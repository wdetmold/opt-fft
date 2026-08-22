# The campaign: four chained competitions

Each phase is a competition between implementer agents, measured by a monitor, with its own
harness directory and its own independent records. `bench/PHASE` names the current one;
`bench/run_current_phase.sh` is the single cron entry point; each phase's
series-completion hook builds the next and flips the pointer, so the chain advances with
nobody watching.

| phase | harness | rounds | hardware | geometries | status |
|---|---|---|---|---|---|
| 1 | `bench/geom` | `panel_rN` | 1 core, Xeon Gold 5218 (exclusive) | L = 6, 8, 17, 36 → + 13, 23, 45, 64 | **running** (r6 of r11) |
| 2 | `bench/mt` | `mt_rN` | 32 cores, same node, pinned | same 8 | generated at handover |
| 3 | `bench/gpu` | `gpu_rN` | one A100 | same 8 | built and validated |
| 4 | `bench/mgpu` | `mgpu_rN` | 8× A100-SXM4 on one node | **L = 64, 96, 128, 192, 256** | designed (below) |

Rounds per phase: 6. One round is 11–19 implementer agents revising code (headless, on
`fable-5`), then a mechanical timing pass, then a monitor verdict (`opus-5`), then promotion
of exemplars and a commit.

## Why the geometries change in phase 4

Phases 1–3 use small cubes because that is the target workload: many batched `L^3` volumes
with `L` ≈ 6–64. Multi-GPU makes no sense there — a 64³ volume is 4.19 MB, and splitting it
across 8 GPUs would spend more time on interconnect than on arithmetic.

Phase 4 therefore moves up, to where distribution is the point:

| L | volume | bytes/volume | why |
|---|---|---|---|
| 64 | 262144 | 4.19 MB | **the bridge**: also in phase 3, so single-GPU and multi-GPU numbers can be compared directly and strong scaling measured honestly |
| 96 | 884736 | 14.2 MB | 2⁵·3, a real LQCD spatial extent |
| 128 | 2.10M | 33.6 MB | 2⁷, the clean power-of-two case |
| 192 | 7.08M | 113 MB | 2⁶·3, where the all-to-all starts to dominate |
| 256 | 16.8M | 268 MB | 2⁸; batched, this is hundreds of MB per GPU and a genuine distributed transform |

## Phase 4 design

**Topology.** One node, 8× A100-SXM4-40GB, NVLink between them. Single process, 8 devices —
not MPI. This is deliberate: it keeps the harness a single binary like every other phase,
and it matches the hardware we can actually hold (`reserve.sh` claims one whole node).

**Decomposition.** Slab: GPU *g* owns a contiguous range of the slowest axis,
`x ∈ [g·L/n, (g+1)·L/n)`, for every volume in the batch. The z and y transforms are local;
the x transform requires a redistribution. That single all-to-all is the whole problem, and
it is where the competition will be decided.

**ABI** (`fft3d_mgpu_api.h`, to be written):

```c
int   fft3d_mgpu_supports(int L, int ngpus);
plan *fft3d_mgpu_create(int L, int batch, int ngpus, const int *devices);
void  fft3d_mgpu_execute(plan *, const double2 *const *in, double2 *const *out);
```

`in[g]` and `out[g]` are device pointers **on device g**, holding that GPU's slab in the same
C order as the single-GPU phases. The driver allocates and distributes; the implementation
owns everything after that.

**Rules.** As before, no FFT library inside execute — not cuFFT, cuFFTDx, VkFFT, cuFFTMp or
heFFTe. Communication is explicitly allowed and is not "the FFT": `cudaMemcpyPeer`,
peer-mapped pointers, NCCL, and CUDA IPC are all fair, because the interesting question is
*which* communication pattern wins, not whether you may move bytes. What you may not do is
call somebody else's transform.

**Baseline.** `cufftXtSetGPUs` + `cufftXtExecDescriptor` — cuFFT's own single-process
multi-GPU path, which is exactly the same job under the same conditions. Where it is
available, heFFTe with the cuFFT backend and cuFFTMp (both installed) give a second,
MPI-based reference point; those need a separate MPI driver and are a stretch goal rather
than a gate.

**What is timed.** Device-resident data on all 8 GPUs, H2D/D2H excluded and reported
separately, CUDA events plus a synchronize across every device so no GPU's work is counted as
free. Reported per transform, plus:

* **strong scaling** against the phase-3 champion at L = 64, on the same data;
* **achieved interconnect bandwidth** during the redistribution, against NVLink peak — the
  number that will actually explain the results.

**Opening strategies** for the panel (two per geometry):

* *slab + peer all-to-all*: the textbook approach; measure what fraction of NVLink you reach.
* *pencil decomposition*: two redistributions instead of one, but each smaller — the standard
  answer at scale, worth testing at 8 GPUs where it is not obviously right.
* *overlap*: split the batch into chunks and overlap chunk *k*'s communication with chunk
  *k+1*'s arithmetic. On paper this hides the all-to-all entirely; the question is whether
  8 GPUs' worth of copies can be kept in flight.
* *transpose-free*: keep the data where it is and change which axis each GPU transforms,
  paying redistribution only once at the end (or folding it into the consumer).

## The GPU node reservation

This cluster is `SelectType=select/linear`, so slurm cannot allocate a single GPU — per-GPU
`--gres` requests are rejected outright, and `--gpus` is unsupported by the plugin. Phases 3
and 4 therefore claim a **whole 8-GPU node** with a placeholder job (`reserve.sh`), and,
because ssh to a node is permitted while you hold an allocation on it, agents run their work
there over ssh:

* `reserve.sh` claims/releases the node and heartbeats so a dead reservation is detectable.
* `gpu_lease.sh` hands out one GPU per agent. A lease is a *directory* (`mkdir` is atomic
  even over NFS, unlike a check-then-create test or `flock`). Eight agents, eight GPUs.
* `on_gpu.sh -- <cmd>` leases a GPU, runs the command on the reserved node with
  `CUDA_VISIBLE_DEVICES` pinned to it, and releases the lease even on failure.
* The monitor's scored runs open a **scoring window** (`acquire-all`), which blocks new
  single-GPU claims and waits for in-flight ones to drain, so nothing else is on the node
  while numbers are taken.

The node is claimed at phase handover, not held through the earlier phases — 8 idle A100s
for eight hours would be antisocial and pointless. If the partition is busy the phase still
runs: rounds fall back to queueing a whole-node job, and implementers fall back to the login
node's A100 (which is the PCIe part at ~1.55 TB/s against the SXM4's ~2 TB/s, so its numbers
are relative only).
