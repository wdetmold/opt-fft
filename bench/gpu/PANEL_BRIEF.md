# Implementer brief — complex 3D FFT on ONE A100

Phase 3 of the project. Phases 1 and 2 built CPU kernels that beat every CPU library at
every geometry. This phase starts over on a single NVIDIA A100, against cuFFT.

Read `fft3d_gpu_api.h` first — it is the contract, and it is **not** the CPU contract.
`driver.cu` decides your score; read it too.

## The job

Forward, unnormalized, complex-**double** 3D DFT of `L^3` over a batch of `B` volumes,
out-of-place, on **device-resident data**:

```
out[b][k0][k1][k2] = sum_j in[b][j0][j1][j2] * exp(-2*pi*i*(k0*j0+k1*j1+k2*j2)/L)
element (b,x,y,z) at index ((b*L + x)*L + y)*L + z,  both buffers cudaMalloc'd (256B aligned)
```

You are assigned one `L`. `fft3d_gpu_supports()` may accept only that one — specialize as
hard as you like.

## Rules

1. **No FFT library inside `fft3d_gpu_execute()`.** Not cuFFT, not cuFFTDx, not VkFFT, not
   cuBLAS or cuTENSOR. The kernels are yours. cuFFTDx is banned *as a dependency* even
   though it is header-only and device-side — study its design, do not link it. If a matrix
   formulation is the right answer, you write the `mma`/`dmma` yourself.
2. **One GPU.** No multi-GPU, no NVSHMEM, no MPI.
3. **All setup in `fft3d_gpu_create()`** — plans, device twiddle tables, kernel selection,
   occupancy probing, autotuning. Excluded from your time, and may be as expensive as you
   like.
4. **Double precision, and the accuracy gate is real**: relative L2 against numpy below
   1e-12, and a correct kernel lands near 1e-16. The half- and single-precision tensor-core
   FFT tricks in the literature (tcFFT, turboFFT) cannot meet this gate — they are not a
   shortcut, and an entry that trades accuracy for speed scores nothing.
5. **Repeatable**: same plan, same input, same answer, every call. The driver checks
   `cudaGetLastError()` after your execute and fails the entry loudly if a kernel faulted,
   so a fast zero-filled buffer will be caught.
6. Anything else is fair: shared memory, registers, warp shuffles, `cp.async`, persistent
   kernels, vectorized `double2` loads, `__ldg`, FP64 tensor cores, inline PTX, launch
   bounds, generated code.

## What is timed

**The transform of device-resident data.** The input is copied to the device once, before
timing, and stays there; H2D and D2H are measured separately and reported alongside, because
the target workload keeps the field on the GPU across many operations. So you are being
scored on the kernel, not on PCIe — but the transfer numbers are on the leaderboard, so an
approach that would need a round trip cannot hide it.

Method, identical in shape to the CPU phases: compilation and `create()` excluded, warmup
discarded, inner repeat count auto-calibrated, ~20 samples, 3 independent processes, **CUDA
events around the whole inner loop plus a device synchronize** so work on a non-blocking
stream cannot be counted as free.

**Batch points are defined by working set**, not by round numbers, so the geometries are
comparable and the methodology matches how the GPU literature benchmarks this:

| point | size | what it measures |
|---|---|---|
| `B=1` | one volume | launch and synchronization overhead. Scored separately and read as such — if your execute launches three kernels, you pay three launches |
| `B_L2` | in+out ≈ 32 MiB | the whole problem inside the A100's 40 MiB L2 |
| `B_HBM` | one buffer ≈ 1 GiB | **the primary score**: it cannot hide in cache, so this is the real bandwidth question |

**Do not shorten the sample length.** We have no permission to lock clocks on this cluster,
and the GPU's boost behaviour has a cliff: measured with cuFFT at L=8/B=64, varying only
`--min-sample-ms`, per-execute time was 22.4 µs at 3 ms, 20.9 µs at 10 ms, and **12.3 µs at
20 ms** where the SM clock pins at 1410 MHz. That is 1.7× of pure measurement artefact. The
default is 20 ms; if you measure something surprisingly slow in `tryout.sh`, check that first.

## The machine

Two A100 variants are involved, and they are not the same part:

| | scored on: **A100-SXM4-40GB** (reserved node) | fallback: A100-PCIE-40GB (login node) |
|---|---|---|
| SMs | 108 | 108 |
| FP64 | 9.7 TFLOP/s vanilla, 19.5 via tensor cores | same |
| HBM2 | 40 GB, **~2.0 TB/s** | 40 GB, ~1.55 TB/s |

A bandwidth-bound kernel therefore measures ~30% faster on the node you are scored on than on
the login node's GPU. Never compare across the two. Everything below is the SXM4 part:

| | A100 (sm_80), per SM |
|---|---|
| L2 | 40 MB |
| shared memory | 164 KB/SM carveout, but **163 KB max per block** and only **48 KB by default** |
| registers | 256 KB/SM (65536 × 32-bit), max 255 per thread |

Working sets per volume (complex double = 16 B/point):

| L | points | bytes | fits one block's 163 KB shared? |
|---|---|---|---|
| 6 | 216 | 3.4 KB | yes, ~48 volumes |
| 8 | 512 | 8.2 KB | yes, ~20 volumes |
| 13 | 2197 | 35 KB | yes, 4 volumes |
| 17 | 4913 | 78.6 KB | yes, 2 volumes |
| 23 | 12167 | 190 KB | **no** |
| 36 | 46656 | 746 KB | no |
| 45 | 91125 | 1.42 MB | no |
| 64 | 262144 | 4.19 MB | no |

That table is the central design fact: for L ≤ 17 a whole volume fits in one SM's shared
memory, so all three axes can be done with **one** global read and one global write. Above
that they cannot, and the transform becomes a memory-traffic problem.

Four hardware facts that the corpus had wrong or that are easy to get wrong, all corrected in
`../../docs/literature/09-gpu-small-batched-a100.md`:

* **You only get 48 KB of shared memory per block unless you ask.** The default is 49152 B;
  anything above it requires
  `cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)` and a
  dynamic allocation. Forget this and a 17³ volume (78.6 KB) simply fails to launch.
* **L=23 does not fit shared memory.** 190.1 KiB against a 163 KB per-block ceiling. An
  earlier version of the corpus quoted "192 KB per block", which is the unified L1+shared per
  *SM*; under that wrong number L=23 looks like it fits. It does not — it is a registers-or-
  tiling problem.
* **The register file is 1.56× shared memory, not 4×.** 256 KB registers against 164 KB
  shared per SM. The 4× figure came from a pre-Volta 32 KB shared figure. The direction still
  favours registers; the magnitude does not decide the question at L = 13, 17, 23.
* **A complex double occupies 4 registers, not 2.** So a 64-point line is 256 registers,
  above the 255-per-thread ceiling: at L=64 one line per thread is impossible and you need
  2–4 threads per line.

## Where to develop, and where you are scored

**You get a whole A100 to yourself.** The project holds an 8-GPU node for this phase, and
because ssh into a node is permitted while you hold an allocation on it, `tryout.sh` leases
one of those GPUs and runs your code there:

```bash
cd /home/lqcd/wdetmold/fft/bench/gpu
./tryout.sh L17_dmma            # infers L from the name, B=64
./tryout.sh L17_dmma 17 512
```

It builds your file, leases a GPU, runs on the reserved node, verifies against numpy, re-runs
to confirm bit-identical output, releases the lease, and prints cuFFT on the same case. Run it
after every change — an uncontended SXM4 A100 gives ~0.02% run-to-run spread, so you can
actually see a 2% improvement.

**Eight agents, eight GPUs, so take one only while you are using it.** `./gpu_lease.sh status`
shows who holds what. If all eight are busy, `tryout.sh` waits. If the reservation is down it
falls back to the shared login-node GPU and says so — those numbers are relative only.

`compute-sanitizer --tool memcheck` and `ncu --set full` work over the lease too, and an
occupancy or bank-conflict number will tell you more than another guess. Profiling counters
need the GPU to yourself, which is exactly what the lease gives you.

**Do not submit slurm jobs.** The reservation is already there; a second allocation would sit
in the queue behind the cluster's real work. The monitor takes the scored measurements inside
a *scoring window* that holds all eight leases, so nothing else is on the node while numbers
are recorded — which is why your lease may briefly have to wait.

## What carries over from the CPU phases

Read `../geom/strategies/` and `../mt/strategies/` — the same kernels, three rounds of
measurement each, and their records say what worked. Two results are worth having in mind:

* **At the prime L=17, a dense conjugate-symmetric matrix-vector product beat Rader** on the
  CPU, by 4.97× over the best library, because a matvec has perfect data flow. On an A100
  there is an FP64 tensor-core path (DMMA) at 2× the vanilla FP64 rate that a matrix
  formulation can reach and a butterfly cannot — and unlike the published half-precision
  tensor-core FFT work, **DMMA is IEEE-compliant FP64 with round-to-nearest**, so it clears
  our 1e-12 gate outright. The corpus works the arithmetic through: a dense per-axis DFT is
  24·L flop/point, which hides entirely under the 32-byte/point HBM floor for L ≤ 8.3 on the
  vanilla pipe and **L ≤ 16.7 on DMMA** — i.e. DMMA moves the "a dense matrix is free"
  frontier from L≈8 to L≈17, exactly where the CPU result lives. Two caveats stated equally
  plainly: there is no prior work on FP64 tensor-core FFT at all, so this is a well-supported
  hypothesis with no empirical check; and the one paper that does program DMMA directly found
  **shared-memory data motion, not FLOPs, was the bottleneck** — which reframes the benefit as
  fewer shared-memory reads of the matrix rather than the 19.5 TFLOP/s headline.
* **The libraries are not weak at 17.** Unlike FFTW and MKL, cuFFT has radix-m building
  blocks for every prime below 128 (Bluestein starts only at 131), and per VkFFT's own
  account cuFFT's radix-17 *is* a dense matvec. VkFFT's real radix set is 2–16 and 32, so six
  of our eight geometries are one or two native stages for it. Do not expect the CPU phase's
  4.97× to reappear here: the libraries already have our algorithm, and we have to implement
  it better rather than differently.
* **The CPU phase plateaued on arithmetic and was decided by layout.** Expect the same here,
  more so: at these sizes with large batches the transform is bandwidth-bound, and the
  measured floor already shows it — a deliberately naive O(L⁴) GPU kernel is only 1.6×
  behind cuFFT at L=8, because neither is FLOP-limited.

`../../docs/literature/09-gpu-small-batched-a100.md` is the GPU section of the corpus:
A100 capacity arithmetic, cuFFT's and VkFFT's design, the tensor-core question, and a
per-geometry opening strategy table. Start there.

## Records

Independent of the CPU phases: code in `impl_N/`, your record in `strategies/`, leaderboards
in `results/gpu_rN/`. Append to your strategy record each round, never overwrite, and say
what you borrowed and from whom — including from the CPU records.

## Read every previous phase — all the code, not just the conclusions

Two CPU phases came before this one, on the same eight geometries and largely the same
kernels. All of it is on disk and you are expected to read it:

| what | where |
|---|---|
| **single-core code, final form** | `../geom/impl_11/` (20 sources) |
| **single-core history, round by round** | `../geom/impl_6/` … `../geom/impl_11/` — `diff` consecutive rounds to see what each change was |
| **multicore code and history** | `../mt/impl_*/` |
| **the reasoning, in the implementers' words** | `../geom/strategies/*.md` and `../mt/strategies/*.md` — including failures with the number that killed them |
| **curated exemplars** | `../geom/exemplars/<round>/`, `../mt/exemplars/<round>/` |
| final CPU standings | `../geom/results/panel_r11/leaderboard.txt`, `../mt/results/mt_r*/leaderboard.txt` |

The CPU times are not comparable to yours and the hardware is different in kind — but the
*algorithmic* work transfers directly, and it is substantial: eleven rounds of measured
decisions about radix choice, conjugate-pair folding, index maps and layout. The single-core
winner at your geometry is the best available statement of what the arithmetic should look
like; your job is to map it onto 108 SMs, not to rediscover it.

Two specifics worth chasing in that code: `../geom/strategies/L17_matrixsimd.md` explains the
conjugate-symmetric folding behind the 5.42× win at L=17 (directly relevant to the DMMA
entry), and `../geom/strategies/L36_pfa.md` records a Good-Thomas kernel that lost by 1.9×
and then recovered to take the lead — the reasons it lost are mostly memory-behaviour
reasons, which are sharper on a GPU.
