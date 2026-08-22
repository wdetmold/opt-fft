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

| | A100-PCIE-40GB (sm_80) |
|---|---|
| SMs | 108 |
| FP64 | 9.7 TFLOP/s vanilla, 19.5 TFLOP/s via FP64 tensor cores |
| HBM2 | 40 GB, ~1.55 TB/s |
| L2 | 40 MB |
| shared memory | 164 KB/SM carveout, but **163 KB max per block** and only **48 KB by default** |
| registers | 256 KB/SM (65536 × 32-bit), max 255 per thread |

Working sets per volume (complex double = 16 B/point):

| L | points | bytes | fits in 164 KB shared? |
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

**Develop here, on the login node** — it has an A100, so unlike the CPU phases you develop
on the same architecture you are scored on. One command:

```bash
cd /home/lqcd/wdetmold/fft/bench/gpu
./tryout.sh L17_dmma            # infers L from the name, B=64
./tryout.sh L17_dmma 17 512
```

It builds only your file, runs it, verifies against numpy, re-runs to confirm bit-identical
output, and prints cuFFT on the same case. The GPU here is **shared**, so treat its timings
as relative: cuFFT's run-to-run spread is ~30% on this node against ~0.3% on a
job-allocated GPU. Use it for "did this change help", never as a reported number.

`compute-sanitizer --tool memcheck` and `ncu --set full` are available and worth your time —
an occupancy or bank-conflict number will tell you more than another guess.

**Do not submit slurm jobs.** The monitor owns the scored measurement, on an a100 partition
with `--gres=gpu:1`. If you need something only a job-allocated GPU can show, say so in your
return value and the monitor will take it.

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
