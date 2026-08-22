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
discarded, inner repeat count auto-calibrated to clear the timer, ~20 samples, 3 independent
processes, **CUDA events around the whole inner loop with an explicit synchronize** so
asynchronous launches cannot be counted as free. Batched and non-batched are scored
separately.

Note what that means for `B=1`: one 6³ volume is 3.4 KB of work on 108 SMs. That case is
dominated by launch and synchronization overhead, and it is a real part of the competition —
if your `execute` launches three kernels, you pay three launches.

## The machine

| | A100-PCIE-40GB (sm_80) |
|---|---|
| SMs | 108 |
| FP64 | 9.7 TFLOP/s vanilla, 19.5 TFLOP/s via FP64 tensor cores |
| HBM2 | 40 GB, ~1.55 TB/s |
| L2 | 40 MB |
| shared memory | up to 164 KB/SM (configurable carveout), 32 banks × 4 B |
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
  there is an FP64 tensor-core path at 2× the vanilla FP64 rate that a matrix formulation
  can reach and a butterfly cannot. Whether that is real for a 17×17 double-precision
  operand is the most interesting open question in this phase.
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
