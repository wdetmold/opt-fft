# 3D complex FFT libraries — survey and local installation

Target workload: **many batched small `L^3` complex transforms** (LQCD-style spatial
volumes, `L` ~ 16–48, batched over time slices / spin-colour / configurations),
across four hardware regimes: single GPU, single-node CPU, multi-node MPI, multi-GPU.

Everything is installed under `ext/install`; `source env.sh` sets up modules, paths
and the Python venv.

## Hardware available

| Where | CPU | GPU | Memory | Notes |
|---|---|---|---|---|
| this login node | 2× Xeon E5-2680 v3 (Haswell, 12c/24t each, 48 threads), AVX2 + FMA, **no AVX-512** | 1× A100-PCIE-40GB (sm_80) | 125 GB | interactive dev + single-GPU work |
| `axxxl` (a80n0, a81n2) | 64 CPU, ~980 GB — **AVX-512 testing target** | none allocated (`gpu:a100:0`) | 980 GB | CPU-only; exact model still to be confirmed (nodes were fully allocated when probed) |
| `a100l` / `a100r` | 64 CPU | **8× A100 per node** (sm_80), 6 + 8 nodes | 980 GB | multi-GPU and multi-node GPU scaling |
| `prod` / `devel` / `long` | 64 CPU (2×16c) | 8× RTX 2080 Ti (sm_75) | 326 GB | large node count (20+); cheap multi-GPU scaling at sm_75 |

Consequences for builds: FFTW is compiled with SSE2/AVX/AVX2/**AVX-512** codelets, all
runtime-dispatched by cpuid, so one build runs optimally on both Haswell and the
AVX-512 nodes. CUDA code targets `sm_75;sm_80` (`$CUDA_ARCHS`). MPI is the CUDA-aware
OpenMPI 4.1.5 (UCX + hcoll + slurm/pmix), which is what heFFTe and cuFFTMp need for
GPU-direct communication.

## Installed — CPU

| Library | Version | Where | Batching interface | Why it matters here |
|---|---|---|---|---|
| **FFTW3** | 3.3.10 (built here: double + single, shared + static, pthreads + OpenMP + MPI, AVX-512) | `ext/install` | `fftw_plan_many_dft`, guru `fftw_plan_guru64_dft` | the reference baseline everyone quotes; guru interface expresses arbitrary batched strides exactly, which matters for spin-colour-inner layouts |
| **oneMKL DFTI** | 2022.0.2 (module `mkl/latest`) | `$MKLROOT` | `DFTI_NUMBER_OF_TRANSFORMS` + input/output distance | usually the fastest CPU option on Intel; native batch descriptor, no plan-per-batch overhead. Caution: MKL's runtime dispatch is conservative on AMD, so verify if `axxxl` turns out to be EPYC |
| **ducc0** | 0.41.0 (pip; C++ source in `ext/src/ducc`) | venv + `ext/src/ducc/src/ducc0/fft` | `c2c` with `axes=`, threads | pocketfft's successor (Reinecke). No planning phase and very strong at small/medium sizes — a serious contender for `L ~ 16–32` where FFTW's planner overhead dominates |
| **mkl_fft** | 2.3.2 (pip) | venv | numpy-like | quick Python access to DFTI for prototyping |
| **pyFFTW** | 0.15.1 (pip) | venv | `FFTW` object with `axes=` | Python access to FFTW incl. wisdom reuse |

Not installed, deliberately: **PFFFT / FFTS / KFR / muFFT** — single-precision, small-size
CPU kernels. Worth revisiting only if we end up hand-writing small-`L` CPU kernels;
none is a credible published baseline for 3D.

## Installed — single GPU

| Library | Version | Where | Batching interface | Why it matters here |
|---|---|---|---|---|
| **cuFFT** | 11.0.8 (CUDA 12.2.1) | module `cuda/12.2.1` | `cufftPlanMany` / `cufftXtMakePlanMany` | the baseline on NVIDIA; batched 3D C2C in one call |
| **VkFFT** | 1.3.4 (header-only, CUDA backend) | `ext/src/VkFFT/vkFFT/vkFFT.h`, `$VKFFT_DIR` | `numberBatches`, `omitDimension` | generates kernels at runtime and routinely beats cuFFT on small batched transforms; bundles its own benchmark suite (`VkFFT_TestSuite.cpp`) for direct comparison |
| **pyvkfft** | 2025.1.1 (CUDA backend only — no OpenCL headers on this system) | venv | cupy arrays | lets us benchmark VkFFT vs cuFFT from Python in a few lines |
| **cuFFTDx** | via `nvidia-mathdx`, headers at `$MATHDX_DIR` | venv site-packages | device-function FFT inside your own kernel | **the most promising route for this workload**: a small `L^3` fits in registers/shared memory, so all three axis transforms — plus whatever contraction follows — can be fused into a single kernel with no global-memory round trip between axes. This is exactly where library-level cuFFT loses |
| **cupy** | 14.2.0 | venv | `cupy.fft.fftn(axes=)` | driver for cuFFT experiments; verified against VkFFT to 1.5e-7 (single precision) on the A100 |

## Installed — distributed (multi-node CPU, multi-GPU)

| Library | Version | Where | Decomposition | Why it matters here |
|---|---|---|---|---|
| **FFTW3-MPI** | 3.3.10 (built here) | `ext/install/lib/libfftw3_mpi.so` | 1D slab | simplest distributed baseline; slab limits parallelism to `L` ranks, which is fine at `L ~ 32` on a few nodes |
| **heFFTe** | git `master` (building: fftw + mkl + cufft + stock/AVX-512 backends) | `ext/install` | pencil / slab, configurable reshapes | the ECP-era standard for distributed FFT, CPU *and* GPU, with GPU-aware MPI. One API to compare all backends, so it doubles as the cross-backend harness |
| **cuFFTMp** | 11.4.0.6 (+ NVSHMEM 3.7.2) | `venv/.../nvidia/cufftmp/cu12` | NVSHMEM-based slab/pencil | the state of the art for multi-GPU/multi-node A100 (`a100l`/`a100r`, 8 GPUs per node); communication runs over NVSHMEM rather than MPI collectives |

Not installed, deliberately: **P3DFFT++**, **2DECOMP&FFT**, **PFFT**, **AccFFT**,
**fftMPI**. All are pencil-decomposition distributed FFTs that predate heFFTe and are
now either unmaintained or strictly narrower (Fortran-only, CPU-only, or FFTW-only
backends). heFFTe plus cuFFTMp covers the same ground with current code. `gfortran` is
present, so 2DECOMP&FFT can be added if a Fortran-side comparison is ever wanted.
oneMKL also ships **cluster DFT** (`mkl_cdft`) and heFFTe already links it; using it
standalone would want Intel MPI (module `mpi/2021.3.0`) rather than OpenMPI.

## Reference implementation

`python/slow_dft.py` is the from-scratch ground truth — no FFT factorization, no library
FFT calls:

- `dft3d_loops` — the definitional 6-fold sum, `O(V^2)`
- `dft3d_matrix` — dense `V×V` apply, same complexity via BLAS
- `dft3d_separable` — row-column with a dense `O(n^2)` DFT matrix per axis, `O(V·Σn)`
- `dft3d_batched`, `idft3d_separable`, plus `rel_error` / `random_field` / `flop_count` helpers

Conventions match `numpy.fft`: forward `exp(-2πi k·j/n)` unnormalized, `1/V` on the
inverse, batch index slowest-varying in `(B, n0, n1, n2)`. All three agree with
`numpy.fft.fftn` to ~2e-15 at `L=6`, and the separable form stays usable to `L=32`.

## What to measure next

1. Small-`L` batched sweep on one A100: cuFFT vs VkFFT vs cuFFTDx, over `L ∈ {16, 24, 32, 48}`
   and batch sizes spanning under- to fully-occupied GPU. cuFFTDx is the one that can
   fuse the three axes; the others cannot.
2. Same sweep on CPU: FFTW (with wisdom) vs MKL DFTI vs ducc0, threads 1→48 here and
   1→64 on `axxxl` to separate AVX-512 gain from core-count gain.
3. Single vs double precision throughout — small `L^3` is bandwidth/occupancy bound, so
   complex64 typically wins outright if the physics tolerates it.
4. Distributed scaling only once single-device numbers are known, so the comparison is
   against a real roofline rather than against another distributed run.
