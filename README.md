# 3D complex FFT — optimization for specific task geometries

Building a 3D complex FFT tuned for fixed geometries, with the installed libraries
(FFTW, oneMKL, ducc0, cuFFT, VkFFT, cuFFTDx, heFFTe, cuFFTMp) used as **state-of-the-art
benchmarks rather than as the implementation**. The target workload is many batched small
`L^3` complex volumes, LQCD-style.

```bash
source env.sh     # modules (cuda 12.2, CUDA-aware OpenMPI 4.1.5, MKL, python 3.12) + venv
```

## Layout

| Path | What |
|---|---|
| `python/slow_dft.py` | the definition, transcribed — naive, dense-matrix and separable DFTs. Correctness ground truth. |
| `python/fft3d.py` | **the textbook FFT**: mixed-radix Cooley-Tukey, Rader, Bluestein, row-column 3D. Any geometry, anisotropic included. Derivations in the docstrings. |
| `python/test_fft3d.py` | 34 checks of `fft3d` against the slow reference and against extended precision |
| `python/verify_backends.py` | every installed library backend checked against the slow reference |
| `bench/geom/` | the fixed-geometry benchmark harness (see below) |
| `bench/smoke/` | minimal link-and-run checks for the distributed installs |
| `docs/SURVEY.md` | library survey, hardware map, what is installed and why |
| `docs/TEXTBOOK_FFT.md` | algorithm map, route per extent, validation summary |
| `docs/LITERATURE.md` + `docs/literature/` | cited corpus (~7,500 lines) on optimizing FFTs for fixed small geometries, with a per-size strategy table |
| `docs/CURATION.md` | what this repo tracks and why; how panel exemplars are promoted |
| `ext/build_*.sh`, `ext/requirements.txt` | rebuild every external library and the venv from scratch |

## The benchmark harness — `bench/geom/`

One ABI (`fft3d_api.h`), one timing driver (`driver.c`), many backends. Every backend is
linked against the same driver and sees bit-identical input, so the comparison is like for
like.

* **Timing** excludes compilation and `fft3d_create()` (reported separately), discards
  warmup, auto-calibrates an inner repeat count so short cases clear the clock, takes ~20
  samples, and repeats the whole process in 3 independent processes. **Batched and
  non-batched are separate measurements.**
* **Correctness** is checked for every backend on every case against `numpy.fft`, whose
  own agreement with the from-scratch definition is established by `python/test_fft3d.py`.
* **Isolation**: `./submit.sh --round TAG --seed N` submits an `--exclusive` slurm job that
  rebuilds on the benchmark node (Xeon Gold 5218, AVX-512) with `-march=native`, generates
  fresh random data, measures everything, and writes `results/TAG/leaderboard.txt`.
* `sota/` holds the library baselines: FFTW at three planner levels, oneMKL 2022 **and**
  2026, ducc0. `impl/` is the panel's scratch area; `impl/baseline_matrix.c` is the
  library-free floor used to validate the harness itself.

Rules for our own implementations: **no FFT library calls inside the transform**,
single-threaded, all precomputation in the plan step. Full brief in
`bench/geom/PANEL_BRIEF.md`.

**Where work happens.** Development and iteration go on `wallaby` (Xeon Gold 6448Y,
Sapphire Rapids, 64 cores, near-idle, full AVX-512, shares this filesystem):
`./tryout.sh --on wallaby <impl>` builds, runs, verifies and times one implementation in
one command. The exclusive benchmark node is reserved for the monitor's cross-checks, so
every scored number comes from one uncontended machine measured one way; `probe_node.sh`
refuses to run without `FFT_MONITOR=1`. Build products live in `build/<hostname>/`, since
this tree is shared between machines with different ISAs.

## Panel rounds

`bench/geom/panel_round.js` runs one round: implementer agents each write one
self-contained optimized implementation for one geometry, then a monitor benchmarks them
all on an isolated node against the library baselines on identical data.

Implementers maintain `bench/geom/strategies/<name>.md` — appended each round, never
overwritten — because the next round optimizes from those records. Entries worth keeping
are promoted with `./promote.sh <round> <name>...` into `bench/geom/exemplars/<round>/`,
carrying their strategy record and measured numbers.

## Status

Verified on this machine: all Python backends agree with the from-scratch reference to
~2e-15 (VkFFT to 9e-8 in single precision); the textbook FFT passes 34 checks including
44 geometries against the slow reference, worst error 1.7e-14; FFTW-MPI roundtrip 1.5e-16
on 4 ranks; heFFTe 32³ double = 4.4 GFlop/s on 2 CPU ranks and 133 GFlop/s on one A100.

Library baselines on the isolated node (per transform, non-batched):

| L | best library | time | GF/s |
|---|---|---|---|
| 6 | MKL 2022 | 0.370 µs | 22.6 |
| 8 | MKL 2022 | 0.653 µs | 35.3 |
| 17 | FFTW patient | 81.7 µs | **3.7** |
| 36 | MKL 2022 | 163.6 µs | 22.1 |

`L=17` is the standing opportunity: every library runs ~6× below its `L=8` efficiency
there, because 17 is prime and the generic path handles it badly.
