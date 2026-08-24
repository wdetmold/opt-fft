# `fft3d_best` — the best single-core kernel for each geometry

One API over the eight kernels that won the single-threaded competition, so a caller who
just wants the fastest available `L^3` complex-double transform can link one library and
call one function.

```c
#include "fft3d_best.h"

fft3d_best_plan *p = fft3d_best_create(17, 256);   /* all setup, may be expensive */
for (...) fft3d_best_execute(p, in, out);          /* out-of-place, repeatable */
fft3d_best_destroy(p);
```

```
make          # libfft3d_best.a + verify_best
make verify   # checks every geometry against the definition
```

No FFT library is involved: the arithmetic is entirely in `kernels/`.

## What is here, and where it came from

Every kernel is **byte-identical to the source that produced the measurement** — copied from
`../impl_11/`, the final round of the single-core series. Provenance matters more than tidiness
here, so nothing was reformatted or merged.

| L | kernel | technique | time (1 vol) | vs best library |
|---|---|---|---|---|
| 6 | `L6_pfa` | Good-Thomas PFA 2×3 per axis, no inter-stage twiddles, 2 complex/ymm | 0.208 µs | **1.79×** MKL |
| 8 | `L8_batchsimd` | radix-8 split, batch-major SIMD across volumes | 0.551 µs | 1.18× MKL |
| 13 | `L13_direct` | conjugate-folded dense 13×13 per axis, lanes = lines | 5.73 µs | 1.31× MKL 2026 |
| 17 | `L17_matrixsimd` | nested cyclic/negacyclic 17-point per axis, pinned sines | 15.07 µs | **5.42×** FFTW |
| 23 | `L23_rader` | Rader-23, folded conjugate pair, two-sweep | 47.61 µs | **5.48×** FFTW |
| 36 | `L36_mixedradix` | PFA 4×9, two-sweep, radix-9 DFT codelet | 114.6 µs | 1.42× MKL |
| 45 | `L45_pfa` | Good-Thomas 9×5, two-sweep | 304.2 µs | **1.99×** MKL |
| 64 | `L64_blocked` | 8×8 two-stage, hugepage odd-line-padded scratch | 952.7 µs | 1.25× MKL |

Times are per transform, non-batched, from `../results/panel_r11/leaderboard.txt`: one
exclusive Xeon Gold 5218 (Cascade Lake, AVX-512), single-threaded, plan setup excluded,
minimum over three independent processes. Every kernel verifies at ~1e-16 relative L2.

**Run `verify_best` on that node, not on a login node.** Several kernels probe the machine in
`create()` and pick a 512-bit or 256-bit path accordingly; on the AVX2 login node they
correctly select 256-bit and report roughly 1.6× the times above.

## Three caveats worth reading before using this

1. **These are the non-batched winners.** At large batch the ranking changes at three
   geometries: L=8 goes to `L8_fusedaxes`, L=36 to `L36_pencilfused`, L=64 to `L64_radix8`.
   All three are in `../impl_11/`, and the batched sections of the leaderboard have the
   numbers. If your workload is throughput-bound rather than latency-bound, use those.
2. **The kernels self-tune in `create()`.** Some race several variants and keep the fastest,
   so `create()` can take a good fraction of a second and its choice depends on the machine
   and the batch size. Create once, reuse the plan.
3. **Only these eight sizes.** `fft3d_best_supports(L)` is false for everything else — there
   is no fallback, deliberately, so a caller cannot silently get a slow generic path. For
   arbitrary geometries use `python/fft3d.py`, which handles any extent (and is a reference,
   not a fast path).

## How the eight coexist in one library

They were written independently and share nine `static` helper names between them; one
(`L17_matrixsimd`) instantiates itself by `#include`ing its own file. So they are **not**
concatenated into a single translation unit — that would collide and break. Instead each is
compiled separately with its six API symbols renamed by `-D` (see the `Makefile`), and
`fft3d_best.c` dispatches on `L` through a small table. The result is one library and one
header, with each kernel still exactly the code that was measured.
