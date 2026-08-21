# Implementer brief — optimized complex 3D FFT for one fixed geometry

You are writing **one self-contained C file** that transforms a fixed cube size as fast
as you can make it. Read `fft3d_api.h` first: it is the contract, and the driver that
times you is `driver.c`.

## The job

Forward, unnormalized, complex-double 3D DFT of `L^3`, over a batch of `B` volumes,
out-of-place, in the exact layout the driver hands you:

```
out[b][k0][k1][k2] = sum_j in[b][j0][j1][j2] * exp(-2*pi*i*(k0*j0+k1*j1+k2*j2)/L)
element (b,x,y,z) at index ((b*L + x)*L + y)*L + z,   both buffers 64-byte aligned
```

You are assigned exactly one `L` from **6, 8, 17, 36**. Specialize as hard as you like:
`fft3d_supports()` may return true only for your `L`. Constants known at compile time
are yours to exploit — that is the entire point of the exercise.

## Rules

1. **No FFT library calls in `fft3d_execute()`.** Not FFTW, not MKL, not cuFFT, not
   ducc0, not a vendor DFT of any kind. The arithmetic must be yours. `libm` for
   `sin`/`cos` during setup is fine.
2. **Single-threaded.** No OpenMP, no pthreads, in this round.
3. **All precomputation goes in `fft3d_create()`**, which is timed separately and
   excluded from your score. It may be as expensive and as clever as you like:
   twiddle tables, permutation tables, generated code selected at runtime, self-tuning
   over variants — all fair.
4. `fft3d_execute()` must be repeatable: same plan, same input, same answer every time.
5. Accuracy: relative L2 error against numpy must be `< 1e-12`. In practice a correct
   implementation lands near `1e-16`; anything near the tolerance means a real problem.
   Reassociating additions is allowed (and expected), but do not trade away digits.
6. Do not modify `driver.c`, `fft3d_api.h`, `Makefile`, `sweep.sh`, `check.py`, or any
   other implementer's file. Your file is `impl/L<L>_<your-tag>.c` and nothing else.

Anything else is permitted: unrolled straight-line codelets, Good-Thomas/PFA index
mapping, Rader, Winograd modules, vector radix, SIMD intrinsics (AVX2 **and AVX-512**,
see the target machine below), in-register transposes, batch-major repacking, cache
blocking, software pipelining, non-temporal stores, generated code, compiler-specific
pragmas. Requesting a different compiler or an autotuning framework is also allowed.

## The machine you are scored on

The benchmark node is an **isolated (`--exclusive`) Intel Xeon Gold 5218**
(Cascade Lake, 2.30 GHz, 2×16 cores, and **full AVX-512**: `avx512f avx512dq avx512bw
avx512vl avx512cd avx512_vnni`). Binaries are compiled *on that node* with
`-O3 -march=native -mtune=native -std=gnu11`, so AVX-512 intrinsics are available and
`__AVX512F__` is defined. Guard anything exotic with `#ifdef` so the build never breaks.

The interactive machine you develop on is a Haswell Xeon E5-2680 v3 (AVX2, **no**
AVX-512), so AVX-512 code paths compile there only under an explicit `-mavx512f` and
cannot be *run* there. Develop the portable path locally, guard the AVX-512 path, and
let the monitor measure it on the real node.

**AVX-512 is not automatically a win on this part, and you must measure it.** The corpus
(§04 §8.1-8.2, and `LITERATURE.md` §4.8 gap 6) documents licence-based downclocking on
Skylake-SP/Cascade Lake: a Xeon Gold 5120 runs 2.7 GHz scalar, 2.3 GHz under AVX2 and
**1.6 GHz under AVX-512** with several cores active, and some SKUs in this family have only
one AVX-512 FMA unit rather than two. Our node is a Gold 5218, same family. So a 512-bit
kernel can lose to a 256-bit one despite twice the width. Build both paths where you can,
and let the monitor's numbers decide rather than assuming wider is faster. No AVX-512
measurement exists anywhere in the corpus — whatever you measure here is new information,
so put it in your strategy record.

Working-set arithmetic (complex double = 16 B/point, 32 KB L1d, 1 MB L2, 22 MB L3):

| L | points | bytes/volume | volumes in L1 | volumes in L2 |
|---|---|---|---|---|
| 6 | 216 | 3.4 KB | ~9 | ~300 |
| 8 | 512 | 8.2 KB | ~4 | ~128 |
| 17 | 4913 | 78.6 KB | 0 (1 volume ≈ 2.4× L1) | ~13 |
| 36 | 46656 | 746 KB | 0 | ~1.4 |

## How you are timed

Compilation is outside the measurement. `fft3d_create()` is excluded. Warmup executes
are discarded. The driver then auto-calibrates an inner repeat count so a timed sample
comfortably exceeds the clock resolution, takes ~20 samples, and the whole process is
repeated 3 times independently; the leaderboard reports the minimum across processes
with the spread beside it. **Non-batched (`B=1`) and batched are scored separately** —
being fast only in one regime is a partial result, so state which you optimized for.

## Develop with a fast local loop — you are expected to iterate

Do not write the whole thing and hope. Build, run and check after every change, on the
login node, as often as you like. One command does it:

```bash
cd /home/lqcd/wdetmold/fft/bench/geom
./tryout.sh L17_rader              # infers L from the name, uses B=8
./tryout.sh L17_rader 17 256       # explicit L and batch
./tryout.sh L8_radix8 8 512 -fno-tree-vectorize    # extra gcc flags pass through
```

`tryout.sh` compiles only your file into a private scratch directory (never the shared
build tree, so it cannot race with the other implementers), generates data, times it,
verifies against numpy, **re-runs and checks the output is identical** (the repeatability
clause — the timing loop calls your execute thousands of times), and prints the best
library on the same case for reference. Run it constantly.

**What local timing can and cannot tell you.** The login node is a shared 48-thread
Haswell with other people's jobs on it. Treat its numbers as *relative only* — good for
"did this change help", useless as a reported result. Three specific traps:

* **No AVX-512.** An `#ifdef __AVX512F__` path does not even compile here without an
  explicit `-mavx512f`, and cannot execute at all. Your AVX-512 code is unverified until
  it runs on the benchmark node.
* **Different cache hierarchy.** Haswell has **256 KB of L2 per core**; the benchmark node
  has **1 MB**. Any blocking or tile size you tune locally is tuned against the wrong
  cache — this matters most at L=36 (746 KB per volume: 2.9× the local L2 but under the
  node's) and for the L=8 batched cases.
* **Shared and noisy.** Run-to-run spread of tens of percent is normal here. Small
  differences measured locally are not real.

**Taking your own measurement on the real machine.** When a decision actually depends on
the target hardware — "is my AVX-512 kernel faster than my AVX2 one", "which tile size
wins at L=36" — measure it there yourself:

```bash
./probe_node.sh L36_pfa                      # sensible batch points for that L
./probe_node.sh L8_batchsimd --batches "1 64 2048"
```

That submits a short `--exclusive` job which builds only your file (plus an MKL baseline)
on the benchmark node, times both on identical data, and verifies correctness. Results
land in `results/probe_<name>/probe-<jobid>.out`.

**Queue etiquette, please respect it:** the `devel` partition has two nodes and other
people share this cluster. One probe at a time, keep it short, never resubmit in a loop
while one is pending. Iterate with `tryout.sh`; use `probe_node.sh` to settle a decision,
not to explore. Put whatever the node tells you into your strategy record — there is no
AVX-512 measurement anywhere in the literature corpus, so those numbers are new
information.

## What you are up against (isolated node, round `sota_r1`, per transform)

| L | regime | best library | time | GF/s | who |
|---|---|---|---|---|---|
| 6 | B=1 | MKL 2022 | 0.370 µs | 22.6 | libraries strong |
| 8 | B=1 | MKL 2022 | 0.653 µs | 35.3 | libraries very strong |
| 8 | B=2048 | MKL 2022 | 1.349 µs | 17.1 | memory-bound; 2× worse than B=1 |
| 17 | B=1 | FFTW patient | 81.68 µs | 3.7 | **every library does badly here** |
| 36 | B=1 | MKL 2022 | 163.6 µs | 22.1 | MKL 1.8× ahead of FFTW |

The library-free floor (`baseline_matrix`, dense DFT matrix per axis) is 22–66× slower
than the best library, so beating *it* means nothing. The bar is the library column.

`L=17` is the standing opportunity: at 3.7 GF/s the libraries are running ~6× below
what they achieve at L=8, because 17 is prime and their generic path handles it badly.

## Keep a strategy record (required)

Maintain `strategies/<your-filename>.md` — e.g. `strategies/L17_rader.md`. This is not
paperwork: downstream rounds optimize *from* these records, and an idea that failed for
a documented reason is as valuable as one that worked, because it stops the next
implementer repeating it. Append a new section each round rather than overwriting, so the
history of what was tried survives.

Each round's entry should carry:

* **Technique** — the algorithm and index mapping, precisely enough to reimplement.
* **Derivation / operation count** — multiply-adds per transform, and how you got there.
  Compare against `python/fft3d.py`'s `line_cost()` where it applies.
* **Layout and SIMD decisions** — what lives in registers, what the vector lanes hold,
  where the data is repacked and what that cost.
* **What was measured** — your local numbers for B=1 and batched, and the node numbers
  once the monitor reports them. State the machine.
* **What was tried and did NOT work** — with the number that killed it. This is the most
  useful part of the record.
* **Next** — the specific thing you would do with another iteration, and why you expect
  it to pay.

## Reference and literature

* `../../python/slow_dft.py` — the definition, transcribed; the ultimate ground truth.
* `../../python/fft3d.py` — the textbook FFT (mixed-radix, Rader, Bluestein) with the
  derivations in its docstrings; `line_cost()` gives exact multiply-add counts.
* `../../docs/LITERATURE.md` and `../../docs/literature/` — the cited corpus on
  small-`n` codelets, prime lengths, vector radix, SIMD layout, blocking, autotuning,
  and register-level fusion. Start from the per-size strategy table in `LITERATURE.md`.
