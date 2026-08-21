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

## Where to develop: wallaby

**Develop and measure on `wallaby`**, not on the login node you land on and not on the
benchmark node. It shares this filesystem, so there is nothing to copy:

```bash
cd /home/lqcd/wdetmold/fft/bench/geom
./tryout.sh --on wallaby L17_rader            # infers L from the name, B=8
./tryout.sh --on wallaby L17_rader 17 256     # explicit L and batch
./tryout.sh --on wallaby L8_radix8 8 512 -fno-tree-vectorize   # gcc flags pass through
```

`tryout.sh` builds only your file into a private scratch directory (never the shared
build tree, so it cannot race with the other implementers), generates data, times it,
verifies against numpy, **re-runs and checks the output is bit-identical** (the
repeatability clause — the timing loop calls your execute thousands of times), and prints
the best library on the same case. Run it after every change.

| | `wombat` (default login) | **`wallaby` (use this)** | benchmark node (monitor only) |
|---|---|---|---|
| CPU | Xeon E5-2680 v3, Haswell | **Xeon Gold 6448Y, Sapphire Rapids** | Xeon Gold 5218, Cascade Lake |
| cores | 24c / 48t, shared and busy | **64c / 128t, near-idle** | 32c / 64t, exclusive |
| AVX-512 | **none** | **full, incl. fp16/bf16/vbmi2** | full (f/dq/bw/vl/cd/vnni) |
| L1d / L2 per core | 32 KB / 256 KB | 48 KB / **2 MB** | 32 KB / **1 MB** |
| run-to-run spread | ~0.4% and up | **~0.04%** | ~0.3% (exclusive) |

wallaby is the right development machine on every count: it is idle, so its numbers are
stable enough to see a small change; and it has AVX-512, so your `#ifdef __AVX512F__`
path can actually be **run and verified**, not merely compiled.

**But wallaby is not the scoring machine, and two differences matter.**

* **L2 is 2 MB per core on wallaby against 1 MB on the benchmark node.** A tile or
  blocking parameter tuned on wallaby can be twice too large for the machine you are
  scored on. This bites hardest at L=36 (746 KB per volume) and in the L=8 batched cases.
  Where a blocking size matters, make it a compile-time constant that is easy to change,
  say why you chose it in your strategy record, and expect the monitor's numbers to
  revise it.
* **AVX-512 frequency behaviour is completely different.** Sapphire Rapids runs 512-bit
  code at essentially full clock; Cascade Lake applies severe licence-based downclocking
  (the corpus documents a Gold 5120 going 2.7 GHz scalar → 2.3 AVX2 → **1.6 AVX-512**).
  So a 512-bit kernel that wins on wallaby can lose to a 256-bit one on the scoring node.
  If you write an AVX-512 path, keep the AVX2 path working and competitive, and say in
  your record which you expect to win where. Nobody has measured this for our sizes.

For calibration: at L=8, B=64 MKL runs at 50.1 GF/s on wallaby and 17.3 GF/s on wombat —
a 2.9× spread on identical code. Never compare a number from one machine to a number from
another.

## The benchmark node is the monitor's

Do **not** submit slurm jobs. The exclusive benchmark node is reserved for the monitor
agent's cross-checks, so that every scored number is taken on an uncontended machine, by
one party, with one method — which is the only way the leaderboard means anything. The
partition is also shared with other people's production work.

`probe_node.sh` exists but refuses to run without `FFT_MONITOR=1`. If you believe you
need a measurement only the benchmark node can give, say so in your return value and in
your strategy record, and the monitor will take it for you in the next round.

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

## Learn from previous generations (from round 2 onwards)

After the first round this stops being an independent competition and becomes a cumulative
one. You are **expected** to read what everyone else has already done and to take whatever
helps:

* `strategies/*.md` — every implementation's own account of what it tried, what it
  measured, and what failed. Read the records for **other geometries too**, not just your
  own: a layout or SIMD trick that won at L=8 often transfers to L=6 or L=36, and the
  L=17 entries have the most unusual structure of the four.
* `exemplars/<round>/` — the code that was judged worth keeping from earlier rounds, with
  its strategy record and its measured numbers alongside.
* `results/<round>/leaderboard.txt` and `VERDICT.md` — where you actually stand, and the
  monitor's reading of what moved.
* `results/<round>/context.md` — an index of all of the above, assembled for you.

Two obligations come with that access. **Attribute what you borrow**: if you adopt an idea
from another entry, name the entry in your strategy record. And **do not rediscover a
documented dead end**: if a record already shows an approach failing, with the number that
killed it, spend your round on something else. The whole point of the records is that the
panel gets smarter each round instead of re-running the same experiments.

Your rivals on your own geometry are fair game too. If another entry for your `L` is
beating you, read its code, understand why, and either adopt its idea or find the thing it
cannot do.

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
