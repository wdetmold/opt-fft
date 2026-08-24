# gen_twiddle — exact-twiddle tables + accuracy audit (library layer)

Scored by ADOPTION. This record doubles as the layer's user manual AND the
panel's twiddle accuracy audit: if you build any trig table anywhere, the
section "The audit report" tells you how much error your current pattern
carries, measured, with the file:line it lives at.

## Round gen_r1

### What shipped

`impl/gen_twiddle.c` was rewritten from the dense stub into two parts, in the
gen_layout single-file style:

1. **The library** (top of file): all-`static inline`, `tw_`-prefixed, zero
   link footprint. Adopt with

   ```c
   #define GEN_TWIDDLE_LIB_ONLY
   #include "gen_twiddle.c"       /* impl/ is the include dir */
   ```

   Compiles clean with `-Wall -Wextra` on wallaby and a80n0.

2. **A demo entry** (below the guard): any-L (2..128) row-column 3D FFT whose
   axes are generic mixed-radix DIT pencils (8 pencils wide, split complex,
   radix 4 preferred then smallest-prime), with every table built by the
   layer's fillers, ulp-audited at create(), and the whole 1-D engine gated at
   create() against the layer's long-double DFT oracle (rel L2 ≤ 5e-15 or the
   plan refuses). It is the layer's living test bench and an existence proof
   for every round-6 surprise size; NOT a contender at any scored size.

### The API (what each piece is, and the lesson it encodes)

| function | what | lesson / source |
|---|---|---|
| `tw_cisl / tw_cis` | exp(−2πi·num/den): exact `num mod den`, OCTANT fold (trig argument always ≤ π/4), long-double cosl/sinl, one rounding to double | ≤ 0.5 ulp per part at ANY den (trig condition ≤ 1 after the fold); quarter turns bit-exact (±1, ±0) so trivial-twiddle special cases stay trivial; FFTW note: "inaccurate twiddles are the most likely reason for FFT inaccuracy" (LITERATURE §04 §5.3, §07 §5.1–5.3) |
| `tw_chirp` | Bluestein chirp exp(−iπk²/n), k² mod 2n reduction done internally, overflow-safe | the k²-reduction is the classic chirp bug; borrowed from gen_bluestein's exact `(j·j) mod 2L` and hardened |
| `tw_fill_ct_split` | CT twiddles w_N^(s·k1), k1-major, s=1..r−1 inner, split re/im | consumption order for a k1-sweeping combine loop (this demo's engine) |
| `tw_fill_ct_int_rowmajor` | j1-major, k2 inner, interleaved pairs | drop-in for gen_planner's `pln_x->tw` layout (`x->tw + 2*(j1-1)*m`, k2 inner) |
| `tw_fill_ct_int_colmajor` | j-major, s inner, interleaved pairs | gen_bluestein's per-stage `*tf++ = cos; *tf++ = sin` order |
| `tw_fill_dft_split` | dense DFT matrix, split, padded row stride | gen_dense_prime / gen_layout-demo shape |
| `tw_fill_dft_cplx` | dense DFT matrix, C99 complex row-major | drop-in for gen_race's `p->w`, gen_powp/gen_rader dense fallbacks, and the `refnd` gate references |
| `tw_splat8` | duplicate a table 8× per value for broadcast-free zmm loads | gen_batchlane / dup-pair styles |
| `tw_ulp_err / tw_cis_err` | error of a stored value vs the exact root, in double ulps; at true zeros (quarter turns) a stray is measured in ulps-at-1.0 | the audit metric; ≤ 0.51 means "as good as correctly rounded" |
| `tw_audit_ct_split / _int_rowmajor / _int_colmajor / tw_audit_dft_split / _cplx` | max ulp over a whole table, recomputed in long double | assert ≤ 0.51 in your create() the way gen_layout has you assert `gl_selftest()` |
| `tw_ld_dft` | O(n²) long-double DFT, octant-exact roots computed on the fly | plan-time ground truth for Rader kernels (`FFT of w^{g^k}`), Bluestein `bh`, and per-stage error budgets. O(n²) trig calls — plan-time only |
| `tw_selftest` | naive-fold cross-check (absolute 1.2e-16 at the \|w\|=1 scale), quarter/eighth-turn bit-exactness, \|w\|²−1 ≤ 5e-19, addition theorem ≤ 2e-18, chirp identities | run once in create(); an octant/sign bug shows as O(1), not 0.5 ulp |

### How to adopt (60 seconds)

```c
#define GEN_TWIDDLE_LIB_ONLY
#include "gen_twiddle.c"

/* CT stage twiddles, your layout: */
tw_fill_ct_int_rowmajor(tw, N, r, m);              /* planner-style */
if (tw_audit_ct_int_rowmajor(tw, N, r, m) > 0.51) abort();  /* in create() */

/* Bluestein: */ tw_chirp(j, L, &chre[j], &chim[j]);
/* Rader:     */ tw_cis(modpow(g, k, p), p, &c, &s);   /* generator-ordered roots */
/* dense:     */ tw_fill_dft_cplx(p->w, L);            /* replaces double cexp    */
/* anything else: tw_cis(num, den, &c, &s) is the primitive; build your own
   order and audit per element with tw_cis_err(). */
```

### THE AUDIT REPORT (measured, max ulp per component, exact reference = octant long double)

Formula patterns over dense k·j mod L tables, L ∈ acceptance set ∪ {127, 128}:

| pattern | max ulp | who uses it |
|---|---|---|
| double `cexp(-2πi·m/L)` | **250.6** | gen_powp.c:460, gen_pfa_large.c:442 (`refnd` gate reference) |
| double `cos/sin(-2π·m/L)` | **250.6** | gen_powp.c:533, gen_rader.c:38 (stub), baseline_matrix.c:37 (all dense fallbacks) |
| long double, no fold (angle ∈ (−2π, 0]) | 0.497 | gen_race.c:791, gen_pow2.c:268, gen_dense_prime.c:70 (positive-angle variant) |
| long double, ±π fold | 0.497 | gen_planner.c:136 `pln_omegal` |
| `tw_cis` (octant fold) | 0.497 | this layer |
| chirp: long double, exact k² mod 2n, no fold | **0.554** | gen_bluestein.c:512 |
| chirp: `tw_chirp` | 0.500 | this layer |

Same long-double patterns as den grows (why the octant fold is the right
primitive for a GENERAL library, not a nicety):

| den | no fold | ±π fold | octant |
|---|---|---|---|
| 256 | 0.483 | 0.483 | 0.483 |
| 1024 | 0.502 | 0.501 | 0.499 |
| 4096 | 1.518 | 0.533 | 0.500 |
| 65536 | 20.9 | 3.91 | **0.500** |
| 1048576 | 132.7 | 83.4 | **0.500** |

Findings, per entry (honesty first: nobody's SHIPPING kernel has a twiddle
accuracy problem this round — every hot-path table in the panel is already
long double and den ≤ 512, i.e. ≤ 0.55 ulp):

- **gen_powp / gen_pfa_large**: the `refnd` create()-gate reference builds W
  with double `cexp` → up to 250 ulp ≈ 5.6e-14 relative per twiddle. As a
  1e-13 gate yardstick that leaves less than one decade of margin at L = 100.
  One-line fix: `tw_fill_dft_cplx(Wt-as-matrix)` or `tw_cis(k, L, ...)` per
  root. Same for the dense fallback fills (gate-visible paths only).
- **gen_bluestein**: chirp measured 0.554 ulp (the no-fold tail at den up to
  512). Their record already says "defer to the twiddle layer" — `tw_chirp`
  is the exact drop-in, and `tw_fill_ct_int_colmajor` matches their stage
  order. Their inverse twiddles are conjugates: fill forward, negate im.
- **gen_race / gen_pow2 / gen_dense_prime / gen_planner**: current tables fine
  at their dens (≤ 0.5 ulp). Adopting buys the audit (regression insurance
  when the planner starts emitting NEW dens) and bit-exact quarter turns
  (no-fold leaves ±i twiddles ~2.5e-20 off zero — invisible numerically,
  but it breaks "trivial twiddle" detection by equality and bit-repeatable
  table diffs).

### What I measured on the node (a80n0, Ice Lake, pinned core, graded chain workload, min per transform)

Demo entry, default build (`gcc -O3 -march=native`, ymm auto-vectorization):

| L | B | m | µs/xform | rel L2 | map-chain |
|---|---|---|---|---|---|
| 12 | 64 | 600 | 27.115 | 2.96e-16 | PASS 4.85e-14 (anchor 3.89e-14, tol 1e-10) |
| 12 | 1 | 600 | 26.681 | 2.99e-16 | |
| 25 | 16 | 256 | 262.284 | 3.70e-16 | |
| 27 | 16 | 200 | 431.773 | 3.88e-16 | PASS 2.98e-14 (anchor 2.57e-14) |
| 31 | 16 | 140 | 1125.602 | 4.63e-16 | |
| 32 | 8 | 250 | 650.895 | 3.25e-16 | |
| 50 | 4 | 128 | 2828.650 | 4.52e-16 | |
| 100 | 1 | 64 | 26786.1 | 4.82e-16 | |

Bit-identical output across runs (verified manually; see harness note below).
Setup ≤ 6 ms at L=100 (selftest + fills + audits + 1-D gate), far under the
60 s budget. Brute long-double 3D check on wallaby: rel L2 1.5–3.7e-16 for
every L ∈ {2..50 all supported}, EXACTLY 0 at L = 2 and 4 (pure trivial
twiddles — the bit-exactness claim made visible); create+execute smoke OK at
64, 100, 108, 125, 127, 128. Per-axis 1-D error is ~1e-16 rms, i.e. ~1/100th
of the 1.5e-14/step contract — the budget is dominated by the transform's own
reassociations, never the tables.

Context, not competition: gen_layout's vectorized dense demo is still faster
below L≈25 (16.15 µs vs my 27.1 at L=12 — hand AVX-512 beats compiler codegen
at tiny n), crossover by L=50 (2940 vs my 2829), and at L=100 the O(L·ΣR) vs
O(L²) per-pencil gap shows: 42496 vs 26786 µs. Class entries beat both by
design.

### What I tried that did not work (with the number that killed it)

- **First engine: generic dense r×r combine, single accumulator pair per
  output row, radix-2-first.** L=12 B=64: 67.7 µs, L=32 B=8: 1878 µs. The
  combine was FMA-latency-bound (one chain of length r per row) and radix-2
  levels paid full complex multiplies by ±1. Fix: radix 4 preferred in the
  factorizer, exact-constant butterflies r=2/3/4/5, dense fallback unrolled 4
  output rows (16 independent FMA chains). Same file, 27.1 µs and 651 µs —
  2.5–2.9× from ILP + trivial-constant structure, zero table changes. Lesson
  for adopters: at these sizes the tables were never the bottleneck; the
  butterfly's dependency structure was.
- **`-mprefer-vector-width=512`**: L=100 B=1: 28679 vs 26786 µs; L=12 B=64:
  30.2 vs 27.1 µs. zmm loses to ymm on this code shape on Ice Lake; default
  kept. (Consistent with gen_layout's intrinsics demo winning at small L —
  512-bit pays only when hand-scheduled.)
- **ulp-vs-naive selftest comparison**: first selftest compared `tw_cis`
  against the naive fold in ULPS OF THE NAIVE VALUE and false-alarmed at
  every quarter turn — naive gives ~5e-20 where the true value is exactly 0,
  and "ulp at 5e-20" is 2^-116. The metric, not the primitive, was wrong.
  Fixed two ways: selftest uses absolute tolerance 1.2e-16 at the |w| = 1
  scale; `tw_ulp_err` measures strays at true zeros in ulps-at-1.0 (lower
  bound of any honest ulp reading, keeps the 0.51 gate meaningful for
  adopters' tables).

### Borrowed from other entries

- **gen_layout**: the single-file `*_LIB_ONLY` adoption pattern wholesale;
  the demo adopts `gl_arena` (THP, prefaulted, staggered) for ALL plan
  memory; their harness notes (tryout's remote map-check `$W` expansion hole,
  slurm PATH for reserve.sh) saved me an hour — both still present this
  round, workaround: run `check.py --map-check` yourself after tryout, and
  note the `&&`-chain means tryout's repeatability check silently never runs
  for chain cases (I verified bit-identical output by hand on the node).
- **gen_planner**: `pln_omegal`'s exact-integer-reduction idea is the seed of
  `tw_cisl`; I generalized the fold from ±π to octant and matched my
  rowmajor filler to their consumption layout so adoption is a two-line diff.
- **gen_bluestein**: the exact `(j·j) mod 2L` chirp reduction — hardened into
  `tw_chirp` (pre-reduce k so k² cannot overflow), and their stage order is
  `tw_fill_ct_int_colmajor`.

### What I would do next (gen_r2)

1. Wire the fillers into the first willing adopters — gen_bluestein asked
   explicitly (their next-steps item 3), gen_powp/gen_pfa_large's `refnd` is
   a one-liner. Adoption is my score; a PR-style diff in each owner's round
   prompt is the plan.
2. `tw_fill_rader(p, g, ...)`: generator-ordered root sequence + its
   long-double FFT via `tw_ld_dft`, so gen_rader's kernel table is one call.
3. Per-stage error-budget helper: given a factorization chain, return the
   predicted rms growth (√Σops model) so the planner can reject chains that
   would eat the 1.5e-14/step budget before racing them.
4. Demo speed is NOT a priority (it is a test bench), but if idle: hoist the
   pencil gather into gen_layout's `gl_pack8` and stop re-gathering axis 1
   per plane.
