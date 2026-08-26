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

## Round gen_r2

### Adoption status (the score)

- **gen_bluestein ADOPTED the layer this round**: `tw_chirp` for their chirp
  table and `tw_fill_ct_int_colmajor` + `tw_audit_ct_int_colmajor` for every
  per-stage twiddle table, forward and (conjugated) inverse, asserted ≤ 0.51
  ulp in their create() (impl/gen_bluestein.c:523–548). Exactly the drop-in
  their r1 record asked for. API kept 100% backward compatible this round so
  their include keeps compiling (verified: `gcc -c impl/gen_bluestein.c` clean
  against the new file).
- **gen_planner changed their CT twiddle layout in their r2** (their record,
  "Notes for adopters": `pln_x->tw` is now k2-major
  `[(k2-1)*(r-1) + (j1-1)]`, k2 ≥ 1 only — my r1 rowmajor filler documented
  their OLD layout). Answered with a new filler this round (below).

### What changed in the library (all additive; nothing existing moved)

1. **`tw_fill_ct_int_k2major` / `tw_audit_ct_int_k2major`** — exactly
   gen_planner's gen_r2 fused-leaf consumption order (verified against their
   fill loop at gen_planner.c:712–720: same index formula, k2 = 1..m-1 outer,
   j1 = 1..r-1 inner, interleaved pairs, no k2 = 0 block). Their `pln_omegal`
   is ±π-fold long double (fine at den ≤ 16384); adopting buys the octant
   fold, bit-exact quarter turns and the audit, as a two-line diff.
2. **Rader helpers** (my r1 next-steps item 2, built for gen_rader's round-3
   any-prime mandate and gen_planner's rad-p nodes):
   - `tw_modpow(b, e, p)` — exact for p < 2³¹;
   - `tw_primroot(p)` — smallest primitive root, trial-factors p−1, returns 0
     on composite input (fatal-signal, not a guess);
   - `tw_fill_rader_seq(w, p, g, dir)` + `tw_audit_rader_seq` — the roots
     w_p^(g^q) (dir ≥ 0, input-gather order) or w_p^(g^−q) (dir < 0, the b
     kernel) as interleaved pairs in generator = consumption order;
   - `tw_fill_rader_fft(V, p, g, scale_inv)` — DFT_{p−1} of the b-sequence
     computed END-TO-END in long double (octant-exact roots + `tw_ld_dft`),
     ONE rounding to double, optional 1/(p−1) folded in. This is the
     plan-time kernel table a Rader convolution multiplies by; gen_rader's
     generalization beyond p=31 is one call plus their index tables.
3. **`tw_selftest` extended** (still the thing you assert once in create()):
   primitive-root orbit checks (g^q visits all p−1 residues and closes) at
   p ∈ {3,5,7,13,17,31,101,127} + composite rejection; Rader table identities
   at p=13 (seq audit ≤ 0.51 ulp both directions; kernel V[0] = Σ nontrivial
   roots = −1 to 1e-14; Parseval Σ|V|² = (p−1)² to 1e-12; scaled variant
   V[0] = −1/(p−1)); a k2-major fill/audit round trip. Cost added: ~150 extra
   tw_cisl calls — negligible next to the existing chirp sweep.

### The demo: zmm lanes via GNU vector extensions (−25% to −49% on the node)

The r1 demo's 8-pencil split-complex loops (`double [8]` + q-loops) were
auto-vectorized by gcc to **ymm only**. gen_bluestein's r1 record called this
exactly ("gcc targets 256-bit even when it does vectorize; explicit 512-bit
is the only reliable path under the fixed harness flags"). Rewrote the hot
paths as explicit 64-byte GNU vector types — `tw_v8` (8 doubles, may_alias),
one vector per lane-row:

- butterflies r=2/3/4/5 and the dense 4-row-unrolled combine: same arithmetic,
  same evaluation order, one zmm op per line (scalar constants broadcast);
- twiddle multiply in the combine sweep: broadcast-FMA on whole lane-rows;
- `twd_gather_i`/`twd_scatter_i` full groups: 2 unaligned loads + 2 two-source
  `__builtin_shuffle` (vpermt2pd) de/interleave per row — was 16 scalar moves;
- `twd_gather_z`/`twd_scatter_z` full groups: in-register 8×8 double transpose
  (`tw_tr8x8`, 24 shuffles per 4 elements — gen_bluestein's tr8x8 idea,
  reimplemented with `__builtin_shuffle`); scalar tail lanes/elements kept.

No intrinsics headers, no #ifdef forest: GNU vector extensions lower to
scalar code on any non-AVX-512 target, so the file stays portable and the
node build gets full zmm. Operation COUNT is unchanged from r1 (same
radix-4-first factorization, same tables); only lane width and the
gather/scatter instruction economy changed.

Measured on the node (a80n0, leased core via tryout.sh, graded chain, min;
sd ≤ 0.7% every run):

| L | B | gen_r1 | gen_r2 | delta | rel L2 |
|---|---|---|---|---|---|
| 10 | 64 | 18.13 | **13.09** | −28% | — |
| 12 | 64 | 27.07 | **19.30** | −29% | 2.96e-16 |
| 12 | 1  | 26.68 | **21.49** | −19% | 2.99e-16 |
| 25 | 16 | 260.7 | **181.9** | −30% | 3.70e-16 |
| 27 | 16 | 431.8 | **299.5** | −30% | 3.88e-16 |
| 31 | 16 | 1125.6 | **566.2** | −49% | 4.63e-16 |
| 50 | 4  | 2828.7 | **2098.3** | −25% | 4.52e-16 |
| 100 | 1 | 26786 | **18938** | −29% | 4.82e-16 |

L=31's dense-leaf −49% is the cleanest read of the ymm→zmm effect (pure
broadcast-FMA). Chain timings still include the driver's scalar map (the demo
does not own fft3d_chain — it is a test bench, not a contender).

Gates, all rechecked manually (tryout's remote map-check still dies on the
`'$W/c.bin'` quoting bug): two-step 9.5e-16 (L=12) / 1.6e-15 (L=27) /
2.9e-15 (L=100) vs tol 3e-14; full chain m=600 at L=12 5.19e-14 (anchor
3.89e-14, tol 1e-10); bit-repeatable across runs (manual cmp); local numpy
sweep of 32 sizes L ∈ {2..32 all, 45, 50, 64, 100, 101, 108, 125, 127, 128}
ALL PASS, worst 1.02e-15 (L=101), typical 3–5e-16. setup ≤ 5 ms at L=100.

### What did not work / superseded negatives

- Nothing tried this round regressed — but the r1 negative
  "`-mprefer-vector-width=512` loses (30.2 vs 27.1 µs at L=12)" is now
  SUPERSEDED and should not deter anyone: explicit vector types measure 19.3
  µs on the same case. The flag's loss was the autovectorizer's zmm codegen
  (masked epilogues, gather choices), not 512-bit itself. If your kernel is
  already 8-lane-shaped, write the vectors explicitly; do not conclude "zmm
  loses on Ice Lake" from the flag experiment.
- `tw_primroot` subtlety worth recording for adopters: the factor test alone
  (`g^((p-1)/q) != 1`) does NOT reject composite p — it can return a bogus
  "root". The shipped version also requires `g^(p-1) = 1` (Fermat), which
  makes composite input return 0, and the selftest asserts exactly that at
  p = 9 and 15. A nonzero return is trustworthy only because of that check;
  do not strip it for speed (it is plan-time).

### Borrowed this round, named

- **gen_bluestein**: the explicit-512-bit lesson (their r1 item 3) — the
  whole demo speedup is that lesson applied; and the 8×8 in-register
  transpose idea (their `tr8x8`) for my z-axis gather/scatter.
- **gen_planner**: the k2-major layout spec straight from their r2 record's
  adopter note — the new filler exists because they documented the change.

### What I would do next (gen_r3)

1. **gen_rader adoption**: their record plans exactly what
   `tw_primroot`/`tw_fill_rader_seq`/`tw_fill_rader_fft` provide ("needs a
   small generator for the index/sign tables"); a worked example in their
   prompt (p=41: primroot, seq both directions, kernel FFT, audit) is the
   cheapest adoption of the round.
2. **gen_powp / gen_pfa_large refnd**: still building their create()-gate
   reference W with double cexp (250 ulp); `tw_fill_dft_cplx` remains the
   one-line fix. Re-pitch with the r1 audit-table numbers.
3. Per-stage error-budget helper (unchanged from r1 item 3).
4. Demo if idle: own fft3d_chain (volume-major in-place + NR-ladder map — the
   panel-standard scheme; the driver's scalar map is most of the gap to
   gen_planner at small L), and stop re-gathering axis 1 per plane.

## Round gen_r3

### Adoption status (the score)

- **gen_bluestein's r2 adoption stands** (tw_chirp + colmajor filler/audit in
  their create()); API kept 100% backward compatible again — `gcc -c
  impl/gen_bluestein.c` verified clean against this round's file.
- **gen_planner rewrote their executor in intrinsics this round and grew two
  NEW table layouts**; both now have exact fillers + audits in the layer,
  **verified slot-for-slot against their own builders** (a test TU included
  both files and compared every slot at n = 7, 11, 13, 31: MATCH):
  - `tw_fill_dense_simd` / `tw_audit_dense_simd` = their `pln_dense_matrix`
    PLN_SIMD layout (WR[np][n] k-major real parts, then (−wi,+wi)
    broadcast_f64x2 pairs at np·n, zero pad rows k ≥ n — the audit also
    checks the pads ARE zero, since the tiled kernel multiplies them).
  - `tw_fill_fold_half` / `tw_audit_fold_half` = their `pln_fold_matrix`
    (folded odd-n half-system: C[k][j] = cos(2πkj/n) rows 1..h padded to hp,
    then S[k][j] = sin(2πkj/n); also the gen_dense_prime / gen_rader /
    gen_layout-demo fold shape). Their `pln_omegal` is ±π-fold long double —
    fine at their dens — so the pitch is the audit + bit-exact quarter turns
    + one less private table builder, as a two-line diff each.
- **NEW for gen_rader's round-3 any-prime mandate**: `tw_fill_rader_half` /
  `tw_audit_rader_half` — the FOLDED Rader form's split real kernels over the
  quotient group Z_p^× mod {±1} (cyclic of order h = (p−1)/2; any primitive
  root of p generates it, g^h = −1 so q = 0..h−1 hits each class {e, p−e}
  exactly once). Emits cos(2π g^{±q}/p) and sin(2π g^{±q}/p) split, in
  generator = consumption order: cw feeds their cyclic-h correlation (E), sw
  the negacyclic one (O); the (−1)^q twist/sign tables stay theirs (engine
  index tables, not trig). With `tw_primroot` + this + `tw_fill_rader_fft`,
  the any-prime plan-time table work is three calls.
- `tw_selftest` extended with fill/audit round trips for all three new
  fillers (including a fold spot value at n=15, k·j = n: cos exactly 1, sin
  exactly 0) and a quotient-orbit coverage check at p=13. Everything
  additive; nothing existing moved (adopters recompile unchanged).

### The demo: owned fft3d_chain — in place, volume-major, map fused (−18% to −40%)

My r2 next-steps item 4, and the last panel-standard chain lesson the demo
had not cashed (gen_planner r2's "own the chain" + gen_rader r1's in-place
insight, applied to this engine): the driver fallback charged the demo
execute (src → T → dst, a full intermediate volume T) + ping-pong between two
volumes + a full-volume scalar-ish map pass per step. The demo's
gather-to-lanes structure makes every axis in-place safe FOR FREE (each
8-pencil group is copied into the lane buffers before anything stores), so
the owned chain runs all three axes in place on the state volume — T is
never touched, the ping-pong is gone — and the map z/(1+|z|) is fused into
the axis-2 scatter: z-pencils are contiguous rows, so c is read interleaved
with NO transpose, and the map runs on the tr8x8 outputs while they are in
registers. Map arithmetic: rsqrt14 + 2 Newton for |z| (~5e-17 rel), then ONE
well-rounded vdivpd (raced against the rcp14+2NR ladder — see below);
1e-300 clamp; exact scalar sqrt/div on tail lanes.

create() gains a CHAIN GATE (ice L17_rader r5 discipline, everyone's
pattern): two owned steps vs execute + the driver's exact scalar map on a
deterministic volume, rel L2 ≤ 1e-12 or the chain silently uses the exact
fallback path (execute src==dst is safe in this engine, so the fallback
needs no extra buffer). The gate has never fired; a malloc failure also
falls back. Setup at L=100 grew 6 → 90 ms (gate = 4 volume transforms) —
still 3 orders under the 60 s budget.

Measured on the node (a80n0, leased core via tryout.sh, graded chain, min;
sd ≤ 0.2% except where noted):

| L | B | gen_r2 | gen_r3 | delta | rel L2 (single) |
|---|---|---|---|---|---|
| 10 | 64 | 13.09 | **10.62** | −19% | 2.9e-16 |
| 12 | 64 | 19.30 | **14.70** (14.7–15.3 across windows) | −24% | 3.0e-16 |
| 12 | 1  | 21.49 | **17.50** | −19% | 3.0e-16 |
| 15 | 32 | 39.12 | **30.08** | −23% | 3.4e-16 |
| 20 | 32 | 79.37 | **59.30** | −25% | 3.3e-16 |
| 25 | 16 | 181.9 | **124.7** | −31% | 3.7e-16 |
| 27 | 16 | 299.5 | **239.0** | −20% | 3.9e-16 |
| 31 | 16 | 566.2 | **463.9** | −18% | 4.6e-16 |
| 32 | 8  | 451.7 | **365.0** | −19% | 3.2e-16 |
| 40 | 8  | 853.4 | **663.9** | −22% | 3.8e-16 |
| 50 | 4  | 2098.3 | **1410.6** | −33% | 4.5e-16 |
| 100 | 1 | 18938 | **11353** | −40% | 4.8e-16 |

The gain grows with L exactly as the deleted traffic predicts: at L=100 the
old path streamed state + T + pong + a separate map sweep; the owned chain
streams state + c only. Same-window MKL for scale: 7.91 (12), 145.1 (27),
7815 (100).

Gates on the shipped binary, run by hand on the node (tryout's remote
map-check leg still dies on the `'$W/c.bin'` quoting bug): two-step
9.5e-16 (12) / 1.6e-15 (27) / 2.9e-15 (100) vs tol 3e-14; full graded chains
5.3e-14 (12, anchor 3.9e-14) / 3.1e-14 (27, anchor 2.6e-14) / 3.7e-14 (100,
anchor 2.4e-14), tol 1e-10; chain AND single outputs bit-identical across
independent node runs (manual cmp of .chain files). Local unit TU
(build/tryout/gen_twiddle/chain_unit.c): owned chain vs execute + exact
scalar map ≤ 8.1e-16 at 12 sizes × B ∈ {1,3,8} × m=3, AVX-512 and scalar
(-march=x86-64) builds both; create+gates smoke at 30 sizes incl. 97, 101,
121, 125, 127, 128 — no refusal, no fallback, 1.6 s total on wallaby.

### What did NOT work / raced off, with the number

- **rcp14 + 2NR reciprocal instead of the map's one vdivpd** (`-DTWD_MAPRCP`,
  kept as a knob): same-window alternating A/B at L=12 B=64: div 14.72 vs
  rcp 15.58 (+6%); tie at L=25 (124.7 vs 124.9). Confirms gen_powp's r2
  verdict on their fused x-pass and REVERSES gen_planner's ladder choice on
  their engine — third data point that the divider-vs-ladder call is
  engine-specific: in a scatter-side fused map the divider pipe is idle and
  one exact vdivpd is both faster and better rounded. Measure on your own
  pass; do not copy anyone's verdict (including mine).
- Two ssh round trips lost to the documented "remote commands land in $HOME"
  trap even though gen_powp's record warns about it — put the cd INSIDE the
  quoted remote script, first line, and check it with `|| exit 1`. Also
  tryout.sh needs `/opt/software/slurm-19.05.8.1/bin` on PATH from wallaby
  or it claims there is no reservation (reserve.sh --status is what fails).

### Borrowed this round, named

- **gen_planner r2 ("own the chain") + gen_rader r1 (everything in place) +
  gen_powp r1/r2 (fused map at last-axis stores, div-vs-ladder A/B
  discipline)**: the whole demo chain design is their program applied to the
  lane-buffer engine; the volume-major batch loop is gen_planner's r2 form.
- **gen_powp r2**: the create()-time m=2-style chain gate against execute +
  exact scalar map (theirs gates the soa family the same way).
- **gen_planner r3**: both new table layouts come straight from their record
  and source (pln_dense_matrix PLN_SIMD, pln_fold_matrix) — the fillers exist
  because their record documented the shapes.

### Operation count (demo chain step, per volume)

Axes: unchanged r2 arithmetic (radix-4-first mixed-radix DIT, 3·L² pencils of
O(L·ΣR) each) but zero T traffic and zero ping-pong — state and c are the
only volume-sized streams. Map: per 4 complex, ~14 FMA-class + 1 vpermilpd +
1 vdivpd + 1 rsqrt14, fused at the axis-2 stores; exact scalar tail ≤ 7
lanes per group row. Everything else identical to gen_r2.

### What I would do next (gen_r4)

1. **gen_rader worked example** (p=41: tw_primroot → 6; rader_half both
   directions; rader_fft kernel; audits) pasted into their round prompt if
   the monitor allows — cheapest adoption on the panel now that the folded
   kernels are one call.
2. **gen_powp / gen_pfa_large refnd** double-cexp gate reference: STILL
   unfixed (250 ulp on a 1e-13 gate yardstick at L=100), still a one-liner
   (`tw_fill_dft_cplx`). Third pitch.
3. Per-stage error-budget helper — deferred three rounds; honestly the races
   decide chains empirically and nobody has asked. Drop unless gen_planner's
   sub-tree diversity work (their r3 item 3) creates a real consumer.
4. Demo small-L per-call overhead (L=10 gap to batchlane is 9x): the lane
   gather/scatter per axis is 6 sweeps/volume; a fused zy sweep per x-plane
   (gen_pfa_small's structure) would halve it. Only if idle — the demo is a
   test bench first.

## Round gen_r4

### Adoption status (the score)

- **gen_bluestein's adoption stands** (tw_chirp + colmajor filler/audit in their
  create()); API kept 100% backward compatible again — `gcc -c
  impl/gen_bluestein.c` verified clean against this round's file.
- **No consumer changed a table layout this round**, so the library gained no
  new fillers (checked every r4 record present at write time): gen_planner's
  fused codelets reuse their existing leaf constants and k2-major twiddles
  ("arithmetic and constants byte-identical to the pln_lv* leaves" — their
  words); gen_layout's r4 additions (gl_alias_drained4k, gl_tr8x8_c2i) are a
  store-buffer model and a shuffle network, not trig. Nothing existing moved;
  adopters recompile unchanged.
- **gen_rader worked example, so the folded-Rader adoption is three lines**
  (my r3 next-steps item 1, delivered here since I cannot write their prompt):
  at p = 41, `tw_primroot(41)` returns **6** (h = (p−1)/2 = 20); then
  `tw_fill_rader_half(cw, sw, 41, 6, +1)` gives the input-gather half-kernels
  cos/sin(2π·6^q/41), q = 0..19, `dir = -1` the b-kernel order, and
  `tw_fill_rader_fft(V, 41, 6, 1)` the DFT_40 convolution table with 1/40
  folded in, all end-to-end long double, audited by
  `tw_audit_rader_half(..., ±1) ≤ 0.51`. The (−1)^q twist/sign tables stay
  engine-side.
- **gen_powp / gen_pfa_large refnd double-cexp gate reference** (250 ulp on a
  1e-13 yardstick at L=100): STILL unfixed, still the one-line
  `tw_fill_dft_cplx` swap. Fourth pitch.

### The demo: gated k-plane-blocked axes-1+2 custody (−0.8% at 100, −3.5% at 50, bit-identical outputs)

An x-plane (LL contiguous complex) holds every y-pencil AND every z-pencil of
that x, so after the global strided axis-0 pass, axes 1 and 2 (+ c + map in
the chain) now run fused per block of `kblk = 8/gcd(L,8)` planes while the
block is cache-hot — one full-volume round trip (state read+write between
axis 1 and axis 2) deleted per step. The block size is **gen_bluestein r4's**
(named borrow): kblk·L rows ≡ 0 (mod 8) keeps the axis-2 8-row group
decomposition IDENTICAL to the unblocked pass, and a partial tail block still
starts at a global multiple of 8, so outputs are **bit-identical to gen_r3**
(verified by cmp at 12/25/27/31/40/100/101/127, singles and m=2 chains, on
wallaby AND chain outputs on the node).

**The gate, added after the numbers demanded it**: blocking is enabled only
when the fused pass's two volume-sized streams outgrow L2
(`32·L³ > TWD_BLK_MIN_BYTES`, default 1.25 MB = ICL L2; -D knob for the
cross-arch race — CLX 1 MB, SPR 2 MB). Below it `kblk = L` reproduces the r3
path exactly. In the suite: 10..32 unblocked, 40/50/100 blocked.

Measured on the node (a80n0, same-core interleaved pairs vs a rebuilt r3
control binary — protocol below; min µs/xform, graded chain):

| L | B | r3 ctl (same window) | gen_r4 ship | delta | note |
|---|---|---|---|---|---|
| 10 | 64 | 10.93 | 10.96 | wash | unblocked (gated) |
| 12 | 64 | 15.00 | 14.88 | wash | unblocked; B=1: 15.15 |
| 15 | 32 | 30.32 | 29.90 | −1.4% | unblocked |
| 20 | 32 | 57.7–58.7 | 56.7–59.4 | wash (sign flips) | unblocked |
| 25 | 16 | 122.1 | 122.6 | wash | unblocked |
| 27 | 16 | 236.3/238.9 | **236.0/236.3** | parity restored | gate case, see below |
| 31 | 16 | 465.1 | 462.1 | wash | unblocked (953 KB < gate) |
| 32 | 8 | 361–366 | 360–368 | wash (4 pairs) | first-session −2.4% was window luck |
| 40 | 8 | 654–661 | 648–669 | wash (±1%, sign flips across sessions) | blocked |
| 50 | 4 | 1415.6/1430.5 | **1363.3/1382.1** | **−3.5%** | blocked |
| 100 | 1 | 11251/11257 | **11157/11169** | **−0.8%** (quiet best 11123) | blocked; MKL same window 7796 |

Gates, final shipped binary, all by hand on the node (tryout's map-check leg
still dies on the `'$W/c.bin'` quoting bug): two-step m=2 9.5e-16 (12) /
1.6e-15 (27) / 2.3e-15 (50) / 2.9e-15 (100) vs tol 3e-14; full graded chains
5.32e-14 (12, anchor 3.89e-14) / 3.08e-14 (27, 2.57e-14) / 4.47e-14 (50,
2.92e-14) / 3.69e-14 (100, 2.42e-14), tol 1e-10; B=1 chain at 12 PASS
6.97e-14 (anchor 5.80e-14); chain outputs bit-identical across independent
node runs at 12/27/50/100; singles 3.9–4.8e-16. Local: 15-size numpy sweep
PASS (worst 1.02e-15 at 101), 19-size surprise smoke (2..128 incl. 97, 101,
121, 125, 127) PASS, scalar -march=x86-64 build PASS, -DTWD_PF knob and
GEN_TWIDDLE_LIB_ONLY adoption compile clean.

### What did NOT work, with the number that killed it

- **Ungated blocking at L=27**: +0.6/+1.9/+3.8/+0.8% (4/4 same-core pairs;
  7/7 counting earlier sessions). At 32·L³ = 630 KB the volume is already
  L2-resident across passes and the fine-grain alternation of the two loop
  bodies only costs. The residency gate (above) is the fix; with it, 27 reads
  parity (even −0.5% in the confirm pairs).
- **Axis-0 gather prefetch (gen_layout r4's fold-load idea, +256 B, 2
  lines/row): REJECTED on measurement.** Same-core pairs, blocking held
  constant: +1.5–2% at L=100 (3/3 pairs), +0.5–1.7% at L=31 (4/4), wash at
  L=12. Their fold walks ~L short row streams per group; my gather reads only
  2 lines per row per group and the OoO window already covers them — the
  extra load-port µops never pay. Kept compilable as `-DTWD_PF` for the CLX
  advisory, default OFF. Do not conclude gen_layout's demo prefetch is wrong
  — it isn't, on their loop shape; the lesson is the transfer failed and only
  the A/B could tell.
- **Method: the tryout core-hop confound (gen_batchlane r4) is real and bit
  me within the hour.** My first tryout read of the new code at L=100 was
  11225 (looked like a win); the first same-core interleave then showed it
  LOSING 3/3 pairs to r3 — that binary carried the prefetch. One later
  session had a neighbor land mid-interleave (readings doubled to 23–25 ms,
  both arms); those pairs were discarded, not averaged. Every keep/kill above
  is from held-lease alternating-binary pairs.

### Borrowed this round, named

- **gen_bluestein r4**: the k-plane block size `8/gcd(L,8)` that makes the
  blocked pass's group decomposition provably identical to the unblocked one
  (their custody design, itself from gen_layout r3's plane window).
- **gen_layout r3/r4**: the cache-custody idea behind the whole change, and
  the prefetch experiment (adopted as a knob, rejected as a default — their
  own L=100 attribution said prefetch was a wash there, which this confirms
  on a second engine).
- **gen_batchlane r4**: the held-lease same-core interleaved A/B protocol —
  every number in this section that decided anything came from it.

### Operation count (delta vs gen_r3)

Arithmetic, tables, lane sweeps: unchanged (identical bits out). Blocked
sizes lose one full-volume state read+write per chain step / per execute
volume (the axis-1→axis-2 round trip becomes L2-hot inside a ≤ 8-plane
block); unblocked sizes are the r3 loops exactly, plus one plan-time branch.

### What I would do next (gen_r5)

1. **Fuse axis 1 → axis 2 through the lane buffers within a plane** (skip the
   plane-sized scatter/gather round trip entirely, not just keep it hot):
   the plane is already resident; the remaining cost is the shuffle traffic
   of scatter_i + gather_z. Needs twd_rec to emit transposed output — a real
   rewrite, only worth it if the demo's mid-size cells matter to anyone.
2. **Cross-arch**: race TWD_BLK_MIN_BYTES and TWD_PF on CLX/SPR when XARCH.md
   lands (CLX's 1 MB L2 moves the gate; its downclock may flip prefetch).
3. **gen_rader**: check their r4 record for the folded-kernel adoption the
   p=41 example above makes three lines; answer whatever layout they ship.
4. refnd pitch #5 if gen_powp / gen_pfa_large still build gate references
   with double cexp.

## Round gen_r5

### Adoption status (the score)

- **gen_bluestein's adoption stands** (tw_chirp + colmajor filler/audit); API
  kept 100% backward compatible again — `gcc -c impl/gen_bluestein.c` verified
  clean against this round's file after every change.
- **NEW library offering: the dual-select FMA twiddle form** (LITERATURE 11
  Tier 1, Bergach arXiv:2604.00567 — "per-twiddle choice between Linzer-Feig
  factorizations keeps every stored ratio <= 1, 235x tighter error bound,
  zero runtime cost"; the brief asked for a first validation in performant
  software and this round delivers it, see below):
  - `tw_cisl_ds / tw_cis_ds(num, den, &m, &t, &swap)` — the root as
    `w = m*(1+i*t)` (swap=0, |Re| >= |Im|) or `w = m*(t+i)` (swap=1), so
    **|t| <= 1 always**. The octant fold hands the select over for free:
    after reduction the argument is in [0, pi/4], the larger component is
    always the cos one, and the octant index alone fixes slot and signs.
    t is ONE long-double `tanl` of the reduced argument (NOT a quotient of
    two rounded doubles), so each stored constant is within ~0.5 ulp.
    Quarter turns (m,t) = (+-1, +0) exactly; eighth turns t = +-1 exactly,
    m = correctly rounded sqrt(1/2). Multiply kernel is 4 FMA-class ops —
    the SAME count as the (c,s) form:
    swap=0: `m*((ur - t*ui) + i*(ui + t*ur))`; swap=1: `m*((t*ur - ui) +
    i*(t*ui + ur))`.
  - `tw_fill_ct_ds_split / tw_audit_ct_ds_split` — CT tables in the k1-major
    split layout; the audit checks both constants <= 0.51 ulp PLUS the
    structural invariants (|t| <= 1, select actually picks the larger
    component; violations return 1e9, not a fraction of an ulp).
  - `tw_selftest` extended: DS reconstruction vs tw_cisl (abs 2.5e-16 at the
    |w|=1 scale), quarter/eighth-turn exactness, select correctness, and a
    fill/audit round trip.
  - **Who should adopt**: the codelet GENERATORS (gen_planner / gen_pow2 /
    gen_powp fused leaves). Their twiddles are plan-time constants, so the
    select is a per-site constant and no branch survives into the kernel —
    the tighter worst-case bound is literally free error headroom against
    the 1.5e-14/step contract. My demo validated the runtime-branch worst
    case (below): even THERE it costs nothing measurable.
- **refnd double-cexp gate reference, pitch #5**: gen_powp.c:1436 and
  gen_pfa_large.c:784 still build W with double `cexp` (250 ulp on a 1e-13
  yardstick at L=100). `tw_fill_dft_cplx` remains the one-line fix.
- **gen_rader**: their r4 record shows the engine port-model-saturated at 31
  and no table-layout change; the p=41 folded-kernel worked example from my
  r4 section still stands for their next-step item 4 (generalized Rader for
  3|h primes — `tw_primroot`/`tw_fill_rader_half`/`tw_fill_rader_fft` are the
  three plan-time calls).

### The demo: conjugate-fold prime butterflies (-42% at 31) and a codegen lesson worth the round

**1. Fold butterfly for r >= 7** (every default-case radix is an odd prime:
leaves 7..127, combine radices 7/11/13). The dense r x r lane butterfly
(16h^2 FMAs, h = (r-1)/2) is replaced by the conjugate-fold half-system —
the gen_layout / gen_dense_prime / gen_rader fold shape, tables from this
layer's own `tw_fill_fold_half` (hp = h, audited at create): fold a_j = x_j +
x_{r-j}, b_j = x_j - x_{r-j} (4h vector adds), then per k-pair four real
h-correlations (C*a_re, C*a_im, S*b_re, S*b_im) and the conjugate-signed
combine X_k = (E_re + O_im, E_im - O_re), X_{r-k} = (E_re - O_im, E_im +
O_re): ~4h^2 FMAs + O(r) adds, 2 k-rows per iteration = the same 8 independent
FMA chains as the old dense unroll. **~3.5x fewer butterfly FMAs at r=31.**
The demo now finally exercises tw_fill_fold_half itself (it had no consumer
in this engine before).

**2. THE LESSON — the case-bloat tax.** First cut had the fold body INLINE
as the default case of `twd_butterfly`. Same-core interleaved pairs vs the
rebuilt r4 binary: L=31 -42% as expected, but 12/50/100 (which never touch
the fold path and whose outputs were cmp-IDENTICAL) regressed +1..+4.5%,
4/4 pairs each. The big new case bloated the function and gcc de-inlined /
re-scheduled the hot r=2..5 paths inside twd_rec. Fix: the fold body is a
SEPARATE `noinline` function; the default case is a 2-line call. That did
not merely restore parity — the final binary now BEATS r4 at the no-fold
sizes with bit-identical outputs (12: 13.97-14.11 vs 14.83-15.25; 50:
1275-1321 vs 1363-1380; 100: 10734-10862 vs 11362-11464; all 4/4 same-core
pairs), because twd_butterfly ends up SMALLER than r4's (the dense default
is gone from the switch). Panel-transferable: keep rare heavy cases out of
hot dispatch functions; if outputs are bit-identical and time moved, it is
code layout, and an A/B against the previous binary is the only detector.

**3. Pair-packed map ladder (gen_pfa_large gen_r5's -45% map win): REJECTED
here on measurement, kept as -DTWD_MAPPAIR.** Packing the 8 distinct |z|^2
of a vector pair into one zmm (2 permutex2var in, 2 vpermpd out, one ladder
for two vectors) is bit-identical by construction (verified by cmp at
12/50/100, singles and chains) — and still LOSES on this engine: it was in
the first-cut binary and accounts for part of the +1..4.5% above; the
isolated mechanism is gen_rader r4's relocation lesson: my map runs inside
`twd_scatter_z_map`, whose tr8x8 transposes already bind port 5, so 4 more
port-5 shuffles per pair cost more than ~11 saved FMA-class ops on ports
the scatter leaves half-idle. gen_pfa_large's win was on a SEQUENTIAL map
pass with "no port-5 competition" (their words). Boundary recorded: pack
the ladder where the map is a standalone/FMA-bound pass, never inside a
shuffle-bound exit.

**4. Dual-select twiddles in the combine sweep (-DTWD_DS): a measured WASH —
which is the validation.** The demo consumes tw_fill_ct_ds_split with a
per-twiddle branch (pattern fixed per plan level, predictor-friendly).
Same-core pairs vs the non-DS ship binary: L=12 13.74-14.32 vs 14.43-14.46
(4/4 but small), L=31 2/4, L=100 3/4 — i.e. zero cost at the resolution of
the machine, mild lean positive. Accuracy: rel L2 identical to 1e-17
(3-5e-16 both ways); the DS claim is the WORST-CASE bound, not rms, and the
runtime-branch engine is the unfavorable case — for codelet generators it
is free by construction. Shipped default OFF to preserve the bit-identity
regression property vs r4 at the no-fold sizes; the knob and these numbers
are the literature validation (lit 11 Tier 1, first in performant code).

### Measured on the node (a80n0; every keep/kill from held-lease same-core
### interleaved pairs, gen_batchlane r4 protocol; sweep = one lease, 5 samples)

| L | B | r4 board | gen_r5 ship | delta | note |
|---|---|---|---|---|---|
| 10 | 64 | 10.94 | **10.33** | -6% | codegen only (bit-identical output) |
| 12 | 64 | 14.97 | **14.60** | -2.5% | pairs: 13.97-14.11 vs r4ctl 14.83-15.25, 4/4; B=1 14.33 |
| 15 | 32 | 29.67 | **29.17** | -1.7% | |
| 20 | 32 | 58.21 | **54.55** | -6.3% | |
| 25 | 16 | 122.09 | **119.77** | -1.9% | |
| 27 | 16 | 235.30 | **222.10** | -5.6% | |
| 31 | 16 | 463.25 | **270.89** | **-41.5%** | fold; pairs 267-277 vs 471-472 (4/4); dense arm == r4 (attribution clean); B=1 271.2 |
| 32 | 8 | 360.71 | **343.81** | -4.7% | |
| 40 | 8 | 647.06 | **625.16** | -3.4% | |
| 50 | 4 | 1367.70 | **1306.90** | -4.4% | pairs 1275-1321 vs 1363-1380 (4/4) |
| 100 | 1 | 11375.76 | **10735.11** | -5.6% | pairs 10734-10862 vs 11362-11464 (4/4); B=1 10757 |

MKL 2022 same windows: 860.6 (31), 7858.8 (100). The demo now beats ducc0 at
every acceptance size and fftw3_measure at 100 — still a test bench, but no
longer an embarrassing one.

### Gates (ship binary, by hand on the node — tryout's map-check leg still
### dies on the '$W/c.bin' quoting bug)

Singles 2.96-4.82e-16 at 12/27/31/50/100 (tol 1e-12); two-step m=2 9.52e-16
(12) / 1.59e-15 (27) / 1.72e-15 (31) / 2.31e-15 (50) / 2.95e-15 (100) vs tol
3e-14; graded chains 5.32e-14 (12, anchor 3.89e-14) / 3.08e-14 (27, 2.57e-14)
/ 2.48e-14 (31, 2.31e-14) / 4.47e-14 (50, 2.92e-14) / 3.69e-14 (100,
2.42e-14), tol 1e-10; B=1 chain at 12 PASS 6.97e-14 (anchor 5.80e-14); all
chains bit-repeatable across runs; **chain AND single outputs bit-identical
to the r4 binary at 12/50/100** (no fold there, map untouched by default —
the strongest regression check available). Fold-path accuracy IMPROVED
(fewer roundings): L=31 single 4.02e-16 vs r4's 4.63e-16. Local: numpy sweep
16 fold-heavy + 9 acceptance sizes ALL PASS (worst 8.68e-16 at 127); m=3
chains at 7/11/49/91/97/121/127 PASS; scalar -march=x86-64 build PASS at
31/127; all knob combinations (DS/DENSEBF/MAPPAIR/PF) compile -Wall -Wextra
clean; setup <= 0.1 s at L=100.

### What did NOT work / incidents, with the number

- **Pair-packed map as default: +1..4.5% at 12/50/100** (with bit-identical
  outputs) — see boundary above. Kept as -DTWD_MAPPAIR for CLX/SPR.
- **Fold inlined into twd_butterfly: the same +1..4.5%** at sizes that never
  execute it (case-bloat tax, mechanism and fix above).
- **Selftest false alarm at eighth turns**: the DS select check compared
  against long-double |cos| vs |sin| at x = pi/4, where the TRUE components
  tie exactly and the reference breaks the tie at ~1e-19 arbitrarily. Either
  factorization is correct there; the check now allows the tie (1e-12 fuzz,
  well under the >= 1e-3 non-tie gap at den <= 1000). Same metric-vs-
  primitive lesson as my r1 quarter-turn ulp false alarm.
- **Round-start hard-link contamination (process warning for everyone)**:
  impl_5/gen_twiddle.c started the round HARD-LINKED to impl_4/gen_twiddle.c
  (same inode). Early in-place edits landed in BOTH paths — i.e. in the
  archived r4 file — until a later whole-file write split the inodes, which
  also silently dropped two edits from the live file. Detected via a
  gen_layout.c include failure in the r4 control build; fixed with
  `git checkout -- impl_4/gen_twiddle.c` (verified clean) and re-verified
  the live file end-to-end. If your r4 control builds strangely, `stat -c
  %h` your impl_4 file before trusting it.

### Borrowed this round, named

- **gen_pfa_large gen_r5**: the pair-packed ladder (adopted as a knob,
  rejected as a default with the boundary contributed back).
- **gen_rader gen_r4**: the port-relocation lesson, used to diagnose the
  map-pack loss instead of burning more windows on variants.
- **gen_batchlane gen_r4**: the held-lease same-core protocol — every
  keep/kill this round is adjacent-pair on one core.
- **Literature 11 Tier 1 (Bergach)**: the dual-select policy itself; the
  octant-fold implementation and the tie handling are this layer's.

### Operation count (demo, delta vs gen_r4)

r >= 7 butterflies: dense 16h^2+8h FMA-class -> fold 4h^2 FMA + ~10h adds
per group call (r=31: ~3844 -> ~1110). Everything else bit-identical to r4
by default (map, gathers, small radices, blocking). Plan tables: +3 doubles
+1 byte per twiddle (DS), +2h^2 doubles per r>=7 level (fold), well under
the 3 KiB/size guidance at acceptance sizes; create() <= 0.1 s at L=100.

### What I would do next (gen_r6)

1. **DS inside a codelet generator**: wire tw_cis_ds into gen_planner's
   fused-leaf constant emission (their twiddles are per-site constants; the
   select costs nothing and buys worst-case headroom). One-call diff pitch
   in their prompt if the monitor allows.
2. **Round-6 surprise insurance**: primes 37..127 now run the fold path
   (~3.5x fewer FMAs than r4's dense); if a surprise draw lands on a large
   prime, check the assembled library actually routes it to gen_rader/
   gen_dense_prime — my demo is the existence fallback, not the contender.
3. **xarch knobs**: race TWD_BLK_MIN_BYTES, TWD_MAPPAIR, TWD_DS on CLX/SPR
   when XARCH.md lands (SPR's second FMA pipe may flip the map-pack verdict;
   its port-5 pressure differs).
4. refnd pitch #6 if still double-cexp.

## Round gen_r6

### Adoption status (the score)

- **gen_bluestein's adoption stands and carried their r6 structural change
  for free**: their new non-power-of-two convolution grid (M = 48/80/96/160/
  192) consumes `tw_fill_ct_int_colmajor` + audit unchanged — their record:
  "Twiddle fills/audits unchanged — tw_fill_ct_int_colmajor takes arbitrary
  N (checked before writing code)". That is what a layer is for. API kept
  100% backward compatible again; `gcc -c impl/gen_bluestein.c` verified
  clean against this round's file after every change.
- **The library is FROZEN this round — zero API changes** (the gen_layout /
  gen_race doctrine, adopted deliberately: churn in a layer others #include
  is its own cost, and no r5/r6 adopter record asked for a new primitive or
  layout).
- **Convergent validation note**: gen_pow2's r5 GP2_FTW ships dual-select
  Linzer-Feig twiddle tables built by THEIR OWN generator, not tw_cis_ds —
  independent arithmetic, same citation (lit 11 Tier 1), same verdict as my
  r5 knob (correct, accuracy-neutral, wall-neutral on a non-port-bound
  engine). Two engines now agree on the claim; the layer's tw_cis_ds remains
  the drop-in for anyone who wants the tables without writing the octant
  tie-handling themselves.
- **refnd double-cexp gate reference, pitch #7**: gen_powp.c:1607 and
  gen_pfa_large.c:1359 still build the create()-gate reference W with double
  `cexp` (250 ulp ≈ 5.6e-14 per twiddle on a 1e-13 yardstick at L=100).
  `tw_fill_dft_cplx` remains the one-line fix.

### The demo: register-resident whole-level codelets for r = 2/3/4/5
### (−30% to −44% at every combine-bearing graded cell, bit-identical)

**The finding (asm audit, the round's whole story).** twd_butterfly was
NEVER inlined into twd_rec — `gcc -O3 -march=icelake-server -S` shows 6
surviving call sites — so every leaf and every combine k1-iteration paid a
real call PLUS a round trip of 2r zmm rows through the tr/ti staging arrays
(8 stores + 8 reloads at r = 4 before any arithmetic), and twd_rec carried a
16.6 KB probed stack frame (tr/ti = 2×128 v8 rows) through every recursion
step, page-probe loop included. I had assumed since r1 that the compiler
elides the staging; it does not, and five rounds of measurements sat on top
of that overhead. Check your asm for un-inlined hot dispatch: a big switch
in a called-in-a-loop function does NOT get inlined even at two call sites,
and "the compiler will forward those stores" is not a plan.

**The fix.** Whole-level noinline codelets: `twd_leaf2/3/4/5` (strided loads
straight into registers, butterfly, contiguous stores) and `twd_comb2/3/4/5`
(the ENTIRE k1 = 0..m−1 combine loop: load r rows, twiddle-multiply,
butterfly, store — all register-resident, zero rsp-zmm spills, asm-audited).
The generic paths (odd-prime fold leaves, combine radices 7/11/13, and all
radices under -DTWD_DS) moved unchanged into `twd_leaf_gen`/`twd_comb_gen`,
also noinline, so the staging arrays' frame is paid only there: twd_rec is
now a thin dispatcher with a 392-BYTE frame (was 16.6 KB + probe loop).
Every expression is twd_butterfly's cases and the generic combine's twiddle
multiply copied VERBATIM (same temporaries, same order → same FMA
contraction), so outputs are **BIT-IDENTICAL to gen_r5**: verified by cmp on
the node (AVX-512 codegen) at 12/27/31/50/100, singles and m=2 chains, and
on wallaby at 16 sizes including fold primes (31/101/127) and a fold-combine
composite (77 = 7·11), singles and m=3 chains. The r5 case-bloat lesson
applied preemptively: every body its own noinline function, nothing added to
the dispatcher — and the fold path read a clean wash as designed.

### Measured on the node (a80n0, ONE held lease, slot 1 core 3, same-core
### interleaved pairs vs the r5 control binary, order rotated per pair
### (gen_pow2 r5's wrinkle); graded chain cells, min µs/xform)

| L | B | r5 ctl (same window) | gen_r6 | delta | note |
|---|---|---|---|---|---|
| 10 | 64 | 10.28 / 10.28 | **6.29 / 6.44** | **−38%** | |
| 12 | 64 | 14.31–14.39 (3 prs) | **8.86–9.19** | **−37%** | B=1: **10.52** (r5 14.33) |
| 15 | 32 | 28.41 / 28.50 | **19.94 / 20.01** | **−30%** | |
| 20 | 32 | 55.5 / 56.4 | **35.59 / 35.64** | **−36%** | |
| 25 | 16 | 120.1 / 120.6 | **79.1 / 80.7** | **−34%** | r=5 combines |
| 27 | 16 | 216.3–219.9 (3 prs) | **130.2–133.4** | **−40%** | r=3 combines |
| 31 | 16 | 267.2–272.1 | 268.0–274.9 | **wash** | fold path untouched — the intended control |
| 32 | 8 | 340.7 / 343.6 | **198.0 / 199.6** | **−42%** | |
| 40 | 8 | 613.0 / 618.3 | **342.3 / 344.2** | **−44%** | |
| 50 | 4 | 1281.7 / 1288.1 | **741.3 / 752.6** | **−42%** | r=2 top combine |
| 100 | 1 | 10821 / 11460 | **7767 / 7783** | **−29%** | quiet read **7631.5**; MKL same window **7734.8** — the demo now BEATS MKL at 100 |

MKL 2022 same windows: 7.73 (12 B=64), 7734.8 (100). The gap that remains at
small L is gather/scatter + per-call structure, not the levels.

### Gates (ship binary, on the node; tryout's map-check leg still dies on
### the '$W/c.bin' quoting bug — run check.py by hand, r2 recipe)

Bit-identity to r5 makes every r5 gate value carry over exactly; re-measured
anyway: singles 2.959e-16 (12) / 4.817e-16 (100) via tryout PASS; two-step
m=2 9.521e-16 (12) / 2.948e-15 (100) vs tol 3e-14; graded chains 5.321e-14
(12, anchor 3.887e-14) / 3.694e-14 (100, anchor 2.416e-14), tol 1e-10; chain
outputs bit-identical across independent node runs. Local: numpy PASS at
12/27/31/100 (B=2) and scalar -march=x86-64 build PASS at 12/31/127; all
knob combinations (DS / DENSEBF / MAPPAIR / PF / DS+DENSEBF) compile -Wall
-Wextra clean; GEN_TWIDDLE_LIB_ONLY adoption (gen_bluestein) compiles clean.
setup unchanged (≤ 0.11 s at L=100).

### What did NOT work / incidents

- Nothing raced off this round — the change is a pure deletion of overhead
  with bit-identical arithmetic, and the L=31 wash was the predicted
  control, not a surprise. One build incident worth a line: a `*/` inside a
  comment ("twd_leaf*/twd_comb*") terminated the comment block mid-file;
  gcc's error pointed at the comment text. Write glob pairs as twd_leafN.

### Borrowed this round, named

- **gen_batchlane gen_r4**: the held-lease same-core interleaved A/B — every
  number above is adjacent-pair on one core.
- **gen_pow2 gen_r5**: the rotate-the-A/B-order wrinkle (first-position bias
  once windows drift); pair order alternated throughout.
- **My own gen_r5 case-bloat lesson**, applied in reverse: moving the
  generic bodies OUT shrank the dispatcher's frame 42× and is where most of
  the small-L win lives.

### Operation count (demo, delta vs gen_r5)

FMA/add/mul count and order: IDENTICAL (bit-identical outputs). Deleted per
combine k1-iteration at radix r: 2r zmm stores + 2r zmm reloads (staging) +
1 call/ret + 1 switch dispatch; per leaf: the same staging round trip;
per twd_rec invocation: a 16.6 KB → 392 B frame (page-probe loop gone).
Tables, plan memory, create() work: unchanged.

### What I would do next (gen_r7 / endgame)

1. **Fold-combine codelet for r = 7/11/13** (the one hot path still on the
   generic staging loop): only composite sizes with a >5 prime factor use it
   (49, 77, 91, 98, 121, ...) — exactly the round-6 surprise-draw territory.
   Twiddle into registers, then feed the fold directly.
2. **FTW-style fused-scale twiddles in the specialized combines**: gen_pow2
   measured the op cut wall-neutral under a staging-dominated engine; now
   that the combines are register-resident and FMA-lean, re-race tw_cis_ds
   with the scale folded into the butterfly (the r5 DS wash may become a
   win — the boundary is worth one A/B).
3. **xarch knobs on CLX/SPR** (TWD_BLK_MIN_BYTES / TWD_MAPPAIR / TWD_DS):
   XARCH.md r5 shows no gen_twiddle-relevant flips, but the new combines
   change the port mix — re-check when the next advisory lands.
4. refnd pitch #8 if gen_powp / gen_pfa_large still build gate references
   with double cexp.

## Round gen_r7

### Adoption status (the score)

- **gen_bluestein's adoption stands** (tw_chirp + colmajor filler/audit); API
  kept 100% backward compatible — the library half of the file is UNCHANGED
  this round (frozen for the second round running, the gen_layout/gen_race
  doctrine), and `gcc -c impl/gen_bluestein.c` verified clean against the
  shipped file.
- **refnd double-cexp gate reference, pitch #8**: gen_powp.c:1728 and
  gen_pfa_large.c:1392 still build the create()-gate reference W with double
  `cexp` (250 ulp ≈ 5.6e-14 per twiddle on a 1e-13 yardstick at L=100).
  `tw_fill_dft_cplx` remains the one-line fix.

### The round's centerpiece is a NEGATIVE result, fully measured: split-custody
### two-axes-per-pass direct feed (brief r7 item 1 / lit 11 Tier 2) LOSES 6-35%

I built the whole program my r4 record queued and the r7 brief names: SPLIT
intermediate volume (axis 0 scatters with plain v8 stores, zero shuffles),
axis 1 feeding the recursion DIRECTLY from the split plane rows (leaves
rewritten to unaligned loads, xstr in doubles — zero gather copy), axes 1+2
fused through ONE in-register 8x8 transpose per tile into a split z-major
block buffer P, and axis 2 direct-feeding from P (zero copy, zero shuffles,
guaranteed-full lanes because kblk·L ≡ 0 mod 8). On paper: −0.75 to −1.0
shuffles/complex, two gather copies deleted, "one volume stream deleted".
Outputs were verified BIT-IDENTICAL to gen_r6 at 29 sizes locally and 5 on
the node (singles and chains) — the arithmetic, group decomposition and lane
values were all preserved. And it lost EVERYWHERE, same-core interleaved
rotated pairs, 4/4 each (graded cells, min µs/xform, r7split vs r6ctl):

| L | B | r6 ctl | r7 split | delta |
|---|---|---|---|---|
| 10 | 64 | 6.38–6.54 | 7.01–7.23 | +8% |
| 12 | 64 | 9.10–9.22 | 9.73–9.82 | +6.5% |
| 12 | 1  | 9.04–9.40 | 9.63–9.86 | +5% |
| 15 | 32 | 19.6–20.1 | 21.7–22.6 | +11% |
| 20 | 32 | 35.7–36.9 | 38.4–39.1 | +7% |
| 25 | 16 | 79.6–81.7 | 94.0–95.4 | +16% |
| 27 | 16 | 130.3–131.8 | 147.6–150.3 | +13% |
| 31 | 16 | 272.7–306.7 | 312.4–330.6 | +14% |
| 32 | 8  | 199.1–212.7 | 222.4–244.9 | +11% |
| 40 | 8  | 343.0–350.6 | 363.2–379.8 | +6% |
| 50 | 4  | 757.0–765.7 | 954.0–965.8 | +26% |
| 100 | 1 | 7923.9–8611.0 | 10843.8–11513.1 | +35% |

**Why (the transferable mechanism, worth more than the diff I reverted):**

1. **The gathers were never overhead — they are the software prefetch AND the
   L1 staging.** The old gather walks the pencil rows SEQUENTIALLY once
   (streaming, prefetcher-friendly) and the recursion then hits an 8L-double
   L1 buffer. Direct feed replaces that with the recursion's own DIT-decimated
   access: leaf s reads rows s, s+m, s+2m, ... — widely strided 64 B touches
   across a plane (160 KB at L=100) or across P, which the hardware
   prefetchers do not follow and the OoO window cannot cover. The loss grows
   with L exactly as the staging buffer's parent (L1 → L2 → L3) recedes.
   This is the ice "no in-sweep gathers" lesson pointing the OTHER way: my
   "zero-copy" feed WAS an in-sweep gather.
2. **The chain's in-place interleaved custody was already DRAM-optimal.**
   I claimed split-T would delete a volume stream; wrong. In gen_r6's chain,
   axis 1 writes its planes back and axis 2 OVERWRITES the same addresses
   while the block is still cache-hot — the dirty lines never reach DRAM
   twice. Real r6 DRAM streams: 5 (axis0 R+W, block R+W, c R). The split-T
   variant also runs 5 but touches st + T + c = 48 MB per step at L=100
   instead of 32 MB — it blew L3 (24 MB) and turned the axes-1+2 pass's
   input reads into misses. The +35% at 100 and +26% at 50 are footprint;
   the +6-16% at resident sizes are mechanism 1.

Do not spend a round rediscovering this: if your engine stages pencils
through a gather, the gather is load-bearing; delete the SHUFFLES inside it
if you can, never the sequential pass itself. (The one place a shuffle-free
split handoff might still pay is behind a gather that stays sequential —
that variant saves only ~16 shuffles/64 complex and I did not burn a window
on it after the main result came in this decisively.)

### What SHIPPED: fused fold-combine codelet for combine radices 7/11/13
### (my r6 next-steps item 1) — −6 to −8% at every fold-composite size, bit-identical

The one hot path still on the generic staging loop: composite sizes with a
factor in {7, 11, 13} ran twd_comb_gen (per k1: twiddle-multiply into
tr/ti stack arrays — 2r zmm stores + 2r reloads — then a call into
twd_butterfly → twd_fold_bf). New `twd_comb_fold` runs the WHOLE k1 = 0..m−1
combine in one noinline call: the twiddle products feed the conjugate fold
directly (a_j/b_j built from w_j·x_j ± w_{r−j}·x_{r−j} as they are computed),
fold rows live in 12 zmm-backed locals (h ≤ 6 since twd_factor caps composite
radices at 13), and the E/O correlation + conjugate-signed combine are
twd_fold_bf's expressions VERBATIM — same temporaries, same order, same FMA
contraction, so outputs are bit-identical to gen_r6 (verified by cmp on the
node at 49/77/98/121/100 and locally at 22 sizes, singles and chains).
Scored acceptance sizes never execute the new path by construction.

Node, same-core interleaved rotated pairs, m=8 chains, min µs/xform, 4/4 each:

| L | B | r6 ctl | gen_r7 | delta | factorization |
|---|---|---|---|---|---|
| 49 | 8 | 1375–1417 | **1270–1282** | **−8%** | 7·7 |
| 77 | 2 | 5546–5779 | **5224–5327** | **−7%** | 7·11 |
| 91 | 2 | 9758–10009 | **9019–9206** | **−8%** | 7·13 |
| 98 | 1 | 12707–12939 | **12063–12140** | **−6%** | 2·7·7 |
| 121 | 1 | 24609–24895 | **23049–23367** | **−6.5%** | 11·11 |

These are exactly the round-6-surprise-draw shapes (the addendum's L=21/44
lesson: never-built primes and their composites are where the library is
weakest), so the win lands where the assembled trunk's fallbacks are thin.

**Scored cells: parity confirmed, with a measurement lesson.** First session
read L=50 r7-high 4/4 (+0.5–1.9%) — a layout-tax scare (my r5 case-bloat
precedent). Six more rotated pairs: signs flip (r7 wins 4 of 6), medians
764.8 vs 768.2 — a WASH; the 4/4 was window luck. 12/27/31/100 were washes
with mixed signs in session one (e.g. 100: 7832.8/7934.5/7939.8/7906.1 vs
7903.5/7874.9/7877.4/7895.1). Ten total pairs at the scare cell before
declaring parity is the protocol takeaway.

### Ship-binary reads via ./tryout.sh (a80n0, leased core, graded cells)

12 B=64: **8.927** µs/xform (sd 0.02%; MKL same window 7.908); 12 B=1:
**10.104** (MKL 8.329); 100 B=1: **8056.7** (noisy window, sd 0.87%, MKL
7967.8 same window); 77 B=2 single-transform: 5981 (MKL 2432 — the demo has
no 7/11-point leaf modules, it runs generic fold leaves; that gap is
gen_dense_prime/gen_rader's round-7 assignment, not mine).

### Gates (ship binary, on the node; tryout's map-check leg still dies on the
### '$W/c.bin' quoting bug — check.py run by hand, r2 recipe)

Bit-identity to gen_r6 makes every r6 gate value carry over exactly;
re-measured anyway: singles 2.959e-16 (12) / 4.817e-16 (100) / 4.519e-16
(77), all PASS tol 1e-12; graded chains 5.321e-14 (12 m=600, anchor
3.887e-14) / 3.694e-14 (100 m=64, anchor 2.416e-14) / 7.408e-15 (77 m=8,
anchor 6.259e-15), tol 1e-10; chain outputs bit-identical across independent
node runs at 100. Local: numpy PASS at 22 sizes incl. 21/35/44/49/77/91/96/
98/119/121/126/127; scalar -march=x86-64 build PASS (77 checked vs numpy);
all knob combinations (DS / DENSEBF / MAPPAIR / PF / DS+DENSEBF) compile
-Wall -Wextra clean (the DS and DENSEBF race arms keep the generic combine
loop by design); GEN_TWIDDLE_LIB_ONLY adoption compiles clean + selftest
PASS; setup unchanged (≤ 0.08 s at L=100).

### What did NOT work / incidents, with the number

- **The split-custody direct-feed program above** — the whole section is the
  negative; reverted wholesale after 12 cells × 4 pairs, all losses. The
  revert was `cat impl_6/gen_twiddle.c > impl/gen_twiddle.c` (content copy,
  inode preserved and checked — the r5 hard-link lesson) and the fold-combine
  was re-applied on top as a clean minimal diff (~120 changed lines).
- **L=50 4/4 window-luck scare** — resolved by 6 more rotated pairs, above.

### Borrowed this round, named

- **Literature 11 Tier 2 / brief r7 item 1** (two-axes-per-pass fusion): the
  attempted design; the negative verdict and the mechanism boundary are this
  engine's contribution back.
- **gen_batchlane gen_r4** (held-lease same-core interleaved pairs) and
  **gen_pow2 gen_r5** (rotate the A/B order): every keep/kill number above.
- **My own gen_r5/r6 lessons**: case-bloat (comb_fold is its own noinline
  function; the dispatcher gained only a 2-line default case) and the r6
  staging-deletion pattern (comb_fold is that pattern applied to the last
  staged hot path).

### Operation count (demo, delta vs gen_r6)

FMA/add/mul count and order: IDENTICAL (bit-identical outputs).  Deleted per
combine k1-iteration at radix r ∈ {7,11,13}: 2r zmm stores + 2r zmm reloads
(tr/ti staging) + 1 call/ret + 1 switch dispatch + the twd_fold_bf frame
round trip.  Tables, plan memory, create() work: unchanged.

### What I would do next (gen_r8 / endgame)

1. **The sequential-gather shuffle-free split handoff** (the untested benign
   variant of this round's negative): axis-1 scatter → split P via tr8x8,
   axis-2 gather from P as a plain sequential copy — saves ~16 shuffles/64
   complex with staging and streaming preserved. One window, keep only if
   ≥2% on 3+ cells.
2. **DS re-race with specialized DS combines** (r6 item 2): under -DTWD_DS
   the combines still fall back to the generic loop, so the knob now measures
   staging, not dual-select. Fold the scale into twd_comb2..5 variants if a
   generator entry asks; otherwise let gen_pow2's independent validation
   stand.
3. **xarch knobs on CLX/SPR** (TWD_BLK_MIN_BYTES / TWD_MAPPAIR / TWD_DS /
   TWD_PF) when the next advisory lands.
4. refnd pitch #9 if gen_powp / gen_pfa_large still build gate references
   with double cexp.

## Round gen_r8

### Adoption status (the score)

- **gen_bluestein's adoption stands** (tw_chirp + colmajor filler/audit); the
  LIBRARY half of the file is UNCHANGED for the third round running (the
  frozen-layer doctrine) — `gcc -c impl/gen_bluestein.c` verified clean
  against the shipped file after every change this round.
- **refnd double-cexp gate reference, pitch #9**: gen_powp.c:1738 and
  gen_pfa_large.c:1474 still build the create()-gate reference W with double
  `cexp` (250 ulp ≈ 5.6e-14 per twiddle on a 1e-13 yardstick at L=100).
  `tw_fill_dft_cplx` remains the one-line fix.
- **Convergence note**: gen_planner's r8 A/B script races a PLN_LIFT5 knob —
  the gen_batchlane r7 lifted-DFT5 idea is spreading through the panel on its
  own; the layer's exact-constant primitives (tw_cisl for sin(2pi/5),
  sin(pi/5); phi and 1.25/sin(pi/5) are one long-double divide away) are
  available to anyone who wants lift constants at <= 0.51 ulp.

### What SHIPPED: split-group handoff for axes 1->2, plan-gated to large L
### (-2..-3.5% at 27, -2.6..-3.2% at 50, -4.6..-5.2% at 100, bit-identical)

My r7 next-steps item 1 — the benign variant of the r7 split-custody
negative. The axis-1 SEQUENTIAL gather stays untouched (the r7 lesson: it is
the software prefetch and the L1 staging; deleting it lost 6-35%). What
changes is only the handoff between axis 1 and axis 2:

- axis-1's writeback no longer interleaves back into the plane
  (twd_scatter_i, 2 shuffles/row): a new `twd_scatter_gf` transposes 8y x 8z
  tiles in registers (two tr8x8) straight into a slab-sized GROUP-FORMAT
  buffer GB — one L-row block per axis-2 8-row group, row j = [re v8][im v8]
  (16 doubles);
- axis-2's gather is DELETED ENTIRELY: twd_rec direct-feeds group G with
  (xr, xi, xstr) = (GB + G*L*16, ..+8, 2) — the existing recursion takes the
  interleaved-row stride with zero code changes, and all loads stay
  64B-aligned.

Per 64 complex the handoff drops from scatter_i(16 shuf + 16 st) +
gather_z(48 shuf + 16 ld + 16 st) to 2x tr8x8(48 shuf) + 16 st: net -16
shuffles, -16 stores, -16 loads, plus gather_z's whole sweep/call overhead.
GB is ceil(kblk*L/8) full group buffers (<= 2 MB at any L), reused per slab:
L2-hot custody, NO new DRAM stream — the r7 negative's footprint mechanism
avoided by construction. Outputs are BIT-IDENTICAL to gen_r7 (verified by
cmp at 16 sizes x {singles, m=3 chains} locally incl. 25/27/31/77/96/101/127,
plus a forced-gate build at 10 small sizes; arithmetic order untouched).

**The plan gate.** The tr8x8 scatter is quantized to 8x8 tiles, so at small
L it costs more shuffles than the gather it deletes (at L=12 roughly 4x the
old scatter's shuffle count, and plane seams / w<8 z-tails don't tile).
Measured ungated: +4% at 12, +1.5% at 25 vs wins at 27/50/100. Ship gates at
`32*L^3 >= TWD_GF_MIN_BYTES` (default 600000 — between 25's 500 KB and 27's
630 KB of fused-pass streams, both measured straddling it); knob raceable
per host like TWD_BLK_MIN_BYTES, and -DTWD_NOGF forces the r7 handoff
everywhere for the cross-arch race. Gate-off sizes run the r7 loops VERBATIM
and INLINE in twd_exec_vol/twd_chain_step (see the layout saga below).

### Measured on the node (a80n0, ONE core per session via slot lease,
### same-core interleaved rotated pairs vs the rebuilt r7 binary,
### gen_batchlane r4 protocol + gen_pow2 r5 rotation; graded chains, min)

Final ship binary (A/B rounds 4-6, pairs pooled):

| L | B | r7 ctl (same window) | gen_r8 ship | delta | note |
|---|---|---|---|---|---|
| 12 | 64 | 8.92-9.39 (9 prs) | 8.99-9.53 | wash (mixed signs) | gate off |
| 25 | 16 | 78.6-81.1 (10 prs) | 80.4-84.2 | **+1..2% residual** | gate off — layout tax, see below |
| 27 | 16 | 129.8-133.9 | **126.3-129.4** | **-2..-3.5%** (8/8) | gated ON (630 KB) |
| 40 | 8 | 338.7-345.1 | 336.3-343.2 | wash (3/4 lean win) | blocked, no seams |
| 50 | 4 | 744.0-756.2 | **725.3-730.6** | **-2.6..-3.2%** (7/7) | |
| 100 | 1 | 7768-7843 | **7354-7451** | **-4.6..-5.2%** (7/7) | the weakest-cell win |

31 raced wash in rounds 1-2 (fold-leaf pencils; handoff is a small fraction
there). Ship-binary tryout reads (different cores/windows, for the record):
12 B=64 **9.342** (sd 0.01%), 12 B=1 **9.473**, 50 **748.6**, 100 **7745.7**
(setup 0.095 s).

### What did NOT work, with the number that killed it

1. **GB sized kblk*L*L complex exactly**: a slab whose row count is not a
   multiple of 8 ends in a partial group that still owns a FULL L-row
   buffer; the scatter/feed overran GB into the twiddle-table takes at odd
   unblocked L — NaNs at 25/27 (o8 vs o7 cmp caught it immediately; L=15's
   210-double overrun happened to land in arena slack and passed, which is
   why the bit-identity sweep must cover MANY sizes). Fix:
   ceil(kblk*L/8) full group buffers.
2. **Staging-copy variant (axis-2 copies GB group -> Gr/Gi sequentially,
   then feeds xstr=1)**: loses to direct feed everywhere it matters — 100:
   8016-8613 vs direct 7532-7759 (+7%); 50: 764.8-770.8 vs 726.1-731.8;
   even at 12 it beat direct only by noise. The direct feed's decimated
   reads over a 12.8 KB L2-resident group buffer cost nothing measurable —
   the r7 in-sweep-gather mechanism needs a BIG target to hurt. Removed.
3. **Masked-store tr8x8 seam tiles** (vectorizing the <= 7-row plane seams
   with 2 tr8x8 + _mm512_mask_storeu_pd): LOST to plain scalar seam stores —
   25: 82.8-84.3 vs 80.2-81.0; 12: 9.63-9.87 vs 9.38-9.72. 48 port-5 uops
   for <= 7 rows of data is the gen_r5 map-pack lesson verbatim. Scalar
   seams shipped.
4. **The code-layout saga (the round's method lesson).** Three arrangements
   of the same bit-identical code, all measured at the gate-OFF cells:
   (a) both handoff arms inlined in the volume functions: twd_exec_vol grew
   7008 -> 12328 B (nm -S), 25 read +2.9%; (b) BOTH arms as noinline pass
   functions (volume functions back under r7 size): 25 STILL +2.9% (5/5) —
   being small is not enough, the r7-path instructions must also sit where
   r7 put them; (c) shipped: gate-off loops INLINE and VERBATIM, only the
   gf arm in noinline functions, plus `aligned(64)` on all 16 hot noinline
   functions (gen_dense_prime r7's code-alignment axis, applied as an
   attribute since harness flags are fixed): 25's tax shrank to +1..2%
   (5/5 lean), 12 to a wash. The residual at 25 is displacement of
   everything behind the ~10 KB of new .text; I spent 15 pairs across three
   builds on it and am shipping with it documented — the 27/50/100 wins
   dwarf it and the cell belongs to gen_powp at 2.6x my speed anyway.
5. **Declined borrow, with the recount: gen_batchlane r7's lifted DFT5
   v-pair.** Their claim is 8 -> 6 vector ops per DFT5; under FMA
   contraction MY forms count equal: original b-pair = (mul+fma) x 4 = 4
   ops/component; lifted u = fnmadd(1) + v2 = mul(1) + v1 = KS1*u + KL5*t4
   = mul+fma(2) — also 4/component. Their measured -0.7..-1.0% is real but
   their record attributes it to register pressure ("two fewer live temps
   keep the sweep pencil spill-free"); my combines are asm-audited
   spill-free (r6), so the mechanism does not transfer. Declined on
   arithmetic, like their own dense-GEMM declination — if a rival engine
   with DFT5 spill pressure wants it, the recount above is the boundary.

### Borrowed this round, named

- **My own gen_r7 negative**: the entire design is its closing note
  executed — keep the sequential gather, collapse only the shuffle pair;
  the footprint mechanism is why GB is slab-sized and L2-resident.
- **gen_bluestein gen_r4** (via my r4): the kblk = 8/gcd(L,8) custody that
  makes slab rows ≡ 0 (mod 8) — it is what lets GB's group decomposition
  match the unblocked pass and keep outputs bit-identical.
- **gen_planner gen_r7**: the `nm -S` symbol-size diff as the 5-second
  layout-drift detector — it found the 7 KB -> 17 KB bloat before a single
  node window was spent on the wrong hypothesis.
- **gen_dense_prime gen_r7**: code alignment as a first-class race axis —
  applied as `aligned(64)` function attributes.
- **gen_batchlane gen_r4 / gen_pow2 gen_r5**: held-lease same-core
  interleaved rotated pairs — all 6 A/B sessions this round.

### Operation count (demo, delta vs gen_r7)

Butterfly/twiddle arithmetic: IDENTICAL (bit-identical outputs everywhere).
Gated sizes (32*L^3 >= 600 KB), per 64 complex of axes-1->2 handoff:
-16 shuffles, -16 stores, -16 loads, -1 gather_z call; +scalar seam stores
at odd-L plane boundaries (<= 2 tiles per plane per z-group) and the w < 8
z-tail rows. Plan memory: +ceil(kblk*L/8)*8*L complex for GB (176 KB at
L=100, 2.0 MB worst case at L=127; nothing at gate-off sizes). Ungated
sizes: the r7 loops verbatim plus one predictable branch per slab.

### Gates (ship binary, on the node; tryout's map-check leg still dies on
### the '$W/c.bin' quoting bug — check.py run by hand, r2 recipe)

Bit-identity to gen_r7 carries every r7 gate value; re-measured anyway:
singles 2.959e-16 (12) / 3.884e-16 (27) / 4.520e-16 (50) / 4.817e-16 (100),
tol 1e-12; two-step m=2 9.521e-16 / 1.594e-15 / 2.313e-15 / 2.948e-15, tol
3e-14; graded chains 5.321e-14 (12 m=600, anchor 3.887e-14) / 3.075e-14
(27 m=200, 2.567e-14) / 4.468e-14 (50 m=128, 2.922e-14) / 3.694e-14
(100 m=64, 2.416e-14), tol 1e-10; B=1 chain at 12 PASS 6.972e-14 (anchor
5.797e-14); all chains bit-repeatable (cmp). Local: 16-size bit-identity
sweep (singles + chains) vs the r7 binary; forced-gate (-DTWD_GF_MIN_BYTES=1)
bit-identity at 10 small sizes; numpy PASS at 21/44/49/91/96/98/121/125/127/
128 (worst 8.7e-16 at 127); scalar -march=x86-64 build PASS vs numpy at
27/31/77; all knob combinations (NOGF / DS / DENSEBF / MAPPAIR / PF /
DS+DENSEBF / GF_MIN_BYTES=1) compile -Wall -Wextra clean;
GEN_TWIDDLE_LIB_ONLY adoption compiles clean. setup <= 0.1 s at L=100.

### What I would do next (post-campaign)

1. **Cross-arch**: race TWD_GF_MIN_BYTES and TWD_NOGF on CLX/SPR with the
   other knobs — CLX's 1 MB L2 moves both custody gates, and its downclock
   changes the shuffle economics the gate encodes.
2. The 25-cell layout residual: the only lever left is whole-binary
   function placement (link order); not worth it for a library layer's
   demo. If anyone chases it, start from the nm -S diff, not benchmarks.
3. refnd pitch #10 (gen_powp.c:1738, gen_pfa_large.c:1474).
4. Do NOT attempt the axis-0->axis-1 split handoff analog: in the chain it
   forces a T write (in-place st would clobber unread sites), which is a
   new full-volume stream — exactly the r7 mechanism-2 killer at 50/100.
   Analyzed this round, not built; recorded so nobody rediscovers it.

## Round gen_r9

### Adoption status (the score)

- **gen_bluestein's adoption stands** (tw_chirp + colmajor filler/audit); the
  LIBRARY half of the file is UNCHANGED for the fourth round running (the
  frozen-layer doctrine) — `gcc -c impl/gen_bluestein.c` verified clean
  against the shipped file.
- **refnd double-cexp gate reference, pitch #10**: gen_powp and gen_pfa_large
  still build the create()-gate reference W with double `cexp` (250 ulp on a
  1e-13 yardstick at L=100). `tw_fill_dft_cplx` remains the one-line fix.
- **Brief avenue 1 (bank the picks) does not apply to this engine**: create()
  contains no plan-time race and no nondeterministic pick — factorization,
  gates and knobs are all deterministic functions of (L, host), and chain
  outputs are bit-repeatable across independent node runs (re-verified this
  round). Nothing to bank; determinism is by construction.

### What SHIPPED: the 8-lane tail cleanup — packed axis-1 across plane seams
### + masked/vector w<8 and j-tails (−6..−20% at seven graded cells, bit-identical)

The engine's 8-pencil structure bled at every L not divisible by 8, in three
places, all fixed this round with lane-composition-only changes (per-pencil
arithmetic is elementwise, so ALL outputs are bit-identical to gen_r8 —
verified by cmp at 25 sizes locally, singles + m=3 chains, scalar and AVX-512
builds, forced-gate and NOGF arms, AND with node codegen at 12/15/25/27/100
singles + graded chains):

1. **Packed axis-1 (the big one).** Axis 2 has always packed its pencil rows
   ACROSS planes (that is what the kblk = 8/gcd(L,8) custody is FOR); axis 1
   never did — it swept per plane, paying ceil(L/8) groups per plane
   including a w = L mod 8 group that ran a FULL recursion on mostly-zero
   lanes through SCALAR gather/scatter loops (47% of pencils at L=15, 33% at
   12, 23% at 31). Now axis-1 pencils are numbered g = (x−x0)·L + z across
   the slab and groups of 8 span plane seams: rec calls drop from
   L·ceil(L/8) to ceil(L²/8) per volume (−35% at 10, −25% at 12, −21% at 25,
   −17% at 20, −15% at 27) and every group runs full 8-lane vector paths.
   Split groups (≤ 2 planes since L ≥ 8; provably never dangle past a slab)
   use new two-pointer masked forms `twd_gather_i2`/`twd_scatter_i2`: per row
   the 16 wanted doubles are [2a from plane A | 16−2a from plane B], built by
   two plain loads based at B−2a (always-mapped: the tail of plane A's same
   row) merged with two masked loads from A, then the ordinary deinterleave
   shuffles — zero scalar lane work. The ≤ 1 partial group per slab
   (pencil count mod 8, only in unblocked custody) stays scalar
   (`twd_*_i2t`). In-place chain safety: every pencil is gathered exactly
   once and written exactly once; split groups write only their own pencils
   into the next plane. L < 8 keeps the per-plane loop (unscored).
2. **Masked w<8 tails in `twd_gather_i`/`twd_scatter_i`** (axis-0's LL mod 8
   group; the gf-arm's per-plane axis-1 z-tails at 27/50/100): two
   zero-masked loads (maskz lanes are +0.0 = exactly the old zero fill) or
   two masked stores replace up to 4w scalar ops per row.
3. **Vectorized axis-2 tails.** The old `gather_z`/`scatter_z`/
   `scatter_z_map` fast path handled only w == 8 AND j+4 ≤ n: the last
   L mod 4 complex of EVERY z-pencil (20% of all map sites at L=15, 11% at
   27) and ALL of any w<8 group ran scalar loops including scalar sqrt/div
   per map site. Now: w<8 groups run the same tr8x8 tile path with zero rows
   for q ≥ w and per-q stores; j-tails load/store masked through one
   transpose. Map arithmetic on vectorized ex-scalar sites uses NEW
   `twd_map8x` — an EXACT vector replica of scalar twd_map1 (mul, fma, sqrt,
   add, div-for-sc, muls, u pair-duplicated from the re lane; matched gcc's
   -O3 contraction on the first try) — so even those sites stay
   bit-identical.

### Measured on the node (NOTE: a81n2 this round — the a80n0 hold expired and
### the queued icehold landed on the sibling Ice Lake node at 23:13; same CPU
### model, absolute times comparable but not identical-silicon to the r8 board.
### ONE held lease, same-core interleaved ROTATED pairs vs the rebuilt r8
### control, gen_batchlane r4 protocol + gen_pow2 r5 rotation; graded chains,
### min us/xform)

| L | B | r8 ctl (same window) | gen_r9 | delta | mechanism |
|---|---|---|---|---|---|
| 10 | 64 | 6.455/6.585 | **5.192/5.571** | **−16..−20%** | packed ax1 (20 → 13 groups) + ax2 w=4 tail |
| 12 | 64 | 9.138/9.366 | **7.602/7.860** | **−15..−18%** | packed ax1 (24 → 18) |
| 15 | 32 | 19.887/20.625 | **17.818/18.586** | **−9..−11%** | w=7 scalar tails deleted + jt=3 ax2 tails |
| 20 | 32 | 36.566/37.458 | **32.785/33.444** | **−10%** | packed ax1 (60 → 50) |
| 25 | 16 | 80.844/81.335 | **74.200/76.936** | **−6..−9%** | packed ax1 (100 → 79, w=1 groups gone) |
| 27 | 16 | 128.811/129.175 | **121.439/121.510** | **−5.7%** (4/4) | masked gf ax1 tails + ax2 jt=3 gather tails |
| 31 | 16 | 270.446/273.033 | **263.706/267.142** | **−2.3%** (4/4) | same, w=7 |
| 32 | 8 | 198.8–202.9 (6 prs) | 198.3–208.7 | wash (signs flip; first-pair +5% was window luck, r7's L=50 lesson re-learned) | no new path executes |
| 40 | 8 | 341.835/345.365 | 339.577/340.240 | −1% lean | no new path (L≡0 mod 8) |
| 50 | 4 | 743.865/745.380 | **728.066/730.945** | **−2.1%** (4/4) | masked gf tails + ax2 jt=2 |
| 100 | 1 | 7430.4/7447.2 | 7411.4/7421.6 | −0.3% lean (traffic-bound, as the PMU audit says) | masked gf tails |

Ship-binary tryout reads (different cores/windows): 12 B=64 **8.109**
(sd 0.01%), 12 B=1 **7.991** (r8 reads: 9.342/9.473 — B=1 −15.6%), 15 B=32
**18.043**. Wallaby (SPR) pre-reads agreed on every verdict, keep/kill was
decided on the ICL pairs.

### What did NOT work, with the number that killed it

- **Vectorizing the scatter_z_map j-tail with a full-width exact map**
  (tr8x8 + twd_map8x per pencil on jt ≤ 3 useful complex): wallaby +8% at
  L=25 (jt=1) and +2% at 27 (jt=3) vs the packed-ax1-only build, 4/4 each.
  Mechanism: **the divider unit prices vsqrtpd/vdivpd per OP, not per useful
  lane** — 8 zmm sqrt+div for 8 scalar sites loses to 8 scalar sqrt/div
  (~3x). Shipped fix: the map's j-tail stays SCALAR; the vector map runs
  only where lanes are full (w<8 full tiles). Boundary for the panel: pack
  divider work densely or leave it scalar; shuffles/loads vectorize at any
  occupancy, sqrt/div does not.
- The nm -S layout check (gen_planner r7's detector) shows twd_exec_vol /
  twd_chain_step grew ~+4.5/+5.2 KB (the packed sweep + i2 forms inlined).
  Watched for the r8-style gate-off layout tax at 32/40 (which execute none
  of the new code): 32 washed over 8 pairs, 40 leaned −1%. No tax this time;
  if a future round sees one, `noinline` on twd_gather_i2/twd_scatter_i2 is
  the first lever.

### Borrowed this round, named

- **My own axis-2 custody design (via gen_bluestein r4's kblk)**: the packed
  axis-1 is literally "do to axis 1 what axis 2 already does" — slabs hold
  kblk·L ≡ 0 (mod 8) pencils, so blocked custody never even needs the
  scalar tail.
- **gen_batchlane r4 / gen_pow2 r5**: held-lease same-core interleaved
  rotated pairs — every keep/kill above.
- **gen_planner r7**: nm -S as the layout-drift detector, run BEFORE spending
  node windows.
- **gen_rader r4 (port-relocation reasoning)**: used to diagnose the map
  j-tail loss as divider pricing without burning extra windows on variants.

### Operation count (demo, delta vs gen_r8)

Butterfly/twiddle arithmetic: IDENTICAL (bit-identical outputs everywhere).
Per volume at unblocked sizes: axis-1 rec invocations drop
L·ceil(L/8) → ceil(L²/8); ALL per-plane scalar tail loops (4w ops/row,
gather+scatter) deleted; split groups cost +2 loads/row (gather) and
+2 masked stores/row (scatter) vs aligned groups. Axis-2: scalar tail sites
(8·(L mod 4) per group + all w<8 groups) become tr8x8-tile vector work except
the map's j-tail sites (scalar by measurement, above). Axis-0 and gf-arm
axis-1 z-tails: 4w scalar ops/row → 2 masked ops + 2 shuffles/row. Plan
memory, tables, create() work: unchanged (setup ≤ 0.1 s at L=100 stands).

### Gates (ship binary, on the node; tryout's map-check leg still dies on the
### '$W/c.bin' quoting bug — check.py run by hand, r2 recipe)

Bit-identity to gen_r8 carries every r8 gate value (incl. two-step m=2);
re-measured anyway on a81n2: singles 2.959e-16 (12) / 3.375e-16 (15) /
3.698e-16 (25) / 3.884e-16 (27) / 4.817e-16 (100), tol 1e-12; graded chains
5.321e-14 (12 m=600, anchor 3.887e-14) / 5.802e-14 (15 m=600, 4.784e-14) /
3.809e-14 (25 m=256, 2.796e-14) / 3.082e-14 (27 m=200, 2.567e-14) /
3.694e-14 (100 m=64, 2.416e-14), tol 1e-10; all chains bit-repeatable.
Local: 25-size bit-identity sweep vs the r8 binary (native AND
-march=x86-64 scalar builds, singles + m=3 chains, incl. 9/21/44/49/77/96/
101/121/127/128); NOGF-arm bit-identity at 27/31/40/50/100 (multi-slab
packed custody); forced-gate bit-identity at 9/10/12/15/20/25; all knob
combinations (NOGF / DS / DENSEBF / MAPPAIR / PF / DS+DENSEBF /
GF_MIN_BYTES=1) compile -Wall -Wextra clean; GEN_TWIDDLE_LIB_ONLY adoption
compiles clean.

### What I would do next (gen_r10)

1. **Packed axis-1 for the gf arm** — analyzed, not built: split groups need
   a two-rowbase scatter_gf (per-lane GB phases differ by L mod 8; the
   aligned-tile path needs a per-lane alignr + 2 masked stores when the
   lane's phase ≠ 0). Group-count math: 27: 108→92, 50: 28→25, 100: 26→25
   per slab-set, but at gcd(L,8)=1 seven of eight groups pay the split-store
   overhead — net estimate only +1-2% at 27, less elsewhere. Marginal;
   build only if a window is spare.
2. **PMU signature reads** (brief avenue 3): /tmp/perf was staged on a80n0;
   the hold now lands on a81n2 — ask the monitor to re-stage before
   counter-directed work.
3. The L=100 cell remains traffic-bound (0.82/2.0 p0+p5, PMU audit) and this
   engine's streams are already minimal (r7/r8 analysis); the remaining
   lever there belongs to two-axes fusion inside a class engine, not to this
   demo's handoff.
4. refnd pitch #11 if still double-cexp.

## Round gen_r10

### Adoption status (the score)

- **gen_bluestein's adoption stands** (tw_chirp + colmajor filler/audit); the
  LIBRARY half of the file is UNCHANGED for the fifth round running (the
  frozen-layer doctrine) — `gcc -c impl/gen_bluestein.c` verified clean
  against the shipped file.
- **gen_pow2's adoption noted for the record**: their r9 strategy record
  credits "gen_twiddle gen_r5: their tanl-based dual-select constant
  generator" — tw_cis_ds is the constant generator behind their GP2_FTW
  dual-select tables. Second confirmed adopter alongside gen_bluestein.
- **Library doc note owed to gen_planner** (their r8 analysis of shear
  twiddles, generalized): dual-select forms whose SELECT changes the opcode
  pattern are only branch-free for codelets with plan-time-constant twiddles;
  a runtime-table consumer pays a per-twiddle branch (my r5 demo measured
  that branch a wash on this engine, but the boundary belongs in the docs —
  see tw_cisl_ds's comment, which already states the per-site-constant case
  is the intended adopter).
- **refnd double-cexp gate reference, pitch #11**: gen_powp and gen_pfa_large
  still build the create()-gate reference W with double `cexp`.
  `tw_fill_dft_cplx` remains the one-line fix.

### What SHIPPED (two changes)

**1. Radix-8 levels: twd_factor prefers r = 8 while 8 | n; new twd_leaf8 +
whole-level register-resident twd_comb8 codelets** (the gen_r6 codelet
treatment at radix 8; DFT8 = even/odd DFT4s + a W8 combine, the only
non-trivial constant c8 = sqrt(1/2) from tw_cis(1,8), correctly rounded).
One whole combine pass over the lane buffers is deleted wherever 8 | L:
32 = 8·4 and 40 = 8·5 drop from 3 recursion levels to 2 (per pencil-group at
32: one full 32-row zmm read+write round trip through L1 gone, twiddle
multiplies 33 → 21, 12 fewer twd_rec dispatches); 64 = 8·8, 96 = 8·12,
128 = 8·16·(2), 24/56/72/88/104/120 similarly. Scored cells with 8 ∤ L are
untouched by construction (verified bit-identical, below). twd_butterfly
gains a case 8 so the -DTWD_DS / -DTWD_DENSEBF race arms (generic combine)
stay correct; create() skips fold/dense tables for r = 8 (exact-constant
codelet, no tables beyond the standard CT twiddles).

**2. Extract-to-memory transpose stores** (ADOPTED from gen_pow2 gen_r9, the
round's #1 steal): the LAST stage of tw_tr8x8 only moves aligned 256-bit
halves (B[k] = [U_k lo | U_{k+4} lo], B[k+4] = the highs), and at the
transpose-into-store sites its only consumer IS the store — so the stage
becomes 16 x 256-bit stores (vmovupd ymm lows + vextractf64x4-TO-MEMORY
highs, pure store-port ops, p237+p4, no p5 uop): −8 port-5 shuffles per
transpose, front-end and store-bandwidth neutral, bytes at every address
IDENTICAL. Applied at twd_scatter_gf full tiles (−16 p5 per 8x8 tile; the
gf-arm chain cells 27/31/40/50/100) and twd_scatter_z full groups (execute
path). NOT applied at twd_gather_z by default (see negatives).

### Measured on the node (a81n2, ONE held lease slot 1 core 3, same-core
### interleaved ROTATED pairs vs the r9 control built from impl_9 — NOT the
### live impl symlink (gen_rader r9's trap, heeded); 2 warmups per cell
### (gen_race r9's rule); graded chains, min µs/xform, --samples 5)

| L | B | r9 ctl (same window) | gen_r10 | delta | mechanism |
|---|---|---|---|---|---|
| 12 | 64 | 7.545–7.959 (6 prs) | 7.593–8.246 | wash-to-lean +0.5–1% (mixed signs after the relocation fix; was 4/4 +1.1–3.0% before it) | no new code executes; residual layout tax, r8-precedent |
| 25 | 16 | 74.12–76.91 | 74.76–76.38 | wash (mixed signs) | no new code executes |
| 27 | 16 | 121.02–122.80 | 121.91–125.59 | wash | scatter_gf extract (small fraction of pass) |
| 32 | 8 | 195.4–198.0 (7 prs) | **153.4–155.8** | **−21.4%** (7/7) | radix-8: 3 → 2 levels |
| 40 | 8 | 336.7–342.2 | **286.0–291.9** | **−14.5%** (6/6) | radix-8: 3 → 2 levels |
| 50 | 4 | 714.5–721.6 | **708.5–714.2** | **−0.4..−1.5%** (4/4) | scatter_gf extract-to-memory |
| 100 | 1 | 7412–8000 (noisy window) | 7301–7766 | lean win, mixed | scatter_gf extract |

An early L=32 session had a neighbor's job inflating both arms ~2x
(readings 398/302 dropping to 197/155 mid-session when it finished); the
contaminated pairs showed the same ratio but only the quiet pairs were
counted (the r4 protocol).

Ship-binary tryout reads (fresh cores/windows, sd ≤ 0.27%):

| case | gen_r10 | MKL same window | note |
|---|---|---|---|
| 32 B=8 m=250 | **158.256** | 177.279 | demo now BEATS MKL at 32 (r9: 196.5 vs 172.1, behind) |
| 32 B=1 m=250 | **157.161** | 158.756 | beats MKL at B=1 too |
| 40 B=8 m=128 | **290.222** | 412.544 | r9 board 336.7 |
| 12 B=64 m=600 | 7.992 | 7.912 | window-dependent ±0.5% vs MKL; A/B verdict above |
| 12 B=1 m=600 | 7.702 | 7.317 | |

Wallaby (SPR) advisory agreed on every keep/kill and the radix-8
magnitudes (−13%/−9% at 32/40) before any node window was spent.

### Gates (ship binary; tryout's map-check leg still dies on the
### '$W/c.bin' quoting bug — check.py run by hand on the node, r2 recipe)

Radix-8 cells re-gated in full: singles 3.286e-16 (32) / 3.758e-16 (40),
B=1 3.296e-16 / 3.753e-16, tol 1e-12; two-step m=2 1.540e-15 (32) /
1.918e-15 (40) vs tol 3e-14; FULL graded chains 3.277e-14 (32 m=250,
anchor 0) / 3.924e-14 (40 m=128, anchor 2.612e-14), tol 1e-10; chains
bit-repeatable across runs (cmp). Everything with 8 ∤ L carries the r9
gate values exactly: NODE-codegen bit-identity vs the r9 binary verified
by cmp at 12/25/27/31/50/100, singles AND m=8 chains, plus wallaby
bit-identity at 18 sizes (singles + m=3 chains) and numpy PASS at all of
8/12/16/24/27/31/32/40/48/56/64/72/88/96/100/104/120/128 (worst 4.8e-16;
radix-8 composites 24..128 all exercise twd_comb8/twd_leaf8). Local: DS
and DENSEBF race arms PASS at 32/40/96; scalar -march=x86-64 build PASS
at 32/40/96; all knob combinations (DS / DENSEBF / DS+DENSEBF / MAPPAIR /
PF / NOGF / XSTG / GF_MIN_BYTES=1) compile -Wall -Wextra clean;
GEN_TWIDDLE_LIB_ONLY adoption compiles clean. create() determinism: picks
are closed-form functions of (L, host) as before — no plan-time race, so
brief avenue 1 stays discharged by construction (bit-repeatable chains
are the receipt).

### What did NOT work, with the number that killed it

- **Extract-to-memory at the GATHER site (twd_gather_z lane-buffer fill)**:
  +0.7–1% at L=12 on wallaby (r10-with-gather-extract 4.480–4.570 vs
  r10-without 4.436–4.545 vs r9ctl 4.384–4.484, interleaved). Mechanism:
  these stores are re-read by twd_rec almost immediately, and a zmm load
  spanning two ymm stores CANNOT store-forward — the SF-fail stall beats
  the −8 p5. gen_pow2's win had a full plane between store and consumer.
  Boundary for the panel: extract-to-memory pays only where the consumer
  is far away; at a staging buffer the wide store IS the forwarding path.
  Kept compilable as -DTWD_XSTG (opt-in) for the cross-arch race.
- **First build placed the new codelets mid-file and let the case-8
  butterfly inline into twd_leaf_gen**: L=12 read +1.1–3.0%, 4/4 node
  pairs, on bit-identical output — the r5/r8 code-layout tax a third
  time. nm -S fingered it: twd_leaf_gen +657 B (case-8 inlined into its
  twd_butterfly copy) sitting between the hot functions. Fix, two parts:
  (a) twd_butterfly's case 8 is compiled ONLY under -DTWD_DS (default
  builds route r = 8 to twd_leaf8/twd_comb8 in twd_rec's switches, so the
  generic path can never see it) — leaf_gen back to its r9 size to the
  byte; (b) twd_leaf8/twd_comb8 DEFINITIONS moved to the end of the file
  (declarations up top), so emission order = the r9 hot text unchanged,
  new code appended. After the fix L=12 reads mixed signs, median +0.5–1%
  (r8-precedent residual, documented and accepted — the 32/40 wins dwarf
  it and gen_race/gen_pfa_small own that cell at 1.9 µs anyway).
- **Two session-hygiene incidents worth a line**: an early L=32 window was
  contaminated by a neighbor job (readings 2x, both arms — discarded per
  the r4 protocol, quiet pairs only); and a missing in31.bin in the node
  workdir made cmp report a phantom DIFF at L=31 (cmp of two nonexistent
  files "differs") — generate inputs before trusting any cmp verdict.

### Borrowed this round, named

- **gen_pow2 gen_r9**: extract-to-memory for store-consumed shuffles — the
  whole change-2 design (their −2.3% at 32; here −0.4..−1.5% at 50 and a
  share of the 32/40 wins via scatter_gf). The gather-site negative above
  is this engine's boundary contributed back.
- **gen_rader gen_r9**: the impl-symlink trap — the r9 control was built
  from impl_9/gen_twiddle.c (a true archive, verified different inode and
  content from the live impl/), never the snapshot dir.
- **gen_race gen_r9**: the 2-warmup rule for interleaved batteries.
- **gen_batchlane gen_r4 / gen_pow2 gen_r5**: held-lease same-core
  interleaved rotated pairs — every keep/kill above.
- **gen_planner gen_r7**: nm -S as the layout-drift detector — it found
  the leaf_gen +657 B before a second node window was spent.
- **Declined after reading their records**: the φ-lifted DFT5 (my r8
  recount stands — my combines are spill-free so the mechanism does not
  transfer); port-1 ymm co-issue (closed twice: gen_pfa_large's portcal3
  measures 256-bit FP stealing 512-bit FMA slots 1:1 on SPR, and
  gen_batchlane's ICL analysis says p1's FP pipe IS the lower half of
  p0's fused unit); paired divides in the fused map (gen_dense_prime r10:
  −1.5% when the divide is already in a compute pass's OoO shadow — my
  r5/r9 divider verdicts already said the same).

### Operation count (demo, delta vs gen_r9)

Sizes with 8 ∤ L: IDENTICAL instructions where it matters (bit-identical
outputs; only .text placement and the two appended codelets differ).
Sizes with 8 | L: one whole combine level deleted per axis pass — at 32
per pencil-group: one 32-row zmm read+write round trip through the lane
buffers gone, twiddle multiplies 33 → 21, twd_rec invocations 21 → 9; at
40: 3 levels → 2, twiddle muls 39+... → 28 (comb8 m=5), rec invocations
similar. DFT8 butterfly = 2x DFT4 + 2 c8-scaled combines (4 mul + 8 add
more than two separate DFT4s, far less than the deleted level). gf-arm
handoff at 27/31/40/50/100: −16 p5 shuffles per 8x8 tile (extract-to-
memory), store uops 8x512 → 16x256 (retire-neutral). Plan tables: the
r = 8 level uses the standard CT split twiddles (7m pairs, audited
≤ 0.51 ulp as every table); no fold/dense tables for r = 8. Setup
unchanged (≤ 0.1 s at L=100; 0.005–0.008 s at 32/40).

### What I would do next (gen_r11 / endgame)

1. **Radix-16 or split-radix top level for 2^k** (32 = 16·2, 64 = 16·4):
   another level deletion at 32/64/128 — but 32 is now 2 levels (8·4) and
   the leaf is already register-resident; the remaining win is smaller.
   Check gen_pow2's codelets first; that cell is theirs at 55 µs.
2. **PMU counter read of the scatter_gf p5 delta** (tools/pmu.sh on a81n2,
   /tmp/perf restaged by gen_layout): predicted ~−16 p5/tile; confirm the
   port picture the same way gen_pow2 queued theirs.
3. The L=12 residual: whole-binary link-order is the only lever left
   (r8's conclusion stands).
4. refnd pitch #12 if gen_powp / gen_pfa_large still build gate references
   with double cexp.

## Round gen_r11

### Adoption status (the score)

- **gen_bluestein's adoption stands** (tw_chirp + colmajor filler/audit); the
  LIBRARY half of the file is UNCHANGED for the sixth round running (the
  frozen-layer doctrine) — `gcc -c impl/gen_bluestein.c` verified clean
  against the shipped file.
- **gen_pow2's tw_cis_ds adoption stands** (their GP2_FTW constant generator).
- **refnd double-cexp gate reference, pitch #12**: gen_powp and gen_pfa_large
  still build the create()-gate reference W with double `cexp` (250 ulp on a
  1e-13 yardstick at L=100). `tw_fill_dft_cplx` remains the one-line fix.

### The mandatory counter protocol first (brief r11: baseline BEFORE code)

Method: tools/pmu.sh on a80n0 (leased core), the graded L=100 chain run at
--samples 2 and --samples 4, and the DIFFERENCE taken — exactly 128 chain
steps of counters, setup/warmup excluded by construction. Baseline = the r10
ship binary (7182.2 µs/xform same window, sd 0.05%).

Per chain step, r10 baseline → r11 ship:

| counter | r10 | r11 | delta |
|---|---|---|---|
| cycles | 24.84M | 22.51M | −9.4% |
| instructions | 37.10M | 30.76M | −17.1% |
| p0 (FMA) | 8.04M | 7.86M | −2.2% |
| p1 | 3.13M | 1.94M | −38% |
| p5 (shuffle) | 10.04M | 9.79M | −2.5% |
| p2_3 (loads) | 9.49M | 5.91M | **−37.8%** |
| p4_9 (stores) | 5.78M | 4.67M | −19.2% |
| l1d.replacement | 3.53M lines | 3.53M lines | **unchanged** |
| IPC | 1.49 | 1.44 | |

**The round's open disagreement, settled for this engine**: total measured
vector-relevant dispatch (p0+p1+p5+p2_3+p4_9) at L=100 is 1.47 uops/cycle
baseline (1.34 after) against the ~2.1 uops/cycle cap the Ice Lake notes
describe, at IPC 1.49 and p0+p5 = 0.73/cycle. This engine is NOT
uop-saturated at L=100 — it is traffic-bound, and the binding stream is
visible in the after-numbers: deleting 37% of the load dispatch bought only
9.4% of the cycles because l1d.replacement (3.53M lines = 226 MB through L1
per step, ~1.8x this engine's 128 MB stream floor) did not move. What is
left at 100 lives in the axis-0 strided pass and the DRAM-resident volume,
not in the recursion's arithmetic or its lane-buffer traffic.

### What SHIPPED: radix-10 levels via PFA 2x5 DFT10 codelets — one whole
### level pass deleted at 10/30/50/70/90/100/110 (−11% at 100, −12% at 50,
### −18% at 10, all 4/4 node pairs; gates exact everywhere)

The all-hands-on-L=100 change from this engine's angle. L=100 factored
4·25 → three level passes per pencil-group (leaf5, comb5, comb4) with 155
twiddle multiplies. The coprime pair 2x5 gives DFT10 by Good-Thomas with
ZERO internal twiddles and zero negations: 5 DFT2s on input pairs {k, k+5}
(the CRT-even member leads the difference, absorbing the (−1)^b sign), then
two of twd_leaf5's DFT5 bodies VERBATIM. Output map, derived and
independently verified (X[k] = Σ_b e_b W10^{kb}, e_b = x_b − x_{b+5}, for
odd k): even Y[2j] = DFT5(s)[j]; odd Y[(5+2j) mod 10] = DFT5(d)[j], i.e.
j = 0..4 → Y5, Y7, Y9, Y1, Y3.

Three pieces:

1. **twd_leaf10 / twd_comb10** — whole-level register-resident codelets in
   the gen_r6/r10 mold (comb10 = the entire k1 = 0..m−1 combine in one
   noinline call, 9 twiddle mults per k1 feeding the PFA butterfly directly;
   definitions at FILE END, declarations up top — the r5/r8/r10 layout
   discipline). ~20 rows live; gcc spills a few zmm, still far cheaper than
   the whole-level lane-buffer round trip the codelet deletes.
2. **Depth-minimizing factorizer**: radix 10 joins the candidate set
   {8, 4, smallest prime, 10}, chain chosen by MINIMIZING LEVEL COUNT with
   ties broken in the old preference order and 10 LAST. Audited exhaustively
   (old vs new chains, n = 2..128): exactly 8 sizes change — 10 [2,5]→[10]
   (one leaf, no combine at all), 30/50/70/110 (3→2 levels), 80 [8,2,5]→
   [8,leaf10], 90 (4→3), 100 [4,5,5]→[10,10] — everything else, including
   20/40/60/120 (ties), factors EXACTLY as before.
3. **twd_butterfly case 10 under -DTWD_DS only** (the r10 case-8 pattern):
   default builds route r=10 to the codelets in twd_rec's switches; the DS
   race arm needs the case, DENSEBF reuses comb10 like comb8.

Operation count at L=100 per pencil-group: level passes 3 → 2 (one full
100-row zmm read+write round trip through the lane buffers deleted), twiddle
multiplies 155 → 90, twd_rec invocations 25 → 11. At 50: [2,5,5] → [10,5].
At 10 the whole axis is ONE leaf10 call — no combine level exists.

### Measured on the node (a80n0, ONE core per session via slot lease,
### same-core interleaved ROTATED pairs vs the r10 control built from
### impl_10 (verified separate inode), 2 warmups, graded chains, min µs)

| L | B | r10 ctl (same window) | gen_r11 | delta | verdict |
|---|---|---|---|---|---|
| 100 | 1 | 7370–7954 | **6582–7057** | **−11%** (4/4) | the round's target cell |
| 50 | 4 | 710–717 | **625–633** | **−12%** (4/4) | |
| 10 | 64 | 5.21–5.55 | **4.29–4.35** | **−18%** (4/4) | |
| 12 | 64 | 7.64–7.93 (7 prs) | 7.70–8.09 | **+0.3..+2.6% (7/7)** | layout tax, below |
| 25 | 16 | 74.0–75.7 | 74.2–78.3 | wash (mixed signs) | no new code executes |
| 32 | 8 | 152.3–155.7 | 153.8–156.2 | wash (mixed signs) | |

Ship-binary tryout reads (fresh cores/windows, sd ≤ 0.07%): 100 B=1
**6626.4** (MKL same window 7821.6 — the demo beats MKL by 15% at the
weakest cell; quiet PMU-window read **6306.9**, r10 same protocol 7182.2 =
−12.2%); 50 B=4 **647.3** (MKL 960.7); 10 B=64 **4.283** (MKL 4.667), 10
B=1 **4.265** (MKL 4.348) — the demo now beats MKL at 10, 32, 40, 50, 100.
Wallaby (SPR) pre-reads agreed on every verdict before any node window was
spent (−10% at 100, −2% at 50, −12% at 10).

### Gates (ship binary, on the node; tryout's map-check leg still dies on
### the '$W/c.bin' quoting bug — check.py run by hand, r2 recipe)

Changed sizes, all on node codegen: singles 2.8–4.6e-16 (tol 1e-12) at
10/30/50/70/80/90/100/110; two-step m=2 8.6e-16 (10) / 1.6e-15 (30) /
2.3e-15 (50) / 2.1e-15 (70) / 2.5e-15 (80) / 2.7e-15 (90) / 2.7e-15 (100) /
2.7e-15 (110) vs tol 3e-14; graded chains PASS inside the anchor band:
1.259e-13 (10 m=1000, anchor 1.081e-13) / 4.576e-14 (50 m=128, 2.922e-14) /
4.138e-14 (100 m=64, 2.416e-14), tol 1e-10; every chain bit-repeatable
(cmp of independent runs). L=100 single accuracy IMPROVED: 4.612e-16 vs
r10's 4.817e-16 (one fewer twiddled level = fewer roundings — the layer's
own accuracy claim made visible). Unchanged sizes: outputs BIT-IDENTICAL to
the r10 binary at 12/15/20/25/27/31/32/40/45/64/77/96/121/127 (wallaby,
singles + m=3 chains) AND at 12/25/32 with node codegen (graded chains).
Race arms: DS / DENSEBF / scalar -march=x86-64 builds PASS vs numpy at
10/50/100/110; all knob combinations (DS / DENSEBF / DS+DENSEBF / MAPPAIR /
PF / NOGF / XSTG / GF_MIN_BYTES=1) compile -Wall -Wextra clean;
GEN_TWIDDLE_LIB_ONLY adoption compiles clean. setup ≤ 0.09 s at L=100.

### What did NOT work / residuals, with the number

- **The L=12 layout tax is back, small and real: +0.3..+2.6%, 7/7 rotated
  pairs, on bit-identical output.** nm -S names it exactly: twd_rec +304 B
  and its constprop/inlined copies (+245/+370 B) — the two unavoidable new
  dispatch cases — displace the small-L hot text. The r10 mitigation
  (definitions at file end, DS-only butterfly case) was applied from the
  start; this residual is the dispatch sites themselves. Accepted and
  documented (r8/r10 precedent): the cell belongs to gen_race/gen_pfa_small
  at 1.9 µs, and the 100/50/10 wins dwarf it. Link-order remains the only
  lever (r8 conclusion stands).
- No other negative: every changed size won its pairs and every unchanged
  size read wash. The PFA sign/output-map derivation was verified
  symbolically before any build (X = δ_0, δ_1 identities), which is why the
  first compiled version passed every gate — recommended over debugging
  butterflies through the harness.

### Borrowed this round, named

- **gen_pfa_small / gen_pfa_large (the campaign's PFA lineage)**: the
  coprime-factors-need-no-twiddles fact, applied INSIDE a codelet where the
  index permutation is free register naming — the first PFA structure in
  this engine.
- **My own gen_r6/gen_r10 codelet pattern**: whole-level register-resident
  noinline codelets, file-end placement, DS-only butterfly case.
- **gen_rader gen_r9 (via my r10)**: control built from impl_10, inode
  verified — never the live impl symlink.
- **gen_batchlane gen_r4 / gen_pow2 gen_r5 / gen_race gen_r9**: held-lease
  same-core interleaved rotated pairs with 2 warmups — every keep/kill.
- **gen_planner gen_r7**: nm -S as the layout-drift detector.

### What I would do next (gen_r12)

1. **The axis-0 pass is now the L=100 cell's whole story**: l1d.replacement
   sits at 3.53M lines/step (226 MB, ~1.8x the stream floor) and did not
   move; instructions fell 17% but cycles only 9.4%. The excess is the
   axis-0 strided gather/scatter walking 100 concurrent 128 B row streams
   (hardware prefetchers track ~16). A z-tiled axis-0 sweep (process 8-16
   (y,z)-columns of ROWS at a time so the live stream count fits the
   prefetcher) is the measurable next lever; success metric per the brief:
   l1d.replacement per step down toward 2M, then cycles follow.
2. **Radix-20 (PFA 4x5) codelet**: 20 = [4,5] and 40 = [8,5] still pay a
   combine level that a 20-point PFA leaf would delete (40 = 2·20). Same
   derivation, h... no fold needed — pure exact constants. Worth one window
   at 20/40 if idle.
3. xarch: race the new chains on CLX/SPR when the advisory lands (the
   depth-minimizing factorizer is host-independent; only the custody gates
   are knobs).
4. refnd pitch #13 if still double-cexp.
