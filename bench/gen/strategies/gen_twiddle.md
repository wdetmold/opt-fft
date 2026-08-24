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
