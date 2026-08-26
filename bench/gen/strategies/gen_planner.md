# gen_planner -- factorization search + candidate enumeration (library layer, scored by adoption)

## Round gen_r1 -- from the dense-floor stub to the actual planner library

### What shipped

`impl/gen_planner.c` is now a dual-mode single file:

1. **The library** (`#define GEN_PLANNER_LIB 1` then `#include "gen_planner.c"`
   from any other entry in impl/ -- everything is `static pln_*`, no fft3d_*
   symbols, composes with any class entry):
   - `pln_enumerate(arena, L, cand[], maxc)` -- L -> up to maxc candidate
     algorithm TREES over {mixed-radix CT, Good-Thomas PFA, Rader, Bluestein,
     dense}, sorted by a measurement-calibrated cost model. Each candidate
     carries a canonical name string (`c8(d4)`, `gt(d4,c5(d5))`,
     `rad31(c3(c5(d2)))`, `bs64(c8(d8))`) intended verbatim as the per-(host,L)
     wisdom key for gen_race, plus its model cost and the tree.
   - `pln_xplan_build(tree, tile)` / `pln_xplan_exec` -- a generic stride-capable
     1-D executor that runs ANY candidate tree exactly: rows of `w <= 64`
     contiguous complex, arbitrary row stride, so one engine serves all three
     axes. CT reads its decimated sub-sequences directly through strides (no
     gather pass); PFA gathers/scatters through precomputed CRT permutations;
     Rader = gather by primitive-root powers + two sub-FFTs against a
     plan-time kernel; Bluestein = chirp + two sub-FFTs at padded pow2 M.
   - `pln_p3d_build(L, tree)` / `pln_p3d_exec` -- the full 3-D row-column
     wrapper (axis 0 stride L^2, axis 1 stride L, axis 2 via blocked L x L
     transpose), any 2 <= L <= 128. This is the correctness oracle / round-6
     existence fallback every class owner can adopt for free.
   - All twiddles/matrices/kernels/chirps computed at plan time in long double
     with integer argument reduction (`pln_omegal`); Rader V and Bluestein H
     via an O(n^2) long-double DFT -- exact tables for free, per the brief's
     twiddle contract. No trig in any execute path, everything in consumption
     order.
2. **The standalone entry**: `fft3d_supports()` = ANY 2 <= L <= 128 (the roster
   scopes this class to "all"). create() enumerates, picks cand[0]
   deterministically (so independent processes are bit-identical), builds the
   3-D engine. `GEN_PLANNER_RACE=1` switches create() to timing the top 4
   candidates on a scratch volume and picking the measured winner -- the
   gen_race composition, demonstrated and working, default-off only for
   cross-process repeatability of the dev harness.

### Measured on the reserved Ice Lake node (a80n0) via tryout.sh, graded chain

All eleven acceptance cases, single generic engine, no per-size code:

| case | us/xform (min) | GF/s | rel L2 |
|---|---|---|---|
| L=10  B=64 m=1000 | 8.59  | 5.8 | 2.8e-16 |
| L=12  B=64 m=600  | 14.14 | 6.6 | 3.0e-16 |
| L=15  B=32 m=600  | 29.26 | 6.8 | 3.2e-16 |
| L=20  B=32 m=256  | 62.12 | 8.4 | 3.2e-16 |
| L=25  B=16 m=256  | 137.4 | 7.9 | 3.7e-16 |
| L=27  B=16 m=200  | 216.9 | 6.5 | 4.0e-16 |
| L=31  B=16 m=140  | 609.4 | 3.6 | 4.9e-16 |
| L=32  B=8  m=250  | 307.1 | 8.0 | 3.3e-16 |
| L=40  B=8  m=128  | 669.8 | 7.6 | 3.8e-16 |
| L=50  B=4  m=128  | 1543.7| 6.9 | 4.3e-16 |
| L=100 B=1  m=64   | 15266 | 6.5 | 4.5e-16 |

B=1 costs nothing extra per transform (per-volume engine): L=10 9.07, L=25
151.3, L=32 304.1 us -- unlike the batch-lane engines this layer never pays a
remainder penalty, which matters for round-6 draws at odd batches.

Gates: full local sweep L=2..128 ALL PASS vs numpy (worst 1.1e-15 at L=113);
two-step map gate 8.5e-16..2.8e-15 (tol 3e-14) at 10/25/31/100; full graded
chains 3.2e-14..1.0e-13 against honest anchors 2.3e-14..1.1e-13 (tol 1e-10) --
drift is AT the anchor, exact tier. Bit-repeatable across processes (checked
by cmp at 10/25/31/100). setup <= 1 ms at every L including Bluestein-127 --
the 60 s cold / 50 ms warm plan budgets are met by 3-4 orders of magnitude
without even needing the wisdom cache.

### Cost-model calibration (the real work of the round)

I timed EVERY enumerated candidate at all 11 acceptance sizes on the node
(`build/tryout/gen_planner/candbench.c`, kept there for reuse). Raw flop
counts mis-ordered several sizes; three corrections fixed the ordering:

- **CT needs a per-leaf-invocation term `+30*m`**: larger leading leaf radix
  measures faster than flops predict (c8(d4) 198 us vs c4(d8) 221 at L=32;
  c5(c4(d5)) 15.6 ms vs c4(c5(d5)) 16.0 at L=100; c5(d3) < c3(d5) at 15).
- **PFA copies are expensive: permutation penalty 10n -> 30n.** Good-Thomas
  never wins in this executor (gt(d5,d8) 574 us vs c8(d5) 468 at L=40) -- the
  gather+scatter passes eat the twiddle savings. Without the fix the CT term
  above would have wrongly promoted gt at L=40. (PFA stays enumerated: a
  class engine with fused permutations, e.g. gen_pfa_*'s baked slot offsets,
  has the opposite economics -- the model constant is executor-specific and
  gen_race should re-race it per engine.)
- **Rader overhead `10(p-1)+12p -> 24(p-1)+16p`**: the raw model preferred
  rad7 over dense-7, which is absurd at w-wide vectorization. After the fix,
  dense wins model-first up to p=13 and Rader from p>=17; at the scored L=31
  the measurement agrees emphatically: rad31 551 us vs bs64 906 vs d31 1376.

After calibration cand[0] = measured winner at 9 of 11 acceptance sizes; the
two misses (10: c5(d2) picked vs c2(d5) 5% faster; 20: dead tie) are inside
run-to-run spread. That is what the default deterministic pick ships on.

- **Tile width is size-dependent**: scratch-row width 32 complex wins for
  L < 40 (L=10: 4.79 vs 5.71 us at 64), 64 wins for L >= 40 (L=100: 12.0 vs
  13.5 ms; L=50: 1.10 vs 1.18 ms). Tile is now a runtime plan parameter
  (max PLN_TI=64), crossover at L=40. TI=8 loses everywhere (tested).

### What did NOT work / dead ends, with numbers

- **Uncalibrated flop-count cost model**: inverted c8/c4 order at 32 (10%),
  preferred rad7 to d7, and after the CT fix alone would have flipped L=40 to
  PFA (574 vs 468 us = 23% loss). Pure op-counting does not order candidates
  on this machine; every constant above is fitted to node measurements.
- **PLN_TI=8**: loses 6-15% everywhere (L=25: 102.6 vs 91.1 us) -- per-call
  overhead dominates under row width 16 doubles.
- **Fixed PLN_TI=16 (initial choice)**: left 6-12% on the table at both ends
  of the size range; see crossover above.

### Borrowed, plainly

- **gen_batchlane's harness notes** (strategies/gen_batchlane.md): the wallaby
  squeue PATH shim for reserve.sh --status, the tryout.sh `$W`-unbound-under-
  `set -u` workaround (export W first), and the knowledge that the remote
  map-check line dies on the `'$W/c.bin'` quoting bug -- I ran all map-checks
  manually on the shared FS instead of rediscovering any of this.
- The create()-time race idea is the fixed-size campaigns' fft3d_best
  choose()/trial() (my seed material), reduced to a 4-candidate min-of-3 race.

### Notes for class owners and the monitor

- The executor is deliberately generic (interleaved complex, autovectorized
  over w); it is a correctness oracle and floor, not a rival to your codelets
  -- MKL is still ~2-4x faster at most sizes and batchlane beats me 7x at
  L=10 B=64. Adopt `pln_enumerate` for decomposition choice + names, and
  `pln_p3d` as your any-L fallback so round 6 can never zero you.
- For gen_race: `pln_cand.name` is the wisdom key; enumeration is
  deterministic; sub-trees below the root currently take their single
  model-best form, so root candidates are diverse but sub-decompositions are
  not raced independently yet (see next).

### Next round, in order of expected value

1. **Kill the axis-2 transposes**: leaf kernels with a column-stride parameter
   (or in-register 8x8 transposes) so z-pencils feed the engine directly;
   the two L^2 copy passes per plane are the largest non-arithmetic cost at
   L >= 50.
2. **Symmetric odd-prime leaf** (half-matrix via t_j = x_j + x_{p-j},
   u_j = x_j - x_{p-j}): halves dense work at 7/11/13/31 and inside Rader's
   dense sub-blocks; directly moves the scored L=31 and the d31 crossover.
3. **Sub-tree diversity in enumeration**: emit top-2 sub-candidates at each
   composite child so gen_race can race e.g. rad31(c2(c3(d5))) vs
   rad31(gt(d5,c3(d2))); the arena/memo structure already supports it.
4. **Batch-lane composition**: at B >= 8, hand whole batched pencil sets to a
   gen_batchlane-style SoA engine chosen by the same candidate trees --
   coordinate with gen_batchlane, whose record already flags this seam.
5. **Wisdom persistence hook**: create() currently re-picks by model each
   process; once gen_race lands results/wisdom_<host>.json, read it keyed by
   my candidate names and the default pick becomes the raced winner without
   losing determinism.

## Round gen_r2 -- own the chain, run everything in place, race what is graded

### What changed (all in impl/gen_planner.c; library API unchanged)

1. **The executor is now unconditionally in-place safe at any root** (every
   node kind reads all of src before writing dst): dense leaves stage their
   rows through the arena (which also replaces p strided sweeps over a
   far-strided source with one), Rader stashes x0 before writing X[0], and
   CT/PFA/Bluestein already consumed src into the arena first. This is the
   enabling property for everything below.
2. **Per-plane fused axes 1+2, tmp volume deleted.** pln_p3d_exec is now:
   axis 0 streams in -> out once; then each x-plane gets axis 1 in place, a
   transpose into ONE scratch plane with an odd-cache-line row pitch (the
   gen_pfa_large / L23_rader anti-4K-alias pitch, applied to my axis-2 row
   stride), axis 2 in place there, transpose back. The plane stays
   cache-resident across both passes; the gen_r1 structure paid a full extra
   volume write+read (axis1 -> tmp) plus a second scratch plane.
3. **CT twiddles fused into the leaf loads.** Twiddle tables moved to k2-major
   consumption order and each hard leaf (2/3/4/5/8) gained a _tw variant that
   multiplies row j1 by w^(j1*k2) while loading. Kills the separate rowscale
   read+modify+write pass over the CT buffer per node. (Dense radixes keep the
   two-phase path.) NOTE for gen_twiddle: pln_x->tw layout changed from
   j1-major [(j1-1)*m + k2] to k2-major [(k2-1)*(r-1) + (j1-1)], k2 >= 1 only.
4. **fft3d_chain exported -- the round's biggest win.** The graded m-step map
   chain now runs VOLUME-MAJOR (volumes are independent, the map is pointwise):
   each volume is copied once into final_out and its whole m-step chain runs
   fully IN PLACE there -- state and c are the only volume-sized streams, and
   at L<=27 both stay L2-resident across all m steps. The map
   z/(1+|z|) is fused per x-plane right after that plane's axis-2 pass while
   the plane is hot, computed with rsqrt14/rcp14 + 2 Newton steps each
   (~1e-16 rel; the campaign-standard ladder -- plain vsqrtpd/vdivpd measured
   ~4x slower at this op mix, worth ~40 us of the old 137 at L=25). The
   driver fallback previously charged me execute (with the tmp round trip) +
   a full-volume sqrt/div map + ping-pong traffic every step. A create()-time
   gate (ice L17_rader r5 discipline) checks the owned step against execute +
   the driver's exact scalar map over two steps on a random volume (tol 1e-12)
   and falls back to execute + scalar map per step if it ever disagrees.
5. **Folded odd dense leaf/radix (n >= 11)**: conjugate fold s_j = x_j+x_{n-j},
   d_j = x_j-x_{n-j}; X_k = E_k -+ iO_k with E/O from REAL (h x h) cos/sin
   matrices, 2n^2 flops instead of 8n^2 -- the gen_rader / gen_dense_prime
   half-system, generalized to any odd length and usable as a CT radix.
   d31 went 1376 -> 594 us (2.3x) -- still loses to rad31 (below), but every
   odd prime 11..63 leaf and Rader/Bluestein dense sub-block gets it free.
6. **The create()-time race is ON by default and races the CHAIN STEP**
   (gen_pfa_large's "race what is graded" lesson), top 4 candidates, min of 3,
   2% simplest-first hysteresis. GEN_PLANNER_RACE=0 restores the deterministic
   model pick; GEN_PLANNER_VERBOSE=1 prints the pick; GEN_PLANNER_TILE
   overrides the row width. Setup stays trivial: worst 0.29 s at L=100
   (cold budget 60 s; the 50 ms warm target still wants gen_race's wisdom).

### Measured on the node (a80n0, graded chain, min over samples; final binary)

| case | gen_r1 | gen_r2 | delta | MKL 2022 | vs MKL | picked (node race) |
|---|---|---|---|---|---|---|
| L=10  B=64 m=1000 | 8.59  | **6.28**  | -27% | 4.55  | 0.72x | c2(d5) |
| L=12  B=64 m=600  | 14.15 | **10.72** | -24% | 7.74  | 0.72x | c4(d3) |
| L=15  B=32 m=600  | 29.26 | **21.70** | -26% | 16.49 | 0.76x | c5(d3) |
| L=20  B=32 m=256  | 62.12 | **46.58** | -25% | 58.83 | **1.26x** | c5(d4) |
| L=25  B=16 m=256  | 137.5 | **105.6** | -23% | 121.1 | **1.15x** | c5(d5) |
| L=27  B=16 m=200  | 216.9 | **155.0** | -29% | 149.7 | 0.97x | c3(c3(d3)) |
| L=31  B=16 m=140  | 606.8 | **533.6** | -12% | 850.7 | **1.59x** | rad31(c3(c5(d2))) |
| L=32  B=8  m=250  | 306.0 | **236.9** | -23% | 176.2 | 0.74x | c4(d8) |
| L=40  B=8  m=128  | 666.3 | **503.5** | -24% | 412.9 | 0.82x | (race; c8(d5) family) |
| L=50  B=4  m=128  | 1535.4| **1222.3**| -20% | 954.0 | 0.78x | c5(c5(d2)) |
| L=100 B=1  m=64   | 15044.6| **9607.6**| -36% | 7901.2| 0.82x | c4(c5(d5)) |

The generic engine now BEATS MKL at 20, 25, 31 and ties it at 27 -- gen_r1
beat it only at 31-by-default (everything beat MKL at 31). B=1 == batched per
transform (the volume-major chain never pays a batch remainder): 6.23 (10),
105.2 (25), 234.6 (32), 1174.9 (50).

Gates, final binary on the node: single-call rel L2 2.9e-16..4.9e-16 at all
11 cases; map chains 1.4e-13 (10, anchor 1.1e-13), 3.1e-14 (27, anchor
2.6e-14), 3.5e-14 (31, anchor 2.3e-14), 5.5e-14 (100, anchor 2.4e-14) -- all
within 1.3-2.3x of the honest anchor, tol 1e-10. Local numpy sweep L=2..128
(76 sizes incl. every odd prime and odd-radix CT): ALL PASS. The create()
chain gate passed at every size tried (it has never fired the fallback).

### What did NOT work / went wrong, with the number

- **Folded dense as a small CT radix**: fold-9 inside c9(d3) at L=27 took the
  chain 155 -> 228 us. At h=4 the stage pass + per-k accumulator setup costs
  more than the 81 -> ~40 flop saving. Fold now requires n >= 11. Between the
  same builds d31 folded was 2.3x faster than plain d31 -- the fold is a
  large-n device.
- **Fold model cost at flop parity (2n^2) promoted d31 over rad31**: measured
  d31 594 vs rad31 531/533. Calibrated to 5.5n^2 + 8n, which keeps rad31
  first at 31 AND keeps the dense->Rader crossover at p=17 (r1's measured
  boundary). Raw op counts keep failing to order candidates on this machine
  -- same lesson as gen_r1, now including my own new kernel.
- **The r1-calibrated CT ordering inverted under the fused executor**: model
  cand[0] c8(d4) at 32 measured 282 vs raced c4(d8) 236 (-16%); model
  c5(c5(d4)) at 100 measured 10185 vs raced c4(c5(d5)) 9574 (-6%). I did not
  re-fit the CT constants; the chain-step race is the fix and is now the
  default. The model's remaining job is candidate ORDER (the race sees top 4).
- **ssh sessions land in $HOME** (gen_powp warned; I lost one round trip
  anyway): cd explicitly in every remote command. The tryout map-check c.bin
  quoting bug is still there; all map-checks above were run manually on the
  shared FS.

### Borrowed this round, named

- **gen_pfa_large / gen_rader / gen_powp**: the whole "own the chain" program
  -- in-place passes, volume-resident chain, map fused where residency allows,
  race the chain step not execute, create()-time chain gate. My numbers above
  are substantially their r1 lessons applied to a generic engine.
- **gen_rader + gen_dense_prime**: the conjugate fold arithmetic (their
  half-system form), generalized into pln_dense_fold_apply.
- **gen_pfa_large (via L23_rader)**: odd-cache-line plane pitch for the
  transposed axis-2 scratch.
- **The campaign-standard NR map ladder** (everyone's records): rsqrt14/rcp14
  + 2 Newton each, 1e-300 clamp.

### Notes for adopters

- pln_enumerate / pln_p3d_build / pln_p3d_exec unchanged. NEW:
  pln_p3d_step(p3, state, cfield) = one full in-place graded chain step
  (FFT + c + map) on one volume -- adopt it and fft3d_chain is three lines.
- pln_x->tw is now k2-major (see item 3) -- gen_twiddle's drop-in comment
  refers to the old layout.
- The executor is in-place safe at every node; class owners can call it
  src==dst on any pencil set.

### Next round, in order of expected value

1. **L=100 is still 18% behind MKL and axis 0 + transposes are the residue**:
   stage far-strided axis-0 pencils through the arena (the dense-leaf staging
   generalized -- one strided pass + contiguous tree instead of a strided
   tree), and vectorize the transposes (AVX-512 4x4 complex blocks).
2. **Small-L batch lanes**: at L=10/12/15 the executor's per-call overhead
   dominates (batchlane is 5x faster). Composing the candidate trees with an
   SoA 8-volume engine at B >= 8 is the seam gen_batchlane's record already
   flags; the volume-major chain makes the pack/unpack once-per-chain.
3. **Sub-tree diversity in enumeration** (unchanged from r1): race
   rad31(c2(c3(d5))) vs rad31(gt(d5,c3(d2))) etc.
4. **Wisdom persistence**: hand the race result to gen_race's per-host cache
   keyed by cand.name; create() then meets the 50 ms warm budget with the
   raced pick instead of re-racing.

## Round gen_r3 -- the executor goes explicit AVX-512; the compiler was the bottleneck

### The diagnosis that drove the round

gen_layout's r2 record showed their O(L^4) DENSE demo beating this true FFT at
L=10-31 (264 vs my 531 at 31). Same machine, worse algorithm, better code --
so the deficit had to be vectorization quality, not decomposition. objdump on
the node build confirmed it: pln_xexec (with the _tw leaves inlined) compiled
to 277 vmovsd / 68 vmulsd / 63 vsubsd -- HALF SCALAR -- plus 240 shuffle-class
ops against ~130 packed FMAs; pln_dense_fold_apply kept its E/O accumulators
in stack arrays and emitted 295 vmovupd against 70 FMAs (4+ loads per FMA).
gcc-11 -O3 -march=native cannot vectorize interleaved-complex butterflies.
Everything below follows from replacing those loops with intrinsics.

### What changed (all in impl/gen_planner.c; pln_* API unchanged)

1. **Explicit AVX-512 complex-vector layer** (`pv` = 4 interleaved complex per
   zmm): cmul by a scalar twiddle = 1 vpermilpd + vmul + vfmaddsub (broadcasts
   hoisted out of the column loop); `a +- i*b` = 1 vpermilpd + 1
   fmaddsub/fmsubadd; conj = 1 vxorpd. Every column loop is fully masked
   (maskz loads / mask stores), so tails are exact, OOB-safe at buffer edges,
   and there is NO scalar residue path anywhere in the executor.
2. **All five hard leaves (2/3/4/5/8) rewritten once each** as always_inline
   bodies with a compile-time HAS-twiddle flag -- the plain and _tw variants
   are two instantiations of the same code, so the k2=0 column stays
   twiddle-free for free.
3. **Register-tiled matrix kernels** (gen_dense_prime's k-quad x wide-tile
   shape, generalized): fold and plain-dense now accumulate 4 output rows x 2
   zmm of columns = 16 accumulators in registers; per input row j: 4 row loads
   + 8 broadcasts + 16 FMAs (was ~2 loads + 1 store per FMA). Matrix row
   counts are padded to a multiple of 4 (PLN_HPAD, zero rows) so there is no
   K-tail variant -- padded outputs are computed and not stored. Plain dense
   gets a new plan-time layout: WR[np][n] real parts + 16-byte (-wi,+wi)
   pairs for one broadcast_f64x2 straight into the alternating-sign FMA.
   Rader's rowscale_conj, its X = x0 + conj(a) combine, and both Bluestein
   chirp passes got the same treatment.
4. **Transposes vectorized**: 4x4 complex blocks via 4 vpermutex2var + 4
   vshuff64x2 (4 loads, 4 full-line stores, 8 shuffles per 16 complex instead
   of 16 scalar 16-byte moves); scalar edges only at L%4 rows/columns.
5. **Tile default is now 32 at every L** -- the r2 "64 for L>=40" crossover
   vanished with the intrinsic kernels (100: 6869@32 vs 6955@64; 40/50 a
   wash; 25: 70.3@32 vs 77.5@64). GEN_PLANNER_TILE still overrides.
6. **gen_race string-wisdom adoption** (their r2 hook, written for this
   entry): the raced tree name is persisted per (host, L) under
   `gen_planner/tree/L<L>`; a wisdom hit builds that tree with no race, so
   warm create() measured 4 ms (50 ms budget) and the driver's two-process
   repeatability is STRUCTURAL (different trees round differently; r1/r2
   relied on the race being quiet-stable). Stores only after the chain gate
   passes (gen_powp's discipline); GEN_PLANNER_RACE=0 skips wisdom entirely
   so the deterministic dev path stays file-free. At round end I stripped all
   gen_planner/ keys from results/wisdom_a80n0.json (flock held, format
   preserved) so the monitor's scoring run cold-races in its full-quiet
   window -- gen_powp's precedent; absent entries are deliberate.

### Measured on the node (a80n0, tryout.sh leased core, graded chain, min)

| case | gen_r2 | gen_r3 | delta | MKL 2022 same window | vs MKL | picked |
|---|---|---|---|---|---|---|
| L=10  B=64 m=1000 | 6.28  | **4.72**  | -25% | ~4.6  | 0.97x | c2(d5) |
| L=12  B=64 m=600  | 10.72 | **6.89**  | -36% | 7.7   | **1.12x** | c3(d4) |
| L=15  B=32 m=600  | 21.70 | **14.96** | -31% | 16.5  | **1.10x** | c5(d3) |
| L=20  B=32 m=256  | 46.58 | **27.92** | -40% | 58.8  | **2.11x** | c5(d4) |
| L=25  B=16 m=256  | 105.6 | **69.5**  | -34% | 121   | **1.74x** | c5(d5) |
| L=27  B=16 m=200  | 155.0 | **104.5** | -33% | 150   | **1.43x** | c3(c3(d3)) |
| L=31  B=16 m=140  | 533.6 | **204.9** | -62% | 857.6 | **4.18x** | **d31** |
| L=32  B=8  m=250  | 236.9 | **131.3** | -45% | 194.0 | **1.48x** | c4(d8) |
| L=40  B=8  m=128  | 503.5 | **282.1** | -44% | 444.8 | **1.58x** | c5(d8) |
| L=50  B=4  m=128  | 1222.3| **785.1** | -36% | 998.0 | **1.27x** | c5(c5(d2)) |
| L=100 B=1  m=64   | 9607.6| **6655.8**| -31% | 8046.6| **1.21x** | c5(c5(d4)) |

The generic engine now beats MKL at 10 of 11 acceptance cases (r2: 3). The
headline flip: **the race now picks d31 -- the register-tiled FOLD -- over
rad31 at L=31** (203 vs r2's rad31 533; rad31 also got faster but lost its
own race). The dense->Rader crossover moved past p=31; leafF's fold constant
recalibrated 5.5n^2 -> 3.0n^2 accordingly. B=1 (same code path, known ~10%
short-unit core-ramp): 5.43 (10), 78.7 (25), 144.7 (32), 878.5 (50).

Gates, final binary: single call 2.9-4.8e-16 at all 11 cases; two-step m=2
gate 0.9-3.0e-15 at 10/25/31/100 (tol 3e-14, 10-30x margin); full graded
chains 2.5e-14 (31, anchor 2.3e-14), 3.4e-14 (25, anchor 2.8e-14), 5.5e-14
(100, anchor 2.4e-14), 1.6e-13 (10, anchor 1.1e-13) -- all 1.06-2.3x the
honest anchor, tol 1e-10. Full local sweep L=2..128: ALL 127 PASS, worst
1.07e-15 (L=113, Bluestein). Chain and single outputs bit-identical across
two node processes (wisdom pins process 2; cmp verified). Setup: 0.23 s cold
at L=100, 4 ms warm.

### What did NOT work, with the numbers

- **Software prefetch of the next axis-0 tile at L=100** (the volume's L=100
  strided row streams defeat the HW prefetcher, so this looked right):
  interleaved A/B on a leased core, pf-off 6182/6460 vs pf-on 6547/6812 --
  a consistent ~5% LOSS. The axis-0 pass already saturates MLP with its own
  demand misses; 800 extra load-port uops per tile only queue behind them.
  Removed entirely. Same lesson as gen_pfa_small's "no prefetch in
  issue-bound passes", now measured on a miss-bound pass too.
- **Tile 64 at large L** (the r2 crossover): now loses or ties everywhere
  (numbers in item 5). The wider tile's win was amortizing per-call SCALAR
  overhead that no longer exists; 32 keeps the arena smaller and the plane
  hotter. Tile 16 also loses (73.6 vs 70.3 at 25).
- The r2 fold-cost constant 5.5n^2 kept d31 ranked behind rad31, which the
  tiled kernel inverted on the node. Only 3 candidates enumerate at 31 so the
  race caught it regardless -- but any size where a d-leaf must make a top-4
  cut needed the recalibration (now 3.0n^2 + 8n).

### Borrowed this round, named

- **gen_dense_prime** (via gen_race's demo tile4x8 and gen_layout's r2 fold
  engine): the k-quad x wide-tile register-blocking shape -- their measured
  155->33 us axpy-vs-tile gap is exactly what pln_fold_applyv/pln_dense_applyv
  fix; also the objdump spill/scalar-audit discipline that started the round.
- **gen_layout r2**: the existence proof that split-lane broadcast-FMA code
  beats my autovectorized executor at equal-or-worse algorithm -- their fold
  demo numbers were the round's target line. (Their gl_ arena itself remains
  unadopted here: my scratch is small and their own A/B ladder shows ~0 for
  compute-staged kernels.)
- **gen_race r2**: gr_wisdom_get_str/put_str, adopted exactly as their record
  invites (they widened GR_NAME_MAX to 128 for my names in advance); the
  "wisdom pins run 2 to run 1" repeatability rationale is now load-bearing in
  my entry.
- **gen_powp r2**: the store-only-gate-passed-picks wisdom discipline and the
  round-end strip of my own keys from the shared wisdom file.

### Operation counts (per pencil of w complex, vector ops, lanes of 4)

Hard leaf r: r maskz loads + (r-1) cmul (3 ops) + butterfly (~r log r
FMA-class) + r mask stores, zero scalar ops, zero gathers. Fold n (h=n/2):
staging h*(2 loads+2 ALU+2 stores)/4 cols + tiled E/O: per 4 k-rows x 8 cols
x j<=h: 4 loads + 8 broadcasts + 16 FMA (FMA:load-port = 16:12); combine 2
shuffles + 2 fmaddsub per (k, 4 cols). Dense n: same tile, 2 cmul-FMAs per
(k,j) per 4 cols. Transpose: 12 shuffle-class per 16 complex. Map unchanged
(rsqrt14/rcp14 + 2NR).

### What I would do next (ranked)

1. **Small-L per-call overhead** (L=10 is the one case still behind MKL, and
   batchlane is 4x ahead): the c2(d5) step at L=10 is ~34 pln_xexec root
   calls per volume plus 20 plane transposes; a fused small-L path (whole
   plane through registers, or the batch-lane SoA seam gen_batchlane's record
   flags) is where the next factor lives. Coordinate with gen_batchlane.
2. **L=100 axis-0 pass**: still the residue at 6.66 ms vs gen_pfa_large's
   5.0. Prefetch measurably fails; the structural fixes are a z-split state
   layout or plane-pair streaming (their r2 item 2). Watch their r3 record
   first.
3. **Sub-tree diversity in enumeration** (deferred three rounds now): with
   wisdom in place the race can afford 8+ candidates; emit top-2
   sub-candidates per composite child so rad31(gt(...)) vs rad31(c...) race.
4. **xarch guard**: the intrinsic layer is guarded on AVX512F+DQ with the
   full scalar executor as fallback; CLX/SPR have both, but the tile/fold
   crossovers should be re-raced per host -- which the wisdom cache now does
   by construction.

## Round gen_r4 -- fused register-resident small-n codelets; the arena round trip was the small-L tax

### What changed (all in impl/gen_planner.c; pln_* API unchanged)

1. **FUSED two-level CT codelets** (the round's one big idea): a CT node with
   n = r*m <= 25 and BOTH stages hard leaves (r, m in {2,3,4,5,8}) now runs as
   a single register-resident kernel per column chunk -- load all n rows once,
   child m-DFTs + twiddle cmuls + r-DFTs entirely in zmm, store n rows once.
   No arena staging, no per-leaf calls, no separate twiddle pass. One
   always_inline body (`pln_fusedv`) instantiated for all 19 legal (r,m)
   pairs; arithmetic and constants byte-identical to the pln_lv* leaves, so
   the precision budget is unchanged. All loads precede all stores per chunk,
   preserving the executor's universal in-place safety. asm audit
   (icelake-server, gcc 11): ZERO zmm spills up to n=20; n=25 spills 4 zmm
   (pln_f_5_5) and still wins big (below). This deletes, per former CT node
   call: 2n row stores + 2n row loads of arena traffic and ~r+m call
   dispatches -- exactly the per-call overhead my r3 record named as the
   small-L residue.
2. **Fusion reaches INNER nodes everywhere**: 50 = c5(c2(d5)) runs its inner
   10 fused, 100 = c5(c4(d5))/c5(c5(d4)) its inner 20/25, 27 = c3(c3(d3)) its
   inner 9, 75 = c5(c3(d5)) its inner 15 -- the round-6 surprise draws get
   this for free through the ordinary planner.
3. **Sub-tree diversity in enumeration** (my r1 next-list item, deferred three
   rounds): the arena memo now keeps best2[n]/bestc2[n], and every CT root is
   also emitted with the runner-up child tree. The race widened from top-4 to
   top-6 (wisdom amortizes the extra trials; cold setup at L=100 is 0.25 s
   vs the 60 s budget). Model cost for fused nodes:
   m*leafF(r) + r*leafF(m) + 4(m-1)(r-1) + 12 (no buffer/per-call terms).
4. **Masked transpose edges**: pln_tr4x4m handles nr x nc <= 4x4 tail blocks
   with masked loads/stores (zero-fill lanes die in rows/columns the masks
   drop); the scalar edge loops are gone. At L=10, 36 of 100 elements per
   plane transpose went through scalar copies, twice per plane per step.

### Measured on the node (a80n0, leased core via tryout.sh, graded chain, min; MKL 2022 same window)

| case | gen_r3 | gen_r4 | delta | MKL | vs MKL | picked (node race) |
|---|---|---|---|---|---|---|
| L=10  B=64 m=1000 | 4.72  | **3.49**  | -26% | 4.61  | **1.32x** | c2(d5) fused |
| L=12  B=64 m=600  | 6.89  | **5.18**  | -25% | 7.93  | **1.53x** | c3(d4) fused |
| L=15  B=32 m=600  | 14.96 | **11.79** | -21% | 16.50 | **1.40x** | c3(d5) fused |
| L=20  B=32 m=256  | 27.92 | **24.69** | -12% | 60.63 | **2.46x** | c4(d5) fused |
| L=25  B=16 m=256  | 69.5  | **60.2**  | -13% | 125.6 | **2.09x** | c5(d5) fused (4 spills, still -15% in the A/B) |
| L=27  B=16 m=200  | 104.5 | **86.5**  | -17% | 145.6 | **1.68x** | c3(c3(d3)), inner 9 fused |
| L=31  B=16 m=140  | 204.9 | 205.6     | 0    | 858.5 | 4.18x | d31 (unchanged path) |
| L=32  B=8  m=250  | 131.3 | 129.7     | -1%  | 181.1 | 1.40x | c4(d8) (unchanged path) |
| L=40  B=8  m=128  | 282.1 | 283-292   | wash | 415.9 | ~1.45x | c5(d8) (same tree; interleaved r3-vs-r4 A/B 3 pairs: deltas alternate sign) |
| L=50  B=4  m=128  | 785.1 | **639.3** | -19% | 943.4 | **1.48x** | c5(c5(d2))/c5(c2(d5)) near-tie, inner 10 fused |
| L=100 B=1  m=64   | 6655.8| **5528.4**| -17% | 7762.4| **1.40x** | c5(c5(d4))/c5(c4(d5)) near-tie, inner 20 fused |

Now ahead of MKL at ALL 11 acceptance cases (r3: 10 of 11; L=10 was the last
red cell). The remaining gaps to class winners narrowed: L=100 5528 vs
gen_powp's r3 5021 (was 6656 vs 5021); L=25 60.2 vs their soa 32.4. B=1 ==
batched as always (3.53 at 10, 85.8 at 27, 655 at 50). fftw3_guru is not a
threat at the sizes checked (L=100 B=1: 14691 us in its r3 json).

Gates, final binary: single call 2.9-4.7e-16 at all 11 cases (tol 1e-12);
two-step m=2 gate 0.95-2.97e-15 at 10/20/25/27/50/100 (tol 3e-14, 10-30x
margin); full graded map-chains 2.46e-14 (31, anchor 2.31e-14) to 5.88e-14
(20, anchor 2.84e-14), all within 1.06-2.1x of honest anchors, tol 1e-10.
Full local sweep L=2..128: ALL 127 PASS (both before and after the n<=25
fusion cap). Two independent node processes bit-identical at L=100
(race+wisdom path; wisdom pins process 2). Setup: 0.25 s cold at L=100,
wisdom hit ~6 ms. Non-SIMD and GEN_PLANNER_LIB adoption modes still compile.
Round end: gen_planner/ keys stripped from results/wisdom_a80n0.json under
the flock (gen_powp protocol; the file was already otherwise emptied by the
other entries' own round-end strips -- absent entries are deliberate, the
monitor cold-races on its quiet window).

### What did NOT work / open items, with numbers

- **L=40 shows no gain and wobbled +2% in the first window**: the picked tree
  c5(d8) contains no fusable node (n=40 > 25, d8 child of c5 makes r*m=40)
  and 40%4==0 so the transpose edges never fire. Interleaved A/B against the
  rebuilt r3 binary (3 pairs, same core): r3 281.0-287.6 vs r4 283.1-292.3
  with the sign flipping between pairs -- window noise on identical kernels,
  not a regression. A c5-outer/fused-8-inner shape does not exist; the
  structural next move at 40 is a fused (5,8) variant with spill-managed 13+
  live vecs or the batchlane seam, not more racing.
- **GEN_PLANNER_TILE=64 at L=100 under the fused inner nodes**: 5524@32 vs
  5553@64 (quiet pair; the busy pair agreed) -- tile 32 stays, third round
  running.
- The fresh 50/100 races flip between near-tie trees (c5(c5(d2)) vs
  c5(c2(d5)); c5(c5(d4)) vs c5(c4(d5))) depending on window: the diversity
  candidates are real contenders, and the monitor's cold quiet-window race
  picks whichever is true there. That is working as designed.

### Borrowed this round, named

- **gen_batchlane r1 harness notes** (again): the wallaby squeue PATH shim
  (recreated verbatim, heartbeat-gated) and the by-hand map-check for
  tryout's still-unfixed '$W/c.bin' quoting bug.
- **gen_dense_prime r3 / gen_rader r3**: the interleaved control-first A/B
  protocol for bimodal windows (used at 40 and for the fusion-cap A/B).
- **gen_pow2 r3's DSB warning** shaped the fused design: one column chunk per
  loop body, kernels stay ~1-2 KB; no multi-column unrolling was attempted.
- The fused codelet idea itself is the fixed-size campaigns' whole-transform-
  in-registers doctrine (every class winner does this); this round
  industrializes it inside the generic executor, picked up by the ordinary
  planner enumeration rather than per-size code.

### Operation counts (fused node, per 4-column chunk)

n rows loaded (masked), r child DFT_m butterflies + (r-1)(m-1) cmuls
(3 FMA-class + 2 broadcast loads each) + m cross DFT_r butterflies, n rows
stored (masked). Zero arena traffic, zero shuffles beyond the leaves' own
addi/subi permutes. vs r3 staged CT: saves 4n row moves (2n stores + 2n
loads) and r+m dispatches per node call. Transposes: edge blocks now cost
the same 8 shuffles + masked I/O as interior blocks.

### What I would do next (ranked)

1. **L=100 axis-0/plane-pass residue vs gen_powp's 5.0 ms**: the gap is now
   10%; the remaining structure is their codelet-grade x-pass. A PMU session
   (gen_pfa_large's r3 advice: attribute port-5 vs DRAM before speculating)
   should decide between wider fused shapes and traffic work.
2. **Fused (5,8)/(8,5) with explicit spill scheduling** for L=40's inner 40
   -- the n=25 result (4 spills, -15%) says the register cliff is soft;
   n=40 needs ~20 planned spills and may still beat the arena round trip.
3. **Batch-lane seam at 10/12/15** (gen_batchlane is still 3x ahead at 10):
   the fused kernels cut my per-call overhead, but their SoA-8 layout
   amortizes across volumes -- composition remains the unexplored axis.
4. **xarch**: the fused kernels are pure AVX-512F+DQ, guarded; CLX/SPR
   re-race via wisdom by construction. Check XARCH.md when it lands for the
   tile and fusion-cap crossovers (SPR's second FMA pipe may move the n=25
   spill trade).

## Round gen_r5 -- the split-group batch engine: the batch-lane seam, done generically

### The round in one line

At batch >= 8, groups of 8 volumes now run site-major SPLIT-complex (site =
{8 re, 8 im} = 2 zmm, lane = volume): every pencil DFT is full-width with
broadcast-scalar twiddles -- ZERO shuffles (asm-audited), zero masked tails,
zero transposes (axis 2 runs directly on strided rows), and a map ladder that
sees 8 distinct sites per zmm instead of 4.  This is the composition seam my
r2/r3/r4 next-lists kept deferring, built as three generic levels that enter
the create() race as extra arms.

### What changed (all in impl/gen_planner.c; pln_* API unchanged, additive)

1. **Split-lane codelet layer** (`pw_*`): split mirrors of the pv leaves
   (2/3/4/5/8), the fused two-level CT codelet, and the folded odd dense
   kernel.  Arithmetic is op-for-op the interleaved form (fmsub/fmadd pairs
   reproduce fmaddsub's per-lane rounding), so split outputs are
   BIT-IDENTICAL to the per-volume path on the same tree -- verified by cmp
   through the whole graded chain.  cmul = 2 mul + 2 fma (no permute);
   a +- i*b = add + sub (no permute).  asm audit (icelake-server, -O3
   -funroll-loops): shuffle-class ops in every split codelet = 0; spills
   20/24/96/128 rsp-zmm-ops at (2,5)/(3,4)/(4,5)/(5,5); leaves spill-free.
2. **Three group levels, raced per (host, L, batch-regime)**:
   - `@s1` fused root (n = r*m <= 25, both hard): one register-resident
     codelet call per pencil, all three axes in place.  Picks: 10, 12, 15,
     20, 25.
   - `@s2` two-pass CT (r hard; child = hard leaf or fused CT): pass 1
     child DFT_m (undoes decimation, src->dst), pass 2 twiddle-fused leaf_r
     in place; the step runs as TWO fused volume sweeps -- axes 0+1 back to
     back per z-SLAB (both axes live inside the fixed-z slab, L^2 x 128 B,
     L2-hot between them), axis 2 per x-plane.  The slab merge took 27 from
     an 86-us tie with pv to 78.8-79.3 vs 86.0-86.3 same-core (-8.5%): the
     lev-2 step was L3-traffic-bound, and 3 sweeps -> 2 is the fix.  Pick:
     27 = c3(c3(d3)) with fused-9 child.
   - `@s3` folded odd dense root: the split fold with 4-k-row tiled E/O
     accumulation, per-pencil 4 KB staging, single pass per axis in place.
     Pick: 31 = d31, 141 us vs r4's 205 (-31%), 6.0x MKL.
3. **Map fused into the final-axis stores** (`pw_stmap`), gated by residency:
   same-core pairs show -8% at 25, -4% at 15, -9% at 31, but +5% at 10
   (L2-resident group; the ladder's ~8 temps push the fused codelet into
   spills) and a wash at 12.  Shipped rule: fuse iff L^3 > 1728.  Identical
   ladder ops either way -- bit-identical, and the split ladder is HALF the
   interleaved ladder work per site (no duplicated |z| lanes).
4. **The create() race is INTERLEAVED sample-major** (gen_race r4's protocol
   fix, adopted): all arms built and warmed first, then min-of-3 round-robin
   rounds.  Group arms are timed on a real 8-volume group and scored /8.
   Arm order (pv by model cost, then split by level) preserves the 2%
   simplest-first hysteresis.
5. **Wisdom is keyed by batch REGIME** (`gen_planner/tree/L<n>/g8` vs
   `.../L<n>`), value = tree name + `@s1/@s2/@s3` tag.  Caught in dev: with
   one key per L, a B=1 create() re-raced and CLOBBERED the batched @s1
   pick, silently reverting the batched cell to pv on the next wisdom hit.
6. **Chain plumbing**: groups of 8 pack once (scalar interleave, 16 moves/
   site -- twice per chain per group, amortized over m >= 128; measured
   negligible), run all m steps in the group buffer(s), unpack once; the
   B % 8 remainder takes the r4 per-volume path.  The split engine gets its
   OWN create()-time two-step gate (pack -> 2 group steps -> unpack vs
   per-volume execute + exact scalar map, tol 1e-12) and is dropped -- never
   shipped -- on any disagreement; wisdom stores only after it passes.
7. **R=2 fused specialization**: no y double-buffer (peak live 2n instead of
   2n + 2M); same-core pairs win 3/3 at L=10, outputs bit-identical.

### Measured on the node (a80n0, tryout leased cores, graded chain, min; MKL 2022 same window)

| case | gen_r4 | gen_r5 | delta | MKL | vs MKL | picked (race) |
|---|---|---|---|---|---|---|
| L=10  B=64 m=1000 | 3.49  | **1.43** (same-core floor 1.41) | **-59%** | 4.56  | **3.2x** | c2(d5)@s1 |
| L=12  B=64 m=600  | 5.18  | **2.47**  | **-52%** | 7.74  | **3.1x** | c3(d4)@s1 |
| L=15  B=32 m=600  | 11.79 | **5.65**  | **-52%** | 16.48 | **2.9x** | c3(d5)@s1 |
| L=20  B=32 m=256  | 24.69 | **18.50** | -25% | 59.7  | **3.2x** | c4(d5)@s1 |
| L=25  B=16 m=256  | 60.2  | **40.6**  | -33% | 121.6 | **3.0x** | c5(d5)@s1 |
| L=27  B=16 m=200  | 86.5  | **78.8**  | -9%  | 145.1 | 1.8x | c3(c3(d3))@s2 |
| L=31  B=16 m=140  | 205.6 | **141.1** | -31% | 849.6 | **6.0x** | d31@s3 |
| L=32  B=8  m=250  | 129.7 | 130.5     | 0    | 183.7 | 1.4x | c4(d8) pv (race rejects the group's L3 set) |
| L=40  B=8  m=128  | 283   | 285.9     | 0    | 403.8 | 1.4x | pv |
| L=50  B=4  m=128  | 639.3 | 650.8     | 0 (window) | 974.8 | 1.5x | c5(c2(d5)) pv (B < 8: no group) |
| L=100 B=1  m=64   | 5528  | 5953 (MKL +3.7% same window) | ~0 | 8046 | 1.4x | c5(c4(d5)) pv |

B=1 (per-volume path, group never engages): 3.96 (10; MKL B=1 4.93), 5.89
(12), 11.68 (15), 68.0 (25), 203.5 (31).  The B=1 cells are unchanged code;
the 10-15% wobble vs r4's B=1 reads is the known short-unit core-ramp.

Gates, final binary, all on the node: single call 2.9-4.7e-16 at all 11
cases (tol 1e-12); two-step m=2 gate 9.2e-16 / 1.52e-15 / 1.60e-15 /
1.73e-15 at 10/25/27/31 (tol 3e-14, 17-30x margin); graded map-chains
4.88e-14 (12, anchor 3.89e-14), 3.10e-14 (27, anchor 2.57e-14), 2.60e-14
(31, anchor 2.31e-14), tol 1e-10; MIXED batch group+remainder chains PASS at
B=12 (12, 31) and B=9 (15); full local sweep L=2..128 vs numpy: ALL 127
PASS; graded chains bit-identical across independent node processes at 12
(@s1), 27 (@s2, ping-pong parity) and 31 (@s3); non-AVX512 build (AVX2
login host) PASSES end-to-end incl. chain at L=9 B=12; round-6-style draws
24/45/63 at B=8 all plan, race and pass (setup <= 0.15 s).  Setup: cold
0.001-0.147 s (60 s budget), warm wisdom ~2-4 ms (50 ms budget).  Round
end: all gen_planner keys stripped from results/wisdom_a80n0.json under the
flock (the sweep alone had stored ~127 of them) -- the monitor cold-races
in its quiet window; absent entries are deliberate.

### What did NOT work / went wrong, with the numbers

- **Fused map at L=10: +5%** (1.40 vs 1.45-1.49 same-core).  The group is
  L2-resident there, so the deleted sweep was cheap, and the ladder's temps
  push the n=10 codelet into spills.  Fused map is a RESIDENCY play, not a
  free win -- hence the L^3 > 1728 gate (12 measured a wash, 15 wins -4%).
- **Two-pass @s2 at 25: 80 vs 46 us** for the spilling fused @s1 -- extra
  volume sweeps lose to register pressure at L2/L3-boundary sizes.  The
  race sees both; @s2 exists for sizes the fused form cannot reach (27, and
  round-6 composites like r*fused(<=25)).
- **The one-key wisdom CLOBBER bug**: a B=1 create() at an already-raced L
  re-raced (no group arms at B=1) and overwrote the batched @s1 pick; the
  next batched run silently ran pv.  Fixed by the /g8 regime key.  Anyone
  adding mode tags to wisdom values: key the REGIME too.
- **A 98-vs-86 "regression" at 27 that was a hot core state**: pinned
  same-core pairs read pv 86.5/86.7 vs @s2 86.8/107.6 -- a tie plus a
  mid-state window, exactly gen_batchlane r4's confound.  Every verdict in
  this record is from held-lease alternation; cross-invocation tryout pairs
  were treated as smoke only.
- **The R=2 despill did not zero the asm counter** (the MAP instantiation
  still spills ~20) but wins 3/3 same-core pairs at 10 anyway; kept on the
  measurement, not the audit.

### Borrowed this round, named

- **gen_batchlane / gen_pfa_small / gen_powp**: the whole SoA-8 split-lane
  batch-group concept (their class engines are the existence proof and the
  target line), the map-fused-in-final-stores placement (the ONE fusion
  geometry that keeps winning on this panel), and gen_layout's r2 proof
  that split-lane broadcast-FMA beats interleaved at equal algorithm.
  My contribution is making it GENERIC: any candidate tree with a hard-leaf
  root, raced against the per-volume engine per (host, L, batch-regime),
  with pack/remainder/gating handled once for everyone.
- **gen_race gen_r4**: the interleaved sample-major race -- my create()
  race was candidate-major, i.e. exactly the protocol their r4 receipt
  showed produces wisdom-pinned flip-flops.
- **gen_batchlane gen_r4 / gen_pfa_large gen_r4 lesson 1**: the same-core
  held-lease alternation A/B; it acquitted the false 27 regression and
  killed a wrong fused-map-everywhere default before it shipped.
- **gen_powp r2 protocol** (again): store-only-gate-passed wisdom, round-end
  key strip.

### Operation count (split group, per pencil of n rows x 8 volumes)

2n zmm loads + child/root butterfly FMA ops (identical per-point count to
the interleaved codelets) + (r-1)(m-1) cmul at 4 FMA-port ops each + 2n zmm
stores; shuffle-class ops: ZERO (asm-audited per codelet); masks: none.
Map: ~19 FMA-port ops per 8 SITES (half the interleaved rate), fused into
the final-axis stores for L > 12.  @s2 step: 2 fused volume sweeps (z-slab
axes 0+1, x-plane axis 2) instead of 3.  Pack/unpack: 16 scalar moves per
site, twice per chain per group.  Transposes per step: zero (the per-volume
path keeps its 4x4-block transposes; the group path has none).

### Notes for adopters

- `pln_s8_build(L, tree, lev)` / `pln_s8_step` / `pln_s8_pack/unpack` are
  in the GEN_PLANNER_LIB surface; batchlane/pfa_small-class engines can
  drive their own trees through them, and the fold (`pln_foldw`) accepts
  any odd n >= 11 (round-6 primes at B >= 8 get d_p for free).
- Wisdom values may now carry `@s1/@s2/@s3`; unknown tags force a re-race
  (forward-compatible).  Keys are regime-split: `.../L<n>` and
  `.../L<n>/g8`.

### What I would do next (ranked)

1. **20/25 arithmetic**: pfa_small/powp still lead 1.4x there with
   hand-derived real-factor codelets (powp's ~218-op lines vs my ~500 at
   27); a Winograd-style split DFT9/DFT25 module inside the fused codelet
   is the next arithmetic-level lever the generic engine can host.
2. **B=1 lane-spatial engine** -- the whole panel's standing hole (my B=1
   at 10 is 2.8x the batched cell); the ice L6_pfa shape, build once,
   everyone adopts.  Round 6 draws unknown batches.
3. **Half-group G=4 (ymm lanes or 2-site zmm) for B=4** -- L=50 is the one
   batched scored cell the group engine cannot touch.
4. **Large L unchanged since r4** (100: 1.4x MKL): gen_pfa_large's PMU
   attribution advice stands; the split engine does not reach B=1.
5. **xarch**: the split layer is AVX-512F+DQ-guarded with the pv fallback;
   the fusemap boundary (L^3 > 1728) is an ICX L2 number -- CLX (1 MB L2)
   and SPR (2 MB) should re-race it via wisdom, which the /g8 keys now do
   by construction.

## Round gen_r6 -- the map leaves its own pass everywhere, and @s2 loses its ping-pong

### The round in one line

Three changes, all mid/large-L: the per-volume chain's separate map pass is
gone (fused into the transpose-out exit at 12 < L <= 80, pair-packed ladder
elsewhere), and a fourth split-group level @s4 runs the two-pass CT as ONE
in-place volume sweep per axis through an L1 staging block -- 27 drops 24%,
32/40/50 drop 13-17%, 100 drops ~6%, small L and 31 unchanged.

### What changed (all in impl/gen_planner.c; pln_* public surface unchanged)

1. **Pair-packed map ladder in pln_map_span** (ADOPTED from gen_pfa_large
   gen_r5's map_step_pair, near verbatim): the interleaved ladder ran its
   ~19 ops on pair-DUPLICATED |z|^2; two vectors' 8 distinct |z|^2 now pack
   into one zmm (2 permutex2var), ONE rcp14-form ladder serves both, and the
   scales unpack pair-duplicated (2 permutexvar).  BIT-IDENTICAL per element
   (the packed lane sums the same two squares in the same order; everything
   else is elementwise) -- verified by cmp against the r5 binary through a
   full graded chain at L=100.  This is what the unfused path (L <= 12,
   L > 80, remainders, fallback) runs.
2. **Map fused into the pv transpose-out exit** (ADOPTED shape: gen_layout
   gen_r5's gl_map8 register-exit fusion -- pair-compressed |z|^2, rsqrt14 +
   2NR for the magnitude, ONE exact vdivpd per 8 complex; the exit is
   shuffle-bound and the divider idle, exactly gen_batchlane r5's
   "div-vs-rcp is a property of the surrounding codelet").  The chain step's
   axis-2 values already pass through pln_transpose_out; adding c and
   mapping them in registers deletes the separate per-plane map pass (plane
   read + c read + plane write) entirely.  pln_p3d_plane takes an optional
   c pointer; pln_p3d_exec never maps.  Gate: 12 < L <= 80, measured (see
   dead ends for the 100 story).  GEN_PLANNER_PVFUSE=0/1 overrides for A/B.
3. **Split-group level @s4: staged single-sweep in-place two-pass CT** (same
   trees as @s2: r hard, child hard leaf or fused CT, n > PLN_FUSEMAX).
   Blocks of P pencils (P = 4 for n <= 40, 2 to 64, else 1) stage through an
   L1 buffer: pass 1 (child DFT_m, undoing the decimation) reads the volume
   rows ONCE into the stage, pass 2 (twiddle-fused leaf_r) reads the stage
   and writes the volume rows ONCE, map fused in the final-axis stores.
   No ping-pong buffer (G2 not allocated for @s4 picks), each axis touches
   the volume once instead of @s2's twice; axes 0+1 stay slab-fused.  Stage
   row stride 16P + 8 doubles = an odd number of cache lines, so pass 2's
   m*srs stride never lands 4K-uniform (the ice odd-line rule, applied at
   plan birth).  Enabling change: pw_leaf / pln_fusedw grew separate in/out
   pencil strides (ics/ocs); all pre-r6 call sites pass ics == ocs, adopters
   of the pln_s8_* surface are unaffected (gen_race.c compiles unmodified).
   @s4 enters the race at batch >= 8 (arm cap 4 -> 6), wisdom tag @s4
   (unknown-tag re-race guard already covers forward compat).

### Measured on the node (a80n0 core 3, held lease, INTERLEAVED r5-binary-vs-
### r6-binary adjacent pairs, 3 pairs each, min per rep; graded chains)

| case | r5 same-window (3 reps) | gen_r6 (3 reps) | delta | pick |
|---|---|---|---|---|
| L=27  B=16 m=200 | 79.1-79.4 | **60.2-60.3** | **-24%** | c3(c3(d3))@s4 (NEW) |
| L=31  B=16 m=140 | 142.2-143.1 | **140.4-141.8** | tie | d31@s3 (unchanged path) |
| L=32  B=8  m=250 | 129.2-131.7 | **108.3-110.0** | **-17%** | c4(d8) pv, fused exit |
| L=40  B=8  m=128 | 281.5-285.2 | **239.2-243.6** | **-15%** | c4(c2(d5)) pv (race found a NEW tree under the fused exit) |
| L=50  B=4  m=128 | 645.9-678.9 | **565.0-568.3** | **-13%** | c5(c2(d5)) pv, fused exit |
| L=100 B=1  m=64  | 5554-5812 | **5173-5459** | ~-6% | c5(c4(d5)) pv, pair-packed map (unfused; outputs bit-identical to r5) |

Small L, full-battery reads (paths untouched, @s1 picks unchanged): 10:
1.414, 12: 2.484, 15: 5.593, 20: 18.300, 25: 40.570 -- all at the r5 board
within window noise.  B=1 chains (m=64): 4.50 (12), 110.4 (27), 190.0 (31),
574.2 (50) -- the 12/27/31 cells ride the pair-packed span, 50 the fused
exit.  Fused-exit A/B boundary data (same-core interleaved, GEN_PLANNER_
PVFUSE): 40 -6.5% (3/3), 50 -4.8% (3/3), 64 -10..19% (2/2), 80 -2% (2/2),
100 +8..10% (3/3 LOSS) -> gate 12 < L <= 80.

Gates, final binary, all on the node: full numpy sweep L=2..128 single call
ALL 127 PASS; two-step m=2 gate 9.2e-16 (10) / 1.33e-15 (20) / 1.52e-15
(25) / 1.73e-15 (31) / 1.54e-15 (32) / 1.83e-15 (40) / 2.25e-15 (50) /
2.70e-15 (100) vs tol 3e-14; graded map-chains PASS at all 11 acceptance
cases (3.10e-14 at 27 -- @s4 is op-identical to @s2, the staged pass
reorders nothing -- 2.60e-14 at 31, 3.35e-14 at 32, 2.78e-14 at 40,
3.86e-14 at 50, 2.99e-14 at 100, tol 1e-10); mixed batch B=12 at 27
(group-of-8 @s4 + 4-volume pv remainder) PASS; chain outputs bit-identical
across independent processes at 27, and the wisdom-pinned second process is
bit-identical with warm setup 15 ms (budget 50).  AVX2 wallaby build (no
AVX-512: fusemap forced 0, scalar exits) PASSes singles + m=20 chain at
L=9 B=12 and L=14.  Round-6 drill at B=8, all through the ordinary race:
45/48/54/63/96/104/127 all plan, pass single + m=8 map-chain, setup
0.04-3.2 s (worst L=96) vs the 60 s budget.  Round end: gen_planner keys
stripped from results/wisdom_a80n0.json under the flock (1 key this time --
I ran the whole session under GEN_RACE_NO_WISDOM=1, which the protocol
should have been doing all along).

### What did NOT work / went wrong, with the numbers

- **The fused exit at L=100: +8-10%, 3/3 interleaved pairs** (5918-6055
  fused vs 5328-5628 unfused).  The blockwise (z0 outer, y0 inner) exit
  walk re-reads the c plane L/4 times per plane (160 KB x 25 at L=100) and
  the win from the deleted map pass drowns in the extra L2 traffic; at 80
  (102 KB plane) it still wins -2%.  gen_layout's r5 saw a wash at 100 on
  the same shape with NT stores; mine has none, and loses outright.  Gate
  12 < L <= 80, boundary measured at 80/100, 81..99 untested -- a round-6
  draw there gets the conservative side of an unmeasured boundary.
- **Growing pln_s8_step silently un-inlined pln_foldw and cost @s3 at
  L=31 ~5-9%** (r5 binary 140-142 vs first r6 build 147-163, outputs
  bit-identical).  gcc-11 had been constprop-cloning pln_foldw into map and
  no-map forms on its own; the new @s4 branch pushed pln_s8_step past the
  inlining budget and the generic form kept both store paths live inside
  the 16-accumulator E/O tile.  Fixed by making the MAP specialization
  explicit (pln_foldw_m0/_m1 wrappers over an always_inline body) -- the
  objdump symbol audit (gen_dense_prime r3's discipline) is what caught
  it.  LESSON, for everyone: after growing any dispatch function, diff the
  symbol table against the previous build; an optimizer courtesy you never
  asked for can be revoked without a warning.
- **@s4 wins ONLY at 27 among the scored group cells**: at 32/40 (and the
  drill's 45/48/54/63/96) the race kept pv -- above vol ~20k the group's
  16x working set loses to the pv volume-major chain's L2 residency no
  matter how few sweeps the group makes.  Expected boundary (gen_race r5
  recorded the same for @s1/@s2), now measured one level deeper.  Recorded
  so nobody builds @s5 expecting different economics.

### Borrowed this round, named

- **gen_pfa_large gen_r5**: map_step_pair -- the pair-packed ladder,
  adopted near verbatim into pln_map_span; also the "map is ~40% of step
  arithmetic" accounting that made it obviously first-priority.
- **gen_layout gen_r5**: the whole register-exit map-fusion shape
  (gl_map8: pair-compressed |z|^2 + one vdivpd per 8 complex), its
  "the money is the L2/L3 middle, not DRAM sizes" boundary doctrine
  (reproduced exactly: my win band is 15..80), and the warning that
  gl_map16-style two-pencil pairing loses in a fused exit (not attempted).
- **gen_batchlane gen_r5**: the div-vs-rcp codelet-locality lesson (my
  fused exit divs, my separate pass keeps rcp14), and (with
  gen_dense_prime r3/r5) the held-lease interleaved adjacent-pair protocol
  used for every verdict above, including acquitting the false L=31 window
  read and convicting the real inlining regression behind it.
- **gen_powp / gen_race gen_r5 protocol**: GEN_RACE_NO_WISDOM for the whole
  dev session + flock strip at round end; the ice odd-line pitch rule,
  re-applied to the @s4 stage stride.

### Operation count (deltas vs gen_r5)

pln_map_span: ~21 arith + 4 shuffles per PAIR of vectors (was 38 + 2).
Fused pv exit: transpose shuffles unchanged; + 4 c loads, ~14 FMA-class,
2 pack shuffles, 2 dup shuffles, 1 vdivpd per 8 complex, fused into
existing stores; DELETES per plane: L^2 state loads + L^2 c loads + L^2
state stores + the span-loop overhead.  @s4 per axis per pencil: n volume-
row loads + n volume-row stores (was 2n + 2n in @s2's two passes) + n
stage stores + n stage loads (L1-resident); FFT arithmetic identical to
@s2; G2 (16 * vol doubles) not allocated.

### What I would do next (ranked)

1. **B=1 lane-spatial split engine** -- six rounds on this list, still the
   panel's standing hole (my 27 B=1 is 1.8x the batched cell).  The fused
   exit narrowed pv's deficit but the structural answer is 8 spatial sites
   per zmm with an in-register axis-2 stage.  Round 6 draws unknown batches.
2. **L=100 structurally** (lit 11 Tier 2 two-axes-per-pass): the pair-packed
   map got the residue to ~5.2-5.5 ms vs gen_pfa_large's 4.5; the fused exit
   measurably does NOT transfer there.  Coordinate with gen_pfa_large /
   gen_powp before anyone burns the round -- three entries now queue the
   same idea.
3. **Race the fusemap axis per host**: the 12/80 boundaries are ICX numbers;
   CLX's 1 MB L2 and SPR's 2 MB will move both (gen_pow2 flags MAPDIV as a
   CLX flip candidate for the same reason).  One extra pv arm with the
   opposite fusemap at the boundary sizes would let wisdom decide.
4. **Winograd/real-factor DFT9 and DFT25 modules** inside the fused
   codelets: powp's 218-op hand lines vs my ~500 at 27 says the remaining
   1.16x at 27 is arithmetic, not scheduling.

## Round gen_r7 -- fused Good-Thomas codelets (PFA economics, hosted generically) and DFT7 joins the hard-leaf set

### The round in one line

A PFA node whose two coprime factors are both hard leaves and whose product
is <= 25 now runs as a register-resident TWIDDLE-FREE codelet (pv and pw
forms) -- the CRT permutations become compile-time index selection, so the
gather/scatter passes that made generic PFA lose in gen_r1 do not exist --
and 7 is a hard leaf: 10/12/15/20 drop 2-5%, 40 drops 2% (GT child), and
the 7-containing surprise-class sizes drop 27-80% (L=14: 5x).

### What changed (all in impl/gen_planner.c; pln_* public surface additive)

1. **Fused GT codelets** (`pln_gtv` / `pln_gtw` + wrappers for the 8
   canonical coprime hard pairs (2,3),(2,5),(2,7),(3,4),(3,5),(3,7),(3,8),
   (4,5) -> n = 6,10,14,12,15,21,24,20).  Ruritanian maps both sides:
   input n = (M n1 + R n2) % N, output k = (M k1 + R k2) % N, which makes
   the factorization twiddle-free with PLAIN child DFTs read/written at
   permuted frequency indices ((R k2) % M, (M k1) % R) -- all index
   arithmetic on compile-time constants, so it folds to immediates and the
   permutation costs ZERO instructions.  vs the fused CT on the same pair:
   the (R-1)(M-1) twiddle cmuls are deleted (-20..25% of pencil arithmetic
   at 10..20) and register pressure drops (no twiddle broadcasts).  The pw
   form has the R=2 no-double-buffer specialization and the same ics/ocs +
   fused-map surface as pln_fusedw, so fused-GT nodes serve as @s1 roots
   and as @s2/@s4 children; in pv they serve as executor roots and CT
   children (50 = c5(gt(d2,d5)), 100 = c5(gt(d4,d5))).  Enumeration: a
   PFA candidate with pln_gt_fused children costs the fused-CT model minus
   its 4(m-1)(r-1) twiddle term; the PLN_PFA name ("gt(dR,dM)") and the
   wisdom machinery are unchanged.
2. **DFT7 is a hard leaf** (scalar pair, pv pln_lv7 with HAS-twiddle flag,
   pv_dft7r, pw_dft7s, pw_l7; symmetric fold t_j = x_j+x_{7-j},
   u_j = x_j-x_{7-j}, X_k = E_k -+ iO_k, exact 20-digit constants): the r6
   surprise test ran d7 as an 8n^2 dense matrix inside L=21/44-class sizes
   and the addendum priced that hole at 2-3x.  New fused pairs (2,7),(7,2),
   (3,7),(7,3) instantiate in both CT and GT forms; leaf-7 CT covers
   28/35/42/49/56/63... through the ordinary planner.
3. **The DFT7 leaf dispatch entries are NOINLINE wrappers** (pln_lv7_nt /
   pln_lv7_tw) -- see dead-ends below for the +1..3% regression this fixed.
4. Race pool and everything else unchanged; setup stayed 0.001-0.245 s
   cold (60 s budget), wisdom warm path untouched.

### Measured on the node (a80n0 core 2, held lease slot 0, interleaved
### r6-binary-vs-r7-binary adjacent pairs, graded chains, min us/xform)

| case | r6ref same-window | gen_r7 | delta | pick |
|---|---|---|---|---|
| L=10  B=64 m=1000 | 1.395-1.594 | **1.337-1.343** | **-4.4%** (6/6) | gt(d2,d5)@s1 |
| L=12  B=64 m=600  | 2.467-2.517 | **2.341-2.368** | **-5.2%** (6/6) | gt(d3,d4)@s1 |
| L=15  B=32 m=600  | 5.547-5.783 | **5.525-5.647** | -1% (3/3 after pin; wash before) | gt(d3,d5)@s1 |
| L=20  B=32 m=256  | 18.90-19.85 | **18.31-18.82** | **-2..4%** | gt(d4,d5)@s1 |
| L=25  B=16 m=256  | 42.48-44.19 | 42.28-43.20 | tie/slight win | c5(d5)@s1 (unchanged) |
| L=27  B=16 m=200  | 64.31-64.43 | 64.09-64.38 | tie (3/4 by <=0.4%) | c3(c3(d3))@s4 (unchanged) |
| L=31  B=16 m=140  | 144.6-150.0 | 144.7-150.0 | tie | d31@s3 (unchanged) |
| L=32  B=8  m=250  | 108.6-109.2 | **107.4-108.2** | -1.4% (layout luck; same tree) | c4(d8) pv |
| L=40  B=8  m=128  | 248.3-250.0 | **243.5-245.3** | **-2%** (3/3) | c4(gt(d2,d5)) pv (NEW tree) |
| L=50  B=4  m=128  | 578.8-582.7 | **574.3-579.2** | -0.7% (2/2) | c5(gt(d2,d5)) pv |
| L=100 B=1  m=64   | 5212-5224 | **5168-5180** | -0.9% (2/2) | c5(gt(d4,d5)) pv |

Surprise-class sizes (the round's real payoff; same protocol):
L=14 B=8: 23.58 -> **4.71-5.42** (gt(d2,d7)@s1, ~**5x**; r6 ran c7(d2) with a
dense-7 radix); L=21 B=8: 38.9-39.2 -> **24.3-24.8** (c3(d7)@s1, **-38%**);
L=35 B=8: 254.7-255.6 -> **162.5-163.1** (c5(d7)@s4, **-36%**); L=63 B=2:
1832-1851 -> **1338-1342** (**-27%**, SAME tree c7(c3(d3)) -- the win is
purely dense-radix-7 -> hard-leaf-7 with fused twiddles).  B=1 (pv path):
10: 3.17 -> **2.94-2.98** (gt, -6.5%); 20: 18.9 -> **17.8-18.3** (gt, -4%).

Gates, final binary, all on the node: single call 2.8-4.6e-16 at all 11
acceptance cases + 14/21/35/63 (tol 1e-12); two-step m=2 gate
0.90-2.90e-15 everywhere (tol 3e-14, >10x margin); full graded chains
PASS at all 15 cases (e.g. 3.10e-14 at 27, 2.60e-14 at 31, 4.06e-14 at
100), all within ~2x of honest anchors (tol 1e-10); wisdom-pinned
cross-process chains bit-identical (cmp) at every case, scratch wisdom
file (GEN_RACE_WISDOM override), removed after; node sweep L=2..128
single-call vs numpy ALL 127 PASS (and the same sweep on wallaby/SPR);
non-AVX512 build (-march=x86-64-v2, scalar leaf7 + generic PFA paths)
passes m=6 chains at 7/10/14/21/27; GEN_PLANNER_LIB adoption mode and
gen_race's include both compile.  Shared results/wisdom_a80n0.json checked
clean of gen_planner keys (whole session under GEN_RACE_NO_WISDOM).

### What did NOT work / went wrong, with the numbers

- **Adding `case 7: pln_lv7(...)` inline to the two leaf dispatchers cost
  the 7-FREE tree c8(d5) at L=40 +1..3% (3/3 RACE=0 same-tree pairs:
  253.9/255.5/254.7 r6 vs 257.7/257.9/262.4 r7)** -- pure code-layout
  drift: the constprop clones of pln_leaf_apply/_tw grew 3198 -> 3867
  bytes and gcc rebalanced inlining across pln_xexec.  With the race on,
  this compounded: the pv arms' trials slowed enough that the race
  mis-picked c5(d8)@s4 at 40 (252.3 vs r6's pv 242-248, +3-4% on the
  cell).  FIX: the DFT7 dispatch entries are noinline wrappers
  (pln_lv7_nt/_tw) -- dispatcher clones return to r6 size (3116), the
  RACE=0 gap collapses to +0.2..0.5%, and the raced cell flips to a 3/3
  WIN at 243.5-245.3 picking c4(gt(d2,d5)).  This is gen_twiddle r5's
  case-bloat hazard + gen_pfa_large r6's pin-structurally lesson, now
  measured in this entry: a new leaf must enter hot dispatchers as a CALL,
  not a body.
- **nm -S symbol-size diff against the previous binary found it before
  the benchmarks did** (pln_leaf_apply.constprop 3198->3867 was visible
  immediately); the audit also showed pln_s8_ct2_set halving (2463->1206,
  pw leaves un-inlined) -- @s2 is not on any scored path and 27 (@s4)
  A/B'd clean, so that one was left alone.  Diff the symbol table after
  ANY dispatch growth; it is a 5-second test that localizes layout drift.
- **ssh sessions land in $HOME** (r2 lesson, bitten AGAIN by an inline
  ssh sweep -- two wasted node runs): every remote command block must
  start with the explicit cd.  My r7ab.sh/r7gates.sh scripts cd
  internally; inline ssh one-liners are where it keeps happening.
- GT at 15 was a wash BEFORE the noinline pin (5.55-5.63 vs 5.55-5.78
  mixed signs) and only a small win after (-1%): at (3,5) the y[15]+v[5]
  pw working set already spills, so the 8 deleted cmuls partly hide under
  spill traffic -- the GT gain shrinks as the pair's register pressure
  grows (10/12 spill-free: -4..5%; 20 spilling: -2..4%; 15: -1%).

### Borrowed this round, named

- **gen_pfa_small / gen_batchlane**: the entire premise -- their PFA/GT
  engines are why 10/12/15/20 belonged to them; this round hosts their
  twiddle-free economics in the generic engine (my r1 record already
  noted "a class engine with fused permutations has the opposite
  economics"; literature 11 Tier 1's GT-as-vectorization-first is the
  citation).  Gap to gen_batchlane at 10 narrows from 1.23x to ~1.16x.
- **gen_pfa_large gen_r6**: DFTODDM's direct symmetric odd-N form is the
  same fold my DFT7 uses (independently the ice-era standard); more
  importantly their "pin hot callsites structurally, diff nm -S per
  function" discipline is what turned the L=40 incident from a shipped
  regression into a one-hour fix.
- **gen_twiddle gen_r5** (via gen_pow2 r6's citation): the case-bloat /
  code-layout hazard -- new code moves sizes that never execute it.
- **gen_batchlane gen_r4 / the panel standard**: held-lease same-core
  interleaved adjacent-pair A/B for every verdict above.
- **The surprise-test addendum** (monitor, r6): the explicit pricing of
  the missing 7-point module is what put DFT7 at the top of this round.

### Operation count (deltas vs gen_r6)

Fused GT (R,M), per pencil: n row loads + R x DFT_M + M x DFT_R + n row
stores, ZERO twiddle ops, ZERO permutation instructions (index selection
folds at compile time) -- vs fused CT: -(R-1)(M-1) cmuls (pv: 3 FMA-class
+ 2 broadcast loads each; pw: 4 FMA-port ops + 2 broadcasts).  At n=20
that is -48 of ~192 FMA-port pencil ops (-25%).  DFT7 leaf (pv): 7 masked
loads, 6 add/sub fold, 3 adds (X0), 3 x (6 FMA + 1 mul + 2 shuffle-class
+-i combines), 7 masked stores, ~40 FMA-class total -- vs the old d7
dense: 8n^2-model matrix with arena staging (the register-tiled kernel's
~2n^2 real FMA + staging + per-node dispatch).  Everything else unchanged.

### What I would do next (ranked)

1. **B=1 lane-spatial split engine** -- seventh round on this list; the
   B=1 cells still run 1.5-2x behind their batched selves and round-8
   draws unknown batches.
2. **Winograd/real-factor DFT9/DFT25 modules** in the fused codelets
   (unchanged from r6; 25/27 are now my largest arithmetic gaps to powp).
3. **GT pairs above the 25-cap** ((4,7)=28, (5,7)=35, (5,8)=40 as
   two-level @s4-style staged GT: child pass + root pass, still
   twiddle-free): 35 already dropped 36% just from leaf-7; a staged GT
   would delete its 24 root cmuls too.  Needs the ct1_set machinery
   generalized from CT to GT maps -- moderate.
4. **Race the fusemap axis per host** (r6 item, still open; CLX/SPR
   advisories will move the 12/80 boundaries).

## Round gen_r8 -- the z-slab hypothesis refuted with numbers; the lifted DFT5 ships pair-scoped

### The round in one line

The round's big idea (fusing @s1/@s3 axes 0+1 per z-slab to delete one of the
step's three volume sweeps) measured a 7-12% LOSS at every size it engages and
was reverted -- the refutation is the deliverable; what ships is
gen_batchlane's lifted DFT5 v-pair, scoped to the ONE codelet pair where it
wins on this engine: gt(2,5) (-1.2% at L=10 batched, -0.4% at L=50, 4/4
pairs), with every other DFT5 site left bit-identical to gen_r7.

### What was tried FIRST and REFUTED: z-slab fusion of @s1/@s3 axes 0+1

My own r5 @s2 slab merge (axes 0+1 back to back per fixed-z slab, -8.5% at
27) had never been applied to levels 1 and 3, whose steps still make THREE
volume sweeps (axis 0 whole-volume; then axes 1+2 per x-plane).  The reorder
is bit-identical (verified by cmp through graded chains at 10/20/25/31
against the r7 binary with picks pinned via a scratch wisdom file) and the
traffic model said -10-15% at 20/25/31 where the group (state+c = 256 L^3 B)
leaves the 1.25 MB L2.  Held-lease interleaved pairs on the node
(a80n0 core 2, 3 pairs, min of --samples 4) said otherwise:

| case | r7ref | slabbed | verdict |
|---|---|---|---|
| L=25 B=16 m=256 | 42.96-44.38 | 48.28-49.37 | +10-12% LOSS 3/3 |
| L=31 B=16 m=140 | 144.9-146.8 | 156.2-156.8 | +7% LOSS 3/3 |
| L=20 B=32 m=256 | 19.23-19.39 | 19.23-20.63 | wash-to-loss |
| L=15 B=32 m=600 (forced on) | 5.50-5.55 | 5.64-5.76 | +2% LOSS |

Mechanism, recorded so nobody re-derives: @s1/@s3's axis-0 pass walks its
pencils CONTIGUOUSLY (pcs=16 site-major, i.e. L parallel perfectly-sequential
streams -- ideal L2-streamer food); the slab replaces that with pencil walks
strided 128*L bytes (3200 B at L=25), past the ICX streamer's ~2 KB stride
reach, so every line becomes a demand miss.  @s2's slab win at 27 does NOT
transfer because @s2's two-pass structure never had a contiguous-stream pass
to lose.  The deleted volume sweep was worth less than the prefetch it cost.
The code is reverted out (not a knob) -- a comment in pln_s8_build carries
the numbers.  Corollary for anyone eyeing the same traffic model: the win
band for pass-fusion reorders on this engine is "only where the baseline is
already stride-blind", and @s1/@s3 baselines are not.

### What ships: lifted DFT5 v-pair, PAIR-SCOPED (adopted from gen_batchlane gen_r7)

sin(2pi/5)/sin(pi/5) = 2cos(pi/5) = PHI exactly, so the v-pair
v1 = S1*sa + S2*sb, v2 = S2*sa - S1*sb factors through u = sa - PHI*sb as
v2 = S2*u, v1 = S1*u + KL5*sb (KL5 = 1.25/sin(pi/5); their exact-to-last-bit
constants taken verbatim).  One fewer live temp per DFT5.  Measured on MY
codelets the verdict is pair-specific (4-pair held-lease batteries, r8nl =
same-layout-no-lift control arm so layout drift is separated from arithmetic):

- FULL lift everywhere (first attempt): 10: -1.0% 4/4 WIN; 15 (gt(3,5)):
  +2-3% 4/4 LOSS; 20 (gt(4,5)) and 25 (c5(d5)): wash; 100 (gt(4,5) child):
  +0.6% 4/4 slight loss.  Same boundary class as gen_powp's r7 finding
  (lift loses on their slot engine, wins on batchlane's): the win needs
  surrounding ILP AND a codelet that is not already spill-free-saturated;
  at (3,5) the pw y[15]+v[5] working set spills and the lift's extra serial
  link through u costs more than the freed register.
- SHIPPED form: the L5 flag threaded compile-time through pv_dftNr_i /
  pw_dftNs_i, requested ONLY by the (2,5) GT codelets (pln_gtv / pln_gtw,
  both the R=2 specialization and pv fused-GT-as-CT-child).  Final node
  battery (4 pairs each, r7ref control first):
  10 B=64: 1.320-1.344 vs 1.339-1.349 (-1.2%, 3/4);
  50 B=4: 565.6-566.8 vs 566.7-569.1 (-0.4%, 4/4 -- via the gt(d2,d5) child
  of c5);  10 B=1: wash;  40 (gt(d2,d5) child): wash;
  15 / 27 / 100: parity (bit-identical paths).
  Every size whose picked tree lacks gt(2,5) produces BIT-IDENTICAL chains
  to gen_r7 (cmp-verified at 15/20/25/27/31 with pinned picks); nm -S symbol
  audit: exactly ONE function changed size (pw_g_2_5, 2909 -> 2919 B), zero
  drift elsewhere -- the r7 case-bloat lesson applied preventively.
  PLN_LIFT5=0 kills the lift everywhere for cross-arch races.

### What went WRONG mid-round and the lesson

My first lift edit hoisted the dd = -S2*u computation ABOVE the v[1]/v[4]
store pair (to share the #if block).  That cost +1-2% at 15 in the
NO-LIFT control arm -- identical arithmetic, pure schedule/live-range drift
in the already-spilling (3,5) codelet -- which briefly made the lift look
guilty at sizes it never touched.  Restoring the original store interleaving
(dd computed after cc, where r7 had it) collapsed the control arm back to
parity.  Lesson: when adding a #if variant to a register-tight codelet,
keep the UNCHANGED branch's statement order byte-for-byte -- the compiler's
schedule is part of the tuned artifact, and an A/B without a same-layout
control arm would have shipped the confound.

### Gates (final binary, node a80n0 leased core, scratch wisdom)

All 11 acceptance cases at their exact graded m: single call 2.96-4.63e-16
(tol 1e-12); two-step m=2 gate 0.94e-15 (10) .. 2.90e-15 (100) vs tol 3e-14
(>10x margin everywhere -- the lift moved 10's from 9.2e-16 to 9.4e-16, i.e.
nothing); full graded chains PASS at all 11 (1.32e-13 at 10 vs anchor
1.08e-13; 2.60e-14 at 31 vs 2.31e-14; 4.06e-14 at 100 vs 2.42e-14 -- all
within ~2x of the honest anchors, tol 1e-10); chain outputs bit-repeatable
across independent runs at every case (wisdom-pinned); mixed batch B=12 at
L=10 (group-of-8 lift path + 4-volume pv lift path) PASS at 9.3e-15; full
node sweep L=2..128 single call vs numpy: ALL 127 PASS, zero failures.
Non-AVX512 build (-march=x86-64-v2) passes chains at 10/14 on wallaby;
GEN_PLANNER_LIB adoption mode compiles and runs; gen_race.c compiles against
the new file unmodified.  Setup unchanged (no race-pool or wisdom-format
changes).  Round end: gen_planner keys stripped from results/wisdom_a80n0.json
under the flock (the r7 leaderboard run had left 11), scratch wisdom only all
session.

### Borrowed this round, named

- **gen_batchlane gen_r7**: the lifted DFT5 v-pair, constants verbatim
  (PHI, KL5).  Their measured -0.8..-1.0% transfers to my gt(2,5) codelets
  only; the pair-scoped shipping form and the (3,5)-loses boundary are this
  round's addition to the shared picture (gen_powp r7 recorded the
  engine-level boundary; this is the pair-level one).
- **gen_powp gen_r7**: the build-verification trap (never pipe gcc; capture
  stderr and echo $? -- followed all round) and the precedent of shipping a
  lift as a knob when the verdict is shape-specific.
- **gen_pfa_large gen_r6/r7**: nm -S per-function symbol diff as the layout
  arbiter; their "rank evidence does not transfer, race machinery does"
  doctrine is the same lesson my slab refutation re-teaches for traffic
  models.
- **The panel-standard held-lease interleaved adjacent-pair protocol** for
  every verdict above, including the r8nl same-layout control arm (an
  extension other entries may want: it is what caught the store-interleaving
  confound).

### Operation count (deltas vs gen_r7)

gt(2,5) codelets only (pv and pw forms): per DFT5, the v-pair becomes
u (1 FMA) + v2 (1 mul) + v1 (1 mul + 1 FMA) = same 4-op count as r7 but one
fewer live temp (t3 dies at u) and the b/d pair depends on one shared u
instead of both t3/t4 -- the gain is register/schedule, not raw ops (the
measured -1.2% at 10 against a +10-byte body).  All other codelets,
transposes, maps, twiddles: byte-identical to gen_r7.

### What I would do next (ranked)

1. **B=1 lane-spatial split engine** -- eighth round on this list.  The B=1
   cells still run 1.5-2x behind batched; nobody on the panel has built it.
2. **Winograd/real-factor DFT9/DFT25 modules** in the fused codelets --
   the 25/27 arithmetic gap to gen_powp (1.30-1.36x) is real arithmetic,
   not scheduling; powp's r7 3-shear twiddles will not transfer to my
   runtime-table codelets (each shear's half-turn reduction changes OPCODES,
   not constants, so a table-driven form needs a per-twiddle branch in the
   inner loop -- analyzed this round, declined by arithmetic; a Winograd
   module sidesteps twiddles entirely).
3. **uiCA env is broken** (ModuleNotFoundError: instrData -- needs its
   setup step run); llvm-mca/OSACA untried this round.  Whoever fixes the
   uiCA install first should note it in their record; the tools would have
   priced the slab reorder's port picture before I spent three lease
   sessions on it.
4. **Protect**: the engine is now r7 + one 10-byte codelet delta; any r9
   work should start from the standings, not from this file's queue alone.

## Round gen_r10 -- three refutations, the r9 machinery node-proven, and a banked tile playoff

### First, the missing gen_r9 record (written here because the r9 session never appended it)

The r9 session ran entirely node-blind (both Ice Lake nodes held by other
users all round; NOTICE-directed model-side development) and its section was
never written -- its final log entry is a status update from inside the
wait.  What it CODED, wallaby-validated, and shipped in impl_9:

1. **Noise-gated create() race** (PMU-audit avenue 1): the min-of-3
   interleaved race extends itself (contenders within 10% of the winner
   only, up to 6 extra rounds) whenever the leader's margin sits inside the
   pair's own measured trial noise; at the cap the deterministic
   2%-simplest-first hysteresis breaks the tie instead of window luck.
   Wallaby proof: 5/5 identical cold picks at 25/50/100.
2. **c-line custody at L >= 89** (adopted from gen_pfa_large's c-bypass):
   clflushopt each x-plane's consumed c lines in the separate-map path so
   the read-once c stream stops evicting m-step-resident state; sysfs-L3
   default gate plus a custody playoff in the race banking @f0/@f1 through
   wisdom.  Cache op only, outputs bit-identical.  NEVER MEASURED on the
   node in r9 -- see below for how that went.

The r9 leaderboard (first board on a81n2, the replacement Ice Lake node)
scored the r9 binary at 10: 1.320, 25: 39.589, 31: 138.384, 32: 105.865,
40: 240.053, 50: 563.424, 100: 4976.4 us -- and its own quiet-window race
banked `c5(gt(d4,d5))@f0` at 100: the custody playoff had already measured
custody OFF as the winner on the very node the sysfs heuristic predicted a
win.  That was the tell for this round.

### gen_r10: what was tried, in order, with the numbers

**All verdicts below: a81n2 (Ice Lake, our hold), held-lease same-core
interleaved adjacent pairs, min of --samples 4, shared scratch wisdom so
both arms run the same tree; outputs cmp'd bit-identical in every A/B.**

1. **gen_pow2 r9's ZST transpose stores -- REFUTED ON ICL (+2..6%).**
   Their extract-to-memory trick (store the 4x4 transpose's combine rows as
   256-bit halves: ymm mov low + vextractf64x4-to-mem high, both
   store-port-only, deleting 4 vshuff64x2/block, front-end neutral,
   bit-identical bytes) applied to pln_tr4x4.  Node: 32: ctl 113.1-116.5 vs
   zst 117.3-117.9 (3/3 LOSS); 40: 240.6-244.5 vs 245.5-249.6 (3/3); 100:
   5534-5761 vs 5860-5861 (3/3).  Mechanism I believe: their custody-row
   destinations are 64B-aligned; my transpose destinations are complex-
   granular strided rows (16B-aligned), so half the split 256-bit stores
   line-cross where the one zmm store was a single access.  Ships as
   PLN_TRZST=0 default (knob kept as an xarch arm).  NOTE FOR gen_pow2:
   their r9 ZST shipped default-ON on SPR evidence with the ICL check
   pending -- this is an adjacent-kernel ICL data point that disagrees;
   their own node re-race should decide their default.
2. **The r9 custody -- REFUTED (+6..11% at L=100, 3/3).**  GEN_PLANNER_
   CFLUSH=0: 5565/5606/5612 vs =1: 5919/5926/6224.  On the exact 24 MB-L3
   node the r9 sysfs heuristic targeted, the ~2.5K clflushopt uops per
   plane cost more than the 16 MB read-once c stream's L3 pressure ever
   did.  Default now OFF everywhere; eligibility + playoff kept so a host
   where it wins can still bank @f1.  (The r9 scoring race had already
   picked @f0 -- the playoff machinery worked; the default was wrong.)
3. **Fused exit at L > 80, walk-order repaired -- STILL LOSES (0.2-2.2%,
   3/3).**  My r6 dead-end blamed the fused exit's loss at 100 on c-plane
   re-reads; re-analysis says the z0-outer walk STRIDES the two DRAM
   streams (output rows + c, 64 B chunks at 16L-byte steps) while
   streaming the L2-resident P scratch -- backwards.  Built the y0-outer
   exit (pln_transpose_out_map_yo, bit-identical to the z0 form): it
   repairs most of the pathology (z0 5722-6018 -> yo 5146-5589 at 100) --
   the r6 mechanism story was wrong, the walk order was the bulk of it --
   but the separate pair-packed map still wins 3/3 (5073/5414/5466 vs
   5146/5424/5589): two sequential full-plane sweeps ride the prefetcher
   at effectively no cost, and the exit's strided P reads + vdivpd beat
   the 32 MB of traffic fusion deletes.  Default unchanged (unfused at
   L > 80); PVFUSE=2 keeps the yo arm reachable for xarch races.
4. **TILE PLAYOFF -- the round's one shipped win (-2.5..4% at 40, -2.6..3%
   at 50).**  gen_race's r9 scoring keys raced MY row-width knob on its own
   composition and banked t16 at 40/50 (margins 2.3/2.8%) -- my race never
   raced the tile axis (fixed 32 since r3).  Node A/B confirmed: 40: t16
   232.9-236.6 vs t32 240.6-243.1 (3/3); 50: 567.6-570.2 vs 583.6-585.9
   (3/3); t16 LOSES at 32 (+3.5..5.9%) and 100 (+4..13%) -- per-(host, L)
   truth, exactly what playoff + wisdom is for.  Implementation: after the
   main race, a pv winner is rebuilt at the alternate width (pln_tile_req
   beats the default, never the GEN_PLANNER_TILE env pin) and the chain
   step raced interleaved, min-of-12-rep cap; the verdict rides the wisdom
   value as @t16.  Wisdom values now COMPOSE tags (name@f0@t16); the
   parser handles any order, unknown tags still force a re-race (old
   binaries reading @t re-race -- forward compatible).  Tile is
   bit-neutral (t16-vs-t32 chains cmp-identical; column chunking never
   reorders arithmetic within a pencil), so all gate digits carry.
5. **Adoption gate hardening after a caught determinism regression.**  The
   playoff's first form (adopt on capped min comparison) flipped 2-of-5
   cold picks at 50 -- a 2.6% truth at a 2% hysteresis edge is a coin flip
   under contention, gen_pfa_large's r9 lesson replayed in my own code.
   Shipped form: adoption requires the gate to SETTLE (gap >= max(2%,
   2x the pair's trial noise)); a capped-out noisy playoff keeps the
   default deterministically.  Honest boundary, recorded plainly: in a
   CONTENDED window a cold create at 40/50 may keep t32 (settled-vs-capped
   varies with neighbor load; 100 and all pre-playoff cells stay 5/5);
   the deployment path is unaffected -- the quiet scoring window settles,
   banks @t16, and wisdom pins every later create() (gen_powp r9's
   banking-pins-the-verdict doctrine, adopted with the mechanism).

### Node validation of the whole engine (final binary, a81n2)

- 5x cold create determinism: 5/5 identical picks AND bit-identical
  outputs at 25 (c5(d5)@s1), 50 (c5(gt(d2,d5))), 100 (c5(gt(d4,d5))@f0)
  pre-playoff; with the playoff live: 5/5 at 100, 4/5 at 40 where the
  MAIN race flipped trees once under 4-agent dev contention (near-tied
  arm set, r4-r8 known; the noise-gated extension caps at 6 rounds).
- Two-step m=2 gate: 0.94e-15 (10) .. 2.90e-15 (100) at 10/20/25/27/31/
  32/40/50/100, tol 3e-14, >10x margin everywhere.
- Graded chains: ALL 11 acceptance cases PASS, drift 1.06-2.0x honest
  anchors (e.g. 2.60e-14 at 31 vs anchor 2.31e-14; 4.06e-14 at 100 vs
  2.42e-14).  Chain times in the (contended) dev windows: 1.51 (10),
  2.34 (12), 5.62 (15), 18.17 (20), 44.6 (25), 64.2 (27), 145.6 (31),
  107.5 (32), 233.0 (40 @t16), 573.6 (50 @t16), 5126-5501 (100 @f0).
- Full numpy sweep L=2..128 single call: ALL 127 PASS (run twice, before
  and after the playoff landed).
- Mixed batch B=12 at 27 (@s4 + pv remainder) and B=9 at 15: PASS.
- r9-vs-r10 chain outputs BIT-IDENTICAL at 25/27/100 with pinned picks
  (the true impl_9 source rebuilt as control -- gen_rader's r9 symlink
  trap respected: impl/ is live, snapshots are the control).
- x86-64-v2 (no AVX-512) build passes chains at 10/14 B=12; GEN_PLANNER_
  LIB adoption mode and gen_race.c compile unmodified.
- Wisdom hygiene: whole session on scratch files / GEN_RACE_NO_WISDOM; my
  11 stale r9-shape keys stripped from results/wisdom_a81n2.json under
  the flock (45 foreign entries preserved, file re-validated) BECAUSE the
  race shape changed -- a warm hit on a pre-playoff value would silently
  skip the tile playoff forever.  The r9+ gen_race parser is layout-
  independent (their r9 note), so the python round-trip is safe.

### Operation count (deltas vs gen_r8/r9 arithmetic: NONE in any shipped path)

Every shipped execute/chain path is bit-identical to gen_r8 (cmp-proven
through graded chains).  The refuted arms live behind PLN_TRZST=1 /
GEN_PLANNER_CFLUSH=1 / PVFUSE=1,2 only.  The tile playoff changes plan-time
work: +1 pln_p3d_build + 8-24 interleaved chain steps at pv-winning sizes
(cold setup at 100 measured 0.35 s incl. all playoffs; budget 60 s).

### Borrowed this round, named

- **gen_pow2 gen_r9**: the ZST extract-to-memory store idea, tried
  faithfully (their asm-audit method included) -- refuted on ICL for my
  kernel; the negative plus the alignment mechanism is my return gift.
- **gen_race gen_r9 scoring keys**: the tile9/fm9 axis data that pointed
  at the t16 win -- adoption-by-reading-their-wisdom, a new channel.
- **gen_pfa_large gen_r9**: the determinism lesson (in-window spread
  cannot gate a hysteresis-edge verdict; needs floors/settling + evidence
  the challenger has not seen) -- applied after reproducing their exact
  failure; also their portcal3 result (ymm co-issue steals 512-bit slots
  1:1 on SPR) which struck avenue 4 off my list for free.
- **gen_powp gen_r9**: the banking-pins-noisy-truths doctrine and the
  honest 4/5-is-correct framing, both load-bearing in item 5.
- **gen_rader gen_r9**: the impl-symlink control trap -- my r9-vs-r10
  bit-identity control was built from impl_9/, not impl/.
- **gen_batchlane r1/r9**: the slurm PATH shim (reserve.sh --status reads
  dead without it -- bit me again at session start) and the a81n2 no-perf
  status (time-only A/Bs this round, PMU dashboard still blocked).

### What I would do next (ranked)

1. **Winograd/real-factor DFT9/DFT25 modules** in the fused codelets --
   the 25/27 arithmetic gap to gen_powp (1.27/1.37x on the r9 board) has
   been the top unspent arithmetic item for four rounds.
2. **B=1 lane-spatial split engine** -- ninth round on this list; B=1
   cells still run 1.5-2x behind batched and nobody on the panel has
   built it.
3. **The L=31/32 class gaps** (1.6-1.9x behind gen_rader/gen_pow2) are
   hand-codelet arithmetic, not scheduling -- adopt their modules as @s
   levels only if their records show transferable kernels; do not retune
   scheduling there, the PMU audit says those engines are near the port
   floor.
4. **xarch**: @t16/@f verdicts re-race per host by construction; check
   the next CLX/SPR advisory picks them up (CLX's 1 MB L2 should flip
   the tile crossover hard).
