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
