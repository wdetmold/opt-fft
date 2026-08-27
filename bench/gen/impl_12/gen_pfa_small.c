/* gen_pfa_small.c -- PFA of coprime pairs, small: L = 10, 12, 15, 20,
 * plus (round 3, class duty) any coprime P*Q with modules in
 * {2,3,4,5,7,8,9}: 6, 14, 18, 21, 24, 28, 35, 36, 45, 56, 63.
 *
 * ROUND gen_r12 delta (rounds 11-12 are the all-hands-on-L=100 rounds):
 * - WITHIN-VOLUME SoA chain (gstep_wvs; BORROWED: gen_batchlane gen_r11,
 *   the brief's approach #4 -- their engine took the L=100 cell at 4059 us
 *   vs my r11 slab form's 6009): for EVERY generic L >= 8 at small batch
 *   (r = B%8 in 1..GSPLIT_RMAX), the per-volume chain runs in NS =
 *   ceil(L/8) slabs of the batched engine's interleaved site format with
 *   LANES = 8 X-PLANES of one volume.  zy sweeps are the batched engine's
 *   shuffle-free elementwise pencils per L2-resident slab; the x pass
 *   tr8-gathers each (y,8z) column into a 16 KB scratch, runs the generic
 *   pencil with the map fused at its stage-2 stores (c prepacked once per
 *   chain in consumption order), and tr8-scatters back in place.  One
 *   128 B-site stream replaces the split path's two re/im line streams
 *   (my r11 next-list #1).  Pads are exact zeros and provably stay zero.
 *   Beats the r8 rotation form at every measured size (-16..-25%) and the
 *   r11 slab form at 50/100 (-25/-14%).  At 50/100 the pencils are
 *   FORMULA-baked gpencilf (GT/CRT indices as pragma-unrolled literals --
 *   the differential PMU read 29.5M insn/step through the table pencils
 *   vs gen_batchlane's 13.4M; formulas + forced unroll cut mine to
 *   24.7M) and 100 runs role-SWAPPED (25,4).  Knobs: -DGWVS=0 (r11 slab
 *   form), -DGWVS_MIN, -DGWVSSW(50), -DGWVSPF (gather prefetch, raced
 *   OFF).
 *
 * ROUND gen_r11 deltas (the all-hands-on-L=100 round; cross-class entry):
 * - CLAIMS 50 = 2x25 (B=4 graded) and 100 = 4x25 (B=1 graded) -- both are
 *   coprime pairs this class always could serve and had deliberately left
 *   to gen_pfa_large/gen_powp; the r11 brief invites cross-class entries
 *   and lets the race pick.  Both route through the per-volume split
 *   chain (B=1 and B%8=4 <= GSPLIT_RMAX).
 * - Module 25 = twiddled 5x5 Cooley-Tukey (gdft25; BORROWED: gen_pfa_large
 *   gen_r1's DFT25 decomposition, stage A stores U[5*k1+n2]): ~404 vector
 *   FP vs the h=12 fold's ~650.  First twiddled stage in this entry;
 *   16 nontrivial W25 twiddles, long-double tables at create() in
 *   consumption order.  -DG25CT=0 races the fold back.  Also speeds the
 *   existing 75 = 3x25 cell.
 * - gstep_slab: slab-FUSED split step for L >= GSLAB_MIN (default 44) --
 *   two-axes-per-pass fusion (lit 11 Tier 2) from this class's angle:
 *   pass 1 (stride L^2) IN PLACE over the state; then per x-slab, pass 2
 *   (stride L) into ONE reused L2-resident scratch slab and pass 3
 *   (stride 1) on the hot scratch back into the state slab, map fused.
 *   Kills both full-volume ping-pong round trips and the r8 rotated
 *   stores' L miss streams (the shape gen_pfa_large r1 measured losing
 *   45% at L=100).  Counter-verified at L=100 (differential over 128
 *   graded steps): LLC misses 2.94M -> 0.44M lines/step, l2_lines_in
 *   1.81M -> 0.80M, time 14.5 -> ~6.0-6.6 ms/step.  Sizes below
 *   GSLAB_MIN keep r8 rotation behavior bit-identically.
 * - r11b: pass-3 HALF-TURN stores (my r7 trick per slab: -DGSLABSW=0
 *   races back) -- inner-dim parity alternates, c read in the matching
 *   parity, one unpermute when m is odd; kills the transpose-back tr8s.
 *   Slab c-warm prefetch loop before pass 2 (-DGWARMC, default 1; hides
 *   pass-3's c stream behind pass-2 compute, -2..-8% at 100; GWARMS=1
 *   -- warming S itself -- LOSES ~+3%, no compute to hide behind).
 * - r11c: gspencil_ip -- the r4 batched IPOK rule ported to the split
 *   pencils (both claimed pairs have Q == 1 mod P): stage 1 writes back
 *   to its own slots, stage 2 inmap-reads/CRT-scatters, no tr/ti round
 *   trip.  Used by gstep_slab passes 1+3 for IPOK pairs; buffered
 *   gspencil stays for pass 2 and non-IPOK pairs.
 * - Raced and kept div (PS_RCPMAP LOSES +2-3% even in this traffic-bound
 *   cell -- fifth codelet-local map-tail confirmation).
 *
 * ROUND gen_r10 deltas:
 * - FACTOR-SWAPPED fused-map x-pencils at 10/15/20 (BORROWED: gen_batchlane
 *   gen_r9, taken with their slot tables and their ICL race verdicts):
 *   large factor in stage 1 (map-free), small factor in stage 2 where the
 *   map fuses, so each fused-map store block is a DFT2/3/4 tail instead of
 *   a DFT5's.  Same FP count, same ld/st count; the win is dependency
 *   shape (their ICL numbers: -2.2% at 10, -1.0% at 15, -1.4..1.6% at 20;
 *   swap LOSES +3.5..4.7% at 12 -- dft12_swm kept buildable, default OFF).
 *   Swapped in-place hazards worked out here: 15's stage 2 is a 4-CYCLE
 *   (c1->c2->c4->c3->c1), resolved by keeping one group pre-loaded ahead
 *   of each store block; 20's is two mutual pairs (the D5X2 shape); 10/12
 *   are register-explicit (stage 2 reads registers -- no hazard).  Knobs:
 *   -DSWAP10/12/15/20={0,1}, map body+tail -DMT<L>S (same encoding as
 *   MT<L>).  Sweeps, split path, generic engine untouched; L=12 batched
 *   and ALL split/generic paths ship bit-identical to r9.
 *
 * ROUND gen_r9 deltas:
 * - PHI-LIFTED DFT5 v-pair (BORROWED: gen_batchlane gen_r7, lit 08 6.3):
 *   sin(2pi/5) = phi * sin(pi/5) exactly, so v1/v2 factor through one
 *   shared u = sa - PHI5*sb -- 6 vector ops instead of 8 per DFT5, zero
 *   latency cost, two new exact constants (PHI5, KL5).  Applied in BOTH
 *   DFT5 forms: D5CORE (tuned batched stage-2 at 10/15/20 incl. the D5X2
 *   hazard pair and the X5 register-explicit binds) and M_DFT5 (the split
 *   B=1/B%8 pencils at 10/15/20).  Pencil FP per 8 vols: 10: 88 -> 84,
 *   15: 162 -> 156, 20: 216 -> 208; L=12 has no DFT5 and ships
 *   bit-identical.  NOT bit-transparent (same exact values, different
 *   rounding), so all gates re-run.  -DLIFT5=0 restores r8 arithmetic for
 *   the cross-arch races.  Generic-engine module 5 stays the h=2 fold
 *   (table-driven, different arithmetic; untouched).
 *
 * ROUND gen_r8 deltas (ALL tuned paths -- batched AND split -- ship
 * bit-identical to r7, cmp-verified on full graded chains at 10 B=64,
 * 20 B=32, 12 B=1, 15 B=9):
 * - The r7 rotation fused-map split chain PORTED TO THE GENERIC ENGINE
 *   (my r7 next-list #1; the surprise addendum's L=21/44-class gap).
 *   gstep_split = DEF_STEP's three passes with runtime L, using new
 *   out-of-place split-complex gspencil_<P>_<Q> instantiations (same
 *   GT maps/modules as gpencil, always buffered; the (P,Q) list is now
 *   one X-macro, GP_LIST).  Chain remainder volumes r = B%8 in
 *   1..GSPLIT_RMAX (default 4, raced) run per-volume through it; r >= 5
 *   keeps the lane-replicated SoA group (flat cost wins there: +18..+29%
 *   for split at r=5..7, raced at 21/34).  L=6 (< 8 lanes/slab row)
 *   always lane-replicates.  B=1 graded-shape chains, same-core vs MKL:
 *   14: 49.8 -> 11.10 us (MKL 12.65); 21: 184.1 -> 40.93 (72.29);
 *   34: 1296.7 -> 233.0 (792.0); 44: 2825.6 -> 520.9 (580.4) -- 4.5-5.6x
 *   over r7's lane replication, 1.1-3.4x FASTER than MKL where r7 lost
 *   3-4x.  A first cut reusing the in-place gpen on a gathered site
 *   buffer was 10-20% slower than the dedicated pencils (its 4L-vector
 *   round trip per pass-1/2 pencil) and was replaced.
 *
 * ROUND gen_r7 deltas (B=1/B%8 split-path chain rebuilt; ALL batched paths
 * bit-identical to the r6 ship, cmp-verified on full graded chains):
 * - Fused-map ROTATION step (step_<L>): three passes S->D->S->D; pass 3
 *   transforms the stride-1 dim and stores the volume ROTATED
 *   ([P0][P1][P2] -> [P2][P0][P1]), so its rows are uniformly strided over
 *   the whole volume -- ceil(L^2/8) row-blocks with ONE overlap tail
 *   instead of L*ceil(L/8) per-slab blocks -- and only the IN-transposes
 *   remain (tr8 count 160 -> 52 per volume-step at L=10).  The graded map
 *   is fused into pass-3's contiguous stores (map8c; the r2 "map at the
 *   stores of the last axis" lesson finally applied to the split path):
 *   the separate map_span state+c round trip per step is gone.  Layout
 *   cycles with period 3; the chain keeps c in all three rotations
 *   (Cr/Cra/Crb) and un-rotates once at chain end when m % 3 != 0.
 * - B=1 graded chains, same-core vs MKL: 2.57/3.36/10.76/21.7 us at
 *   10/12/15/20 (r6: 3.88/5.31/14.0/32.3) -- 1.7/2.2/1.5/2.5x FASTER than
 *   MKL at B=1.  -DSPLITZ<L>=0 rebuilds the r6 sandwich+map_span path.
 * - DENSE stage-matrix broadcast-FMA pass-3 (dense3_slab, lit 11 Tier 2
 *   stage-as-outer-product; -DSPLITZ<L>=2): measured and REJECTED at every
 *   size (+26..+41% vs the half-turn) -- the "dense GEMM wins at L<=16"
 *   crossover claim is dead on AVX-512 even in the masked-lane B=1 regime;
 *   kept buildable for the cross-arch races.
 *
 * ROUND gen_r6 deltas (the surprise-round coverage widening; tuned 10/12/
 * 15/20 paths untouched, bit-identical to the r5 ship):
 * - Generic module set widened to {2,3,4,5,7,8,9,11,13,15,16,21,25,27}:
 *   odd modules go through the same conjugate-pair-fold kernel (h = n/2 now
 *   up to 13, cos/sin tables long-double at create(), h*h <= 169 doubles);
 *   NEW exact-constant DFT16 (two gdft8 + W16 combine, cos/sin(pi/8)
 *   literals).  New coprime pairs and sizes:
 *     22=2x11 26=2x13 30=2x15 33=3x11 39=3x13 42=2x21 44=4x11 48=3x16
 *     52=4x13 54=2x27 55=5x11 60=4x15 65=5x13 72=8x9 75=3x25 77=7x11
 *     84=4x21 88=8x11 91=7x13 99=9x11 104=8x13 105=7x15 108=4x27 112=7x16
 *     117=9x13 120=8x15
 *   (72=8x9 was a plain OMISSION in the r3 list -- both modules existed.)
 *   IPOK (in-place, Q==1 mod P) holds automatically at 22,26,30,39,42,48,
 *   52,55,72,75,84,105.  50/80/100 (=2x25/5x16/4x25) are deliberately NOT
 *   claimed: they are gen_pfa_large / gen_powp scored cells.
 * - COMPOSITE odd modules (21,33,35,39,45,51,55,57,63) run as a NESTED
 *   twiddle-free GT-PFA (gmodpfa; module-internal qin/qout maps built at
 *   create()): DFT21 drops ~850 -> ~530 vector ops vs the h=10 fold
 *   (-12% at 42, -11% at 84, raced same-core).  Module 15's smaller cut
 *   LOSES to the fold's straight-line FMA stream (+10% at 30, +2-3% at
 *   60/105/120) -- 15 stays a fold, -DGM15PFA=1 re-races it cross-arch.
 *   Unlocks 2 x composite-odd sizes (all IPOK): 66,70,78,90,102,110,114,126.
 * - PRIME modules 17,19,23,29,31 via the same fold (h <= 15): libraries
 *   collapse at these factors (MKL is 10x behind at L=31), so their
 *   composites are prime surprise-draw material: 34,38,46,51,57,58,62,68,
 *   69,76,85,87,92,93,95,115,116,119,124.
 *   Modules 25/27 stay direct folds (prime powers need twiddled CT --
 *   gen_powp's territory; 54/75/108 route better there if they claim them).
 * - Split per-volume buffers are now allocated for the TUNED sizes only;
 *   the generic path never touches them (saves 6*L^3 doubles at create for
 *   the big draws, e.g. 83 MB at L=120).
 *
 * ROUND gen_r5 deltas (all raced with the SAME-CORE interleaved protocol --
 * one held slot lease, variants alternated on one core; BORROWED:
 * gen_batchlane gen_r4 / gen_pfa_large gen_r4):
 * - map8 now carries TWO ladder bodies, selected per size like the tail
 *   (MT<L> bit 0 = rcp, bit 1 = bl body): the bl body (BORROWED verbatim:
 *   gen_batchlane map8, bl8 r4 lineage -- hs-form Newton saves one mul/site,
 *   vector-extension arith w/ static-const constants schedules better under
 *   sched-pressure) + rcp tail is -2.8% at 12 (1.917 vs 1.973, closing the
 *   whole gap to batchlane's 1.915); bl body + div tail is -0.5% at 15 and
 *   -0.6% at 20; L=10 KEEPS the legacy body + div (bl body +0.6% there) and
 *   is bit-identical to the r4 ship.  The r4 div-at-12 verdict was a
 *   core-hop artifact on the costlier ladder: on the bl body the verdict
 *   FLIPS (rcp 1.913 vs div 1.955, five interleaved pairs).
 * - L=15 hybrid sweep (batchlane gen_r5's BL_MEM15=2) A/B-ed and REJECTED
 *   here: memory sweep 4.413-4.432 vs register sweep 4.445-4.449 (their
 *   shipped hybrid reads 4.602-4.620 same-core).  -DMEM15SW=1 keeps it
 *   buildable for the cross-arch races.
 *
 * ROUND gen_r4 deltas:
 * - Register-explicit pencils at 10/12 (BORROWED: gen_batchlane gen_r3):
 *   stage 1 memory -> named registers, stage 2 registers -> memory, 2L ld +
 *   2L st, no stage-1 store / stage-2 reload.  Measured a WASH here (my
 *   pencils were already fully inlined, unlike their out-lined r2 form);
 *   kept for the leaner store count.  At 15 the same rewrite REGRESSES
 *   +12.6% (fast-state 5.02 vs 4.46, same window) -- 30 live site registers
 *   + DFT5 temps spill; 15 stays the r3 memory form with the D5X2 fusion.
 * - Map tail re-raced on the new codelets: vdivpd WINS again at every size
 *   (12: 1.969/1.971 div vs 2.005/2.010 rcp, paired runs; rcp knob -DMTxx=1
 *   stays for cross-arch).  sched-pressure re-raced: keep on 10/12 (off
 *   costs +1.7% at 12), keep OFF at 15/20.
 * - Generic engine: IPOK in-place pencils when Q == 1 mod P (14, 18, 21,
 *   36, 56) -- stage 1 writes back to its input slots, stage 2 reads them
 *   via inmap[j2*P + k1]; kills the 2L-vector tr/ti round trip (-1.9% at 36
 *   same-window, wash at 14; all gates pass incl. chains at 14/21/56).
 * - Chain m-loop moved INSIDE the SCHED step function (soa_chain_L,
 *   gen_batchlane's chainsteps shape): constants/base addresses hoisted
 *   across steps, bit-identical outputs, -1.3% at 20 (13.123 vs 13.30),
 *   10/12/15 within noise.
 *
 * ROUND gen_r3 deltas:
 * - GENERIC runtime-table coprime-pair engine (gpencil + gtabs): same
 *   Good-Thomas slot algebra as the tuned codelets but tables built at
 *   create() (CRT coefficients A = Q*inv(Q mod P,P), B = P*inv(P mod Q,Q))
 *   and the pencil buffered through v8 temps, in-place safe for ANY pair.
 *   Odd modules 3/5/7/9 are one conjugate-pair-fold kernel with per-n
 *   cos/sin tables computed long double at create(); 2/4/8 exact-constant.
 *   Remainder volumes (B%8, B=1) at generic sizes lane-replicate
 *   (gen_batchlane gen_r1's scheme).  Gates at all 11 generic sizes:
 *   single call 2-5e-16; L=14 two-step 9.7e-16, chain m=100 under the
 *   honest anchor, bit-repeatable.
 * - sched-pressure per-function attribute on the 10/12 families
 *   (gen_batchlane gen_r2's revision): ~0 at 10, -0.4% at 12 here.
 * - RE-TESTED and REJECTED on this engine, same window, control second:
 *   rcp14+2NR map reciprocal (gen_batchlane gen_r2's -8%): LOSES 2.4-4.3%
 *   at every size here -- the x-pass saturates FMA ports, the divider is
 *   free.  Consumption-order (column-major) c layout at L=20: +4-10% --
 *   the natural layout already streams c as 20 sequential per-plane
 *   streams (consecutive columns read ADJACENT 128 B blocks per plane).
 *
 * ROUND gen_r2: the r1 engine (three full-volume passes, split-complex
 * ROUND gen_r2: the r1 engine (three full-volume passes, split-complex
 * qr/qi arrays, whole-pencil temp buffers, map in a separate reload loop)
 * measured 1.43/2.50/6.18/16.9 us and lost L=10/12/15 to gen_batchlane's
 * bl8-lineage engine.  This round adopts that engine's structure wholesale
 * (credited: gen_batchlane gen_r1, itself from ice bl8 / rivals v5_cb7847fb,
 * 8dc1a96d) and extends it to L=20, which batchlane does not cover:
 *
 * 1. ONE interleaved site arena: site s = re[8] | im[8] (128 B), 8 volumes
 *    in the zmm lanes.  Half the memory streams of split qr/qi arrays, and
 *    a site's re/im share a 128 B block (adjacent-line prefetch pair).
 * 2. PADDED plane stride PL (sites), plane bytes == 256 (mod 4096):
 *    PL = 130/162/226/418 for L = 10/12/15/20.  Unpadded, L=12 and L=20
 *    planes are == 2048 (mod 4096): the x-pass pencil's column loads stack
 *    into TWO L1 sets (L=20: 10+ lines/set > 12-way) and thrash.
 * 3. IN-PLACE slot modules, PFA maps baked into the slot lists -- no
 *    tr[]/ti[] whole-pencil temp arrays (those spill: 30-40 live v8 at
 *    L=15/20).  In-place safety: stage-2 group c reads slots {(Qc+Pb)%L}
 *    and writes {(Q inv(Q) c + P inv(P) d)%L}; both sets are the residue
 *    class {== c mod P} iff Q == 1 mod P.  Holds for 10=2*5 (5==1 mod 2),
 *    12 as 3-then-4 (4==1 mod 3), 20=4*5 (5==1 mod 4).  15=3*5 fails
 *    (5==2 mod 3): stage-2 groups c=1,2 have EQUAL read/write slot sets,
 *    so they are one fused load-both-then-store-both codelet (DFT5X2,
 *    batchlane's exact hazard and fix).
 * 4. TWO volume sweeps per step: zy sweep per x-plane (12.8..50 KiB,
 *    L1/L2-resident; z pencils stride 1 site, y pencils stride L), then
 *    the x pass per (y,z) column at stride PL, with the graded map fused
 *    IN REGISTERS into the stage-2 stores (map8, always_inline).  The r1
 *    map-as-a-separate-span-loop measured 1.2 us/vol at L=15 and 5.4
 *    us/vol at L=20 -- 20-32% of the whole step.
 * 5. Map ladder = bl8's r4 ladder: s = a^2+b^2+1e-300, rsqrt14 + two
 *    quadratic Newtons, d = fma(s,y,1) = 1+sqrt(s), ONE vdivpd (the
 *    divider unit is idle in this pass; the r1 rcp14+2NR ladder's 5 extra
 *    uops competed with the pencil FMAs).
 * 6. (C - S) == 2048 (mod 4096) de-alias offset between state and c.
 * 7. DFT5 in the 4-constant Winograd form (f +- KQ5*q), 34 instrs vs 36.
 *
 * PFA slot maps (input n, output k; a is the stage-1 module index):
 *   L=10: n=(5a+2b)%10, k=(5c+6d)%10        stage 1: 5xDFT2, stage 2: 2xDFT5
 *   L=12: n=(4a+3b)%12, k=(4c+9d)%12        stage 1: 4xDFT3, stage 2: 3xDFT4
 *   L=15: n=(5a+3b)%15, k=(10c+6d)%15       stage 1: 5xDFT3, stage 2: 3xDFT5
 *   L=20: n=(5a+4b)%20, k=(5c+16d)%20       stage 1: 5xDFT4, stage 2: 4xDFT5
 * (10/12/15 lists verbatim from gen_batchlane gen_r1; 20 derived here and
 * verified against numpy at create-time by the single-call gate.)
 *
 * B % 8 REMAINDERS AND B = 1: unchanged r1 per-volume split-complex path
 * (buffered pencils, ping-pong passes, overlapped idempotent tails).
 *
 * Correctness: single call rel L2 ~3e-16 vs numpy; the chain map is
 * arithmetically the driver fallback's own formula.
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/mman.h>
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#include "../fft3d_api.h"

const char *fft3d_name(void) { return "gen_pfa_small"; }
const char *fft3d_description(void)
{
    return "PFA coprime (10=2x5,12=3x4,15=3x5,20=4x5), no twiddles; interleaved "
           "site SoA 8 vols/zmm, padded planes 256 mod 4096, in-place slot "
           "codelets, zy sweep + x-pass w/ in-register fused map; B%8 split "
           "path; r3: generic runtime-table coprime P*Q engine (modules "
           "2,3,4,5,7,8,9) for 6,14,18,21,24,28,35,36,45,56,63; r4: "
           "register-explicit 10/12 pencils, in-place generic pencils where "
           "Q==1 mod P (14,18,21,36,56); r5: per-size map ladder BODY+tail "
           "(bl hs-form + rcp at 12, bl + div at 15/20, legacy + div at 10), "
           "raced same-core; r6: modules widened to {2,3,4,5,7,8,9,11,13,15,"
           "16,17,19,21,23,25,27,29,31} + nested-PFA composite odd modules "
           "(21,33,35,39,45,51,55,57,63) -- 53 new sizes, all coprime P*Q "
           "in 14..127 except 50/80/100 (pfa_large/powp cells); r7: B=1/B%8 "
           "chain rebuilt -- map fused into the stride-1 pass (map_span pass "
           "gone), half-turn z-pass (transpose-in only, swapped stores, "
           "parity-alternating c) vs dense stage-matrix broadcast-FMA pencil "
           "(lit 11 Tier 2) raced per size via -DSPLITZ<L>; r8: the rotation "
           "fused-map split chain ported to the GENERIC engine (site-buffer "
           "gather + the existing in-place gpen codelets, L >= 8) -- B%8 "
           "remainder volumes r <= GSPLIT_RMAX no longer lane-replicate; "
           "r9: phi-lifted DFT5 v-pair (6 ops vs 8; borrowed gen_batchlane "
           "r7 / lit 08 6.3) in D5CORE + M_DFT5 -- pencil FP 84/156/208 at "
           "10/15/20 batched and in the split pencils; -DLIFT5=0 races back; "
           "r10: factor-swapped fused-map x-pencils at 10/15/20 (small "
           "factor in stage 2 where the map fuses; borrowed gen_batchlane "
           "r9 incl. their 12-loses verdict), -DSWAP<L> races back; "
           "r11 (all hands on L=100): CLAIMS 50=2x25 and 100=4x25 -- "
           "module 25 as twiddled 5x5 CT (borrowed gen_pfa_large r1's "
           "DFT25 shape; long-double tables; -DG25CT=0 races the fold) and "
           "a slab-FUSED split step for L>=GSLAB_MIN (passes 2+3 per "
           "L2-resident x-slab, natural-order map-fused stores, no "
           "rotation: one volume round trip per step gone vs r8); "
           "r12: WITHIN-VOLUME SoA chain for EVERY generic size at small "
           "batch (borrowed gen_batchlane r11, approach #4): lanes = 8 "
           "x-planes of one volume in the batched engine's site format -- "
           "shuffle-free zy sweeps per L2-resident slab, x-pass via "
           "tr8-bracketed 16 KB scratch pencil with the map fused at its "
           "stage-2 stores, c prepacked in consumption order, exact-zero "
           "pads; beats the r8 rotation form at every measured size "
           "(-16..-25%); at 50/100 the pencils are FORMULA-baked "
           "(pragma-unrolled GT/CRT index literals, no table loads) and "
           "100 runs role-SWAPPED (gdft25 stage 1 map-free, map at the "
           "DFT4 stage-2 -- the r10 swap verdict; 50 keeps the map on the "
           "DFT25 side per gen_batchlane r10); -DGWVS=0 races the r11 "
           "slab form back, -DGWVS_MIN/-DGWVSSW/-DGWVSSW50/-DGWVSPF race "
           "per host";
}
static int gfactor(int L, int *P, int *Q);
int fft3d_supports(int L)
{
    int P, Q;
    return L == 10 || L == 12 || L == 15 || L == 20 || gfactor(L, &P, &Q);
}

/* ------------------------------------------------------------------ SIMD */

typedef double v8 __attribute__((vector_size(64)));
typedef long long v8i __attribute__((vector_size(64)));

static inline v8 vload(const double *p) { v8 v; __builtin_memcpy(&v, p, 64); return v; }
static inline void vstore(double *p, v8 v) { __builtin_memcpy(p, &v, 64); }

#define SHUF(a, b, ...) __builtin_shuffle((a), (b), (v8i){__VA_ARGS__})

/* 8x8 doubles transpose, in place on m[0..7]: m'[j] = old column j. */
static inline void tr8(v8 *m)
{
    v8 u0 = SHUF(m[0], m[1], 0, 8, 2, 10, 4, 12, 6, 14);
    v8 u1 = SHUF(m[0], m[1], 1, 9, 3, 11, 5, 13, 7, 15);
    v8 u2 = SHUF(m[2], m[3], 0, 8, 2, 10, 4, 12, 6, 14);
    v8 u3 = SHUF(m[2], m[3], 1, 9, 3, 11, 5, 13, 7, 15);
    v8 u4 = SHUF(m[4], m[5], 0, 8, 2, 10, 4, 12, 6, 14);
    v8 u5 = SHUF(m[4], m[5], 1, 9, 3, 11, 5, 13, 7, 15);
    v8 u6 = SHUF(m[6], m[7], 0, 8, 2, 10, 4, 12, 6, 14);
    v8 u7 = SHUF(m[6], m[7], 1, 9, 3, 11, 5, 13, 7, 15);
    v8 w0 = SHUF(u0, u2, 0, 1, 8, 9, 4, 5, 12, 13);
    v8 w2 = SHUF(u0, u2, 2, 3, 10, 11, 6, 7, 14, 15);
    v8 w1 = SHUF(u1, u3, 0, 1, 8, 9, 4, 5, 12, 13);
    v8 w3 = SHUF(u1, u3, 2, 3, 10, 11, 6, 7, 14, 15);
    v8 w4 = SHUF(u4, u6, 0, 1, 8, 9, 4, 5, 12, 13);
    v8 w6 = SHUF(u4, u6, 2, 3, 10, 11, 6, 7, 14, 15);
    v8 w5 = SHUF(u5, u7, 0, 1, 8, 9, 4, 5, 12, 13);
    v8 w7 = SHUF(u5, u7, 2, 3, 10, 11, 6, 7, 14, 15);
    m[0] = SHUF(w0, w4, 0, 1, 2, 3, 8, 9, 10, 11);
    m[4] = SHUF(w0, w4, 4, 5, 6, 7, 12, 13, 14, 15);
    m[1] = SHUF(w1, w5, 0, 1, 2, 3, 8, 9, 10, 11);
    m[5] = SHUF(w1, w5, 4, 5, 6, 7, 12, 13, 14, 15);
    m[2] = SHUF(w2, w6, 0, 1, 2, 3, 8, 9, 10, 11);
    m[6] = SHUF(w2, w6, 4, 5, 6, 7, 12, 13, 14, 15);
    m[3] = SHUF(w3, w7, 0, 1, 2, 3, 8, 9, 10, 11);
    m[7] = SHUF(w3, w7, 4, 5, 6, 7, 12, 13, 14, 15);
}

/* ------------------------------------------------- exact module constants */

#define K3  0.86602540378443864676   /* sin(pi/3)   */
#define K25 0.25
#define KQ5 0.55901699437494742410   /* sqrt(5)/4   */
#define S51 0.95105651629515357212   /* sin(2pi/5)  */
#define S52 0.58778525229247312917   /* sin(4pi/5)  */
#define C51 0.30901699437494742410   /* cos(2pi/5)  */
#define C52 (-0.80901699437494742410) /* cos(4pi/5) */

/* r9 lifted DFT5 v-pair (BORROWED: gen_batchlane gen_r7, lit 08 6.3):
 * sin(2pi/5)/sin(pi/5) = 2cos(pi/5) = phi EXACTLY, so the scaled-reflection
 * pair v1 = S51*sa + S52*sb, v2 = S52*sa - S51*sb factors through one shared
 * term at the same dependency depth:
 *     u  = sa - PHI5*sb            (FMA)
 *     v2 = S52*u                   (mul)
 *     v1 = S51*u + KL5*sb          (FMA), KL5 = (S51^2+S52^2)/S52 = 1.25/S52
 * 6 vector ops instead of 8 per DFT5 (both components).  Constants exact to
 * the last bit (50-digit Decimal; KL5 - S51*PHI5 - S52 == 0 in double).
 * PER-SIZE knobs (the MT<L>/SPLITZ<L> pattern): -DLIFT5=0 strips it
 * globally, -DLIFT5_10/15/20=0 per size -- wallaby (SPR) measures the lift
 * WINNING at 15 (-0.9%) and 20 (-1.2%) but LOSING +0.6% at 10 (four clean
 * interleaved pairs each), so the SPR advisory race should flip only 10. */
#ifndef LIFT5
#define LIFT5 1
#endif
#ifndef LIFT5_10
#define LIFT5_10 LIFT5
#endif
#ifndef LIFT5_15
#define LIFT5_15 LIFT5
#endif
#ifndef LIFT5_20
#define LIFT5_20 LIFT5
#endif
#define PHI5 1.61803398874989484820   /* (1+sqrt5)/2               */
#define KL5  2.12662702088009983045   /* 1.25/sin(pi/5)            */

/* ------------------------------------------------------------- the map
 * Exactly the driver fallback's arithmetic: sc = 1/(1+sqrt(re^2+im^2)).
 * map8: one site in registers, c site at cp (re at cp, im at cp+8).
 * BORROWED: gen_batchlane gen_r1's ladder (bl8 r4 lineage): additive
 * 1e-300 guard, rsqrt14 + 2 quadratic Newtons, d = fma(s,y,1), one exact
 * vdivpd on the otherwise-idle divider unit. */

#if defined(__AVX512F__)
/* 1/d tail, split path (map_span): ONE vdivpd on the otherwise-idle
 * divider; -DPS_RCPMAP builds the ladder for the cross-arch reruns. */
static inline __attribute__((always_inline)) __m512d recip8(__m512d d)
{
#if defined(PS_RCPMAP)
    __m512d t = _mm512_rcp14_pd(d);
    t = _mm512_fmadd_pd(t, _mm512_fnmadd_pd(d, t, _mm512_set1_pd(1.0)), t);
    t = _mm512_fmadd_pd(t, _mm512_fnmadd_pd(d, t, _mm512_set1_pd(1.0)), t);
    return t;
#else
    return _mm512_div_pd(_mm512_set1_pd(1.0), d);
#endif
}

/* The fused map's 1/d tail is a PER-SIZE compile-time choice (r4, following
 * gen_batchlane gen_r3's per-size split): the div-vs-rcp verdict is a
 * property of the surrounding codelet's port pressure, measured to flip
 * with codelet structure twice already (my r3 record).  rcp=1 builds
 * rcp14 + 2 Newton residual steps (+5 FMA-port ops, no divider); rcp=0 one
 * exact vdivpd.  always_inline + constant arg = dead branch eliminated.
 * MT10/12/15/20 defaults below were raced on the r4 register-explicit
 * codelets; override with -DMT10=1 etc. for the cross-arch reruns. */
static inline __attribute__((always_inline)) void
map8(v8 *zr, v8 *zi, const double *restrict cp, const int rcp, const int blf)
{
    /* r5: TWO ladder bodies, selected per size like the tail (both raced
     * same-core; the form verdict is codelet-local, exactly like div/rcp).
     * blf=1: verbatim gen_batchlane map8 (bl8 r4 lineage) -- hs-form Newton
     * (hs = s/2 hoisted once, y *= (1.5 - hs*y^2), one mul/site fewer) in
     * vector-extension arithmetic w/ static-const vector constants, which
     * schedules better under sched-pressure than set1 intrinsics: -2.4% at
     * 12 (w/ rcp tail), -0.4% at 15, wash at 20.  blf=0: the r2-r4 set1
     * 3-mul-Newton transcription, which stays FASTER at 10 (1.154-1.156 vs
     * 1.162-1.165, three interleaved rounds) -- L=10's 20-live-register
     * pencil leaves sched-pressure nothing to fix and the static-const
     * loads just add L1 pressure. */
    if (blf) {
        static const v8 eps = { 1e-300, 1e-300, 1e-300, 1e-300,
                                1e-300, 1e-300, 1e-300, 1e-300 };
        static const v8 half = { 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
        static const v8 c15 = { 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5 };
        static const v8 one = { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
        const v8 wr = *zr + *(const v8 *)cp;
        const v8 wi = *zi + *(const v8 *)(cp + 8);
        v8 s = wr * wr + eps;
        s = wi * wi + s;
        v8 y = (v8)_mm512_rsqrt14_pd((__m512d)s);
        const v8 hs = s * half;
        v8 u = y * y;
        y = y * (c15 - hs * u);
        u = y * y;
        y = y * (c15 - hs * u);
        const v8 d = s * y + one;      /* 1 + |w| */
        v8 t;
        if (rcp) {
            t = (v8)_mm512_rcp14_pd((__m512d)d);
            t = t + t * (one - d * t); /* residual Newton: 2^-14 -> 2^-28 */
            t = t + t * (one - d * t); /* -> full double */
        } else {
            t = one / d;               /* one exact vdivpd */
        }
        *zr = wr * t;
        *zi = wi * t;
    } else {
        __m512d a = _mm512_add_pd((__m512d)*zr, _mm512_loadu_pd(cp));
        __m512d b = _mm512_add_pd((__m512d)*zi, _mm512_loadu_pd(cp + 8));
        __m512d m = _mm512_fmadd_pd(a, a,
                        _mm512_fmadd_pd(b, b, _mm512_set1_pd(1e-300)));
        __m512d r = _mm512_rsqrt14_pd(m);
        __m512d t = _mm512_mul_pd(m, r);
        r = _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(0.5), r),
                          _mm512_fnmadd_pd(t, r, _mm512_set1_pd(3.0)));
        t = _mm512_mul_pd(m, r);
        r = _mm512_mul_pd(_mm512_mul_pd(_mm512_set1_pd(0.5), r),
                          _mm512_fnmadd_pd(t, r, _mm512_set1_pd(3.0)));
        __m512d d = _mm512_fmadd_pd(m, r, _mm512_set1_pd(1.0));
        __m512d y;
        if (rcp) {
            y = _mm512_rcp14_pd(d);
            y = _mm512_fmadd_pd(y, _mm512_fnmadd_pd(d, y, _mm512_set1_pd(1.0)), y);
            y = _mm512_fmadd_pd(y, _mm512_fnmadd_pd(d, y, _mm512_set1_pd(1.0)), y);
        } else {
            y = _mm512_div_pd(_mm512_set1_pd(1.0), d);
        }
        *zr = (v8)_mm512_mul_pd(a, y);
        *zi = (v8)_mm512_mul_pd(b, y);
    }
}
#else
static inline void map8(v8 *zr, v8 *zi, const double *cp, const int rcp,
                        const int blf)
{
    (void)rcp; (void)blf;
    for (int k = 0; k < 8; ++k) {
        double a = (*zr)[k] + cp[k], b = (*zi)[k] + cp[8 + k];
        double sc = 1.0 / (1.0 + sqrt(a * a + b * b));
        (*zr)[k] = a * sc;
        (*zi)[k] = b * sc;
    }
}
#endif

/* Per-size map defaults, re-raced same-core in r5 (one held lease, variants
 * alternated on ONE core -- gen_batchlane r4's protocol; my r4 verdicts came
 * from core-hopping tryout invocations).  The div-vs-rcp verdict FLIPPED at
 * 12 on the bl-form body (rcp 1.913 vs div 1.955, five interleaved pairs);
 * 10 keeps the r4 legacy body + div (bl body costs +0.6% there); 15/20 take
 * the bl body with the div tail.  Override with -DMT<L>=<0..3>. */
/* Values: 0 = legacy body + vdivpd, 1 = legacy + rcp ladder,
 *         2 = bl body + vdivpd,     3 = bl body + rcp ladder. */
#ifndef GMT              /* generic coprime-pair engine's map */
#define GMT 2
#endif
#ifndef MT10
#define MT10 0
#endif
#ifndef MT12
#define MT12 3
#endif
#ifndef MT15
#define MT15 2
#endif
#ifndef MT20
#define MT20 2
#endif

/* Contiguous split spans (the B%8 split path); n need not be 8-aligned. */
static inline void map_span(double *zr, double *zi,
                            const double *cr, const double *ci, ptrdiff_t n)
{
    ptrdiff_t i = 0;
#if defined(__AVX512F__)
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d half = _mm512_set1_pd(0.5), c15 = _mm512_set1_pd(1.5);
    const __m512d tiny = _mm512_set1_pd(1e-300);
    for (; i + 8 <= n; i += 8) {
        __m512d a = _mm512_add_pd(_mm512_loadu_pd(zr + i), _mm512_loadu_pd(cr + i));
        __m512d b = _mm512_add_pd(_mm512_loadu_pd(zi + i), _mm512_loadu_pd(ci + i));
        __m512d m = _mm512_fmadd_pd(a, a, _mm512_fmadd_pd(b, b, tiny));
        __m512d r = _mm512_rsqrt14_pd(m);
        __m512d hm = _mm512_mul_pd(m, half);
        __m512d u = _mm512_mul_pd(r, r);
        r = _mm512_mul_pd(r, _mm512_fnmadd_pd(hm, u, c15));
        u = _mm512_mul_pd(r, r);
        r = _mm512_mul_pd(r, _mm512_fnmadd_pd(hm, u, c15));
        __m512d d = _mm512_fmadd_pd(m, r, one);
        __m512d y = recip8(d);
        _mm512_storeu_pd(zr + i, _mm512_mul_pd(a, y));
        _mm512_storeu_pd(zi + i, _mm512_mul_pd(b, y));
    }
#endif
    for (; i < n; ++i) {
        double a = zr[i] + cr[i], b = zi[i] + ci[i];
        double sc = 1.0 / (1.0 + sqrt(a * a + b * b));
        zr[i] = a * sc;
        zi[i] = b * sc;
    }
}

/* r7: the split-path fused map -- map_span's exact ladder (hs-form Newtons,
 * recip8 tail) on v8 values with the c site pre-loaded by the caller.  Used
 * by the new split step's last pass so the chain loses its separate
 * map_span reload pass (a full state+c round trip per step). */
#if defined(__AVX512F__)
static inline __attribute__((always_inline)) void
map8c(v8 *zr, v8 *zi, v8 crv, v8 civ)
{
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d half = _mm512_set1_pd(0.5), c15 = _mm512_set1_pd(1.5);
    const __m512d tiny = _mm512_set1_pd(1e-300);
    __m512d a = _mm512_add_pd((__m512d)*zr, (__m512d)crv);
    __m512d b = _mm512_add_pd((__m512d)*zi, (__m512d)civ);
    __m512d m = _mm512_fmadd_pd(a, a, _mm512_fmadd_pd(b, b, tiny));
    __m512d r = _mm512_rsqrt14_pd(m);
    __m512d hm = _mm512_mul_pd(m, half);
    __m512d u = _mm512_mul_pd(r, r);
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(hm, u, c15));
    u = _mm512_mul_pd(r, r);
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(hm, u, c15));
    __m512d d = _mm512_fmadd_pd(m, r, one);
    __m512d y = recip8(d);
    *zr = (v8)_mm512_mul_pd(a, y);
    *zi = (v8)_mm512_mul_pd(b, y);
}
#else
static inline void map8c(v8 *zr, v8 *zi, v8 crv, v8 civ)
{
    for (int k = 0; k < 8; ++k) {
        double a = (*zr)[k] + crv[k], b = (*zi)[k] + civ[k];
        double sc = 1.0 / (1.0 + sqrt(a * a + b * b));
        (*zr)[k] = a * sc;
        (*zi)[k] = b * sc;
    }
}
#endif

/* --------------------------------------- in-place site modules (SoA path)
 * A pencil lives at base p with stride st doubles between sites; slot k's
 * re vector is at p + k*st, im at p + k*st + 8.  Modules load all their
 * slots before storing, so a module is always in-place safe; the slot
 * lists in the dftL_ip functions carry the cross-group safety argument
 * from the file header. */

#define QR_(p, st, k) (*(v8 *)((p) + (size_t)(k) * (st)))
#define QI_(p, st, k) (*(v8 *)((p) + (size_t)(k) * (st) + 8))

/* MT encodes tail and body: bit 0 = rcp tail, bit 1 = bl-form body (r5). */
#define STM(p, st, cp, MT, o, rr, ii) do {                                    \
        v8 zr_ = (rr), zi_ = (ii);                                            \
        map8(&zr_, &zi_, (cp) + (size_t)(o) * (st), (MT) & 1, ((MT) >> 1) & 1); \
        QR_(p, st, o) = zr_;  QI_(p, st, o) = zi_;                            \
    } while (0)

/* --------- r4 register-explicit pencils for 10/12/15 (BORROWED:
 * gen_batchlane gen_r3, transitively gen_pow2 r1's count-the-stores asm
 * audit).  Stage 1 reads memory and writes NAMED registers xr<k>/xi<k>;
 * stage 2 reads registers and stores straight to memory (map fused in the
 * *_ipm variants).  Exactly 2L zmm loads + 2L zmm stores per pencil, no
 * stage-1 store / stage-2 reload round trip (gcc's DSE left 27/39/60 dead
 * stores per pencil at 10/12/15 in the r3 memory form -- their audit).
 * Bonus: the L=15 stage-2 equal-slot-set hazard (r2's fused DFT5X2) is
 * GONE -- stage 2 never reads memory, so no store/load ordering exists. */

/* stage-1 DFT2 slots a,b: memory -> registers */
#define X2L(p, st, a, b) do {                                                 \
        v8 t0r = QR_(p, st, a), t0i = QI_(p, st, a);                          \
        v8 t1r = QR_(p, st, b), t1i = QI_(p, st, b);                          \
        xr##a = t0r + t1r;  xi##a = t0i + t1i;                                \
        xr##b = t0r - t1r;  xi##b = t0i - t1i;                                \
    } while (0)

/* stage-1 DFT3 slots a,b,c: memory -> registers */
#define X3L(p, st, a, b, c) do {                                              \
        v8 t0r = QR_(p, st, a), t0i = QI_(p, st, a);                          \
        v8 t1r = QR_(p, st, b), t1i = QI_(p, st, b);                          \
        v8 t2r = QR_(p, st, c), t2i = QI_(p, st, c);                          \
        v8 tr_ = t1r + t2r, ti_ = t1i + t2i;                                  \
        v8 ur_ = t1r - t2r, ui_ = t1i - t2i;                                  \
        v8 hr_ = t0r - 0.5 * tr_, hi_ = t0i - 0.5 * ti_;                      \
        xr##a = t0r + tr_;        xi##a = t0i + ti_;                          \
        xr##b = hr_ + K3 * ui_;   xi##b = hi_ - K3 * ur_;                     \
        xr##c = hr_ - K3 * ui_;   xi##c = hi_ + K3 * ur_;                     \
    } while (0)

/* stage-2 DFT4: registers i0..i3 -> memory o0..o3 (X4STM fuses the map) */
#define X4CORE(i0, i1, i2, i3)                                                \
        v8 t0r = xr##i0 + xr##i2, t0i = xi##i0 + xi##i2;                      \
        v8 t1r = xr##i0 - xr##i2, t1i = xi##i0 - xi##i2;                      \
        v8 t2r = xr##i1 + xr##i3, t2i = xi##i1 + xi##i3;                      \
        v8 t3r = xr##i1 - xr##i3, t3i = xi##i1 - xi##i3;                      \
        v8 y0r = t0r + t2r, y0i = t0i + t2i;                                  \
        v8 y2r = t0r - t2r, y2i = t0i - t2i;                                  \
        v8 y1r = t1r + t3i, y1i = t1i - t3r;                                  \
        v8 y3r = t1r - t3i, y3i = t1i + t3r;

#define X4ST(p, st, i0, i1, i2, i3, o0, o1, o2, o3) do {                      \
        X4CORE(i0, i1, i2, i3)                                                \
        QR_(p, st, o0) = y0r;  QI_(p, st, o0) = y0i;                          \
        QR_(p, st, o1) = y1r;  QI_(p, st, o1) = y1i;                          \
        QR_(p, st, o2) = y2r;  QI_(p, st, o2) = y2i;                          \
        QR_(p, st, o3) = y3r;  QI_(p, st, o3) = y3i;                          \
    } while (0)

#define X4STM(p, st, cp, MT, i0, i1, i2, i3, o0, o1, o2, o3) do {             \
        X4CORE(i0, i1, i2, i3)                                                \
        STM(p, st, cp, MT, o0, y0r, y0i);                                     \
        STM(p, st, cp, MT, o1, y1r, y1i);                                     \
        STM(p, st, cp, MT, o2, y2r, y2i);                                     \
        STM(p, st, cp, MT, o3, y3r, y3i);                                     \
    } while (0)

/* stage-2 DFT5 input bind: registers i0..i4 -> the D5CORE operand names */
#define X5B(T, i0, i1, i2, i3, i4)                                            \
        v8 T##x0r = xr##i0, T##x0i = xi##i0;                                  \
        v8 T##x1r = xr##i1, T##x1i = xi##i1;                                  \
        v8 T##x2r = xr##i2, T##x2i = xi##i2;                                  \
        v8 T##x3r = xr##i3, T##x3i = xi##i3;                                  \
        v8 T##x4r = xr##i4, T##x4i = xi##i4;

#define X5ST(T, p, st, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4) do {           \
        X5B(T, i0, i1, i2, i3, i4)                                            \
        D5CORE(T)                                                             \
        D5STORE(T, p, st, o0, o1, o2, o3, o4)                                 \
    } while (0)

#define X5STM(T, p, st, cp, MT, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4) do {  \
        X5B(T, i0, i1, i2, i3, i4)                                            \
        D5CORE(T)                                                             \
        D5STOREM(T, p, st, cp, MT, o0, o1, o2, o3, o4)                        \
    } while (0)

#define XDECL10 v8 xr0, xi0, xr1, xi1, xr2, xi2, xr3, xi3, xr4, xi4,          \
                   xr5, xi5, xr6, xi6, xr7, xi7, xr8, xi8, xr9, xi9
#define XDECL12 XDECL10, xr10, xi10, xr11, xi11
#define XDECL15 XDECL12, xr12, xi12, xr13, xi13, xr14, xi14

#define D4CORE(p, st, i0, i1, i2, i3)                                         \
        v8 x0r = QR_(p, st, i0), x0i = QI_(p, st, i0);                        \
        v8 x1r = QR_(p, st, i1), x1i = QI_(p, st, i1);                        \
        v8 x2r = QR_(p, st, i2), x2i = QI_(p, st, i2);                        \
        v8 x3r = QR_(p, st, i3), x3i = QI_(p, st, i3);                        \
        v8 t0r = x0r + x2r, t0i = x0i + x2i;                                  \
        v8 t1r = x0r - x2r, t1i = x0i - x2i;                                  \
        v8 t2r = x1r + x3r, t2i = x1i + x3i;                                  \
        v8 t3r = x1r - x3r, t3i = x1i - x3i;                                  \
        v8 y0r = t0r + t2r, y0i = t0i + t2i;                                  \
        v8 y2r = t0r - t2r, y2i = t0i - t2i;                                  \
        v8 y1r = t1r + t3i, y1i = t1i - t3r;                                  \
        v8 y3r = t1r - t3i, y3i = t1i + t3r;

#define D4S(p, st, i0, i1, i2, i3, o0, o1, o2, o3) do {                       \
        D4CORE(p, st, i0, i1, i2, i3)                                         \
        QR_(p, st, o0) = y0r;  QI_(p, st, o0) = y0i;                          \
        QR_(p, st, o1) = y1r;  QI_(p, st, o1) = y1i;                          \
        QR_(p, st, o2) = y2r;  QI_(p, st, o2) = y2i;                          \
        QR_(p, st, o3) = y3r;  QI_(p, st, o3) = y3i;                          \
    } while (0)

/* DFT5, 4-constant Winograd split (34 instrs): f = x0 - p/4,
 * A1,A2 = f +- (sqrt5/4) q; equal to cos-form e1/e2 with 2 fewer FMAs. */
#define D5LOAD(T, p, st, i0, i1, i2, i3, i4)                                  \
        v8 T##x0r = QR_(p, st, i0), T##x0i = QI_(p, st, i0);                  \
        v8 T##x1r = QR_(p, st, i1), T##x1i = QI_(p, st, i1);                  \
        v8 T##x2r = QR_(p, st, i2), T##x2i = QI_(p, st, i2);                  \
        v8 T##x3r = QR_(p, st, i3), T##x3i = QI_(p, st, i3);                  \
        v8 T##x4r = QR_(p, st, i4), T##x4i = QI_(p, st, i4);

/* D5LIFT_ is an enum constant in the enclosing function's scope (set per
 * size from LIFT5_<L>); the dead branch folds away under always_inline. */
#define D5VPAIR(T)                                                            \
        v8 T##v1r, T##v1i, T##v2r, T##v2i;                                    \
        if (D5LIFT_) {                                                        \
            v8 T##ulr = T##sar - PHI5 * T##sbr;                               \
            v8 T##uli = T##sai - PHI5 * T##sbi;                               \
            T##v2r = S52 * T##ulr;            T##v2i = S52 * T##uli;          \
            T##v1r = S51 * T##ulr + KL5 * T##sbr;                             \
            T##v1i = S51 * T##uli + KL5 * T##sbi;                             \
        } else {                                                              \
            T##v1r = S51 * T##sar + S52 * T##sbr;                             \
            T##v1i = S51 * T##sai + S52 * T##sbi;                             \
            T##v2r = S52 * T##sar - S51 * T##sbr;                             \
            T##v2i = S52 * T##sai - S51 * T##sbi;                             \
        }

#define D5CORE(T)                                                             \
        v8 T##tar = T##x1r + T##x4r, T##tai = T##x1i + T##x4i;                \
        v8 T##tbr = T##x2r + T##x3r, T##tbi = T##x2i + T##x3i;                \
        v8 T##sar = T##x1r - T##x4r, T##sai = T##x1i - T##x4i;                \
        v8 T##sbr = T##x2r - T##x3r, T##sbi = T##x2i - T##x3i;                \
        v8 T##pr = T##tar + T##tbr, T##pi = T##tai + T##tbi;                  \
        v8 T##qr = T##tar - T##tbr, T##qi = T##tai - T##tbi;                  \
        v8 T##X0r = T##x0r + T##pr, T##X0i = T##x0i + T##pi;                  \
        v8 T##fr = T##x0r - K25 * T##pr, T##fi = T##x0i - K25 * T##pi;        \
        v8 T##A1r = T##fr + KQ5 * T##qr, T##A1i = T##fi + KQ5 * T##qi;        \
        v8 T##A2r = T##fr - KQ5 * T##qr, T##A2i = T##fi - KQ5 * T##qi;        \
        D5VPAIR(T)

#define D5STORE(T, p, st, o0, o1, o2, o3, o4)                                 \
        QR_(p, st, o0) = T##X0r;           QI_(p, st, o0) = T##X0i;           \
        QR_(p, st, o1) = T##A1r + T##v1i;  QI_(p, st, o1) = T##A1i - T##v1r;  \
        QR_(p, st, o4) = T##A1r - T##v1i;  QI_(p, st, o4) = T##A1i + T##v1r;  \
        QR_(p, st, o2) = T##A2r + T##v2i;  QI_(p, st, o2) = T##A2i - T##v2r;  \
        QR_(p, st, o3) = T##A2r - T##v2i;  QI_(p, st, o3) = T##A2i + T##v2r;

#define D5STOREM(T, p, st, cp, MT, o0, o1, o2, o3, o4)                        \
        STM(p, st, cp, MT, o0, T##X0r, T##X0i);                               \
        STM(p, st, cp, MT, o1, T##A1r + T##v1i, T##A1i - T##v1r);             \
        STM(p, st, cp, MT, o4, T##A1r - T##v1i, T##A1i + T##v1r);             \
        STM(p, st, cp, MT, o2, T##A2r + T##v2i, T##A2i - T##v2r);             \
        STM(p, st, cp, MT, o3, T##A2r - T##v2i, T##A2i + T##v2r);

#define D5S(p, st, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4) do {               \
        D5LOAD(a_, p, st, i0, i1, i2, i3, i4)                                 \
        D5CORE(a_)                                                            \
        D5STORE(a_, p, st, o0, o1, o2, o3, o4)                                \
    } while (0)

#define D5SM(p, st, cp, MT, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4) do {      \
        D5LOAD(a_, p, st, i0, i1, i2, i3, i4)                                 \
        D5CORE(a_)                                                            \
        D5STOREM(a_, p, st, cp, MT, o0, o1, o2, o3, o4)                       \
    } while (0)

/* ------------------------------- length-L in-place pencils, maps baked in
 * always_inline, NO optimize attribute here: the caller's SCHED attr governs
 * the inlined body (gen_batchlane gen_r3: an optimize-attr callee is what
 * out-lined their r2 pencils). */
#define AIN static inline __attribute__((always_inline))

AIN void dft10_ip(double *restrict p, const ptrdiff_t st)
{
    enum { D5LIFT_ = LIFT5_10 };
    XDECL10;
    X2L(p, st, 0, 5); X2L(p, st, 2, 7); X2L(p, st, 4, 9);
    X2L(p, st, 6, 1); X2L(p, st, 8, 3);
    X5ST(a_, p, st, 0, 2, 4, 6, 8,  0, 6, 2, 8, 4);
    X5ST(b_, p, st, 5, 7, 9, 1, 3,  5, 1, 7, 3, 9);
}
AIN void dft10_ipm(double *restrict p, const ptrdiff_t st,
                   const double *restrict cp)
{
    enum { D5LIFT_ = LIFT5_10 };
    XDECL10;
    X2L(p, st, 0, 5); X2L(p, st, 2, 7); X2L(p, st, 4, 9);
    X2L(p, st, 6, 1); X2L(p, st, 8, 3);
    X5STM(a_, p, st, cp, MT10, 0, 2, 4, 6, 8,  0, 6, 2, 8, 4);
    X5STM(b_, p, st, cp, MT10, 5, 7, 9, 1, 3,  5, 1, 7, 3, 9);
}

AIN void dft12_ip(double *restrict p, const ptrdiff_t st)
{
    XDECL12;
    X3L(p, st, 0, 4, 8);  X3L(p, st, 3, 7, 11);
    X3L(p, st, 6, 10, 2); X3L(p, st, 9, 1, 5);
    X4ST(p, st, 0, 3, 6, 9,   0, 9, 6, 3);
    X4ST(p, st, 4, 7, 10, 1,  4, 1, 10, 7);
    X4ST(p, st, 8, 11, 2, 5,  8, 5, 2, 11);
}
AIN void dft12_ipm(double *restrict p, const ptrdiff_t st,
                   const double *restrict cp)
{
    XDECL12;
    X3L(p, st, 0, 4, 8);  X3L(p, st, 3, 7, 11);
    X3L(p, st, 6, 10, 2); X3L(p, st, 9, 1, 5);
    X4STM(p, st, cp, MT12, 0, 3, 6, 9,   0, 9, 6, 3);
    X4STM(p, st, cp, MT12, 4, 7, 10, 1,  4, 1, 10, 7);
    X4STM(p, st, cp, MT12, 8, 11, 2, 5,  8, 5, 2, 11);
}

/* L=15 stays the r3 MEMORY form: the register-explicit rewrite (identical
 * to gen_batchlane r3's shipped 15) was A/B-ed in r4 and REGRESSES here,
 * +12.6% (5.02-5.05 vs 4.46 r3-form control, same window, fast-state
 * confirmed by a mixed-state run whose fast samples still read 5.02): 30
 * live site registers + DFT5 temps spill, while the memory form's stage-1
 * stores are near-free at 2 stores/cycle.  Stage-2 groups c=1,2 read and
 * write the SAME slot set (5 != 1 mod 3): fused D5X2SM, batchlane's hazard. */
#define D3S(p, st, i0, i1, i2) do {                                           \
        v8 x0r = QR_(p, st, i0), x0i = QI_(p, st, i0);                        \
        v8 x1r = QR_(p, st, i1), x1i = QI_(p, st, i1);                        \
        v8 x2r = QR_(p, st, i2), x2i = QI_(p, st, i2);                        \
        v8 tr_ = x1r + x2r, ti_ = x1i + x2i;                                  \
        v8 ur_ = x1r - x2r, ui_ = x1i - x2i;                                  \
        v8 hr_ = x0r - 0.5 * tr_, hi_ = x0i - 0.5 * ti_;                      \
        QR_(p, st, i0) = x0r + tr_;        QI_(p, st, i0) = x0i + ti_;        \
        QR_(p, st, i1) = hr_ + K3 * ui_;   QI_(p, st, i1) = hi_ - K3 * ur_;   \
        QR_(p, st, i2) = hr_ - K3 * ui_;   QI_(p, st, i2) = hi_ + K3 * ur_;   \
    } while (0)

#define D5X2S(p, st, i0,i1,i2,i3,i4, o0,o1,o2,o3,o4,                          \
                     j0,j1,j2,j3,j4, w0,w1,w2,w3,w4) do {                     \
        D5LOAD(a_, p, st, i0, i1, i2, i3, i4)                                 \
        D5LOAD(b_, p, st, j0, j1, j2, j3, j4)                                 \
        D5CORE(a_)                                                            \
        D5CORE(b_)                                                            \
        D5STORE(a_, p, st, o0, o1, o2, o3, o4)                                \
        D5STORE(b_, p, st, w0, w1, w2, w3, w4)                                \
    } while (0)

#define D5X2SM(p, st, cp, MT, i0,i1,i2,i3,i4, o0,o1,o2,o3,o4,                 \
                             j0,j1,j2,j3,j4, w0,w1,w2,w3,w4) do {             \
        D5LOAD(a_, p, st, i0, i1, i2, i3, i4)                                 \
        D5LOAD(b_, p, st, j0, j1, j2, j3, j4)                                 \
        D5CORE(a_)                                                            \
        D5CORE(b_)                                                            \
        D5STOREM(a_, p, st, cp, MT, o0, o1, o2, o3, o4)                       \
        D5STOREM(b_, p, st, cp, MT, w0, w1, w2, w3, w4)                       \
    } while (0)

AIN void dft15_ip(double *restrict p, const ptrdiff_t st)
{
    enum { D5LIFT_ = LIFT5_15 };
    D3S(p, st, 0, 5, 10);  D3S(p, st, 3, 8, 13); D3S(p, st, 6, 11, 1);
    D3S(p, st, 9, 14, 4);  D3S(p, st, 12, 2, 7);
    D5S(p, st, 0, 3, 6, 9, 12,  0, 6, 12, 3, 9);
    D5X2S(p, st, 5, 8, 11, 14, 2,   10, 1, 7, 13, 4,
                 10, 13, 1, 4, 7,    5, 11, 2, 8, 14);
}
AIN void dft15_ipm(double *restrict p, const ptrdiff_t st,
                   const double *restrict cp)
{
    enum { D5LIFT_ = LIFT5_15 };
    D3S(p, st, 0, 5, 10);  D3S(p, st, 3, 8, 13); D3S(p, st, 6, 11, 1);
    D3S(p, st, 9, 14, 4);  D3S(p, st, 12, 2, 7);
    D5SM(p, st, cp, MT15, 0, 3, 6, 9, 12,  0, 6, 12, 3, 9);
    D5X2SM(p, st, cp, MT15, 5, 8, 11, 14, 2,   10, 1, 7, 13, 4,
                            10, 13, 1, 4, 7,    5, 11, 2, 8, 14);
}

/* r5 HYBRID sweep pencil for 15 (BORROWED: gen_batchlane gen_r5's BL_MEM15=2):
 * the r4 A/B rejected the register-explicit form for BOTH passes (+12.6%),
 * but the spills live in the FUSED-MAP x-pencil (30 site regs + ~7 map temps
 * + constants); the map-free sweep pencil fits and takes 2L ld + 2L st vs the
 * memory form's 4L + 4L.  Stage 2 reads only registers, so the equal-slot-set
 * hazard of groups c=1,2 needs no fused DFT5X2 here.  x-pass keeps the r3
 * memory-form dft15_ipm.  -DMEM15SW=0 restores the r3 memory sweep. */
AIN void dft15_ipr(double *restrict p, const ptrdiff_t st)
{
    enum { D5LIFT_ = LIFT5_15 };
    XDECL15;
    X3L(p, st, 0, 5, 10);  X3L(p, st, 3, 8, 13); X3L(p, st, 6, 11, 1);
    X3L(p, st, 9, 14, 4);  X3L(p, st, 12, 2, 7);
    X5ST(a_, p, st, 0, 3, 6, 9, 12,   0, 6, 12, 3, 9);
    X5ST(b_, p, st, 5, 8, 11, 14, 2,  10, 1, 7, 13, 4);
    X5ST(c_, p, st, 10, 13, 1, 4, 7,   5, 11, 2, 8, 14);
}
/* r5 same-core race: the hybrid LOSES here -- memory sweep 4.413-4.432 vs
 * register sweep 4.445-4.449 vs gen_batchlane's shipped hybrid 4.602-4.620
 * (five interleaved rounds, all three in every round).  My 30-site register
 * sweep pencil spills even without the map (30 sites + ~14 D5CORE temps),
 * while the memory form's stage-1 stores ride the 2-store/cycle port.
 * Default 0 = r3/r4 memory sweep; -DMEM15SW=1 builds the hybrid for the
 * cross-arch races (CLX/SPR may flip it -- batchlane's does win on THEIR
 * codelet, whose D5X2 pair differs). */
#ifndef MEM15SW
#define MEM15SW 0
#endif
#if MEM15SW
#define DFT15_SWEEP dft15_ipr
#else
#define DFT15_SWEEP dft15_ip
#endif

/* L=20: 20 sites = 40 live site registers, too many for the register-
 * explicit form (gen_batchlane r3 concurs: guaranteed heavy spill), so 20
 * keeps the memory round trip; its stage-2 reloads are not dead stores. */
AIN void dft20_ip(double *restrict p, const ptrdiff_t st)
{
    enum { D5LIFT_ = LIFT5_20 };
    D4S(p, st, 0, 5, 10, 15,   0, 5, 10, 15);
    D4S(p, st, 4, 9, 14, 19,   4, 9, 14, 19);
    D4S(p, st, 8, 13, 18, 3,   8, 13, 18, 3);
    D4S(p, st, 12, 17, 2, 7,   12, 17, 2, 7);
    D4S(p, st, 16, 1, 6, 11,   16, 1, 6, 11);
    D5S(p, st, 0, 4, 8, 12, 16,    0, 16, 12, 8, 4);
    D5S(p, st, 5, 9, 13, 17, 1,    5, 1, 17, 13, 9);
    D5S(p, st, 10, 14, 18, 2, 6,   10, 6, 2, 18, 14);
    D5S(p, st, 15, 19, 3, 7, 11,   15, 11, 7, 3, 19);
}
AIN void dft20_ipm(double *restrict p, const ptrdiff_t st,
                   const double *restrict cp)
{
    enum { D5LIFT_ = LIFT5_20 };
    D4S(p, st, 0, 5, 10, 15,   0, 5, 10, 15);
    D4S(p, st, 4, 9, 14, 19,   4, 9, 14, 19);
    D4S(p, st, 8, 13, 18, 3,   8, 13, 18, 3);
    D4S(p, st, 12, 17, 2, 7,   12, 17, 2, 7);
    D4S(p, st, 16, 1, 6, 11,   16, 1, 6, 11);
    D5SM(p, st, cp, MT20, 0, 4, 8, 12, 16,    0, 16, 12, 8, 4);
    D5SM(p, st, cp, MT20, 5, 9, 13, 17, 1,    5, 1, 17, 13, 9);
    D5SM(p, st, cp, MT20, 10, 14, 18, 2, 6,   10, 6, 2, 18, 14);
    D5SM(p, st, cp, MT20, 15, 19, 3, 7, 11,   15, 11, 7, 3, 19);
}

/* --------- r10 factor-SWAPPED fused-map x-pencils (BORROWED: gen_batchlane
 * gen_r9).  PFA order symmetry: LARGE factor in stage 1 (map-free), SMALL
 * factor in stage 2 where the map fuses -- each fused-map store block is a
 * DFT2/3/4 tail after its ladder instead of a DFT5's 5-output block, and
 * the map-free stage-1 DFT5s interleave with the previous column's
 * in-flight ladders.  Same FP, same ld/st counts.  Swapped slot algebra
 * (their tables, re-derived and checked here; n input, k output):
 *   10 sw: n=(2a+5b)%10, k=(6c+5d)%10   stage 1: 2xDFT5, stage 2: 5xDFT2
 *   12 sw: n=(3a+4b)%12, k=(9c+4d)%12   stage 1: 3xDFT4, stage 2: 4xDFT3
 *   15 sw: n=(3a+5b)%15, k=(6c+10d)%15  stage 1: 3xDFT5, stage 2: 5xDFT3
 *   20 sw: n=(4a+5b)%20, k=(16c+5d)%20  stage 1: 4xDFT5, stage 2: 5xDFT4
 * Stage-1 groups read and write their OWN slot set (in-place safe alone).
 * Swapped stage-2 read/write sets (read = {P c + Q b}, write = {A c + B d}):
 *   15: c0 self; the 4-cycle c1 w{6,1,11}=c2 reads, c2 w{12,7,2}=c4 reads,
 *       c4 w{9,4,14}=c3 reads, c3 w{3,13,8}=c1 reads -- schedule keeps ONE
 *       group pre-loaded ahead of each store block (max 2 groups live);
 *   20: c0 self; mutual pairs (c1,c4) and (c2,c3) -- load both, store both
 *       (the 15-unswapped D5X2 hazard shape);
 *   10/12: register-explicit (stage 2 reads only registers) -- no hazard.
 * Forms follow the r4 live-register rule: 10/12 register (2L ld + 2L st),
 * 15/20 memory (4L + 4L). */

/* stage-1 DFT5 slots i0..i4: memory -> named registers xr<k>/xi<k> */
#define X5L(T, p, st, i0, i1, i2, i3, i4) do {                                \
        D5LOAD(T, p, st, i0, i1, i2, i3, i4)                                  \
        D5CORE(T)                                                             \
        xr##i0 = T##X0r;           xi##i0 = T##X0i;                           \
        xr##i1 = T##A1r + T##v1i;  xi##i1 = T##A1i - T##v1r;                  \
        xr##i4 = T##A1r - T##v1i;  xi##i4 = T##A1i + T##v1r;                  \
        xr##i2 = T##A2r + T##v2i;  xi##i2 = T##A2i - T##v2r;                  \
        xr##i3 = T##A2r - T##v2i;  xi##i3 = T##A2i + T##v2r;                  \
    } while (0)

/* stage-2 DFT2 from registers a,b -> map-fused stores at o0,o1 */
#define X2STM(p, st, cp, MT, a, b, o0, o1) do {                               \
        v8 s0r_ = xr##a + xr##b, s0i_ = xi##a + xi##b;                        \
        v8 s1r_ = xr##a - xr##b, s1i_ = xi##a - xi##b;                        \
        STM(p, st, cp, MT, o0, s0r_, s0i_);                                   \
        STM(p, st, cp, MT, o1, s1r_, s1i_);                                   \
    } while (0)

/* stage-1 DFT4 slots a,b,c,d: memory -> named registers */
#define X4L(p, st, a, b, c, d) do {                                           \
        v8 t0r = QR_(p, st, a), t0i = QI_(p, st, a);                          \
        v8 t1r = QR_(p, st, b), t1i = QI_(p, st, b);                          \
        v8 t2r = QR_(p, st, c), t2i = QI_(p, st, c);                          \
        v8 t3r = QR_(p, st, d), t3i = QI_(p, st, d);                          \
        v8 u0r = t0r + t2r, u0i = t0i + t2i;                                  \
        v8 u1r = t0r - t2r, u1i = t0i - t2i;                                  \
        v8 u2r = t1r + t3r, u2i = t1i + t3i;                                  \
        v8 u3r = t1r - t3r, u3i = t1i - t3i;                                  \
        xr##a = u0r + u2r;  xi##a = u0i + u2i;                                \
        xr##b = u1r + u3i;  xi##b = u1i - u3r;                                \
        xr##c = u0r - u2r;  xi##c = u0i - u2i;                                \
        xr##d = u1r - u3i;  xi##d = u1i + u3r;                                \
    } while (0)

/* stage-2 DFT3 from registers a,b,c -> map-fused stores at o0,o1,o2 */
#define X3STM(p, st, cp, MT, a, b, c, o0, o1, o2) do {                        \
        v8 tr_ = xr##b + xr##c, ti_ = xi##b + xi##c;                          \
        v8 ur_ = xr##b - xr##c, ui_ = xi##b - xi##c;                          \
        v8 hr_ = xr##a - 0.5 * tr_, hi_ = xi##a - 0.5 * ti_;                  \
        STM(p, st, cp, MT, o0, xr##a + tr_, xi##a + ti_);                     \
        STM(p, st, cp, MT, o1, hr_ + K3 * ui_, hi_ - K3 * ur_);               \
        STM(p, st, cp, MT, o2, hr_ - K3 * ui_, hi_ + K3 * ur_);               \
    } while (0)

/* memory-form swapped stage 2: named group loads (declarations, so a group
 * can stay live across another group's store block) + map-fused stores.
 * Loads of a slot precede any store to the SAME slot in program order, so
 * the restrict-safe reordering gcc may do cannot break the hazard chains. */
#define X3LD(T, p, st, a, b, c)                                               \
        v8 T##r0 = QR_(p, st, a), T##i0 = QI_(p, st, a);                      \
        v8 T##r1 = QR_(p, st, b), T##i1 = QI_(p, st, b);                      \
        v8 T##r2 = QR_(p, st, c), T##i2 = QI_(p, st, c);

#define D3CM(T, p, st, cp, MT, o0, o1, o2) do {                               \
        v8 tr_ = T##r1 + T##r2, ti_ = T##i1 + T##i2;                          \
        v8 ur_ = T##r1 - T##r2, ui_ = T##i1 - T##i2;                          \
        v8 hr_ = T##r0 - 0.5 * tr_, hi_ = T##i0 - 0.5 * ti_;                  \
        STM(p, st, cp, MT, o0, T##r0 + tr_, T##i0 + ti_);                     \
        STM(p, st, cp, MT, o1, hr_ + K3 * ui_, hi_ - K3 * ur_);               \
        STM(p, st, cp, MT, o2, hr_ - K3 * ui_, hi_ + K3 * ur_);               \
    } while (0)

#define X4LD(T, p, st, a, b, c, d)                                            \
        v8 T##r0 = QR_(p, st, a), T##i0 = QI_(p, st, a);                      \
        v8 T##r1 = QR_(p, st, b), T##i1 = QI_(p, st, b);                      \
        v8 T##r2 = QR_(p, st, c), T##i2 = QI_(p, st, c);                      \
        v8 T##r3 = QR_(p, st, d), T##i3 = QI_(p, st, d);

#define D4CM(T, p, st, cp, MT, o0, o1, o2, o3) do {                           \
        v8 t0r = T##r0 + T##r2, t0i = T##i0 + T##i2;                          \
        v8 t1r = T##r0 - T##r2, t1i = T##i0 - T##i2;                          \
        v8 t2r = T##r1 + T##r3, t2i = T##i1 + T##i3;                          \
        v8 t3r = T##r1 - T##r3, t3i = T##i1 - T##i3;                          \
        STM(p, st, cp, MT, o0, t0r + t2r, t0i + t2i);                         \
        STM(p, st, cp, MT, o1, t1r + t3i, t1i - t3r);                         \
        STM(p, st, cp, MT, o2, t0r - t2r, t0i - t2i);                         \
        STM(p, st, cp, MT, o3, t1r - t3i, t1i + t3r);                         \
    } while (0)

/* Ship defaults = gen_batchlane gen_r9's ICL race verdicts (5/5 rounds each
 * at 10/15/20; 12 LOSES +3.5..4.7% there -- their negative result adopted
 * unre-measured, per their record's explicit instruction).  Wallaby (SPR)
 * advisory races this round: see the r10 strategy record. */
#ifndef SWAP10
#define SWAP10 1
#endif
#ifndef SWAP12
#define SWAP12 0
#endif
#ifndef SWAP15
#define SWAP15 1
#endif
#ifndef SWAP20
#define SWAP20 1
#endif
/* Map body+tail for the swapped codelets, MT<L> encoding (bit 0 = rcp tail,
 * bit 1 = bl body).  batchlane's swapped-form tails are div everywhere
 * (their register-swap 10 FLIPPED rcp->div; the codelet-local rule again).
 * MT10S=2 (bl body + div): on the SWAPPED 10 the r5 body verdict flips --
 * legacy body LOSES +2.3% to bl on SPR (0.926 vs 0.905, four interleaved
 * rounds), and bl+div is the exact codelet batchlane's ICL -2.2% was
 * measured on.  SWAP20: SPR FLIPS it (+1.6% there, ICL -1.4..1.6% win,
 * llvm-mca ICL wash) -- the SPR advisory race should build -DSWAP20=0. */
#ifndef MT10S
#define MT10S 2
#endif
#ifndef MT12S
#define MT12S 2
#endif
#ifndef MT15S
#define MT15S 2
#endif
#ifndef MT20S
#define MT20S 2
#endif

AIN void dft10_swm(double *restrict p, const ptrdiff_t st,
                   const double *restrict cp)
{
    enum { D5LIFT_ = LIFT5_10 };
    XDECL10;
    X5L(a_, p, st, 0, 2, 4, 6, 8);
    X5L(b_, p, st, 5, 7, 9, 1, 3);
    X2STM(p, st, cp, MT10S, 0, 5,  0, 5);
    X2STM(p, st, cp, MT10S, 2, 7,  6, 1);
    X2STM(p, st, cp, MT10S, 4, 9,  2, 7);
    X2STM(p, st, cp, MT10S, 6, 1,  8, 3);
    X2STM(p, st, cp, MT10S, 8, 3,  4, 9);
}

AIN void dft12_swm(double *restrict p, const ptrdiff_t st,
                   const double *restrict cp)
{
    XDECL12;
    X4L(p, st, 0, 3, 6, 9);
    X4L(p, st, 4, 7, 10, 1);
    X4L(p, st, 8, 11, 2, 5);
    X3STM(p, st, cp, MT12S, 0, 4, 8,   0, 4, 8);
    X3STM(p, st, cp, MT12S, 3, 7, 11,  9, 1, 5);
    X3STM(p, st, cp, MT12S, 6, 10, 2,  6, 10, 2);
    X3STM(p, st, cp, MT12S, 9, 1, 5,   3, 7, 11);
}

AIN void dft15_swm(double *restrict p, const ptrdiff_t st,
                   const double *restrict cp)
{
    enum { D5LIFT_ = LIFT5_15 };
    D5S(p, st, 0, 3, 6, 9, 12,   0, 3, 6, 9, 12);
    D5S(p, st, 5, 8, 11, 14, 2,  5, 8, 11, 14, 2);
    D5S(p, st, 10, 13, 1, 4, 7,  10, 13, 1, 4, 7);
    {
        X3LD(g0_, p, st, 0, 5, 10)
        D3CM(g0_, p, st, cp, MT15S, 0, 10, 5);
    }
    {
        X3LD(g2_, p, st, 6, 11, 1)
        {
            X3LD(g1_, p, st, 3, 8, 13)
            D3CM(g1_, p, st, cp, MT15S, 6, 1, 11);
        }
        X3LD(g4_, p, st, 12, 2, 7)
        D3CM(g2_, p, st, cp, MT15S, 12, 7, 2);
        X3LD(g3_, p, st, 9, 14, 4)
        D3CM(g4_, p, st, cp, MT15S, 9, 4, 14);
        D3CM(g3_, p, st, cp, MT15S, 3, 13, 8);
    }
}

AIN void dft20_swm(double *restrict p, const ptrdiff_t st,
                   const double *restrict cp)
{
    enum { D5LIFT_ = LIFT5_20 };
    D5S(p, st, 0, 4, 8, 12, 16,   0, 4, 8, 12, 16);
    D5S(p, st, 5, 9, 13, 17, 1,   5, 9, 13, 17, 1);
    D5S(p, st, 10, 14, 18, 2, 6,  10, 14, 18, 2, 6);
    D5S(p, st, 15, 19, 3, 7, 11,  15, 19, 3, 7, 11);
    {
        X4LD(h0_, p, st, 0, 5, 10, 15)
        D4CM(h0_, p, st, cp, MT20S, 0, 5, 10, 15);
    }
    {
        X4LD(h4_, p, st, 16, 1, 6, 11)
        {
            X4LD(h1_, p, st, 4, 9, 14, 19)
            D4CM(h1_, p, st, cp, MT20S, 16, 1, 6, 11);
        }
        D4CM(h4_, p, st, cp, MT20S, 4, 9, 14, 19);
    }
    {
        X4LD(h3_, p, st, 12, 17, 2, 7)
        {
            X4LD(h2_, p, st, 8, 13, 18, 3)
            D4CM(h2_, p, st, cp, MT20S, 12, 17, 2, 7);
        }
        D4CM(h3_, p, st, cp, MT20S, 8, 13, 18, 3);
    }
}

#if SWAP10
#define DFT10_XM dft10_swm
#else
#define DFT10_XM dft10_ipm
#endif
#if SWAP12
#define DFT12_XM dft12_swm
#else
#define DFT12_XM dft12_ipm
#endif
#if SWAP15
#define DFT15_XM dft15_swm
#else
#define DFT15_XM dft15_ipm
#endif
#if SWAP20
#define DFT20_XM dft20_swm
#else
#define DFT20_XM dft20_ipm
#endif

/* -------------------------------------------- the sweeps, one per L
 * PL = padded plane stride in sites; plane bytes == 256 (mod 4096). */

/* Per-function pre-RA scheduling.  10/12: kept from r3 (-0.4% at 12, ~0 at
 * 10).  15: OFF -- on the memory-form codelet it is a wash-to-loss (r2/r3;
 * re-checked in r4 on the rejected register-explicit 15, where it helped
 * ~1.5% but the form itself lost 12%).  20 stays default (both records
 * agree).  Knobs: -DPS_NOSCHED1012 strips 10/12, -DPS_SCHED15 enables 15,
 * for the monitor's cross-arch reruns. */
#define SCHEDP __attribute__((optimize("schedule-insns", "sched-pressure")))
#if !defined(PS_NOSCHED1012)
#define SCHED1012 SCHEDP
#else
#define SCHED1012
#endif
#if defined(PS_SCHED15)
#define SCHED15A SCHEDP
#else
#define SCHED15A
#endif

#define DEF_ENGINE(L, PLV, ATTR, PEN, PENM)                                   \
static ATTR void sweep_zy_##L(double *restrict pl)                            \
{                                                                             \
    for (int y = 0; y < (L); ++y)                                             \
        PEN(pl + (size_t)y * (L) * 16, 16);                                  \
    for (int z = 0; z < (L); ++z)                                             \
        PEN(pl + (size_t)z * 16, (ptrdiff_t)(L) * 16);                       \
}                                                                             \
static ATTR void soa_fft_##L(double *restrict S)                              \
{                                                                             \
    for (int x = 0; x < (L); ++x)                                             \
        sweep_zy_##L(S + (size_t)x * (PLV) * 16);                            \
    for (int c = 0; c < (L) * (L); ++c)                                       \
        PEN(S + (size_t)c * 16, (ptrdiff_t)(PLV) * 16);                     \
}                                                                             \
static ATTR void soa_step_##L(double *restrict S, const double *restrict C)   \
{                                                                             \
    for (int x = 0; x < (L); ++x)                                             \
        sweep_zy_##L(S + (size_t)x * (PLV) * 16);                            \
    for (int c = 0; c < (L) * (L); ++c)                                       \
        PENM(S + (size_t)c * 16, (ptrdiff_t)(PLV) * 16,                      \
             C + (size_t)c * 16);                                             \
}                                                                             \
static ATTR void soa_chain_##L(double *restrict S, const double *restrict C,  \
                               int m)                                         \
{                                                                             \
    for (int s = 0; s < m; ++s) {                                             \
        for (int x = 0; x < (L); ++x)                                         \
            sweep_zy_##L(S + (size_t)x * (PLV) * 16);                        \
        for (int c = 0; c < (L) * (L); ++c)                                   \
            PENM(S + (size_t)c * 16, (ptrdiff_t)(PLV) * 16,                  \
                 C + (size_t)c * 16);                                         \
    }                                                                         \
}

DEF_ENGINE(10, 130, SCHED1012, dft10_ip, DFT10_XM)
DEF_ENGINE(12, 162, SCHED1012, dft12_ip, DFT12_XM)
DEF_ENGINE(15, 226, SCHED15A,  DFT15_SWEEP, DFT15_XM)
DEF_ENGINE(20, 418, ,          dft20_ip, DFT20_XM)

static int plane_stride_sites(int L)   /* L^2 padded to == 2 (mod 32) sites */
{
    int pl = L * L;                    /* plane bytes == 256 (mod 4096)     */
    pl += ((2 - pl) % 32 + 32) % 32;   /* 130/162/226/418 at 10/12/15/20    */
    return pl;
}

/* --------------------------------------- split-path pencils (B%8, B=1)
 * The r1 buffered out-of-place pencils with the equivalent IN/OUT index
 * tables; used only by the ping-pong per-volume path. */

static const int IN10[2][5]  = {{0, 2, 4, 6, 8}, {5, 7, 9, 1, 3}};
static const int OUT10[2][5] = {{0, 6, 2, 8, 4}, {5, 1, 7, 3, 9}};

static const int IN12[4][3]  = {{0, 4, 8}, {3, 7, 11}, {6, 10, 2}, {9, 1, 5}};
static const int OUT12[4][3] = {{0, 4, 8}, {9, 1, 5}, {6, 10, 2}, {3, 7, 11}};

static const int IN15[3][5]  = {{0, 3, 6, 9, 12}, {5, 8, 11, 14, 2}, {10, 13, 1, 4, 7}};
static const int OUT15[3][5] = {{0, 6, 12, 3, 9}, {10, 1, 7, 13, 4}, {5, 11, 2, 8, 14}};

static const int IN20[4][5]  = {{0, 4, 8, 12, 16}, {5, 9, 13, 17, 1},
                                {10, 14, 18, 2, 6}, {15, 19, 3, 7, 11}};
static const int OUT20[4][5] = {{0, 16, 12, 8, 4}, {5, 1, 17, 13, 9},
                                {10, 6, 2, 18, 14}, {15, 11, 7, 3, 19}};

#define M_DFT2(r0, i0, r1, i1) do {                                   \
        v8 tr_ = r0 - r1, ti_ = i0 - i1;                              \
        r0 += r1; i0 += i1; r1 = tr_; i1 = ti_;                       \
    } while (0)

#define M_DFT3(r0, i0, r1, i1, r2, i2) do {                           \
        v8 ar_ = r1 + r2, ai_ = i1 + i2;                              \
        v8 dr_ = r1 - r2, di_ = i1 - i2;                              \
        v8 er_ = r0 - 0.5 * ar_, ei_ = i0 - 0.5 * ai_;                \
        r0 += ar_; i0 += ai_;                                         \
        r1 = er_ + K3 * di_; i1 = ei_ - K3 * dr_;                     \
        r2 = er_ - K3 * di_; i2 = ei_ + K3 * dr_;                     \
    } while (0)

#define M_DFT4(r0, i0, r1, i1, r2, i2, r3, i3) do {                   \
        v8 t0r_ = r0 + r2, t0i_ = i0 + i2;                            \
        v8 t1r_ = r0 - r2, t1i_ = i0 - i2;                            \
        v8 t2r_ = r1 + r3, t2i_ = i1 + i3;                            \
        v8 t3r_ = r1 - r3, t3i_ = i1 - i3;                            \
        r0 = t0r_ + t2r_; i0 = t0i_ + t2i_;                           \
        r2 = t0r_ - t2r_; i2 = t0i_ - t2i_;                           \
        r1 = t1r_ + t3i_; i1 = t1i_ - t3r_;                           \
        r3 = t1r_ - t3i_; i3 = t1i_ + t3r_;                           \
    } while (0)

/* r9: the same lifted v-pair for the split-path pencils (M_DFT5);
 * D5LIFT_ comes from the enclosing pencil function, like D5VPAIR. */
#define M5VPAIR(cr_, ci_, dr_, di_)                                   \
        v8 o1r_, o1i_, o2r_, o2i_;                                    \
        if (D5LIFT_) {                                                \
            v8 ulr_ = cr_ - PHI5 * dr_, uli_ = ci_ - PHI5 * di_;      \
            o2r_ = S52 * ulr_;            o2i_ = S52 * uli_;          \
            o1r_ = S51 * ulr_ + KL5 * dr_;                            \
            o1i_ = S51 * uli_ + KL5 * di_;                            \
        } else {                                                      \
            o1r_ = S51 * cr_ + S52 * dr_; o1i_ = S51 * ci_ + S52 * di_;\
            o2r_ = S52 * cr_ - S51 * dr_; o2i_ = S52 * ci_ - S51 * di_;\
        }

#define M_DFT5(r0, i0, r1, i1, r2, i2, r3, i3, r4, i4) do {           \
        v8 ar_ = r1 + r4, ai_ = i1 + i4;                              \
        v8 cr_ = r1 - r4, ci_ = i1 - i4;                              \
        v8 br_ = r2 + r3, bi_ = i2 + i3;                              \
        v8 dr_ = r2 - r3, di_ = i2 - i3;                              \
        v8 pr_ = ar_ + br_, pi_ = ai_ + bi_;                          \
        v8 qr_ = ar_ - br_, qi_ = ai_ - bi_;                          \
        v8 fr_ = r0 - K25 * pr_, fi_ = i0 - K25 * pi_;                \
        v8 e1r_ = fr_ + KQ5 * qr_, e1i_ = fi_ + KQ5 * qi_;            \
        v8 e2r_ = fr_ - KQ5 * qr_, e2i_ = fi_ - KQ5 * qi_;            \
        r0 += pr_; i0 += pi_;                                         \
        M5VPAIR(cr_, ci_, dr_, di_)                                   \
        r1 = e1r_ + o1i_; i1 = e1i_ - o1r_;                           \
        r4 = e1r_ - o1i_; i4 = e1i_ + o1r_;                           \
        r2 = e2r_ + o2i_; i2 = e2i_ - o2r_;                           \
        r3 = e2r_ - o2i_; i3 = e2i_ + o2r_;                           \
    } while (0)

static inline void pencil10(const double *sr, const double *si,
                            double *dr, double *di, ptrdiff_t s)
{
    enum { D5LIFT_ = LIFT5_10 };
    v8 tr[10], ti[10];
    for (int j2 = 0; j2 < 5; ++j2) {
        v8 r0 = vload(sr + IN10[0][j2] * s), i0 = vload(si + IN10[0][j2] * s);
        v8 r1 = vload(sr + IN10[1][j2] * s), i1 = vload(si + IN10[1][j2] * s);
        M_DFT2(r0, i0, r1, i1);
        tr[j2] = r0; ti[j2] = i0; tr[5 + j2] = r1; ti[5 + j2] = i1;
    }
    for (int k1 = 0; k1 < 2; ++k1) {
        v8 r0 = tr[5 * k1 + 0], i0 = ti[5 * k1 + 0];
        v8 r1 = tr[5 * k1 + 1], i1 = ti[5 * k1 + 1];
        v8 r2 = tr[5 * k1 + 2], i2 = ti[5 * k1 + 2];
        v8 r3 = tr[5 * k1 + 3], i3 = ti[5 * k1 + 3];
        v8 r4 = tr[5 * k1 + 4], i4 = ti[5 * k1 + 4];
        M_DFT5(r0, i0, r1, i1, r2, i2, r3, i3, r4, i4);
        vstore(dr + OUT10[k1][0] * s, r0); vstore(di + OUT10[k1][0] * s, i0);
        vstore(dr + OUT10[k1][1] * s, r1); vstore(di + OUT10[k1][1] * s, i1);
        vstore(dr + OUT10[k1][2] * s, r2); vstore(di + OUT10[k1][2] * s, i2);
        vstore(dr + OUT10[k1][3] * s, r3); vstore(di + OUT10[k1][3] * s, i3);
        vstore(dr + OUT10[k1][4] * s, r4); vstore(di + OUT10[k1][4] * s, i4);
    }
}

static inline void pencil12(const double *sr, const double *si,
                            double *dr, double *di, ptrdiff_t s)
{
    v8 tr[12], ti[12];
    for (int j2 = 0; j2 < 3; ++j2) {
        v8 r0 = vload(sr + IN12[0][j2] * s), i0 = vload(si + IN12[0][j2] * s);
        v8 r1 = vload(sr + IN12[1][j2] * s), i1 = vload(si + IN12[1][j2] * s);
        v8 r2 = vload(sr + IN12[2][j2] * s), i2 = vload(si + IN12[2][j2] * s);
        v8 r3 = vload(sr + IN12[3][j2] * s), i3 = vload(si + IN12[3][j2] * s);
        M_DFT4(r0, i0, r1, i1, r2, i2, r3, i3);
        tr[j2] = r0; ti[j2] = i0; tr[3 + j2] = r1; ti[3 + j2] = i1;
        tr[6 + j2] = r2; ti[6 + j2] = i2; tr[9 + j2] = r3; ti[9 + j2] = i3;
    }
    for (int k1 = 0; k1 < 4; ++k1) {
        v8 r0 = tr[3 * k1 + 0], i0 = ti[3 * k1 + 0];
        v8 r1 = tr[3 * k1 + 1], i1 = ti[3 * k1 + 1];
        v8 r2 = tr[3 * k1 + 2], i2 = ti[3 * k1 + 2];
        M_DFT3(r0, i0, r1, i1, r2, i2);
        vstore(dr + OUT12[k1][0] * s, r0); vstore(di + OUT12[k1][0] * s, i0);
        vstore(dr + OUT12[k1][1] * s, r1); vstore(di + OUT12[k1][1] * s, i1);
        vstore(dr + OUT12[k1][2] * s, r2); vstore(di + OUT12[k1][2] * s, i2);
    }
}

static inline void pencil15(const double *sr, const double *si,
                            double *dr, double *di, ptrdiff_t s)
{
    enum { D5LIFT_ = LIFT5_15 };
    v8 tr[15], ti[15];
    for (int j2 = 0; j2 < 5; ++j2) {
        v8 r0 = vload(sr + IN15[0][j2] * s), i0 = vload(si + IN15[0][j2] * s);
        v8 r1 = vload(sr + IN15[1][j2] * s), i1 = vload(si + IN15[1][j2] * s);
        v8 r2 = vload(sr + IN15[2][j2] * s), i2 = vload(si + IN15[2][j2] * s);
        M_DFT3(r0, i0, r1, i1, r2, i2);
        tr[j2] = r0; ti[j2] = i0; tr[5 + j2] = r1; ti[5 + j2] = i1;
        tr[10 + j2] = r2; ti[10 + j2] = i2;
    }
    for (int k1 = 0; k1 < 3; ++k1) {
        v8 r0 = tr[5 * k1 + 0], i0 = ti[5 * k1 + 0];
        v8 r1 = tr[5 * k1 + 1], i1 = ti[5 * k1 + 1];
        v8 r2 = tr[5 * k1 + 2], i2 = ti[5 * k1 + 2];
        v8 r3 = tr[5 * k1 + 3], i3 = ti[5 * k1 + 3];
        v8 r4 = tr[5 * k1 + 4], i4 = ti[5 * k1 + 4];
        M_DFT5(r0, i0, r1, i1, r2, i2, r3, i3, r4, i4);
        vstore(dr + OUT15[k1][0] * s, r0); vstore(di + OUT15[k1][0] * s, i0);
        vstore(dr + OUT15[k1][1] * s, r1); vstore(di + OUT15[k1][1] * s, i1);
        vstore(dr + OUT15[k1][2] * s, r2); vstore(di + OUT15[k1][2] * s, i2);
        vstore(dr + OUT15[k1][3] * s, r3); vstore(di + OUT15[k1][3] * s, i3);
        vstore(dr + OUT15[k1][4] * s, r4); vstore(di + OUT15[k1][4] * s, i4);
    }
}

static inline void pencil20(const double *sr, const double *si,
                            double *dr, double *di, ptrdiff_t s)
{
    enum { D5LIFT_ = LIFT5_20 };
    v8 tr[20], ti[20];
    for (int j2 = 0; j2 < 5; ++j2) {
        v8 r0 = vload(sr + IN20[0][j2] * s), i0 = vload(si + IN20[0][j2] * s);
        v8 r1 = vload(sr + IN20[1][j2] * s), i1 = vload(si + IN20[1][j2] * s);
        v8 r2 = vload(sr + IN20[2][j2] * s), i2 = vload(si + IN20[2][j2] * s);
        v8 r3 = vload(sr + IN20[3][j2] * s), i3 = vload(si + IN20[3][j2] * s);
        M_DFT4(r0, i0, r1, i1, r2, i2, r3, i3);
        tr[j2] = r0; ti[j2] = i0; tr[5 + j2] = r1; ti[5 + j2] = i1;
        tr[10 + j2] = r2; ti[10 + j2] = i2; tr[15 + j2] = r3; ti[15 + j2] = i3;
    }
    for (int k1 = 0; k1 < 4; ++k1) {
        v8 r0 = tr[5 * k1 + 0], i0 = ti[5 * k1 + 0];
        v8 r1 = tr[5 * k1 + 1], i1 = ti[5 * k1 + 1];
        v8 r2 = tr[5 * k1 + 2], i2 = ti[5 * k1 + 2];
        v8 r3 = tr[5 * k1 + 3], i3 = ti[5 * k1 + 3];
        v8 r4 = tr[5 * k1 + 4], i4 = ti[5 * k1 + 4];
        M_DFT5(r0, i0, r1, i1, r2, i2, r3, i3, r4, i4);
        vstore(dr + OUT20[k1][0] * s, r0); vstore(di + OUT20[k1][0] * s, i0);
        vstore(dr + OUT20[k1][1] * s, r1); vstore(di + OUT20[k1][1] * s, i1);
        vstore(dr + OUT20[k1][2] * s, r2); vstore(di + OUT20[k1][2] * s, i2);
        vstore(dr + OUT20[k1][3] * s, r3); vstore(di + OUT20[k1][3] * s, i3);
        vstore(dr + OUT20[k1][4] * s, r4); vstore(di + OUT20[k1][4] * s, i4);
    }
}

/* --------------------------- per-volume split-complex path (B=1, B%8)
 * Ping-pong passes S->D, D->S, S->D; lanes are 8 consecutive inner points,
 * tails handled by overlapped (idempotent, out-of-place) chunks.  The z
 * pass turns lanes into y via in-register 8x8 transposes. */

#define DEF_SPLIT(L)                                                            \
static void split_fft_##L(double *Sr, double *Si, double *Dr, double *Di)       \
{                                                                               \
    /* x pass: lanes = flat (y,z) index, stride L^2 */                          \
    for (int b = 0; b < (L) * (L); b += 8) {                                    \
        int o = (b + 8 <= (L) * (L)) ? b : (L) * (L) - 8;                       \
        pencil##L(Sr + o, Si + o, Dr + o, Di + o, (ptrdiff_t)(L) * (L));        \
    }                                                                           \
    /* y pass: per x slab, lanes = z chunk, stride L */                         \
    for (int x = 0; x < (L); ++x) {                                             \
        ptrdiff_t xo = (ptrdiff_t)x * (L) * (L);                                \
        for (int z = 0; z < (L); z += 8) {                                      \
            int o = (z + 8 <= (L)) ? z : (L) - 8;                               \
            pencil##L(Dr + xo + o, Di + xo + o, Sr + xo + o, Si + xo + o,       \
                      (ptrdiff_t)(L));                                          \
        }                                                                       \
    }                                                                           \
    /* z pass: per x slab, lanes = y via 8x8 transposes */                      \
    for (int x = 0; x < (L); ++x) {                                             \
        ptrdiff_t xo = (ptrdiff_t)x * (L) * (L);                                \
        for (int y = 0; y < (L); y += 8) {                                      \
            int yo = (y + 8 <= (L)) ? y : (L) - 8;                              \
            v8 pzr[L], pzi[L], por[L], poi[L];                                  \
            for (int z = 0; z < (L); z += 8) {                                  \
                int zo = (z + 8 <= (L)) ? z : (L) - 8;                          \
                v8 blk[8];                                                      \
                for (int q = 0; q < 8; ++q)                                     \
                    blk[q] = vload(Sr + xo + (ptrdiff_t)(yo + q) * (L) + zo);   \
                tr8(blk);                                                       \
                for (int q = 0; q < 8; ++q) pzr[zo + q] = blk[q];               \
                for (int q = 0; q < 8; ++q)                                     \
                    blk[q] = vload(Si + xo + (ptrdiff_t)(yo + q) * (L) + zo);   \
                tr8(blk);                                                       \
                for (int q = 0; q < 8; ++q) pzi[zo + q] = blk[q];               \
            }                                                                   \
            pencil##L((const double *)pzr, (const double *)pzi,                 \
                      (double *)por, (double *)poi, 8);                         \
            for (int z = 0; z < (L); z += 8) {                                  \
                int zo = (z + 8 <= (L)) ? z : (L) - 8;                          \
                v8 blk[8];                                                      \
                for (int q = 0; q < 8; ++q) blk[q] = por[zo + q];               \
                tr8(blk);                                                       \
                for (int q = 0; q < 8; ++q)                                     \
                    vstore(Dr + xo + (ptrdiff_t)(yo + q) * (L) + zo, blk[q]);   \
                for (int q = 0; q < 8; ++q) blk[q] = poi[zo + q];               \
                tr8(blk);                                                       \
                for (int q = 0; q < 8; ++q)                                     \
                    vstore(Di + xo + (ptrdiff_t)(yo + q) * (L) + zo, blk[q]);   \
            }                                                                   \
        }                                                                       \
    }                                                                           \
}

DEF_SPLIT(10)
DEF_SPLIT(12)
DEF_SPLIT(15)
DEF_SPLIT(20)

/* ------------------------- r7: fused-map split chain step (B=1, B%8)
 * Three passes, ping-pong S->D, D->S, S->D, with the graded map FUSED into
 * the last pass's stores (the r2 lesson -- a separate map pass is a full
 * state+c round trip -- finally applied to the split path).  The last pass
 * transforms the STRIDE-1 dim; SPLITZ<L> picks its form:
 *
 *   1 = HALF-TURN: transpose IN only (tr8 8x8 blocks), pencil on the
 *       buffered lanes, then PLAIN contiguous stores with the two inner
 *       dims SWAPPED -- the back-transposes of the r1-r6 sandwich are gone
 *       (half the port-5 bill).  Each step flips the inner-dim parity; the
 *       chain alternates a transposed c copy and un-permutes once at the
 *       end.  Values bit-identical to the sandwich (same pencil, same
 *       ladder).
 *   2 = DENSE stage-matrix broadcast-FMA pencil (literature 11 Tier 2,
 *       stage-as-outer-product; first x86 test of the "dense GEMM wins at
 *       L<=16" crossover claim): per pencil, broadcast each input scalar
 *       and FMA rows of the compiled DFT matrix (create()-time long-double
 *       tables, dead columns zero), masked stores, map fused, NO shuffles
 *       and no layout turn.  4*L*ceil(L/8) FMAs/pencil vs the PFA's
 *       ~11-27 -- the claim only has a chance because the sandwich pays
 *       port-5, not FMA.
 *   0 = r6 behavior (sandwich split_fft + separate map_span), kept for
 *       the cross-arch races.
 * Pass-2 lane waste (ceil(L/8)*8/L) is unchanged this round. */

#if defined(__AVX512F__)
static inline __attribute__((always_inline)) void
dense3_slab(const double *restrict Sr, const double *restrict Si,
            double *restrict Dr, double *restrict Di,
            const double *restrict cr, const double *restrict ci,
            const double *restrict wtr, const double *restrict wti,
            const int L, const int NV)
{
    for (int a = 0; a < L; ++a) {
        const double *br = Sr + (ptrdiff_t)a * L;
        const double *bi = Si + (ptrdiff_t)a * L;
        v8 accr[3], acci[3];
        for (int v = 0; v < NV; ++v) {
            accr[v] = (v8){0, 0, 0, 0, 0, 0, 0, 0};
            acci[v] = accr[v];
        }
        for (int n = 0; n < L; ++n) {
            const v8 xrb = (v8)_mm512_set1_pd(br[n]);
            const v8 xib = (v8)_mm512_set1_pd(bi[n]);
            const double *wr = wtr + (size_t)n * NV * 8;
            const double *wi = wti + (size_t)n * NV * 8;
            for (int v = 0; v < NV; ++v) {
                v8 wrv = vload(wr + 8 * v), wiv = vload(wi + 8 * v);
                accr[v] += xrb * wrv;
                accr[v] -= xib * wiv;
                acci[v] += xrb * wiv;
                acci[v] += xib * wrv;
            }
        }
        for (int v = 0; v < NV; ++v) {
            int rem = L - 8 * v;
            __mmask8 mk = rem >= 8 ? (__mmask8)0xFF
                                   : (__mmask8)((1u << rem) - 1u);
            v8 crv = (v8)_mm512_maskz_loadu_pd(mk, cr + (ptrdiff_t)a * L + 8 * v);
            v8 civ = (v8)_mm512_maskz_loadu_pd(mk, ci + (ptrdiff_t)a * L + 8 * v);
            v8 zr = accr[v], zi = acci[v];
            map8c(&zr, &zi, crv, civ);
            _mm512_mask_storeu_pd(Dr + (ptrdiff_t)a * L + 8 * v, mk, (__m512d)zr);
            _mm512_mask_storeu_pd(Di + (ptrdiff_t)a * L + 8 * v, mk, (__m512d)zi);
        }
    }
}
#else
static inline void
dense3_slab(const double *restrict Sr, const double *restrict Si,
            double *restrict Dr, double *restrict Di,
            const double *restrict cr, const double *restrict ci,
            const double *restrict wtr, const double *restrict wti,
            const int L, const int NV)
{
    for (int a = 0; a < L; ++a)
        for (int k = 0; k < L; ++k) {
            double yr = 0, yi = 0;
            for (int n = 0; n < L; ++n) {
                double wr = wtr[((size_t)n * NV) * 8 + k];
                double wi = wti[((size_t)n * NV) * 8 + k];
                yr += Sr[(ptrdiff_t)a * L + n] * wr - Si[(ptrdiff_t)a * L + n] * wi;
                yi += Sr[(ptrdiff_t)a * L + n] * wi + Si[(ptrdiff_t)a * L + n] * wr;
            }
            double aa = yr + cr[(ptrdiff_t)a * L + k];
            double bb = yi + ci[(ptrdiff_t)a * L + k];
            double sc = 1.0 / (1.0 + sqrt(aa * aa + bb * bb));
            Dr[(ptrdiff_t)a * L + k] = aa * sc;
            Di[(ptrdiff_t)a * L + k] = bb * sc;
        }
}
#endif

/* Per-size pass-3 selection: 1 = half-turn, 2 = dense, 0 = r6 sandwich +
 * map_span (raced same-core on wallaby; override for the cross-arch runs). */
#ifndef SPLITZ10
#define SPLITZ10 1
#endif
#ifndef SPLITZ12
#define SPLITZ12 1
#endif
#ifndef SPLITZ15
#define SPLITZ15 1
#endif
#ifndef SPLITZ20
#define SPLITZ20 1
#endif

#define DEF_STEP(L)                                                            \
static void step_##L(double *restrict Sr, double *restrict Si,                 \
                     double *restrict Dr, double *restrict Di,                 \
                     const double *restrict cr, const double *restrict ci,     \
                     const double *restrict wtr, const double *restrict wti)   \
{                                                                              \
    /* pass 1: stride-L^2 dim, lanes = flat 8 of the inner L^2 */              \
    for (int b = 0; b < (L) * (L); b += 8) {                                   \
        int o = (b + 8 <= (L) * (L)) ? b : (L) * (L) - 8;                      \
        pencil##L(Sr + o, Si + o, Dr + o, Di + o, (ptrdiff_t)(L) * (L));       \
    }                                                                          \
    /* pass 2: stride-L dim, lanes = 8 contiguous stride-1 positions */        \
    for (int x = 0; x < (L); ++x) {                                            \
        ptrdiff_t xo = (ptrdiff_t)x * (L) * (L);                               \
        for (int z = 0; z < (L); z += 8) {                                     \
            int o = (z + 8 <= (L)) ? z : (L) - 8;                              \
            pencil##L(Dr + xo + o, Di + xo + o, Sr + xo + o, Si + xo + o,      \
                      (ptrdiff_t)(L));                                         \
        }                                                                      \
    }                                                                          \
    /* pass 3: stride-1 dim, map fused into the stores.  Half-turn ROTATES
     * the layout ([P0][P1][P2] -> [P2][P0][P1]): the transform's rows are
     * uniformly strided over the WHOLE volume (R = 0..L^2-1 at stride L),
     * so lane blocks are 8 consecutive rows regardless of slab boundaries
     * -- ceil(L^2/8) blocks with ONE overlap tail instead of L*ceil(L/8)
     * per-slab blocks (13 vs 20 at L=10), and the rotated store lanes
     * (k*L^2 + R) stay contiguous across slab crossings. */                   \
    if (SPLITZ##L == 2) {                                                      \
        for (int x = 0; x < (L); ++x) {                                        \
            ptrdiff_t xo = (ptrdiff_t)x * (L) * (L);                           \
            dense3_slab(Sr + xo, Si + xo, Dr + xo, Di + xo,                    \
                        cr + xo, ci + xo, wtr, wti, (L), ((L) + 7) / 8);       \
        }                                                                      \
    } else {                                                                   \
        for (int rb = 0; rb < (L) * (L); rb += 8) {                            \
            int R0 = (rb + 8 <= (L) * (L)) ? rb : (L) * (L) - 8;               \
            v8 pzr[L], pzi[L], por[L], poi[L];                                 \
            for (int z = 0; z < (L); z += 8) {                                 \
                int zo = (z + 8 <= (L)) ? z : (L) - 8;                         \
                v8 blk[8];                                                     \
                for (int q = 0; q < 8; ++q)                                    \
                    blk[q] = vload(Sr + (ptrdiff_t)(R0 + q) * (L) + zo);       \
                tr8(blk);                                                      \
                for (int q = 0; q < 8; ++q) pzr[zo + q] = blk[q];              \
                for (int q = 0; q < 8; ++q)                                    \
                    blk[q] = vload(Si + (ptrdiff_t)(R0 + q) * (L) + zo);       \
                tr8(blk);                                                      \
                for (int q = 0; q < 8; ++q) pzi[zo + q] = blk[q];              \
            }                                                                  \
            pencil##L((const double *)pzr, (const double *)pzi,                \
                      (double *)por, (double *)poi, 8);                        \
            for (int k = 0; k < (L); ++k) {                                    \
                v8 zr = por[k], zi = poi[k];                                   \
                map8c(&zr, &zi,                                                \
                      vload(cr + (ptrdiff_t)k * (L) * (L) + R0),               \
                      vload(ci + (ptrdiff_t)k * (L) * (L) + R0));              \
                vstore(Dr + (ptrdiff_t)k * (L) * (L) + R0, zr);                \
                vstore(Di + (ptrdiff_t)k * (L) * (L) + R0, zi);                \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

DEF_STEP(10)
DEF_STEP(12)
DEF_STEP(15)
DEF_STEP(20)

/* --------------------------------------------- generic coprime-pair engine
 * Round-3 class duty: accept ANY small coprime-pair composite the driver
 * asks for.  L = P*Q, gcd(P,Q)=1, modules in {2,3,4,5,7,8,9}: covers
 * 6,14,18,21,24,28,35,36,45,56,63 beyond the four tuned sizes.  Same
 * Good-Thomas maps as the tuned codelets, but slot tables are built at
 * create() and the pencil is BUFFERED (whole pencil in v8 temps), which is
 * in-place safe for any pair -- no Q == 1 mod P constraint.  Runs on the
 * same padded SoA-8 arena with the same fused map; remainder volumes
 * (B % 8, incl. B = 1) replicate the last volume into dead lanes
 * (gen_batchlane gen_r1's scheme: correct, pays up to 8x on that group).
 * Odd-module constants are computed at create() in long double (the
 * brief's twiddle-exactness rule); 2/4/8 use exact +-1, +-i, sqrt(1/2). */

#define GMAXL 126

#define K8 0.70710678118654752440   /* sqrt(1/2) */
#define C16 0.92387953251128675613  /* cos(pi/8) */
#define S16 0.38268343236508977173  /* sin(pi/8) */

/* n-point DFT, n odd (3..31), conjugate-pair fold, split complex on v8
 * lanes; cs/sn are the h*h cos/sin tables, h = n/2, row k-1, col j-1. */
static inline __attribute__((always_inline)) void
gdftodd(v8 *xr, v8 *xi, int n, const double *cs, const double *sn)
{
    int h = n >> 1;
    v8 ar[15], ai[15], sr[15], si[15];
    for (int j = 1; j <= h; ++j) {
        ar[j-1] = xr[j] + xr[n-j];  ai[j-1] = xi[j] + xi[n-j];
        sr[j-1] = xr[j] - xr[n-j];  si[j-1] = xi[j] - xi[n-j];
    }
    v8 x0r = xr[0], x0i = xi[0];
    v8 X0r = x0r, X0i = x0i;
    for (int j = 0; j < h; ++j) { X0r += ar[j]; X0i += ai[j]; }
    for (int k = 1; k <= h; ++k) {
        const double *c = cs + (size_t)(k - 1) * h;
        const double *s = sn + (size_t)(k - 1) * h;
        v8 Cr = x0r + c[0] * ar[0], Ci = x0i + c[0] * ai[0];
        v8 Sr = s[0] * sr[0],       Si = s[0] * si[0];
        for (int j = 1; j < h; ++j) {
            Cr += c[j] * ar[j];  Ci += c[j] * ai[j];
            Sr += s[j] * sr[j];  Si += s[j] * si[j];
        }
        xr[k]     = Cr + Si;  xi[k]     = Ci - Sr;
        xr[n - k] = Cr - Si;  xi[n - k] = Ci + Sr;
    }
    xr[0] = X0r;  xi[0] = X0i;
}

/* 8-point: two DFT4s + W8 twiddle combine, exact constants only. */
static inline __attribute__((always_inline)) void gdft8(v8 *xr, v8 *xi)
{
    v8 e0r = xr[0], e0i = xi[0], e1r = xr[2], e1i = xi[2];
    v8 e2r = xr[4], e2i = xi[4], e3r = xr[6], e3i = xi[6];
    M_DFT4(e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i);
    v8 o0r = xr[1], o0i = xi[1], o1r = xr[3], o1i = xi[3];
    v8 o2r = xr[5], o2i = xi[5], o3r = xr[7], o3i = xi[7];
    M_DFT4(o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i);
    v8 t1r = K8 * (o1r + o1i), t1i = K8 * (o1i - o1r);
    v8 t2r = o2i,              t2i = -o2r;
    v8 t3r = K8 * (o3i - o3r), t3i = -(K8 * (o3r + o3i));
    xr[0] = e0r + o0r;  xi[0] = e0i + o0i;
    xr[4] = e0r - o0r;  xi[4] = e0i - o0i;
    xr[1] = e1r + t1r;  xi[1] = e1i + t1i;
    xr[5] = e1r - t1r;  xi[5] = e1i - t1i;
    xr[2] = e2r + t2r;  xi[2] = e2i + t2i;
    xr[6] = e2r - t2r;  xi[6] = e2i - t2i;
    xr[3] = e3r + t3r;  xi[3] = e3i + t3i;
    xr[7] = e3r - t3r;  xi[7] = e3i - t3i;
}

/* 16-point (r6): two natural-order gdft8 halves + W16 combine.  All
 * constants exact literals (1, 0, sqrt(1/2), cos/sin(pi/8)); with the loop
 * unrolled the k=0/2/4/6 rotations constant-fold to trivial forms. */
static inline __attribute__((always_inline)) void gdft16(v8 *xr, v8 *xi)
{
    v8 er[8], ei[8], odr[8], odi[8];
    for (int j = 0; j < 8; ++j) {
        er[j]  = xr[2 * j];      ei[j]  = xi[2 * j];
        odr[j] = xr[2 * j + 1];  odi[j] = xi[2 * j + 1];
    }
    gdft8(er, ei);
    gdft8(odr, odi);
    static const double wc[8] = { 1.0, C16, K8, S16, 0.0, -S16, -K8, -C16 };
    static const double ws[8] = { 0.0, S16, K8, C16, 1.0,  C16,  K8,  S16 };
    for (int k = 0; k < 8; ++k) {   /* w16^k = (wc[k], -ws[k]) */
        v8 tr = wc[k] * odr[k] + ws[k] * odi[k];
        v8 ti = wc[k] * odi[k] - ws[k] * odr[k];
        xr[k]     = er[k] + tr;  xi[k]     = ei[k] + ti;
        xr[k + 8] = er[k] - tr;  xi[k + 8] = ei[k] - ti;
    }
}

/* r11: module 25 as a twiddled 5x5 Cooley-Tukey (BORROWED: gen_pfa_large
 * gen_r1's DFT25 -- the decomposition X[k1+5k2] = DFT5_{n2}(W25^{n2 k1} *
 * DFT5_m(x[n2+5m])[k1]) with stage A storing U[5*k1+n2] so stage B reads 5
 * contiguous slots; 16 nontrivial twiddles).  Split-complex v8 form with
 * the lifted M_DFT5: ~404 vector FP vs the h=12 fold's ~650 (a 1.6x cut --
 * the r6 module-15 lesson says that ratio is marginal, so the fold stays
 * one knob away: -DG25CT=0).  This is the class's first twiddled stage;
 * tables tc/ts computed long double at create() in consumption order
 * ([n2*5+k1]), per the brief's twiddle-exactness rule.
 * W25^{n2 k1} = c - i*s:  r' = c*r + s*i,  i' = c*i - s*r. */
#ifndef G25CT
#define G25CT 1
#endif
static inline __attribute__((always_inline)) void
gdft25(v8 *xr, v8 *xi, const double *restrict tc, const double *restrict ts)
{
    enum { D5LIFT_ = LIFT5 };
    v8 ur[25], ui[25];
#pragma GCC unroll 5
    for (int n2 = 0; n2 < 5; ++n2) {
        v8 r0 = xr[n2],      i0 = xi[n2];
        v8 r1 = xr[n2 + 5],  i1 = xi[n2 + 5];
        v8 r2 = xr[n2 + 10], i2 = xi[n2 + 10];
        v8 r3 = xr[n2 + 15], i3 = xi[n2 + 15];
        v8 r4 = xr[n2 + 20], i4 = xi[n2 + 20];
        M_DFT5(r0, i0, r1, i1, r2, i2, r3, i3, r4, i4);
        ur[n2] = r0;  ui[n2] = i0;
        if (n2 == 0) {
            ur[5]  = r1;  ui[5]  = i1;
            ur[10] = r2;  ui[10] = i2;
            ur[15] = r3;  ui[15] = i3;
            ur[20] = r4;  ui[20] = i4;
        } else {
#define GTW25(k1, rr, ii) do {                                                \
            const double c_ = tc[n2 * 5 + (k1)], s_ = ts[n2 * 5 + (k1)];      \
            ur[5 * (k1) + n2] = c_ * (rr) + s_ * (ii);                        \
            ui[5 * (k1) + n2] = c_ * (ii) - s_ * (rr);                        \
        } while (0)
            GTW25(1, r1, i1);
            GTW25(2, r2, i2);
            GTW25(3, r3, i3);
            GTW25(4, r4, i4);
#undef GTW25
        }
    }
#pragma GCC unroll 5
    for (int k1 = 0; k1 < 5; ++k1) {
        v8 r0 = ur[5 * k1],     i0 = ui[5 * k1];
        v8 r1 = ur[5 * k1 + 1], i1 = ui[5 * k1 + 1];
        v8 r2 = ur[5 * k1 + 2], i2 = ui[5 * k1 + 2];
        v8 r3 = ur[5 * k1 + 3], i3 = ui[5 * k1 + 3];
        v8 r4 = ur[5 * k1 + 4], i4 = ui[5 * k1 + 4];
        M_DFT5(r0, i0, r1, i1, r2, i2, r3, i3, r4, i4);
        xr[k1]      = r0;  xi[k1]      = i0;
        xr[k1 + 5]  = r1;  xi[k1 + 5]  = i1;
        xr[k1 + 10] = r2;  xi[k1 + 10] = i2;
        xr[k1 + 15] = r3;  xi[k1 + 15] = i3;
        xr[k1 + 20] = r4;  xi[k1 + 20] = i4;
    }
}

static inline __attribute__((always_inline)) void
gmod(v8 *xr, v8 *xi, int n, const double *cs, const double *sn)
{
    switch (n) {
    case 2:
        M_DFT2(xr[0], xi[0], xr[1], xi[1]);
        break;
    case 4:
        M_DFT4(xr[0], xi[0], xr[1], xi[1], xr[2], xi[2], xr[3], xi[3]);
        break;
    case 8:
        gdft8(xr, xi);
        break;
    case 16:
        gdft16(xr, xi);
        break;
    default:
        gdftodd(xr, xi, n, cs, sn);
    }
}

struct gtabs {
    int P, Q;
    int16_t inmap[GMAXL];   /* [j2*P + j1] = (Q*j1 + P*j2) mod L  */
    int16_t outmap[GMAXL];  /* [k1*Q + j2] = (A*k1 + B*j2) mod L  */
    double csP[225], snP[225], csQ[225], snQ[225];  /* h*h, h <= 15 */
    /* r6: composite odd module Q = q1*q2 coprime runs as a NESTED
     * twiddle-free GT-PFA (qin/qout are the module-internal maps); the
     * O(h^2) fold for DFT15/21/... is 2-3x the ops of its PFA form. */
    int q1, q2;             /* 0 when Q is not split                */
    int16_t qin[63], qout[63];
    double csq1[36], snq1[36], csq2[81], snq2[81];  /* q1 <= 7, q2 <= 19 */
    double t25c[25], t25s[25]; /* r11: W25 twiddles, [n2*5+k1], long-double */
};

/* Composite odd module as nested PFA: natural-order in/out DFT_n on the
 * caller's slot arrays, n = q1*q2 coprime, sub-modules through gmod (odd
 * fold / exact kernels).  Stage 1 buffers fully, so the in-place stage-2
 * scatter through qout is hazard-free. */
static inline __attribute__((always_inline)) void
gmodpfa(v8 *xr, v8 *xi, const struct gtabs *g, const int q1, const int q2)
{
    v8 tr[63], ti[63], yr[13], yi[13];
    for (int b = 0; b < q2; ++b) {
        const int16_t *im = g->qin + (size_t)b * q1;
        for (int a = 0; a < q1; ++a) { yr[a] = xr[im[a]]; yi[a] = xi[im[a]]; }
        gmod(yr, yi, q1, g->csq1, g->snq1);
        for (int a = 0; a < q1; ++a) {
            tr[(size_t)a * q2 + b] = yr[a];
            ti[(size_t)a * q2 + b] = yi[a];
        }
    }
    for (int a = 0; a < q1; ++a) {   /* stage 2 in place on the tr rows */
        gmod(tr + (size_t)a * q2, ti + (size_t)a * q2, q2,
             g->csq2, g->snq2);
        const int16_t *om = g->qout + (size_t)a * q2;
        for (int b = 0; b < q2; ++b) {
            xr[om[b]] = tr[(size_t)a * q2 + b];
            xi[om[b]] = ti[(size_t)a * q2 + b];
        }
    }
}

/* Coprime split of the composite odd modules; 0 = not split (fold/exact).
 * Module 15 defaults to the FOLD: the nested form's op cut (~450 -> ~280
 * incl. buffer moves) loses to the fold's straight-line FMA stream, raced
 * same-core at 30 (+10%), 60/105/120 (+2-3%).  Module 21's bigger cut
 * (~850 -> ~530) wins (-12% at 42, -11% at 84).  -DGM15PFA=1 re-enables
 * the 15-split for the cross-arch races. */
#ifndef GM15PFA
#define GM15PFA 0
#endif
static int gsplit(int n, int *a, int *b)
{
    switch (n) {
    case 15: if (!GM15PFA) return 0;
             *a = 3; *b = 5;  return 1;
    case 21: *a = 3; *b = 7;  return 1;
    case 33: *a = 3; *b = 11; return 1;
    case 35: *a = 5; *b = 7;  return 1;
    case 39: *a = 3; *b = 13; return 1;
    case 45: *a = 5; *b = 9;  return 1;
    case 51: *a = 3; *b = 17; return 1;
    case 55: *a = 5; *b = 11; return 1;
    case 57: *a = 3; *b = 19; return 1;
    case 63: *a = 7; *b = 9;  return 1;
    }
    return 0;
}

/* r12: stage-1 module dispatch -- identical to gmod except module 25
 * routes to the gdft25 CT (needed by the role-SWAPPED pairs (25,2)/(25,4)
 * below, where the wide factor runs in stage 1).  For every pre-r12 pair
 * P is in {2..9}, where this is gmod verbatim (compile-time constant P:
 * same codegen). */
static inline __attribute__((always_inline)) void
gmodP(v8 *xr, v8 *xi, const int P, const struct gtabs *g)
{
    if (P == 25 && G25CT) { gdft25(xr, xi, g->t25c, g->t25s); return; }
    gmod(xr, xi, P, g->csP, g->snP);
}

/* Stage-2 module dispatch: composite odd Q -> nested PFA, else the flat
 * kernel.  Q is a compile-time constant per GP_DEF instantiation, so the
 * switch resolves statically.  -DGMODPFA=0 restores the flat fold for the
 * cross-arch races (66..126 then lose coverage of nothing -- the fold
 * handles any odd n -- but run 1.5-2.5x more module ops). */
#ifndef GMODPFA
#define GMODPFA 1
#endif
static inline __attribute__((always_inline)) void
gmodQ(v8 *xr, v8 *xi, const int Q, const struct gtabs *g)
{
    if (Q == 25 && G25CT) { gdft25(xr, xi, g->t25c, g->t25s); return; }
    if (!GMODPFA) { gmod(xr, xi, Q, g->csQ, g->snQ); return; }
    switch (Q) {
    case 15: if (!GM15PFA) { gmod(xr, xi, Q, g->csQ, g->snQ); break; }
             gmodpfa(xr, xi, g, 3, 5);  break;
    case 21: gmodpfa(xr, xi, g, 3, 7);  break;
    case 33: gmodpfa(xr, xi, g, 3, 11); break;
    case 35: gmodpfa(xr, xi, g, 5, 7);  break;
    case 39: gmodpfa(xr, xi, g, 3, 13); break;
    case 45: gmodpfa(xr, xi, g, 5, 9);  break;
    case 51: gmodpfa(xr, xi, g, 3, 17); break;
    case 55: gmodpfa(xr, xi, g, 5, 11); break;
    case 57: gmodpfa(xr, xi, g, 3, 19); break;
    case 63: gmodpfa(xr, xi, g, 7, 9);  break;
    default: gmod(xr, xi, Q, g->csQ, g->snQ);
    }
}

/* One length-L pencil on the SoA arena at slot stride st doubles; cp !=
 * NULL fuses the map into the stage-2 stores exactly like the tuned STM
 * path.  The body is always_inline and instantiated once per (P,Q) pair
 * below with CONSTANT P and Q, so gcc unrolls every loop and resolves
 * gmod's switch at compile time -- measured 1.3-2.4x over the runtime-loop
 * version (6: 1.42->0.58, 14: 16.7->7.3, 24: 105.6->55.1, 63: 2611->2041
 * us, B=8 execute).
 *
 * IPOK (r4, compile-time): when Q == 1 mod P the r2 disjointness rule
 * holds -- stage-2 group k1's read slot set {(Q k1 + P j2) mod L} and CRT
 * write set are both the residue class {== k1 mod P}, groups mutually
 * disjoint -- so the pencil runs IN PLACE like the tuned codelets: stage 1
 * writes back to its input slots, stage 2 reads those slots directly
 * (input j2 of group k1 sits at inmap[j2*P + k1]).  Kills the 2L-vector
 * tr/ti temp round trip per pencil.  IPOK sizes: 14, 18, 21, 36, 56.
 * Q != 1 mod P pairs keep the buffered form (in-place safe for any pair). */
static inline __attribute__((always_inline)) void
gpencil_body(double *restrict p, const ptrdiff_t st,
             const double *restrict cp, const struct gtabs *g,
             const int P, const int Q, const int IPOK)
{
    v8 tr[GMAXL], ti[GMAXL];
    for (int j2 = 0; j2 < Q; ++j2) {
        v8 xr[27], xi[27];
        const int16_t *im = g->inmap + (size_t)j2 * P;
        for (int j1 = 0; j1 < P; ++j1) {
            xr[j1] = QR_(p, st, im[j1]);
            xi[j1] = QI_(p, st, im[j1]);
        }
        gmodP(xr, xi, P, g);
        if (IPOK) {
            for (int j1 = 0; j1 < P; ++j1) {
                QR_(p, st, im[j1]) = xr[j1];
                QI_(p, st, im[j1]) = xi[j1];
            }
        } else {
            for (int j1 = 0; j1 < P; ++j1) {
                tr[(size_t)j1 * Q + j2] = xr[j1];
                ti[(size_t)j1 * Q + j2] = xi[j1];
            }
        }
    }
    for (int k1 = 0; k1 < P; ++k1) {
        v8 xr[63], xi[63];
        if (IPOK) {
            for (int j2 = 0; j2 < Q; ++j2) {
                int s = g->inmap[(size_t)j2 * P + k1];
                xr[j2] = QR_(p, st, s);
                xi[j2] = QI_(p, st, s);
            }
        } else {
            for (int j2 = 0; j2 < Q; ++j2) {
                xr[j2] = tr[(size_t)k1 * Q + j2];
                xi[j2] = ti[(size_t)k1 * Q + j2];
            }
        }
        gmodQ(xr, xi, Q, g);
        const int16_t *om = g->outmap + (size_t)k1 * Q;
        if (cp) {
            for (int j2 = 0; j2 < Q; ++j2) {
                int o = om[j2];
                v8 zr = xr[j2], zi = xi[j2];
                map8(&zr, &zi, cp + (size_t)o * st, GMT & 1, (GMT >> 1) & 1);
                QR_(p, st, o) = zr;  QI_(p, st, o) = zi;
            }
        } else {
            for (int j2 = 0; j2 < Q; ++j2) {
                int o = om[j2];
                QR_(p, st, o) = xr[j2];  QI_(p, st, o) = xi[j2];
            }
        }
    }
}

typedef void (*gpen_fn)(double *restrict, ptrdiff_t,
                        const double *restrict, const struct gtabs *);

/* r12c: FORMULA-baked pencil for the WVS 25-pairs.  The differential PMU
 * at 100 read 29.5M instructions/step against gen_batchlane's 13.4M at
 * equal FP dispatch and equal (tiny, 2.7 MB/step) DRAM traffic -- the
 * generic pencil's runtime int16 inmap/outmap loads and per-slot address
 * arithmetic are the gap.  Here the slot indices are the GT/CRT FORMULAS
 * with compile-time P, Q, A, B: after unrolling every index is a literal
 * and every access a constant offset, the tuned-codelet treatment the
 * generic engine never got.  A = Q*inv(Q mod P, P) mod L and B = P*inv(P
 * mod Q, Q) mod L are baked per instantiation (values cross-checked
 * against gtabs_init: chain outputs are cmp-identical to the table
 * pencils).  Only instantiated for the 25-pairs the WVS path serves. */
static inline __attribute__((always_inline)) void
gpencilf_body(double *restrict p, const ptrdiff_t st,
              const double *restrict cp, const struct gtabs *g,
              const int P, const int Q, const int A, const int B,
              const int IPOK)
{
    const int L = P * Q;
    v8 tr[GMAXL], ti[GMAXL];
    /* the pragmas are DEMANDS, not hints: -funroll-loops refuses these
     * big-body loops, and only full unrolling folds the index formulas
     * into literal offsets (a rolled body computes % L per slot on port 1
     * -- measured, 4.2M p1 uops/step at 100) */
#pragma GCC unroll 32
    for (int j2 = 0; j2 < Q; ++j2) {
        v8 xr[27], xi[27];
#pragma GCC unroll 32
        for (int j1 = 0; j1 < P; ++j1) {
            const int sIn = (Q * j1 + P * j2) % L;
            xr[j1] = QR_(p, st, sIn);
            xi[j1] = QI_(p, st, sIn);
        }
        gmodP(xr, xi, P, g);
        if (IPOK) {
#pragma GCC unroll 32
            for (int j1 = 0; j1 < P; ++j1) {
                const int sIn = (Q * j1 + P * j2) % L;
                QR_(p, st, sIn) = xr[j1];
                QI_(p, st, sIn) = xi[j1];
            }
        } else {
#pragma GCC unroll 32
            for (int j1 = 0; j1 < P; ++j1) {
                tr[j1 * Q + j2] = xr[j1];
                ti[j1 * Q + j2] = xi[j1];
            }
        }
    }
#pragma GCC unroll 32
    for (int k1 = 0; k1 < P; ++k1) {
        v8 xr[63], xi[63];
        if (IPOK) {
#pragma GCC unroll 32
            for (int j2 = 0; j2 < Q; ++j2) {
                const int sIn = (Q * k1 + P * j2) % L;
                xr[j2] = QR_(p, st, sIn);
                xi[j2] = QI_(p, st, sIn);
            }
        } else {
#pragma GCC unroll 32
            for (int j2 = 0; j2 < Q; ++j2) {
                xr[j2] = tr[(size_t)k1 * Q + j2];
                xi[j2] = ti[(size_t)k1 * Q + j2];
            }
        }
        gmodQ(xr, xi, Q, g);
        if (cp) {
#pragma GCC unroll 32
            for (int j2 = 0; j2 < Q; ++j2) {
                const int o = (A * k1 + B * j2) % L;
                v8 zr = xr[j2], zi = xi[j2];
                map8(&zr, &zi, cp + (size_t)o * st, GMT & 1, (GMT >> 1) & 1);
                QR_(p, st, o) = zr;
                QI_(p, st, o) = zi;
            }
        } else {
#pragma GCC unroll 32
            for (int j2 = 0; j2 < Q; ++j2) {
                const int o = (A * k1 + B * j2) % L;
                QR_(p, st, o) = xr[j2];
                QI_(p, st, o) = xi[j2];
            }
        }
    }
}

#define GPF_DEF(Pv, Qv, Av, Bv)                                               \
static void gpencilf_##Pv##_##Qv(double *restrict p, ptrdiff_t st,            \
                                 const double *restrict cp,                   \
                                 const struct gtabs *g)                       \
{ gpencilf_body(p, st, cp, g, Pv, Qv, Av, Bv, (Qv) % (Pv) == 1); }
GPF_DEF(4, 25, 25, 76)   /* inv(25%4,4)=1, inv(4,25)=19: A=25, B=76 */
GPF_DEF(25, 4, 76, 25)   /* roles swapped: A=4*19=76, B=25          */
GPF_DEF(2, 25, 25, 26)   /* inv(25%2,2)=1, inv(2,25)=13: A=25, B=26 */
GPF_DEF(25, 2, 26, 25)
/* r8: out-of-place split-complex pencil (separate re/im arrays, no map) */
typedef void (*gspen_fn)(const double *restrict, const double *restrict,
                         double *restrict, double *restrict, ptrdiff_t,
                         const struct gtabs *);

/* r8: the same two-stage GT-PFA pencil on SPLIT-COMPLEX arrays, out of
 * place (src re/im -> dst re/im at slot stride st doubles), no map -- the
 * pencil the per-volume split path uses.  Same maps, same modules, always
 * buffered (out-of-place needs no IPOK rule). */
static inline __attribute__((always_inline)) void
gspencil_body(const double *restrict sr, const double *restrict si,
              double *restrict dr, double *restrict di, const ptrdiff_t st,
              const struct gtabs *g, const int P, const int Q)
{
    v8 tr[GMAXL], ti[GMAXL];
    for (int j2 = 0; j2 < Q; ++j2) {
        v8 xr[27], xi[27];
        const int16_t *im = g->inmap + (size_t)j2 * P;
        for (int j1 = 0; j1 < P; ++j1) {
            xr[j1] = vload(sr + (size_t)im[j1] * st);
            xi[j1] = vload(si + (size_t)im[j1] * st);
        }
        gmodP(xr, xi, P, g);
        for (int j1 = 0; j1 < P; ++j1) {
            tr[(size_t)j1 * Q + j2] = xr[j1];
            ti[(size_t)j1 * Q + j2] = xi[j1];
        }
    }
    for (int k1 = 0; k1 < P; ++k1) {
        gmodQ(tr + (size_t)k1 * Q, ti + (size_t)k1 * Q, Q, g);
        const int16_t *om = g->outmap + (size_t)k1 * Q;
        for (int j2 = 0; j2 < Q; ++j2) {
            vstore(dr + (size_t)om[j2] * st, tr[(size_t)k1 * Q + j2]);
            vstore(di + (size_t)om[j2] * st, ti[(size_t)k1 * Q + j2]);
        }
    }
}

/* r11c: IN-PLACE split-complex pencil for IPOK pairs (Q == 1 mod P) -- the
 * r4 batched IPOK rule ported to the split path.  One array pair, stage 1
 * writes back to its own slots, stage 2 reads inmap[j2*P+k1] and scatters
 * to the CRT slots (group read/write sets are the same residue class, so
 * groups never clobber each other -- the r2 disjointness rule).  vs the
 * buffered gspencil this deletes the 2L-v8 tr/ti round trip AND its 16 KB
 * of stack footprint per call: the r11 counters read 41% more loads / 79%
 * more stores per step than gen_pfa_large's engine, and the buffered
 * pencil's stack (tr/ti + gdft25's ur/ui + the pass-3 lane buffers =
 * ~45 KB) simply does not fit the 48 KB L1 next to the data.  Both
 * claimed pairs (2,25) and (4,25) are IPOK. */
static inline __attribute__((always_inline)) void
gspencil_ip_body(double *restrict dr, double *restrict di,
                 const ptrdiff_t st, const struct gtabs *g,
                 const int P, const int Q)
{
    for (int j2 = 0; j2 < Q; ++j2) {
        v8 xr[27], xi[27];
        const int16_t *im = g->inmap + (size_t)j2 * P;
        for (int j1 = 0; j1 < P; ++j1) {
            xr[j1] = vload(dr + (size_t)im[j1] * st);
            xi[j1] = vload(di + (size_t)im[j1] * st);
        }
        gmodP(xr, xi, P, g);
        for (int j1 = 0; j1 < P; ++j1) {
            vstore(dr + (size_t)im[j1] * st, xr[j1]);
            vstore(di + (size_t)im[j1] * st, xi[j1]);
        }
    }
    for (int k1 = 0; k1 < P; ++k1) {
        v8 xr[63], xi[63];
        for (int j2 = 0; j2 < Q; ++j2) {
            int s = g->inmap[(size_t)j2 * P + k1];
            xr[j2] = vload(dr + (size_t)s * st);
            xi[j2] = vload(di + (size_t)s * st);
        }
        gmodQ(xr, xi, Q, g);
        const int16_t *om = g->outmap + (size_t)k1 * Q;
        for (int j2 = 0; j2 < Q; ++j2) {
            vstore(dr + (size_t)om[j2] * st, xr[j2]);
            vstore(di + (size_t)om[j2] * st, xi[j2]);
        }
    }
}

/* The (P,Q) instantiation list, X-macro form (r8; the literal GP_DEF /
 * GP_CASE lists of r3-r7 are unchanged in content and order):
 * r3 seed pairs; r6 widening (modules 11/13/15/16/21/25/27); r6 nested-PFA
 * modules as Q (66..126 = 2 x {33..63}); r6 prime modules 17..31 + nested
 * 51/57. */
#define GP_LIST(X)                                                            \
    X(2, 3)  X(2, 7)  X(2, 9)  X(3, 7)  X(3, 8)                               \
    X(4, 7)  X(4, 9)  X(5, 7)  X(5, 9)  X(7, 8)                               \
    X(7, 9)                                                                   \
    X(2, 11) X(2, 13) X(2, 15) X(2, 21) X(2, 25) X(2, 27)                     \
    X(3, 11) X(3, 13) X(3, 16) X(3, 25)                                       \
    X(4, 11) X(4, 13) X(4, 15) X(4, 21) X(4, 25) X(4, 27)                     \
    X(5, 11) X(5, 13)                                                         \
    X(7, 11) X(7, 13) X(7, 15) X(7, 16)                                       \
    X(8, 9)  X(8, 11) X(8, 13) X(8, 15)                                       \
    X(9, 11) X(9, 13)                                                         \
    X(2, 33) X(2, 35) X(2, 39) X(2, 45) X(2, 55)                              \
    X(2, 63)                                                                  \
    X(2, 17) X(2, 19) X(2, 23) X(2, 29) X(2, 31)                              \
    X(2, 51) X(2, 57)                                                         \
    X(3, 17) X(3, 19) X(3, 23) X(3, 29) X(3, 31)                              \
    X(4, 17) X(4, 19) X(4, 23) X(4, 29) X(4, 31)                              \
    X(5, 17) X(5, 19) X(5, 23)                                                \
    X(7, 17)                                                                  \
    X(25, 2) X(25, 4)  /* r12: role-swapped 50/100 (wide factor stage 1,
                          map at the small-DFT stage-2 stores -- the r10
                          factor-swap verdict for the WVS x pass) */

#define GP_DEF(Pv, Qv)                                                        \
static void gpencil_##Pv##_##Qv(double *restrict p, ptrdiff_t st,             \
                                const double *restrict cp,                    \
                                const struct gtabs *g)                        \
{ gpencil_body(p, st, cp, g, Pv, Qv, (Qv) % (Pv) == 1); }
GP_LIST(GP_DEF)

#define GS_DEF(Pv, Qv)                                                        \
static void gspencil_##Pv##_##Qv(const double *restrict sr,                   \
                                 const double *restrict si,                   \
                                 double *restrict dr, double *restrict di,    \
                                 ptrdiff_t st, const struct gtabs *g)         \
{ gspencil_body(sr, si, dr, di, st, g, Pv, Qv); }
GP_LIST(GS_DEF)

/* r11c: in-place split pencils, IPOK pairs only (empty otherwise; the
 * lookup returns NULL for non-IPOK pairs, so the empties are never called) */
typedef void (*gspenip_fn)(double *restrict, double *restrict, ptrdiff_t,
                           const struct gtabs *);
#define GSIP_DEF(Pv, Qv)                                                      \
static void gspencil_ip_##Pv##_##Qv(double *restrict dr,                      \
                                    double *restrict di,                      \
                                    ptrdiff_t st, const struct gtabs *g)      \
{ if ((Qv) % (Pv) == 1) gspencil_ip_body(dr, di, st, g, Pv, Qv); }
GP_LIST(GSIP_DEF)

static gspenip_fn gspenip_lookup(int P, int Q)
{
    if (Q % P != 1) return NULL;
#define GSIP_CASE(Pv, Qv) case (Pv) * 128 + (Qv): return gspencil_ip_##Pv##_##Qv;
    switch (P * 128 + Q) {
    GP_LIST(GSIP_CASE)
    }
#undef GSIP_CASE
    return NULL;
}

static gpen_fn gpen_lookup(int P, int Q)
{
#define GP_CASE(Pv, Qv) case (Pv) * 128 + (Qv): return gpencil_##Pv##_##Qv;
    switch (P * 128 + Q) {
    GP_LIST(GP_CASE)
    }
#undef GP_CASE
    return NULL;
}

static gspen_fn gspen_lookup(int P, int Q)
{
#define GS_CASE(Pv, Qv) case (Pv) * 128 + (Qv): return gspencil_##Pv##_##Qv;
    switch (P * 128 + Q) {
    GP_LIST(GS_CASE)
    }
#undef GS_CASE
    return NULL;
}

/* Generic two-sweep step over the padded arena: identical structure to the
 * tuned DEF_ENGINE (zy sweep per x-plane, then the x pass per column with
 * the map fused into stage-2 stores). */
static void gsweep_zy(double *restrict pl, int L, const struct gtabs *g,
                      gpen_fn pen)
{
    for (int y = 0; y < L; ++y)
        pen(pl + (size_t)y * L * 16, 16, NULL, g);
    for (int z = 0; z < L; ++z)
        pen(pl + (size_t)z * 16, (ptrdiff_t)L * 16, NULL, g);
}
static void gsoa_fft(double *restrict S, int L, int PL, const struct gtabs *g,
                     gpen_fn pen)
{
    for (int x = 0; x < L; ++x)
        gsweep_zy(S + (size_t)x * PL * 16, L, g, pen);
    for (int c = 0; c < L * L; ++c)
        pen(S + (size_t)c * 16, (ptrdiff_t)PL * 16, NULL, g);
}
static void gsoa_step(double *restrict S, const double *restrict C,
                      int L, int PL, const struct gtabs *g, gpen_fn pen)
{
    for (int x = 0; x < L; ++x)
        gsweep_zy(S + (size_t)x * PL * 16, L, g, pen);
    for (int c = 0; c < L * L; ++c)
        pen(S + (size_t)c * 16, (ptrdiff_t)PL * 16, C + (size_t)c * 16, g);
}

/* --------------------- r8: generic per-volume split chain step (B%8, B=1)
 * The r7 rotation fused-map step ported to the GENERIC engine (my r7
 * next-list #1; the surprise-test addendum names exactly these cells --
 * L=21/44-class at small batch -- as the library's weak spot).  Structure
 * is DEF_STEP verbatim: three ping-pong passes, pass 3 transforms the
 * stride-1 dim and stores the volume ROTATED ([P0][P1][P2]->[P2][P0][P1])
 * with the graded map fused into its contiguous stores.  Pencils are the
 * r8 out-of-place split-complex gspencil instantiations (same maps and
 * modules as gpen; a first cut that gathered lanes into a site buffer and
 * reused the in-place gpen paid a 4L-vector round trip per pass-1/2 pencil
 * -- numbers in the r8 strategy record).  Requires L >= 8 (8 stride-1
 * lanes must fit in one slab row); L=6, the only generic size below 8,
 * keeps the lane-replicated path. */

/* One chain step, out of place S -> D, output rotated one turn; cr == NULL
 * skips the map (plain FFT + rotation, for a future execute() route). */
static void gstep_split(double *restrict Sr, double *restrict Si,
                        double *restrict Dr, double *restrict Di,
                        const double *restrict cr, const double *restrict ci,
                        const int L, const struct gtabs *g, gspen_fn pen)
{
    const int LL = L * L;
    /* pass 1: stride-L^2 dim, lanes = flat 8 of the inner L^2 */
    for (int b = 0; b < LL; b += 8) {
        int o = (b + 8 <= LL) ? b : LL - 8;
        pen(Sr + o, Si + o, Dr + o, Di + o, (ptrdiff_t)LL, g);
    }
    /* pass 2: stride-L dim per x slab, lanes = 8 contiguous stride-1 */
    for (int x = 0; x < L; ++x) {
        ptrdiff_t xo = (ptrdiff_t)x * LL;
        for (int z = 0; z < L; z += 8) {
            int o = (z + 8 <= L) ? z : L - 8;
            pen(Dr + xo + o, Di + xo + o, Sr + xo + o, Si + xo + o,
                (ptrdiff_t)L, g);
        }
    }
    /* pass 3: stride-1 dim; transpose-in to buffered lanes, pencil at
     * stride 8, map fused into ROTATED stores (rows R = 0..L^2-1 of
     * [R][z]; output k goes to k*L^2 + R, the r7 minimal-block form). */
    for (int rb = 0; rb < LL; rb += 8) {
        int R0 = (rb + 8 <= LL) ? rb : LL - 8;
        v8 pzr[GMAXL], pzi[GMAXL], por[GMAXL], poi[GMAXL];
        for (int z = 0; z < L; z += 8) {
            int zo = (z + 8 <= L) ? z : L - 8;
            v8 blk[8];
            for (int q = 0; q < 8; ++q)
                blk[q] = vload(Sr + (ptrdiff_t)(R0 + q) * L + zo);
            tr8(blk);
            for (int q = 0; q < 8; ++q) pzr[zo + q] = blk[q];
            for (int q = 0; q < 8; ++q)
                blk[q] = vload(Si + (ptrdiff_t)(R0 + q) * L + zo);
            tr8(blk);
            for (int q = 0; q < 8; ++q) pzi[zo + q] = blk[q];
        }
        pen((const double *)pzr, (const double *)pzi,
            (double *)por, (double *)poi, 8, g);
        for (int k = 0; k < L; ++k) {
            v8 zr = por[k], zi = poi[k];
            if (cr)
                map8c(&zr, &zi,
                      vload(cr + (ptrdiff_t)k * LL + R0),
                      vload(ci + (ptrdiff_t)k * LL + R0));
            vstore(Dr + (ptrdiff_t)k * LL + R0, zr);
            vstore(Di + (ptrdiff_t)k * LL + R0, zi);
        }
    }
}

/* --------------------- r11: large-L slab-FUSED split chain step
 * Two-axes-per-pass fusion (lit 11 Tier 2; the rounds-11 mandate) from
 * this class's angle.  Per step, on the state volume S:
 *   pass 1: stride-L^2 dim, IN PLACE (see the note at the code);
 *   per x-slab: pass 2 (stride L) S slab -> a single reused slab of D
 *   (an L2-resident scratch, 160 KB at L=100), then pass 3 (stride 1) on
 *   the hot scratch back into the S slab with the map fused into
 *   natural-order stores.
 * vs the r8 rotation step this deletes BOTH full-volume ping-pong round
 * trips (pass-2 write + pass-3 re-read now live in L2), the L-stream
 * rotated stores (at L=100: 100 concurrent 64 B write streams + 100
 * rotated c read streams -- the miss-stream shape gen_pfa_large's r1
 * measured losing 45% on this cell), the Cra/Crb copies and the
 * un-rotate; c is read in natural order.  Costs 2x the tr8 of the
 * rotation form (transpose back), small at L >= 44 against the saved
 * traffic; at 10..21-class sizes the rotation form's minimal block count
 * wins (r7 numbers), so this path is selected by L >= GSLAB_MIN and the
 * small sizes keep r8 behavior bit-identically.
 * Overlap tails stay idempotent: pass 2 and pass 3 are out of place
 * (slab -> scratch, scratch -> slab), pass 1's tail is staged through D
 * (see code), and the map is a pure function of the recomputed pencil
 * output and c. */
#ifdef GPHASET                  /* dev-only per-pass rdtscp accounting */
#include <stdio.h>
static unsigned long long gph_[3];
static void gph_print_(void)
{
    fprintf(stderr, "GPHASET p1=%llu p2=%llu p3=%llu (cycles)\n",
            gph_[0], gph_[1], gph_[2]);
}
static unsigned long long gph_now_(void)
{
    unsigned lo, hi, aux;
    __asm__ volatile("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux));
    return ((unsigned long long)hi << 32) | lo;
}
static void gph_init_(void)
{
    static int once;
    if (!once) { once = 1; atexit(gph_print_); }
}
#define GPH_T(i, ...) do { gph_init_();                                      \
        unsigned long long t0_ = gph_now_(); { __VA_ARGS__ }                 \
        gph_[i] += gph_now_() - t0_; } while (0)
#else
#define GPH_T(i, ...) do { __VA_ARGS__ } while (0)
#endif

/* r11b: pass-3 store form -- 1 = half-turn swapped stores (default),
 * 0 = tr8-back + natural stores, kept for the cross-arch races. */
#ifndef GSLABSW
#define GSLABSW 1
#endif
/* r11b: slab warm prefetch loops (see the comment at the use site).
 * Node race at 100 (three interleaved rounds, min): GWARMC -2..-8% and
 * best in every round (6088 vs 6552-7818); GWARMS +3% alone (no compute
 * to hide behind) and worse combined.  Ship c-warm only. */
#ifndef GWARMC
#define GWARMC 1
#endif
#ifndef GWARMS
#define GWARMS 0
#endif

static void gstep_slab(double *restrict Sr, double *restrict Si,
                       double *restrict Dr, double *restrict Di,
                       const double *restrict cr, const double *restrict ci,
                       const int L, const struct gtabs *g, gspen_fn pen,
                       gspenip_fn pen_ip)
{
    const int LL = L * L;
    const int LT = LL & ~7;
    /* pass 1 IN PLACE (S -> S): the x pass is the step's miss-stream
     * problem (2 x L read streams at stride L^2 x 8 B); in place the store
     * hits the line the load just brought in, so there are no separate RFO
     * write streams (gen_pfa_large keeps its x pass in place for the same
     * reason -- their r1 measured the out-of-place fused form losing 45%
     * at this size).  Safe with one pointer: the pencil buffers everything
     * through tr/ti before its first store.  The overlap tail (LL % 8 !=
     * 0; absent at L=100, 4 lanes at L=50) cannot recompute in place: it
     * runs FIRST, out of place into the D scratch, and its lanes are
     * copied back after the full blocks. */
    if (LL & 7) {
        int o = LL - 8;
        pen(Sr + o, Si + o, Dr + o, Di + o, (ptrdiff_t)LL, g);
    }
    GPH_T(0,
    if (pen_ip)     /* r11c: no tr/ti round trip, half the stack footprint */
        for (int b = 0; b < LT; b += 8)
            pen_ip(Sr + b, Si + b, (ptrdiff_t)LL, g);
    else
        for (int b = 0; b < LT; b += 8)
            pen(Sr + b, Si + b, Sr + b, Si + b, (ptrdiff_t)LL, g);
    );
    if (LL & 7) {
        int o = LL - 8;
        for (int k = 0; k < L; ++k) {
            vstore(Sr + (ptrdiff_t)k * LL + o,
                   vload(Dr + (ptrdiff_t)k * LL + o));
            vstore(Si + (ptrdiff_t)k * LL + o,
                   vload(Di + (ptrdiff_t)k * LL + o));
        }
    }
    for (int x = 0; x < L; ++x) {
        ptrdiff_t xo = (ptrdiff_t)x * LL;
        /* r11b slab warms, RACEABLE against the standing no-prefetch rule
         * (which was set in issue-bound passes; these are latency-bound
         * cold 800 B-stride walks the L2 streamer cannot follow across
         * pages).  GWARMC prefetches the c slab BEFORE pass 2 so the miss
         * latency overlaps pass-2 compute (pass 3 reads c); GWARMS
         * prefetches the S slab pass 2 is about to walk. */
#if GWARMC
        if (cr)
            for (int off = 0; off < LL; off += 8) {
                __builtin_prefetch(cr + xo + off, 0, 3);
                __builtin_prefetch(ci + xo + off, 0, 3);
            }
#endif
#if GWARMS
        for (int off = 0; off < LL; off += 8) {
            __builtin_prefetch(Sr + xo + off, 0, 3);
            __builtin_prefetch(Si + xo + off, 0, 3);
        }
#endif
        /* pass 2: stride-L dim within the slab, S slab -> the FIRST slab
         * of D, reused for every x -- an L2-resident scratch (2*L^2*8 B =
         * 160 KB at L=100), so the r8 form's second full-volume ping-pong
         * round trip never touches DRAM */
        GPH_T(1,
        for (int z = 0; z < L; z += 8) {
            int o = (z + 8 <= L) ? z : L - 8;
            pen(Sr + xo + o, Si + xo + o, Dr + o, Di + o, (ptrdiff_t)L, g);
        }
        );
        /* pass 3: stride-1 dim on the hot scratch, D[0..LL) -> S slab;
         * tr8 in, pencil on buffered lanes, tr8 back, map fused into
         * natural-order slab-local stores (c read in natural order) */
        GPH_T(2,
        for (int y = 0; y < L; y += 8) {
            int yo = (y + 8 <= L) ? y : L - 8;
            v8 pzr[GMAXL], pzi[GMAXL], por[GMAXL], poi[GMAXL];
            for (int z = 0; z < L; z += 8) {
                int zo = (z + 8 <= L) ? z : L - 8;
                v8 blk[8];
                for (int q = 0; q < 8; ++q)
                    blk[q] = vload(Dr + (ptrdiff_t)(yo + q) * L + zo);
                tr8(blk);
                for (int q = 0; q < 8; ++q) pzr[zo + q] = blk[q];
                for (int q = 0; q < 8; ++q)
                    blk[q] = vload(Di + (ptrdiff_t)(yo + q) * L + zo);
                tr8(blk);
                for (int q = 0; q < 8; ++q) pzi[zo + q] = blk[q];
            }
            const v8 *outr, *outi;
            if (pen_ip) {   /* r11c: in place on the lane buffers */
                pen_ip((double *)pzr, (double *)pzi, 8, g);
                outr = pzr;  outi = pzi;
            } else {
                pen((const double *)pzr, (const double *)pzi,
                    (double *)por, (double *)poi, 8, g);
                outr = por;  outi = poi;
            }
            if (GSLABSW) {
                /* r11b: HALF-TURN stores (my r7 trick, per slab): por[k]'s
                 * lanes are y = yo..yo+7, so storing it whole at slab
                 * offset k*L + yo writes the slab with the inner dims
                 * SWAPPED -- the transpose-back tr8s (half the pass-3
                 * port-5 bill; the counters read p5 4.69G vs p0 3.76G with
                 * them) are gone.  Layout parity alternates per step; the
                 * chain passes c in the matching parity and un-swaps once
                 * at the end when m is odd. */
                for (int k = 0; k < L; ++k) {
                    v8 zr = outr[k], zi = outi[k];
                    ptrdiff_t o = xo + (ptrdiff_t)k * L + yo;
                    if (cr)
                        map8c(&zr, &zi, vload(cr + o), vload(ci + o));
                    vstore(Sr + o, zr);
                    vstore(Si + o, zi);
                }
            } else {
                for (int z = 0; z < L; z += 8) {
                    int zo = (z + 8 <= L) ? z : L - 8;
                    v8 br[8], bi[8];
                    for (int q = 0; q < 8; ++q) br[q] = outr[zo + q];
                    tr8(br);
                    for (int q = 0; q < 8; ++q) bi[q] = outi[zo + q];
                    tr8(bi);
                    for (int q = 0; q < 8; ++q) {
                        ptrdiff_t o = xo + (ptrdiff_t)(yo + q) * L + zo;
                        if (cr)
                            map8c(&br[q], &bi[q], vload(cr + o), vload(ci + o));
                        vstore(Sr + o, br[q]);
                        vstore(Si + o, bi[q]);
                    }
                }
            }
        }
        );
    }
}

/* Slab-fused vs rotation crossover: rotation keeps the minimal block count
 * (its r7 win at 10..21-class sizes); slab fusion wins once the volume
 * stops being cache-resident and traffic dominates.  Raced on the node at
 * 44/50/100 this round; override per host with -DGSLAB_MIN=<L> (999 = never,
 * 8 = always). */
#ifndef GSLAB_MIN
#define GSLAB_MIN 44
#endif

/* --------------------- r12: WITHIN-VOLUME SoA chain step (B=1/B%8, large L)
 * BORROWED: gen_batchlane gen_r11 -- the engine that took the L=100 cell
 * at 4059 us vs my r11 slab form's 6009: lanes = 8 x-PLANES of ONE volume
 * instead of 8 volumes (the brief's approach #4, "the batch-lane trick
 * without a batch").  It is also exactly my own r11 next-list #1 (a
 * site-interleaved slab variant of the split path to halve line touches):
 * the state lives in NS = ceil(L/8) slabs of the SAME interleaved site
 * format as the batched engine (site = re[8]|im[8], slab s lanes =
 * x-planes 8s..8s+7), so
 *   - z-pencils (stride 1 site) and y-pencils (stride ZP sites) are the
 *     batched engine's pure elementwise lane code -- zero shuffles, zero
 *     masked-lane waste -- run back to back per L2-resident slab (the
 *     two-axes-per-pass fusion, from the lane side);
 *   - ONE stream of 128 B sites replaces the split path's two separate
 *     re/im 64 B-line streams -- the 2x line-touch tax the r11 counters
 *     measured against gen_pfa_large (5.60M vs 3.97M loads/step) is gone
 *     by construction;
 *   - only the x pass crosses lanes: per (y, 8z) column, tr8 gather from
 *     the NS slabs into a <=16 KB site-format scratch pencil, then the
 *     EXISTING in-place IPOK gpencil (gdft25 CT, fused map at its stage-2
 *     stores; c prepacked once per chain in consumption order), then tr8
 *     scatter back.  2*NS tr8 in + 2*NS out per column -- gen_batchlane's
 *     1248-shuffle bill at L=100, unavoidable and hidden under the pass's
 *     streams (their PMU: stalls_mem_any binds, not p5).
 * Layout: z rows padded to ZP8 = 8*ceil(L/8) sites so the x-pass z-blocks
 * are NON-OVERLAPPING (an in-place pass cannot recompute an overlap tail:
 * the gather would read post-map values), plus 1 more so the y-pencil
 * stride is an ODD site count (32 distinct 4K residues -- the r2 pad
 * rule, stronger form); slab stride padded to == 2 mod 32 sites (slab
 * bytes == 256 mod 4096, house rule).  PAD LANES AND PAD Z-SITES ARE
 * ZERO AND STAY ZERO EXACTLY: pass A only loops real y/z, pad-x scratch
 * slots sit above the pencil's L slots, the FFT is linear (0 -> 0), and
 * the fused map fixes 0 when c == 0 (c pads packed zero) -- so nothing
 * unmapped ever grows, NaNs cannot form, and no pad value can reach a
 * real lane.
 * Selected for generic L >= GWVS_MIN when the batch leaves a remainder
 * group r <= GSPLIT_RMAX (the graded 100 B=1 and 50 B=4 shapes; B > 1
 * loops volumes).  -DGWVS=0 races the r11 gstep_slab back; sizes below
 * GWVS_MIN keep r11 behavior bit-identically. */
#ifndef GWVS
#define GWVS 1
#endif
/* GWVS_MIN raced DOWN from the planned 50: the within-volume form beats
 * the r8 rotation split at EVERY generic size measured (14: 9.7 vs 12.8
 * us; 21: 33.1 vs 41.8; 28: 73.7 vs 87.4; 36: 198 vs 258; 44: 410 vs 440
 * -- shuffle-free sweeps + one site stream + no rotated store streams
 * beat the rotation form's minimal block count even at 43% pad-lane waste
 * (L=14's second slab)).  8 = every generic size (L < 8 cannot feed a z
 * block; L=6 keeps lane replication). */
#ifndef GWVS_MIN
#define GWVS_MIN 8
#endif
/* r12b: role-SWAPPED WVS pencil -- gtabs built as (25, P), so stage 1 is
 * the wide gdft25 (map-free) and stage 2 the small DFT4/DFT2 blocks where
 * the map fuses (gen_batchlane's r9/r10 factor-swap verdict, which their
 * r11 dft100 uses; my r10 adopted it for the tuned sizes only).  The
 * swapped pair is NOT IPOK (P % 25 != 1), so the pencil is the BUFFERED
 * single-touch form: strided slab slots are read once and written once
 * (tr/ti live on the L1 stack) instead of the IPOK form's two RMW touches
 * per slot -- the y-sweep's slots span the whole slab, where a second
 * touch costs L2.  Default ON at 100, OFF at 50 (batchlane r10/r11: a
 * DFT2-width map stage-2 LOSES; their dft50 keeps the map on the DFT25
 * side). */
#ifndef GWVSSW
#define GWVSSW 1
#endif
#ifndef GWVSSW50
#define GWVSSW50 0
#endif
/* r12c: pass-B gather prefetch (next z-block chunk per slab).  RACED AND
 * OFF: loses ~1% (5587 vs 5540 min-of-mins, 4 interleaved rounds) -- the
 * gather is issue-bound enough that the standing no-prefetch rule wins;
 * the r11 GWARMC exception does not extend here. */
#ifndef GWVSPF
#define GWVSPF 0
#endif
#define GWVS_MAXNS ((GMAXL + 7) / 8)

static void gstep_wvs(double *restrict W, const double *restrict Wc,
                      const int L, const int ZP, const int NS,
                      const int SLST, const int NBZ,
                      const struct gtabs *g, gpen_fn pen,
                      const struct gtabs *gw, gpen_fn penw)
{
    /* pass A: per slab, z-pencils then y-pencils (elementwise lanes).
     * z-pencil slots are CONTIGUOUS (12.8 KB, L1): the unswapped IPOK
     * form's second slot touch is an L1 hit and it carries no tr/ti
     * buffer traffic -- keep it.  y-pencil slots span the whole slab
     * (L2): the swapped BUFFERED form touches each slot once. */
    for (int s = 0; s < NS; ++s) {
        double *pl = W + (size_t)s * SLST * 16;
        GPH_T(0,
        for (int y = 0; y < L; ++y)
            pen(pl + (size_t)y * ZP * 16, 16, NULL, g);
        );
        GPH_T(1,
        for (int z = 0; z < L; ++z)
            penw(pl + (size_t)z * 16, (ptrdiff_t)ZP * 16, NULL, gw);
        );
    }
    /* pass B: x pass per (y, z-block): tr8 gather -> in-place pencil with
     * the map fused into its stage-2 stores (L1-hot scratch) -> tr8
     * scatter back to the same sites (in place, blocks disjoint) */
    double scr[GWVS_MAXNS * 8 * 16] __attribute__((aligned(64)));
    GPH_T(2,
    for (int y = 0; y < L; ++y) {
        for (int zb = 0; zb < NBZ; ++zb) {
            const double *cpb = Wc + ((size_t)y * NBZ + zb) * (size_t)L * 16;
            for (int s = 0; s < NS; ++s) {
                const double *src = W + ((size_t)s * SLST + (size_t)y * ZP
                                         + (size_t)zb * 8) * 16;
#if GWVSPF
                /* warm the NEXT z-block's chunk of this slab (16 lines;
                 * every other line -- adjacent-line prefetch pairs them).
                 * The gather's 1 KB chunks at 1.3 MB stride are exactly
                 * the latency-bound cold walk the r11 GWARMC scope covers;
                 * stalls_mem_any read 42% of cycles without it. */
                if (zb + 1 < NBZ)
                    for (int q = 0; q < 8; ++q)
                        __builtin_prefetch(src + 128 + q * 16, 0, 3);
#endif
                v8 br[8], bi[8];
                for (int q = 0; q < 8; ++q) br[q] = vload(src + q * 16);
                for (int q = 0; q < 8; ++q) bi[q] = vload(src + q * 16 + 8);
                tr8(br);
                tr8(bi);
                double *dst = scr + (size_t)s * 8 * 16;
                for (int j = 0; j < 8; ++j) {
                    vstore(dst + j * 16, br[j]);
                    vstore(dst + j * 16 + 8, bi[j]);
                }
            }
            penw(scr, 16, cpb, gw);
            for (int s = 0; s < NS; ++s) {
                double *dst = W + ((size_t)s * SLST + (size_t)y * ZP
                                   + (size_t)zb * 8) * 16;
                const double *src = scr + (size_t)s * 8 * 16;
                v8 br[8], bi[8];
                for (int j = 0; j < 8; ++j) {
                    br[j] = vload(src + j * 16);
                    bi[j] = vload(src + j * 16 + 8);
                }
                tr8(br);
                tr8(bi);
                for (int q = 0; q < 8; ++q) {
                    vstore(dst + q * 16, br[q]);
                    vstore(dst + q * 16 + 8, bi[q]);
                }
            }
        }
    }
    );
}

/* Remainder volumes 1..GSPLIT_RMAX go per-volume through gstep_split;
 * larger remainders keep the one lane-replicated SoA group (its cost is
 * flat in r, so it wins once r * t_split exceeds one group).  Raced
 * same-core at L=21/34: split wins to r=4 (-3..-8%), replicated wins from
 * r=5 (+18% at 21 r=5, +29% at 21 r=7, +21% at 34 r=6).  -DGSPLIT_RMAX=0
 * restores the r6/r7 lane-replicated behavior everywhere; re-race the
 * threshold per host (the split/SoA balance is tr8-vs-FMA shaped). */
#ifndef GSPLIT_RMAX
#define GSPLIT_RMAX 4
#endif

/* Remainder-group pack/unpack (r < 8 real volumes): lanes >= r replicate
 * the last real volume; unpack writes only real lanes.  Scalar -- the
 * remainder group is the correctness fallback, not the fast path. */
static void gpack_plane(const double _Complex *in, size_t lane_stride,
                        size_t n, double *q, int r)
{
    const double *base = (const double *)in;
    for (size_t p = 0; p < n; ++p)
        for (int k = 0; k < 8; ++k) {
            size_t v = (k < r) ? (size_t)k : (size_t)(r - 1);
            q[p * 16 + k]     = base[2 * (v * lane_stride + p)];
            q[p * 16 + 8 + k] = base[2 * (v * lane_stride + p) + 1];
        }
}
static void gunpack_plane(const double *q, double _Complex *out,
                          size_t lane_stride, size_t n, int r)
{
    double *base = (double *)out;
    for (size_t p = 0; p < n; ++p)
        for (int k = 0; k < r; ++k) {
            base[2 * ((size_t)k * lane_stride + p)]     = q[p * 16 + k];
            base[2 * ((size_t)k * lane_stride + p) + 1] = q[p * 16 + 8 + k];
        }
}

/* Coprime factorization table for the generic sizes; modular inverse for
 * the CRT output coefficients A = Q*inv(Q mod P, P), B = P*inv(P mod Q, Q). */
static int gfactor(int L, int *P, int *Q)
{
    switch (L) {
    case 6:  *P = 2; *Q = 3; return 1;
    case 14: *P = 2; *Q = 7; return 1;
    case 18: *P = 2; *Q = 9; return 1;
    case 21: *P = 3; *Q = 7; return 1;
    case 24: *P = 3; *Q = 8; return 1;
    case 28: *P = 4; *Q = 7; return 1;
    case 35: *P = 5; *Q = 7; return 1;
    case 36: *P = 4; *Q = 9; return 1;
    case 45: *P = 5; *Q = 9; return 1;
    case 56: *P = 7; *Q = 8; return 1;
    case 63: *P = 7; *Q = 9; return 1;
    /* r6 widening (50/80/100 were deliberately absent: pfa_large/powp
     * cells.  r11 CLAIMS 50 and 100 -- the all-hands-on-L=100 rounds
     * explicitly invite cross-class entries; the race picks winners.
     * 80=5x16 stays unclaimed: unscored, and pfa_large covers it.) */
    case 50:  *P = 2; *Q = 25; return 1;   /* r11 */
    case 100: *P = 4; *Q = 25; return 1;   /* r11 */
    case 22:  *P = 2; *Q = 11; return 1;
    case 26:  *P = 2; *Q = 13; return 1;
    case 30:  *P = 2; *Q = 15; return 1;
    case 33:  *P = 3; *Q = 11; return 1;
    case 39:  *P = 3; *Q = 13; return 1;
    case 42:  *P = 2; *Q = 21; return 1;
    case 44:  *P = 4; *Q = 11; return 1;
    case 48:  *P = 3; *Q = 16; return 1;
    case 52:  *P = 4; *Q = 13; return 1;
    case 54:  *P = 2; *Q = 27; return 1;
    case 55:  *P = 5; *Q = 11; return 1;
    case 60:  *P = 4; *Q = 15; return 1;
    case 65:  *P = 5; *Q = 13; return 1;
    case 72:  *P = 8; *Q = 9;  return 1;
    case 75:  *P = 3; *Q = 25; return 1;
    case 77:  *P = 7; *Q = 11; return 1;
    case 84:  *P = 4; *Q = 21; return 1;
    case 88:  *P = 8; *Q = 11; return 1;
    case 91:  *P = 7; *Q = 13; return 1;
    case 99:  *P = 9; *Q = 11; return 1;
    case 104: *P = 8; *Q = 13; return 1;
    case 105: *P = 7; *Q = 15; return 1;
    case 108: *P = 4; *Q = 27; return 1;
    case 112: *P = 7; *Q = 16; return 1;
    case 117: *P = 9; *Q = 13; return 1;
    case 120: *P = 8; *Q = 15; return 1;
    /* r6 nested-PFA-module sizes (2 x composite odd, all IPOK); the flat
     * fold build (-DGMODPFA=0) cannot serve these (tables sized h <= 13) */
    case 66:  if (!GMODPFA) break; *P = 2; *Q = 33; return 1;
    case 70:  if (!GMODPFA) break; *P = 2; *Q = 35; return 1;
    case 78:  if (!GMODPFA) break; *P = 2; *Q = 39; return 1;
    case 90:  if (!GMODPFA) break; *P = 2; *Q = 45; return 1;
    case 110: if (!GMODPFA) break; *P = 2; *Q = 55; return 1;
    case 126: if (!GMODPFA) break; *P = 2; *Q = 63; return 1;
    /* r6 prime modules 17..31 (dense fold); libraries collapse at these
     * factors (MKL 10x behind at L=31), so 2^a*p and 3/5/7*p composites
     * are prime round-6 draw material */
    case 34:  *P = 2; *Q = 17; return 1;
    case 38:  *P = 2; *Q = 19; return 1;
    case 46:  *P = 2; *Q = 23; return 1;
    case 51:  *P = 3; *Q = 17; return 1;
    case 57:  *P = 3; *Q = 19; return 1;
    case 58:  *P = 2; *Q = 29; return 1;
    case 62:  *P = 2; *Q = 31; return 1;
    case 68:  *P = 4; *Q = 17; return 1;
    case 69:  *P = 3; *Q = 23; return 1;
    case 76:  *P = 4; *Q = 19; return 1;
    case 85:  *P = 5; *Q = 17; return 1;
    case 87:  *P = 3; *Q = 29; return 1;
    case 92:  *P = 4; *Q = 23; return 1;
    case 93:  *P = 3; *Q = 31; return 1;
    case 95:  *P = 5; *Q = 19; return 1;
    case 102: if (!GMODPFA) break; *P = 2; *Q = 51; return 1;
    case 114: if (!GMODPFA) break; *P = 2; *Q = 57; return 1;
    case 115: *P = 5; *Q = 23; return 1;
    case 116: *P = 4; *Q = 29; return 1;
    case 119: *P = 7; *Q = 17; return 1;
    case 124: *P = 4; *Q = 31; return 1;
    }
    return 0;
}
static int ginv(int a, int m)
{
    a %= m;
    for (int t = 1; t < m; ++t)
        if (a * t % m == 1) return t;
    return 1; /* m == 1 */
}
static void godd_tables(int n, double *cs, double *sn)
{
    if (!(n & 1)) return;
    int h = n >> 1;
    const long double TP = 2.0L * acosl(-1.0L);
    for (int k = 1; k <= h; ++k)
        for (int j = 1; j <= h; ++j) {
            int m = (k * j) % n;
            cs[(size_t)(k - 1) * h + j - 1] = (double)cosl(TP * m / n);
            sn[(size_t)(k - 1) * h + j - 1] = (double)sinl(TP * m / n);
        }
}
static void gtabs_init(struct gtabs *g, int L, int P, int Q)
{
    g->P = P;  g->Q = Q;
    int A = Q * ginv(Q % P, P) % L;
    int B = P * ginv(P % Q, Q) % L;
    for (int j2 = 0; j2 < Q; ++j2)
        for (int j1 = 0; j1 < P; ++j1)
            g->inmap[(size_t)j2 * P + j1] = (int16_t)((Q * j1 + P * j2) % L);
    for (int k1 = 0; k1 < P; ++k1)
        for (int j2 = 0; j2 < Q; ++j2)
            g->outmap[(size_t)k1 * Q + j2] = (int16_t)((A * k1 + B * j2) % L);
    godd_tables(P, g->csP, g->snP);
    if (GMODPFA && gsplit(Q, &g->q1, &g->q2)) { /* nested-PFA tables (r6) */
        int q1 = g->q1, q2 = g->q2;
        int Am = q2 * ginv(q2 % q1, q1) % Q;
        int Bm = q1 * ginv(q1 % q2, q2) % Q;
        for (int b = 0; b < q2; ++b)
            for (int a = 0; a < q1; ++a)
                g->qin[(size_t)b * q1 + a] = (int16_t)((q2 * a + q1 * b) % Q);
        for (int a = 0; a < q1; ++a)
            for (int b = 0; b < q2; ++b)
                g->qout[(size_t)a * q2 + b] = (int16_t)((Am * a + Bm * b) % Q);
        godd_tables(q1, g->csq1, g->snq1);
        godd_tables(q2, g->csq2, g->snq2);
    } else {
        g->q1 = g->q2 = 0;
        godd_tables(Q, g->csQ, g->snQ);
    }
    if (Q == 25 || P == 25) {   /* r11: W25 CT twiddles, consumption order */
        const long double TP = 2.0L * acosl(-1.0L);
        for (int n2 = 0; n2 < 5; ++n2)
            for (int k1 = 0; k1 < 5; ++k1) {
                int m = (n2 * k1) % 25;
                g->t25c[n2 * 5 + k1] = (double)cosl(TP * m / 25);
                g->t25s[n2 * 5 + k1] = (double)sinl(TP * m / 25);
            }
    }
}

/* ----------------------------------------------------- packing helpers */

/* One x-plane of 8 interleaved volumes (lane stride vol complex) -> an
 * interleaved site arena (site = re[8]|im[8]) at site stride sst doubles:
 * sst = 16 packs a contiguous plane (the state arena); sst = L*16 packs
 * site c of plane x to q0 + x*16 + c*L*16, i.e. the x-pass consumption
 * order used for the chain's c field. */
static void pack8_plane(const double _Complex *in, size_t lane_stride,
                        size_t n, double *q, size_t sst)
{
    const double *base = (const double *)in;
    size_t p = 0;
    for (; p + 4 <= n; p += 4) {
        v8 rows[8];
        for (int k = 0; k < 8; ++k)
            rows[k] = vload(base + 2 * ((size_t)k * lane_stride + p));
        tr8(rows);
        for (int j = 0; j < 4; ++j) {
            vstore(q + (p + j) * sst,     rows[2 * j]);
            vstore(q + (p + j) * sst + 8, rows[2 * j + 1]);
        }
    }
    for (; p < n; ++p)
        for (int k = 0; k < 8; ++k) {
            q[p * sst + k]     = base[2 * ((size_t)k * lane_stride + p)];
            q[p * sst + 8 + k] = base[2 * ((size_t)k * lane_stride + p) + 1];
        }
}

static void unpack8_plane(const double *q, double _Complex *out,
                          size_t lane_stride, size_t n)
{
    double *base = (double *)out;
    size_t p = 0;
    for (; p + 4 <= n; p += 4) {
        v8 rows[8];
        for (int j = 0; j < 4; ++j) {
            rows[2 * j]     = vload(q + (p + j) * 16);
            rows[2 * j + 1] = vload(q + (p + j) * 16 + 8);
        }
        tr8(rows);
        for (int k = 0; k < 8; ++k)
            vstore(base + 2 * ((size_t)k * lane_stride + p), rows[k]);
    }
    for (; p < n; ++p)
        for (int k = 0; k < 8; ++k) {
            base[2 * ((size_t)k * lane_stride + p)]     = q[p * 16 + k];
            base[2 * ((size_t)k * lane_stride + p) + 1] = q[p * 16 + 8 + k];
        }
}

static void deinterleave(const double _Complex *x, double *r, double *i, size_t n)
{
    const double *p = (const double *)x;
    for (size_t k = 0; k < n; ++k) { r[k] = p[2 * k]; i[k] = p[2 * k + 1]; }
}

static void interleave(const double *r, const double *i, double _Complex *x, size_t n)
{
    double *p = (double *)x;
    for (size_t k = 0; k < n; ++k) { p[2 * k] = r[k]; p[2 * k + 1] = i[k]; }
}

/* r7: the half-turn step ROTATES the layout each step (natural [x][y][z]
 * -> [z][x][y] -> [y][z][x] -> natural, period 3), so a chain of m steps
 * ends in rotation p = m mod 3 and the one-time unpack undoes it (scalar;
 * ~L^3 once per chain, not per step).  p=1: state mem [z][x][y]; p=2:
 * state mem [y][z][x]. */
static void interleave_rot(const double *r, const double *i,
                           double _Complex *x, int L, int p)
{
    double *o = (double *)x;
    for (int xx = 0; xx < L; ++xx)
        for (int yy = 0; yy < L; ++yy)
            for (int zz = 0; zz < L; ++zz) {
                size_t src = (p == 1)
                    ? (size_t)zz * L * L + (size_t)xx * L + yy
                    : (size_t)yy * L * L + (size_t)zz * L + xx;
                size_t dst = (size_t)xx * L * L + (size_t)yy * L + zz;
                o[2 * dst]     = r[src];
                o[2 * dst + 1] = i[src];
            }
}

/* r11b: the slab step's half-turn stores SWAP the two inner dims each step
 * (period 2); a chain of odd m ends in the swapped layout [x][z][y] and
 * this one-time unpack undoes it (scalar, once per chain). */
static void interleave_swp(const double *r, const double *i,
                           double _Complex *x, int L)
{
    double *o = (double *)x;
    for (int xx = 0; xx < L; ++xx)
        for (int yy = 0; yy < L; ++yy)
            for (int zz = 0; zz < L; ++zz) {
                size_t src = (size_t)xx * L * L + (size_t)zz * L + yy;
                size_t dst = (size_t)xx * L * L + (size_t)yy * L + zz;
                o[2 * dst]     = r[src];
                o[2 * dst + 1] = i[src];
            }
}

/* r12 WVS pack/unpack: natural volume <-> the NS-slab site arena.  Pad x
 * lanes and pad z sites are ZEROED at pack (see the gstep_wvs header note
 * for why they then stay zero through every step). */
static void wvs_row_tail(const double _Complex *in, size_t lane_stride,
                         size_t n, double *q, int r)
{
    const double *base = (const double *)in;
    for (size_t p = 0; p < n; ++p)
        for (int k = 0; k < 8; ++k) {
            double re = 0.0, im = 0.0;
            if (k < r) {
                re = base[2 * ((size_t)k * lane_stride + p)];
                im = base[2 * ((size_t)k * lane_stride + p) + 1];
            }
            q[p * 16 + k]     = re;
            q[p * 16 + 8 + k] = im;
        }
}

static void wvs_pack_state(const double _Complex *x0, double *W, int L,
                           int ZP, int ZP8, int NS, int SLST)
{
    const size_t LL = (size_t)L * L;
    for (int s = 0; s < NS; ++s) {
        int rs = L - 8 * s;
        if (rs > 8) rs = 8;
        double *pl = W + (size_t)s * SLST * 16;
        const double _Complex *src = x0 + (size_t)(8 * s) * LL;
        for (int y = 0; y < L; ++y) {
            double *row = pl + (size_t)y * ZP * 16;
            if (rs == 8)
                pack8_plane(src + (size_t)y * L, LL, (size_t)L, row, 16);
            else
                wvs_row_tail(src + (size_t)y * L, LL, (size_t)L, row, rs);
            if (ZP8 > L)
                memset(row + (size_t)L * 16, 0,
                       (size_t)(ZP8 - L) * 16 * sizeof(double));
        }
    }
}

/* c in x-pass consumption order: block (y, zb) holds L sites (index = x,
 * lanes = z in zb*8..zb*8+7), so the pencil's fused map reads c at
 * cpb + x*16 exactly like the batched engine's C.  Built by packing each
 * slab row (site z, lanes x -- the state layout, cheap via pack8_plane)
 * and tr8-ing each 8-site block once; pad z lanes come from the zeroed
 * row pads, pad x rows are simply not stored. */
static void wvs_pack_c(const double _Complex *c, double *Wc, int L,
                       int ZP8, int NS, int NBZ)
{
    const size_t LL = (size_t)L * L;
    double row[16 * 8 * 16] __attribute__((aligned(64)));  /* ZP8 <= 128 */
    for (int s = 0; s < NS; ++s) {
        int rs = L - 8 * s;
        if (rs > 8) rs = 8;
        const double _Complex *src = c + (size_t)(8 * s) * LL;
        for (int y = 0; y < L; ++y) {
            if (rs == 8)
                pack8_plane(src + (size_t)y * L, LL, (size_t)L, row, 16);
            else
                wvs_row_tail(src + (size_t)y * L, LL, (size_t)L, row, rs);
            if (ZP8 > L)
                memset(row + (size_t)L * 16, 0,
                       (size_t)(ZP8 - L) * 16 * sizeof(double));
            for (int zb = 0; zb < NBZ; ++zb) {
                const double *rp = row + (size_t)zb * 8 * 16;
                v8 br[8], bi[8];
                for (int q = 0; q < 8; ++q) br[q] = vload(rp + q * 16);
                for (int q = 0; q < 8; ++q) bi[q] = vload(rp + q * 16 + 8);
                tr8(br);
                tr8(bi);
                double *dst = Wc + ((size_t)y * NBZ + zb) * (size_t)L * 16;
                for (int j = 0; j < rs; ++j) {
                    vstore(dst + (size_t)(8 * s + j) * 16, br[j]);
                    vstore(dst + (size_t)(8 * s + j) * 16 + 8, bi[j]);
                }
            }
        }
    }
}

static void wvs_unpack(const double *W, double _Complex *out, int L,
                       int ZP, int NS, int SLST)
{
    const size_t LL = (size_t)L * L;
    for (int s = 0; s < NS; ++s) {
        int rs = L - 8 * s;
        if (rs > 8) rs = 8;
        const double *pl = W + (size_t)s * SLST * 16;
        double _Complex *dst = out + (size_t)(8 * s) * LL;
        for (int y = 0; y < L; ++y) {
            if (rs == 8)
                unpack8_plane(pl + (size_t)y * ZP * 16,
                              dst + (size_t)y * L, LL, (size_t)L);
            else
                gunpack_plane(pl + (size_t)y * ZP * 16,
                              dst + (size_t)y * L, LL, (size_t)L, rs);
        }
    }
}

/* ------------------------------------------------------------------ plan */

/* 2 MiB-aligned anonymous mapping + MADV_HUGEPAGE (node THP is madvise
 * mode): with 4K pages the arena's physical page coloring varies per run
 * and L=15 measured 4.50-5.87 us run to run (in-run sd 0.05%); a huge-page
 * arena is physically contiguous, so L2 indexing is deterministic and
 * matches the best-case runs. */
static double *arena_alloc(size_t bytes, size_t *out_len)
{
    const size_t HP = (size_t)1 << 21;
    size_t len = (bytes + HP - 1) & ~(HP - 1);
    char *raw = mmap(NULL, len + HP, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw == MAP_FAILED) return NULL;
    uintptr_t a = ((uintptr_t)raw + HP - 1) & ~(uintptr_t)(HP - 1);
    size_t head = a - (uintptr_t)raw;
    if (head) munmap(raw, head);
    if (HP - head) munmap((char *)a + len, HP - head);
#ifdef MADV_HUGEPAGE
    madvise((void *)a, len, MADV_HUGEPAGE);
#endif
    *out_len = len;
    return (double *)a;
}

struct fft3d_plan {
    int L, batch, PL, generic;
    size_t vol, arena_len;
    double *arena;
    double *S, *C;                       /* interleaved site arenas        */
    double *Sr, *Si, *Dr, *Di, *Cr, *Ci; /* split per-volume: vol each     */
    double *Cra, *Cia, *Crb, *Cib;       /* r7: c in the two rotations     */
    double *wtr, *wti;                   /* r7: dense DFT-matrix rows      */
    void (*soa_fft)(double *);
    void (*soa_step)(double *, const double *);
    void (*soa_chain)(double *, const double *, int);
    void (*split_fft)(double *, double *, double *, double *);
    struct gtabs gt;                     /* generic coprime-pair tables    */
    gpen_fn gpen;                        /* specialized (P,Q) pencil       */
    gspen_fn gspen;                      /* r8: out-of-place split pencil  */
    gspenip_fn gspen_ip;                 /* r11c: in-place (IPOK) or NULL  */
    double *Wq, *Wcq;                    /* r12: within-volume SoA arenas  */
    int wvs_zp, wvs_zp8, wvs_ns, wvs_slst, wvs_nbz;
    struct gtabs gtw;                    /* r12b: role-swapped tables      */
    gpen_fn gpen_w;                      /* r12b: swapped WVS pencil/NULL  */
    gpen_fn gpen_z;                      /* r12c: WVS z-sweep pencil/NULL  */
};

fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L) || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->vol = (size_t)L * L * L;
    p->PL = plane_stride_sites(L);

    /* Arena is 4096-aligned; S sits at offset 0, C at an offset == 2048
     * (mod 4096) so state and c never collide in the low address bits
     * (gen_batchlane / bl8's de-alias offset).  Split buffers keep the r1
     * one-line stagger between components. */
    int tuned = (L == 10 || L == 12 || L == 15 || L == 20);
    size_t soa = (size_t)L * p->PL * 16;               /* doubles, one arena */
    size_t coff = ((soa + 511) / 512) * 512 + 256;
    size_t svol = p->vol + 8;
    /* split per-volume buffers for the tuned sizes and (r8) for generic
     * sizes whose batch leaves a remainder group the rotation split chain
     * will serve (r6 rule otherwise: the generic path never touches them,
     * saving 6*L^3 doubles at the big pure-batched draws).
     * r7 adds the two rotated c copies (4 svol) + dense DFT-matrix tables. */
    int gsplit_bufs = !tuned && L >= 8 && batch % 8 != 0 &&
                      batch % 8 <= GSPLIT_RMAX;
    size_t wsz = (size_t)L * (((size_t)L + 7) / 8) * 8;
    /* r12: within-volume SoA arenas (see gstep_wvs).  wvsok implies
     * gsplit_bufs, so the r11 slab path stays raceable (-DGWVS=0). */
    int wvsok = GWVS && gsplit_bufs && L >= GWVS_MIN;
    size_t wvs_q = 0, wvs_c = 0, woff = 0;
    if (wvsok) {
        p->wvs_ns  = (L + 7) / 8;
        p->wvs_zp8 = (L + 7) & ~7;
        p->wvs_zp  = p->wvs_zp8 + 1;          /* odd y-pencil site stride */
        int sl = L * p->wvs_zp;
        sl += ((2 - sl) % 32 + 32) % 32;      /* slab bytes == 256 mod 4096 */
        p->wvs_slst = sl;
        p->wvs_nbz  = p->wvs_zp8 / 8;
        wvs_q = (size_t)p->wvs_ns * sl * 16;
        wvs_c = (size_t)L * p->wvs_nbz * L * 16;
    }
    size_t total = coff + soa + ((tuned || gsplit_bufs) ? 10 * svol : 0) +
                   (tuned ? 2 * wsz : 0);
    if (wvsok) {
        woff = (total + 15) & ~(size_t)15;    /* 128 B site alignment */
        total = woff + wvs_q + 16 + wvs_c;    /* one-site stagger between */
    }
    p->arena = arena_alloc(total * sizeof(double), &p->arena_len);
    if (!p->arena) {
        free(p);
        return NULL;
    }
    memset(p->arena, 0, total * sizeof(double));  /* fault in as huge pages */
    p->S = p->arena;
    p->C = p->arena + coff;
    if (wvsok) {
        p->Wq  = p->arena + woff;
        p->Wcq = p->Wq + wvs_q + 16;
    }
    if (tuned || gsplit_bufs) {
        p->Sr = p->C + soa;
        p->Si = p->Sr + svol;
        p->Dr = p->Si + svol;
        p->Di = p->Dr + svol;
        p->Cr = p->Di + svol;
        p->Ci = p->Cr + svol;
        p->Cra = p->Ci + svol;
        p->Cia = p->Cra + svol;
        p->Crb = p->Cia + svol;
        p->Cib = p->Crb + svol;
    }
    if (tuned) {
        p->wtr = p->Cib + svol;
        p->wti = p->wtr + wsz;
        /* dense pass-3 tables: row n = e^{-2pi i n k / L} over k, padded
         * columns k >= L ZERO so masked-out accumulator lanes stay exactly
         * 0 through the map ladder (no denormal/NaN slow paths). */
        {
            const long double TP = 2.0L * acosl(-1.0L);
            int nv8 = (int)(wsz / L);
            for (int n = 0; n < L; ++n)
                for (int k = 0; k < nv8; ++k) {
                    size_t idx = (size_t)n * nv8 + k;
                    if (k < L) {
                        int mm = (n * k) % L;
                        p->wtr[idx] = (double)cosl(TP * mm / L);
                        p->wti[idx] = -(double)sinl(TP * mm / L);
                    } else {
                        p->wtr[idx] = 0.0;
                        p->wti[idx] = 0.0;
                    }
                }
        }
    }

    switch (L) {
    case 10: p->soa_fft = soa_fft_10; p->soa_step = soa_step_10; p->soa_chain = soa_chain_10; p->split_fft = split_fft_10; break;
    case 12: p->soa_fft = soa_fft_12; p->soa_step = soa_step_12; p->soa_chain = soa_chain_12; p->split_fft = split_fft_12; break;
    case 15: p->soa_fft = soa_fft_15; p->soa_step = soa_step_15; p->soa_chain = soa_chain_15; p->split_fft = split_fft_15; break;
    case 20: p->soa_fft = soa_fft_20; p->soa_step = soa_step_20; p->soa_chain = soa_chain_20; p->split_fft = split_fft_20; break;
    default: {
        int P, Q;
        gfactor(L, &P, &Q);
        gtabs_init(&p->gt, L, P, Q);
        p->gpen = gpen_lookup(P, Q);
        p->gspen = gspen_lookup(P, Q);
        p->gspen_ip = gspenip_lookup(P, Q);
        /* r12b/c: WVS pencil routing at the 25-pairs -- z-sweep takes the
         * formula-baked IPOK pencil; the y-sweep and x-pass take the
         * formula-baked SWAPPED pencil where the swap is wanted (see the
         * GWVSSW note), else the same unswapped one. */
        if (wvsok && Q == 25 && (P == 2 || P == 4)) {
            p->gpen_z = (P == 4) ? gpencilf_4_25 : gpencilf_2_25;
            if ((P == 4 && GWVSSW) || (P == 2 && GWVSSW50)) {
                p->gpen_w = (P == 4) ? gpencilf_25_4 : gpencilf_25_2;
                gtabs_init(&p->gtw, L, Q, P);
            } else {
                p->gpen_w = p->gpen_z;
                p->gtw = p->gt;
            }
        }
        p->generic = 1;
        break;
    }
    }
    return p;
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    const size_t vol = p->vol;
    const size_t LL = (size_t)p->L * p->L;
    const size_t pstride = (size_t)p->PL * 16;
    const int g8 = p->batch / 8;

    for (int g = 0; g < g8; ++g) {
        const double _Complex *src = in + (size_t)g * 8 * vol;
        double _Complex *dst = out + (size_t)g * 8 * vol;
        for (int x = 0; x < p->L; ++x)
            pack8_plane(src + x * LL, vol, LL, p->S + x * pstride, 16);
        if (p->generic) gsoa_fft(p->S, p->L, p->PL, &p->gt, p->gpen);
        else            p->soa_fft(p->S);
        for (int x = 0; x < p->L; ++x)
            unpack8_plane(p->S + x * pstride, dst + x * LL, vol, LL);
    }
    if (p->generic) {           /* remainder group: lane-replicated SoA */
        int r = p->batch - g8 * 8;
        if (r > 0) {
            const double _Complex *src = in + (size_t)g8 * 8 * vol;
            double _Complex *dst = out + (size_t)g8 * 8 * vol;
            for (int x = 0; x < p->L; ++x)
                gpack_plane(src + x * LL, vol, LL, p->S + x * pstride, r);
            gsoa_fft(p->S, p->L, p->PL, &p->gt, p->gpen);
            for (int x = 0; x < p->L; ++x)
                gunpack_plane(p->S + x * pstride, dst + x * LL, vol, LL, r);
        }
        return;
    }
    for (int v = g8 * 8; v < p->batch; ++v) {
        deinterleave(in + (size_t)v * vol, p->Sr, p->Si, vol);
        p->split_fft(p->Sr, p->Si, p->Dr, p->Di);
        interleave(p->Dr, p->Di, out + (size_t)v * vol, vol);
    }
}

/* The whole graded chain: state <- (FFT(state)+c)/(1+|FFT(state)+c|), m
 * times, final MAPPED state to final_out.  State lives in the site arena
 * across all m steps: pack twice, unpack once, map fused into the x pass. */
void fft3d_chain(fft3d_plan *p, const double _Complex *x0,
                 const double _Complex *c, double _Complex *final_out, int m)
{
    const size_t vol = p->vol;
    const size_t LL = (size_t)p->L * p->L;
    const size_t pstride = (size_t)p->PL * 16;
    const int g8 = p->batch / 8;

    for (int g = 0; g < g8; ++g) {
        const size_t off = (size_t)g * 8 * vol;
        for (int x = 0; x < p->L; ++x) {
            pack8_plane(x0 + off + x * LL, vol, LL, p->S + x * pstride, 16);
            pack8_plane(c + off + x * LL, vol, LL, p->C + x * pstride, 16);
        }
        if (p->generic)
            for (int s = 0; s < m; ++s)
                gsoa_step(p->S, p->C, p->L, p->PL, &p->gt, p->gpen);
        else
            p->soa_chain(p->S, p->C, m);
        for (int x = 0; x < p->L; ++x)
            unpack8_plane(p->S + x * pstride, final_out + off + x * LL, vol, LL);
    }
    if (p->generic) {
        int r = p->batch - g8 * 8;
        if (r > 0 && p->Wq) {
            /* r12: within-volume SoA chain (lanes = 8 x-planes; see
             * gstep_wvs).  Everything in place, natural layout, no
             * rotation/parity bookkeeping; pack twice + unpack once per
             * volume, all m steps in the slab arena. */
            const int L = p->L;
            for (int v = g8 * 8; v < p->batch; ++v) {
                const size_t off = (size_t)v * vol;
                wvs_pack_state(x0 + off, p->Wq, L, p->wvs_zp, p->wvs_zp8,
                               p->wvs_ns, p->wvs_slst);
                wvs_pack_c(c + off, p->Wcq, L, p->wvs_zp8, p->wvs_ns,
                           p->wvs_nbz);
                gpen_fn zpen = p->gpen_z ? p->gpen_z : p->gpen;
                const struct gtabs *wg = p->gpen_w ? &p->gtw : &p->gt;
                gpen_fn wpen = p->gpen_w ? p->gpen_w : p->gpen;
                for (int s = 0; s < m; ++s)
                    gstep_wvs(p->Wq, p->Wcq, L, p->wvs_zp, p->wvs_ns,
                              p->wvs_slst, p->wvs_nbz, &p->gt, zpen,
                              wg, wpen);
                wvs_unpack(p->Wq, final_out + off, L, p->wvs_zp,
                           p->wvs_ns, p->wvs_slst);
            }
        } else if (r > 0 && p->Sr && p->L >= GSLAB_MIN) {
            /* r11: large-L slab-fused split chain (natural layout, no
             * rotation bookkeeping; see gstep_slab) */
            const int L = p->L;
            for (int v = g8 * 8; v < p->batch; ++v) {
                const size_t off = (size_t)v * vol;
                deinterleave(x0 + off, p->Sr, p->Si, vol);
                deinterleave(c + off, p->Cr, p->Ci, vol);
                if (GSLABSW)    /* c in the inner-swapped layout (Cra) */
                    for (int xx = 0; xx < L; ++xx)
                        for (int yy = 0; yy < L; ++yy)
                            for (int zz = 0; zz < L; ++zz) {
                                size_t sN = (size_t)xx * L * L
                                          + (size_t)yy * L + zz;
                                size_t sW = (size_t)xx * L * L
                                          + (size_t)zz * L + yy;
                                p->Cra[sW] = p->Cr[sN];
                                p->Cia[sW] = p->Ci[sN];
                            }
                /* state stays in S the whole chain (pass 1 in place, D is
                 * a slab scratch); half-turn stores flip the inner-dim
                 * parity each step: even steps store into the swapped
                 * layout (map reads Cra), odd steps store back to natural
                 * (Cr); one unpermute at chain end when m is odd */
                for (int s = 0; s < m; ++s) {
                    const double *ucr = (GSLABSW && !(s & 1)) ? p->Cra : p->Cr;
                    const double *uci = (GSLABSW && !(s & 1)) ? p->Cia : p->Ci;
                    gstep_slab(p->Sr, p->Si, p->Dr, p->Di, ucr, uci,
                               L, &p->gt, p->gspen, p->gspen_ip);
                }
                if (GSLABSW && (m & 1))
                    interleave_swp(p->Sr, p->Si, final_out + off, L);
                else
                    interleave(p->Sr, p->Si, final_out + off, vol);
            }
        } else if (r > 0 && p->Sr) { /* r8: per-volume rotation split chain */
            const int L = p->L;
            for (int v = g8 * 8; v < p->batch; ++v) {
                const size_t off = (size_t)v * vol;
                deinterleave(x0 + off, p->Sr, p->Si, vol);
                deinterleave(c + off, p->Cr, p->Ci, vol);
                for (int xx = 0; xx < L; ++xx)      /* c in both rotations */
                    for (int yy = 0; yy < L; ++yy)
                        for (int zz = 0; zz < L; ++zz) {
                            size_t sN = (size_t)xx * L * L + (size_t)yy * L + zz;
                            size_t sA = (size_t)zz * L * L + (size_t)xx * L + yy;
                            size_t sB = (size_t)yy * L * L + (size_t)zz * L + xx;
                            p->Cra[sA] = p->Cr[sN];  p->Cia[sA] = p->Ci[sN];
                            p->Crb[sB] = p->Cr[sN];  p->Cib[sB] = p->Ci[sN];
                        }
                const double *crs[3] = { p->Cr, p->Cra, p->Crb };
                const double *cis[3] = { p->Ci, p->Cia, p->Cib };
                double *sr = p->Sr, *si = p->Si, *dr = p->Dr, *di = p->Di;
                for (int s = 0; s < m; ++s) {
                    const int q = (s + 1) % 3;  /* this step's output rot */
                    gstep_split(sr, si, dr, di, crs[q], cis[q],
                                L, &p->gt, p->gspen);
                    double *t;
                    t = sr; sr = dr; dr = t;
                    t = si; si = di; di = t;
                }
                const int pr = m % 3;
                if (pr) interleave_rot(sr, si, final_out + off, L, pr);
                else    interleave(sr, si, final_out + off, vol);
            }
        } else if (r > 0) {     /* remainder group: lane-replicated SoA */
            const size_t off = (size_t)g8 * 8 * vol;
            for (int x = 0; x < p->L; ++x) {
                gpack_plane(x0 + off + x * LL, vol, LL, p->S + x * pstride, r);
                gpack_plane(c + off + x * LL, vol, LL, p->C + x * pstride, r);
            }
            for (int s = 0; s < m; ++s)
                gsoa_step(p->S, p->C, p->L, p->PL, &p->gt, p->gpen);
            for (int x = 0; x < p->L; ++x)
                gunpack_plane(p->S + x * pstride,
                              final_out + off + x * LL, vol, LL, r);
        }
        return;
    }
    /* r7: fused-map split steps.  SPLITZ<L>=1 (half-turn) ROTATES the
     * layout each step (period 3): the map's c must match the step's
     * OUTPUT layout, so steps cycle through c, c-rot1, c-rot2, and the
     * chain un-rotates once at the end if m % 3 != 0.  SPLITZ<L>=2 (dense)
     * and =0 (r6 sandwich + map_span) keep the natural layout throughout. */
    const int zv = (p->L == 10) ? SPLITZ10 : (p->L == 12) ? SPLITZ12
                 : (p->L == 15) ? SPLITZ15 : SPLITZ20;
    for (int v = g8 * 8; v < p->batch; ++v) {
        const size_t off = (size_t)v * vol;
        deinterleave(x0 + off, p->Sr, p->Si, vol);
        deinterleave(c + off, p->Cr, p->Ci, vol);
        double *sr = p->Sr, *si = p->Si, *dr = p->Dr, *di = p->Di;
        if (zv == 0) {
            for (int s = 0; s < m; ++s) {
                switch (p->L) {
                case 10: split_fft_10(sr, si, dr, di); break;
                case 12: split_fft_12(sr, si, dr, di); break;
                case 15: split_fft_15(sr, si, dr, di); break;
                case 20: split_fft_20(sr, si, dr, di); break;
                }
                map_span(dr, di, p->Cr, p->Ci, (ptrdiff_t)vol);
                double *t;
                t = sr; sr = dr; dr = t;
                t = si; si = di; di = t;
            }
            interleave(sr, si, final_out + off, vol);
            continue;
        }
        if (zv == 1) {          /* c in the two rotated layouts */
            const int L = p->L;
            for (int xx = 0; xx < L; ++xx)
                for (int yy = 0; yy < L; ++yy)
                    for (int zz = 0; zz < L; ++zz) {
                        size_t sN = (size_t)xx * L * L + (size_t)yy * L + zz;
                        size_t sA = (size_t)zz * L * L + (size_t)xx * L + yy;
                        size_t sB = (size_t)yy * L * L + (size_t)zz * L + xx;
                        p->Cra[sA] = p->Cr[sN];  p->Cia[sA] = p->Ci[sN];
                        p->Crb[sB] = p->Cr[sN];  p->Cib[sB] = p->Ci[sN];
                    }
        }
        const double *crs[3] = { p->Cr, p->Cra, p->Crb };
        const double *cis[3] = { p->Ci, p->Cia, p->Cib };
        for (int s = 0; s < m; ++s) {
            /* this step's output rotation is (s+1) mod 3 under half-turn */
            const int q = (zv == 1) ? (s + 1) % 3 : 0;
            switch (p->L) {
            case 10: step_10(sr, si, dr, di, crs[q], cis[q], p->wtr, p->wti); break;
            case 12: step_12(sr, si, dr, di, crs[q], cis[q], p->wtr, p->wti); break;
            case 15: step_15(sr, si, dr, di, crs[q], cis[q], p->wtr, p->wti); break;
            case 20: step_20(sr, si, dr, di, crs[q], cis[q], p->wtr, p->wti); break;
            }
            double *t;
            t = sr; sr = dr; dr = t;
            t = si; si = di; di = t;
        }
        const int pr = (zv == 1) ? m % 3 : 0;
        if (pr) interleave_rot(sr, si, final_out + off, p->L, pr);
        else    interleave(sr, si, final_out + off, vol);
    }
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    munmap(p->arena, p->arena_len);
    free(p);
}
