/* gen_powp -- prime-power Cooley-Tukey with general exact twiddles:
 * L = 25 (5x5), 27 (3x9, DFT9 = 3x3), 50 (PFA 2 x 25), 100 (PFA 4 x 25),
 * and from gen_r3 the whole odd-p^k class: 49 (7x7), 81 (9x9), 121 (11x11),
 * 125 (5x25) -- any prime-power draw in round 6's 14..127 range that is not
 * gen_pow2's 2^k is served here.
 *
 * ROUND gen_r5 (changes on top of r4):
 *
 *   1. PAIR-PACKED map ladder in map_span (ADOPTED from gen_pfa_large
 *      gen_r5, their map_step_pair): the sequential map paths ran the NR
 *      ladder on vectors where each |z|^2 is duplicated in both complex
 *      lanes -- half the ladder lanes were redundant.  Two vectors' 8
 *      distinct |z|^2 now pack into ONE zmm (2 shuffles), one ladder runs
 *      for both, reciprocals unpack pair-duplicated (2 shuffles): ~21
 *      arithmetic ops + 4 shuffles per PAIR vs 36 + 2.  BIT-IDENTICAL per
 *      element (q_re + q_im commutes; max/rsqrt14/rcp14/fma elementwise),
 *      so every ip* and ipp chain family and the trailing map keep their r4
 *      gate numbers.  Applies to ip* map passes, the ipp prepass, and the
 *      deferred-chain trailing span at ALL sizes.
 *   2. c-stream L3-BYPASS chain families ipq1/ipk1/iqn1 at 50/100 and ipq0
 *      at the lite sizes (ADOPTED from gen_pfa_large gen_r5): c is read
 *      exactly once per step; its only reuse is across steps, so caching it
 *      is pure pollution wherever state+c exceeds the 24 MB L3 (L=100:
 *      32 MB; 121/125: 57/62 MB).  ipq* = ipp* with the prepass fetching c
 *      via PREFETCHNTA (SKX+ NTA fills L1 only, skips L2/LLC on eviction);
 *      ipk1 = CLFLUSHOPT-ing clean c lines after use (architectural
 *      semantics where NTA policy is implementation-defined); iqn1 = ip1
 *      with the NTA-c map pass.  All bit-identical (hints/flushes move no
 *      data); ranked last, the per-host race arbitrates -- at 50 B=4 the
 *      whole batch's c is L3-resident reuse and they should lose there.
 *      pf ids 11/12/13/14; wisdom tag chain4 -> chain5.
 *   3. PAIRED-vdivpd SoA x-pass map at 25/27 (reciprocal-product trick from
 *      gen_layout gen_r5's gl_map16, adapted to the split-complex site):
 *      the final-stage map shares ONE vdivpd between two sites via
 *      q = 1/(d0 d1), r0 = q d1, r1 = q d0 -- x-pass divider occupancy
 *      -40% at 25 (15 vs 25 vdivpd/pencil), -33% at 27, for 3 extra vmulpd
 *      per pair (~1-2 ulp on the reciprocal, budget is ~60 ulp/step).
 *      Knob -DGENPWP_NOMAPPAIR restores the r2-r4 one-div-per-site form.
 *
 * ROUND gen_r4 (changes on top of r3):
 *
 *   1. VOLUME-MAJOR chain schedule for the interleaved families (ADOPTED
 *      from gen_pfa_large gen_r4, who adopted it from gen_dense_prime /
 *      gen_rader / gen_layout): fft3d_chain runs ALL m steps on one volume
 *      before touching the next, so the per-step working set drops from the
 *      whole batch's state+c to one volume's slice (0.5 MB at 25 -- L2! --
 *      4 MB at 50 B=4).  Per-volume FFT op order unchanged; bit-identical
 *      at 50/100.  The soa engine was group-major from birth (unchanged).
 *   2. ipp* plane-prepass deferred-map chain family (ADOPTED from
 *      gen_pfa_large gen_r4): the ipm schedule, but the map runs as
 *      map_span's sequential per-plane prepass into an L2-resident scratch
 *      plane (M's base) that p1body then consumes -- same traffic accounting
 *      as ipm without the ladder's port/latency footprint inside the
 *      granule-load stream (their measured ipm failure at L=100, confirmed
 *      by my r3 numbers).  Raced as ipp0/ipp1 at 50/100 (ranked between the
 *      base families and ipm), ipp0 at 25/27 and the lite sizes.
 *   3. The create() race times the VOLUME-MAJOR shape (their race fix): per
 *      volume one unmeasured warm step then R timed steps in place on that
 *      volume (R = 8/6/4 by volume bytes <=2/<=8/>8 MiB), min over 4
 *      interleaved rounds.  Wisdom tag chain2 -> chain4 so a stale
 *      step-major verdict can never be installed.
 *
 * ROUND gen_r3 (changes on top of r2):
 *
 *   1. ipm* deferred-map chain family ADOPTED from gen_pfa_large gen_r3
 *      (their file, this morning -- the r2 lesson about re-reading the seed
 *      entry's CURRENT source, applied): the graded map z/(1+|z|) is applied
 *      to the NEXT step's phase-1 z-subpass LOADS (v = map(z' + c) right
 *      after the granule load, before TRNC), so the state buffer holds the
 *      raw FFT output between steps and the separate map pass's full-volume
 *      read + RFO + write disappears (at L=100 that pass is ~32 MB of the
 *      ~112 MB DRAM-bound step).  fft3d_chain runs their schedule: step 1 =
 *      plain execute, steps 2..m = p1m+p2ip, one trailing map_span.  Raced
 *      as ipm0/ipm1 at 50/100 and ipm0 at the new lite sizes; bit-identical
 *      to the ip* families' op order.  Their two race fixes come along
 *      (both bit them first): a DISTINCT c-field buffer tcf (tin-as-c made
 *      ipm's state+c loads share lines and halved its apparent traffic),
 *      and trials run IN PLACE on tout (cfn(tout,tout,tcf) -- the graded
 *      steady state) instead of tin->tout.
 *   2. Round-3 any-size-in-class duty: the engine template is instantiated
 *      at 49/81/121/125 with new line codelets -- DFT49C = 7x7 CT (DFT7K
 *      conjugate-pair fold module, 33 FMA-port ops), DFT81C = 9x9 CT (DFT9K
 *      = 3x3 CT reusing the compile-time W9 tables), DFT121C = 11x11 CT
 *      (DFT11K fold, 75 ops), DFT125C = 5x25 CT (stage A 25 x DFT5 + 96
 *      W125 twiddles, stage B 5 x DFT25M).  Twiddle/constant tables for the
 *      new sizes are computed ONCE at create() in long double (cosl/sinl,
 *      ~19 digits -- gen_pfa_small r3's precedent for satisfying the
 *      brief's exactness rule with runtime tables), stored as {cos} +
 *      {sin,-sin} pairs so a twiddle store is VSPLAT + one 128-bit
 *      broadcast (no per-use sign flip).  The lite sizes instantiate a
 *      REDUCED template (GENLITE: ip0/ipf/f0/ipm0 candidates only) to keep
 *      compile size sane; all four pass the same create()-time refnd gate
 *      and race as the tuned sizes.  Unscored; they exist for round 6.
 *
 * ROUND gen_r2 (changes on top of the r1 design described below):
 *
 *   1. SoA-8 BATCH-LANE chain engine for L = 25/27 at batch % 8 == 0 (the
 *      graded 25/27 cases are B=16), raced as a 7th candidate "soa":
 *      8 volumes fill the zmm lanes, split-complex 128 B sites (re[8]|im[8]),
 *      ZERO shuffles inside the transform (structure ADOPTED from
 *      gen_pfa_small gen_r2 / gen_batchlane gen_r1, lineage ice bl8; their
 *      records explicitly invited this entry to take it, twiddles becoming
 *      broadcast-FMA operands).  NEW here -- the twiddle problem inside that
 *      structure: prime-power CT needs a digit permutation somewhere, so the
 *      pencil is IN PLACE via classic DIF (natural in -> digit-reversed out)
 *      on odd steps and DIT (digit-reversed in -> natural out) on even steps;
 *      every stage's read and write slot sets coincide (no whole-pencil
 *      register buffering, no self-sort pass).  The map is fused into the
 *      x-pass stores; c is packed ONCE per chain in BOTH layouts (natural for
 *      DIT steps, digit-reversed for DIF steps).  Slot algebra and twiddle
 *      placement verified against numpy before coding (DIF/DIT, both sizes).
 *      L=25: 2 stages of 5 x DFT5 (Winograd 4-constant split form, from
 *      gen_pfa_small) + 16 W25 twiddles; L=27: 3 stages of 9 x DFT3 + 28
 *      W27/W9 twiddles; twiddle store = 4 broadcast FMAs, exponents fold at
 *      compile time through the same product-indexed tables as r1.
 *   2. DFT25 stage-B outputs fused straight through the PFA wrappers' ST
 *      (ADOPTED from gen_pfa_large gen_r2, who did it first -- it was queued
 *      in both records): dft25v + R_[25] round-trip replaced by their
 *      DFT25M(LDX, STO, KMAP) macro; helps L=50/100.
 *   3. gen_race ADOPTION: the create()-race result is persisted per host in
 *      results/wisdom_<host>.json via gr_keyf/gr_sig/gr_wisdom_lookup/store.
 *      Warm create() is a file read (50 ms budget); and the pick is PINNED
 *      across processes, which is what makes a non-bit-identical candidate
 *      (soa computes in split-complex order) safe against the driver's
 *      two-process repeatability cmp -- gen_race's documented rationale.
 *   4. gen_layout ADOPTION: THP arena (gl_arena) with staggered mod-4096
 *      phases for the three SoA buffers; gl_pack8/gl_unpack8/gl_tr8x8 for the
 *      lane pack/unpack at the chain ends.
 *
 * ROUND gen_r1 (first real round; the previous file was the dense O(L^4)
 * validation stub).  This class is the campaign's twiddle problem: 25 and 27
 * are prime powers, so Good-Thomas cannot split them and every factorization
 * pays general W_N twiddles.  Technique:
 *
 *   Row-column 3D DFT, two sweeps per volume:
 *     phase 1, per x-plane:
 *       z transform: lanes = 4 y-rows, 4x4 complex-granule register
 *                    transposes on load and store, into plane scratch
 *                    pl[y][kz] (row pitch an ODD number of cache lines);
 *       y transform: lanes = 4 kz (contiguous in pl), store into the plan's
 *                    padded mid volume M (+64 B per plane -> odd line count,
 *                    no fixed mod-4096 relation to the driver's buffers).
 *     phase 2:
 *       x transform, lanes = 4 kz, tiled over the FLAT (y,z) index.  25^2
 *       and 27^2 are 1 mod 4, so the last tile OVERLAPS (recomputes 3
 *       lanes).  Out of place (M -> out) the overlap is trivially
 *       idempotent; IN PLACE the tail's inputs coincide with the previous
 *       tile's outputs, so the tail's GENL input vectors are STASHED before
 *       any tile runs and the tail recomputes from the stash.
 *
 *   The graded chain step (z = FFT3(x) + c; x <- z/(1+|z|)) is raced at
 *   create() across THREE families (winner differs by size and host):
 *     ip*: everything in place + a sequential vectorized map pass; the
 *          state volume is the only volume-sized object touched besides c
 *          (gen_pfa_large's measured winner at L=100, where fusing the map
 *          into the GENL-stream x-pass doubles the miss-stream count);
 *     ipf: in place AND the map fused into phase 2's stores -- no mid
 *          volume, no separate map pass (NEW; targets the cache-resident
 *          cases 25/27 where the extra pass is pure cost);
 *     f*:  map fused into phase 2 routed through the padded mid volume M
 *          (+64 B/plane -> odd line count, no fixed mod-4096 relation to
 *          the driver's buffers).
 *
 *   Line codelets (interleaved complex, lanes = a spectator axis, all index
 *   maps folded at compile time):
 *     L = 25:  DFT25 = 5x5 Cooley-Tukey, W25 twiddles: 16 nontrivial per
 *              line, exact long-double literals.  DFT5 is the FFTW n1_5 FMA
 *              form (16 FMA-port ops).  192 FMA-port ops + 36 swaps / line.
 *     L = 27:  DFT27 = 3x9 Cooley-Tukey (NEW this round): stage A = 9 x DFT3
 *              + 16 nontrivial W27 twiddles into U[9*k1+n2]; stage B = 3 x
 *              DFT9, where DFT9 is itself 3x3 CT with 4 nontrivial W9
 *              twiddles (short live ranges, 15 live vectors max).
 *              218 FMA-port ops + 55 swaps / line.
 *     L = 50 = 25 x 2 (coprime -> Good-Thomas, no inter-stage twiddles):
 *              stage 1: 25 x DFT2, stage 2: 2 x DFT25.  434 ops / line.
 *     L = 100 = 25 x 4: stage 1: 25 x DFT4, stage 2: 4 x DFT25.  968 / line.
 *
 * ATTRIBUTION (this file is deliberately cumulative):
 *   - The ENTIRE engine shell is adopted from gen_pfa_large (this round):
 *     two-sweep plane-fused structure, spectator lanes of 4, TRNC granule
 *     transpose, opaque-base asm barrier, heap plane scratch, padded mid
 *     volume (their measured L=100 4K-alias fix), fused NR map chain
 *     (rsqrt14/rcp14 + 2 Newton steps), create()-time scalar-reference gate
 *     + interleaved min-of-rounds race with simplest-first hysteresis, and
 *     the PFA50C/PFA100C/DFT25/DFT5/DFT4 codelets verbatim.  gen_pfa_large
 *     in turn credits L45_pfa (panel_r11), L45_mixedradix, L23_rader.
 *   - Fused map at the final-axis stores (not a separate pass): the
 *     campaign-wide lesson via gen_pfa_small / gen_batchlane / gen_pfa_large.
 *   - NEW here: the DFT27 = 3x9(3x3) codelet with its exact W27/W9 tables,
 *     the direct DFT25 line codelet for L=25, and the overlapping phase-2
 *     tail that lets the engine run at L^2 % 4 != 0 (odd prime powers).
 *
 * OPERATION COUNT (vector FMA-port ops per volume, lanes of 4):
 *   L=25:  3 *  625 * 192 =   360,000    (plus 2 granule transposes/elt)
 *   L=27:  3 *  729 * 218 =   476,766
 *   L=50:  3 * 1250 * 434 / 2 = 813,750  (+ ~8% phase-1 overlap recompute)
 *   L=100: 3 * 2500 * 968 = 7,260,000
 *
 * ACCURACY: twiddles are compile-time literals from long-double cosl/sinl
 * (~19 correct digits, exact-to-0.5ulp doubles).  create() gates every
 * candidate against an independent scalar O(L^2)-per-line reference at
 * 1e-13 rel L2 (first AND last arena volume) and gates the fused chain step
 * against execute + the driver's scalar map, falling back on any mismatch.
 *
 * Falls back to the dense O(L^4) matrix path if AVX-512 is unavailable.
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "../fft3d_api.h"

#ifndef GEN_POWP_ONCE               /* ============ COMMON, first pass ===== */
#define GEN_POWP_ONCE

/* library layers adopted this round (gen_r2); both are all-static includes */
#define GEN_LAYOUT_LIB_ONLY
#include "gen_layout.c"       /* gl_arena/gl_map_huge, gl_tr8x8, gl_(un)pack8 */
#define GEN_RACE_LIB_ONLY
#include "gen_race.c"         /* gr_keyf/gr_bucket/gr_sig + per-host wisdom   */

#ifdef __AVX512F__
#include <immintrin.h>

/* ---- one vector layer: 4 interleaved complex per zmm ------------------- */
typedef double    vec  __attribute__((vector_size(64)));
typedef double    uvec __attribute__((vector_size(64), aligned(8)));
typedef long long veci __attribute__((vector_size(64)));

#ifdef __clang__
# define VSH(a,b,...) __builtin_shufflevector(a, b, __VA_ARGS__)
#else
# define VSH(a,b,...) __builtin_shuffle(a, b, (veci){__VA_ARGS__})
#endif
#define LDU(p)      ((vec)*(const uvec *)(p))
#define STU(p, v)   (*(uvec *)(p) = (uvec)(v))
#define VSPLAT(a)   ((vec){(a),(a),(a),(a),(a),(a),(a),(a)})
#define VPAIR(a,b)  ((vec){(a),(b),(a),(b),(a),(b),(a),(b)})
#define SWAP(v)     VSH((v),(v), 1,0,3,2,5,4,7,6)
#define VFMA(a,b,c)  ((vec)_mm512_fmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
#define VFNMA(a,b,c) ((vec)_mm512_fnmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))

/* 4x4 transpose of 128-bit complex granules (involution) -- L45_pfa's TRNC */
#define TRNC(r, c) do {                                                      \
    vec u0_ = VSH((r)[0], (r)[1], 0,1,8,9,4,5,12,13);                        \
    vec u1_ = VSH((r)[0], (r)[1], 2,3,10,11,6,7,14,15);                      \
    vec u2_ = VSH((r)[2], (r)[3], 0,1,8,9,4,5,12,13);                        \
    vec u3_ = VSH((r)[2], (r)[3], 2,3,10,11,6,7,14,15);                      \
    (c)[0] = VSH(u0_, u2_, 0,1,2,3,8,9,10,11);                               \
    (c)[2] = VSH(u0_, u2_, 4,5,6,7,12,13,14,15);                             \
    (c)[1] = VSH(u1_, u3_, 0,1,2,3,8,9,10,11);                               \
    (c)[3] = VSH(u1_, u3_, 4,5,6,7,12,13,14,15);                             \
} while (0)

/* ---- module constants --------------------------------------------------- */
/* 5-point (FFTW n1_5 FMA form, via L45_pfa / gen_pfa_large) */
#define K59  0.55901699437494742410229341718282   /* sqrt(5)/4               */
#define KIG  0.61803398874989484820458683436564   /* sin(4pi/5)/sin(2pi/5)   */
#define KS5  0.95105651629515357211665325776975   /* sin(2pi/5)              */
/* 3-point */
#define KS3  0.86602540378443864676372317075294   /* sin(2pi/3) = sqrt(3)/2  */

/* W25^j = cos(2 pi j / 25) - i sin(2 pi j / 25), j = n2*k1 for n2,k1 in
 * 1..4 -> j in {1,2,3,4,6,8,9,12,16}.  Literals from long-double cosl/sinl
 * (~19 correct digits).  Indexed by the compile-time product, so every
 * access folds to a constant after unrolling. */
static const double C25T[17] = {
    [1]  =  9.685831611286311195e-01, [2]  =  8.763066800438635873e-01,
    [3]  =  7.289686274214115231e-01, [4]  =  5.358267949789966183e-01,
    [6]  =  6.279051952931337601e-02, [8]  = -4.257792915650726488e-01,
    [9]  = -6.374239897486897102e-01, [12] = -9.921147013144778311e-01,
    [16] = -6.374239897486897102e-01,
};
static const double S25T[17] = {
    [1]  =  2.486898871648547882e-01, [2]  =  4.817536741017152750e-01,
    [3]  =  6.845471059286886738e-01, [4]  =  8.443279255020150785e-01,
    [6]  =  9.980267284282715619e-01, [8]  =  9.048270524660195277e-01,
    [9]  =  7.705132427757892308e-01, [12] =  1.253332335643042452e-01,
    [16] = -7.705132427757892307e-01,
};

/* W27^j, j = n2*k1 for n2 in 1..8, k1 in 1..2 -> j in {1..8,10,12,14,16} */
static const double C27T[17] = {
    [1]  =  9.730448705798238388e-01, [2]  =  8.936326403234122482e-01,
    [3]  =  7.660444431189780352e-01, [4]  =  5.971585917027861649e-01,
    [5]  =  3.960797660391568237e-01, [6]  =  1.736481776669303488e-01,
    [7]  = -5.814482891047582855e-02, [8]  = -2.868032327110902531e-01,
    [10] = -6.862416378687335857e-01, [12] = -9.396926207859083841e-01,
    [14] = -9.932383577419429885e-01, [16] = -8.354878114129364197e-01,
};
static const double S27T[17] = {
    [1]  =  2.306158707424401784e-01, [2]  =  4.487991802004621728e-01,
    [3]  =  6.427876096865393263e-01, [4]  =  8.021231927550437851e-01,
    [5]  =  9.182161068802740148e-01, [6]  =  9.848077530122080594e-01,
    [7]  =  9.983081582712682080e-01, [8]  =  9.579895123154888744e-01,
    [10] =  7.273736415730486960e-01, [12] =  3.420201433256687330e-01,
    [14] = -1.160929141252302297e-01, [16] = -5.495089780708060352e-01,
};

/* W9^j, j = b*q for b,q in 1..2 -> j in {1,2,4} */
static const double C9T[5] = {
    [1] =  7.660444431189780352e-01, [2] =  1.736481776669303488e-01,
    [4] = -9.396926207859083841e-01,
};
static const double S9T[5] = {
    [1] =  6.427876096865393263e-01, [2] =  9.848077530122080594e-01,
    [4] =  3.420201433256687330e-01,
};

/* v * (C - iS): 2 FMA-port ops + 1 swap */
#define CMULC(v, C, S) VFMA(SWAP(v), VPAIR((S), -(S)), (v) * VSPLAT(C))

/* One step of the graded chain map on a vector of 4 complex:
 *     z -> z / (1 + |z|)
 * |z| and the reciprocal via rsqrt14/rcp14 + TWO Newton steps each (final
 * relative error ~1e-16, comfortably inside the 1.5e-14/step contract),
 * instead of vsqrtpd+vdivpd whose zmm throughput is not pipelined.  The
 * max() guard keeps z = 0 exact (rsqrt(0) would make 0 * inf = NaN). */
static void map_scalar(const double *z, const double *c, double *o, size_t npts);

static inline __attribute__((always_inline))
vec map_step_v(vec z)
{
    vec q  = z * z;
    vec ms = q + SWAP(q);                        /* |z|^2 in both lanes */
    ms = (vec)_mm512_max_pd((__m512d)ms, (__m512d)VSPLAT(1e-300));
    vec y = (vec)_mm512_rsqrt14_pd((__m512d)ms);
    vec t_ = ms * y;
    y = (y * VSPLAT(0.5)) * VFNMA(t_, y, VSPLAT(3.0));
    t_ = ms * y;
    y = (y * VSPLAT(0.5)) * VFNMA(t_, y, VSPLAT(3.0));
    vec d = VFMA(ms, y, VSPLAT(1.0));            /* 1 + |z|             */
    vec r = (vec)_mm512_rcp14_pd((__m512d)d);
    r = r * VFNMA(d, r, VSPLAT(2.0));
    r = r * VFNMA(d, r, VSPLAT(2.0));
    return z * r;
}

/* One map step for TWO vectors at once (gen_r5, ADOPTED from gen_pfa_large
 * gen_r5's map_step_pair).  map_step_v runs the NR ladder on a vector where
 * each |z|^2 sits duplicated in both complex lanes -- half its ladder lanes
 * compute nothing new.  Here the 8 distinct |z|^2 of a vector PAIR pack
 * into one zmm (2 shuffles), ONE ladder runs for both vectors, and the
 * reciprocals unpack pair-duplicated (2 shuffles): ~21 arithmetic ops + 4
 * shuffles per pair vs 36 + 2 with two map_step_v calls.  BIT-IDENTICAL to
 * map_step_v per element: q_re + q_im = q_im + q_re exactly (IEEE addition
 * commutes), max/rsqrt14/rcp14/fma are elementwise, and the NR expressions
 * are verbatim -- so mixed use across chain families cannot break
 * repeatability. */
static inline __attribute__((always_inline))
void map_step_pair(vec v0, vec v1, vec *o0, vec *o1)
{
    vec q0 = v0 * v0, q1 = v1 * v1;
    vec ms = VSH(q0, q1, 0,2,4,6,8,10,12,14)
           + VSH(q0, q1, 1,3,5,7,9,11,13,15);        /* 8 distinct |z|^2 */
    ms = (vec)_mm512_max_pd((__m512d)ms, (__m512d)VSPLAT(1e-300));
    vec y = (vec)_mm512_rsqrt14_pd((__m512d)ms);
    vec t_ = ms * y;
    y = (y * VSPLAT(0.5)) * VFNMA(t_, y, VSPLAT(3.0));
    t_ = ms * y;
    y = (y * VSPLAT(0.5)) * VFNMA(t_, y, VSPLAT(3.0));
    vec d = VFMA(ms, y, VSPLAT(1.0));                /* 1 + |z|          */
    vec r = (vec)_mm512_rcp14_pd((__m512d)d);
    r = r * VFNMA(d, r, VSPLAT(2.0));
    r = r * VFNMA(d, r, VSPLAT(2.0));
    *o0 = v0 * VSH(r, r, 0,0,1,1,2,2,3,3);
    *o1 = v1 * VSH(r, r, 4,4,5,5,6,6,7,7);
}

/* sequential vectorized map over one contiguous span: o = (z+c)/(1+|z+c|).
 * 2-3 perfectly sequential streams (in-place z==o legal).  BORROWED from
 * gen_pfa_large's ip* chain family: at sizes where the volume does not stay
 * cache-resident, folding the map into the GENL-stream x-pass doubles the
 * miss-stream count; a separate sequential pass is cheaper.  The 25^3/27^3
 * volumes are not multiples of 8 doubles: exact scalar tail (1-3 complex).
 * gen_r5: pair-packed ladder (map_step_pair); odd vector counts (the L=50
 * plane prepass is 625 vectors) fall to one map_step_v. */
static void map_span(const double *z, const double *c, double *o, size_t nd)
{
    const size_t nv = nd / 8;
    size_t i = 0;
    for (; i + 2 <= nv; i += 2) {
        vec v0 = LDU(z + 8 * i)     + LDU(c + 8 * i);
        vec v1 = LDU(z + 8 * i + 8) + LDU(c + 8 * i + 8);
        vec o0, o1;
        map_step_pair(v0, v1, &o0, &o1);
        STU(o + 8 * i,     o0);
        STU(o + 8 * i + 8, o1);
    }
    if (i < nv) {
        vec v = LDU(z + 8 * i) + LDU(c + 8 * i);
        STU(o + 8 * i, map_step_v(v));
    }
    map_scalar(z + nv * 8, c + nv * 8, o + nv * 8, (nd - nv * 8) / 2);
}

/* map_span with the c stream fetched PREFETCHNTA (gen_r5, the ipq/iqn
 * families, ADOPTED from gen_pfa_large gen_r5): on SKX+ non-inclusive-LLC
 * parts, NTA lines fill L1 only and are dropped on eviction instead of
 * allocated into L2/LLC.  c has NO reuse within a step -- keeping it out of
 * L3 leaves the whole LLC to the state volume, which p2 and the next step's
 * prepass re-read.  Values identical to map_span (a hint moves no data). */
static void map_span_nta(const double *z, const double *c, double *o,
                         size_t nd)
{
    const size_t nv = nd / 8;
    size_t i = 0;
    for (; i + 2 <= nv; i += 2) {
        _mm_prefetch((const char *)(c + 8 * i + 512), _MM_HINT_NTA);
        _mm_prefetch((const char *)(c + 8 * i + 520), _MM_HINT_NTA);
        vec v0 = LDU(z + 8 * i)     + LDU(c + 8 * i);
        vec v1 = LDU(z + 8 * i + 8) + LDU(c + 8 * i + 8);
        vec o0, o1;
        map_step_pair(v0, v1, &o0, &o1);
        STU(o + 8 * i,     o0);
        STU(o + 8 * i + 8, o1);
    }
    if (i < nv) {
        vec v = LDU(z + 8 * i) + LDU(c + 8 * i);
        STU(o + 8 * i, map_step_v(v));
    }
    map_scalar(z + nv * 8, c + nv * 8, o + nv * 8, (nd - nv * 8) / 2);
}

/* map_span with the c lines CLFLUSHOPT-ed one pair after use (gen_r5, the
 * ipk family, ADOPTED from gen_pfa_large gen_r5): the guaranteed-semantics
 * variant of the same L3 bypass -- NTA fill policy is implementation-
 * defined, an explicit flush of a clean line is architectural and writes
 * nothing back.  Safe: nothing writes c, so a flush racing an in-flight
 * load re-reads identical bytes. */
static void map_span_cfl(const double *z, const double *c, double *o,
                         size_t nd)
{
    const size_t nv = nd / 8;
    size_t i = 0;
    for (; i + 2 <= nv; i += 2) {
        vec v0 = LDU(z + 8 * i)     + LDU(c + 8 * i);
        vec v1 = LDU(z + 8 * i + 8) + LDU(c + 8 * i + 8);
        vec o0, o1;
        map_step_pair(v0, v1, &o0, &o1);
        STU(o + 8 * i,     o0);
        STU(o + 8 * i + 8, o1);
#ifdef __CLFLUSHOPT__
        if (i >= 2) {
            _mm_clflushopt((void *)(c + 8 * (i - 2)));
            _mm_clflushopt((void *)(c + 8 * (i - 2) + 8));
        }
#endif
    }
    if (i < nv) {
        vec v = LDU(z + 8 * i) + LDU(c + 8 * i);
        STU(o + 8 * i, map_step_v(v));
    }
    map_scalar(z + nv * 8, c + nv * 8, o + nv * 8, (nd - nv * 8) / 2);
}

/* DFT5, FFTW n1_5 FMA form: 16 FMA-port ops + 2 swaps.  Outputs are
 * lvalues; temps block-scoped so the macro can be used repeatedly. */
#define DFT5M(x0,x1,x2,x3,x4, o0,o1,o2,o3,o4) do {                           \
    vec t1_ = (x1) + (x4), t4_ = (x1) - (x4);                                \
    vec t2_ = (x2) + (x3), t7_ = (x2) - (x3);                                \
    vec te_ = t1_ + t2_,   ta_ = t1_ - t2_;                                  \
    (o0) = (x0) + te_;                                                       \
    vec tm_ = VFNMA(te_, VSPLAT(0.25), (x0));                                \
    vec tp_ = VFMA (ta_, VSPLAT(K59), tm_);                                  \
    vec tq_ = VFNMA(ta_, VSPLAT(K59), tm_);                                  \
    vec tv_ = VFMA (t7_, VSPLAT(KIG), t4_);                                  \
    vec tw_ = VFNMA(t4_, VSPLAT(KIG), t7_);                                  \
    vec sv_ = SWAP(tv_), sw_ = SWAP(tw_);                                    \
    (o1) = VFMA (sv_, VPAIR(KS5, -KS5), tp_);                                \
    (o2) = VFNMA(sw_, VPAIR(KS5, -KS5), tq_);                                \
    (o3) = VFMA (sw_, VPAIR(KS5, -KS5), tq_);                                \
    (o4) = VFNMA(sv_, VPAIR(KS5, -KS5), tp_);                                \
} while (0)

/* DFT4: 8 FMA-port ops + 1 swap */
#define DFT4M(x0,x1,x2,x3, o0,o1,o2,o3) do {                                 \
    vec c0_ = (x0) + (x2), c1_ = (x0) - (x2);                                \
    vec c2_ = (x1) + (x3), c3_ = (x1) - (x3);                                \
    vec cm_ = SWAP(c3_);                                                     \
    (o0) = c0_ + c2_;                                                        \
    (o2) = c0_ - c2_;                                                        \
    (o1) = VFMA (cm_, VPAIR(1.0, -1.0), c1_);                                \
    (o3) = VFNMA(cm_, VPAIR(1.0, -1.0), c1_);                                \
} while (0)

/* DFT3: 6 FMA-port ops + 1 swap.
 *   X1 = (x0 - t/2) - i*KS3*(x1-x2),  X2 = conj-pair form with +i. */
#define DFT3M(x0,x1,x2, o0,o1,o2) do {                                       \
    vec ts_ = (x1) + (x2), ds_ = (x1) - (x2);                                \
    (o0) = (x0) + ts_;                                                       \
    vec tm3_ = VFNMA(ts_, VSPLAT(0.5), (x0));                                \
    vec sd3_ = SWAP(ds_);                                                    \
    (o1) = VFMA (sd3_, VPAIR(KS3, -KS3), tm3_);                              \
    (o2) = VFNMA(sd3_, VPAIR(KS3, -KS3), tm3_);                              \
} while (0)

/* DFT25 = 5x5 Cooley-Tukey with exact twiddles, stage-B outputs handed
 * STRAIGHT to the caller's store macro (gen_r2: ADOPTED from gen_pfa_large
 * gen_r2, replacing the dft25v function whose r[25] every PFA wrapper then
 * re-read to route through the CRT map -- a 25-store + 25-load L1 round-trip
 * per call).  LDX(n) yields input n (stride 1 in n), STO(k, v) consumes
 * natural-order output k; KMAP is applied by the caller inside STO.  Stage A
 * stores U_[5*k1 + n2] so stage B reads 5 contiguous hot slots.
 * 192 FMA-port ops + 36 swaps. */
#define DFT25M(LDX, STO, KMAP) do {                                          \
    vec U_[25];                                                              \
    _Pragma("GCC unroll 5")                                                  \
    for (int c5_ = 0; c5_ < 5; ++c5_) {                                      \
        vec y0_, y1_, y2_, y3_, y4_;                                         \
        DFT5M(LDX(c5_), LDX(c5_ + 5), LDX(c5_ + 10), LDX(c5_ + 15),          \
              LDX(c5_ + 20), y0_, y1_, y2_, y3_, y4_);                       \
        U_[c5_]      = y0_;                                                  \
        U_[5 + c5_]  = c5_ ? CMULC(y1_, C25T[c5_],     S25T[c5_])     : y1_; \
        U_[10 + c5_] = c5_ ? CMULC(y2_, C25T[2 * c5_], S25T[2 * c5_]) : y2_; \
        U_[15 + c5_] = c5_ ? CMULC(y3_, C25T[3 * c5_], S25T[3 * c5_]) : y3_; \
        U_[20 + c5_] = c5_ ? CMULC(y4_, C25T[4 * c5_], S25T[4 * c5_]) : y4_; \
    }                                                                        \
    _Pragma("GCC unroll 5")                                                  \
    for (int k1_ = 0; k1_ < 5; ++k1_) {                                      \
        vec r0_, r1_, r2_, r3_, r4_;                                         \
        DFT5M(U_[5 * k1_], U_[5 * k1_ + 1], U_[5 * k1_ + 2],                 \
              U_[5 * k1_ + 3], U_[5 * k1_ + 4],                              \
              r0_, r1_, r2_, r3_, r4_);                                      \
        STO(KMAP(k1_),      r0_);                                            \
        STO(KMAP(k1_ + 5),  r1_);                                            \
        STO(KMAP(k1_ + 10), r2_);                                            \
        STO(KMAP(k1_ + 15), r3_);                                            \
        STO(KMAP(k1_ + 20), r4_);                                            \
    }                                                                        \
} while (0)

/* stage-A input and CRT output-index helpers for the two DFT25M users;
 * expanded inside the PFA wrappers where T_ and k2_ are in scope (their
 * preprocessor note applies: the store macro must reach DFT25M as a
 * PARAMETER, not by textual name) */
#define LDT25(n)   T_[25 * k2_ + (n)]
#define K50MAP(k)  ((26 * (k) + 25 * k2_) % 50)
#define K100MAP(k) ((76 * (k) + 25 * k2_) % 100)

/* ---- per-size line codelets, LD/ST as macro parameters ------------------ */

/* L=25: direct DFT25 = 5x5 CT, stage-B outputs handed straight to ST
 * (no result-array round trip).  X[k1 + 5*k2]. */
#define DFT25C(LD, ST) do {                                                  \
    vec U_[25];                                                              \
    _Pragma("GCC unroll 5")                                                  \
    for (int c_ = 0; c_ < 5; ++c_) {                                         \
        vec y0_, y1_, y2_, y3_, y4_;                                         \
        DFT5M(LD(c_), LD(c_ + 5), LD(c_ + 10), LD(c_ + 15), LD(c_ + 20),     \
              y0_, y1_, y2_, y3_, y4_);                                      \
        U_[c_]      = y0_;                                                   \
        U_[5 + c_]  = c_ ? CMULC(y1_, C25T[c_],     S25T[c_])     : y1_;     \
        U_[10 + c_] = c_ ? CMULC(y2_, C25T[2 * c_], S25T[2 * c_]) : y2_;     \
        U_[15 + c_] = c_ ? CMULC(y3_, C25T[3 * c_], S25T[3 * c_]) : y3_;     \
        U_[20 + c_] = c_ ? CMULC(y4_, C25T[4 * c_], S25T[4 * c_]) : y4_;     \
    }                                                                        \
    _Pragma("GCC unroll 5")                                                  \
    for (int k1_ = 0; k1_ < 5; ++k1_) {                                      \
        vec o0_, o1_, o2_, o3_, o4_;                                         \
        DFT5M(U_[5 * k1_], U_[5 * k1_ + 1], U_[5 * k1_ + 2],                 \
              U_[5 * k1_ + 3], U_[5 * k1_ + 4],                              \
              o0_, o1_, o2_, o3_, o4_);                                      \
        ST(k1_,      o0_); ST(k1_ +  5, o1_); ST(k1_ + 10, o2_);             \
        ST(k1_ + 15, o3_); ST(k1_ + 20, o4_);                                \
    }                                                                        \
} while (0)

/* L=27: DFT27 = 3x9 Cooley-Tukey (NEW).  n = 9*n1 + n2, k = k1 + 3*k2:
 *   X[k1+3k2] = DFT9_{n2}( W27^{n2 k1} * DFT3_{n1}( x[9n1+n2] )[k1] )[k2]
 * Stage A: 9 x DFT3 + 16 nontrivial W27 twiddles into U[9*k1 + n2].
 * Stage B: 3 x DFT9, DFT9 itself 3x3 CT (n2 = 3a+b, k2 = q+3r) with 4
 * nontrivial W9 twiddles.  218 FMA-port ops + 55 swaps. */
#define DFT27C(LD, ST) do {                                                  \
    vec U_[27];                                                              \
    _Pragma("GCC unroll 9")                                                  \
    for (int c_ = 0; c_ < 9; ++c_) {                                         \
        vec y0_, y1_, y2_;                                                   \
        DFT3M(LD(c_), LD(c_ + 9), LD(c_ + 18), y0_, y1_, y2_);               \
        U_[c_]      = y0_;                                                   \
        U_[9 + c_]  = c_ ? CMULC(y1_, C27T[c_],     S27T[c_])     : y1_;     \
        U_[18 + c_] = c_ ? CMULC(y2_, C27T[2 * c_], S27T[2 * c_]) : y2_;     \
    }                                                                        \
    _Pragma("GCC unroll 3")                                                  \
    for (int k1_ = 0; k1_ < 3; ++k1_) {                                      \
        const vec *u_ = U_ + 9 * k1_;                                        \
        vec T9_[9];                                                          \
        _Pragma("GCC unroll 3")                                              \
        for (int b_ = 0; b_ < 3; ++b_) {                                     \
            vec t0_, t1_, t2_;                                               \
            DFT3M(u_[b_], u_[3 + b_], u_[6 + b_], t0_, t1_, t2_);            \
            T9_[b_]     = t0_;                                               \
            T9_[3 + b_] = b_ ? CMULC(t1_, C9T[b_],     S9T[b_])     : t1_;   \
            T9_[6 + b_] = b_ ? CMULC(t2_, C9T[2 * b_], S9T[2 * b_]) : t2_;   \
        }                                                                    \
        _Pragma("GCC unroll 3")                                              \
        for (int q_ = 0; q_ < 3; ++q_) {                                     \
            vec v0_, v1_, v2_;                                               \
            DFT3M(T9_[3 * q_], T9_[3 * q_ + 1], T9_[3 * q_ + 2],             \
                  v0_, v1_, v2_);                                            \
            ST(k1_ + 3 * q_,      v0_);                                      \
            ST(k1_ + 3 * q_ +  9, v1_);                                      \
            ST(k1_ + 3 * q_ + 18, v2_);                                      \
        }                                                                    \
    }                                                                        \
} while (0)

/* L=50 = 25x2 Good-Thomas.  Stage 1: 25 x DFT2 into T_[25*k2 + n1]; stage
 * 2: 2 x DFT25, stage-B outputs stored straight through ST via the CRT map
 * (gen_r2: ADOPTED from gen_pfa_large gen_r2 -- no R_[25] round-trip). */
#define PFA50C(LD, ST) do {                                                  \
    vec T_[50];                                                              \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 25; ++n1_) {                                     \
        vec a_ = LD((2 * n1_     ) % 50);                                    \
        vec b_ = LD((2 * n1_ + 25) % 50);                                    \
        T_[n1_]      = a_ + b_;                                              \
        T_[25 + n1_] = a_ - b_;                                              \
    }                                                                        \
    _Pragma("GCC unroll 2")                                                  \
    for (int k2_ = 0; k2_ < 2; ++k2_)                                        \
        DFT25M(LDT25, ST, K50MAP);                                           \
} while (0)

/* L=100 = 25x4 Good-Thomas.  Stage 1: 25 x DFT4 into T_[25*k2 + n1]; stage
 * 2: 4 x DFT25, stage-B outputs stored straight through ST (gen_r2). */
#define PFA100C(LD, ST) do {                                                 \
    vec T_[100];                                                             \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 25; ++n1_) {                                     \
        vec a0_ = LD((4 * n1_     ) % 100);                                  \
        vec a1_ = LD((4 * n1_ + 25) % 100);                                  \
        vec a2_ = LD((4 * n1_ + 50) % 100);                                  \
        vec a3_ = LD((4 * n1_ + 75) % 100);                                  \
        DFT4M(a0_, a1_, a2_, a3_,                                            \
              T_[n1_], T_[25 + n1_], T_[50 + n1_], T_[75 + n1_]);            \
    }                                                                        \
    _Pragma("GCC unroll 4")                                                  \
    for (int k2_ = 0; k2_ < 4; ++k2_)                                        \
        DFT25M(LDT25, ST, K100MAP);                                          \
} while (0)

/* ==== gen_r3: runtime exact tables + modules for the odd-p^k class ========
 * Tables are filled once at create() from long double cosl/sinl (~19 correct
 * digits, exact-to-0.5ulp doubles -- gen_pfa_small r3's precedent for the
 * brief's exactness rule).  Twiddle tables are stored as C[j] plus the pair
 * {sin, -sin} so W^j = C - iS costs VSPLAT + ONE 128-bit broadcast. */
static double C7F[7],     S7F[7];             /* module constants, DFT7K    */
static double C11F[11],   S11F[11];           /* module constants, DFT11K   */
static double C49F[49],   S49P[49][2];        /* W49 twiddles               */
static double C81F[81],   S81P[81][2];        /* W81 twiddles               */
static double C121F[121], S121P[121][2];      /* W121 twiddles              */
static double C125F[125], S125P[125][2];      /* W125 twiddles              */

static void powp_rt_tabs(void)
{
    static int done;
    if (done) return;
    const long double TP = 6.283185307179586476925286766559L;   /* 2 pi */
    for (int j = 0; j < 7; ++j) {
        C7F[j]  = (double)cosl(TP * j / 7);
        S7F[j]  = (double)sinl(TP * j / 7);
    }
    for (int j = 0; j < 11; ++j) {
        C11F[j] = (double)cosl(TP * j / 11);
        S11F[j] = (double)sinl(TP * j / 11);
    }
#define PWPTAB(N, C, SP) do {                                                \
    for (int j_ = 0; j_ < (N); ++j_) {                                       \
        C[j_] = (double)cosl(TP * j_ / (N));                                 \
        double s_ = (double)sinl(TP * j_ / (N));                             \
        SP[j_][0] = s_; SP[j_][1] = -s_;                                     \
    }                                                                        \
} while (0)
    PWPTAB(49,  C49F,  S49P);
    PWPTAB(81,  C81F,  S81P);
    PWPTAB(121, C121F, S121P);
    PWPTAB(125, C125F, S125P);
#undef PWPTAB
    done = 1;
}

#ifdef __AVX512F__
/* broadcast a 16 B {s,-s} pair to all four complex lanes (AVX512DQ) */
#define VPAIRP(p)  ((vec)_mm512_broadcast_f64x2(_mm_loadu_pd(p)))
/* v * (C - iS) with C/S from runtime tables: 2 FMA-port ops + 1 swap + the
 * two broadcasts (VSPLAT of a scalar is a vbroadcastsd memory operand) */
#define CMULCT(v, CT, SP, j) \
    VFMA(SWAP(v), VPAIRP((SP)[(j)]), (v) * VSPLAT((CT)[(j)]))

#define KID(k) (k)                            /* identity output map        */

/* DFT7, conjugate-pair fold over runtime C7F/S7F: X_k = A_k - i B_k with
 * A_k = x0 + sum_j cos(2pi jk/7) t_j, B_k = sum_j sin(2pi jk/7) u_j,
 * t_j = x_j + x_{7-j}, u_j = x_j - x_{7-j}; X_{7-k} is the conjugate-pair
 * partner.  Full tables indexed by (j k) mod 7 carry the sine signs.
 * 33 FMA-port ops + 3 swaps.  Outputs routed STO(KM(k), simple-var). */
#define DFT7K(LDX, STO, KM) do {                                             \
    vec x70_ = LDX(0);                                                       \
    vec t71_ = LDX(1) + LDX(6), u71_ = LDX(1) - LDX(6);                      \
    vec t72_ = LDX(2) + LDX(5), u72_ = LDX(2) - LDX(5);                      \
    vec t73_ = LDX(3) + LDX(4), u73_ = LDX(3) - LDX(4);                      \
    vec y70_ = x70_ + t71_ + t72_ + t73_;                                    \
    STO(KM(0), y70_);                                                        \
    _Pragma("GCC unroll 3")                                                  \
    for (int k7_ = 1; k7_ <= 3; ++k7_) {                                     \
        vec A7_ = VFMA(t71_, VSPLAT(C7F[k7_]),                               \
                  VFMA(t72_, VSPLAT(C7F[(2 * k7_) % 7]),                     \
                  VFMA(t73_, VSPLAT(C7F[(3 * k7_) % 7]), x70_)));            \
        vec B7_ = VFMA(u71_, VSPLAT(S7F[k7_]),                               \
                  VFMA(u72_, VSPLAT(S7F[(2 * k7_) % 7]),                     \
                       u73_ * VSPLAT(S7F[(3 * k7_) % 7])));                  \
        vec sb7_ = SWAP(B7_);                                                \
        vec o7a_ = VFMA (sb7_, VPAIR(1.0, -1.0), A7_);                       \
        vec o7b_ = VFNMA(sb7_, VPAIR(1.0, -1.0), A7_);                       \
        STO(KM(k7_),     o7a_);                                              \
        STO(KM(7 - k7_), o7b_);                                              \
    }                                                                        \
} while (0)

/* DFT11, same fold shape, h = 5: 75 FMA-port ops + 5 swaps */
#define DFT11K(LDX, STO, KM) do {                                            \
    vec xb0_ = LDX(0);                                                       \
    vec tb1_ = LDX(1) + LDX(10), ub1_ = LDX(1) - LDX(10);                    \
    vec tb2_ = LDX(2) + LDX(9),  ub2_ = LDX(2) - LDX(9);                     \
    vec tb3_ = LDX(3) + LDX(8),  ub3_ = LDX(3) - LDX(8);                     \
    vec tb4_ = LDX(4) + LDX(7),  ub4_ = LDX(4) - LDX(7);                     \
    vec tb5_ = LDX(5) + LDX(6),  ub5_ = LDX(5) - LDX(6);                     \
    vec yb0_ = xb0_ + tb1_ + tb2_ + tb3_ + tb4_ + tb5_;                      \
    STO(KM(0), yb0_);                                                        \
    _Pragma("GCC unroll 5")                                                  \
    for (int kb_ = 1; kb_ <= 5; ++kb_) {                                     \
        vec Ab_ = VFMA(tb1_, VSPLAT(C11F[kb_]),                              \
                  VFMA(tb2_, VSPLAT(C11F[(2 * kb_) % 11]),                   \
                  VFMA(tb3_, VSPLAT(C11F[(3 * kb_) % 11]),                   \
                  VFMA(tb4_, VSPLAT(C11F[(4 * kb_) % 11]),                   \
                  VFMA(tb5_, VSPLAT(C11F[(5 * kb_) % 11]), xb0_)))));        \
        vec Bb_ = VFMA(ub1_, VSPLAT(S11F[kb_]),                              \
                  VFMA(ub2_, VSPLAT(S11F[(2 * kb_) % 11]),                   \
                  VFMA(ub3_, VSPLAT(S11F[(3 * kb_) % 11]),                   \
                  VFMA(ub4_, VSPLAT(S11F[(4 * kb_) % 11]),                   \
                       ub5_ * VSPLAT(S11F[(5 * kb_) % 11])))));              \
        vec sbb_ = SWAP(Bb_);                                                \
        vec oba_ = VFMA (sbb_, VPAIR(1.0, -1.0), Ab_);                       \
        vec obb_ = VFNMA(sbb_, VPAIR(1.0, -1.0), Ab_);                       \
        STO(KM(kb_),      oba_);                                             \
        STO(KM(11 - kb_), obb_);                                             \
    }                                                                        \
} while (0)

/* DFT9 = 3x3 CT (n = 3 n1 + n2, k = k1 + 3 k2), the compile-time W9 tables
 * from the L=27 codelet: stage a DFT3 over n1 per n2 + W9^{n2 k1} into
 * V9_[3 k1 + n2], stage b DFT3 over n2 per k1.  32 FMA-port ops + 8 swaps. */
#define DFT9K(LDX, STO, KM) do {                                             \
    vec V9_[9];                                                              \
    _Pragma("GCC unroll 3")                                                  \
    for (int n9_ = 0; n9_ < 3; ++n9_) {                                      \
        vec y90_, y91_, y92_;                                                \
        DFT3M(LDX(n9_), LDX(n9_ + 3), LDX(n9_ + 6), y90_, y91_, y92_);       \
        V9_[n9_]     = y90_;                                                 \
        V9_[3 + n9_] = n9_ ? CMULC(y91_, C9T[n9_],     S9T[n9_])     : y91_; \
        V9_[6 + n9_] = n9_ ? CMULC(y92_, C9T[2 * n9_], S9T[2 * n9_]) : y92_; \
    }                                                                        \
    _Pragma("GCC unroll 3")                                                  \
    for (int k9_ = 0; k9_ < 3; ++k9_) {                                      \
        vec z90_, z91_, z92_;                                                \
        DFT3M(V9_[3 * k9_], V9_[3 * k9_ + 1], V9_[3 * k9_ + 2],              \
              z90_, z91_, z92_);                                             \
        STO(KM(k9_),     z90_);                                              \
        STO(KM(k9_ + 3), z91_);                                              \
        STO(KM(k9_ + 6), z92_);                                              \
    }                                                                        \
} while (0)

/* stage helpers for the composite codelets (reference locals X_/U_ and the
 * loop vars n2_/k1_/q5_ in scope at the expansion site, like LDT25) */
#define LA49(j)     X_[7 * (j) + n2_]
#define SA49(k, v)  (U_[7 * (k) + n2_] = ((k) && n2_)                        \
                     ? CMULCT((v), C49F, S49P, ((k) * n2_) % 49) : (v))
#define LB49(j)     U_[7 * k1_ + (j)]
#define K49B(k)     (k1_ + 7 * (k))
#define LA81(j)     X_[9 * (j) + n2_]
#define SA81(k, v)  (U_[9 * (k) + n2_] = ((k) && n2_)                        \
                     ? CMULCT((v), C81F, S81P, (k) * n2_) : (v))
#define LB81(j)     U_[9 * k1_ + (j)]
#define K81B(k)     (k1_ + 9 * (k))
#define LA121(j)    X_[11 * (j) + n2_]
#define SA121(k, v) (U_[11 * (k) + n2_] = ((k) && n2_)                       \
                     ? CMULCT((v), C121F, S121P, (k) * n2_) : (v))
#define LB121(j)    U_[11 * k1_ + (j)]
#define K121B(k)    (k1_ + 11 * (k))
#define L125B(n)    T_[25 * q5_ + (n)]     /* T_, NOT U_: DFT25M's internal
                                              U_[25] would shadow it (the
                                              PFA50C/T_ rule)              */
#define K125B(k)    (q5_ + 5 * (k))

/* L=49: 7x7 CT, X[k1 + 7 k2], 36 nontrivial W49 twiddles.  534 FMA-port. */
#define DFT49C(LD, ST) do {                                                  \
    vec X_[49], U_[49];                                                      \
    for (int n_ = 0; n_ < 49; ++n_) X_[n_] = LD(n_);                         \
    _Pragma("GCC unroll 7")                                                  \
    for (int n2_ = 0; n2_ < 7; ++n2_) DFT7K(LA49, SA49, KID);                \
    _Pragma("GCC unroll 7")                                                  \
    for (int k1_ = 0; k1_ < 7; ++k1_) DFT7K(LB49, ST, K49B);                 \
} while (0)

/* L=81: 9x9 CT, 64 nontrivial W81 twiddles.  ~850 FMA-port. */
#define DFT81C(LD, ST) do {                                                  \
    vec X_[81], U_[81];                                                      \
    for (int n_ = 0; n_ < 81; ++n_) X_[n_] = LD(n_);                         \
    _Pragma("GCC unroll 9")                                                  \
    for (int n2_ = 0; n2_ < 9; ++n2_) DFT9K(LA81, SA81, KID);                \
    _Pragma("GCC unroll 9")                                                  \
    for (int k1_ = 0; k1_ < 9; ++k1_) DFT9K(LB81, ST, K81B);                 \
} while (0)

/* L=121: 11x11 CT, 100 nontrivial W121 twiddles.  ~1850 FMA-port. */
#define DFT121C(LD, ST) do {                                                 \
    vec X_[121], U_[121];                                                    \
    for (int n_ = 0; n_ < 121; ++n_) X_[n_] = LD(n_);                        \
    _Pragma("GCC unroll 11")                                                 \
    for (int n2_ = 0; n2_ < 11; ++n2_) DFT11K(LA121, SA121, KID);            \
    _Pragma("GCC unroll 11")                                                 \
    for (int k1_ = 0; k1_ < 11; ++k1_) DFT11K(LB121, ST, K121B);             \
} while (0)

/* L=125: 5x25 CT (n = 25 n1 + n2, k = k1 + 5 k2): stage A 25 x DFT5 + 96
 * W125 twiddles into U_[25 k1 + n2], stage B 5 x DFT25M straight through
 * ST via k1 + 5 k2.  1552 FMA-port. */
#define DFT125C(LD, ST) do {                                                 \
    vec X_[125], T_[125];                                                    \
    for (int n_ = 0; n_ < 125; ++n_) X_[n_] = LD(n_);                        \
    _Pragma("GCC unroll 25")                                                 \
    for (int n2_ = 0; n2_ < 25; ++n2_) {                                     \
        vec y0_, y1_, y2_, y3_, y4_;                                         \
        DFT5M(X_[n2_], X_[25 + n2_], X_[50 + n2_], X_[75 + n2_],             \
              X_[100 + n2_], y0_, y1_, y2_, y3_, y4_);                       \
        T_[n2_]       = y0_;                                                 \
        T_[25 + n2_]  = n2_ ? CMULCT(y1_, C125F, S125P, n2_)     : y1_;      \
        T_[50 + n2_]  = n2_ ? CMULCT(y2_, C125F, S125P, 2 * n2_) : y2_;      \
        T_[75 + n2_]  = n2_ ? CMULCT(y3_, C125F, S125P, 3 * n2_) : y3_;      \
        T_[100 + n2_] = n2_ ? CMULCT(y4_, C125F, S125P, 4 * n2_) : y4_;      \
    }                                                                        \
    _Pragma("GCC unroll 5")                                                  \
    for (int q5_ = 0; q5_ < 5; ++q5_)                                        \
        DFT25M(L125B, ST, K125B);                                            \
} while (0)
#endif /* __AVX512F__ */

#define GCAT_(a,b) a##b
#define GCAT(a,b)  GCAT_(a,b)

/* instantiate the engine for the four sizes */
#define GENL 25
#define GPP  28                       /* pl row pitch: 448 B = 7 lines, odd  */
#define PFAL DFT25C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 27
#define GPP  28                       /* 448 B = 7 lines, odd                */
#define PFAL DFT27C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 50
#define GPP  52                       /* 832 B = 13 lines, odd               */
#define PFAL PFA50C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 100
#define GPP  108                      /* 1728 B = 27 lines, odd              */
#define PFAL PFA100C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

/* gen_r3 odd-p^k class sizes: REDUCED template (GENLITE -> ip0/ipf/f0/ipm0
 * candidates only; halves the compile of these unscored round-6 sizes) */
#define GENLITE 1

#define GENL 49
#define GPP  52                       /* 832 B = 13 lines, odd               */
#define PFAL DFT49C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 81
#define GPP  84                       /* 1344 B = 21 lines, odd              */
#define PFAL DFT81C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 121
#define GPP  124                      /* 1984 B = 31 lines, odd              */
#define PFAL DFT121C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 125
#define GPP  132                      /* 2112 B = 33 lines, odd              */
#define PFAL DFT125C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#undef GENLITE

/* ================= SoA-8 batch-lane chain engine, L = 25 / 27 =============
 * (gen_r2) 8 volumes in the zmm lanes, split-complex 128 B sites
 * (re[8] | im[8]), zero shuffles inside the transform -- the structure of
 * gen_pfa_small gen_r2 / gen_batchlane gen_r1 (ice bl8 lineage), extended to
 * prime powers.  The prime-power novelty: general CT needs a digit
 * permutation, so pencils are the classic IN-PLACE forms -- DIF (natural in,
 * digit-reversed out) and DIT (digit-reversed in, natural out) -- whose
 * stages read and write the SAME slot sets (no whole-pencil buffering).  The
 * chain alternates DIF/DIT steps; c is packed once per chain in both site
 * layouts.  Twiddles are the same exact product-indexed tables as the
 * interleaved engine, applied at the stores as 4 broadcast FMAs (re' =
 * C*r + S*i, im' = C*i - S*r for W^j = C - iS).  Slot algebra and twiddle
 * placement verified against numpy (both sizes, both directions). */

/* site slot k of a pencil at base p, stride st DOUBLES between sites */
#define BR(k) (*(vec *)((p) + (size_t)(k) * (st)))
#define BI(k) (*(vec *)((p) + (size_t)(k) * (st) + 8))

/* store slot k, twiddled by CT_/ST_[j] (j == 0 compile-time: plain store) */
#define SSTW(k, j, CT_, ST_, rr, ii) do {                                    \
    vec rt_ = (rr), it_ = (ii);                                              \
    if ((j) != 0) {                                                          \
        BR(k) = VFMA (VSPLAT(ST_[j]), it_, rt_ * VSPLAT(CT_[j]));            \
        BI(k) = VFNMA(VSPLAT(ST_[j]), rt_, it_ * VSPLAT(CT_[j]));            \
    } else { BR(k) = rt_; BI(k) = it_; }                                     \
} while (0)

/* split-lane map step: state (zr,zi) <- map(z + c), c site at cp (re at +0,
 * im at +8, same stride as the pencil).  Same rsqrt14/rcp14 + 2-Newton
 * ladder as the interleaved engine (additive 1e-300 guard as in
 * gen_pfa_small/gen_batchlane -- folds into the |z|^2 FMA chain). */
static inline __attribute__((always_inline))
void map8s(vec *zr, vec *zi, const double *cp)
{
    vec a  = *zr + LDU(cp);
    vec b  = *zi + LDU(cp + 8);
    vec ms = VFMA(a, a, VFMA(b, b, VSPLAT(1e-300)));
    vec y  = (vec)_mm512_rsqrt14_pd((__m512d)ms);
    vec t_ = ms * y;
    y = (y * VSPLAT(0.5)) * VFNMA(t_, y, VSPLAT(3.0));
    t_ = ms * y;
    y = (y * VSPLAT(0.5)) * VFNMA(t_, y, VSPLAT(3.0));
    vec d = VFMA(ms, y, VSPLAT(1.0));
#ifdef GENPWP_MAPRCP
    /* the interleaved engine's rcp14 + 2-Newton reciprocal, kept as the A/B
     * control.  Node verdict (this round, control-first pairs): the single
     * exact vdivpd below wins at 25 (34.2 vs 34.9-35.7, and 32.0 vs 35.6)
     * and ties at 27 -- gen_pfa_small's "divider is idle in this pass"
     * argument transfers to this engine; gen_batchlane's opposite result
     * does not. */
    vec r = (vec)_mm512_rcp14_pd((__m512d)d);
    r = r * VFNMA(d, r, VSPLAT(2.0));
    r = r * VFNMA(d, r, VSPLAT(2.0));
#else
    /* ADOPTED from gen_pfa_small: ONE exact vdivpd on the otherwise-idle
     * divider unit (also exact-tier: better rounding than the ladder) */
    vec r = (vec)_mm512_div_pd((__m512d)VSPLAT(1.0), (__m512d)d);
#endif
    *zr = a * r;
    *zi = b * r;
}

/* TWO split-lane map steps sharing ONE vdivpd (gen_r5): the reciprocal-
 * product trick from gen_layout gen_r5's gl_map16 (built there from
 * gen_dense_prime r4's item 3), adapted to the split-complex site.  Both
 * NR rsqrt ladders run (their lanes are all distinct -- nothing to pack in
 * split complex); only the divide is shared: q = 1/(d0 d1), r0 = q d1,
 * r1 = q d0.  Costs 3 extra vmulpd per pair for one saved vdivpd (~13-16
 * occupied divider cycles on ICL) and adds ~1-2 ulp to the reciprocal --
 * the per-step budget is 1.5e-14 (~60 ulp) and the m=2 soa gate checks the
 * composition at 1e-13.  |z| stays O(1) in the graded chain (the map is a
 * contraction into the unit disc), so d0*d1 cannot overflow. */
static inline __attribute__((always_inline))
void map8s_pair(vec *zr0, vec *zi0, const double *cp0,
                vec *zr1, vec *zi1, const double *cp1)
{
    vec a0 = *zr0 + LDU(cp0),  b0 = *zi0 + LDU(cp0 + 8);
    vec a1 = *zr1 + LDU(cp1),  b1 = *zi1 + LDU(cp1 + 8);
    vec m0 = VFMA(a0, a0, VFMA(b0, b0, VSPLAT(1e-300)));
    vec m1 = VFMA(a1, a1, VFMA(b1, b1, VSPLAT(1e-300)));
    vec y0 = (vec)_mm512_rsqrt14_pd((__m512d)m0);
    vec y1 = (vec)_mm512_rsqrt14_pd((__m512d)m1);
    vec t0 = m0 * y0, t1 = m1 * y1;
    y0 = (y0 * VSPLAT(0.5)) * VFNMA(t0, y0, VSPLAT(3.0));
    y1 = (y1 * VSPLAT(0.5)) * VFNMA(t1, y1, VSPLAT(3.0));
    t0 = m0 * y0; t1 = m1 * y1;
    y0 = (y0 * VSPLAT(0.5)) * VFNMA(t0, y0, VSPLAT(3.0));
    y1 = (y1 * VSPLAT(0.5)) * VFNMA(t1, y1, VSPLAT(3.0));
    vec d0 = VFMA(m0, y0, VSPLAT(1.0));
    vec d1 = VFMA(m1, y1, VSPLAT(1.0));
    vec q  = (vec)_mm512_div_pd((__m512d)VSPLAT(1.0), (__m512d)(d0 * d1));
    vec r0 = q * d1, r1 = q * d0;
    *zr0 = a0 * r0; *zi0 = b0 * r0;
    *zr1 = a1 * r1; *zi1 = b1 * r1;
}

/* map store for the final stage of the x-pass (j/CT_/ST_ ignored: the final
 * stage is twiddle-free); cp/st are in scope in the *_m pencils */
#define SSTM(k, j, CT_, ST_, rr, ii) do {                                    \
    vec zr_ = (rr), zi_ = (ii);                                              \
    map8s(&zr_, &zi_, cp + (size_t)(k) * (st));                              \
    BR(k) = zr_; BI(k) = zi_;                                                \
} while (0)

/* DFT5 split-complex core, Winograd 4-constant form (34 FMA-port ops for
 * both components; from gen_pfa_small gen_r1/r2, verbatim algebra).
 * Declares X0*, A1*, A2*, v1*, v2*; outputs in spectral order are
 * X0, A1+iv1x, A2+iv2x, A2-iv2x, A1-iv1x with the i-cross as re+=v?i,
 * im-=v?r on the + side. */
#define KS52 0.58778525229247312917   /* sin(4pi/5)                          */
#define D5SC(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i)                        \
    vec tar_ = (x1r) + (x4r), tai_ = (x1i) + (x4i);                          \
    vec tbr_ = (x2r) + (x3r), tbi_ = (x2i) + (x3i);                          \
    vec sar_ = (x1r) - (x4r), sai_ = (x1i) - (x4i);                          \
    vec sbr_ = (x2r) - (x3r), sbi_ = (x2i) - (x3i);                          \
    vec pr_  = tar_ + tbr_,   pi_  = tai_ + tbi_;                            \
    vec qr_  = tar_ - tbr_,   qi_  = tai_ - tbi_;                            \
    vec X0r  = (x0r) + pr_,   X0i  = (x0i) + pi_;                            \
    vec fr_  = (x0r) - 0.25 * pr_, fi_ = (x0i) - 0.25 * pi_;                 \
    vec A1r  = fr_ + K59 * qr_,  A1i = fi_ + K59 * qi_;                      \
    vec A2r  = fr_ - K59 * qr_,  A2i = fi_ - K59 * qi_;                      \
    vec v1r  = KS5 * sar_ + KS52 * sbr_, v1i = KS5 * sai_ + KS52 * sbi_;     \
    vec v2r  = KS52 * sar_ - KS5 * sbr_, v2i = KS52 * sai_ - KS5 * sbi_;

/* one in-place DFT5 stage over slots {BASE + STEP*i}: output k gets twiddle
 * exponent (TJ)*k in the W25 tables through the MST store macro */
#define D5STAGE(BASE, STEP, TJ, MST) do {                                    \
    const int b5_ = (BASE), s5_ = (STEP);                                    \
    vec x0r_ = BR(b5_),           x0i_ = BI(b5_);                            \
    vec x1r_ = BR(b5_ + s5_),     x1i_ = BI(b5_ + s5_);                      \
    vec x2r_ = BR(b5_ + 2 * s5_), x2i_ = BI(b5_ + 2 * s5_);                  \
    vec x3r_ = BR(b5_ + 3 * s5_), x3i_ = BI(b5_ + 3 * s5_);                  \
    vec x4r_ = BR(b5_ + 4 * s5_), x4i_ = BI(b5_ + 4 * s5_);                  \
    D5SC(x0r_, x0i_, x1r_, x1i_, x2r_, x2i_, x3r_, x3i_, x4r_, x4i_)         \
    MST(b5_,           0,        C25T, S25T, X0r, X0i);                      \
    MST(b5_ +     s5_, (TJ),     C25T, S25T, A1r + v1i, A1i - v1r);          \
    MST(b5_ + 2 * s5_, 2 * (TJ), C25T, S25T, A2r + v2i, A2i - v2r);          \
    MST(b5_ + 3 * s5_, 3 * (TJ), C25T, S25T, A2r - v2i, A2i + v2r);          \
    MST(b5_ + 4 * s5_, 4 * (TJ), C25T, S25T, A1r - v1i, A1i + v1r);          \
} while (0)

/* the map-fused FINAL DFT5 stage with paired divides (gen_r5): same loads,
 * same D5SC core, same store slots and per-element map values as
 * D5STAGE(..., SSTM) except outputs 1+2 and 3+4 each share one vdivpd via
 * map8s_pair -- 3 divides per 5 sites instead of 5. */
#define D5STAGEMP(BASE, STEP) do {                                           \
    const int b5_ = (BASE), s5_ = (STEP);                                    \
    vec x0r_ = BR(b5_),           x0i_ = BI(b5_);                            \
    vec x1r_ = BR(b5_ + s5_),     x1i_ = BI(b5_ + s5_);                      \
    vec x2r_ = BR(b5_ + 2 * s5_), x2i_ = BI(b5_ + 2 * s5_);                  \
    vec x3r_ = BR(b5_ + 3 * s5_), x3i_ = BI(b5_ + 3 * s5_);                  \
    vec x4r_ = BR(b5_ + 4 * s5_), x4i_ = BI(b5_ + 4 * s5_);                  \
    D5SC(x0r_, x0i_, x1r_, x1i_, x2r_, x2i_, x3r_, x3i_, x4r_, x4i_)         \
    vec z0r_ = X0r,       z0i_ = X0i;                                        \
    map8s(&z0r_, &z0i_, cp + (size_t)b5_ * st);                              \
    BR(b5_) = z0r_; BI(b5_) = z0i_;                                          \
    vec z1r_ = A1r + v1i, z1i_ = A1i - v1r;                                  \
    vec z2r_ = A2r + v2i, z2i_ = A2i - v2r;                                  \
    map8s_pair(&z1r_, &z1i_, cp + (size_t)(b5_ +     s5_) * st,              \
               &z2r_, &z2i_, cp + (size_t)(b5_ + 2 * s5_) * st);             \
    BR(b5_ +     s5_) = z1r_; BI(b5_ +     s5_) = z1i_;                      \
    BR(b5_ + 2 * s5_) = z2r_; BI(b5_ + 2 * s5_) = z2i_;                      \
    vec z3r_ = A2r - v2i, z3i_ = A2i + v2r;                                  \
    vec z4r_ = A1r - v1i, z4i_ = A1i + v1r;                                  \
    map8s_pair(&z3r_, &z3i_, cp + (size_t)(b5_ + 3 * s5_) * st,              \
               &z4r_, &z4i_, cp + (size_t)(b5_ + 4 * s5_) * st);             \
    BR(b5_ + 3 * s5_) = z3r_; BI(b5_ + 3 * s5_) = z3i_;                      \
    BR(b5_ + 4 * s5_) = z4r_; BI(b5_ + 4 * s5_) = z4i_;                      \
} while (0)

/* DFT3 split-complex core (12 FMA-port ops; gen_pfa_small's D3S algebra) */
#define D3SC(x0r,x0i,x1r,x1i,x2r,x2i)                                        \
    vec t3r_ = (x1r) + (x2r), t3i_ = (x1i) + (x2i);                          \
    vec u3r_ = (x1r) - (x2r), u3i_ = (x1i) - (x2i);                          \
    vec h3r_ = (x0r) - 0.5 * t3r_, h3i_ = (x0i) - 0.5 * t3i_;                \
    vec Y0r  = (x0r) + t3r_, Y0i = (x0i) + t3i_;                             \
    vec Y1r  = h3r_ + KS3 * u3i_, Y1i = h3i_ - KS3 * u3r_;                   \
    vec Y2r  = h3r_ - KS3 * u3i_, Y2i = h3i_ + KS3 * u3r_;

/* one in-place DFT3 stage over slots {BASE + STEP*i}: outputs 0/1/2 get
 * twiddle exponents J0/J1/J2 in the CT_/ST_ tables through MST.  J0 is
 * nonzero in exactly one place -- DIT27 stage 2, whose twiddle
 * W27^{n1 (k1 + 3 k2)} carries the constant offset n1*k1 at k2 = 0. */
#define D3STAGE(BASE, STEP, J0, J1, J2, CT_, ST_, MST) do {                  \
    const int b3_ = (BASE), s3_ = (STEP);                                    \
    vec x0r_ = BR(b3_),           x0i_ = BI(b3_);                            \
    vec x1r_ = BR(b3_ + s3_),     x1i_ = BI(b3_ + s3_);                      \
    vec x2r_ = BR(b3_ + 2 * s3_), x2i_ = BI(b3_ + 2 * s3_);                  \
    D3SC(x0r_, x0i_, x1r_, x1i_, x2r_, x2i_)                                 \
    MST(b3_,           (J0), CT_, ST_, Y0r, Y0i);                            \
    MST(b3_ +     s3_, (J1), CT_, ST_, Y1r, Y1i);                            \
    MST(b3_ + 2 * s3_, (J2), CT_, ST_, Y2r, Y2i);                            \
} while (0)

/* the map-fused FINAL DFT3 stage with a paired divide (gen_r5): same loads,
 * same D3SC core, same store slots and per-element map values as
 * D3STAGE(..., SSTM) except outputs 1+2 share one vdivpd via map8s_pair --
 * 2 divides per 3 sites instead of 3. */
#define D3STAGEMP(BASE, STEP) do {                                           \
    const int b3_ = (BASE), s3_ = (STEP);                                    \
    vec x0r_ = BR(b3_),           x0i_ = BI(b3_);                            \
    vec x1r_ = BR(b3_ + s3_),     x1i_ = BI(b3_ + s3_);                      \
    vec x2r_ = BR(b3_ + 2 * s3_), x2i_ = BI(b3_ + 2 * s3_);                  \
    D3SC(x0r_, x0i_, x1r_, x1i_, x2r_, x2i_)                                 \
    vec z0r_ = Y0r, z0i_ = Y0i;                                              \
    map8s(&z0r_, &z0i_, cp + (size_t)b3_ * st);                              \
    BR(b3_) = z0r_; BI(b3_) = z0i_;                                          \
    vec z1r_ = Y1r, z1i_ = Y1i;                                              \
    vec z2r_ = Y2r, z2i_ = Y2i;                                              \
    map8s_pair(&z1r_, &z1i_, cp + (size_t)(b3_ +     s3_) * st,              \
               &z2r_, &z2i_, cp + (size_t)(b3_ + 2 * s3_) * st);             \
    BR(b3_ +     s3_) = z1r_; BI(b3_ +     s3_) = z1i_;                      \
    BR(b3_ + 2 * s3_) = z2r_; BI(b3_ + 2 * s3_) = z2i_;                      \
} while (0)

/* ---- L=25 pencils: n = n1 + 5 n2, k = 5 k1 + k2 (DIF) / k1 + 5 k2 (DIT).
 * DIF, natural in -> digit-swapped out (X[k] lands at slot (k%5)*5 + k/5):
 *   stage 1, group n1: slots {n1 + 5i}, DFT5 over n2 -> k2, tw W25^{n1 k2}
 *   stage 2, group k2: slots {5k2 + i}, DFT5 over n1 -> k1, plain (or map)
 * DIT, digit-swapped in -> natural out:
 *   stage 1, group n1: slots {5n1 + i}, DFT5 over n2 -> k1, tw W25^{n1 k1}
 *   stage 2, group k1: slots {k1 + 5i}, DFT5 over n1 -> k2, plain (or map) */
/* final-stage forms for the two DFT5 pencil flavours: plain (z/y passes)
 * and map-fused (x pass).  gen_r5: the map-fused form defaults to the
 * paired-divide D5STAGEMP; -DGENPWP_NOMAPPAIR restores the r2-r4
 * one-vdivpd-per-site SSTM form for A/B and cross-arch races. */
#define D5FS_PLAIN(B, S) D5STAGE(B, S, 0, SSTW)
#ifdef GENPWP_NOMAPPAIR
#define D5FS_MAP(B, S)   D5STAGE(B, S, 0, SSTM)
#else
#define D5FS_MAP(B, S)   D5STAGEMP(B, S)
#endif

#define DEF_P25(NAME, FS5)                                                   \
static inline __attribute__((always_inline))                                 \
void NAME##_dif(double *restrict p, const ptrdiff_t st,                      \
                const double *restrict cp)                                   \
{                                                                            \
    (void)cp;                                                                \
    _Pragma("GCC unroll 5")                                                  \
    for (int n1_ = 0; n1_ < 5; ++n1_) D5STAGE(n1_, 5, n1_, SSTW);            \
    _Pragma("GCC unroll 5")                                                  \
    for (int k2_ = 0; k2_ < 5; ++k2_) FS5(5 * k2_, 1);                       \
}                                                                            \
static inline __attribute__((always_inline))                                 \
void NAME##_dit(double *restrict p, const ptrdiff_t st,                      \
                const double *restrict cp)                                   \
{                                                                            \
    (void)cp;                                                                \
    _Pragma("GCC unroll 5")                                                  \
    for (int n1_ = 0; n1_ < 5; ++n1_) D5STAGE(5 * n1_, 1, n1_, SSTW);        \
    _Pragma("GCC unroll 5")                                                  \
    for (int k1_ = 0; k1_ < 5; ++k1_) FS5(k1_, 5);                           \
}
DEF_P25(p25,  D5FS_PLAIN)                 /* plain final stage (z/y passes) */
DEF_P25(p25m, D5FS_MAP)                   /* map-fused final stage (x pass) */

/* ---- L=27 pencils: n = n1 + 3 n2 + 9 n3; trit-reversal is the involution.
 * DIF, natural in -> trit-reversed out (X[k] at slot rev(k)):
 *   stage 1, group (n2,n3): slots {3n2+n3 + 9i}, DFT3 over n1 -> k1,
 *            tw W27^{k1 (3n2+n3)}
 *   stage 2, group (k1,n3): slots {9k1+n3 + 3i}, DFT3 over n2 -> k2,
 *            tw W9^{k2 n3}
 *   stage 3, group (k1,k2): slots {9k1+3k2 + i}, DFT3 over n3 -> k3, plain
 * DIT, trit-reversed in -> natural out:
 *   stage 1, group (n1,n2): slots {9n1+3n2 + i}, DFT3 over n3 -> k1,
 *            tw W9^{n2 k1}
 *   stage 2, group (n1,k1): slots {9n1+k1 + 3i}, DFT3 over n2 -> k2,
 *            tw W27^{n1 (k1+3k2)}
 *   stage 3, group (k1,k2): slots {3k2+k1 + 9i}, DFT3 over n1 -> k3, plain */
/* final-stage forms for the DFT3 pencils (see D5FS_* above) */
#define D3FS_PLAIN(B, S) D3STAGE(B, S, 0, 0, 0, C27T, S27T, SSTW)
#ifdef GENPWP_NOMAPPAIR
#define D3FS_MAP(B, S)   D3STAGE(B, S, 0, 0, 0, C27T, S27T, SSTM)
#else
#define D3FS_MAP(B, S)   D3STAGEMP(B, S)
#endif

#define DEF_P27(NAME, FS3)                                                   \
static inline __attribute__((always_inline))                                 \
void NAME##_dif(double *restrict p, const ptrdiff_t st,                      \
                const double *restrict cp)                                   \
{                                                                            \
    (void)cp;                                                                \
    _Pragma("GCC unroll 3")                                                  \
    for (int n2_ = 0; n2_ < 3; ++n2_)                                        \
        _Pragma("GCC unroll 3")                                              \
        for (int n3_ = 0; n3_ < 3; ++n3_)                                    \
            D3STAGE(3 * n2_ + n3_, 9, 0,                                     \
                    3 * n2_ + n3_, 2 * (3 * n2_ + n3_), C27T, S27T, SSTW);   \
    _Pragma("GCC unroll 3")                                                  \
    for (int k1_ = 0; k1_ < 3; ++k1_)                                        \
        _Pragma("GCC unroll 3")                                              \
        for (int n3_ = 0; n3_ < 3; ++n3_)                                    \
            D3STAGE(9 * k1_ + n3_, 3, 0, n3_, 2 * n3_, C9T, S9T, SSTW);      \
    _Pragma("GCC unroll 3")                                                  \
    for (int k1_ = 0; k1_ < 3; ++k1_)                                        \
        _Pragma("GCC unroll 3")                                              \
        for (int k2_ = 0; k2_ < 3; ++k2_)                                    \
            FS3(9 * k1_ + 3 * k2_, 1);                                       \
}                                                                            \
static inline __attribute__((always_inline))                                 \
void NAME##_dit(double *restrict p, const ptrdiff_t st,                      \
                const double *restrict cp)                                   \
{                                                                            \
    (void)cp;                                                                \
    _Pragma("GCC unroll 3")                                                  \
    for (int n1_ = 0; n1_ < 3; ++n1_)                                        \
        _Pragma("GCC unroll 3")                                              \
        for (int n2_ = 0; n2_ < 3; ++n2_)                                    \
            D3STAGE(9 * n1_ + 3 * n2_, 1, 0, n2_, 2 * n2_, C9T, S9T, SSTW);  \
    _Pragma("GCC unroll 3")                                                  \
    for (int n1_ = 0; n1_ < 3; ++n1_)                                        \
        _Pragma("GCC unroll 3")                                              \
        for (int k1_ = 0; k1_ < 3; ++k1_)                                    \
            D3STAGE(9 * n1_ + k1_, 3, n1_ * k1_,                             \
                    n1_ * (k1_ + 3), n1_ * (k1_ + 6), C27T, S27T, SSTW);     \
    _Pragma("GCC unroll 3")                                                  \
    for (int k1_ = 0; k1_ < 3; ++k1_)                                        \
        _Pragma("GCC unroll 3")                                              \
        for (int k2_ = 0; k2_ < 3; ++k2_)                                    \
            FS3(3 * k2_ + k1_, 9);                                           \
}
DEF_P27(p27,  D3FS_PLAIN)
DEF_P27(p27m, D3FS_MAP)

/* padded plane strides, sites: L^2 rounded up to == 2 (mod 32) so plane
 * bytes == 256 (mod 4096) -- gen_pfa_small/gen_batchlane's anti-alias pad */
#define SOAPL25 642
#define SOAPL27 738

/* per-axis digit permutations (X[k] slot after a DIF pass; involutions) */
static const unsigned char g_pm25[25] = {
     0,  5, 10, 15, 20,  1,  6, 11, 16, 21,  2,  7, 12, 17, 22,
     3,  8, 13, 18, 23,  4,  9, 14, 19, 24 };
static const unsigned char g_pm27[27] = {
     0,  9, 18,  3, 12, 21,  6, 15, 24,  1, 10, 19,  4, 13, 22,
     7, 16, 25,  2, 11, 20,  5, 14, 23,  8, 17, 26 };

/* one chain step over one 8-volume group: zy sweep per x-plane (in place),
 * then the x pass with the map fused into the final-stage stores.  DIF and
 * DIT variants; C must be packed in the matching site layout (reversed for
 * DIF, natural for DIT). */
/* -DGENPWP_SOASCHED: pre-RA pressure scheduling on the soa_step wrappers
 * (gen_batchlane r3 measured -5.7..-10% on their register-explicit pencils).
 * Node verdict this round, control-first pairs with MKL steady: LOSES 32%
 * at 25 (34.3 -> 45.4) and 22% at 27 (47.8 -> 58.3) -- their win is specific
 * to spill-heavy codelets; these pencils hold <= 16 live vecs and gcc's
 * default schedule already forwards ~106 loads / kills ~61 stores per step
 * function (objdump audit).  Default stays OFF. */
#ifdef GENPWP_SOASCHED
#define SOAOPT __attribute__((optimize("schedule-insns", "sched-pressure")))
#else
#define SOAOPT
#endif

#define DEF_SOASTEP(L, PLV, PZ, PZM)                                         \
static void SOAOPT soa_step_dif_##L(double *restrict S, const double *restrict C) \
{                                                                            \
    for (int x = 0; x < (L); ++x) {                                          \
        double *pl = S + (size_t)x * ((PLV) * 16);                           \
        for (int y = 0; y < (L); ++y)                                        \
            PZ##_dif(pl + (size_t)y * ((L) * 16), 16, 0);                    \
        for (int z = 0; z < (L); ++z)                                        \
            PZ##_dif(pl + (size_t)z * 16, (L) * 16, 0);                      \
    }                                                                        \
    for (int c = 0; c < (L) * (L); ++c)                                      \
        PZM##_dif(S + (size_t)c * 16, (ptrdiff_t)((PLV) * 16),               \
                  C + (size_t)c * 16);                                       \
}                                                                            \
static void SOAOPT soa_step_dit_##L(double *restrict S, const double *restrict C) \
{                                                                            \
    for (int x = 0; x < (L); ++x) {                                          \
        double *pl = S + (size_t)x * ((PLV) * 16);                           \
        for (int y = 0; y < (L); ++y)                                        \
            PZ##_dit(pl + (size_t)y * ((L) * 16), 16, 0);                    \
        for (int z = 0; z < (L); ++z)                                        \
            PZ##_dit(pl + (size_t)z * 16, (L) * 16, 0);                      \
    }                                                                        \
    for (int c = 0; c < (L) * (L); ++c)                                      \
        PZM##_dit(S + (size_t)c * 16, (ptrdiff_t)((PLV) * 16),               \
                  C + (size_t)c * 16);                                       \
}
DEF_SOASTEP(25, SOAPL25, p25, p25m)
DEF_SOASTEP(27, SOAPL27, p27, p27m)

/* pack 8 interleaved volumes (volume stride vstr complex) into the padded
 * SoA arena; pm = per-axis site permutation, NULL = natural.  Natural rows
 * go through gen_layout's gl_pack8; permuted rows scatter sites through the
 * same gl_tr8x8 network. */
static void soa_pack(const double _Complex *src, size_t vstr, double *S,
                     int L, int PL, const unsigned char *pm)
{
    for (int x = 0; x < L; ++x)
        for (int y = 0; y < L; ++y) {
            const double _Complex *row = src + ((size_t)x * L + y) * L;
            double *dst = S + ((size_t)(pm ? pm[x] : x) * PL
                             + (size_t)(pm ? pm[y] : y) * L) * 16;
            if (!pm) { gl_pack8(row, vstr, dst, (size_t)L); continue; }
            int s = 0;
            for (; s + 4 <= L; s += 4) {
                __m512d r[8];
                for (int t = 0; t < 8; ++t)
                    r[t] = _mm512_loadu_pd((const double *)(row + t * vstr + s));
                gl_tr8x8(r);
                for (int q = 0; q < 4; ++q) {
                    _mm512_storeu_pd(dst + (size_t)pm[s + q] * 16,     r[2 * q]);
                    _mm512_storeu_pd(dst + (size_t)pm[s + q] * 16 + 8, r[2 * q + 1]);
                }
            }
            for (; s < L; ++s)
                for (int t = 0; t < 8; ++t) {
                    dst[(size_t)pm[s] * 16 + t]     = creal(row[t * vstr + s]);
                    dst[(size_t)pm[s] * 16 + 8 + t] = cimag(row[t * vstr + s]);
                }
        }
}

/* exact inverse: SoA arena (site layout pm, NULL = natural) -> 8 volumes */
static void soa_unpack(const double *S, double _Complex *dst, size_t vstr,
                       int L, int PL, const unsigned char *pm)
{
    for (int x = 0; x < L; ++x)
        for (int y = 0; y < L; ++y) {
            double _Complex *row = dst + ((size_t)x * L + y) * L;
            const double *sr = S + ((size_t)(pm ? pm[x] : x) * PL
                                  + (size_t)(pm ? pm[y] : y) * L) * 16;
            if (!pm) { gl_unpack8(sr, row, vstr, (size_t)L); continue; }
            int s = 0;
            for (; s + 4 <= L; s += 4) {
                __m512d r[8];
                for (int q = 0; q < 4; ++q) {
                    r[2 * q]     = _mm512_loadu_pd(sr + (size_t)pm[s + q] * 16);
                    r[2 * q + 1] = _mm512_loadu_pd(sr + (size_t)pm[s + q] * 16 + 8);
                }
                gl_tr8x8(r);
                for (int t = 0; t < 8; ++t)
                    _mm512_storeu_pd((double *)(row + t * vstr + s), r[t]);
            }
            for (; s < L; ++s)
                for (int t = 0; t < 8; ++t)
                    row[t * vstr + s] = sr[(size_t)pm[s] * 16 + t]
                                      + I * sr[(size_t)pm[s] * 16 + 8 + t];
        }
}

#endif /* __AVX512F__ */

/* ---- plan, gate, race, API ---------------------------------------------- */

typedef void (*execpl_fn)(const double *in, double *out, long nvol,
                          double *M, double *P);
typedef void (*chainpl_fn)(const double *cur, double *dst, const double *cf,
                           long nvol, double *M, double *P);

struct fft3d_plan {
    int L, batch;
    execpl_fn  fn;                /* NULL -> dense fallback                  */
    chainpl_fn cfn;               /* NULL -> execute + scalar map            */
    int use_soa;                  /* chain owned by the SoA-8 lane engine    */
    int dm;                       /* cfn is a deferred-map (ipm) step        */
    double *P;                    /* plane scratch (heap: stack 4K-aliases)  */
    double *M;                    /* padded mid volume (phase 1 -> phase 2)  */
    double *X;                    /* batch state volumes for chain ping-pong */
    double *SA, *CN, *CR;         /* SoA state + c (natural / digit-reversed)*/
    gl_arena soa_ar;              /* THP arena for the three (gen_layout)    */
    int soa_live;
    void   *rawP, *rawM, *rawX;
    double _Complex *w, *tmp;     /* dense fallback state                    */
};

/* the driver's MAP_STEP, verbatim semantics: z -> (z+c)/(1+|z+c|) */
static void map_scalar(const double *z, const double *c, double *o, size_t npts)
{
    for (size_t i = 0; i < npts; ++i) {
        double re = z[2 * i]     + c[2 * i];
        double im = z[2 * i + 1] + c[2 * i + 1];
        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
        o[2 * i]     = re * sc;
        o[2 * i + 1] = im * sc;
    }
}

const char *fft3d_name(void) { return "gen_powp"; }

static char g_desc[224];
const char *fft3d_description(void)
{
    return g_desc[0] ? g_desc
        : "prime-power CT, exact twiddles: OWN 25 (5x5), 27 (3x9), 50 (GT 25x2), "
          "100 (GT 25x4) + odd-p^k class 49/81/121/125 (round 6); two-sweep "
          "zmm lanes + SoA-8 DIF/DIT lane chain at 25/27 (paired-div map), "
          "volume-major chain, deferred-map ipm/ipp (pair-packed ladder), "
          "c-bypass ipq/ipk/iqn, gate+race+wisdom in create()";
}
int fft3d_supports(int L)
{
    return L == 25 || L == 27 || L == 50 || L == 100 ||
           L == 49 || L == 81 || L == 121 || L == 125;
}

/* scalar O(L^2)-per-line reference: independent ground truth for the
 * create()-time gate (from gen_pfa_large / L45_pfa's ref3d) */
static void __attribute__((unused))       /* unused in the no-AVX512 build */
refnd(int L, const double _Complex *in, double _Complex *out)
{
    const size_t NP = (size_t)L * L;
    double _Complex Wt[128], buf[128];
    for (int k = 0; k < L; ++k)
        Wt[k] = cexp(-2.0 * M_PI * I * (double)k / (double)L);
    for (int x = 0; x < L; ++x)                       /* z axis: in -> out  */
        for (int y = 0; y < L; ++y) {
            const double _Complex *r = in  + ((size_t)x * L + y) * L;
            double _Complex       *w = out + ((size_t)x * L + y) * L;
            for (int k = 0; k < L; ++k) {
                double _Complex s = 0;
                for (int j = 0; j < L; ++j) s += r[j] * Wt[(j * k) % L];
                w[k] = s;
            }
        }
    for (int x = 0; x < L; ++x)                       /* y axis, in place   */
        for (int z = 0; z < L; ++z) {
            double _Complex *base = out + (size_t)x * NP + z;
            for (int j = 0; j < L; ++j) buf[j] = base[(size_t)j * L];
            for (int k = 0; k < L; ++k) {
                double _Complex s = 0;
                for (int j = 0; j < L; ++j) s += buf[j] * Wt[(j * k) % L];
                base[(size_t)k * L] = s;
            }
        }
    for (int y = 0; y < L; ++y)                       /* x axis, in place   */
        for (int z = 0; z < L; ++z) {
            double _Complex *base = out + (size_t)y * L + z;
            for (int j = 0; j < L; ++j) buf[j] = base[(size_t)j * NP];
            for (int k = 0; k < L; ++k) {
                double _Complex s = 0;
                for (int j = 0; j < L; ++j) s += buf[j] * Wt[(j * k) % L];
                base[(size_t)k * NP] = s;
            }
        }
}

/* dense fallback (the old stub): floor of last resort, known correct */
static void dense_contract(const double _Complex *w, int L,
                           const double _Complex *in, double _Complex *out,
                           int inner)
{
    for (int k = 0; k < L; ++k)
        for (int c = 0; c < inner; ++c) {
            double _Complex acc = 0.0;
            for (int j = 0; j < L; ++j)
                acc += w[(size_t)k * L + j] * in[(size_t)j * inner + c];
            out[(size_t)k * inner + c] = acc;
        }
}

static void dense_exec(const fft3d_plan *p, const double _Complex *in,
                       double _Complex *out)
{
    const int L = p->L;
    const size_t volume = (size_t)L * L * L;
    for (int b = 0; b < p->batch; ++b) {
        const double _Complex *src = in + (size_t)b * volume;
        double _Complex *dst = out + (size_t)b * volume;
        dense_contract(p->w, L, src, dst, L * L);
        for (int x = 0; x < L; ++x)
            dense_contract(p->w, L, dst + (size_t)x * L * L,
                           p->tmp + (size_t)x * L * L, L);
        for (size_t row = 0; row < volume / (size_t)L; ++row)
            dense_contract(p->w, L, p->tmp + row * (size_t)L,
                           dst + row * (size_t)L, 1);
    }
}

static int dense_setup(fft3d_plan *p)
{
    const int L = p->L;
    p->w   = malloc((size_t)L * L * sizeof *p->w);
    p->tmp = malloc((size_t)L * L * L * sizeof *p->tmp);
    if (!p->w || !p->tmp) return 0;
    for (int k = 0; k < L; ++k)
        for (int j = 0; j < L; ++j) {
            double phase = -2.0 * M_PI * (double)((k * j) % L) / (double)L;
            p->w[(size_t)k * L + j] = cos(phase) + I * sin(phase);
        }
    return 1;
}

#ifdef __AVX512F__

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int rel_ok(const double *got, const double *ref, size_t n)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = got[i] - ref[i];
        num += d * d; den += ref[i] * ref[i];
    }
    return num <= den * 1e-26;                        /* rel L2 < 1e-13 */
}

struct candpl { execpl_fn fn; chainpl_fn cfn; int pf, rank, dm;
                const char *nm; };

/* candidates are raced on the CHAIN step (the graded workload); the exec fn
 * rides along with the winning candidate.  FIVE chain families:
 *   ip*: everything in place + a sequential map pass (one volume-sized
 *        working set; wins when miss streams bind -- gen_pfa_large L=100)
 *   ipf: in place AND map fused into phase 2's stores (no map pass)
 *   f*:  map fused into phase 2, routed through the padded mid volume M
 *   ipm: (gen_r3, ADOPTED from gen_pfa_large gen_r3) deferred map applied
 *        on the NEXT step's p1 loads -- dm == 1, needs the whole-chain
 *        schedule in fft3d_chain; highest rank (must clearly win)
 *   ipp: (gen_r4, ADOPTED from gen_pfa_large gen_r4) deferred map as a
 *        per-plane sequential PREPASS into an L2 scratch plane (M's base);
 *        dm == 1 like ipm, ranked between the base families and ipm
 *   soa: (25/27, batch %% 8 == 0) the SoA-8 batch-lane whole-chain engine;
 *        cfn == NULL marks it, must beat the best interleaved by > 3% */
static const struct candpl g_c25[]  = {
    { x_ip0_25,  xc_ip0_25,   0, 0, 0, "l25-ip0"   },
    { x_ip1_25,  xc_ip1_25,   1, 1, 0, "l25-ip1"   },
    { x_ip0_25,  xc_ipf_25,   2, 2, 0, "l25-ipf"   },
    { x_pf0_25,  xc_pf0_25,   3, 3, 0, "l25-f0"    },
    { x_pf1_25,  xc_pfr_25,   4, 4, 0, "l25-fr"    },
    { x_pf1_25,  xc_pfrw_25,  5, 5, 0, "l25-frw"   },
    { x_ip0_25,  0,           6, 6, 0, "l25-soa"   },
    { x_ip0_25,  xc_ipp0_25,  9, 7, 1, "l25-ipp0"  } };
static const struct candpl g_c27[]  = {
    { x_ip0_27,  xc_ip0_27,   0, 0, 0, "l27-ip0"   },
    { x_ip1_27,  xc_ip1_27,   1, 1, 0, "l27-ip1"   },
    { x_ip0_27,  xc_ipf_27,   2, 2, 0, "l27-ipf"   },
    { x_pf0_27,  xc_pf0_27,   3, 3, 0, "l27-f0"    },
    { x_pf1_27,  xc_pfr_27,   4, 4, 0, "l27-fr"    },
    { x_pf1_27,  xc_pfrw_27,  5, 5, 0, "l27-frw"   },
    { x_ip0_27,  0,           6, 6, 0, "l27-soa"   },
    { x_ip0_27,  xc_ipp0_27,  9, 7, 1, "l27-ipp0"  } };
/* rank note (gen_r4): at 50/100 the ipp family is ranked FIRST -- 5-pair
 * same-core alternation on the node (batchlane r4's protocol) puts ipp1 at
 * -4.0..-4.5% vs ip0 at 100 (quiet floors 4905-4929 vs 5110-5160 us), and
 * both the node race and wallaby put ipp ahead at 50; a busy-window race
 * margin can shrink under the 3% hysteresis, and the tie must then fall to
 * the measured winner, not the simpler loser. */
static const struct candpl g_c50[]  = {
    { x_ip0_50,  xc_ipp0_50,  9, 0, 1, "l50-ipp0"  },
    { x_ip1_50,  xc_ipp1_50, 10, 1, 1, "l50-ipp1"  },
    { x_ip0_50,  xc_ip0_50,   0, 2, 0, "l50-ip0"   },
    { x_ip1_50,  xc_ip1_50,   1, 3, 0, "l50-ip1"   },
    { x_ip0_50,  xc_ipf_50,   2, 4, 0, "l50-ipf"   },
    { x_pf0_50,  xc_pf0_50,   3, 5, 0, "l50-f0"    },
    { x_pf1_50,  xc_pfr_50,   4, 6, 0, "l50-fr"    },
    { x_pf1_50,  xc_pfrw_50,  5, 7, 0, "l50-frw"   },
    { x_ip0_50,  xc_ipm0_50,  7, 8, 1, "l50-ipm0"  },
    { x_ip1_50,  xc_ipm1_50,  8, 9, 1, "l50-ipm1"  },
    { x_ip1_50,  xc_ipq1_50, 11, 10, 1, "l50-ipq1" },
    { x_ip1_50,  xc_ipk1_50, 12, 11, 1, "l50-ipk1" },
    { x_ip1_50,  xc_iqn1_50, 13, 12, 0, "l50-iqn1" } };
static const struct candpl g_c100[] = {
    { x_ip1_100, xc_ipp1_100,10, 0, 1, "l100-ipp1" },
    { x_ip0_100, xc_ipp0_100, 9, 1, 1, "l100-ipp0" },
    { x_ip0_100, xc_ip0_100,  0, 2, 0, "l100-ip0"  },
    { x_ip1_100, xc_ip1_100,  1, 3, 0, "l100-ip1"  },
    { x_ip0_100, xc_ipf_100,  2, 4, 0, "l100-ipf"  },
    { x_pf0_100, xc_pf0_100,  3, 5, 0, "l100-f0"   },
    { x_pf1_100, xc_pfr_100,  4, 6, 0, "l100-fr"   },
    { x_pf1_100, xc_pfrw_100, 5, 7, 0, "l100-frw"  },
    { x_ip0_100, xc_ipm0_100, 7, 8, 1, "l100-ipm0" },
    { x_ip1_100, xc_ipm1_100, 8, 9, 1, "l100-ipm1" },
    { x_ip1_100, xc_ipq1_100,11, 10, 1, "l100-ipq1" },
    { x_ip1_100, xc_ipk1_100,12, 11, 1, "l100-ipk1" },
    { x_ip1_100, xc_iqn1_100,13, 12, 0, "l100-iqn1" } };
/* gen_r3 lite sizes (round-6 coverage): reduced candidate pool */
static const struct candpl g_c49[]  = {
    { x_ip0_49,  xc_ip0_49,   0, 0, 0, "l49-ip0"   },
    { x_ip0_49,  xc_ipp0_49,  9, 1, 1, "l49-ipp0"  },
    { x_ip0_49,  xc_ipf_49,   2, 2, 0, "l49-ipf"   },
    { x_pf0_49,  xc_pf0_49,   3, 3, 0, "l49-f0"    },
    { x_ip0_49,  xc_ipm0_49,  7, 4, 1, "l49-ipm0"  },
    { x_ip0_49,  xc_ipq0_49, 14, 5, 1, "l49-ipq0"  } };
static const struct candpl g_c81[]  = {
    { x_ip0_81,  xc_ip0_81,   0, 0, 0, "l81-ip0"   },
    { x_ip0_81,  xc_ipp0_81,  9, 1, 1, "l81-ipp0"  },
    { x_ip0_81,  xc_ipf_81,   2, 2, 0, "l81-ipf"   },
    { x_pf0_81,  xc_pf0_81,   3, 3, 0, "l81-f0"    },
    { x_ip0_81,  xc_ipm0_81,  7, 4, 1, "l81-ipm0"  },
    { x_ip0_81,  xc_ipq0_81, 14, 5, 1, "l81-ipq0"  } };
static const struct candpl g_c121[] = {
    { x_ip0_121, xc_ip0_121,  0, 0, 0, "l121-ip0"  },
    { x_ip0_121, xc_ipp0_121, 9, 1, 1, "l121-ipp0" },
    { x_ip0_121, xc_ipf_121,  2, 2, 0, "l121-ipf"  },
    { x_pf0_121, xc_pf0_121,  3, 3, 0, "l121-f0"   },
    { x_ip0_121, xc_ipm0_121, 7, 4, 1, "l121-ipm0" },
    { x_ip0_121, xc_ipq0_121,14, 5, 1, "l121-ipq0" } };
static const struct candpl g_c125[] = {
    { x_ip0_125, xc_ip0_125,  0, 0, 0, "l125-ip0"  },
    { x_ip0_125, xc_ipp0_125, 9, 1, 1, "l125-ipp0" },
    { x_ip0_125, xc_ipf_125,  2, 2, 0, "l125-ipf"  },
    { x_pf0_125, xc_pf0_125,  3, 3, 0, "l125-f0"   },
    { x_ip0_125, xc_ipm0_125, 7, 4, 1, "l125-ipm0" },
    { x_ip0_125, xc_ipq0_125,14, 5, 1, "l125-ipq0" } };
#define NCMAX 13

/* the whole graded chain through the SoA-8 lane engine: per group of 8
 * volumes, pack once (x0 natural; c in BOTH site layouts), alternate
 * DIF/DIT steps in place, unpack once (layout by m's parity) */
static void soa_chain_n(fft3d_plan *p, const double _Complex *x0,
                        const double _Complex *c, double _Complex *out,
                        int m, long nvol)
{
    const int L  = p->L;
    const int PL = (L == 25) ? SOAPL25 : SOAPL27;
    const unsigned char *pm = (L == 25) ? g_pm25 : g_pm27;
    void (*stepd)(double *restrict, const double *restrict) =
        (L == 25) ? soa_step_dif_25 : soa_step_dif_27;
    void (*stept)(double *restrict, const double *restrict) =
        (L == 25) ? soa_step_dit_25 : soa_step_dit_27;
    const size_t vstr = (size_t)L * L * L;
    for (long g = 0; g < nvol / 8; ++g) {
        const double _Complex *xg = x0 + (size_t)(8 * g) * vstr;
        const double _Complex *cg = c  + (size_t)(8 * g) * vstr;
        soa_pack(xg, vstr, p->SA, L, PL, NULL);
        soa_pack(cg, vstr, p->CN, L, PL, NULL);
        soa_pack(cg, vstr, p->CR, L, PL, pm);
        for (int s = 0; s < m; ++s)
            if (s & 1) stept(p->SA, p->CN);
            else       stepd(p->SA, p->CR);
        soa_unpack(p->SA, out + (size_t)(8 * g) * vstr, vstr, L, PL,
                   (m & 1) ? pm : NULL);
    }
}

static const char *powp_fam(int L)
{
    switch (L) {
    case 25:  return "5x5";          case 27:  return "3x9(3x3)";
    case 50:  return "GT 25x2(5x5)"; case 100: return "GT 25x4(5x5)";
    case 49:  return "7x7";          case 81:  return "9x9(3x3)";
    case 121: return "11x11";        default:  return "5x25(5x5)";
    }
}

static void install_pick(fft3d_plan *p, const struct candpl *cd, int pick,
                         int trust_chain)
{
    p->fn = cd[pick].fn;
    p->cfn = NULL;
    p->use_soa = 0;
    p->dm = 0;
    if (!getenv("GENPWP_NOFUSE")) {
        if (!cd[pick].cfn)           p->use_soa = 1;
        else if (trust_chain) {      p->cfn = cd[pick].cfn;
                                     p->dm  = cd[pick].dm; }
        /* trust_chain == 0: the cold path gates cfn itself afterwards */
    }
    snprintf(g_desc, sizeof g_desc,
             "powp CT %s exact tw%s; pick: %s (B=%d)",
             powp_fam(p->L),
             p->use_soa ? ", SoA-8 lane chain (DIF/DIT in place)"
             : p->dm    ? ", two-sweep, deferred map"
                        : ", two-sweep",
             cd[pick].nm, p->batch);
}

/* gate + race + wisdom; installs p->fn (or leaves NULL: dense fallback) */
static void tune(fft3d_plan *p)
{
    const int L = p->L;
    const size_t VD = (size_t)2 * L * L * L;
    const struct candpl *cd;
    int NC;
    switch (L) {
    case 25:  cd = g_c25;  NC = 8; break;
    case 27:  cd = g_c27;  NC = 8; break;
    case 50:  cd = g_c50;  NC = 13; break;
    case 100: cd = g_c100; NC = 13; break;
    case 49:  cd = g_c49;  NC = 6; break;
    case 81:  cd = g_c81;  NC = 6; break;
    case 121: cd = g_c121; NC = 6; break;
    default:  cd = g_c125; NC = 6; break;
    }

    int    live[NCMAX];
    double tc[NCMAX];
    for (int c = 0; c < NC; ++c) { live[c] = 1; tc[c] = 1e300; }
    for (int c = 0; c < NC; ++c)
        if (!cd[c].cfn && !p->SA) live[c] = 0;        /* soa needs its arena */

    { const char *e = getenv("GENPWP_PF");            /* monitor forcing */
      if (e) { int v = atoi(e);
               for (int c = 0; c < NC; ++c) if (cd[c].pf != v) live[c] = 0;
               int any = 0;
               for (int c = 0; c < NC; ++c) any |= live[c];
               if (!any) for (int c = 0; c < NC; ++c)
                   live[c] = cd[c].cfn || p->SA ? 1 : 0; } }

    /* per-host wisdom (ADOPTED: gen_race).  The key carries the signature of
     * the OFFERED candidate set, so a changed pool (or a GENPWP_PF filter)
     * misses instead of silently replaying stale wisdom.  A hit is the 50 ms
     * warm-create path AND pins the pick across processes -- which is what
     * makes the non-bit-identical soa family safe against the driver's
     * two-process repeatability cmp. */
    char fullkey[GR_KEY_MAX + 16];
    { char key[GR_KEY_MAX];
      gr_cand sc[NCMAX]; int ns = 0;
      memset(sc, 0, sizeof sc);
      for (int c = 0; c < NC; ++c)
          if (live[c]) sc[ns++].name = cd[c].nm;
      /* gen_r4: tag bumped chain2 -> chain4 (gen_pfa_large r4's practice) --
       * the race now times the VOLUME-MAJOR schedule, so a stale step-major
       * verdict must never be installed from wisdom.  gen_r5: chain4 ->
       * chain5 (pool gained the c-bypass families; the soa x-pass map is
       * paired-div, so a stale soa-vs-interleaved margin must re-race). */
      gr_keyf(key, sizeof key, "gen_powp", "chain5", L, gr_bucket(p->batch));
      snprintf(fullkey, sizeof fullkey, "%s#%08x", key, gr_sig(sc, ns)); }

    if (!getenv("GEN_RACE_NO_WISDOM") && !getenv("GEN_RACE_REFRESH")) {
        char wname[64]; int widx = -1, wtie = 0; double wus = 0.0;
        if (gr_wisdom_lookup(fullkey, wname, sizeof wname, &widx, &wtie, &wus))
            for (int c = 0; c < NC; ++c)
                if (live[c] && !strcmp(cd[c].nm, wname)) {
                    install_pick(p, cd, c, 1);   /* warm: no gate, no race */
                    return;
                }
    }

    /* arena: stream realistically at large batch, cap the footprint; a
     * multiple of 8 volumes whenever the soa candidate is offered.  tcf is
     * a DISTINCT c-field buffer (gen_r3, ADOPTED from gen_pfa_large's r3
     * fix: passing tin as both state and c made ipm's back-to-back state+c
     * loads hit the same lines, halving its apparent read traffic -- race
     * what is graded, including the STREAMS). */
    const int cap = (int)(1 + (size_t)32 * 1024 * 1024 / (VD * 8));
    int nv  = p->batch < cap ? p->batch : cap;
    int soa_on = 0;
    for (int c = 0; c < NC; ++c) if (live[c] && !cd[c].cfn) soa_on = 1;
    if (soa_on && nv >= 8) nv &= ~7;
    void *ri = NULL, *ro = NULL, *rc = NULL, *r0 = NULL, *r1 = NULL;
    if (posix_memalign(&ri, 64, (size_t)nv * VD * sizeof(double)) ||
        posix_memalign(&ro, 64, (size_t)nv * VD * sizeof(double)) ||
        posix_memalign(&rc, 64, (size_t)nv * VD * sizeof(double)) ||
        posix_memalign(&r0, 64, VD * sizeof(double)) ||
        (nv > 1 && posix_memalign(&r1, 64, VD * sizeof(double)))) {
        free(ri); free(ro); free(rc); free(r0);
        return;                                       /* dense fallback */
    }
    double *tin = ri, *tout = ro, *tcf = rc, *ref0 = r0, *refN = r1;
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < (size_t)nv * VD; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        tin[i] = (double)(long long)(s >> 11) * 0x1p-53;
    }
    for (size_t i = 0; i < (size_t)nv * VD; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        tcf[i] = 0.1 * (double)(long long)(s >> 11) * 0x1p-53;
    }
    refnd(L, (const double _Complex *)tin, (double _Complex *)ref0);
    if (nv > 1)          /* gate the LAST volume too: M/P-reuse bugs cannot hide */
        refnd(L, (const double _Complex *)(tin + (size_t)(nv - 1) * VD),
              (double _Complex *)refN);

    for (int c = 0; c < NC; ++c) {
        if (!live[c] || !cd[c].cfn) continue;
        memset(tout, 0, (size_t)nv * VD * sizeof(double));
        cd[c].fn(tin, tout, nv, p->M, p->P);
        if (!rel_ok(tout, ref0, VD)) live[c] = 0;
        if (nv > 1 && live[c] &&
            !rel_ok(tout + (size_t)(nv - 1) * VD, refN, VD)) live[c] = 0;
    }

    /* soa gate: the whole m=2 chain on the first 8 volumes against an
     * INDEPENDENT scalar reference (refnd + the driver's exact map) -- this
     * exercises pack, DIF, the reversed-c map, DIT, the natural-c map and
     * unpack in one shot.  Lanes 1..7 catch pack/unpack lane bugs. */
    if (soa_on && nv >= 8) {
        void *w1 = NULL, *w2 = NULL;
        if (!posix_memalign(&w1, 64, (size_t)8 * VD * sizeof(double)) &&
            !posix_memalign(&w2, 64, (size_t)8 * VD * sizeof(double))) {
            double *z1 = w1, *z2 = w2;
            for (int v = 0; v < 8; ++v)
                refnd(L, (const double _Complex *)(tin + (size_t)v * VD),
                      (double _Complex *)(z1 + (size_t)v * VD));
            map_scalar(z1, tcf, z1, 8 * VD / 2);
            for (int v = 0; v < 8; ++v)
                refnd(L, (const double _Complex *)(z1 + (size_t)v * VD),
                      (double _Complex *)(z2 + (size_t)v * VD));
            map_scalar(z2, tcf, z2, 8 * VD / 2);
            soa_chain_n(p, (const double _Complex *)tin,
                        (const double _Complex *)tcf,
                        (double _Complex *)tout, 2, 8);
            for (int c = 0; c < NC; ++c)
                if (!cd[c].cfn && !rel_ok(tout, z2, 8 * VD)) live[c] = 0;
        } else {
            for (int c = 0; c < NC; ++c) if (!cd[c].cfn) live[c] = 0;
        }
        free(w1); free(w2);
    }

    /* min over interleaved rounds; race the CHAIN step in its VOLUME-MAJOR
     * shape (gen_r4, ADOPTED from gen_pfa_large gen_r4's race fix, as their
     * fix to MY r2 tin-as-c bug came the other way): the graded chain now
     * runs all m steps on one volume before the next, so the trial working
     * set must be ONE volume's state + c slice, not the whole batch --
     * racing the old step-major shape would rank candidates on a working
     * set the chain no longer has (at resident sizes the fused families'
     * economics change).  Per volume: one unmeasured warm step, then R
     * timed steps IN PLACE on that tout volume with the DISTINCT tcf slice
     * as the c field (the graded steady state); R = 8/6/4 by volume bytes
     * <=2/<=8/>8 MiB; min over NR interleaved rounds.  tc[] is seconds per
     * VOLUME-step.  State values stay bounded across trials: every family
     * re-maps before the FFT (ipm/ipp) or right after it (all others).
     * The soa trial runs one DIF + one DIT step per group (2*nv
     * volume-steps), on the arenas the m=2 gate just packed -- the soa
     * engine was group-major from birth, so its shape is unchanged. */
    void (*sd_)(double *restrict, const double *restrict) =
        (L == 25) ? soa_step_dif_25 : soa_step_dif_27;
    void (*st_)(double *restrict, const double *restrict) =
        (L == 25) ? soa_step_dit_25 : soa_step_dit_27;
    memcpy(tout, tin, (size_t)nv * VD * sizeof(double));
    const size_t vbytes = VD * sizeof(double);
    const int R  = vbytes <= ((size_t)2 << 20) ? 8
                 : vbytes <= ((size_t)8 << 20) ? 6 : 4;
    const int NR = 4;
    for (int round = 0; round < NR; ++round)
        for (int c = 0; c < NC; ++c) {
            if (!live[c]) continue;
            double t;
            if (!cd[c].cfn) {                         /* soa whole-chain    */
                for (long g_ = 0; g_ < nv / 8; ++g_)
                    { sd_(p->SA, p->CR); st_(p->SA, p->CN); }   /* warm     */
                double t0 = now_s();
                for (int r = 0; r < R; ++r)
                    for (long g_ = 0; g_ < nv / 8; ++g_)
                        { sd_(p->SA, p->CR); st_(p->SA, p->CN); }
                t = (now_s() - t0) / ((double)R * 2.0 * (double)nv);
            } else {                                  /* volume-major       */
                double tsum = 0.0;
                for (int b = 0; b < nv; ++b) {
                    double       *tv = tout + (size_t)b * VD;
                    const double *cv = tcf  + (size_t)b * VD;
                    cd[c].cfn(tv, tv, cv, 1, p->M, p->P);       /* warm     */
                    double t0 = now_s();
                    for (int r = 0; r < R; ++r)
                        cd[c].cfn(tv, tv, cv, 1, p->M, p->P);
                    tsum += now_s() - t0;
                }
                t = tsum / ((double)R * (double)nv);
            }
            if (t < tc[c]) tc[c] = t;
        }
    int best = -1;
    for (int c = 0; c < NC; ++c)
        if (live[c] && (best < 0 || tc[c] < tc[best])) best = c;
    if (best >= 0) {
        int pick = best;                              /* 3% simplest-first */
        for (int c = 0; c < NC; ++c)
            if (live[c] && tc[c] <= tc[best] * 1.03 &&
                cd[c].rank < cd[pick].rank) pick = c;
        install_pick(p, cd, pick, 0);
        /* gate the picked interleaved chain step against execute + the
         * driver's own scalar map on volume 0 (tcf is the c field); on any
         * mismatch fall back to the execute+scalar path.  Deferred-map
         * (ipm) picks gate the full m=2 chain COMPOSITION (execute, one ipm
         * step, trailing map_span) -- a single ipm step's output is a raw
         * FFT volume, not a chain state.  The soa pick was already gated
         * above (m=2, independent reference). */
        int chain_ok = p->use_soa;
        if (cd[pick].cfn && !getenv("GENPWP_NOFUSE")) {
            void *e1 = NULL, *e2 = NULL;
            if (!posix_memalign(&e1, 64, VD * sizeof(double)) &&
                !posix_memalign(&e2, 64, VD * sizeof(double))) {
                double *zed = e1, *exp_ = e2;
                if (cd[pick].dm) {
                    p->fn(tin, zed, 1, p->M, p->P);
                    map_scalar(zed, tcf, exp_, VD / 2);          /* x1      */
                    p->fn(exp_, zed, 1, p->M, p->P);
                    map_scalar(zed, tcf, exp_, VD / 2);          /* x2 ref  */
                    p->fn(tin, tout, 1, p->M, p->P);
                    cd[pick].cfn(tout, tout, tcf, 1, p->M, p->P);
                    map_span(tout, tcf, tout, VD);               /* x2 got  */
                } else {
                    p->fn(tin, zed, 1, p->M, p->P);
                    map_scalar(zed, tcf, exp_, VD / 2);
                    cd[pick].cfn(tin, tout, tcf, 1, p->M, p->P);
                }
                if (rel_ok(tout, exp_, VD)) {
                    p->cfn = cd[pick].cfn;
                    p->dm  = cd[pick].dm;
                    chain_ok = 1;
                }
            }
            free(e1); free(e2);
            snprintf(g_desc, sizeof g_desc,
                     "powp CT %s exact tw, two-sweep%s; pick: %s (B=%d)",
                     powp_fam(L),
                     p->cfn ? (p->dm ? " + owned chain (deferred NR map)"
                                     : " + owned chain (NR map)") : "",
                     cd[pick].nm, p->batch);
        }
        /* persist the decision (gen_race wisdom) -- only a pick whose chain
         * path passed its gate, so a warm hit can trust it */
        if (chain_ok && !getenv("GEN_RACE_NO_WISDOM")) {
            double runner = -1.0;
            for (int c = 0; c < NC; ++c)
                if (live[c] && c != pick && (runner < 0 || tc[c] < runner))
                    runner = tc[c];
            double margin = runner > 0 ? (runner - tc[pick]) / tc[pick] : 0.0;
            gr_wisdom_store(fullkey, cd[pick].nm, pick,
                            margin < 0.03 ? 1 : 0,
                            tc[pick] * 1e6, margin);
        }
    }
    if (getenv("GENPWP_VERBOSE")) {
        for (int c = 0; c < NC; ++c)
            fprintf(stderr, "gen_powp L=%d: %-9s %s %.1f us/vol\n",
                    L, cd[c].nm, live[c] ? "ok " : "OUT",
                    live[c] ? tc[c] * 1e6 : 0.0);
    }
    free(ri); free(ro); free(rc); free(r0); free(r1);
}
#endif /* __AVX512F__ */

fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L) || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    const size_t VD = (size_t)2 * L * L * L;
    /* chain ping-pong state for the non-fused fallback path */
    if (posix_memalign(&p->rawX, 4096,
                       (size_t)batch * VD * sizeof(double)) != 0) {
        free(p);
        return NULL;
    }
    p->X = (double *)p->rawX;
#ifdef __AVX512F__
    /* SoA-8 lane arenas (gen_layout THP arena, staggered mod-4096 phases):
     * only for 25/27 at whole groups of 8 -- the roster's B >= 8 scope */
    if ((L == 25 || L == 27) && batch >= 8 && batch % 8 == 0) {
        const int PL = (L == 25) ? SOAPL25 : SOAPL27;
        const size_t bytes = (size_t)L * PL * 16 * sizeof(double);
        if (gl_arena_init(&p->soa_ar, 3 * (bytes + 8192) + (64 << 10)) == 0) {
            p->soa_live = 1;
            p->SA = (double *)gl_arena_take(&p->soa_ar, bytes + 4096);
            p->CN = (double *)gl_arena_take(&p->soa_ar, bytes + 4096);
            p->CR = (double *)gl_arena_take(&p->soa_ar, bytes + 4096);
            if (!p->SA || !p->CN || !p->CR) {
                gl_arena_destroy(&p->soa_ar);
                p->soa_live = 0;
                p->SA = p->CN = p->CR = NULL;
            }
        }
    }
    powp_rt_tabs();                   /* gen_r3 runtime exact tables (~us)   */
    const int pp = (L == 25) ? 28 : (L == 27) ? 28 : (L == 50) ? 52
                 : (L == 49) ? 52 : (L == 81) ? 84 : (L == 121) ? 124
                 : (L == 125) ? 132 : 108;
    if (posix_memalign(&p->rawP, 4096,
                       (size_t)L * pp * 2 * sizeof(double)) == 0) {
        p->P = (double *)p->rawP;
        memset(p->P, 0, (size_t)L * pp * 2 * sizeof(double));
        if (posix_memalign(&p->rawM, 4096,
                           ((size_t)(2 * L * L + 8) * L + 296)
                               * sizeof(double)) == 0) {
            p->M = (double *)p->rawM + 296;       /* padded mid volume: +64 B
                                                     per plane, base 2368 B
                                                     off the page           */
            tune(p);
        }
    }
#endif
    if (!p->fn && !dense_setup(p)) {                  /* fallback of last resort */
        free(p->rawP); free(p->rawM); free(p->rawX); free(p->w); free(p->tmp);
        free(p);
        return NULL;
    }
    return p;
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    if (p->fn) p->fn((const double *)in, (double *)out, p->batch, p->M, p->P);
    else       dense_exec(p, in, out);
}

/* the whole graded m-step chain: state <- (FFT(state)+c)/(1+|FFT(state)+c|).
 *
 * Fused path: the state lives in `out` for the whole chain (the engine
 * consumes each input volume completely -- into the plane scratch, then into
 * the mid volume M -- before writing the matching output volume, so
 * cur == dst is safe per volume).  One state buffer instead of a ping-pong
 * pair keeps the chain working set at state+M+c volumes, which matters at
 * L=100 where that is already the size of the node's L3.
 *
 * Fallback path (no fused kernel): fft3d_execute needs distinct buffers, so
 * steps alternate between `out` and the plan's X, ending in `out`. */
void fft3d_chain(fft3d_plan *p, const double _Complex *x0,
                 const double _Complex *c, double _Complex *out, int m)
{
    const size_t VDv = (size_t)2 * p->L * p->L * p->L;   /* doubles/volume  */
    const size_t VD  = VDv * (size_t)p->batch;
    if (m < 1) { memmove(out, x0, VD * sizeof(double)); return; }
#ifdef __AVX512F__
    if (p->use_soa) {                 /* SoA-8 lane engine owns the chain
                                       * (group-major from birth: all m
                                       * steps per 8-volume group)          */
        soa_chain_n(p, x0, c, out, m, p->batch);
        return;
    }
    /* gen_r4: VOLUME-MAJOR schedule for the interleaved families (ADOPTED
     * from gen_pfa_large gen_r4, who took it from gen_dense_prime /
     * gen_rader / gen_layout -- the corpus-consensus chain shape).  Volumes
     * are independent in the chain algebra, so each volume runs ALL m steps
     * while its state + c slice stays resident (0.5 MB at L=25 -- inside
     * L2 -- to 4 MB at L=50 B=4), instead of every step churning the whole
     * batch working set.  Per-volume FFT op order is unchanged; at 50/100
     * (VDv % 8 == 0) outputs are bit-identical to the r3 step-major
     * schedule, at 25/27 the per-volume map_span tail moves 1-3 complex
     * from the vector ladder to the exact scalar map (gate-checked as
     * always; the graded 25/27 picks are soa and unaffected). */
    if (p->cfn && p->dm) {
        /* deferred-map schedule (gen_r3 ipm / gen_r4 ipp): between steps
         * the state volume holds the RAW FFT output z' = FFT3(x_s); the map
         * z/(1+|z|) with z = z' + c is applied by the NEXT step's phase 1
         * (at its loads for ipm, in a per-plane prepass for ipp), and once
         * by map_span after the last step.  m FFT passes + m map
         * applications, exactly the chain's algebra, zero separate map
         * passes in steady state. */
        for (int b = 0; b < p->batch; ++b) {
            const double *xb = (const double *)x0 + (size_t)b * VDv;
            const double *cb = (const double *)c  + (size_t)b * VDv;
            double       *ob = (double *)out      + (size_t)b * VDv;
            p->fn(xb, ob, 1, p->M, p->P);
            for (int s = 1; s < m; ++s)
                p->cfn(ob, ob, cb, 1, p->M, p->P);
            map_span(ob, cb, ob, VDv);
        }
        return;
    }
#endif
    if (p->cfn) {
        for (int b = 0; b < p->batch; ++b) {
            const double *xb = (const double *)x0 + (size_t)b * VDv;
            const double *cb = (const double *)c  + (size_t)b * VDv;
            double       *ob = (double *)out      + (size_t)b * VDv;
            const double *cur = xb;
            for (int s = 0; s < m; ++s) {
                p->cfn(cur, ob, cb, 1, p->M, p->P);
                cur = ob;
            }
        }
        return;
    }
    const double _Complex *cur = x0;
    for (int s = 0; s < m; ++s) {
        double _Complex *dst = (((m - 1 - s) & 1) == 0) ? out
                             : (double _Complex *)p->X;
        fft3d_execute(p, cur, dst);
        map_scalar((const double *)dst, (const double *)c, (double *)dst,
                   VD / 2);
        cur = dst;
    }
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    if (p->soa_live) gl_arena_destroy(&p->soa_ar);
    free(p->rawP);
    free(p->rawM);
    free(p->rawX);
    free(p->w);
    free(p->tmp);
    free(p);
}

#else /* ================ ENGINE TEMPLATE, one size (GENL) ================== */

#define NPL   (GENL * GENL)
#define PLNDL (2 * GENL * GENL)          /* doubles per x-plane              */
#define MPLND (PLNDL + 8)                /* mid volume plane pitch: +64 B ->
                                            an ODD cache-line count, no fixed
                                            mod-4096 relation to in/out      */
#define VDL   ((size_t)2 * GENL * NPL)   /* doubles per volume               */
#define NFULL (GENL / 4)                 /* full 4-lane groups per subpass   */
#define NYG   (NFULL + (GENL % 4 ? 1 : 0))
#define NTFL  (NPL / 4)                  /* full flat tiles in phase 2       */
#define FN(n) GCAT(n, GCAT(_, GENL))

/* pre-RA scheduling with register-pressure awareness helps the L=25 family
 * only (measured on the node: 25: -5%, 27: +2%, 100: +48%(!) when applied
 * globally) -- gen_batchlane's SCHED15 trick, same mechanism: the 5x5 CT
 * body holds 25+ live vectors and spills; pressure-aware scheduling cuts
 * the spill traffic where the codelet is pressure-bound. */
#if GENL == 25
#define PWPOPT __attribute__((optimize("schedule-insns", "sched-pressure")))
#else
#define PWPOPT
#endif

/* phase 1, ONE x-plane: z transform (lanes = 4 y-rows, granule transposes)
 * into plane scratch pl[y][kz], then y transform (lanes = 4 kz, contiguous)
 * into the mid volume M at the padded pitch.  GENL % 4 != 0 (25, 27, 50)
 * adds one OVERLAPPING group per subpass (recompute of 4 - GENL%4 lanes;
 * every store is idempotent).  gen_r4: the body is factored over (px, mx)
 * plane bases so the ipp family can feed it a mapped scratch plane
 * (gen_pfa_large gen_r4's shape). */
static inline __attribute__((always_inline))
void FN(p1body)(const double *restrict px, double *restrict mx,
                double *restrict pld)
{
    for (int yg = 0; yg < NYG; ++yg) {
        const int yb = (yg == NFULL) ? (GENL - 4) : 4 * yg;
        /* one runtime base per block (the L45 r7 single-base fix) */
        const double *rows = px  + (size_t)yb * (2 * GENL);
        double       *prow = pld + (size_t)yb * (2 * GPP);
        vec Zv[GENL], Wv[GENL];
        _Pragma("GCC unroll 25")
        for (int zg = 0; zg < NFULL; ++zg) {
            vec r_[4];
            _Pragma("GCC unroll 4")
            for (int j = 0; j < 4; ++j)
                r_[j] = LDU(rows + (size_t)j * (2 * GENL) + 8 * zg);
            TRNC(r_, &Zv[4 * zg]);
        }
#if GENL % 4
        {   /* overlapping last z-granule: columns GENL-4 .. GENL-1 */
            vec r_[4];
            _Pragma("GCC unroll 4")
            for (int j = 0; j < 4; ++j)
                r_[j] = LDU(rows + (size_t)j * (2 * GENL) + 2 * (GENL - 4));
            TRNC(r_, &Zv[GENL - 4]);
        }
#endif
#define LD1(n)    Zv[n]
#define ST1(k, v) (Wv[k] = (v))
        PFAL(LD1, ST1);
#undef LD1
#undef ST1
        _Pragma("GCC unroll 25")
        for (int zg = 0; zg < NFULL; ++zg) {
            vec r_[4];
            TRNC(&Wv[4 * zg], r_);
            _Pragma("GCC unroll 4")
            for (int j = 0; j < 4; ++j)
                STU(prow + (size_t)j * (2 * GPP) + 8 * zg, r_[j]);
        }
#if GENL % 4
        {   vec r_[4];
            TRNC(&Wv[GENL - 4], r_);
            _Pragma("GCC unroll 4")
            for (int j = 0; j < 4; ++j)
                STU(prow + (size_t)j * (2 * GPP) + 2 * (GENL - 4), r_[j]);
        }
#endif
    }

    for (int zg = 0; zg < NYG; ++zg) {
        const int zb = (zg == NFULL) ? (GENL - 4) : 4 * zg;
        const double *pcol = pld + 2 * zb;
        double       *mcol = mx  + 2 * zb;
        /* opaque-base barrier: stops gcc hoisting+spilling GENL row leas
         * (the L45 r6 offset-table pathology) */
        __asm__("" : "+r"(pcol), "+r"(mcol));
#define LD2(n)    LDU(pcol + (size_t)(n) * (2 * GPP))
#define ST2(k, v) STU(mcol + (size_t)(k) * (2 * GENL), (v))
        PFAL(LD2, ST2);
#undef LD2
#undef ST2
    }
}

static inline __attribute__((always_inline))
void FN(p1)(const double *restrict in, double *restrict mid,
            double *restrict pld, int x, const long mpln)
{
    FN(p1body)(in + (size_t)x * PLNDL, mid + (size_t)x * mpln, pld);
}

/* p1 with the graded map APPLIED TO THE Z-SUBPASS LOADS (gen_r3, the ipm
 * deferred-map family, ADOPTED from gen_pfa_large gen_r3): the state
 * entering this step is the PREVIOUS step's raw FFT output z' (no +c, no
 * map); every element is loaded here exactly once, so v = map(z' + c)
 * right after the granule load IS the chain map, in map_span's exact op
 * order (bit-identical results).  The ladder latency sits behind loads
 * feeding compute -- thousands of independent chains per plane the OoO
 * window runs ahead of -- NOT gating stores in the miss-bound x-pass (the
 * measured ipf failure at L=100).  Deletes the separate map pass entirely.
 * Overlap groups at GENL % 4 != 0 recompute the map on the repeated lanes:
 * pure function, idempotent. */
static inline __attribute__((always_inline))
void FN(p1m)(const double *restrict in, double *restrict mid,
             double *restrict pld, int x, const long mpln,
             const double *restrict cf)
{
    const double *px = in  + (size_t)x * PLNDL;
    const double *pc = cf  + (size_t)x * PLNDL;
    double       *mx = mid + (size_t)x * mpln;

    for (int yg = 0; yg < NYG; ++yg) {
        const int yb = (yg == NFULL) ? (GENL - 4) : 4 * yg;
        const double *rows = px  + (size_t)yb * (2 * GENL);
        const double *crow = pc  + (size_t)yb * (2 * GENL);
        double       *prow = pld + (size_t)yb * (2 * GPP);
        vec Zv[GENL], Wv[GENL];
        _Pragma("GCC unroll 25")
        for (int zg = 0; zg < NFULL; ++zg) {
            vec r_[4];
            _Pragma("GCC unroll 4")
            for (int j = 0; j < 4; ++j)
                r_[j] = map_step_v(LDU(rows + (size_t)j * (2 * GENL) + 8 * zg)
                                 + LDU(crow + (size_t)j * (2 * GENL) + 8 * zg));
            TRNC(r_, &Zv[4 * zg]);
        }
#if GENL % 4
        {   /* overlapping last z-granule: columns GENL-4 .. GENL-1 */
            vec r_[4];
            _Pragma("GCC unroll 4")
            for (int j = 0; j < 4; ++j)
                r_[j] = map_step_v(
                    LDU(rows + (size_t)j * (2 * GENL) + 2 * (GENL - 4))
                  + LDU(crow + (size_t)j * (2 * GENL) + 2 * (GENL - 4)));
            TRNC(r_, &Zv[GENL - 4]);
        }
#endif
#define LD1(n)    Zv[n]
#define ST1(k, v) (Wv[k] = (v))
        PFAL(LD1, ST1);
#undef LD1
#undef ST1
        _Pragma("GCC unroll 25")
        for (int zg = 0; zg < NFULL; ++zg) {
            vec r_[4];
            TRNC(&Wv[4 * zg], r_);
            _Pragma("GCC unroll 4")
            for (int j = 0; j < 4; ++j)
                STU(prow + (size_t)j * (2 * GPP) + 8 * zg, r_[j]);
        }
#if GENL % 4
        {   vec r_[4];
            TRNC(&Wv[GENL - 4], r_);
            _Pragma("GCC unroll 4")
            for (int j = 0; j < 4; ++j)
                STU(prow + (size_t)j * (2 * GPP) + 2 * (GENL - 4), r_[j]);
        }
#endif
    }

    for (int zg = 0; zg < NYG; ++zg) {
        const int zb = (zg == NFULL) ? (GENL - 4) : 4 * zg;
        const double *pcol = pld + 2 * zb;
        double       *mcol = mx  + 2 * zb;
        __asm__("" : "+r"(pcol), "+r"(mcol));
#define LD2(n)    LDU(pcol + (size_t)(n) * (2 * GPP))
#define ST2(k, v) STU(mcol + (size_t)(k) * (2 * GENL), (v))
        PFAL(LD2, ST2);
#undef LD2
#undef ST2
    }
}

/* phase 2: x transform OUT of place, mid (padded pitch) -> out (contract
 * layout), tiled over the FLAT (y,z) index.  NPL % 4 != 0 (25, 27) adds one
 * OVERLAPPING tail tile at flat base NPL-4 -- idempotent because the source
 * M is not written here.  pf=1 pokes the GENL read streams one cache line
 * ahead (more streams than the L2 prefetcher tracks). */
static inline __attribute__((always_inline))
void FN(p2o)(const double *restrict mid, double *restrict dst, const int pf)
{
#define LD3(n)    LDU(s_ + (size_t)(n) * MPLND)
#define ST3(k, v) STU(d_ + (size_t)(k) * PLNDL, (v))
    for (int t = 0; t < NTFL; ++t) {
        const double *s_ = mid + (size_t)t * 8;
        double       *d_ = dst + (size_t)t * 8;
        if (pf) {
            _Pragma("GCC unroll 100")
            for (int n_ = 0; n_ < GENL; ++n_)
                __builtin_prefetch(s_ + (size_t)n_ * MPLND + 8, 0, 3);
        }
        PFAL(LD3, ST3);
    }
#if NPL % 4
    {   const double *s_ = mid + (size_t)(NPL - 4) * 2;
        double       *d_ = dst + (size_t)(NPL - 4) * 2;
        PFAL(LD3, ST3);
    }
#endif
#undef LD3
#undef ST3
}

/* phase 2 IN PLACE (the ip* chain family and its execute).  The line
 * codelet reads all GENL inputs before its first store, so in-place is safe
 * per tile -- EXCEPT the overlapping tail at NPL % 4 != 0 (25, 27), whose
 * inputs partially coincide with the previous tile's outputs.  Fix: stash
 * the tail tile's GENL input vectors BEFORE any tile runs; the tail then
 * recomputes from pristine inputs and its stores repeat the overlap
 * columns' values exactly (idempotent). */
static inline __attribute__((always_inline))
void FN(p2ip)(double *io, const int pf)
{
#define LD3(n)    LDU(s_ + (size_t)(n) * PLNDL)
#define ST3(k, v) STU(s_ + (size_t)(k) * PLNDL, (v))
#if NPL % 4
    vec tl_[GENL];
    {   const double *s_ = io + (size_t)(NPL - 4) * 2;
        _Pragma("GCC unroll 100")
        for (int n_ = 0; n_ < GENL; ++n_)
            tl_[n_] = LD3(n_);
    }
#endif
    for (int t = 0; t < NTFL; ++t) {
        double *s_ = io + (size_t)t * 8;
        if (pf) {
            _Pragma("GCC unroll 100")
            for (int n_ = 0; n_ < GENL; ++n_)
                __builtin_prefetch(s_ + (size_t)n_ * PLNDL + 8, 1, 3);
        }
        PFAL(LD3, ST3);
    }
#if NPL % 4
    {   double *s_ = io + (size_t)(NPL - 4) * 2;
#define LDT(n) tl_[n]
        PFAL(LDT, ST3);
#undef LDT
    }
#endif
#undef LD3
#undef ST3
}

/* phase 2 IN PLACE with the chain map fused into the stores (NEW variant:
 * in-place working set AND no separate map pass).  Wins when the volume is
 * cache-resident (the map pass is pure extra L2 traffic); expected to lose
 * at L=100 where the fused map doubles the miss-stream count
 * (gen_pfa_large's measured lesson) -- the race decides. */
static inline __attribute__((always_inline))
void FN(p2ipf)(double *io, const double *restrict cf, const int pf)
{
#define LD3(n)    LDU(s_ + (size_t)(n) * PLNDL)
#define ST3(k, v) do {                                                       \
    vec z_ = (v) + LDU(g_ + (size_t)(k) * PLNDL);                            \
    STU(s_ + (size_t)(k) * PLNDL, map_step_v(z_));                           \
} while (0)
#if NPL % 4
    vec tl_[GENL];
    {   const double *s_ = io + (size_t)(NPL - 4) * 2;
        _Pragma("GCC unroll 100")
        for (int n_ = 0; n_ < GENL; ++n_)
            tl_[n_] = LD3(n_);
    }
#endif
    for (int t = 0; t < NTFL; ++t) {
        double       *s_ = io + (size_t)t * 8;
        const double *g_ = cf + (size_t)t * 8;
        if (pf) {
            _Pragma("GCC unroll 100")
            for (int n_ = 0; n_ < GENL; ++n_)
                __builtin_prefetch(s_ + (size_t)n_ * PLNDL + 8, 1, 3);
        }
        PFAL(LD3, ST3);
    }
#if NPL % 4
    {   double       *s_ = io + (size_t)(NPL - 4) * 2;
        const double *g_ = cf + (size_t)(NPL - 4) * 2;
#define LDT(n) tl_[n]
        PFAL(LDT, ST3);
#undef LDT
    }
#endif
#undef LD3
#undef ST3
}

/* phase 2 for the fused chain: x transform OUT of place, mid -> dst, with
 * z = v + c and the map z/(1+|z|) applied at every store -- deletes the
 * driver's separate full-volume map pass (read z + read c + write state).
 * Overlapping tail is idempotent for the same reason as p2o (map reads only
 * v and c, never dst).  pf pokes the mid read streams / dst write streams
 * (RFO) one line ahead; c and the tile bases advance contiguously. */
static inline __attribute__((always_inline))
void FN(p2c)(const double *restrict mid, double *restrict dst,
             const double *restrict cf, const int pfr, const int pfw)
{
#define LD3(n)    LDU(s_ + (size_t)(n) * MPLND)
#define ST3(k, v) do {                                                       \
    vec z_ = (v) + LDU(g_ + (size_t)(k) * PLNDL);                            \
    STU(d_ + (size_t)(k) * PLNDL, map_step_v(z_));                           \
} while (0)
    for (int t = 0; t < NTFL; ++t) {
        const double *s_ = mid + (size_t)t * 8;
        double       *d_ = dst + (size_t)t * 8;
        const double *g_ = cf  + (size_t)t * 8;
        if (pfr) {
            _Pragma("GCC unroll 100")
            for (int n_ = 0; n_ < GENL; ++n_)
                __builtin_prefetch(s_ + (size_t)n_ * MPLND + 8, 0, 3);
        }
        if (pfw) {
            _Pragma("GCC unroll 100")
            for (int n_ = 0; n_ < GENL; ++n_)
                __builtin_prefetch(d_ + (size_t)n_ * PLNDL + 8, 1, 3);
        }
        PFAL(LD3, ST3);
    }
#if NPL % 4
    {   const double *s_ = mid + (size_t)(NPL - 4) * 2;
        double       *d_ = dst + (size_t)(NPL - 4) * 2;
        const double *g_ = cf  + (size_t)(NPL - 4) * 2;
        PFAL(LD3, ST3);
    }
#endif
#undef LD3
#undef ST3
}

/* fused-through-M chain step for the whole batch: state cur -> state dst.
 * Per volume: phase 1 into the plan's single mid volume M (padded pitch),
 * then the fused map phase 2 into dst. */
static void PWPOPT FN(xc_pf0)(const double *cur, double *dst, const double *cf,
                       long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, M, P, x, MPLND);
        FN(p2c)(M, dst + (size_t)b * VDL, cf + (size_t)b * VDL, 0, 0);
    }
}

#ifndef GENLITE
static void PWPOPT FN(xc_pfr)(const double *cur, double *dst, const double *cf,
                       long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, M, P, x, MPLND);
        FN(p2c)(M, dst + (size_t)b * VDL, cf + (size_t)b * VDL, 1, 0);
    }
}

static void PWPOPT FN(xc_pfrw)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, M, P, x, MPLND);
        FN(p2c)(M, dst + (size_t)b * VDL, cf + (size_t)b * VDL, 1, 1);
    }
}
#endif /* !GENLITE */

/* in-place chain step (BORROWED from gen_pfa_large's ip* family): p1 in
 * place (each plane is fully consumed into the plane scratch before being
 * rewritten; on the first chain step cur != dst and p1 simply reads cur,
 * writes dst), p2 in place, then the sequential vectorized map in place --
 * the state buffer is the ONLY volume-sized object touched besides c. */
static void PWPOPT FN(xc_ip0)(const double *cur, double *dst, const double *cf,
                       long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, d, P, x, PLNDL);
        FN(p2ip)(d, 0);
        map_span(d, cf + (size_t)b * VDL, d, VDL);
    }
}

#ifndef GENLITE
static void PWPOPT FN(xc_ip1)(const double *cur, double *dst, const double *cf,
                       long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, d, P, x, PLNDL);
        FN(p2ip)(d, 1);
        map_span(d, cf + (size_t)b * VDL, d, VDL);
    }
}
#endif /* !GENLITE */

/* deferred-map chain step (gen_r3, ipm family, ADOPTED from gen_pfa_large
 * gen_r3): the incoming state is the previous step's RAW FFT output z';
 * p1m applies map(z' + c) at its loads, p2 runs plain, and NO map pass
 * follows -- the volume written is again a raw FFT output, mapped by the
 * NEXT step's p1m (or by fft3d_chain's one trailing map_span after the
 * last step).  In-place safe exactly as ip* (each plane fully consumed
 * into the scratch before being rewritten). */
static void PWPOPT FN(xc_ipm0)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1m)(i, d, P, x, PLNDL, g);
        FN(p2ip)(d, 0);
    }
}

#ifndef GENLITE
static void PWPOPT FN(xc_ipm1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1m)(i, d, P, x, PLNDL, g);
        FN(p2ip)(d, 1);
    }
}
#endif /* !GENLITE */

/* plane-prepass deferred-map chain step (gen_r4, the ipp family, ADOPTED
 * from gen_pfa_large gen_r4): the ipm schedule (state holds the raw FFT
 * output between steps; fft3d_chain runs step 1 plain and one trailing
 * map_span) but the map runs as map_span's perfectly sequential per-plane
 * PREPASS into the plan's M base -- an L2-resident scratch plane (10-250 KB
 * at these sizes) that p1's z-subpass then consumes.  Their measured
 * rationale at L=100: ipm and ipp have IDENTICAL per-step traffic
 * accounting, and ipm still lost 11-16% to ip -- so ipm's loss is the map
 * ladder's port/latency footprint INSIDE the granule-load stream, not
 * traffic; ipp keeps the traffic cut and pays the interference only at the
 * plane seam (~plane-sized granularity, invisible to the OoO window).
 * In-place safe: the prepass copies the plane out of the state before
 * p1body overwrites it. */
static void PWPOPT FN(xc_ipp0)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x) {
            map_span(i + (size_t)x * PLNDL, g + (size_t)x * PLNDL, M, PLNDL);
            FN(p1body)(M, d + (size_t)x * PLNDL, P);
        }
        FN(p2ip)(d, 0);
    }
}

#ifndef GENLITE
static void PWPOPT FN(xc_ipp1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x) {
            map_span(i + (size_t)x * PLNDL, g + (size_t)x * PLNDL, M, PLNDL);
            FN(p1body)(M, d + (size_t)x * PLNDL, P);
        }
        FN(p2ip)(d, 1);
    }
}
#endif /* !GENLITE */

/* ipp with the prepass c stream NTA-prefetched (gen_r5, the ipq family,
 * ADOPTED from gen_pfa_large gen_r5): c never enters L2/L3, so the LLC
 * holds the state volume across the whole step and p2's reads hit L3
 * instead of DRAM.  Wins only where state+c exceed L3 (100: 32 MB;
 * 121/125: 57/62 MB); at 50 B=4 the batch's c is L3-resident reuse across
 * all m steps and bypassing it costs a full c re-read per step -- ranked
 * last, the race arbitrates. */
static void PWPOPT FN(xc_ipq0)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x) {
            map_span_nta(i + (size_t)x * PLNDL, g + (size_t)x * PLNDL,
                         M, PLNDL);
            FN(p1body)(M, d + (size_t)x * PLNDL, P);
        }
        FN(p2ip)(d, 0);
    }
}

#ifndef GENLITE
static void PWPOPT FN(xc_ipq1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x) {
            map_span_nta(i + (size_t)x * PLNDL, g + (size_t)x * PLNDL,
                         M, PLNDL);
            FN(p1body)(M, d + (size_t)x * PLNDL, P);
        }
        FN(p2ip)(d, 1);
    }
}

/* ipp with the prepass c lines CLFLUSHOPT-ed after use (gen_r5, ipk):
 * the architectural-semantics variant of ipq's L3 bypass. */
static void PWPOPT FN(xc_ipk1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x) {
            map_span_cfl(i + (size_t)x * PLNDL, g + (size_t)x * PLNDL,
                         M, PLNDL);
            FN(p1body)(M, d + (size_t)x * PLNDL, P);
        }
        FN(p2ip)(d, 1);
    }
}

/* ip1 with the NTA-c map pass (gen_r5, iqn): the same c bypass for the
 * non-deferred schedule (hosts/sizes where ip beats ipp). */
static void PWPOPT FN(xc_iqn1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, d, P, x, PLNDL);
        FN(p2ip)(d, 1);
        map_span_nta(d, cf + (size_t)b * VDL, d, VDL);
    }
}
#endif /* !GENLITE */

/* in-place chain step with the map fused into phase 2's stores (NEW):
 * no mid volume, no separate map pass. */
static void PWPOPT FN(xc_ipf)(const double *cur, double *dst, const double *cf,
                       long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, d, P, x, PLNDL);
        FN(p2ipf)(d, cf + (size_t)b * VDL, 0);
    }
}

static void PWPOPT FN(x_pf0)(const double *in, double *out, long nvol,
                      double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDL;
        double       *o = out + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, M, P, x, MPLND);
        FN(p2o)(M, o, 0);
    }
}

#ifndef GENLITE
static void PWPOPT FN(x_pf1)(const double *in, double *out, long nvol,
                      double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDL;
        double       *o = out + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, M, P, x, MPLND);
        FN(p2o)(M, o, 1);
    }
}
#endif /* !GENLITE */

/* out-of-place execute via the in-place phase 2: p1 in -> out, p2 in place
 * in out (rides with the ip* chain candidates) */
static void PWPOPT FN(x_ip0)(const double *in, double *out, long nvol,
                      double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDL;
        double       *o = out + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, o, P, x, PLNDL);
        FN(p2ip)(o, 0);
    }
}

#ifndef GENLITE
static void PWPOPT FN(x_ip1)(const double *in, double *out, long nvol,
                      double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDL;
        double       *o = out + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, o, P, x, PLNDL);
        FN(p2ip)(o, 1);
    }
}
#endif /* !GENLITE */

#undef PWPOPT
#undef FN
#undef NTFL
#undef NYG
#undef NFULL
#undef VDL
#undef MPLND
#undef PLNDL
#undef NPL

#endif /* template */
