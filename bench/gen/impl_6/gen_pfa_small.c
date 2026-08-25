/* gen_pfa_small.c -- PFA of coprime pairs, small: L = 10, 12, 15, 20,
 * plus (round 3, class duty) any coprime P*Q with modules in
 * {2,3,4,5,7,8,9}: 6, 14, 18, 21, 24, 28, 35, 36, 45, 56, 63.
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
           "in 14..127 except 50/80/100 (pfa_large/powp cells)";
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
        v8 T##v1r = S51 * T##sar + S52 * T##sbr;                              \
        v8 T##v1i = S51 * T##sai + S52 * T##sbi;                              \
        v8 T##v2r = S52 * T##sar - S51 * T##sbr;                              \
        v8 T##v2i = S52 * T##sai - S51 * T##sbi;

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
    XDECL10;
    X2L(p, st, 0, 5); X2L(p, st, 2, 7); X2L(p, st, 4, 9);
    X2L(p, st, 6, 1); X2L(p, st, 8, 3);
    X5ST(a_, p, st, 0, 2, 4, 6, 8,  0, 6, 2, 8, 4);
    X5ST(b_, p, st, 5, 7, 9, 1, 3,  5, 1, 7, 3, 9);
}
AIN void dft10_ipm(double *restrict p, const ptrdiff_t st,
                   const double *restrict cp)
{
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
    D3S(p, st, 0, 5, 10);  D3S(p, st, 3, 8, 13); D3S(p, st, 6, 11, 1);
    D3S(p, st, 9, 14, 4);  D3S(p, st, 12, 2, 7);
    D5S(p, st, 0, 3, 6, 9, 12,  0, 6, 12, 3, 9);
    D5X2S(p, st, 5, 8, 11, 14, 2,   10, 1, 7, 13, 4,
                 10, 13, 1, 4, 7,    5, 11, 2, 8, 14);
}
AIN void dft15_ipm(double *restrict p, const ptrdiff_t st,
                   const double *restrict cp)
{
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

DEF_ENGINE(10, 130, SCHED1012, dft10_ip, dft10_ipm)
DEF_ENGINE(12, 162, SCHED1012, dft12_ip, dft12_ipm)
DEF_ENGINE(15, 226, SCHED15A,  DFT15_SWEEP, dft15_ipm)
DEF_ENGINE(20, 418, ,          dft20_ip, dft20_ipm)

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
        v8 o1r_ = S51 * cr_ + S52 * dr_, o1i_ = S51 * ci_ + S52 * di_;\
        v8 o2r_ = S52 * cr_ - S51 * dr_, o2i_ = S52 * ci_ - S51 * di_;\
        r1 = e1r_ + o1i_; i1 = e1i_ - o1r_;                           \
        r4 = e1r_ - o1i_; i4 = e1i_ + o1r_;                           \
        r2 = e2r_ + o2i_; i2 = e2i_ - o2r_;                           \
        r3 = e2r_ - o2i_; i3 = e2i_ + o2r_;                           \
    } while (0)

static inline void pencil10(const double *sr, const double *si,
                            double *dr, double *di, ptrdiff_t s)
{
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
        gmod(xr, xi, P, g->csP, g->snP);
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

#define GP_DEF(Pv, Qv)                                                        \
static void gpencil_##Pv##_##Qv(double *restrict p, ptrdiff_t st,             \
                                const double *restrict cp,                    \
                                const struct gtabs *g)                        \
{ gpencil_body(p, st, cp, g, Pv, Qv, (Qv) % (Pv) == 1); }

GP_DEF(2, 3) GP_DEF(2, 7) GP_DEF(2, 9) GP_DEF(3, 7) GP_DEF(3, 8)
GP_DEF(4, 7) GP_DEF(4, 9) GP_DEF(5, 7) GP_DEF(5, 9) GP_DEF(7, 8)
GP_DEF(7, 9)
/* r6 widening: modules 11/13/15/16/21/25/27 */
GP_DEF(2, 11) GP_DEF(2, 13) GP_DEF(2, 15) GP_DEF(2, 21) GP_DEF(2, 27)
GP_DEF(3, 11) GP_DEF(3, 13) GP_DEF(3, 16) GP_DEF(3, 25)
GP_DEF(4, 11) GP_DEF(4, 13) GP_DEF(4, 15) GP_DEF(4, 21) GP_DEF(4, 27)
GP_DEF(5, 11) GP_DEF(5, 13)
GP_DEF(7, 11) GP_DEF(7, 13) GP_DEF(7, 15) GP_DEF(7, 16)
GP_DEF(8, 9)  GP_DEF(8, 11) GP_DEF(8, 13) GP_DEF(8, 15)
GP_DEF(9, 11) GP_DEF(9, 13)
/* r6 nested-PFA modules as Q: 66,70,78,90,110,126 = 2 x {33,35,39,45,55,63} */
GP_DEF(2, 33) GP_DEF(2, 35) GP_DEF(2, 39) GP_DEF(2, 45) GP_DEF(2, 55)
GP_DEF(2, 63)
/* r6 prime modules 17..31 (dense fold, h <= 15) + nested 51/57:
 * 34,38,46,51,57,58,62,68,69,76,85,87,92,93,95,102,114,115,116,119,124 */
GP_DEF(2, 17) GP_DEF(2, 19) GP_DEF(2, 23) GP_DEF(2, 29) GP_DEF(2, 31)
GP_DEF(2, 51) GP_DEF(2, 57)
GP_DEF(3, 17) GP_DEF(3, 19) GP_DEF(3, 23) GP_DEF(3, 29) GP_DEF(3, 31)
GP_DEF(4, 17) GP_DEF(4, 19) GP_DEF(4, 23) GP_DEF(4, 29) GP_DEF(4, 31)
GP_DEF(5, 17) GP_DEF(5, 19) GP_DEF(5, 23)
GP_DEF(7, 17)

static gpen_fn gpen_lookup(int P, int Q)
{
#define GP_CASE(Pv, Qv) case (Pv) * 128 + (Qv): return gpencil_##Pv##_##Qv;
    switch (P * 128 + Q) {
    GP_CASE(2, 3)  GP_CASE(2, 7)  GP_CASE(2, 9)  GP_CASE(3, 7)  GP_CASE(3, 8)
    GP_CASE(4, 7)  GP_CASE(4, 9)  GP_CASE(5, 7)  GP_CASE(5, 9)  GP_CASE(7, 8)
    GP_CASE(7, 9)
    GP_CASE(2, 11) GP_CASE(2, 13) GP_CASE(2, 15) GP_CASE(2, 21) GP_CASE(2, 27)
    GP_CASE(3, 11) GP_CASE(3, 13) GP_CASE(3, 16) GP_CASE(3, 25)
    GP_CASE(4, 11) GP_CASE(4, 13) GP_CASE(4, 15) GP_CASE(4, 21) GP_CASE(4, 27)
    GP_CASE(5, 11) GP_CASE(5, 13)
    GP_CASE(7, 11) GP_CASE(7, 13) GP_CASE(7, 15) GP_CASE(7, 16)
    GP_CASE(8, 9)  GP_CASE(8, 11) GP_CASE(8, 13) GP_CASE(8, 15)
    GP_CASE(9, 11) GP_CASE(9, 13)
    GP_CASE(2, 33) GP_CASE(2, 35) GP_CASE(2, 39) GP_CASE(2, 45) GP_CASE(2, 55)
    GP_CASE(2, 63)
    GP_CASE(2, 17) GP_CASE(2, 19) GP_CASE(2, 23) GP_CASE(2, 29) GP_CASE(2, 31)
    GP_CASE(2, 51) GP_CASE(2, 57)
    GP_CASE(3, 17) GP_CASE(3, 19) GP_CASE(3, 23) GP_CASE(3, 29) GP_CASE(3, 31)
    GP_CASE(4, 17) GP_CASE(4, 19) GP_CASE(4, 23) GP_CASE(4, 29) GP_CASE(4, 31)
    GP_CASE(5, 17) GP_CASE(5, 19) GP_CASE(5, 23)
    GP_CASE(7, 17)
    }
#undef GP_CASE
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
    /* r6 widening (50/80/100 deliberately absent: pfa_large/powp cells) */
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
    void (*soa_fft)(double *);
    void (*soa_step)(double *, const double *);
    void (*soa_chain)(double *, const double *, int);
    void (*split_fft)(double *, double *, double *, double *);
    struct gtabs gt;                     /* generic coprime-pair tables    */
    gpen_fn gpen;                        /* specialized (P,Q) pencil       */
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
    /* split per-volume buffers only for the tuned sizes: the generic path
     * never touches them (r6; saves 6*L^3 doubles at the big draws) */
    size_t total = coff + soa + (tuned ? 6 * svol : 0);
    p->arena = arena_alloc(total * sizeof(double), &p->arena_len);
    if (!p->arena) {
        free(p);
        return NULL;
    }
    memset(p->arena, 0, total * sizeof(double));  /* fault in as huge pages */
    p->S = p->arena;
    p->C = p->arena + coff;
    if (tuned) {
        p->Sr = p->C + soa;
        p->Si = p->Sr + svol;
        p->Dr = p->Si + svol;
        p->Di = p->Dr + svol;
        p->Cr = p->Di + svol;
        p->Ci = p->Cr + svol;
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
    if (p->generic) {           /* remainder group: lane-replicated SoA */
        int r = p->batch - g8 * 8;
        if (r > 0) {
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
    for (int v = g8 * 8; v < p->batch; ++v) {
        const size_t off = (size_t)v * vol;
        deinterleave(x0 + off, p->Sr, p->Si, vol);
        deinterleave(c + off, p->Cr, p->Ci, vol);
        double *sr = p->Sr, *si = p->Si, *dr = p->Dr, *di = p->Di;
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
    }
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    munmap(p->arena, p->arena_len);
    free(p);
}
