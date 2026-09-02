/* gen_batchlane -- SoA 8-volumes-per-zmm batch-lane engine for small L at B >= 8.
 *
 * LINEAGE: ice_r7/r8 "bl8" chain inside L8_fusedaxes.c (itself adopted from rivals
 * v5_cb7847fb / 8dc1a96d), generalized from L=8 to the class sizes 10, 12, 15,
 * (r3) 20, (r6) the 7-smooth coprime sizes 14, 21, 28, 35 via a DFT7
 * module and SAFE PLACEMENT (stage-1 store permutation; see the M2IP block),
 * which makes the two-stage PFA hazard-free at ANY coprime split, and (r8)
 * the 11-smooth sizes 22, 33, 44, 55 via a DFT11 module (the surprise-test
 * addendum's named gap: no panel had an 11-point module).
 *
 * THE IDEA
 *   Fill the eight zmm lanes with EIGHT VOLUMES of the batch.  Every 1D DFT of the
 *   3D transform is then elementwise across registers: ZERO shuffles per step,
 *   lane-invariant constants, and the only transposes in the whole graded chain are
 *   the pack/unpack at the two ends (once per m = 256..1000 steps).
 *
 * WHY THESE SIZES NEED NO TWIDDLES
 *   10 = 2 x 5, 12 = 3 x 4 (coprime), 15 = 3 x 5, 20 = 4 x 5: every axis DFT is a
 *   two-stage Good-Thomas PFA.  Input map n = (Qa + Pb) mod L (P*Q = L), output
 *   map k = (c, d) by CRT; both permutations are folded into the compile-time slot
 *   offsets of the codelet loads/stores.  No twiddle table exists anywhere in this
 *   file -- the campaign's twiddle problem is dodged entirely inside this class.
 *   Codelet costs (vector FP instrs, FMA-contracted, per pencil per 8 volumes;
 *   gen_r7 LIFTED DFT5 v-pair saves 2/DFT5 -- see the KPHI/KL5 block):
 *      L=10: 5xDFT2 + 2xDFT5              = 84  (was 88)
 *      L=12: 4xDFT3 + 3xDFT4              = 96
 *      L=15: 5xDFT3 + 3xDFT5              = 156 (was 162)
 *      L=20: 5xDFT4 + 4xDFT5              = 208 (was 216)
 *
 * PENCIL FORMS (r3, revised r5).  10/12 pencils are REGISTER-EXPLICIT single
 *   codelets: stage 1 reads memory and writes NAMED registers, stage 2 reads
 *   registers and stores to memory -- exactly 2L zmm loads + 2L zmm stores per
 *   pencil (r3; the r1/r2 in-place slot form compiled to dead stage-1 stores
 *   and out-lined the sweep pencils).  15 and 20 use the MEMORY form (in-place
 *   slot modules, stage-2 groups load-then-store): register-explicit pays only
 *   when 2L site regs + module temps fit ~32 zmm -- 10 (20 live) and 12 (24)
 *   fit, 15 (30) and 20 (40) spill (gen_pfa_small gen_r4's rule, confirmed
 *   here r5 same-core: mem+div 4.41 vs reg forms 4.42-4.57 at 15).  L=15
 *   stage-2 groups c=1,2 have EQUAL slot sets and are one fused
 *   load-both-store-both codelet (DFT5X2 hazard); L=20 is plain in-place
 *   (Q=5 == 1 mod P=4, disjoint residue classes -- gen_pfa_small r2's rule).
 *   -DBL_MEM15=0/1/2 races register / memory / hybrid at 15 (2 = register
 *   sweeps + memory map x-pass; lost by 0.3% r5, WINS by 0.25% r7 under
 *   the lifted DFT5 -- now the default, see the BL_MEM15 block).
 *
 * STRUCTURE (per group of 8 volumes)
 *   Scratch is split-complex SoA: site vector = 16 doubles (re[8] | im[8]), site
 *   order natural (x*L^2 + y*L + z) with the PLANE STRIDE PADDED so that
 *   plane_bytes == 256 (mod 4096) -- bl8's anti-alias pad, kept: it breaks the
 *   inter-plane residue degeneracy that would stack the x-pass column loads in a
 *   few L1 sets.  PL2 = 130 / 162 / 226 / 418 site-vectors for L = 10/12/15/20.
 *      sweep zy : per x-plane (L1-resident at 10/12/15): z-pencils stride 1
 *                 site, then y-pencils stride L sites, both in place.
 *      pass x   : per (y,z) column, stride PL2 sites (an L2 stream the HW
 *                 prefetcher tracks; no software prefetch -- bl8 measured pf as
 *                 pure loss on this node).
 *   fft3d_execute: pack plane -> sweep zy (fused, plane still L1-hot), pass x,
 *   transpose-unpack to interleaved out.
 *
 * THE CHAIN IS THE SCORE:  fft3d_chain owns the whole m-step graded map
 *      state <- (FFT(state) + c) / (1 + |FFT(state) + c|)
 *   State and the pre-split c live in SoA for the entire chain; both fit L2
 *   together at 10/12/15 (2 x 424 KiB at L=15, node L2 = 1.25 MiB; at L=20 they
 *   exceed it and c streams -- interleaving c INTO the site was measured at
 *   +40%, see the strategy record) and (C - S) == 2048 (mod 4096) de-aliases
 *   the paired column streams (bl8's offset).  The map is EAGER and
 *   in-register: applied to each x-pass output right before its store (bl8/L17
 *   lesson: lazy map loses -- it fronts the next step's critical path).  Map
 *   form: s = wr^2 + wi^2 + 1e-300, vrsqrt14pd seed, TWO quadratic Newtons
 *   (2^-14 -> 2^-27.4 -> 2^-53.8, full double), d = fma(s,y,1) = 1 + |w|, then
 *   PER SIZE (r5 re-race, same-core): 10/12 vrcp14pd + TWO residual Newtons
 *   (no divider op; div +0.9/+2.1% on the register pencils); 15/20 one exact
 *   vdivpd (the memory-form pencils leave the ICL divider idle; div -3.9% at
 *   15; -DBL_MAPDIV/-DBL_MAPRCP force one form for cross-arch races).  Map placement: fused in-register at the
 *   x-pass stores at ALL sizes (r4 raced a per-column map_col epilogue --
 *   spill-free pencil + tiny L1-hot loop -- and fused won every size under
 *   the same-core protocol; BL_EPI<N>=1 keeps the epilogue raceable).
 *
 * BATCH HANDLING
 *   B % 8 == 0 (every scored case: 10:64, 12:64, 15:32, 20:32) runs full
 *   groups.  A remainder group (incl. B = 1) replicates its last volume into
 *   the unused lanes and unpacks only the real ones -- correct for any B >= 1,
 *   paying up to 8x arithmetic on the remainder group only.
 *
 * ISA: one arithmetic path via 64-byte GCC vector extensions (zmm on the node);
 *   only vrsqrt14pd/vrcp14pd are __AVX512F__-guarded (scalar fallback elsewhere).
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "../fft3d_api.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

/* gen_layout layer, adopted r2: THP 2MiB-page arena for the SoA scratch.  The
 * chain re-touches ~420 4K pages of state+c every step at L=15 (dTLB-L1 is 64
 * entries); one huge page each ends that walk. */
#define GEN_LAYOUT_LIB_ONLY
#include "gen_layout.c"

typedef double v8d __attribute__((vector_size(64)));
typedef double v8du __attribute__((vector_size(64), aligned(8), may_alias));
typedef long long v8i __attribute__((vector_size(64)));

#define LD(p)    (*(const v8du *)(const void *)(p))
#define ST(p, v) (*(v8du *)(void *)(p) = (v))
#if defined(__clang__)
#define SH(a, b, ...) __builtin_shufflevector((a), (b), __VA_ARGS__)
#else
#define SH(a, b, ...) __builtin_shuffle((a), (b), (v8i){__VA_ARGS__})
#endif

/* lane primitives + transpose networks, verbatim from L8_fusedaxes.c (bl8) */
#define T1_LO 0,8,2,10,4,12,6,14
#define T1_HI 1,9,3,11,5,13,7,15
#define T2_LO 0,1,2,3,8,9,10,11
#define T2_HI 4,5,6,7,12,13,14,15
#define T3_LO 0,1,4,5,8,9,12,13
#define T3_HI 2,3,6,7,10,11,14,15

#define BF(a, b, LO, HI)                                                      \
    do {                                                                      \
        const v8d bf_ = SH((a), (b), LO);                                     \
        (b) = SH((a), (b), HI);                                               \
        (a) = bf_;                                                            \
    } while (0)

/* 8x8 transpose: in reg x = one volume's 8 doubles (4 complex) of a site run;
 * out reg j = double j of that run across volumes, lane l = volume lanex[l],
 * lanex = 0,1,4,5,2,3,6,7 (self-inverse; composed into the unpack table). */
static inline void trans8(v8d *restrict m)
{
    BF(m[0], m[4], T2_LO, T2_HI);  BF(m[1], m[5], T2_LO, T2_HI);
    BF(m[2], m[6], T2_LO, T2_HI);  BF(m[3], m[7], T2_LO, T2_HI);
    BF(m[0], m[2], T3_LO, T3_HI);  BF(m[1], m[3], T3_LO, T3_HI);
    BF(m[4], m[6], T3_LO, T3_HI);  BF(m[5], m[7], T3_LO, T3_HI);
    BF(m[0], m[1], T1_LO, T1_HI);  BF(m[2], m[3], T1_LO, T1_HI);
    BF(m[4], m[5], T1_LO, T1_HI);  BF(m[6], m[7], T1_LO, T1_HI);
}

/* inverse transpose + complex re-interleave: in r[j]/q[j] = re/im of site j of an
 * 8-site run (lane = volume, lanex order); out = 16 ready-to-store registers, each
 * 4 interleaved complex, destination (volume, half) given by UO_V/UO_H below. */
static inline void untrans_interleave(v8d *restrict r, v8d *restrict q)
{
    BF(r[0], r[1], T3_LO, T3_HI);  BF(r[2], r[3], T3_LO, T3_HI);
    BF(r[4], r[5], T3_LO, T3_HI);  BF(r[6], r[7], T3_LO, T3_HI);
    BF(q[0], q[1], T3_LO, T3_HI);  BF(q[2], q[3], T3_LO, T3_HI);
    BF(q[4], q[5], T3_LO, T3_HI);  BF(q[6], q[7], T3_LO, T3_HI);

    BF(r[0], r[2], T3_LO, T3_HI);  BF(r[1], r[3], T3_LO, T3_HI);
    BF(r[4], r[6], T3_LO, T3_HI);  BF(r[5], r[7], T3_LO, T3_HI);
    BF(q[0], q[2], T3_LO, T3_HI);  BF(q[1], q[3], T3_LO, T3_HI);
    BF(q[4], q[6], T3_LO, T3_HI);  BF(q[5], q[7], T3_LO, T3_HI);

    BF(r[0], q[0], T1_LO, T1_HI);  BF(r[1], q[1], T1_LO, T1_HI);
    BF(r[2], q[2], T1_LO, T1_HI);  BF(r[3], q[3], T1_LO, T1_HI);
    BF(r[4], q[4], T1_LO, T1_HI);  BF(r[5], q[5], T1_LO, T1_HI);
    BF(r[6], q[6], T1_LO, T1_HI);  BF(r[7], q[7], T1_LO, T1_HI);
}

/* untrans_interleave output t (r[0..7] then q[0..7]) -> (volume, half-of-8-sites) */
static const unsigned char UO_V[16] = { 0,4,2,6,0,4,2,6,1,5,3,7,1,5,3,7 };
static const unsigned char UO_H[16] = { 0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1 };

/* ---- pack / unpack between driver AoS (interleaved complex per volume) and the
 * SoA group scratch.  nsites >= 8; tails re-do a full overlapping block (stores are
 * idempotent copies of already-final data, so the overlap is safe). */
static void pack_plane(const double *const vp[8], double *restrict Spl, int nsites)
{
    for (int s = 0; s < nsites; s += 4) {
        if (s + 4 > nsites) s = nsites - 4;
        v8d m[8];
        for (int l = 0; l < 8; ++l) m[l] = LD(vp[l] + 2 * (size_t)s);
        trans8(m);
        double *d = Spl + (size_t)s * 16;
        for (int j = 0; j < 8; ++j)
            ST(d + (size_t)(j >> 1) * 16 + (size_t)(j & 1) * 8, m[j]);
    }
}

static void unpack_plane(double *const vp[8], const double *restrict Spl,
                         int nsites, int nvol)
{
    for (int s = 0; s < nsites; s += 8) {
        if (s + 8 > nsites) s = nsites - 8;
        v8d r[8], q[8];
        const double *b = Spl + (size_t)s * 16;
        for (int j = 0; j < 8; ++j) {
            r[j] = *(const v8d *)(b + (size_t)j * 16);
            q[j] = *(const v8d *)(b + (size_t)j * 16 + 8);
        }
        untrans_interleave(r, q);
        for (int t = 0; t < 16; ++t) {
            const int v = UO_V[t];
            if (v < nvol)
                ST(vp[v] + 2 * (size_t)s + (size_t)UO_H[t] * 8,
                   t < 8 ? r[t] : q[t - 8]);
        }
    }
}

/* ---- the graded map: rsqrt14 + 2 quadratic Newtons for |w|, then rcp14 + 2
 * residual Newtons for 1/(1+|w|) -- NO divider op anywhere (gen_pfa_small r1's
 * ladder; vdivpd zmm is unpipelined ~16 cyc/op and the x-pass issues one per
 * output vector).  eps guards rsqrt(0). */
#define V8C(x) { (x), (x), (x), (x), (x), (x), (x), (x) }

static inline __attribute__((always_inline)) void
map8(v8d *restrict vr, v8d *restrict vi, const double *restrict cp,
     const int usediv)
{
    static const v8d eps = V8C(1e-300), half = V8C(0.5), c15 = V8C(1.5),
                     one = V8C(1.0);
    const v8d wr = *vr + *(const v8d *)cp;
    const v8d wi = *vi + *(const v8d *)(cp + 8);
    v8d s = wr * wr + eps;
    s = wi * wi + s;
#if defined(__AVX512F__)
    v8d y = (v8d)_mm512_rsqrt14_pd((__m512d)s);
    const v8d hs = s * half;
    v8d u = y * y;
    y = y * (c15 - hs * u);
    u = y * y;
    y = y * (c15 - hs * u);
    const v8d d = s * y + one;          /* 1 + |w| */
    v8d t;
    if (usediv) {
        /* one exact vdivpd riding the otherwise-idle ICL divider (~8 cyc/zmm).
         * r3 re-race per size: WINS at L=20 (13.7 vs 14.3 us, the memory-bound
         * x-pass leaves the FMA ports hungrier than the divider), LOSES at
         * L=15 (+10%) and washes at 10/12 on the register pencils. */
        t = one / d;
    } else {
        t = (v8d)_mm512_rcp14_pd((__m512d)d);
        t = t + t * (one - d * t);      /* residual Newton: 2^-14 -> 2^-28 */
        t = t + t * (one - d * t);      /* -> full double, no vdivpd */
    }
#else
    (void)half; (void)c15; (void)usediv;
    v8d d;
    for (int k = 0; k < 8; ++k) d[k] = 1.0 + __builtin_sqrt(s[k]);
    const v8d t = one / d;
#endif
    *vr = wr * t;
    *vi = wi * t;
}

/* per-size map tail: 10/12 = rcp14 ladder (register pencils; re-raced r5
 * same-core: div +0.9%/+2.1%), 15 = vdivpd on the r5 memory-form pencil
 * (same-core: div 4.41 vs rcp 4.59, -3.9% -- the tail verdict is a property
 * of the surrounding codelet, gen_pfa_small r3's lesson now measured on this
 * entry too), 20 = vdivpd (r3/r4).  -DBL_MAPDIV / -DBL_MAPRCP force one form
 * everywhere (cross-arch race knobs). */
#if defined(BL_MAPDIV)
#define MAPTAIL_REG  1
#define MAPTAIL_SW10 1
#define MAPTAIL_SW12 1
#define MAPTAIL_15  1
#define MAPTAIL_20  1
#define MAPTAIL_GEN 1
#elif defined(BL_MAPRCP)
#define MAPTAIL_REG  0
#define MAPTAIL_SW10 0
#define MAPTAIL_SW12 0
#define MAPTAIL_15  0
#define MAPTAIL_20  0
#define MAPTAIL_GEN 0
#else
#define MAPTAIL_REG 0
/* gen_r9: on the factor-SWAPPED register codelets the tail verdict flips to
 * div (10: swap+div 1.122 vs swap+rcp 1.132 vs ship 1.147; 12: swap+div
 * 1.984 vs swap+rcp 2.008, both behind the unswapped ship there) -- the tiny
 * DFT2/DFT3+map stage-2 codelets leave the divider idle while the rcp
 * ladder's 5 extra FMA-port ops now cost (the codelet-local rule, again). */
#define MAPTAIL_SW10 1
#define MAPTAIL_SW12 1
#define MAPTAIL_15  1
#define MAPTAIL_20  1
#define MAPTAIL_GEN 1   /* 14/21/28/35: memory-form pencils, divider idle (15/20's verdict) */
#endif

/* ---- codelet constants (exact to the last bit of double) */
static const v8d KH  = V8C(0.5);
static const v8d KS3 = V8C(0.86602540378443864676);   /* sin(pi/3) */
static const v8d K25 = V8C(0.25);
static const v8d KQ5 = V8C(0.55901699437494742410);   /* sqrt(5)/4 */
static const v8d KS1 = V8C(0.95105651629515357212);   /* sin(2pi/5) */
static const v8d KS2 = V8C(0.58778525229247312917);   /* sin(4pi/5) */
/* gen_r7 LIFTED v-pair constants (lit 08 6.3's op cut, adapted to the
 * Winograd v-pair): KS1/KS2 = sin(2pi/5)/sin(pi/5) = 2cos(pi/5) = PHI, the
 * golden ratio, EXACTLY -- so v1 = KS1*sa + KS2*sb, v2 = KS2*sa - KS1*sb
 * factor through u = sa - PHI*sb as v2 = KS2*u, v1 = KS1*u + KL5*sb with
 * KL5 = (KS1^2+KS2^2)/KS2 = 1.25/sin(pi/5): 6 vector ops instead of 8 per
 * DFT5, same dependency depth (2).  Constants exact to the last bit of
 * double (Decimal series, 50 digits). */
static const v8d KPHI = V8C(1.61803398874989484820);
static const v8d KL5  = V8C(2.12662702088009983045);

/* DFT-7 rotations (gen_r6), exact to the last bit of double (Decimal series,
 * 22 digits): c_j = cos(2pi j/7) signed, s_j = sin(2pi j/7) positive. */
static const v8d KC71 = V8C(0.62348980185873353053);
static const v8d KC72 = V8C(-0.22252093395631440429);
static const v8d KC73 = V8C(-0.90096886790241912624);
static const v8d KS71 = V8C(0.78183148246802980871);
static const v8d KS72 = V8C(0.97492791218182360702);
static const v8d KS73 = V8C(0.43388373911755812048);

/* DFT-11 rotations (gen_r8), exact to the last bit of double (Decimal series,
 * 60 digits, cross-checked vs libm): c_j/s_j = cos/sin(2pi j/11), j=1..5. */
static const v8d KC111 = V8C(0.8412535328311812);
static const v8d KC112 = V8C(0.41541501300188644);
static const v8d KC113 = V8C(-0.14231483827328514);
static const v8d KC114 = V8C(-0.6548607339452851);
static const v8d KC115 = V8C(-0.9594929736144974);
static const v8d KS111 = V8C(0.5406408174555976);
static const v8d KS112 = V8C(0.9096319953545183);
static const v8d KS113 = V8C(0.9898214418809327);
static const v8d KS114 = V8C(0.7557495743542583);
static const v8d KS115 = V8C(0.28173255684142967);

/* site-vector accessors: slot k of a pencil at base p, stride st doubles */
#define PR(p, st, k)  (*(v8d *)((p) + (size_t)(k) * (st)))
#define PI_(p, st, k) (*(v8d *)((p) + (size_t)(k) * (st) + 8))

/* ---- register-explicit pencil modules (r3).  The r1/r2 in-place slot macros
 * round-tripped every slot through L1 between the two PFA stages; the compiled
 * code forwarded the stage-2 LOADS in registers but kept most stage-1 stores
 * as dead stores (asm audit: 27/39/60 zmm stores per pencil where 20/24/30
 * suffice) and out-lined the sweep pencils behind call/ret.  Now stage 1 reads
 * memory and writes NAMED registers xr<k>/xi<k>; stage 2 reads registers and
 * stores straight to memory (map fused there in the chain variant).  Exactly
 * 2L loads + 2L stores per pencil, no store-forward round trip, and the L=15
 * stage-2 alias hazard (groups c=1,2 with equal slot sets) vanishes: stage 2
 * never reads memory. */

#define SLR(p, st, k) LD((p) + (size_t)(k) * (st))
#define SLI(p, st, k) LD((p) + (size_t)(k) * (st) + 8)

/* stage-1 DFT-2 on slots a,b: memory -> registers */
#define R2L(p, st, a, b)                                                       \
    do {                                                                       \
        const v8d t0r = SLR(p, st, a), t0i = SLI(p, st, a);                    \
        const v8d t1r = SLR(p, st, b), t1i = SLI(p, st, b);                    \
        xr##a = t0r + t1r;  xi##a = t0i + t1i;                                 \
        xr##b = t0r - t1r;  xi##b = t0i - t1i;                                 \
    } while (0)

/* stage-1 forward DFT-3 on slots a,b,c: memory -> registers (12 vector instrs) */
#define R3L(p, st, a, b, c)                                                    \
    do {                                                                       \
        const v8d t0r = SLR(p, st, a), t0i = SLI(p, st, a);                    \
        const v8d t1r = SLR(p, st, b), t1i = SLI(p, st, b);                    \
        const v8d t2r = SLR(p, st, c), t2i = SLI(p, st, c);                    \
        const v8d tr = t1r + t2r, ti = t1i + t2i;                              \
        const v8d ur = t1r - t2r, ui = t1i - t2i;                              \
        const v8d hr = t0r - KH * tr, hi = t0i - KH * ti;                      \
        xr##a = t0r + tr;         xi##a = t0i + ti;                            \
        xr##b = hr + KS3 * ui;    xi##b = hi - KS3 * ur;                       \
        xr##c = hr - KS3 * ui;    xi##c = hi + KS3 * ur;                       \
    } while (0)

/* stage-2 forward DFT-4: registers i0..i3 (PFA b-order) -> memory o0..o3 */
#define R4CORE(i0, i1, i2, i3)                                                 \
        const v8d t0r = xr##i0 + xr##i2, t0i = xi##i0 + xi##i2;                \
        const v8d t1r = xr##i0 - xr##i2, t1i = xi##i0 - xi##i2;                \
        const v8d t2r = xr##i1 + xr##i3, t2i = xi##i1 + xi##i3;                \
        const v8d t3r = xr##i1 - xr##i3, t3i = xi##i1 - xi##i3;                \
        const v8d y0r = t0r + t2r, y0i = t0i + t2i;                            \
        const v8d y2r = t0r - t2r, y2i = t0i - t2i;                            \
        const v8d y1r = t1r + t3i, y1i = t1i - t3r;                            \
        const v8d y3r = t1r - t3i, y3i = t1i + t3r;

#define R4ST(p, st, i0, i1, i2, i3, o0, o1, o2, o3)                            \
    do {                                                                       \
        R4CORE(i0, i1, i2, i3)                                                 \
        PR(p, st, o0) = y0r;  PI_(p, st, o0) = y0i;                            \
        PR(p, st, o1) = y1r;  PI_(p, st, o1) = y1i;                            \
        PR(p, st, o2) = y2r;  PI_(p, st, o2) = y2i;                            \
        PR(p, st, o3) = y3r;  PI_(p, st, o3) = y3i;                            \
    } while (0)

#define R4STM(p, st, cp, i0, i1, i2, i3, o0, o1, o2, o3)                       \
    do {                                                                       \
        R4CORE(i0, i1, i2, i3)                                                 \
        v8d zr, zi;                                                            \
        zr = y0r; zi = y0i; map8(&zr, &zi, (cp) + (size_t)(o0) * (st), MAPTAIL_REG);        \
        PR(p, st, o0) = zr;  PI_(p, st, o0) = zi;                              \
        zr = y1r; zi = y1i; map8(&zr, &zi, (cp) + (size_t)(o1) * (st), MAPTAIL_REG);        \
        PR(p, st, o1) = zr;  PI_(p, st, o1) = zi;                              \
        zr = y2r; zi = y2i; map8(&zr, &zi, (cp) + (size_t)(o2) * (st), MAPTAIL_REG);        \
        PR(p, st, o2) = zr;  PI_(p, st, o2) = zi;                              \
        zr = y3r; zi = y3i; map8(&zr, &zi, (cp) + (size_t)(o3) * (st), MAPTAIL_REG);        \
        PR(p, st, o3) = zr;  PI_(p, st, o3) = zi;                              \
    } while (0)

/* forward DFT-5 core: inputs T##x0r.., outputs T##X0r/A1/A2/v1/v2 (34 instrs
 * with the output combines; 32 in the gen_r7 LIFTED form below).  Winograd
 * split: c1,c2 = (-1 +- sqrt5)/4.  BL_LIFT5=0 restores the r1-r6 direct
 * v-pair (cross-arch race knob); outputs differ from the lifted form only
 * in rounding (same exact values), so the knob needs a full gate re-check,
 * not a cmp. */
#ifndef BL_LIFT5
#define BL_LIFT5 1
#endif
#if BL_LIFT5
#define DFT5VPAIR(T)                                                           \
        const v8d T##uur = T##sar - KPHI * T##sbr;                             \
        const v8d T##uui = T##sai - KPHI * T##sbi;                             \
        const v8d T##v2r = KS2 * T##uur, T##v2i = KS2 * T##uui;                \
        const v8d T##v1r = KS1 * T##uur + KL5 * T##sbr;                        \
        const v8d T##v1i = KS1 * T##uui + KL5 * T##sbi;
#else
#define DFT5VPAIR(T)                                                           \
        const v8d T##v1r = KS1 * T##sar + KS2 * T##sbr;                        \
        const v8d T##v1i = KS1 * T##sai + KS2 * T##sbi;                        \
        const v8d T##v2r = KS2 * T##sar - KS1 * T##sbr;                        \
        const v8d T##v2i = KS2 * T##sai - KS1 * T##sbi;
#endif
#define DFT5CORE(T)                                                            \
        const v8d T##tar = T##x1r + T##x4r, T##tai = T##x1i + T##x4i;          \
        const v8d T##tbr = T##x2r + T##x3r, T##tbi = T##x2i + T##x3i;          \
        const v8d T##sar = T##x1r - T##x4r, T##sai = T##x1i - T##x4i;          \
        const v8d T##sbr = T##x2r - T##x3r, T##sbi = T##x2i - T##x3i;          \
        const v8d T##pr = T##tar + T##tbr, T##pi = T##tai + T##tbi;            \
        const v8d T##qr = T##tar - T##tbr, T##qi = T##tai - T##tbi;            \
        const v8d T##X0r = T##x0r + T##pr, T##X0i = T##x0i + T##pi;            \
        const v8d T##fr = T##x0r - K25 * T##pr, T##fi = T##x0i - K25 * T##pi;  \
        const v8d T##A1r = T##fr + KQ5 * T##qr, T##A1i = T##fi + KQ5 * T##qi;  \
        const v8d T##A2r = T##fr - KQ5 * T##qr, T##A2i = T##fi - KQ5 * T##qi;  \
        DFT5VPAIR(T)

/* stage-2 DFT-5 input bind: registers i0..i4 -> the DFT5CORE operand names */
#define R5BIND(T, i0, i1, i2, i3, i4)                                          \
        const v8d T##x0r = xr##i0, T##x0i = xi##i0;                            \
        const v8d T##x1r = xr##i1, T##x1i = xi##i1;                            \
        const v8d T##x2r = xr##i2, T##x2i = xi##i2;                            \
        const v8d T##x3r = xr##i3, T##x3i = xi##i3;                            \
        const v8d T##x4r = xr##i4, T##x4i = xi##i4;

#define DFT5STORE(T, p, st, o0, o1, o2, o3, o4)                                \
        PR(p, st, o0) = T##X0r;              PI_(p, st, o0) = T##X0i;          \
        PR(p, st, o1) = T##A1r + T##v1i;     PI_(p, st, o1) = T##A1i - T##v1r; \
        PR(p, st, o4) = T##A1r - T##v1i;     PI_(p, st, o4) = T##A1i + T##v1r; \
        PR(p, st, o2) = T##A2r + T##v2i;     PI_(p, st, o2) = T##A2i - T##v2r; \
        PR(p, st, o3) = T##A2r - T##v2i;     PI_(p, st, o3) = T##A2i + T##v2r;

#define DFT5STOREM(T, p, st, cp, UD, o0, o1, o2, o3, o4)                       \
    do {                                                                       \
        v8d zr, zi;                                                            \
        zr = T##X0r;           zi = T##X0i;                                    \
        map8(&zr, &zi, (cp) + (size_t)(o0) * (st), UD);                            \
        PR(p, st, o0) = zr;  PI_(p, st, o0) = zi;                              \
        zr = T##A1r + T##v1i;  zi = T##A1i - T##v1r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o1) * (st), UD);                            \
        PR(p, st, o1) = zr;  PI_(p, st, o1) = zi;                              \
        zr = T##A1r - T##v1i;  zi = T##A1i + T##v1r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o4) * (st), UD);                            \
        PR(p, st, o4) = zr;  PI_(p, st, o4) = zi;                              \
        zr = T##A2r + T##v2i;  zi = T##A2i - T##v2r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o2) * (st), UD);                            \
        PR(p, st, o2) = zr;  PI_(p, st, o2) = zi;                              \
        zr = T##A2r - T##v2i;  zi = T##A2i + T##v2r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o3) * (st), UD);                            \
        PR(p, st, o3) = zr;  PI_(p, st, o3) = zi;                              \
    } while (0)

/* ---- r4: per-column map EPILOGUE (BL_EPI* path).  The fused R5STM/R4STM
 * x-pass pencil spills: 30 site regs + ~7 map temps + 10 constants overflow
 * the 32 zmm (asm audit r4: 27 spill stores + 15 spill loads + 12 folded
 * rsp operands per pencil at L=15, 31 rsp touches at L=12; the map-free
 * sweep pencils spill ZERO).  Un-fusing the map into this tiny loop right
 * after the spill-free plain DFT pencil keeps every value L1-hot (the loads
 * store-forward from the pencil stores) and swaps the ~54 spill ops for
 * 2L ld + 2L st of clean 15-way-ILP work.  Per-site map arithmetic is
 * unchanged, so chain outputs stay bit-identical either way. */
#define AIN static inline __attribute__((always_inline))

AIN void map_col(double *restrict p, const ptrdiff_t st,
                 const double *restrict cp, const int n, const int usediv)
{
    for (int k = 0; k < n; ++k) {
        v8d zr = SLR(p, st, k), zi = SLI(p, st, k);
        map8(&zr, &zi, cp + (size_t)k * st, usediv);
        PR(p, st, k) = zr;
        PI_(p, st, k) = zi;
    }
}

/* ---- L=20 modules (r3, size adopted from gen_pfa_small's r2 extension of
 * this engine; slot algebra re-derived, driver-verified).  20 sites = 40 live
 * site-vector registers, too many for the register-explicit form, so L=20
 * keeps the r2 memory round trip between stages: stage 1 = in-place DFT4
 * (read 4 slots, write the same 4), stage 2 = load-5-then-store-5 DFT5.
 * Q = 5 == 1 mod P = 4, so stage-2 groups are mutually disjoint residue
 * classes {m == c mod 4}: fully in-place safe, no fusion needed. */

/* in-place DFT-4 on slots a0..a3: memory -> memory, output c at slot a_c */
#define M4IP(p, st, a0, a1, a2, a3)                                            \
    do {                                                                       \
        const v8d x0r = SLR(p, st, a0), x0i = SLI(p, st, a0);                  \
        const v8d x1r = SLR(p, st, a1), x1i = SLI(p, st, a1);                  \
        const v8d x2r = SLR(p, st, a2), x2i = SLI(p, st, a2);                  \
        const v8d x3r = SLR(p, st, a3), x3i = SLI(p, st, a3);                  \
        const v8d t0r = x0r + x2r, t0i = x0i + x2i;                            \
        const v8d t1r = x0r - x2r, t1i = x0i - x2i;                            \
        const v8d t2r = x1r + x3r, t2i = x1i + x3i;                            \
        const v8d t3r = x1r - x3r, t3i = x1i - x3i;                            \
        PR(p, st, a0) = t0r + t2r;  PI_(p, st, a0) = t0i + t2i;                \
        PR(p, st, a1) = t1r + t3i;  PI_(p, st, a1) = t1i - t3r;                \
        PR(p, st, a2) = t0r - t2r;  PI_(p, st, a2) = t0i - t2i;                \
        PR(p, st, a3) = t1r - t3i;  PI_(p, st, a3) = t1i + t3r;                \
    } while (0)

/* DFT-5 input bind from memory slots (stage 2 at L=20) */
#define M5BIND(T, p, st, i0, i1, i2, i3, i4)                                   \
        const v8d T##x0r = SLR(p, st, i0), T##x0i = SLI(p, st, i0);            \
        const v8d T##x1r = SLR(p, st, i1), T##x1i = SLI(p, st, i1);            \
        const v8d T##x2r = SLR(p, st, i2), T##x2i = SLI(p, st, i2);            \
        const v8d T##x3r = SLR(p, st, i3), T##x3i = SLI(p, st, i3);            \
        const v8d T##x4r = SLR(p, st, i4), T##x4i = SLI(p, st, i4);

/* stage-2 DFT-5: registers i0..i4 -> memory o0..o4 (map variant fuses map8) */
#define R5ST(T, p, st, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4)                 \
    do {                                                                       \
        R5BIND(T, i0, i1, i2, i3, i4)                                          \
        DFT5CORE(T)                                                            \
        DFT5STORE(T, p, st, o0, o1, o2, o3, o4)                                \
    } while (0)

#define R5STM(T, p, st, cp, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4)            \
    do {                                                                       \
        R5BIND(T, i0, i1, i2, i3, i4)                                          \
        DFT5CORE(T)                                                            \
        DFT5STOREM(T, p, st, cp, MAPTAIL_REG, o0, o1, o2, o3, o4);             \
    } while (0)

/* memory-bound stage-2 DFT-5 for L=20 (see M4IP block comment) */
#define M5ST(T, p, st, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4)                 \
    do {                                                                       \
        M5BIND(T, p, st, i0, i1, i2, i3, i4)                                   \
        DFT5CORE(T)                                                            \
        DFT5STORE(T, p, st, o0, o1, o2, o3, o4)                                \
    } while (0)

#define M5STMU(T, p, st, cp, UD, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4)       \
    do {                                                                       \
        M5BIND(T, p, st, i0, i1, i2, i3, i4)                                   \
        DFT5CORE(T)                                                            \
        DFT5STOREM(T, p, st, cp, UD, o0, o1, o2, o3, o4);                      \
    } while (0)

#define M5STM(T, p, st, cp, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4)            \
    M5STMU(T, p, st, cp, MAPTAIL_20, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4)

/* ==== gen_r11: the DFT25 = 5x5 COOLEY-TUKEY stages for the L=100 pencil.
 * First twiddles ever in this file -- as COMPILED-IN broadcast constants
 * (lit 11 Tier 1 constant-per-site routing; only 9 distinct w25^m pairs
 * exist), never a table load.  The CT digit reversal cannot be made
 * in-place-safe by the r6 placement alone (stage B reads rows, the outer
 * safe placement wants column-permuted outputs -- a genuine transpose), so
 * stage A stores to a 25-sv L1 scratch and stage B reads it back: the SAME
 * 50-load/50-store count a pure in-place two-stage module would pay, just
 * at a different address, and every hazard is gone.
 *   stage A codelet a2: DFT5 over positions {5a1+a2} of the group, outputs
 *     d1 twiddled by w25^(a2*d1), stored at scratch position a2 + 5*d1;
 *   stage B codelet d1: DFT5 over scratch row {5d1+a2}, output d2 stored to
 *     the group slot of PFA output d = d1 + 5*d2 (safe placement baked in). */
#define M5CTA0(T, p, st, sc, a2, i0, i1, i2, i3, i4)                           \
    do {                                                                       \
        M5BIND(T, p, st, i0, i1, i2, i3, i4)                                   \
        DFT5CORE(T)                                                            \
        DFT5STORE(T, (sc) + (a2) * 16, 80, 0, 1, 2, 3, 4)                      \
    } while (0)

#define M5CTA(T, p, st, sc, a2, i0, i1, i2, i3, i4,                            \
              W1R, W1I, W2R, W2I, W3R, W3I, W4R, W4I)                          \
    do {                                                                       \
        M5BIND(T, p, st, i0, i1, i2, i3, i4)                                   \
        DFT5CORE(T)                                                            \
        double *const scA_ = (sc) + (a2) * 16;                                 \
        PR(scA_, 80, 0) = T##X0r;            PI_(scA_, 80, 0) = T##X0i;        \
        { const v8d t1r_ = T##A1r + T##v1i, t1i_ = T##A1i - T##v1r;            \
          PR(scA_, 80, 1) = t1r_ * (W1R) - t1i_ * (W1I);                       \
          PI_(scA_, 80, 1) = t1r_ * (W1I) + t1i_ * (W1R); }                    \
        { const v8d t4r_ = T##A1r - T##v1i, t4i_ = T##A1i + T##v1r;            \
          PR(scA_, 80, 4) = t4r_ * (W4R) - t4i_ * (W4I);                       \
          PI_(scA_, 80, 4) = t4r_ * (W4I) + t4i_ * (W4R); }                    \
        { const v8d t2r_ = T##A2r + T##v2i, t2i_ = T##A2i - T##v2r;            \
          PR(scA_, 80, 2) = t2r_ * (W2R) - t2i_ * (W2I);                       \
          PI_(scA_, 80, 2) = t2r_ * (W2I) + t2i_ * (W2R); }                    \
        { const v8d t3r_ = T##A2r - T##v2i, t3i_ = T##A2i + T##v2r;            \
          PR(scA_, 80, 3) = t3r_ * (W3R) - t3i_ * (W3I);                       \
          PI_(scA_, 80, 3) = t3r_ * (W3I) + t3i_ * (W3R); }                    \
    } while (0)

#define M5CTB(T, p, st, sc, d1, o0, o1, o2, o3, o4)                            \
    do {                                                                       \
        M5BIND(T, (sc) + (d1) * 80, 16, 0, 1, 2, 3, 4)                         \
        DFT5CORE(T)                                                            \
        DFT5STORE(T, p, st, o0, o1, o2, o3, o4)                                \
    } while (0)

/* stage-B DFT5 with the graded map fused at its stores (L=50: the map rides
 * the wide factor's CT stage B at 5 ladders/codelet -- the winning stage-2
 * ladder width; a DFT2 stage-2 is the r10 L=14 losing shape) */
#define M5CTBM(T, p, st, sc, d1, cp, o0, o1, o2, o3, o4)                       \
    do {                                                                       \
        M5BIND(T, (sc) + (d1) * 80, 16, 0, 1, 2, 3, 4)                         \
        DFT5CORE(T)                                                            \
        DFT5STOREM(T, p, st, cp, MAPTAIL_GEN, o0, o1, o2, o3, o4);             \
    } while (0)

/* ---- L=15 MEMORY-FORM modules (gen_r5).  The r2 in-place slot form, brought
 * back after gen_pfa_small's gen_r4 same-core A/B measured our register-
 * explicit 15 pencil at +12.6% (their memory form 4.44-4.46 vs reg 5.02): 30
 * live site registers + ~14 DFT5CORE temps + map temps spill hard, while the
 * memory form's stage-1 stores ride ICL's 2-stores/cycle at near-zero cost.
 * Their rule, adopted: register-explicit pays only when 2L + module temps
 * <= ~32 zmm (10 and 12 fit; 15 and 20 do not).  Arithmetic order is
 * identical to the register form, so outputs stay bit-identical across the
 * BL_MEM15 knob.  Stage-2 groups c=1,c=2 read/write EQUAL slot sets, so they
 * are one fused load-both-then-store-both codelet (r2's DFT5X2 hazard). */

/* in-place forward DFT-3 on slots i0,i1,i2: memory -> memory (12 instrs) */
#define M3IP(p, st, i0, i1, i2)                                                \
    do {                                                                       \
        const v8d x0r = SLR(p, st, i0), x0i = SLI(p, st, i0);                  \
        const v8d x1r = SLR(p, st, i1), x1i = SLI(p, st, i1);                  \
        const v8d x2r = SLR(p, st, i2), x2i = SLI(p, st, i2);                  \
        const v8d tr = x1r + x2r, ti = x1i + x2i;                              \
        const v8d ur = x1r - x2r, ui = x1i - x2i;                              \
        const v8d hr = x0r - KH * tr, hi = x0i - KH * ti;                      \
        PR(p, st, i0) = x0r + tr;         PI_(p, st, i0) = x0i + ti;           \
        PR(p, st, i1) = hr + KS3 * ui;    PI_(p, st, i1) = hi - KS3 * ur;      \
        PR(p, st, i2) = hr - KS3 * ui;    PI_(p, st, i2) = hi + KS3 * ur;      \
    } while (0)

#define M5X2ST(p, st, i0,i1,i2,i3,i4, o0,o1,o2,o3,o4,                          \
                       j0,j1,j2,j3,j4, w0,w1,w2,w3,w4)                         \
    do {                                                                       \
        M5BIND(b_, p, st, i0, i1, i2, i3, i4)                                  \
        M5BIND(c_, p, st, j0, j1, j2, j3, j4)                                  \
        DFT5CORE(b_)                                                           \
        DFT5CORE(c_)                                                           \
        DFT5STORE(b_, p, st, o0, o1, o2, o3, o4)                               \
        DFT5STORE(c_, p, st, w0, w1, w2, w3, w4)                               \
    } while (0)

#define M5X2STM(p, st, cp, i0,i1,i2,i3,i4, o0,o1,o2,o3,o4,                     \
                           j0,j1,j2,j3,j4, w0,w1,w2,w3,w4)                     \
    do {                                                                       \
        M5BIND(b_, p, st, i0, i1, i2, i3, i4)                                  \
        M5BIND(c_, p, st, j0, j1, j2, j3, j4)                                  \
        DFT5CORE(b_)                                                           \
        DFT5CORE(c_)                                                           \
        DFT5STOREM(b_, p, st, cp, MAPTAIL_15, o0, o1, o2, o3, o4);             \
        DFT5STOREM(c_, p, st, cp, MAPTAIL_15, w0, w1, w2, w3, w4);              \
    } while (0)

AIN void dft15_pencil_mem(double *restrict p, const ptrdiff_t st)
{
    M3IP(p, st, 0, 5, 10);  M3IP(p, st, 3, 8, 13); M3IP(p, st, 6, 11, 1);
    M3IP(p, st, 9, 14, 4);  M3IP(p, st, 12, 2, 7);
    M5ST(a_, p, st, 0, 3, 6, 9, 12,  0, 6, 12, 3, 9);
    M5X2ST(p, st, 5, 8, 11, 14, 2,   10, 1, 7, 13, 4,
                  10, 13, 1, 4, 7,    5, 11, 2, 8, 14);
}

AIN void dft15_pencil_map_mem(double *restrict p, const ptrdiff_t st,
                              const double *restrict cp)
{
    M3IP(p, st, 0, 5, 10);  M3IP(p, st, 3, 8, 13); M3IP(p, st, 6, 11, 1);
    M3IP(p, st, 9, 14, 4);  M3IP(p, st, 12, 2, 7);
    M5STMU(a_, p, st, cp, MAPTAIL_15, 0, 3, 6, 9, 12,  0, 6, 12, 3, 9);
    M5X2STM(p, st, cp, 5, 8, 11, 14, 2,   10, 1, 7, 13, 4,
                       10, 13, 1, 4, 7,    5, 11, 2, 8, 14);
}

/* ==== gen_r6: SAFE PLACEMENT -- two-stage PFA with NO stage-2 hazard at any
 * coprime split.  Stage 1 group b stays in place on slots {(Qa+Pb)%L}, but
 * output c is stored at the slot whose a = (Q^-1 c) mod P, i.e. the slot
 * CONGRUENT TO c mod P.  Stage-2 group c then reads and writes exactly the
 * residue class {m == c mod P} -- read set == write set per group, groups
 * disjoint, plain load-all-then-store-all in-place safe for ANY (P, Q).
 * (Verified against a reference DFT for every size below.)  This retires the
 * r2 DFT5X2 equal-slot-set fusion at L=15 (raced as BL_SAFE15) and is what
 * makes the new 7-smooth sizes 14/21/28/35 mechanical.  The permutation is
 * baked into stage-1 STORE slots: zero extra ops, values unchanged, so
 * outputs are bit-identical to the fused form where both exist. */

/* in-place DFT-2 on slots a,b: memory -> memory (stage 1 at L=14) */
#define M2IP(p, st, a, b)                                                      \
    do {                                                                       \
        const v8d t0r = SLR(p, st, a), t0i = SLI(p, st, a);                    \
        const v8d t1r = SLR(p, st, b), t1i = SLI(p, st, b);                    \
        PR(p, st, a) = t0r + t1r;  PI_(p, st, a) = t0i + t1i;                  \
        PR(p, st, b) = t0r - t1r;  PI_(p, st, b) = t0i - t1i;                  \
    } while (0)

/* in-place DFT-3 with the safe store permutation X0->i0, X1->i2, X2->i1
 * (L=15: Q=5 == 2 mod P=3, Q^-1 = 2, so output c lands at a = 2c mod 3) */
#define M3IPS(p, st, i0, i1, i2)                                               \
    do {                                                                       \
        const v8d x0r = SLR(p, st, i0), x0i = SLI(p, st, i0);                  \
        const v8d x1r = SLR(p, st, i1), x1i = SLI(p, st, i1);                  \
        const v8d x2r = SLR(p, st, i2), x2i = SLI(p, st, i2);                  \
        const v8d tr = x1r + x2r, ti = x1i + x2i;                              \
        const v8d ur = x1r - x2r, ui = x1i - x2i;                              \
        const v8d hr = x0r - KH * tr, hi = x0i - KH * ti;                      \
        PR(p, st, i0) = x0r + tr;         PI_(p, st, i0) = x0i + ti;           \
        PR(p, st, i2) = hr + KS3 * ui;    PI_(p, st, i2) = hi - KS3 * ur;      \
        PR(p, st, i1) = hr - KS3 * ui;    PI_(p, st, i1) = hi + KS3 * ur;      \
    } while (0)

/* in-place DFT-4 with explicit output slots (L=28: Q=7 == 3 mod P=4,
 * Q^-1 = 3, output c at a = 3c mod 4 -> o = (i0, i3, i2, i1)) */
#define M4IPO(p, st, a0, a1, a2, a3, o0, o1, o2, o3)                           \
    do {                                                                       \
        const v8d x0r = SLR(p, st, a0), x0i = SLI(p, st, a0);                  \
        const v8d x1r = SLR(p, st, a1), x1i = SLI(p, st, a1);                  \
        const v8d x2r = SLR(p, st, a2), x2i = SLI(p, st, a2);                  \
        const v8d x3r = SLR(p, st, a3), x3i = SLI(p, st, a3);                  \
        const v8d t0r = x0r + x2r, t0i = x0i + x2i;                            \
        const v8d t1r = x0r - x2r, t1i = x0i - x2i;                            \
        const v8d t2r = x1r + x3r, t2i = x1i + x3i;                            \
        const v8d t3r = x1r - x3r, t3i = x1i - x3i;                            \
        PR(p, st, o0) = t0r + t2r;  PI_(p, st, o0) = t0i + t2i;                \
        PR(p, st, o1) = t1r + t3i;  PI_(p, st, o1) = t1i - t3r;                \
        PR(p, st, o2) = t0r - t2r;  PI_(p, st, o2) = t0i - t2i;                \
        PR(p, st, o3) = t1r - t3i;  PI_(p, st, o3) = t1i + t3r;                \
    } while (0)

/* ---- forward DFT-7 core (gen_r6).  Symmetric/antisymmetric split:
 *   t_j = x_j + x_{7-j}, u_j = x_j - x_{7-j}, j = 1..3;
 *   A_k = x0 + sum_j cos(2pi kj/7) t_j,  B_k = sum_j sin(2pi kj/7) u_j;
 *   X_k = A_k - i B_k, X_{7-k} = A_k + i B_k;
 * exponents kj folded to j' in {1,2,3}: A2 = (c2,c3,c1), A3 = (c3,c1,c2),
 * B2 = (s2,-s3,-s1), B3 = (s3,-s1,s2).  66 vector FP per DFT7 per 8 vols
 * (12 t/u + 6 X0 + 18 A-FMA + 18 B + 12 output combines). */
#define DFT7CORE(T)                                                            \
        const v8d T##t1r = T##x1r + T##x6r, T##t1i = T##x1i + T##x6i;          \
        const v8d T##u1r = T##x1r - T##x6r, T##u1i = T##x1i - T##x6i;          \
        const v8d T##t2r = T##x2r + T##x5r, T##t2i = T##x2i + T##x5i;          \
        const v8d T##u2r = T##x2r - T##x5r, T##u2i = T##x2i - T##x5i;          \
        const v8d T##t3r = T##x3r + T##x4r, T##t3i = T##x3i + T##x4i;          \
        const v8d T##u3r = T##x3r - T##x4r, T##u3i = T##x3i - T##x4i;          \
        const v8d T##X0r = T##x0r + T##t1r + T##t2r + T##t3r;                  \
        const v8d T##X0i = T##x0i + T##t1i + T##t2i + T##t3i;                  \
        const v8d T##A1r = T##x0r + KC71 * T##t1r + KC72 * T##t2r + KC73 * T##t3r; \
        const v8d T##A1i = T##x0i + KC71 * T##t1i + KC72 * T##t2i + KC73 * T##t3i; \
        const v8d T##A2r = T##x0r + KC72 * T##t1r + KC73 * T##t2r + KC71 * T##t3r; \
        const v8d T##A2i = T##x0i + KC72 * T##t1i + KC73 * T##t2i + KC71 * T##t3i; \
        const v8d T##A3r = T##x0r + KC73 * T##t1r + KC71 * T##t2r + KC72 * T##t3r; \
        const v8d T##A3i = T##x0i + KC73 * T##t1i + KC71 * T##t2i + KC72 * T##t3i; \
        const v8d T##B1r = KS71 * T##u1r + KS72 * T##u2r + KS73 * T##u3r;      \
        const v8d T##B1i = KS71 * T##u1i + KS72 * T##u2i + KS73 * T##u3i;      \
        const v8d T##B2r = KS72 * T##u1r - KS73 * T##u2r - KS71 * T##u3r;      \
        const v8d T##B2i = KS72 * T##u1i - KS73 * T##u2i - KS71 * T##u3i;      \
        const v8d T##B3r = KS73 * T##u1r - KS71 * T##u2r + KS72 * T##u3r;      \
        const v8d T##B3i = KS73 * T##u1i - KS71 * T##u2i + KS72 * T##u3i;

#define M7BIND(T, p, st, i0, i1, i2, i3, i4, i5, i6)                           \
        const v8d T##x0r = SLR(p, st, i0), T##x0i = SLI(p, st, i0);            \
        const v8d T##x1r = SLR(p, st, i1), T##x1i = SLI(p, st, i1);            \
        const v8d T##x2r = SLR(p, st, i2), T##x2i = SLI(p, st, i2);            \
        const v8d T##x3r = SLR(p, st, i3), T##x3i = SLI(p, st, i3);            \
        const v8d T##x4r = SLR(p, st, i4), T##x4i = SLI(p, st, i4);            \
        const v8d T##x5r = SLR(p, st, i5), T##x5i = SLI(p, st, i5);            \
        const v8d T##x6r = SLR(p, st, i6), T##x6i = SLI(p, st, i6);

#define DFT7STORE(T, p, st, o0, o1, o2, o3, o4, o5, o6)                        \
        PR(p, st, o0) = T##X0r;              PI_(p, st, o0) = T##X0i;          \
        PR(p, st, o1) = T##A1r + T##B1i;     PI_(p, st, o1) = T##A1i - T##B1r; \
        PR(p, st, o6) = T##A1r - T##B1i;     PI_(p, st, o6) = T##A1i + T##B1r; \
        PR(p, st, o2) = T##A2r + T##B2i;     PI_(p, st, o2) = T##A2i - T##B2r; \
        PR(p, st, o5) = T##A2r - T##B2i;     PI_(p, st, o5) = T##A2i + T##B2r; \
        PR(p, st, o3) = T##A3r + T##B3i;     PI_(p, st, o3) = T##A3i - T##B3r; \
        PR(p, st, o4) = T##A3r - T##B3i;     PI_(p, st, o4) = T##A3i + T##B3r;

#define M7ST(T, p, st, i0, i1, i2, i3, i4, i5, i6, o0, o1, o2, o3, o4, o5, o6) \
    do {                                                                       \
        M7BIND(T, p, st, i0, i1, i2, i3, i4, i5, i6)                           \
        DFT7CORE(T)                                                            \
        DFT7STORE(T, p, st, o0, o1, o2, o3, o4, o5, o6)                        \
    } while (0)

#define DFT7STOREM(T, p, st, cp, UD, o0, o1, o2, o3, o4, o5, o6)               \
    do {                                                                       \
        v8d zr, zi;                                                            \
        zr = T##X0r;           zi = T##X0i;                                    \
        map8(&zr, &zi, (cp) + (size_t)(o0) * (st), UD);                        \
        PR(p, st, o0) = zr;  PI_(p, st, o0) = zi;                              \
        zr = T##A1r + T##B1i;  zi = T##A1i - T##B1r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o1) * (st), UD);                        \
        PR(p, st, o1) = zr;  PI_(p, st, o1) = zi;                              \
        zr = T##A1r - T##B1i;  zi = T##A1i + T##B1r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o6) * (st), UD);                        \
        PR(p, st, o6) = zr;  PI_(p, st, o6) = zi;                              \
        zr = T##A2r + T##B2i;  zi = T##A2i - T##B2r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o2) * (st), UD);                        \
        PR(p, st, o2) = zr;  PI_(p, st, o2) = zi;                              \
        zr = T##A2r - T##B2i;  zi = T##A2i + T##B2r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o5) * (st), UD);                        \
        PR(p, st, o5) = zr;  PI_(p, st, o5) = zi;                              \
        zr = T##A3r + T##B3i;  zi = T##A3i - T##B3r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o3) * (st), UD);                        \
        PR(p, st, o3) = zr;  PI_(p, st, o3) = zi;                              \
        zr = T##A3r - T##B3i;  zi = T##A3i + T##B3r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o4) * (st), UD);                        \
        PR(p, st, o4) = zr;  PI_(p, st, o4) = zi;                              \
    } while (0)

#define M7STM(T, p, st, cp, UD, i0, i1, i2, i3, i4, i5, i6,                    \
                             o0, o1, o2, o3, o4, o5, o6)                       \
    do {                                                                       \
        M7BIND(T, p, st, i0, i1, i2, i3, i4, i5, i6)                           \
        DFT7CORE(T)                                                            \
        DFT7STOREM(T, p, st, cp, UD, o0, o1, o2, o3, o4, o5, o6);              \
    } while (0)

/* ---- forward DFT-11 core (gen_r8).  Same symmetric/antisymmetric split as
 * DFT7, five rotation pairs: t_j = x_j + x_{11-j}, u_j = x_j - x_{11-j},
 * j = 1..5; A_k = x0 + sum_j cos(2pi kj/11) t_j, B_k = sum_j sin(2pi kj/11)
 * u_j; X_k = A_k - i B_k, X_{11-k} = A_k + i B_k.  Exponent folding (kj mod
 * 11, cos even / sin sign-flipped past 5) generated and verified in Python
 * against a reference DFT.  150 vector FP per DFT11 per 8 volumes
 * (20 t/u + 10 X0 + 50 A-FMA + 50 B + 20 output combines). */
#define DFT11CORE(T)                                                           \
        const v8d T##t1r = T##x1r + T##x10r, T##t1i = T##x1i + T##x10i;        \
        const v8d T##u1r = T##x1r - T##x10r, T##u1i = T##x1i - T##x10i;        \
        const v8d T##t2r = T##x2r + T##x9r,  T##t2i = T##x2i + T##x9i;         \
        const v8d T##u2r = T##x2r - T##x9r,  T##u2i = T##x2i - T##x9i;         \
        const v8d T##t3r = T##x3r + T##x8r,  T##t3i = T##x3i + T##x8i;         \
        const v8d T##u3r = T##x3r - T##x8r,  T##u3i = T##x3i - T##x8i;         \
        const v8d T##t4r = T##x4r + T##x7r,  T##t4i = T##x4i + T##x7i;         \
        const v8d T##u4r = T##x4r - T##x7r,  T##u4i = T##x4i - T##x7i;         \
        const v8d T##t5r = T##x5r + T##x6r,  T##t5i = T##x5i + T##x6i;         \
        const v8d T##u5r = T##x5r - T##x6r,  T##u5i = T##x5i - T##x6i;         \
        const v8d T##X0r = T##x0r + T##t1r + T##t2r + T##t3r + T##t4r + T##t5r;\
        const v8d T##X0i = T##x0i + T##t1i + T##t2i + T##t3i + T##t4i + T##t5i;\
        const v8d T##A1r = T##x0r + KC111 * T##t1r + KC112 * T##t2r            \
                         + KC113 * T##t3r + KC114 * T##t4r + KC115 * T##t5r;   \
        const v8d T##A1i = T##x0i + KC111 * T##t1i + KC112 * T##t2i            \
                         + KC113 * T##t3i + KC114 * T##t4i + KC115 * T##t5i;   \
        const v8d T##A2r = T##x0r + KC112 * T##t1r + KC114 * T##t2r            \
                         + KC115 * T##t3r + KC113 * T##t4r + KC111 * T##t5r;   \
        const v8d T##A2i = T##x0i + KC112 * T##t1i + KC114 * T##t2i            \
                         + KC115 * T##t3i + KC113 * T##t4i + KC111 * T##t5i;   \
        const v8d T##A3r = T##x0r + KC113 * T##t1r + KC115 * T##t2r            \
                         + KC112 * T##t3r + KC111 * T##t4r + KC114 * T##t5r;   \
        const v8d T##A3i = T##x0i + KC113 * T##t1i + KC115 * T##t2i            \
                         + KC112 * T##t3i + KC111 * T##t4i + KC114 * T##t5i;   \
        const v8d T##A4r = T##x0r + KC114 * T##t1r + KC113 * T##t2r            \
                         + KC111 * T##t3r + KC115 * T##t4r + KC112 * T##t5r;   \
        const v8d T##A4i = T##x0i + KC114 * T##t1i + KC113 * T##t2i            \
                         + KC111 * T##t3i + KC115 * T##t4i + KC112 * T##t5i;   \
        const v8d T##A5r = T##x0r + KC115 * T##t1r + KC111 * T##t2r            \
                         + KC114 * T##t3r + KC112 * T##t4r + KC113 * T##t5r;   \
        const v8d T##A5i = T##x0i + KC115 * T##t1i + KC111 * T##t2i            \
                         + KC114 * T##t3i + KC112 * T##t4i + KC113 * T##t5i;   \
        const v8d T##B1r = KS111 * T##u1r + KS112 * T##u2r + KS113 * T##u3r    \
                         + KS114 * T##u4r + KS115 * T##u5r;                    \
        const v8d T##B1i = KS111 * T##u1i + KS112 * T##u2i + KS113 * T##u3i    \
                         + KS114 * T##u4i + KS115 * T##u5i;                    \
        const v8d T##B2r = KS112 * T##u1r + KS114 * T##u2r - KS115 * T##u3r    \
                         - KS113 * T##u4r - KS111 * T##u5r;                    \
        const v8d T##B2i = KS112 * T##u1i + KS114 * T##u2i - KS115 * T##u3i    \
                         - KS113 * T##u4i - KS111 * T##u5i;                    \
        const v8d T##B3r = KS113 * T##u1r - KS115 * T##u2r - KS112 * T##u3r    \
                         + KS111 * T##u4r + KS114 * T##u5r;                    \
        const v8d T##B3i = KS113 * T##u1i - KS115 * T##u2i - KS112 * T##u3i    \
                         + KS111 * T##u4i + KS114 * T##u5i;                    \
        const v8d T##B4r = KS114 * T##u1r - KS113 * T##u2r + KS111 * T##u3r    \
                         + KS115 * T##u4r - KS112 * T##u5r;                    \
        const v8d T##B4i = KS114 * T##u1i - KS113 * T##u2i + KS111 * T##u3i    \
                         + KS115 * T##u4i - KS112 * T##u5i;                    \
        const v8d T##B5r = KS115 * T##u1r - KS111 * T##u2r + KS114 * T##u3r    \
                         - KS112 * T##u4r + KS113 * T##u5r;                    \
        const v8d T##B5i = KS115 * T##u1i - KS111 * T##u2i + KS114 * T##u3i    \
                         - KS112 * T##u4i + KS113 * T##u5i;

#define M11BIND(T, p, st, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10)         \
        const v8d T##x0r = SLR(p, st, i0),  T##x0i = SLI(p, st, i0);           \
        const v8d T##x1r = SLR(p, st, i1),  T##x1i = SLI(p, st, i1);           \
        const v8d T##x2r = SLR(p, st, i2),  T##x2i = SLI(p, st, i2);           \
        const v8d T##x3r = SLR(p, st, i3),  T##x3i = SLI(p, st, i3);           \
        const v8d T##x4r = SLR(p, st, i4),  T##x4i = SLI(p, st, i4);           \
        const v8d T##x5r = SLR(p, st, i5),  T##x5i = SLI(p, st, i5);           \
        const v8d T##x6r = SLR(p, st, i6),  T##x6i = SLI(p, st, i6);           \
        const v8d T##x7r = SLR(p, st, i7),  T##x7i = SLI(p, st, i7);           \
        const v8d T##x8r = SLR(p, st, i8),  T##x8i = SLI(p, st, i8);           \
        const v8d T##x9r = SLR(p, st, i9),  T##x9i = SLI(p, st, i9);           \
        const v8d T##x10r = SLR(p, st, i10), T##x10i = SLI(p, st, i10);

#define DFT11STORE(T, p, st, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10)      \
        PR(p, st, o0)  = T##X0r;             PI_(p, st, o0)  = T##X0i;         \
        PR(p, st, o1)  = T##A1r + T##B1i;    PI_(p, st, o1)  = T##A1i - T##B1r;\
        PR(p, st, o10) = T##A1r - T##B1i;    PI_(p, st, o10) = T##A1i + T##B1r;\
        PR(p, st, o2)  = T##A2r + T##B2i;    PI_(p, st, o2)  = T##A2i - T##B2r;\
        PR(p, st, o9)  = T##A2r - T##B2i;    PI_(p, st, o9)  = T##A2i + T##B2r;\
        PR(p, st, o3)  = T##A3r + T##B3i;    PI_(p, st, o3)  = T##A3i - T##B3r;\
        PR(p, st, o8)  = T##A3r - T##B3i;    PI_(p, st, o8)  = T##A3i + T##B3r;\
        PR(p, st, o4)  = T##A4r + T##B4i;    PI_(p, st, o4)  = T##A4i - T##B4r;\
        PR(p, st, o7)  = T##A4r - T##B4i;    PI_(p, st, o7)  = T##A4i + T##B4r;\
        PR(p, st, o5)  = T##A5r + T##B5i;    PI_(p, st, o5)  = T##A5i - T##B5r;\
        PR(p, st, o6)  = T##A5r - T##B5i;    PI_(p, st, o6)  = T##A5i + T##B5r;

#define M11ST(T, p, st, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10,           \
                        o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10)           \
    do {                                                                       \
        M11BIND(T, p, st, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10)         \
        DFT11CORE(T)                                                           \
        DFT11STORE(T, p, st, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10)      \
    } while (0)

#define D11M(T, p, st, cp, UD, o, er, ei)                                      \
    do {                                                                       \
        v8d zr = (er), zi = (ei);                                              \
        map8(&zr, &zi, (cp) + (size_t)(o) * (st), UD);                         \
        PR(p, st, o) = zr;  PI_(p, st, o) = zi;                                \
    } while (0)

#define DFT11STOREM(T, p, st, cp, UD, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9,  \
                    o10)                                                       \
        D11M(T, p, st, cp, UD, o0,  T##X0r,           T##X0i);                 \
        D11M(T, p, st, cp, UD, o1,  T##A1r + T##B1i,  T##A1i - T##B1r);        \
        D11M(T, p, st, cp, UD, o10, T##A1r - T##B1i,  T##A1i + T##B1r);        \
        D11M(T, p, st, cp, UD, o2,  T##A2r + T##B2i,  T##A2i - T##B2r);        \
        D11M(T, p, st, cp, UD, o9,  T##A2r - T##B2i,  T##A2i + T##B2r);        \
        D11M(T, p, st, cp, UD, o3,  T##A3r + T##B3i,  T##A3i - T##B3r);        \
        D11M(T, p, st, cp, UD, o8,  T##A3r - T##B3i,  T##A3i + T##B3r);        \
        D11M(T, p, st, cp, UD, o4,  T##A4r + T##B4i,  T##A4i - T##B4r);        \
        D11M(T, p, st, cp, UD, o7,  T##A4r - T##B4i,  T##A4i + T##B4r);        \
        D11M(T, p, st, cp, UD, o5,  T##A5r + T##B5i,  T##A5i - T##B5r);        \
        D11M(T, p, st, cp, UD, o6,  T##A5r - T##B5i,  T##A5i + T##B5r);

#define M11STM(T, p, st, cp, UD, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10,  \
                                 o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10)  \
    do {                                                                       \
        M11BIND(T, p, st, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10)         \
        DFT11CORE(T)                                                           \
        DFT11STOREM(T, p, st, cp, UD, o0, o1, o2, o3, o4, o5, o6, o7, o8, o9,  \
                    o10)                                                       \
    } while (0)

/* ==== gen_r9: FACTOR-SWAPPED map pencils (BL_SWAP10/12/15/20).  The r4 asm
 * audit measured the fused-map x-pencil's residual as REGISTER SPILLS (15:
 * 27 spill stores + 15 spill loads + 12 folded rsp operands per pencil; 12:
 * 31 rsp touches): the map's ~7 temps ride on top of the WIDE factor's
 * stage-2 codelet (DFT5X2 = 2 cores + 10 map ladders at 15).  The never-
 * raced axis: swap the PFA factor ORDER in the map pencil only -- run the
 * LARGE factor in stage 1 (map-free, in place) and the SMALL factor in
 * stage 2 where the map fuses.  Same op count, same 4L ld + 4L st, but the
 * fused-map codelet shrinks from DFT5+5 ladders (or X2 pair + 10) to
 * DFT2/DFT3/DFT4 + 2/3/4 ladders: peak pressure collapses, spills vanish,
 * and the OOO window finds the cross-codelet ILP that the X2 fusion bought
 * with spills.  Sweep pencils keep the shipped order (they spill zero).
 * Slot tables generated and verified against a reference DFT (same Python
 * sim as r6/r8: load-all-store-all group semantics, in-place and
 * disjointness invariants asserted):
 *   L=10 sw: n=(2a+5b)%10, k=(6c+5d)%10, sigma(c)=3c mod 5 -> stage-2 pairs
 *            land naturally in place.
 *   L=12 sw: n=(3a+4b)%12, k=(9c+4d)%12, sigma(c)=3c mod 4.
 *   L=15 sw: n=(3a+5b)%15, k=(6c+10d)%15, sigma(c)=2c mod 5; stage-1 groups
 *            are EXACTLY the shipped stage-2 DFT5 groups.
 *   L=20 sw: n=(4a+5b)%20, k=(16c+5d)%20, sigma(c)=4c mod 5; stage-2 DFT4
 *            groups land naturally in place.
 * NOT bit-transparent (different summation order): full gate re-check, not
 * cmp.  Defaults set by the gen_r9 same-core races -- see strategy record. */

/* memory-form DFT3 + fused map: load 3 slots, core, map each output, store */
#define M3STM(p, st, cp, UD, i0, i1, i2, o0, o1, o2)                           \
    do {                                                                       \
        const v8d x0r = SLR(p, st, i0), x0i = SLI(p, st, i0);                  \
        const v8d x1r = SLR(p, st, i1), x1i = SLI(p, st, i1);                  \
        const v8d x2r = SLR(p, st, i2), x2i = SLI(p, st, i2);                  \
        const v8d tr = x1r + x2r, ti = x1i + x2i;                              \
        const v8d ur = x1r - x2r, ui = x1i - x2i;                              \
        const v8d hr = x0r - KH * tr, hi = x0i - KH * ti;                      \
        v8d zr, zi;                                                            \
        zr = x0r + tr;         zi = x0i + ti;                                  \
        map8(&zr, &zi, (cp) + (size_t)(o0) * (st), UD);                        \
        PR(p, st, o0) = zr;  PI_(p, st, o0) = zi;                              \
        zr = hr + KS3 * ui;    zi = hi - KS3 * ur;                             \
        map8(&zr, &zi, (cp) + (size_t)(o1) * (st), UD);                        \
        PR(p, st, o1) = zr;  PI_(p, st, o1) = zi;                              \
        zr = hr - KS3 * ui;    zi = hi + KS3 * ur;                             \
        map8(&zr, &zi, (cp) + (size_t)(o2) * (st), UD);                        \
        PR(p, st, o2) = zr;  PI_(p, st, o2) = zi;                              \
    } while (0)

/* memory-form DFT4 + fused map */
#define M4STM(p, st, cp, UD, i0, i1, i2, i3, o0, o1, o2, o3)                   \
    do {                                                                       \
        const v8d x0r = SLR(p, st, i0), x0i = SLI(p, st, i0);                  \
        const v8d x1r = SLR(p, st, i1), x1i = SLI(p, st, i1);                  \
        const v8d x2r = SLR(p, st, i2), x2i = SLI(p, st, i2);                  \
        const v8d x3r = SLR(p, st, i3), x3i = SLI(p, st, i3);                  \
        const v8d t0r = x0r + x2r, t0i = x0i + x2i;                            \
        const v8d t1r = x0r - x2r, t1i = x0i - x2i;                            \
        const v8d t2r = x1r + x3r, t2i = x1i + x3i;                            \
        const v8d t3r = x1r - x3r, t3i = x1i - x3i;                            \
        v8d zr, zi;                                                            \
        zr = t0r + t2r;  zi = t0i + t2i;                                       \
        map8(&zr, &zi, (cp) + (size_t)(o0) * (st), UD);                        \
        PR(p, st, o0) = zr;  PI_(p, st, o0) = zi;                              \
        zr = t1r + t3i;  zi = t1i - t3r;                                       \
        map8(&zr, &zi, (cp) + (size_t)(o1) * (st), UD);                        \
        PR(p, st, o1) = zr;  PI_(p, st, o1) = zi;                              \
        zr = t0r - t2r;  zi = t0i - t2i;                                       \
        map8(&zr, &zi, (cp) + (size_t)(o2) * (st), UD);                        \
        PR(p, st, o2) = zr;  PI_(p, st, o2) = zi;                              \
        zr = t1r - t3i;  zi = t1i + t3r;                                       \
        map8(&zr, &zi, (cp) + (size_t)(o3) * (st), UD);                        \
        PR(p, st, o3) = zr;  PI_(p, st, o3) = zi;                              \
    } while (0)

/* memory-form DFT2 + fused map (gen_r10: swapped stage-2 at 14/22) */
#define M2STM(p, st, cp, UD, i0, i1, o0, o1)                                   \
    do {                                                                       \
        const v8d x0r = SLR(p, st, i0), x0i = SLI(p, st, i0);                  \
        const v8d x1r = SLR(p, st, i1), x1i = SLI(p, st, i1);                  \
        v8d zr, zi;                                                            \
        zr = x0r + x1r;  zi = x0i + x1i;                                       \
        map8(&zr, &zi, (cp) + (size_t)(o0) * (st), UD);                        \
        PR(p, st, o0) = zr;  PI_(p, st, o0) = zi;                              \
        zr = x0r - x1r;  zi = x0i - x1i;                                       \
        map8(&zr, &zi, (cp) + (size_t)(o1) * (st), UD);                        \
        PR(p, st, o1) = zr;  PI_(p, st, o1) = zi;                              \
    } while (0)

/* stage-1 DFT-4 memory -> NAMED registers (register form, L=12 swap) */
#define R4L(p, st, a0, a1, a2, a3, o0, o1, o2, o3)                             \
    do {                                                                       \
        const v8d x0r = SLR(p, st, a0), x0i = SLI(p, st, a0);                  \
        const v8d x1r = SLR(p, st, a1), x1i = SLI(p, st, a1);                  \
        const v8d x2r = SLR(p, st, a2), x2i = SLI(p, st, a2);                  \
        const v8d x3r = SLR(p, st, a3), x3i = SLI(p, st, a3);                  \
        const v8d t0r = x0r + x2r, t0i = x0i + x2i;                            \
        const v8d t1r = x0r - x2r, t1i = x0i - x2i;                            \
        const v8d t2r = x1r + x3r, t2i = x1i + x3i;                            \
        const v8d t3r = x1r - x3r, t3i = x1i - x3i;                            \
        xr##o0 = t0r + t2r;  xi##o0 = t0i + t2i;                               \
        xr##o1 = t1r + t3i;  xi##o1 = t1i - t3r;                               \
        xr##o2 = t0r - t2r;  xi##o2 = t0i - t2i;                               \
        xr##o3 = t1r - t3i;  xi##o3 = t1i + t3r;                               \
    } while (0)

/* stage-1 DFT-5 memory -> NAMED registers (register form, L=10 swap) */
#define R5L(T, p, st, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4)                  \
    do {                                                                       \
        M5BIND(T, p, st, i0, i1, i2, i3, i4)                                   \
        DFT5CORE(T)                                                            \
        xr##o0 = T##X0r;             xi##o0 = T##X0i;                          \
        xr##o1 = T##A1r + T##v1i;    xi##o1 = T##A1i - T##v1r;                 \
        xr##o4 = T##A1r - T##v1i;    xi##o4 = T##A1i + T##v1r;                 \
        xr##o2 = T##A2r + T##v2i;    xi##o2 = T##A2i - T##v2r;                 \
        xr##o3 = T##A2r - T##v2i;    xi##o3 = T##A2i + T##v2r;                 \
    } while (0)

/* stage-2 DFT-3 registers -> memory + fused map (register form) */
#define R3STM(p, st, cp, i0, i1, i2, o0, o1, o2)                               \
    do {                                                                       \
        const v8d tr = xr##i1 + xr##i2, ti = xi##i1 + xi##i2;                  \
        const v8d ur = xr##i1 - xr##i2, ui = xi##i1 - xi##i2;                  \
        const v8d hr = xr##i0 - KH * tr, hi = xi##i0 - KH * ti;                \
        v8d zr, zi;                                                            \
        zr = xr##i0 + tr;      zi = xi##i0 + ti;                               \
        map8(&zr, &zi, (cp) + (size_t)(o0) * (st), MAPTAIL_SW12);              \
        PR(p, st, o0) = zr;  PI_(p, st, o0) = zi;                              \
        zr = hr + KS3 * ui;    zi = hi - KS3 * ur;                             \
        map8(&zr, &zi, (cp) + (size_t)(o1) * (st), MAPTAIL_SW12);              \
        PR(p, st, o1) = zr;  PI_(p, st, o1) = zi;                              \
        zr = hr - KS3 * ui;    zi = hi + KS3 * ur;                             \
        map8(&zr, &zi, (cp) + (size_t)(o2) * (st), MAPTAIL_SW12);              \
        PR(p, st, o2) = zr;  PI_(p, st, o2) = zi;                              \
    } while (0)

/* stage-2 DFT-2 registers -> memory + fused map (register form) */
#define R2STM(p, st, cp, i0, i1, o0, o1)                                       \
    do {                                                                       \
        v8d zr, zi;                                                            \
        zr = xr##i0 + xr##i1;  zi = xi##i0 + xi##i1;                           \
        map8(&zr, &zi, (cp) + (size_t)(o0) * (st), MAPTAIL_SW10);              \
        PR(p, st, o0) = zr;  PI_(p, st, o0) = zi;                              \
        zr = xr##i0 - xr##i1;  zi = xi##i0 - xi##i1;                           \
        map8(&zr, &zi, (cp) + (size_t)(o1) * (st), MAPTAIL_SW10);              \
        PR(p, st, o1) = zr;  PI_(p, st, o1) = zi;                              \
    } while (0)

/* ---- gen_r6: L=15 SAFE-PLACEMENT pencils (BL_SAFE15).  Same memory-form
 * modules as the r5 ship, but stage 1 uses the M3IPS store permutation so
 * stage 2 is three INDEPENDENT in-place DFT5 groups instead of one plain
 * group + the forced DFT5X2 fused pair (2 live DFT5COREs + 2 map ladders in
 * one codelet).  Values and store slots per output k are unchanged ->
 * chain outputs bit-identical to the fused form; only peak register
 * pressure in the fused-map x-pencil drops. */
AIN void dft15_pencil_mem_safe(double *restrict p, const ptrdiff_t st)
{
    M3IPS(p, st, 0, 5, 10);  M3IPS(p, st, 3, 8, 13); M3IPS(p, st, 6, 11, 1);
    M3IPS(p, st, 9, 14, 4);  M3IPS(p, st, 12, 2, 7);
    M5ST(a_, p, st,  0, 3, 6, 9, 12,   0, 6, 12, 3, 9);
    M5ST(b_, p, st, 10, 13, 1, 4, 7,  10, 1, 7, 13, 4);
    M5ST(c_, p, st,  5, 8, 11, 14, 2,  5, 11, 2, 8, 14);
}

AIN void dft15_pencil_map_mem_safe(double *restrict p, const ptrdiff_t st,
                                   const double *restrict cp)
{
    M3IPS(p, st, 0, 5, 10);  M3IPS(p, st, 3, 8, 13); M3IPS(p, st, 6, 11, 1);
    M3IPS(p, st, 9, 14, 4);  M3IPS(p, st, 12, 2, 7);
    M5STMU(a_, p, st, cp, MAPTAIL_15,  0, 3, 6, 9, 12,   0, 6, 12, 3, 9);
    M5STMU(b_, p, st, cp, MAPTAIL_15, 10, 13, 1, 4, 7,  10, 1, 7, 13, 4);
    M5STMU(c_, p, st, cp, MAPTAIL_15,  5, 8, 11, 14, 2,  5, 11, 2, 8, 14);
}

/* ---- gen_r6: the 7-smooth class sizes, all memory form (2L site registers
 * exceed 32 zmm at every one), all safe-placement (no fusions anywhere).
 * Slot tables generated and verified against a reference DFT.
 *   L=14 = 2x7: n=(7a+2b)%14, k=(7c+8d)%14,  Q==1 mod P: natural placement.
 *   L=21 = 3x7: n=(7a+3b)%21, k=(7c+15d)%21, Q==1 mod P: natural placement.
 *   L=28 = 4x7: n=(7a+4b)%28, k=(21c+8d)%28, Q^-1=3 mod 4: M4IPO(i0,i3,i2,i1).
 *   L=35 = 5x7: n=(7a+5b)%35, k=(21c+15d)%35, Q^-1=3 mod 5: X_c at a=3c mod 5,
 *               stage-1 DFT5 via M5ST with o=(i0,i3,i1,i4,i2).
 * Codelet costs (vector FP per pencil per 8 vols): 14: 7xDFT2+2xDFT7 = 160;
 * 21: 7xDFT3+3xDFT7 = 282; 28: 7xDFT4+4xDFT7 = 376; 35: 7xDFT5+5xDFT7 = 568
 * (554 with the r7 lifted DFT5). */
AIN void dft14_pencil(double *restrict p, const ptrdiff_t st)
{
    M2IP(p, st, 0, 7);  M2IP(p, st, 2, 9);   M2IP(p, st, 4, 11);
    M2IP(p, st, 6, 13); M2IP(p, st, 8, 1);   M2IP(p, st, 10, 3);
    M2IP(p, st, 12, 5);
    M7ST(a_, p, st, 0, 2, 4, 6, 8, 10, 12,  0, 8, 2, 10, 4, 12, 6);
    M7ST(b_, p, st, 7, 9, 11, 13, 1, 3, 5,  7, 1, 9, 3, 11, 5, 13);
}

AIN void dft14_pencil_map(double *restrict p, const ptrdiff_t st,
                          const double *restrict cp)
{
    M2IP(p, st, 0, 7);  M2IP(p, st, 2, 9);   M2IP(p, st, 4, 11);
    M2IP(p, st, 6, 13); M2IP(p, st, 8, 1);   M2IP(p, st, 10, 3);
    M2IP(p, st, 12, 5);
    M7STM(a_, p, st, cp, MAPTAIL_GEN, 0, 2, 4, 6, 8, 10, 12,
                                      0, 8, 2, 10, 4, 12, 6);
    M7STM(b_, p, st, cp, MAPTAIL_GEN, 7, 9, 11, 13, 1, 3, 5,
                                      7, 1, 9, 3, 11, 5, 13);
}

/* gen_r7: DFT7X2 -- both stage-2 DFT7 groups of L=14 in ONE codelet (load
 * both, run both cores interleaved, store both), the r6 next-step #1: the
 * DFT5X2-at-15 lesson says two interleaved dependency chains beat two
 * serial groups by ~1% when the surrounding pressure allows.  Groups are
 * disjoint (natural placement, Q=7 == 1 mod 2), so the fusion is purely an
 * ILP play; outputs bit-identical to the serial form.  BL_X214 knob. */
AIN void dft14_pencil_x2(double *restrict p, const ptrdiff_t st)
{
    M2IP(p, st, 0, 7);  M2IP(p, st, 2, 9);   M2IP(p, st, 4, 11);
    M2IP(p, st, 6, 13); M2IP(p, st, 8, 1);   M2IP(p, st, 10, 3);
    M2IP(p, st, 12, 5);
    M7BIND(a_, p, st, 0, 2, 4, 6, 8, 10, 12)
    M7BIND(b_, p, st, 7, 9, 11, 13, 1, 3, 5)
    DFT7CORE(a_)
    DFT7CORE(b_)
    DFT7STORE(a_, p, st, 0, 8, 2, 10, 4, 12, 6)
    DFT7STORE(b_, p, st, 7, 1, 9, 3, 11, 5, 13)
}

AIN void dft14_pencil_map_x2(double *restrict p, const ptrdiff_t st,
                             const double *restrict cp)
{
    M2IP(p, st, 0, 7);  M2IP(p, st, 2, 9);   M2IP(p, st, 4, 11);
    M2IP(p, st, 6, 13); M2IP(p, st, 8, 1);   M2IP(p, st, 10, 3);
    M2IP(p, st, 12, 5);
    M7BIND(a_, p, st, 0, 2, 4, 6, 8, 10, 12)
    M7BIND(b_, p, st, 7, 9, 11, 13, 1, 3, 5)
    DFT7CORE(a_)
    DFT7CORE(b_)
    DFT7STOREM(a_, p, st, cp, MAPTAIL_GEN, 0, 8, 2, 10, 4, 12, 6);
    DFT7STOREM(b_, p, st, cp, MAPTAIL_GEN, 7, 1, 9, 3, 11, 5, 13);
}

AIN void dft21_pencil(double *restrict p, const ptrdiff_t st)
{
    M3IP(p, st, 0, 7, 14);   M3IP(p, st, 3, 10, 17);  M3IP(p, st, 6, 13, 20);
    M3IP(p, st, 9, 16, 2);   M3IP(p, st, 12, 19, 5);  M3IP(p, st, 15, 1, 8);
    M3IP(p, st, 18, 4, 11);
    M7ST(a_, p, st,  0, 3, 6, 9, 12, 15, 18,   0, 15, 9, 3, 18, 12, 6);
    M7ST(b_, p, st,  7, 10, 13, 16, 19, 1, 4,  7, 1, 16, 10, 4, 19, 13);
    M7ST(c_, p, st, 14, 17, 20, 2, 5, 8, 11,  14, 8, 2, 17, 11, 5, 20);
}

AIN void dft21_pencil_map(double *restrict p, const ptrdiff_t st,
                          const double *restrict cp)
{
    M3IP(p, st, 0, 7, 14);   M3IP(p, st, 3, 10, 17);  M3IP(p, st, 6, 13, 20);
    M3IP(p, st, 9, 16, 2);   M3IP(p, st, 12, 19, 5);  M3IP(p, st, 15, 1, 8);
    M3IP(p, st, 18, 4, 11);
    M7STM(a_, p, st, cp, MAPTAIL_GEN,  0, 3, 6, 9, 12, 15, 18,
                                       0, 15, 9, 3, 18, 12, 6);
    M7STM(b_, p, st, cp, MAPTAIL_GEN,  7, 10, 13, 16, 19, 1, 4,
                                       7, 1, 16, 10, 4, 19, 13);
    M7STM(c_, p, st, cp, MAPTAIL_GEN, 14, 17, 20, 2, 5, 8, 11,
                                      14, 8, 2, 17, 11, 5, 20);
}

AIN void dft28_pencil(double *restrict p, const ptrdiff_t st)
{
    M4IPO(p, st, 0, 7, 14, 21,    0, 21, 14, 7);
    M4IPO(p, st, 4, 11, 18, 25,   4, 25, 18, 11);
    M4IPO(p, st, 8, 15, 22, 1,    8, 1, 22, 15);
    M4IPO(p, st, 12, 19, 26, 5,   12, 5, 26, 19);
    M4IPO(p, st, 16, 23, 2, 9,    16, 9, 2, 23);
    M4IPO(p, st, 20, 27, 6, 13,   20, 13, 6, 27);
    M4IPO(p, st, 24, 3, 10, 17,   24, 17, 10, 3);
    M7ST(a_, p, st,  0, 4, 8, 12, 16, 20, 24,   0, 8, 16, 24, 4, 12, 20);
    M7ST(b_, p, st, 21, 25, 1, 5, 9, 13, 17,   21, 1, 9, 17, 25, 5, 13);
    M7ST(c_, p, st, 14, 18, 22, 26, 2, 6, 10,  14, 22, 2, 10, 18, 26, 6);
    M7ST(d_, p, st,  7, 11, 15, 19, 23, 27, 3,  7, 15, 23, 3, 11, 19, 27);
}

AIN void dft28_pencil_map(double *restrict p, const ptrdiff_t st,
                          const double *restrict cp)
{
    M4IPO(p, st, 0, 7, 14, 21,    0, 21, 14, 7);
    M4IPO(p, st, 4, 11, 18, 25,   4, 25, 18, 11);
    M4IPO(p, st, 8, 15, 22, 1,    8, 1, 22, 15);
    M4IPO(p, st, 12, 19, 26, 5,   12, 5, 26, 19);
    M4IPO(p, st, 16, 23, 2, 9,    16, 9, 2, 23);
    M4IPO(p, st, 20, 27, 6, 13,   20, 13, 6, 27);
    M4IPO(p, st, 24, 3, 10, 17,   24, 17, 10, 3);
    M7STM(a_, p, st, cp, MAPTAIL_GEN,  0, 4, 8, 12, 16, 20, 24,
                                       0, 8, 16, 24, 4, 12, 20);
    M7STM(b_, p, st, cp, MAPTAIL_GEN, 21, 25, 1, 5, 9, 13, 17,
                                      21, 1, 9, 17, 25, 5, 13);
    M7STM(c_, p, st, cp, MAPTAIL_GEN, 14, 18, 22, 26, 2, 6, 10,
                                      14, 22, 2, 10, 18, 26, 6);
    M7STM(d_, p, st, cp, MAPTAIL_GEN,  7, 11, 15, 19, 23, 27, 3,
                                       7, 15, 23, 3, 11, 19, 27);
}

AIN void dft35_pencil(double *restrict p, const ptrdiff_t st)
{
    M5ST(sa, p, st,  0, 7, 14, 21, 28,   0, 21, 7, 28, 14);
    M5ST(sb, p, st,  5, 12, 19, 26, 33,  5, 26, 12, 33, 19);
    M5ST(sc, p, st, 10, 17, 24, 31, 3,  10, 31, 17, 3, 24);
    M5ST(sd, p, st, 15, 22, 29, 1, 8,   15, 1, 22, 8, 29);
    M5ST(se, p, st, 20, 27, 34, 6, 13,  20, 6, 27, 13, 34);
    M5ST(sf, p, st, 25, 32, 4, 11, 18,  25, 11, 32, 18, 4);
    M5ST(sg, p, st, 30, 2, 9, 16, 23,   30, 16, 2, 23, 9);
    M7ST(a_, p, st,  0, 5, 10, 15, 20, 25, 30,   0, 15, 30, 10, 25, 5, 20);
    M7ST(b_, p, st, 21, 26, 31, 1, 6, 11, 16,   21, 1, 16, 31, 11, 26, 6);
    M7ST(c_, p, st,  7, 12, 17, 22, 27, 32, 2,   7, 22, 2, 17, 32, 12, 27);
    M7ST(d_, p, st, 28, 33, 3, 8, 13, 18, 23,   28, 8, 23, 3, 18, 33, 13);
    M7ST(e_, p, st, 14, 19, 24, 29, 34, 4, 9,   14, 29, 9, 24, 4, 19, 34);
}

AIN void dft35_pencil_map(double *restrict p, const ptrdiff_t st,
                          const double *restrict cp)
{
    M5ST(sa, p, st,  0, 7, 14, 21, 28,   0, 21, 7, 28, 14);
    M5ST(sb, p, st,  5, 12, 19, 26, 33,  5, 26, 12, 33, 19);
    M5ST(sc, p, st, 10, 17, 24, 31, 3,  10, 31, 17, 3, 24);
    M5ST(sd, p, st, 15, 22, 29, 1, 8,   15, 1, 22, 8, 29);
    M5ST(se, p, st, 20, 27, 34, 6, 13,  20, 6, 27, 13, 34);
    M5ST(sf, p, st, 25, 32, 4, 11, 18,  25, 11, 32, 18, 4);
    M5ST(sg, p, st, 30, 2, 9, 16, 23,   30, 16, 2, 23, 9);
    M7STM(a_, p, st, cp, MAPTAIL_GEN,  0, 5, 10, 15, 20, 25, 30,
                                       0, 15, 30, 10, 25, 5, 20);
    M7STM(b_, p, st, cp, MAPTAIL_GEN, 21, 26, 31, 1, 6, 11, 16,
                                      21, 1, 16, 31, 11, 26, 6);
    M7STM(c_, p, st, cp, MAPTAIL_GEN,  7, 12, 17, 22, 27, 32, 2,
                                       7, 22, 2, 17, 32, 12, 27);
    M7STM(d_, p, st, cp, MAPTAIL_GEN, 28, 33, 3, 8, 13, 18, 23,
                                      28, 8, 23, 3, 18, 33, 13);
    M7STM(e_, p, st, cp, MAPTAIL_GEN, 14, 19, 24, 29, 34, 4, 9,
                                      14, 29, 9, 24, 4, 19, 34);
}

/* ---- gen_r8: the 11-smooth class sizes 22/33/44/55, memory form, safe
 * placement, div map tail, stock scheduler (the memory-form family verdicts).
 * Slot tables generated and verified against a reference DFT (Python sim of
 * the exact load-all-store-all group semantics below).
 *   L=22 = 2x11: n=(11a+2b)%22,  Q==1 mod P:  natural placement (M2IP).
 *   L=33 = 3x11: n=(11a+3b)%33,  Q^-1=2 mod 3: M3IPS (the L=15 perm).
 *   L=44 = 4x11: n=(11a+4b)%44,  Q^-1=3 mod 4: M4IPO(i0,i3,i2,i1) (L=28's).
 *   L=55 = 5x11: n=(11a+5b)%55,  Q==1 mod P:  natural placement (M5ST id).
 * Codelet costs (vector FP per pencil per 8 vols): 22: 11xDFT2 + 2xDFT11 =
 * 388; 33: 11xDFT3 + 3xDFT11 = 582; 44: 11xDFT4 + 4xDFT11 = 776;
 * 55: 11xDFT5(lifted) + 5xDFT11 = 1102.  L=77 = 7x11 was left out: an
 * 11xDFT7 + 7xDFT11 = 1776-FP always_inline pencil (x2 forms) is a real
 * compile-time hazard for a size no case names; its verified slot table is
 * in the strategy record if a round ever wants it. */
AIN void dft22_pencil(double *restrict p, const ptrdiff_t st)
{
    M2IP(p, st, 0, 11);  M2IP(p, st, 2, 13);  M2IP(p, st, 4, 15);
    M2IP(p, st, 6, 17);  M2IP(p, st, 8, 19);  M2IP(p, st, 10, 21);
    M2IP(p, st, 12, 1);  M2IP(p, st, 14, 3);  M2IP(p, st, 16, 5);
    M2IP(p, st, 18, 7);  M2IP(p, st, 20, 9);
    M11ST(a_, p, st,  0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                      0, 12, 2, 14, 4, 16, 6, 18, 8, 20, 10);
    M11ST(b_, p, st, 11, 13, 15, 17, 19, 21, 1, 3, 5, 7, 9,
                     11, 1, 13, 3, 15, 5, 17, 7, 19, 9, 21);
}

AIN void dft22_pencil_map(double *restrict p, const ptrdiff_t st,
                          const double *restrict cp)
{
    M2IP(p, st, 0, 11);  M2IP(p, st, 2, 13);  M2IP(p, st, 4, 15);
    M2IP(p, st, 6, 17);  M2IP(p, st, 8, 19);  M2IP(p, st, 10, 21);
    M2IP(p, st, 12, 1);  M2IP(p, st, 14, 3);  M2IP(p, st, 16, 5);
    M2IP(p, st, 18, 7);  M2IP(p, st, 20, 9);
    M11STM(a_, p, st, cp, MAPTAIL_GEN,  0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                                        0, 12, 2, 14, 4, 16, 6, 18, 8, 20, 10);
    M11STM(b_, p, st, cp, MAPTAIL_GEN, 11, 13, 15, 17, 19, 21, 1, 3, 5, 7, 9,
                                       11, 1, 13, 3, 15, 5, 17, 7, 19, 9, 21);
}

AIN void dft33_pencil(double *restrict p, const ptrdiff_t st)
{
    M3IPS(p, st, 0, 11, 22);  M3IPS(p, st, 3, 14, 25);  M3IPS(p, st, 6, 17, 28);
    M3IPS(p, st, 9, 20, 31);  M3IPS(p, st, 12, 23, 1);  M3IPS(p, st, 15, 26, 4);
    M3IPS(p, st, 18, 29, 7);  M3IPS(p, st, 21, 32, 10); M3IPS(p, st, 24, 2, 13);
    M3IPS(p, st, 27, 5, 16);  M3IPS(p, st, 30, 8, 19);
    M11ST(a_, p, st,  0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30,
                      0, 12, 24, 3, 15, 27, 6, 18, 30, 9, 21);
    M11ST(b_, p, st, 22, 25, 28, 31, 1, 4, 7, 10, 13, 16, 19,
                     22, 1, 13, 25, 4, 16, 28, 7, 19, 31, 10);
    M11ST(c_, p, st, 11, 14, 17, 20, 23, 26, 29, 32, 2, 5, 8,
                     11, 23, 2, 14, 26, 5, 17, 29, 8, 20, 32);
}

AIN void dft33_pencil_map(double *restrict p, const ptrdiff_t st,
                          const double *restrict cp)
{
    M3IPS(p, st, 0, 11, 22);  M3IPS(p, st, 3, 14, 25);  M3IPS(p, st, 6, 17, 28);
    M3IPS(p, st, 9, 20, 31);  M3IPS(p, st, 12, 23, 1);  M3IPS(p, st, 15, 26, 4);
    M3IPS(p, st, 18, 29, 7);  M3IPS(p, st, 21, 32, 10); M3IPS(p, st, 24, 2, 13);
    M3IPS(p, st, 27, 5, 16);  M3IPS(p, st, 30, 8, 19);
    M11STM(a_, p, st, cp, MAPTAIL_GEN,  0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30,
                                        0, 12, 24, 3, 15, 27, 6, 18, 30, 9, 21);
    M11STM(b_, p, st, cp, MAPTAIL_GEN, 22, 25, 28, 31, 1, 4, 7, 10, 13, 16, 19,
                                       22, 1, 13, 25, 4, 16, 28, 7, 19, 31, 10);
    M11STM(c_, p, st, cp, MAPTAIL_GEN, 11, 14, 17, 20, 23, 26, 29, 32, 2, 5, 8,
                                       11, 23, 2, 14, 26, 5, 17, 29, 8, 20, 32);
}

AIN void dft44_pencil(double *restrict p, const ptrdiff_t st)
{
    M4IPO(p, st, 0, 11, 22, 33,   0, 33, 22, 11);
    M4IPO(p, st, 4, 15, 26, 37,   4, 37, 26, 15);
    M4IPO(p, st, 8, 19, 30, 41,   8, 41, 30, 19);
    M4IPO(p, st, 12, 23, 34, 1,   12, 1, 34, 23);
    M4IPO(p, st, 16, 27, 38, 5,   16, 5, 38, 27);
    M4IPO(p, st, 20, 31, 42, 9,   20, 9, 42, 31);
    M4IPO(p, st, 24, 35, 2, 13,   24, 13, 2, 35);
    M4IPO(p, st, 28, 39, 6, 17,   28, 17, 6, 39);
    M4IPO(p, st, 32, 43, 10, 21,  32, 21, 10, 43);
    M4IPO(p, st, 36, 3, 14, 25,   36, 25, 14, 3);
    M4IPO(p, st, 40, 7, 18, 29,   40, 29, 18, 7);
    M11ST(a_, p, st,  0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40,
                      0, 12, 24, 36, 4, 16, 28, 40, 8, 20, 32);
    M11ST(b_, p, st, 33, 37, 41, 1, 5, 9, 13, 17, 21, 25, 29,
                     33, 1, 13, 25, 37, 5, 17, 29, 41, 9, 21);
    M11ST(c_, p, st, 22, 26, 30, 34, 38, 42, 2, 6, 10, 14, 18,
                     22, 34, 2, 14, 26, 38, 6, 18, 30, 42, 10);
    M11ST(d_, p, st, 11, 15, 19, 23, 27, 31, 35, 39, 43, 3, 7,
                     11, 23, 35, 3, 15, 27, 39, 7, 19, 31, 43);
}

AIN void dft44_pencil_map(double *restrict p, const ptrdiff_t st,
                          const double *restrict cp)
{
    M4IPO(p, st, 0, 11, 22, 33,   0, 33, 22, 11);
    M4IPO(p, st, 4, 15, 26, 37,   4, 37, 26, 15);
    M4IPO(p, st, 8, 19, 30, 41,   8, 41, 30, 19);
    M4IPO(p, st, 12, 23, 34, 1,   12, 1, 34, 23);
    M4IPO(p, st, 16, 27, 38, 5,   16, 5, 38, 27);
    M4IPO(p, st, 20, 31, 42, 9,   20, 9, 42, 31);
    M4IPO(p, st, 24, 35, 2, 13,   24, 13, 2, 35);
    M4IPO(p, st, 28, 39, 6, 17,   28, 17, 6, 39);
    M4IPO(p, st, 32, 43, 10, 21,  32, 21, 10, 43);
    M4IPO(p, st, 36, 3, 14, 25,   36, 25, 14, 3);
    M4IPO(p, st, 40, 7, 18, 29,   40, 29, 18, 7);
    M11STM(a_, p, st, cp, MAPTAIL_GEN,  0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40,
                                        0, 12, 24, 36, 4, 16, 28, 40, 8, 20, 32);
    M11STM(b_, p, st, cp, MAPTAIL_GEN, 33, 37, 41, 1, 5, 9, 13, 17, 21, 25, 29,
                                       33, 1, 13, 25, 37, 5, 17, 29, 41, 9, 21);
    M11STM(c_, p, st, cp, MAPTAIL_GEN, 22, 26, 30, 34, 38, 42, 2, 6, 10, 14, 18,
                                       22, 34, 2, 14, 26, 38, 6, 18, 30, 42, 10);
    M11STM(d_, p, st, cp, MAPTAIL_GEN, 11, 15, 19, 23, 27, 31, 35, 39, 43, 3, 7,
                                       11, 23, 35, 3, 15, 27, 39, 7, 19, 31, 43);
}

AIN void dft55_pencil(double *restrict p, const ptrdiff_t st)
{
    M5ST(sa, p, st,  0, 11, 22, 33, 44,   0, 11, 22, 33, 44);
    M5ST(sb, p, st,  5, 16, 27, 38, 49,   5, 16, 27, 38, 49);
    M5ST(sc, p, st, 10, 21, 32, 43, 54,  10, 21, 32, 43, 54);
    M5ST(sd, p, st, 15, 26, 37, 48, 4,   15, 26, 37, 48, 4);
    M5ST(se, p, st, 20, 31, 42, 53, 9,   20, 31, 42, 53, 9);
    M5ST(sf, p, st, 25, 36, 47, 3, 14,   25, 36, 47, 3, 14);
    M5ST(sg, p, st, 30, 41, 52, 8, 19,   30, 41, 52, 8, 19);
    M5ST(sh, p, st, 35, 46, 2, 13, 24,   35, 46, 2, 13, 24);
    M5ST(si, p, st, 40, 51, 7, 18, 29,   40, 51, 7, 18, 29);
    M5ST(sj, p, st, 45, 1, 12, 23, 34,   45, 1, 12, 23, 34);
    M5ST(sk, p, st, 50, 6, 17, 28, 39,   50, 6, 17, 28, 39);
    M11ST(a_, p, st,  0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                      0, 45, 35, 25, 15, 5, 50, 40, 30, 20, 10);
    M11ST(b_, p, st, 11, 16, 21, 26, 31, 36, 41, 46, 51, 1, 6,
                     11, 1, 46, 36, 26, 16, 6, 51, 41, 31, 21);
    M11ST(c_, p, st, 22, 27, 32, 37, 42, 47, 52, 2, 7, 12, 17,
                     22, 12, 2, 47, 37, 27, 17, 7, 52, 42, 32);
    M11ST(d_, p, st, 33, 38, 43, 48, 53, 3, 8, 13, 18, 23, 28,
                     33, 23, 13, 3, 48, 38, 28, 18, 8, 53, 43);
    M11ST(e_, p, st, 44, 49, 54, 4, 9, 14, 19, 24, 29, 34, 39,
                     44, 34, 24, 14, 4, 49, 39, 29, 19, 9, 54);
}

AIN void dft55_pencil_map(double *restrict p, const ptrdiff_t st,
                          const double *restrict cp)
{
    M5ST(sa, p, st,  0, 11, 22, 33, 44,   0, 11, 22, 33, 44);
    M5ST(sb, p, st,  5, 16, 27, 38, 49,   5, 16, 27, 38, 49);
    M5ST(sc, p, st, 10, 21, 32, 43, 54,  10, 21, 32, 43, 54);
    M5ST(sd, p, st, 15, 26, 37, 48, 4,   15, 26, 37, 48, 4);
    M5ST(se, p, st, 20, 31, 42, 53, 9,   20, 31, 42, 53, 9);
    M5ST(sf, p, st, 25, 36, 47, 3, 14,   25, 36, 47, 3, 14);
    M5ST(sg, p, st, 30, 41, 52, 8, 19,   30, 41, 52, 8, 19);
    M5ST(sh, p, st, 35, 46, 2, 13, 24,   35, 46, 2, 13, 24);
    M5ST(si, p, st, 40, 51, 7, 18, 29,   40, 51, 7, 18, 29);
    M5ST(sj, p, st, 45, 1, 12, 23, 34,   45, 1, 12, 23, 34);
    M5ST(sk, p, st, 50, 6, 17, 28, 39,   50, 6, 17, 28, 39);
    M11STM(a_, p, st, cp, MAPTAIL_GEN,  0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                                        0, 45, 35, 25, 15, 5, 50, 40, 30, 20, 10);
    M11STM(b_, p, st, cp, MAPTAIL_GEN, 11, 16, 21, 26, 31, 36, 41, 46, 51, 1, 6,
                                       11, 1, 46, 36, 26, 16, 6, 51, 41, 31, 21);
    M11STM(c_, p, st, cp, MAPTAIL_GEN, 22, 27, 32, 37, 42, 47, 52, 2, 7, 12, 17,
                                       22, 12, 2, 47, 37, 27, 17, 7, 52, 42, 32);
    M11STM(d_, p, st, cp, MAPTAIL_GEN, 33, 38, 43, 48, 53, 3, 8, 13, 18, 23, 28,
                                       33, 23, 13, 3, 48, 38, 28, 18, 8, 53, 43);
    M11STM(e_, p, st, cp, MAPTAIL_GEN, 44, 49, 54, 4, 9, 14, 19, 24, 29, 34, 39,
                                       44, 34, 24, 14, 4, 49, 39, 29, 19, 9, 54);
}

/* ==== gen_r10: FACTOR-SWAPPED map pencils for the wide-module memory-form
 * family (the r9 swap extended to 7-smooth and 11-smooth sizes -- my own r9
 * next-step #2).  Mechanism as at 15/20, but the codelet delta is much
 * larger here: the fused-map stage-2 shrinks from DFT7/DFT11 + 7/11 map
 * ladders per group (a ~24/~40-temp core with the ladder block riding on
 * top) to DFT2/3/4/5 + 2..5 ladders, while stage 1 becomes the map-free
 * wide DFT.  Swapped input map n=(Pa+Qb)%L (P small, Q wide), CRT output
 * k=(e1 c + e2 d)%L, safe placement sigma(c)=(P^-1 mod Q)c.  With this map
 * the plain module output j IS the true index (no exponent scrambling), and
 * the swapped stage-1 tables come out IDENTICAL to the shipped unswapped
 * stage-2 tables (the r9 mirror symmetry at 15, now at every size).  All
 * eight tables generated and verified in Python against a reference DFT
 * (exact load-all-store-all group semantics, in-place + disjointness
 * asserted, 3 seeds each, err ~2e-15; script in
 * build/tryout/gen_batchlane/gen_swap_tables.py).  Map pencil ONLY; sweeps
 * keep the shipped order.  NOT bit-transparent: full gate re-check. */
AIN void dft14_pencil_map_swap(double *restrict p, const ptrdiff_t st,
                               const double *restrict cp)
{
    M7ST(a_, p, st, 0, 2, 4, 6, 8, 10, 12,  0, 8, 2, 10, 4, 12, 6);
    M7ST(b_, p, st, 7, 9, 11, 13, 1, 3, 5,  7, 1, 9, 3, 11, 5, 13);
    M2STM(p, st, cp, MAPTAIL_GEN, 0, 7,    0, 7);
    M2STM(p, st, cp, MAPTAIL_GEN, 8, 1,    8, 1);
    M2STM(p, st, cp, MAPTAIL_GEN, 2, 9,    2, 9);
    M2STM(p, st, cp, MAPTAIL_GEN, 10, 3,   10, 3);
    M2STM(p, st, cp, MAPTAIL_GEN, 4, 11,   4, 11);
    M2STM(p, st, cp, MAPTAIL_GEN, 12, 5,   12, 5);
    M2STM(p, st, cp, MAPTAIL_GEN, 6, 13,   6, 13);
}

AIN void dft21_pencil_map_swap(double *restrict p, const ptrdiff_t st,
                               const double *restrict cp)
{
    M7ST(a_, p, st,  0, 3, 6, 9, 12, 15, 18,   0, 15, 9, 3, 18, 12, 6);
    M7ST(b_, p, st,  7, 10, 13, 16, 19, 1, 4,  7, 1, 16, 10, 4, 19, 13);
    M7ST(c_, p, st, 14, 17, 20, 2, 5, 8, 11,  14, 8, 2, 17, 11, 5, 20);
    M3STM(p, st, cp, MAPTAIL_GEN,  0, 7, 14,    0, 7, 14);
    M3STM(p, st, cp, MAPTAIL_GEN, 15, 1, 8,    15, 1, 8);
    M3STM(p, st, cp, MAPTAIL_GEN,  9, 16, 2,    9, 16, 2);
    M3STM(p, st, cp, MAPTAIL_GEN,  3, 10, 17,   3, 10, 17);
    M3STM(p, st, cp, MAPTAIL_GEN, 18, 4, 11,   18, 4, 11);
    M3STM(p, st, cp, MAPTAIL_GEN, 12, 19, 5,   12, 19, 5);
    M3STM(p, st, cp, MAPTAIL_GEN,  6, 13, 20,   6, 13, 20);
}

AIN void dft28_pencil_map_swap(double *restrict p, const ptrdiff_t st,
                               const double *restrict cp)
{
    M7ST(a_, p, st,  0, 4, 8, 12, 16, 20, 24,   0, 8, 16, 24, 4, 12, 20);
    M7ST(b_, p, st,  7, 11, 15, 19, 23, 27, 3,  7, 15, 23, 3, 11, 19, 27);
    M7ST(c_, p, st, 14, 18, 22, 26, 2, 6, 10,  14, 22, 2, 10, 18, 26, 6);
    M7ST(d_, p, st, 21, 25, 1, 5, 9, 13, 17,   21, 1, 9, 17, 25, 5, 13);
    M4STM(p, st, cp, MAPTAIL_GEN,  0, 7, 14, 21,    0, 21, 14, 7);
    M4STM(p, st, cp, MAPTAIL_GEN,  8, 15, 22, 1,    8, 1, 22, 15);
    M4STM(p, st, cp, MAPTAIL_GEN, 16, 23, 2, 9,    16, 9, 2, 23);
    M4STM(p, st, cp, MAPTAIL_GEN, 24, 3, 10, 17,   24, 17, 10, 3);
    M4STM(p, st, cp, MAPTAIL_GEN,  4, 11, 18, 25,   4, 25, 18, 11);
    M4STM(p, st, cp, MAPTAIL_GEN, 12, 19, 26, 5,   12, 5, 26, 19);
    M4STM(p, st, cp, MAPTAIL_GEN, 20, 27, 6, 13,   20, 13, 6, 27);
}

AIN void dft35_pencil_map_swap(double *restrict p, const ptrdiff_t st,
                               const double *restrict cp)
{
    M7ST(a_, p, st,  0, 5, 10, 15, 20, 25, 30,   0, 15, 30, 10, 25, 5, 20);
    M7ST(b_, p, st,  7, 12, 17, 22, 27, 32, 2,   7, 22, 2, 17, 32, 12, 27);
    M7ST(c_, p, st, 14, 19, 24, 29, 34, 4, 9,   14, 29, 9, 24, 4, 19, 34);
    M7ST(d_, p, st, 21, 26, 31, 1, 6, 11, 16,   21, 1, 16, 31, 11, 26, 6);
    M7ST(e_, p, st, 28, 33, 3, 8, 13, 18, 23,   28, 8, 23, 3, 18, 33, 13);
    M5STMU(sa, p, st, cp, MAPTAIL_GEN,  0, 7, 14, 21, 28,   0, 21, 7, 28, 14);
    M5STMU(sb, p, st, cp, MAPTAIL_GEN, 15, 22, 29, 1, 8,   15, 1, 22, 8, 29);
    M5STMU(sc, p, st, cp, MAPTAIL_GEN, 30, 2, 9, 16, 23,   30, 16, 2, 23, 9);
    M5STMU(sd, p, st, cp, MAPTAIL_GEN, 10, 17, 24, 31, 3,  10, 31, 17, 3, 24);
    M5STMU(se, p, st, cp, MAPTAIL_GEN, 25, 32, 4, 11, 18,  25, 11, 32, 18, 4);
    M5STMU(sf, p, st, cp, MAPTAIL_GEN,  5, 12, 19, 26, 33,  5, 26, 12, 33, 19);
    M5STMU(sg, p, st, cp, MAPTAIL_GEN, 20, 27, 34, 6, 13,  20, 6, 27, 13, 34);
}

AIN void dft22_pencil_map_swap(double *restrict p, const ptrdiff_t st,
                               const double *restrict cp)
{
    M11ST(a_, p, st,  0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20,
                      0, 12, 2, 14, 4, 16, 6, 18, 8, 20, 10);
    M11ST(b_, p, st, 11, 13, 15, 17, 19, 21, 1, 3, 5, 7, 9,
                     11, 1, 13, 3, 15, 5, 17, 7, 19, 9, 21);
    M2STM(p, st, cp, MAPTAIL_GEN,  0, 11,    0, 11);
    M2STM(p, st, cp, MAPTAIL_GEN, 12, 1,    12, 1);
    M2STM(p, st, cp, MAPTAIL_GEN,  2, 13,    2, 13);
    M2STM(p, st, cp, MAPTAIL_GEN, 14, 3,    14, 3);
    M2STM(p, st, cp, MAPTAIL_GEN,  4, 15,    4, 15);
    M2STM(p, st, cp, MAPTAIL_GEN, 16, 5,    16, 5);
    M2STM(p, st, cp, MAPTAIL_GEN,  6, 17,    6, 17);
    M2STM(p, st, cp, MAPTAIL_GEN, 18, 7,    18, 7);
    M2STM(p, st, cp, MAPTAIL_GEN,  8, 19,    8, 19);
    M2STM(p, st, cp, MAPTAIL_GEN, 20, 9,    20, 9);
    M2STM(p, st, cp, MAPTAIL_GEN, 10, 21,   10, 21);
}

AIN void dft33_pencil_map_swap(double *restrict p, const ptrdiff_t st,
                               const double *restrict cp)
{
    M11ST(a_, p, st,  0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30,
                      0, 12, 24, 3, 15, 27, 6, 18, 30, 9, 21);
    M11ST(b_, p, st, 11, 14, 17, 20, 23, 26, 29, 32, 2, 5, 8,
                     11, 23, 2, 14, 26, 5, 17, 29, 8, 20, 32);
    M11ST(c_, p, st, 22, 25, 28, 31, 1, 4, 7, 10, 13, 16, 19,
                     22, 1, 13, 25, 4, 16, 28, 7, 19, 31, 10);
    M3STM(p, st, cp, MAPTAIL_GEN,  0, 11, 22,    0, 22, 11);
    M3STM(p, st, cp, MAPTAIL_GEN, 12, 23, 1,    12, 1, 23);
    M3STM(p, st, cp, MAPTAIL_GEN, 24, 2, 13,    24, 13, 2);
    M3STM(p, st, cp, MAPTAIL_GEN,  3, 14, 25,    3, 25, 14);
    M3STM(p, st, cp, MAPTAIL_GEN, 15, 26, 4,    15, 4, 26);
    M3STM(p, st, cp, MAPTAIL_GEN, 27, 5, 16,    27, 16, 5);
    M3STM(p, st, cp, MAPTAIL_GEN,  6, 17, 28,    6, 28, 17);
    M3STM(p, st, cp, MAPTAIL_GEN, 18, 29, 7,    18, 7, 29);
    M3STM(p, st, cp, MAPTAIL_GEN, 30, 8, 19,    30, 19, 8);
    M3STM(p, st, cp, MAPTAIL_GEN,  9, 20, 31,    9, 31, 20);
    M3STM(p, st, cp, MAPTAIL_GEN, 21, 32, 10,   21, 10, 32);
}

AIN void dft44_pencil_map_swap(double *restrict p, const ptrdiff_t st,
                               const double *restrict cp)
{
    M11ST(a_, p, st,  0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40,
                      0, 12, 24, 36, 4, 16, 28, 40, 8, 20, 32);
    M11ST(b_, p, st, 11, 15, 19, 23, 27, 31, 35, 39, 43, 3, 7,
                     11, 23, 35, 3, 15, 27, 39, 7, 19, 31, 43);
    M11ST(c_, p, st, 22, 26, 30, 34, 38, 42, 2, 6, 10, 14, 18,
                     22, 34, 2, 14, 26, 38, 6, 18, 30, 42, 10);
    M11ST(d_, p, st, 33, 37, 41, 1, 5, 9, 13, 17, 21, 25, 29,
                     33, 1, 13, 25, 37, 5, 17, 29, 41, 9, 21);
    M4STM(p, st, cp, MAPTAIL_GEN,  0, 11, 22, 33,    0, 33, 22, 11);
    M4STM(p, st, cp, MAPTAIL_GEN, 12, 23, 34, 1,    12, 1, 34, 23);
    M4STM(p, st, cp, MAPTAIL_GEN, 24, 35, 2, 13,    24, 13, 2, 35);
    M4STM(p, st, cp, MAPTAIL_GEN, 36, 3, 14, 25,    36, 25, 14, 3);
    M4STM(p, st, cp, MAPTAIL_GEN,  4, 15, 26, 37,    4, 37, 26, 15);
    M4STM(p, st, cp, MAPTAIL_GEN, 16, 27, 38, 5,    16, 5, 38, 27);
    M4STM(p, st, cp, MAPTAIL_GEN, 28, 39, 6, 17,    28, 17, 6, 39);
    M4STM(p, st, cp, MAPTAIL_GEN, 40, 7, 18, 29,    40, 29, 18, 7);
    M4STM(p, st, cp, MAPTAIL_GEN,  8, 19, 30, 41,    8, 41, 30, 19);
    M4STM(p, st, cp, MAPTAIL_GEN, 20, 31, 42, 9,    20, 9, 42, 31);
    M4STM(p, st, cp, MAPTAIL_GEN, 32, 43, 10, 21,   32, 21, 10, 43);
}

AIN void dft55_pencil_map_swap(double *restrict p, const ptrdiff_t st,
                               const double *restrict cp)
{
    M11ST(a_, p, st,  0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                      0, 45, 35, 25, 15, 5, 50, 40, 30, 20, 10);
    M11ST(b_, p, st, 11, 16, 21, 26, 31, 36, 41, 46, 51, 1, 6,
                     11, 1, 46, 36, 26, 16, 6, 51, 41, 31, 21);
    M11ST(c_, p, st, 22, 27, 32, 37, 42, 47, 52, 2, 7, 12, 17,
                     22, 12, 2, 47, 37, 27, 17, 7, 52, 42, 32);
    M11ST(d_, p, st, 33, 38, 43, 48, 53, 3, 8, 13, 18, 23, 28,
                     33, 23, 13, 3, 48, 38, 28, 18, 8, 53, 43);
    M11ST(e_, p, st, 44, 49, 54, 4, 9, 14, 19, 24, 29, 34, 39,
                     44, 34, 24, 14, 4, 49, 39, 29, 19, 9, 54);
    M5STMU(sa, p, st, cp, MAPTAIL_GEN,  0, 11, 22, 33, 44,   0, 11, 22, 33, 44);
    M5STMU(sb, p, st, cp, MAPTAIL_GEN, 45, 1, 12, 23, 34,   45, 1, 12, 23, 34);
    M5STMU(sc, p, st, cp, MAPTAIL_GEN, 35, 46, 2, 13, 24,   35, 46, 2, 13, 24);
    M5STMU(sd, p, st, cp, MAPTAIL_GEN, 25, 36, 47, 3, 14,   25, 36, 47, 3, 14);
    M5STMU(se, p, st, cp, MAPTAIL_GEN, 15, 26, 37, 48, 4,   15, 26, 37, 48, 4);
    M5STMU(sf, p, st, cp, MAPTAIL_GEN,  5, 16, 27, 38, 49,   5, 16, 27, 38, 49);
    M5STMU(sg, p, st, cp, MAPTAIL_GEN, 50, 6, 17, 28, 39,   50, 6, 17, 28, 39);
    M5STMU(sh, p, st, cp, MAPTAIL_GEN, 40, 51, 7, 18, 29,   40, 51, 7, 18, 29);
    M5STMU(si, p, st, cp, MAPTAIL_GEN, 30, 41, 52, 8, 19,   30, 41, 52, 8, 19);
    M5STMU(sj, p, st, cp, MAPTAIL_GEN, 20, 31, 42, 53, 9,   20, 31, 42, 53, 9);
    M5STMU(sk, p, st, cp, MAPTAIL_GEN, 10, 21, 32, 43, 54,  10, 21, 32, 43, 54);
}

/* ==== gen_r11: the L=100 pencil -- PFA(4 x 25) with the DFT25 = 5x5 CT above.
 * ALL HANDS ON L=100 round; this is the brief's approach #4 engine (within-
 * volume SoA) and the module it runs.  Outer PFA: n = (4a + 25b) % 100,
 * stage 1 = 4 x DFT25 over a (map-free, wide factor first -- the r9/r10
 * factor-swap verdict), safe placement: output d of group b stored at slot
 * (4*((19d)%25) + 25b) % 100 (19 = 4^-1 mod 25), so stage 2 = 25 x DFT4
 * over b on slots {(76d+25b)%100}, NATURALLY in place with natural output
 * order (k = (25c+76d)%100 lands at input position c), map fused there --
 * 4 ladders per codelet, the winning stage-2 width from r9/r10.  Slot
 * tables + the exact macro-semantics module were simulated in Python
 * against numpy (3 seeds, rel L2 3.6e-16; build/tryout/gen_batchlane/
 * gen100.py) before any C ran -- the r8/r9/r10 method.
 * Cost per pencil per 8 lanes: 4 x DFT25 (404 = 10 lifted DFT5 + 16
 * compiled cmults) + 25 x DFT4 (16) = 2016 vector FP, zero shuffles. */
/* W25 twiddles for the 5x5 CT inside DFT25 (gen_r11): w25^m = e^{-2pi i m/25},
 * KW25R_m = cos(2pi m/25), KW25I_m = -sin(2pi m/25); exact to the last bit
 * (Decimal Machin pi + Taylor series, 60 digits; cross-checked vs libm). */
static const v8d KW25R_1 = V8C(0.96858316112863108);
static const v8d KW25I_1 = V8C(-0.24868988716485479);
static const v8d KW25R_2 = V8C(0.87630668004386358);
static const v8d KW25I_2 = V8C(-0.48175367410171527);
static const v8d KW25R_3 = V8C(0.72896862742141155);
static const v8d KW25I_3 = V8C(-0.68454710592868873);
static const v8d KW25R_4 = V8C(0.53582679497899666);
static const v8d KW25I_4 = V8C(-0.84432792550201508);
static const v8d KW25R_6 = V8C(0.062790519529313374);
static const v8d KW25I_6 = V8C(-0.99802672842827156);
static const v8d KW25R_8 = V8C(-0.42577929156507266);
static const v8d KW25I_8 = V8C(-0.90482705246601958);
static const v8d KW25R_9 = V8C(-0.63742398974868975);
static const v8d KW25I_9 = V8C(-0.77051324277578925);
static const v8d KW25R_12 = V8C(-0.99211470131447788);
static const v8d KW25I_12 = V8C(-0.12533323356430426);
static const v8d KW25R_16 = V8C(-0.63742398974868975);
static const v8d KW25I_16 = V8C(0.77051324277578925);

AIN void dft100_pencil(double *restrict p, const ptrdiff_t st)
{
    double sc[25 * 16] __attribute__((aligned(64)));
    M5CTA0(a_, p, st, sc, 0, 0, 20, 40, 60, 80);
    M5CTA(b_, p, st, sc, 1, 4, 24, 44, 64, 84,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 8, 28, 48, 68, 88,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 12, 32, 52, 72, 92,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 16, 36, 56, 76, 96,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 0, 80, 60, 40, 20);
    M5CTB(g_, p, st, sc, 1, 76, 56, 36, 16, 96);
    M5CTB(h_, p, st, sc, 2, 52, 32, 12, 92, 72);
    M5CTB(i_, p, st, sc, 3, 28, 8, 88, 68, 48);
    M5CTB(j_, p, st, sc, 4, 4, 84, 64, 44, 24);
    M5CTA0(a_, p, st, sc, 0, 25, 45, 65, 85, 5);
    M5CTA(b_, p, st, sc, 1, 29, 49, 69, 89, 9,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 33, 53, 73, 93, 13,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 37, 57, 77, 97, 17,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 41, 61, 81, 1, 21,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 25, 5, 85, 65, 45);
    M5CTB(g_, p, st, sc, 1, 1, 81, 61, 41, 21);
    M5CTB(h_, p, st, sc, 2, 77, 57, 37, 17, 97);
    M5CTB(i_, p, st, sc, 3, 53, 33, 13, 93, 73);
    M5CTB(j_, p, st, sc, 4, 29, 9, 89, 69, 49);
    M5CTA0(a_, p, st, sc, 0, 50, 70, 90, 10, 30);
    M5CTA(b_, p, st, sc, 1, 54, 74, 94, 14, 34,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 58, 78, 98, 18, 38,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 62, 82, 2, 22, 42,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 66, 86, 6, 26, 46,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 50, 30, 10, 90, 70);
    M5CTB(g_, p, st, sc, 1, 26, 6, 86, 66, 46);
    M5CTB(h_, p, st, sc, 2, 2, 82, 62, 42, 22);
    M5CTB(i_, p, st, sc, 3, 78, 58, 38, 18, 98);
    M5CTB(j_, p, st, sc, 4, 54, 34, 14, 94, 74);
    M5CTA0(a_, p, st, sc, 0, 75, 95, 15, 35, 55);
    M5CTA(b_, p, st, sc, 1, 79, 99, 19, 39, 59,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 83, 3, 23, 43, 63,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 87, 7, 27, 47, 67,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 91, 11, 31, 51, 71,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 75, 55, 35, 15, 95);
    M5CTB(g_, p, st, sc, 1, 51, 31, 11, 91, 71);
    M5CTB(h_, p, st, sc, 2, 27, 7, 87, 67, 47);
    M5CTB(i_, p, st, sc, 3, 3, 83, 63, 43, 23);
    M5CTB(j_, p, st, sc, 4, 79, 59, 39, 19, 99);
    M4IP(p, st, 0, 25, 50, 75);
    M4IP(p, st, 76, 1, 26, 51);
    M4IP(p, st, 52, 77, 2, 27);
    M4IP(p, st, 28, 53, 78, 3);
    M4IP(p, st, 4, 29, 54, 79);
    M4IP(p, st, 80, 5, 30, 55);
    M4IP(p, st, 56, 81, 6, 31);
    M4IP(p, st, 32, 57, 82, 7);
    M4IP(p, st, 8, 33, 58, 83);
    M4IP(p, st, 84, 9, 34, 59);
    M4IP(p, st, 60, 85, 10, 35);
    M4IP(p, st, 36, 61, 86, 11);
    M4IP(p, st, 12, 37, 62, 87);
    M4IP(p, st, 88, 13, 38, 63);
    M4IP(p, st, 64, 89, 14, 39);
    M4IP(p, st, 40, 65, 90, 15);
    M4IP(p, st, 16, 41, 66, 91);
    M4IP(p, st, 92, 17, 42, 67);
    M4IP(p, st, 68, 93, 18, 43);
    M4IP(p, st, 44, 69, 94, 19);
    M4IP(p, st, 20, 45, 70, 95);
    M4IP(p, st, 96, 21, 46, 71);
    M4IP(p, st, 72, 97, 22, 47);
    M4IP(p, st, 48, 73, 98, 23);
    M4IP(p, st, 24, 49, 74, 99);
}

AIN void dft100_pencil_map(double *restrict p, const ptrdiff_t st,
                           const double *restrict cp)
{
    double sc[25 * 16] __attribute__((aligned(64)));
    M5CTA0(a_, p, st, sc, 0, 0, 20, 40, 60, 80);
    M5CTA(b_, p, st, sc, 1, 4, 24, 44, 64, 84,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 8, 28, 48, 68, 88,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 12, 32, 52, 72, 92,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 16, 36, 56, 76, 96,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 0, 80, 60, 40, 20);
    M5CTB(g_, p, st, sc, 1, 76, 56, 36, 16, 96);
    M5CTB(h_, p, st, sc, 2, 52, 32, 12, 92, 72);
    M5CTB(i_, p, st, sc, 3, 28, 8, 88, 68, 48);
    M5CTB(j_, p, st, sc, 4, 4, 84, 64, 44, 24);
    M5CTA0(a_, p, st, sc, 0, 25, 45, 65, 85, 5);
    M5CTA(b_, p, st, sc, 1, 29, 49, 69, 89, 9,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 33, 53, 73, 93, 13,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 37, 57, 77, 97, 17,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 41, 61, 81, 1, 21,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 25, 5, 85, 65, 45);
    M5CTB(g_, p, st, sc, 1, 1, 81, 61, 41, 21);
    M5CTB(h_, p, st, sc, 2, 77, 57, 37, 17, 97);
    M5CTB(i_, p, st, sc, 3, 53, 33, 13, 93, 73);
    M5CTB(j_, p, st, sc, 4, 29, 9, 89, 69, 49);
    M5CTA0(a_, p, st, sc, 0, 50, 70, 90, 10, 30);
    M5CTA(b_, p, st, sc, 1, 54, 74, 94, 14, 34,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 58, 78, 98, 18, 38,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 62, 82, 2, 22, 42,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 66, 86, 6, 26, 46,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 50, 30, 10, 90, 70);
    M5CTB(g_, p, st, sc, 1, 26, 6, 86, 66, 46);
    M5CTB(h_, p, st, sc, 2, 2, 82, 62, 42, 22);
    M5CTB(i_, p, st, sc, 3, 78, 58, 38, 18, 98);
    M5CTB(j_, p, st, sc, 4, 54, 34, 14, 94, 74);
    M5CTA0(a_, p, st, sc, 0, 75, 95, 15, 35, 55);
    M5CTA(b_, p, st, sc, 1, 79, 99, 19, 39, 59,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 83, 3, 23, 43, 63,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 87, 7, 27, 47, 67,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 91, 11, 31, 51, 71,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 75, 55, 35, 15, 95);
    M5CTB(g_, p, st, sc, 1, 51, 31, 11, 91, 71);
    M5CTB(h_, p, st, sc, 2, 27, 7, 87, 67, 47);
    M5CTB(i_, p, st, sc, 3, 3, 83, 63, 43, 23);
    M5CTB(j_, p, st, sc, 4, 79, 59, 39, 19, 99);
    M4STM(p, st, cp, MAPTAIL_GEN, 0, 25, 50, 75,   0, 25, 50, 75);
    M4STM(p, st, cp, MAPTAIL_GEN, 76, 1, 26, 51,   76, 1, 26, 51);
    M4STM(p, st, cp, MAPTAIL_GEN, 52, 77, 2, 27,   52, 77, 2, 27);
    M4STM(p, st, cp, MAPTAIL_GEN, 28, 53, 78, 3,   28, 53, 78, 3);
    M4STM(p, st, cp, MAPTAIL_GEN, 4, 29, 54, 79,   4, 29, 54, 79);
    M4STM(p, st, cp, MAPTAIL_GEN, 80, 5, 30, 55,   80, 5, 30, 55);
    M4STM(p, st, cp, MAPTAIL_GEN, 56, 81, 6, 31,   56, 81, 6, 31);
    M4STM(p, st, cp, MAPTAIL_GEN, 32, 57, 82, 7,   32, 57, 82, 7);
    M4STM(p, st, cp, MAPTAIL_GEN, 8, 33, 58, 83,   8, 33, 58, 83);
    M4STM(p, st, cp, MAPTAIL_GEN, 84, 9, 34, 59,   84, 9, 34, 59);
    M4STM(p, st, cp, MAPTAIL_GEN, 60, 85, 10, 35,   60, 85, 10, 35);
    M4STM(p, st, cp, MAPTAIL_GEN, 36, 61, 86, 11,   36, 61, 86, 11);
    M4STM(p, st, cp, MAPTAIL_GEN, 12, 37, 62, 87,   12, 37, 62, 87);
    M4STM(p, st, cp, MAPTAIL_GEN, 88, 13, 38, 63,   88, 13, 38, 63);
    M4STM(p, st, cp, MAPTAIL_GEN, 64, 89, 14, 39,   64, 89, 14, 39);
    M4STM(p, st, cp, MAPTAIL_GEN, 40, 65, 90, 15,   40, 65, 90, 15);
    M4STM(p, st, cp, MAPTAIL_GEN, 16, 41, 66, 91,   16, 41, 66, 91);
    M4STM(p, st, cp, MAPTAIL_GEN, 92, 17, 42, 67,   92, 17, 42, 67);
    M4STM(p, st, cp, MAPTAIL_GEN, 68, 93, 18, 43,   68, 93, 18, 43);
    M4STM(p, st, cp, MAPTAIL_GEN, 44, 69, 94, 19,   44, 69, 94, 19);
    M4STM(p, st, cp, MAPTAIL_GEN, 20, 45, 70, 95,   20, 45, 70, 95);
    M4STM(p, st, cp, MAPTAIL_GEN, 96, 21, 46, 71,   96, 21, 46, 71);
    M4STM(p, st, cp, MAPTAIL_GEN, 72, 97, 22, 47,   72, 97, 22, 47);
    M4STM(p, st, cp, MAPTAIL_GEN, 48, 73, 98, 23,   48, 73, 98, 23);
    M4STM(p, st, cp, MAPTAIL_GEN, 24, 49, 74, 99,   24, 49, 74, 99);
}

/* ==== gen_r11: the L=50 pencil -- PFA(2 x 25), the DFT25 machinery reused.
 * n = (25a + 2b) % 50: stage 1 = 25 x DFT2, NATURALLY in place (slot parity
 * IS the output index parity), stage 2 = 2 x DFT25 on the parity classes
 * {(25c+2b)%50}, CT output d placed at position 13d mod 25 (13 = 2^-1 mod
 * 25) -> site k = (25c+26d)%50.  The map fuses in the DFT25 stage-B stores
 * (M5CTBM, 5 ladders/codelet).  Tables simulated vs numpy (3 seeds,
 * 3.4e-16; build/tryout/gen_batchlane/gen50.py).
 * Cost per pencil per 8 lanes: 25 x DFT2 (8) + 2 x DFT25 (404) = 1008
 * vector FP, zero shuffles. */
AIN void dft50_pencil(double *restrict p, const ptrdiff_t st)
{
    double sc[25 * 16] __attribute__((aligned(64)));
    M2IP(p, st, 0, 25);
    M2IP(p, st, 2, 27);
    M2IP(p, st, 4, 29);
    M2IP(p, st, 6, 31);
    M2IP(p, st, 8, 33);
    M2IP(p, st, 10, 35);
    M2IP(p, st, 12, 37);
    M2IP(p, st, 14, 39);
    M2IP(p, st, 16, 41);
    M2IP(p, st, 18, 43);
    M2IP(p, st, 20, 45);
    M2IP(p, st, 22, 47);
    M2IP(p, st, 24, 49);
    M2IP(p, st, 26, 1);
    M2IP(p, st, 28, 3);
    M2IP(p, st, 30, 5);
    M2IP(p, st, 32, 7);
    M2IP(p, st, 34, 9);
    M2IP(p, st, 36, 11);
    M2IP(p, st, 38, 13);
    M2IP(p, st, 40, 15);
    M2IP(p, st, 42, 17);
    M2IP(p, st, 44, 19);
    M2IP(p, st, 46, 21);
    M2IP(p, st, 48, 23);
    M5CTA0(a_, p, st, sc, 0, 0, 10, 20, 30, 40);
    M5CTA(b_, p, st, sc, 1, 2, 12, 22, 32, 42,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 4, 14, 24, 34, 44,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 6, 16, 26, 36, 46,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 8, 18, 28, 38, 48,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 0, 30, 10, 40, 20);
    M5CTB(g_, p, st, sc, 1, 26, 6, 36, 16, 46);
    M5CTB(h_, p, st, sc, 2, 2, 32, 12, 42, 22);
    M5CTB(i_, p, st, sc, 3, 28, 8, 38, 18, 48);
    M5CTB(j_, p, st, sc, 4, 4, 34, 14, 44, 24);
    M5CTA0(a_, p, st, sc, 0, 25, 35, 45, 5, 15);
    M5CTA(b_, p, st, sc, 1, 27, 37, 47, 7, 17,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 29, 39, 49, 9, 19,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 31, 41, 1, 11, 21,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 33, 43, 3, 13, 23,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 25, 5, 35, 15, 45);
    M5CTB(g_, p, st, sc, 1, 1, 31, 11, 41, 21);
    M5CTB(h_, p, st, sc, 2, 27, 7, 37, 17, 47);
    M5CTB(i_, p, st, sc, 3, 3, 33, 13, 43, 23);
    M5CTB(j_, p, st, sc, 4, 29, 9, 39, 19, 49);
}

AIN void dft50_pencil_map(double *restrict p, const ptrdiff_t st,
                          const double *restrict cp)
{
    double sc[25 * 16] __attribute__((aligned(64)));
    M2IP(p, st, 0, 25);
    M2IP(p, st, 2, 27);
    M2IP(p, st, 4, 29);
    M2IP(p, st, 6, 31);
    M2IP(p, st, 8, 33);
    M2IP(p, st, 10, 35);
    M2IP(p, st, 12, 37);
    M2IP(p, st, 14, 39);
    M2IP(p, st, 16, 41);
    M2IP(p, st, 18, 43);
    M2IP(p, st, 20, 45);
    M2IP(p, st, 22, 47);
    M2IP(p, st, 24, 49);
    M2IP(p, st, 26, 1);
    M2IP(p, st, 28, 3);
    M2IP(p, st, 30, 5);
    M2IP(p, st, 32, 7);
    M2IP(p, st, 34, 9);
    M2IP(p, st, 36, 11);
    M2IP(p, st, 38, 13);
    M2IP(p, st, 40, 15);
    M2IP(p, st, 42, 17);
    M2IP(p, st, 44, 19);
    M2IP(p, st, 46, 21);
    M2IP(p, st, 48, 23);
    M5CTA0(a_, p, st, sc, 0, 0, 10, 20, 30, 40);
    M5CTA(b_, p, st, sc, 1, 2, 12, 22, 32, 42,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 4, 14, 24, 34, 44,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 6, 16, 26, 36, 46,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 8, 18, 28, 38, 48,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTBM(f_, p, st, sc, 0, cp, 0, 30, 10, 40, 20);
    M5CTBM(g_, p, st, sc, 1, cp, 26, 6, 36, 16, 46);
    M5CTBM(h_, p, st, sc, 2, cp, 2, 32, 12, 42, 22);
    M5CTBM(i_, p, st, sc, 3, cp, 28, 8, 38, 18, 48);
    M5CTBM(j_, p, st, sc, 4, cp, 4, 34, 14, 44, 24);
    M5CTA0(a_, p, st, sc, 0, 25, 35, 45, 5, 15);
    M5CTA(b_, p, st, sc, 1, 27, 37, 47, 7, 17,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 29, 39, 49, 9, 19,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 31, 41, 1, 11, 21,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 33, 43, 3, 13, 23,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTBM(f_, p, st, sc, 0, cp, 25, 5, 35, 15, 45);
    M5CTBM(g_, p, st, sc, 1, cp, 1, 31, 11, 41, 21);
    M5CTBM(h_, p, st, sc, 2, cp, 27, 7, 37, 17, 47);
    M5CTBM(i_, p, st, sc, 3, cp, 3, 33, 13, 43, 23);
    M5CTBM(j_, p, st, sc, 4, cp, 29, 9, 39, 19, 49);
}

/* Pressure-aware scheduling, re-raced r4 under the same-core interleaved
 * protocol (see strategy record: tryout.sh leases a fresh slot per
 * invocation, so cross-invocation A/B pairs can land on different cores in
 * different turbo states -- r1-r3 sched verdicts had that confound).
 * Same-core verdicts: 10: attr wins 1.159 vs 1.218 (-4.8%); 12: attr wins
 * 1.921 vs 2.037 (-5.7%); 15: attr LOSES 4.77-4.88 vs 4.57-4.61 (+4.3%,
 * six consecutive interleaved pairs) -- r3 had it backwards at 15.  The
 * 15-family now defaults to the stock scheduler; -DBL_SCHED15 re-enables
 * (cross-arch race knob), -DBL_NOSCHED1012 strips 10/12. */
#define SCHEDP __attribute__((optimize("schedule-insns", "sched-pressure")))
#if defined(BL_SCHED15)
#define SCHED15 SCHEDP
#else
#define SCHED15
#endif
#if defined(BL_NOSCHED1012)
#define SCHED1012
#else
#define SCHED1012 SCHEDP
#endif

/* ---- length-L pencil DFTs, PFA slot maps baked in (see file header) ----
 * L=10: n=(5a+2b)%10, k=(5c+6d)%10.  L=12: n=(4a+3b)%12, k=(4c+9d)%12.
 * L=15: n=(5a+3b)%15, k=(10c+6d)%15.
 * always_inline (no optimize attr here -- the caller's SCHED attr governs the
 * inlined body; an optimize-attr callee is what out-lined the r2 pencils). */
#define XDECL10 v8d xr0, xi0, xr1, xi1, xr2, xi2, xr3, xi3, xr4, xi4,          \
                    xr5, xi5, xr6, xi6, xr7, xi7, xr8, xi8, xr9, xi9
#define XDECL12 XDECL10, xr10, xi10, xr11, xi11
#define XDECL15 XDECL12, xr12, xi12, xr13, xi13, xr14, xi14

AIN void dft10_pencil(double *restrict p, const ptrdiff_t st)
{
    XDECL10;
    R2L(p, st, 0, 5); R2L(p, st, 2, 7); R2L(p, st, 4, 9);
    R2L(p, st, 6, 1); R2L(p, st, 8, 3);
    R5ST(a_, p, st, 0, 2, 4, 6, 8,  0, 6, 2, 8, 4);
    R5ST(b_, p, st, 5, 7, 9, 1, 3,  5, 1, 7, 3, 9);
}

AIN void dft10_pencil_map(double *restrict p, const ptrdiff_t st,
                          const double *restrict cp)
{
    XDECL10;
    R2L(p, st, 0, 5); R2L(p, st, 2, 7); R2L(p, st, 4, 9);
    R2L(p, st, 6, 1); R2L(p, st, 8, 3);
    R5STM(a_, p, st, cp, 0, 2, 4, 6, 8,  0, 6, 2, 8, 4);
    R5STM(b_, p, st, cp, 5, 7, 9, 1, 3,  5, 1, 7, 3, 9);
}

/* gen_r9 factor swap at 10: stage 1 = 2 lifted DFT5s (mem -> regs), stage 2 =
 * 5 DFT2+map (2 ladders per codelet vs the shipped DFT5+5). */
AIN void dft10_pencil_map_swap(double *restrict p, const ptrdiff_t st,
                               const double *restrict cp)
{
    XDECL10;
    R5L(a_, p, st, 0, 2, 4, 6, 8,  0, 6, 2, 8, 4);
    R5L(b_, p, st, 5, 7, 9, 1, 3,  5, 1, 7, 3, 9);
    R2STM(p, st, cp, 0, 5,  0, 5);
    R2STM(p, st, cp, 6, 1,  6, 1);
    R2STM(p, st, cp, 2, 7,  2, 7);
    R2STM(p, st, cp, 8, 3,  8, 3);
    R2STM(p, st, cp, 4, 9,  4, 9);
}

AIN void dft12_pencil(double *restrict p, const ptrdiff_t st)
{
    XDECL12;
    R3L(p, st, 0, 4, 8);  R3L(p, st, 3, 7, 11);
    R3L(p, st, 6, 10, 2); R3L(p, st, 9, 1, 5);
    R4ST(p, st, 0, 3, 6, 9,   0, 9, 6, 3);
    R4ST(p, st, 4, 7, 10, 1,  4, 1, 10, 7);
    R4ST(p, st, 8, 11, 2, 5,  8, 5, 2, 11);
}

AIN void dft12_pencil_map(double *restrict p, const ptrdiff_t st,
                          const double *restrict cp)
{
    XDECL12;
    R3L(p, st, 0, 4, 8);  R3L(p, st, 3, 7, 11);
    R3L(p, st, 6, 10, 2); R3L(p, st, 9, 1, 5);
    R4STM(p, st, cp, 0, 3, 6, 9,   0, 9, 6, 3);
    R4STM(p, st, cp, 4, 7, 10, 1,  4, 1, 10, 7);
    R4STM(p, st, cp, 8, 11, 2, 5,  8, 5, 2, 11);
}

/* gen_r9 factor swap at 12: stage 1 = 3 DFT4s (mem -> regs), stage 2 =
 * 4 DFT3+map (3 ladders per codelet vs the shipped DFT4+4). */
AIN void dft12_pencil_map_swap(double *restrict p, const ptrdiff_t st,
                               const double *restrict cp)
{
    XDECL12;
    R4L(p, st, 0, 3, 6, 9,   0, 9, 6, 3);
    R4L(p, st, 4, 7, 10, 1,  4, 1, 10, 7);
    R4L(p, st, 8, 11, 2, 5,  8, 5, 2, 11);
    R3STM(p, st, cp, 0, 4, 8,   0, 4, 8);
    R3STM(p, st, cp, 9, 1, 5,   9, 1, 5);
    R3STM(p, st, cp, 6, 10, 2,  6, 10, 2);
    R3STM(p, st, cp, 3, 7, 11,  3, 7, 11);
}

/* gen_r9 factor swap at 15 (memory form): stage 1 = 3 in-place lifted DFT5s
 * (the shipped stage-2 groups, safe placement sigma(c)=2c mod 5), stage 2 =
 * 5 DFT3+map (3 ladders per codelet vs the shipped X2 pair's 2 cores + 10). */
AIN void dft15_pencil_map_swap(double *restrict p, const ptrdiff_t st,
                               const double *restrict cp)
{
    M5ST(g0, p, st,  0, 3, 6, 9, 12,   0, 6, 12, 3, 9);
    M5ST(g1, p, st,  5, 8, 11, 14, 2,  5, 11, 2, 8, 14);
    M5ST(g2, p, st, 10, 13, 1, 4, 7,  10, 1, 7, 13, 4);
    M3STM(p, st, cp, MAPTAIL_15,  0, 5, 10,   0, 10, 5);
    M3STM(p, st, cp, MAPTAIL_15,  6, 11, 1,   6, 1, 11);
    M3STM(p, st, cp, MAPTAIL_15, 12, 2, 7,   12, 7, 2);
    M3STM(p, st, cp, MAPTAIL_15,  3, 8, 13,   3, 13, 8);
    M3STM(p, st, cp, MAPTAIL_15,  9, 14, 4,   9, 4, 14);
}

/* gen_r9 factor swap at 20 (memory form): stage 1 = 4 in-place lifted DFT5s
 * (sigma(c)=4c mod 5), stage 2 = 5 DFT4+map, groups naturally in place. */
AIN void dft20_pencil_map_swap(double *restrict p, const ptrdiff_t st,
                               const double *restrict cp)
{
    M5ST(g0, p, st,  0, 4, 8, 12, 16,   0, 16, 12, 8, 4);
    M5ST(g1, p, st,  5, 9, 13, 17, 1,   5, 1, 17, 13, 9);
    M5ST(g2, p, st, 10, 14, 18, 2, 6,  10, 6, 2, 18, 14);
    M5ST(g3, p, st, 15, 19, 3, 7, 11,  15, 11, 7, 3, 19);
    M4STM(p, st, cp, MAPTAIL_20,  0, 5, 10, 15,   0, 5, 10, 15);
    M4STM(p, st, cp, MAPTAIL_20, 16, 1, 6, 11,   16, 1, 6, 11);
    M4STM(p, st, cp, MAPTAIL_20, 12, 17, 2, 7,   12, 17, 2, 7);
    M4STM(p, st, cp, MAPTAIL_20,  8, 13, 18, 3,   8, 13, 18, 3);
    M4STM(p, st, cp, MAPTAIL_20,  4, 9, 14, 19,   4, 9, 14, 19);
}

/* L=15 stage-2 groups c=1,c=2 read/write equal SLOT sets, but stage 2 reads
 * only registers here, so no fusion or ordering hazard remains (r2's DFT5X2). */
AIN void dft15_pencil(double *restrict p, const ptrdiff_t st)
{
    XDECL15;
    R3L(p, st, 0, 5, 10);  R3L(p, st, 3, 8, 13); R3L(p, st, 6, 11, 1);
    R3L(p, st, 9, 14, 4);  R3L(p, st, 12, 2, 7);
    R5ST(a_, p, st, 0, 3, 6, 9, 12,   0, 6, 12, 3, 9);
    R5ST(b_, p, st, 5, 8, 11, 14, 2,  10, 1, 7, 13, 4);
    R5ST(c_, p, st, 10, 13, 1, 4, 7,   5, 11, 2, 8, 14);
}

AIN void dft15_pencil_map(double *restrict p, const ptrdiff_t st,
                          const double *restrict cp)
{
    XDECL15;
    R3L(p, st, 0, 5, 10);  R3L(p, st, 3, 8, 13); R3L(p, st, 6, 11, 1);
    R3L(p, st, 9, 14, 4);  R3L(p, st, 12, 2, 7);
    R5STM(a_, p, st, cp, 0, 3, 6, 9, 12,   0, 6, 12, 3, 9);
    R5STM(b_, p, st, cp, 5, 8, 11, 14, 2,  10, 1, 7, 13, 4);
    R5STM(c_, p, st, cp, 10, 13, 1, 4, 7,   5, 11, 2, 8, 14);
}

/* L=20: n=(5a+4b)%20, k=(5c+16d)%20.  Stage-1 group b reads/writes slots
 * {(5a+4b)%20}; stage-2 group c reads {(5c+4b)%20} in b order, writes
 * {(5c+16d)%20} at output d.  X[k] = sum_b W5^{bd} [sum_a W4^{ac} x_n]:
 * k == c mod 4 and == d mod 5, both exponent identities check out. */
AIN void dft20_pencil(double *restrict p, const ptrdiff_t st)
{
    M4IP(p, st, 0, 5, 10, 15);  M4IP(p, st, 4, 9, 14, 19);
    M4IP(p, st, 8, 13, 18, 3);  M4IP(p, st, 12, 17, 2, 7);
    M4IP(p, st, 16, 1, 6, 11);
    M5ST(a_, p, st,  0, 4, 8, 12, 16,   0, 16, 12, 8, 4);
    M5ST(b_, p, st,  5, 9, 13, 17, 1,   5, 1, 17, 13, 9);
    M5ST(c_, p, st, 10, 14, 18, 2, 6,  10, 6, 2, 18, 14);
    M5ST(d_, p, st, 15, 19, 3, 7, 11,  15, 11, 7, 3, 19);
}

AIN void dft20_pencil_map(double *restrict p, const ptrdiff_t st,
                          const double *restrict cp)
{
    M4IP(p, st, 0, 5, 10, 15);  M4IP(p, st, 4, 9, 14, 19);
    M4IP(p, st, 8, 13, 18, 3);  M4IP(p, st, 12, 17, 2, 7);
    M4IP(p, st, 16, 1, 6, 11);
    M5STM(a_, p, st, cp,  0, 4, 8, 12, 16,   0, 16, 12, 8, 4);
    M5STM(b_, p, st, cp,  5, 9, 13, 17, 1,   5, 1, 17, 13, 9);
    M5STM(c_, p, st, cp, 10, 14, 18, 2, 6,  10, 6, 2, 18, 14);
    M5STM(d_, p, st, cp, 15, 19, 3, 7, 11,  15, 11, 7, 3, 19);
}

/* per-size fused-vs-epilogue map placement (r4 A/B, see map_col comment);
 * override with -DBL_EPI<N>=0/1 for the cross-arch races */
#ifndef BL_EPI10
#define BL_EPI10 0
#endif
#ifndef BL_EPI12
#define BL_EPI12 0
#endif
#ifndef BL_EPI15
#define BL_EPI15 0
#endif
#ifndef BL_EPI20
#define BL_EPI20 0
#endif

/* ---- L=15 pencil-form selection (gen_r5).  BL_MEM15:
 *   0 = register-explicit both passes (the r4 ship; loses to 1 -- see above),
 *   1 = memory form both passes (gen_pfa_small's shipped 15),
 *   2 = HYBRID (default): register-explicit pencils in the map-free zy sweep
 *       (asm audit r4: they spill ZERO and take 2L ld + 2L st vs the memory
 *       form's 4L + 4L) + the memory-form fused-map pencil in the x-pass
 *       (where 30 site regs + ~7 map temps + constants are what spill). */
/* gen_r7: default flipped 1 -> 2.  The r5 hybrid verdict (+0.35%, rejected)
 * REVERSES under the lifted DFT5 v-pair: hybrid wins 9 of 12 same-core
 * pairs, 4.373-4.395 vs memory-form 4.385-4.411 (-0.25%) -- the lift's two
 * fewer live temps per DFT5 are enough to keep the register-explicit sweep
 * pencil spill-free where it previously tied.  Bit-transparent (same
 * arithmetic order both forms; verified by cmp on the m=600 chain). */
#ifndef BL_MEM15
#define BL_MEM15 2
#endif
/* gen_r6: BL_SAFE15=1 replaces the memory form's forced DFT5X2 fused pair
 * with three independent in-place DFT5 groups via the M3IPS store
 * permutation.  Bit-identical outputs (verified by cmp on the m=600 chain).
 * RACED same-core r6 and LOST: 4.458-4.654 vs fused 4.414-4.480 (+1%, five
 * interleaved rounds) -- the X2 pair's two interleaved dependency chains
 * (2 DFT5COREs + 2 map ladders in flight) beat three serial groups on ILP.
 * Default 0; knob kept for the cross-arch races. */
#ifndef BL_SAFE15
#define BL_SAFE15 0
#endif
#if BL_SAFE15 && BL_MEM15 != 0
#define DFT15_MEM_SWEEP dft15_pencil_mem_safe
#define DFT15_MEM_MAP   dft15_pencil_map_mem_safe
#else
#define DFT15_MEM_SWEEP dft15_pencil_mem
#define DFT15_MEM_MAP   dft15_pencil_map_mem
#endif
#if BL_MEM15 == 0
#define DFT15_SWEEP dft15_pencil
#define DFT15_MAP_BASE dft15_pencil_map
#elif BL_MEM15 == 1
#define DFT15_SWEEP DFT15_MEM_SWEEP
#define DFT15_MAP_BASE DFT15_MEM_MAP
#else
#define DFT15_SWEEP dft15_pencil
#define DFT15_MAP_BASE DFT15_MEM_MAP
#endif

/* ---- gen_r9 factor-swap knobs (see the M3STM block comment).  Swap applies
 * to the fused-map x-pencil ONLY; sweep pencils keep the shipped order.
 * Defaults from the gen_r9 same-core races (strategy record). */
/* gen_r9 same-core race verdicts (a81n2 core 2, interleaved --samples 4,
 * first invocation discarded): swap WINS at 10 (1.122-1.125 w/ div tail vs
 * ship 1.147-1.150, -2.1%), 15 (4.332-4.362 vs 4.377-4.392, -1.0%), 20
 * (13.237-13.281 vs 13.469-13.519, -1.6%); swap LOSES at 12 (2.005-2.009
 * rcp / 1.983-1.986 div vs ship 1.916-1.918) -- the shipped DFT4+map
 * stage-2's 4-ladder ILP beats the DFT3+map form there.  Sched attributes
 * re-raced on the new codelets: unchanged (on at 10, stock at 15/20). */
#ifndef BL_SWAP10
#define BL_SWAP10 1
#endif
#ifndef BL_SWAP12
#define BL_SWAP12 0
#endif
#ifndef BL_SWAP15
#define BL_SWAP15 1
#endif
#ifndef BL_SWAP20
#define BL_SWAP20 1
#endif
#if BL_SWAP10
#define DFT10_MAP dft10_pencil_map_swap
#else
#define DFT10_MAP dft10_pencil_map
#endif
#if BL_SWAP12
#define DFT12_MAP dft12_pencil_map_swap
#else
#define DFT12_MAP dft12_pencil_map
#endif
#if BL_SWAP15
#define DFT15_MAP dft15_pencil_map_swap
#else
#define DFT15_MAP DFT15_MAP_BASE
#endif
#if BL_SWAP20
#define DFT20_MAP dft20_pencil_map_swap
#else
#define DFT20_MAP dft20_pencil_map
#endif

/* ---- the sweeps, one instantiation per L (strides compile-time constant) ----
 * PL2 = padded plane stride in site-vectors, plane bytes == 256 (mod 4096).
 * PEN/PENM = plain / fused-map pencil functions for this size.
 * EPIV: 0 = map fused into the x-pencil stage-2 stores (r3 form),
 *       1 = spill-free plain x-pencil + map_col epilogue per column (r4). */
#define DEF_ENGINE(N, PL2V, ATTR, EPIV, MT, PEN, PENM)                         \
static ATTR void sweep_zy_##N(double *restrict pl)                             \
{                                                                              \
    for (int y = 0; y < N; ++y)                                                \
        PEN(pl + (size_t)y * (N * 16), 16);                                    \
    for (int z = 0; z < N; ++z)                                                \
        PEN(pl + (size_t)z * 16, N * 16);                                      \
}                                                                              \
static ATTR void xpass_##N(double *restrict S)                                 \
{                                                                              \
    for (int c = 0; c < N * N; ++c)                                            \
        PEN(S + (size_t)c * 16, PL2V * 16);                                    \
}                                                                              \
static ATTR void chainsteps_##N(double *restrict S, const double *restrict C,  \
                                int m)                                         \
{                                                                              \
    for (int s = 0; s < m; ++s) {                                              \
        for (int x = 0; x < N; ++x)                                            \
            sweep_zy_##N(S + (size_t)x * ((size_t)PL2V * 16));                 \
        if (EPIV) {                                                            \
            for (int c = 0; c < N * N; ++c) {                                  \
                PEN(S + (size_t)c * 16, PL2V * 16);                            \
                map_col(S + (size_t)c * 16, PL2V * 16,                         \
                        C + (size_t)c * 16, N, MT);                            \
            }                                                                  \
        } else {                                                               \
            for (int c = 0; c < N * N; ++c)                                    \
                PENM(S + (size_t)c * 16, PL2V * 16,                            \
                     C + (size_t)c * 16);                                      \
        }                                                                      \
    }                                                                          \
}

/* L=20 defaults to the STOCK scheduler: sched-pressure measured +4..8% there
 * (14.77-14.96 vs 14.35 us same window), matching gen_pfa_small's r2 verdict
 * on this memory-round-trip codelet shape.  -DBL_SCHED20 re-enables to race
 * it on other hosts. */
#if defined(BL_SCHED20)
#define SCHED20 SCHEDP
#else
#define SCHED20
#endif

DEF_ENGINE(10, 130, SCHED1012, BL_EPI10, MAPTAIL_REG, dft10_pencil, DFT10_MAP)
DEF_ENGINE(12, 162, SCHED1012, BL_EPI12, MAPTAIL_REG, dft12_pencil, DFT12_MAP)
DEF_ENGINE(15, 226, SCHED15,   BL_EPI15, MAPTAIL_15,  DFT15_SWEEP,  DFT15_MAP)
DEF_ENGINE(20, 418, SCHED20,   BL_EPI20, MAPTAIL_20,  dft20_pencil, DFT20_MAP)
/* gen_r6: the 7-smooth sizes, memory form, stock scheduler (the 15/20
 * verdict for this codelet shape), fused map, div tail.
 * gen_r7: BL_X214 selects the fused DFT7X2 stage-2 at 14 (raced this
 * round; see strategy record for the verdict and default). */
/* RACED gen_r7 and LOST: fused DFT7X2 3.83-4.01 vs serial 3.62-3.64 us
 * (+6-10%, five same-core pairs) -- 28 slot loads + two DFT7COREs' ~24
 * temps each spill far past 32 zmm; the DFT5X2-at-15 ILP win does NOT
 * transfer to modules this wide.  Default 0; knob kept for cross-arch. */
#ifndef BL_X214
#define BL_X214 0
#endif
/* gen_r10 factor-swap knobs for the wide-module family (see the swapped
 * pencil block).  Defaults from the gen_r10 same-core races on a81n2
 * (core 5, interleaved --samples 4 minima, first invocation discarded):
 * swap WINS at 21 (17.16-17.32 vs 17.48-17.62, -1.9%, 4/4), 22 (24.21-24.77
 * vs 24.70-24.81, -1.7%, 4/4), 28 (44.95-44.97 vs 46.80-46.89, -4.0%, 4/4),
 * 33 (88.04-88.54 vs 92.40-93.10, -4.7%, 4/4), 35 (96.06-96.25 vs
 * 98.54-98.75, -2.5%, 3/3), 44 (284.9 vs 288.2 quiet; 304-347 vs 374-382
 * under LLC contention, 6/6) and 55 (1089-1230 vs 1140-1241, 2/2, noisy
 * bandwidth-bound cell); swap LOSES at 14 (+0.3-0.4%, 6/7 pairs: 3.613-3.633
 * ctl vs 3.628-3.654 -- the DFT2 stage-2's two-ladder codelet is too thin,
 * the same tiny-stage-2 failure as BL_SWAP12).  Div map tail re-confirmed
 * on the swapped 21 codelet (rcp +1.7%, 2/3 clean pairs). */
#ifndef BL_SWAP14
#define BL_SWAP14 0
#endif
#ifndef BL_SWAP21
#define BL_SWAP21 1
#endif
#ifndef BL_SWAP28
#define BL_SWAP28 1
#endif
#ifndef BL_SWAP35
#define BL_SWAP35 1
#endif
#ifndef BL_SWAP22
#define BL_SWAP22 1
#endif
#ifndef BL_SWAP33
#define BL_SWAP33 1
#endif
#ifndef BL_SWAP44
#define BL_SWAP44 1
#endif
#ifndef BL_SWAP55
#define BL_SWAP55 1
#endif
#if BL_SWAP14
#define DFT14_MAP dft14_pencil_map_swap
#elif BL_X214
#define DFT14_MAP dft14_pencil_map_x2
#else
#define DFT14_MAP dft14_pencil_map
#endif
#if BL_SWAP21
#define DFT21_MAP dft21_pencil_map_swap
#else
#define DFT21_MAP dft21_pencil_map
#endif
#if BL_SWAP28
#define DFT28_MAP dft28_pencil_map_swap
#else
#define DFT28_MAP dft28_pencil_map
#endif
#if BL_SWAP35
#define DFT35_MAP dft35_pencil_map_swap
#else
#define DFT35_MAP dft35_pencil_map
#endif
#if BL_SWAP22
#define DFT22_MAP dft22_pencil_map_swap
#else
#define DFT22_MAP dft22_pencil_map
#endif
#if BL_SWAP33
#define DFT33_MAP dft33_pencil_map_swap
#else
#define DFT33_MAP dft33_pencil_map
#endif
#if BL_SWAP44
#define DFT44_MAP dft44_pencil_map_swap
#else
#define DFT44_MAP dft44_pencil_map
#endif
#if BL_SWAP55
#define DFT55_MAP dft55_pencil_map_swap
#else
#define DFT55_MAP dft55_pencil_map
#endif
#if BL_X214
DEF_ENGINE(14, 226,  , 0, MAPTAIL_GEN, dft14_pencil_x2, DFT14_MAP)
#else
DEF_ENGINE(14, 226,  , 0, MAPTAIL_GEN, dft14_pencil, DFT14_MAP)
#endif
DEF_ENGINE(21, 450,  , 0, MAPTAIL_GEN, dft21_pencil, DFT21_MAP)
DEF_ENGINE(28, 802,  , 0, MAPTAIL_GEN, dft28_pencil, DFT28_MAP)
DEF_ENGINE(35, 1250, , 0, MAPTAIL_GEN, dft35_pencil, DFT35_MAP)
/* gen_r8: the 11-smooth sizes, same memory-form defaults */
DEF_ENGINE(22, 514,  , 0, MAPTAIL_GEN, dft22_pencil, DFT22_MAP)
DEF_ENGINE(33, 1090, , 0, MAPTAIL_GEN, dft33_pencil, DFT33_MAP)
DEF_ENGINE(44, 1954, , 0, MAPTAIL_GEN, dft44_pencil, DFT44_MAP)
DEF_ENGINE(55, 3042, , 0, MAPTAIL_GEN, dft55_pencil, DFT55_MAP)

/* ==== gen_r11: the L=100 WITHIN-VOLUME engine (brief approach #4 -- the
 * batch-lane trick without a batch; the scored case is B=1).  The eight zmm
 * lanes carry eight X-PLANES of one volume instead of eight volumes: lane l
 * of site-vector (xb, y, z) is x = 8*xb + l.  Then
 *   - z-pencils are stride-1 sv runs and y-pencils stride-ZP100 sv runs,
 *     both ELEMENTWISE across lanes -- zero shuffles, zero twiddle swaps,
 *     zero spills where the AoS engines pay 569 shuffles + ~458 spill slots
 *     per z-line (gen_powp r8's port census of the shared pfa_large shell);
 *   - the two sweeps run back to back on one ~1.3 MB slab = the two-axes-
 *     per-pass fusion (one DRAM read+write for BOTH axes);
 *   - only the x-pass crosses lanes: per (y, 8z) column, 26 trans8 in +
 *     26 out around a scratch-pencil dft100 -- 1248 shuffles per 100-site
 *     column, hidden under the pass's DRAM streams;
 *   - the map fuses in-register at the x-pass stores as at every other size
 *     (gen_pfa_large runs it as a separate pass costing 35% of their step;
 *     their fused variants lost on AoS in-place streams -- this shape is
 *     the one their record says never got tried).
 * DRAM accounting matches gen_pfa_large's proven 80 MB/step floor (zy:
 * read+write state; x: read+write state + read c); the win is bought in
 * uops and L1 round trips, not passes.
 * Geometry: 13 x-slabs (slab 12 has 4 pad lanes, packed replicated, never
 * unpacked); row stride ZP100 = 101 sv (odd multiple -> 4K phases rotate);
 * slab stride SLST100 = 10114 sv (slab bytes == 256 mod 4096, house rule).
 * trans8 lane order: out reg j lane l = in reg lanex[l] element j, lanex =
 * {0,1,4,5,2,3,6,7} (measured, build/tryout/gen_batchlane/t8.c) -- scratch
 * slots stay NATURAL x order, scratch lanes are z-offsets in lanex order;
 * the c pack bakes the same lanex into its deinterleave shuffle, and the
 * inverse transpose is trans8 with lanex-permuted loads and stores. */
#define ZP100   101
#define SLST100 10114
#define CST100  101

static void sweep_zy_100(double *restrict slab)
{
    for (int y = 0; y < 100; ++y)
        dft100_pencil(slab + (size_t)y * (ZP100 * 16), 16);      /* z-pencils */
    for (int z = 0; z < 100; ++z)
        dft100_pencil(slab + (size_t)z * 16, ZP100 * 16);        /* y-pencils */
}

static const int LX8[8] = { 0, 1, 4, 5, 2, 3, 6, 7 };            /* lanex */

/* gather column (y, z=8zg..) into the 104-sv scratch pencil: slot s = x
 * (natural), lane l = z-offset lanex[l]; pad z (zg=12, dz>=4) enters as
 * zeros, pad x (slots 100..103) is whatever the pad lanes held (finite,
 * ignored by the module). */
AIN void xcol_load_100(const double *restrict S, double *restrict Pn,
                       int y, int zg, int ndz)
{
    const v8d vz = V8C(0.0);
    for (int xb = 0; xb < 13; ++xb) {
        const double *base = S + ((size_t)xb * SLST100 + (size_t)y * ZP100
                                  + (size_t)zg * 8) * 16;
        v8d r[8], q[8];
        for (int dz = 0; dz < 8; ++dz) {
            if (dz < ndz) {
                r[dz] = LD(base + (size_t)dz * 16);
                q[dz] = LD(base + (size_t)dz * 16 + 8);
            } else {
                r[dz] = vz;
                q[dz] = vz;
            }
        }
        trans8(r);
        trans8(q);
        /* pack_plane's slab convention is lane l = x-offset lanex[l], so
         * trans8's output register j carries x = 8xb + lanex[j]: store it at
         * slot lanex[j] and the scratch slots come out NATURAL x order. */
        double *dst = Pn + (size_t)xb * (8 * 16);
        for (int j = 0; j < 8; ++j) {
            ST(dst + (size_t)LX8[j] * 16, r[j]);
            ST(dst + (size_t)LX8[j] * 16 + 8, q[j]);
        }
    }
}

AIN void xcol_store_100(double *restrict S, const double *restrict Pn,
                        int y, int zg, int ndz)
{
    for (int xb = 0; xb < 13; ++xb) {
        const double *src = Pn + (size_t)xb * (8 * 16);
        v8d r[8], q[8];
        for (int i = 0; i < 8; ++i) {
            r[i] = LD(src + (size_t)i * 16);
            q[i] = LD(src + (size_t)i * 16 + 8);
        }
        trans8(r);
        trans8(q);
        double *base = S + ((size_t)xb * SLST100 + (size_t)y * ZP100
                            + (size_t)zg * 8) * 16;
        for (int j = 0; j < 8; ++j) {
            const int dz = LX8[j];
            if (dz < ndz) {
                ST(base + (size_t)dz * 16, r[j]);
                ST(base + (size_t)dz * 16 + 8, q[j]);
            }
        }
    }
}

/* BL_EPI100: 0 = map fused into the dft100 stage-2 stores (house law),
 * 1 = plain dft100 + map_col epilogue on the L1-hot scratch pencil
 * (raceable: gen_pfa_large's AoS engine wants the map SEPARATE there). */
#ifndef BL_EPI100
#define BL_EPI100 0
#endif

static void xpass_100(double *restrict S)
{
    double Pn[104 * 16] __attribute__((aligned(64)));
    for (int y = 0; y < 100; ++y)
        for (int zg = 0; zg < 13; ++zg) {
            const int ndz = (zg == 12) ? 4 : 8;
            xcol_load_100(S, Pn, y, zg, ndz);
            dft100_pencil(Pn, 16);
            xcol_store_100(S, Pn, y, zg, ndz);
        }
}

static void xpass_map_100(double *restrict S, const double *restrict CT)
{
    double Pn[104 * 16] __attribute__((aligned(64)));
    for (int y = 0; y < 100; ++y)
        for (int zg = 0; zg < 13; ++zg) {
            const int ndz = (zg == 12) ? 4 : 8;
            const double *cp = CT + ((size_t)y * 13 + zg) * (CST100 * 16);
            xcol_load_100(S, Pn, y, zg, ndz);
            if (BL_EPI100) {
                dft100_pencil(Pn, 16);
                map_col(Pn, 16, cp, 100, MAPTAIL_GEN);
            } else {
                dft100_pencil_map(Pn, 16, cp);
            }
            xcol_store_100(S, Pn, y, zg, ndz);
        }
}

static void pack_vol_100(const double *restrict vin, double *restrict S)
{
    for (int s = 0; s < 13; ++s) {
        const int nl = (s == 12) ? 4 : 8;
        for (int y = 0; y < 100; ++y) {
            const double *vp[8];
            for (int l = 0; l < 8; ++l) {
                const int x = 8 * s + (l < nl ? l : nl - 1);
                vp[l] = vin + ((size_t)x * 100 + y) * 200;
            }
            pack_plane(vp, S + ((size_t)s * SLST100 + (size_t)y * ZP100) * 16,
                       100);
        }
    }
}

static void unpack_vol_100(double *restrict vout, const double *restrict S)
{
    for (int s = 0; s < 13; ++s) {
        const int nl = (s == 12) ? 4 : 8;
        for (int y = 0; y < 100; ++y) {
            double *op[8];
            for (int l = 0; l < 8; ++l) {
                const int x = 8 * s + (l < nl ? l : nl - 1);
                op[l] = vout + ((size_t)x * 100 + y) * 200;
            }
            unpack_plane(op, S + ((size_t)s * SLST100 + (size_t)y * ZP100) * 16,
                         100, nl);
        }
    }
}

/* c packed ONCE per chain into x-pass consumption order: column (y, zg),
 * slot x, lane l = z = 8zg + lanex[l] -- the lanex is baked into the
 * deinterleave shuffle indices (re: 2*lanex[l], im: +1), so alignment with
 * the state scratch costs zero extra ops.  zg=12 pulls its high half from
 * a zero vector (indices >= 8 select operand b), which both avoids reading
 * past the caller's array and zeroes the pad lanes to match the state. */
static void pack_ct_100(const double *restrict c, double *restrict CT)
{
    const v8d vz = V8C(0.0);
    for (int y = 0; y < 100; ++y)
        for (int zg = 0; zg < 13; ++zg) {
            double *col = CT + ((size_t)y * 13 + zg) * (CST100 * 16);
            const size_t zb = (size_t)zg * 8;
            for (int x = 0; x < 100; ++x) {
                const double *src = c + (((size_t)x * 100 + y) * 100 + zb) * 2;
                const v8d a = LD(src);
                const v8d b = (zg == 12) ? vz : (v8d)LD(src + 8);
                ST(col + (size_t)x * 16, SH(a, b, 0, 2, 8, 10, 4, 6, 12, 14));
                ST(col + (size_t)x * 16 + 8, SH(a, b, 1, 3, 9, 11, 5, 7, 13, 15));
            }
        }
}

static void chainsteps_100(double *restrict S, const double *restrict CT, int m)
{
    for (int step = 0; step < m; ++step) {
        for (int s = 0; s < 13; ++s)
            sweep_zy_100(S + (size_t)s * (SLST100 * 16));
        xpass_map_100(S, CT);
    }
}

/* ==== gen_r12: the ONE-SWEEP FUSED chain step at L=100 (BL_FUSE100).
 * ADOPTED from gen_pow2 gen_r11 (their L=128 one-sweep fused step, -61%
 * demand DRAM reads / -14% wall, and their explicit transfer note naming
 * L=100 as the same regime): split the x-FFT into two CT stages ACROSS the
 * chain-step boundary so each step crosses DRAM once, not twice --
 *     x-stage-2(s) + map(s) + z(s+1) + y(s+1) + x-stage-1(s+1)
 * per 10-plane tile.  DRAM/step: state r+w once + c read (~50 MB) vs the
 * r11 two-pass engine's ~84 MB, against gen_pfa_large r11's measurement
 * that this cell is ~88% DRAM-BW-bound at ~20 GB/s.
 *
 * WHY CT 10x10 AND NOT THE r11 PFA 4x25: gen_pow2's label algebra needs a
 * tile's stage-2 OUTPUT set to be a stage-1 input GROUP of the next step.
 * Under PFA the two digit sets are CRT-orthogonal (a tile's outputs are one
 * mod-4 class; every DFT25 stage-1 group needs all four classes), so no
 * tile below the whole volume closes -- PFA cannot fuse across steps.  CT
 * with EQUAL radices can: 100 = 10x10 is the exact analogue of their
 * involutive 8x8 at L=64.  x-DFT100 = 10 DFT10 (stage 1 over j, n=10j+b)
 * x w100^(b*k0) twiddles x 10 DFT10 (stage 2 over b) -- +150 vector FP per
 * pencil vs the PFA (2166 vs 2016), irrelevant at a BW-bound cell.  The 36
 * w100^(b*k0) products are COMPILED-IN broadcast constants (lit 11 Tier 1
 * routing, the r11 w25 precedent), exact to the last bit (60-digit Decimal
 * Machin pi + Taylor, cross-checked vs libm; w100^25 = -i exactly).
 *
 * LAYOUT (new for the fused engine; the r11 slab layout stays for
 * execute()): state = 100 x-planes, each (y, z) row-major, site-vector = 8
 * CONSECUTIVE Z of one row (re8|im8), row = FROW100 = 13 sv (z pad slots
 * 100..103), plane stride FPS100 = 1314 sv (plane bytes == 256 mod 4096,
 * house rule).  Within a plane both in-plane passes reuse the r11 dft100
 * module: y-pencils run it at row stride (shuffle-free), z-pencils through
 * a trans8-bracketed 104-sv scratch (the r11 xcol pattern; lanes here are
 * NATURAL because the LX8 permutation is folded into the gather row/slot
 * order -- lanex is self-inverse).  The x stages are VERTICAL 10-stream
 * DFT10s (PFA 2x5, the file's L=10 slot algebra: pairs (0,5)(2,7)(4,9)
 * (6,1)(8,3), stage-2 groups (0,2,4,6,8)->(0,6,2,8,4) and (5,7,9,1,3)->
 * (5,1,7,3,9), in-place safe), elementwise on site-vectors: zero shuffles.
 *
 * BOOKKEEPING (simulated vs numpy at m=1..4 BEFORE any C ran -- the r8..r11
 * method; build/tryout/gen_batchlane/gen100f.py):
 *   between sweeps, physical plane p = 10*k0 + b holds CT label (b, k0);
 *   parity-0 sweeps take tiles of 10 CONSECUTIVE planes, parity-1 sweeps
 *   tiles at stride 10 (the 10x10 digit map is an involution, so a parity
 *   flag suffices -- gen_pow2's 8x8 case, not their 8x16 perm);
 *   head: DFT10 over positions (=b) + map, output k1 stored AT position k1
 *     (logical x-plane n = 10*k1 + t; after a parity-1 head the volume is
 *     in natural order, so even-m chains unpack natural, odd-m transposed);
 *   tail: DFT10 over positions (=j) with w100^(t*k0) folded into the
 *     stores, output k0 at position k0.  Tile t's outputs are exactly
 *     next-step stage-1 group t -- the property PFA lacks.
 *   prologue (after pack, planes natural): z+y per plane + a parity-1-style
 *     tail; the final sweep is head-only.
 * c is packed once per chain in HEAD-consumption order CT[t][site][k1]
 * (k1 fastest: one sequential stream per tile, both parities -- gen_pow2's
 * tile-order-c rule, the part of their r11 that flipped +10% to -14%). */

#define FROW100 13
#define FPS100  1314

/* BL_FUSE100: 1 = one-sweep fused chain step (this section), 0 = the r11
 * two-pass within-volume chain (kept raceable; execute() always uses it) */
#ifndef BL_FUSE100
#define BL_FUSE100 1
#endif

/* w100^m = cos(2pi m/100) - i sin(2pi m/100), m = b*k0 for b,k0 in 1..9 */
static const v8d KW100R_1 = V8C(0.9980267284282716);
static const v8d KW100I_1 = V8C(-0.06279051952931337);
static const v8d KW100R_2 = V8C(0.9921147013144779);
static const v8d KW100I_2 = V8C(-0.12533323356430426);
static const v8d KW100R_3 = V8C(0.9822872507286887);
static const v8d KW100I_3 = V8C(-0.18738131458572463);
static const v8d KW100R_4 = V8C(0.9685831611286311);
static const v8d KW100I_4 = V8C(-0.2486898871648548);
static const v8d KW100R_5 = V8C(0.9510565162951535);
static const v8d KW100I_5 = V8C(-0.30901699437494745);
static const v8d KW100R_6 = V8C(0.9297764858882515);
static const v8d KW100I_6 = V8C(-0.368124552684678);
static const v8d KW100R_7 = V8C(0.9048270524660196);
static const v8d KW100I_7 = V8C(-0.42577929156507266);
static const v8d KW100R_8 = V8C(0.8763066800438636);
static const v8d KW100I_8 = V8C(-0.48175367410171527);
static const v8d KW100R_9 = V8C(0.8443279255020151);
static const v8d KW100I_9 = V8C(-0.5358267949789967);
static const v8d KW100R_10 = V8C(0.8090169943749475);
static const v8d KW100I_10 = V8C(-0.5877852522924731);
static const v8d KW100R_12 = V8C(0.7289686274214116);
static const v8d KW100I_12 = V8C(-0.6845471059286887);
static const v8d KW100R_14 = V8C(0.6374239897486897);
static const v8d KW100I_14 = V8C(-0.7705132427757893);
static const v8d KW100R_15 = V8C(0.5877852522924731);
static const v8d KW100I_15 = V8C(-0.8090169943749475);
static const v8d KW100R_16 = V8C(0.5358267949789967);
static const v8d KW100I_16 = V8C(-0.8443279255020151);
static const v8d KW100R_18 = V8C(0.42577929156507266);
static const v8d KW100I_18 = V8C(-0.9048270524660196);
static const v8d KW100R_20 = V8C(0.30901699437494745);
static const v8d KW100I_20 = V8C(-0.9510565162951535);
static const v8d KW100R_21 = V8C(0.2486898871648548);
static const v8d KW100I_21 = V8C(-0.9685831611286311);
static const v8d KW100R_24 = V8C(0.06279051952931337);
static const v8d KW100I_24 = V8C(-0.9980267284282716);
static const v8d KW100R_25 = V8C(0.0);
static const v8d KW100I_25 = V8C(-1.0);
static const v8d KW100R_27 = V8C(-0.12533323356430426);
static const v8d KW100I_27 = V8C(-0.9921147013144779);
static const v8d KW100R_28 = V8C(-0.18738131458572463);
static const v8d KW100I_28 = V8C(-0.9822872507286887);
static const v8d KW100R_30 = V8C(-0.30901699437494745);
static const v8d KW100I_30 = V8C(-0.9510565162951535);
static const v8d KW100R_32 = V8C(-0.42577929156507266);
static const v8d KW100I_32 = V8C(-0.9048270524660196);
static const v8d KW100R_35 = V8C(-0.5877852522924731);
static const v8d KW100I_35 = V8C(-0.8090169943749475);
static const v8d KW100R_36 = V8C(-0.6374239897486897);
static const v8d KW100I_36 = V8C(-0.7705132427757893);
static const v8d KW100R_40 = V8C(-0.8090169943749475);
static const v8d KW100I_40 = V8C(-0.5877852522924731);
static const v8d KW100R_42 = V8C(-0.8763066800438636);
static const v8d KW100I_42 = V8C(-0.48175367410171527);
static const v8d KW100R_45 = V8C(-0.9510565162951535);
static const v8d KW100I_45 = V8C(-0.30901699437494745);
static const v8d KW100R_48 = V8C(-0.9921147013144779);
static const v8d KW100I_48 = V8C(-0.12533323356430426);
static const v8d KW100R_49 = V8C(-0.9980267284282716);
static const v8d KW100I_49 = V8C(-0.06279051952931337);
static const v8d KW100R_54 = V8C(-0.9685831611286311);
static const v8d KW100I_54 = V8C(0.2486898871648548);
static const v8d KW100R_56 = V8C(-0.9297764858882515);
static const v8d KW100I_56 = V8C(0.368124552684678);
static const v8d KW100R_63 = V8C(-0.6845471059286887);
static const v8d KW100I_63 = V8C(0.7289686274214116);
static const v8d KW100R_64 = V8C(-0.6374239897486897);
static const v8d KW100I_64 = V8C(0.7705132427757893);
static const v8d KW100R_72 = V8C(-0.18738131458572463);
static const v8d KW100I_72 = V8C(0.9822872507286887);
static const v8d KW100R_81 = V8C(0.368124552684678);
static const v8d KW100I_81 = V8C(0.9297764858882515);

/* map store with the c slots packed CONTIGUOUS per output digit (stride 16
 * doubles: the head's CT[t][site][k1] layout), not at the state stride */
#define DFT5STOREMC(T, p, st, cp, UD, o0, o1, o2, o3, o4)                      \
    do {                                                                       \
        v8d zr, zi;                                                            \
        zr = T##X0r;           zi = T##X0i;                                    \
        map8(&zr, &zi, (cp) + (size_t)(o0) * 16, UD);                          \
        PR(p, st, o0) = zr;  PI_(p, st, o0) = zi;                              \
        zr = T##A1r + T##v1i;  zi = T##A1i - T##v1r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o1) * 16, UD);                          \
        PR(p, st, o1) = zr;  PI_(p, st, o1) = zi;                              \
        zr = T##A1r - T##v1i;  zi = T##A1i + T##v1r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o4) * 16, UD);                          \
        PR(p, st, o4) = zr;  PI_(p, st, o4) = zi;                              \
        zr = T##A2r + T##v2i;  zi = T##A2i - T##v2r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o2) * 16, UD);                          \
        PR(p, st, o2) = zr;  PI_(p, st, o2) = zi;                              \
        zr = T##A2r - T##v2i;  zi = T##A2i + T##v2r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o3) * 16, UD);                          \
        PR(p, st, o3) = zr;  PI_(p, st, o3) = zi;                              \
    } while (0)

#define M5STMC(T, p, st, cp, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4)           \
    do {                                                                       \
        M5BIND(T, p, st, i0, i1, i2, i3, i4)                                   \
        DFT5CORE(T)                                                            \
        DFT5STOREMC(T, p, st, cp, MAPTAIL_GEN, o0, o1, o2, o3, o4);            \
    } while (0)

/* twiddled store: out_o = val * (WR + i WI) */
#define CSTW(p, st, o, tr, ti, WR, WI)                                         \
    do {                                                                       \
        const v8d twr_ = (tr), twi_ = (ti);                                    \
        PR(p, st, o)  = twr_ * (WR) - twi_ * (WI);                             \
        PI_(p, st, o) = twr_ * (WI) + twi_ * (WR);                             \
    } while (0)

/* stage-2 DFT5 with twiddles folded into the stores; W_n pairs with store
 * slot o_n (the M5STTW0 form leaves o0 -- the k=0 output -- untwiddled) */
#define M5STTW0(T, p, st, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4,              \
                W1R, W1I, W2R, W2I, W3R, W3I, W4R, W4I)                        \
    do {                                                                       \
        M5BIND(T, p, st, i0, i1, i2, i3, i4)                                   \
        DFT5CORE(T)                                                            \
        PR(p, st, o0) = T##X0r;  PI_(p, st, o0) = T##X0i;                      \
        CSTW(p, st, o1, T##A1r + T##v1i, T##A1i - T##v1r, W1R, W1I);           \
        CSTW(p, st, o4, T##A1r - T##v1i, T##A1i + T##v1r, W4R, W4I);           \
        CSTW(p, st, o2, T##A2r + T##v2i, T##A2i - T##v2r, W2R, W2I);           \
        CSTW(p, st, o3, T##A2r - T##v2i, T##A2i + T##v2r, W3R, W3I);           \
    } while (0)

#define M5STTW5(T, p, st, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4,              \
                W0R, W0I, W1R, W1I, W2R, W2I, W3R, W3I, W4R, W4I)              \
    do {                                                                       \
        M5BIND(T, p, st, i0, i1, i2, i3, i4)                                   \
        DFT5CORE(T)                                                            \
        CSTW(p, st, o0, T##X0r, T##X0i, W0R, W0I);                             \
        CSTW(p, st, o1, T##A1r + T##v1i, T##A1i - T##v1r, W1R, W1I);           \
        CSTW(p, st, o4, T##A1r - T##v1i, T##A1i + T##v1r, W4R, W4I);           \
        CSTW(p, st, o2, T##A2r + T##v2i, T##A2i - T##v2r, W2R, W2I);           \
        CSTW(p, st, o3, T##A2r - T##v2i, T##A2i + T##v2r, W3R, W3I);           \
    } while (0)

/* vertical in-place DFT10 (PFA 2x5) on positions 0..9 at stride st doubles.
 * Stage-1 M2IP pairs put the sum at the even slot (natural placement, Q=5
 * == 1 mod P=2: in-place safe); stage-2 groups read/write their own parity
 * class, output k stored AT position k. */
#define DFT10V_S1(p, st)                                                       \
    do {                                                                       \
        M2IP(p, st, 0, 5);  M2IP(p, st, 2, 7);  M2IP(p, st, 4, 9);             \
        M2IP(p, st, 6, 1);  M2IP(p, st, 8, 3);                                 \
    } while (0)

AIN void dft10v_map(double *restrict p, const ptrdiff_t st,
                    const double *restrict cp)
{
    DFT10V_S1(p, st);
    M5STMC(va, p, st, cp, 0, 2, 4, 6, 8,   0, 6, 2, 8, 4);
    M5STMC(vb, p, st, cp, 5, 7, 9, 1, 3,   5, 1, 7, 3, 9);
}

AIN void dft10v_plain(double *restrict p, const ptrdiff_t st)
{
    DFT10V_S1(p, st);
    M5ST(va, p, st, 0, 2, 4, 6, 8,   0, 6, 2, 8, 4);
    M5ST(vb, p, st, 5, 7, 9, 1, 3,   5, 1, 7, 3, 9);
}

/* tail form for group b: output k twiddled by w100^(b*k); WkR/WkI below are
 * w100^(b*k), so the a_ group's store slots (6,2,8,4) take (W6,W2,W8,W4) */
#define DFT10V_TW_FN(NAME, W1R, W1I, W2R, W2I, W3R, W3I, W4R, W4I, W5R, W5I,   \
                     W6R, W6I, W7R, W7I, W8R, W8I, W9R, W9I)                   \
    AIN void NAME(double *restrict p, const ptrdiff_t st)                      \
    {                                                                          \
        DFT10V_S1(p, st);                                                      \
        M5STTW0(va, p, st, 0, 2, 4, 6, 8,   0, 6, 2, 8, 4,                     \
                W6R, W6I, W2R, W2I, W8R, W8I, W4R, W4I);                       \
        M5STTW5(vb, p, st, 5, 7, 9, 1, 3,   5, 1, 7, 3, 9,                     \
                W5R, W5I, W1R, W1I, W7R, W7I, W3R, W3I, W9R, W9I);             \
    }

DFT10V_TW_FN(dft10v_tw1,
    KW100R_1, KW100I_1, KW100R_2, KW100I_2, KW100R_3, KW100I_3,
    KW100R_4, KW100I_4, KW100R_5, KW100I_5, KW100R_6, KW100I_6,
    KW100R_7, KW100I_7, KW100R_8, KW100I_8, KW100R_9, KW100I_9)
DFT10V_TW_FN(dft10v_tw2,
    KW100R_2, KW100I_2, KW100R_4, KW100I_4, KW100R_6, KW100I_6,
    KW100R_8, KW100I_8, KW100R_10, KW100I_10, KW100R_12, KW100I_12,
    KW100R_14, KW100I_14, KW100R_16, KW100I_16, KW100R_18, KW100I_18)
DFT10V_TW_FN(dft10v_tw3,
    KW100R_3, KW100I_3, KW100R_6, KW100I_6, KW100R_9, KW100I_9,
    KW100R_12, KW100I_12, KW100R_15, KW100I_15, KW100R_18, KW100I_18,
    KW100R_21, KW100I_21, KW100R_24, KW100I_24, KW100R_27, KW100I_27)
DFT10V_TW_FN(dft10v_tw4,
    KW100R_4, KW100I_4, KW100R_8, KW100I_8, KW100R_12, KW100I_12,
    KW100R_16, KW100I_16, KW100R_20, KW100I_20, KW100R_24, KW100I_24,
    KW100R_28, KW100I_28, KW100R_32, KW100I_32, KW100R_36, KW100I_36)
DFT10V_TW_FN(dft10v_tw5,
    KW100R_5, KW100I_5, KW100R_10, KW100I_10, KW100R_15, KW100I_15,
    KW100R_20, KW100I_20, KW100R_25, KW100I_25, KW100R_30, KW100I_30,
    KW100R_35, KW100I_35, KW100R_40, KW100I_40, KW100R_45, KW100I_45)
DFT10V_TW_FN(dft10v_tw6,
    KW100R_6, KW100I_6, KW100R_12, KW100I_12, KW100R_18, KW100I_18,
    KW100R_24, KW100I_24, KW100R_30, KW100I_30, KW100R_36, KW100I_36,
    KW100R_42, KW100I_42, KW100R_48, KW100I_48, KW100R_54, KW100I_54)
DFT10V_TW_FN(dft10v_tw7,
    KW100R_7, KW100I_7, KW100R_14, KW100I_14, KW100R_21, KW100I_21,
    KW100R_28, KW100I_28, KW100R_35, KW100I_35, KW100R_42, KW100I_42,
    KW100R_49, KW100I_49, KW100R_56, KW100I_56, KW100R_63, KW100I_63)
DFT10V_TW_FN(dft10v_tw8,
    KW100R_8, KW100I_8, KW100R_16, KW100I_16, KW100R_24, KW100I_24,
    KW100R_32, KW100I_32, KW100R_40, KW100I_40, KW100R_48, KW100I_48,
    KW100R_56, KW100I_56, KW100R_64, KW100I_64, KW100R_72, KW100I_72)
DFT10V_TW_FN(dft10v_tw9,
    KW100R_9, KW100I_9, KW100R_18, KW100I_18, KW100R_27, KW100I_27,
    KW100R_36, KW100I_36, KW100R_45, KW100I_45, KW100R_54, KW100I_54,
    KW100R_63, KW100I_63, KW100R_72, KW100I_72, KW100R_81, KW100I_81)

/* REGISTER-FORM verticals (BL_F100RV=1): the memory-form DFT10's stage-1
 * stores + stage-2 reloads are a pure L1 round trip (20 sv st + 20 sv ld
 * per site-column); the register form (R2L stage 1 into named registers,
 * stage 2 straight from registers -- the shape that shipped at the batched
 * L=10 cell since r3) pays exactly 2L ld + 2L st.  Arithmetic order is
 * identical, so the knob is bit-transparent.  RACED AND LOST on the node
 * (worst arm in 6/6 rotated rounds, +5..14% vs the memory form): at 10
 * VERTICAL streams the 20 live site registers + DFT5CORE temps + map
 * temps spill harder than the store-forwarded round trip costs -- the r9
 * "do not trust spill counts as a proxy" lesson, fourth sighting.
 * DEFAULT 0 (memory form); knob kept for CLX/SPR. */
#ifndef BL_F100RV
#define BL_F100RV 0
#endif

#define R5STTWMC(T, p, st, cp, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4)         \
    do {                                                                       \
        R5BIND(T, i0, i1, i2, i3, i4)                                          \
        DFT5CORE(T)                                                            \
        DFT5STOREMC(T, p, st, cp, MAPTAIL_GEN, o0, o1, o2, o3, o4);            \
    } while (0)

AIN void dft10v_map_r(double *restrict p, const ptrdiff_t st,
                      const double *restrict cp)
{
    XDECL10;
    R2L(p, st, 0, 5); R2L(p, st, 2, 7); R2L(p, st, 4, 9);
    R2L(p, st, 6, 1); R2L(p, st, 8, 3);
    R5STTWMC(va, p, st, cp, 0, 2, 4, 6, 8,   0, 6, 2, 8, 4);
    R5STTWMC(vb, p, st, cp, 5, 7, 9, 1, 3,   5, 1, 7, 3, 9);
}

AIN void dft10v_plain_r(double *restrict p, const ptrdiff_t st)
{
    XDECL10;
    R2L(p, st, 0, 5); R2L(p, st, 2, 7); R2L(p, st, 4, 9);
    R2L(p, st, 6, 1); R2L(p, st, 8, 3);
    R5ST(va, p, st, 0, 2, 4, 6, 8,   0, 6, 2, 8, 4);
    R5ST(vb, p, st, 5, 7, 9, 1, 3,   5, 1, 7, 3, 9);
}

#define R5STTW0(T, p, st, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4,              \
                W1R, W1I, W2R, W2I, W3R, W3I, W4R, W4I)                        \
    do {                                                                       \
        R5BIND(T, i0, i1, i2, i3, i4)                                          \
        DFT5CORE(T)                                                            \
        PR(p, st, o0) = T##X0r;  PI_(p, st, o0) = T##X0i;                      \
        CSTW(p, st, o1, T##A1r + T##v1i, T##A1i - T##v1r, W1R, W1I);           \
        CSTW(p, st, o4, T##A1r - T##v1i, T##A1i + T##v1r, W4R, W4I);           \
        CSTW(p, st, o2, T##A2r + T##v2i, T##A2i - T##v2r, W2R, W2I);           \
        CSTW(p, st, o3, T##A2r - T##v2i, T##A2i + T##v2r, W3R, W3I);           \
    } while (0)

#define R5STTW5(T, p, st, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4,              \
                W0R, W0I, W1R, W1I, W2R, W2I, W3R, W3I, W4R, W4I)              \
    do {                                                                       \
        R5BIND(T, i0, i1, i2, i3, i4)                                          \
        DFT5CORE(T)                                                            \
        CSTW(p, st, o0, T##X0r, T##X0i, W0R, W0I);                             \
        CSTW(p, st, o1, T##A1r + T##v1i, T##A1i - T##v1r, W1R, W1I);           \
        CSTW(p, st, o4, T##A1r - T##v1i, T##A1i + T##v1r, W4R, W4I);           \
        CSTW(p, st, o2, T##A2r + T##v2i, T##A2i - T##v2r, W2R, W2I);           \
        CSTW(p, st, o3, T##A2r - T##v2i, T##A2i + T##v2r, W3R, W3I);           \
    } while (0)

#define DFT10V_TW_FN_R(NAME, W1R, W1I, W2R, W2I, W3R, W3I, W4R, W4I, W5R, W5I, \
                       W6R, W6I, W7R, W7I, W8R, W8I, W9R, W9I)                 \
    AIN void NAME(double *restrict p, const ptrdiff_t st)                      \
    {                                                                          \
        XDECL10;                                                               \
        R2L(p, st, 0, 5); R2L(p, st, 2, 7); R2L(p, st, 4, 9);                  \
        R2L(p, st, 6, 1); R2L(p, st, 8, 3);                                    \
        R5STTW0(va, p, st, 0, 2, 4, 6, 8,   0, 6, 2, 8, 4,                     \
                W6R, W6I, W2R, W2I, W8R, W8I, W4R, W4I);                       \
        R5STTW5(vb, p, st, 5, 7, 9, 1, 3,   5, 1, 7, 3, 9,                     \
                W5R, W5I, W1R, W1I, W7R, W7I, W3R, W3I, W9R, W9I);             \
    }

DFT10V_TW_FN_R(dft10v_tw1r,
    KW100R_1, KW100I_1, KW100R_2, KW100I_2, KW100R_3, KW100I_3,
    KW100R_4, KW100I_4, KW100R_5, KW100I_5, KW100R_6, KW100I_6,
    KW100R_7, KW100I_7, KW100R_8, KW100I_8, KW100R_9, KW100I_9)
DFT10V_TW_FN_R(dft10v_tw2r,
    KW100R_2, KW100I_2, KW100R_4, KW100I_4, KW100R_6, KW100I_6,
    KW100R_8, KW100I_8, KW100R_10, KW100I_10, KW100R_12, KW100I_12,
    KW100R_14, KW100I_14, KW100R_16, KW100I_16, KW100R_18, KW100I_18)
DFT10V_TW_FN_R(dft10v_tw3r,
    KW100R_3, KW100I_3, KW100R_6, KW100I_6, KW100R_9, KW100I_9,
    KW100R_12, KW100I_12, KW100R_15, KW100I_15, KW100R_18, KW100I_18,
    KW100R_21, KW100I_21, KW100R_24, KW100I_24, KW100R_27, KW100I_27)
DFT10V_TW_FN_R(dft10v_tw4r,
    KW100R_4, KW100I_4, KW100R_8, KW100I_8, KW100R_12, KW100I_12,
    KW100R_16, KW100I_16, KW100R_20, KW100I_20, KW100R_24, KW100I_24,
    KW100R_28, KW100I_28, KW100R_32, KW100I_32, KW100R_36, KW100I_36)
DFT10V_TW_FN_R(dft10v_tw5r,
    KW100R_5, KW100I_5, KW100R_10, KW100I_10, KW100R_15, KW100I_15,
    KW100R_20, KW100I_20, KW100R_25, KW100I_25, KW100R_30, KW100I_30,
    KW100R_35, KW100I_35, KW100R_40, KW100I_40, KW100R_45, KW100I_45)
DFT10V_TW_FN_R(dft10v_tw6r,
    KW100R_6, KW100I_6, KW100R_12, KW100I_12, KW100R_18, KW100I_18,
    KW100R_24, KW100I_24, KW100R_30, KW100I_30, KW100R_36, KW100I_36,
    KW100R_42, KW100I_42, KW100R_48, KW100I_48, KW100R_54, KW100I_54)
DFT10V_TW_FN_R(dft10v_tw7r,
    KW100R_7, KW100I_7, KW100R_14, KW100I_14, KW100R_21, KW100I_21,
    KW100R_28, KW100I_28, KW100R_35, KW100I_35, KW100R_42, KW100I_42,
    KW100R_49, KW100I_49, KW100R_56, KW100I_56, KW100R_63, KW100I_63)
DFT10V_TW_FN_R(dft10v_tw8r,
    KW100R_8, KW100I_8, KW100R_16, KW100I_16, KW100R_24, KW100I_24,
    KW100R_32, KW100I_32, KW100R_40, KW100I_40, KW100R_48, KW100I_48,
    KW100R_56, KW100I_56, KW100R_64, KW100I_64, KW100R_72, KW100I_72)
DFT10V_TW_FN_R(dft10v_tw9r,
    KW100R_9, KW100I_9, KW100R_18, KW100I_18, KW100R_27, KW100I_27,
    KW100R_36, KW100I_36, KW100R_45, KW100I_45, KW100R_54, KW100I_54,
    KW100R_63, KW100I_63, KW100R_72, KW100I_72, KW100R_81, KW100I_81)

/* ---- in-plane z-pass gather/scatter.  Plane site-vectors hold 8
 * CONSECUTIVE Z in natural element order, so the LX8 fold goes on the
 * gather's ROW index (load) and the scratch SLOT index (store): trans8's
 * out[j][l] = in[lanex[l]][j] with lanex self-inverse then yields natural
 * slots AND natural lanes on both trips (measured semantics, r11 t8.c). */
AIN void zcol_load_f100(const double *restrict pl, double *restrict Pn,
                        int y0, int nry)
{
    const v8d vz = V8C(0.0);
    for (int zb = 0; zb < 13; ++zb) {
        v8d r[8], q[8];
        for (int i = 0; i < 8; ++i) {
            const int dy = LX8[i];
            if (dy < nry) {
                const double *b = pl + ((size_t)(y0 + dy) * FROW100 + zb) * 16;
                r[i] = LD(b);
                q[i] = LD(b + 8);
            } else {
                r[i] = vz;
                q[i] = vz;
            }
        }
        trans8(r);
        trans8(q);
        double *dst = Pn + (size_t)zb * (8 * 16);
        for (int j = 0; j < 8; ++j) {
            ST(dst + (size_t)j * 16, r[j]);
            ST(dst + (size_t)j * 16 + 8, q[j]);
        }
    }
}

AIN void zcol_store_f100(double *restrict pl, const double *restrict Pn,
                         int y0, int nry)
{
    for (int zb = 0; zb < 13; ++zb) {
        const double *src = Pn + (size_t)zb * (8 * 16);
        v8d r[8], q[8];
        for (int i = 0; i < 8; ++i) {
            r[i] = LD(src + (size_t)LX8[i] * 16);
            q[i] = LD(src + (size_t)LX8[i] * 16 + 8);
        }
        trans8(r);
        trans8(q);
        for (int j = 0; j < 8; ++j) {
            if (j < nry) {
                double *b = pl + ((size_t)(y0 + j) * FROW100 + zb) * 16;
                ST(b, r[j]);
                ST(b + 8, q[j]);
            }
        }
    }
}

/* z + y DFTs of one 166-KB plane (L2-resident across both passes): z via
 * the trans8-bracketed scratch pencil, y as row-stride pencils -- both the
 * verified r11 dft100 module.  The y0=96 group carries 4 zero lanes (the
 * xcol ndz pattern); pad z slots 100..103 never contaminate real slots. */
static void fuse100_zy(double *restrict pl)
{
    double Pn[104 * 16] __attribute__((aligned(64)));
    for (int y0 = 0; y0 < 100; y0 += 8) {
        const int nry = (y0 == 96) ? 4 : 8;
        zcol_load_f100(pl, Pn, y0, nry);
        dft100_pencil(Pn, 16);
        zcol_store_f100(pl, Pn, y0, nry);
    }
    for (int zb = 0; zb < 13; ++zb)
        dft100_pencil(pl + (size_t)zb * 16, FROW100 * 16);
}

/* NOTE (raced, clean window, 4/4 rotated pairs): routing these hot loops
 * through runtime-constant dispatch / a per-column tail function cost +4%
 * (4215-4253 vs 4057-4086 us) with BIT-IDENTICAL output -- gcc-11 codegen
 * is that sensitive here (gen_planner r11's map-span lesson, this entry's
 * sighting).  Knob variants are therefore selected by PREPROCESSOR only
 * and the tail keeps the v1 whole-tile loop with the switch outside. */
#if BL_F100RV
#define DFT10V_MAP_FN dft10v_map_r
#define DFT10V_TW(N)  dft10v_tw##N##r
#define DFT10V_PLAIN  dft10v_plain_r
#else
#define DFT10V_MAP_FN dft10v_map
#define DFT10V_TW(N)  dft10v_tw##N
#define DFT10V_PLAIN  dft10v_plain
#endif

static void fuse100_head(double *restrict base, const ptrdiff_t pst_sv,
                         const double *restrict ct)
{
    const ptrdiff_t st = pst_sv * 16;
    for (int i = 0; i < 1300; ++i)
        DFT10V_MAP_FN(base + (size_t)i * 16, st, ct + (size_t)i * 160);
}

static void fuse100_tail(double *restrict base, const ptrdiff_t pst_sv,
                         const int b)
{
    const ptrdiff_t st = pst_sv * 16;
#define BL_TLOOP(FN)                                                           \
    for (int i = 0; i < 1300; ++i) FN(base + (size_t)i * 16, st)
    switch (b) {
    case 0: BL_TLOOP(DFT10V_PLAIN); break;
    case 1: BL_TLOOP(DFT10V_TW(1)); break;
    case 2: BL_TLOOP(DFT10V_TW(2)); break;
    case 3: BL_TLOOP(DFT10V_TW(3)); break;
    case 4: BL_TLOOP(DFT10V_TW(4)); break;
    case 5: BL_TLOOP(DFT10V_TW(5)); break;
    case 6: BL_TLOOP(DFT10V_TW(6)); break;
    case 7: BL_TLOOP(DFT10V_TW(7)); break;
    case 8: BL_TLOOP(DFT10V_TW(8)); break;
    default: BL_TLOOP(DFT10V_TW(9)); break;
    }
#undef BL_TLOOP
}

/* BL_F100TILE=1 interleaves the verticals INTO the in-plane passes (head
 * per 8-row group before its z-pencils, y-pencils zb-outer before that
 * column's tail verticals) -- the residual gen_pow2's r11 record flags as
 * their own next step.  RACED AND LOST on the node (+8..13%, 3/3 vs the
 * phased form): the "hot" reuse windows are 130 KB -- L2-sized, not L1 --
 * so nothing is saved, while the burst-interleaved verticals break the
 * head's linear plane streams.  Kept as a cross-arch knob; default is the
 * PHASED sweep (head, then z+y per plane, then tail: the tile stays
 * L2/L3-resident). */
#ifndef BL_F100TILE
#define BL_F100TILE 0
#endif

#if BL_F100TILE
static void fuse100_tail_col(double *restrict col, const ptrdiff_t pst_sv,
                             const int b)
{
    const ptrdiff_t st = pst_sv * 16;
#define BL_TCLOOP(FN)                                                          \
    for (int y = 0; y < 100; ++y) FN(col + (size_t)y * (FROW100 * 16), st)
    switch (b) {
    case 0: BL_TCLOOP(DFT10V_PLAIN); break;
    case 1: BL_TCLOOP(DFT10V_TW(1)); break;
    case 2: BL_TCLOOP(DFT10V_TW(2)); break;
    case 3: BL_TCLOOP(DFT10V_TW(3)); break;
    case 4: BL_TCLOOP(DFT10V_TW(4)); break;
    case 5: BL_TCLOOP(DFT10V_TW(5)); break;
    case 6: BL_TCLOOP(DFT10V_TW(6)); break;
    case 7: BL_TCLOOP(DFT10V_TW(7)); break;
    case 8: BL_TCLOOP(DFT10V_TW(8)); break;
    default: BL_TCLOOP(DFT10V_TW(9)); break;
    }
#undef BL_TCLOOP
}

static void fuse100_tile_headz(double *restrict base, const ptrdiff_t pst_sv,
                               const double *restrict ct)
{
    const ptrdiff_t st = pst_sv * 16;
    double Pn[104 * 16] __attribute__((aligned(64)));
    for (int y0 = 0; y0 < 100; y0 += 8) {
        const int nry = (y0 == 96) ? 4 : 8;
        for (int dy = 0; dy < nry; ++dy) {
            const size_t i0 = (size_t)(y0 + dy) * 13;
            for (int zb = 0; zb < 13; ++zb)
                DFT10V_MAP_FN(base + (i0 + zb) * 16, st, ct + (i0 + zb) * 160);
        }
        for (int s = 0; s < 10; ++s) {
            double *pl = base + (size_t)s * pst_sv * 16;
            zcol_load_f100(pl, Pn, y0, nry);
            dft100_pencil(Pn, 16);
            zcol_store_f100(pl, Pn, y0, nry);
        }
    }
}

static void fuse100_tile_ytail(double *restrict base, const ptrdiff_t pst_sv,
                               const int b)
{
    for (int zb = 0; zb < 13; ++zb) {
        for (int s = 0; s < 10; ++s)
            dft100_pencil(base + (size_t)s * pst_sv * 16 + (size_t)zb * 16,
                          FROW100 * 16);
        fuse100_tail_col(base + (size_t)zb * 16, pst_sv, b);
    }
}
#endif /* BL_F100TILE */

static void fuse100_sweep(double *restrict S, const double *restrict CT,
                          const int parity, const int do_tail)
{
    const ptrdiff_t tstr = (parity ? 1 : 10) * (ptrdiff_t)FPS100; /* tile base */
    const ptrdiff_t pstr = (parity ? 10 : 1) * (ptrdiff_t)FPS100; /* in-tile */
    for (int t = 0; t < 10; ++t) {
        double *base = S + (size_t)t * tstr * 16;
#if BL_F100TILE
        if (do_tail) {
            fuse100_tile_headz(base, pstr, CT + (size_t)t * (13000 * 16));
            fuse100_tile_ytail(base, pstr, t);
        } else {
            fuse100_head(base, pstr, CT + (size_t)t * (13000 * 16));
        }
#else
        fuse100_head(base, pstr, CT + (size_t)t * (13000 * 16));
        if (do_tail) {
            for (int s = 0; s < 10; ++s)
                fuse100_zy(base + (size_t)s * pstr * 16);
            fuse100_tail(base, pstr, t);
        }
#endif
    }
}

static void fuse100_prologue(double *restrict S)
{
    for (int p = 0; p < 100; ++p)
        fuse100_zy(S + (size_t)p * (FPS100 * 16));
    for (int t = 0; t < 10; ++t)
        fuse100_tail(S + (size_t)t * (FPS100 * 16), 10 * FPS100, t);
}

static void chainsteps_f100(double *restrict S, const double *restrict CT,
                            int m)
{
    fuse100_prologue(S);
    for (int step = 1; step <= m; ++step)
        fuse100_sweep(S, CT, (step - 1) & 1, step < m);
}

/* pack/unpack for the fused layout: plain deinterleave (lanes = natural
 * consecutive z), no trans8 -- cheaper than the r11 slab pack */
static void pack_vol_f100(const double *restrict vin, double *restrict S)
{
    const v8d vz = V8C(0.0);
    for (int x = 0; x < 100; ++x) {
        double *pl = S + (size_t)x * (FPS100 * 16);
        for (int y = 0; y < 100; ++y) {
            const double *src = vin + ((size_t)x * 100 + y) * 200;
            double *row = pl + (size_t)y * (FROW100 * 16);
            for (int zb = 0; zb < 12; ++zb) {
                const v8d a = LD(src + zb * 16);
                const v8d b = LD(src + zb * 16 + 8);
                ST(row + zb * 16, SH(a, b, 0, 2, 4, 6, 8, 10, 12, 14));
                ST(row + zb * 16 + 8, SH(a, b, 1, 3, 5, 7, 9, 11, 13, 15));
            }
            const v8d a = LD(src + 192);        /* z = 96..99; pads zero */
            ST(row + 192, SH(a, vz, 0, 2, 4, 6, 8, 9, 10, 11));
            ST(row + 200, SH(a, vz, 1, 3, 5, 7, 8, 9, 10, 11));
        }
    }
}

static void unpack_vol_f100(double *restrict vout, const double *restrict S,
                            int natural)
{
    for (int n = 0; n < 100; ++n) {
        const int p = natural ? n : 10 * (n % 10) + n / 10;
        const double *pl = S + (size_t)p * (FPS100 * 16);
        for (int y = 0; y < 100; ++y) {
            const double *row = pl + (size_t)y * (FROW100 * 16);
            double *dst = vout + ((size_t)n * 100 + y) * 200;
            for (int zb = 0; zb < 12; ++zb) {
                const v8d re = LD(row + zb * 16);
                const v8d im = LD(row + zb * 16 + 8);
                ST(dst + zb * 16, SH(re, im, 0, 8, 1, 9, 2, 10, 3, 11));
                ST(dst + zb * 16 + 8, SH(re, im, 4, 12, 5, 13, 6, 14, 7, 15));
            }
            const v8d re = LD(row + 192);
            const v8d im = LD(row + 200);
            ST(dst + 192, SH(re, im, 0, 8, 1, 9, 2, 10, 3, 11));
        }
    }
}

/* c in head-consumption order CT[t = x%10][site i = y*13+zb][k1 = x/10],
 * k1 fastest -- one sequential stream per tile head, both parities */
static void pack_ct_f100(const double *restrict c, double *restrict CT)
{
    const v8d vz = V8C(0.0);
    for (int x = 0; x < 100; ++x) {
        const int t = x % 10, k1 = x / 10;
        double *ctt = CT + ((size_t)t * 13000 + (size_t)k1) * 16;
        for (int y = 0; y < 100; ++y) {
            const double *src = c + ((size_t)x * 100 + y) * 200;
            for (int zb = 0; zb < 13; ++zb) {
                double *d = ctt + ((size_t)y * 13 + (size_t)zb) * 160;
                if (zb < 12) {
                    const v8d a = LD(src + zb * 16);
                    const v8d b = LD(src + zb * 16 + 8);
                    ST(d, SH(a, b, 0, 2, 4, 6, 8, 10, 12, 14));
                    ST(d + 8, SH(a, b, 1, 3, 5, 7, 9, 11, 13, 15));
                } else {
                    const v8d a = LD(src + 192);
                    ST(d, SH(a, vz, 0, 2, 4, 6, 8, 9, 10, 11));
                    ST(d + 8, SH(a, vz, 1, 3, 5, 7, 8, 9, 10, 11));
                }
            }
        }
    }
}

/* ==== gen_r11: the L=50 WITHIN-VOLUME engine -- the 100 engine at half size
 * (the brief: wins at 100 transfer to 50).  B=4 is the scored case; the
 * per-volume chain (volume-major, the r5 group-major residency idea) has a
 * state+c working set of 4.6 MB -- fully L3-resident for all m steps.
 * Geometry: 7 x-slabs (slab 6 = 2 real lanes), ZP50 = 51 sv rows,
 * SLST50 = 2562 sv (slab bytes == 256 mod 4096), zg=6 carries 2 real z. */
#define ZP50   51
#define SLST50 2562
#define CST50  51

static void sweep_zy_50(double *restrict slab)
{
    for (int y = 0; y < 50; ++y)
        dft50_pencil(slab + (size_t)y * (ZP50 * 16), 16);
    for (int z = 0; z < 50; ++z)
        dft50_pencil(slab + (size_t)z * 16, ZP50 * 16);
}

AIN void xcol_load_50(const double *restrict S, double *restrict Pn,
                      int y, int zg, int ndz)
{
    const v8d vz = V8C(0.0);
    for (int xb = 0; xb < 7; ++xb) {
        const double *base = S + ((size_t)xb * SLST50 + (size_t)y * ZP50
                                  + (size_t)zg * 8) * 16;
        v8d r[8], q[8];
        for (int dz = 0; dz < 8; ++dz) {
            if (dz < ndz) {
                r[dz] = LD(base + (size_t)dz * 16);
                q[dz] = LD(base + (size_t)dz * 16 + 8);
            } else {
                r[dz] = vz;
                q[dz] = vz;
            }
        }
        trans8(r);
        trans8(q);
        double *dst = Pn + (size_t)xb * (8 * 16);
        for (int j = 0; j < 8; ++j) {
            ST(dst + (size_t)LX8[j] * 16, r[j]);
            ST(dst + (size_t)LX8[j] * 16 + 8, q[j]);
        }
    }
}

AIN void xcol_store_50(double *restrict S, const double *restrict Pn,
                       int y, int zg, int ndz)
{
    for (int xb = 0; xb < 7; ++xb) {
        const double *src = Pn + (size_t)xb * (8 * 16);
        v8d r[8], q[8];
        for (int i = 0; i < 8; ++i) {
            r[i] = LD(src + (size_t)i * 16);
            q[i] = LD(src + (size_t)i * 16 + 8);
        }
        trans8(r);
        trans8(q);
        double *base = S + ((size_t)xb * SLST50 + (size_t)y * ZP50
                            + (size_t)zg * 8) * 16;
        for (int j = 0; j < 8; ++j) {
            const int dz = LX8[j];
            if (dz < ndz) {
                ST(base + (size_t)dz * 16, r[j]);
                ST(base + (size_t)dz * 16 + 8, q[j]);
            }
        }
    }
}

static void xpass_50(double *restrict S)
{
    double Pn[56 * 16] __attribute__((aligned(64)));
    for (int y = 0; y < 50; ++y)
        for (int zg = 0; zg < 7; ++zg) {
            const int ndz = (zg == 6) ? 2 : 8;
            xcol_load_50(S, Pn, y, zg, ndz);
            dft50_pencil(Pn, 16);
            xcol_store_50(S, Pn, y, zg, ndz);
        }
}

static void xpass_map_50(double *restrict S, const double *restrict CT)
{
    double Pn[56 * 16] __attribute__((aligned(64)));
    for (int y = 0; y < 50; ++y)
        for (int zg = 0; zg < 7; ++zg) {
            const int ndz = (zg == 6) ? 2 : 8;
            xcol_load_50(S, Pn, y, zg, ndz);
            dft50_pencil_map(Pn, 16, CT + ((size_t)y * 7 + zg) * (CST50 * 16));
            xcol_store_50(S, Pn, y, zg, ndz);
        }
}

static void pack_vol_50(const double *restrict vin, double *restrict S)
{
    for (int s = 0; s < 7; ++s) {
        const int nl = (s == 6) ? 2 : 8;
        for (int y = 0; y < 50; ++y) {
            const double *vp[8];
            for (int l = 0; l < 8; ++l) {
                const int x = 8 * s + (l < nl ? l : nl - 1);
                vp[l] = vin + ((size_t)x * 50 + y) * 100;
            }
            pack_plane(vp, S + ((size_t)s * SLST50 + (size_t)y * ZP50) * 16, 50);
        }
    }
}

static void unpack_vol_50(double *restrict vout, const double *restrict S)
{
    for (int s = 0; s < 7; ++s) {
        const int nl = (s == 6) ? 2 : 8;
        for (int y = 0; y < 50; ++y) {
            double *op[8];
            for (int l = 0; l < 8; ++l) {
                const int x = 8 * s + (l < nl ? l : nl - 1);
                op[l] = vout + ((size_t)x * 50 + y) * 100;
            }
            unpack_plane(op, S + ((size_t)s * SLST50 + (size_t)y * ZP50) * 16,
                         50, nl);
        }
    }
}

/* zg=6 holds only z=48,49: a vector load of the 8-complex block would run
 * past the caller's array at (x,y)=(49,49), so the last group packs by
 * scalar lane (once per chain; the lanex order matches the vector path). */
static void pack_ct_50(const double *restrict c, double *restrict CT)
{
    const v8d vz = V8C(0.0);
    for (int y = 0; y < 50; ++y)
        for (int zg = 0; zg < 7; ++zg) {
            double *col = CT + ((size_t)y * 7 + zg) * (CST50 * 16);
            const size_t zb = (size_t)zg * 8;
            for (int x = 0; x < 50; ++x) {
                const double *src = c + (((size_t)x * 50 + y) * 50 + zb) * 2;
                if (zg == 6) {
                    for (int l = 0; l < 8; ++l) {
                        const int lx = LX8[l];
                        col[(size_t)x * 16 + l] = lx < 2 ? src[2 * lx] : 0.0;
                        col[(size_t)x * 16 + 8 + l] = lx < 2 ? src[2 * lx + 1] : 0.0;
                    }
                } else {
                    const v8d a = LD(src);
                    const v8d b = (v8d)LD(src + 8);
                    (void)vz;
                    ST(col + (size_t)x * 16, SH(a, b, 0, 2, 8, 10, 4, 6, 12, 14));
                    ST(col + (size_t)x * 16 + 8,
                       SH(a, b, 1, 3, 9, 11, 5, 7, 13, 15));
                }
            }
        }
}

static void chainsteps_50(double *restrict S, const double *restrict CT, int m)
{
    for (int step = 0; step < m; ++step) {
        for (int s = 0; s < 7; ++s)
            sweep_zy_50(S + (size_t)s * (SLST50 * 16));
        xpass_map_50(S, CT);
    }
}

/* ==== gen_r13: WITHIN-VOLUME engines for L=10 and L=12 -- close the
 * benchFFT-exposed B=1 gap (new scored cells 10:1:16384, 12:1:12288).
 * This is the r11 L=100 within-volume trick at small scale (the brief names
 * it for this entry): ONE volume fills the lanes instead of eight volumes.
 *
 * LAYOUT (z-in-lanes rows, the r12 f100 discipline, chosen over x-planes
 * because it minimizes transposes: cross-lane groups number ceil(L^2/8),
 * not L*ceil(L/8) -- 13 vs 20 at L=10, 18 vs 24 at 12):
 *   row r = x*L + y, row = 2 site-vectors of 8 CONSECUTIVE z (re8|im8);
 *   z slots L..15 and (L=10) pad rows 100..103 are packed EXACT ZEROS and
 *   provably stay zero (DFT of 0 = 0; map is only applied to real z slots
 *   at real+zero c, and map(0, c=0) = 0): no NaN/denormal hazard, ever.
 *   State: 104*2 sv (26.6 KB) at 10 / 144*2 sv (36.9 KB) at 12 -- with the
 *   consumption-order c both are L1/L2-resident for the whole chain.
 *   - y-pencils: stride 2 sv at fixed (x, zb)   -- elementwise, no shuffles
 *   - x-pencils: stride 2L sv at fixed (y, zb)  -- elementwise, no shuffles
 *   - z-pass LAST (so the graded map fuses into its stores): per 8 rows,
 *     trans8-bracketed 16-sv L1 scratch (the r11/r12 zcol pattern: LX8 on
 *     the row index going in, natural slots/lanes both ways), the shipped
 *     fused-map pencil on the scratch, c packed once per chain as
 *     CT[group][k] (stride 16 doubles = the scratch stride).
 * Per step at L=10: 40 elementwise + 13 scratch DFT10s (53 vs the 37.5/vol
 * the B=64 groups pay -- the price of no batch) + 104 trans8; at 12:
 * 48 + 18 DFT12s + 144 trans8.  vs the replicated-lane fallback (8x
 * arithmetic for one volume) this is ~5x less work.
 * ROUTING: full groups keep the batch engine; a remainder group runs
 * within-volume per volume iff nv <= BL_WVMAX.  Crossover from the node
 * measurements: within-volume = 2.09 / 3.21 us per volume-step at 10/12 vs
 * a replicated 8-lane group's 9.2 / 15.3 -- within wins up to nv = 4 at
 * both sizes and loses from 5 (both engines' arithmetic scales measured,
 * not modeled).  -DBL_NOWV=1 restores the replicated path everywhere. */
#ifndef BL_NOWV
#define BL_NOWV 0
#endif
#ifndef BL_WVMAX
#define BL_WVMAX 4
#endif
/* map-pencil forms on the z-scratch: the batched x-pass verdicts (swap+div
 * at 10, unswapped+rcp at 12) DO NOT TRANSFER to this shape -- the r3/r9
 * codelet-local rule, measured again.  Same-core interleaved race on the
 * scored B=1 chains (a80n0 core 7, --samples 4 minima, first invocation
 * discarded): 10 UNSWAPPED beats swap 2.128 vs 2.200 (-3.3%), and DIV tail
 * beats the rcp ladder on that unswapped register codelet 2.085-2.096 vs
 * 2.129-2.132 (-1.7%); 12 SWAPPED beats unswapped 3.202-3.208 vs
 * 3.219-3.222 (-0.5%).  Reading: the wv z-scratch pencil sits between two
 * trans8 networks whose 24-shuffle blocks saturate p5, so the wider DFT5
 * stage-2 (5 interleaved ladders) schedules better than the batched
 * verdict's thin DFT2 stage-2, and the divider is idle where the batched
 * x-pass keeps it warm. */
#ifndef BL_WVSWAP10
#define BL_WVSWAP10 0
#endif
#ifndef BL_WVSWAP12
#define BL_WVSWAP12 1
#endif
/* WV-local map tail for the unswapped 10 pencil (div won the race above);
 * follows the global cross-arch overrides when they are set */
#if defined(BL_MAPDIV)
#define MAPTAIL_WV10 1
#elif defined(BL_MAPRCP)
#define MAPTAIL_WV10 0
#elif !defined(MAPTAIL_WV10)
#define MAPTAIL_WV10 1
#endif

#define R5STMW(T, p, st, cp, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4)           \
    do {                                                                       \
        R5BIND(T, i0, i1, i2, i3, i4)                                          \
        DFT5CORE(T)                                                            \
        DFT5STOREM(T, p, st, cp, MAPTAIL_WV10, o0, o1, o2, o3, o4);            \
    } while (0)

/* unswapped 10 map pencil with the WV tail (arithmetic order identical to
 * dft10_pencil_map; only the reciprocal path differs per MAPTAIL_WV10) */
AIN void dft10_pencil_map_wv(double *restrict p, const ptrdiff_t st,
                             const double *restrict cp)
{
    XDECL10;
    R2L(p, st, 0, 5); R2L(p, st, 2, 7); R2L(p, st, 4, 9);
    R2L(p, st, 6, 1); R2L(p, st, 8, 3);
    R5STMW(a_, p, st, cp, 0, 2, 4, 6, 8,  0, 6, 2, 8, 4);
    R5STMW(b_, p, st, cp, 5, 7, 9, 1, 3,  5, 1, 7, 3, 9);
}

#if BL_WVSWAP10
#define WV10_MAP dft10_pencil_map_swap
#else
#define WV10_MAP dft10_pencil_map_wv
#endif
#if BL_WVSWAP12
#define WV12_MAP dft12_pencil_map_swap
#else
#define WV12_MAP dft12_pencil_map
#endif
/* sched-pressure on the trans8-bracketed z-pass (the elementwise passes
 * inherit SCHED1012 like the batched sweeps; the shuffle networks were the
 * r1 reason global scheduling lost, so the z-pass defaults to stock) */
#ifndef BL_WVZSCHED
#define BL_WVZSCHED 0
#endif
#if BL_WVZSCHED
#define WVZATTR SCHEDP
#else
#define WVZATTR
#endif

/* gather 8 rows (r0..r0+7) into the 16-sv scratch: slot s = z (natural),
 * lanes = rows (natural: LX8 on the load row index, lanex self-inverse) */
AIN void wv_zload(const double *restrict S, double *restrict Pn, int r0)
{
    for (int zb = 0; zb < 2; ++zb) {
        v8d r[8], q[8];
        for (int i = 0; i < 8; ++i) {
            const double *b = S + ((size_t)(r0 + LX8[i]) * 2 + zb) * 16;
            r[i] = LD(b);
            q[i] = LD(b + 8);
        }
        trans8(r);
        trans8(q);
        double *dst = Pn + (size_t)zb * (8 * 16);
        for (int j = 0; j < 8; ++j) {
            ST(dst + (size_t)j * 16, r[j]);
            ST(dst + (size_t)j * 16 + 8, q[j]);
        }
    }
}

AIN void wv_zstore(double *restrict S, const double *restrict Pn, int r0)
{
    for (int zb = 0; zb < 2; ++zb) {
        const double *src = Pn + (size_t)zb * (8 * 16);
        v8d r[8], q[8];
        for (int i = 0; i < 8; ++i) {
            r[i] = LD(src + (size_t)LX8[i] * 16);
            q[i] = LD(src + (size_t)LX8[i] * 16 + 8);
        }
        trans8(r);
        trans8(q);
        for (int j = 0; j < 8; ++j) {
            double *b = S + ((size_t)(r0 + j) * 2 + zb) * 16;
            ST(b, r[j]);
            ST(b + 8, q[j]);
        }
    }
}

/* NR = real rows (L^2), NG = ceil(NR/8) z-groups; row slots NG*8 */
#define DEF_WV_ENGINE(N, NG, PEN, PENM)                                        \
static SCHED1012 void yxpass_wv_##N(double *restrict S)                        \
{                                                                              \
    for (int zb = 0; zb < 2; ++zb)                                             \
        for (int x = 0; x < N; ++x)                                            \
            PEN(S + ((size_t)x * N * 2 + zb) * 16, 2 * 16);                    \
    for (int zb = 0; zb < 2; ++zb)                                             \
        for (int y = 0; y < N; ++y)                                            \
            PEN(S + ((size_t)y * 2 + zb) * 16, N * 2 * 16);                    \
}                                                                              \
static WVZATTR void zpass_wv_##N(double *restrict S)                           \
{                                                                              \
    double Pn[16 * 16] __attribute__((aligned(64)));                           \
    for (int g = 0; g < NG; ++g) {                                             \
        wv_zload(S, Pn, 8 * g);                                                \
        PEN(Pn, 16);                                                           \
        wv_zstore(S, Pn, 8 * g);                                               \
    }                                                                          \
}                                                                              \
static WVZATTR void zpass_map_wv_##N(double *restrict S,                       \
                                     const double *restrict CT)                \
{                                                                              \
    double Pn[16 * 16] __attribute__((aligned(64)));                           \
    for (int g = 0; g < NG; ++g) {                                             \
        wv_zload(S, Pn, 8 * g);                                                \
        PENM(Pn, 16, CT + (size_t)g * (N * 16));                               \
        wv_zstore(S, Pn, 8 * g);                                               \
    }                                                                          \
}                                                                              \
static void chainsteps_wv_##N(double *restrict S, const double *restrict CT,   \
                              int m)                                           \
{                                                                              \
    for (int s = 0; s < m; ++s) {                                              \
        yxpass_wv_##N(S);                                                      \
        zpass_map_wv_##N(S, CT);                                               \
    }                                                                          \
}

DEF_WV_ENGINE(10, 13, dft10_pencil, WV10_MAP)
DEF_WV_ENGINE(12, 18, dft12_pencil, WV12_MAP)

/* row r = 20 doubles at L=10; tail sv from an overlapped LD(src+12) so the
 * last row of the volume never reads past the caller's array */
static void pack_vol_wv_10(const double *restrict vin, double *restrict S)
{
    const v8d vz = V8C(0.0);
    for (int r = 0; r < 100; ++r) {
        const double *src = vin + (size_t)r * 20;
        double *row = S + (size_t)r * 32;
        const v8d a = LD(src), b = LD(src + 8), t = LD(src + 12);
        ST(row,      SH(a, b, 0, 2, 4, 6, 8, 10, 12, 14));
        ST(row + 8,  SH(a, b, 1, 3, 5, 7, 9, 11, 13, 15));
        ST(row + 16, SH(t, vz, 4, 6, 8, 9, 10, 11, 12, 13));
        ST(row + 24, SH(t, vz, 5, 7, 8, 9, 10, 11, 12, 13));
    }
    for (int r = 100; r < 104; ++r)     /* pad rows: exact zeros, stay zero */
        for (int k = 0; k < 32; k += 8)
            ST(S + (size_t)r * 32 + k, vz);
}

static void unpack_vol_wv_10(double *restrict vout, const double *restrict S)
{
    for (int r = 0; r < 100; ++r) {
        const double *row = S + (size_t)r * 32;
        double *dst = vout + (size_t)r * 20;
        const v8d re = LD(row), im = LD(row + 8);
        ST(dst,     SH(re, im, 0, 8, 1, 9, 2, 10, 3, 11));
        ST(dst + 8, SH(re, im, 4, 12, 5, 13, 6, 14, 7, 15));
        dst[16] = row[16]; dst[17] = row[24];   /* z=8,9: 4 scalar stores */
        dst[18] = row[17]; dst[19] = row[25];
    }
}

static void pack_vol_wv_12(const double *restrict vin, double *restrict S)
{
    const v8d vz = V8C(0.0);
    for (int r = 0; r < 144; ++r) {
        const double *src = vin + (size_t)r * 24;
        double *row = S + (size_t)r * 32;
        const v8d a = LD(src), b = LD(src + 8), t = LD(src + 16);
        ST(row,      SH(a, b, 0, 2, 4, 6, 8, 10, 12, 14));
        ST(row + 8,  SH(a, b, 1, 3, 5, 7, 9, 11, 13, 15));
        ST(row + 16, SH(t, vz, 0, 2, 4, 6, 8, 9, 10, 11));
        ST(row + 24, SH(t, vz, 1, 3, 5, 7, 8, 9, 10, 11));
    }
}

static void unpack_vol_wv_12(double *restrict vout, const double *restrict S)
{
    for (int r = 0; r < 144; ++r) {
        const double *row = S + (size_t)r * 32;
        double *dst = vout + (size_t)r * 24;
        const v8d re0 = LD(row), im0 = LD(row + 8);
        const v8d re1 = LD(row + 16), im1 = LD(row + 24);
        ST(dst,      SH(re0, im0, 0, 8, 1, 9, 2, 10, 3, 11));
        ST(dst + 8,  SH(re0, im0, 4, 12, 5, 13, 6, 14, 7, 15));
        ST(dst + 16, SH(re1, im1, 0, 8, 1, 9, 2, 10, 3, 11));
    }
}

/* c in z-pass consumption order CT[g][k], lanes = rows 8g.. (natural); pad
 * rows get c = 0 (with state 0 the map then fixes them at exactly 0).
 * Scalar: once per chain, amortized over m = 12288..16384 steps. */
static void pack_ct_wv(const double *restrict c, double *restrict CT,
                       int N, int NG, int NR)
{
    for (int g = 0; g < NG; ++g)
        for (int k = 0; k < N; ++k) {
            double *d = CT + ((size_t)g * N + k) * 16;
            for (int l = 0; l < 8; ++l) {
                const int r = 8 * g + l;
                d[l]     = r < NR ? c[((size_t)r * N + k) * 2] : 0.0;
                d[8 + l] = r < NR ? c[((size_t)r * N + k) * 2 + 1] : 0.0;
            }
        }
}

static int plane_stride_sv(int L)   /* PL2: L^2 padded to == 2 (mod 32) sv */
{
    switch (L) {
    case 10: return 130;
    case 12: return 162;
    case 14: return 226;
    case 15: return 226;
    case 20: return 418;
    case 21: return 450;
    case 22: return 514;
    case 28: return 802;
    case 33: return 1090;
    case 35: return 1250;
    case 44: return 1954;
    default: return 3042;   /* 55 */
    }
}

struct fft3d_plan {
    int L, batch, PL2;
    double *S, *C;      /* one 8-volume group each, split-complex SoA */
    gl_map map;         /* gen_layout THP mapping (posix_memalign fallback) */
};

const char *fft3d_name(void) { return "gen_batchlane"; }
const char *fft3d_description(void)
{
    return "SoA 8-vol/zmm batch-lane (bl8 lineage): twiddle-free 2-stage PFA "
           "pencils (10=2x5,12=3x4,15=3x5,20=4x5; r6 adds 7-smooth 14/21/28/35 "
           "via a DFT7 module; r8 adds 11-smooth 22/33/44/55 via a DFT11 "
           "module; r7 LIFTED DFT5 v-pair -- sin72=phi*sin36 exact, "
           "6 ops not 8, lit 08 6.3), register-explicit at 10/12 (2L ld + 2L "
           "st) and in the 15 zy-sweep (r7 hybrid), memory form elsewhere with "
           "r6 SAFE PLACEMENT (stage-1 store permutation makes every stage-2 "
           "group in-place on its own mod-P residue class), L1 zy-sweep + "
           "x-pass, fused chain in SoA with eager rsqrt14 map (rcp14 ladder "
           "at 12, vdivpd elsewhere), r9 FACTOR-SWAPPED map x-pencils at "
           "10/15/20 (large factor stage 1 in place, small factor + map "
           "stage 2: kills the fused-map spills, -1..-2%), r10 extends the "
           "swap to the wide-module family 21/22/28/33/35/44/55 (-1.7..-4.7%; "
           "14 keeps the unswapped order), r11 adds L=100 WITHIN-VOLUME SoA "
           "(lanes = 8 x-planes of one volume, B=1 native: shuffle-free zy "
           "sweeps fused per slab, trans8-bracketed x-pass, PFA 4x25 with "
           "DFT25=5x5 CT through an L1 scratch, 9 compiled-in w25 twiddle "
           "constants -- the file's first twiddles), r12 L=100 chain goes "
           "ONE-SWEEP FUSED (gen_pow2 r11's step-boundary x-split, CT 10x10 "
           "-- the involutive equal-radix case PFA provably cannot tile: "
           "z-in-lanes planes, vertical DFT10 stage-2+map head / twiddled "
           "stage-1 tail per 10-plane tile, parity-alternating tiling, "
           "tile-order c, 36 compiled-in w100 constants; DRAM crossings "
           "2->1 per step, LLC loads -19%), r13 WITHIN-VOLUME B=1 engines at "
           "10/12 (z-in-lanes rows, elementwise y/x passes, trans8-bracketed "
           "z-pass with fused map -- the r11 trick at small scale for the new "
           "B=1 cells; remainder groups route per-volume), sched-pressure on "
           "10/12 only, THP arena (gen_layout), plane stride 256 mod 4096";
}
int fft3d_supports(int L)
{
    return L == 10 || L == 12 || L == 14 || L == 15 || L == 20 || L == 21 ||
           L == 22 || L == 28 || L == 33 || L == 35 || L == 44 || L == 55 ||
           L == 50 || L == 100;
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L) || batch < 1) return NULL;
    fft3d_plan *p = malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;

    if (L == 100 || L == 50) {
        /* within-volume engine: ONE volume of state + the x-consumption-order
         * c buffer; batches loop volumes through it (per-volume chains are
         * independent, so volume-major is exact) */
        p->PL2 = 0;
        const size_t st_bytes = (L == 100 ? (size_t)13 * SLST100
                                          : (size_t)7 * SLST50) *
                                16 * sizeof(double);
        const size_t ct_bytes = (L == 100 ? (size_t)100 * 13 * CST100
                                          : (size_t)50 * 7 * CST50) *
                                16 * sizeof(double);
        const size_t c_off = ((st_bytes + 4095) & ~(size_t)4095) + 2048;
        void *base = gl_map_huge(&p->map, c_off + ct_bytes);
        if (!base) {
            free(p);
            return NULL;
        }
        p->S = (double *)base;
        p->C = (double *)((char *)base + c_off);
        return p;
    }

    p->PL2 = plane_stride_sv(L);

    const size_t grp_bytes = (size_t)L * p->PL2 * 16 * sizeof(double);
    const size_t c_off = ((grp_bytes + 4095) & ~(size_t)4095) + 2048;  /* bl8 pad */
    void *base = gl_map_huge(&p->map, c_off + grp_bytes);  /* zeroed, prefaulted */
    if (!base) {
        free(p);
        return NULL;
    }
    p->S = (double *)base;
    p->C = (double *)((char *)base + c_off);
    return p;
}

/* group volume pointers: lanes past the batch end replicate the last volume */
static void group_vols(const double *buf, size_t vol_doubles, int g0, int nv,
                       const double *vp[8])
{
    for (int l = 0; l < 8; ++l) {
        const int v = g0 + (l < nv ? l : nv - 1);
        vp[l] = buf + (size_t)v * vol_doubles;
    }
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    const int L = p->L, B = p->batch, LL = L * L, PL2 = p->PL2;
    const size_t vold = (size_t)L * LL * 2;
    const size_t psz = (size_t)PL2 * 16;

    if (L == 100) {
        for (int v = 0; v < B; ++v) {
            pack_vol_100((const double *)in + (size_t)v * vold, p->S);
            for (int s = 0; s < 13; ++s)
                sweep_zy_100(p->S + (size_t)s * (SLST100 * 16));
            xpass_100(p->S);
            unpack_vol_100((double *)out + (size_t)v * vold, p->S);
        }
        return;
    }
    if (L == 50) {
        for (int v = 0; v < B; ++v) {
            pack_vol_50((const double *)in + (size_t)v * vold, p->S);
            for (int s = 0; s < 7; ++s)
                sweep_zy_50(p->S + (size_t)s * (SLST50 * 16));
            xpass_50(p->S);
            unpack_vol_50((double *)out + (size_t)v * vold, p->S);
        }
        return;
    }

    /* gen_r13: small remainders (incl. B=1) at 10/12 go WITHIN-VOLUME
     * instead of paying 8x on replicated lanes; full groups and remainders
     * past the measured crossover (nv > BL_WVMAX) unchanged */
    int Bgrp = B;
    if (!BL_NOWV && (L == 10 || L == 12) && (B & 7) && (B & 7) <= BL_WVMAX)
        Bgrp = B & ~7;

    for (int g0 = 0; g0 < Bgrp; g0 += 8) {
        const int nv = (B - g0 < 8) ? B - g0 : 8;
        const double *vin[8];
        group_vols((const double *)in, vold, g0, nv, vin);

        for (int x = 0; x < L; ++x) {
            const double *pp[8];
            for (int l = 0; l < 8; ++l) pp[l] = vin[l] + (size_t)x * LL * 2;
            double *pl = p->S + (size_t)x * psz;
            pack_plane(pp, pl, LL);
            switch (L) {
            case 10: sweep_zy_10(pl); break;
            case 12: sweep_zy_12(pl); break;
            case 14: sweep_zy_14(pl); break;
            case 15: sweep_zy_15(pl); break;
            case 20: sweep_zy_20(pl); break;
            case 21: sweep_zy_21(pl); break;
            case 22: sweep_zy_22(pl); break;
            case 28: sweep_zy_28(pl); break;
            case 33: sweep_zy_33(pl); break;
            case 35: sweep_zy_35(pl); break;
            case 44: sweep_zy_44(pl); break;
            default: sweep_zy_55(pl); break;
            }
        }
        switch (L) {
        case 10: xpass_10(p->S); break;
        case 12: xpass_12(p->S); break;
        case 14: xpass_14(p->S); break;
        case 15: xpass_15(p->S); break;
        case 20: xpass_20(p->S); break;
        case 21: xpass_21(p->S); break;
        case 22: xpass_22(p->S); break;
        case 28: xpass_28(p->S); break;
        case 33: xpass_33(p->S); break;
        case 35: xpass_35(p->S); break;
        case 44: xpass_44(p->S); break;
        default: xpass_55(p->S); break;
        }
        for (int x = 0; x < L; ++x) {
            double *op[8];
            for (int l = 0; l < 8; ++l)
                op[l] = (double *)out + (size_t)(g0 + (l < nv ? l : nv - 1)) * vold
                        + (size_t)x * LL * 2;
            unpack_plane(op, p->S + (size_t)x * psz, LL, nv);
        }
    }

    for (int v = Bgrp; v < B; ++v) {
        const double *iv = (const double *)in + (size_t)v * vold;
        double *ov = (double *)out + (size_t)v * vold;
        if (L == 10) {
            pack_vol_wv_10(iv, p->S);
            yxpass_wv_10(p->S);
            zpass_wv_10(p->S);
            unpack_vol_wv_10(ov, p->S);
        } else {
            pack_vol_wv_12(iv, p->S);
            yxpass_wv_12(p->S);
            zpass_wv_12(p->S);
            unpack_vol_wv_12(ov, p->S);
        }
    }
}

/* the whole graded chain: state <- (FFT(state)+c)/(1+|FFT(state)+c|), m steps.
 * Pack once, run every step in SoA (state and c both L2-resident), unpack once. */
void fft3d_chain(fft3d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    const int L = p->L, B = p->batch, LL = L * L;
    const size_t vold = (size_t)L * LL * 2;
    const size_t psz = (size_t)p->PL2 * 16;

    if (L == 100) {
        /* per-volume chains are independent: volume-major keeps one volume's
         * state + c (33.6 MB) as the whole working set for all m steps */
        for (int v = 0; v < B; ++v) {
#if BL_FUSE100
            /* gen_r12 one-sweep fused steps; final parity is 1 (natural
             * plane order) exactly when m is even */
            pack_vol_f100((const double *)x0 + (size_t)v * vold, p->S);
            pack_ct_f100((const double *)c + (size_t)v * vold, p->C);
            chainsteps_f100(p->S, p->C, m);
            unpack_vol_f100((double *)final_out + (size_t)v * vold, p->S,
                            (m & 1) == 0);
#else
            pack_vol_100((const double *)x0 + (size_t)v * vold, p->S);
            pack_ct_100((const double *)c + (size_t)v * vold, p->C);
            chainsteps_100(p->S, p->C, m);
            unpack_vol_100((double *)final_out + (size_t)v * vold, p->S);
#endif
        }
        return;
    }
    if (L == 50) {
        /* volume-major: one volume's state + c = 4.6 MB, L3-resident chain */
        for (int v = 0; v < B; ++v) {
            pack_vol_50((const double *)x0 + (size_t)v * vold, p->S);
            pack_ct_50((const double *)c + (size_t)v * vold, p->C);
            chainsteps_50(p->S, p->C, m);
            unpack_vol_50((double *)final_out + (size_t)v * vold, p->S);
        }
        return;
    }

    /* gen_r13: small remainders (incl. the scored B=1 cells) at 10/12 run
     * per-volume within-volume chains; full groups and nv > BL_WVMAX
     * remainders keep the batch engine (see the crossover note above) */
    int Bgrp = B;
    if (!BL_NOWV && (L == 10 || L == 12) && (B & 7) && (B & 7) <= BL_WVMAX)
        Bgrp = B & ~7;

    for (int g0 = 0; g0 < Bgrp; g0 += 8) {
        const int nv = (B - g0 < 8) ? B - g0 : 8;
        const double *vx[8], *vc[8];
        group_vols((const double *)x0, vold, g0, nv, vx);
        group_vols((const double *)c, vold, g0, nv, vc);

        for (int x = 0; x < L; ++x) {
            const double *px[8], *pc[8];
            for (int l = 0; l < 8; ++l) {
                px[l] = vx[l] + (size_t)x * LL * 2;
                pc[l] = vc[l] + (size_t)x * LL * 2;
            }
            pack_plane(px, p->S + (size_t)x * psz, LL);
            pack_plane(pc, p->C + (size_t)x * psz, LL);
        }

        switch (L) {
        case 10: chainsteps_10(p->S, p->C, m); break;
        case 12: chainsteps_12(p->S, p->C, m); break;
        case 14: chainsteps_14(p->S, p->C, m); break;
        case 15: chainsteps_15(p->S, p->C, m); break;
        case 20: chainsteps_20(p->S, p->C, m); break;
        case 21: chainsteps_21(p->S, p->C, m); break;
        case 22: chainsteps_22(p->S, p->C, m); break;
        case 28: chainsteps_28(p->S, p->C, m); break;
        case 33: chainsteps_33(p->S, p->C, m); break;
        case 35: chainsteps_35(p->S, p->C, m); break;
        case 44: chainsteps_44(p->S, p->C, m); break;
        default: chainsteps_55(p->S, p->C, m); break;
        }

        for (int x = 0; x < L; ++x) {
            double *op[8];
            for (int l = 0; l < 8; ++l)
                op[l] = (double *)final_out
                        + (size_t)(g0 + (l < nv ? l : nv - 1)) * vold
                        + (size_t)x * LL * 2;
            unpack_plane(op, p->S + (size_t)x * psz, LL, nv);
        }
    }

    for (int v = Bgrp; v < B; ++v) {
        const double *xv = (const double *)x0 + (size_t)v * vold;
        const double *cv = (const double *)c + (size_t)v * vold;
        double *ov = (double *)final_out + (size_t)v * vold;
        if (L == 10) {
            pack_vol_wv_10(xv, p->S);
            pack_ct_wv(cv, p->C, 10, 13, 100);
            chainsteps_wv_10(p->S, p->C, m);
            unpack_vol_wv_10(ov, p->S);
        } else {
            pack_vol_wv_12(xv, p->S);
            pack_ct_wv(cv, p->C, 12, 18, 144);
            chainsteps_wv_12(p->S, p->C, m);
            unpack_vol_wv_12(ov, p->S);
        }
    }
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    gl_unmap(&p->map);
    free(p);
}
