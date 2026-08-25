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
#define MAPTAIL_REG 1
#define MAPTAIL_15  1
#define MAPTAIL_20  1
#define MAPTAIL_GEN 1
#elif defined(BL_MAPRCP)
#define MAPTAIL_REG 0
#define MAPTAIL_15  0
#define MAPTAIL_20  0
#define MAPTAIL_GEN 0
#else
#define MAPTAIL_REG 0
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
#define DFT15_MAP   dft15_pencil_map
#elif BL_MEM15 == 1
#define DFT15_SWEEP DFT15_MEM_SWEEP
#define DFT15_MAP   DFT15_MEM_MAP
#else
#define DFT15_SWEEP dft15_pencil
#define DFT15_MAP   DFT15_MEM_MAP
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

DEF_ENGINE(10, 130, SCHED1012, BL_EPI10, MAPTAIL_REG, dft10_pencil, dft10_pencil_map)
DEF_ENGINE(12, 162, SCHED1012, BL_EPI12, MAPTAIL_REG, dft12_pencil, dft12_pencil_map)
DEF_ENGINE(15, 226, SCHED15,   BL_EPI15, MAPTAIL_15,  DFT15_SWEEP,  DFT15_MAP)
DEF_ENGINE(20, 418, SCHED20,   BL_EPI20, MAPTAIL_20,  dft20_pencil, dft20_pencil_map)
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
#if BL_X214
DEF_ENGINE(14, 226,  , 0, MAPTAIL_GEN, dft14_pencil_x2, dft14_pencil_map_x2)
#else
DEF_ENGINE(14, 226,  , 0, MAPTAIL_GEN, dft14_pencil, dft14_pencil_map)
#endif
DEF_ENGINE(21, 450,  , 0, MAPTAIL_GEN, dft21_pencil, dft21_pencil_map)
DEF_ENGINE(28, 802,  , 0, MAPTAIL_GEN, dft28_pencil, dft28_pencil_map)
DEF_ENGINE(35, 1250, , 0, MAPTAIL_GEN, dft35_pencil, dft35_pencil_map)
/* gen_r8: the 11-smooth sizes, same memory-form defaults */
DEF_ENGINE(22, 514,  , 0, MAPTAIL_GEN, dft22_pencil, dft22_pencil_map)
DEF_ENGINE(33, 1090, , 0, MAPTAIL_GEN, dft33_pencil, dft33_pencil_map)
DEF_ENGINE(44, 1954, , 0, MAPTAIL_GEN, dft44_pencil, dft44_pencil_map)
DEF_ENGINE(55, 3042, , 0, MAPTAIL_GEN, dft55_pencil, dft55_pencil_map)

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
           "at 10/12, vdivpd elsewhere), sched-pressure on 10/12 only, THP "
           "arena (gen_layout), plane stride 256 mod 4096";
}
int fft3d_supports(int L)
{
    return L == 10 || L == 12 || L == 14 || L == 15 || L == 20 || L == 21 ||
           L == 22 || L == 28 || L == 33 || L == 35 || L == 44 || L == 55;
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L) || batch < 1) return NULL;
    fft3d_plan *p = malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
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

    for (int g0 = 0; g0 < B; g0 += 8) {
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
}

/* the whole graded chain: state <- (FFT(state)+c)/(1+|FFT(state)+c|), m steps.
 * Pack once, run every step in SoA (state and c both L2-resident), unpack once. */
void fft3d_chain(fft3d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    const int L = p->L, B = p->batch, LL = L * L;
    const size_t vold = (size_t)L * LL * 2;
    const size_t psz = (size_t)p->PL2 * 16;

    for (int g0 = 0; g0 < B; g0 += 8) {
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
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    gl_unmap(&p->map);
    free(p);
}
