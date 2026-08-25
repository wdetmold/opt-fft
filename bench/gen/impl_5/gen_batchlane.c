/* gen_batchlane -- SoA 8-volumes-per-zmm batch-lane engine for small L at B >= 8.
 *
 * LINEAGE: ice_r7/r8 "bl8" chain inside L8_fusedaxes.c (itself adopted from rivals
 * v5_cb7847fb / 8dc1a96d), generalized from L=8 to the class sizes 10, 12, 15
 * and (r3) 20.
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
 *   Codelet costs (vector FP instrs, FMA-contracted, per pencil per 8 volumes):
 *      L=10: 5xDFT2 + 2xDFT5              = 88
 *      L=12: 4xDFT3 + 3xDFT4              = 96
 *      L=15: 5xDFT3 + 3xDFT5              = 162
 *      L=20: 5xDFT4 + 4xDFT5              = 216
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
 *   sweeps + memory map x-pass, lost by 0.3% r5).
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
#elif defined(BL_MAPRCP)
#define MAPTAIL_REG 0
#define MAPTAIL_15  0
#define MAPTAIL_20  0
#else
#define MAPTAIL_REG 0
#define MAPTAIL_15  1
#define MAPTAIL_20  1
#endif

/* ---- codelet constants (exact to the last bit of double) */
static const v8d KH  = V8C(0.5);
static const v8d KS3 = V8C(0.86602540378443864676);   /* sin(pi/3) */
static const v8d K25 = V8C(0.25);
static const v8d KQ5 = V8C(0.55901699437494742410);   /* sqrt(5)/4 */
static const v8d KS1 = V8C(0.95105651629515357212);   /* sin(2pi/5) */
static const v8d KS2 = V8C(0.58778525229247312917);   /* sin(4pi/5) */

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

/* forward DFT-5 core: inputs T##x0r.., outputs T##X0r/A1/A2/v1/v2 (34 instrs with
 * the output combines).  Winograd split: c1,c2 = (-1 +- sqrt5)/4. */
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
        const v8d T##v1r = KS1 * T##sar + KS2 * T##sbr;                        \
        const v8d T##v1i = KS1 * T##sai + KS2 * T##sbi;                        \
        const v8d T##v2r = KS2 * T##sar - KS1 * T##sbr;                        \
        const v8d T##v2i = KS2 * T##sai - KS1 * T##sbi;

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
#ifndef BL_MEM15
#define BL_MEM15 1
#endif
#if BL_MEM15 == 0
#define DFT15_SWEEP dft15_pencil
#define DFT15_MAP   dft15_pencil_map
#elif BL_MEM15 == 1
#define DFT15_SWEEP dft15_pencil_mem
#define DFT15_MAP   dft15_pencil_map_mem
#else
#define DFT15_SWEEP dft15_pencil
#define DFT15_MAP   dft15_pencil_map_mem
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

static int plane_stride_sv(int L)   /* PL2: L^2 padded to == 2 (mod 32) sv */
{
    return L == 10 ? 130 : L == 12 ? 162 : L == 15 ? 226 : 418;
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
           "pencils (10=2x5,12=3x4,15=3x5,20=4x5), register-explicit at 10/12 "
           "(2L ld + 2L st), memory form at 15/20 (r5: reg spills past 24 live "
           "sites; 15 has the fused DFT5X2 equal-slot pair), L1 zy-sweep + "
           "x-pass, fused chain in SoA with eager rsqrt14 map (rcp14 ladder at "
           "10/12, vdivpd at 15/20 -- r5 same-core re-race), sched-pressure on "
           "10/12 only, THP arena (gen_layout), plane stride 256 mod 4096";
}
int fft3d_supports(int L) { return L == 10 || L == 12 || L == 15 || L == 20; }

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
            case 15: sweep_zy_15(pl); break;
            default: sweep_zy_20(pl); break;
            }
        }
        switch (L) {
        case 10: xpass_10(p->S); break;
        case 12: xpass_12(p->S); break;
        case 15: xpass_15(p->S); break;
        default: xpass_20(p->S); break;
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
        case 15: chainsteps_15(p->S, p->C, m); break;
        default: chainsteps_20(p->S, p->C, m); break;
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
