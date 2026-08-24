/* gen_batchlane -- SoA 8-volumes-per-zmm batch-lane engine for small L at B >= 8.
 *
 * LINEAGE: ice_r7/r8 "bl8" chain inside L8_fusedaxes.c (itself adopted from rivals
 * v5_cb7847fb / 8dc1a96d), generalized from L=8 to the class sizes 10, 12, 15.
 *
 * THE IDEA
 *   Fill the eight zmm lanes with EIGHT VOLUMES of the batch.  Every 1D DFT of the
 *   3D transform is then elementwise across registers: ZERO shuffles per step,
 *   lane-invariant constants, and the only transposes in the whole graded chain are
 *   the pack/unpack at the two ends (once per m = 600..1000 steps).
 *
 * WHY THESE SIZES NEED NO TWIDDLES
 *   10 = 2 x 5, 12 = 3 x 4 (coprime), 15 = 3 x 5: every axis DFT is a two-stage
 *   Good-Thomas PFA.  Input map n = (Qa + Pb) mod L (P*Q = L), output map
 *   k = (c, d) by CRT; both permutations are folded into the compile-time slot
 *   offsets of the codelet loads/stores.  No twiddle table exists anywhere in this
 *   file -- the campaign's twiddle problem is dodged entirely inside this class.
 *   Codelet costs (vector instrs, FMA-contracted, per pencil per 8 volumes):
 *      L=10: 5xDFT2 + 2xDFT5              = 88
 *      L=12: 4xDFT3 + 3xDFT4              = 96
 *      L=15: 5xDFT3 + 3xDFT5              = 162
 *
 * STRUCTURE (per group of 8 volumes)
 *   Scratch is split-complex SoA: site vector = 16 doubles (re[8] | im[8]), site
 *   order natural (x*L^2 + y*L + z) with the PLANE STRIDE PADDED so that
 *   plane_bytes == 256 (mod 4096) -- bl8's anti-alias pad, kept: it breaks the
 *   inter-plane residue degeneracy that would stack the x-pass column loads in a
 *   few L1 sets.  PL2 = 130 / 162 / 226 site-vectors for L = 10 / 12 / 15.
 *      sweep zy : per x-plane (12.8..28.9 KiB, L1-resident): z-pencils stride 1
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
 *   together (2 x 424 KiB at L=15, node L2 = 1.25 MiB) and (C - S) == 2048
 *   (mod 4096) de-aliases the paired column streams (bl8's offset).  The map is
 *   EAGER and in-register: applied to each x-pass output right before its store
 *   (bl8/L17 lesson: lazy map loses -- it puts the ~40-cycle map chain in front of
 *   the next step's critical path).  Map form (r2, adopted from gen_pfa_small r1):
 *   s = wr^2 + wi^2 + 1e-300, vrsqrt14pd seed, TWO quadratic Newtons (2^-14 ->
 *   2^-27.4 -> 2^-53.8, full double), d = fma(s,y,1) = 1 + |w|, then vrcp14pd +
 *   TWO residual Newtons for 1/d -- NO divider op anywhere (vdivpd zmm is
 *   unpipelined ~16 cyc/op and the x-pass issues one per output vector; the
 *   swap bought -8.1/-8.8/-4.7% of the whole chain at L=10/12/15).
 *
 * BATCH HANDLING
 *   B % 8 == 0 (every scored case: 10:64, 12:64, 15:32) runs full groups.  A
 *   remainder group (incl. B = 1) replicates its last volume into the unused
 *   lanes and unpacks only the real ones -- correct for any B >= 1, paying up to
 *   8x arithmetic on the remainder group only.
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
map8(v8d *restrict vr, v8d *restrict vi, const double *restrict cp)
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
    v8d t = (v8d)_mm512_rcp14_pd((__m512d)d);
    t = t + t * (one - d * t);          /* residual Newton: 2^-14 -> 2^-28 */
    t = t + t * (one - d * t);          /* -> full double, no vdivpd */
#else
    (void)half; (void)c15;
    v8d d;
    for (int k = 0; k < 8; ++k) d[k] = 1.0 + __builtin_sqrt(s[k]);
    const v8d t = one / d;
#endif
    *vr = wr * t;
    *vi = wi * t;
}

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

/* in-place DFT-2 on slots a,b */
#define DFT2S(p, st, a, b)                                                     \
    do {                                                                       \
        const v8d x0r = PR(p, st, a), x0i = PI_(p, st, a);                     \
        const v8d x1r = PR(p, st, b), x1i = PI_(p, st, b);                     \
        PR(p, st, a) = x0r + x1r;  PI_(p, st, a) = x0i + x1i;                  \
        PR(p, st, b) = x0r - x1r;  PI_(p, st, b) = x0i - x1i;                  \
    } while (0)

/* in-place forward DFT-3 on slots i0,i1,i2 (12 vector instrs) */
#define DFT3S(p, st, i0, i1, i2)                                               \
    do {                                                                       \
        const v8d x0r = PR(p, st, i0), x0i = PI_(p, st, i0);                   \
        const v8d x1r = PR(p, st, i1), x1i = PI_(p, st, i1);                   \
        const v8d x2r = PR(p, st, i2), x2i = PI_(p, st, i2);                   \
        const v8d tr = x1r + x2r, ti = x1i + x2i;                              \
        const v8d ur = x1r - x2r, ui = x1i - x2i;                              \
        const v8d hr = x0r - KH * tr, hi = x0i - KH * ti;                      \
        PR(p, st, i0) = x0r + tr;         PI_(p, st, i0) = x0i + ti;           \
        PR(p, st, i1) = hr + KS3 * ui;    PI_(p, st, i1) = hi - KS3 * ur;      \
        PR(p, st, i2) = hr - KS3 * ui;    PI_(p, st, i2) = hi + KS3 * ur;      \
    } while (0)

/* forward DFT-4, read slots i0..i3 (PFA b-order), write slots o0..o3 (16 instrs) */
#define DFT4CORE(p, st, i0, i1, i2, i3)                                        \
        const v8d x0r = PR(p, st, i0), x0i = PI_(p, st, i0);                   \
        const v8d x1r = PR(p, st, i1), x1i = PI_(p, st, i1);                   \
        const v8d x2r = PR(p, st, i2), x2i = PI_(p, st, i2);                   \
        const v8d x3r = PR(p, st, i3), x3i = PI_(p, st, i3);                   \
        const v8d t0r = x0r + x2r, t0i = x0i + x2i;                            \
        const v8d t1r = x0r - x2r, t1i = x0i - x2i;                            \
        const v8d t2r = x1r + x3r, t2i = x1i + x3i;                            \
        const v8d t3r = x1r - x3r, t3i = x1i - x3i;                            \
        const v8d y0r = t0r + t2r, y0i = t0i + t2i;                            \
        const v8d y2r = t0r - t2r, y2i = t0i - t2i;                            \
        const v8d y1r = t1r + t3i, y1i = t1i - t3r;                            \
        const v8d y3r = t1r - t3i, y3i = t1i + t3r;

#define DFT4S(p, st, i0, i1, i2, i3, o0, o1, o2, o3)                           \
    do {                                                                       \
        DFT4CORE(p, st, i0, i1, i2, i3)                                        \
        PR(p, st, o0) = y0r;  PI_(p, st, o0) = y0i;                            \
        PR(p, st, o1) = y1r;  PI_(p, st, o1) = y1i;                            \
        PR(p, st, o2) = y2r;  PI_(p, st, o2) = y2i;                            \
        PR(p, st, o3) = y3r;  PI_(p, st, o3) = y3i;                            \
    } while (0)

#define DFT4SM(p, st, cp, i0, i1, i2, i3, o0, o1, o2, o3)                      \
    do {                                                                       \
        DFT4CORE(p, st, i0, i1, i2, i3)                                        \
        v8d zr, zi;                                                            \
        zr = y0r; zi = y0i; map8(&zr, &zi, (cp) + (size_t)(o0) * (st));        \
        PR(p, st, o0) = zr;  PI_(p, st, o0) = zi;                              \
        zr = y1r; zi = y1i; map8(&zr, &zi, (cp) + (size_t)(o1) * (st));        \
        PR(p, st, o1) = zr;  PI_(p, st, o1) = zi;                              \
        zr = y2r; zi = y2i; map8(&zr, &zi, (cp) + (size_t)(o2) * (st));        \
        PR(p, st, o2) = zr;  PI_(p, st, o2) = zi;                              \
        zr = y3r; zi = y3i; map8(&zr, &zi, (cp) + (size_t)(o3) * (st));        \
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

#define DFT5LOAD(T, p, st, i0, i1, i2, i3, i4)                                 \
        const v8d T##x0r = PR(p, st, i0), T##x0i = PI_(p, st, i0);             \
        const v8d T##x1r = PR(p, st, i1), T##x1i = PI_(p, st, i1);             \
        const v8d T##x2r = PR(p, st, i2), T##x2i = PI_(p, st, i2);             \
        const v8d T##x3r = PR(p, st, i3), T##x3i = PI_(p, st, i3);             \
        const v8d T##x4r = PR(p, st, i4), T##x4i = PI_(p, st, i4);

#define DFT5STORE(T, p, st, o0, o1, o2, o3, o4)                                \
        PR(p, st, o0) = T##X0r;              PI_(p, st, o0) = T##X0i;          \
        PR(p, st, o1) = T##A1r + T##v1i;     PI_(p, st, o1) = T##A1i - T##v1r; \
        PR(p, st, o4) = T##A1r - T##v1i;     PI_(p, st, o4) = T##A1i + T##v1r; \
        PR(p, st, o2) = T##A2r + T##v2i;     PI_(p, st, o2) = T##A2i - T##v2r; \
        PR(p, st, o3) = T##A2r - T##v2i;     PI_(p, st, o3) = T##A2i + T##v2r;

#define DFT5STOREM(T, p, st, cp, o0, o1, o2, o3, o4)                           \
    do {                                                                       \
        v8d zr, zi;                                                            \
        zr = T##X0r;           zi = T##X0i;                                    \
        map8(&zr, &zi, (cp) + (size_t)(o0) * (st));                            \
        PR(p, st, o0) = zr;  PI_(p, st, o0) = zi;                              \
        zr = T##A1r + T##v1i;  zi = T##A1i - T##v1r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o1) * (st));                            \
        PR(p, st, o1) = zr;  PI_(p, st, o1) = zi;                              \
        zr = T##A1r - T##v1i;  zi = T##A1i + T##v1r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o4) * (st));                            \
        PR(p, st, o4) = zr;  PI_(p, st, o4) = zi;                              \
        zr = T##A2r + T##v2i;  zi = T##A2i - T##v2r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o2) * (st));                            \
        PR(p, st, o2) = zr;  PI_(p, st, o2) = zi;                              \
        zr = T##A2r - T##v2i;  zi = T##A2i + T##v2r;                           \
        map8(&zr, &zi, (cp) + (size_t)(o3) * (st));                            \
        PR(p, st, o3) = zr;  PI_(p, st, o3) = zi;                              \
    } while (0)

#define DFT5S(p, st, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4)                   \
    do {                                                                       \
        DFT5LOAD(a_, p, st, i0, i1, i2, i3, i4)                                \
        DFT5CORE(a_)                                                           \
        DFT5STORE(a_, p, st, o0, o1, o2, o3, o4)                               \
    } while (0)

#define DFT5SM(p, st, cp, i0, i1, i2, i3, i4, o0, o1, o2, o3, o4)              \
    do {                                                                       \
        DFT5LOAD(a_, p, st, i0, i1, i2, i3, i4)                                \
        DFT5CORE(a_)                                                           \
        DFT5STOREM(a_, p, st, cp, o0, o1, o2, o3, o4);                         \
    } while (0)

/* L=15 stage-2 groups c=1,c=2 have EQUAL read and write slot sets ({slots != 0 mod
 * 3}), so both groups' inputs must be loaded before either group stores: fused. */
#define DFT5X2S(p, st, i0,i1,i2,i3,i4, o0,o1,o2,o3,o4,                         \
                        j0,j1,j2,j3,j4, w0,w1,w2,w3,w4)                        \
    do {                                                                       \
        DFT5LOAD(a_, p, st, i0, i1, i2, i3, i4)                                \
        DFT5LOAD(b_, p, st, j0, j1, j2, j3, j4)                                \
        DFT5CORE(a_)                                                           \
        DFT5CORE(b_)                                                           \
        DFT5STORE(a_, p, st, o0, o1, o2, o3, o4)                               \
        DFT5STORE(b_, p, st, w0, w1, w2, w3, w4)                               \
    } while (0)

#define DFT5X2SM(p, st, cp, i0,i1,i2,i3,i4, o0,o1,o2,o3,o4,                    \
                          j0,j1,j2,j3,j4, w0,w1,w2,w3,w4)                      \
    do {                                                                       \
        DFT5LOAD(a_, p, st, i0, i1, i2, i3, i4)                                \
        DFT5LOAD(b_, p, st, j0, j1, j2, j3, j4)                                \
        DFT5CORE(a_)                                                           \
        DFT5CORE(b_)                                                           \
        DFT5STOREM(a_, p, st, cp, o0, o1, o2, o3, o4);                         \
        DFT5STOREM(b_, p, st, cp, w0, w1, w2, w3, w4);                         \
    } while (0)

/* Pressure-aware scheduling on ALL pencil/sweep/chainstep families (r2).
 * r1 measured global -fschedule-insns -fsched-pressure as +17%/+16% at 10/12
 * (it also rescheduled the shuffle-heavy pack/unpack) and kept it for 15 only;
 * r2 re-measured as a per-FUNCTION attribute with the longer rcp14 map ladder:
 * 10: 1.222 -> 1.165 (-4.7%), 12: 2.068 -> 1.936 (-6.4%), and stripping it
 * from 15 costs +4.3% -- so all three families now carry it, pack/unpack does
 * not.  A/B knobs (dev only): -DBL_NOSCHED1012 / -DBL_NOSCHED15 strip it. */
#define SCHEDP __attribute__((optimize("schedule-insns", "sched-pressure")))
#if defined(BL_NOSCHED15)
#define SCHED15
#else
#define SCHED15 SCHEDP
#endif
#if defined(BL_NOSCHED1012)
#define SCHED1012
#else
#define SCHED1012 SCHEDP
#endif

/* ---- length-L pencil DFTs, PFA slot maps baked in (see file header) ----
 * L=10: n=(5a+2b)%10, k=(5c+6d)%10.  L=12: n=(4a+3b)%12, k=(4c+9d)%12.
 * L=15: n=(5a+3b)%15, k=(10c+6d)%15. */
static inline SCHED1012 void dft10_pencil(double *restrict p, const ptrdiff_t st)
{
    DFT2S(p, st, 0, 5); DFT2S(p, st, 2, 7); DFT2S(p, st, 4, 9);
    DFT2S(p, st, 6, 1); DFT2S(p, st, 8, 3);
    DFT5S(p, st, 0, 2, 4, 6, 8,  0, 6, 2, 8, 4);
    DFT5S(p, st, 5, 7, 9, 1, 3,  5, 1, 7, 3, 9);
}

static inline SCHED1012 void dft10_pencil_map(double *restrict p, const ptrdiff_t st,
                                              const double *restrict cp)
{
    DFT2S(p, st, 0, 5); DFT2S(p, st, 2, 7); DFT2S(p, st, 4, 9);
    DFT2S(p, st, 6, 1); DFT2S(p, st, 8, 3);
    DFT5SM(p, st, cp, 0, 2, 4, 6, 8,  0, 6, 2, 8, 4);
    DFT5SM(p, st, cp, 5, 7, 9, 1, 3,  5, 1, 7, 3, 9);
}

static inline SCHED1012 void dft12_pencil(double *restrict p, const ptrdiff_t st)
{
    DFT3S(p, st, 0, 4, 8);  DFT3S(p, st, 3, 7, 11);
    DFT3S(p, st, 6, 10, 2); DFT3S(p, st, 9, 1, 5);
    DFT4S(p, st, 0, 3, 6, 9,   0, 9, 6, 3);
    DFT4S(p, st, 4, 7, 10, 1,  4, 1, 10, 7);
    DFT4S(p, st, 8, 11, 2, 5,  8, 5, 2, 11);
}

static inline SCHED1012 void dft12_pencil_map(double *restrict p, const ptrdiff_t st,
                                              const double *restrict cp)
{
    DFT3S(p, st, 0, 4, 8);  DFT3S(p, st, 3, 7, 11);
    DFT3S(p, st, 6, 10, 2); DFT3S(p, st, 9, 1, 5);
    DFT4SM(p, st, cp, 0, 3, 6, 9,   0, 9, 6, 3);
    DFT4SM(p, st, cp, 4, 7, 10, 1,  4, 1, 10, 7);
    DFT4SM(p, st, cp, 8, 11, 2, 5,  8, 5, 2, 11);
}

static inline SCHED15 void dft15_pencil(double *restrict p, const ptrdiff_t st)
{
    DFT3S(p, st, 0, 5, 10);  DFT3S(p, st, 3, 8, 13); DFT3S(p, st, 6, 11, 1);
    DFT3S(p, st, 9, 14, 4);  DFT3S(p, st, 12, 2, 7);
    DFT5S(p, st, 0, 3, 6, 9, 12,  0, 6, 12, 3, 9);
    DFT5X2S(p, st, 5, 8, 11, 14, 2,   10, 1, 7, 13, 4,
                   10, 13, 1, 4, 7,    5, 11, 2, 8, 14);
}

static inline SCHED15 void dft15_pencil_map(double *restrict p, const ptrdiff_t st,
                                            const double *restrict cp)
{
    DFT3S(p, st, 0, 5, 10);  DFT3S(p, st, 3, 8, 13); DFT3S(p, st, 6, 11, 1);
    DFT3S(p, st, 9, 14, 4);  DFT3S(p, st, 12, 2, 7);
    DFT5SM(p, st, cp, 0, 3, 6, 9, 12,  0, 6, 12, 3, 9);
    DFT5X2SM(p, st, cp, 5, 8, 11, 14, 2,   10, 1, 7, 13, 4,
                        10, 13, 1, 4, 7,    5, 11, 2, 8, 14);
}

/* ---- the sweeps, one instantiation per L (strides compile-time constant) ----
 * PL2 = padded plane stride in site-vectors, plane bytes == 256 (mod 4096). */
#define DEF_ENGINE(N, PL2V, ATTR)                                              \
static ATTR void sweep_zy_##N(double *restrict pl)                             \
{                                                                              \
    for (int y = 0; y < N; ++y)                                                \
        dft##N##_pencil(pl + (size_t)y * (N * 16), 16);                        \
    for (int z = 0; z < N; ++z)                                                \
        dft##N##_pencil(pl + (size_t)z * 16, N * 16);                          \
}                                                                              \
static ATTR void xpass_##N(double *restrict S)                                 \
{                                                                              \
    for (int c = 0; c < N * N; ++c)                                            \
        dft##N##_pencil(S + (size_t)c * 16, PL2V * 16);                        \
}                                                                              \
static ATTR void chainsteps_##N(double *restrict S, const double *restrict C,  \
                                int m)                                         \
{                                                                              \
    for (int s = 0; s < m; ++s) {                                              \
        for (int x = 0; x < N; ++x)                                            \
            sweep_zy_##N(S + (size_t)x * ((size_t)PL2V * 16));                 \
        for (int c = 0; c < N * N; ++c)                                        \
            dft##N##_pencil_map(S + (size_t)c * 16, PL2V * 16,                 \
                                C + (size_t)c * 16);                           \
    }                                                                          \
}

DEF_ENGINE(10, 130, SCHED1012)
DEF_ENGINE(12, 162, SCHED1012)
DEF_ENGINE(15, 226, SCHED15)

static int plane_stride_sv(int L)   /* PL2: L^2 padded to == 2 (mod 32) sv */
{
    return L == 10 ? 130 : L == 12 ? 162 : 226;
}

struct fft3d_plan {
    int L, batch, PL2;
    double *S, *C;      /* one 8-volume group each, split-complex SoA */
    gl_map map;         /* gen_layout THP mapping (posix_memalign fallback) */
};

const char *fft3d_name(void) { return "gen_batchlane"; }
const char *fft3d_description(void)
{
    return "SoA 8-vol/zmm batch-lane (bl8 lineage): twiddle-free 2-stage PFA pencils "
           "(10=2x5,12=3x4,15=3x5), L1 zy-sweep + x-pass, fused chain in SoA with "
           "eager divider-free rsqrt14/rcp14 map, sched-pressure codelets, THP "
           "arena (gen_layout), plane stride 256 mod 4096";
}
int fft3d_supports(int L) { return L == 10 || L == 12 || L == 15; }

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
            default: sweep_zy_15(pl); break;
            }
        }
        switch (L) {
        case 10: xpass_10(p->S); break;
        case 12: xpass_12(p->S); break;
        default: xpass_15(p->S); break;
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
        default: chainsteps_15(p->S, p->C, m); break;
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
