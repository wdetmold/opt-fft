/* d1_batchlane: SoA 8-lane-per-zmm batch-lane engine.
 *
 * Class mandate (survey vein 1, batched #1-2, SPIRAL DFT_n (x) I_v): vectorize ACROSS the
 * batch in split-complex SoA -- lane j = transform j, so every butterfly is a vertical
 * vector op with ZERO in-register shuffles. Shuffles happen only at the batch boundary:
 * one 8x8 transpose in, one out, per group of 8 transforms.
 *
 * Kernels (straight-line, natural-order in/out, twiddles from plan-time sincos tables):
 *   13, 31 : dense symmetric-pair DFT (u_j = x_j + x_{L-j}, v_j = x_j - x_{L-j};
 *            X_k = x0 + A_k -/+ i B_k), halves the dense work, all real*complex FMAs.
 *   32     : Cooley-Tukey 4 x FFT8 -> twiddle -> 8 x FFT4.
 *   64     : Cooley-Tukey 8 x FFT8 -> twiddle -> 8 x FFT8.
 *   128    : 2 x FFT64 (stride 2) -> W128 twiddle -> 64 x FFT2.
 *   60     : Good-Thomas PFA 3x4x5, TWIDDLE-FREE. In SoA the CRT permutations are just
 *            index tables -- no shuffles, no gathers, plain indexed loads.
 *
 * fft1d_chain owns the whole m-step map chain: per group of 8 transforms the state, the
 * c field and the scratch all stay L1-resident across ALL m steps (~32 KB at L=64), while
 * the driver fallback streams the full batch through memory once per step. The map
 * (z+c)/(1+|z+c|) uses exact 512-bit vsqrt/vdiv intrinsics (gcc auto-vectorizes the loop
 * at ymm width only), masked tail so odd L pays no scalar sqrt chain.
 *
 * B=1 / batch-remainder transforms use single-transform kernels in which the SIMD lanes
 * carry an INTERNAL index instead of the batch index:
 *   32/64/128: four-step, lanes = outer decimation residue n1 (32 = 4x8 ymm,
 *              64 = 8x8 zmm, 128 = 8x16 zmm); only shuffles are the middle transpose.
 *   60:        four-step 4x15 (ymm lanes), inner FFT15 = twiddle-free PFA 3x5.
 *   13/31:     densesym with lanes = output index k, u_j/v_j broadcast (dsk8).
 *
 * The kernel file is instantiated three times via self-inclusion: V = v8 (batch groups
 * of 8), V = v4 (ymm building blocks), V = double (scalar fallback).
 */

#ifndef BL_COMMON
#define BL_COMMON

#include <complex.h>
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../fft1d_api.h"

typedef double v8  __attribute__((vector_size(64), aligned(64)));
typedef double v8u __attribute__((vector_size(64), aligned(8)));   /* unaligned ld/st */
typedef long long v8i __attribute__((vector_size(64)));
typedef double v4  __attribute__((vector_size(32), aligned(32)));
typedef double v4u __attribute__((vector_size(32), aligned(8)));
typedef long long v4i __attribute__((vector_size(32)));

struct fft1d_plan {
    int L, batch;
    double *tw;                 /* size-specific twiddle/constant tables */
    double *twv;                /* vector twiddles for the single-transform kernels */
    int *im, *om;               /* PFA index maps (L=60) */
    v8 *xr, *xi, *yr, *yi, *wr, *wi, *cr, *ci;   /* SoA scratch, each L vectors */
    void *scratch_base;
};

#define S2H  0.70710678118654752440   /* sqrt(2)/2 */
#define S3H  0.86602540378443864676   /* sin(2pi/3) */
#define C5_1 0.30901699437494742410   /* cos(2pi/5) */
#define C5_2 (-0.80901699437494742410) /* cos(4pi/5) */
#define S5_1 0.95105651629515357212   /* sin(2pi/5) */
#define S5_2 0.58778525229247312917   /* sin(4pi/5) */

/* ---- instantiate the kernels for 8 zmm lanes ---- */
#define V v8
#define VW 8
#define KN(name) name##_v8
#define VZ ((v8){0})
#include "d1_batchlane.c"
#undef V
#undef VW
#undef KN
#undef VZ

/* ---- instantiate the kernels for 4 ymm lanes ---- */
#define V v4
#define VW 4
#define KN(name) name##_v4
#define VZ ((v4){0})
#include "d1_batchlane.c"
#undef V
#undef VW
#undef KN
#undef VZ

/* ---- instantiate the kernels for a single scalar lane ---- */
#define V double
#define VW 1
#define KN(name) name##_s
#define VZ 0.0
#include "d1_batchlane.c"
#undef V
#undef VW
#undef KN
#undef VZ

/* ================= 8x8 double transpose (the ONLY shuffles in the engine) ========= */
static inline void tr8(v8 r[8])
{
    const v8i u_lo = {0,8,2,10,4,12,6,14}, u_hi = {1,9,3,11,5,13,7,15};
    const v8i s_lo = {0,1,8,9,4,5,12,13},  s_hi = {2,3,10,11,6,7,14,15};
    const v8i q_lo = {0,1,2,3,8,9,10,11},  q_hi = {4,5,6,7,12,13,14,15};
    v8 t0 = __builtin_shuffle(r[0], r[1], u_lo), t1 = __builtin_shuffle(r[0], r[1], u_hi);
    v8 t2 = __builtin_shuffle(r[2], r[3], u_lo), t3 = __builtin_shuffle(r[2], r[3], u_hi);
    v8 t4 = __builtin_shuffle(r[4], r[5], u_lo), t5 = __builtin_shuffle(r[4], r[5], u_hi);
    v8 t6 = __builtin_shuffle(r[6], r[7], u_lo), t7 = __builtin_shuffle(r[6], r[7], u_hi);
    v8 s0 = __builtin_shuffle(t0, t2, s_lo), s2 = __builtin_shuffle(t0, t2, s_hi);
    v8 s1 = __builtin_shuffle(t1, t3, s_lo), s3 = __builtin_shuffle(t1, t3, s_hi);
    v8 s4 = __builtin_shuffle(t4, t6, s_lo), s6 = __builtin_shuffle(t4, t6, s_hi);
    v8 s5 = __builtin_shuffle(t5, t7, s_lo), s7 = __builtin_shuffle(t5, t7, s_hi);
    r[0] = __builtin_shuffle(s0, s4, q_lo); r[4] = __builtin_shuffle(s0, s4, q_hi);
    r[1] = __builtin_shuffle(s1, s5, q_lo); r[5] = __builtin_shuffle(s1, s5, q_hi);
    r[2] = __builtin_shuffle(s2, s6, q_lo); r[6] = __builtin_shuffle(s2, s6, q_hi);
    r[3] = __builtin_shuffle(s3, s7, q_lo); r[7] = __builtin_shuffle(s3, s7, q_hi);
}

/* 4x4 double transpose (ymm) */
static inline void tr4(v4 r[4])
{
    const v4i p_lo = {0,4,2,6}, p_hi = {1,5,3,7};
    const v4i q_lo = {0,1,4,5}, q_hi = {2,3,6,7};
    v4 s0 = __builtin_shuffle(r[0], r[1], p_lo), s1 = __builtin_shuffle(r[0], r[1], p_hi);
    v4 s2 = __builtin_shuffle(r[2], r[3], p_lo), s3 = __builtin_shuffle(r[2], r[3], p_hi);
    r[0] = __builtin_shuffle(s0, s2, q_lo);
    r[1] = __builtin_shuffle(s1, s3, q_lo);
    r[2] = __builtin_shuffle(s0, s2, q_hi);
    r[3] = __builtin_shuffle(s1, s3, q_hi);
}

/* one AoS row <-> split-complex SoA double arrays (SoA side 64B-aligned) */
static void deint8(const double *aos, double *re, double *im, int L)
{
    const v8i ev = {0,2,4,6,8,10,12,14}, od = {1,3,5,7,9,11,13,15};
    int j = 0;
    for (; j + 8 <= L; j += 8) {
        v8 a = *(const v8u *)(aos + 2*j), b = *(const v8u *)(aos + 2*j + 8);
        *(v8 *)(re + j) = __builtin_shuffle(a, b, ev);
        *(v8 *)(im + j) = __builtin_shuffle(a, b, od);
    }
    for (; j < L; ++j) { re[j] = aos[2*j]; im[j] = aos[2*j + 1]; }
}
static void inter8(double *aos, const double *re, const double *im, int L)
{
    const v8i lo = {0,8,1,9,2,10,3,11}, hi = {4,12,5,13,6,14,7,15};
    int j = 0;
    for (; j + 8 <= L; j += 8) {
        v8 r = *(const v8 *)(re + j), i = *(const v8 *)(im + j);
        *(v8u *)(aos + 2*j)     = __builtin_shuffle(r, i, lo);
        *(v8u *)(aos + 2*j + 8) = __builtin_shuffle(r, i, hi);
    }
    for (; j < L; ++j) { aos[2*j] = re[j]; aos[2*j + 1] = im[j]; }
}

/* ===== single-transform four-step kernels: the SAME zero-shuffle vector FFTs, with
 * the lanes carrying the outer decimation residue n1 instead of the batch index.
 * SoA in/out, so the only shuffles are the middle transpose. =================== */

/* L=64 = 8(n1, zmm lanes) x 8(n2): x[8 n2 + n1]; out X[k2 + 8 k1] in natural order */
static void fs64(const fft1d_plan *p, const double *re, const double *im,
                 double *qr, double *qi)
{
    const v8 *tw = (const v8 *)p->twv;    /* [k2-1]=re, [7+k2-1]=im, lane n1 */
    v8 Ar[8], Ai[8], Br[8], Bi[8];
    for (int n2 = 0; n2 < 8; ++n2) {
        Ar[n2] = *(const v8 *)(re + 8*n2);
        Ai[n2] = *(const v8 *)(im + 8*n2);
    }
    fft8_v8(Ar, Ai, 1, Br, Bi, 1);            /* inner FFT over n2, per lane n1 */
    for (int k2 = 1; k2 < 8; ++k2) {
        v8 c = tw[k2-1], s = tw[7 + k2-1];
        v8 a = Br[k2], b = Bi[k2];
        Br[k2] = a*c - b*s;
        Bi[k2] = a*s + b*c;
    }
    tr8(Br); tr8(Bi);                       /* [k2][n1] -> [n1][k2] */
    fft8_v8(Br, Bi, 1, Ar, Ai, 1);            /* outer FFT over n1 */
    for (int k1 = 0; k1 < 8; ++k1) {
        *(v8 *)(qr + 8*k1) = Ar[k1];
        *(v8 *)(qi + 8*k1) = Ai[k1];
    }
}

/* L=64 single-shot straight from/to AoS: fs64 with the de/re-interleave shuffles fused
 * into the load/store steps (no deint8/inter8 scratch round trip, no store-forward
 * hazards). Same shuffle count as deint8+inter8 -- the memory traffic is what goes. */
static void fs64_aos(const fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    const v8i ev = {0,2,4,6,8,10,12,14}, od = {1,3,5,7,9,11,13,15};
    const v8i lo = {0,8,1,9,2,10,3,11},  hi = {4,12,5,13,6,14,7,15};
    const double *a = (const double *)in;
    double *o = (double *)out;
    const v8 *tw = (const v8 *)p->twv;
    v8 Ar[8], Ai[8], Br[8], Bi[8];
    for (int n2 = 0; n2 < 8; ++n2) {
        v8 u = *(const v8u *)(a + 16*n2), w = *(const v8u *)(a + 16*n2 + 8);
        Ar[n2] = __builtin_shuffle(u, w, ev);
        Ai[n2] = __builtin_shuffle(u, w, od);
    }
    fft8_v8(Ar, Ai, 1, Br, Bi, 1);
    for (int k2 = 1; k2 < 8; ++k2) {
        v8 c = tw[k2-1], s = tw[7 + k2-1];
        v8 x = Br[k2], y = Bi[k2];
        Br[k2] = x*c - y*s;
        Bi[k2] = x*s + y*c;
    }
    tr8(Br); tr8(Bi);
    fft8_v8(Br, Bi, 1, Ar, Ai, 1);
    for (int k1 = 0; k1 < 8; ++k1) {
        *(v8u *)(o + 16*k1)     = __builtin_shuffle(Ar[k1], Ai[k1], lo);
        *(v8u *)(o + 16*k1 + 8) = __builtin_shuffle(Ar[k1], Ai[k1], hi);
    }
}

/* L=128 single-shot, same fusion */
static void fs128_aos(const fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    const v8i ev = {0,2,4,6,8,10,12,14}, od = {1,3,5,7,9,11,13,15};
    const v8i lo = {0,8,1,9,2,10,3,11},  hi = {4,12,5,13,6,14,7,15};
    const double *a = (const double *)in;
    double *o = (double *)out;
    const v8 *tw = (const v8 *)p->twv;
    v8 Ar[16], Ai[16], Br[16], Bi[16];
    for (int n2 = 0; n2 < 16; ++n2) {
        v8 u = *(const v8u *)(a + 16*n2), w = *(const v8u *)(a + 16*n2 + 8);
        Ar[n2] = __builtin_shuffle(u, w, ev);
        Ai[n2] = __builtin_shuffle(u, w, od);
    }
    fft16_v8(p->tw + 256, Ar, Ai, 1, Br, Bi, 1);
    for (int k2 = 1; k2 < 16; ++k2) {
        v8 c = tw[k2-1], s = tw[15 + k2-1];
        v8 x = Br[k2], y = Bi[k2];
        Br[k2] = x*c - y*s;
        Bi[k2] = x*s + y*c;
    }
    tr8(Br); tr8(Br + 8); tr8(Bi); tr8(Bi + 8);
    fft8_v8(Br, Bi, 1, Ar, Ai, 1);
    fft8_v8(Br + 8, Bi + 8, 1, Ar + 8, Ai + 8, 1);
    for (int k1 = 0; k1 < 8; ++k1) {
        *(v8u *)(o + 32*k1)      = __builtin_shuffle(Ar[k1], Ai[k1], lo);
        *(v8u *)(o + 32*k1 + 8)  = __builtin_shuffle(Ar[k1], Ai[k1], hi);
        *(v8u *)(o + 32*k1 + 16) = __builtin_shuffle(Ar[8+k1], Ai[8+k1], lo);
        *(v8u *)(o + 32*k1 + 24) = __builtin_shuffle(Ar[8+k1], Ai[8+k1], hi);
    }
}

/* L=32 = 4(n1, ymm lanes) x 8(n2): x[4 n2 + n1]; out X[k2 + 8 k1] */
static void fs32(const fft1d_plan *p, const double *re, const double *im,
                 double *qr, double *qi)
{
    const v4 *tw = (const v4 *)p->twv;    /* [k2-1]=re, [7+k2-1]=im, lane n1 */
    v4 Ar[8], Ai[8], Br[8], Bi[8];
    for (int n2 = 0; n2 < 8; ++n2) {
        Ar[n2] = *(const v4 *)(re + 4*n2);
        Ai[n2] = *(const v4 *)(im + 4*n2);
    }
    fft8_v4(Ar, Ai, 1, Br, Bi, 1);
    for (int k2 = 1; k2 < 8; ++k2) {
        v4 c = tw[k2-1], s = tw[7 + k2-1];
        v4 a = Br[k2], b = Bi[k2];
        Br[k2] = a*c - b*s;
        Bi[k2] = a*s + b*c;
    }
    tr4(Br); tr4(Br + 4); tr4(Bi); tr4(Bi + 4);   /* two 4x4 blocks per component */
    fft4_v4(Br, Bi, 1, Ar, Ai, 1);                /* outer over n1, lanes k2 = 0..3 */
    fft4_v4(Br + 4, Bi + 4, 1, Ar + 4, Ai + 4, 1);/* lanes k2 = 4..7 */
    for (int k1 = 0; k1 < 4; ++k1) {
        *(v4 *)(qr + 8*k1)     = Ar[k1];
        *(v4 *)(qi + 8*k1)     = Ai[k1];
        *(v4 *)(qr + 8*k1 + 4) = Ar[4 + k1];
        *(v4 *)(qi + 8*k1 + 4) = Ai[4 + k1];
    }
}

/* L=32 single-shot straight from/to AoS (same fusion as fs64_aos, ymm blocks) */
static void fs32_aos(const fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    const v4i ev4 = {0,2,4,6}, od4 = {1,3,5,7}, lo4 = {0,4,1,5}, hi4 = {2,6,3,7};
    const double *a = (const double *)in;
    double *o = (double *)out;
    const v4 *tw = (const v4 *)p->twv;
    v4 Ar[8], Ai[8], Br[8], Bi[8];
    for (int n2 = 0; n2 < 8; ++n2) {
        v4 u = *(const v4u *)(a + 8*n2), w = *(const v4u *)(a + 8*n2 + 4);
        Ar[n2] = __builtin_shuffle(u, w, ev4);
        Ai[n2] = __builtin_shuffle(u, w, od4);
    }
    fft8_v4(Ar, Ai, 1, Br, Bi, 1);
    for (int k2 = 1; k2 < 8; ++k2) {
        v4 c = tw[k2-1], s = tw[7 + k2-1];
        v4 x = Br[k2], y = Bi[k2];
        Br[k2] = x*c - y*s;
        Bi[k2] = x*s + y*c;
    }
    tr4(Br); tr4(Br + 4); tr4(Bi); tr4(Bi + 4);
    fft4_v4(Br, Bi, 1, Ar, Ai, 1);
    fft4_v4(Br + 4, Bi + 4, 1, Ar + 4, Ai + 4, 1);
    for (int k1 = 0; k1 < 4; ++k1) {
        *(v4u *)(o + 16*k1)      = __builtin_shuffle(Ar[k1], Ai[k1], lo4);
        *(v4u *)(o + 16*k1 + 4)  = __builtin_shuffle(Ar[k1], Ai[k1], hi4);
        *(v4u *)(o + 16*k1 + 8)  = __builtin_shuffle(Ar[4+k1], Ai[4+k1], lo4);
        *(v4u *)(o + 16*k1 + 12) = __builtin_shuffle(Ar[4+k1], Ai[4+k1], hi4);
    }
}

/* L=60 single-shot straight from/to AoS (fs60 with fused de/re-interleave) */
static void fs60_aos(const fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    static const int in15[15] = {0,3,6,9,12, 5,8,11,14,2, 10,13,1,4,7};
    static const int om15[15] = {0,6,12,3,9, 10,1,7,13,4, 5,11,2,8,14};
    const v4i ev4 = {0,2,4,6}, od4 = {1,3,5,7}, lo4 = {0,4,1,5}, hi4 = {2,6,3,7};
    const double *a = (const double *)in;
    double *o = (double *)out;
    const v4 *tw = (const v4 *)p->twv;
    v4 Xr[15], Xi[15], Pr[15], Pi[15], Wr[16], Wi[16];
    for (int n2 = 0; n2 < 15; ++n2) {
        v4 u = *(const v4u *)(a + 8*n2), w = *(const v4u *)(a + 8*n2 + 4);
        Xr[n2] = __builtin_shuffle(u, w, ev4);
        Xi[n2] = __builtin_shuffle(u, w, od4);
    }
    for (int t = 0; t < 15; ++t) { Pr[t] = Xr[in15[t]]; Pi[t] = Xi[in15[t]]; }
    for (int j = 0; j < 5; ++j) fft3_v4(Pr + j, Pi + j, 5);
    for (int t = 0; t < 3; ++t) fft5_v4(Pr + 5*t, Pi + 5*t, 1);
    for (int t = 0; t < 15; ++t) { Wr[om15[t]] = Pr[t]; Wi[om15[t]] = Pi[t]; }
    for (int k2 = 1; k2 < 15; ++k2) {
        v4 c = tw[k2-1], s = tw[14 + k2-1];
        v4 x = Wr[k2], y = Wi[k2];
        Wr[k2] = x*c - y*s;
        Wi[k2] = x*s + y*c;
    }
    Wr[15] = Wr[0]; Wi[15] = Wi[0];
    for (int g = 0; g < 4; ++g) { tr4(Wr + 4*g); tr4(Wi + 4*g); }
    for (int g = 0; g < 4; ++g) {
        v4 Rr[4], Ri[4];
        fft4_v4(Wr + 4*g, Wi + 4*g, 1, Rr, Ri, 1);
        for (int k1 = 0; k1 < 4; ++k1) {
            double *q = o + 2*(15*k1 + 4*g);
            if (g < 3) {
                *(v4u *)q       = __builtin_shuffle(Rr[k1], Ri[k1], lo4);
                *(v4u *)(q + 4) = __builtin_shuffle(Rr[k1], Ri[k1], hi4);
            } else {
                *(v4u *)q = __builtin_shuffle(Rr[k1], Ri[k1], lo4);
                q[4] = Rr[k1][2]; q[5] = Ri[k1][2];
            }
        }
    }
}

/* L=128 = 8(n1, zmm lanes) x 16(n2): x[8 n2 + n1]; out X[k2 + 16 k1] */
static void fs128(const fft1d_plan *p, const double *re, const double *im,
                  double *qr, double *qi)
{
    const v8 *tw = (const v8 *)p->twv;    /* [k2-1]=re, [15+k2-1]=im, k2=1..15 */
    v8 Ar[16], Ai[16], Br[16], Bi[16];
    for (int n2 = 0; n2 < 16; ++n2) {
        Ar[n2] = *(const v8 *)(re + 8*n2);
        Ai[n2] = *(const v8 *)(im + 8*n2);
    }
    fft16_v8(p->tw + 256, Ar, Ai, 1, Br, Bi, 1);  /* inner FFT16 over n2, per lane n1 */
    for (int k2 = 1; k2 < 16; ++k2) {
        v8 c = tw[k2-1], s = tw[15 + k2-1];
        v8 a = Br[k2], b = Bi[k2];
        Br[k2] = a*c - b*s;
        Bi[k2] = a*s + b*c;
    }
    tr8(Br); tr8(Br + 8); tr8(Bi); tr8(Bi + 8);
    fft8_v8(Br, Bi, 1, Ar, Ai, 1);                /* outer over n1, lanes k2 = 0..7 */
    fft8_v8(Br + 8, Bi + 8, 1, Ar + 8, Ai + 8, 1);/* lanes k2 = 8..15 */
    for (int k1 = 0; k1 < 8; ++k1) {
        *(v8 *)(qr + 16*k1)     = Ar[k1];
        *(v8 *)(qi + 16*k1)     = Ai[k1];
        *(v8 *)(qr + 16*k1 + 8) = Ar[8 + k1];
        *(v8 *)(qi + 16*k1 + 8) = Ai[8 + k1];
    }
}

/* L=60 = 4(n1, ymm lanes) x 15(n2): x[4 n2 + n1]; inner FFT15 = twiddle-free PFA 3x5;
 * out X[k2 + 15 k1] */
static void fs60(const fft1d_plan *p, const double *re, const double *im,
                 double *qr, double *qi)
{
    static const int in15[15] = {0,3,6,9,12, 5,8,11,14,2, 10,13,1,4,7};
    static const int om15[15] = {0,6,12,3,9, 10,1,7,13,4, 5,11,2,8,14};
    const v4 *tw = (const v4 *)p->twv;    /* [k2-1]=re, [14+k2-1]=im, lane n1 */
    v4 Xr[15], Xi[15], Pr[15], Pi[15], Wr[16], Wi[16];
    for (int n2 = 0; n2 < 15; ++n2) {
        Xr[n2] = *(const v4 *)(re + 4*n2);
        Xi[n2] = *(const v4 *)(im + 4*n2);
    }
    for (int t = 0; t < 15; ++t) { Pr[t] = Xr[in15[t]]; Pi[t] = Xi[in15[t]]; }
    for (int j = 0; j < 5; ++j) fft3_v4(Pr + j, Pi + j, 5);
    for (int t = 0; t < 3; ++t) fft5_v4(Pr + 5*t, Pi + 5*t, 1);
    for (int t = 0; t < 15; ++t) { Wr[om15[t]] = Pr[t]; Wi[om15[t]] = Pi[t]; }
    for (int k2 = 1; k2 < 15; ++k2) {
        v4 c = tw[k2-1], s = tw[14 + k2-1];
        v4 a = Wr[k2], b = Wi[k2];
        Wr[k2] = a*c - b*s;
        Wi[k2] = a*s + b*c;
    }
    Wr[15] = Wr[0]; Wi[15] = Wi[0];       /* pad lane for the 4th transpose block */
    for (int g = 0; g < 4; ++g) { tr4(Wr + 4*g); tr4(Wi + 4*g); }
    for (int g = 0; g < 4; ++g) {
        v4 Rr[4], Ri[4];
        fft4_v4(Wr + 4*g, Wi + 4*g, 1, Rr, Ri, 1);
        for (int k1 = 0; k1 < 4; ++k1) {
            if (g < 3) {
                *(v4u *)(qr + 15*k1 + 4*g) = Rr[k1];
                *(v4u *)(qi + 15*k1 + 4*g) = Ri[k1];
            } else {
                for (int l = 0; l < 3; ++l) {
                    qr[15*k1 + 12 + l] = Rr[k1][l];
                    qi[15*k1 + 12 + l] = Ri[k1][l];
                }
            }
        }
    }
}

/* primes 13/31, single transform: the SAME symmetric-pair dense DFT, but with the zmm
 * lanes carrying the OUTPUT index k (u_j/v_j broadcast), so one transform fills the
 * machine. Fold and epilogue are vector permute-reversals, not scalar loops, and the
 * L=13 FMA chain is split even/odd-j (d1_prime's split-accumulator lesson: the 4-chain
 * x 6-deep loop was FMA-latency-bound). Table lanes beyond H are zero-padded. */
static void dsk13(const double *Ct, const double *St,
                  const double *re, const double *im, double *qr, double *qi)
{
    const __m512i rev6 = _mm512_setr_epi64(5, 4, 3, 2, 1, 0, 6, 7);
    const __mmask8 m6 = 0x3F;
    __m512d fr = _mm512_maskz_loadu_pd(m6, re + 1);
    __m512d gr = _mm512_permutexvar_pd(rev6, _mm512_maskz_loadu_pd(m6, re + 7));
    __m512d fi = _mm512_maskz_loadu_pd(m6, im + 1);
    __m512d gi = _mm512_permutexvar_pd(rev6, _mm512_maskz_loadu_pd(m6, im + 7));
    __m512d urv = _mm512_add_pd(fr, gr), vrv = _mm512_sub_pd(fr, gr);
    __m512d uiv = _mm512_add_pd(fi, gi), viv = _mm512_sub_pd(fi, gi);
    double s0r = re[0], s0i = im[0];
    qr[0] = s0r + _mm512_reduce_add_pd(urv);   /* pad lanes are zero */
    qi[0] = s0i + _mm512_reduce_add_pd(uiv);
    /* lane broadcasts stay in-register (vpermpd): a store + 8B reload of a fresh 64B
     * store blocks store-forwarding and cost 13 us/call measured 0.023 -> 0.036 */
    __m512d A0r = _mm512_setzero_pd(), A0i = A0r, B0r = A0r, B0i = A0r;
    __m512d A1r = A0r, A1i = A0r, B1r = A0r, B1i = A0r;
    for (int j = 0; j < 6; j += 2) {
        __m512i bj0 = _mm512_set1_epi64(j), bj1 = _mm512_set1_epi64(j + 1);
        __m512d c0 = _mm512_load_pd(Ct + 8*j), s0 = _mm512_load_pd(St + 8*j);
        __m512d c1 = _mm512_load_pd(Ct + 8*j + 8), s1 = _mm512_load_pd(St + 8*j + 8);
        A0r = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj0, urv), c0, A0r);
        A0i = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj0, uiv), c0, A0i);
        B0r = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj0, vrv), s0, B0r);
        B0i = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj0, viv), s0, B0i);
        A1r = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj1, urv), c1, A1r);
        A1i = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj1, uiv), c1, A1i);
        B1r = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj1, vrv), s1, B1r);
        B1i = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj1, viv), s1, B1i);
    }
    __m512d Ar = _mm512_add_pd(A0r, A1r), Ai = _mm512_add_pd(A0i, A1i);
    __m512d Br = _mm512_add_pd(B0r, B1r), Bi = _mm512_add_pd(B0i, B1i);
    __m512d s0rv = _mm512_set1_pd(s0r), s0iv = _mm512_set1_pd(s0i);
    __m512d qlr = _mm512_add_pd(s0rv, _mm512_add_pd(Ar, Bi));
    __m512d qli = _mm512_sub_pd(_mm512_add_pd(s0iv, Ai), Br);
    __m512d qhr = _mm512_sub_pd(_mm512_add_pd(s0rv, Ar), Bi);
    __m512d qhi = _mm512_add_pd(_mm512_add_pd(s0iv, Ai), Br);
    _mm512_mask_storeu_pd(qr + 1, m6, qlr);
    _mm512_mask_storeu_pd(qi + 1, m6, qli);
    _mm512_mask_storeu_pd(qr + 7, m6, _mm512_permutexvar_pd(rev6, qhr));
    _mm512_mask_storeu_pd(qi + 7, m6, _mm512_permutexvar_pd(rev6, qhi));
}

/* L=13 single-shot straight from/to the interleaved AoS buffers (no deint8/inter8):
 * the fold rows come out of the raw complex loads via ONE vpermt2pd each (d1_prime's
 * prologue trick). Rationale: dsk13 after deint8 re-reads bytes of freshly written
 * mixed vector+scalar stores -- a guaranteed store-forward stall, measured 0.023 ->
 * 0.036 us on the B=1 m=1 cell. Reading the caller's input directly avoids it. */
static void fs13_aos(const double *Ct, const double *St,
                     const double _Complex *in, double _Complex *out)
{
    const double *a = (const double *)in;
    double *o = (double *)out;
    const __mmask8 m4 = 0x0F, m6 = 0x3F;
    const __m512i fev = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i fod = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    const __m512i gev = _mm512_setr_epi64(10, 8, 6, 4, 2, 0, 12, 14);
    const __m512i god = _mm512_setr_epi64(11, 9, 7, 5, 3, 1, 13, 15);
    __m512d l0 = _mm512_loadu_pd(a + 2), l1 = _mm512_maskz_loadu_pd(m4, a + 10);
    __m512d h0 = _mm512_loadu_pd(a + 14), h1 = _mm512_maskz_loadu_pd(m4, a + 22);
    __m512d fr = _mm512_permutex2var_pd(l0, fev, l1);   /* x1..x6 re, pad 0 */
    __m512d fi = _mm512_permutex2var_pd(l0, fod, l1);
    __m512d gr = _mm512_permutex2var_pd(h0, gev, h1);   /* x12..x7 re, pad 0 */
    __m512d gi = _mm512_permutex2var_pd(h0, god, h1);
    __m512d urv = _mm512_add_pd(fr, gr), vrv = _mm512_sub_pd(fr, gr);
    __m512d uiv = _mm512_add_pd(fi, gi), viv = _mm512_sub_pd(fi, gi);
    double s0r = a[0], s0i = a[1];
    double x0r = s0r + _mm512_reduce_add_pd(urv);
    double x0i = s0i + _mm512_reduce_add_pd(uiv);
    __m512d A0r = _mm512_setzero_pd(), A0i = A0r, B0r = A0r, B0i = A0r;
    __m512d A1r = A0r, A1i = A0r, B1r = A0r, B1i = A0r;
    for (int j = 0; j < 6; j += 2) {
        __m512i bj0 = _mm512_set1_epi64(j), bj1 = _mm512_set1_epi64(j + 1);
        __m512d c0 = _mm512_load_pd(Ct + 8*j), s0 = _mm512_load_pd(St + 8*j);
        __m512d c1 = _mm512_load_pd(Ct + 8*j + 8), s1 = _mm512_load_pd(St + 8*j + 8);
        A0r = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj0, urv), c0, A0r);
        A0i = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj0, uiv), c0, A0i);
        B0r = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj0, vrv), s0, B0r);
        B0i = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj0, viv), s0, B0i);
        A1r = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj1, urv), c1, A1r);
        A1i = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj1, uiv), c1, A1i);
        B1r = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj1, vrv), s1, B1r);
        B1i = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj1, viv), s1, B1i);
    }
    __m512d Ar = _mm512_add_pd(A0r, A1r), Ai = _mm512_add_pd(A0i, A1i);
    __m512d Br = _mm512_add_pd(B0r, B1r), Bi = _mm512_add_pd(B0i, B1i);
    __m512d s0rv = _mm512_set1_pd(s0r), s0iv = _mm512_set1_pd(s0i);
    __m512d qlr = _mm512_add_pd(s0rv, _mm512_add_pd(Ar, Bi));   /* X1..X6 */
    __m512d qli = _mm512_sub_pd(_mm512_add_pd(s0iv, Ai), Br);
    __m512d qhr = _mm512_sub_pd(_mm512_add_pd(s0rv, Ar), Bi);   /* X12..X7 (lane k-1) */
    __m512d qhi = _mm512_add_pd(_mm512_add_pd(s0iv, Ai), Br);
    const __m512i ilo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i ihi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    const __m512i hlo = _mm512_setr_epi64(5, 13, 4, 12, 3, 11, 2, 10);
    const __m512i hhi = _mm512_setr_epi64(1, 9, 0, 8, 2, 10, 3, 11);
    o[0] = x0r; o[1] = x0i;
    _mm512_storeu_pd(o + 2, _mm512_permutex2var_pd(qlr, ilo, qli));      /* X1..X4  */
    _mm512_mask_storeu_pd(o + 10, m4, _mm512_permutex2var_pd(qlr, ihi, qli)); /* X5,X6 */
    _mm512_storeu_pd(o + 14, _mm512_permutex2var_pd(qhr, hlo, qhi));     /* X7..X10 */
    _mm512_mask_storeu_pd(o + 22, m4, _mm512_permutex2var_pd(qhr, hhi, qhi)); /* X11,X12 */
    (void)m6;
}

static void dsk31(const double *Ct, const double *St,
                  const double *re, const double *im, double *qr, double *qi)
{
    const __m512i rev8 = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    const __m512i rev7 = _mm512_setr_epi64(6, 5, 4, 3, 2, 1, 0, 7);
    const __mmask8 m7 = 0x7F;
    __m512d f0r = _mm512_loadu_pd(re + 1);                    /* x[1..8] */
    __m512d f1r = _mm512_maskz_loadu_pd(m7, re + 9);          /* x[9..15] */
    __m512d g0r = _mm512_permutexvar_pd(rev8, _mm512_loadu_pd(re + 23));  /* x[30..23] */
    __m512d g1r = _mm512_permutexvar_pd(rev7, _mm512_maskz_loadu_pd(m7, re + 16));
    __m512d f0i = _mm512_loadu_pd(im + 1);
    __m512d f1i = _mm512_maskz_loadu_pd(m7, im + 9);
    __m512d g0i = _mm512_permutexvar_pd(rev8, _mm512_loadu_pd(im + 23));
    __m512d g1i = _mm512_permutexvar_pd(rev7, _mm512_maskz_loadu_pd(m7, im + 16));
    __m512d ur0 = _mm512_add_pd(f0r, g0r), vr0 = _mm512_sub_pd(f0r, g0r);
    __m512d ur1 = _mm512_add_pd(f1r, g1r), vr1 = _mm512_sub_pd(f1r, g1r);
    __m512d ui0 = _mm512_add_pd(f0i, g0i), vi0 = _mm512_sub_pd(f0i, g0i);
    __m512d ui1 = _mm512_add_pd(f1i, g1i), vi1 = _mm512_sub_pd(f1i, g1i);
    double us[16] __attribute__((aligned(64))), uis[16] __attribute__((aligned(64)));
    double vs[16] __attribute__((aligned(64))), vis[16] __attribute__((aligned(64)));
    _mm512_store_pd(us, ur0);      _mm512_store_pd(us + 8, ur1);
    _mm512_store_pd(uis, ui0);     _mm512_store_pd(uis + 8, ui1);
    _mm512_store_pd(vs, vr0);      _mm512_store_pd(vs + 8, vr1);
    _mm512_store_pd(vis, vi0);     _mm512_store_pd(vis + 8, vi1);
    double s0r = re[0], s0i = im[0];
    qr[0] = s0r + _mm512_reduce_add_pd(_mm512_add_pd(ur0, ur1));
    qi[0] = s0i + _mm512_reduce_add_pd(_mm512_add_pd(ui0, ui1));
    __m512d Ar0 = _mm512_setzero_pd(), Ai0 = Ar0, Br0 = Ar0, Bi0 = Ar0;
    __m512d Ar1 = Ar0, Ai1 = Ar0, Br1 = Ar0, Bi1 = Ar0;
    for (int j = 0; j < 15; ++j) {
        __m512d uu_r = _mm512_set1_pd(us[j]),  uu_i = _mm512_set1_pd(uis[j]);
        __m512d vv_r = _mm512_set1_pd(vs[j]),  vv_i = _mm512_set1_pd(vis[j]);
        __m512d c0 = _mm512_load_pd(Ct + 16*j), c1 = _mm512_load_pd(Ct + 16*j + 8);
        __m512d s0 = _mm512_load_pd(St + 16*j), s1 = _mm512_load_pd(St + 16*j + 8);
        Ar0 = _mm512_fmadd_pd(uu_r, c0, Ar0); Ar1 = _mm512_fmadd_pd(uu_r, c1, Ar1);
        Ai0 = _mm512_fmadd_pd(uu_i, c0, Ai0); Ai1 = _mm512_fmadd_pd(uu_i, c1, Ai1);
        Br0 = _mm512_fmadd_pd(vv_r, s0, Br0); Br1 = _mm512_fmadd_pd(vv_r, s1, Br1);
        Bi0 = _mm512_fmadd_pd(vv_i, s0, Bi0); Bi1 = _mm512_fmadd_pd(vv_i, s1, Bi1);
    }
    __m512d s0rv = _mm512_set1_pd(s0r), s0iv = _mm512_set1_pd(s0i);
    __m512d ql0r = _mm512_add_pd(s0rv, _mm512_add_pd(Ar0, Bi0));   /* k = 1..8 */
    __m512d ql0i = _mm512_sub_pd(_mm512_add_pd(s0iv, Ai0), Br0);
    __m512d qh0r = _mm512_sub_pd(_mm512_add_pd(s0rv, Ar0), Bi0);   /* k = 30..23 */
    __m512d qh0i = _mm512_add_pd(_mm512_add_pd(s0iv, Ai0), Br0);
    __m512d ql1r = _mm512_add_pd(s0rv, _mm512_add_pd(Ar1, Bi1));   /* k = 9..15 */
    __m512d ql1i = _mm512_sub_pd(_mm512_add_pd(s0iv, Ai1), Br1);
    __m512d qh1r = _mm512_sub_pd(_mm512_add_pd(s0rv, Ar1), Bi1);   /* k = 22..16 */
    __m512d qh1i = _mm512_add_pd(_mm512_add_pd(s0iv, Ai1), Br1);
    _mm512_storeu_pd(qr + 1, ql0r);
    _mm512_storeu_pd(qi + 1, ql0i);
    _mm512_mask_storeu_pd(qr + 9, m7, ql1r);
    _mm512_mask_storeu_pd(qi + 9, m7, ql1i);
    _mm512_storeu_pd(qr + 23, _mm512_permutexvar_pd(rev8, qh0r));
    _mm512_storeu_pd(qi + 23, _mm512_permutexvar_pd(rev8, qh0i));
    _mm512_mask_storeu_pd(qr + 16, m7, _mm512_permutexvar_pd(rev7, qh1r));
    _mm512_mask_storeu_pd(qi + 16, m7, _mm512_permutexvar_pd(rev7, qh1i));
}

/* L=31 single-shot, same fused AoS prologue/epilogue as fs13_aos (two vectors/row) */
static void fs31_aos(const double *Ct, const double *St,
                     const double _Complex *in, double _Complex *out)
{
    const double *a = (const double *)in;
    double *o = (double *)out;
    const __mmask8 m6 = 0x3F;
    const __m512i fev = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i fod = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    const __m512i gev = _mm512_setr_epi64(14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i god = _mm512_setr_epi64(15, 13, 11, 9, 7, 5, 3, 1);
    const __m512i g7e = _mm512_setr_epi64(12, 10, 8, 6, 4, 2, 0, 14);
    const __m512i g7o = _mm512_setr_epi64(13, 11, 9, 7, 5, 3, 1, 15);
    __m512d m0 = _mm512_loadu_pd(a + 2),  m1 = _mm512_loadu_pd(a + 10);
    __m512d m2 = _mm512_loadu_pd(a + 18), m3 = _mm512_maskz_loadu_pd(m6, a + 26);
    __m512d n0 = _mm512_loadu_pd(a + 46), n1 = _mm512_loadu_pd(a + 54);
    __m512d n2 = _mm512_loadu_pd(a + 32), n3 = _mm512_maskz_loadu_pd(m6, a + 40);
    __m512d f0r = _mm512_permutex2var_pd(m0, fev, m1);  /* x1..x8 re */
    __m512d f0i = _mm512_permutex2var_pd(m0, fod, m1);
    __m512d f1r = _mm512_permutex2var_pd(m2, fev, m3);  /* x9..x15 re, pad 0 */
    __m512d f1i = _mm512_permutex2var_pd(m2, fod, m3);
    __m512d g0r = _mm512_permutex2var_pd(n0, gev, n1);  /* x30..x23 re */
    __m512d g0i = _mm512_permutex2var_pd(n0, god, n1);
    __m512d g1r = _mm512_permutex2var_pd(n2, g7e, n3);  /* x22..x16 re, pad 0 */
    __m512d g1i = _mm512_permutex2var_pd(n2, g7o, n3);
    __m512d ur0 = _mm512_add_pd(f0r, g0r), vr0 = _mm512_sub_pd(f0r, g0r);
    __m512d ur1 = _mm512_add_pd(f1r, g1r), vr1 = _mm512_sub_pd(f1r, g1r);
    __m512d ui0 = _mm512_add_pd(f0i, g0i), vi0 = _mm512_sub_pd(f0i, g0i);
    __m512d ui1 = _mm512_add_pd(f1i, g1i), vi1 = _mm512_sub_pd(f1i, g1i);
    double s0r = a[0], s0i = a[1];
    double x0r = s0r + _mm512_reduce_add_pd(_mm512_add_pd(ur0, ur1));
    double x0i = s0i + _mm512_reduce_add_pd(_mm512_add_pd(ui0, ui1));
    double us[16] __attribute__((aligned(64))), uis[16] __attribute__((aligned(64)));
    double vs[16] __attribute__((aligned(64))), vis[16] __attribute__((aligned(64)));
    _mm512_store_pd(us, ur0);      _mm512_store_pd(us + 8, ur1);
    _mm512_store_pd(uis, ui0);     _mm512_store_pd(uis + 8, ui1);
    _mm512_store_pd(vs, vr0);      _mm512_store_pd(vs + 8, vr1);
    _mm512_store_pd(vis, vi0);     _mm512_store_pd(vis + 8, vi1);
    __m512d Ar0 = _mm512_setzero_pd(), Ai0 = Ar0, Br0 = Ar0, Bi0 = Ar0;
    __m512d Ar1 = Ar0, Ai1 = Ar0, Br1 = Ar0, Bi1 = Ar0;
    for (int j = 0; j < 15; ++j) {
        __m512d uu_r = _mm512_set1_pd(us[j]),  uu_i = _mm512_set1_pd(uis[j]);
        __m512d vv_r = _mm512_set1_pd(vs[j]),  vv_i = _mm512_set1_pd(vis[j]);
        __m512d c0 = _mm512_load_pd(Ct + 16*j), c1 = _mm512_load_pd(Ct + 16*j + 8);
        __m512d s0 = _mm512_load_pd(St + 16*j), s1 = _mm512_load_pd(St + 16*j + 8);
        Ar0 = _mm512_fmadd_pd(uu_r, c0, Ar0); Ar1 = _mm512_fmadd_pd(uu_r, c1, Ar1);
        Ai0 = _mm512_fmadd_pd(uu_i, c0, Ai0); Ai1 = _mm512_fmadd_pd(uu_i, c1, Ai1);
        Br0 = _mm512_fmadd_pd(vv_r, s0, Br0); Br1 = _mm512_fmadd_pd(vv_r, s1, Br1);
        Bi0 = _mm512_fmadd_pd(vv_i, s0, Bi0); Bi1 = _mm512_fmadd_pd(vv_i, s1, Bi1);
    }
    __m512d s0rv = _mm512_set1_pd(s0r), s0iv = _mm512_set1_pd(s0i);
    __m512d ql0r = _mm512_add_pd(s0rv, _mm512_add_pd(Ar0, Bi0));   /* X1..X8   */
    __m512d ql0i = _mm512_sub_pd(_mm512_add_pd(s0iv, Ai0), Br0);
    __m512d qh0r = _mm512_sub_pd(_mm512_add_pd(s0rv, Ar0), Bi0);   /* X30..X23 */
    __m512d qh0i = _mm512_add_pd(_mm512_add_pd(s0iv, Ai0), Br0);
    __m512d ql1r = _mm512_add_pd(s0rv, _mm512_add_pd(Ar1, Bi1));   /* X9..X15  */
    __m512d ql1i = _mm512_sub_pd(_mm512_add_pd(s0iv, Ai1), Br1);
    __m512d qh1r = _mm512_sub_pd(_mm512_add_pd(s0rv, Ar1), Bi1);   /* X22..X16 */
    __m512d qh1i = _mm512_add_pd(_mm512_add_pd(s0iv, Ai1), Br1);
    const __m512i ilo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i ihi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    const __m512i r16 = _mm512_setr_epi64(6, 14, 5, 13, 4, 12, 3, 11);
    const __m512i r20 = _mm512_setr_epi64(2, 10, 1, 9, 0, 8, 3, 11);
    const __m512i r23 = _mm512_setr_epi64(7, 15, 6, 14, 5, 13, 4, 12);
    const __m512i r27 = _mm512_setr_epi64(3, 11, 2, 10, 1, 9, 0, 8);
    o[0] = x0r; o[1] = x0i;
    _mm512_storeu_pd(o + 2,  _mm512_permutex2var_pd(ql0r, ilo, ql0i));  /* X1..X4   */
    _mm512_storeu_pd(o + 10, _mm512_permutex2var_pd(ql0r, ihi, ql0i));  /* X5..X8   */
    _mm512_storeu_pd(o + 18, _mm512_permutex2var_pd(ql1r, ilo, ql1i));  /* X9..X12  */
    _mm512_mask_storeu_pd(o + 26, m6,
                          _mm512_permutex2var_pd(ql1r, ihi, ql1i));     /* X13..X15 */
    _mm512_storeu_pd(o + 32, _mm512_permutex2var_pd(qh1r, r16, qh1i));  /* X16..X19 */
    _mm512_mask_storeu_pd(o + 40, m6,
                          _mm512_permutex2var_pd(qh1r, r20, qh1i));     /* X20..X22 */
    _mm512_storeu_pd(o + 46, _mm512_permutex2var_pd(qh0r, r23, qh0i));  /* X23..X26 */
    _mm512_storeu_pd(o + 54, _mm512_permutex2var_pd(qh0r, r27, qh0i));  /* X27..X30 */
}

/* dispatch: 1 if a fast single-transform kernel exists for this L */
static int fs_one(const fft1d_plan *p, const double *re, const double *im,
                  double *qr, double *qi)
{
    switch (p->L) {
    case 13:  dsk13(p->twv, p->twv + 6*8, re, im, qr, qi);      return 1;
    case 31:  dsk31(p->twv, p->twv + 15*2*8, re, im, qr, qi);   return 1;
    case 32:  fs32(p, re, im, qr, qi);  return 1;
    case 60:  fs60(p, re, im, qr, qi);  return 1;
    case 64:  fs64(p, re, im, qr, qi);  return 1;
    case 128: fs128(p, re, im, qr, qi); return 1;
    default: return 0;
    }
}

/* L=128 batched, m=1 only: the last radix-2 combine fused into the transposing
 * store. The plain path's in-place combine + separate tstore8 writes and re-reads a
 * 16 KB y round trip per group of 8 -- with x+y+w scratch already at the 48 KB L1
 * capacity that round trip is all misses. Here the combine feeds tr8 in registers. */
static void fft128_fused(const fft1d_plan *p, double _Complex *dst)
{
    const double *t2 = p->tw + 128;
    v8 *yr = p->yr, *yi = p->yi;
    double *base = (double *)dst;
    fft64s_v8(p->tw, p->xr,     p->xi,     2, yr,      yi,      p->wr, p->wi);
    fft64s_v8(p->tw, p->xr + 1, p->xi + 1, 2, yr + 64, yi + 64, p->wr, p->wi);
    for (int kb = 0; kb < 64; kb += 4) {
        v8 lo[8], hi[8];
        for (int l = 0; l < 4; ++l) {
            int k = kb + l;
            double c = t2[2*k], s = t2[2*k + 1];
            v8 br = yr[64 + k], bi = yi[64 + k];
            v8 tr = c * br - s * bi, ti = c * bi + s * br;
            v8 ar = yr[k], ai = yi[k];
            lo[2*l] = ar + tr; lo[2*l + 1] = ai + ti;
            hi[2*l] = ar - tr; hi[2*l + 1] = ai - ti;
        }
        tr8(lo); tr8(hi);
        for (int i = 0; i < 8; ++i) {
            *(v8u *)(base + (size_t)i*256 + 2*kb)        = lo[i];
            *(v8u *)(base + (size_t)i*256 + 2*(64 + kb)) = hi[i];
        }
    }
}

/* AoS rows (8 consecutive transforms) -> split-complex SoA vectors */
static void tload8(const double _Complex *src, int L, v8 *xr, v8 *xi)
{
    const double *base = (const double *)src;
    int jb = 0;
    for (; jb + 4 <= L; jb += 4) {
        v8 r[8];
        for (int i = 0; i < 8; ++i) r[i] = *(const v8u *)(base + (size_t)i*2*L + 2*jb);
        tr8(r);
        xr[jb]   = r[0]; xi[jb]   = r[1];
        xr[jb+1] = r[2]; xi[jb+1] = r[3];
        xr[jb+2] = r[4]; xi[jb+2] = r[5];
        xr[jb+3] = r[6]; xi[jb+3] = r[7];
    }
    if (jb < L) {   /* masked tr8 tail: no scalar stores for the kernel loads to
                       trip over (partial-vector store-forward stalls) */
        int rem = L - jb;
        __mmask8 mk = (__mmask8)((1u << (2*rem)) - 1);
        v8 r[8];
        for (int i = 0; i < 8; ++i)
            r[i] = (v8)_mm512_maskz_loadu_pd(mk, base + (size_t)i*2*L + 2*jb);
        tr8(r);
        for (int t = 0; t < rem; ++t) { xr[jb + t] = r[2*t]; xi[jb + t] = r[2*t + 1]; }
    }
}

static void tstore8(double _Complex *dst, int L, const v8 *yr, const v8 *yi)
{
    double *base = (double *)dst;
    int jb = 0;
    for (; jb + 4 <= L; jb += 4) {
        v8 r[8];
        r[0] = yr[jb];   r[1] = yi[jb];
        r[2] = yr[jb+1]; r[3] = yi[jb+1];
        r[4] = yr[jb+2]; r[5] = yi[jb+2];
        r[6] = yr[jb+3]; r[7] = yi[jb+3];
        tr8(r);
        for (int i = 0; i < 8; ++i) *(v8u *)(base + (size_t)i*2*L + 2*jb) = r[i];
    }
    if (jb < L) {
        int rem = L - jb;
        __mmask8 mk = (__mmask8)((1u << (2*rem)) - 1);
        v8 r[8];
        for (int t = 0; t < 4; ++t) { r[2*t] = (v8){0}; r[2*t + 1] = (v8){0}; }
        for (int t = 0; t < rem; ++t) { r[2*t] = yr[jb + t]; r[2*t + 1] = yi[jb + t]; }
        tr8(r);
        for (int i = 0; i < 8; ++i)
            _mm512_mask_storeu_pd(base + (size_t)i*2*L + 2*jb, mk, (__m512d)r[i]);
    }
}

/* ============ the chain map, on contiguous split-complex arrays ====================
 * Explicit 512-bit intrinsics: gcc -march=native auto-vectorizes this loop at ymm width
 * only (256-bit preference), which halves sqrt/div throughput -- and the map is the
 * per-step floor of every chained cell. All pointers are 64B-aligned scratch.
 *
 * 1/(1+sqrt(h)) is computed WITHOUT the unpipelined divider (borrowed from
 * d1_composite / d1_prime): rsqrt14 + 2 Newton, an exact-residual FMA correction on
 * sqrt = h*r, then rcp14 + 2 residual-form Newton rounds on 1/(1+sqrt). All FMA-port
 * work that pipelines across the L map vectors; h clamped at 1e-300 so h=0 stays
 * finite. Accuracy ~1 ulp (residual refinements per d1_pow2's chain-gate fight). */
static inline __m512d map_scale(__m512d h)
{
    const __m512d one = _mm512_set1_pd(1.0), half = _mm512_set1_pd(0.5);
    const __m512d th  = _mm512_set1_pd(1.5);
    __m512d hc = _mm512_max_pd(h, _mm512_set1_pd(1e-300));
    __m512d hh = _mm512_mul_pd(hc, half);
    __m512d r  = _mm512_rsqrt14_pd(hc);
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hh, r), r, th));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hh, r), r, th));
    __m512d t = _mm512_mul_pd(hc, r);                       /* ~sqrt(h) */
    t = _mm512_fmadd_pd(_mm512_mul_pd(half, r), _mm512_fnmadd_pd(t, t, hc), t);
    __m512d d = _mm512_add_pd(one, t);
    __m512d c = _mm512_rcp14_pd(d);
    c = _mm512_fmadd_pd(c, _mm512_fnmadd_pd(d, c, one), c);
    c = _mm512_fmadd_pd(c, _mm512_fnmadd_pd(d, c, one), c);
    return c;
}

static void map_apply(const double *restrict zr, const double *restrict zi,
                      const double *restrict cr, const double *restrict ci,
                      double *restrict sr, double *restrict si, int n)
{
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d re = _mm512_add_pd(_mm512_load_pd(zr + i), _mm512_load_pd(cr + i));
        __m512d im = _mm512_add_pd(_mm512_load_pd(zi + i), _mm512_load_pd(ci + i));
        __m512d h  = _mm512_fmadd_pd(re, re, _mm512_mul_pd(im, im));
        __m512d sc = map_scale(h);
        _mm512_store_pd(sr + i, _mm512_mul_pd(re, sc));
        _mm512_store_pd(si + i, _mm512_mul_pd(im, sc));
    }
    if (i < n) {   /* masked tail (zeroing loads: no junk-lane denormal/NaN assists) */
        __mmask8 mk = (__mmask8)((1u << (n - i)) - 1);
        __m512d re = _mm512_add_pd(_mm512_maskz_load_pd(mk, zr + i),
                                   _mm512_maskz_load_pd(mk, cr + i));
        __m512d im = _mm512_add_pd(_mm512_maskz_load_pd(mk, zi + i),
                                   _mm512_maskz_load_pd(mk, ci + i));
        __m512d h  = _mm512_fmadd_pd(re, re, _mm512_mul_pd(im, im));
        const __m512d one = _mm512_set1_pd(1.0);
        __m512d sc = _mm512_div_pd(one, _mm512_add_pd(one, _mm512_sqrt_pd(h)));
        _mm512_mask_store_pd(sr + i, mk, _mm512_mul_pd(re, sc));
        _mm512_mask_store_pd(si + i, mk, _mm512_mul_pd(im, sc));
    }
}

/* Twiddles from long-double phase (d1_pow2's chain-gate lesson: double M_PI carries a
 * biased ~2e-16 phase error that a long map chain amplifies; 80-bit pi gives
 * correctly-rounded-double tables). r is the mod-reduced integer numerator. */
static double tw_cosl(long r, long L)
{ return (double)cosl(2.0L * 3.141592653589793238462643383279502884L * r / L); }
static double tw_sinl(long r, long L)
{ return (double)sinl(2.0L * 3.141592653589793238462643383279502884L * r / L); }

/* =============================== API ============================================= */
const char *fft1d_name(void) { return "d1_batchlane"; }
const char *fft1d_description(void)
{
    return "SoA 8-lane zmm batch-lane engine: split-complex across-batch kernels "
           "(densesym 13/31, CT 32/64/128, PFA 60), fused-AoS single-shot kernels, "
           "L1-blocked fused FFT+Newton-map chain";
}

int fft1d_supports(int L)
{
    return L == 13 || L == 31 || L == 32 || L == 60 || L == 64 || L == 128;
}

fft1d_plan *fft1d_create(int L, int batch)
{
    if (!fft1d_supports(L) || batch < 1) return NULL;
    fft1d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L; p->batch = batch;

    /* SoA scratch: 8 arrays of L vectors each, 64B aligned */
    void *sb = NULL;
    if (posix_memalign(&sb, 64, (size_t)8 * L * sizeof(v8))) { free(p); return NULL; }
    p->scratch_base = sb;
    v8 *s = (v8 *)sb;
    p->xr = s;       p->xi = s + L;
    p->yr = s + 2*L; p->yi = s + 3*L;
    p->wr = s + 4*L; p->wi = s + 5*L;
    p->cr = s + 6*L; p->ci = s + 7*L;

    if (L == 13 || L == 31) {
        int H = (L - 1) / 2;
        p->tw = malloc((size_t)2 * H * H * sizeof(double));
        if (!p->tw) { fft1d_destroy(p); return NULL; }
        double *C = p->tw, *S = p->tw + H*H;
        for (int j = 1; j <= H; ++j)
            for (int k = 1; k <= H; ++k) {
                C[(j-1)*H + (k-1)] = tw_cosl((j * k) % L, L);
                S[(j-1)*H + (k-1)] = tw_sinl((j * k) % L, L);
            }
    } else if (L == 32 || L == 64) {
        int R = (L == 32) ? 4 : 8;             /* outer radix over n1 */
        p->tw = malloc((size_t)2 * R * 8 * sizeof(double));
        if (!p->tw) { fft1d_destroy(p); return NULL; }
        for (int n1 = 0; n1 < R; ++n1)
            for (int k2 = 0; k2 < 8; ++k2) {
                p->tw[2*(n1*8 + k2)]     = tw_cosl((n1 * k2) % L, L);
                p->tw[2*(n1*8 + k2) + 1] = -tw_sinl((n1 * k2) % L, L);
            }
    } else if (L == 128) {
        /* [0..127]  : FFT64's W64^{n1 k2} table (n1,k2 in 0..7)
           [128..255]: W128^k for k = 0..63
           [256..273]: FFT16's W16^{n1 k2} table (n1,k2 in 1..3) */
        p->tw = malloc((size_t)(128 + 128 + 18) * sizeof(double));
        if (!p->tw) { fft1d_destroy(p); return NULL; }
        for (int n1 = 0; n1 < 8; ++n1)
            for (int k2 = 0; k2 < 8; ++k2) {
                p->tw[2*(n1*8 + k2)]     = tw_cosl((n1 * k2) % 64, 64);
                p->tw[2*(n1*8 + k2) + 1] = -tw_sinl((n1 * k2) % 64, 64);
            }
        for (int k = 0; k < 64; ++k) {
            p->tw[128 + 2*k]     = tw_cosl(k, 128);
            p->tw[128 + 2*k + 1] = -tw_sinl(k, 128);
        }
        for (int n1 = 1; n1 < 4; ++n1)
            for (int k2 = 1; k2 < 4; ++k2) {
                p->tw[256 + 2*((n1-1)*3 + (k2-1))]     = tw_cosl((n1 * k2) % 16, 16);
                p->tw[256 + 2*((n1-1)*3 + (k2-1)) + 1] = -tw_sinl((n1 * k2) % 16, 16);
            }
    } else if (L == 60) {
        /* Good-Thomas maps: input ruritanian, output CRT -> plain DFTs, no twiddles.
           w laid out as [n3][n4][n5] = n3*20 + n4*5 + n5. */
        p->im = malloc(60 * sizeof(int));
        p->om = malloc(60 * sizeof(int));
        if (!p->im || !p->om) { fft1d_destroy(p); return NULL; }
        for (int n3 = 0; n3 < 3; ++n3)
            for (int n4 = 0; n4 < 4; ++n4)
                for (int n5 = 0; n5 < 5; ++n5) {
                    p->im[n3*20 + n4*5 + n5] = (20*n3 + 15*n4 + 12*n5) % 60;
                    p->om[n3*20 + n4*5 + n5] = (40*n3 + 45*n4 + 36*n5) % 60;
                }
    }

    /* k-lane densesym tables for the single-transform prime kernels: lane = k-1 */
    if (L == 13 || L == 31) {
        int H = (L - 1) / 2, G = (H + 7) / 8;
        void *tv = NULL;
        if (posix_memalign(&tv, 64, (size_t)2 * H * G * 8 * sizeof(double))) {
            fft1d_destroy(p); return NULL;
        }
        p->twv = tv;
        double *Cv = p->twv, *Sv = p->twv + (size_t)H * G * 8;
        for (int j = 0; j < H; ++j)
            for (int g = 0; g < G; ++g)
                for (int l = 0; l < 8; ++l) {
                    int k = g*8 + l + 1;
                    double c = 0.0, s = 0.0;
                    if (k <= H) {
                        c = tw_cosl(((j+1) * k) % L, L);
                        s = tw_sinl(((j+1) * k) % L, L);
                    }
                    Cv[(j*G + g)*8 + l] = c;
                    Sv[(j*G + g)*8 + l] = s;
                }
    }

    /* vector twiddles for the single-transform four-step kernels: lane = n1 */
    if (L == 32 || L == 60 || L == 64 || L == 128) {
        int lanes = (L == 32 || L == 60) ? 4 : 8;   /* outer radix = lane count */
        int s = L / lanes;                           /* inner FFT length */
        void *tv = NULL;
        if (posix_memalign(&tv, 64, (size_t)2 * (s - 1) * lanes * sizeof(double))) {
            fft1d_destroy(p); return NULL;
        }
        p->twv = tv;
        double *tre = p->twv, *tim = p->twv + (size_t)(s - 1) * lanes;
        for (int k2 = 1; k2 < s; ++k2)
            for (int n1 = 0; n1 < lanes; ++n1) {
                tre[(k2-1)*lanes + n1] = tw_cosl((n1 * k2) % L, L);
                tim[(k2-1)*lanes + n1] = -tw_sinl((n1 * k2) % L, L);
            }
    }
    return p;
}

void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    const int L = p->L;
    int b = 0;
    for (; b + 8 <= p->batch; b += 8) {
        tload8(in + (size_t)b * L, L, p->xr, p->xi);
        if (L == 128) { fft128_fused(p, out + (size_t)b * L); continue; }
        dofft_v8(p, p->xr, p->xi, p->yr, p->yi, p->wr, p->wi);
        tstore8(out + (size_t)b * L, L, p->yr, p->yi);
    }
    for (; b < p->batch; ++b) {
        if (L == 13) { fs13_aos(p->twv, p->twv + 48, in + (size_t)b * L,
                                out + (size_t)b * L); continue; }
        if (L == 31) { fs31_aos(p->twv, p->twv + 240, in + (size_t)b * L,
                                out + (size_t)b * L); continue; }
        if (L == 32) { fs32_aos(p, in + (size_t)b * L, out + (size_t)b * L); continue; }
        if (L == 60) { fs60_aos(p, in + (size_t)b * L, out + (size_t)b * L); continue; }
        if (L == 64) { fs64_aos(p, in + (size_t)b * L, out + (size_t)b * L); continue; }
        if (L == 128) { fs128_aos(p, in + (size_t)b * L, out + (size_t)b * L); continue; }
        double *sxr = (double *)p->xr, *sxi = (double *)p->xi;
        double *syr = (double *)p->yr, *syi = (double *)p->yi;
        double *swr = (double *)p->wr, *swi = (double *)p->wi;
        deint8((const double *)(in + (size_t)b * L), sxr, sxi, L);
        if (!fs_one(p, sxr, sxi, syr, syi))
            dofft_s(p, sxr, sxi, syr, syi, swr, swi);
        inter8((double *)(out + (size_t)b * L), syr, syi, L);
    }
}

/* Own the whole m-step chain: per group of 8 transforms, state + c + scratch stay
 * L1-resident across ALL m steps; boundary transposes happen once, not per step. */
void fft1d_chain(fft1d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    const int L = p->L;
    int b = 0;
    for (; b + 8 <= p->batch; b += 8) {
        tload8(x0 + (size_t)b * L, L, p->xr, p->xi);
        tload8(c  + (size_t)b * L, L, p->cr, p->ci);
        for (int s = 0; s < m; ++s) {
            dofft_v8(p, p->xr, p->xi, p->yr, p->yi, p->wr, p->wi);
            map_apply((double *)p->yr, (double *)p->yi, (double *)p->cr, (double *)p->ci,
                      (double *)p->xr, (double *)p->xi, L * 8);
        }
        tstore8(final_out + (size_t)b * L, L, p->xr, p->xi);
    }
    for (; b < p->batch; ++b) {
        double *sxr = (double *)p->xr, *sxi = (double *)p->xi;
        double *syr = (double *)p->yr, *syi = (double *)p->yi;
        double *swr = (double *)p->wr, *swi = (double *)p->wi;
        double *scr = (double *)p->cr, *sci = (double *)p->ci;
        deint8((const double *)(x0 + (size_t)b * L), sxr, sxi, L);
        deint8((const double *)(c  + (size_t)b * L), scr, sci, L);
        for (int s = 0; s < m; ++s) {
            if (!fs_one(p, sxr, sxi, syr, syi))
                dofft_s(p, sxr, sxi, syr, syi, swr, swi);
            map_apply(syr, syi, scr, sci, sxr, sxi, L);
        }
        inter8((double *)(final_out + (size_t)b * L), sxr, sxi, L);
    }
}

void fft1d_destroy(fft1d_plan *p)
{
    if (!p) return;
    free(p->tw); free(p->twv); free(p->im); free(p->om); free(p->scratch_base);
    free(p);
}

#else  /* ======================= kernel pass: parameterized on V ================== */

/* natural-order 4-point DFT, separate in/out strides (in-place safe: reads first) */
static inline void KN(fft4)(const V *xr, const V *xi, long xs, V *yr, V *yi, long ys)
{
    V a0r = xr[0],    a0i = xi[0];
    V a1r = xr[xs],   a1i = xi[xs];
    V a2r = xr[2*xs], a2i = xi[2*xs];
    V a3r = xr[3*xs], a3i = xi[3*xs];
    V t0r = a0r + a2r, t0i = a0i + a2i;
    V t1r = a0r - a2r, t1i = a0i - a2i;
    V t2r = a1r + a3r, t2i = a1i + a3i;
    V t3r = a1r - a3r, t3i = a1i - a3i;
    yr[0]    = t0r + t2r; yi[0]    = t0i + t2i;
    yr[2*ys] = t0r - t2r; yi[2*ys] = t0i - t2i;
    yr[ys]   = t1r + t3i; yi[ys]   = t1i - t3r;   /* X1 = t1 - i t3 */
    yr[3*ys] = t1r - t3i; yi[3*ys] = t1i + t3r;   /* X3 = t1 + i t3 */
}

/* natural-order 8-point DFT (4 real mults by sqrt2/2), separate in/out strides */
static inline void KN(fft8)(const V *xr, const V *xi, long xs, V *yr, V *yi, long ys)
{
    V a0r = xr[0],    a0i = xi[0],    a1r = xr[xs],   a1i = xi[xs];
    V a2r = xr[2*xs], a2i = xi[2*xs], a3r = xr[3*xs], a3i = xi[3*xs];
    V a4r = xr[4*xs], a4i = xi[4*xs], a5r = xr[5*xs], a5i = xi[5*xs];
    V a6r = xr[6*xs], a6i = xi[6*xs], a7r = xr[7*xs], a7i = xi[7*xs];
    V t0r = a0r + a4r, t0i = a0i + a4i, t1r = a0r - a4r, t1i = a0i - a4i;
    V t2r = a2r + a6r, t2i = a2i + a6i, t3r = a2r - a6r, t3i = a2i - a6i;
    V t4r = a1r + a5r, t4i = a1i + a5i, t5r = a1r - a5r, t5i = a1i - a5i;
    V t6r = a3r + a7r, t6i = a3i + a7i, t7r = a3r - a7r, t7i = a3i - a7i;
    V E0r = t0r + t2r, E0i = t0i + t2i, E2r = t0r - t2r, E2i = t0i - t2i;
    V E1r = t1r + t3i, E1i = t1i - t3r, E3r = t1r - t3i, E3i = t1i + t3r;
    V O0r = t4r + t6r, O0i = t4i + t6i, O2r = t4r - t6r, O2i = t4i - t6i;
    V O1r = t5r + t7i, O1i = t5i - t7r, O3r = t5r - t7i, O3i = t5i + t7r;
    V o1r = S2H * (O1r + O1i), o1i = S2H * (O1i - O1r);          /* W8^1 * O1 */
    V o2r = O2i,               o2i = -O2r;                        /* W8^2 * O2 */
    V o3r = S2H * (O3i - O3r), o3i = -(S2H * (O3r + O3i));        /* W8^3 * O3 */
    yr[0]    = E0r + O0r; yi[0]    = E0i + O0i;
    yr[4*ys] = E0r - O0r; yi[4*ys] = E0i - O0i;
    yr[ys]   = E1r + o1r; yi[ys]   = E1i + o1i;
    yr[5*ys] = E1r - o1r; yi[5*ys] = E1i - o1i;
    yr[2*ys] = E2r + o2r; yi[2*ys] = E2i + o2i;
    yr[6*ys] = E2r - o2r; yi[6*ys] = E2i - o2i;
    yr[3*ys] = E3r + o3r; yi[3*ys] = E3i + o3i;
    yr[7*ys] = E3r - o3r; yi[7*ys] = E3i - o3i;
}

/* natural-order 3-point DFT, single stride, in-place safe */
static inline void KN(fft3)(V *zr, V *zi, long s)
{
    V a0r = zr[0], a0i = zi[0], a1r = zr[s], a1i = zi[s], a2r = zr[2*s], a2i = zi[2*s];
    V tr = a1r + a2r, ti = a1i + a2i;
    V mr = a0r - 0.5 * tr, mi = a0i - 0.5 * ti;
    V dr = S3H * (a1r - a2r), di = S3H * (a1i - a2i);
    zr[0]   = a0r + tr;  zi[0]   = a0i + ti;
    zr[s]   = mr + di;   zi[s]   = mi - dr;    /* X1 = m - i*s3*d */
    zr[2*s] = mr - di;   zi[2*s] = mi + dr;    /* X2 = m + i*s3*d */
}

/* 3-point DFT reading through an index map (PFA input permutation folded into the
 * first stage: no separate 60-element copy pass), writing contiguously at stride s */
static inline void KN(fft3io)(const V *xr, const V *xi, const int *ix,
                              V *zr, V *zi, long s)
{
    V a0r = xr[ix[0]],  a0i = xi[ix[0]];
    V a1r = xr[ix[20]], a1i = xi[ix[20]];
    V a2r = xr[ix[40]], a2i = xi[ix[40]];
    V tr = a1r + a2r, ti = a1i + a2i;
    V mr = a0r - 0.5 * tr, mi = a0i - 0.5 * ti;
    V dr = S3H * (a1r - a2r), di = S3H * (a1i - a2i);
    zr[0]   = a0r + tr;  zi[0]   = a0i + ti;
    zr[s]   = mr + di;   zi[s]   = mi - dr;
    zr[2*s] = mr - di;   zi[2*s] = mi + dr;
}

/* natural-order 5-point DFT, single stride, in-place safe */
static inline void KN(fft5)(V *zr, V *zi, long s)
{
    V a0r = zr[0],   a0i = zi[0];
    V a1r = zr[s],   a1i = zi[s],   a2r = zr[2*s], a2i = zi[2*s];
    V a3r = zr[3*s], a3i = zi[3*s], a4r = zr[4*s], a4i = zi[4*s];
    V t1r = a1r + a4r, t1i = a1i + a4i, t3r = a1r - a4r, t3i = a1i - a4i;
    V t2r = a2r + a3r, t2i = a2i + a3i, t4r = a2r - a3r, t4i = a2i - a3i;
    V m1r = a0r + C5_1 * t1r + C5_2 * t2r, m1i = a0i + C5_1 * t1i + C5_2 * t2i;
    V m2r = a0r + C5_2 * t1r + C5_1 * t2r, m2i = a0i + C5_2 * t1i + C5_1 * t2i;
    V n1r = S5_1 * t3r + S5_2 * t4r, n1i = S5_1 * t3i + S5_2 * t4i;
    V n2r = S5_2 * t3r - S5_1 * t4r, n2i = S5_2 * t3i - S5_1 * t4i;
    zr[0]   = a0r + t1r + t2r; zi[0]   = a0i + t1i + t2i;
    zr[s]   = m1r + n1i;       zi[s]   = m1i - n1r;    /* X1 = m1 - i n1 */
    zr[4*s] = m1r - n1i;       zi[4*s] = m1i + n1r;
    zr[2*s] = m2r + n2i;       zi[2*s] = m2i - n2r;    /* X2 = m2 - i n2 */
    zr[3*s] = m2r - n2i;       zi[3*s] = m2i + n2r;
}

/* 5-point DFT reading contiguously, writing through an index map (PFA output
 * permutation folded into the last stage: no separate 60-element copy pass) */
static inline void KN(fft5o)(const V *zr, const V *zi, const int *o, V *yr, V *yi)
{
    V a0r = zr[0], a0i = zi[0];
    V a1r = zr[1], a1i = zi[1], a2r = zr[2], a2i = zi[2];
    V a3r = zr[3], a3i = zi[3], a4r = zr[4], a4i = zi[4];
    V t1r = a1r + a4r, t1i = a1i + a4i, t3r = a1r - a4r, t3i = a1i - a4i;
    V t2r = a2r + a3r, t2i = a2i + a3i, t4r = a2r - a3r, t4i = a2i - a3i;
    V m1r = a0r + C5_1 * t1r + C5_2 * t2r, m1i = a0i + C5_1 * t1i + C5_2 * t2i;
    V m2r = a0r + C5_2 * t1r + C5_1 * t2r, m2i = a0i + C5_2 * t1i + C5_1 * t2i;
    V n1r = S5_1 * t3r + S5_2 * t4r, n1i = S5_1 * t3i + S5_2 * t4i;
    V n2r = S5_2 * t3r - S5_1 * t4r, n2i = S5_2 * t3i - S5_1 * t4i;
    yr[o[0]] = a0r + t1r + t2r; yi[o[0]] = a0i + t1i + t2i;
    yr[o[1]] = m1r + n1i;       yi[o[1]] = m1i - n1r;
    yr[o[4]] = m1r - n1i;       yi[o[4]] = m1i + n1r;
    yr[o[2]] = m2r + n2i;       yi[o[2]] = m2i - n2r;
    yr[o[3]] = m2r - n2i;       yi[o[3]] = m2i + n2r;
}

/* natural-order 16-point DFT = 4 x 4 Cooley-Tukey; tw = 18 doubles, W16^{n1 k2}
 * (cos, -sin) for n1,k2 in 1..3 at [2*((n1-1)*3 + (k2-1))] */
static inline void KN(fft16)(const double *tw, const V *xr, const V *xi, long xs,
                             V *yr, V *yi, long ys)
{
    V wr[16], wi[16];
    for (int n1 = 0; n1 < 4; ++n1)
        KN(fft4)(xr + xs*n1, xi + xs*n1, 4*xs, wr + 4*n1, wi + 4*n1, 1);
    for (int n1 = 1; n1 < 4; ++n1)
        for (int k2 = 1; k2 < 4; ++k2) {
            double c = tw[2*((n1-1)*3 + (k2-1))], s = tw[2*((n1-1)*3 + (k2-1)) + 1];
            V a = wr[4*n1 + k2], b = wi[4*n1 + k2];
            wr[4*n1 + k2] = c * a - s * b;
            wi[4*n1 + k2] = c * b + s * a;
        }
    for (int k2 = 0; k2 < 4; ++k2)
        KN(fft4)(wr + k2, wi + k2, 4, yr + k2*ys, yi + k2*ys, 4*ys);
}

/* dense symmetric-pair prime DFT (P = 13 or 31, H = (P-1)/2):
 *   u_j = x_j + x_{P-j},  v_j = x_j - x_{P-j}
 *   A_k = sum_j cos(2pi jk/P) u_j,  B_k = sum_j sin(2pi jk/P) v_j
 *   X_k = x0 + A_k - i B_k,  X_{P-k} = x0 + A_k + i B_k */
static void KN(fft_densesym)(int P, int H, const double *C, const double *S,
                             const V *xr, const V *xi, V *yr, V *yi)
{
    V ur[15], ui[15], vr[15], vi[15];
    V s0r = xr[0], s0i = xi[0];
    V ar0 = s0r, ai0 = s0i;
    for (int j = 1; j <= H; ++j) {
        ur[j-1] = xr[j] + xr[P-j]; ui[j-1] = xi[j] + xi[P-j];
        vr[j-1] = xr[j] - xr[P-j]; vi[j-1] = xi[j] - xi[P-j];
        ar0 += ur[j-1]; ai0 += ui[j-1];
    }
    yr[0] = ar0; yi[0] = ai0;
    /* k blocked by 3 (d1_prime's trick): each u/v row load feeds 12 FMAs instead of 4,
     * so the loop is FMA-bound, not load-port-bound. H is 6 or 15 -- always 3 | H. */
    for (int k = 1; k + 2 <= H + 1; k += 3) {
        V a0r = s0r, a0i = s0i, b0r = VZ, b0i = VZ;
        V a1r = s0r, a1i = s0i, b1r = VZ, b1i = VZ;
        V a2r = s0r, a2i = s0i, b2r = VZ, b2i = VZ;
        const double *Cr = C + (k-1), *Sr = S + (k-1);
        for (int j = 0; j < H; ++j) {
            V u_r = ur[j], u_i = ui[j], v_r = vr[j], v_i = vi[j];
            double c0 = Cr[j*H], c1 = Cr[j*H + 1], c2 = Cr[j*H + 2];
            double s0 = Sr[j*H], s1 = Sr[j*H + 1], s2 = Sr[j*H + 2];
            a0r += c0 * u_r; a0i += c0 * u_i; b0r += s0 * v_r; b0i += s0 * v_i;
            a1r += c1 * u_r; a1i += c1 * u_i; b1r += s1 * v_r; b1i += s1 * v_i;
            a2r += c2 * u_r; a2i += c2 * u_i; b2r += s2 * v_r; b2i += s2 * v_i;
        }
        yr[k]     = a0r + b0i; yi[k]     = a0i - b0r;
        yr[P-k]   = a0r - b0i; yi[P-k]   = a0i + b0r;
        yr[k+1]   = a1r + b1i; yi[k+1]   = a1i - b1r;
        yr[P-k-1] = a1r - b1i; yi[P-k-1] = a1i + b1r;
        yr[k+2]   = a2r + b2i; yi[k+2]   = a2i - b2r;
        yr[P-k-2] = a2r - b2i; yi[P-k-2] = a2i + b2r;
    }
}

/* 32 = 8 (inner, over n2) x 4 (outer, over n1); n = 4 n2 + n1; X[k2 + 8 k1] */
static void KN(fft32)(const double *tw, const V *xr, const V *xi,
                      V *yr, V *yi, V *wr, V *wi)
{
    for (int n1 = 0; n1 < 4; ++n1)
        KN(fft8)(xr + n1, xi + n1, 4, wr + 8*n1, wi + 8*n1, 1);
    for (int n1 = 1; n1 < 4; ++n1)
        for (int k2 = 1; k2 < 8; ++k2) {
            double c = tw[2*(n1*8 + k2)], s = tw[2*(n1*8 + k2) + 1];
            V a = wr[n1*8 + k2], b = wi[n1*8 + k2];
            wr[n1*8 + k2] = c * a - s * b;
            wi[n1*8 + k2] = c * b + s * a;
        }
    for (int k2 = 0; k2 < 8; ++k2)
        KN(fft4)(wr + k2, wi + k2, 8, yr + k2, yi + k2, 8);
}

/* 64 = 8 x 8; input stride xs so 128 can call it on even/odd decimations */
static void KN(fft64s)(const double *tw, const V *xr, const V *xi, long xs,
                       V *yr, V *yi, V *wr, V *wi)
{
    for (int n1 = 0; n1 < 8; ++n1)
        KN(fft8)(xr + xs*n1, xi + xs*n1, 8*xs, wr + 8*n1, wi + 8*n1, 1);
    for (int n1 = 1; n1 < 8; ++n1)
        for (int k2 = 1; k2 < 8; ++k2) {
            double c = tw[2*(n1*8 + k2)], s = tw[2*(n1*8 + k2) + 1];
            V a = wr[n1*8 + k2], b = wi[n1*8 + k2];
            wr[n1*8 + k2] = c * a - s * b;
            wi[n1*8 + k2] = c * b + s * a;
        }
    for (int k2 = 0; k2 < 8; ++k2)
        KN(fft8)(wr + k2, wi + k2, 8, yr + k2, yi + k2, 8);
}

/* 128 = 2 x 64: Y0 = FFT64(evens), Y1 = FFT64(odds); X[k] = Y0 +/- W128^k Y1 */
static void KN(fft128)(const double *tw, const V *xr, const V *xi,
                       V *yr, V *yi, V *wr, V *wi)
{
    KN(fft64s)(tw, xr,     xi,     2, yr,      yi,      wr, wi);
    KN(fft64s)(tw, xr + 1, xi + 1, 2, yr + 64, yi + 64, wr, wi);
    const double *t2 = tw + 128;
    for (int k = 0; k < 64; ++k) {
        double c = t2[2*k], s = t2[2*k + 1];
        V br = yr[64 + k], bi = yi[64 + k];
        V tr = c * br - s * bi, ti = c * bi + s * br;
        V ar = yr[k], ai = yi[k];
        yr[k]      = ar + tr; yi[k]      = ai + ti;
        yr[64 + k] = ar - tr; yi[64 + k] = ai - ti;
    }
}

/* 60 = Good-Thomas 3 x 4 x 5, twiddle-free; w layout [n3][n4][n5] = n3*20+n4*5+n5 */
static void KN(fft60)(const int *im, const int *om, const V *xr, const V *xi,
                      V *yr, V *yi, V *wr, V *wi)
{
    for (int j = 0; j < 20; ++j)
        KN(fft3io)(xr, xi, im + j, wr + j, wi + j, 20);
    for (int n3 = 0; n3 < 3; ++n3)
        for (int n5 = 0; n5 < 5; ++n5)
            KN(fft4)(wr + n3*20 + n5, wi + n3*20 + n5, 5,
                     wr + n3*20 + n5, wi + n3*20 + n5, 5);
    for (int t = 0; t < 12; ++t)
        KN(fft5o)(wr + 5*t, wi + 5*t, om + 5*t, yr, yi);
}

static void KN(dofft)(const fft1d_plan *p, const V *xr, const V *xi,
                      V *yr, V *yi, V *wr, V *wi)
{
    switch (p->L) {
    case 13:  KN(fft_densesym)(13, 6,  p->tw, p->tw + 36,  xr, xi, yr, yi); break;
    case 31:  KN(fft_densesym)(31, 15, p->tw, p->tw + 225, xr, xi, yr, yi); break;
    case 32:  KN(fft32)(p->tw, xr, xi, yr, yi, wr, wi); break;
    case 60:  KN(fft60)(p->im, p->om, xr, xi, yr, yi, wr, wi); break;
    case 64:  KN(fft64s)(p->tw, xr, xi, 1, yr, yi, wr, wi); break;
    case 128: KN(fft128)(p->tw, xr, xi, yr, yi, wr, wi); break;
    }
}

#endif /* BL_COMMON */
