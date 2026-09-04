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
 * B=1 chains are REGISTER-RESIDENT (r3): at 13/31 the state lives across steps in
 * fold-ready A/B rows (d1_prime's design), at 32/64/128 in the four-step kernels'
 * natural row layout, which those kernels map onto itself -- no per-step scratch round
 * trip, map fused on rows. Batched m=1 at 31/64/128 loops the fused-AoS single-shot
 * kernels instead of the SoA group path (one memory pass, ~no port-5 transposes).
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
    double *twc;                /* chain-layout densesym tables (L=13/31, k-lane rows
                                   with a k=0 column and a trailing x0 ones-row) */
    double *tp;                 /* interleaved-pair tables for the 13/31 m=1 kernels
                                   (d1_prime r3): coefficients duplicated per 128-bit
                                   pair, sin stored (+s,-s) */
    double *tw32;               /* L=32 in-register codelet table (d1_pow2): per
                                   4-p group [w1r|w1p|w2r|w2p|w3r|w3p] x 8 doubles */
    double *tw64a, *tw64b;      /* L=64 in-register codelet tables (d1_pow2, r5):
                                   s=1 dup-format table (4 groups x 48 dbl) and the
                                   radix-4 n=16 bc table (9 dbl per p, p=0..3) */
    v8 *xr, *xi, *yr, *yi, *wr, *wi, *cr, *ci;   /* SoA scratch, each L vectors */
    void *scratch_base;
};

/* PFA-60 CRT maps as compile-time constants (d1_composite's r2 lesson: heap-pointer
 * index tables keep gcc's loops rolled with movslq address arithmetic; static const
 * tables + forced unroll constant-fold every index into a displacement).
 * IM60[n3*20+n4*5+n5] = (20n3+15n4+12n5)%60, OM60 = (40n3+45n4+36n5)%60. */
static const int IM60[60] = {
     0,12,24,36,48, 15,27,39,51, 3, 30,42,54, 6,18, 45,57, 9,21,33,
    20,32,44,56, 8, 35,47,59,11,23, 50, 2,14,26,38,  5,17,29,41,53,
    40,52, 4,16,28, 55, 7,19,31,43, 10,22,34,46,58, 25,37,49, 1,13};
static const int OM60[60] = {
     0,36,12,48,24, 45,21,57,33, 9, 30, 6,42,18,54, 15,51,27, 3,39,
    40,16,52,28, 4, 25, 1,37,13,49, 10,46,22,58,34, 55,31, 7,43,19,
    20,56,32, 8,44,  5,41,17,53,29, 50,26, 2,38,14, 35,11,47,23,59};

#define S2H  0.70710678118654752440   /* sqrt(2)/2 */
#define S3H  0.86602540378443864676   /* sin(2pi/3) */
#define C5_1 0.30901699437494742410   /* cos(2pi/5) */
#define C5_2 (-0.80901699437494742410) /* cos(4pi/5) */
#define S5_1 0.95105651629515357212   /* sin(2pi/5) */
#define S5_2 0.58778525229247312917   /* sin(4pi/5) */

/* ---- L=60 m=1 kernels TAKEN FROM d1_composite (r4 source, ported verbatim):
 * their CRT tables and Winograd constants. idx = n1*15 + n2*5 + n3;
 * P60[idx] = (15n1+20n2+12n3) mod 60 (stage-A reads), K60[idx] = (45n1+40n2+36n3)
 * mod 60 (stage-C writes). Constants as in their file. */
static const int P60[60] = {0,12,24,36,48,20,32,44,56,8,40,52,4,16,28,15,27,39,51,3,35,47,59,11,23,55,7,19,31,43,30,42,54,6,18,50,2,14,26,38,10,22,34,46,58,45,57,9,21,33,5,17,29,41,53,25,37,49,1,13};
static const int K60[60] = {0,36,12,48,24,40,16,52,28,4,20,56,32,8,44,45,21,57,33,9,25,1,37,13,49,5,41,17,53,29,30,6,42,18,54,10,46,22,58,34,50,26,2,38,14,15,51,27,3,39,55,31,7,43,19,35,11,47,23,59};
#define C51 (0.30901699437494745126)   /* cos(2pi/5) */
#define C52 (-0.80901699437494734024)  /* cos(4pi/5) */
#define S51 (0.95105651629515353118)   /* sin(2pi/5) */
#define S52 (0.58778525229247324813)   /* sin(4pi/5) */
#define S3W (0.86602540378443870761)   /* sin(2pi/3) */

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

/* ===== L=60 m=1 kernels, TAKEN FROM d1_composite (r4 fft60_ymm1/fft60_zmm2x2,
 * ported verbatim with renamed tables). These replace fs60_aos (B=1: node 0.0687
 * vs their 0.0489 on the r4 board) and the SoA group execute at 60 (B=512: node
 * 0.0699 vs their 0.0578). Twiddle-free Good-Thomas PFA, straight-line via
 * _Pragma unroll; stage A is broadcast-fed signed FMA (zero cross-lane shuffles),
 * stage C leaves via pure stores. */

/* single transform: one complex per 128-bit lane, PAIRED over n1 (2/ymm).
 * Stage A: X_j = (xj|xj) via vbroadcastf64x2 (one load uop); P = X0 + E1*X2,
 * Q = X1 + E1*X3 (signed FMA, exact); R = permil(Q,6); wp0/1 = P +- R*E4. */
__attribute__((aligned(64), hot)) static void fft60_ymm1(const double *restrict x, double *restrict y)
{
    __m256d wp[2][15];   /* [n1-pair][n2*5+n3] */
    {
        const __m256d E1 = _mm256_set_pd(-1.0, -1.0, 1.0, 1.0);
        const __m256d E4 = _mm256_set_pd(-1.0, 1.0, 1.0, 1.0);
        _Pragma("GCC unroll 15") for (int col = 0; col < 15; ++col) {
            __m256d X0 = _mm256_broadcast_f64x2(_mm_loadu_pd(x + 2*P60[col]));
            __m256d X1 = _mm256_broadcast_f64x2(_mm_loadu_pd(x + 2*P60[15+col]));
            __m256d X2 = _mm256_broadcast_f64x2(_mm_loadu_pd(x + 2*P60[30+col]));
            __m256d X3 = _mm256_broadcast_f64x2(_mm_loadu_pd(x + 2*P60[45+col]));
            __m256d P = _mm256_fmadd_pd(X2, E1, X0);
            __m256d Q = _mm256_fmadd_pd(X3, E1, X1);
            __m256d R = _mm256_permute_pd(Q, 0x6);
            wp[0][col] = _mm256_fmadd_pd(R, E4, P);
            wp[1][col] = _mm256_fnmadd_pd(R, E4, P);
        }
    }
    {
        const __m256d half = _mm256_set1_pd(0.5);
        const __m256d S3E = _mm256_set_pd(-S3W, S3W, -S3W, S3W);
        _Pragma("GCC unroll 2") for (int pr = 0; pr < 2; ++pr)
            _Pragma("GCC unroll 5") for (int n3 = 0; n3 < 5; ++n3) {
                __m256d x1 = wp[pr][5+n3], x2 = wp[pr][10+n3];
                __m256d t = _mm256_add_pd(x1, x2), u = _mm256_sub_pd(x1, x2);
                __m256d x0 = wp[pr][n3];
                wp[pr][n3] = _mm256_add_pd(x0, t);
                __m256d vv = _mm256_fnmadd_pd(half, t, x0);
                __m256d swu = _mm256_permute_pd(u, 0x5);
                wp[pr][5+n3]  = _mm256_fmadd_pd(swu, S3E, vv);
                wp[pr][10+n3] = _mm256_fnmadd_pd(swu, S3E, vv);
            }
        const __m256d c51v = _mm256_set1_pd(C51), c52v = _mm256_set1_pd(C52);
        const __m256d S1E = _mm256_set_pd(-S51, S51, -S51, S51);
        const __m256d S2E = _mm256_set_pd(-S52, S52, -S52, S52);
        _Pragma("GCC unroll 2") for (int pr = 0; pr < 2; ++pr)
            _Pragma("GCC unroll 3") for (int n2 = 0; n2 < 3; ++n2) {
                const int b = 5*n2;                        /* base in 15-space */
                const int bl = (2*pr)*15 + 5*n2, bh = (2*pr+1)*15 + 5*n2; /* 60-space */
                __m256d x0 = wp[pr][b];
                __m256d t1 = _mm256_add_pd(wp[pr][b+1], wp[pr][b+4]);
                __m256d t3 = _mm256_sub_pd(wp[pr][b+1], wp[pr][b+4]);
                __m256d t2 = _mm256_add_pd(wp[pr][b+2], wp[pr][b+3]);
                __m256d t4 = _mm256_sub_pd(wp[pr][b+2], wp[pr][b+3]);
#define STP(k3, v) do { __m256d v_ = (v);                                        \
        _mm_storeu_pd(y + 2*K60[bl+(k3)], _mm256_castpd256_pd128(v_));           \
        _mm_storeu_pd(y + 2*K60[bh+(k3)], _mm256_extractf128_pd(v_, 1)); } while (0)
                STP(0, _mm256_add_pd(x0, _mm256_add_pd(t1, t2)));
                __m256d a1 = _mm256_fmadd_pd(c52v, t2, _mm256_fmadd_pd(c51v, t1, x0));
                __m256d a2 = _mm256_fmadd_pd(c51v, t2, _mm256_fmadd_pd(c52v, t1, x0));
                __m256d sw3 = _mm256_permute_pd(t3, 0x5), sw4 = _mm256_permute_pd(t4, 0x5);
                __m256d m1 = _mm256_fmadd_pd(sw4, S2E, _mm256_mul_pd(sw3, S1E));
                __m256d m2 = _mm256_fnmadd_pd(sw4, S1E, _mm256_mul_pd(sw3, S2E));
                STP(1, _mm256_add_pd(a1, m1));
                STP(4, _mm256_sub_pd(a1, m1));
                STP(2, _mm256_add_pd(a2, m2));
                STP(3, _mm256_sub_pd(a2, m2));
#undef STP
            }
    }
}

/* batched: TWO transforms x TWO n1-pairs per zmm (transform pair stride 120
 * doubles). Stage A operands arrive DUAL-BROADCAST (masked vbroadcastf64x2 =
 * one load uop for the second transform's lanes); stages B/C run at half the
 * per-transform op count of a 2-wide kernel; stage C exits via memory-form
 * vextractf64x2 = pure stores. ~287 instructions/transform, ~18 shuffle uops
 * (d1_composite r4 disassembly; won d1_race's on-node kernel race in r3). */
__attribute__((aligned(64), hot, unused)) static void fft60_zmm2x2(const double *restrict x, double *restrict y)
{
    __m512d wp[2][15];
#define LDD(p) _mm512_mask_broadcast_f64x2(                                    \
        _mm512_broadcast_f64x2(_mm_loadu_pd(x + 2*(p))), 0xF0,                 \
        _mm_loadu_pd(x + 120 + 2*(p)))
    {
        const __m512d E1 = _mm512_set4_pd(-1.0, -1.0, 1.0, 1.0);
        const __m512d E4 = _mm512_set4_pd(-1.0, 1.0, 1.0, 1.0);
        _Pragma("GCC unroll 15") for (int col = 0; col < 15; ++col) {
            __m512d X0 = LDD(P60[col]),    X1 = LDD(P60[15+col]);
            __m512d X2 = LDD(P60[30+col]), X3 = LDD(P60[45+col]);
            __m512d P = _mm512_fmadd_pd(X2, E1, X0);
            __m512d Q = _mm512_fmadd_pd(X3, E1, X1);
            __m512d R = _mm512_permute_pd(Q, 0x66);
            wp[0][col] = _mm512_fmadd_pd(R, E4, P);
            wp[1][col] = _mm512_fnmadd_pd(R, E4, P);
        }
    }
    {
        const __m512d half = _mm512_set1_pd(0.5);
        const __m512d S3E = _mm512_set4_pd(-S3W, S3W, -S3W, S3W);
        _Pragma("GCC unroll 2") for (int pr = 0; pr < 2; ++pr)
            _Pragma("GCC unroll 5") for (int n3 = 0; n3 < 5; ++n3) {
                __m512d x1 = wp[pr][5+n3], x2 = wp[pr][10+n3];
                __m512d t = _mm512_add_pd(x1, x2), u = _mm512_sub_pd(x1, x2);
                __m512d x0 = wp[pr][n3];
                wp[pr][n3] = _mm512_add_pd(x0, t);
                __m512d vv = _mm512_fnmadd_pd(half, t, x0);
                __m512d swu = _mm512_permute_pd(u, 0x55);
                wp[pr][5+n3]  = _mm512_fmadd_pd(swu, S3E, vv);
                wp[pr][10+n3] = _mm512_fnmadd_pd(swu, S3E, vv);
            }
        const __m512d c51v = _mm512_set1_pd(C51), c52v = _mm512_set1_pd(C52);
        const __m512d S1E = _mm512_set4_pd(-S51, S51, -S51, S51);
        const __m512d S2E = _mm512_set4_pd(-S52, S52, -S52, S52);
        _Pragma("GCC unroll 2") for (int pr = 0; pr < 2; ++pr)
            _Pragma("GCC unroll 3") for (int n2 = 0; n2 < 3; ++n2) {
                const int b = 5*n2;
                const int bl = (2*pr)*15 + 5*n2, bh = (2*pr+1)*15 + 5*n2;
                __m512d x0 = wp[pr][b];
                __m512d t1 = _mm512_add_pd(wp[pr][b+1], wp[pr][b+4]);
                __m512d t3 = _mm512_sub_pd(wp[pr][b+1], wp[pr][b+4]);
                __m512d t2 = _mm512_add_pd(wp[pr][b+2], wp[pr][b+3]);
                __m512d t4 = _mm512_sub_pd(wp[pr][b+2], wp[pr][b+3]);
#define STPZ(k3, v) do { __m512d v_ = (v);                                        \
        _mm_storeu_pd(y + 2*K60[bl+(k3)], _mm512_castpd512_pd128(v_));            \
        _mm_storeu_pd(y + 2*K60[bh+(k3)], _mm512_extractf64x2_pd(v_, 1));         \
        _mm_storeu_pd(y + 120 + 2*K60[bl+(k3)], _mm512_extractf64x2_pd(v_, 2));   \
        _mm_storeu_pd(y + 120 + 2*K60[bh+(k3)], _mm512_extractf64x2_pd(v_, 3)); } while (0)
                STPZ(0, _mm512_add_pd(x0, _mm512_add_pd(t1, t2)));
                __m512d a1 = _mm512_fmadd_pd(c52v, t2, _mm512_fmadd_pd(c51v, t1, x0));
                __m512d a2 = _mm512_fmadd_pd(c51v, t2, _mm512_fmadd_pd(c52v, t1, x0));
                __m512d sw3 = _mm512_permute_pd(t3, 0x55), sw4 = _mm512_permute_pd(t4, 0x55);
                __m512d m1 = _mm512_fmadd_pd(sw4, S2E, _mm512_mul_pd(sw3, S1E));
                __m512d m2 = _mm512_fnmadd_pd(sw4, S1E, _mm512_mul_pd(sw3, S2E));
                STPZ(1, _mm512_add_pd(a1, m1));
                STPZ(4, _mm512_sub_pd(a1, m1));
                STPZ(2, _mm512_add_pd(a2, m2));
                STPZ(3, _mm512_sub_pd(a2, m2));
#undef STPZ
            }
    }
#undef LDD
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

/* ===== interleaved-pair m=1 kernels for 13/31 (TAKEN FROM d1_prime, r3 record):
 * each 128-bit lane pair carries one complex OUTPUT (4 per zmm), so the natural
 * interleaved layout IS the compute layout -- no deinterleave prologue or
 * epilogue, and ONE in-register vshuff64x2 pair-broadcast per (u_j|v_j) instead
 * of two scalar broadcasts. Coefficients pair-duplicated at plan time (p->tp);
 * sin stored (+s,-s) so S accumulates (Br,-Bi) and one in-lane vpermilpd swap
 * gives (-Bi,Br):  na = P - swap(S) = X[k],  nb = P + swap(S) = X[L-k].
 * These replace r2's dsk13/dsk31/fs13_aos/fs31_aos, whose u/v fold went through
 * stack arrays reloaded as {1to8} broadcasts: an 8B broadcast load from a fresh
 * 64B store does NOT store-forward on ICX (d1_prime's r3 node bisect, 0.021 vs
 * 0.015 at 13:512) even though SPR forwards it -- which is exactly why my 31
 * m=1 cells degraded ~2x from wallaby to the scoring node in r2/r3. */
#define BSTEP13(jj, UW, VW, tt, PA_, PB_, SA_, SB_) do {                      \
    __m512d ub_ = _mm512_shuffle_f64x2(UW, UW, (tt)*0x55);                    \
    __m512d vb_ = _mm512_shuffle_f64x2(VW, VW, (tt)*0x55);                    \
    PA_ = _mm512_fmadd_pd(_mm512_load_pd(tp + 32*(jj) +  0), ub_, PA_);       \
    PB_ = _mm512_fmadd_pd(_mm512_load_pd(tp + 32*(jj) +  8), ub_, PB_);       \
    SA_ = _mm512_fmadd_pd(_mm512_load_pd(tp + 32*(jj) + 16), vb_, SA_);       \
    SB_ = _mm512_fmadd_pd(_mm512_load_pd(tp + 32*(jj) + 24), vb_, SB_);       \
} while (0)

static inline __attribute__((always_inline)) void
bl13p_body(const double *restrict x, double *restrict y,
           const double *restrict tp, const int NSET)
{
    __m512d F1 = _mm512_loadu_pd(x + 2);              /* (x1)(x2)(x3)(x4)   */
    __m512d F2 = _mm512_loadu_pd(x + 10);             /* (x5)(x6)(x7)(x8)   */
    __m512d Z  = _mm512_loadu_pd(x + 18);             /* (x9)(x10)(x11)(x12)*/
    __m512d R1 = _mm512_shuffle_f64x2(Z,  Z,  0x1B);  /* (x12)(x11)(x10)(x9)*/
    __m512d R2 = _mm512_shuffle_f64x2(F2, F2, 0xBB);  /* (x8)(x7)(x8)(x7)   */
    __m512d U1 = _mm512_add_pd(F1, R1), V1 = _mm512_sub_pd(F1, R1);
    __m512d U2 = _mm512_add_pd(F2, R2), V2 = _mm512_sub_pd(F2, R2);
    /* x0 seeds the P accumulators, so no separate pre = x0 + P add at the end */
    __m512d x0p = _mm512_broadcast_f64x2(_mm_loadu_pd(x));
    __m512d PA = x0p, PB = x0p, SA = _mm512_setzero_pd(), SB = SA;
    if (NSET == 2) {   /* two sets halve the FMA depth: the B=1 critical path */
        __m512d PA2 = _mm512_setzero_pd(), PB2 = PA2, SA2 = PA2, SB2 = PA2;
        BSTEP13(0, U1, V1, 0, PA,  PB,  SA,  SB );
        BSTEP13(1, U1, V1, 1, PA2, PB2, SA2, SB2);
        BSTEP13(2, U1, V1, 2, PA,  PB,  SA,  SB );
        BSTEP13(3, U1, V1, 3, PA2, PB2, SA2, SB2);
        BSTEP13(4, U2, V2, 0, PA,  PB,  SA,  SB );
        BSTEP13(5, U2, V2, 1, PA2, PB2, SA2, SB2);
        PA = _mm512_add_pd(PA, PA2); PB = _mm512_add_pd(PB, PB2);
        SA = _mm512_add_pd(SA, SA2); SB = _mm512_add_pd(SB, SB2);
    } else {
        BSTEP13(0, U1, V1, 0, PA, PB, SA, SB);
        BSTEP13(1, U1, V1, 1, PA, PB, SA, SB);
        BSTEP13(2, U1, V1, 2, PA, PB, SA, SB);
        BSTEP13(3, U1, V1, 3, PA, PB, SA, SB);
        BSTEP13(4, U2, V2, 0, PA, PB, SA, SB);
        BSTEP13(5, U2, V2, 1, PA, PB, SA, SB);
    }
    __m512d swA  = _mm512_permute_pd(SA, 0x55), swB = _mm512_permute_pd(SB, 0x55);
    __m512d naA  = _mm512_sub_pd(PA, swA), nbA = _mm512_add_pd(PA, swA);
    __m512d naB  = _mm512_sub_pd(PB, swB), nbB = _mm512_add_pd(PB, swB);
    /* naA = X1..X4, naB = X5,X6,X0,--, nbA = X12..X9, nbB = X8,X7,--,-- */
    const __m512i IDX0 = _mm512_setr_epi64(12,13,0,1,2,3,4,5);      /* X0,X1..X3   */
    const __m512i IDXT = _mm512_setr_epi64(0,1,2,3,10,11,10,11);    /* X5,X6,X7    */
    const __m512i IDX1 = _mm512_setr_epi64(6,7,8,9,10,11,12,13);    /* X4,X5,X6,X7 */
    const __m512i IDX2 = _mm512_setr_epi64(0,1,14,15,12,13,10,11);  /* X8..X11     */
    _mm512_storeu_pd(y,      _mm512_permutex2var_pd(naA, IDX0, naB));
    __m512d t = _mm512_permutex2var_pd(naB, IDXT, nbB);
    _mm512_storeu_pd(y + 8,  _mm512_permutex2var_pd(naA, IDX1, t));
    _mm512_storeu_pd(y + 16, _mm512_permutex2var_pd(nbB, IDX2, nbA));
    _mm512_mask_storeu_pd(y + 24, 0x03, nbA);                       /* X12 */
}

/* two transforms per body: table rows loaded once per pair, all loads grouped
 * ahead of all stores (keeps load->store distance up across the batch loop) */
#define BSTEP13X2(jj, UW, VW, UX, VX, tt) do {                                \
    __m512d c1_ = _mm512_load_pd(tp + 32*(jj) +  0);                          \
    __m512d c2_ = _mm512_load_pd(tp + 32*(jj) +  8);                          \
    __m512d s1_ = _mm512_load_pd(tp + 32*(jj) + 16);                          \
    __m512d s2_ = _mm512_load_pd(tp + 32*(jj) + 24);                          \
    __m512d ub_ = _mm512_shuffle_f64x2(UW, UW, (tt)*0x55);                    \
    __m512d vb_ = _mm512_shuffle_f64x2(VW, VW, (tt)*0x55);                    \
    __m512d uc_ = _mm512_shuffle_f64x2(UX, UX, (tt)*0x55);                    \
    __m512d vc_ = _mm512_shuffle_f64x2(VX, VX, (tt)*0x55);                    \
    PA = _mm512_fmadd_pd(c1_, ub_, PA);  QA = _mm512_fmadd_pd(c1_, uc_, QA);  \
    PB = _mm512_fmadd_pd(c2_, ub_, PB);  QB = _mm512_fmadd_pd(c2_, uc_, QB);  \
    SA = _mm512_fmadd_pd(s1_, vb_, SA);  TA = _mm512_fmadd_pd(s1_, vc_, TA);  \
    SB = _mm512_fmadd_pd(s2_, vb_, SB);  TB = _mm512_fmadd_pd(s2_, vc_, TB);  \
} while (0)

static inline __attribute__((always_inline)) void
bl13p_b2(const double *restrict x, double *restrict y, const double *restrict tp)
{
    const double *restrict x2 = x + 26;
    double *restrict y2 = y + 26;
    __m512d F1 = _mm512_loadu_pd(x + 2),  G1 = _mm512_loadu_pd(x2 + 2);
    __m512d F2 = _mm512_loadu_pd(x + 10), G2 = _mm512_loadu_pd(x2 + 10);
    __m512d Z  = _mm512_loadu_pd(x + 18), W  = _mm512_loadu_pd(x2 + 18);
    __m512d x0p = _mm512_broadcast_f64x2(_mm_loadu_pd(x));
    __m512d x0q = _mm512_broadcast_f64x2(_mm_loadu_pd(x2));
    __m512d R1 = _mm512_shuffle_f64x2(Z,  Z,  0x1B), S1r = _mm512_shuffle_f64x2(W,  W,  0x1B);
    __m512d R2 = _mm512_shuffle_f64x2(F2, F2, 0xBB), S2r = _mm512_shuffle_f64x2(G2, G2, 0xBB);
    __m512d U1 = _mm512_add_pd(F1, R1),  V1 = _mm512_sub_pd(F1, R1);
    __m512d U2 = _mm512_add_pd(F2, R2),  V2 = _mm512_sub_pd(F2, R2);
    __m512d X1 = _mm512_add_pd(G1, S1r), W1 = _mm512_sub_pd(G1, S1r);
    __m512d X2 = _mm512_add_pd(G2, S2r), W2 = _mm512_sub_pd(G2, S2r);
    __m512d PA = x0p, PB = x0p, SA = _mm512_setzero_pd(), SB = SA;
    __m512d QA = x0q, QB = x0q, TA = SA, TB = SA;
    BSTEP13X2(0, U1, V1, X1, W1, 0);
    BSTEP13X2(1, U1, V1, X1, W1, 1);
    BSTEP13X2(2, U1, V1, X1, W1, 2);
    BSTEP13X2(3, U1, V1, X1, W1, 3);
    BSTEP13X2(4, U2, V2, X2, W2, 0);
    BSTEP13X2(5, U2, V2, X2, W2, 1);
    const __m512i IDX0 = _mm512_setr_epi64(12,13,0,1,2,3,4,5);
    const __m512i IDXT = _mm512_setr_epi64(0,1,2,3,10,11,10,11);
    const __m512i IDX1 = _mm512_setr_epi64(6,7,8,9,10,11,12,13);
    const __m512i IDX2 = _mm512_setr_epi64(0,1,14,15,12,13,10,11);
    __m512d swA = _mm512_permute_pd(SA, 0x55), swB = _mm512_permute_pd(SB, 0x55);
    __m512d naA = _mm512_sub_pd(PA, swA), nbA = _mm512_add_pd(PA, swA);
    __m512d naB = _mm512_sub_pd(PB, swB), nbB = _mm512_add_pd(PB, swB);
    __m512d swC = _mm512_permute_pd(TA, 0x55), swD = _mm512_permute_pd(TB, 0x55);
    __m512d naC = _mm512_sub_pd(QA, swC), nbC = _mm512_add_pd(QA, swC);
    __m512d naD = _mm512_sub_pd(QB, swD), nbD = _mm512_add_pd(QB, swD);
    _mm512_storeu_pd(y,       _mm512_permutex2var_pd(naA, IDX0, naB));
    __m512d t  = _mm512_permutex2var_pd(naB, IDXT, nbB);
    _mm512_storeu_pd(y + 8,   _mm512_permutex2var_pd(naA, IDX1, t));
    _mm512_storeu_pd(y + 16,  _mm512_permutex2var_pd(nbB, IDX2, nbA));
    _mm512_mask_storeu_pd(y + 24, 0x03, nbA);
    _mm512_storeu_pd(y2,      _mm512_permutex2var_pd(naC, IDX0, naD));
    __m512d t2 = _mm512_permutex2var_pd(naD, IDXT, nbD);
    _mm512_storeu_pd(y2 + 8,  _mm512_permutex2var_pd(naC, IDX1, t2));
    _mm512_storeu_pd(y2 + 16, _mm512_permutex2var_pd(nbD, IDX2, nbC));
    _mm512_mask_storeu_pd(y2 + 24, 0x03, nbC);
}

/* interleaved-pair L=31 kernel, same scheme. Rows: A=(X1..X4) B=(X5..X8)
 * C=(X9..X12) D=(X13,X14,X15,X0); conjugate rows nbA=(X30..X27) nbB=(X26..X23)
 * nbC=(X22..X19) nbD=(X18,X17,X16,--). na rows store back DIRECTLY (natural
 * order, already interleaved); nb rows need one 0x1B pair-reversal each. */
#define BSTEP31(jj, UW, VW, tt) do {                                          \
    __m512d ub_ = _mm512_shuffle_f64x2(UW, UW, (tt)*0x55);                    \
    __m512d vb_ = _mm512_shuffle_f64x2(VW, VW, (tt)*0x55);                    \
    PA = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) +  0), ub_, PA);         \
    PB = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) +  8), ub_, PB);         \
    PC = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 16), ub_, PC);         \
    PD = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 24), ub_, PD);         \
    SA = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 32), vb_, SA);         \
    SB = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 40), vb_, SB);         \
    SC = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 48), vb_, SC);         \
    SD = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 56), vb_, SD);         \
} while (0)

static inline __attribute__((always_inline)) void
bl31p_body(const double *restrict x, double *restrict y, const double *restrict tq)
{
    __m512d F1 = _mm512_loadu_pd(x + 2);    /* (x1)(x2)(x3)(x4)     */
    __m512d F2 = _mm512_loadu_pd(x + 10);   /* (x5)(x6)(x7)(x8)     */
    __m512d F3 = _mm512_loadu_pd(x + 18);   /* (x9)(x10)(x11)(x12)  */
    __m512d F4 = _mm512_loadu_pd(x + 26);   /* (x13)(x14)(x15)(x16) */
    __m512d Z1 = _mm512_loadu_pd(x + 54);   /* (x27)(x28)(x29)(x30) */
    __m512d Z2 = _mm512_loadu_pd(x + 46);   /* (x23)(x24)(x25)(x26) */
    __m512d Z3 = _mm512_loadu_pd(x + 38);   /* (x19)(x20)(x21)(x22) */
    __m512d Z4 = _mm512_loadu_pd(x + 30);   /* (x15)(x16)(x17)(x18) */
    __m512d R1 = _mm512_shuffle_f64x2(Z1, Z1, 0x1B);   /* (x30)(x29)(x28)(x27) */
    __m512d R2 = _mm512_shuffle_f64x2(Z2, Z2, 0x1B);   /* (x26)(x25)(x24)(x23) */
    __m512d R3 = _mm512_shuffle_f64x2(Z3, Z3, 0x1B);   /* (x22)(x21)(x20)(x19) */
    __m512d R4 = _mm512_shuffle_f64x2(Z4, Z4, 0x1B);   /* (x18)(x17)(x16)(x15) */
    __m512d U1 = _mm512_add_pd(F1, R1), V1 = _mm512_sub_pd(F1, R1);
    __m512d U2 = _mm512_add_pd(F2, R2), V2 = _mm512_sub_pd(F2, R2);
    __m512d U3 = _mm512_add_pd(F3, R3), V3 = _mm512_sub_pd(F3, R3);
    __m512d U4 = _mm512_add_pd(F4, R4), V4 = _mm512_sub_pd(F4, R4);  /* pair 3 junk */
    __m512d x0p = _mm512_broadcast_f64x2(_mm_loadu_pd(x));
    __m512d PA = x0p, PB = x0p, PC = x0p, PD = x0p;
    __m512d SA = _mm512_setzero_pd(), SB = SA, SC = SA, SD = SA;
    BSTEP31( 0, U1, V1, 0);  BSTEP31( 1, U1, V1, 1);
    BSTEP31( 2, U1, V1, 2);  BSTEP31( 3, U1, V1, 3);
    BSTEP31( 4, U2, V2, 0);  BSTEP31( 5, U2, V2, 1);
    BSTEP31( 6, U2, V2, 2);  BSTEP31( 7, U2, V2, 3);
    BSTEP31( 8, U3, V3, 0);  BSTEP31( 9, U3, V3, 1);
    BSTEP31(10, U3, V3, 2);  BSTEP31(11, U3, V3, 3);
    BSTEP31(12, U4, V4, 0);  BSTEP31(13, U4, V4, 1);
    BSTEP31(14, U4, V4, 2);
    __m512d swA = _mm512_permute_pd(SA, 0x55), swB = _mm512_permute_pd(SB, 0x55);
    __m512d swC = _mm512_permute_pd(SC, 0x55), swD = _mm512_permute_pd(SD, 0x55);
    __m512d naA = _mm512_sub_pd(PA, swA), nbA = _mm512_add_pd(PA, swA);
    __m512d naB = _mm512_sub_pd(PB, swB), nbB = _mm512_add_pd(PB, swB);
    __m512d naC = _mm512_sub_pd(PC, swC), nbC = _mm512_add_pd(PC, swC);
    __m512d naD = _mm512_sub_pd(PD, swD), nbD = _mm512_add_pd(PD, swD);
    _mm_storeu_pd(y, _mm512_extractf64x2_pd(naD, 3));            /* X0 */
    _mm512_storeu_pd(y + 2,  naA);                               /* X1..X4   */
    _mm512_storeu_pd(y + 10, naB);                               /* X5..X8   */
    _mm512_storeu_pd(y + 18, naC);                               /* X9..X12  */
    _mm512_mask_storeu_pd(y + 26, 0x3F, naD);                    /* X13..X15 */
    _mm512_mask_storeu_pd(y + 30, 0xFC,
        _mm512_shuffle_f64x2(nbD, nbD, 0x1B));                   /* X16..X18 */
    _mm512_storeu_pd(y + 38, _mm512_shuffle_f64x2(nbC, nbC, 0x1B)); /* X19..X22 */
    _mm512_storeu_pd(y + 46, _mm512_shuffle_f64x2(nbB, nbB, 0x1B)); /* X23..X26 */
    _mm512_storeu_pd(y + 54, _mm512_shuffle_f64x2(nbA, nbA, 0x1B)); /* X27..X30 */
}

/* ===== L=32 all-in-register interleaved-AoS codelet (TAKEN FROM d1_pow2):
 * whole transform in 8 zmm of 4 complexes: a stride-1 radix-4 stage (two quads
 * with full precomputed w/w^2/w^3 dup-format tables and an in-register 4x4
 * complex transpose) feeding one twiddle-free radix-8, natural-order out.
 * Replaces the ymm four-step fs32/fs32_aos whose boundary de/re-interleave +
 * transpose shuffles lost 2.06x to MKL on ICX (pow2's identical structure ran
 * 0.0154 vs my 0.0315 on the r3 board at 32:512). */
static inline __m512d bl_cmul(__m512d u, __m512d wr, __m512d wp)
{   /* u * w with wr = dup(re w), wp = (+im-conj form): (s,-s) per pair */
    return _mm512_fmadd_pd(_mm512_permute_pd(u, 0x55), wp, _mm512_mul_pd(u, wr));
}

#define BLR8_CONSTS                                                                    \
    const __m512d ONE = _mm512_set1_pd(1.0);                                           \
    const __m512d Cq = _mm512_set1_pd(0.70710678118654752440);                         \
    const __m512d CPN = _mm512_setr_pd(0.70710678118654752440, -0.70710678118654752440,\
                                       0.70710678118654752440, -0.70710678118654752440,\
                                       0.70710678118654752440, -0.70710678118654752440,\
                                       0.70710678118654752440, -0.70710678118654752440)

/* radix-8 DIF butterfly on 8 vectors (32 complexes), no twiddles */
#define BLR8_BODY(X0, X1, X2, X3, X4, X5, X6, X7)                                      \
    __m512d s0 = _mm512_add_pd(X0, X4), s1 = _mm512_add_pd(X1, X5);                    \
    __m512d s2 = _mm512_add_pd(X2, X6), s3 = _mm512_add_pd(X3, X7);                    \
    __m512d d0 = _mm512_sub_pd(X0, X4), d1 = _mm512_sub_pd(X1, X5);                    \
    __m512d d2 = _mm512_sub_pd(X2, X6), d3 = _mm512_sub_pd(X3, X7);                    \
    __m512d apc = _mm512_add_pd(s0, s2), amc = _mm512_sub_pd(s0, s2);                  \
    __m512d bpd = _mm512_add_pd(s1, s3), bmd = _mm512_sub_pd(s1, s3);                  \
    __m512d swe = _mm512_permute_pd(bmd, 0x55);                                        \
    __m512d u0 = _mm512_add_pd(apc, bpd);                                              \
    __m512d u4 = _mm512_sub_pd(apc, bpd);                                              \
    __m512d u2 = _mm512_fmsubadd_pd(amc, ONE, swe);                                    \
    __m512d u6 = _mm512_fmaddsub_pd(amc, ONE, swe);                                    \
    __m512d e1 = _mm512_mul_pd(_mm512_fmsubadd_pd(d1, ONE, _mm512_permute_pd(d1, 0x55)), Cq); \
    __m512d e3 = _mm512_mul_pd(                                                        \
        _mm512_permute_pd(_mm512_fmsubadd_pd(d3, ONE, _mm512_permute_pd(d3, 0x55)), 0x55), CPN); \
    __m512d sw2 = _mm512_permute_pd(d2, 0x55);                                         \
    __m512d apo = _mm512_fmsubadd_pd(d0, ONE, sw2); /* d0 + (-i)d2 */                  \
    __m512d amo = _mm512_fmaddsub_pd(d0, ONE, sw2); /* d0 - (-i)d2 */                  \
    __m512d bpo = _mm512_add_pd(e1, e3), bmo = _mm512_sub_pd(e1, e3);                  \
    __m512d swo = _mm512_permute_pd(bmo, 0x55);                                        \
    __m512d u1 = _mm512_add_pd(apo, bpo);                                              \
    __m512d u5 = _mm512_sub_pd(apo, bpo);                                              \
    __m512d u3 = _mm512_fmsubadd_pd(amo, ONE, swo);                                    \
    __m512d u7 = _mm512_fmaddsub_pd(amo, ONE, swo)

#define BLTRANSP4(R0, R1, R2, R3, O0, O1, O2, O3)                                      \
    do {                                                                               \
        __m512d tp0_ = _mm512_permutex2var_pd(R0, idxA, R1);                           \
        __m512d tp1_ = _mm512_permutex2var_pd(R0, idxB, R1);                           \
        __m512d tp2_ = _mm512_permutex2var_pd(R2, idxA, R3);                           \
        __m512d tp3_ = _mm512_permutex2var_pd(R2, idxB, R3);                           \
        O0 = _mm512_shuffle_f64x2(tp0_, tp2_, 0x44);                                   \
        O1 = _mm512_shuffle_f64x2(tp1_, tp3_, 0x44);                                   \
        O2 = _mm512_shuffle_f64x2(tp0_, tp2_, 0xEE);                                   \
        O3 = _mm512_shuffle_f64x2(tp1_, tp3_, 0xEE);                                   \
    } while (0)

/* stride-1 radix-4 butterfly quad, all three twiddles from the full table */
#define BLS1QUADT(A, B, C, D, TP, O0, O1, O2, O3)                                      \
    do {                                                                               \
        __m512d apc_ = _mm512_add_pd(A, C), amc_ = _mm512_sub_pd(A, C);                \
        __m512d bpd_ = _mm512_add_pd(B, D), bmd_ = _mm512_sub_pd(B, D);                \
        __m512d sw_ = _mm512_permute_pd(bmd_, 0x55);                                   \
        __m512d q0_ = _mm512_add_pd(apc_, bpd_);                                       \
        __m512d q2_ = _mm512_sub_pd(apc_, bpd_);                                       \
        __m512d q1_ = _mm512_fmsubadd_pd(amc_, ONE, sw_);                              \
        __m512d q3_ = _mm512_fmaddsub_pd(amc_, ONE, sw_);                              \
        __m512d r1_ = bl_cmul(q1_, _mm512_load_pd(TP), _mm512_load_pd((TP) + 8));      \
        __m512d r2_ = bl_cmul(q2_, _mm512_load_pd((TP) + 16), _mm512_load_pd((TP) + 24)); \
        __m512d r3_ = bl_cmul(q3_, _mm512_load_pd((TP) + 32), _mm512_load_pd((TP) + 40)); \
        BLTRANSP4(q0_, r1_, r2_, r3_, O0, O1, O2, O3);                                 \
    } while (0)

static void fft32_codelet(const double *restrict tf, const double *restrict in,
                          double *restrict out, int batch)
{
    BLR8_CONSTS;
    const __m512i idxA = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    const __m512i idxB = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
    for (int b = 0; b < batch; ++b) {
        const double *x = in + 64L * b;
        double *y = out + 64L * b;
        __m512d v0 = _mm512_loadu_pd(x),      v1 = _mm512_loadu_pd(x + 8);
        __m512d v2 = _mm512_loadu_pd(x + 16), v3 = _mm512_loadu_pd(x + 24);
        __m512d v4 = _mm512_loadu_pd(x + 32), v5 = _mm512_loadu_pd(x + 40);
        __m512d v6 = _mm512_loadu_pd(x + 48), v7 = _mm512_loadu_pd(x + 56);
        __m512d ya0, ya1, ya2, ya3, yb0, yb1, yb2, yb3;
        BLS1QUADT(v0, v2, v4, v6, tf,      ya0, ya1, ya2, ya3);
        BLS1QUADT(v1, v3, v5, v7, tf + 48, yb0, yb1, yb2, yb3);
        BLR8_BODY(ya0, ya1, ya2, ya3, yb0, yb1, yb2, yb3);
        _mm512_storeu_pd(y,      u0); _mm512_storeu_pd(y + 8,  u1);
        _mm512_storeu_pd(y + 16, u2); _mm512_storeu_pd(y + 24, u3);
        _mm512_storeu_pd(y + 32, u4); _mm512_storeu_pd(y + 40, u5);
        _mm512_storeu_pd(y + 48, u6); _mm512_storeu_pd(y + 56, u7);
    }
}

/* ===== L=64 all-in-register interleaved-AoS codelet (TAKEN FROM d1_pow2's
 * fft64_execute, r5 port): whole transform in 16 zmm of 4 complexes. Stage 1 =
 * stride-1 radix-4 (four S1QUADT quads, full dup-format tables, in-register
 * transpose); stage 2 = radix-4 n=16 s=4 with broadcast twiddles (p'=0
 * twiddle-free); stage 3 = twiddle-free radix-4, natural order out. ~65 shuffle
 * uops/transform vs fs64_aos's ~128 boundary+transpose shuffles -- pow2 measured
 * 0.037 us at 64:512 on a80n0 where my fs64_aos loop read 0.053-0.066. */
#define BLR4Q(A, B, C, D, O0, O1, O2, O3)                                              \
    do {                                                                               \
        __m512d apc_ = _mm512_add_pd(A, C), amc_ = _mm512_sub_pd(A, C);                \
        __m512d bpd_ = _mm512_add_pd(B, D), bmd_ = _mm512_sub_pd(B, D);                \
        __m512d sw_ = _mm512_permute_pd(bmd_, 0x55);                                   \
        O0 = _mm512_add_pd(apc_, bpd_);                                                \
        O2 = _mm512_sub_pd(apc_, bpd_);                                                \
        O1 = _mm512_fmsubadd_pd(amc_, ONE, sw_);                                       \
        O3 = _mm512_fmaddsub_pd(amc_, ONE, sw_);                                       \
    } while (0)

static void fft64_codelet(const double *restrict tf, const double *restrict t2,
                          const double *restrict in, double *restrict out, int batch)
{
    const __m512d ONE = _mm512_set1_pd(1.0);
    const __m512i idxA = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    const __m512i idxB = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
    __m512d g2r[4][3], g2p[4][3];
    for (int pp = 1; pp < 4; ++pp)
        for (int r = 0; r < 3; ++r) {
            g2r[pp][r] = _mm512_set1_pd(t2[9 * pp + r]);
            g2p[pp][r] = _mm512_broadcast_f64x2(_mm_loadu_pd(t2 + 9 * pp + 3 + 2 * r));
        }
    for (int b = 0; b < batch; ++b) {
        const double *x = in + 128L * b;
        double *yo = out + 128L * b;
        __m512d Y[16], Z[16];
        /* stage 1: p-quads (v[j], v[j+4], v[j+8], v[j+12]), twiddle group j */
        for (int j = 0; j < 4; ++j) {
            __m512d a  = _mm512_loadu_pd(x + 8L * j);
            __m512d bq = _mm512_loadu_pd(x + 8L * (j + 4));
            __m512d cq = _mm512_loadu_pd(x + 8L * (j + 8));
            __m512d dq = _mm512_loadu_pd(x + 8L * (j + 12));
            BLS1QUADT(a, bq, cq, dq, tf + 48 * j, Y[4 * j], Y[4 * j + 1], Y[4 * j + 2],
                      Y[4 * j + 3]);
        }
        /* stage 2: radix-4 n=16 s=4, broadcast twiddles; p'=0 twiddle-free */
        BLR4Q(Y[0], Y[4], Y[8], Y[12], Z[0], Z[1], Z[2], Z[3]);
        for (int pp = 1; pp < 4; ++pp) {
            __m512d z0, z1, z2, z3;
            BLR4Q(Y[pp], Y[pp + 4], Y[pp + 8], Y[pp + 12], z0, z1, z2, z3);
            Z[4 * pp] = z0;
            Z[4 * pp + 1] = bl_cmul(z1, g2r[pp][0], g2p[pp][0]);
            Z[4 * pp + 2] = bl_cmul(z2, g2r[pp][1], g2p[pp][1]);
            Z[4 * pp + 3] = bl_cmul(z3, g2r[pp][2], g2p[pp][2]);
        }
        /* stage 3: twiddle-free radix-4, s=16 */
        for (int j = 0; j < 4; ++j) {
            __m512d z0, z1, z2, z3;
            BLR4Q(Z[j], Z[j + 4], Z[j + 8], Z[j + 12], z0, z1, z2, z3);
            _mm512_storeu_pd(yo + 8L * j, z0);
            _mm512_storeu_pd(yo + 8L * (j + 4), z1);
            _mm512_storeu_pd(yo + 8L * (j + 8), z2);
            _mm512_storeu_pd(yo + 8L * (j + 12), z3);
        }
    }
}

/* ===== L=128 all-in-register codelet (r5, extending the d1_pow2 64-codelet
 * structure one factor up): stage 1 = stride-1 radix-4 (eight S1QUADT quads,
 * W128^{p t} dup tables), stage 2 = radix-4 n=32 s=4 with bc-table twiddles
 * (W32^{m t}), stage 3 = twiddle-free radix-8 (BLR8_BODY), natural order out.
 * 32 data zmm so gcc spills Y[]/Z[] partially, but the boundary de/interleave
 * and the 4x tr8 of fs128_aos (~192 cross-lane p5 uops) become ~156 mostly
 * in-lane shuffles and the fft16 sub-kernel dispatch disappears. */
static void fft128_codelet(const double *restrict tf, const double *restrict t2,
                           const double *restrict in, double *restrict out, int batch)
{
    BLR8_CONSTS;
    const __m512i idxA = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    const __m512i idxB = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
    for (int b = 0; b < batch; ++b) {
        const double *x = in + 256L * b;
        double *yo = out + 256L * b;
        __m512d Y[32], Z[32];
        /* stage 1: quads (n, n+32, n+64, n+96), twiddle group j covers p=4j..4j+3 */
        for (int j = 0; j < 8; ++j) {
            __m512d a  = _mm512_loadu_pd(x + 8L * j);
            __m512d bq = _mm512_loadu_pd(x + 8L * (j + 8));
            __m512d cq = _mm512_loadu_pd(x + 8L * (j + 16));
            __m512d dq = _mm512_loadu_pd(x + 8L * (j + 24));
            BLS1QUADT(a, bq, cq, dq, tf + 48 * j, Y[4 * j], Y[4 * j + 1], Y[4 * j + 2],
                      Y[4 * j + 3]);
        }
        /* stage 2: radix-4 across vector index (n=32), twiddles W32^{m t};
         * outputs grouped by branch t so stage 3 reads contiguously */
        {
            __m512d z0, z1, z2, z3;
            BLR4Q(Y[0], Y[8], Y[16], Y[24], z0, z1, z2, z3);
            Z[0] = z0; Z[8] = z1; Z[16] = z2; Z[24] = z3;
        }
        for (int m = 1; m < 8; ++m) {
            __m512d z0, z1, z2, z3;
            BLR4Q(Y[m], Y[m + 8], Y[m + 16], Y[m + 24], z0, z1, z2, z3);
            Z[m] = z0;
            Z[m + 8]  = bl_cmul(z1, _mm512_set1_pd(t2[9*m]),
                                _mm512_broadcast_f64x2(_mm_loadu_pd(t2 + 9*m + 3)));
            Z[m + 16] = bl_cmul(z2, _mm512_set1_pd(t2[9*m + 1]),
                                _mm512_broadcast_f64x2(_mm_loadu_pd(t2 + 9*m + 5)));
            Z[m + 24] = bl_cmul(z3, _mm512_set1_pd(t2[9*m + 2]),
                                _mm512_broadcast_f64x2(_mm_loadu_pd(t2 + 9*m + 7)));
        }
        /* stage 3: twiddle-free radix-8 over m for each branch t2;
         * output k of branch t2 is X32-index 4k+t2 -> complexes 4*(4k+t2) */
        for (int t2 = 0; t2 < 4; ++t2) {
            BLR8_BODY(Z[8*t2], Z[8*t2 + 1], Z[8*t2 + 2], Z[8*t2 + 3],
                      Z[8*t2 + 4], Z[8*t2 + 5], Z[8*t2 + 6], Z[8*t2 + 7]);
            _mm512_storeu_pd(yo + 8L * t2,        u0);
            _mm512_storeu_pd(yo + 8L * (4 + t2),  u1);
            _mm512_storeu_pd(yo + 8L * (8 + t2),  u2);
            _mm512_storeu_pd(yo + 8L * (12 + t2), u3);
            _mm512_storeu_pd(yo + 8L * (16 + t2), u4);
            _mm512_storeu_pd(yo + 8L * (20 + t2), u5);
            _mm512_storeu_pd(yo + 8L * (24 + t2), u6);
            _mm512_storeu_pd(yo + 8L * (28 + t2), u7);
        }
    }
}

/* dispatch: 1 if a fast single-transform kernel exists for this L */
static int fs_one(const fft1d_plan *p, const double *re, const double *im,
                  double *qr, double *qi)
{
    switch (p->L) {
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
    if (jb < L) {   /* gather tail: 2 gathers per remaining column instead of a full
                       24-shuffle tr8 block (at L=13 that block extracted ONE column;
                       every one of those shuffles is a port-5 uop on the scoring
                       node). Column t of the 8 transforms is stride 2L doubles. */
        __m512i vi = _mm512_mullo_epi64(_mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm512_set1_epi64(2 * L));
        for (int t = jb; t < L; ++t) {
            xr[t] = (v8)_mm512_i64gather_pd(vi, base + 2*t, 8);
            xi[t] = (v8)_mm512_i64gather_pd(vi, base + 2*t + 1, 8);
        }
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
    if (jb < L) {   /* scatter tail, mirror of the gather tail in tload8 */
        __m512i vi = _mm512_mullo_epi64(_mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7),
                                        _mm512_set1_epi64(2 * L));
        for (int t = jb; t < L; ++t) {
            _mm512_i64scatter_pd(base + 2*t,     vi, (__m512d)yr[t], 8);
            _mm512_i64scatter_pd(base + 2*t + 1, vi, (__m512d)yi[t], 8);
        }
    }
}

/* ============ the chain map, on contiguous split-complex arrays ====================
 * Explicit 512-bit intrinsics: gcc -march=native auto-vectorizes this loop at ymm width
 * only (256-bit preference), which halves sqrt/div throughput -- and the map is the
 * per-step floor of every chained cell. All pointers are 64B-aligned scratch.
 *
 * 1/(1+sqrt(h)) is computed WITHOUT the unpipelined divider (borrowed from
 * d1_composite / d1_prime): rsqrt14/rcp14 + Newton, all FMA-port work that
 * pipelines across the L map vectors. The floor must NOT be 1e-300: rsqrt14(1e-300)
 * ~1e150 drives the Newton arithmetic into FP assists, ~250 cycles PER map_scale on
 * any vector with a zeroed junk lane (measured 85 vs 4 ns standalone) -- this was
 * r2's undiagnosed "Newton in the masked tail is catastrophically slow". At 1e-100
 * the substitution error for real h < 1e-100 is |z|*1e-50, nothing at a 1e-10 gate.
 *
 * r6: latency-shaped map TAKEN FROM d1_prime (r5 map_scale_h31, offered to the
 * panel): the map is on the SERIAL per-step critical path of every chain, so
 * shape the dependence graph, not the op count.
 *   1. sqrt via GOLDSCHMIDT (x = m2*y, h = y/2; twice r = 0.5 - x*h, x += x*r,
 *      h += h*r): each iteration is fnmadd->fma (8 cy) instead of NR's
 *      t=r*r -> fnmadd -> mul (12 cy). Not self-correcting like NR, but 2
 *      iterations from the 2^-14 seed land ~2-3 ulp -- nothing at 1e-10 gates.
 *   2. the reciprocal seed q0 = rcp14(1 + m2*y) is taken from the RAW rsqrt14
 *      estimate (available ~20 cy early) and refined by 2 Newton rounds against
 *      the TRUE d = 1 + sqrt: reciprocal NR converges to 1/d regardless of the
 *      seed (1.2e-4 -> 1.4e-8 -> ~1e-16), so the rcp chain overlaps the
 *      Goldschmidt refinement instead of serializing behind it.
 *   3. the 1e-100 junk-lane floor (my r3 clamp lesson, prime's r5 additive
 *      form) is folded into the caller's m2 FMA -- see BL_M2 -- so the max
 *      leaves the critical path. Perturbation <= 1e-100/m2, invisible.
 * The argument is therefore the PRE-FLOORED m2; callers build it with BL_M2. */
#define BL_M2(zr_, zi_) _mm512_fmadd_pd(zr_, zr_, \
        _mm512_fmadd_pd(zi_, zi_, _mm512_set1_pd(1e-100)))
static inline __m512d map_scale_fast(__m512d m2)
{
    const __m512d half = _mm512_set1_pd(0.5), one = _mm512_set1_pd(1.0);
    const __m512d two  = _mm512_set1_pd(2.0);
    __m512d y  = _mm512_rsqrt14_pd(m2);
    __m512d x  = _mm512_mul_pd(m2, y);
    __m512d h  = _mm512_mul_pd(y, half);
    __m512d q  = _mm512_rcp14_pd(_mm512_add_pd(x, one));    /* early seed */
    __m512d r1 = _mm512_fnmadd_pd(x, h, half);
    x = _mm512_fmadd_pd(x, r1, x);  h = _mm512_fmadd_pd(h, r1, h);
    __m512d r2 = _mm512_fnmadd_pd(x, h, half);
    x = _mm512_fmadd_pd(x, r2, x);                          /* x = sqrt(m2) */
    __m512d d = _mm512_add_pd(x, one);
    q = _mm512_mul_pd(q, _mm512_fnmadd_pd(d, q, two));
    q = _mm512_mul_pd(q, _mm512_fnmadd_pd(d, q, two));
    return q;
}

static void map_apply(const double *restrict zr, const double *restrict zi,
                      const double *restrict cr, const double *restrict ci,
                      double *restrict sr, double *restrict si, int n)
{
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d re = _mm512_add_pd(_mm512_load_pd(zr + i), _mm512_load_pd(cr + i));
        __m512d im = _mm512_add_pd(_mm512_load_pd(zi + i), _mm512_load_pd(ci + i));
        __m512d h  = BL_M2(re, im);
        __m512d sc = map_scale_fast(h);    /* r4: 2NR-only map here too; pow2's r3
                                              re-measurement (their change 3) showed
                                              the residual refinements were noise
                                              once twiddles are long-double, and the
                                              gates keep >= 2 decades (verified at
                                              every graded chained cell) */
        _mm512_store_pd(sr + i, _mm512_mul_pd(re, sc));
        _mm512_store_pd(si + i, _mm512_mul_pd(im, sc));
    }
    if (i < n) {   /* masked tail. Newton would be SAFE here now (the 1e-100 clamp
                      removed the junk-lane FP assists that r2 blamed) but exact
                      sqrt/div measured ~4% faster on the 60 B=1 chain -- one
                      latency-bound vector, and vsqrt+vdiv latency < the Newton chain. */
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

/* ===== fused FFT+map steps for the batched (SoA group) chain =====
 * r4: the generic group step was dofft_v8 into y then a separate map_apply pass
 * y+c -> x -- a full extra read+write of the state per step. These per-size
 * steps apply the map AT the final stage's output writes instead (same
 * arithmetic, same map_scale_fast), so the y round trip disappears. The idea is
 * d1_pow2's "map fused into the final butterfly stage" (their r1 design),
 * applied to my SoA group layout. */

/* map one v8 output row (z = y + c), writing the state row */
static inline void map_row(__m512d zr, __m512d zi, const v8 *crow, const v8 *cirow,
                           v8 *xrow, v8 *xirow)
{
    zr = _mm512_add_pd(zr, (__m512d)*crow);
    zi = _mm512_add_pd(zi, (__m512d)*cirow);
    __m512d h  = BL_M2(zr, zi);
    __m512d sc = map_scale_fast(h);
    *xrow  = (v8)_mm512_mul_pd(zr, sc);
    *xirow = (v8)_mm512_mul_pd(zi, sc);
}

/* densesym (13/31) with the map fused into every output write; state in
 * xr/xi, in place (the fold reads all of xr before the first output lands) */
static void chain_dsym_grp_step(fft1d_plan *p, int P, int H,
                                const double *C, const double *S)
{
    v8 *xr = p->xr, *xi = p->xi, *cr = p->cr, *ci = p->ci;
    v8 ur[15], ui[15], vr[15], vi[15];
    v8 s0r = xr[0], s0i = xi[0];
    v8 ar0 = s0r, ai0 = s0i;
    for (int j = 1; j <= H; ++j) {
        ur[j-1] = xr[j] + xr[P-j]; ui[j-1] = xi[j] + xi[P-j];
        vr[j-1] = xr[j] - xr[P-j]; vi[j-1] = xi[j] - xi[P-j];
        ar0 += ur[j-1]; ai0 += ui[j-1];
    }
    map_row((__m512d)ar0, (__m512d)ai0, cr, ci, xr, xi);
    for (int k = 1; k + 2 <= H + 1; k += 3) {
        v8 a0r = s0r, a0i = s0i, b0r = (v8){0}, b0i = (v8){0};
        v8 a1r = s0r, a1i = s0i, b1r = (v8){0}, b1i = (v8){0};
        v8 a2r = s0r, a2i = s0i, b2r = (v8){0}, b2i = (v8){0};
        const double *Cr = C + (k-1), *Sr = S + (k-1);
        for (int j = 0; j < H; ++j) {
            v8 u_r = ur[j], u_i = ui[j], v_r = vr[j], v_i = vi[j];
            double c0 = Cr[j*H], c1 = Cr[j*H + 1], c2 = Cr[j*H + 2];
            double s0 = Sr[j*H], s1 = Sr[j*H + 1], s2 = Sr[j*H + 2];
            a0r += c0 * u_r; a0i += c0 * u_i; b0r += s0 * v_r; b0i += s0 * v_i;
            a1r += c1 * u_r; a1i += c1 * u_i; b1r += s1 * v_r; b1i += s1 * v_i;
            a2r += c2 * u_r; a2i += c2 * u_i; b2r += s2 * v_r; b2i += s2 * v_i;
        }
        map_row((__m512d)(a0r + b0i), (__m512d)(a0i - b0r), cr + k,     ci + k,     xr + k,     xi + k);
        map_row((__m512d)(a0r - b0i), (__m512d)(a0i + b0r), cr + P-k,   ci + P-k,   xr + P-k,   xi + P-k);
        map_row((__m512d)(a1r + b1i), (__m512d)(a1i - b1r), cr + k+1,   ci + k+1,   xr + k+1,   xi + k+1);
        map_row((__m512d)(a1r - b1i), (__m512d)(a1i + b1r), cr + P-k-1, ci + P-k-1, xr + P-k-1, xi + P-k-1);
        map_row((__m512d)(a2r + b2i), (__m512d)(a2i - b2r), cr + k+2,   ci + k+2,   xr + k+2,   xi + k+2);
        map_row((__m512d)(a2r - b2i), (__m512d)(a2i + b2r), cr + P-k-2, ci + P-k-2, xr + P-k-2, xi + P-k-2);
    }
}

/* fft5o with the map fused into its five permuted output writes */
static inline void fft5om(const v8 *zr, const v8 *zi, const int *o,
                          const v8 *cr, const v8 *ci, v8 *xr, v8 *xi)
{
    v8 a0r = zr[0], a0i = zi[0];
    v8 a1r = zr[1], a1i = zi[1], a2r = zr[2], a2i = zi[2];
    v8 a3r = zr[3], a3i = zi[3], a4r = zr[4], a4i = zi[4];
    v8 t1r = a1r + a4r, t1i = a1i + a4i, t3r = a1r - a4r, t3i = a1i - a4i;
    v8 t2r = a2r + a3r, t2i = a2i + a3i, t4r = a2r - a3r, t4i = a2i - a3i;
    v8 m1r = a0r + C5_1 * t1r + C5_2 * t2r, m1i = a0i + C5_1 * t1i + C5_2 * t2i;
    v8 m2r = a0r + C5_2 * t1r + C5_1 * t2r, m2i = a0i + C5_2 * t1i + C5_1 * t2i;
    v8 n1r = S5_1 * t3r + S5_2 * t4r, n1i = S5_1 * t3i + S5_2 * t4i;
    v8 n2r = S5_2 * t3r - S5_1 * t4r, n2i = S5_2 * t3i - S5_1 * t4i;
    map_row((__m512d)(a0r + t1r + t2r), (__m512d)(a0i + t1i + t2i),
            cr + o[0], ci + o[0], xr + o[0], xi + o[0]);
    map_row((__m512d)(m1r + n1i), (__m512d)(m1i - n1r),
            cr + o[1], ci + o[1], xr + o[1], xi + o[1]);
    map_row((__m512d)(m1r - n1i), (__m512d)(m1i + n1r),
            cr + o[4], ci + o[4], xr + o[4], xi + o[4]);
    map_row((__m512d)(m2r + n2i), (__m512d)(m2i - n2r),
            cr + o[2], ci + o[2], xr + o[2], xi + o[2]);
    map_row((__m512d)(m2r - n2i), (__m512d)(m2i + n2r),
            cr + o[3], ci + o[3], xr + o[3], xi + o[3]);
}

/* PFA-60 step with the map fused into stage C (stage A reads all of xr first) */
static void chain60_grp_step(fft1d_plan *p)
{
    v8 *xr = p->xr, *xi = p->xi, *wr = p->wr, *wi = p->wi;
    _Pragma("GCC unroll 20")
    for (int j = 0; j < 20; ++j)
        fft3io_v8(xr, xi, IM60 + j, wr + j, wi + j, 20);
    _Pragma("GCC unroll 3")
    for (int n3 = 0; n3 < 3; ++n3)
        _Pragma("GCC unroll 5")
        for (int n5 = 0; n5 < 5; ++n5)
            fft4_v8(wr + n3*20 + n5, wi + n3*20 + n5, 5,
                    wr + n3*20 + n5, wi + n3*20 + n5, 5);
    _Pragma("GCC unroll 12")
    for (int t = 0; t < 12; ++t)
        fft5om(wr + 5*t, wi + 5*t, OM60 + 5*t, p->cr, p->ci, xr, xi);
}

/* fft8 (v8 rows) with the map fused into its eight output writes */
static void fft8m_v8(const v8 *xr, const v8 *xi, long xs,
                     const v8 *cr, const v8 *ci, long cs, v8 *yr, v8 *yi, long ys)
{
    v8 a0r = xr[0],    a0i = xi[0],    a1r = xr[xs],   a1i = xi[xs];
    v8 a2r = xr[2*xs], a2i = xi[2*xs], a3r = xr[3*xs], a3i = xi[3*xs];
    v8 a4r = xr[4*xs], a4i = xi[4*xs], a5r = xr[5*xs], a5i = xi[5*xs];
    v8 a6r = xr[6*xs], a6i = xi[6*xs], a7r = xr[7*xs], a7i = xi[7*xs];
    v8 t0r = a0r + a4r, t0i = a0i + a4i, t1r = a0r - a4r, t1i = a0i - a4i;
    v8 t2r = a2r + a6r, t2i = a2i + a6i, t3r = a2r - a6r, t3i = a2i - a6i;
    v8 t4r = a1r + a5r, t4i = a1i + a5i, t5r = a1r - a5r, t5i = a1i - a5i;
    v8 t6r = a3r + a7r, t6i = a3i + a7i, t7r = a3r - a7r, t7i = a3i - a7i;
    v8 E0r = t0r + t2r, E0i = t0i + t2i, E2r = t0r - t2r, E2i = t0i - t2i;
    v8 E1r = t1r + t3i, E1i = t1i - t3r, E3r = t1r - t3i, E3i = t1i + t3r;
    v8 O0r = t4r + t6r, O0i = t4i + t6i, O2r = t4r - t6r, O2i = t4i - t6i;
    v8 O1r = t5r + t7i, O1i = t5i - t7r, O3r = t5r - t7i, O3i = t5i + t7r;
    v8 o1r = S2H * (O1r + O1i), o1i = S2H * (O1i - O1r);
    v8 o2r = O2i,               o2i = -O2r;
    v8 o3r = S2H * (O3i - O3r), o3i = -(S2H * (O3r + O3i));
    map_row((__m512d)(E0r + O0r), (__m512d)(E0i + O0i), cr,        ci,        yr,        yi);
    map_row((__m512d)(E0r - O0r), (__m512d)(E0i - O0i), cr + 4*cs, ci + 4*cs, yr + 4*ys, yi + 4*ys);
    map_row((__m512d)(E1r + o1r), (__m512d)(E1i + o1i), cr + cs,   ci + cs,   yr + ys,   yi + ys);
    map_row((__m512d)(E1r - o1r), (__m512d)(E1i - o1i), cr + 5*cs, ci + 5*cs, yr + 5*ys, yi + 5*ys);
    map_row((__m512d)(E2r + o2r), (__m512d)(E2i + o2i), cr + 2*cs, ci + 2*cs, yr + 2*ys, yi + 2*ys);
    map_row((__m512d)(E2r - o2r), (__m512d)(E2i - o2i), cr + 6*cs, ci + 6*cs, yr + 6*ys, yi + 6*ys);
    map_row((__m512d)(E3r + o3r), (__m512d)(E3i + o3i), cr + 3*cs, ci + 3*cs, yr + 3*ys, yi + 3*ys);
    map_row((__m512d)(E3r - o3r), (__m512d)(E3i - o3i), cr + 7*cs, ci + 7*cs, yr + 7*ys, yi + 7*ys);
}

/* L=64 step: outer fft8 columns write MAPPED state directly (stage 1 reads all
 * of xr before the first mapped output lands back in it) */
static void chain64_grp_step(fft1d_plan *p)
{
    v8 *xr = p->xr, *xi = p->xi, *wr = p->wr, *wi = p->wi;
    const double *tw = p->tw;
    for (int n1 = 0; n1 < 8; ++n1)
        fft8_v8(xr + n1, xi + n1, 8, wr + 8*n1, wi + 8*n1, 1);
    for (int n1 = 1; n1 < 8; ++n1)
        for (int k2 = 1; k2 < 8; ++k2) {
            double c = tw[2*(n1*8 + k2)], s = tw[2*(n1*8 + k2) + 1];
            v8 a = wr[n1*8 + k2], b = wi[n1*8 + k2];
            wr[n1*8 + k2] = c * a - s * b;
            wi[n1*8 + k2] = c * b + s * a;
        }
    for (int k2 = 0; k2 < 8; ++k2)
        fft8m_v8(wr + k2, wi + k2, 8,
                 p->cr + k2, p->ci + k2, 8, xr + k2, xi + k2, 8);
}

/* L=128 step: the final radix-2 combine and the map in ONE pass over y
 * (previously combine read+wrote all 128 y rows, then map re-read them) */
static void chain128_grp_step(fft1d_plan *p)
{
    const double *t2 = p->tw + 128;
    v8 *yr = p->yr, *yi = p->yi;
    fft64s_v8(p->tw, p->xr,     p->xi,     2, yr,      yi,      p->wr, p->wi);
    fft64s_v8(p->tw, p->xr + 1, p->xi + 1, 2, yr + 64, yi + 64, p->wr, p->wi);
    for (int k = 0; k < 64; ++k) {
        __m512d c  = _mm512_set1_pd(t2[2*k]), sn = _mm512_set1_pd(t2[2*k + 1]);
        __m512d br = (__m512d)yr[64 + k],     bi = (__m512d)yi[64 + k];
        __m512d tr = _mm512_fnmadd_pd(sn, bi, _mm512_mul_pd(c, br));
        __m512d ti = _mm512_fmadd_pd(sn, br, _mm512_mul_pd(c, bi));
        __m512d ar = (__m512d)yr[k],          ai = (__m512d)yi[k];
        map_row(_mm512_add_pd(ar, tr), _mm512_add_pd(ai, ti),
                p->cr + k,      p->ci + k,      p->xr + k,      p->xi + k);
        map_row(_mm512_sub_pd(ar, tr), _mm512_sub_pd(ai, ti),
                p->cr + 64 + k, p->ci + 64 + k, p->xr + 64 + k, p->xi + 64 + k);
    }
}

/* ===== in-register B=1 chains for the dense primes (borrowed from d1_prime's
 * chain1_body design, r1 record): the state lives ACROSS steps in fold-ready rows —
 * F = (x1..xH, x0 in a spare lane), G = (x_{L-1}..x_{L-H}) — so the symmetric fold is
 * u = F+G, v = F-G on whole registers, the output rows Xlo = A+iB* / Xhi = A-iB* land
 * DIRECTLY in next-step F/G order, and the k=0 output rides the spare lane through a
 * k=0 table column + trailing all-ones x0 row. ZERO shuffles per step; u/v broadcasts
 * go through aligned scratch so gcc emits {1to8} FMA memory operands instead of
 * port-5 vpermpd (d1_prime's r2 lesson). The map runs on the same rows, junk lanes
 * masked to zero at the G multiply. */
static __m512d row8i(const double *a, const int idx[8])
{
    double t[8] __attribute__((aligned(64)));
    for (int l = 0; l < 8; ++l) t[l] = idx[l] < 0 ? 0.0 : a[idx[l]];
    return _mm512_load_pd(t);
}

static void chain13_reg(const fft1d_plan *p, const double _Complex *x0,
                        const double _Complex *cc, double _Complex *out, int m)
{
    static const int iFr[8] = {2,4,6,8,10,12, 0,-1}, iFi[8] = {3,5,7,9,11,13, 1,-1};
    static const int iGr[8] = {24,22,20,18,16,14,-1,-1};
    static const int iGi[8] = {25,23,21,19,17,15,-1,-1};
    const double *Ct = p->twc, *St = p->twc + 7*8;
    const double *a = (const double *)x0, *cv = (const double *)cc;
    const __mmask8 m6 = 0x3F;
    __m512d Fr = row8i(a, iFr),  Fi = row8i(a, iFi);
    __m512d Gr = row8i(a, iGr),  Gi = row8i(a, iGi);
    __m512d cFr = row8i(cv, iFr), cFi = row8i(cv, iFi);
    __m512d cGr = row8i(cv, iGr), cGi = row8i(cv, iGi);
    /* Broadcasts are in-register vpermpd, NOT store + {1to8} reload: an 8B load two
     * instructions behind a fresh 64B zmm store blocks store-forwarding, and 12 such
     * loads on the critical path measured 0.17 us/step vs 0.03 (chain31 gets away with
     * the scratch because its loads trail eight stores back). */
    for (int s = 0; s < m; ++s) {
        __m512d u_r = _mm512_add_pd(Fr, Gr), u_i = _mm512_add_pd(Fi, Gi);
        __m512d v_r = _mm512_sub_pd(Fr, Gr), v_i = _mm512_sub_pd(Fi, Gi);
        __m512d A0r, A0i, A1r, A1i, B0r, B0i, B1r, B1i;
        {
            const __m512i b0 = _mm512_set1_epi64(0), b1 = _mm512_set1_epi64(1);
            __m512d c0 = _mm512_load_pd(Ct),      s0 = _mm512_load_pd(St);
            __m512d c1 = _mm512_load_pd(Ct + 8),  s1 = _mm512_load_pd(St + 8);
            A0r = _mm512_mul_pd(_mm512_permutexvar_pd(b0, u_r), c0);
            A0i = _mm512_mul_pd(_mm512_permutexvar_pd(b0, u_i), c0);
            B0r = _mm512_mul_pd(_mm512_permutexvar_pd(b0, v_r), s0);
            B0i = _mm512_mul_pd(_mm512_permutexvar_pd(b0, v_i), s0);
            A1r = _mm512_mul_pd(_mm512_permutexvar_pd(b1, u_r), c1);
            A1i = _mm512_mul_pd(_mm512_permutexvar_pd(b1, u_i), c1);
            B1r = _mm512_mul_pd(_mm512_permutexvar_pd(b1, v_r), s1);
            B1i = _mm512_mul_pd(_mm512_permutexvar_pd(b1, v_i), s1);
        }
        for (int j = 2; j < 6; j += 2) {
            const __m512i bj0 = _mm512_set1_epi64(j), bj1 = _mm512_set1_epi64(j + 1);
            __m512d c0 = _mm512_load_pd(Ct + 8*j),     s0 = _mm512_load_pd(St + 8*j);
            __m512d c1 = _mm512_load_pd(Ct + 8*j + 8), s1 = _mm512_load_pd(St + 8*j + 8);
            A0r = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj0, u_r), c0, A0r);
            A0i = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj0, u_i), c0, A0i);
            B0r = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj0, v_r), s0, B0r);
            B0i = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj0, v_i), s0, B0i);
            A1r = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj1, u_r), c1, A1r);
            A1i = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj1, u_i), c1, A1i);
            B1r = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj1, v_r), s1, B1r);
            B1i = _mm512_fmadd_pd(_mm512_permutexvar_pd(bj1, v_i), s1, B1i);
        }
        {   /* x0 row: adds x0 (= u lane 6) to every output incl. the k=0 lane */
            const __m512i b6 = _mm512_set1_epi64(6);
            __m512d c6 = _mm512_load_pd(Ct + 8*6);
            A0r = _mm512_fmadd_pd(_mm512_permutexvar_pd(b6, u_r), c6, A0r);
            A0i = _mm512_fmadd_pd(_mm512_permutexvar_pd(b6, u_i), c6, A0i);
        }
        __m512d Ar = _mm512_add_pd(A0r, A1r), Ai = _mm512_add_pd(A0i, A1i);
        __m512d Br = _mm512_add_pd(B0r, B1r), Bi = _mm512_add_pd(B0i, B1i);
        /* Xlo lanes = X1..X6, X0 in lane 6; Xhi lanes = X12..X7 (next G order) */
        __m512d zr = _mm512_add_pd(_mm512_add_pd(Ar, Bi), cFr);
        __m512d zi = _mm512_add_pd(_mm512_sub_pd(Ai, Br), cFi);
        __m512d h  = BL_M2(zr, zi);
        __m512d sc = map_scale_fast(h);
        Fr = _mm512_mul_pd(zr, sc); Fi = _mm512_mul_pd(zi, sc);
        zr = _mm512_add_pd(_mm512_sub_pd(Ar, Bi), cGr);
        zi = _mm512_add_pd(_mm512_add_pd(Ai, Br), cGi);
        h  = BL_M2(zr, zi);
        sc = map_scale_fast(h);
        Gr = _mm512_maskz_mul_pd(m6, zr, sc); Gi = _mm512_maskz_mul_pd(m6, zi, sc);
    }
    double fr[8] __attribute__((aligned(64))), fi[8] __attribute__((aligned(64)));
    double gr[8] __attribute__((aligned(64))), gi[8] __attribute__((aligned(64)));
    _mm512_store_pd(fr, Fr); _mm512_store_pd(fi, Fi);
    _mm512_store_pd(gr, Gr); _mm512_store_pd(gi, Gi);
    double *o = (double *)out;
    o[0] = fr[6]; o[1] = fi[6];
    for (int k = 1; k <= 6; ++k) {
        o[2*k] = fr[k-1];        o[2*k+1] = fi[k-1];
        o[2*(13-k)] = gr[k-1];   o[2*(13-k)+1] = gi[k-1];
    }
}

static void chain31_reg(const fft1d_plan *p, const double _Complex *x0,
                        const double _Complex *cc, double _Complex *out, int m)
{
    static const int iF0r[8] = {2,4,6,8,10,12,14,16};
    static const int iF1r[8] = {18,20,22,24,26,28,30, 0};
    static const int iG0r[8] = {60,58,56,54,52,50,48,46};
    static const int iG1r[8] = {44,42,40,38,36,34,32,-1};
    int idx[8];
    const double *Ct = p->twc, *St = p->twc + 16*16;
    const double *a = (const double *)x0, *cv = (const double *)cc;
    const __mmask8 m7 = 0x7F;
    __m512d F0r, F0i, F1r, F1i, G0r, G0i, G1r, G1i;
    __m512d cF0r, cF0i, cF1r, cF1i, cG0r, cG0i, cG1r, cG1i;
#define ODD(src, dst_r, dst_i) do { \
        for (int l = 0; l < 8; ++l) idx[l] = (src)[l] < 0 ? -1 : (src)[l] + 1; \
        dst_r = row8i(a, src);  dst_i = row8i(a, idx); } while (0)
    ODD(iF0r, F0r, F0i); ODD(iF1r, F1r, F1i);
    ODD(iG0r, G0r, G0i); ODD(iG1r, G1r, G1i);
#undef ODD
#define ODDC(src, dst_r, dst_i) do { \
        for (int l = 0; l < 8; ++l) idx[l] = (src)[l] < 0 ? -1 : (src)[l] + 1; \
        dst_r = row8i(cv, src); dst_i = row8i(cv, idx); } while (0)
    ODDC(iF0r, cF0r, cF0i); ODDC(iF1r, cF1r, cF1i);
    ODDC(iG0r, cG0r, cG0i); ODDC(iG1r, cG1r, cG1i);
#undef ODDC
    double us[16] __attribute__((aligned(64))), uis[16] __attribute__((aligned(64)));
    double vs[16] __attribute__((aligned(64))), vis[16] __attribute__((aligned(64)));
    for (int s = 0; s < m; ++s) {
        _mm512_store_pd(us,      _mm512_add_pd(F0r, G0r));
        _mm512_store_pd(us + 8,  _mm512_add_pd(F1r, G1r));
        _mm512_store_pd(uis,     _mm512_add_pd(F0i, G0i));
        _mm512_store_pd(uis + 8, _mm512_add_pd(F1i, G1i));
        _mm512_store_pd(vs,      _mm512_sub_pd(F0r, G0r));
        _mm512_store_pd(vs + 8,  _mm512_sub_pd(F1r, G1r));
        _mm512_store_pd(vis,     _mm512_sub_pd(F0i, G0i));
        _mm512_store_pd(vis + 8, _mm512_sub_pd(F1i, G1i));
        __asm__("" : "+m"(*(double (*)[16])us), "+m"(*(double (*)[16])uis),
                     "+m"(*(double (*)[16])vs), "+m"(*(double (*)[16])vis));
        __m512d A0r = _mm512_setzero_pd(), A0i = A0r, A1r = A0r, A1i = A0r;
        __m512d B0r = A0r, B0i = A0r, B1r = A0r, B1i = A0r;
        _Pragma("GCC unroll 15")
        for (int j = 0; j < 15; ++j) {
            __m512d c0 = _mm512_load_pd(Ct + 16*j), c1 = _mm512_load_pd(Ct + 16*j + 8);
            __m512d s0 = _mm512_load_pd(St + 16*j), s1 = _mm512_load_pd(St + 16*j + 8);
            A0r = _mm512_fmadd_pd(_mm512_set1_pd(us[j]),  c0, A0r);
            A1r = _mm512_fmadd_pd(_mm512_set1_pd(us[j]),  c1, A1r);
            A0i = _mm512_fmadd_pd(_mm512_set1_pd(uis[j]), c0, A0i);
            A1i = _mm512_fmadd_pd(_mm512_set1_pd(uis[j]), c1, A1i);
            B0r = _mm512_fmadd_pd(_mm512_set1_pd(vs[j]),  s0, B0r);
            B1r = _mm512_fmadd_pd(_mm512_set1_pd(vs[j]),  s1, B1r);
            B0i = _mm512_fmadd_pd(_mm512_set1_pd(vis[j]), s0, B0i);
            B1i = _mm512_fmadd_pd(_mm512_set1_pd(vis[j]), s1, B1i);
        }
        {   /* x0 row (u lane 15 = x0): all-ones C over both halves incl. k=0 lane */
            __m512d c0 = _mm512_load_pd(Ct + 16*15), c1 = _mm512_load_pd(Ct + 16*15 + 8);
            A0r = _mm512_fmadd_pd(_mm512_set1_pd(us[15]),  c0, A0r);
            A1r = _mm512_fmadd_pd(_mm512_set1_pd(us[15]),  c1, A1r);
            A0i = _mm512_fmadd_pd(_mm512_set1_pd(uis[15]), c0, A0i);
            A1i = _mm512_fmadd_pd(_mm512_set1_pd(uis[15]), c1, A1i);
        }
        /* Xlo0 = X1..X8, Xlo1 = X9..X15 + X0 lane 7; Xhi0 = X30..X23, Xhi1 = X22..X16 */
        __m512d zr = _mm512_add_pd(_mm512_add_pd(A0r, B0i), cF0r);
        __m512d zi = _mm512_add_pd(_mm512_sub_pd(A0i, B0r), cF0i);
        __m512d h  = BL_M2(zr, zi);
        __m512d sc = map_scale_fast(h);
        F0r = _mm512_mul_pd(zr, sc); F0i = _mm512_mul_pd(zi, sc);
        zr = _mm512_add_pd(_mm512_add_pd(A1r, B1i), cF1r);
        zi = _mm512_add_pd(_mm512_sub_pd(A1i, B1r), cF1i);
        h  = BL_M2(zr, zi);
        sc = map_scale_fast(h);
        F1r = _mm512_mul_pd(zr, sc); F1i = _mm512_mul_pd(zi, sc);
        zr = _mm512_add_pd(_mm512_sub_pd(A0r, B0i), cG0r);
        zi = _mm512_add_pd(_mm512_add_pd(A0i, B0r), cG0i);
        h  = BL_M2(zr, zi);
        sc = map_scale_fast(h);
        G0r = _mm512_mul_pd(zr, sc); G0i = _mm512_mul_pd(zi, sc);
        zr = _mm512_add_pd(_mm512_sub_pd(A1r, B1i), cG1r);
        zi = _mm512_add_pd(_mm512_add_pd(A1i, B1r), cG1i);
        h  = BL_M2(zr, zi);
        sc = map_scale_fast(h);
        G1r = _mm512_maskz_mul_pd(m7, zr, sc); G1i = _mm512_maskz_mul_pd(m7, zi, sc);
    }
    double fr[16] __attribute__((aligned(64))), fi[16] __attribute__((aligned(64)));
    double gr[16] __attribute__((aligned(64))), gi[16] __attribute__((aligned(64)));
    _mm512_store_pd(fr, F0r);     _mm512_store_pd(fi, F0i);
    _mm512_store_pd(fr + 8, F1r); _mm512_store_pd(fi + 8, F1i);
    _mm512_store_pd(gr, G0r);     _mm512_store_pd(gi, G0i);
    _mm512_store_pd(gr + 8, G1r); _mm512_store_pd(gi + 8, G1i);
    double *o = (double *)out;
    o[0] = fr[15]; o[1] = fi[15];
    for (int k = 1; k <= 15; ++k) {
        o[2*k] = fr[k-1];        o[2*k+1] = fi[k-1];
        o[2*(31-k)] = gr[k-1];   o[2*(31-k)+1] = gi[k-1];
    }
}

/* Register-resident B=1 chain for L=64: the four-step 8x8 kernel maps natural-order
 * rows (row r = elements 8r..8r+7) to natural-order rows, so the 16 state vectors
 * stay in zmm across ALL m steps -- no SoA scratch round trip per step and no separate
 * map pass. Only the c-field rows are re-loaded each step (L1-resident scratch). The
 * one tr8 pair per step is internal to the transform, exactly as in fs64. */
static void chain64_reg(fft1d_plan *p, const double _Complex *x0,
                        const double _Complex *cc, double _Complex *out, int m)
{
    double *sxr = (double *)p->xr, *sxi = (double *)p->xi;
    double *scr = (double *)p->cr, *sci = (double *)p->ci;
    deint8((const double *)x0, sxr, sxi, 64);
    deint8((const double *)cc, scr, sci, 64);
    const v8 *tw = (const v8 *)p->twv;    /* [k2-1]=re, [7+k2-1]=im, lane n1 */
    v8 Sr[8], Si[8];
    for (int r = 0; r < 8; ++r) {
        Sr[r] = *(v8 *)(sxr + 8*r);
        Si[r] = *(v8 *)(sxi + 8*r);
    }
    for (int s = 0; s < m; ++s) {
        v8 Br[8], Bi[8];
        fft8_v8(Sr, Si, 1, Br, Bi, 1);           /* inner FFT over rows n2, lanes n1 */
        for (int k2 = 1; k2 < 8; ++k2) {
            v8 c = tw[k2-1], sn = tw[7 + k2-1];
            v8 a = Br[k2], b = Bi[k2];
            Br[k2] = a*c - b*sn;
            Bi[k2] = a*sn + b*c;
        }
        tr8(Br); tr8(Bi);
        fft8_v8(Br, Bi, 1, Sr, Si, 1);           /* outer FFT: rows k1, lanes k2 */
        for (int r = 0; r < 8; ++r) {
            __m512d zr = _mm512_add_pd((__m512d)Sr[r], _mm512_load_pd(scr + 8*r));
            __m512d zi = _mm512_add_pd((__m512d)Si[r], _mm512_load_pd(sci + 8*r));
            __m512d h  = BL_M2(zr, zi);
            __m512d sc = map_scale_fast(h);
            Sr[r] = (v8)_mm512_mul_pd(zr, sc);
            Si[r] = (v8)_mm512_mul_pd(zi, sc);
        }
    }
    for (int r = 0; r < 8; ++r) {
        *(v8 *)(sxr + 8*r) = Sr[r];
        *(v8 *)(sxi + 8*r) = Si[r];
    }
    inter8((double *)out, sxr, sxi, 64);
}

/* ymm variant of the r6 map for the L=32 register chain (state rows are v4);
 * same Goldschmidt + early-seeded-rcp shape, argument pre-floored via BL_M2_4 */
#define BL_M2_4(zr_, zi_) _mm256_fmadd_pd(zr_, zr_, \
        _mm256_fmadd_pd(zi_, zi_, _mm256_set1_pd(1e-100)))
static inline __m256d map_scale_fast4(__m256d m2)
{
    const __m256d half = _mm256_set1_pd(0.5), one = _mm256_set1_pd(1.0);
    const __m256d two  = _mm256_set1_pd(2.0);
    __m256d y  = _mm256_rsqrt14_pd(m2);
    __m256d x  = _mm256_mul_pd(m2, y);
    __m256d h  = _mm256_mul_pd(y, half);
    __m256d q  = _mm256_rcp14_pd(_mm256_add_pd(x, one));
    __m256d r1 = _mm256_fnmadd_pd(x, h, half);
    x = _mm256_fmadd_pd(x, r1, x);  h = _mm256_fmadd_pd(h, r1, h);
    __m256d r2 = _mm256_fnmadd_pd(x, h, half);
    x = _mm256_fmadd_pd(x, r2, x);
    __m256d d = _mm256_add_pd(x, one);
    q = _mm256_mul_pd(q, _mm256_fnmadd_pd(d, q, two));
    q = _mm256_mul_pd(q, _mm256_fnmadd_pd(d, q, two));
    return q;
}

/* L=32 register chain: fs32's natural v4-row layout is also closed under the
 * transform (8 rows of 4 elements in and out), so 16 ymm state rows persist across
 * steps. ymm mixes spread across p0/p1/p5 (d1_composite's lesson), and the map runs
 * per-row at ymm width -- fine here, the whole map is 8 rows. */
static void chain32_reg(fft1d_plan *p, const double _Complex *x0,
                        const double _Complex *cc, double _Complex *out, int m)
{
    double *sxr = (double *)p->xr, *sxi = (double *)p->xi;
    double *scr = (double *)p->cr, *sci = (double *)p->ci;
    deint8((const double *)x0, sxr, sxi, 32);
    deint8((const double *)cc, scr, sci, 32);
    const v4 *tw = (const v4 *)p->twv;    /* [k2-1]=re, [7+k2-1]=im, lane n1 */
    v4 Sr[8], Si[8];
    for (int r = 0; r < 8; ++r) {
        Sr[r] = *(v4 *)(sxr + 4*r);
        Si[r] = *(v4 *)(sxi + 4*r);
    }
    for (int s = 0; s < m; ++s) {
        v4 Br[8], Bi[8];
        fft8_v4(Sr, Si, 1, Br, Bi, 1);
        for (int k2 = 1; k2 < 8; ++k2) {
            v4 c = tw[k2-1], sn = tw[7 + k2-1];
            v4 a = Br[k2], b = Bi[k2];
            Br[k2] = a*c - b*sn;
            Bi[k2] = a*sn + b*c;
        }
        tr4(Br); tr4(Br + 4); tr4(Bi); tr4(Bi + 4);
        /* output row of X[8 k1 + k2]: lanes k2 0..3 land in natural v4-row 2k1,
         * lanes 4..7 in row 2k1+1 -- write with row stride 2 so Sr[t] stays the
         * natural row t that the next step's inner FFT expects */
        fft4_v4(Br, Bi, 1, Sr, Si, 2);
        fft4_v4(Br + 4, Bi + 4, 1, Sr + 1, Si + 1, 2);
        for (int r = 0; r < 8; ++r) {
            __m256d zr = _mm256_add_pd((__m256d)Sr[r], _mm256_load_pd(scr + 4*r));
            __m256d zi = _mm256_add_pd((__m256d)Si[r], _mm256_load_pd(sci + 4*r));
            __m256d h  = BL_M2_4(zr, zi);
            __m256d sc = map_scale_fast4(h);
            Sr[r] = (v4)_mm256_mul_pd(zr, sc);
            Si[r] = (v4)_mm256_mul_pd(zi, sc);
        }
    }
    for (int r = 0; r < 8; ++r) {
        *(v4 *)(sxr + 4*r) = Sr[r];
        *(v4 *)(sxi + 4*r) = Si[r];
    }
    inter8((double *)out, sxr, sxi, 32);
}

/* L=128 register chain, same natural-rows-closed argument (fs128's 16-row layout);
 * 32 state vectors exceed the register file so gcc spills some, but the SoA scratch
 * round trip and the separate map pass still go away. */
static void chain128_reg(fft1d_plan *p, const double _Complex *x0,
                         const double _Complex *cc, double _Complex *out, int m)
{
    double *sxr = (double *)p->xr, *sxi = (double *)p->xi;
    double *scr = (double *)p->cr, *sci = (double *)p->ci;
    deint8((const double *)x0, sxr, sxi, 128);
    deint8((const double *)cc, scr, sci, 128);
    const v8 *tw = (const v8 *)p->twv;    /* [k2-1]=re, [15+k2-1]=im, lane n1 */
    v8 Sr[16], Si[16];
    for (int r = 0; r < 16; ++r) {
        Sr[r] = *(v8 *)(sxr + 8*r);
        Si[r] = *(v8 *)(sxi + 8*r);
    }
    for (int s = 0; s < m; ++s) {
        v8 Br[16], Bi[16];
        fft16_v8(p->tw + 256, Sr, Si, 1, Br, Bi, 1);
        for (int k2 = 1; k2 < 16; ++k2) {
            v8 c = tw[k2-1], sn = tw[15 + k2-1];
            v8 a = Br[k2], b = Bi[k2];
            Br[k2] = a*c - b*sn;
            Bi[k2] = a*sn + b*c;
        }
        tr8(Br); tr8(Br + 8); tr8(Bi); tr8(Bi + 8);
        /* X[16 k1 + k2]: lanes k2 0..7 = natural row 2k1, lanes 8..15 = row 2k1+1 */
        fft8_v8(Br, Bi, 1, Sr, Si, 2);
        fft8_v8(Br + 8, Bi + 8, 1, Sr + 1, Si + 1, 2);
        for (int r = 0; r < 16; ++r) {
            __m512d zr = _mm512_add_pd((__m512d)Sr[r], _mm512_load_pd(scr + 8*r));
            __m512d zi = _mm512_add_pd((__m512d)Si[r], _mm512_load_pd(sci + 8*r));
            __m512d h  = BL_M2(zr, zi);
            __m512d sc = map_scale_fast(h);
            Sr[r] = (v8)_mm512_mul_pd(zr, sc);
            Si[r] = (v8)_mm512_mul_pd(zi, sc);
        }
    }
    for (int r = 0; r < 16; ++r) {
        *(v8 *)(sxr + 8*r) = Sr[r];
        *(v8 *)(sxi + 8*r) = Si[r];
    }
    inter8((double *)out, sxr, sxi, 128);
}

/* ===== L=60 B=1 chain, TAKEN FROM d1_composite (r3 chain60_step_v4 + CH_RNAT/
 * CH_CBNAT, ported near-verbatim; their map calls rewired to my r6 Goldschmidt
 * map). The structural idea is theirs: natural indices decompose into 15 cosets
 * {r, r+15, r+30, r+45}; both the stage-A operand pairs {p,p+15}/{p+30,p+45}
 * AND the stage-C emitted ymm pairs {k,k+45} live inside cosets, and the coset
 * class of an emission is (10*n2+6*k3) mod 15 independent of pr -- so the state
 * lives as 15 DFT-4-READY zmm rows. Per step: stage A = 30 aligned 32B loads,
 * zero inserts; stage C pairs pr0/pr1 ymms into coset zmms, maps 8-wide, and
 * ONE output permute pair lands each coset back in row layout (15 aligned 64B
 * row stores; every next-step load forwards 1:1 or contained). This replaces
 * my generic fs60+map_apply remainder chain (0.138 vs their 0.11 on the r5
 * board at 60 B=1 chained). CH_RNAT[q] = natural index at state position q,
 * CH_CBNAT = c-permutation in emission-group order. */
static const int CH_RNAT[60] = {0,15,30,45,12,27,42,57,24,39,54,9,36,51,6,21,48,3,18,33,20,35,50,5,32,47,2,17,44,59,14,29,56,11,26,41,8,23,38,53,40,55,10,25,52,7,22,37,4,19,34,49,16,31,46,1,28,43,58,13};
static const int CH_CBNAT[60] = {0,45,30,15,36,21,6,51,24,9,54,39,12,57,42,27,40,25,10,55,16,1,46,31,4,49,34,19,52,37,22,7,48,33,18,3,28,13,58,43,20,5,50,35,56,41,26,11,44,29,14,59,32,17,2,47,8,53,38,23};

__attribute__((aligned(64), hot)) static void chain60_step_v4(double *restrict st, const double *restrict cb)
{
    __m256d wp[2][15];
    {
        const __m256d E4 = _mm256_set_pd(-1.0, 1.0, 1.0, 1.0);
        _Pragma("GCC unroll 15") for (int col = 0; col < 15; ++col) {
            __m256d A = _mm256_load_pd(st + 8*col);
            __m256d B = _mm256_load_pd(st + 8*col + 4);
            __m256d S = _mm256_add_pd(A, B), D = _mm256_sub_pd(A, B);
            __m256d P = _mm256_permute2f128_pd(S, D, 0x20);
            __m256d R = _mm256_permute_pd(_mm256_permute2f128_pd(S, D, 0x31), 6);
            wp[0][col] = _mm256_fmadd_pd(R, E4, P);
            wp[1][col] = _mm256_fnmadd_pd(R, E4, P);
        }
        const __m256d half = _mm256_set1_pd(0.5);
        const __m256d S3E = _mm256_set_pd(-S3W, S3W, -S3W, S3W);
        _Pragma("GCC unroll 2") for (int pr = 0; pr < 2; ++pr)
            _Pragma("GCC unroll 5") for (int n3 = 0; n3 < 5; ++n3) {
                __m256d x1 = wp[pr][5+n3], x2 = wp[pr][10+n3];
                __m256d t = _mm256_add_pd(x1, x2), u = _mm256_sub_pd(x1, x2);
                __m256d x0 = wp[pr][n3];
                wp[pr][n3] = _mm256_add_pd(x0, t);
                __m256d vv = _mm256_fnmadd_pd(half, t, x0);
                __m256d swu = _mm256_permute_pd(u, 0x5);
                wp[pr][5+n3]  = _mm256_fmadd_pd(swu, S3E, vv);
                wp[pr][10+n3] = _mm256_fnmadd_pd(swu, S3E, vv);
            }
    }
    {
        const __m256d c51v = _mm256_set1_pd(C51), c52v = _mm256_set1_pd(C52);
        const __m256d S1E = _mm256_set_pd(-S51, S51, -S51, S51);
        const __m256d S2E = _mm256_set_pd(-S52, S52, -S52, S52);
        const __m512i IRE  = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
        const __m512i IIM  = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
        /* row-layout output permutes: uniform across all 7 groups (their generator) */
        const __m512i IDXA = _mm512_setr_epi64(0, 8, 3, 11, 2, 10, 1, 9);
        const __m512i IDXB = _mm512_setr_epi64(4, 12, 7, 15, 6, 14, 5, 13);
        const __m512i IDX4 = _mm512_setr_epi64(0, 1, 6, 7, 4, 5, 2, 3);
        /* the 5 stage-C output ymms for (pr,n2), k3 order 0,1,4,2,3 */
#define STAGEC5(pr, n2, e0, e1, e2, e3, e4) do {                               \
        const int b_ = 5*(n2);                                                 \
        __m256d x0 = wp[pr][b_];                                               \
        __m256d t1 = _mm256_add_pd(wp[pr][b_+1], wp[pr][b_+4]);                \
        __m256d t3 = _mm256_sub_pd(wp[pr][b_+1], wp[pr][b_+4]);                \
        __m256d t2 = _mm256_add_pd(wp[pr][b_+2], wp[pr][b_+3]);                \
        __m256d t4 = _mm256_sub_pd(wp[pr][b_+2], wp[pr][b_+3]);                \
        e0 = _mm256_add_pd(x0, _mm256_add_pd(t1, t2));                         \
        __m256d a1 = _mm256_fmadd_pd(c52v, t2, _mm256_fmadd_pd(c51v, t1, x0)); \
        __m256d a2 = _mm256_fmadd_pd(c51v, t2, _mm256_fmadd_pd(c52v, t1, x0)); \
        __m256d sw3 = _mm256_permute_pd(t3, 0x5), sw4 = _mm256_permute_pd(t4, 0x5); \
        __m256d m1 = _mm256_fmadd_pd(sw4, S2E, _mm256_mul_pd(sw3, S1E));       \
        __m256d m2 = _mm256_fnmadd_pd(sw4, S1E, _mm256_mul_pd(sw3, S2E));      \
        e1 = _mm256_add_pd(a1, m1);   /* k3=1 */                               \
        e2 = _mm256_sub_pd(a1, m1);   /* k3=4 */                               \
        e3 = _mm256_add_pd(a2, m2);   /* k3=2 */                               \
        e4 = _mm256_sub_pd(a2, m2);   /* k3=3 */ } while (0)
        /* 8 complexes (two coset zmms) at cb offset `base`: add c interleaved,
         * deinterleave IN REGISTERS, 8-wide map, and the output permutes land
         * each coset in row layout: two aligned 64B row stores */
#define EMIT8Z(base, rA, rB, za, zb) do {                                      \
        __m512d zA = _mm512_add_pd((za), _mm512_load_pd(cb + (base)));         \
        __m512d zB = _mm512_add_pd((zb), _mm512_load_pd(cb + (base) + 8));     \
        __m512d zr = _mm512_permutex2var_pd(zA, IRE, zB);                      \
        __m512d zi = _mm512_permutex2var_pd(zA, IIM, zB);                      \
        __m512d s_ = map_scale_fast(BL_M2(zr, zi));                            \
        zr = _mm512_mul_pd(zr, s_);  zi = _mm512_mul_pd(zi, s_);               \
        _mm512_store_pd(st + 8*(rA), _mm512_permutex2var_pd(zr, IDXA, zi));    \
        _mm512_store_pd(st + 8*(rB), _mm512_permutex2var_pd(zr, IDXB, zi));    \
        } while (0)
        /* the last coset (n2=2, k3=3): q duplicated per 128-lane, one store;
         * the 1e-100 junk-lane floor is an explicit add here (no BL_M2 form) */
#define EMIT4Z(base, r, za) do {                                               \
        __m512d z_ = _mm512_add_pd((za), _mm512_load_pd(cb + (base)));         \
        __m512d q_ = _mm512_mul_pd(z_, z_);                                    \
        q_ = _mm512_add_pd(q_, _mm512_permute_pd(q_, 0x55));                   \
        q_ = _mm512_add_pd(q_, _mm512_set1_pd(1e-100));                        \
        z_ = _mm512_mul_pd(z_, map_scale_fast(q_));                            \
        _mm512_store_pd(st + 8*(r), _mm512_permutexvar_pd(IDX4, z_)); } while (0)
#define CZ(a, b) _mm512_insertf64x4(_mm512_castpd256_pd512(a), (b), 1)
        /* group schedule (rows from their generator): n2=0 -> G0(rows 0,3),
         * G1(2,1), hold k3=3; n2=1 -> G2(10,13), G3(12,11), G4(4,14) pairs the
         * two held cosets; n2=2 -> G5(5,8), G6(7,6), G7(row 9) 4-wide. */
        __m512d pend = _mm512_setzero_pd();
        _Pragma("GCC unroll 3") for (int n2 = 0; n2 < 3; ++n2) {
            __m256d eA0, eA1, eA2, eA3, eA4, eB0, eB1, eB2, eB3, eB4;
            STAGEC5(0, n2, eA0, eA1, eA2, eA3, eA4);
            STAGEC5(1, n2, eB0, eB1, eB2, eB3, eB4);
            __m512d z4 = CZ(eA4, eB4);
            if (n2 == 0) {
                EMIT8Z(0,  0, 3, CZ(eA0, eB0), CZ(eA1, eB1));
                EMIT8Z(16, 2, 1, CZ(eA2, eB2), CZ(eA3, eB3));
                pend = z4;
            } else if (n2 == 1) {
                EMIT8Z(32, 10, 13, CZ(eA0, eB0), CZ(eA1, eB1));
                EMIT8Z(48, 12, 11, CZ(eA2, eB2), CZ(eA3, eB3));
                EMIT8Z(64, 4, 14, pend, z4);
            } else {
                EMIT8Z(80, 5, 8, CZ(eA0, eB0), CZ(eA1, eB1));
                EMIT8Z(96, 7, 6, CZ(eA2, eB2), CZ(eA3, eB3));
                EMIT4Z(112, 9, z4);
            }
        }
#undef STAGEC5
#undef EMIT8Z
#undef EMIT4Z
#undef CZ
    }
}

/* whole chain for one transform: state kept in row layout in a local aligned
 * buffer across all m steps; permute in/out once per chain */
static void chain60_row(const double *restrict x0, const double *restrict c,
                        double *restrict out, int m)
{
    _Alignas(64) double st[120], cb[120];
    for (int q = 0; q < 60; ++q) {
        st[2*q] = x0[2*CH_RNAT[q]];  st[2*q+1] = x0[2*CH_RNAT[q]+1];
        cb[2*q] = c[2*CH_CBNAT[q]];  cb[2*q+1] = c[2*CH_CBNAT[q]+1];
    }
    for (int s = 0; s < m; ++s) chain60_step_v4(st, cb);
    for (int q = 0; q < 60; ++q) {
        out[2*CH_RNAT[q]] = st[2*q];  out[2*CH_RNAT[q]+1] = st[2*q+1];
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
    return "SoA 8-lane zmm batch-lane engine: interleaved-pair m=1 kernels at 13/31 "
           "(d1_prime), in-register AoS codelets at 32/64 (d1_pow2), broadcast-fed "
           "PFA ymm1 kernel at 60 (d1_composite), register-resident B=1 chains "
           "(fold-ready A/B rows at 13/31, natural-row four-step at 32/64/128, "
           "coset-row PFA at 60 per d1_composite), L1-blocked fused FFT+map batched "
           "chain with Goldschmidt early-seeded-rcp map (d1_prime)";
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

    /* interleaved-pair tables for the 13/31 m=1 kernels (d1_prime's r3 layout):
     * coefficients duplicated per 128-bit pair, sin stored (+s,-s), k=0 column
     * (cos=1, sin=0) riding a spare pair so X0 = x0 + sum u falls out free. */
    if (L == 13) {
        void *tpp = NULL;
        if (posix_memalign(&tpp, 64, (size_t)6 * 4 * 8 * sizeof(double))) {
            fft1d_destroy(p); return NULL;
        }
        p->tp = tpp;
        memset(p->tp, 0, (size_t)6 * 4 * 8 * sizeof(double));
        for (int j = 1; j <= 6; ++j) {
            double *cpA = p->tp + (size_t)(j-1)*32, *cpB = cpA + 8;
            double *spA = cpB + 8,                  *spB = spA + 8;
            for (int k = 1; k <= 6; ++k) {
                double c = tw_cosl((j * k) % 13, 13);
                double s = tw_sinl((j * k) % 13, 13);
                if (k <= 4) { cpA[2*(k-1)] = cpA[2*(k-1)+1] = c;
                              spA[2*(k-1)] = s;  spA[2*(k-1)+1] = -s; }
                else        { cpB[2*(k-5)] = cpB[2*(k-5)+1] = c;
                              spB[2*(k-5)] = s;  spB[2*(k-5)+1] = -s; }
            }
            cpB[4] = cpB[5] = 1.0;   /* k=0 column rides pair 2 of the B row */
        }
    } else if (L == 31) {
        /* [j][8][8]: rows cpA..cpD then spA..spD; A = k 1..4, B = 5..8,
         * C = 9..12, D = 13..15 + k=0 column in pair 3 */
        void *tpp = NULL;
        if (posix_memalign(&tpp, 64, (size_t)15 * 8 * 8 * sizeof(double))) {
            fft1d_destroy(p); return NULL;
        }
        p->tp = tpp;
        memset(p->tp, 0, (size_t)15 * 8 * 8 * sizeof(double));
        for (int j = 1; j <= 15; ++j) {
            double *base = p->tp + (size_t)(j-1)*64;
            for (int k = 1; k <= 15; ++k) {
                double c = tw_cosl((j * k) % 31, 31);
                double s = tw_sinl((j * k) % 31, 31);
                int row = (k-1) >> 2, t = (k-1) & 3;
                base[8*row + 2*t]     = c;  base[8*row + 2*t + 1]     =  c;
                base[8*(row+4) + 2*t] = s;  base[8*(row+4) + 2*t + 1] = -s;
            }
            base[8*3 + 6] = base[8*3 + 7] = 1.0;   /* k=0 column: pair 3 of row D */
        }
    }

    /* L=32 codelet table (d1_pow2's dup format): per 4-p group
     * [w1r|w1p|w2r|w2p|w3r|w3p] x 8 doubles, wp = (+s,-s) for w = e^{-i th} */
    if (L == 32) {
        void *t32 = NULL;
        if (posix_memalign(&t32, 64, 96 * sizeof(double))) {
            fft1d_destroy(p); return NULL;
        }
        p->tw32 = t32;
        for (int q = 0; q < 8; ++q) {
            int g = q / 4, j = q % 4;
            for (int r = 1; r <= 3; ++r) {
                double *base = p->tw32 + g*48 + (r-1)*16;
                base[2*j] = base[2*j + 1] = tw_cosl((q * r) % 32, 32);
                base[8 + 2*j]     =  tw_sinl((q * r) % 32, 32);
                base[8 + 2*j + 1] = -tw_sinl((q * r) % 32, 32);
            }
        }
    }

    /* L=64 codelet tables (d1_pow2's formats, long-double twiddles):
     * tw64a = s=1 stage, 4 groups of 4 p x [w1r|w1p|w2r|w2p|w3r|w3p] x 8 dbl
     *         (W64^{p r}, r=1..3, cos duplicated per pair, sin as (+s,-s));
     * tw64b = radix-4 n=16 stage, 9 dbl per p: [c1,c2,c3, s1,-s1, s2,-s2, s3,-s3] */
    if (L == 64) {
        void *ta = NULL, *tb = NULL;
        if (posix_memalign(&ta, 64, 192 * sizeof(double)) ||
            posix_memalign(&tb, 64, 40 * sizeof(double))) {
            free(ta); fft1d_destroy(p); return NULL;
        }
        p->tw64a = ta; p->tw64b = tb;
        for (int q = 0; q < 16; ++q) {
            int g = q / 4, j = q % 4;
            for (int r = 1; r <= 3; ++r) {
                double *base = p->tw64a + g*48 + (r-1)*16;
                base[2*j] = base[2*j + 1] = tw_cosl((q * r) % 64, 64);
                base[8 + 2*j]     =  tw_sinl((q * r) % 64, 64);
                base[8 + 2*j + 1] = -tw_sinl((q * r) % 64, 64);
            }
        }
        memset(p->tw64b, 0, 40 * sizeof(double));
        for (int pp = 0; pp < 4; ++pp)
            for (int r = 1; r <= 3; ++r) {
                p->tw64b[9*pp + (r-1)]       = tw_cosl((pp * r) % 16, 16);
                p->tw64b[9*pp + 3 + 2*(r-1)] = tw_sinl((pp * r) % 16, 16);
                p->tw64b[9*pp + 4 + 2*(r-1)] = -tw_sinl((pp * r) % 16, 16);
            }
    }

    /* L=128 codelet tables, same formats one factor up: tw64a = s=1 stage
     * (8 groups x 48 dbl, W128^{p r}), tw64b = radix-4 n=32 bc table
     * (9 dbl per m, m=0..7, W32^{m r}) */
    if (L == 128) {
        void *ta = NULL, *tb = NULL;
        if (posix_memalign(&ta, 64, 384 * sizeof(double)) ||
            posix_memalign(&tb, 64, 72 * sizeof(double))) {
            free(ta); fft1d_destroy(p); return NULL;
        }
        p->tw64a = ta; p->tw64b = tb;
        for (int q = 0; q < 32; ++q) {
            int g = q / 4, j = q % 4;
            for (int r = 1; r <= 3; ++r) {
                double *base = p->tw64a + g*48 + (r-1)*16;
                base[2*j] = base[2*j + 1] = tw_cosl((q * r) % 128, 128);
                base[8 + 2*j]     =  tw_sinl((q * r) % 128, 128);
                base[8 + 2*j + 1] = -tw_sinl((q * r) % 128, 128);
            }
        }
        for (int m = 0; m < 8; ++m)
            for (int r = 1; r <= 3; ++r) {
                p->tw64b[9*m + (r-1)]       = tw_cosl((m * r) % 32, 32);
                p->tw64b[9*m + 3 + 2*(r-1)] = tw_sinl((m * r) % 32, 32);
                p->tw64b[9*m + 4 + 2*(r-1)] = -tw_sinl((m * r) % 32, 32);
            }
    }

    /* chain-layout tables for the in-register B=1 chain kernels (13/31): rows over j
     * (j = 1..H plus a final all-ones x0 row), lanes = output k. L=13: lane 0..5 =
     * k 1..6, lane 6 = the k=0 column (cos=1, sin=0), lane 7 = 0. L=31: two vectors
     * per row, lanes k=1..8 and k=9..15 + k=0 column in lane 7 of the second. */
    if (L == 13) {
        void *tc = NULL;
        if (posix_memalign(&tc, 64, 2 * 7 * 8 * sizeof(double))) {
            fft1d_destroy(p); return NULL;
        }
        p->twc = tc;
        double *C = p->twc, *S = p->twc + 7*8;
        memset(p->twc, 0, 2 * 7 * 8 * sizeof(double));
        for (int j = 0; j < 6; ++j) {
            for (int k = 1; k <= 6; ++k) {
                C[8*j + k-1] = tw_cosl(((j+1) * k) % 13, 13);
                S[8*j + k-1] = tw_sinl(((j+1) * k) % 13, 13);
            }
            C[8*j + 6] = 1.0;      /* k=0 column */
        }
        for (int l = 0; l < 7; ++l) C[8*6 + l] = 1.0;   /* x0 row */
    } else if (L == 31) {
        void *tc = NULL;
        if (posix_memalign(&tc, 64, 2 * 16 * 16 * sizeof(double))) {
            fft1d_destroy(p); return NULL;
        }
        p->twc = tc;
        double *C = p->twc, *S = p->twc + 16*16;
        memset(p->twc, 0, 2 * 16 * 16 * sizeof(double));
        for (int j = 0; j < 15; ++j) {
            for (int k = 1; k <= 8; ++k) {
                C[16*j + k-1] = tw_cosl(((j+1) * k) % 31, 31);
                S[16*j + k-1] = tw_sinl(((j+1) * k) % 31, 31);
            }
            for (int k = 9; k <= 15; ++k) {
                C[16*j + 8 + k-9] = tw_cosl(((j+1) * k) % 31, 31);
                S[16*j + 8 + k-9] = tw_sinl(((j+1) * k) % 31, 31);
            }
            C[16*j + 15] = 1.0;    /* k=0 column */
        }
        for (int l = 0; l < 16; ++l) C[16*15 + l] = 1.0; /* x0 row */
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

/* Batched m=1 dispatch: on the ICX scoring node every 512-bit shuffle lands on port 5
 * (also an FMA port), so the SoA group path's boundary transposes (~96 p5 uops per
 * transform at 64) plus its multi-pass scratch traffic lose to simply looping the fused
 * single-shot AoS kernels, which touch each transform's memory exactly once. r2 a80n0
 * evidence: 64 0.0725 (SoA) vs 0.0553 (fs64_aos incl. call overhead), 128 0.2462 vs
 * 0.0991, 31 0.0767 vs 0.0685. 32/60 stay SoA (measured the other way). */
#ifdef BL_FORCE_SOA
static int aos_batch(int L) { (void)L; return 0; }
#elif defined(BL_FORCE_AOS)
static int aos_batch(int L) { (void)L; return 1; }
#else
static int aos_batch(int L) { return L == 128; }   /* 13/31/32/60/64 now
                                 dispatch to their own kernels before this */
#endif

void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    const int L = p->L;
    const double *a = (const double *)in;
    double *o = (double *)out;
    if (L == 13) {   /* interleaved-pair kernels at every B (d1_prime r3) */
        if (p->batch == 1) { bl13p_body(a, o, p->tp, 2); return; }
        int b = 0;
        for (; b + 2 <= p->batch; b += 2)
            bl13p_b2(a + 26*(size_t)b, o + 26*(size_t)b, p->tp);
        for (; b < p->batch; ++b)
            bl13p_body(a + 26*(size_t)b, o + 26*(size_t)b, p->tp, 1);
        return;
    }
    if (L == 31) {   /* single-body loop: a two-transform variant needs >32 zmm */
        for (int b = 0; b < p->batch; ++b)
            bl31p_body(a + 62*(size_t)b, o + 62*(size_t)b, p->tp);
        return;
    }
    if (L == 32) { fft32_codelet(p->tw32, a, o, p->batch); return; }
#ifndef BL60_SOA
    if (L == 60) {   /* d1_composite's kernels. r6: their r5 on-node A/B settled the
                        width question -- the plain per-transform ymm1 loop reads
                        0.045-0.046 us/xf vs zmm2x2's 0.052-0.053 on ICX (every
                        512-bit FMA-class op dispatches on p0+p5 only, so zmm2x2's
                        287-instr diet still loses to ymm1's three-wide 256-bit mix).
                        Adopted with their flag map: zmm2x2 kept under -DBL60_ZMM2X2. */
#ifdef BL60_ZMM2X2
        int b = 0;
        for (; b + 2 <= p->batch; b += 2)
            fft60_zmm2x2(a + 120*(size_t)b, o + 120*(size_t)b);
        for (; b < p->batch; ++b)
            fft60_ymm1(a + 120*(size_t)b, o + 120*(size_t)b);
#else
        for (int b = 0; b < p->batch; ++b)
            fft60_ymm1(a + 120*(size_t)b, o + 120*(size_t)b);
#endif
        return;
    }
#endif
    if (L == 64) {   /* B=1 keeps fs64_aos (node r4: 0.049 vs pow2 codelet
                        0.058-0.067); everything else loops the codelet */
#ifndef BL64_CODELET_B1
        if (p->batch == 1) { fs64_aos(p, in, out); return; }
#endif
        fft64_codelet(p->tw64a, p->tw64b, a, o, p->batch);
        return;
    }
#ifndef BL128_AOS
    if (L == 128) {  /* interleaved same-core A/B on a80n0 (r5): codelet
                        0.154-0.157 vs fs128_aos loop 0.165-0.169 at B=512;
                        at B=1 a wash (0.098-0.110 vs steady 0.102-0.103),
                        so B=1 keeps the four-step */
        if (p->batch == 1) { fs128_aos(p, in, out); return; }
        fft128_codelet(p->tw64a, p->tw64b, a, o, p->batch);
        return;
    }
#endif
    int b = 0;
    if (!aos_batch(L))
    for (; b + 8 <= p->batch; b += 8) {
        tload8(in + (size_t)b * L, L, p->xr, p->xi);
        if (L == 128) { fft128_fused(p, out + (size_t)b * L); continue; }
        dofft_v8(p, p->xr, p->xi, p->yr, p->yi, p->wr, p->wi);
        tstore8(out + (size_t)b * L, L, p->yr, p->yi);
    }
    for (; b < p->batch; ++b) {
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
        switch (L) {   /* fused FFT+map steps: no separate map pass, no y round trip */
        case 13:
            for (int s = 0; s < m; ++s) chain_dsym_grp_step(p, 13, 6, p->tw, p->tw + 36);
            break;
        case 31:
            for (int s = 0; s < m; ++s) chain_dsym_grp_step(p, 31, 15, p->tw, p->tw + 225);
            break;
        case 60:  for (int s = 0; s < m; ++s) chain60_grp_step(p);        break;
        case 64:  for (int s = 0; s < m; ++s) chain64_grp_step(p);        break;
        case 128: for (int s = 0; s < m; ++s) chain128_grp_step(p);       break;
        default:
            for (int s = 0; s < m; ++s) {
                dofft_v8(p, p->xr, p->xi, p->yr, p->yi, p->wr, p->wi);
                map_apply((double *)p->yr, (double *)p->yi, (double *)p->cr,
                          (double *)p->ci, (double *)p->xr, (double *)p->xi, L * 8);
            }
        }
        tstore8(final_out + (size_t)b * L, L, p->xr, p->xi);
    }
    for (; b < p->batch; ++b) {
        if (L == 13) { chain13_reg(p, x0 + (size_t)b * L, c + (size_t)b * L,
                                   final_out + (size_t)b * L, m); continue; }
        if (L == 31) { chain31_reg(p, x0 + (size_t)b * L, c + (size_t)b * L,
                                   final_out + (size_t)b * L, m); continue; }
        if (L == 32) { chain32_reg(p, x0 + (size_t)b * L, c + (size_t)b * L,
                                   final_out + (size_t)b * L, m); continue; }
        if (L == 60) { chain60_row((const double *)(x0 + (size_t)b * L),
                                   (const double *)(c + (size_t)b * L),
                                   (double *)(final_out + (size_t)b * L), m); continue; }
        if (L == 64) { chain64_reg(p, x0 + (size_t)b * L, c + (size_t)b * L,
                                   final_out + (size_t)b * L, m); continue; }
        if (L == 128) { chain128_reg(p, x0 + (size_t)b * L, c + (size_t)b * L,
                                     final_out + (size_t)b * L, m); continue; }
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
    free(p->tw); free(p->twv); free(p->twc); free(p->tp); free(p->tw32);
    free(p->tw64a); free(p->tw64b);
    free(p->im); free(p->om);
    free(p->scratch_base);
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

/* 60 = Good-Thomas 3 x 4 x 5, twiddle-free; w layout [n3][n4][n5] = n3*20+n4*5+n5.
 * The im/om args are ignored in favor of the static IM60/OM60 so full unrolling
 * constant-folds every CRT index into an addressing displacement. */
static void KN(fft60)(const int *im, const int *om, const V *xr, const V *xi,
                      V *yr, V *yi, V *wr, V *wi)
{
    (void)im; (void)om;
    _Pragma("GCC unroll 20")
    for (int j = 0; j < 20; ++j)
        KN(fft3io)(xr, xi, IM60 + j, wr + j, wi + j, 20);
    _Pragma("GCC unroll 3")
    for (int n3 = 0; n3 < 3; ++n3)
        _Pragma("GCC unroll 5")
        for (int n5 = 0; n5 < 5; ++n5)
            KN(fft4)(wr + n3*20 + n5, wi + n3*20 + n5, 5,
                     wr + n3*20 + n5, wi + n3*20 + n5, 5);
    _Pragma("GCC unroll 12")
    for (int t = 0; t < 12; ++t)
        KN(fft5o)(wr + 5*t, wi + 5*t, OM60 + 5*t, yr, yi);
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
