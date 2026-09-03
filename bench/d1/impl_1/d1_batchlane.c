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
 * machine. G = ceil(H/8) accumulator groups; table lanes beyond H are zero-padded. */
static void dsk8(int P, int H, int G, const double *Ct, const double *St,
                 const double *re, const double *im, double *qr, double *qi)
{
    const __m512d *C = (const __m512d *)Ct, *S = (const __m512d *)St;
    __m512d Arv[2], Aiv[2], Brv[2], Biv[2];
    for (int g = 0; g < G; ++g) {
        Arv[g] = _mm512_setzero_pd(); Aiv[g] = _mm512_setzero_pd();
        Brv[g] = _mm512_setzero_pd(); Biv[g] = _mm512_setzero_pd();
    }
    double s0r = re[0], s0i = im[0], sumr = s0r, sumi = s0i;
    for (int j = 0; j < H; ++j) {
        double ur = re[1+j] + re[P-1-j], ui = im[1+j] + im[P-1-j];
        double vr = re[1+j] - re[P-1-j], vi = im[1+j] - im[P-1-j];
        sumr += ur; sumi += ui;
        __m512d urv = _mm512_set1_pd(ur), uiv = _mm512_set1_pd(ui);
        __m512d vrv = _mm512_set1_pd(vr), viv = _mm512_set1_pd(vi);
        for (int g = 0; g < G; ++g) {
            Arv[g] = _mm512_fmadd_pd(urv, C[j*G + g], Arv[g]);
            Aiv[g] = _mm512_fmadd_pd(uiv, C[j*G + g], Aiv[g]);
            Brv[g] = _mm512_fmadd_pd(vrv, S[j*G + g], Brv[g]);
            Biv[g] = _mm512_fmadd_pd(viv, S[j*G + g], Biv[g]);
        }
    }
    qr[0] = sumr; qi[0] = sumi;
    double ar[16] __attribute__((aligned(64))), ai[16] __attribute__((aligned(64)));
    double br[16] __attribute__((aligned(64))), bi[16] __attribute__((aligned(64)));
    for (int g = 0; g < G; ++g) {
        _mm512_store_pd(ar + 8*g, Arv[g]); _mm512_store_pd(ai + 8*g, Aiv[g]);
        _mm512_store_pd(br + 8*g, Brv[g]); _mm512_store_pd(bi + 8*g, Biv[g]);
    }
    for (int k = 1; k <= H; ++k) {
        qr[k]     = s0r + ar[k-1] + bi[k-1];
        qi[k]     = s0i + ai[k-1] - br[k-1];
        qr[P - k] = s0r + ar[k-1] - bi[k-1];
        qi[P - k] = s0i + ai[k-1] + br[k-1];
    }
}

/* dispatch: 1 if a fast single-transform kernel exists for this L */
static int fs_one(const fft1d_plan *p, const double *re, const double *im,
                  double *qr, double *qi)
{
    switch (p->L) {
    case 13:  dsk8(13, 6, 1, p->twv, p->twv + 6*8, re, im, qr, qi);   return 1;
    case 31:  dsk8(31, 15, 2, p->twv, p->twv + 15*2*8, re, im, qr, qi); return 1;
    case 32:  fs32(p, re, im, qr, qi);  return 1;
    case 60:  fs60(p, re, im, qr, qi);  return 1;
    case 64:  fs64(p, re, im, qr, qi);  return 1;
    case 128: fs128(p, re, im, qr, qi); return 1;
    default: return 0;
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
    for (; jb < L; ++jb)
        for (int i = 0; i < 8; ++i) {
            xr[jb][i] = base[(size_t)i*2*L + 2*jb];
            xi[jb][i] = base[(size_t)i*2*L + 2*jb + 1];
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
    for (; jb < L; ++jb)
        for (int i = 0; i < 8; ++i) {
            base[(size_t)i*2*L + 2*jb]     = yr[jb][i];
            base[(size_t)i*2*L + 2*jb + 1] = yi[jb][i];
        }
}

/* ============ the chain map, on contiguous split-complex arrays ====================
 * Explicit 512-bit intrinsics: gcc -march=native auto-vectorizes this loop at ymm width
 * only (256-bit preference), which halves sqrt/div throughput -- and the map is the
 * per-step floor of every chained cell. All pointers are 64B-aligned scratch. */
static void map_apply(const double *restrict zr, const double *restrict zi,
                      const double *restrict cr, const double *restrict ci,
                      double *restrict sr, double *restrict si, int n)
{
    const __m512d one = _mm512_set1_pd(1.0);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d re = _mm512_add_pd(_mm512_load_pd(zr + i), _mm512_load_pd(cr + i));
        __m512d im = _mm512_add_pd(_mm512_load_pd(zi + i), _mm512_load_pd(ci + i));
        __m512d h  = _mm512_fmadd_pd(re, re, _mm512_mul_pd(im, im));
        __m512d sc = _mm512_div_pd(one, _mm512_add_pd(one, _mm512_sqrt_pd(h)));
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
        __m512d sc = _mm512_div_pd(one, _mm512_add_pd(one, _mm512_sqrt_pd(h)));
        _mm512_mask_store_pd(sr + i, mk, _mm512_mul_pd(re, sc));
        _mm512_mask_store_pd(si + i, mk, _mm512_mul_pd(im, sc));
    }
}

/* =============================== API ============================================= */
const char *fft1d_name(void) { return "d1_batchlane"; }
const char *fft1d_description(void)
{
    return "SoA 8-lane zmm batch-lane engine: split-complex across-batch kernels "
           "(densesym 13/31, CT 32/64/128, PFA 60), L1-blocked fused FFT+map chain";
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
                double ph = 2.0 * M_PI * ((double)((j * k) % L)) / (double)L;
                C[(j-1)*H + (k-1)] = cos(ph);
                S[(j-1)*H + (k-1)] = sin(ph);
            }
    } else if (L == 32 || L == 64) {
        int R = (L == 32) ? 4 : 8;             /* outer radix over n1 */
        p->tw = malloc((size_t)2 * R * 8 * sizeof(double));
        if (!p->tw) { fft1d_destroy(p); return NULL; }
        for (int n1 = 0; n1 < R; ++n1)
            for (int k2 = 0; k2 < 8; ++k2) {
                double ph = 2.0 * M_PI * ((double)((n1 * k2) % L)) / (double)L;
                p->tw[2*(n1*8 + k2)]     = cos(ph);
                p->tw[2*(n1*8 + k2) + 1] = -sin(ph);
            }
    } else if (L == 128) {
        /* [0..127]  : FFT64's W64^{n1 k2} table (n1,k2 in 0..7)
           [128..255]: W128^k for k = 0..63
           [256..273]: FFT16's W16^{n1 k2} table (n1,k2 in 1..3) */
        p->tw = malloc((size_t)(128 + 128 + 18) * sizeof(double));
        if (!p->tw) { fft1d_destroy(p); return NULL; }
        for (int n1 = 0; n1 < 8; ++n1)
            for (int k2 = 0; k2 < 8; ++k2) {
                double ph = 2.0 * M_PI * ((double)((n1 * k2) % 64)) / 64.0;
                p->tw[2*(n1*8 + k2)]     = cos(ph);
                p->tw[2*(n1*8 + k2) + 1] = -sin(ph);
            }
        for (int k = 0; k < 64; ++k) {
            double ph = 2.0 * M_PI * (double)k / 128.0;
            p->tw[128 + 2*k]     = cos(ph);
            p->tw[128 + 2*k + 1] = -sin(ph);
        }
        for (int n1 = 1; n1 < 4; ++n1)
            for (int k2 = 1; k2 < 4; ++k2) {
                double ph = 2.0 * M_PI * ((double)((n1 * k2) % 16)) / 16.0;
                p->tw[256 + 2*((n1-1)*3 + (k2-1))]     = cos(ph);
                p->tw[256 + 2*((n1-1)*3 + (k2-1)) + 1] = -sin(ph);
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
                        double ph = 2.0 * M_PI * ((double)(((j+1) * k) % L)) / (double)L;
                        c = cos(ph); s = sin(ph);
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
                double ph = 2.0 * M_PI * ((double)((n1 * k2) % L)) / (double)L;
                tre[(k2-1)*lanes + n1] = cos(ph);
                tim[(k2-1)*lanes + n1] = -sin(ph);
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
        dofft_v8(p, p->xr, p->xi, p->yr, p->yi, p->wr, p->wi);
        tstore8(out + (size_t)b * L, L, p->yr, p->yi);
    }
    for (; b < p->batch; ++b) {
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
    for (int k = 1; k <= H; ++k) {
        V ar = s0r, ai = s0i, br = VZ, bi = VZ;
        const double *Cr = C + (0)*H + (k-1), *Sr = S + (0)*H + (k-1);
        for (int j = 0; j < H; ++j) {
            double cc = Cr[j*H], ss = Sr[j*H];
            ar += cc * ur[j]; ai += cc * ui[j];
            br += ss * vr[j]; bi += ss * vi[j];
        }
        yr[k]   = ar + bi; yi[k]   = ai - br;
        yr[P-k] = ar - bi; yi[P-k] = ai + br;
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
    for (int i = 0; i < 60; ++i) { wr[i] = xr[im[i]]; wi[i] = xi[im[i]]; }
    for (int j = 0; j < 20; ++j)
        KN(fft3)(wr + j, wi + j, 20);
    for (int n3 = 0; n3 < 3; ++n3)
        for (int n5 = 0; n5 < 5; ++n5)
            KN(fft4)(wr + n3*20 + n5, wi + n3*20 + n5, 5,
                     wr + n3*20 + n5, wi + n3*20 + n5, 5);
    for (int t = 0; t < 12; ++t)
        KN(fft5)(wr + 5*t, wi + 5*t, 1);
    for (int i = 0; i < 60; ++i) { yr[om[i]] = wr[i]; yi[om[i]] = wi[i]; }
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
