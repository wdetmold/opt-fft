// Iterated batched 3D complex FFT engine, specialized for L in {6,8,13,17,23,36,45,64}.
// Planar (split re/im) state, AVX-512 vector lanes across the contiguous axis,
// per-volume cache-resident iteration, nonlinear map fused into the last pass.
// All transform arithmetic is our own (no FFT library anywhere).

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef double v8 __attribute__((vector_size(64), may_alias, aligned(64)));

#define VL(p)    (*(const v8 *)(p))
#define VS(p, x) (*(v8 *)(p) = (x))
static inline v8 vbc(double x) { return (v8){x, x, x, x, x, x, x, x}; }
static inline v8 vsqrt8(v8 x) { return (v8)_mm512_sqrt_pd((__m512d)x); }

// ---------------------------------------------------------------- constants
static double K_HALF = 0.5;
static double K_S3;            // sin(2pi/3) = sqrt(3)/2
static double K_R2;            // sqrt(2)/2
static double K_C51, K_C52, K_S51, K_S52;   // cos/sin 2pi/5, 4pi/5
static double K_W9R[5], K_W9I[5];           // w9^k, k=1,2,4 used
// CT twiddles  tw[n1*N2+k2] = w_N^(n1*k2)
static double TW36R[36], TW36I[36];
static double TW45R[45], TW45I[45];
static double TW64R[64], TW64I[64];
// prime tables: C[k-1][j-1] = cos(2pi j k / p), rows padded to PH
#define PH13 8
#define PH17 8
#define PH23 12
static double C13[PH13 * 6], S13[PH13 * 6];
static double C17[PH17 * 8], S17[PH17 * 8];
static double C23[PH23 * 11], S23[PH23 * 11];

// ---------------------------------------------------------------- scratch
static double SZA[64 * 8] __attribute__((aligned(64)));   // tile scratch re
static double SZB[64 * 8] __attribute__((aligned(64)));   // tile scratch im
static double T1R[64 * 8] __attribute__((aligned(64)));   // CT stage scratch
static double T1I[64 * 8] __attribute__((aligned(64)));
static double SAR[12 * 8] __attribute__((aligned(64)));   // prime a_j re
static double SAI[12 * 8] __attribute__((aligned(64)));
static double SBR[12 * 8] __attribute__((aligned(64)));
static double SBI[12 * 8] __attribute__((aligned(64)));

static double *Xre, *Xim, *Cre, *Cim;   // planar volume buffers (max size)

// ---------------------------------------------------------------- 8x8 double transpose
#define TR8(a0, a1, a2, a3, a4, a5, a6, a7)                                             \
  {                                                                                     \
    __m512d _t0 = _mm512_unpacklo_pd((__m512d)(a0), (__m512d)(a1));                     \
    __m512d _t1 = _mm512_unpackhi_pd((__m512d)(a0), (__m512d)(a1));                     \
    __m512d _t2 = _mm512_unpacklo_pd((__m512d)(a2), (__m512d)(a3));                     \
    __m512d _t3 = _mm512_unpackhi_pd((__m512d)(a2), (__m512d)(a3));                     \
    __m512d _t4 = _mm512_unpacklo_pd((__m512d)(a4), (__m512d)(a5));                     \
    __m512d _t5 = _mm512_unpackhi_pd((__m512d)(a4), (__m512d)(a5));                     \
    __m512d _t6 = _mm512_unpacklo_pd((__m512d)(a6), (__m512d)(a7));                     \
    __m512d _t7 = _mm512_unpackhi_pd((__m512d)(a6), (__m512d)(a7));                     \
    __m512d _u0 = _mm512_shuffle_f64x2(_t0, _t2, 0x88);                                 \
    __m512d _u1 = _mm512_shuffle_f64x2(_t1, _t3, 0x88);                                 \
    __m512d _u2 = _mm512_shuffle_f64x2(_t0, _t2, 0xDD);                                 \
    __m512d _u3 = _mm512_shuffle_f64x2(_t1, _t3, 0xDD);                                 \
    __m512d _u4 = _mm512_shuffle_f64x2(_t4, _t6, 0x88);                                 \
    __m512d _u5 = _mm512_shuffle_f64x2(_t5, _t7, 0x88);                                 \
    __m512d _u6 = _mm512_shuffle_f64x2(_t4, _t6, 0xDD);                                 \
    __m512d _u7 = _mm512_shuffle_f64x2(_t5, _t7, 0xDD);                                 \
    (a0) = (v8)_mm512_shuffle_f64x2(_u0, _u4, 0x88);                                    \
    (a1) = (v8)_mm512_shuffle_f64x2(_u1, _u5, 0x88);                                    \
    (a2) = (v8)_mm512_shuffle_f64x2(_u2, _u6, 0x88);                                    \
    (a3) = (v8)_mm512_shuffle_f64x2(_u3, _u7, 0x88);                                    \
    (a4) = (v8)_mm512_shuffle_f64x2(_u0, _u4, 0xDD);                                    \
    (a5) = (v8)_mm512_shuffle_f64x2(_u1, _u5, 0xDD);                                    \
    (a6) = (v8)_mm512_shuffle_f64x2(_u2, _u6, 0xDD);                                    \
    (a7) = (v8)_mm512_shuffle_f64x2(_u3, _u7, 0xDD);                                    \
  }

// load an 8-row x LP-col tile (row stride ld doubles) transposed into scratch s:
// s slot c (c < LP) holds column c across the 8 rows.
static inline __attribute__((always_inline)) void tile_load(const double *b, long ld,
                                                            double *s, int ncb) {
  for (int cb = 0; cb < ncb; cb++) {
    v8 r0 = VL(b + 0 * ld + cb * 8), r1 = VL(b + 1 * ld + cb * 8);
    v8 r2 = VL(b + 2 * ld + cb * 8), r3 = VL(b + 3 * ld + cb * 8);
    v8 r4 = VL(b + 4 * ld + cb * 8), r5 = VL(b + 5 * ld + cb * 8);
    v8 r6 = VL(b + 6 * ld + cb * 8), r7 = VL(b + 7 * ld + cb * 8);
    TR8(r0, r1, r2, r3, r4, r5, r6, r7);
    VS(s + (cb * 8 + 0) * 8, r0); VS(s + (cb * 8 + 1) * 8, r1);
    VS(s + (cb * 8 + 2) * 8, r2); VS(s + (cb * 8 + 3) * 8, r3);
    VS(s + (cb * 8 + 4) * 8, r4); VS(s + (cb * 8 + 5) * 8, r5);
    VS(s + (cb * 8 + 6) * 8, r6); VS(s + (cb * 8 + 7) * 8, r7);
  }
}

// store scratch back: columns >= L get junk (slot 0) - harmless padding.
static inline __attribute__((always_inline)) void tile_store(double *b, long ld,
                                                             const double *s, int ncb, int L) {
  for (int cb = 0; cb < ncb; cb++) {
    v8 c0 = VL(s + ((cb * 8 + 0) < L ? (cb * 8 + 0) : 0) * 8);
    v8 c1 = VL(s + ((cb * 8 + 1) < L ? (cb * 8 + 1) : 0) * 8);
    v8 c2 = VL(s + ((cb * 8 + 2) < L ? (cb * 8 + 2) : 0) * 8);
    v8 c3 = VL(s + ((cb * 8 + 3) < L ? (cb * 8 + 3) : 0) * 8);
    v8 c4 = VL(s + ((cb * 8 + 4) < L ? (cb * 8 + 4) : 0) * 8);
    v8 c5 = VL(s + ((cb * 8 + 5) < L ? (cb * 8 + 5) : 0) * 8);
    v8 c6 = VL(s + ((cb * 8 + 6) < L ? (cb * 8 + 6) : 0) * 8);
    v8 c7 = VL(s + ((cb * 8 + 7) < L ? (cb * 8 + 7) : 0) * 8);
    TR8(c0, c1, c2, c3, c4, c5, c6, c7);
    VS(b + 0 * ld + cb * 8, c0); VS(b + 1 * ld + cb * 8, c1);
    VS(b + 2 * ld + cb * 8, c2); VS(b + 3 * ld + cb * 8, c3);
    VS(b + 4 * ld + cb * 8, c4); VS(b + 5 * ld + cb * 8, c5);
    VS(b + 6 * ld + cb * 8, c6); VS(b + 7 * ld + cb * 8, c7);
  }
}

// ---------------------------------------------------------------- map store
#define STORE_PLAIN(pr, pi, vr, vi) { VS((pr), (vr)); VS((pi), (vi)); }
#define STORE_MAP(pr, pi, vr, vi, pcr, pci)                    \
  {                                                            \
    v8 _zr = (vr) + VL(pcr), _zi = (vi) + VL(pci);             \
    v8 _mg = vsqrt8(_zr * _zr + _zi * _zi);                    \
    v8 _sc = vone / (vone + _mg);                              \
    VS((pr), _zr * _sc); VS((pi), _zi * _sc);                  \
  }
#define STORE_X(pr, pi, vr, vi, off)                           \
  {                                                            \
    if (domap) { STORE_MAP((pr) + (off), (pi) + (off), vr, vi, cre + (off), cim + (off)) } \
    else       { STORE_PLAIN((pr) + (off), (pi) + (off), vr, vi) }                          \
  }

// ---------------------------------------------------------------- codelet cores
// complex var = pair of v8 named <x>r, <x>i ; macros operate on base names.

#define FFT3_CORE(x0, x1, x2)                                 \
  {                                                           \
    v8 _tr = x1##r + x2##r, _ti = x1##i + x2##i;              \
    v8 _dr = x1##r - x2##r, _di = x1##i - x2##i;              \
    v8 _mr = x0##r - vhalf * _tr, _mi = x0##i - vhalf * _ti;  \
    x0##r += _tr; x0##i += _ti;                               \
    v8 _sr = vs3 * _dr, _si = vs3 * _di;                      \
    x1##r = _mr + _si; x1##i = _mi - _sr;                     \
    x2##r = _mr - _si; x2##i = _mi + _sr;                     \
  }

#define FFT5_CORE(x0, x1, x2, x3, x4)                                      \
  {                                                                        \
    v8 _t1r = x1##r + x4##r, _t1i = x1##i + x4##i;                         \
    v8 _d1r = x1##r - x4##r, _d1i = x1##i - x4##i;                         \
    v8 _t2r = x2##r + x3##r, _t2i = x2##i + x3##i;                         \
    v8 _d2r = x2##r - x3##r, _d2i = x2##i - x3##i;                         \
    v8 _u1r = x0##r + vc51 * _t1r + vc52 * _t2r;                           \
    v8 _u1i = x0##i + vc51 * _t1i + vc52 * _t2i;                           \
    v8 _u2r = x0##r + vc52 * _t1r + vc51 * _t2r;                           \
    v8 _u2i = x0##i + vc52 * _t1i + vc51 * _t2i;                           \
    x0##r += _t1r + _t2r; x0##i += _t1i + _t2i;                            \
    v8 _v1r = vs51 * _d1r + vs52 * _d2r, _v1i = vs51 * _d1i + vs52 * _d2i; \
    v8 _v2r = vs52 * _d1r - vs51 * _d2r, _v2i = vs52 * _d1i - vs51 * _d2i; \
    x1##r = _u1r + _v1i; x1##i = _u1i - _v1r;                              \
    x4##r = _u1r - _v1i; x4##i = _u1i + _v1r;                              \
    x2##r = _u2r + _v2i; x2##i = _u2i - _v2r;                              \
    x3##r = _u2r - _v2i; x3##i = _u2i + _v2r;                              \
  }

// FFT6: outputs in natural order
#define FFT6_CORE(x0, x1, x2, x3, x4, x5)                      \
  {                                                            \
    v8 g0r = x0##r, g0i = x0##i, g1r = x2##r, g1i = x2##i, g2r = x4##r, g2i = x4##i;  \
    v8 h0r = x1##r, h0i = x1##i, h1r = x3##r, h1i = x3##i, h2r = x5##r, h2i = x5##i;  \
    FFT3_CORE(g0, g1, g2);                                     \
    FFT3_CORE(h0, h1, h2);                                     \
    x0##r = g0r + h0r; x0##i = g0i + h0i;                      \
    x3##r = g0r - h0r; x3##i = g0i - h0i;                      \
    v8 _w1r = vhalf * h1r + vs3 * h1i, _w1i = vhalf * h1i - vs3 * h1r;   \
    x1##r = g1r + _w1r; x1##i = g1i + _w1i;                    \
    x4##r = g1r - _w1r; x4##i = g1i - _w1i;                    \
    v8 _w2r = vs3 * h2i - vhalf * h2r, _w2i = -(vhalf * h2i + vs3 * h2r); \
    x2##r = g2r + _w2r; x2##i = g2i + _w2i;                    \
    x5##r = g2r - _w2r; x5##i = g2i - _w2i;                    \
  }

// FFT8: outputs in natural order
#define FFT8_CORE(x0, x1, x2, x3, x4, x5, x6, x7)              \
  {                                                            \
    v8 a0r = x0##r + x4##r, a0i = x0##i + x4##i, b0r = x0##r - x4##r, b0i = x0##i - x4##i; \
    v8 a1r = x1##r + x5##r, a1i = x1##i + x5##i, b1r = x1##r - x5##r, b1i = x1##i - x5##i; \
    v8 a2r = x2##r + x6##r, a2i = x2##i + x6##i, b2r = x2##r - x6##r, b2i = x2##i - x6##i; \
    v8 a3r = x3##r + x7##r, a3i = x3##i + x7##i, b3r = x3##r - x7##r, b3i = x3##i - x7##i; \
    v8 u0r = a0r + a2r, u0i = a0i + a2i, u1r = a0r - a2r, u1i = a0i - a2i; \
    v8 u2r = a1r + a3r, u2i = a1i + a3i, u3r = a1r - a3r, u3i = a1i - a3i; \
    x0##r = u0r + u2r; x0##i = u0i + u2i;                      \
    x4##r = u0r - u2r; x4##i = u0i - u2i;                      \
    x2##r = u1r + u3i; x2##i = u1i - u3r;                      \
    x6##r = u1r - u3i; x6##i = u1i + u3r;                      \
    v8 c1r = (b1r + b1i) * vr2, c1i = (b1i - b1r) * vr2;       \
    v8 c2r = b2i, c2i = -b2r;                                  \
    v8 c3r = (b3i - b3r) * vr2, c3i = -(b3r + b3i) * vr2;      \
    v8 v0r = b0r + c2r, v0i = b0i + c2i, v1r = b0r - c2r, v1i = b0i - c2i; \
    v8 v2r = c1r + c3r, v2i = c1i + c3i, v3r = c1r - c3r, v3i = c1i - c3i; \
    x1##r = v0r + v2r; x1##i = v0i + v2i;                      \
    x5##r = v0r - v2r; x5##i = v0i - v2i;                      \
    x3##r = v1r + v3i; x3##i = v1i - v3r;                      \
    x7##r = v1r - v3i; x7##i = v1i + v3r;                      \
  }

// FFT9 on vars y0..y8; OUTPUT PERMUTED: X[k] lives in var P[k], P = {0,3,6,1,4,7,2,5,8}
#define FFT9_CORE(y0, y1, y2, y3, y4, y5, y6, y7, y8)         \
  {                                                           \
    FFT3_CORE(y0, y3, y6);                                    \
    FFT3_CORE(y1, y4, y7);                                    \
    FFT3_CORE(y2, y5, y8);                                    \
    CMULT(y4##r, y4##i, y4##r, y4##i, vw91r, vw91i);          \
    CMULT(y7##r, y7##i, y7##r, y7##i, vw92r, vw92i);          \
    CMULT(y5##r, y5##i, y5##r, y5##i, vw92r, vw92i);          \
    CMULT(y8##r, y8##i, y8##r, y8##i, vw94r, vw94i);          \
    FFT3_CORE(y0, y1, y2);                                    \
    FFT3_CORE(y3, y4, y5);                                    \
    FFT3_CORE(y6, y7, y8);                                    \
  }

#define CMULT(dr, di, ar, ai, wr, wi)            \
  {                                              \
    v8 _cr = (ar) * (wr) - (ai) * (wi);          \
    v8 _ci = (ar) * (wi) + (ai) * (wr);          \
    (dr) = _cr; (di) = _ci;                      \
  }

#define DEFCONSTS                                  \
  const v8 vhalf = vbc(K_HALF), vs3 = vbc(K_S3), vr2 = vbc(K_R2);     \
  const v8 vc51 = vbc(K_C51), vc52 = vbc(K_C52), vs51 = vbc(K_S51), vs52 = vbc(K_S52); \
  const v8 vw91r = vbc(K_W9R[1]), vw91i = vbc(K_W9I[1]);              \
  const v8 vw92r = vbc(K_W9R[2]), vw92i = vbc(K_W9I[2]);              \
  const v8 vw94r = vbc(K_W9R[4]), vw94i = vbc(K_W9I[4]);              \
  const v8 vone = vbc(1.0), vzero = vbc(0.0);                         \
  (void)vhalf; (void)vs3; (void)vr2; (void)vc51; (void)vc52;          \
  (void)vs51; (void)vs52; (void)vw91r; (void)vw91i; (void)vw92r;      \
  (void)vw92i; (void)vw94r; (void)vw94i; (void)vone; (void)vzero;

// ---------------------------------------------------------------- line FFTs
// element j of the line lives at re[j*s], im[j*s]; s is a compile-time constant
// at every call site.  domap: fuse z = X + c ; x = z/(1+|z|) into the store.

static inline __attribute__((always_inline)) void
fft6_line(double *re, double *im, const long s, const int domap,
          const double *cre, const double *cim) {
  DEFCONSTS
  v8 x0r = VL(re + 0 * s), x0i = VL(im + 0 * s);
  v8 x1r = VL(re + 1 * s), x1i = VL(im + 1 * s);
  v8 x2r = VL(re + 2 * s), x2i = VL(im + 2 * s);
  v8 x3r = VL(re + 3 * s), x3i = VL(im + 3 * s);
  v8 x4r = VL(re + 4 * s), x4i = VL(im + 4 * s);
  v8 x5r = VL(re + 5 * s), x5i = VL(im + 5 * s);
  FFT6_CORE(x0, x1, x2, x3, x4, x5);
  STORE_X(re, im, x0r, x0i, 0 * s);
  STORE_X(re, im, x1r, x1i, 1 * s);
  STORE_X(re, im, x2r, x2i, 2 * s);
  STORE_X(re, im, x3r, x3i, 3 * s);
  STORE_X(re, im, x4r, x4i, 4 * s);
  STORE_X(re, im, x5r, x5i, 5 * s);
}

static inline __attribute__((always_inline)) void
fft8_line(double *re, double *im, const long s, const int domap,
          const double *cre, const double *cim) {
  DEFCONSTS
  v8 x0r = VL(re + 0 * s), x0i = VL(im + 0 * s);
  v8 x1r = VL(re + 1 * s), x1i = VL(im + 1 * s);
  v8 x2r = VL(re + 2 * s), x2i = VL(im + 2 * s);
  v8 x3r = VL(re + 3 * s), x3i = VL(im + 3 * s);
  v8 x4r = VL(re + 4 * s), x4i = VL(im + 4 * s);
  v8 x5r = VL(re + 5 * s), x5i = VL(im + 5 * s);
  v8 x6r = VL(re + 6 * s), x6i = VL(im + 6 * s);
  v8 x7r = VL(re + 7 * s), x7i = VL(im + 7 * s);
  FFT8_CORE(x0, x1, x2, x3, x4, x5, x6, x7);
  STORE_X(re, im, x0r, x0i, 0 * s);
  STORE_X(re, im, x1r, x1i, 1 * s);
  STORE_X(re, im, x2r, x2i, 2 * s);
  STORE_X(re, im, x3r, x3i, 3 * s);
  STORE_X(re, im, x4r, x4i, 4 * s);
  STORE_X(re, im, x5r, x5i, 5 * s);
  STORE_X(re, im, x6r, x6i, 6 * s);
  STORE_X(re, im, x7r, x7i, 7 * s);
}

// generic prime codelet body (P = 2H+1), BLK in {3,4}
#define PRIME_LINE(P, H, PHN, CT, ST, BLK)                                        \
  DEFCONSTS                                                                       \
  v8 x0r = VL(re), x0i = VL(im);                                                  \
  v8 sumr = x0r, sumi = x0i;                                                      \
  _Pragma("GCC unroll 16") for (int j = 1; j <= H; j++) {                         \
    v8 ar = VL(re + j * s), ai = VL(im + j * s);                                  \
    v8 br = VL(re + (P - j) * s), bi = VL(im + (P - j) * s);                      \
    v8 par = ar + br, pai = ai + bi, pbr = ar - br, pbi = ai - bi;                \
    VS(SAR + (j - 1) * 8, par); VS(SAI + (j - 1) * 8, pai);                       \
    VS(SBR + (j - 1) * 8, pbr); VS(SBI + (j - 1) * 8, pbi);                       \
    sumr += par; sumi += pai;                                                     \
  }                                                                               \
  STORE_X(re, im, sumr, sumi, 0);                                                 \
  for (int k0 = 1; k0 <= H; k0 += BLK) {                                          \
    v8 Crr[BLK], Cii[BLK], Srr[BLK], Sii[BLK];                                    \
    _Pragma("GCC unroll 8") for (int t = 0; t < BLK; t++) {                       \
      Crr[t] = x0r; Cii[t] = x0i; Srr[t] = vzero; Sii[t] = vzero;                 \
    }                                                                             \
    _Pragma("GCC unroll 16") for (int j = 0; j < H; j++) {                        \
      v8 par = VL(SAR + j * 8), pai = VL(SAI + j * 8);                            \
      v8 pbr = VL(SBR + j * 8), pbi = VL(SBI + j * 8);                            \
      _Pragma("GCC unroll 8") for (int t = 0; t < BLK; t++) {                     \
        v8 ck = vbc(CT[(k0 - 1 + t) * H + j]);                                    \
        v8 sk = vbc(ST[(k0 - 1 + t) * H + j]);                                    \
        Crr[t] += ck * par; Cii[t] += ck * pai;                                   \
        Srr[t] += sk * pbr; Sii[t] += sk * pbi;                                   \
      }                                                                           \
    }                                                                             \
    int nk = H - k0 + 1; if (nk > BLK) nk = BLK;                                  \
    _Pragma("GCC unroll 8") for (int t = 0; t < nk; t++) {                        \
      long k = k0 + t;                                                            \
      v8 xr1 = Crr[t] + Sii[t], xi1 = Cii[t] - Srr[t];                            \
      v8 xr2 = Crr[t] - Sii[t], xi2 = Cii[t] + Srr[t];                            \
      STORE_X(re, im, xr1, xi1, k * s);                                           \
      STORE_X(re, im, xr2, xi2, (P - k) * s);                                     \
    }                                                                             \
  }

static inline __attribute__((always_inline)) void
fft13_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim) {
  PRIME_LINE(13, 6, PH13, C13, S13, 3)
}
static inline __attribute__((always_inline)) void
fft17_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim) {
  PRIME_LINE(17, 8, PH17, C17, S17, 4)
}
static inline __attribute__((always_inline)) void
fft23_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim) {
  PRIME_LINE(23, 11, PH23, C23, S23, 4)
}

// 36 = 6 x 6
static inline __attribute__((always_inline)) void
fft36_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim) {
  DEFCONSTS
  _Pragma("GCC unroll 6") for (int n1 = 0; n1 < 6; n1++) {
    v8 x0r = VL(re + (n1 + 0) * s), x0i = VL(im + (n1 + 0) * s);
    v8 x1r = VL(re + (n1 + 6) * s), x1i = VL(im + (n1 + 6) * s);
    v8 x2r = VL(re + (n1 + 12) * s), x2i = VL(im + (n1 + 12) * s);
    v8 x3r = VL(re + (n1 + 18) * s), x3i = VL(im + (n1 + 18) * s);
    v8 x4r = VL(re + (n1 + 24) * s), x4i = VL(im + (n1 + 24) * s);
    v8 x5r = VL(re + (n1 + 30) * s), x5i = VL(im + (n1 + 30) * s);
    FFT6_CORE(x0, x1, x2, x3, x4, x5);
    if (n1) {
      CMULT(x1r, x1i, x1r, x1i, vbc(TW36R[n1 * 6 + 1]), vbc(TW36I[n1 * 6 + 1]));
      CMULT(x2r, x2i, x2r, x2i, vbc(TW36R[n1 * 6 + 2]), vbc(TW36I[n1 * 6 + 2]));
      CMULT(x3r, x3i, x3r, x3i, vbc(TW36R[n1 * 6 + 3]), vbc(TW36I[n1 * 6 + 3]));
      CMULT(x4r, x4i, x4r, x4i, vbc(TW36R[n1 * 6 + 4]), vbc(TW36I[n1 * 6 + 4]));
      CMULT(x5r, x5i, x5r, x5i, vbc(TW36R[n1 * 6 + 5]), vbc(TW36I[n1 * 6 + 5]));
    }
    VS(T1R + (0 * 6 + n1) * 8, x0r); VS(T1I + (0 * 6 + n1) * 8, x0i);
    VS(T1R + (1 * 6 + n1) * 8, x1r); VS(T1I + (1 * 6 + n1) * 8, x1i);
    VS(T1R + (2 * 6 + n1) * 8, x2r); VS(T1I + (2 * 6 + n1) * 8, x2i);
    VS(T1R + (3 * 6 + n1) * 8, x3r); VS(T1I + (3 * 6 + n1) * 8, x3i);
    VS(T1R + (4 * 6 + n1) * 8, x4r); VS(T1I + (4 * 6 + n1) * 8, x4i);
    VS(T1R + (5 * 6 + n1) * 8, x5r); VS(T1I + (5 * 6 + n1) * 8, x5i);
  }
  _Pragma("GCC unroll 6") for (int k2 = 0; k2 < 6; k2++) {
    v8 x0r = VL(T1R + (k2 * 6 + 0) * 8), x0i = VL(T1I + (k2 * 6 + 0) * 8);
    v8 x1r = VL(T1R + (k2 * 6 + 1) * 8), x1i = VL(T1I + (k2 * 6 + 1) * 8);
    v8 x2r = VL(T1R + (k2 * 6 + 2) * 8), x2i = VL(T1I + (k2 * 6 + 2) * 8);
    v8 x3r = VL(T1R + (k2 * 6 + 3) * 8), x3i = VL(T1I + (k2 * 6 + 3) * 8);
    v8 x4r = VL(T1R + (k2 * 6 + 4) * 8), x4i = VL(T1I + (k2 * 6 + 4) * 8);
    v8 x5r = VL(T1R + (k2 * 6 + 5) * 8), x5i = VL(T1I + (k2 * 6 + 5) * 8);
    FFT6_CORE(x0, x1, x2, x3, x4, x5);
    STORE_X(re, im, x0r, x0i, (0 * 6 + k2) * s);
    STORE_X(re, im, x1r, x1i, (1 * 6 + k2) * s);
    STORE_X(re, im, x2r, x2i, (2 * 6 + k2) * s);
    STORE_X(re, im, x3r, x3i, (3 * 6 + k2) * s);
    STORE_X(re, im, x4r, x4i, (4 * 6 + k2) * s);
    STORE_X(re, im, x5r, x5i, (5 * 6 + k2) * s);
  }
}

// 45 = 5 x 9 : n = 5*n2 + n1 ; stage1: FFT9 per n1 (stride 5), twiddle w45^(n1*k2),
// store T[k2*5+n1]; stage2: FFT5 per k2 -> out[9*k1 + k2]
static inline __attribute__((always_inline)) void
fft45_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim) {
  DEFCONSTS
  _Pragma("GCC unroll 5") for (int n1 = 0; n1 < 5; n1++) {
    v8 y0r = VL(re + (n1 + 0) * s), y0i = VL(im + (n1 + 0) * s);
    v8 y1r = VL(re + (n1 + 5) * s), y1i = VL(im + (n1 + 5) * s);
    v8 y2r = VL(re + (n1 + 10) * s), y2i = VL(im + (n1 + 10) * s);
    v8 y3r = VL(re + (n1 + 15) * s), y3i = VL(im + (n1 + 15) * s);
    v8 y4r = VL(re + (n1 + 20) * s), y4i = VL(im + (n1 + 20) * s);
    v8 y5r = VL(re + (n1 + 25) * s), y5i = VL(im + (n1 + 25) * s);
    v8 y6r = VL(re + (n1 + 30) * s), y6i = VL(im + (n1 + 30) * s);
    v8 y7r = VL(re + (n1 + 35) * s), y7i = VL(im + (n1 + 35) * s);
    v8 y8r = VL(re + (n1 + 40) * s), y8i = VL(im + (n1 + 40) * s);
    FFT9_CORE(y0, y1, y2, y3, y4, y5, y6, y7, y8);
    // X[k] is in var P[k], P = {0,3,6,1,4,7,2,5,8}
#define TW45ST(k2, var)                                                              \
    {                                                                                \
      v8 _rr = var##r, _ri = var##i;                                                 \
      if (n1 && k2) CMULT(_rr, _ri, _rr, _ri, vbc(TW45R[n1 * 9 + k2]), vbc(TW45I[n1 * 9 + k2])); \
      VS(T1R + ((k2) * 5 + n1) * 8, _rr); VS(T1I + ((k2) * 5 + n1) * 8, _ri);        \
    }
    TW45ST(0, y0) TW45ST(1, y3) TW45ST(2, y6) TW45ST(3, y1) TW45ST(4, y4)
    TW45ST(5, y7) TW45ST(6, y2) TW45ST(7, y5) TW45ST(8, y8)
#undef TW45ST
  }
  _Pragma("GCC unroll 9") for (int k2 = 0; k2 < 9; k2++) {
    v8 x0r = VL(T1R + (k2 * 5 + 0) * 8), x0i = VL(T1I + (k2 * 5 + 0) * 8);
    v8 x1r = VL(T1R + (k2 * 5 + 1) * 8), x1i = VL(T1I + (k2 * 5 + 1) * 8);
    v8 x2r = VL(T1R + (k2 * 5 + 2) * 8), x2i = VL(T1I + (k2 * 5 + 2) * 8);
    v8 x3r = VL(T1R + (k2 * 5 + 3) * 8), x3i = VL(T1I + (k2 * 5 + 3) * 8);
    v8 x4r = VL(T1R + (k2 * 5 + 4) * 8), x4i = VL(T1I + (k2 * 5 + 4) * 8);
    FFT5_CORE(x0, x1, x2, x3, x4);
    STORE_X(re, im, x0r, x0i, (0 * 9 + k2) * s);
    STORE_X(re, im, x1r, x1i, (1 * 9 + k2) * s);
    STORE_X(re, im, x2r, x2i, (2 * 9 + k2) * s);
    STORE_X(re, im, x3r, x3i, (3 * 9 + k2) * s);
    STORE_X(re, im, x4r, x4i, (4 * 9 + k2) * s);
  }
}

// 64 = 8 x 8
static inline __attribute__((always_inline)) void
fft64_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim) {
  DEFCONSTS
  _Pragma("GCC unroll 8") for (int n1 = 0; n1 < 8; n1++) {
    v8 x0r = VL(re + (n1 + 0) * s), x0i = VL(im + (n1 + 0) * s);
    v8 x1r = VL(re + (n1 + 8) * s), x1i = VL(im + (n1 + 8) * s);
    v8 x2r = VL(re + (n1 + 16) * s), x2i = VL(im + (n1 + 16) * s);
    v8 x3r = VL(re + (n1 + 24) * s), x3i = VL(im + (n1 + 24) * s);
    v8 x4r = VL(re + (n1 + 32) * s), x4i = VL(im + (n1 + 32) * s);
    v8 x5r = VL(re + (n1 + 40) * s), x5i = VL(im + (n1 + 40) * s);
    v8 x6r = VL(re + (n1 + 48) * s), x6i = VL(im + (n1 + 48) * s);
    v8 x7r = VL(re + (n1 + 56) * s), x7i = VL(im + (n1 + 56) * s);
    FFT8_CORE(x0, x1, x2, x3, x4, x5, x6, x7);
    if (n1) {
      CMULT(x1r, x1i, x1r, x1i, vbc(TW64R[n1 * 8 + 1]), vbc(TW64I[n1 * 8 + 1]));
      CMULT(x2r, x2i, x2r, x2i, vbc(TW64R[n1 * 8 + 2]), vbc(TW64I[n1 * 8 + 2]));
      CMULT(x3r, x3i, x3r, x3i, vbc(TW64R[n1 * 8 + 3]), vbc(TW64I[n1 * 8 + 3]));
      CMULT(x4r, x4i, x4r, x4i, vbc(TW64R[n1 * 8 + 4]), vbc(TW64I[n1 * 8 + 4]));
      CMULT(x5r, x5i, x5r, x5i, vbc(TW64R[n1 * 8 + 5]), vbc(TW64I[n1 * 8 + 5]));
      CMULT(x6r, x6i, x6r, x6i, vbc(TW64R[n1 * 8 + 6]), vbc(TW64I[n1 * 8 + 6]));
      CMULT(x7r, x7i, x7r, x7i, vbc(TW64R[n1 * 8 + 7]), vbc(TW64I[n1 * 8 + 7]));
    }
    VS(T1R + (0 * 8 + n1) * 8, x0r); VS(T1I + (0 * 8 + n1) * 8, x0i);
    VS(T1R + (1 * 8 + n1) * 8, x1r); VS(T1I + (1 * 8 + n1) * 8, x1i);
    VS(T1R + (2 * 8 + n1) * 8, x2r); VS(T1I + (2 * 8 + n1) * 8, x2i);
    VS(T1R + (3 * 8 + n1) * 8, x3r); VS(T1I + (3 * 8 + n1) * 8, x3i);
    VS(T1R + (4 * 8 + n1) * 8, x4r); VS(T1I + (4 * 8 + n1) * 8, x4i);
    VS(T1R + (5 * 8 + n1) * 8, x5r); VS(T1I + (5 * 8 + n1) * 8, x5i);
    VS(T1R + (6 * 8 + n1) * 8, x6r); VS(T1I + (6 * 8 + n1) * 8, x6i);
    VS(T1R + (7 * 8 + n1) * 8, x7r); VS(T1I + (7 * 8 + n1) * 8, x7i);
  }
  _Pragma("GCC unroll 8") for (int k2 = 0; k2 < 8; k2++) {
    v8 x0r = VL(T1R + (k2 * 8 + 0) * 8), x0i = VL(T1I + (k2 * 8 + 0) * 8);
    v8 x1r = VL(T1R + (k2 * 8 + 1) * 8), x1i = VL(T1I + (k2 * 8 + 1) * 8);
    v8 x2r = VL(T1R + (k2 * 8 + 2) * 8), x2i = VL(T1I + (k2 * 8 + 2) * 8);
    v8 x3r = VL(T1R + (k2 * 8 + 3) * 8), x3i = VL(T1I + (k2 * 8 + 3) * 8);
    v8 x4r = VL(T1R + (k2 * 8 + 4) * 8), x4i = VL(T1I + (k2 * 8 + 4) * 8);
    v8 x5r = VL(T1R + (k2 * 8 + 5) * 8), x5i = VL(T1I + (k2 * 8 + 5) * 8);
    v8 x6r = VL(T1R + (k2 * 8 + 6) * 8), x6i = VL(T1I + (k2 * 8 + 6) * 8);
    v8 x7r = VL(T1R + (k2 * 8 + 7) * 8), x7i = VL(T1I + (k2 * 8 + 7) * 8);
    FFT8_CORE(x0, x1, x2, x3, x4, x5, x6, x7);
    STORE_X(re, im, x0r, x0i, (0 * 8 + k2) * s);
    STORE_X(re, im, x1r, x1i, (1 * 8 + k2) * s);
    STORE_X(re, im, x2r, x2i, (2 * 8 + k2) * s);
    STORE_X(re, im, x3r, x3i, (3 * 8 + k2) * s);
    STORE_X(re, im, x4r, x4i, (4 * 8 + k2) * s);
    STORE_X(re, im, x5r, x5i, (5 * 8 + k2) * s);
    STORE_X(re, im, x6r, x6i, (6 * 8 + k2) * s);
    STORE_X(re, im, x7r, x7i, (7 * 8 + k2) * s);
  }
}

// ---------------------------------------------------------------- engines
#define DEF_ENGINE(LN, L_, LP_, MP_, FFTLINE)                                         \
  static void sweepA_##LN(void) {                                                    \
    for (long x0 = 0; x0 < L_; x0 += 8) {                                             \
      long r0 = x0 * L_, r1 = (x0 + 8) * (long)L_;                                    \
      if (r1 > MP_) r1 = MP_;                                                         \
      for (long r = r0; r < r1; r += 8) {                                             \
        tile_load(Xre + r * LP_, LP_, SZA, LP_ / 8);                                  \
        tile_load(Xim + r * LP_, LP_, SZB, LP_ / 8);                                  \
        FFTLINE(SZA, SZB, 8, 0, 0, 0);                                                \
        tile_store(Xre + r * LP_, LP_, SZA, LP_ / 8, L_);                             \
        tile_store(Xim + r * LP_, LP_, SZB, LP_ / 8, L_);                             \
      }                                                                               \
      long xe = x0 + 8; if (xe > L_) xe = L_;                                         \
      for (long x = x0; x < xe; x++) {                                                \
        double *br = Xre + x * (long)(L_ * LP_), *bi = Xim + x * (long)(L_ * LP_);    \
        for (long zc = 0; zc < LP_; zc += 8) FFTLINE(br + zc, bi + zc, LP_, 0, 0, 0); \
      }                                                                               \
    }                                                                                 \
  }                                                                                   \
  static void sweepB_##LN(void) {                                                     \
    for (long y = 0; y < L_; y++) {                                                   \
      long base = y * LP_;                                                            \
      for (long zc = 0; zc < LP_; zc += 8) {                                          \
        long off = base + zc;                                                         \
        FFTLINE(Xre + off, Xim + off, (long)(L_ * LP_), 1, Cre + off, Cim + off);     \
      }                                                                               \
    }                                                                                 \
  }

DEF_ENGINE(6, 6, 8, 40, fft6_line)
DEF_ENGINE(8, 8, 8, 64, fft8_line)
DEF_ENGINE(13, 13, 16, 176, fft13_line)
DEF_ENGINE(17, 17, 24, 296, fft17_line)
DEF_ENGINE(23, 23, 24, 536, fft23_line)
DEF_ENGINE(36, 36, 40, 1296, fft36_line)
DEF_ENGINE(45, 45, 48, 2032, fft45_line)
DEF_ENGINE(64, 64, 64, 4096, fft64_line)

// ---------------------------------------------------------------- conversions
static void load_vol(long L, long Lp, const double *xin, double *dre, double *dim) {
  long n2 = L * L;
  for (long r = 0; r < n2; r++) {
    const double *p = xin + r * L * 2;
    double *qr = dre + r * Lp, *qi = dim + r * Lp;
    for (long z = 0; z < L; z++) { qr[z] = p[2 * z]; qi[z] = p[2 * z + 1]; }
  }
}
static void store_vol(long L, long Lp, const double *sre, const double *sim, double *xout) {
  long n2 = L * L;
  for (long r = 0; r < n2; r++) {
    double *p = xout + r * L * 2;
    const double *qr = sre + r * Lp, *qi = sim + r * Lp;
    for (long z = 0; z < L; z++) { p[2 * z] = qr[z]; p[2 * z + 1] = qi[z]; }
  }
}

// ---------------------------------------------------------------- init
static void fill_ct(double *tr, double *ti, int N1, int N2, int N) {
  const long double PI2 = 6.283185307179586476925286766559005768L;
  for (int n1 = 0; n1 < N1; n1++)
    for (int k2 = 0; k2 < N2; k2++) {
      long rr = ((long)n1 * k2) % N;
      long double a = -PI2 * (long double)rr / (long double)N;
      tr[n1 * N2 + k2] = (double)cosl(a);
      ti[n1 * N2 + k2] = (double)sinl(a);
    }
}
static void fill_prime(double *ct, double *st, int p, int h, int ph) {
  const long double PI2 = 6.283185307179586476925286766559005768L;
  memset(ct, 0, sizeof(double) * ph * h);
  memset(st, 0, sizeof(double) * ph * h);
  for (int k = 1; k <= h; k++)
    for (int j = 1; j <= h; j++) {
      long rr = ((long)j * k) % p;
      long double a = PI2 * (long double)rr / (long double)p;
      ct[(k - 1) * h + (j - 1)] = (double)cosl(a);
      st[(k - 1) * h + (j - 1)] = (double)sinl(a);
    }
}

void mp_init(void) {
  const long double PI2 = 6.283185307179586476925286766559005768L;
  K_S3 = (double)sinl(PI2 / 3.0L);
  K_R2 = (double)(sqrtl(2.0L) / 2.0L);
  K_C51 = (double)cosl(PI2 / 5.0L);  K_S51 = (double)sinl(PI2 / 5.0L);
  K_C52 = (double)cosl(2.0L * PI2 / 5.0L); K_S52 = (double)sinl(2.0L * PI2 / 5.0L);
  for (int k = 0; k < 5; k++) {
    K_W9R[k] = (double)cosl(-PI2 * k / 9.0L);
    K_W9I[k] = (double)sinl(-PI2 * k / 9.0L);
  }
  fill_ct(TW36R, TW36I, 6, 6, 36);
  fill_ct(TW45R, TW45I, 5, 9, 45);
  fill_ct(TW64R, TW64I, 8, 8, 64);
  fill_prime(C13, S13, 13, 6, PH13);
  fill_prime(C17, S17, 17, 8, PH17);
  fill_prime(C23, S23, 23, 11, PH23);
  if (!Xre) {
    size_t one = 4096 * 64 * sizeof(double);   // 2 MiB, max volume component
    char *blk = (char *)aligned_alloc(1 << 21, 4 * one);
    memset(blk, 0, 4 * one);
    Xre = (double *)blk; Xim = (double *)(blk + one);
    Cre = (double *)(blk + 2 * one); Cim = (double *)(blk + 3 * one);
  }
}

// ---------------------------------------------------------------- driver
#define RUN_CASE(LN, L_, LP_)                                        \
  case L_: {                                                         \
    long n = (long)L_ * L_ * L_;                                     \
    for (long b = 0; b < B; b++) {                                   \
      load_vol(L_, LP_, xin + b * 2 * n, Xre, Xim);                  \
      load_vol(L_, LP_, cin + b * 2 * n, Cre, Cim);                  \
      for (long it = 0; it < m; it++) {                              \
        sweepA_##LN();                                               \
        sweepB_##LN();                                               \
        if (it == 0) store_vol(L_, LP_, Xre, Xim, out1 + b * 2 * n); \
      }                                                              \
      store_vol(L_, LP_, Xre, Xim, outm + b * 2 * n);                \
    }                                                                \
  } break;

void mp_run(long L, long B, long m, const double *xin, const double *cin,
            double *out1, double *outm) {
  switch (L) {
    RUN_CASE(6, 6, 8)
    RUN_CASE(8, 8, 8)
    RUN_CASE(13, 13, 16)
    RUN_CASE(17, 17, 24)
    RUN_CASE(23, 23, 24)
    RUN_CASE(36, 36, 40)
    RUN_CASE(45, 45, 48)
    RUN_CASE(64, 64, 64)
    default: break;
  }
}

// ---------------------------------------------------------------- unit tests
#ifdef UNIT_TEST
#include <stdio.h>
#include <complex.h>

static void naive_dft(const double complex *in, double complex *out, int N) {
  for (int k = 0; k < N; k++) {
    double complex acc = 0;
    for (int j = 0; j < N; j++) {
      long double a = -6.283185307179586476925286766559005768L * ((long)j * k % N) / N;
      acc += in[j] * (cosl(a) + I * sinl(a));
    }
    out[k] = acc;
  }
}

static double LBUFR[64 * 8] __attribute__((aligned(64)));
static double LBUFI[64 * 8] __attribute__((aligned(64)));

typedef void (*linefn)(double *, double *, long, int, const double *, const double *);
static void t6(double *r, double *i, long s, int d, const double *a, const double *b) { fft6_line(r, i, s, d, a, b); }
static void t8(double *r, double *i, long s, int d, const double *a, const double *b) { fft8_line(r, i, s, d, a, b); }
static void t13(double *r, double *i, long s, int d, const double *a, const double *b) { fft13_line(r, i, s, d, a, b); }
static void t17(double *r, double *i, long s, int d, const double *a, const double *b) { fft17_line(r, i, s, d, a, b); }
static void t23(double *r, double *i, long s, int d, const double *a, const double *b) { fft23_line(r, i, s, d, a, b); }
static void t36(double *r, double *i, long s, int d, const double *a, const double *b) { fft36_line(r, i, s, d, a, b); }
static void t45(double *r, double *i, long s, int d, const double *a, const double *b) { fft45_line(r, i, s, d, a, b); }
static void t64(double *r, double *i, long s, int d, const double *a, const double *b) { fft64_line(r, i, s, d, a, b); }

int main(void) {
  mp_init();
  int Ls[8] = {6, 8, 13, 17, 23, 36, 45, 64};
  linefn fns[8] = {t6, t8, t13, t17, t23, t36, t45, t64};
  unsigned rr = 12345;
  for (int t = 0; t < 8; t++) {
    int N = Ls[t];
    double complex in[64][8], ref[64][8];
    for (int j = 0; j < N; j++)
      for (int l = 0; l < 8; l++) {
        rr = rr * 1103515245u + 12345u;
        double a = (double)(rr >> 8) / (1 << 24) - 0.5;
        rr = rr * 1103515245u + 12345u;
        double b = (double)(rr >> 8) / (1 << 24) - 0.5;
        in[j][l] = a + I * b;
        LBUFR[j * 8 + l] = a; LBUFI[j * 8 + l] = b;
      }
    for (int l = 0; l < 8; l++) {
      double complex colin[64], colout[64];
      for (int j = 0; j < N; j++) colin[j] = in[j][l];
      naive_dft(colin, colout, N);
      for (int j = 0; j < N; j++) ref[j][l] = colout[j];
    }
    fns[t](LBUFR, LBUFI, 8, 0, 0, 0);
    double err = 0, nrm = 0;
    for (int j = 0; j < N; j++)
      for (int l = 0; l < 8; l++) {
        double complex got = LBUFR[j * 8 + l] + I * LBUFI[j * 8 + l];
        double complex d = got - ref[j][l];
        err += creal(d) * creal(d) + cimag(d) * cimag(d);
        nrm += creal(ref[j][l]) * creal(ref[j][l]) + cimag(ref[j][l]) * cimag(ref[j][l]);
      }
    printf("L=%2d  rel err %.3e\n", N, sqrt(err / nrm));
  }
  // transpose test
  double tin[8 * 8], tout[8 * 8];
  for (int i = 0; i < 64; i++) tin[i] = i;
  {
    v8 a0 = VL(tin + 0), a1 = VL(tin + 8), a2 = VL(tin + 16), a3 = VL(tin + 24);
    v8 a4 = VL(tin + 32), a5 = VL(tin + 40), a6 = VL(tin + 48), a7 = VL(tin + 56);
    TR8(a0, a1, a2, a3, a4, a5, a6, a7);
    VS(tout + 0, a0); VS(tout + 8, a1); VS(tout + 16, a2); VS(tout + 24, a3);
    VS(tout + 32, a4); VS(tout + 40, a5); VS(tout + 48, a6); VS(tout + 56, a7);
  }
  int ok = 1;
  for (int r = 0; r < 8; r++)
    for (int c = 0; c < 8; c++) ok &= (tout[r * 8 + c] == tin[c * 8 + r]);
  printf("transpose %s\n", ok ? "OK" : "FAIL");
  return 0;
}
#endif
