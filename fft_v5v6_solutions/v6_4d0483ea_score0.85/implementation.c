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
#define PH17 12
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

#define PLANE 295424L   // doubles per plane (= 64*4616); buffers are contiguous
static double *Xre, *Xim, *Cre, *Cim;   // planar volume buffers (max size)
#ifndef XPFHINT
#define XPFHINT _MM_HINT_T0
#endif
#define PF1(p) _mm_prefetch((const char *)(p), XPFHINT)
#ifndef CPFHINT
#define CPFHINT _MM_HINT_T1
#endif
#ifndef PFD
#define PFD 8
#endif
#ifndef PFD2
#define PFD2 0
#endif
#define PFC(p) _mm_prefetch((const char *)(p), CPFHINT)

#define PSTR(x) #x
#define PRAGMA_UNROLL(n) _Pragma(PSTR(GCC unroll n))
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
#ifndef UTILE
#define UTILE 1
#endif
static inline __attribute__((always_inline)) void tile_load(const double *b, long ld,
                                                            double *s, int ncb) {
  PRAGMA_UNROLL(UTILE) for (int cb = 0; cb < ncb; cb++) {
    PF1(b + (PFD + 0) * ld + cb * 8); PF1(b + (PFD + 1) * ld + cb * 8);
    PF1(b + (PFD + 2) * ld + cb * 8); PF1(b + (PFD + 3) * ld + cb * 8);
    PF1(b + (PFD + 4) * ld + cb * 8); PF1(b + (PFD + 5) * ld + cb * 8);
    PF1(b + (PFD + 6) * ld + cb * 8); PF1(b + (PFD + 7) * ld + cb * 8);
#if PFD2
    _mm_prefetch((const char *)(b + (PFD2 + 0) * ld + cb * 8), _MM_HINT_T1);
    _mm_prefetch((const char *)(b + (PFD2 + 1) * ld + cb * 8), _MM_HINT_T1);
    _mm_prefetch((const char *)(b + (PFD2 + 2) * ld + cb * 8), _MM_HINT_T1);
    _mm_prefetch((const char *)(b + (PFD2 + 3) * ld + cb * 8), _MM_HINT_T1);
    _mm_prefetch((const char *)(b + (PFD2 + 4) * ld + cb * 8), _MM_HINT_T1);
    _mm_prefetch((const char *)(b + (PFD2 + 5) * ld + cb * 8), _MM_HINT_T1);
    _mm_prefetch((const char *)(b + (PFD2 + 6) * ld + cb * 8), _MM_HINT_T1);
    _mm_prefetch((const char *)(b + (PFD2 + 7) * ld + cb * 8), _MM_HINT_T1);
#endif
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
  PRAGMA_UNROLL(UTILE) for (int cb = 0; cb < ncb; cb++) {
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
#define STORE_MAP(pr, pi, vr, vi, pcr, pci, MV)                             \
  {                                                                         \
    v8 _zr = (vr) + VL(pcr), _zi = (vi) + VL(pci);                          \
    v8 _t = _zr * _zr + _zi * _zi;                                          \
    v8 _d, _r;                                                              \
    if ((MV) == 1) {                                                        \
      _d = vone + vsqrt8(_t);                                               \
      _r = (v8)_mm512_rcp14_pd((__m512d)_d);                                \
      _r = _r * (vtwo - _d * _r);                                           \
      _r = _r * (vtwo - _d * _r);                                           \
    } else {                                                                \
      v8 _t2 = _t + vtiny;                                                  \
      v8 _y = (v8)_mm512_rsqrt14_pd((__m512d)_t2);                          \
      _y = _y * (vc32 - vhalf * _t2 * _y * _y);                             \
      _y = _y * (vc32 - vhalf * _t2 * _y * _y);                             \
      _d = vone + _t2 * _y;                                                 \
      _r = (v8)_mm512_rcp14_pd((__m512d)_d);                                \
      _r = _r * (vtwo - _d * _r);                                           \
      _r = _r * (vtwo - _d * _r);                                           \
    }                                                                       \
    VS((pr), _zr * _r); VS((pi), _zi * _r);                                 \
  }
#define STORE_X(pr, pi, vr, vi, off)                           \
  {                                                            \
    if (domap) { STORE_MAP((pr) + (off), (pi) + (off), vr, vi, cre + (off), cim + (off), domap) } \
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

#define FFT4_CORE(x0, x1, x2, x3)                              \
  {                                                            \
    v8 _t0r = x0##r + x2##r, _t0i = x0##i + x2##i;             \
    v8 _t1r = x0##r - x2##r, _t1i = x0##i - x2##i;             \
    v8 _t2r = x1##r + x3##r, _t2i = x1##i + x3##i;             \
    v8 _t3r = x1##r - x3##r, _t3i = x1##i - x3##i;             \
    x0##r = _t0r + _t2r; x0##i = _t0i + _t2i;                  \
    x2##r = _t0r - _t2r; x2##i = _t0i - _t2i;                  \
    x1##r = _t1r + _t3i; x1##i = _t1i - _t3r;                  \
    x3##r = _t1r - _t3i; x3##i = _t1i + _t3r;                  \
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
  const v8 vc32 = vbc(1.5), vtwo = vbc(2.0), vtiny = vbc(1e-300);     \
  (void)vc32; (void)vtwo; (void)vtiny;                                \
  (void)vhalf; (void)vs3; (void)vr2; (void)vc51; (void)vc52;          \
  (void)vs51; (void)vs52; (void)vw91r; (void)vw91i; (void)vw92r;      \
  (void)vw92i; (void)vw94r; (void)vw94i; (void)vone; (void)vzero;

#ifndef JU13
#define JU13 16
#endif
#ifndef U36S1
#define U36S1 4
#endif
#ifndef U36S2
#define U36S2 9
#endif
#ifndef U45S1
#define U45S1 1
#endif
#ifndef U45S2
#define U45S2 1
#endif
#ifndef JROLL13
#define JROLL13 0
#endif
#ifndef JP17
#define JP17 16
#endif
#ifndef JP23
#define JP23 16
#endif
#ifndef BLK17
#define BLK17 6
#endif
#ifndef BLK23
#define BLK23 6
#endif
#ifndef JU17
#define JU17 2
#endif
#ifndef JU23
#define JU23 1
#endif
// PFA index tables (input (9*n1+K*n2)%N, output (A*q1+B*q2)%N)
static const long MI36[4][9] = {
  {0, 4, 8, 12, 16, 20, 24, 28, 32},
  {9, 13, 17, 21, 25, 29, 33, 1, 5},
  {18, 22, 26, 30, 34, 2, 6, 10, 14},
  {27, 31, 35, 3, 7, 11, 15, 19, 23}};
static const long MO36[9][4] = {
  {0, 9, 18, 27},
  {28, 1, 10, 19},
  {20, 29, 2, 11},
  {12, 21, 30, 3},
  {4, 13, 22, 31},
  {32, 5, 14, 23},
  {24, 33, 6, 15},
  {16, 25, 34, 7},
  {8, 17, 26, 35}};
static const long MI45[5][9] = {
  {0, 5, 10, 15, 20, 25, 30, 35, 40},
  {9, 14, 19, 24, 29, 34, 39, 44, 4},
  {18, 23, 28, 33, 38, 43, 3, 8, 13},
  {27, 32, 37, 42, 2, 7, 12, 17, 22},
  {36, 41, 1, 6, 11, 16, 21, 26, 31}};
static const long MO45[9][5] = {
  {0, 36, 27, 18, 9},
  {10, 1, 37, 28, 19},
  {20, 11, 2, 38, 29},
  {30, 21, 12, 3, 39},
  {40, 31, 22, 13, 4},
  {5, 41, 32, 23, 14},
  {15, 6, 42, 33, 24},
  {25, 16, 7, 43, 34},
  {35, 26, 17, 8, 44}};
// ---------------------------------------------------------------- line FFTs
// element j of the line lives at re[j*s], im[j*s]; s is a compile-time constant
// at every call site.  domap: fuse z = X + c ; x = z/(1+|z|) into the store.

static inline __attribute__((always_inline)) void
fft6_line(double *re, double *im, const long s, const int domap,
          const double *cre, const double *cim, const double *pf) { (void)pf;
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
          const double *cre, const double *cim, const double *pf) { (void)pf;
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
#define PRIME_LINE(P, H, PHN, CT, ST, BLK, JU, JP)                                        \
  DEFCONSTS                                                                       \
  v8 x0r = VL(re), x0i = VL(im);                                                  \
  v8 sumr = x0r, sumi = x0i;                                                      \
  PRAGMA_UNROLL(JP) for (int j = 1; j <= H; j++) {                                \
    v8 ar = VL(re + j * s), ai = VL(im + j * s);                                  \
    v8 br = VL(re + (P - j) * s), bi = VL(im + (P - j) * s);                      \
    v8 par = ar + br, pai = ai + bi, pbr = ar - br, pbi = ai - bi;                \
    VS(SAR + (j - 1) * 8, par); VS(SAI + (j - 1) * 8, pai);                       \
    VS(SBR + (j - 1) * 8, pbr); VS(SBI + (j - 1) * 8, pbi);                       \
    sumr += par; sumi += pai;                                                     \
  }                                                                               \
  STORE_X(re, im, sumr, sumi, 0);                                                 \
  _Pragma("GCC unroll 4") for (int k0 = 1; k0 <= H; k0 += BLK) {                  \
    v8 Crr[BLK], Cii[BLK], Srr[BLK], Sii[BLK];                                    \
    _Pragma("GCC unroll 8") for (int t = 0; t < BLK; t++) {                       \
      Crr[t] = x0r; Cii[t] = x0i; Srr[t] = vzero; Sii[t] = vzero;                 \
    }                                                                             \
    PRAGMA_UNROLL(JU) for (int j = 0; j < H; j++) {                               \
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
           const double *cre, const double *cim, const double *pf) { (void)pf;
  DEFCONSTS
  v8 x0r = VL(re), x0i = VL(im);
#define LDAB(j, a, b)                                              \
  v8 a##r, a##i, b##r, b##i;                                       \
  {                                                                \
    v8 _ur = VL(re + (j) * s), _ui = VL(im + (j) * s);             \
    v8 _vr = VL(re + (13 - (j)) * s), _vi = VL(im + (13 - (j)) * s); \
    a##r = _ur + _vr; a##i = _ui + _vi;                            \
    b##r = _ur - _vr; b##i = _ui - _vi;                            \
  }
  LDAB(1, a1, b1) LDAB(2, a2, b2) LDAB(3, a3, b3)
  LDAB(4, a4, b4) LDAB(5, a5, b5) LDAB(6, a6, b6)
#undef LDAB
  {
    v8 sr = x0r + a1r + a2r + a3r + a4r + a5r + a6r;
    v8 si = x0i + a1i + a2i + a3i + a4i + a5i + a6i;
    STORE_X(re, im, sr, si, 0);
  }
#define KROW(k)                                                               \
  {                                                                           \
    const double *ct = C13 + ((k) - 1) * 6, *st = S13 + ((k) - 1) * 6;        \
    v8 crr = x0r, cii = x0i, srr, sii;                                        \
    crr += vbc(ct[0]) * a1r; cii += vbc(ct[0]) * a1i;                         \
    srr = vbc(st[0]) * b1r; sii = vbc(st[0]) * b1i;                           \
    crr += vbc(ct[1]) * a2r; cii += vbc(ct[1]) * a2i;                         \
    srr += vbc(st[1]) * b2r; sii += vbc(st[1]) * b2i;                         \
    crr += vbc(ct[2]) * a3r; cii += vbc(ct[2]) * a3i;                         \
    srr += vbc(st[2]) * b3r; sii += vbc(st[2]) * b3i;                         \
    crr += vbc(ct[3]) * a4r; cii += vbc(ct[3]) * a4i;                         \
    srr += vbc(st[3]) * b4r; sii += vbc(st[3]) * b4i;                         \
    crr += vbc(ct[4]) * a5r; cii += vbc(ct[4]) * a5i;                         \
    srr += vbc(st[4]) * b5r; sii += vbc(st[4]) * b5i;                         \
    crr += vbc(ct[5]) * a6r; cii += vbc(ct[5]) * a6i;                         \
    srr += vbc(st[5]) * b6r; sii += vbc(st[5]) * b6i;                         \
    v8 xr1 = crr + sii, xi1 = cii - srr;                                      \
    v8 xr2 = crr - sii, xi2 = cii + srr;                                      \
    STORE_X(re, im, xr1, xi1, (k) * s);                                       \
    STORE_X(re, im, xr2, xi2, (13 - (k)) * s);                                \
  }
#if JROLL13
  for (int k = 1; k <= 6; k++) KROW(k)
#else
  KROW(1) KROW(2) KROW(3) KROW(4) KROW(5) KROW(6)
#endif
#undef KROW
}
static inline __attribute__((always_inline)) void
fft17_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim, const double *pf) {
  PRIME_LINE(17, 8, PH17, C17, S17, BLK17, JU17, JP17)
}
static inline __attribute__((always_inline)) void
fft23_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim, const double *pf) {
  PRIME_LINE(23, 11, PH23, C23, S23, BLK23, JU23, JP23)
}

// 36 = 4 (x) 9  PFA: no twiddles (rolled stage loops)
static inline __attribute__((always_inline)) void
fft36_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim, const double *pf) {
  DEFCONSTS
  PRAGMA_UNROLL(U36S1) for (int n1 = 0; n1 < 4; n1++) {
    const long *mi = MI36[n1];
    if (pf) {
      _Pragma("GCC unroll 9") for (int q = 0; q < 9; q++) {
        PF1(pf + (n1 * 9 + q) * s); PF1(pf + PLANE + (n1 * 9 + q) * s);
      }
    }
    v8 y0r = VL(re + mi[0] * s), y0i = VL(im + mi[0] * s);
    v8 y1r = VL(re + mi[1] * s), y1i = VL(im + mi[1] * s);
    v8 y2r = VL(re + mi[2] * s), y2i = VL(im + mi[2] * s);
    v8 y3r = VL(re + mi[3] * s), y3i = VL(im + mi[3] * s);
    v8 y4r = VL(re + mi[4] * s), y4i = VL(im + mi[4] * s);
    v8 y5r = VL(re + mi[5] * s), y5i = VL(im + mi[5] * s);
    v8 y6r = VL(re + mi[6] * s), y6i = VL(im + mi[6] * s);
    v8 y7r = VL(re + mi[7] * s), y7i = VL(im + mi[7] * s);
    v8 y8r = VL(re + mi[8] * s), y8i = VL(im + mi[8] * s);
    FFT9_CORE(y0, y1, y2, y3, y4, y5, y6, y7, y8);
    // X9[q2] lives in var P[q2], P = {0,3,6,1,4,7,2,5,8}
    double *t1r = T1R + n1 * 8, *t1i = T1I + n1 * 8;
#define ST36(q2, var)  { VS(t1r + (q2) * 32, var##r); VS(t1i + (q2) * 32, var##i); }
    ST36(0, y0) ST36(1, y3) ST36(2, y6) ST36(3, y1) ST36(4, y4)
    ST36(5, y7) ST36(6, y2) ST36(7, y5) ST36(8, y8)
#undef ST36
  }
  PRAGMA_UNROLL(U36S2) for (int q2 = 0; q2 < 9; q2++) {
    const long *mo = MO36[q2];
    if (pf && domap) {
      _Pragma("GCC unroll 4") for (int q = 0; q < 4; q++) {
        PFC(pf + 2 * PLANE + (q2 * 4 + q) * s); PFC(pf + 3 * PLANE + (q2 * 4 + q) * s);
      }
    }
    const double *t1r = T1R + q2 * 32, *t1i = T1I + q2 * 32;
    v8 x0r = VL(t1r + 0 * 8), x0i = VL(t1i + 0 * 8);
    v8 x1r = VL(t1r + 1 * 8), x1i = VL(t1i + 1 * 8);
    v8 x2r = VL(t1r + 2 * 8), x2i = VL(t1i + 2 * 8);
    v8 x3r = VL(t1r + 3 * 8), x3i = VL(t1i + 3 * 8);
    FFT4_CORE(x0, x1, x2, x3);
    STORE_X(re, im, x0r, x0i, mo[0] * s);
    STORE_X(re, im, x1r, x1i, mo[1] * s);
    STORE_X(re, im, x2r, x2i, mo[2] * s);
    STORE_X(re, im, x3r, x3i, mo[3] * s);
  }
}

// 45 = 5 (x) 9  PFA (rolled stage loops)
static inline __attribute__((always_inline)) void
fft45_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim, const double *pf) {
  DEFCONSTS
  PRAGMA_UNROLL(U45S1) for (int n1 = 0; n1 < 5; n1++) {
    const long *mi = MI45[n1];
    if (pf) {
      _Pragma("GCC unroll 9") for (int q = 0; q < 9; q++) {
        PF1(pf + (n1 * 9 + q) * s); PF1(pf + PLANE + (n1 * 9 + q) * s);
      }
    }
    v8 y0r = VL(re + mi[0] * s), y0i = VL(im + mi[0] * s);
    v8 y1r = VL(re + mi[1] * s), y1i = VL(im + mi[1] * s);
    v8 y2r = VL(re + mi[2] * s), y2i = VL(im + mi[2] * s);
    v8 y3r = VL(re + mi[3] * s), y3i = VL(im + mi[3] * s);
    v8 y4r = VL(re + mi[4] * s), y4i = VL(im + mi[4] * s);
    v8 y5r = VL(re + mi[5] * s), y5i = VL(im + mi[5] * s);
    v8 y6r = VL(re + mi[6] * s), y6i = VL(im + mi[6] * s);
    v8 y7r = VL(re + mi[7] * s), y7i = VL(im + mi[7] * s);
    v8 y8r = VL(re + mi[8] * s), y8i = VL(im + mi[8] * s);
    FFT9_CORE(y0, y1, y2, y3, y4, y5, y6, y7, y8);
    double *t1r = T1R + n1 * 8, *t1i = T1I + n1 * 8;
#define ST45(q2, var)  { VS(t1r + (q2) * 40, var##r); VS(t1i + (q2) * 40, var##i); }
    ST45(0, y0) ST45(1, y3) ST45(2, y6) ST45(3, y1) ST45(4, y4)
    ST45(5, y7) ST45(6, y2) ST45(7, y5) ST45(8, y8)
#undef ST45
  }
  PRAGMA_UNROLL(U45S2) for (int q2 = 0; q2 < 9; q2++) {
    const long *mo = MO45[q2];
    if (pf && domap) {
      _Pragma("GCC unroll 5") for (int q = 0; q < 5; q++) {
        PFC(pf + 2 * PLANE + (q2 * 5 + q) * s); PFC(pf + 3 * PLANE + (q2 * 5 + q) * s);
      }
    }
    const double *t1r = T1R + q2 * 40, *t1i = T1I + q2 * 40;
    v8 x0r = VL(t1r + 0 * 8), x0i = VL(t1i + 0 * 8);
    v8 x1r = VL(t1r + 1 * 8), x1i = VL(t1i + 1 * 8);
    v8 x2r = VL(t1r + 2 * 8), x2i = VL(t1i + 2 * 8);
    v8 x3r = VL(t1r + 3 * 8), x3i = VL(t1i + 3 * 8);
    v8 x4r = VL(t1r + 4 * 8), x4i = VL(t1i + 4 * 8);
    FFT5_CORE(x0, x1, x2, x3, x4);
    STORE_X(re, im, x0r, x0i, mo[0] * s);
    STORE_X(re, im, x1r, x1i, mo[1] * s);
    STORE_X(re, im, x2r, x2i, mo[2] * s);
    STORE_X(re, im, x3r, x3i, mo[3] * s);
    STORE_X(re, im, x4r, x4i, mo[4] * s);
  }
}

// 64 = 8 x 8   (stage loops rolled: front-end friendly)
static inline __attribute__((always_inline)) void
fft64_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim, const double *pf) {
  DEFCONSTS
  // n1 = 0 peeled (no twiddles)
  {
    v8 x0r = VL(re + 0 * s), x0i = VL(im + 0 * s);
    v8 x1r = VL(re + 8 * s), x1i = VL(im + 8 * s);
    v8 x2r = VL(re + 16 * s), x2i = VL(im + 16 * s);
    v8 x3r = VL(re + 24 * s), x3i = VL(im + 24 * s);
    v8 x4r = VL(re + 32 * s), x4i = VL(im + 32 * s);
    v8 x5r = VL(re + 40 * s), x5i = VL(im + 40 * s);
    v8 x6r = VL(re + 48 * s), x6i = VL(im + 48 * s);
    v8 x7r = VL(re + 56 * s), x7i = VL(im + 56 * s);
    if (pf) {
      _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++) {
        PF1(pf + (0 * 8 + q) * s); PF1(pf + PLANE + (0 * 8 + q) * s);
      }
    }
    FFT8_CORE(x0, x1, x2, x3, x4, x5, x6, x7);
    VS(T1R + (0 * 8 + 0) * 8, x0r); VS(T1I + (0 * 8 + 0) * 8, x0i);
    VS(T1R + (1 * 8 + 0) * 8, x1r); VS(T1I + (1 * 8 + 0) * 8, x1i);
    VS(T1R + (2 * 8 + 0) * 8, x2r); VS(T1I + (2 * 8 + 0) * 8, x2i);
    VS(T1R + (3 * 8 + 0) * 8, x3r); VS(T1I + (3 * 8 + 0) * 8, x3i);
    VS(T1R + (4 * 8 + 0) * 8, x4r); VS(T1I + (4 * 8 + 0) * 8, x4i);
    VS(T1R + (5 * 8 + 0) * 8, x5r); VS(T1I + (5 * 8 + 0) * 8, x5i);
    VS(T1R + (6 * 8 + 0) * 8, x6r); VS(T1I + (6 * 8 + 0) * 8, x6i);
    VS(T1R + (7 * 8 + 0) * 8, x7r); VS(T1I + (7 * 8 + 0) * 8, x7i);
  }
  for (int n1 = 1; n1 < 8; n1++) {
    const double *rr = re + n1 * s, *ii = im + n1 * s;
    v8 x0r = VL(rr + 0 * s), x0i = VL(ii + 0 * s);
    v8 x1r = VL(rr + 8 * s), x1i = VL(ii + 8 * s);
    v8 x2r = VL(rr + 16 * s), x2i = VL(ii + 16 * s);
    v8 x3r = VL(rr + 24 * s), x3i = VL(ii + 24 * s);
    v8 x4r = VL(rr + 32 * s), x4i = VL(ii + 32 * s);
    v8 x5r = VL(rr + 40 * s), x5i = VL(ii + 40 * s);
    v8 x6r = VL(rr + 48 * s), x6i = VL(ii + 48 * s);
    v8 x7r = VL(rr + 56 * s), x7i = VL(ii + 56 * s);
    if (pf) {
      _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++) {
        PF1(pf + (n1 * 8 + q) * s); PF1(pf + PLANE + (n1 * 8 + q) * s);
      }
    }
    FFT8_CORE(x0, x1, x2, x3, x4, x5, x6, x7);
    const double *twr = TW64R + n1 * 8, *twi = TW64I + n1 * 8;
    CMULT(x1r, x1i, x1r, x1i, vbc(twr[1]), vbc(twi[1]));
    CMULT(x2r, x2i, x2r, x2i, vbc(twr[2]), vbc(twi[2]));
    CMULT(x3r, x3i, x3r, x3i, vbc(twr[3]), vbc(twi[3]));
    CMULT(x4r, x4i, x4r, x4i, vbc(twr[4]), vbc(twi[4]));
    CMULT(x5r, x5i, x5r, x5i, vbc(twr[5]), vbc(twi[5]));
    CMULT(x6r, x6i, x6r, x6i, vbc(twr[6]), vbc(twi[6]));
    CMULT(x7r, x7i, x7r, x7i, vbc(twr[7]), vbc(twi[7]));
    double *t1r = T1R + n1 * 8, *t1i = T1I + n1 * 8;
    VS(t1r + 0 * 64, x0r); VS(t1i + 0 * 64, x0i);
    VS(t1r + 1 * 64, x1r); VS(t1i + 1 * 64, x1i);
    VS(t1r + 2 * 64, x2r); VS(t1i + 2 * 64, x2i);
    VS(t1r + 3 * 64, x3r); VS(t1i + 3 * 64, x3i);
    VS(t1r + 4 * 64, x4r); VS(t1i + 4 * 64, x4i);
    VS(t1r + 5 * 64, x5r); VS(t1i + 5 * 64, x5i);
    VS(t1r + 6 * 64, x6r); VS(t1i + 6 * 64, x6i);
    VS(t1r + 7 * 64, x7r); VS(t1i + 7 * 64, x7i);
  }
  for (int k2 = 0; k2 < 8; k2++) {
    const double *t1r = T1R + k2 * 64, *t1i = T1I + k2 * 64;
    v8 x0r = VL(t1r + 0 * 8), x0i = VL(t1i + 0 * 8);
    v8 x1r = VL(t1r + 1 * 8), x1i = VL(t1i + 1 * 8);
    v8 x2r = VL(t1r + 2 * 8), x2i = VL(t1i + 2 * 8);
    v8 x3r = VL(t1r + 3 * 8), x3i = VL(t1i + 3 * 8);
    v8 x4r = VL(t1r + 4 * 8), x4i = VL(t1i + 4 * 8);
    v8 x5r = VL(t1r + 5 * 8), x5i = VL(t1i + 5 * 8);
    v8 x6r = VL(t1r + 6 * 8), x6i = VL(t1i + 6 * 8);
    v8 x7r = VL(t1r + 7 * 8), x7i = VL(t1i + 7 * 8);
    if (pf && domap) {
      _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++) {
        PFC(pf + 2 * PLANE + (k2 * 8 + q) * s); PFC(pf + 3 * PLANE + (k2 * 8 + q) * s);
      }
    }
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
#ifdef EXPOSE_SWEEPS
#define SWEEPVIS
#else
#define SWEEPVIS static
#endif
// layout: element (x,y,z) at  x*SS + y*R + z  (doubles) in each plane.
// LP = vector-padded z extent, LPS = padded rows per slice (junk rows stay 0),
// R = row stride, SS = slice stride; R and SS chosen to spread cache sets.
#define DEF_SWEEPA(LN, L_, LP_, R_, LPS_, SS_, FFTLINE)                               \
  SWEEPVIS void sweepA_##LN(void) {                                                   \
    for (long x = 0; x < L_; x++) {                                                   \
      double *br = Xre + x * (long)SS_, *bi = Xim + x * (long)SS_;                    \
      for (long y0 = 0; y0 < LPS_; y0 += 8) {                                         \
        tile_load(br + y0 * R_, R_, SZA, LP_ / 8);                                    \
        tile_load(bi + y0 * R_, R_, SZB, LP_ / 8);                                    \
        FFTLINE(SZA, SZB, 8, 0, 0, 0, 0);                                             \
        tile_store(br + y0 * R_, R_, SZA, LP_ / 8, L_);                               \
        tile_store(bi + y0 * R_, R_, SZB, LP_ / 8, L_);                               \
      }                                                                               \
      for (long zc = 0; zc < LP_; zc += 8) FFTLINE(br + zc, bi + zc, R_, 0, 0, 0, 0); \
    }                                                                                 \
  }

#define DEF_SWEEPA_G(LN, L_, LP_, R_, MP_, FFTLINE)                                   \
  SWEEPVIS void sweepA_##LN(void) {                                                   \
    long t = 0;                                                                       \
    for (long x = 0; x < L_; x++) {                                                   \
      long lim = (x + 1) * L_;                                                        \
      for (; t < lim; t += 8) {                                                       \
        tile_load(Xre + t * R_, R_, SZA, LP_ / 8);                                    \
        tile_load(Xim + t * R_, R_, SZB, LP_ / 8);                                    \
        FFTLINE(SZA, SZB, 8, 0, 0, 0, 0);                                             \
        tile_store(Xre + t * R_, R_, SZA, LP_ / 8, L_);                               \
        tile_store(Xim + t * R_, R_, SZB, LP_ / 8, L_);                               \
      }                                                                               \
      double *br = Xre + x * (long)(L_ * R_), *bi = Xim + x * (long)(L_ * R_);        \
      for (long zc = 0; zc < LP_; zc += 8) FFTLINE(br + zc, bi + zc, R_, 0, 0, 0, 0); \
    }                                                                                 \
  }

#define DEF_SWEEPB(LN, L_, LP_, R_, LPS_, SS_, FFTLINE, MAPV)                               \
  SWEEPVIS void sweepB_##LN(void) {                                                     \
    for (long y = 0; y < L_; y++) {                                                   \
      long base = y * R_;                                                             \
      for (long zc = 0; zc < LP_; zc += 8) {                                          \
        long off = base + zc;                                                         \
        long zc2 = zc + 8, y2 = y;                                                    \
        if (zc2 >= LP_) { zc2 = 0; y2 = (y + 1 < L_) ? y + 1 : 0; }                   \
        FFTLINE(Xre + off, Xim + off, (long)SS_, MAPV, Cre + off, Cim + off,          \
                Xre + y2 * R_ + zc2);                                                 \
      }                                                                               \
    }                                                                                 \
  }


// register-resident fused Z+Y sweep for tiny sizes (slice fits in registers)
SWEEPVIS void sweepA_6(void) {
  DEFCONSTS
  for (long x = 0; x < 6; x++) {
    double *br = Xre + x * 72, *bi = Xim + x * 72;
    v8 y0r = VL(br + 0 * 8), y0i = VL(bi + 0 * 8);
    v8 y1r = VL(br + 1 * 8), y1i = VL(bi + 1 * 8);
    v8 y2r = VL(br + 2 * 8), y2i = VL(bi + 2 * 8);
    v8 y3r = VL(br + 3 * 8), y3i = VL(bi + 3 * 8);
    v8 y4r = VL(br + 4 * 8), y4i = VL(bi + 4 * 8);
    v8 y5r = VL(br + 5 * 8), y5i = VL(bi + 5 * 8);
    FFT6_CORE(y0, y1, y2, y3, y4, y5);       // pass Y (across rows)
    v8 y6r = vzero, y7r = vzero, y6i = vzero, y7i = vzero;
    TR8(y0r, y1r, y2r, y3r, y4r, y5r, y6r, y7r);
    TR8(y0i, y1i, y2i, y3i, y4i, y5i, y6i, y7i);
    FFT6_CORE(y0, y1, y2, y3, y4, y5);       // pass Z (across columns)
    TR8(y0r, y1r, y2r, y3r, y4r, y5r, y6r, y7r);
    TR8(y0i, y1i, y2i, y3i, y4i, y5i, y6i, y7i);
    VS(br + 0 * 8, y0r); VS(bi + 0 * 8, y0i);
    VS(br + 1 * 8, y1r); VS(bi + 1 * 8, y1i);
    VS(br + 2 * 8, y2r); VS(bi + 2 * 8, y2i);
    VS(br + 3 * 8, y3r); VS(bi + 3 * 8, y3i);
    VS(br + 4 * 8, y4r); VS(bi + 4 * 8, y4i);
    VS(br + 5 * 8, y5r); VS(bi + 5 * 8, y5i);
  }
}
SWEEPVIS void sweepA_8(void) {
  DEFCONSTS
  for (long x = 0; x < 8; x++) {
    double *br = Xre + x * 72, *bi = Xim + x * 72;
    v8 y0r = VL(br + 0 * 8), y0i = VL(bi + 0 * 8);
    v8 y1r = VL(br + 1 * 8), y1i = VL(bi + 1 * 8);
    v8 y2r = VL(br + 2 * 8), y2i = VL(bi + 2 * 8);
    v8 y3r = VL(br + 3 * 8), y3i = VL(bi + 3 * 8);
    v8 y4r = VL(br + 4 * 8), y4i = VL(bi + 4 * 8);
    v8 y5r = VL(br + 5 * 8), y5i = VL(bi + 5 * 8);
    v8 y6r = VL(br + 6 * 8), y6i = VL(bi + 6 * 8);
    v8 y7r = VL(br + 7 * 8), y7i = VL(bi + 7 * 8);
    FFT8_CORE(y0, y1, y2, y3, y4, y5, y6, y7);
    TR8(y0r, y1r, y2r, y3r, y4r, y5r, y6r, y7r);
    TR8(y0i, y1i, y2i, y3i, y4i, y5i, y6i, y7i);
    FFT8_CORE(y0, y1, y2, y3, y4, y5, y6, y7);
    TR8(y0r, y1r, y2r, y3r, y4r, y5r, y6r, y7r);
    TR8(y0i, y1i, y2i, y3i, y4i, y5i, y6i, y7i);
    VS(br + 0 * 8, y0r); VS(bi + 0 * 8, y0i);
    VS(br + 1 * 8, y1r); VS(bi + 1 * 8, y1i);
    VS(br + 2 * 8, y2r); VS(bi + 2 * 8, y2i);
    VS(br + 3 * 8, y3r); VS(bi + 3 * 8, y3i);
    VS(br + 4 * 8, y4r); VS(bi + 4 * 8, y4i);
    VS(br + 5 * 8, y5r); VS(bi + 5 * 8, y5i);
    VS(br + 6 * 8, y6r); VS(bi + 6 * 8, y6i);
    VS(br + 7 * 8, y7r); VS(bi + 7 * 8, y7i);
  }
}

DEF_SWEEPB(6, 6, 8, 8, 8, 72, fft6_line, 1)
DEF_SWEEPB(8, 8, 8, 8, 8, 72, fft8_line, 1)
DEF_SWEEPA_G(13, 13, 16, 16, 176, fft13_line)
DEF_SWEEPB(13, 13, 16, 16, 16, 208, fft13_line, 1)
DEF_SWEEPA_G(17, 17, 24, 24, 296, fft17_line)
DEF_SWEEPB(17, 17, 24, 24, 24, 408, fft17_line, 1)
DEF_SWEEPA_G(23, 23, 24, 24, 536, fft23_line)
DEF_SWEEPB(23, 23, 24, 24, 24, 552, fft23_line, 1)
DEF_SWEEPA_G(36, 36, 40, 40, 1296, fft36_line)
DEF_SWEEPB(36, 36, 40, 40, 40, 1440, fft36_line, 1)
DEF_SWEEPA(45, 45, 48, 48, 48, 2312, fft45_line)
DEF_SWEEPB(45, 45, 48, 48, 48, 2312, fft45_line, 1)
DEF_SWEEPA(64, 64, 64, 72, 64, 4616, fft64_line)
DEF_SWEEPB(64, 64, 64, 72, 64, 4616, fft64_line, 1)

#ifdef EXPOSE_SWEEPS
void micro_line(int which, long n) {   // run codelet on L1-resident scratch
  for (long i = 0; i < n; i++) {
    switch (which) {
      case 6:  fft6_line(SZA, SZB, 8, 0, 0, 0, 0); break;
      case 8:  fft8_line(SZA, SZB, 8, 0, 0, 0, 0); break;
      case 13: fft13_line(SZA, SZB, 8, 0, 0, 0, 0); break;
      case 17: fft17_line(SZA, SZB, 8, 0, 0, 0, 0); break;
      case 23: fft23_line(SZA, SZB, 8, 0, 0, 0, 0); break;
      case 36: fft36_line(SZA, SZB, 8, 0, 0, 0, 0); break;
      case 45: fft45_line(SZA, SZB, 8, 0, 0, 0, 0); break;
      case 64: fft64_line(SZA, SZB, 8, 0, 0, 0, 0); break;
    }
  }
}
#endif


// ---------------------------------------------------------------- SoA batch-of-8 engine
// For small sizes with B >= 8: lanes = 8 volumes. Layout: plane[x][y*L+z][lane],
// slice stride SP = 8*(L*L+1) (anti-aliasing pad). No transposes, no pad lanes.
#define DEF_SOA(LN, L_, SP_, MAPV)                                                  \
  static void soa_iter_##LN(void) {                                          \
    for (long x = 0; x < L_; x++) {                                          \
      double *br = Xre + x * SP_, *bi = Xim + x * SP_;                       \
      for (long y = 0; y < L_; y++)                                          \
        fft##LN##_line(br + y * (L_ * 8), bi + y * (L_ * 8), 8, 0, 0, 0, 0); \
      for (long z = 0; z < L_; z++)                                          \
        fft##LN##_line(br + z * 8, bi + z * 8, L_ * 8, 0, 0, 0, 0);          \
    }                                                                        \
    for (long y = 0; y < L_; y++)                                            \
      for (long z = 0; z < L_; z++) {                                        \
        long off = (y * L_ + z) * 8;                                         \
        fft##LN##_line(Xre + off, Xim + off, SP_, MAPV, Cre + off, Cim + off, 0); \
      }                                                                      \
  }

DEF_SOA(6, 6, 8 * (36 + 1), 2)
DEF_SOA(8, 8, 8 * (64 + 1), 2)
DEF_SOA(13, 13, 8 * (169 + 1), 1)
DEF_SOA(17, 17, 8 * (289 + 1), 2)

static void soa_load(long L, long SP, const double *xin, double *dre, double *dim) {
  long n = L * L * L, n2 = L * L;
  for (long v = 0; v < 8; v++) {
    const double *p = xin + v * 2 * n;
    for (long x = 0; x < L; x++) {
      double *qr = dre + x * SP + v, *qi = dim + x * SP + v;
      const double *px = p + x * n2 * 2;
      for (long i = 0; i < n2; i++) { qr[i * 8] = px[2 * i]; qi[i * 8] = px[2 * i + 1]; }
    }
  }
}
static void soa_store(long L, long SP, const double *sre, const double *sim, double *xout) {
  long n = L * L * L, n2 = L * L;
  for (long v = 0; v < 8; v++) {
    double *p = xout + v * 2 * n;
    for (long x = 0; x < L; x++) {
      const double *qr = sre + x * SP + v, *qi = sim + x * SP + v;
      double *px = p + x * n2 * 2;
      for (long i = 0; i < n2; i++) { px[2 * i] = qr[i * 8]; px[2 * i + 1] = qi[i * 8]; }
    }
  }
}

// ---------------------------------------------------------------- conversions
// interleaved complex <-> planar, vectorized (2 permutes per 8 complexes)
static const long long IDX_RE[8] = {0, 2, 4, 6, 8, 10, 12, 14};
static const long long IDX_IM[8] = {1, 3, 5, 7, 9, 11, 13, 15};
static const long long IDX_LO[8] = {0, 8, 1, 9, 2, 10, 3, 11};
static const long long IDX_HI[8] = {4, 12, 5, 13, 6, 14, 7, 15};

static inline void row_deinter(const double *p, double *qr, double *qi, long L) {
  __m512i ire = _mm512_loadu_si512(IDX_RE), iim = _mm512_loadu_si512(IDX_IM);
  long z = 0;
  for (; z + 8 <= L; z += 8) {
    __m512d v0 = _mm512_loadu_pd(p + 2 * z), v1 = _mm512_loadu_pd(p + 2 * z + 8);
    _mm512_store_pd(qr + z, _mm512_permutex2var_pd(v0, ire, v1));
    _mm512_store_pd(qi + z, _mm512_permutex2var_pd(v0, iim, v1));
  }
  for (; z < L; z++) { qr[z] = p[2 * z]; qi[z] = p[2 * z + 1]; }
}
static inline void row_inter(double *p, const double *qr, const double *qi, long L) {
  __m512i ilo = _mm512_loadu_si512(IDX_LO), ihi = _mm512_loadu_si512(IDX_HI);
  long z = 0;
  for (; z + 8 <= L; z += 8) {
    __m512d vr = _mm512_load_pd(qr + z), vi = _mm512_load_pd(qi + z);
    _mm512_storeu_pd(p + 2 * z, _mm512_permutex2var_pd(vr, ilo, vi));
    _mm512_storeu_pd(p + 2 * z + 8, _mm512_permutex2var_pd(vr, ihi, vi));
  }
  for (; z < L; z++) { p[2 * z] = qr[z]; p[2 * z + 1] = qi[z]; }
}
static void load_vol(long L, long R, long SS, const double *xin, double *dre, double *dim) {
  for (long x = 0; x < L; x++)
    for (long y = 0; y < L; y++)
      row_deinter(xin + (x * L + y) * L * 2, dre + x * SS + y * R, dim + x * SS + y * R, L);
}
static void store_vol(long L, long R, long SS, const double *sre, const double *sim, double *xout) {
  for (long x = 0; x < L; x++)
    for (long y = 0; y < L; y++)
      row_inter(xout + (x * L + y) * L * 2, sre + x * SS + y * R, sim + x * SS + y * R, L);
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
    size_t one = PLANE * sizeof(double);   // max volume component (L=64 padded)
    size_t tot = ((4 * one + (1 << 21) - 1) >> 21) << 21;
    char *blk = (char *)aligned_alloc(1 << 21, tot);
    memset(blk, 0, tot);
    Xre = (double *)blk; Xim = (double *)(blk + one);
    Cre = (double *)(blk + 2 * one); Cim = (double *)(blk + 3 * one);
  }
}

// ---------------------------------------------------------------- driver
#define RUN_CASE_SOA(LN, L_, R_, SS_, SP_)                          \
  case L_: {                                                         \
    long n = (long)L_ * L_ * L_;                                     \
    long b0 = 0;                                                     \
    while (B - b0 >= 8) {                                            \
      soa_load(L_, SP_, xin + b0 * 2 * n, Xre, Xim);                 \
      soa_load(L_, SP_, cin + b0 * 2 * n, Cre, Cim);                 \
      for (long it = 0; it < m; it++) {                              \
        soa_iter_##LN();                                             \
        if (it == 0) soa_store(L_, SP_, Xre, Xim, out1 + b0 * 2 * n);\
      }                                                              \
      soa_store(L_, SP_, Xre, Xim, outm + b0 * 2 * n);               \
      b0 += 8;                                                       \
    }                                                                \
    for (long b = b0; b < B; b++) {                                  \
      load_vol(L_, R_, SS_, xin + b * 2 * n, Xre, Xim);              \
      load_vol(L_, R_, SS_, cin + b * 2 * n, Cre, Cim);              \
      for (long it = 0; it < m; it++) {                              \
        sweepA_##LN();                                               \
        sweepB_##LN();                                               \
        if (it == 0) store_vol(L_, R_, SS_, Xre, Xim, out1 + b * 2 * n); \
      }                                                              \
      store_vol(L_, R_, SS_, Xre, Xim, outm + b * 2 * n);            \
    }                                                                \
  } break;

#define RUN_CASE(LN, L_, R_, SS_)                                  \
  case L_: {                                                         \
    long n = (long)L_ * L_ * L_;                                     \
    for (long b = 0; b < B; b++) {                                   \
      load_vol(L_, R_, SS_, xin + b * 2 * n, Xre, Xim);              \
      load_vol(L_, R_, SS_, cin + b * 2 * n, Cre, Cim);              \
      for (long it = 0; it < m; it++) {                              \
        sweepA_##LN();                                               \
        sweepB_##LN();                                               \
        if (it == 0) store_vol(L_, R_, SS_, Xre, Xim, out1 + b * 2 * n); \
      }                                                              \
      store_vol(L_, R_, SS_, Xre, Xim, outm + b * 2 * n);            \
    }                                                                \
  } break;

void mp_run(long L, long B, long m, const double *xin, const double *cin,
            double *out1, double *outm) {
  switch (L) {
    RUN_CASE_SOA(6, 6, 8, 72, 8 * (36 + 1))
    RUN_CASE_SOA(8, 8, 8, 72, 8 * (64 + 1))
    RUN_CASE_SOA(13, 13, 16, 208, 8 * (169 + 1))
    RUN_CASE_SOA(17, 17, 24, 408, 8 * (289 + 1))
    RUN_CASE(23, 23, 24, 552)
    RUN_CASE(36, 36, 40, 1440)
    RUN_CASE(45, 45, 48, 2312)
    RUN_CASE(64, 64, 72, 4616)
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
static void t6(double *r, double *i, long s, int d, const double *a, const double *b) { fft6_line(r, i, s, d, a, b, 0); }
static void t8(double *r, double *i, long s, int d, const double *a, const double *b) { fft8_line(r, i, s, d, a, b, 0); }
static void t13(double *r, double *i, long s, int d, const double *a, const double *b) { fft13_line(r, i, s, d, a, b, 0); }
static void t17(double *r, double *i, long s, int d, const double *a, const double *b) { fft17_line(r, i, s, d, a, b, 0); }
static void t23(double *r, double *i, long s, int d, const double *a, const double *b) { fft23_line(r, i, s, d, a, b, 0); }
static void t36(double *r, double *i, long s, int d, const double *a, const double *b) { fft36_line(r, i, s, d, a, b, 0); }
static void t45(double *r, double *i, long s, int d, const double *a, const double *b) { fft45_line(r, i, s, d, a, b, 0); }
static void t64(double *r, double *i, long s, int d, const double *a, const double *b) { fft64_line(r, i, s, d, a, b, 0); }

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
