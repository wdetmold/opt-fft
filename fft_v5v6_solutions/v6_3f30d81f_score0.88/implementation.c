// Iterated batched 3D complex FFT for fixed cube sizes 6,8,13,17,23,36,45,64.
// Hand-written AVX-512 specialization; no FFT library code anywhere.
//
// Layouts (all "paired split complex": logical vd slot q holds 8 real parts
// at memory vd index 2q and 8 imaginary parts at 2q+1):
//  * L in {6,8,13,17}: lanes = 8 volumes of the batch (AoSoA).  A block of 8
//    volumes is iterated m times while resident in L1/L2.
//  * L in {23,36,45}: in-volume, lanes = 8 consecutive z (z padded to ZP).
//    z-axis DFT via in-register 8x8 tile transposes; x-axis DFT runs directly
//    on strided lines (strides chosen to avoid 4K aliasing); the elementwise
//    map is fused into the x-pass, the next iteration's z-pass behind it.
//  * L = 64: in-volume, lanes = low 3 bits of x.  z- and y-axis DFTs are
//    plain vertical radix-8^2 kernels; the x-axis DFT is a four-step with one
//    in-register 8x8 transpose, 8 sequential streams, fused map, and the next
//    iteration's z-pass pipelined behind it.
// All twiddles are generated in long double with exact mod-L reduction.

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <sys/mman.h>

#if !defined(__AVX512F__)
#error "This implementation requires AVX-512F (compile with -march=native on the grading machine)"
#endif

typedef double vd __attribute__((vector_size(64), aligned(64)));
#define VL 8

// ------------------------------------------------------------------
// Tables
// ------------------------------------------------------------------
static double S3;                        // sin(2pi/3)
static double C8W;                       // sqrt(2)/2
static double C5_1, C5_2, S5_1, S5_2;    // cos/sin 2pi/5, 4pi/5
static double W9R[3][3], W9I[3][3];      // w9^{b*k1}
static double TW36R[6][6], TW36I[6][6];  // [b][k1] w36^{b*k1}
static double TW45R[9][5], TW45I[9][5];  // [b][k1] w45^{b*k1}
static double TW64R[8][8], TW64I[8][8];  // [b][k1] w64^{b*k1}
static vd     ZTWR[8], ZTWI[8];          // [k1], lane n2: w64^{n2*k1}
static double T13C[6][6],  T13S[6][6];
static double CW13[7],  SW13[7];    // cos/sin(2pi m/13), m=1..6
static double CW17[9],  SW17[9];
static double CW23[12], SW23[12];   // [k-1][j-1]: cos/sin(2pi jk/13)
static double T17C[8][8],  T17S[8][8];
static double T23C[11][11], T23S[11][11];

static long double angl(long long num, long long den) {
    const long double TWO_PI = 6.283185307179586476925286766559005768394L;
    long long r = num % den; if (r < 0) r += den;
    return -TWO_PI * (long double)r / (long double)den;
}

static vd *XP_, *CP_;   // state and constant, paired layout
static void *big_alloc(size_t bytes) {
    size_t sz = (bytes + (2u<<20) - 1) & ~(size_t)((2u<<20) - 1);
    void *p = mmap(NULL, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { void *q = 0; if (posix_memalign(&q, 64, bytes)) return 0; return q; }
    madvise(p, sz, MADV_HUGEPAGE);
    memset(p, 0, sz);
    return p;
}
#define ARENA_SLOTS 33856   /* 8*4232 padded slots for L=64 */

void init_tables(void) {
    const long double TWO_PI = 6.283185307179586476925286766559005768394L;
    if (!XP_) {
        XP_ = (vd*)big_alloc((size_t)2 * ARENA_SLOTS * sizeof(vd));
        CP_ = (vd*)big_alloc((size_t)2 * ARENA_SLOTS * sizeof(vd));
    }
    S3  = (double)sinl(TWO_PI / 3.0L);
    C8W = (double)cosl(TWO_PI / 8.0L);
    C5_1 = (double)cosl(TWO_PI / 5.0L);        S5_1 = (double)sinl(TWO_PI / 5.0L);
    C5_2 = (double)cosl(2.0L * TWO_PI / 5.0L); S5_2 = (double)sinl(2.0L * TWO_PI / 5.0L);
    for (int b = 0; b < 3; b++) for (int k = 0; k < 3; k++) {
        long double a = angl((long long)b * k, 9);
        W9R[b][k] = (double)cosl(a); W9I[b][k] = (double)sinl(a);
    }
    for (int b = 0; b < 6; b++) for (int k = 0; k < 6; k++) {
        long double a = angl((long long)b * k, 36);
        TW36R[b][k] = (double)cosl(a); TW36I[b][k] = (double)sinl(a);
    }
    for (int b = 0; b < 9; b++) for (int k = 0; k < 5; k++) {
        long double a = angl((long long)b * k, 45);
        TW45R[b][k] = (double)cosl(a); TW45I[b][k] = (double)sinl(a);
    }
    for (int b = 0; b < 8; b++) for (int k = 0; k < 8; k++) {
        long double a = angl((long long)b * k, 64);
        TW64R[b][k] = (double)cosl(a); TW64I[b][k] = (double)sinl(a);
    }
    for (int k = 0; k < 8; k++) for (int n2 = 0; n2 < 8; n2++) {
        long double a = angl((long long)n2 * k, 64);
        ((double*)&ZTWR[k])[n2] = (double)cosl(a);
        ((double*)&ZTWI[k])[n2] = (double)sinl(a);
    }
    for (int m = 1; m <= 6; m++) {
        long double a = -angl(m, 13);
        CW13[m] = (double)cosl(a); SW13[m] = (double)sinl(a);
    }
    for (int m = 1; m <= 8; m++) {
        long double a = -angl(m, 17);
        CW17[m] = (double)cosl(a); SW17[m] = (double)sinl(a);
    }
    for (int m = 1; m <= 11; m++) {
        long double a = -angl(m, 23);
        CW23[m] = (double)cosl(a); SW23[m] = (double)sinl(a);
    }
    for (int k = 1; k <= 6; k++) for (int j = 1; j <= 6; j++) {
        long double a = -angl((long long)j * k, 13);
        T13C[k-1][j-1] = (double)cosl(a); T13S[k-1][j-1] = (double)sinl(a);
    }
    for (int k = 1; k <= 8; k++) for (int j = 1; j <= 8; j++) {
        long double a = -angl((long long)j * k, 17);
        T17C[k-1][j-1] = (double)cosl(a); T17S[k-1][j-1] = (double)sinl(a);
    }
    for (int k = 1; k <= 11; k++) for (int j = 1; j <= 11; j++) {
        long double a = -angl((long long)j * k, 23);
        T23C[k-1][j-1] = (double)cosl(a); T23S[k-1][j-1] = (double)sinl(a);
    }
}

// ------------------------------------------------------------------
// Elementwise map x = z/(1+|z|)
// ------------------------------------------------------------------
#ifndef MAP_STYLE
#define MAP_STYLE 2   /* 0 = NR rsqrt+rcp, 1 = sqrt+div, 2 = sqrt + rcp-NR (hybrid) */
#endif
static inline void mapv(vd *zr, vd *zi) {
    vd r2 = (*zr) * (*zr) + (*zi) * (*zi);
#if MAP_STYLE == 1
    __m512d mag = _mm512_sqrt_pd((__m512d)r2);
    __m512d den = _mm512_add_pd(mag, _mm512_set1_pd(1.0));
    __m512d u = _mm512_div_pd(_mm512_set1_pd(1.0), den);
#elif MAP_STYLE == 2
    __m512d mag = _mm512_sqrt_pd((__m512d)r2);
    __m512d den = _mm512_add_pd(mag, _mm512_set1_pd(1.0));
    __m512d u = _mm512_rcp14_pd(den);
    u = _mm512_mul_pd(u, _mm512_fnmadd_pd(den, u, _mm512_set1_pd(2.0)));
    u = _mm512_mul_pd(u, _mm512_fnmadd_pd(den, u, _mm512_set1_pd(2.0)));
#else
    __m512d r2m = _mm512_max_pd((__m512d)r2, _mm512_set1_pd(1e-300));
    __m512d t = _mm512_rsqrt14_pd(r2m);
    __m512d hr = _mm512_mul_pd(r2m, _mm512_set1_pd(0.5));
    t = _mm512_mul_pd(t, _mm512_fnmadd_pd(hr, _mm512_mul_pd(t, t), _mm512_set1_pd(1.5)));
    t = _mm512_mul_pd(t, _mm512_fnmadd_pd(hr, _mm512_mul_pd(t, t), _mm512_set1_pd(1.5)));
    __m512d mag = _mm512_mul_pd(r2m, t);
    __m512d den = _mm512_add_pd(mag, _mm512_set1_pd(1.0));
    __m512d u = _mm512_rcp14_pd(den);
    u = _mm512_mul_pd(u, _mm512_fnmadd_pd(den, u, _mm512_set1_pd(2.0)));
    u = _mm512_mul_pd(u, _mm512_fnmadd_pd(den, u, _mm512_set1_pd(2.0)));
#endif
    *zr *= (vd)u; *zi *= (vd)u;
}

#define KSTORE(dstR, dstI, vr_, vi_, cR, cI, DOMAP) do { \
    vd _r = (vr_), _i = (vi_);                            \
    if (DOMAP) { _r += (cR); _i += (cI); mapv(&_r, &_i); } \
    (dstR) = _r; (dstI) = _i;                             \
} while (0)

// ------------------------------------------------------------------
// Micro-codelets
// ------------------------------------------------------------------
static inline void bf3(vd *x0r, vd *x0i, vd *x1r, vd *x1i, vd *x2r, vd *x2i) {
    vd tr = *x1r + *x2r, ti = *x1i + *x2i;
    vd ur = *x1r - *x2r, ui = *x1i - *x2i;
    vd mr = *x0r - 0.5 * tr, mi = *x0i - 0.5 * ti;
    vd sr = S3 * ur, si = S3 * ui;
    *x0r += tr; *x0i += ti;
    *x1r = mr + si; *x1i = mi - sr;
    *x2r = mr - si; *x2i = mi + sr;
}

static inline void bf5(vd *x0r, vd *x0i, vd *x1r, vd *x1i, vd *x2r, vd *x2i,
                       vd *x3r, vd *x3i, vd *x4r, vd *x4i) {
    vd t1r = *x1r + *x4r, t1i = *x1i + *x4i;
    vd t4r = *x1r - *x4r, t4i = *x1i - *x4i;
    vd t2r = *x2r + *x3r, t2i = *x2i + *x3i;
    vd t3r = *x2r - *x3r, t3i = *x2i - *x3i;
    vd z0r = *x0r, z0i = *x0i;
    *x0r = z0r + t1r + t2r; *x0i = z0i + t1i + t2i;
    vd Pr = z0r + C5_1 * t1r + C5_2 * t2r;
    vd Pi = z0i + C5_1 * t1i + C5_2 * t2i;
    vd Qr = S5_1 * t4r + S5_2 * t3r;
    vd Qi = S5_1 * t4i + S5_2 * t3i;
    *x1r = Pr + Qi; *x1i = Pi - Qr;
    *x4r = Pr - Qi; *x4i = Pi + Qr;
    vd Rr = z0r + C5_2 * t1r + C5_1 * t2r;
    vd Ri = z0i + C5_2 * t1i + C5_1 * t2i;
    vd Sr = S5_2 * t4r - S5_1 * t3r;
    vd Si = S5_2 * t4i - S5_1 * t3i;
    *x2r = Rr + Si; *x2i = Ri - Sr;
    *x3r = Rr - Si; *x3i = Ri + Sr;
}

static inline void bf6(vd *x0r, vd *x0i, vd *x1r, vd *x1i, vd *x2r, vd *x2i,
                       vd *x3r, vd *x3i, vd *x4r, vd *x4i, vd *x5r, vd *x5i) {
    // PFA 2x3: no twiddle multiplies
    vd s0r = *x0r + *x3r, s0i = *x0i + *x3i;
    vd d0r = *x0r - *x3r, d0i = *x0i - *x3i;
    vd s1r = *x2r + *x5r, s1i = *x2i + *x5i;
    vd d1r = *x2r - *x5r, d1i = *x2i - *x5i;
    vd s2r = *x4r + *x1r, s2i = *x4i + *x1i;
    vd d2r = *x4r - *x1r, d2i = *x4i - *x1i;
    bf3(&s0r, &s0i, &s1r, &s1i, &s2r, &s2i);
    bf3(&d0r, &d0i, &d1r, &d1i, &d2r, &d2i);
    *x0r = s0r; *x0i = s0i;   // k=0
    *x4r = s1r; *x4i = s1i;   // k=4
    *x2r = s2r; *x2i = s2i;   // k=2
    *x3r = d0r; *x3i = d0i;   // k=3
    *x1r = d1r; *x1i = d1i;   // k=1
    *x5r = d2r; *x5i = d2i;   // k=5
}

static inline void bf8(vd *x0r, vd *x0i, vd *x1r, vd *x1i, vd *x2r, vd *x2i,
                       vd *x3r, vd *x3i, vd *x4r, vd *x4i, vd *x5r, vd *x5i,
                       vd *x6r, vd *x6i, vd *x7r, vd *x7i) {
    vd t0r = *x0r + *x4r, t0i = *x0i + *x4i;
    vd t1r = *x0r - *x4r, t1i = *x0i - *x4i;
    vd t2r = *x2r + *x6r, t2i = *x2i + *x6i;
    vd t3r = *x2r - *x6r, t3i = *x2i - *x6i;
    vd E0r = t0r + t2r, E0i = t0i + t2i;
    vd E2r = t0r - t2r, E2i = t0i - t2i;
    vd E1r = t1r + t3i, E1i = t1i - t3r;
    vd E3r = t1r - t3i, E3i = t1i + t3r;
    vd u0r = *x1r + *x5r, u0i = *x1i + *x5i;
    vd u1r = *x1r - *x5r, u1i = *x1i - *x5i;
    vd u2r = *x3r + *x7r, u2i = *x3i + *x7i;
    vd u3r = *x3r - *x7r, u3i = *x3i - *x7i;
    vd O0r = u0r + u2r, O0i = u0i + u2i;
    vd O2r = u0r - u2r, O2i = u0i - u2i;
    vd O1r = u1r + u3i, O1i = u1i - u3r;
    vd O3r = u1r - u3i, O3i = u1i + u3r;
    vd W1r = C8W * (O1r + O1i), W1i = C8W * (O1i - O1r);
    vd W2r = O2i, W2i = -O2r;
    vd W3r = C8W * (O3i - O3r), W3i = -C8W * (O3r + O3i);
    *x0r = E0r + O0r; *x0i = E0i + O0i;
    *x4r = E0r - O0r; *x4i = E0i - O0i;
    *x1r = E1r + W1r; *x1i = E1i + W1i;
    *x5r = E1r - W1r; *x5i = E1i - W1i;
    *x2r = E2r + W2r; *x2i = E2i + W2i;
    *x6r = E2r - W2r; *x6i = E2i - W2i;
    *x3r = E3r + W3r; *x3i = E3i + W3i;
    *x7r = E3r - W3r; *x7i = E3i - W3i;
}

// ------------------------------------------------------------------
// Line kernels: xr/xi pointers with stride s, all in vd units (memory).
// For the paired layout call with xi = xr+1 and s = 2*slotstride.
// ------------------------------------------------------------------
static __attribute__((always_inline)) inline
void k6(vd *restrict xr, vd *restrict xi, ptrdiff_t s,
        const vd *restrict cr, const vd *restrict ci, const int domap) {
    vd r0 = xr[0],   i0 = xi[0];
    vd r1 = xr[s],   i1 = xi[s];
    vd r2 = xr[2*s], i2 = xi[2*s];
    vd r3 = xr[3*s], i3 = xi[3*s];
    vd r4 = xr[4*s], i4 = xi[4*s];
    vd r5 = xr[5*s], i5 = xi[5*s];
    bf6(&r0,&i0,&r1,&i1,&r2,&i2,&r3,&i3,&r4,&i4,&r5,&i5);
    KSTORE(xr[0],   xi[0],   r0, i0, cr[0],   ci[0],   domap);
    KSTORE(xr[s],   xi[s],   r1, i1, cr[s],   ci[s],   domap);
    KSTORE(xr[2*s], xi[2*s], r2, i2, cr[2*s], ci[2*s], domap);
    KSTORE(xr[3*s], xi[3*s], r3, i3, cr[3*s], ci[3*s], domap);
    KSTORE(xr[4*s], xi[4*s], r4, i4, cr[4*s], ci[4*s], domap);
    KSTORE(xr[5*s], xi[5*s], r5, i5, cr[5*s], ci[5*s], domap);
}

static __attribute__((always_inline)) inline
void k8(vd *restrict xr, vd *restrict xi, ptrdiff_t s,
        const vd *restrict cr, const vd *restrict ci, const int domap) {
    vd r0 = xr[0],   i0 = xi[0];
    vd r1 = xr[s],   i1 = xi[s];
    vd r2 = xr[2*s], i2 = xi[2*s];
    vd r3 = xr[3*s], i3 = xi[3*s];
    vd r4 = xr[4*s], i4 = xi[4*s];
    vd r5 = xr[5*s], i5 = xi[5*s];
    vd r6 = xr[6*s], i6 = xi[6*s];
    vd r7 = xr[7*s], i7 = xi[7*s];
    bf8(&r0,&i0,&r1,&i1,&r2,&i2,&r3,&i3,&r4,&i4,&r5,&i5,&r6,&i6,&r7,&i7);
    KSTORE(xr[0],   xi[0],   r0, i0, cr[0],   ci[0],   domap);
    KSTORE(xr[s],   xi[s],   r1, i1, cr[s],   ci[s],   domap);
    KSTORE(xr[2*s], xi[2*s], r2, i2, cr[2*s], ci[2*s], domap);
    KSTORE(xr[3*s], xi[3*s], r3, i3, cr[3*s], ci[3*s], domap);
    KSTORE(xr[4*s], xi[4*s], r4, i4, cr[4*s], ci[4*s], domap);
    KSTORE(xr[5*s], xi[5*s], r5, i5, cr[5*s], ci[5*s], domap);
    KSTORE(xr[6*s], xi[6*s], r6, i6, cr[6*s], ci[6*s], domap);
    KSTORE(xr[7*s], xi[7*s], r7, i7, cr[7*s], ci[7*s], domap);
}

// Odd-prime DFT via symmetric conjugate pairs (k blocked by 2, generated code,
// (p-1)/2 distinct twiddle magnitudes kept in registers with sign-folded FMAs).
#define SPLAT(x) ((vd)_mm512_set1_pd(x))

// generated: 13-point DFT, k-block 2, twiddles: both
static __attribute__((always_inline)) inline
void k13(vd *restrict xr, vd *restrict xi, ptrdiff_t s,
          const vd *restrict cr, const vd *restrict ci, const int domap) {
    vd ur[7], ui[7], vr[7], vi[7];
    vd x0r = xr[0], x0i = xi[0];
    vd s0r = x0r, s0i = x0i, s1r = (vd){0}, s1i = (vd){0};
    _Pragma("GCC unroll 16")
    for (int j = 1; j <= 6; j++) {
        vd ar = xr[(ptrdiff_t)j*s],      ai = xi[(ptrdiff_t)j*s];
        vd br = xr[(ptrdiff_t)(13-j)*s], bi = xi[(ptrdiff_t)(13-j)*s];
        ur[j] = ar + br; ui[j] = ai + bi;
        vr[j] = ar - br; vi[j] = ai - bi;
        if (j & 1) { s0r += ur[j]; s0i += ui[j]; }
        else       { s1r += ur[j]; s1i += ui[j]; }
    }
    KSTORE(xr[0], xi[0], s0r + s1r, s0i + s1i, cr[0], ci[0], domap);
    const vd cw1 = SPLAT(CW13[1]), sw1 = SPLAT(SW13[1]);
    const vd cw2 = SPLAT(CW13[2]), sw2 = SPLAT(SW13[2]);
    const vd cw3 = SPLAT(CW13[3]), sw3 = SPLAT(SW13[3]);
    const vd cw4 = SPLAT(CW13[4]), sw4 = SPLAT(SW13[4]);
    const vd cw5 = SPLAT(CW13[5]), sw5 = SPLAT(SW13[5]);
    const vd cw6 = SPLAT(CW13[6]), sw6 = SPLAT(SW13[6]);
    { // k = 1..2
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          A0 += u*cw1; C0 += iu*cw1; B0 += iv*sw1; D0 += v*sw1;
          A1 += u*cw2; C1 += iu*cw2; B1 += iv*sw2; D1 += v*sw2;
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          A0 += u*cw2; C0 += iu*cw2; B0 += iv*sw2; D0 += v*sw2;
          A1 += u*cw4; C1 += iu*cw4; B1 += iv*sw4; D1 += v*sw4;
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          A0 += u*cw3; C0 += iu*cw3; B0 += iv*sw3; D0 += v*sw3;
          A1 += u*cw6; C1 += iu*cw6; B1 += iv*sw6; D1 += v*sw6;
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          A0 += u*cw4; C0 += iu*cw4; B0 += iv*sw4; D0 += v*sw4;
          A1 += u*cw5; C1 += iu*cw5; B1 -= iv*sw5; D1 -= v*sw5;
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          A0 += u*cw5; C0 += iu*cw5; B0 += iv*sw5; D0 += v*sw5;
          A1 += u*cw3; C1 += iu*cw3; B1 -= iv*sw3; D1 -= v*sw3;
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          A0 += u*cw6; C0 += iu*cw6; B0 += iv*sw6; D0 += v*sw6;
          A1 += u*cw1; C1 += iu*cw1; B1 -= iv*sw1; D1 -= v*sw1;
        }
        { ptrdiff_t ka = (ptrdiff_t)1*s, kb2 = (ptrdiff_t)12*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)2*s, kb2 = (ptrdiff_t)11*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
    { // k = 3..4
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          A0 += u*cw3; C0 += iu*cw3; B0 += iv*sw3; D0 += v*sw3;
          A1 += u*cw4; C1 += iu*cw4; B1 += iv*sw4; D1 += v*sw4;
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          A0 += u*cw6; C0 += iu*cw6; B0 += iv*sw6; D0 += v*sw6;
          A1 += u*cw5; C1 += iu*cw5; B1 -= iv*sw5; D1 -= v*sw5;
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          A0 += u*cw4; C0 += iu*cw4; B0 -= iv*sw4; D0 -= v*sw4;
          A1 += u*cw1; C1 += iu*cw1; B1 -= iv*sw1; D1 -= v*sw1;
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          A0 += u*cw1; C0 += iu*cw1; B0 -= iv*sw1; D0 -= v*sw1;
          A1 += u*cw3; C1 += iu*cw3; B1 += iv*sw3; D1 += v*sw3;
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          A0 += u*cw2; C0 += iu*cw2; B0 += iv*sw2; D0 += v*sw2;
          A1 += u*cw6; C1 += iu*cw6; B1 -= iv*sw6; D1 -= v*sw6;
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          A0 += u*cw5; C0 += iu*cw5; B0 += iv*sw5; D0 += v*sw5;
          A1 += u*cw2; C1 += iu*cw2; B1 -= iv*sw2; D1 -= v*sw2;
        }
        { ptrdiff_t ka = (ptrdiff_t)3*s, kb2 = (ptrdiff_t)10*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)4*s, kb2 = (ptrdiff_t)9*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
    { // k = 5..6
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          A0 += u*cw5; C0 += iu*cw5; B0 += iv*sw5; D0 += v*sw5;
          A1 += u*cw6; C1 += iu*cw6; B1 += iv*sw6; D1 += v*sw6;
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          A0 += u*cw3; C0 += iu*cw3; B0 -= iv*sw3; D0 -= v*sw3;
          A1 += u*cw1; C1 += iu*cw1; B1 -= iv*sw1; D1 -= v*sw1;
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          A0 += u*cw2; C0 += iu*cw2; B0 += iv*sw2; D0 += v*sw2;
          A1 += u*cw5; C1 += iu*cw5; B1 += iv*sw5; D1 += v*sw5;
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          A0 += u*cw6; C0 += iu*cw6; B0 -= iv*sw6; D0 -= v*sw6;
          A1 += u*cw2; C1 += iu*cw2; B1 -= iv*sw2; D1 -= v*sw2;
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          A0 += u*cw1; C0 += iu*cw1; B0 -= iv*sw1; D0 -= v*sw1;
          A1 += u*cw4; C1 += iu*cw4; B1 += iv*sw4; D1 += v*sw4;
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          A0 += u*cw4; C0 += iu*cw4; B0 += iv*sw4; D0 += v*sw4;
          A1 += u*cw3; C1 += iu*cw3; B1 -= iv*sw3; D1 -= v*sw3;
        }
        { ptrdiff_t ka = (ptrdiff_t)5*s, kb2 = (ptrdiff_t)8*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)6*s, kb2 = (ptrdiff_t)7*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
}

// generated: 17-point DFT, k-block 3, twiddles: cos
static __attribute__((always_inline)) inline
void k17(vd *restrict xr, vd *restrict xi, ptrdiff_t s,
          const vd *restrict cr, const vd *restrict ci, const int domap) {
    vd ur[9], ui[9], vr[9], vi[9];
    vd x0r = xr[0], x0i = xi[0];
    vd s0r = x0r, s0i = x0i, s1r = (vd){0}, s1i = (vd){0};
    _Pragma("GCC unroll 16")
    for (int j = 1; j <= 8; j++) {
        vd ar = xr[(ptrdiff_t)j*s],      ai = xi[(ptrdiff_t)j*s];
        vd br = xr[(ptrdiff_t)(17-j)*s], bi = xi[(ptrdiff_t)(17-j)*s];
        ur[j] = ar + br; ui[j] = ai + bi;
        vr[j] = ar - br; vi[j] = ai - bi;
        if (j & 1) { s0r += ur[j]; s0i += ui[j]; }
        else       { s1r += ur[j]; s1i += ui[j]; }
    }
    KSTORE(xr[0], xi[0], s0r + s1r, s0i + s1i, cr[0], ci[0], domap);
    const vd cw1 = SPLAT(CW17[1]);
    const vd cw2 = SPLAT(CW17[2]);
    const vd cw3 = SPLAT(CW17[3]);
    const vd cw4 = SPLAT(CW17[4]);
    const vd cw5 = SPLAT(CW17[5]);
    const vd cw6 = SPLAT(CW17[6]);
    const vd cw7 = SPLAT(CW17[7]);
    const vd cw8 = SPLAT(CW17[8]);
    { // k = 1..3
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        vd A2 = x0r, C2 = x0i, B2 = (vd){0}, D2 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW17[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[2]);
            A1 += u*cw2; C1 += iu*cw2; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[3]);
            A2 += u*cw3; C2 += iu*cw3; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW17[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[4]);
            A1 += u*cw4; C1 += iu*cw4; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[6]);
            A2 += u*cw6; C2 += iu*cw6; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW17[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[6]);
            A1 += u*cw6; C1 += iu*cw6; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[8]);
            A2 += u*cw8; C2 += iu*cw8; B2 -= iv*sb; D2 -= v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW17[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[8]);
            A1 += u*cw8; C1 += iu*cw8; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[5]);
            A2 += u*cw5; C2 += iu*cw5; B2 -= iv*sb; D2 -= v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW17[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[7]);
            A1 += u*cw7; C1 += iu*cw7; B1 -= iv*sb; D1 -= v*sb; }
          { vd sb = SPLAT(SW17[2]);
            A2 += u*cw2; C2 += iu*cw2; B2 -= iv*sb; D2 -= v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW17[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[5]);
            A1 += u*cw5; C1 += iu*cw5; B1 -= iv*sb; D1 -= v*sb; }
          { vd sb = SPLAT(SW17[1]);
            A2 += u*cw1; C2 += iu*cw1; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW17[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[3]);
            A1 += u*cw3; C1 += iu*cw3; B1 -= iv*sb; D1 -= v*sb; }
          { vd sb = SPLAT(SW17[4]);
            A2 += u*cw4; C2 += iu*cw4; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW17[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[1]);
            A1 += u*cw1; C1 += iu*cw1; B1 -= iv*sb; D1 -= v*sb; }
          { vd sb = SPLAT(SW17[7]);
            A2 += u*cw7; C2 += iu*cw7; B2 += iv*sb; D2 += v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)1*s, kb2 = (ptrdiff_t)16*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)2*s, kb2 = (ptrdiff_t)15*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)3*s, kb2 = (ptrdiff_t)14*s;
          KSTORE(xr[ka], xi[ka], A2 + B2, C2 - D2, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A2 - B2, C2 + D2, cr[kb2], ci[kb2], domap); }
    }
    { // k = 4..6
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        vd A2 = x0r, C2 = x0i, B2 = (vd){0}, D2 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW17[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[5]);
            A1 += u*cw5; C1 += iu*cw5; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[6]);
            A2 += u*cw6; C2 += iu*cw6; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW17[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[7]);
            A1 += u*cw7; C1 += iu*cw7; B1 -= iv*sb; D1 -= v*sb; }
          { vd sb = SPLAT(SW17[5]);
            A2 += u*cw5; C2 += iu*cw5; B2 -= iv*sb; D2 -= v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW17[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW17[2]);
            A1 += u*cw2; C1 += iu*cw2; B1 -= iv*sb; D1 -= v*sb; }
          { vd sb = SPLAT(SW17[1]);
            A2 += u*cw1; C2 += iu*cw1; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW17[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW17[3]);
            A1 += u*cw3; C1 += iu*cw3; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[7]);
            A2 += u*cw7; C2 += iu*cw7; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW17[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[8]);
            A1 += u*cw8; C1 += iu*cw8; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[4]);
            A2 += u*cw4; C2 += iu*cw4; B2 -= iv*sb; D2 -= v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW17[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[4]);
            A1 += u*cw4; C1 += iu*cw4; B1 -= iv*sb; D1 -= v*sb; }
          { vd sb = SPLAT(SW17[2]);
            A2 += u*cw2; C2 += iu*cw2; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW17[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW17[1]);
            A1 += u*cw1; C1 += iu*cw1; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[8]);
            A2 += u*cw8; C2 += iu*cw8; B2 += iv*sb; D2 += v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW17[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW17[6]);
            A1 += u*cw6; C1 += iu*cw6; B1 += iv*sb; D1 += v*sb; }
          { vd sb = SPLAT(SW17[3]);
            A2 += u*cw3; C2 += iu*cw3; B2 -= iv*sb; D2 -= v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)4*s, kb2 = (ptrdiff_t)13*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)5*s, kb2 = (ptrdiff_t)12*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)6*s, kb2 = (ptrdiff_t)11*s;
          KSTORE(xr[ka], xi[ka], A2 + B2, C2 - D2, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A2 - B2, C2 + D2, cr[kb2], ci[kb2], domap); }
    }
    { // k = 7..8
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW17[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[8]);
            A1 += u*cw8; C1 += iu*cw8; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW17[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW17[1]);
            A1 += u*cw1; C1 += iu*cw1; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW17[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[7]);
            A1 += u*cw7; C1 += iu*cw7; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW17[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW17[2]);
            A1 += u*cw2; C1 += iu*cw2; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW17[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[6]);
            A1 += u*cw6; C1 += iu*cw6; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW17[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[3]);
            A1 += u*cw3; C1 += iu*cw3; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW17[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW17[5]);
            A1 += u*cw5; C1 += iu*cw5; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW17[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW17[4]);
            A1 += u*cw4; C1 += iu*cw4; B1 -= iv*sb; D1 -= v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)7*s, kb2 = (ptrdiff_t)10*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)8*s, kb2 = (ptrdiff_t)9*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
}

// generated: 23-point DFT, k-block 2, twiddles: cos
static __attribute__((always_inline)) inline
void k23(vd *restrict xr, vd *restrict xi, ptrdiff_t s,
          const vd *restrict cr, const vd *restrict ci, const int domap) {
    vd ur[12], ui[12], vr[12], vi[12];
    vd x0r = xr[0], x0i = xi[0];
    vd s0r = x0r, s0i = x0i, s1r = (vd){0}, s1i = (vd){0};
    _Pragma("GCC unroll 16")
    for (int j = 1; j <= 11; j++) {
        vd ar = xr[(ptrdiff_t)j*s],      ai = xi[(ptrdiff_t)j*s];
        vd br = xr[(ptrdiff_t)(23-j)*s], bi = xi[(ptrdiff_t)(23-j)*s];
        ur[j] = ar + br; ui[j] = ai + bi;
        vr[j] = ar - br; vi[j] = ai - bi;
        if (j & 1) { s0r += ur[j]; s0i += ui[j]; }
        else       { s1r += ur[j]; s1i += ui[j]; }
    }
    KSTORE(xr[0], xi[0], s0r + s1r, s0i + s1i, cr[0], ci[0], domap);
    const vd cw1 = SPLAT(CW23[1]);
    const vd cw2 = SPLAT(CW23[2]);
    const vd cw3 = SPLAT(CW23[3]);
    const vd cw4 = SPLAT(CW23[4]);
    const vd cw5 = SPLAT(CW23[5]);
    const vd cw6 = SPLAT(CW23[6]);
    const vd cw7 = SPLAT(CW23[7]);
    const vd cw8 = SPLAT(CW23[8]);
    const vd cw9 = SPLAT(CW23[9]);
    const vd cw10 = SPLAT(CW23[10]);
    const vd cw11 = SPLAT(CW23[11]);
    { // k = 1..2
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW23[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[2]);
            A1 += u*cw2; C1 += iu*cw2; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW23[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[4]);
            A1 += u*cw4; C1 += iu*cw4; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW23[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[6]);
            A1 += u*cw6; C1 += iu*cw6; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW23[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[8]);
            A1 += u*cw8; C1 += iu*cw8; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW23[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[10]);
            A1 += u*cw10; C1 += iu*cw10; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW23[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[11]);
            A1 += u*cw11; C1 += iu*cw11; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW23[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[9]);
            A1 += u*cw9; C1 += iu*cw9; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW23[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[7]);
            A1 += u*cw7; C1 += iu*cw7; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[9], iu = ui[9], v = vr[9], iv = vi[9];
          { vd sb = SPLAT(SW23[9]);
            A0 += u*cw9; C0 += iu*cw9; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[5]);
            A1 += u*cw5; C1 += iu*cw5; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[10], iu = ui[10], v = vr[10], iv = vi[10];
          { vd sb = SPLAT(SW23[10]);
            A0 += u*cw10; C0 += iu*cw10; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[3]);
            A1 += u*cw3; C1 += iu*cw3; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[11], iu = ui[11], v = vr[11], iv = vi[11];
          { vd sb = SPLAT(SW23[11]);
            A0 += u*cw11; C0 += iu*cw11; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[1]);
            A1 += u*cw1; C1 += iu*cw1; B1 -= iv*sb; D1 -= v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)1*s, kb2 = (ptrdiff_t)22*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)2*s, kb2 = (ptrdiff_t)21*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
    { // k = 3..4
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW23[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[4]);
            A1 += u*cw4; C1 += iu*cw4; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW23[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[8]);
            A1 += u*cw8; C1 += iu*cw8; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW23[9]);
            A0 += u*cw9; C0 += iu*cw9; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[11]);
            A1 += u*cw11; C1 += iu*cw11; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW23[11]);
            A0 += u*cw11; C0 += iu*cw11; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[7]);
            A1 += u*cw7; C1 += iu*cw7; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW23[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[3]);
            A1 += u*cw3; C1 += iu*cw3; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW23[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[1]);
            A1 += u*cw1; C1 += iu*cw1; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW23[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[5]);
            A1 += u*cw5; C1 += iu*cw5; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW23[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[9]);
            A1 += u*cw9; C1 += iu*cw9; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[9], iu = ui[9], v = vr[9], iv = vi[9];
          { vd sb = SPLAT(SW23[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[10]);
            A1 += u*cw10; C1 += iu*cw10; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[10], iu = ui[10], v = vr[10], iv = vi[10];
          { vd sb = SPLAT(SW23[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[6]);
            A1 += u*cw6; C1 += iu*cw6; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[11], iu = ui[11], v = vr[11], iv = vi[11];
          { vd sb = SPLAT(SW23[10]);
            A0 += u*cw10; C0 += iu*cw10; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[2]);
            A1 += u*cw2; C1 += iu*cw2; B1 -= iv*sb; D1 -= v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)3*s, kb2 = (ptrdiff_t)20*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)4*s, kb2 = (ptrdiff_t)19*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
    { // k = 5..6
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW23[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[6]);
            A1 += u*cw6; C1 += iu*cw6; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW23[10]);
            A0 += u*cw10; C0 += iu*cw10; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[11]);
            A1 += u*cw11; C1 += iu*cw11; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW23[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[5]);
            A1 += u*cw5; C1 += iu*cw5; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW23[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[1]);
            A1 += u*cw1; C1 += iu*cw1; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW23[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[7]);
            A1 += u*cw7; C1 += iu*cw7; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW23[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[10]);
            A1 += u*cw10; C1 += iu*cw10; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW23[11]);
            A0 += u*cw11; C0 += iu*cw11; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[4]);
            A1 += u*cw4; C1 += iu*cw4; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW23[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[2]);
            A1 += u*cw2; C1 += iu*cw2; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[9], iu = ui[9], v = vr[9], iv = vi[9];
          { vd sb = SPLAT(SW23[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[8]);
            A1 += u*cw8; C1 += iu*cw8; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[10], iu = ui[10], v = vr[10], iv = vi[10];
          { vd sb = SPLAT(SW23[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[9]);
            A1 += u*cw9; C1 += iu*cw9; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[11], iu = ui[11], v = vr[11], iv = vi[11];
          { vd sb = SPLAT(SW23[9]);
            A0 += u*cw9; C0 += iu*cw9; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[3]);
            A1 += u*cw3; C1 += iu*cw3; B1 -= iv*sb; D1 -= v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)5*s, kb2 = (ptrdiff_t)18*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)6*s, kb2 = (ptrdiff_t)17*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
    { // k = 7..8
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW23[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[8]);
            A1 += u*cw8; C1 += iu*cw8; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW23[9]);
            A0 += u*cw9; C0 += iu*cw9; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[7]);
            A1 += u*cw7; C1 += iu*cw7; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW23[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[1]);
            A1 += u*cw1; C1 += iu*cw1; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW23[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[9]);
            A1 += u*cw9; C1 += iu*cw9; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW23[11]);
            A0 += u*cw11; C0 += iu*cw11; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[6]);
            A1 += u*cw6; C1 += iu*cw6; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW23[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[2]);
            A1 += u*cw2; C1 += iu*cw2; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW23[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[10]);
            A1 += u*cw10; C1 += iu*cw10; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW23[10]);
            A0 += u*cw10; C0 += iu*cw10; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[5]);
            A1 += u*cw5; C1 += iu*cw5; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[9], iu = ui[9], v = vr[9], iv = vi[9];
          { vd sb = SPLAT(SW23[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[3]);
            A1 += u*cw3; C1 += iu*cw3; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[10], iu = ui[10], v = vr[10], iv = vi[10];
          { vd sb = SPLAT(SW23[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[11]);
            A1 += u*cw11; C1 += iu*cw11; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[11], iu = ui[11], v = vr[11], iv = vi[11];
          { vd sb = SPLAT(SW23[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[4]);
            A1 += u*cw4; C1 += iu*cw4; B1 -= iv*sb; D1 -= v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)7*s, kb2 = (ptrdiff_t)16*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)8*s, kb2 = (ptrdiff_t)15*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
    { // k = 9..10
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        vd A1 = x0r, C1 = x0i, B1 = (vd){0}, D1 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW23[9]);
            A0 += u*cw9; C0 += iu*cw9; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[10]);
            A1 += u*cw10; C1 += iu*cw10; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW23[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[3]);
            A1 += u*cw3; C1 += iu*cw3; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW23[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[7]);
            A1 += u*cw7; C1 += iu*cw7; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW23[10]);
            A0 += u*cw10; C0 += iu*cw10; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[6]);
            A1 += u*cw6; C1 += iu*cw6; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW23[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[4]);
            A1 += u*cw4; C1 += iu*cw4; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW23[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[9]);
            A1 += u*cw9; C1 += iu*cw9; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW23[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[1]);
            A1 += u*cw1; C1 += iu*cw1; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW23[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[11]);
            A1 += u*cw11; C1 += iu*cw11; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[9], iu = ui[9], v = vr[9], iv = vi[9];
          { vd sb = SPLAT(SW23[11]);
            A0 += u*cw11; C0 += iu*cw11; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[2]);
            A1 += u*cw2; C1 += iu*cw2; B1 -= iv*sb; D1 -= v*sb; }
        }
        { vd u = ur[10], iu = ui[10], v = vr[10], iv = vi[10];
          { vd sb = SPLAT(SW23[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 -= iv*sb; D0 -= v*sb; }
          { vd sb = SPLAT(SW23[8]);
            A1 += u*cw8; C1 += iu*cw8; B1 += iv*sb; D1 += v*sb; }
        }
        { vd u = ur[11], iu = ui[11], v = vr[11], iv = vi[11];
          { vd sb = SPLAT(SW23[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
          { vd sb = SPLAT(SW23[5]);
            A1 += u*cw5; C1 += iu*cw5; B1 -= iv*sb; D1 -= v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)9*s, kb2 = (ptrdiff_t)14*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
        { ptrdiff_t ka = (ptrdiff_t)10*s, kb2 = (ptrdiff_t)13*s;
          KSTORE(xr[ka], xi[ka], A1 + B1, C1 - D1, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A1 - B1, C1 + D1, cr[kb2], ci[kb2], domap); }
    }
    { // k = 11..11
        vd A0 = x0r, C0 = x0i, B0 = (vd){0}, D0 = (vd){0};
        { vd u = ur[1], iu = ui[1], v = vr[1], iv = vi[1];
          { vd sb = SPLAT(SW23[11]);
            A0 += u*cw11; C0 += iu*cw11; B0 += iv*sb; D0 += v*sb; }
        }
        { vd u = ur[2], iu = ui[2], v = vr[2], iv = vi[2];
          { vd sb = SPLAT(SW23[1]);
            A0 += u*cw1; C0 += iu*cw1; B0 -= iv*sb; D0 -= v*sb; }
        }
        { vd u = ur[3], iu = ui[3], v = vr[3], iv = vi[3];
          { vd sb = SPLAT(SW23[10]);
            A0 += u*cw10; C0 += iu*cw10; B0 += iv*sb; D0 += v*sb; }
        }
        { vd u = ur[4], iu = ui[4], v = vr[4], iv = vi[4];
          { vd sb = SPLAT(SW23[2]);
            A0 += u*cw2; C0 += iu*cw2; B0 -= iv*sb; D0 -= v*sb; }
        }
        { vd u = ur[5], iu = ui[5], v = vr[5], iv = vi[5];
          { vd sb = SPLAT(SW23[9]);
            A0 += u*cw9; C0 += iu*cw9; B0 += iv*sb; D0 += v*sb; }
        }
        { vd u = ur[6], iu = ui[6], v = vr[6], iv = vi[6];
          { vd sb = SPLAT(SW23[3]);
            A0 += u*cw3; C0 += iu*cw3; B0 -= iv*sb; D0 -= v*sb; }
        }
        { vd u = ur[7], iu = ui[7], v = vr[7], iv = vi[7];
          { vd sb = SPLAT(SW23[8]);
            A0 += u*cw8; C0 += iu*cw8; B0 += iv*sb; D0 += v*sb; }
        }
        { vd u = ur[8], iu = ui[8], v = vr[8], iv = vi[8];
          { vd sb = SPLAT(SW23[4]);
            A0 += u*cw4; C0 += iu*cw4; B0 -= iv*sb; D0 -= v*sb; }
        }
        { vd u = ur[9], iu = ui[9], v = vr[9], iv = vi[9];
          { vd sb = SPLAT(SW23[7]);
            A0 += u*cw7; C0 += iu*cw7; B0 += iv*sb; D0 += v*sb; }
        }
        { vd u = ur[10], iu = ui[10], v = vr[10], iv = vi[10];
          { vd sb = SPLAT(SW23[5]);
            A0 += u*cw5; C0 += iu*cw5; B0 -= iv*sb; D0 -= v*sb; }
        }
        { vd u = ur[11], iu = ui[11], v = vr[11], iv = vi[11];
          { vd sb = SPLAT(SW23[6]);
            A0 += u*cw6; C0 += iu*cw6; B0 += iv*sb; D0 += v*sb; }
        }
        { ptrdiff_t ka = (ptrdiff_t)11*s, kb2 = (ptrdiff_t)12*s;
          KSTORE(xr[ka], xi[ka], A0 + B0, C0 - D0, cr[ka], ci[ka], domap);
          KSTORE(xr[kb2], xi[kb2], A0 - B0, C0 + D0, cr[kb2], ci[kb2], domap); }
    }
}

static __attribute__((always_inline)) inline
void k36(vd *restrict xr, vd *restrict xi, ptrdiff_t s,
         const vd *restrict cr, const vd *restrict ci, const int domap) {
    vd br[36], bi[36];
    for (int b = 0; b < 6; b++) {
        vd r0 = xr[b*s],        i0 = xi[b*s];
        vd r1 = xr[(6+b)*s],    i1 = xi[(6+b)*s];
        vd r2 = xr[(12+b)*s],   i2 = xi[(12+b)*s];
        vd r3 = xr[(18+b)*s],   i3 = xi[(18+b)*s];
        vd r4 = xr[(24+b)*s],   i4 = xi[(24+b)*s];
        vd r5 = xr[(30+b)*s],   i5 = xi[(30+b)*s];
        bf6(&r0,&i0,&r1,&i1,&r2,&i2,&r3,&i3,&r4,&i4,&r5,&i5);
        br[b] = r0; bi[b] = i0;
        if (b == 0) {
            br[6]=r1; bi[6]=i1; br[12]=r2; bi[12]=i2;
            br[18]=r3; bi[18]=i3; br[24]=r4; bi[24]=i4;
            br[30]=r5; bi[30]=i5;
        } else {
            br[6+b]  = r1*TW36R[b][1] - i1*TW36I[b][1];  bi[6+b]  = r1*TW36I[b][1] + i1*TW36R[b][1];
            br[12+b] = r2*TW36R[b][2] - i2*TW36I[b][2];  bi[12+b] = r2*TW36I[b][2] + i2*TW36R[b][2];
            br[18+b] = r3*TW36R[b][3] - i3*TW36I[b][3];  bi[18+b] = r3*TW36I[b][3] + i3*TW36R[b][3];
            br[24+b] = r4*TW36R[b][4] - i4*TW36I[b][4];  bi[24+b] = r4*TW36I[b][4] + i4*TW36R[b][4];
            br[30+b] = r5*TW36R[b][5] - i5*TW36I[b][5];  bi[30+b] = r5*TW36I[b][5] + i5*TW36R[b][5];
        }
    }
    for (int k1 = 0; k1 < 6; k1++) {
        vd r0 = br[6*k1],   i0 = bi[6*k1];
        vd r1 = br[6*k1+1], i1 = bi[6*k1+1];
        vd r2 = br[6*k1+2], i2 = bi[6*k1+2];
        vd r3 = br[6*k1+3], i3 = bi[6*k1+3];
        vd r4 = br[6*k1+4], i4 = bi[6*k1+4];
        vd r5 = br[6*k1+5], i5 = bi[6*k1+5];
        bf6(&r0,&i0,&r1,&i1,&r2,&i2,&r3,&i3,&r4,&i4,&r5,&i5);
        ptrdiff_t p0 = (k1)*s,    p1 = (k1+6)*s,  p2 = (k1+12)*s;
        ptrdiff_t p3 = (k1+18)*s, p4 = (k1+24)*s, p5 = (k1+30)*s;
        KSTORE(xr[p0], xi[p0], r0, i0, cr[p0], ci[p0], domap);
        KSTORE(xr[p1], xi[p1], r1, i1, cr[p1], ci[p1], domap);
        KSTORE(xr[p2], xi[p2], r2, i2, cr[p2], ci[p2], domap);
        KSTORE(xr[p3], xi[p3], r3, i3, cr[p3], ci[p3], domap);
        KSTORE(xr[p4], xi[p4], r4, i4, cr[p4], ci[p4], domap);
        KSTORE(xr[p5], xi[p5], r5, i5, cr[p5], ci[p5], domap);
    }
}

static __attribute__((always_inline)) inline
void k45(vd *restrict xr, vd *restrict xi, ptrdiff_t s,
         const vd *restrict cr, const vd *restrict ci, const int domap) {
    vd br[45], bi[45];
    for (int b = 0; b < 9; b++) {
        vd r0 = xr[b*s],      i0 = xi[b*s];
        vd r1 = xr[(9+b)*s],  i1 = xi[(9+b)*s];
        vd r2 = xr[(18+b)*s], i2 = xi[(18+b)*s];
        vd r3 = xr[(27+b)*s], i3 = xi[(27+b)*s];
        vd r4 = xr[(36+b)*s], i4 = xi[(36+b)*s];
        bf5(&r0,&i0,&r1,&i1,&r2,&i2,&r3,&i3,&r4,&i4);
        br[b] = r0; bi[b] = i0;
        if (b == 0) {
            br[9]=r1; bi[9]=i1; br[18]=r2; bi[18]=i2;
            br[27]=r3; bi[27]=i3; br[36]=r4; bi[36]=i4;
        } else {
            br[9+b]  = r1*TW45R[b][1] - i1*TW45I[b][1];  bi[9+b]  = r1*TW45I[b][1] + i1*TW45R[b][1];
            br[18+b] = r2*TW45R[b][2] - i2*TW45I[b][2];  bi[18+b] = r2*TW45I[b][2] + i2*TW45R[b][2];
            br[27+b] = r3*TW45R[b][3] - i3*TW45I[b][3];  bi[27+b] = r3*TW45I[b][3] + i3*TW45R[b][3];
            br[36+b] = r4*TW45R[b][4] - i4*TW45I[b][4];  bi[36+b] = r4*TW45I[b][4] + i4*TW45R[b][4];
        }
    }
    for (int k1 = 0; k1 < 5; k1++) {
        vd *R = br + 9*k1, *I = bi + 9*k1;
        for (int b2 = 0; b2 < 3; b2++)
            bf3(&R[b2], &I[b2], &R[b2+3], &I[b2+3], &R[b2+6], &I[b2+6]);
        for (int kk = 1; kk < 3; kk++)
            for (int b2 = 1; b2 < 3; b2++) {
                vd gr = R[3*kk + b2], gi = I[3*kk + b2];
                R[3*kk + b2] = gr*W9R[b2][kk] - gi*W9I[b2][kk];
                I[3*kk + b2] = gr*W9I[b2][kk] + gi*W9R[b2][kk];
            }
        for (int kk = 0; kk < 3; kk++)
            bf3(&R[3*kk], &I[3*kk], &R[3*kk+1], &I[3*kk+1], &R[3*kk+2], &I[3*kk+2]);
        for (int kk = 0; kk < 3; kk++)
            for (int k2b = 0; k2b < 3; k2b++) {
                ptrdiff_t pos = (ptrdiff_t)(k1 + 5*(kk + 3*k2b)) * s;
                KSTORE(xr[pos], xi[pos], R[3*kk + k2b], I[3*kk + k2b], cr[pos], ci[pos], domap);
            }
    }
}

static __attribute__((always_inline)) inline
void k64(vd *restrict xr, vd *restrict xi, ptrdiff_t s,
         const vd *restrict cr, const vd *restrict ci, const int domap) {
    vd br[64], bi[64];
    for (int b = 0; b < 8; b++) {
        vd r0 = xr[b*s],      i0 = xi[b*s];
        vd r1 = xr[(8+b)*s],  i1 = xi[(8+b)*s];
        vd r2 = xr[(16+b)*s], i2 = xi[(16+b)*s];
        vd r3 = xr[(24+b)*s], i3 = xi[(24+b)*s];
        vd r4 = xr[(32+b)*s], i4 = xi[(32+b)*s];
        vd r5 = xr[(40+b)*s], i5 = xi[(40+b)*s];
        vd r6 = xr[(48+b)*s], i6 = xi[(48+b)*s];
        vd r7 = xr[(56+b)*s], i7 = xi[(56+b)*s];
        bf8(&r0,&i0,&r1,&i1,&r2,&i2,&r3,&i3,&r4,&i4,&r5,&i5,&r6,&i6,&r7,&i7);
        br[b] = r0; bi[b] = i0;
        if (b == 0) {
            br[8]=r1; bi[8]=i1;   br[16]=r2; bi[16]=i2;
            br[24]=r3; bi[24]=i3; br[32]=r4; bi[32]=i4;
            br[40]=r5; bi[40]=i5; br[48]=r6; bi[48]=i6;
            br[56]=r7; bi[56]=i7;
        } else {
            br[8+b]  = r1*TW64R[b][1] - i1*TW64I[b][1];  bi[8+b]  = r1*TW64I[b][1] + i1*TW64R[b][1];
            br[16+b] = r2*TW64R[b][2] - i2*TW64I[b][2];  bi[16+b] = r2*TW64I[b][2] + i2*TW64R[b][2];
            br[24+b] = r3*TW64R[b][3] - i3*TW64I[b][3];  bi[24+b] = r3*TW64I[b][3] + i3*TW64R[b][3];
            br[32+b] = r4*TW64R[b][4] - i4*TW64I[b][4];  bi[32+b] = r4*TW64I[b][4] + i4*TW64R[b][4];
            br[40+b] = r5*TW64R[b][5] - i5*TW64I[b][5];  bi[40+b] = r5*TW64I[b][5] + i5*TW64R[b][5];
            br[48+b] = r6*TW64R[b][6] - i6*TW64I[b][6];  bi[48+b] = r6*TW64I[b][6] + i6*TW64R[b][6];
            br[56+b] = r7*TW64R[b][7] - i7*TW64I[b][7];  bi[56+b] = r7*TW64I[b][7] + i7*TW64R[b][7];
        }
    }
    for (int k1 = 0; k1 < 8; k1++) {
        vd r0 = br[8*k1],   i0 = bi[8*k1];
        vd r1 = br[8*k1+1], i1 = bi[8*k1+1];
        vd r2 = br[8*k1+2], i2 = bi[8*k1+2];
        vd r3 = br[8*k1+3], i3 = bi[8*k1+3];
        vd r4 = br[8*k1+4], i4 = bi[8*k1+4];
        vd r5 = br[8*k1+5], i5 = bi[8*k1+5];
        vd r6 = br[8*k1+6], i6 = bi[8*k1+6];
        vd r7 = br[8*k1+7], i7 = bi[8*k1+7];
        bf8(&r0,&i0,&r1,&i1,&r2,&i2,&r3,&i3,&r4,&i4,&r5,&i5,&r6,&i6,&r7,&i7);
        ptrdiff_t p0 = (k1)*s,    p1 = (k1+8)*s,  p2 = (k1+16)*s, p3 = (k1+24)*s;
        ptrdiff_t p4 = (k1+32)*s, p5 = (k1+40)*s, p6 = (k1+48)*s, p7 = (k1+56)*s;
        KSTORE(xr[p0], xi[p0], r0, i0, cr[p0], ci[p0], domap);
        KSTORE(xr[p1], xi[p1], r1, i1, cr[p1], ci[p1], domap);
        KSTORE(xr[p2], xi[p2], r2, i2, cr[p2], ci[p2], domap);
        KSTORE(xr[p3], xi[p3], r3, i3, cr[p3], ci[p3], domap);
        KSTORE(xr[p4], xi[p4], r4, i4, cr[p4], ci[p4], domap);
        KSTORE(xr[p5], xi[p5], r5, i5, cr[p5], ci[p5], domap);
        KSTORE(xr[p6], xi[p6], r6, i6, cr[p6], ci[p6], domap);
        KSTORE(xr[p7], xi[p7], r7, i7, cr[p7], ci[p7], domap);
    }
}

#define GEN_WRAPPERS(KK, LL, PF)                                                \
static __attribute__((always_inline)) inline                                    \
void KK##_p(vd *restrict xr, vd *restrict xi, ptrdiff_t s) {                    \
    KK(xr, xi, s, (const vd*)xr, (const vd*)xi, 0);                             \
}                                                                               \
static __attribute__((always_inline)) inline                                    \
void KK##_m(vd *restrict xr, vd *restrict xi, ptrdiff_t s,                      \
                   const vd *restrict cr, const vd *restrict ci) {              \
    if (PF) for (int i = 0; i < LL; i++) {                                      \
        _mm_prefetch((const char*)(cr + (ptrdiff_t)i*s), _MM_HINT_T0);          \
        _mm_prefetch((const char*)(cr + (ptrdiff_t)i*s) + 64, _MM_HINT_T0);     \
    }                                                                           \
    KK(xr, xi, s, cr, ci, 1);                                                   \
}
GEN_WRAPPERS(k6, 6, 0)   GEN_WRAPPERS(k8, 8, 0)   GEN_WRAPPERS(k13, 13, 0) GEN_WRAPPERS(k17, 17, 0)
GEN_WRAPPERS(k23, 23, 0) GEN_WRAPPERS(k36, 36, 0) GEN_WRAPPERS(k45, 45, 1) GEN_WRAPPERS(k64, 64, 0)

// ------------------------------------------------------------------
// 8x8 transpose of vd rows
// ------------------------------------------------------------------
static __attribute__((always_inline)) inline void tr8(vd *a) {
    __m512d r0=(__m512d)a[0], r1=(__m512d)a[1], r2=(__m512d)a[2], r3=(__m512d)a[3];
    __m512d r4=(__m512d)a[4], r5=(__m512d)a[5], r6=(__m512d)a[6], r7=(__m512d)a[7];
    __m512d t0 = _mm512_unpacklo_pd(r0, r1), t1 = _mm512_unpackhi_pd(r0, r1);
    __m512d t2 = _mm512_unpacklo_pd(r2, r3), t3 = _mm512_unpackhi_pd(r2, r3);
    __m512d t4 = _mm512_unpacklo_pd(r4, r5), t5 = _mm512_unpackhi_pd(r4, r5);
    __m512d t6 = _mm512_unpacklo_pd(r6, r7), t7 = _mm512_unpackhi_pd(r6, r7);
    __m512d v0 = _mm512_shuffle_f64x2(t0, t2, 0x88), v2 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
    __m512d v1 = _mm512_shuffle_f64x2(t1, t3, 0x88), v3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
    __m512d v4 = _mm512_shuffle_f64x2(t4, t6, 0x88), v6 = _mm512_shuffle_f64x2(t4, t6, 0xDD);
    __m512d v5 = _mm512_shuffle_f64x2(t5, t7, 0x88), v7 = _mm512_shuffle_f64x2(t5, t7, 0xDD);
    a[0] = (vd)_mm512_shuffle_f64x2(v0, v4, 0x88);
    a[1] = (vd)_mm512_shuffle_f64x2(v1, v5, 0x88);
    a[2] = (vd)_mm512_shuffle_f64x2(v2, v6, 0x88);
    a[3] = (vd)_mm512_shuffle_f64x2(v3, v7, 0x88);
    a[4] = (vd)_mm512_shuffle_f64x2(v0, v4, 0xDD);
    a[5] = (vd)_mm512_shuffle_f64x2(v1, v5, 0xDD);
    a[6] = (vd)_mm512_shuffle_f64x2(v2, v6, 0xDD);
    a[7] = (vd)_mm512_shuffle_f64x2(v3, v7, 0xDD);
}

// ------------------------------------------------------------------
// z-pass tile for in-volume 23/36/45: 8 y-rows (cnt real), lanes<->rows
// base: memory vd pointer to row0 slot0 (re); rows are RS slots apart.
// ------------------------------------------------------------------
#define GEN_ZPASS_TILE(NAME, LL, ZPV, KERNP)                                    \
static void NAME(vd *restrict base, ptrdiff_t RS, int cnt) {                    \
    vd bufR[(ZPV)*8], bufI[(ZPV)*8];                                           \
    for (int blk = 0; blk < (ZPV); blk++) {                                    \
        vd tR[8], tI[8];                                                       \
        for (int r = 0; r < 8; r++) {                                          \
            int rr = r < cnt ? r : 0;                                          \
            tR[r] = base[2*(rr*RS + blk)];                                     \
            tI[r] = base[2*(rr*RS + blk) + 1];                                 \
        }                                                                      \
        tr8(tR); tr8(tI);                                                      \
        for (int c = 0; c < 8; c++) { bufR[8*blk+c] = tR[c]; bufI[8*blk+c] = tI[c]; } \
    }                                                                          \
    KERNP(bufR, bufI, 1);                                                      \
    for (int blk = 0; blk < (ZPV); blk++) {                                    \
        vd tR[8], tI[8];                                                       \
        for (int c = 0; c < 8; c++) { tR[c] = bufR[8*blk+c]; tI[c] = bufI[8*blk+c]; } \
        tr8(tR); tr8(tI);                                                      \
        for (int r = 0; r < cnt; r++) {                                        \
            base[2*(r*RS + blk)]     = tR[r];                                  \
            base[2*(r*RS + blk) + 1] = tI[r];                                  \
        }                                                                      \
    }                                                                          \
}
GEN_ZPASS_TILE(zpass23, 23, 3, k23_p)
GEN_ZPASS_TILE(zpass36, 36, 5, k36_p)
GEN_ZPASS_TILE(zpass45, 45, 6, k45_p)

// ------------------------------------------------------------------
// Conversions
// ------------------------------------------------------------------
// AoSoA: X[2*i][lane l] = re of volume l element i.
// Vectorized: 8 rows of 4 complexes (one per lane) -> 8x8 transpose ->
// 8 consecutive vds (re/im pairs of 4 slots).
static void conv_in_aosoaC(const double *restrict src, long nl, long N, long CNT, vd *restrict d) {
    long i = 0;
    for (; i + 4 <= CNT; i += 4) {
        vd t[8];
        for (long l = 0; l < nl; l++)
            t[l] = (vd)_mm512_loadu_pd(src + (l*N + i)*2);
        for (long l = nl; l < 8; l++)
            t[l] = (vd)_mm512_setzero_pd();
        tr8(t);
        vd *o = d + 2*i;
        o[0]=t[0]; o[1]=t[1]; o[2]=t[2]; o[3]=t[3];
        o[4]=t[4]; o[5]=t[5]; o[6]=t[6]; o[7]=t[7];
    }
    long tail_i = CNT & 3;
    for (long t2 = 0; t2 < tail_i; t2++, i++) {
        double *R = (double*)(d + 2*i), *I = (double*)(d + 2*i + 1);
        const double *sp = src + (size_t)i*2;
        for (long l = 0; l < nl; l++) {
            R[l] = sp[(size_t)l*(size_t)N*2];
            I[l] = sp[(size_t)l*(size_t)N*2 + 1];
        }
        for (long l = nl; l < 8; l++) { R[l] = 0.0; I[l] = 0.0; }
    }
}
static void conv_out_aosoaC(const vd *restrict d, double *restrict dst, long nl, long N, long CNT) {
    long i = 0;
    for (; i + 4 <= CNT; i += 4) {
        vd t[8];
        const vd *o = d + 2*i;
        t[0]=o[0]; t[1]=o[1]; t[2]=o[2]; t[3]=o[3];
        t[4]=o[4]; t[5]=o[5]; t[6]=o[6]; t[7]=o[7];
        tr8(t);
        for (long l = 0; l < nl; l++)
            _mm512_storeu_pd(dst + (l*N + i)*2, (__m512d)t[l]);
    }
    long tail_o = CNT & 3;
    for (long t2 = 0; t2 < tail_o; t2++, i++) {
        const double *R = (const double*)(d + 2*i), *I = (const double*)(d + 2*i + 1);
        double *dp = dst + (size_t)i*2;
        for (long l = 0; l < nl; l++) {
            dp[(size_t)l*(size_t)N*2]     = R[l];
            dp[(size_t)l*(size_t)N*2 + 1] = I[l];
        }
    }
}

static void conv_in_aosoa(const double *restrict src, long nl, long N, vd *restrict d) {
    conv_in_aosoaC(src, nl, N, N, d);
}
static void conv_out_aosoa(const vd *restrict d, double *restrict dst, long nl, long N) {
    conv_out_aosoaC(d, dst, nl, N, N);
}

// in-volume z-lanes (23/36/45): slot(x,y,zc) = (x*L+y)*ZPV + zc
static void conv_in_vol(const double *restrict src, int L, int ZPV, vd *restrict X) {
    const __m512i IRE = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    const __m512i IIM = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    int ZP = ZPV*8;
    for (int xy = 0; xy < L*L; xy++) {
        const double *s = src + (long)xy*L*2;
        vd *r = X + (long)xy*ZPV*2;
        int z = 0;
        for (; z + 8 <= L; z += 8) {
            __m512d v0 = _mm512_loadu_pd(s + 2*z);
            __m512d v1 = _mm512_loadu_pd(s + 2*z + 8);
            r[(z>>3)*2]     = (vd)_mm512_permutex2var_pd(v0, IRE, v1);
            r[(z>>3)*2 + 1] = (vd)_mm512_permutex2var_pd(v0, IIM, v1);
        }
        if (z < L) {
            double *R = (double*)(r + (z>>3)*2), *I = R + 8;
            int zl = 0;
            for (; z < L; z++, zl++) { R[zl] = s[2*z]; I[zl] = s[2*z+1]; }
            for (; zl < 8; zl++) { R[zl] = 0.0; I[zl] = 0.0; }
        }
        for (int zc = (L+7)>>3; zc < ZPV; zc++) {
            r[zc*2] = (vd)_mm512_setzero_pd(); r[zc*2+1] = (vd)_mm512_setzero_pd();
        }
    }
    (void)ZP;
}
static void conv_out_vol(const vd *restrict X, int L, int ZPV, double *restrict dst) {
    const __m512i ILO = _mm512_set_epi64(11,3,10,2,9,1,8,0);
    const __m512i IHI = _mm512_set_epi64(15,7,14,6,13,5,12,4);
    for (int xy = 0; xy < L*L; xy++) {
        const vd *r = X + (long)xy*ZPV*2;
        double *d = dst + (long)xy*L*2;
        int z = 0;
        for (; z + 8 <= L; z += 8) {
            __m512d re = (__m512d)r[(z>>3)*2], im = (__m512d)r[(z>>3)*2 + 1];
            _mm512_storeu_pd(d + 2*z,     _mm512_permutex2var_pd(re, ILO, im));
            _mm512_storeu_pd(d + 2*z + 8, _mm512_permutex2var_pd(re, IHI, im));
        }
        if (z < L) {
            const double *R = (const double*)(r + (z>>3)*2), *I = R + 8;
            for (int zl = 0; z < L; z++, zl++) { d[2*z] = R[zl]; d[2*z+1] = I[zl]; }
        }
    }
}

// 64: slot(a,y,z) = (a*64+y)*64+z, lane = b, x = 8a+b
#define RY64 66
#define SA64 4232    /* 64*66 + 8 : breaks 4K aliasing between a-slabs */
static void conv_in_64(const double *restrict src, vd *restrict X) {
    for (int a = 0; a < 8; a++)
        for (int y = 0; y < 64; y++)
            conv_in_aosoaC(src + (long)a*8*4096*2 + (long)y*128, 8, 4096, 64,
                           X + 2*((long)a*SA64 + (long)y*RY64));
}
static void conv_out_64(const vd *restrict X, double *restrict dst) {
    for (int a = 0; a < 8; a++)
        for (int y = 0; y < 64; y++)
            conv_out_aosoaC(X + 2*((long)a*SA64 + (long)y*RY64),
                            dst + (long)a*8*4096*2 + (long)y*128, 8, 4096, 64);
}

// ------------------------------------------------------------------
// AoSoA drivers: L in {6,8,13,17}
// ------------------------------------------------------------------
// AoSoA pipeline: sweepY, then sweepX (fused map) per y-row with the next
// iteration's z-pass pipelined behind each completed row (skipped on snapshot
// iterations, where a standalone z sweep restarts the pipeline).
// XS = padded x-slab stride in slots (breaks 4K aliasing for L=8).
#define GEN_RUN_AOSOA(LL, XS, KP, KM)                                           \
static void sweepZ_A##LL(vd *restrict X) {                                      \
    for (int x = 0; x < LL; x++)                                                \
        for (int y = 0; y < LL; y++) {                                          \
            vd *p = X + 2*((ptrdiff_t)x*(XS) + y*LL);                           \
            KP(p, p + 1, 2);                                                    \
        }                                                                       \
}                                                                               \
static void sweepY_A##LL(vd *restrict X) {                                      \
    for (int x = 0; x < LL; x++)                                                \
        for (int z = 0; z < LL; z++) {                                          \
            vd *p = X + 2*((ptrdiff_t)x*(XS) + z);                              \
            KP(p, p + 1, 2*LL);                                                 \
        }                                                                       \
}                                                                               \
static void sweepX_A##LL(vd *restrict X, const vd *restrict C, int fuse_z) {    \
    for (int y = 0; y < LL; y++) {                                              \
        for (int z = 0; z < LL; z++) {                                          \
            ptrdiff_t q = (ptrdiff_t)y*LL + z;                                  \
            KM(X + 2*q, X + 2*q + 1, 2*(XS), C + 2*q, C + 2*q + 1);             \
        }                                                                       \
        if (fuse_z)                                                             \
            for (int x = 0; x < LL; x++) {                                      \
                vd *p = X + 2*((ptrdiff_t)x*(XS) + y*LL);                       \
                KP(p, p + 1, 2);                                                \
            }                                                                   \
    }                                                                           \
}                                                                               \
void run##LL(long long Bll, long long mll, const double *x0, const double *c0, \
             double *out1, double *outm) {                                      \
    const long N = (long)LL*LL*LL; const long B = (long)Bll;                    \
    long m = (long)mll; if (m < 1) m = 1;                                       \
    for (long b0 = 0; b0 < B; b0 += VL) {                                       \
        long nl = B - b0 < VL ? B - b0 : VL;                                    \
        for (int x = 0; x < LL; x++) {                                          \
            conv_in_aosoaC(x0 + (long long)b0*N*2 + (long)x*LL*LL*2, nl, N,     \
                           (long)LL*LL, XP_ + 2*(ptrdiff_t)x*(XS));             \
            conv_in_aosoaC(c0 + (long long)b0*N*2 + (long)x*LL*LL*2, nl, N,     \
                           (long)LL*LL, CP_ + 2*(ptrdiff_t)x*(XS));             \
        }                                                                       \
        sweepZ_A##LL(XP_);                                                      \
        for (long it = 1; it <= m; it++) {                                      \
            int snap = (it == 1) || (it == m);                                  \
            sweepY_A##LL(XP_);                                                  \
            sweepX_A##LL(XP_, CP_, !snap);                                      \
            if (snap) {                                                         \
                for (int x = 0; x < LL; x++) {                                  \
                    if (it == 1)                                                \
                        conv_out_aosoaC(XP_ + 2*(ptrdiff_t)x*(XS),              \
                                        out1 + (long long)b0*N*2 + (long)x*LL*LL*2, \
                                        nl, N, (long)LL*LL);                    \
                    if (it == m)                                                \
                        conv_out_aosoaC(XP_ + 2*(ptrdiff_t)x*(XS),              \
                                        outm + (long long)b0*N*2 + (long)x*LL*LL*2, \
                                        nl, N, (long)LL*LL);                    \
                }                                                               \
                if (it < m) sweepZ_A##LL(XP_);                                  \
            }                                                                   \
        }                                                                       \
    }                                                                           \
}
GEN_RUN_AOSOA(6,  36,  k6_p,  k6_m)
GEN_RUN_AOSOA(8,  68,  k8_p,  k8_m)
GEN_RUN_AOSOA(13, 169, k13_p, k13_m)
GEN_RUN_AOSOA(17, 289, k17_p, k17_m)


// ------------------------------------------------------------------
// In-volume drivers 23/36/45.
// Per iteration: sweepY, then sweepX (x-DFT + map) in y-row groups of 8,
// with the NEXT iteration's z-pass fused right behind each completed group
// (skipped on snapshot iterations, where a standalone z sweep follows).
// ------------------------------------------------------------------
// x-line DFT for a group of y rows [y0, y0+cnt) over all zc, either direct
// (23) or tiled through a contiguous buffer (36/45).

#define GEN_VOL(LL, ZPV, ZPASS, KP, KM, XGROUP)                                 \
static void sweepZ_##LL(vd *restrict X) {                                       \
    const ptrdiff_t SR = (ZPV);                                                 \
    for (int x = 0; x < LL; x++)                                                \
        for (int y0 = 0; y0 < LL; y0 += 8) {                                    \
            int cnt = LL - y0 < 8 ? LL - y0 : 8;                                \
            ZPASS(X + 2*((ptrdiff_t)(x*LL + y0) * SR), SR, cnt);                \
        }                                                                       \
}                                                                               \
static void sweepY_##LL(vd *restrict X) {                                       \
    for (int x = 0; x < LL; x++)                                                \
        for (int zc = 0; zc < (ZPV); zc++) {                                    \
            vd *p = X + 2*((ptrdiff_t)x*LL*(ZPV) + zc);                         \
            KP(p, p + 1, 2*(ZPV));                                              \
        }                                                                       \
}                                                                               \
static void sweepX_##LL(vd *restrict X, const vd *restrict C, int fuse_z) {     \
    const ptrdiff_t SR = (ZPV);                                                 \
    for (int y0 = 0; y0 < LL; y0 += 8) {                                        \
        int cnt = LL - y0 < 8 ? LL - y0 : 8;                                    \
        XGROUP(X, C, y0, cnt);                                                  \
        if (fuse_z)                                                             \
            for (int x = 0; x < LL; x++)                                        \
                ZPASS(X + 2*((ptrdiff_t)(x*LL + y0) * SR), SR, cnt);            \
    }                                                                           \
}                                                                               \
void run##LL(long long Bll, long long mll, const double *x0, const double *c0, \
             double *out1, double *outm) {                                      \
    const long NV = (long)LL*LL*LL; const long B = (long)Bll;                   \
    long m = (long)mll; if (m < 1) m = 1;                                       \
    for (long v = 0; v < B; v++) {                                              \
        conv_in_vol(x0 + (long long)v*NV*2, LL, ZPV, XP_);                      \
        conv_in_vol(c0 + (long long)v*NV*2, LL, ZPV, CP_);                      \
        sweepZ_##LL(XP_);                                                       \
        for (long it = 1; it <= m; it++) {                                      \
            int snap = (it == 1) || (it == m);                                  \
            sweepY_##LL(XP_);                                                   \
            sweepX_##LL(XP_, CP_, !snap);                                       \
            if (snap) {                                                         \
                if (it == 1) conv_out_vol(XP_, LL, ZPV, out1 + (long long)v*NV*2); \
                if (it == m) conv_out_vol(XP_, LL, ZPV, outm + (long long)v*NV*2); \
                if (it < m) sweepZ_##LL(XP_);                                   \
            }                                                                   \
        }                                                                       \
    }                                                                           \
}

// direct x-group for 23 (block is L2 resident)
static void xgroup_23(vd *restrict X, const vd *restrict C, int y0, int cnt) {
    const ptrdiff_t SX = 23*3;
    for (ptrdiff_t q = (ptrdiff_t)y0*3; q < (ptrdiff_t)(y0+cnt)*3; q++)
        k23_m(X + 2*q, X + 2*q + 1, 2*SX, C + 2*q, C + 2*q + 1);
}

// direct x-groups for 36/45 (state is L2/L3 resident; strides break 4K aliasing)
static void xgroup_36(vd *restrict X, const vd *restrict C, int y0, int cnt) {
    const ptrdiff_t SX = 180;
    for (ptrdiff_t q = (ptrdiff_t)y0*5; q < (ptrdiff_t)(y0+cnt)*5; q++)
        k36_m(X + 2*q, X + 2*q + 1, 2*SX, C + 2*q, C + 2*q + 1);
}
static void xgroup_45(vd *restrict X, const vd *restrict C, int y0, int cnt) {
    const ptrdiff_t SX = 270;
    for (ptrdiff_t q = (ptrdiff_t)y0*6; q < (ptrdiff_t)(y0+cnt)*6; q++)
        k45_m(X + 2*q, X + 2*q + 1, 2*SX, C + 2*q, C + 2*q + 1);
}

GEN_VOL(23, 3, zpass23, k23_p, k23_m, xgroup_23)
GEN_VOL(36, 5, zpass36, k36_p, k36_m, xgroup_36)
GEN_VOL(45, 6, zpass45, k45_p, k45_m, xgroup_45)

// ------------------------------------------------------------------
// L = 64 driver: slot(a,y,z) = (a*64+y)*64+z, lanes = b (x = 8a+b)
// ------------------------------------------------------------------
// x-line four-step with fused +c/map; operates on slots q, q+4096, ...
static __attribute__((always_inline)) inline
void x64lane(vd *restrict X, const vd *restrict C, ptrdiff_t q) {
    const ptrdiff_t S = 2*SA64;
    vd R[8], I[8];
    vd *p = X + 2*q;
    const vd *c = C + 2*q;
    for (int a = 0; a < 8; a++) { R[a] = p[a*S]; I[a] = p[a*S + 1]; }
    bf8(&R[0],&I[0],&R[1],&I[1],&R[2],&I[2],&R[3],&I[3],
        &R[4],&I[4],&R[5],&I[5],&R[6],&I[6],&R[7],&I[7]);
    for (int k1 = 1; k1 < 8; k1++) {
        vd gr = R[k1], gi = I[k1];
        R[k1] = gr * ZTWR[k1] - gi * ZTWI[k1];
        I[k1] = gr * ZTWI[k1] + gi * ZTWR[k1];
    }
    tr8(R); tr8(I);
    bf8(&R[0],&I[0],&R[1],&I[1],&R[2],&I[2],&R[3],&I[3],
        &R[4],&I[4],&R[5],&I[5],&R[6],&I[6],&R[7],&I[7]);
    for (int k2 = 0; k2 < 8; k2++) {
        vd zr = R[k2] + c[k2*S], zi = I[k2] + c[k2*S + 1];
        mapv(&zr, &zi);
        p[k2*S] = zr; p[k2*S + 1] = zi;
    }
}

static void sweepZ_64(vd *restrict X) {
    for (int a = 0; a < 8; a++)
        for (int y = 0; y < 64; y++) {
            vd *p = X + 2*((ptrdiff_t)a*SA64 + (ptrdiff_t)y*RY64);
            k64_p(p, p + 1, 2);
        }
}
static void sweepZrow_64(vd *restrict X, int y) {
    for (int a = 0; a < 8; a++) {
        vd *p = X + 2*((ptrdiff_t)a*SA64 + (ptrdiff_t)y*RY64);
        k64_p(p, p + 1, 2);
    }
}
static void sweepY_64(vd *restrict X) {
    for (int a = 0; a < 8; a++)
        for (int z = 0; z < 64; z++) {
            vd *p = X + 2*((ptrdiff_t)a*SA64 + z);
            k64_p(p, p + 1, 2*RY64);
        }
}
static void sweepX_64(vd *restrict X, const vd *restrict C, int fuse_z) {
    for (int y = 0; y < 64; y++) {
        for (int z = 0; z < 64; z++)
            x64lane(X, C, (ptrdiff_t)y*RY64 + z);
        if (fuse_z) sweepZrow_64(X, y);
    }
}

void run64(long long Bll, long long mll, const double *x0, const double *c0,
           double *out1, double *outm) {
    const long NV = 262144; const long B = (long)Bll;
    long m = (long)mll; if (m < 1) m = 1;
    for (long v = 0; v < B; v++) {
        conv_in_64(x0 + (long long)v*NV*2, XP_);
        conv_in_64(c0 + (long long)v*NV*2, CP_);
        sweepZ_64(XP_);
        for (long it = 1; it <= m; it++) {
            int snap = (it == 1) || (it == m);
            sweepY_64(XP_);
            sweepX_64(XP_, CP_, !snap);
            if (snap) {
                if (it == 1) conv_out_64(XP_, out1 + (long long)v*NV*2);
                if (it == m) conv_out_64(XP_, outm + (long long)v*NV*2);
                if (it < m) sweepZ_64(XP_);
            }
        }
    }
}

int probe(void) { return 512; }
