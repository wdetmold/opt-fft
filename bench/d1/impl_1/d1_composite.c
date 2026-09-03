/* d1_composite: Good-Thomas PFA for L=60 = 4*3*5 (coprime, twiddle-FREE).
 *
 * The 60-point DFT is re-indexed by the CRT into a 4x3x5 3D DFT: input
 * x[(15n1+20n2+12n3) mod 60] -> W[n1][n2][n3], output Y at k = (45k1+40k2+36k3) mod 60.
 * No twiddle factors between stages at all -- the whole transform is 15 DFT-4s
 * (add-only), 20 DFT-3s and 12 DFT-5s (Winograd-style real-constant kernels).
 *
 * Code paths:
 *   1. scalar fully-unrolled kernel (B=1 execute and batch remainders);
 *   2. AVX-512 across-batch path for execute at B>=8: 8 transforms per zmm in
 *      SPLIT-COMPLEX SoA (lane j = transform j -- zero shuffles in the
 *      arithmetic, x(-i) is an array swap, x(real const) is one vmulpd);
 *      gathers in, scatters out;
 *   3. an owned fft1d_chain:
 *      - B>=8: the chain STATE LIVES IN SoA across all m steps -- transposed in
 *        once, then every step is pure full-width vector loads/stores. The
 *        inter-step permutation SROW is only row renaming, and it maps the 12
 *        DFT-5 output blocks onto each other as an INVOLUTION (pairs 1<->2,
 *        3<->9, 4<->11, 5<->10, 7<->8; 0 and 6 fixed), so the step runs fully
 *        in place with the map fused. Transposed out once at the end.
 *      - B<8: per-transform SPLIT-complex state (sr[60]/si[60] natural order),
 *        scalar PFA stages + an 8-wide shuffle-free vector map.
 *
 * L=12/24/36 keep the dense O(L^2) floor (not in the measured case list).
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../fft1d_api.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#define HAVE_AVX512 1
#else
#define HAVE_AVX512 0
#endif

/* ---- CRT index tables (verified against numpy: PFA err 7e-15) ----
 * idx = n1*15 + n2*5 + n3.
 * PIN[idx]  = (15n1+20n2+12n3) mod 60 : where stage A reads x.
 * KOUT[idx] = (45n1+40n2+36n3) mod 60 : where stage C writes y (n3 = k3 there).
 * SROW[idx] = PININV[KOUT[idx]]       : chain only -- the state row the mapped
 *   stage-C output of slot idx lands in so the NEXT step's stage A is contiguous. */
static const int PIN[60] = {0,12,24,36,48,20,32,44,56,8,40,52,4,16,28,15,27,39,51,3,35,47,59,11,23,55,7,19,31,43,30,42,54,6,18,50,2,14,26,38,10,22,34,46,58,45,57,9,21,33,5,17,29,41,53,25,37,49,1,13};
static const int KOUT[60] = {0,36,12,48,24,40,16,52,28,4,20,56,32,8,44,45,21,57,33,9,25,1,37,13,49,5,41,17,53,29,30,6,42,18,54,10,46,22,58,34,50,26,2,38,14,15,51,27,3,39,55,31,7,43,19,35,11,47,23,59};
static const int SROW[60] = {0,3,1,4,2,10,13,11,14,12,5,8,6,9,7,45,48,46,49,47,55,58,56,59,57,50,53,51,54,52,30,33,31,34,32,40,43,41,44,42,35,38,36,39,37,15,18,16,19,17,25,28,26,29,27,20,23,21,24,22};
/* stage-C block pairs under SROW (verified involution) */
static const int CPAIR[7][2] = {{0,0},{6,6},{1,2},{3,9},{4,11},{5,10},{7,8}};

/* DFT-5 / DFT-3 constants (w = e^{-2pi i/N}, forward) */
#define C51 (0.30901699437494745126)   /* cos(2pi/5) */
#define C52 (-0.80901699437494734024)  /* cos(4pi/5) */
#define S51 (0.95105651629515353118)   /* sin(2pi/5) */
#define S52 (0.58778525229247324813)   /* sin(4pi/5) */
#define S3  (0.86602540378443870761)   /* sin(2pi/3) */

struct fft1d_plan {
    int L, batch;
    double _Complex *w;   /* dense fallback matrix for L=12/24/36 */
#if HAVE_AVX512
    double *soa_state;    /* chain SoA state: ngroup * 60 rows * 16 doubles (re8,im8) */
    double *soa_c;        /* chain SoA c, pre-permuted to stage-C slot order */
#endif
};

const char *fft1d_name(void){ return "d1_composite"; }
const char *fft1d_description(void){
    return "Good-Thomas PFA 60=4x3x5 twiddle-free; n1-paired ymm kernels; SoA-resident in-place chain with fused rsqrt/rcp map";
}
int fft1d_supports(int L){ return L == 24 || L == 60 || L == 12 || L == 36; }

/* ================= scalar PFA-60 stage bodies =================
 * Shared by the interleaved-output kernel (execute) and the split-state chain
 * step. Stage A gathers via LOADR/LOADI expressions; stages B and C work on
 * split work arrays ar/ai. */
#define SCALAR_STAGE_A(ar, ai, LOADR, LOADI)                                             \
    for (int col = 0; col < 15; ++col) {                                                 \
        const int p0 = PIN[col], p1 = PIN[15+col], p2 = PIN[30+col], p3 = PIN[45+col];   \
        const double x0r = LOADR(p0), x0i = LOADI(p0), x1r = LOADR(p1), x1i = LOADI(p1); \
        const double x2r = LOADR(p2), x2i = LOADI(p2), x3r = LOADR(p3), x3i = LOADI(p3); \
        const double t0r = x0r + x2r, t0i = x0i + x2i;                                   \
        const double t1r = x0r - x2r, t1i = x0i - x2i;                                   \
        const double t2r = x1r + x3r, t2i = x1i + x3i;                                   \
        const double t3r = x1r - x3r, t3i = x1i - x3i;                                   \
        ar[col]    = t0r + t2r;  ai[col]    = t0i + t2i;                                 \
        ar[30+col] = t0r - t2r;  ai[30+col] = t0i - t2i;                                 \
        ar[15+col] = t1r + t3i;  ai[15+col] = t1i - t3r;   /* y1 = t1 - i*t3 */          \
        ar[45+col] = t1r - t3i;  ai[45+col] = t1i + t3r;   /* y3 = t1 + i*t3 */          \
    }

#define SCALAR_STAGE_B(ar, ai)                                                           \
    for (int n1 = 0; n1 < 4; ++n1)                                                       \
        for (int n3 = 0; n3 < 5; ++n3) {                                                 \
            const int b = n1*15 + n3;                                                    \
            const double x1r = ar[b+5], x1i = ai[b+5], x2r = ar[b+10], x2i = ai[b+10];   \
            const double tr = x1r + x2r, ti = x1i + x2i;                                 \
            const double ur = x1r - x2r, ui = x1i - x2i;                                 \
            const double x0r = ar[b], x0i = ai[b];                                       \
            ar[b] = x0r + tr;  ai[b] = x0i + ti;                                         \
            const double vr = x0r - 0.5*tr, vi = x0i - 0.5*ti;                           \
            const double sr = S3*ur, si = S3*ui;                                         \
            ar[b+5]  = vr + si;  ai[b+5]  = vi - sr;   /* y1 = v - i*s */                \
            ar[b+10] = vr - si;  ai[b+10] = vi + sr;   /* y2 = v + i*s */                \
        }

/* stage C body for block c: computes y0..y4 (re/im doubles) then STORE(k3, yr, yi) */
#define SCALAR_STAGE_C(ar, ai, STORE)                                                    \
    for (int c = 0; c < 12; ++c) {                                                       \
        const int b = 5*c;                                                               \
        const double x0r = ar[b], x0i = ai[b];                                           \
        const double t1r = ar[b+1]+ar[b+4], t1i = ai[b+1]+ai[b+4];                       \
        const double t2r = ar[b+2]+ar[b+3], t2i = ai[b+2]+ai[b+3];                       \
        const double t3r = ar[b+1]-ar[b+4], t3i = ai[b+1]-ai[b+4];                       \
        const double t4r = ar[b+2]-ar[b+3], t4i = ai[b+2]-ai[b+3];                       \
        STORE(0, x0r + t1r + t2r, x0i + t1i + t2i);                                      \
        const double a1r = x0r + C51*t1r + C52*t2r, a1i = x0i + C51*t1i + C52*t2i;       \
        const double a2r = x0r + C52*t1r + C51*t2r, a2i = x0i + C52*t1i + C51*t2i;       \
        const double b1r = S51*t3r + S52*t4r, b1i = S51*t3i + S52*t4i;                   \
        const double b2r = S52*t3r - S51*t4r, b2i = S52*t3i - S51*t4i;                   \
        STORE(1, a1r + b1i, a1i - b1r);   /* y1 = a1 - i*b1 */                           \
        STORE(4, a1r - b1i, a1i + b1r);   /* y4 = a1 + i*b1 */                           \
        STORE(2, a2r + b2i, a2i - b2r);   /* y2 = a2 - i*b2 */                           \
        STORE(3, a2r - b2i, a2i + b2r);   /* y3 = a2 + i*b2 */                           \
    }

/* full scalar transform: interleaved complex x -> interleaved complex y.
 * Reads all of x in stage A before writing y, so x == y is safe. */
static void fft60_scalar(const double *restrict x, double *restrict y) __attribute__((unused));
static void fft60_scalar(const double *restrict x, double *restrict y)
{
    double ar[60], ai[60];
#define LDR(p) x[2*(p)]
#define LDI(p) x[2*(p)+1]
    SCALAR_STAGE_A(ar, ai, LDR, LDI)
#undef LDR
#undef LDI
    SCALAR_STAGE_B(ar, ai)
#define ST(k3, yr, yi) do { const int k = 2*KOUT[b+(k3)]; y[k] = (yr); y[k+1] = (yi); } while (0)
    SCALAR_STAGE_C(ar, ai, ST)
#undef ST
}

/* ================= the map: s = 1/(1 + |z|), 8-wide shuffle-free ============ */
#if HAVE_AVX512
static inline __m512d map_scale8(__m512d zr, __m512d zi)
{
    const __m512d one = _mm512_set1_pd(1.0);
    __m512d q = _mm512_fmadd_pd(zr, zr, _mm512_mul_pd(zi, zi));
#ifndef EXACT_MAP
    /* |z| and the reciprocal via rsqrt14/rcp14 + Newton (2 rounds each -> ~1ulp),
     * avoiding the unpipelined sqrt+div pair. q clamped so q=0 stays finite.
     * A/B vs sqrt+div: chain rel_l2 6.6e-13 vs 6.3e-13 (m=600) -- indistinguishable. */
    const __m512d c15 = _mm512_set1_pd(1.5), two = _mm512_set1_pd(2.0);
    q = _mm512_max_pd(q, _mm512_set1_pd(1e-300));
    __m512d h = _mm512_mul_pd(q, _mm512_set1_pd(0.5));
    __m512d r = _mm512_rsqrt14_pd(q);
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(h, _mm512_mul_pd(r, r), c15));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(h, _mm512_mul_pd(r, r), c15));
    __m512d d = _mm512_fmadd_pd(q, r, one);          /* 1 + q*rsqrt(q) = 1 + |z| */
    __m512d y = _mm512_rcp14_pd(d);
    y = _mm512_mul_pd(y, _mm512_fnmadd_pd(d, y, two));
    y = _mm512_mul_pd(y, _mm512_fnmadd_pd(d, y, two));
    return y;
#else
    return _mm512_div_pd(one, _mm512_add_pd(one, _mm512_sqrt_pd(q)));
#endif
}

/* ============ single-transform kernel, 128-bit complex pairs ============
 * One complex per xmm: a complex add/sub is ONE vaddpd/vsubpd (half the scalar
 * op count), a real-constant multiply is one vfmadd, and (-i)*t costs one
 * in-lane vpermilpd plus a sign-folded constant:
 *   -i*t = swap(t)*( 1,-1)        +i*t = -swap(t)*( 1,-1)
 * so   y = a - i*b  ->  y = a + swap(b)*SE,   y' = a + i*b -> a - swap(b)*SE
 * with the sines folded into SE = (s,-s). */
#define CSWP(v) _mm_permute_pd((v), 1)

/* single-transform ymm kernel, paired over n1: n1 only matters in stage A (the
 * DFT-4 direction), so stage A runs xmm and two vinsertf128 per column build
 * rows (n1=0|n1=1) and (n1=2|n1=3); stages B (over n2) and C (over n3) then run
 * two-wide with no repacking at all. */
static void fft60_ymm1(const double *restrict x, double *restrict y)
{
    __m256d wp[2][15];   /* [n1-pair][n2*5+n3] */
#ifndef STAGEA_XMM
    {
        /* DFT-4 in ymm: with A=(x0|x1), B=(x2|x3): S=A+B=(t0|t2), D=A-B=(t1|t3);
         * P=(t0|t1), Q=(t2|t3); R=permil(Q,6)=(t2 | swap(t3)); E4=(1,1,1,-1):
         * P+R*E4 = (t0+t2 | t1-i*t3) = (y0|y1), P-R*E4 = (y2|y3). */
        const __m256d E4 = _mm256_set_pd(-1.0, 1.0, 1.0, 1.0);
        for (int col = 0; col < 15; ++col) {
            __m256d A = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm_loadu_pd(x + 2*PIN[col])),
                _mm_loadu_pd(x + 2*PIN[15+col]), 1);
            __m256d B = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm_loadu_pd(x + 2*PIN[30+col])),
                _mm_loadu_pd(x + 2*PIN[45+col]), 1);
            __m256d S = _mm256_add_pd(A, B), D = _mm256_sub_pd(A, B);
            __m256d P = _mm256_permute2f128_pd(S, D, 0x20);
            __m256d R = _mm256_permute_pd(_mm256_permute2f128_pd(S, D, 0x31), 6);
            wp[0][col] = _mm256_fmadd_pd(R, E4, P);
            wp[1][col] = _mm256_fnmadd_pd(R, E4, P);
        }
    }
#else
    {
        const __m128d E = _mm_set_pd(-1.0, 1.0);
        for (int col = 0; col < 15; ++col) {
            __m128d x0 = _mm_loadu_pd(x + 2*PIN[col]);
            __m128d x1 = _mm_loadu_pd(x + 2*PIN[15+col]);
            __m128d x2 = _mm_loadu_pd(x + 2*PIN[30+col]);
            __m128d x3 = _mm_loadu_pd(x + 2*PIN[45+col]);
            __m128d t0 = _mm_add_pd(x0, x2), t1 = _mm_sub_pd(x0, x2);
            __m128d t2 = _mm_add_pd(x1, x3), t3 = _mm_sub_pd(x1, x3);
            __m128d sw = CSWP(t3);
            wp[0][col] = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm_add_pd(t0, t2)), _mm_fmadd_pd(sw, E, t1), 1);
            wp[1][col] = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm_sub_pd(t0, t2)), _mm_fnmadd_pd(sw, E, t1), 1);
        }
    }
#endif
    {
        const __m256d half = _mm256_set1_pd(0.5);
        const __m256d S3E = _mm256_set_pd(-S3, S3, -S3, S3);
        for (int pr = 0; pr < 2; ++pr)
            for (int n3 = 0; n3 < 5; ++n3) {
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
        for (int pr = 0; pr < 2; ++pr)
            for (int n2 = 0; n2 < 3; ++n2) {
                const int b = 5*n2;                        /* base in 15-space */
                const int bl = (2*pr)*15 + 5*n2, bh = (2*pr+1)*15 + 5*n2; /* 60-space */
                __m256d x0 = wp[pr][b];
                __m256d t1 = _mm256_add_pd(wp[pr][b+1], wp[pr][b+4]);
                __m256d t3 = _mm256_sub_pd(wp[pr][b+1], wp[pr][b+4]);
                __m256d t2 = _mm256_add_pd(wp[pr][b+2], wp[pr][b+3]);
                __m256d t4 = _mm256_sub_pd(wp[pr][b+2], wp[pr][b+3]);
#define STP(k3, v) do { __m256d v_ = (v);                                        \
        _mm_storeu_pd(y + 2*KOUT[bl+(k3)], _mm256_castpd256_pd128(v_));          \
        _mm_storeu_pd(y + 2*KOUT[bh+(k3)], _mm256_extractf128_pd(v_, 1)); } while (0)
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

/* batched execute: TWO transforms per ymm (transform b in the low 128 lane,
 * b+1 in the high lane). Same dataflow as the xmm kernel -- the +-i swap is the
 * in-lane _mm256_permute_pd(v,5) -- at half the arithmetic per transform, for
 * one vinsertf128 per load pair and one vextractf128 per store pair. */
static void fft60_ymm2(const double *restrict x, double *restrict y)
{
    __m256d w[60];
#define LD2(p) _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd((x) + 2*(p))), \
                                    _mm_loadu_pd((x) + 120 + 2*(p)), 1)
#define ST2(p, v) do { __m256d v_ = (v);                                   \
        _mm_storeu_pd((y) + 2*(p), _mm256_castpd256_pd128(v_));            \
        _mm_storeu_pd((y) + 120 + 2*(p), _mm256_extractf128_pd(v_, 1)); } while (0)
#define CSWP2(v) _mm256_permute_pd((v), 0x5)
    {
        const __m256d E = _mm256_set_pd(-1.0, 1.0, -1.0, 1.0);
        const __m256d half = _mm256_set1_pd(0.5);
        const __m256d S3E = _mm256_set_pd(-S3, S3, -S3, S3);
        for (int n3 = 0; n3 < 5; ++n3) {
            __m256d v[4][3];
            for (int n2 = 0; n2 < 3; ++n2) {
                const int col = n2*5 + n3;
                __m256d x0 = LD2(PIN[col]),    x1 = LD2(PIN[15+col]);
                __m256d x2 = LD2(PIN[30+col]), x3 = LD2(PIN[45+col]);
                __m256d t0 = _mm256_add_pd(x0, x2), t1 = _mm256_sub_pd(x0, x2);
                __m256d t2 = _mm256_add_pd(x1, x3), t3 = _mm256_sub_pd(x1, x3);
                v[0][n2] = _mm256_add_pd(t0, t2);
                v[2][n2] = _mm256_sub_pd(t0, t2);
                __m256d sw = CSWP2(t3);
                v[1][n2] = _mm256_fmadd_pd(sw, E, t1);
                v[3][n2] = _mm256_fnmadd_pd(sw, E, t1);
            }
            for (int n1 = 0; n1 < 4; ++n1) {
                const int b = n1*15 + n3;
                __m256d t = _mm256_add_pd(v[n1][1], v[n1][2]);
                __m256d u = _mm256_sub_pd(v[n1][1], v[n1][2]);
                w[b] = _mm256_add_pd(v[n1][0], t);
                __m256d vv = _mm256_fnmadd_pd(half, t, v[n1][0]);
                __m256d swu = CSWP2(u);
                w[b+5]  = _mm256_fmadd_pd(swu, S3E, vv);
                w[b+10] = _mm256_fnmadd_pd(swu, S3E, vv);
            }
        }
        const __m256d c51v = _mm256_set1_pd(C51), c52v = _mm256_set1_pd(C52);
        const __m256d S1E = _mm256_set_pd(-S51, S51, -S51, S51);
        const __m256d S2E = _mm256_set_pd(-S52, S52, -S52, S52);
        for (int c = 0; c < 12; ++c) {
            const int b = 5*c;
            __m256d x0 = w[b];
            __m256d t1 = _mm256_add_pd(w[b+1], w[b+4]), t3 = _mm256_sub_pd(w[b+1], w[b+4]);
            __m256d t2 = _mm256_add_pd(w[b+2], w[b+3]), t4 = _mm256_sub_pd(w[b+2], w[b+3]);
            ST2(KOUT[b], _mm256_add_pd(x0, _mm256_add_pd(t1, t2)));
            __m256d a1 = _mm256_fmadd_pd(c52v, t2, _mm256_fmadd_pd(c51v, t1, x0));
            __m256d a2 = _mm256_fmadd_pd(c51v, t2, _mm256_fmadd_pd(c52v, t1, x0));
            __m256d sw3 = CSWP2(t3), sw4 = CSWP2(t4);
            __m256d m1 = _mm256_fmadd_pd(sw4, S2E, _mm256_mul_pd(sw3, S1E));
            __m256d m2 = _mm256_fnmadd_pd(sw4, S1E, _mm256_mul_pd(sw3, S2E));
            ST2(KOUT[b+1], _mm256_add_pd(a1, m1));
            ST2(KOUT[b+4], _mm256_sub_pd(a1, m1));
            ST2(KOUT[b+2], _mm256_add_pd(a2, m2));
            ST2(KOUT[b+3], _mm256_sub_pd(a2, m2));
        }
    }
#undef LD2
#undef ST2
#undef CSWP2
}

/* batched execute: FOUR transforms per zmm (complex of transform b+l in 128-bit
 * lane l). Same dataflow as fft60_ymm2 widened to 512 bits. */
static void fft60_zmm4(const double *restrict x, double *restrict y) __attribute__((unused));
static void fft60_zmm4(const double *restrict x, double *restrict y)
{
    __m512d w[60];
#define LD4(p) _mm512_insertf64x4(                                                     \
        _mm512_castpd256_pd512(_mm256_insertf128_pd(                                   \
            _mm256_castpd128_pd256(_mm_loadu_pd((x) + 2*(p))),                         \
            _mm_loadu_pd((x) + 120 + 2*(p)), 1)),                                      \
        _mm256_insertf128_pd(                                                          \
            _mm256_castpd128_pd256(_mm_loadu_pd((x) + 240 + 2*(p))),                   \
            _mm_loadu_pd((x) + 360 + 2*(p)), 1), 1)
#define ST4(p, v) do { __m512d v_ = (v);                                               \
        __m256d lo_ = _mm512_castpd512_pd256(v_), hi_ = _mm512_extractf64x4_pd(v_, 1); \
        _mm_storeu_pd((y) + 2*(p),       _mm256_castpd256_pd128(lo_));                 \
        _mm_storeu_pd((y) + 120 + 2*(p), _mm256_extractf128_pd(lo_, 1));               \
        _mm_storeu_pd((y) + 240 + 2*(p), _mm256_castpd256_pd128(hi_));                 \
        _mm_storeu_pd((y) + 360 + 2*(p), _mm256_extractf128_pd(hi_, 1)); } while (0)
#define CSWP4(v) _mm512_permute_pd((v), 0x55)
    {
        const __m512d E = _mm512_set4_pd(-1.0, 1.0, -1.0, 1.0);
        const __m512d half = _mm512_set1_pd(0.5);
        const __m512d S3E = _mm512_set4_pd(-S3, S3, -S3, S3);
        for (int n3 = 0; n3 < 5; ++n3) {
            __m512d v[4][3];
            for (int n2 = 0; n2 < 3; ++n2) {
                const int col = n2*5 + n3;
                __m512d x0 = LD4(PIN[col]),    x1 = LD4(PIN[15+col]);
                __m512d x2 = LD4(PIN[30+col]), x3 = LD4(PIN[45+col]);
                __m512d t0 = _mm512_add_pd(x0, x2), t1 = _mm512_sub_pd(x0, x2);
                __m512d t2 = _mm512_add_pd(x1, x3), t3 = _mm512_sub_pd(x1, x3);
                v[0][n2] = _mm512_add_pd(t0, t2);
                v[2][n2] = _mm512_sub_pd(t0, t2);
                __m512d sw = CSWP4(t3);
                v[1][n2] = _mm512_fmadd_pd(sw, E, t1);
                v[3][n2] = _mm512_fnmadd_pd(sw, E, t1);
            }
            for (int n1 = 0; n1 < 4; ++n1) {
                const int b = n1*15 + n3;
                __m512d t = _mm512_add_pd(v[n1][1], v[n1][2]);
                __m512d u = _mm512_sub_pd(v[n1][1], v[n1][2]);
                w[b] = _mm512_add_pd(v[n1][0], t);
                __m512d vv = _mm512_fnmadd_pd(half, t, v[n1][0]);
                __m512d swu = CSWP4(u);
                w[b+5]  = _mm512_fmadd_pd(swu, S3E, vv);
                w[b+10] = _mm512_fnmadd_pd(swu, S3E, vv);
            }
        }
        const __m512d c51v = _mm512_set1_pd(C51), c52v = _mm512_set1_pd(C52);
        const __m512d S1E = _mm512_set4_pd(-S51, S51, -S51, S51);
        const __m512d S2E = _mm512_set4_pd(-S52, S52, -S52, S52);
        for (int c = 0; c < 12; ++c) {
            const int b = 5*c;
            __m512d x0 = w[b];
            __m512d t1 = _mm512_add_pd(w[b+1], w[b+4]), t3 = _mm512_sub_pd(w[b+1], w[b+4]);
            __m512d t2 = _mm512_add_pd(w[b+2], w[b+3]), t4 = _mm512_sub_pd(w[b+2], w[b+3]);
            ST4(KOUT[b], _mm512_add_pd(x0, _mm512_add_pd(t1, t2)));
            __m512d a1 = _mm512_fmadd_pd(c52v, t2, _mm512_fmadd_pd(c51v, t1, x0));
            __m512d a2 = _mm512_fmadd_pd(c51v, t2, _mm512_fmadd_pd(c52v, t1, x0));
            __m512d sw3 = CSWP4(t3), sw4 = CSWP4(t4);
            __m512d m1 = _mm512_fmadd_pd(sw4, S2E, _mm512_mul_pd(sw3, S1E));
            __m512d m2 = _mm512_fnmadd_pd(sw4, S1E, _mm512_mul_pd(sw3, S2E));
            ST4(KOUT[b+1], _mm512_add_pd(a1, m1));
            ST4(KOUT[b+4], _mm512_sub_pd(a1, m1));
            ST4(KOUT[b+2], _mm512_add_pd(a2, m2));
            ST4(KOUT[b+3], _mm512_sub_pd(a2, m2));
        }
    }
#undef LD4
#undef ST4
#undef CSWP4
}

/* one chain step in place on interleaved state st; c pre-split (64-padded, zeros).
 * Same n1-paired ymm dataflow as fft60_ymm1, but stage C lands the outputs in
 * SPLIT zr/zi buffers so the fused map runs 8-wide with zero shuffles. */
static void chain60_ymm_step(double *restrict st,
                             const double *restrict cr, const double *restrict ci)
{
    _Alignas(64) double zr[64], zi[64];
    __m256d wp[2][15];
    {
        const __m256d E4 = _mm256_set_pd(-1.0, 1.0, 1.0, 1.0);
        for (int col = 0; col < 15; ++col) {
            __m256d A = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm_loadu_pd(st + 2*PIN[col])),
                _mm_loadu_pd(st + 2*PIN[15+col]), 1);
            __m256d B = _mm256_insertf128_pd(
                _mm256_castpd128_pd256(_mm_loadu_pd(st + 2*PIN[30+col])),
                _mm_loadu_pd(st + 2*PIN[45+col]), 1);
            __m256d S = _mm256_add_pd(A, B), D = _mm256_sub_pd(A, B);
            __m256d P = _mm256_permute2f128_pd(S, D, 0x20);
            __m256d R = _mm256_permute_pd(_mm256_permute2f128_pd(S, D, 0x31), 6);
            wp[0][col] = _mm256_fmadd_pd(R, E4, P);
            wp[1][col] = _mm256_fnmadd_pd(R, E4, P);
        }
        const __m256d half = _mm256_set1_pd(0.5);
        const __m256d S3E = _mm256_set_pd(-S3, S3, -S3, S3);
        for (int pr = 0; pr < 2; ++pr)
            for (int n3 = 0; n3 < 5; ++n3) {
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
        for (int pr = 0; pr < 2; ++pr)
            for (int n2 = 0; n2 < 3; ++n2) {
                const int b = 5*n2;
                const int bl = (2*pr)*15 + 5*n2, bh = (2*pr+1)*15 + 5*n2;
                __m256d x0 = wp[pr][b];
                __m256d t1 = _mm256_add_pd(wp[pr][b+1], wp[pr][b+4]);
                __m256d t3 = _mm256_sub_pd(wp[pr][b+1], wp[pr][b+4]);
                __m256d t2 = _mm256_add_pd(wp[pr][b+2], wp[pr][b+3]);
                __m256d t4 = _mm256_sub_pd(wp[pr][b+2], wp[pr][b+3]);
#define STZ(k3, v) do { __m256d v_ = (v);                                     \
        __m128d lo_ = _mm256_castpd256_pd128(v_);                             \
        __m128d hi_ = _mm256_extractf128_pd(v_, 1);                           \
        _mm_storel_pd(zr + KOUT[bl+(k3)], lo_); _mm_storeh_pd(zi + KOUT[bl+(k3)], lo_); \
        _mm_storel_pd(zr + KOUT[bh+(k3)], hi_); _mm_storeh_pd(zi + KOUT[bh+(k3)], hi_); } while (0)
                STZ(0, _mm256_add_pd(x0, _mm256_add_pd(t1, t2)));
                __m256d a1 = _mm256_fmadd_pd(c52v, t2, _mm256_fmadd_pd(c51v, t1, x0));
                __m256d a2 = _mm256_fmadd_pd(c51v, t2, _mm256_fmadd_pd(c52v, t1, x0));
                __m256d sw3 = _mm256_permute_pd(t3, 0x5), sw4 = _mm256_permute_pd(t4, 0x5);
                __m256d m1 = _mm256_fmadd_pd(sw4, S2E, _mm256_mul_pd(sw3, S1E));
                __m256d m2 = _mm256_fnmadd_pd(sw4, S1E, _mm256_mul_pd(sw3, S2E));
                STZ(1, _mm256_add_pd(a1, m1));
                STZ(4, _mm256_sub_pd(a1, m1));
                STZ(2, _mm256_add_pd(a2, m2));
                STZ(3, _mm256_sub_pd(a2, m2));
#undef STZ
            }
    }
    zr[60]=zr[61]=zr[62]=zr[63]=0.0; zi[60]=zi[61]=zi[62]=zi[63]=0.0;
    /* interleave indices: (r0,i0,r1,i1,...) from (r-vector, i-vector) */
    const __m512i ILO = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i IHI = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    for (int i = 0; i < 56; i += 8) {
        __m512d vr = _mm512_add_pd(_mm512_load_pd(zr+i), _mm512_loadu_pd(cr+i));
        __m512d vi = _mm512_add_pd(_mm512_load_pd(zi+i), _mm512_loadu_pd(ci+i));
        __m512d s = map_scale8(vr, vi);
        vr = _mm512_mul_pd(vr, s); vi = _mm512_mul_pd(vi, s);
        _mm512_storeu_pd(st + 2*i,     _mm512_permutex2var_pd(vr, ILO, vi));
        _mm512_storeu_pd(st + 2*i + 8, _mm512_permutex2var_pd(vr, IHI, vi));
    }
    {   /* tail: complexes 56..59 (padded lanes are zero: s = 1, harmless) */
        __m512d vr = _mm512_add_pd(_mm512_load_pd(zr+56), _mm512_loadu_pd(cr+56));
        __m512d vi = _mm512_add_pd(_mm512_load_pd(zi+56), _mm512_loadu_pd(ci+56));
        __m512d s = map_scale8(vr, vi);
        vr = _mm512_mul_pd(vr, s); vi = _mm512_mul_pd(vi, s);
        _mm512_storeu_pd(st + 112, _mm512_permutex2var_pd(vr, ILO, vi));
    }
}

/* whole chain for one transform, state living in `out` (interleaved) */
static void chain60_scalar(const double *restrict x0, const double *restrict c,
                           double *restrict out, int m)
{
    _Alignas(64) double cr[64], ci[64];
    for (int i = 0; i < 60; ++i) { cr[i] = c[2*i]; ci[i] = c[2*i+1]; }
    for (int i = 60; i < 64; ++i) { cr[i] = 0.0; ci[i] = 0.0; }
    memcpy(out, x0, 60*sizeof(double _Complex));
    for (int s = 0; s < m; ++s) chain60_ymm_step(out, cr, ci);
}

#else  /* !HAVE_AVX512: portable scalar chain */

static void chain60_scalar(const double *restrict x0, const double *restrict c,
                           double *restrict out, int m)
{
    double z[120];
    memcpy(out, x0, 60*sizeof(double _Complex));
    for (int s = 0; s < m; ++s) {
        fft60_scalar(out, z);
        for (int i = 0; i < 60; ++i) {
            const double vr = z[2*i] + c[2*i], vi = z[2*i+1] + c[2*i+1];
            const double sc = 1.0 / (1.0 + sqrt(vr*vr + vi*vi));
            out[2*i] = vr * sc;  out[2*i+1] = vi * sc;
        }
    }
}
#endif

#if HAVE_AVX512
/* ================= AVX-512 across-batch path: 8 transforms per register =====
 * Row layout: row j = 16 doubles at p + 16*j -- re lanes at +0, im lanes at +8.
 * Lane l = transform b0+l. All arithmetic is full-width and shuffle-free. */

static const long long GIDX8[8] = {0,120,240,360,480,600,720,840}; /* doubles: b*60 complexes */

#define ROWR(p, j) ((p) + 16*(j))
#define ROWI(p, j) ((p) + 16*(j) + 8)

/* stages A and B, in place on a row buffer -- fused per n3-slice like the xmm
 * kernel: the 12 DFT-4 outputs of cols {n3,n3+5,n3+10} feed the four DFT-3s at
 * that n3 directly from registers. */
static inline void stages_AB_soa(double *restrict w)
{
    const __m512d half = _mm512_set1_pd(0.5), s3v = _mm512_set1_pd(S3);
    for (int n3 = 0; n3 < 5; ++n3) {
        __m512d vr[4][3], vi[4][3];
        for (int n2 = 0; n2 < 3; ++n2) {
            const int col = n2*5 + n3;
            __m512d x0r = _mm512_load_pd(ROWR(w,col)),    x0i = _mm512_load_pd(ROWI(w,col));
            __m512d x1r = _mm512_load_pd(ROWR(w,15+col)), x1i = _mm512_load_pd(ROWI(w,15+col));
            __m512d x2r = _mm512_load_pd(ROWR(w,30+col)), x2i = _mm512_load_pd(ROWI(w,30+col));
            __m512d x3r = _mm512_load_pd(ROWR(w,45+col)), x3i = _mm512_load_pd(ROWI(w,45+col));
            __m512d t0r = _mm512_add_pd(x0r, x2r), t0i = _mm512_add_pd(x0i, x2i);
            __m512d t1r = _mm512_sub_pd(x0r, x2r), t1i = _mm512_sub_pd(x0i, x2i);
            __m512d t2r = _mm512_add_pd(x1r, x3r), t2i = _mm512_add_pd(x1i, x3i);
            __m512d t3r = _mm512_sub_pd(x1r, x3r), t3i = _mm512_sub_pd(x1i, x3i);
            vr[0][n2] = _mm512_add_pd(t0r, t2r);  vi[0][n2] = _mm512_add_pd(t0i, t2i);
            vr[2][n2] = _mm512_sub_pd(t0r, t2r);  vi[2][n2] = _mm512_sub_pd(t0i, t2i);
            vr[1][n2] = _mm512_add_pd(t1r, t3i);  vi[1][n2] = _mm512_sub_pd(t1i, t3r);
            vr[3][n2] = _mm512_sub_pd(t1r, t3i);  vi[3][n2] = _mm512_add_pd(t1i, t3r);
        }
        for (int n1 = 0; n1 < 4; ++n1) {
            const int b = n1*15 + n3;
            __m512d tr = _mm512_add_pd(vr[n1][1], vr[n1][2]), ti = _mm512_add_pd(vi[n1][1], vi[n1][2]);
            __m512d ur = _mm512_sub_pd(vr[n1][1], vr[n1][2]), ui = _mm512_sub_pd(vi[n1][1], vi[n1][2]);
            _mm512_store_pd(ROWR(w,b), _mm512_add_pd(vr[n1][0], tr));
            _mm512_store_pd(ROWI(w,b), _mm512_add_pd(vi[n1][0], ti));
            __m512d wr = _mm512_fnmadd_pd(half, tr, vr[n1][0]), wi = _mm512_fnmadd_pd(half, ti, vi[n1][0]);
            __m512d sr = _mm512_mul_pd(s3v, ur), si = _mm512_mul_pd(s3v, ui);
            _mm512_store_pd(ROWR(w,b+5),  _mm512_add_pd(wr, si)); _mm512_store_pd(ROWI(w,b+5),  _mm512_sub_pd(wi, sr));
            _mm512_store_pd(ROWR(w,b+10), _mm512_sub_pd(wr, si)); _mm512_store_pd(ROWI(w,b+10), _mm512_add_pd(wi, sr));
        }
    }
}

/* stage C for one 5-row block of w: the five outputs land in yr[0..4]/yi[0..4].
 * Needs c51v/c52v/s51v/s52v in scope. */
#define STAGE_C_BLOCK(w, b, yr, yi)                                                     \
    do {                                                                                \
        __m512d x0r = _mm512_load_pd(ROWR(w,b)),   x0i = _mm512_load_pd(ROWI(w,b));     \
        __m512d p1r = _mm512_load_pd(ROWR(w,(b)+1)), p1i = _mm512_load_pd(ROWI(w,(b)+1)); \
        __m512d p2r = _mm512_load_pd(ROWR(w,(b)+2)), p2i = _mm512_load_pd(ROWI(w,(b)+2)); \
        __m512d p3r = _mm512_load_pd(ROWR(w,(b)+3)), p3i = _mm512_load_pd(ROWI(w,(b)+3)); \
        __m512d p4r = _mm512_load_pd(ROWR(w,(b)+4)), p4i = _mm512_load_pd(ROWI(w,(b)+4)); \
        __m512d t1r = _mm512_add_pd(p1r, p4r), t1i = _mm512_add_pd(p1i, p4i);           \
        __m512d t2r = _mm512_add_pd(p2r, p3r), t2i = _mm512_add_pd(p2i, p3i);           \
        __m512d t3r = _mm512_sub_pd(p1r, p4r), t3i = _mm512_sub_pd(p1i, p4i);           \
        __m512d t4r = _mm512_sub_pd(p2r, p3r), t4i = _mm512_sub_pd(p2i, p3i);           \
        yr[0] = _mm512_add_pd(x0r, _mm512_add_pd(t1r, t2r));                            \
        yi[0] = _mm512_add_pd(x0i, _mm512_add_pd(t1i, t2i));                            \
        __m512d a1r = _mm512_fmadd_pd(c52v, t2r, _mm512_fmadd_pd(c51v, t1r, x0r));      \
        __m512d a1i = _mm512_fmadd_pd(c52v, t2i, _mm512_fmadd_pd(c51v, t1i, x0i));      \
        __m512d a2r = _mm512_fmadd_pd(c51v, t2r, _mm512_fmadd_pd(c52v, t1r, x0r));      \
        __m512d a2i = _mm512_fmadd_pd(c51v, t2i, _mm512_fmadd_pd(c52v, t1i, x0i));      \
        __m512d b1r = _mm512_fmadd_pd(s52v, t4r, _mm512_mul_pd(s51v, t3r));             \
        __m512d b1i = _mm512_fmadd_pd(s52v, t4i, _mm512_mul_pd(s51v, t3i));             \
        __m512d b2r = _mm512_fnmadd_pd(s51v, t4r, _mm512_mul_pd(s52v, t3r));            \
        __m512d b2i = _mm512_fnmadd_pd(s51v, t4i, _mm512_mul_pd(s52v, t3i));            \
        yr[1] = _mm512_add_pd(a1r, b1i); yi[1] = _mm512_sub_pd(a1i, b1r);               \
        yr[4] = _mm512_sub_pd(a1r, b1i); yi[4] = _mm512_add_pd(a1i, b1r);               \
        yr[2] = _mm512_add_pd(a2r, b2i); yi[2] = _mm512_sub_pd(a2i, b2r);               \
        yr[3] = _mm512_sub_pd(a2r, b2i); yi[3] = _mm512_add_pd(a2i, b2r);               \
    } while (0)

#define DECL_C_CONSTS                                                     \
    const __m512d c51v = _mm512_set1_pd(C51), c52v = _mm512_set1_pd(C52); \
    const __m512d s51v = _mm512_set1_pd(S51), s52v = _mm512_set1_pd(S52)

/* one chain step for one 8-transform group, fully in place on the SoA state.
 * Stage C blocks are processed in SROW-involution pairs: both blocks' outputs
 * are computed and mapped before either block's rows are overwritten. */
static void chain60_soa8_step(double *restrict st, const double *restrict cs)
{
    stages_AB_soa(st);
    DECL_C_CONSTS;
    for (int p = 0; p < 7; ++p) {
        const int bP = 5*CPAIR[p][0], bQ = 5*CPAIR[p][1];
        __m512d Pr[5], Pi[5];
        STAGE_C_BLOCK(st, bP, Pr, Pi);
        for (int j = 0; j < 5; ++j) {   /* map P, hold in registers */
            const int slot = bP + j;
            __m512d zr = _mm512_add_pd(Pr[j], _mm512_load_pd(cs + 16*slot));
            __m512d zi = _mm512_add_pd(Pi[j], _mm512_load_pd(cs + 16*slot + 8));
            __m512d s = map_scale8(zr, zi);
            Pr[j] = _mm512_mul_pd(zr, s);  Pi[j] = _mm512_mul_pd(zi, s);
        }
        if (bQ != bP) {
            __m512d Qr[5], Qi[5];
            STAGE_C_BLOCK(st, bQ, Qr, Qi);
            for (int j = 0; j < 5; ++j) {
                const int slot = bQ + j;
                __m512d zr = _mm512_add_pd(Qr[j], _mm512_load_pd(cs + 16*slot));
                __m512d zi = _mm512_add_pd(Qi[j], _mm512_load_pd(cs + 16*slot + 8));
                __m512d s = map_scale8(zr, zi);
                _mm512_store_pd(ROWR(st, SROW[slot]), _mm512_mul_pd(zr, s));
                _mm512_store_pd(ROWI(st, SROW[slot]), _mm512_mul_pd(zi, s));
            }
        }
        for (int j = 0; j < 5; ++j) {
            _mm512_store_pd(ROWR(st, SROW[bP+j]), Pr[j]);
            _mm512_store_pd(ROWI(st, SROW[bP+j]), Pi[j]);
        }
    }
}
#endif /* HAVE_AVX512 */

/* ================= plan / execute / chain / destroy ================= */

fft1d_plan *fft1d_create(int L, int batch)
{
    if (!fft1d_supports(L)) return NULL;
    fft1d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L; p->batch = batch;
    if (L != 60) {          /* dense floor for the unmeasured sizes */
        p->w = malloc((size_t)L*L*sizeof *p->w);
        if (!p->w) { free(p); return NULL; }
        for (int k = 0; k < L; ++k)
            for (int j = 0; j < L; ++j) {
                double ph = -2.0*M_PI*((k*j)%L)/L;
                p->w[(size_t)k*L+j] = cos(ph) + I*sin(ph);
            }
    }
#if HAVE_AVX512
    if (L == 60 && batch >= 8) {
        const size_t ngroup = (size_t)(batch / 8);
        p->soa_state = aligned_alloc(64, ngroup * 60 * 16 * sizeof(double));
        p->soa_c     = aligned_alloc(64, ngroup * 60 * 16 * sizeof(double));
        if (!p->soa_state || !p->soa_c) { free(p->soa_state); free(p->soa_c); free(p); return NULL; }
    }
#endif
    return p;
}

void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    const int L = p->L;
    if (L == 60) {
        /* pairwise ymm kernel, xmm for the remainder. A/B history: the 8-lane
         * gather/scatter SoA path lost to the plain xmm loop (0.129 vs 0.061 us
         * at B=512) -- cross-batch transposes cost more than wide lanes save in
         * a single pass (chains are a different story). */
        int b = 0;
#if HAVE_AVX512
#ifdef USE_ZMM4   /* A/B on wallaby (SPR): statistical tie with ymm2; ymm2 kept
                     as default to avoid 512-bit clock licensing in this cell */
        for (; b + 4 <= p->batch; b += 4)
            fft60_zmm4((const double*)(in + (size_t)b*60), (double*)(out + (size_t)b*60));
#endif
        for (; b + 2 <= p->batch; b += 2)
            fft60_ymm2((const double*)(in + (size_t)b*60), (double*)(out + (size_t)b*60));
        for (; b < p->batch; ++b)
            fft60_ymm1((const double*)(in + (size_t)b*60), (double*)(out + (size_t)b*60));
#else
        for (; b < p->batch; ++b)
            fft60_scalar((const double*)(in + (size_t)b*60), (double*)(out + (size_t)b*60));
#endif
        return;
    }
    for (int b = 0; b < p->batch; ++b) {
        const double _Complex *x = in + (size_t)b*L;
        double _Complex *y = out + (size_t)b*L;
        for (int k = 0; k < L; ++k) {
            double _Complex s = 0;
            const double _Complex *wrow = p->w + (size_t)k*L;
            for (int j = 0; j < L; ++j) s += wrow[j]*x[j];
            y[k] = s;
        }
    }
}

/* own the whole m-step map chain: state <- (FFT(state)+c)/(1+|FFT(state)+c|) */
void fft1d_chain(fft1d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    const int L = p->L, B = p->batch;
    if (L != 60) {   /* generic fallback for the dense sizes */
        double _Complex *z = malloc((size_t)L*sizeof *z);
        for (int b = 0; b < B; ++b) {
            const double _Complex *src = x0 + (size_t)b*L;
            double _Complex *st = final_out + (size_t)b*L;
            const double *cb = (const double*)(c + (size_t)b*L);
            for (int s = 0; s < m; ++s) {
                for (int k = 0; k < L; ++k) {
                    double _Complex acc = 0;
                    const double _Complex *wrow = p->w + (size_t)k*L;
                    for (int j = 0; j < L; ++j) acc += wrow[j]*src[j];
                    z[k] = acc;
                }
                double *zd = (double*)z;
                for (int i = 0; i < L; ++i) {
                    double zr = zd[2*i] + cb[2*i], zi = zd[2*i+1] + cb[2*i+1];
                    double sc = 1.0/(1.0 + sqrt(zr*zr + zi*zi));
                    ((double*)st)[2*i] = zr*sc; ((double*)st)[2*i+1] = zi*sc;
                }
                src = st;
            }
        }
        free(z);
        return;
    }
    int b = 0;
#if HAVE_AVX512
    if (p->soa_state) {
        const __m512i vg = _mm512_loadu_si512((const void*)GIDX8);
        const int ngroup = B / 8;
        /* transpose in: state rows in PFA input order; c rows in stage-C slot order */
        for (int g = 0; g < ngroup; ++g) {
            const double *xb = (const double*)(x0 + (size_t)g*8*60);
            const double *cb = (const double*)(c  + (size_t)g*8*60);
            double *st = p->soa_state + (size_t)g*60*16;
            double *cs = p->soa_c     + (size_t)g*60*16;
            for (int j = 0; j < 60; ++j) {
                _mm512_store_pd(ROWR(st,j), _mm512_i64gather_pd(vg, xb + 2*PIN[j],      8));
                _mm512_store_pd(ROWI(st,j), _mm512_i64gather_pd(vg, xb + 2*PIN[j] + 1,  8));
                _mm512_store_pd(ROWR(cs,j), _mm512_i64gather_pd(vg, cb + 2*KOUT[j],     8));
                _mm512_store_pd(ROWI(cs,j), _mm512_i64gather_pd(vg, cb + 2*KOUT[j] + 1, 8));
            }
        }
        for (int s = 0; s < m; ++s)
            for (int g = 0; g < ngroup; ++g)
                chain60_soa8_step(p->soa_state + (size_t)g*60*16, p->soa_c + (size_t)g*60*16);
        /* transpose out: state row j holds natural index PIN[j] */
        for (int g = 0; g < ngroup; ++g) {
            double *yb = (double*)(final_out + (size_t)g*8*60);
            const double *st = p->soa_state + (size_t)g*60*16;
            for (int j = 0; j < 60; ++j) {
                _mm512_i64scatter_pd(yb + 2*PIN[j],     vg, _mm512_load_pd(ROWR(st,j)), 8);
                _mm512_i64scatter_pd(yb + 2*PIN[j] + 1, vg, _mm512_load_pd(ROWI(st,j)), 8);
            }
        }
        b = ngroup * 8;
    }
#endif
    for (; b < B; ++b)   /* B<8 and remainders: split-state scalar chain */
        chain60_scalar((const double*)(x0 + (size_t)b*60), (const double*)(c + (size_t)b*60),
                       (double*)(final_out + (size_t)b*60), m);
}

void fft1d_destroy(fft1d_plan *p)
{
    if (!p) return;
    free(p->w);
#if HAVE_AVX512
    free(p->soa_state); free(p->soa_c);
#endif
    free(p);
}
