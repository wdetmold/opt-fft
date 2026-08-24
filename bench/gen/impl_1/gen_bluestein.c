/* gen_bluestein: Bluestein chirp-Z existence fallback for ANY L.
 *
 * X_k = ch[k] * (a (*) b)[k],   a_j = x_j * ch[j],  ch[n] = exp(-i pi n^2 / L),
 * b_n = exp(+i pi n^2 / L) embedded circularly in M = next_pow2(2L-1).
 *
 * The circular convolution runs through our own power-of-two FFT:
 *   forward  = in-place radix-4 DIF (radix-2 tail when log2 M is odd),
 *   pointwise multiply by FFT(b)/M in the SAME scrambled order,
 *   inverse  = the exact stage-by-stage inverse (radix-4 DIT, mirrored order).
 * Because the inverse is the literal inverse of the forward pipeline, no
 * bit/digit-reversal permutation is ever materialized and correctness does not
 * depend on the scramble order at all.
 *
 * Rows are transformed 8 at a time: work buffers are split re/im with 8 row
 * lanes contiguous per element (one zmm of doubles), so every butterfly is a
 * unit-stride 8-wide loop the compiler vectorizes.  Chirp and twiddle tables
 * are built in long double with exact integer phase reduction (j^2 mod 2L).
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include "../fft3d_api.h"

#define VL 8 /* row lanes per group: 8 doubles = one zmm */

static const long double PIL = 3.141592653589793238462643383279502884L;

struct fft3d_plan {
    int L, batch, M;
    double *chre, *chim; /* chirp exp(-i pi j^2/L), length L */
    double *bhre, *bhim; /* forward-DIF(b_pad)/M, scrambled order, length M */
    double *twf;         /* forward twiddles, consumption order */
    double *twi;         /* inverse twiddles (conjugates), consumption order */
    double *wr, *wi;     /* work: M * VL each */
};

const char *fft3d_name(void) { return "gen_bluestein"; }
const char *fft3d_description(void)
{
    return "Bluestein chirp-Z for ANY L: pow2 radix-4 DIF/DIT convolution (no bit-reversal), 8-row SoA lanes, exact long-double chirp tables";
}
int fft3d_supports(int L) { return L >= 2 && L <= 2048; }

/* ---------------- pow2 FFT stages over VL-lane split-complex data --------- */

static void dif4_stage(int M, int len, const double *restrict tw,
                       double *restrict wr, double *restrict wi)
{
    const int S = len >> 2;
#ifdef __AVX512F__
    for (int blk = 0; blk < M; blk += len) {
        const double *t = tw;
        double *r0r = wr + (size_t)blk * VL, *r0i = wi + (size_t)blk * VL;
        double *r1r = r0r + (size_t)S * VL, *r1i = r0i + (size_t)S * VL;
        double *r2r = r1r + (size_t)S * VL, *r2i = r1i + (size_t)S * VL;
        double *r3r = r2r + (size_t)S * VL, *r3i = r2i + (size_t)S * VL;
        for (int j = 0; j < S; ++j, t += 6) {
            __m512d ar = _mm512_load_pd(r0r), ai = _mm512_load_pd(r0i);
            __m512d br = _mm512_load_pd(r1r), bi = _mm512_load_pd(r1i);
            __m512d cr = _mm512_load_pd(r2r), ci = _mm512_load_pd(r2i);
            __m512d dr = _mm512_load_pd(r3r), di = _mm512_load_pd(r3i);
            __m512d t0r = _mm512_add_pd(ar, cr), t0i = _mm512_add_pd(ai, ci);
            __m512d t1r = _mm512_sub_pd(ar, cr), t1i = _mm512_sub_pd(ai, ci);
            __m512d t2r = _mm512_add_pd(br, dr), t2i = _mm512_add_pd(bi, di);
            __m512d t3r = _mm512_sub_pd(br, dr), t3i = _mm512_sub_pd(bi, di);
            _mm512_store_pd(r0r, _mm512_add_pd(t0r, t2r));
            _mm512_store_pd(r0i, _mm512_add_pd(t0i, t2i));
            __m512d w1r = _mm512_set1_pd(t[0]), w1i = _mm512_set1_pd(t[1]);
            __m512d w2r = _mm512_set1_pd(t[2]), w2i = _mm512_set1_pd(t[3]);
            __m512d w3r = _mm512_set1_pd(t[4]), w3i = _mm512_set1_pd(t[5]);
            __m512d u2r = _mm512_sub_pd(t0r, t2r), u2i = _mm512_sub_pd(t0i, t2i);
            _mm512_store_pd(r2r, _mm512_fmsub_pd(u2r, w2r, _mm512_mul_pd(u2i, w2i)));
            _mm512_store_pd(r2i, _mm512_fmadd_pd(u2r, w2i, _mm512_mul_pd(u2i, w2r)));
            __m512d u1r = _mm512_add_pd(t1r, t3i), u1i = _mm512_sub_pd(t1i, t3r);
            _mm512_store_pd(r1r, _mm512_fmsub_pd(u1r, w1r, _mm512_mul_pd(u1i, w1i)));
            _mm512_store_pd(r1i, _mm512_fmadd_pd(u1r, w1i, _mm512_mul_pd(u1i, w1r)));
            __m512d u3r = _mm512_sub_pd(t1r, t3i), u3i = _mm512_add_pd(t1i, t3r);
            _mm512_store_pd(r3r, _mm512_fmsub_pd(u3r, w3r, _mm512_mul_pd(u3i, w3i)));
            _mm512_store_pd(r3i, _mm512_fmadd_pd(u3r, w3i, _mm512_mul_pd(u3i, w3r)));
            r0r += VL; r0i += VL; r1r += VL; r1i += VL;
            r2r += VL; r2i += VL; r3r += VL; r3i += VL;
        }
    }
#else
    for (int blk = 0; blk < M; blk += len) {
        const double *t = tw;
        double *r0r = wr + (size_t)blk * VL, *r0i = wi + (size_t)blk * VL;
        double *r1r = r0r + (size_t)S * VL, *r1i = r0i + (size_t)S * VL;
        double *r2r = r1r + (size_t)S * VL, *r2i = r1i + (size_t)S * VL;
        double *r3r = r2r + (size_t)S * VL, *r3i = r2i + (size_t)S * VL;
        for (int j = 0; j < S; ++j, t += 6) {
            const double w1r = t[0], w1i = t[1], w2r = t[2], w2i = t[3],
                         w3r = t[4], w3i = t[5];
            for (int v = 0; v < VL; ++v) {
                double ar = r0r[v], ai = r0i[v], br = r1r[v], bi = r1i[v];
                double cr = r2r[v], ci = r2i[v], dr = r3r[v], di = r3i[v];
                double t0r = ar + cr, t0i = ai + ci, t1r = ar - cr, t1i = ai - ci;
                double t2r = br + dr, t2i = bi + di, t3r = br - dr, t3i = bi - di;
                r0r[v] = t0r + t2r;
                r0i[v] = t0i + t2i;
                double u2r = t0r - t2r, u2i = t0i - t2i;
                r2r[v] = u2r * w2r - u2i * w2i;
                r2i[v] = u2r * w2i + u2i * w2r;
                double u1r = t1r + t3i, u1i = t1i - t3r; /* t1 - i*t3 */
                r1r[v] = u1r * w1r - u1i * w1i;
                r1i[v] = u1r * w1i + u1i * w1r;
                double u3r = t1r - t3i, u3i = t1i + t3r; /* t1 + i*t3 */
                r3r[v] = u3r * w3r - u3i * w3i;
                r3i[v] = u3r * w3i + u3i * w3r;
            }
            r0r += VL; r0i += VL; r1r += VL; r1i += VL;
            r2r += VL; r2i += VL; r3r += VL; r3i += VL;
        }
    }
#endif
}

/* exact stage inverse of dif4_stage (unnormalized; the 1/M lives in bh) */
static void dit4_stage(int M, int len, const double *restrict tw,
                       double *restrict wr, double *restrict wi)
{
    const int S = len >> 2;
#ifdef __AVX512F__
    for (int blk = 0; blk < M; blk += len) {
        const double *t = tw;
        double *r0r = wr + (size_t)blk * VL, *r0i = wi + (size_t)blk * VL;
        double *r1r = r0r + (size_t)S * VL, *r1i = r0i + (size_t)S * VL;
        double *r2r = r1r + (size_t)S * VL, *r2i = r1i + (size_t)S * VL;
        double *r3r = r2r + (size_t)S * VL, *r3i = r2i + (size_t)S * VL;
        for (int j = 0; j < S; ++j, t += 6) {
            __m512d w1r = _mm512_set1_pd(t[0]), w1i = _mm512_set1_pd(t[1]);
            __m512d w2r = _mm512_set1_pd(t[2]), w2i = _mm512_set1_pd(t[3]);
            __m512d w3r = _mm512_set1_pd(t[4]), w3i = _mm512_set1_pd(t[5]);
            __m512d u0r = _mm512_load_pd(r0r), u0i = _mm512_load_pd(r0i);
            __m512d y1r = _mm512_load_pd(r1r), y1i = _mm512_load_pd(r1i);
            __m512d y2r = _mm512_load_pd(r2r), y2i = _mm512_load_pd(r2i);
            __m512d y3r = _mm512_load_pd(r3r), y3i = _mm512_load_pd(r3i);
            __m512d u1r = _mm512_fmsub_pd(y1r, w1r, _mm512_mul_pd(y1i, w1i));
            __m512d u1i = _mm512_fmadd_pd(y1r, w1i, _mm512_mul_pd(y1i, w1r));
            __m512d u2r = _mm512_fmsub_pd(y2r, w2r, _mm512_mul_pd(y2i, w2i));
            __m512d u2i = _mm512_fmadd_pd(y2r, w2i, _mm512_mul_pd(y2i, w2r));
            __m512d u3r = _mm512_fmsub_pd(y3r, w3r, _mm512_mul_pd(y3i, w3i));
            __m512d u3i = _mm512_fmadd_pd(y3r, w3i, _mm512_mul_pd(y3i, w3r));
            __m512d s0r = _mm512_add_pd(u0r, u2r), s0i = _mm512_add_pd(u0i, u2i);
            __m512d s1r = _mm512_sub_pd(u0r, u2r), s1i = _mm512_sub_pd(u0i, u2i);
            __m512d s2r = _mm512_add_pd(u1r, u3r), s2i = _mm512_add_pd(u1i, u3i);
            __m512d s3r = _mm512_sub_pd(u1r, u3r), s3i = _mm512_sub_pd(u1i, u3i);
            _mm512_store_pd(r0r, _mm512_add_pd(s0r, s2r));
            _mm512_store_pd(r0i, _mm512_add_pd(s0i, s2i));
            _mm512_store_pd(r2r, _mm512_sub_pd(s0r, s2r));
            _mm512_store_pd(r2i, _mm512_sub_pd(s0i, s2i));
            _mm512_store_pd(r1r, _mm512_sub_pd(s1r, s3i));
            _mm512_store_pd(r1i, _mm512_add_pd(s1i, s3r));
            _mm512_store_pd(r3r, _mm512_add_pd(s1r, s3i));
            _mm512_store_pd(r3i, _mm512_sub_pd(s1i, s3r));
            r0r += VL; r0i += VL; r1r += VL; r1i += VL;
            r2r += VL; r2i += VL; r3r += VL; r3i += VL;
        }
    }
#else
    for (int blk = 0; blk < M; blk += len) {
        const double *t = tw;
        double *r0r = wr + (size_t)blk * VL, *r0i = wi + (size_t)blk * VL;
        double *r1r = r0r + (size_t)S * VL, *r1i = r0i + (size_t)S * VL;
        double *r2r = r1r + (size_t)S * VL, *r2i = r1i + (size_t)S * VL;
        double *r3r = r2r + (size_t)S * VL, *r3i = r2i + (size_t)S * VL;
        for (int j = 0; j < S; ++j, t += 6) {
            const double w1r = t[0], w1i = t[1], w2r = t[2], w2i = t[3],
                         w3r = t[4], w3i = t[5];
            for (int v = 0; v < VL; ++v) {
                double u0r = r0r[v], u0i = r0i[v];
                double y1r = r1r[v], y1i = r1i[v];
                double y2r = r2r[v], y2i = r2i[v];
                double y3r = r3r[v], y3i = r3i[v];
                double u1r = y1r * w1r - y1i * w1i, u1i = y1r * w1i + y1i * w1r;
                double u2r = y2r * w2r - y2i * w2i, u2i = y2r * w2i + y2i * w2r;
                double u3r = y3r * w3r - y3i * w3i, u3i = y3r * w3i + y3i * w3r;
                double s0r = u0r + u2r, s0i = u0i + u2i;
                double s1r = u0r - u2r, s1i = u0i - u2i;
                double s2r = u1r + u3r, s2i = u1i + u3i;
                double s3r = u1r - u3r, s3i = u1i - u3i;
                r0r[v] = s0r + s2r;
                r0i[v] = s0i + s2i;
                r2r[v] = s0r - s2r;
                r2i[v] = s0i - s2i;
                r1r[v] = s1r - s3i; /* s1 + i*s3 */
                r1i[v] = s1i + s3r;
                r3r[v] = s1r + s3i; /* s1 - i*s3 */
                r3i[v] = s1i - s3r;
            }
            r0r += VL; r0i += VL; r1r += VL; r1i += VL;
            r2r += VL; r2i += VL; r3r += VL; r3i += VL;
        }
    }
#endif
}

/* len == 2 stage, twiddle-free; it is its own (unnormalized) inverse */
static void r2_stage(int M, double *restrict wr, double *restrict wi)
{
#ifdef __AVX512F__
    for (int blk = 0; blk < M; blk += 2) {
        double *ar = wr + (size_t)blk * VL, *ai = wi + (size_t)blk * VL;
        __m512d xr = _mm512_load_pd(ar), xi = _mm512_load_pd(ai);
        __m512d yr = _mm512_load_pd(ar + VL), yi = _mm512_load_pd(ai + VL);
        _mm512_store_pd(ar, _mm512_add_pd(xr, yr));
        _mm512_store_pd(ai, _mm512_add_pd(xi, yi));
        _mm512_store_pd(ar + VL, _mm512_sub_pd(xr, yr));
        _mm512_store_pd(ai + VL, _mm512_sub_pd(xi, yi));
    }
#else
    for (int blk = 0; blk < M; blk += 2) {
        double *ar = wr + (size_t)blk * VL, *ai = wi + (size_t)blk * VL;
        double *br = ar + VL, *bi = ai + VL;
        for (int v = 0; v < VL; ++v) {
            double xr = ar[v], xi = ai[v], yr = br[v], yi = bi[v];
            ar[v] = xr + yr;
            ai[v] = xi + yi;
            br[v] = xr - yr;
            bi[v] = xi - yi;
        }
    }
#endif
}

#ifdef __AVX512F__
/* First forward stage (len == M, one block) exploiting the Bluestein zero-pad:
 * inputs j >= L are structurally zero and are NEVER READ (so the pad region of
 * the work buffer need not be cleared).  Since M >= 2L-1: c = d = 0 always,
 * and b = x[j+M/4] is zero for j >= L - M/4. */
static void dif4_first(int M, int L, const double *restrict tw,
                       double *restrict wr, double *restrict wi)
{
    const int S = M >> 2;
    int jful = L - S; /* < S because M >= 2L-1; >= 1 because M < 4L */
    const double *t = tw;
    double *r0r = wr, *r0i = wi;
    double *r1r = wr + (size_t)S * VL, *r1i = wi + (size_t)S * VL;
    double *r2r = wr + (size_t)2 * S * VL, *r2i = wi + (size_t)2 * S * VL;
    double *r3r = wr + (size_t)3 * S * VL, *r3i = wi + (size_t)3 * S * VL;
    for (int j = 0; j < jful; ++j, t += 6) {
        __m512d ar = _mm512_load_pd(r0r), ai = _mm512_load_pd(r0i);
        __m512d br = _mm512_load_pd(r1r), bi = _mm512_load_pd(r1i);
        __m512d w1r = _mm512_set1_pd(t[0]), w1i = _mm512_set1_pd(t[1]);
        __m512d w2r = _mm512_set1_pd(t[2]), w2i = _mm512_set1_pd(t[3]);
        __m512d w3r = _mm512_set1_pd(t[4]), w3i = _mm512_set1_pd(t[5]);
        _mm512_store_pd(r0r, _mm512_add_pd(ar, br));
        _mm512_store_pd(r0i, _mm512_add_pd(ai, bi));
        __m512d u2r = _mm512_sub_pd(ar, br), u2i = _mm512_sub_pd(ai, bi);
        _mm512_store_pd(r2r, _mm512_fmsub_pd(u2r, w2r, _mm512_mul_pd(u2i, w2i)));
        _mm512_store_pd(r2i, _mm512_fmadd_pd(u2r, w2i, _mm512_mul_pd(u2i, w2r)));
        __m512d u1r = _mm512_add_pd(ar, bi), u1i = _mm512_sub_pd(ai, br); /* a - i*b */
        _mm512_store_pd(r1r, _mm512_fmsub_pd(u1r, w1r, _mm512_mul_pd(u1i, w1i)));
        _mm512_store_pd(r1i, _mm512_fmadd_pd(u1r, w1i, _mm512_mul_pd(u1i, w1r)));
        __m512d u3r = _mm512_sub_pd(ar, bi), u3i = _mm512_add_pd(ai, br); /* a + i*b */
        _mm512_store_pd(r3r, _mm512_fmsub_pd(u3r, w3r, _mm512_mul_pd(u3i, w3i)));
        _mm512_store_pd(r3i, _mm512_fmadd_pd(u3r, w3i, _mm512_mul_pd(u3i, w3r)));
        r0r += VL; r0i += VL; r1r += VL; r1i += VL;
        r2r += VL; r2i += VL; r3r += VL; r3i += VL;
    }
    for (int j = jful; j < S; ++j, t += 6) { /* b = c = d = 0 */
        __m512d ar = _mm512_load_pd(r0r), ai = _mm512_load_pd(r0i);
        __m512d w1r = _mm512_set1_pd(t[0]), w1i = _mm512_set1_pd(t[1]);
        __m512d w2r = _mm512_set1_pd(t[2]), w2i = _mm512_set1_pd(t[3]);
        __m512d w3r = _mm512_set1_pd(t[4]), w3i = _mm512_set1_pd(t[5]);
        _mm512_store_pd(r0r, ar);
        _mm512_store_pd(r0i, ai);
        _mm512_store_pd(r1r, _mm512_fmsub_pd(ar, w1r, _mm512_mul_pd(ai, w1i)));
        _mm512_store_pd(r1i, _mm512_fmadd_pd(ar, w1i, _mm512_mul_pd(ai, w1r)));
        _mm512_store_pd(r2r, _mm512_fmsub_pd(ar, w2r, _mm512_mul_pd(ai, w2i)));
        _mm512_store_pd(r2i, _mm512_fmadd_pd(ar, w2i, _mm512_mul_pd(ai, w2r)));
        _mm512_store_pd(r3r, _mm512_fmsub_pd(ar, w3r, _mm512_mul_pd(ai, w3i)));
        _mm512_store_pd(r3i, _mm512_fmadd_pd(ar, w3i, _mm512_mul_pd(ai, w3r)));
        r0r += VL; r0i += VL; r1r += VL; r1i += VL;
        r2r += VL; r2i += VL; r3r += VL; r3i += VL;
    }
}

/* Last inverse stage (len == M): only outputs k < L are ever consumed by the
 * scatter, and L <= M/2, so the r2/r3 quarters (k >= M/2) are never stored;
 * the r1 output (k = j + M/4) only while j < L - M/4. */
static void dit4_last(int M, int L, const double *restrict tw,
                      double *restrict wr, double *restrict wi)
{
    const int S = M >> 2;
    int jful = L - S;
    const double *t = tw;
    double *r0r = wr, *r0i = wi;
    double *r1r = wr + (size_t)S * VL, *r1i = wi + (size_t)S * VL;
    double *r2r = wr + (size_t)2 * S * VL, *r2i = wi + (size_t)2 * S * VL;
    double *r3r = wr + (size_t)3 * S * VL, *r3i = wi + (size_t)3 * S * VL;
    for (int j = 0; j < S; ++j, t += 6) {
        __m512d w1r = _mm512_set1_pd(t[0]), w1i = _mm512_set1_pd(t[1]);
        __m512d w2r = _mm512_set1_pd(t[2]), w2i = _mm512_set1_pd(t[3]);
        __m512d w3r = _mm512_set1_pd(t[4]), w3i = _mm512_set1_pd(t[5]);
        __m512d u0r = _mm512_load_pd(r0r), u0i = _mm512_load_pd(r0i);
        __m512d y1r = _mm512_load_pd(r1r), y1i = _mm512_load_pd(r1i);
        __m512d y2r = _mm512_load_pd(r2r), y2i = _mm512_load_pd(r2i);
        __m512d y3r = _mm512_load_pd(r3r), y3i = _mm512_load_pd(r3i);
        __m512d u1r = _mm512_fmsub_pd(y1r, w1r, _mm512_mul_pd(y1i, w1i));
        __m512d u1i = _mm512_fmadd_pd(y1r, w1i, _mm512_mul_pd(y1i, w1r));
        __m512d u2r = _mm512_fmsub_pd(y2r, w2r, _mm512_mul_pd(y2i, w2i));
        __m512d u2i = _mm512_fmadd_pd(y2r, w2i, _mm512_mul_pd(y2i, w2r));
        __m512d u3r = _mm512_fmsub_pd(y3r, w3r, _mm512_mul_pd(y3i, w3i));
        __m512d u3i = _mm512_fmadd_pd(y3r, w3i, _mm512_mul_pd(y3i, w3r));
        _mm512_store_pd(r0r, _mm512_add_pd(_mm512_add_pd(u0r, u2r),
                                           _mm512_add_pd(u1r, u3r)));
        _mm512_store_pd(r0i, _mm512_add_pd(_mm512_add_pd(u0i, u2i),
                                           _mm512_add_pd(u1i, u3i)));
        if (j < jful) { /* r1out = s1 + i*s3 */
            __m512d s1r = _mm512_sub_pd(u0r, u2r), s1i = _mm512_sub_pd(u0i, u2i);
            __m512d s3r = _mm512_sub_pd(u1r, u3r), s3i = _mm512_sub_pd(u1i, u3i);
            _mm512_store_pd(r1r, _mm512_sub_pd(s1r, s3i));
            _mm512_store_pd(r1i, _mm512_add_pd(s1i, s3r));
        }
        r0r += VL; r0i += VL; r1r += VL; r1i += VL;
        r2r += VL; r2i += VL; r3r += VL; r3i += VL;
    }
}
#endif /* __AVX512F__ */

#ifdef __AVX512F__
/* Fused convolution middle: last forward stage (twiddle-free) + pointwise
 * multiply by bh + first inverse stage (twiddle-free), one pass, in registers.
 * conv_mid4 for even log2(M) (len-4 tail), conv_mid2 for odd (radix-2 tail). */
static void conv_mid4(int M, const double *restrict bhre, const double *restrict bhim,
                      double *restrict wr, double *restrict wi)
{
    for (int blk = 0; blk < M; blk += 4) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        __m512d ar = _mm512_load_pd(qr), ai = _mm512_load_pd(qi);
        __m512d br = _mm512_load_pd(qr + VL), bi = _mm512_load_pd(qi + VL);
        __m512d cr = _mm512_load_pd(qr + 2 * VL), ci = _mm512_load_pd(qi + 2 * VL);
        __m512d dr = _mm512_load_pd(qr + 3 * VL), di = _mm512_load_pd(qi + 3 * VL);
        __m512d t0r = _mm512_add_pd(ar, cr), t0i = _mm512_add_pd(ai, ci);
        __m512d t1r = _mm512_sub_pd(ar, cr), t1i = _mm512_sub_pd(ai, ci);
        __m512d t2r = _mm512_add_pd(br, dr), t2i = _mm512_add_pd(bi, di);
        __m512d t3r = _mm512_sub_pd(br, dr), t3i = _mm512_sub_pd(bi, di);
        __m512d y0r = _mm512_add_pd(t0r, t2r), y0i = _mm512_add_pd(t0i, t2i);
        __m512d y2r = _mm512_sub_pd(t0r, t2r), y2i = _mm512_sub_pd(t0i, t2i);
        __m512d y1r = _mm512_add_pd(t1r, t3i), y1i = _mm512_sub_pd(t1i, t3r);
        __m512d y3r = _mm512_sub_pd(t1r, t3i), y3i = _mm512_add_pd(t1i, t3r);
        __m512d h0r = _mm512_set1_pd(bhre[blk]), h0i = _mm512_set1_pd(bhim[blk]);
        __m512d h1r = _mm512_set1_pd(bhre[blk + 1]), h1i = _mm512_set1_pd(bhim[blk + 1]);
        __m512d h2r = _mm512_set1_pd(bhre[blk + 2]), h2i = _mm512_set1_pd(bhim[blk + 2]);
        __m512d h3r = _mm512_set1_pd(bhre[blk + 3]), h3i = _mm512_set1_pd(bhim[blk + 3]);
        __m512d u0r = _mm512_fmsub_pd(y0r, h0r, _mm512_mul_pd(y0i, h0i));
        __m512d u0i = _mm512_fmadd_pd(y0r, h0i, _mm512_mul_pd(y0i, h0r));
        __m512d u1r = _mm512_fmsub_pd(y1r, h1r, _mm512_mul_pd(y1i, h1i));
        __m512d u1i = _mm512_fmadd_pd(y1r, h1i, _mm512_mul_pd(y1i, h1r));
        __m512d u2r = _mm512_fmsub_pd(y2r, h2r, _mm512_mul_pd(y2i, h2i));
        __m512d u2i = _mm512_fmadd_pd(y2r, h2i, _mm512_mul_pd(y2i, h2r));
        __m512d u3r = _mm512_fmsub_pd(y3r, h3r, _mm512_mul_pd(y3i, h3i));
        __m512d u3i = _mm512_fmadd_pd(y3r, h3i, _mm512_mul_pd(y3i, h3r));
        __m512d s0r = _mm512_add_pd(u0r, u2r), s0i = _mm512_add_pd(u0i, u2i);
        __m512d s1r = _mm512_sub_pd(u0r, u2r), s1i = _mm512_sub_pd(u0i, u2i);
        __m512d s2r = _mm512_add_pd(u1r, u3r), s2i = _mm512_add_pd(u1i, u3i);
        __m512d s3r = _mm512_sub_pd(u1r, u3r), s3i = _mm512_sub_pd(u1i, u3i);
        _mm512_store_pd(qr, _mm512_add_pd(s0r, s2r));
        _mm512_store_pd(qi, _mm512_add_pd(s0i, s2i));
        _mm512_store_pd(qr + 2 * VL, _mm512_sub_pd(s0r, s2r));
        _mm512_store_pd(qi + 2 * VL, _mm512_sub_pd(s0i, s2i));
        _mm512_store_pd(qr + VL, _mm512_sub_pd(s1r, s3i)); /* s1 + i*s3 */
        _mm512_store_pd(qi + VL, _mm512_add_pd(s1i, s3r));
        _mm512_store_pd(qr + 3 * VL, _mm512_add_pd(s1r, s3i)); /* s1 - i*s3 */
        _mm512_store_pd(qi + 3 * VL, _mm512_sub_pd(s1i, s3r));
    }
}

static void conv_mid2(int M, const double *restrict bhre, const double *restrict bhim,
                      double *restrict wr, double *restrict wi)
{
    for (int blk = 0; blk < M; blk += 2) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        __m512d ar = _mm512_load_pd(qr), ai = _mm512_load_pd(qi);
        __m512d br = _mm512_load_pd(qr + VL), bi = _mm512_load_pd(qi + VL);
        __m512d y0r = _mm512_add_pd(ar, br), y0i = _mm512_add_pd(ai, bi);
        __m512d y1r = _mm512_sub_pd(ar, br), y1i = _mm512_sub_pd(ai, bi);
        __m512d h0r = _mm512_set1_pd(bhre[blk]), h0i = _mm512_set1_pd(bhim[blk]);
        __m512d h1r = _mm512_set1_pd(bhre[blk + 1]), h1i = _mm512_set1_pd(bhim[blk + 1]);
        __m512d u0r = _mm512_fmsub_pd(y0r, h0r, _mm512_mul_pd(y0i, h0i));
        __m512d u0i = _mm512_fmadd_pd(y0r, h0i, _mm512_mul_pd(y0i, h0r));
        __m512d u1r = _mm512_fmsub_pd(y1r, h1r, _mm512_mul_pd(y1i, h1i));
        __m512d u1i = _mm512_fmadd_pd(y1r, h1i, _mm512_mul_pd(y1i, h1r));
        _mm512_store_pd(qr, _mm512_add_pd(u0r, u1r));
        _mm512_store_pd(qi, _mm512_add_pd(u0i, u1i));
        _mm512_store_pd(qr + VL, _mm512_sub_pd(u0r, u1r));
        _mm512_store_pd(qi + VL, _mm512_sub_pd(u0i, u1i));
    }
}
#endif /* __AVX512F__ */

/* Full (unpruned) forward, used at create() on b_pad, which is NOT zero-padded. */
static void fwd_fft_full(const fft3d_plan *p, double *wr, double *wi)
{
    const double *t = p->twf;
    int len = p->M;
    while (len >= 4) {
        dif4_stage(p->M, len, t, wr, wi);
        t += (size_t)(len >> 2) * 6;
        len >>= 2;
    }
    if (len == 2) r2_stage(p->M, wr, wi);
}

/* The whole per-row-group circular convolution: forward (input-pruned),
 * fused middle, inverse (output-pruned).  Scalar fallback runs the generic
 * pipeline and needs the caller to have zeroed the pad. */
static void conv_rows(const fft3d_plan *p, double *restrict wr, double *restrict wi)
{
    const int M = p->M;
#ifdef __AVX512F__
    const int L = p->L;
    const double *t = p->twf;
    dif4_first(M, L, t, wr, wi);
    t += (size_t)(M >> 2) * 6;
    int len = M >> 2;
    while (len >= 8) {
        dif4_stage(M, len, t, wr, wi);
        t += (size_t)(len >> 2) * 6;
        len >>= 2;
    }
    if (len == 4)
        conv_mid4(M, p->bhre, p->bhim, wr, wi);
    else if (len == 2)
        conv_mid2(M, p->bhre, p->bhim, wr, wi);
    else { /* M == 4: dif4_first was the whole forward */
        for (int m = 0; m < M; ++m) {
            __m512d hr = _mm512_set1_pd(p->bhre[m]), hi = _mm512_set1_pd(p->bhim[m]);
            double *qr = wr + (size_t)m * VL, *qi = wi + (size_t)m * VL;
            __m512d xr = _mm512_load_pd(qr), xi = _mm512_load_pd(qi);
            _mm512_store_pd(qr, _mm512_fmsub_pd(xr, hr, _mm512_mul_pd(xi, hi)));
            _mm512_store_pd(qi, _mm512_fmadd_pd(xr, hi, _mm512_mul_pd(xi, hr)));
        }
    }
    const double *ti = p->twi;
    if (M > 4) {
        int start = (len == 4) ? 16 : 8;
        if (len == 4) ti += 6; /* skip the fused len-4 stage's twiddle chunk */
        for (int l2 = start; l2 < M; l2 <<= 2) {
            dit4_stage(M, l2, ti, wr, wi);
            ti += (size_t)(l2 >> 2) * 6;
        }
    }
    dit4_last(M, L, ti, wr, wi);
#else
    fwd_fft_full(p, wr, wi);
    for (int m = 0; m < M; ++m) {
        const double br = p->bhre[m], bi = p->bhim[m];
        double *restrict qr = wr + (size_t)m * VL;
        double *restrict qi = wi + (size_t)m * VL;
        for (int v = 0; v < VL; ++v) {
            double xr = qr[v], xi = qi[v];
            qr[v] = xr * br - xi * bi;
            qi[v] = xr * bi + xi * br;
        }
    }
    int len = M, odd;
    while (len >= 4) len >>= 2;
    odd = (len == 2);
    if (odd) r2_stage(M, wr, wi);
    const double *t = p->twi;
    for (len = odd ? 8 : 4; len <= M; len <<= 2) {
        dit4_stage(M, len, t, wr, wi);
        t += (size_t)(len >> 2) * 6;
    }
#endif
}

/* ---------------- plan construction --------------------------------------- */

static double *xalloc(size_t n) { return aligned_alloc(64, (n * sizeof(double) + 63) & ~(size_t)63); }

fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L) || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    int M = 4;
    while (M < 2 * L - 1) M <<= 1;
    p->M = M;

    /* twiddle table sizes: sum over radix-4 stages of (len/4)*6 doubles */
    size_t twn = 0;
    for (int len = M; len >= 4; len >>= 2) twn += (size_t)(len >> 2) * 6;

    p->chre = xalloc(L);
    p->chim = xalloc(L);
    p->bhre = xalloc(M);
    p->bhim = xalloc(M);
    p->twf = xalloc(twn ? twn : 1);
    p->twi = xalloc(twn ? twn : 1);
    p->wr = xalloc((size_t)M * VL);
    p->wi = xalloc((size_t)M * VL);
    if (!p->chre || !p->chim || !p->bhre || !p->bhim || !p->twf || !p->twi ||
        !p->wr || !p->wi) {
        fft3d_destroy(p);
        return NULL;
    }

    /* chirp: exact integer phase reduction, long-double evaluation */
    for (int j = 0; j < L; ++j) {
        long r = ((long)j * j) % (2L * L);
        long double th = -(long double)PIL * (long double)r / (long double)L;
        p->chre[j] = (double)cosl(th);
        p->chim[j] = (double)sinl(th);
    }

    /* forward twiddles (consumption order), inverse = conjugates, mirrored order */
    {
        double *tf = p->twf;
        for (int len = M; len >= 4; len >>= 2) {
            int S = len >> 2;
            for (int j = 0; j < S; ++j) {
                for (int r = 1; r <= 3; ++r) {
                    long double th = -2.0L * PIL * (long double)(r * j) / (long double)len;
                    *tf++ = (double)cosl(th);
                    *tf++ = (double)sinl(th);
                }
            }
        }
        double *ti = p->twi;
        int odd_len = M;
        while (odd_len >= 4) odd_len >>= 2;
        for (int len = (odd_len == 2) ? 8 : 4; len <= M; len <<= 2) {
            int S = len >> 2;
            for (int j = 0; j < S; ++j) {
                for (int r = 1; r <= 3; ++r) {
                    long double th = 2.0L * PIL * (long double)(r * j) / (long double)len;
                    *ti++ = (double)cosl(th);
                    *ti++ = (double)sinl(th);
                }
            }
        }
    }

    /* b_pad = chirp conjugate embedded circularly; bh = fwd_fft(b_pad)/M in the
     * pipeline's own scrambled order (run our own forward, lane-replicated). */
    memset(p->wr, 0, (size_t)M * VL * sizeof(double));
    memset(p->wi, 0, (size_t)M * VL * sizeof(double));
    for (int n = 0; n < L; ++n) {
        double br = p->chre[n], bi = -p->chim[n]; /* exp(+i pi n^2/L) */
        for (int v = 0; v < VL; ++v) {
            p->wr[(size_t)n * VL + v] = br;
            p->wi[(size_t)n * VL + v] = bi;
            if (n) {
                p->wr[(size_t)(M - n) * VL + v] = br;
                p->wi[(size_t)(M - n) * VL + v] = bi;
            }
        }
    }
    fwd_fft_full(p, p->wr, p->wi);
    for (int m = 0; m < M; ++m) {
        p->bhre[m] = p->wr[(size_t)m * VL] / (double)M;
        p->bhim[m] = p->wi[(size_t)m * VL] / (double)M;
    }
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->chre); free(p->chim);
    free(p->bhre); free(p->bhim);
    free(p->twf); free(p->twi);
    free(p->wr); free(p->wi);
    free(p);
}

/* ---------------- execution ------------------------------------------------ */

#ifdef __AVX512F__
/* transpose an 8x8 block of doubles held in 8 zmm registers (self-inverse) */
static inline void tr8x8(__m512d r[8])
{
    __m512d t0 = _mm512_unpacklo_pd(r[0], r[1]), t1 = _mm512_unpackhi_pd(r[0], r[1]);
    __m512d t2 = _mm512_unpacklo_pd(r[2], r[3]), t3 = _mm512_unpackhi_pd(r[2], r[3]);
    __m512d t4 = _mm512_unpacklo_pd(r[4], r[5]), t5 = _mm512_unpackhi_pd(r[4], r[5]);
    __m512d t6 = _mm512_unpacklo_pd(r[6], r[7]), t7 = _mm512_unpackhi_pd(r[6], r[7]);
    __m512d s0 = _mm512_shuffle_f64x2(t0, t2, 0x88), s1 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
    __m512d s2 = _mm512_shuffle_f64x2(t1, t3, 0x88), s3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
    __m512d s4 = _mm512_shuffle_f64x2(t4, t6, 0x88), s5 = _mm512_shuffle_f64x2(t4, t6, 0xDD);
    __m512d s6 = _mm512_shuffle_f64x2(t5, t7, 0x88), s7 = _mm512_shuffle_f64x2(t5, t7, 0xDD);
    r[0] = _mm512_shuffle_f64x2(s0, s4, 0x88);
    r[4] = _mm512_shuffle_f64x2(s0, s4, 0xDD);
    r[2] = _mm512_shuffle_f64x2(s1, s5, 0x88);
    r[6] = _mm512_shuffle_f64x2(s1, s5, 0xDD);
    r[1] = _mm512_shuffle_f64x2(s2, s6, 0x88);
    r[5] = _mm512_shuffle_f64x2(s2, s6, 0xDD);
    r[3] = _mm512_shuffle_f64x2(s3, s7, 0x88);
    r[7] = _mm512_shuffle_f64x2(s3, s7, 0xDD);
}
#endif

/* One Bluestein pass over all rows of one axis.  Row r (0..nrows-1) starts at
 * complex offset (r/div)*A + (r%div)*B from base and strides by `stride`
 * complex elements.  src rows are read, dst rows written at the SAME offsets
 * (safe in place: each row depends only on itself). */
static void axis_pass(fft3d_plan *p, const double *src, double *dst,
                      long rlo, long rhi, long div, long A, long B, long stride)
{
    const int L = p->L;
    double *restrict wr = p->wr, *restrict wi = p->wi;
    const double *restrict chre = p->chre, *restrict chim = p->chim;
    const long s2 = 2 * stride;

    for (long r0 = rlo; r0 < rhi; r0 += VL) {
        int nv = (rhi - r0 < VL) ? (int)(rhi - r0) : VL;
        const double *sbase[VL];
        double *dbase[VL];
        for (int v = 0; v < VL; ++v) {
            long r = r0 + (v < nv ? v : 0);
            long off = 2 * ((r / div) * A + (r % div) * B);
            sbase[v] = src + off;
            dbase[v] = dst + off;
        }

#ifdef __AVX512F__
        /* lanes contiguous in memory?  (axes 0/1 except at outer-run seams) */
        const int contig = (B == 1) && (nv == VL) && (r0 % div) + VL <= div;
#endif

        /* gather * chirp, zero-pad implicit (pruned first stage never reads it) */
#ifdef __AVX512F__
        if (contig) {
            const double *q = sbase[0];
            const __m512i IDXE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
            const __m512i IDXO = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
            for (int j = 0; j < L; ++j) {
                __m512d z0 = _mm512_loadu_pd(q + (size_t)j * s2);
                __m512d z1 = _mm512_loadu_pd(q + (size_t)j * s2 + 8);
                __m512d xr = _mm512_permutex2var_pd(z0, IDXE, z1);
                __m512d xi = _mm512_permutex2var_pd(z0, IDXO, z1);
                __m512d cr = _mm512_set1_pd(chre[j]), ci = _mm512_set1_pd(chim[j]);
                _mm512_store_pd(wr + (size_t)j * VL,
                                _mm512_fmsub_pd(xr, cr, _mm512_mul_pd(xi, ci)));
                _mm512_store_pd(wi + (size_t)j * VL,
                                _mm512_fmadd_pd(xr, ci, _mm512_mul_pd(xi, cr)));
            }
        } else if (stride == 1) {
            /* axis 2: rows contiguous, lanes strided -> 8x8 transpose blocks */
            const int nd = 2 * L, full = nd & ~7;
            for (int k = 0; k < full; k += 8) {
                __m512d r[8];
                for (int v = 0; v < VL; ++v) r[v] = _mm512_loadu_pd(sbase[v] + k);
                tr8x8(r);
                for (int c = 0; c < 4; ++c) {
                    const int j = (k >> 1) + c;
                    __m512d xr = r[2 * c], xi = r[2 * c + 1];
                    __m512d cr = _mm512_set1_pd(chre[j]), ci = _mm512_set1_pd(chim[j]);
                    _mm512_store_pd(wr + (size_t)j * VL,
                                    _mm512_fmsub_pd(xr, cr, _mm512_mul_pd(xi, ci)));
                    _mm512_store_pd(wi + (size_t)j * VL,
                                    _mm512_fmadd_pd(xr, ci, _mm512_mul_pd(xi, cr)));
                }
            }
            for (int j = full >> 1; j < L; ++j) {
                const double cr = chre[j], ci = chim[j];
                const long o = (long)j * s2;
                double *restrict qr = wr + (size_t)j * VL;
                double *restrict qi = wi + (size_t)j * VL;
                for (int v = 0; v < VL; ++v) {
                    double xr = sbase[v][o], xi = sbase[v][o + 1];
                    qr[v] = xr * cr - xi * ci;
                    qi[v] = xr * ci + xi * cr;
                }
            }
        } else
#endif
        {
            for (int j = 0; j < L; ++j) {
                const double cr = chre[j], ci = chim[j];
                const long o = (long)j * s2;
                double *restrict qr = wr + (size_t)j * VL;
                double *restrict qi = wi + (size_t)j * VL;
                for (int v = 0; v < VL; ++v) {
                    double xr = sbase[v][o], xi = sbase[v][o + 1];
                    qr[v] = xr * cr - xi * ci;
                    qi[v] = xr * ci + xi * cr;
                }
            }
        }
#ifndef __AVX512F__
        memset(wr + (size_t)L * VL, 0, (size_t)(p->M - L) * VL * sizeof(double));
        memset(wi + (size_t)L * VL, 0, (size_t)(p->M - L) * VL * sizeof(double));
#endif
        conv_rows(p, wr, wi);

        /* scatter * chirp */
#ifdef __AVX512F__
        if (contig) {
            double *q = dbase[0];
            const __m512i IDXL = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
            const __m512i IDXH = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
            for (int k = 0; k < L; ++k) {
                __m512d xr = _mm512_load_pd(wr + (size_t)k * VL);
                __m512d xi = _mm512_load_pd(wi + (size_t)k * VL);
                __m512d cr = _mm512_set1_pd(chre[k]), ci = _mm512_set1_pd(chim[k]);
                __m512d yr = _mm512_fmsub_pd(xr, cr, _mm512_mul_pd(xi, ci));
                __m512d yi = _mm512_fmadd_pd(xr, ci, _mm512_mul_pd(xi, cr));
                _mm512_storeu_pd(q + (size_t)k * s2, _mm512_permutex2var_pd(yr, IDXL, yi));
                _mm512_storeu_pd(q + (size_t)k * s2 + 8, _mm512_permutex2var_pd(yr, IDXH, yi));
            }
        } else if (stride == 1) {
            const int nd = 2 * L, full = nd & ~7;
            for (int k = 0; k < full; k += 8) {
                __m512d r[8];
                for (int c = 0; c < 4; ++c) {
                    const int j = (k >> 1) + c;
                    __m512d xr = _mm512_load_pd(wr + (size_t)j * VL);
                    __m512d xi = _mm512_load_pd(wi + (size_t)j * VL);
                    __m512d cr = _mm512_set1_pd(chre[j]), ci = _mm512_set1_pd(chim[j]);
                    r[2 * c] = _mm512_fmsub_pd(xr, cr, _mm512_mul_pd(xi, ci));
                    r[2 * c + 1] = _mm512_fmadd_pd(xr, ci, _mm512_mul_pd(xi, cr));
                }
                tr8x8(r);
                for (int v = 0; v < nv; ++v) _mm512_storeu_pd(dbase[v] + k, r[v]);
            }
            for (int k = full >> 1; k < L; ++k) {
                const double cr = chre[k], ci = chim[k];
                const long o = (long)k * s2;
                const double *restrict qr = wr + (size_t)k * VL;
                const double *restrict qi = wi + (size_t)k * VL;
                for (int v = 0; v < nv; ++v) {
                    double xr = qr[v], xi = qi[v];
                    dbase[v][o] = xr * cr - xi * ci;
                    dbase[v][o + 1] = xr * ci + xi * cr;
                }
            }
        } else
#endif
        for (int k = 0; k < L; ++k) {
            const double cr = chre[k], ci = chim[k];
            const long o = (long)k * s2;
            const double *restrict qr = wr + (size_t)k * VL;
            const double *restrict qi = wi + (size_t)k * VL;
            for (int v = 0; v < nv; ++v) {
                double xr = qr[v], xi = qi[v];
                dbase[v][o] = xr * cr - xi * ci;
                dbase[v][o + 1] = xr * ci + xi * cr;
            }
        }
    }
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    const long L = p->L, B = p->batch;
    const double *src = (const double *)in;
    double *dst = (double *)out;
    const long nrows = B * L * L;

    /* axis 2 (z, stride 1): row r starts at r*L      -> in  to out */
    axis_pass(p, src, dst, 0, nrows, 1, L, 0, 1);
    /* axis 1 (y, stride L): row r -> (r/L)*L^2 + r%L -> out in place */
    axis_pass(p, dst, dst, 0, nrows, L, L * L, 1, L);
    /* axis 0 (x, stride L^2): row r -> (r/L^2)*L^3 + r%L^2 */
    axis_pass(p, dst, dst, 0, nrows, L * L, L * L * L, 1, L * L);
}
