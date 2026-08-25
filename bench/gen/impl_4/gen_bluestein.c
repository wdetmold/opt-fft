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

/* gen_twiddle library layer: exact octant-folded tables + ulp audit
 * (adopted gen_r2; tw_chirp / tw_fill_ct_int_colmajor were built for this
 * entry's exact consumption order -- see strategies/gen_twiddle.md). */
#define GEN_TWIDDLE_LIB_ONLY
#include "gen_twiddle.c"

#define VL 8 /* row lanes per group: 8 doubles = one zmm */

struct fft3d_plan {
    int L, batch, M;
    double *chre, *chim;    /* chirp exp(-i pi j^2/L), length L */
    double *bhre, *bhim;    /* forward-DIF(b_pad)/M, scrambled order, length M */
    double *twf;            /* forward twiddles, consumption order */
    double *twi;            /* inverse twiddles (conjugates), consumption order */
    const double *twi_last; /* the len == M chunk of twi (consumed by dit4_last) */
    double *wr, *wi;        /* work: M * VL each */
};

const char *fft3d_name(void) { return "gen_bluestein"; }
const char *fft3d_description(void)
{
    return "Bluestein chirp-Z for ANY L: pow2 radix-4/16 DIF/DIT convolution (no bit-reversal), 8-row SoA lanes, gather/scatter fused into the pruned end stages (masked dual-run loads keep seam groups vectorized), owned in-place map chain -- map fused into the axis-0 scatter while state+c fit LLC, else axis-0-first k-plane-blocked custody (axes 2+1 + sequential map sweep per L2-hot block), gen_twiddle exact tables";
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

#ifdef __AVX512F__
/* Fused pair of forward stages (radix-16 = two radix-4 layers in registers):
 * stage `len` (twiddles tw1) immediately followed by stage len/4 (twiddles
 * tw2), one buffer pass instead of two.  Tile: p(q,t) = blk + q*S1 + t*S2 + j2
 * is closed under both layers (layer 1 = legs across q, layer 2 = legs across
 * t).  16 live complex vectors: some spill, but the spills are L1 round trips
 * we were paying as full stage stores anyway. */
#define BST_CMUL(rr, ri, xr, xi, wre, wim)                                        \
    do {                                                                          \
        rr = _mm512_fmsub_pd(xr, wre, _mm512_mul_pd(xi, wim));                    \
        ri = _mm512_fmadd_pd(xr, wim, _mm512_mul_pd(xi, wre));                    \
    } while (0)

#ifdef BST_SCHED
/* gen_batchlane's SCHED15 trick: pre-RA scheduling w/ pressure awareness on
 * the register-heavy fused stages only */
#define BST_SCHED_ATTR __attribute__((optimize("schedule-insns", "sched-pressure")))
#else
#define BST_SCHED_ATTR
#endif

BST_SCHED_ATTR
static void dif16_stage(int M, int len, const double *restrict tw1,
                        const double *restrict tw2,
                        double *restrict wr, double *restrict wi)
{
    const int S1 = len >> 2, S2 = len >> 4;
    for (int blk = 0; blk < M; blk += len) {
        for (int j2 = 0; j2 < S2; ++j2) {
            double *br_ = wr + (size_t)(blk + j2) * VL;
            double *bi_ = wi + (size_t)(blk + j2) * VL;
            __m512d yr[16], yi[16];
            for (int t = 0; t < 4; ++t) { /* layer 1: legs across q */
                const double *w = tw1 + (size_t)(t * S2 + j2) * 6;
                const size_t o = (size_t)(t * S2) * VL, qs = (size_t)S1 * VL;
                __m512d ar = _mm512_load_pd(br_ + o), ai = _mm512_load_pd(bi_ + o);
                __m512d brr = _mm512_load_pd(br_ + o + qs), bri = _mm512_load_pd(bi_ + o + qs);
                __m512d cr = _mm512_load_pd(br_ + o + 2 * qs), ci = _mm512_load_pd(bi_ + o + 2 * qs);
                __m512d dr = _mm512_load_pd(br_ + o + 3 * qs), di = _mm512_load_pd(bi_ + o + 3 * qs);
                __m512d t0r = _mm512_add_pd(ar, cr), t0i = _mm512_add_pd(ai, ci);
                __m512d t1r = _mm512_sub_pd(ar, cr), t1i = _mm512_sub_pd(ai, ci);
                __m512d t2r = _mm512_add_pd(brr, dr), t2i = _mm512_add_pd(bri, di);
                __m512d t3r = _mm512_sub_pd(brr, dr), t3i = _mm512_sub_pd(bri, di);
                yr[t] = _mm512_add_pd(t0r, t2r);
                yi[t] = _mm512_add_pd(t0i, t2i);
                __m512d w1r = _mm512_set1_pd(w[0]), w1i = _mm512_set1_pd(w[1]);
                __m512d w2r = _mm512_set1_pd(w[2]), w2i = _mm512_set1_pd(w[3]);
                __m512d w3r = _mm512_set1_pd(w[4]), w3i = _mm512_set1_pd(w[5]);
                __m512d u2r = _mm512_sub_pd(t0r, t2r), u2i = _mm512_sub_pd(t0i, t2i);
                BST_CMUL(yr[8 + t], yi[8 + t], u2r, u2i, w2r, w2i);
                __m512d u1r = _mm512_add_pd(t1r, t3i), u1i = _mm512_sub_pd(t1i, t3r);
                BST_CMUL(yr[4 + t], yi[4 + t], u1r, u1i, w1r, w1i);
                __m512d u3r = _mm512_sub_pd(t1r, t3i), u3i = _mm512_add_pd(t1i, t3r);
                BST_CMUL(yr[12 + t], yi[12 + t], u3r, u3i, w3r, w3i);
            }
            const double *w = tw2 + (size_t)j2 * 6;
            __m512d w1r = _mm512_set1_pd(w[0]), w1i = _mm512_set1_pd(w[1]);
            __m512d w2r = _mm512_set1_pd(w[2]), w2i = _mm512_set1_pd(w[3]);
            __m512d w3r = _mm512_set1_pd(w[4]), w3i = _mm512_set1_pd(w[5]);
            for (int q = 0; q < 4; ++q) { /* layer 2: legs across t */
                const size_t o = (size_t)(q * S1) * VL, ts = (size_t)S2 * VL;
                __m512d ar = yr[4 * q], ai = yi[4 * q];
                __m512d brr = yr[4 * q + 1], bri = yi[4 * q + 1];
                __m512d cr = yr[4 * q + 2], ci = yi[4 * q + 2];
                __m512d dr = yr[4 * q + 3], di = yi[4 * q + 3];
                __m512d t0r = _mm512_add_pd(ar, cr), t0i = _mm512_add_pd(ai, ci);
                __m512d t1r = _mm512_sub_pd(ar, cr), t1i = _mm512_sub_pd(ai, ci);
                __m512d t2r = _mm512_add_pd(brr, dr), t2i = _mm512_add_pd(bri, di);
                __m512d t3r = _mm512_sub_pd(brr, dr), t3i = _mm512_sub_pd(bri, di);
                _mm512_store_pd(br_ + o, _mm512_add_pd(t0r, t2r));
                _mm512_store_pd(bi_ + o, _mm512_add_pd(t0i, t2i));
                __m512d u2r = _mm512_sub_pd(t0r, t2r), u2i = _mm512_sub_pd(t0i, t2i);
                __m512d rr, ri;
                BST_CMUL(rr, ri, u2r, u2i, w2r, w2i);
                _mm512_store_pd(br_ + o + 2 * ts, rr);
                _mm512_store_pd(bi_ + o + 2 * ts, ri);
                __m512d u1r = _mm512_add_pd(t1r, t3i), u1i = _mm512_sub_pd(t1i, t3r);
                BST_CMUL(rr, ri, u1r, u1i, w1r, w1i);
                _mm512_store_pd(br_ + o + ts, rr);
                _mm512_store_pd(bi_ + o + ts, ri);
                __m512d u3r = _mm512_sub_pd(t1r, t3i), u3i = _mm512_add_pd(t1i, t3r);
                BST_CMUL(rr, ri, u3r, u3i, w3r, w3i);
                _mm512_store_pd(br_ + o + 3 * ts, rr);
                _mm512_store_pd(bi_ + o + 3 * ts, ri);
            }
        }
    }
}

/* Fused pair of inverse stages: stage len/4 (twiddles ti1, applied on load)
 * then stage `len` (twiddles ti2).  Exact inverse of dif16_stage. */
BST_SCHED_ATTR
static void dit16_stage(int M, int len, const double *restrict ti1,
                        const double *restrict ti2,
                        double *restrict wr, double *restrict wi)
{
    const int S1 = len >> 2, S2 = len >> 4;
    for (int blk = 0; blk < M; blk += len) {
        for (int j2 = 0; j2 < S2; ++j2) {
            double *br_ = wr + (size_t)(blk + j2) * VL;
            double *bi_ = wi + (size_t)(blk + j2) * VL;
            __m512d yr[16], yi[16];
            const double *wa = ti1 + (size_t)j2 * 6;
            __m512d v1r = _mm512_set1_pd(wa[0]), v1i = _mm512_set1_pd(wa[1]);
            __m512d v2r = _mm512_set1_pd(wa[2]), v2i = _mm512_set1_pd(wa[3]);
            __m512d v3r = _mm512_set1_pd(wa[4]), v3i = _mm512_set1_pd(wa[5]);
            for (int q = 0; q < 4; ++q) { /* layer A: stage len/4, legs across t */
                const size_t o = (size_t)(q * S1) * VL, ts = (size_t)S2 * VL;
                __m512d u0r = _mm512_load_pd(br_ + o), u0i = _mm512_load_pd(bi_ + o);
                __m512d p1r = _mm512_load_pd(br_ + o + ts), p1i = _mm512_load_pd(bi_ + o + ts);
                __m512d p2r = _mm512_load_pd(br_ + o + 2 * ts), p2i = _mm512_load_pd(bi_ + o + 2 * ts);
                __m512d p3r = _mm512_load_pd(br_ + o + 3 * ts), p3i = _mm512_load_pd(bi_ + o + 3 * ts);
                __m512d u1r, u1i, u2r, u2i, u3r, u3i;
                BST_CMUL(u1r, u1i, p1r, p1i, v1r, v1i);
                BST_CMUL(u2r, u2i, p2r, p2i, v2r, v2i);
                BST_CMUL(u3r, u3i, p3r, p3i, v3r, v3i);
                __m512d s0r = _mm512_add_pd(u0r, u2r), s0i = _mm512_add_pd(u0i, u2i);
                __m512d s1r = _mm512_sub_pd(u0r, u2r), s1i = _mm512_sub_pd(u0i, u2i);
                __m512d s2r = _mm512_add_pd(u1r, u3r), s2i = _mm512_add_pd(u1i, u3i);
                __m512d s3r = _mm512_sub_pd(u1r, u3r), s3i = _mm512_sub_pd(u1i, u3i);
                yr[4 * q] = _mm512_add_pd(s0r, s2r);
                yi[4 * q] = _mm512_add_pd(s0i, s2i);
                yr[4 * q + 2] = _mm512_sub_pd(s0r, s2r);
                yi[4 * q + 2] = _mm512_sub_pd(s0i, s2i);
                yr[4 * q + 1] = _mm512_sub_pd(s1r, s3i);
                yi[4 * q + 1] = _mm512_add_pd(s1i, s3r);
                yr[4 * q + 3] = _mm512_add_pd(s1r, s3i);
                yi[4 * q + 3] = _mm512_sub_pd(s1i, s3r);
            }
            for (int t = 0; t < 4; ++t) { /* layer B: stage len, legs across q */
                const double *w = ti2 + (size_t)(t * S2 + j2) * 6;
                const size_t o = (size_t)(t * S2) * VL, qs = (size_t)S1 * VL;
                __m512d w1r = _mm512_set1_pd(w[0]), w1i = _mm512_set1_pd(w[1]);
                __m512d w2r = _mm512_set1_pd(w[2]), w2i = _mm512_set1_pd(w[3]);
                __m512d w3r = _mm512_set1_pd(w[4]), w3i = _mm512_set1_pd(w[5]);
                __m512d u0r = yr[t], u0i = yi[t];
                __m512d u1r, u1i, u2r, u2i, u3r, u3i;
                BST_CMUL(u1r, u1i, yr[4 + t], yi[4 + t], w1r, w1i);
                BST_CMUL(u2r, u2i, yr[8 + t], yi[8 + t], w2r, w2i);
                BST_CMUL(u3r, u3i, yr[12 + t], yi[12 + t], w3r, w3i);
                __m512d s0r = _mm512_add_pd(u0r, u2r), s0i = _mm512_add_pd(u0i, u2i);
                __m512d s1r = _mm512_sub_pd(u0r, u2r), s1i = _mm512_sub_pd(u0i, u2i);
                __m512d s2r = _mm512_add_pd(u1r, u3r), s2i = _mm512_add_pd(u1i, u3i);
                __m512d s3r = _mm512_sub_pd(u1r, u3r), s3i = _mm512_sub_pd(u1i, u3i);
                _mm512_store_pd(br_ + o, _mm512_add_pd(s0r, s2r));
                _mm512_store_pd(bi_ + o, _mm512_add_pd(s0i, s2i));
                _mm512_store_pd(br_ + o + 2 * qs, _mm512_sub_pd(s0r, s2r));
                _mm512_store_pd(bi_ + o + 2 * qs, _mm512_sub_pd(s0i, s2i));
                _mm512_store_pd(br_ + o + qs, _mm512_sub_pd(s1r, s3i));
                _mm512_store_pd(bi_ + o + qs, _mm512_add_pd(s1i, s3r));
                _mm512_store_pd(br_ + o + 3 * qs, _mm512_add_pd(s1r, s3i));
                _mm512_store_pd(bi_ + o + 3 * qs, _mm512_sub_pd(s1i, s3r));
            }
        }
    }
}
#endif /* __AVX512F__ */

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

#ifdef __AVX512F__
/* Everything between the pruned first stage and the pruned last stage:
 * remaining forward stages, fused pointwise middle, and the inverse stages
 * short of len == M (which dit4_last / the fused scatters own). */
static void conv_rows_mid(const fft3d_plan *p, double *restrict wr, double *restrict wi)
{
    const int M = p->M;
    const double *t = p->twf + (size_t)(M >> 2) * 6;
    int len = M >> 2;
    while (len >= 8) {
#ifndef BST_NOR16
        if (len >= 32) { /* fuse this stage with the next: one pass, not two */
            dif16_stage(M, len, t, t + (size_t)(len >> 2) * 6, wr, wi);
            t += (size_t)((len >> 2) + (len >> 4)) * 6;
            len >>= 4;
            continue;
        }
#endif
        dif4_stage(M, len, t, wr, wi);
        t += (size_t)(len >> 2) * 6;
        len >>= 2;
    }
    if (len == 4)
        conv_mid4(M, p->bhre, p->bhim, wr, wi);
    else if (len == 2)
        conv_mid2(M, p->bhre, p->bhim, wr, wi);
    else { /* M == 4: the first stage was the whole forward */
        for (int m = 0; m < M; ++m) {
            __m512d hr = _mm512_set1_pd(p->bhre[m]), hi = _mm512_set1_pd(p->bhim[m]);
            double *qr = wr + (size_t)m * VL, *qi = wi + (size_t)m * VL;
            __m512d xr = _mm512_load_pd(qr), xi = _mm512_load_pd(qi);
            _mm512_store_pd(qr, _mm512_fmsub_pd(xr, hr, _mm512_mul_pd(xi, hi)));
            _mm512_store_pd(qi, _mm512_fmadd_pd(xr, hi, _mm512_mul_pd(xi, hr)));
        }
    }
    if (M > 4) {
        const double *ti = p->twi;
        int l2 = (len == 4) ? 16 : 8;
        if (len == 4) ti += 6; /* skip the fused len-4 stage's twiddle chunk */
        while (l2 < M) {
#ifndef BST_NOR16
            if (4 * l2 < M) { /* fuse this stage with 4*l2: one pass, not two */
                dit16_stage(M, 4 * l2, ti, ti + (size_t)(l2 >> 2) * 6, wr, wi);
                ti += (size_t)((l2 >> 2) + l2) * 6;
                l2 <<= 4;
                continue;
            }
#endif
            dit4_stage(M, l2, ti, wr, wi);
            ti += (size_t)(l2 >> 2) * 6;
            l2 <<= 2;
        }
    }
}
#endif /* __AVX512F__ */

/* The whole per-row-group circular convolution: forward (input-pruned),
 * fused middle, inverse (output-pruned).  Scalar fallback runs the generic
 * pipeline and needs the caller to have zeroed the pad. */
static void conv_rows(const fft3d_plan *p, double *restrict wr, double *restrict wi)
{
    const int M = p->M;
#ifdef __AVX512F__
    const int L = p->L;
    dif4_first(M, L, p->twf, wr, wi);
    conv_rows_mid(p, wr, wi);
    dit4_last(M, L, p->twi_last, wr, wi);
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

    p->chre = xalloc(L + 4); /* +4: the tr-mode boundary block chirps b-legs   */
    p->chim = xalloc(L + 4); /*     up to index L+3; those lanes are zero, the */
                             /*     slack entries just need to be finite       */
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

    /* chirp: gen_twiddle's tw_chirp (exact k^2 mod 2L reduction, octant fold) */
    for (int j = 0; j < L; ++j) tw_chirp(j, L, &p->chre[j], &p->chim[j]);
    for (int j = L; j < L + 4; ++j) { p->chre[j] = 1.0; p->chim[j] = 0.0; }

    /* forward twiddles (consumption order), inverse = conjugates, mirrored
     * order; tw_fill_ct_int_colmajor is gen_twiddle's filler for exactly this
     * per-stage layout, audited to <= 0.51 ulp per part at create() */
    {
        double *tf = p->twf;
        for (int len = M; len >= 4; len >>= 2) {
            int S = len >> 2;
            tw_fill_ct_int_colmajor(tf, len, 4, S);
            if (tw_audit_ct_int_colmajor(tf, len, 4, S) > 0.51) {
                fft3d_destroy(p); /* cannot happen; audit is the contract */
                return NULL;
            }
            tf += (size_t)S * 6;
        }
        double *ti = p->twi;
        int odd_len = M;
        while (odd_len >= 4) odd_len >>= 2;
        for (int len = (odd_len == 2) ? 8 : 4; len <= M; len <<= 2) {
            int S = len >> 2;
            tw_fill_ct_int_colmajor(ti, len, 4, S);
            for (int j = 0; j < S * 6; j += 2) ti[j + 1] = -ti[j + 1]; /* conjugate */
            ti += (size_t)S * 6;
        }
        p->twi_last = p->twi + twn - (size_t)(M >> 2) * 6;
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

/* ---- gather/scatter fused into the pruned end stages (gen_r2) --------------
 * The r1 gather pass wrote L elements to w and dif4_first immediately re-read
 * them (and mirrored on the output side): 4L zmm of pure L1 round-trip per
 * group.  These kernels feed the first-stage butterfly straight from the
 * chirped source rows and scatter dit4_last's outputs straight to dst. */

/* first stage butterfly on chirped inputs a (always) and b (j < jful only),
 * writing the four quarters at index j; twiddles t[0..5] */
#define BST_FIRST_STORE(j, ar, ai, br, bi, t, wr, wi, S)                          \
    do {                                                                          \
        __m512d w1r = _mm512_set1_pd((t)[0]), w1i = _mm512_set1_pd((t)[1]);       \
        __m512d w2r = _mm512_set1_pd((t)[2]), w2i = _mm512_set1_pd((t)[3]);       \
        __m512d w3r = _mm512_set1_pd((t)[4]), w3i = _mm512_set1_pd((t)[5]);       \
        _mm512_store_pd((wr) + (size_t)(j) * VL, _mm512_add_pd(ar, br));          \
        _mm512_store_pd((wi) + (size_t)(j) * VL, _mm512_add_pd(ai, bi));          \
        __m512d u2r = _mm512_sub_pd(ar, br), u2i = _mm512_sub_pd(ai, bi);         \
        _mm512_store_pd((wr) + (size_t)((j) + 2 * (S)) * VL,                      \
                        _mm512_fmsub_pd(u2r, w2r, _mm512_mul_pd(u2i, w2i)));      \
        _mm512_store_pd((wi) + (size_t)((j) + 2 * (S)) * VL,                      \
                        _mm512_fmadd_pd(u2r, w2i, _mm512_mul_pd(u2i, w2r)));      \
        __m512d u1r = _mm512_add_pd(ar, bi), u1i = _mm512_sub_pd(ai, br);         \
        _mm512_store_pd((wr) + (size_t)((j) + (S)) * VL,                          \
                        _mm512_fmsub_pd(u1r, w1r, _mm512_mul_pd(u1i, w1i)));      \
        _mm512_store_pd((wi) + (size_t)((j) + (S)) * VL,                          \
                        _mm512_fmadd_pd(u1r, w1i, _mm512_mul_pd(u1i, w1r)));      \
        __m512d u3r = _mm512_sub_pd(ar, bi), u3i = _mm512_add_pd(ai, br);         \
        _mm512_store_pd((wr) + (size_t)((j) + 3 * (S)) * VL,                      \
                        _mm512_fmsub_pd(u3r, w3r, _mm512_mul_pd(u3i, w3i)));      \
        _mm512_store_pd((wi) + (size_t)((j) + 3 * (S)) * VL,                      \
                        _mm512_fmadd_pd(u3r, w3i, _mm512_mul_pd(u3i, w3r)));      \
    } while (0)

/* same, single-leg region (b = 0): four twiddled copies of a */
#define BST_FIRST_STORE1(j, ar, ai, t, wr, wi, S)                                 \
    do {                                                                          \
        __m512d w1r = _mm512_set1_pd((t)[0]), w1i = _mm512_set1_pd((t)[1]);       \
        __m512d w2r = _mm512_set1_pd((t)[2]), w2i = _mm512_set1_pd((t)[3]);       \
        __m512d w3r = _mm512_set1_pd((t)[4]), w3i = _mm512_set1_pd((t)[5]);       \
        _mm512_store_pd((wr) + (size_t)(j) * VL, ar);                             \
        _mm512_store_pd((wi) + (size_t)(j) * VL, ai);                             \
        _mm512_store_pd((wr) + (size_t)((j) + (S)) * VL,                          \
                        _mm512_fmsub_pd(ar, w1r, _mm512_mul_pd(ai, w1i)));        \
        _mm512_store_pd((wi) + (size_t)((j) + (S)) * VL,                          \
                        _mm512_fmadd_pd(ar, w1i, _mm512_mul_pd(ai, w1r)));        \
        _mm512_store_pd((wr) + (size_t)((j) + 2 * (S)) * VL,                      \
                        _mm512_fmsub_pd(ar, w2r, _mm512_mul_pd(ai, w2i)));        \
        _mm512_store_pd((wi) + (size_t)((j) + 2 * (S)) * VL,                      \
                        _mm512_fmadd_pd(ar, w2i, _mm512_mul_pd(ai, w2r)));        \
        _mm512_store_pd((wr) + (size_t)((j) + 3 * (S)) * VL,                      \
                        _mm512_fmsub_pd(ar, w3r, _mm512_mul_pd(ai, w3i)));        \
        _mm512_store_pd((wi) + (size_t)((j) + 3 * (S)) * VL,                      \
                        _mm512_fmadd_pd(ar, w3i, _mm512_mul_pd(ai, w3r)));        \
    } while (0)

/* chirp-multiply a split pair in place: (xr,xi) *= ch[j] */
#define BST_CHIRP(xr, xi, chre, chim, j)                                          \
    do {                                                                          \
        __m512d cr_ = _mm512_set1_pd((chre)[j]), ci_ = _mm512_set1_pd((chim)[j]); \
        __m512d tr_ = _mm512_fmsub_pd(xr, cr_, _mm512_mul_pd(xi, ci_));           \
        xi = _mm512_fmadd_pd(xr, ci_, _mm512_mul_pd(xi, cr_));                    \
        xr = tr_;                                                                 \
    } while (0)

/* The graded map, fused into the axis-0 scatter for fft3d_chain:
 *   (zr,zi) <- (z + c) / (1 + |z + c|)
 * rsqrt14 + 2 quadratic Newtons for |w|, rcp14 + 2 residual Newtons for the
 * reciprocal -- gen_batchlane's ladder verbatim (itself gen_pfa_small r1's):
 * no divider op, exact-tier (their two-step gate measured ~1e-15 vs 3e-14). */
#define BST_MAP8(zr, zi, car, cai)                                                \
    do {                                                                          \
        __m512d wr_ = _mm512_add_pd(zr, car), wi_ = _mm512_add_pd(zi, cai);       \
        __m512d s_ = _mm512_fmadd_pd(wr_, wr_, _mm512_set1_pd(1e-300));           \
        s_ = _mm512_fmadd_pd(wi_, wi_, s_);                                       \
        __m512d y_ = _mm512_rsqrt14_pd(s_);                                       \
        __m512d hs_ = _mm512_mul_pd(s_, _mm512_set1_pd(0.5));                     \
        __m512d u_ = _mm512_mul_pd(y_, y_);                                       \
        y_ = _mm512_mul_pd(y_, _mm512_fnmadd_pd(hs_, u_, _mm512_set1_pd(1.5)));   \
        u_ = _mm512_mul_pd(y_, y_);                                               \
        y_ = _mm512_mul_pd(y_, _mm512_fnmadd_pd(hs_, u_, _mm512_set1_pd(1.5)));   \
        __m512d d_ = _mm512_fmadd_pd(s_, y_, _mm512_set1_pd(1.0)); /* 1 + |w| */  \
        __m512d t_ = _mm512_rcp14_pd(d_);                                         \
        t_ = _mm512_fmadd_pd(                                                     \
            t_, _mm512_fnmadd_pd(d_, t_, _mm512_set1_pd(1.0)), t_);               \
        t_ = _mm512_fmadd_pd(                                                     \
            t_, _mm512_fnmadd_pd(d_, t_, _mm512_set1_pd(1.0)), t_);               \
        zr = _mm512_mul_pd(wr_, t_);                                              \
        zi = _mm512_mul_pd(wi_, t_);                                              \
    } while (0)

/* contiguous-lane rows (axes 0/1 off-seam): gather+chirp+first stage.
 * pf: software-prefetch distance in elements (0 = off).  The 2L read streams
 * of a strided row walk 128 B once and move on -- too many streams for the L2
 * streamer once the volume leaves L2, so on big volumes we prefetch by hand
 * (the ice "prefetch flips back to a win in the L3 regime" lesson). */
static void first_gather_contig(const fft3d_plan *p, const double *q, long s2,
                                double *restrict wr, double *restrict wi, int pf)
{
    const int S = p->M >> 2, jful = p->L - S;
    const double *restrict chre = p->chre, *restrict chim = p->chim;
    const double *t = p->twf;
    const __m512i IDXE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i IDXO = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    for (int j = 0; j < jful; ++j, t += 6) {
        if (pf) {
            _mm_prefetch((const char *)(q + (size_t)(j + pf) * s2), _MM_HINT_T0);
            _mm_prefetch((const char *)(q + (size_t)(j + pf) * s2 + 8), _MM_HINT_T0);
            _mm_prefetch((const char *)(q + (size_t)(j + S + pf) * s2), _MM_HINT_T0);
            _mm_prefetch((const char *)(q + (size_t)(j + S + pf) * s2 + 8), _MM_HINT_T0);
        }
        __m512d z0 = _mm512_loadu_pd(q + (size_t)j * s2);
        __m512d z1 = _mm512_loadu_pd(q + (size_t)j * s2 + 8);
        __m512d ar = _mm512_permutex2var_pd(z0, IDXE, z1);
        __m512d ai = _mm512_permutex2var_pd(z0, IDXO, z1);
        BST_CHIRP(ar, ai, chre, chim, j);
        z0 = _mm512_loadu_pd(q + (size_t)(j + S) * s2);
        z1 = _mm512_loadu_pd(q + (size_t)(j + S) * s2 + 8);
        __m512d br = _mm512_permutex2var_pd(z0, IDXE, z1);
        __m512d bi = _mm512_permutex2var_pd(z0, IDXO, z1);
        BST_CHIRP(br, bi, chre, chim, j + S);
        BST_FIRST_STORE(j, ar, ai, br, bi, t, wr, wi, S);
    }
    for (int j = jful; j < S; ++j, t += 6) {
        if (pf) {
            _mm_prefetch((const char *)(q + (size_t)(j + pf) * s2), _MM_HINT_T0);
            _mm_prefetch((const char *)(q + (size_t)(j + pf) * s2 + 8), _MM_HINT_T0);
        }
        __m512d z0 = _mm512_loadu_pd(q + (size_t)j * s2);
        __m512d z1 = _mm512_loadu_pd(q + (size_t)j * s2 + 8);
        __m512d ar = _mm512_permutex2var_pd(z0, IDXE, z1);
        __m512d ai = _mm512_permutex2var_pd(z0, IDXO, z1);
        BST_CHIRP(ar, ai, chre, chim, j);
        BST_FIRST_STORE1(j, ar, ai, t, wr, wi, S);
    }
}

/* contiguous rows, strided lanes (axis 2): 8x8-transpose gather + first stage.
 * Requires S % 4 == 0 (i.e. M >= 16); boundary block at jful runs per lane. */
static void first_gather_tr(const fft3d_plan *p, const double *const sbase[VL],
                            double *restrict wr, double *restrict wi)
{
    const int S = p->M >> 2, jful = p->L - S;
    const double *restrict chre = p->chre, *restrict chim = p->chim;
    const double *t = p->twf;
    const int jbf = jful & ~3;                            /* full dual blocks   */
    const int jb2 = (jbf < jful) ? jbf + 4 : jbf;         /* after boundary blk */
    for (int j = 0; j < jbf; j += 4) {
        __m512d ra[8], rb[8];
        for (int v = 0; v < VL; ++v) {
            ra[v] = _mm512_loadu_pd(sbase[v] + 2 * (size_t)j);
            rb[v] = _mm512_loadu_pd(sbase[v] + 2 * (size_t)(j + S));
        }
        tr8x8(ra);
        tr8x8(rb);
        for (int c = 0; c < 4; ++c, t += 6) {
            __m512d ar = ra[2 * c], ai = ra[2 * c + 1];
            __m512d br = rb[2 * c], bi = rb[2 * c + 1];
            BST_CHIRP(ar, ai, chre, chim, j + c);
            BST_CHIRP(br, bi, chre, chim, j + c + S);
            BST_FIRST_STORE(j + c, ar, ai, br, bi, t, wr, wi, S);
        }
    }
    if (jbf < jb2) { /* boundary block: b-leg zero-filled by a masked load, so
                      * the dual butterfly degenerates to the single-leg form
                      * exactly where b lanes are zero (chirp slack: see create) */
        const int j = jbf;
        const int nb = jful - j; /* 1..3 valid b elements */
        const __mmask8 bmask = (__mmask8)((1u << (2 * nb)) - 1u);
        __m512d ra[8], rb[8];
        for (int v = 0; v < VL; ++v) {
            ra[v] = _mm512_loadu_pd(sbase[v] + 2 * (size_t)j);
            rb[v] = _mm512_maskz_loadu_pd(bmask, sbase[v] + 2 * (size_t)(j + S));
        }
        tr8x8(ra);
        tr8x8(rb);
        for (int c = 0; c < 4; ++c, t += 6) {
            __m512d ar = ra[2 * c], ai = ra[2 * c + 1];
            __m512d br = rb[2 * c], bi = rb[2 * c + 1];
            BST_CHIRP(ar, ai, chre, chim, j + c);
            BST_CHIRP(br, bi, chre, chim, j + c + S);
            BST_FIRST_STORE(j + c, ar, ai, br, bi, t, wr, wi, S);
        }
    }
    for (int j = jb2; j < S; j += 4) { /* full single-leg blocks */
        __m512d ra[8];
        for (int v = 0; v < VL; ++v)
            ra[v] = _mm512_loadu_pd(sbase[v] + 2 * (size_t)j);
        tr8x8(ra);
        for (int c = 0; c < 4; ++c, t += 6) {
            __m512d ar = ra[2 * c], ai = ra[2 * c + 1];
            BST_CHIRP(ar, ai, chre, chim, j + c);
            BST_FIRST_STORE1(j + c, ar, ai, t, wr, wi, S);
        }
    }
}

/* ---- seam groups (axes 0/1): the 8 lanes straddle ONE div-block boundary:
 * lanes 0..k-1 are contiguous rows at q1, lanes k..7 contiguous rows at q2.
 * gen_r2 sent these to the scalar generic path -- at L=10/15 that was 60%/47%
 * of axis-1 groups.  Here: two fault-suppressed masked loads blended into the
 * same interleaved shape the contig path uses, then the identical pruned
 * pipeline.  Per element: 4 masked loads + 2 permutex2var (vs contig's 2+2).
 * Masks are group constants.  Requires div >= VL (one boundary max).       */
#define BST_SEAM_LOAD(o, xr, xi)                                                   \
    do {                                                                           \
        __m512d z0_ = _mm512_mask_loadu_pd(                                        \
            _mm512_maskz_loadu_pd(nlo, q2l + (o)), mlo, q1 + (o));                 \
        __m512d z1_ = _mm512_mask_loadu_pd(                                        \
            _mm512_maskz_loadu_pd(nhi, q2h + (o)), mhi, q1 + (o) + 8);             \
        xr = _mm512_permutex2var_pd(z0_, IDXE, z1_);                               \
        xi = _mm512_permutex2var_pd(z0_, IDXO, z1_);                               \
    } while (0)

static void first_gather_seam(const fft3d_plan *p, const double *q1,
                              const double *q2, int k, long s2,
                              double *restrict wr, double *restrict wi)
{
    const int S = p->M >> 2, jful = p->L - S;
    const double *restrict chre = p->chre, *restrict chim = p->chim;
    const double *t = p->twf;
    const __m512i IDXE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i IDXO = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    const int tk = 2 * k; /* doubles supplied by run 1 */
    const __mmask8 mlo = (tk >= 8) ? (__mmask8)0xFF : (__mmask8)((1u << tk) - 1u);
    const __mmask8 mhi = (tk <= 8) ? (__mmask8)0 : (__mmask8)((1u << (tk - 8)) - 1u);
    const __mmask8 nlo = (__mmask8)~mlo, nhi = (__mmask8)~mhi;
    const double *q2l = q2 - tk; /* run-2 doubles land at positions >= tk   */
    const double *q2h = q2 + 8 - tk;
    for (int j = 0; j < jful; ++j, t += 6) {
        __m512d ar, ai, br, bi;
        BST_SEAM_LOAD((size_t)j * s2, ar, ai);
        BST_CHIRP(ar, ai, chre, chim, j);
        BST_SEAM_LOAD((size_t)(j + S) * s2, br, bi);
        BST_CHIRP(br, bi, chre, chim, j + S);
        BST_FIRST_STORE(j, ar, ai, br, bi, t, wr, wi, S);
    }
    for (int j = jful; j < S; ++j, t += 6) {
        __m512d ar, ai;
        BST_SEAM_LOAD((size_t)j * s2, ar, ai);
        BST_CHIRP(ar, ai, chre, chim, j);
        BST_FIRST_STORE1(j, ar, ai, t, wr, wi, S);
    }
}

/* last inverse stage: load the four legs at j, twiddle, produce the k = j
 * output (always) and the k = j + S output (j < jful only) in registers */
#define BST_LAST_LOAD(j, t, wr, wi, S, r0r, r0i, r1r, r1i, has1)                  \
    do {                                                                          \
        __m512d w1r = _mm512_set1_pd((t)[0]), w1i = _mm512_set1_pd((t)[1]);       \
        __m512d w2r = _mm512_set1_pd((t)[2]), w2i = _mm512_set1_pd((t)[3]);       \
        __m512d w3r = _mm512_set1_pd((t)[4]), w3i = _mm512_set1_pd((t)[5]);       \
        __m512d u0r = _mm512_load_pd((wr) + (size_t)(j) * VL);                    \
        __m512d u0i = _mm512_load_pd((wi) + (size_t)(j) * VL);                    \
        __m512d y1r = _mm512_load_pd((wr) + (size_t)((j) + (S)) * VL);            \
        __m512d y1i = _mm512_load_pd((wi) + (size_t)((j) + (S)) * VL);            \
        __m512d y2r = _mm512_load_pd((wr) + (size_t)((j) + 2 * (S)) * VL);        \
        __m512d y2i = _mm512_load_pd((wi) + (size_t)((j) + 2 * (S)) * VL);        \
        __m512d y3r = _mm512_load_pd((wr) + (size_t)((j) + 3 * (S)) * VL);        \
        __m512d y3i = _mm512_load_pd((wi) + (size_t)((j) + 3 * (S)) * VL);        \
        __m512d u1r = _mm512_fmsub_pd(y1r, w1r, _mm512_mul_pd(y1i, w1i));         \
        __m512d u1i = _mm512_fmadd_pd(y1r, w1i, _mm512_mul_pd(y1i, w1r));         \
        __m512d u2r = _mm512_fmsub_pd(y2r, w2r, _mm512_mul_pd(y2i, w2i));         \
        __m512d u2i = _mm512_fmadd_pd(y2r, w2i, _mm512_mul_pd(y2i, w2r));         \
        __m512d u3r = _mm512_fmsub_pd(y3r, w3r, _mm512_mul_pd(y3i, w3i));         \
        __m512d u3i = _mm512_fmadd_pd(y3r, w3i, _mm512_mul_pd(y3i, w3r));         \
        r0r = _mm512_add_pd(_mm512_add_pd(u0r, u2r), _mm512_add_pd(u1r, u3r));    \
        r0i = _mm512_add_pd(_mm512_add_pd(u0i, u2i), _mm512_add_pd(u1i, u3i));    \
        if (has1) {                                                               \
            __m512d s1r = _mm512_sub_pd(u0r, u2r), s1i = _mm512_sub_pd(u0i, u2i); \
            __m512d s3r = _mm512_sub_pd(u1r, u3r), s3i = _mm512_sub_pd(u1i, u3i); \
            r1r = _mm512_sub_pd(s1r, s3i);                                        \
            r1i = _mm512_add_pd(s1i, s3r);                                        \
        }                                                                         \
    } while (0)

/* contiguous-lane rows: last stage + chirp + interleaving scatter.
 * cq != NULL (fft3d_chain's axis-0 pass) fuses  dst = (z + c)/(1 + |z + c|)
 * into the same store pass; always_inline + a literal NULL at the execute
 * call site keeps the no-map specialization free of the map branch. */
static inline __attribute__((always_inline)) void
last_scatter_contig_core(const fft3d_plan *p, double *q, long s2,
                         const double *restrict wr, const double *restrict wi,
                         int pf, const double *cq)
{
    const int S = p->M >> 2, jful = p->L - S;
    const double *restrict chre = p->chre, *restrict chim = p->chim;
    const double *t = p->twi_last;
    const __m512i IDXE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i IDXO = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    const __m512i IDXL = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i IDXH = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    for (int j = 0; j < S; ++j, t += 6) {
        if (pf) { /* write-intent prefetch of the strided store targets */
            __builtin_prefetch(q + (size_t)(j + pf) * s2, 1, 3);
            __builtin_prefetch(q + (size_t)(j + pf) * s2 + 8, 1, 3);
            __builtin_prefetch(q + (size_t)(j + S + pf) * s2, 1, 3);
            __builtin_prefetch(q + (size_t)(j + S + pf) * s2 + 8, 1, 3);
        }
        __m512d r0r, r0i, r1r, r1i;
        const int has1 = j < jful;
        BST_LAST_LOAD(j, t, wr, wi, S, r0r, r0i, r1r, r1i, has1);
        BST_CHIRP(r0r, r0i, chre, chim, j);
        if (cq) {
            __m512d z0 = _mm512_loadu_pd(cq + (size_t)j * s2);
            __m512d z1 = _mm512_loadu_pd(cq + (size_t)j * s2 + 8);
            __m512d car = _mm512_permutex2var_pd(z0, IDXE, z1);
            __m512d cai = _mm512_permutex2var_pd(z0, IDXO, z1);
            BST_MAP8(r0r, r0i, car, cai);
        }
        _mm512_storeu_pd(q + (size_t)j * s2, _mm512_permutex2var_pd(r0r, IDXL, r0i));
        _mm512_storeu_pd(q + (size_t)j * s2 + 8,
                         _mm512_permutex2var_pd(r0r, IDXH, r0i));
        if (has1) {
            BST_CHIRP(r1r, r1i, chre, chim, j + S);
            if (cq) {
                __m512d z0 = _mm512_loadu_pd(cq + (size_t)(j + S) * s2);
                __m512d z1 = _mm512_loadu_pd(cq + (size_t)(j + S) * s2 + 8);
                __m512d car = _mm512_permutex2var_pd(z0, IDXE, z1);
                __m512d cai = _mm512_permutex2var_pd(z0, IDXO, z1);
                BST_MAP8(r1r, r1i, car, cai);
            }
            _mm512_storeu_pd(q + (size_t)(j + S) * s2,
                             _mm512_permutex2var_pd(r1r, IDXL, r1i));
            _mm512_storeu_pd(q + (size_t)(j + S) * s2 + 8,
                             _mm512_permutex2var_pd(r1r, IDXH, r1i));
        }
    }
}

static void last_scatter_contig(const fft3d_plan *p, double *q, long s2,
                                const double *restrict wr, const double *restrict wi,
                                int pf)
{
    last_scatter_contig_core(p, q, s2, wr, wi, pf, NULL);
}

static void last_scatter_contig_map(const fft3d_plan *p, double *q, long s2,
                                    const double *restrict wr,
                                    const double *restrict wi, int pf,
                                    const double *cq)
{
    last_scatter_contig_core(p, q, s2, wr, wi, pf, cq);
}

/* seam-group mirror: last stage + chirp (+ optional fused map) + interleave,
 * stored through the group-constant masks back into the two runs. */
static void last_scatter_seam(const fft3d_plan *p, double *q1, double *q2, int k,
                              long s2, const double *restrict wr,
                              const double *restrict wi,
                              const double *cq1, const double *cq2)
{
    const int S = p->M >> 2, jful = p->L - S;
    const double *restrict chre = p->chre, *restrict chim = p->chim;
    const double *t = p->twi_last;
    const __m512i IDXE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i IDXO = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    const __m512i IDXL = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i IDXH = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    const int tk = 2 * k;
    const __mmask8 mlo = (tk >= 8) ? (__mmask8)0xFF : (__mmask8)((1u << tk) - 1u);
    const __mmask8 mhi = (tk <= 8) ? (__mmask8)0 : (__mmask8)((1u << (tk - 8)) - 1u);
    const __mmask8 nlo = (__mmask8)~mlo, nhi = (__mmask8)~mhi;
    double *q2l = q2 - tk, *q2h = q2 + 8 - tk;
    const double *cq2l = cq2 ? cq2 - tk : NULL, *cq2h = cq2 ? cq2 + 8 - tk : NULL;
    for (int j = 0; j < S; ++j, t += 6) {
        __m512d r0r, r0i, r1r, r1i;
        const int has1 = j < jful;
        BST_LAST_LOAD(j, t, wr, wi, S, r0r, r0i, r1r, r1i, has1);
        BST_CHIRP(r0r, r0i, chre, chim, j);
        if (cq1) {
            const size_t o = (size_t)j * s2;
            __m512d z0 = _mm512_mask_loadu_pd(
                _mm512_maskz_loadu_pd(nlo, cq2l + o), mlo, cq1 + o);
            __m512d z1 = _mm512_mask_loadu_pd(
                _mm512_maskz_loadu_pd(nhi, cq2h + o), mhi, cq1 + o + 8);
            __m512d car = _mm512_permutex2var_pd(z0, IDXE, z1);
            __m512d cai = _mm512_permutex2var_pd(z0, IDXO, z1);
            BST_MAP8(r0r, r0i, car, cai);
        }
        {
            const size_t o = (size_t)j * s2;
            __m512d c0 = _mm512_permutex2var_pd(r0r, IDXL, r0i);
            __m512d c1 = _mm512_permutex2var_pd(r0r, IDXH, r0i);
            _mm512_mask_storeu_pd(q1 + o, mlo, c0);
            _mm512_mask_storeu_pd(q1 + o + 8, mhi, c1);
            _mm512_mask_storeu_pd(q2l + o, nlo, c0);
            _mm512_mask_storeu_pd(q2h + o, nhi, c1);
        }
        if (has1) {
            BST_CHIRP(r1r, r1i, chre, chim, j + S);
            const size_t o = (size_t)(j + S) * s2;
            if (cq1) {
                __m512d z0 = _mm512_mask_loadu_pd(
                    _mm512_maskz_loadu_pd(nlo, cq2l + o), mlo, cq1 + o);
                __m512d z1 = _mm512_mask_loadu_pd(
                    _mm512_maskz_loadu_pd(nhi, cq2h + o), mhi, cq1 + o + 8);
                __m512d car = _mm512_permutex2var_pd(z0, IDXE, z1);
                __m512d cai = _mm512_permutex2var_pd(z0, IDXO, z1);
                BST_MAP8(r1r, r1i, car, cai);
            }
            __m512d c0 = _mm512_permutex2var_pd(r1r, IDXL, r1i);
            __m512d c1 = _mm512_permutex2var_pd(r1r, IDXH, r1i);
            _mm512_mask_storeu_pd(q1 + o, mlo, c0);
            _mm512_mask_storeu_pd(q1 + o + 8, mhi, c1);
            _mm512_mask_storeu_pd(q2l + o, nlo, c0);
            _mm512_mask_storeu_pd(q2h + o, nhi, c1);
        }
    }
}

/* contiguous rows, strided lanes: last stage + chirp + 8x8-transpose scatter.
 * Requires S % 4 == 0. */
static void last_scatter_tr(const fft3d_plan *p, double *const dbase[VL], int nv,
                            const double *restrict wr, const double *restrict wi)
{
    const int S = p->M >> 2, jful = p->L - S;
    const double *restrict chre = p->chre, *restrict chim = p->chim;
    const double *t = p->twi_last;
    for (int j = 0; j < S; j += 4, t += 24) {
        __m512d ro[8], r1o[8];
        const int n1 = jful - j; /* r1 outputs live in this block: c < n1 */
        for (int c = 0; c < 4; ++c) {
            __m512d r0r, r0i, r1r, r1i;
            const int has1 = c < n1;
            BST_LAST_LOAD(j + c, t + 6 * c, wr, wi, S, r0r, r0i, r1r, r1i, has1);
            BST_CHIRP(r0r, r0i, chre, chim, j + c);
            ro[2 * c] = r0r;
            ro[2 * c + 1] = r0i;
            if (has1) {
                BST_CHIRP(r1r, r1i, chre, chim, j + c + S);
                r1o[2 * c] = r1r;
                r1o[2 * c + 1] = r1i;
            }
        }
        tr8x8(ro);
        for (int v = 0; v < nv; ++v)
            _mm512_storeu_pd(dbase[v] + 2 * (size_t)j, ro[v]);
        if (n1 >= 4) {
            tr8x8(r1o);
            for (int v = 0; v < nv; ++v)
                _mm512_storeu_pd(dbase[v] + 2 * (size_t)(j + S), r1o[v]);
        } else if (n1 > 0) { /* partial block: transpose, store only valid pairs */
            const __mmask8 m1 = (__mmask8)((1u << (2 * n1)) - 1u);
            for (int c = n1; c < 4; ++c)
                r1o[2 * c] = r1o[2 * c + 1] = _mm512_setzero_pd();
            tr8x8(r1o);
            for (int v = 0; v < nv; ++v)
                _mm512_mask_storeu_pd(dbase[v] + 2 * (size_t)(j + S), m1, r1o[v]);
        }
    }
}
#endif

/* One Bluestein pass over all rows of one axis.  Row r (0..nrows-1) starts at
 * complex offset (r/div)*A + (r%div)*B from base and strides by `stride`
 * complex elements.  src rows are read, dst rows written at the SAME offsets
 * (safe in place: each row depends only on itself).
 * cmap != NULL (fft3d_chain, axis 0 only): the scatter stores
 * (z + c)/(1 + |z + c|) instead of z, c read at the dst offsets. */
static void axis_pass(fft3d_plan *p, const double *src, double *dst,
                      long rlo, long rhi, long div, long A, long B, long stride,
                      const double *cmap)
{
    const int L = p->L;
    double *restrict wr = p->wr, *restrict wi = p->wi;
    const double *restrict chre = p->chre, *restrict chim = p->chim;
    const long s2 = 2 * stride;

    for (long r0 = rlo; r0 < rhi; r0 += VL) {
        int nv = (rhi - r0 < VL) ? (int)(rhi - r0) : VL;
        const double *sbase[VL];
        double *dbase[VL];
        long off0 = 0;
        for (int v = 0; v < VL; ++v) {
            long r = r0 + (v < nv ? v : 0);
            long off = 2 * ((r / div) * A + (r % div) * B);
            if (v == 0) off0 = off;
            sbase[v] = src + off;
            dbase[v] = dst + off;
        }

#ifdef __AVX512F__
        /* lanes contiguous in memory?  (axes 0/1 except at outer-run seams) */
        const int contig = (B == 1) && (nv == VL) && (r0 % div) + VL <= div;
        if (contig) {
#ifdef BST_PF
            /* strided big-volume passes: hand-prefetch.  Raced OFF by default:
             * a same-window A/B at L=100 read 19.48/19.34 ms without vs
             * 19.67/19.63 with -- the OOO window already covers the latency. */
            const int pf = (stride >= 8 &&
                            (size_t)L * L * L * 16 > (size_t)1200 * 1024) ? 8 : 0;
#else
            const int pf = 0;
#endif
            first_gather_contig(p, sbase[0], s2, wr, wi, pf);
            conv_rows_mid(p, wr, wi);
            if (cmap)
                last_scatter_contig_map(p, dbase[0], s2, wr, wi, pf, cmap + off0);
            else
                last_scatter_contig(p, dbase[0], s2, wr, wi, pf);
            continue;
        }
#ifndef BST_NOSEAM
        if ((B == 1) && (nv == VL) && div >= VL) {
            /* seam group: one div-block boundary inside the 8 lanes */
            const int k = (int)(div - (r0 % div)); /* lanes in run 1: 1..7 */
            const long offk = dbase[k] - dst;
            first_gather_seam(p, sbase[0], sbase[k], k, s2, wr, wi);
            conv_rows_mid(p, wr, wi);
            last_scatter_seam(p, dbase[0], dbase[k], k, s2, wr, wi,
                              cmap ? cmap + off0 : NULL,
                              cmap ? cmap + offk : NULL);
            continue;
        }
#endif
        if (stride == 1 && p->M >= 16) {
            /* axis 2: rows contiguous, lanes strided -> 8x8 transpose blocks */
            first_gather_tr(p, sbase, wr, wi);
            conv_rows_mid(p, wr, wi);
            last_scatter_tr(p, dbase, nv, wr, wi);
            continue;
        }
#endif

        /* generic fallback (tail groups, tiny M/div, non-AVX512 builds):
         * gather * chirp, full convolution, scatter * chirp (* map) */
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
#ifndef __AVX512F__
        memset(wr + (size_t)L * VL, 0, (size_t)(p->M - L) * VL * sizeof(double));
        memset(wi + (size_t)L * VL, 0, (size_t)(p->M - L) * VL * sizeof(double));
#endif
        conv_rows(p, wr, wi);

        for (int k = 0; k < L; ++k) {
            const double cr = chre[k], ci = chim[k];
            const long o = (long)k * s2;
            const double *restrict qr = wr + (size_t)k * VL;
            const double *restrict qi = wi + (size_t)k * VL;
            for (int v = 0; v < nv; ++v) {
                double xr = qr[v], xi = qi[v];
                double zr = xr * cr - xi * ci;
                double zi = xr * ci + xi * cr;
                if (cmap) {
                    const double *cv = cmap + (dbase[v] - dst);
                    zr += cv[o];
                    zi += cv[o + 1];
                    double sc = 1.0 / (1.0 + sqrt(zr * zr + zi * zi + 1e-300));
                    zr *= sc;
                    zi *= sc;
                }
                dbase[v][o] = zr;
                dbase[v][o + 1] = zi;
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

#ifndef BST_AXES
#define BST_AXES 3 /* dev knob: 1/2 time a partial (WRONG) transform */
#endif
    /* axis 2 (z, stride 1): row r starts at r*L      -> in  to out */
    axis_pass(p, src, dst, 0, nrows, 1, L, 0, 1, NULL);
#if BST_AXES >= 2
    /* axis 1 (y, stride L): row r -> (r/L)*L^2 + r%L -> out in place */
    axis_pass(p, dst, dst, 0, nrows, L, L * L, 1, L, NULL);
#endif
#if BST_AXES >= 3
    /* axis 0 (x, stride L^2): row r -> (r/L^2)*L^3 + r%L^2 */
    axis_pass(p, dst, dst, 0, nrows, L * L, L * L * L, 1, L * L, NULL);
#endif
}

#ifdef __AVX512F__
/* separate sequential map pass, in place on the state volume: for working
 * sets past LLC reach the axis-0 scatter's c reads are strided (L^2 apart)
 * and latency-bound; a straight-line sweep keeps all three streams (z, c,
 * store) hardware-prefetchable.  Same ladder, same arithmetic. */
static void map_pass_seq(double *restrict zd, const double *restrict cd,
                         size_t count)
{
    const __m512i IDXE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i IDXO = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    const __m512i IDXL = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i IDXH = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m512d z0 = _mm512_loadu_pd(zd + 2 * i);
        __m512d z1 = _mm512_loadu_pd(zd + 2 * i + 8);
        __m512d zr = _mm512_permutex2var_pd(z0, IDXE, z1);
        __m512d zi = _mm512_permutex2var_pd(z0, IDXO, z1);
        __m512d c0 = _mm512_loadu_pd(cd + 2 * i);
        __m512d c1 = _mm512_loadu_pd(cd + 2 * i + 8);
        __m512d car = _mm512_permutex2var_pd(c0, IDXE, c1);
        __m512d cai = _mm512_permutex2var_pd(c0, IDXO, c1);
        BST_MAP8(zr, zi, car, cai);
        _mm512_storeu_pd(zd + 2 * i, _mm512_permutex2var_pd(zr, IDXL, zi));
        _mm512_storeu_pd(zd + 2 * i + 8, _mm512_permutex2var_pd(zr, IDXH, zi));
    }
    for (; i < count; ++i) {
        double zr = zd[2 * i] + cd[2 * i], zi = zd[2 * i + 1] + cd[2 * i + 1];
        double sc = 1.0 / (1.0 + sqrt(zr * zr + zi * zi + 1e-300));
        zd[2 * i] = zr * sc;
        zd[2 * i + 1] = zi * sc;
    }
}
#endif /* __AVX512F__ */

/* The graded m-step map chain, owned (gen_r3): every pass is in place in
 * final_out (each row depends only on itself), and the map + c-add is fused
 * into the axis-0 scatter -- vs the driver fallback this deletes the per-step
 * map pass (one full-volume read + write plus c reread) and its scalar-ish
 * sqrt/div, and the initial memcpy.  x0 is never written.
 * Past LLC reach the fused scatter's strided c reads stop paying (raced on
 * the node, gen_r3) and the map runs as its own sequential sweep; the gate is
 * on state + c COMBINED (both stream through the scatter -- the r3 code gated
 * on state alone, which wrongly kept L=40/50 fused: 1987 vs 1822 us at L=50,
 * same window, gen_r4). */
#ifndef BST_MAPFUSE_MAX_MIB
#define BST_MAPFUSE_MAX_MIB 15
#endif
#ifndef BST_NOCHAIN
void fft3d_chain(fft3d_plan *p, const double _Complex *x0,
                 const double _Complex *c, double _Complex *final_out, int m)
{
    const long L = p->L, B = p->batch;
    const double *cm = (const double *)c;
    double *dst = (double *)final_out;
    const long nrows = B * L * L;
#ifdef __AVX512F__
    const int fuse =
        (size_t)nrows * (size_t)L * 32 <= ((size_t)BST_MAPFUSE_MAX_MIB << 20);
#else
    const int fuse = 1; /* scalar build: generic path maps in its scatter */
#endif

#if defined(__AVX512F__) && !defined(BST_NOBLOCK)
    if (!fuse) {
        /* Separate-map regime, k-plane-blocked custody (gen_layout r3's
         * plane-window idea): axis 0 runs first (global, strided), then per
         * block of k planes -- k*L rows with k = 8/gcd(L,8), so every block
         * is whole 8-row groups and the group decomposition is IDENTICAL to
         * the unblocked passes -- axis 2, axis 1 and the sequential map sweep
         * all run while the block (k*L^2*16 B <= 320 KiB at L <= 100) is
         * L2-hot.  Deletes the axis-1 full-volume round trip and the map's
         * state read+write from DRAM.  The map itself stays a sequential
         * sweep: fusing the ladder into gathers/scatters loses on this node
         * (my r3 race, and gen_pfa_large r3's ipm verdict -- not re-run). */
        long k = 1;
        while ((k * L) & 7) k <<= 1;
        const long rpb = k * L; /* rows per block, multiple of 8 */
        for (int s = 0; s < m; ++s) {
            const double *src = (s == 0) ? (const double *)x0 : dst;
            axis_pass(p, src, dst, 0, nrows, L * L, L * L * L, 1, L * L, NULL);
            for (long r0 = 0; r0 < nrows; r0 += rpb) {
                const long r1 = (r0 + rpb < nrows) ? r0 + rpb : nrows;
                axis_pass(p, dst, dst, r0, r1, 1, L, 0, 1, NULL);
                axis_pass(p, dst, dst, r0, r1, L, L * L, 1, L, NULL);
                map_pass_seq(dst + 2 * (size_t)r0 * L, cm + 2 * (size_t)r0 * L,
                             (size_t)(r1 - r0) * L);
            }
        }
        return;
    }
#endif

    for (int s = 0; s < m; ++s) {
        const double *src = (s == 0) ? (const double *)x0 : dst;
#if defined(__AVX512F__) && defined(BST_BLKFUSE)
        /* raced variant: axes 2+1 plane-blocked in the fused regime too */
        {
            long k = 1;
            while ((k * L) & 7) k <<= 1;
            const long rpb = k * L;
            for (long r0 = 0; r0 < nrows; r0 += rpb) {
                const long r1 = (r0 + rpb < nrows) ? r0 + rpb : nrows;
                axis_pass(p, src, dst, r0, r1, 1, L, 0, 1, NULL);
                axis_pass(p, dst, dst, r0, r1, L, L * L, 1, L, NULL);
            }
        }
#else
        axis_pass(p, src, dst, 0, nrows, 1, L, 0, 1, NULL);
        axis_pass(p, dst, dst, 0, nrows, L, L * L, 1, L, NULL);
#endif
        axis_pass(p, dst, dst, 0, nrows, L * L, L * L * L, 1, L * L,
                  fuse ? cm : NULL);
#ifdef __AVX512F__
        if (!fuse) map_pass_seq(dst, cm, (size_t)nrows * (size_t)L);
#endif
    }
}
#endif /* BST_NOCHAIN */
