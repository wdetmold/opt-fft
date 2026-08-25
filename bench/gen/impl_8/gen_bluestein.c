/* gen_bluestein: Bluestein chirp-Z existence fallback for ANY L.
 *
 * X_k = ch[k] * (a (*) b)[k],   a_j = x_j * ch[j],  ch[n] = exp(-i pi n^2 / L),
 * b_n = exp(+i pi n^2 / L) embedded circularly in M = the smallest 2^k,
 * 3*2^k, 5*2^k, or 7*2^k (k >= 4) >= 2L-1 -- the grid 48,64,80,96,112,128,
 * 160,192,224,256 cuts the convolution up to 25-37% below next_pow2
 * (gen_r6 + gen_r7; the 7-slice covers graded L=50 (M 128->112) and
 * L=100 (256->224)).  A 13*2^(2k) slice (M=208 for L=97..104, DFT-13 tail)
 * exists behind -DBST_M13 but lost its gen_r8 race at L=100 -- see create().
 *
 * The circular convolution runs through our own FFT:
 *   forward  = in-place radix-4 DIF down the 4|len chain, tail = radix-2,
 *              DFT-3, DFT-5, DFT-7, DFT-13, PFA(2x3), PFA(2x5), or PFA(2x7)
 *              blocks (all twiddle-free),
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
    double *cc;             /* custody copy of the chain's c operand, in the
                             * blocked axis-1 scatter's consumption order
                             * (lazily allocated by fft3d_chain) */
};

const char *fft3d_name(void) { return "gen_bluestein"; }
const char *fft3d_description(void)
{
    return "Bluestein chirp-Z for ANY L: {2,3,5,7}*2^k radix-4/16 DIF/DIT convolution (no bit-reversal; twiddle-free DFT-3/5/7 and PFA-6/10/14 fused middles cut M up to 37% below next_pow2), 8-row SoA lanes, gather/scatter fused into the pruned end stages (masked dual-run loads keep seam groups vectorized), owned in-place map chain -- map fused into the axis-0 scatter while state+c fit LLC, else axis-0-first k-plane-blocked custody with the map fused into the axis-1 scatter reading a custody-ordered c (two aligned sequential streams; gen_pow2 GP2_CT), gen_twiddle exact tables";
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

/* ---- twiddle-free tail stages for M = 3*2^k -------------------------------
 * After the radix-4 DIF chain the remaining sub-transforms are independent
 * DFTs on blocks of 3 (k even) or 6 (k odd) consecutive positions, inputs in
 * natural block order.  Outputs go to the pipeline's OWN slot order (DFT-6 =
 * PFA 2x3, no twiddles); bh is computed by running this same forward, so the
 * slot permutation never needs to be named.  The inverses are the exact
 * unnormalized (x3 / x6) stage inverses.  Plain C: these run at create() on
 * b_pad and in the scalar-build execute path only -- the AVX-512 execute path
 * uses the fused conv_mid3 / conv_mid6 below. */
#define BST_K3 0.86602540378443864676372317075293618347 /* sqrt(3)/2 */

static void dft3_fwd_stage(int M, double *restrict wr, double *restrict wi)
{
    for (int blk = 0; blk < M; blk += 3) {
        double *ar = wr + (size_t)blk * VL, *ai = wi + (size_t)blk * VL;
        double *br = ar + VL, *bi = ai + VL;
        double *cr = ar + 2 * VL, *ci = ai + 2 * VL;
        for (int v = 0; v < VL; ++v) {
            double tr = br[v] + cr[v], ti = bi[v] + ci[v];
            double mr = ar[v] - 0.5 * tr, mi = ai[v] - 0.5 * ti;
            double sr = BST_K3 * (br[v] - cr[v]), si = BST_K3 * (bi[v] - ci[v]);
            ar[v] += tr;
            ai[v] += ti;
            br[v] = mr + si; /* y1 = m - i*s */
            bi[v] = mi - sr;
            cr[v] = mr - si; /* y2 = m + i*s */
            ci[v] = mi + sr;
        }
    }
}

static void dft3_inv_stage(int M, double *restrict wr, double *restrict wi)
{
    for (int blk = 0; blk < M; blk += 3) {
        double *ar = wr + (size_t)blk * VL, *ai = wi + (size_t)blk * VL;
        double *br = ar + VL, *bi = ai + VL;
        double *cr = ar + 2 * VL, *ci = ai + 2 * VL;
        for (int v = 0; v < VL; ++v) {
            double tr = br[v] + cr[v], ti = bi[v] + ci[v];
            double mr = ar[v] - 0.5 * tr, mi = ai[v] - 0.5 * ti;
            double sr = BST_K3 * (br[v] - cr[v]), si = BST_K3 * (bi[v] - ci[v]);
            ar[v] += tr;
            ai[v] += ti;
            br[v] = mr - si; /* x1 = m + i*s */
            bi[v] = mi + sr;
            cr[v] = mr + si; /* x2 = m - i*s */
            ci[v] = mi - sr;
        }
    }
}

/* DFT-6 as PFA(2x3): u = DFT3(x0,x2,x4), v = DFT3(x3,x5,x1), slots
 * (u0+v0, u0-v0, u1+v1, u1-v1, u2+v2, u2-v2).  No twiddles anywhere. */
static void dft6_fwd_stage(int M, double *restrict wr, double *restrict wi)
{
    for (int blk = 0; blk < M; blk += 6) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        for (int v = 0; v < VL; ++v) {
            double x0r = qr[v], x0i = qi[v];
            double x1r = qr[VL + v], x1i = qi[VL + v];
            double x2r = qr[2 * VL + v], x2i = qi[2 * VL + v];
            double x3r = qr[3 * VL + v], x3i = qi[3 * VL + v];
            double x4r = qr[4 * VL + v], x4i = qi[4 * VL + v];
            double x5r = qr[5 * VL + v], x5i = qi[5 * VL + v];
            double tur = x2r + x4r, tui = x2i + x4i;
            double u0r = x0r + tur, u0i = x0i + tui;
            double umr = x0r - 0.5 * tur, umi = x0i - 0.5 * tui;
            double usr = BST_K3 * (x2r - x4r), usi = BST_K3 * (x2i - x4i);
            double u1r = umr + usi, u1i = umi - usr;
            double u2r = umr - usi, u2i = umi + usr;
            double tvr = x5r + x1r, tvi = x5i + x1i;
            double v0r = x3r + tvr, v0i = x3i + tvi;
            double vmr = x3r - 0.5 * tvr, vmi = x3i - 0.5 * tvi;
            double vsr = BST_K3 * (x5r - x1r), vsi = BST_K3 * (x5i - x1i);
            double v1r = vmr + vsi, v1i = vmi - vsr;
            double v2r = vmr - vsi, v2i = vmi + vsr;
            qr[v] = u0r + v0r;
            qi[v] = u0i + v0i;
            qr[VL + v] = u0r - v0r;
            qi[VL + v] = u0i - v0i;
            qr[2 * VL + v] = u1r + v1r;
            qi[2 * VL + v] = u1i + v1i;
            qr[3 * VL + v] = u1r - v1r;
            qi[3 * VL + v] = u1i - v1i;
            qr[4 * VL + v] = u2r + v2r;
            qi[4 * VL + v] = u2i + v2i;
            qr[5 * VL + v] = u2r - v2r;
            qi[5 * VL + v] = u2i - v2i;
        }
    }
}

static void dft6_inv_stage(int M, double *restrict wr, double *restrict wi)
{
    for (int blk = 0; blk < M; blk += 6) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        for (int v = 0; v < VL; ++v) {
            double s0r = qr[v], s0i = qi[v];
            double s1r = qr[VL + v], s1i = qi[VL + v];
            double s2r = qr[2 * VL + v], s2i = qi[2 * VL + v];
            double s3r = qr[3 * VL + v], s3i = qi[3 * VL + v];
            double s4r = qr[4 * VL + v], s4i = qi[4 * VL + v];
            double s5r = qr[5 * VL + v], s5i = qi[5 * VL + v];
            double u0r = s0r + s1r, u0i = s0i + s1i; /* 2-pt is self-inverse */
            double v0r = s0r - s1r, v0i = s0i - s1i;
            double u1r = s2r + s3r, u1i = s2i + s3i;
            double v1r = s2r - s3r, v1i = s2i - s3i;
            double u2r = s4r + s5r, u2i = s4i + s5i;
            double v2r = s4r - s5r, v2i = s4i - s5i;
            double tur = u1r + u2r, tui = u1i + u2i;
            double umr = u0r - 0.5 * tur, umi = u0i - 0.5 * tui;
            double usr = BST_K3 * (u1r - u2r), usi = BST_K3 * (u1i - u2i);
            double tvr = v1r + v2r, tvi = v1i + v2i;
            double vmr = v0r - 0.5 * tvr, vmi = v0i - 0.5 * tvi;
            double vsr = BST_K3 * (v1r - v2r), vsi = BST_K3 * (v1i - v2i);
            qr[v] = u0r + tur;             /* inv DFT3(u) -> x0, x2, x4 */
            qi[v] = u0i + tui;
            qr[2 * VL + v] = umr - usi;    /* x2 = m + i*s */
            qi[2 * VL + v] = umi + usr;
            qr[4 * VL + v] = umr + usi;    /* x4 = m - i*s */
            qi[4 * VL + v] = umi - usr;
            qr[3 * VL + v] = v0r + tvr;    /* inv DFT3(v) -> x3, x5, x1 */
            qi[3 * VL + v] = v0i + tvi;
            qr[5 * VL + v] = vmr - vsi;
            qi[5 * VL + v] = vmi + vsr;
            qr[VL + v] = vmr + vsi;
            qi[VL + v] = vmi - vsr;
        }
    }
}

/* DFT-5 constants (correctly rounded) */
#define BST_C51 0.30901699437494742410229341718281905886  /* cos(2pi/5) */
#define BST_C52 (-0.80901699437494742410229341718281905886) /* cos(4pi/5) */
#define BST_S51 0.95105651629515357211643933337938214341  /* sin(2pi/5) */
#define BST_S52 0.58778525229247312916870595463907276860  /* sin(4pi/5) */

/* 5-point DFT on one lane: forward (sgn = -i pairing) writes y in place.
 * Shared by the scalar tail stages below; the AVX-512 conv_mid5/10 mirror
 * this exact operation order. */
#define BST_DFT5_LANE(P0r, P0i, P1r, P1i, P2r, P2i, P3r, P3i, P4r, P4i, SGN)      \
    do {                                                                          \
        double t1r = P1r + P4r, t1i = P1i + P4i;                                  \
        double t2r = P2r + P3r, t2i = P2i + P3i;                                  \
        double d1r = P1r - P4r, d1i = P1i - P4i;                                  \
        double d2r = P2r - P3r, d2i = P2i - P3i;                                  \
        double m1r = P0r + BST_C51 * t1r + BST_C52 * t2r;                         \
        double m1i = P0i + BST_C51 * t1i + BST_C52 * t2i;                         \
        double m2r = P0r + BST_C52 * t1r + BST_C51 * t2r;                         \
        double m2i = P0i + BST_C52 * t1i + BST_C51 * t2i;                         \
        double q1r = BST_S51 * d1r + BST_S52 * d2r;                               \
        double q1i = BST_S51 * d1i + BST_S52 * d2i;                               \
        double q2r = BST_S52 * d1r - BST_S51 * d2r;                               \
        double q2i = BST_S52 * d1i - BST_S51 * d2i;                               \
        P0r += t1r + t2r;                                                         \
        P0i += t1i + t2i;                                                         \
        P1r = m1r + (SGN) * q1i; P1i = m1i - (SGN) * q1r;                         \
        P4r = m1r - (SGN) * q1i; P4i = m1i + (SGN) * q1r;                         \
        P2r = m2r + (SGN) * q2i; P2i = m2i - (SGN) * q2r;                         \
        P3r = m2r - (SGN) * q2i; P3i = m2i + (SGN) * q2r;                         \
    } while (0)

static void dft5_fwd_stage(int M, double *restrict wr, double *restrict wi)
{
    for (int blk = 0; blk < M; blk += 5) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        for (int v = 0; v < VL; ++v)
            BST_DFT5_LANE(qr[v], qi[v], qr[VL + v], qi[VL + v],
                          qr[2 * VL + v], qi[2 * VL + v],
                          qr[3 * VL + v], qi[3 * VL + v],
                          qr[4 * VL + v], qi[4 * VL + v], 1.0);
    }
}

static void dft5_inv_stage(int M, double *restrict wr, double *restrict wi)
{
    for (int blk = 0; blk < M; blk += 5) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        for (int v = 0; v < VL; ++v)
            BST_DFT5_LANE(qr[v], qi[v], qr[VL + v], qi[VL + v],
                          qr[2 * VL + v], qi[2 * VL + v],
                          qr[3 * VL + v], qi[3 * VL + v],
                          qr[4 * VL + v], qi[4 * VL + v], -1.0);
    }
}

/* DFT-10 as PFA(2x5): u = DFT5(x0,x2,x4,x6,x8), v = DFT5(x5,x7,x9,x1,x3),
 * slots (u_k + v_k, u_k - v_k) pairwise; the 2-pt is self-inverse. */
static void dft10_fwd_stage(int M, double *restrict wr, double *restrict wi)
{
    for (int blk = 0; blk < M; blk += 10) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        for (int v = 0; v < VL; ++v) {
            double ur[5], ui[5], vr[5], vi[5];
            static const int ue[5] = {0, 2, 4, 6, 8}, vo[5] = {5, 7, 9, 1, 3};
            for (int k = 0; k < 5; ++k) {
                ur[k] = qr[(size_t)ue[k] * VL + v]; ui[k] = qi[(size_t)ue[k] * VL + v];
                vr[k] = qr[(size_t)vo[k] * VL + v]; vi[k] = qi[(size_t)vo[k] * VL + v];
            }
            BST_DFT5_LANE(ur[0], ui[0], ur[1], ui[1], ur[2], ui[2],
                          ur[3], ui[3], ur[4], ui[4], 1.0);
            BST_DFT5_LANE(vr[0], vi[0], vr[1], vi[1], vr[2], vi[2],
                          vr[3], vi[3], vr[4], vi[4], 1.0);
            for (int k = 0; k < 5; ++k) {
                qr[(size_t)(2 * k) * VL + v] = ur[k] + vr[k];
                qi[(size_t)(2 * k) * VL + v] = ui[k] + vi[k];
                qr[(size_t)(2 * k + 1) * VL + v] = ur[k] - vr[k];
                qi[(size_t)(2 * k + 1) * VL + v] = ui[k] - vi[k];
            }
        }
    }
}

static void dft10_inv_stage(int M, double *restrict wr, double *restrict wi)
{
    for (int blk = 0; blk < M; blk += 10) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        for (int v = 0; v < VL; ++v) {
            double ur[5], ui[5], vr[5], vi[5];
            static const int ue[5] = {0, 2, 4, 6, 8}, vo[5] = {5, 7, 9, 1, 3};
            for (int k = 0; k < 5; ++k) {
                double s0r = qr[(size_t)(2 * k) * VL + v], s0i = qi[(size_t)(2 * k) * VL + v];
                double s1r = qr[(size_t)(2 * k + 1) * VL + v], s1i = qi[(size_t)(2 * k + 1) * VL + v];
                ur[k] = s0r + s1r; ui[k] = s0i + s1i;
                vr[k] = s0r - s1r; vi[k] = s0i - s1i;
            }
            BST_DFT5_LANE(ur[0], ui[0], ur[1], ui[1], ur[2], ui[2],
                          ur[3], ui[3], ur[4], ui[4], -1.0);
            BST_DFT5_LANE(vr[0], vi[0], vr[1], vi[1], vr[2], vi[2],
                          vr[3], vi[3], vr[4], vi[4], -1.0);
            for (int k = 0; k < 5; ++k) {
                qr[(size_t)ue[k] * VL + v] = ur[k]; qi[(size_t)ue[k] * VL + v] = ui[k];
                qr[(size_t)vo[k] * VL + v] = vr[k]; qi[(size_t)vo[k] * VL + v] = vi[k];
            }
        }
    }
}

/* DFT-7 constants (correctly rounded; cos/sin(2 pi k / 7)) */
#define BST_C71 0.62348980185873353052500488400423981063   /* cos(2pi/7) */
#define BST_C72 (-0.22252093395631440428890256449679475947) /* cos(4pi/7) */
#define BST_C73 (-0.90096886790241912623610231950744505116) /* cos(6pi/7) */
#define BST_S71 0.78183148246802980870844452667405775023   /* sin(2pi/7) */
#define BST_S72 0.97492791218182360701813168299393121723   /* sin(4pi/7) */
#define BST_S73 0.43388373911755812047576833284835875461   /* sin(6pi/7) */

/* 7-point DFT on one lane, in place, same shape as BST_DFT5_LANE:
 * t_k = x_k + x_{7-k}, d_k = x_k - x_{7-k};  y_j = m_j -+ i q_j with
 * m_j = x0 + sum cos(2pi jk/7) t_k, q_j = sum sin(2pi jk/7) d_k
 * (index reduction jk mod 7 folded into the constant pattern below).
 * SGN = +1 forward, -1 = the exact unnormalized (x7) inverse. */
#define BST_DFT7_LANE(P0r, P0i, P1r, P1i, P2r, P2i, P3r, P3i, P4r, P4i,           \
                      P5r, P5i, P6r, P6i, SGN)                                    \
    do {                                                                          \
        double t1r = P1r + P6r, t1i = P1i + P6i;                                  \
        double d1r = P1r - P6r, d1i = P1i - P6i;                                  \
        double t2r = P2r + P5r, t2i = P2i + P5i;                                  \
        double d2r = P2r - P5r, d2i = P2i - P5i;                                  \
        double t3r = P3r + P4r, t3i = P3i + P4i;                                  \
        double d3r = P3r - P4r, d3i = P3i - P4i;                                  \
        double m1r = P0r + BST_C71 * t1r + BST_C72 * t2r + BST_C73 * t3r;         \
        double m1i = P0i + BST_C71 * t1i + BST_C72 * t2i + BST_C73 * t3i;         \
        double m2r = P0r + BST_C72 * t1r + BST_C73 * t2r + BST_C71 * t3r;         \
        double m2i = P0i + BST_C72 * t1i + BST_C73 * t2i + BST_C71 * t3i;         \
        double m3r = P0r + BST_C73 * t1r + BST_C71 * t2r + BST_C72 * t3r;         \
        double m3i = P0i + BST_C73 * t1i + BST_C71 * t2i + BST_C72 * t3i;         \
        double q1r = BST_S71 * d1r + BST_S72 * d2r + BST_S73 * d3r;               \
        double q1i = BST_S71 * d1i + BST_S72 * d2i + BST_S73 * d3i;               \
        double q2r = BST_S72 * d1r - BST_S73 * d2r - BST_S71 * d3r;               \
        double q2i = BST_S72 * d1i - BST_S73 * d2i - BST_S71 * d3i;               \
        double q3r = BST_S73 * d1r - BST_S71 * d2r + BST_S72 * d3r;               \
        double q3i = BST_S73 * d1i - BST_S71 * d2i + BST_S72 * d3i;               \
        P0r += t1r + t2r + t3r;                                                   \
        P0i += t1i + t2i + t3i;                                                   \
        P1r = m1r + (SGN) * q1i; P1i = m1i - (SGN) * q1r;                         \
        P6r = m1r - (SGN) * q1i; P6i = m1i + (SGN) * q1r;                         \
        P2r = m2r + (SGN) * q2i; P2i = m2i - (SGN) * q2r;                         \
        P5r = m2r - (SGN) * q2i; P5i = m2i + (SGN) * q2r;                         \
        P3r = m3r + (SGN) * q3i; P3i = m3i - (SGN) * q3r;                         \
        P4r = m3r - (SGN) * q3i; P4i = m3i + (SGN) * q3r;                         \
    } while (0)

static void dft7_fwd_stage(int M, double *restrict wr, double *restrict wi)
{
    for (int blk = 0; blk < M; blk += 7) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        for (int v = 0; v < VL; ++v)
            BST_DFT7_LANE(qr[v], qi[v], qr[VL + v], qi[VL + v],
                          qr[2 * VL + v], qi[2 * VL + v],
                          qr[3 * VL + v], qi[3 * VL + v],
                          qr[4 * VL + v], qi[4 * VL + v],
                          qr[5 * VL + v], qi[5 * VL + v],
                          qr[6 * VL + v], qi[6 * VL + v], 1.0);
    }
}

static void dft7_inv_stage(int M, double *restrict wr, double *restrict wi)
{
    for (int blk = 0; blk < M; blk += 7) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        for (int v = 0; v < VL; ++v)
            BST_DFT7_LANE(qr[v], qi[v], qr[VL + v], qi[VL + v],
                          qr[2 * VL + v], qi[2 * VL + v],
                          qr[3 * VL + v], qi[3 * VL + v],
                          qr[4 * VL + v], qi[4 * VL + v],
                          qr[5 * VL + v], qi[5 * VL + v],
                          qr[6 * VL + v], qi[6 * VL + v], -1.0);
    }
}

/* DFT-14 as PFA(2x7): u = DFT7(x0,x2,..,x12), v = DFT7(x7,x9,..,x5), slots
 * (u_k + v_k, u_k - v_k) pairwise; the 2-pt is self-inverse.  Same PFA slot
 * convention as DFT-10 -- bh is computed by this same forward, so the slot
 * permutation never needs to be named. */
static const int bst_ue14[7] = {0, 2, 4, 6, 8, 10, 12};
static const int bst_vo14[7] = {7, 9, 11, 13, 1, 3, 5};

static void dft14_fwd_stage(int M, double *restrict wr, double *restrict wi)
{
    for (int blk = 0; blk < M; blk += 14) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        for (int v = 0; v < VL; ++v) {
            double ur[7], ui[7], vr[7], vi[7];
            for (int k = 0; k < 7; ++k) {
                ur[k] = qr[(size_t)bst_ue14[k] * VL + v];
                ui[k] = qi[(size_t)bst_ue14[k] * VL + v];
                vr[k] = qr[(size_t)bst_vo14[k] * VL + v];
                vi[k] = qi[(size_t)bst_vo14[k] * VL + v];
            }
            BST_DFT7_LANE(ur[0], ui[0], ur[1], ui[1], ur[2], ui[2], ur[3], ui[3],
                          ur[4], ui[4], ur[5], ui[5], ur[6], ui[6], 1.0);
            BST_DFT7_LANE(vr[0], vi[0], vr[1], vi[1], vr[2], vi[2], vr[3], vi[3],
                          vr[4], vi[4], vr[5], vi[5], vr[6], vi[6], 1.0);
            for (int k = 0; k < 7; ++k) {
                qr[(size_t)(2 * k) * VL + v] = ur[k] + vr[k];
                qi[(size_t)(2 * k) * VL + v] = ui[k] + vi[k];
                qr[(size_t)(2 * k + 1) * VL + v] = ur[k] - vr[k];
                qi[(size_t)(2 * k + 1) * VL + v] = ui[k] - vi[k];
            }
        }
    }
}

static void dft14_inv_stage(int M, double *restrict wr, double *restrict wi)
{
    for (int blk = 0; blk < M; blk += 14) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        for (int v = 0; v < VL; ++v) {
            double ur[7], ui[7], vr[7], vi[7];
            for (int k = 0; k < 7; ++k) {
                double s0r = qr[(size_t)(2 * k) * VL + v];
                double s0i = qi[(size_t)(2 * k) * VL + v];
                double s1r = qr[(size_t)(2 * k + 1) * VL + v];
                double s1i = qi[(size_t)(2 * k + 1) * VL + v];
                ur[k] = s0r + s1r; ui[k] = s0i + s1i;
                vr[k] = s0r - s1r; vi[k] = s0i - s1i;
            }
            BST_DFT7_LANE(ur[0], ui[0], ur[1], ui[1], ur[2], ui[2], ur[3], ui[3],
                          ur[4], ui[4], ur[5], ui[5], ur[6], ui[6], -1.0);
            BST_DFT7_LANE(vr[0], vi[0], vr[1], vi[1], vr[2], vi[2], vr[3], vi[3],
                          vr[4], vi[4], vr[5], vi[5], vr[6], vi[6], -1.0);
            for (int k = 0; k < 7; ++k) {
                qr[(size_t)bst_ue14[k] * VL + v] = ur[k];
                qi[(size_t)bst_ue14[k] * VL + v] = ui[k];
                qr[(size_t)bst_vo14[k] * VL + v] = vr[k];
                qi[(size_t)bst_vo14[k] * VL + v] = vi[k];
            }
        }
    }
}

/* DFT-13 constants (correctly rounded; cos/sin(2 pi k / 13), k = 1..6) */
#define BST_C131 0.88545602565320989590037552201509887860   /* cos( 2pi/13) */
#define BST_C132 0.56806474673115580251180755912751662453   /* cos( 4pi/13) */
#define BST_C133 0.12053668025532305334906768745254358227   /* cos( 6pi/13) */
#define BST_C134 (-0.35460488704253562596963789260001847432) /* cos( 8pi/13) */
#define BST_C135 (-0.74851074817110109863463059970135138385) /* cos(10pi/13) */
#define BST_C136 (-0.97094181742605202715698227629378922725) /* cos(12pi/13) */
#define BST_S131 0.46472317204376854565601533513310477756   /* sin( 2pi/13) */
#define BST_S132 0.82298386589365639457961742343938199066   /* sin( 4pi/13) */
#define BST_S133 0.99270887409805399280075164949252017934   /* sin( 6pi/13) */
#define BST_S134 0.93501624268541482343978459983783072905   /* sin( 8pi/13) */
#define BST_S135 0.66312265824079520237678549266676627952   /* sin(10pi/13) */
#define BST_S136 0.23931566428755776714875372626021189520   /* sin(12pi/13) */

/* 13-point DFT on one lane, arrays XR/XI of 13 doubles, in place; same
 * symmetric t/d form as BST_DFT7_LANE extended to six 6-term dot products
 * (index reduction jk mod 13 folded into the constant/sign pattern; row
 * derivation checked against a direct DFT to 1e-14 before shipping).
 * SGN = +1 forward, -1 = the exact unnormalized (x13) inverse. */
#define BST_DFT13_LANE(XR, XI, SGN)                                               \
    do {                                                                          \
        double t1r = XR[1] + XR[12], t1i = XI[1] + XI[12];                        \
        double d1r = XR[1] - XR[12], d1i = XI[1] - XI[12];                        \
        double t2r = XR[2] + XR[11], t2i = XI[2] + XI[11];                        \
        double d2r = XR[2] - XR[11], d2i = XI[2] - XI[11];                        \
        double t3r = XR[3] + XR[10], t3i = XI[3] + XI[10];                        \
        double d3r = XR[3] - XR[10], d3i = XI[3] - XI[10];                        \
        double t4r = XR[4] + XR[9], t4i = XI[4] + XI[9];                          \
        double d4r = XR[4] - XR[9], d4i = XI[4] - XI[9];                          \
        double t5r = XR[5] + XR[8], t5i = XI[5] + XI[8];                          \
        double d5r = XR[5] - XR[8], d5i = XI[5] - XI[8];                          \
        double t6r = XR[6] + XR[7], t6i = XI[6] + XI[7];                          \
        double d6r = XR[6] - XR[7], d6i = XI[6] - XI[7];                          \
        double m1r = XR[0] + BST_C131 * t1r + BST_C132 * t2r + BST_C133 * t3r    \
                           + BST_C134 * t4r + BST_C135 * t5r + BST_C136 * t6r;   \
        double m1i = XI[0] + BST_C131 * t1i + BST_C132 * t2i + BST_C133 * t3i    \
                           + BST_C134 * t4i + BST_C135 * t5i + BST_C136 * t6i;   \
        double m2r = XR[0] + BST_C132 * t1r + BST_C134 * t2r + BST_C136 * t3r    \
                           + BST_C135 * t4r + BST_C133 * t5r + BST_C131 * t6r;   \
        double m2i = XI[0] + BST_C132 * t1i + BST_C134 * t2i + BST_C136 * t3i    \
                           + BST_C135 * t4i + BST_C133 * t5i + BST_C131 * t6i;   \
        double m3r = XR[0] + BST_C133 * t1r + BST_C136 * t2r + BST_C134 * t3r    \
                           + BST_C131 * t4r + BST_C132 * t5r + BST_C135 * t6r;   \
        double m3i = XI[0] + BST_C133 * t1i + BST_C136 * t2i + BST_C134 * t3i    \
                           + BST_C131 * t4i + BST_C132 * t5i + BST_C135 * t6i;   \
        double m4r = XR[0] + BST_C134 * t1r + BST_C135 * t2r + BST_C131 * t3r    \
                           + BST_C133 * t4r + BST_C136 * t5r + BST_C132 * t6r;   \
        double m4i = XI[0] + BST_C134 * t1i + BST_C135 * t2i + BST_C131 * t3i    \
                           + BST_C133 * t4i + BST_C136 * t5i + BST_C132 * t6i;   \
        double m5r = XR[0] + BST_C135 * t1r + BST_C133 * t2r + BST_C132 * t3r    \
                           + BST_C136 * t4r + BST_C131 * t5r + BST_C134 * t6r;   \
        double m5i = XI[0] + BST_C135 * t1i + BST_C133 * t2i + BST_C132 * t3i    \
                           + BST_C136 * t4i + BST_C131 * t5i + BST_C134 * t6i;   \
        double m6r = XR[0] + BST_C136 * t1r + BST_C131 * t2r + BST_C135 * t3r    \
                           + BST_C132 * t4r + BST_C134 * t5r + BST_C133 * t6r;   \
        double m6i = XI[0] + BST_C136 * t1i + BST_C131 * t2i + BST_C135 * t3i    \
                           + BST_C132 * t4i + BST_C134 * t5i + BST_C133 * t6i;   \
        double q1r = BST_S131 * d1r + BST_S132 * d2r + BST_S133 * d3r            \
                   + BST_S134 * d4r + BST_S135 * d5r + BST_S136 * d6r;           \
        double q1i = BST_S131 * d1i + BST_S132 * d2i + BST_S133 * d3i            \
                   + BST_S134 * d4i + BST_S135 * d5i + BST_S136 * d6i;           \
        double q2r = BST_S132 * d1r + BST_S134 * d2r + BST_S136 * d3r            \
                   - BST_S135 * d4r - BST_S133 * d5r - BST_S131 * d6r;           \
        double q2i = BST_S132 * d1i + BST_S134 * d2i + BST_S136 * d3i            \
                   - BST_S135 * d4i - BST_S133 * d5i - BST_S131 * d6i;           \
        double q3r = BST_S133 * d1r + BST_S136 * d2r - BST_S134 * d3r            \
                   - BST_S131 * d4r + BST_S132 * d5r + BST_S135 * d6r;           \
        double q3i = BST_S133 * d1i + BST_S136 * d2i - BST_S134 * d3i            \
                   - BST_S131 * d4i + BST_S132 * d5i + BST_S135 * d6i;           \
        double q4r = BST_S134 * d1r - BST_S135 * d2r - BST_S131 * d3r            \
                   + BST_S133 * d4r - BST_S136 * d5r - BST_S132 * d6r;           \
        double q4i = BST_S134 * d1i - BST_S135 * d2i - BST_S131 * d3i            \
                   + BST_S133 * d4i - BST_S136 * d5i - BST_S132 * d6i;           \
        double q5r = BST_S135 * d1r - BST_S133 * d2r + BST_S132 * d3r            \
                   - BST_S136 * d4r - BST_S131 * d5r + BST_S134 * d6r;           \
        double q5i = BST_S135 * d1i - BST_S133 * d2i + BST_S132 * d3i            \
                   - BST_S136 * d4i - BST_S131 * d5i + BST_S134 * d6i;           \
        double q6r = BST_S136 * d1r - BST_S131 * d2r + BST_S135 * d3r            \
                   - BST_S132 * d4r + BST_S134 * d5r - BST_S133 * d6r;           \
        double q6i = BST_S136 * d1i - BST_S131 * d2i + BST_S135 * d3i            \
                   - BST_S132 * d4i + BST_S134 * d5i - BST_S133 * d6i;           \
        XR[0] += t1r + t2r + t3r + t4r + t5r + t6r;                              \
        XI[0] += t1i + t2i + t3i + t4i + t5i + t6i;                              \
        XR[1] = m1r + (SGN) * q1i; XI[1] = m1i - (SGN) * q1r;                    \
        XR[12] = m1r - (SGN) * q1i; XI[12] = m1i + (SGN) * q1r;                  \
        XR[2] = m2r + (SGN) * q2i; XI[2] = m2i - (SGN) * q2r;                    \
        XR[11] = m2r - (SGN) * q2i; XI[11] = m2i + (SGN) * q2r;                  \
        XR[3] = m3r + (SGN) * q3i; XI[3] = m3i - (SGN) * q3r;                    \
        XR[10] = m3r - (SGN) * q3i; XI[10] = m3i + (SGN) * q3r;                  \
        XR[4] = m4r + (SGN) * q4i; XI[4] = m4i - (SGN) * q4r;                    \
        XR[9] = m4r - (SGN) * q4i; XI[9] = m4i + (SGN) * q4r;                    \
        XR[5] = m5r + (SGN) * q5i; XI[5] = m5i - (SGN) * q5r;                    \
        XR[8] = m5r - (SGN) * q5i; XI[8] = m5i + (SGN) * q5r;                    \
        XR[6] = m6r + (SGN) * q6i; XI[6] = m6i - (SGN) * q6r;                    \
        XR[7] = m6r - (SGN) * q6i; XI[7] = m6i + (SGN) * q6r;                    \
    } while (0)

static void dft13_fwd_stage(int M, double *restrict wr, double *restrict wi)
{
    for (int blk = 0; blk < M; blk += 13) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        for (int v = 0; v < VL; ++v) {
            double xr[13], xi[13];
            for (int k = 0; k < 13; ++k) {
                xr[k] = qr[(size_t)k * VL + v];
                xi[k] = qi[(size_t)k * VL + v];
            }
            BST_DFT13_LANE(xr, xi, 1.0);
            for (int k = 0; k < 13; ++k) {
                qr[(size_t)k * VL + v] = xr[k];
                qi[(size_t)k * VL + v] = xi[k];
            }
        }
    }
}

static void dft13_inv_stage(int M, double *restrict wr, double *restrict wi)
{
    for (int blk = 0; blk < M; blk += 13) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        for (int v = 0; v < VL; ++v) {
            double xr[13], xi[13];
            for (int k = 0; k < 13; ++k) {
                xr[k] = qr[(size_t)k * VL + v];
                xi[k] = qi[(size_t)k * VL + v];
            }
            BST_DFT13_LANE(xr, xi, -1.0);
            for (int k = 0; k < 13; ++k) {
                qr[(size_t)k * VL + v] = xr[k];
                qi[(size_t)k * VL + v] = xi[k];
            }
        }
    }
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

/* fused middles for the 3*2^k convolution sizes: DFT-3 (resp. PFA-6) blocks +
 * pointwise bh multiply + the exact inverse blocks, in registers, one pass.
 * Same slot order as dft3_fwd_stage / dft6_fwd_stage (bh consistency). */
static void conv_mid3(int M, const double *restrict bhre, const double *restrict bhim,
                      double *restrict wr, double *restrict wi)
{
    const __m512d KH = _mm512_set1_pd(0.5), K3 = _mm512_set1_pd(BST_K3);
    for (int blk = 0; blk < M; blk += 3) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        __m512d ar = _mm512_load_pd(qr), ai = _mm512_load_pd(qi);
        __m512d br = _mm512_load_pd(qr + VL), bi = _mm512_load_pd(qi + VL);
        __m512d cr = _mm512_load_pd(qr + 2 * VL), ci = _mm512_load_pd(qi + 2 * VL);
        __m512d tr = _mm512_add_pd(br, cr), ti = _mm512_add_pd(bi, ci);
        __m512d y0r = _mm512_add_pd(ar, tr), y0i = _mm512_add_pd(ai, ti);
        __m512d mr = _mm512_fnmadd_pd(KH, tr, ar), mi = _mm512_fnmadd_pd(KH, ti, ai);
        __m512d sr = _mm512_mul_pd(K3, _mm512_sub_pd(br, cr));
        __m512d si = _mm512_mul_pd(K3, _mm512_sub_pd(bi, ci));
        __m512d y1r = _mm512_add_pd(mr, si), y1i = _mm512_sub_pd(mi, sr);
        __m512d y2r = _mm512_sub_pd(mr, si), y2i = _mm512_add_pd(mi, sr);
        __m512d h0r = _mm512_set1_pd(bhre[blk]), h0i = _mm512_set1_pd(bhim[blk]);
        __m512d h1r = _mm512_set1_pd(bhre[blk + 1]), h1i = _mm512_set1_pd(bhim[blk + 1]);
        __m512d h2r = _mm512_set1_pd(bhre[blk + 2]), h2i = _mm512_set1_pd(bhim[blk + 2]);
        __m512d u0r, u0i, u1r, u1i, u2r, u2i;
        BST_CMUL(u0r, u0i, y0r, y0i, h0r, h0i);
        BST_CMUL(u1r, u1i, y1r, y1i, h1r, h1i);
        BST_CMUL(u2r, u2i, y2r, y2i, h2r, h2i);
        tr = _mm512_add_pd(u1r, u2r);
        ti = _mm512_add_pd(u1i, u2i);
        mr = _mm512_fnmadd_pd(KH, tr, u0r);
        mi = _mm512_fnmadd_pd(KH, ti, u0i);
        sr = _mm512_mul_pd(K3, _mm512_sub_pd(u1r, u2r));
        si = _mm512_mul_pd(K3, _mm512_sub_pd(u1i, u2i));
        _mm512_store_pd(qr, _mm512_add_pd(u0r, tr));
        _mm512_store_pd(qi, _mm512_add_pd(u0i, ti));
        _mm512_store_pd(qr + VL, _mm512_sub_pd(mr, si)); /* x1 = m + i*s */
        _mm512_store_pd(qi + VL, _mm512_add_pd(mi, sr));
        _mm512_store_pd(qr + 2 * VL, _mm512_add_pd(mr, si)); /* x2 = m - i*s */
        _mm512_store_pd(qi + 2 * VL, _mm512_sub_pd(mi, sr));
    }
}

static void conv_mid6(int M, const double *restrict bhre, const double *restrict bhim,
                      double *restrict wr, double *restrict wi)
{
    const __m512d KH = _mm512_set1_pd(0.5), K3 = _mm512_set1_pd(BST_K3);
    for (int blk = 0; blk < M; blk += 6) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        __m512d x0r = _mm512_load_pd(qr), x0i = _mm512_load_pd(qi);
        __m512d x1r = _mm512_load_pd(qr + VL), x1i = _mm512_load_pd(qi + VL);
        __m512d x2r = _mm512_load_pd(qr + 2 * VL), x2i = _mm512_load_pd(qi + 2 * VL);
        __m512d x3r = _mm512_load_pd(qr + 3 * VL), x3i = _mm512_load_pd(qi + 3 * VL);
        __m512d x4r = _mm512_load_pd(qr + 4 * VL), x4i = _mm512_load_pd(qi + 4 * VL);
        __m512d x5r = _mm512_load_pd(qr + 5 * VL), x5i = _mm512_load_pd(qi + 5 * VL);
        /* u = DFT3(x0,x2,x4), v = DFT3(x3,x5,x1) */
        __m512d tur = _mm512_add_pd(x2r, x4r), tui = _mm512_add_pd(x2i, x4i);
        __m512d u0r = _mm512_add_pd(x0r, tur), u0i = _mm512_add_pd(x0i, tui);
        __m512d umr = _mm512_fnmadd_pd(KH, tur, x0r), umi = _mm512_fnmadd_pd(KH, tui, x0i);
        __m512d usr = _mm512_mul_pd(K3, _mm512_sub_pd(x2r, x4r));
        __m512d usi = _mm512_mul_pd(K3, _mm512_sub_pd(x2i, x4i));
        __m512d u1r = _mm512_add_pd(umr, usi), u1i = _mm512_sub_pd(umi, usr);
        __m512d u2r = _mm512_sub_pd(umr, usi), u2i = _mm512_add_pd(umi, usr);
        __m512d tvr = _mm512_add_pd(x5r, x1r), tvi = _mm512_add_pd(x5i, x1i);
        __m512d v0r = _mm512_add_pd(x3r, tvr), v0i = _mm512_add_pd(x3i, tvi);
        __m512d vmr = _mm512_fnmadd_pd(KH, tvr, x3r), vmi = _mm512_fnmadd_pd(KH, tvi, x3i);
        __m512d vsr = _mm512_mul_pd(K3, _mm512_sub_pd(x5r, x1r));
        __m512d vsi = _mm512_mul_pd(K3, _mm512_sub_pd(x5i, x1i));
        __m512d v1r = _mm512_add_pd(vmr, vsi), v1i = _mm512_sub_pd(vmi, vsr);
        __m512d v2r = _mm512_sub_pd(vmr, vsi), v2i = _mm512_add_pd(vmi, vsr);
        /* slots, pointwise, and the self-inverse 2-pt undo, pairwise */
        __m512d s0r, s0i, s1r, s1i;
#define BST_M6_PAIR(uR, uI, vR, vI, h0, h1)                                       \
        do {                                                                      \
            __m512d p0r = _mm512_add_pd(uR, vR), p0i = _mm512_add_pd(uI, vI);     \
            __m512d p1r = _mm512_sub_pd(uR, vR), p1i = _mm512_sub_pd(uI, vI);     \
            __m512d hr0 = _mm512_set1_pd(bhre[blk + (h0)]);                       \
            __m512d hi0 = _mm512_set1_pd(bhim[blk + (h0)]);                       \
            __m512d hr1 = _mm512_set1_pd(bhre[blk + (h1)]);                       \
            __m512d hi1 = _mm512_set1_pd(bhim[blk + (h1)]);                       \
            BST_CMUL(s0r, s0i, p0r, p0i, hr0, hi0);                               \
            BST_CMUL(s1r, s1i, p1r, p1i, hr1, hi1);                               \
            uR = _mm512_add_pd(s0r, s1r);                                         \
            uI = _mm512_add_pd(s0i, s1i);                                         \
            vR = _mm512_sub_pd(s0r, s1r);                                         \
            vI = _mm512_sub_pd(s0i, s1i);                                         \
        } while (0)
        BST_M6_PAIR(u0r, u0i, v0r, v0i, 0, 1);
        BST_M6_PAIR(u1r, u1i, v1r, v1i, 2, 3);
        BST_M6_PAIR(u2r, u2i, v2r, v2i, 4, 5);
#undef BST_M6_PAIR
        /* inverse DFT3(u) -> x0, x2, x4;  inverse DFT3(v) -> x3, x5, x1 */
        tur = _mm512_add_pd(u1r, u2r);
        tui = _mm512_add_pd(u1i, u2i);
        umr = _mm512_fnmadd_pd(KH, tur, u0r);
        umi = _mm512_fnmadd_pd(KH, tui, u0i);
        usr = _mm512_mul_pd(K3, _mm512_sub_pd(u1r, u2r));
        usi = _mm512_mul_pd(K3, _mm512_sub_pd(u1i, u2i));
        _mm512_store_pd(qr, _mm512_add_pd(u0r, tur));
        _mm512_store_pd(qi, _mm512_add_pd(u0i, tui));
        _mm512_store_pd(qr + 2 * VL, _mm512_sub_pd(umr, usi));
        _mm512_store_pd(qi + 2 * VL, _mm512_add_pd(umi, usr));
        _mm512_store_pd(qr + 4 * VL, _mm512_add_pd(umr, usi));
        _mm512_store_pd(qi + 4 * VL, _mm512_sub_pd(umi, usr));
        tvr = _mm512_add_pd(v1r, v2r);
        tvi = _mm512_add_pd(v1i, v2i);
        vmr = _mm512_fnmadd_pd(KH, tvr, v0r);
        vmi = _mm512_fnmadd_pd(KH, tvi, v0i);
        vsr = _mm512_mul_pd(K3, _mm512_sub_pd(v1r, v2r));
        vsi = _mm512_mul_pd(K3, _mm512_sub_pd(v1i, v2i));
        _mm512_store_pd(qr + 3 * VL, _mm512_add_pd(v0r, tvr));
        _mm512_store_pd(qi + 3 * VL, _mm512_add_pd(v0i, tvi));
        _mm512_store_pd(qr + 5 * VL, _mm512_sub_pd(vmr, vsi));
        _mm512_store_pd(qi + 5 * VL, _mm512_add_pd(vmi, vsr));
        _mm512_store_pd(qr + VL, _mm512_add_pd(vmr, vsi));
        _mm512_store_pd(qi + VL, _mm512_sub_pd(vmi, vsr));
    }
}

/* Fused len-12 middle (M = 48, and any BST_NOR16 chain that lands on 12):
 * the forward len-12 radix-4 layer, four DFT-3 blocks, the pointwise bh
 * multiply, the inverse DFT-3s, and the inverse len-12 layer -- one buffer
 * pass instead of three.  tw = the len-12 forward chunk (S = 3), ti = the
 * len-12 inverse chunk (already conjugated).  Arithmetic is op-for-op
 * dif4_stage(12) + conv_mid3 + dit4_stage(12). */
static void conv_mid12(int M, const double *restrict tw, const double *restrict ti,
                       const double *restrict bhre, const double *restrict bhim,
                       double *restrict wr, double *restrict wi)
{
    const __m512d KH = _mm512_set1_pd(0.5), K3 = _mm512_set1_pd(BST_K3);
    for (int blk = 0; blk < M; blk += 12) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        __m512d zr[12], zi[12];
        for (int j = 0; j < 3; ++j) { /* forward radix-4 layer, legs stride 3 */
            const double *t = tw + 6 * j;
            __m512d ar = _mm512_load_pd(qr + (size_t)j * VL);
            __m512d ai = _mm512_load_pd(qi + (size_t)j * VL);
            __m512d br = _mm512_load_pd(qr + (size_t)(j + 3) * VL);
            __m512d bi = _mm512_load_pd(qi + (size_t)(j + 3) * VL);
            __m512d cr = _mm512_load_pd(qr + (size_t)(j + 6) * VL);
            __m512d ci = _mm512_load_pd(qi + (size_t)(j + 6) * VL);
            __m512d dr = _mm512_load_pd(qr + (size_t)(j + 9) * VL);
            __m512d di = _mm512_load_pd(qi + (size_t)(j + 9) * VL);
            __m512d t0r = _mm512_add_pd(ar, cr), t0i = _mm512_add_pd(ai, ci);
            __m512d t1r = _mm512_sub_pd(ar, cr), t1i = _mm512_sub_pd(ai, ci);
            __m512d t2r = _mm512_add_pd(br, dr), t2i = _mm512_add_pd(bi, di);
            __m512d t3r = _mm512_sub_pd(br, dr), t3i = _mm512_sub_pd(bi, di);
            zr[j] = _mm512_add_pd(t0r, t2r);
            zi[j] = _mm512_add_pd(t0i, t2i);
            __m512d w1r = _mm512_set1_pd(t[0]), w1i = _mm512_set1_pd(t[1]);
            __m512d w2r = _mm512_set1_pd(t[2]), w2i = _mm512_set1_pd(t[3]);
            __m512d w3r = _mm512_set1_pd(t[4]), w3i = _mm512_set1_pd(t[5]);
            __m512d u2r = _mm512_sub_pd(t0r, t2r), u2i = _mm512_sub_pd(t0i, t2i);
            BST_CMUL(zr[j + 6], zi[j + 6], u2r, u2i, w2r, w2i);
            __m512d u1r = _mm512_add_pd(t1r, t3i), u1i = _mm512_sub_pd(t1i, t3r);
            BST_CMUL(zr[j + 3], zi[j + 3], u1r, u1i, w1r, w1i);
            __m512d u3r = _mm512_sub_pd(t1r, t3i), u3i = _mm512_add_pd(t1i, t3r);
            BST_CMUL(zr[j + 9], zi[j + 9], u3r, u3i, w3r, w3i);
        }
        for (int b3 = 0; b3 < 12; b3 += 3) { /* DFT-3, pointwise, inverse DFT-3 */
            __m512d tr = _mm512_add_pd(zr[b3 + 1], zr[b3 + 2]);
            __m512d tvi = _mm512_add_pd(zi[b3 + 1], zi[b3 + 2]);
            __m512d y0r = _mm512_add_pd(zr[b3], tr), y0i = _mm512_add_pd(zi[b3], tvi);
            __m512d mr = _mm512_fnmadd_pd(KH, tr, zr[b3]);
            __m512d mi = _mm512_fnmadd_pd(KH, tvi, zi[b3]);
            __m512d sr = _mm512_mul_pd(K3, _mm512_sub_pd(zr[b3 + 1], zr[b3 + 2]));
            __m512d si = _mm512_mul_pd(K3, _mm512_sub_pd(zi[b3 + 1], zi[b3 + 2]));
            __m512d y1r = _mm512_add_pd(mr, si), y1i = _mm512_sub_pd(mi, sr);
            __m512d y2r = _mm512_sub_pd(mr, si), y2i = _mm512_add_pd(mi, sr);
            __m512d h0r = _mm512_set1_pd(bhre[blk + b3]), h0i = _mm512_set1_pd(bhim[blk + b3]);
            __m512d h1r = _mm512_set1_pd(bhre[blk + b3 + 1]), h1i = _mm512_set1_pd(bhim[blk + b3 + 1]);
            __m512d h2r = _mm512_set1_pd(bhre[blk + b3 + 2]), h2i = _mm512_set1_pd(bhim[blk + b3 + 2]);
            __m512d u0r, u0i, u1r, u1i, u2r, u2i;
            BST_CMUL(u0r, u0i, y0r, y0i, h0r, h0i);
            BST_CMUL(u1r, u1i, y1r, y1i, h1r, h1i);
            BST_CMUL(u2r, u2i, y2r, y2i, h2r, h2i);
            tr = _mm512_add_pd(u1r, u2r);
            tvi = _mm512_add_pd(u1i, u2i);
            mr = _mm512_fnmadd_pd(KH, tr, u0r);
            mi = _mm512_fnmadd_pd(KH, tvi, u0i);
            sr = _mm512_mul_pd(K3, _mm512_sub_pd(u1r, u2r));
            si = _mm512_mul_pd(K3, _mm512_sub_pd(u1i, u2i));
            zr[b3] = _mm512_add_pd(u0r, tr);
            zi[b3] = _mm512_add_pd(u0i, tvi);
            zr[b3 + 1] = _mm512_sub_pd(mr, si);
            zi[b3 + 1] = _mm512_add_pd(mi, sr);
            zr[b3 + 2] = _mm512_add_pd(mr, si);
            zi[b3 + 2] = _mm512_sub_pd(mi, sr);
        }
        for (int j = 0; j < 3; ++j) { /* inverse radix-4 layer (dit4 mirror) */
            const double *t = ti + 6 * j;
            __m512d w1r = _mm512_set1_pd(t[0]), w1i = _mm512_set1_pd(t[1]);
            __m512d w2r = _mm512_set1_pd(t[2]), w2i = _mm512_set1_pd(t[3]);
            __m512d w3r = _mm512_set1_pd(t[4]), w3i = _mm512_set1_pd(t[5]);
            __m512d u0r = zr[j], u0i = zi[j];
            __m512d u1r, u1i, u2r, u2i, u3r, u3i;
            BST_CMUL(u1r, u1i, zr[j + 3], zi[j + 3], w1r, w1i);
            BST_CMUL(u2r, u2i, zr[j + 6], zi[j + 6], w2r, w2i);
            BST_CMUL(u3r, u3i, zr[j + 9], zi[j + 9], w3r, w3i);
            __m512d s0r = _mm512_add_pd(u0r, u2r), s0i = _mm512_add_pd(u0i, u2i);
            __m512d s1r = _mm512_sub_pd(u0r, u2r), s1i = _mm512_sub_pd(u0i, u2i);
            __m512d s2r = _mm512_add_pd(u1r, u3r), s2i = _mm512_add_pd(u1i, u3i);
            __m512d s3r = _mm512_sub_pd(u1r, u3r), s3i = _mm512_sub_pd(u1i, u3i);
            _mm512_store_pd(qr + (size_t)j * VL, _mm512_add_pd(s0r, s2r));
            _mm512_store_pd(qi + (size_t)j * VL, _mm512_add_pd(s0i, s2i));
            _mm512_store_pd(qr + (size_t)(j + 6) * VL, _mm512_sub_pd(s0r, s2r));
            _mm512_store_pd(qi + (size_t)(j + 6) * VL, _mm512_sub_pd(s0i, s2i));
            _mm512_store_pd(qr + (size_t)(j + 3) * VL, _mm512_sub_pd(s1r, s3i));
            _mm512_store_pd(qi + (size_t)(j + 3) * VL, _mm512_add_pd(s1i, s3r));
            _mm512_store_pd(qr + (size_t)(j + 9) * VL, _mm512_add_pd(s1r, s3i));
            _mm512_store_pd(qi + (size_t)(j + 9) * VL, _mm512_sub_pd(s1i, s3r));
        }
    }
}

/* vector 5-point DFT on five (r,i) zmm pairs, in place.  Expects __m512d
 * locals vC51, vC52, vS51, vS52 in scope.  A/S = add/sub for the forward,
 * sub/add for the (unnormalized x5) inverse. */
#define BST_DFT5V(P0r, P0i, P1r, P1i, P2r, P2i, P3r, P3i, P4r, P4i, A, S)         \
    do {                                                                          \
        __m512d t1r = _mm512_add_pd(P1r, P4r), t1i = _mm512_add_pd(P1i, P4i);     \
        __m512d t2r = _mm512_add_pd(P2r, P3r), t2i = _mm512_add_pd(P2i, P3i);     \
        __m512d d1r = _mm512_sub_pd(P1r, P4r), d1i = _mm512_sub_pd(P1i, P4i);     \
        __m512d d2r = _mm512_sub_pd(P2r, P3r), d2i = _mm512_sub_pd(P2i, P3i);     \
        __m512d m1r = _mm512_fmadd_pd(vC52, t2r, _mm512_fmadd_pd(vC51, t1r, P0r));\
        __m512d m1i = _mm512_fmadd_pd(vC52, t2i, _mm512_fmadd_pd(vC51, t1i, P0i));\
        __m512d m2r = _mm512_fmadd_pd(vC51, t2r, _mm512_fmadd_pd(vC52, t1r, P0r));\
        __m512d m2i = _mm512_fmadd_pd(vC51, t2i, _mm512_fmadd_pd(vC52, t1i, P0i));\
        __m512d q1r = _mm512_fmadd_pd(vS52, d2r, _mm512_mul_pd(vS51, d1r));       \
        __m512d q1i = _mm512_fmadd_pd(vS52, d2i, _mm512_mul_pd(vS51, d1i));       \
        __m512d q2r = _mm512_fnmadd_pd(vS51, d2r, _mm512_mul_pd(vS52, d1r));      \
        __m512d q2i = _mm512_fnmadd_pd(vS51, d2i, _mm512_mul_pd(vS52, d1i));      \
        P0r = _mm512_add_pd(P0r, _mm512_add_pd(t1r, t2r));                        \
        P0i = _mm512_add_pd(P0i, _mm512_add_pd(t1i, t2i));                        \
        P1r = A(m1r, q1i); P1i = S(m1i, q1r);                                     \
        P4r = S(m1r, q1i); P4i = A(m1i, q1r);                                     \
        P2r = A(m2r, q2i); P2i = S(m2i, q2r);                                     \
        P3r = S(m2r, q2i); P3i = A(m2i, q2r);                                     \
    } while (0)

/* fused middles for the 5*2^k convolution sizes (M = 80, 160, 320, ...):
 * DFT-5 (resp. PFA-10) blocks + pointwise bh + exact inverse, one pass.
 * Same slot order as dft5_fwd_stage / dft10_fwd_stage. */
static void conv_mid5(int M, const double *restrict bhre, const double *restrict bhim,
                      double *restrict wr, double *restrict wi)
{
    const __m512d vC51 = _mm512_set1_pd(BST_C51), vC52 = _mm512_set1_pd(BST_C52);
    const __m512d vS51 = _mm512_set1_pd(BST_S51), vS52 = _mm512_set1_pd(BST_S52);
    for (int blk = 0; blk < M; blk += 5) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        __m512d zr[5], zi[5];
        for (int k = 0; k < 5; ++k) {
            zr[k] = _mm512_load_pd(qr + (size_t)k * VL);
            zi[k] = _mm512_load_pd(qi + (size_t)k * VL);
        }
        BST_DFT5V(zr[0], zi[0], zr[1], zi[1], zr[2], zi[2], zr[3], zi[3],
                  zr[4], zi[4], _mm512_add_pd, _mm512_sub_pd);
        for (int k = 0; k < 5; ++k) {
            __m512d hr = _mm512_set1_pd(bhre[blk + k]);
            __m512d hi = _mm512_set1_pd(bhim[blk + k]);
            __m512d ur, ui;
            BST_CMUL(ur, ui, zr[k], zi[k], hr, hi);
            zr[k] = ur;
            zi[k] = ui;
        }
        BST_DFT5V(zr[0], zi[0], zr[1], zi[1], zr[2], zi[2], zr[3], zi[3],
                  zr[4], zi[4], _mm512_sub_pd, _mm512_add_pd);
        for (int k = 0; k < 5; ++k) {
            _mm512_store_pd(qr + (size_t)k * VL, zr[k]);
            _mm512_store_pd(qi + (size_t)k * VL, zi[k]);
        }
    }
}

static void conv_mid10(int M, const double *restrict bhre, const double *restrict bhim,
                       double *restrict wr, double *restrict wi)
{
    const __m512d vC51 = _mm512_set1_pd(BST_C51), vC52 = _mm512_set1_pd(BST_C52);
    const __m512d vS51 = _mm512_set1_pd(BST_S51), vS52 = _mm512_set1_pd(BST_S52);
    static const int ue[5] = {0, 2, 4, 6, 8}, vo[5] = {5, 7, 9, 1, 3};
    for (int blk = 0; blk < M; blk += 10) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        __m512d ur[5], ui[5], vr[5], vi[5];
        for (int k = 0; k < 5; ++k) {
            ur[k] = _mm512_load_pd(qr + (size_t)ue[k] * VL);
            ui[k] = _mm512_load_pd(qi + (size_t)ue[k] * VL);
            vr[k] = _mm512_load_pd(qr + (size_t)vo[k] * VL);
            vi[k] = _mm512_load_pd(qi + (size_t)vo[k] * VL);
        }
        BST_DFT5V(ur[0], ui[0], ur[1], ui[1], ur[2], ui[2], ur[3], ui[3],
                  ur[4], ui[4], _mm512_add_pd, _mm512_sub_pd);
        BST_DFT5V(vr[0], vi[0], vr[1], vi[1], vr[2], vi[2], vr[3], vi[3],
                  vr[4], vi[4], _mm512_add_pd, _mm512_sub_pd);
        for (int k = 0; k < 5; ++k) { /* slots, pointwise, 2-pt undo */
            __m512d p0r = _mm512_add_pd(ur[k], vr[k]), p0i = _mm512_add_pd(ui[k], vi[k]);
            __m512d p1r = _mm512_sub_pd(ur[k], vr[k]), p1i = _mm512_sub_pd(ui[k], vi[k]);
            __m512d hr0 = _mm512_set1_pd(bhre[blk + 2 * k]);
            __m512d hi0 = _mm512_set1_pd(bhim[blk + 2 * k]);
            __m512d hr1 = _mm512_set1_pd(bhre[blk + 2 * k + 1]);
            __m512d hi1 = _mm512_set1_pd(bhim[blk + 2 * k + 1]);
            __m512d s0r, s0i, s1r, s1i;
            BST_CMUL(s0r, s0i, p0r, p0i, hr0, hi0);
            BST_CMUL(s1r, s1i, p1r, p1i, hr1, hi1);
            ur[k] = _mm512_add_pd(s0r, s1r);
            ui[k] = _mm512_add_pd(s0i, s1i);
            vr[k] = _mm512_sub_pd(s0r, s1r);
            vi[k] = _mm512_sub_pd(s0i, s1i);
        }
        BST_DFT5V(ur[0], ui[0], ur[1], ui[1], ur[2], ui[2], ur[3], ui[3],
                  ur[4], ui[4], _mm512_sub_pd, _mm512_add_pd);
        BST_DFT5V(vr[0], vi[0], vr[1], vi[1], vr[2], vi[2], vr[3], vi[3],
                  vr[4], vi[4], _mm512_sub_pd, _mm512_add_pd);
        for (int k = 0; k < 5; ++k) {
            _mm512_store_pd(qr + (size_t)ue[k] * VL, ur[k]);
            _mm512_store_pd(qi + (size_t)ue[k] * VL, ui[k]);
            _mm512_store_pd(qr + (size_t)vo[k] * VL, vr[k]);
            _mm512_store_pd(qi + (size_t)vo[k] * VL, vi[k]);
        }
    }
}

/* vector 7-point DFT on seven (r,i) zmm pairs, in place; same operation
 * order as BST_DFT7_LANE with the 3-term dot products as FMA chains.
 * Expects __m512d locals vC71..vC73, vS71..vS73 in scope.  A/S = add/sub
 * for the forward, sub/add for the (unnormalized x7) inverse. */
#define BST_DFT7V(P0r, P0i, P1r, P1i, P2r, P2i, P3r, P3i, P4r, P4i,               \
                  P5r, P5i, P6r, P6i, A, S)                                       \
    do {                                                                          \
        __m512d t1r = _mm512_add_pd(P1r, P6r), t1i = _mm512_add_pd(P1i, P6i);     \
        __m512d d1r = _mm512_sub_pd(P1r, P6r), d1i = _mm512_sub_pd(P1i, P6i);     \
        __m512d t2r = _mm512_add_pd(P2r, P5r), t2i = _mm512_add_pd(P2i, P5i);     \
        __m512d d2r = _mm512_sub_pd(P2r, P5r), d2i = _mm512_sub_pd(P2i, P5i);     \
        __m512d t3r = _mm512_add_pd(P3r, P4r), t3i = _mm512_add_pd(P3i, P4i);     \
        __m512d d3r = _mm512_sub_pd(P3r, P4r), d3i = _mm512_sub_pd(P3i, P4i);     \
        __m512d m1r = _mm512_fmadd_pd(vC73, t3r,                                  \
            _mm512_fmadd_pd(vC72, t2r, _mm512_fmadd_pd(vC71, t1r, P0r)));         \
        __m512d m1i = _mm512_fmadd_pd(vC73, t3i,                                  \
            _mm512_fmadd_pd(vC72, t2i, _mm512_fmadd_pd(vC71, t1i, P0i)));         \
        __m512d m2r = _mm512_fmadd_pd(vC71, t3r,                                  \
            _mm512_fmadd_pd(vC73, t2r, _mm512_fmadd_pd(vC72, t1r, P0r)));         \
        __m512d m2i = _mm512_fmadd_pd(vC71, t3i,                                  \
            _mm512_fmadd_pd(vC73, t2i, _mm512_fmadd_pd(vC72, t1i, P0i)));         \
        __m512d m3r = _mm512_fmadd_pd(vC72, t3r,                                  \
            _mm512_fmadd_pd(vC71, t2r, _mm512_fmadd_pd(vC73, t1r, P0r)));         \
        __m512d m3i = _mm512_fmadd_pd(vC72, t3i,                                  \
            _mm512_fmadd_pd(vC71, t2i, _mm512_fmadd_pd(vC73, t1i, P0i)));         \
        __m512d q1r = _mm512_fmadd_pd(vS73, d3r,                                  \
            _mm512_fmadd_pd(vS72, d2r, _mm512_mul_pd(vS71, d1r)));                \
        __m512d q1i = _mm512_fmadd_pd(vS73, d3i,                                  \
            _mm512_fmadd_pd(vS72, d2i, _mm512_mul_pd(vS71, d1i)));                \
        __m512d q2r = _mm512_fnmadd_pd(vS71, d3r,                                 \
            _mm512_fnmadd_pd(vS73, d2r, _mm512_mul_pd(vS72, d1r)));               \
        __m512d q2i = _mm512_fnmadd_pd(vS71, d3i,                                 \
            _mm512_fnmadd_pd(vS73, d2i, _mm512_mul_pd(vS72, d1i)));               \
        __m512d q3r = _mm512_fmadd_pd(vS72, d3r,                                  \
            _mm512_fnmadd_pd(vS71, d2r, _mm512_mul_pd(vS73, d1r)));               \
        __m512d q3i = _mm512_fmadd_pd(vS72, d3i,                                  \
            _mm512_fnmadd_pd(vS71, d2i, _mm512_mul_pd(vS73, d1i)));               \
        P0r = _mm512_add_pd(P0r, _mm512_add_pd(t1r, _mm512_add_pd(t2r, t3r)));    \
        P0i = _mm512_add_pd(P0i, _mm512_add_pd(t1i, _mm512_add_pd(t2i, t3i)));    \
        P1r = A(m1r, q1i); P1i = S(m1i, q1r);                                     \
        P6r = S(m1r, q1i); P6i = A(m1i, q1r);                                     \
        P2r = A(m2r, q2i); P2i = S(m2i, q2r);                                     \
        P5r = S(m2r, q2i); P5i = A(m2i, q2r);                                     \
        P3r = A(m3r, q3i); P3i = S(m3i, q3r);                                     \
        P4r = S(m3r, q3i); P4i = A(m3i, q3r);                                     \
    } while (0)

#define BST_DFT7V_ARR(zr, zi, A, S)                                               \
    BST_DFT7V(zr[0], zi[0], zr[1], zi[1], zr[2], zi[2], zr[3], zi[3],             \
              zr[4], zi[4], zr[5], zi[5], zr[6], zi[6], A, S)

/* fused middles for the 7*2^k convolution sizes (M = 112, 224, 448, ...):
 * DFT-7 (resp. PFA-14) blocks + pointwise bh + exact inverse, one pass.
 * Same slot order as dft7_fwd_stage / dft14_fwd_stage. */
static void conv_mid7(int M, const double *restrict bhre, const double *restrict bhim,
                      double *restrict wr, double *restrict wi)
{
    const __m512d vC71 = _mm512_set1_pd(BST_C71), vC72 = _mm512_set1_pd(BST_C72);
    const __m512d vC73 = _mm512_set1_pd(BST_C73);
    const __m512d vS71 = _mm512_set1_pd(BST_S71), vS72 = _mm512_set1_pd(BST_S72);
    const __m512d vS73 = _mm512_set1_pd(BST_S73);
    for (int blk = 0; blk < M; blk += 7) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        __m512d zr[7], zi[7];
        for (int k = 0; k < 7; ++k) {
            zr[k] = _mm512_load_pd(qr + (size_t)k * VL);
            zi[k] = _mm512_load_pd(qi + (size_t)k * VL);
        }
        BST_DFT7V_ARR(zr, zi, _mm512_add_pd, _mm512_sub_pd);
        for (int k = 0; k < 7; ++k) {
            __m512d hr = _mm512_set1_pd(bhre[blk + k]);
            __m512d hi = _mm512_set1_pd(bhim[blk + k]);
            __m512d ur, ui;
            BST_CMUL(ur, ui, zr[k], zi[k], hr, hi);
            zr[k] = ur;
            zi[k] = ui;
        }
        BST_DFT7V_ARR(zr, zi, _mm512_sub_pd, _mm512_add_pd);
        for (int k = 0; k < 7; ++k) {
            _mm512_store_pd(qr + (size_t)k * VL, zr[k]);
            _mm512_store_pd(qi + (size_t)k * VL, zi[k]);
        }
    }
}

static void conv_mid14(int M, const double *restrict bhre, const double *restrict bhim,
                       double *restrict wr, double *restrict wi)
{
    const __m512d vC71 = _mm512_set1_pd(BST_C71), vC72 = _mm512_set1_pd(BST_C72);
    const __m512d vC73 = _mm512_set1_pd(BST_C73);
    const __m512d vS71 = _mm512_set1_pd(BST_S71), vS72 = _mm512_set1_pd(BST_S72);
    const __m512d vS73 = _mm512_set1_pd(BST_S73);
    for (int blk = 0; blk < M; blk += 14) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        __m512d ur[7], ui[7], vr[7], vi[7];
        for (int k = 0; k < 7; ++k) {
            ur[k] = _mm512_load_pd(qr + (size_t)bst_ue14[k] * VL);
            ui[k] = _mm512_load_pd(qi + (size_t)bst_ue14[k] * VL);
            vr[k] = _mm512_load_pd(qr + (size_t)bst_vo14[k] * VL);
            vi[k] = _mm512_load_pd(qi + (size_t)bst_vo14[k] * VL);
        }
        BST_DFT7V_ARR(ur, ui, _mm512_add_pd, _mm512_sub_pd);
        BST_DFT7V_ARR(vr, vi, _mm512_add_pd, _mm512_sub_pd);
        for (int k = 0; k < 7; ++k) { /* slots, pointwise, 2-pt undo */
            __m512d p0r = _mm512_add_pd(ur[k], vr[k]), p0i = _mm512_add_pd(ui[k], vi[k]);
            __m512d p1r = _mm512_sub_pd(ur[k], vr[k]), p1i = _mm512_sub_pd(ui[k], vi[k]);
            __m512d hr0 = _mm512_set1_pd(bhre[blk + 2 * k]);
            __m512d hi0 = _mm512_set1_pd(bhim[blk + 2 * k]);
            __m512d hr1 = _mm512_set1_pd(bhre[blk + 2 * k + 1]);
            __m512d hi1 = _mm512_set1_pd(bhim[blk + 2 * k + 1]);
            __m512d s0r, s0i, s1r, s1i;
            BST_CMUL(s0r, s0i, p0r, p0i, hr0, hi0);
            BST_CMUL(s1r, s1i, p1r, p1i, hr1, hi1);
            ur[k] = _mm512_add_pd(s0r, s1r);
            ui[k] = _mm512_add_pd(s0i, s1i);
            vr[k] = _mm512_sub_pd(s0r, s1r);
            vi[k] = _mm512_sub_pd(s0i, s1i);
        }
        BST_DFT7V_ARR(ur, ui, _mm512_sub_pd, _mm512_add_pd);
        BST_DFT7V_ARR(vr, vi, _mm512_sub_pd, _mm512_add_pd);
        for (int k = 0; k < 7; ++k) {
            _mm512_store_pd(qr + (size_t)bst_ue14[k] * VL, ur[k]);
            _mm512_store_pd(qi + (size_t)bst_ue14[k] * VL, ui[k]);
            _mm512_store_pd(qr + (size_t)bst_vo14[k] * VL, vr[k]);
            _mm512_store_pd(qi + (size_t)bst_vo14[k] * VL, vi[k]);
        }
    }
}

/* vector 13-point DFT on arrays of 13 (r,i) zmm pairs, in place; same
 * operation order as BST_DFT13_LANE with the 6-term dot products as FMA
 * chains (24 independent chains per block cover the 6-deep latency).
 * Expects __m512d locals vC131..vC136, vS131..vS136 in scope. */
#define BST_DFT13V_ARR(ZR, ZI, A, S)                                              \
    do {                                                                          \
        __m512d t1r = _mm512_add_pd(ZR[1], ZR[12]), t1i = _mm512_add_pd(ZI[1], ZI[12]); \
        __m512d d1r = _mm512_sub_pd(ZR[1], ZR[12]), d1i = _mm512_sub_pd(ZI[1], ZI[12]); \
        __m512d t2r = _mm512_add_pd(ZR[2], ZR[11]), t2i = _mm512_add_pd(ZI[2], ZI[11]); \
        __m512d d2r = _mm512_sub_pd(ZR[2], ZR[11]), d2i = _mm512_sub_pd(ZI[2], ZI[11]); \
        __m512d t3r = _mm512_add_pd(ZR[3], ZR[10]), t3i = _mm512_add_pd(ZI[3], ZI[10]); \
        __m512d d3r = _mm512_sub_pd(ZR[3], ZR[10]), d3i = _mm512_sub_pd(ZI[3], ZI[10]); \
        __m512d t4r = _mm512_add_pd(ZR[4], ZR[9]), t4i = _mm512_add_pd(ZI[4], ZI[9]);   \
        __m512d d4r = _mm512_sub_pd(ZR[4], ZR[9]), d4i = _mm512_sub_pd(ZI[4], ZI[9]);   \
        __m512d t5r = _mm512_add_pd(ZR[5], ZR[8]), t5i = _mm512_add_pd(ZI[5], ZI[8]);   \
        __m512d d5r = _mm512_sub_pd(ZR[5], ZR[8]), d5i = _mm512_sub_pd(ZI[5], ZI[8]);   \
        __m512d t6r = _mm512_add_pd(ZR[6], ZR[7]), t6i = _mm512_add_pd(ZI[6], ZI[7]);   \
        __m512d d6r = _mm512_sub_pd(ZR[6], ZR[7]), d6i = _mm512_sub_pd(ZI[6], ZI[7]);   \
        __m512d m1r = _mm512_fmadd_pd(vC136, t6r, _mm512_fmadd_pd(vC135, t5r,     \
            _mm512_fmadd_pd(vC134, t4r, _mm512_fmadd_pd(vC133, t3r,               \
            _mm512_fmadd_pd(vC132, t2r, _mm512_fmadd_pd(vC131, t1r, ZR[0]))))));  \
        __m512d m1i = _mm512_fmadd_pd(vC136, t6i, _mm512_fmadd_pd(vC135, t5i,     \
            _mm512_fmadd_pd(vC134, t4i, _mm512_fmadd_pd(vC133, t3i,               \
            _mm512_fmadd_pd(vC132, t2i, _mm512_fmadd_pd(vC131, t1i, ZI[0]))))));  \
        __m512d m2r = _mm512_fmadd_pd(vC131, t6r, _mm512_fmadd_pd(vC133, t5r,     \
            _mm512_fmadd_pd(vC135, t4r, _mm512_fmadd_pd(vC136, t3r,               \
            _mm512_fmadd_pd(vC134, t2r, _mm512_fmadd_pd(vC132, t1r, ZR[0]))))));  \
        __m512d m2i = _mm512_fmadd_pd(vC131, t6i, _mm512_fmadd_pd(vC133, t5i,     \
            _mm512_fmadd_pd(vC135, t4i, _mm512_fmadd_pd(vC136, t3i,               \
            _mm512_fmadd_pd(vC134, t2i, _mm512_fmadd_pd(vC132, t1i, ZI[0]))))));  \
        __m512d m3r = _mm512_fmadd_pd(vC135, t6r, _mm512_fmadd_pd(vC132, t5r,     \
            _mm512_fmadd_pd(vC131, t4r, _mm512_fmadd_pd(vC134, t3r,               \
            _mm512_fmadd_pd(vC136, t2r, _mm512_fmadd_pd(vC133, t1r, ZR[0]))))));  \
        __m512d m3i = _mm512_fmadd_pd(vC135, t6i, _mm512_fmadd_pd(vC132, t5i,     \
            _mm512_fmadd_pd(vC131, t4i, _mm512_fmadd_pd(vC134, t3i,               \
            _mm512_fmadd_pd(vC136, t2i, _mm512_fmadd_pd(vC133, t1i, ZI[0]))))));  \
        __m512d m4r = _mm512_fmadd_pd(vC132, t6r, _mm512_fmadd_pd(vC136, t5r,     \
            _mm512_fmadd_pd(vC133, t4r, _mm512_fmadd_pd(vC131, t3r,               \
            _mm512_fmadd_pd(vC135, t2r, _mm512_fmadd_pd(vC134, t1r, ZR[0]))))));  \
        __m512d m4i = _mm512_fmadd_pd(vC132, t6i, _mm512_fmadd_pd(vC136, t5i,     \
            _mm512_fmadd_pd(vC133, t4i, _mm512_fmadd_pd(vC131, t3i,               \
            _mm512_fmadd_pd(vC135, t2i, _mm512_fmadd_pd(vC134, t1i, ZI[0]))))));  \
        __m512d m5r = _mm512_fmadd_pd(vC134, t6r, _mm512_fmadd_pd(vC131, t5r,     \
            _mm512_fmadd_pd(vC136, t4r, _mm512_fmadd_pd(vC132, t3r,               \
            _mm512_fmadd_pd(vC133, t2r, _mm512_fmadd_pd(vC135, t1r, ZR[0]))))));  \
        __m512d m5i = _mm512_fmadd_pd(vC134, t6i, _mm512_fmadd_pd(vC131, t5i,     \
            _mm512_fmadd_pd(vC136, t4i, _mm512_fmadd_pd(vC132, t3i,               \
            _mm512_fmadd_pd(vC133, t2i, _mm512_fmadd_pd(vC135, t1i, ZI[0]))))));  \
        __m512d m6r = _mm512_fmadd_pd(vC133, t6r, _mm512_fmadd_pd(vC134, t5r,     \
            _mm512_fmadd_pd(vC132, t4r, _mm512_fmadd_pd(vC135, t3r,               \
            _mm512_fmadd_pd(vC131, t2r, _mm512_fmadd_pd(vC136, t1r, ZR[0]))))));  \
        __m512d m6i = _mm512_fmadd_pd(vC133, t6i, _mm512_fmadd_pd(vC134, t5i,     \
            _mm512_fmadd_pd(vC132, t4i, _mm512_fmadd_pd(vC135, t3i,               \
            _mm512_fmadd_pd(vC131, t2i, _mm512_fmadd_pd(vC136, t1i, ZI[0]))))));  \
        __m512d q1r = _mm512_fmadd_pd(vS136, d6r, _mm512_fmadd_pd(vS135, d5r,     \
            _mm512_fmadd_pd(vS134, d4r, _mm512_fmadd_pd(vS133, d3r,               \
            _mm512_fmadd_pd(vS132, d2r, _mm512_mul_pd(vS131, d1r))))));           \
        __m512d q1i = _mm512_fmadd_pd(vS136, d6i, _mm512_fmadd_pd(vS135, d5i,     \
            _mm512_fmadd_pd(vS134, d4i, _mm512_fmadd_pd(vS133, d3i,               \
            _mm512_fmadd_pd(vS132, d2i, _mm512_mul_pd(vS131, d1i))))));           \
        __m512d q2r = _mm512_fnmadd_pd(vS131, d6r, _mm512_fnmadd_pd(vS133, d5r,   \
            _mm512_fnmadd_pd(vS135, d4r, _mm512_fmadd_pd(vS136, d3r,              \
            _mm512_fmadd_pd(vS134, d2r, _mm512_mul_pd(vS132, d1r))))));           \
        __m512d q2i = _mm512_fnmadd_pd(vS131, d6i, _mm512_fnmadd_pd(vS133, d5i,   \
            _mm512_fnmadd_pd(vS135, d4i, _mm512_fmadd_pd(vS136, d3i,              \
            _mm512_fmadd_pd(vS134, d2i, _mm512_mul_pd(vS132, d1i))))));           \
        __m512d q3r = _mm512_fmadd_pd(vS135, d6r, _mm512_fmadd_pd(vS132, d5r,     \
            _mm512_fnmadd_pd(vS131, d4r, _mm512_fnmadd_pd(vS134, d3r,             \
            _mm512_fmadd_pd(vS136, d2r, _mm512_mul_pd(vS133, d1r))))));           \
        __m512d q3i = _mm512_fmadd_pd(vS135, d6i, _mm512_fmadd_pd(vS132, d5i,     \
            _mm512_fnmadd_pd(vS131, d4i, _mm512_fnmadd_pd(vS134, d3i,             \
            _mm512_fmadd_pd(vS136, d2i, _mm512_mul_pd(vS133, d1i))))));           \
        __m512d q4r = _mm512_fnmadd_pd(vS132, d6r, _mm512_fnmadd_pd(vS136, d5r,   \
            _mm512_fmadd_pd(vS133, d4r, _mm512_fnmadd_pd(vS131, d3r,              \
            _mm512_fnmadd_pd(vS135, d2r, _mm512_mul_pd(vS134, d1r))))));          \
        __m512d q4i = _mm512_fnmadd_pd(vS132, d6i, _mm512_fnmadd_pd(vS136, d5i,   \
            _mm512_fmadd_pd(vS133, d4i, _mm512_fnmadd_pd(vS131, d3i,              \
            _mm512_fnmadd_pd(vS135, d2i, _mm512_mul_pd(vS134, d1i))))));          \
        __m512d q5r = _mm512_fmadd_pd(vS134, d6r, _mm512_fnmadd_pd(vS131, d5r,    \
            _mm512_fnmadd_pd(vS136, d4r, _mm512_fmadd_pd(vS132, d3r,              \
            _mm512_fnmadd_pd(vS133, d2r, _mm512_mul_pd(vS135, d1r))))));          \
        __m512d q5i = _mm512_fmadd_pd(vS134, d6i, _mm512_fnmadd_pd(vS131, d5i,    \
            _mm512_fnmadd_pd(vS136, d4i, _mm512_fmadd_pd(vS132, d3i,              \
            _mm512_fnmadd_pd(vS133, d2i, _mm512_mul_pd(vS135, d1i))))));          \
        __m512d q6r = _mm512_fnmadd_pd(vS133, d6r, _mm512_fmadd_pd(vS134, d5r,    \
            _mm512_fnmadd_pd(vS132, d4r, _mm512_fmadd_pd(vS135, d3r,              \
            _mm512_fnmadd_pd(vS131, d2r, _mm512_mul_pd(vS136, d1r))))));          \
        __m512d q6i = _mm512_fnmadd_pd(vS133, d6i, _mm512_fmadd_pd(vS134, d5i,    \
            _mm512_fnmadd_pd(vS132, d4i, _mm512_fmadd_pd(vS135, d3i,              \
            _mm512_fnmadd_pd(vS131, d2i, _mm512_mul_pd(vS136, d1i))))));          \
        ZR[0] = _mm512_add_pd(ZR[0], _mm512_add_pd(                               \
            _mm512_add_pd(t1r, t2r),                                             \
            _mm512_add_pd(_mm512_add_pd(t3r, t4r), _mm512_add_pd(t5r, t6r))));   \
        ZI[0] = _mm512_add_pd(ZI[0], _mm512_add_pd(                               \
            _mm512_add_pd(t1i, t2i),                                             \
            _mm512_add_pd(_mm512_add_pd(t3i, t4i), _mm512_add_pd(t5i, t6i))));   \
        ZR[1] = A(m1r, q1i); ZI[1] = S(m1i, q1r);                                \
        ZR[12] = S(m1r, q1i); ZI[12] = A(m1i, q1r);                              \
        ZR[2] = A(m2r, q2i); ZI[2] = S(m2i, q2r);                                \
        ZR[11] = S(m2r, q2i); ZI[11] = A(m2i, q2r);                              \
        ZR[3] = A(m3r, q3i); ZI[3] = S(m3i, q3r);                                \
        ZR[10] = S(m3r, q3i); ZI[10] = A(m3i, q3r);                              \
        ZR[4] = A(m4r, q4i); ZI[4] = S(m4i, q4r);                                \
        ZR[9] = S(m4r, q4i); ZI[9] = A(m4i, q4r);                                \
        ZR[5] = A(m5r, q5i); ZI[5] = S(m5i, q5r);                                \
        ZR[8] = S(m5r, q5i); ZI[8] = A(m5i, q5r);                                \
        ZR[6] = A(m6r, q6i); ZI[6] = S(m6i, q6r);                                \
        ZR[7] = S(m6r, q6i); ZI[7] = A(m6i, q6r);                                \
    } while (0)

/* fused middle for the 13*2^(2k) convolution sizes (M = 208, 832, ...):
 * DFT-13 blocks + pointwise bh + exact inverse, one pass, mirroring
 * conv_mid7.  Same slot order as dft13_fwd_stage (bh consistency). */
static void conv_mid13(int M, const double *restrict bhre, const double *restrict bhim,
                       double *restrict wr, double *restrict wi)
{
    const __m512d vC131 = _mm512_set1_pd(BST_C131), vC132 = _mm512_set1_pd(BST_C132);
    const __m512d vC133 = _mm512_set1_pd(BST_C133), vC134 = _mm512_set1_pd(BST_C134);
    const __m512d vC135 = _mm512_set1_pd(BST_C135), vC136 = _mm512_set1_pd(BST_C136);
    const __m512d vS131 = _mm512_set1_pd(BST_S131), vS132 = _mm512_set1_pd(BST_S132);
    const __m512d vS133 = _mm512_set1_pd(BST_S133), vS134 = _mm512_set1_pd(BST_S134);
    const __m512d vS135 = _mm512_set1_pd(BST_S135), vS136 = _mm512_set1_pd(BST_S136);
    for (int blk = 0; blk < M; blk += 13) {
        double *qr = wr + (size_t)blk * VL, *qi = wi + (size_t)blk * VL;
        __m512d zr[13], zi[13];
        for (int k = 0; k < 13; ++k) {
            zr[k] = _mm512_load_pd(qr + (size_t)k * VL);
            zi[k] = _mm512_load_pd(qi + (size_t)k * VL);
        }
        BST_DFT13V_ARR(zr, zi, _mm512_add_pd, _mm512_sub_pd);
        for (int k = 0; k < 13; ++k) {
            __m512d hr = _mm512_set1_pd(bhre[blk + k]);
            __m512d hi = _mm512_set1_pd(bhim[blk + k]);
            __m512d ur, ui;
            BST_CMUL(ur, ui, zr[k], zi[k], hr, hi);
            zr[k] = ur;
            zi[k] = ui;
        }
        BST_DFT13V_ARR(zr, zi, _mm512_sub_pd, _mm512_add_pd);
        for (int k = 0; k < 13; ++k) {
            _mm512_store_pd(qr + (size_t)k * VL, zr[k]);
            _mm512_store_pd(qi + (size_t)k * VL, zi[k]);
        }
    }
}
#endif /* __AVX512F__ */

/* Full (unpruned) forward, used at create() on b_pad, which is NOT zero-padded. */
static void fwd_fft_full(const fft3d_plan *p, double *wr, double *wi)
{
    const double *t = p->twf;
    int len = p->M;
    while ((len & 3) == 0) {
        dif4_stage(p->M, len, t, wr, wi);
        t += (size_t)(len >> 2) * 6;
        len >>= 2;
    }
    if (len == 2) r2_stage(p->M, wr, wi);
    else if (len == 3) dft3_fwd_stage(p->M, wr, wi);
    else if (len == 6) dft6_fwd_stage(p->M, wr, wi);
    else if (len == 5) dft5_fwd_stage(p->M, wr, wi);
    else if (len == 10) dft10_fwd_stage(p->M, wr, wi);
    else if (len == 7) dft7_fwd_stage(p->M, wr, wi);
    else if (len == 14) dft14_fwd_stage(p->M, wr, wi);
    else if (len == 13) dft13_fwd_stage(p->M, wr, wi);
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
    while (len >= 8 && (len & 3) == 0) {
#ifdef BST_MID12
        /* raced gen_r6, default OFF: fusing dif4(12)+mid3+dit4(12) into one
         * pass was a wash-to-loss at L=20 B=32 (84.5/84.9/84.5 vs
         * 85.3/84.8/87.6 us control-first pairs) -- the 6 KiB M=48 buffer is
         * L1-resident, so the deleted passes were free and the 24-live-zmm
         * block spills.  Kept for the CLX/SPR cross-arch re-race. */
        if (len == 12) break; /* fused conv_mid12 owns this stage + tail */
#endif
#ifndef BST_NOR16
        if (len >= 32 && (len & 15) == 0) { /* fuse with the next: one pass */
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
    else if (len == 3)
        conv_mid3(M, p->bhre, p->bhim, wr, wi);
    else if (len == 6)
        conv_mid6(M, p->bhre, p->bhim, wr, wi);
    else if (len == 5)
        conv_mid5(M, p->bhre, p->bhim, wr, wi);
    else if (len == 10)
        conv_mid10(M, p->bhre, p->bhim, wr, wi);
    else if (len == 7)
        conv_mid7(M, p->bhre, p->bhim, wr, wi);
    else if (len == 14)
        conv_mid14(M, p->bhre, p->bhim, wr, wi);
    else if (len == 13)
        conv_mid13(M, p->bhre, p->bhim, wr, wi);
    else if (len == 12)
        conv_mid12(M, t, p->twi, p->bhre, p->bhim, wr, wi);
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
        int l2 = (len == 4) ? 16 : 4 * len; /* 8, 12, 24 -- or 48 after mid12 */
        if (len == 4) ti += 6; /* skip the fused len-4 stage's twiddle chunk */
        else if (len == 12) ti += 18; /* mid12 consumed the len-12 chunk */
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
    int tail = M;
    while ((tail & 3) == 0) tail >>= 2; /* 1, 2, 3, 5, 6, 7, 10, 13, or 14 */
    if (tail == 2) r2_stage(M, wr, wi);
    else if (tail == 3) dft3_inv_stage(M, wr, wi);
    else if (tail == 6) dft6_inv_stage(M, wr, wi);
    else if (tail == 5) dft5_inv_stage(M, wr, wi);
    else if (tail == 10) dft10_inv_stage(M, wr, wi);
    else if (tail == 7) dft7_inv_stage(M, wr, wi);
    else if (tail == 14) dft14_inv_stage(M, wr, wi);
    else if (tail == 13) dft13_inv_stage(M, wr, wi);
    const double *t = p->twi;
    for (int len = (tail == 1) ? 4 : 4 * tail; len <= M; len <<= 2) {
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
    {   /* 3*2^k, 5*2^k and 7*2^k candidates, k >= 4 so S = M/4 stays 4-aligned
         * for the transpose gather/scatter path; the grid 48,64,80,96,112,128,
         * 160,192,224,256,... shrinks M up to 25% (33%, 12.5%) on the 3- (5-,
         * 7-) slices of every L-octave.  The 7-slice (gen_r7) covers graded
         * L=50 (128->112) and L=100 (256->224). */
        int M3 = 48, M5 = 80, M7 = 112;
        while (M3 < 2 * L - 1) M3 <<= 1;
        while (M5 < 2 * L - 1) M5 <<= 1;
        while (M7 < 2 * L - 1) M7 <<= 1;
        if (M3 < M) M = M3;
        if (M5 < M) M = M5;
#ifndef BST_NO7 /* attribution control: -DBST_NO7 = the gen_r6 grid */
        if (M7 < M) M = M7;
#endif
#ifdef BST_M13 /* raced OFF gen_r8: M=208 (13*2^(2k), tail DFT-13) lost 3/3
                * control-first same-core pairs at L=100 B=1 m=64 (16000/
                * 16845/17194 vs 14419/14589/16288 us) -- the 6-term DFT-13
                * dot products are FMA-port-bound (llvm-mca: 460 cyc/blk vs
                * PFA-14's 313, +58%/pt) and eat the 7% data cut.  Kept as a
                * create()-side knob for the CLX/SPR cross-arch re-race. */
        {   /* 13*2^(2k) only (<<= 2 keeps the tail at 13, never 26):
             * the M = 208 slice is L = 97..104 */
            int M13 = 208;
            while (M13 < 2 * L - 1) M13 <<= 2;
            if (M13 < M) M = M13;
        }
#endif
    }
    p->M = M;

    /* the radix-4 DIF chain ends at tail = 1 (even-log2 pow2), 2 (odd-log2
     * pow2), or 3/6 (the 3*2^k sizes); the tail stage is twiddle-free */
    int tail = M;
    while ((tail & 3) == 0) tail >>= 2;

    /* twiddle table sizes: sum over radix-4 stages of (len/4)*6 doubles
     * (the forward and inverse chains consume the same chunk lengths) */
    size_t twn = 0;
    for (int len = M; (len & 3) == 0; len >>= 2) twn += (size_t)(len >> 2) * 6;

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
        for (int len = M; (len & 3) == 0; len >>= 2) {
            int S = len >> 2;
            tw_fill_ct_int_colmajor(tf, len, 4, S);
            if (tw_audit_ct_int_colmajor(tf, len, 4, S) > 0.51) {
                fft3d_destroy(p); /* cannot happen; audit is the contract */
                return NULL;
            }
            tf += (size_t)S * 6;
        }
        double *ti = p->twi;
        for (int len = (tail == 1) ? 4 : 4 * tail; len <= M; len <<= 2) {
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
    free(p->cc);
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
 * into the same store pass, c read STRIDED at the dst offsets; cch != NULL
 * (blocked axis-1 pass) reads c from the custody chunk instead: [8 re][8 im]
 * per element k, two aligned sequential 128-B streams (gen_pow2 GP2_CT).
 * always_inline + literal NULLs at the call sites keep each specialization
 * free of the branches it does not use. */
static inline __attribute__((always_inline)) void
last_scatter_contig_core(const fft3d_plan *p, double *q, long s2,
                         const double *restrict wr, const double *restrict wi,
                         int pf, const double *cq, const double *cch)
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
        if (cch) {
            __m512d car = _mm512_load_pd(cch + (size_t)j * 2 * VL);
            __m512d cai = _mm512_load_pd(cch + (size_t)j * 2 * VL + VL);
            BST_MAP8(r0r, r0i, car, cai);
        } else if (cq) {
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
            if (cch) {
                __m512d car = _mm512_load_pd(cch + (size_t)(j + S) * 2 * VL);
                __m512d cai = _mm512_load_pd(cch + (size_t)(j + S) * 2 * VL + VL);
                BST_MAP8(r1r, r1i, car, cai);
            } else if (cq) {
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
    last_scatter_contig_core(p, q, s2, wr, wi, pf, NULL, NULL);
}

static void last_scatter_contig_map(const fft3d_plan *p, double *q, long s2,
                                    const double *restrict wr,
                                    const double *restrict wi, int pf,
                                    const double *cq)
{
    last_scatter_contig_core(p, q, s2, wr, wi, pf, cq, NULL);
}

static void last_scatter_contig_cust(const fft3d_plan *p, double *q, long s2,
                                     const double *restrict wr,
                                     const double *restrict wi, int pf,
                                     const double *cch)
{
    last_scatter_contig_core(p, q, s2, wr, wi, pf, NULL, cch);
}

/* seam-group mirror: last stage + chirp (+ optional fused map) + interleave,
 * stored through the group-constant masks back into the two runs. */
static void last_scatter_seam(const fft3d_plan *p, double *q1, double *q2, int k,
                              long s2, const double *restrict wr,
                              const double *restrict wi,
                              const double *cq1, const double *cq2,
                              const double *cch)
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
        if (cch) {
            __m512d car = _mm512_load_pd(cch + (size_t)j * 2 * VL);
            __m512d cai = _mm512_load_pd(cch + (size_t)j * 2 * VL + VL);
            BST_MAP8(r0r, r0i, car, cai);
        } else if (cq1) {
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
            if (cch) {
                __m512d car = _mm512_load_pd(cch + (size_t)(j + S) * 2 * VL);
                __m512d cai = _mm512_load_pd(cch + (size_t)(j + S) * 2 * VL + VL);
                BST_MAP8(r1r, r1i, car, cai);
            } else if (cq1) {
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
 * cmap != NULL (fft3d_chain, fused regime, axis 0 only): the scatter stores
 * (z + c)/(1 + |z + c|) instead of z, c read at the dst offsets.
 * ccust != NULL (fft3d_chain, blocked regime, axis 1 only): same map, but c
 * comes from the custody buffer -- chunk r0/8 holds the group's c in element
 * order, split [8 re][8 im] per k (build_ccust).  rlo must be 8-aligned. */
static void axis_pass(fft3d_plan *p, const double *src, double *dst,
                      long rlo, long rhi, long div, long A, long B, long stride,
                      const double *cmap, const double *ccust)
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
        const double *cch =
            ccust ? ccust + (size_t)(r0 / VL) * (size_t)L * 2 * VL : NULL;
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
            if (cch)
                last_scatter_contig_cust(p, dbase[0], s2, wr, wi, pf, cch);
            else if (cmap)
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
                              cmap ? cmap + offk : NULL, cch);
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
                if (cch) {
                    zr += cch[(size_t)k * 2 * VL + v];
                    zi += cch[(size_t)k * 2 * VL + VL + v];
                    double sc = 1.0 / (1.0 + sqrt(zr * zr + zi * zi + 1e-300));
                    zr *= sc;
                    zi *= sc;
                } else if (cmap) {
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
    axis_pass(p, src, dst, 0, nrows, 1, L, 0, 1, NULL, NULL);
#if BST_AXES >= 2
    /* axis 1 (y, stride L): row r -> (r/L)*L^2 + r%L -> out in place */
    axis_pass(p, dst, dst, 0, nrows, L, L * L, 1, L, NULL, NULL);
#endif
#if BST_AXES >= 3
    /* axis 0 (x, stride L^2): row r -> (r/L^2)*L^3 + r%L^2 */
    axis_pass(p, dst, dst, 0, nrows, L * L, L * L * L, 1, L * L, NULL, NULL);
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

/* Build the custody copy of c for the blocked axis-1 fused-map scatter
 * (gen_pow2 gen_r4's GP2_CT idea: store the chain operand in the LAST
 * pass's consumption order).  Chunk g (= group r0 = 8g of the axis-1 row
 * order) holds, for k = 0..L-1, [8 re][8 im] of c at the group's 8 row
 * offsets, element k.  The fused scatter then reads c as two aligned
 * interleaved sequential streams (+128 B each) the L2 streamer covers,
 * instead of 2L cold 128-B touches at stride 16L B per group -- the
 * pattern that killed axis-0 map fusion past LLC reach (gen_r3).  Runs
 * once per fft3d_chain call; the group decomposition below must mirror
 * axis_pass exactly (same contig test, same lane-offset formula). */
static void build_ccust(const fft3d_plan *p, const double *cm, double *cc)
{
    const long L = p->L, B = p->batch;
    const long nrows = B * L * L;
    const long div = L, A = L * L;
    const long s2 = 2 * L;
    const __m512i IDXE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i IDXO = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    for (long r0 = 0; r0 < nrows; r0 += VL) {
        const int nv = (nrows - r0 < VL) ? (int)(nrows - r0) : VL;
        double *ch = cc + (size_t)(r0 / VL) * (size_t)L * 2 * VL;
        if (nv == VL && (r0 % div) + VL <= div) { /* contig group */
            const double *q = cm + 2 * ((r0 / div) * A + (r0 % div));
            for (int k = 0; k < L; ++k) {
                __m512d z0 = _mm512_loadu_pd(q + (size_t)k * s2);
                __m512d z1 = _mm512_loadu_pd(q + (size_t)k * s2 + 8);
                _mm512_store_pd(ch + (size_t)k * 2 * VL,
                                _mm512_permutex2var_pd(z0, IDXE, z1));
                _mm512_store_pd(ch + (size_t)k * 2 * VL + VL,
                                _mm512_permutex2var_pd(z0, IDXO, z1));
            }
            continue;
        }
        for (int v = 0; v < VL; ++v) { /* seam / tail: scalar, rare */
            long r = r0 + (v < nv ? v : 0);
            const double *q = cm + 2 * ((r / div) * A + (r % div));
            for (int k = 0; k < L; ++k) {
                ch[(size_t)k * 2 * VL + v] = q[(size_t)k * s2];
                ch[(size_t)k * 2 * VL + VL + v] = q[(size_t)k * s2 + 1];
            }
        }
    }
}
#endif /* __AVX512F__ */

/* The graded m-step map chain, owned (gen_r3): every pass is in place in
 * final_out (each row depends only on itself), and the map + c-add is fused
 * into the axis-0 scatter -- vs the driver fallback this deletes the per-step
 * map pass (one full-volume read + write plus c reread) and its scalar-ish
 * sqrt/div, and the initial memcpy.  x0 is never written.
 * Past LLC reach the fused scatter's strided c reads stop paying (raced on
 * the node, gen_r3) and the chain moves to the blocked-custody regime; the
 * gate is on state + c COMBINED (both stream through the scatter -- the r3
 * code gated on state alone, which wrongly kept L=40/50 fused: 1987 vs 1822
 * us at L=50, same window, gen_r4).  gen_r5: gate lowered 15 -> 14 MiB --
 * with the axis-1 custody-fused map the blocked regime now beats the fused
 * one at L=31 B=16 (14.55 MiB): 5/6 same-core pairs, mean -2.9%, best 288.6
 * vs 302.2 us. */
#ifndef BST_MAPFUSE_MAX_MIB
#define BST_MAPFUSE_MAX_MIB 14
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
         * the unblocked passes -- axis 2 and axis 1 run while the block
         * (k*L^2*16 B <= 320 KiB at L <= 100) is L2-hot.  Deletes the axis-1
         * full-volume round trip from DRAM.
         * gen_r5: the map is fused into the axis-1 scatter (the block's last
         * touch), reading c from a CUSTODY buffer built once per chain call
         * in the scatter's exact consumption order (gen_pow2 gen_r4's GP2_CT
         * adopted).  Sequential custody reads dodge the strided-cold-c
         * latency that made the r3 race send this regime to a separate
         * sweep; the sweep's block read+write and its de/re-interleave
         * permutes are deleted.  Per-element map arithmetic identical =>
         * chain output bit-identical to gen_r4.  Knobs: -DBST_NOCFUSE
         * restores the r4 sweep; -DBST_CSTRIDED fuses but reads c strided
         * (the attribution arm). */
        long k = 1;
        while ((k * L) & 7) k <<= 1;
        const long rpb = k * L; /* rows per block, multiple of 8 */
        const double *cm1 = NULL; /* axis-1 strided cmap (attribution arm) */
        const double *ccb = NULL; /* axis-1 custody base */
#if defined(BST_CSTRIDED)
        cm1 = cm;
#elif !defined(BST_NOCFUSE)
        if (!p->cc)
            p->cc = xalloc((size_t)((nrows + VL - 1) / VL) * (size_t)L * 2 * VL);
        if (p->cc) {
            build_ccust(p, cm, p->cc);
            ccb = p->cc;
        } /* alloc failure -> r4 separate-sweep path below */
#endif
        for (int s = 0; s < m; ++s) {
            const double *src = (s == 0) ? (const double *)x0 : dst;
            axis_pass(p, src, dst, 0, nrows, L * L, L * L * L, 1, L * L,
                      NULL, NULL);
            for (long r0 = 0; r0 < nrows; r0 += rpb) {
                const long r1 = (r0 + rpb < nrows) ? r0 + rpb : nrows;
                axis_pass(p, dst, dst, r0, r1, 1, L, 0, 1, NULL, NULL);
                if (ccb || cm1) {
                    axis_pass(p, dst, dst, r0, r1, L, L * L, 1, L, cm1, ccb);
                } else {
                    axis_pass(p, dst, dst, r0, r1, L, L * L, 1, L, NULL, NULL);
                    map_pass_seq(dst + 2 * (size_t)r0 * L,
                                 cm + 2 * (size_t)r0 * L,
                                 (size_t)(r1 - r0) * L);
                }
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
                axis_pass(p, src, dst, r0, r1, 1, L, 0, 1, NULL, NULL);
                axis_pass(p, dst, dst, r0, r1, L, L * L, 1, L, NULL, NULL);
            }
        }
#else
        axis_pass(p, src, dst, 0, nrows, 1, L, 0, 1, NULL, NULL);
        axis_pass(p, dst, dst, 0, nrows, L, L * L, 1, L, NULL, NULL);
#endif
        axis_pass(p, dst, dst, 0, nrows, L * L, L * L * L, 1, L * L,
                  fuse ? cm : NULL, NULL);
#ifdef __AVX512F__
        if (!fuse) map_pass_seq(dst, cm, (size_t)nrows * (size_t)L);
#endif
    }
}
#endif /* BST_NOCHAIN */
