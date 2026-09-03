/* d1_pow2: power-of-two 1D complex-double FFT.
 *
 * Algorithm: Stockham autosort (no bit-reversal, ping-pong out-of-place), DIF
 * radix-4 stages with a twiddle-free radix-4 or radix-2 codelet as the final
 * stage.  All stages except the first have butterfly stride s >= 4 complexes,
 * so they vectorize over q with one zmm = 4 interleaved complexes and
 * BROADCAST twiddles (no per-lane twiddle traffic).  The first stage (s == 1)
 * vectorizes over the twiddle index p with full per-lane twiddle tables
 * (w^p, w^2p, w^3p precomputed -- deriving them in-loop by squaring was both
 * slower and a source of BIASED rounding that a long map chain amplifies)
 * and a 4x4 complex-lane transpose before the store.
 *
 * Complex multiply on interleaved data: twiddles are stored as a broadcastable
 * scalar re part plus a (-im,+im) pair, so u*w = fmadd(swap(u), wpair, u*wre):
 * one in-lane shuffle + mul + fma, no sign-mask xors.
 *
 * fft1d_chain (the fused map chain) is owned:
 *   - the chain is separable per transform (the FFT is batched-independent and
 *     the map is pointwise), so each transform is driven through ALL m steps
 *     while its ~4L*16 bytes of working set stays cache-resident.  Libraries
 *     timed through the driver fallback must stream the full B*L batch three
 *     times per step, which is DRAM-bound at the large batched cells.
 *   - the map z/(1+|z|) is fused into the final butterfly stage (no extra
 *     read+write pass) and computed with rsqrt14+rcp14 plus two Newton steps
 *     each instead of vsqrtpd+vdivpd: ~4x the throughput at <=2 ulp.
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../fft1d_api.h"

#if defined(__AVX512F__) && defined(__AVX512DQ__)
#define D1_AVX512 1
#include <immintrin.h>
#endif

/* long-double pi: twiddles must be correctly rounded doubles -- an M_PI-based
 * angle carries a BIASED ~2e-16 phase error that a 10^4-step map chain
 * amplifies past the 1e-10 gate (measured at L=128, m=30000) */
#define PIL 3.14159265358979323846264338327950288L

enum { ST_S1, ST_GEN, ST_GEN8, ST_F4, ST_F2, ST_F8 };

typedef struct {
    int type, n, s;
    const double *tw;
} stage_t;

struct fft1d_plan {
    int L, batch, T;
    stage_t st[10];
    double *twmem;
    double *scratch;   /* 2L doubles: ping-pong buffer */
    double *state;     /* 2L doubles: chain state for one transform */
    double *tws1full;  /* L = 32/64 codelets: full dup-format w,w^2,w^3 table */
    int nt;            /* stream (non-temporal) final-stage stores in execute */
};

const char *fft1d_name(void) { return "d1_pow2"; }
const char *fft1d_description(void)
{
    return "Stockham autosort radix-4/8 DIF, AVX-512 4-complex lanes, broadcast twiddles, "
           "in-register codelets at L=32/64; fused chain: per-transform m-step blocking + "
           "rsqrt/rcp Newton+residual map in the final butterfly stage";
}

int fft1d_supports(int L) { return L >= 16 && L <= 65536 && (L & (L - 1)) == 0; }

/* ---------------------------------------------------------------- kernels */
#ifdef D1_AVX512

#define PSWAP 0x55 /* vpermilpd: swap re/im inside each 128-bit pair */

static inline __m512d cmul_bc(__m512d u, __m512d wr, __m512d wp)
{
    /* u * w with wr = dup(re w), wp = (-im w, +im w) per pair */
    return _mm512_fmadd_pd(_mm512_permute_pd(u, PSWAP), wp, _mm512_mul_pd(u, wr));
}

/* first stage: s == 1, vectorized over p (4 butterflies), table holds w^p only */
static void stage_s1(const double *restrict x, double *restrict y, int n,
                     const double *restrict tw)
{
    const int m = n >> 2;
    const __m512d ONE = _mm512_set1_pd(1.0);
    const __m512i idxA = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    const __m512i idxB = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
    for (int p = 0; p < m; p += 4) {
        __m512d a = _mm512_load_pd(x + 2 * p);
        __m512d b = _mm512_load_pd(x + 2 * (p + m));
        __m512d c = _mm512_load_pd(x + 2 * (p + 2 * m));
        __m512d d = _mm512_load_pd(x + 2 * (p + 3 * m));
        __m512d apc = _mm512_add_pd(a, c), amc = _mm512_sub_pd(a, c);
        __m512d bpd = _mm512_add_pd(b, d), bmd = _mm512_sub_pd(b, d);
        __m512d sw = _mm512_permute_pd(bmd, PSWAP);
        __m512d r0 = _mm512_add_pd(apc, bpd);
        __m512d u2 = _mm512_sub_pd(apc, bpd);
        __m512d u1 = _mm512_fmsubadd_pd(amc, ONE, sw); /* amc - i*bmd */
        __m512d u3 = _mm512_fmaddsub_pd(amc, ONE, sw); /* amc + i*bmd */
        const double *t = tw + 12 * p; /* 48 doubles per 4 p: w,w^2,w^3 tables */
        __m512d r1 = cmul_bc(u1, _mm512_load_pd(t), _mm512_load_pd(t + 8));
        __m512d r2 = cmul_bc(u2, _mm512_load_pd(t + 16), _mm512_load_pd(t + 24));
        __m512d r3 = cmul_bc(u3, _mm512_load_pd(t + 32), _mm512_load_pd(t + 40));
        /* transpose 4 lanes x 4 outputs so y[4p..4p+15] stores contiguously */
        __m512d p0 = _mm512_permutex2var_pd(r0, idxA, r1);
        __m512d p1 = _mm512_permutex2var_pd(r0, idxB, r1);
        __m512d p2 = _mm512_permutex2var_pd(r2, idxA, r3);
        __m512d p3 = _mm512_permutex2var_pd(r2, idxB, r3);
        _mm512_store_pd(y + 8 * p, _mm512_shuffle_f64x2(p0, p2, 0x44));
        _mm512_store_pd(y + 8 * p + 8, _mm512_shuffle_f64x2(p1, p3, 0x44));
        _mm512_store_pd(y + 8 * p + 16, _mm512_shuffle_f64x2(p0, p2, 0xEE));
        _mm512_store_pd(y + 8 * p + 24, _mm512_shuffle_f64x2(p1, p3, 0xEE));
    }
}

/* general radix-4 stage, s >= 4: vectorized over q, broadcast twiddles */
static void stage_gen(const double *restrict x, double *restrict y, int n, int s,
                      const double *restrict tw)
{
    const int m = n >> 2;
    const long S = 2L * s;
    const __m512d ONE = _mm512_set1_pd(1.0);
    for (int p = 0; p < m; ++p) {
        const double *t = tw + 9 * p;
        const __m512d w1r = _mm512_set1_pd(t[0]);
        const __m512d w2r = _mm512_set1_pd(t[1]);
        const __m512d w3r = _mm512_set1_pd(t[2]);
        const __m512d w1p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 3));
        const __m512d w2p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 5));
        const __m512d w3p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 7));
        const double *xa = x + S * p, *xb = xa + S * m, *xc = xb + S * m, *xd = xc + S * m;
        double *ya = y + S * 4 * p, *yb = ya + S, *yc = yb + S, *yd = yc + S;
        for (long q = 0; q < S; q += 8) {
            __m512d a = _mm512_load_pd(xa + q), b = _mm512_load_pd(xb + q);
            __m512d c = _mm512_load_pd(xc + q), d = _mm512_load_pd(xd + q);
            __m512d apc = _mm512_add_pd(a, c), amc = _mm512_sub_pd(a, c);
            __m512d bpd = _mm512_add_pd(b, d), bmd = _mm512_sub_pd(b, d);
            __m512d sw = _mm512_permute_pd(bmd, PSWAP);
            _mm512_store_pd(ya + q, _mm512_add_pd(apc, bpd));
            _mm512_store_pd(yb + q, cmul_bc(_mm512_fmsubadd_pd(amc, ONE, sw), w1r, w1p));
            _mm512_store_pd(yc + q, cmul_bc(_mm512_sub_pd(apc, bpd), w2r, w2p));
            _mm512_store_pd(yd + q, cmul_bc(_mm512_fmaddsub_pd(amc, ONE, sw), w3r, w3p));
        }
    }
}

/* radix-8 DIF butterfly on 8 input vectors (32 complexes), no twiddles.
 * u[2k]   = DFT4 of (x_j + x_{j+4});  u[2k+1] = DFT4 of ((x_j - x_{j+4}) * w8^j). */
#define R8_BODY(X0, X1, X2, X3, X4, X5, X6, X7)                                        \
    __m512d s0 = _mm512_add_pd(X0, X4), s1 = _mm512_add_pd(X1, X5);                    \
    __m512d s2 = _mm512_add_pd(X2, X6), s3 = _mm512_add_pd(X3, X7);                    \
    __m512d d0 = _mm512_sub_pd(X0, X4), d1 = _mm512_sub_pd(X1, X5);                    \
    __m512d d2 = _mm512_sub_pd(X2, X6), d3 = _mm512_sub_pd(X3, X7);                    \
    __m512d apc = _mm512_add_pd(s0, s2), amc = _mm512_sub_pd(s0, s2);                  \
    __m512d bpd = _mm512_add_pd(s1, s3), bmd = _mm512_sub_pd(s1, s3);                  \
    __m512d swe = _mm512_permute_pd(bmd, PSWAP);                                       \
    __m512d u0 = _mm512_add_pd(apc, bpd);                                              \
    __m512d u4 = _mm512_sub_pd(apc, bpd);                                              \
    __m512d u2 = _mm512_fmsubadd_pd(amc, ONE, swe);                                    \
    __m512d u6 = _mm512_fmaddsub_pd(amc, ONE, swe);                                    \
    __m512d e1 = _mm512_mul_pd(_mm512_fmsubadd_pd(d1, ONE, _mm512_permute_pd(d1, PSWAP)), Cq); \
    __m512d e3 = _mm512_mul_pd(                                                        \
        _mm512_permute_pd(_mm512_fmsubadd_pd(d3, ONE, _mm512_permute_pd(d3, PSWAP)), PSWAP), CPN); \
    __m512d sw2 = _mm512_permute_pd(d2, PSWAP);                                        \
    __m512d apo = _mm512_fmsubadd_pd(d0, ONE, sw2); /* d0 + (-i)d2 */                  \
    __m512d amo = _mm512_fmaddsub_pd(d0, ONE, sw2); /* d0 - (-i)d2 */                  \
    __m512d bpo = _mm512_add_pd(e1, e3), bmo = _mm512_sub_pd(e1, e3);                  \
    __m512d swo = _mm512_permute_pd(bmo, PSWAP);                                       \
    __m512d u1 = _mm512_add_pd(apo, bpo);                                              \
    __m512d u5 = _mm512_sub_pd(apo, bpo);                                              \
    __m512d u3 = _mm512_fmsubadd_pd(amo, ONE, swo);                                    \
    __m512d u7 = _mm512_fmaddsub_pd(amo, ONE, swo)

#define R8_CONSTS                                                                      \
    const __m512d ONE = _mm512_set1_pd(1.0);                                           \
    const __m512d Cq = _mm512_set1_pd(0.70710678118654752440);                         \
    const __m512d CPN = _mm512_setr_pd(0.70710678118654752440, -0.70710678118654752440,\
                                       0.70710678118654752440, -0.70710678118654752440,\
                                       0.70710678118654752440, -0.70710678118654752440,\
                                       0.70710678118654752440, -0.70710678118654752440)

/* general radix-8 stage, s >= 4: vectorized over q, broadcast twiddles (21 dbl/p) */
static void stage_gen8(const double *restrict x, double *restrict y, int n, int s,
                       const double *restrict tw)
{
    const int m = n >> 3;
    const long S = 2L * s;
    R8_CONSTS;
    for (int p = 0; p < m; ++p) {
        const double *t = tw + 21 * p;
        __m512d wr1 = _mm512_set1_pd(t[0]), wr2 = _mm512_set1_pd(t[1]);
        __m512d wr3 = _mm512_set1_pd(t[2]), wr4 = _mm512_set1_pd(t[3]);
        __m512d wr5 = _mm512_set1_pd(t[4]), wr6 = _mm512_set1_pd(t[5]);
        __m512d wr7 = _mm512_set1_pd(t[6]);
        __m512d wp1 = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 7));
        __m512d wp2 = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 9));
        __m512d wp3 = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 11));
        __m512d wp4 = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 13));
        __m512d wp5 = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 15));
        __m512d wp6 = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 17));
        __m512d wp7 = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 19));
        const double *xa = x + S * p;
        double *ya = y + S * 8 * p;
        const long Sm = S * m;
        for (long q = 0; q < S; q += 8) {
            __m512d x0 = _mm512_load_pd(xa + q), x1 = _mm512_load_pd(xa + Sm + q);
            __m512d x2 = _mm512_load_pd(xa + 2 * Sm + q), x3 = _mm512_load_pd(xa + 3 * Sm + q);
            __m512d x4 = _mm512_load_pd(xa + 4 * Sm + q), x5 = _mm512_load_pd(xa + 5 * Sm + q);
            __m512d x6 = _mm512_load_pd(xa + 6 * Sm + q), x7 = _mm512_load_pd(xa + 7 * Sm + q);
            R8_BODY(x0, x1, x2, x3, x4, x5, x6, x7);
            _mm512_store_pd(ya + q, u0);
            _mm512_store_pd(ya + S + q, cmul_bc(u1, wr1, wp1));
            _mm512_store_pd(ya + 2 * S + q, cmul_bc(u2, wr2, wp2));
            _mm512_store_pd(ya + 3 * S + q, cmul_bc(u3, wr3, wp3));
            _mm512_store_pd(ya + 4 * S + q, cmul_bc(u4, wr4, wp4));
            _mm512_store_pd(ya + 5 * S + q, cmul_bc(u5, wr5, wp5));
            _mm512_store_pd(ya + 6 * S + q, cmul_bc(u6, wr6, wp6));
            _mm512_store_pd(ya + 7 * S + q, cmul_bc(u7, wr7, wp7));
        }
    }
}

/* The chain map w = z/(1+|z|), z = v + c, sqrt/div-free.
 *
 * Accuracy matters more than a lone transform here: at pow2 sizes the two
 * numpy reference paths of the chain gate agree bitwise, so the tolerance
 * floors at 1e-10 and per-step map error accumulates through a weakly chaotic
 * chain.  Plain 2-Newton rsqrt/rcp (~2-3 ulp) MEASURED 6.7e-10 at L=128
 * m=30000 -- FAILED.  Each approximation therefore ends with an exact-residual
 * FMA refinement (Heron for sqrt, residual corrections for 1/t and for the
 * quotient), which lands every elementary result at ~0.5-1 ulp, the same
 * quality as the driver's sqrt/div map. */
static inline __m512d map_vec(__m512d v, __m512d cv)
{
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d th = _mm512_set1_pd(1.5);
    const __m512d two = _mm512_set1_pd(2.0);
    __m512d z = _mm512_add_pd(v, cv);
    __m512d zz = _mm512_mul_pd(z, z);
    __m512d s = _mm512_add_pd(zz, _mm512_permute_pd(zz, PSWAP)); /* |z|^2 dup'd per pair */
    s = _mm512_max_pd(s, _mm512_set1_pd(1e-300));                /* rsqrt(0) guard */
    __m512d r = _mm512_rsqrt14_pd(s);
    __m512d hs = _mm512_mul_pd(s, half);
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hs, r), r, th));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hs, r), r, th));
    __m512d u = _mm512_mul_pd(s, r);                            /* ~sqrt(s) */
    u = _mm512_fmadd_pd(_mm512_mul_pd(half, r), _mm512_fnmadd_pd(u, u, s), u);
    __m512d t = _mm512_add_pd(one, u);                          /* 1 + |z| */
    __m512d rc = _mm512_rcp14_pd(t);
    rc = _mm512_mul_pd(rc, _mm512_fnmadd_pd(t, rc, two));
    rc = _mm512_fmadd_pd(rc, _mm512_fnmadd_pd(t, rc, one), rc);
    __m512d q = _mm512_mul_pd(z, rc);
    return _mm512_fmadd_pd(_mm512_fnmadd_pd(q, t, z), rc, q);
}

/* final radix-4, n == 4, twiddle-free; optionally fused with the chain map */
static void stage_f4(const double *restrict x, double *restrict y, int s,
                     const double *restrict cf, int nt)
{
    const long S = 2L * s;
    const __m512d ONE = _mm512_set1_pd(1.0);
    const double *xa = x, *xb = x + S, *xc = x + 2 * S, *xd = x + 3 * S;
    double *ya = y, *yb = y + S, *yc = y + 2 * S, *yd = y + 3 * S;
    if (!cf) {
        for (long q = 0; q < S; q += 8) {
            __m512d a = _mm512_load_pd(xa + q), b = _mm512_load_pd(xb + q);
            __m512d c = _mm512_load_pd(xc + q), d = _mm512_load_pd(xd + q);
            __m512d apc = _mm512_add_pd(a, c), amc = _mm512_sub_pd(a, c);
            __m512d bpd = _mm512_add_pd(b, d), bmd = _mm512_sub_pd(b, d);
            __m512d sw = _mm512_permute_pd(bmd, PSWAP);
            __m512d r0 = _mm512_add_pd(apc, bpd);
            __m512d r1 = _mm512_fmsubadd_pd(amc, ONE, sw);
            __m512d r2 = _mm512_sub_pd(apc, bpd);
            __m512d r3 = _mm512_fmaddsub_pd(amc, ONE, sw);
            if (nt) {
                _mm512_stream_pd(ya + q, r0);
                _mm512_stream_pd(yb + q, r1);
                _mm512_stream_pd(yc + q, r2);
                _mm512_stream_pd(yd + q, r3);
            } else {
                _mm512_store_pd(ya + q, r0);
                _mm512_store_pd(yb + q, r1);
                _mm512_store_pd(yc + q, r2);
                _mm512_store_pd(yd + q, r3);
            }
        }
    } else {
        const double *ca = cf, *cb = cf + S, *cc = cf + 2 * S, *cd = cf + 3 * S;
        for (long q = 0; q < S; q += 8) {
            __m512d a = _mm512_load_pd(xa + q), b = _mm512_load_pd(xb + q);
            __m512d c = _mm512_load_pd(xc + q), d = _mm512_load_pd(xd + q);
            __m512d apc = _mm512_add_pd(a, c), amc = _mm512_sub_pd(a, c);
            __m512d bpd = _mm512_add_pd(b, d), bmd = _mm512_sub_pd(b, d);
            __m512d sw = _mm512_permute_pd(bmd, PSWAP);
            _mm512_store_pd(ya + q, map_vec(_mm512_add_pd(apc, bpd), _mm512_load_pd(ca + q)));
            _mm512_store_pd(yb + q, map_vec(_mm512_fmsubadd_pd(amc, ONE, sw), _mm512_load_pd(cb + q)));
            _mm512_store_pd(yc + q, map_vec(_mm512_sub_pd(apc, bpd), _mm512_load_pd(cc + q)));
            _mm512_store_pd(yd + q, map_vec(_mm512_fmaddsub_pd(amc, ONE, sw), _mm512_load_pd(cd + q)));
        }
    }
}

/* final radix-8, n == 8, twiddle-free; optionally fused with the chain map */
static void stage_f8(const double *restrict x, double *restrict y, int s,
                     const double *restrict cf, int nt)
{
    const long S = 2L * s;
    R8_CONSTS;
    if (!cf) {
        for (long q = 0; q < S; q += 8) {
            __m512d x0 = _mm512_load_pd(x + q), x1 = _mm512_load_pd(x + S + q);
            __m512d x2 = _mm512_load_pd(x + 2 * S + q), x3 = _mm512_load_pd(x + 3 * S + q);
            __m512d x4 = _mm512_load_pd(x + 4 * S + q), x5 = _mm512_load_pd(x + 5 * S + q);
            __m512d x6 = _mm512_load_pd(x + 6 * S + q), x7 = _mm512_load_pd(x + 7 * S + q);
            R8_BODY(x0, x1, x2, x3, x4, x5, x6, x7);
            if (nt) {
                _mm512_stream_pd(y + q, u0);
                _mm512_stream_pd(y + S + q, u1);
                _mm512_stream_pd(y + 2 * S + q, u2);
                _mm512_stream_pd(y + 3 * S + q, u3);
                _mm512_stream_pd(y + 4 * S + q, u4);
                _mm512_stream_pd(y + 5 * S + q, u5);
                _mm512_stream_pd(y + 6 * S + q, u6);
                _mm512_stream_pd(y + 7 * S + q, u7);
            } else {
                _mm512_store_pd(y + q, u0);
                _mm512_store_pd(y + S + q, u1);
                _mm512_store_pd(y + 2 * S + q, u2);
                _mm512_store_pd(y + 3 * S + q, u3);
                _mm512_store_pd(y + 4 * S + q, u4);
                _mm512_store_pd(y + 5 * S + q, u5);
                _mm512_store_pd(y + 6 * S + q, u6);
                _mm512_store_pd(y + 7 * S + q, u7);
            }
        }
    } else {
        for (long q = 0; q < S; q += 8) {
            __m512d x0 = _mm512_load_pd(x + q), x1 = _mm512_load_pd(x + S + q);
            __m512d x2 = _mm512_load_pd(x + 2 * S + q), x3 = _mm512_load_pd(x + 3 * S + q);
            __m512d x4 = _mm512_load_pd(x + 4 * S + q), x5 = _mm512_load_pd(x + 5 * S + q);
            __m512d x6 = _mm512_load_pd(x + 6 * S + q), x7 = _mm512_load_pd(x + 7 * S + q);
            R8_BODY(x0, x1, x2, x3, x4, x5, x6, x7);
            _mm512_store_pd(y + q, map_vec(u0, _mm512_load_pd(cf + q)));
            _mm512_store_pd(y + S + q, map_vec(u1, _mm512_load_pd(cf + S + q)));
            _mm512_store_pd(y + 2 * S + q, map_vec(u2, _mm512_load_pd(cf + 2 * S + q)));
            _mm512_store_pd(y + 3 * S + q, map_vec(u3, _mm512_load_pd(cf + 3 * S + q)));
            _mm512_store_pd(y + 4 * S + q, map_vec(u4, _mm512_load_pd(cf + 4 * S + q)));
            _mm512_store_pd(y + 5 * S + q, map_vec(u5, _mm512_load_pd(cf + 5 * S + q)));
            _mm512_store_pd(y + 6 * S + q, map_vec(u6, _mm512_load_pd(cf + 6 * S + q)));
            _mm512_store_pd(y + 7 * S + q, map_vec(u7, _mm512_load_pd(cf + 7 * S + q)));
        }
    }
}

/* final radix-2, n == 2, twiddle-free; optionally fused with the chain map */
static void stage_f2(const double *restrict x, double *restrict y, int s,
                     const double *restrict cf, int nt)
{
    (void)nt;
    const long S = 2L * s;
    const double *xa = x, *xb = x + S;
    double *ya = y, *yb = y + S;
    if (!cf) {
        for (long q = 0; q < S; q += 8) {
            __m512d a = _mm512_load_pd(xa + q), b = _mm512_load_pd(xb + q);
            _mm512_store_pd(ya + q, _mm512_add_pd(a, b));
            _mm512_store_pd(yb + q, _mm512_sub_pd(a, b));
        }
    } else {
        const double *ca = cf, *cb = cf + S;
        for (long q = 0; q < S; q += 8) {
            __m512d a = _mm512_load_pd(xa + q), b = _mm512_load_pd(xb + q);
            _mm512_store_pd(ya + q, map_vec(_mm512_add_pd(a, b), _mm512_load_pd(ca + q)));
            _mm512_store_pd(yb + q, map_vec(_mm512_sub_pd(a, b), _mm512_load_pd(cb + q)));
        }
    }
}

/* ---------------- all-in-register codelets for L = 32 and L = 64 ----------
 * The whole transform lives in 8 (L=32) or 16 (L=64) zmm registers; stages
 * pass values register-to-register (the stride-1 stage's store transpose
 * becomes an in-register transpose), so per-transform cost is loads + ALU +
 * stores with no intermediate memory traffic and no stage dispatch. */

#define TRANSP4(R0, R1, R2, R3, O0, O1, O2, O3)                                        \
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

/* stride-1 radix-4 butterfly quad with all three twiddles from a full table */
#define S1QUADT(A, B, C, D, TP, O0, O1, O2, O3)                                        \
    do {                                                                               \
        __m512d apc_ = _mm512_add_pd(A, C), amc_ = _mm512_sub_pd(A, C);                \
        __m512d bpd_ = _mm512_add_pd(B, D), bmd_ = _mm512_sub_pd(B, D);                \
        __m512d sw_ = _mm512_permute_pd(bmd_, PSWAP);                                  \
        __m512d q0_ = _mm512_add_pd(apc_, bpd_);                                       \
        __m512d q2_ = _mm512_sub_pd(apc_, bpd_);                                       \
        __m512d q1_ = _mm512_fmsubadd_pd(amc_, ONE, sw_);                              \
        __m512d q3_ = _mm512_fmaddsub_pd(amc_, ONE, sw_);                              \
        __m512d r1_ = cmul_bc(q1_, _mm512_load_pd(TP), _mm512_load_pd((TP) + 8));      \
        __m512d r2_ = cmul_bc(q2_, _mm512_load_pd((TP) + 16), _mm512_load_pd((TP) + 24)); \
        __m512d r3_ = cmul_bc(q3_, _mm512_load_pd((TP) + 32), _mm512_load_pd((TP) + 40)); \
        TRANSP4(q0_, r1_, r2_, r3_, O0, O1, O2, O3);                                   \
    } while (0)

/* one stride-1 radix-4 butterfly quad (4 twiddle lanes) + in-register transpose */
#define S1QUAD(A, B, C, D, W1R, W1P, O0, O1, O2, O3)                                   \
    do {                                                                               \
        __m512d apc_ = _mm512_add_pd(A, C), amc_ = _mm512_sub_pd(A, C);                \
        __m512d bpd_ = _mm512_add_pd(B, D), bmd_ = _mm512_sub_pd(B, D);                \
        __m512d sw_ = _mm512_permute_pd(bmd_, PSWAP);                                  \
        __m512d q0_ = _mm512_add_pd(apc_, bpd_);                                       \
        __m512d q2_ = _mm512_sub_pd(apc_, bpd_);                                       \
        __m512d q1_ = _mm512_fmsubadd_pd(amc_, ONE, sw_);                              \
        __m512d q3_ = _mm512_fmaddsub_pd(amc_, ONE, sw_);                              \
        __m512d w2r_ = _mm512_fmsub_pd(W1R, W1R, _mm512_mul_pd(W1P, W1P));             \
        __m512d w2p_ = _mm512_mul_pd(_mm512_add_pd(W1R, W1R), W1P);                    \
        __m512d w3r_ = _mm512_fnmadd_pd(W1P, w2p_, _mm512_mul_pd(W1R, w2r_));          \
        __m512d w3p_ = _mm512_fmadd_pd(W1R, w2p_, _mm512_mul_pd(w2r_, W1P));           \
        __m512d r1_ = cmul_bc(q1_, W1R, W1P);                                          \
        __m512d r2_ = cmul_bc(q2_, w2r_, w2p_);                                        \
        __m512d r3_ = cmul_bc(q3_, w3r_, w3p_);                                        \
        TRANSP4(q0_, r1_, r2_, r3_, O0, O1, O2, O3);                                   \
    } while (0)

/* FFT(32) on 8 registers: stride-1 radix-4 (two quads) then twiddle-free radix-8 */
#define FFT32_REGS(V0, V1, V2, V3, V4, V5, V6, V7, U)                                  \
    do {                                                                               \
        __m512d ya0_, ya1_, ya2_, ya3_, yb0_, yb1_, yb2_, yb3_;                        \
        S1QUADT(V0, V2, V4, V6, tf, ya0_, ya1_, ya2_, ya3_);                           \
        S1QUADT(V1, V3, V5, V7, tf + 48, yb0_, yb1_, yb2_, yb3_);                      \
        R8_BODY(ya0_, ya1_, ya2_, ya3_, yb0_, yb1_, yb2_, yb3_);                       \
        U##0 = u0; U##1 = u1; U##2 = u2; U##3 = u3;                                    \
        U##4 = u4; U##5 = u5; U##6 = u6; U##7 = u7;                                    \
    } while (0)

static void fft32_execute(const fft1d_plan *p, const double *restrict in,
                          double *restrict out, int batch)
{
    R8_CONSTS;
    const __m512i idxA = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    const __m512i idxB = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
    const double *tf = p->tws1full;
    for (int b = 0; b < batch; ++b) {
        const double *x = in + 64L * b;
        double *y = out + 64L * b;
        __m512d v0 = _mm512_load_pd(x), v1 = _mm512_load_pd(x + 8);
        __m512d v2 = _mm512_load_pd(x + 16), v3 = _mm512_load_pd(x + 24);
        __m512d v4 = _mm512_load_pd(x + 32), v5 = _mm512_load_pd(x + 40);
        __m512d v6 = _mm512_load_pd(x + 48), v7 = _mm512_load_pd(x + 56);
        __m512d o0, o1, o2, o3, o4, o5, o6, o7;
        FFT32_REGS(v0, v1, v2, v3, v4, v5, v6, v7, o);
        _mm512_store_pd(y, o0); _mm512_store_pd(y + 8, o1);
        _mm512_store_pd(y + 16, o2); _mm512_store_pd(y + 24, o3);
        _mm512_store_pd(y + 32, o4); _mm512_store_pd(y + 40, o5);
        _mm512_store_pd(y + 48, o6); _mm512_store_pd(y + 56, o7);
    }
}

static void fft32_chain(const fft1d_plan *p, const double *restrict x0,
                        const double *restrict c, double *restrict final_out,
                        int batch, int m)
{
    R8_CONSTS;
    const __m512i idxA = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    const __m512i idxB = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
    const double *tf = p->tws1full;
    for (int b = 0; b < batch; ++b) {
        const double *x = x0 + 64L * b;
        const double *cb = c + 64L * b;
        double *y = final_out + 64L * b;
        __m512d c0 = _mm512_load_pd(cb), c1 = _mm512_load_pd(cb + 8);
        __m512d c2 = _mm512_load_pd(cb + 16), c3 = _mm512_load_pd(cb + 24);
        __m512d c4 = _mm512_load_pd(cb + 32), c5 = _mm512_load_pd(cb + 40);
        __m512d c6 = _mm512_load_pd(cb + 48), c7 = _mm512_load_pd(cb + 56);
        __m512d v0 = _mm512_load_pd(x), v1 = _mm512_load_pd(x + 8);
        __m512d v2 = _mm512_load_pd(x + 16), v3 = _mm512_load_pd(x + 24);
        __m512d v4 = _mm512_load_pd(x + 32), v5 = _mm512_load_pd(x + 40);
        __m512d v6 = _mm512_load_pd(x + 48), v7 = _mm512_load_pd(x + 56);
        for (int step = 0; step < m; ++step) {
            __m512d o0, o1, o2, o3, o4, o5, o6, o7;
            FFT32_REGS(v0, v1, v2, v3, v4, v5, v6, v7, o);
            v0 = map_vec(o0, c0); v1 = map_vec(o1, c1);
            v2 = map_vec(o2, c2); v3 = map_vec(o3, c3);
            v4 = map_vec(o4, c4); v5 = map_vec(o5, c5);
            v6 = map_vec(o6, c6); v7 = map_vec(o7, c7);
        }
        _mm512_store_pd(y, v0); _mm512_store_pd(y + 8, v1);
        _mm512_store_pd(y + 16, v2); _mm512_store_pd(y + 24, v3);
        _mm512_store_pd(y + 32, v4); _mm512_store_pd(y + 40, v5);
        _mm512_store_pd(y + 48, v6); _mm512_store_pd(y + 56, v7);
    }
}

/* FFT(64) on 16 registers: stride-1 radix-4 (four quads), radix-4 with
 * broadcast twiddles, twiddle-free radix-4 final. Y[] indices are vectors of
 * 4 consecutive complexes. */
#define R4Q(A, B, C, D, O0, O1, O2, O3)                                                \
    do {                                                                               \
        __m512d apc_ = _mm512_add_pd(A, C), amc_ = _mm512_sub_pd(A, C);                \
        __m512d bpd_ = _mm512_add_pd(B, D), bmd_ = _mm512_sub_pd(B, D);                \
        __m512d sw_ = _mm512_permute_pd(bmd_, PSWAP);                                  \
        O0 = _mm512_add_pd(apc_, bpd_);                                                \
        O2 = _mm512_sub_pd(apc_, bpd_);                                                \
        O1 = _mm512_fmsubadd_pd(amc_, ONE, sw_);                                       \
        O3 = _mm512_fmaddsub_pd(amc_, ONE, sw_);                                       \
    } while (0)

static void fft64_execute(const fft1d_plan *p, const double *restrict in,
                          double *restrict out, int batch)
{
    const __m512d ONE = _mm512_set1_pd(1.0);
    const __m512i idxA = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    const __m512i idxB = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
    const double *tf = p->tws1full;
    const double *t2 = p->st[1].tw;  /* radix-4 n=16 stage: 9 doubles per p */
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
            __m512d a = _mm512_load_pd(x + 8L * j);
            __m512d bq = _mm512_load_pd(x + 8L * (j + 4));
            __m512d cq = _mm512_load_pd(x + 8L * (j + 8));
            __m512d dq = _mm512_load_pd(x + 8L * (j + 12));
            S1QUADT(a, bq, cq, dq, tf + 48 * j, Y[4 * j], Y[4 * j + 1], Y[4 * j + 2],
                    Y[4 * j + 3]);
        }
        /* stage 2: radix-4 n=16 s=4, broadcast twiddles; p'=0 twiddle-free */
        R4Q(Y[0], Y[4], Y[8], Y[12], Z[0], Z[1], Z[2], Z[3]);
        for (int pp = 1; pp < 4; ++pp) {
            __m512d z0, z1, z2, z3;
            R4Q(Y[pp], Y[pp + 4], Y[pp + 8], Y[pp + 12], z0, z1, z2, z3);
            Z[4 * pp] = z0;
            Z[4 * pp + 1] = cmul_bc(z1, g2r[pp][0], g2p[pp][0]);
            Z[4 * pp + 2] = cmul_bc(z2, g2r[pp][1], g2p[pp][1]);
            Z[4 * pp + 3] = cmul_bc(z3, g2r[pp][2], g2p[pp][2]);
        }
        /* stage 3: twiddle-free radix-4, s=16 */
        for (int j = 0; j < 4; ++j) {
            __m512d z0, z1, z2, z3;
            R4Q(Z[j], Z[j + 4], Z[j + 8], Z[j + 12], z0, z1, z2, z3);
            _mm512_store_pd(yo + 8L * j, z0);
            _mm512_store_pd(yo + 8L * (j + 4), z1);
            _mm512_store_pd(yo + 8L * (j + 8), z2);
            _mm512_store_pd(yo + 8L * (j + 12), z3);
        }
    }
}

static void fft64_chain(const fft1d_plan *p, const double *restrict x0,
                        const double *restrict c, double *restrict final_out,
                        int batch, int m)
{
    const __m512d ONE = _mm512_set1_pd(1.0);
    const __m512i idxA = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    const __m512i idxB = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
    const double *tf = p->tws1full;
    const double *t2 = p->st[1].tw;
    __m512d g2r[4][3], g2p[4][3];
    for (int pp = 1; pp < 4; ++pp)
        for (int r = 0; r < 3; ++r) {
            g2r[pp][r] = _mm512_set1_pd(t2[9 * pp + r]);
            g2p[pp][r] = _mm512_broadcast_f64x2(_mm_loadu_pd(t2 + 9 * pp + 3 + 2 * r));
        }
    for (int b = 0; b < batch; ++b) {
        const double *x = x0 + 128L * b;
        const double *cb = c + 128L * b;
        double *yo = final_out + 128L * b;
        __m512d V[16];
        for (int j = 0; j < 16; ++j) V[j] = _mm512_load_pd(x + 8L * j);
        for (int step = 0; step < m; ++step) {
            __m512d Y[16], Z[16];
            for (int j = 0; j < 4; ++j)
                S1QUADT(V[j], V[j + 4], V[j + 8], V[j + 12], tf + 48 * j, Y[4 * j],
                        Y[4 * j + 1], Y[4 * j + 2], Y[4 * j + 3]);
            R4Q(Y[0], Y[4], Y[8], Y[12], Z[0], Z[1], Z[2], Z[3]);
            for (int pp = 1; pp < 4; ++pp) {
                __m512d z0, z1, z2, z3;
                R4Q(Y[pp], Y[pp + 4], Y[pp + 8], Y[pp + 12], z0, z1, z2, z3);
                Z[4 * pp] = z0;
                Z[4 * pp + 1] = cmul_bc(z1, g2r[pp][0], g2p[pp][0]);
                Z[4 * pp + 2] = cmul_bc(z2, g2r[pp][1], g2p[pp][1]);
                Z[4 * pp + 3] = cmul_bc(z3, g2r[pp][2], g2p[pp][2]);
            }
            for (int j = 0; j < 4; ++j) {
                __m512d z0, z1, z2, z3;
                R4Q(Z[j], Z[j + 4], Z[j + 8], Z[j + 12], z0, z1, z2, z3);
                V[j] = map_vec(z0, _mm512_load_pd(cb + 8L * j));
                V[j + 4] = map_vec(z1, _mm512_load_pd(cb + 8L * (j + 4)));
                V[j + 8] = map_vec(z2, _mm512_load_pd(cb + 8L * (j + 8)));
                V[j + 12] = map_vec(z3, _mm512_load_pd(cb + 8L * (j + 12)));
            }
        }
        for (int j = 0; j < 16; ++j) _mm512_store_pd(yo + 8L * j, V[j]);
    }
}

#else /* ------------------------------------------------- scalar fallback */

static void stage_s1(const double *restrict xd_, double *restrict yd_, int n,
                     const double *restrict tw)
{
    (void)tw;
    const double _Complex *x = (const double _Complex *)xd_;
    double _Complex *y = (double _Complex *)yd_;
    const int m = n / 4;
    for (int p = 0; p < m; ++p) {
        double _Complex w1 = cexp(-2.0 * I * M_PI * p / n);
        double _Complex w2 = w1 * w1, w3 = w2 * w1;
        double _Complex a = x[p], b = x[p + m], c = x[p + 2 * m], d = x[p + 3 * m];
        double _Complex apc = a + c, amc = a - c, bpd = b + d, jb = I * (b - d);
        y[4 * p] = apc + bpd;
        y[4 * p + 1] = w1 * (amc - jb);
        y[4 * p + 2] = w2 * (apc - bpd);
        y[4 * p + 3] = w3 * (amc + jb);
    }
}

static void stage_gen(const double *restrict xd_, double *restrict yd_, int n, int s,
                      const double *restrict tw)
{
    (void)tw;
    const double _Complex *x = (const double _Complex *)xd_;
    double _Complex *y = (double _Complex *)yd_;
    const int m = n / 4;
    for (int p = 0; p < m; ++p) {
        double _Complex w1 = cexp(-2.0 * I * M_PI * p / n);
        double _Complex w2 = w1 * w1, w3 = w2 * w1;
        for (int q = 0; q < s; ++q) {
            double _Complex a = x[q + s * p], b = x[q + s * (p + m)];
            double _Complex c = x[q + s * (p + 2 * m)], d = x[q + s * (p + 3 * m)];
            double _Complex apc = a + c, amc = a - c, bpd = b + d, jb = I * (b - d);
            y[q + s * (4 * p)] = apc + bpd;
            y[q + s * (4 * p + 1)] = w1 * (amc - jb);
            y[q + s * (4 * p + 2)] = w2 * (apc - bpd);
            y[q + s * (4 * p + 3)] = w3 * (amc + jb);
        }
    }
}

static void r8_core(const double _Complex *v, double _Complex *u)
{
    const double cq = 0.70710678118654752440;
    double _Complex s0 = v[0] + v[4], s1 = v[1] + v[5], s2 = v[2] + v[6], s3 = v[3] + v[7];
    double _Complex d0 = v[0] - v[4], d1 = v[1] - v[5], d2 = v[2] - v[6], d3 = v[3] - v[7];
    double _Complex e1 = d1 * (cq - cq * I), e2 = d2 * (-I), e3 = d3 * (-cq - cq * I);
    double _Complex apc = s0 + s2, amc = s0 - s2, bpd = s1 + s3, jb = I * (s1 - s3);
    u[0] = apc + bpd; u[2] = amc - jb; u[4] = apc - bpd; u[6] = amc + jb;
    apc = d0 + e2; amc = d0 - e2; bpd = e1 + e3; jb = I * (e1 - e3);
    u[1] = apc + bpd; u[3] = amc - jb; u[5] = apc - bpd; u[7] = amc + jb;
}

static void stage_gen8(const double *restrict xd_, double *restrict yd_, int n, int s,
                       const double *restrict tw)
{
    (void)tw;
    const double _Complex *x = (const double _Complex *)xd_;
    double _Complex *y = (double _Complex *)yd_;
    const int m = n / 8;
    for (int p = 0; p < m; ++p) {
        double _Complex w[8];
        w[0] = 1.0;
        w[1] = cexp(-2.0 * I * M_PI * p / n);
        for (int r = 2; r < 8; ++r) w[r] = w[r - 1] * w[1];
        for (int q = 0; q < s; ++q) {
            double _Complex v[8], u[8];
            for (int r = 0; r < 8; ++r) v[r] = x[q + s * (p + r * m)];
            r8_core(v, u);
            for (int r = 0; r < 8; ++r) y[q + s * (8 * p + r)] = w[r] * u[r];
        }
    }
}

static inline double _Complex map_sc(double _Complex v, double _Complex c)
{
    double _Complex z = v + c;
    return z / (1.0 + cabs(z));
}

static void stage_f4(const double *restrict xd_, double *restrict yd_, int s,
                     const double *restrict cfd_, int nt)
{
    (void)nt;
    const double _Complex *x = (const double _Complex *)xd_;
    const double _Complex *cf = (const double _Complex *)cfd_;
    double _Complex *y = (double _Complex *)yd_;
    for (int q = 0; q < s; ++q) {
        double _Complex a = x[q], b = x[q + s], c = x[q + 2 * s], d = x[q + 3 * s];
        double _Complex apc = a + c, amc = a - c, bpd = b + d, jb = I * (b - d);
        double _Complex r0 = apc + bpd, r1 = amc - jb, r2 = apc - bpd, r3 = amc + jb;
        if (cf) {
            y[q] = map_sc(r0, cf[q]);
            y[q + s] = map_sc(r1, cf[q + s]);
            y[q + 2 * s] = map_sc(r2, cf[q + 2 * s]);
            y[q + 3 * s] = map_sc(r3, cf[q + 3 * s]);
        } else {
            y[q] = r0; y[q + s] = r1; y[q + 2 * s] = r2; y[q + 3 * s] = r3;
        }
    }
}

static void stage_f8(const double *restrict xd_, double *restrict yd_, int s,
                     const double *restrict cfd_, int nt)
{
    (void)nt;
    const double _Complex *x = (const double _Complex *)xd_;
    const double _Complex *cf = (const double _Complex *)cfd_;
    double _Complex *y = (double _Complex *)yd_;
    for (int q = 0; q < s; ++q) {
        double _Complex v[8], u[8];
        for (int r = 0; r < 8; ++r) v[r] = x[q + r * s];
        r8_core(v, u);
        for (int r = 0; r < 8; ++r)
            y[q + r * s] = cf ? map_sc(u[r], cf[q + r * s]) : u[r];
    }
}

static void stage_f2(const double *restrict xd_, double *restrict yd_, int s,
                     const double *restrict cfd_, int nt)
{
    (void)nt;
    const double _Complex *x = (const double _Complex *)xd_;
    const double _Complex *cf = (const double _Complex *)cfd_;
    double _Complex *y = (double _Complex *)yd_;
    for (int q = 0; q < s; ++q) {
        double _Complex r0 = x[q] + x[q + s], r1 = x[q] - x[q + s];
        if (cf) { y[q] = map_sc(r0, cf[q]); y[q + s] = map_sc(r1, cf[q + s]); }
        else { y[q] = r0; y[q + s] = r1; }
    }
}

#endif /* D1_AVX512 */

/* ------------------------------------------------------------ stage driver */

/* Runs all T stages src -> ... -> final.  Intermediate stages ping-pong between
 * bufodd (stage indices 0,2,..) and bufeven (1,3,..); the last stage always
 * writes `final`, with the chain map fused when cf != NULL.  Chosen so no stage
 * ever writes the buffer it reads. */
static void run_stages(const fft1d_plan *p, const double *src, double *bufodd,
                       double *bufeven, double *final_, const double *cf, int nt)
{
    const double *sp = src;
    for (int t = 0; t < p->T; ++t) {
        const stage_t *st = &p->st[t];
        int last = (t == p->T - 1);
        double *d = last ? final_ : ((t & 1) ? bufeven : bufodd);
        const double *cft = last ? cf : NULL;
        switch (st->type) {
        case ST_S1: stage_s1(sp, d, st->n, st->tw); break;
        case ST_GEN: stage_gen(sp, d, st->n, st->s, st->tw); break;
        case ST_GEN8: stage_gen8(sp, d, st->n, st->s, st->tw); break;
        case ST_F4: stage_f4(sp, d, st->s, cft, nt); break;
        case ST_F2: stage_f2(sp, d, st->s, cft, nt); break;
        case ST_F8: stage_f8(sp, d, st->s, cft, nt); break;
        }
        sp = d;
    }
}

/* --------------------------------------------------------------- plan/exec */

fft1d_plan *fft1d_create(int L, int batch)
{
    if (!fft1d_supports(L) || batch < 1) return NULL;
    fft1d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;

    /* schedule: radix-4 first stage (s == 1, per-lane twiddles), then radix-8
     * stages while possible, a radix-4 fixup at n == 16, and a twiddle-free
     * final codelet (n = 8, 4 or 2).  Fewest passes over the data:
     * 32 -> 2, 64/128/256 -> 3, 1024 -> 4, 4096/16384 -> 5. */
    int n = L, s = 1, T = 0;
    size_t twd = 0;
    p->st[T].n = n; p->st[T].s = s; p->st[T].type = ST_S1;
    twd += (size_t)(n / 4) * 12;
    ++T; n >>= 2; s <<= 2;
#ifdef D1_R4ONLY /* A/B harness: pure radix-4 pipeline after the first stage */
    while (n >= 8) {
        p->st[T].n = n; p->st[T].s = s; p->st[T].type = ST_GEN;
        twd += (size_t)(n / 4) * 9;
        ++T; n >>= 2; s <<= 2;
    }
#endif
    /* When log2(n) % 3 != 0 a radix-4 stage is owed somewhere anyway; spend it
     * at s == 4 so every radix-8 stage runs at s >= 16, where its 14 twiddle
     * broadcasts amortize over >= 4 vector iterations.  Same total pass count. */
    {
        int k = 0;
        for (int nn = n; nn > 1; nn >>= 1) ++k;
        if (n > 16 && k % 3 != 0) {
            p->st[T].n = n; p->st[T].s = s; p->st[T].type = ST_GEN;
            twd += (size_t)(n / 4) * 9;
            ++T; n >>= 2; s <<= 2;
        }
    }
    while (n > 16) {
        p->st[T].n = n; p->st[T].s = s; p->st[T].type = ST_GEN8;
        twd += (size_t)(n / 8) * 21;
        ++T; n >>= 3; s <<= 3;
    }
    if (n == 16) {
        p->st[T].n = n; p->st[T].s = s; p->st[T].type = ST_GEN;
        twd += (size_t)(n / 4) * 9;
        ++T; n >>= 2; s <<= 2;
    }
    p->st[T].n = n;
    p->st[T].s = s;
    p->st[T].type = (n == 8) ? ST_F8 : (n == 4) ? ST_F4 : ST_F2;
    ++T;
    p->T = T;
    /* Non-temporal final-stage stores were tried for the big batched cells and
     * MEASURED 3x SLOWER (L=1024 B=512: 5.95us vs 2.1us on wallaby): in+out of
     * every graded batched cell fits L3 (<= 32 MB vs 60/24 MB), so regular
     * stores hit L3 while NT forces DRAM writes.  Keep the path, disabled. */
    p->nt = 0;

    twd += 8; /* padding so vector loads at the tail stay in bounds */
    if (posix_memalign((void **)&p->twmem, 64, twd * sizeof(double)) ||
        posix_memalign((void **)&p->scratch, 64, 2 * (size_t)L * sizeof(double)) ||
        posix_memalign((void **)&p->state, 64, 2 * (size_t)L * sizeof(double))) {
        free(p->twmem); free(p->scratch); free(p->state); free(p);
        return NULL;
    }
    memset(p->twmem, 0, twd * sizeof(double));

    double *w = p->twmem;
    for (int t = 0; t < T; ++t) {
        stage_t *st = &p->st[t];
        int nn = st->n, m = nn / 4;
        if (st->type == ST_S1) {
            st->tw = w;
            for (int q = 0; q < m; ++q) {
                int g = (q / 4) * 48, j = q % 4;
                for (int r = 1; r <= 3; ++r) {
                    long double th = -2.0L * PIL * (long double)((long)q * r % nn) / (long double)nn;
                    double cr = (double)cosl(th), si = (double)sinl(th);
                    w[g + 16 * (r - 1) + 2 * j] = cr;
                    w[g + 16 * (r - 1) + 2 * j + 1] = cr;
                    w[g + 16 * (r - 1) + 8 + 2 * j] = -si;
                    w[g + 16 * (r - 1) + 8 + 2 * j + 1] = si;
                }
            }
            w += (size_t)m * 12;
        } else if (st->type == ST_GEN) {
            st->tw = w;
            for (int q = 0; q < m; ++q) {
                for (int r = 1; r <= 3; ++r) {
                    long double th = -2.0L * PIL * (long double)((long)q * r % nn) / (long double)nn;
                    w[9 * q + (r - 1)] = (double)cosl(th);
                    w[9 * q + 3 + 2 * (r - 1)] = -(double)sinl(th);
                    w[9 * q + 4 + 2 * (r - 1)] = (double)sinl(th);
                }
            }
            w += (size_t)m * 9;
        } else if (st->type == ST_GEN8) {
            int m8 = nn / 8;
            st->tw = w;
            for (int q = 0; q < m8; ++q) {
                for (int r = 1; r <= 7; ++r) {
                    long double th = -2.0L * PIL * (long double)((long)q * r % nn) / (long double)nn;
                    w[21 * q + (r - 1)] = (double)cosl(th);
                    w[21 * q + 7 + 2 * (r - 1)] = -(double)sinl(th);
                    w[21 * q + 8 + 2 * (r - 1)] = (double)sinl(th);
                }
            }
            w += (size_t)m8 * 21;
        }
    }
    if (L == 32 || L == 64) {
        /* full dup-format first-stage table for the in-register codelets:
         * per 4-p group: [w1r|w1p|w2r|w2p|w3r|w3p] x 8 doubles */
        int m = L / 4;
        if (posix_memalign((void **)&p->tws1full, 64, (size_t)(m / 4) * 48 * sizeof(double))) {
            fft1d_destroy(p);
            return NULL;
        }
        for (int q = 0; q < m; ++q) {
            int g = q / 4, j = q % 4;
            for (int r = 1; r <= 3; ++r) {
                long double th = -2.0L * PIL * (long double)(q * r % L) / (long double)L;
                double *base = p->tws1full + g * 48 + (r - 1) * 16;
                base[2 * j] = (double)cosl(th);
                base[2 * j + 1] = (double)cosl(th);
                base[8 + 2 * j] = -(double)sinl(th);
                base[8 + 2 * j + 1] = (double)sinl(th);
            }
        }
    }
    return p;
}

void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    const size_t L = (size_t)p->L;
#ifdef D1_AVX512
    if (p->L == 32) { fft32_execute(p, (const double *)in, (double *)out, p->batch); return; }
    if (p->L == 64) { fft64_execute(p, (const double *)in, (double *)out, p->batch); return; }
#endif
    for (int b = 0; b < p->batch; ++b) {
        const double *src = (const double *)(in + (size_t)b * L);
        double *dst = (double *)(out + (size_t)b * L);
        double *bo = (p->T & 1) ? dst : p->scratch;
        double *be = (p->T & 1) ? p->scratch : dst;
        run_stages(p, src, bo, be, dst, NULL, p->nt);
    }
#ifdef D1_AVX512
    if (p->nt) _mm_sfence();
#endif
}

/* Fused m-step map chain: state <- (FFT(state)+c) / (1+|FFT(state)+c|).
 * The chain is independent per transform, so each transform runs all m steps
 * back-to-back while it is cache-resident; final_out doubles as a ping buffer
 * until the last copy. */
void fft1d_chain(fft1d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    const size_t L = (size_t)p->L;
#ifdef D1_AVX512
    if (p->L == 32) { fft32_chain(p, (const double *)x0, (const double *)c, (double *)final_out, p->batch, m); return; }
    if (p->L == 64) { fft64_chain(p, (const double *)x0, (const double *)c, (double *)final_out, p->batch, m); return; }
#endif
    for (int b = 0; b < p->batch; ++b) {
        const double *xb = (const double *)(x0 + (size_t)b * L);
        const double *cb = (const double *)(c + (size_t)b * L);
        double *fb = (double *)(final_out + (size_t)b * L);
        for (int step = 0; step < m; ++step) {
            const double *src = step ? p->state : xb;
            run_stages(p, src, fb, p->scratch, p->state, cb, 0);
        }
        memcpy(fb, p->state, L * sizeof(double _Complex));
    }
}

void fft1d_destroy(fft1d_plan *p)
{
    if (!p) return;
    free(p->twmem);
    free(p->scratch);
    free(p->state);
    free(p->tws1full);
    free(p);
}
