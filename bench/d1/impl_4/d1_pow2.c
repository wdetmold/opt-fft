/* d1_pow2: power-of-two 1D complex-double FFT.
 *
 * Algorithm: Stockham autosort (no bit-reversal, ping-pong out-of-place), DIF.
 * Two engines:
 *   - L >= 128 (r2): a blocked SPLIT-COMPLEX radix-8 pipeline.  Interleaved
 *     AoS pays one port-5 shuffle per complex multiply, and on the scoring
 *     node every 512-bit shuffle shares port 5 with an FMA unit.  So the
 *     stride-1 first stage writes [8 re | 8 im] blocks directly (its store
 *     transpose costs the same either way), one paired-p radix-8/4 stage
 *     covers s == 4, all further stages are pure vertical FMA with broadcast
 *     twiddles and ZERO shuffles, and the twiddle-free final re-interleaves
 *     on store.  Twiddles from long-double sincos at plan time (M_PI-based
 *     angles carry a BIASED ~2e-16 phase error that a 10^4-step chain
 *     amplifies past the 1e-10 gate).
 *   - L <= 64: interleaved-AoS in-register codelets (whole transform in 8/16
 *     zmm), cmul = fmadd(swap(u), wpair, u*wre).
 *
 * fft1d_chain (the fused map chain) is owned:
 *   - batch >= 8, L <= 2048: across-batch SoA groups of 8 transforms (taken
 *     from d1_batchlane's r1 design): lane j = transform j, one 8x8 transpose
 *     in/out per group per chain, every step shuffle-free, group working set
 *     cache-resident.  Above 2048 the 3 x 16L-double group set spills L2 and
 *     the per-transform path below wins (measured 2x at 16384).
 *   - otherwise per-transform m-step blocking: each transform runs ALL m
 *     steps while its working set stays cache-resident; libraries through
 *     the driver fallback stream the full B*L batch three times per step.
 *   - the map z/(1+|z|) is fused into the final butterfly stage (no extra
 *     read+write pass) and computed with rsqrt14+rcp14 plus two Newton steps
 *     each; in split form it needs no pair-swap shuffle.  (The r1/r2
 *     exact-residual refinements are compile-time optional: with long-double
 *     twiddles they moved graded gates by <25% and cost 10-27% of chain time.)
 *
 * r3: at L=4096 the two middle radix-8 stages fuse into one L1-tiled radix-64
 * pass.  Execute uses non-temporal final stores when in+out exceeds the
 * scoring node's 24 MB L3 (4096xB256, 16384xB64), ping-ponging through two
 * private scratches so the NT target is never dirty in cache (NT into freshly
 * dirtied lines measured 1.5x SLOWER than no NT).
 *
 * r4 (first round with the scoring node itself leased -- every number below
 * is a80n0/ICX): PMU showed the large-L cells are L1-fill and L2-capacity
 * bound, not port bound: at 16384 B=1 the five passes retired 60k
 * l1d.replacement per transform (one fill per 3 cycles = the entire runtime),
 * and the dup-format twiddle tables (s1s 384 KB + s48 229 KB at 16384) pushed
 * the working set past the 1.25 MB L2 (46k cycles of stalls_l2_miss).  Fixes:
 *   - COMPACT (c,s)-pair twiddle tables for s1s/s48/s44 at L >= 1024 (2x/4x
 *     smaller), expanded in-register: s1s cmul re-expressed as
 *     fmaddsub(u, dup(c), swap(u)*dup(s)); s48/s44 pair twiddles rebuilt with
 *     one broadcast_f64x4 + two permutexvar.  16384 B=1: 49.8 -> 38.4 us.
 *     (Same trick at L <= 512 measured SLOWER -- port 5 dups cost more than
 *     the tiny-table savings -- so the dup format stays below 1024.)
 *   - the stride-1 first stage FUSES with the s == 4 stage through a 256-
 *     double L1 tile at L >= 4096 (ST_SX48/ST_SX44): one array pass instead
 *     of two, bitwise-identical output.  With the r3 radix-64 fusion (now on
 *     at 16384 too -- it is a win once the first-stage pair is fused) 4096
 *     and 16384 run in THREE passes.  16384 B=1 lands at ~33.9 us vs FFTW
 *     patient 32.6 on the same core (was 52.9/1.64x in r3).
 * A full Bailey four-step engine (column FFTs in an L1 tile, fused diagonal
 * twiddle, transposed mid array) was built and measured this round: correct,
 * fills cut 60k -> 38k, but its tile ladder doubled the load count (short
 * q-loops re-broadcast twiddles per group) and it LOST to the compacted
 * Stockham everywhere (16384 B=1: 52.7 vs 38.4) -- removed, see the r4
 * strategy record before rebuilding it.
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

enum { ST_S1, ST_GEN, ST_GEN8, ST_F4, ST_F2, ST_F8,
       ST_S1S, ST_S48, ST_S44, ST_SS8, ST_SF8, ST_SF4, ST_SS64,
       ST_SX48, ST_SX44 };

typedef struct {
    int type, n, s;
    const double *tw;
    const double *tw2;  /* ST_SS64: table of the second (fused) radix-8 level */
} stage_t;

struct fft1d_plan {
    int L, batch, T;
    stage_t st[10];
    double *twmem;
    double *tile;      /* ST_SS64 L1 tile: 128*s doubles (64 blocks of s complexes) */
    double *scratch;   /* 2L doubles: ping-pong buffer */
    double *scratch2;  /* nt only: second ping-pong, so NT final stores hit a
                        * buffer with no dirty cached lines of its own */
    double *state;     /* 2L doubles: chain state for one transform */
    double *tws1full;  /* L = 32/64 codelets: full dup-format w,w^2,w^3 table */
    int nt;            /* stream (non-temporal) final-stage stores in execute */
    int cmpt;          /* compact (c,s)-pair twiddle tables in s1s/s48/s44 */
    int fastmap;       /* L <= 128: drop the map's exact-residual refinements */
    /* across-batch SoA chain path (idea taken from d1_batchlane): groups of 8
     * transforms, zmm lane = batch index, split-complex planes.  Zero shuffles
     * per chain step; one 8x8 transpose in and out per group per chain. */
    int soa_T;             /* stages of the pure radix-4 SoA ladder (0 = off) */
    double *soa_twmem;     /* 6 doubles per butterfly: (cr,si) x w^1,w^2,w^3 */
    const double *soa_tw[16];
    double *soa_a, *soa_b, *soa_c; /* 2*L*8 doubles each: re plane, im plane */
};


const char *fft1d_name(void) { return "d1_pow2"; }
const char *fft1d_description(void)
{
    return "Stockham autosort DIF: blocked split-complex radix-8 engine at L>=128 "
           "(zero-shuffle middle stages, conversion fused into the stride-1 stage), "
           "compact (c,s)-pair twiddle tables at L>=1024 (halved L2 footprint -- the "
           "ICX large-L bottleneck was table-bloated L1/L2 fill traffic, not ports), "
           "first-stage pair fused through an L1 tile at L>=4096 and the two middle "
           "radix-8 stages fused into a radix-64 tile pass at 4096/16384 (three "
           "array passes total), in-register AoS codelets at L=32/64, NT final "
           "stores above the scoring node's L3; chains: across-batch SoA groups of "
           "8 (L<=2048, from d1_batchlane) or per-transform blocking, with a "
           "2-Newton rsqrt/rcp map fused into the final stage in split form";
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
 * Accuracy history (r1): plain 2-Newton rsqrt/rcp measured 6.7e-10 at the
 * unlucky L=128 m=30000 B=8 config -- but refining the map only moved that to
 * 6.1e-10; the real culprit was BIASED M_PI TWIDDLES (long-double tables:
 * 1.56e-10).  With exact twiddles the graded gates sit 17x-64000x under the
 * 1e-10 floor either way, so r3 default is the fast 2-Newton map. */
/* `precise` adds the exact-residual refinements (Heron on sqrt, residual on
 * the final quotient).  r1's accuracy fight showed the dominant chain error
 * was BIASED TWIDDLES, not these ~1-ulp map terms (2NR-only: 6.7e-10, +map
 * refinements: 6.1e-10, +long-double twiddles: 1.56e-10 at the unlucky
 * config) -- so with exact twiddles the graded gates at L <= 128 sit at
 * 1.8e-15..2.9e-12 vs the 1e-10 floor and afford the fast map.  L >= 1024
 * keeps the precise map (gate margin at 1024 B=1 is only ~20x). */
static inline __m512d map_vec_p(__m512d v, __m512d cv, int precise)
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
    if (precise)
        u = _mm512_fmadd_pd(_mm512_mul_pd(half, r), _mm512_fnmadd_pd(u, u, s), u);
    __m512d t = _mm512_add_pd(one, u);                          /* 1 + |z| */
    __m512d rc = _mm512_rcp14_pd(t);
    rc = _mm512_mul_pd(rc, _mm512_fnmadd_pd(t, rc, two));
    rc = _mm512_fmadd_pd(rc, _mm512_fnmadd_pd(t, rc, one), rc);
    __m512d q = _mm512_mul_pd(z, rc);
    if (!precise) return q;
    return _mm512_fmadd_pd(_mm512_fnmadd_pd(q, t, z), rc, q);
}
static inline __m512d map_vec(__m512d v, __m512d cv) { return map_vec_p(v, cv, 1); }

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
    const int pr = !p->fastmap;
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
            v0 = map_vec_p(o0, c0, pr); v1 = map_vec_p(o1, c1, pr);
            v2 = map_vec_p(o2, c2, pr); v3 = map_vec_p(o3, c3, pr);
            v4 = map_vec_p(o4, c4, pr); v5 = map_vec_p(o5, c5, pr);
            v6 = map_vec_p(o6, c6, pr); v7 = map_vec_p(o7, c7, pr);
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
    const int pr = !p->fastmap;
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
                V[j] = map_vec_p(z0, _mm512_load_pd(cb + 8L * j), pr);
                V[j + 4] = map_vec_p(z1, _mm512_load_pd(cb + 8L * (j + 4)), pr);
                V[j + 8] = map_vec_p(z2, _mm512_load_pd(cb + 8L * (j + 8)), pr);
                V[j + 12] = map_vec_p(z3, _mm512_load_pd(cb + 8L * (j + 12)), pr);
            }
        }
        for (int j = 0; j < 16; ++j) _mm512_store_pd(yo + 8L * j, V[j]);
    }
}

static inline void soa_map_p(__m512d *vr, __m512d *vi, __m512d cr, __m512d ci, int precise);

/* ---------------- split-complex single-transform engine (L >= 128) --------
 * Interleaved AoS pays one shuffle per complex multiply, and on the scoring
 * node (Ice Lake) every 512-bit shuffle lands on port 5, which is also an FMA
 * port.  So after the (unavoidably shuffling) stride-1 first stage, the
 * transform runs in split-complex planes: the second stage deinterleaves as
 * it loads (2 shuffles per 8 complexes), the middle stages are pure vertical
 * FMA with broadcast twiddles and zero shuffles, and the twiddle-free final
 * stage re-interleaves as it stores (map fused in split form for chains).
 * Split data is BLOCKED AoSoA -- blocks of 8 complexes as [8 re | 8 im] --
 * not two big planes: with planes every radix-8 stage runs 16 read + 16
 * write streams, which exhausts the fill buffers once a stage streams from
 * L2 (MEASURED: plane format lost 42% at L=4096 B=1, 9.6 vs 6.7 us AoS;
 * blocks keep the AoS engine's 8+8 streams).  Complex index c lives at
 * double offset 2c - (c&7) for re and +8 for im. */

#define SCMUL(RR, RI, CR, SI)                                                          \
    do {                                                                               \
        __m512d tr_ = _mm512_fnmadd_pd(RI, SI, _mm512_mul_pd(RR, CR));                 \
        RI = _mm512_fmadd_pd(RR, SI, _mm512_mul_pd(RI, CR));                           \
        RR = tr_;                                                                      \
    } while (0)

/* split radix-8 DIF butterfly: (x0r..x7r, x0i..x7i) -> u0r..u7r/u0i..u7i */
#define R8S_BODY(CQ)                                                                   \
    __m512d s0r = _mm512_add_pd(x0r, x4r), s0i = _mm512_add_pd(x0i, x4i);              \
    __m512d s1r = _mm512_add_pd(x1r, x5r), s1i = _mm512_add_pd(x1i, x5i);              \
    __m512d s2r = _mm512_add_pd(x2r, x6r), s2i = _mm512_add_pd(x2i, x6i);              \
    __m512d s3r = _mm512_add_pd(x3r, x7r), s3i = _mm512_add_pd(x3i, x7i);              \
    __m512d d0r = _mm512_sub_pd(x0r, x4r), d0i = _mm512_sub_pd(x0i, x4i);              \
    __m512d d1r = _mm512_sub_pd(x1r, x5r), d1i = _mm512_sub_pd(x1i, x5i);              \
    __m512d d2r = _mm512_sub_pd(x2r, x6r), d2i = _mm512_sub_pd(x2i, x6i);              \
    __m512d d3r = _mm512_sub_pd(x3r, x7r), d3i = _mm512_sub_pd(x3i, x7i);              \
    __m512d apcr = _mm512_add_pd(s0r, s2r), apci = _mm512_add_pd(s0i, s2i);            \
    __m512d amcr = _mm512_sub_pd(s0r, s2r), amci = _mm512_sub_pd(s0i, s2i);            \
    __m512d bpdr = _mm512_add_pd(s1r, s3r), bpdi = _mm512_add_pd(s1i, s3i);            \
    __m512d bmdr = _mm512_sub_pd(s1r, s3r), bmdi = _mm512_sub_pd(s1i, s3i);            \
    __m512d u0r = _mm512_add_pd(apcr, bpdr), u0i = _mm512_add_pd(apci, bpdi);          \
    __m512d u4r = _mm512_sub_pd(apcr, bpdr), u4i = _mm512_sub_pd(apci, bpdi);          \
    __m512d u2r = _mm512_add_pd(amcr, bmdi), u2i = _mm512_sub_pd(amci, bmdr);          \
    __m512d u6r = _mm512_sub_pd(amcr, bmdi), u6i = _mm512_add_pd(amci, bmdr);          \
    __m512d e1r = _mm512_mul_pd(CQ, _mm512_add_pd(d1r, d1i));                          \
    __m512d e1i = _mm512_mul_pd(CQ, _mm512_sub_pd(d1i, d1r));                          \
    __m512d e3r = _mm512_mul_pd(CQ, _mm512_sub_pd(d3i, d3r));                          \
    __m512d e3i = _mm512_mul_pd(CQ, _mm512_add_pd(d3r, d3i));                          \
    /* e3i above is -im(e3); fold the sign into the +/- combines below */             \
    __m512d apor = _mm512_add_pd(d0r, d2i), apoi = _mm512_sub_pd(d0i, d2r);            \
    __m512d amor = _mm512_sub_pd(d0r, d2i), amoi = _mm512_add_pd(d0i, d2r);            \
    __m512d bpor = _mm512_add_pd(e1r, e3r), bpoi = _mm512_sub_pd(e1i, e3i);            \
    __m512d bmor = _mm512_sub_pd(e1r, e3r), bmoi = _mm512_add_pd(e1i, e3i);            \
    __m512d u1r = _mm512_add_pd(apor, bpor), u1i = _mm512_add_pd(apoi, bpoi);          \
    __m512d u5r = _mm512_sub_pd(apor, bpor), u5i = _mm512_sub_pd(apoi, bpoi);          \
    __m512d u3r = _mm512_add_pd(amor, bmoi), u3i = _mm512_sub_pd(amoi, bmor);          \
    __m512d u7r = _mm512_sub_pd(amor, bmoi), u7i = _mm512_add_pd(amoi, bmor)

#define DEINT_IDX                                                                      \
    const __m512i IRE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);                  \
    const __m512i IIM = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15)
#define INT_IDX                                                                        \
    const __m512i ILO = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);                   \
    const __m512i IHI = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15)

/* stride-1 radix-4 first stage that writes BLOCKED SPLIT output directly:
 * identical butterfly + per-lane dup-format twiddles as stage_s1, but the
 * output transpose targets [8 re | 8 im] blocks -- same shuffle count (8 per
 * 16 complexes) as stage_s1's AoS store transpose, so the AoS->split
 * conversion costs nothing and no separate conversion stage exists. */
/* one stride-1 radix-4 quad (4 p's, 16 complexes) of the first stage,
 * writing two [8 re | 8 im] blocks to yq (contiguous 32 doubles): the body of
 * stage_s1s, split out so the fused first-stage-pair passes can aim it at an
 * L1 tile. */
static inline void s1s_quad(const double *restrict x, double *restrict yq,
                            int m, int p, const double *restrict tw, int compact)
{
    const __m512d ONE = _mm512_set1_pd(1.0);
    const __m512i TRE = _mm512_setr_epi64(0, 8, 2, 10, 4, 12, 6, 14);
    const __m512i TIM = _mm512_setr_epi64(1, 9, 3, 11, 5, 13, 7, 15);
    const __m512i BL0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    const __m512i BL1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
    {
        __m512d a = _mm512_load_pd(x + 2 * p);
        __m512d b = _mm512_load_pd(x + 2 * (p + m));
        __m512d c = _mm512_load_pd(x + 2 * (p + 2 * m));
        __m512d d = _mm512_load_pd(x + 2 * (p + 3 * m));
        __m512d apc = _mm512_add_pd(a, c), amc = _mm512_sub_pd(a, c);
        __m512d bpd = _mm512_add_pd(b, d), bmd = _mm512_sub_pd(b, d);
        __m512d sw = _mm512_permute_pd(bmd, PSWAP);
        __m512d r0 = _mm512_add_pd(apc, bpd);
        __m512d u2 = _mm512_sub_pd(apc, bpd);
        __m512d u1 = _mm512_fmsubadd_pd(amc, ONE, sw);
        __m512d u3 = _mm512_fmaddsub_pd(amc, ONE, sw);
        __m512d r1, r2, r3;
        if (compact) {
            /* table is (c,s) pairs, 24 doubles per 4-p group: half the dup
             * format's bytes and loads.  cmul re-expressed as
             * fmaddsub(u, dup(c), swap(u)*dup(s)): even ur*c - ui*s,
             * odd ui*c + ur*s -- same values, two extra port-5 dups paid for
             * a halved table stream at the L >= 1024 sizes where stage 1 is
             * fill-bound (ICX PMU, r4). */
            const double *t = tw + 6 * p;
            __m512d t1 = _mm512_load_pd(t), t2 = _mm512_load_pd(t + 8);
            __m512d t3 = _mm512_load_pd(t + 16);
            r1 = _mm512_fmaddsub_pd(u1, _mm512_permute_pd(t1, 0x00),
                     _mm512_mul_pd(_mm512_permute_pd(u1, PSWAP), _mm512_permute_pd(t1, 0xFF)));
            r2 = _mm512_fmaddsub_pd(u2, _mm512_permute_pd(t2, 0x00),
                     _mm512_mul_pd(_mm512_permute_pd(u2, PSWAP), _mm512_permute_pd(t2, 0xFF)));
            r3 = _mm512_fmaddsub_pd(u3, _mm512_permute_pd(t3, 0x00),
                     _mm512_mul_pd(_mm512_permute_pd(u3, PSWAP), _mm512_permute_pd(t3, 0xFF)));
        } else {
        const double *t = tw + 12 * p;
        r1 = cmul_bc(u1, _mm512_load_pd(t), _mm512_load_pd(t + 8));
        r2 = cmul_bc(u2, _mm512_load_pd(t + 16), _mm512_load_pd(t + 24));
        r3 = cmul_bc(u3, _mm512_load_pd(t + 32), _mm512_load_pd(t + 40));
        }
        __m512d t01r = _mm512_permutex2var_pd(r0, TRE, r1);
        __m512d t23r = _mm512_permutex2var_pd(r2, TRE, r3);
        __m512d t01i = _mm512_permutex2var_pd(r0, TIM, r1);
        __m512d t23i = _mm512_permutex2var_pd(r2, TIM, r3);
        _mm512_store_pd(yq, _mm512_permutex2var_pd(t01r, BL0, t23r));
        _mm512_store_pd(yq + 8, _mm512_permutex2var_pd(t01i, BL0, t23i));
        _mm512_store_pd(yq + 16, _mm512_permutex2var_pd(t01r, BL1, t23r));
        _mm512_store_pd(yq + 24, _mm512_permutex2var_pd(t01i, BL1, t23i));
    }
}

static void stage_s1s(const double *restrict x, double *restrict y, int n,
                      const double *restrict tw, int compact)
{
    const int m = n >> 2;
    for (int p = 0; p < m; p += 4)
        s1s_quad(x, y + 8 * p, m, p, tw, compact);
}

/* split radix-8 at s == 4, vectorized over a PAIR of p (lanes 0-3 = p, lanes
 * 4-7 = p+1); tw per pair, r = 1..7: [c x8][s x8].  Outputs of an r-pair
 * share one block, so the stores are pair-merged full zmm. */
#define CS_IDX                                                                         \
    const __m512i ICC = _mm512_setr_epi64(0, 0, 0, 0, 2, 2, 2, 2);                     \
    const __m512i ISS = _mm512_setr_epi64(1, 1, 1, 1, 3, 3, 3, 3)

/* compact p-pair twiddle: 4 doubles [c(p) s(p) c(p+1) s(p+1)] -> two lane
 * vectors [c(p) x4 | c(p+1) x4], [s(p) x4 | s(p+1) x4] (2 dups per r, paid
 * for a 4x smaller table stream) */
#define CSPAIR(T, CV, SV)                                                              \
    __m512d CV, SV;                                                                    \
    do {                                                                               \
        __m512d v_ = _mm512_broadcast_f64x4(_mm256_load_pd(T));                        \
        CV = _mm512_permutexvar_pd(ICC, v_);                                           \
        SV = _mm512_permutexvar_pd(ISS, v_);                                           \
    } while (0)

/* one p-pair of the split radix-8 s == 4 stage, legs M doubles apart on
 * input; t is the pair's twiddle pointer (compact or dup format) */
static inline void s48_pair(const double *restrict xp, long M,
                            double *restrict yp, const double *restrict t8,
                            int compact)
{
    const __m512d CQ = _mm512_set1_pd(0.70710678118654752440);
    CS_IDX;
    {
#define LD(J)                                                                          \
        __m512d x##J##r = _mm512_load_pd(xp + M * (J));                                \
        __m512d x##J##i = _mm512_load_pd(xp + M * (J) + 8)
        LD(0); LD(1); LD(2); LD(3); LD(4); LD(5); LD(6); LD(7);
#undef LD
        R8S_BODY(CQ);
        if (compact) {
            const double *t = t8;
            CSPAIR(t, c1, s1); SCMUL(u1r, u1i, c1, s1);
            CSPAIR(t + 4, c2, s2); SCMUL(u2r, u2i, c2, s2);
            CSPAIR(t + 8, c3, s3); SCMUL(u3r, u3i, c3, s3);
            CSPAIR(t + 12, c4, s4); SCMUL(u4r, u4i, c4, s4);
            CSPAIR(t + 16, c5, s5); SCMUL(u5r, u5i, c5, s5);
            CSPAIR(t + 20, c6, s6); SCMUL(u6r, u6i, c6, s6);
            CSPAIR(t + 24, c7, s7); SCMUL(u7r, u7i, c7, s7);
        } else {
        const double *t = t8;
        SCMUL(u1r, u1i, _mm512_load_pd(t), _mm512_load_pd(t + 8));
        SCMUL(u2r, u2i, _mm512_load_pd(t + 16), _mm512_load_pd(t + 24));
        SCMUL(u3r, u3i, _mm512_load_pd(t + 32), _mm512_load_pd(t + 40));
        SCMUL(u4r, u4i, _mm512_load_pd(t + 48), _mm512_load_pd(t + 56));
        SCMUL(u5r, u5i, _mm512_load_pd(t + 64), _mm512_load_pd(t + 72));
        SCMUL(u6r, u6i, _mm512_load_pd(t + 80), _mm512_load_pd(t + 88));
        SCMUL(u7r, u7i, _mm512_load_pd(t + 96), _mm512_load_pd(t + 104));
        }
#define ST2(J, K, BI)                                                                  \
        _mm512_store_pd(yp + 16 * (BI), _mm512_shuffle_f64x2(u##J##r, u##K##r, 0x44)); \
        _mm512_store_pd(yp + 16 * (BI) + 8,                                            \
                        _mm512_shuffle_f64x2(u##J##i, u##K##i, 0x44));                 \
        _mm512_store_pd(yp + 64 + 16 * (BI),                                           \
                        _mm512_shuffle_f64x2(u##J##r, u##K##r, 0xEE));                 \
        _mm512_store_pd(yp + 64 + 16 * (BI) + 8,                                       \
                        _mm512_shuffle_f64x2(u##J##i, u##K##i, 0xEE))
        ST2(0, 1, 0); ST2(2, 3, 1); ST2(4, 5, 2); ST2(6, 7, 3);
#undef ST2
    }
}

static void stage_s48(const double *restrict x, double *restrict y, int n,
                      const double *restrict tw, int compact)
{
    const int m = n >> 3;
    const long M = 8L * m;
    for (int p = 0; p < m; p += 2)
        s48_pair(x + 8L * p, M, y + 64L * p,
                 tw + (compact ? 14L : 56L) * p, compact);
}

/* split radix-4 at s == 4, p-pair vectorized; tw: r = 1..3: [c x8][s x8] */
static inline void s44_pair(const double *restrict xp, long M,
                            double *restrict yp, const double *restrict t8,
                            int compact)
{
    CS_IDX;
    {
#define LD(J)                                                                          \
        __m512d x##J##r = _mm512_load_pd(xp + M * (J));                                \
        __m512d x##J##i = _mm512_load_pd(xp + M * (J) + 8)
        LD(0); LD(1); LD(2); LD(3);
#undef LD
        __m512d apcr = _mm512_add_pd(x0r, x2r), apci = _mm512_add_pd(x0i, x2i);
        __m512d amcr = _mm512_sub_pd(x0r, x2r), amci = _mm512_sub_pd(x0i, x2i);
        __m512d bpdr = _mm512_add_pd(x1r, x3r), bpdi = _mm512_add_pd(x1i, x3i);
        __m512d bmdr = _mm512_sub_pd(x1r, x3r), bmdi = _mm512_sub_pd(x1i, x3i);
        __m512d u0r = _mm512_add_pd(apcr, bpdr), u0i = _mm512_add_pd(apci, bpdi);
        __m512d u1r = _mm512_add_pd(amcr, bmdi), u1i = _mm512_sub_pd(amci, bmdr);
        __m512d u2r = _mm512_sub_pd(apcr, bpdr), u2i = _mm512_sub_pd(apci, bpdi);
        __m512d u3r = _mm512_sub_pd(amcr, bmdi), u3i = _mm512_add_pd(amci, bmdr);
        if (compact) {
            const double *t = t8;
            CSPAIR(t, c1, s1); SCMUL(u1r, u1i, c1, s1);
            CSPAIR(t + 4, c2, s2); SCMUL(u2r, u2i, c2, s2);
            CSPAIR(t + 8, c3, s3); SCMUL(u3r, u3i, c3, s3);
        } else {
        const double *t = t8;
        SCMUL(u1r, u1i, _mm512_load_pd(t), _mm512_load_pd(t + 8));
        SCMUL(u2r, u2i, _mm512_load_pd(t + 16), _mm512_load_pd(t + 24));
        SCMUL(u3r, u3i, _mm512_load_pd(t + 32), _mm512_load_pd(t + 40));
        }
#define ST2(J, K, BI)                                                                  \
        _mm512_store_pd(yp + 16 * (BI), _mm512_shuffle_f64x2(u##J##r, u##K##r, 0x44)); \
        _mm512_store_pd(yp + 16 * (BI) + 8,                                            \
                        _mm512_shuffle_f64x2(u##J##i, u##K##i, 0x44));                 \
        _mm512_store_pd(yp + 32 + 16 * (BI),                                           \
                        _mm512_shuffle_f64x2(u##J##r, u##K##r, 0xEE));                 \
        _mm512_store_pd(yp + 32 + 16 * (BI) + 8,                                       \
                        _mm512_shuffle_f64x2(u##J##i, u##K##i, 0xEE))
        ST2(0, 1, 0); ST2(2, 3, 1);
#undef ST2
    }
}

static void stage_s44(const double *restrict x, double *restrict y, int n,
                      const double *restrict tw, int compact)
{
    const int m = n >> 2;
    const long M = 8L * m;
    for (int p = 0; p < m; p += 2)
        s44_pair(x + 8L * p, M, y + 32L * p,
                 tw + (compact ? 6L : 24L) * p, compact);
}

/* fused first-stage pair: the stride-1 radix-4 stage feeds the s == 4
 * radix-8 (or radix-4) stage through a 16- (8-) block L1 tile, so the array
 * is read and written ONCE where the unfused schedule made two full passes.
 * For each second-stage p-pair (p2, p2+1), the 8 (4) feeder leg block-pairs
 * are produced by first-stage quads p = 2*(p2 + J*m2); the quad's two output
 * blocks land in tile slot J, and the second stage runs from the tile with a
 * 32-double leg stride.  Requires m2 even, i.e. L >= 256 (radix-8 second) /
 * L >= 128 (radix-4 second).  Same butterflies, same twiddles, same
 * operation order -- output bitwise identical to the unfused pair. */
static void stage_sx48(const double *restrict x, double *restrict y, int n,
                       const double *restrict tw1, const double *restrict tw2,
                       int compact, double *restrict tile)
{
    const int m = n >> 2;   /* first-stage butterflies = second-stage size */
    const int m2 = m >> 3;  /* second-stage groups */
    const long t2s = compact ? 14L : 56L;
    /* one first-stage quad (16 complexes, 2 blocks) covers leg J of TWO
     * consecutive second-stage pairs, so the tile serves four p2 at a time */
    for (int p2 = 0; p2 < m2; p2 += 4) {
        for (int J = 0; J < 8; ++J)
            s1s_quad(x, tile + 32 * J, m, p2 + J * m2, tw1, compact);
        s48_pair(tile, 32, y + 64L * p2, tw2 + t2s * p2, compact);
        s48_pair(tile + 16, 32, y + 64L * (p2 + 2), tw2 + t2s * (p2 + 2), compact);
    }
}

static void stage_sx44(const double *restrict x, double *restrict y, int n,
                       const double *restrict tw1, const double *restrict tw2,
                       int compact, double *restrict tile)
{
    const int m = n >> 2;
    const int m2 = m >> 2;
    const long t2s = compact ? 6L : 24L;
    for (int p2 = 0; p2 < m2; p2 += 4) {
        for (int J = 0; J < 4; ++J)
            s1s_quad(x, tile + 32 * J, m, p2 + J * m2, tw1, compact);
        s44_pair(tile, 32, y + 32L * p2, tw2 + t2s * p2, compact);
        s44_pair(tile + 16, 32, y + 32L * (p2 + 2), tw2 + t2s * (p2 + 2), compact);
    }
}


/* split -> split radix-8, s >= 8, broadcast twiddles (reuses the 21/p table) */
static void stage_ss8(const double *restrict x, double *restrict y, int n,
                      int s, const double *restrict tw)
{
    const int m = n >> 3;
    const long SD = 2L * s; /* doubles per s complexes in block format */
    const __m512d CQ = _mm512_set1_pd(0.70710678118654752440);
    for (int p = 0; p < m; ++p) {
        const double *t = tw + 21 * p;
        const __m512d c1 = _mm512_set1_pd(t[0]), s1 = _mm512_set1_pd(t[8]);
        const __m512d c2 = _mm512_set1_pd(t[1]), s2 = _mm512_set1_pd(t[10]);
        const __m512d c3 = _mm512_set1_pd(t[2]), s3 = _mm512_set1_pd(t[12]);
        const __m512d c4 = _mm512_set1_pd(t[3]), s4 = _mm512_set1_pd(t[14]);
        const __m512d c5 = _mm512_set1_pd(t[4]), s5 = _mm512_set1_pd(t[16]);
        const __m512d c6 = _mm512_set1_pd(t[5]), s6 = _mm512_set1_pd(t[18]);
        const __m512d c7 = _mm512_set1_pd(t[6]), s7 = _mm512_set1_pd(t[20]);
        const double *xp = x + SD * p;
        double *yp = y + SD * 8 * p;
        const long SmD = SD * m;
        for (long q = 0; q < SD; q += 16) {
#define LD(J)                                                                          \
            __m512d x##J##r = _mm512_load_pd(xp + (J) * SmD + q);                      \
            __m512d x##J##i = _mm512_load_pd(xp + (J) * SmD + q + 8)
            LD(0); LD(1); LD(2); LD(3); LD(4); LD(5); LD(6); LD(7);
#undef LD
            R8S_BODY(CQ);
            SCMUL(u1r, u1i, c1, s1);
            SCMUL(u2r, u2i, c2, s2);
            SCMUL(u3r, u3i, c3, s3);
            SCMUL(u4r, u4i, c4, s4);
            SCMUL(u5r, u5i, c5, s5);
            SCMUL(u6r, u6i, c6, s6);
            SCMUL(u7r, u7i, c7, s7);
#define ST(J)                                                                          \
            _mm512_store_pd(yp + (J) * SD + q, u##J##r);                               \
            _mm512_store_pd(yp + (J) * SD + q + 8, u##J##i)
            ST(0); ST(1); ST(2); ST(3); ST(4); ST(5); ST(6); ST(7);
#undef ST
        }
    }
}

/* one radix-8 split group: 8 legs of `len` doubles, legs `inLeg` apart on
 * input and `outLeg` apart on output, broadcast twiddles from a 21/p table.
 * This is stage_ss8's body with the strides freed, so the SS64 tile pass can
 * reuse it for both fused levels. */
static inline void ss8_group(const double *restrict xp, long inLeg,
                             double *restrict yp, long outLeg, long len,
                             const double *restrict t)
{
    const __m512d CQ = _mm512_set1_pd(0.70710678118654752440);
    const __m512d c1 = _mm512_set1_pd(t[0]), s1 = _mm512_set1_pd(t[8]);
    const __m512d c2 = _mm512_set1_pd(t[1]), s2 = _mm512_set1_pd(t[10]);
    const __m512d c3 = _mm512_set1_pd(t[2]), s3 = _mm512_set1_pd(t[12]);
    const __m512d c4 = _mm512_set1_pd(t[3]), s4 = _mm512_set1_pd(t[14]);
    const __m512d c5 = _mm512_set1_pd(t[4]), s5 = _mm512_set1_pd(t[16]);
    const __m512d c6 = _mm512_set1_pd(t[5]), s6 = _mm512_set1_pd(t[18]);
    const __m512d c7 = _mm512_set1_pd(t[6]), s7 = _mm512_set1_pd(t[20]);
    for (long q = 0; q < len; q += 16) {
#define LD(J)                                                                          \
        __m512d x##J##r = _mm512_load_pd(xp + (J) * inLeg + q);                        \
        __m512d x##J##i = _mm512_load_pd(xp + (J) * inLeg + q + 8)
        LD(0); LD(1); LD(2); LD(3); LD(4); LD(5); LD(6); LD(7);
#undef LD
        R8S_BODY(CQ);
        SCMUL(u1r, u1i, c1, s1);
        SCMUL(u2r, u2i, c2, s2);
        SCMUL(u3r, u3i, c3, s3);
        SCMUL(u4r, u4i, c4, s4);
        SCMUL(u5r, u5i, c5, s5);
        SCMUL(u6r, u6i, c6, s6);
        SCMUL(u7r, u7i, c7, s7);
#define ST(J)                                                                          \
        _mm512_store_pd(yp + (J) * outLeg + q, u##J##r);                               \
        _mm512_store_pd(yp + (J) * outLeg + q + 8, u##J##i)
        ST(0); ST(1); ST(2); ST(3); ST(4); ST(5); ST(6); ST(7);
#undef ST
    }
}

/* fused pair of split radix-8 stages (a radix-64 pass through an L1 tile).
 * Stage A is (n, s), stage B is (n/8, 8s); run back to back over the whole
 * array they cost two full read+write passes, and at L = 4096/16384 those
 * passes stream L2 (the array is far beyond L1).  Fused: for each stage-B
 * group p2, run stage A for its 8 feeder groups p = p2 + j*(n/64) into a
 * 64-block tile (64*s complexes = 16/32 KB, L1-resident), then stage B
 * straight from the tile.  Same butterflies, same twiddles, same per-value
 * operation order -- output is bitwise identical; the array is read and
 * written ONCE instead of twice. */
static void stage_ss64(const double *restrict x, double *restrict y, int n,
                       int s, const double *restrict twA,
                       const double *restrict twB, double *restrict tile)
{
    const int m = n >> 3, m2 = n >> 6;
    const long SD = 2L * s;
    for (int p2 = 0; p2 < m2; ++p2) {
        for (int j2 = 0; j2 < 8; ++j2) {
            const int pa = p2 + j2 * m2;
            ss8_group(x + SD * pa, SD * m, tile + 8 * SD * j2, SD, SD,
                      twA + 21 * pa);
        }
        ss8_group(tile, 8 * SD, y + 64L * SD * p2, 8 * SD, 8 * SD,
                  twB + 21 * p2);
    }
}

/* split final radix-4 (n == 4, twiddle-free): interleaves to AoS on store;
 * optionally fuses the chain map (in split form, c deinterleaved on load) */
static void stage_sf4(const double *restrict x, double *restrict y, int s,
                      const double *restrict cf, int nt, int fm)
{
    const long SD = 2L * s;
    DEINT_IDX;
    INT_IDX;
    for (long q = 0; q < SD; q += 16) {
        __m512d x0r = _mm512_load_pd(x + q), x0i = _mm512_load_pd(x + q + 8);
        __m512d x1r = _mm512_load_pd(x + SD + q), x1i = _mm512_load_pd(x + SD + q + 8);
        __m512d x2r = _mm512_load_pd(x + 2 * SD + q), x2i = _mm512_load_pd(x + 2 * SD + q + 8);
        __m512d x3r = _mm512_load_pd(x + 3 * SD + q), x3i = _mm512_load_pd(x + 3 * SD + q + 8);
        __m512d apcr = _mm512_add_pd(x0r, x2r), apci = _mm512_add_pd(x0i, x2i);
        __m512d amcr = _mm512_sub_pd(x0r, x2r), amci = _mm512_sub_pd(x0i, x2i);
        __m512d bpdr = _mm512_add_pd(x1r, x3r), bpdi = _mm512_add_pd(x1i, x3i);
        __m512d bmdr = _mm512_sub_pd(x1r, x3r), bmdi = _mm512_sub_pd(x1i, x3i);
        __m512d u0r = _mm512_add_pd(apcr, bpdr), u0i = _mm512_add_pd(apci, bpdi);
        __m512d u1r = _mm512_add_pd(amcr, bmdi), u1i = _mm512_sub_pd(amci, bmdr);
        __m512d u2r = _mm512_sub_pd(apcr, bpdr), u2i = _mm512_sub_pd(apci, bpdi);
        __m512d u3r = _mm512_sub_pd(amcr, bmdi), u3i = _mm512_add_pd(amci, bmdr);
#define FIN(J, R)                                                                      \
        do {                                                                           \
            if (cf) {                                                                  \
                __m512d ca_ = _mm512_load_pd(cf + (R) * SD + q);                       \
                __m512d cb_ = _mm512_load_pd(cf + (R) * SD + q + 8);                   \
                soa_map_p(&u##J##r, &u##J##i, _mm512_permutex2var_pd(ca_, IRE, cb_),   \
                          _mm512_permutex2var_pd(ca_, IIM, cb_), !fm);                        \
            }                                                                          \
            __m512d lo_ = _mm512_permutex2var_pd(u##J##r, ILO, u##J##i);               \
            __m512d hi_ = _mm512_permutex2var_pd(u##J##r, IHI, u##J##i);               \
            if (nt) {                                                                  \
                _mm512_stream_pd(y + (R) * SD + q, lo_);                               \
                _mm512_stream_pd(y + (R) * SD + q + 8, hi_);                           \
            } else {                                                                   \
                _mm512_store_pd(y + (R) * SD + q, lo_);                                \
                _mm512_store_pd(y + (R) * SD + q + 8, hi_);                            \
            }                                                                          \
        } while (0)
        FIN(0, 0); FIN(1, 1); FIN(2, 2); FIN(3, 3);
#undef FIN
    }
}

/* split final radix-8 (n == 8, twiddle-free): interleave to AoS, optional map */
static void stage_sf8(const double *restrict x, double *restrict y, int s,
                      const double *restrict cf, int nt, int fm)
{
    const long SD = 2L * s;
    const __m512d CQ = _mm512_set1_pd(0.70710678118654752440);
    DEINT_IDX;
    INT_IDX;
    for (long q = 0; q < SD; q += 16) {
#define LD(J)                                                                          \
        __m512d x##J##r = _mm512_load_pd(x + (J) * SD + q);                            \
        __m512d x##J##i = _mm512_load_pd(x + (J) * SD + q + 8)
        LD(0); LD(1); LD(2); LD(3); LD(4); LD(5); LD(6); LD(7);
#undef LD
        R8S_BODY(CQ);
#define FIN(J)                                                                         \
        do {                                                                           \
            if (cf) {                                                                  \
                __m512d ca_ = _mm512_load_pd(cf + (J) * SD + q);                       \
                __m512d cb_ = _mm512_load_pd(cf + (J) * SD + q + 8);                   \
                soa_map_p(&u##J##r, &u##J##i, _mm512_permutex2var_pd(ca_, IRE, cb_),   \
                          _mm512_permutex2var_pd(ca_, IIM, cb_), !fm);                        \
            }                                                                          \
            __m512d lo_ = _mm512_permutex2var_pd(u##J##r, ILO, u##J##i);               \
            __m512d hi_ = _mm512_permutex2var_pd(u##J##r, IHI, u##J##i);               \
            if (nt) {                                                                  \
                _mm512_stream_pd(y + (J) * SD + q, lo_);                               \
                _mm512_stream_pd(y + (J) * SD + q + 8, hi_);                           \
            } else {                                                                   \
                _mm512_store_pd(y + (J) * SD + q, lo_);                                \
                _mm512_store_pd(y + (J) * SD + q + 8, hi_);                            \
            }                                                                          \
        } while (0)
        FIN(0); FIN(1); FIN(2); FIN(3); FIN(4); FIN(5); FIN(6); FIN(7);
#undef FIN
    }
}

/* -------------------- across-batch SoA chain path (from d1_batchlane) -----
 * Lane j of every vector is transform (b0+j); data is split-complex (separate
 * re/im planes of L vectors).  Twiddles are scalar broadcasts, the complex
 * multiply is 2 mul + 2 fma with NO shuffle, and the map loses its pair-swap
 * too.  One 8x8 double transpose each way per group of 8 transforms per whole
 * chain, amortized over all m steps. */

/* 8x8 double transpose: v0..v7 in place (24 shuffles) */
#define TR8(v0, v1, v2, v3, v4, v5, v6, v7)                                            \
    do {                                                                               \
        __m512d t0_ = _mm512_unpacklo_pd(v0, v1), t1_ = _mm512_unpackhi_pd(v0, v1);    \
        __m512d t2_ = _mm512_unpacklo_pd(v2, v3), t3_ = _mm512_unpackhi_pd(v2, v3);    \
        __m512d t4_ = _mm512_unpacklo_pd(v4, v5), t5_ = _mm512_unpackhi_pd(v4, v5);    \
        __m512d t6_ = _mm512_unpacklo_pd(v6, v7), t7_ = _mm512_unpackhi_pd(v6, v7);    \
        __m512d u0_ = _mm512_shuffle_f64x2(t0_, t2_, 0x88);                            \
        __m512d u1_ = _mm512_shuffle_f64x2(t1_, t3_, 0x88);                            \
        __m512d u2_ = _mm512_shuffle_f64x2(t0_, t2_, 0xDD);                            \
        __m512d u3_ = _mm512_shuffle_f64x2(t1_, t3_, 0xDD);                            \
        __m512d u4_ = _mm512_shuffle_f64x2(t4_, t6_, 0x88);                            \
        __m512d u5_ = _mm512_shuffle_f64x2(t5_, t7_, 0x88);                            \
        __m512d u6_ = _mm512_shuffle_f64x2(t4_, t6_, 0xDD);                            \
        __m512d u7_ = _mm512_shuffle_f64x2(t5_, t7_, 0xDD);                            \
        v0 = _mm512_shuffle_f64x2(u0_, u4_, 0x88);                                     \
        v1 = _mm512_shuffle_f64x2(u1_, u5_, 0x88);                                     \
        v2 = _mm512_shuffle_f64x2(u2_, u6_, 0x88);                                     \
        v3 = _mm512_shuffle_f64x2(u3_, u7_, 0x88);                                     \
        v4 = _mm512_shuffle_f64x2(u0_, u4_, 0xDD);                                     \
        v5 = _mm512_shuffle_f64x2(u1_, u5_, 0xDD);                                     \
        v6 = _mm512_shuffle_f64x2(u2_, u6_, 0xDD);                                     \
        v7 = _mm512_shuffle_f64x2(u3_, u7_, 0xDD);                                     \
    } while (0)

/* AoS group (8 transforms, 4 consecutive complexes each) -> SoA planes */
static void soa_tr_in(const double *restrict x, long strideL, double *restrict re,
                      double *restrict im, int L)
{
    for (int k = 0; k < L; k += 4) {
        __m512d v0 = _mm512_load_pd(x + 0 * strideL + 2 * k);
        __m512d v1 = _mm512_load_pd(x + 1 * strideL + 2 * k);
        __m512d v2 = _mm512_load_pd(x + 2 * strideL + 2 * k);
        __m512d v3 = _mm512_load_pd(x + 3 * strideL + 2 * k);
        __m512d v4 = _mm512_load_pd(x + 4 * strideL + 2 * k);
        __m512d v5 = _mm512_load_pd(x + 5 * strideL + 2 * k);
        __m512d v6 = _mm512_load_pd(x + 6 * strideL + 2 * k);
        __m512d v7 = _mm512_load_pd(x + 7 * strideL + 2 * k);
        TR8(v0, v1, v2, v3, v4, v5, v6, v7);
        _mm512_store_pd(re + 8L * k, v0);      _mm512_store_pd(im + 8L * k, v1);
        _mm512_store_pd(re + 8L * (k + 1), v2); _mm512_store_pd(im + 8L * (k + 1), v3);
        _mm512_store_pd(re + 8L * (k + 2), v4); _mm512_store_pd(im + 8L * (k + 2), v5);
        _mm512_store_pd(re + 8L * (k + 3), v6); _mm512_store_pd(im + 8L * (k + 3), v7);
    }
}

static void soa_tr_out(const double *restrict re, const double *restrict im,
                       double *restrict y, long strideL, int L)
{
    for (int k = 0; k < L; k += 4) {
        __m512d v0 = _mm512_load_pd(re + 8L * k), v1 = _mm512_load_pd(im + 8L * k);
        __m512d v2 = _mm512_load_pd(re + 8L * (k + 1)), v3 = _mm512_load_pd(im + 8L * (k + 1));
        __m512d v4 = _mm512_load_pd(re + 8L * (k + 2)), v5 = _mm512_load_pd(im + 8L * (k + 2));
        __m512d v6 = _mm512_load_pd(re + 8L * (k + 3)), v7 = _mm512_load_pd(im + 8L * (k + 3));
        TR8(v0, v1, v2, v3, v4, v5, v6, v7);
        _mm512_store_pd(y + 0 * strideL + 2 * k, v0);
        _mm512_store_pd(y + 1 * strideL + 2 * k, v1);
        _mm512_store_pd(y + 2 * strideL + 2 * k, v2);
        _mm512_store_pd(y + 3 * strideL + 2 * k, v3);
        _mm512_store_pd(y + 4 * strideL + 2 * k, v4);
        _mm512_store_pd(y + 5 * strideL + 2 * k, v5);
        _mm512_store_pd(y + 6 * strideL + 2 * k, v6);
        _mm512_store_pd(y + 7 * strideL + 2 * k, v7);
    }
}

/* split-complex map, componentwise version of map_vec_p (same refinements) */
static inline void soa_map_p(__m512d *vr, __m512d *vi, __m512d cr, __m512d ci,
                             int precise)
{
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d th = _mm512_set1_pd(1.5);
    const __m512d two = _mm512_set1_pd(2.0);
    __m512d zr = _mm512_add_pd(*vr, cr), zi = _mm512_add_pd(*vi, ci);
    __m512d s = _mm512_fmadd_pd(zr, zr, _mm512_mul_pd(zi, zi));
    s = _mm512_max_pd(s, _mm512_set1_pd(1e-300));
    __m512d r = _mm512_rsqrt14_pd(s);
    __m512d hs = _mm512_mul_pd(s, half);
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hs, r), r, th));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hs, r), r, th));
    __m512d u = _mm512_mul_pd(s, r);
    if (precise)
        u = _mm512_fmadd_pd(_mm512_mul_pd(half, r), _mm512_fnmadd_pd(u, u, s), u);
    __m512d t = _mm512_add_pd(one, u);
    __m512d rc = _mm512_rcp14_pd(t);
    rc = _mm512_mul_pd(rc, _mm512_fnmadd_pd(t, rc, two));
    rc = _mm512_fmadd_pd(rc, _mm512_fnmadd_pd(t, rc, one), rc);
    __m512d qr = _mm512_mul_pd(zr, rc), qi = _mm512_mul_pd(zi, rc);
    if (!precise) { *vr = qr; *vi = qi; return; }
    *vr = _mm512_fmadd_pd(_mm512_fnmadd_pd(qr, t, zr), rc, qr);
    *vi = _mm512_fmadd_pd(_mm512_fnmadd_pd(qi, t, zi), rc, qi);
}

/* generic SoA radix-4 DIF stage; tw = 6 doubles per p (cr,si per w^1,w^2,w^3) */
static void soa_stage4(const double *restrict xr, const double *restrict xi,
                       double *restrict yr, double *restrict yi, int n, int s,
                       const double *restrict tw)
{
    const int m = n >> 2;
    for (int p = 0; p < m; ++p) {
        const double *t = tw + 6 * p;
        const __m512d c1 = _mm512_set1_pd(t[0]), s1 = _mm512_set1_pd(t[1]);
        const __m512d c2 = _mm512_set1_pd(t[2]), s2 = _mm512_set1_pd(t[3]);
        const __m512d c3 = _mm512_set1_pd(t[4]), s3 = _mm512_set1_pd(t[5]);
        const double *ar = xr + 8L * s * p, *ai = xi + 8L * s * p;
        double *br = yr + 8L * s * 4 * p, *bi = yi + 8L * s * 4 * p;
        const long M = 8L * s * m, S = 8L * s;
        for (long q = 0; q < S; q += 8) {
            __m512d a_r = _mm512_load_pd(ar + q), a_i = _mm512_load_pd(ai + q);
            __m512d b_r = _mm512_load_pd(ar + M + q), b_i = _mm512_load_pd(ai + M + q);
            __m512d c_r = _mm512_load_pd(ar + 2 * M + q), c_i = _mm512_load_pd(ai + 2 * M + q);
            __m512d d_r = _mm512_load_pd(ar + 3 * M + q), d_i = _mm512_load_pd(ai + 3 * M + q);
            __m512d apcr = _mm512_add_pd(a_r, c_r), apci = _mm512_add_pd(a_i, c_i);
            __m512d amcr = _mm512_sub_pd(a_r, c_r), amci = _mm512_sub_pd(a_i, c_i);
            __m512d bpdr = _mm512_add_pd(b_r, d_r), bpdi = _mm512_add_pd(b_i, d_i);
            __m512d bmdr = _mm512_sub_pd(b_r, d_r), bmdi = _mm512_sub_pd(b_i, d_i);
            _mm512_store_pd(br + q, _mm512_add_pd(apcr, bpdr));
            _mm512_store_pd(bi + q, _mm512_add_pd(apci, bpdi));
            /* r1 = amc - i*bmd, r3 = amc + i*bmd, r2 = apc - bpd */
            __m512d r1r = _mm512_add_pd(amcr, bmdi), r1i = _mm512_sub_pd(amci, bmdr);
            __m512d r2r = _mm512_sub_pd(apcr, bpdr), r2i = _mm512_sub_pd(apci, bpdi);
            __m512d r3r = _mm512_sub_pd(amcr, bmdi), r3i = _mm512_add_pd(amci, bmdr);
            _mm512_store_pd(br + S + q, _mm512_fnmadd_pd(r1i, s1, _mm512_mul_pd(r1r, c1)));
            _mm512_store_pd(bi + S + q, _mm512_fmadd_pd(r1r, s1, _mm512_mul_pd(r1i, c1)));
            _mm512_store_pd(br + 2 * S + q, _mm512_fnmadd_pd(r2i, s2, _mm512_mul_pd(r2r, c2)));
            _mm512_store_pd(bi + 2 * S + q, _mm512_fmadd_pd(r2r, s2, _mm512_mul_pd(r2i, c2)));
            _mm512_store_pd(br + 3 * S + q, _mm512_fnmadd_pd(r3i, s3, _mm512_mul_pd(r3r, c3)));
            _mm512_store_pd(bi + 3 * S + q, _mm512_fmadd_pd(r3r, s3, _mm512_mul_pd(r3i, c3)));
        }
    }
}

/* final twiddle-free SoA radix-4 (n==4) with fused map */
static void soa_final4(const double *restrict xr, const double *restrict xi,
                       double *restrict yr, double *restrict yi, int s,
                       const double *restrict cfr, const double *restrict cfi, int fm)
{
    const long S = 8L * s;
    for (long q = 0; q < S; q += 8) {
        __m512d a_r = _mm512_load_pd(xr + q), a_i = _mm512_load_pd(xi + q);
        __m512d b_r = _mm512_load_pd(xr + S + q), b_i = _mm512_load_pd(xi + S + q);
        __m512d c_r = _mm512_load_pd(xr + 2 * S + q), c_i = _mm512_load_pd(xi + 2 * S + q);
        __m512d d_r = _mm512_load_pd(xr + 3 * S + q), d_i = _mm512_load_pd(xi + 3 * S + q);
        __m512d apcr = _mm512_add_pd(a_r, c_r), apci = _mm512_add_pd(a_i, c_i);
        __m512d amcr = _mm512_sub_pd(a_r, c_r), amci = _mm512_sub_pd(a_i, c_i);
        __m512d bpdr = _mm512_add_pd(b_r, d_r), bpdi = _mm512_add_pd(b_i, d_i);
        __m512d bmdr = _mm512_sub_pd(b_r, d_r), bmdi = _mm512_sub_pd(b_i, d_i);
        __m512d r0r = _mm512_add_pd(apcr, bpdr), r0i = _mm512_add_pd(apci, bpdi);
        __m512d r1r = _mm512_add_pd(amcr, bmdi), r1i = _mm512_sub_pd(amci, bmdr);
        __m512d r2r = _mm512_sub_pd(apcr, bpdr), r2i = _mm512_sub_pd(apci, bpdi);
        __m512d r3r = _mm512_sub_pd(amcr, bmdi), r3i = _mm512_add_pd(amci, bmdr);
        soa_map_p(&r0r, &r0i, _mm512_load_pd(cfr + q), _mm512_load_pd(cfi + q), !fm);
        soa_map_p(&r1r, &r1i, _mm512_load_pd(cfr + S + q), _mm512_load_pd(cfi + S + q), !fm);
        soa_map_p(&r2r, &r2i, _mm512_load_pd(cfr + 2 * S + q), _mm512_load_pd(cfi + 2 * S + q), !fm);
        soa_map_p(&r3r, &r3i, _mm512_load_pd(cfr + 3 * S + q), _mm512_load_pd(cfi + 3 * S + q), !fm);
        _mm512_store_pd(yr + q, r0r);          _mm512_store_pd(yi + q, r0i);
        _mm512_store_pd(yr + S + q, r1r);      _mm512_store_pd(yi + S + q, r1i);
        _mm512_store_pd(yr + 2 * S + q, r2r);  _mm512_store_pd(yi + 2 * S + q, r2i);
        _mm512_store_pd(yr + 3 * S + q, r3r);  _mm512_store_pd(yi + 3 * S + q, r3i);
    }
}

/* final twiddle-free SoA radix-2 (n==2) with fused map */
static void soa_final2(const double *restrict xr, const double *restrict xi,
                       double *restrict yr, double *restrict yi, int s,
                       const double *restrict cfr, const double *restrict cfi, int fm)
{
    const long S = 8L * s;
    for (long q = 0; q < S; q += 8) {
        __m512d a_r = _mm512_load_pd(xr + q), a_i = _mm512_load_pd(xi + q);
        __m512d b_r = _mm512_load_pd(xr + S + q), b_i = _mm512_load_pd(xi + S + q);
        __m512d r0r = _mm512_add_pd(a_r, b_r), r0i = _mm512_add_pd(a_i, b_i);
        __m512d r1r = _mm512_sub_pd(a_r, b_r), r1i = _mm512_sub_pd(a_i, b_i);
        soa_map_p(&r0r, &r0i, _mm512_load_pd(cfr + q), _mm512_load_pd(cfi + q), !fm);
        soa_map_p(&r1r, &r1i, _mm512_load_pd(cfr + S + q), _mm512_load_pd(cfi + S + q), !fm);
        _mm512_store_pd(yr + q, r0r);     _mm512_store_pd(yi + q, r0i);
        _mm512_store_pd(yr + S + q, r1r); _mm512_store_pd(yi + S + q, r1i);
    }
}

/* whole batched chain in SoA groups of 8 transforms */
static void soa_chain(fft1d_plan *p, const double *restrict x0,
                      const double *restrict c, double *restrict out, int m)
{
    const int L = p->L;
    const long strideL = 2L * L;
    double *A = p->soa_a, *B = p->soa_b;
    for (int b0 = 0; b0 + 8 <= p->batch; b0 += 8) {
        soa_tr_in(x0 + b0 * strideL, strideL, A, A + 8L * L, L);
        soa_tr_in(c + b0 * strideL, strideL, p->soa_c, p->soa_c + 8L * L, L);
        for (int step = 0; step < m; ++step) {
            double *sr = A, *dr = B;
            int n = L, s = 1;
            for (int t = 0; t < p->soa_T - 1; ++t) {
                soa_stage4(sr, sr + 8L * L, dr, dr + 8L * L, n, s, p->soa_tw[t]);
                { double *tmp = sr; sr = dr; dr = tmp; }
                n >>= 2; s <<= 2;
            }
            if (n == 4)
                soa_final4(sr, sr + 8L * L, dr, dr + 8L * L, s, p->soa_c, p->soa_c + 8L * L, p->fastmap);
            else
                soa_final2(sr, sr + 8L * L, dr, dr + 8L * L, s, p->soa_c, p->soa_c + 8L * L, p->fastmap);
            /* state for the next step is dr; keep names so A is always state */
            A = dr; B = sr;
        }
        soa_tr_out(A, A + 8L * L, out + b0 * strideL, strideL, L);
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
 * ever writes the buffer it reads, and so only ONE scratch buffer joins in/out
 * in the working set (a second scratch MEASURED +20% at L=1024 B=1: the L1
 * footprint goes 48 -> 64 KB). */
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
#ifdef D1_AVX512
        case ST_S1S: stage_s1s(sp, d, st->n, st->tw, p->cmpt); break;
        case ST_S48: stage_s48(sp, d, st->n, st->tw, p->cmpt); break;
        case ST_S44: stage_s44(sp, d, st->n, st->tw, p->cmpt); break;
        case ST_SS8: stage_ss8(sp, d, st->n, st->s, st->tw); break;
        case ST_SS64: stage_ss64(sp, d, st->n, st->s, st->tw, st->tw2, p->tile); break;
        case ST_SX48: stage_sx48(sp, d, st->n, st->tw, st->tw2, p->cmpt, p->tile); break;
        case ST_SX44: stage_sx44(sp, d, st->n, st->tw, st->tw2, p->cmpt, p->tile); break;
        case ST_SF8: stage_sf8(sp, d, st->s, cft, nt, p->fastmap); break;
        case ST_SF4: stage_sf4(sp, d, st->s, cft, nt, p->fastmap); break;
#endif
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
#ifndef D1_SPLIT_MIN
#define D1_SPLIT_MIN 128
#endif
#ifdef D1_AVX512
    if (L >= D1_SPLIT_MIN) {
        /* split-complex pipeline: AoS s1 (radix-4), one AoS->split stage at
         * s == 4 (radix 8 or 4, chosen so the residue factors as 8^a * (4|8)),
         * zero-shuffle split radix-8 stages, split final radix-8/4 that
         * re-interleaves on store.  Pass counts match the old AoS schedule. */
        int k = 0;
        for (int nn = L; nn > 1; nn >>= 1) ++k;
        p->cmpt = (L >= 1024);
        p->st[T].n = n; p->st[T].s = s; p->st[T].type = ST_S1S;
        twd += (size_t)(n / 4) * (p->cmpt ? 6 : 12);
        ++T; n >>= 2; s <<= 2;
        if ((k - 2) % 3 == 1) {
            p->st[T].n = n; p->st[T].s = s; p->st[T].type = ST_S44;
            twd += (size_t)(n / 4) * (p->cmpt ? 6 : 24);
            ++T; n >>= 2; s <<= 2;
        } else {
            p->st[T].n = n; p->st[T].s = s; p->st[T].type = ST_S48;
            twd += (size_t)(n / 8) * (p->cmpt ? 14 : 56);
            ++T; n >>= 3; s <<= 3;
        }
        while (n > 8) {
            p->st[T].n = n; p->st[T].s = s; p->st[T].type = ST_SS8;
            twd += (size_t)(n / 8) * 21;
            ++T; n >>= 3; s <<= 3;
        }
        p->st[T].n = n;
        p->st[T].s = s;
        p->st[T].type = (n == 8) ? ST_SF8 : ST_SF4;
        ++T;
        /* Fuse consecutive SS8 pairs into one L1-tiled radix-64 pass: one
         * fewer full read+write of the array, identical arithmetic.  GATED to
         * L == 4096: there it MEASURED -5.7% (6.67 -> 6.29 us B=1 wallaby),
         * but at 16384 the same fusion measured +5% (27.0 -> 28.3) -- the
         * fused traversal reads 64 interleaved 512 B bursts 4 KB apart, too
         * short for the prefetcher (explicit T0 prefetch of the next feeder
         * group did not recover it: 28.7).  Same stream-count lesson as r2's
         * plane-format failure. */
        if (L == 4096 || L == 16384)
            for (int t = 0; t + 1 < T; ++t)
                if (p->st[t].type == ST_SS8 && p->st[t + 1].type == ST_SS8) {
                    p->st[t].type = ST_SS64;
                    for (int u = t + 1; u + 1 < T; ++u) p->st[u] = p->st[u + 1];
                    --T;
                }
        p->T = T;
        goto sched_done;
    }
#endif
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
#ifdef D1_AVX512
sched_done:
#endif
    /* Non-temporal final-stage stores: r1 measured them 3x SLOWER when the
     * batch working set fits L3 (1024 B=512, 16 MB: 5.95 vs 2.1 us -- regular
     * stores hit L3, NT forces DRAM) -- but that verdict was taken on
     * wallaby's 60 MB L3.  The scoring node has 24 MB, and the graded
     * 4096xB=256 / 16384xB=64 cells are 33.5 MB of in+out: there the output
     * RFO reads are pure waste and NT frees L3 for the input.  Enable NT only
     * above the scoring node's L3 (verified on wallaby by pushing the batch
     * past ITS L3: 16384 B=256, 128 MB, NT wins ~17%). */
#ifndef D1_NT_MIN_BYTES
#define D1_NT_MIN_BYTES (25.0 * 1024 * 1024)
#endif
    p->nt = ((double)L * (double)batch * 32.0 >= (double)(D1_NT_MIN_BYTES));
    /* Fast map at every size: dropping the two exact-residual refinements
     * moved the graded chain gates by <25% (e.g. 1024:1:4000 4.54e-12 ->
     * 5.68e-12 vs the 1e-10 floor, m=2 gates ~1e-15 vs 3e-14) while gaining
     * 10-27% on every chained cell.  If a scoring-node seed ever fails a
     * gate, rebuild with -DD1_FASTMAP_MAX_L=0 to restore the precise map. */
#ifndef D1_FASTMAP_MAX_L
#define D1_FASTMAP_MAX_L 65536
#endif
    p->fastmap = (L <= D1_FASTMAP_MAX_L);

#ifndef D1_SX_MIN
#define D1_SX_MIN 1024
#endif
    size_t tiled = 0;
    for (int t = 0; t < p->T; ++t)
        if (p->st[t].type == ST_SS64 && 128u * (size_t)p->st[t].s > tiled)
            tiled = 128u * (size_t)p->st[t].s;
    if (L >= D1_SX_MIN && p->st[0].type == ST_S1S && tiled < 256)
        tiled = 256;

    twd += 8; /* padding so vector loads at the tail stay in bounds */
    if (posix_memalign((void **)&p->twmem, 64, twd * sizeof(double)) ||
        posix_memalign((void **)&p->scratch, 64, 2 * (size_t)L * sizeof(double)) ||
        posix_memalign((void **)&p->state, 64, 2 * (size_t)L * sizeof(double)) ||
        (tiled && posix_memalign((void **)&p->tile, 64, tiled * sizeof(double))) ||
        (p->nt && posix_memalign((void **)&p->scratch2, 64, 2 * (size_t)L * sizeof(double)))) {
        free(p->twmem); free(p->scratch); free(p->state); free(p->tile);
        free(p->scratch2); free(p);
        return NULL;
    }
    memset(p->twmem, 0, twd * sizeof(double));

    double *w = p->twmem;
    for (int t = 0; t < T; ++t) {
        stage_t *st = &p->st[t];
        int nn = st->n, m = nn / 4;
        if (st->type == ST_S1S && p->cmpt) {
            st->tw = w;
            for (int q = 0; q < m; ++q) {
                int g = (q / 4) * 24, j = q % 4;
                for (int r = 1; r <= 3; ++r) {
                    long double th = -2.0L * PIL * (long double)((long)q * r % nn) / (long double)nn;
                    w[g + 8 * (r - 1) + 2 * j] = (double)cosl(th);
                    w[g + 8 * (r - 1) + 2 * j + 1] = (double)sinl(th);
                }
            }
            w += (size_t)m * 6;
        } else if (st->type == ST_S1 || st->type == ST_S1S) {
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
        } else if (st->type == ST_GEN8 || st->type == ST_SS8 || st->type == ST_SS64) {
            /* SS64 carries the tables of BOTH fused radix-8 levels: level A
             * over nn (nn/8 groups), then level B over nn/8 (nn/64 groups) --
             * exactly the two tables the unfused stages would have had. */
            int lev = (st->type == ST_SS64) ? 2 : 1;
            for (int l = 0; l < lev; ++l) {
                int nl = (l == 0) ? nn : nn / 8;
                int m8 = nl / 8;
                if (l == 0) st->tw = w; else st->tw2 = w;
                for (int q = 0; q < m8; ++q) {
                    for (int r = 1; r <= 7; ++r) {
                        long double th = -2.0L * PIL * (long double)((long)q * r % nl) / (long double)nl;
                        w[21 * q + (r - 1)] = (double)cosl(th);
                        w[21 * q + 7 + 2 * (r - 1)] = -(double)sinl(th);
                        w[21 * q + 8 + 2 * (r - 1)] = (double)sinl(th);
                    }
                }
                w += (size_t)m8 * 21;
            }
        } else if ((st->type == ST_S48 || st->type == ST_S44) && p->cmpt) {
            /* compact p-pair twiddles: per pair, r = 1..R: [c(p) s(p) c(p+1) s(p+1)] */
            int R = (st->type == ST_S48) ? 7 : 3;
            int mm = nn / (R + 1);
            st->tw = w;
            for (int q = 0; q < mm; ++q) {
                double *base = w + (size_t)(q >> 1) * (4 * R) + (q & 1) * 2;
                for (int r = 1; r <= R; ++r) {
                    long double th = -2.0L * PIL * (long double)((long)q * r % nn) / (long double)nn;
                    base[4 * (r - 1)] = (double)cosl(th);
                    base[4 * (r - 1) + 1] = (double)sinl(th);
                }
            }
            w += (size_t)(mm / 2) * (4 * R);
        } else if (st->type == ST_S48 || st->type == ST_S44) {
            /* p-pair vector twiddles: per pair, r = 1..R: [c(p) x4 | c(p+1) x4]
             * then [s(p) x4 | s(p+1) x4] at +8 (R = 7 or 3) */
            int R = (st->type == ST_S48) ? 7 : 3;
            int mm = nn / (R + 1);
            st->tw = w;
            for (int q = 0; q < mm; ++q) {
                double *base = w + (size_t)(q >> 1) * (16 * R) + (q & 1) * 4;
                for (int r = 1; r <= R; ++r) {
                    long double th = -2.0L * PIL * (long double)((long)q * r % nn) / (long double)nn;
                    double cr = (double)cosl(th), si = (double)sinl(th);
                    for (int j = 0; j < 4; ++j) {
                        base[16 * (r - 1) + j] = cr;
                        base[16 * (r - 1) + 8 + j] = si;
                    }
                }
            }
            w += (size_t)(mm / 2) * (16 * R);
        }
    }
#ifdef D1_AVX512
    /* fuse the first two stages through the L1 tile (one array pass instead
     * of two); tables are already filled, the merged stage carries both */
    if (L >= D1_SX_MIN && p->T >= 3 && p->st[0].type == ST_S1S &&
        (p->st[1].type == ST_S48 || p->st[1].type == ST_S44)) {
        p->st[0].type = (p->st[1].type == ST_S48) ? ST_SX48 : ST_SX44;
        p->st[0].tw2 = p->st[1].tw;
        for (int u = 1; u + 1 < p->T; ++u) p->st[u] = p->st[u + 1];
        --p->T;
    }
#endif
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
#ifdef D1_AVX512
    /* across-batch SoA chain path: pure radix-4 ladder + final radix-4/2.
     * Gated at L <= 2048: the group working set is 3 x 16L doubles (state,
     * ping, c); at 4096 that is 1.5 MB, past the scoring node's 1.25 MB L2,
     * and at 16384 it MEASURED 2x slower than the per-transform AoS path
     * (98.0 vs 48.8 us on wallaby).  Below that it wins outright
     * (32: 0.064 -> 0.039, 64: 0.116 -> 0.062, 128: 0.233 -> 0.146). */
    if (batch >= 8 && L <= 2048) {
        int T2 = 0, n2 = L;
        while (n2 > 4) { ++T2; n2 >>= 2; }
        size_t twd2 = 0, nn = L;
        for (int t = 0; t < T2; ++t) { twd2 += (nn / 4) * 6; nn >>= 2; }
        if (posix_memalign((void **)&p->soa_twmem, 64, (twd2 + 8) * sizeof(double)) ||
            posix_memalign((void **)&p->soa_a, 64, 16 * (size_t)L * sizeof(double)) ||
            posix_memalign((void **)&p->soa_b, 64, 16 * (size_t)L * sizeof(double)) ||
            posix_memalign((void **)&p->soa_c, 64, 16 * (size_t)L * sizeof(double))) {
            free(p->soa_twmem); free(p->soa_a); free(p->soa_b); free(p->soa_c);
            p->soa_twmem = p->soa_a = p->soa_b = p->soa_c = NULL;
            p->soa_T = 0; /* AoS chain path still works */
        } else {
            p->soa_T = T2 + 1;
            double *w2 = p->soa_twmem;
            int n3 = L;
            for (int t = 0; t < T2; ++t) {
                p->soa_tw[t] = w2;
                for (int q = 0; q < n3 / 4; ++q)
                    for (int r = 1; r <= 3; ++r) {
                        long double th = -2.0L * PIL * (long double)((long)q * r % n3) /
                                         (long double)n3;
                        w2[6 * q + 2 * (r - 1)] = (double)cosl(th);
                        w2[6 * q + 2 * (r - 1) + 1] = (double)sinl(th);
                    }
                w2 += (size_t)(n3 / 4) * 6;
                n3 >>= 2;
            }
        }
    }
#endif
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
        /* With NT final stores the intermediates must NOT ping-pong through
         * dst: NT streaming into lines this call just dirtied in cache
         * MEASURED 1.5x slower than no NT at all (16384 B=256, 128 MB).
         * Two private scratches keep dst untouched until the final stream. */
        double *bo = p->nt ? p->scratch : ((p->T & 1) ? dst : p->scratch);
        double *be = p->nt ? p->scratch2 : ((p->T & 1) ? p->scratch : dst);
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
    int b0 = 0;
#ifdef D1_AVX512
    if (p->soa_T && p->batch >= 8) {
        soa_chain(p, (const double *)x0, (const double *)c, (double *)final_out, m);
        b0 = p->batch & ~7;
        if (b0 == p->batch) return;
    }
    if (p->L == 32) { fft32_chain(p, (const double *)(x0 + (size_t)b0 * L), (const double *)(c + (size_t)b0 * L), (double *)(final_out + (size_t)b0 * L), p->batch - b0, m); return; }
    if (p->L == 64) { fft64_chain(p, (const double *)(x0 + (size_t)b0 * L), (const double *)(c + (size_t)b0 * L), (double *)(final_out + (size_t)b0 * L), p->batch - b0, m); return; }
#endif
    for (int b = b0; b < p->batch; ++b) {
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
    free(p->tile);
    free(p->scratch);
    free(p->scratch2);
    free(p->state);
    free(p->tws1full);
    free(p->soa_twmem);
    free(p->soa_a);
    free(p->soa_b);
    free(p->soa_c);
    free(p);
}
