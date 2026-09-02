/* d1_prime: small-prime 1D FFT (L = 7, 11, 13, 17, 31; graded sizes 13 and 31).
 *
 * Strategy (round d1_r1): dense conjugate-symmetric kernel, NOT Rader/Winograd.
 * For prime L the DFT matrix pairs outputs y[k] / y[L-k] through
 *   y[k]   = x0 + sum_j ( s_j cos(2pi jk/L) - i d_j sin(2pi jk/L) )
 *   y[L-k] = x0 + sum_j ( s_j cos(2pi jk/L) + i d_j sin(2pi jk/L) )
 * with s_j = x_j + x_{L-j}, d_j = x_j - x_{L-j}, j,k = 1..h = (L-1)/2.
 * That is 4 real h x h matvecs = 4h^2 FMAs -- 144 (L=13) / 900 (L=31) flops as
 * 24 / 120 zmm FMA instructions when vectorized over k.  The literature corpus
 * (docs/LITERATURE.md 3.4, 08 5.5/5.7) says direct multiplication is what VkFFT
 * and cuFFT themselves choose at tiny primes; instructions, not flops, are the
 * currency, so Winograd/WFTA is deliberately not attempted.
 *
 * Vectorization: over the OUTPUT index k (h lanes), with the (real) cos/sin
 * tables pair-duplicated so one 128-bit broadcast of the complex s_j / d_j
 * feeds both re and im lanes of the accumulator (L=13), or plain k-lane form
 * with scalar broadcasts (L=31, h=15 needs 2 zmm per accumulator).  All table
 * loads are L1-resident (< 4 KiB per size).
 *
 * fft1d_chain is owned: the map state <- (FFT(state)+c)/(1+|FFT(state)+c|) is
 * applied in registers on the store-ordered output vectors, so a chain step
 * never round-trips y through memory or pays a second pass / call.
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../fft1d_api.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#define HAVE_AVX512 1
#endif

/* -------------------------------------------------------------------------- */
/* plan                                                                       */
/* -------------------------------------------------------------------------- */
struct fft1d_plan {
    int L, batch;
    /* generic-path tables (L = 7, 11, 17, and any L without a hand kernel):
       ct[j*h + (k-1)] = cos(2pi (j+1) k / L), st likewise sin. */
    double *ct, *st;
};

const char *fft1d_name(void) { return "d1_prime"; }
const char *fft1d_description(void)
{
    return "small-prime dense conjugate-symmetric kernel: y[k]/y[L-k] pairing, "
           "k-vectorized AVX-512 FMA (pair-duplicated tables at L=13, split "
           "k-lane at L=31), fused-map chain in registers";
}
int fft1d_supports(int L) { return L == 7 || L == 11 || L == 13 || L == 17 || L == 31; }

/* -------------------------------------------------------------------------- */
/* trig tables for the hand kernels (filled once, exact integer reduction)    */
/* -------------------------------------------------------------------------- */
#ifdef HAVE_AVX512
/* L=13, h=6.  Pair-duplicated over k so a 128-bit broadcast of (re,im) works:
 *   CTD13[j][0] lanes = c(j,1) c(j,1) c(j,2) c(j,2) c(j,3) c(j,3) c(j,4) c(j,4)
 *   CTD13H[j]   lanes = c(j,5) c(j,5) c(j,6) c(j,6)                      (ymm) */
static double CTD13[6][8] __attribute__((aligned(64)));
static double STD13[6][8] __attribute__((aligned(64)));
static double CTD13H[6][4] __attribute__((aligned(64)));
static double STD13H[6][4] __attribute__((aligned(64)));
/* L=31, h=15, plain k-lane form: CT31[j][k-1] = cos(2pi (j+1) k / 31), 16 lanes. */
static double CT31[15][16] __attribute__((aligned(64)));
static double ST31[15][16] __attribute__((aligned(64)));
static int tables_ready = 0;

static void fill_tables(void)
{
    if (tables_ready) return;
    for (int j = 1; j <= 6; ++j)
        for (int k = 1; k <= 6; ++k) {
            double ph = 2.0 * M_PI * (double)((j * k) % 13) / 13.0;
            double c = cos(ph), s = sin(ph);
            if (k <= 4) {
                CTD13[j-1][2*(k-1)] = CTD13[j-1][2*(k-1)+1] = c;
                STD13[j-1][2*(k-1)] = STD13[j-1][2*(k-1)+1] = s;
            } else {
                CTD13H[j-1][2*(k-5)] = CTD13H[j-1][2*(k-5)+1] = c;
                STD13H[j-1][2*(k-5)] = STD13H[j-1][2*(k-5)+1] = s;
            }
        }
    for (int j = 1; j <= 15; ++j)
        for (int k = 1; k <= 15; ++k) {
            double ph = 2.0 * M_PI * (double)((j * k) % 31) / 31.0;
            CT31[j-1][k-1] = cos(ph);
            ST31[j-1][k-1] = sin(ph);
        }
    tables_ready = 1;
}

/* -------------------------------------------------------------------------- */
/* L = 13 kernel.  X, Y are interleaved complex (26 doubles).                 */
/* MAPPED != 0 fuses one chaotic-map step (C = the map's constant field).     */
/* -------------------------------------------------------------------------- */
/* Packed-map helper: two 4-complex vectors za/zb share one sqrt and one div on
 * their 8 unique |z|^2 values (a pair-duplicated map would burn 2x the divider).
 * Returns the per-complex reciprocal scales, still packed. */
static inline __m512d packed_recip(__m512d zza, __m512d zzb)
{
    const __m512i EVEN = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i ODD  = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    __m512d n2 = _mm512_add_pd(_mm512_permutex2var_pd(zza, EVEN, zzb),
                               _mm512_permutex2var_pd(zza, ODD, zzb));
    __m512d dn = _mm512_add_pd(_mm512_set1_pd(1.0), _mm512_sqrt_pd(n2));
    return _mm512_div_pd(_mm512_set1_pd(1.0), dn);
}
static inline __m512d dup_lo(__m512d r)
{
    return _mm512_permutexvar_pd(_mm512_setr_epi64(0, 0, 1, 1, 2, 2, 3, 3), r);
}
static inline __m512d dup_hi(__m512d r)
{
    return _mm512_permutexvar_pd(_mm512_setr_epi64(4, 4, 5, 5, 6, 6, 7, 7), r);
}

static inline void kern13(const double *restrict X, double *restrict Y,
                          const double *restrict C, int mapped)
{
    /* pre: s_j = x_j + x_{13-j}, d_j = x_j - x_{13-j}, kept in registers */
    const __m512i REVZ = _mm512_setr_epi64(6, 7, 4, 5, 2, 3, 0, 1);
    __m512d xlo = _mm512_loadu_pd(X + 2);                    /* x1..x4  */
    __m512d xhr = _mm512_permutexvar_pd(REVZ, _mm512_loadu_pd(X + 18)); /* x12..x9 */
    __m256d xm1 = _mm256_loadu_pd(X + 10);                   /* x5 x6   */
    __m256d xm2 = _mm256_permute4x64_pd(_mm256_loadu_pd(X + 14), 0x4E); /* x8 x7 */
    __m512d slo = _mm512_add_pd(xlo, xhr), dlo = _mm512_sub_pd(xlo, xhr);
    __m512d shi = _mm512_castpd256_pd512(_mm256_add_pd(xm1, xm2));
    __m512d dhi = _mm512_castpd256_pd512(_mm256_sub_pd(xm1, xm2));

    /* y0 = x0 + sum s_j */
    __m512d t = _mm512_add_pd(slo, _mm512_insertf64x4(_mm512_setzero_pd(),
                                                      _mm512_castpd512_pd256(shi), 0));
    __m256d u = _mm256_add_pd(_mm512_castpd512_pd256(t), _mm512_extractf64x4_pd(t, 1));
    __m128d v = _mm_add_pd(_mm256_castpd256_pd128(u), _mm256_extractf128_pd(u, 1));
    __m128d y0 = _mm_add_pd(v, _mm_loadu_pd(X));

    /* main: A = x0 + sum s_j c_jk ; B = sum d_j s_jk   (pair-duplicated lanes).
       Broadcasts come from registers via vpermpd, never through memory: a 16 B
       broadcast load from a just-written 64 B store risks a failed
       store-forward every transform. */
    const __m512i BC0 = _mm512_setr_epi64(0, 1, 0, 1, 0, 1, 0, 1);
    const __m512i BC1 = _mm512_setr_epi64(2, 3, 2, 3, 2, 3, 2, 3);
    const __m512i BC2 = _mm512_setr_epi64(4, 5, 4, 5, 4, 5, 4, 5);
    const __m512i BC3 = _mm512_setr_epi64(6, 7, 6, 7, 6, 7, 6, 7);
    __m512d Alo = _mm512_broadcast_f64x2(_mm_loadu_pd(X));
    __m256d Ahi = _mm256_broadcast_pd((const __m128d *)X);
    __m512d Blo = _mm512_setzero_pd();
    __m256d Bhi = _mm256_setzero_pd();
#define PJ13(j, src_s, src_d, idx)                                                   \
    do {                                                                             \
        __m512d bs = _mm512_permutexvar_pd(idx, src_s);                              \
        __m512d bd = _mm512_permutexvar_pd(idx, src_d);                              \
        Alo = _mm512_fmadd_pd(bs, _mm512_load_pd(CTD13[j]), Alo);                    \
        Blo = _mm512_fmadd_pd(bd, _mm512_load_pd(STD13[j]), Blo);                    \
        Ahi = _mm256_fmadd_pd(_mm512_castpd512_pd256(bs), _mm256_load_pd(CTD13H[j]), Ahi); \
        Bhi = _mm256_fmadd_pd(_mm512_castpd512_pd256(bd), _mm256_load_pd(STD13H[j]), Bhi); \
    } while (0)
    PJ13(0, slo, dlo, BC0);
    PJ13(1, slo, dlo, BC1);
    PJ13(2, slo, dlo, BC2);
    PJ13(3, slo, dlo, BC3);
    PJ13(4, shi, dhi, BC0);
    PJ13(5, shi, dhi, BC1);
#undef PJ13

    /* combine: y[k] = (Ar+Bi, Ai-Br), y[13-k] = (Ar-Bi, Ai+Br) */
    const __m512d SGN = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    __m512d swlo = _mm512_permute_pd(Blo, 0x55);
    __m256d swhi = _mm256_permute_pd(Bhi, 0x5);
    __m512d flo = _mm512_fmadd_pd(swlo, SGN, Alo);   /* y1..y4  */
    __m512d rlo = _mm512_fnmadd_pd(swlo, SGN, Alo);  /* y12..y9 in k order */
    __m256d fhi = _mm256_fmadd_pd(swhi, _mm512_castpd512_pd256(SGN), Ahi);  /* y5 y6 */
    __m256d rhi = _mm256_fnmadd_pd(swhi, _mm512_castpd512_pd256(SGN), Ahi); /* y8 y7 */

    /* store order: rev side is k = 6..1 */
    const __m512i RV0 = _mm512_setr_epi64(10, 11, 8, 9, 6, 7, 4, 5); /* k6 k5 k4 k3 */
    __m512d g0 = _mm512_permutex2var_pd(rlo, RV0, _mm512_castpd256_pd512(rhi));
    __m256d g1 = _mm256_permute4x64_pd(_mm512_castpd512_pd256(rlo), 0x4E); /* k2 k1 */

    if (!mapped) {
        _mm_storeu_pd(Y, y0);
        _mm512_storeu_pd(Y + 2, flo);
        _mm256_storeu_pd(Y + 10, fhi);
        _mm512_storeu_pd(Y + 14, g0);
        _mm256_storeu_pd(Y + 22, g1);
    } else {
        /* fused map: z = y + c; y' = z / (1 + |z|).  13 magnitudes share
           2 zmm sqrt + 2 zmm div via lane packing. */
        __m512d mid = _mm512_insertf64x4(_mm512_castpd256_pd512(fhi), g1, 1);
        __m512d cmid = _mm512_insertf64x4(
            _mm512_castpd256_pd512(_mm256_loadu_pd(C + 10)),
            _mm256_loadu_pd(C + 22), 1);
        __m512d za = _mm512_add_pd(flo, _mm512_loadu_pd(C + 2));
        __m512d zb = _mm512_add_pd(g0, _mm512_loadu_pd(C + 14));
        __m512d zc = _mm512_add_pd(mid, cmid);
        __m128d z0 = _mm_add_pd(y0, _mm_loadu_pd(C));
        __m512d z0w = _mm512_zextpd128_pd512(z0);   /* upper lanes zero, so the
                                                       shared sqrt sees no garbage */
        __m512d rab = packed_recip(_mm512_mul_pd(za, za), _mm512_mul_pd(zb, zb));
        __m512d rc0 = packed_recip(_mm512_mul_pd(zc, zc), _mm512_mul_pd(z0w, z0w));
        _mm512_storeu_pd(Y + 2, _mm512_mul_pd(za, dup_lo(rab)));
        _mm512_storeu_pd(Y + 14, _mm512_mul_pd(zb, dup_hi(rab)));
        __m512d mm = _mm512_mul_pd(zc, dup_lo(rc0));
        _mm256_storeu_pd(Y + 10, _mm512_castpd512_pd256(mm));
        _mm256_storeu_pd(Y + 22, _mm512_extractf64x4_pd(mm, 1));
        __m128d r0 = _mm512_castpd512_pd128(
            _mm512_permutexvar_pd(_mm512_setr_epi64(4, 4, 0, 0, 0, 0, 0, 0), rc0));
        _mm_storeu_pd(Y, _mm_mul_pd(z0, r0));
    }
}

/* -------------------------------------------------------------------------- */
/* L = 31 kernel, h = 15: k lanes 1..8 / 9..15 in two zmm per accumulator.    */
/* -------------------------------------------------------------------------- */
static inline void kern31(const double *restrict X, double *restrict Y,
                          const double *restrict C, int mapped,
                          double *restrict sb, double *restrict db)
{
    /* pre: s/d for j = 1..15, interleaved in caller scratch */
    const __m512i REVZ = _mm512_setr_epi64(6, 7, 4, 5, 2, 3, 0, 1);
    __m512d a0 = _mm512_loadu_pd(X + 2);   /* x1..x4   */
    __m512d a1 = _mm512_loadu_pd(X + 10);  /* x5..x8   */
    __m512d a2 = _mm512_loadu_pd(X + 18);  /* x9..x12  */
    __m512d b0 = _mm512_permutexvar_pd(REVZ, _mm512_loadu_pd(X + 54)); /* x30..x27 */
    __m512d b1 = _mm512_permutexvar_pd(REVZ, _mm512_loadu_pd(X + 46)); /* x26..x23 */
    __m512d b2 = _mm512_permutexvar_pd(REVZ, _mm512_loadu_pd(X + 38)); /* x22..x19 */
    _mm512_store_pd(sb,      _mm512_add_pd(a0, b0));
    _mm512_store_pd(sb + 8,  _mm512_add_pd(a1, b1));
    _mm512_store_pd(sb + 16, _mm512_add_pd(a2, b2));
    _mm512_store_pd(db,      _mm512_sub_pd(a0, b0));
    _mm512_store_pd(db + 8,  _mm512_sub_pd(a1, b1));
    _mm512_store_pd(db + 16, _mm512_sub_pd(a2, b2));
    /* j = 13, 14, 15: x13..x15 vs x18..x16 */
    __m256d m1 = _mm256_loadu_pd(X + 26);                                /* x13 x14 */
    __m256d m2 = _mm256_permute4x64_pd(_mm256_loadu_pd(X + 34), 0x4E);   /* x18 x17 */
    _mm256_store_pd(sb + 24, _mm256_add_pd(m1, m2));
    _mm256_store_pd(db + 24, _mm256_sub_pd(m1, m2));
    {
        __m128d e1 = _mm_loadu_pd(X + 30), e2 = _mm_loadu_pd(X + 32);    /* x15 x16 */
        _mm_store_pd(sb + 28, _mm_add_pd(e1, e2));
        _mm_store_pd(db + 28, _mm_sub_pd(e1, e2));
        sb[30] = sb[31] = db[30] = db[31] = 0.0;
    }

    /* y0 = x0 + sum s_j */
    __m512d ts = _mm512_add_pd(_mm512_add_pd(_mm512_load_pd(sb), _mm512_load_pd(sb + 8)),
                               _mm512_add_pd(_mm512_load_pd(sb + 16), _mm512_load_pd(sb + 24)));
    __m256d us = _mm256_add_pd(_mm512_castpd512_pd256(ts), _mm512_extractf64x4_pd(ts, 1));
    __m128d vs = _mm_add_pd(_mm256_castpd256_pd128(us), _mm256_extractf128_pd(us, 1));
    __m128d y0 = _mm_add_pd(vs, _mm_loadu_pd(X));

    /* main: 8 accumulators, scalar broadcasts, 8 FMA per j */
    __m512d Ar0 = _mm512_set1_pd(X[0]), Ar1 = Ar0;
    __m512d Ai0 = _mm512_set1_pd(X[1]), Ai1 = Ai0;
    __m512d Br0 = _mm512_setzero_pd(), Br1 = Br0, Bi0 = Br0, Bi1 = Br0;
#pragma GCC unroll 15
    for (int j = 0; j < 15; ++j) {
        __m512d c0 = _mm512_load_pd(CT31[j]),     c1 = _mm512_load_pd(CT31[j] + 8);
        __m512d s0 = _mm512_load_pd(ST31[j]),     s1 = _mm512_load_pd(ST31[j] + 8);
        __m512d sr = _mm512_set1_pd(sb[2 * j]),   si = _mm512_set1_pd(sb[2 * j + 1]);
        __m512d dr = _mm512_set1_pd(db[2 * j]),   di = _mm512_set1_pd(db[2 * j + 1]);
        Ar0 = _mm512_fmadd_pd(sr, c0, Ar0);  Ar1 = _mm512_fmadd_pd(sr, c1, Ar1);
        Ai0 = _mm512_fmadd_pd(si, c0, Ai0);  Ai1 = _mm512_fmadd_pd(si, c1, Ai1);
        Br0 = _mm512_fmadd_pd(dr, s0, Br0);  Br1 = _mm512_fmadd_pd(dr, s1, Br1);
        Bi0 = _mm512_fmadd_pd(di, s0, Bi0);  Bi1 = _mm512_fmadd_pd(di, s1, Bi1);
    }

    /* combine */
    __m512d P0 = _mm512_add_pd(Ar0, Bi0), P1 = _mm512_add_pd(Ar1, Bi1);
    __m512d Q0 = _mm512_sub_pd(Ai0, Br0), Q1 = _mm512_sub_pd(Ai1, Br1);
    __m512d R0 = _mm512_sub_pd(Ar0, Bi0), R1 = _mm512_sub_pd(Ar1, Bi1);
    __m512d S0 = _mm512_add_pd(Ai0, Br0), S1 = _mm512_add_pd(Ai1, Br1);

    /* interleave / reverse permutes (pad lanes duplicate a valid lane so the
       fused map's sqrt never sees garbage) */
    const __m512i ILA = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i ILB = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    const __m512i ILC = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 6, 14);
    const __m512i RVA = _mm512_setr_epi64(6, 14, 5, 13, 4, 12, 3, 11);
    const __m512i RVB = _mm512_setr_epi64(2, 10, 1, 9, 0, 8, 0, 8);
    const __m512i RVC = _mm512_setr_epi64(7, 15, 6, 14, 5, 13, 4, 12);
    const __m512i RVD = _mm512_setr_epi64(3, 11, 2, 10, 1, 9, 0, 8);
    __m512d f0 = _mm512_permutex2var_pd(P0, ILA, Q0);   /* y1..y4   -> Y+2   */
    __m512d f1 = _mm512_permutex2var_pd(P0, ILB, Q0);   /* y5..y8   -> Y+10  */
    __m512d f2 = _mm512_permutex2var_pd(P1, ILA, Q1);   /* y9..y12  -> Y+18  */
    __m512d f3 = _mm512_permutex2var_pd(P1, ILC, Q1);   /* y13..y15 -> Y+26 (6) */
    __m512d g0 = _mm512_permutex2var_pd(R1, RVA, S1);   /* y16..y19 -> Y+32  */
    __m512d g1 = _mm512_permutex2var_pd(R1, RVB, S1);   /* y20..y22 -> Y+40 (6) */
    __m512d g2 = _mm512_permutex2var_pd(R0, RVC, S0);   /* y23..y26 -> Y+46  */
    __m512d g3 = _mm512_permutex2var_pd(R0, RVD, S0);   /* y27..y30 -> Y+54  */

    if (!mapped) {
        _mm_storeu_pd(Y, y0);
        _mm512_storeu_pd(Y + 2, f0);
        _mm512_storeu_pd(Y + 10, f1);
        _mm512_storeu_pd(Y + 18, f2);
        _mm512_mask_storeu_pd(Y + 26, 0x3F, f3);
        _mm512_storeu_pd(Y + 32, g0);
        _mm512_mask_storeu_pd(Y + 40, 0x3F, g1);
        _mm512_storeu_pd(Y + 46, g2);
        _mm512_storeu_pd(Y + 54, g3);
    } else {
        /* fused map, 31 magnitudes shared over 4 zmm sqrt + 4 zmm div */
        __m512d zf0 = _mm512_add_pd(f0, _mm512_loadu_pd(C + 2));
        __m512d zf1 = _mm512_add_pd(f1, _mm512_loadu_pd(C + 10));
        __m512d zf2 = _mm512_add_pd(f2, _mm512_loadu_pd(C + 18));
        __m512d zf3 = _mm512_add_pd(f3, _mm512_maskz_loadu_pd(0x3F, C + 26));
        __m512d zg0 = _mm512_add_pd(g0, _mm512_loadu_pd(C + 32));
        __m512d zg1 = _mm512_add_pd(g1, _mm512_maskz_loadu_pd(0x3F, C + 40));
        __m512d zg2 = _mm512_add_pd(g2, _mm512_loadu_pd(C + 46));
        __m512d zg3 = _mm512_add_pd(g3, _mm512_loadu_pd(C + 54));
        __m128d z0 = _mm_add_pd(y0, _mm_loadu_pd(C));
        /* ride y0 in zf3's two pad lanes so all 31 magnitudes fit 4 sqrt+div */
        __m512d z0w = _mm512_zextpd128_pd512(z0);
        __m512d z0b = _mm512_permutexvar_pd(
            _mm512_setr_epi64(0, 1, 0, 1, 0, 1, 0, 1), z0w);
        zf3 = _mm512_mask_blend_pd(0xC0, zf3, z0b);
        __m512d r01 = packed_recip(_mm512_mul_pd(zf0, zf0), _mm512_mul_pd(zf1, zf1));
        __m512d r23 = packed_recip(_mm512_mul_pd(zf2, zf2), _mm512_mul_pd(zf3, zf3));
        __m512d r45 = packed_recip(_mm512_mul_pd(zg0, zg0), _mm512_mul_pd(zg1, zg1));
        __m512d r67 = packed_recip(_mm512_mul_pd(zg2, zg2), _mm512_mul_pd(zg3, zg3));
        _mm512_storeu_pd(Y + 2, _mm512_mul_pd(zf0, dup_lo(r01)));
        _mm512_storeu_pd(Y + 10, _mm512_mul_pd(zf1, dup_hi(r01)));
        _mm512_storeu_pd(Y + 18, _mm512_mul_pd(zf2, dup_lo(r23)));
        _mm512_mask_storeu_pd(Y + 26, 0x3F, _mm512_mul_pd(zf3, dup_hi(r23)));
        _mm512_storeu_pd(Y + 32, _mm512_mul_pd(zg0, dup_lo(r45)));
        _mm512_mask_storeu_pd(Y + 40, 0x3F, _mm512_mul_pd(zg1, dup_hi(r45)));
        _mm512_storeu_pd(Y + 46, _mm512_mul_pd(zg2, dup_lo(r67)));
        _mm512_storeu_pd(Y + 54, _mm512_mul_pd(zg3, dup_hi(r67)));
        __m128d r0 = _mm512_castpd512_pd128(
            _mm512_permutexvar_pd(_mm512_setr_epi64(7, 7, 0, 0, 0, 0, 0, 0), r23));
        _mm_storeu_pd(Y, _mm_mul_pd(z0, r0));
    }
}
#endif /* HAVE_AVX512 */

/* -------------------------------------------------------------------------- */
/* generic scalar path (L = 7, 11, 17; and everything without AVX-512)        */
/* -------------------------------------------------------------------------- */
static void kern_generic(const fft1d_plan *p, const double *restrict X,
                         double *restrict Y)
{
    const int L = p->L, h = (L - 1) / 2;
    double sr[16], si[16], dr[16], di[16];
    double y0r = X[0], y0i = X[1];
    for (int j = 1; j <= h; ++j) {
        double ar = X[2 * j], ai = X[2 * j + 1];
        double br = X[2 * (L - j)], bi = X[2 * (L - j) + 1];
        sr[j-1] = ar + br; si[j-1] = ai + bi;
        dr[j-1] = ar - br; di[j-1] = ai - bi;
        y0r += sr[j-1]; y0i += si[j-1];
    }
    Y[0] = y0r; Y[1] = y0i;
    for (int k = 1; k <= h; ++k) {
        double Ar = X[0], Ai = X[1], Br = 0.0, Bi = 0.0;
        const double *c = p->ct + (size_t)(k - 1) * h;
        const double *s = p->st + (size_t)(k - 1) * h;
        for (int j = 0; j < h; ++j) {
            Ar += sr[j] * c[j]; Ai += si[j] * c[j];
            Br += dr[j] * s[j]; Bi += di[j] * s[j];
        }
        Y[2*k] = Ar + Bi;         Y[2*k + 1] = Ai - Br;
        Y[2*(L-k)] = Ar - Bi;     Y[2*(L-k) + 1] = Ai + Br;
    }
}

static void map_generic(const double *restrict z, const double *restrict c,
                        double *restrict o, int L)
{
    for (int i = 0; i < L; ++i) {
        double re = z[2*i] + c[2*i], im = z[2*i+1] + c[2*i+1];
        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
        o[2*i] = re * sc; o[2*i+1] = im * sc;
    }
}

/* -------------------------------------------------------------------------- */
/* API                                                                        */
/* -------------------------------------------------------------------------- */
fft1d_plan *fft1d_create(int L, int batch)
{
    if (!fft1d_supports(L)) return NULL;
    fft1d_plan *p = malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L; p->batch = batch; p->ct = p->st = NULL;
#ifdef HAVE_AVX512
    fill_tables();
    if (L == 13 || L == 31) return p;
#endif
    {
        int h = (L - 1) / 2;
        p->ct = malloc((size_t)2 * h * h * sizeof(double));
        if (!p->ct) { free(p); return NULL; }
        p->st = p->ct + (size_t)h * h;
        for (int k = 1; k <= h; ++k)
            for (int j = 1; j <= h; ++j) {
                double ph = 2.0 * M_PI * (double)((j * k) % L) / L;
                p->ct[(size_t)(k-1) * h + (j-1)] = cos(ph);
                p->st[(size_t)(k-1) * h + (j-1)] = sin(ph);
            }
    }
    return p;
}

void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    const int L = p->L, B = p->batch;
    const double *X = (const double *)in;
    double *Y = (double *)out;
#ifdef HAVE_AVX512
    /* two scratch sets, alternated per volume, so consecutive iterations do not
       serialize through store-forwarding on one stack address */
    if (L == 13) {
        for (int b = 0; b < B; ++b)
            kern13(X + 26 * b, Y + 26 * b, NULL, 0);
        return;
    }
    if (L == 31) {
        double scr[4][32] __attribute__((aligned(64)));
        for (int b = 0; b < B; ++b)
            kern31(X + 62 * b, Y + 62 * b, NULL, 0, scr[2 * (b & 1)], scr[2 * (b & 1) + 1]);
        return;
    }
#endif
    for (int b = 0; b < B; ++b)
        kern_generic(p, X + 2 * (size_t)L * b, Y + 2 * (size_t)L * b);
}

/* Own the whole m-step map chain: state <- (FFT(state)+c)/(1+|FFT(state)+c|).
 * final_out is used as the state buffer; each fused kernel is in-place safe
 * (all state loads complete before the first store). */
void fft1d_chain(fft1d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    const int L = p->L, B = p->batch;
    const size_t n2 = 2 * (size_t)L;
    double *S = (double *)final_out;
    const double *C = (const double *)c;
    memcpy(S, x0, (size_t)L * B * sizeof(double _Complex));
#ifdef HAVE_AVX512
    if (L == 13) {
        for (int s = 0; s < m; ++s)
            for (int b = 0; b < B; ++b)
                kern13(S + 26 * b, S + 26 * b, C + 26 * b, 1);
        return;
    }
    if (L == 31) {
        double scr[4][32] __attribute__((aligned(64)));
        for (int s = 0; s < m; ++s)
            for (int b = 0; b < B; ++b) {
                int r = (b + s) & 1;
                kern31(S + 62 * b, S + 62 * b, C + 62 * b, 1, scr[2 * r], scr[2 * r + 1]);
            }
        return;
    }
#endif
    {
        double tmp[64];
        for (int s = 0; s < m; ++s)
            for (int b = 0; b < B; ++b) {
                kern_generic(p, S + n2 * b, tmp);
                map_generic(tmp, C + n2 * b, S + n2 * b, L);
            }
    }
}

void fft1d_destroy(fft1d_plan *p)
{
    if (!p) return;
    free(p->ct);
    free(p);
}
