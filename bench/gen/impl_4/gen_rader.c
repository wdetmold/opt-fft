/* gen_rader -- Rader-class prime entry, round gen_r1.  Owns L=31.
 *
 * FOLDED RADER: the conjugate-pair fold (s_j = x_j + x_{31-j}, d_j = x_j - x_{31-j})
 * turns the 31-point DFT into
 *     X_k     = x_0 + E_k - i O_k,   X_{31-k} = x_0 + E_k + i O_k,
 *     E_k = sum_j cos(2pi jk/31) s_j,   O_k = sum_j sin(2pi jk/31) d_j,  j,k = 1..15.
 * Indexing both j and k through the multiplicative quotient group Z31* mod {+-1}
 * (cyclic of order 15, generator 3) makes the E system a CYCLIC-15 correlation of
 * s with the cos kernel and the O system a NEGACYCLIC-15 correlation of sign-twisted
 * d with the sin kernel; because 15 is odd, the negacyclic one converts to cyclic via
 * diagonal +-1 twists that fold entirely into the (compile-time) load/store index
 * tables.  Correlation -> convolution by reversing the (precomputed) kernel.
 *
 * Each cyclic-15 convolution is computed as Winograd-C3 (4 block products, 11 block
 * adds) NESTED OVER dense cyclic-5 blocks (25 FMA each): 100 FMA + 65 vector adds
 * per convolution, ALL constants real, so on interleaved complex data every op is a
 * plain zmm add/FMA -- no complex-multiply shuffles anywhere.  Per 4 pencils per
 * axis: ~435 zmm ops vs the folded-dense entry's ~555, with ~40 broadcast constants
 * instead of ~450 table loads.  (Dense C5 blocks beat Winograd C5 blocks on FMA
 * hardware: 25 FMA < 10 mul + 31 add.  The full-Winograd C15 at 40 mul + 179 add
 * would LOSE to this hybrid -- same lesson as ice L23_rader's "121 fused FMAs beat
 * any sub-quadratic length-11 convolution".)
 *
 * Chassis (pass order, z-row kernel, chain scheme, map) ADOPTED from
 * gen_dense_prime gen_r1 (itself from the ice-campaign records):
 *   P1 z rows contiguous (their zpass31 row-pair GEMM, verbatim), P2 x-axis
 *   inner=961, P3 y-axis per-plane in place; volume-resident fused chain;
 *   map = pair-compressed |w|^2, rsqrt14+2NR, ONE vdivpd per 8 points.
 * gen_r2: the chain runs on a fully PADDED private state (z-rows 31 -> 32
 * complex, planes -> 1148 complex == 124 mod 256) -- every access in every
 * pass 64B-aligned, the x-pass's exact 4K store->load aliases (row stride
 * 961*16: store row j+4 == next chunk's load of row j in low-12 bits) pushed
 * outside the 31-row system, z = 8 uniform quads/plane (row 32 is a zeroed
 * pad row), y and map tail-free.  See R31_ZP/R31_PP and r31_chain_volume.
 *
 * gen_r3: the class takes ANY odd prime 3 <= p <= 127 (the round-3 duty).
 * Primes != 31 run a GENERIC folded half-system engine (rp_*): the same
 * conjugate fold, C_k = x0 + sum_j cos(2pi jk/p) u_j, S_k = sum_j sin v_j,
 * X_k = C_k -+ iS_k, computed by a runtime-(p,h) column-chunk kernel with
 * k in quads (4 C + 4 S accumulators sharing each u_j/v_j load; all-real
 * broadcast constants, no complex-multiply shuffles) and the z axis through
 * the 4x4-complex transpose quad (r31_tp4 reused).  The chain runs fully in
 * place on the out volume (r1 form), map per plane.  The 31 fast path is
 * unchanged.  st/cpad for 31 now live in ONE 2MiB huge-page arena with the
 * c mirror at a +2048 B page phase (gen_layout gl_map_huge; gen_dense_prime
 * r3 found two same-phase ~500 KB aligned_allocs make the map's c loads
 * 4K-alias the y-pass state stores).
 *
 * gen_r4: PLANE CUSTODY in both chains (gen_layout r3 / gen_bluestein r4's
 * window idea): step s+1's z-pass is plane-local, so it runs right after each
 * plane's map while the plane is cache-hot -- one full-state read per step
 * deleted, bit-identical outputs (verified by cmp against the r3 binary).
 * Raced and rejected same-core (gen_batchlane r4 protocol): x-pass software
 * prefetch (-DR31_PFX, +0.7%) and the map fused into the z-quads' transpose-in
 * loads (-DR31_ZMAPF, +4% -- the panel's FIFTH map-fusion negative, first on
 * the load side).  -DR31_R3CHAIN / -DRP_R3CHAIN restore the r3 pass order.
 *
 * create() SELF-CHECKS the fast engine against a dense reference volume at 1e-13
 * and falls back to the (slow, correct) dense-matrix path if the check fails --
 * a fast wrong answer scores nothing.
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include "../fft3d_api.h"

typedef double _Complex cplx;

static const long double PIL = 3.141592653589793238462643383279502884L;

/* ---- index tables (derived + verified against a reference DFT offline) ----
 * slot n = 5*i + j maps to CRT point ((10i+6j) mod 15) of the cyclic-15 system.
 * JS:  E-sweep fill:  S[n] = x[JS] + x[31-JS]              (s fold, reversal baked in)
 * JDP/JDM: O-sweep fill: S[n] = x[JDP] - x[JDM]            (d fold; (-1)^q eps_q twist
 *                                                           baked into the p/m swap)
 * KP/KM: output rows: X[KP] = T - iO~, X[KM] = T + iO~     (eps_t (-1)^t twist baked
 *                                                           into the pair order)    */
static const int R31_JS[15]  = {1, 2, 4, 8, 15, 5, 10, 11, 9, 13, 6, 12, 7, 14, 3};
static const int R31_JDP[15] = {1, 2, 4, 8, 16, 5, 10, 20, 9, 18, 25, 19, 7, 14, 28};
static const int R31_JDM[15] = {30, 29, 27, 23, 15, 26, 21, 11, 22, 13, 6, 12, 24, 17, 3};
static const int R31_KP[15]  = {1, 16, 8, 4, 2, 25, 28, 14, 7, 19, 5, 18, 9, 20, 10};
static const int R31_KM[15]  = {30, 15, 23, 27, 29, 6, 3, 17, 24, 12, 26, 13, 22, 11, 21};

/* Padded chain-state layout.  z-rows padded 31 -> R31_ZP = 32 complex (512 B:
 * every z/y-pass row access 64 B-aligned; natural 496 B rows line-split 3 of 4
 * accesses).  Planes padded 992 -> R31_PP = 1148 complex, == 124 mod 256, so
 * the x-pass row stride is 1148*16 B == 31*64 B mod 4096: the nearest 4K
 * store->load alias sits at row distance 33 -- outside the 31-row system --
 * where the natural 961-pitch put a store to row j+4 at EXACTLY the low-12
 * bits of the next chunk's load of row j.  Pad slots are zeroed once at
 * create(); every pass maps zeros to zeros (columns never mix), so they stay
 * zero and cost only ~3% extra x-pass/map lanes -- bought back by tail-free
 * uniform chunks everywhere (992 = 124 x 8, 32 cols = 8 chunks, z = 8 quads). */
#define R31_ZP 32
#define R31_PP 1148

struct fft3d_plan {
    int L, batch;
    int h;               /* (L-1)/2 */
    int fast;            /* 1: AVX-512 Rader engine passed the self-check */
    double *ke, *ko;     /* transformed conv kernels: 4 blocks x 5 doubles each */
    double *ctd, *std_;  /* z-pass duplicated-pair trig tables [15][32] */
    double *gct, *gst;   /* generic-prime fold tables, k-major [h][h] (L != 31) */
    cplx *t1;            /* scratch volume */
    cplx *st;            /* padded chain state, 31 planes x R31_PP (pads zeroed) */
    cplx *cpad;          /* padded mirror of the chain's c volume, same layout */
    cplx *w;             /* dense LxL DFT matrix (self-check + fallback) */
    cplx *tmp;           /* fallback scratch volume */
    void *arena;         /* 2MiB huge-page arena backing st+cpad (may be NULL) */
    size_t alen;
};

const char *fft3d_name(void) { return "gen_rader"; }
const char *fft3d_description(void)
{
    return "Rader-class primes 3..127: at 31, conjugate fold -> cyclic-15 (cos) + "
           "negacyclic-15 (sin; odd-N sign-twist), Winograd-C3 x dense-C5 on a fully "
           "padded huge-page arena (64B-aligned, anti-4K pitch, c mirror phase-split); "
           "any other prime via a generic folded half-system engine (runtime k-quad "
           "chunk kernel + transpose z-quads), in-place chain, self-check gated; "
           "s6 map adopted from gen_dense_prime";
}
static int rp_is_prime(int n)
{
    if (n < 2) return 0;
    for (int d = 2; d * d <= n; ++d)
        if (n % d == 0) return 0;
    return 1;
}
int fft3d_supports(int L) { return L >= 3 && L <= 127 && rp_is_prime(L); }

/* ---------------- plan-time tables ---------------- */

/* Winograd-C3 kernel-side transform of a cyclic-15 kernel K (long double in):
 * blocks H_i[j] = K[(10i+6j) mod 15]; out = { (H0+H1+H2)/3, (H0-H2)/3,
 * (H1-H2)/3, (H0-H1)/3 }, 20 doubles. */
static void r31_kernel_transform(const long double *K, double *out)
{
    long double H[3][5];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 5; ++j)
            H[i][j] = K[(10 * i + 6 * j) % 15];
    for (int j = 0; j < 5; ++j) {
        out[j]      = (double)((H[0][j] + H[1][j] + H[2][j]) / 3.0L);
        out[5 + j]  = (double)((H[0][j] - H[2][j]) / 3.0L);
        out[10 + j] = (double)((H[1][j] - H[2][j]) / 3.0L);
        out[15 + j] = (double)((H[0][j] - H[1][j]) / 3.0L);
    }
}

/* E kernel: CC_r = cos(2pi 3^r/31); O kernel: SN'_r = (-1)^r sin(2pi 3^r/31). */
static void r31_build_kernels(double *ke, double *ko)
{
    long double CC[15], SN[15];
    long g = 1;
    for (int r = 0; r < 15; ++r) {
        long double th = 2.0L * PIL * (long double)g / 31.0L;
        CC[r] = cosl(th);
        SN[r] = (r & 1) ? -sinl(th) : sinl(th);
        g = (g * 3) % 31;
    }
    r31_kernel_transform(CC, ke);
    r31_kernel_transform(SN, ko);
}

/* duplicated-pair trig layout for the z-pass (from gen_dense_prime):
 * row j-1 holds (w_{j,0}, w_{j,0}, ..., w_{j,15}, w_{j,15}) = 32 doubles */
static double *r31_trig_dup(int want_sin)
{
    double *t = aligned_alloc(64, 15 * 32 * sizeof(double));
    if (!t) return NULL;
    for (int j = 1; j <= 15; ++j)
        for (int k = 0; k <= 15; ++k) {
            long m = ((long)j * k) % 31;
            long double th = 2.0L * PIL * (long double)m / 31.0L;
            double w = want_sin ? (double)sinl(th) : (double)cosl(th);
            t[(size_t)(j - 1) * 32 + 2 * k]     = w;
            t[(size_t)(j - 1) * 32 + 2 * k + 1] = w;
        }
    return t;
}

/* generic-prime fold tables, k-major so a k-quad walks 4 linear rows:
 * t[(k-1)*h + (j-1)] = cos/sin(2pi jk/p), j,k = 1..h, exact long-double args */
static double *rp_trig(int p, int h, int want_sin)
{
    double *t = aligned_alloc(64, ((size_t)h * h * sizeof(double) + 63) & ~(size_t)63);
    if (!t) return NULL;
    for (int k = 1; k <= h; ++k)
        for (int j = 1; j <= h; ++j) {
            long m = ((long)j * k) % p;
            long double th = 2.0L * PIL * (long double)m / (long double)p;
            t[(size_t)(k - 1) * h + (j - 1)] =
                want_sin ? (double)sinl(th) : (double)cosl(th);
        }
    return t;
}

#ifdef __AVX512F__

/* ---------------- the Winograd C3 x dense-C5 cyclic-15 convolution ----------------
 * S[15] in CRT slot order -> Y[15] same order; kt = 20 transformed kernel doubles.
 * If esum != NULL, also emits sum of the E block (= sum of all 15 inputs). */

#define R31_C5(DST, SRC, KB) do {                                              \
    __m512d k0 = _mm512_set1_pd((KB)[0]), k1 = _mm512_set1_pd((KB)[1]);        \
    __m512d k2 = _mm512_set1_pd((KB)[2]), k3 = _mm512_set1_pd((KB)[3]);        \
    __m512d k4 = _mm512_set1_pd((KB)[4]);                                      \
    DST[0] = _mm512_mul_pd(SRC[0], k0);                                        \
    DST[1] = _mm512_mul_pd(SRC[0], k1);                                        \
    DST[2] = _mm512_mul_pd(SRC[0], k2);                                        \
    DST[3] = _mm512_mul_pd(SRC[0], k3);                                        \
    DST[4] = _mm512_mul_pd(SRC[0], k4);                                        \
    DST[0] = _mm512_fmadd_pd(SRC[1], k4, DST[0]);                              \
    DST[1] = _mm512_fmadd_pd(SRC[1], k0, DST[1]);                              \
    DST[2] = _mm512_fmadd_pd(SRC[1], k1, DST[2]);                              \
    DST[3] = _mm512_fmadd_pd(SRC[1], k2, DST[3]);                              \
    DST[4] = _mm512_fmadd_pd(SRC[1], k3, DST[4]);                              \
    DST[0] = _mm512_fmadd_pd(SRC[2], k3, DST[0]);                              \
    DST[1] = _mm512_fmadd_pd(SRC[2], k4, DST[1]);                              \
    DST[2] = _mm512_fmadd_pd(SRC[2], k0, DST[2]);                              \
    DST[3] = _mm512_fmadd_pd(SRC[2], k1, DST[3]);                              \
    DST[4] = _mm512_fmadd_pd(SRC[2], k2, DST[4]);                              \
    DST[0] = _mm512_fmadd_pd(SRC[3], k2, DST[0]);                              \
    DST[1] = _mm512_fmadd_pd(SRC[3], k3, DST[1]);                              \
    DST[2] = _mm512_fmadd_pd(SRC[3], k4, DST[2]);                              \
    DST[3] = _mm512_fmadd_pd(SRC[3], k0, DST[3]);                              \
    DST[4] = _mm512_fmadd_pd(SRC[3], k1, DST[4]);                              \
    DST[0] = _mm512_fmadd_pd(SRC[4], k1, DST[0]);                              \
    DST[1] = _mm512_fmadd_pd(SRC[4], k2, DST[1]);                              \
    DST[2] = _mm512_fmadd_pd(SRC[4], k3, DST[2]);                              \
    DST[3] = _mm512_fmadd_pd(SRC[4], k4, DST[3]);                              \
    DST[4] = _mm512_fmadd_pd(SRC[4], k0, DST[4]);                              \
} while (0)

static inline __attribute__((always_inline))
void r31_wino15(const __m512d *S, const double *kt, __m512d *Y, __m512d *esum)
{
    __m512d E[5], A[5], B[5], C[5], M0[5], M1[5], M2[5], M3[5];
    for (int j = 0; j < 5; ++j) {
        E[j] = _mm512_add_pd(_mm512_add_pd(S[j], S[5 + j]), S[10 + j]);
        A[j] = _mm512_sub_pd(S[j], S[10 + j]);
        B[j] = _mm512_sub_pd(S[5 + j], S[10 + j]);
        C[j] = _mm512_sub_pd(S[j], S[5 + j]);
    }
    if (esum)
        *esum = _mm512_add_pd(_mm512_add_pd(_mm512_add_pd(E[0], E[1]),
                                            _mm512_add_pd(E[2], E[3])), E[4]);
    R31_C5(M0, E, kt);
    R31_C5(M1, A, kt + 5);
    R31_C5(M2, B, kt + 10);
    R31_C5(M3, C, kt + 15);
    const __m512d TWO = _mm512_set1_pd(2.0);
    for (int j = 0; j < 5; ++j) {
        __m512d t01 = _mm512_add_pd(M0[j], M1[j]);
        Y[j]      = _mm512_fnmadd_pd(M2[j], TWO, _mm512_add_pd(t01, M3[j]));
        Y[5 + j]  = _mm512_fnmadd_pd(M3[j], TWO, _mm512_add_pd(t01, M2[j]));
        __m512d t23 = _mm512_add_pd(M2[j], M3[j]);
        Y[10 + j] = _mm512_fnmadd_pd(M1[j], TWO, _mm512_add_pd(M0[j], t23));
    }
}

/* s6 map ladder pieces (arithmetic identical to map_volume; every ladder op is
 * elementwise, so where it runs cannot change per-point bits).  Pair form: one
 * vdivpd per two output vectors; single form for the lone X0 vector. */
/* hwdiv != 0: end the ladder in vdivpd (bit-identical to map_volume);
 * hwdiv == 0: divider-free rcp14 + 2 Newton (sub-ulp) -- the ice L23 r4
 * lesson: an eager STORE-side fused map must not end in vdivpd right before
 * the stores.  Compile-time constant at every call site. */
static inline __attribute__((always_inline)) __m512d r31_map_rec(__m512d m2,
                                                                 const int hwdiv)
{
    const __m512d ONE  = _mm512_set1_pd(1.0);
    const __m512d TH   = _mm512_set1_pd(1.5);
    const __m512d HALF = _mm512_set1_pd(0.5);
    const __m512d TINY = _mm512_set1_pd(1e-300);
    __m512d m2c = _mm512_max_pd(m2, TINY);
    __m512d r = _mm512_rsqrt14_pd(m2c);
    __m512d hm = _mm512_mul_pd(m2c, HALF);
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hm, r), r, TH));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hm, r), r, TH));
    __m512d d = _mm512_fmadd_pd(m2c, r, ONE);                  /* 1 + |w| */
    if (hwdiv)
        return _mm512_div_pd(ONE, d);
    const __m512d TWO = _mm512_set1_pd(2.0);
    __m512d rec = _mm512_rcp14_pd(d);
    rec = _mm512_mul_pd(rec, _mm512_fnmadd_pd(d, rec, TWO));
    rec = _mm512_mul_pd(rec, _mm512_fnmadd_pd(d, rec, TWO));
    return rec;
}

#ifdef R31_FUSE_DIV
#define R31_FMDIV 1     /* store-side-fusion map style: raced, LOSES with div */
#else
#define R31_FMDIV 0
#endif

static inline __attribute__((always_inline))
void r31_map2(__m512d wa, __m512d wb, __m512d *oa, __m512d *ob, const int hwdiv)
{
    __m512d pa = _mm512_mul_pd(wa, wa), pb = _mm512_mul_pd(wb, wb);
    __m512d m2 = _mm512_add_pd(_mm512_unpacklo_pd(pa, pb),
                               _mm512_unpackhi_pd(pa, pb));
    __m512d rec = r31_map_rec(m2, hwdiv);
    *oa = _mm512_mul_pd(wa, _mm512_unpacklo_pd(rec, rec));
    *ob = _mm512_mul_pd(wb, _mm512_unpackhi_pd(rec, rec));
}

static inline __attribute__((always_inline)) __m512d r31_map1(__m512d w,
                                                              const int hwdiv)
{
    __m512d p = _mm512_mul_pd(w, w);
    __m512d m2 = _mm512_add_pd(p, _mm512_permute_pd(p, 0x55));
    return _mm512_mul_pd(w, r31_map_rec(m2, hwdiv));
}

/* one column chunk (up to 4 complex = 8 doubles wide) of a 31 x inner pass.
 * sx/dx already offset to the chunk; rs = row stride in doubles.  In-place safe:
 * within the chunk every load of a row precedes every store to it (E sweep writes
 * only row 0, which the O sweep never reads).  NO restrict here -- the y-pass runs
 * dst == src and the compiler must keep the load/store order.
 * mapc == NULL: plain FFT stores.  mapc != NULL: chain mode -- every output is
 * mapped ((X+c)/(1+|X+c|)) at the store, c chunk at mapc, same row strides. */
static inline __attribute__((always_inline))
void r31_chunk(const double *sx, double *dx, const double *mapc,
               const ptrdiff_t rs, int full,
               __mmask8 msk, const double *ke, const double *ko)
{
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    __m512d S[15], Y[15], T[15];
#ifdef R31_ONEBODY
    (void)full;   /* single always-masked body: halves the hot code footprint */
#define R31_LD(px, off) _mm512_maskz_loadu_pd(msk, (px) + (off))
#define R31_ST(off, v)  _mm512_mask_storeu_pd(dx + (off), msk, v)
#else
#define R31_LD(px, off) (full ? _mm512_loadu_pd((px) + (off)) \
                              : _mm512_maskz_loadu_pd(msk, (px) + (off)))
#define R31_ST(off, v) do { if (full) _mm512_storeu_pd(dx + (off), v); \
                            else _mm512_mask_storeu_pd(dx + (off), msk, v); } while (0)
#endif
    __m512d x0 = R31_LD(sx, 0);
    for (int n = 0; n < 15; ++n) {
        const ptrdiff_t j = R31_JS[n];
        S[n] = _mm512_add_pd(R31_LD(sx, j * rs), R31_LD(sx, (31 - j) * rs));
    }
    __m512d esum;
    r31_wino15(S, ke, Y, &esum);
    __m512d X0 = _mm512_add_pd(x0, esum);
    if (mapc)
        X0 = r31_map1(_mm512_add_pd(X0, R31_LD(mapc, 0)), R31_FMDIV);
    R31_ST(0, X0);
    for (int n = 0; n < 15; ++n) T[n] = _mm512_add_pd(x0, Y[n]);
    for (int n = 0; n < 15; ++n)
        S[n] = _mm512_sub_pd(R31_LD(sx, (ptrdiff_t)R31_JDP[n] * rs),
                             R31_LD(sx, (ptrdiff_t)R31_JDM[n] * rs));
    r31_wino15(S, ko, Y, NULL);
    for (int n = 0; n < 15; ++n) {
        __m512d o = _mm512_permute_pd(Y[n], 0x55);              /* swap re/im */
        __m512d xp = _mm512_fmadd_pd(o, SG, T[n]);
        __m512d xm = _mm512_fnmadd_pd(o, SG, T[n]);
        if (mapc) {
            xp = _mm512_add_pd(xp, R31_LD(mapc, (ptrdiff_t)R31_KP[n] * rs));
            xm = _mm512_add_pd(xm, R31_LD(mapc, (ptrdiff_t)R31_KM[n] * rs));
            r31_map2(xp, xm, &xp, &xm, R31_FMDIV);
        }
        R31_ST((ptrdiff_t)R31_KP[n] * rs, xp);
        R31_ST((ptrdiff_t)R31_KM[n] * rs, xm);
    }
#undef R31_LD
#undef R31_ST
}

/* contract the slowest axis of a (31 x ncols) complex block whose rows sit at
 * `pitch` complex apart (pitch == ncols: flat; pitch > ncols: padded rows);
 * dst may equal src when c == NULL (plain).  c != NULL: map-fused stores
 * (dst must be distinct; c rows at the same pitch).
 * pf != 0: software-prefetch each row's line `pf` bytes ahead of the current
 * chunk's loads (gen_layout gen_r4: the fold's ~62 row streams outrun what
 * the DCU prefetcher tracks; T0 pays where the streams miss L1 -> L2). */
static inline __attribute__((always_inline))
void r31_pass_core(const cplx *src, cplx *dst, const cplx *c,
                   const ptrdiff_t ncols, const ptrdiff_t pitch,
                   const int pf, const double *ke, const double *ko)
{
    const ptrdiff_t rs = 2 * pitch;
    const ptrdiff_t nd = 2 * ncols;
    const double *sx = (const double *)src;
    const double *cx = (const double *)c;
    double *dx = (double *)dst;
#ifdef R31_ONEBODY
    for (ptrdiff_t d = 0; d < nd; d += 8) {
        __mmask8 msk = (nd - d >= 8) ? (__mmask8)0xFF
                                     : (__mmask8)((1u << (nd - d)) - 1);
        r31_chunk(sx + d, dx + d, cx ? cx + d : NULL, rs, 0, msk, ke, ko);
    }
    (void)pf;
#else
    ptrdiff_t d = 0;
    for (; d + 8 <= nd; d += 8) {
        if (pf)
            for (int j = 0; j < 31; ++j)
                _mm_prefetch((const char *)(sx + (ptrdiff_t)j * rs + d) + pf,
                             _MM_HINT_T0);
        r31_chunk(sx + d, dx + d, cx ? cx + d : NULL, rs, 1, (__mmask8)0xFF, ke, ko);
    }
    if (d < nd)
        r31_chunk(sx + d, dx + d, cx ? cx + d : NULL, rs, 0,
                  (__mmask8)((1u << (nd - d)) - 1), ke, ko);
#endif
}

/* -DR31_SCHEDP: pre-RA pressure scheduling on the chunk instantiators only
 * (gen_batchlane r2 / gen_powp r1: pays on spill-bound bodies, as attribute
 * not global flags; this kernel spills ~29 moves/chunk). */
#ifdef R31_SCHEDP
#define R31_SCHED_ATTR __attribute__((optimize("schedule-insns", "sched-pressure")))
#else
#define R31_SCHED_ATTR
#endif

/* x-pass prefetch distance in bytes (0 disables; 128 = 2 chunks ahead).
 * Raced gen_r4 same-core: +0.7% LOSS at the graded cell (the 31 extra
 * port-2/3 uops/chunk cost more than the L2-hit latency they hide -- the
 * state is L2-resident, unlike gen_layout's DRAM-resident demo where the
 * same recipe won).  Default OFF; knob kept for the cross-arch race. */
#ifndef R31_PFX
#define R31_PFX 0
#endif

static R31_SCHED_ATTR
void r31_pass_x(const cplx *s, cplx *d, const double *ke, const double *ko)
{ r31_pass_core(s, d, NULL, 31 * 31, 31 * 31, R31_PFX, ke, ko); }

/* x pass over the padded state: 992 columns per plane (31 pad columns of
 * zeros ride along -- tail-free), planes R31_PP apart */
static R31_SCHED_ATTR
void r31_pass_xp(cplx *st, const double *ke, const double *ko)
{ r31_pass_core(st, st, NULL, 31 * R31_ZP, R31_PP, R31_PFX, ke, ko); }

static R31_SCHED_ATTR
void r31_pass_y(const cplx *s, cplx *d, const double *ke, const double *ko)
{ r31_pass_core(s, d, NULL, 31, 31, 0, ke, ko); }

/* y pass on one padded plane: rows at R31_ZP, 32 columns -> 8 full chunks */
static R31_SCHED_ATTR
void r31_pass_yp(cplx *pl, const double *ke, const double *ko)
{ r31_pass_core(pl, pl, NULL, R31_ZP, R31_ZP, 0, ke, ko); }

/* y pass with the map fused at every store: src = x-pass output plane (t1),
 * dst = state plane, c = map-constant plane */
static __attribute__((unused))
void r31_pass_ym(const cplx *s, cplx *d, const cplx *c,
                 const double *ke, const double *ko)
{ r31_pass_core(s, d, c, 31, 31, 0, ke, ko); }

/* ---------------- z-axis pass, Rader form: 4 rows via 4x4-complex transposes ----
 * Four contiguous rows are transposed (8 vshuff64x2 per 4x4-complex tile) into a
 * stack array where element j of the 4 pencils is one zmm at stride 8 doubles,
 * the SAME r31_chunk kernel runs on it (all offsets compile-time), and the result
 * transposes back.  ~110 arith/pencil + 32 shuffles/pencil vs the dense row-GEMM's
 * ~240 FMA/pencil.  In-place safe: all loads of the 4 rows precede all stores. */
static inline __attribute__((always_inline))
void r31_tp4(const __m512d a, const __m512d b, const __m512d c, const __m512d d,
             __m512d *o0, __m512d *o1, __m512d *o2, __m512d *o3)
{
    __m512d t0 = _mm512_shuffle_f64x2(a, b, 0x44);
    __m512d t1 = _mm512_shuffle_f64x2(a, b, 0xEE);
    __m512d t2 = _mm512_shuffle_f64x2(c, d, 0x44);
    __m512d t3 = _mm512_shuffle_f64x2(c, d, 0xEE);
    *o0 = _mm512_shuffle_f64x2(t0, t2, 0x88);
    *o1 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
    *o2 = _mm512_shuffle_f64x2(t1, t3, 0x88);
    *o3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
}

/* srd/drd: source/dest row strides in DOUBLES (62 flat, 64 padded); always
 * called with compile-time constants so each wrapper folds its offsets. */
static inline __attribute__((always_inline))
void r31_zquad_core(const cplx *src, cplx *dst, const ptrdiff_t srd,
                    const ptrdiff_t drd, const double *ke, const double *ko)
{
    __attribute__((aligned(64))) double xt[32 * 8], yt[32 * 8];
    const double *x = (const double *)src;
    double *y = (double *)dst;
    for (int t = 0; t < 8; ++t) {           /* transpose in: tile t = elements 4t..4t+3 */
        __m512d r0, r1, r2, r3;
        if (t < 7) {
            r0 = _mm512_loadu_pd(x + 8 * t);
            r1 = _mm512_loadu_pd(x + srd + 8 * t);
            r2 = _mm512_loadu_pd(x + 2 * srd + 8 * t);
            r3 = _mm512_loadu_pd(x + 3 * srd + 8 * t);
        } else {                            /* elements 28..30 only */
            r0 = _mm512_maskz_loadu_pd(0x3F, x + 56);
            r1 = _mm512_maskz_loadu_pd(0x3F, x + srd + 56);
            r2 = _mm512_maskz_loadu_pd(0x3F, x + 2 * srd + 56);
            r3 = _mm512_maskz_loadu_pd(0x3F, x + 3 * srd + 56);
        }
        __m512d o0, o1, o2, o3;
        r31_tp4(r0, r1, r2, r3, &o0, &o1, &o2, &o3);
        _mm512_store_pd(xt + (4 * t) * 8, o0);
        _mm512_store_pd(xt + (4 * t + 1) * 8, o1);
        _mm512_store_pd(xt + (4 * t + 2) * 8, o2);
        _mm512_store_pd(xt + (4 * t + 3) * 8, o3);
    }
    r31_chunk(xt, yt, NULL, 8, 1, (__mmask8)0xFF, ke, ko);
    for (int t = 0; t < 8; ++t) {           /* transpose out */
        __m512d o0, o1, o2, o3;
        r31_tp4(_mm512_load_pd(yt + (4 * t) * 8),
                _mm512_load_pd(yt + (4 * t + 1) * 8),
                _mm512_load_pd(yt + (4 * t + 2) * 8),
                (t < 7) ? _mm512_load_pd(yt + (4 * t + 3) * 8) : _mm512_setzero_pd(),
                &o0, &o1, &o2, &o3);
        if (t < 7) {
            _mm512_storeu_pd(y + 8 * t, o0);
            _mm512_storeu_pd(y + drd + 8 * t, o1);
            _mm512_storeu_pd(y + 2 * drd + 8 * t, o2);
            _mm512_storeu_pd(y + 3 * drd + 8 * t, o3);
        } else {
            _mm512_mask_storeu_pd(y + 56, 0x3F, o0);
            _mm512_mask_storeu_pd(y + drd + 56, 0x3F, o1);
            _mm512_mask_storeu_pd(y + 2 * drd + 56, 0x3F, o2);
            _mm512_mask_storeu_pd(y + 3 * drd + 56, 0x3F, o3);
        }
    }
}

static void r31_zquad(const cplx *src, cplx *dst, const double *ke, const double *ko)
{ r31_zquad_core(src, dst, 62, 62, ke, ko); }           /* flat, execute() */

static void r31_zquad_pp(const cplx *src, cplx *dst, const double *ke, const double *ko)
{ r31_zquad_core(src, dst, 2 * R31_ZP, 2 * R31_ZP, ke, ko); }  /* padded in place */

static void r31_zquad_fp(const cplx *src, cplx *dst, const double *ke, const double *ko)
{ r31_zquad_core(src, dst, 62, 2 * R31_ZP, ke, ko); }   /* step 0: flat x0 -> padded */

/* padded in-place z quad with the s6 MAP APPLIED AT THE TRANSPOSE-IN LOADS:
 * computes zquad(map(x + c)) for 4 rows, replacing map_volume + r31_zquad_pp
 * on those rows.  The map is elementwise and every ladder op is lanewise, so
 * per-element bits are IDENTICAL to the separate sweep (only which lanes share
 * a zmm changes); pad lanes/rows stay zero (maskz load -> w=0 -> map 0).
 * Unlike the four store-side map fusions this panel measured and killed
 * (r31 y-stores r1, dense_prime r2, rp y r3, batchlane epilogue r4), this is
 * LOAD-side fusion into a register-light transpose phase: ~10 live zmm before
 * the ladder, not ~30. */
static __attribute__((unused))
void r31_zquad_mp(const cplx *src, cplx *dst, const cplx *c,
                  const double *ke, const double *ko)
{
    __attribute__((aligned(64))) double xt[32 * 8], yt[32 * 8];
    const ptrdiff_t rd = 2 * R31_ZP;
    const double *x = (const double *)src;
    const double *cx = (const double *)c;
    double *y = (double *)dst;
    for (int t = 0; t < 8; ++t) {
        __m512d r0, r1, r2, r3;
        if (t < 7) {
            r0 = _mm512_add_pd(_mm512_loadu_pd(x + 8 * t),
                               _mm512_loadu_pd(cx + 8 * t));
            r1 = _mm512_add_pd(_mm512_loadu_pd(x + rd + 8 * t),
                               _mm512_loadu_pd(cx + rd + 8 * t));
            r2 = _mm512_add_pd(_mm512_loadu_pd(x + 2 * rd + 8 * t),
                               _mm512_loadu_pd(cx + 2 * rd + 8 * t));
            r3 = _mm512_add_pd(_mm512_loadu_pd(x + 3 * rd + 8 * t),
                               _mm512_loadu_pd(cx + 3 * rd + 8 * t));
        } else {                              /* elements 28..30 + pad col */
            r0 = _mm512_add_pd(_mm512_maskz_loadu_pd(0x3F, x + 56),
                               _mm512_maskz_loadu_pd(0x3F, cx + 56));
            r1 = _mm512_add_pd(_mm512_maskz_loadu_pd(0x3F, x + rd + 56),
                               _mm512_maskz_loadu_pd(0x3F, cx + rd + 56));
            r2 = _mm512_add_pd(_mm512_maskz_loadu_pd(0x3F, x + 2 * rd + 56),
                               _mm512_maskz_loadu_pd(0x3F, cx + 2 * rd + 56));
            r3 = _mm512_add_pd(_mm512_maskz_loadu_pd(0x3F, x + 3 * rd + 56),
                               _mm512_maskz_loadu_pd(0x3F, cx + 3 * rd + 56));
        }
        r31_map2(r0, r1, &r0, &r1, 1);      /* vdivpd: bit-identical to the
                                             * separate map_volume sweep */
        r31_map2(r2, r3, &r2, &r3, 1);
        __m512d o0, o1, o2, o3;
        r31_tp4(r0, r1, r2, r3, &o0, &o1, &o2, &o3);
        _mm512_store_pd(xt + (4 * t) * 8, o0);
        _mm512_store_pd(xt + (4 * t + 1) * 8, o1);
        _mm512_store_pd(xt + (4 * t + 2) * 8, o2);
        _mm512_store_pd(xt + (4 * t + 3) * 8, o3);
    }
    r31_chunk(xt, yt, NULL, 8, 1, (__mmask8)0xFF, ke, ko);
    for (int t = 0; t < 8; ++t) {
        __m512d o0, o1, o2, o3;
        r31_tp4(_mm512_load_pd(yt + (4 * t) * 8),
                _mm512_load_pd(yt + (4 * t + 1) * 8),
                _mm512_load_pd(yt + (4 * t + 2) * 8),
                (t < 7) ? _mm512_load_pd(yt + (4 * t + 3) * 8) : _mm512_setzero_pd(),
                &o0, &o1, &o2, &o3);
        if (t < 7) {
            _mm512_storeu_pd(y + 8 * t, o0);
            _mm512_storeu_pd(y + rd + 8 * t, o1);
            _mm512_storeu_pd(y + 2 * rd + 8 * t, o2);
            _mm512_storeu_pd(y + 3 * rd + 8 * t, o3);
        } else {
            _mm512_mask_storeu_pd(y + 56, 0x3F, o0);
            _mm512_mask_storeu_pd(y + rd + 56, 0x3F, o1);
            _mm512_mask_storeu_pd(y + 2 * rd + 56, 0x3F, o2);
            _mm512_mask_storeu_pd(y + 3 * rd + 56, 0x3F, o3);
        }
    }
}

/* ---------------- z-axis pass: contiguous rows ----------------
 * ADOPTED VERBATIM from gen_dense_prime gen_r1 (their zpass31/zpass31_pair):
 * k dimension in 4 zmm per half-spectrum, u/v broadcast as 128-bit pairs,
 * rows in PAIRS sharing the 8 table loads per j. */
static void r31_zrow_pair(const cplx *restrict src, cplx *restrict dst,
                          ptrdiff_t sp, ptrdiff_t dp,
                          const double *restrict ctd, const double *restrict std)
{
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    const double *xA = (const double *)src;
    const double *xB = (const double *)(src + sp);
    double *yA = (double *)dst;
    double *yB = (double *)(dst + dp);

    __attribute__((aligned(64))) double ua[32], va[32], ub2[32], vb2[32];
    for (int j = 1; j <= 15; ++j) {
        __m128d a = _mm_loadu_pd(xA + 2 * j), b = _mm_loadu_pd(xA + 2 * (31 - j));
        _mm_store_pd(ua + 2 * j, _mm_add_pd(a, b));
        _mm_store_pd(va + 2 * j, _mm_sub_pd(a, b));
        __m128d c = _mm_loadu_pd(xB + 2 * j), e = _mm_loadu_pd(xB + 2 * (31 - j));
        _mm_store_pd(ub2 + 2 * j, _mm_add_pd(c, e));
        _mm_store_pd(vb2 + 2 * j, _mm_sub_pd(c, e));
    }
    __m512d xa0 = _mm512_broadcast_f64x2(_mm_loadu_pd(xA));
    __m512d xb0 = _mm512_broadcast_f64x2(_mm_loadu_pd(xB));
    __m512d CA0 = xa0, CA1 = xa0, CA2 = xa0, CA3 = xa0;
    __m512d CB0 = xb0, CB1 = xb0, CB2 = xb0, CB3 = xb0;
    __m512d SA0 = _mm512_setzero_pd(), SA1 = SA0, SA2 = SA0, SA3 = SA0;
    __m512d SB0 = SA0, SB1 = SA0, SB2 = SA0, SB3 = SA0;
    for (int j = 1; j <= 15; ++j) {
        const double *cr = ctd + (size_t)(j - 1) * 32;
        const double *sr = std + (size_t)(j - 1) * 32;
        __m512d c0 = _mm512_load_pd(cr + 0),  c1 = _mm512_load_pd(cr + 8);
        __m512d c2 = _mm512_load_pd(cr + 16), c3 = _mm512_load_pd(cr + 24);
        __m512d s0 = _mm512_load_pd(sr + 0),  s1 = _mm512_load_pd(sr + 8);
        __m512d s2 = _mm512_load_pd(sr + 16), s3 = _mm512_load_pd(sr + 24);
        __m512d uA = _mm512_broadcast_f64x2(_mm_load_pd(ua + 2 * j));
        __m512d vA = _mm512_broadcast_f64x2(_mm_load_pd(va + 2 * j));
        __m512d uB = _mm512_broadcast_f64x2(_mm_load_pd(ub2 + 2 * j));
        __m512d vB = _mm512_broadcast_f64x2(_mm_load_pd(vb2 + 2 * j));
        CA0 = _mm512_fmadd_pd(c0, uA, CA0); CA1 = _mm512_fmadd_pd(c1, uA, CA1);
        CA2 = _mm512_fmadd_pd(c2, uA, CA2); CA3 = _mm512_fmadd_pd(c3, uA, CA3);
        SA0 = _mm512_fmadd_pd(s0, vA, SA0); SA1 = _mm512_fmadd_pd(s1, vA, SA1);
        SA2 = _mm512_fmadd_pd(s2, vA, SA2); SA3 = _mm512_fmadd_pd(s3, vA, SA3);
        CB0 = _mm512_fmadd_pd(c0, uB, CB0); CB1 = _mm512_fmadd_pd(c1, uB, CB1);
        CB2 = _mm512_fmadd_pd(c2, uB, CB2); CB3 = _mm512_fmadd_pd(c3, uB, CB3);
        SB0 = _mm512_fmadd_pd(s0, vB, SB0); SB1 = _mm512_fmadd_pd(s1, vB, SB1);
        SB2 = _mm512_fmadd_pd(s2, vB, SB2); SB3 = _mm512_fmadd_pd(s3, vB, SB3);
    }
#define R31_ZSTORE(y, C0, C1, C2, C3, S0, S1, S2, S3) do {                     \
        __m512d T0 = _mm512_permute_pd(S0, 0x55);                              \
        __m512d T1 = _mm512_permute_pd(S1, 0x55);                              \
        __m512d T2 = _mm512_permute_pd(S2, 0x55);                              \
        __m512d T3 = _mm512_permute_pd(S3, 0x55);                              \
        _mm512_storeu_pd((y) + 0,  _mm512_fmadd_pd(T0, SG, C0));               \
        _mm512_storeu_pd((y) + 8,  _mm512_fmadd_pd(T1, SG, C1));               \
        _mm512_storeu_pd((y) + 16, _mm512_fmadd_pd(T2, SG, C2));               \
        _mm512_storeu_pd((y) + 24, _mm512_fmadd_pd(T3, SG, C3));               \
        __m512d h0 = _mm512_fnmadd_pd(T0, SG, C0);                             \
        __m512d h1 = _mm512_fnmadd_pd(T1, SG, C1);                             \
        __m512d h2 = _mm512_fnmadd_pd(T2, SG, C2);                             \
        __m512d h3 = _mm512_fnmadd_pd(T3, SG, C3);                             \
        _mm512_storeu_pd((y) + 32, _mm512_shuffle_f64x2(h3, h3, 0x1B));        \
        _mm512_storeu_pd((y) + 40, _mm512_shuffle_f64x2(h2, h2, 0x1B));        \
        _mm512_storeu_pd((y) + 48, _mm512_shuffle_f64x2(h1, h1, 0x1B));        \
        _mm512_mask_storeu_pd((y) + 56, 0x3F, _mm512_shuffle_f64x2(h0, h0, 0x1B)); \
    } while (0)
    R31_ZSTORE(yA, CA0, CA1, CA2, CA3, SA0, SA1, SA2, SA3);
    R31_ZSTORE(yB, CB0, CB1, CB2, CB3, SB0, SB1, SB2, SB3);
}

static void r31_zpass(const cplx *restrict src, cplx *restrict dst, size_t nrows,
                      ptrdiff_t sp, ptrdiff_t dp,
                      const double *restrict ctd, const double *restrict std)
{
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    size_t r = 0;
    for (; r + 2 <= nrows; r += 2)
        r31_zrow_pair(src + r * sp, dst + r * dp, sp, dp, ctd, std);
    for (; r < nrows; ++r) {
        const double *x = (const double *)(src + r * sp);
        double *y = (double *)(dst + r * dp);
        __attribute__((aligned(64))) double ub[32], vb[32];
        for (int j = 1; j <= 15; ++j) {
            __m128d a = _mm_loadu_pd(x + 2 * j);
            __m128d b = _mm_loadu_pd(x + 2 * (31 - j));
            _mm_store_pd(ub + 2 * j, _mm_add_pd(a, b));
            _mm_store_pd(vb + 2 * j, _mm_sub_pd(a, b));
        }
        __m512d x0 = _mm512_broadcast_f64x2(_mm_loadu_pd(x));
        __m512d C0 = x0, C1 = x0, C2 = x0, C3 = x0;
        __m512d S0 = _mm512_setzero_pd(), S1 = S0, S2 = S0, S3 = S0;
        for (int j = 1; j <= 15; ++j) {
            __m512d u = _mm512_broadcast_f64x2(_mm_load_pd(ub + 2 * j));
            __m512d v = _mm512_broadcast_f64x2(_mm_load_pd(vb + 2 * j));
            const double *cr = ctd + (size_t)(j - 1) * 32;
            const double *sr = std + (size_t)(j - 1) * 32;
            C0 = _mm512_fmadd_pd(_mm512_load_pd(cr + 0),  u, C0);
            C1 = _mm512_fmadd_pd(_mm512_load_pd(cr + 8),  u, C1);
            C2 = _mm512_fmadd_pd(_mm512_load_pd(cr + 16), u, C2);
            C3 = _mm512_fmadd_pd(_mm512_load_pd(cr + 24), u, C3);
            S0 = _mm512_fmadd_pd(_mm512_load_pd(sr + 0),  v, S0);
            S1 = _mm512_fmadd_pd(_mm512_load_pd(sr + 8),  v, S1);
            S2 = _mm512_fmadd_pd(_mm512_load_pd(sr + 16), v, S2);
            S3 = _mm512_fmadd_pd(_mm512_load_pd(sr + 24), v, S3);
        }
        __m512d T0 = _mm512_permute_pd(S0, 0x55);
        __m512d T1 = _mm512_permute_pd(S1, 0x55);
        __m512d T2 = _mm512_permute_pd(S2, 0x55);
        __m512d T3 = _mm512_permute_pd(S3, 0x55);
        _mm512_storeu_pd(y + 0,  _mm512_fmadd_pd(T0, SG, C0));
        _mm512_storeu_pd(y + 8,  _mm512_fmadd_pd(T1, SG, C1));
        _mm512_storeu_pd(y + 16, _mm512_fmadd_pd(T2, SG, C2));
        _mm512_storeu_pd(y + 24, _mm512_fmadd_pd(T3, SG, C3));
        __m512d h0 = _mm512_fnmadd_pd(T0, SG, C0);
        __m512d h1 = _mm512_fnmadd_pd(T1, SG, C1);
        __m512d h2 = _mm512_fnmadd_pd(T2, SG, C2);
        __m512d h3 = _mm512_fnmadd_pd(T3, SG, C3);
        _mm512_storeu_pd(y + 32, _mm512_shuffle_f64x2(h3, h3, 0x1B));
        _mm512_storeu_pd(y + 40, _mm512_shuffle_f64x2(h2, h2, 0x1B));
        _mm512_storeu_pd(y + 48, _mm512_shuffle_f64x2(h1, h1, 0x1B));
        _mm512_mask_storeu_pd(y + 56, 0x3F, _mm512_shuffle_f64x2(h0, h0, 0x1B));
    }
}
/* z-pass dispatcher: Rader-quad form by default, dense row-GEMM with -DR31_ZDENSE */
static void r31_zpass_main(const struct fft3d_plan *p, const cplx *src, cplx *dst,
                           size_t nrows)
{
#ifdef R31_ZDENSE
    r31_zpass(src, dst, nrows, 31, 31, p->ctd, p->std_);
#else
    size_t r = 0;
    for (; r + 4 <= nrows; r += 4)
        r31_zquad(src + r * 31, dst + r * 31, p->ke, p->ko);
    if (r < nrows)
        r31_zpass(src + r * 31, dst + r * 31, nrows - r, 31, 31, p->ctd, p->std_);
#endif
}

/* ---------------- generic odd-prime folded half-system engine (rp_*) ----------------
 * Round-3 class duty: any odd prime 3 <= p <= 127, p != 31 (which keeps the
 * tuned Winograd path above).  Same fold arithmetic as the r31 z-pass, in
 * column-chunk form with runtime loops: per chunk of 4 complex columns, all
 * p rows are loaded ONCE into stack u/v arrays (loads-all-then-stores => the
 * kernel is in-place safe for any dst == src), then k runs in QUADS of 4
 * (C,S) accumulator pairs sharing each u_j/v_j reload: per j per quad,
 * 2 stack loads + 8 broadcast constants feeding 8 FMAs.  ~2h^2 zmm FMA per
 * chunk per axis -- the settled folded-dense count on FMA hardware. */

#define RP_MAXH 63   /* (127-1)/2 */

static inline __attribute__((always_inline))
void rp_chunk(const double *sx, double *dx, const double *mapc,
              const ptrdiff_t rs, const int p, const int h,
              const double *ct, const double *st,
              int full, __mmask8 msk)
{
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    __m512d U[RP_MAXH + 1], V[RP_MAXH + 1];
#define RP_LD(px, off) (full ? _mm512_loadu_pd((px) + (off)) \
                             : _mm512_maskz_loadu_pd(msk, (px) + (off)))
#define RP_ST(off, v_) do { if (full) _mm512_storeu_pd(dx + (off), v_); \
                            else _mm512_mask_storeu_pd(dx + (off), msk, v_); } while (0)
    __m512d x0 = RP_LD(sx, 0);
    __m512d e0 = x0, e1 = _mm512_setzero_pd();
    for (int j = 1; j <= h; ++j) {
        __m512d a = RP_LD(sx, (ptrdiff_t)j * rs);
        __m512d b = RP_LD(sx, (ptrdiff_t)(p - j) * rs);
        U[j] = _mm512_add_pd(a, b);
        V[j] = _mm512_sub_pd(a, b);
        if (j & 1) e1 = _mm512_add_pd(e1, U[j]);
        else       e0 = _mm512_add_pd(e0, U[j]);
    }
    __m512d X0 = _mm512_add_pd(e0, e1);
    if (mapc) X0 = r31_map1(_mm512_add_pd(X0, RP_LD(mapc, 0)), R31_FMDIV);
    RP_ST(0, X0);
/* X_k = C - iS, X_{p-k} = C + iS on interleaved lanes: o = swap(S),
 * X_k = C + SG*o, X_{p-k} = C - SG*o (SG = +1,-1,...) */
#define RP_OUT(KK, C, S) do {                                                  \
        __m512d o_ = _mm512_permute_pd(S, 0x55);                               \
        __m512d xp_ = _mm512_fmadd_pd(o_, SG, C);                              \
        __m512d xm_ = _mm512_fnmadd_pd(o_, SG, C);                             \
        if (mapc) {                                                            \
            xp_ = _mm512_add_pd(xp_, RP_LD(mapc, (ptrdiff_t)(KK) * rs));       \
            xm_ = _mm512_add_pd(xm_, RP_LD(mapc, (ptrdiff_t)(p - (KK)) * rs)); \
            r31_map2(xp_, xm_, &xp_, &xm_, R31_FMDIV);                         \
        }                                                                      \
        RP_ST((ptrdiff_t)(KK) * rs, xp_);                                      \
        RP_ST((ptrdiff_t)(p - (KK)) * rs, xm_);                                \
    } while (0)
    int k = 1;
    for (; k + 3 <= h; k += 4) {
        const double *c0 = ct + (size_t)(k - 1) * h, *c1 = c0 + h,
                     *c2 = c1 + h, *c3 = c2 + h;
        const double *s0 = st + (size_t)(k - 1) * h, *s1 = s0 + h,
                     *s2 = s1 + h, *s3 = s2 + h;
        __m512d C0 = x0, C1 = x0, C2 = x0, C3 = x0;
        __m512d S0 = _mm512_setzero_pd(), S1 = S0, S2 = S0, S3 = S0;
        for (int j = 1; j <= h; ++j) {
            __m512d uj = U[j], vj = V[j];
            C0 = _mm512_fmadd_pd(_mm512_set1_pd(c0[j - 1]), uj, C0);
            S0 = _mm512_fmadd_pd(_mm512_set1_pd(s0[j - 1]), vj, S0);
            C1 = _mm512_fmadd_pd(_mm512_set1_pd(c1[j - 1]), uj, C1);
            S1 = _mm512_fmadd_pd(_mm512_set1_pd(s1[j - 1]), vj, S1);
            C2 = _mm512_fmadd_pd(_mm512_set1_pd(c2[j - 1]), uj, C2);
            S2 = _mm512_fmadd_pd(_mm512_set1_pd(s2[j - 1]), vj, S2);
            C3 = _mm512_fmadd_pd(_mm512_set1_pd(c3[j - 1]), uj, C3);
            S3 = _mm512_fmadd_pd(_mm512_set1_pd(s3[j - 1]), vj, S3);
        }
        RP_OUT(k, C0, S0);
        RP_OUT(k + 1, C1, S1);
        RP_OUT(k + 2, C2, S2);
        RP_OUT(k + 3, C3, S3);
    }
    for (; k <= h; ++k) {
        const double *c0 = ct + (size_t)(k - 1) * h;
        const double *s0 = st + (size_t)(k - 1) * h;
        __m512d C0 = x0, S0 = _mm512_setzero_pd();
        for (int j = 1; j <= h; ++j) {
            C0 = _mm512_fmadd_pd(_mm512_set1_pd(c0[j - 1]), U[j], C0);
            S0 = _mm512_fmadd_pd(_mm512_set1_pd(s0[j - 1]), V[j], S0);
        }
        RP_OUT(k, C0, S0);
    }
#undef RP_OUT
#undef RP_LD
#undef RP_ST
}

/* one p x ncols pass, rows `pitch` complex apart; in-place safe (dst may ==
 * src); c != NULL fuses the map at every store (c rows at the same pitch) */
static void rp_pass(const cplx *src, cplx *dst, const cplx *c,
                    const ptrdiff_t ncols, const ptrdiff_t pitch,
                    const int p, const int h,
                    const double *ct, const double *st)
{
    const ptrdiff_t rs = 2 * pitch, nd = 2 * ncols;
    const double *sx = (const double *)src;
    const double *cx = (const double *)c;
    double *dx = (double *)dst;
    ptrdiff_t d = 0;
    for (; d + 8 <= nd; d += 8)
        rp_chunk(sx + d, dx + d, cx ? cx + d : NULL, rs, p, h, ct, st,
                 1, (__mmask8)0xFF);
    if (d < nd)
        rp_chunk(sx + d, dx + d, cx ? cx + d : NULL, rs, p, h, ct, st,
                 0, (__mmask8)((1u << (nd - d)) - 1));
}

/* z-axis: up to 4 contiguous rows via 4x4-complex transposes into a stack
 * pencil array, the SAME chunk kernel at rs = 8, transpose back (r31_zquad
 * generalized to runtime p and a row count 1..4).  All loads precede all
 * stores => in-place safe. */
static void rp_zquad(const cplx *src, cplx *dst, int nrows,
                     const ptrdiff_t srd, const ptrdiff_t drd,
                     const int p, const int h,
                     const double *ct, const double *st)
{
    /* 4*ceil(p/4) <= 128 pencil rows of 8 doubles each */
    __attribute__((aligned(64))) double xt[128 * 8], yt[128 * 8];
    const int nt = (p + 3) / 4;            /* 4-complex tiles per row */
    const int tail = p & 3;                /* complex in the last tile (0 = full) */
    const __mmask8 tmsk = tail ? (__mmask8)((1u << (2 * tail)) - 1) : (__mmask8)0xFF;
    const double *x = (const double *)src;
    double *y = (double *)dst;
    for (int t = 0; t < nt; ++t) {
        const int ft = (t < nt - 1) || !tail;
        const __mmask8 mk = ft ? (__mmask8)0xFF : tmsk;
        __m512d r0 = _mm512_setzero_pd(), r1 = r0, r2 = r0, r3 = r0;
        r0 = _mm512_maskz_loadu_pd(mk, x + 8 * t);
        if (nrows > 1) r1 = _mm512_maskz_loadu_pd(mk, x + srd + 8 * t);
        if (nrows > 2) r2 = _mm512_maskz_loadu_pd(mk, x + 2 * srd + 8 * t);
        if (nrows > 3) r3 = _mm512_maskz_loadu_pd(mk, x + 3 * srd + 8 * t);
        __m512d o0, o1, o2, o3;
        r31_tp4(r0, r1, r2, r3, &o0, &o1, &o2, &o3);
        _mm512_store_pd(xt + (4 * t) * 8, o0);
        _mm512_store_pd(xt + (4 * t + 1) * 8, o1);
        _mm512_store_pd(xt + (4 * t + 2) * 8, o2);
        _mm512_store_pd(xt + (4 * t + 3) * 8, o3);
    }
    rp_chunk(xt, yt, NULL, 8, p, h, ct, st, 1, (__mmask8)0xFF);
    for (int t = 0; t < nt; ++t) {
        const int ft = (t < nt - 1) || !tail;
        const __mmask8 mk = ft ? (__mmask8)0xFF : tmsk;
        __m512d o0, o1, o2, o3;
        /* rows >= p of yt are never written by rp_chunk; their lanes land in
         * masked-out columns (shuffles only -- no arithmetic on them) */
        r31_tp4(_mm512_load_pd(yt + (4 * t) * 8),
                _mm512_load_pd(yt + (4 * t + 1) * 8),
                _mm512_load_pd(yt + (4 * t + 2) * 8),
                _mm512_load_pd(yt + (4 * t + 3) * 8),
                &o0, &o1, &o2, &o3);
        _mm512_mask_storeu_pd(y + 8 * t, mk, o0);
        if (nrows > 1) _mm512_mask_storeu_pd(y + drd + 8 * t, mk, o1);
        if (nrows > 2) _mm512_mask_storeu_pd(y + 2 * drd + 8 * t, mk, o2);
        if (nrows > 3) _mm512_mask_storeu_pd(y + 3 * drd + 8 * t, mk, o3);
    }
}

/* forward 3D volume, generic prime: z rows (quads + tail), x in place
 * (inner = p^2), y in place per plane (inner = p) */
static void rp_volume(const fft3d_plan *pl, const cplx *src, cplx *dst)
{
    const int p = pl->L, h = pl->h;
    const size_t LL = (size_t)p * p;
    size_t r = 0;
    for (; r + 4 <= LL; r += 4)
        rp_zquad(src + r * p, dst + r * p, 4, 2 * p, 2 * p, p, h, pl->gct, pl->gst);
    if (r < LL)
        rp_zquad(src + r * p, dst + r * p, (int)(LL - r), 2 * p, 2 * p,
                 p, h, pl->gct, pl->gst);
    rp_pass(dst, dst, NULL, (ptrdiff_t)LL, (ptrdiff_t)LL, p, h, pl->gct, pl->gst);
    for (int xpl = 0; xpl < p; ++xpl)
        rp_pass(dst + (size_t)xpl * LL, dst + (size_t)xpl * LL, NULL,
                p, p, p, h, pl->gct, pl->gst);
}

#endif /* __AVX512F__ */

/* ---------------- dense fallback / reference (the round-0 stub engine) ---------------- */

static void ref_contract(const cplx *w, int L, const cplx *in, cplx *out, int inner)
{
    for (int k = 0; k < L; ++k)
        for (int c = 0; c < inner; ++c) {
            cplx acc = 0.0;
            for (int j = 0; j < L; ++j)
                acc += w[(size_t)k * L + j] * in[(size_t)j * inner + c];
            out[(size_t)k * inner + c] = acc;
        }
}

/* full reference volume: src -> dst (src untouched; uses p->tmp) */
static void ref_volume(fft3d_plan *p, const cplx *src, cplx *dst)
{
    const int L = p->L;
    const size_t LL = (size_t)L * L;
    ref_contract(p->w, L, src, dst, (int)LL);
    for (int x = 0; x < L; ++x)
        ref_contract(p->w, L, dst + (size_t)x * LL, p->tmp + (size_t)x * LL, L);
    for (size_t row = 0; row < LL; ++row)
        ref_contract(p->w, L, p->tmp + row * L, dst + row * L, 1);
}

/* ---------------- one volume, forward 3D (fast path) ----------------
 * z rows: src -> dst (row-local, out-of-place or in-place safe); x and y IN
 * PLACE on dst (the chunk kernel loads every row before storing any). */
static void fast_volume(fft3d_plan *p, const cplx *src, cplx *dst)
{
#ifdef __AVX512F__
    if (p->L != 31) { rp_volume(p, src, dst); return; }
    const size_t LL = 31 * 31;
    r31_zpass_main(p, src, dst, LL);
    r31_pass_x(dst, dst, p->ke, p->ko);
    for (int x = 0; x < 31; ++x)
        r31_pass_y(dst + (size_t)x * LL, dst + (size_t)x * LL, p->ke, p->ko);
#else
    (void)p; (void)src; (void)dst;
#endif
}

void fft3d_execute(fft3d_plan *p, const cplx *in, cplx *out)
{
    const size_t vol = (size_t)p->L * p->L * p->L;
    for (int b = 0; b < p->batch; ++b) {
        if (p->fast)
            fast_volume(p, in + (size_t)b * vol, out + (size_t)b * vol);
        else
            ref_volume(p, in + (size_t)b * vol, out + (size_t)b * vol);
    }
}

/* ---------------- fused map chain (shape from gen_dense_prime / ice s6) ---------------- */

/* z and o may alias (in-place map): elementwise, loads precede the store per
 * point -- deliberately NOT restrict-qualified */
static void map_volume(const cplx *z, const cplx *restrict c,
                       cplx *o, size_t npts)
{
    const double *zp = (const double *)z;
    const double *cp = (const double *)c;
    double *op = (double *)o;
    size_t i = 0;
#ifdef __AVX512F__
    const __m512d ONE  = _mm512_set1_pd(1.0);
    const __m512d TH   = _mm512_set1_pd(1.5);
    const __m512d HALF = _mm512_set1_pd(0.5);
    const __m512d TINY = _mm512_set1_pd(1e-300);
    for (; i + 8 <= npts; i += 8) {
        __m512d w0 = _mm512_add_pd(_mm512_loadu_pd(zp + 2 * i),
                                   _mm512_loadu_pd(cp + 2 * i));
        __m512d w1 = _mm512_add_pd(_mm512_loadu_pd(zp + 2 * i + 8),
                                   _mm512_loadu_pd(cp + 2 * i + 8));
        __m512d p0 = _mm512_mul_pd(w0, w0), p1 = _mm512_mul_pd(w1, w1);
        __m512d m2 = _mm512_add_pd(_mm512_unpacklo_pd(p0, p1),
                                   _mm512_unpackhi_pd(p0, p1));
        __m512d m2c = _mm512_max_pd(m2, TINY);
        __m512d r = _mm512_rsqrt14_pd(m2c);
        __m512d hm = _mm512_mul_pd(m2c, HALF);
        r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hm, r), r, TH));
        r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hm, r), r, TH));
        __m512d d = _mm512_fmadd_pd(m2c, r, ONE);           /* 1 + |w| */
        __m512d rec = _mm512_div_pd(ONE, d);                /* the one divide */
        _mm512_storeu_pd(op + 2 * i,     _mm512_mul_pd(w0, _mm512_unpacklo_pd(rec, rec)));
        _mm512_storeu_pd(op + 2 * i + 8, _mm512_mul_pd(w1, _mm512_unpackhi_pd(rec, rec)));
    }
#endif
    for (; i < npts; ++i) {
        double re = zp[2 * i] + cp[2 * i];
        double im = zp[2 * i + 1] + cp[2 * i + 1];
        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
        op[2 * i] = re * sc;
        op[2 * i + 1] = im * sc;
    }
}

#ifdef __AVX512F__
/* One volume's whole fused chain on the padded private state (raw engine; the
 * caller gates on p->fast).  Step 0's z pass reads flat x0 straight into the
 * arena; steps run z (8 uniform quads/plane -- the 32nd "row" is the zeroed
 * pad row, DFT(0)=0), x across planes at the anti-alias pitch, then per plane
 * y + map (c from the padded mirror, filled once per volume).  The last step's
 * map stays in the arena and the rows are copied out flat. */
static void r31_chain_volume(fft3d_plan *p, const cplx *x0v, const cplx *cv,
                             cplx *outv, int m)
{
    const size_t LL = 31 * 31;
    cplx *st = p->st, *cp = p->cpad;
    for (int x = 0; x < 31; ++x)
        for (int y = 0; y < 31; ++y)
            memcpy(cp + (size_t)x * R31_PP + (size_t)y * R31_ZP,
                   cv + (size_t)x * LL + (size_t)y * 31, 31 * sizeof(cplx));
#ifdef R31_R3CHAIN
    /* gen_r3 pass order (control arm): z sweep / x sweep / y+map sweep --
     * step s+1's z re-reads the whole state from L2 after the map sweep. */
    for (int s = 0; s < m; ++s) {
        for (int x = 0; x < 31; ++x) {
            cplx *pl = st + (size_t)x * R31_PP;
            if (s == 0) {
                const cplx *sp = x0v + (size_t)x * LL;
                for (int q = 0; q < 7; ++q)
                    r31_zquad_fp(sp + (size_t)q * 4 * 31,
                                 pl + (size_t)q * 4 * R31_ZP, p->ke, p->ko);
                r31_zpass(sp + 28 * 31, pl + 28 * R31_ZP, 3, 31, R31_ZP,
                          p->ctd, p->std_);
            } else {
                for (int q = 0; q < 8; ++q)
                    r31_zquad_pp(pl + (size_t)q * 4 * R31_ZP,
                                 pl + (size_t)q * 4 * R31_ZP, p->ke, p->ko);
            }
        }
        r31_pass_xp(st, p->ke, p->ko);
        for (int x = 0; x < 31; ++x) {
            cplx *pl = st + (size_t)x * R31_PP;
            r31_pass_yp(pl, p->ke, p->ko);
            map_volume(pl, cp + (size_t)x * R31_PP, pl, 31 * R31_ZP);
            if (s == m - 1) {
                cplx *op = outv + (size_t)x * LL;
                for (int y = 0; y < 31; ++y)
                    memcpy(op + (size_t)y * 31, pl + (size_t)y * R31_ZP,
                           31 * sizeof(cplx));
            }
        }
    }
#else
    /* gen_r4 PLANE CUSTODY (gen_layout r3's window idea, gen_bluestein r4's
     * confirmation): z contracts within a plane, so step s+1's z-pass runs on
     * each plane RIGHT AFTER that plane's map, while it is still L1/L2-hot --
     * one full-state L2 read per step deleted vs the r3 order.  Identical
     * arithmetic, identical per-pass order within each plane => bit-identical.
     * -DR31_ZMAPF additionally fuses the map into the z-quads' transpose-in
     * loads (deletes the separate map sweep; vdivpd form => still
     * bit-identical). */
    for (int x = 0; x < 31; ++x) {          /* prologue: z_0 from flat x0 */
        cplx *pl = st + (size_t)x * R31_PP;
        const cplx *sp = x0v + (size_t)x * LL;
        for (int q = 0; q < 7; ++q)
            r31_zquad_fp(sp + (size_t)q * 4 * 31,
                         pl + (size_t)q * 4 * R31_ZP, p->ke, p->ko);
        r31_zpass(sp + 28 * 31, pl + 28 * R31_ZP, 3, 31, R31_ZP,
                  p->ctd, p->std_);
    }
    for (int s = 0; s < m; ++s) {
        r31_pass_xp(st, p->ke, p->ko);
        for (int x = 0; x < 31; ++x) {
            cplx *pl = st + (size_t)x * R31_PP;
            const cplx *cpl = cp + (size_t)x * R31_PP;
            r31_pass_yp(pl, p->ke, p->ko);
            if (s == m - 1) {
                map_volume(pl, cpl, pl, 31 * R31_ZP);
                cplx *op = outv + (size_t)x * LL;
                for (int y = 0; y < 31; ++y)
                    memcpy(op + (size_t)y * 31, pl + (size_t)y * R31_ZP,
                           31 * sizeof(cplx));
            } else {
#ifdef R31_ZMAPF
                for (int q = 0; q < 8; ++q)
                    r31_zquad_mp(pl + (size_t)q * 4 * R31_ZP,
                                 pl + (size_t)q * 4 * R31_ZP,
                                 cpl + (size_t)q * 4 * R31_ZP, p->ke, p->ko);
#else
                map_volume(pl, cpl, pl, 31 * R31_ZP);
                for (int q = 0; q < 8; ++q)
                    r31_zquad_pp(pl + (size_t)q * 4 * R31_ZP,
                                 pl + (size_t)q * 4 * R31_ZP, p->ke, p->ko);
#endif
            }
        }
    }
#endif
}

/* generic-prime fused chain: fully in place on the out volume (the r1 form;
 * every rp pass and the map are in-place safe), map per plane right after its
 * y pass while the plane is cache-hot.  -DRP_YMAPFUSE instead fuses the map
 * into the y-pass stores (raceable; the r31 engine lost this one to register
 * pressure -- the generic kernel's combine is leaner, so it stays a knob). */
/* z-pass over one plane's p pencils, in place: quads + a per-plane tail */
static void rp_zplane(cplx *plane, const int p, const int h,
                      const double *ct, const double *st)
{
    int r = 0;
    for (; r + 4 <= p; r += 4)
        rp_zquad(plane + (size_t)r * p, plane + (size_t)r * p, 4,
                 2 * p, 2 * p, p, h, ct, st);
    if (r < p)
        rp_zquad(plane + (size_t)r * p, plane + (size_t)r * p, p - r,
                 2 * p, 2 * p, p, h, ct, st);
}

static void rp_chain_volume(fft3d_plan *pl, const cplx *x0v, const cplx *cv,
                            cplx *stv, int m)
{
    const int p = pl->L, h = pl->h;
    const size_t LL = (size_t)p * p, vol = LL * p;
    memcpy(stv, x0v, vol * sizeof(cplx));
#ifdef RP_R3CHAIN
    /* gen_r3 order (control arm): global z sweep per step */
    for (int s = 0; s < m; ++s) {
        size_t r = 0;
        for (; r + 4 <= LL; r += 4)
            rp_zquad(stv + r * p, stv + r * p, 4, 2 * p, 2 * p,
                     p, h, pl->gct, pl->gst);
        if (r < LL)
            rp_zquad(stv + r * p, stv + r * p, (int)(LL - r), 2 * p, 2 * p,
                     p, h, pl->gct, pl->gst);
        rp_pass(stv, stv, NULL, (ptrdiff_t)LL, (ptrdiff_t)LL,
                p, h, pl->gct, pl->gst);
        for (int xpl = 0; xpl < p; ++xpl) {
            cplx *plane = stv + (size_t)xpl * LL;
            const cplx *cpl = cv + (size_t)xpl * LL;
#ifdef RP_YMAPFUSE
            rp_pass(plane, plane, cpl, p, p, p, h, pl->gct, pl->gst);
#else
            rp_pass(plane, plane, NULL, p, p, p, h, pl->gct, pl->gst);
            map_volume(plane, cpl, plane, LL);
#endif
        }
    }
#else
    /* gen_r4 PLANE CUSTODY (same move as the r31 chain): step s+1's z runs
     * per plane right after that plane's map, while the plane is cache-hot.
     * At DRAM-resident sizes (p >= ~50, volume > L2) this deletes a whole
     * volume read per step.  Per-pencil arithmetic identical; only the quad
     * GROUPING changes at p % 4 != 0 (pencils are lane-independent =>
     * bit-identical outputs). */
    for (int xpl = 0; xpl < p; ++xpl)       /* prologue: z_0 per plane */
        rp_zplane(stv + (size_t)xpl * LL, p, h, pl->gct, pl->gst);
    for (int s = 0; s < m; ++s) {
        rp_pass(stv, stv, NULL, (ptrdiff_t)LL, (ptrdiff_t)LL,
                p, h, pl->gct, pl->gst);
        for (int xpl = 0; xpl < p; ++xpl) {
            cplx *plane = stv + (size_t)xpl * LL;
            const cplx *cpl = cv + (size_t)xpl * LL;
#ifdef RP_YMAPFUSE
            rp_pass(plane, plane, cpl, p, p, p, h, pl->gct, pl->gst);
#else
            rp_pass(plane, plane, NULL, p, p, p, h, pl->gct, pl->gst);
            map_volume(plane, cpl, plane, LL);
#endif
            if (s < m - 1)
                rp_zplane(plane, p, h, pl->gct, pl->gst);
        }
    }
#endif
}
#endif /* __AVX512F__ */

void fft3d_chain(fft3d_plan *p, const cplx *x0, const cplx *c,
                 cplx *final_out, int m)
{
    const size_t LL = (size_t)p->L * p->L, vol = LL * p->L;
    for (int b = 0; b < p->batch; ++b) {
        cplx *stv = final_out + (size_t)b * vol;    /* state lives in the out volume */
        const cplx *cv = c + (size_t)b * vol;
        if (p->fast && m > 0 && p->L != 31) {
#ifdef __AVX512F__
            rp_chain_volume(p, x0 + (size_t)b * vol, cv, stv, m);
#endif
        } else if (p->fast && m > 0) {
#ifdef __AVX512F__
#if defined(R31_FLATCHAIN) || defined(R31_FUSEMAP)
            /* gen_r1 shipped form: all passes in place on the FLAT state in the
             * out volume (953 KB working set).  Kept for A/B (-DR31_FLATCHAIN);
             * the natural 961-complex plane pitch makes every x-pass access
             * line-split AND 4K-aliases each chunk's stores against the next
             * chunk's loads at row distance 4 (961*16*4 == 64 mod 4096). */
            memcpy(stv, x0 + (size_t)b * vol, vol * sizeof(cplx));
            for (int s = 0; s < m; ++s) {
                r31_zpass_main(p, stv, stv, LL);
                r31_pass_x(stv, stv, p->ke, p->ko);
#ifdef R31_FUSEMAP
                for (int x = 0; x < 31; ++x)
                    r31_pass_ym(stv + (size_t)x * LL, p->t1 + (size_t)x * LL,
                                cv + (size_t)x * LL, p->ke, p->ko);
                memcpy(stv, p->t1, vol * sizeof(cplx));
#else
                for (int x = 0; x < 31; ++x)
                    r31_pass_y(stv + (size_t)x * LL, stv + (size_t)x * LL,
                               p->ke, p->ko);
                map_volume(stv, cv, stv, vol);
#endif
            }
#else
            /* gen_r2 form: fully padded/aligned private state -- see
             * r31_chain_volume.  Working set st + cpad = 1.14 MB < L2. */
            r31_chain_volume(p, x0 + (size_t)b * vol, cv, stv, m);
#endif
#endif
        } else {
            memcpy(stv, x0 + (size_t)b * vol, vol * sizeof(cplx));
            for (int s = 0; s < m; ++s) {
                ref_volume(p, stv, p->t1);
                map_volume(p->t1, cv, stv, vol);
            }
        }
    }
}

/* ---------------- plan lifecycle ---------------- */

static void *xalloc(size_t bytes)
{
    return aligned_alloc(64, (bytes + 63) & ~(size_t)63);
}

/* deterministic pseudo-random volume; compare fast engine vs dense reference
 * for both execute AND one padded chain step (a pad/stride bug must fall back,
 * not ship).  Transcription bugs in the Rader tables would show at ~1e0; the
 * correct engines differ by rounding only (~1e-15). */
static int self_check(fft3d_plan *p)
{
#ifdef __AVX512F__
    const size_t vol = (size_t)p->L * p->L * p->L;
    cplx *a = xalloc(vol * sizeof(cplx));
    cplx *cc = xalloc(vol * sizeof(cplx));
    cplx *rf = xalloc(vol * sizeof(cplx));
    cplx *ff = xalloc(vol * sizeof(cplx));
    if (!a || !cc || !rf || !ff) { free(a); free(cc); free(rf); free(ff); return 0; }
    unsigned long long st = 0x9e3779b97f4a7c15ull;
    double *ad = (double *)a, *cd = (double *)cc;
    for (size_t i = 0; i < 2 * vol; ++i) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        ad[i] = (double)(long long)(st % 2000001ull) / 1000000.0 - 1.0;
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        cd[i] = (double)(long long)(st % 2000001ull) / 1000000.0 - 1.0;
    }
    ref_volume(p, a, rf);
    fast_volume(p, a, ff);
    long double num = 0, den = 0;
    for (size_t i = 0; i < vol; ++i) {
        cplx dd = ff[i] - rf[i];
        num += creal(dd) * creal(dd) + cimag(dd) * cimag(dd);
        den += creal(rf[i]) * creal(rf[i]) + cimag(rf[i]) * cimag(rf[i]);
    }
    int ok = den > 0 && sqrtl(num / den) < 1e-13L;
    if (ok) {
        for (size_t i = 0; i < vol; ++i) {      /* scalar-map the reference */
            double re = creal(rf[i]) + creal(cc[i]);
            double im = cimag(rf[i]) + cimag(cc[i]);
            double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
            rf[i] = re * sc + I * im * sc;
        }
        int do_chain_leg = 1;
#if defined(R31_FLATCHAIN) || defined(R31_FUSEMAP)
        if (p->L == 31) do_chain_leg = 0;
#endif
        if (do_chain_leg) {
            if (p->L == 31) r31_chain_volume(p, a, cc, ff, 1);
            else            rp_chain_volume(p, a, cc, ff, 1);
            num = den = 0;
            for (size_t i = 0; i < vol; ++i) {
                cplx dd = ff[i] - rf[i];
                num += creal(dd) * creal(dd) + cimag(dd) * cimag(dd);
                den += creal(rf[i]) * creal(rf[i]) + cimag(rf[i]) * cimag(rf[i]);
            }
            ok = den > 0 && sqrtl(num / den) < 1e-13L;
        }
    }
    free(a); free(cc); free(rf); free(ff);
    return ok;
#else
    (void)p;
    return 0;
#endif
}

/* st + cpad in ONE 2MiB huge-page arena, c mirror at page phase +2048 B
 * (gen_layout gl_map_huge recipe; kills the map's c-load / y-store 4K alias
 * two same-phase aligned_allocs produce).  Heap fallback keeps r2 behavior. */
static void r31_arena_init(fft3d_plan *p)
{
    const size_t one = (size_t)31 * R31_PP * sizeof(cplx);
    const size_t stb = (one + 4095) & ~(size_t)4095;
    const size_t HP = (size_t)2 << 20;
    const size_t len = (stb + 2048 + one + HP - 1) & ~(HP - 1);
    void *raw = mmap(0, len + HP, PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (raw != MAP_FAILED) {
        char *al = (char *)(((uintptr_t)raw + HP - 1) & ~(uintptr_t)(HP - 1));
        size_t head = (size_t)(al - (char *)raw);
        if (head) munmap(raw, head);
        size_t tl = (size_t)(((char *)raw + len + HP) - (al + len));
        if (tl) munmap(al + len, tl);
        madvise(al, len, MADV_HUGEPAGE);
        memset(al, 0, len);            /* prefault now: faults belong in create() */
        p->arena = al;
        p->alen = len;
        p->st = (cplx *)al;
        p->cpad = (cplx *)(al + stb + 2048);
        return;
    }
    p->st = xalloc(one);
    p->cpad = xalloc(one);
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L)) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->h = (L - 1) / 2;

    const size_t vol = (size_t)L * L * L;
    p->t1 = xalloc(vol * sizeof(cplx));
    p->w = xalloc((size_t)L * L * sizeof(cplx));
    p->tmp = xalloc(vol * sizeof(cplx));
    int okm = p->t1 && p->w && p->tmp;
    if (L == 31) {
        p->ke = xalloc(20 * sizeof(double));
        p->ko = xalloc(20 * sizeof(double));
        p->ctd = r31_trig_dup(0);
        p->std_ = r31_trig_dup(1);
        r31_arena_init(p);
        okm = okm && p->ke && p->ko && p->ctd && p->std_ && p->st && p->cpad;
        if (okm) {
            /* pad slots must be zero (not garbage/denormals) and then stay zero */
            memset(p->st, 0, (size_t)31 * R31_PP * sizeof(cplx));
            memset(p->cpad, 0, (size_t)31 * R31_PP * sizeof(cplx));
            r31_build_kernels(p->ke, p->ko);
        }
    } else {
        p->gct = rp_trig(L, p->h, 0);
        p->gst = rp_trig(L, p->h, 1);
        okm = okm && p->gct && p->gst;
    }
    if (!okm) {
        fft3d_destroy(p);
        return NULL;
    }
    for (int k = 0; k < L; ++k)
        for (int j = 0; j < L; ++j) {
            long double th = -2.0L * PIL * (long double)((k * j) % L) / (long double)L;
            p->w[(size_t)k * L + j] = (double)cosl(th) + I * (double)sinl(th);
        }
    p->fast = self_check(p);
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->ke); free(p->ko); free(p->ctd); free(p->std_);
    free(p->gct); free(p->gst);
    free(p->t1); free(p->w); free(p->tmp);
    if (p->arena) munmap(p->arena, p->alen);
    else { free(p->st); free(p->cpad); }
    free(p);
}
