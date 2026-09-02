/* d1_pow2: radix-4/2 Stockham autosort DIF for L = 16/32/64/128/256.
 *
 * Layout: interleaved complex double, 4 complex per zmm. All kernels are
 * register-blocked with compile-time trip counts: L=16/32/64 run entirely in
 * registers (one 4x4 complex transpose after the first stage is the only
 * shuffle work); L=128/256 run as TWO fused passes (stages 1+2, stages 3+4)
 * through one L1 scratch buffer instead of one pass per stage. Twiddles are
 * precomputed at plan time in exactly the form consumed: pair-duplicated
 * vectors for the s=1 first stage, compact scalars broadcast at run time
 * (embedded-broadcast operands) for the later stages.
 *
 * fft1d_chain is owned: the normalizing map z/(1+|z|) is fused into the final
 * FFT stage (no separate map pass, no round trip), computed with
 * rsqrt14/rcp14 + 2 Newton steps each (~1e-15 relative, inside every gate)
 * so nothing serializes on the divider port; and the chain runs batch-outer
 * so one lane's entire m-step evolution stays L1-resident.
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../fft1d_api.h"

#if defined(__AVX512F__)
#include <immintrin.h>
#define D1P_AVX512 1
#endif

typedef double _Complex cd;

struct fft1d_plan {
    int L, batch;
    double *tw1;              /* first stage (s=1): 48 doubles per 4-wide p group */
    double *twA, *twB;        /* later radix-4 stages: 6 doubles per p */
    double *s1, *s2, *stt;    /* scratch + chain state, 2L doubles each */
    void (*kern)(const struct fft1d_plan *, const double *, double *);
    void (*kmap)(const struct fft1d_plan *, const double *, double *, const double *);
};

const char *fft1d_name(void) { return "d1_pow2"; }
const char *fft1d_description(void)
{
    return "radix-4/2 Stockham DIF, register-resident zmm kernels (2-pass at 128/256), "
           "NR-map fused chain (batch-outer)";
}
int fft1d_supports(int L) { return L == 16 || L == 32 || L == 64 || L == 128 || L == 256; }

#ifdef D1P_AVX512
/* ---------------- AVX-512 kernels ---------------- */

static inline __m512d vswap(__m512d v) { return _mm512_shuffle_pd(v, v, 0x55); }

/* y * w with vector twiddles (wr,wi already lane-placed, pair-duplicated) */
static inline __m512d cmulv(__m512d y, __m512d wr, __m512d wi)
{
    return _mm512_fmaddsub_pd(wr, y, _mm512_mul_pd(wi, vswap(y)));
}

/* z -> (z+c)/(1+|z+c|), 4 complex per vector. sqrt and the reciprocal are done
 * with rsqrt14/rcp14 plus two Newton steps each: ~21 FMA-class ops that
 * pipeline with the FFT instead of vsqrtpd+vdivpd serializing on the divider
 * port (measured 1.7-2.1x on whole chain steps, divergence unchanged). */
static inline __m512d mapv(__m512d z, __m512d cv)
{
    const __m512d one = _mm512_set1_pd(1.0), half = _mm512_set1_pd(0.5),
                  th = _mm512_set1_pd(1.5);
    __m512d w = _mm512_add_pd(z, cv);
    __m512d t = _mm512_mul_pd(w, w);
    __m512d h = _mm512_add_pd(t, vswap(t));            /* |w|^2, pair-duplicated */
#ifdef D1P_EXACT_MAP
    (void)half; (void)th;
    return _mm512_div_pd(w, _mm512_add_pd(_mm512_sqrt_pd(h), one));
#else
    h = _mm512_max_pd(h, _mm512_set1_pd(1e-300));      /* rsqrt(0) would be inf */
    __m512d r = _mm512_rsqrt14_pd(h);
    __m512d hh = _mm512_mul_pd(h, half);
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hh, r), r, th));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hh, r), r, th));
    __m512d den = _mm512_fmadd_pd(h, r, one);          /* 1 + sqrt(h) */
    __m512d rc = _mm512_rcp14_pd(den);
    rc = _mm512_fmadd_pd(rc, _mm512_fnmadd_pd(den, rc, one), rc);
    rc = _mm512_fmadd_pd(rc, _mm512_fnmadd_pd(den, rc, one), rc);
    return _mm512_mul_pd(w, rc);
#endif
}

/* radix-4 butterfly on 4 vectors, results pre-twiddle in y0..y3 */
#define BFLY4(a, b, c, d, y0, y1, y2, y3, one)                    \
    do {                                                          \
        __m512d apc_ = _mm512_add_pd(a, c), amc_ = _mm512_sub_pd(a, c); \
        __m512d bpd_ = _mm512_add_pd(b, d), bmd_ = _mm512_sub_pd(b, d); \
        __m512d sw_ = vswap(bmd_);                                \
        y0 = _mm512_add_pd(apc_, bpd_);                           \
        y1 = _mm512_fmsubadd_pd(one, amc_, sw_);                  \
        y2 = _mm512_sub_pd(apc_, bpd_);                           \
        y3 = _mm512_fmaddsub_pd(one, amc_, sw_);                  \
    } while (0)

/* first-stage group: twiddle by tw1 group table, then 4x4 complex transpose */
#define TWID_TRANSPOSE(t, y0, y1, y2, y3, z0, z1, z2, z3)         \
    do {                                                          \
        __m512d w1_ = cmulv(y1, _mm512_loadu_pd(t), _mm512_loadu_pd((t) + 8));   \
        __m512d w2_ = cmulv(y2, _mm512_loadu_pd((t) + 16), _mm512_loadu_pd((t) + 24)); \
        __m512d w3_ = cmulv(y3, _mm512_loadu_pd((t) + 32), _mm512_loadu_pd((t) + 40)); \
        __m512d t0_ = _mm512_shuffle_f64x2(y0, w1_, 0x44);        \
        __m512d t1_ = _mm512_shuffle_f64x2(w2_, w3_, 0x44);       \
        __m512d t2_ = _mm512_shuffle_f64x2(y0, w1_, 0xEE);        \
        __m512d t3_ = _mm512_shuffle_f64x2(w2_, w3_, 0xEE);       \
        z0 = _mm512_shuffle_f64x2(t0_, t1_, 0x88);                \
        z1 = _mm512_shuffle_f64x2(t0_, t1_, 0xDD);                \
        z2 = _mm512_shuffle_f64x2(t2_, t3_, 0x88);                \
        z3 = _mm512_shuffle_f64x2(t2_, t3_, 0xDD);                \
    } while (0)

/* twiddled radix-4 output set from compact table entry p */
#define CMUL3(tw, pp, y1, y2, y3)                                 \
    do {                                                          \
        y1 = cmulv(y1, _mm512_set1_pd((tw)[6 * (pp) + 0]), _mm512_set1_pd((tw)[6 * (pp) + 1])); \
        y2 = cmulv(y2, _mm512_set1_pd((tw)[6 * (pp) + 2]), _mm512_set1_pd((tw)[6 * (pp) + 3])); \
        y3 = cmulv(y3, _mm512_set1_pd((tw)[6 * (pp) + 4]), _mm512_set1_pd((tw)[6 * (pp) + 5])); \
    } while (0)

/* ---- L=16: 4 zmm, two stages, fully register-resident ---- */
static inline void fft16_regs(const fft1d_plan *p, const double *restrict x,
                              __m512d v[4])
{
    const __m512d one = _mm512_set1_pd(1.0);
    __m512d y0, y1, y2, y3, u0, u1, u2, u3;
    BFLY4(_mm512_loadu_pd(x), _mm512_loadu_pd(x + 8),
          _mm512_loadu_pd(x + 16), _mm512_loadu_pd(x + 24), y0, y1, y2, y3, one);
    TWID_TRANSPOSE(p->tw1, y0, y1, y2, y3, u0, u1, u2, u3);
    BFLY4(u0, u1, u2, u3, v[0], v[1], v[2], v[3], one);
}

static void k16r(const fft1d_plan *p, const double *restrict x, double *restrict y)
{
    __m512d v[4];
    fft16_regs(p, x, v);
    for (int j = 0; j < 4; ++j) _mm512_storeu_pd(y + 8 * j, v[j]);
}

static void k16rm(const fft1d_plan *p, const double *restrict x, double *restrict y,
                  const double *restrict cf)
{
    __m512d v[4];
    fft16_regs(p, x, v);
    for (int j = 0; j < 4; ++j)
        _mm512_storeu_pd(y + 8 * j, mapv(v[j], _mm512_loadu_pd(cf + 8 * j)));
}

/* ---- L=32: 8 zmm, three stages, fully register-resident ---- */
static inline void fft32_regs(const fft1d_plan *p, const double *restrict x,
                              __m512d v[8])
{
    const __m512d one = _mm512_set1_pd(1.0);
    __m512d u[8];
    for (int j = 0; j < 8; ++j) v[j] = _mm512_loadu_pd(x + 8 * j);
    /* stage 1: n=32, s=1, two p-groups */
    {
        __m512d y0, y1, y2, y3;
        BFLY4(v[0], v[2], v[4], v[6], y0, y1, y2, y3, one);
        TWID_TRANSPOSE(p->tw1, y0, y1, y2, y3, u[0], u[1], u[2], u[3]);
        BFLY4(v[1], v[3], v[5], v[7], y0, y1, y2, y3, one);
        TWID_TRANSPOSE(p->tw1 + 48, y0, y1, y2, y3, u[4], u[5], u[6], u[7]);
    }
    /* stage 2: n=8, s=4, m=2 */
    {
        __m512d y0, y1, y2, y3;
        BFLY4(u[0], u[2], u[4], u[6], y0, y1, y2, y3, one);
        v[0] = y0; v[1] = y1; v[2] = y2; v[3] = y3;
        BFLY4(u[1], u[3], u[5], u[7], y0, y1, y2, y3, one);
        CMUL3(p->twA, 1, y1, y2, y3);
        v[4] = y0; v[5] = y1; v[6] = y2; v[7] = y3;
    }
    /* stage 3: n=2, s=16 */
    for (int j = 0; j < 4; ++j) {
        __m512d a = v[j], b = v[j + 4];
        v[j] = _mm512_add_pd(a, b);
        v[j + 4] = _mm512_sub_pd(a, b);
    }
}

static void k32r(const fft1d_plan *p, const double *restrict x, double *restrict y)
{
    __m512d v[8];
    fft32_regs(p, x, v);
    for (int j = 0; j < 8; ++j) _mm512_storeu_pd(y + 8 * j, v[j]);
}

static void k32rm(const fft1d_plan *p, const double *restrict x, double *restrict y,
                  const double *restrict cf)
{
    __m512d v[8];
    fft32_regs(p, x, v);
    for (int j = 0; j < 8; ++j)
        _mm512_storeu_pd(y + 8 * j, mapv(v[j], _mm512_loadu_pd(cf + 8 * j)));
}

/* ---- L=64: 16 zmm, three stages, fully register-resident ---- */
static inline void fft64_regs(const fft1d_plan *p, const double *restrict x,
                              __m512d v[16])
{
    const __m512d one = _mm512_set1_pd(1.0);
    __m512d u[16];
    for (int j = 0; j < 16; ++j) v[j] = _mm512_loadu_pd(x + 8 * j);
    /* stage 1: n=64, s=1, four p-groups */
    for (int g = 0; g < 4; ++g) {
        __m512d y0, y1, y2, y3;
        BFLY4(v[g], v[g + 4], v[g + 8], v[g + 12], y0, y1, y2, y3, one);
        TWID_TRANSPOSE(p->tw1 + 48 * g, y0, y1, y2, y3,
                       u[4 * g], u[4 * g + 1], u[4 * g + 2], u[4 * g + 3]);
    }
    /* stage 2: n=16, s=4, m=4 */
    for (int pp = 0; pp < 4; ++pp) {
        __m512d y0, y1, y2, y3;
        BFLY4(u[pp], u[pp + 4], u[pp + 8], u[pp + 12], y0, y1, y2, y3, one);
        if (pp) CMUL3(p->twA, pp, y1, y2, y3);
        v[4 * pp] = y0; v[4 * pp + 1] = y1; v[4 * pp + 2] = y2; v[4 * pp + 3] = y3;
    }
    /* stage 3: n=4, s=16, m=1 */
    for (int j = 0; j < 4; ++j) {
        __m512d y0, y1, y2, y3;
        BFLY4(v[j], v[j + 4], v[j + 8], v[j + 12], y0, y1, y2, y3, one);
        v[j] = y0; v[j + 4] = y1; v[j + 8] = y2; v[j + 12] = y3;
    }
}

static void k64r(const fft1d_plan *p, const double *restrict x, double *restrict y)
{
    __m512d v[16];
    fft64_regs(p, x, v);
    for (int j = 0; j < 16; ++j) _mm512_storeu_pd(y + 8 * j, v[j]);
}

static void k64rm(const fft1d_plan *p, const double *restrict x, double *restrict y,
                  const double *restrict cf)
{
    __m512d v[16];
    fft64_regs(p, x, v);
    for (int j = 0; j < 16; ++j)
        _mm512_storeu_pd(y + 8 * j, mapv(v[j], _mm512_loadu_pd(cf + 8 * j)));
}

/* ---- L=128: two fused passes (stages 1+2, stages 3+4) via L1 scratch ----
 * Stage-index algebra: stage2's p-th butterfly consumes exactly stage1-output
 * vectors {p, p+8, p+16, p+24}, i.e. stage1 groups {0,2,4,6} feed stage2
 * p=0..3 and groups {1,3,5,7} feed p=4..7 -- so each half fuses in 16 live
 * vectors. Stage4's (q', q'+64) pairs are exactly stage3's p=0/p=1 outputs
 * for the same q-vector, so pass B never touches memory between them. */
static inline void fft128_passA(const fft1d_plan *p, const double *restrict x,
                                double *restrict s)
{
    const __m512d one = _mm512_set1_pd(1.0);
    for (int h = 0; h < 2; ++h) {
        __m512d U[4][4];
        for (int gi = 0; gi < 4; ++gi) {
            const int g = 2 * gi + h;
            __m512d y0, y1, y2, y3;
            BFLY4(_mm512_loadu_pd(x + 8 * g), _mm512_loadu_pd(x + 8 * (g + 8)),
                  _mm512_loadu_pd(x + 8 * (g + 16)), _mm512_loadu_pd(x + 8 * (g + 24)),
                  y0, y1, y2, y3, one);
            TWID_TRANSPOSE(p->tw1 + 48 * g, y0, y1, y2, y3,
                           U[gi][0], U[gi][1], U[gi][2], U[gi][3]);
        }
        for (int k = 0; k < 4; ++k) {
            const int pp = k + 4 * h;
            __m512d y0, y1, y2, y3;
            BFLY4(U[0][k], U[1][k], U[2][k], U[3][k], y0, y1, y2, y3, one);
            if (pp) CMUL3(p->twA, pp, y1, y2, y3);
            _mm512_storeu_pd(s + 8 * (4 * pp + 0), y0);
            _mm512_storeu_pd(s + 8 * (4 * pp + 1), y1);
            _mm512_storeu_pd(s + 8 * (4 * pp + 2), y2);
            _mm512_storeu_pd(s + 8 * (4 * pp + 3), y3);
        }
    }
}

/* pass B for one q-vector j: out[r] -> vec j+4r, out[4+r] -> vec j+16+4r */
static inline void fft128_passB_j(const fft1d_plan *p, const double *restrict s,
                                  const int j, __m512d out[8])
{
    const __m512d one = _mm512_set1_pd(1.0);
    __m512d A0, A1, A2, A3, B0, B1, B2, B3;
    BFLY4(_mm512_loadu_pd(s + 8 * j), _mm512_loadu_pd(s + 8 * (j + 8)),
          _mm512_loadu_pd(s + 8 * (j + 16)), _mm512_loadu_pd(s + 8 * (j + 24)),
          A0, A1, A2, A3, one);
    BFLY4(_mm512_loadu_pd(s + 8 * (j + 4)), _mm512_loadu_pd(s + 8 * (j + 12)),
          _mm512_loadu_pd(s + 8 * (j + 20)), _mm512_loadu_pd(s + 8 * (j + 28)),
          B0, B1, B2, B3, one);
    CMUL3(p->twB, 1, B1, B2, B3);
    out[0] = _mm512_add_pd(A0, B0); out[4] = _mm512_sub_pd(A0, B0);
    out[1] = _mm512_add_pd(A1, B1); out[5] = _mm512_sub_pd(A1, B1);
    out[2] = _mm512_add_pd(A2, B2); out[6] = _mm512_sub_pd(A2, B2);
    out[3] = _mm512_add_pd(A3, B3); out[7] = _mm512_sub_pd(A3, B3);
}

static void k128f(const fft1d_plan *p, const double *restrict x, double *restrict y)
{
    fft128_passA(p, x, p->s1);
    for (int j = 0; j < 4; ++j) {
        __m512d out[8];
        fft128_passB_j(p, p->s1, j, out);
        for (int r = 0; r < 4; ++r) {
            _mm512_storeu_pd(y + 8 * (j + 4 * r), out[r]);
            _mm512_storeu_pd(y + 8 * (j + 16 + 4 * r), out[4 + r]);
        }
    }
}

static void k128fm(const fft1d_plan *p, const double *restrict x, double *restrict y,
                   const double *restrict cf)
{
    fft128_passA(p, x, p->s1);
    for (int j = 0; j < 4; ++j) {
        __m512d out[8];
        fft128_passB_j(p, p->s1, j, out);
        for (int r = 0; r < 4; ++r) {
            _mm512_storeu_pd(y + 8 * (j + 4 * r),
                mapv(out[r], _mm512_loadu_pd(cf + 8 * (j + 4 * r))));
            _mm512_storeu_pd(y + 8 * (j + 16 + 4 * r),
                mapv(out[4 + r], _mm512_loadu_pd(cf + 8 * (j + 16 + 4 * r))));
        }
    }
}

/* ---- L=256: same two-fused-pass structure ----
 * Stage2 p consumes stage1-output vectors {p, p+16, p+32, p+48}: groups
 * {q,q+4,q+8,q+12} feed p=4q..4q+3. Stage3 p's outputs land at vec
 * j+16p+4r, which for fixed (j,r) are exactly stage4's four inputs. */
static inline void fft256_passA(const fft1d_plan *p, const double *restrict x,
                                double *restrict s)
{
    const __m512d one = _mm512_set1_pd(1.0);
    for (int qtr = 0; qtr < 4; ++qtr) {
        __m512d U[4][4];
        for (int gi = 0; gi < 4; ++gi) {
            const int g = qtr + 4 * gi;
            __m512d y0, y1, y2, y3;
            BFLY4(_mm512_loadu_pd(x + 8 * g), _mm512_loadu_pd(x + 8 * (g + 16)),
                  _mm512_loadu_pd(x + 8 * (g + 32)), _mm512_loadu_pd(x + 8 * (g + 48)),
                  y0, y1, y2, y3, one);
            TWID_TRANSPOSE(p->tw1 + 48 * g, y0, y1, y2, y3,
                           U[gi][0], U[gi][1], U[gi][2], U[gi][3]);
        }
        for (int k = 0; k < 4; ++k) {
            const int pp = 4 * qtr + k;
            __m512d y0, y1, y2, y3;
            BFLY4(U[0][k], U[1][k], U[2][k], U[3][k], y0, y1, y2, y3, one);
            if (pp) CMUL3(p->twA, pp, y1, y2, y3);
            _mm512_storeu_pd(s + 8 * (4 * pp + 0), y0);
            _mm512_storeu_pd(s + 8 * (4 * pp + 1), y1);
            _mm512_storeu_pd(s + 8 * (4 * pp + 2), y2);
            _mm512_storeu_pd(s + 8 * (4 * pp + 3), y3);
        }
    }
}

static inline void fft256_passB_j(const fft1d_plan *p, const double *restrict s,
                                  const int j, __m512d out[16])
{
    const __m512d one = _mm512_set1_pd(1.0);
    __m512d M[4][4];
    for (int pp = 0; pp < 4; ++pp) {
        __m512d y0, y1, y2, y3;
        BFLY4(_mm512_loadu_pd(s + 8 * (j + 4 * pp)),
              _mm512_loadu_pd(s + 8 * (j + 4 * pp + 16)),
              _mm512_loadu_pd(s + 8 * (j + 4 * pp + 32)),
              _mm512_loadu_pd(s + 8 * (j + 4 * pp + 48)),
              y0, y1, y2, y3, one);
        if (pp) CMUL3(p->twB, pp, y1, y2, y3);
        M[pp][0] = y0; M[pp][1] = y1; M[pp][2] = y2; M[pp][3] = y3;
    }
    for (int r = 0; r < 4; ++r)     /* out[4u+r] -> vec j+4r+16u */
        BFLY4(M[0][r], M[1][r], M[2][r], M[3][r],
              out[r], out[4 + r], out[8 + r], out[12 + r], one);
}

static void k256f(const fft1d_plan *p, const double *restrict x, double *restrict y)
{
    fft256_passA(p, x, p->s1);
    for (int j = 0; j < 4; ++j) {
        __m512d out[16];
        fft256_passB_j(p, p->s1, j, out);
        for (int u = 0; u < 4; ++u)
            for (int r = 0; r < 4; ++r)
                _mm512_storeu_pd(y + 8 * (j + 4 * r + 16 * u), out[4 * u + r]);
    }
}

static void k256fm(const fft1d_plan *p, const double *restrict x, double *restrict y,
                   const double *restrict cf)
{
    fft256_passA(p, x, p->s1);
    for (int j = 0; j < 4; ++j) {
        __m512d out[16];
        fft256_passB_j(p, p->s1, j, out);
        for (int u = 0; u < 4; ++u)
            for (int r = 0; r < 4; ++r) {
                const int o = 8 * (j + 4 * r + 16 * u);
                _mm512_storeu_pd(y + o, mapv(out[4 * u + r], _mm512_loadu_pd(cf + o)));
            }
    }
}

#else
/* ---------------- scalar fallback: iterative radix-2 Stockham ---------------- */

static void scalar_fft(const fft1d_plan *p, const double *in, double *out)
{
    const int L = p->L;
    cd *X = (cd *)p->s1, *Y = (cd *)p->s2;
    memcpy(X, in, (size_t)L * sizeof(cd));
    int n = L, s = 1;
    while (n > 1) {
        const int m = n / 2;
        for (int pp = 0; pp < m; ++pp) {
            const double ang = -2.0 * M_PI * (double)pp / (double)n;
            const cd w = cos(ang) + I * sin(ang);
            for (int q = 0; q < s; ++q) {
                cd a = X[q + s * pp], b = X[q + s * (pp + m)];
                Y[q + s * (2 * pp)] = a + b;
                Y[q + s * (2 * pp + 1)] = (a - b) * w;
            }
        }
        n = m; s *= 2;
        cd *tmp = X; X = Y; Y = tmp;
    }
    memcpy(out, X, (size_t)L * sizeof(cd));
}

static void sk(const fft1d_plan *p, const double *in, double *out)
{ scalar_fft(p, in, out); }

static void skm(const fft1d_plan *p, const double *in, double *out, const double *cf)
{
    const int L = p->L;
    scalar_fft(p, in, (double *)p->stt);
    const cd *z = (const cd *)p->stt, *c = (const cd *)cf;
    cd *o = (cd *)out;
    for (int i = 0; i < L; ++i) {
        cd w = z[i] + c[i];
        double re = creal(w), im = cimag(w);
        o[i] = w / (1.0 + sqrt(re * re + im * im));
    }
}
#endif

/* ---------------- plan setup ---------------- */

static double *amalloc(size_t doubles)
{
    void *q = NULL;
    if (posix_memalign(&q, 64, doubles * sizeof(double)) != 0) return NULL;
    return (double *)q;
}

/* twiddle w^{jp} for stage size n, j = 1..3: compact 6 doubles per p */
static void fill_compact(double *t, int n)
{
    for (int p = 0; p < n / 4; ++p)
        for (int j = 1; j <= 3; ++j) {
            double ang = -2.0 * M_PI * (double)(j * p) / (double)n;
            t[6 * p + 2 * (j - 1)] = cos(ang);
            t[6 * p + 2 * (j - 1) + 1] = sin(ang);
        }
}

fft1d_plan *fft1d_create(int L, int batch)
{
    if (!fft1d_supports(L) || batch < 1) return NULL;
    fft1d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L; p->batch = batch;

    const int m1 = L / 4;
    p->tw1 = amalloc((size_t)(m1 / 4) * 48);
    p->s1 = amalloc(2 * (size_t)L);
    p->s2 = amalloc(2 * (size_t)L);
    p->stt = amalloc(2 * (size_t)L);
    if (!p->tw1 || !p->s1 || !p->s2 || !p->stt) { fft1d_destroy(p); return NULL; }
    /* first stage: pair-duplicated vector layout, 48 doubles per 4 p's:
       [w1r x8][w1i x8][w2r x8][w2i x8][w3r x8][w3i x8] */
    for (int g = 0; g < m1 / 4; ++g)
        for (int k = 0; k < 4; ++k) {
            int pp = 4 * g + k;
            for (int j = 1; j <= 3; ++j) {
                double ang = -2.0 * M_PI * (double)(j * pp) / (double)L;
                double *blk = p->tw1 + 48 * g + 16 * (j - 1);
                blk[2 * k] = blk[2 * k + 1] = cos(ang);
                blk[8 + 2 * k] = blk[8 + 2 * k + 1] = sin(ang);
            }
        }
    /* stage 2 (n = L/4) and, for L >= 128, stage 3 (n = L/16) compact tables */
    if (L >= 32) {
        p->twA = amalloc(6 * (size_t)(L / 16));
        if (!p->twA) { fft1d_destroy(p); return NULL; }
        fill_compact(p->twA, L / 4);
    }
    if (L >= 128) {
        p->twB = amalloc(6 * (size_t)(L / 64));
        if (!p->twB) { fft1d_destroy(p); return NULL; }
        fill_compact(p->twB, L / 16);
    }
#ifdef D1P_AVX512
    switch (L) {
    case 16:  p->kern = k16r;  p->kmap = k16rm;  break;
    case 32:  p->kern = k32r;  p->kmap = k32rm;  break;
    case 64:  p->kern = k64r;  p->kmap = k64rm;  break;
    case 128: p->kern = k128f; p->kmap = k128fm; break;
    default:  p->kern = k256f; p->kmap = k256fm; break;
    }
#else
    p->kern = sk; p->kmap = skm;
#endif
    return p;
}

void fft1d_execute(fft1d_plan *p, const cd *in, cd *out)
{
    const int L = p->L;
    for (int b = 0; b < p->batch; ++b)
        p->kern(p, (const double *)(in + (size_t)b * L), (double *)(out + (size_t)b * L));
}

/* Owned chain: state <- (FFT(state)+c)/(1+|FFT(state)+c|), m steps.
 * Batch-outer: each lane's whole chain runs with its state, c-slice and the
 * FFT scratch L1-resident; the map is fused into the final FFT stage. */
void fft1d_chain(fft1d_plan *p, const cd *x0, const cd *c, cd *final_out, int m)
{
    const int L = p->L;
    if (m <= 0) { memcpy(final_out, x0, (size_t)L * p->batch * sizeof(cd)); return; }
    for (int b = 0; b < p->batch; ++b) {
        const double *xb = (const double *)(x0 + (size_t)b * L);
        const double *cb = (const double *)(c + (size_t)b * L);
        double *ob = (double *)(final_out + (size_t)b * L);
        const double *src = xb;
        for (int s = 0; s < m; ++s) {
            double *dst = (s == m - 1) ? ob : (double *)p->stt;
            p->kmap(p, src, dst, cb);
            src = (const double *)p->stt;
        }
    }
}

void fft1d_destroy(fft1d_plan *p)
{
    if (!p) return;
    free(p->tw1); free(p->twA); free(p->twB);
    free(p->s1); free(p->s2); free(p->stt);
    free(p);
}
