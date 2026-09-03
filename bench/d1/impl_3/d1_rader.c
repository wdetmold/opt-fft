/* d1_rader: prime-N 1D complex FFT by Rader's reduction to an (N-1)-point
 * cyclic convolution, engineered around the factorization of N-1.
 *
 *   N=65537: N-1 = 2^16  -> UNPADDED conv at M=65536, pure pow2 stages.
 *            The class headline: a Bluestein fallback pads to >= 131071.
 *   N=1021:  N-1 = 1020 -> UNPADDED conv, stages [4,3,5,17] with a dense
 *            symmetric-fold radix-17 final stage (no pad to 2048).
 *   N=127:   N-1 = 126 (7)  -> padded conv at M=256.
 *   N=13,31: N-1 = 12, 30 smooth -> UNPADDED conv at M=12, 30.
 *
 * Runtime per transform: gather x[g^q] fused into the first conv-FFT stage,
 * forward FFT, kernel-spectrum multiply fused into the inverse's first stage
 * (inverse = forward on swapped re/im planes, 1/M folded into the kernel),
 * inverse FFT, and the output scatter X[g^-m] = x[0] + conv[m] fused into the
 * last stage.  X[0] = x[0] + (forward conv DC bin) -- free.
 *
 * Conv core: split-complex mixed-radix (2/3/4/5/8) Stockham, per-stage
 * plan-time twiddle tables, out-of-place ping-pong. Core structure and the
 * ivdep / per-function-512-bit lessons adopted from d1_bluestein (round
 * d1_r1 record).
 */
#include <complex.h>
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../fft1d_api.h"

#define HOT __attribute__((target("arch=icelake-server,prefer-vector-width=512")))

#define MAXSTAGE 40

/* ---------------- split-complex mixed-radix Stockham core ---------------- */

typedef struct {
    int n;
    int nstage;
    int radix[MAXSTAGE];
    double *twr[MAXSTAGE];   /* per-stage twiddles, blocks [j-1][p], j=1..r-1 */
    double *twi[MAXSTAGE];
    double *twstore;
} core_plan;

static void *amalloc(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
}

/* Factor n into stages. Entry stage is radix 4 (or 2) so the gather / kernel
   fused variants apply; radix-8 workhorses in the middle; the pow2 leftover
   is arranged so the LAST stage is radix 4 or 2 whenever the exit scatter
   wants to fuse (pow2 leftover of 3 becomes [4,2], of 4 becomes [4,4]). */
static int core_factor(int n, int *radix, int *nstage)
{
    int ns = 0, a = 0, seventeen = 0;
    if (n == 65536) {   /* the 65537 conv: 10 array passes instead of 12 */
        static const int st65536[5] = { 4, 16, 16, 16, 4 };
        for (int i = 0; i < 5; ++i) radix[i] = st65536[i];
        *nstage = 5;
        return 1;
    }
    if (n % 17 == 0) { n /= 17; seventeen = 1; }   /* one dense radix-17, last */
    while (n % 2 == 0) { ++a; n /= 2; }
    if (a >= 2) { radix[ns++] = 4; a -= 2; }
    else if (a == 1) { radix[ns++] = 2; a -= 1; }
    while (a > 4) { radix[ns++] = 8; a -= 3; }
    while (n % 3 == 0) { radix[ns++] = 3; n /= 3; }
    while (n % 5 == 0) { radix[ns++] = 5; n /= 5; }
    switch (a) {
    case 4: radix[ns++] = 4; radix[ns++] = 4; break;
    case 3: radix[ns++] = 4; radix[ns++] = 2; break;   /* fused pruned exit beat [.,8] by ~5% at M=2048 */
    case 2: radix[ns++] = 4; break;
    case 1: radix[ns++] = 2; break;
    default: break;
    }
    if (seventeen) radix[ns++] = 17;   /* m == 1 there: st17 assumes unit twiddles */
    *nstage = ns;
    return n == 1 && ns >= 1 && ns <= MAXSTAGE;
}

static int core_init_stages(core_plan *c, int n, const int *stages, int nstages)
{
    c->n = n;
    if (stages) {
        c->nstage = nstages;
        for (int i = 0; i < nstages; ++i) c->radix[i] = stages[i];
    } else if (!core_factor(n, c->radix, &c->nstage)) return 0;
    size_t tot = 0;
    int ncur = n;
    for (int st = 0; st < c->nstage; ++st) {
        int r = c->radix[st], m = ncur / r;
        tot += (size_t)(r - 1) * m;
        ncur = m;
    }
    c->twstore = amalloc(2 * tot * sizeof(double));
    if (!c->twstore) return 0;
    double *wr = c->twstore, *wi = c->twstore + tot;
    ncur = n;
    for (int st = 0; st < c->nstage; ++st) {
        int r = c->radix[st], m = ncur / r;
        c->twr[st] = wr; c->twi[st] = wi;
        for (int j = 1; j < r; ++j)
            for (int p = 0; p < m; ++p) {
                double ph = -2.0 * M_PI * ((double)p * j) / (double)ncur;
                wr[(size_t)(j - 1) * m + p] = cos(ph);
                wi[(size_t)(j - 1) * m + p] = sin(ph);
            }
        wr += (size_t)(r - 1) * m; wi += (size_t)(r - 1) * m;
        ncur = m;
    }
    return 1;
}

static int core_init(core_plan *c, int n) { return core_init_stages(c, n, NULL, 0); }

static void core_free(core_plan *c) { free(c->twstore); c->twstore = NULL; }

/* radix-2 stage */
static void st2(int m, int s, const double *restrict wr, const double *restrict wi,
                const double *restrict xr, const double *restrict xi,
                double *restrict yr, double *restrict yi)
{
    if (s == 1) {
#pragma GCC ivdep
        for (int p = 0; p < m; ++p) {
            double ar = xr[p], ai = xi[p], br = xr[p + m], bi = xi[p + m];
            double ur = ar - br, ui = ai - bi;
            yr[2 * p] = ar + br;             yi[2 * p] = ai + bi;
            yr[2 * p + 1] = ur * wr[p] - ui * wi[p];
            yi[2 * p + 1] = ur * wi[p] + ui * wr[p];
        }
        return;
    }
    for (int p = 0; p < m; ++p) {
        const double w1r = wr[p], w1i = wi[p];
        const double *restrict x0r = xr + (size_t)s * p;
        const double *restrict x0i = xi + (size_t)s * p;
        const double *restrict x1r = x0r + (size_t)s * m;
        const double *restrict x1i = x0i + (size_t)s * m;
        double *restrict o0r = yr + (size_t)s * 2 * p;
        double *restrict o0i = yi + (size_t)s * 2 * p;
        double *restrict o1r = o0r + s;
        double *restrict o1i = o0i + s;
#pragma GCC ivdep
        for (int q = 0; q < s; ++q) {
            double ar = x0r[q], ai = x0i[q], br = x1r[q], bi = x1i[q];
            double ur = ar - br, ui = ai - bi;
            o0r[q] = ar + br;  o0i[q] = ai + bi;
            o1r[q] = ur * w1r - ui * w1i;
            o1i[q] = ur * w1i + ui * w1r;
        }
    }
}

/* radix-4 stage: X0=t0+t2, X1=t1-i t3, X2=t0-t2, X3=t1+i t3 */
static void st4(int m, int s, const double *restrict wr, const double *restrict wi,
                const double *restrict xr, const double *restrict xi,
                double *restrict yr, double *restrict yi)
{
    const double *restrict w1r = wr,        *restrict w1i = wi;
    const double *restrict w2r = wr + m,    *restrict w2i = wi + m;
    const double *restrict w3r = wr + 2 * m, *restrict w3i = wi + 2 * m;
    if (s == 1) {
#pragma GCC ivdep
        for (int p = 0; p < m; ++p) {
            double ar = xr[p],         ai = xi[p];
            double br = xr[p + m],     bi = xi[p + m];
            double cr = xr[p + 2 * m], ci = xi[p + 2 * m];
            double dr = xr[p + 3 * m], di = xi[p + 3 * m];
            double t0r = ar + cr, t0i = ai + ci, t1r = ar - cr, t1i = ai - ci;
            double t2r = br + dr, t2i = bi + di, t3r = br - dr, t3i = bi - di;
            double y1r = t1r + t3i, y1i = t1i - t3r;
            double y2r = t0r - t2r, y2i = t0i - t2i;
            double y3r = t1r - t3i, y3i = t1i + t3r;
            yr[4 * p] = t0r + t2r;  yi[4 * p] = t0i + t2i;
            yr[4 * p + 1] = y1r * w1r[p] - y1i * w1i[p];
            yi[4 * p + 1] = y1r * w1i[p] + y1i * w1r[p];
            yr[4 * p + 2] = y2r * w2r[p] - y2i * w2i[p];
            yi[4 * p + 2] = y2r * w2i[p] + y2i * w2r[p];
            yr[4 * p + 3] = y3r * w3r[p] - y3i * w3i[p];
            yi[4 * p + 3] = y3r * w3i[p] + y3i * w3r[p];
        }
        return;
    }
    for (int p = 0; p < m; ++p) {
        const double u1r = w1r[p], u1i = w1i[p];
        const double u2r = w2r[p], u2i = w2i[p];
        const double u3r = w3r[p], u3i = w3i[p];
        const double *restrict x0r = xr + (size_t)s * p, *restrict x0i = xi + (size_t)s * p;
        const double *restrict x1r = x0r + (size_t)s * m, *restrict x1i = x0i + (size_t)s * m;
        const double *restrict x2r = x1r + (size_t)s * m, *restrict x2i = x1i + (size_t)s * m;
        const double *restrict x3r = x2r + (size_t)s * m, *restrict x3i = x2i + (size_t)s * m;
        double *restrict o0r = yr + (size_t)s * 4 * p, *restrict o0i = yi + (size_t)s * 4 * p;
        double *restrict o1r = o0r + s, *restrict o1i = o0i + s;
        double *restrict o2r = o1r + s, *restrict o2i = o1i + s;
        double *restrict o3r = o2r + s, *restrict o3i = o2i + s;
#pragma GCC ivdep
        for (int q = 0; q < s; ++q) {
            double ar = x0r[q], ai = x0i[q];
            double br = x1r[q], bi = x1i[q];
            double cr = x2r[q], ci = x2i[q];
            double dr = x3r[q], di = x3i[q];
            double t0r = ar + cr, t0i = ai + ci, t1r = ar - cr, t1i = ai - ci;
            double t2r = br + dr, t2i = bi + di, t3r = br - dr, t3i = bi - di;
            double y1r = t1r + t3i, y1i = t1i - t3r;
            double y2r = t0r - t2r, y2i = t0i - t2i;
            double y3r = t1r - t3i, y3i = t1i + t3r;
            o0r[q] = t0r + t2r;  o0i[q] = t0i + t2i;
            o1r[q] = y1r * u1r - y1i * u1i;  o1i[q] = y1r * u1i + y1i * u1r;
            o2r[q] = y2r * u2r - y2i * u2i;  o2i[q] = y2r * u2i + y2i * u2r;
            o3r[q] = y3r * u3r - y3i * u3i;  o3i[q] = y3r * u3i + y3i * u3r;
        }
    }
}

/* radix-8 as even/odd DFT4 + combine */
static void st8(int m, int s, const double *restrict wr, const double *restrict wi,
                const double *restrict xr, const double *restrict xi,
                double *restrict yr, double *restrict yi)
{
    const double C8 = 0.70710678118654752440;
    for (int p = 0; p < m; ++p) {
        const double v1r = wr[p], v1i = wi[p];
        const double v2r = wr[m + p], v2i = wi[m + p];
        const double v3r = wr[2 * m + p], v3i = wi[2 * m + p];
        const double v4r = wr[3 * m + p], v4i = wi[3 * m + p];
        const double v5r = wr[4 * m + p], v5i = wi[4 * m + p];
        const double v6r = wr[5 * m + p], v6i = wi[5 * m + p];
        const double v7r = wr[6 * m + p], v7i = wi[6 * m + p];
        const double *restrict x0r = xr + (size_t)s * p, *restrict x0i = xi + (size_t)s * p;
        const size_t sm = (size_t)s * m;
        double *restrict o0r = yr + (size_t)s * 8 * p, *restrict o0i = yi + (size_t)s * 8 * p;
#pragma GCC ivdep
        for (int q = 0; q < s; ++q) {
            double a0r = x0r[q],          a0i = x0i[q];
            double a1r = x0r[q + sm],     a1i = x0i[q + sm];
            double a2r = x0r[q + 2 * sm], a2i = x0i[q + 2 * sm];
            double a3r = x0r[q + 3 * sm], a3i = x0i[q + 3 * sm];
            double a4r = x0r[q + 4 * sm], a4i = x0i[q + 4 * sm];
            double a5r = x0r[q + 5 * sm], a5i = x0i[q + 5 * sm];
            double a6r = x0r[q + 6 * sm], a6i = x0i[q + 6 * sm];
            double a7r = x0r[q + 7 * sm], a7i = x0i[q + 7 * sm];
            double t0r = a0r + a4r, t0i = a0i + a4i, t1r = a0r - a4r, t1i = a0i - a4i;
            double t2r = a2r + a6r, t2i = a2i + a6i, t3r = a2r - a6r, t3i = a2i - a6i;
            double e0r = t0r + t2r, e0i = t0i + t2i;
            double e1r = t1r + t3i, e1i = t1i - t3r;
            double e2r = t0r - t2r, e2i = t0i - t2i;
            double e3r = t1r - t3i, e3i = t1i + t3r;
            double u0r = a1r + a5r, u0i = a1i + a5i, u1r = a1r - a5r, u1i = a1i - a5i;
            double u2r = a3r + a7r, u2i = a3i + a7i, u3r = a3r - a7r, u3i = a3i - a7i;
            double f0r = u0r + u2r, f0i = u0i + u2i;
            double f1r = u1r + u3i, f1i = u1i - u3r;
            double f2r = u0r - u2r, f2i = u0i - u2i;
            double f3r = u1r - u3i, f3i = u1i + u3r;
            double g1r = C8 * (f1r + f1i), g1i = C8 * (f1i - f1r);
            double g2r = f2i,              g2i = -f2r;
            double g3r = C8 * (f3i - f3r), g3i = -C8 * (f3r + f3i);
            double z0r = e0r + f0r, z0i = e0i + f0i;
            double z4r = e0r - f0r, z4i = e0i - f0i;
            double z1r = e1r + g1r, z1i = e1i + g1i;
            double z5r = e1r - g1r, z5i = e1i - g1i;
            double z2r = e2r + g2r, z2i = e2i + g2i;
            double z6r = e2r - g2r, z6i = e2i - g2i;
            double z3r = e3r + g3r, z3i = e3i + g3i;
            double z7r = e3r - g3r, z7i = e3i - g3i;
            o0r[q] = z0r;  o0i[q] = z0i;
            o0r[q + s]     = z1r * v1r - z1i * v1i;  o0i[q + s]     = z1r * v1i + z1i * v1r;
            o0r[q + 2 * s] = z2r * v2r - z2i * v2i;  o0i[q + 2 * s] = z2r * v2i + z2i * v2r;
            o0r[q + 3 * s] = z3r * v3r - z3i * v3i;  o0i[q + 3 * s] = z3r * v3i + z3i * v3r;
            o0r[q + 4 * s] = z4r * v4r - z4i * v4i;  o0i[q + 4 * s] = z4r * v4i + z4i * v4r;
            o0r[q + 5 * s] = z5r * v5r - z5i * v5i;  o0i[q + 5 * s] = z5r * v5i + z5i * v5r;
            o0r[q + 6 * s] = z6r * v6r - z6i * v6i;  o0i[q + 6 * s] = z6r * v6i + z6i * v6r;
            o0r[q + 7 * s] = z7r * v7r - z7i * v7i;  o0i[q + 7 * s] = z7r * v7i + z7i * v7r;
        }
    }
}

/* radix-16 as two fused radix-4 layers through a 2 KB L1 tile (the st17
 * two-pass shape: keeping all 16 complex lane-vectors live is the whole
 * zmm file and gcc spills).  Layer 1: DFT4 over v2 of x[v1+4v2], times
 * W16^(u1 v1) (compile-time constants).  Layer 2: DFT4 over v1, times the
 * stage twiddle table.  One radix-16 pass replaces two radix-8-ish passes
 * over the M-length arrays -- the 65536 conv is L3-bandwidth-bound, so
 * pass count is what the cell pays for. */
#define C16A 0.92387953251128675613   /* cos(pi/8) */
#define C16B 0.38268343236508977173   /* sin(pi/8) */
#define C16H 0.70710678118654752440   /* sqrt(2)/2 */

__attribute__((always_inline)) static inline void
st16_block(size_t sm, int s, int m, int q0, __mmask8 mk,
           const double *restrict wrp, const double *restrict wip,
           const double *restrict xpr, const double *restrict xpi,
           double *restrict ypr, double *restrict ypi)
{
    __attribute__((aligned(64))) double zr[16][8], zi[16][8];
    const __m512d cA = _mm512_set1_pd(C16A), cB = _mm512_set1_pd(C16B);
    const __m512d cH = _mm512_set1_pd(C16H);
    /* layer 1: v1 groups; z[v1][u1] = W16^(u1 v1) * DFT4_v2(x[v1+4v2]) */
#pragma GCC unroll 4
    for (int v1 = 0; v1 < 4; ++v1) {
        __m512d ar = _mm512_maskz_loadu_pd(mk, xpr + q0 + sm * v1);
        __m512d ai = _mm512_maskz_loadu_pd(mk, xpi + q0 + sm * v1);
        __m512d br = _mm512_maskz_loadu_pd(mk, xpr + q0 + sm * (v1 + 4));
        __m512d bi = _mm512_maskz_loadu_pd(mk, xpi + q0 + sm * (v1 + 4));
        __m512d cr = _mm512_maskz_loadu_pd(mk, xpr + q0 + sm * (v1 + 8));
        __m512d ci = _mm512_maskz_loadu_pd(mk, xpi + q0 + sm * (v1 + 8));
        __m512d dr = _mm512_maskz_loadu_pd(mk, xpr + q0 + sm * (v1 + 12));
        __m512d di = _mm512_maskz_loadu_pd(mk, xpi + q0 + sm * (v1 + 12));
        __m512d t0r = _mm512_add_pd(ar, cr), t0i = _mm512_add_pd(ai, ci);
        __m512d t1r = _mm512_sub_pd(ar, cr), t1i = _mm512_sub_pd(ai, ci);
        __m512d t2r = _mm512_add_pd(br, dr), t2i = _mm512_add_pd(bi, di);
        __m512d t3r = _mm512_sub_pd(br, dr), t3i = _mm512_sub_pd(bi, di);
        __m512d y0r = _mm512_add_pd(t0r, t2r), y0i = _mm512_add_pd(t0i, t2i);
        __m512d y1r = _mm512_add_pd(t1r, t3i), y1i = _mm512_sub_pd(t1i, t3r);
        __m512d y2r = _mm512_sub_pd(t0r, t2r), y2i = _mm512_sub_pd(t0i, t2i);
        __m512d y3r = _mm512_sub_pd(t1r, t3i), y3i = _mm512_add_pd(t1i, t3r);
        __m512d z1r, z1i, z2r, z2i, z3r, z3i;
        switch (v1) {
        default:   /* v1 = 0: unit twiddles */
            z1r = y1r; z1i = y1i; z2r = y2r; z2i = y2i; z3r = y3r; z3i = y3i;
            break;
        case 1:    /* W16^1, W16^2, W16^3 */
            z1r = _mm512_fmadd_pd(cA, y1r, _mm512_mul_pd(cB, y1i));
            z1i = _mm512_fmsub_pd(cA, y1i, _mm512_mul_pd(cB, y1r));
            z2r = _mm512_mul_pd(cH, _mm512_add_pd(y2r, y2i));
            z2i = _mm512_mul_pd(cH, _mm512_sub_pd(y2i, y2r));
            z3r = _mm512_fmadd_pd(cB, y3r, _mm512_mul_pd(cA, y3i));
            z3i = _mm512_fmsub_pd(cB, y3i, _mm512_mul_pd(cA, y3r));
            break;
        case 2:    /* W16^2, W16^4 = -i, W16^6 */
            z1r = _mm512_mul_pd(cH, _mm512_add_pd(y1r, y1i));
            z1i = _mm512_mul_pd(cH, _mm512_sub_pd(y1i, y1r));
            z2r = y2i;
            z2i = _mm512_sub_pd(_mm512_setzero_pd(), y2r);
            z3r = _mm512_mul_pd(cH, _mm512_sub_pd(y3i, y3r));
            z3i = _mm512_sub_pd(_mm512_setzero_pd(),
                                _mm512_mul_pd(cH, _mm512_add_pd(y3r, y3i)));
            break;
        case 3:    /* W16^3, W16^6, W16^9 = -W16^1 */
            z1r = _mm512_fmadd_pd(cB, y1r, _mm512_mul_pd(cA, y1i));
            z1i = _mm512_fmsub_pd(cB, y1i, _mm512_mul_pd(cA, y1r));
            z2r = _mm512_mul_pd(cH, _mm512_sub_pd(y2i, y2r));
            z2i = _mm512_sub_pd(_mm512_setzero_pd(),
                                _mm512_mul_pd(cH, _mm512_add_pd(y2r, y2i)));
            z3r = _mm512_sub_pd(_mm512_setzero_pd(),
                      _mm512_fmadd_pd(cA, y3r, _mm512_mul_pd(cB, y3i)));
            z3i = _mm512_fmsub_pd(cB, y3r, _mm512_mul_pd(cA, y3i));
            break;
        }
        _mm512_store_pd(zr[v1],      y0r);  _mm512_store_pd(zi[v1],      y0i);
        _mm512_store_pd(zr[v1 + 4],  z1r);  _mm512_store_pd(zi[v1 + 4],  z1i);
        _mm512_store_pd(zr[v1 + 8],  z2r);  _mm512_store_pd(zi[v1 + 8],  z2i);
        _mm512_store_pd(zr[v1 + 12], z3r);  _mm512_store_pd(zi[v1 + 12], z3i);
    }
    /* layer 2: u1 groups; y[u1+4u2] = tw[u-1] * DFT4_v1(z[v1][u1]) */
#pragma GCC unroll 4
    for (int u1 = 0; u1 < 4; ++u1) {
        /* slot(v1,u1) = v1 + 4*u1, as stored by layer 1 */
        __m512d ar = _mm512_load_pd(zr[4 * u1 + 0]), ai = _mm512_load_pd(zi[4 * u1 + 0]);
        __m512d br = _mm512_load_pd(zr[4 * u1 + 1]), bi = _mm512_load_pd(zi[4 * u1 + 1]);
        __m512d cr = _mm512_load_pd(zr[4 * u1 + 2]), ci = _mm512_load_pd(zi[4 * u1 + 2]);
        __m512d dr = _mm512_load_pd(zr[4 * u1 + 3]), di = _mm512_load_pd(zi[4 * u1 + 3]);
        __m512d t0r = _mm512_add_pd(ar, cr), t0i = _mm512_add_pd(ai, ci);
        __m512d t1r = _mm512_sub_pd(ar, cr), t1i = _mm512_sub_pd(ai, ci);
        __m512d t2r = _mm512_add_pd(br, dr), t2i = _mm512_add_pd(bi, di);
        __m512d t3r = _mm512_sub_pd(br, dr), t3i = _mm512_sub_pd(bi, di);
        __m512d o0r = _mm512_add_pd(t0r, t2r), o0i = _mm512_add_pd(t0i, t2i);
        __m512d o1r = _mm512_add_pd(t1r, t3i), o1i = _mm512_sub_pd(t1i, t3r);
        __m512d o2r = _mm512_sub_pd(t0r, t2r), o2i = _mm512_sub_pd(t0i, t2i);
        __m512d o3r = _mm512_sub_pd(t1r, t3i), o3i = _mm512_add_pd(t1i, t3r);
        if (u1 == 0) {
            _mm512_mask_storeu_pd(ypr + q0, mk, o0r);
            _mm512_mask_storeu_pd(ypi + q0, mk, o0i);
        } else {
            __m512d w = _mm512_set1_pd(wrp[(size_t)(u1 - 1) * m]);
            __m512d v = _mm512_set1_pd(wip[(size_t)(u1 - 1) * m]);
            _mm512_mask_storeu_pd(ypr + q0 + (size_t)s * u1, mk,
                _mm512_fmsub_pd(o0r, w, _mm512_mul_pd(o0i, v)));
            _mm512_mask_storeu_pd(ypi + q0 + (size_t)s * u1, mk,
                _mm512_fmadd_pd(o0r, v, _mm512_mul_pd(o0i, w)));
        }
#define ST16_TWSTORE(ov_r, ov_i, u)                                            \
        do {                                                                   \
            __m512d w_ = _mm512_set1_pd(wrp[(size_t)((u) - 1) * m]);           \
            __m512d v_ = _mm512_set1_pd(wip[(size_t)((u) - 1) * m]);           \
            _mm512_mask_storeu_pd(ypr + q0 + (size_t)s * (u), mk,              \
                _mm512_fmsub_pd((ov_r), w_, _mm512_mul_pd((ov_i), v_)));       \
            _mm512_mask_storeu_pd(ypi + q0 + (size_t)s * (u), mk,              \
                _mm512_fmadd_pd((ov_r), v_, _mm512_mul_pd((ov_i), w_)));       \
        } while (0)
        ST16_TWSTORE(o1r, o1i, u1 + 4);
        ST16_TWSTORE(o2r, o2i, u1 + 8);
        ST16_TWSTORE(o3r, o3i, u1 + 12);
#undef ST16_TWSTORE
    }
}

/* s == 4 path: two p-groups per zmm (8 contiguous doubles are two radix-16
 * inputs at s=4), twiddles pair-broadcast, outputs stored as 256-bit halves
 * 64 apart -- the st3_s4 shape.  This stage runs twice per 65537 transform
 * (forward + inverse) and the masked-4 form wastes half of every vector. */
static void st16_s4(int m, const double *restrict wr, const double *restrict wi,
                    const double *restrict xr, const double *restrict xi,
                    double *restrict yr, double *restrict yi)
{
    const size_t sm = 4 * (size_t)m;
    const __m512i PAIR = _mm512_setr_epi64(0, 0, 0, 0, 1, 1, 1, 1);
    const __m512d cA = _mm512_set1_pd(C16A), cB = _mm512_set1_pd(C16B);
    const __m512d cH = _mm512_set1_pd(C16H);
    for (int p = 0; p + 2 <= m; p += 2) {
        const double *restrict xpr = xr + 4 * (size_t)p;
        const double *restrict xpi = xi + 4 * (size_t)p;
        double *restrict ypr = yr + 64 * (size_t)p;
        double *restrict ypi = yi + 64 * (size_t)p;
        __attribute__((aligned(64))) double zr[16][8], zi[16][8];
#pragma GCC unroll 4
        for (int v1 = 0; v1 < 4; ++v1) {
            __m512d ar = _mm512_loadu_pd(xpr + sm * v1);
            __m512d ai = _mm512_loadu_pd(xpi + sm * v1);
            __m512d br = _mm512_loadu_pd(xpr + sm * (v1 + 4));
            __m512d bi = _mm512_loadu_pd(xpi + sm * (v1 + 4));
            __m512d cr = _mm512_loadu_pd(xpr + sm * (v1 + 8));
            __m512d ci = _mm512_loadu_pd(xpi + sm * (v1 + 8));
            __m512d dr = _mm512_loadu_pd(xpr + sm * (v1 + 12));
            __m512d di = _mm512_loadu_pd(xpi + sm * (v1 + 12));
            __m512d t0r = _mm512_add_pd(ar, cr), t0i = _mm512_add_pd(ai, ci);
            __m512d t1r = _mm512_sub_pd(ar, cr), t1i = _mm512_sub_pd(ai, ci);
            __m512d t2r = _mm512_add_pd(br, dr), t2i = _mm512_add_pd(bi, di);
            __m512d t3r = _mm512_sub_pd(br, dr), t3i = _mm512_sub_pd(bi, di);
            __m512d y0r = _mm512_add_pd(t0r, t2r), y0i = _mm512_add_pd(t0i, t2i);
            __m512d y1r = _mm512_add_pd(t1r, t3i), y1i = _mm512_sub_pd(t1i, t3r);
            __m512d y2r = _mm512_sub_pd(t0r, t2r), y2i = _mm512_sub_pd(t0i, t2i);
            __m512d y3r = _mm512_sub_pd(t1r, t3i), y3i = _mm512_add_pd(t1i, t3r);
            __m512d z1r, z1i, z2r, z2i, z3r, z3i;
            switch (v1) {
            default:
                z1r = y1r; z1i = y1i; z2r = y2r; z2i = y2i; z3r = y3r; z3i = y3i;
                break;
            case 1:
                z1r = _mm512_fmadd_pd(cA, y1r, _mm512_mul_pd(cB, y1i));
                z1i = _mm512_fmsub_pd(cA, y1i, _mm512_mul_pd(cB, y1r));
                z2r = _mm512_mul_pd(cH, _mm512_add_pd(y2r, y2i));
                z2i = _mm512_mul_pd(cH, _mm512_sub_pd(y2i, y2r));
                z3r = _mm512_fmadd_pd(cB, y3r, _mm512_mul_pd(cA, y3i));
                z3i = _mm512_fmsub_pd(cB, y3i, _mm512_mul_pd(cA, y3r));
                break;
            case 2:
                z1r = _mm512_mul_pd(cH, _mm512_add_pd(y1r, y1i));
                z1i = _mm512_mul_pd(cH, _mm512_sub_pd(y1i, y1r));
                z2r = y2i;
                z2i = _mm512_sub_pd(_mm512_setzero_pd(), y2r);
                z3r = _mm512_mul_pd(cH, _mm512_sub_pd(y3i, y3r));
                z3i = _mm512_sub_pd(_mm512_setzero_pd(),
                                    _mm512_mul_pd(cH, _mm512_add_pd(y3r, y3i)));
                break;
            case 3:
                z1r = _mm512_fmadd_pd(cB, y1r, _mm512_mul_pd(cA, y1i));
                z1i = _mm512_fmsub_pd(cB, y1i, _mm512_mul_pd(cA, y1r));
                z2r = _mm512_mul_pd(cH, _mm512_sub_pd(y2i, y2r));
                z2i = _mm512_sub_pd(_mm512_setzero_pd(),
                                    _mm512_mul_pd(cH, _mm512_add_pd(y2r, y2i)));
                z3r = _mm512_sub_pd(_mm512_setzero_pd(),
                          _mm512_fmadd_pd(cA, y3r, _mm512_mul_pd(cB, y3i)));
                z3i = _mm512_fmsub_pd(cB, y3r, _mm512_mul_pd(cA, y3i));
                break;
            }
            _mm512_store_pd(zr[v1],      y0r);  _mm512_store_pd(zi[v1],      y0i);
            _mm512_store_pd(zr[v1 + 4],  z1r);  _mm512_store_pd(zi[v1 + 4],  z1i);
            _mm512_store_pd(zr[v1 + 8],  z2r);  _mm512_store_pd(zi[v1 + 8],  z2i);
            _mm512_store_pd(zr[v1 + 12], z3r);  _mm512_store_pd(zi[v1 + 12], z3i);
        }
#pragma GCC unroll 4
        for (int u1 = 0; u1 < 4; ++u1) {
            __m512d ar = _mm512_load_pd(zr[4 * u1 + 0]), ai = _mm512_load_pd(zi[4 * u1 + 0]);
            __m512d br = _mm512_load_pd(zr[4 * u1 + 1]), bi = _mm512_load_pd(zi[4 * u1 + 1]);
            __m512d cr = _mm512_load_pd(zr[4 * u1 + 2]), ci = _mm512_load_pd(zi[4 * u1 + 2]);
            __m512d dr = _mm512_load_pd(zr[4 * u1 + 3]), di = _mm512_load_pd(zi[4 * u1 + 3]);
            __m512d t0r = _mm512_add_pd(ar, cr), t0i = _mm512_add_pd(ai, ci);
            __m512d t1r = _mm512_sub_pd(ar, cr), t1i = _mm512_sub_pd(ai, ci);
            __m512d t2r = _mm512_add_pd(br, dr), t2i = _mm512_add_pd(bi, di);
            __m512d t3r = _mm512_sub_pd(br, dr), t3i = _mm512_sub_pd(bi, di);
            __m512d o0r = _mm512_add_pd(t0r, t2r), o0i = _mm512_add_pd(t0i, t2i);
            __m512d o1r = _mm512_add_pd(t1r, t3i), o1i = _mm512_sub_pd(t1i, t3r);
            __m512d o2r = _mm512_sub_pd(t0r, t2r), o2i = _mm512_sub_pd(t0i, t2i);
            __m512d o3r = _mm512_sub_pd(t1r, t3i), o3i = _mm512_add_pd(t1i, t3r);
#define ST16_S4STORE(ov_r, ov_i, u)                                            \
            do {                                                               \
                __m512d fr_ = (ov_r), fi_ = (ov_i);                            \
                if ((u) > 0) {                                                 \
                    __m512d w_ = _mm512_permutexvar_pd(PAIR,                   \
                        _mm512_castpd128_pd512(_mm_loadu_pd(wr + (size_t)((u) - 1) * m + p))); \
                    __m512d v_ = _mm512_permutexvar_pd(PAIR,                   \
                        _mm512_castpd128_pd512(_mm_loadu_pd(wi + (size_t)((u) - 1) * m + p))); \
                    __m512d tr_ = _mm512_fmsub_pd(fr_, w_, _mm512_mul_pd(fi_, v_)); \
                    fi_ = _mm512_fmadd_pd(fr_, v_, _mm512_mul_pd(fi_, w_));    \
                    fr_ = tr_;                                                 \
                }                                                              \
                _mm256_storeu_pd(ypr + 4 * (u),      _mm512_castpd512_pd256(fr_)); \
                _mm256_storeu_pd(ypr + 4 * (u) + 64, _mm512_extractf64x4_pd(fr_, 1)); \
                _mm256_storeu_pd(ypi + 4 * (u),      _mm512_castpd512_pd256(fi_)); \
                _mm256_storeu_pd(ypi + 4 * (u) + 64, _mm512_extractf64x4_pd(fi_, 1)); \
            } while (0)
            ST16_S4STORE(o0r, o0i, u1);
            ST16_S4STORE(o1r, o1i, u1 + 4);
            ST16_S4STORE(o2r, o2i, u1 + 8);
            ST16_S4STORE(o3r, o3i, u1 + 12);
#undef ST16_S4STORE
        }
    }
}

static void st16(int m, int s, const double *restrict wr, const double *restrict wi,
                 const double *restrict xr, const double *restrict xi,
                 double *restrict yr, double *restrict yi)
{
    if (s == 4 && (m & 1) == 0) { st16_s4(m, wr, wi, xr, xi, yr, yi); return; }
    const size_t sm = (size_t)s * m;
    for (int p = 0; p < m; ++p) {
        const double *restrict xpr = xr + (size_t)s * p;
        const double *restrict xpi = xi + (size_t)s * p;
        double *restrict ypr = yr + (size_t)s * 16 * p;
        double *restrict ypi = yi + (size_t)s * 16 * p;
        int q0 = 0;
        for (; q0 + 8 <= s; q0 += 8)
            st16_block(sm, s, m, q0, (__mmask8)0xFF, wr + p, wi + p,
                       xpr, xpi, ypr, ypi);
        if (q0 < s)
            st16_block(sm, s, m, q0, (__mmask8)((1u << (s - q0)) - 1),
                       wr + p, wi + p, xpr, xpi, ypr, ypi);
    }
}

/* store 4 lane-vectors interleaved as y[4p+u], p = p0..p0+7, u = 0..3:
   the Stockham radix-4 output pattern, done with 12 shuffles per plane */
__attribute__((always_inline)) static inline void
interleave4_store(double *restrict y, int p0, __m512d A, __m512d B, __m512d C, __m512d D)
{
    const __m512i I0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    const __m512i I1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
    const __m512i J0 = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
    const __m512i J1 = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);
    __m512d T0 = _mm512_unpacklo_pd(A, B), T1 = _mm512_unpackhi_pd(A, B);
    __m512d T2 = _mm512_unpacklo_pd(C, D), T3 = _mm512_unpackhi_pd(C, D);
    __m512d P0 = _mm512_permutex2var_pd(T0, I0, T2);
    __m512d P1 = _mm512_permutex2var_pd(T1, I0, T3);
    __m512d P2 = _mm512_permutex2var_pd(T0, I1, T2);
    __m512d P3 = _mm512_permutex2var_pd(T1, I1, T3);
    _mm512_storeu_pd(y + 4 * (size_t)p0,      _mm512_permutex2var_pd(P0, J0, P1));
    _mm512_storeu_pd(y + 4 * (size_t)p0 + 8,  _mm512_permutex2var_pd(P0, J1, P1));
    _mm512_storeu_pd(y + 4 * (size_t)p0 + 16, _mm512_permutex2var_pd(P2, J0, P3));
    _mm512_storeu_pd(y + 4 * (size_t)p0 + 24, _mm512_permutex2var_pd(P2, J1, P3));
}

/* radix-3 */
static void st3_scalar(int m, int s, const double *restrict wr, const double *restrict wi,
                const double *restrict xr, const double *restrict xi,
                double *restrict yr, double *restrict yi, int p0)
{
    const double s3 = 0.86602540378443864676;
    const double *restrict w1r = wr,     *restrict w1i = wi;
    const double *restrict w2r = wr + m, *restrict w2i = wi + m;
    for (int p = p0; p < m; ++p) {
        const double u1r = w1r[p], u1i = w1i[p];
        const double u2r = w2r[p], u2i = w2i[p];
        const double *restrict x0r = xr + (size_t)s * p, *restrict x0i = xi + (size_t)s * p;
        const double *restrict x1r = x0r + (size_t)s * m, *restrict x1i = x0i + (size_t)s * m;
        const double *restrict x2r = x1r + (size_t)s * m, *restrict x2i = x1i + (size_t)s * m;
        double *restrict o0r = yr + (size_t)s * 3 * p, *restrict o0i = yi + (size_t)s * 3 * p;
        double *restrict o1r = o0r + s, *restrict o1i = o0i + s;
        double *restrict o2r = o1r + s, *restrict o2i = o1i + s;
#pragma GCC ivdep
        for (int q = 0; q < s; ++q) {
            double ar = x0r[q], ai = x0i[q];
            double br = x1r[q], bi = x1i[q];
            double cr = x2r[q], ci = x2i[q];
            double tr = br + cr, ti = bi + ci;
            double ur = br - cr, ui = bi - ci;
            double mr = ar - 0.5 * tr, mi = ai - 0.5 * ti;
            double y1r = mr + s3 * ui, y1i = mi - s3 * ur;
            double y2r = mr - s3 * ui, y2i = mi + s3 * ur;
            o0r[q] = ar + tr;  o0i[q] = ai + ti;
            o1r[q] = y1r * u1r - y1i * u1i;  o1i[q] = y1r * u1i + y1i * u1r;
            o2r[q] = y2r * u2r - y2i * u2i;  o2i[q] = y2r * u2i + y2i * u2r;
        }
    }
}

/* radix-3, s == 4 vector path: two p-groups per zmm (8 contiguous doubles),
   twiddles pair-broadcast with one vpermpd, outputs stored as 256-bit halves
   at 12p + 4u.  gcc never vectorized the s=4 inner loop (measured scalar). */
static void st3_s4(int m, const double *restrict wr, const double *restrict wi,
                   const double *restrict xr, const double *restrict xi,
                   double *restrict yr, double *restrict yi)
{
    const __m512d S3 = _mm512_set1_pd(0.86602540378443864676);
    const __m512d H  = _mm512_set1_pd(0.5);
    const __m512i PAIR = _mm512_setr_epi64(0, 0, 0, 0, 1, 1, 1, 1);
    const double *restrict w1r = wr,     *restrict w1i = wi;
    const double *restrict w2r = wr + m, *restrict w2i = wi + m;
    const size_t sm = 4 * (size_t)m;
    int p = 0;
    for (; p + 2 <= m; p += 2) {
        size_t b = 4 * (size_t)p;
        __m512d ar = _mm512_loadu_pd(xr + b),          ai = _mm512_loadu_pd(xi + b);
        __m512d br = _mm512_loadu_pd(xr + b + sm),     bi = _mm512_loadu_pd(xi + b + sm);
        __m512d cr = _mm512_loadu_pd(xr + b + 2 * sm), ci = _mm512_loadu_pd(xi + b + 2 * sm);
        __m512d tr = _mm512_add_pd(br, cr), ti = _mm512_add_pd(bi, ci);
        __m512d ur = _mm512_sub_pd(br, cr), ui = _mm512_sub_pd(bi, ci);
        __m512d mr = _mm512_fnmadd_pd(H, tr, ar), mi = _mm512_fnmadd_pd(H, ti, ai);
        __m512d y1r = _mm512_fmadd_pd(S3, ui, mr), y1i = _mm512_fnmadd_pd(S3, ur, mi);
        __m512d y2r = _mm512_fnmadd_pd(S3, ui, mr), y2i = _mm512_fmadd_pd(S3, ur, mi);
        __m512d o0r = _mm512_add_pd(ar, tr), o0i = _mm512_add_pd(ai, ti);
        __m512d u1r = _mm512_permutexvar_pd(PAIR,
                          _mm512_castpd128_pd512(_mm_loadu_pd(w1r + p)));
        __m512d u1i = _mm512_permutexvar_pd(PAIR,
                          _mm512_castpd128_pd512(_mm_loadu_pd(w1i + p)));
        __m512d u2r = _mm512_permutexvar_pd(PAIR,
                          _mm512_castpd128_pd512(_mm_loadu_pd(w2r + p)));
        __m512d u2i = _mm512_permutexvar_pd(PAIR,
                          _mm512_castpd128_pd512(_mm_loadu_pd(w2i + p)));
        __m512d o1r = _mm512_fmsub_pd(y1r, u1r, _mm512_mul_pd(y1i, u1i));
        __m512d o1i = _mm512_fmadd_pd(y1r, u1i, _mm512_mul_pd(y1i, u1r));
        __m512d o2r = _mm512_fmsub_pd(y2r, u2r, _mm512_mul_pd(y2i, u2i));
        __m512d o2i = _mm512_fmadd_pd(y2r, u2i, _mm512_mul_pd(y2i, u2r));
        double *restrict outr = yr + 12 * (size_t)p;
        double *restrict outi = yi + 12 * (size_t)p;
        _mm256_storeu_pd(outr,      _mm512_castpd512_pd256(o0r));
        _mm256_storeu_pd(outr + 12, _mm512_extractf64x4_pd(o0r, 1));
        _mm256_storeu_pd(outr + 4,  _mm512_castpd512_pd256(o1r));
        _mm256_storeu_pd(outr + 16, _mm512_extractf64x4_pd(o1r, 1));
        _mm256_storeu_pd(outr + 8,  _mm512_castpd512_pd256(o2r));
        _mm256_storeu_pd(outr + 20, _mm512_extractf64x4_pd(o2r, 1));
        _mm256_storeu_pd(outi,      _mm512_castpd512_pd256(o0i));
        _mm256_storeu_pd(outi + 12, _mm512_extractf64x4_pd(o0i, 1));
        _mm256_storeu_pd(outi + 4,  _mm512_castpd512_pd256(o1i));
        _mm256_storeu_pd(outi + 16, _mm512_extractf64x4_pd(o1i, 1));
        _mm256_storeu_pd(outi + 8,  _mm512_castpd512_pd256(o2i));
        _mm256_storeu_pd(outi + 20, _mm512_extractf64x4_pd(o2i, 1));
    }
    if (p < m) st3_scalar(m, 4, wr, wi, xr, xi, yr, yi, p);
}

static void st3(int m, int s, const double *restrict wr, const double *restrict wi,
                const double *restrict xr, const double *restrict xi,
                double *restrict yr, double *restrict yi)
{
    if (s == 4) st3_s4(m, wr, wi, xr, xi, yr, yi);
    else        st3_scalar(m, s, wr, wi, xr, xi, yr, yi, 0);
}

/* radix-5, vector path: per p, q in masked 8-lane blocks; twiddles broadcast.
   Handles any s (1021 uses s=12: one full + one masked-4 block per p). */
static void st5_vec(int m, int s, const double *restrict wr, const double *restrict wi,
                    const double *restrict xr, const double *restrict xi,
                    double *restrict yr, double *restrict yi)
{
    const __m512d C1 = _mm512_set1_pd(0.30901699437494742410);
    const __m512d S1 = _mm512_set1_pd(0.95105651629515357212);
    const __m512d C2 = _mm512_set1_pd(-0.80901699437494742410);
    const __m512d S2 = _mm512_set1_pd(0.58778525229247312917);
    const size_t sm = (size_t)s * m;
    for (int p = 0; p < m; ++p) {
        const __m512d u1r = _mm512_set1_pd(wr[p]),         u1i = _mm512_set1_pd(wi[p]);
        const __m512d u2r = _mm512_set1_pd(wr[m + p]),     u2i = _mm512_set1_pd(wi[m + p]);
        const __m512d u3r = _mm512_set1_pd(wr[2 * m + p]), u3i = _mm512_set1_pd(wi[2 * m + p]);
        const __m512d u4r = _mm512_set1_pd(wr[3 * m + p]), u4i = _mm512_set1_pd(wi[3 * m + p]);
        const double *restrict x0r = xr + (size_t)s * p, *restrict x0i = xi + (size_t)s * p;
        double *restrict o0r = yr + (size_t)s * 5 * p, *restrict o0i = yi + (size_t)s * 5 * p;
        for (int q0 = 0; q0 < s; q0 += 8) {
            __mmask8 mk = (s - q0 >= 8) ? (__mmask8)0xFF
                                        : (__mmask8)((1u << (s - q0)) - 1);
            __m512d ar = _mm512_maskz_loadu_pd(mk, x0r + q0);
            __m512d ai = _mm512_maskz_loadu_pd(mk, x0i + q0);
            __m512d br = _mm512_maskz_loadu_pd(mk, x0r + q0 + sm);
            __m512d bi = _mm512_maskz_loadu_pd(mk, x0i + q0 + sm);
            __m512d cr = _mm512_maskz_loadu_pd(mk, x0r + q0 + 2 * sm);
            __m512d ci = _mm512_maskz_loadu_pd(mk, x0i + q0 + 2 * sm);
            __m512d dr = _mm512_maskz_loadu_pd(mk, x0r + q0 + 3 * sm);
            __m512d di = _mm512_maskz_loadu_pd(mk, x0i + q0 + 3 * sm);
            __m512d er = _mm512_maskz_loadu_pd(mk, x0r + q0 + 4 * sm);
            __m512d ei = _mm512_maskz_loadu_pd(mk, x0i + q0 + 4 * sm);
            __m512d t1r = _mm512_add_pd(br, er), t1i = _mm512_add_pd(bi, ei);
            __m512d t2r = _mm512_add_pd(cr, dr), t2i = _mm512_add_pd(ci, di);
            __m512d v1r = _mm512_sub_pd(br, er), v1i = _mm512_sub_pd(bi, ei);
            __m512d v2r = _mm512_sub_pd(cr, dr), v2i = _mm512_sub_pd(ci, di);
            __m512d m1r = _mm512_fmadd_pd(C1, t1r, _mm512_fmadd_pd(C2, t2r, ar));
            __m512d m1i = _mm512_fmadd_pd(C1, t1i, _mm512_fmadd_pd(C2, t2i, ai));
            __m512d m2r = _mm512_fmadd_pd(C2, t1r, _mm512_fmadd_pd(C1, t2r, ar));
            __m512d m2i = _mm512_fmadd_pd(C2, t1i, _mm512_fmadd_pd(C1, t2i, ai));
            __m512d n1r = _mm512_fmadd_pd(S1, v1r, _mm512_mul_pd(S2, v2r));
            __m512d n1i = _mm512_fmadd_pd(S1, v1i, _mm512_mul_pd(S2, v2i));
            __m512d n2r = _mm512_fmsub_pd(S2, v1r, _mm512_mul_pd(S1, v2r));
            __m512d n2i = _mm512_fmsub_pd(S2, v1i, _mm512_mul_pd(S1, v2i));
            __m512d y1r = _mm512_add_pd(m1r, n1i), y1i = _mm512_sub_pd(m1i, n1r);
            __m512d y4r = _mm512_sub_pd(m1r, n1i), y4i = _mm512_add_pd(m1i, n1r);
            __m512d y2r = _mm512_add_pd(m2r, n2i), y2i = _mm512_sub_pd(m2i, n2r);
            __m512d y3r = _mm512_sub_pd(m2r, n2i), y3i = _mm512_add_pd(m2i, n2r);
            _mm512_mask_storeu_pd(o0r + q0, mk,
                _mm512_add_pd(ar, _mm512_add_pd(t1r, t2r)));
            _mm512_mask_storeu_pd(o0i + q0, mk,
                _mm512_add_pd(ai, _mm512_add_pd(t1i, t2i)));
            _mm512_mask_storeu_pd(o0r + q0 + s, mk,
                _mm512_fmsub_pd(y1r, u1r, _mm512_mul_pd(y1i, u1i)));
            _mm512_mask_storeu_pd(o0i + q0 + s, mk,
                _mm512_fmadd_pd(y1r, u1i, _mm512_mul_pd(y1i, u1r)));
            _mm512_mask_storeu_pd(o0r + q0 + 2 * s, mk,
                _mm512_fmsub_pd(y2r, u2r, _mm512_mul_pd(y2i, u2i)));
            _mm512_mask_storeu_pd(o0i + q0 + 2 * s, mk,
                _mm512_fmadd_pd(y2r, u2i, _mm512_mul_pd(y2i, u2r)));
            _mm512_mask_storeu_pd(o0r + q0 + 3 * s, mk,
                _mm512_fmsub_pd(y3r, u3r, _mm512_mul_pd(y3i, u3i)));
            _mm512_mask_storeu_pd(o0i + q0 + 3 * s, mk,
                _mm512_fmadd_pd(y3r, u3i, _mm512_mul_pd(y3i, u3r)));
            _mm512_mask_storeu_pd(o0r + q0 + 4 * s, mk,
                _mm512_fmsub_pd(y4r, u4r, _mm512_mul_pd(y4i, u4i)));
            _mm512_mask_storeu_pd(o0i + q0 + 4 * s, mk,
                _mm512_fmadd_pd(y4r, u4i, _mm512_mul_pd(y4i, u4r)));
        }
    }
}

static void st5_small(int m, int s, const double *restrict wr, const double *restrict wi,
                const double *restrict xr, const double *restrict xi,
                double *restrict yr, double *restrict yi);

static void st5(int m, int s, const double *restrict wr, const double *restrict wi,
                const double *restrict xr, const double *restrict xi,
                double *restrict yr, double *restrict yi)
{
    if (s >= 4) { st5_vec(m, s, wr, wi, xr, xi, yr, yi); return; }
    st5_small(m, s, wr, wi, xr, xi, yr, yi);
}

/* radix-5, scalar (small s fallback) */
static void st5_small(int m, int s, const double *restrict wr, const double *restrict wi,
                const double *restrict xr, const double *restrict xi,
                double *restrict yr, double *restrict yi)
{
    const double c1 = 0.30901699437494742410, s1 = 0.95105651629515357212;
    const double c2 = -0.80901699437494742410, s2 = 0.58778525229247312917;
    for (int p = 0; p < m; ++p) {
        const double u1r_ = wr[p], u1i_ = wi[p];
        const double u2r_ = wr[m + p], u2i_ = wi[m + p];
        const double u3r_ = wr[2 * m + p], u3i_ = wi[2 * m + p];
        const double u4r_ = wr[3 * m + p], u4i_ = wi[3 * m + p];
        const double *restrict x0r = xr + (size_t)s * p, *restrict x0i = xi + (size_t)s * p;
        const double *restrict x1r = x0r + (size_t)s * m, *restrict x1i = x0i + (size_t)s * m;
        const double *restrict x2r = x1r + (size_t)s * m, *restrict x2i = x1i + (size_t)s * m;
        const double *restrict x3r = x2r + (size_t)s * m, *restrict x3i = x2i + (size_t)s * m;
        const double *restrict x4r = x3r + (size_t)s * m, *restrict x4i = x3i + (size_t)s * m;
        double *restrict o0r = yr + (size_t)s * 5 * p, *restrict o0i = yi + (size_t)s * 5 * p;
        double *restrict o1r = o0r + s, *restrict o1i = o0i + s;
        double *restrict o2r = o1r + s, *restrict o2i = o1i + s;
        double *restrict o3r = o2r + s, *restrict o3i = o2i + s;
        double *restrict o4r = o3r + s, *restrict o4i = o3i + s;
#pragma GCC ivdep
        for (int q = 0; q < s; ++q) {
            double ar = x0r[q], ai = x0i[q];
            double br = x1r[q], bi = x1i[q];
            double cr = x2r[q], ci = x2i[q];
            double dr = x3r[q], di = x3i[q];
            double er = x4r[q], ei = x4i[q];
            double t1r = br + er, t1i = bi + ei;
            double t2r = cr + dr, t2i = ci + di;
            double v1r = br - er, v1i = bi - ei;
            double v2r = cr - dr, v2i = ci - di;
            double m1r = ar + c1 * t1r + c2 * t2r, m1i = ai + c1 * t1i + c2 * t2i;
            double m2r = ar + c2 * t1r + c1 * t2r, m2i = ai + c2 * t1i + c1 * t2i;
            double n1r = s1 * v1r + s2 * v2r, n1i = s1 * v1i + s2 * v2i;
            double n2r = s2 * v1r - s1 * v2r, n2i = s2 * v1i - s1 * v2i;
            double y1r = m1r + n1i, y1i = m1i - n1r;
            double y4r = m1r - n1i, y4i = m1i + n1r;
            double y2r = m2r + n2i, y2i = m2i - n2r;
            double y3r = m2r - n2i, y3i = m2i + n2r;
            o0r[q] = ar + t1r + t2r;  o0i[q] = ai + t1i + t2i;
            o1r[q] = y1r * u1r_ - y1i * u1i_;  o1i[q] = y1r * u1i_ + y1i * u1r_;
            o2r[q] = y2r * u2r_ - y2i * u2i_;  o2i[q] = y2r * u2i_ + y2i * u2r_;
            o3r[q] = y3r * u3r_ - y3i * u3i_;  o3i[q] = y3r * u3i_ + y3i * u3r_;
            o4r[q] = y4r * u4r_ - y4i * u4i_;  o4i[q] = y4r * u4i_ + y4i * u4r_;
        }
    }
}

/* dense radix-17, LAST stage only (m = 1, unit twiddles): symmetric-pair
   real-coefficient 17-point DFT (the d1_prime fold, applied per stage lane):
   u_j = x_j + x_{17-j}, v_j = x_j - x_{17-j};
   X_k = x_0 + sum_j c[k][j] u_j - i sum_j s[k][j] v_j,  X_{17-k} the +i twin. */
static double c17t[8][8], s17t[8][8];
static int c17_ready = 0;

static void c17_init(void)
{
    if (c17_ready) return;
    for (int k = 1; k <= 8; ++k)
        for (int j = 1; j <= 8; ++j) {
            double ph = -2.0 * M_PI * (double)((j * k) % 17) / 17.0;
            c17t[k - 1][j - 1] = cos(ph);
            s17t[k - 1][j - 1] = -sin(ph);   /* s[k][j] = sin(2 pi jk / 17) */
        }
    c17_ready = 1;
}

/* the graded map, on one value */
#define CHAIN_MAP(re, im, outr, outi)                          \
    do {                                                       \
        double sc_ = 1.0 / (1.0 + sqrt((re) * (re) + (im) * (im))); \
        (outr) = (re) * sc_;                                   \
        (outi) = (im) * sc_;                                   \
    } while (0)

/* 8-wide graded map WITHOUT the divider: vsqrtpd+vdivpd zmm cost ~34 cycles
 * back-to-back on the single Ice Lake divide unit, and the chain exits run
 * one pair per 8 lanes.  rsqrt14 + 2 Newton (-> ~2^-56) gives sqrt(m) = m*r,
 * rcp14 + 2 Newton gives 1/(1+s); everything lands on the two FMA ports.
 * max(m, 1e-300) guards m = 0 (rsqrt(0) = inf would make m*inf = NaN). */
__attribute__((always_inline)) static inline void
chain_map8(__m512d re, __m512d im, __m512d *outr, __m512d *outi)
{
    const __m512d ONE = _mm512_set1_pd(1.0);
    const __m512d HALF3 = _mm512_set1_pd(1.5);
    const __m512d MHALF = _mm512_set1_pd(-0.5);
    __m512d m = _mm512_fmadd_pd(re, re, _mm512_mul_pd(im, im));
    __m512d mz = _mm512_max_pd(m, _mm512_set1_pd(1e-300));
    __m512d r = _mm512_rsqrt14_pd(mz);
    /* r <- r (1.5 - 0.5 m r^2), twice */
    __m512d h = _mm512_mul_pd(_mm512_mul_pd(mz, MHALF), _mm512_mul_pd(r, r));
    r = _mm512_mul_pd(r, _mm512_add_pd(HALF3, h));
    h = _mm512_mul_pd(_mm512_mul_pd(mz, MHALF), _mm512_mul_pd(r, r));
    r = _mm512_mul_pd(r, _mm512_add_pd(HALF3, h));
    __m512d d = _mm512_fmadd_pd(mz, r, ONE);          /* 1 + sqrt(m) */
    __m512d y = _mm512_rcp14_pd(d);
    /* y <- y (2 - d y), twice */
    y = _mm512_mul_pd(y, _mm512_fnmadd_pd(d, y, _mm512_set1_pd(2.0)));
    y = _mm512_mul_pd(y, _mm512_fnmadd_pd(d, y, _mm512_set1_pd(2.0)));
    *outr = _mm512_mul_pd(re, y);
    *outi = _mm512_mul_pd(im, y);
}

/* st17 rewritten (round d1_r2): the old single-loop form kept ur/ui/vr/vi --
 * 32 zmm-vectors of lane state -- live across the whole k-loop, which is the
 * entire AVX-512 register file; gcc spilled and the stage ran at ~10% of FMA
 * peak on the scoring Ice Lake (6.5 us of the 14.7 us 1021 transform, both
 * calls together).  New form: per 8-lane block, pass A folds u/v into a 2 KB
 * L1 tile, pass B loops k with only 4 accumulators + 2 broadcasts + 4 loads
 * live.  Store variants fuse the Rader exit (scatter+x0) and the chain map
 * into the inverse's last stage so the separate M-length exit pass vanishes. */
enum { ST17_PLAIN, ST17_SCATTER, ST17_CHAIN };

typedef struct {                 /* exit-fusion parameters, unused when PLAIN */
    const int *oidx;             /* SCATTER: conv index -> natural index */
    double x0r, x0i;             /* SCATTER: X[k] = x[0] + conv[k] */
    const double *cfr, *cfi;     /* CHAIN: permuted c-field, conv order */
    double s0r, s0i;             /* CHAIN: state[0] scalar */
    double *dr, *di;             /* CHAIN: conv-order state out */
    double *yd;                  /* SCATTER: interleaved natural-order out */
} st17_sink;

/* One masked 8-lane block, explicit AVX-512.  gcc 11 never vectorizes the
 * lane loop of the scalar form across the nested j/k loops (measured 3.3 GF/s
 * = scalar FMA throughput on the scoring Ice Lake), so the lanes are __m512d
 * by hand.  u-half and v-half run sequentially so live state stays ~22 regs.
 * mode is a compile-time constant at every call site. */
#define ST17_SINKSTORE(mode, snk, mk, idxbase, o_r, o_i)                       \
    do {                                                                       \
        if ((mode) == ST17_SCATTER) {                                          \
            /* planes swapped: yd[2o] = im-slot + x0r, yd[2o+1] = re-slot + x0i.  \
               Ice Lake vscatterdpd is microcoded (~5x cost of this): stage    \
               through the stack and store scalar. */                          \
            double tr_[8], ti_[8];                                             \
            _mm512_storeu_pd(tr_, _mm512_add_pd((o_i), _mm512_set1_pd((snk)->x0r))); \
            _mm512_storeu_pd(ti_, _mm512_add_pd((o_r), _mm512_set1_pd((snk)->x0i))); \
            const int *restrict ox_ = (snk)->oidx + (idxbase);                 \
            int nl_ = __builtin_popcount(mk);                                  \
            for (int l_ = 0; l_ < nl_; ++l_) {                                 \
                (snk)->yd[2 * ox_[l_]]     = tr_[l_];                          \
                (snk)->yd[2 * ox_[l_] + 1] = ti_[l_];                          \
            }                                                                  \
        } else {                                                               \
            __m512d re_ = _mm512_add_pd(_mm512_add_pd((o_i),                   \
                              _mm512_set1_pd((snk)->s0r)),                     \
                              _mm512_maskz_loadu_pd((mk), (snk)->cfr + (idxbase))); \
            __m512d im_ = _mm512_add_pd(_mm512_add_pd((o_r),                   \
                              _mm512_set1_pd((snk)->s0i)),                     \
                              _mm512_maskz_loadu_pd((mk), (snk)->cfi + (idxbase))); \
            __m512d mr_, mi_;                                                  \
            chain_map8(re_, im_, &mr_, &mi_);                                  \
            _mm512_mask_storeu_pd((snk)->dr + (idxbase), (mk), mr_);           \
            _mm512_mask_storeu_pd((snk)->di + (idxbase), (mk), mi_);           \
        }                                                                      \
    } while (0)

__attribute__((always_inline))
static inline void st17_vblock(int s, int q0, __mmask8 mk, int mode, const st17_sink *snk,
                               const double *restrict xr, const double *restrict xi,
                               double *restrict yr, double *restrict yi)
{
    const double *restrict xrq = xr + q0, *restrict xiq = xi + q0;
    __m512d ur[8], ui[8];
    __m512d x0r = _mm512_maskz_loadu_pd(mk, xrq);
    __m512d x0i = _mm512_maskz_loadu_pd(mk, xiq);
    __m512d sur = x0r, sui = x0i;
#pragma GCC unroll 8
    for (int j = 1; j <= 8; ++j) {
        __m512d ar = _mm512_maskz_loadu_pd(mk, xrq + (size_t)s * j);
        __m512d ai = _mm512_maskz_loadu_pd(mk, xiq + (size_t)s * j);
        __m512d br = _mm512_maskz_loadu_pd(mk, xrq + (size_t)s * (17 - j));
        __m512d bi = _mm512_maskz_loadu_pd(mk, xiq + (size_t)s * (17 - j));
        ur[j - 1] = _mm512_add_pd(ar, br);
        ui[j - 1] = _mm512_add_pd(ai, bi);
        sur = _mm512_add_pd(sur, ur[j - 1]);
        sui = _mm512_add_pd(sui, ui[j - 1]);
    }
    if (mode == ST17_PLAIN) {
        _mm512_mask_storeu_pd(yr + q0, mk, sur);
        _mm512_mask_storeu_pd(yi + q0, mk, sui);
    } else {
        ST17_SINKSTORE(mode, snk, mk, q0, sur, sui);
    }
    /* A_k = x0 + sum_j c[k][j] u_j  -> parked in BOTH twin slots of y */
#pragma GCC unroll 8
    for (int k = 1; k <= 8; ++k) {
        __m512d Ar = x0r, Ai = x0i;
#pragma GCC unroll 8
        for (int j = 0; j < 8; ++j) {
            __m512d cc = _mm512_set1_pd(c17t[k - 1][j]);
            Ar = _mm512_fmadd_pd(cc, ur[j], Ar);
            Ai = _mm512_fmadd_pd(cc, ui[j], Ai);
        }
        _mm512_mask_storeu_pd(yr + q0 + (size_t)s * k, mk, Ar);
        _mm512_mask_storeu_pd(yi + q0 + (size_t)s * k, mk, Ai);
        _mm512_mask_storeu_pd(yr + q0 + (size_t)s * (17 - k), mk, Ar);
        _mm512_mask_storeu_pd(yi + q0 + (size_t)s * (17 - k), mk, Ai);
    }
    /* v-half: B_k = sum_j s[k][j] v_j; X_k = A - iB, X_{17-k} = A + iB */
    __m512d vr[8], vi[8];
#pragma GCC unroll 8
    for (int j = 1; j <= 8; ++j) {
        vr[j - 1] = _mm512_sub_pd(_mm512_maskz_loadu_pd(mk, xrq + (size_t)s * j),
                                  _mm512_maskz_loadu_pd(mk, xrq + (size_t)s * (17 - j)));
        vi[j - 1] = _mm512_sub_pd(_mm512_maskz_loadu_pd(mk, xiq + (size_t)s * j),
                                  _mm512_maskz_loadu_pd(mk, xiq + (size_t)s * (17 - j)));
    }
#pragma GCC unroll 8
    for (int k = 1; k <= 8; ++k) {
        __m512d Br = _mm512_setzero_pd(), Bi = _mm512_setzero_pd();
#pragma GCC unroll 8
        for (int j = 0; j < 8; ++j) {
            __m512d ss = _mm512_set1_pd(s17t[k - 1][j]);
            Br = _mm512_fmadd_pd(ss, vr[j], Br);
            Bi = _mm512_fmadd_pd(ss, vi[j], Bi);
        }
        size_t i1 = q0 + (size_t)s * k, i2 = q0 + (size_t)s * (17 - k);
        __m512d Ar = _mm512_maskz_loadu_pd(mk, yr + i1);
        __m512d Ai = _mm512_maskz_loadu_pd(mk, yi + i1);
        __m512d o1r = _mm512_add_pd(Ar, Bi), o1i = _mm512_sub_pd(Ai, Br);
        __m512d o2r = _mm512_sub_pd(Ar, Bi), o2i = _mm512_add_pd(Ai, Br);
        if (mode == ST17_PLAIN) {
            _mm512_mask_storeu_pd(yr + i1, mk, o1r);
            _mm512_mask_storeu_pd(yi + i1, mk, o1i);
            _mm512_mask_storeu_pd(yr + i2, mk, o2r);
            _mm512_mask_storeu_pd(yi + i2, mk, o2i);
        } else {
            ST17_SINKSTORE(mode, snk, mk, i1, o1r, o1i);
            ST17_SINKSTORE(mode, snk, mk, i2, o2r, o2i);
        }
    }
}

__attribute__((always_inline))
static inline void st17_body(int s, int mode, const st17_sink *snk,
                             const double *restrict xr, const double *restrict xi,
                             double *restrict yr, double *restrict yi)
{
    int q = 0;
    for (; q + 8 <= s; q += 8)
        st17_vblock(s, q, (__mmask8)0xFF, mode, snk, xr, xi, yr, yi);
    if (q < s)
        st17_vblock(s, q, (__mmask8)((1u << (s - q)) - 1), mode, snk, xr, xi, yr, yi);
}

/* three HOT instantiations so `mode` is a compile-time constant in each */
static void st17(int s, const double *restrict xr, const double *restrict xi,
                     double *restrict yr, double *restrict yi)
{
    st17_body(s, ST17_PLAIN, NULL, xr, xi, yr, yi);
}

static void st17_scatter(int s, const st17_sink *snk,
                         const double *restrict xr, const double *restrict xi,
                         double *restrict tr, double *restrict ti)
{
    st17_body(s, ST17_SCATTER, snk, xr, xi, tr, ti);
}

static void st17_chain(int s, const st17_sink *snk,
                           const double *restrict xr, const double *restrict xi,
                           double *restrict tr, double *restrict ti)
{
    st17_body(s, ST17_CHAIN, snk, xr, xi, tr, ti);
}

static void core_exec_range(const core_plan *c, int st0, int st1, int n, int s,
                            double *ar, double *ai, double *br, double *bi,
                            double **outr, double **outi)
{
    double *xr = ar, *xi = ai, *yr = br, *yi = bi, *t;
    for (int st = st0; st < st1; ++st) {
        int r = c->radix[st], m = n / r;
        switch (r) {
        case 2: st2(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break;
        case 3: st3(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break;
        case 4: st4(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break;
        case 8: st8(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break;
        case 16: st16(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break;
        case 17: st17(s, xr, xi, yr, yi); break;   /* last stage only, m == 1 */
        default: st5(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break;
        }
        n = m; s *= r;
        t = xr; xr = yr; yr = t;
        t = xi; xi = yi; yi = t;
    }
    *outr = xr; *outi = xi;
}

static void core_exec(const core_plan *c, double *ar, double *ai,
                      double *br, double *bi, double **outr, double **outi)
{
    core_exec_range(c, 0, c->nstage, c->n, 1, ar, ai, br, bi, outr, outi);
}

/* ---------------- Rader fused entry / exit stages ----------------
 *
 * Entry: a[t] = x[iidx[t]] gathered straight into the stage-0 butterfly.
 * Padded conv (M >= 2P-1, pow2): a[t] = 0 for t >= P and P <= M/2, so radix-4
 * blocks u=2,3 vanish and block 1 is partial -- same degenerate butterfly
 * as d1_bluestein's zero-pruned entry.
 * Exit: last-stage butterfly (twiddles = 1) on the swapped planes, unswapped
 * at the store, + x[0], scattered to y[oidx[k]]. */

/* radix-4 gather entry, vector path: index streams as vgatherdpd (index 2*i,
   scale 8; +1 base for the imaginary plane), butterfly and twiddles 8-wide,
   outputs re-interleaved to y[4p+u] with the shuffle helper. */
static void st4_gather_full(int m, const double *restrict wr, const double *restrict wi,
        const double *restrict xd, const int *restrict idx,
        double *restrict yr, double *restrict yi)
{
    const double *restrict w1r = wr,         *restrict w1i = wi;
    const double *restrict w2r = wr + m,     *restrict w2i = wi + m;
    const double *restrict w3r = wr + 2 * m, *restrict w3i = wi + 2 * m;
    const __m512i EVN = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i ODD = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    /* 8 random complex points -> split re/im vectors: 128-bit pair loads
       assembled with vinsertf64x2, then one parity permute per plane.
       Beats vgatherdpd (microcoded on Ice Lake) for L1/L2-resident input. */
#define GATHER8(ix, vre, vim)                                                  \
    do {                                                                       \
        __m512d g0_ = _mm512_castpd128_pd512(_mm_loadu_pd(xd + 2 * (ix)[0]));  \
        g0_ = _mm512_insertf64x2(g0_, _mm_loadu_pd(xd + 2 * (ix)[1]), 1);      \
        g0_ = _mm512_insertf64x2(g0_, _mm_loadu_pd(xd + 2 * (ix)[2]), 2);      \
        g0_ = _mm512_insertf64x2(g0_, _mm_loadu_pd(xd + 2 * (ix)[3]), 3);      \
        __m512d g1_ = _mm512_castpd128_pd512(_mm_loadu_pd(xd + 2 * (ix)[4]));  \
        g1_ = _mm512_insertf64x2(g1_, _mm_loadu_pd(xd + 2 * (ix)[5]), 1);      \
        g1_ = _mm512_insertf64x2(g1_, _mm_loadu_pd(xd + 2 * (ix)[6]), 2);      \
        g1_ = _mm512_insertf64x2(g1_, _mm_loadu_pd(xd + 2 * (ix)[7]), 3);      \
        (vre) = _mm512_permutex2var_pd(g0_, EVN, g1_);                         \
        (vim) = _mm512_permutex2var_pd(g0_, ODD, g1_);                         \
    } while (0)
    int p = 0;
    for (; p + 8 <= m; p += 8) {
        __m512d ar, ai, br, bi, cr, ci, dr, di;
        GATHER8(idx + p,         ar, ai);
        GATHER8(idx + p + m,     br, bi);
        GATHER8(idx + p + 2 * m, cr, ci);
        GATHER8(idx + p + 3 * m, dr, di);
        __m512d t0r = _mm512_add_pd(ar, cr), t0i = _mm512_add_pd(ai, ci);
        __m512d t1r = _mm512_sub_pd(ar, cr), t1i = _mm512_sub_pd(ai, ci);
        __m512d t2r = _mm512_add_pd(br, dr), t2i = _mm512_add_pd(bi, di);
        __m512d t3r = _mm512_sub_pd(br, dr), t3i = _mm512_sub_pd(bi, di);
        __m512d y1r = _mm512_add_pd(t1r, t3i), y1i = _mm512_sub_pd(t1i, t3r);
        __m512d y2r = _mm512_sub_pd(t0r, t2r), y2i = _mm512_sub_pd(t0i, t2i);
        __m512d y3r = _mm512_sub_pd(t1r, t3i), y3i = _mm512_add_pd(t1i, t3r);
        __m512d v1r = _mm512_loadu_pd(w1r + p), v1i = _mm512_loadu_pd(w1i + p);
        __m512d v2r = _mm512_loadu_pd(w2r + p), v2i = _mm512_loadu_pd(w2i + p);
        __m512d v3r = _mm512_loadu_pd(w3r + p), v3i = _mm512_loadu_pd(w3i + p);
        __m512d X0r = _mm512_add_pd(t0r, t2r);
        __m512d X0i = _mm512_add_pd(t0i, t2i);
        __m512d X1r = _mm512_fmsub_pd(y1r, v1r, _mm512_mul_pd(y1i, v1i));
        __m512d X1i = _mm512_fmadd_pd(y1r, v1i, _mm512_mul_pd(y1i, v1r));
        __m512d X2r = _mm512_fmsub_pd(y2r, v2r, _mm512_mul_pd(y2i, v2i));
        __m512d X2i = _mm512_fmadd_pd(y2r, v2i, _mm512_mul_pd(y2i, v2r));
        __m512d X3r = _mm512_fmsub_pd(y3r, v3r, _mm512_mul_pd(y3i, v3i));
        __m512d X3i = _mm512_fmadd_pd(y3r, v3i, _mm512_mul_pd(y3i, v3r));
        interleave4_store(yr, p, X0r, X1r, X2r, X3r);
        interleave4_store(yi, p, X0i, X1i, X2i, X3i);
    }
#pragma GCC ivdep
    for (; p < m; ++p) {
        int ia = idx[p], ib = idx[p + m], ic = idx[p + 2 * m], id = idx[p + 3 * m];
        double ar = xd[2 * ia], ai = xd[2 * ia + 1];
        double br = xd[2 * ib], bi = xd[2 * ib + 1];
        double cr = xd[2 * ic], ci = xd[2 * ic + 1];
        double dr = xd[2 * id], di = xd[2 * id + 1];
        double t0r = ar + cr, t0i = ai + ci, t1r = ar - cr, t1i = ai - ci;
        double t2r = br + dr, t2i = bi + di, t3r = br - dr, t3i = bi - di;
        double y1r = t1r + t3i, y1i = t1i - t3r;
        double y2r = t0r - t2r, y2i = t0i - t2i;
        double y3r = t1r - t3i, y3i = t1i + t3r;
        yr[4 * p] = t0r + t2r;  yi[4 * p] = t0i + t2i;
        yr[4 * p + 1] = y1r * w1r[p] - y1i * w1i[p];
        yi[4 * p + 1] = y1r * w1i[p] + y1i * w1r[p];
        yr[4 * p + 2] = y2r * w2r[p] - y2i * w2i[p];
        yi[4 * p + 2] = y2r * w2i[p] + y2i * w2r[p];
        yr[4 * p + 3] = y3r * w3r[p] - y3i * w3i[p];
        yi[4 * p + 3] = y3r * w3i[p] + y3i * w3r[p];
    }
}

static void st4_gather_pruned(int m, int P,
        const double *restrict wr, const double *restrict wi,
        const double *restrict xd, const int *restrict idx,
        double *restrict yr, double *restrict yi)
{
    const double *restrict w1r = wr,         *restrict w1i = wi;
    const double *restrict w2r = wr + m,     *restrict w2i = wi + m;
    const double *restrict w3r = wr + 2 * m, *restrict w3i = wi + 2 * m;
    const int nb = P - m;    /* block 1 has nb live entries; blocks 2,3 zero */
#pragma GCC ivdep
    for (int p = 0; p < nb; ++p) {
        int ia = idx[p], ib = idx[p + m];
        double a_r = xd[2 * ia], a_i = xd[2 * ia + 1];
        double b_r = xd[2 * ib], b_i = xd[2 * ib + 1];
        double y1r = a_r + b_i, y1i = a_i - b_r;
        double y2r = a_r - b_r, y2i = a_i - b_i;
        double y3r = a_r - b_i, y3i = a_i + b_r;
        yr[4 * p] = a_r + b_r;  yi[4 * p] = a_i + b_i;
        yr[4 * p + 1] = y1r * w1r[p] - y1i * w1i[p];
        yi[4 * p + 1] = y1r * w1i[p] + y1i * w1r[p];
        yr[4 * p + 2] = y2r * w2r[p] - y2i * w2i[p];
        yi[4 * p + 2] = y2r * w2i[p] + y2i * w2r[p];
        yr[4 * p + 3] = y3r * w3r[p] - y3i * w3i[p];
        yi[4 * p + 3] = y3r * w3i[p] + y3i * w3r[p];
    }
#pragma GCC ivdep
    for (int p = nb; p < m; ++p) {
        int ia = idx[p];
        double a_r = xd[2 * ia], a_i = xd[2 * ia + 1];
        yr[4 * p] = a_r;  yi[4 * p] = a_i;
        yr[4 * p + 1] = a_r * w1r[p] - a_i * w1i[p];
        yi[4 * p + 1] = a_r * w1i[p] + a_i * w1r[p];
        yr[4 * p + 2] = a_r * w2r[p] - a_i * w2i[p];
        yi[4 * p + 2] = a_r * w2i[p] + a_i * w2r[p];
        yr[4 * p + 3] = a_r * w3r[p] - a_i * w3i[p];
        yi[4 * p + 3] = a_r * w3i[p] + a_i * w3r[p];
    }
}

static void st2_gather_full(int m, const double *restrict wr, const double *restrict wi,
        const double *restrict xd, const int *restrict idx,
        double *restrict yr, double *restrict yi)
{
#pragma GCC ivdep
    for (int p = 0; p < m; ++p) {
        int ia = idx[p], ib = idx[p + m];
        double ar = xd[2 * ia], ai = xd[2 * ia + 1];
        double br = xd[2 * ib], bi = xd[2 * ib + 1];
        double ur = ar - br, ui = ai - bi;
        yr[2 * p] = ar + br;  yi[2 * p] = ai + bi;
        yr[2 * p + 1] = ur * wr[p] - ui * wi[p];
        yi[2 * p + 1] = ur * wi[p] + ui * wr[p];
    }
}

/* inverse entry: multiply forward spectrum R by kernel spectrum B (1/M folded
   in), swap planes (inverse-as-forward), do the stage-0 butterfly. */
static void st4_first_bhat(int m,
        const double *restrict wr, const double *restrict wi,
        const double *restrict Rr, const double *restrict Ri,
        const double *restrict br, const double *restrict bi,
        double *restrict yr, double *restrict yi)
{
    const double *restrict w1r = wr,         *restrict w1i = wi;
    const double *restrict w2r = wr + m,     *restrict w2i = wi + m;
    const double *restrict w3r = wr + 2 * m, *restrict w3i = wi + 2 * m;
    int p = 0;
    for (; p + 8 <= m; p += 8) {
        __m512d ar, ai, brr, bii, cr, ci, dr, di;
        {   /* swap(R*B) per block: re' = Rr*bi + Ri*br, im' = Rr*br - Ri*bi */
            __m512d R0r = _mm512_loadu_pd(Rr + p),         R0i = _mm512_loadu_pd(Ri + p);
            __m512d K0r = _mm512_loadu_pd(br + p),         K0i = _mm512_loadu_pd(bi + p);
            ar  = _mm512_fmadd_pd(R0r, K0i, _mm512_mul_pd(R0i, K0r));
            ai  = _mm512_fmsub_pd(R0r, K0r, _mm512_mul_pd(R0i, K0i));
            __m512d R1r = _mm512_loadu_pd(Rr + p + m),     R1i = _mm512_loadu_pd(Ri + p + m);
            __m512d K1r = _mm512_loadu_pd(br + p + m),     K1i = _mm512_loadu_pd(bi + p + m);
            brr = _mm512_fmadd_pd(R1r, K1i, _mm512_mul_pd(R1i, K1r));
            bii = _mm512_fmsub_pd(R1r, K1r, _mm512_mul_pd(R1i, K1i));
            __m512d R2r = _mm512_loadu_pd(Rr + p + 2 * m), R2i = _mm512_loadu_pd(Ri + p + 2 * m);
            __m512d K2r = _mm512_loadu_pd(br + p + 2 * m), K2i = _mm512_loadu_pd(bi + p + 2 * m);
            cr  = _mm512_fmadd_pd(R2r, K2i, _mm512_mul_pd(R2i, K2r));
            ci  = _mm512_fmsub_pd(R2r, K2r, _mm512_mul_pd(R2i, K2i));
            __m512d R3r = _mm512_loadu_pd(Rr + p + 3 * m), R3i = _mm512_loadu_pd(Ri + p + 3 * m);
            __m512d K3r = _mm512_loadu_pd(br + p + 3 * m), K3i = _mm512_loadu_pd(bi + p + 3 * m);
            dr  = _mm512_fmadd_pd(R3r, K3i, _mm512_mul_pd(R3i, K3r));
            di  = _mm512_fmsub_pd(R3r, K3r, _mm512_mul_pd(R3i, K3i));
        }
        __m512d t0r = _mm512_add_pd(ar, cr),  t0i = _mm512_add_pd(ai, ci);
        __m512d t1r = _mm512_sub_pd(ar, cr),  t1i = _mm512_sub_pd(ai, ci);
        __m512d t2r = _mm512_add_pd(brr, dr), t2i = _mm512_add_pd(bii, di);
        __m512d t3r = _mm512_sub_pd(brr, dr), t3i = _mm512_sub_pd(bii, di);
        __m512d y1r = _mm512_add_pd(t1r, t3i), y1i = _mm512_sub_pd(t1i, t3r);
        __m512d y2r = _mm512_sub_pd(t0r, t2r), y2i = _mm512_sub_pd(t0i, t2i);
        __m512d y3r = _mm512_sub_pd(t1r, t3i), y3i = _mm512_add_pd(t1i, t3r);
        __m512d v1r = _mm512_loadu_pd(w1r + p), v1i = _mm512_loadu_pd(w1i + p);
        __m512d v2r = _mm512_loadu_pd(w2r + p), v2i = _mm512_loadu_pd(w2i + p);
        __m512d v3r = _mm512_loadu_pd(w3r + p), v3i = _mm512_loadu_pd(w3i + p);
        __m512d X0r = _mm512_add_pd(t0r, t2r);
        __m512d X0i = _mm512_add_pd(t0i, t2i);
        __m512d X1r = _mm512_fmsub_pd(y1r, v1r, _mm512_mul_pd(y1i, v1i));
        __m512d X1i = _mm512_fmadd_pd(y1r, v1i, _mm512_mul_pd(y1i, v1r));
        __m512d X2r = _mm512_fmsub_pd(y2r, v2r, _mm512_mul_pd(y2i, v2i));
        __m512d X2i = _mm512_fmadd_pd(y2r, v2i, _mm512_mul_pd(y2i, v2r));
        __m512d X3r = _mm512_fmsub_pd(y3r, v3r, _mm512_mul_pd(y3i, v3i));
        __m512d X3i = _mm512_fmadd_pd(y3r, v3i, _mm512_mul_pd(y3i, v3r));
        interleave4_store(yr, p, X0r, X1r, X2r, X3r);
        interleave4_store(yi, p, X0i, X1i, X2i, X3i);
    }
#pragma GCC ivdep
    for (; p < m; ++p) {
        int t1_ = p + m, t2_ = p + 2 * m, t3_ = p + 3 * m;
        double ar = Rr[p] * bi[p] + Ri[p] * br[p];
        double ai = Rr[p] * br[p] - Ri[p] * bi[p];
        double brr = Rr[t1_] * bi[t1_] + Ri[t1_] * br[t1_];
        double bii = Rr[t1_] * br[t1_] - Ri[t1_] * bi[t1_];
        double cr = Rr[t2_] * bi[t2_] + Ri[t2_] * br[t2_];
        double ci = Rr[t2_] * br[t2_] - Ri[t2_] * bi[t2_];
        double dr = Rr[t3_] * bi[t3_] + Ri[t3_] * br[t3_];
        double di = Rr[t3_] * br[t3_] - Ri[t3_] * bi[t3_];
        double t0r = ar + cr, t0i = ai + ci, t1r = ar - cr, t1i = ai - ci;
        double t2r = brr + dr, t2i = bii + di, t3r = brr - dr, t3i = bii - di;
        double y1r = t1r + t3i, y1i = t1i - t3r;
        double y2r = t0r - t2r, y2i = t0i - t2i;
        double y3r = t1r - t3i, y3i = t1i + t3r;
        yr[4 * p] = t0r + t2r;  yi[4 * p] = t0i + t2i;
        yr[4 * p + 1] = y1r * w1r[p] - y1i * w1i[p];
        yi[4 * p + 1] = y1r * w1i[p] + y1i * w1r[p];
        yr[4 * p + 2] = y2r * w2r[p] - y2i * w2i[p];
        yi[4 * p + 2] = y2r * w2i[p] + y2i * w2r[p];
        yr[4 * p + 3] = y3r * w3r[p] - y3i * w3i[p];
        yi[4 * p + 3] = y3r * w3i[p] + y3i * w3r[p];
    }
}

static void st2_first_bhat(int m,
        const double *restrict wr, const double *restrict wi,
        const double *restrict Rr, const double *restrict Ri,
        const double *restrict br, const double *restrict bi,
        double *restrict yr, double *restrict yi)
{
#pragma GCC ivdep
    for (int p = 0; p < m; ++p) {
        int t1_ = p + m;
        double ar = Rr[p] * bi[p] + Ri[p] * br[p];
        double ai = Rr[p] * br[p] - Ri[p] * bi[p];
        double brr = Rr[t1_] * bi[t1_] + Ri[t1_] * br[t1_];
        double bii = Rr[t1_] * br[t1_] - Ri[t1_] * bi[t1_];
        double ur = ar - brr, ui = ai - bii;
        yr[2 * p] = ar + brr;  yi[2 * p] = ai + bii;
        yr[2 * p + 1] = ur * wr[p] - ui * wi[p];
        yi[2 * p + 1] = ur * wi[p] + ui * wr[p];
    }
}

/* exit stages: planes swapped (xr = W_imag, xi = W_real), last-stage
   twiddles are 1; unswap + add x0 + scatter to y[oidx[k]]. */
static void st4_last_scatter(int s, const double *restrict xr, const double *restrict xi,
        const int *restrict oidx, double x0r, double x0i, double *restrict yd)
{
#pragma GCC ivdep
    for (int q = 0; q < s; ++q) {
        double a_r = xr[q],         a_i = xi[q];
        double b_r = xr[q + s],     b_i = xi[q + s];
        double c_r = xr[q + 2 * s], c_i = xi[q + 2 * s];
        double d_r = xr[q + 3 * s], d_i = xi[q + 3 * s];
        double t0r = a_r + c_r, t0i = a_i + c_i, t1r = a_r - c_r, t1i = a_i - c_i;
        double t2r = b_r + d_r, t2i = b_i + d_i, t3r = b_r - d_r, t3i = b_i - d_i;
        /* forward butterfly on swapped planes; unswap at the store */
        double z0R = t0r + t2r, z0I = t0i + t2i;
        double z1R = t1r + t3i, z1I = t1i - t3r;
        double z2R = t0r - t2r, z2I = t0i - t2i;
        double z3R = t1r - t3i, z3I = t1i + t3r;
        int o0 = oidx[q], o1 = oidx[q + s], o2 = oidx[q + 2 * s], o3 = oidx[q + 3 * s];
        yd[2 * o0] = z0I + x0r;  yd[2 * o0 + 1] = z0R + x0i;
        yd[2 * o1] = z1I + x0r;  yd[2 * o1 + 1] = z1R + x0i;
        yd[2 * o2] = z2I + x0r;  yd[2 * o2 + 1] = z2R + x0i;
        yd[2 * o3] = z3I + x0r;  yd[2 * o3 + 1] = z3R + x0i;
    }
}

/* radix-2 exit, pruned: only outputs k < P are real conv outputs (P <= s) */
static void st2_last_scatter(int s, int P, const double *restrict xr, const double *restrict xi,
        const int *restrict oidx, double x0r, double x0i, double *restrict yd)
{
#pragma GCC ivdep
    for (int q = 0; q < P; ++q) {
        double w_i = xr[q] + xr[q + s];
        double w_r = xi[q] + xi[q + s];
        int o = oidx[q];
        yd[2 * o] = w_r + x0r;  yd[2 * o + 1] = w_i + x0i;
    }
}

/* ---------------- chain-mode entry / exit stages ----------------
 *
 * For the fused map chain the state lives in CONV-OUTPUT order between steps:
 * d[k] = state[oidx[k]].  Because iidx[q] = g^q = g^-(P-q) = oidx[(P-q) mod P],
 * the next step's gather a[q] = state[iidx[q]] is just d[(P-q) mod P] -- a
 * sequential index REVERSAL instead of a random gather, and the elementwise
 * map commutes with the permutation.  The map itself fuses into the exit
 * stage; only the first step reads the natural-order input and only a final
 * P-sized pass materializes the natural-order interleaved state. */

static void st4_rev_full(int m, int P,
        const double *restrict wr, const double *restrict wi,
        const double *restrict dr, const double *restrict di,
        double *restrict yr, double *restrict yi)
{
    const double *restrict w1r = wr,         *restrict w1i = wi;
    const double *restrict w2r = wr + m,     *restrict w2i = wi + m;
    const double *restrict w3r = wr + 2 * m, *restrict w3i = wi + 2 * m;
#pragma GCC ivdep
    for (int p = 0; p < m; ++p) {
        /* a[t] = t ? d[P-t] : d[0], t = p + m*u */
        int ta = p ? P - p : 0;
        double ar = dr[ta],           ai = di[ta];
        double br = dr[P - (p + m)],     bi = di[P - (p + m)];
        double cr = dr[P - (p + 2 * m)], ci = di[P - (p + 2 * m)];
        double dr_ = dr[P - (p + 3 * m)], di_ = di[P - (p + 3 * m)];
        double t0r = ar + cr, t0i = ai + ci, t1r = ar - cr, t1i = ai - ci;
        double t2r = br + dr_, t2i = bi + di_, t3r = br - dr_, t3i = bi - di_;
        double y1r = t1r + t3i, y1i = t1i - t3r;
        double y2r = t0r - t2r, y2i = t0i - t2i;
        double y3r = t1r - t3i, y3i = t1i + t3r;
        yr[4 * p] = t0r + t2r;  yi[4 * p] = t0i + t2i;
        yr[4 * p + 1] = y1r * w1r[p] - y1i * w1i[p];
        yi[4 * p + 1] = y1r * w1i[p] + y1i * w1r[p];
        yr[4 * p + 2] = y2r * w2r[p] - y2i * w2i[p];
        yi[4 * p + 2] = y2r * w2i[p] + y2i * w2r[p];
        yr[4 * p + 3] = y3r * w3r[p] - y3i * w3i[p];
        yi[4 * p + 3] = y3r * w3i[p] + y3i * w3r[p];
    }
}

static void st4_rev_pruned(int m, int P,
        const double *restrict wr, const double *restrict wi,
        const double *restrict dr, const double *restrict di,
        double *restrict yr, double *restrict yi)
{
    const double *restrict w1r = wr,         *restrict w1i = wi;
    const double *restrict w2r = wr + m,     *restrict w2i = wi + m;
    const double *restrict w3r = wr + 2 * m, *restrict w3i = wi + 2 * m;
    const int nb = P - m;
#pragma GCC ivdep
    for (int p = 0; p < nb; ++p) {
        int ta = p ? P - p : 0;
        double a_r = dr[ta], a_i = di[ta];
        double b_r = dr[P - (p + m)], b_i = di[P - (p + m)];
        double y1r = a_r + b_i, y1i = a_i - b_r;
        double y2r = a_r - b_r, y2i = a_i - b_i;
        double y3r = a_r - b_i, y3i = a_i + b_r;
        yr[4 * p] = a_r + b_r;  yi[4 * p] = a_i + b_i;
        yr[4 * p + 1] = y1r * w1r[p] - y1i * w1i[p];
        yi[4 * p + 1] = y1r * w1i[p] + y1i * w1r[p];
        yr[4 * p + 2] = y2r * w2r[p] - y2i * w2i[p];
        yi[4 * p + 2] = y2r * w2i[p] + y2i * w2r[p];
        yr[4 * p + 3] = y3r * w3r[p] - y3i * w3i[p];
        yi[4 * p + 3] = y3r * w3i[p] + y3i * w3r[p];
    }
#pragma GCC ivdep
    for (int p = nb; p < m; ++p) {
        int ta = p ? P - p : 0;
        double a_r = dr[ta], a_i = di[ta];
        yr[4 * p] = a_r;  yi[4 * p] = a_i;
        yr[4 * p + 1] = a_r * w1r[p] - a_i * w1i[p];
        yi[4 * p + 1] = a_r * w1i[p] + a_i * w1r[p];
        yr[4 * p + 2] = a_r * w2r[p] - a_i * w2i[p];
        yi[4 * p + 2] = a_r * w2i[p] + a_i * w2r[p];
        yr[4 * p + 3] = a_r * w3r[p] - a_i * w3i[p];
        yi[4 * p + 3] = a_r * w3i[p] + a_i * w3r[p];
    }
}

static void st2_rev_full(int m, int P,
        const double *restrict wr, const double *restrict wi,
        const double *restrict dr, const double *restrict di,
        double *restrict yr, double *restrict yi)
{
#pragma GCC ivdep
    for (int p = 0; p < m; ++p) {
        int ta = p ? P - p : 0;
        double ar = dr[ta], ai = di[ta];
        double br = dr[P - (p + m)], bi = di[P - (p + m)];
        double ur = ar - br, ui = ai - bi;
        yr[2 * p] = ar + br;  yi[2 * p] = ai + bi;
        yr[2 * p + 1] = ur * wr[p] - ui * wi[p];
        yi[2 * p + 1] = ur * wi[p] + ui * wr[p];
    }
}

/* exit + map, split sequential output d[k] (planes swapped on input).
 * 8-wide with the Newton map: s is a multiple of 8 for every plan that
 * fuses radix-4 last (65536: s = 16384). */
static void st4_last_chain(int s, const double *restrict xr, const double *restrict xi,
        const double *restrict cfr, const double *restrict cfi,
        double s0r, double s0i, double *restrict dr, double *restrict di)
{
    const __m512d S0R = _mm512_set1_pd(s0r), S0I = _mm512_set1_pd(s0i);
    int q = 0;
    for (; q + 8 <= s; q += 8) {
        __m512d ar = _mm512_loadu_pd(xr + q),         ai = _mm512_loadu_pd(xi + q);
        __m512d br = _mm512_loadu_pd(xr + q + s),     bi = _mm512_loadu_pd(xi + q + s);
        __m512d cr = _mm512_loadu_pd(xr + q + 2 * s), ci = _mm512_loadu_pd(xi + q + 2 * s);
        __m512d er = _mm512_loadu_pd(xr + q + 3 * s), ei = _mm512_loadu_pd(xi + q + 3 * s);
        __m512d t0r = _mm512_add_pd(ar, cr), t0i = _mm512_add_pd(ai, ci);
        __m512d t1r = _mm512_sub_pd(ar, cr), t1i = _mm512_sub_pd(ai, ci);
        __m512d t2r = _mm512_add_pd(br, er), t2i = _mm512_add_pd(bi, ei);
        __m512d t3r = _mm512_sub_pd(br, er), t3i = _mm512_sub_pd(bi, ei);
        __m512d re, im, mr, mi;
        /* unswap at use: state re comes from the imag-plane butterfly */
        re = _mm512_add_pd(_mm512_add_pd(t0i, t2i),
                           _mm512_add_pd(S0R, _mm512_loadu_pd(cfr + q)));
        im = _mm512_add_pd(_mm512_add_pd(t0r, t2r),
                           _mm512_add_pd(S0I, _mm512_loadu_pd(cfi + q)));
        chain_map8(re, im, &mr, &mi);
        _mm512_storeu_pd(dr + q, mr);  _mm512_storeu_pd(di + q, mi);
        re = _mm512_add_pd(_mm512_sub_pd(t1i, t3r),
                           _mm512_add_pd(S0R, _mm512_loadu_pd(cfr + q + s)));
        im = _mm512_add_pd(_mm512_add_pd(t1r, t3i),
                           _mm512_add_pd(S0I, _mm512_loadu_pd(cfi + q + s)));
        chain_map8(re, im, &mr, &mi);
        _mm512_storeu_pd(dr + q + s, mr);  _mm512_storeu_pd(di + q + s, mi);
        re = _mm512_add_pd(_mm512_sub_pd(t0i, t2i),
                           _mm512_add_pd(S0R, _mm512_loadu_pd(cfr + q + 2 * s)));
        im = _mm512_add_pd(_mm512_sub_pd(t0r, t2r),
                           _mm512_add_pd(S0I, _mm512_loadu_pd(cfi + q + 2 * s)));
        chain_map8(re, im, &mr, &mi);
        _mm512_storeu_pd(dr + q + 2 * s, mr);  _mm512_storeu_pd(di + q + 2 * s, mi);
        re = _mm512_add_pd(_mm512_add_pd(t1i, t3r),
                           _mm512_add_pd(S0R, _mm512_loadu_pd(cfr + q + 3 * s)));
        im = _mm512_add_pd(_mm512_sub_pd(t1r, t3i),
                           _mm512_add_pd(S0I, _mm512_loadu_pd(cfi + q + 3 * s)));
        chain_map8(re, im, &mr, &mi);
        _mm512_storeu_pd(dr + q + 3 * s, mr);  _mm512_storeu_pd(di + q + 3 * s, mi);
    }
    for (; q < s; ++q) {
        double a_r = xr[q],         a_i = xi[q];
        double b_r = xr[q + s],     b_i = xi[q + s];
        double c_r = xr[q + 2 * s], c_i = xi[q + 2 * s];
        double d_r = xr[q + 3 * s], d_i = xi[q + 3 * s];
        double t0r = a_r + c_r, t0i = a_i + c_i, t1r = a_r - c_r, t1i = a_i - c_i;
        double t2r = b_r + d_r, t2i = b_i + d_i, t3r = b_r - d_r, t3i = b_i - d_i;
        double z0R = t0r + t2r, z0I = t0i + t2i;
        double z1R = t1r + t3i, z1I = t1i - t3r;
        double z2R = t0r - t2r, z2I = t0i - t2i;
        double z3R = t1r - t3i, z3I = t1i + t3r;
        double re, im;
        re = z0I + s0r + cfr[q];         im = z0R + s0i + cfi[q];
        CHAIN_MAP(re, im, dr[q], di[q]);
        re = z1I + s0r + cfr[q + s];     im = z1R + s0i + cfi[q + s];
        CHAIN_MAP(re, im, dr[q + s], di[q + s]);
        re = z2I + s0r + cfr[q + 2 * s]; im = z2R + s0i + cfi[q + 2 * s];
        CHAIN_MAP(re, im, dr[q + 2 * s], di[q + 2 * s]);
        re = z3I + s0r + cfr[q + 3 * s]; im = z3R + s0i + cfi[q + 3 * s];
        CHAIN_MAP(re, im, dr[q + 3 * s], di[q + 3 * s]);
    }
}

static void st2_last_chain(int s, int P, const double *restrict xr, const double *restrict xi,
        const double *restrict cfr, const double *restrict cfi,
        double s0r, double s0i, double *restrict dr, double *restrict di)
{
    const __m512d S0R = _mm512_set1_pd(s0r), S0I = _mm512_set1_pd(s0i);
    for (int q = 0; q < P; q += 8) {
        __mmask8 mk = (P - q >= 8) ? (__mmask8)0xFF : (__mmask8)((1u << (P - q)) - 1);
        __m512d wi = _mm512_add_pd(_mm512_maskz_loadu_pd(mk, xr + q),
                                   _mm512_maskz_loadu_pd(mk, xr + q + s));
        __m512d wr = _mm512_add_pd(_mm512_maskz_loadu_pd(mk, xi + q),
                                   _mm512_maskz_loadu_pd(mk, xi + q + s));
        __m512d re = _mm512_add_pd(wr, _mm512_add_pd(S0R, _mm512_maskz_loadu_pd(mk, cfr + q)));
        __m512d im = _mm512_add_pd(wi, _mm512_add_pd(S0I, _mm512_maskz_loadu_pd(mk, cfi + q)));
        __m512d mr, mi;
        chain_map8(re, im, &mr, &mi);
        _mm512_mask_storeu_pd(dr + q, mk, mr);
        _mm512_mask_storeu_pd(di + q, mk, mi);
    }
}

static void generic_last_chain(int P, const double *restrict xr, const double *restrict xi,
        const double *restrict cfr, const double *restrict cfi,
        double s0r, double s0i, double *restrict dr, double *restrict di)
{
    const double *restrict wr = xi, *restrict wi = xr;   /* unswap */
#pragma GCC ivdep
    for (int k = 0; k < P; ++k) {
        double re = wr[k] + s0r + cfr[k], im = wi[k] + s0i + cfi[k];
        CHAIN_MAP(re, im, dr[k], di[k]);
    }
}

/* ---------------- four-step conv core for M = 65536 = 256 x 256 ----------------
 *
 * Transpose-free: every pass walks 8-lane column tiles (contiguous 8 doubles,
 * stride 256), the 8-lane FFT-256 rides the SAME Stockham stage kernels with
 * the initial stride set to 8 (lanes live in the inert q-position and never
 * mix).  Forward output lands digit-swapped: X[k2 + 256 k1] at p = k1*256+k2;
 * the kernel spectrum is precomputed in that order, so pointwise never cares.
 * The middle twiddle table W^(a*b) is symmetric in (a,b), so both the forward
 * store and the inverse store read it contiguously.  Pointwise kernel multiply
 * + plane swap fuse into the inverse's first tile load; the unswap + x[0] add
 * + Rader scatter fuse into the last tile store.  4 array passes instead of 12. */

#define FSR 256          /* row/column length */
#define FSL 8            /* lanes per tile */


/* forward step 1: gather a[j1+l + 256 j2] over j2 (lanes l), lane FFT,
 * multiply W^((j1+l) k2), store A1[(j1+l)*256 + k2].
 * xd != NULL: interleaved input gathered through iidx (execute path).
 * xd == NULL: split direct input (plan-time kernel spectrum). */
HOT static void fs_f1(const core_plan *lane, const int *restrict iidx,
                      const double *restrict twm,   /* re then im, 65536 each */
                      const double *restrict xd,
                      const double *restrict sr, const double *restrict si,
                      double *restrict outr, double *restrict outi,
                      double *t0r, double *t0i, double *t1r, double *t1i)
{
    const double *restrict twr = twm, *restrict twi = twm + FSR * FSR;
    for (int j1 = 0; j1 < FSR; j1 += FSL) {
        if (xd) {
            const int *restrict ix = iidx + j1;
#pragma GCC ivdep
            for (int j2 = 0; j2 < FSR; ++j2)
                for (int l = 0; l < FSL; ++l) {
                    int t = ix[l + FSR * j2];
                    t0r[j2 * FSL + l] = xd[2 * t];
                    t0i[j2 * FSL + l] = xd[2 * t + 1];
                }
        } else {
            const double *restrict ar = sr + j1, *restrict ai = si + j1;
#pragma GCC ivdep
            for (int j2 = 0; j2 < FSR; ++j2)
                for (int l = 0; l < FSL; ++l) {
                    t0r[j2 * FSL + l] = ar[l + FSR * j2];
                    t0i[j2 * FSL + l] = ai[l + FSR * j2];
                }
        }
        double *Rr, *Ri;
        core_exec_range(lane, 0, lane->nstage, FSR, FSL, t0r, t0i, t1r, t1i, &Rr, &Ri);
        for (int k2 = 0; k2 < FSR; ++k2) {
            const double *restrict wr = twr + (size_t)k2 * FSR + j1;
            const double *restrict wi = twi + (size_t)k2 * FSR + j1;
            const double *restrict vr = Rr + k2 * FSL;
            const double *restrict vi = Ri + k2 * FSL;
#pragma GCC ivdep
            for (int l = 0; l < FSL; ++l) {
                double zr = vr[l], zi = vi[l];
                outr[(size_t)(j1 + l) * FSR + k2] = zr * wr[l] - zi * wi[l];
                outi[(size_t)(j1 + l) * FSR + k2] = zr * wi[l] + zi * wr[l];
            }
        }
    }
}

/* forward step 2: lane FFT over j1 (columns of A1, lanes = k2), store
 * U[k1*256 + k2+l] (contiguous both sides). */
HOT static void fs_f2(const core_plan *lane,
                      const double *restrict inr, const double *restrict ini,
                      double *restrict outr, double *restrict outi,
                      double *t0r, double *t0i, double *t1r, double *t1i)
{
    for (int k2 = 0; k2 < FSR; k2 += FSL) {
        const double *restrict ar = inr + k2, *restrict ai = ini + k2;
#pragma GCC ivdep
        for (int j1 = 0; j1 < FSR; ++j1)
            for (int l = 0; l < FSL; ++l) {
                t0r[j1 * FSL + l] = ar[l + FSR * j1];
                t0i[j1 * FSL + l] = ai[l + FSR * j1];
            }
        double *Rr, *Ri;
        core_exec_range(lane, 0, lane->nstage, FSR, FSL, t0r, t0i, t1r, t1i, &Rr, &Ri);
#pragma GCC ivdep
        for (int k1 = 0; k1 < FSR; ++k1)
            for (int l = 0; l < FSL; ++l) {
                outr[(size_t)k1 * FSR + k2 + l] = Rr[k1 * FSL + l];
                outi[(size_t)k1 * FSR + k2 + l] = Ri[k1 * FSL + l];
            }
    }
}

/* inverse step 1: load U columns (lanes = k2), multiply by the kernel
 * spectrum (1/M folded in) and SWAP planes in the same load, lane FFT,
 * multiply the swapped value by W^(m1 (k2+l)) (plain product = conjugate
 * twiddle in the true planes), store E[(k2+l)*256 + m1]. */
HOT static void fs_i1(const core_plan *lane,
                      const double *restrict twm,
                      const double *restrict br, const double *restrict bi,
                      const double *restrict Ur, const double *restrict Ui,
                      double *restrict outr, double *restrict outi,
                      double *t0r, double *t0i, double *t1r, double *t1i)
{
    const double *restrict twr = twm, *restrict twi = twm + FSR * FSR;
    for (int k2 = 0; k2 < FSR; k2 += FSL) {
#pragma GCC ivdep
        for (int k1 = 0; k1 < FSR; ++k1)
            for (int l = 0; l < FSL; ++l) {
                size_t pp = (size_t)k1 * FSR + k2 + l;
                double zr = Ur[pp], zi = Ui[pp];
                double kr = br[pp], ki = bi[pp];
                t0r[k1 * FSL + l] = zr * ki + zi * kr;   /* swap(Z*K) */
                t0i[k1 * FSL + l] = zr * kr - zi * ki;
            }
        double *Rr, *Ri;
        core_exec_range(lane, 0, lane->nstage, FSR, FSL, t0r, t0i, t1r, t1i, &Rr, &Ri);
        for (int m1 = 0; m1 < FSR; ++m1) {
            const double *restrict wr = twr + (size_t)m1 * FSR + k2;
            const double *restrict wi = twi + (size_t)m1 * FSR + k2;
            const double *restrict vr = Rr + m1 * FSL;
            const double *restrict vi = Ri + m1 * FSL;
#pragma GCC ivdep
            for (int l = 0; l < FSL; ++l) {
                double zr = vr[l], zi = vi[l];
                outr[(size_t)(k2 + l) * FSR + m1] = zr * wr[l] - zi * wi[l];
                outi[(size_t)(k2 + l) * FSR + m1] = zr * wi[l] + zi * wr[l];
            }
        }
    }
}

/* inverse step 2: lane FFT over k2 (columns of E, lanes = m1), unswap the
 * planes, add x[0], scatter y[oidx[(m1+l) + 256 m2]]. */
HOT static void fs_i2(const core_plan *lane, const int *restrict oidx,
                      const double *restrict inr, const double *restrict ini,
                      double x0r, double x0i, double *restrict yd,
                      double *t0r, double *t0i, double *t1r, double *t1i)
{
    for (int m1 = 0; m1 < FSR; m1 += FSL) {
        const double *restrict ar = inr + m1, *restrict ai = ini + m1;
#pragma GCC ivdep
        for (int k2 = 0; k2 < FSR; ++k2)
            for (int l = 0; l < FSL; ++l) {
                t0r[k2 * FSL + l] = ar[l + FSR * k2];
                t0i[k2 * FSL + l] = ai[l + FSR * k2];
            }
        double *Rr, *Ri;
        core_exec_range(lane, 0, lane->nstage, FSR, FSL, t0r, t0i, t1r, t1i, &Rr, &Ri);
        const int *restrict ox = oidx + m1;
#pragma GCC ivdep
        for (int m2 = 0; m2 < FSR; ++m2)
            for (int l = 0; l < FSL; ++l) {
                int o = ox[l + FSR * m2];
                yd[2 * o] = Ri[m2 * FSL + l] + x0r;      /* unswap */
                yd[2 * o + 1] = Rr[m2 * FSL + l] + x0i;
            }
    }
}

/* ------------------------------ number theory ---------------------------- */

static unsigned long long modpow(unsigned long long b, unsigned long long e,
                                 unsigned long long m)
{
    unsigned long long r = 1;
    b %= m;
    while (e) {
        if (e & 1) r = r * b % m;
        b = b * b % m;
        e >>= 1;
    }
    return r;
}

static int find_generator(int N)
{
    int P = N - 1;
    int fac[16], nf = 0, t = P;
    for (int d = 2; (long)d * d <= t; ++d)
        if (t % d == 0) {
            fac[nf++] = d;
            while (t % d == 0) t /= d;
        }
    if (t > 1) fac[nf++] = t;
    for (int g = 2; g < N; ++g) {
        int ok = 1;
        for (int i = 0; i < nf && ok; ++i)
            if (modpow((unsigned long long)g, (unsigned long long)(P / fac[i]),
                       (unsigned long long)N) == 1)
                ok = 0;
        if (ok) return g;
    }
    return 0;
}

/* ------------------------------- plan ------------------------------------ */

struct fft1d_plan {
    int N, batch;
    int P, M, padded;
    int entry_r;             /* stage-0 radix: 4 or 2 */
    int fuse_last;           /* 4 = full radix-4 exit, 2 = pruned radix-2 exit, 0 = none */
    int fourstep;            /* M == 65536: 256x256 tiled path */
    core_plan core;          /* full-M core (unused when fourstep) */
    core_plan lane;          /* 256-point lane core (fourstep) */
    int *iidx;               /* gather:  a[q] = x[iidx[q]], iidx[q] = g^q mod N   */
    int *oidx;               /* scatter: y[oidx[m]] = x0 + conv[m], oidx = g^-m   */
    int *rev;                /* small-N codelets: chain reversal perm, CRT order  */
    double *br, *bi;         /* kernel spectrum / M (fourstep: digit-swapped order) */
    double *twm;             /* fourstep middle twiddles W^(ab), re[65536] im[65536] */
    double *s0r, *s0i, *s1r, *s1i;   /* scratch, length M */
    double *t0r, *t0i, *t1r, *t1i;   /* fourstep tile ping-pong, 2048 each */
    double *cdr, *cdi;               /* chain state in conv order, length P (lazy) */
    double *cfpr, *cfpi;             /* chain: c-field permuted by oidx, length P (lazy) */
};

static int small_crt_setup(fft1d_plan *p);   /* defined with the codelets below */

const char *fft1d_name(void) { return "d1_rader"; }
const char *fft1d_description(void)
{
    return "Rader prime->cyclic conv sized by N-1: UNPADDED conv 65536=[4,16,16,16,4] at "
           "N=65537 (two-layer radix-16 through an L1 tile, 10 array passes not 12; "
           "paired-p s=4 variant), unpadded 1020=[4,3,5,17] at 1021 (AVX-512 two-half "
           "radix-17, scatter-fused exit), CRT codelets Z12=Z4xZ3 / Z30=Z2xZ3xZ5 at 13/31 "
           "(Agarwal-Cooley) with ACROSS-BATCH SoA: 8 transforms per zmm, transpose only "
           "at entry/exit; pad 256 at 127; split-complex Stockham core (from d1_bluestein), "
           "kernel-fused inverse entry, X[0] free from conv DC; fused map chain keeps state "
           "in conv order (gather o scatter = index reversal), Newton rsqrt/rcp map "
           "(no divider) in every vector chain exit";
}
int fft1d_supports(int L)
{
    return L == 13 || L == 31 || L == 127 || L == 1021 || L == 65537;
}

fft1d_plan *fft1d_create(int L, int batch)
{
    if (!fft1d_supports(L) || batch < 1) return NULL;
    fft1d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->N = L; p->batch = batch;
    const int P = L - 1;
    p->P = P;

    /* conv length: N-1 itself when the core can run it directly (2/3/4/5/8
       stages plus one dense radix-17: 1020 = [4,3,5,17]), else pow2 pad */
    c17_init();
    int M;
    int tmpr[MAXSTAGE], tmpn;
    if (core_factor(P, tmpr, &tmpn) && (tmpr[0] == 4 || tmpr[0] == 2)) {
        M = P; p->padded = 0;
    } else {
        M = 1;
        while (M < 2 * P - 1) M <<= 1;
        p->padded = 1;
    }
    p->M = M;
    /* four-step 256x256 path: correct, but measured SLOWER than the plain
       6-stage Stockham on wallaby (1017 vs 703 us at 65537 B=1) -- the tile
       load/store passes cost more than the traffic they save while the 1 MB
       working set still fits L3. Kept for a lower-bandwidth target; off. */
    /* four-step 256x256: correct but SLOWER on both machines measured --
       wallaby 1017 vs 703 us (r1), Ice Lake a80n0 1635 vs 1338 us B=1 and
       1965 vs 1657 us B=16 (r3, same-minute A/B).  The tile load/store
       passes cost more than the traffic they save at this size.  Off. */
    p->fourstep = 0 && (M == FSR * FSR);
    if (p->fourstep) {
        static const int lane_stages[3] = { 8, 8, 4 };
        if (!core_init_stages(&p->lane, FSR, lane_stages, 3)) goto fail;
        p->twm = amalloc(2 * (size_t)FSR * FSR * sizeof(double));
        p->t0r = amalloc((size_t)FSR * FSL * sizeof(double));
        p->t0i = amalloc((size_t)FSR * FSL * sizeof(double));
        p->t1r = amalloc((size_t)FSR * FSL * sizeof(double));
        p->t1i = amalloc((size_t)FSR * FSL * sizeof(double));
        if (!p->twm || !p->t0r || !p->t0i || !p->t1r || !p->t1i) goto fail;
        double *twr = p->twm, *twi = p->twm + (size_t)FSR * FSR;
        for (int a = 0; a < FSR; ++a)
            for (int b = 0; b < FSR; ++b) {
                double ph = -2.0 * M_PI * ((double)(a * b) / (double)M);
                twr[(size_t)a * FSR + b] = cos(ph);
                twi[(size_t)a * FSR + b] = sin(ph);
            }
    } else {
        if (!core_init(&p->core, M)) goto fail;
        p->entry_r = p->core.radix[0];

        int lastr = p->core.radix[p->core.nstage - 1];
        if (p->core.nstage >= 2 && lastr == 4 && P == M) p->fuse_last = 4;
        else if (p->core.nstage >= 2 && lastr == 2 && P <= M / 2) p->fuse_last = 2;
        else if (p->core.nstage >= 2 && lastr == 17 && P == M) p->fuse_last = 17;
        else p->fuse_last = 0;
        /* pruned radix-4 gather entry needs m < P <= 2m; guaranteed for the pow2
           pad (2P-1 <= M < 4P-2), but keep the check honest */
        if (p->padded && (p->entry_r != 4 || P <= M / 4 || P > M / 2)) goto fail;
    }

    int g = find_generator(L);
    if (!g) goto fail;
    p->iidx = amalloc((size_t)M * sizeof(int));   /* M >= P; tail unused */
    p->oidx = amalloc((size_t)M * sizeof(int));
    p->br = amalloc((size_t)M * sizeof(double));
    p->bi = amalloc((size_t)M * sizeof(double));
    p->s0r = amalloc((size_t)M * sizeof(double));
    p->s0i = amalloc((size_t)M * sizeof(double));
    p->s1r = amalloc((size_t)M * sizeof(double));
    p->s1i = amalloc((size_t)M * sizeof(double));
    if (!p->iidx || !p->oidx || !p->br || !p->bi ||
        !p->s0r || !p->s0i || !p->s1r || !p->s1i) goto fail;

    unsigned long long ginv = modpow((unsigned long long)g, (unsigned long long)(P - 1),
                                     (unsigned long long)L);
    unsigned long long t = 1, ti = 1;
    for (int q = 0; q < P; ++q) {
        p->iidx[q] = (int)t;
        p->oidx[q] = (int)ti;
        t = t * (unsigned long long)g % (unsigned long long)L;
        ti = ti * ginv % (unsigned long long)L;
    }

    /* kernel b[r] = exp(-2 pi i g^-r / N); wrapped copy for the padded pad:
       bhat[M-j] = b[P-j], j = 1..P-1 (disjoint from [0,P) since M >= 2P-1) */
    memset(p->s0r, 0, (size_t)M * sizeof(double));
    memset(p->s0i, 0, (size_t)M * sizeof(double));
    for (int r = 0; r < P; ++r) {
        double ph = -2.0 * M_PI * ((double)p->oidx[r] / (double)L);
        p->s0r[r] = cos(ph);
        p->s0i[r] = sin(ph);
    }
    if (p->padded)
        for (int j = 1; j < P; ++j) {
            p->s0r[M - j] = p->s0r[P - j];
            p->s0i[M - j] = p->s0i[P - j];
        }
    double invM = 1.0 / (double)M;
    if (p->fourstep) {
        /* kernel spectrum through the same four-step forward, so it lands in
           the digit-swapped order the pointwise multiply will see */
        fs_f1(&p->lane, NULL, p->twm, NULL, p->s0r, p->s0i, p->s1r, p->s1i,
              p->t0r, p->t0i, p->t1r, p->t1i);
        fs_f2(&p->lane, p->s1r, p->s1i, p->br, p->bi,
              p->t0r, p->t0i, p->t1r, p->t1i);
        for (int k = 0; k < M; ++k) { p->br[k] *= invM; p->bi[k] *= invM; }
    } else {
        double *Rr, *Ri;
        core_exec(&p->core, p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
        for (int k = 0; k < M; ++k) { p->br[k] = Rr[k] * invM; p->bi[k] = Ri[k] * invM; }
    }
    if ((L == 13 || L == 31) && !small_crt_setup(p)) goto fail;
    return p;
fail:
    fft1d_destroy(p);
    return NULL;
}

static void rader_one(fft1d_plan *p, const double _Complex *x, double _Complex *y)
{
    const core_plan *c = &p->core;
    const int M = p->M, P = p->P;
    const double *xd = (const double *)x;
    double *yd = (double *)y;

    if (p->fourstep) {
        fs_f1(&p->lane, p->iidx, p->twm, xd, NULL, NULL, p->s0r, p->s0i,
              p->t0r, p->t0i, p->t1r, p->t1i);
        fs_f2(&p->lane, p->s0r, p->s0i, p->s1r, p->s1i,
              p->t0r, p->t0i, p->t1r, p->t1i);
        const double sr = p->s1r[0], si = p->s1i[0];     /* forward DC bin */
        fs_i1(&p->lane, p->twm, p->br, p->bi, p->s1r, p->s1i, p->s0r, p->s0i,
              p->t0r, p->t0i, p->t1r, p->t1i);
        const double xr0 = xd[0], xi0 = xd[1];
        fs_i2(&p->lane, p->oidx, p->s0r, p->s0i, xr0, xi0, yd,
              p->t0r, p->t0i, p->t1r, p->t1i);
        yd[0] = xr0 + sr;
        yd[1] = xi0 + si;
        return;
    }

    const int r0 = p->entry_r, m0 = M / r0;
    /* forward conv FFT, gather fused into stage 0 */
    if (r0 == 4) {
        if (p->padded)
            st4_gather_pruned(m0, P, c->twr[0], c->twi[0], xd, p->iidx, p->s0r, p->s0i);
        else
            st4_gather_full(m0, c->twr[0], c->twi[0], xd, p->iidx, p->s0r, p->s0i);
    } else {
        st2_gather_full(m0, c->twr[0], c->twi[0], xd, p->iidx, p->s0r, p->s0i);
    }
    double *Rr, *Ri;
    core_exec_range(c, 1, c->nstage, m0, r0, p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);

    const double sumr = Rr[0], sumi = Ri[0];   /* DC bin = sum of gathered a */

    /* inverse entry: kernel multiply + plane swap + stage-0 butterfly */
    double *dr = (Rr == p->s0r) ? p->s1r : p->s0r;
    double *di = (Rr == p->s0r) ? p->s1i : p->s0i;
    if (r0 == 4)
        st4_first_bhat(m0, c->twr[0], c->twi[0], Rr, Ri, p->br, p->bi, dr, di);
    else
        st2_first_bhat(m0, c->twr[0], c->twi[0], Rr, Ri, p->br, p->bi, dr, di);
    double *er = (dr == p->s0r) ? p->s1r : p->s0r;
    double *ei = (dr == p->s0r) ? p->s1i : p->s0i;
    double *Vr, *Vi;
    core_exec_range(c, 1, p->fuse_last ? c->nstage - 1 : c->nstage,
                    m0, r0, dr, di, er, ei, &Vr, &Vi);

    const double x0r = xd[0], x0i = xd[1];
    if (p->fuse_last == 4)
        st4_last_scatter(M / 4, Vr, Vi, p->oidx, x0r, x0i, yd);
    else if (p->fuse_last == 2)
        st2_last_scatter(M / 2, P, Vr, Vi, p->oidx, x0r, x0i, yd);
    else if (p->fuse_last == 17) {
        /* A/B'd against plain st17 + the generic exit pass below: fused
           scalar-store scatter wins, 7.85 vs 8.28 us at 1021 B=1. */
        st17_sink snk = { .oidx = p->oidx, .x0r = x0r, .x0i = x0i, .yd = yd };
        double *tr = (Vr == p->s0r) ? p->s1r : p->s0r;
        double *ti = (Vr == p->s0r) ? p->s1i : p->s0i;
        st17_scatter(M / 17, &snk, Vr, Vi, tr, ti);
    }
    else {
        const double *restrict wr = Vi, *restrict wi = Vr;   /* unswap */
        const int *restrict oidx = p->oidx;
#pragma GCC ivdep
        for (int k = 0; k < P; ++k) {
            int o = oidx[k];
            yd[2 * o] = wr[k] + x0r;
            yd[2 * o + 1] = wi[k] + x0i;
        }
    }
    yd[0] = x0r + sumr;
    yd[1] = x0i + sumi;
}

/* ---- register-resident small-N codelets (round d1_r2) ----
 *
 * At M=12/30 even a flattened stage ladder loses to store-forward latency:
 * every stage round-trips the 12/30 points through the s-buffers.  Instead:
 * the Rader conv is bracketed by permutations we own, so map the cyclic ring
 * by CRT (Agarwal-Cooley): Z12 = Z4 x Z3, Z30 = Z2 x Z3 x Z5.  Cyclic conv
 * of length 12/30 becomes a multidimensional cyclic conv = twiddle-FREE
 * DFT4(x)DFT3 / DFT2(x)DFT3(x)DFT5, straight-line in registers.  The CRT
 * index maps fold into iidx/oidx at plan time; the kernel spectrum is
 * computed by the same codelet DFT, so the ordering is consistent by
 * construction.  Inverse = forward on swapped planes, 1/M in the kernel. */

/* The DFT bodies are type-parametric: dftv_t is `double` in the scalar
 * codelets and `__m512d` in the across-batch SoA codelets (GNU C infix
 * arithmetic on vector types; scalar constants broadcast). */
#define DFT12_2D(re, im)                                                       \
    do {                                                                       \
        for (int n2_ = 0; n2_ < 3; ++n2_) {   /* DFT4 along n1 (stride 3) */   \
            dftv_t ar = re[n2_],     ai = im[n2_];                             \
            dftv_t br = re[n2_ + 3], bi = im[n2_ + 3];                         \
            dftv_t cr = re[n2_ + 6], ci = im[n2_ + 6];                         \
            dftv_t dr = re[n2_ + 9], di = im[n2_ + 9];                         \
            dftv_t t0r = ar + cr, t0i = ai + ci, t1r = ar - cr, t1i = ai - ci; \
            dftv_t t2r = br + dr, t2i = bi + di, t3r = br - dr, t3i = bi - di; \
            re[n2_]     = t0r + t2r;  im[n2_]     = t0i + t2i;                 \
            re[n2_ + 3] = t1r + t3i;  im[n2_ + 3] = t1i - t3r;                 \
            re[n2_ + 6] = t0r - t2r;  im[n2_ + 6] = t0i - t2i;                 \
            re[n2_ + 9] = t1r - t3i;  im[n2_ + 9] = t1i + t3r;                 \
        }                                                                      \
        for (int k1_ = 0; k1_ < 4; ++k1_) {   /* DFT3 along n2 (stride 1) */   \
            const int b_ = 3 * k1_;   /* direct indexing only: pointer         \
                                         aliases here defeat gcc's SROA */    \
            dftv_t tr = re[b_ + 1] + re[b_ + 2], ti = im[b_ + 1] + im[b_ + 2]; \
            dftv_t ur = re[b_ + 1] - re[b_ + 2], ui = im[b_ + 1] - im[b_ + 2]; \
            dftv_t mr = re[b_] - 0.5 * tr, mi = im[b_] - 0.5 * ti;             \
            re[b_] += tr;               im[b_] += ti;                          \
            re[b_ + 1] = mr + CS3 * ui; im[b_ + 1] = mi - CS3 * ur;            \
            re[b_ + 2] = mr - CS3 * ui; im[b_ + 2] = mi + CS3 * ur;            \
        }                                                                      \
    } while (0)

#define DFT30_3D(re, im)                                                       \
    do {                                                                       \
        for (int j_ = 0; j_ < 15; ++j_) {     /* DFT2 along n1 (stride 15) */  \
            dftv_t ar = re[j_], ai = im[j_];                                   \
            dftv_t br = re[j_ + 15], bi = im[j_ + 15];                         \
            re[j_] = ar + br;       im[j_] = ai + bi;                          \
            re[j_ + 15] = ar - br;  im[j_ + 15] = ai - bi;                     \
        }                                                                      \
        for (int b_ = 0; b_ < 30; b_ += 15)   /* DFT3 along n2 (stride 5) */   \
            for (int n3_ = 0; n3_ < 5; ++n3_) {                                \
                const int q_ = b_ + n3_;                                       \
                dftv_t tr = re[q_ + 5] + re[q_ + 10], ti = im[q_ + 5] + im[q_ + 10]; \
                dftv_t ur = re[q_ + 5] - re[q_ + 10], ui = im[q_ + 5] - im[q_ + 10]; \
                dftv_t mr = re[q_] - 0.5 * tr, mi = im[q_] - 0.5 * ti;         \
                re[q_] += tr;                im[q_] += ti;                     \
                re[q_ + 5]  = mr + CS3 * ui; im[q_ + 5]  = mi - CS3 * ur;      \
                re[q_ + 10] = mr - CS3 * ui; im[q_ + 10] = mi + CS3 * ur;      \
            }                                                                  \
        for (int b_ = 0; b_ < 30; b_ += 5) {  /* DFT5 along n3 (stride 1) */   \
            dftv_t t1r = re[b_ + 1] + re[b_ + 4], t1i = im[b_ + 1] + im[b_ + 4]; \
            dftv_t t2r = re[b_ + 2] + re[b_ + 3], t2i = im[b_ + 2] + im[b_ + 3]; \
            dftv_t v1r = re[b_ + 1] - re[b_ + 4], v1i = im[b_ + 1] - im[b_ + 4]; \
            dftv_t v2r = re[b_ + 2] - re[b_ + 3], v2i = im[b_ + 2] - im[b_ + 3]; \
            dftv_t ar = re[b_], ai = im[b_];                                   \
            dftv_t m1r = ar + C51 * t1r + C52 * t2r, m1i = ai + C51 * t1i + C52 * t2i; \
            dftv_t m2r = ar + C52 * t1r + C51 * t2r, m2i = ai + C52 * t1i + C51 * t2i; \
            dftv_t n1r = S51 * v1r + S52 * v2r, n1i = S51 * v1i + S52 * v2i;   \
            dftv_t n2r = S52 * v1r - S51 * v2r, n2i = S52 * v1i - S51 * v2i;   \
            re[b_] = ar + t1r + t2r;  im[b_] = ai + t1i + t2i;                 \
            re[b_ + 1] = m1r + n1i;  im[b_ + 1] = m1i - n1r;                   \
            re[b_ + 4] = m1r - n1i;  im[b_ + 4] = m1i + n1r;                   \
            re[b_ + 2] = m2r + n2i;  im[b_ + 2] = m2i - n2r;                   \
            re[b_ + 3] = m2r - n2i;  im[b_ + 3] = m2i + n2r;                   \
        }                                                                      \
    } while (0)

#define CS3 0.86602540378443864676
#define C51 0.30901699437494742410
#define S51 0.95105651629515357212
#define C52 (-0.80901699437494742410)
#define S52 0.58778525229247312917

/* CRT position of conv index t */
static inline int crt_pos(int P, int t)
{
    return (P == 12) ? 3 * (t % 4) + (t % 3)
                     : 15 * (t % 2) + 5 * (t % 3) + (t % 5);
}

#define RADER_SMALL_ONE(PN, DFT)                                               \
    do {                                                                       \
        typedef double dftv_t;                                                 \
        double Rr[PN], Ri[PN], Sr[PN], Si[PN];                                 \
        const int *restrict iidx = p->iidx, *restrict oidx = p->oidx;          \
        for (int i = 0; i < PN; ++i) {                                         \
            int t = iidx[i];                                                   \
            Rr[i] = xd[2 * t];  Ri[i] = xd[2 * t + 1];                         \
        }                                                                      \
        DFT(Rr, Ri);                                                           \
        const double sumr = Rr[0], sumi = Ri[0];                               \
        const double *restrict br = p->br, *restrict bi = p->bi;               \
        for (int i = 0; i < PN; ++i) {        /* swap(R*B), 1/M in B */        \
            Sr[i] = Rr[i] * bi[i] + Ri[i] * br[i];                             \
            Si[i] = Rr[i] * br[i] - Ri[i] * bi[i];                             \
        }                                                                      \
        DFT(Sr, Si);                                                           \
        const double x0r = xd[0], x0i = xd[1];                                 \
        for (int i = 0; i < PN; ++i) {        /* unswap + x0 + scatter */      \
            int o = oidx[i];                                                   \
            yd[2 * o] = Si[i] + x0r;                                           \
            yd[2 * o + 1] = Sr[i] + x0i;                                       \
        }                                                                      \
        yd[0] = x0r + sumr;                                                    \
        yd[1] = x0i + sumi;                                                    \
    } while (0)

/* rebuild iidx/oidx/kernel-spectrum in CRT order, build the reversal table;
   the kernel goes through the SAME codelet DFT so ordering is consistent */
static int small_crt_setup(fft1d_plan *p)
{
    typedef double dftv_t;
    const int P = p->P, L = p->N;
    p->rev = amalloc((size_t)P * sizeof(int));
    if (!p->rev) return 0;
    int ii2[30], oo2[30];
    double bkr[30], bki[30];
    for (int t = 0; t < P; ++t) {
        int i = crt_pos(P, t);
        ii2[i] = p->iidx[t];
        oo2[i] = p->oidx[t];
        double ph = -2.0 * M_PI * ((double)p->oidx[t] / (double)L);
        bkr[i] = cos(ph);  bki[i] = sin(ph);
        p->rev[i] = crt_pos(P, (P - t) % P);
    }
    memcpy(p->iidx, ii2, (size_t)P * sizeof(int));
    memcpy(p->oidx, oo2, (size_t)P * sizeof(int));
    if (P == 12) { DFT12_2D(bkr, bki); }
    else         { DFT30_3D(bkr, bki); }
    for (int i = 0; i < P; ++i) {
        p->br[i] = bkr[i] / (double)P;
        p->bi[i] = bki[i] / (double)P;
    }
    return 1;
}

__attribute__((optimize("no-tree-vectorize,no-tree-slp-vectorize")))
static void rader_one_13(const fft1d_plan *restrict p,
                         const double *restrict xd, double *restrict yd)
{
    RADER_SMALL_ONE(12, DFT12_2D);
}

__attribute__((optimize("no-tree-vectorize,no-tree-slp-vectorize")))
static void rader_one_31(const fft1d_plan *restrict p,
                         const double *restrict xd, double *restrict yd)
{
    RADER_SMALL_ONE(30, DFT30_3D);
}

/* fused map chain, small-N: state lives split in conv(2D)-order registers/L1;
   interior steps replace the gather with the plan-time reversal permutation
   (gather o scatter = t -> P-t, in CRT order = p->rev). */
/* Two map styles for the scalar chain codelets.  SCALAR keeps the 12-element
 * state fully register-resident (touching it with intrinsic loads spills it:
 * 0.167 vs 0.099 us/step at L=13, measured).  VEC8 wins at L=31 where the
 * 30-element arrays are memory-backed anyway and 30 scalar sqrt+div per step
 * choke the divider (0.250 -> 0.206 us/step, measured). */
#define SMALL_MAP_SCALAR(PN)                                                   \
            for (int i = 0; i < PN; ++i) {                                     \
                double re = Si[i] + st0r + cfr[i];                             \
                double im = Sr[i] + st0i + cfi[i];                             \
                CHAIN_MAP(re, im, dR[i], dI[i]);                               \
            }

#define SMALL_MAP_VEC8(PN)                                                     \
            for (int i = 0; i < PN; i += 8) {                                  \
                __mmask8 mk_ = (PN - i >= 8) ? (__mmask8)0xFF                  \
                                             : (__mmask8)((1u << (PN - i)) - 1); \
                __m512d re_ = _mm512_add_pd(_mm512_maskz_loadu_pd(mk_, Si + i), \
                                  _mm512_add_pd(_mm512_set1_pd(st0r),          \
                                      _mm512_maskz_loadu_pd(mk_, cfr + i)));   \
                __m512d im_ = _mm512_add_pd(_mm512_maskz_loadu_pd(mk_, Sr + i), \
                                  _mm512_add_pd(_mm512_set1_pd(st0i),          \
                                      _mm512_maskz_loadu_pd(mk_, cfi + i)));   \
                __m512d mr_, mi_;                                              \
                chain_map8(re_, im_, &mr_, &mi_);                              \
                _mm512_mask_storeu_pd(dR + i, mk_, mr_);                       \
                _mm512_mask_storeu_pd(dI + i, mk_, mi_);                       \
            }

#define RADER_SMALL_CHAIN(PN, DFT, MAP)                                        \
    do {                                                                       \
        typedef double dftv_t;                                                 \
        double dR[PN], dI[PN], cfr[PN], cfi[PN];                               \
        if (m <= 0) {   /* no steps: state = input */                          \
            memcpy(od, xd, 2u * (PN + 1) * sizeof(double));                    \
            return;                                                            \
        }                                                                      \
        const int *restrict iidx = p->iidx, *restrict oidx = p->oidx;          \
        const int *restrict rev = p->rev;                                      \
        for (int i = 0; i < PN; ++i) {                                         \
            int o = oidx[i];                                                   \
            cfr[i] = cd[2 * o];  cfi[i] = cd[2 * o + 1];                       \
        }                                                                      \
        const double cf0r = cd[0], cf0i = cd[1];                               \
        double st0r = xd[0], st0i = xd[1];                                     \
        for (int step = 0; step < m; ++step) {                                 \
            double Rr[PN], Ri[PN], Sr[PN], Si[PN];                             \
            if (step == 0) {                                                   \
                for (int i = 0; i < PN; ++i) {                                 \
                    int t = iidx[i];                                           \
                    Rr[i] = xd[2 * t];  Ri[i] = xd[2 * t + 1];                 \
                }                                                              \
            } else {                                                           \
                for (int i = 0; i < PN; ++i) {                                 \
                    int t = rev[i];                                            \
                    Rr[i] = dR[t];  Ri[i] = dI[t];                             \
                }                                                              \
            }                                                                  \
            DFT(Rr, Ri);                                                       \
            const double sumr = Rr[0], sumi = Ri[0];                           \
            const double *restrict br = p->br, *restrict bi = p->bi;           \
            for (int i = 0; i < PN; ++i) {                                     \
                Sr[i] = Rr[i] * bi[i] + Ri[i] * br[i];                         \
                Si[i] = Rr[i] * br[i] - Ri[i] * bi[i];                         \
            }                                                                  \
            DFT(Sr, Si);                                                       \
            MAP(PN);                                                           \
            double re0 = st0r + sumr + cf0r, im0 = st0i + sumi + cf0i;         \
            CHAIN_MAP(re0, im0, st0r, st0i);                                   \
        }                                                                      \
        for (int i = 0; i < PN; ++i) {                                         \
            int o = oidx[i];                                                   \
            od[2 * o] = dR[i];  od[2 * o + 1] = dI[i];                         \
        }                                                                      \
        od[0] = st0r;  od[1] = st0i;                                           \
    } while (0)

__attribute__((optimize("no-tree-vectorize,no-tree-slp-vectorize")))
static void rader_chain_13(const fft1d_plan *restrict p, const double *restrict xd,
                           const double *restrict cd, double *restrict od, int m)
{
    RADER_SMALL_CHAIN(12, DFT12_2D, SMALL_MAP_SCALAR);
}

__attribute__((optimize("no-tree-vectorize,no-tree-slp-vectorize")))
static void rader_chain_31(const fft1d_plan *restrict p, const double *restrict xd,
                           const double *restrict cd, double *restrict od, int m)
{
    RADER_SMALL_CHAIN(30, DFT30_3D, SMALL_MAP_VEC8);
}

/* ---- across-batch SoA codelets (round d1_r3) ----
 *
 * The batched cells at 13/31 were losing to fftw_custom_soa / d1_batchlane,
 * whose lever is lane j = transform j: 8 batch elements per zmm, ZERO in-lane
 * shuffles (the survey's top batched-regime lever, confirmed by d1_batchlane's
 * record).  Here the SAME CRT codelet bodies run with dftv_t = __m512d --
 * every butterfly line does 8 transforms.  The only lane traffic is the AoS
 * transpose at entry/exit (128-bit pair loads + 2 parity permutes per point,
 * the GATHER8 shape; scatter is its mirror).  The chain keeps 8 lanes of
 * conv-order state live across all m steps -- interior steps have no
 * transpose at all.  The map is vector sqrt/div: unlike the r2 single-lane
 * case (where scalar won, 0.099 vs 0.110), each vsqrtpd here serves 8
 * transforms, so the throughput math flips. */

__attribute__((always_inline)) static inline void
soa_gather8(const double *restrict xd, size_t off, size_t stride,
            __m512d *vre, __m512d *vim)
{
    const __m512i EVN = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i ODD = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    __m512d g0 = _mm512_castpd128_pd512(_mm_loadu_pd(xd + off));
    g0 = _mm512_insertf64x2(g0, _mm_loadu_pd(xd + off + stride), 1);
    g0 = _mm512_insertf64x2(g0, _mm_loadu_pd(xd + off + 2 * stride), 2);
    g0 = _mm512_insertf64x2(g0, _mm_loadu_pd(xd + off + 3 * stride), 3);
    __m512d g1 = _mm512_castpd128_pd512(_mm_loadu_pd(xd + off + 4 * stride));
    g1 = _mm512_insertf64x2(g1, _mm_loadu_pd(xd + off + 5 * stride), 1);
    g1 = _mm512_insertf64x2(g1, _mm_loadu_pd(xd + off + 6 * stride), 2);
    g1 = _mm512_insertf64x2(g1, _mm_loadu_pd(xd + off + 7 * stride), 3);
    *vre = _mm512_permutex2var_pd(g0, EVN, g1);
    *vim = _mm512_permutex2var_pd(g0, ODD, g1);
}

__attribute__((always_inline)) static inline void
soa_scatter8(double *restrict yd, size_t off, size_t stride,
             __m512d vre, __m512d vim)
{
    const __m512i LO = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i HI = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    __m512d lo = _mm512_permutex2var_pd(vre, LO, vim);
    __m512d hi = _mm512_permutex2var_pd(vre, HI, vim);
    _mm_storeu_pd(yd + off,              _mm512_castpd512_pd128(lo));
    _mm_storeu_pd(yd + off + stride,     _mm512_extractf64x2_pd(lo, 1));
    _mm_storeu_pd(yd + off + 2 * stride, _mm512_extractf64x2_pd(lo, 2));
    _mm_storeu_pd(yd + off + 3 * stride, _mm512_extractf64x2_pd(lo, 3));
    _mm_storeu_pd(yd + off + 4 * stride, _mm512_castpd512_pd128(hi));
    _mm_storeu_pd(yd + off + 5 * stride, _mm512_extractf64x2_pd(hi, 1));
    _mm_storeu_pd(yd + off + 6 * stride, _mm512_extractf64x2_pd(hi, 2));
    _mm_storeu_pd(yd + off + 7 * stride, _mm512_extractf64x2_pd(hi, 3));
}

#define CHAIN_MAP_V(re, im, outr, outi)  chain_map8((re), (im), &(outr), &(outi))

#define RADER_SOA8_ONE(PN, DFT)                                                \
    do {                                                                       \
        typedef __m512d dftv_t;                                                \
        const size_t STR = 2u * (size_t)p->N;                                  \
        __m512d Rr[PN], Ri[PN], Sr[PN], Si[PN];                                \
        const int *restrict iidx = p->iidx, *restrict oidx = p->oidx;          \
        for (int i = 0; i < PN; ++i)                                           \
            soa_gather8(xd, 2u * (size_t)iidx[i], STR, &Rr[i], &Ri[i]);        \
        DFT(Rr, Ri);                                                           \
        const __m512d sumr = Rr[0], sumi = Ri[0];                              \
        const double *restrict br = p->br, *restrict bi = p->bi;               \
        for (int i = 0; i < PN; ++i) {        /* swap(R*B), 1/M in B */        \
            Sr[i] = Rr[i] * bi[i] + Ri[i] * br[i];                             \
            Si[i] = Rr[i] * br[i] - Ri[i] * bi[i];                             \
        }                                                                      \
        DFT(Sr, Si);                                                           \
        __m512d x0r, x0i;                                                      \
        soa_gather8(xd, 0, STR, &x0r, &x0i);                                   \
        for (int i = 0; i < PN; ++i)          /* unswap + x0 + scatter */      \
            soa_scatter8(yd, 2u * (size_t)oidx[i], STR,                        \
                         Si[i] + x0r, Sr[i] + x0i);                            \
        soa_scatter8(yd, 0, STR, x0r + sumr, x0i + sumi);                      \
    } while (0)

static void rader_soa8_13(const fft1d_plan *restrict p,
                          const double *restrict xd, double *restrict yd)
{
    RADER_SOA8_ONE(12, DFT12_2D);
}

static void rader_soa8_31(const fft1d_plan *restrict p,
                          const double *restrict xd, double *restrict yd)
{
    RADER_SOA8_ONE(30, DFT30_3D);
}

#define RADER_SOA8_CHAIN(PN, DFT)                                              \
    do {                                                                       \
        typedef __m512d dftv_t;                                                \
        const size_t STR = 2u * (size_t)p->N;                                  \
        __m512d dR[PN], dI[PN], cfr[PN], cfi[PN];                              \
        const int *restrict iidx = p->iidx, *restrict oidx = p->oidx;          \
        const int *restrict rev = p->rev;                                      \
        for (int i = 0; i < PN; ++i)                                           \
            soa_gather8(cd, 2u * (size_t)oidx[i], STR, &cfr[i], &cfi[i]);      \
        __m512d cf0r, cf0i, st0r, st0i;                                        \
        soa_gather8(cd, 0, STR, &cf0r, &cf0i);                                 \
        soa_gather8(xd, 0, STR, &st0r, &st0i);                                 \
        for (int step = 0; step < m; ++step) {                                 \
            __m512d Rr[PN], Ri[PN], Sr[PN], Si[PN];                            \
            if (step == 0) {                                                   \
                for (int i = 0; i < PN; ++i)                                   \
                    soa_gather8(xd, 2u * (size_t)iidx[i], STR, &Rr[i], &Ri[i]); \
            } else {                                                           \
                for (int i = 0; i < PN; ++i) {                                 \
                    int t = rev[i];                                            \
                    Rr[i] = dR[t];  Ri[i] = dI[t];                             \
                }                                                              \
            }                                                                  \
            DFT(Rr, Ri);                                                       \
            const __m512d sumr = Rr[0], sumi = Ri[0];                          \
            const double *restrict br = p->br, *restrict bi = p->bi;           \
            for (int i = 0; i < PN; ++i) {                                     \
                Sr[i] = Rr[i] * bi[i] + Ri[i] * br[i];                         \
                Si[i] = Rr[i] * br[i] - Ri[i] * bi[i];                         \
            }                                                                  \
            DFT(Sr, Si);                                                       \
            for (int i = 0; i < PN; ++i) {                                     \
                __m512d re = Si[i] + st0r + cfr[i];                            \
                __m512d im = Sr[i] + st0i + cfi[i];                            \
                CHAIN_MAP_V(re, im, dR[i], dI[i]);                             \
            }                                                                  \
            __m512d re0 = st0r + sumr + cf0r, im0 = st0i + sumi + cf0i;        \
            CHAIN_MAP_V(re0, im0, st0r, st0i);                                 \
        }                                                                      \
        for (int i = 0; i < PN; ++i)                                           \
            soa_scatter8(od, 2u * (size_t)oidx[i], STR, dR[i], dI[i]);         \
        soa_scatter8(od, 0, STR, st0r, st0i);                                  \
    } while (0)

static void rader_chain_soa8_13(const fft1d_plan *restrict p, const double *restrict xd,
                                const double *restrict cd, double *restrict od, int m)
{
    RADER_SOA8_CHAIN(12, DFT12_2D);
}

static void rader_chain_soa8_31(const fft1d_plan *restrict p, const double *restrict xd,
                                const double *restrict cd, double *restrict od, int m)
{
    RADER_SOA8_CHAIN(30, DFT30_3D);
}

void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    const int L = p->N;
    if (L == 13 || L == 31) {
        const int B = p->batch;
        int b = 0;
        for (; b + 8 <= B; b += 8) {
            const double *xb = (const double *)(in + (size_t)b * L);
            double *yb = (double *)(out + (size_t)b * L);
            if (L == 13) rader_soa8_13(p, xb, yb);
            else         rader_soa8_31(p, xb, yb);
        }
        for (; b < B; ++b) {
            const double *xb = (const double *)(in + (size_t)b * L);
            double *yb = (double *)(out + (size_t)b * L);
            if (L == 13) rader_one_13(p, xb, yb);
            else         rader_one_31(p, xb, yb);
        }
        return;
    }
    for (int b = 0; b < p->batch; ++b)
        rader_one(p, in + (size_t)b * L, out + (size_t)b * L);
}

/* Fused m-step map chain: state <- (FFT(state)+c) / (1+|FFT(state)+c|).
 * State stays split, in conv-output order, across steps (see the chain-mode
 * kernels above); each batch element runs its whole chain back to back so the
 * P-sized buffers stay hot. Falls back to nothing gracefully: if the lazy
 * buffers cannot be allocated the plain execute+map loop runs instead. */
void fft1d_chain(fft1d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    const int L = p->N, B = p->batch, M = p->M, P = p->P;
    const core_plan *cp = &p->core;

    if (L == 13 || L == 31) {
        int b = 0;
        if (m >= 1)
            for (; b + 8 <= B; b += 8) {
                const double *xb = (const double *)(x0 + (size_t)b * L);
                const double *cb = (const double *)(c + (size_t)b * L);
                double *ob = (double *)(final_out + (size_t)b * L);
                if (L == 13) rader_chain_soa8_13(p, xb, cb, ob, m);
                else         rader_chain_soa8_31(p, xb, cb, ob, m);
            }
        for (; b < B; ++b) {
            const double *xb = (const double *)(x0 + (size_t)b * L);
            const double *cb = (const double *)(c + (size_t)b * L);
            double *ob = (double *)(final_out + (size_t)b * L);
            if (L == 13) rader_chain_13(p, xb, cb, ob, m);
            else         rader_chain_31(p, xb, cb, ob, m);
        }
        return;
    }

    if (!p->cdr) {
        p->cdr = amalloc((size_t)P * sizeof(double));
        p->cdi = amalloc((size_t)P * sizeof(double));
        p->cfpr = amalloc((size_t)P * sizeof(double));
        p->cfpi = amalloc((size_t)P * sizeof(double));
    }
    if (!p->cdr || !p->cdi || !p->cfpr || !p->cfpi || p->fourstep) {
        /* plain fallback: execute + vectorized map, ping-pong via s buffers is
           not possible (wrong size), so use final_out and a lazy y buffer */
        const size_t count = (size_t)L * B;
        double _Complex *y = malloc(count * sizeof(double _Complex));
        if (!y) return;   /* nothing sane to do */
        memcpy(final_out, x0, count * sizeof(double _Complex));
        for (int s = 0; s < m; ++s) {
            fft1d_execute(p, final_out, y);
            const double *restrict zr = (const double *)y;
            const double *restrict cr = (const double *)c;
            double *restrict o = (double *)final_out;
#pragma GCC ivdep
            for (size_t i = 0; i < count; ++i) {
                double re = zr[2 * i] + cr[2 * i];
                double im = zr[2 * i + 1] + cr[2 * i + 1];
                double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                o[2 * i] = re * sc;
                o[2 * i + 1] = im * sc;
            }
        }
        free(y);
        return;
    }

    const int r0 = p->entry_r, m0 = M / r0;
    for (int b = 0; b < B; ++b) {
        const double *restrict xd = (const double *)(x0 + (size_t)b * L);
        const double *restrict cd = (const double *)(c + (size_t)b * L);
        double *restrict od = (double *)(final_out + (size_t)b * L);
        /* permute the constant field once per batch element */
        const int *restrict oidx = p->oidx;
#pragma GCC ivdep
        for (int k = 0; k < P; ++k) {
            int o = oidx[k];
            p->cfpr[k] = cd[2 * o];
            p->cfpi[k] = cd[2 * o + 1];
        }
        const double cf0r = cd[0], cf0i = cd[1];
        double st0r = xd[0], st0i = xd[1];   /* state[0], carried as a scalar */

        for (int step = 0; step < m; ++step) {
            /* entry stage */
            if (step == 0) {
                if (r0 == 4) {
                    if (p->padded)
                        st4_gather_pruned(m0, P, cp->twr[0], cp->twi[0], xd, p->iidx, p->s0r, p->s0i);
                    else
                        st4_gather_full(m0, cp->twr[0], cp->twi[0], xd, p->iidx, p->s0r, p->s0i);
                } else {
                    st2_gather_full(m0, cp->twr[0], cp->twi[0], xd, p->iidx, p->s0r, p->s0i);
                }
            } else {
                if (r0 == 4) {
                    if (p->padded)
                        st4_rev_pruned(m0, P, cp->twr[0], cp->twi[0], p->cdr, p->cdi, p->s0r, p->s0i);
                    else
                        st4_rev_full(m0, P, cp->twr[0], cp->twi[0], p->cdr, p->cdi, p->s0r, p->s0i);
                } else {
                    st2_rev_full(m0, P, cp->twr[0], cp->twi[0], p->cdr, p->cdi, p->s0r, p->s0i);
                }
            }
            double *Rr, *Ri;
            core_exec_range(cp, 1, cp->nstage, m0, r0, p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
            const double sumr = Rr[0], sumi = Ri[0];
            double *dr = (Rr == p->s0r) ? p->s1r : p->s0r;
            double *di = (Rr == p->s0r) ? p->s1i : p->s0i;
            if (r0 == 4)
                st4_first_bhat(m0, cp->twr[0], cp->twi[0], Rr, Ri, p->br, p->bi, dr, di);
            else
                st2_first_bhat(m0, cp->twr[0], cp->twi[0], Rr, Ri, p->br, p->bi, dr, di);
            double *er = (dr == p->s0r) ? p->s1r : p->s0r;
            double *ei = (dr == p->s0r) ? p->s1i : p->s0i;
            double *Vr, *Vi;
            core_exec_range(cp, 1, p->fuse_last ? cp->nstage - 1 : cp->nstage,
                            m0, r0, dr, di, er, ei, &Vr, &Vi);
            /* exit + map into the conv-order state */
            if (p->fuse_last == 4)
                st4_last_chain(M / 4, Vr, Vi, p->cfpr, p->cfpi, st0r, st0i, p->cdr, p->cdi);
            else if (p->fuse_last == 2)
                st2_last_chain(M / 2, P, Vr, Vi, p->cfpr, p->cfpi, st0r, st0i, p->cdr, p->cdi);
            else if (p->fuse_last == 17) {
                st17_sink snk = { .cfr = p->cfpr, .cfi = p->cfpi,
                                  .s0r = st0r, .s0i = st0i,
                                  .dr = p->cdr, .di = p->cdi };
                double *tr = (Vr == p->s0r) ? p->s1r : p->s0r;
                double *ti = (Vr == p->s0r) ? p->s1i : p->s0i;
                st17_chain(M / 17, &snk, Vr, Vi, tr, ti);
            }
            else
                generic_last_chain(P, Vr, Vi, p->cfpr, p->cfpi, st0r, st0i, p->cdr, p->cdi);
            /* state[0] update uses the OLD state[0] */
            double re0 = st0r + sumr + cf0r, im0 = st0i + sumi + cf0i;
            CHAIN_MAP(re0, im0, st0r, st0i);
        }
        /* materialize the natural-order interleaved final state */
#pragma GCC ivdep
        for (int k = 0; k < P; ++k) {
            int o = oidx[k];
            od[2 * o] = p->cdr[k];
            od[2 * o + 1] = p->cdi[k];
        }
        od[0] = st0r;
        od[1] = st0i;
    }
}

void fft1d_destroy(fft1d_plan *p)
{
    if (!p) return;
    core_free(&p->core);
    core_free(&p->lane);
    free(p->iidx); free(p->oidx); free(p->rev);
    free(p->br); free(p->bi);
    free(p->twm);
    free(p->s0r); free(p->s0i); free(p->s1r); free(p->s1i);
    free(p->t0r); free(p->t0i); free(p->t1r); free(p->t1i);
    free(p->cdr); free(p->cdi); free(p->cfpr); free(p->cfpi);
    free(p);
}
