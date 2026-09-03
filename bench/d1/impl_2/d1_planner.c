/*
 * d1_planner -- LIBRARY LAYER (adoption-scored): 1D factorization -> candidate
 * algorithm chains.
 *
 * ROUND r2 ENGINE: the r1 scalar interleaved kernels were 2-10x behind at every
 * cell; this round the whole engine is SPLIT-COMPLEX (separate re/im planes) --
 * the stage kernel family (st2/3/4/5/8), the per-stage twiddle layout, the
 * inverse-as-forward-on-swapped-planes trick, the Bluestein entry/exit fusions,
 * #pragma GCC ivdep on every hot loop and the per-function 512-bit target
 * attribute are ADOPTED FROM d1_bluestein (see strategies/d1_planner.md; d1_rader
 * took the same core, which is this layer working as intended in reverse).
 * What the planner ADDS on top of that core:
 *   - a GENERIC odd-prime stage (any prime r <= 61) using d1_prime's
 *     symmetric-pair real-coefficient fold, so "smooth" is a much wider set:
 *     1020 = [4,3,5,17] runs UNPADDED as the Rader conv of 1021, and every
 *     prime <= 61 is a single dense stage (no Bluestein detour at 13/31).
 *   - the factorization dispatch: L smooth (factors 2/3/5 + primes <= 61) ->
 *     direct Stockham; L prime with smooth L-1 -> unpadded Rader (65537 ->
 *     conv 2^16, 1021 -> conv 1020); anything else -> Bluestein with the
 *     minimal smooth pad M = 3^a 5^b 2^c >= 2L-1, 4|M (10007 -> 20480,
 *     100003 -> 204800; choose_M is d1_bluestein's).
 *   - across-batch 8-lane execution for the direct path (enter the SAME stage
 *     kernels with initial stride 8 on lane-blocked data; twiddles are stride-
 *     independent), gated to cache-resident sizes.
 *   - fused split-state map chains for all three kinds (transform-outer, per
 *     d1_pow2's residency lesson): state stays split between steps, interleaved
 *     output materializes only at the final step; the Bluestein chain keeps the
 *     state chirp-premultiplied (d1_bluestein's scheme, ported).
 * Twiddle/chirp/dense tables are generated with long-double phase (cosl/sinl)
 * after exact integer phase reduction -- d1_pow2's finding that M_PI's rounding
 * is a biased ~2e-16 phase error.
 *
 * Compile-time self test:  gcc -DPLANNER_TEST -O2 d1_planner.c -lm
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../fft1d_api.h"

/* The graded build's flags leave gcc at 256-bit vector preference; these
   shuffle-free split loops want zmm. Per-function attribute only -- a file-wide
   pragma resets the ISA (d1_bluestein's hard-won lesson). */
#define HOT __attribute__((target("arch=icelake-server,prefer-vector-width=512")))

#define MAXSTAGE 48
#define GENR_MAX 61            /* largest prime run as a direct dense-fold stage */
#define LANEV    8             /* across-batch lanes per group */
#ifndef LANE_MAX_N
#define LANE_MAX_N 1024        /* lane path only when 8 transforms stay cache-resident */
#endif

static const long double PIL = 3.141592653589793238462643383279502884L;

static void *amalloc(size_t bytes)
{
    void *p = NULL;
    if (bytes == 0) bytes = 64;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
}

/* ------------------- split-complex mixed-radix Stockham core -------------------
 * Stage order: leading 4 (or 2), radix-8 workhorses, 3s, 5s, generic primes
 * ascending (late, where the stride is long and their inner loops vectorize),
 * pow2 leftover (4/2) LAST (the Bluestein fused pruned exit wants it). */

typedef struct {
    int n, nstage;
    int radix[MAXSTAGE];
    double *twr[MAXSTAGE], *twi[MAXSTAGE];  /* blocks [j-1][p], j=1..r-1 */
    double *gC[MAXSTAGE], *gS[MAXSTAGE];    /* generic stages: h*h fold tables */
    double *twstore;
} core_plan;

static int core_factor(int n, int *radix, int *nstage)
{
    int ns = 0, a = 0;
    while (n % 2 == 0) { ++a; n /= 2; }
    if (a >= 2)      { radix[ns++] = 4; a -= 2; }
    else if (a == 1) { radix[ns++] = 2; a -= 1; }
    while (a >= 3 && ns < MAXSTAGE) { radix[ns++] = 8; a -= 3; }
    while (n % 3 == 0 && ns < MAXSTAGE) { radix[ns++] = 3; n /= 3; }
    while (n % 5 == 0 && ns < MAXSTAGE) { radix[ns++] = 5; n /= 5; }
    for (int p = 7; p <= GENR_MAX && ns < MAXSTAGE; p += 2)
        while (n % p == 0 && ns < MAXSTAGE) { radix[ns++] = p; n /= p; }
    if (a == 2 && ns < MAXSTAGE)      radix[ns++] = 4;
    else if (a == 1 && ns < MAXSTAGE) radix[ns++] = 2;
    *nstage = ns;
    return n == 1 && ns <= MAXSTAGE;
}

static int core_is_generic(int r) { return r != 2 && r != 3 && r != 4 && r != 5 && r != 8; }

static void core_free(core_plan *c)
{
    free(c->twstore);
    for (int st = 0; st < c->nstage; ++st) { free(c->gC[st]); free(c->gS[st]); }
    memset(c, 0, sizeof *c);
}

static int core_init(core_plan *c, int n)
{
    memset(c, 0, sizeof *c);
    c->n = n;
    if (!core_factor(n, c->radix, &c->nstage)) return 0;
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
                long ph = ((long)p * j) % ncur;         /* exact integer phase */
                long double t = -2.0L * PIL * (long double)ph / (long double)ncur;
                wr[(size_t)(j - 1) * m + p] = (double)cosl(t);
                wi[(size_t)(j - 1) * m + p] = (double)sinl(t);
            }
        wr += (size_t)(r - 1) * m; wi += (size_t)(r - 1) * m;
        if (core_is_generic(r)) {
            int h = (r - 1) / 2;
            c->gC[st] = amalloc((size_t)h * h * sizeof(double));
            c->gS[st] = amalloc((size_t)h * h * sizeof(double));
            if (!c->gC[st] || !c->gS[st]) return 0;
            for (int j = 1; j <= h; ++j)
                for (int i = 1; i <= h; ++i) {
                    long ph = ((long)i * j) % r;
                    long double t = 2.0L * PIL * (long double)ph / (long double)r;
                    c->gC[st][(size_t)(j - 1) * h + (i - 1)] = (double)cosl(t);
                    c->gS[st][(size_t)(j - 1) * h + (i - 1)] = (double)sinl(t);
                }
        }
        ncur = m;
    }
    return 1;
}

/* scratch (doubles) needed by generic stages at initial stride s0 */
static size_t core_gen_scratch(const core_plan *c, int s0)
{
    size_t mx = 0, s = (size_t)s0;
    for (int st = 0; st < c->nstage; ++st) {
        int r = c->radix[st];
        if (core_is_generic(r)) {
            size_t need = (size_t)(4 * ((r - 1) / 2) + 4) * s;
            if (need > mx) mx = need;
        }
        s *= r;
    }
    return mx;
}

/* ------------------------------ stage kernels ------------------------------
 * st2/st3/st4/st5/st8 adopted from d1_bluestein (impl/d1_bluestein.c, r1). */

HOT static void st2(int m, int s, const double *restrict wr, const double *restrict wi,
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

HOT static void st4(int m, int s, const double *restrict wr, const double *restrict wi,
                const double *restrict xr, const double *restrict xi,
                double *restrict yr, double *restrict yi)
{
    const double *restrict w1r = wr,         *restrict w1i = wi;
    const double *restrict w2r = wr + m,     *restrict w2i = wi + m;
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

HOT static void st8(int m, int s, const double *restrict wr, const double *restrict wi,
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

HOT static void st3(int m, int s, const double *restrict wr, const double *restrict wi,
                const double *restrict xr, const double *restrict xi,
                double *restrict yr, double *restrict yi)
{
    const double s3 = 0.86602540378443864676;
    const double *restrict w1r = wr,     *restrict w1i = wi;
    const double *restrict w2r = wr + m, *restrict w2i = wi + m;
    for (int p = 0; p < m; ++p) {
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

HOT static void st5(int m, int s, const double *restrict wr, const double *restrict wi,
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

/* Generic odd-prime stage r <= 61, symmetric-pair real-coefficient fold
 * (d1_prime's dense trick, applied per stage): u_i = x_i + x_{r-i},
 * v_i = x_i - x_{r-i}; X_j = x0 + sum_i C_ji u_i - i * sum_i S_ji v_i, and
 * X_{r-j} flips the sign of the S sum. All multiplies are real x complex.
 * scratch: 4*h*s (u/v planes) + 4*s (A/B accumulators) doubles. */
HOT static void stg(int r, int m, int s,
        const double *restrict wr, const double *restrict wi,
        const double *restrict C, const double *restrict S,
        double *restrict scratch,
        const double *restrict xr, const double *restrict xi,
        double *restrict yr, double *restrict yi)
{
    const int h = (r - 1) / 2;
    double *restrict ur = scratch;
    double *restrict ui = ur + (size_t)h * s;
    double *restrict vr = ui + (size_t)h * s;
    double *restrict vi = vr + (size_t)h * s;
    double *restrict Ar = vi + (size_t)h * s;
    double *restrict Ai = Ar + s;
    double *restrict Br = Ai + s;
    double *restrict Bi = Br + s;
    const size_t sm = (size_t)s * m;
    for (int p = 0; p < m; ++p) {
        const double *restrict x0r = xr + (size_t)s * p;
        const double *restrict x0i = xi + (size_t)s * p;
        double *restrict o0r = yr + (size_t)s * r * p;
        double *restrict o0i = yi + (size_t)s * r * p;
#pragma GCC ivdep
        for (int q = 0; q < s; ++q) { o0r[q] = x0r[q]; o0i[q] = x0i[q]; }
        for (int i = 1; i <= h; ++i) {
            const double *restrict xar = x0r + sm * i, *restrict xai = x0i + sm * i;
            const double *restrict xbr = x0r + sm * (r - i), *restrict xbi = x0i + sm * (r - i);
            double *restrict u_r = ur + (size_t)(i - 1) * s, *restrict u_i = ui + (size_t)(i - 1) * s;
            double *restrict v_r = vr + (size_t)(i - 1) * s, *restrict v_i = vi + (size_t)(i - 1) * s;
#pragma GCC ivdep
            for (int q = 0; q < s; ++q) {
                double aur = xar[q] + xbr[q], aui = xai[q] + xbi[q];
                u_r[q] = aur; u_i[q] = aui;
                v_r[q] = xar[q] - xbr[q]; v_i[q] = xai[q] - xbi[q];
                o0r[q] += aur; o0i[q] += aui;
            }
        }
        for (int j = 1; j <= h; ++j) {
            const double *restrict Cj = C + (size_t)(j - 1) * h;
            const double *restrict Sj = S + (size_t)(j - 1) * h;
#pragma GCC ivdep
            for (int q = 0; q < s; ++q) { Ar[q] = x0r[q]; Ai[q] = x0i[q]; Br[q] = 0.0; Bi[q] = 0.0; }
            for (int i = 0; i < h; ++i) {
                const double c_ = Cj[i], s_ = Sj[i];
                const double *restrict u_r = ur + (size_t)i * s, *restrict u_i = ui + (size_t)i * s;
                const double *restrict v_r = vr + (size_t)i * s, *restrict v_i = vi + (size_t)i * s;
#pragma GCC ivdep
                for (int q = 0; q < s; ++q) {
                    Ar[q] += c_ * u_r[q]; Ai[q] += c_ * u_i[q];
                    Br[q] += s_ * v_r[q]; Bi[q] += s_ * v_i[q];
                }
            }
            const double w1r = wr[(size_t)(j - 1) * m + p], w1i = wi[(size_t)(j - 1) * m + p];
            const double w2r = wr[(size_t)(r - j - 1) * m + p], w2i = wi[(size_t)(r - j - 1) * m + p];
            double *restrict oar = yr + (size_t)s * ((size_t)r * p + j);
            double *restrict oai = yi + (size_t)s * ((size_t)r * p + j);
            double *restrict obr = yr + (size_t)s * ((size_t)r * p + (r - j));
            double *restrict obi = yi + (size_t)s * ((size_t)r * p + (r - j));
#pragma GCC ivdep
            for (int q = 0; q < s; ++q) {
                double e1r = Ar[q] + Bi[q], e1i = Ai[q] - Br[q];
                double e2r = Ar[q] - Bi[q], e2i = Ai[q] + Br[q];
                oar[q] = e1r * w1r - e1i * w1i;  oai[q] = e1r * w1i + e1i * w1r;
                obr[q] = e2r * w2r - e2i * w2i;  obi[q] = e2r * w2i + e2i * w2r;
            }
        }
    }
}

/* Stages [st0, st1), sub-length n, initial stride s (s = LANEV for the lane
 * path -- the twiddle tables are stride-independent). */
static void core_exec_range(const core_plan *c, double *gs, int st0, int st1, int n, int s,
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
        case 5: st5(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break;
        case 8: st8(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break;
        default: stg(r, m, s, c->twr[st], c->twi[st], c->gC[st], c->gS[st], gs,
                     xr, xi, yr, yi); break;
        }
        n = m; s *= r;
        t = xr; xr = yr; yr = t;
        t = xi; xi = yi; yi = t;
    }
    *outr = xr; *outi = xi;
}

/* --------- direct-path fused entry stages (deinterleave folded in) ---------- */

HOT static void st4_first_deint(int m, const double *restrict wr, const double *restrict wi,
        const double *restrict xd, double *restrict yr, double *restrict yi)
{
    const double *restrict w1r = wr,         *restrict w1i = wi;
    const double *restrict w2r = wr + m,     *restrict w2i = wi + m;
    const double *restrict w3r = wr + 2 * m, *restrict w3i = wi + 2 * m;
#pragma GCC ivdep
    for (int p = 0; p < m; ++p) {
        double ar = xd[2 * p],           ai = xd[2 * p + 1];
        double br = xd[2 * (p + m)],     bi = xd[2 * (p + m) + 1];
        double cr = xd[2 * (p + 2 * m)], ci = xd[2 * (p + 2 * m) + 1];
        double dr = xd[2 * (p + 3 * m)], di = xd[2 * (p + 3 * m) + 1];
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

HOT static void st2_first_deint(int m, const double *restrict wr, const double *restrict wi,
        const double *restrict xd, double *restrict yr, double *restrict yi)
{
#pragma GCC ivdep
    for (int p = 0; p < m; ++p) {
        double ar = xd[2 * p],       ai = xd[2 * p + 1];
        double br = xd[2 * (p + m)], bi = xd[2 * (p + m) + 1];
        double u_r = ar - br, u_i = ai - bi;
        yr[2 * p] = ar + br;  yi[2 * p] = ai + bi;
        yr[2 * p + 1] = u_r * wr[p] - u_i * wi[p];
        yi[2 * p + 1] = u_r * wi[p] + u_i * wr[p];
    }
}

/* ---- direct-path fused exit stages: the FINAL stage (m=1, unit twiddles)
 * with the interleaved store folded in -- saves one full read+write pass ---- */

HOT static void st2_last_int(int s, const double *restrict xr, const double *restrict xi,
                             double *restrict yd)
{
#pragma GCC ivdep
    for (int q = 0; q < s; ++q) {
        double ar = xr[q], ai = xi[q], br = xr[q + s], bi = xi[q + s];
        yd[2 * q]           = ar + br;  yd[2 * q + 1]           = ai + bi;
        yd[2 * (q + s)]     = ar - br;  yd[2 * (q + s) + 1]     = ai - bi;
    }
}

HOT static void st3_last_int(int s, const double *restrict xr, const double *restrict xi,
                             double *restrict yd)
{
    const double s3 = 0.86602540378443864676;
#pragma GCC ivdep
    for (int q = 0; q < s; ++q) {
        double ar = xr[q], ai = xi[q];
        double br = xr[q + s], bi = xi[q + s];
        double cr = xr[q + 2 * s], ci = xi[q + 2 * s];
        double tr = br + cr, ti = bi + ci;
        double ur = br - cr, ui = bi - ci;
        double mr = ar - 0.5 * tr, mi = ai - 0.5 * ti;
        yd[2 * q] = ar + tr;  yd[2 * q + 1] = ai + ti;
        yd[2 * (q + s)]         = mr + s3 * ui;  yd[2 * (q + s) + 1]         = mi - s3 * ur;
        yd[2 * (q + 2 * s)]     = mr - s3 * ui;  yd[2 * (q + 2 * s) + 1]     = mi + s3 * ur;
    }
}

HOT static void st4_last_int(int s, const double *restrict xr, const double *restrict xi,
                             double *restrict yd)
{
#pragma GCC ivdep
    for (int q = 0; q < s; ++q) {
        double ar = xr[q], ai = xi[q];
        double br = xr[q + s], bi = xi[q + s];
        double cr = xr[q + 2 * s], ci = xi[q + 2 * s];
        double dr = xr[q + 3 * s], di = xi[q + 3 * s];
        double t0r = ar + cr, t0i = ai + ci, t1r = ar - cr, t1i = ai - ci;
        double t2r = br + dr, t2i = bi + di, t3r = br - dr, t3i = bi - di;
        yd[2 * q]               = t0r + t2r;  yd[2 * q + 1]               = t0i + t2i;
        yd[2 * (q + s)]         = t1r + t3i;  yd[2 * (q + s) + 1]         = t1i - t3r;
        yd[2 * (q + 2 * s)]     = t0r - t2r;  yd[2 * (q + 2 * s) + 1]     = t0i - t2i;
        yd[2 * (q + 3 * s)]     = t1r - t3i;  yd[2 * (q + 3 * s) + 1]     = t1i + t3r;
    }
}

HOT static void st5_last_int(int s, const double *restrict xr, const double *restrict xi,
                             double *restrict yd)
{
    const double c1 = 0.30901699437494742410, s1 = 0.95105651629515357212;
    const double c2 = -0.80901699437494742410, s2 = 0.58778525229247312917;
#pragma GCC ivdep
    for (int q = 0; q < s; ++q) {
        double ar = xr[q], ai = xi[q];
        double br = xr[q + s], bi = xi[q + s];
        double cr = xr[q + 2 * s], ci = xi[q + 2 * s];
        double dr = xr[q + 3 * s], di = xi[q + 3 * s];
        double er = xr[q + 4 * s], ei = xi[q + 4 * s];
        double t1r = br + er, t1i = bi + ei;
        double t2r = cr + dr, t2i = ci + di;
        double v1r = br - er, v1i = bi - ei;
        double v2r = cr - dr, v2i = ci - di;
        double m1r = ar + c1 * t1r + c2 * t2r, m1i = ai + c1 * t1i + c2 * t2i;
        double m2r = ar + c2 * t1r + c1 * t2r, m2i = ai + c2 * t1i + c1 * t2i;
        double n1r = s1 * v1r + s2 * v2r, n1i = s1 * v1i + s2 * v2i;
        double n2r = s2 * v1r - s1 * v2r, n2i = s2 * v1i - s1 * v2i;
        yd[2 * q] = ar + t1r + t2r;  yd[2 * q + 1] = ai + t1i + t2i;
        yd[2 * (q + s)]         = m1r + n1i;  yd[2 * (q + s) + 1]         = m1i - n1r;
        yd[2 * (q + 2 * s)]     = m2r + n2i;  yd[2 * (q + 2 * s) + 1]     = m2i - n2r;
        yd[2 * (q + 3 * s)]     = m2r - n2i;  yd[2 * (q + 3 * s) + 1]     = m2i + n2r;
        yd[2 * (q + 4 * s)]     = m1r - n1i;  yd[2 * (q + 4 * s) + 1]     = m1i + n1r;
    }
}

HOT static void st8_last_int(int s, const double *restrict xr, const double *restrict xi,
                             double *restrict yd)
{
    const double C8 = 0.70710678118654752440;
#pragma GCC ivdep
    for (int q = 0; q < s; ++q) {
        double a0r = xr[q],         a0i = xi[q];
        double a1r = xr[q + s],     a1i = xi[q + s];
        double a2r = xr[q + 2 * s], a2i = xi[q + 2 * s];
        double a3r = xr[q + 3 * s], a3i = xi[q + 3 * s];
        double a4r = xr[q + 4 * s], a4i = xi[q + 4 * s];
        double a5r = xr[q + 5 * s], a5i = xi[q + 5 * s];
        double a6r = xr[q + 6 * s], a6i = xi[q + 6 * s];
        double a7r = xr[q + 7 * s], a7i = xi[q + 7 * s];
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
        yd[2 * q]               = e0r + f0r;  yd[2 * q + 1]               = e0i + f0i;
        yd[2 * (q + s)]         = e1r + g1r;  yd[2 * (q + s) + 1]         = e1i + g1i;
        yd[2 * (q + 2 * s)]     = e2r + g2r;  yd[2 * (q + 2 * s) + 1]     = e2i + g2i;
        yd[2 * (q + 3 * s)]     = e3r + g3r;  yd[2 * (q + 3 * s) + 1]     = e3i + g3i;
        yd[2 * (q + 4 * s)]     = e0r - f0r;  yd[2 * (q + 4 * s) + 1]     = e0i - f0i;
        yd[2 * (q + 5 * s)]     = e1r - g1r;  yd[2 * (q + 5 * s) + 1]     = e1i - g1i;
        yd[2 * (q + 6 * s)]     = e2r - g2r;  yd[2 * (q + 6 * s) + 1]     = e2i - g2i;
        yd[2 * (q + 7 * s)]     = e3r - g3r;  yd[2 * (q + 7 * s) + 1]     = e3i - g3i;
    }
}

/* dense symmetric-fold DFT for a single generic stage at B=1 (interleaved in/out).
 * acc: 4*h doubles of scratch. */
HOT static void dense_sym(int L, const double *restrict C, const double *restrict S,
                          double *restrict acc,
                          const double *restrict xd, double *restrict yd)
{
    const int h = (L - 1) / 2;
    double *restrict Ar = acc, *restrict Ai = acc + h;
    double *restrict Br = acc + 2 * h, *restrict Bi = acc + 3 * h;
    const double x0r = xd[0], x0i = xd[1];
    double s0r = x0r, s0i = x0i;
#pragma GCC ivdep
    for (int k = 0; k < h; ++k) { Ar[k] = x0r; Ai[k] = x0i; Br[k] = 0.0; Bi[k] = 0.0; }
    for (int j = 1; j <= h; ++j) {
        double ar_ = xd[2 * j], ai_ = xd[2 * j + 1];
        double br_ = xd[2 * (L - j)], bi_ = xd[2 * (L - j) + 1];
        double ujr = ar_ + br_, uji = ai_ + bi_;
        double vjr = ar_ - br_, vji = ai_ - bi_;
        s0r += ujr; s0i += uji;
        const double *restrict Cj = C + (size_t)(j - 1) * h;
        const double *restrict Sj = S + (size_t)(j - 1) * h;
#pragma GCC ivdep
        for (int k = 0; k < h; ++k) {
            Ar[k] += Cj[k] * ujr; Ai[k] += Cj[k] * uji;
            Br[k] += Sj[k] * vjr; Bi[k] += Sj[k] * vji;
        }
    }
    yd[0] = s0r; yd[1] = s0i;
#pragma GCC ivdep
    for (int k = 1; k <= h; ++k) {
        yd[2 * k]           = Ar[k - 1] + Bi[k - 1];
        yd[2 * k + 1]       = Ai[k - 1] - Br[k - 1];
        yd[2 * (L - k)]     = Ar[k - 1] - Bi[k - 1];
        yd[2 * (L - k) + 1] = Ai[k - 1] + Br[k - 1];
    }
}

/* ------------- Bluestein fused entry / exit stages (from d1_bluestein) ------ */

HOT static void st4_first_chirp(int m, int L,
        const double *restrict wr, const double *restrict wi,
        const double *restrict xd,
        const double *restrict car, const double *restrict cai,
        double *restrict yr, double *restrict yi)
{
    const double *restrict w1r = wr,         *restrict w1i = wi;
    const double *restrict w2r = wr + m,     *restrict w2i = wi + m;
    const double *restrict w3r = wr + 2 * m, *restrict w3i = wi + 2 * m;
    const int nb = L - m;
#pragma GCC ivdep
    for (int p = 0; p < nb; ++p) {
        double x0r = xd[2 * p], x0i = xd[2 * p + 1];
        double a_r = x0r * car[p] - x0i * cai[p];
        double a_i = x0r * cai[p] + x0i * car[p];
        double x1r = xd[2 * (p + m)], x1i = xd[2 * (p + m) + 1];
        double b_r = x1r * car[p + m] - x1i * cai[p + m];
        double b_i = x1r * cai[p + m] + x1i * car[p + m];
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
        double x0r = xd[2 * p], x0i = xd[2 * p + 1];
        double a_r = x0r * car[p] - x0i * cai[p];
        double a_i = x0r * cai[p] + x0i * car[p];
        yr[4 * p] = a_r;  yi[4 * p] = a_i;
        yr[4 * p + 1] = a_r * w1r[p] - a_i * w1i[p];
        yi[4 * p + 1] = a_r * w1i[p] + a_i * w1r[p];
        yr[4 * p + 2] = a_r * w2r[p] - a_i * w2i[p];
        yi[4 * p + 2] = a_r * w2i[p] + a_i * w2r[p];
        yr[4 * p + 3] = a_r * w3r[p] - a_i * w3i[p];
        yi[4 * p + 3] = a_r * w3i[p] + a_i * w3r[p];
    }
}

/* same entry stage, input already chirp-premultiplied and split (chain path) */
HOT static void st4_first_pre(int m, int L,
        const double *restrict wr, const double *restrict wi,
        const double *restrict pr, const double *restrict pi,
        double *restrict yr, double *restrict yi)
{
    const double *restrict w1r = wr,         *restrict w1i = wi;
    const double *restrict w2r = wr + m,     *restrict w2i = wi + m;
    const double *restrict w3r = wr + 2 * m, *restrict w3i = wi + 2 * m;
    const int nb = L - m;
#pragma GCC ivdep
    for (int p = 0; p < nb; ++p) {
        double a_r = pr[p], a_i = pi[p];
        double b_r = pr[p + m], b_i = pi[p + m];
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
        double a_r = pr[p], a_i = pi[p];
        yr[4 * p] = a_r;  yi[4 * p] = a_i;
        yr[4 * p + 1] = a_r * w1r[p] - a_i * w1i[p];
        yi[4 * p + 1] = a_r * w1i[p] + a_i * w1r[p];
        yr[4 * p + 2] = a_r * w2r[p] - a_i * w2i[p];
        yi[4 * p + 2] = a_r * w2i[p] + a_i * w2r[p];
        yr[4 * p + 3] = a_r * w3r[p] - a_i * w3i[p];
        yi[4 * p + 3] = a_r * w3i[p] + a_i * w3r[p];
    }
}

/* inverse entry: pointwise multiply by the kernel spectrum, swap planes
 * (inverse-as-forward), and the radix-4 stage-0 butterfly, one pass */
HOT static void st4_first_bhat(int m,
        const double *restrict wr, const double *restrict wi,
        const double *restrict Rr, const double *restrict Ri,
        const double *restrict br, const double *restrict bi,
        double *restrict yr, double *restrict yi)
{
    const double *restrict w1r = wr,         *restrict w1i = wi;
    const double *restrict w2r = wr + m,     *restrict w2i = wi + m;
    const double *restrict w3r = wr + 2 * m, *restrict w3i = wi + 2 * m;
#pragma GCC ivdep
    for (int p = 0; p < m; ++p) {
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

/* ---------------- Rader fused entry / exit stages (d1_rader's shape) --------
 * entry: the g^q gather feeds the radix-4 stage-0 butterfly directly (the
 * gathered sequence never hits memory); the DC sum accumulates in the same
 * pass. exit: the inverse's final radix-4 stage (m=1, unit twiddles, planes
 * swapped) scatters X[qout[t]] = x0 + conv[t] as it computes. */
static void st4_first_gather(int m, const double *restrict wr, const double *restrict wi,
        const double *restrict xd, const int *restrict qin,
        double *restrict yr, double *restrict yi, double *sumr, double *sumi)
{
    const double *restrict w1r = wr,         *restrict w1i = wi;
    const double *restrict w2r = wr + m,     *restrict w2i = wi + m;
    const double *restrict w3r = wr + 2 * m, *restrict w3i = wi + 2 * m;
    double sr = 0.0, si = 0.0;
    for (int p = 0; p < m; ++p) {
        int j0 = qin[p], j1 = qin[p + m], j2 = qin[p + 2 * m], j3 = qin[p + 3 * m];
        double ar = xd[2 * j0], ai = xd[2 * j0 + 1];
        double br = xd[2 * j1], bi = xd[2 * j1 + 1];
        double cr = xd[2 * j2], ci = xd[2 * j2 + 1];
        double dr = xd[2 * j3], di = xd[2 * j3 + 1];
        sr += ar + br + cr + dr;  si += ai + bi + ci + di;
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
    *sumr = sr; *sumi = si;
}

/* same, reading a split source through the gather permutation (chain path) */
static void st4_first_gather_split(int m, const double *restrict wr, const double *restrict wi,
        const double *restrict sxr, const double *restrict sxi, const int *restrict qin,
        double *restrict yr, double *restrict yi, double *sumr, double *sumi)
{
    const double *restrict w1r = wr,         *restrict w1i = wi;
    const double *restrict w2r = wr + m,     *restrict w2i = wi + m;
    const double *restrict w3r = wr + 2 * m, *restrict w3i = wi + 2 * m;
    double sr = 0.0, si = 0.0;
    for (int p = 0; p < m; ++p) {
        int j0 = qin[p], j1 = qin[p + m], j2 = qin[p + 2 * m], j3 = qin[p + 3 * m];
        double ar = sxr[j0], ai = sxi[j0];
        double br = sxr[j1], bi = sxi[j1];
        double cr = sxr[j2], ci = sxi[j2];
        double dr = sxr[j3], di = sxi[j3];
        sr += ar + br + cr + dr;  si += ai + bi + ci + di;
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
    *sumr = sr; *sumi = si;
}

/* final inverse radix-4 stage (m=1, s=P/4, unit twiddles, swapped planes) with
 * the Rader output scatter folded in: yd[2*qout[t]] = x0 + conv[t] */
static void st4_last_scatter(int s, const double *restrict xr, const double *restrict xi,
        const int *restrict qout, double x0r, double x0i, double *restrict yd)
{
    for (int q = 0; q < s; ++q) {
        /* butterfly per plane; conv_r comes from the i-plane (unswap) */
        double ar = xr[q], ai = xi[q];
        double br = xr[q + s], bi = xi[q + s];
        double cr = xr[q + 2 * s], ci = xi[q + 2 * s];
        double dr = xr[q + 3 * s], di = xi[q + 3 * s];
        double t0r = ar + cr, t0i = ai + ci, t1r = ar - cr, t1i = ai - ci;
        double t2r = br + dr, t2i = bi + di, t3r = br - dr, t3i = bi - di;
        double z0r = t0r + t2r, z0i = t0i + t2i;
        double z1r = t1r + t3i, z1i = t1i - t3r;
        double z2r = t0r - t2r, z2i = t0i - t2i;
        double z3r = t1r - t3i, z3i = t1i + t3r;
        int k0 = qout[q], k1 = qout[q + s], k2 = qout[q + 2 * s], k3 = qout[q + 3 * s];
        yd[2 * k0] = x0r + z0i;  yd[2 * k0 + 1] = x0i + z0r;
        yd[2 * k1] = x0r + z1i;  yd[2 * k1 + 1] = x0i + z1r;
        yd[2 * k2] = x0r + z2i;  yd[2 * k2 + 1] = x0i + z2r;
        yd[2 * k3] = x0r + z3i;  yd[2 * k3 + 1] = x0i + z3r;
    }
}

/* pruned exit stages: planes swapped; only k < L outputs exist; last-stage
 * twiddles are 1; the output chirp and interleave (or split store) fold in */
HOT static void st2_last_chirp(int s, const double *restrict xr, const double *restrict xi,
        const double *restrict car, const double *restrict cai,
        double *restrict yd, int L)
{
#pragma GCC ivdep
    for (int k = 0; k < L; ++k) {
        double w_i = xr[k] + xr[k + s];
        double w_r = xi[k] + xi[k + s];
        yd[2 * k]     = w_r * car[k] - w_i * cai[k];
        yd[2 * k + 1] = w_r * cai[k] + w_i * car[k];
    }
}

HOT static void st4_last_chirp(int s, const double *restrict xr, const double *restrict xi,
        const double *restrict car, const double *restrict cai,
        double *restrict yd, int L)
{
#pragma GCC ivdep
    for (int q = 0; q < s; ++q) {
        double w_i = xr[q] + xr[q + s] + xr[q + 2 * s] + xr[q + 3 * s];
        double w_r = xi[q] + xi[q + s] + xi[q + 2 * s] + xi[q + 3 * s];
        yd[2 * q]     = w_r * car[q] - w_i * cai[q];
        yd[2 * q + 1] = w_r * cai[q] + w_i * car[q];
    }
    const int n1 = L - s;
#pragma GCC ivdep
    for (int q = 0; q < n1; ++q) {
        double t1r = xr[q] - xr[q + 2 * s], t1i = xi[q] - xi[q + 2 * s];
        double t3r = xr[q + s] - xr[q + 3 * s], t3i = xi[q + s] - xi[q + 3 * s];
        double w_i = t1r + t3i;
        double w_r = t1i - t3r;
        int k = s + q;
        yd[2 * k]     = w_r * car[k] - w_i * cai[k];
        yd[2 * k + 1] = w_r * cai[k] + w_i * car[k];
    }
}

HOT static void st2_last_chirp_split(int s, const double *restrict xr, const double *restrict xi,
        const double *restrict car, const double *restrict cai,
        double *restrict zr, double *restrict zi, int L)
{
#pragma GCC ivdep
    for (int k = 0; k < L; ++k) {
        double w_i = xr[k] + xr[k + s];
        double w_r = xi[k] + xi[k + s];
        zr[k] = w_r * car[k] - w_i * cai[k];
        zi[k] = w_r * cai[k] + w_i * car[k];
    }
}

HOT static void st4_last_chirp_split(int s, const double *restrict xr, const double *restrict xi,
        const double *restrict car, const double *restrict cai,
        double *restrict zr, double *restrict zi, int L)
{
#pragma GCC ivdep
    for (int q = 0; q < s; ++q) {
        double w_i = xr[q] + xr[q + s] + xr[q + 2 * s] + xr[q + 3 * s];
        double w_r = xi[q] + xi[q + s] + xi[q + 2 * s] + xi[q + 3 * s];
        zr[q] = w_r * car[q] - w_i * cai[q];
        zi[q] = w_r * cai[q] + w_i * car[q];
    }
    const int n1 = L - s;
#pragma GCC ivdep
    for (int q = 0; q < n1; ++q) {
        double t1r = xr[q] - xr[q + 2 * s], t1i = xi[q] - xi[q + 2 * s];
        double t3r = xr[q + s] - xr[q + 3 * s], t3i = xi[q + s] - xi[q + 3 * s];
        double w_i = t1r + t3i;
        double w_r = t1i - t3r;
        int k = s + q;
        zr[k] = w_r * car[k] - w_i * cai[k];
        zi[k] = w_r * cai[k] + w_i * car[k];
    }
}

/* ---------------------------- number theory bits ---------------------------- */

static long modpow(long b, long e, long mod)
{
    long r = 1; b %= mod;
    while (e) { if (e & 1) r = r * b % mod; b = b * b % mod; e >>= 1; }
    return r;
}

static int is_prime(int n)
{
    if (n < 2) return 0;
    if (n % 2 == 0) return n == 2;
    for (long d = 3; d * d <= n; d += 2) if (n % d == 0) return 0;
    return 1;
}

static int primitive_root(int p)
{
    int pf[16], npf = 0, t = p - 1;
    for (int d = 2; (long)d * d <= t; d = (d == 2) ? 3 : d + 2)
        if (t % d == 0) { pf[npf++] = d; while (t % d == 0) t /= d; }
    if (t > 1) pf[npf++] = t;
    for (int g = 2;; ++g) {
        int ok = 1;
        for (int i = 0; i < npf; ++i)
            if (modpow(g, (p - 1) / pf[i], p) == 1) { ok = 0; break; }
        if (ok) return g;
    }
}

/* Bluestein pad: smallest 3^a 5^b 2^c >= 2L-1 with 4 | M (d1_bluestein's
 * choose_M -- 10007 -> 20480, 100003 -> 204800) */
static int choose_M(int L)
{
    long need = 2L * (long)L - 1;
    long best = -1;
    long p3 = 1;
    for (int a = 0; a <= 3; ++a, p3 *= 3) {
        long p5 = 1;
        for (int b = 0; b <= 3; ++b, p5 *= 5) {
            long base = p3 * p5, M = base;
            while (M < need || M % 4) M *= 2;
            if (best < 0 || M < best) best = M;
        }
    }
    return (int)best;
}

/* ------------------------------- the top plan ------------------------------- */

struct fft1d_plan {
    int L, batch;
    int kind;                 /* 0 direct, 1 Rader, 2 Bluestein, 3 trivial L==1 */
    core_plan core;           /* length L (kind 0), P = L-1 (kind 1), M (kind 2) */
    int dense;                /* kind 0: single generic stage -> dense_sym at B=1 */
    /* Rader */
    int P;
    int rentry, rexit;        /* fused radix-4 gather entry / scatter exit */
    int *qin, *qout;          /* qin[q] = g^q mod L ; qout[t] = g^{-t} mod L */
    /* Bluestein */
    int M;
    int fuse_last;            /* 2/4: fused pruned exit radix; 0: none */
    double *ar, *ai;          /* chirp, length L */
    double *br, *bi;          /* kernel spectrum (1/n folded), length P or M */
    /* work */
    double *s0r, *s0i, *s1r, *s1i;   /* ping-pong planes, core length */
    double *gs;               /* generic-stage scratch (per-vector) */
    /* lanes (kind 0, batch >= LANEV, L <= LANE_MAX_N) */
    int lanes;
    double *l0r, *l0i, *l1r, *l1i;   /* LANEV*L each */
    double *lgs;              /* generic-stage scratch at stride LANEV */
    /* chain state (lazy; transform-outer, so one transform's worth suffices) */
    double *pre_r, *pre_i;    /* split state, length L */
    double *tzr, *tzi;        /* one transform's split output, length L */
    double *lcr, *lci;        /* lane-blocked c field, LANEV*L */
};

const char *fft1d_name(void) { return "d1_planner"; }
const char *fft1d_description(void)
{
    return "LIBRARY LAYER (adoption-scored): factorization planner on a split-complex "
           "Stockham core (2/3/4/5/8 + generic prime<=61 sym-fold stage) -- direct smooth / "
           "unpadded Rader (smooth L-1) / smooth-padded Bluestein; 8-lane batched path; "
           "fused split-state map chains";
}

int fft1d_supports(int L) { return L >= 1; }

fft1d_plan *fft1d_create(int L, int batch)
{
    if (L < 1 || batch < 1) return NULL;
    fft1d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L; p->batch = batch;

    if (L == 1) { p->kind = 3; return p; }

    int radix[MAXSTAGE], ns;
    if (core_factor(L, radix, &ns)) {                       /* direct smooth */
        p->kind = 0;
        if (!core_init(&p->core, L)) goto fail;
        p->dense = (p->core.nstage == 1 && core_is_generic(p->core.radix[0]));
        p->s0r = amalloc((size_t)L * sizeof(double));
        p->s0i = amalloc((size_t)L * sizeof(double));
        p->s1r = amalloc((size_t)L * sizeof(double));
        p->s1i = amalloc((size_t)L * sizeof(double));
        if (!p->s0r || !p->s0i || !p->s1r || !p->s1i) goto fail;
        size_t g = core_gen_scratch(&p->core, 1);
        if (p->dense) { size_t d = 4 * (size_t)((L - 1) / 2); if (d > g) g = d; }
        if (g) { p->gs = amalloc(g * sizeof(double)); if (!p->gs) goto fail; }
        /* lanes only pay where the per-vector path runs scalar-ish: the dense
         * single-generic-stage plans (13/31/...). For multi-stage smooth plans
         * the per-vector split kernels already vectorize and the two lane
         * transposes lose (A/B wallaby: 32: 0.049 vs 0.041, 64: 0.108 vs 0.080,
         * 1024: 5.1 vs 2.4 us; but 13: 0.037 vs 0.042, 31: 0.139 vs 0.160). */
        if (batch >= LANEV && L <= LANE_MAX_N && p->dense) {
            p->lanes = 1;
            p->l0r = amalloc((size_t)LANEV * L * sizeof(double));
            p->l0i = amalloc((size_t)LANEV * L * sizeof(double));
            p->l1r = amalloc((size_t)LANEV * L * sizeof(double));
            p->l1i = amalloc((size_t)LANEV * L * sizeof(double));
            if (!p->l0r || !p->l0i || !p->l1r || !p->l1i) goto fail;
            size_t lg = core_gen_scratch(&p->core, LANEV);
            if (lg) { p->lgs = amalloc(lg * sizeof(double)); if (!p->lgs) goto fail; }
        }
        return p;
    }

    if (is_prime(L)) {                                      /* prime: try Rader */
        int P = L - 1;
        if (core_factor(P, radix, &ns)) {
            p->kind = 1;
            p->P = P;
            if (!core_init(&p->core, P)) goto fail;
            p->qin  = malloc((size_t)P * sizeof(int));
            p->qout = malloc((size_t)P * sizeof(int));
            p->br  = amalloc((size_t)P * sizeof(double));
            p->bi  = amalloc((size_t)P * sizeof(double));
            p->s0r = amalloc((size_t)P * sizeof(double));
            p->s0i = amalloc((size_t)P * sizeof(double));
            p->s1r = amalloc((size_t)P * sizeof(double));
            p->s1i = amalloc((size_t)P * sizeof(double));
            if (!p->qin || !p->qout || !p->br || !p->bi ||
                !p->s0r || !p->s0i || !p->s1r || !p->s1i) goto fail;
            size_t g = core_gen_scratch(&p->core, 1);
            if (g) { p->gs = amalloc(g * sizeof(double)); if (!p->gs) goto fail; }
            long gg = primitive_root(L), gi = modpow(gg, L - 2, L);
            long a = 1, b = 1;
            for (int q = 0; q < P; ++q) {
                p->qin[q]  = (int)a;
                p->qout[q] = (int)b;
                a = a * gg % L;  b = b * gi % L;
            }
            /* kernel b_t = exp(-2 pi i qout[t] / L); spectrum / P folded */
            for (int t = 0; t < P; ++t) {
                long double th = -2.0L * PIL * (long double)p->qout[t] / (long double)L;
                p->s0r[t] = (double)cosl(th);
                p->s0i[t] = (double)sinl(th);
            }
            double *Rr, *Ri;
            core_exec_range(&p->core, p->gs, 0, p->core.nstage, P, 1,
                            p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
            double invP = 1.0 / (double)P;
            for (int t = 0; t < P; ++t) { p->br[t] = Rr[t] * invP; p->bi[t] = Ri[t] * invP; }
            /* Fused gather-entry/scatter-exit measured a WASH on wallaby at
             * 65537 (fused 828-831 vs unfused 795-843 us, interleaved A/B):
             * the random-access fusion de-vectorizes the stage it joins, which
             * cancels the pass it saves. Kept off; worth one retry on the
             * scoring node's smaller L2 where the traffic matters more. */
            p->rentry = 0 && (p->core.nstage >= 2 && p->core.radix[0] == 4);
            p->rexit  = 0 && (p->rentry && p->core.radix[p->core.nstage - 1] == 4);
            return p;
        }
    }

    /* Bluestein */
    p->kind = 2;
    {
        int M = p->M = choose_M(L);
        if (M <= 0 || !core_init(&p->core, M)) goto fail;
        p->ar  = amalloc((size_t)L * sizeof(double));
        p->ai  = amalloc((size_t)L * sizeof(double));
        p->br  = amalloc((size_t)M * sizeof(double));
        p->bi  = amalloc((size_t)M * sizeof(double));
        p->s0r = amalloc((size_t)M * sizeof(double));
        p->s0i = amalloc((size_t)M * sizeof(double));
        p->s1r = amalloc((size_t)M * sizeof(double));
        p->s1i = amalloc((size_t)M * sizeof(double));
        if (!p->ar || !p->ai || !p->br || !p->bi ||
            !p->s0r || !p->s0i || !p->s1r || !p->s1i) goto fail;
        int lastr = p->core.radix[p->core.nstage - 1];
        p->fuse_last = (p->core.nstage >= 2 && (lastr == 2 || lastr == 4)) ? lastr : 0;
        for (long k = 0; k < L; ++k) {
            long m2 = (k * k) % (2L * (long)L);              /* exact int k^2 mod 2L */
            long double th = -PIL * (long double)m2 / (long double)L;
            p->ar[k] = (double)cosl(th); p->ai[k] = (double)sinl(th);
        }
        memset(p->s0r, 0, (size_t)M * sizeof(double));
        memset(p->s0i, 0, (size_t)M * sizeof(double));
        for (long j = 0; j < L; ++j) {
            p->s0r[j] = p->ar[j]; p->s0i[j] = -p->ai[j];
            if (j) { p->s0r[M - j] = p->s0r[j]; p->s0i[M - j] = p->s0i[j]; }
        }
        double *Rr, *Ri;
        core_exec_range(&p->core, NULL, 0, p->core.nstage, M, 1,
                        p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
        double invM = 1.0 / (double)M;
        for (int k = 0; k < M; ++k) { p->br[k] = Rr[k] * invM; p->bi[k] = Ri[k] * invM; }
        return p;
    }

fail:
    fft1d_destroy(p);
    return NULL;
}

/* --------------------------------- execute --------------------------------- */

/* direct path, one vector, interleaved in/out; entry deinterleave fused when
 * the first radix is 4 or 2, else a separate split pass */
static void direct_one(fft1d_plan *p, const double *restrict xd, double *restrict yd)
{
    const int L = p->L;
    const core_plan *c = &p->core;
    if (p->dense) { dense_sym(L, c->gC[0], c->gS[0], p->gs, xd, yd); return; }
    double *Rr, *Ri;
    const int r0 = c->radix[0], rl = c->radix[c->nstage - 1];
    const int fx = (c->nstage >= 2 && !core_is_generic(rl));   /* fused interleave exit */
    const int last = fx ? c->nstage - 1 : c->nstage;
    if (r0 == 4) {
        st4_first_deint(L / 4, c->twr[0], c->twi[0], xd, p->s0r, p->s0i);
        core_exec_range(c, p->gs, 1, last, L / 4, 4,
                        p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
    } else if (r0 == 2) {
        st2_first_deint(L / 2, c->twr[0], c->twi[0], xd, p->s0r, p->s0i);
        core_exec_range(c, p->gs, 1, last, L / 2, 2,
                        p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
    } else {
        double *restrict sr = p->s0r, *restrict si = p->s0i;
#pragma GCC ivdep
        for (int t = 0; t < L; ++t) { sr[t] = xd[2 * t]; si[t] = xd[2 * t + 1]; }
        core_exec_range(c, p->gs, 0, last, L, 1,
                        p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
    }
    if (fx) {
        const int s = L / rl;
        switch (rl) {
        case 2: st2_last_int(s, Rr, Ri, yd); break;
        case 3: st3_last_int(s, Rr, Ri, yd); break;
        case 4: st4_last_int(s, Rr, Ri, yd); break;
        case 5: st5_last_int(s, Rr, Ri, yd); break;
        default: st8_last_int(s, Rr, Ri, yd); break;
        }
        return;
    }
    const double *restrict rr = Rr, *restrict ri = Ri;
#pragma GCC ivdep
    for (int k = 0; k < L; ++k) { yd[2 * k] = rr[k]; yd[2 * k + 1] = ri[k]; }
}

/* Rader inverse half: from the forward spectrum R (living in one scratch pair)
 * to the conv result V (swapped planes: conv_r = (*Vi)[t], conv_i = (*Vr)[t]).
 * fused_exit != 0 stops one stage early for st4_last_scatter to finish. */
static void rader_inv(fft1d_plan *p, double *Rr, double *Ri, int fused_exit,
                      double **Vr, double **Vi)
{
    const core_plan *c = &p->core;
    const int P = p->P, nst = c->nstage;
    double *dr = (Rr == p->s0r) ? p->s1r : p->s0r;
    double *di = (Rr == p->s0r) ? p->s1i : p->s0i;
    double *er = (dr == p->s0r) ? p->s1r : p->s0r;
    double *ei = (dr == p->s0r) ? p->s1i : p->s0i;
    if (c->radix[0] == 4) {
        st4_first_bhat(P / 4, c->twr[0], c->twi[0], Rr, Ri, p->br, p->bi, dr, di);
        core_exec_range(c, p->gs, 1, fused_exit ? nst - 1 : nst, P / 4, 4,
                        dr, di, er, ei, Vr, Vi);
    } else {
        const double *restrict rr = Rr, *restrict ri = Ri;
        const double *restrict br = p->br, *restrict bi = p->bi;
        double *restrict zr = dr, *restrict zi = di;
#pragma GCC ivdep
        for (int t = 0; t < P; ++t) {          /* swapped product */
            zr[t] = rr[t] * bi[t] + ri[t] * br[t];
            zi[t] = rr[t] * br[t] - ri[t] * bi[t];
        }
        core_exec_range(c, p->gs, 0, nst, P, 1, dr, di, er, ei, Vr, Vi);
    }
}

static void rader_one(fft1d_plan *p, const double *restrict xd, double *restrict yd)
{
    const int P = p->P;
    const core_plan *c = &p->core;
    const int nst = c->nstage;
    const int *restrict qin = p->qin, *restrict qout = p->qout;
    double sr, si;
    double *Rr, *Ri;
    if (p->rentry) {
        st4_first_gather(P / 4, c->twr[0], c->twi[0], xd, qin,
                         p->s0r, p->s0i, &sr, &si);
        core_exec_range(c, p->gs, 1, nst, P / 4, 4,
                        p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
    } else {
        double *restrict gr = p->s0r, *restrict gi = p->s0i;
        sr = 0.0; si = 0.0;
        for (int q = 0; q < P; ++q) {
            int j = qin[q];
            double xr = xd[2 * j], xi = xd[2 * j + 1];
            gr[q] = xr; gi[q] = xi;
            sr += xr; si += xi;
        }
        core_exec_range(c, p->gs, 0, nst, P, 1,
                        p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
    }
    const double x0r = xd[0], x0i = xd[1];
    double *Vr, *Vi;
    rader_inv(p, Rr, Ri, p->rexit, &Vr, &Vi);
    yd[0] = x0r + sr; yd[1] = x0i + si;
    if (p->rexit) {
        st4_last_scatter(P / 4, Vr, Vi, qout, x0r, x0i, yd);
    } else {
        const double *restrict cr = Vi, *restrict ci = Vr;   /* unswap */
        for (int t = 0; t < P; ++t) {
            int k = qout[t];
            yd[2 * k]     = x0r + cr[t];
            yd[2 * k + 1] = x0i + ci[t];
        }
    }
}

/* Bluestein middle (from d1_bluestein): forward stages 1.., fused kernel-mul
 * inverse entry, inverse stages 1..(nst-1 if fused exit) */
static void bl_middle(fft1d_plan *p, double **Vr, double **Vi)
{
    const core_plan *c = &p->core;
    const int nst = c->nstage, m0 = p->M / 4;
    double *Rr, *Ri;
    core_exec_range(c, NULL, 1, nst, m0, 4, p->s1r, p->s1i, p->s0r, p->s0i, &Rr, &Ri);
    double *dr = (Rr == p->s0r) ? p->s1r : p->s0r;
    double *di = (Rr == p->s0r) ? p->s1i : p->s0i;
    st4_first_bhat(m0, c->twr[0], c->twi[0], Rr, Ri, p->br, p->bi, dr, di);
    double *er = (dr == p->s0r) ? p->s1r : p->s0r;
    double *ei = (dr == p->s0r) ? p->s1i : p->s0i;
    core_exec_range(c, NULL, 1, p->fuse_last ? nst - 1 : nst, m0, 4, dr, di, er, ei, Vr, Vi);
}

static void bl_one(fft1d_plan *p, const double *restrict xd, double *restrict yd)
{
    const int L = p->L, M = p->M;
    const core_plan *c = &p->core;
    st4_first_chirp(M / 4, L, c->twr[0], c->twi[0], xd, p->ar, p->ai, p->s1r, p->s1i);
    double *Vr, *Vi;
    bl_middle(p, &Vr, &Vi);
    if (p->fuse_last == 2)
        st2_last_chirp(M / 2, Vr, Vi, p->ar, p->ai, yd, L);
    else if (p->fuse_last == 4)
        st4_last_chirp(M / 4, Vr, Vi, p->ar, p->ai, yd, L);
    else {
        const double *restrict wr = Vi, *restrict wi = Vr;
        const double *restrict car = p->ar, *restrict cai = p->ai;
#pragma GCC ivdep
        for (int k = 0; k < L; ++k) {
            yd[2 * k]     = wr[k] * car[k] - wi[k] * cai[k];
            yd[2 * k + 1] = wr[k] * cai[k] + wi[k] * car[k];
        }
    }
}

/* lane gather/scatter: 8 interleaved vectors (stride 2L doubles) <-> lane-
 * blocked split planes T[e*8+v] */
HOT static void lane_gather(const double *restrict xd, int L,
                            double *restrict lr, double *restrict li)
{
    for (int v = 0; v < LANEV; ++v) {
        const double *restrict x = xd + (size_t)v * 2 * L;
#pragma GCC ivdep
        for (int e = 0; e < L; ++e) {
            lr[(size_t)e * LANEV + v] = x[2 * e];
            li[(size_t)e * LANEV + v] = x[2 * e + 1];
        }
    }
}

HOT static void lane_scatter(const double *restrict lr, const double *restrict li,
                             int L, double *restrict yd)
{
    for (int v = 0; v < LANEV; ++v) {
        double *restrict y = yd + (size_t)v * 2 * L;
#pragma GCC ivdep
        for (int e = 0; e < L; ++e) {
            y[2 * e]     = lr[(size_t)e * LANEV + v];
            y[2 * e + 1] = li[(size_t)e * LANEV + v];
        }
    }
}

void fft1d_execute(fft1d_plan *p, const double _Complex *cin, double _Complex *cout)
{
    const double *in = (const double *)cin;
    double *out = (double *)cout;
    const int L = p->L;
    const size_t stride = 2 * (size_t)L;
    if (p->kind == 3) { memcpy(out, in, (size_t)p->batch * stride * sizeof(double)); return; }
    int b = 0;
    if (p->kind == 0 && p->lanes) {
        for (; b + LANEV <= p->batch; b += LANEV) {
            lane_gather(in + (size_t)b * stride, L, p->l0r, p->l0i);
            double *Rr, *Ri;
            core_exec_range(&p->core, p->lgs, 0, p->core.nstage, L, LANEV,
                            p->l0r, p->l0i, p->l1r, p->l1i, &Rr, &Ri);
            lane_scatter(Rr, Ri, L, out + (size_t)b * stride);
        }
    }
    for (; b < p->batch; ++b) {
        const double *x = in + (size_t)b * stride;
        double *y = out + (size_t)b * stride;
        switch (p->kind) {
        case 0:  direct_one(p, x, y); break;
        case 1:  rader_one(p, x, y);  break;
        default: bl_one(p, x, y);     break;
        }
    }
}

/* ------------------------------- fused chain --------------------------------
 * state <- (FFT(state) + c) / (1 + |FFT(state) + c|), m steps, transform-outer
 * so one transform's state stays cache-resident for its whole chain. The state
 * lives SPLIT between steps; interleaved output only at the final step.
 * (Bluestein: state lives chirp-premultiplied -- d1_bluestein's scheme.) */

void fft1d_chain(fft1d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    const int L = p->L, B = p->batch;
    const double *cin = (const double *)c;

    if (p->kind == 3) {                      /* L == 1: FFT is identity */
        for (int b = 0; b < B; ++b) {
            double re = ((const double *)x0)[2 * b], im = ((const double *)x0)[2 * b + 1];
            double cr = cin[2 * b], ci = cin[2 * b + 1];
            for (int s = 0; s < m; ++s) {
                double zr = re + cr, zi = im + ci;
                double sc = 1.0 / (1.0 + sqrt(zr * zr + zi * zi));
                re = zr * sc; im = zi * sc;
            }
            ((double *)final_out)[2 * b] = re;
            ((double *)final_out)[2 * b + 1] = im;
        }
        return;
    }

    if (!p->tzr) {
        p->tzr = amalloc((size_t)L * sizeof(double));
        p->tzi = amalloc((size_t)L * sizeof(double));
    }
    if (!p->pre_r) {
        p->pre_r = amalloc((size_t)L * sizeof(double));
        p->pre_i = amalloc((size_t)L * sizeof(double));
    }
    if (p->kind == 0 && p->lanes && !p->lcr) {
        p->lcr = amalloc((size_t)LANEV * L * sizeof(double));
        p->lci = amalloc((size_t)LANEV * L * sizeof(double));
    }
    if (!p->tzr || !p->tzi || !p->pre_r || !p->pre_i ||
        (p->kind == 0 && p->lanes && (!p->lcr || !p->lci))) {
        /* allocation failed (never in practice): unfused ping-pong fallback */
        const size_t count = (size_t)L * B;
        double _Complex *tmp = amalloc(count * sizeof(double _Complex));
        if (!tmp) return;
        memcpy(final_out, x0, count * sizeof(double _Complex));
        for (int s = 0; s < m; ++s) {
            fft1d_execute(p, final_out, tmp);
            const double *restrict zr = (const double *)tmp;
            double *restrict o = (double *)final_out;
            for (size_t i = 0; i < count; ++i) {
                double re = zr[2 * i] + cin[2 * i];
                double im = zr[2 * i + 1] + cin[2 * i + 1];
                double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                o[2 * i] = re * sc;
                o[2 * i + 1] = im * sc;
            }
        }
        free(tmp);
        return;
    }

    if (p->kind == 2) {
        /* Bluestein: chirp-premultiplied split state (ported from d1_bluestein,
         * reordered transform-outer) */
        const int M = p->M, m0 = M / 4;
        const core_plan *cp = &p->core;
        const double *restrict car = p->ar, *restrict cai = p->ai;
        for (int b = 0; b < B; ++b) {
            const double *restrict xd = (const double *)(x0 + (size_t)b * L);
            const double *restrict cd = cin + (size_t)b * 2 * L;
            double *restrict pr = p->pre_r;
            double *restrict pi = p->pre_i;
#pragma GCC ivdep
            for (int k = 0; k < L; ++k) {
                double xr = xd[2 * k], xi = xd[2 * k + 1];
                pr[k] = xr * car[k] - xi * cai[k];
                pi[k] = xr * cai[k] + xi * car[k];
            }
            for (int s = 0; s < m; ++s) {
                st4_first_pre(m0, L, cp->twr[0], cp->twi[0], pr, pi, p->s1r, p->s1i);
                double *Vr, *Vi;
                bl_middle(p, &Vr, &Vi);
                if (p->fuse_last == 2)
                    st2_last_chirp_split(M / 2, Vr, Vi, car, cai, p->tzr, p->tzi, L);
                else if (p->fuse_last == 4)
                    st4_last_chirp_split(M / 4, Vr, Vi, car, cai, p->tzr, p->tzi, L);
                else {
                    const double *restrict wr = Vi, *restrict wi = Vr;
                    double *restrict tzr = p->tzr, *restrict tzi = p->tzi;
#pragma GCC ivdep
                    for (int k = 0; k < L; ++k) {
                        tzr[k] = wr[k] * car[k] - wi[k] * cai[k];
                        tzi[k] = wr[k] * cai[k] + wi[k] * car[k];
                    }
                }
                const double *restrict tzr = p->tzr, *restrict tzi = p->tzi;
                if (s == m - 1) {
                    double *restrict od = (double *)(final_out + (size_t)b * L);
#pragma GCC ivdep
                    for (int k = 0; k < L; ++k) {
                        double re = tzr[k] + cd[2 * k], im = tzi[k] + cd[2 * k + 1];
                        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                        od[2 * k] = re * sc;
                        od[2 * k + 1] = im * sc;
                    }
                } else {
#pragma GCC ivdep
                    for (int k = 0; k < L; ++k) {
                        double re = tzr[k] + cd[2 * k], im = tzi[k] + cd[2 * k + 1];
                        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                        double gr = re * sc, gi = im * sc;
                        pr[k] = gr * car[k] - gi * cai[k];
                        pi[k] = gr * cai[k] + gi * car[k];
                    }
                }
            }
        }
        return;
    }

    if (p->kind == 1) {
        /* Rader: split natural-order state; gather feeds the conv directly */
        const int P = p->P;
        const int *restrict qin = p->qin, *restrict qout = p->qout;
        for (int b = 0; b < B; ++b) {
            const double *restrict xd = (const double *)(x0 + (size_t)b * L);
            const double *restrict cd = cin + (size_t)b * 2 * L;
            double *restrict str = p->pre_r;
            double *restrict sti = p->pre_i;
#pragma GCC ivdep
            for (int k = 0; k < L; ++k) { str[k] = xd[2 * k]; sti[k] = xd[2 * k + 1]; }
            for (int s = 0; s < m; ++s) {
                const double x0r = str[0], x0i = sti[0];
                double sr, si;
                double *Rr, *Ri;
                const core_plan *cc = &p->core;
                if (p->rentry) {
                    st4_first_gather_split(P / 4, cc->twr[0], cc->twi[0],
                                           str, sti, qin, p->s0r, p->s0i, &sr, &si);
                    core_exec_range(cc, p->gs, 1, cc->nstage, P / 4, 4,
                                    p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
                } else {
                    double *restrict gr = p->s0r, *restrict gi = p->s0i;
                    sr = 0.0; si = 0.0;
                    for (int q = 0; q < P; ++q) {
                        int j = qin[q];
                        double xr = str[j], xi = sti[j];
                        gr[q] = xr; gi[q] = xi;
                        sr += xr; si += xi;
                    }
                    core_exec_range(cc, p->gs, 0, cc->nstage, P, 1,
                                    p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
                }
                double *Vr, *Vi;
                rader_inv(p, Rr, Ri, 0, &Vr, &Vi);
                double *restrict tzr = p->tzr, *restrict tzi = p->tzi;
                tzr[0] = x0r + sr; tzi[0] = x0i + si;
                const double *restrict cvr = Vi, *restrict cvi = Vr;
                for (int t = 0; t < P; ++t) {
                    int k = qout[t];
                    tzr[k] = x0r + cvr[t];
                    tzi[k] = x0i + cvi[t];
                }
                if (s == m - 1) {
                    double *restrict od = (double *)(final_out + (size_t)b * L);
#pragma GCC ivdep
                    for (int k = 0; k < L; ++k) {
                        double re = tzr[k] + cd[2 * k], im = tzi[k] + cd[2 * k + 1];
                        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                        od[2 * k] = re * sc;
                        od[2 * k + 1] = im * sc;
                    }
                } else {
#pragma GCC ivdep
                    for (int k = 0; k < L; ++k) {
                        double re = tzr[k] + cd[2 * k], im = tzi[k] + cd[2 * k + 1];
                        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                        str[k] = re * sc;
                        sti[k] = im * sc;
                    }
                }
            }
        }
        return;
    }

    /* kind 0: direct. Lane groups first (state + c lane-blocked, whole chain
     * L1/L2-resident per group), then per-vector remainder with split state. */
    int b = 0;
    if (p->lanes) {
        for (; b + LANEV <= B; b += LANEV) {
            lane_gather((const double *)(x0 + (size_t)b * L), L, p->l0r, p->l0i);
            lane_gather(cin + (size_t)b * 2 * L, L, p->lcr, p->lci);
            double *sr_ = p->l0r, *si_ = p->l0i;
            for (int s = 0; s < m; ++s) {
                double *Rr, *Ri;
                double *ar = sr_, *ai = si_;
                double *br_ = (ar == p->l0r) ? p->l1r : p->l0r;
                double *bi_ = (ai == p->l0i) ? p->l1i : p->l0i;
                core_exec_range(&p->core, p->lgs, 0, p->core.nstage, L, LANEV,
                                ar, ai, br_, bi_, &Rr, &Ri);
                const double *restrict lcr = p->lcr, *restrict lci = p->lci;
                double *rr = Rr, *ri = Ri;
#pragma GCC ivdep
                for (int e = 0; e < L * LANEV; ++e) {
                    double re = rr[e] + lcr[e], im = ri[e] + lci[e];
                    double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                    rr[e] = re * sc;
                    ri[e] = im * sc;
                }
                sr_ = Rr; si_ = Ri;
            }
            lane_scatter(sr_, si_, L, (double *)(final_out + (size_t)b * L));
        }
    }
    for (; b < B; ++b) {
        const double *restrict xd = (const double *)(x0 + (size_t)b * L);
        const double *restrict cd = cin + (size_t)b * 2 * L;
        /* state ping-pongs through the scratch planes; the map runs in place on
         * the FFT result (same-index elementwise), so no state copy per step */
        double *ar = p->s0r, *ai = p->s0i, *br_ = p->s1r, *bi_ = p->s1i;
        {
            double *restrict sr_ = ar, *restrict si_ = ai;
#pragma GCC ivdep
            for (int k = 0; k < L; ++k) { sr_[k] = xd[2 * k]; si_[k] = xd[2 * k + 1]; }
        }
        for (int s = 0; s < m; ++s) {
            double *Rr, *Ri;
            if (p->dense) {
                stg(L, 1, 1, p->core.twr[0], p->core.twi[0],
                    p->core.gC[0], p->core.gS[0], p->gs, ar, ai, br_, bi_);
                Rr = br_; Ri = bi_;
            } else {
                core_exec_range(&p->core, p->gs, 0, p->core.nstage, L, 1,
                                ar, ai, br_, bi_, &Rr, &Ri);
            }
            double *rr = Rr, *ri = Ri;
            if (s == m - 1) {
                double *restrict od = (double *)(final_out + (size_t)b * L);
#pragma GCC ivdep
                for (int k = 0; k < L; ++k) {
                    double re = rr[k] + cd[2 * k], im = ri[k] + cd[2 * k + 1];
                    double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                    od[2 * k] = re * sc;
                    od[2 * k + 1] = im * sc;
                }
            } else {
#pragma GCC ivdep
                for (int k = 0; k < L; ++k) {
                    double re = rr[k] + cd[2 * k], im = ri[k] + cd[2 * k + 1];
                    double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                    rr[k] = re * sc;
                    ri[k] = im * sc;
                }
                ar = Rr; ai = Ri;
                br_ = (ar == p->s0r) ? p->s1r : p->s0r;
                bi_ = (ai == p->s0i) ? p->s1i : p->s0i;
            }
        }
    }
}

void fft1d_destroy(fft1d_plan *p)
{
    if (!p) return;
    core_free(&p->core);
    free(p->qin); free(p->qout);
    free(p->ar); free(p->ai); free(p->br); free(p->bi);
    free(p->s0r); free(p->s0i); free(p->s1r); free(p->s1i);
    free(p->gs);
    free(p->l0r); free(p->l0i); free(p->l1r); free(p->l1i); free(p->lgs);
    free(p->pre_r); free(p->pre_i); free(p->tzr); free(p->tzi);
    free(p->lcr); free(p->lci);
    free(p);
}

/* --------------------------------- self test -------------------------------- */
#ifdef PLANNER_TEST
#include <stdio.h>

static void naive_dft(int n, const double *x, double *y)
{
    for (int k = 0; k < n; ++k) {
        double sr = 0, si = 0;
        for (int j = 0; j < n; ++j) {
            long ph = ((long)j * k) % n;
            double th = -2.0 * M_PI * (double)ph / (double)n;
            double wr = cos(th), wi = sin(th);
            sr += x[2*j] * wr - x[2*j+1] * wi;
            si += x[2*j] * wi + x[2*j+1] * wr;
        }
        y[2*k] = sr; y[2*k+1] = si;
    }
}

int main(void)
{
    int sizes[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 24, 31, 32, 36, 45,
                    49, 60, 61, 64, 77, 100, 120, 127, 128, 240, 243, 256, 343,
                    510, 512, 1000, 1013, 1019, 1020, 1021, 1024, 1153, 2038, 2048, 2053,
#ifdef PLANNER_TEST_BIG
                    4096, 10007, 16384, 65537,
#endif
    };
    srand(12345);
    int nfail = 0;
    for (unsigned si = 0; si < sizeof sizes / sizeof *sizes; ++si) {
        int n = sizes[si], B = 19;   /* 2 lane groups + remainder of 3 */
        double _Complex *in  = malloc((size_t)n * B * sizeof *in);
        double _Complex *out = malloc((size_t)n * B * sizeof *out);
        double *ref = malloc(2 * (size_t)n * sizeof *ref);
        for (int i = 0; i < n * B; ++i)
            in[i] = (2.0 * rand() / RAND_MAX - 1.0) + (2.0 * rand() / RAND_MAX - 1.0) * I;
        fft1d_plan *p = fft1d_create(n, B);
        if (!p) { printf("n=%5d  CREATE FAILED\n", n); ++nfail; continue; }
        fft1d_execute(p, in, out);
        double worst = 0;
        for (int b = 0; b < B; ++b) {
            naive_dft(n, (const double *)(in + (size_t)b * n), ref);
            double e2 = 0, r2 = 0;
            for (int i = 0; i < n; ++i) {
                double dr = creal(out[(size_t)b*n+i]) - ref[2*i];
                double di = cimag(out[(size_t)b*n+i]) - ref[2*i+1];
                e2 += dr*dr + di*di;
                r2 += ref[2*i]*ref[2*i] + ref[2*i+1]*ref[2*i+1];
            }
            double rel = sqrt(e2 / (r2 > 0 ? r2 : 1));
            if (rel > worst) worst = rel;
        }
        /* chain smoke test: fft1d_chain vs execute+map, m=3 */
        double _Complex *cf  = malloc((size_t)n * B * sizeof *cf);
        double _Complex *st  = malloc((size_t)n * B * sizeof *st);
        double _Complex *tmp = malloc((size_t)n * B * sizeof *tmp);
        double _Complex *ch  = malloc((size_t)n * B * sizeof *ch);
        for (int i = 0; i < n * B; ++i)
            cf[i] = 0.1 * ((2.0 * rand() / RAND_MAX - 1.0) + (2.0 * rand() / RAND_MAX - 1.0) * I);
        memcpy(st, in, (size_t)n * B * sizeof *st);
        for (int s = 0; s < 3; ++s) {
            fft1d_execute(p, st, tmp);
            for (int i = 0; i < n * B; ++i) {
                double _Complex z = tmp[i] + cf[i];
                st[i] = z / (1.0 + cabs(z));
            }
        }
        fft1d_chain(p, in, cf, ch, 3);
        double ce2 = 0, cr2 = 0;
        for (int i = 0; i < n * B; ++i) {
            double _Complex d = ch[i] - st[i];
            ce2 += creal(d)*creal(d) + cimag(d)*cimag(d);
            cr2 += creal(st[i])*creal(st[i]) + cimag(st[i])*cimag(st[i]);
        }
        double crel = sqrt(ce2 / (cr2 > 0 ? cr2 : 1));
        const char *k = p->kind == 0 ? "direct" : p->kind == 1 ? "rader " :
                        p->kind == 2 ? "blues " : "triv  ";
        printf("n=%5d  %s relL2=%.3e chain=%.3e  %s\n", n, k, worst, crel,
               (worst < 1e-12 && crel < 1e-11) ? "ok" : "FAIL");
        if (worst >= 1e-12 || crel >= 1e-11) ++nfail;
        fft1d_destroy(p);
        free(in); free(out); free(ref); free(cf); free(st); free(tmp); free(ch);
    }
    printf(nfail ? "== %d FAILURES ==\n" : "== all ok ==\n", nfail);
    return nfail != 0;
}
#endif
