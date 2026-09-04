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
 * ROUND r4: (1) the stage kernels are now NOINLINE so their 512-bit-preference
 * target attribute actually survives (d1_bluestein's r3 discovery; the whole r3
 * engine had silently scored at ymm width -- every cell gained 3-23% on the
 * node); (2) L=13/31 run d1_prime's r3 interleaved-pair zmm kernels and their
 * r1 A/B-row + SoA-block fused chains (3-6x on all eight 13/31 cells); (3)
 * L=32/64 run d1_pow2's r1 in-register codelets and register-resident chains
 * (2.5-3.3x); (4) fused exits NT-stream when in+out exceeds the node's L3
 * (d1_pow2 r3); (5) lanes serve the batched CHAIN (transpose amortizes over m)
 * for small multi-stage smooth sizes, never their execute.
 *
 * ROUND r7: (1) chain map fused into every step's FINAL stage (m=1, unit
 * twiddles, contiguous outputs -> zero-shuffle map fusion; st{2,4,5,8,16,64}
 * _last_map, swapped-plane st4_last_map_rader for the conv-order Rader chain,
 * map_chirp_split for the Bluestein interior) -- the per-step map pass is
 * gone everywhere; (2) tile EXITS st{16,64}_last_int(+NT): 4096=[4,64,16],
 * 16384=[4,64,64] run 3 passes; (3) lane chains get 2-pass chain-only
 * schedules (64=[8,8], 128=[16,8]) and take the 60/64 batched chains; (4)
 * Goldschmidt + early-seeded-rcp map (d1_prime r5 via d1_batchlane r6); (5)
 * arena rounded to whole 2MB pages + pre-faulted (d1_pow2's r6 THP finding:
 * the old arena got 0 huge pages at 16384, partial above).
 *
 * Compile-time self test:  gcc -DPLANNER_TEST -O2 d1_planner.c -lm
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <sys/mman.h>
#include "../fft1d_api.h"

/* The graded build's flags leave gcc at 256-bit vector preference; these
   shuffle-free split loops want zmm. Per-function attribute only -- a file-wide
   pragma resets the ISA (d1_bluestein's hard-won lesson). NOINLINE IS LOAD-
   BEARING (r4, from d1_bluestein's r3 finding): when a target()-attributed
   kernel inlines into a caller without the attribute, gcc silently drops the
   512-bit preference -- objdump of the r3 build showed core_exec_range at
   0 zmm / 846 ymm, i.e. the whole engine scored r3 at half width. */
#define HOT __attribute__((noinline, target("arch=icelake-server,prefer-vector-width=512")))

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

/* Deterministic plane layout for the big plans -- ADOPTED FROM d1_bluestein
 * (r2): separate posix_memalign(64) planes land at the same offset mod 4K, and
 * with 4K pages the L2 set index rides on random physical page coloring, so
 * multi-plane streaming plans are BIMODAL across invocations (their 10007
 * measured 110 or 213 us per process, plan-time luck). Fix: all work planes in
 * ONE 2MB-aligned block under MADV_HUGEPAGE (physical set indexing then follows
 * the virtual layout), plane stride rounded up to the 128 KB L2-way period plus
 * a 32KB+192B skew so equal-length planes never share L1/L2 sets. */
typedef struct { char *base; size_t off; } carver;

static size_t carve_stride(size_t nd)
{
    size_t bytes = nd * sizeof(double);
    return ((bytes + 131071) & ~(size_t)131071) + 32768 + 192;
}

static double *carve(carver *cv, size_t nd)
{
    double *p = (double *)(cv->base + cv->off);
    cv->off += carve_stride(nd);
    return p;
}

static void *arena_alloc(size_t bytes)
{
    /* r7, after d1_pow2's r6 finding (returned to the arena's adopters): THP
     * can only back FULL 2MB-aligned, 2MB-sized subranges. The r3-r6 arena was
     * 2MB-aligned but odd-sized, so its tail -- and at 16384 the WHOLE 1.3MB
     * arena -- got zero huge pages (verified: AnonHugePages=0/partial on the
     * node). Round the size to whole huge pages and pre-fault at plan time so
     * the backing (and the set-skew layout it enables) is deterministic. */
    void *p = NULL;
    bytes = (bytes + ((size_t)2 << 20) - 1) & ~(((size_t)2 << 20) - 1);
    if (posix_memalign(&p, 2 * 1024 * 1024, bytes) != 0) return NULL;
#ifdef MADV_HUGEPAGE
    madvise(p, bytes, MADV_HUGEPAGE);
#endif
    memset(p, 0, bytes);
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

static int core_is_generic(int r) { return r != 2 && r != 3 && r != 4 && r != 5 &&
                                           r != 8 && r != 16 && r != 64; }

/* Hand-picked schedules with radix-16/64 L1-tile stages (r6, kernels ADOPTED
 * FROM d1_rader r3/r5 via d1_bluestein r5's flavor verdicts). Entry stays 4
 * and the tail stays 2/4, so every fused entry / mid / pruned-exit path in
 * this file applies unchanged; the tile stages only replace radix-8 middles.
 * Donor-verified rows: 20480 = [4,16,16,5,4] (d1_bluestein r5, -11.5% at
 * 10007), 65536 = [4,64,64,4] (d1_rader r5, 7 M-passes instead of 9).
 * 1024 = [4,64,4] was TRIED and dropped (node interleaved A/B: wash-to-worse,
 * 1.89-1.97 vs 1.83-2.00 B=1, 2.19+ vs 2.14 B=512) -- d1_bluestein's finding
 * that L2-resident 1024 gains nothing from a tile stage holds on this core
 * too.  PLN_SCHED=0 restores the plain radix-8 factorization for A/B. */
static int core_schedule(int n, int *radix, int *nstage)
{
    /* r7: the direct pow2 rows drop the radix-4 tail -- st{16,64}_last_int/
     * _map make the TILE the final stage, so 4096/16384 run THREE passes.
     * Conv rows (65536/20480/204800) keep their 2/4 tails: the Rader/Bluestein
     * mid and exit fusions require them. PLN_SCHED=0 -> plain factorization,
     * PLN_SCHED=2 -> the r6 rows (radix-4 tails) for A/B. */
    struct sched_row { int n, ns, r[6]; };
    /* 1024 = [4,16,16] TRIED r7 and dropped: B=1 wash-to-slightly-better
     * (1.57-1.62 vs 1.62-1.65) but B=512 regresses (2.12-2.63 vs 2.12-2.36)
     * and the chain is a wash -- d1_bluestein's "L2-resident 1024 gains
     * nothing from a tile stage" verdict holds for the 16-tile too. */
    static const struct sched_row tab[] = {
        { 4096,   3, {4, 64, 16} },
        { 16384,  3, {4, 64, 64} },
        { 65536,  4, {4, 64, 64, 4} },
        { 20480,  5, {4, 16, 16, 5, 4} },
        { 204800, 6, {4, 64, 16, 5, 5, 2} },
    };
    static const struct sched_row tab6[] = {
        { 4096,   4, {4, 16, 16, 4} },
        { 16384,  4, {4, 64, 16, 4} },
        { 65536,  4, {4, 64, 64, 4} },
        { 20480,  5, {4, 16, 16, 5, 4} },
        { 204800, 6, {4, 64, 16, 5, 5, 2} },
    };
    static int mode = -1;
    if (mode < 0) { const char *e = getenv("PLN_SCHED"); mode = e ? atoi(e) : 1; }
    if (mode == 0) return 0;
    const struct sched_row *t = (mode == 2) ? tab6 : tab;
    for (unsigned i = 0; i < sizeof tab / sizeof tab[0]; ++i)
        if (t[i].n == n) {
            for (int k = 0; k < t[i].ns; ++k) radix[k] = t[i].r[k];
            *nstage = t[i].ns;
            return 1;
        }
    return 0;
}

static void core_free(core_plan *c)
{
    free(c->twstore);
    for (int st = 0; st < c->nstage; ++st) { free(c->gC[st]); free(c->gS[st]); }
    memset(c, 0, sizeof *c);
}

static void w64_init(void);

static int core_init_with(core_plan *c, int n, const int *radix, int ns)
{
    memset(c, 0, sizeof *c);
    c->n = n;
    c->nstage = ns;
    for (int st = 0; st < ns; ++st) c->radix[st] = radix[st];
    for (int st = 0; st < c->nstage; ++st)
        if (c->radix[st] == 64) { w64_init(); break; }
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

static int core_init(core_plan *c, int n)
{
    int radix[MAXSTAGE], ns;
    if (!core_schedule(n, radix, &ns) && !core_factor(n, radix, &ns)) return 0;
    return core_init_with(c, n, radix, ns);
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

/* radix-8 at s == 4, two p-groups per zmm -- ADOPTED FROM d1_bluestein (r4
 * st8_s4, near-verbatim), itself d1_rader's r3 st16_s4/st3_s4 paired-p trick
 * applied to radix 8. Every plan here with >= 5 factors of 2 runs its SECOND
 * stage as radix-8 at s=4 (1024/4096/16384, conv 65536, Bluestein pads 20480/
 * 204800, and 128 = [4,8,4]): the q-loop is 4-wide there with per-p scalar
 * twiddle loads, measured ~2x ideal cost by d1_bluestein's PMU work. Two
 * p-groups are 8 contiguous doubles per input stream; twiddles pair-broadcast
 * [w(p) x4 | w(p+1) x4]; outputs recombine to full contiguous zmm stores via
 * 128-bit-lane shuffles -- no masked half-stores. Plain target attribute (not
 * HOT): intrinsics + arch= form do not mix (the file's r3 lesson). */
__attribute__((target("avx512f")))
static void st8_s4(int m, const double *restrict wr, const double *restrict wi,
                   const double *restrict xr, const double *restrict xi,
                   double *restrict yr, double *restrict yi)
{
    const __m512d C8 = _mm512_set1_pd(0.70710678118654752440);
    const __m512i PAIR = _mm512_setr_epi64(0, 0, 0, 0, 1, 1, 1, 1);
#define TWPAIR(base) _mm512_permutexvar_pd(PAIR, \
        _mm512_castpd128_pd512(_mm_loadu_pd(base)))
    for (int p = 0; p < m; p += 2) {
        const size_t mm = (size_t)m;
        __m512d a0r = _mm512_loadu_pd(xr + 4 * p);
        __m512d a0i = _mm512_loadu_pd(xi + 4 * p);
        __m512d a1r = _mm512_loadu_pd(xr + 4 * p + 4 * mm);
        __m512d a1i = _mm512_loadu_pd(xi + 4 * p + 4 * mm);
        __m512d a2r = _mm512_loadu_pd(xr + 4 * p + 8 * mm);
        __m512d a2i = _mm512_loadu_pd(xi + 4 * p + 8 * mm);
        __m512d a3r = _mm512_loadu_pd(xr + 4 * p + 12 * mm);
        __m512d a3i = _mm512_loadu_pd(xi + 4 * p + 12 * mm);
        __m512d a4r = _mm512_loadu_pd(xr + 4 * p + 16 * mm);
        __m512d a4i = _mm512_loadu_pd(xi + 4 * p + 16 * mm);
        __m512d a5r = _mm512_loadu_pd(xr + 4 * p + 20 * mm);
        __m512d a5i = _mm512_loadu_pd(xi + 4 * p + 20 * mm);
        __m512d a6r = _mm512_loadu_pd(xr + 4 * p + 24 * mm);
        __m512d a6i = _mm512_loadu_pd(xi + 4 * p + 24 * mm);
        __m512d a7r = _mm512_loadu_pd(xr + 4 * p + 28 * mm);
        __m512d a7i = _mm512_loadu_pd(xi + 4 * p + 28 * mm);
        /* even DFT4 on (a0,a2,a4,a6) */
        __m512d t0r = _mm512_add_pd(a0r, a4r), t0i = _mm512_add_pd(a0i, a4i);
        __m512d t1r = _mm512_sub_pd(a0r, a4r), t1i = _mm512_sub_pd(a0i, a4i);
        __m512d t2r = _mm512_add_pd(a2r, a6r), t2i = _mm512_add_pd(a2i, a6i);
        __m512d t3r = _mm512_sub_pd(a2r, a6r), t3i = _mm512_sub_pd(a2i, a6i);
        __m512d e0r = _mm512_add_pd(t0r, t2r), e0i = _mm512_add_pd(t0i, t2i);
        __m512d e1r = _mm512_add_pd(t1r, t3i), e1i = _mm512_sub_pd(t1i, t3r);
        __m512d e2r = _mm512_sub_pd(t0r, t2r), e2i = _mm512_sub_pd(t0i, t2i);
        __m512d e3r = _mm512_sub_pd(t1r, t3i), e3i = _mm512_add_pd(t1i, t3r);
        /* odd DFT4 on (a1,a3,a5,a7) */
        __m512d u0r = _mm512_add_pd(a1r, a5r), u0i = _mm512_add_pd(a1i, a5i);
        __m512d u1r = _mm512_sub_pd(a1r, a5r), u1i = _mm512_sub_pd(a1i, a5i);
        __m512d u2r = _mm512_add_pd(a3r, a7r), u2i = _mm512_add_pd(a3i, a7i);
        __m512d u3r = _mm512_sub_pd(a3r, a7r), u3i = _mm512_sub_pd(a3i, a7i);
        __m512d f0r = _mm512_add_pd(u0r, u2r), f0i = _mm512_add_pd(u0i, u2i);
        __m512d f1r = _mm512_add_pd(u1r, u3i), f1i = _mm512_sub_pd(u1i, u3r);
        __m512d f2r = _mm512_sub_pd(u0r, u2r), f2i = _mm512_sub_pd(u0i, u2i);
        __m512d f3r = _mm512_sub_pd(u1r, u3i), f3i = _mm512_add_pd(u1i, u3r);
        /* rotate odds: f1 *= w8, f2 *= -i, f3 *= w8^3 */
        __m512d g1r = _mm512_mul_pd(C8, _mm512_add_pd(f1r, f1i));
        __m512d g1i = _mm512_mul_pd(C8, _mm512_sub_pd(f1i, f1r));
        __m512d g2r = f2i;
        __m512d g2i = _mm512_sub_pd(_mm512_setzero_pd(), f2r);
        __m512d g3r = _mm512_mul_pd(C8, _mm512_sub_pd(f3i, f3r));
        __m512d g3i = _mm512_sub_pd(_mm512_setzero_pd(),
                                    _mm512_mul_pd(C8, _mm512_add_pd(f3r, f3i)));
        __m512d z0r = _mm512_add_pd(e0r, f0r), z0i = _mm512_add_pd(e0i, f0i);
        __m512d z4r = _mm512_sub_pd(e0r, f0r), z4i = _mm512_sub_pd(e0i, f0i);
        __m512d z1r = _mm512_add_pd(e1r, g1r), z1i = _mm512_add_pd(e1i, g1i);
        __m512d z5r = _mm512_sub_pd(e1r, g1r), z5i = _mm512_sub_pd(e1i, g1i);
        __m512d z2r = _mm512_add_pd(e2r, g2r), z2i = _mm512_add_pd(e2i, g2i);
        __m512d z6r = _mm512_sub_pd(e2r, g2r), z6i = _mm512_sub_pd(e2i, g2i);
        __m512d z3r = _mm512_add_pd(e3r, g3r), z3i = _mm512_add_pd(e3i, g3i);
        __m512d z7r = _mm512_sub_pd(e3r, g3r), z7i = _mm512_sub_pd(e3i, g3i);
        /* twiddles j=1..7, pair-broadcast, applied in place */
#define TWMUL(J) do { \
        __m512d vr_ = TWPAIR(wr + (size_t)(J - 1) * mm + p); \
        __m512d vi_ = TWPAIR(wi + (size_t)(J - 1) * mm + p); \
        __m512d nr_ = _mm512_fmsub_pd(z##J##r, vr_, _mm512_mul_pd(z##J##i, vi_)); \
        z##J##i = _mm512_fmadd_pd(z##J##r, vi_, _mm512_mul_pd(z##J##i, vr_)); \
        z##J##r = nr_; } while (0)
        TWMUL(1); TWMUL(2); TWMUL(3); TWMUL(4); TWMUL(5); TWMUL(6); TWMUL(7);
#undef TWMUL
        /* store: group p occupies y[32p + 4j + q], group p+1 y[32p+32+4j+q];
           recombine adjacent j via 128-bit-lane shuffles into full zmm runs */
        double *restrict or_ = yr + 32 * (size_t)p;
        double *restrict oi_ = yi + 32 * (size_t)p;
#define STPAIR(K, ZA_r, ZA_i, ZB_r, ZB_i) do { \
        _mm512_storeu_pd(or_ + 8 * K,      _mm512_shuffle_f64x2(ZA_r, ZB_r, 0x44)); \
        _mm512_storeu_pd(oi_ + 8 * K,      _mm512_shuffle_f64x2(ZA_i, ZB_i, 0x44)); \
        _mm512_storeu_pd(or_ + 32 + 8 * K, _mm512_shuffle_f64x2(ZA_r, ZB_r, 0xEE)); \
        _mm512_storeu_pd(oi_ + 32 + 8 * K, _mm512_shuffle_f64x2(ZA_i, ZB_i, 0xEE)); \
        } while (0)
        STPAIR(0, z0r, z0i, z1r, z1i);
        STPAIR(1, z2r, z2i, z3r, z3i);
        STPAIR(2, z4r, z4i, z5r, z5i);
        STPAIR(3, z6r, z6i, z7r, z7i);
#undef STPAIR
    }
#undef TWPAIR
}

/* ---------------- radix-16 / radix-64 L1-tile stages (round r6) ----------------
 * ADOPTED FROM d1_rader near-verbatim: st16_block/st16_s4/st16 are their r3
 * kernels (imported by d1_bluestein in r5 with the flavor verdicts this file's
 * core_schedule uses), dft8v/st64_block/st64_s4/st64 their r5 radix-64 pair.
 * Radix-16 = two fused radix-4 layers through a 2 KB stack tile (layer-1
 * twiddles W16^(u1 v1) are compile-time constants); radix-64 = two fused
 * radix-8 layers through an 8 KB tile with a plan-time 8x8 W64 table.  One
 * tile pass replaces two radix-4/8-ish passes over the M-length planes -- the
 * big convs and direct pow2 sizes are bandwidth-bound, so pass count is what
 * those cells pay for.  Twiddle tables are this file's own [j-1][m] blocks
 * (identical layout to d1_rader's -- their r1 borrowing coming home). */
#define C16A 0.92387953251128675613   /* cos(pi/8) */
#define C16B 0.38268343236508977173   /* sin(pi/8) */
#define C16H 0.70710678118654752440   /* sqrt(2)/2 */

__attribute__((always_inline, target("avx512f"))) static inline void
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

/* s == 4 paired-p path: two p-groups per zmm, twiddles pair-broadcast, outputs
 * as 256-bit halves 64 apart (the st3_s4/st8_s4 shape at radix 16) */
__attribute__((target("avx512f")))
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

__attribute__((target("avx512f")))
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

/* radix-64 layer-1 twiddle table W64^(v u); long-double phases (this file's
 * exact-integer-phase rule -- the donor used M_PI doubles here) */
static double w64r[8][8], w64i[8][8];
static int w64_ready = 0;

static void w64_init(void)
{
    if (w64_ready) return;
    for (int v = 0; v < 8; ++v)
        for (int u = 0; u < 8; ++u) {
            long double t = -2.0L * PIL * (long double)(v * u % 64) / 64.0L;
            w64r[v][u] = (double)cosl(t);
            w64i[v][u] = (double)sinl(t);
        }
    w64_ready = 1;
}

/* one 8-lane DFT8 (forward, natural output order), outputs in registers */
__attribute__((always_inline, target("avx512f"))) static inline void
dft8v(__m512d a0r, __m512d a0i, __m512d a1r, __m512d a1i,
      __m512d a2r, __m512d a2i, __m512d a3r, __m512d a3i,
      __m512d a4r, __m512d a4i, __m512d a5r, __m512d a5i,
      __m512d a6r, __m512d a6i, __m512d a7r, __m512d a7i,
      __m512d *z0r, __m512d *z0i, __m512d *z1r, __m512d *z1i,
      __m512d *z2r, __m512d *z2i, __m512d *z3r, __m512d *z3i,
      __m512d *z4r, __m512d *z4i, __m512d *z5r, __m512d *z5i,
      __m512d *z6r, __m512d *z6i, __m512d *z7r, __m512d *z7i)
{
    const __m512d C8 = _mm512_set1_pd(0.70710678118654752440);
    __m512d t0r = _mm512_add_pd(a0r, a4r), t0i = _mm512_add_pd(a0i, a4i);
    __m512d t1r = _mm512_sub_pd(a0r, a4r), t1i = _mm512_sub_pd(a0i, a4i);
    __m512d t2r = _mm512_add_pd(a2r, a6r), t2i = _mm512_add_pd(a2i, a6i);
    __m512d t3r = _mm512_sub_pd(a2r, a6r), t3i = _mm512_sub_pd(a2i, a6i);
    __m512d e0r = _mm512_add_pd(t0r, t2r), e0i = _mm512_add_pd(t0i, t2i);
    __m512d e1r = _mm512_add_pd(t1r, t3i), e1i = _mm512_sub_pd(t1i, t3r);
    __m512d e2r = _mm512_sub_pd(t0r, t2r), e2i = _mm512_sub_pd(t0i, t2i);
    __m512d e3r = _mm512_sub_pd(t1r, t3i), e3i = _mm512_add_pd(t1i, t3r);
    __m512d u0r = _mm512_add_pd(a1r, a5r), u0i = _mm512_add_pd(a1i, a5i);
    __m512d u1r = _mm512_sub_pd(a1r, a5r), u1i = _mm512_sub_pd(a1i, a5i);
    __m512d u2r = _mm512_add_pd(a3r, a7r), u2i = _mm512_add_pd(a3i, a7i);
    __m512d u3r = _mm512_sub_pd(a3r, a7r), u3i = _mm512_sub_pd(a3i, a7i);
    __m512d f0r = _mm512_add_pd(u0r, u2r), f0i = _mm512_add_pd(u0i, u2i);
    __m512d f1r = _mm512_add_pd(u1r, u3i), f1i = _mm512_sub_pd(u1i, u3r);
    __m512d f2r = _mm512_sub_pd(u0r, u2r), f2i = _mm512_sub_pd(u0i, u2i);
    __m512d f3r = _mm512_sub_pd(u1r, u3i), f3i = _mm512_add_pd(u1i, u3r);
    __m512d g1r = _mm512_mul_pd(C8, _mm512_add_pd(f1r, f1i));
    __m512d g1i = _mm512_mul_pd(C8, _mm512_sub_pd(f1i, f1r));
    __m512d g2r = f2i;
    __m512d g2i = _mm512_sub_pd(_mm512_setzero_pd(), f2r);
    __m512d g3r = _mm512_mul_pd(C8, _mm512_sub_pd(f3i, f3r));
    __m512d g3i = _mm512_sub_pd(_mm512_setzero_pd(),
                                _mm512_mul_pd(C8, _mm512_add_pd(f3r, f3i)));
    *z0r = _mm512_add_pd(e0r, f0r);  *z0i = _mm512_add_pd(e0i, f0i);
    *z4r = _mm512_sub_pd(e0r, f0r);  *z4i = _mm512_sub_pd(e0i, f0i);
    *z1r = _mm512_add_pd(e1r, g1r);  *z1i = _mm512_add_pd(e1i, g1i);
    *z5r = _mm512_sub_pd(e1r, g1r);  *z5i = _mm512_sub_pd(e1i, g1i);
    *z2r = _mm512_add_pd(e2r, g2r);  *z2i = _mm512_add_pd(e2i, g2i);
    *z6r = _mm512_sub_pd(e2r, g2r);  *z6i = _mm512_sub_pd(e2i, g2i);
    *z3r = _mm512_add_pd(e3r, g3r);  *z3i = _mm512_add_pd(e3i, g3i);
    *z7r = _mm512_sub_pd(e3r, g3r);  *z7i = _mm512_sub_pd(e3i, g3i);
}

__attribute__((always_inline, target("avx512f"))) static inline void
st64_block(size_t sm, int s, int m, int q0, __mmask8 mk,
           const double *restrict wrp, const double *restrict wip,
           const double *restrict xpr, const double *restrict xpi,
           double *restrict ypr, double *restrict ypi)
{
    __attribute__((aligned(64))) double zr[64][8], zi[64][8];
    /* layer 1: for each v1, DFT8 over v2, times W64^(v1 u1); slot v1 + 8 u1 */
    for (int v1 = 0; v1 < 8; ++v1) {
        __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
        __m512d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;
        dft8v(_mm512_maskz_loadu_pd(mk, xpr + q0 + sm * v1),
              _mm512_maskz_loadu_pd(mk, xpi + q0 + sm * v1),
              _mm512_maskz_loadu_pd(mk, xpr + q0 + sm * (v1 + 8)),
              _mm512_maskz_loadu_pd(mk, xpi + q0 + sm * (v1 + 8)),
              _mm512_maskz_loadu_pd(mk, xpr + q0 + sm * (v1 + 16)),
              _mm512_maskz_loadu_pd(mk, xpi + q0 + sm * (v1 + 16)),
              _mm512_maskz_loadu_pd(mk, xpr + q0 + sm * (v1 + 24)),
              _mm512_maskz_loadu_pd(mk, xpi + q0 + sm * (v1 + 24)),
              _mm512_maskz_loadu_pd(mk, xpr + q0 + sm * (v1 + 32)),
              _mm512_maskz_loadu_pd(mk, xpi + q0 + sm * (v1 + 32)),
              _mm512_maskz_loadu_pd(mk, xpr + q0 + sm * (v1 + 40)),
              _mm512_maskz_loadu_pd(mk, xpi + q0 + sm * (v1 + 40)),
              _mm512_maskz_loadu_pd(mk, xpr + q0 + sm * (v1 + 48)),
              _mm512_maskz_loadu_pd(mk, xpi + q0 + sm * (v1 + 48)),
              _mm512_maskz_loadu_pd(mk, xpr + q0 + sm * (v1 + 56)),
              _mm512_maskz_loadu_pd(mk, xpi + q0 + sm * (v1 + 56)),
              &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
              &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
        _mm512_store_pd(zr[v1], y0r);  _mm512_store_pd(zi[v1], y0i);
        if (v1 == 0) {
#define ST64_RAW(u1, yur, yui)                                                 \
            do {                                                               \
                _mm512_store_pd(zr[8 * (u1)], (yur));                          \
                _mm512_store_pd(zi[8 * (u1)], (yui));                          \
            } while (0)
            ST64_RAW(1, y1r, y1i); ST64_RAW(2, y2r, y2i); ST64_RAW(3, y3r, y3i);
            ST64_RAW(4, y4r, y4i); ST64_RAW(5, y5r, y5i); ST64_RAW(6, y6r, y6i);
            ST64_RAW(7, y7r, y7i);
#undef ST64_RAW
        } else {
#define ST64_W64(u1, yur, yui)                                                 \
            do {                                                               \
                __m512d c_ = _mm512_set1_pd(w64r[v1][u1]);                     \
                __m512d s_ = _mm512_set1_pd(w64i[v1][u1]);                     \
                _mm512_store_pd(zr[v1 + 8 * (u1)],                             \
                    _mm512_fmsub_pd((yur), c_, _mm512_mul_pd((yui), s_)));     \
                _mm512_store_pd(zi[v1 + 8 * (u1)],                             \
                    _mm512_fmadd_pd((yur), s_, _mm512_mul_pd((yui), c_)));     \
            } while (0)
            ST64_W64(1, y1r, y1i); ST64_W64(2, y2r, y2i); ST64_W64(3, y3r, y3i);
            ST64_W64(4, y4r, y4i); ST64_W64(5, y5r, y5i); ST64_W64(6, y6r, y6i);
            ST64_W64(7, y7r, y7i);
#undef ST64_W64
        }
    }
    /* layer 2: for each u1, DFT8 over v1 (slots 8u1 + v1); stage twiddle at
       output u = u1 + 8 u2, y position q0 + s*u */
    for (int u1 = 0; u1 < 8; ++u1) {
        __m512d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
        __m512d o4r, o4i, o5r, o5i, o6r, o6i, o7r, o7i;
        dft8v(_mm512_load_pd(zr[8 * u1 + 0]), _mm512_load_pd(zi[8 * u1 + 0]),
              _mm512_load_pd(zr[8 * u1 + 1]), _mm512_load_pd(zi[8 * u1 + 1]),
              _mm512_load_pd(zr[8 * u1 + 2]), _mm512_load_pd(zi[8 * u1 + 2]),
              _mm512_load_pd(zr[8 * u1 + 3]), _mm512_load_pd(zi[8 * u1 + 3]),
              _mm512_load_pd(zr[8 * u1 + 4]), _mm512_load_pd(zi[8 * u1 + 4]),
              _mm512_load_pd(zr[8 * u1 + 5]), _mm512_load_pd(zi[8 * u1 + 5]),
              _mm512_load_pd(zr[8 * u1 + 6]), _mm512_load_pd(zi[8 * u1 + 6]),
              _mm512_load_pd(zr[8 * u1 + 7]), _mm512_load_pd(zi[8 * u1 + 7]),
              &o0r, &o0i, &o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
              &o4r, &o4i, &o5r, &o5i, &o6r, &o6i, &o7r, &o7i);
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
#define ST64_TWSTORE(ov_r, ov_i, u)                                            \
        do {                                                                   \
            __m512d w_ = _mm512_set1_pd(wrp[(size_t)((u) - 1) * m]);           \
            __m512d v_ = _mm512_set1_pd(wip[(size_t)((u) - 1) * m]);           \
            _mm512_mask_storeu_pd(ypr + q0 + (size_t)s * (u), mk,              \
                _mm512_fmsub_pd((ov_r), w_, _mm512_mul_pd((ov_i), v_)));       \
            _mm512_mask_storeu_pd(ypi + q0 + (size_t)s * (u), mk,              \
                _mm512_fmadd_pd((ov_r), v_, _mm512_mul_pd((ov_i), w_)));       \
        } while (0)
        ST64_TWSTORE(o1r, o1i, u1 + 8);
        ST64_TWSTORE(o2r, o2i, u1 + 16);
        ST64_TWSTORE(o3r, o3i, u1 + 24);
        ST64_TWSTORE(o4r, o4i, u1 + 32);
        ST64_TWSTORE(o5r, o5i, u1 + 40);
        ST64_TWSTORE(o6r, o6i, u1 + 48);
        ST64_TWSTORE(o7r, o7i, u1 + 56);
#undef ST64_TWSTORE
    }
}

/* s == 4 paired-p path: two p-groups per zmm, twiddles pair-broadcast,
 * outputs as 256-bit halves 256 apart -- st16_s4's shape at radix 64 */
__attribute__((target("avx512f")))
static void st64_s4(int m, const double *restrict wr, const double *restrict wi,
                    const double *restrict xr, const double *restrict xi,
                    double *restrict yr, double *restrict yi)
{
    const size_t sm = 4 * (size_t)m;
    const __m512i PAIR = _mm512_setr_epi64(0, 0, 0, 0, 1, 1, 1, 1);
    for (int p = 0; p + 2 <= m; p += 2) {
        const double *restrict xpr = xr + 4 * (size_t)p;
        const double *restrict xpi = xi + 4 * (size_t)p;
        double *restrict ypr = yr + 256 * (size_t)p;
        double *restrict ypi = yi + 256 * (size_t)p;
        __attribute__((aligned(64))) double zr[64][8], zi[64][8];
        for (int v1 = 0; v1 < 8; ++v1) {
            __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;
            __m512d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;
            dft8v(_mm512_loadu_pd(xpr + sm * v1),
                  _mm512_loadu_pd(xpi + sm * v1),
                  _mm512_loadu_pd(xpr + sm * (v1 + 8)),
                  _mm512_loadu_pd(xpi + sm * (v1 + 8)),
                  _mm512_loadu_pd(xpr + sm * (v1 + 16)),
                  _mm512_loadu_pd(xpi + sm * (v1 + 16)),
                  _mm512_loadu_pd(xpr + sm * (v1 + 24)),
                  _mm512_loadu_pd(xpi + sm * (v1 + 24)),
                  _mm512_loadu_pd(xpr + sm * (v1 + 32)),
                  _mm512_loadu_pd(xpi + sm * (v1 + 32)),
                  _mm512_loadu_pd(xpr + sm * (v1 + 40)),
                  _mm512_loadu_pd(xpi + sm * (v1 + 40)),
                  _mm512_loadu_pd(xpr + sm * (v1 + 48)),
                  _mm512_loadu_pd(xpi + sm * (v1 + 48)),
                  _mm512_loadu_pd(xpr + sm * (v1 + 56)),
                  _mm512_loadu_pd(xpi + sm * (v1 + 56)),
                  &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,
                  &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);
            _mm512_store_pd(zr[v1], y0r);  _mm512_store_pd(zi[v1], y0i);
            if (v1 == 0) {
#define ST64_RAW(u1, yur, yui)                                                 \
                do {                                                           \
                    _mm512_store_pd(zr[8 * (u1)], (yur));                      \
                    _mm512_store_pd(zi[8 * (u1)], (yui));                      \
                } while (0)
                ST64_RAW(1, y1r, y1i); ST64_RAW(2, y2r, y2i); ST64_RAW(3, y3r, y3i);
                ST64_RAW(4, y4r, y4i); ST64_RAW(5, y5r, y5i); ST64_RAW(6, y6r, y6i);
                ST64_RAW(7, y7r, y7i);
#undef ST64_RAW
            } else {
#define ST64_W64(u1, yur, yui)                                                 \
                do {                                                           \
                    __m512d c_ = _mm512_set1_pd(w64r[v1][u1]);                 \
                    __m512d s_ = _mm512_set1_pd(w64i[v1][u1]);                 \
                    _mm512_store_pd(zr[v1 + 8 * (u1)],                         \
                        _mm512_fmsub_pd((yur), c_, _mm512_mul_pd((yui), s_))); \
                    _mm512_store_pd(zi[v1 + 8 * (u1)],                         \
                        _mm512_fmadd_pd((yur), s_, _mm512_mul_pd((yui), c_))); \
                } while (0)
                ST64_W64(1, y1r, y1i); ST64_W64(2, y2r, y2i); ST64_W64(3, y3r, y3i);
                ST64_W64(4, y4r, y4i); ST64_W64(5, y5r, y5i); ST64_W64(6, y6r, y6i);
                ST64_W64(7, y7r, y7i);
#undef ST64_W64
            }
        }
        for (int u1 = 0; u1 < 8; ++u1) {
            __m512d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;
            __m512d o4r, o4i, o5r, o5i, o6r, o6i, o7r, o7i;
            dft8v(_mm512_load_pd(zr[8 * u1 + 0]), _mm512_load_pd(zi[8 * u1 + 0]),
                  _mm512_load_pd(zr[8 * u1 + 1]), _mm512_load_pd(zi[8 * u1 + 1]),
                  _mm512_load_pd(zr[8 * u1 + 2]), _mm512_load_pd(zi[8 * u1 + 2]),
                  _mm512_load_pd(zr[8 * u1 + 3]), _mm512_load_pd(zi[8 * u1 + 3]),
                  _mm512_load_pd(zr[8 * u1 + 4]), _mm512_load_pd(zi[8 * u1 + 4]),
                  _mm512_load_pd(zr[8 * u1 + 5]), _mm512_load_pd(zi[8 * u1 + 5]),
                  _mm512_load_pd(zr[8 * u1 + 6]), _mm512_load_pd(zi[8 * u1 + 6]),
                  _mm512_load_pd(zr[8 * u1 + 7]), _mm512_load_pd(zi[8 * u1 + 7]),
                  &o0r, &o0i, &o1r, &o1i, &o2r, &o2i, &o3r, &o3i,
                  &o4r, &o4i, &o5r, &o5i, &o6r, &o6i, &o7r, &o7i);
#define ST64_S4STORE(ov_r, ov_i, u)                                            \
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
                _mm256_storeu_pd(ypr + 4 * (u),       _mm512_castpd512_pd256(fr_)); \
                _mm256_storeu_pd(ypr + 4 * (u) + 256, _mm512_extractf64x4_pd(fr_, 1)); \
                _mm256_storeu_pd(ypi + 4 * (u),       _mm512_castpd512_pd256(fi_)); \
                _mm256_storeu_pd(ypi + 4 * (u) + 256, _mm512_extractf64x4_pd(fi_, 1)); \
            } while (0)
            ST64_S4STORE(o0r, o0i, u1);
            ST64_S4STORE(o1r, o1i, u1 + 8);
            ST64_S4STORE(o2r, o2i, u1 + 16);
            ST64_S4STORE(o3r, o3i, u1 + 24);
            ST64_S4STORE(o4r, o4i, u1 + 32);
            ST64_S4STORE(o5r, o5i, u1 + 40);
            ST64_S4STORE(o6r, o6i, u1 + 48);
            ST64_S4STORE(o7r, o7i, u1 + 56);
#undef ST64_S4STORE
        }
    }
}

__attribute__((target("avx512f")))
static void st64(int m, int s, const double *restrict wr, const double *restrict wi,
                 const double *restrict xr, const double *restrict xi,
                 double *restrict yr, double *restrict yi)
{
    if (s == 4 && (m & 1) == 0) { st64_s4(m, wr, wi, xr, xi, yr, yi); return; }
    const size_t sm = (size_t)s * m;
    for (int p = 0; p < m; ++p) {
        const double *restrict xpr = xr + (size_t)s * p;
        const double *restrict xpi = xi + (size_t)s * p;
        double *restrict ypr = yr + (size_t)s * 64 * p;
        double *restrict ypi = yi + (size_t)s * 64 * p;
        int q0 = 0;
        /* (donor r5 note kept: prefetching the 64 stream heads a block ahead
           LOST -- the planes are L2-resident, 128 extra uops/block only cost) */
        for (; q0 + 8 <= s; q0 += 8)
            st64_block(sm, s, m, q0, (__mmask8)0xFF, wr + p, wi + p,
                       xpr, xpi, ypr, ypi);
        if (q0 < s)
            st64_block(sm, s, m, q0, (__mmask8)((1u << (s - q0)) - 1),
                       wr + p, wi + p, xpr, xpi, ypr, ypi);
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

/* radix-3 at s == 4, two p-groups per zmm -- ADOPTED FROM d1_rader (r2/r3
 * st3_s4, near-verbatim; the scalar tail is this file's st3 body for one p).
 * gcc never vectorizes the 4-wide s=4 inner loop (their measured-scalar
 * diagnosis); pair-broadcast twiddles + 256-bit half stores fix it. Runs in
 * 1020 = [4,3,5,17] (stage 1, m=85) and 60 = [4,3,5] (stage 1, m=5). */
__attribute__((target("avx512f")))
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
    for (; p < m; ++p) {     /* odd-m tail: one p, scalar (st3's body, s=4) */
        const double s3 = 0.86602540378443864676;
        const double u1r_ = w1r[p], u1i_ = w1i[p];
        const double u2r_ = w2r[p], u2i_ = w2i[p];
        const double *restrict x0r = xr + 4 * (size_t)p, *restrict x0i = xi + 4 * (size_t)p;
        const double *restrict x1r = x0r + sm, *restrict x1i = x0i + sm;
        const double *restrict x2r = x1r + sm, *restrict x2i = x1i + sm;
        double *restrict o0r = yr + 12 * (size_t)p, *restrict o0i = yi + 12 * (size_t)p;
        for (int q = 0; q < 4; ++q) {
            double ar = x0r[q], ai = x0i[q];
            double br = x1r[q], bi = x1i[q];
            double cr = x2r[q], ci = x2i[q];
            double tr = br + cr, ti = bi + ci;
            double ur = br - cr, ui = bi - ci;
            double mr = ar - 0.5 * tr, mi = ai - 0.5 * ti;
            double y1r = mr + s3 * ui, y1i = mi - s3 * ur;
            double y2r = mr - s3 * ui, y2i = mi + s3 * ur;
            o0r[q] = ar + tr;  o0i[q] = ai + ti;
            o0r[q + 4] = y1r * u1r_ - y1i * u1i_;  o0i[q + 4] = y1r * u1i_ + y1i * u1r_;
            o0r[q + 8] = y2r * u2r_ - y2i * u2i_;  o0i[q + 8] = y2r * u2i_ + y2i * u2r_;
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

/* one 8-lane column of a radix-5 butterfly: full loads at off + k*sm,
 * twiddled outputs returned in registers -- ADOPTED FROM d1_rader (r4
 * st5_col/st5_vec/st5_s12, near-verbatim). */
__attribute__((always_inline, target("avx512f"))) static inline void
st5_col(const double *restrict xr, const double *restrict xi, size_t off, size_t sm,
        __m512d u1r, __m512d u1i, __m512d u2r, __m512d u2i,
        __m512d u3r, __m512d u3i, __m512d u4r, __m512d u4i,
        __m512d *o0r, __m512d *o0i, __m512d *o1r, __m512d *o1i,
        __m512d *o2r, __m512d *o2i, __m512d *o3r, __m512d *o3i,
        __m512d *o4r, __m512d *o4i)
{
    const __m512d C1 = _mm512_set1_pd(0.30901699437494742410);
    const __m512d S1 = _mm512_set1_pd(0.95105651629515357212);
    const __m512d C2 = _mm512_set1_pd(-0.80901699437494742410);
    const __m512d S2 = _mm512_set1_pd(0.58778525229247312917);
    __m512d ar = _mm512_loadu_pd(xr + off),          ai = _mm512_loadu_pd(xi + off);
    __m512d br = _mm512_loadu_pd(xr + off + sm),     bi = _mm512_loadu_pd(xi + off + sm);
    __m512d cr = _mm512_loadu_pd(xr + off + 2 * sm), ci = _mm512_loadu_pd(xi + off + 2 * sm);
    __m512d dr = _mm512_loadu_pd(xr + off + 3 * sm), di = _mm512_loadu_pd(xi + off + 3 * sm);
    __m512d er = _mm512_loadu_pd(xr + off + 4 * sm), ei = _mm512_loadu_pd(xi + off + 4 * sm);
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
    *o0r = _mm512_add_pd(ar, _mm512_add_pd(t1r, t2r));
    *o0i = _mm512_add_pd(ai, _mm512_add_pd(t1i, t2i));
    *o1r = _mm512_fmsub_pd(y1r, u1r, _mm512_mul_pd(y1i, u1i));
    *o1i = _mm512_fmadd_pd(y1r, u1i, _mm512_mul_pd(y1i, u1r));
    *o2r = _mm512_fmsub_pd(y2r, u2r, _mm512_mul_pd(y2i, u2i));
    *o2i = _mm512_fmadd_pd(y2r, u2i, _mm512_mul_pd(y2i, u2r));
    *o3r = _mm512_fmsub_pd(y3r, u3r, _mm512_mul_pd(y3i, u3i));
    *o3i = _mm512_fmadd_pd(y3r, u3i, _mm512_mul_pd(y3i, u3r));
    *o4r = _mm512_fmsub_pd(y4r, u4r, _mm512_mul_pd(y4i, u4i));
    *o4i = _mm512_fmadd_pd(y4r, u4i, _mm512_mul_pd(y4i, u4r));
}

/* radix-5, masked 8-lane blocks per p from p0 (tail path for st5_s12) */
__attribute__((target("avx512f")))
static void st5_vec(int m, int s, int p0, const double *restrict wr, const double *restrict wi,
                    const double *restrict xr, const double *restrict xi,
                    double *restrict yr, double *restrict yi)
{
    const __m512d C1 = _mm512_set1_pd(0.30901699437494742410);
    const __m512d S1 = _mm512_set1_pd(0.95105651629515357212);
    const __m512d C2 = _mm512_set1_pd(-0.80901699437494742410);
    const __m512d S2 = _mm512_set1_pd(0.58778525229247312917);
    const size_t sm = (size_t)s * m;
    for (int p = p0; p < m; ++p) {
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

/* radix-5, s == 12 paired-p path (d1_rader r4: the 1020 stage-2, m=17): the
 * masked 8+4 form wastes a third of every second block. Two p-groups are 24
 * contiguous lanes = 3 FULL zmm per input stream: column 0 = q0..7 of p,
 * column 1 = q8..11 of p then q0..3 of p+1 (twiddles pair-broadcast, outputs
 * as 256-bit halves 60 apart -- the st3_s4 shape), column 2 = q4..11 of p+1.
 * Odd-m tail falls back to the masked path. */
__attribute__((target("avx512f")))
static void st5_s12(int m, const double *restrict wr, const double *restrict wi,
                    const double *restrict xr, const double *restrict xi,
                    double *restrict yr, double *restrict yi)
{
    const __m512i PAIR = _mm512_setr_epi64(0, 0, 0, 0, 1, 1, 1, 1);
    const size_t sm = 12 * (size_t)m;
    int p = 0;
    for (; p + 2 <= m; p += 2) {
        const size_t ib = 12 * (size_t)p;
        double *restrict o_r = yr + 60 * (size_t)p;
        double *restrict o_i = yi + 60 * (size_t)p;
        __m512d z0r, z0i, z1r, z1i, z2r, z2i, z3r, z3i, z4r, z4i;
        /* column 0: all lanes p */
        st5_col(xr, xi, ib, sm,
                _mm512_set1_pd(wr[p]),         _mm512_set1_pd(wi[p]),
                _mm512_set1_pd(wr[m + p]),     _mm512_set1_pd(wi[m + p]),
                _mm512_set1_pd(wr[2 * m + p]), _mm512_set1_pd(wi[2 * m + p]),
                _mm512_set1_pd(wr[3 * m + p]), _mm512_set1_pd(wi[3 * m + p]),
                &z0r, &z0i, &z1r, &z1i, &z2r, &z2i, &z3r, &z3i, &z4r, &z4i);
        _mm512_storeu_pd(o_r,      z0r);  _mm512_storeu_pd(o_i,      z0i);
        _mm512_storeu_pd(o_r + 12, z1r);  _mm512_storeu_pd(o_i + 12, z1i);
        _mm512_storeu_pd(o_r + 24, z2r);  _mm512_storeu_pd(o_i + 24, z2i);
        _mm512_storeu_pd(o_r + 36, z3r);  _mm512_storeu_pd(o_i + 36, z3i);
        _mm512_storeu_pd(o_r + 48, z4r);  _mm512_storeu_pd(o_i + 48, z4i);
        /* column 1: lanes 0-3 = p, lanes 4-7 = p+1 */
#define ST5_PAIRTW(tab, u_) \
        _mm512_permutexvar_pd(PAIR, _mm512_castpd128_pd512(_mm_loadu_pd((tab) + (size_t)(u_) * m + p)))
        st5_col(xr, xi, ib + 8, sm,
                ST5_PAIRTW(wr, 0), ST5_PAIRTW(wi, 0),
                ST5_PAIRTW(wr, 1), ST5_PAIRTW(wi, 1),
                ST5_PAIRTW(wr, 2), ST5_PAIRTW(wi, 2),
                ST5_PAIRTW(wr, 3), ST5_PAIRTW(wi, 3),
                &z0r, &z0i, &z1r, &z1i, &z2r, &z2i, &z3r, &z3i, &z4r, &z4i);
#undef ST5_PAIRTW
#define ST5_HALFST(v_, u_) \
        do { \
            _mm256_storeu_pd(o_r + 12 * (u_) + 8,  _mm512_castpd512_pd256(v_##r)); \
            _mm256_storeu_pd(o_r + 60 + 12 * (u_), _mm512_extractf64x4_pd(v_##r, 1)); \
            _mm256_storeu_pd(o_i + 12 * (u_) + 8,  _mm512_castpd512_pd256(v_##i)); \
            _mm256_storeu_pd(o_i + 60 + 12 * (u_), _mm512_extractf64x4_pd(v_##i, 1)); \
        } while (0)
        ST5_HALFST(z0, 0); ST5_HALFST(z1, 1); ST5_HALFST(z2, 2);
        ST5_HALFST(z3, 3); ST5_HALFST(z4, 4);
#undef ST5_HALFST
        /* column 2: all lanes p+1 */
        st5_col(xr, xi, ib + 16, sm,
                _mm512_set1_pd(wr[p + 1]),         _mm512_set1_pd(wi[p + 1]),
                _mm512_set1_pd(wr[m + p + 1]),     _mm512_set1_pd(wi[m + p + 1]),
                _mm512_set1_pd(wr[2 * m + p + 1]), _mm512_set1_pd(wi[2 * m + p + 1]),
                _mm512_set1_pd(wr[3 * m + p + 1]), _mm512_set1_pd(wi[3 * m + p + 1]),
                &z0r, &z0i, &z1r, &z1i, &z2r, &z2i, &z3r, &z3i, &z4r, &z4i);
        _mm512_storeu_pd(o_r + 64,  z0r);  _mm512_storeu_pd(o_i + 64,  z0i);
        _mm512_storeu_pd(o_r + 76,  z1r);  _mm512_storeu_pd(o_i + 76,  z1i);
        _mm512_storeu_pd(o_r + 88,  z2r);  _mm512_storeu_pd(o_i + 88,  z2i);
        _mm512_storeu_pd(o_r + 100, z3r);  _mm512_storeu_pd(o_i + 100, z3i);
        _mm512_storeu_pd(o_r + 112, z4r);  _mm512_storeu_pd(o_i + 112, z4i);
    }
    if (p < m) st5_vec(m, 12, p, wr, wi, xr, xi, yr, yi);
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

/* Dense radix-17 FINAL stage (m == 1, unit twiddles), explicit AVX-512 8-lane
 * blocks -- ADOPTED FROM d1_rader (r2 st17_vblock, taken nearly verbatim; only
 * the coefficient tables changed to this file's per-stage gC/gS layout, which
 * is the same 8x8 row-major fold table). Their r2 finding, reconfirmed here:
 * gcc 11 never vectorizes stg's lane loop across the nested j/i loops on the
 * scoring Ice Lake (the stage ran at scalar FMA throughput and was ~44% of the
 * 1021 transform). u-half and v-half run sequentially so live state stays ~22
 * regs; A_k parks in BOTH twin output slots, the v-half combines by RMW.
 * NO target attribute: intrinsics refuse to inline into functions carrying
 * target("arch=icelake-server") under -march=native (their hard-won note). */
__attribute__((always_inline, target("avx512f")))
static inline void st17_vblock(int s, int q0, __mmask8 mk,
        const double *restrict C, const double *restrict S,
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
    _mm512_mask_storeu_pd(yr + q0, mk, sur);
    _mm512_mask_storeu_pd(yi + q0, mk, sui);
    /* A_k = x0 + sum_j C[k][j] u_j  -> parked in BOTH twin slots of y */
#pragma GCC unroll 8
    for (int k = 1; k <= 8; ++k) {
        __m512d Ar = x0r, Ai = x0i;
#pragma GCC unroll 8
        for (int j = 0; j < 8; ++j) {
            __m512d cc = _mm512_set1_pd(C[(size_t)(k - 1) * 8 + j]);
            Ar = _mm512_fmadd_pd(cc, ur[j], Ar);
            Ai = _mm512_fmadd_pd(cc, ui[j], Ai);
        }
        _mm512_mask_storeu_pd(yr + q0 + (size_t)s * k, mk, Ar);
        _mm512_mask_storeu_pd(yi + q0 + (size_t)s * k, mk, Ai);
        _mm512_mask_storeu_pd(yr + q0 + (size_t)s * (17 - k), mk, Ar);
        _mm512_mask_storeu_pd(yi + q0 + (size_t)s * (17 - k), mk, Ai);
    }
    /* v-half: B_k = sum_j S[k][j] v_j; X_k = A - iB, X_{17-k} = A + iB */
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
            __m512d ss = _mm512_set1_pd(S[(size_t)(k - 1) * 8 + j]);
            Br = _mm512_fmadd_pd(ss, vr[j], Br);
            Bi = _mm512_fmadd_pd(ss, vi[j], Bi);
        }
        size_t i1 = q0 + (size_t)s * k, i2 = q0 + (size_t)s * (17 - k);
        __m512d Ar = _mm512_maskz_loadu_pd(mk, yr + i1);
        __m512d Ai = _mm512_maskz_loadu_pd(mk, yi + i1);
        _mm512_mask_storeu_pd(yr + i1, mk, _mm512_add_pd(Ar, Bi));
        _mm512_mask_storeu_pd(yi + i1, mk, _mm512_sub_pd(Ai, Br));
        _mm512_mask_storeu_pd(yr + i2, mk, _mm512_sub_pd(Ar, Bi));
        _mm512_mask_storeu_pd(yi + i2, mk, _mm512_add_pd(Ai, Br));
    }
}

__attribute__((target("avx512f")))
static void st17(int s, const double *restrict C, const double *restrict S,
                 const double *restrict xr, const double *restrict xi,
                 double *restrict yr, double *restrict yi)
{
    int q = 0;
    for (; q + 8 <= s; q += 8)
        st17_vblock(s, q, (__mmask8)0xFF, C, S, xr, xi, yr, yi);
    if (q < s)
        st17_vblock(s, q, (__mmask8)((1u << (s - q)) - 1), C, S, xr, xi, yr, yi);
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
        case 3: if (s == 4) st3_s4(m, c->twr[st], c->twi[st], xr, xi, yr, yi);
                else st3(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi);
                break;
        case 4: st4(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break;
        case 5: if (s == 12) st5_s12(m, c->twr[st], c->twi[st], xr, xi, yr, yi);
                else st5(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi);
                break;
        case 8: if (s == 4 && !(m & 1))
                    st8_s4(m, c->twr[st], c->twi[st], xr, xi, yr, yi);
                else st8(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi);
                break;
        case 16: st16(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break;
        case 64: st64(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break;
        default:
            if (r == 17 && m == 1)
                st17(s, c->gC[st], c->gS[st], xr, xi, yr, yi);
            else
                stg(r, m, s, c->twr[st], c->twi[st], c->gC[st], c->gS[st], gs,
                    xr, xi, yr, yi);
            break;
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

/* NT-streamed fused exits -- ADOPTED FROM d1_pow2 (r3 change 2: NT final
 * stores once in+out exceeds the scoring node's 24 MB L3 kill the RFO reads;
 * their 16384 B=256 went 71.9 -> 50.7 us) and d1_bluestein (r3 NT exit
 * scatter). Same butterflies as st{2,4,8}_last_int, outputs interleaved
 * in-register and streamed. Preconditions checked by the caller: s % 8 == 0,
 * yd 64-byte aligned, and my intermediates never touch yd (so the stream
 * targets no dirty lines -- the bug that poisoned d1_pow2's r1 NT verdict). */
__attribute__((target("avx512f")))
static void st2_last_int_nt(int s, const double *restrict xr, const double *restrict xi,
                            double *restrict yd)
{
    const __m512i ILO = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i IHI = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    for (int q = 0; q < s; q += 8) {
        __m512d ar = _mm512_loadu_pd(xr + q),     ai = _mm512_loadu_pd(xi + q);
        __m512d br = _mm512_loadu_pd(xr + q + s), bi = _mm512_loadu_pd(xi + q + s);
        __m512d z0r = _mm512_add_pd(ar, br), z0i = _mm512_add_pd(ai, bi);
        __m512d z1r = _mm512_sub_pd(ar, br), z1i = _mm512_sub_pd(ai, bi);
        _mm512_stream_pd(yd + 2 * q,           _mm512_permutex2var_pd(z0r, ILO, z0i));
        _mm512_stream_pd(yd + 2 * q + 8,       _mm512_permutex2var_pd(z0r, IHI, z0i));
        _mm512_stream_pd(yd + 2 * (q + s),     _mm512_permutex2var_pd(z1r, ILO, z1i));
        _mm512_stream_pd(yd + 2 * (q + s) + 8, _mm512_permutex2var_pd(z1r, IHI, z1i));
    }
    _mm_sfence();
}

__attribute__((target("avx512f")))
static void st4_last_int_nt(int s, const double *restrict xr, const double *restrict xi,
                            double *restrict yd)
{
    const __m512i ILO = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i IHI = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    for (int q = 0; q < s; q += 8) {
        __m512d ar = _mm512_loadu_pd(xr + q),         ai = _mm512_loadu_pd(xi + q);
        __m512d br = _mm512_loadu_pd(xr + q + s),     bi = _mm512_loadu_pd(xi + q + s);
        __m512d cr = _mm512_loadu_pd(xr + q + 2 * s), ci = _mm512_loadu_pd(xi + q + 2 * s);
        __m512d dr = _mm512_loadu_pd(xr + q + 3 * s), di = _mm512_loadu_pd(xi + q + 3 * s);
        __m512d t0r = _mm512_add_pd(ar, cr), t0i = _mm512_add_pd(ai, ci);
        __m512d t1r = _mm512_sub_pd(ar, cr), t1i = _mm512_sub_pd(ai, ci);
        __m512d t2r = _mm512_add_pd(br, dr), t2i = _mm512_add_pd(bi, di);
        __m512d t3r = _mm512_sub_pd(br, dr), t3i = _mm512_sub_pd(bi, di);
        __m512d z0r = _mm512_add_pd(t0r, t2r), z0i = _mm512_add_pd(t0i, t2i);
        __m512d z1r = _mm512_add_pd(t1r, t3i), z1i = _mm512_sub_pd(t1i, t3r);
        __m512d z2r = _mm512_sub_pd(t0r, t2r), z2i = _mm512_sub_pd(t0i, t2i);
        __m512d z3r = _mm512_sub_pd(t1r, t3i), z3i = _mm512_add_pd(t1i, t3r);
        _mm512_stream_pd(yd + 2 * q,               _mm512_permutex2var_pd(z0r, ILO, z0i));
        _mm512_stream_pd(yd + 2 * q + 8,           _mm512_permutex2var_pd(z0r, IHI, z0i));
        _mm512_stream_pd(yd + 2 * (q + s),         _mm512_permutex2var_pd(z1r, ILO, z1i));
        _mm512_stream_pd(yd + 2 * (q + s) + 8,     _mm512_permutex2var_pd(z1r, IHI, z1i));
        _mm512_stream_pd(yd + 2 * (q + 2 * s),     _mm512_permutex2var_pd(z2r, ILO, z2i));
        _mm512_stream_pd(yd + 2 * (q + 2 * s) + 8, _mm512_permutex2var_pd(z2r, IHI, z2i));
        _mm512_stream_pd(yd + 2 * (q + 3 * s),     _mm512_permutex2var_pd(z3r, ILO, z3i));
        _mm512_stream_pd(yd + 2 * (q + 3 * s) + 8, _mm512_permutex2var_pd(z3r, IHI, z3i));
    }
    _mm_sfence();
}

__attribute__((target("avx512f")))
static void st8_last_int_nt(int s, const double *restrict xr, const double *restrict xi,
                            double *restrict yd)
{
    const __m512d C8v = _mm512_set1_pd(0.70710678118654752440);
    const __m512i ILO = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i IHI = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    for (int q = 0; q < s; q += 8) {
        __m512d a0r = _mm512_loadu_pd(xr + q),         a0i = _mm512_loadu_pd(xi + q);
        __m512d a1r = _mm512_loadu_pd(xr + q + s),     a1i = _mm512_loadu_pd(xi + q + s);
        __m512d a2r = _mm512_loadu_pd(xr + q + 2 * s), a2i = _mm512_loadu_pd(xi + q + 2 * s);
        __m512d a3r = _mm512_loadu_pd(xr + q + 3 * s), a3i = _mm512_loadu_pd(xi + q + 3 * s);
        __m512d a4r = _mm512_loadu_pd(xr + q + 4 * s), a4i = _mm512_loadu_pd(xi + q + 4 * s);
        __m512d a5r = _mm512_loadu_pd(xr + q + 5 * s), a5i = _mm512_loadu_pd(xi + q + 5 * s);
        __m512d a6r = _mm512_loadu_pd(xr + q + 6 * s), a6i = _mm512_loadu_pd(xi + q + 6 * s);
        __m512d a7r = _mm512_loadu_pd(xr + q + 7 * s), a7i = _mm512_loadu_pd(xi + q + 7 * s);
        __m512d t0r = _mm512_add_pd(a0r, a4r), t0i = _mm512_add_pd(a0i, a4i);
        __m512d t1r = _mm512_sub_pd(a0r, a4r), t1i = _mm512_sub_pd(a0i, a4i);
        __m512d t2r = _mm512_add_pd(a2r, a6r), t2i = _mm512_add_pd(a2i, a6i);
        __m512d t3r = _mm512_sub_pd(a2r, a6r), t3i = _mm512_sub_pd(a2i, a6i);
        __m512d e0r = _mm512_add_pd(t0r, t2r), e0i = _mm512_add_pd(t0i, t2i);
        __m512d e1r = _mm512_add_pd(t1r, t3i), e1i = _mm512_sub_pd(t1i, t3r);
        __m512d e2r = _mm512_sub_pd(t0r, t2r), e2i = _mm512_sub_pd(t0i, t2i);
        __m512d e3r = _mm512_sub_pd(t1r, t3i), e3i = _mm512_add_pd(t1i, t3r);
        __m512d u0r = _mm512_add_pd(a1r, a5r), u0i = _mm512_add_pd(a1i, a5i);
        __m512d u1r = _mm512_sub_pd(a1r, a5r), u1i = _mm512_sub_pd(a1i, a5i);
        __m512d u2r = _mm512_add_pd(a3r, a7r), u2i = _mm512_add_pd(a3i, a7i);
        __m512d u3r = _mm512_sub_pd(a3r, a7r), u3i = _mm512_sub_pd(a3i, a7i);
        __m512d f0r = _mm512_add_pd(u0r, u2r), f0i = _mm512_add_pd(u0i, u2i);
        __m512d f1r = _mm512_add_pd(u1r, u3i), f1i = _mm512_sub_pd(u1i, u3r);
        __m512d f2r = _mm512_sub_pd(u0r, u2r), f2i = _mm512_sub_pd(u0i, u2i);
        __m512d f3r = _mm512_sub_pd(u1r, u3i), f3i = _mm512_add_pd(u1i, u3r);
        __m512d g1r = _mm512_mul_pd(C8v, _mm512_add_pd(f1r, f1i));
        __m512d g1i = _mm512_mul_pd(C8v, _mm512_sub_pd(f1i, f1r));
        __m512d g2r = f2i;
        __m512d g2i = _mm512_sub_pd(_mm512_setzero_pd(), f2r);
        __m512d g3r = _mm512_mul_pd(C8v, _mm512_sub_pd(f3i, f3r));
        __m512d g3i = _mm512_sub_pd(_mm512_setzero_pd(),
                                    _mm512_mul_pd(C8v, _mm512_add_pd(f3r, f3i)));
        __m512d zr, zi;
#define NTOUT(jj, rr, ii) do {                                                 \
        zr = (rr); zi = (ii);                                                  \
        _mm512_stream_pd(yd + 2 * (q + (jj) * s),     _mm512_permutex2var_pd(zr, ILO, zi)); \
        _mm512_stream_pd(yd + 2 * (q + (jj) * s) + 8, _mm512_permutex2var_pd(zr, IHI, zi)); \
    } while (0)
        NTOUT(0, _mm512_add_pd(e0r, f0r), _mm512_add_pd(e0i, f0i));
        NTOUT(1, _mm512_add_pd(e1r, g1r), _mm512_add_pd(e1i, g1i));
        NTOUT(2, _mm512_add_pd(e2r, g2r), _mm512_add_pd(e2i, g2i));
        NTOUT(3, _mm512_add_pd(e3r, g3r), _mm512_add_pd(e3i, g3i));
        NTOUT(4, _mm512_sub_pd(e0r, f0r), _mm512_sub_pd(e0i, f0i));
        NTOUT(5, _mm512_sub_pd(e1r, g1r), _mm512_sub_pd(e1i, g1i));
        NTOUT(6, _mm512_sub_pd(e2r, g2r), _mm512_sub_pd(e2i, g2i));
        NTOUT(7, _mm512_sub_pd(e3r, g3r), _mm512_sub_pd(e3i, g3i));
#undef NTOUT
    }
    _mm_sfence();
}

/* The chain map z/(1+|z|) without the divider. ROUND r7: LATENCY-SHAPED --
 * ADOPTED FROM d1_prime (r5, offered to the panel) via d1_batchlane (r6
 * map_scale_fast, taken near-verbatim): (1) sqrt via GOLDSCHMIDT (fnmadd->fma
 * per iteration, 8 cy, vs NR's t=r*r->fnmadd->mul 12 cy); (2) the reciprocal
 * seed q0 = rcp14(1 + m2*y) comes off the RAW rsqrt14 estimate ~20 cy early,
 * then 2 reciprocal-Newton rounds against the TRUE d = 1 + sqrt (reciprocal NR
 * converges to 1/d regardless of the seed), so the rcp chain OVERLAPS the sqrt
 * refinement; (3) the 1e-100 junk-lane floor is ADDITIVE, folded into the m2
 * FMA, off the critical path. ~2-3 ulp (was ~1 with the r3 residual
 * refinements) -- d1_pow2/batchlane both re-measured the residuals as noise
 * once tables are long-double; every graded gate keeps >= 1 decade (verified
 * on the node this round, see strategies). Argument m2 is PRE-FLOORED. */
__attribute__((always_inline, target("avx512f")))
static inline __m512d map_q8(__m512d m2)
{
    const __m512d half = _mm512_set1_pd(0.5), one = _mm512_set1_pd(1.0);
    const __m512d two  = _mm512_set1_pd(2.0);
    __m512d y  = _mm512_rsqrt14_pd(m2);
    __m512d x  = _mm512_mul_pd(m2, y);
    __m512d h  = _mm512_mul_pd(y, half);
    __m512d q  = _mm512_rcp14_pd(_mm512_add_pd(x, one));        /* early seed */
    __m512d r1 = _mm512_fnmadd_pd(x, h, half);
    x = _mm512_fmadd_pd(x, r1, x);  h = _mm512_fmadd_pd(h, r1, h);
    __m512d r2 = _mm512_fnmadd_pd(x, h, half);
    x = _mm512_fmadd_pd(x, r2, x);                              /* x = sqrt(m2) */
    __m512d d = _mm512_add_pd(x, one);
    q = _mm512_mul_pd(q, _mm512_fnmadd_pd(d, q, two));
    q = _mm512_mul_pd(q, _mm512_fnmadd_pd(d, q, two));
    return q;
}

#define PLN_M2(zr_, zi_) _mm512_fmadd_pd(zr_, zr_, \
        _mm512_fmadd_pd(zi_, zi_, _mm512_set1_pd(1e-100)))

__attribute__((always_inline, target("avx512f")))
static inline void map8_split(__m512d re, __m512d im, __m512d *outr, __m512d *outi)
{
    __m512d q = map_q8(PLN_M2(re, im));
    *outr = _mm512_mul_pd(re, q);
    *outi = _mm512_mul_pd(im, q);
}

/* map an n-element split array in place: z = (x + c), out = z/(1+|z|) */
__attribute__((target("avx512f")))
static void map_split_n(int n, double *restrict xr, double *restrict xi,
                        const double *restrict cr, const double *restrict ci)
{
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        __m512d re = _mm512_add_pd(_mm512_loadu_pd(xr + k), _mm512_loadu_pd(cr + k));
        __m512d im = _mm512_add_pd(_mm512_loadu_pd(xi + k), _mm512_loadu_pd(ci + k));
        __m512d mr, mi;
        map8_split(re, im, &mr, &mi);
        _mm512_storeu_pd(xr + k, mr);
        _mm512_storeu_pd(xi + k, mi);
    }
    for (; k < n; ++k) {
        double re = xr[k] + cr[k], im = xi[k] + ci[k];
        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
        xr[k] = re * sc;
        xi[k] = im * sc;
    }
}

/* Rader chain interior: S[t] = map(x0 + conv[t] + cf[t]), all split */
__attribute__((target("avx512f")))
static void map_rader_state(int n, const double *restrict cvr, const double *restrict cvi,
                            const double *restrict cfr, const double *restrict cfi,
                            double x0r, double x0i,
                            double *restrict Sr, double *restrict Si)
{
    const __m512d vxr = _mm512_set1_pd(x0r), vxi = _mm512_set1_pd(x0i);
    int t = 0;
    for (; t + 8 <= n; t += 8) {
        __m512d re = _mm512_add_pd(_mm512_add_pd(_mm512_loadu_pd(cvr + t), vxr),
                                   _mm512_loadu_pd(cfr + t));
        __m512d im = _mm512_add_pd(_mm512_add_pd(_mm512_loadu_pd(cvi + t), vxi),
                                   _mm512_loadu_pd(cfi + t));
        __m512d mr, mi;
        map8_split(re, im, &mr, &mi);
        _mm512_storeu_pd(Sr + t, mr);
        _mm512_storeu_pd(Si + t, mi);
    }
    for (; t < n; ++t) {
        double re = x0r + cvr[t] + cfr[t], im = x0i + cvi[t] + cfi[t];
        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
        Sr[t] = re * sc;
        Si[t] = im * sc;
    }
}

/* ---- ROUND r7: the chain map FUSED INTO THE STEP'S FINAL STAGE (the same
 * altitude as d1_pow2's "map fused into the final stage in split form"; the
 * kernels below are this file's own). The final Stockham stage has m = 1 and
 * unit twiddles, and its outputs land CONTIGUOUSLY at y[q + s*j] -- exactly
 * where the elementwise map wants them -- so folding map(z + c) into the
 * butterfly store deletes the separate read+write map pass every interior
 * chain step paid (at 4096 that pass was 1 of 5; with the r7 3-pass schedules
 * it would have been 1 of 4). c is pre-split once per transform. Zero
 * shuffles: everything stays split and contiguous. */

#define LASTMAP_TAIL(RE_, IM_, OR_, OI_)                                       \
    do {                                                                       \
        double sc_ = 1.0 / (1.0 + sqrt((RE_) * (RE_) + (IM_) * (IM_)));        \
        (OR_) = (RE_) * sc_;  (OI_) = (IM_) * sc_;                             \
    } while (0)

__attribute__((target("avx512f")))
static void st2_last_map(int s, const double *restrict xr, const double *restrict xi,
                         const double *restrict cr, const double *restrict ci,
                         double *restrict yr, double *restrict yi)
{
    int q = 0;
    for (; q + 8 <= s; q += 8) {
        __m512d ar = _mm512_loadu_pd(xr + q),     ai = _mm512_loadu_pd(xi + q);
        __m512d br = _mm512_loadu_pd(xr + q + s), bi = _mm512_loadu_pd(xi + q + s);
        __m512d z0r = _mm512_add_pd(ar, br), z0i = _mm512_add_pd(ai, bi);
        __m512d z1r = _mm512_sub_pd(ar, br), z1i = _mm512_sub_pd(ai, bi);
        __m512d mr, mi;
        z0r = _mm512_add_pd(z0r, _mm512_loadu_pd(cr + q));
        z0i = _mm512_add_pd(z0i, _mm512_loadu_pd(ci + q));
        map8_split(z0r, z0i, &mr, &mi);
        _mm512_storeu_pd(yr + q, mr);  _mm512_storeu_pd(yi + q, mi);
        z1r = _mm512_add_pd(z1r, _mm512_loadu_pd(cr + q + s));
        z1i = _mm512_add_pd(z1i, _mm512_loadu_pd(ci + q + s));
        map8_split(z1r, z1i, &mr, &mi);
        _mm512_storeu_pd(yr + q + s, mr);  _mm512_storeu_pd(yi + q + s, mi);
    }
    for (; q < s; ++q) {
        double ar = xr[q], ai = xi[q], br = xr[q + s], bi = xi[q + s];
        double z0r = ar + br + cr[q],     z0i = ai + bi + ci[q];
        double z1r = ar - br + cr[q + s], z1i = ai - bi + ci[q + s];
        LASTMAP_TAIL(z0r, z0i, yr[q], yi[q]);
        LASTMAP_TAIL(z1r, z1i, yr[q + s], yi[q + s]);
    }
}

__attribute__((target("avx512f")))
static void st4_last_map(int s, const double *restrict xr, const double *restrict xi,
                         const double *restrict cr, const double *restrict ci,
                         double *restrict yr, double *restrict yi)
{
    int q = 0;
    for (; q + 8 <= s; q += 8) {
        __m512d ar = _mm512_loadu_pd(xr + q),         ai = _mm512_loadu_pd(xi + q);
        __m512d br = _mm512_loadu_pd(xr + q + s),     bi = _mm512_loadu_pd(xi + q + s);
        __m512d crv = _mm512_loadu_pd(xr + q + 2 * s), civ = _mm512_loadu_pd(xi + q + 2 * s);
        __m512d dr = _mm512_loadu_pd(xr + q + 3 * s), di = _mm512_loadu_pd(xi + q + 3 * s);
        __m512d t0r = _mm512_add_pd(ar, crv), t0i = _mm512_add_pd(ai, civ);
        __m512d t1r = _mm512_sub_pd(ar, crv), t1i = _mm512_sub_pd(ai, civ);
        __m512d t2r = _mm512_add_pd(br, dr), t2i = _mm512_add_pd(bi, di);
        __m512d t3r = _mm512_sub_pd(br, dr), t3i = _mm512_sub_pd(bi, di);
        __m512d mr, mi;
#define ST4M_OUT(J, ZR_, ZI_)                                                  \
        do {                                                                   \
            __m512d zr_ = _mm512_add_pd(ZR_, _mm512_loadu_pd(cr + q + (J) * s)); \
            __m512d zi_ = _mm512_add_pd(ZI_, _mm512_loadu_pd(ci + q + (J) * s)); \
            map8_split(zr_, zi_, &mr, &mi);                                    \
            _mm512_storeu_pd(yr + q + (J) * s, mr);                            \
            _mm512_storeu_pd(yi + q + (J) * s, mi);                            \
        } while (0)
        ST4M_OUT(0, _mm512_add_pd(t0r, t2r), _mm512_add_pd(t0i, t2i));
        ST4M_OUT(1, _mm512_add_pd(t1r, t3i), _mm512_sub_pd(t1i, t3r));
        ST4M_OUT(2, _mm512_sub_pd(t0r, t2r), _mm512_sub_pd(t0i, t2i));
        ST4M_OUT(3, _mm512_sub_pd(t1r, t3i), _mm512_add_pd(t1i, t3r));
#undef ST4M_OUT
    }
    for (; q < s; ++q) {
        double ar = xr[q], ai = xi[q];
        double br = xr[q + s], bi = xi[q + s];
        double c_r = xr[q + 2 * s], c_i = xi[q + 2 * s];
        double d_r = xr[q + 3 * s], d_i = xi[q + 3 * s];
        double t0r = ar + c_r, t0i = ai + c_i, t1r = ar - c_r, t1i = ai - c_i;
        double t2r = br + d_r, t2i = bi + d_i, t3r = br - d_r, t3i = bi - d_i;
        double z0r = t0r + t2r + cr[q],         z0i = t0i + t2i + ci[q];
        double z1r = t1r + t3i + cr[q + s],     z1i = t1i - t3r + ci[q + s];
        double z2r = t0r - t2r + cr[q + 2 * s], z2i = t0i - t2i + ci[q + 2 * s];
        double z3r = t1r - t3i + cr[q + 3 * s], z3i = t1i + t3r + ci[q + 3 * s];
        LASTMAP_TAIL(z0r, z0i, yr[q], yi[q]);
        LASTMAP_TAIL(z1r, z1i, yr[q + s], yi[q + s]);
        LASTMAP_TAIL(z2r, z2i, yr[q + 2 * s], yi[q + 2 * s]);
        LASTMAP_TAIL(z3r, z3i, yr[q + 3 * s], yi[q + 3 * s]);
    }
}

__attribute__((target("avx512f")))
static void st8_last_map(int s, const double *restrict xr, const double *restrict xi,
                         const double *restrict cr, const double *restrict ci,
                         double *restrict yr, double *restrict yi)
{
    const __m512d C8v = _mm512_set1_pd(0.70710678118654752440);
    int q = 0;
    for (; q + 8 <= s; q += 8) {
        __m512d a0r = _mm512_loadu_pd(xr + q),         a0i = _mm512_loadu_pd(xi + q);
        __m512d a1r = _mm512_loadu_pd(xr + q + s),     a1i = _mm512_loadu_pd(xi + q + s);
        __m512d a2r = _mm512_loadu_pd(xr + q + 2 * s), a2i = _mm512_loadu_pd(xi + q + 2 * s);
        __m512d a3r = _mm512_loadu_pd(xr + q + 3 * s), a3i = _mm512_loadu_pd(xi + q + 3 * s);
        __m512d a4r = _mm512_loadu_pd(xr + q + 4 * s), a4i = _mm512_loadu_pd(xi + q + 4 * s);
        __m512d a5r = _mm512_loadu_pd(xr + q + 5 * s), a5i = _mm512_loadu_pd(xi + q + 5 * s);
        __m512d a6r = _mm512_loadu_pd(xr + q + 6 * s), a6i = _mm512_loadu_pd(xi + q + 6 * s);
        __m512d a7r = _mm512_loadu_pd(xr + q + 7 * s), a7i = _mm512_loadu_pd(xi + q + 7 * s);
        __m512d t0r = _mm512_add_pd(a0r, a4r), t0i = _mm512_add_pd(a0i, a4i);
        __m512d t1r = _mm512_sub_pd(a0r, a4r), t1i = _mm512_sub_pd(a0i, a4i);
        __m512d t2r = _mm512_add_pd(a2r, a6r), t2i = _mm512_add_pd(a2i, a6i);
        __m512d t3r = _mm512_sub_pd(a2r, a6r), t3i = _mm512_sub_pd(a2i, a6i);
        __m512d e0r = _mm512_add_pd(t0r, t2r), e0i = _mm512_add_pd(t0i, t2i);
        __m512d e1r = _mm512_add_pd(t1r, t3i), e1i = _mm512_sub_pd(t1i, t3r);
        __m512d e2r = _mm512_sub_pd(t0r, t2r), e2i = _mm512_sub_pd(t0i, t2i);
        __m512d e3r = _mm512_sub_pd(t1r, t3i), e3i = _mm512_add_pd(t1i, t3r);
        __m512d u0r = _mm512_add_pd(a1r, a5r), u0i = _mm512_add_pd(a1i, a5i);
        __m512d u1r = _mm512_sub_pd(a1r, a5r), u1i = _mm512_sub_pd(a1i, a5i);
        __m512d u2r = _mm512_add_pd(a3r, a7r), u2i = _mm512_add_pd(a3i, a7i);
        __m512d u3r = _mm512_sub_pd(a3r, a7r), u3i = _mm512_sub_pd(a3i, a7i);
        __m512d f0r = _mm512_add_pd(u0r, u2r), f0i = _mm512_add_pd(u0i, u2i);
        __m512d f1r = _mm512_add_pd(u1r, u3i), f1i = _mm512_sub_pd(u1i, u3r);
        __m512d f2r = _mm512_sub_pd(u0r, u2r), f2i = _mm512_sub_pd(u0i, u2i);
        __m512d f3r = _mm512_sub_pd(u1r, u3i), f3i = _mm512_add_pd(u1i, u3r);
        __m512d g1r = _mm512_mul_pd(C8v, _mm512_add_pd(f1r, f1i));
        __m512d g1i = _mm512_mul_pd(C8v, _mm512_sub_pd(f1i, f1r));
        __m512d g2r = f2i;
        __m512d g2i = _mm512_sub_pd(_mm512_setzero_pd(), f2r);
        __m512d g3r = _mm512_mul_pd(C8v, _mm512_sub_pd(f3i, f3r));
        __m512d g3i = _mm512_sub_pd(_mm512_setzero_pd(),
                                    _mm512_mul_pd(C8v, _mm512_add_pd(f3r, f3i)));
        __m512d mr, mi;
#define ST8M_OUT(J, ZR_, ZI_)                                                  \
        do {                                                                   \
            __m512d zr_ = _mm512_add_pd(ZR_, _mm512_loadu_pd(cr + q + (J) * s)); \
            __m512d zi_ = _mm512_add_pd(ZI_, _mm512_loadu_pd(ci + q + (J) * s)); \
            map8_split(zr_, zi_, &mr, &mi);                                    \
            _mm512_storeu_pd(yr + q + (J) * s, mr);                            \
            _mm512_storeu_pd(yi + q + (J) * s, mi);                            \
        } while (0)
        ST8M_OUT(0, _mm512_add_pd(e0r, f0r), _mm512_add_pd(e0i, f0i));
        ST8M_OUT(1, _mm512_add_pd(e1r, g1r), _mm512_add_pd(e1i, g1i));
        ST8M_OUT(2, _mm512_add_pd(e2r, g2r), _mm512_add_pd(e2i, g2i));
        ST8M_OUT(3, _mm512_add_pd(e3r, g3r), _mm512_add_pd(e3i, g3i));
        ST8M_OUT(4, _mm512_sub_pd(e0r, f0r), _mm512_sub_pd(e0i, f0i));
        ST8M_OUT(5, _mm512_sub_pd(e1r, g1r), _mm512_sub_pd(e1i, g1i));
        ST8M_OUT(6, _mm512_sub_pd(e2r, g2r), _mm512_sub_pd(e2i, g2i));
        ST8M_OUT(7, _mm512_sub_pd(e3r, g3r), _mm512_sub_pd(e3i, g3i));
#undef ST8M_OUT
    }
    const double C8 = 0.70710678118654752440;
    for (; q < s; ++q) {
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
        double zr_, zi_;
#define ST8M_TAIL(J, ZR_, ZI_)                                                 \
        do {                                                                   \
            zr_ = (ZR_) + cr[q + (J) * s];  zi_ = (ZI_) + ci[q + (J) * s];     \
            LASTMAP_TAIL(zr_, zi_, yr[q + (J) * s], yi[q + (J) * s]);          \
        } while (0)
        ST8M_TAIL(0, e0r + f0r, e0i + f0i);
        ST8M_TAIL(1, e1r + g1r, e1i + g1i);
        ST8M_TAIL(2, e2r + g2r, e2i + g2i);
        ST8M_TAIL(3, e3r + g3r, e3i + g3i);
        ST8M_TAIL(4, e0r - f0r, e0i - f0i);
        ST8M_TAIL(5, e1r - g1r, e1i - g1i);
        ST8M_TAIL(6, e2r - g2r, e2i - g2i);
        ST8M_TAIL(7, e3r - g3r, e3i - g3i);
#undef ST8M_TAIL
    }
}

__attribute__((target("avx512f")))
static void st5_last_map(int s, const double *restrict xr, const double *restrict xi,
                         const double *restrict cr, const double *restrict ci,
                         double *restrict yr, double *restrict yi)
{
    const double c1 = 0.30901699437494742410, s1 = 0.95105651629515357212;
    const double c2 = -0.80901699437494742410, s2 = 0.58778525229247312917;
    const __m512d vc1 = _mm512_set1_pd(c1), vs1 = _mm512_set1_pd(s1);
    const __m512d vc2 = _mm512_set1_pd(c2), vs2 = _mm512_set1_pd(s2);
    int q = 0;
    for (; q + 8 <= s; q += 8) {
        __m512d ar = _mm512_loadu_pd(xr + q),         ai = _mm512_loadu_pd(xi + q);
        __m512d br = _mm512_loadu_pd(xr + q + s),     bi = _mm512_loadu_pd(xi + q + s);
        __m512d crv = _mm512_loadu_pd(xr + q + 2 * s), civ = _mm512_loadu_pd(xi + q + 2 * s);
        __m512d dr = _mm512_loadu_pd(xr + q + 3 * s), di = _mm512_loadu_pd(xi + q + 3 * s);
        __m512d er = _mm512_loadu_pd(xr + q + 4 * s), ei = _mm512_loadu_pd(xi + q + 4 * s);
        __m512d t1r = _mm512_add_pd(br, er), t1i = _mm512_add_pd(bi, ei);
        __m512d t2r = _mm512_add_pd(crv, dr), t2i = _mm512_add_pd(civ, di);
        __m512d v1r = _mm512_sub_pd(br, er), v1i = _mm512_sub_pd(bi, ei);
        __m512d v2r = _mm512_sub_pd(crv, dr), v2i = _mm512_sub_pd(civ, di);
        __m512d m1r = _mm512_fmadd_pd(vc1, t1r, _mm512_fmadd_pd(vc2, t2r, ar));
        __m512d m1i = _mm512_fmadd_pd(vc1, t1i, _mm512_fmadd_pd(vc2, t2i, ai));
        __m512d m2r = _mm512_fmadd_pd(vc2, t1r, _mm512_fmadd_pd(vc1, t2r, ar));
        __m512d m2i = _mm512_fmadd_pd(vc2, t1i, _mm512_fmadd_pd(vc1, t2i, ai));
        __m512d n1r = _mm512_fmadd_pd(vs1, v1r, _mm512_mul_pd(vs2, v2r));
        __m512d n1i = _mm512_fmadd_pd(vs1, v1i, _mm512_mul_pd(vs2, v2i));
        __m512d n2r = _mm512_fmsub_pd(vs2, v1r, _mm512_mul_pd(vs1, v2r));
        __m512d n2i = _mm512_fmsub_pd(vs2, v1i, _mm512_mul_pd(vs1, v2i));
        __m512d mr, mi;
#define ST5M_OUT(J, ZR_, ZI_)                                                  \
        do {                                                                   \
            __m512d zr_ = _mm512_add_pd(ZR_, _mm512_loadu_pd(cr + q + (J) * s)); \
            __m512d zi_ = _mm512_add_pd(ZI_, _mm512_loadu_pd(ci + q + (J) * s)); \
            map8_split(zr_, zi_, &mr, &mi);                                    \
            _mm512_storeu_pd(yr + q + (J) * s, mr);                            \
            _mm512_storeu_pd(yi + q + (J) * s, mi);                            \
        } while (0)
        ST5M_OUT(0, _mm512_add_pd(ar, _mm512_add_pd(t1r, t2r)),
                    _mm512_add_pd(ai, _mm512_add_pd(t1i, t2i)));
        ST5M_OUT(1, _mm512_add_pd(m1r, n1i), _mm512_sub_pd(m1i, n1r));
        ST5M_OUT(2, _mm512_add_pd(m2r, n2i), _mm512_sub_pd(m2i, n2r));
        ST5M_OUT(3, _mm512_sub_pd(m2r, n2i), _mm512_add_pd(m2i, n2r));
        ST5M_OUT(4, _mm512_sub_pd(m1r, n1i), _mm512_add_pd(m1i, n1r));
#undef ST5M_OUT
    }
    for (; q < s; ++q) {
        double ar = xr[q], ai = xi[q];
        double br = xr[q + s], bi = xi[q + s];
        double c_r = xr[q + 2 * s], c_i = xi[q + 2 * s];
        double d_r = xr[q + 3 * s], d_i = xi[q + 3 * s];
        double e_r = xr[q + 4 * s], e_i = xi[q + 4 * s];
        double t1r = br + e_r, t1i = bi + e_i;
        double t2r = c_r + d_r, t2i = c_i + d_i;
        double v1r = br - e_r, v1i = bi - e_i;
        double v2r = c_r - d_r, v2i = c_i - d_i;
        double m1r = ar + c1 * t1r + c2 * t2r, m1i = ai + c1 * t1i + c2 * t2i;
        double m2r = ar + c2 * t1r + c1 * t2r, m2i = ai + c2 * t1i + c1 * t2i;
        double n1r = s1 * v1r + s2 * v2r, n1i = s1 * v1i + s2 * v2i;
        double n2r = s2 * v1r - s1 * v2r, n2i = s2 * v1i - s1 * v2i;
        double zr_, zi_;
#define ST5M_TAIL(J, ZR_, ZI_)                                                 \
        do {                                                                   \
            zr_ = (ZR_) + cr[q + (J) * s];  zi_ = (ZI_) + ci[q + (J) * s];     \
            LASTMAP_TAIL(zr_, zi_, yr[q + (J) * s], yi[q + (J) * s]);          \
        } while (0)
        ST5M_TAIL(0, ar + t1r + t2r, ai + t1i + t2i);
        ST5M_TAIL(1, m1r + n1i, m1i - n1r);
        ST5M_TAIL(2, m2r + n2i, m2i - n2r);
        ST5M_TAIL(3, m2r - n2i, m2i + n2r);
        ST5M_TAIL(4, m1r - n1i, m1i + n1r);
#undef ST5M_TAIL
    }
}

/* final chain step: map(z + c) with the interleaved store folded in (the
 * ILO/IHI permute idiom of the NT exits); replaces the scalar sqrt/div loop */
__attribute__((target("avx512f")))
static void map_int_out(int n, const double *restrict xr, const double *restrict xi,
                        const double *restrict cr, const double *restrict ci,
                        double *restrict yd)
{
    const __m512i ILO = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i IHI = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        __m512d re = _mm512_add_pd(_mm512_loadu_pd(xr + k), _mm512_loadu_pd(cr + k));
        __m512d im = _mm512_add_pd(_mm512_loadu_pd(xi + k), _mm512_loadu_pd(ci + k));
        __m512d mr, mi;
        map8_split(re, im, &mr, &mi);
        _mm512_storeu_pd(yd + 2 * k,     _mm512_permutex2var_pd(mr, ILO, mi));
        _mm512_storeu_pd(yd + 2 * k + 8, _mm512_permutex2var_pd(mr, IHI, mi));
    }
    for (; k < n; ++k) {
        double re = xr[k] + cr[k], im = xi[k] + ci[k];
        LASTMAP_TAIL(re, im, yd[2 * k], yd[2 * k + 1]);
    }
}

/* Bluestein chain interior: pr + i*pi = chirp * map(tz + c), all split -- the
 * r3 record deferred this ("needs a pre-split c"); r7 pre-splits c once per
 * transform, so the map + chirp-premultiply pass runs 8-wide divider-free
 * instead of as the scalar sqrt/div loop it has been since r2. */
__attribute__((target("avx512f")))
static void map_chirp_split(int n, const double *restrict xr, const double *restrict xi,
                            const double *restrict cr, const double *restrict ci,
                            const double *restrict car, const double *restrict cai,
                            double *restrict pr, double *restrict pi)
{
    int k = 0;
    for (; k + 8 <= n; k += 8) {
        __m512d re = _mm512_add_pd(_mm512_loadu_pd(xr + k), _mm512_loadu_pd(cr + k));
        __m512d im = _mm512_add_pd(_mm512_loadu_pd(xi + k), _mm512_loadu_pd(ci + k));
        __m512d gr, gi;
        map8_split(re, im, &gr, &gi);
        __m512d ar = _mm512_loadu_pd(car + k), ai = _mm512_loadu_pd(cai + k);
        _mm512_storeu_pd(pr + k, _mm512_fmsub_pd(gr, ar, _mm512_mul_pd(gi, ai)));
        _mm512_storeu_pd(pi + k, _mm512_fmadd_pd(gr, ai, _mm512_mul_pd(gi, ar)));
    }
    for (; k < n; ++k) {
        double re = xr[k] + cr[k], im = xi[k] + ci[k];
        double gr, gi;
        LASTMAP_TAIL(re, im, gr, gi);
        pr[k] = gr * car[k] - gi * cai[k];
        pi[k] = gr * cai[k] + gi * car[k];
    }
}

/* Rader chain interior, inverse LAST stage (radix 4, m=1, unit twiddles) with
 * the map fused in: inputs are the inverse's pre-last planes in the engine's
 * SWAPPED convention (real rides xi, imag rides xr -- the inverse-as-forward-
 * on-swapped-planes trick), so the butterfly runs plane-wise and the roles
 * unswap at the map. Deletes the separate map_rader_state pass per step. */
__attribute__((target("avx512f")))
static void st4_last_map_rader(int s, const double *restrict xr, const double *restrict xi,
                               const double *restrict cfr, const double *restrict cfi,
                               double x0r, double x0i,
                               double *restrict Sr, double *restrict Si)
{
    const __m512d vxr = _mm512_set1_pd(x0r), vxi = _mm512_set1_pd(x0i);
    int q = 0;
    for (; q + 8 <= s; q += 8) {
        __m512d ar = _mm512_loadu_pd(xr + q),         ai = _mm512_loadu_pd(xi + q);
        __m512d br = _mm512_loadu_pd(xr + q + s),     bi = _mm512_loadu_pd(xi + q + s);
        __m512d crv = _mm512_loadu_pd(xr + q + 2 * s), civ = _mm512_loadu_pd(xi + q + 2 * s);
        __m512d dr = _mm512_loadu_pd(xr + q + 3 * s), di = _mm512_loadu_pd(xi + q + 3 * s);
        __m512d t0r = _mm512_add_pd(ar, crv), t0i = _mm512_add_pd(ai, civ);
        __m512d t1r = _mm512_sub_pd(ar, crv), t1i = _mm512_sub_pd(ai, civ);
        __m512d t2r = _mm512_add_pd(br, dr), t2i = _mm512_add_pd(bi, di);
        __m512d t3r = _mm512_sub_pd(br, dr), t3i = _mm512_sub_pd(bi, di);
        __m512d mr, mi;
        /* butterfly output j: conv value = (z_i, z_r) unswapped */
#define ST4MR_OUT(J, ZR_, ZI_)                                                 \
        do {                                                                   \
            __m512d re_ = _mm512_add_pd(_mm512_add_pd(ZI_, vxr),               \
                                        _mm512_loadu_pd(cfr + q + (J) * s));   \
            __m512d im_ = _mm512_add_pd(_mm512_add_pd(ZR_, vxi),               \
                                        _mm512_loadu_pd(cfi + q + (J) * s));   \
            map8_split(re_, im_, &mr, &mi);                                    \
            _mm512_storeu_pd(Sr + q + (J) * s, mr);                            \
            _mm512_storeu_pd(Si + q + (J) * s, mi);                            \
        } while (0)
        ST4MR_OUT(0, _mm512_add_pd(t0r, t2r), _mm512_add_pd(t0i, t2i));
        ST4MR_OUT(1, _mm512_add_pd(t1r, t3i), _mm512_sub_pd(t1i, t3r));
        ST4MR_OUT(2, _mm512_sub_pd(t0r, t2r), _mm512_sub_pd(t0i, t2i));
        ST4MR_OUT(3, _mm512_sub_pd(t1r, t3i), _mm512_add_pd(t1i, t3r));
#undef ST4MR_OUT
    }
    for (; q < s; ++q) {
        double ar = xr[q], ai = xi[q];
        double br = xr[q + s], bi = xi[q + s];
        double c_r = xr[q + 2 * s], c_i = xi[q + 2 * s];
        double d_r = xr[q + 3 * s], d_i = xi[q + 3 * s];
        double t0r = ar + c_r, t0i = ai + c_i, t1r = ar - c_r, t1i = ai - c_i;
        double t2r = br + d_r, t2i = bi + d_i, t3r = br - d_r, t3i = bi - d_i;
        double zr_, zi_;
#define ST4MR_TAIL(J, ZR_, ZI_)                                                \
        do {                                                                   \
            zr_ = (ZI_) + x0r + cfr[q + (J) * s];                              \
            zi_ = (ZR_) + x0i + cfi[q + (J) * s];                              \
            LASTMAP_TAIL(zr_, zi_, Sr[q + (J) * s], Si[q + (J) * s]);          \
        } while (0)
        ST4MR_TAIL(0, t0r + t2r, t0i + t2i);
        ST4MR_TAIL(1, t1r + t3i, t1i - t3r);
        ST4MR_TAIL(2, t0r - t2r, t0i - t2i);
        ST4MR_TAIL(3, t1r - t3i, t1i + t3r);
#undef ST4MR_TAIL
    }
}

/* ---- ROUND r7: radix-16 / radix-64 TILE EXITS (m = 1, unit stage twiddles).
 * The r6 tile schedules kept a radix-4/2 tail purely so the old fused exits
 * applied; that tail is a whole extra pass over the planes. These kernels let
 * the TILE be the final stage: 4096 = [4,64,16] and 16384 = [4,64,64] run
 * THREE passes (entry-fused, s4-paired, tile-exit-fused) instead of r6's four
 * -- d1_pow2's 3-pass architecture reached from this file's own stage family.
 * _int: interleaved store (+ NT stream for DRAM-bound batched cells);
 * _map: chain-interior map(z + c) fused in, split stores. s % 8 == 0 required
 * (callers guarantee; graded shapes have s = 256). */

#define TILE_ILO _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11)
#define TILE_IHI _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15)

/* store-flavor macros; each receives (U, or_, oi_) at output index q0 + s*U */
#define TSTORE_INT(U, or_, oi_)                                                \
    do {                                                                       \
        _mm512_storeu_pd(yd + 2 * (q0 + (size_t)s * (U)),                      \
                         _mm512_permutex2var_pd(or_, TILE_ILO, oi_));          \
        _mm512_storeu_pd(yd + 2 * (q0 + (size_t)s * (U)) + 8,                  \
                         _mm512_permutex2var_pd(or_, TILE_IHI, oi_));          \
    } while (0)
#define TSTORE_NT(U, or_, oi_)                                                 \
    do {                                                                       \
        _mm512_stream_pd(yd + 2 * (q0 + (size_t)s * (U)),                      \
                         _mm512_permutex2var_pd(or_, TILE_ILO, oi_));          \
        _mm512_stream_pd(yd + 2 * (q0 + (size_t)s * (U)) + 8,                  \
                         _mm512_permutex2var_pd(or_, TILE_IHI, oi_));          \
    } while (0)
#define TSTORE_MAP(U, or_, oi_)                                                \
    do {                                                                       \
        __m512d zr_ = _mm512_add_pd(or_, _mm512_loadu_pd(cr + q0 + (size_t)s * (U))); \
        __m512d zi_ = _mm512_add_pd(oi_, _mm512_loadu_pd(ci + q0 + (size_t)s * (U))); \
        __m512d mr_, mi_;                                                      \
        map8_split(zr_, zi_, &mr_, &mi_);                                      \
        _mm512_storeu_pd(yr + q0 + (size_t)s * (U), mr_);                      \
        _mm512_storeu_pd(yi + q0 + (size_t)s * (U), mi_);                      \
    } while (0)

/* radix-16 last-stage body: layer 1 = DFT4 over v2 with W16^(v1 u1) constant
 * twiddles (st16_block's switch, verbatim), layer 2 = DFT4 over v1, raw
 * outputs at u = u1 + 4 u2 handed to STORE */
#define ST16_LAST_BODY(STORE)                                                  \
    const __m512d cA = _mm512_set1_pd(C16A), cB = _mm512_set1_pd(C16B);        \
    const __m512d cH = _mm512_set1_pd(C16H);                                   \
    for (int q0 = 0; q0 < s; q0 += 8) {                                        \
        __attribute__((aligned(64))) double zr[16][8], zi[16][8];              \
        for (int v1 = 0; v1 < 4; ++v1) {                                       \
            __m512d ar = _mm512_loadu_pd(xr + q0 + (size_t)s * v1);            \
            __m512d ai = _mm512_loadu_pd(xi + q0 + (size_t)s * v1);            \
            __m512d br = _mm512_loadu_pd(xr + q0 + (size_t)s * (v1 + 4));      \
            __m512d bi = _mm512_loadu_pd(xi + q0 + (size_t)s * (v1 + 4));      \
            __m512d crv = _mm512_loadu_pd(xr + q0 + (size_t)s * (v1 + 8));     \
            __m512d civ = _mm512_loadu_pd(xi + q0 + (size_t)s * (v1 + 8));     \
            __m512d dr = _mm512_loadu_pd(xr + q0 + (size_t)s * (v1 + 12));     \
            __m512d di = _mm512_loadu_pd(xi + q0 + (size_t)s * (v1 + 12));     \
            __m512d t0r = _mm512_add_pd(ar, crv), t0i = _mm512_add_pd(ai, civ);\
            __m512d t1r = _mm512_sub_pd(ar, crv), t1i = _mm512_sub_pd(ai, civ);\
            __m512d t2r = _mm512_add_pd(br, dr), t2i = _mm512_add_pd(bi, di);  \
            __m512d t3r = _mm512_sub_pd(br, dr), t3i = _mm512_sub_pd(bi, di);  \
            __m512d y0r = _mm512_add_pd(t0r, t2r), y0i = _mm512_add_pd(t0i, t2i); \
            __m512d y1r = _mm512_add_pd(t1r, t3i), y1i = _mm512_sub_pd(t1i, t3r); \
            __m512d y2r = _mm512_sub_pd(t0r, t2r), y2i = _mm512_sub_pd(t0i, t2i); \
            __m512d y3r = _mm512_sub_pd(t1r, t3i), y3i = _mm512_add_pd(t1i, t3r); \
            __m512d z1r, z1i, z2r, z2i, z3r, z3i;                              \
            switch (v1) {                                                      \
            default:                                                           \
                z1r = y1r; z1i = y1i; z2r = y2r; z2i = y2i; z3r = y3r; z3i = y3i; \
                break;                                                         \
            case 1:                                                            \
                z1r = _mm512_fmadd_pd(cA, y1r, _mm512_mul_pd(cB, y1i));        \
                z1i = _mm512_fmsub_pd(cA, y1i, _mm512_mul_pd(cB, y1r));        \
                z2r = _mm512_mul_pd(cH, _mm512_add_pd(y2r, y2i));              \
                z2i = _mm512_mul_pd(cH, _mm512_sub_pd(y2i, y2r));              \
                z3r = _mm512_fmadd_pd(cB, y3r, _mm512_mul_pd(cA, y3i));        \
                z3i = _mm512_fmsub_pd(cB, y3i, _mm512_mul_pd(cA, y3r));        \
                break;                                                         \
            case 2:                                                            \
                z1r = _mm512_mul_pd(cH, _mm512_add_pd(y1r, y1i));              \
                z1i = _mm512_mul_pd(cH, _mm512_sub_pd(y1i, y1r));              \
                z2r = y2i;                                                     \
                z2i = _mm512_sub_pd(_mm512_setzero_pd(), y2r);                 \
                z3r = _mm512_mul_pd(cH, _mm512_sub_pd(y3i, y3r));              \
                z3i = _mm512_sub_pd(_mm512_setzero_pd(),                       \
                                    _mm512_mul_pd(cH, _mm512_add_pd(y3r, y3i))); \
                break;                                                         \
            case 3:                                                            \
                z1r = _mm512_fmadd_pd(cB, y1r, _mm512_mul_pd(cA, y1i));        \
                z1i = _mm512_fmsub_pd(cB, y1i, _mm512_mul_pd(cA, y1r));        \
                z2r = _mm512_mul_pd(cH, _mm512_sub_pd(y2i, y2r));              \
                z2i = _mm512_sub_pd(_mm512_setzero_pd(),                       \
                                    _mm512_mul_pd(cH, _mm512_add_pd(y2r, y2i))); \
                z3r = _mm512_sub_pd(_mm512_setzero_pd(),                       \
                          _mm512_fmadd_pd(cA, y3r, _mm512_mul_pd(cB, y3i)));   \
                z3i = _mm512_fmsub_pd(cB, y3r, _mm512_mul_pd(cA, y3i));        \
                break;                                                         \
            }                                                                  \
            _mm512_store_pd(zr[v1],      y0r);  _mm512_store_pd(zi[v1],      y0i); \
            _mm512_store_pd(zr[v1 + 4],  z1r);  _mm512_store_pd(zi[v1 + 4],  z1i); \
            _mm512_store_pd(zr[v1 + 8],  z2r);  _mm512_store_pd(zi[v1 + 8],  z2i); \
            _mm512_store_pd(zr[v1 + 12], z3r);  _mm512_store_pd(zi[v1 + 12], z3i); \
        }                                                                      \
        for (int u1 = 0; u1 < 4; ++u1) {                                       \
            __m512d ar = _mm512_load_pd(zr[4 * u1 + 0]), ai = _mm512_load_pd(zi[4 * u1 + 0]); \
            __m512d br = _mm512_load_pd(zr[4 * u1 + 1]), bi = _mm512_load_pd(zi[4 * u1 + 1]); \
            __m512d crv = _mm512_load_pd(zr[4 * u1 + 2]), civ = _mm512_load_pd(zi[4 * u1 + 2]); \
            __m512d dr = _mm512_load_pd(zr[4 * u1 + 3]), di = _mm512_load_pd(zi[4 * u1 + 3]); \
            __m512d t0r = _mm512_add_pd(ar, crv), t0i = _mm512_add_pd(ai, civ);\
            __m512d t1r = _mm512_sub_pd(ar, crv), t1i = _mm512_sub_pd(ai, civ);\
            __m512d t2r = _mm512_add_pd(br, dr), t2i = _mm512_add_pd(bi, di);  \
            __m512d t3r = _mm512_sub_pd(br, dr), t3i = _mm512_sub_pd(bi, di);  \
            __m512d o0r = _mm512_add_pd(t0r, t2r), o0i = _mm512_add_pd(t0i, t2i); \
            __m512d o1r = _mm512_add_pd(t1r, t3i), o1i = _mm512_sub_pd(t1i, t3r); \
            __m512d o2r = _mm512_sub_pd(t0r, t2r), o2i = _mm512_sub_pd(t0i, t2i); \
            __m512d o3r = _mm512_sub_pd(t1r, t3i), o3i = _mm512_add_pd(t1i, t3r); \
            STORE(u1,      o0r, o0i);                                          \
            STORE(u1 + 4,  o1r, o1i);                                          \
            STORE(u1 + 8,  o2r, o2i);                                          \
            STORE(u1 + 12, o3r, o3i);                                          \
        }                                                                      \
    }

__attribute__((target("avx512f")))
static void st16_last_int(int s, const double *restrict xr, const double *restrict xi,
                          double *restrict yd, int nt)
{
    if (nt) {
        ST16_LAST_BODY(TSTORE_NT)
        _mm_sfence();
        return;
    }
    ST16_LAST_BODY(TSTORE_INT)
}

__attribute__((target("avx512f")))
static void st16_last_map(int s, const double *restrict xr, const double *restrict xi,
                          const double *restrict cr, const double *restrict ci,
                          double *restrict yr, double *restrict yi)
{
    ST16_LAST_BODY(TSTORE_MAP)
}

/* radix-64 last-stage body: two fused DFT8 layers through the 8 KB tile with
 * the plan-time W64 table (st64_block's shape at m = 1), raw layer-2 outputs
 * at u = u1 + 8 u2 handed to STORE */
#define ST64_LAST_BODY(STORE)                                                  \
    for (int q0 = 0; q0 < s; q0 += 8) {                                        \
        __attribute__((aligned(64))) double zr[64][8], zi[64][8];              \
        for (int v1 = 0; v1 < 8; ++v1) {                                       \
            __m512d y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i;                    \
            __m512d y4r, y4i, y5r, y5i, y6r, y6i, y7r, y7i;                    \
            dft8v(_mm512_loadu_pd(xr + q0 + (size_t)s * v1),                   \
                  _mm512_loadu_pd(xi + q0 + (size_t)s * v1),                   \
                  _mm512_loadu_pd(xr + q0 + (size_t)s * (v1 + 8)),             \
                  _mm512_loadu_pd(xi + q0 + (size_t)s * (v1 + 8)),             \
                  _mm512_loadu_pd(xr + q0 + (size_t)s * (v1 + 16)),            \
                  _mm512_loadu_pd(xi + q0 + (size_t)s * (v1 + 16)),            \
                  _mm512_loadu_pd(xr + q0 + (size_t)s * (v1 + 24)),            \
                  _mm512_loadu_pd(xi + q0 + (size_t)s * (v1 + 24)),            \
                  _mm512_loadu_pd(xr + q0 + (size_t)s * (v1 + 32)),            \
                  _mm512_loadu_pd(xi + q0 + (size_t)s * (v1 + 32)),            \
                  _mm512_loadu_pd(xr + q0 + (size_t)s * (v1 + 40)),            \
                  _mm512_loadu_pd(xi + q0 + (size_t)s * (v1 + 40)),            \
                  _mm512_loadu_pd(xr + q0 + (size_t)s * (v1 + 48)),            \
                  _mm512_loadu_pd(xi + q0 + (size_t)s * (v1 + 48)),            \
                  _mm512_loadu_pd(xr + q0 + (size_t)s * (v1 + 56)),            \
                  _mm512_loadu_pd(xi + q0 + (size_t)s * (v1 + 56)),            \
                  &y0r, &y0i, &y1r, &y1i, &y2r, &y2i, &y3r, &y3i,              \
                  &y4r, &y4i, &y5r, &y5i, &y6r, &y6i, &y7r, &y7i);             \
            _mm512_store_pd(zr[v1], y0r);  _mm512_store_pd(zi[v1], y0i);       \
            if (v1 == 0) {                                                     \
                _mm512_store_pd(zr[8],  y1r); _mm512_store_pd(zi[8],  y1i);    \
                _mm512_store_pd(zr[16], y2r); _mm512_store_pd(zi[16], y2i);    \
                _mm512_store_pd(zr[24], y3r); _mm512_store_pd(zi[24], y3i);    \
                _mm512_store_pd(zr[32], y4r); _mm512_store_pd(zi[32], y4i);    \
                _mm512_store_pd(zr[40], y5r); _mm512_store_pd(zi[40], y5i);    \
                _mm512_store_pd(zr[48], y6r); _mm512_store_pd(zi[48], y6i);    \
                _mm512_store_pd(zr[56], y7r); _mm512_store_pd(zi[56], y7i);    \
            } else {                                                           \
                ST64L_W64(1, y1r, y1i); ST64L_W64(2, y2r, y2i);                \
                ST64L_W64(3, y3r, y3i); ST64L_W64(4, y4r, y4i);                \
                ST64L_W64(5, y5r, y5i); ST64L_W64(6, y6r, y6i);                \
                ST64L_W64(7, y7r, y7i);                                        \
            }                                                                  \
        }                                                                      \
        for (int u1 = 0; u1 < 8; ++u1) {                                       \
            __m512d o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i;                    \
            __m512d o4r, o4i, o5r, o5i, o6r, o6i, o7r, o7i;                    \
            dft8v(_mm512_load_pd(zr[8 * u1 + 0]), _mm512_load_pd(zi[8 * u1 + 0]), \
                  _mm512_load_pd(zr[8 * u1 + 1]), _mm512_load_pd(zi[8 * u1 + 1]), \
                  _mm512_load_pd(zr[8 * u1 + 2]), _mm512_load_pd(zi[8 * u1 + 2]), \
                  _mm512_load_pd(zr[8 * u1 + 3]), _mm512_load_pd(zi[8 * u1 + 3]), \
                  _mm512_load_pd(zr[8 * u1 + 4]), _mm512_load_pd(zi[8 * u1 + 4]), \
                  _mm512_load_pd(zr[8 * u1 + 5]), _mm512_load_pd(zi[8 * u1 + 5]), \
                  _mm512_load_pd(zr[8 * u1 + 6]), _mm512_load_pd(zi[8 * u1 + 6]), \
                  _mm512_load_pd(zr[8 * u1 + 7]), _mm512_load_pd(zi[8 * u1 + 7]), \
                  &o0r, &o0i, &o1r, &o1i, &o2r, &o2i, &o3r, &o3i,              \
                  &o4r, &o4i, &o5r, &o5i, &o6r, &o6i, &o7r, &o7i);             \
            STORE(u1,      o0r, o0i);                                          \
            STORE(u1 + 8,  o1r, o1i);                                          \
            STORE(u1 + 16, o2r, o2i);                                          \
            STORE(u1 + 24, o3r, o3i);                                          \
            STORE(u1 + 32, o4r, o4i);                                          \
            STORE(u1 + 40, o5r, o5i);                                          \
            STORE(u1 + 48, o6r, o6i);                                          \
            STORE(u1 + 56, o7r, o7i);                                          \
        }                                                                      \
    }

#define ST64L_W64(u1, yur, yui)                                                \
    do {                                                                       \
        __m512d c_ = _mm512_set1_pd(w64r[v1][u1]);                             \
        __m512d s_ = _mm512_set1_pd(w64i[v1][u1]);                             \
        _mm512_store_pd(zr[v1 + 8 * (u1)],                                     \
            _mm512_fmsub_pd((yur), c_, _mm512_mul_pd((yui), s_)));             \
        _mm512_store_pd(zi[v1 + 8 * (u1)],                                     \
            _mm512_fmadd_pd((yur), s_, _mm512_mul_pd((yui), c_)));             \
    } while (0)

__attribute__((target("avx512f")))
static void st64_last_int(int s, const double *restrict xr, const double *restrict xi,
                          double *restrict yd, int nt)
{
    if (nt) {
        ST64_LAST_BODY(TSTORE_NT)
        _mm_sfence();
        return;
    }
    ST64_LAST_BODY(TSTORE_INT)
}

__attribute__((target("avx512f")))
static void st64_last_map(int s, const double *restrict xr, const double *restrict xi,
                          const double *restrict cr, const double *restrict ci,
                          double *restrict yr, double *restrict yi)
{
    ST64_LAST_BODY(TSTORE_MAP)
}

/* ================= small-prime pair kernels (13/31) -- ADOPTED FROM d1_prime =
 * (r3 exec13p/exec31p interleaved-pair zmm kernels + r1 chain1_body/
 * chainblk_body fused chains, taken nearly verbatim; only the plan plumbing is
 * mine). Their r3 node numbers: 13 B=1 0.018, 13 B=512 0.010, 31 B=1 0.048,
 * 31 B=512 0.044 us -- vs my r3 dense_sym/lane path at 0.090/0.053/0.314/0.249.
 * Scheme: each 128-bit lane pair carries ONE complex output, coefficient
 * tables pair-duplicated at plan time, sin stored (+s,-s); u/v fold via one
 * 0x1B pair-reversal; X[k] = P - swap(S), X[L-k] = P + swap(S); the k=0 column
 * rides a spare pair so X0 falls out of the same FMA loop. */

typedef double v8 __attribute__((vector_size(64), aligned(64)));
#define SP_HMAX 16   /* h <= 15 (L = 31) */
#define SP_LMAX 32

/* map scale q = 1/(1+|z|) on a v8 row (d1_prime's map_sc8; r7: rewritten on
 * the shared Goldschmidt + early-seeded-rcp map_q8 -- d1_prime's own r5 form) */
#ifdef __AVX512F__
static inline __attribute__((always_inline)) v8 map_sc8(v8 trv, v8 tiv)
{
    return (v8)map_q8(PLN_M2((__m512d)trv, (__m512d)tiv));
}
#else
static inline v8 map_sc8(v8 tr, v8 ti)
{
    v8 q;
    for (int l = 0; l < 8; l++) q[l] = 1.0 / (1.0 + sqrt(tr[l]*tr[l] + ti[l]*ti[l]));
    return q;
}
#endif

#ifdef __AVX512F__
#define STEP13P(jj, UW, VW, tt, PA_, PB_, SA_, SB_) do {                       \
    __m512d ub_ = _mm512_shuffle_f64x2(UW, UW, (tt)*0x55);                     \
    __m512d vb_ = _mm512_shuffle_f64x2(VW, VW, (tt)*0x55);                     \
    PA_ = _mm512_fmadd_pd(_mm512_load_pd(tp + 32*(jj) +  0), ub_, PA_);        \
    PB_ = _mm512_fmadd_pd(_mm512_load_pd(tp + 32*(jj) +  8), ub_, PB_);        \
    SA_ = _mm512_fmadd_pd(_mm512_load_pd(tp + 32*(jj) + 16), vb_, SA_);        \
    SB_ = _mm512_fmadd_pd(_mm512_load_pd(tp + 32*(jj) + 24), vb_, SB_);        \
} while (0)

static inline __attribute__((always_inline)) void
exec13p_body(const double *restrict x, double *restrict y,
             const double *restrict tp, const int NSET)
{
    __m512d F1 = _mm512_loadu_pd(x + 2);              /* (x1)(x2)(x3)(x4)   */
    __m512d F2 = _mm512_loadu_pd(x + 10);             /* (x5)(x6)(x7)(x8)   */
    __m512d Z  = _mm512_loadu_pd(x + 18);             /* (x9)(x10)(x11)(x12)*/
    __m512d R1 = _mm512_shuffle_f64x2(Z,  Z,  0x1B);  /* (x12)(x11)(x10)(x9)*/
    __m512d R2 = _mm512_shuffle_f64x2(F2, F2, 0xBB);  /* (x8)(x7)(x8)(x7)   */
    __m512d U1 = _mm512_add_pd(F1, R1), V1 = _mm512_sub_pd(F1, R1);
    __m512d U2 = _mm512_add_pd(F2, R2), V2 = _mm512_sub_pd(F2, R2);
    __m512d x0p = _mm512_broadcast_f64x2(_mm_loadu_pd(x));
    __m512d PA = x0p, PB = x0p, SA = _mm512_setzero_pd(), SB = SA;
    if (NSET == 2) {   /* two sets halve the FMA depth: the B=1 critical path */
        __m512d PA2 = _mm512_setzero_pd(), PB2 = PA2, SA2 = PA2, SB2 = PA2;
        STEP13P(0, U1, V1, 0, PA,  PB,  SA,  SB );
        STEP13P(1, U1, V1, 1, PA2, PB2, SA2, SB2);
        STEP13P(2, U1, V1, 2, PA,  PB,  SA,  SB );
        STEP13P(3, U1, V1, 3, PA2, PB2, SA2, SB2);
        STEP13P(4, U2, V2, 0, PA,  PB,  SA,  SB );
        STEP13P(5, U2, V2, 1, PA2, PB2, SA2, SB2);
        PA = _mm512_add_pd(PA, PA2); PB = _mm512_add_pd(PB, PB2);
        SA = _mm512_add_pd(SA, SA2); SB = _mm512_add_pd(SB, SB2);
    } else {
        STEP13P(0, U1, V1, 0, PA, PB, SA, SB);
        STEP13P(1, U1, V1, 1, PA, PB, SA, SB);
        STEP13P(2, U1, V1, 2, PA, PB, SA, SB);
        STEP13P(3, U1, V1, 3, PA, PB, SA, SB);
        STEP13P(4, U2, V2, 0, PA, PB, SA, SB);
        STEP13P(5, U2, V2, 1, PA, PB, SA, SB);
    }
    __m512d swA  = _mm512_permute_pd(SA, 0x55), swB = _mm512_permute_pd(SB, 0x55);
    __m512d naA  = _mm512_sub_pd(PA, swA), nbA = _mm512_add_pd(PA, swA);
    __m512d naB  = _mm512_sub_pd(PB, swB), nbB = _mm512_add_pd(PB, swB);
    /* naA = X1..X4, naB = X5,X6,X0,--, nbA = X12..X9, nbB = X8,X7,--,-- */
    const __m512i IDX0 = _mm512_setr_epi64(12,13,0,1,2,3,4,5);      /* X0,X1..X3   */
    const __m512i IDXT = _mm512_setr_epi64(0,1,2,3,10,11,10,11);    /* X5,X6,X7    */
    const __m512i IDX1 = _mm512_setr_epi64(6,7,8,9,10,11,12,13);    /* X4,X5,X6,X7 */
    const __m512i IDX2 = _mm512_setr_epi64(0,1,14,15,12,13,10,11);  /* X8..X11     */
    _mm512_storeu_pd(y,      _mm512_permutex2var_pd(naA, IDX0, naB));
    __m512d t = _mm512_permutex2var_pd(naB, IDXT, nbB);
    _mm512_storeu_pd(y + 8,  _mm512_permutex2var_pd(naA, IDX1, t));
    _mm512_storeu_pd(y + 16, _mm512_permutex2var_pd(nbB, IDX2, nbA));
    _mm512_mask_storeu_pd(y + 24, 0x03, nbA);                       /* X12 */
}

/* two transforms per body: table rows loaded once per pair, loads grouped
 * ahead of stores (d1_prime r3: the inline attribute alone was worth ~9%) */
#define STEP13P2(jj, UW, VW, UX, VX, tt) do {                                  \
    __m512d c1_ = _mm512_load_pd(tp + 32*(jj) +  0);                          \
    __m512d c2_ = _mm512_load_pd(tp + 32*(jj) +  8);                          \
    __m512d s1_ = _mm512_load_pd(tp + 32*(jj) + 16);                          \
    __m512d s2_ = _mm512_load_pd(tp + 32*(jj) + 24);                          \
    __m512d ub_ = _mm512_shuffle_f64x2(UW, UW, (tt)*0x55);                    \
    __m512d vb_ = _mm512_shuffle_f64x2(VW, VW, (tt)*0x55);                    \
    __m512d uc_ = _mm512_shuffle_f64x2(UX, UX, (tt)*0x55);                    \
    __m512d vc_ = _mm512_shuffle_f64x2(VX, VX, (tt)*0x55);                    \
    PA = _mm512_fmadd_pd(c1_, ub_, PA);  QA = _mm512_fmadd_pd(c1_, uc_, QA);  \
    PB = _mm512_fmadd_pd(c2_, ub_, PB);  QB = _mm512_fmadd_pd(c2_, uc_, QB);  \
    SA = _mm512_fmadd_pd(s1_, vb_, SA);  TA = _mm512_fmadd_pd(s1_, vc_, TA);  \
    SB = _mm512_fmadd_pd(s2_, vb_, SB);  TB = _mm512_fmadd_pd(s2_, vc_, TB);  \
} while (0)

static inline __attribute__((always_inline)) void
exec13p_b2(const double *restrict x, double *restrict y, const double *restrict tp)
{
    const double *restrict x2 = x + 26;
    double *restrict y2 = y + 26;
    __m512d F1 = _mm512_loadu_pd(x + 2),  G1 = _mm512_loadu_pd(x2 + 2);
    __m512d F2 = _mm512_loadu_pd(x + 10), G2 = _mm512_loadu_pd(x2 + 10);
    __m512d Z  = _mm512_loadu_pd(x + 18), W  = _mm512_loadu_pd(x2 + 18);
    __m512d x0p = _mm512_broadcast_f64x2(_mm_loadu_pd(x));
    __m512d x0q = _mm512_broadcast_f64x2(_mm_loadu_pd(x2));
    __m512d R1 = _mm512_shuffle_f64x2(Z,  Z,  0x1B), S1r = _mm512_shuffle_f64x2(W,  W,  0x1B);
    __m512d R2 = _mm512_shuffle_f64x2(F2, F2, 0xBB), S2r = _mm512_shuffle_f64x2(G2, G2, 0xBB);
    __m512d U1 = _mm512_add_pd(F1, R1),  V1 = _mm512_sub_pd(F1, R1);
    __m512d U2 = _mm512_add_pd(F2, R2),  V2 = _mm512_sub_pd(F2, R2);
    __m512d X1 = _mm512_add_pd(G1, S1r), W1 = _mm512_sub_pd(G1, S1r);
    __m512d X2 = _mm512_add_pd(G2, S2r), W2 = _mm512_sub_pd(G2, S2r);
    __m512d PA = x0p, PB = x0p, SA = _mm512_setzero_pd(), SB = SA;
    __m512d QA = x0q, QB = x0q, TA = SA, TB = SA;
    STEP13P2(0, U1, V1, X1, W1, 0);
    STEP13P2(1, U1, V1, X1, W1, 1);
    STEP13P2(2, U1, V1, X1, W1, 2);
    STEP13P2(3, U1, V1, X1, W1, 3);
    STEP13P2(4, U2, V2, X2, W2, 0);
    STEP13P2(5, U2, V2, X2, W2, 1);
    const __m512i IDX0 = _mm512_setr_epi64(12,13,0,1,2,3,4,5);
    const __m512i IDXT = _mm512_setr_epi64(0,1,2,3,10,11,10,11);
    const __m512i IDX1 = _mm512_setr_epi64(6,7,8,9,10,11,12,13);
    const __m512i IDX2 = _mm512_setr_epi64(0,1,14,15,12,13,10,11);
    __m512d swA = _mm512_permute_pd(SA, 0x55), swB = _mm512_permute_pd(SB, 0x55);
    __m512d naA = _mm512_sub_pd(PA, swA), nbA = _mm512_add_pd(PA, swA);
    __m512d naB = _mm512_sub_pd(PB, swB), nbB = _mm512_add_pd(PB, swB);
    __m512d swC = _mm512_permute_pd(TA, 0x55), swD = _mm512_permute_pd(TB, 0x55);
    __m512d naC = _mm512_sub_pd(QA, swC), nbC = _mm512_add_pd(QA, swC);
    __m512d naD = _mm512_sub_pd(QB, swD), nbD = _mm512_add_pd(QB, swD);
    _mm512_storeu_pd(y,       _mm512_permutex2var_pd(naA, IDX0, naB));
    __m512d t  = _mm512_permutex2var_pd(naB, IDXT, nbB);
    _mm512_storeu_pd(y + 8,   _mm512_permutex2var_pd(naA, IDX1, t));
    _mm512_storeu_pd(y + 16,  _mm512_permutex2var_pd(nbB, IDX2, nbA));
    _mm512_mask_storeu_pd(y + 24, 0x03, nbA);
    _mm512_storeu_pd(y2,      _mm512_permutex2var_pd(naC, IDX0, naD));
    __m512d t2 = _mm512_permutex2var_pd(naD, IDXT, nbD);
    _mm512_storeu_pd(y2 + 8,  _mm512_permutex2var_pd(naC, IDX1, t2));
    _mm512_storeu_pd(y2 + 16, _mm512_permutex2var_pd(nbD, IDX2, nbC));
    _mm512_mask_storeu_pd(y2 + 24, 0x03, nbC);
}

/* interleaved-pair L=31 kernel. Rows: A=(X1..X4) B=(X5..X8) C=(X9..X12)
 * D=(X13,X14,X15,X0); conjugate rows reverse-stored with one 0x1B each. */
#define STEP31P(jj, UW, VW, tt) do {                                          \
    __m512d ub_ = _mm512_shuffle_f64x2(UW, UW, (tt)*0x55);                    \
    __m512d vb_ = _mm512_shuffle_f64x2(VW, VW, (tt)*0x55);                    \
    PA = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) +  0), ub_, PA);         \
    PB = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) +  8), ub_, PB);         \
    PC = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 16), ub_, PC);         \
    PD = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 24), ub_, PD);         \
    SA = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 32), vb_, SA);         \
    SB = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 40), vb_, SB);         \
    SC = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 48), vb_, SC);         \
    SD = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 56), vb_, SD);         \
} while (0)

static inline __attribute__((always_inline)) void
exec31p_body(const double *restrict x, double *restrict y, const double *restrict tq)
{
    __m512d F1 = _mm512_loadu_pd(x + 2);    /* (x1)(x2)(x3)(x4)     */
    __m512d F2 = _mm512_loadu_pd(x + 10);   /* (x5)(x6)(x7)(x8)     */
    __m512d F3 = _mm512_loadu_pd(x + 18);   /* (x9)(x10)(x11)(x12)  */
    __m512d F4 = _mm512_loadu_pd(x + 26);   /* (x13)(x14)(x15)(x16) */
    __m512d Z1 = _mm512_loadu_pd(x + 54);   /* (x27)(x28)(x29)(x30) */
    __m512d Z2 = _mm512_loadu_pd(x + 46);   /* (x23)(x24)(x25)(x26) */
    __m512d Z3 = _mm512_loadu_pd(x + 38);   /* (x19)(x20)(x21)(x22) */
    __m512d Z4 = _mm512_loadu_pd(x + 30);   /* (x15)(x16)(x17)(x18) */
    __m512d R1 = _mm512_shuffle_f64x2(Z1, Z1, 0x1B);   /* (x30)(x29)(x28)(x27) */
    __m512d R2 = _mm512_shuffle_f64x2(Z2, Z2, 0x1B);   /* (x26)(x25)(x24)(x23) */
    __m512d R3 = _mm512_shuffle_f64x2(Z3, Z3, 0x1B);   /* (x22)(x21)(x20)(x19) */
    __m512d R4 = _mm512_shuffle_f64x2(Z4, Z4, 0x1B);   /* (x18)(x17)(x16)(x15) */
    __m512d U1 = _mm512_add_pd(F1, R1), V1 = _mm512_sub_pd(F1, R1);
    __m512d U2 = _mm512_add_pd(F2, R2), V2 = _mm512_sub_pd(F2, R2);
    __m512d U3 = _mm512_add_pd(F3, R3), V3 = _mm512_sub_pd(F3, R3);
    __m512d U4 = _mm512_add_pd(F4, R4), V4 = _mm512_sub_pd(F4, R4);  /* pair 3 junk */
    __m512d x0p = _mm512_broadcast_f64x2(_mm_loadu_pd(x));
    __m512d PA = x0p, PB = x0p, PC = x0p, PD = x0p;
    __m512d SA = _mm512_setzero_pd(), SB = SA, SC = SA, SD = SA;
    STEP31P( 0, U1, V1, 0);  STEP31P( 1, U1, V1, 1);
    STEP31P( 2, U1, V1, 2);  STEP31P( 3, U1, V1, 3);
    STEP31P( 4, U2, V2, 0);  STEP31P( 5, U2, V2, 1);
    STEP31P( 6, U2, V2, 2);  STEP31P( 7, U2, V2, 3);
    STEP31P( 8, U3, V3, 0);  STEP31P( 9, U3, V3, 1);
    STEP31P(10, U3, V3, 2);  STEP31P(11, U3, V3, 3);
    STEP31P(12, U4, V4, 0);  STEP31P(13, U4, V4, 1);
    STEP31P(14, U4, V4, 2);
    __m512d swA = _mm512_permute_pd(SA, 0x55), swB = _mm512_permute_pd(SB, 0x55);
    __m512d swC = _mm512_permute_pd(SC, 0x55), swD = _mm512_permute_pd(SD, 0x55);
    __m512d naA = _mm512_sub_pd(PA, swA), nbA = _mm512_add_pd(PA, swA);
    __m512d naB = _mm512_sub_pd(PB, swB), nbB = _mm512_add_pd(PB, swB);
    __m512d naC = _mm512_sub_pd(PC, swC), nbC = _mm512_add_pd(PC, swC);
    __m512d naD = _mm512_sub_pd(PD, swD), nbD = _mm512_add_pd(PD, swD);
    _mm_storeu_pd(y, _mm512_extractf64x2_pd(naD, 3));            /* X0 */
    _mm512_storeu_pd(y + 2,  naA);                               /* X1..X4   */
    _mm512_storeu_pd(y + 10, naB);                               /* X5..X8   */
    _mm512_storeu_pd(y + 18, naC);                               /* X9..X12  */
    _mm512_mask_storeu_pd(y + 26, 0x3F, naD);                    /* X13..X15 */
    _mm512_mask_storeu_pd(y + 30, 0xFC,
        _mm512_shuffle_f64x2(nbD, nbD, 0x1B));                   /* X16..X18 */
    _mm512_storeu_pd(y + 38, _mm512_shuffle_f64x2(nbC, nbC, 0x1B)); /* X19..X22 */
    _mm512_storeu_pd(y + 46, _mm512_shuffle_f64x2(nbB, nbB, 0x1B)); /* X23..X26 */
    _mm512_storeu_pd(y + 54, _mm512_shuffle_f64x2(nbA, nbA, 0x1B)); /* X27..X30 */
}
#endif /* __AVX512F__ */

/* fused chain, B=1: state lives ACROSS steps in the A/B row representation
 * (A rows: lane k-1 = state[k], lane h = state[0]; B rows: lane k-1 =
 * state[L-k]), so the next step's fold is u = A+B, v = A-B on whole registers
 * -- no scatter, no gather, no reversal permute in the steady state. */
static inline __attribute__((always_inline)) void
sp_chain1_body(const int L, const int h, const int HP8,
               const double *restrict tc, const double *restrict ts,
               const double *restrict x0, const double *restrict c,
               double *restrict out, int m)
{
    const int r_h = h >> 3, l_h = h & 7;
    v8 Ar[2], Ai[2], Br[2], Bi[2];
    v8 cAr[2], cAi[2], cBr[2], cBi[2];
    for (int r = 0; r < HP8; r++){
        Ar[r] = (v8){0}; Ai[r] = (v8){0}; Br[r] = (v8){0}; Bi[r] = (v8){0};
        cAr[r] = (v8){0}; cAi[r] = (v8){0}; cBr[r] = (v8){0}; cBi[r] = (v8){0};
    }
    for (int k = 1; k <= h; k++) {
        const int r = (k-1) >> 3, l = (k-1) & 7;
        Ar[r][l]  = x0[2*k];       Ai[r][l]  = x0[2*k+1];
        Br[r][l]  = x0[2*(L-k)];   Bi[r][l]  = x0[2*(L-k)+1];
        cAr[r][l] = c[2*k];        cAi[r][l] = c[2*k+1];
        cBr[r][l] = c[2*(L-k)];    cBi[r][l] = c[2*(L-k)+1];
    }
    Ar[r_h][l_h]  = x0[0];  Ai[r_h][l_h]  = x0[1];
    cAr[r_h][l_h] = c[0];   cAi[r_h][l_h] = c[1];

    for (int step = 0; step < m; step++) {
        const double x0r = Ar[r_h][l_h], x0i = Ai[r_h][l_h];
        double ur[SP_HMAX] __attribute__((aligned(64))), ui[SP_HMAX] __attribute__((aligned(64)));
        double vr[SP_HMAX] __attribute__((aligned(64))), vi[SP_HMAX] __attribute__((aligned(64)));
        for (int r = 0; r < HP8; r++) {
            *(v8 *)(ur + 8*r) = Ar[r] + Br[r];
            *(v8 *)(ui + 8*r) = Ai[r] + Bi[r];
            *(v8 *)(vr + 8*r) = Ar[r] - Br[r];
            *(v8 *)(vi + 8*r) = Ai[r] - Bi[r];
        }
        /* d1_prime's barrier: u/v broadcasts from memory ({1to8} FMA operands),
         * not register vpermpd (port 5); no ICX downside measured by them */
        __asm__("" : : : "memory");
        v8 Pr[2], Pi[2], Rr[2], Si[2];
        v8 Pr2[2], Pi2[2], Rr2[2], Si2[2];
        for (int r = 0; r < HP8; r++){
            Pr[r] = (v8){0}; Pi[r] = (v8){0}; Rr[r] = (v8){0}; Si[r] = (v8){0};
            Pr2[r] = (v8){0}; Pi2[r] = (v8){0}; Rr2[r] = (v8){0}; Si2[r] = (v8){0};
        }
        /* split accumulators only when one row set fits (HP8==1): at HP8==2
         * the extra 8 rows spill (their measured 0.050 -> 0.057 at L=31) */
        const int hh = (HP8 == 1) ? (h >> 1) : h;
        for (int j = 0; j < hh; j++) {
            const v8 *restrict cj = (const v8 *)(tc + (size_t)j*8*HP8);
            const v8 *restrict sj = (const v8 *)(ts + (size_t)j*8*HP8);
            const double a = ur[j], b = ui[j], e = vr[j], f = vi[j];
            for (int r = 0; r < HP8; r++) {
                Pr[r] += cj[r]*a;  Pi[r] += cj[r]*b;
                Rr[r] += sj[r]*e;  Si[r] += sj[r]*f;
            }
        }
        for (int j = hh; j < h; j++) {
            const v8 *restrict cj = (const v8 *)(tc + (size_t)j*8*HP8);
            const v8 *restrict sj = (const v8 *)(ts + (size_t)j*8*HP8);
            const double a = ur[j], b = ui[j], e = vr[j], f = vi[j];
            for (int r = 0; r < HP8; r++) {
                Pr2[r] += cj[r]*a;  Pi2[r] += cj[r]*b;
                Rr2[r] += sj[r]*e;  Si2[r] += sj[r]*f;
            }
        }
        for (int r = 0; r < HP8; r++) {
            Pr[r] += Pr2[r]; Pi[r] += Pi2[r]; Rr[r] += Rr2[r]; Si[r] += Si2[r];
        }
        for (int r = 0; r < HP8; r++) {
            v8 pre = x0r + Pr[r], pim = x0i + Pi[r];
            v8 nar = pre + Si[r] + cAr[r];
            v8 nai = pim - Rr[r] + cAi[r];
            v8 nbr = pre - Si[r] + cBr[r];
            v8 nbi = pim + Rr[r] + cBi[r];
            v8 qa = map_sc8(nar, nai);
            Ar[r] = nar * qa;  Ai[r] = nai * qa;
            v8 qb = map_sc8(nbr, nbi);
            Br[r] = nbr * qb;  Bi[r] = nbi * qb;
        }
    }
    out[0] = Ar[r_h][l_h]; out[1] = Ai[r_h][l_h];
    for (int k = 1; k <= h; k++) {
        const int r = (k-1) >> 3, l = (k-1) & 7;
        out[2*k]       = Ar[r][l];  out[2*k+1]       = Ai[r][l];
        out[2*(L-k)]   = Br[r][l];  out[2*(L-k)+1]   = Bi[r][l];
    }
}

/* fused chain, batched: 8 chains per lane-block, transposed to SoA once,
 * L1-resident for all m steps; tail lanes clamped (computed, never stored) */
static inline __attribute__((always_inline)) void
sp_chainblk_body(const int L, const int h,
                 const double *restrict ck, const double *restrict sk,
                 const double *restrict x0, const double *restrict c,
                 double *restrict out, int m, int batch)
{
    for (int b0 = 0; b0 < batch; b0 += 8) {
        const int lanes = (batch - b0 < 8) ? batch - b0 : 8;
        v8 xr[SP_LMAX], xi[SP_LMAX], cr[SP_LMAX], ci[SP_LMAX];
        for (int l = 0; l < 8; l++) {
            const size_t bb = (size_t)(b0 + (l < lanes ? l : lanes - 1)) * L;
            for (int j = 0; j < L; j++) {
                xr[j][l] = x0[2*(bb+j)];  xi[j][l] = x0[2*(bb+j)+1];
                cr[j][l] = c [2*(bb+j)];  ci[j][l] = c [2*(bb+j)+1];
            }
        }
        for (int step = 0; step < m; step++) {
            v8 ur[SP_HMAX], ui[SP_HMAX], vr[SP_HMAX], vi[SP_HMAX];
            const v8 x0r = xr[0], x0i = xi[0];
            v8 s0r = x0r, s0i = x0i;
            for (int j = 1; j <= h; j++) {
                v8 ar = xr[j],   ai = xi[j];
                v8 br = xr[L-j], bi = xi[L-j];
                ur[j-1] = ar + br;  ui[j-1] = ai + bi;
                vr[j-1] = ar - br;  vi[j-1] = ai - bi;
                s0r += ur[j-1];     s0i += ui[j-1];
            }
            s0r += cr[0]; s0i += ci[0];
            v8 q0 = map_sc8(s0r, s0i);
            xr[0] = s0r * q0; xi[0] = s0i * q0;
            /* k blocked by 3: each u/v row load feeds 12 FMAs (h = 6 and 15
             * both divide by 3, so the remainder loop is dead at 13/31) */
            int k = 1;
            for (; k + 2 <= h; k += 3) {
                v8 Pr0=(v8){0}, Pi0=(v8){0}, Rr0=(v8){0}, Si0=(v8){0};
                v8 Pr1=(v8){0}, Pi1=(v8){0}, Rr1=(v8){0}, Si1=(v8){0};
                v8 Pr2=(v8){0}, Pi2=(v8){0}, Rr2=(v8){0}, Si2=(v8){0};
                const double *restrict ck0 = ck + (size_t)(k-1)*h, *restrict sk0 = sk + (size_t)(k-1)*h;
                const double *restrict ck1 = ck0 + h, *restrict sk1 = sk0 + h;
                const double *restrict ck2 = ck1 + h, *restrict sk2 = sk1 + h;
                for (int j = 0; j < h; j++) {
                    const v8 uj = ur[j], wj = ui[j], ej = vr[j], fj = vi[j];
                    Pr0 += ck0[j]*uj;  Pi0 += ck0[j]*wj;  Rr0 += sk0[j]*ej;  Si0 += sk0[j]*fj;
                    Pr1 += ck1[j]*uj;  Pi1 += ck1[j]*wj;  Rr1 += sk1[j]*ej;  Si1 += sk1[j]*fj;
                    Pr2 += ck2[j]*uj;  Pi2 += ck2[j]*wj;  Rr2 += sk2[j]*ej;  Si2 += sk2[j]*fj;
                }
                for (int t = 0; t < 3; t++) {
                    const int kk = k + t;
                    v8 Pr = t==0?Pr0:(t==1?Pr1:Pr2), Pi = t==0?Pi0:(t==1?Pi1:Pi2);
                    v8 Rr = t==0?Rr0:(t==1?Rr1:Rr2), Si = t==0?Si0:(t==1?Si1:Si2);
                    v8 pre = x0r + Pr, pim = x0i + Pi;
                    v8 Arow = pre + Si + cr[kk],   Airow = pim - Rr + ci[kk];
                    v8 Brow = pre - Si + cr[L-kk], Birow = pim + Rr + ci[L-kk];
                    v8 qa = map_sc8(Arow, Airow);
                    xr[kk]   = Arow * qa;  xi[kk]   = Airow * qa;
                    v8 qb = map_sc8(Brow, Birow);
                    xr[L-kk] = Brow * qb;  xi[L-kk] = Birow * qb;
                }
            }
            for (; k <= h; k++) {
                v8 Pr = (v8){0}, Pi = (v8){0}, Rr = (v8){0}, Si = (v8){0};
                const double *restrict ckr = ck + (size_t)(k-1)*h;
                const double *restrict skr = sk + (size_t)(k-1)*h;
                for (int j = 0; j < h; j++) {
                    const double cc = ckr[j], ss = skr[j];
                    Pr += cc*ur[j];  Pi += cc*ui[j];
                    Rr += ss*vr[j];  Si += ss*vi[j];
                }
                v8 pre = x0r + Pr, pim = x0i + Pi;
                v8 Arow = pre + Si + cr[k],   Airow = pim - Rr + ci[k];
                v8 Brow = pre - Si + cr[L-k], Birow = pim + Rr + ci[L-k];
                v8 qa = map_sc8(Arow, Airow);
                xr[k]   = Arow * qa;  xi[k]   = Airow * qa;
                v8 qb = map_sc8(Brow, Birow);
                xr[L-k] = Brow * qb;  xi[L-k] = Birow * qb;
            }
        }
        for (int l = 0; l < lanes; l++) {
            const size_t bb = (size_t)(b0 + l) * L;
            for (int j = 0; j < L; j++){ out[2*(bb+j)] = xr[j][l]; out[2*(bb+j)+1] = xi[j][l]; }
        }
    }
}

/* ============== end of the d1_prime lift (kernels; plumbing below) ============ */

/* ============ L=32/64 in-register codelets -- ADOPTED FROM d1_pow2 ============
 * (r1 fft32/fft64 execute + chain codelets, taken nearly verbatim: whole
 * transform in 8/16 zmm, stride-1 stage's store transpose done in-register,
 * full dup-format w/w^2/w^3 first-stage tables -- their r1 finding that the
 * on-the-fly w^2/w^3 variant is slower AND a rounding-bias source. Chain map
 * is their r3 fast map (2NR rsqrt/rcp, no residual refinements): their gate
 * margins at 32/64 are >=2 decades and my tryout gates below reconfirm.
 * Aligned load/store relaxed to loadu/storeu: costs nothing when aligned. */
#ifdef __AVX512F__
#define PW_PSWAP 0x55 /* vpermilpd: swap re/im inside each 128-bit pair */

static inline __attribute__((always_inline)) __m512d
pw_cmul_bc(__m512d u, __m512d wr, __m512d wp)
{
    return _mm512_fmadd_pd(_mm512_permute_pd(u, PW_PSWAP), wp, _mm512_mul_pd(u, wr));
}

/* AoS map z -> z/(1+|z|), z = v + cv (d1_pow2's map_vec; r7: Goldschmidt +
 * early-seeded rcp via map_q8, |z|^2 pair-duplicated by the PSWAP permute) */
static inline __attribute__((always_inline)) __m512d
pw_map_vec(__m512d v, __m512d cv)
{
    __m512d z = _mm512_add_pd(v, cv);
    __m512d zz = _mm512_fmadd_pd(z, z, _mm512_set1_pd(1e-100));
    __m512d s = _mm512_add_pd(zz, _mm512_permute_pd(zz, PW_PSWAP));
    return _mm512_mul_pd(z, map_q8(s));
}

#define PW_R8_BODY(X0, X1, X2, X3, X4, X5, X6, X7)                                     \
    __m512d s0 = _mm512_add_pd(X0, X4), s1 = _mm512_add_pd(X1, X5);                    \
    __m512d s2 = _mm512_add_pd(X2, X6), s3 = _mm512_add_pd(X3, X7);                    \
    __m512d d0 = _mm512_sub_pd(X0, X4), d1 = _mm512_sub_pd(X1, X5);                    \
    __m512d d2 = _mm512_sub_pd(X2, X6), d3 = _mm512_sub_pd(X3, X7);                    \
    __m512d apc = _mm512_add_pd(s0, s2), amc = _mm512_sub_pd(s0, s2);                  \
    __m512d bpd = _mm512_add_pd(s1, s3), bmd = _mm512_sub_pd(s1, s3);                  \
    __m512d swe = _mm512_permute_pd(bmd, PW_PSWAP);                                    \
    __m512d u0 = _mm512_add_pd(apc, bpd);                                              \
    __m512d u4 = _mm512_sub_pd(apc, bpd);                                              \
    __m512d u2 = _mm512_fmsubadd_pd(amc, ONE, swe);                                    \
    __m512d u6 = _mm512_fmaddsub_pd(amc, ONE, swe);                                    \
    __m512d e1 = _mm512_mul_pd(_mm512_fmsubadd_pd(d1, ONE, _mm512_permute_pd(d1, PW_PSWAP)), Cq); \
    __m512d e3 = _mm512_mul_pd(                                                        \
        _mm512_permute_pd(_mm512_fmsubadd_pd(d3, ONE, _mm512_permute_pd(d3, PW_PSWAP)), PW_PSWAP), CPN); \
    __m512d sw2 = _mm512_permute_pd(d2, PW_PSWAP);                                     \
    __m512d apo = _mm512_fmsubadd_pd(d0, ONE, sw2);                                    \
    __m512d amo = _mm512_fmaddsub_pd(d0, ONE, sw2);                                    \
    __m512d bpo = _mm512_add_pd(e1, e3), bmo = _mm512_sub_pd(e1, e3);                  \
    __m512d swo = _mm512_permute_pd(bmo, PW_PSWAP);                                    \
    __m512d u1 = _mm512_add_pd(apo, bpo);                                              \
    __m512d u5 = _mm512_sub_pd(apo, bpo);                                              \
    __m512d u3 = _mm512_fmsubadd_pd(amo, ONE, swo);                                    \
    __m512d u7 = _mm512_fmaddsub_pd(amo, ONE, swo)

#define PW_R8_CONSTS                                                                   \
    const __m512d ONE = _mm512_set1_pd(1.0);                                           \
    const __m512d Cq = _mm512_set1_pd(0.70710678118654752440);                         \
    const __m512d CPN = _mm512_setr_pd(0.70710678118654752440, -0.70710678118654752440,\
                                       0.70710678118654752440, -0.70710678118654752440,\
                                       0.70710678118654752440, -0.70710678118654752440,\
                                       0.70710678118654752440, -0.70710678118654752440)

#define PW_TRANSP4(R0, R1, R2, R3, O0, O1, O2, O3)                                     \
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

#define PW_S1QUADT(A, B, C, D, TP, O0, O1, O2, O3)                                     \
    do {                                                                               \
        __m512d apc_ = _mm512_add_pd(A, C), amc_ = _mm512_sub_pd(A, C);                \
        __m512d bpd_ = _mm512_add_pd(B, D), bmd_ = _mm512_sub_pd(B, D);                \
        __m512d sw_ = _mm512_permute_pd(bmd_, PW_PSWAP);                               \
        __m512d q0_ = _mm512_add_pd(apc_, bpd_);                                       \
        __m512d q2_ = _mm512_sub_pd(apc_, bpd_);                                       \
        __m512d q1_ = _mm512_fmsubadd_pd(amc_, ONE, sw_);                              \
        __m512d q3_ = _mm512_fmaddsub_pd(amc_, ONE, sw_);                              \
        __m512d r1_ = pw_cmul_bc(q1_, _mm512_loadu_pd(TP), _mm512_loadu_pd((TP) + 8)); \
        __m512d r2_ = pw_cmul_bc(q2_, _mm512_loadu_pd((TP) + 16), _mm512_loadu_pd((TP) + 24)); \
        __m512d r3_ = pw_cmul_bc(q3_, _mm512_loadu_pd((TP) + 32), _mm512_loadu_pd((TP) + 40)); \
        PW_TRANSP4(q0_, r1_, r2_, r3_, O0, O1, O2, O3);                                \
    } while (0)

#define PW_R4Q(A, B, C, D, O0, O1, O2, O3)                                             \
    do {                                                                               \
        __m512d apc_ = _mm512_add_pd(A, C), amc_ = _mm512_sub_pd(A, C);                \
        __m512d bpd_ = _mm512_add_pd(B, D), bmd_ = _mm512_sub_pd(B, D);                \
        __m512d sw_ = _mm512_permute_pd(bmd_, PW_PSWAP);                               \
        O0 = _mm512_add_pd(apc_, bpd_);                                                \
        O2 = _mm512_sub_pd(apc_, bpd_);                                                \
        O1 = _mm512_fmsubadd_pd(amc_, ONE, sw_);                                       \
        O3 = _mm512_fmaddsub_pd(amc_, ONE, sw_);                                       \
    } while (0)

#define PW_FFT32_REGS(V0, V1, V2, V3, V4, V5, V6, V7, U)                               \
    do {                                                                               \
        __m512d ya0_, ya1_, ya2_, ya3_, yb0_, yb1_, yb2_, yb3_;                        \
        PW_S1QUADT(V0, V2, V4, V6, tf, ya0_, ya1_, ya2_, ya3_);                        \
        PW_S1QUADT(V1, V3, V5, V7, tf + 48, yb0_, yb1_, yb2_, yb3_);                   \
        PW_R8_BODY(ya0_, ya1_, ya2_, ya3_, yb0_, yb1_, yb2_, yb3_);                    \
        U##0 = u0; U##1 = u1; U##2 = u2; U##3 = u3;                                    \
        U##4 = u4; U##5 = u5; U##6 = u6; U##7 = u7;                                    \
    } while (0)

static void fft32_execute(const double *restrict tf, const double *restrict in,
                          double *restrict out, int batch)
{
    PW_R8_CONSTS;
    const __m512i idxA = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    const __m512i idxB = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
    for (int b = 0; b < batch; ++b) {
        const double *x = in + 64L * b;
        double *y = out + 64L * b;
        __m512d v0 = _mm512_loadu_pd(x), v1 = _mm512_loadu_pd(x + 8);
        __m512d v2 = _mm512_loadu_pd(x + 16), v3 = _mm512_loadu_pd(x + 24);
        __m512d v4 = _mm512_loadu_pd(x + 32), v5 = _mm512_loadu_pd(x + 40);
        __m512d v6 = _mm512_loadu_pd(x + 48), v7 = _mm512_loadu_pd(x + 56);
        __m512d o0, o1, o2, o3, o4, o5, o6, o7;
        PW_FFT32_REGS(v0, v1, v2, v3, v4, v5, v6, v7, o);
        _mm512_storeu_pd(y, o0); _mm512_storeu_pd(y + 8, o1);
        _mm512_storeu_pd(y + 16, o2); _mm512_storeu_pd(y + 24, o3);
        _mm512_storeu_pd(y + 32, o4); _mm512_storeu_pd(y + 40, o5);
        _mm512_storeu_pd(y + 48, o6); _mm512_storeu_pd(y + 56, o7);
    }
}

static void fft32_chain(const double *restrict tf, const double *restrict x0,
                        const double *restrict c, double *restrict final_out,
                        int batch, int m)
{
    PW_R8_CONSTS;
    const __m512i idxA = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    const __m512i idxB = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
    for (int b = 0; b < batch; ++b) {
        const double *x = x0 + 64L * b;
        const double *cb = c + 64L * b;
        double *y = final_out + 64L * b;
        __m512d c0 = _mm512_loadu_pd(cb), c1 = _mm512_loadu_pd(cb + 8);
        __m512d c2 = _mm512_loadu_pd(cb + 16), c3 = _mm512_loadu_pd(cb + 24);
        __m512d c4 = _mm512_loadu_pd(cb + 32), c5 = _mm512_loadu_pd(cb + 40);
        __m512d c6 = _mm512_loadu_pd(cb + 48), c7 = _mm512_loadu_pd(cb + 56);
        __m512d v0 = _mm512_loadu_pd(x), v1 = _mm512_loadu_pd(x + 8);
        __m512d v2 = _mm512_loadu_pd(x + 16), v3 = _mm512_loadu_pd(x + 24);
        __m512d v4 = _mm512_loadu_pd(x + 32), v5 = _mm512_loadu_pd(x + 40);
        __m512d v6 = _mm512_loadu_pd(x + 48), v7 = _mm512_loadu_pd(x + 56);
        for (int step = 0; step < m; ++step) {
            __m512d o0, o1, o2, o3, o4, o5, o6, o7;
            PW_FFT32_REGS(v0, v1, v2, v3, v4, v5, v6, v7, o);
            v0 = pw_map_vec(o0, c0); v1 = pw_map_vec(o1, c1);
            v2 = pw_map_vec(o2, c2); v3 = pw_map_vec(o3, c3);
            v4 = pw_map_vec(o4, c4); v5 = pw_map_vec(o5, c5);
            v6 = pw_map_vec(o6, c6); v7 = pw_map_vec(o7, c7);
        }
        _mm512_storeu_pd(y, v0); _mm512_storeu_pd(y + 8, v1);
        _mm512_storeu_pd(y + 16, v2); _mm512_storeu_pd(y + 24, v3);
        _mm512_storeu_pd(y + 32, v4); _mm512_storeu_pd(y + 40, v5);
        _mm512_storeu_pd(y + 48, v6); _mm512_storeu_pd(y + 56, v7);
    }
}

static void fft64_execute(const double *restrict tf, const double *restrict t2,
                          const double *restrict in, double *restrict out, int batch)
{
    const __m512d ONE = _mm512_set1_pd(1.0);
    const __m512i idxA = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    const __m512i idxB = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
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
        for (int j = 0; j < 4; ++j) {
            __m512d a = _mm512_loadu_pd(x + 8L * j);
            __m512d bq = _mm512_loadu_pd(x + 8L * (j + 4));
            __m512d cq = _mm512_loadu_pd(x + 8L * (j + 8));
            __m512d dq = _mm512_loadu_pd(x + 8L * (j + 12));
            PW_S1QUADT(a, bq, cq, dq, tf + 48 * j, Y[4 * j], Y[4 * j + 1], Y[4 * j + 2],
                       Y[4 * j + 3]);
        }
        PW_R4Q(Y[0], Y[4], Y[8], Y[12], Z[0], Z[1], Z[2], Z[3]);
        for (int pp = 1; pp < 4; ++pp) {
            __m512d z0, z1, z2, z3;
            PW_R4Q(Y[pp], Y[pp + 4], Y[pp + 8], Y[pp + 12], z0, z1, z2, z3);
            Z[4 * pp] = z0;
            Z[4 * pp + 1] = pw_cmul_bc(z1, g2r[pp][0], g2p[pp][0]);
            Z[4 * pp + 2] = pw_cmul_bc(z2, g2r[pp][1], g2p[pp][1]);
            Z[4 * pp + 3] = pw_cmul_bc(z3, g2r[pp][2], g2p[pp][2]);
        }
        for (int j = 0; j < 4; ++j) {
            __m512d z0, z1, z2, z3;
            PW_R4Q(Z[j], Z[j + 4], Z[j + 8], Z[j + 12], z0, z1, z2, z3);
            _mm512_storeu_pd(yo + 8L * j, z0);
            _mm512_storeu_pd(yo + 8L * (j + 4), z1);
            _mm512_storeu_pd(yo + 8L * (j + 8), z2);
            _mm512_storeu_pd(yo + 8L * (j + 12), z3);
        }
    }
}

static void fft64_chain(const double *restrict tf, const double *restrict t2,
                        const double *restrict x0, const double *restrict c,
                        double *restrict final_out, int batch, int m)
{
    const __m512d ONE = _mm512_set1_pd(1.0);
    const __m512i idxA = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    const __m512i idxB = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
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
        for (int j = 0; j < 16; ++j) V[j] = _mm512_loadu_pd(x + 8L * j);
        for (int step = 0; step < m; ++step) {
            __m512d Y[16], Z[16];
            for (int j = 0; j < 4; ++j)
                PW_S1QUADT(V[j], V[j + 4], V[j + 8], V[j + 12], tf + 48 * j, Y[4 * j],
                           Y[4 * j + 1], Y[4 * j + 2], Y[4 * j + 3]);
            PW_R4Q(Y[0], Y[4], Y[8], Y[12], Z[0], Z[1], Z[2], Z[3]);
            for (int pp = 1; pp < 4; ++pp) {
                __m512d z0, z1, z2, z3;
                PW_R4Q(Y[pp], Y[pp + 4], Y[pp + 8], Y[pp + 12], z0, z1, z2, z3);
                Z[4 * pp] = z0;
                Z[4 * pp + 1] = pw_cmul_bc(z1, g2r[pp][0], g2p[pp][0]);
                Z[4 * pp + 2] = pw_cmul_bc(z2, g2r[pp][1], g2p[pp][1]);
                Z[4 * pp + 3] = pw_cmul_bc(z3, g2r[pp][2], g2p[pp][2]);
            }
            for (int j = 0; j < 4; ++j) {
                __m512d z0, z1, z2, z3;
                PW_R4Q(Z[j], Z[j + 4], Z[j + 8], Z[j + 12], z0, z1, z2, z3);
                V[j]      = pw_map_vec(z0, _mm512_loadu_pd(cb + 8L * j));
                V[j + 4]  = pw_map_vec(z1, _mm512_loadu_pd(cb + 8L * (j + 4)));
                V[j + 8]  = pw_map_vec(z2, _mm512_loadu_pd(cb + 8L * (j + 8)));
                V[j + 12] = pw_map_vec(z3, _mm512_loadu_pd(cb + 8L * (j + 12)));
            }
        }
        for (int j = 0; j < 16; ++j) _mm512_storeu_pd(yo + 8L * j, V[j]);
    }
}
#endif /* __AVX512F__ */

/* =============== end of the d1_pow2 lift (kernels; plumbing below) ============ */

/* dense symmetric-fold DFT for a single generic stage at B=1 (interleaved in/out).
 * acc: 4*h doubles of scratch. */
/* ------------- L=60 Good-Thomas PFA kernel (ADOPTED FROM d1_composite) ------
 * 60 = 4x3x5 coprime -> CRT re-index kills ALL twiddles: 15 DFT-4s (add-only),
 * 20 DFT-3s, 12 DFT-5s with real constants. Lifted near-verbatim from their
 * fft60_ymm1 (r4/r5 form): stage A operands arrive (xj|xj) via
 * vbroadcastf64x2 (one load uop, no insert blends) and the DFT-4 lane split
 * is two signed FMAs (exact, E=+-1) plus one in-lane vpermilpd -- zero
 * cross-lane shuffles; stages B/C run n1-paired ymm with sign-folded sine
 * constants. Their node A/B: this kernel in a plain per-transform loop wins
 * at EVERY batch on ICX (256-bit mix fills p1; 0.045 us at B=512 vs 0.052
 * for the zmm pair kernel). My r4 Stockham [4,3,5] ran 0.135/0.131 (B=1/512)
 * on the same node. Table names kept (PIN/KOUT). */
static const int PIN60[60] = {0,12,24,36,48,20,32,44,56,8,40,52,4,16,28,15,27,39,51,3,35,47,59,11,23,55,7,19,31,43,30,42,54,6,18,50,2,14,26,38,10,22,34,46,58,45,57,9,21,33,5,17,29,41,53,25,37,49,1,13};
static const int KOUT60[60] = {0,36,12,48,24,40,16,52,28,4,20,56,32,8,44,45,21,57,33,9,25,1,37,13,49,5,41,17,53,29,30,6,42,18,54,10,46,22,58,34,50,26,2,38,14,15,51,27,3,39,55,31,7,43,19,35,11,47,23,59};

__attribute__((target("avx512f,avx512dq,avx512vl")))
static void fft60_pfa(const double *restrict x, double *restrict y)
{
    const double C51 = 0.30901699437494745126;   /* cos(2pi/5) */
    const double C52 = -0.80901699437494734024;  /* cos(4pi/5) */
    const double S51 = 0.95105651629515353118;   /* sin(2pi/5) */
    const double S52 = 0.58778525229247324813;   /* sin(4pi/5) */
    const double S3_ = 0.86602540378443870761;   /* sin(2pi/3) */
    __m256d wp[2][15];   /* [n1-pair][n2*5+n3] */
    {
        const __m256d E1 = _mm256_set_pd(-1.0, -1.0, 1.0, 1.0);
        const __m256d E4 = _mm256_set_pd(-1.0, 1.0, 1.0, 1.0);
        _Pragma("GCC unroll 15") for (int col = 0; col < 15; ++col) {
            __m256d X0 = _mm256_broadcast_f64x2(_mm_loadu_pd(x + 2*PIN60[col]));
            __m256d X1 = _mm256_broadcast_f64x2(_mm_loadu_pd(x + 2*PIN60[15+col]));
            __m256d X2 = _mm256_broadcast_f64x2(_mm_loadu_pd(x + 2*PIN60[30+col]));
            __m256d X3 = _mm256_broadcast_f64x2(_mm_loadu_pd(x + 2*PIN60[45+col]));
            __m256d P = _mm256_fmadd_pd(X2, E1, X0);
            __m256d Q = _mm256_fmadd_pd(X3, E1, X1);
            __m256d R = _mm256_permute_pd(Q, 0x6);
            wp[0][col] = _mm256_fmadd_pd(R, E4, P);
            wp[1][col] = _mm256_fnmadd_pd(R, E4, P);
        }
    }
    {
        const __m256d half = _mm256_set1_pd(0.5);
        const __m256d S3E = _mm256_set_pd(-S3_, S3_, -S3_, S3_);
        _Pragma("GCC unroll 2") for (int pr = 0; pr < 2; ++pr)
            _Pragma("GCC unroll 5") for (int n3 = 0; n3 < 5; ++n3) {
                __m256d x1 = wp[pr][5+n3], x2 = wp[pr][10+n3];
                __m256d t = _mm256_add_pd(x1, x2), u = _mm256_sub_pd(x1, x2);
                __m256d x0 = wp[pr][n3];
                wp[pr][n3] = _mm256_add_pd(x0, t);
                __m256d vv = _mm256_fnmadd_pd(half, t, x0);
                __m256d swu = _mm256_permute_pd(u, 0x5);
                wp[pr][5+n3]  = _mm256_fmadd_pd(swu, S3E, vv);
                wp[pr][10+n3] = _mm256_fnmadd_pd(swu, S3E, vv);
            }
        const __m256d c51v = _mm256_set1_pd(C51), c52v = _mm256_set1_pd(C52);
        const __m256d S1E = _mm256_set_pd(-S51, S51, -S51, S51);
        const __m256d S2E = _mm256_set_pd(-S52, S52, -S52, S52);
        _Pragma("GCC unroll 2") for (int pr = 0; pr < 2; ++pr)
            _Pragma("GCC unroll 3") for (int n2 = 0; n2 < 3; ++n2) {
                const int b = 5*n2;                        /* base in 15-space */
                const int bl = (2*pr)*15 + 5*n2, bh = (2*pr+1)*15 + 5*n2; /* 60-space */
                __m256d x0 = wp[pr][b];
                __m256d t1 = _mm256_add_pd(wp[pr][b+1], wp[pr][b+4]);
                __m256d t3 = _mm256_sub_pd(wp[pr][b+1], wp[pr][b+4]);
                __m256d t2 = _mm256_add_pd(wp[pr][b+2], wp[pr][b+3]);
                __m256d t4 = _mm256_sub_pd(wp[pr][b+2], wp[pr][b+3]);
#define STP(k3, v) do { __m256d v_ = (v);                                        \
        _mm_storeu_pd(y + 2*KOUT60[bl+(k3)], _mm256_castpd256_pd128(v_));        \
        _mm_storeu_pd(y + 2*KOUT60[bh+(k3)], _mm256_extractf128_pd(v_, 1)); } while (0)
                STP(0, _mm256_add_pd(x0, _mm256_add_pd(t1, t2)));
                __m256d a1 = _mm256_fmadd_pd(c52v, t2, _mm256_fmadd_pd(c51v, t1, x0));
                __m256d a2 = _mm256_fmadd_pd(c51v, t2, _mm256_fmadd_pd(c52v, t1, x0));
                __m256d sw3 = _mm256_permute_pd(t3, 0x5), sw4 = _mm256_permute_pd(t4, 0x5);
                __m256d m1 = _mm256_fmadd_pd(sw4, S2E, _mm256_mul_pd(sw3, S1E));
                __m256d m2 = _mm256_fnmadd_pd(sw4, S1E, _mm256_mul_pd(sw3, S2E));
                STP(1, _mm256_add_pd(a1, m1));
                STP(4, _mm256_sub_pd(a1, m1));
                STP(2, _mm256_add_pd(a2, m2));
                STP(3, _mm256_sub_pd(a2, m2));
#undef STP
            }
    }
}

/* L=60 fused chain on the PFA kernel: state stays INTERLEAVED (the codelet's
 * native format), map runs 8-wide on interleaved data with |z|^2 duplicated
 * per 128-bit lane (d1_composite's map_scale_q idiom; r7: Goldschmidt +
 * early-seeded rcp via map_q8) -- 120 doubles = exactly 15 zmm per step. */
__attribute__((target("avx512f,avx512dq,avx512vl")))
static void fft60_chain(const double *restrict x0, const double *restrict cd,
                        double *restrict od, int B, int m)
{
    double st[120] __attribute__((aligned(64)));
    double z[120] __attribute__((aligned(64)));
    for (int b = 0; b < B; ++b) {
        const double *restrict xb = x0 + (size_t)b * 120;
        const double *restrict cb = cd + (size_t)b * 120;
        double *restrict ob = od + (size_t)b * 120;
        memcpy(st, xb, sizeof st);
        for (int s = 0; s < m; ++s) {
            fft60_pfa(st, z);
            double *restrict dst = (s == m - 1) ? ob : st;
            for (int k = 0; k < 120; k += 8) {
                __m512d zv = _mm512_add_pd(_mm512_load_pd(z + k),
                                           _mm512_loadu_pd(cb + k));
                __m512d z2 = _mm512_fmadd_pd(zv, zv, _mm512_set1_pd(1e-100));
                __m512d q = _mm512_add_pd(z2, _mm512_permute_pd(z2, 0x55));
                _mm512_storeu_pd(dst + k, _mm512_mul_pd(zv, map_q8(q)));
            }
        }
    }
}

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

/* --------- fused middle: forward-last x kernel-multiply x inverse-entry ----
 * ADOPTED FROM d1_bluestein (r4 stmid2/stmid4), which executed d1_rader's r3
 * "fuse the kernel multiply into the LAST forward stage" idea: the last
 * forward Stockham stage has m = 1 (twiddles all 1) and its output blocks
 * k = q + j*s line up exactly with the radix-4 inverse entry's reads
 * p + u*M/4, so forward-last butterfly, kernel multiply with the re/im swap
 * (inverse-as-forward), and the inverse stage-0 radix-4 butterfly run in ONE
 * pass -- the spectrum is never written or read back (one full plane-pair
 * round trip gone per transform / per chain step). Two forward-last radix
 * cases: rl=4 (s = M/4, one radix-4 butterfly) and rl=2 (s = M/2, two
 * radix-2 butterflies at q = p and q = p + M/4). Serves both the Bluestein
 * middle (10007 M=20480 rl=2, 100003 M=204800 rl=4) and the Rader conv
 * (65537 P=65536 rl=4; the spectrum DC bin, needed for X[0], is re-summed
 * from the same four inputs at p=0 by the caller -- d1_rader's "X[0] free"). */
#define STMID_BODY(BUTTERFLY) \
{ \
    const double *restrict w1r = wr,           *restrict w1i = wi; \
    const double *restrict w2r = wr + m0,      *restrict w2i = wi + m0; \
    const double *restrict w3r = wr + 2 * m0,  *restrict w3i = wi + 2 * m0; \
_Pragma("GCC ivdep") \
    for (int p = 0; p < m0; ++p) { \
        int t1_ = p + m0, t2_ = p + 2 * m0, t3_ = p + 3 * m0; \
        double ar = xr[p],   ai_ = xi[p]; \
        double br = xr[t1_], bi = xi[t1_]; \
        double cr = xr[t2_], ci = xi[t2_]; \
        double dr = xr[t3_], di = xi[t3_]; \
        double B0r, B0i, B1r, B1i, B2r, B2i, B3r, B3i; \
        BUTTERFLY \
        /* kernel multiply, swapped planes: A = swap(B * bhat) */ \
        double A0r = brh[p] * B0i + bih[p] * B0r,    A0i = brh[p] * B0r - bih[p] * B0i; \
        double A1r = brh[t1_] * B1i + bih[t1_] * B1r, A1i = brh[t1_] * B1r - bih[t1_] * B1i; \
        double A2r = brh[t2_] * B2i + bih[t2_] * B2r, A2i = brh[t2_] * B2r - bih[t2_] * B2i; \
        double A3r = brh[t3_] * B3i + bih[t3_] * B3r, A3i = brh[t3_] * B3r - bih[t3_] * B3i; \
        /* inverse entry radix-4 with stage-0 twiddles */ \
        double u0r = A0r + A2r, u0i = A0i + A2i, u1r = A0r - A2r, u1i = A0i - A2i; \
        double u2r = A1r + A3r, u2i = A1i + A3i, u3r = A1r - A3r, u3i = A1i - A3i; \
        double y1r = u1r + u3i, y1i = u1i - u3r; \
        double y2r = u0r - u2r, y2i = u0i - u2i; \
        double y3r = u1r - u3i, y3i = u1i + u3r; \
        yr[4 * p] = u0r + u2r;  yi[4 * p] = u0i + u2i; \
        yr[4 * p + 1] = y1r * w1r[p] - y1i * w1i[p]; \
        yi[4 * p + 1] = y1r * w1i[p] + y1i * w1r[p]; \
        yr[4 * p + 2] = y2r * w2r[p] - y2i * w2i[p]; \
        yi[4 * p + 2] = y2r * w2i[p] + y2i * w2r[p]; \
        yr[4 * p + 3] = y3r * w3r[p] - y3i * w3i[p]; \
        yi[4 * p + 3] = y3r * w3i[p] + y3i * w3r[p]; \
    } \
}

/* rl=4: forward-last radix-4 (s = m0), B_j lands at k = p + j*m0 */
#define STMID_B4 \
        double t0r = ar + cr, t0i = ai_ + ci, t1r = ar - cr, t1i = ai_ - ci; \
        double t2r = br + dr, t2i = bi + di,  t3r = br - dr, t3i = bi - di; \
        B0r = t0r + t2r; B0i = t0i + t2i; \
        B1r = t1r + t3i; B1i = t1i - t3r; \
        B2r = t0r - t2r; B2i = t0i - t2i; \
        B3r = t1r - t3i; B3i = t1i + t3r;

/* rl=2: forward-last radix-2 (s = 2*m0), butterflies at q=p and q=p+m0:
   B[p] = a+c, B[p+m0] = b+d, B[p+2m0] = a-c, B[p+3m0] = b-d */
#define STMID_B2 \
        B0r = ar + cr; B0i = ai_ + ci; \
        B1r = br + dr; B1i = bi + di; \
        B2r = ar - cr; B2i = ai_ - ci; \
        B3r = br - dr; B3i = bi - di;

#define STMID_ARGS int m0, const double *restrict wr, const double *restrict wi, \
        const double *restrict xr, const double *restrict xi, \
        const double *restrict brh, const double *restrict bih, \
        double *restrict yr, double *restrict yi

HOT static void stmid4(STMID_ARGS) STMID_BODY(STMID_B4)
HOT static void stmid2(STMID_ARGS) STMID_BODY(STMID_B2)

/* ---------------- Rader fused entry / exit stages (d1_rader's shape) --------
 * entry: the g^q gather feeds the radix-4 stage-0 butterfly directly (the
 * gathered sequence never hits memory). The DC sum is NOT accumulated here:
 * it is the forward conv's bin 0 (d1_rader's "X[0] free from conv DC").
 * GATHER8 and interleave4_store are ADOPTED FROM d1_rader (r2): 8 random
 * complex points loaded as 128-bit pairs assembled with vinsertf64x2 + one
 * parity permute per plane -- beats vgatherdpd (microcoded on Ice Lake) for
 * cache-resident input; their measured 2.0 vs 2.8 us inside a 9.6 us case. */
__attribute__((always_inline, target("avx512f,avx512dq")))
static inline void interleave4_store(double *restrict y, int p0,
                                     __m512d A, __m512d B, __m512d C, __m512d D)
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

/* pf (r6, ADOPTED FROM d1_rader r5): prefetch the NEXT iteration's 32 random
 * read targets, T0.  Pays only when the batch working set is DRAM-resident
 * (their a80n0 A/B: B=16 -26 us, B=1 +16 us) -- so the caller gates it on
 * batch >= 2. */
__attribute__((target("avx512f,avx512dq")))
static void st4_first_gather(int m, const double *restrict wr, const double *restrict wi,
        const double *restrict xd, const int *restrict qin,
        double *restrict yr, double *restrict yi, int pf)
{
    const double *restrict w1r = wr,         *restrict w1i = wi;
    const double *restrict w2r = wr + m,     *restrict w2i = wi + m;
    const double *restrict w3r = wr + 2 * m, *restrict w3i = wi + 2 * m;
    const __m512i EVN = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i ODD = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
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
        if (pf && p + 16 <= m) {
#pragma GCC unroll 8
            for (int f = 0; f < 8; ++f) {
                __builtin_prefetch(xd + 2 * qin[p + 8 + f], 0, 3);
                __builtin_prefetch(xd + 2 * qin[p + m + 8 + f], 0, 3);
                __builtin_prefetch(xd + 2 * qin[p + 2 * m + 8 + f], 0, 3);
                __builtin_prefetch(xd + 2 * qin[p + 3 * m + 8 + f], 0, 3);
            }
        }
        GATHER8(qin + p,         ar, ai);
        GATHER8(qin + p + m,     br, bi);
        GATHER8(qin + p + 2 * m, cr, ci);
        GATHER8(qin + p + 3 * m, dr, di);
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
#undef GATHER8
#pragma GCC ivdep
    for (; p < m; ++p) {
        int j0 = qin[p], j1 = qin[p + m], j2 = qin[p + 2 * m], j3 = qin[p + 3 * m];
        double ar = xd[2 * j0], ai = xd[2 * j0 + 1];
        double br = xd[2 * j1], bi = xd[2 * j1 + 1];
        double cr = xd[2 * j2], ci = xd[2 * j2 + 1];
        double dr = xd[2 * j3], di = xd[2 * j3 + 1];
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

/* Rader CHAIN entry: same radix-4 stage-0 but reading the conv-order state
 * through the index REVERSAL g[q] = S[(P-q) mod P] (d1_rader's r1 insight:
 * gather-of-scatter between chain steps is a pure reversal, so interior steps
 * have no random access at all). Four backwards-contiguous streams. */
HOT static void st4_first_rev(int m, const double *restrict wr, const double *restrict wi,
        const double *restrict sxr, const double *restrict sxi,
        double *restrict yr, double *restrict yi)
{
    const double *restrict w1r = wr,         *restrict w1i = wi;
    const double *restrict w2r = wr + m,     *restrict w2i = wi + m;
    const double *restrict w3r = wr + 2 * m, *restrict w3i = wi + 2 * m;
    const int P = 4 * m;
    {   /* p == 0: g[0] = S[0], the others fall on the backwards streams */
        double ar = sxr[0], ai = sxi[0];
        double br = sxr[P - m], bi = sxi[P - m];
        double cr = sxr[P - 2 * m], ci = sxi[P - 2 * m];
        double dr = sxr[P - 3 * m], di = sxi[P - 3 * m];
        double t0r = ar + cr, t0i = ai + ci, t1r = ar - cr, t1i = ai - ci;
        double t2r = br + dr, t2i = bi + di, t3r = br - dr, t3i = bi - di;
        double y1r = t1r + t3i, y1i = t1i - t3r;
        double y2r = t0r - t2r, y2i = t0i - t2i;
        double y3r = t1r - t3i, y3i = t1i + t3r;
        yr[0] = t0r + t2r;  yi[0] = t0i + t2i;
        yr[1] = y1r * w1r[0] - y1i * w1i[0];
        yi[1] = y1r * w1i[0] + y1i * w1r[0];
        yr[2] = y2r * w2r[0] - y2i * w2i[0];
        yi[2] = y2r * w2i[0] + y2i * w2r[0];
        yr[3] = y3r * w3r[0] - y3i * w3i[0];
        yi[3] = y3r * w3i[0] + y3i * w3r[0];
    }
    const double *restrict a0r = sxr + P, *restrict a0i = sxi + P;
    const double *restrict a1r = sxr + P - m, *restrict a1i = sxi + P - m;
    const double *restrict a2r = sxr + P - 2 * m, *restrict a2i = sxi + P - 2 * m;
    const double *restrict a3r = sxr + P - 3 * m, *restrict a3i = sxi + P - 3 * m;
#pragma GCC ivdep
    for (int p = 1; p < m; ++p) {
        double ar = a0r[-p], ai = a0i[-p];
        double br = a1r[-p], bi = a1i[-p];
        double cr = a2r[-p], ci = a2i[-p];
        double dr = a3r[-p], di = a3i[-p];
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

/* final inverse radix-4 stage (m=1, s=P/4, unit twiddles, swapped planes) with
 * the Rader output scatter folded in: yd[2*qout[t]] = x0 + conv[t].
 * Restructured (d1_rader r2's SINKSTORE lesson): the butterfly is computed in
 * vector-friendly blocks staged through the stack, only the random-index
 * stores are scalar -- Ice Lake vscatterdpd is microcoded and loses. */
HOT static void st4_last_scatter(int s, const double *restrict xr, const double *restrict xi,
        const int *restrict qout, double x0r, double x0i, double *restrict yd)
{
    int q = 0;
    for (; q + 8 <= s; q += 8) {
        /* r6, ADOPTED FROM d1_rader r5: prefetch the write targets 16 points
         * ahead with the exclusive hint (builtin rw=1 -> prefetchw where PRFCHW is on).  Pays at every batch (their A/B:
         * B=1 889->853, B=16 1175->1002): the output lines are cold even at
         * B=1, evicted by the transform's own ~2 MB of ping-pong traffic --
         * an RFO to L3/DRAM per pair, hideable and never hidden before. */
        if (q + 24 <= s) {
#pragma GCC unroll 8
            for (int f = 0; f < 8; ++f) {
                __builtin_prefetch(yd + 2 * qout[q + 16 + f], 1, 3);
                __builtin_prefetch(yd + 2 * qout[q + 16 + f + s], 1, 3);
                __builtin_prefetch(yd + 2 * qout[q + 16 + f + 2 * s], 1, 3);
                __builtin_prefetch(yd + 2 * qout[q + 16 + f + 3 * s], 1, 3);
            }
        }
        double zr[4][8], zi[4][8];
#pragma GCC ivdep
        for (int l = 0; l < 8; ++l) {
            int t = q + l;
            /* butterfly per plane; conv_r comes from the i-plane (unswap) */
            double ar = xr[t], ai = xi[t];
            double br = xr[t + s], bi = xi[t + s];
            double cr = xr[t + 2 * s], ci = xi[t + 2 * s];
            double dr = xr[t + 3 * s], di = xi[t + 3 * s];
            double t0r = ar + cr, t0i = ai + ci, t1r = ar - cr, t1i = ai - ci;
            double t2r = br + dr, t2i = bi + di, t3r = br - dr, t3i = bi - di;
            zr[0][l] = x0r + (t0i + t2i);  zi[0][l] = x0i + (t0r + t2r);
            zr[1][l] = x0r + (t1i - t3r);  zi[1][l] = x0i + (t1r + t3i);
            zr[2][l] = x0r + (t0i - t2i);  zi[2][l] = x0i + (t0r - t2r);
            zr[3][l] = x0r + (t1i + t3r);  zi[3][l] = x0i + (t1r - t3i);
        }
        for (int j = 0; j < 4; ++j) {
            const int *restrict ox = qout + q + j * s;
            for (int l = 0; l < 8; ++l) {
                yd[2 * ox[l]]     = zr[j][l];
                yd[2 * ox[l] + 1] = zi[j][l];
            }
        }
    }
    for (; q < s; ++q) {
        double ar = xr[q], ai = xi[q];
        double br = xr[q + s], bi = xi[q + s];
        double cr = xr[q + 2 * s], ci = xi[q + 2 * s];
        double dr = xr[q + 3 * s], di = xi[q + 3 * s];
        double t0r = ar + cr, t0i = ai + ci, t1r = ar - cr, t1i = ai - ci;
        double t2r = br + dr, t2i = bi + di, t3r = br - dr, t3i = bi - di;
        int k0 = qout[q], k1 = qout[q + s], k2 = qout[q + 2 * s], k3 = qout[q + 3 * s];
        yd[2 * k0] = x0r + t0i + t2i;  yd[2 * k0 + 1] = x0i + t0r + t2r;
        yd[2 * k1] = x0r + t1i - t3r;  yd[2 * k1 + 1] = x0i + t1r + t3i;
        yd[2 * k2] = x0r + t0i - t2i;  yd[2 * k2 + 1] = x0i + t0r - t2r;
        yd[2 * k3] = x0r + t1i + t3r;  yd[2 * k3 + 1] = x0i + t1r - t3i;
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

/* NT-streamed variants of the pruned chirp exits (r6, ADOPTED FROM
 * d1_bluestein r5 near-verbatim): for batched mode-2 cells whose output is
 * DRAM-resident (10007 x B=64 = 10 MB out, 100003 x B=8 = 12.8 MB), y is
 * written once and never re-read, so streaming stores skip the RFO -- a third
 * of the exit traffic.  The output is CONTIGUOUS, so alignment is bought by
 * peeling scalar head complexes until the store address is 64B-aligned;
 * all source reads are loadu.  sfence at the end (this file's convention). */
__attribute__((target("avx512f")))
static void st2_last_chirp_nt(int s, const double *restrict xr, const double *restrict xi,
        const double *restrict car, const double *restrict cai,
        double *restrict yd, int L)
{
    const __m512i ixlo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i ixhi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    const int pad = (int)(((64 - ((uintptr_t)yd & 63)) & 63) >> 3);  /* doubles */
    const int h = pad >> 1;      /* head complexes (pad even: yd is 16B-aligned) */
    int k = 0;
    for (; k < h && k < L; ++k) {
        double w_i = xr[k] + xr[k + s];
        double w_r = xi[k] + xi[k + s];
        yd[2 * k]     = w_r * car[k] - w_i * cai[k];
        yd[2 * k + 1] = w_r * cai[k] + w_i * car[k];
    }
    for (; k + 8 <= L; k += 8) {
        __m512d w_i = _mm512_add_pd(_mm512_loadu_pd(xr + k), _mm512_loadu_pd(xr + k + s));
        __m512d w_r = _mm512_add_pd(_mm512_loadu_pd(xi + k), _mm512_loadu_pd(xi + k + s));
        __m512d vcr = _mm512_loadu_pd(car + k), vci = _mm512_loadu_pd(cai + k);
        __m512d re = _mm512_fmsub_pd(w_r, vcr, _mm512_mul_pd(w_i, vci));
        __m512d im = _mm512_fmadd_pd(w_r, vci, _mm512_mul_pd(w_i, vcr));
        _mm512_stream_pd(yd + 2 * k,     _mm512_permutex2var_pd(re, ixlo, im));
        _mm512_stream_pd(yd + 2 * k + 8, _mm512_permutex2var_pd(re, ixhi, im));
    }
    for (; k < L; ++k) {
        double w_i = xr[k] + xr[k + s];
        double w_r = xi[k] + xi[k + s];
        yd[2 * k]     = w_r * car[k] - w_i * cai[k];
        yd[2 * k + 1] = w_r * cai[k] + w_i * car[k];
    }
    _mm_sfence();
}

/* radix-4 twin: two contiguous output segments, j = 0 at yd[0..2s) and j = 1
 * at yd[2s..2L); s = M/4 is a multiple of 8 here so both share the head pad */
__attribute__((target("avx512f")))
static void st4_last_chirp_nt(int s, const double *restrict xr, const double *restrict xi,
        const double *restrict car, const double *restrict cai,
        double *restrict yd, int L)
{
    const __m512i ixlo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i ixhi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    const int pad = (int)(((64 - ((uintptr_t)yd & 63)) & 63) >> 3);
    const int h = pad >> 1;
    int q = 0;
    for (; q < h && q < s; ++q) {           /* j = 0: X0 = a+b+c+d */
        double w_i = xr[q] + xr[q + s] + xr[q + 2 * s] + xr[q + 3 * s];
        double w_r = xi[q] + xi[q + s] + xi[q + 2 * s] + xi[q + 3 * s];
        yd[2 * q]     = w_r * car[q] - w_i * cai[q];
        yd[2 * q + 1] = w_r * cai[q] + w_i * car[q];
    }
    for (; q + 8 <= s; q += 8) {
        __m512d w_i = _mm512_add_pd(
            _mm512_add_pd(_mm512_loadu_pd(xr + q), _mm512_loadu_pd(xr + q + s)),
            _mm512_add_pd(_mm512_loadu_pd(xr + q + 2 * s), _mm512_loadu_pd(xr + q + 3 * s)));
        __m512d w_r = _mm512_add_pd(
            _mm512_add_pd(_mm512_loadu_pd(xi + q), _mm512_loadu_pd(xi + q + s)),
            _mm512_add_pd(_mm512_loadu_pd(xi + q + 2 * s), _mm512_loadu_pd(xi + q + 3 * s)));
        __m512d vcr = _mm512_loadu_pd(car + q), vci = _mm512_loadu_pd(cai + q);
        __m512d re = _mm512_fmsub_pd(w_r, vcr, _mm512_mul_pd(w_i, vci));
        __m512d im = _mm512_fmadd_pd(w_r, vci, _mm512_mul_pd(w_i, vcr));
        _mm512_stream_pd(yd + 2 * q,     _mm512_permutex2var_pd(re, ixlo, im));
        _mm512_stream_pd(yd + 2 * q + 8, _mm512_permutex2var_pd(re, ixhi, im));
    }
    for (; q < s; ++q) {
        double w_i = xr[q] + xr[q + s] + xr[q + 2 * s] + xr[q + 3 * s];
        double w_r = xi[q] + xi[q + s] + xi[q + 2 * s] + xi[q + 3 * s];
        yd[2 * q]     = w_r * car[q] - w_i * cai[q];
        yd[2 * q + 1] = w_r * cai[q] + w_i * car[q];
    }
    const int n1 = L - s;                   /* j = 1: X1 = (a-c) - i(b-d) */
    q = 0;
    for (; q < h && q < n1; ++q) {
        double t1r = xr[q] - xr[q + 2 * s], t1i = xi[q] - xi[q + 2 * s];
        double t3r = xr[q + s] - xr[q + 3 * s], t3i = xi[q + s] - xi[q + 3 * s];
        double w_i = t1r + t3i, w_r = t1i - t3r;
        int k = s + q;
        yd[2 * k]     = w_r * car[k] - w_i * cai[k];
        yd[2 * k + 1] = w_r * cai[k] + w_i * car[k];
    }
    for (; q + 8 <= n1; q += 8) {
        __m512d t1r = _mm512_sub_pd(_mm512_loadu_pd(xr + q), _mm512_loadu_pd(xr + q + 2 * s));
        __m512d t1i = _mm512_sub_pd(_mm512_loadu_pd(xi + q), _mm512_loadu_pd(xi + q + 2 * s));
        __m512d t3r = _mm512_sub_pd(_mm512_loadu_pd(xr + q + s), _mm512_loadu_pd(xr + q + 3 * s));
        __m512d t3i = _mm512_sub_pd(_mm512_loadu_pd(xi + q + s), _mm512_loadu_pd(xi + q + 3 * s));
        __m512d w_i = _mm512_add_pd(t1r, t3i);
        __m512d w_r = _mm512_sub_pd(t1i, t3r);
        __m512d vcr = _mm512_loadu_pd(car + s + q), vci = _mm512_loadu_pd(cai + s + q);
        __m512d re = _mm512_fmsub_pd(w_r, vcr, _mm512_mul_pd(w_i, vci));
        __m512d im = _mm512_fmadd_pd(w_r, vci, _mm512_mul_pd(w_i, vcr));
        _mm512_stream_pd(yd + 2 * (s + q),     _mm512_permutex2var_pd(re, ixlo, im));
        _mm512_stream_pd(yd + 2 * (s + q) + 8, _mm512_permutex2var_pd(re, ixhi, im));
    }
    for (; q < n1; ++q) {
        double t1r = xr[q] - xr[q + 2 * s], t1i = xi[q] - xi[q + 2 * s];
        double t3r = xr[q + s] - xr[q + 3 * s], t3i = xi[q + s] - xi[q + 3 * s];
        double w_i = t1r + t3i, w_r = t1i - t3r;
        int k = s + q;
        yd[2 * k]     = w_r * car[k] - w_i * cai[k];
        yd[2 * k + 1] = w_r * cai[k] + w_i * car[k];
    }
    _mm_sfence();
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
    int nt2;                  /* NT-stream the pruned chirp exit (batched out >= 8 MB) */
    int fuse_last;            /* 2/4: fused pruned exit radix; 0: none */
    int fuse_mid;             /* 2/4: fused fwd-last x kernel x inv-entry; 0: none */
    double *ar, *ai;          /* chirp, length L */
    double *br, *bi;          /* kernel spectrum (1/n folded), length P or M */
    /* work */
    double *s0r, *s0i, *s1r, *s1i;   /* ping-pong planes, core length */
    double *gs;               /* generic-stage scratch (per-vector) */
    /* lanes (kind 0, batch >= LANEV, L <= LANE_MAX_N) */
    int lanes;
    int lanes_chain_only;     /* multi-stage smooth: lanes win the chain (the
                                 transpose amortizes over m) but lose execute */
    int nt;                   /* NT-stream the fused exit (in+out > node L3) */
    double *l0r, *l0i, *l1r, *l1i;   /* LANEV*L each */
    double *lgs;              /* generic-stage scratch at stride LANEV */
    core_plan lcore;          /* r7: lane-CHAIN-only schedule (64=[8,8],
                                 128=[16,8] -> 2 fused passes/step; the execute
                                 plan keeps its radix-4 entry for the fused
                                 deinterleave, which the chain never needs) */
    int lcore_on;
    /* chain state (lazy; transform-outer, so one transform's worth suffices) */
    double *pre_r, *pre_i;    /* split state, length L */
    double *tzr, *tzi;        /* one transform's split output, length L */
    double *lcr, *lci;        /* lane-blocked c field, LANEV*L */
    double *c2r, *c2i;        /* Bluestein chain: split c field, length L (r7) */
    double *cfr, *cfi;        /* Rader chain: c field permuted to conv order, length P */
    void *arena;              /* single skewed huge-page block owning the planes above */
    /* small-prime pair path (L = 13/31, ADOPTED FROM d1_prime r3) */
    int sp_h, sp_hp;          /* h = (L-1)/2, hp = (h+1+7) & ~7 */
    double *sp_tp;            /* pair tables: 13 -> [j][4][8], 31 -> [j][8][8] */
    double *sp_tc, *sp_ts;    /* [h][hp] k-contiguous, k=0 col in lane h (chain B=1) */
    double *sp_ck, *sp_sk;    /* [h][h] j-contiguous (batched chain) */
    /* pow2 codelet path (L = 32/64, ADOPTED FROM d1_pow2 r1) */
    double *pw_tf;            /* dup-format first-stage w/w^2/w^3, 48 dbl per 4-p */
    double *pw_t2;            /* L=64 stage-2 radix-4 n=16 table, 9 dbl per p */
};

const char *fft1d_name(void) { return "d1_planner"; }
const char *fft1d_description(void)
{
    return "LIBRARY LAYER (adoption-scored): factorization planner on a split-complex "
           "Stockham core (true-zmm noinline kernels; 2/3/4/5/8 + radix-16/64 L1-tile "
           "stages + generic prime<=61 stage; paired-p zmm s=4 kernels at 3/8/16/64) -- "
           "direct smooth / unpadded Rader (smooth L-1) / smooth-padded Bluestein, both "
           "with the fused fwd-last x kernel x inv-entry mid pass; TILE EXITS "
           "st{16,64}_last_{int,map} make the tile the final stage: 4096=[4,64,16] and "
           "16384=[4,64,64] run 3 passes (conv rows keep 2/4 tails for the mid/exit "
           "fusions); every chain step fuses map(z+c) into its final stage (split, "
           "zero-shuffle; Rader conv-order chain included via the swapped-plane "
           "st4_last_map_rader), lane chains on 2-pass chain-only schedules "
           "(64=[8,8], 128=[16,8]); Goldschmidt + early-seeded-rcp map everywhere; "
           "whole-2MB-page pre-faulted arena; Rader gather T0 (batch>=2) + scatter ET0 "
           "prefetch; adopted codelets: d1_prime 13/31, d1_pow2 32/64, d1_composite "
           "PFA-60 (B<8 chain tail); NT fused exits above L3 incl. the tile exits";
}

int fft1d_supports(int L) { return L >= 1; }

/* build the d1_prime-style tables for the 13/31 pair path; on any failure the
 * sp_* fields stay NULL and the generic dense path serves the size instead */
static void sp_build(fft1d_plan *p, int L)
{
    const int h = (L - 1) / 2, hp = (h + 1 + 7) & ~7;
    p->sp_h = h; p->sp_hp = hp;
    size_t tpn = (L == 13) ? (size_t)6 * 4 * 8 : (size_t)15 * 8 * 8;
    p->sp_tp = amalloc(tpn * sizeof(double));
    p->sp_tc = amalloc((size_t)h * hp * sizeof(double));
    p->sp_ts = amalloc((size_t)h * hp * sizeof(double));
    p->sp_ck = amalloc((size_t)h * h * sizeof(double));
    p->sp_sk = amalloc((size_t)h * h * sizeof(double));
    if (!p->sp_tp || !p->sp_tc || !p->sp_ts || !p->sp_ck || !p->sp_sk) {
        free(p->sp_tp); free(p->sp_tc); free(p->sp_ts); free(p->sp_ck); free(p->sp_sk);
        p->sp_tp = p->sp_tc = p->sp_ts = p->sp_ck = p->sp_sk = NULL;
        return;
    }
    memset(p->sp_tp, 0, tpn * sizeof(double));
    memset(p->sp_tc, 0, (size_t)h * hp * sizeof(double));
    memset(p->sp_ts, 0, (size_t)h * hp * sizeof(double));
    for (int k = 1; k <= h; k++)
        for (int j = 1; j <= h; j++) {
            int r = (k * j) % L;
            int rr = (2 * r > L) ? r - L : r;    /* reduced angle: cos even, sin exact */
            long double th = 2.0L * PIL * (long double)rr / (long double)L;
            double c = (double)cosl(th), s = (double)sinl(th);
            p->sp_tc[(size_t)(j-1)*hp + (k-1)] = c;
            p->sp_ts[(size_t)(j-1)*hp + (k-1)] = s;
            p->sp_ck[(size_t)(k-1)*h + (j-1)] = c;
            p->sp_sk[(size_t)(k-1)*h + (j-1)] = s;
        }
    for (int j = 1; j <= h; j++) {
        p->sp_tc[(size_t)(j-1)*hp + h] = 1.0;   /* k=0 column: X0 = x0 + sum u */
        p->sp_ts[(size_t)(j-1)*hp + h] = 0.0;
    }
    if (L == 13) {
        for (int j = 1; j <= 6; j++) {
            double *cpA = p->sp_tp + (size_t)(j-1)*32, *cpB = cpA + 8;
            double *spA = cpB + 8,                     *spB = spA + 8;
            for (int k = 1; k <= 6; k++) {
                double c = p->sp_ck[(size_t)(k-1)*h + (j-1)];
                double s = p->sp_sk[(size_t)(k-1)*h + (j-1)];
                if (k <= 4) { cpA[2*(k-1)] = cpA[2*(k-1)+1] = c;
                              spA[2*(k-1)] = s;  spA[2*(k-1)+1] = -s; }
                else        { cpB[2*(k-5)] = cpB[2*(k-5)+1] = c;
                              spB[2*(k-5)] = s;  spB[2*(k-5)+1] = -s; }
            }
            cpB[4] = cpB[5] = 1.0;   /* k=0 column rides pair 2 of the B row */
        }
    } else {  /* L == 31: [j][8][8], rows cpA..cpD then spA..spD */
        for (int j = 1; j <= 15; j++) {
            double *base = p->sp_tp + (size_t)(j-1)*64;
            for (int k = 1; k <= 15; k++) {
                double c = p->sp_ck[(size_t)(k-1)*h + (j-1)];
                double s = p->sp_sk[(size_t)(k-1)*h + (j-1)];
                int row = (k-1) >> 2, t = (k-1) & 3;
                base[8*row + 2*t]      = c;  base[8*row + 2*t + 1]      =  c;
                base[8*(row+4) + 2*t]  = s;  base[8*(row+4) + 2*t + 1]  = -s;
            }
            base[8*3 + 6] = base[8*3 + 7] = 1.0;   /* k=0 col: pair 3 of row D */
        }
    }
}

/* d1_pow2's dup-format first-stage table (per 4-p group: [w1r|w1p|w2r|w2p|
 * w3r|w3p] x 8 doubles, wp = (-sin,+sin)) and, for L=64, the 9-doubles-per-p
 * broadcast table of the radix-4 n=16 stage. NULL fields on failure ->
 * generic path serves the size. */
static void pw_build(fft1d_plan *p, int L)
{
    int m = L / 4;
    p->pw_tf = amalloc((size_t)(m / 4) * 48 * sizeof(double));
    if (!p->pw_tf) return;
    for (int q = 0; q < m; ++q) {
        int g = q / 4, j = q % 4;
        for (int r = 1; r <= 3; ++r) {
            long double th = -2.0L * PIL * (long double)(q * r % L) / (long double)L;
            double *base = p->pw_tf + g * 48 + (r - 1) * 16;
            base[2 * j]     = (double)cosl(th);
            base[2 * j + 1] = (double)cosl(th);
            base[8 + 2 * j]     = -(double)sinl(th);
            base[8 + 2 * j + 1] =  (double)sinl(th);
        }
    }
    if (L == 64) {
        p->pw_t2 = amalloc((size_t)4 * 9 * sizeof(double));
        if (!p->pw_t2) { free(p->pw_tf); p->pw_tf = NULL; return; }
        for (int pp = 0; pp < 4; ++pp)
            for (int r = 1; r <= 3; ++r) {
                long double th = -2.0L * PIL * (long double)(pp * r % 16) / 16.0L;
                p->pw_t2[9 * pp + (r - 1)] = (double)cosl(th);
                p->pw_t2[9 * pp + 3 + 2 * (r - 1)]     = -(double)sinl(th);
                p->pw_t2[9 * pp + 3 + 2 * (r - 1) + 1] =  (double)sinl(th);
            }
    }
}

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
        if (L == 13 || L == 31) sp_build(p, L);   /* d1_prime pair path */
        if (L == 32 || L == 64) pw_build(p, L);   /* d1_pow2 codelet path */
        /* NT exit once in+out exceeds the scoring node's 24 MB L3 (d1_pow2 r3):
         * catches 4096xB=256 and 16384xB=64 (33.5 MB), leaves 1024xB=512 alone */
        p->nt = ((size_t)32 * L * batch >= (size_t)25 * 1024 * 1024);
        if (L >= 512) {
            p->arena = arena_alloc(8 * carve_stride((size_t)L));
            if (!p->arena) goto fail;
            carver cv = { p->arena, 0 };
            p->s0r = carve(&cv, L);  p->s0i = carve(&cv, L);
            p->s1r = carve(&cv, L);  p->s1i = carve(&cv, L);
            p->pre_r = carve(&cv, L);  p->pre_i = carve(&cv, L);
            p->tzr = carve(&cv, L);    p->tzi = carve(&cv, L);
        } else {
            p->s0r = amalloc((size_t)L * sizeof(double));
            p->s0i = amalloc((size_t)L * sizeof(double));
            p->s1r = amalloc((size_t)L * sizeof(double));
            p->s1i = amalloc((size_t)L * sizeof(double));
            if (!p->s0r || !p->s0i || !p->s1r || !p->s1i) goto fail;
        }
        size_t g = core_gen_scratch(&p->core, 1);
        if (p->dense) { size_t d = 4 * (size_t)((L - 1) / 2); if (d > g) g = d; }
        if (g) { p->gs = amalloc(g * sizeof(double)); if (!p->gs) goto fail; }
        /* lanes only pay where the per-vector path runs scalar-ish: the dense
         * single-generic-stage plans (13/31/...). For multi-stage smooth plans
         * the per-vector split kernels already vectorize and the two lane
         * transposes lose (A/B wallaby: 32: 0.049 vs 0.041, 64: 0.108 vs 0.080,
         * 1024: 5.1 vs 2.4 us; but 13: 0.037 vs 0.042, 31: 0.139 vs 0.160). */
        /* lanes serve: dense plans without a pair path (execute + chain), and
         * the L=32/64 batched chains (r7: with the map fused into the lane
         * steps' final stage, the lane path beats the 64 codelet chain too --
         * r2's "64 stays on the codelet" verdict was for the UNFUSED lane
         * loop; PLN_64L=0 restores it for A/B) */
        /* chain-only lanes for small multi-stage smooth L (r4 node A/B: 60
         * B=512 chain 0.212 -> 0.130, 128 chain 0.342 -> 0.288, but execute
         * 0.153 -> 0.206 / 0.287 -> 0.413 -- the two transposes only pay for
         * themselves when they amortize over the m chain steps) */
        p->lanes_chain_only = (batch >= LANEV && !p->dense && !p->pw_tf && L <= 128);
        {
            const char *e64 = getenv("PLN_64L");
            int lane64 = !(e64 && *e64 == '0');
            if (batch >= LANEV && L <= LANE_MAX_N &&
                ((p->dense && !p->sp_tp) || (p->pw_tf && L == 32) ||
                 (p->pw_tf && L == 64 && lane64) ||
                 p->lanes_chain_only)) {
            p->lanes = 1;
            p->l0r = amalloc((size_t)LANEV * L * sizeof(double));
            p->l0i = amalloc((size_t)LANEV * L * sizeof(double));
            p->l1r = amalloc((size_t)LANEV * L * sizeof(double));
            p->l1i = amalloc((size_t)LANEV * L * sizeof(double));
            if (!p->l0r || !p->l0i || !p->l1r || !p->l1i) goto fail;
            size_t lg = core_gen_scratch(&p->core, LANEV);
            if (lg) { p->lgs = amalloc(lg * sizeof(double)); if (!p->lgs) goto fail; }
            const char *elc = getenv("PLN_LC");
            if (!(elc && *elc == '0') && (L == 64 || L == 128)) {
                static const int r64[2] = {8, 8}, r128[2] = {16, 8};
                if (core_init_with(&p->lcore, L, (L == 64) ? r64 : r128, 2))
                    p->lcore_on = 1;
                else
                    core_free(&p->lcore);
            }
            }
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
            if (!p->qin || !p->qout) goto fail;
            if (P >= 512) {
                p->arena = arena_alloc(8 * carve_stride((size_t)P) +
                                       4 * carve_stride((size_t)L));
                if (!p->arena) goto fail;
                carver cv = { p->arena, 0 };
                p->s0r = carve(&cv, P);  p->s0i = carve(&cv, P);
                p->s1r = carve(&cv, P);  p->s1i = carve(&cv, P);
                p->br  = carve(&cv, P);  p->bi  = carve(&cv, P);
                p->cfr = carve(&cv, P);  p->cfi = carve(&cv, P);
                p->pre_r = carve(&cv, L);  p->pre_i = carve(&cv, L);
                p->tzr = carve(&cv, L);    p->tzi = carve(&cv, L);
            } else {
                p->br  = amalloc((size_t)P * sizeof(double));
                p->bi  = amalloc((size_t)P * sizeof(double));
                p->s0r = amalloc((size_t)P * sizeof(double));
                p->s0i = amalloc((size_t)P * sizeof(double));
                p->s1r = amalloc((size_t)P * sizeof(double));
                p->s1i = amalloc((size_t)P * sizeof(double));
                if (!p->br || !p->bi ||
                    !p->s0r || !p->s0i || !p->s1r || !p->s1i) goto fail;
            }
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
            /* r2 measured the SCALAR fused gather/scatter a wash on wallaby;
             * r3 enables both with d1_rader's vectorized shapes (GATHER8
             * insert-assembled entry, block-staged scalar-store exit), which
             * their node A/Bs showed winning -- the saved P-length passes are
             * ~2x1MB of traffic each at 65537 on the scoring node's small L2. */
            p->rentry = (p->core.nstage >= 2 && p->core.radix[0] == 4);
            p->rexit  = (p->rentry && p->core.radix[p->core.nstage - 1] == 4);
            /* fused mid (fwd-last x kernel x inv-entry, r5 from d1_rader r4 /
             * d1_bluestein r4): needs entry radix 4 and last radix 2/4.
             * 65536 = [4,8,...,4] qualifies; 1020 ends in 17 and does not. */
            {
                int rl_ = p->core.radix[p->core.nstage - 1];
                p->fuse_mid = (p->rentry && (rl_ == 2 || rl_ == 4)) ? rl_ : 0;
            }
            return p;
        }
    }

    /* Bluestein */
    p->kind = 2;
    {
        int M = p->M = choose_M(L);
        if (M <= 0 || !core_init(&p->core, M)) goto fail;
        p->arena = arena_alloc(6 * carve_stride((size_t)M) +
                               6 * carve_stride((size_t)L));
        if (!p->arena) goto fail;
        carver cv = { p->arena, 0 };
        p->s0r = carve(&cv, M);  p->s0i = carve(&cv, M);
        p->s1r = carve(&cv, M);  p->s1i = carve(&cv, M);
        p->br  = carve(&cv, M);  p->bi  = carve(&cv, M);
        p->ar  = carve(&cv, L);  p->ai  = carve(&cv, L);
        p->pre_r = carve(&cv, L);  p->pre_i = carve(&cv, L);
        p->tzr = carve(&cv, L);    p->tzi = carve(&cv, L);
        int lastr = p->core.radix[p->core.nstage - 1];
        p->fuse_last = (p->core.nstage >= 2 && (lastr == 2 || lastr == 4)) ? lastr : 0;
        p->fuse_mid  = p->fuse_last;   /* same condition: entry 4 (4|M), last 2/4 */
        /* NT chirp exit (r6, d1_bluestein r5): batched output alone >= 8 MB is
         * DRAM-resident on the node's 24 MB L3 (10007 B=64, 100003 B=8); the
         * radix-4 form needs s = M/4 % 8 == 0 so both output segments share
         * the head pad.  Execute-only (bl_one); the chain re-reads its state. */
        p->nt2 = ((size_t)batch * (size_t)L * 16 >= ((size_t)8 << 20)) &&
                 (p->fuse_last == 2 || (p->fuse_last == 4 && (M / 4) % 8 == 0)) &&
                 !getenv("PLN_NONT");
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
    /* fused interleave exit: radices with a st*_last_int kernel; the r7 tile
       exits (16/64) need s % 8 == 0 (their blocks are maskless) */
    const int fx = (c->nstage >= 2 && !core_is_generic(rl) &&
                    ((rl != 16 && rl != 64) || (L / rl) % 8 == 0));
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
        const int ntok = p->nt && (s & 7) == 0 && (((uintptr_t)yd & 63) == 0);
        if (rl == 16 || rl == 64) {
            if (rl == 16) st16_last_int(s, Rr, Ri, yd, ntok);
            else          st64_last_int(s, Rr, Ri, yd, ntok);
            return;
        }
        if (ntok && (rl == 2 || rl == 4 || rl == 8)) {
            switch (rl) {
            case 2: st2_last_int_nt(s, Rr, Ri, yd); break;
            case 4: st4_last_int_nt(s, Rr, Ri, yd); break;
            default: st8_last_int_nt(s, Rr, Ri, yd); break;
            }
            return;
        }
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

/* forward stages 1.. (post-entry data in s0), kernel multiply, inverse -- with
 * the fused mid pass when enabled (r5, from d1_rader r4 st4_mid_bhat /
 * d1_bluestein r4 stmid): the forward stops one stage early and stmid runs
 * fwd-last x kernel x inv-entry in one pass; the spectrum DC bin (needed for
 * X[0] = x[0] + DC) is re-summed from the same last-stage inputs (their
 * "X[0] comes free from the partial sums"). Only entered when rentry (entry
 * radix 4); dcr/dci get the spectrum bin 0 either way. */
static void rader_mid(fft1d_plan *p, int fused_exit, double *dcr, double *dci,
                      double **Vr, double **Vi)
{
    const core_plan *c = &p->core;
    const int nst = c->nstage, m0 = p->P / 4;
    double *Rr, *Ri;
    if (p->fuse_mid) {
        core_exec_range(c, p->gs, 1, nst - 1, m0, 4,
                        p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
        if (p->fuse_mid == 4) {
            *dcr = Rr[0] + Rr[m0] + Rr[2 * m0] + Rr[3 * m0];
            *dci = Ri[0] + Ri[m0] + Ri[2 * m0] + Ri[3 * m0];
        } else {
            *dcr = Rr[0] + Rr[2 * m0];
            *dci = Ri[0] + Ri[2 * m0];
        }
        double *dr = (Rr == p->s0r) ? p->s1r : p->s0r;
        double *di = (Rr == p->s0r) ? p->s1i : p->s0i;
        if (p->fuse_mid == 4)
            stmid4(m0, c->twr[0], c->twi[0], Rr, Ri, p->br, p->bi, dr, di);
        else
            stmid2(m0, c->twr[0], c->twi[0], Rr, Ri, p->br, p->bi, dr, di);
        double *er = (dr == p->s0r) ? p->s1r : p->s0r;
        double *ei = (dr == p->s0r) ? p->s1i : p->s0i;
        core_exec_range(c, p->gs, 1, fused_exit ? nst - 1 : nst, m0, 4,
                        dr, di, er, ei, Vr, Vi);
        return;
    }
    core_exec_range(c, p->gs, 1, nst, m0, 4,
                    p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
    *dcr = Rr[0]; *dci = Ri[0];
    rader_inv(p, Rr, Ri, fused_exit, Vr, Vi);
}

static void rader_one(fft1d_plan *p, const double *restrict xd, double *restrict yd)
{
    const int P = p->P;
    const core_plan *c = &p->core;
    const int nst = c->nstage;
    const int *restrict qin = p->qin, *restrict qout = p->qout;
    double sr, si;
    const double x0r = xd[0], x0i = xd[1];
    double *Vr, *Vi;
    if (p->rentry) {
        st4_first_gather(P / 4, c->twr[0], c->twi[0], xd, qin,
                         p->s0r, p->s0i, p->batch >= 2);
        rader_mid(p, p->rexit, &sr, &si, &Vr, &Vi);
    } else {
        double *restrict gr = p->s0r, *restrict gi = p->s0i;
        double *Rr, *Ri;
        sr = 0.0; si = 0.0;
        for (int q = 0; q < P; ++q) {
            int j = qin[q];
            double xr = xd[2 * j], xi = xd[2 * j + 1];
            gr[q] = xr; gi[q] = xi;
            sr += xr; si += xi;
        }
        core_exec_range(c, p->gs, 0, nst, P, 1,
                        p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
        rader_inv(p, Rr, Ri, p->rexit, &Vr, &Vi);
    }
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
    if (p->fuse_mid) {
        /* forward stops one stage early; stmid runs fwd-last x kernel x
         * inv-entry in one pass (r5, from d1_bluestein r4 / d1_rader r4) */
        core_exec_range(c, NULL, 1, nst - 1, m0, 4, p->s1r, p->s1i, p->s0r, p->s0i, &Rr, &Ri);
        double *dr = (Rr == p->s0r) ? p->s1r : p->s0r;
        double *di = (Rr == p->s0r) ? p->s1i : p->s0i;
        if (p->fuse_mid == 4)
            stmid4(m0, c->twr[0], c->twi[0], Rr, Ri, p->br, p->bi, dr, di);
        else
            stmid2(m0, c->twr[0], c->twi[0], Rr, Ri, p->br, p->bi, dr, di);
        double *er = (dr == p->s0r) ? p->s1r : p->s0r;
        double *ei = (dr == p->s0r) ? p->s1i : p->s0i;
        core_exec_range(c, NULL, 1, p->fuse_last ? nst - 1 : nst, m0, 4, dr, di, er, ei, Vr, Vi);
        return;
    }
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
        p->nt2 ? st2_last_chirp_nt(M / 2, Vr, Vi, p->ar, p->ai, yd, L)
               : st2_last_chirp(M / 2, Vr, Vi, p->ar, p->ai, yd, L);
    else if (p->fuse_last == 4)
        p->nt2 ? st4_last_chirp_nt(M / 4, Vr, Vi, p->ar, p->ai, yd, L)
               : st4_last_chirp(M / 4, Vr, Vi, p->ar, p->ai, yd, L);
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
#ifdef __AVX512F__
    if (p->sp_tp) {           /* 13/31 pair kernels (d1_prime r3 dispatch) */
        const int B = p->batch;
        const double *restrict tp = p->sp_tp;
        if (L == 13) {
            if (B == 1) exec13p_body(in, out, tp, 2);
            else if (B < 8) { for (int b = 0; b < B; b++)
                                  exec13p_body(in + 26*(size_t)b, out + 26*(size_t)b, tp, 2); }
            else { int b = 0;
                   for (; b + 2 <= B; b += 2) exec13p_b2(in + 26*(size_t)b, out + 26*(size_t)b, tp);
                   for (; b < B; b++)         exec13p_body(in + 26*(size_t)b, out + 26*(size_t)b, tp, 1); }
        } else {              /* L == 31 */
            for (int b = 0; b < B; b++)
                exec31p_body(in + 62*(size_t)b, out + 62*(size_t)b, tp);
        }
        return;
    }
    if (p->pw_tf) {           /* 32/64 in-register codelets (d1_pow2 r1) */
        if (L == 32) fft32_execute(p->pw_tf, in, out, p->batch);
        else         fft64_execute(p->pw_tf, p->pw_t2, in, out, p->batch);
        return;
    }
    if (L == 60 && p->kind == 0) {   /* PFA codelet (d1_composite), all batches */
        for (int b60 = 0; b60 < p->batch; ++b60)
            fft60_pfa(in + (size_t)b60 * stride, out + (size_t)b60 * stride);
        return;
    }
#endif
    int b = 0;
    if (p->kind == 0 && p->lanes && !p->lanes_chain_only) {
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

    if (p->sp_tc) {           /* 13/31 fused chains (d1_prime r1/r3) */
        const double *xd = (const double *)x0;
        double *od = (double *)final_out;
        if (B == 1) {
            if (L == 13) sp_chain1_body(13, 6, 1, p->sp_tc, p->sp_ts, xd, cin, od, m);
            else         sp_chain1_body(31, 15, 2, p->sp_tc, p->sp_ts, xd, cin, od, m);
        } else {
            if (L == 13) sp_chainblk_body(13, 6, p->sp_ck, p->sp_sk, xd, cin, od, m, B);
            else         sp_chainblk_body(31, 15, p->sp_ck, p->sp_sk, xd, cin, od, m, B);
        }
        return;
    }

#ifdef __AVX512F__
    /* 60: PFA fused chain (r5) -- r7: full lane groups go through the fused
     * lane path below instead (st5_last_map makes it 3 passes/step); the PFA
     * codelet keeps B < 8 and the odd tail. PLN_60L=0 restores r6's
     * all-batches PFA chain for A/B. */
    if (L == 60 && p->kind == 0) {
        const char *e60 = getenv("PLN_60L");
        if ((e60 && *e60 == '0') || !p->lanes || B < LANEV) {
            fft60_chain((const double *)x0, cin, (double *)final_out, B, m);
            return;
        }
    }
#endif

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
         * reordered transform-outer). r7: c is pre-SPLIT once per transform,
         * so the interior map + chirp-premultiply and the final map +
         * interleave run 8-wide divider-free (map_chirp_split / map_int_out)
         * instead of as scalar sqrt/div loops. */
        const int M = p->M, m0 = M / 4;
        const core_plan *cp = &p->core;
        const double *restrict car = p->ar, *restrict cai = p->ai;
        if (!p->c2r) {
            p->c2r = amalloc((size_t)L * sizeof(double));
            p->c2i = amalloc((size_t)L * sizeof(double));
            if (!p->c2r || !p->c2i) return;
        }
        for (int b = 0; b < B; ++b) {
            const double *restrict xd = (const double *)(x0 + (size_t)b * L);
            const double *restrict cd = cin + (size_t)b * 2 * L;
            double *restrict pr = p->pre_r;
            double *restrict pi = p->pre_i;
            double *restrict c2r = p->c2r, *restrict c2i = p->c2i;
#pragma GCC ivdep
            for (int k = 0; k < L; ++k) {
                double xr = xd[2 * k], xi = xd[2 * k + 1];
                pr[k] = xr * car[k] - xi * cai[k];
                pi[k] = xr * cai[k] + xi * car[k];
                c2r[k] = cd[2 * k];  c2i[k] = cd[2 * k + 1];
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
                if (s == m - 1)
                    map_int_out(L, p->tzr, p->tzi, c2r, c2i,
                                (double *)(final_out + (size_t)b * L));
                else
                    map_chirp_split(L, p->tzr, p->tzi, c2r, c2i, car, cai, pr, pi);
            }
        }
        return;
    }

    if (p->kind == 1) {
        /* Rader chain in CONV ORDER -- ADOPTED FROM d1_rader (r1): between
         * steps, gather∘scatter is a pure index reversal (qin[q] = g^q =
         * qout[(P-q) mod P]) and the elementwise map commutes with any
         * permutation, so the state lives split in conv order for the whole
         * chain: S[t] = state[qout[t]], state[0] rides as a scalar pair, the
         * c field is pre-permuted once per transform. Interior steps have NO
         * random gather and NO random scatter (r2's chain paid both per step:
         * 965 vs 813 us execute at 65537); one natural-order pass remains at
         * chain entry and one at exit. */
        const int P = p->P;
        const int *restrict qout = p->qout;
        const core_plan *cc = &p->core;
        const int nst = cc->nstage;
        /* r7: interior steps stop the inverse one stage early and run
         * st4_last_map_rader (inverse-last radix-4 butterfly + x0 + c + map in
         * ONE pass) -- the separate map_rader_state pass is gone. PLN_NOFM=1
         * restores the r6 shape for A/B. */
        const int fmr = p->rentry && cc->radix[nst - 1] == 4 &&
                        !(getenv("PLN_NOFM") && getenv("PLN_NOFM")[0] == '1');
        if (!p->cfr) {
            p->cfr = amalloc((size_t)P * sizeof(double));
            p->cfi = amalloc((size_t)P * sizeof(double));
            if (!p->cfr || !p->cfi) return;
        }
        double *restrict Sr = p->pre_r, *restrict Si = p->pre_i;
        double *restrict cfr = p->cfr, *restrict cfi = p->cfi;
        for (int b = 0; b < B; ++b) {
            const double *restrict xd = (const double *)(x0 + (size_t)b * L);
            const double *restrict cd = cin + (size_t)b * 2 * L;
            double st0r = xd[0], st0i = xd[1];
            const double cf0r = cd[0], cf0i = cd[1];
            for (int t = 0; t < P; ++t) {
                int k = qout[t];
                Sr[t] = xd[2 * k];  Si[t] = xd[2 * k + 1];
                cfr[t] = cd[2 * k]; cfi[t] = cd[2 * k + 1];
            }
            for (int s = 0; s < m; ++s) {
                double sr, si;
                double *Vr, *Vi;
                const int fuse_step = fmr && s < m - 1;
                if (p->rentry) {
                    st4_first_rev(P / 4, cc->twr[0], cc->twi[0], Sr, Si,
                                  p->s0r, p->s0i);
                    rader_mid(p, fuse_step, &sr, &si, &Vr, &Vi);
                } else {
                    double *Rr, *Ri;
                    double *restrict gr = p->s0r, *restrict gi = p->s0i;
                    gr[0] = Sr[0]; gi[0] = Si[0];
#pragma GCC ivdep
                    for (int q = 1; q < P; ++q) { gr[q] = Sr[P - q]; gi[q] = Si[P - q]; }
                    core_exec_range(cc, p->gs, 0, nst, P, 1,
                                    p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
                    sr = Rr[0]; si = Ri[0];   /* DC of the gathered seq */
                    rader_inv(p, Rr, Ri, 0, &Vr, &Vi);
                }
                const double x0r = st0r, x0i = st0i;
                {
                    double re = x0r + sr + cf0r, im = x0i + si + cf0i;
                    double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                    st0r = re * sc; st0i = im * sc;
                }
                if (fuse_step) {
                    /* Vr/Vi are the inverse's PRE-last planes (swapped) */
                    st4_last_map_rader(P / 4, Vr, Vi, cfr, cfi, x0r, x0i, Sr, Si);
                    continue;
                }
                const double *restrict cvr = Vi, *restrict cvi = Vr;  /* unswap */
                if (s == m - 1) {
                    double *restrict od = (double *)(final_out + (size_t)b * L);
                    od[0] = st0r; od[1] = st0i;
                    for (int t = 0; t < P; ++t) {
                        int k = qout[t];
                        double re = x0r + cvr[t] + cfr[t];
                        double im = x0i + cvi[t] + cfi[t];
                        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                        od[2 * k]     = re * sc;
                        od[2 * k + 1] = im * sc;
                    }
                } else {
                    map_rader_state(P, cvr, cvi, cfr, cfi, x0r, x0i, Sr, Si);
                }
            }
        }
        return;
    }

    /* kind 0: direct. Lane groups first (state + c lane-blocked, whole chain
     * L1/L2-resident per group), then per-vector remainder with split state.
     * For 32/64 the lane path also serves batched chains (transpose amortizes
     * over all m steps); the register codelet chain takes B < 8 and tails.
     * r7: every step fuses map(z + c) into the step's FINAL stage
     * (st{2,4,8,16,64}_last_map) -- the separate map pass is gone. The final
     * per-vector step instead runs the full FFT + map_int_out (vector map with
     * the interleave folded in; one extra pass on 1 of m steps).
     * PLN_NOFM=1 restores the r6 unfused shape for A/B. */
    const int nst0 = p->core.nstage;
    const int rl0 = p->core.radix[nst0 - 1];
    const int nofm = (getenv("PLN_NOFM") && getenv("PLN_NOFM")[0] == '1');
    const int fm = !nofm && !p->dense && nst0 >= 2 &&
                   (rl0 == 2 || rl0 == 4 || rl0 == 5 || rl0 == 8 ||
                    ((rl0 == 16 || rl0 == 64) && (L / rl0) % 8 == 0));
    int b = 0;
    if (p->lanes) {
        const core_plan *lc = p->lcore_on ? &p->lcore : &p->core;
        const int lnst = lc->nstage, lrl = lc->radix[lnst - 1];
        const int fml = !nofm && !p->dense && lnst >= 2 &&
                        (lrl == 2 || lrl == 4 || lrl == 5 || lrl == 8);
        const int sl = LANEV * (L / lrl);
        for (; b + LANEV <= B; b += LANEV) {
            lane_gather((const double *)(x0 + (size_t)b * L), L, p->l0r, p->l0i);
            lane_gather(cin + (size_t)b * 2 * L, L, p->lcr, p->lci);
            double *sr_ = p->l0r, *si_ = p->l0i;
            for (int s = 0; s < m; ++s) {
                double *Rr, *Ri;
                double *ar = sr_, *ai = si_;
                double *br_ = (ar == p->l0r) ? p->l1r : p->l0r;
                double *bi_ = (ai == p->l0i) ? p->l1i : p->l0i;
                if (fml) {
                    core_exec_range(lc, p->lgs, 0, lnst - 1, L, LANEV,
                                    ar, ai, br_, bi_, &Rr, &Ri);
                    double *Wr = (Rr == p->l0r) ? p->l1r : p->l0r;
                    double *Wi = (Ri == p->l0i) ? p->l1i : p->l0i;
                    switch (lrl) {
                    case 2:  st2_last_map(sl, Rr, Ri, p->lcr, p->lci, Wr, Wi); break;
                    case 4:  st4_last_map(sl, Rr, Ri, p->lcr, p->lci, Wr, Wi); break;
                    case 5:  st5_last_map(sl, Rr, Ri, p->lcr, p->lci, Wr, Wi); break;
                    default: st8_last_map(sl, Rr, Ri, p->lcr, p->lci, Wr, Wi); break;
                    }
                    sr_ = Wr; si_ = Wi;
                } else {
                    core_exec_range(lc, p->lgs, 0, lnst, L, LANEV,
                                    ar, ai, br_, bi_, &Rr, &Ri);
                    map_split_n(L * LANEV, Rr, Ri, p->lcr, p->lci);
                    sr_ = Rr; si_ = Ri;
                }
            }
            lane_scatter(sr_, si_, L, (double *)(final_out + (size_t)b * L));
        }
    }
#ifdef __AVX512F__
    if (L == 60 && p->kind == 0) {   /* odd tail of the r7 lane-60 chain */
        if (b < B)
            fft60_chain((const double *)(x0 + (size_t)b * L),
                        cin + (size_t)b * 2 * L,
                        (double *)(final_out + (size_t)b * L), B - b, m);
        return;
    }
    if (p->pw_tf) {           /* 32/64 register-resident chains (d1_pow2 r1) */
        if (b < B) {
            const double *xd = (const double *)(x0 + (size_t)b * L);
            const double *cd = cin + (size_t)b * 2 * L;
            double *od = (double *)(final_out + (size_t)b * L);
            if (L == 32) fft32_chain(p->pw_tf, xd, cd, od, B - b, m);
            else         fft64_chain(p->pw_tf, p->pw_t2, xd, cd, od, B - b, m);
        }
        return;
    }
#endif
    for (; b < B; ++b) {
        const double *restrict xd = (const double *)(x0 + (size_t)b * L);
        const double *restrict cd = cin + (size_t)b * 2 * L;
        /* state ping-pongs through the scratch planes; the map runs in place on
         * the FFT result (same-index elementwise), so no state copy per step.
         * c is pre-split ONCE per transform so the interior map runs shuffle-
         * free through map8_split. */
        double *ar = p->s0r, *ai = p->s0i, *br_ = p->s1r, *bi_ = p->s1i;
        double *restrict pcr = p->pre_r, *restrict pci = p->pre_i;
        {
            double *restrict sr_ = ar, *restrict si_ = ai;
#pragma GCC ivdep
            for (int k = 0; k < L; ++k) {
                sr_[k] = xd[2 * k];  si_[k] = xd[2 * k + 1];
                pcr[k] = cd[2 * k];  pci[k] = cd[2 * k + 1];
            }
        }
        for (int s = 0; s < m; ++s) {
            double *Rr, *Ri;
            if (fm && s < m - 1) {
                /* interior step: final stage + map fused, split -> split */
                core_exec_range(&p->core, p->gs, 0, nst0 - 1, L, 1,
                                ar, ai, br_, bi_, &Rr, &Ri);
                double *Wr = (Rr == p->s0r) ? p->s1r : p->s0r;
                double *Wi = (Ri == p->s0i) ? p->s1i : p->s0i;
                const int sl = L / rl0;
                switch (rl0) {
                case 2:  st2_last_map(sl, Rr, Ri, pcr, pci, Wr, Wi); break;
                case 4:  st4_last_map(sl, Rr, Ri, pcr, pci, Wr, Wi); break;
                case 5:  st5_last_map(sl, Rr, Ri, pcr, pci, Wr, Wi); break;
                case 8:  st8_last_map(sl, Rr, Ri, pcr, pci, Wr, Wi); break;
                case 16: st16_last_map(sl, Rr, Ri, pcr, pci, Wr, Wi); break;
                default: st64_last_map(sl, Rr, Ri, pcr, pci, Wr, Wi); break;
                }
                ar = Wr; ai = Wi;
                br_ = (ar == p->s0r) ? p->s1r : p->s0r;
                bi_ = (ai == p->s0i) ? p->s1i : p->s0i;
                continue;
            }
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
                map_int_out(L, rr, ri, pcr, pci,
                            (double *)(final_out + (size_t)b * L));
            } else {
                if (L >= 32) {
                    map_split_n(L, rr, ri, pcr, pci);
                } else {
                    /* tiny L: the vector map's latency loses (d1_rader saw the
                     * same in their codelets: 0.110 vs 0.099 us at 13) */
#pragma GCC ivdep
                    for (int k = 0; k < L; ++k) {
                        double re = rr[k] + pcr[k], im = ri[k] + pci[k];
                        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                        rr[k] = re * sc;
                        ri[k] = im * sc;
                    }
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
    if (p->lcore_on) core_free(&p->lcore);
    free(p->qin); free(p->qout);
    if (p->arena) {
        /* the double planes are carved from the arena, not owned individually */
        free(p->arena);
    } else {
        free(p->ar); free(p->ai); free(p->br); free(p->bi);
        free(p->s0r); free(p->s0i); free(p->s1r); free(p->s1i);
        free(p->pre_r); free(p->pre_i); free(p->tzr); free(p->tzi);
        free(p->cfr); free(p->cfi);
    }
    free(p->c2r); free(p->c2i);
    free(p->gs);
    free(p->l0r); free(p->l0i); free(p->l1r); free(p->l1i); free(p->lgs);
    free(p->lcr); free(p->lci);
    free(p->sp_tp); free(p->sp_tc); free(p->sp_ts); free(p->sp_ck); free(p->sp_sk);
    free(p->pw_tf); free(p->pw_t2);
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
