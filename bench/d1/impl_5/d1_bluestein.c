/* d1_bluestein: any-L 1D complex FFT.
 *
 * Four paths, chosen at plan time:
 *   - smooth L (factors 2/3/5 only, L >= 4): direct split-complex mixed-radix
 *     Stockham (radix 4/2/3/5), natural-order output, out-of-place ping-pong.
 *   - non-smooth L <= DENSE_MAX: dense O(L^2) DFT (tiny primes: 7, 11, 13, ...).
 *   - Bluestein chirp-Z, single-pass conv while M fits L2. Pad M = smallest
 *     3^a*5^b*2^c >= 2L-1 (L=10007 -> M=20480, not 32768). Chirp phase k^2
 *     reduced mod 2L in 64-bit integers before the trig call. Kernel spectrum
 *     precomputed at plan time with the 1/M of the inverse folded in. Inverse
 *     FFT = forward FFT with re/im swapped.
 *   - Bluestein with an Agarwal-Cooley coprime 2D convolution once the
 *     single-pass M outgrows L2 (M > 32768): M = M1*M2, M1 = pow2 <= 8192,
 *     M2 = odd <= 27 (65537 -> 8192x25, 100003 -> 8192x25). CRT maps the
 *     cyclic conv to an exact 2D cyclic conv with NO inter-axis twiddles;
 *     the M-array is crossed O(1) times instead of once per Stockham stage.
 *
 * All hot loops are split-complex (separate re/im arrays) with restrict
 * pointers: complex multiply needs no in-register swizzle and gcc
 * auto-vectorizes to full-width FMA. All large arrays live in single
 * huge-page blocks with fixed inter-plane skew so L1/L2 set conflicts are
 * deterministic (physical-page luck was a measured 2x run-to-run swing).
 */
#include <complex.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include "../fft1d_api.h"

/* The graded build uses the shared Makefile flags, which leave gcc at its
   256-bit vector preference on Ice Lake; the split-complex loops here have no
   shuffles and win from full zmm width, so ask for it per hot function (a
   whole-file pragma conflicts with the fortified memset wrappers). Both the
   scoring node (Ice Lake 6326) and the dev node (SPR 6448Y) run this ISA.
   Two r3 gcc-11 findings about this attribute:
   - target("prefer-vector-width=512") WITHOUT the arch= part measurably
     degrades the same loops (100003: 2.7 -> 3.7 ms) -- keep the arch=.
   - the arch= form breaks always_inline AVX-512 intrinsics inside the
     function ("target specific option mismatch"), so intrinsic code lives in
     separate NTF-attributed functions (feature-form target merges cleanly). */
#define HOT __attribute__((target("arch=icelake-server,prefer-vector-width=512")))
#define NTF __attribute__((target("avx512f,prefer-vector-width=512")))

#define DENSE_MAX 16
#define MAXSTAGE 40

/* Agarwal-Cooley coprime 2D convolution path (mode 3): split M = M1 * M2 with
   M1 = pow2 part, M2 = odd part, gcd = 1. Used when the conv working set
   outgrows L2: the M2 axis runs in L1-resident 8-wide column tiles and the M1
   rows are contiguous, so each plane crosses memory O(1) times instead of
   once per Stockham stage. */
#define AC_M2MAX 135      /* largest odd axis handled by the stack tiles */

/* plan-time trig in long double: M_PI's rounding is a biased ~2e-16 phase
   error (d1_pow2's r1 lesson); cosl/sinl with an 80-bit pi gives
   correctly-rounded double tables for free at plan time. */
static const long double PI_L = 3.14159265358979323846264338327950288L;

/* ---------------- split-complex mixed-radix Stockham core ---------------- */

typedef struct {
    int n;                   /* transform length = product of radices */
    int nstage;
    int radix[MAXSTAGE];
    double *twr[MAXSTAGE];   /* per-stage twiddles, blocks [j-1][p], j=1..r-1 */
    double *twi[MAXSTAGE];
    double *twstore;         /* single allocation backing all tables */
} core_plan;

static void *amalloc(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
}

/* n same-sized planes in ONE hugepage block, each plane shifted by 32KB+192B
   modulo the 128KB L2-way period. Separate posix_memalign planes land at
   identical offsets mod 4K, and with 4K pages the L2 set index depends on
   random physical page coloring -- measured as a 2x bimodal invocation-to-
   invocation swing at M=147456 (stable within an invocation, set at plan
   time). Huge pages make physical set indexing follow the virtual layout, and
   the fixed skew makes that layout conflict-free.
   r3, on the scoring node: posix_memalign(2MB) blocks there get only PARTIAL
   hugepage grants (glibc reuses fragmented 4K-backed heap), and every block
   starting 2MB-aligned puts all blocks' planes in the SAME L2 sets -- 100003
   stayed bimodal 2.7/3.8 ms across invocations. A fresh mmap gets full grants
   (verified via smaps), the block is pre-faulted so the grant happens before
   timing, and a per-block rotating skew decollides the blocks. */
static size_t blk_skew_ctr;

static void *mmap_block(size_t bytes)
{
    const size_t HP = 2u << 20;
    size_t skew = (blk_skew_ctr++ * (4096 + 64)) & (128 * 1024 - 1);
    size_t tot = (bytes + skew + 64 + HP - 1) & ~(HP - 1);
    char *b = mmap(NULL, tot + HP, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (b == MAP_FAILED) return NULL;
    char *base = (char *)(((size_t)b + HP - 1) & ~(HP - 1));
    madvise(base, tot, MADV_HUGEPAGE);
    memset(base, 0, tot);                /* fault in (and grant) up front */
    char *ret = base + ((skew + 64 + 63) & ~(size_t)63);
    ((size_t *)ret)[-8] = (size_t)0x1dfeb10c5;   /* magic */
    ((size_t *)ret)[-7] = (size_t)b;
    ((size_t *)ret)[-6] = tot + HP;
    return ret;
}

static void big_free(void *p)
{
    if (!p) return;
    if (((size_t *)p)[-8] == (size_t)0x1dfeb10c5)
        munmap((void *)((size_t *)p)[-7], ((size_t *)p)[-6]);
    else
        free(p);
}

static void *big_alloc(size_t bytes) { return mmap_block(bytes); }

static double *planes_alloc(double *out[], int n, size_t len)
{
    const size_t WAY = 128 * 1024;       /* L2 set period (2MB/16w, 1.25MB/10w) */
    size_t bytes = (len * sizeof(double) + WAY - 1) & ~(WAY - 1);
    size_t stride = bytes + 32 * 1024 + 192;
    char *b = mmap_block((size_t)n * stride);
    if (!b) return NULL;
    for (int q = 0; q < n; ++q)
        out[q] = (double *)(b + (size_t)q * stride);
    return (double *)b;
}

/* factor n into stages. Order: one radix-4 first (the Bluestein fused entry
   stage assumes it when 4|n), then the pow2 workhorses, 3s and 5s, and the
   pow2 leftover (4 or 2) LAST -- a radix-2/4 final stage is what the fused
   pruned output stage of the Bluestein path wants (L <= n/2 means only the
   j=0,1 output blocks exist). Returns 0 if not smooth.
   The pow2 workhorse choice depends on the plan FLAVOR, because the fusions
   each flavor relies on constrain the tail:
     CF_DIRECT (mode-1 smooth path): nothing fuses, so just minimize stage
       count with radix-16 (1024 = [4,16,16], 16384 = [4,16,16,16]).
     CF_CONV (mode-2 conv): the pruned chirp exit and the fused middle need a
       2/4 tail, so 16s are only taken while a tail survives
       (20480 = [4,16,16,5,4], 2048 = [4,16,8,4]).
     CF_ROWS (AC row FFTs): stmid8 fuses a trailing radix-8, so when the
       exponent lands exactly on 16^k * 8 the rows drop two passes
       (8192 = [4,16,16,8]); otherwise the old radix-8 schedule stands
       (1024 stays [4,8,8,4] -- st8 is pure-register, st16 bounces L1).
     CF_TILE (AC odd axis): radix-9 on (the L1 tile is pass-count-bound).
   Radix-16 needs a >= 8 to beat the radix-8 schedule at all; BLU_NO16
   disables it for A/B. */
enum { CF_DIRECT = 0, CF_CONV = 1, CF_ROWS = 2, CF_TILE = 3 };

static int core_factor(int n, int *radix, int *nstage, int flavor)
{
    int ns = 0;
    int a = 0;
    while (n % 2 == 0) { ++a; n /= 2; }
    if (a >= 2) { radix[ns++] = 4; a -= 2; }
    else if (a == 1) { radix[ns++] = 2; a = 0; }
    const int r16 = a >= 8 && !getenv("BLU_NO16") && !getenv("AC_R4");
    if (r16 && flavor == CF_DIRECT && a >= 10) {
        /* a = 8 (L = 1024) measured WORSE with 16s on the node (2.22 vs 1.90
           us interleaved): L2-resident, pass count moot, the st16 L1-tile
           bounce loses to st8's registers.  4096/16384 (a >= 10) won 8-9%. */
        while (a >= 4) { radix[ns++] = 16; a -= 4; }
    } else if (r16 && flavor == CF_ROWS && a >= 7 && (a - 3) % 4 == 0) {
        while (a > 3) { radix[ns++] = 16; a -= 4; }
        radix[ns++] = 8; a = 0;                     /* stmid8 fuses this */
    } else if (r16 && flavor == CF_CONV) {
        while (a >= 6) { radix[ns++] = 16; a -= 4; }
        while (a >= 5) { radix[ns++] = 8; a -= 3; }
        while (a >= 3) { radix[ns++] = 4; a -= 2; } /* leaves a in {1,2} */
    }
    if (!getenv("AC_R4"))
        while (a >= 3) { radix[ns++] = 8; a -= 3; }
    while (a >= 4) { radix[ns++] = 4; a -= 2; }
    if (flavor == CF_TILE)
        while (n % 9 == 0) { radix[ns++] = 9; n /= 9; }
    /* radix-15 (Good-Thomas 3x5 in one pass) measured 20-40% SLOWER than
       [3,5] two-pass at M2=135 on the scoring node (65537 B=1 interleaved
       A/B: 2126/2129/1897 vs 1776/1526/1592): 15 live complex = 30 zmm plus
       temps overruns the register file and the spills cost more than the
       saved tile pass (radix-16 dodges this with an L1 tile bounce -- doing
       the same for 15 is untried). Kept opt-in for re-testing. */
    if (flavor == CF_TILE && getenv("AC_R15"))
        while (n % 15 == 0) { radix[ns++] = 15; n /= 15; }
    while (n % 3 == 0) { radix[ns++] = 3; n /= 3; }
    while (n % 5 == 0) { radix[ns++] = 5; n /= 5; }
    if (a == 2) radix[ns++] = 4;
    else if (a == 1) radix[ns++] = 2;
    *nstage = ns;
    return n == 1 && ns <= MAXSTAGE;
}

static int core_init(core_plan *c, int n, int flavor)
{
    c->n = n;
    if (!core_factor(n, c->radix, &c->nstage, flavor)) return 0;
    /* twiddle storage: stage st with current length ncur, radix r needs
       (r-1) * (ncur/r) complex entries */
    size_t tot = 0;
    int ncur = n;
    for (int st = 0; st < c->nstage; ++st) {
        int r = c->radix[st], m = ncur / r;
        tot += (size_t)(r - 1) * m;
        ncur = m;
    }
    c->twstore = big_alloc(2 * tot * sizeof(double));
    if (!c->twstore) return 0;
    double *wr = c->twstore, *wi = c->twstore + tot;
    ncur = n;
    for (int st = 0; st < c->nstage; ++st) {
        int r = c->radix[st], m = ncur / r;
        c->twr[st] = wr; c->twi[st] = wi;
        for (int j = 1; j < r; ++j)
            for (int p = 0; p < m; ++p) {
                long double ph = -2.0L * PI_L
                                 * (long double)(((long)p * j) % ncur)
                                 / (long double)ncur;
                wr[(size_t)(j - 1) * m + p] = (double)cosl(ph);
                wi[(size_t)(j - 1) * m + p] = (double)sinl(ph);
            }
        wr += (size_t)(r - 1) * m; wi += (size_t)(r - 1) * m;
        ncur = m;
    }
    return 1;
}

static void core_free(core_plan *c) { big_free(c->twstore); c->twstore = NULL; }

/* radix-2 stage: y[q+s(2p+j)] = (x[q+s p] +- x[q+s(p+m)]) * w^{p j} */
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

/* radix-4 DFT: X0=t0+t2, X1=t1-i t3, X2=t0-t2, X3=t1+i t3
   with t0=a+c, t1=a-c, t2=b+d, t3=b-d */
HOT static void st4(int m, int s, const double *restrict wr, const double *restrict wi,
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

/* radix-8 as even/odd DFT4 + combine: X_j = E_j + w8^j O_j, X_{j+4} = E_j - w8^j O_j */
HOT static void st8(int m, int s, const double *restrict wr, const double *restrict wi,
                const double *restrict xr, const double *restrict xi,
                double *restrict yr, double *restrict yi)
{
    const double C8 = 0.70710678118654752440;   /* sqrt(2)/2 */
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
            /* even DFT4 on (a0,a2,a4,a6) */
            double t0r = a0r + a4r, t0i = a0i + a4i, t1r = a0r - a4r, t1i = a0i - a4i;
            double t2r = a2r + a6r, t2i = a2i + a6i, t3r = a2r - a6r, t3i = a2i - a6i;
            double e0r = t0r + t2r, e0i = t0i + t2i;
            double e1r = t1r + t3i, e1i = t1i - t3r;
            double e2r = t0r - t2r, e2i = t0i - t2i;
            double e3r = t1r - t3i, e3i = t1i + t3r;
            /* odd DFT4 on (a1,a3,a5,a7) */
            double u0r = a1r + a5r, u0i = a1i + a5i, u1r = a1r - a5r, u1i = a1i - a5i;
            double u2r = a3r + a7r, u2i = a3i + a7i, u3r = a3r - a7r, u3i = a3i - a7i;
            double f0r = u0r + u2r, f0i = u0i + u2i;
            double f1r = u1r + u3i, f1i = u1i - u3r;
            double f2r = u0r - u2r, f2i = u0i - u2i;
            double f3r = u1r - u3i, f3i = u1i + u3r;
            /* rotate odd outputs: f1 *= w8, f2 *= -i, f3 *= w8^3 */
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

/* radix-3: t=b+c, u=b-c; X1 = a - t/2 - i*(s3)*u, X2 = a - t/2 + i*(s3)*u */
HOT static void st3(int m, int s, const double *restrict wr, const double *restrict wi,
                const double *restrict xr, const double *restrict xi,
                double *restrict yr, double *restrict yi)
{
    const double s3 = 0.86602540378443864676;   /* sqrt(3)/2 */
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
            double y1r = mr + s3 * ui, y1i = mi - s3 * ur;   /* a - t/2 - i s3 u */
            double y2r = mr - s3 * ui, y2i = mi + s3 * ur;
            o0r[q] = ar + tr;  o0i[q] = ai + ti;
            o1r[q] = y1r * u1r - y1i * u1i;  o1i[q] = y1r * u1i + y1i * u1r;
            o2r[q] = y2r * u2r - y2i * u2i;  o2i[q] = y2r * u2i + y2i * u2r;
        }
    }
}

/* radix-9 as 3x3 Cooley-Tukey in registers (two radix-3 levels + 4 inner
   W9 twiddles), one memory pass instead of two radix-3 passes. Used only in
   the AC tile plan (M2 = 135/27/45), where the r4 phase split showed the
   entry/exit tile FFTs dominating (65537: entry 745 + exit 610 vs rows 547
   us) -- the shape is d1_rader's r3 "fuse two layers per pass" st16 idea at
   radix 3. */
HOT static void st9(int m, int s, const double *restrict wr, const double *restrict wi,
                const double *restrict xr, const double *restrict xi,
                double *restrict yr, double *restrict yi)
{
    const double s3 = 0.86602540378443864676;               /* sqrt(3)/2 */
    const double W1c = 0.76604444311897803520, W1s = 0.64278760968653932632;
    const double W2c = 0.17364817766693034885, W2s = 0.98480775301220805937;
    const double W4c = -0.93969262078590838405, W4s = 0.34202014332566873304;
    for (int p = 0; p < m; ++p) {
        const double *restrict xp_r = xr + (size_t)s * p;
        const double *restrict xp_i = xi + (size_t)s * p;
        double *restrict op_r = yr + (size_t)s * 9 * p;
        double *restrict op_i = yi + (size_t)s * 9 * p;
        const size_t sm = (size_t)s * m;
        double tw_r[8], tw_i[8];
        for (int j = 1; j < 9; ++j) {
            tw_r[j - 1] = wr[(size_t)(j - 1) * m + p];
            tw_i[j - 1] = wi[(size_t)(j - 1) * m + p];
        }
#pragma GCC ivdep
        for (int q = 0; q < s; ++q) {
            double a0r = xp_r[q],          a0i = xp_i[q];
            double a1r = xp_r[q + sm],     a1i = xp_i[q + sm];
            double a2r = xp_r[q + 2 * sm], a2i = xp_i[q + 2 * sm];
            double a3r = xp_r[q + 3 * sm], a3i = xp_i[q + 3 * sm];
            double a4r = xp_r[q + 4 * sm], a4i = xp_i[q + 4 * sm];
            double a5r = xp_r[q + 5 * sm], a5i = xp_i[q + 5 * sm];
            double a6r = xp_r[q + 6 * sm], a6i = xp_i[q + 6 * sm];
            double a7r = xp_r[q + 7 * sm], a7i = xp_i[q + 7 * sm];
            double a8r = xp_r[q + 8 * sm], a8i = xp_i[q + 8 * sm];
            /* DFT3 over each residue class b: inputs (a_b, a_{b+3}, a_{b+6}) */
            double t0r = a3r + a6r, t0i = a3i + a6i, u0r = a3r - a6r, u0i = a3i - a6i;
            double S00r = a0r + t0r, S00i = a0i + t0i;
            double m0r = a0r - 0.5 * t0r, m0i = a0i - 0.5 * t0i;
            double S01r = m0r + s3 * u0i, S01i = m0i - s3 * u0r;
            double S02r = m0r - s3 * u0i, S02i = m0i + s3 * u0r;
            double t1r = a4r + a7r, t1i = a4i + a7i, u1r = a4r - a7r, u1i = a4i - a7i;
            double S10r = a1r + t1r, S10i = a1i + t1i;
            double m1r = a1r - 0.5 * t1r, m1i = a1i - 0.5 * t1i;
            double S11r = m1r + s3 * u1i, S11i = m1i - s3 * u1r;
            double S12r = m1r - s3 * u1i, S12i = m1i + s3 * u1r;
            double t2r = a5r + a8r, t2i = a5i + a8i, u2r = a5r - a8r, u2i = a5i - a8i;
            double S20r = a2r + t2r, S20i = a2i + t2i;
            double m2r = a2r - 0.5 * t2r, m2i = a2i - 0.5 * t2i;
            double S21r = m2r + s3 * u2i, S21i = m2i - s3 * u2r;
            double S22r = m2r - s3 * u2i, S22i = m2i + s3 * u2r;
            /* inner twiddles: T_bj = S_bj * W9^{bj}, W9^x = (cos, -sin) */
            double T11r = S11r * W1c + S11i * W1s, T11i = S11i * W1c - S11r * W1s;
            double T12r = S12r * W2c + S12i * W2s, T12i = S12i * W2c - S12r * W2s;
            double T21r = S21r * W2c + S21i * W2s, T21i = S21i * W2c - S21r * W2s;
            double T22r = S22r * W4c + S22i * W4s, T22i = S22i * W4c - S22r * W4s;
            /* DFT3 over b for each j: X[j + 3t] */
            double X[9][2];
            { double tr = S10r + S20r, ti = S10i + S20i;
              double ur = S10r - S20r, ui = S10i - S20i;
              double mr = S00r - 0.5 * tr, mi = S00i - 0.5 * ti;
              X[0][0] = S00r + tr;      X[0][1] = S00i + ti;
              X[3][0] = mr + s3 * ui;   X[3][1] = mi - s3 * ur;
              X[6][0] = mr - s3 * ui;   X[6][1] = mi + s3 * ur; }
            { double tr = T11r + T21r, ti = T11i + T21i;
              double ur = T11r - T21r, ui = T11i - T21i;
              double mr = S01r - 0.5 * tr, mi = S01i - 0.5 * ti;
              X[1][0] = S01r + tr;      X[1][1] = S01i + ti;
              X[4][0] = mr + s3 * ui;   X[4][1] = mi - s3 * ur;
              X[7][0] = mr - s3 * ui;   X[7][1] = mi + s3 * ur; }
            { double tr = T12r + T22r, ti = T12i + T22i;
              double ur = T12r - T22r, ui = T12i - T22i;
              double mr = S02r - 0.5 * tr, mi = S02i - 0.5 * ti;
              X[2][0] = S02r + tr;      X[2][1] = S02i + ti;
              X[5][0] = mr + s3 * ui;   X[5][1] = mi - s3 * ur;
              X[8][0] = mr - s3 * ui;   X[8][1] = mi + s3 * ur; }
            op_r[q] = X[0][0];  op_i[q] = X[0][1];
#pragma GCC unroll 8
            for (int k = 1; k < 9; ++k) {
                op_r[q + (size_t)s * k] = X[k][0] * tw_r[k - 1] - X[k][1] * tw_i[k - 1];
                op_i[q + (size_t)s * k] = X[k][0] * tw_i[k - 1] + X[k][1] * tw_r[k - 1];
            }
        }
    }
}

/* radix-15 as Good-Thomas 3x5 PFA in registers: 5 DFT3s + 3 DFT5s and NO
   inner twiddles at all (the coprime CRT index maps replace them), one memory
   pass instead of two.  Input u = (5 n1 + 3 n2) mod 15, output
   k = (10 k1 + 6 k2) mod 15 (exponent check: n k = 5 n1 k1 + 3 n2 k2 mod 15,
   so the 2D transform separates exactly).  MEASURED SLOWER than the [3,5]
   two-pass on the scoring node (see core_factor) -- 15 live complex spill the
   zmm file -- so this ships DISABLED (env AC_R15 re-enables for A/B). */
HOT static void st15(int m, int s, const double *restrict wr, const double *restrict wi,
                 const double *restrict xr, const double *restrict xi,
                 double *restrict yr, double *restrict yi)
{
    const double s3 = 0.86602540378443864676;   /* sqrt(3)/2 */
    const double c1 = 0.30901699437494742410, s1 = 0.95105651629515357212;
    const double c2 = -0.80901699437494742410, s2 = 0.58778525229247312917;
    for (int p = 0; p < m; ++p) {
        const double *restrict xp_r = xr + (size_t)s * p;
        const double *restrict xp_i = xi + (size_t)s * p;
        double *restrict op_r = yr + (size_t)s * 15 * p;
        double *restrict op_i = yi + (size_t)s * 15 * p;
        const size_t sm = (size_t)s * m;
        double tw_r[14], tw_i[14];
        if (m > 1)
            for (int j = 1; j < 15; ++j) {
                tw_r[j - 1] = wr[(size_t)(j - 1) * m + p];
                tw_i[j - 1] = wi[(size_t)(j - 1) * m + p];
            }
#pragma GCC ivdep
        for (int q = 0; q < s; ++q) {
            double A[3][5][2];      /* A[k1][n2] = DFT3 over n1 of x[(5n1+3n2)%15] */
#define DFT3COL(N2, U0, U1, U2) do { \
            double ar = xp_r[q + (U0) * sm], ai = xp_i[q + (U0) * sm]; \
            double br = xp_r[q + (U1) * sm], bi = xp_i[q + (U1) * sm]; \
            double cr = xp_r[q + (U2) * sm], ci = xp_i[q + (U2) * sm]; \
            double tr = br + cr, ti = bi + ci; \
            double ur = br - cr, ui = bi - ci; \
            double mr = ar - 0.5 * tr, mi = ai - 0.5 * ti; \
            A[0][N2][0] = ar + tr;       A[0][N2][1] = ai + ti; \
            A[1][N2][0] = mr + s3 * ui;  A[1][N2][1] = mi - s3 * ur; \
            A[2][N2][0] = mr - s3 * ui;  A[2][N2][1] = mi + s3 * ur; \
            } while (0)
            DFT3COL(0, 0, 5, 10);
            DFT3COL(1, 3, 8, 13);
            DFT3COL(2, 6, 11, 1);
            DFT3COL(3, 9, 14, 4);
            DFT3COL(4, 12, 2, 7);
#undef DFT3COL
            double X[15][2];        /* DFT5 over n2 at each k1; k = (10k1+6k2)%15 */
#define DFT5ROW(K1, O0, O1, O2, O3, O4) do { \
            double ar = A[K1][0][0], ai = A[K1][0][1]; \
            double br = A[K1][1][0], bi = A[K1][1][1]; \
            double cr = A[K1][2][0], ci = A[K1][2][1]; \
            double dr = A[K1][3][0], di = A[K1][3][1]; \
            double er = A[K1][4][0], ei = A[K1][4][1]; \
            double t1r = br + er, t1i = bi + ei; \
            double t2r = cr + dr, t2i = ci + di; \
            double v1r = br - er, v1i = bi - ei; \
            double v2r = cr - dr, v2i = ci - di; \
            double m1r = ar + c1 * t1r + c2 * t2r, m1i = ai + c1 * t1i + c2 * t2i; \
            double m2r = ar + c2 * t1r + c1 * t2r, m2i = ai + c2 * t1i + c1 * t2i; \
            double n1r = s1 * v1r + s2 * v2r, n1i = s1 * v1i + s2 * v2i; \
            double n2r = s2 * v1r - s1 * v2r, n2i = s2 * v1i - s1 * v2i; \
            X[O0][0] = ar + t1r + t2r;  X[O0][1] = ai + t1i + t2i; \
            X[O1][0] = m1r + n1i;  X[O1][1] = m1i - n1r; \
            X[O2][0] = m2r + n2i;  X[O2][1] = m2i - n2r; \
            X[O3][0] = m2r - n2i;  X[O3][1] = m2i + n2r; \
            X[O4][0] = m1r - n1i;  X[O4][1] = m1i + n1r; \
            } while (0)
            DFT5ROW(0, 0, 6, 12, 3, 9);
            DFT5ROW(1, 10, 1, 7, 13, 4);
            DFT5ROW(2, 5, 11, 2, 8, 14);
#undef DFT5ROW
            op_r[q] = X[0][0];  op_i[q] = X[0][1];
            if (m == 1) {
#pragma GCC unroll 14
                for (int k = 1; k < 15; ++k) {
                    op_r[q + (size_t)s * k] = X[k][0];
                    op_i[q + (size_t)s * k] = X[k][1];
                }
            } else {
#pragma GCC unroll 14
                for (int k = 1; k < 15; ++k) {
                    op_r[q + (size_t)s * k] = X[k][0] * tw_r[k - 1] - X[k][1] * tw_i[k - 1];
                    op_i[q + (size_t)s * k] = X[k][0] * tw_i[k - 1] + X[k][1] * tw_r[k - 1];
                }
            }
        }
    }
}

/* radix-5 with t1=b+e, t2=c+d, u1=b-e, u2=c-d:
   X1/X4 = a + c1 t1 + c2 t2 -/+ i(s1 u1 + s2 u2)
   X2/X3 = a + c2 t1 + c1 t2 -/+ i(s2 u1 - s1 u2) */
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
            double y1r = m1r + n1i, y1i = m1i - n1r;   /* m1 - i n1 */
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

/* radix-8 at s=4, two p-groups per zmm (lanes 0-3 = group p, 4-7 = group
   p+1): the q-loop at s=4 otherwise runs half-width with per-p twiddle
   overhead. d1_rader's r3 st16_s4 trick (their st3_s4 generalized), applied
   to my radix-8: inputs for the pair are contiguous 8-double runs, twiddles
   are pair-broadcast [w(p) x4 | w(p+1) x4], outputs recombine to contiguous
   zmm stores via 128-bit-lane shuffles. NTF because always_inline intrinsics
   refuse to inline into arch=-form HOT callers (r3 lesson). */
NTF static void st8_s4(int m, const double *restrict wr, const double *restrict wi,
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

/* radix-16 as two fused radix-4 layers through a 2 KB L1 tile -- ported
   near-verbatim from d1_rader (their r3 st16 / st16_s4; the same kernels
   carry their 65536 conv at 915 us).  The tile is the fix my radix-15
   attempt lacked: keeping all 16 complex lane-vectors live is the whole zmm
   file and gcc spills; bouncing layer 1 through an aligned stack tile keeps
   the live set at 8 vectors.  One radix-16 pass replaces two radix-8-ish
   passes wherever the pow2 run is long enough (rows at M1=8192, the direct
   pow2 path, the mode-2 convs). */
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

/* s == 4 path: two p-groups per zmm, twiddles pair-broadcast, outputs as
   256-bit halves 64 apart (d1_rader's st16_s4, their st3_s4 shape) */
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

/* Stages [st0, st1) of the forward DFT, entered at sub-length n / stride s,
   ping-ponging between the two pairs. Result pointers land in outr/outi.
   TWO instantiations: the stage kernels get INLINED here, and when they do,
   the inliner silently drops their per-function prefer-vector-width=512 (the
   caller's target wins -- found by objdump in r3: zero zmm in the old single
   copy despite HOT on every st*). zmm wins where the pass data is L2-resident
   (the AC row/tile FFTs: -21% instructions, -25% time at 65537); ymm wins on
   the L3-bound single-pass conv (10007: zmm cost 15%, 210 -> 243 us on the
   node). So: _z for mode 3, _y for modes 1/2. */
#define CORE_EXEC_BODY \
{ \
    double *xr = ar, *xi = ai, *yr = br, *yi = bi, *t; \
    for (int st = st0; st < st1; ++st) { \
        int r = c->radix[st], m = n / r; \
        switch (r) { \
        case 2: st2(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break; \
        case 3: st3(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break; \
        case 4: st4(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break; \
        case 8: if (s == 4 && !(m & 1)) \
                    st8_s4(m, c->twr[st], c->twi[st], xr, xi, yr, yi); \
                else st8(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); \
                break; \
        case 9: st9(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break; \
        case 15: st15(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break; \
        case 16: st16(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break; \
        default: st5(m, s, c->twr[st], c->twi[st], xr, xi, yr, yi); break; \
        } \
        n = m; s *= r; \
        t = xr; xr = yr; yr = t; \
        t = xi; xi = yi; yi = t; \
    } \
    *outr = xr; *outi = xi; \
}

HOT static void core_exec_range_z(const core_plan *c, int st0, int st1, int n, int s,
                            double *ar, double *ai, double *br, double *bi,
                            double **outr, double **outi)
CORE_EXEC_BODY

static void core_exec_range_y(const core_plan *c, int st0, int st1, int n, int s,
                            double *ar, double *ai, double *br, double *bi,
                            double **outr, double **outi)
CORE_EXEC_BODY

static void core_exec(const core_plan *c, double *ar, double *ai,
                      double *br, double *bi, double **outr, double **outi)
{
    core_exec_range_y(c, 0, c->nstage, c->n, 1, ar, ai, br, bi, outr, outi);
}

/* ------------- Bluestein-fused entry / exit stages (radix-4/2) -------------
 *
 * The padded input has M-L trailing zeros and L <= M/2, so in the radix-4
 * entry stage (m = M/4, L > m always for our minimal pads) only the u=0,1
 * input blocks are ever nonzero: the butterfly degenerates to
 *   X0 = a+b, X1 = a-ib, X2 = a-b, X3 = a+ib   (b = 0 above p = L-m).
 * The input chirp multiply and (for the plain call) the deinterleave are
 * folded into the same pass, so the padded buffer is never materialized.
 *
 * The inverse runs as forward-on-swapped-planes; its entry stage folds in the
 * kernel-spectrum multiply and the swap, and its exit stage (radix 2 or 4,
 * arranged by core_factor) only needs the j*s < L output blocks, folding in
 * the output chirp and the interleave. */

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

/* inverse entry stage: pointwise multiply by the kernel spectrum, swap the
   planes (inverse-as-forward trick), and do the radix-4 stage-0 butterfly */
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
        /* a..d are the swapped (imag,real) of R*Bhat at t = p + m*u */
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

/* exit stages: planes are swapped (plane_r = W_imag, plane_i = W_real);
   only outputs k = q + s*j < L exist, twiddles at the last stage are 1.
   The output chirp a[k] and the interleave (or the split store for the
   chain) are folded in. */
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
    for (int q = 0; q < s; ++q) {           /* j = 0: X0 = a+b+c+d; s < L always */
        double w_i = xr[q] + xr[q + s] + xr[q + 2 * s] + xr[q + 3 * s];
        double w_r = xi[q] + xi[q + s] + xi[q + 2 * s] + xi[q + 3 * s];
        yd[2 * q]     = w_r * car[q] - w_i * cai[q];
        yd[2 * q + 1] = w_r * cai[q] + w_i * car[q];
    }
    const int n1 = L - s;                   /* j = 1: X1 = (a-c) - i(b-d) */
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

/* NT-streamed variant of the radix-2 fused exit, for batched mode-2 cells
   whose in+out working set is DRAM-resident (10007 x B=64 = 19.5 MB): y is
   written once and never re-read, so streaming stores skip the RFO read of
   the output -- a third of the exit traffic.  The mode-3 scatter can only NT
   the 1-in-4 batch members whose y base is 64B-aligned, but THIS output is
   contiguous, so alignment is bought by peeling: the vector loop starts at
   the first complex whose store address is 64B-aligned (head/tail complexes
   go through regular scalar stores) and every 16-double NT store after that
   is aligned by construction.  All source reads are loadu at arbitrary
   offset.  Caller issues the sfence. */
NTF static void st2_last_chirp_nt(int s, const double *restrict xr, const double *restrict xi,
        const double *restrict car, const double *restrict cai,
        double *restrict yd, int L)
{
    const __m512i ixlo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i ixhi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    const int pad = (int)(((64 - ((uintptr_t)yd & 63)) & 63) >> 3);  /* doubles */
    const int h = pad >> 1;              /* head complexes (pad is even: yd is 16B-aligned) */
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
}

/* radix-4 twin (the CF_CONV radix-16 schedules end [.., 4]): two contiguous
   output segments, j = 0 at yd[0..2s) and j = 1 at yd[2s..2L); s = M/4 is a
   multiple of 4 so both segments share the same head pad. */
NTF static void st4_last_chirp_nt(int s, const double *restrict xr, const double *restrict xi,
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
}

/* same idea for the direct path's final interleave (mode 1: 4096 x B=256 and
   16384 x B=64 stream 33.5 MB past L3; d1_pow2/d1_twiddle measured -17% from
   NT last-stage stores there) */
NTF static void interleave_nt(const double *restrict rr, const double *restrict ri,
                              double *restrict yd, int L)
{
    const __m512i ixlo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i ixhi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    const int pad = (int)(((64 - ((uintptr_t)yd & 63)) & 63) >> 3);
    const int h = pad >> 1;
    int k = 0;
    for (; k < h && k < L; ++k) { yd[2 * k] = rr[k]; yd[2 * k + 1] = ri[k]; }
    for (; k + 8 <= L; k += 8) {
        __m512d re = _mm512_loadu_pd(rr + k), im = _mm512_loadu_pd(ri + k);
        _mm512_stream_pd(yd + 2 * k,     _mm512_permutex2var_pd(re, ixlo, im));
        _mm512_stream_pd(yd + 2 * k + 8, _mm512_permutex2var_pd(re, ixhi, im));
    }
    for (; k < L; ++k) { yd[2 * k] = rr[k]; yd[2 * k + 1] = ri[k]; }
}

/* split-output exit stages for the chain: z = a*W is the actual FFT output */
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

/* --------- fused middle: forward-last x kernel-multiply x inverse-entry ---
 *
 * The LAST forward Stockham stage has m = 1, so its twiddles are all 1, and
 * its output block structure (k = q + j*s) lines up exactly with the radix-4
 * inverse entry's read pattern (p + u*M/4).  So the forward-last butterfly,
 * the kernel-spectrum multiply with the re/im swap (inverse-as-forward), and
 * the inverse stage-0 radix-4 butterfly all fuse into ONE pass: one full
 * write+read round trip over the M-plane disappears per transform (mode 2)
 * or per row (mode 3).  d1_rader's r3 record listed exactly this fusion as
 * their untried next item; the algebra here is st4_first_bhat's with the
 * spectrum produced in registers instead of read back from a plane.
 * Two radix cases for the forward-last stage (core_factor puts the pow2
 * leftover last): rl=4 (s = M/4, one radix-4 butterfly) and rl=2 (s = M/2,
 * two radix-2 butterflies at q = p and q = p + M/4).  Both read the same
 * four indices p + u*M/4.  Instantiated _z (zmm, L2-resident AC rows) and
 * _y (ymm, L3-bound single-pass conv) like CORE_EXEC_BODY, and for the same
 * r3 reason: the inliner drops a callee's vector-width preference. */
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
        /* kernel multiply, swapped planes: A = swap(R * B) */ \
        double A0r = Rr[p] * B0i + Ri[p] * B0r,    A0i = Rr[p] * B0r - Ri[p] * B0i; \
        double A1r = Rr[t1_] * B1i + Ri[t1_] * B1r, A1i = Rr[t1_] * B1r - Ri[t1_] * B1i; \
        double A2r = Rr[t2_] * B2i + Ri[t2_] * B2r, A2i = Rr[t2_] * B2r - Ri[t2_] * B2i; \
        double A3r = Rr[t3_] * B3i + Ri[t3_] * B3r, A3i = Rr[t3_] * B3r - Ri[t3_] * B3i; \
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
        const double *restrict Rr, const double *restrict Ri, \
        double *restrict yr, double *restrict yi

HOT static void stmid4_z(STMID_ARGS) STMID_BODY(STMID_B4)
HOT static void stmid2_z(STMID_ARGS) STMID_BODY(STMID_B2)
static void stmid4_y(STMID_ARGS) STMID_BODY(STMID_B4)
static void stmid2_y(STMID_ARGS) STMID_BODY(STMID_B2)

/* rl=8 fused middle, for the CF_ROWS [4,16,16,8] schedule: the forward-last
   radix-8 butterfly at q produces spectrum bins k = q + j*(M/8); its even-j
   outputs are exactly the radix-4 inverse entry's reads at p = q, its odd-j
   outputs those at p = q + M/8.  So one radix-8 butterfly + 8 kernel
   multiplies (with the inverse's plane swap) + TWO twiddled radix-4 inverse
   entry butterflies run per point, in one pass -- the spectrum never touches
   a plane, same as stmid2/4 but one radix level up.  mm8 = M/8; the stage-0
   twiddle blocks are m0 = M/4 = 2*mm8 long and get indexed at p and p+mm8. */
HOT static void stmid8_z(int mm8,
        const double *restrict wr, const double *restrict wi,
        const double *restrict xr, const double *restrict xi,
        const double *restrict Rr, const double *restrict Ri,
        double *restrict yr, double *restrict yi)
{
    const int m0 = 2 * mm8;
    const double *restrict w1r = wr,          *restrict w1i = wi;
    const double *restrict w2r = wr + m0,     *restrict w2i = wi + m0;
    const double *restrict w3r = wr + 2 * m0, *restrict w3i = wi + 2 * m0;
    const double C8 = 0.70710678118654752440;
#pragma GCC ivdep
    for (int p = 0; p < mm8; ++p) {
        /* forward-last radix-8, m = 1 so twiddles are all 1 */
        double a0r = xr[p],           a0i = xi[p];
        double a1r = xr[p + mm8],     a1i = xi[p + mm8];
        double a2r = xr[p + 2 * mm8], a2i = xi[p + 2 * mm8];
        double a3r = xr[p + 3 * mm8], a3i = xi[p + 3 * mm8];
        double a4r = xr[p + 4 * mm8], a4i = xi[p + 4 * mm8];
        double a5r = xr[p + 5 * mm8], a5i = xi[p + 5 * mm8];
        double a6r = xr[p + 6 * mm8], a6i = xi[p + 6 * mm8];
        double a7r = xr[p + 7 * mm8], a7i = xi[p + 7 * mm8];
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
        double B0r = e0r + f0r, B0i = e0i + f0i;
        double B4r = e0r - f0r, B4i = e0i - f0i;
        double B1r = e1r + g1r, B1i = e1i + g1i;
        double B5r = e1r - g1r, B5i = e1i - g1i;
        double B2r = e2r + g2r, B2i = e2i + g2i;
        double B6r = e2r - g2r, B6i = e2i - g2i;
        double B3r = e3r + g3r, B3i = e3i + g3i;
        double B7r = e3r - g3r, B7i = e3i - g3i;
        /* kernel multiply at k = p + j*mm8, swapped planes: A = swap(B * R) */
#define KM(J) \
        double A##J##r = B##J##r * Ri[p + J * mm8] + B##J##i * Rr[p + J * mm8]; \
        double A##J##i = B##J##r * Rr[p + J * mm8] - B##J##i * Ri[p + J * mm8];
        KM(0) KM(1) KM(2) KM(3) KM(4) KM(5) KM(6) KM(7)
#undef KM
        /* inverse radix-4 entry at p' = p (even j) and p' = p + mm8 (odd j) */
#define INV4(pp, I0r, I0i, I1r, I1i, I2r, I2i, I3r, I3i) do { \
        double v0r = I0r + I2r, v0i = I0i + I2i, v1r = I0r - I2r, v1i = I0i - I2i; \
        double v2r = I1r + I3r, v2i = I1i + I3i, v3r = I1r - I3r, v3i = I1i - I3i; \
        double y1r = v1r + v3i, y1i = v1i - v3r; \
        double y2r = v0r - v2r, y2i = v0i - v2i; \
        double y3r = v1r - v3i, y3i = v1i + v3r; \
        yr[4 * (pp)] = v0r + v2r;  yi[4 * (pp)] = v0i + v2i; \
        yr[4 * (pp) + 1] = y1r * w1r[pp] - y1i * w1i[pp]; \
        yi[4 * (pp) + 1] = y1r * w1i[pp] + y1i * w1r[pp]; \
        yr[4 * (pp) + 2] = y2r * w2r[pp] - y2i * w2i[pp]; \
        yi[4 * (pp) + 2] = y2r * w2i[pp] + y2i * w2r[pp]; \
        yr[4 * (pp) + 3] = y3r * w3r[pp] - y3i * w3i[pp]; \
        yi[4 * (pp) + 3] = y3r * w3i[pp] + y3i * w3r[pp]; \
        } while (0)
        INV4(p,       A0r, A0i, A2r, A2i, A4r, A4i, A6r, A6i);
        INV4(p + mm8, A1r, A1i, A3r, A3i, A5r, A5i, A7r, A7i);
#undef INV4
    }
}

/* ------------------------------- plan ------------------------------------ */

struct fft1d_plan {
    int L, batch;
    int mode;                /* 0 dense, 1 direct smooth, 2 bluestein, 3 AC-2D bluestein */
    int M;                   /* convolution length (modes 2/3) */
    int fuse_last;           /* 2/4: fused pruned exit stage radix; 0: none */
    int fuse_mid;            /* 2/4: fused fwd-last x kernel x inv-entry; 0: none */
    int nt_out;              /* modes 1/2: NT-stream the final interleaved store
                                (batched working set past L3; y never re-read) */
    core_plan core;          /* size M (mode 2) or L (mode 1) */
    double *ar, *ai;         /* chirp a[k] = exp(-i pi k^2 / L), length L */
    double *br, *bi;         /* DFT(kernel)/M, length M */
    double *s0r, *s0i, *s1r, *s1i;   /* scratch, length M (or L direct, M1 AC) */
    double _Complex *w;      /* dense DFT matrix (mode 0) */
    double _Complex *chain_y;/* chain scratch, L*batch (lazily sized) */
    double *pre_r, *pre_i;   /* chain: chirp-premultiplied state, L*batch */
    double *tzr, *tzi;       /* chain: one transform's output, L */
    /* mode 3: Agarwal-Cooley M = M1*M2, gcd(M1,M2)=1; layout [M2 rows][RS] */
    int M1, M2, RS;
    core_plan c1, c2;        /* row plan (pow2 M1), column plan (odd M2) */
    double *gr, *gi;         /* 2D work planes, M2*RS each */
    double *bakr, *baki;     /* kernel spectrum / M, same layout */
    double *ctr, *cti;       /* rotation fix C[k][r]=e^{-2pi i rk/M2}, [M2][M2+8] */
    int *acrow;              /* run t -> tile row (t*M1) mod M2 */
    double *blkA, *blkB, *blkC;  /* skewed backing blocks (planes_alloc) */
    double *blk_pre, *blk_tz;    /* chain scratch backing blocks */
};

static int is_smooth(int n)
{
    while (n % 2 == 0) n /= 2;
    while (n % 3 == 0) n /= 3;
    while (n % 5 == 0) n /= 5;
    return n == 1;
}

static int choose_M(int L)
{
    /* smallest 3^a * 5^b * 2^c >= 2L-1 with 4 | M (the fused entry stage is
       radix-4), a,b <= 3 */
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

/* smallest M = M2 * M1 with M2 in {3,5,9,15,25,27}, M1 pow2 in [512, 8192],
   M >= 2L-1. The M1 cap keeps the per-row working set (4 ping-pong planes +
   row + kernel row ~ 64 B/point) comfortably inside a 1.25 MB L2: measured at
   L=65537, M=204800 (8192x25) runs a deterministic 1790 us on wallaby while
   the smaller M=147456 (16384x9) swings 1570-2440 with physical-page luck.
   Returns 0 if no split fits. */
static int ac_choose_M(int L)
{
    /* dev override: AC_SPLIT="M1xM2" forces the split for A/B on the node */
    const char *ov = getenv("AC_SPLIT");
    if (ov) {
        int m1 = 0, m2 = 0;
        if (sscanf(ov, "%dx%d", &m1, &m2) == 2 &&
            (long)m1 * m2 >= 2L * (long)L - 1)
            return m1 * m2;
    }
    /* minimize total points M: measured on the scoring node (Ice Lake 6326),
       AC cost is ~linear in M across M1 = 1024..8192 at fixed M >= 1.3e5
       (65537: 8192x25 = 4360 us, 2048x75 = 3353, 1024x135 = 2303 -- and the
       last is also the smallest M). The r2 "M1 <= 8192" lesson still holds:
       16384x9 = 147456 was slower than 8192x25 = 204800 despite fewer points. */
    static const int odds[10] = {3, 5, 9, 15, 25, 27, 45, 75, 81, 135};
    long need = 2L * (long)L - 1;
    long best = -1;
    for (int i = 0; i < 10; ++i)
        for (int m1 = 512; m1 <= 8192; m1 *= 2) {
            long M = (long)odds[i] * m1;
            if (M >= need && (best < 0 || M < best)) best = M;
        }
    return best < 0 ? 0 : (int)best;
}

/* ---------------- mode 3: Agarwal-Cooley coprime 2D convolution -----------
 *
 * CRT: n <-> (n1, n2) = (n mod M1, n mod M2), gcd(M1,M2)=1, maps a length-M
 * cyclic convolution to an M1 x M2 2D cyclic convolution EXACTLY -- no
 * inter-axis twiddles (unlike four-step), so the only new per-point work is a
 * tiny rotation fix (below). Layout: row n2 (M2 rows, odd, <= AC_M2MAX),
 * column n1 (M1 = pow2, contiguous, row stride RS = M1+8 to stagger L1 sets).
 *
 * Runs of 8 consecutive n (n = c + t*M1 + j, j = 0..7) sit in 8 consecutive
 * COLUMNS but on a diagonal of rows: n2 = (c + t*M1 + j) mod M2. Storing the
 * tile by the diagonal index rho = (n2 - n1) mod M2 = (t*M1) mod M2 makes
 * every gather/scatter run contiguous and 8-wide; the price is that each tile
 * column j is the true column cyclically shifted by r_j = (c+j) mod M2, so
 * after the tile DFT the true spectrum is U[k]*C[k][r_j] with
 * C[k][r] = e^{-2pi i r k / M2} (an M2 x M2 table, L1-resident), and the
 * inverse pre-multiplies by conj(C) to come back out rotated. One extra
 * complex multiply per point per direction buys fully streaming access.
 *
 * Pipeline per transform: fused entry (chirp + CRT gather + M2-axis tile FFT,
 * one write pass) -> per-row: pow2 forward FFT, kernel-multiply fused into the
 * inverse entry (st4_first_bhat, plane swap), inverse back into the row (one
 * row round-trip through L2) -> fused exit (M2-axis inverse tile FFT + output
 * chirp + CRT scatter, one read pass). The M-array is crossed O(1) times
 * instead of once per Stockham stage. */

/* fused entry, interleaved-x source: chirp multiply + gather + M2-axis FFT */
HOT static void ac_cols_fwd_x(const fft1d_plan *p, const double *restrict xd)
{
    const int M1 = p->M1, M2 = p->M2, RS = p->RS, L = p->L;
    const core_plan *c2 = &p->c2;
    const int TW = M2 + 8;
    const double *restrict car = p->ar, *restrict cai = p->ai;
    for (int c = 0; c < M1; c += 8) {
        double t0r[AC_M2MAX * 8] __attribute__((aligned(64)));
        double t0i[AC_M2MAX * 8] __attribute__((aligned(64)));
        double t1r[AC_M2MAX * 8] __attribute__((aligned(64)));
        double t1i[AC_M2MAX * 8] __attribute__((aligned(64)));
        const int cm = c % M2;
        for (int t = 0; t < M2; ++t) {
            double *restrict tr = t0r + 8 * p->acrow[t];
            double *restrict ti = t0i + 8 * p->acrow[t];
            const int nlo = c + t * M1;
            int cnt = L - nlo;
            if (cnt > 8) cnt = 8; else if (cnt < 0) cnt = 0;
            const double *restrict xs = xd + 2 * (size_t)nlo;
            const double *restrict cr = car + nlo;
            const double *restrict ci = cai + nlo;
            __builtin_prefetch(xs + 16, 0, 2);      /* next tile's gather */
            __builtin_prefetch(xs + 24, 0, 2);
            __builtin_prefetch(cr + 8, 0, 2);
            __builtin_prefetch(ci + 8, 0, 2);
#pragma GCC ivdep
            for (int j = 0; j < cnt; ++j) {
                double xr = xs[2 * j], xi = xs[2 * j + 1];
                tr[j] = xr * cr[j] - xi * ci[j];
                ti[j] = xr * ci[j] + xi * cr[j];
            }
            for (int j = cnt; j < 8; ++j) { tr[j] = 0.0; ti[j] = 0.0; }
        }
        double *Ur, *Ui;
        core_exec_range_z(c2, 0, c2->nstage, M2, 8, t0r, t0i, t1r, t1i, &Ur, &Ui);
        for (int k = 0; k < M2; ++k) {
            const double *restrict ur = Ur + 8 * k, *restrict ui = Ui + 8 * k;
            const double *restrict qr = p->ctr + (size_t)k * TW + cm;
            const double *restrict qi = p->cti + (size_t)k * TW + cm;
            double *restrict or_ = p->gr + (size_t)k * RS + c;
            double *restrict oi_ = p->gi + (size_t)k * RS + c;
            __builtin_prefetch(or_ + 8, 1, 2);      /* next tile's scatter */
            __builtin_prefetch(oi_ + 8, 1, 2);
#pragma GCC ivdep
            for (int j = 0; j < 8; ++j) {
                or_[j] = ur[j] * qr[j] - ui[j] * qi[j];
                oi_[j] = ur[j] * qi[j] + ui[j] * qr[j];
            }
        }
    }
}

/* fused entry, split source of length srclen (chain state already chirped;
   plan-time kernel sequence): no chirp, otherwise identical */
HOT static void ac_cols_fwd_split(const fft1d_plan *p, const double *restrict sr,
                                  const double *restrict si, int srclen)
{
    const int M1 = p->M1, M2 = p->M2, RS = p->RS;
    const core_plan *c2 = &p->c2;
    const int TW = M2 + 8;
    for (int c = 0; c < M1; c += 8) {
        double t0r[AC_M2MAX * 8] __attribute__((aligned(64)));
        double t0i[AC_M2MAX * 8] __attribute__((aligned(64)));
        double t1r[AC_M2MAX * 8] __attribute__((aligned(64)));
        double t1i[AC_M2MAX * 8] __attribute__((aligned(64)));
        const int cm = c % M2;
        for (int t = 0; t < M2; ++t) {
            double *restrict tr = t0r + 8 * p->acrow[t];
            double *restrict ti = t0i + 8 * p->acrow[t];
            const int nlo = c + t * M1;
            int cnt = srclen - nlo;
            if (cnt > 8) cnt = 8; else if (cnt < 0) cnt = 0;
            const double *restrict vr = sr + nlo, *restrict vi = si + nlo;
            __builtin_prefetch(vr + 8, 0, 2);
            __builtin_prefetch(vi + 8, 0, 2);
#pragma GCC ivdep
            for (int j = 0; j < cnt; ++j) { tr[j] = vr[j]; ti[j] = vi[j]; }
            for (int j = cnt; j < 8; ++j) { tr[j] = 0.0; ti[j] = 0.0; }
        }
        double *Ur, *Ui;
        core_exec_range_z(c2, 0, c2->nstage, M2, 8, t0r, t0i, t1r, t1i, &Ur, &Ui);
        for (int k = 0; k < M2; ++k) {
            const double *restrict ur = Ur + 8 * k, *restrict ui = Ui + 8 * k;
            const double *restrict qr = p->ctr + (size_t)k * TW + cm;
            const double *restrict qi = p->cti + (size_t)k * TW + cm;
            double *restrict or_ = p->gr + (size_t)k * RS + c;
            double *restrict oi_ = p->gi + (size_t)k * RS + c;
            __builtin_prefetch(or_ + 8, 1, 2);
            __builtin_prefetch(oi_ + 8, 1, 2);
#pragma GCC ivdep
            for (int j = 0; j < 8; ++j) {
                or_[j] = ur[j] * qr[j] - ui[j] * qi[j];
                oi_[j] = ur[j] * qi[j] + ui[j] * qr[j];
            }
        }
    }
}

/* per-row middle: forward M1-FFT, kernel multiply fused into the inverse
   entry (plane swap), inverse M1-FFT back into the row. Each row makes one
   round trip; all ping-pong scratch is M1-sized (L2-resident). */
HOT static void ac_rows_middle(fft1d_plan *p)
{
    const core_plan *c1 = &p->c1;
    const int nst = c1->nstage, m0 = p->M1 / 4;
    const int rlm = c1->radix[nst - 1];      /* pow2 M1: 2/4, or 8 (CF_ROWS 16s) */
    for (int k2 = 0; k2 < p->M2; ++k2) {
        double *rr = p->gr + (size_t)k2 * p->RS;
        double *ri = p->gi + (size_t)k2 * p->RS;
        st4(m0, 1, c1->twr[0], c1->twi[0], rr, ri, p->s0r, p->s0i);
        double *Rr, *Ri;
        core_exec_range_z(c1, 1, nst - 1, m0, 4, p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
        double *dr = (Rr == p->s0r) ? p->s1r : p->s0r;
        double *di = (Rr == p->s0r) ? p->s1i : p->s0i;
        if (rlm == 8)
            stmid8_z(p->M1 / 8, c1->twr[0], c1->twi[0], Rr, Ri,
                     p->bakr + (size_t)k2 * p->RS, p->baki + (size_t)k2 * p->RS,
                     dr, di);
        else if (rlm == 4)
            stmid4_z(m0, c1->twr[0], c1->twi[0], Rr, Ri,
                     p->bakr + (size_t)k2 * p->RS, p->baki + (size_t)k2 * p->RS,
                     dr, di);
        else
            stmid2_z(m0, c1->twr[0], c1->twi[0], Rr, Ri,
                     p->bakr + (size_t)k2 * p->RS, p->baki + (size_t)k2 * p->RS,
                     dr, di);
        double *er = (dr == p->s0r) ? p->s1r : p->s0r;
        double *ei = (dr == p->s0r) ? p->s1i : p->s0i;
        double *Vr, *Vi;
        core_exec_range_z(c1, 1, nst - 1, m0, 4, dr, di, er, ei, &Vr, &Vi);
        const int rl = c1->radix[nst - 1], sl = p->M1 / rl;
        switch (rl) {
        case 2: st2(1, sl, c1->twr[nst - 1], c1->twi[nst - 1], Vr, Vi, rr, ri); break;
        case 3: st3(1, sl, c1->twr[nst - 1], c1->twi[nst - 1], Vr, Vi, rr, ri); break;
        case 4: st4(1, sl, c1->twr[nst - 1], c1->twi[nst - 1], Vr, Vi, rr, ri); break;
        case 8: st8(1, sl, c1->twr[nst - 1], c1->twi[nst - 1], Vr, Vi, rr, ri); break;
        default: st5(1, sl, c1->twr[nst - 1], c1->twi[nst - 1], Vr, Vi, rr, ri); break;
        }
    }
}

/* plan-time: forward row FFTs of the kernel sequence, scaled, into bak */
HOT static void ac_rows_fwd_store(fft1d_plan *p, double scale)
{
    const core_plan *c1 = &p->c1;
    const int nst = c1->nstage, m0 = p->M1 / 4;
    for (int k2 = 0; k2 < p->M2; ++k2) {
        double *rr = p->gr + (size_t)k2 * p->RS;
        double *ri = p->gi + (size_t)k2 * p->RS;
        st4(m0, 1, c1->twr[0], c1->twi[0], rr, ri, p->s0r, p->s0i);
        double *Rr, *Ri;
        core_exec_range_z(c1, 1, nst, m0, 4, p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
        double *br_ = p->bakr + (size_t)k2 * p->RS;
        double *bi_ = p->baki + (size_t)k2 * p->RS;
        for (int i = 0; i < p->M1; ++i) { br_[i] = Rr[i] * scale; bi_[i] = Ri[i] * scale; }
    }
}

/* CRT scatter of one 8-column tile to interleaved y: unswap + output chirp.
   Two bodies: the NT one streams full 128B runs (y is written exactly once
   and never re-read -- the batched cells' y lives in DRAM, so skipping the
   read-for-ownership saves a third of the scatter traffic); it must live in
   its own NTF function because always_inline intrinsics refuse to inline into
   an arch=-form HOT function. Callable only when yd is 64B-aligned (every run
   starts at nlo % 4 == 0, so alignment is set by the base pointer alone). */
NTF static void exit_scatter_nt(const fft1d_plan *p,
        const double *restrict Or, const double *restrict Oi,
        const double *restrict car, const double *restrict cai,
        double *restrict yd, int c)
{
    const int M1 = p->M1, M2 = p->M2, L = p->L;
    const __m512i ixlo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i ixhi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    for (int t = 0; t < M2; ++t) {
        const int nlo = c + t * M1;
        if (nlo >= L) continue;
        int cnt = L - nlo;
        if (cnt > 8) cnt = 8;
        const double *restrict wr = Oi + 8 * p->acrow[t];   /* unswap */
        const double *restrict wi = Or + 8 * p->acrow[t];
        const double *restrict cr = car + nlo;
        const double *restrict ci = cai + nlo;
        double *restrict ys = yd + 2 * (size_t)nlo;
        if (cnt == 8) {
            __m512d vwr = _mm512_load_pd(wr), vwi = _mm512_load_pd(wi);
            __m512d vcr = _mm512_loadu_pd(cr), vci = _mm512_loadu_pd(ci);
            __m512d re = _mm512_fmsub_pd(vwr, vcr, _mm512_mul_pd(vwi, vci));
            __m512d im = _mm512_fmadd_pd(vwr, vci, _mm512_mul_pd(vwi, vcr));
            _mm512_stream_pd(ys,     _mm512_permutex2var_pd(re, ixlo, im));
            _mm512_stream_pd(ys + 8, _mm512_permutex2var_pd(re, ixhi, im));
        } else {
            for (int j = 0; j < cnt; ++j) {
                ys[2 * j]     = wr[j] * cr[j] - wi[j] * ci[j];
                ys[2 * j + 1] = wr[j] * ci[j] + wi[j] * cr[j];
            }
        }
    }
}

/* Newton map scale: v = 1/(1 + sqrt(re^2+im^2)) via rsqrt14+2NR and
   rcp14+2NR on the FMA ports instead of vsqrtpd+vdivpd serializing on the
   divider. Adopted from d1_rader r3 (chain_map8) / d1_pow2 r3 (the shipped
   "2NR-only fast map"); both records show every graded chain gate holding
   with >=3 decades of margin. max(m,1e-100) guards rsqrt(0)=inf; the clamp
   is 1e-100 and not 1e-300 because rsqrt14 of a true denormal-range operand
   lands in FP-assist territory (~250 cycles/lane on a zeroed input --
   d1_batchlane's r3 headline, re-confirmed by d1_twiddle r4). */
NTF static inline __m512d nmap_scale(__m512d re, __m512d im)
{
    const __m512d half = _mm512_set1_pd(0.5), three = _mm512_set1_pd(3.0);
    const __m512d one = _mm512_set1_pd(1.0), two = _mm512_set1_pd(2.0);
    const __m512d tiny = _mm512_set1_pd(1e-100);
    __m512d m = _mm512_fmadd_pd(re, re, _mm512_mul_pd(im, im));
    m = _mm512_max_pd(m, tiny);
    __m512d r = _mm512_rsqrt14_pd(m);
    r = _mm512_mul_pd(_mm512_mul_pd(half, r),
        _mm512_fnmadd_pd(m, _mm512_mul_pd(r, r), three));
    r = _mm512_mul_pd(_mm512_mul_pd(half, r),
        _mm512_fnmadd_pd(m, _mm512_mul_pd(r, r), three));
    __m512d t = _mm512_fmadd_pd(m, r, one);        /* 1 + sqrt(m) */
    __m512d v = _mm512_rcp14_pd(t);
    v = _mm512_mul_pd(v, _mm512_fnmadd_pd(t, v, two));
    v = _mm512_mul_pd(v, _mm512_fnmadd_pd(t, v, two));
    return v;
}

/* one full 8-run of the AC chain step tail: z = a*W + c, Newton map, then
   either premultiply by the chirp for the next step (pre) or interleave into
   the final output (out). wr/wi are 64B-aligned tile rows; everything else
   is loadu/storeu (batched b*L offsets are odd multiples of 8B). */
NTF static void chain_map8_pre(const double *restrict wr, const double *restrict wi,
        const double *restrict cr, const double *restrict ci,
        const double *restrict cs, double *restrict pr, double *restrict pi)
{
    const __m512i evn = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    __m512d vwr = _mm512_load_pd(wr), vwi = _mm512_load_pd(wi);
    __m512d vcr = _mm512_loadu_pd(cr), vci = _mm512_loadu_pd(ci);
    __m512d c0 = _mm512_loadu_pd(cs), c1 = _mm512_loadu_pd(cs + 8);
    __m512d re = _mm512_add_pd(_mm512_fmsub_pd(vwr, vcr, _mm512_mul_pd(vwi, vci)),
                               _mm512_permutex2var_pd(c0, evn, c1));
    __m512d im = _mm512_add_pd(_mm512_fmadd_pd(vwr, vci, _mm512_mul_pd(vwi, vcr)),
                               _mm512_permutex2var_pd(c0, odd, c1));
    __m512d v = nmap_scale(re, im);
    __m512d gr = _mm512_mul_pd(re, v), gi = _mm512_mul_pd(im, v);
    _mm512_storeu_pd(pr, _mm512_fmsub_pd(gr, vcr, _mm512_mul_pd(gi, vci)));
    _mm512_storeu_pd(pi, _mm512_fmadd_pd(gr, vci, _mm512_mul_pd(gi, vcr)));
}

NTF static void chain_map8_out(const double *restrict wr, const double *restrict wi,
        const double *restrict cr, const double *restrict ci,
        const double *restrict cs, double *restrict ys)
{
    const __m512i evn = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    const __m512i ixlo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i ixhi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    __m512d vwr = _mm512_load_pd(wr), vwi = _mm512_load_pd(wi);
    __m512d vcr = _mm512_loadu_pd(cr), vci = _mm512_loadu_pd(ci);
    __m512d c0 = _mm512_loadu_pd(cs), c1 = _mm512_loadu_pd(cs + 8);
    __m512d re = _mm512_add_pd(_mm512_fmsub_pd(vwr, vcr, _mm512_mul_pd(vwi, vci)),
                               _mm512_permutex2var_pd(c0, evn, c1));
    __m512d im = _mm512_add_pd(_mm512_fmadd_pd(vwr, vci, _mm512_mul_pd(vwi, vcr)),
                               _mm512_permutex2var_pd(c0, odd, c1));
    __m512d v = nmap_scale(re, im);
    __m512d gr = _mm512_mul_pd(re, v), gi = _mm512_mul_pd(im, v);
    _mm512_storeu_pd(ys,     _mm512_permutex2var_pd(gr, ixlo, gi));
    _mm512_storeu_pd(ys + 8, _mm512_permutex2var_pd(gr, ixhi, gi));
}

/* full-pass Newton-map chain tail for the mode-2 fused chain: z = tz + c,
   map, then chirp-premultiply into pr/pi (not last) or interleave into od
   (last). Scalar precise-map tail for L mod 8. */
NTF static void chain_map_pass(const double *restrict tzr, const double *restrict tzi,
        const double *restrict cd, const double *restrict car, const double *restrict cai,
        double *restrict pr, double *restrict pi, double *restrict od,
        int L, int last)
{
    const __m512i evn = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    const __m512i ixlo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i ixhi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    int k = 0;
    for (; k + 8 <= L; k += 8) {
        __m512d c0 = _mm512_loadu_pd(cd + 2 * k), c1 = _mm512_loadu_pd(cd + 2 * k + 8);
        __m512d re = _mm512_add_pd(_mm512_loadu_pd(tzr + k),
                                   _mm512_permutex2var_pd(c0, evn, c1));
        __m512d im = _mm512_add_pd(_mm512_loadu_pd(tzi + k),
                                   _mm512_permutex2var_pd(c0, odd, c1));
        __m512d v = nmap_scale(re, im);
        __m512d gr = _mm512_mul_pd(re, v), gi = _mm512_mul_pd(im, v);
        if (last) {
            _mm512_storeu_pd(od + 2 * k,     _mm512_permutex2var_pd(gr, ixlo, gi));
            _mm512_storeu_pd(od + 2 * k + 8, _mm512_permutex2var_pd(gr, ixhi, gi));
        } else {
            __m512d vcr = _mm512_loadu_pd(car + k), vci = _mm512_loadu_pd(cai + k);
            _mm512_storeu_pd(pr + k, _mm512_fmsub_pd(gr, vcr, _mm512_mul_pd(gi, vci)));
            _mm512_storeu_pd(pi + k, _mm512_fmadd_pd(gr, vci, _mm512_mul_pd(gi, vcr)));
        }
    }
    for (; k < L; ++k) {
        double re = tzr[k] + cd[2 * k], im = tzi[k] + cd[2 * k + 1];
        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
        double gr = re * sc, gi = im * sc;
        if (last) { od[2 * k] = gr; od[2 * k + 1] = gi; }
        else {
            pr[k] = gr * car[k] - gi * cai[k];
            pi[k] = gr * cai[k] + gi * car[k];
        }
    }
}

HOT static void exit_scatter(const fft1d_plan *p,
        const double *restrict Or, const double *restrict Oi,
        const double *restrict car, const double *restrict cai,
        double *restrict yd, int c)
{
    const int M1 = p->M1, M2 = p->M2, L = p->L;
    for (int t = 0; t < M2; ++t) {
        const int nlo = c + t * M1;
        if (nlo >= L) continue;
        int cnt = L - nlo;
        if (cnt > 8) cnt = 8;
        const double *restrict wr = Oi + 8 * p->acrow[t];   /* unswap */
        const double *restrict wi = Or + 8 * p->acrow[t];
        const double *restrict cr = car + nlo;
        const double *restrict ci = cai + nlo;
        double *restrict ys = yd + 2 * (size_t)nlo;
        __builtin_prefetch(ys + 16, 1, 2);
        __builtin_prefetch(ys + 24, 1, 2);
#pragma GCC ivdep
        for (int j = 0; j < cnt; ++j) {
            ys[2 * j]     = wr[j] * cr[j] - wi[j] * ci[j];
            ys[2 * j + 1] = wr[j] * ci[j] + wi[j] * cr[j];
        }
    }
}

/* fused exit, interleaved-y sink: conj-rotation premultiply on the swapped
   planes, inverse M2-axis tile FFT, unswap + output chirp + CRT scatter
   (only k < L stored) */
HOT static void ac_cols_inv_y(const fft1d_plan *p, double *restrict yd)
{
    const int M1 = p->M1, M2 = p->M2, RS = p->RS, L = p->L;
    const core_plan *c2 = &p->c2;
    const int TW = M2 + 8;
    const double *restrict car = p->ar, *restrict cai = p->ai;
    /* every scatter run starts at nlo % 4 == 0 (64B-multiple offset), so
       alignment is uniform across the call and set by the base pointer;
       batched calls pass yd = out + b*L, which is unaligned for odd b*L.
       An interior-line NT variant for those members (mask-store head/tail,
       stream the one covered line) measured 1.7-1.8x SLOWER on the node
       (65537 B=16: 2864 vs 1665; 100003 B=8: 4492 vs 2498 us, interleaved
       A/B r5): per-run mixing of RFO masked stores with NT lines defeats
       both write paths. Misaligned members keep the plain scatter. */
    const int y_aligned = ((uintptr_t)yd & 63) == 0;
    for (int c = 0; c < M1; c += 8) {
        double t0r[AC_M2MAX * 8] __attribute__((aligned(64)));
        double t0i[AC_M2MAX * 8] __attribute__((aligned(64)));
        double t1r[AC_M2MAX * 8] __attribute__((aligned(64)));
        double t1i[AC_M2MAX * 8] __attribute__((aligned(64)));
        const int cm = c % M2;
        for (int k = 0; k < M2; ++k) {
            /* planes are swapped: gr = Im W, gi = Re W; multiply W by
               conj(C) staying in swapped representation */
            const double *restrict pr = p->gr + (size_t)k * RS + c;
            const double *restrict pi_ = p->gi + (size_t)k * RS + c;
            const double *restrict qr = p->ctr + (size_t)k * TW + cm;
            const double *restrict qi = p->cti + (size_t)k * TW + cm;
            double *restrict tr = t0r + 8 * k, *restrict ti = t0i + 8 * k;
            /* one line each of the NEXT tile's gather, issued k-spread so the
               L3 latency overlaps this tile's work instead of stalling its
               gather loop (41% of exit cycles were L1-miss-pending) */
            __builtin_prefetch(pr + 8, 0, 2);
            __builtin_prefetch(pi_ + 8, 0, 2);
#pragma GCC ivdep
            for (int j = 0; j < 8; ++j) {
                tr[j] = pr[j] * qr[j] - pi_[j] * qi[j];
                ti[j] = pi_[j] * qr[j] + pr[j] * qi[j];
            }
        }
        double *Or, *Oi;
        core_exec_range_z(c2, 0, c2->nstage, M2, 8, t0r, t0i, t1r, t1i, &Or, &Oi);
        if (y_aligned)
            exit_scatter_nt(p, Or, Oi, car, cai, yd, c);
        else
            exit_scatter(p, Or, Oi, car, cai, yd, c);
    }
    if (y_aligned) __asm__ __volatile__("sfence" ::: "memory");
}

/* fused exit for the chain: z = a*W, then the whole map step
   g = (z+c)/(1+|z+c|) and either the NEXT step's chirp premultiply (into
   pre) or the final interleaved store -- no separate L-sized map pass. */
HOT static void ac_cols_inv_chain(const fft1d_plan *p, const double *restrict cd,
                                  double *restrict prer, double *restrict prei,
                                  double *restrict od, int last)
{
    const int M1 = p->M1, M2 = p->M2, RS = p->RS, L = p->L;
    const core_plan *c2 = &p->c2;
    const int TW = M2 + 8;
    const double *restrict car = p->ar, *restrict cai = p->ai;
    for (int c = 0; c < M1; c += 8) {
        double t0r[AC_M2MAX * 8] __attribute__((aligned(64)));
        double t0i[AC_M2MAX * 8] __attribute__((aligned(64)));
        double t1r[AC_M2MAX * 8] __attribute__((aligned(64)));
        double t1i[AC_M2MAX * 8] __attribute__((aligned(64)));
        const int cm = c % M2;
        for (int k = 0; k < M2; ++k) {
            const double *restrict pr = p->gr + (size_t)k * RS + c;
            const double *restrict pi_ = p->gi + (size_t)k * RS + c;
            const double *restrict qr = p->ctr + (size_t)k * TW + cm;
            const double *restrict qi = p->cti + (size_t)k * TW + cm;
            double *restrict tr = t0r + 8 * k, *restrict ti = t0i + 8 * k;
            __builtin_prefetch(pr + 8, 0, 2);
            __builtin_prefetch(pi_ + 8, 0, 2);
#pragma GCC ivdep
            for (int j = 0; j < 8; ++j) {
                tr[j] = pr[j] * qr[j] - pi_[j] * qi[j];
                ti[j] = pi_[j] * qr[j] + pr[j] * qi[j];
            }
        }
        double *Or, *Oi;
        core_exec_range_z(c2, 0, c2->nstage, M2, 8, t0r, t0i, t1r, t1i, &Or, &Oi);
        for (int t = 0; t < M2; ++t) {
            const int nlo = c + t * M1;
            if (nlo >= L) continue;
            int cnt = L - nlo;
            if (cnt > 8) cnt = 8;
            const double *restrict wr = Oi + 8 * p->acrow[t];   /* unswap */
            const double *restrict wi = Or + 8 * p->acrow[t];
            const double *restrict cr = car + nlo;
            const double *restrict ci = cai + nlo;
            const double *restrict cs = cd + 2 * (size_t)nlo;
            __builtin_prefetch(cs + 16, 0, 2);
            __builtin_prefetch(cs + 24, 0, 2);
            if (last) {
                double *restrict ys = od + 2 * (size_t)nlo;
                if (cnt == 8) { chain_map8_out(wr, wi, cr, ci, cs, ys); continue; }
#pragma GCC ivdep
                for (int j = 0; j < cnt; ++j) {
                    double re = wr[j] * cr[j] - wi[j] * ci[j] + cs[2 * j];
                    double im = wr[j] * ci[j] + wi[j] * cr[j] + cs[2 * j + 1];
                    double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                    ys[2 * j] = re * sc;
                    ys[2 * j + 1] = im * sc;
                }
            } else {
                double *restrict pr = prer + nlo, *restrict pi_ = prei + nlo;
                if (cnt == 8) { chain_map8_pre(wr, wi, cr, ci, cs, pr, pi_); continue; }
#pragma GCC ivdep
                for (int j = 0; j < cnt; ++j) {
                    double re = wr[j] * cr[j] - wi[j] * ci[j] + cs[2 * j];
                    double im = wr[j] * ci[j] + wi[j] * cr[j] + cs[2 * j + 1];
                    double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                    double gr_ = re * sc, gi_ = im * sc;
                    pr[j] = gr_ * cr[j] - gi_ * ci[j];
                    pi_[j] = gr_ * ci[j] + gi_ * cr[j];
                }
            }
        }
    }
}

static double phase_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec * 1e-3;
}

static void ac_one(fft1d_plan *p, const double _Complex *x, double _Complex *y)
{
    if (getenv("AC_PHASES")) {           /* dev-only phase split, off by default */
        double t0 = phase_now();
        ac_cols_fwd_x(p, (const double *)x);
        double t1 = phase_now();
        ac_rows_middle(p);
        double t2 = phase_now();
        ac_cols_inv_y(p, (double *)y);
        double t3 = phase_now();
        fprintf(stderr, "AC_PHASES entry=%.1f rows=%.1f exit=%.1f us\n",
                t1 - t0, t2 - t1, t3 - t2);
        return;
    }
    ac_cols_fwd_x(p, (const double *)x);
    ac_rows_middle(p);
    ac_cols_inv_y(p, (double *)y);
}

const char *fft1d_name(void) { return "d1_bluestein"; }
const char *fft1d_description(void)
{
    return "Bluestein chirp-Z any-L: split-complex mixed-radix(2/3/4/5/8/9/16) Stockham conv "
           "core, minimal smooth pad (10007->20480); large M via Agarwal-Cooley coprime 2D conv "
           "(CRT, no inter-axis twiddles; minimal-M split over odd 3^a5^b <= 135: "
           "65537->138240=1024x135, 100003->204800=8192x25) with chirp/CRT-fused entry+exit, "
           "NT-streamed exit scatter; radix-16 two-layer L1-tile stages (d1_rader's) with "
           "flavor-specific schedules (rows 8192=[4,16,16,8], conv 20480=[4,16,16,5,4]); "
           "fwd-last x kernel-mul x inv-entry fused into ONE pass at last radix 2/4/8 "
           "(stmid8 covers the trailing-8 row schedule); paired-p zmm radix-8 at the narrow "
           "s=4 stage; radix-9 tile stages; zero-pruned+chirp-fused entry and pruned "
           "chirp-fused exit for small M, NT-streamed (input-offset realigned) for "
           "DRAM-bound batched cells; direct Stockham for smooth L with NT batched "
           "interleave; dense tiny-L floor; fused map chain with rsqrt/rcp+2NR Newton map "
           "(1e-100 clamp); all planes in pre-faulted hugepage mmap blocks with L2-way skew";
}
int fft1d_supports(int L) { return L >= 2; }

fft1d_plan *fft1d_create(int L, int batch)
{
    if (L < 2 || batch < 1) return NULL;
    fft1d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L; p->batch = batch;

    /* NT final stores once the batched output alone is a decisive fraction of
       the node's 24 MB L3: below that the repeated timing calls keep y
       cache-resident and regular stores win. BLU_NONT=1 disables for A/B. */
    const int nt_big = (size_t)batch * (size_t)L * 16 >= ((size_t)8 << 20)
                       && !getenv("BLU_NONT");

    if (is_smooth(L) && L >= 4) {
        p->mode = 1;
        p->nt_out = nt_big;
        if (!core_init(&p->core, L, CF_DIRECT)) goto fail;
        double *pl[4];
        p->blkB = planes_alloc(pl, 4, (size_t)L);
        if (!p->blkB) goto fail;
        p->s0r = pl[0]; p->s0i = pl[1]; p->s1r = pl[2]; p->s1i = pl[3];
        return p;
    }
    if (L <= DENSE_MAX) {
        /* split-format dense DFT, accumulated across output index k so the
           inner loops vectorize without needing float reassociation */
        p->mode = 0;
        p->s0r = amalloc((size_t)L * L * sizeof(double));
        p->s0i = amalloc((size_t)L * L * sizeof(double));
        p->s1r = amalloc((size_t)L * sizeof(double));
        p->s1i = amalloc((size_t)L * sizeof(double));
        if (!p->s0r || !p->s0i || !p->s1r || !p->s1i) goto fail;
        for (int j = 0; j < L; ++j)
            for (int k = 0; k < L; ++k) {
                long double ph = -2.0L * PI_L * (long double)((j * k) % L)
                                 / (long double)L;
                p->s0r[(size_t)j * L + k] = (double)cosl(ph);
                p->s0i[(size_t)j * L + k] = (double)sinl(ph);
            }
        return p;
    }

    int M = choose_M(L);
    int acM = ac_choose_M(L);
    /* AC only once the single-pass conv working set has outgrown L2 (traffic
       then dominates: 65537 pays 204800 AC points over 147456 single-pass
       and still wins 15%+). Below that the single-pass fusions win: at
       M=20480 the good-mode single-pass runs 110 us on wallaby vs AC's 160. */
    if (acM && (M > 32768 || getenv("AC_SPLIT"))) M = acM;
    else acM = 0;
    int odd = M;
    while (odd % 2 == 0) odd /= 2;
    if (acM && odd >= 3 && odd <= AC_M2MAX && M / odd >= 512) {
        /* Agarwal-Cooley coprime 2D convolution */
        p->mode = 3;
        p->M = M; p->M2 = odd; p->M1 = M / odd; p->RS = p->M1 + 8;
        const int TW = odd + 8;
        if (!core_init(&p->c1, p->M1, CF_ROWS)) goto fail;
        if (!core_init(&p->c2, odd, CF_TILE)) goto fail;
        double *plA[4], *plB[4], *plC[2] = {0, 0};
        p->blkA = planes_alloc(plA, 4, (size_t)odd * p->RS);
        p->blkB = planes_alloc(plB, 4, (size_t)p->M1);
        p->blkC = planes_alloc(plC, 2, (size_t)L);
        p->ctr = amalloc((size_t)odd * TW * sizeof(double));
        p->cti = amalloc((size_t)odd * TW * sizeof(double));
        p->acrow = malloc((size_t)odd * sizeof(int));
        if (!p->blkA || !p->blkB || !p->blkC ||
            !p->ctr || !p->cti || !p->acrow) goto fail;
        p->gr = plA[0]; p->gi = plA[1]; p->bakr = plA[2]; p->baki = plA[3];
        p->s0r = plB[0]; p->s0i = plB[1]; p->s1r = plB[2]; p->s1i = plB[3];
        p->ar = plC[0]; p->ai = plC[1];
        for (int t = 0; t < odd; ++t)
            p->acrow[t] = (int)(((long)t * p->M1) % odd);
        for (int k = 0; k < odd; ++k)
            for (int r = 0; r < TW; ++r) {
                int rk = (int)(((long)(r % odd) * k) % odd);
                long double ph = -2.0L * PI_L * (long double)rk / (long double)odd;
                p->ctr[(size_t)k * TW + r] = (double)cosl(ph);
                p->cti[(size_t)k * TW + r] = (double)sinl(ph);
            }
        for (long k = 0; k < L; ++k) {
            long m2 = (k * k) % (2L * (long)L);
            long double ph = -PI_L * (long double)m2 / (long double)L;
            p->ar[k] = (double)cosl(ph); p->ai[k] = (double)sinl(ph);
        }
        /* kernel spectrum in the same 2D layout, 1/M folded in */
        double *ksr = calloc((size_t)M, sizeof(double));
        double *ksi = calloc((size_t)M, sizeof(double));
        if (!ksr || !ksi) { free(ksr); free(ksi); goto fail; }
        for (long j = 0; j < L; ++j) {
            ksr[j] = p->ar[j]; ksi[j] = -p->ai[j];
            if (j) { ksr[M - j] = ksr[j]; ksi[M - j] = ksi[j]; }
        }
        ac_cols_fwd_split(p, ksr, ksi, M);
        ac_rows_fwd_store(p, 1.0 / (double)M);
        free(ksr); free(ksi);
        return p;
    }

    p->mode = 2;
    p->M = M;
    if (!core_init(&p->core, M, CF_CONV)) goto fail;
    double *pl[6], *plC[2];
    p->blkB = planes_alloc(pl, 6, (size_t)M);
    p->blkC = planes_alloc(plC, 2, (size_t)L);
    if (!p->blkB || !p->blkC) goto fail;
    p->s0r = pl[0]; p->s0i = pl[1]; p->s1r = pl[2]; p->s1i = pl[3];
    p->br = pl[4]; p->bi = pl[5];
    p->ar = plC[0]; p->ai = plC[1];
    int lastr = p->core.radix[p->core.nstage - 1];
    p->fuse_last = (p->core.nstage >= 2 && (lastr == 2 || lastr == 4)) ? lastr : 0;
    p->nt_out = nt_big && p->fuse_last != 0;
    /* fwd-last x kernel x inv-entry fusion needs the same 2/4 last radix and
       a radix-4 stage 0 (guaranteed: choose_M enforces 4|M) */
    p->fuse_mid = (p->core.nstage >= 3 && (lastr == 2 || lastr == 4)) ? lastr : 0;

    /* chirp a[k] = exp(-i pi (k^2 mod 2L) / L); reduce in integers first */
    for (long k = 0; k < L; ++k) {
        long m2 = (k * k) % (2L * (long)L);
        long double ph = -PI_L * (long double)m2 / (long double)L;
        p->ar[k] = (double)cosl(ph); p->ai[k] = (double)sinl(ph);
    }
    /* kernel b[j] = conj(a[j]) at positions j and M-j, then spectrum / M */
    memset(p->s0r, 0, (size_t)M * sizeof(double));
    memset(p->s0i, 0, (size_t)M * sizeof(double));
    for (long j = 0; j < L; ++j) {
        p->s0r[j] = p->ar[j]; p->s0i[j] = -p->ai[j];
        if (j) { p->s0r[M - j] = p->s0r[j]; p->s0i[M - j] = p->s0i[j]; }
    }
    double *Rr, *Ri;
    core_exec(&p->core, p->s0r, p->s0i, p->s1r, p->s1i, &Rr, &Ri);
    double invM = 1.0 / (double)M;
    for (int k = 0; k < M; ++k) { p->br[k] = Rr[k] * invM; p->bi[k] = Ri[k] * invM; }
    return p;
fail:
    fft1d_destroy(p);
    return NULL;
}

/* Run the shared middle of one Bluestein transform: forward stages 1..end,
   fused kernel-multiply entry of the inverse, inverse stages 1..(last-1 if the
   exit stage is fused). On return: if fused exit, Vr/Vi hold the swapped
   planes ready for a st{2,4}_last_chirp* kernel; else they hold the full
   inverse result (swapped planes: W_r = (*Vi)[k], W_i = (*Vr)[k]). */
static void bl_middle(fft1d_plan *p, double **Vr, double **Vi)
{
    const core_plan *c = &p->core;
    const int nst = c->nstage, m0 = p->M / 4;
    double *Rr, *Ri;
    double *dr, *di;
    if (p->fuse_mid) {
        /* forward stages 1..nst-2, then the fused fwd-last/kernel/inv-entry */
        core_exec_range_y(c, 1, nst - 1, m0, 4, p->s1r, p->s1i, p->s0r, p->s0i, &Rr, &Ri);
        dr = (Rr == p->s0r) ? p->s1r : p->s0r;
        di = (Rr == p->s0r) ? p->s1i : p->s0i;
        if (p->fuse_mid == 4)
            stmid4_y(m0, c->twr[0], c->twi[0], Rr, Ri, p->br, p->bi, dr, di);
        else
            stmid2_y(m0, c->twr[0], c->twi[0], Rr, Ri, p->br, p->bi, dr, di);
    } else {
        core_exec_range_y(c, 1, nst, m0, 4, p->s1r, p->s1i, p->s0r, p->s0i, &Rr, &Ri);
        dr = (Rr == p->s0r) ? p->s1r : p->s0r;
        di = (Rr == p->s0r) ? p->s1i : p->s0i;
        st4_first_bhat(m0, c->twr[0], c->twi[0], Rr, Ri, p->br, p->bi, dr, di);
    }
    double *er = (dr == p->s0r) ? p->s1r : p->s0r;
    double *ei = (dr == p->s0r) ? p->s1i : p->s0i;
    core_exec_range_y(c, 1, p->fuse_last ? nst - 1 : nst, m0, 4, dr, di, er, ei, Vr, Vi);
}

static void bl_one(fft1d_plan *p, const double _Complex *x, double _Complex *y)
{
    const int L = p->L, M = p->M;
    const core_plan *c = &p->core;
    st4_first_chirp(M / 4, L, c->twr[0], c->twi[0], (const double *)x,
                    p->ar, p->ai, p->s1r, p->s1i);
    double *Vr, *Vi;
    bl_middle(p, &Vr, &Vi);
    if (p->fuse_last == 2 && p->nt_out)
        st2_last_chirp_nt(M / 2, Vr, Vi, p->ar, p->ai, (double *)y, L);
    else if (p->fuse_last == 4 && p->nt_out)
        st4_last_chirp_nt(M / 4, Vr, Vi, p->ar, p->ai, (double *)y, L);
    else if (p->fuse_last == 2)
        st2_last_chirp(M / 2, Vr, Vi, p->ar, p->ai, (double *)y, L);
    else if (p->fuse_last == 4)
        st4_last_chirp(M / 4, Vr, Vi, p->ar, p->ai, (double *)y, L);
    else {
        const double *restrict wr = Vi, *restrict wi = Vr;  /* unswap */
        const double *restrict car = p->ar, *restrict cai = p->ai;
        double *restrict yd = (double *)y;
        for (int k = 0; k < L; ++k) {
            yd[2 * k]     = wr[k] * car[k] - wi[k] * cai[k];
            yd[2 * k + 1] = wr[k] * cai[k] + wi[k] * car[k];
        }
    }
}

static void direct_one(fft1d_plan *p, const double _Complex *x, double _Complex *y)
{
    const int L = p->L;
    const double *restrict xd = (const double *)x;
    double *restrict s0r = p->s0r, *restrict s0i = p->s0i;
    for (int t = 0; t < L; ++t) { s0r[t] = xd[2 * t]; s0i[t] = xd[2 * t + 1]; }
    double *Rr, *Ri;
    core_exec(&p->core, s0r, s0i, p->s1r, p->s1i, &Rr, &Ri);
    const double *restrict rr = Rr, *restrict ri = Ri;
    double *restrict yd = (double *)y;
    if (p->nt_out) { interleave_nt(rr, ri, yd, L); return; }
    for (int k = 0; k < L; ++k) { yd[2 * k] = rr[k]; yd[2 * k + 1] = ri[k]; }
}

HOT static void dense_one(fft1d_plan *p, const double _Complex *x, double _Complex *y)
{
    const int L = p->L;
    const double *restrict xd = (const double *)x;
    const double *restrict wr = p->s0r, *restrict wi = p->s0i;
    double *restrict accr = p->s1r, *restrict acci = p->s1i;
    const double x0r = xd[0], x0i = xd[1];
#pragma GCC ivdep
    for (int k = 0; k < L; ++k) { accr[k] = x0r; acci[k] = x0i; }
    for (int j = 1; j < L; ++j) {
        const double xr = xd[2 * j], xi = xd[2 * j + 1];
        const double *restrict rr = wr + (size_t)j * L;
        const double *restrict ri = wi + (size_t)j * L;
#pragma GCC ivdep
        for (int k = 0; k < L; ++k) {
            accr[k] += xr * rr[k] - xi * ri[k];
            acci[k] += xr * ri[k] + xi * rr[k];
        }
    }
    double *restrict yd = (double *)y;
#pragma GCC ivdep
    for (int k = 0; k < L; ++k) { yd[2 * k] = accr[k]; yd[2 * k + 1] = acci[k]; }
}

void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    const int L = p->L;
    for (int b = 0; b < p->batch; ++b) {
        const double _Complex *x = in + (size_t)b * L;
        double _Complex *y = out + (size_t)b * L;
        if (p->mode == 3) ac_one(p, x, y);
        else if (p->mode == 2) bl_one(p, x, y);
        else if (p->mode == 1) direct_one(p, x, y);
        else dense_one(p, x, y);
    }
    if (p->nt_out) __asm__ __volatile__("sfence" ::: "memory");
}

/* Fused chain: state <- (FFT(state)+c)/(1+|FFT(state)+c|), m steps.
 * For the Bluestein path the output chirp multiply, the map, and the NEXT
 * step's input chirp multiply + deinterleave are fused into one pass, so the
 * interleaved state array is only materialized at the final step. */
void fft1d_chain(fft1d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    const int L = p->L, B = p->batch;
    const size_t count = (size_t)L * B;
    const size_t bytes = count * sizeof(double _Complex);

    if (p->mode != 2 && p->mode != 3) {
        /* plain loop, ping-pong via chain_y -- which the map loop re-reads
           immediately, so the execute path's NT stores must stay off here */
        const int nt_save = p->nt_out; p->nt_out = 0;
        if (!p->chain_y) p->chain_y = amalloc(bytes);
        double _Complex *y = p->chain_y;
        memcpy(final_out, x0, bytes);
        for (int s = 0; s < m; ++s) {
            fft1d_execute(p, final_out, y);
            const double *restrict zr = (const double *)y;
            const double *restrict cr = (const double *)c;
            double *restrict o = (double *)final_out;
            for (size_t i = 0; i < count; ++i) {
                double re = zr[2 * i] + cr[2 * i];
                double im = zr[2 * i + 1] + cr[2 * i + 1];
                double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                o[2 * i] = re * sc;
                o[2 * i + 1] = im * sc;
            }
        }
        p->nt_out = nt_save;
        return;
    }

    /* Bluestein fused chain: the entry stage reads the chirp-premultiplied
     * split state directly (no padded buffer, no memset), the exit stage
     * produces z = a*W split, and the map + next step's chirp premultiply are
     * one more L-sized pass. */
    const int M = p->M;
    const core_plan *cp = &p->core;
    const int m0 = M / 4;
    if (!p->pre_r) {
        /* chain scratch joins the skewed hugepage layout: these planes are in
           the per-step hot loop, and a 64B-aligned 4K-page allocation here
           reintroduces the L2-set page luck the r2 layout fix removed */
        double *pl[2], *pl2[2];
        p->blk_pre = planes_alloc(pl, 2, (size_t)L * B);
        if (p->blk_pre) { p->pre_r = pl[0]; p->pre_i = pl[1]; }
        p->blk_tz = planes_alloc(pl2, 2, (size_t)L);
        if (p->blk_tz) { p->tzr = pl2[0]; p->tzi = pl2[1]; }
    }
    double *pre_r = p->pre_r, *pre_i = p->pre_i, *tzr = p->tzr, *tzi = p->tzi;
    if (!pre_r || !pre_i || !tzr || !tzi) {
        /* unfused fallback (chain_y is re-read: keep NT off, as above) */
        const int nt_save = p->nt_out; p->nt_out = 0;
        if (!p->chain_y) p->chain_y = amalloc(bytes);
        double _Complex *y = p->chain_y;
        memcpy(final_out, x0, bytes);
        for (int s = 0; s < m; ++s) {
            fft1d_execute(p, final_out, y);
            const double *restrict zr = (const double *)y;
            const double *restrict cr = (const double *)c;
            double *restrict o = (double *)final_out;
            for (size_t i = 0; i < count; ++i) {
                double re = zr[2 * i] + cr[2 * i];
                double im = zr[2 * i + 1] + cr[2 * i + 1];
                double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                o[2 * i] = re * sc;
                o[2 * i + 1] = im * sc;
            }
        }
        p->nt_out = nt_save;
        return;
    }
    const double *restrict car = p->ar, *restrict cai = p->ai;
    for (int b = 0; b < B; ++b) {
        const double *restrict xd = (const double *)(x0 + (size_t)b * L);
        double *restrict pr = pre_r + (size_t)b * L, *restrict pi = pre_i + (size_t)b * L;
        for (int k = 0; k < L; ++k) {
            double xr = xd[2 * k], xi = xd[2 * k + 1];
            pr[k] = xr * car[k] - xi * cai[k];
            pi[k] = xr * cai[k] + xi * car[k];
        }
    }
    for (int s = 0; s < m; ++s) {
        const int last = (s == m - 1);
        for (int b = 0; b < B; ++b) {
            double *restrict pr = pre_r + (size_t)b * L, *restrict pi = pre_i + (size_t)b * L;
            if (p->mode == 3) {
                ac_cols_fwd_split(p, pr, pi, L);
                ac_rows_middle(p);
                ac_cols_inv_chain(p, (const double *)(c + (size_t)b * L), pr, pi,
                                  (double *)(final_out + (size_t)b * L), last);
                continue;
            } else {
                st4_first_pre(m0, L, cp->twr[0], cp->twi[0], pr, pi, p->s1r, p->s1i);
                double *Vr, *Vi;
                bl_middle(p, &Vr, &Vi);
                if (p->fuse_last == 2)
                    st2_last_chirp_split(M / 2, Vr, Vi, car, cai, tzr, tzi, L);
                else if (p->fuse_last == 4)
                    st4_last_chirp_split(M / 4, Vr, Vi, car, cai, tzr, tzi, L);
                else {
                    const double *restrict wr = Vi, *restrict wi = Vr;
                    for (int k = 0; k < L; ++k) {
                        tzr[k] = wr[k] * car[k] - wi[k] * cai[k];
                        tzi[k] = wr[k] * cai[k] + wi[k] * car[k];
                    }
                }
            }
            const double *restrict cd = (const double *)(c + (size_t)b * L);
            chain_map_pass(tzr, tzi, cd, car, cai, pr, pi,
                           (double *)(final_out + (size_t)b * L), L, last);
        }
    }
}

void fft1d_destroy(fft1d_plan *p)
{
    if (!p) return;
    core_free(&p->core); core_free(&p->c1); core_free(&p->c2);
    if (!p->blkB) { free(p->s0r); free(p->s0i); free(p->s1r); free(p->s1i); }
    free(p->w); free(p->chain_y);
    big_free(p->blk_pre); big_free(p->blk_tz);
    free(p->ctr); free(p->cti); free(p->acrow);
    big_free(p->blkA); big_free(p->blkB); big_free(p->blkC);
    free(p);
}
