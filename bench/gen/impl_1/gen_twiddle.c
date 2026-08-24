/* =============================================================================
 * GEN_TWIDDLE -- exact-twiddle tables + accuracy audit (library layer, gen_r1)
 * =============================================================================
 *
 * LIBRARY LAYER, scored by ADOPTION.  Class owners: adopt with
 *
 *     #define GEN_TWIDDLE_LIB_ONLY
 *     #include "gen_twiddle.c"          // impl/ is the include dir; everything
 *                                       // is `static`, prefix tw_, no symbols leak
 *
 * WHAT IS IN HERE AND WHY (FFTW accuracy notes: "inaccurate twiddles are the
 * most likely reason for FFT inaccuracy"; LITERATURE 04 SS5.3, 07 SS5.1-5.3):
 *
 *  1. tw_cis / tw_cisl        -- THE primitive: exp(-2*pi*i*num/den) with exact
 *     integer reduction num mod den, then OCTANT folding (|arg| <= pi/4) before
 *     long-double cosl/sinl.  Every value lands within 0.5 ulp of the true
 *     real/imag parts INDEPENDENT OF den (trig condition number <= 1 after the
 *     fold), and quarter turns come out EXACTLY (+-1, +-0), so W^0, +-i, -1
 *     twiddles are bit-exact and trivial-twiddle special cases stay trivial.
 *     Measured against the panel's current patterns (see strategies/
 *     gen_twiddle.md): double cexp / cos+sin = up to 250 ulp at L <= 128 (do
 *     not build tables that way); ANY long-double path holds ~0.5 ulp at
 *     den <= 1024, but no-fold degrades to 21 ulp and [-pi,pi]-fold to 3.9 ulp
 *     by den = 65536, while this primitive stays at 0.500 at every den.
 *  2. tw_chirp                -- Bluestein chirp exp(-i*pi*k^2/n) with the
 *     k^2 mod 2n reduction done for you (kills the overflow/precision bug).
 *  3. tw_fill_* / tw_audit_*  -- table fillers in CONSUMPTION ORDER (ice
 *     lesson: no in-sweep gathers) for the layouts the panel actually uses:
 *       tw_fill_ct_split        k1-major, s=1..r-1 inner, split re/im
 *                               (this file's demo engine)
 *       tw_fill_ct_int_rowmajor j1-major, k2 inner, interleaved pairs
 *                               (drop-in for gen_planner's pln_x->tw)
 *       tw_fill_ct_int_colmajor j-major, r inner, interleaved pairs
 *                               (gen_bluestein's per-stage layout)
 *       tw_fill_dft_split/cplx  dense DFT matrix, padded row stride
 *                               (gen_dense_prime / gen_race / gen_layout demo)
 *     plus matching tw_audit_* that recompute in long double and return the
 *     max error in double ulps -- assert <= 0.51 in your create() the way
 *     gen_layout has you assert gl_selftest().
 *  4. tw_ld_dft               -- O(n^2) long-double DFT with octant-exact
 *     roots: plan-time ground truth for Rader/Bluestein kernel tables and for
 *     per-stage error budgets (the 1.5e-14/step contract).
 *  5. tw_selftest             -- independent cross-check of the primitive
 *     (naive-fold agreement, quarter/eighth-turn exactness, |w|=1, addition
 *     theorem, chirp identities).  Run it once in create(); it is how you know
 *     the octant logic is right on your build.
 *
 * THE DEMO ENTRY (when not GEN_TWIDDLE_LIB_ONLY): any-L (2..128) row-column
 * 3D FFT whose axes are generic smallest-prime-first mixed-radix DIT pencils
 * (8 pencils wide, split complex), with EVERY table built by the fillers
 * above, audited at create(), and each 1-D stage gated at create() against
 * tw_ld_dft.  It exists to prove the layer's accuracy claim end to end on
 * every L including the round-6 surprise sizes, and as the layer's living
 * test bench.  It adopts gen_layout's THP arena for all plan memory.  Not a
 * contender at any scored size; the class entries are.
 * ============================================================================= */

#ifndef GEN_TWIDDLE_C_INCLUDED
#define GEN_TWIDDLE_C_INCLUDED

#include <complex.h>
#include <math.h>
#include <stddef.h>

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440084436210485
#endif

#define TW_PI4L 0.78539816339744830961566084581988L /* pi/4, 36 digits */

/* ---- 1. the primitive ------------------------------------------------------ */
/* exp(-2*pi*i*num/den) in long double.  den > 0, |den| < 2^60; num any long.
 * Exact integer reduction, octant fold: cosl/sinl only ever see |x| <= pi/4,
 * so the argument carries ~1e-19 relative error and the trig condition number
 * is <= 1.  Quarter turns are exact by construction (sinl(0) == 0). */
static inline void tw_cisl(long num, long den, long double *cr, long double *ci)
{
    long m = num % den;
    if (m < 0) m += den;
    long t = 8 * m;             /* theta = (pi/4) * (t / den), t in [0, 8*den) */
    long q = t / den;           /* octant 0..7 */
    long u = t % den;           /* even q: phi = (pi/4)(u/den) in [0, pi/4)    */
    if (q & 1) u = den - u;     /* odd q: psi = (pi/4)(u/den) in (0, pi/4]     */
    long double c, s;
    if (u == 0) { c = 1.0L; s = 0.0L; }  /* quarter turns exact, kill -0 */
    else {
        long double x = (TW_PI4L * (long double)u) / (long double)den;
        c = cosl(x);
        s = sinl(x);
    }
    long double C, S; /* cos(theta), sin(theta) with theta = 2*pi*m/den >= 0 */
    switch ((int)q) {
    default:
    case 0: C =  c; S =  s; break;      /* theta =        phi */
    case 1: C =  s; S =  c; break;      /* theta = pi/2 - psi */
    case 2: C = -s; S =  c; break;      /* theta = pi/2 + phi */
    case 3: C = -c; S =  s; break;      /* theta = pi   - psi */
    case 4: C = -c; S = -s; break;      /* theta = pi   + phi */
    case 5: C = -s; S = -c; break;      /* theta = 3pi/2- psi */
    case 6: C =  s; S = -c; break;      /* theta = 3pi/2+ phi */
    case 7: C =  c; S = -s; break;      /* theta = 2pi  - psi */
    }
    *cr = C;
    *ci = (S == 0.0L) ? 0.0L : -S;      /* exp(-i theta); canonical +0 */
}

/* same, rounded to double: within ~0.502 ulp of the exact value, per part */
static inline void tw_cis(long num, long den, double *cr, double *ci)
{
    long double c, s;
    tw_cisl(num, den, &c, &s);
    *cr = (double)c;
    *ci = (double)s;
}

/* ---- 2. Bluestein chirp ----------------------------------------------------- */
/* exp(-i*pi*k^2/n) = exp(-2*pi*i*(k^2 mod 2n)/(2n)); reduce k mod 2n FIRST so
 * the square cannot overflow (valid for n < 2^30). */
static inline void tw_chirp(long k, long n, double *cr, double *ci)
{
    long d = 2 * n;
    long r = k % d;
    if (r < 0) r += d;
    tw_cis((r * r) % d, d, cr, ci);
}

/* ---- 3. fillers, consumption order ------------------------------------------ */
/* CT stage of length N = r*m, twiddles w_N^(s*k1).  k1-major, s = 1..r-1 inner
 * (k1 = 0 entries are the exact 1+0i, kept for uniform indexing):
 *     wr/wi[k1*(r-1) + (s-1)]                                                  */
static inline void tw_fill_ct_split(double *wr, double *wi, long N, int r, int m)
{
    for (long k1 = 0; k1 < m; ++k1)
        for (long s = 1; s < r; ++s)
            tw_cis(s * k1, N, &wr[k1 * (r - 1) + s - 1], &wi[k1 * (r - 1) + s - 1]);
}

/* j1-major, k2 = 0..m-1 inner, interleaved (re,im) pairs -- the layout
 * gen_planner's executor consumes at x->tw + 2*(j1-1)*m:
 *     w[2*((j1-1)*m + k2)] = Re w_N^(j1*k2),  w[..+1] = Im                      */
static inline void tw_fill_ct_int_rowmajor(double *w, long N, int r, int m)
{
    for (long j1 = 1; j1 < r; ++j1)
        for (long k2 = 0; k2 < m; ++k2)
            tw_cis(j1 * k2, N, &w[2 * ((j1 - 1) * m + k2)],
                   &w[2 * ((j1 - 1) * m + k2) + 1]);
}

/* j-major, s = 1..r-1 inner, interleaved pairs -- gen_bluestein's per-stage
 * order (their len = N, S = m, r = 4):  *tf++ = cos, *tf++ = sin              */
static inline void tw_fill_ct_int_colmajor(double *w, long N, int r, int m)
{
    for (long j = 0; j < m; ++j)
        for (long s = 1; s < r; ++s) {
            tw_cis(s * j, N, &w[0], &w[1]);
            w += 2;
        }
}

/* dense DFT matrix, split, row k at wr/wi + k*rowstride (pad rows for SIMD) */
static inline void tw_fill_dft_split(double *wr, double *wi, int n, size_t rowstride)
{
    for (long k = 0; k < n; ++k)
        for (long j = 0; j < n; ++j)
            tw_cis(k * j, n, &wr[(size_t)k * rowstride + j],
                   &wi[(size_t)k * rowstride + j]);
}

/* dense DFT matrix, C99 complex, row-major n x n (gen_race / dense fallbacks) */
static inline void tw_fill_dft_cplx(double _Complex *w, int n)
{
    for (long k = 0; k < n; ++k)
        for (long j = 0; j < n; ++j) {
            double c, s;
            tw_cis(k * j, n, &c, &s);
            w[(size_t)k * n + j] = c + s * I;
        }
}

/* splat a table 8x for broadcast-free zmm loads: dst[8*i + lane] = src[i] */
static inline void tw_splat8(const double *src, double *dst, long n)
{
    for (long i = 0; i < n; ++i)
        for (int q = 0; q < 8; ++q) dst[8 * i + q] = src[i];
}

/* ---- 4. audit --------------------------------------------------------------- */
/* |got - ref| in units of the double ulp at ref's binade.  Where the exact
 * value is 0 (quarter turns) the binade ulp is meaningless, so a stray there
 * is measured in ulps at 1.0 (2^-53) -- the natural scale on the unit circle
 * and a lower bound of any true-ulp reading, so the 0.51 gate stays honest. */
static inline double tw_ulp_err(double got, long double ref)
{
    if (ref == 0.0L)
        return (double)(fabsl((long double)got) / 1.1102230246251565e-16L);
    int e;
    frexpl(fabsl(ref), &e);                    /* |ref| in [2^(e-1), 2^e) */
    long double ulp = ldexpl(1.0L, e - 53);
    return (double)(fabsl((long double)got - ref) / ulp);
}

/* max ulp error of a stored (cos, sin) pair against the exact w^(num/den) */
static inline double tw_cis_err(double gc, double gs, long num, long den)
{
    long double c, s;
    tw_cisl(num, den, &c, &s);
    double ec = tw_ulp_err(gc, c), es = tw_ulp_err(gs, s);
    return ec > es ? ec : es;
}

static inline double tw_audit_ct_split(const double *wr, const double *wi,
                                long N, int r, int m)
{
    double e = 0;
    for (long k1 = 0; k1 < m; ++k1)
        for (long s = 1; s < r; ++s) {
            double u = tw_cis_err(wr[k1 * (r - 1) + s - 1],
                                  wi[k1 * (r - 1) + s - 1], s * k1, N);
            if (u > e) e = u;
        }
    return e;
}

static inline double tw_audit_ct_int_rowmajor(const double *w, long N, int r, int m)
{
    double e = 0;
    for (long j1 = 1; j1 < r; ++j1)
        for (long k2 = 0; k2 < m; ++k2) {
            double u = tw_cis_err(w[2 * ((j1 - 1) * m + k2)],
                                  w[2 * ((j1 - 1) * m + k2) + 1], j1 * k2, N);
            if (u > e) e = u;
        }
    return e;
}

static inline double tw_audit_ct_int_colmajor(const double *w, long N, int r, int m)
{
    double e = 0;
    for (long j = 0; j < m; ++j)
        for (long s = 1; s < r; ++s, w += 2) {
            double u = tw_cis_err(w[0], w[1], s * j, N);
            if (u > e) e = u;
        }
    return e;
}

static inline double tw_audit_dft_split(const double *wr, const double *wi,
                                 int n, size_t rowstride)
{
    double e = 0;
    for (long k = 0; k < n; ++k)
        for (long j = 0; j < n; ++j) {
            double u = tw_cis_err(wr[(size_t)k * rowstride + j],
                                  wi[(size_t)k * rowstride + j], k * j, n);
            if (u > e) e = u;
        }
    return e;
}

static inline double tw_audit_dft_cplx(const double _Complex *w, int n)
{
    double e = 0;
    for (long k = 0; k < n; ++k)
        for (long j = 0; j < n; ++j) {
            double u = tw_cis_err(creal(w[(size_t)k * n + j]),
                                  cimag(w[(size_t)k * n + j]), k * j, n);
            if (u > e) e = u;
        }
    return e;
}

/* ---- 5. long-double ground truth -------------------------------------------- */
/* forward DFT, O(n^2), octant-exact roots computed on the fly: plan-time only
 * (Rader/Bluestein kernel tables, per-stage error budgets) */
static inline void tw_ld_dft(int n, const long double *xr, const long double *xi,
                      long double *Xr, long double *Xi)
{
    for (long k = 0; k < n; ++k) {
        long double ar = 0, ai = 0;
        for (long j = 0; j < n; ++j) {
            long double c, s;
            tw_cisl(j * k, n, &c, &s);
            ar += xr[j] * c - xi[j] * s;
            ai += xr[j] * s + xi[j] * c;
        }
        Xr[k] = ar;
        Xi[k] = ai;
    }
}

/* ---- 6. selftest ------------------------------------------------------------ */
/* Independent cross-checks of the octant logic; returns 1 on pass.  The naive
 * path (fold to [-pi,pi], no octant map) shares no branch with tw_cisl, so an
 * octant/sign bug shows up as millions of ulps, not 0.5. */
static inline int tw_selftest(void)
{
    static const long dens[] = { 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17,
                                 20, 24, 25, 27, 31, 32, 45, 50, 64, 100, 108,
                                 125, 127, 128, 200, 254, 256, 512, 1000 };
    const int nd = (int)(sizeof dens / sizeof dens[0]);

    for (int i = 0; i < nd; ++i) {
        long den = dens[i];
        for (long num = -3; num <= den + 3; ++num) {
            double c, s;
            tw_cis(num, den, &c, &s);
            /* against the independent naive fold, in long double.  Absolute
             * tolerance at the |w| = 1 scale (~0.54 ulp of 1.0): the naive ref
             * is a hair off exactly where tw_cis is exact (quarter turns), so
             * a per-component ulp metric against IT would false-alarm there.
             * An octant/sign bug shows up as an O(1) difference. */
            long m = num % den;
            if (m < 0) m += den;
            if (2 * m > den) m -= den;
            long double a = -8.0L * TW_PI4L * (long double)m / (long double)den;
            if (fabsl((long double)c - cosl(a)) > 1.2e-16L ||
                fabsl((long double)s - sinl(a)) > 1.2e-16L) return 0;
            /* |w| == 1 and the addition theorem, in long double */
            long double cr, ci, dr, di, er, ei;
            tw_cisl(num, den, &cr, &ci);
            if (fabsl(cr * cr + ci * ci - 1.0L) > 5e-19L) return 0;
            tw_cisl(num + 7, den, &dr, &di);
            tw_cisl(7, den, &er, &ei);
            if (fabsl(dr - (cr * er - ci * ei)) > 2e-18L ||
                fabsl(di - (cr * ei + ci * er)) > 2e-18L) return 0;
        }
        /* quarter turns bit-exact */
        double c, s;
        tw_cis(0, den, &c, &s);
        if (c != 1.0 || s != 0.0) return 0;
        if (den % 4 == 0) {
            tw_cis(den / 4, den, &c, &s);
            if (c != 0.0 || s != -1.0) return 0;
            tw_cis(den / 2, den, &c, &s);
            if (c != -1.0 || s != 0.0) return 0;
            tw_cis(3 * (den / 4), den, &c, &s);
            if (c != 0.0 || s != 1.0) return 0;
        }
        /* eighth turns = correctly rounded sqrt(1/2) */
        if (den % 8 == 0) {
            tw_cis(den / 8, den, &c, &s);
            if (c != M_SQRT1_2 || s != -M_SQRT1_2) return 0;
        }
    }
    /* chirp: definition match, symmetry, period */
    static const long ns[] = { 5, 13, 31, 100, 127 };
    for (int i = 0; i < 5; ++i) {
        long n = ns[i];
        for (long k = -n; k <= 3 * n; ++k) {
            double c1, s1, c2, s2;
            tw_chirp(k, n, &c1, &s1);
            long r = k % (2 * n);
            if (r < 0) r += 2 * n;
            tw_cis((r * r) % (2 * n), 2 * n, &c2, &s2);
            if (c1 != c2 || s1 != s2) return 0;
            tw_chirp(-k, n, &c2, &s2);
            if (c1 != c2 || s1 != s2) return 0;
            tw_chirp(k + 2 * n, n, &c2, &s2);
            if (c1 != c2 || s1 != s2) return 0;
        }
    }
    return 1;
}

#endif /* GEN_TWIDDLE_C_INCLUDED */

/* =============================================================================
 * DEMO ENTRY: any-L mixed-radix row-column 3D FFT, every table from the layer,
 * audited and 1-D-gated at create().  Compiled only when this file is the
 * entry translation unit.
 * ============================================================================= */
#ifndef GEN_TWIDDLE_LIB_ONLY

#define GEN_LAYOUT_LIB_ONLY /* adopt gen_layout's THP arena for plan memory */
#include "gen_layout.c"

#include <stdlib.h>

#include "../fft3d_api.h"

#define TWD_MAXLEV 8   /* 4,4,4,2 covers 128; deepest chain is 2^7 */
#define TWD_MAXR 128   /* leaves (m == 1) may be any prime <= 127; the combine
                        * radix of a composite is 4 or a prime <= 11 */

struct twd_level {
    int n, r, m;               /* n = r*m; r = 4 if 4|n else smallest prime  */
    double *twr, *twi;         /* w_n^(s*k1), k1-major, s inner, (r-1)*m each */
    double *Rr, *Ri;           /* dense r x r combine matrix w_r^(k2*s)       */
};

struct fft3d_plan {
    int L, batch, nlev;
    struct twd_level lev[TWD_MAXLEV];
    double s3;                 /* butterfly constants, from tw_cis: sin(2pi/3) */
    double c15, s15, c25, s25; /* cos/sin(2pi/5), cos/sin(4pi/5)               */
    double _Complex *T;        /* interleaved intermediate volume             */
    double *Gr, *Gi, *Sr, *Si, *Yr, *Yi; /* 8-wide split pencil buffers, 8L each */
    gl_arena ar;
};

const char *fft3d_name(void) { return "gen_twiddle"; }
const char *fft3d_description(void)
{
    return "LIBRARY LAYER (scored by adoption): octant-folded exact twiddles <=0.51 ulp "
           "(tw_cis/tw_chirp), consumption-order CT/DFT/chirp fillers + ulp audits + "
           "long-double DFT oracle (adopt: #define GEN_TWIDDLE_LIB_ONLY + #include "
           "gen_twiddle.c); entry = any-L mixed-radix demo, self-audited at create()";
}
int fft3d_supports(int L) { return L >= 2 && L <= 128; }

/* factorization chain: radix 4 while 4 | n, else smallest prime factor;
 * returns level count or 0 */
static int twd_factor(int L, struct twd_level *lev)
{
    int n = L, nl = 0;
    while (n > 1) {
        if (nl >= TWD_MAXLEV) return 0;
        int r = n;
        if (n % 4 == 0) r = 4;
        else
            for (int d = 2; d * d <= n; ++d)
                if (n % d == 0) { r = d; break; }
        lev[nl].n = n;
        lev[nl].r = r;
        lev[nl].m = n / r;
        if (lev[nl].m > 1 && r > 13) return 0; /* composite: 4 or prime <= 11 */
        ++nl;
        n /= r;
        if (n == 1) break;
    }
    return nl;
}

/* ---- the 8-wide split mixed-radix DIT recursion ---------------------------- */
/* x: rows j at xr/xi + 8*j*xstr (read-only view into the gather buffer);
 * y: this call's output, n rows contiguous; s: scratch, n rows contiguous.
 * Sub-calls write into s-blocks using the matching y-blocks as THEIR scratch,
 * so buffers ping-pong down the recursion with no overlap. */
/* forward DFT_r of the rows in tr/ti, results to y rows k2 at 8*(k2*ystep):
 * exact-constant butterflies for r = 2,3,4,5 (constants from tw_cis, held in
 * the plan); dense r x r with 4-row unroll otherwise (prime leaves, r = 7/11) */
static void twd_butterfly(const struct fft3d_plan *p, const struct twd_level *lv,
                          const double tr[][8], const double ti[][8],
                          double *yr, double *yi, long ystep)
{
    const int r = lv->r;
    const long st = 8 * ystep;
    switch (r) {
    case 2:
        for (int q = 0; q < 8; ++q) {
            yr[q] = tr[0][q] + tr[1][q];
            yi[q] = ti[0][q] + ti[1][q];
            yr[st + q] = tr[0][q] - tr[1][q];
            yi[st + q] = ti[0][q] - ti[1][q];
        }
        return;
    case 3: {
        const double s3 = p->s3;
        for (int q = 0; q < 8; ++q) {
            double sr_ = tr[1][q] + tr[2][q], si_ = ti[1][q] + ti[2][q];
            double vr = tr[0][q] - 0.5 * sr_, vi = ti[0][q] - 0.5 * si_;
            double wr = s3 * (tr[1][q] - tr[2][q]), wi = s3 * (ti[1][q] - ti[2][q]);
            yr[q] = tr[0][q] + sr_;
            yi[q] = ti[0][q] + si_;
            yr[st + q] = vr + wi;          /* v - i*w */
            yi[st + q] = vi - wr;
            yr[2 * st + q] = vr - wi;      /* v + i*w */
            yi[2 * st + q] = vi + wr;
        }
        return;
    }
    case 4:
        for (int q = 0; q < 8; ++q) {
            double ar = tr[0][q] + tr[2][q], ai = ti[0][q] + ti[2][q];
            double br = tr[0][q] - tr[2][q], bi = ti[0][q] - ti[2][q];
            double cr = tr[1][q] + tr[3][q], ci = ti[1][q] + ti[3][q];
            double dr = tr[1][q] - tr[3][q], di = ti[1][q] - ti[3][q];
            yr[q] = ar + cr;
            yi[q] = ai + ci;
            yr[st + q] = br + di;          /* b - i*d */
            yi[st + q] = bi - dr;
            yr[2 * st + q] = ar - cr;
            yi[2 * st + q] = ai - ci;
            yr[3 * st + q] = br - di;      /* b + i*d */
            yi[3 * st + q] = bi + dr;
        }
        return;
    case 5: {
        const double c1 = p->c15, s1 = p->s15, c2 = p->c25, s2 = p->s25;
        for (int q = 0; q < 8; ++q) {
            double t1r = tr[1][q] + tr[4][q], t1i = ti[1][q] + ti[4][q];
            double t2r = tr[2][q] + tr[3][q], t2i = ti[2][q] + ti[3][q];
            double t3r = tr[1][q] - tr[4][q], t3i = ti[1][q] - ti[4][q];
            double t4r = tr[2][q] - tr[3][q], t4i = ti[2][q] - ti[3][q];
            yr[q] = tr[0][q] + t1r + t2r;
            yi[q] = ti[0][q] + t1i + t2i;
            double a1r = tr[0][q] + c1 * t1r + c2 * t2r;
            double a1i = ti[0][q] + c1 * t1i + c2 * t2i;
            double b1r = s1 * t3r + s2 * t4r, b1i = s1 * t3i + s2 * t4i;
            double a2r = tr[0][q] + c2 * t1r + c1 * t2r;
            double a2i = ti[0][q] + c2 * t1i + c1 * t2i;
            double b2r = s2 * t3r - s1 * t4r, b2i = s2 * t3i - s1 * t4i;
            yr[st + q] = a1r + b1i;        /* a1 - i*b1 */
            yi[st + q] = a1i - b1r;
            yr[4 * st + q] = a1r - b1i;
            yi[4 * st + q] = a1i + b1r;
            yr[2 * st + q] = a2r + b2i;    /* a2 - i*b2 */
            yi[2 * st + q] = a2i - b2r;
            yr[3 * st + q] = a2r - b2i;
            yi[3 * st + q] = a2i + b2r;
        }
        return;
    }
    default: {                 /* dense, 4 output rows at a time for ILP */
        int k = 0;
        for (; k + 4 <= r; k += 4) {
            const double *R0 = lv->Rr + (size_t)k * r, *I0 = lv->Ri + (size_t)k * r;
            const double *R1 = R0 + r, *I1 = I0 + r;
            const double *R2 = R1 + r, *I2 = I1 + r;
            const double *R3 = R2 + r, *I3 = I2 + r;
            double a0r[8] = { 0 }, a0i[8] = { 0 }, a1r[8] = { 0 }, a1i[8] = { 0 };
            double a2r[8] = { 0 }, a2i[8] = { 0 }, a3r[8] = { 0 }, a3i[8] = { 0 };
            for (int s = 0; s < r; ++s) {
                const double w0 = R0[s], v0 = I0[s], w1 = R1[s], v1 = I1[s];
                const double w2 = R2[s], v2 = I2[s], w3 = R3[s], v3 = I3[s];
                for (int q = 0; q < 8; ++q) {
                    const double xr_ = tr[s][q], xi_ = ti[s][q];
                    a0r[q] += w0 * xr_ - v0 * xi_;
                    a0i[q] += w0 * xi_ + v0 * xr_;
                    a1r[q] += w1 * xr_ - v1 * xi_;
                    a1i[q] += w1 * xi_ + v1 * xr_;
                    a2r[q] += w2 * xr_ - v2 * xi_;
                    a2i[q] += w2 * xi_ + v2 * xr_;
                    a3r[q] += w3 * xr_ - v3 * xi_;
                    a3i[q] += w3 * xi_ + v3 * xr_;
                }
            }
            for (int q = 0; q < 8; ++q) {
                yr[(k + 0) * st + q] = a0r[q];
                yi[(k + 0) * st + q] = a0i[q];
                yr[(k + 1) * st + q] = a1r[q];
                yi[(k + 1) * st + q] = a1i[q];
                yr[(k + 2) * st + q] = a2r[q];
                yi[(k + 2) * st + q] = a2i[q];
                yr[(k + 3) * st + q] = a3r[q];
                yi[(k + 3) * st + q] = a3i[q];
            }
        }
        for (; k < r; ++k) {
            const double *Rr = lv->Rr + (size_t)k * r, *Ri = lv->Ri + (size_t)k * r;
            double ar[8] = { 0 }, ai[8] = { 0 };
            for (int s = 0; s < r; ++s) {
                const double wr = Rr[s], wi = Ri[s];
                for (int q = 0; q < 8; ++q) {
                    ar[q] += wr * tr[s][q] - wi * ti[s][q];
                    ai[q] += wr * ti[s][q] + wi * tr[s][q];
                }
            }
            for (int q = 0; q < 8; ++q) {
                yr[k * st + q] = ar[q];
                yi[k * st + q] = ai[q];
            }
        }
        return;
    }
    }
}

static void twd_rec(const struct fft3d_plan *p, int li,
                    const double *xr, const double *xi, long xstr,
                    double *yr, double *yi, double *sr, double *si)
{
    const struct twd_level *lv = &p->lev[li];
    const int r = lv->r, m = lv->m;
    double tr[TWD_MAXR][8] __attribute__((aligned(64)));
    double ti[TWD_MAXR][8] __attribute__((aligned(64)));

    if (m == 1) {              /* leaf: DFT_r of the strided input rows */
        for (int s = 0; s < r; ++s) {
            const double *br = xr + 8l * s * xstr, *bi = xi + 8l * s * xstr;
            for (int q = 0; q < 8; ++q) { tr[s][q] = br[q]; ti[s][q] = bi[q]; }
        }
        twd_butterfly(p, lv, tr, ti, yr, yi, 1);
        return;
    }

    for (int s = 0; s < r; ++s) /* sub-DFTs of the decimated subsequences */
        twd_rec(p, li + 1, xr + 8l * s * xstr, xi + 8l * s * xstr, xstr * r,
                sr + 8l * s * m, si + 8l * s * m,
                yr + 8l * s * m, yi + 8l * s * m);

    /* combine: X[k2*m + k1] = sum_s w_n^(s*k1) * w_r^(s*k2) * A_s[k1];
     * twiddles consumed sequentially, k1-major (consumption order).  k1 = 0
     * twiddles are exactly 1: copy, don't multiply. */
    for (int q = 0; q < 8; ++q) {
        tr[0][q] = sr[q];
        ti[0][q] = si[q];
    }
    for (int s = 1; s < r; ++s) {
        const double *ur = sr + 8l * s * m, *ui = si + 8l * s * m;
        for (int q = 0; q < 8; ++q) { tr[s][q] = ur[q]; ti[s][q] = ui[q]; }
    }
    twd_butterfly(p, lv, tr, ti, yr, yi, m);
    for (int k1 = 1; k1 < m; ++k1) {
        const double *twr = lv->twr + (size_t)k1 * (r - 1);
        const double *twi = lv->twi + (size_t)k1 * (r - 1);
        for (int q = 0; q < 8; ++q) {
            tr[0][q] = sr[8 * k1 + q];
            ti[0][q] = si[8 * k1 + q];
        }
        for (int s = 1; s < r; ++s) {
            const double *ur = sr + 8l * (s * m + k1), *ui = si + 8l * (s * m + k1);
            const double wr = twr[s - 1], wi = twi[s - 1];
            for (int q = 0; q < 8; ++q) {
                tr[s][q] = wr * ur[q] - wi * ui[q];
                ti[s][q] = wr * ui[q] + wi * ur[q];
            }
        }
        twd_butterfly(p, lv, tr, ti, yr + 8l * k1, yi + 8l * k1, m);
    }
}

/* ---- gathers/scatters: w <= 8 pencils per group, tail lanes zeroed ---------- */
/* rows contiguous in the fast index: pencil q is src + q, row j at j*rowstride */
static void twd_gather_i(const double _Complex *src, long rowstride, int n, int w,
                         double *gr, double *gi)
{
    for (int j = 0; j < n; ++j) {
        const double *sp = (const double *)(src + (size_t)j * rowstride);
        int q = 0;
        for (; q < w; ++q) { gr[8 * j + q] = sp[2 * q]; gi[8 * j + q] = sp[2 * q + 1]; }
        for (; q < 8; ++q) { gr[8 * j + q] = 0.0; gi[8 * j + q] = 0.0; }
    }
}

static void twd_scatter_i(double _Complex *dst, long rowstride, int n, int w,
                          const double *yr, const double *yi)
{
    for (int j = 0; j < n; ++j) {
        double *dp = (double *)(dst + (size_t)j * rowstride);
        for (int q = 0; q < w; ++q) {
            dp[2 * q] = yr[8 * j + q];
            dp[2 * q + 1] = yi[8 * j + q];
        }
    }
}

/* pencils ARE rows (fastest axis): pencil q at src + q*n, element j contiguous */
static void twd_gather_z(const double _Complex *src, int n, int w,
                         double *gr, double *gi)
{
    for (int j = 0; j < n; ++j) {
        int q = 0;
        for (; q < w; ++q) {
            const double *sp = (const double *)(src + (size_t)q * n + j);
            gr[8 * j + q] = sp[0];
            gi[8 * j + q] = sp[1];
        }
        for (; q < 8; ++q) { gr[8 * j + q] = 0.0; gi[8 * j + q] = 0.0; }
    }
}

static void twd_scatter_z(double _Complex *dst, int n, int w,
                          const double *yr, const double *yi)
{
    for (int q = 0; q < w; ++q)
        for (int j = 0; j < n; ++j) {
            double *dp = (double *)(dst + (size_t)q * n + j);
            dp[0] = yr[8 * j + q];
            dp[1] = yi[8 * j + q];
        }
}

/* ---- create: build every table with the layer, audit it, gate the 1-D engine */
fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L) || batch <= 0) return NULL;
    if (!tw_selftest()) return NULL;   /* the layer must prove itself first */

    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->nlev = twd_factor(L, p->lev);
    if (!p->nlev) { free(p); return NULL; }

    {   /* butterfly constants, from the layer (validated by the 1-D gate) */
        double c, s;
        tw_cis(1, 3, &c, &s);
        p->s3 = -s;                        /* sin(2pi/3) */
        tw_cis(1, 5, &c, &s);
        p->c15 = c;
        p->s15 = -s;                       /* cos/sin(2pi/5) */
        tw_cis(2, 5, &c, &s);
        p->c25 = c;
        p->s25 = -s;                       /* cos/sin(4pi/5) */
    }

    const size_t V = (size_t)L * L * L;
    size_t total = V * sizeof(double _Complex) + 6 * (8 * (size_t)L) * sizeof(double);
    for (int i = 0; i < p->nlev; ++i) {
        const struct twd_level *lv = &p->lev[i];
        total += (2 * (size_t)(lv->r - 1) * lv->m + 2 * (size_t)lv->r * lv->r)
                 * sizeof(double);
    }
    total += (size_t)(p->nlev * 4 + 8) * (GL_PAGE + GL_LINE) + (64u << 10);
    if (gl_arena_init(&p->ar, total) != 0) { free(p); return NULL; }

    p->T = gl_arena_take(&p->ar, V * sizeof(double _Complex));
    p->Gr = gl_arena_take(&p->ar, 8 * (size_t)L * sizeof(double));
    p->Gi = gl_arena_take(&p->ar, 8 * (size_t)L * sizeof(double));
    p->Sr = gl_arena_take(&p->ar, 8 * (size_t)L * sizeof(double));
    p->Si = gl_arena_take(&p->ar, 8 * (size_t)L * sizeof(double));
    p->Yr = gl_arena_take(&p->ar, 8 * (size_t)L * sizeof(double));
    p->Yi = gl_arena_take(&p->ar, 8 * (size_t)L * sizeof(double));
    if (!p->T || !p->Gr || !p->Gi || !p->Sr || !p->Si || !p->Yr || !p->Yi) {
        fft3d_destroy(p);
        return NULL;
    }

    for (int i = 0; i < p->nlev; ++i) {
        struct twd_level *lv = &p->lev[i];
        const size_t ntw = (size_t)(lv->r - 1) * lv->m;
        lv->twr = gl_arena_take(&p->ar, (ntw ? ntw : 1) * sizeof(double));
        lv->twi = gl_arena_take(&p->ar, (ntw ? ntw : 1) * sizeof(double));
        lv->Rr = gl_arena_take(&p->ar, (size_t)lv->r * lv->r * sizeof(double));
        lv->Ri = gl_arena_take(&p->ar, (size_t)lv->r * lv->r * sizeof(double));
        if (!lv->twr || !lv->twi || !lv->Rr || !lv->Ri) {
            fft3d_destroy(p);
            return NULL;
        }
        if (lv->m > 1) tw_fill_ct_split(lv->twr, lv->twi, lv->n, lv->r, lv->m);
        tw_fill_dft_split(lv->Rr, lv->Ri, lv->r, (size_t)lv->r);
        /* the audit gate: every table within 0.51 ulp or the plan refuses */
        if (lv->m > 1 &&
            tw_audit_ct_split(lv->twr, lv->twi, lv->n, lv->r, lv->m) > 0.51) {
            fft3d_destroy(p);
            return NULL;
        }
        if (tw_audit_dft_split(lv->Rr, lv->Ri, lv->r, (size_t)lv->r) > 0.51) {
            fft3d_destroy(p);
            return NULL;
        }
    }

    /* 1-D gate: one deterministic pencil through the engine vs tw_ld_dft.
     * Budget: the per-step contract is 1.5e-14; a single 1-D stage must sit
     * far under it (measured ~1e-16 rms; gate at 5e-15 leaves alarm margin). */
    {
        long double xr[128], xi[128], Xr[128], Xi[128];
        for (int j = 0; j < L; ++j) {  /* fixed quasi-random, no libm needed */
            xr[j] = (long double)((j * 2654435761u) % 1000003u) / 1000003.0L - 0.5L;
            xi[j] = (long double)((j * 40503u + 12345u) % 999983u) / 999983.0L - 0.5L;
        }
        tw_ld_dft(L, xr, xi, Xr, Xi);
        for (int j = 0; j < L; ++j)
            for (int q = 0; q < 8; ++q) {
                p->Gr[8 * j + q] = (double)xr[j];
                p->Gi[8 * j + q] = (double)xi[j];
            }
        twd_rec(p, 0, p->Gr, p->Gi, 1, p->Yr, p->Yi, p->Sr, p->Si);
        long double e2 = 0, n2 = 0;
        for (int k = 0; k < L; ++k) {
            long double dr = (long double)p->Yr[8 * k] - Xr[k];
            long double di = (long double)p->Yi[8 * k] - Xi[k];
            e2 += dr * dr + di * di;
            n2 += Xr[k] * Xr[k] + Xi[k] * Xi[k];
        }
        if (!(e2 <= 25e-30L * n2)) {   /* rel L2 <= 5e-15 */
            fft3d_destroy(p);
            return NULL;
        }
    }
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    gl_arena_destroy(&p->ar);
    free(p);
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    const int L = p->L;
    const long LL = (long)L * L;
    const size_t V = (size_t)LL * L;

    for (int b = 0; b < p->batch; ++b) {
        const double _Complex *src = in + (size_t)b * V;
        double _Complex *dst = out + (size_t)b * V;

        /* axis 0 (x): row j at j*L^2, pencil = one (y,z) site, groups of 8
         * consecutive sites are contiguous within each row */
        for (long c = 0; c < LL; c += 8) {
            int w = (LL - c < 8) ? (int)(LL - c) : 8;
            twd_gather_i(src + c, LL, L, w, p->Gr, p->Gi);
            twd_rec(p, 0, p->Gr, p->Gi, 1, p->Yr, p->Yi, p->Sr, p->Si);
            twd_scatter_i(p->T + c, LL, L, w, p->Yr, p->Yi);
        }
        /* axis 1 (y): per x-plane, row j at j*L, pencils indexed by z */
        for (int x = 0; x < L; ++x) {
            double _Complex *pl = p->T + (size_t)x * LL;
            for (int c = 0; c < L; c += 8) {
                int w = (L - c < 8) ? (L - c) : 8;
                twd_gather_i(pl + c, L, L, w, p->Gr, p->Gi);
                twd_rec(p, 0, p->Gr, p->Gi, 1, p->Yr, p->Yi, p->Sr, p->Si);
                twd_scatter_i(pl + c, L, L, w, p->Yr, p->Yi);
            }
        }
        /* axis 2 (z): pencils are contiguous rows, 8 rows per group */
        for (long c = 0; c < LL; c += 8) {
            int w = (LL - c < 8) ? (int)(LL - c) : 8;
            twd_gather_z(p->T + c * L, L, w, p->Gr, p->Gi);
            twd_rec(p, 0, p->Gr, p->Gi, 1, p->Yr, p->Yi, p->Sr, p->Si);
            twd_scatter_z(dst + c * L, L, w, p->Yr, p->Yi);
        }
    }
}

#endif /* GEN_TWIDDLE_LIB_ONLY */
