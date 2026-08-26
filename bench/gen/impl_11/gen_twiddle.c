/* =============================================================================
 * GEN_TWIDDLE -- exact-twiddle tables + accuracy audit (library layer, gen_r2)
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
 *  1b. tw_cisl_ds / tw_cis_ds -- the SAME root in dual-select Linzer-Feig
 *     FMA form (NEW gen_r5; LITERATURE 11 Tier 1, Bergach arXiv:2604.00567,
 *     first validation in performant software): w = m*(1+i*t) or m*(t+i),
 *     select per twiddle so |t| <= 1 ALWAYS, t one long-double tanl of the
 *     octant-reduced argument.  Same 4-FMA multiply count as (c,s); the
 *     tighter worst-case error bound is free.  tw_fill_ct_ds_split /
 *     tw_audit_ct_ds_split fill/gate the k1-major CT layout; the demo's
 *     combine consumes them under -DTWD_DS (raced; see strategy record).
 *  2. tw_chirp                -- Bluestein chirp exp(-i*pi*k^2/n) with the
 *     k^2 mod 2n reduction done for you (kills the overflow/precision bug).
 *     ADOPTED gen_r2 by gen_bluestein (with tw_fill_ct_int_colmajor).
 *  3. tw_fill_* / tw_audit_*  -- table fillers in CONSUMPTION ORDER (ice
 *     lesson: no in-sweep gathers) for the layouts the panel actually uses:
 *       tw_fill_ct_split        k1-major, s=1..r-1 inner, split re/im
 *                               (this file's demo engine)
 *       tw_fill_ct_int_rowmajor j1-major, k2 inner, interleaved pairs
 *                               (gen_planner's OLD gen_r1 pln_x->tw)
 *       tw_fill_ct_int_k2major  k2-major (k2 >= 1), j1 inner, interleaved --
 *                               gen_planner's gen_r2 fused-leaf layout
 *       tw_fill_ct_int_colmajor j-major, r inner, interleaved pairs
 *                               (gen_bluestein's per-stage layout; conjugate
 *                               the odd slots for their inverse stages)
 *       tw_fill_dft_split/cplx  dense DFT matrix, padded row stride
 *                               (gen_dense_prime / gen_race / gen_layout demo;
 *                               the one-line fix for double-cexp refnd gates)
 *       tw_fill_dense_simd      gen_planner's gen_r3 tiled-kernel dense layout
 *                               (NEW gen_r3): WR[np][n] real parts + (-wi,+wi)
 *                               broadcast_f64x2 pairs, zero pad rows audited
 *       tw_fill_fold_half       folded odd-n half-system C/S real matrices
 *                               (NEW gen_r3): pln_fold_matrix's exact layout;
 *                               the gen_rader / gen_dense_prime fold shape
 *     plus matching tw_audit_* that recompute in long double and return the
 *     max error in double ulps -- assert <= 0.51 in your create() the way
 *     gen_layout has you assert gl_selftest().
 *  4. RADER helpers (NEW gen_r2, for gen_rader's any-prime round-3 mandate
 *     and gen_planner's rad-p nodes):
 *       tw_modpow / tw_primroot   exact integer modpow; smallest primitive
 *                                 root of a prime (plan-time)
 *       tw_fill_rader_seq         w_p^{g^q} or w_p^{g^-q}, q = 0..p-2, in
 *                                 generator (consumption) order, + audit
 *       tw_fill_rader_half        (NEW gen_r3) the FOLDED form's split real
 *                                 kernels over Z_p^x mod {+-1}: cos/sin(2pi g^q/p),
 *                                 q = 0..(p-3)/2, generator order, + audit --
 *                                 gen_rader's any-prime cyclic-h correlations
 *       tw_fill_rader_fft         DFT_{p-1} of the b-sequence w_p^{g^-q},
 *                                 computed END-TO-END in long double
 *                                 (octant-exact roots + tw_ld_dft), one
 *                                 rounding to double; optional 1/(p-1) scale
 *  5. tw_ld_dft               -- O(n^2) long-double DFT with octant-exact
 *     roots: plan-time ground truth for Rader/Bluestein kernel tables and for
 *     per-stage error budgets (the 1.5e-14/step contract).
 *  6. tw_selftest             -- independent cross-check of the primitive
 *     (naive-fold agreement, quarter/eighth-turn exactness, |w|=1, addition
 *     theorem, chirp identities, primitive-root orbits, Rader kernel
 *     identities V[0] = -1 and Parseval).  Run it once in create().
 *
 * THE DEMO ENTRY (when not GEN_TWIDDLE_LIB_ONLY): any-L (2..128) row-column
 * 3D FFT whose axes are generic smallest-prime-first mixed-radix DIT pencils
 * (8 pencils wide, split complex, one zmm per lane-row via GNU vector
 * extensions), with EVERY table built by the fillers above, audited at
 * create(), and each 1-D stage gated at create() against tw_ld_dft.  It
 * exists to prove the layer's accuracy claim end to end on every L including
 * the round-6 surprise sizes, and as the layer's living test bench.  It
 * adopts gen_layout's THP arena for all plan memory.  Not a contender at any
 * scored size; the class entries are.
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

/* ---- 1b. dual-select FMA form (NEW gen_r5) ---------------------------------- */
/* LITERATURE 11 Tier 1 (Bergach, arXiv:2604.00567): the same root
 * w = exp(-2*pi*i*num/den) returned as the Linzer-Feig factorization pair an
 * FMA butterfly wants, with the factorization SELECTED PER TWIDDLE so the
 * stored ratio never exceeds 1 (the paper's worst case 163 -> 1.0, "235x
 * tighter error bound", zero runtime cost):
 *     swap = 0:  w = m*(1 + i*t)    (|Re w| >= |Im w|; m = Re w, signed)
 *     swap = 1:  w = m*(t + i)      (|Im w| >  |Re w|; m = Im w, signed)
 * so |t| <= 1 ALWAYS.  The octant fold hands us the select for free: after
 * reduction the argument x is in [0, pi/4], the larger component is always
 * the cos(x) one, and the octant q alone decides which slot it lands in and
 * with which signs.  m is the primitive's own <= 0.5 ulp component; t is ONE
 * long-double tanl of the reduced argument (condition number <= pi/2 there,
 * |tan| <= 1), NOT the quotient of two already-rounded doubles, so t is also
 * within ~0.5 ulp of the exact ratio.  Quarter turns: m = +-1, t = +0
 * exactly.  Eighth turns: t = +-1 exactly, m = correctly rounded sqrt(1/2).
 * The multiply kernel, 4 FMA-class ops -- SAME count as the (c,s) form, the
 * tighter bound is free:
 *     swap = 0:  w*u = m*((ur - t*ui) + i*(ui + t*ur))
 *     swap = 1:  w*u = m*((t*ur - ui) + i*(t*ui + ur))
 * For codelet GENERATORS (gen_planner / gen_pow2 / gen_powp fused leaves,
 * where twiddles are compile-time or plan-time constants) the select is a
 * per-site constant: no branch survives into the kernel. */
static inline void tw_cisl_ds(long num, long den, long double *m,
                              long double *t, int *swap)
{
    long mm = num % den;
    if (mm < 0) mm += den;
    long tt = 8 * mm;
    long q = tt / den;
    long u = tt % den;
    if (q & 1) u = den - u;
    long double c, tn;
    if (u == 0) { c = 1.0L; tn = 0.0L; }                    /* quarter turns  */
    else if (u == den) { c = cosl(TW_PI4L); tn = 1.0L; }    /* x = pi/4 exact */
    else {
        long double x = (TW_PI4L * (long double)u) / (long double)den;
        c = cosl(x);
        tn = tanl(x);
    }
    /* octant tables, derived from tw_cisl's switch with w = C - i*S:
     * the larger component is the cos(x) one; SW says which slot, SM/ST the
     * signs of m and t there. */
    static const signed char SM[8]   = { +1, -1, -1, -1, -1, +1, +1, +1 };
    static const signed char ST[8]   = { -1, -1, +1, +1, -1, -1, +1, +1 };
    static const unsigned char SW[8] = {  0,  1,  1,  0,  0,  1,  1,  0 };
    *m = SM[q] * c;
    *t = (tn == 0.0L) ? 0.0L : ST[q] * tn;    /* canonical +0 at quarter turns */
    *swap = SW[q];
}

/* same, rounded to double: each stored constant within ~0.502 ulp */
static inline void tw_cis_ds(long num, long den, double *m, double *t, int *swap)
{
    long double lm, lt;
    tw_cisl_ds(num, den, &lm, &lt, swap);
    *m = (double)lm;
    *t = (double)lt;
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

/* dual-select FMA tables for the same CT stage and the same k1-major,
 * s = 1..r-1 inner indexing as tw_fill_ct_split (NEW gen_r5, lit 11 Tier 1):
 *     m/t[k1*(r-1) + (s-1)]  the scale and |.|<=1 ratio,  sw[..] the select.
 * k1 = 0 entries are exactly (m,t,sw) = (1, +0, 0), kept for uniform
 * indexing, so trivial-twiddle special cases stay trivial. */
static inline void tw_fill_ct_ds_split(double *m, double *t, unsigned char *sw,
                                       long N, int r, int mlen)
{
    for (long k1 = 0; k1 < mlen; ++k1)
        for (long s = 1; s < r; ++s) {
            int w_;
            tw_cis_ds(s * k1, N, &m[k1 * (r - 1) + s - 1],
                      &t[k1 * (r - 1) + s - 1], &w_);
            sw[k1 * (r - 1) + s - 1] = (unsigned char)w_;
        }
}

/* j1-major, k2 = 0..m-1 inner, interleaved (re,im) pairs -- gen_planner's OLD
 * (gen_r1) executor layout at x->tw + 2*(j1-1)*m:
 *     w[2*((j1-1)*m + k2)] = Re w_N^(j1*k2),  w[..+1] = Im                      */
static inline void tw_fill_ct_int_rowmajor(double *w, long N, int r, int m)
{
    for (long j1 = 1; j1 < r; ++j1)
        for (long k2 = 0; k2 < m; ++k2)
            tw_cis(j1 * k2, N, &w[2 * ((j1 - 1) * m + k2)],
                   &w[2 * ((j1 - 1) * m + k2) + 1]);
}

/* k2-major, j1 = 1..r-1 inner, k2 >= 1 ONLY, interleaved pairs -- gen_planner's
 * gen_r2 fused-leaf layout (their record, item 3: "pln_x->tw layout changed
 * from j1-major to k2-major [(k2-1)*(r-1) + (j1-1)], k2 >= 1 only"):
 *     w[2*((k2-1)*(r-1) + (j1-1))] = Re w_N^(j1*k2),  w[..+1] = Im             */
static inline void tw_fill_ct_int_k2major(double *w, long N, int r, int m)
{
    for (long k2 = 1; k2 < m; ++k2)
        for (long j1 = 1; j1 < r; ++j1) {
            tw_cis(j1 * k2, N, &w[0], &w[1]);
            w += 2;
        }
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

/* gen_planner's gen_r3 SIMD dense layout (pln_dense_matrix, PLN_SIMD): one
 * buffer of 3*np*n doubles, np = row count padded for the 4-row-tiled kernel.
 *   WR block: m[k*n + j] = Re w_n^(k*j), k-major; rows k >= n stay ZERO
 *   WP block at m + np*n: 16-byte pairs for one broadcast_f64x2 into the
 *   alternating-sign FMA: m[np*n + 2*(k*n+j)] = -Im w, [..+1] = +Im w
 * Call on a calloc'd/zeroed buffer so the pad rows stay zero (the tiled
 * kernel multiplies them; the audit checks they ARE zero). */
static inline void tw_fill_dense_simd(double *m, int n, int np)
{
    for (long k = 0; k < n; ++k)
        for (long j = 0; j < n; ++j) {
            double c, s;
            tw_cis(k * j, n, &c, &s);
            m[(size_t)k * n + j] = c;
            m[(size_t)np * n + 2 * ((size_t)k * n + j)] = (s == 0.0) ? 0.0 : -s;
            m[(size_t)np * n + 2 * ((size_t)k * n + j) + 1] = s;
        }
}

/* folded odd-n half-system REAL matrices (gen_planner pln_fold_matrix; the
 * gen_rader / gen_dense_prime / gen_layout conjugate-fold shape): h = n/2
 * (n odd), one buffer of 2*hp*h doubles, hp = padded row count (pads zero):
 *   C block: m[(k-1)*h + (j-1)]        = cos(2*pi*k*j/n),  k,j = 1..h
 *   S block: m[hp*h + (k-1)*h + (j-1)] = sin(2*pi*k*j/n)
 * Call on a zeroed buffer, same pad contract as tw_fill_dense_simd. */
static inline void tw_fill_fold_half(double *m, int n, int hp)
{
    const int h = n / 2;
    for (long k = 1; k <= h; ++k)
        for (long j = 1; j <= h; ++j) {
            double c, s;
            tw_cis(k * j, n, &c, &s);
            m[(size_t)(k - 1) * h + (j - 1)] = c;
            m[(size_t)hp * h + (size_t)(k - 1) * h + (j - 1)] =
                (s == 0.0) ? 0.0 : -s;      /* sin(2*pi*kj/n) = -Im w_n^(kj) */
        }
}

/* ---- 4. Rader: integer helpers + generator-ordered tables ------------------- */
/* b^e mod p, exact for p < 2^31 (products stay under 2^62) */
static inline long tw_modpow(long b, long e, long p)
{
    long r = 1 % p;
    b %= p;
    if (b < 0) b += p;
    while (e > 0) {
        if (e & 1) r = (r * b) % p;
        b = (b * b) % p;
        e >>= 1;
    }
    return r;
}

/* smallest primitive root of the odd prime p (plan-time; trial-factors p-1,
 * requires g^((p-1)/q) != 1 for every prime q | p-1).  Returns 0 if p is not
 * prime (a composite slipped past your factorizer -- treat as fatal). */
static inline long tw_primroot(long p)
{
    if (p == 2) return 1;
    long f[16];
    int nf = 0;
    long n = p - 1;
    for (long d = 2; d * d <= n; ++d)
        if (n % d == 0) {
            f[nf++] = d;
            while (n % d == 0) n /= d;
        }
    if (n > 1) f[nf++] = n;
    for (long g = 2; g < p; ++g) {
        int ok = (tw_modpow(g, p - 1, p) == 1);   /* fails when p not prime */
        for (int i = 0; i < nf && ok; ++i)
            if (tw_modpow(g, (p - 1) / f[i], p) == 1) ok = 0;
        if (ok) return g;
    }
    return 0;
}

/* generator-ordered roots, interleaved (re,im) pairs, q = 0..p-2:
 *     dir >= 0:  w[2q] + i*w[2q+1] = exp(-2*pi*i * g^{ q} / p)   (input gather)
 *     dir <  0:  ...              = exp(-2*pi*i * g^{-q} / p)   (the b kernel)
 * This is the consumption order of a Rader gather/convolution sweep. */
static inline void tw_fill_rader_seq(double *w, long p, long g, int dir)
{
    long step = (dir < 0) ? tw_modpow(g, p - 2, p) : (g % p); /* g^-1 = g^(p-2) */
    long e = 1;
    for (long q = 0; q < p - 1; ++q) {
        tw_cis(e, p, &w[2 * q], &w[2 * q + 1]);
        e = (e * step) % p;
    }
}

/* FOLDED Rader (gen_rader's any-prime round-3 form): the conjugate fold maps
 * the p-point DFT onto the quotient group Z_p^x mod {+-1}, cyclic of order
 * h = (p-1)/2, and ANY primitive root g of p generates it (g^q, q = 0..h-1,
 * visits each {e, p-e} class exactly once because g^h = -1).  Emits the split
 * REAL kernels in generator = consumption order, q = 0..h-1, e = g^{+-q} mod p:
 *     cw[q] = cos(2*pi*e/p)   (class-invariant: cos(-x) = cos(x))
 *     sw[q] = sin(2*pi*e/p)   (sign belongs to the representative e in 0..p-1;
 *                              your fold's (-1)^q twist tables consume it)
 * cw feeds the cyclic-h correlation for E, sw the negacyclic one for O. */
static inline void tw_fill_rader_half(double *cw, double *sw, long p, long g, int dir)
{
    long h = (p - 1) / 2;
    long step = (dir < 0) ? tw_modpow(g, p - 2, p) : (g % p);
    long e = 1;
    for (long q = 0; q < h; ++q) {
        double c, s;
        tw_cis(e, p, &c, &s);
        cw[q] = c;
        sw[q] = (s == 0.0) ? 0.0 : -s;      /* sin(2*pi*e/p) */
        e = (e * step) % p;
    }
}

/* ---- 5. audit --------------------------------------------------------------- */
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

/* dual-select audit: each stored constant against its exact long-double
 * value in ulps (both must be correctly rounded, <= 0.51), PLUS the
 * structural invariants |t| <= 1 and the select flag itself (a wrong select
 * or an out-of-range ratio returns 1e9, not a fraction of an ulp). */
static inline double tw_audit_ct_ds_split(const double *m, const double *t,
                                          const unsigned char *sw,
                                          long N, int r, int mlen)
{
    double e = 0;
    for (long k1 = 0; k1 < mlen; ++k1)
        for (long s = 1; s < r; ++s) {
            long double lm, lt;
            int w_;
            tw_cisl_ds(s * k1, N, &lm, &lt, &w_);
            size_t ix = (size_t)k1 * (r - 1) + s - 1;
            if (sw[ix] != (unsigned char)w_ || fabs(t[ix]) > 1.0) return 1e9;
            double u = tw_ulp_err(m[ix], lm), v = tw_ulp_err(t[ix], lt);
            if (v > u) u = v;
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

static inline double tw_audit_ct_int_k2major(const double *w, long N, int r, int m)
{
    double e = 0;
    for (long k2 = 1; k2 < m; ++k2)
        for (long j1 = 1; j1 < r; ++j1, w += 2) {
            double u = tw_cis_err(w[0], w[1], j1 * k2, N);
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

/* audits every component of the SIMD dense layout: WR vs cos, both WP pair
 * slots vs -+sin, AND that the pad rows k = n..np-1 are exactly zero (the
 * tiled kernel multiplies them; a stray NaN there is 1e9 "ulps", not 0.5) */
static inline double tw_audit_dense_simd(const double *m, int n, int np)
{
    double e = 0;
    const double *wp = m + (size_t)np * n;
    for (long k = 0; k < n; ++k)
        for (long j = 0; j < n; ++j) {
            long double c, s;
            tw_cisl(k * j, n, &c, &s);
            double u = tw_ulp_err(m[(size_t)k * n + j], c);
            double v = tw_ulp_err(m[(size_t)np * n + 2 * ((size_t)k * n + j) + 1], s);
            double x = tw_ulp_err(-m[(size_t)np * n + 2 * ((size_t)k * n + j)], s);
            if (v > u) u = v;
            if (x > u) u = x;
            if (u > e) e = u;
        }
    for (size_t i = (size_t)n * n; i < (size_t)np * n; ++i)
        if (m[i] != 0.0) return 1e9;
    for (size_t i = 2 * (size_t)n * n; i < 2 * (size_t)np * n; ++i)
        if (wp[i] != 0.0) return 1e9;
    return e;
}

static inline double tw_audit_fold_half(const double *m, int n, int hp)
{
    const int h = n / 2;
    double e = 0;
    for (long k = 1; k <= h; ++k)
        for (long j = 1; j <= h; ++j) {
            long double c, s;
            tw_cisl(k * j, n, &c, &s);   /* s = -sin(2*pi*kj/n) */
            double u = tw_ulp_err(m[(size_t)(k - 1) * h + (j - 1)], c);
            double v = tw_ulp_err(-m[(size_t)hp * h + (size_t)(k - 1) * h + (j - 1)], s);
            if (v > u) u = v;
            if (u > e) e = u;
        }
    for (size_t i = (size_t)h * h; i < (size_t)hp * h; ++i)
        if (m[i] != 0.0 || m[(size_t)hp * h + i] != 0.0) return 1e9;
    return e;
}

static inline double tw_audit_rader_seq(const double *w, long p, long g, int dir)
{
    long step = (dir < 0) ? tw_modpow(g, p - 2, p) : (g % p);
    long e = 1;
    double err = 0;
    for (long q = 0; q < p - 1; ++q) {
        double u = tw_cis_err(w[2 * q], w[2 * q + 1], e, p);
        if (u > err) err = u;
        e = (e * step) % p;
    }
    return err;
}

static inline double tw_audit_rader_half(const double *cw, const double *sw,
                                         long p, long g, int dir)
{
    long h = (p - 1) / 2;
    long step = (dir < 0) ? tw_modpow(g, p - 2, p) : (g % p);
    long e = 1;
    double err = 0;
    for (long q = 0; q < h; ++q) {
        double u = tw_cis_err(cw[q], -sw[q], e, p);   /* Im w_p^e = -sin */
        if (u > err) err = u;
        e = (e * step) % p;
    }
    return err;
}

/* ---- 6. long-double ground truth -------------------------------------------- */
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

/* Rader kernel table: V = DFT_{p-1}(b), b_q = exp(-2*pi*i * g^{-q} / p),
 * computed END-TO-END in long double (octant-exact roots + tw_ld_dft) and
 * rounded ONCE to double -- the plan-time table a Rader convolution multiplies
 * by.  scale_inv != 0 folds the inverse-FFT 1/(p-1) into the kernel.  p <= 128
 * (the campaign's L cap; O(p^2) trig, plan-time only). */
static inline void tw_fill_rader_fft(double _Complex *V, long p, long g,
                                     int scale_inv)
{
    long double br[128], bi[128], Br[128], Bi[128];
    long ginv = tw_modpow(g, p - 2, p);
    long e = 1;
    for (long q = 0; q < p - 1; ++q) {
        tw_cisl(e, p, &br[q], &bi[q]);
        e = (e * ginv) % p;
    }
    tw_ld_dft((int)(p - 1), br, bi, Br, Bi);
    long double s = scale_inv ? 1.0L / (long double)(p - 1) : 1.0L;
    for (long t = 0; t < p - 1; ++t)
        V[t] = (double)(Br[t] * s) + (double)(Bi[t] * s) * I;
}

/* ---- 7. selftest ------------------------------------------------------------ */
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
    /* primitive roots: the orbit g^0..g^{p-2} must visit every nonzero
     * residue exactly once and close (g^{p-1} = 1) */
    {
        static const long ps[] = { 3, 5, 7, 13, 17, 31, 101, 127 };
        for (int i = 0; i < 8; ++i) {
            long p = ps[i], g = tw_primroot(p);
            if (!g) return 0;
            char seen[128] = { 0 };
            long e = 1;
            for (long q = 0; q < p - 1; ++q) {
                if (seen[e]) return 0;
                seen[e] = 1;
                e = (e * g) % p;
            }
            if (e != 1) return 0;
        }
        if (tw_primroot(9) != 0 || tw_primroot(15) != 0) return 0; /* composites */
    }
    /* Rader tables (p = 13): the seq audit must sit at the primitive's own
     * accuracy, and the kernel FFT must satisfy V[0] = sum of all nontrivial
     * roots = -1 and Parseval sum|V|^2 = (p-1)^2 */
    {
        const long p = 13, g = tw_primroot(p);
        double w[2 * 12];
        double _Complex V[12];
        tw_fill_rader_seq(w, p, g, -1);
        if (tw_audit_rader_seq(w, p, g, -1) > 0.51) return 0;
        tw_fill_rader_seq(w, p, g, +1);
        if (tw_audit_rader_seq(w, p, g, +1) > 0.51) return 0;
        tw_fill_rader_fft(V, p, g, 0);
        if (cabs(V[0] + 1.0) > 1e-14) return 0;
        long double ps2 = 0;
        for (int t = 0; t < 12; ++t)
            ps2 += (long double)creal(V[t]) * creal(V[t]) +
                   (long double)cimag(V[t]) * cimag(V[t]);
        if (fabsl(ps2 - 144.0L) > 1e-12L) return 0;
        tw_fill_rader_fft(V, p, g, 1);           /* scaled variant: /(p-1) */
        if (cabs(V[0] + 1.0 / 12.0) > 1e-15) return 0;
    }
    /* the k2-major filler agrees with the audit (layout regression guard) */
    {
        double w[2 * 4 * 3];                     /* N=20, r=4, m=5 */
        tw_fill_ct_int_k2major(w, 20, 4, 5);
        if (tw_audit_ct_int_k2major(w, 20, 4, 5) > 0.51) return 0;
    }
    /* SIMD dense + fold-half fillers: fill/audit round trip incl. pad rows,
     * and a spot value (n=15, k=3, j=5: kj=15 -> w=1 exactly, sin exactly 0) */
    {
        double d[3 * 8 * 7] = { 0 };             /* n=7, np=8 */
        tw_fill_dense_simd(d, 7, 8);
        if (tw_audit_dense_simd(d, 7, 8) > 0.51) return 0;
        double f[2 * 8 * 7] = { 0 };             /* n=15, h=7, hp=8 */
        tw_fill_fold_half(f, 15, 8);
        if (tw_audit_fold_half(f, 15, 8) > 0.51) return 0;
        if (f[(3 - 1) * 7 + (5 - 1)] != 1.0 ||
            f[8 * 7 + (3 - 1) * 7 + (5 - 1)] != 0.0) return 0;
    }
    /* folded-Rader half sequence: audit both directions at p=13, and the
     * orbit g^0..g^{h-1} must visit each class {e, p-e} exactly once */
    {
        const long p = 13, g = tw_primroot(p), h = 6;
        double cw[6], sw[6];
        tw_fill_rader_half(cw, sw, p, g, +1);
        if (tw_audit_rader_half(cw, sw, p, g, +1) > 0.51) return 0;
        tw_fill_rader_half(cw, sw, p, g, -1);
        if (tw_audit_rader_half(cw, sw, p, g, -1) > 0.51) return 0;
        char seen[7] = { 0 };
        long e = 1;
        for (long q = 0; q < h; ++q) {
            long cls = (e <= p / 2) ? e : p - e;
            if (seen[cls]) return 0;
            seen[cls] = 1;
            e = (e * g) % p;
        }
    }
    /* dual-select form (NEW gen_r5): for every den in the sweep, reconstruct
     * w from (m, t, swap) and compare against tw_cisl.  Tolerance is absolute
     * at the |w| = 1 scale: the minor component m*t carries both constants'
     * roundings (<= ~1.2e-16), the major is m itself (<= 0.6e-16).  Also the
     * structural invariants: |t| <= 1 everywhere; the select really picks the
     * larger component; quarter turns are (m,t) = (+-1, +0) exactly; eighth
     * turns have t = +-1 exactly and m = correctly rounded sqrt(1/2). */
    for (int i = 0; i < nd; ++i) {
        long den = dens[i];
        for (long num = -3; num <= den + 3; ++num) {
            double m, t;
            int sw;
            tw_cis_ds(num, den, &m, &t, &sw);
            if (fabs(t) > 1.0) return 0;
            long double cr, ci;
            tw_cisl(num, den, &cr, &ci);
            long double rr = sw ? (long double)m * t : (long double)m;
            long double ri = sw ? (long double)m : (long double)m * t;
            if (fabsl(rr - cr) > 2.5e-16L || fabsl(ri - ci) > 2.5e-16L) return 0;
            /* the select must pick the larger component -- up to the exact
             * tie at eighth turns, where the long-double reference breaks
             * the tie by ~1e-19 arbitrarily and either factorization is
             * correct (tolerance well under any non-tie gap, which is
             * >= ~1e-5 at den <= 1000) */
            if (fabsl(ci) > fabsl(cr) + 1e-12L && !sw) return 0;
            if (fabsl(cr) > fabsl(ci) + 1e-12L && sw) return 0;
        }
        double m, t;
        int sw;
        tw_cis_ds(0, den, &m, &t, &sw);
        if (m != 1.0 || t != 0.0 || sw != 0) return 0;
        if (den % 4 == 0) {
            tw_cis_ds(den / 4, den, &m, &t, &sw);         /* w = -i */
            if (m != -1.0 || t != 0.0 || sw != 1) return 0;
            tw_cis_ds(den / 2, den, &m, &t, &sw);         /* w = -1 */
            if (m != -1.0 || t != 0.0 || sw != 0) return 0;
        }
        if (den % 8 == 0) {
            tw_cis_ds(den / 8, den, &m, &t, &sw);         /* w = (1-i)/sqrt2 */
            if (m != -M_SQRT1_2 || t != -1.0 || sw != 1) return 0;
        }
    }
    /* the ds filler agrees with its audit (layout regression guard) */
    {
        double m[4 * 3], t[4 * 3];
        unsigned char sw[4 * 3];
        tw_fill_ct_ds_split(m, t, sw, 20, 4, 5);          /* N=20, r=4, m=5 */
        if (tw_audit_ct_ds_split(m, t, sw, 20, 4, 5) > 0.51) return 0;
    }
    return 1;
}

#endif /* GEN_TWIDDLE_C_INCLUDED */

/* =============================================================================
 * DEMO ENTRY: any-L mixed-radix row-column 3D FFT, every table from the layer,
 * audited and 1-D-gated at create().  Compiled only when this file is the
 * entry translation unit.
 *
 * gen_r2: the 8-pencil split-complex lanes are now explicit 64-byte GNU
 * vector types (one zmm per lane-row) instead of double[8] loops -- gcc was
 * auto-vectorizing those to ymm only (gen_bluestein's r1 lesson: explicit
 * 512-bit is the only reliable path under the fixed harness flags).  Gathers
 * and scatters de/interleave with two-source shuffles; the z-axis uses an
 * 8x8 in-register transpose.  Arithmetic order per butterfly is unchanged.
 * ============================================================================= */
#ifndef GEN_TWIDDLE_LIB_ONLY

#define GEN_LAYOUT_LIB_ONLY /* adopt gen_layout's THP arena for plan memory */
#include "gen_layout.c"

#include <stdlib.h>

#include "../fft3d_api.h"

#define TWD_MAXLEV 8   /* 4,4,4,2 covers 128; deepest chain is 2^7 */
#define TWD_MAXR 128   /* leaves (m == 1) may be any prime <= 127; the combine
                        * radix of a composite is 4 or a prime <= 11 */
#define TWD_MAXH 63    /* fold half-system rows: h = (r-1)/2 <= 63 at r <= 127 */

/* one zmm of lanes; may_alias so casting the split double buffers is legal */
typedef double tw_v8 __attribute__((vector_size(64), aligned(64), may_alias));
typedef long long tw_v8l __attribute__((vector_size(64)));

#if defined(__AVX512F__)
#include <immintrin.h>   /* masked seam stores (twd_scatter_gf) + map ladder */
#endif

static inline tw_v8 tw_loadu8(const double *q)      /* unaligned load  */
{
    tw_v8 v;
    __builtin_memcpy(&v, q, sizeof v);
    return v;
}
static inline void tw_storeu8(double *q, tw_v8 v)   /* unaligned store */
{
    __builtin_memcpy(q, &v, sizeof v);
}

/* 8x8 double transpose, B[i][j] = A[j][i], 24 two-source shuffles */
static inline void tw_tr8x8(tw_v8 *B, const tw_v8 *A)
{
    tw_v8 t0 = __builtin_shuffle(A[0], A[1], (tw_v8l){0, 8, 2, 10, 4, 12, 6, 14});
    tw_v8 t1 = __builtin_shuffle(A[0], A[1], (tw_v8l){1, 9, 3, 11, 5, 13, 7, 15});
    tw_v8 t2 = __builtin_shuffle(A[2], A[3], (tw_v8l){0, 8, 2, 10, 4, 12, 6, 14});
    tw_v8 t3 = __builtin_shuffle(A[2], A[3], (tw_v8l){1, 9, 3, 11, 5, 13, 7, 15});
    tw_v8 t4 = __builtin_shuffle(A[4], A[5], (tw_v8l){0, 8, 2, 10, 4, 12, 6, 14});
    tw_v8 t5 = __builtin_shuffle(A[4], A[5], (tw_v8l){1, 9, 3, 11, 5, 13, 7, 15});
    tw_v8 t6 = __builtin_shuffle(A[6], A[7], (tw_v8l){0, 8, 2, 10, 4, 12, 6, 14});
    tw_v8 t7 = __builtin_shuffle(A[6], A[7], (tw_v8l){1, 9, 3, 11, 5, 13, 7, 15});
    tw_v8 u0 = __builtin_shuffle(t0, t2, (tw_v8l){0, 1, 8, 9, 4, 5, 12, 13});
    tw_v8 u2 = __builtin_shuffle(t0, t2, (tw_v8l){2, 3, 10, 11, 6, 7, 14, 15});
    tw_v8 u1 = __builtin_shuffle(t1, t3, (tw_v8l){0, 1, 8, 9, 4, 5, 12, 13});
    tw_v8 u3 = __builtin_shuffle(t1, t3, (tw_v8l){2, 3, 10, 11, 6, 7, 14, 15});
    tw_v8 u4 = __builtin_shuffle(t4, t6, (tw_v8l){0, 1, 8, 9, 4, 5, 12, 13});
    tw_v8 u6 = __builtin_shuffle(t4, t6, (tw_v8l){2, 3, 10, 11, 6, 7, 14, 15});
    tw_v8 u5 = __builtin_shuffle(t5, t7, (tw_v8l){0, 1, 8, 9, 4, 5, 12, 13});
    tw_v8 u7 = __builtin_shuffle(t5, t7, (tw_v8l){2, 3, 10, 11, 6, 7, 14, 15});
    B[0] = __builtin_shuffle(u0, u4, (tw_v8l){0, 1, 2, 3, 8, 9, 10, 11});
    B[4] = __builtin_shuffle(u0, u4, (tw_v8l){4, 5, 6, 7, 12, 13, 14, 15});
    B[1] = __builtin_shuffle(u1, u5, (tw_v8l){0, 1, 2, 3, 8, 9, 10, 11});
    B[5] = __builtin_shuffle(u1, u5, (tw_v8l){4, 5, 6, 7, 12, 13, 14, 15});
    B[2] = __builtin_shuffle(u2, u6, (tw_v8l){0, 1, 2, 3, 8, 9, 10, 11});
    B[6] = __builtin_shuffle(u2, u6, (tw_v8l){4, 5, 6, 7, 12, 13, 14, 15});
    B[3] = __builtin_shuffle(u3, u7, (tw_v8l){0, 1, 2, 3, 8, 9, 10, 11});
    B[7] = __builtin_shuffle(u3, u7, (tw_v8l){4, 5, 6, 7, 12, 13, 14, 15});
}

#if defined(__AVX512F__)
/* gen_r10, ADOPTED from gen_pow2 gen_r9 (extract-to-memory): the LAST stage
 * of tw_tr8x8 only ever moves aligned 256-bit halves -- B[k] = [U_k lo|U_k+4
 * lo], B[k+4] = [U_k hi|U_k+4 hi] -- and at the transpose-into-store sites
 * (twd_gather_z lane-buffer fill, twd_scatter_z, twd_scatter_gf) its only
 * consumer IS the store.  So the stage becomes 16 x 256-bit stores: vmovupd
 * ymm for lows, vextractf64x4-TO-MEMORY for highs -- pure store-port ops
 * (p237+p4, no p5 uop on ICL/SPR per uops.info).  Per transpose: -8 port-5
 * shuffles, front-end and store-bandwidth neutral (256-bit stores retire
 * 2/cyc), bytes at every address IDENTICAL (bit-identity preserved).
 * tw_tr8x8_u is tw_tr8x8's first two stages verbatim. */
static inline void tw_tr8x8_u(tw_v8 *U, const tw_v8 *A)
{
    tw_v8 t0 = __builtin_shuffle(A[0], A[1], (tw_v8l){0, 8, 2, 10, 4, 12, 6, 14});
    tw_v8 t1 = __builtin_shuffle(A[0], A[1], (tw_v8l){1, 9, 3, 11, 5, 13, 7, 15});
    tw_v8 t2 = __builtin_shuffle(A[2], A[3], (tw_v8l){0, 8, 2, 10, 4, 12, 6, 14});
    tw_v8 t3 = __builtin_shuffle(A[2], A[3], (tw_v8l){1, 9, 3, 11, 5, 13, 7, 15});
    tw_v8 t4 = __builtin_shuffle(A[4], A[5], (tw_v8l){0, 8, 2, 10, 4, 12, 6, 14});
    tw_v8 t5 = __builtin_shuffle(A[4], A[5], (tw_v8l){1, 9, 3, 11, 5, 13, 7, 15});
    tw_v8 t6 = __builtin_shuffle(A[6], A[7], (tw_v8l){0, 8, 2, 10, 4, 12, 6, 14});
    tw_v8 t7 = __builtin_shuffle(A[6], A[7], (tw_v8l){1, 9, 3, 11, 5, 13, 7, 15});
    U[0] = __builtin_shuffle(t0, t2, (tw_v8l){0, 1, 8, 9, 4, 5, 12, 13});
    U[2] = __builtin_shuffle(t0, t2, (tw_v8l){2, 3, 10, 11, 6, 7, 14, 15});
    U[1] = __builtin_shuffle(t1, t3, (tw_v8l){0, 1, 8, 9, 4, 5, 12, 13});
    U[3] = __builtin_shuffle(t1, t3, (tw_v8l){2, 3, 10, 11, 6, 7, 14, 15});
    U[4] = __builtin_shuffle(t4, t6, (tw_v8l){0, 1, 8, 9, 4, 5, 12, 13});
    U[6] = __builtin_shuffle(t4, t6, (tw_v8l){2, 3, 10, 11, 6, 7, 14, 15});
    U[5] = __builtin_shuffle(t5, t7, (tw_v8l){0, 1, 8, 9, 4, 5, 12, 13});
    U[7] = __builtin_shuffle(t5, t7, (tw_v8l){2, 3, 10, 11, 6, 7, 14, 15});
}
#define TW_STLO(pp, v) _mm256_storeu_pd((pp), _mm512_castpd512_pd256((__m512d)(v)))
#define TW_STHI(pp, v) _mm256_storeu_pd((pp), _mm512_extractf64x4_pd((__m512d)(v), 1))
#endif

struct twd_level {
    int n, r, m;               /* n = r*m; r = 4 if 4|n else smallest prime  */
    double *twr, *twi;         /* w_n^(s*k1), k1-major, s inner, (r-1)*m each */
    double *dm, *dt;           /* dual-select twiddles, same indexing (gen_r5:
                                * scale m, ratio |t| <= 1; -DTWD_DS consumes) */
    unsigned char *dsw;        /* the per-twiddle Linzer-Feig select flags    */
    double *Rr, *Ri;           /* dense r x r combine matrix w_r^(k2*s)
                                * (r >= 7 only; -DTWD_DENSEBF control arm)    */
    double *Cf, *Sf;           /* fold half-system C/S[h][h] (r >= 7, odd r:
                                * tw_fill_fold_half with hp = h; the gen_r5
                                * butterfly, ~4h^2 FMAs vs the dense 16h^2)   */
};

struct fft3d_plan {
    int L, batch, nlev;
    int kblk;                  /* planes per axes-1+2 custody block: 8/gcd(L,8),
                                * so blocks hold kblk*L rows == 0 (mod 8) and the
                                * axis-2 group decomposition is IDENTICAL to the
                                * unblocked pass (gen_bluestein r4's block size) */
    int gf;                    /* gen_r8: use the split-group handoff (GB) for
                                * axes 1->2; plan-gated to large L, see create() */
    int chain_fb;              /* create()'s chain gate failed: use the exact
                                * execute + scalar-map fallback inside chain */
    struct twd_level lev[TWD_MAXLEV];
    double s3;                 /* butterfly constants, from tw_cis: sin(2pi/3) */
    double c15, s15, c25, s25; /* cos/sin(2pi/5), cos/sin(4pi/5)               */
    double c8;                 /* cos(pi/4) = sqrt(1/2), the W8 magnitude
                                * (gen_r10 radix-8 levels)                     */
    double _Complex *T;        /* interleaved intermediate volume             */
    double *Gr, *Gi, *Sr, *Si, *Yr, *Yi; /* 8-wide split pencil buffers, 8L each */
    double *GB;                /* NEW gen_r8: slab-sized group-format handoff
                                * buffer for axes 1->2 (kblk*L*L complex): one
                                * L-row block per axis-2 8-row group, row j =
                                * [re v8][im v8], so axis 2 direct-feeds with
                                * xstr = 2 -- zero gather, zero shuffles      */
    gl_arena ar;
};

const char *fft3d_name(void) { return "gen_twiddle"; }
const char *fft3d_description(void)
{
    return "LIBRARY LAYER (scored by adoption): octant-folded exact twiddles <=0.51 ulp "
           "(tw_cis/tw_chirp) + NEW dual-select FMA form tw_cis_ds (lit 11 Tier 1: every "
           "stored ratio <=1, first performant validation), consumption-order CT/DFT/"
           "Rader(+folded)/chirp/fold-half/SIMD-dense fillers + ulp audits + primitive "
           "roots + long-double DFT oracle (adopt: #define GEN_TWIDDLE_LIB_ONLY + #include "
           "gen_twiddle.c); entry = any-L mixed-radix zmm-lane demo (gen_r5: conjugate-fold "
           "prime butterflies; gen_r6: register-resident whole-level codelets for r=2/3/4/5, "
           "-30..-44% bit-identical; gen_r7: fused fold-combine codelet for combine radices "
           "7/11/13; gen_r8: shuffle-free split-group handoff axes 1->2, axis-2 gather "
           "deleted, bit-identical; gen_r9: axis-1 pencil groups PACKED across plane "
           "seams via two-pointer masked gather/scatter + masked w<8 tails, per-plane "
           "scalar z-tails deleted, bit-identical; gen_r10: radix-8 levels -- one whole "
           "combine pass deleted at 8|L -- + transpose last stage as extract-to-memory "
           "stores, -8 p5/tr8x8, from gen_pow2 r9; gen_r11: radix-10 levels via PFA 2x5 "
           "DFT10 codelets, zero internal twiddles -- depth-minimizing factorizer deletes "
           "one whole level pass at 10/30/50/70/90/100/110, tw muls 155->90 per group at "
           "100), self-audited at create(), owned in-place fused-map chain";
}
int fft3d_supports(int L) { return L >= 2 && L <= 128; }

/* factorization chain (gen_r10: radix 8 preferred -- one whole combine level
 * deleted at 32/40/64/96/128; the r6 lesson says level passes ARE the cost).
 * NEW gen_r11: radix 10 joins the candidate set {8, 4, smallest prime, 10}
 * and the chain is chosen by MINIMIZING LEVEL COUNT, ties broken in the old
 * preference order with 10 LAST -- so every size whose old chain was already
 * depth-minimal factors EXACTLY as before (bit-identical outputs), and only
 * sizes where a 10-level strictly deletes a pass change: 10 (2 levels -> one
 * leaf10), 30/50/70/110 (3 -> 2), 100 (4*25's 3 -> 10*10's 2), 90 (4 -> 3),
 * 80 (8*(2*5) -> 8*leaf10).  DFT10 is PFA 2x5 (coprime factors: ZERO internal
 * twiddles), so the deleted level's twiddle multiplies go too -- at 100 the
 * per-group count drops 155 -> 90. */
static int twd_depth(int n)
{
    if (n <= 1) return 0;
    int p = n;
    for (int d = 2; d * d <= n; ++d)
        if (n % d == 0) { p = d; break; }
    int best = 1 + twd_depth(n / p);            /* smallest prime always works */
    if (n % 8 == 0) { int c = 1 + twd_depth(n / 8); if (c < best) best = c; }
    if (n % 4 == 0) { int c = 1 + twd_depth(n / 4); if (c < best) best = c; }
    if (n % 10 == 0) { int c = 1 + twd_depth(n / 10); if (c < best) best = c; }
    return best;
}

static int twd_factor(int L, struct twd_level *lev)
{
    int n = L, nl = 0;
    while (n > 1) {
        if (nl >= TWD_MAXLEV) return 0;
        int p = n;
        for (int d = 2; d * d <= n; ++d)
            if (n % d == 0) { p = d; break; }
        /* old preference order first, 10 last: 10 wins only on strict depth */
        int r = 0, bd = 1 << 20;
        if (n % 8 == 0) { r = 8; bd = 1 + twd_depth(n / 8); }
        if (n % 4 == 0) { int c = 1 + twd_depth(n / 4); if (c < bd) { r = 4; bd = c; } }
        { int c = 1 + twd_depth(n / p); if (c < bd) { r = p; bd = c; } }
        if (n % 10 == 0) { int c = 1 + twd_depth(n / 10); if (c < bd) { r = 10; bd = c; } }
        lev[nl].n = n;
        lev[nl].r = r;
        lev[nl].m = n / r;
        if (lev[nl].m > 1 && r > 13) return 0; /* composite: 4/8/10 or prime <= 13 */
        ++nl;
        n /= r;
        if (n == 1) break;
    }
    return nl;
}

/* ---- the 8-wide split mixed-radix DIT recursion ---------------------------- */
/* odd r >= 7: conjugate-fold half-system (NEW gen_r5) -- the gen_layout /
 * gen_dense_prime / gen_rader fold shape, tables from this layer's own
 * tw_fill_fold_half.  X_k = x0 + sum_j (a_j*C_kj - i*b_j*S_kj) with
 * a_j = x_j + x_{r-j}, b_j = x_j - x_{r-j}, and X_{r-k} the conjugate-signed
 * combine: ~4h^2 FMAs + O(r) adds instead of the dense 16h^2, h = (r-1)/2.
 * Two k-rows at a time = 8 independent FMA chains, the old dense unroll's
 * ILP shape.  A SEPARATE noinline function on purpose: inlined into
 * twd_butterfly it bloats the switch and gcc then de-inlines the r = 2..5
 * hot cases from twd_rec -- measured +1-4.5% on the NO-fold sizes
 * (12/50/100) with bit-identical outputs before this split. */
#ifndef TWD_DENSEBF
static __attribute__((noinline, aligned(64))) void twd_fold_bf(const struct twd_level *lv,
                                                  const tw_v8 *tr, const tw_v8 *ti,
                                                  tw_v8 *yr, tw_v8 *yi, long st)
{
    const int r = lv->r, h = (r - 1) / 2;
    const double *Cf = lv->Cf, *Sf = lv->Sf;
    tw_v8 fr[2 * TWD_MAXH], fi[2 * TWD_MAXH];   /* a rows, then b rows */
    tw_v8 s0r = tr[0], s0i = ti[0];
    for (int j = 1; j <= h; ++j) {
        fr[j - 1] = tr[j] + tr[r - j];
        fi[j - 1] = ti[j] + ti[r - j];
        fr[h + j - 1] = tr[j] - tr[r - j];
        fi[h + j - 1] = ti[j] - ti[r - j];
        s0r += fr[j - 1];
        s0i += fi[j - 1];
    }
    yr[0] = s0r;
    yi[0] = s0i;
    int k = 1;
    for (; k + 1 <= h; k += 2) {
        const double *C0 = Cf + (size_t)(k - 1) * h, *S0 = Sf + (size_t)(k - 1) * h;
        const double *C1 = C0 + h, *S1 = S0 + h;
        tw_v8 e0r = tr[0], e0i = ti[0], e1r = tr[0], e1i = ti[0];
        tw_v8 o0r = { 0 }, o0i = { 0 }, o1r = { 0 }, o1i = { 0 };
        for (int j = 0; j < h; ++j) {
            const tw_v8 axr = fr[j], axi = fi[j];
            const tw_v8 bxr = fr[h + j], bxi = fi[h + j];
            const double c0 = C0[j], s0 = S0[j], c1 = C1[j], s1 = S1[j];
            e0r += c0 * axr;
            e0i += c0 * axi;
            o0r += s0 * bxr;
            o0i += s0 * bxi;
            e1r += c1 * axr;
            e1i += c1 * axi;
            o1r += s1 * bxr;
            o1i += s1 * bxi;
        }
        yr[(long)k * st] = e0r + o0i;
        yi[(long)k * st] = e0i - o0r;
        yr[(long)(r - k) * st] = e0r - o0i;
        yi[(long)(r - k) * st] = e0i + o0r;
        yr[(long)(k + 1) * st] = e1r + o1i;
        yi[(long)(k + 1) * st] = e1i - o1r;
        yr[(long)(r - k - 1) * st] = e1r - o1i;
        yi[(long)(r - k - 1) * st] = e1i + o1r;
    }
    for (; k <= h; ++k) {
        const double *C0 = Cf + (size_t)(k - 1) * h, *S0 = Sf + (size_t)(k - 1) * h;
        tw_v8 er = tr[0], ei = ti[0], our = { 0 }, oui = { 0 };
        for (int j = 0; j < h; ++j) {
            const double c0 = C0[j], s0 = S0[j];
            er += c0 * fr[j];
            ei += c0 * fi[j];
            our += s0 * fr[h + j];
            oui += s0 * fi[h + j];
        }
        yr[(long)k * st] = er + oui;
        yi[(long)k * st] = ei - our;
        yr[(long)(r - k) * st] = er - oui;
        yi[(long)(r - k) * st] = ei + our;
    }
}
#endif /* !TWD_DENSEBF */

/* ---- specialized register-resident levels for r = 2,3,4,5 (NEW gen_r6) ------
 * twd_butterfly is NOT inlined into twd_rec (asm-audited: 6 call sites
 * survive -O3), so every leaf and every combine k1-iteration was paying a
 * real call plus a round trip of 2r zmm rows through the tr/ti stack arrays
 * (8 stores + 8 reloads at r = 4) before any arithmetic ran.  These
 * functions run a WHOLE level -- leaf, or the full k1 = 0..m-1 combine --
 * inside one noinline call with every row held in registers: no staging, no
 * per-k1 dispatch.  The expressions are twd_butterfly's cases and the
 * generic combine's twiddle multiply copied VERBATIM (same temporaries, same
 * order), so gcc's FMA contraction is identical and outputs are
 * BIT-IDENTICAL to gen_r5 (verified by cmp on the node).  Separate noinline
 * functions on purpose -- the gen_r5 case-bloat lesson: keep heavy bodies
 * out of the recursive dispatcher, which also shrinks twd_rec's own frame.
 * Under -DTWD_DS the combines fall back to the generic loop (the DS knob is
 * raced, default off); leaves have no twiddles and specialize either way. */
static __attribute__((noinline, aligned(64))) void twd_leaf2(const double *xr, const double *xi,
                                                long xstr, tw_v8 *Yr, tw_v8 *Yi)
{
    tw_v8 t0r = *(const tw_v8 *)xr, t0i = *(const tw_v8 *)xi;
    tw_v8 t1r = *(const tw_v8 *)(xr + 8l * xstr), t1i = *(const tw_v8 *)(xi + 8l * xstr);
    Yr[0] = t0r + t1r;
    Yi[0] = t0i + t1i;
    Yr[1] = t0r - t1r;
    Yi[1] = t0i - t1i;
}

static __attribute__((noinline, aligned(64))) void twd_leaf3(const struct fft3d_plan *p,
                                                const double *xr, const double *xi,
                                                long xstr, tw_v8 *Yr, tw_v8 *Yi)
{
    const double s3 = p->s3;
    tw_v8 t0r = *(const tw_v8 *)xr, t0i = *(const tw_v8 *)xi;
    tw_v8 t1r = *(const tw_v8 *)(xr + 8l * xstr), t1i = *(const tw_v8 *)(xi + 8l * xstr);
    tw_v8 t2r = *(const tw_v8 *)(xr + 16l * xstr), t2i = *(const tw_v8 *)(xi + 16l * xstr);
    tw_v8 sr_ = t1r + t2r, si_ = t1i + t2i;
    tw_v8 vr = t0r - 0.5 * sr_, vi = t0i - 0.5 * si_;
    tw_v8 wr = s3 * (t1r - t2r), wi = s3 * (t1i - t2i);
    Yr[0] = t0r + sr_;
    Yi[0] = t0i + si_;
    Yr[1] = vr + wi;
    Yi[1] = vi - wr;
    Yr[2] = vr - wi;
    Yi[2] = vi + wr;
}

static __attribute__((noinline, aligned(64))) void twd_leaf4(const double *xr, const double *xi,
                                                long xstr, tw_v8 *Yr, tw_v8 *Yi)
{
    tw_v8 t0r = *(const tw_v8 *)xr, t0i = *(const tw_v8 *)xi;
    tw_v8 t1r = *(const tw_v8 *)(xr + 8l * xstr), t1i = *(const tw_v8 *)(xi + 8l * xstr);
    tw_v8 t2r = *(const tw_v8 *)(xr + 16l * xstr), t2i = *(const tw_v8 *)(xi + 16l * xstr);
    tw_v8 t3r = *(const tw_v8 *)(xr + 24l * xstr), t3i = *(const tw_v8 *)(xi + 24l * xstr);
    tw_v8 ar = t0r + t2r, ai = t0i + t2i;
    tw_v8 br = t0r - t2r, bi = t0i - t2i;
    tw_v8 cr = t1r + t3r, ci = t1i + t3i;
    tw_v8 dr = t1r - t3r, di = t1i - t3i;
    Yr[0] = ar + cr;
    Yi[0] = ai + ci;
    Yr[1] = br + di;
    Yi[1] = bi - dr;
    Yr[2] = ar - cr;
    Yi[2] = ai - ci;
    Yr[3] = br - di;
    Yi[3] = bi + dr;
}

static __attribute__((noinline, aligned(64))) void twd_leaf5(const struct fft3d_plan *p,
                                                const double *xr, const double *xi,
                                                long xstr, tw_v8 *Yr, tw_v8 *Yi)
{
    const double c1 = p->c15, s1 = p->s15, c2 = p->c25, s2 = p->s25;
    tw_v8 t0r = *(const tw_v8 *)xr, t0i = *(const tw_v8 *)xi;
    tw_v8 x1r = *(const tw_v8 *)(xr + 8l * xstr), x1i = *(const tw_v8 *)(xi + 8l * xstr);
    tw_v8 x2r = *(const tw_v8 *)(xr + 16l * xstr), x2i = *(const tw_v8 *)(xi + 16l * xstr);
    tw_v8 x3r = *(const tw_v8 *)(xr + 24l * xstr), x3i = *(const tw_v8 *)(xi + 24l * xstr);
    tw_v8 x4r = *(const tw_v8 *)(xr + 32l * xstr), x4i = *(const tw_v8 *)(xi + 32l * xstr);
    tw_v8 t1r = x1r + x4r, t1i = x1i + x4i;
    tw_v8 t2r = x2r + x3r, t2i = x2i + x3i;
    tw_v8 t3r = x1r - x4r, t3i = x1i - x4i;
    tw_v8 t4r = x2r - x3r, t4i = x2i - x3i;
    Yr[0] = t0r + t1r + t2r;
    Yi[0] = t0i + t1i + t2i;
    tw_v8 a1r = t0r + c1 * t1r + c2 * t2r;
    tw_v8 a1i = t0i + c1 * t1i + c2 * t2i;
    tw_v8 b1r = s1 * t3r + s2 * t4r, b1i = s1 * t3i + s2 * t4i;
    tw_v8 a2r = t0r + c2 * t1r + c1 * t2r;
    tw_v8 a2i = t0i + c2 * t1i + c1 * t2i;
    tw_v8 b2r = s2 * t3r - s1 * t4r, b2i = s2 * t3i - s1 * t4i;
    Yr[1] = a1r + b1i;
    Yi[1] = a1i - b1r;
    Yr[4] = a1r - b1i;
    Yi[4] = a1i + b1r;
    Yr[2] = a2r + b2i;
    Yi[2] = a2i - b2r;
    Yr[3] = a2r - b2i;
    Yi[3] = a2i + b2r;
}

/* NEW gen_r10: radix-8 codelets -- DFT8 as even/odd DFT4s + a W8 combine,
 * all in registers (the only non-trivial constant is c8 = sqrt(1/2), from
 * tw_cis(1,8), correctly rounded).  Sign convention as twd_leaf4:
 * X_k = E_k + W8^k O_k, X_{k+4} = E_k - W8^k O_k, W8 = (1-i)*c8.
 * DEFINITIONS AT FILE END on purpose: emission order = the r9 hot text
 * unchanged, new code appended after it (first build had them here and read
 * +1-3% at L=12, 4/4 node pairs, on bit-identical output -- the r5/r8
 * code-layout tax again). */
static void twd_leaf8(const struct fft3d_plan *p,
                      const double *xr, const double *xi,
                      long xstr, tw_v8 *Yr, tw_v8 *Yi);
#ifndef TWD_DS
static void twd_comb8(const struct fft3d_plan *p, const struct twd_level *lv,
                      const double *sr, const double *si,
                      double *yr, double *yi);
#endif

/* NEW gen_r11: radix-10 codelets -- DFT10 as PFA 2x5 (Good-Thomas over the
 * coprime pair, so ZERO twiddles and zero negations inside the butterfly:
 * 5 DFT2s on pairs {k, k+5} with the CRT-even member leading, then two of
 * twd_leaf5's DFT5 bodies; even outputs Y[2j] = DFT5(s)[j], odd outputs
 * Y[(5+2j) mod 10] = DFT5(d)[j]).  Same file-end placement discipline as
 * the r10 radix-8 codelets (declarations here, definitions after the hot
 * text -- the r5/r8/r10 code-layout tax lesson). */
static void twd_leaf10(const struct fft3d_plan *p,
                       const double *xr, const double *xi,
                       long xstr, tw_v8 *Yr, tw_v8 *Yi);
#ifndef TWD_DS
static void twd_comb10(const struct fft3d_plan *p, const struct twd_level *lv,
                       const double *sr, const double *si,
                       double *yr, double *yi);
#endif

#ifndef TWD_DS
static __attribute__((noinline, aligned(64))) void twd_comb2(const struct twd_level *lv,
                                                const double *sr, const double *si,
                                                double *yr, double *yi)
{
    const int m = lv->m;
    const tw_v8 *S_r = (const tw_v8 *)sr, *S_i = (const tw_v8 *)si;
    tw_v8 *Y_r = (tw_v8 *)yr, *Y_i = (tw_v8 *)yi;
    {
        tw_v8 t0r = S_r[0], t0i = S_i[0];
        tw_v8 t1r = S_r[m], t1i = S_i[m];
        Y_r[0] = t0r + t1r;
        Y_i[0] = t0i + t1i;
        Y_r[m] = t0r - t1r;
        Y_i[m] = t0i - t1i;
    }
    for (int k1 = 1; k1 < m; ++k1) {
        const double wr = lv->twr[k1], wi = lv->twi[k1];
        tw_v8 t0r = S_r[k1], t0i = S_i[k1];
        const tw_v8 ur = S_r[m + k1], ui = S_i[m + k1];
        tw_v8 t1r = wr * ur - wi * ui;
        tw_v8 t1i = wr * ui + wi * ur;
        Y_r[k1] = t0r + t1r;
        Y_i[k1] = t0i + t1i;
        Y_r[m + k1] = t0r - t1r;
        Y_i[m + k1] = t0i - t1i;
    }
}

static __attribute__((noinline, aligned(64))) void twd_comb3(const struct fft3d_plan *p,
                                                const struct twd_level *lv,
                                                const double *sr, const double *si,
                                                double *yr, double *yi)
{
    const int m = lv->m;
    const double s3 = p->s3;
    const tw_v8 *S_r = (const tw_v8 *)sr, *S_i = (const tw_v8 *)si;
    tw_v8 *Y_r = (tw_v8 *)yr, *Y_i = (tw_v8 *)yi;
    for (int k1 = 0; k1 < m; ++k1) {
        tw_v8 t0r = S_r[k1], t0i = S_i[k1];
        tw_v8 t1r, t1i, t2r, t2i;
        if (k1 == 0) {
            t1r = S_r[m];
            t1i = S_i[m];
            t2r = S_r[2 * m];
            t2i = S_i[2 * m];
        } else {
            const double *twr = lv->twr + (size_t)k1 * 2;
            const double *twi = lv->twi + (size_t)k1 * 2;
            const tw_v8 u1r = S_r[m + k1], u1i = S_i[m + k1];
            const tw_v8 u2r = S_r[2 * m + k1], u2i = S_i[2 * m + k1];
            t1r = twr[0] * u1r - twi[0] * u1i;
            t1i = twr[0] * u1i + twi[0] * u1r;
            t2r = twr[1] * u2r - twi[1] * u2i;
            t2i = twr[1] * u2i + twi[1] * u2r;
        }
        tw_v8 sr_ = t1r + t2r, si_ = t1i + t2i;
        tw_v8 vr = t0r - 0.5 * sr_, vi = t0i - 0.5 * si_;
        tw_v8 wr = s3 * (t1r - t2r), wi = s3 * (t1i - t2i);
        Y_r[k1] = t0r + sr_;
        Y_i[k1] = t0i + si_;
        Y_r[m + k1] = vr + wi;
        Y_i[m + k1] = vi - wr;
        Y_r[2 * m + k1] = vr - wi;
        Y_i[2 * m + k1] = vi + wr;
    }
}

static __attribute__((noinline, aligned(64))) void twd_comb4(const struct twd_level *lv,
                                                const double *sr, const double *si,
                                                double *yr, double *yi)
{
    const int m = lv->m;
    const tw_v8 *S_r = (const tw_v8 *)sr, *S_i = (const tw_v8 *)si;
    tw_v8 *Y_r = (tw_v8 *)yr, *Y_i = (tw_v8 *)yi;
    for (int k1 = 0; k1 < m; ++k1) {
        tw_v8 t0r = S_r[k1], t0i = S_i[k1];
        tw_v8 t1r, t1i, t2r, t2i, t3r, t3i;
        if (k1 == 0) {
            t1r = S_r[m];
            t1i = S_i[m];
            t2r = S_r[2 * m];
            t2i = S_i[2 * m];
            t3r = S_r[3 * m];
            t3i = S_i[3 * m];
        } else {
            const double *twr = lv->twr + (size_t)k1 * 3;
            const double *twi = lv->twi + (size_t)k1 * 3;
            const tw_v8 u1r = S_r[m + k1], u1i = S_i[m + k1];
            const tw_v8 u2r = S_r[2 * m + k1], u2i = S_i[2 * m + k1];
            const tw_v8 u3r = S_r[3 * m + k1], u3i = S_i[3 * m + k1];
            t1r = twr[0] * u1r - twi[0] * u1i;
            t1i = twr[0] * u1i + twi[0] * u1r;
            t2r = twr[1] * u2r - twi[1] * u2i;
            t2i = twr[1] * u2i + twi[1] * u2r;
            t3r = twr[2] * u3r - twi[2] * u3i;
            t3i = twr[2] * u3i + twi[2] * u3r;
        }
        tw_v8 ar = t0r + t2r, ai = t0i + t2i;
        tw_v8 br = t0r - t2r, bi = t0i - t2i;
        tw_v8 cr = t1r + t3r, ci = t1i + t3i;
        tw_v8 dr = t1r - t3r, di = t1i - t3i;
        Y_r[k1] = ar + cr;
        Y_i[k1] = ai + ci;
        Y_r[m + k1] = br + di;
        Y_i[m + k1] = bi - dr;
        Y_r[2 * m + k1] = ar - cr;
        Y_i[2 * m + k1] = ai - ci;
        Y_r[3 * m + k1] = br - di;
        Y_i[3 * m + k1] = bi + dr;
    }
}

static __attribute__((noinline, aligned(64))) void twd_comb5(const struct fft3d_plan *p,
                                                const struct twd_level *lv,
                                                const double *sr, const double *si,
                                                double *yr, double *yi)
{
    const int m = lv->m;
    const double c1 = p->c15, s1 = p->s15, c2 = p->c25, s2 = p->s25;
    const tw_v8 *S_r = (const tw_v8 *)sr, *S_i = (const tw_v8 *)si;
    tw_v8 *Y_r = (tw_v8 *)yr, *Y_i = (tw_v8 *)yi;
    for (int k1 = 0; k1 < m; ++k1) {
        tw_v8 t0r = S_r[k1], t0i = S_i[k1];
        tw_v8 x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i;
        if (k1 == 0) {
            x1r = S_r[m];
            x1i = S_i[m];
            x2r = S_r[2 * m];
            x2i = S_i[2 * m];
            x3r = S_r[3 * m];
            x3i = S_i[3 * m];
            x4r = S_r[4 * m];
            x4i = S_i[4 * m];
        } else {
            const double *twr = lv->twr + (size_t)k1 * 4;
            const double *twi = lv->twi + (size_t)k1 * 4;
            const tw_v8 u1r = S_r[m + k1], u1i = S_i[m + k1];
            const tw_v8 u2r = S_r[2 * m + k1], u2i = S_i[2 * m + k1];
            const tw_v8 u3r = S_r[3 * m + k1], u3i = S_i[3 * m + k1];
            const tw_v8 u4r = S_r[4 * m + k1], u4i = S_i[4 * m + k1];
            x1r = twr[0] * u1r - twi[0] * u1i;
            x1i = twr[0] * u1i + twi[0] * u1r;
            x2r = twr[1] * u2r - twi[1] * u2i;
            x2i = twr[1] * u2i + twi[1] * u2r;
            x3r = twr[2] * u3r - twi[2] * u3i;
            x3i = twr[2] * u3i + twi[2] * u3r;
            x4r = twr[3] * u4r - twi[3] * u4i;
            x4i = twr[3] * u4i + twi[3] * u4r;
        }
        tw_v8 t1r = x1r + x4r, t1i = x1i + x4i;
        tw_v8 t2r = x2r + x3r, t2i = x2i + x3i;
        tw_v8 t3r = x1r - x4r, t3i = x1i - x4i;
        tw_v8 t4r = x2r - x3r, t4i = x2i - x3i;
        Y_r[k1] = t0r + t1r + t2r;
        Y_i[k1] = t0i + t1i + t2i;
        tw_v8 a1r = t0r + c1 * t1r + c2 * t2r;
        tw_v8 a1i = t0i + c1 * t1i + c2 * t2i;
        tw_v8 b1r = s1 * t3r + s2 * t4r, b1i = s1 * t3i + s2 * t4i;
        tw_v8 a2r = t0r + c2 * t1r + c1 * t2r;
        tw_v8 a2i = t0i + c2 * t1i + c1 * t2i;
        tw_v8 b2r = s2 * t3r - s1 * t4r, b2i = s2 * t3i - s1 * t4i;
        Y_r[m + k1] = a1r + b1i;
        Y_i[m + k1] = a1i - b1r;
        Y_r[4 * m + k1] = a1r - b1i;
        Y_i[4 * m + k1] = a1i + b1r;
        Y_r[2 * m + k1] = a2r + b2i;
        Y_i[2 * m + k1] = a2i - b2r;
        Y_r[3 * m + k1] = a2r - b2i;
        Y_i[3 * m + k1] = a2i + b2r;
    }
}
#endif /* !TWD_DS */

/* forward DFT_r of the lane-rows in tr/ti, results to y rows k2 at k2*st (v8
 * units): exact-constant butterflies for r = 2,3,4,5 (constants from tw_cis,
 * held in the plan); conjugate-fold half-system otherwise (odd prime leaves
 * and combine radices 7/11/13; -DTWD_DENSEBF races the old dense form).
 * From gen_r6 the r = 2,3,4,5 leaf/combine hot paths never reach this
 * function (twd_leafN / twd_combN above); it remains the r >= 7 path and
 * the generic combine's kernel. */
static void twd_butterfly(const struct fft3d_plan *p, const struct twd_level *lv,
                          const tw_v8 *tr, const tw_v8 *ti,
                          tw_v8 *yr, tw_v8 *yi, long st)
{
    const int r = lv->r;
    switch (r) {
    case 2:
        yr[0] = tr[0] + tr[1];
        yi[0] = ti[0] + ti[1];
        yr[st] = tr[0] - tr[1];
        yi[st] = ti[0] - ti[1];
        return;
    case 3: {
        const double s3 = p->s3;
        tw_v8 sr_ = tr[1] + tr[2], si_ = ti[1] + ti[2];
        tw_v8 vr = tr[0] - 0.5 * sr_, vi = ti[0] - 0.5 * si_;
        tw_v8 wr = s3 * (tr[1] - tr[2]), wi = s3 * (ti[1] - ti[2]);
        yr[0] = tr[0] + sr_;
        yi[0] = ti[0] + si_;
        yr[st] = vr + wi;              /* v - i*w */
        yi[st] = vi - wr;
        yr[2 * st] = vr - wi;          /* v + i*w */
        yi[2 * st] = vi + wr;
        return;
    }
    case 4: {
        tw_v8 ar = tr[0] + tr[2], ai = ti[0] + ti[2];
        tw_v8 br = tr[0] - tr[2], bi = ti[0] - ti[2];
        tw_v8 cr = tr[1] + tr[3], ci = ti[1] + ti[3];
        tw_v8 dr = tr[1] - tr[3], di = ti[1] - ti[3];
        yr[0] = ar + cr;
        yi[0] = ai + ci;
        yr[st] = br + di;              /* b - i*d */
        yi[st] = bi - dr;
        yr[2 * st] = ar - cr;
        yi[2 * st] = ai - ci;
        yr[3 * st] = br - di;          /* b + i*d */
        yi[3 * st] = bi + dr;
        return;
    }
    case 5: {
        const double c1 = p->c15, s1 = p->s15, c2 = p->c25, s2 = p->s25;
        tw_v8 t1r = tr[1] + tr[4], t1i = ti[1] + ti[4];
        tw_v8 t2r = tr[2] + tr[3], t2i = ti[2] + ti[3];
        tw_v8 t3r = tr[1] - tr[4], t3i = ti[1] - ti[4];
        tw_v8 t4r = tr[2] - tr[3], t4i = ti[2] - ti[3];
        yr[0] = tr[0] + t1r + t2r;
        yi[0] = ti[0] + t1i + t2i;
        tw_v8 a1r = tr[0] + c1 * t1r + c2 * t2r;
        tw_v8 a1i = ti[0] + c1 * t1i + c2 * t2i;
        tw_v8 b1r = s1 * t3r + s2 * t4r, b1i = s1 * t3i + s2 * t4i;
        tw_v8 a2r = tr[0] + c2 * t1r + c1 * t2r;
        tw_v8 a2i = ti[0] + c2 * t1i + c1 * t2i;
        tw_v8 b2r = s2 * t3r - s1 * t4r, b2i = s2 * t3i - s1 * t4i;
        yr[st] = a1r + b1i;            /* a1 - i*b1 */
        yi[st] = a1i - b1r;
        yr[4 * st] = a1r - b1i;
        yi[4 * st] = a1i + b1r;
        yr[2 * st] = a2r + b2i;        /* a2 - i*b2 */
        yi[2 * st] = a2i - b2r;
        yr[3 * st] = a2r - b2i;
        yi[3 * st] = a2i + b2r;
        return;
    }
#ifdef TWD_DS
    case 8: {                  /* gen_r10 radix-8, DS race arm ONLY: default
                                * builds route r = 8 to twd_leaf8/twd_comb8
                                * before this function can see it, so the
                                * case is compiled out to keep twd_leaf_gen's
                                * inlined copy at its r9 size (the case-bloat
                                * lesson: +657 B here displaced the hot text
                                * and read +1-3% at L=12, bit-identical).
                                * Expressions = twd_leaf8's VERBATIM. */
        const double c8 = p->c8;
        tw_v8 ar = tr[0] + tr[4], ai = ti[0] + ti[4];
        tw_v8 br = tr[0] - tr[4], bi = ti[0] - ti[4];
        tw_v8 cr = tr[2] + tr[6], ci = ti[2] + ti[6];
        tw_v8 dr = tr[2] - tr[6], di = ti[2] - ti[6];
        tw_v8 e0r = ar + cr, e0i = ai + ci;
        tw_v8 e1r = br + di, e1i = bi - dr;
        tw_v8 e2r = ar - cr, e2i = ai - ci;
        tw_v8 e3r = br - di, e3i = bi + dr;
        tw_v8 gr = tr[1] + tr[5], gi = ti[1] + ti[5];
        tw_v8 hr = tr[1] - tr[5], hi = ti[1] - ti[5];
        tw_v8 kr = tr[3] + tr[7], ki = ti[3] + ti[7];
        tw_v8 lr = tr[3] - tr[7], li = ti[3] - ti[7];
        tw_v8 o0r = gr + kr, o0i = gi + ki;
        tw_v8 o1r = hr + li, o1i = hi - lr;
        tw_v8 o2r = gr - kr, o2i = gi - ki;
        tw_v8 o3r = hr - li, o3i = hi + lr;
        yr[0] = e0r + o0r;
        yi[0] = e0i + o0i;
        yr[4 * st] = e0r - o0r;
        yi[4 * st] = e0i - o0i;
        tw_v8 p1r = c8 * (o1r + o1i), p1i = c8 * (o1i - o1r);
        yr[st] = e1r + p1r;
        yi[st] = e1i + p1i;
        yr[5 * st] = e1r - p1r;
        yi[5 * st] = e1i - p1i;
        yr[2 * st] = e2r + o2i;
        yi[2 * st] = e2i - o2r;
        yr[6 * st] = e2r - o2i;
        yi[6 * st] = e2i + o2r;
        tw_v8 p3r = c8 * (o3i - o3r), p3i = c8 * (o3r + o3i);
        yr[3 * st] = e3r + p3r;
        yi[3 * st] = e3i - p3i;
        yr[7 * st] = e3r - p3r;
        yi[7 * st] = e3i + p3i;
        return;
    }
    case 10: {                 /* gen_r11 radix-10, DS race arm ONLY (default
                                * builds route r = 10 to twd_leaf10/twd_comb10
                                * in twd_rec's switches -- same reasoning as
                                * case 8 above).  Expressions = twd_leaf10's
                                * VERBATIM: PFA 2x5, zero twiddles. */
        const double cA = p->c15, sA = p->s15, cB = p->c25, sB = p->s25;
        tw_v8 s0r = tr[0] + tr[5], s0i = ti[0] + ti[5];
        tw_v8 s1r = tr[1] + tr[6], s1i = ti[1] + ti[6];
        tw_v8 s2r = tr[2] + tr[7], s2i = ti[2] + ti[7];
        tw_v8 s3r = tr[3] + tr[8], s3i = ti[3] + ti[8];
        tw_v8 s4r = tr[4] + tr[9], s4i = ti[4] + ti[9];
        tw_v8 d0r = tr[0] - tr[5], d0i = ti[0] - ti[5];
        tw_v8 d1r = tr[6] - tr[1], d1i = ti[6] - ti[1];
        tw_v8 d2r = tr[2] - tr[7], d2i = ti[2] - ti[7];
        tw_v8 d3r = tr[8] - tr[3], d3i = ti[8] - ti[3];
        tw_v8 d4r = tr[4] - tr[9], d4i = ti[4] - ti[9];
        {
            tw_v8 u1r = s1r + s4r, u1i = s1i + s4i;
            tw_v8 u2r = s2r + s3r, u2i = s2i + s3i;
            tw_v8 u3r = s1r - s4r, u3i = s1i - s4i;
            tw_v8 u4r = s2r - s3r, u4i = s2i - s3i;
            yr[0] = s0r + u1r + u2r;
            yi[0] = s0i + u1i + u2i;
            tw_v8 a1r = s0r + cA * u1r + cB * u2r;
            tw_v8 a1i = s0i + cA * u1i + cB * u2i;
            tw_v8 b1r = sA * u3r + sB * u4r, b1i = sA * u3i + sB * u4i;
            tw_v8 a2r = s0r + cB * u1r + cA * u2r;
            tw_v8 a2i = s0i + cB * u1i + cA * u2i;
            tw_v8 b2r = sB * u3r - sA * u4r, b2i = sB * u3i - sA * u4i;
            yr[2 * st] = a1r + b1i;            /* DFT5(s)[1] -> Y2 */
            yi[2 * st] = a1i - b1r;
            yr[8 * st] = a1r - b1i;            /* DFT5(s)[4] -> Y8 */
            yi[8 * st] = a1i + b1r;
            yr[4 * st] = a2r + b2i;            /* DFT5(s)[2] -> Y4 */
            yi[4 * st] = a2i - b2r;
            yr[6 * st] = a2r - b2i;            /* DFT5(s)[3] -> Y6 */
            yi[6 * st] = a2i + b2r;
        }
        {
            tw_v8 u1r = d1r + d4r, u1i = d1i + d4i;
            tw_v8 u2r = d2r + d3r, u2i = d2i + d3i;
            tw_v8 u3r = d1r - d4r, u3i = d1i - d4i;
            tw_v8 u4r = d2r - d3r, u4i = d2i - d3i;
            yr[5 * st] = d0r + u1r + u2r;      /* DFT5(d)[0] -> Y5 */
            yi[5 * st] = d0i + u1i + u2i;
            tw_v8 a1r = d0r + cA * u1r + cB * u2r;
            tw_v8 a1i = d0i + cA * u1i + cB * u2i;
            tw_v8 b1r = sA * u3r + sB * u4r, b1i = sA * u3i + sB * u4i;
            tw_v8 a2r = d0r + cB * u1r + cA * u2r;
            tw_v8 a2i = d0i + cB * u1i + cA * u2i;
            tw_v8 b2r = sB * u3r - sA * u4r, b2i = sB * u3i - sA * u4i;
            yr[7 * st] = a1r + b1i;            /* DFT5(d)[1] -> Y7 */
            yi[7 * st] = a1i - b1r;
            yr[3 * st] = a1r - b1i;            /* DFT5(d)[4] -> Y3 */
            yi[3 * st] = a1i + b1r;
            yr[9 * st] = a2r + b2i;            /* DFT5(d)[2] -> Y9 */
            yi[9 * st] = a2i - b2r;
            yr[st] = a2r - b2i;                /* DFT5(d)[3] -> Y1 */
            yi[st] = a2i + b2r;
        }
        return;
    }
#endif /* TWD_DS */
#ifndef TWD_DENSEBF
    default:
        twd_fold_bf(lv, tr, ti, yr, yi, st);
        return;
#else
    default: {                 /* dense, 4 output rows at a time for ILP */
        int k = 0;
        for (; k + 4 <= r; k += 4) {
            const double *R0 = lv->Rr + (size_t)k * r, *I0 = lv->Ri + (size_t)k * r;
            const double *R1 = R0 + r, *I1 = I0 + r;
            const double *R2 = R1 + r, *I2 = I1 + r;
            const double *R3 = R2 + r, *I3 = I2 + r;
            tw_v8 a0r = { 0 }, a0i = { 0 }, a1r = { 0 }, a1i = { 0 };
            tw_v8 a2r = { 0 }, a2i = { 0 }, a3r = { 0 }, a3i = { 0 };
            for (int s = 0; s < r; ++s) {
                const double w0 = R0[s], v0 = I0[s], w1 = R1[s], v1 = I1[s];
                const double w2 = R2[s], v2 = I2[s], w3 = R3[s], v3 = I3[s];
                const tw_v8 xr_ = tr[s], xi_ = ti[s];
                a0r += w0 * xr_ - v0 * xi_;
                a0i += w0 * xi_ + v0 * xr_;
                a1r += w1 * xr_ - v1 * xi_;
                a1i += w1 * xi_ + v1 * xr_;
                a2r += w2 * xr_ - v2 * xi_;
                a2i += w2 * xi_ + v2 * xr_;
                a3r += w3 * xr_ - v3 * xi_;
                a3i += w3 * xi_ + v3 * xr_;
            }
            yr[(k + 0) * st] = a0r;
            yi[(k + 0) * st] = a0i;
            yr[(k + 1) * st] = a1r;
            yi[(k + 1) * st] = a1i;
            yr[(k + 2) * st] = a2r;
            yi[(k + 2) * st] = a2i;
            yr[(k + 3) * st] = a3r;
            yi[(k + 3) * st] = a3i;
        }
        for (; k < r; ++k) {
            const double *Rr = lv->Rr + (size_t)k * r, *Ri = lv->Ri + (size_t)k * r;
            tw_v8 ar = { 0 }, ai = { 0 };
            for (int s = 0; s < r; ++s) {
                const double wr = Rr[s], wi = Ri[s];
                ar += wr * tr[s] - wi * ti[s];
                ai += wr * ti[s] + wi * tr[s];
            }
            yr[k * st] = ar;
            yi[k * st] = ai;
        }
        return;
    }
#endif /* TWD_DENSEBF */
    }
}

/* generic leaf, r >= 7 (odd prime -> fold butterfly).  noinline and OUTSIDE
 * twd_rec (gen_r6) so the tr/ti staging arrays' 16 KB probed stack frame is
 * paid only on this cold-ish path, not on every recursion step. */
static __attribute__((noinline, aligned(64))) void twd_leaf_gen(const struct fft3d_plan *p,
                    const struct twd_level *lv,
                    const double *xr, const double *xi, long xstr,
                    tw_v8 *Y_r, tw_v8 *Y_i)
{
    const int r = lv->r;
    tw_v8 tr[TWD_MAXR], ti[TWD_MAXR];
    for (int s = 0; s < r; ++s) {
        tr[s] = *(const tw_v8 *)(xr + 8l * s * xstr);
        ti[s] = *(const tw_v8 *)(xi + 8l * s * xstr);
    }
    twd_butterfly(p, lv, tr, ti, Y_r, Y_i, 1);
}

/* generic combine (gen_r7: only the -DTWD_DS and -DTWD_DENSEBF race arms
 * reach it -- default builds use twd_comb_fold below for radices 7/11/13):
 * X[k2*m + k1] = sum_s w_n^(s*k1) * w_r^(s*k2) * A_s[k1]; twiddles consumed
 * sequentially, k1-major (consumption order).  k1 = 0 twiddles are exactly
 * 1: copy, don't multiply.  Same noinline frame rationale as twd_leaf_gen. */
static __attribute__((noinline, aligned(64))) void twd_comb_gen(const struct fft3d_plan *p,
                    const struct twd_level *lv,
                    const double *sr, const double *si, double *yr, double *yi)
{
    const int r = lv->r, m = lv->m;
    tw_v8 *Y_r = (tw_v8 *)yr, *Y_i = (tw_v8 *)yi;
    tw_v8 tr[TWD_MAXR], ti[TWD_MAXR];
    const tw_v8 *S_r = (const tw_v8 *)sr, *S_i = (const tw_v8 *)si;
    for (int s = 0; s < r; ++s) {
        tr[s] = S_r[(long)s * m];
        ti[s] = S_i[(long)s * m];
    }
    twd_butterfly(p, lv, tr, ti, Y_r, Y_i, m);
    for (int k1 = 1; k1 < m; ++k1) {
#ifdef TWD_DS
        /* dual-select FMA twiddles (lit 11 Tier 1; tables from
         * tw_fill_ct_ds_split): same 4 FMA-class ops, every constant ratio
         * <= 1.  The select branch's pattern is fixed per plan level, so the
         * predictor sees a constant sequence per sweep. */
        const double *dm = lv->dm + (size_t)k1 * (r - 1);
        const double *dt = lv->dt + (size_t)k1 * (r - 1);
        const unsigned char *dw = lv->dsw + (size_t)k1 * (r - 1);
        tr[0] = S_r[k1];
        ti[0] = S_i[k1];
        for (int s = 1; s < r; ++s) {
            const tw_v8 ur = S_r[(long)s * m + k1], ui = S_i[(long)s * m + k1];
            const double mm = dm[s - 1], tt = dt[s - 1];
            tw_v8 pr, pi;
            if (!dw[s - 1]) { pr = ur - tt * ui; pi = ui + tt * ur; }
            else            { pr = tt * ur - ui; pi = tt * ui + ur; }
            tr[s] = mm * pr;
            ti[s] = mm * pi;
        }
#else
        const double *twr = lv->twr + (size_t)k1 * (r - 1);
        const double *twi = lv->twi + (size_t)k1 * (r - 1);
        tr[0] = S_r[k1];
        ti[0] = S_i[k1];
        for (int s = 1; s < r; ++s) {
            const tw_v8 ur = S_r[(long)s * m + k1], ui = S_i[(long)s * m + k1];
            const double wr = twr[s - 1], wi = twi[s - 1];
            tr[s] = wr * ur - wi * ui;
            ti[s] = wr * ui + wi * ur;
        }
#endif
        twd_butterfly(p, lv, tr, ti, Y_r + k1, Y_i + k1, m);
    }
}

/* fold combine for the odd-prime combine radices 7/11/13 (NEW gen_r7): the
 * WHOLE k1 = 0..m-1 combine loop in one noinline call, the twiddle multiply
 * feeding the conjugate fold directly -- no tr/ti staging round trip (2r zmm
 * stores + 2r reloads per k1 deleted) and no per-k1 butterfly dispatch.  This
 * is the gen_r6 codelet treatment applied to the one hot path that was still
 * on the generic loop; only composite sizes with a factor in {7, 11, 13} ever
 * run it (49, 77, 91, 98, 121, ... -- the surprise-draw territory), so the
 * scored acceptance cells are untouched by construction.  Twiddle-product and
 * fold expressions are twd_comb_gen's and twd_fold_bf's VERBATIM (same
 * temporaries, same order -> same FMA contraction), so outputs stay
 * bit-identical to gen_r6.  Combine radices are <= 13 by twd_factor, so
 * h <= 6 and the fold rows live in 12 zmm registers, not a frame. */
#if !defined(TWD_DS) && !defined(TWD_DENSEBF)
static __attribute__((noinline, aligned(64))) void twd_comb_fold(const struct twd_level *lv,
                    const double *sr, const double *si, double *yr, double *yi)
{
    const int r = lv->r, m = lv->m, h = (r - 1) / 2;
    const double *Cf = lv->Cf, *Sf = lv->Sf;
    tw_v8 *Y_r = (tw_v8 *)yr, *Y_i = (tw_v8 *)yi;
    const tw_v8 *S_r = (const tw_v8 *)sr, *S_i = (const tw_v8 *)si;
    for (int k1 = 0; k1 < m; ++k1) {
        tw_v8 fr[12], fi[12];              /* a rows then b rows; h <= 6 */
        const tw_v8 x0r = S_r[k1], x0i = S_i[k1];
        tw_v8 s0r = x0r, s0i = x0i;
        if (k1 == 0) {                     /* k1 = 0 twiddles exactly 1: copy */
            for (int j = 1; j <= h; ++j) {
                const tw_v8 t1r = S_r[(long)j * m], t1i = S_i[(long)j * m];
                const tw_v8 t2r = S_r[(long)(r - j) * m], t2i = S_i[(long)(r - j) * m];
                fr[j - 1] = t1r + t2r;
                fi[j - 1] = t1i + t2i;
                fr[h + j - 1] = t1r - t2r;
                fi[h + j - 1] = t1i - t2i;
                s0r += fr[j - 1];
                s0i += fi[j - 1];
            }
        } else {
            const double *twr = lv->twr + (size_t)k1 * (r - 1);
            const double *twi = lv->twi + (size_t)k1 * (r - 1);
            for (int j = 1; j <= h; ++j) {
                const tw_v8 u1r = S_r[(long)j * m + k1], u1i = S_i[(long)j * m + k1];
                const tw_v8 u2r = S_r[(long)(r - j) * m + k1];
                const tw_v8 u2i = S_i[(long)(r - j) * m + k1];
                const double w1r = twr[j - 1], w1i = twi[j - 1];
                const double w2r = twr[r - j - 1], w2i = twi[r - j - 1];
                const tw_v8 t1r = w1r * u1r - w1i * u1i;
                const tw_v8 t1i = w1r * u1i + w1i * u1r;
                const tw_v8 t2r = w2r * u2r - w2i * u2i;
                const tw_v8 t2i = w2r * u2i + w2i * u2r;
                fr[j - 1] = t1r + t2r;
                fi[j - 1] = t1i + t2i;
                fr[h + j - 1] = t1r - t2r;
                fi[h + j - 1] = t1i - t2i;
                s0r += fr[j - 1];
                s0i += fi[j - 1];
            }
        }
        Y_r[k1] = s0r;
        Y_i[k1] = s0i;
        int k = 1;
        for (; k + 1 <= h; k += 2) {
            const double *C0 = Cf + (size_t)(k - 1) * h, *S0 = Sf + (size_t)(k - 1) * h;
            const double *C1 = C0 + h, *S1 = S0 + h;
            tw_v8 e0r = x0r, e0i = x0i, e1r = x0r, e1i = x0i;
            tw_v8 o0r = { 0 }, o0i = { 0 }, o1r = { 0 }, o1i = { 0 };
            for (int j = 0; j < h; ++j) {
                const tw_v8 axr = fr[j], axi = fi[j];
                const tw_v8 bxr = fr[h + j], bxi = fi[h + j];
                const double c0 = C0[j], s0 = S0[j], c1 = C1[j], s1 = S1[j];
                e0r += c0 * axr;
                e0i += c0 * axi;
                o0r += s0 * bxr;
                o0i += s0 * bxi;
                e1r += c1 * axr;
                e1i += c1 * axi;
                o1r += s1 * bxr;
                o1i += s1 * bxi;
            }
            Y_r[(long)k * m + k1] = e0r + o0i;
            Y_i[(long)k * m + k1] = e0i - o0r;
            Y_r[(long)(r - k) * m + k1] = e0r - o0i;
            Y_i[(long)(r - k) * m + k1] = e0i + o0r;
            Y_r[(long)(k + 1) * m + k1] = e1r + o1i;
            Y_i[(long)(k + 1) * m + k1] = e1i - o1r;
            Y_r[(long)(r - k - 1) * m + k1] = e1r - o1i;
            Y_i[(long)(r - k - 1) * m + k1] = e1i + o1r;
        }
        for (; k <= h; ++k) {
            const double *C0 = Cf + (size_t)(k - 1) * h, *S0 = Sf + (size_t)(k - 1) * h;
            tw_v8 er = x0r, ei = x0i, our = { 0 }, oui = { 0 };
            for (int j = 0; j < h; ++j) {
                const double c0 = C0[j], s0 = S0[j];
                er += c0 * fr[j];
                ei += c0 * fi[j];
                our += s0 * fr[h + j];
                oui += s0 * fi[h + j];
            }
            Y_r[(long)k * m + k1] = er + oui;
            Y_i[(long)k * m + k1] = ei - our;
            Y_r[(long)(r - k) * m + k1] = er - oui;
            Y_i[(long)(r - k) * m + k1] = ei + our;
        }
    }
}
#endif /* !TWD_DS && !TWD_DENSEBF */

/* x: rows j at xr/xi + 8*j*xstr (read-only view into the gather buffer);
 * y: this call's output, n rows contiguous; s: scratch, n rows contiguous.
 * Sub-calls write into s-blocks using the matching y-blocks as THEIR scratch,
 * so buffers ping-pong down the recursion with no overlap.  gen_r6: this is
 * now a thin dispatcher -- every level body lives in a noinline function, so
 * the recursion itself carries no zmm arrays and a small frame. */
static void twd_rec(const struct fft3d_plan *p, int li,
                    const double *xr, const double *xi, long xstr,
                    double *yr, double *yi, double *sr, double *si)
{
    const struct twd_level *lv = &p->lev[li];
    const int r = lv->r, m = lv->m;
    tw_v8 *Y_r = (tw_v8 *)yr, *Y_i = (tw_v8 *)yi;

    if (m == 1) {              /* leaf: DFT_r of the strided input rows */
        switch (r) {           /* register-resident small leaves (gen_r6) */
        case 2: twd_leaf2(xr, xi, xstr, Y_r, Y_i); return;
        case 3: twd_leaf3(p, xr, xi, xstr, Y_r, Y_i); return;
        case 4: twd_leaf4(xr, xi, xstr, Y_r, Y_i); return;
        case 5: twd_leaf5(p, xr, xi, xstr, Y_r, Y_i); return;
        case 8: twd_leaf8(p, xr, xi, xstr, Y_r, Y_i); return;  /* gen_r10 */
        case 10: twd_leaf10(p, xr, xi, xstr, Y_r, Y_i); return; /* gen_r11 */
        default:               /* odd prime leaves 7..127: fold path */
            twd_leaf_gen(p, lv, xr, xi, xstr, Y_r, Y_i);
            return;
        }
    }

    for (int s = 0; s < r; ++s) /* sub-DFTs of the decimated subsequences */
        twd_rec(p, li + 1, xr + 8l * s * xstr, xi + 8l * s * xstr, xstr * r,
                sr + 8l * s * m, si + 8l * s * m,
                yr + 8l * s * m, yi + 8l * s * m);

#ifndef TWD_DS
    switch (r) {               /* whole-level register-resident combines
                                * (gen_r6); DS builds race the generic loop */
    case 2: twd_comb2(lv, sr, si, yr, yi); return;
    case 3: twd_comb3(p, lv, sr, si, yr, yi); return;
    case 4: twd_comb4(lv, sr, si, yr, yi); return;
    case 5: twd_comb5(p, lv, sr, si, yr, yi); return;
    case 8: twd_comb8(p, lv, sr, si, yr, yi); return;  /* gen_r10 */
    case 10: twd_comb10(p, lv, sr, si, yr, yi); return; /* gen_r11 */
#ifndef TWD_DENSEBF
    default:                   /* combine radices 7/11/13: fused fold combine
                                * (gen_r7); the dense race arm keeps generic */
        twd_comb_fold(lv, sr, si, yr, yi);
        return;
#else
    default: break;
#endif
    }
#endif
    twd_comb_gen(p, lv, sr, si, yr, yi);
}

/* ---- gathers/scatters: w <= 8 pencils per group, tail lanes zeroed ---------- */
/* rows contiguous in the fast index: pencil q is src + q, row j at j*rowstride.
 * Full groups deinterleave 16 consecutive doubles with two-source shuffles.
 * pf != 0 (+ -DTWD_PF): prefetch each row's next-next group lines (+256 B,
 * 2 lines) on the strided axis-0 pass (gen_layout r4's fold-load idea).  OFF
 * by default: raced on the node and it LOSES here -- +1.5-2% at L=100,
 * +0.5-1.7% at L=31, wash at L=12 (same-core interleaved pairs; this gather's
 * 2-line rows ride the OoO window fine, the extra load-port uops do not pay).
 * Kept compilable as a cross-arch race candidate (CLX's smaller L2). */
static void twd_gather_i(const double _Complex *src, long rowstride, int n, int w,
                         double *gr, double *gi, int pf)
{
    tw_v8 *Gr = (tw_v8 *)gr, *Gi = (tw_v8 *)gi;
    if (w == 8) {
#ifdef TWD_PF
        if (pf)
            for (int j = 0; j < n; ++j) {
                const double *sp = (const double *)(src + (size_t)j * rowstride);
                __builtin_prefetch(sp + 32, 0, 3);
                __builtin_prefetch(sp + 40, 0, 3);
                tw_v8 a = tw_loadu8(sp), b = tw_loadu8(sp + 8);
                Gr[j] = __builtin_shuffle(a, b, (tw_v8l){0, 2, 4, 6, 8, 10, 12, 14});
                Gi[j] = __builtin_shuffle(a, b, (tw_v8l){1, 3, 5, 7, 9, 11, 13, 15});
            }
        else
#else
        (void)pf;
#endif
        for (int j = 0; j < n; ++j) {
            const double *sp = (const double *)(src + (size_t)j * rowstride);
            tw_v8 a = tw_loadu8(sp), b = tw_loadu8(sp + 8);
            Gr[j] = __builtin_shuffle(a, b, (tw_v8l){0, 2, 4, 6, 8, 10, 12, 14});
            Gi[j] = __builtin_shuffle(a, b, (tw_v8l){1, 3, 5, 7, 9, 11, 13, 15});
        }
        return;
    }
#if defined(__AVX512F__)
    /* gen_r9: masked w < 8 tail -- two zero-masked loads feed the full-group
     * deinterleave shuffles (maskz lanes are +0.0, exactly the scalar loop's
     * zero fill, so outputs stay bit-identical).  Replaces up to 4w scalar
     * ops per row (the axis-0 LL tail and the gf-arm axis-1 z-tails at
     * 27/50/100 ran this loop on every plane). */
    {
        const int d = 2 * w;
        const __mmask8 klo = (__mmask8)((d >= 8) ? 0xFFu : ((1u << d) - 1u));
        const __mmask8 khi = (__mmask8)((d > 8) ? ((1u << (d - 8)) - 1u) : 0u);
        for (int j = 0; j < n; ++j) {
            const double *sp = (const double *)(src + (size_t)j * rowstride);
            tw_v8 a = (tw_v8)_mm512_maskz_loadu_pd(klo, sp);
            tw_v8 b = (tw_v8)_mm512_maskz_loadu_pd(khi, sp + 8);
            Gr[j] = __builtin_shuffle(a, b, (tw_v8l){0, 2, 4, 6, 8, 10, 12, 14});
            Gi[j] = __builtin_shuffle(a, b, (tw_v8l){1, 3, 5, 7, 9, 11, 13, 15});
        }
    }
#else
    for (int j = 0; j < n; ++j) {
        const double *sp = (const double *)(src + (size_t)j * rowstride);
        int q = 0;
        for (; q < w; ++q) { gr[8 * j + q] = sp[2 * q]; gi[8 * j + q] = sp[2 * q + 1]; }
        for (; q < 8; ++q) { gr[8 * j + q] = 0.0; gi[8 * j + q] = 0.0; }
    }
#endif
}

static void twd_scatter_i(double _Complex *dst, long rowstride, int n, int w,
                          const double *yr, const double *yi)
{
    const tw_v8 *Yr = (const tw_v8 *)yr, *Yi = (const tw_v8 *)yi;
    if (w == 8) {
        for (int j = 0; j < n; ++j) {
            double *dp = (double *)(dst + (size_t)j * rowstride);
            tw_v8 re = Yr[j], im = Yi[j];
            tw_storeu8(dp, __builtin_shuffle(re, im,
                                             (tw_v8l){0, 8, 1, 9, 2, 10, 3, 11}));
            tw_storeu8(dp + 8, __builtin_shuffle(re, im,
                                             (tw_v8l){4, 12, 5, 13, 6, 14, 7, 15}));
        }
        return;
    }
#if defined(__AVX512F__)
    {   /* gen_r9: masked w < 8 tail, mirror of the gather's (2w scalar
         * stores -> 2 shuffles + 2 masked stores per row) */
        const int d = 2 * w;
        const __mmask8 klo = (__mmask8)((d >= 8) ? 0xFFu : ((1u << d) - 1u));
        const __mmask8 khi = (__mmask8)((d > 8) ? ((1u << (d - 8)) - 1u) : 0u);
        for (int j = 0; j < n; ++j) {
            double *dp = (double *)(dst + (size_t)j * rowstride);
            tw_v8 re = Yr[j], im = Yi[j];
            _mm512_mask_storeu_pd(dp, klo, (__m512d)__builtin_shuffle(re, im,
                                             (tw_v8l){0, 8, 1, 9, 2, 10, 3, 11}));
            _mm512_mask_storeu_pd(dp + 8, khi, (__m512d)__builtin_shuffle(re, im,
                                             (tw_v8l){4, 12, 5, 13, 6, 14, 7, 15}));
        }
    }
#else
    for (int j = 0; j < n; ++j) {
        double *dp = (double *)(dst + (size_t)j * rowstride);
        for (int q = 0; q < w; ++q) {
            dp[2 * q] = yr[8 * j + q];
            dp[2 * q + 1] = yi[8 * j + q];
        }
    }
#endif
}

/* ---- gen_r9: two-plane split-lane forms for the PACKED axis-1 sweep --------
 * Axis 2 has always packed its rows ACROSS planes (the kblk custody exists to
 * keep that decomposition exact); axis 1 never did -- it swept per plane, so
 * every plane paid ceil(L/8) groups and a scalar w = L mod 8 tail (at L=15
 * that tail is 47% of the pencils).  These two functions let an 8-pencil
 * group span a plane seam: lanes 0..a-1 are plane A's columns c..c+a-1 (pa
 * points at plane A row 0, column c), lanes a..7 are plane B's columns
 * 0..7-a (pb = plane B row 0), a in 1..7.  Per row the 16 wanted doubles are
 * [2a from pa | 16-2a from pb]: two plain unaligned loads based at pb - 2a
 * (always mapped -- that address is the tail of the PRECEDING plane's same
 * row inside the same volume) merged with two masked loads from pa, then the
 * full-group deinterleave shuffles.  Zero scalar lane work; SIMD lanes are
 * independent, so per-pencil results are bit-identical to the per-plane
 * sweep whatever the lane composition.  The scatter mirrors it with four
 * masked stores (the two lane sets are disjoint, nothing is stored twice;
 * masked-off lanes of the pb - 2a store never touch plane A). */
static void twd_gather_i2(const double _Complex *pa, const double _Complex *pb,
                          int a, long rowstride, int n, double *gr, double *gi)
{
#if defined(__AVX512F__)
    tw_v8 *Gr = (tw_v8 *)gr, *Gi = (tw_v8 *)gi;
    const int d = 2 * a;
    const __mmask8 klo = (__mmask8)((d >= 8) ? 0xFFu : ((1u << d) - 1u));
    const __mmask8 khi = (__mmask8)((d > 8) ? ((1u << (d - 8)) - 1u) : 0u);
    for (int j = 0; j < n; ++j) {
        const double *ap = (const double *)(pa + (size_t)j * rowstride);
        const double *bp = (const double *)(pb + (size_t)j * rowstride) - d;
        tw_v8 lo = (tw_v8)_mm512_mask_loadu_pd(_mm512_loadu_pd(bp), klo, ap);
        tw_v8 hi = (tw_v8)_mm512_mask_loadu_pd(_mm512_loadu_pd(bp + 8), khi, ap + 8);
        Gr[j] = __builtin_shuffle(lo, hi, (tw_v8l){0, 2, 4, 6, 8, 10, 12, 14});
        Gi[j] = __builtin_shuffle(lo, hi, (tw_v8l){1, 3, 5, 7, 9, 11, 13, 15});
    }
#else
    for (int j = 0; j < n; ++j)
        for (int q = 0; q < 8; ++q) {
            const double *sp = (q < a)
                ? (const double *)(pa + (size_t)j * rowstride) + 2 * q
                : (const double *)(pb + (size_t)j * rowstride) + 2 * (q - a);
            gr[8 * j + q] = sp[0];
            gi[8 * j + q] = sp[1];
        }
#endif
}

static void twd_scatter_i2(double _Complex *pa, double _Complex *pb, int a,
                           long rowstride, int n,
                           const double *yr, const double *yi)
{
#if defined(__AVX512F__)
    const tw_v8 *Yr = (const tw_v8 *)yr, *Yi = (const tw_v8 *)yi;
    const int d = 2 * a;
    const __mmask8 klo = (__mmask8)((d >= 8) ? 0xFFu : ((1u << d) - 1u));
    const __mmask8 khi = (__mmask8)((d > 8) ? ((1u << (d - 8)) - 1u) : 0u);
    for (int j = 0; j < n; ++j) {
        double *ap = (double *)(pa + (size_t)j * rowstride);
        double *bp = (double *)(pb + (size_t)j * rowstride) - d;
        tw_v8 re = Yr[j], im = Yi[j];
        tw_v8 lo = __builtin_shuffle(re, im, (tw_v8l){0, 8, 1, 9, 2, 10, 3, 11});
        tw_v8 hi = __builtin_shuffle(re, im, (tw_v8l){4, 12, 5, 13, 6, 14, 7, 15});
        _mm512_mask_storeu_pd(ap, klo, (__m512d)lo);
        _mm512_mask_storeu_pd(ap + 8, khi, (__m512d)hi);
        _mm512_mask_storeu_pd(bp, (__mmask8)~klo, (__m512d)lo);
        _mm512_mask_storeu_pd(bp + 8, (__mmask8)~khi, (__m512d)hi);
    }
#else
    for (int j = 0; j < n; ++j)
        for (int q = 0; q < 8; ++q) {
            double *dp = (q < a)
                ? (double *)(pa + (size_t)j * rowstride) + 2 * q
                : (double *)(pb + (size_t)j * rowstride) + 2 * (q - a);
            dp[0] = yr[8 * j + q];
            dp[1] = yi[8 * j + q];
        }
#endif
}

/* the packed sweep's slab tail: the last w < 8 pencils, global pencil index
 * g = g0..g0+w-1, pencil g = (plane g/L, column g%L) relative to base.
 * Scalar on purpose -- at most ONE such group per slab, and only when the
 * slab's pencil count is not a multiple of 8 (never under blocked custody,
 * whose slabs hold kblk*L = 0 mod 8 pencils). */
static void twd_gather_i2t(const double _Complex *base, long L, long g0, int w,
                           double *gr, double *gi)
{
    const long LL = L * L;
    for (long j = 0; j < L; ++j)
        for (int q = 0; q < 8; ++q)
            if (q < w) {
                const long g = g0 + q;
                const double *sp =
                    (const double *)(base + (g / L) * LL + j * L + g % L);
                gr[8 * j + q] = sp[0];
                gi[8 * j + q] = sp[1];
            } else {
                gr[8 * j + q] = 0.0;
                gi[8 * j + q] = 0.0;
            }
}

static void twd_scatter_i2t(double _Complex *base, long L, long g0, int w,
                            const double *yr, const double *yi)
{
    const long LL = L * L;
    for (long j = 0; j < L; ++j)
        for (int q = 0; q < w; ++q) {
            const long g = g0 + q;
            double *dp = (double *)(base + (g / L) * LL + j * L + g % L);
            dp[0] = yr[8 * j + q];
            dp[1] = yi[8 * j + q];
        }
}

/* NEW gen_r8: axis-1 writeback straight into the axes-1->2 handoff buffer GB
 * in GROUP FORMAT -- the layout axis 2 consumes with plain aligned loads:
 * one L-row block per axis-2 8-row group, row j at [re v8][im v8] (16
 * doubles), so twd_rec direct-feeds group G with (xr, xi, xstr) =
 * (GB + G*L*16, ..+8, 2).  This is the r7 negative's benign variant (that
 * record, closing note): the SEQUENTIAL axis-1 gather stays (it is the
 * software prefetch + L1 staging -- deleting it lost 6-35% in r7), only the
 * axis1-scatter + axis2-gather SHUFFLE pair is collapsed into one
 * transpose: per 64 complex, scatter_i(16 shuf) + gather_z(48 shuf + 16
 * ld + 16 st) becomes 2x tr8x8(48 shuf) + 16 st -- minus 16 shuffles, 16
 * stores, 16 loads, and the whole gather_z sweep.  GB is slab-sized and
 * reused per slab: L2-hot custody, NO new volume stream (the r7 negative's
 * footprint mechanism avoided by construction).
 *
 * Yr/Yi hold L rows (y) of 8 z-lanes (z = c..c+w-1); output slab rows are
 * rowbase + y (rowbase = plane-in-slab * L, so groups straddle plane
 * boundaries exactly as the kblk custody requires).  Full lane-0-aligned
 * 8x8 (y x z) tiles go through two tr8x8 with aligned stores; the <= 2
 * lane-misaligned seam tiles per (plane, z-group) at odd-L plane seams and
 * every w < 8 z-tail take the SCALAR path -- raced against a masked-store
 * tr8x8 seam variant and scalar WON everywhere (25: 80.2-81.0 vs 82.8-84.3;
 * 12: 9.38-9.72 vs 9.63-9.87 us; 2 tr8x8 is 48 port-5 uops for <= 7 rows,
 * the gen_r5 map-pack lesson again).  The tile quantization is also why the
 * whole handoff is PLAN-GATED to large L (TWD_GF_MIN_BYTES below): at L=12
 * the per-tile transposes cost ~4x the old scatter_i's shuffles and the
 * deleted gather_z was smaller than that -- measured +4% at 12, +1.5% at
 * 25, vs -2.3..-2.9% wins at 27/50/100. */
static __attribute__((noinline, aligned(64))) void twd_scatter_gf(double *GB, long L,
                           long rowbase, int c, int w,
                           const double *yr, const double *yi)
{
    long y = 0;
    const long s0 = rowbase & 7;
    if (w == 8) {
        if (s0) {                       /* head seam: lanes s0..7 of group */
            long cnt = 8 - s0;
            if (cnt > L) cnt = L;
            double *g = GB + ((rowbase >> 3) * L + c) * 16;
            for (int t = 0; t < 8; ++t)
                for (long i = 0; i < cnt; ++i) {
                    g[t * 16 + s0 + i] = yr[8 * i + t];
                    g[t * 16 + 8 + s0 + i] = yi[8 * i + t];
                }
            y = cnt;
        }
#if defined(__AVX512F__)
        for (; y + 8 <= L; y += 8) {    /* full tiles: lane 0 aligned.
                                         * gen_r10: last transpose stage as
                                         * extract-to-memory stores (-16 p5
                                         * shuffles per tile, bit-identical) */
            tw_v8 Ur[8], Ui[8];
            tw_tr8x8_u(Ur, (const tw_v8 *)yr + y);
            tw_tr8x8_u(Ui, (const tw_v8 *)yi + y);
            double *g = GB + (((rowbase + y) >> 3) * L + c) * 16;
            for (int t = 0; t < 4; ++t) {
                TW_STLO(g + t * 16, Ur[t]);
                TW_STLO(g + t * 16 + 4, Ur[t + 4]);
                TW_STLO(g + t * 16 + 8, Ui[t]);
                TW_STLO(g + t * 16 + 12, Ui[t + 4]);
                TW_STHI(g + (t + 4) * 16, Ur[t]);
                TW_STHI(g + (t + 4) * 16 + 4, Ur[t + 4]);
                TW_STHI(g + (t + 4) * 16 + 8, Ui[t]);
                TW_STHI(g + (t + 4) * 16 + 12, Ui[t + 4]);
            }
        }
#else
        for (; y + 8 <= L; y += 8) {    /* full tiles: lane 0 aligned */
            tw_v8 Br[8], Bi[8];
            tw_tr8x8(Br, (const tw_v8 *)yr + y);
            tw_tr8x8(Bi, (const tw_v8 *)yi + y);
            double *g = GB + (((rowbase + y) >> 3) * L + c) * 16;
            for (int t = 0; t < 8; ++t) {
                *(tw_v8 *)(g + t * 16) = Br[t];
                *(tw_v8 *)(g + t * 16 + 8) = Bi[t];
            }
        }
#endif
    }
    for (; y < L; ++y) {                /* tail seam, and all of any w < 8 */
        const long G = (rowbase + y) >> 3, lane = (rowbase + y) & 7;
        double *g = GB + (G * L + c) * 16 + lane;
        for (int t = 0; t < w; ++t) {
            g[t * 16] = yr[8 * y + t];
            g[t * 16 + 8] = yi[8 * y + t];
        }
    }
}

/* pencils ARE rows (fastest axis): pencil q at src + q*n, element j contiguous.
 * Full groups load 4 complex per pencil and 8x8-transpose in registers. */
static void twd_gather_z(const double _Complex *src, int n, int w,
                         double *gr, double *gi)
{
    tw_v8 *Gr = (tw_v8 *)gr, *Gi = (tw_v8 *)gi;
    int j = 0;
    if (w == 8) {
#if defined(__AVX512F__) && defined(TWD_XSTG)
        /* gen_r10 extract-to-memory at the GATHER: default OFF (opt-in race
         * arm).  Unlike scatter_z/gf these stores are re-read by twd_rec
         * immediately, and a zmm load spanning two ymm stores cannot
         * store-forward -- measured on wallaby: +0.7-1% at L=12 vs the
         * one-zmm-store form (r9ctl parity), the SF-fail stall beats the
         * -8 p5.  Kept compilable for the cross-arch race. */
        for (; j + 4 <= n; j += 4) {
            tw_v8 A[8], U[8];
            for (int q = 0; q < 8; ++q)
                A[q] = tw_loadu8((const double *)(src + (size_t)q * n + j));
            tw_tr8x8_u(U, A);
            TW_STLO((double *)(Gr + j), U[0]);
            TW_STLO((double *)(Gr + j) + 4, U[4]);
            TW_STLO((double *)(Gi + j), U[1]);
            TW_STLO((double *)(Gi + j) + 4, U[5]);
            TW_STLO((double *)(Gr + j + 1), U[2]);
            TW_STLO((double *)(Gr + j + 1) + 4, U[6]);
            TW_STLO((double *)(Gi + j + 1), U[3]);
            TW_STLO((double *)(Gi + j + 1) + 4, U[7]);
            TW_STHI((double *)(Gr + j + 2), U[0]);
            TW_STHI((double *)(Gr + j + 2) + 4, U[4]);
            TW_STHI((double *)(Gi + j + 2), U[1]);
            TW_STHI((double *)(Gi + j + 2) + 4, U[5]);
            TW_STHI((double *)(Gr + j + 3), U[2]);
            TW_STHI((double *)(Gr + j + 3) + 4, U[6]);
            TW_STHI((double *)(Gi + j + 3), U[3]);
            TW_STHI((double *)(Gi + j + 3) + 4, U[7]);
        }
#else
        for (; j + 4 <= n; j += 4) {
            tw_v8 A[8], B[8];
            for (int q = 0; q < 8; ++q)
                A[q] = tw_loadu8((const double *)(src + (size_t)q * n + j));
            tw_tr8x8(B, A);
            Gr[j] = B[0];
            Gi[j] = B[1];
            Gr[j + 1] = B[2];
            Gi[j + 1] = B[3];
            Gr[j + 2] = B[4];
            Gi[j + 2] = B[5];
            Gr[j + 3] = B[6];
            Gi[j + 3] = B[7];
        }
#endif
#if defined(__AVX512F__)
        if (j < n) {            /* gen_r9: masked j-tail (n mod 4 complex) */
            const int jt = n - j;
            const __mmask8 kj = (__mmask8)((1u << (2 * jt)) - 1u);
            tw_v8 A[8], B[8];
            for (int q = 0; q < 8; ++q)
                A[q] = (tw_v8)_mm512_maskz_loadu_pd(
                    kj, (const double *)(src + (size_t)q * n + j));
            tw_tr8x8(B, A);
            for (int t = 0; t < jt; ++t) {
                Gr[j + t] = B[2 * t];
                Gi[j + t] = B[2 * t + 1];
            }
            return;
        }
#endif
    }
#if defined(__AVX512F__)
    else {                      /* gen_r9: w < 8 vectorized -- pencils q >= w
                                 * are zero rows into the transpose, exactly
                                 * the scalar zero fill (was ALL scalar) */
        for (; j + 4 <= n; j += 4) {
            tw_v8 A[8], B[8];
            int q = 0;
            for (; q < w; ++q)
                A[q] = tw_loadu8((const double *)(src + (size_t)q * n + j));
            for (; q < 8; ++q) A[q] = (tw_v8){ 0 };
            tw_tr8x8(B, A);
            Gr[j] = B[0];
            Gi[j] = B[1];
            Gr[j + 1] = B[2];
            Gi[j + 1] = B[3];
            Gr[j + 2] = B[4];
            Gi[j + 2] = B[5];
            Gr[j + 3] = B[6];
            Gi[j + 3] = B[7];
        }
        if (j < n) {
            const int jt = n - j;
            const __mmask8 kj = (__mmask8)((1u << (2 * jt)) - 1u);
            tw_v8 A[8], B[8];
            int q = 0;
            for (; q < w; ++q)
                A[q] = (tw_v8)_mm512_maskz_loadu_pd(
                    kj, (const double *)(src + (size_t)q * n + j));
            for (; q < 8; ++q) A[q] = (tw_v8){ 0 };
            tw_tr8x8(B, A);
            for (int t = 0; t < jt; ++t) {
                Gr[j + t] = B[2 * t];
                Gi[j + t] = B[2 * t + 1];
            }
        }
        return;
    }
#endif
    for (; j < n; ++j) {
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
    const tw_v8 *Yr = (const tw_v8 *)yr, *Yi = (const tw_v8 *)yi;
    if (w == 8) {
        int j = 0;
#if defined(__AVX512F__)
        /* gen_r10 extract-to-memory: -8 p5 shuffles per transpose (dst is
         * consumed a pass/step later -- no store-forward exposure) */
        for (; j + 4 <= n; j += 4) {
            tw_v8 A[8], U[8];
            A[0] = Yr[j];
            A[1] = Yi[j];
            A[2] = Yr[j + 1];
            A[3] = Yi[j + 1];
            A[4] = Yr[j + 2];
            A[5] = Yi[j + 2];
            A[6] = Yr[j + 3];
            A[7] = Yi[j + 3];
            tw_tr8x8_u(U, A);
            for (int q = 0; q < 4; ++q) {
                double *dlo = (double *)(dst + (size_t)q * n + j);
                double *dhi = (double *)(dst + (size_t)(q + 4) * n + j);
                TW_STLO(dlo, U[q]);
                TW_STLO(dlo + 4, U[q + 4]);
                TW_STHI(dhi, U[q]);
                TW_STHI(dhi + 4, U[q + 4]);
            }
        }
#else
        for (; j + 4 <= n; j += 4) {
            tw_v8 A[8], B[8];
            A[0] = Yr[j];
            A[1] = Yi[j];
            A[2] = Yr[j + 1];
            A[3] = Yi[j + 1];
            A[4] = Yr[j + 2];
            A[5] = Yi[j + 2];
            A[6] = Yr[j + 3];
            A[7] = Yi[j + 3];
            tw_tr8x8(B, A);   /* B[q] = pencil q's 4 complex, interleaved */
            for (int q = 0; q < 8; ++q)
                tw_storeu8((double *)(dst + (size_t)q * n + j), B[q]);
        }
#endif
#if defined(__AVX512F__)
        if (j < n) {            /* gen_r9: masked j-tail via one transpose */
            const int jt = n - j;
            const __mmask8 kj = (__mmask8)((1u << (2 * jt)) - 1u);
            tw_v8 A[8], B[8];
            for (int t = 0; t < jt; ++t) {
                A[2 * t] = Yr[j + t];
                A[2 * t + 1] = Yi[j + t];
            }
            for (int t = 2 * jt; t < 8; ++t) A[t] = (tw_v8){ 0 };
            tw_tr8x8(B, A);
            for (int q = 0; q < 8; ++q)
                _mm512_mask_storeu_pd((double *)(dst + (size_t)q * n + j),
                                      kj, (__m512d)B[q]);
        }
        return;
#else
        for (; j < n; ++j)
            for (int q = 0; q < 8; ++q) {
                double *dp = (double *)(dst + (size_t)q * n + j);
                dp[0] = yr[8 * j + q];
                dp[1] = yi[8 * j + q];
            }
        return;
#endif
    }
#if defined(__AVX512F__)
    {                           /* gen_r9: w < 8 vectorized (was all scalar) */
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            tw_v8 A[8], B[8];
            A[0] = Yr[j];
            A[1] = Yi[j];
            A[2] = Yr[j + 1];
            A[3] = Yi[j + 1];
            A[4] = Yr[j + 2];
            A[5] = Yi[j + 2];
            A[6] = Yr[j + 3];
            A[7] = Yi[j + 3];
            tw_tr8x8(B, A);
            for (int q = 0; q < w; ++q)
                tw_storeu8((double *)(dst + (size_t)q * n + j), B[q]);
        }
        if (j < n) {
            const int jt = n - j;
            const __mmask8 kj = (__mmask8)((1u << (2 * jt)) - 1u);
            tw_v8 A[8], B[8];
            for (int t = 0; t < jt; ++t) {
                A[2 * t] = Yr[j + t];
                A[2 * t + 1] = Yi[j + t];
            }
            for (int t = 2 * jt; t < 8; ++t) A[t] = (tw_v8){ 0 };
            tw_tr8x8(B, A);
            for (int q = 0; q < w; ++q)
                _mm512_mask_storeu_pd((double *)(dst + (size_t)q * n + j),
                                      kj, (__m512d)B[q]);
        }
    }
#else
    for (int q = 0; q < w; ++q)
        for (int j = 0; j < n; ++j) {
            double *dp = (double *)(dst + (size_t)q * n + j);
            dp[0] = yr[8 * j + q];
            dp[1] = yi[8 * j + q];
        }
#endif
}

/* ---- the graded map z/(1+|z|), fused forms (NEW gen_r3) -------------------- */
/* 4 interleaved complex per zmm.  |z| via rsqrt14 + 2 Newton (rel ~5e-17,
 * the campaign-standard ladder), then ONE well-rounded vdivpd -- the divider
 * is idle in this scatter-side op mix (gen_powp's r2 verdict on their fused
 * x-pass; raced here against the rcp14 ladder, see the strategy record). */
#if defined(__AVX512F__)
#include <immintrin.h>
#endif
static inline tw_v8 twd_map8(tw_v8 zv)
{
#if defined(__AVX512F__)
    __m512d z = (__m512d)zv;
    __m512d t = _mm512_mul_pd(z, z);
    __m512d u = _mm512_add_pd(t, _mm512_permute_pd(t, 0x55)); /* re^2+im^2, per pair */
    u = _mm512_max_pd(u, _mm512_set1_pd(1e-300));             /* rsqrt(0) guard */
    const __m512d th = _mm512_set1_pd(1.5);
    __m512d hu = _mm512_mul_pd(_mm512_set1_pd(0.5), u);
    __m512d r = _mm512_rsqrt14_pd(u);
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(hu, _mm512_mul_pd(r, r), th));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(hu, _mm512_mul_pd(r, r), th));
    __m512d den = _mm512_fmadd_pd(u, r, _mm512_set1_pd(1.0)); /* 1 + |z| */
#ifdef TWD_MAPRCP        /* A/B knob: rcp14 + 2 Newton instead of the divide */
    __m512d rc = _mm512_rcp14_pd(den);
    rc = _mm512_mul_pd(rc, _mm512_fnmadd_pd(den, rc, _mm512_set1_pd(2.0)));
    rc = _mm512_mul_pd(rc, _mm512_fnmadd_pd(den, rc, _mm512_set1_pd(2.0)));
    return (tw_v8)_mm512_mul_pd(z, rc);
#else
    return (tw_v8)_mm512_div_pd(z, den);
#endif
#else
    tw_v8 t = zv * zv;
    tw_v8 u = t + __builtin_shuffle(t, (tw_v8l){1, 0, 3, 2, 5, 4, 7, 6});
    tw_v8 r;
    for (int i = 0; i < 8; ++i) r[i] = sqrt(u[i]);
    return zv / (1.0 + r);
#endif
}

/* PAIR-PACKED map (NEW gen_r5; gen_pfa_large gen_r5's ladder, named borrow):
 * the plain twd_map8 runs the whole rsqrt ladder on lanes where every |z|^2
 * sits DUPLICATED in both halves of a complex pair -- half the ladder lanes
 * compute nothing new.  Here the 8 distinct |z|^2 of TWO vectors pack into
 * one zmm (2 permutex2var), ONE ladder serves both, and the denominators
 * unpack pair-duplicated (2 vpermpd).  Every per-element operation and
 * operand is identical to twd_map8's (re^2+im^2 add, max, rsqrt14, the two
 * Newton steps, the fmadd, the divide), so outputs are BIT-IDENTICAL; only
 * the duplicate-lane arithmetic is deleted (~11 FMA-class ops per 8 complex
 * for 2 extra port-5 shuffles). */
static inline void twd_map8p(tw_v8 zav, tw_v8 zbv, tw_v8 *oa, tw_v8 *ob)
{
#if defined(__AVX512F__)
    __m512d za = (__m512d)zav, zb = (__m512d)zbv;
    __m512d ta = _mm512_mul_pd(za, za), tb = _mm512_mul_pd(zb, zb);
    __m512d e = _mm512_permutex2var_pd(
        ta, _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), tb);
    __m512d o = _mm512_permutex2var_pd(
        ta, _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15), tb);
    __m512d u = _mm512_add_pd(e, o);          /* 8 distinct re^2+im^2 */
    u = _mm512_max_pd(u, _mm512_set1_pd(1e-300));
    const __m512d th = _mm512_set1_pd(1.5);
    __m512d hu = _mm512_mul_pd(_mm512_set1_pd(0.5), u);
    __m512d r = _mm512_rsqrt14_pd(u);
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(hu, _mm512_mul_pd(r, r), th));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(hu, _mm512_mul_pd(r, r), th));
    __m512d den = _mm512_fmadd_pd(u, r, _mm512_set1_pd(1.0));
#ifdef TWD_MAPRCP
    __m512d rc = _mm512_rcp14_pd(den);
    rc = _mm512_mul_pd(rc, _mm512_fnmadd_pd(den, rc, _mm512_set1_pd(2.0)));
    rc = _mm512_mul_pd(rc, _mm512_fnmadd_pd(den, rc, _mm512_set1_pd(2.0)));
    __m512d ra = _mm512_permutexvar_pd(_mm512_setr_epi64(0, 0, 1, 1, 2, 2, 3, 3), rc);
    __m512d rb = _mm512_permutexvar_pd(_mm512_setr_epi64(4, 4, 5, 5, 6, 6, 7, 7), rc);
    *oa = (tw_v8)_mm512_mul_pd(za, ra);
    *ob = (tw_v8)_mm512_mul_pd(zb, rb);
#else
    __m512d da = _mm512_permutexvar_pd(_mm512_setr_epi64(0, 0, 1, 1, 2, 2, 3, 3), den);
    __m512d db = _mm512_permutexvar_pd(_mm512_setr_epi64(4, 4, 5, 5, 6, 6, 7, 7), den);
    *oa = (tw_v8)_mm512_div_pd(za, da);
    *ob = (tw_v8)_mm512_div_pd(zb, db);
#endif
#else
    *oa = twd_map8(zav);
    *ob = twd_map8(zbv);
#endif
}

static inline void twd_map1(double *re, double *im)
{
    double zr = *re, zi = *im;
    double sc = 1.0 / (1.0 + sqrt(zr * zr + zi * zi));
    *re = zr * sc;
    *im = zi * sc;
}

#if defined(__AVX512F__)
/* gen_r9: EXACT-map vector replica of twd_map1 -- the same operations in the
 * same order as gcc's -O3 scalar codegen (mul zr*zr, fma zi*zi in, sqrt, add
 * 1, div for sc, two muls), with u pair-duplicated from the re lane so both
 * components use ONE sc like the scalar does.  Vectorized TAIL sites (the
 * n mod 4 j-tail of every pencil and whole w < 8 groups used to run scalar
 * twd_map1 site by site) stay bit-identical.  Tails only: the hot path keeps
 * the rsqrt ladder (vsqrt/vdiv zmm throughput is the ladder's whole reason). */
static inline tw_v8 twd_map8x(tw_v8 zv)
{
    __m512d z = (__m512d)zv;
    __m512d t = _mm512_mul_pd(z, z);
    __m512d s = _mm512_permute_pd(z, 0x55);
    __m512d u = _mm512_permute_pd(_mm512_fmadd_pd(s, s, t), 0x00);
    __m512d den = _mm512_add_pd(_mm512_set1_pd(1.0), _mm512_sqrt_pd(u));
    __m512d sc = _mm512_div_pd(_mm512_set1_pd(1.0), den);
    return (tw_v8)_mm512_mul_pd(z, sc);
}
#endif

/* axis-2 scatter with z = y + c and the map fused at the stores: the pencils
 * ARE contiguous rows, so c is read interleaved, no transpose.  Layout as
 * twd_scatter_z. */
static void twd_scatter_z_map(double _Complex *dst, const double _Complex *cf,
                              int n, int w, const double *yr, const double *yi)
{
    const tw_v8 *Yr = (const tw_v8 *)yr, *Yi = (const tw_v8 *)yi;
    if (w == 8) {
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            tw_v8 A[8], B[8];
            A[0] = Yr[j];
            A[1] = Yi[j];
            A[2] = Yr[j + 1];
            A[3] = Yi[j + 1];
            A[4] = Yr[j + 2];
            A[5] = Yi[j + 2];
            A[6] = Yr[j + 3];
            A[7] = Yi[j + 3];
            tw_tr8x8(B, A);   /* B[q] = pencil q's 4 complex, interleaved */
#ifdef TWD_MAPPAIR
            /* pair-packed ladder (gen_pfa_large gen_r5): bit-identical, but
             * it LOSES HERE -- +1-4.5% at 12/50/100 (4/4 same-core pairs vs
             * the r4 binary, outputs cmp-identical): the 4 extra port-5
             * shuffles per pair land inside the tr8x8-bound scatter, the
             * saved FMA-class ops were never the binder (gen_rader r4's
             * relocation lesson; gen_pfa_large's win was on a SEQUENTIAL map
             * pass with no port-5 competition).  Kept raceable for CLX/SPR. */
            for (int q = 0; q < 8; q += 2) {
                const double *c0 = (const double *)(cf + (size_t)q * n + j);
                const double *c1 = (const double *)(cf + (size_t)(q + 1) * n + j);
                tw_v8 za = B[q] + tw_loadu8(c0);
                tw_v8 zb = B[q + 1] + tw_loadu8(c1);
                tw_v8 ma, mb;
                twd_map8p(za, zb, &ma, &mb);
                tw_storeu8((double *)(dst + (size_t)q * n + j), ma);
                tw_storeu8((double *)(dst + (size_t)(q + 1) * n + j), mb);
            }
#else
            for (int q = 0; q < 8; ++q) {
                const double *cp = (const double *)(cf + (size_t)q * n + j);
                tw_v8 z = B[q] + tw_loadu8(cp);
                tw_storeu8((double *)(dst + (size_t)q * n + j), twd_map8(z));
            }
#endif
        }
        /* j-tail stays SCALAR on purpose (gen_r9 measured): a full-width
         * vsqrtpd/vdivpd per pencil on jt <= 3 useful complex loses to 8
         * scalar sqrt/div -- wallaby A/B: vectorizing this tail cost +8% at
         * L=25 (jt=1) and +2% at 27 (jt=3).  The divider unit prices vector
         * ops per OP, not per useful lane. */
        for (; j < n; ++j)
            for (int q = 0; q < 8; ++q) {
                double *dp = (double *)(dst + (size_t)q * n + j);
                const double *cp = (const double *)(cf + (size_t)q * n + j);
                double re = yr[8 * j + q] + cp[0], im = yi[8 * j + q] + cp[1];
                twd_map1(&re, &im);
                dp[0] = re;
                dp[1] = im;
            }
        return;
    }
#if defined(__AVX512F__)
    {                           /* gen_r9: w < 8 vectorized, exact-map vector
                                 * (was scalar map1 site by site) */
        int j = 0;
        for (; j + 4 <= n; j += 4) {
            tw_v8 A[8], B[8];
            A[0] = Yr[j];
            A[1] = Yi[j];
            A[2] = Yr[j + 1];
            A[3] = Yi[j + 1];
            A[4] = Yr[j + 2];
            A[5] = Yi[j + 2];
            A[6] = Yr[j + 3];
            A[7] = Yi[j + 3];
            tw_tr8x8(B, A);
            for (int q = 0; q < w; ++q) {
                tw_v8 z = B[q] +
                    tw_loadu8((const double *)(cf + (size_t)q * n + j));
                tw_storeu8((double *)(dst + (size_t)q * n + j), twd_map8x(z));
            }
        }
        for (; j < n; ++j)      /* j-tail scalar (same divider-pricing note) */
            for (int q = 0; q < w; ++q) {
                double *dp = (double *)(dst + (size_t)q * n + j);
                const double *cp = (const double *)(cf + (size_t)q * n + j);
                double re = yr[8 * j + q] + cp[0], im = yi[8 * j + q] + cp[1];
                twd_map1(&re, &im);
                dp[0] = re;
                dp[1] = im;
            }
    }
#else
    for (int q = 0; q < w; ++q)
        for (int j = 0; j < n; ++j) {
            double *dp = (double *)(dst + (size_t)q * n + j);
            const double *cp = (const double *)(cf + (size_t)q * n + j);
            double re = yr[8 * j + q] + cp[0], im = yi[8 * j + q] + cp[1];
            twd_map1(&re, &im);
            dp[0] = re;
            dp[1] = im;
        }
#endif
}

static void twd_exec_vol(fft3d_plan *p, const double _Complex *src,
                         double _Complex *dst);
static void twd_chain_step(fft3d_plan *p, double _Complex *st,
                           const double _Complex *cv);

/* ---- whole-pass bodies for the gf handoff (gen_r8) --------------------------
 * The gf-arm passes are their OWN noinline functions (r5/r6 case-bloat
 * lesson: with both handoff arms inlined the volume functions grew
 * 7 KB -> 17 KB and the gate-off cells 12/25 drifted +2% on bit-identical
 * code paths); the gate-OFF arm stays INLINE in twd_exec_vol /
 * twd_chain_step as the r7 loops verbatim, which is what holds the small-L
 * cells nearest r7 parity. */
static __attribute__((noinline, aligned(64))) void twd_ax1_gf(fft3d_plan *p,
                    const double _Complex *pl, long rowbase)
{
    const int L = p->L;
    for (int c = 0; c < L; c += 8) {
        int w = (L - c < 8) ? (L - c) : 8;
        twd_gather_i(pl + c, L, L, w, p->Gr, p->Gi, 0);
        twd_rec(p, 0, p->Gr, p->Gi, 1, p->Yr, p->Yi, p->Sr, p->Si);
        twd_scatter_gf(p->GB, L, rowbase, c, w, p->Yr, p->Yi);
    }
}


/* axis 2 from GB (direct feed, gather deleted); rows = slab row count */
static __attribute__((noinline, aligned(64))) void twd_ax2_gf(fft3d_plan *p,
                    double _Complex *dst, long c0, long rows)
{
    const int L = p->L;
    for (long rg = 0; rg < rows; rg += 8) {
        int w = (rows - rg < 8) ? (int)(rows - rg) : 8;
        const double *g = p->GB + (rg >> 3) * (size_t)L * 16;
        twd_rec(p, 0, g, g + 8, 2, p->Yr, p->Yi, p->Sr, p->Si);
        twd_scatter_z(dst + (c0 + rg) * L, L, w, p->Yr, p->Yi);
    }
}


/* the same two, with c added and the map fused at the scatter (chain) */
static __attribute__((noinline, aligned(64))) void twd_ax2m_gf(fft3d_plan *p,
                    double _Complex *st, const double _Complex *cv,
                    long c0, long rows)
{
    const int L = p->L;
    for (long rg = 0; rg < rows; rg += 8) {
        int w = (rows - rg < 8) ? (int)(rows - rg) : 8;
        const double *g = p->GB + (rg >> 3) * (size_t)L * 16;
        const long c = c0 + rg;
        twd_rec(p, 0, g, g + 8, 2, p->Yr, p->Yi, p->Sr, p->Si);
        twd_scatter_z_map(st + c * L, cv + c * L, L, w, p->Yr, p->Yi);
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
    /* axes-1+2 custody blocks pay only when the fused pass's two volume-sized
     * streams (state + c, or T + dst) outgrow L2; below that the volume is
     * already resident across passes and fine-grain loop alternation LOSES
     * ~1-2% (measured at L=27, 7/7 same-core pairs).  kblk = L (one block =
     * whole volume) reproduces the unblocked pass exactly.  Threshold is an
     * ICL L2 (1.25 MB); knob for the cross-arch race (CLX 1 MB, SPR 2 MB). */
#ifndef TWD_BLK_MIN_BYTES
#define TWD_BLK_MIN_BYTES (1310720u)
#endif
    p->kblk = (L % 8 == 0) ? 1 : (L % 4 == 0) ? 2 : (L % 2 == 0) ? 4 : 8;
    if (32ull * L * L * L <= TWD_BLK_MIN_BYTES) p->kblk = L;
    /* gen_r8 split-group handoff gate: the tr8x8 scatter is quantized to
     * 8x8 tiles, so at small L it costs more shuffles than the gather_z it
     * deletes (measured on the node, same-core pairs: +4% at 12, +1.5% at
     * 25, -2.7% at 27, wash at 31, -2.3% at 50, -2.9% at 100).  Threshold
     * sits between 25's 500 KB and 27's 630 KB of fused-pass streams; knob
     * for the cross-arch race like TWD_BLK_MIN_BYTES. */
#ifndef TWD_GF_MIN_BYTES
#define TWD_GF_MIN_BYTES (600000u)
#endif
    p->gf = (32ull * L * L * L >= TWD_GF_MIN_BYTES);
#ifdef TWD_NOGF /* race arm: force the r7 interleaved handoff everywhere */
    p->gf = 0;
#endif

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
        tw_cis(1, 8, &c, &s);
        p->c8 = c;                         /* sqrt(1/2), correctly rounded */
    }

    const size_t V = (size_t)L * L * L;
    size_t total = V * sizeof(double _Complex) + 6 * (8 * (size_t)L) * sizeof(double);
    /* GB: one FULL L-row group buffer per axis-2 8-row group of the largest
     * slab -- ceil(kblk*L/8) of them (a slab whose row count is not a
     * multiple of 8 ends in a partial group that still owns a full buffer;
     * sizing this kblk*L*L complex exactly overran into the twiddle tables
     * at odd unblocked L -- caught by the r8 bit-identity sweep at 25/27) */
    const size_t gbdbl = ((size_t)p->kblk * L + 7) / 8 * (16 * (size_t)L);
    if (p->gf) total += gbdbl * sizeof(double);
    for (int i = 0; i < p->nlev; ++i) {
        const struct twd_level *lv = &p->lev[i];
        const size_t ntw = (size_t)(lv->r - 1) * lv->m;
        const size_t h = (size_t)(lv->r - 1) / 2;
        total += 4 * ntw * sizeof(double) + ntw;   /* twr/twi + dm/dt + dsw */
        if (lv->r >= 7 && (lv->r & 1))             /* fold C/S + dense arm
                                                    * (odd primes; r = 8 has
                                                    * exact-constant codelets) */
            total += (2 * h * h + 2 * (size_t)lv->r * lv->r) * sizeof(double);
    }
    total += (size_t)(p->nlev * 8 + 8) * (GL_PAGE + GL_LINE) + (64u << 10);
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
    if (p->gf) {
        p->GB = gl_arena_take(&p->ar, gbdbl * sizeof(double));
        if (!p->GB) {
            fft3d_destroy(p);
            return NULL;
        }
    }

    for (int i = 0; i < p->nlev; ++i) {
        struct twd_level *lv = &p->lev[i];
        const size_t ntw = (size_t)(lv->r - 1) * lv->m;
        lv->twr = gl_arena_take(&p->ar, (ntw ? ntw : 1) * sizeof(double));
        lv->twi = gl_arena_take(&p->ar, (ntw ? ntw : 1) * sizeof(double));
        lv->dm = gl_arena_take(&p->ar, (ntw ? ntw : 1) * sizeof(double));
        lv->dt = gl_arena_take(&p->ar, (ntw ? ntw : 1) * sizeof(double));
        lv->dsw = gl_arena_take(&p->ar, ntw ? ntw : 1);
        if (!lv->twr || !lv->twi || !lv->dm || !lv->dt || !lv->dsw) {
            fft3d_destroy(p);
            return NULL;
        }
        if (lv->m > 1) {
            tw_fill_ct_split(lv->twr, lv->twi, lv->n, lv->r, lv->m);
            tw_fill_ct_ds_split(lv->dm, lv->dt, lv->dsw, lv->n, lv->r, lv->m);
            /* the audit gate: every table within 0.51 ulp or the plan refuses */
            if (tw_audit_ct_split(lv->twr, lv->twi, lv->n, lv->r, lv->m) > 0.51 ||
                tw_audit_ct_ds_split(lv->dm, lv->dt, lv->dsw,
                                     lv->n, lv->r, lv->m) > 0.51) {
                fft3d_destroy(p);
                return NULL;
            }
        }
        if (lv->r >= 7 && (lv->r & 1)) {   /* fold half-system + the dense
                                            * control arm (odd primes only;
                                            * r = 8 needs no tables) */
            const int h = (lv->r - 1) / 2;
            double *fb = gl_arena_take(&p->ar, 2 * (size_t)h * h * sizeof(double));
            lv->Rr = gl_arena_take(&p->ar, (size_t)lv->r * lv->r * sizeof(double));
            lv->Ri = gl_arena_take(&p->ar, (size_t)lv->r * lv->r * sizeof(double));
            if (!fb || !lv->Rr || !lv->Ri) {
                fft3d_destroy(p);
                return NULL;
            }
            lv->Cf = fb;
            lv->Sf = fb + (size_t)h * h;
            tw_fill_fold_half(fb, lv->r, h);           /* hp = h: no pad rows */
            tw_fill_dft_split(lv->Rr, lv->Ri, lv->r, (size_t)lv->r);
            if (tw_audit_fold_half(fb, lv->r, h) > 0.51 ||
                tw_audit_dft_split(lv->Rr, lv->Ri, lv->r, (size_t)lv->r) > 0.51) {
                fft3d_destroy(p);
                return NULL;
            }
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

    /* chain gate: two owned in-place fused-map steps vs execute + the exact
     * scalar map, one deterministic volume.  Only the fused ladder differs
     * (~1e-16/step); 1e-12 catches any structural bug and flips the chain to
     * the exact fallback path instead of shipping a fast wrong answer. */
    {
        double _Complex *a = malloc(V * sizeof *a);
        double _Complex *rf = malloc(V * sizeof *rf);
        double _Complex *cf = malloc(V * sizeof *cf);
        if (a && rf && cf) {
            unsigned u = 0x9e3779b9u;
            for (size_t i = 0; i < V; ++i) {
                u = u * 1664525u + 1013904223u;
                double re = (double)(u >> 8) / 16777216.0 - 0.5;
                u = u * 1664525u + 1013904223u;
                double im = (double)(u >> 8) / 16777216.0 - 0.5;
                a[i] = re + im * I;
                u = u * 1664525u + 1013904223u;
                cf[i] = 0.1 * ((double)(u >> 8) / 16777216.0 - 0.5);
            }
            __builtin_memcpy(rf, a, V * sizeof *rf);
            for (int s = 0; s < 2; ++s) twd_chain_step(p, a, cf);
            for (int s = 0; s < 2; ++s) {
                twd_exec_vol(p, rf, rf);
                for (size_t i = 0; i < V; ++i) {
                    double re = creal(rf[i]) + creal(cf[i]);
                    double im = cimag(rf[i]) + cimag(cf[i]);
                    twd_map1(&re, &im);
                    rf[i] = re + im * I;
                }
            }
            long double e2 = 0, n2 = 0;
            for (size_t i = 0; i < V; ++i) {
                long double dr = (long double)creal(a[i]) - creal(rf[i]);
                long double di = (long double)cimag(a[i]) - cimag(rf[i]);
                e2 += dr * dr + di * di;
                n2 += (long double)creal(rf[i]) * creal(rf[i]) +
                      (long double)cimag(rf[i]) * cimag(rf[i]);
            }
            p->chain_fb = !(e2 <= 1e-24L * n2);   /* rel L2 <= 1e-12 */
        } else {
            p->chain_fb = 1;                      /* cannot gate: exact path */
        }
        free(a);
        free(rf);
        free(cf);
    }
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    gl_arena_destroy(&p->ar);
    free(p);
}

/* one volume, out of place through p->T; src == dst is safe (src is consumed
 * into T by axis 0 before axis 2 writes dst).
 *
 * NEW gen_r4: axes 1 and 2 are both plane-local for a given x (an x-plane is
 * LL contiguous complex holding every y-pencil AND every z-pencil of that x),
 * so after the global strided axis-0 pass they run fused per block of kblk
 * planes while the block is cache-hot -- one full-volume round trip deleted
 * per volume (gen_bluestein r4's k-plane-blocked custody; kblk*L rows == 0
 * mod 8 keeps the axis-2 group decomposition and outputs bit-identical). */
static void twd_exec_vol(fft3d_plan *p, const double _Complex *src,
                         double _Complex *dst)
{
    const int L = p->L;
    const long LL = (long)L * L;

    /* axis 0 (x): row j at j*L^2, pencil = one (y,z) site, groups of 8
     * consecutive sites are contiguous within each row */
    for (long c = 0; c < LL; c += 8) {
        int w = (LL - c < 8) ? (int)(LL - c) : 8;
        twd_gather_i(src + c, LL, L, w, p->Gr, p->Gi, 1);
        twd_rec(p, 0, p->Gr, p->Gi, 1, p->Yr, p->Yi, p->Sr, p->Si);
        twd_scatter_i(p->T + c, LL, L, w, p->Yr, p->Yi);
    }
    /* axes 1 + 2, blocked: kblk planes of T get axis 1 (y: row j at j*L,
     * pencils indexed by z), then their z-rows go through axis 2 into dst
     * while still resident.  gen_r8: axis 1 writes GROUP FORMAT into GB
     * (slab-sized, L2-hot, reused per slab) instead of interleaving back
     * into the plane, and axis 2 direct-feeds from GB -- its gather is
     * deleted entirely (twd_scatter_gf comment has the op accounting). */
    for (int x0 = 0; x0 < L; x0 += p->kblk) {
        const int xe = (x0 + p->kblk < L) ? x0 + p->kblk : L;
        if (p->gf) {            /* gen_r8 split-group handoff, noinline passes */
            for (int x = x0; x < xe; ++x)
                twd_ax1_gf(p, p->T + (size_t)x * LL, (long)(x - x0) * L);
            twd_ax2_gf(p, dst, (long)x0 * L, (long)(xe - x0) * L);
            continue;
        }
        /* gate off, gen_r9: axis 1 PACKED across plane seams (still INLINE,
         * the r8 layout lesson) -- pencils numbered g = (x-x0)*L + z across
         * the slab, groups of 8 span seams via the two-pointer masked
         * gather/scatter (L >= 8 keeps any group inside two planes).  Group
         * count drops from L*ceil(L/8) to ceil(L^2/8) per volume and the
         * per-plane scalar z-tails vanish; outputs bit-identical (lane
         * composition changes, per-pencil arithmetic is elementwise). */
        if (L >= 8) {
            double _Complex *s0 = p->T + (size_t)x0 * LL;
            const long NP = (long)(xe - x0) * L;
            long g = 0;
            for (; g + 8 <= NP; g += 8) {
                const long z = g % L;
                double _Complex *pl = s0 + (g / L) * LL;
                if (z + 8 <= L) {
                    twd_gather_i(pl + z, L, L, 8, p->Gr, p->Gi, 0);
                    twd_rec(p, 0, p->Gr, p->Gi, 1, p->Yr, p->Yi, p->Sr, p->Si);
                    twd_scatter_i(pl + z, L, L, 8, p->Yr, p->Yi);
                } else {
                    const int a = (int)(L - z);
                    twd_gather_i2(pl + z, pl + LL, a, L, L, p->Gr, p->Gi);
                    twd_rec(p, 0, p->Gr, p->Gi, 1, p->Yr, p->Yi, p->Sr, p->Si);
                    twd_scatter_i2(pl + z, pl + LL, a, L, L, p->Yr, p->Yi);
                }
            }
            if (g < NP) {
                twd_gather_i2t(s0, L, g, (int)(NP - g), p->Gr, p->Gi);
                twd_rec(p, 0, p->Gr, p->Gi, 1, p->Yr, p->Yi, p->Sr, p->Si);
                twd_scatter_i2t(s0, L, g, (int)(NP - g), p->Yr, p->Yi);
            }
        } else {
            for (int x = x0; x < xe; ++x) {   /* L < 8 (unscored): per plane */
                double _Complex *pl = p->T + (size_t)x * LL;
                for (int c = 0; c < L; c += 8) {
                    int w = (L - c < 8) ? (L - c) : 8;
                    twd_gather_i(pl + c, L, L, w, p->Gr, p->Gi, 0);
                    twd_rec(p, 0, p->Gr, p->Gi, 1, p->Yr, p->Yi, p->Sr, p->Si);
                    twd_scatter_i(pl + c, L, L, w, p->Yr, p->Yi);
                }
            }
        }
        const long re = (long)xe * L;            /* block rows: contiguous z-pencils */
        for (long c = (long)x0 * L; c < re; c += 8) {
            int w = (re - c < 8) ? (int)(re - c) : 8;
            twd_gather_z(p->T + c * L, L, w, p->Gr, p->Gi);
            twd_rec(p, 0, p->Gr, p->Gi, 1, p->Yr, p->Yi, p->Sr, p->Si);
            twd_scatter_z(dst + c * L, L, w, p->Yr, p->Yi);
        }
    }
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    const size_t V = (size_t)p->L * p->L * p->L;
    for (int b = 0; b < p->batch; ++b)
        twd_exec_vol(p, in + (size_t)b * V, out + (size_t)b * V);
}

/* ---- the owned graded chain (NEW gen_r3): state <- (FFT3(state)+c)/(1+|.|) --
 * Volume-major, and every axis runs IN PLACE on the state volume (each
 * 8-pencil group is gathered to the lane buffers before anything stores), so
 * the intermediate volume T and the driver's ping-pong are gone: state and c
 * are the only volume-sized streams, and the map runs fused in the axis-2
 * scatter while the rows are in registers -- the panel-standard chain scheme
 * (gen_planner / gen_rader / gen_powp lineage). */
static void twd_chain_step(fft3d_plan *p, double _Complex *st,
                           const double _Complex *cv)
{
    const int L = p->L;
    const long LL = (long)L * L;

    for (long c = 0; c < LL; c += 8) {          /* axis 0, in place */
        int w = (LL - c < 8) ? (int)(LL - c) : 8;
        twd_gather_i(st + c, LL, L, w, p->Gr, p->Gi, 1);
        twd_rec(p, 0, p->Gr, p->Gi, 1, p->Yr, p->Yi, p->Sr, p->Si);
        twd_scatter_i(st + c, LL, L, w, p->Yr, p->Yi);
    }
    /* axes 1 + 2 + c + map, blocked per kblk planes (see twd_exec_vol): the
     * slab stays cache-hot between its axis-1 writeback and its axis-2
     * re-read.  gen_r8: the writeback goes to GB in group format (axis 1
     * reads the st plane but no longer dirties it); axis 2 direct-feeds
     * from GB and its fused-map scatter writes the final st values --
     * stream count unchanged, the axis-2 gather deleted. */
    for (int x0 = 0; x0 < L; x0 += p->kblk) {
        const int xe = (x0 + p->kblk < L) ? x0 + p->kblk : L;
        if (p->gf) {            /* gen_r8 split-group handoff, noinline passes */
            for (int x = x0; x < xe; ++x)
                twd_ax1_gf(p, st + (size_t)x * LL, (long)(x - x0) * L);
            twd_ax2m_gf(p, st, cv, (long)x0 * L, (long)(xe - x0) * L);
            continue;
        }
        /* gate off, gen_r9: axis 1 PACKED across plane seams, in place (see
         * twd_exec_vol; in-place safe -- every pencil is gathered exactly
         * once and written exactly once, split groups write only their own
         * pencils into the next plane) */
        if (L >= 8) {
            double _Complex *s0 = st + (size_t)x0 * LL;
            const long NP = (long)(xe - x0) * L;
            long g = 0;
            for (; g + 8 <= NP; g += 8) {
                const long z = g % L;
                double _Complex *pl = s0 + (g / L) * LL;
                if (z + 8 <= L) {
                    twd_gather_i(pl + z, L, L, 8, p->Gr, p->Gi, 0);
                    twd_rec(p, 0, p->Gr, p->Gi, 1, p->Yr, p->Yi, p->Sr, p->Si);
                    twd_scatter_i(pl + z, L, L, 8, p->Yr, p->Yi);
                } else {
                    const int a = (int)(L - z);
                    twd_gather_i2(pl + z, pl + LL, a, L, L, p->Gr, p->Gi);
                    twd_rec(p, 0, p->Gr, p->Gi, 1, p->Yr, p->Yi, p->Sr, p->Si);
                    twd_scatter_i2(pl + z, pl + LL, a, L, L, p->Yr, p->Yi);
                }
            }
            if (g < NP) {
                twd_gather_i2t(s0, L, g, (int)(NP - g), p->Gr, p->Gi);
                twd_rec(p, 0, p->Gr, p->Gi, 1, p->Yr, p->Yi, p->Sr, p->Si);
                twd_scatter_i2t(s0, L, g, (int)(NP - g), p->Yr, p->Yi);
            }
        } else {
            for (int x = x0; x < xe; ++x) {     /* L < 8 (unscored): per plane */
                double _Complex *pl = st + (size_t)x * LL;
                for (int c = 0; c < L; c += 8) {
                    int w = (L - c < 8) ? (L - c) : 8;
                    twd_gather_i(pl + c, L, L, w, p->Gr, p->Gi, 0);
                    twd_rec(p, 0, p->Gr, p->Gi, 1, p->Yr, p->Yi, p->Sr, p->Si);
                    twd_scatter_i(pl + c, L, L, w, p->Yr, p->Yi);
                }
            }
        }
        const long re = (long)xe * L;
        for (long c = (long)x0 * L; c < re; c += 8) {   /* axis 2, in place */
            int w = (re - c < 8) ? (int)(re - c) : 8;
            twd_gather_z(st + c * L, L, w, p->Gr, p->Gi);
            twd_rec(p, 0, p->Gr, p->Gi, 1, p->Yr, p->Yi, p->Sr, p->Si);
            twd_scatter_z_map(st + c * L, cv + c * L, L, w, p->Yr, p->Yi);
        }
    }
}


/* ---- gen_r10 radix-8 codelet DEFINITIONS (declared above; placed after the
 * hot text on purpose -- see the declaration comment) ---------------------- */
static __attribute__((noinline, aligned(64))) void twd_leaf8(const struct fft3d_plan *p,
                                                const double *xr, const double *xi,
                                                long xstr, tw_v8 *Yr, tw_v8 *Yi)
{
    const double c8 = p->c8;
    tw_v8 t0r = *(const tw_v8 *)xr, t0i = *(const tw_v8 *)xi;
    tw_v8 t1r = *(const tw_v8 *)(xr + 8l * xstr), t1i = *(const tw_v8 *)(xi + 8l * xstr);
    tw_v8 t2r = *(const tw_v8 *)(xr + 16l * xstr), t2i = *(const tw_v8 *)(xi + 16l * xstr);
    tw_v8 t3r = *(const tw_v8 *)(xr + 24l * xstr), t3i = *(const tw_v8 *)(xi + 24l * xstr);
    tw_v8 t4r = *(const tw_v8 *)(xr + 32l * xstr), t4i = *(const tw_v8 *)(xi + 32l * xstr);
    tw_v8 t5r = *(const tw_v8 *)(xr + 40l * xstr), t5i = *(const tw_v8 *)(xi + 40l * xstr);
    tw_v8 t6r = *(const tw_v8 *)(xr + 48l * xstr), t6i = *(const tw_v8 *)(xi + 48l * xstr);
    tw_v8 t7r = *(const tw_v8 *)(xr + 56l * xstr), t7i = *(const tw_v8 *)(xi + 56l * xstr);
    tw_v8 ar = t0r + t4r, ai = t0i + t4i;      /* even DFT4 on (0,2,4,6) */
    tw_v8 br = t0r - t4r, bi = t0i - t4i;
    tw_v8 cr = t2r + t6r, ci = t2i + t6i;
    tw_v8 dr = t2r - t6r, di = t2i - t6i;
    tw_v8 e0r = ar + cr, e0i = ai + ci;
    tw_v8 e1r = br + di, e1i = bi - dr;
    tw_v8 e2r = ar - cr, e2i = ai - ci;
    tw_v8 e3r = br - di, e3i = bi + dr;
    tw_v8 gr = t1r + t5r, gi = t1i + t5i;      /* odd DFT4 on (1,3,5,7) */
    tw_v8 hr = t1r - t5r, hi = t1i - t5i;
    tw_v8 kr = t3r + t7r, ki = t3i + t7i;
    tw_v8 lr = t3r - t7r, li = t3i - t7i;
    tw_v8 o0r = gr + kr, o0i = gi + ki;
    tw_v8 o1r = hr + li, o1i = hi - lr;
    tw_v8 o2r = gr - kr, o2i = gi - ki;
    tw_v8 o3r = hr - li, o3i = hi + lr;
    Yr[0] = e0r + o0r;
    Yi[0] = e0i + o0i;
    Yr[4] = e0r - o0r;
    Yi[4] = e0i - o0i;
    tw_v8 p1r = c8 * (o1r + o1i), p1i = c8 * (o1i - o1r);   /* W8^1 O1 */
    Yr[1] = e1r + p1r;
    Yi[1] = e1i + p1i;
    Yr[5] = e1r - p1r;
    Yi[5] = e1i - p1i;
    Yr[2] = e2r + o2i;                                       /* W8^2 = -i */
    Yi[2] = e2i - o2r;
    Yr[6] = e2r - o2i;
    Yi[6] = e2i + o2r;
    tw_v8 p3r = c8 * (o3i - o3r), p3i = c8 * (o3r + o3i);   /* W8^3 O3 */
    Yr[3] = e3r + p3r;
    Yi[3] = e3i - p3i;
    Yr[7] = e3r - p3r;
    Yi[7] = e3i + p3i;
}

#ifndef TWD_DS
/* whole-level register-resident radix-8 combine, like twd_comb2..5 (the
 * k1 = 0..m-1 loop in one noinline call, twiddle products feeding the DFT8
 * directly).  Butterfly expressions are twd_leaf8's VERBATIM; twiddles
 * consumed k1-major, s = 1..7 inner (tw_fill_ct_split). */
static __attribute__((noinline, aligned(64))) void twd_comb8(const struct fft3d_plan *p,
                                                const struct twd_level *lv,
                                                const double *sr, const double *si,
                                                double *yr, double *yi)
{
    const int m = lv->m;
    const double c8 = p->c8;
    const tw_v8 *S_r = (const tw_v8 *)sr, *S_i = (const tw_v8 *)si;
    tw_v8 *Y_r = (tw_v8 *)yr, *Y_i = (tw_v8 *)yi;
    for (int k1 = 0; k1 < m; ++k1) {
        tw_v8 t0r = S_r[k1], t0i = S_i[k1];
        tw_v8 t1r, t1i, t2r, t2i, t3r, t3i, t4r, t4i;
        tw_v8 t5r, t5i, t6r, t6i, t7r, t7i;
        if (k1 == 0) {
            t1r = S_r[m];
            t1i = S_i[m];
            t2r = S_r[2 * m];
            t2i = S_i[2 * m];
            t3r = S_r[3 * m];
            t3i = S_i[3 * m];
            t4r = S_r[4 * m];
            t4i = S_i[4 * m];
            t5r = S_r[5 * m];
            t5i = S_i[5 * m];
            t6r = S_r[6 * m];
            t6i = S_i[6 * m];
            t7r = S_r[7 * m];
            t7i = S_i[7 * m];
        } else {
            const double *twr = lv->twr + (size_t)k1 * 7;
            const double *twi = lv->twi + (size_t)k1 * 7;
            const tw_v8 u1r = S_r[m + k1], u1i = S_i[m + k1];
            const tw_v8 u2r = S_r[2 * m + k1], u2i = S_i[2 * m + k1];
            const tw_v8 u3r = S_r[3 * m + k1], u3i = S_i[3 * m + k1];
            const tw_v8 u4r = S_r[4 * m + k1], u4i = S_i[4 * m + k1];
            const tw_v8 u5r = S_r[5 * m + k1], u5i = S_i[5 * m + k1];
            const tw_v8 u6r = S_r[6 * m + k1], u6i = S_i[6 * m + k1];
            const tw_v8 u7r = S_r[7 * m + k1], u7i = S_i[7 * m + k1];
            t1r = twr[0] * u1r - twi[0] * u1i;
            t1i = twr[0] * u1i + twi[0] * u1r;
            t2r = twr[1] * u2r - twi[1] * u2i;
            t2i = twr[1] * u2i + twi[1] * u2r;
            t3r = twr[2] * u3r - twi[2] * u3i;
            t3i = twr[2] * u3i + twi[2] * u3r;
            t4r = twr[3] * u4r - twi[3] * u4i;
            t4i = twr[3] * u4i + twi[3] * u4r;
            t5r = twr[4] * u5r - twi[4] * u5i;
            t5i = twr[4] * u5i + twi[4] * u5r;
            t6r = twr[5] * u6r - twi[5] * u6i;
            t6i = twr[5] * u6i + twi[5] * u6r;
            t7r = twr[6] * u7r - twi[6] * u7i;
            t7i = twr[6] * u7i + twi[6] * u7r;
        }
        tw_v8 ar = t0r + t4r, ai = t0i + t4i;
        tw_v8 br = t0r - t4r, bi = t0i - t4i;
        tw_v8 cr = t2r + t6r, ci = t2i + t6i;
        tw_v8 dr = t2r - t6r, di = t2i - t6i;
        tw_v8 e0r = ar + cr, e0i = ai + ci;
        tw_v8 e1r = br + di, e1i = bi - dr;
        tw_v8 e2r = ar - cr, e2i = ai - ci;
        tw_v8 e3r = br - di, e3i = bi + dr;
        tw_v8 gr = t1r + t5r, gi = t1i + t5i;
        tw_v8 hr = t1r - t5r, hi = t1i - t5i;
        tw_v8 kr = t3r + t7r, ki = t3i + t7i;
        tw_v8 lr = t3r - t7r, li = t3i - t7i;
        tw_v8 o0r = gr + kr, o0i = gi + ki;
        tw_v8 o1r = hr + li, o1i = hi - lr;
        tw_v8 o2r = gr - kr, o2i = gi - ki;
        tw_v8 o3r = hr - li, o3i = hi + lr;
        Y_r[k1] = e0r + o0r;
        Y_i[k1] = e0i + o0i;
        Y_r[4 * m + k1] = e0r - o0r;
        Y_i[4 * m + k1] = e0i - o0i;
        tw_v8 p1r = c8 * (o1r + o1i), p1i = c8 * (o1i - o1r);
        Y_r[m + k1] = e1r + p1r;
        Y_i[m + k1] = e1i + p1i;
        Y_r[5 * m + k1] = e1r - p1r;
        Y_i[5 * m + k1] = e1i - p1i;
        Y_r[2 * m + k1] = e2r + o2i;
        Y_i[2 * m + k1] = e2i - o2r;
        Y_r[6 * m + k1] = e2r - o2i;
        Y_i[6 * m + k1] = e2i + o2r;
        tw_v8 p3r = c8 * (o3i - o3r), p3i = c8 * (o3r + o3i);
        Y_r[3 * m + k1] = e3r + p3r;
        Y_i[3 * m + k1] = e3i - p3i;
        Y_r[7 * m + k1] = e3r - p3r;
        Y_i[7 * m + k1] = e3i + p3i;
    }
}
#endif /* !TWD_DS */

/* ---- gen_r11 radix-10 codelet DEFINITIONS (declared above; appended after
 * the hot text like the r10 radix-8 pair -- the code-layout tax lesson).
 *
 * DFT10 via PFA over the coprime pair 2x5 (Good-Thomas): NO internal
 * twiddles, no negations.  Front: 5 DFT2s on input pairs {k, k+5} with the
 * CRT-even member leading the difference (d_b = (-1)^b (x_b - x_{b+5}),
 * the sign absorbed by operand order).  Back: two of twd_leaf5's DFT5
 * bodies VERBATIM (same temporaries, same order -> same FMA contraction).
 * Output map (derived and verified: X[k] = sum_b e_b W10^{kb} with
 * e_b = x_b - x_{b+5} for odd k, so DFT5(d)[j] = X[(5+2j) mod 10]):
 *   even  Y[2j]            = DFT5(s)[j]
 *   odd   Y[(5+2j) mod 10] = DFT5(d)[j]:  j = 0..4 -> Y5, Y7, Y9, Y1, Y3 */
static __attribute__((noinline, aligned(64))) void twd_leaf10(const struct fft3d_plan *p,
                                                const double *xr, const double *xi,
                                                long xstr, tw_v8 *Yr, tw_v8 *Yi)
{
    const double cA = p->c15, sA = p->s15, cB = p->c25, sB = p->s25;
    tw_v8 t0r = *(const tw_v8 *)xr, t0i = *(const tw_v8 *)xi;
    tw_v8 t1r = *(const tw_v8 *)(xr + 8l * xstr), t1i = *(const tw_v8 *)(xi + 8l * xstr);
    tw_v8 t2r = *(const tw_v8 *)(xr + 16l * xstr), t2i = *(const tw_v8 *)(xi + 16l * xstr);
    tw_v8 t3r = *(const tw_v8 *)(xr + 24l * xstr), t3i = *(const tw_v8 *)(xi + 24l * xstr);
    tw_v8 t4r = *(const tw_v8 *)(xr + 32l * xstr), t4i = *(const tw_v8 *)(xi + 32l * xstr);
    tw_v8 t5r = *(const tw_v8 *)(xr + 40l * xstr), t5i = *(const tw_v8 *)(xi + 40l * xstr);
    tw_v8 t6r = *(const tw_v8 *)(xr + 48l * xstr), t6i = *(const tw_v8 *)(xi + 48l * xstr);
    tw_v8 t7r = *(const tw_v8 *)(xr + 56l * xstr), t7i = *(const tw_v8 *)(xi + 56l * xstr);
    tw_v8 t8r = *(const tw_v8 *)(xr + 64l * xstr), t8i = *(const tw_v8 *)(xi + 64l * xstr);
    tw_v8 t9r = *(const tw_v8 *)(xr + 72l * xstr), t9i = *(const tw_v8 *)(xi + 72l * xstr);
    tw_v8 s0r = t0r + t5r, s0i = t0i + t5i;
    tw_v8 s1r = t1r + t6r, s1i = t1i + t6i;
    tw_v8 s2r = t2r + t7r, s2i = t2i + t7i;
    tw_v8 s3r = t3r + t8r, s3i = t3i + t8i;
    tw_v8 s4r = t4r + t9r, s4i = t4i + t9i;
    tw_v8 d0r = t0r - t5r, d0i = t0i - t5i;
    tw_v8 d1r = t6r - t1r, d1i = t6i - t1i;
    tw_v8 d2r = t2r - t7r, d2i = t2i - t7i;
    tw_v8 d3r = t8r - t3r, d3i = t8i - t3i;
    tw_v8 d4r = t4r - t9r, d4i = t4i - t9i;
    {   /* DFT5(s) -> Y0, Y2, Y4, Y6, Y8 (twd_leaf5's body) */
        tw_v8 u1r = s1r + s4r, u1i = s1i + s4i;
        tw_v8 u2r = s2r + s3r, u2i = s2i + s3i;
        tw_v8 u3r = s1r - s4r, u3i = s1i - s4i;
        tw_v8 u4r = s2r - s3r, u4i = s2i - s3i;
        Yr[0] = s0r + u1r + u2r;
        Yi[0] = s0i + u1i + u2i;
        tw_v8 a1r = s0r + cA * u1r + cB * u2r;
        tw_v8 a1i = s0i + cA * u1i + cB * u2i;
        tw_v8 b1r = sA * u3r + sB * u4r, b1i = sA * u3i + sB * u4i;
        tw_v8 a2r = s0r + cB * u1r + cA * u2r;
        tw_v8 a2i = s0i + cB * u1i + cA * u2i;
        tw_v8 b2r = sB * u3r - sA * u4r, b2i = sB * u3i - sA * u4i;
        Yr[2] = a1r + b1i;                     /* DFT5(s)[1] -> Y2 */
        Yi[2] = a1i - b1r;
        Yr[8] = a1r - b1i;                     /* DFT5(s)[4] -> Y8 */
        Yi[8] = a1i + b1r;
        Yr[4] = a2r + b2i;                     /* DFT5(s)[2] -> Y4 */
        Yi[4] = a2i - b2r;
        Yr[6] = a2r - b2i;                     /* DFT5(s)[3] -> Y6 */
        Yi[6] = a2i + b2r;
    }
    {   /* DFT5(d) -> Y5, Y7, Y9, Y1, Y3 */
        tw_v8 u1r = d1r + d4r, u1i = d1i + d4i;
        tw_v8 u2r = d2r + d3r, u2i = d2i + d3i;
        tw_v8 u3r = d1r - d4r, u3i = d1i - d4i;
        tw_v8 u4r = d2r - d3r, u4i = d2i - d3i;
        Yr[5] = d0r + u1r + u2r;               /* DFT5(d)[0] -> Y5 */
        Yi[5] = d0i + u1i + u2i;
        tw_v8 a1r = d0r + cA * u1r + cB * u2r;
        tw_v8 a1i = d0i + cA * u1i + cB * u2i;
        tw_v8 b1r = sA * u3r + sB * u4r, b1i = sA * u3i + sB * u4i;
        tw_v8 a2r = d0r + cB * u1r + cA * u2r;
        tw_v8 a2i = d0i + cB * u1i + cA * u2i;
        tw_v8 b2r = sB * u3r - sA * u4r, b2i = sB * u3i - sA * u4i;
        Yr[7] = a1r + b1i;                     /* DFT5(d)[1] -> Y7 */
        Yi[7] = a1i - b1r;
        Yr[3] = a1r - b1i;                     /* DFT5(d)[4] -> Y3 */
        Yi[3] = a1i + b1r;
        Yr[9] = a2r + b2i;                     /* DFT5(d)[2] -> Y9 */
        Yi[9] = a2i - b2r;
        Yr[1] = a2r - b2i;                     /* DFT5(d)[3] -> Y1 */
        Yi[1] = a2i + b2r;
    }
}

#ifndef TWD_DS
/* whole-level register-resident radix-10 combine, like twd_comb2..5/8 (the
 * k1 = 0..m-1 loop in one noinline call, twiddle products feeding the PFA
 * DFT10 directly).  Butterfly expressions are twd_leaf10's VERBATIM;
 * twiddles consumed k1-major, s = 1..9 inner (tw_fill_ct_split).  20 rows
 * live plus DFT5 temporaries -- gcc spills a few zmm here, which is still
 * far cheaper than the whole-level lane-buffer round trip this codelet
 * deletes (the r6 accounting). */
static __attribute__((noinline, aligned(64))) void twd_comb10(const struct fft3d_plan *p,
                                                const struct twd_level *lv,
                                                const double *sr, const double *si,
                                                double *yr, double *yi)
{
    const int m = lv->m;
    const double cA = p->c15, sA = p->s15, cB = p->c25, sB = p->s25;
    const tw_v8 *S_r = (const tw_v8 *)sr, *S_i = (const tw_v8 *)si;
    tw_v8 *Y_r = (tw_v8 *)yr, *Y_i = (tw_v8 *)yi;
    for (int k1 = 0; k1 < m; ++k1) {
        tw_v8 t0r = S_r[k1], t0i = S_i[k1];
        tw_v8 t1r, t1i, t2r, t2i, t3r, t3i, t4r, t4i;
        tw_v8 t5r, t5i, t6r, t6i, t7r, t7i, t8r, t8i, t9r, t9i;
        if (k1 == 0) {
            t1r = S_r[m];
            t1i = S_i[m];
            t2r = S_r[2 * m];
            t2i = S_i[2 * m];
            t3r = S_r[3 * m];
            t3i = S_i[3 * m];
            t4r = S_r[4 * m];
            t4i = S_i[4 * m];
            t5r = S_r[5 * m];
            t5i = S_i[5 * m];
            t6r = S_r[6 * m];
            t6i = S_i[6 * m];
            t7r = S_r[7 * m];
            t7i = S_i[7 * m];
            t8r = S_r[8 * m];
            t8i = S_i[8 * m];
            t9r = S_r[9 * m];
            t9i = S_i[9 * m];
        } else {
            const double *twr = lv->twr + (size_t)k1 * 9;
            const double *twi = lv->twi + (size_t)k1 * 9;
            const tw_v8 u1r = S_r[m + k1], u1i = S_i[m + k1];
            const tw_v8 u2r = S_r[2 * m + k1], u2i = S_i[2 * m + k1];
            const tw_v8 u3r = S_r[3 * m + k1], u3i = S_i[3 * m + k1];
            const tw_v8 u4r = S_r[4 * m + k1], u4i = S_i[4 * m + k1];
            const tw_v8 u5r = S_r[5 * m + k1], u5i = S_i[5 * m + k1];
            const tw_v8 u6r = S_r[6 * m + k1], u6i = S_i[6 * m + k1];
            const tw_v8 u7r = S_r[7 * m + k1], u7i = S_i[7 * m + k1];
            const tw_v8 u8r = S_r[8 * m + k1], u8i = S_i[8 * m + k1];
            const tw_v8 u9r = S_r[9 * m + k1], u9i = S_i[9 * m + k1];
            t1r = twr[0] * u1r - twi[0] * u1i;
            t1i = twr[0] * u1i + twi[0] * u1r;
            t2r = twr[1] * u2r - twi[1] * u2i;
            t2i = twr[1] * u2i + twi[1] * u2r;
            t3r = twr[2] * u3r - twi[2] * u3i;
            t3i = twr[2] * u3i + twi[2] * u3r;
            t4r = twr[3] * u4r - twi[3] * u4i;
            t4i = twr[3] * u4i + twi[3] * u4r;
            t5r = twr[4] * u5r - twi[4] * u5i;
            t5i = twr[4] * u5i + twi[4] * u5r;
            t6r = twr[5] * u6r - twi[5] * u6i;
            t6i = twr[5] * u6i + twi[5] * u6r;
            t7r = twr[6] * u7r - twi[6] * u7i;
            t7i = twr[6] * u7i + twi[6] * u7r;
            t8r = twr[7] * u8r - twi[7] * u8i;
            t8i = twr[7] * u8i + twi[7] * u8r;
            t9r = twr[8] * u9r - twi[8] * u9i;
            t9i = twr[8] * u9i + twi[8] * u9r;
        }
        tw_v8 s0r = t0r + t5r, s0i = t0i + t5i;
        tw_v8 s1r = t1r + t6r, s1i = t1i + t6i;
        tw_v8 s2r = t2r + t7r, s2i = t2i + t7i;
        tw_v8 s3r = t3r + t8r, s3i = t3i + t8i;
        tw_v8 s4r = t4r + t9r, s4i = t4i + t9i;
        tw_v8 d0r = t0r - t5r, d0i = t0i - t5i;
        tw_v8 d1r = t6r - t1r, d1i = t6i - t1i;
        tw_v8 d2r = t2r - t7r, d2i = t2i - t7i;
        tw_v8 d3r = t8r - t3r, d3i = t8i - t3i;
        tw_v8 d4r = t4r - t9r, d4i = t4i - t9i;
        {   /* DFT5(s) -> rows 0, 2, 4, 6, 8 */
            tw_v8 u1r = s1r + s4r, u1i = s1i + s4i;
            tw_v8 u2r = s2r + s3r, u2i = s2i + s3i;
            tw_v8 u3r = s1r - s4r, u3i = s1i - s4i;
            tw_v8 u4r = s2r - s3r, u4i = s2i - s3i;
            Y_r[k1] = s0r + u1r + u2r;
            Y_i[k1] = s0i + u1i + u2i;
            tw_v8 a1r = s0r + cA * u1r + cB * u2r;
            tw_v8 a1i = s0i + cA * u1i + cB * u2i;
            tw_v8 b1r = sA * u3r + sB * u4r, b1i = sA * u3i + sB * u4i;
            tw_v8 a2r = s0r + cB * u1r + cA * u2r;
            tw_v8 a2i = s0i + cB * u1i + cA * u2i;
            tw_v8 b2r = sB * u3r - sA * u4r, b2i = sB * u3i - sA * u4i;
            Y_r[2 * m + k1] = a1r + b1i;
            Y_i[2 * m + k1] = a1i - b1r;
            Y_r[8 * m + k1] = a1r - b1i;
            Y_i[8 * m + k1] = a1i + b1r;
            Y_r[4 * m + k1] = a2r + b2i;
            Y_i[4 * m + k1] = a2i - b2r;
            Y_r[6 * m + k1] = a2r - b2i;
            Y_i[6 * m + k1] = a2i + b2r;
        }
        {   /* DFT5(d) -> rows 5, 7, 9, 1, 3 */
            tw_v8 u1r = d1r + d4r, u1i = d1i + d4i;
            tw_v8 u2r = d2r + d3r, u2i = d2i + d3i;
            tw_v8 u3r = d1r - d4r, u3i = d1i - d4i;
            tw_v8 u4r = d2r - d3r, u4i = d2i - d3i;
            Y_r[5 * m + k1] = d0r + u1r + u2r;
            Y_i[5 * m + k1] = d0i + u1i + u2i;
            tw_v8 a1r = d0r + cA * u1r + cB * u2r;
            tw_v8 a1i = d0i + cA * u1i + cB * u2i;
            tw_v8 b1r = sA * u3r + sB * u4r, b1i = sA * u3i + sB * u4i;
            tw_v8 a2r = d0r + cB * u1r + cA * u2r;
            tw_v8 a2i = d0i + cB * u1i + cA * u2i;
            tw_v8 b2r = sB * u3r - sA * u4r, b2i = sB * u3i - sA * u4i;
            Y_r[7 * m + k1] = a1r + b1i;
            Y_i[7 * m + k1] = a1i - b1r;
            Y_r[3 * m + k1] = a1r - b1i;
            Y_i[3 * m + k1] = a1i + b1r;
            Y_r[9 * m + k1] = a2r + b2i;
            Y_i[9 * m + k1] = a2i - b2r;
            Y_r[m + k1] = a2r - b2i;
            Y_i[m + k1] = a2i + b2r;
        }
    }
}
#endif /* !TWD_DS */

void fft3d_chain(fft3d_plan *p, const double _Complex *x0,
                 const double _Complex *c, double _Complex *out, int m)
{
    const size_t V = (size_t)p->L * p->L * p->L;
    for (int b = 0; b < p->batch; ++b) {
        double _Complex *st = out + (size_t)b * V;
        const double _Complex *cv = c + (size_t)b * V;
        __builtin_memcpy(st, x0 + (size_t)b * V, V * sizeof *st);
        if (!p->chain_fb) {
            for (int s = 0; s < m; ++s) twd_chain_step(p, st, cv);
        } else {                    /* gate failed: exact reference path */
            for (int s = 0; s < m; ++s) {
                twd_exec_vol(p, st, st);
                for (size_t i = 0; i < V; ++i) {
                    double re = creal(st[i]) + creal(cv[i]);
                    double im = cimag(st[i]) + cimag(cv[i]);
                    twd_map1(&re, &im);
                    st[i] = re + im * I;
                }
            }
        }
    }
}

#endif /* GEN_TWIDDLE_LIB_ONLY */
