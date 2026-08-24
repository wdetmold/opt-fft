/* gen_planner -- LIBRARY LAYER: factorization search + candidate enumeration.
 *
 * WHAT THIS FILE IS
 * -----------------
 * The planner turns a length L into an ordered list of candidate per-axis
 * algorithm trees (mixed-radix Cooley-Tukey, Good-Thomas PFA, Rader, Bluestein,
 * direct dense), each with a model cost and a canonical name suitable as a
 * wisdom-cache key for the race layer (gen_race).  It also ships a generic
 * stride-capable EXECUTOR that runs any candidate tree exactly, so every class
 * owner has (a) a decomposition oracle, (b) a correctness reference for any L,
 * and (c) a guaranteed fallback for the round-6 surprise sizes.
 *
 * ADOPTION (this layer is scored by adoption -- this is the interface)
 * --------
 *     #define GEN_PLANNER_LIB 1
 *     #include "gen_planner.c"            //  from another entry in impl/
 *
 *     pln_arena *A = calloc(1, sizeof *A);
 *     pln_cand cand[12];
 *     int k = pln_enumerate(A, L, cand, 12);   // sorted by model cost
 *     // cand[i].name  : canonical key, e.g. "gt(c2(d5),gt(d4,d25))" -- use as
 *     //                 the per-(host,L) wisdom key in gen_race
 *     // cand[i].cost  : model flops/pencil (heuristic; race for the truth)
 *     // cand[i].t     : the tree, feed to the executor or to your own codelets
 *     pln_p3d *ref = pln_p3d_build(L, cand[0].t);   // full 3-D engine, any L
 *     pln_p3d_exec(ref, in, out);                    // one L^3 volume, out-of-place
 *     pln_p3d_free(ref);  free(A);
 *
 * With GEN_PLANNER_LIB defined only static pln_* symbols are emitted (no
 * fft3d_* API), so it composes with any class entry.  Compiled standalone it
 * is a complete entry that supports EVERY 2 <= L <= 128: it enumerates,
 * picks the cheapest candidate by model cost (deterministic, so independent
 * processes produce bit-identical output), and runs the generic executor.
 * Set GEN_PLANNER_RACE=1 in the environment to instead time the top
 * candidates at create() and pick the measured winner (the gen_race
 * composition, demonstrated here; off by default only for cross-process
 * bit-repeatability of the dev harness).
 *
 * EXECUTOR MODEL
 * --------------
 * Everything operates on blocks of ROWS: a row is w <= PLN_TI contiguous
 * complex doubles (interleaved), consecutive rows a caller-given stride
 * apart.  A 1-D DFT of length n transforms n rows elementwise across the
 * row direction, vectorizing over the w columns -- the vector-FFT
 * formulation, so the same code serves axis 0 (stride L^2), axis 1
 * (stride L) and, after an L x L transpose, axis 2.  All twiddles, dense
 * matrices, Rader kernels and Bluestein chirps are computed at plan time in
 * long double (argument-reduced), per the campaign's twiddle contract.
 */
#ifndef GEN_PLANNER_C_INCLUDED
#define GEN_PLANNER_C_INCLUDED

#include <complex.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifndef PLN_TI
#define PLN_TI 64               /* MAX row width in complex; the actual tile is a
                                   per-plan runtime choice (32 small L, 64 large:
                                   measured -6..-12% each vs the other's regime) */
#endif
#define PLN_NMAX 256            /* largest sub-length (Bluestein M for L=128) */
#define PLN_POOL 8192
#define PLN_PIL 3.141592653589793238462643383279502884L

enum { PLN_LEAF, PLN_CT, PLN_PFA, PLN_RADER, PLN_BLU };

typedef struct pln_node {
    int kind, n;
    int r;                       /* CT: leaf radix; PFA: n1 */
    int M;                       /* Bluestein: padded length */
    const struct pln_node *s1, *s2;   /* CT: m-side; PFA: n1,n2; Rader: n-1; Blu: M */
} pln_node;

typedef struct pln_arena {       /* calloc one of these; holds trees + memo */
    int used;
    const pln_node *best[PLN_NMAX + 1];
    double bestc[PLN_NMAX + 1];
    pln_node pool[PLN_POOL];
} pln_arena;

typedef struct pln_cand {
    const pln_node *t;
    double cost;
    char name[96];
} pln_cand;

/* ------------------------------------------------------------------ */
/* small number theory                                                  */
/* ------------------------------------------------------------------ */
static int pln_gcd(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; }

static int pln_is_prime(int n)
{
    if (n < 2) return 0;
    for (int d = 2; d * d <= n; ++d) if (n % d == 0) return 0;
    return 1;
}

static int pln_max_prime(int n)
{
    int mp = 1;
    for (int d = 2; d * d <= n; ++d) while (n % d == 0) { mp = d; n /= d; }
    return n > 1 ? n : mp;
}

static int pln_modpow(int b, int e, int m)
{
    long r = 1, x = b % m;
    while (e) { if (e & 1) r = r * x % m; x = x * x % m; e >>= 1; }
    return (int)r;
}

static int pln_modinv(int a, int m)          /* m small; brute force is fine */
{
    a %= m; if (a < 0) a += m;
    for (int x = 1; x < m; ++x) if (a * x % m == 1) return x;
    return 0;
}

static int pln_primitive_root(int p)
{
    int q[8], nq = 0, t = p - 1;
    for (int d = 2; d * d <= t; ++d) if (t % d == 0) { q[nq++] = d; while (t % d == 0) t /= d; }
    if (t > 1) q[nq++] = t;
    for (int g = 2; g < p; ++g) {
        int ok = 1;
        for (int i = 0; i < nq && ok; ++i) if (pln_modpow(g, (p - 1) / q[i], p) == 1) ok = 0;
        if (ok) return g;
    }
    return 0;                                 /* unreachable for prime p */
}

/* exact e^{-2*pi*i*num/den}: integer reduction, long-double evaluation */
static void pln_omegal(long num, long den, long double *re, long double *im)
{
    long r = num % den; if (r < 0) r += den;
    if (2 * r > den) r -= den;                /* fold to [-den/2, den/2] */
    long double a = -2.0L * PLN_PIL * (long double)r / (long double)den;
    *re = cosl(a); *im = sinl(a);
}

static void pln_omega(long num, long den, double *re, double *im)
{
    long double lr, li;
    pln_omegal(num, den, &lr, &li);
    *re = (double)lr; *im = (double)li;
}

/* forward DFT in long double, O(n^2) -- plan-time only (Rader/Bluestein tables) */
static void pln_ld_dft(int n, const long double *xr, const long double *xi,
                       long double *Xr, long double *Xi)
{
    long double wr[PLN_NMAX], wi[PLN_NMAX];
    for (int t = 0; t < n; ++t) pln_omegal(t, n, &wr[t], &wi[t]);
    for (int k = 0; k < n; ++k) {
        long double ar = 0, ai = 0;
        for (int j = 0; j < n; ++j) {
            int t = (int)((long)j * k % n);
            ar += xr[j] * wr[t] - xi[j] * wi[t];
            ai += xr[j] * wi[t] + xi[j] * wr[t];
        }
        Xr[k] = ar; Xi[k] = ai;
    }
}

/* ------------------------------------------------------------------ */
/* planner: candidate enumeration with a flop-model cost                */
/* ------------------------------------------------------------------ */
static int pln_leaf_hard(int r) { return r == 2 || r == 3 || r == 4 || r == 5 || r == 8; }

static double pln_leafF(int r)   /* model flops per pencil for one leaf DFT_r */
{
    switch (r) {
    case 2: return 6;  case 3: return 18; case 4: return 18;
    case 5: return 40; case 8: return 52;
    default: return 8.0 * r * r;
    }
}

static pln_node *pln_new(pln_arena *A, int kind, int n, int r, int M,
                         const pln_node *s1, const pln_node *s2)
{
    if (A->used >= PLN_POOL) return NULL;
    pln_node *t = &A->pool[A->used++];
    t->kind = kind; t->n = n; t->r = r; t->M = M; t->s1 = s1; t->s2 = s2;
    return t;
}

static int pln_blu_M(int n) { int M = 1; while (M < 2 * n - 1) M <<= 1; return M; }

static const pln_node *pln_best(pln_arena *A, int n);
static double pln_bestcost(pln_arena *A, int n) { pln_best(A, n); return A->bestc[n]; }

/* all root candidates for length n (children take their memoized best tree) */
static int pln_cands(pln_arena *A, int n, pln_cand *out, int cap)
{
    int cnt = 0;
#define PLN_ADD(node, c) do { const pln_node *_t = (node); \
        if (_t && cnt < cap) { out[cnt].t = _t; out[cnt].cost = (c); ++cnt; } } while (0)

    if (pln_leaf_hard(n))
        PLN_ADD(pln_new(A, PLN_LEAF, n, 0, 0, NULL, NULL), pln_leafF(n));
    else if (n <= 40)
        PLN_ADD(pln_new(A, PLN_LEAF, n, 0, 0, NULL, NULL), 8.0 * n * n);

    for (int r = 2; r <= 32 && r < n; ++r) {          /* mixed-radix CT: n = r*m */
        if (n % r) continue;
        int m = n / r;
        if (m < 2) continue;
        double c = r * pln_bestcost(A, m) + m * pln_leafF(r)
                 + 6.0 * m * (r - 1) + 4.0 * n + 30.0 * m;
        PLN_ADD(pln_new(A, PLN_CT, n, r, 0, pln_best(A, m), NULL), c);
    }
    for (int a = 2; a * a <= n; ++a) {                /* Good-Thomas: coprime split */
        if (n % a) continue;
        int b = n / a;
        if (b < 2 || pln_gcd(a, b) != 1) continue;
        double c = b * pln_bestcost(A, a) + a * pln_bestcost(A, b) + 30.0 * n;
        PLN_ADD(pln_new(A, PLN_PFA, n, a, 0, pln_best(A, a), pln_best(A, b)), c);
    }
    if (pln_is_prime(n) && n >= 5) {                  /* Rader via length n-1 conv */
        double c = 2.0 * pln_bestcost(A, n - 1) + 24.0 * (n - 1) + 16.0 * n;
        PLN_ADD(pln_new(A, PLN_RADER, n, 0, 0, pln_best(A, n - 1), NULL), c);
    }
    if (n >= 7 && pln_max_prime(n) > 13) {            /* Bluestein existence fallback */
        int M = pln_blu_M(n);
        if (M <= PLN_NMAX) {
            double c = 2.0 * pln_bestcost(A, M) + 12.0 * M + 14.0 * n;
            PLN_ADD(pln_new(A, PLN_BLU, n, 0, M, pln_best(A, M), NULL), c);
        }
    }
#undef PLN_ADD
    for (int i = 1; i < cnt; ++i) {                   /* insertion sort by cost */
        pln_cand key = out[i];
        int j = i - 1;
        while (j >= 0 && out[j].cost > key.cost) { out[j + 1] = out[j]; --j; }
        out[j + 1] = key;
    }
    return cnt;
}

static const pln_node *pln_best(pln_arena *A, int n)
{
    if (n < 2 || n > PLN_NMAX) return NULL;
    if (A->best[n]) return A->best[n];
    pln_cand tmp[64];
    int k = pln_cands(A, n, tmp, 64);
    if (!k) return NULL;
    A->best[n] = tmp[0].t;
    A->bestc[n] = tmp[0].cost;
    return A->best[n];
}

static int pln_describe_rec(const pln_node *t, char *buf, size_t cap, size_t at)
{
    int m;
    switch (t->kind) {
    case PLN_LEAF:  m = snprintf(buf + at, cap - at, "d%d", t->n); return (int)at + m;
    case PLN_CT:
        m = snprintf(buf + at, cap - at, "c%d(", t->r); at += m;
        at = pln_describe_rec(t->s1, buf, cap, at);
        m = snprintf(buf + at, cap - at, ")"); return (int)at + m;
    case PLN_PFA:
        m = snprintf(buf + at, cap - at, "gt("); at += m;
        at = pln_describe_rec(t->s1, buf, cap, at);
        m = snprintf(buf + at, cap - at, ","); at += m;
        at = pln_describe_rec(t->s2, buf, cap, at);
        m = snprintf(buf + at, cap - at, ")"); return (int)at + m;
    case PLN_RADER:
        m = snprintf(buf + at, cap - at, "rad%d(", t->n); at += m;
        at = pln_describe_rec(t->s1, buf, cap, at);
        m = snprintf(buf + at, cap - at, ")"); return (int)at + m;
    default:
        m = snprintf(buf + at, cap - at, "bs%d(", t->M); at += m;
        at = pln_describe_rec(t->s1, buf, cap, at);
        m = snprintf(buf + at, cap - at, ")"); return (int)at + m;
    }
}

static __attribute__((unused)) void pln_describe(const pln_node *t, char *buf, size_t cap)
{
    if (cap) { buf[0] = 0; pln_describe_rec(t, buf, cap, 0); }
}

/* THE planner entry point: top candidates for length n, sorted by model cost */
static __attribute__((unused)) int pln_enumerate(pln_arena *A, int n, pln_cand *out, int maxc)
{
    pln_cand tmp[64];
    int k = pln_cands(A, n, tmp, 64);
    if (k > maxc) k = maxc;
    for (int i = 0; i < k; ++i) {
        out[i] = tmp[i];
        pln_describe(out[i].t, out[i].name, sizeof out[i].name);
    }
    return k;
}

/* ------------------------------------------------------------------ */
/* leaf codelets: n rows in, n rows out; strides in DOUBLES             */
/* ------------------------------------------------------------------ */
static void pln_leaf2(const double *restrict in, ptrdiff_t is,
                      double *restrict out, ptrdiff_t os, int w)
{
    for (int c = 0; c < w; ++c) {
        double ar = in[2*c], ai = in[2*c+1], br = in[is+2*c], bi = in[is+2*c+1];
        out[2*c] = ar + br; out[2*c+1] = ai + bi;
        out[os+2*c] = ar - br; out[os+2*c+1] = ai - bi;
    }
}

static void pln_leaf3(const double *restrict in, ptrdiff_t is,
                      double *restrict out, ptrdiff_t os, int w)
{
    const double C = -0.5, S = 0.86602540378443864676;   /* sin(2*pi/3) */
    for (int c = 0; c < w; ++c) {
        double x0r = in[2*c], x0i = in[2*c+1];
        double x1r = in[is+2*c], x1i = in[is+2*c+1];
        double x2r = in[2*is+2*c], x2i = in[2*is+2*c+1];
        double tr = x1r + x2r, ti = x1i + x2i;
        double ur = x1r - x2r, ui = x1i - x2i;
        out[2*c] = x0r + tr; out[2*c+1] = x0i + ti;
        double vr = x0r + C*tr, vi = x0i + C*ti;
        out[os+2*c]     = vr + S*ui; out[os+2*c+1]     = vi - S*ur;
        out[2*os+2*c]   = vr - S*ui; out[2*os+2*c+1]   = vi + S*ur;
    }
}

static void pln_leaf4(const double *restrict in, ptrdiff_t is,
                      double *restrict out, ptrdiff_t os, int w)
{
    for (int c = 0; c < w; ++c) {
        double x0r = in[2*c],      x0i = in[2*c+1];
        double x1r = in[is+2*c],   x1i = in[is+2*c+1];
        double x2r = in[2*is+2*c], x2i = in[2*is+2*c+1];
        double x3r = in[3*is+2*c], x3i = in[3*is+2*c+1];
        double t0r = x0r + x2r, t0i = x0i + x2i;
        double t1r = x0r - x2r, t1i = x0i - x2i;
        double t2r = x1r + x3r, t2i = x1i + x3i;
        double t3r = x1r - x3r, t3i = x1i - x3i;
        out[2*c] = t0r + t2r;        out[2*c+1] = t0i + t2i;
        out[os+2*c] = t1r + t3i;     out[os+2*c+1] = t1i - t3r;      /* t1 - i t3 */
        out[2*os+2*c] = t0r - t2r;   out[2*os+2*c+1] = t0i - t2i;
        out[3*os+2*c] = t1r - t3i;   out[3*os+2*c+1] = t1i + t3r;    /* t1 + i t3 */
    }
}

static void pln_leaf5(const double *restrict in, ptrdiff_t is,
                      double *restrict out, ptrdiff_t os, int w)
{
    const double C1 = 0.30901699437494742410, C2 = -0.80901699437494742410;
    const double S1 = 0.95105651629515357212, S2 =  0.58778525229247312917;
    for (int c = 0; c < w; ++c) {
        double x0r = in[2*c],      x0i = in[2*c+1];
        double x1r = in[is+2*c],   x1i = in[is+2*c+1];
        double x2r = in[2*is+2*c], x2i = in[2*is+2*c+1];
        double x3r = in[3*is+2*c], x3i = in[3*is+2*c+1];
        double x4r = in[4*is+2*c], x4i = in[4*is+2*c+1];
        double t1r = x1r + x4r, t1i = x1i + x4i;
        double t2r = x2r + x3r, t2i = x2i + x3i;
        double t3r = x1r - x4r, t3i = x1i - x4i;
        double t4r = x2r - x3r, t4i = x2i - x3i;
        out[2*c] = x0r + t1r + t2r; out[2*c+1] = x0i + t1i + t2i;
        double ar = x0r + C1*t1r + C2*t2r, ai = x0i + C1*t1i + C2*t2i;
        double br = -S1*t3r - S2*t4r,      bi = -S1*t3i - S2*t4i;
        out[os+2*c]   = ar - bi;  out[os+2*c+1]   = ai + br;
        out[4*os+2*c] = ar + bi;  out[4*os+2*c+1] = ai - br;
        double cr = x0r + C2*t1r + C1*t2r, ci = x0i + C2*t1i + C1*t2i;
        double dr = -S2*t3r + S1*t4r,      di = -S2*t3i + S1*t4i;
        out[2*os+2*c] = cr - di;  out[2*os+2*c+1] = ci + dr;
        out[3*os+2*c] = cr + di;  out[3*os+2*c+1] = ci - dr;
    }
}

static void pln_leaf8(const double *restrict in, ptrdiff_t is,
                      double *restrict out, ptrdiff_t os, int w)
{
    const double S = 0.70710678118654752440;
    for (int c = 0; c < w; ++c) {
        double ar[4], ai[4], br[4], bi[4];
        for (int j = 0; j < 4; ++j) {
            double xr = in[j*is+2*c],       xi = in[j*is+2*c+1];
            double yr = in[(j+4)*is+2*c],   yi = in[(j+4)*is+2*c+1];
            ar[j] = xr + yr; ai[j] = xi + yi;
            br[j] = xr - yr; bi[j] = xi - yi;
        }
        /* b_j *= w8^j:  j=1: S*(1-i), j=2: -i, j=3: S*(-1-i) */
        { double r = br[1], i = bi[1]; br[1] = S*(r + i); bi[1] = S*(i - r); }
        { double r = br[2], i = bi[2]; br[2] = i;         bi[2] = -r; }
        { double r = br[3], i = bi[3]; br[3] = S*(i - r); bi[3] = S*(-i - r); }
        /* even outputs = DFT4(a), odd outputs = DFT4(b) */
        double t0r = ar[0]+ar[2], t0i = ai[0]+ai[2], t1r = ar[0]-ar[2], t1i = ai[0]-ai[2];
        double t2r = ar[1]+ar[3], t2i = ai[1]+ai[3], t3r = ar[1]-ar[3], t3i = ai[1]-ai[3];
        out[2*c] = t0r+t2r;         out[2*c+1] = t0i+t2i;
        out[2*os+2*c] = t1r+t3i;    out[2*os+2*c+1] = t1i-t3r;
        out[4*os+2*c] = t0r-t2r;    out[4*os+2*c+1] = t0i-t2i;
        out[6*os+2*c] = t1r-t3i;    out[6*os+2*c+1] = t1i+t3r;
        t0r = br[0]+br[2]; t0i = bi[0]+bi[2]; t1r = br[0]-br[2]; t1i = bi[0]-bi[2];
        t2r = br[1]+br[3]; t2i = bi[1]+bi[3]; t3r = br[1]-br[3]; t3i = bi[1]-bi[3];
        out[os+2*c] = t0r+t2r;      out[os+2*c+1] = t0i+t2i;
        out[3*os+2*c] = t1r+t3i;    out[3*os+2*c+1] = t1i-t3r;
        out[5*os+2*c] = t0r-t2r;    out[5*os+2*c+1] = t0i-t2i;
        out[7*os+2*c] = t1r-t3i;    out[7*os+2*c+1] = t1i+t3r;
    }
}

static void pln_dense_apply(const double *restrict mat, int p,
                            const double *restrict in, ptrdiff_t is,
                            double *restrict out, ptrdiff_t os, int w)
{
    double acc[2 * PLN_TI];
    for (int k = 0; k < p; ++k) {
        for (int c = 0; c < 2 * w; ++c) acc[c] = 0.0;
        const double *m = mat + 2 * (size_t)k * p;
        for (int j = 0; j < p; ++j) {
            const double wr = m[2*j], wi = m[2*j+1];
            const double *r = in + (ptrdiff_t)j * is;
            for (int c = 0; c < w; ++c) {
                acc[2*c]   += wr * r[2*c]   - wi * r[2*c+1];
                acc[2*c+1] += wr * r[2*c+1] + wi * r[2*c];
            }
        }
        double *o = out + (ptrdiff_t)k * os;
        for (int c = 0; c < 2 * w; ++c) o[c] = acc[c];
    }
}

static void pln_leaf_apply(int r, const double *mat,
                           const double *in, ptrdiff_t is,
                           double *out, ptrdiff_t os, int w)
{
    switch (r) {
    case 2: pln_leaf2(in, is, out, os, w); break;
    case 3: pln_leaf3(in, is, out, os, w); break;
    case 4: pln_leaf4(in, is, out, os, w); break;
    case 5: pln_leaf5(in, is, out, os, w); break;
    case 8: pln_leaf8(in, is, out, os, w); break;
    default: pln_dense_apply(mat, r, in, is, out, os, w); break;
    }
}

/* ------------------------------------------------------------------ */
/* executable plan: tree + plan-time tables + shared scratch arena      */
/* ------------------------------------------------------------------ */
typedef struct pln_x {
    int kind, n, r, M;
    struct pln_x *s1, *s2;
    double *tw;                  /* CT: (r-1) x m twiddles, consumption order */
    double *mat;                 /* dense matrix for LEAF / CT leaf radix */
    int *pin, *pout;             /* PFA gather/scatter; Rader gather/scatter */
    double *vt;                  /* Rader FFT(v)/(p-1)  or  Bluestein FFT(h)/M */
    double *cj;                  /* Bluestein chirp c_j, j < n */
    size_t off;                  /* this node's scratch offset (doubles) */
} pln_x;

typedef struct pln_xplan {
    pln_x *root;
    double *arena;
    size_t need;                 /* doubles */
    int tile;                    /* row width in complex, <= PLN_TI */
} pln_xplan;

static double *pln_dense_matrix(int n)
{
    double *m = malloc(2 * (size_t)n * n * sizeof *m);
    if (!m) return NULL;
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            pln_omega((long)k * j, n, &m[2*((size_t)k*n + j)], &m[2*((size_t)k*n + j) + 1]);
    return m;
}

static pln_x *pln_xbuild_rec(const pln_node *t, size_t base, size_t *hi, int tile)
{
    pln_x *x = calloc(1, sizeof *x);
    if (!x) return NULL;
    x->kind = t->kind; x->n = t->n; x->r = t->r; x->M = t->M; x->off = base;
    size_t own = 0;
    switch (t->kind) {
    case PLN_LEAF:
        if (!pln_leaf_hard(t->n)) x->mat = pln_dense_matrix(t->n);
        break;
    case PLN_CT: {
        int r = t->r, m = t->n / t->r;
        own = 2 * (size_t)tile * t->n;
        x->tw = malloc(2 * (size_t)(r - 1) * m * sizeof *x->tw);
        if (x->tw)
            for (int j1 = 1; j1 < r; ++j1)
                for (int k2 = 0; k2 < m; ++k2)
                    pln_omega((long)j1 * k2, t->n,
                              &x->tw[2*((size_t)(j1-1)*m + k2)],
                              &x->tw[2*((size_t)(j1-1)*m + k2) + 1]);
        if (!pln_leaf_hard(r)) x->mat = pln_dense_matrix(r);
        x->s1 = pln_xbuild_rec(t->s1, base + own, hi, tile);
        break;
    }
    case PLN_PFA: {
        int n1 = t->r, n2 = t->n / t->r, n = t->n;
        own = 4 * (size_t)tile * n;
        x->pin = malloc((size_t)n * sizeof *x->pin);
        x->pout = malloc((size_t)n * sizeof *x->pout);
        int e1 = n2 % n1 ? n2 * pln_modinv(n2, n1) : 0;
        int e2 = n1 % n2 ? n1 * pln_modinv(n1, n2) : 0;
        for (int a = 0; a < n1; ++a)
            for (int b = 0; b < n2; ++b) {
                x->pin[a * n2 + b] = (int)(((long)n2 * a + (long)n1 * b) % n);
                x->pout[a * n2 + b] = (int)(((long)e1 * a + (long)e2 * b) % n);
            }
        x->s1 = pln_xbuild_rec(t->s1, base + own, hi, tile);
        x->s2 = pln_xbuild_rec(t->s2, base + own, hi, tile);
        break;
    }
    case PLN_RADER: {
        int p = t->n, P1 = p - 1;
        own = 4 * (size_t)tile * P1;
        int g = pln_primitive_root(p);
        x->pin = malloc((size_t)P1 * sizeof *x->pin);    /* jg[q] = g^q mod p */
        x->pout = malloc((size_t)P1 * sizeof *x->pout);  /* g^{-m} mod p */
        long pw = 1;
        for (int q = 0; q < P1; ++q) { x->pin[q] = (int)pw; pw = pw * g % p; }
        for (int m = 0; m < P1; ++m) x->pout[m] = x->pin[(P1 - m) % P1];
        /* v_t = w_p^{g^{-t}}; V = DFT(v)/P1 in long double */
        long double vr[PLN_NMAX] = {0}, vi[PLN_NMAX] = {0}, Vr[PLN_NMAX], Vi[PLN_NMAX];
        for (int q = 0; q < P1; ++q)
            pln_omegal(x->pin[(P1 - q) % P1], p, &vr[q], &vi[q]);
        pln_ld_dft(P1, vr, vi, Vr, Vi);
        x->vt = malloc(2 * (size_t)P1 * sizeof *x->vt);
        for (int q = 0; q < P1; ++q) {
            x->vt[2*q]   = (double)(Vr[q] / P1);
            x->vt[2*q+1] = (double)(Vi[q] / P1);
        }
        x->s1 = pln_xbuild_rec(t->s1, base + own, hi, tile);
        break;
    }
    case PLN_BLU: {
        int n = t->n, M = t->M;
        own = 4 * (size_t)tile * M;
        x->cj = malloc(2 * (size_t)n * sizeof *x->cj);
        long double hr[PLN_NMAX], hi_[PLN_NMAX], Hr[PLN_NMAX], Hi[PLN_NMAX];
        for (int j = 0; j < M; ++j) { hr[j] = 0; hi_[j] = 0; }
        for (int j = 0; j < n; ++j) {
            long q = ((long)j * j) % (2L * n);
            long double cr, ci;
            pln_omegal(q, 2L * n, &cr, &ci);             /* c_j = e^{-i pi j^2/n} */
            x->cj[2*j] = (double)cr; x->cj[2*j+1] = (double)ci;
            hr[j] = cr; hi_[j] = -ci;                    /* h_j = conj(c_j) */
            if (j > 0) { hr[M - j] = cr; hi_[M - j] = -ci; }
        }
        pln_ld_dft(M, hr, hi_, Hr, Hi);
        x->vt = malloc(2 * (size_t)M * sizeof *x->vt);
        for (int k = 0; k < M; ++k) {
            x->vt[2*k]   = (double)(Hr[k] / M);
            x->vt[2*k+1] = (double)(Hi[k] / M);
        }
        x->s1 = pln_xbuild_rec(t->s1, base + own, hi, tile);
        break;
    }
    }
    if (base + own > *hi) *hi = base + own;
    return x;
}

static void pln_xfree_rec(pln_x *x)
{
    if (!x) return;
    pln_xfree_rec(x->s1); pln_xfree_rec(x->s2);
    free(x->tw); free(x->mat); free(x->pin); free(x->pout); free(x->vt); free(x->cj);
    free(x);
}

/* tile <= PLN_TI is the row width in complex; tile <= 0 picks the default */
static __attribute__((unused)) pln_xplan *pln_xplan_build(const pln_node *t, int tile)
{
    pln_xplan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    if (tile <= 0) tile = 32;
    if (tile > PLN_TI) tile = PLN_TI;
    p->tile = tile;
    size_t hi = 0;
    p->root = pln_xbuild_rec(t, 0, &hi, tile);
    p->need = hi ? hi : 1;
    if (posix_memalign((void **)&p->arena, 64, p->need * sizeof(double)) != 0)
        p->arena = NULL;
    if (!p->root || !p->arena) { pln_xfree_rec(p->root); free(p->arena); free(p); return NULL; }
    memset(p->arena, 0, p->need * sizeof(double));
    return p;
}

static __attribute__((unused)) void pln_xplan_free(pln_xplan *p)
{
    if (!p) return;
    pln_xfree_rec(p->root); free(p->arena); free(p);
}

/* row helpers; strides in doubles, rows are 2w doubles */
static void pln_rowscale(double *row, double wr, double wi, int w)
{
    for (int c = 0; c < w; ++c) {
        double re = row[2*c], im = row[2*c+1];
        row[2*c] = re * wr - im * wi;
        row[2*c+1] = re * wi + im * wr;
    }
}

static void pln_rowscale_conj(double *row, double wr, double wi, int w)
{   /* row <- conj(row * (wr + i wi)) */
    for (int c = 0; c < w; ++c) {
        double re = row[2*c], im = row[2*c+1];
        row[2*c] = re * wr - im * wi;
        row[2*c+1] = -(re * wi + im * wr);
    }
}

/* the generic executor: n rows of w complex, src/dst strides in DOUBLES;
 * bs = scratch row stride in doubles (2 * plan tile) */
static void pln_xexec(const pln_x *x, const double *restrict src, ptrdiff_t ss,
                      double *restrict dst, ptrdiff_t ds, int w,
                      double *restrict arena, const ptrdiff_t bs)
{
    switch (x->kind) {
    case PLN_LEAF:
        pln_leaf_apply(x->n, x->mat, src, ss, dst, ds, w);
        break;
    case PLN_CT: {
        const int r = x->r, m = x->n / x->r;
        double *buf = arena + x->off;
        for (int j1 = 0; j1 < r; ++j1)
            pln_xexec(x->s1, src + j1 * ss, ss * r, buf + (size_t)j1 * m * bs, bs, w, arena, bs);
        for (int j1 = 1; j1 < r; ++j1) {
            const double *t = x->tw + 2 * (size_t)(j1 - 1) * m;
            double *row = buf + (size_t)j1 * m * bs;
            for (int k2 = 1; k2 < m; ++k2)               /* k2 = 0: twiddle = 1 */
                pln_rowscale(row + (size_t)k2 * bs, t[2*k2], t[2*k2+1], w);
        }
        for (int k2 = 0; k2 < m; ++k2)
            pln_leaf_apply(r, x->mat, buf + (size_t)k2 * bs, m * bs,
                           dst + (size_t)k2 * ds, m * ds, w);
        break;
    }
    case PLN_PFA: {
        const int n1 = x->r, n2 = x->n / x->r, n = x->n;
        double *A = arena + x->off, *B = A + (size_t)n * bs;
        for (int j = 0; j < n; ++j)
            memcpy(A + (size_t)j * bs, src + (ptrdiff_t)x->pin[j] * ss, 2 * w * sizeof(double));
        for (int a = 0; a < n1; ++a)
            pln_xexec(x->s2, A + (size_t)a * n2 * bs, bs, B + (size_t)a * n2 * bs, bs, w, arena, bs);
        for (int b = 0; b < n2; ++b)
            pln_xexec(x->s1, B + (size_t)b * bs, n2 * bs, A + (size_t)b * bs, n2 * bs, w, arena, bs);
        for (int k = 0; k < n; ++k)
            memcpy(dst + (ptrdiff_t)x->pout[k] * ds, A + (size_t)k * bs, 2 * w * sizeof(double));
        break;
    }
    case PLN_RADER: {
        const int p = x->n, P1 = p - 1;
        double *A = arena + x->off, *B = A + (size_t)P1 * bs;
        double sum[2 * PLN_TI];
        memcpy(sum, src, 2 * w * sizeof(double));        /* x[0] */
        for (int q = 0; q < P1; ++q) {
            const double *r = src + (ptrdiff_t)x->pin[q] * ss;
            double *a = A + (size_t)q * bs;
            for (int c = 0; c < 2 * w; ++c) { a[c] = r[c]; sum[c] += r[c]; }
        }
        for (int c = 0; c < 2 * w; ++c) dst[c] = sum[c]; /* X[0] = sum of all */
        pln_xexec(x->s1, A, bs, B, bs, w, arena, bs);    /* U = FFT(u) */
        for (int q = 0; q < P1; ++q)                     /* B <- conj(U * V) */
            pln_rowscale_conj(B + (size_t)q * bs, x->vt[2*q], x->vt[2*q+1], w);
        pln_xexec(x->s1, B, bs, A, bs, w, arena, bs);    /* conv = conj(FFT(B)) */
        for (int m = 0; m < P1; ++m) {
            const double *a = A + (size_t)m * bs;
            double *o = dst + (ptrdiff_t)x->pout[m] * ds;
            for (int c = 0; c < w; ++c) {                /* X[g^-m] = x0 + conj(a) */
                o[2*c]   = src[2*c]   + a[2*c];
                o[2*c+1] = src[2*c+1] - a[2*c+1];
            }
        }
        break;
    }
    case PLN_BLU: {
        const int n = x->n, M = x->M;
        double *A = arena + x->off, *B = A + (size_t)M * bs;
        for (int j = 0; j < n; ++j) {                    /* a_j = x_j * c_j */
            const double *r = src + (ptrdiff_t)j * ss;
            double *a = A + (size_t)j * bs;
            const double cr = x->cj[2*j], ci = x->cj[2*j+1];
            for (int c = 0; c < w; ++c) {
                a[2*c]   = r[2*c] * cr - r[2*c+1] * ci;
                a[2*c+1] = r[2*c] * ci + r[2*c+1] * cr;
            }
        }
        for (int j = n; j < M; ++j)
            memset(A + (size_t)j * bs, 0, 2 * w * sizeof(double));
        pln_xexec(x->s1, A, bs, B, bs, w, arena, bs);
        for (int k = 0; k < M; ++k)
            pln_rowscale_conj(B + (size_t)k * bs, x->vt[2*k], x->vt[2*k+1], w);
        pln_xexec(x->s1, B, bs, A, bs, w, arena, bs);
        for (int k = 0; k < n; ++k) {                    /* X_k = c_k * conj(a_k) */
            const double *a = A + (size_t)k * bs;
            double *o = dst + (ptrdiff_t)k * ds;
            const double cr = x->cj[2*k], ci = x->cj[2*k+1];
            for (int c = 0; c < w; ++c) {
                o[2*c]   = a[2*c] * cr + a[2*c+1] * ci;
                o[2*c+1] = a[2*c] * ci - a[2*c+1] * cr;
            }
        }
        break;
    }
    }
}

/* public 1-D call: strides in COMPLEX elements */
static __attribute__((unused)) void pln_xplan_exec(pln_xplan *p,
        const double _Complex *src, ptrdiff_t ss,
        double _Complex *dst, ptrdiff_t ds, int w)
{
    pln_xexec(p->root, (const double *)src, 2 * ss, (double *)dst, 2 * ds, w,
              p->arena, 2 * (ptrdiff_t)p->tile);
}

/* ------------------------------------------------------------------ */
/* 3-D row-column wrapper: axis 0, axis 1, then transpose + axis 2      */
/* ------------------------------------------------------------------ */
typedef struct pln_p3d {
    int L;
    pln_xplan *xp;
    double *tmp;                 /* one volume */
    double *P, *Q;               /* two L x L planes */
} pln_p3d;

static __attribute__((unused)) pln_p3d *pln_p3d_build(int L, const pln_node *t)
{
    pln_p3d *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->xp = pln_xplan_build(t, L >= 40 ? 64 : 32);   /* measured crossover */
    size_t vol = (size_t)L * L * L, pl = (size_t)L * L;
    int a = posix_memalign((void **)&p->tmp, 64, 2 * vol * sizeof(double));
    int b = posix_memalign((void **)&p->P, 64, 2 * pl * sizeof(double));
    int c = posix_memalign((void **)&p->Q, 64, 2 * pl * sizeof(double));
    if (!p->xp || a || b || c) {
        if (p->xp) pln_xplan_free(p->xp);
        free(p->tmp); free(p->P); free(p->Q); free(p);
        return NULL;
    }
    return p;
}

static __attribute__((unused)) void pln_p3d_free(pln_p3d *p)
{
    if (!p) return;
    pln_xplan_free(p->xp); free(p->tmp); free(p->P); free(p->Q); free(p);
}

static void pln_transpose(const double *restrict s, double *restrict d, int L)
{
    for (int y0 = 0; y0 < L; y0 += 16)
        for (int z0 = 0; z0 < L; z0 += 16) {
            int y1 = y0 + 16 < L ? y0 + 16 : L, z1 = z0 + 16 < L ? z0 + 16 : L;
            for (int y = y0; y < y1; ++y)
                for (int z = z0; z < z1; ++z) {
                    d[2*((size_t)z*L + y)]     = s[2*((size_t)y*L + z)];
                    d[2*((size_t)z*L + y) + 1] = s[2*((size_t)y*L + z) + 1];
                }
        }
}

/* one L^3 volume, out-of-place, in unchanged */
static __attribute__((unused)) void pln_p3d_exec(pln_p3d *p,
        const double _Complex *in, double _Complex *out)
{
    const int L = p->L;
    const size_t L2 = (size_t)L * L;
    const double *src = (const double *)in;
    double *dst = (double *)out;
    pln_x *root = p->xp->root;
    double *arena = p->xp->arena;
    const int T = p->xp->tile;
    const ptrdiff_t bs = 2 * (ptrdiff_t)T;

    for (size_t off = 0; off < L2; off += T) {               /* axis 0: in -> out */
        int w = (int)(L2 - off < (size_t)T ? L2 - off : (size_t)T);
        pln_xexec(root, src + 2*off, 2*(ptrdiff_t)L2, dst + 2*off, 2*(ptrdiff_t)L2, w, arena, bs);
    }
    for (int xx = 0; xx < L; ++xx) {                         /* axis 1: out -> tmp */
        const double *s = dst + 2 * (size_t)xx * L2;
        double *d = p->tmp + 2 * (size_t)xx * L2;
        for (int off = 0; off < L; off += T) {
            int w = L - off < T ? L - off : T;
            pln_xexec(root, s + 2*off, 2*L, d + 2*off, 2*L, w, arena, bs);
        }
    }
    for (int xx = 0; xx < L; ++xx) {                         /* axis 2: tmp -> out */
        pln_transpose(p->tmp + 2 * (size_t)xx * L2, p->P, L);
        for (int off = 0; off < L; off += T) {
            int w = L - off < T ? L - off : T;
            pln_xexec(root, p->P + 2*off, 2*L, p->Q + 2*off, 2*L, w, arena, bs);
        }
        pln_transpose(p->Q, dst + 2 * (size_t)xx * L2, L);
    }
}

/* ================================================================== */
/* standalone fft3d entry (omitted when adopted as a library)          */
/* ================================================================== */
#ifndef GEN_PLANNER_LIB

#include <time.h>
#include "../fft3d_api.h"

struct fft3d_plan {
    int L, batch;
    pln_p3d *p3;
    char picked[96];
};

const char *fft3d_name(void) { return "gen_planner"; }
const char *fft3d_description(void)
{
    return "planner layer: L -> {ct,gt-pfa,rader,bluestein,dense} candidate trees + generic "
           "strided-row executor, any 2<=L<=128; adopt via GEN_PLANNER_LIB include";
}
int fft3d_supports(int L) { return L >= 2 && L <= 128; }

static double pln_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* create()-time race over the top candidates: the gen_race composition demo.
 * Off by default so independent processes pick identical plans (bit-identical
 * output across runs); enable with GEN_PLANNER_RACE=1. */
static pln_p3d *pln_race(int L, pln_cand *cand, int k, char *picked, size_t pcap)
{
    if (k > 4) k = 4;
    size_t vol = (size_t)L * L * L;
    double _Complex *a, *b;
    if (posix_memalign((void **)&a, 64, vol * sizeof *a)) return NULL;
    if (posix_memalign((void **)&b, 64, vol * sizeof *b)) { free(a); return NULL; }
    unsigned long s = 12345;
    double *ad = (double *)a;
    for (size_t i = 0; i < 2 * vol; ++i) {
        s = s * 6364136223846793005UL + 1442695040888963407UL;
        ad[i] = (double)(s >> 12) / (double)(1UL << 52) - 0.5;
    }
    pln_p3d *bestp = NULL;
    double bestt = 1e300;
    for (int i = 0; i < k; ++i) {
        pln_p3d *p = pln_p3d_build(L, cand[i].t);
        if (!p) continue;
        pln_p3d_exec(p, a, b);                            /* warm */
        double t = 1e300;
        for (int rep = 0; rep < 3; ++rep) {
            double t0 = pln_now();
            pln_p3d_exec(p, a, b);
            double dt = pln_now() - t0;
            if (dt < t) t = dt;
        }
        if (t < bestt) {
            if (bestp) pln_p3d_free(bestp);
            bestp = p; bestt = t;
            snprintf(picked, pcap, "%s", cand[i].name);
        } else {
            pln_p3d_free(p);
        }
    }
    free(a); free(b);
    return bestp;
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L)) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    pln_arena *A = calloc(1, sizeof *A);
    if (!p || !A) { free(p); free(A); return NULL; }
    p->L = L; p->batch = batch;

    pln_cand cand[12];
    int k = pln_enumerate(A, L, cand, 12);
    if (k <= 0) { free(p); free(A); return NULL; }

    const char *race = getenv("GEN_PLANNER_RACE");
    if (race && race[0] == '1')
        p->p3 = pln_race(L, cand, k, p->picked, sizeof p->picked);
    if (!p->p3) {
        p->p3 = pln_p3d_build(L, cand[0].t);
        snprintf(p->picked, sizeof p->picked, "%s", cand[0].name);
    }
    free(A);                     /* exec plan owns copies of everything it needs */
    if (!p->p3) { free(p); return NULL; }
    return p;
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    const size_t vol = (size_t)p->L * p->L * p->L;
    for (int b = 0; b < p->batch; ++b)
        pln_p3d_exec(p->p3, in + (size_t)b * vol, out + (size_t)b * vol);
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    pln_p3d_free(p->p3);
    free(p);
}

#endif /* GEN_PLANNER_LIB */
#endif /* GEN_PLANNER_C_INCLUDED */
