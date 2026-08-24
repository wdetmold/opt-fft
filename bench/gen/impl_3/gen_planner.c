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
 * is a complete entry that supports EVERY 2 <= L <= 128: it enumerates, then
 * RACES the top 4 candidates on the actual chain step at create() (min-of-3,
 * 2% simplest-first hysteresis) and picks the measured winner -- the brief's
 * plan-time-race deliverable.  The raced pick is persisted per (host, L)
 * through gen_race's string wisdom (gen_r3 adoption): a second process reuses
 * the first's tree, so cross-process repeatability is structural and warm
 * create() is a file read + plan build (~4 ms measured).  Set
 * GEN_PLANNER_RACE=0 for the deterministic model pick with no wisdom I/O
 * (bit-identical plans across processes, dev-harness cmp).
 * GEN_PLANNER_VERBOSE=1 prints the picked tree; GEN_PLANNER_TILE overrides
 * the scratch row width for A/B runs.  It also exports fft3d_chain: the whole
 * graded m-step map chain runs VOLUME-MAJOR and fully in place (state + c are
 * the only volume-sized streams), map fused per x-plane while the plane is
 * cache-hot, gated at create() against execute + the exact scalar map.
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
#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifndef PLN_TI
#define PLN_TI 64               /* MAX row width in complex; the actual tile is a
                                   per-plan runtime choice (32 everywhere since
                                   gen_r3's intrinsic kernels; GEN_PLANNER_TILE
                                   overrides for A/B) */
#endif
#define PLN_NMAX 256            /* largest sub-length (Bluestein M for L=128) */
#define PLN_POOL 8192
#define PLN_PIL 3.141592653589793238462643383279502884L
#define PLN_HPAD(h) (((h) + 3) & ~3)   /* matrix row-count pad for the k-tiled kernels */

/* ------------------------------------------------------------------ */
/* AVX-512 complex-vector helpers (gen_r3).  A zmm holds 4 interleaved  */
/* complex.  cmul by a scalar twiddle = 1 permute + mul + fmaddsub;     */
/* a +- i*b = 1 permute + fmaddsub/fmsubadd -- no even/odd gathers, no  */
/* scalar residue.  Masked loads/stores make the column tail exact and  */
/* OOB-safe (the last tile's rows may end at the buffer edge).          */
/* ------------------------------------------------------------------ */
#if defined(__AVX512F__) && defined(__AVX512DQ__)
#define PLN_SIMD 1
typedef __m512d pv;
#define PLN_MK(w, c) ((w) - (c) >= 4 ? (__mmask8)0xFF \
                                     : (__mmask8)((1u << (2*((w)-(c)))) - 1))
static inline pv pv_ld(__mmask8 k, const double *p) { return _mm512_maskz_loadu_pd(k, p); }
static inline void pv_st(double *p, __mmask8 k, pv v) { _mm512_mask_storeu_pd(p, k, v); }
static inline pv pv_swap(pv v) { return _mm512_permute_pd(v, 0x55); }
/* v * (wr + i*wi): even lane r*wr - i*wi, odd lane i*wr + r*wi */
static inline pv pv_cmul(pv v, pv wr, pv wi)
{ return _mm512_fmaddsub_pd(v, wr, _mm512_mul_pd(pv_swap(v), wi)); }
static inline pv pv_addi(pv a, pv b)      /* a + i*b */
{ return _mm512_fmaddsub_pd(a, _mm512_set1_pd(1.0), pv_swap(b)); }
static inline pv pv_subi(pv a, pv b)      /* a - i*b */
{ return _mm512_fmsubadd_pd(a, _mm512_set1_pd(1.0), pv_swap(b)); }
static inline pv pv_conj(pv v)            /* negate odd lanes */
{
    const pv s = _mm512_castsi512_pd(_mm512_set_epi64(
        (long long)0x8000000000000000LL, 0, (long long)0x8000000000000000LL, 0,
        (long long)0x8000000000000000LL, 0, (long long)0x8000000000000000LL, 0));
    return _mm512_xor_pd(v, s);
}
static inline pv pv_mulmi(pv v)           /* v * (-i): (r,i) -> (i,-r) */
{ return pv_conj(pv_swap(v)); }
#else
#define PLN_SIMD 0
#endif

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
    /* folded odd dense (n >= 11): flops are 2n^2; the gen_r3 register-tiled
     * kernel runs near the port floor, so the measured constant dropped from
     * 5.5n^2 to ~3n^2 -- on the node the race now picks d31 OVER rad31
     * (203 vs the r2 rad31 533), i.e. the dense->Rader crossover moved past
     * p = 31.  Constant set so d-leaves rank realistically in top-4 cuts. */
    default: return (r & 1) && r >= 11 ? 3.0 * r * r + 8.0 * r
                                       : 8.0 * r * r;
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
    else if ((n & 1) ? n <= 63 : n <= 40)     /* folded odd dense reaches further */
        PLN_ADD(pln_new(A, PLN_LEAF, n, 0, 0, NULL, NULL), pln_leafF(n));

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

/* fused-twiddle leaf variants: row j1 is multiplied by tw[j1-1] while loading
 * (kills the separate rowscale pass over the CT buffer -- one less read+write
 * of every point, and 4 FMA-friendly flops instead of 6 rowscale flops) */
#define PLN_LDTW(vr, vi, base, tj)                                            \
    double vr, vi;                                                            \
    do { double _r = (base)[2*c], _i = (base)[2*c+1];                         \
         double _wr = tw[2*(tj)], _wi = tw[2*(tj)+1];                         \
         vr = _r * _wr - _i * _wi; vi = _r * _wi + _i * _wr; } while (0)

static void pln_leaf2_tw(const double *restrict in, ptrdiff_t is,
                         double *restrict out, ptrdiff_t os, int w,
                         const double *restrict tw)
{
    for (int c = 0; c < w; ++c) {
        double ar = in[2*c], ai = in[2*c+1];
        PLN_LDTW(br, bi, in + is, 0);
        out[2*c] = ar + br; out[2*c+1] = ai + bi;
        out[os+2*c] = ar - br; out[os+2*c+1] = ai - bi;
    }
}

static void pln_leaf3_tw(const double *restrict in, ptrdiff_t is,
                         double *restrict out, ptrdiff_t os, int w,
                         const double *restrict tw)
{
    const double C = -0.5, S = 0.86602540378443864676;
    for (int c = 0; c < w; ++c) {
        double x0r = in[2*c], x0i = in[2*c+1];
        PLN_LDTW(x1r, x1i, in + is, 0);
        PLN_LDTW(x2r, x2i, in + 2*is, 1);
        double tr = x1r + x2r, ti = x1i + x2i;
        double ur = x1r - x2r, ui = x1i - x2i;
        out[2*c] = x0r + tr; out[2*c+1] = x0i + ti;
        double vr = x0r + C*tr, vi = x0i + C*ti;
        out[os+2*c]     = vr + S*ui; out[os+2*c+1]     = vi - S*ur;
        out[2*os+2*c]   = vr - S*ui; out[2*os+2*c+1]   = vi + S*ur;
    }
}

static void pln_leaf4_tw(const double *restrict in, ptrdiff_t is,
                         double *restrict out, ptrdiff_t os, int w,
                         const double *restrict tw)
{
    for (int c = 0; c < w; ++c) {
        double x0r = in[2*c], x0i = in[2*c+1];
        PLN_LDTW(x1r, x1i, in + is, 0);
        PLN_LDTW(x2r, x2i, in + 2*is, 1);
        PLN_LDTW(x3r, x3i, in + 3*is, 2);
        double t0r = x0r + x2r, t0i = x0i + x2i;
        double t1r = x0r - x2r, t1i = x0i - x2i;
        double t2r = x1r + x3r, t2i = x1i + x3i;
        double t3r = x1r - x3r, t3i = x1i - x3i;
        out[2*c] = t0r + t2r;        out[2*c+1] = t0i + t2i;
        out[os+2*c] = t1r + t3i;     out[os+2*c+1] = t1i - t3r;
        out[2*os+2*c] = t0r - t2r;   out[2*os+2*c+1] = t0i - t2i;
        out[3*os+2*c] = t1r - t3i;   out[3*os+2*c+1] = t1i + t3r;
    }
}

static void pln_leaf5_tw(const double *restrict in, ptrdiff_t is,
                         double *restrict out, ptrdiff_t os, int w,
                         const double *restrict tw)
{
    const double C1 = 0.30901699437494742410, C2 = -0.80901699437494742410;
    const double S1 = 0.95105651629515357212, S2 =  0.58778525229247312917;
    for (int c = 0; c < w; ++c) {
        double x0r = in[2*c], x0i = in[2*c+1];
        PLN_LDTW(x1r, x1i, in + is, 0);
        PLN_LDTW(x2r, x2i, in + 2*is, 1);
        PLN_LDTW(x3r, x3i, in + 3*is, 2);
        PLN_LDTW(x4r, x4i, in + 4*is, 3);
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

static void pln_leaf8_tw(const double *restrict in, ptrdiff_t is,
                         double *restrict out, ptrdiff_t os, int w,
                         const double *restrict tw)
{
    const double S = 0.70710678118654752440;
    for (int c = 0; c < w; ++c) {
        double ar[4], ai[4], br[4], bi[4];
        {   /* j = 0 untwiddled */
            double xr = in[2*c], xi = in[2*c+1];
            PLN_LDTW(yr, yi, in + 4*is, 3);
            ar[0] = xr + yr; ai[0] = xi + yi;
            br[0] = xr - yr; bi[0] = xi - yi;
        }
        for (int j = 1; j < 4; ++j) {
            PLN_LDTW(xr, xi, in + j*is, j - 1);
            PLN_LDTW(yr, yi, in + (j+4)*is, j + 3);
            ar[j] = xr + yr; ai[j] = xi + yi;
            br[j] = xr - yr; bi[j] = xi - yi;
        }
        { double r = br[1], i = bi[1]; br[1] = S*(r + i); bi[1] = S*(i - r); }
        { double r = br[2], i = bi[2]; br[2] = i;         bi[2] = -r; }
        { double r = br[3], i = bi[3]; br[3] = S*(i - r); bi[3] = S*(-i - r); }
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

#if PLN_SIMD
/* ------------------------------------------------------------------ */
/* explicit AVX-512 leaves (gen_r3).  One body per radix; HAS is a      */
/* compile-time constant through always_inline, so the twiddle loads    */
/* and the branch fold away in the plain instantiation.  gcc-11's       */
/* autovectorizer left these loops half scalar (277 vmovsd / 68 vmulsd  */
/* in pln_xexec on the node build) -- hence hand intrinsics.            */
/* ------------------------------------------------------------------ */
static inline __attribute__((always_inline)) void pln_lv2(
    const double *restrict in, ptrdiff_t is, double *restrict out, ptrdiff_t os,
    int w, const double *restrict tw, int HAS)
{
    pv w1r = _mm512_setzero_pd(), w1i = w1r;
    if (HAS) { w1r = _mm512_set1_pd(tw[0]); w1i = _mm512_set1_pd(tw[1]); }
    for (int c = 0; c < w; c += 4) {
        __mmask8 mk = PLN_MK(w, c);
        pv a = pv_ld(mk, in + 2*c);
        pv b = pv_ld(mk, in + is + 2*c);
        if (HAS) b = pv_cmul(b, w1r, w1i);
        pv_st(out + 2*c, mk, _mm512_add_pd(a, b));
        pv_st(out + os + 2*c, mk, _mm512_sub_pd(a, b));
    }
}

static inline __attribute__((always_inline)) void pln_lv3(
    const double *restrict in, ptrdiff_t is, double *restrict out, ptrdiff_t os,
    int w, const double *restrict tw, int HAS)
{
    const pv C = _mm512_set1_pd(-0.5), S = _mm512_set1_pd(0.86602540378443864676);
    pv wr[2], wi[2];
    if (HAS) for (int t = 0; t < 2; ++t) {
        wr[t] = _mm512_set1_pd(tw[2*t]); wi[t] = _mm512_set1_pd(tw[2*t+1]);
    }
    for (int c = 0; c < w; c += 4) {
        __mmask8 mk = PLN_MK(w, c);
        pv x0 = pv_ld(mk, in + 2*c);
        pv x1 = pv_ld(mk, in + is + 2*c);
        pv x2 = pv_ld(mk, in + 2*is + 2*c);
        if (HAS) { x1 = pv_cmul(x1, wr[0], wi[0]); x2 = pv_cmul(x2, wr[1], wi[1]); }
        pv t = _mm512_add_pd(x1, x2), u = _mm512_sub_pd(x1, x2);
        pv_st(out + 2*c, mk, _mm512_add_pd(x0, t));
        pv v = _mm512_fmadd_pd(C, t, x0);
        pv su = _mm512_mul_pd(S, u);
        pv_st(out + os + 2*c, mk, pv_subi(v, su));
        pv_st(out + 2*os + 2*c, mk, pv_addi(v, su));
    }
}

static inline __attribute__((always_inline)) void pln_lv4(
    const double *restrict in, ptrdiff_t is, double *restrict out, ptrdiff_t os,
    int w, const double *restrict tw, int HAS)
{
    pv wr[3], wi[3];
    if (HAS) for (int t = 0; t < 3; ++t) {
        wr[t] = _mm512_set1_pd(tw[2*t]); wi[t] = _mm512_set1_pd(tw[2*t+1]);
    }
    for (int c = 0; c < w; c += 4) {
        __mmask8 mk = PLN_MK(w, c);
        pv x0 = pv_ld(mk, in + 2*c);
        pv x1 = pv_ld(mk, in + is + 2*c);
        pv x2 = pv_ld(mk, in + 2*is + 2*c);
        pv x3 = pv_ld(mk, in + 3*is + 2*c);
        if (HAS) {
            x1 = pv_cmul(x1, wr[0], wi[0]);
            x2 = pv_cmul(x2, wr[1], wi[1]);
            x3 = pv_cmul(x3, wr[2], wi[2]);
        }
        pv t0 = _mm512_add_pd(x0, x2), t1 = _mm512_sub_pd(x0, x2);
        pv t2 = _mm512_add_pd(x1, x3), t3 = _mm512_sub_pd(x1, x3);
        pv_st(out + 2*c, mk, _mm512_add_pd(t0, t2));
        pv_st(out + os + 2*c, mk, pv_subi(t1, t3));
        pv_st(out + 2*os + 2*c, mk, _mm512_sub_pd(t0, t2));
        pv_st(out + 3*os + 2*c, mk, pv_addi(t1, t3));
    }
}

static inline __attribute__((always_inline)) void pln_lv5(
    const double *restrict in, ptrdiff_t is, double *restrict out, ptrdiff_t os,
    int w, const double *restrict tw, int HAS)
{
    const pv C1 = _mm512_set1_pd(0.30901699437494742410);
    const pv C2 = _mm512_set1_pd(-0.80901699437494742410);
    const pv S1 = _mm512_set1_pd(0.95105651629515357212);
    const pv S2 = _mm512_set1_pd(0.58778525229247312917);
    pv wr[4], wi[4];
    if (HAS) for (int t = 0; t < 4; ++t) {
        wr[t] = _mm512_set1_pd(tw[2*t]); wi[t] = _mm512_set1_pd(tw[2*t+1]);
    }
    for (int c = 0; c < w; c += 4) {
        __mmask8 mk = PLN_MK(w, c);
        pv x0 = pv_ld(mk, in + 2*c);
        pv x1 = pv_ld(mk, in + is + 2*c);
        pv x2 = pv_ld(mk, in + 2*is + 2*c);
        pv x3 = pv_ld(mk, in + 3*is + 2*c);
        pv x4 = pv_ld(mk, in + 4*is + 2*c);
        if (HAS) {
            x1 = pv_cmul(x1, wr[0], wi[0]);
            x2 = pv_cmul(x2, wr[1], wi[1]);
            x3 = pv_cmul(x3, wr[2], wi[2]);
            x4 = pv_cmul(x4, wr[3], wi[3]);
        }
        pv t1 = _mm512_add_pd(x1, x4), t2 = _mm512_add_pd(x2, x3);
        pv t3 = _mm512_sub_pd(x1, x4), t4 = _mm512_sub_pd(x2, x3);
        pv_st(out + 2*c, mk, _mm512_add_pd(x0, _mm512_add_pd(t1, t2)));
        pv a = _mm512_fmadd_pd(C1, t1, _mm512_fmadd_pd(C2, t2, x0));
        pv b = _mm512_fnmadd_pd(S1, t3, _mm512_mul_pd(_mm512_set1_pd(-0.58778525229247312917), t4));
        pv_st(out + os + 2*c, mk, pv_addi(a, b));
        pv_st(out + 4*os + 2*c, mk, pv_subi(a, b));
        pv cc = _mm512_fmadd_pd(C2, t1, _mm512_fmadd_pd(C1, t2, x0));
        pv d = _mm512_fnmadd_pd(S2, t3, _mm512_mul_pd(S1, t4));
        pv_st(out + 2*os + 2*c, mk, pv_addi(cc, d));
        pv_st(out + 3*os + 2*c, mk, pv_subi(cc, d));
    }
}

static inline __attribute__((always_inline)) void pln_lv8(
    const double *restrict in, ptrdiff_t is, double *restrict out, ptrdiff_t os,
    int w, const double *restrict tw, int HAS)
{
    const double Sc = 0.70710678118654752440;
    const pv Sp = _mm512_set1_pd(Sc), Sn = _mm512_set1_pd(-Sc);
    for (int c = 0; c < w; c += 4) {
        __mmask8 mk = PLN_MK(w, c);
        pv a[4], b[4];
        for (int j = 0; j < 4; ++j) {
            pv x = pv_ld(mk, in + j*is + 2*c);
            pv y = pv_ld(mk, in + (j+4)*is + 2*c);
            if (HAS) {
                if (j) x = pv_cmul(x, _mm512_set1_pd(tw[2*(j-1)]),
                                      _mm512_set1_pd(tw[2*(j-1)+1]));
                y = pv_cmul(y, _mm512_set1_pd(tw[2*(j+3)]),
                               _mm512_set1_pd(tw[2*(j+3)+1]));
            }
            a[j] = _mm512_add_pd(x, y);
            b[j] = _mm512_sub_pd(x, y);
        }
        b[1] = pv_cmul(b[1], Sp, Sn);       /* * S*(1-i) */
        b[2] = pv_mulmi(b[2]);              /* * -i      */
        b[3] = pv_cmul(b[3], Sn, Sn);       /* * S*(-1-i) */
        pv t0 = _mm512_add_pd(a[0], a[2]), t1 = _mm512_sub_pd(a[0], a[2]);
        pv t2 = _mm512_add_pd(a[1], a[3]), t3 = _mm512_sub_pd(a[1], a[3]);
        pv_st(out + 2*c, mk, _mm512_add_pd(t0, t2));
        pv_st(out + 2*os + 2*c, mk, pv_subi(t1, t3));
        pv_st(out + 4*os + 2*c, mk, _mm512_sub_pd(t0, t2));
        pv_st(out + 6*os + 2*c, mk, pv_addi(t1, t3));
        t0 = _mm512_add_pd(b[0], b[2]); t1 = _mm512_sub_pd(b[0], b[2]);
        t2 = _mm512_add_pd(b[1], b[3]); t3 = _mm512_sub_pd(b[1], b[3]);
        pv_st(out + os + 2*c, mk, _mm512_add_pd(t0, t2));
        pv_st(out + 3*os + 2*c, mk, pv_subi(t1, t3));
        pv_st(out + 5*os + 2*c, mk, _mm512_sub_pd(t0, t2));
        pv_st(out + 7*os + 2*c, mk, pv_addi(t1, t3));
    }
}
#endif /* PLN_SIMD */

static void pln_leaf_apply_tw(int r, const double *in, ptrdiff_t is,
                              double *out, ptrdiff_t os, int w, const double *tw)
{
#if PLN_SIMD
    switch (r) {
    case 2: pln_lv2(in, is, out, os, w, tw, 1); break;
    case 3: pln_lv3(in, is, out, os, w, tw, 1); break;
    case 4: pln_lv4(in, is, out, os, w, tw, 1); break;
    case 5: pln_lv5(in, is, out, os, w, tw, 1); break;
    default: pln_lv8(in, is, out, os, w, tw, 1); break;
    }
#else
    switch (r) {
    case 2: pln_leaf2_tw(in, is, out, os, w, tw); break;
    case 3: pln_leaf3_tw(in, is, out, os, w, tw); break;
    case 4: pln_leaf4_tw(in, is, out, os, w, tw); break;
    case 5: pln_leaf5_tw(in, is, out, os, w, tw); break;
    default: pln_leaf8_tw(in, is, out, os, w, tw); break;
    }
#endif
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

/* folded odd-n dense DFT: X_k = E_k - i O_k, X_{n-k} = E_k + i O_k with
 * E_k = x0 + sum_j C[k][j] (x_j + x_{n-j}),  O_k = sum_j S[k][j] (x_j - x_{n-j})
 * -- 2n^2 flops instead of 8n^2, all coefficients REAL (one add/FMA per zmm on
 * interleaved data; the gen_rader / gen_dense_prime conjugate fold, generalized).
 * A = arena staging for the 2h+1 fold rows; all src reads precede dst writes,
 * so src == dst is safe. */
static void pln_dense_fold_apply(const double *restrict mat, int n,
                                 const double *restrict in, ptrdiff_t is,
                                 double *restrict out, ptrdiff_t os, int w,
                                 double *restrict A, ptrdiff_t bs)
{
    const int h = n / 2;
    memcpy(A, in, 2 * w * sizeof(double));               /* row 0 */
    for (int j = 1; j <= h; ++j) {
        const double *xj = in + (ptrdiff_t)j * is, *xn = in + (ptrdiff_t)(n - j) * is;
        double *sj = A + (size_t)j * bs, *dj = A + (size_t)(h + j) * bs;
        for (int c = 0; c < 2 * w; ++c) { sj[c] = xj[c] + xn[c]; dj[c] = xj[c] - xn[c]; }
    }
    {
        double acc[2 * PLN_TI];
        memcpy(acc, A, 2 * w * sizeof(double));
        for (int j = 1; j <= h; ++j) {
            const double *sj = A + (size_t)j * bs;
            for (int c = 0; c < 2 * w; ++c) acc[c] += sj[c];
        }
        for (int c = 0; c < 2 * w; ++c) out[c] = acc[c];
    }
    const double *C = mat, *S = mat + (size_t)PLN_HPAD(h) * h;
    for (int k = 1; k <= h; ++k) {
        double er[2 * PLN_TI], oc[2 * PLN_TI];
        memcpy(er, A, 2 * w * sizeof(double));           /* E starts from x0 */
        for (int c = 0; c < 2 * w; ++c) oc[c] = 0.0;
        const double *Ck = C + (size_t)(k - 1) * h, *Sk = S + (size_t)(k - 1) * h;
        for (int j = 1; j <= h; ++j) {
            const double cc = Ck[j-1], ss = Sk[j-1];
            const double *sj = A + (size_t)j * bs, *dj = A + (size_t)(h + j) * bs;
            for (int c = 0; c < 2 * w; ++c) { er[c] += cc * sj[c]; oc[c] += ss * dj[c]; }
        }
        double *ok_ = out + (ptrdiff_t)k * os, *onk = out + (ptrdiff_t)(n - k) * os;
        for (int c = 0; c < w; ++c) {
            ok_[2*c]   = er[2*c]   + oc[2*c+1];
            ok_[2*c+1] = er[2*c+1] - oc[2*c];
            onk[2*c]   = er[2*c]   - oc[2*c+1];
            onk[2*c+1] = er[2*c+1] + oc[2*c];
        }
    }
}

#if PLN_SIMD
/* ------------------------------------------------------------------ */
/* register-tiled matrix kernels (gen_r3).  The gen_r2 loops kept their */
/* accumulators in stack arrays, so gcc emitted ~4 loads per FMA (295   */
/* vmovupd vs 70 FMA in the fold kernel).  Tiling 4 output rows x 2 zmm */
/* of columns holds 16 accumulators in registers: per input row j the   */
/* kernel now does 16 FMA against 4 row loads + 8 broadcasts.  Matrix   */
/* row counts are padded to a multiple of 4 (PLN_HPAD, zero rows), so   */
/* there is no K-tail variant -- padded outputs are computed and simply */
/* not stored.  gen_dense_prime's k-quad x wide-tile shape, generalized.*/
/* ------------------------------------------------------------------ */

/* dense complex matrix, SIMD layout (pln_dense_matrix): WR[np][n] real
 * parts k-major, then WP[np][n] 16-byte pairs (-wi, +wi) for a
 * broadcast_f64x2 straight into the alternating-sign FMA. */
static void pln_dense_applyv(const double *restrict mat, int n,
                             const double *restrict in, ptrdiff_t is,
                             double *restrict out, ptrdiff_t os, int w)
{
    const int np = PLN_HPAD(n);
    const double *WR = mat, *WP = mat + (size_t)np * n;
    for (int kb = 0; kb < n; kb += 4) {
        for (int c = 0; c < w; c += 8) {
            __mmask8 m0 = PLN_MK(w, c);
            __mmask8 m1 = (w - c > 4) ? PLN_MK(w, c + 4) : 0;
            pv acc[4][2];
            for (int t = 0; t < 4; ++t)
                acc[t][0] = acc[t][1] = _mm512_setzero_pd();
            for (int j = 0; j < n; ++j) {
                const double *r = in + (ptrdiff_t)j * is + 2*c;
                pv v0 = pv_ld(m0, r), v1 = pv_ld(m1, r + 8);
                pv s0 = pv_swap(v0), s1 = pv_swap(v1);
                for (int t = 0; t < 4; ++t) {
                    const size_t e = (size_t)(kb + t) * n + j;
                    pv wr = _mm512_set1_pd(WR[e]);
                    pv wp = _mm512_broadcast_f64x2(_mm_loadu_pd(WP + 2*e));
                    acc[t][0] = _mm512_fmadd_pd(v0, wr, acc[t][0]);
                    acc[t][0] = _mm512_fmadd_pd(s0, wp, acc[t][0]);
                    acc[t][1] = _mm512_fmadd_pd(v1, wr, acc[t][1]);
                    acc[t][1] = _mm512_fmadd_pd(s1, wp, acc[t][1]);
                }
            }
            for (int t = 0; t < 4 && kb + t < n; ++t) {
                double *o = out + (ptrdiff_t)(kb + t) * os + 2*c;
                pv_st(o, m0, acc[t][0]);
                if (m1) pv_st(o + 8, m1, acc[t][1]);
            }
        }
    }
}

/* folded odd dense, SIMD: same fold algebra as the scalar version, the
 * E/O accumulation tiled 4 k-rows x 2 zmm.  Staging pass builds A and
 * X[0] in one sweep per column chunk (in-place safe: only row 0 of out
 * -- fully consumed for that chunk -- is written before A is complete). */
static void pln_fold_applyv(const double *restrict mat, int n,
                            const double *restrict in, ptrdiff_t is,
                            double *restrict out, ptrdiff_t os, int w,
                            double *restrict A, ptrdiff_t bs)
{
    const int h = n / 2;
    const double *C = mat, *S = mat + (size_t)PLN_HPAD(h) * h;
    for (int c = 0; c < w; c += 4) {
        __mmask8 mk = PLN_MK(w, c);
        pv x0 = pv_ld(mk, in + 2*c);
        pv_st(A + 2*c, mk, x0);
        pv acc = x0;
        for (int j = 1; j <= h; ++j) {
            pv xj = pv_ld(mk, in + (ptrdiff_t)j * is + 2*c);
            pv xn = pv_ld(mk, in + (ptrdiff_t)(n - j) * is + 2*c);
            pv s = _mm512_add_pd(xj, xn);
            pv_st(A + (size_t)j * bs + 2*c, mk, s);
            pv_st(A + (size_t)(h + j) * bs + 2*c, mk, _mm512_sub_pd(xj, xn));
            acc = _mm512_add_pd(acc, s);
        }
        pv_st(out + 2*c, mk, acc);
    }
    for (int kb = 1; kb <= h; kb += 4) {
        for (int c = 0; c < w; c += 8) {
            __mmask8 m0 = PLN_MK(w, c);
            __mmask8 m1 = (w - c > 4) ? PLN_MK(w, c + 4) : 0;
            pv e0 = pv_ld(m0, A + 2*c), e1 = pv_ld(m1, A + 2*c + 8);
            pv er[4][2], oc[4][2];
            for (int t = 0; t < 4; ++t) {
                er[t][0] = e0; er[t][1] = e1;
                oc[t][0] = oc[t][1] = _mm512_setzero_pd();
            }
            for (int j = 1; j <= h; ++j) {
                const double *sj = A + (size_t)j * bs + 2*c;
                const double *dj = A + (size_t)(h + j) * bs + 2*c;
                pv s0 = pv_ld(m0, sj), s1 = pv_ld(m1, sj + 8);
                pv d0 = pv_ld(m0, dj), d1 = pv_ld(m1, dj + 8);
                for (int t = 0; t < 4; ++t) {
                    const size_t e = (size_t)(kb - 1 + t) * h + (j - 1);
                    pv cc = _mm512_set1_pd(C[e]), ss = _mm512_set1_pd(S[e]);
                    er[t][0] = _mm512_fmadd_pd(cc, s0, er[t][0]);
                    er[t][1] = _mm512_fmadd_pd(cc, s1, er[t][1]);
                    oc[t][0] = _mm512_fmadd_pd(ss, d0, oc[t][0]);
                    oc[t][1] = _mm512_fmadd_pd(ss, d1, oc[t][1]);
                }
            }
            for (int t = 0; t < 4 && kb + t <= h; ++t) {
                const int k = kb + t;
                double *ok_ = out + (ptrdiff_t)k * os + 2*c;
                double *onk = out + (ptrdiff_t)(n - k) * os + 2*c;
                pv_st(ok_, m0, pv_subi(er[t][0], oc[t][0]));   /* X_k = E - iO */
                pv_st(onk, m0, pv_addi(er[t][0], oc[t][0]));   /* X_{n-k} = E + iO */
                if (m1) {
                    pv_st(ok_ + 8, m1, pv_subi(er[t][1], oc[t][1]));
                    pv_st(onk + 8, m1, pv_addi(er[t][1], oc[t][1]));
                }
            }
        }
    }
}
#endif /* PLN_SIMD */

/* dispatch shims: one call site shape for both builds */
static void pln_dense_run(const double *mat, int p, const double *in, ptrdiff_t is,
                          double *out, ptrdiff_t os, int w)
{
#if PLN_SIMD
    pln_dense_applyv(mat, p, in, is, out, os, w);
#else
    pln_dense_apply(mat, p, in, is, out, os, w);
#endif
}

static void pln_fold_run(const double *mat, int n, const double *in, ptrdiff_t is,
                         double *out, ptrdiff_t os, int w, double *A, ptrdiff_t bs)
{
#if PLN_SIMD
    pln_fold_applyv(mat, n, in, is, out, os, w, A, bs);
#else
    pln_dense_fold_apply(mat, n, in, is, out, os, w, A, bs);
#endif
}

static void pln_leaf_apply(int r, const double *mat,
                           const double *in, ptrdiff_t is,
                           double *out, ptrdiff_t os, int w)
{
#if PLN_SIMD
    switch (r) {
    case 2: pln_lv2(in, is, out, os, w, NULL, 0); break;
    case 3: pln_lv3(in, is, out, os, w, NULL, 0); break;
    case 4: pln_lv4(in, is, out, os, w, NULL, 0); break;
    case 5: pln_lv5(in, is, out, os, w, NULL, 0); break;
    case 8: pln_lv8(in, is, out, os, w, NULL, 0); break;
    default: pln_dense_run(mat, r, in, is, out, os, w); break;
    }
#else
    switch (r) {
    case 2: pln_leaf2(in, is, out, os, w); break;
    case 3: pln_leaf3(in, is, out, os, w); break;
    case 4: pln_leaf4(in, is, out, os, w); break;
    case 5: pln_leaf5(in, is, out, os, w); break;
    case 8: pln_leaf8(in, is, out, os, w); break;
    default: pln_dense_apply(mat, r, in, is, out, os, w); break;
    }
#endif
}

/* ------------------------------------------------------------------ */
/* executable plan: tree + plan-time tables + shared scratch arena      */
/* ------------------------------------------------------------------ */
typedef struct pln_x {
    int kind, n, r, M;
    int fold;                    /* dense odd leaf/radix uses the folded form */
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
#if PLN_SIMD
    /* tiled-kernel layout: WR[np][n] (k-major, np = PLN_HPAD(n), zero pad
     * rows), then WP[np][n] 16-byte (-wi, +wi) pairs -- see pln_dense_applyv */
    const int np = PLN_HPAD(n);
    double *m = calloc(3 * (size_t)np * n, sizeof *m);
    if (!m) return NULL;
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j) {
            double wr, wi;
            pln_omega((long)k * j, n, &wr, &wi);
            m[(size_t)k*n + j] = wr;
            m[(size_t)np*n + 2*((size_t)k*n + j)]     = -wi;
            m[(size_t)np*n + 2*((size_t)k*n + j) + 1] =  wi;
        }
    return m;
#else
    double *m = malloc(2 * (size_t)n * n * sizeof *m);
    if (!m) return NULL;
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            pln_omega((long)k * j, n, &m[2*((size_t)k*n + j)], &m[2*((size_t)k*n + j) + 1]);
    return m;
#endif
}

/* folded odd-n tables: C[k][j] = cos(2*pi*k*j/n), S[k][j] = sin(2*pi*k*j/n),
 * k,j = 1..h, h = (n-1)/2; C first, then S (real h x h matrices, row count
 * padded to PLN_HPAD(h) with zero rows for the 4-row-tiled kernel) */
static double *pln_fold_matrix(int n)
{
    int h = n / 2, hp = PLN_HPAD(h);
    double *m = calloc(2 * (size_t)hp * h, sizeof *m);
    if (!m) return NULL;
    for (int k = 1; k <= h; ++k)
        for (int j = 1; j <= h; ++j) {
            long double re, im;
            pln_omegal((long)k * j, n, &re, &im);   /* e^{-2pi i kj/n} */
            m[(size_t)(k-1)*h + (j-1)] = (double)re;
            m[(size_t)hp*h + (size_t)(k-1)*h + (j-1)] = (double)(-im);
        }
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
        if (!pln_leaf_hard(t->n)) {
            x->fold = (t->n & 1) && t->n >= 11;   /* below 11 plain dense wins:
                                                     fold overhead beats the
                                                     flop saving (c9(d3) at 27:
                                                     155 -> 228 us folded) */
            x->mat = x->fold ? pln_fold_matrix(t->n) : pln_dense_matrix(t->n);
            own = 2 * (size_t)tile * t->n;    /* staging rows: in-place safety
                                                 + one strided pass instead of p */
        }
        break;
    case PLN_CT: {
        int r = t->r, m = t->n / t->r;
        own = 2 * (size_t)tile * t->n;
        if (!pln_leaf_hard(r) && (r & 1) && r >= 11)
            own += 2 * (size_t)tile * r;      /* fold staging after the buffer */
        /* k2-major, k2 = 1..m-1 (k2 = 0 is twiddle-free): consumption order
           for the fused-twiddle leaves */
        x->tw = malloc(2 * (size_t)(m - 1) * (r - 1) * sizeof *x->tw);
        if (x->tw)
            for (int k2 = 1; k2 < m; ++k2)
                for (int j1 = 1; j1 < r; ++j1)
                    pln_omega((long)j1 * k2, t->n,
                              &x->tw[2*((size_t)(k2-1)*(r-1) + (j1-1))],
                              &x->tw[2*((size_t)(k2-1)*(r-1) + (j1-1)) + 1]);
        if (!pln_leaf_hard(r)) {
            x->fold = (r & 1) && r >= 11;
            x->mat = x->fold ? pln_fold_matrix(r) : pln_dense_matrix(r);
        }
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
#if PLN_SIMD
    const pv vr = _mm512_set1_pd(wr), vi = _mm512_set1_pd(wi);
    for (int c = 0; c < w; c += 4) {
        __mmask8 mk = PLN_MK(w, c);
        pv_st(row + 2*c, mk, pv_cmul(pv_ld(mk, row + 2*c), vr, vi));
    }
#else
    for (int c = 0; c < w; ++c) {
        double re = row[2*c], im = row[2*c+1];
        row[2*c] = re * wr - im * wi;
        row[2*c+1] = re * wi + im * wr;
    }
#endif
}

static void pln_rowscale_conj(double *row, double wr, double wi, int w)
{   /* row <- conj(row * (wr + i wi)) */
#if PLN_SIMD
    const pv vr = _mm512_set1_pd(wr), vi = _mm512_set1_pd(wi);
    for (int c = 0; c < w; c += 4) {
        __mmask8 mk = PLN_MK(w, c);
        pv_st(row + 2*c, mk, pv_conj(pv_cmul(pv_ld(mk, row + 2*c), vr, vi)));
    }
#else
    for (int c = 0; c < w; ++c) {
        double re = row[2*c], im = row[2*c+1];
        row[2*c] = re * wr - im * wi;
        row[2*c+1] = -(re * wi + im * wr);
    }
#endif
}

/* the generic executor: n rows of w complex, src/dst strides in DOUBLES;
 * bs = scratch row stride in doubles (2 * plan tile) */
static void pln_xexec(const pln_x *x, const double *restrict src, ptrdiff_t ss,
                      double *restrict dst, ptrdiff_t ds, int w,
                      double *restrict arena, const ptrdiff_t bs)
{
    switch (x->kind) {
    case PLN_LEAF:
        if (x->mat) {
            double *buf = arena + x->off;
            if (x->fold) {
                pln_fold_run(x->mat, x->n, src, ss, dst, ds, w, buf, bs);
            } else {
                /* stage through the arena: makes the dense leaf in-place safe
                   and replaces p strided sweeps over src with one */
                for (int j = 0; j < x->n; ++j)
                    memcpy(buf + (size_t)j * bs, src + (ptrdiff_t)j * ss,
                           2 * w * sizeof(double));
                pln_dense_run(x->mat, x->n, buf, bs, dst, ds, w);
            }
        } else {
            pln_leaf_apply(x->n, NULL, src, ss, dst, ds, w);
        }
        break;
    case PLN_CT: {
        const int r = x->r, m = x->n / x->r;
        double *buf = arena + x->off;
        for (int j1 = 0; j1 < r; ++j1)
            pln_xexec(x->s1, src + j1 * ss, ss * r, buf + (size_t)j1 * m * bs, bs, w, arena, bs);
        if (x->mat) {                          /* dense radix: twiddles unfused */
            for (int k2 = 1; k2 < m; ++k2) {
                const double *t = x->tw + 2 * (size_t)(k2 - 1) * (r - 1);
                for (int j1 = 1; j1 < r; ++j1)
                    pln_rowscale(buf + ((size_t)j1 * m + k2) * bs,
                                 t[2*(j1-1)], t[2*(j1-1)+1], w);
            }
            for (int k2 = 0; k2 < m; ++k2) {
                if (x->fold)
                    pln_fold_run(x->mat, r, buf + (size_t)k2 * bs, m * bs,
                                 dst + (size_t)k2 * ds, m * ds, w,
                                 buf + (size_t)x->n * bs, bs);
                else
                    pln_dense_run(x->mat, r, buf + (size_t)k2 * bs, m * bs,
                                  dst + (size_t)k2 * ds, m * ds, w);
            }
        } else {                                         /* twiddles fused into the leaf */
            pln_leaf_apply(r, NULL, buf, m * bs, dst, m * ds, w);
            for (int k2 = 1; k2 < m; ++k2)
                pln_leaf_apply_tw(r, buf + (size_t)k2 * bs, m * bs,
                                  dst + (size_t)k2 * ds, m * ds, w,
                                  x->tw + 2 * (size_t)(k2 - 1) * (r - 1));
        }
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
        double sum[2 * PLN_TI], x0s[2 * PLN_TI];
        memcpy(sum, src, 2 * w * sizeof(double));        /* x[0] */
        memcpy(x0s, src, 2 * w * sizeof(double));        /* stashed: in-place safety */
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
#if PLN_SIMD
            for (int c = 0; c < w; c += 4) {             /* X[g^-m] = x0 + conj(a) */
                __mmask8 mk = PLN_MK(w, c);
                pv_st(o + 2*c, mk,
                      _mm512_fmsubadd_pd(pv_ld(mk, x0s + 2*c),
                                         _mm512_set1_pd(1.0), pv_ld(mk, a + 2*c)));
            }
#else
            for (int c = 0; c < w; ++c) {                /* X[g^-m] = x0 + conj(a) */
                o[2*c]   = x0s[2*c]   + a[2*c];
                o[2*c+1] = x0s[2*c+1] - a[2*c+1];
            }
#endif
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
#if PLN_SIMD
            const pv vr = _mm512_set1_pd(cr), vi = _mm512_set1_pd(ci);
            for (int c = 0; c < w; c += 4) {
                __mmask8 mk = PLN_MK(w, c);
                pv_st(a + 2*c, mk, pv_cmul(pv_ld(mk, r + 2*c), vr, vi));
            }
#else
            for (int c = 0; c < w; ++c) {
                a[2*c]   = r[2*c] * cr - r[2*c+1] * ci;
                a[2*c+1] = r[2*c] * ci + r[2*c+1] * cr;
            }
#endif
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
#if PLN_SIMD
            const pv vr = _mm512_set1_pd(cr), vi = _mm512_set1_pd(ci);
            for (int c = 0; c < w; c += 4) {
                __mmask8 mk = PLN_MK(w, c);
                pv v = pv_ld(mk, a + 2*c);
                pv_st(o + 2*c, mk,
                      _mm512_fmsubadd_pd(pv_swap(v), vi, _mm512_mul_pd(v, vr)));
            }
#else
            for (int c = 0; c < w; ++c) {
                o[2*c]   = a[2*c] * cr + a[2*c+1] * ci;
                o[2*c+1] = a[2*c] * ci - a[2*c+1] * cr;
            }
#endif
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
/* 3-D row-column wrapper.  gen_r2 structure: axis 0 streams the volume */
/* once (in -> out, or in place), then each x-plane gets axis 1 IN      */
/* PLACE, a transpose into a padded scratch plane, axis 2 in place      */
/* there, and the transpose back -- the plane stays cache-resident      */
/* across both passes and the tmp volume of gen_r1 is gone.  Every      */
/* pln_xexec node reads all of src before writing dst, so src == dst    */
/* is legal at any root (dense leaf stages via arena; Rader stashes     */
/* x0) -- that is what the in-place passes and the chain rely on.       */
/* ------------------------------------------------------------------ */
typedef struct pln_p3d {
    int L;
    int pitch;                   /* padded row stride of P, complex units:
                                    odd number of cache lines, so the axis-2
                                    strided rows never 4K-alias */
    pln_xplan *xp;
    double *P;                   /* one transposed (z,y) plane, pitch x L */
} pln_p3d;

static __attribute__((unused)) pln_p3d *pln_p3d_build(int L, const pln_node *t)
{
    pln_p3d *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    int lines = (L + 3) / 4;
    if (!(lines & 1)) ++lines;
    p->pitch = 4 * lines;
    {
        int tile = 32;                               /* gen_r3: the r2 L>=40
                                                        crossover to 64 vanished
                                                        with the intrinsic
                                                        kernels (100: 6869@32 vs
                                                        6955@64; 40/50 a wash;
                                                        25: 70.3 vs 77.5) */
        const char *e = getenv("GEN_PLANNER_TILE");  /* dev A/B override */
        if (e && atoi(e) >= 4) tile = atoi(e);
        p->xp = pln_xplan_build(t, tile);
    }
    int b = posix_memalign((void **)&p->P, 64,
                           2 * (size_t)p->pitch * L * sizeof(double));
    if (!p->xp || b) {
        if (p->xp) pln_xplan_free(p->xp);
        free(p->P); free(p);
        return NULL;
    }
    return p;
}

static __attribute__((unused)) void pln_p3d_free(pln_p3d *p)
{
    if (!p) return;
    pln_xplan_free(p->xp); free(p->P); free(p);
}

#if PLN_SIMD
/* 4x4 complex block transpose, strides in doubles: 4 loads, 8 shuffles,
 * 4 full-line stores instead of 16 scalar 16-byte moves */
static inline void pln_tr4x4(const double *restrict s, ptrdiff_t srow,
                             double *restrict d, ptrdiff_t drow)
{
    const __m512i i0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    const __m512i i1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
    pv r0 = _mm512_loadu_pd(s);
    pv r1 = _mm512_loadu_pd(s + srow);
    pv r2 = _mm512_loadu_pd(s + 2*srow);
    pv r3 = _mm512_loadu_pd(s + 3*srow);
    pv a = _mm512_permutex2var_pd(r0, i0, r1);   /* r0g0 r1g0 r0g1 r1g1 */
    pv b = _mm512_permutex2var_pd(r2, i0, r3);
    pv e = _mm512_permutex2var_pd(r0, i1, r1);   /* r0g2 r1g2 r0g3 r1g3 */
    pv f = _mm512_permutex2var_pd(r2, i1, r3);
    _mm512_storeu_pd(d,          _mm512_shuffle_f64x2(a, b, 0x44));
    _mm512_storeu_pd(d + drow,   _mm512_shuffle_f64x2(a, b, 0xEE));
    _mm512_storeu_pd(d + 2*drow, _mm512_shuffle_f64x2(e, f, 0x44));
    _mm512_storeu_pd(d + 3*drow, _mm512_shuffle_f64x2(e, f, 0xEE));
}
#endif

/* (y,z) plane, row stride L -> (z,y) plane, row stride pitch */
static void pln_transpose_in(const double *restrict s, double *restrict d,
                             int L, int pitch)
{
#if PLN_SIMD
    const int L4 = L & ~3;
    for (int y0 = 0; y0 < L4; y0 += 4) {
        for (int z0 = 0; z0 < L4; z0 += 4)
            pln_tr4x4(s + 2*((size_t)y0*L + z0), 2*(ptrdiff_t)L,
                      d + 2*((size_t)z0*pitch + y0), 2*(ptrdiff_t)pitch);
        for (int z = L4; z < L; ++z)
            for (int y = y0; y < y0 + 4; ++y)
                ((double _Complex *)d)[(size_t)z*pitch + y] =
                    ((const double _Complex *)s)[(size_t)y*L + z];
    }
    for (int y = L4; y < L; ++y)
        for (int z = 0; z < L; ++z)
            ((double _Complex *)d)[(size_t)z*pitch + y] =
                ((const double _Complex *)s)[(size_t)y*L + z];
#else
    for (int y0 = 0; y0 < L; y0 += 16)
        for (int z0 = 0; z0 < L; z0 += 16) {
            int y1 = y0 + 16 < L ? y0 + 16 : L, z1 = z0 + 16 < L ? z0 + 16 : L;
            for (int y = y0; y < y1; ++y)
                for (int z = z0; z < z1; ++z)
                    ((double _Complex *)d)[(size_t)z*pitch + y] =
                        ((const double _Complex *)s)[(size_t)y*L + z];
        }
#endif
}

static void pln_transpose_out(const double *restrict s, double *restrict d,
                              int L, int pitch)
{
#if PLN_SIMD
    const int L4 = L & ~3;
    for (int z0 = 0; z0 < L4; z0 += 4) {
        for (int y0 = 0; y0 < L4; y0 += 4)
            pln_tr4x4(s + 2*((size_t)z0*pitch + y0), 2*(ptrdiff_t)pitch,
                      d + 2*((size_t)y0*L + z0), 2*(ptrdiff_t)L);
        for (int y = L4; y < L; ++y)
            for (int z = z0; z < z0 + 4; ++z)
                ((double _Complex *)d)[(size_t)y*L + z] =
                    ((const double _Complex *)s)[(size_t)z*pitch + y];
    }
    for (int z = L4; z < L; ++z)
        for (int y = 0; y < L; ++y)
            ((double _Complex *)d)[(size_t)y*L + z] =
                ((const double _Complex *)s)[(size_t)z*pitch + y];
#else
    for (int z0 = 0; z0 < L; z0 += 16)
        for (int y0 = 0; y0 < L; y0 += 16) {
            int y1 = y0 + 16 < L ? y0 + 16 : L, z1 = z0 + 16 < L ? z0 + 16 : L;
            for (int z = z0; z < z1; ++z)
                for (int y = y0; y < y1; ++y)
                    ((double _Complex *)d)[(size_t)y*L + z] =
                        ((const double _Complex *)s)[(size_t)z*pitch + y];
        }
#endif
}

/* axes 1+2 of one x-plane, in place, plane cache-resident */
static void pln_p3d_plane(pln_p3d *p, double *pl)
{
    const int L = p->L, T = p->xp->tile, pitch = p->pitch;
    pln_x *root = p->xp->root;
    double *arena = p->xp->arena;
    const ptrdiff_t bs = 2 * (ptrdiff_t)T;
    for (int off = 0; off < L; off += T) {                   /* axis 1, in place */
        int w = L - off < T ? L - off : T;
        pln_xexec(root, pl + 2*off, 2*L, pl + 2*off, 2*L, w, arena, bs);
    }
    pln_transpose_in(pl, p->P, L, pitch);
    for (int off = 0; off < L; off += T) {                   /* axis 2, in place */
        int w = L - off < T ? L - off : T;
        pln_xexec(root, p->P + 2*off, 2*(ptrdiff_t)pitch, p->P + 2*off,
                  2*(ptrdiff_t)pitch, w, arena, bs);
    }
    pln_transpose_out(p->P, pl, L, pitch);
}

/* one L^3 volume, out-of-place, in unchanged */
static __attribute__((unused)) void pln_p3d_exec(pln_p3d *p,
        const double _Complex *in, double _Complex *out)
{
    const int L = p->L, T = p->xp->tile;
    const size_t L2 = (size_t)L * L;
    const double *src = (const double *)in;
    double *dst = (double *)out;
    pln_x *root = p->xp->root;
    double *arena = p->xp->arena;
    const ptrdiff_t bs = 2 * (ptrdiff_t)T;

    for (size_t off = 0; off < L2; off += T) {               /* axis 0: in -> out */
        int w = (int)(L2 - off < (size_t)T ? L2 - off : (size_t)T);
        pln_xexec(root, src + 2*off, 2*(ptrdiff_t)L2, dst + 2*off, 2*(ptrdiff_t)L2, w, arena, bs);
    }
    for (int xx = 0; xx < L; ++xx)                           /* axes 1+2 per plane */
        pln_p3d_plane(p, dst + 2 * (size_t)xx * L2);
}

/* ------------------------------------------------------------------ */
/* the graded map  z <- z / (1 + |z|)  with  z = FFT(state) + c,        */
/* applied to one contiguous span while it is still cache-hot.          */
/* rsqrt14/rcp14 + 2 Newton steps each (~1e-16 rel, campaign standard); */
/* zmm vsqrtpd/vdivpd are unpipelined and ~4x slower here.              */
/* ------------------------------------------------------------------ */
static void pln_map_span(double *restrict st, const double *restrict cf, size_t npts)
{
    size_t i = 0, n = 2 * npts;
#if defined(__AVX512F__)
    const __m512d half = _mm512_set1_pd(0.5), th = _mm512_set1_pd(1.5);
    const __m512d one = _mm512_set1_pd(1.0), two = _mm512_set1_pd(2.0);
    const __m512d tiny = _mm512_set1_pd(1e-300);
    for (; i + 8 <= n; i += 8) {
        __m512d v = _mm512_add_pd(_mm512_loadu_pd(st + i), _mm512_loadu_pd(cf + i));
        __m512d t = _mm512_mul_pd(v, v);
        __m512d h = _mm512_add_pd(t, _mm512_permute_pd(t, 0x55)); /* re^2+im^2, per pair */
        h = _mm512_max_pd(h, tiny);
        __m512d r = _mm512_rsqrt14_pd(h);                         /* ~2^-14 */
        r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(_mm512_mul_pd(half, h), r), r, th));
        r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(_mm512_mul_pd(half, h), r), r, th));
        __m512d d = _mm512_fmadd_pd(h, r, one);                   /* 1 + |z| */
        __m512d s = _mm512_rcp14_pd(d);
        s = _mm512_mul_pd(s, _mm512_fnmadd_pd(d, s, two));
        s = _mm512_mul_pd(s, _mm512_fnmadd_pd(d, s, two));
        _mm512_storeu_pd(st + i, _mm512_mul_pd(v, s));
    }
#endif
    for (; i < n; i += 2) {                                       /* exact tail */
        double re = st[i] + cf[i], im = st[i+1] + cf[i+1];
        double sc = 1.0 / (1.0 + sqrt(re*re + im*im));
        st[i] = re * sc; st[i+1] = im * sc;
    }
}

/* one full chained step IN PLACE on one volume:
 *     state <- (FFT3(state) + c) / (1 + |FFT3(state) + c|)
 * The map runs per x-plane right after that plane's axis-2 pass, while the
 * plane is still in cache -- the whole step touches state and c exactly once. */
static __attribute__((unused)) void pln_p3d_step(pln_p3d *p,
        double *state, const double *cfield)
{
    const int L = p->L, T = p->xp->tile;
    const size_t L2 = (size_t)L * L;
    pln_x *root = p->xp->root;
    double *arena = p->xp->arena;
    const ptrdiff_t bs = 2 * (ptrdiff_t)T;

    for (size_t off = 0; off < L2; off += T) {               /* axis 0 in place */
        int w = (int)(L2 - off < (size_t)T ? L2 - off : (size_t)T);
        pln_xexec(root, state + 2*off, 2*(ptrdiff_t)L2, state + 2*off, 2*(ptrdiff_t)L2, w, arena, bs);
    }
    for (int xx = 0; xx < L; ++xx) {
        double *pl = state + 2 * (size_t)xx * L2;
        pln_p3d_plane(p, pl);
        pln_map_span(pl, cfield + 2 * (size_t)xx * L2, L2);  /* plane still hot */
    }
}

/* ================================================================== */
/* standalone fft3d entry (omitted when adopted as a library)          */
/* ================================================================== */
#ifndef GEN_PLANNER_LIB

#include <time.h>
#include "../fft3d_api.h"

/* gen_race adoption (their gen_r2 string-wisdom hook, written for this entry):
 * persist the raced tree name per (host, L) so a second process builds the
 * same plan from a file read -- repeatability across the driver's two
 * processes becomes structural (different trees round differently), and warm
 * create() skips the race.  GEN_RACE_NO_WISDOM/REFRESH pins pass through. */
#define GEN_RACE_LIB_ONLY
#include "gen_race.c"

struct fft3d_plan {
    int L, batch;
    pln_p3d *p3;
    int chain_ok;                /* create()-time gate passed: own the chain */
    double _Complex *scratch;    /* one volume, only for the !chain_ok fallback */
    char picked[96];
};

const char *fft3d_name(void) { return "gen_planner"; }
const char *fft3d_description(void)
{
    return "planner layer: L -> {ct,gt-pfa,rader,bluestein,dense} candidate trees + generic "
           "strided-row executor (in-place, fused twiddles) + volume-resident fused chain, "
           "any 2<=L<=128; adopt via GEN_PLANNER_LIB include";
}
int fft3d_supports(int L) { return L >= 2 && L <= 128; }

static double pln_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* create()-time race over the top candidates: the gen_race composition demo.
 * Races the CHAIN STEP (what is graded), not raw execute -- gen_pfa_large's
 * gen_r1 lesson: the two can order candidates differently.
 * Off by default so independent processes pick identical plans (bit-identical
 * output across runs); enable with GEN_PLANNER_RACE=1. */
static pln_p3d *pln_race(int L, pln_cand *cand, int k, char *picked, size_t pcap)
{
    if (k > 4) k = 4;
    size_t vol = (size_t)L * L * L;
    double *a, *st, *cf;
    if (posix_memalign((void **)&a, 64, 2 * vol * sizeof *a)) return NULL;
    if (posix_memalign((void **)&st, 64, 2 * vol * sizeof *st)) { free(a); return NULL; }
    if (posix_memalign((void **)&cf, 64, 2 * vol * sizeof *cf)) { free(a); free(st); return NULL; }
    unsigned long s = 12345;
    for (size_t i = 0; i < 2 * vol; ++i) {
        s = s * 6364136223846793005UL + 1442695040888963407UL;
        a[i] = (double)(s >> 12) / (double)(1UL << 52) - 0.5;
        s = s * 6364136223846793005UL + 1442695040888963407UL;
        cf[i] = 0.1 * ((double)(s >> 12) / (double)(1UL << 52) - 0.5);
    }
    pln_p3d *bestp = NULL;
    double bestt = 1e300;
    for (int i = 0; i < k; ++i) {
        pln_p3d *p = pln_p3d_build(L, cand[i].t);
        if (!p) continue;
        memcpy(st, a, 2 * vol * sizeof *st);
        pln_p3d_step(p, st, cf);                          /* warm */
        double t = 1e300;
        for (int rep = 0; rep < 3; ++rep) {
            double t0 = pln_now();
            pln_p3d_step(p, st, cf);
            double dt = pln_now() - t0;
            if (dt < t) t = dt;
        }
        /* 2% simplest-first hysteresis (gen_pfa_large's tuner discipline):
           model order breaks ties, so quiet hosts pick stably */
        if (t < bestt * 0.98) {
            if (bestp) pln_p3d_free(bestp);
            bestp = p; bestt = t;
            snprintf(picked, pcap, "%s", cand[i].name);
        } else {
            pln_p3d_free(p);
        }
    }
    free(a); free(st); free(cf);
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

    /* Plan-time race ON BY DEFAULT from gen_r2 (deliverable #3 in the brief):
     * the model mis-orders candidates the race trivially separates (c4(d8)
     * 236 vs c8(d4) 282 us at L=32; c4(c5(d5)) -6% at L=100 on the node).
     * GEN_PLANNER_RACE=0 restores the deterministic model pick (and skips
     * the wisdom cache entirely, keeping that path bit-repeatable offline). */
    const char *race = getenv("GEN_PLANNER_RACE");
    int do_race = !(race && race[0] == '0');
    char wkey[64];
    int raced = 0;
    snprintf(wkey, sizeof wkey, "gen_planner/tree/L%d", L);
    if (do_race) {
        char wname[GR_NAME_MAX];
        if (gr_wisdom_get_str(wkey, wname, sizeof wname)) {
            for (int i = 0; i < k; ++i)
                if (!strcmp(cand[i].name, wname)) {   /* stale names re-race */
                    p->p3 = pln_p3d_build(L, cand[i].t);
                    snprintf(p->picked, sizeof p->picked, "%s", cand[i].name);
                    break;
                }
        }
        if (!p->p3) {
            p->p3 = pln_race(L, cand, k, p->picked, sizeof p->picked);
            raced = (p->p3 != NULL);
        }
    }
    if (!p->p3) {
        p->p3 = pln_p3d_build(L, cand[0].t);
        snprintf(p->picked, sizeof p->picked, "%s", cand[0].name);
    }
    free(A);                     /* exec plan owns copies of everything it needs */
    if (!p->p3) { free(p); return NULL; }
    {
        const char *v = getenv("GEN_PLANNER_VERBOSE");
        if (v && v[0] == '1')
            fprintf(stderr, "gen_planner: L=%d picked %s (of %d candidates)\n",
                    L, p->picked, k);
    }

    /* gate the owned chain step against execute + the exact scalar map on a
     * random volume, two steps (the ice L17_rader r5 discipline: a fast wrong
     * chain must be structurally impossible to ship) */
    {
        size_t vol = (size_t)L * L * L;
        double *a = NULL, *st = NULL, *rf = NULL, *cf = NULL;
        int ok = !posix_memalign((void **)&a, 64, 2 * vol * sizeof(double)) &&
                 !posix_memalign((void **)&st, 64, 2 * vol * sizeof(double)) &&
                 !posix_memalign((void **)&rf, 64, 2 * vol * sizeof(double)) &&
                 !posix_memalign((void **)&cf, 64, 2 * vol * sizeof(double));
        if (ok) {
            unsigned long s = 987654321;
            for (size_t i = 0; i < 2 * vol; ++i) {
                s = s * 6364136223846793005UL + 1442695040888963407UL;
                a[i] = (double)(s >> 12) / (double)(1UL << 52) - 0.5;
                s = s * 6364136223846793005UL + 1442695040888963407UL;
                cf[i] = 0.1 * ((double)(s >> 12) / (double)(1UL << 52) - 0.5);
            }
            memcpy(st, a, 2 * vol * sizeof(double));
            memcpy(rf, a, 2 * vol * sizeof(double));
            double num = 0, den = 0;
            for (int step = 0; step < 2; ++step) {
                pln_p3d_step(p->p3, st, cf);
                double *z = a;   /* reuse a as the reference FFT output */
                pln_p3d_exec(p->p3, (const double _Complex *)rf, (double _Complex *)z);
                for (size_t i = 0; i < 2 * vol; i += 2) {
                    double re = z[i] + cf[i], im = z[i+1] + cf[i+1];
                    double sc = 1.0 / (1.0 + sqrt(re*re + im*im));
                    rf[i] = re * sc; rf[i+1] = im * sc;
                }
            }
            for (size_t i = 0; i < 2 * vol; ++i) {
                double d = st[i] - rf[i];
                num += d * d; den += rf[i] * rf[i];
            }
            p->chain_ok = (den > 0 && sqrt(num / den) < 1e-12);
        }
        free(a); free(st); free(rf); free(cf);
        if (!p->chain_ok &&
            posix_memalign((void **)&p->scratch, 64, vol * sizeof *p->scratch))
            p->scratch = NULL;
    }
    /* persist the raced pick only after its chain gate passed (gen_powp's
     * wisdom discipline); wisdom hits above skip both race and store */
    if (raced && p->chain_ok)
        gr_wisdom_put_str(wkey, p->picked);
    return p;
}

/* Owned graded chain: state <- (FFT3(state) + c) / (1 + |FFT3(state) + c|),
 * m steps.  VOLUME-MAJOR (volumes are independent, c is pointwise): each
 * volume runs its whole m-step chain in place in final_out while it is
 * cache-resident -- the gen_rader / gen_dense_prime chain scheme, generalized.
 * Falls back to execute + exact scalar map if the create() gate failed. */
void fft3d_chain(fft3d_plan *p, const double _Complex *x0,
                 const double _Complex *c, double _Complex *final_out, int m)
{
    const size_t vol = (size_t)p->L * p->L * p->L;
    for (int b = 0; b < p->batch; ++b) {
        double *st = (double *)(final_out + (size_t)b * vol);
        const double *cf = (const double *)(c + (size_t)b * vol);
        memcpy(st, x0 + (size_t)b * vol, vol * sizeof *x0);
        if (p->chain_ok || !p->scratch) {
            for (int s = 0; s < m; ++s)
                pln_p3d_step(p->p3, st, cf);
        } else {
            double *z = (double *)p->scratch;
            for (int s = 0; s < m; ++s) {
                pln_p3d_exec(p->p3, (const double _Complex *)st, p->scratch);
                for (size_t i = 0; i < 2 * vol; i += 2) {
                    double re = z[i] + cf[i], im = z[i+1] + cf[i+1];
                    double sc = 1.0 / (1.0 + sqrt(re*re + im*im));
                    st[i] = re * sc; st[i+1] = im * sc;
                }
            }
        }
    }
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
    free(p->scratch);
    free(p);
}

#endif /* GEN_PLANNER_LIB */
#endif /* GEN_PLANNER_C_INCLUDED */
