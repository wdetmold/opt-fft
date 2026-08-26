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
 * gen_r4: CT nodes with n = r*m <= 20 and both stages hard leaves run as
 * FUSED register-resident codelets (no arena round trip); enumeration also
 * emits each CT root with the runner-up child tree (sub-tree diversity for
 * the race); transpose edges are masked 4x4 blocks, no scalar loops.
 *
 * gen_r5: SPLIT-GROUP batch engine (the pw_* / pln_s8_* layer): at
 * batch >= 8, groups of 8 volumes are packed site-major split-complex
 * (site = {8 re, 8 im} = 2 zmm, lane = volume) and the whole chain step
 * runs shuffle-free, mask-free and TRANSPOSE-free with broadcast-scalar
 * twiddles; the graded map is fused into the final-axis stores above
 * L = 12 and its ladder sees 8 distinct sites per zmm.  Three levels --
 * fused root (n <= 25), two-pass CT in two fused volume sweeps (z-slab
 * axes 0+1, x-plane axis 2), folded odd dense -- enter the create() race
 * as extra arms and are picked per (host, L, batch-regime) through the
 * wisdom cache (value tag @s1/@s2/@s3).  The race itself is interleaved
 * sample-major since gen_r5 (gen_race r4's protocol fix).
 *
 * gen_r6: (1) the per-volume chain fuses the graded map into the axis-2
 * transpose-out exit (gen_layout r5's gl_map8 shape: pair-compressed
 * |z|^2, rsqrt14+2NR for the magnitude, ONE exact vdivpd per 8 complex --
 * the exit is shuffle-bound with an idle divider), deleting the separate
 * per-plane map pass at 12 < L <= 80; (2) the separate-pass ladder that remains
 * (small L, non-fused paths) is PAIR-PACKED (gen_pfa_large r5's
 * map_step_pair, bit-identical per element): two vectors' 8 distinct
 * |z|^2 share one NR ladder; (3) a fourth split-group level @s4 -- the
 * two-pass CT run as ONE volume sweep per axis, fully in place, through
 * an L1-resident staging block of P pencils (child pass volume->stage,
 * twiddled root leaf stage->volume, map fused in the final-axis stores).
 * @s4 deletes @s2's ping-pong buffer and halves its per-axis volume
 * sweeps; it races alongside the other levels at batch >= 8.
 *
 * gen_r7: (1) FUSED GOOD-THOMAS codelets -- a PFA node whose two factors
 * are both hard leaves and whose product is <= PLN_FUSEMAX runs as a
 * register-resident twiddle-FREE codelet (pv and pw forms): the CRT
 * permutations become compile-time index selection, so the gather/scatter
 * passes that made generic PFA lose (gen_r1: gt(d5,d8) 574 vs c8(d5) 468)
 * do not exist, and the (r-1)(m-1) twiddle cmuls of the equivalent fused
 * CT are simply deleted (-20..25% of pencil arithmetic at 10/12/15/20).
 * This is gen_pfa_small / gen_batchlane's PFA economics hosted in the
 * generic engine; fused-GT nodes also serve as CT children (50, 100) and
 * as @s1 roots / @s2/@s4 children in the split-group engine.  (2) DFT7 is
 * a HARD LEAF (symmetric-fold codelet, pv/pw/scalar): the r6 surprise
 * test showed 7-containing sizes (21, 44) running d7 as an 8n^2 dense
 * matrix; 7 in the hard set gives every 7-multiple the leaf/fused/GT
 * machinery (14, 21 register-resident; 28/35/42/56/63... leaf-7 CT).
 *
 * gen_r9 (the counter-directed round; see results/PMU_AUDIT.md): (1) the
 * create() race is NOISE-GATED -- when the hysteresis winner and a
 * contender sit closer than the pair's own measured trial noise, the race
 * extends (contenders only, up to 6 extra interleaved rounds) until the
 * gap clears the noise or the deterministic 2%-simplest-first hysteresis
 * breaks the tie; a stored wisdom verdict is therefore never a min-of-3
 * coin flip (the audit's L=25 plan-pick instability, banked against).
 * (2) c-line CUSTODY (adopted from gen_pfa_large's c-bypass): clflushopt
 * each x-plane's consumed c lines so the read-once c stream stops evicting
 * m-step-resident state.  gen_r10 NODE VERDICT: custody-on LOSES at L=100
 * (+6..11%, 3/3 pairs on a81n2) -- default OFF everywhere, the race's
 * custody playoff still measures per host and banks @f0/@f1 via wisdom;
 * GEN_PLANNER_CFLUSH=0/1 pins for A/B.
 *
 * gen_r10 (counter-directed, part 2; three measured refutations + node
 * validation of the r9 machinery): (1) gen_pow2 r9's extract-to-memory
 * transpose stores: bit-identical, -4 p5 uops per 4x4 block, and +2..6%
 * SLOWER at 32/40/100 on the node (16B-aligned destinations line-cross
 * where the zmm store was one access) -- PLN_TRZST=0 default, knob kept.
 * (2) custody: refuted, above.  (3) the r6 fused-exit loss at L > 80 was
 * mostly the WALK ORDER (y0-outer exit pln_transpose_out_map_yo repairs
 * 5722-6018 -> 5146-5589 at 100) but the separate pair-packed map still
 * wins 3/3 by 0.2-2.2% -- default unchanged, PVFUSE=2 keeps the yo arm.
 * (4) avenue-1 acceptance PROVEN on the node: 5 consecutive cold create()
 * cycles pick identically at 25/50/100, outputs bit-identical.
 */
#ifndef GEN_PLANNER_C_INCLUDED
#define GEN_PLANNER_C_INCLUDED

#include <complex.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(__AVX512F__) || defined(__CLFLUSHOPT__)
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

/* ---- graded-map helpers on interleaved vectors (gen_r6) ----
 * pln_mag_pair packs the 8 DISTINCT |z|^2 of two interleaved vectors into
 * one zmm (gen_pfa_large r5's map_step_pair packing: the old per-vector
 * form ran the whole NR ladder on pair-DUPLICATED magnitudes).  Lane 2j
 * of the old h was t[2j]+t[2j+1] = re^2+im^2; the packed lane is the same
 * two operands in the same add, so ladders fed from it are bit-identical
 * per element.  pln_sdup unpacks a packed scale back to pair-duplicated. */
static inline __attribute__((always_inline)) pv pln_mag_pair(pv za, pv zb)
{
    const __m512i IE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i IO = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    pv ta = _mm512_mul_pd(za, za), tb = _mm512_mul_pd(zb, zb);
    return _mm512_add_pd(_mm512_permutex2var_pd(ta, IE, tb),
                         _mm512_permutex2var_pd(ta, IO, tb));
}
static inline __attribute__((always_inline)) void pln_sdup(pv s, pv *sa, pv *sb)
{
    const __m512i DLO = _mm512_setr_epi64(0, 0, 1, 1, 2, 2, 3, 3);
    const __m512i DHI = _mm512_setr_epi64(4, 4, 5, 5, 6, 6, 7, 7);
    *sa = _mm512_permutexvar_pd(DLO, s);
    *sb = _mm512_permutexvar_pd(DHI, s);
}
/* s = 1/(1 + sqrt(h)) for 8 packed |z|^2: rsqrt14 + 2 Newton for the
 * magnitude (as always), then ONE exact vdivpd -- used only at exits that
 * are shuffle/store-bound with an idle divider (gen_layout r5's gl_map8;
 * gen_batchlane r5's "div-vs-rcp is a property of the surrounding
 * codelet").  The separate-pass ladder keeps the rcp14 form. */
static inline __attribute__((always_inline)) pv pln_mapdiv8(pv h)
{
    const pv half = _mm512_set1_pd(0.5), th = _mm512_set1_pd(1.5);
    const pv one = _mm512_set1_pd(1.0), tiny = _mm512_set1_pd(1e-300);
    h = _mm512_max_pd(h, tiny);
    pv r = _mm512_rsqrt14_pd(h);
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(_mm512_mul_pd(half, h), r), r, th));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(_mm512_mul_pd(half, h), r), r, th));
    return _mm512_div_pd(one, _mm512_fmadd_pd(h, r, one));
}
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
    const pln_node *best2[PLN_NMAX + 1];  /* runner-up sub-tree (gen_r4): CT roots
                                             also enumerate their second-best child
                                             so the race sees sub-tree diversity */
    double bestc2[PLN_NMAX + 1];
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
static int pln_leaf_hard(int r)
{ return r == 2 || r == 3 || r == 4 || r == 5 || r == 7 || r == 8; }

static double pln_leafF(int r)   /* model flops per pencil for one leaf DFT_r */
{
    switch (r) {
    case 2: return 6;  case 3: return 18; case 4: return 18;
    case 5: return 40; case 7: return 100; case 8: return 52;
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

/* a CT node is FUSED (gen_r4) when both stages are hard leaves and n <= 20:
 * the whole n-point DFT runs in registers per column chunk -- no arena
 * round trip, no per-leaf call.  Detection mirrors pln_xbuild_rec. */
#ifndef PLN_FUSEMAX
#define PLN_FUSEMAX 25          /* n = 25 spills a few zmm but still beats the
                                   arena round trip (A/B'd on the node; set 20
                                   to confine fusion to the spill-free sizes) */
#endif
static int pln_ct_fused(int r, const pln_node *s1)
{
    return pln_leaf_hard(r) && s1 && s1->kind == PLN_LEAF &&
           pln_leaf_hard(s1->n) && r * s1->n <= PLN_FUSEMAX;
}

/* a PFA node is FUSED (gen_r7) when both coprime factors are hard leaves
 * and the product is <= PLN_FUSEMAX: the whole n-point DFT runs as one
 * register-resident TWIDDLE-FREE codelet -- the CRT permutations are
 * compile-time index selection, so the arena gather/scatter passes that
 * made generic PFA lose (gen_r1's 30n penalty) do not exist. */
static int pln_gt_fused(const pln_node *s1, const pln_node *s2)
{
    return s1 && s2 && s1->kind == PLN_LEAF && s2->kind == PLN_LEAF &&
           pln_leaf_hard(s1->n) && pln_leaf_hard(s2->n) &&
           s1->n * s2->n <= PLN_FUSEMAX;
}

static double pln_ct_cost(int r, int m, const pln_node *s1, double childc)
{
    if (pln_ct_fused(r, s1))                  /* register-resident: overheads gone */
        return m * pln_leafF(r) + r * pln_leafF(m) + 4.0 * (m - 1) * (r - 1) + 12.0;
    return r * childc + m * pln_leafF(r) + 6.0 * m * (r - 1) + 4.0 * (r * m) + 30.0 * m;
}

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
        const pln_node *s1 = pln_best(A, m);
        double cb = A->bestc[m];
        PLN_ADD(pln_new(A, PLN_CT, n, r, 0, s1, NULL), pln_ct_cost(r, m, s1, cb));
        /* sub-tree diversity (gen_r4, deferred since r1): also race the
           runner-up child decomposition -- the model keeps mis-ordering
           equal-flop children and the race is the arbiter */
        if (A->best2[m] && A->best2[m] != s1)
            PLN_ADD(pln_new(A, PLN_CT, n, r, 0, A->best2[m], NULL),
                    pln_ct_cost(r, m, A->best2[m], A->bestc2[m]));
    }
    for (int a = 2; a * a <= n; ++a) {                /* Good-Thomas: coprime split */
        if (n % a) continue;
        int b = n / a;
        if (b < 2 || pln_gcd(a, b) != 1) continue;
        const pln_node *sa = pln_best(A, a), *sb = pln_best(A, b);
        /* fused GT (gen_r7): register-resident and twiddle-free -- the
         * fused-CT cost minus its 4(m-1)(r-1) twiddle term */
        double c = pln_gt_fused(sa, sb)
                     ? b * pln_leafF(a) + a * pln_leafF(b) + 12.0
                     : b * pln_bestcost(A, a) + a * pln_bestcost(A, b) + 30.0 * n;
        PLN_ADD(pln_new(A, PLN_PFA, n, a, 0, sa, sb), c);
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
    if (k > 1) { A->best2[n] = tmp[1].t; A->bestc2[n] = tmp[1].cost; }
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

/* DFT7 via the symmetric fold (gen_r7): t_j = x_j + x_{7-j}, u_j = x_j -
 * x_{7-j}; X_k = E_k -+ i O_k with E_k = x0 + sum_j cos(2pi jk/7) t_j and
 * O_k = sum_j sin(2pi jk/7) u_j -- the leaf-5 pattern one prime up. */
#define PLN_D7C1  0.62348980185873353053
#define PLN_D7C2 -0.22252093395631440429
#define PLN_D7C3 -0.90096886790241912624
#define PLN_D7S1  0.78183148246802980871
#define PLN_D7S2  0.97492791218182360702
#define PLN_D7S3  0.43388373911755812048

/* LIFTED DFT5 v-pair (gen_r8, ADOPTED from gen_batchlane gen_r7):
 * sin(2pi/5)/sin(pi/5) = 2cos(pi/5) = PHI exactly, so the sine pair
 *   v1 = S1*sa + S2*sb,  v2 = S2*sa - S1*sb
 * factors through u = sa - PHI*sb as v2 = S2*u, v1 = S1*u + KL5*sb with
 * KL5 = S2 + S1*PHI = 1.25/sin(pi/5) -- one fewer live temp per DFT5.
 * On THIS engine the verdict is PAIR-SPECIFIC (gen_r8 held-lease pairs):
 * gt(2,5) -1.0% 4/4 at L=10, gt(3,5) +2% 4/4 LOSS at L=15, (4,5) and
 * c5(d5) a wash -- so only the (2,5) GT codelets take the lift (the L5
 * flag through pv_dftNr_i / pw_dftNs_i) and every other DFT5 site stays
 * bit-identical to gen_r7.  Constants exact to the last bit of double
 * (gen_batchlane's 50-digit Decimal values).  PLN_LIFT5=0 kills the lift
 * everywhere for cross-arch races. */
#ifndef PLN_LIFT5
#define PLN_LIFT5 1
#endif
#define PLN_PHI5  1.61803398874989484820
#define PLN_KL5   2.12662702088009983045

static void pln_leaf7(const double *restrict in, ptrdiff_t is,
                      double *restrict out, ptrdiff_t os, int w)
{
    const double C1 = PLN_D7C1, C2 = PLN_D7C2, C3 = PLN_D7C3;
    const double S1 = PLN_D7S1, S2 = PLN_D7S2, S3 = PLN_D7S3;
    for (int c = 0; c < w; ++c) {
        double x0r = in[2*c],      x0i = in[2*c+1];
        double x1r = in[is+2*c],   x1i = in[is+2*c+1];
        double x2r = in[2*is+2*c], x2i = in[2*is+2*c+1];
        double x3r = in[3*is+2*c], x3i = in[3*is+2*c+1];
        double x4r = in[4*is+2*c], x4i = in[4*is+2*c+1];
        double x5r = in[5*is+2*c], x5i = in[5*is+2*c+1];
        double x6r = in[6*is+2*c], x6i = in[6*is+2*c+1];
        double t1r = x1r + x6r, t1i = x1i + x6i, u1r = x1r - x6r, u1i = x1i - x6i;
        double t2r = x2r + x5r, t2i = x2i + x5i, u2r = x2r - x5r, u2i = x2i - x5i;
        double t3r = x3r + x4r, t3i = x3i + x4i, u3r = x3r - x4r, u3i = x3i - x4i;
        out[2*c]   = x0r + t1r + t2r + t3r;
        out[2*c+1] = x0i + t1i + t2i + t3i;
        double e1r = x0r + C1*t1r + C2*t2r + C3*t3r;
        double e1i = x0i + C1*t1i + C2*t2i + C3*t3i;
        double o1r = S1*u1r + S2*u2r + S3*u3r;
        double o1i = S1*u1i + S2*u2i + S3*u3i;
        out[os+2*c]     = e1r + o1i;  out[os+2*c+1]     = e1i - o1r;
        out[6*os+2*c]   = e1r - o1i;  out[6*os+2*c+1]   = e1i + o1r;
        double e2r = x0r + C2*t1r + C3*t2r + C1*t3r;
        double e2i = x0i + C2*t1i + C3*t2i + C1*t3i;
        double o2r = S2*u1r - S3*u2r - S1*u3r;
        double o2i = S2*u1i - S3*u2i - S1*u3i;
        out[2*os+2*c]   = e2r + o2i;  out[2*os+2*c+1]   = e2i - o2r;
        out[5*os+2*c]   = e2r - o2i;  out[5*os+2*c+1]   = e2i + o2r;
        double e3r = x0r + C3*t1r + C1*t2r + C2*t3r;
        double e3i = x0i + C3*t1i + C1*t2i + C2*t3i;
        double o3r = S3*u1r - S1*u2r + S2*u3r;
        double o3i = S3*u1i - S1*u2i + S2*u3i;
        out[3*os+2*c]   = e3r + o3i;  out[3*os+2*c+1]   = e3i - o3r;
        out[4*os+2*c]   = e3r - o3i;  out[4*os+2*c+1]   = e3i + o3r;
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

static void pln_leaf7_tw(const double *restrict in, ptrdiff_t is,
                         double *restrict out, ptrdiff_t os, int w,
                         const double *restrict tw)
{
    const double C1 = PLN_D7C1, C2 = PLN_D7C2, C3 = PLN_D7C3;
    const double S1 = PLN_D7S1, S2 = PLN_D7S2, S3 = PLN_D7S3;
    for (int c = 0; c < w; ++c) {
        double x0r = in[2*c], x0i = in[2*c+1];
        PLN_LDTW(x1r, x1i, in + is, 0);
        PLN_LDTW(x2r, x2i, in + 2*is, 1);
        PLN_LDTW(x3r, x3i, in + 3*is, 2);
        PLN_LDTW(x4r, x4i, in + 4*is, 3);
        PLN_LDTW(x5r, x5i, in + 5*is, 4);
        PLN_LDTW(x6r, x6i, in + 6*is, 5);
        double t1r = x1r + x6r, t1i = x1i + x6i, u1r = x1r - x6r, u1i = x1i - x6i;
        double t2r = x2r + x5r, t2i = x2i + x5i, u2r = x2r - x5r, u2i = x2i - x5i;
        double t3r = x3r + x4r, t3i = x3i + x4i, u3r = x3r - x4r, u3i = x3i - x4i;
        out[2*c]   = x0r + t1r + t2r + t3r;
        out[2*c+1] = x0i + t1i + t2i + t3i;
        double e1r = x0r + C1*t1r + C2*t2r + C3*t3r;
        double e1i = x0i + C1*t1i + C2*t2i + C3*t3i;
        double o1r = S1*u1r + S2*u2r + S3*u3r;
        double o1i = S1*u1i + S2*u2i + S3*u3i;
        out[os+2*c]     = e1r + o1i;  out[os+2*c+1]     = e1i - o1r;
        out[6*os+2*c]   = e1r - o1i;  out[6*os+2*c+1]   = e1i + o1r;
        double e2r = x0r + C2*t1r + C3*t2r + C1*t3r;
        double e2i = x0i + C2*t1i + C3*t2i + C1*t3i;
        double o2r = S2*u1r - S3*u2r - S1*u3r;
        double o2i = S2*u1i - S3*u2i - S1*u3i;
        out[2*os+2*c]   = e2r + o2i;  out[2*os+2*c+1]   = e2i - o2r;
        out[5*os+2*c]   = e2r - o2i;  out[5*os+2*c+1]   = e2i + o2r;
        double e3r = x0r + C3*t1r + C1*t2r + C2*t3r;
        double e3i = x0i + C3*t1i + C1*t2i + C2*t3i;
        double o3r = S3*u1r - S1*u2r + S2*u3r;
        double o3i = S3*u1i - S1*u2i + S2*u3i;
        out[3*os+2*c]   = e3r + o3i;  out[3*os+2*c+1]   = e3i - o3r;
        out[4*os+2*c]   = e3r - o3i;  out[4*os+2*c+1]   = e3i + o3r;
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

static inline __attribute__((always_inline)) void pln_lv7(
    const double *restrict in, ptrdiff_t is, double *restrict out, ptrdiff_t os,
    int w, const double *restrict tw, int HAS)
{
    const pv C1 = _mm512_set1_pd(PLN_D7C1), C2 = _mm512_set1_pd(PLN_D7C2);
    const pv C3 = _mm512_set1_pd(PLN_D7C3);
    const pv S1 = _mm512_set1_pd(PLN_D7S1), S2 = _mm512_set1_pd(PLN_D7S2);
    const pv S3 = _mm512_set1_pd(PLN_D7S3);
    pv wr[6], wi[6];
    if (HAS) for (int t = 0; t < 6; ++t) {
        wr[t] = _mm512_set1_pd(tw[2*t]); wi[t] = _mm512_set1_pd(tw[2*t+1]);
    }
    for (int c = 0; c < w; c += 4) {
        __mmask8 mk = PLN_MK(w, c);
        pv x0 = pv_ld(mk, in + 2*c);
        pv x1 = pv_ld(mk, in + is + 2*c);
        pv x2 = pv_ld(mk, in + 2*is + 2*c);
        pv x3 = pv_ld(mk, in + 3*is + 2*c);
        pv x4 = pv_ld(mk, in + 4*is + 2*c);
        pv x5 = pv_ld(mk, in + 5*is + 2*c);
        pv x6 = pv_ld(mk, in + 6*is + 2*c);
        if (HAS) {
            x1 = pv_cmul(x1, wr[0], wi[0]);
            x2 = pv_cmul(x2, wr[1], wi[1]);
            x3 = pv_cmul(x3, wr[2], wi[2]);
            x4 = pv_cmul(x4, wr[3], wi[3]);
            x5 = pv_cmul(x5, wr[4], wi[4]);
            x6 = pv_cmul(x6, wr[5], wi[5]);
        }
        pv t1 = _mm512_add_pd(x1, x6), u1 = _mm512_sub_pd(x1, x6);
        pv t2 = _mm512_add_pd(x2, x5), u2 = _mm512_sub_pd(x2, x5);
        pv t3 = _mm512_add_pd(x3, x4), u3 = _mm512_sub_pd(x3, x4);
        pv_st(out + 2*c, mk,
              _mm512_add_pd(x0, _mm512_add_pd(t1, _mm512_add_pd(t2, t3))));
        pv e1 = _mm512_fmadd_pd(C1, t1, _mm512_fmadd_pd(C2, t2, _mm512_fmadd_pd(C3, t3, x0)));
        pv o1 = _mm512_fmadd_pd(S1, u1, _mm512_fmadd_pd(S2, u2, _mm512_mul_pd(S3, u3)));
        pv_st(out + os + 2*c, mk, pv_subi(e1, o1));
        pv_st(out + 6*os + 2*c, mk, pv_addi(e1, o1));
        pv e2 = _mm512_fmadd_pd(C2, t1, _mm512_fmadd_pd(C3, t2, _mm512_fmadd_pd(C1, t3, x0)));
        pv o2 = _mm512_fnmadd_pd(S1, u3, _mm512_fnmadd_pd(S3, u2, _mm512_mul_pd(S2, u1)));
        pv_st(out + 2*os + 2*c, mk, pv_subi(e2, o2));
        pv_st(out + 5*os + 2*c, mk, pv_addi(e2, o2));
        pv e3 = _mm512_fmadd_pd(C3, t1, _mm512_fmadd_pd(C1, t2, _mm512_fmadd_pd(C2, t3, x0)));
        pv o3 = _mm512_fmadd_pd(S2, u3, _mm512_fnmadd_pd(S1, u2, _mm512_mul_pd(S3, u1)));
        pv_st(out + 3*os + 2*c, mk, pv_subi(e3, o3));
        pv_st(out + 4*os + 2*c, mk, pv_addi(e3, o3));
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

/* ------------------------------------------------------------------ */
/* FUSED two-level CT codelets (gen_r4): n = r*m <= 20, both stages     */
/* hard leaves.  The whole n-point DFT runs register-resident per       */
/* column chunk: load n rows once, child m-DFTs + twiddles + r-DFTs     */
/* in registers, store n rows once.  Deletes the CT arena round trip    */
/* (2n rows stored+reloaded per root call) and the per-leaf call        */
/* overhead that dominated L=10..20.  All loads precede all stores in   */
/* each chunk, so src == dst stays legal.  Arithmetic and constants     */
/* are byte-identical to the pln_lv* leaves.                            */
/* ------------------------------------------------------------------ */
static inline __attribute__((always_inline)) void pv_dft2r(pv *v)
{
    pv a = v[0];
    v[0] = _mm512_add_pd(a, v[1]);
    v[1] = _mm512_sub_pd(a, v[1]);
}

static inline __attribute__((always_inline)) void pv_dft3r(pv *v)
{
    const pv C = _mm512_set1_pd(-0.5), S = _mm512_set1_pd(0.86602540378443864676);
    pv t = _mm512_add_pd(v[1], v[2]), u = _mm512_sub_pd(v[1], v[2]);
    pv x0 = v[0];
    v[0] = _mm512_add_pd(x0, t);
    pv vv = _mm512_fmadd_pd(C, t, x0);
    pv su = _mm512_mul_pd(S, u);
    v[1] = pv_subi(vv, su);
    v[2] = pv_addi(vv, su);
}

static inline __attribute__((always_inline)) void pv_dft4r(pv *v)
{
    pv t0 = _mm512_add_pd(v[0], v[2]), t1 = _mm512_sub_pd(v[0], v[2]);
    pv t2 = _mm512_add_pd(v[1], v[3]), t3 = _mm512_sub_pd(v[1], v[3]);
    v[0] = _mm512_add_pd(t0, t2);
    v[1] = pv_subi(t1, t3);
    v[2] = _mm512_sub_pd(t0, t2);
    v[3] = pv_addi(t1, t3);
}

/* L5 (compile-time through every wrapper): use the LIFTED v-pair.  The
 * lift's win is PAIR-specific (gen_r8 held-lease pairs): gt(2,5) at L=10
 * -1.0% 4/4, but gt(3,5) at L=15 +2% 4/4 LOSS and (4,5)/c5(d5) a wash --
 * so only the (2,5) GT codelets request it and every other DFT5 stays
 * bit-identical to gen_r7. */
static inline __attribute__((always_inline)) void pv_dft5r_i(pv *v, const int L5)
{
    const pv C1 = _mm512_set1_pd(0.30901699437494742410);
    const pv C2 = _mm512_set1_pd(-0.80901699437494742410);
    const pv S1 = _mm512_set1_pd(0.95105651629515357212);
    const pv S2 = _mm512_set1_pd(0.58778525229247312917);
    const pv S2n = _mm512_set1_pd(-0.58778525229247312917);
    pv t1 = _mm512_add_pd(v[1], v[4]), t2 = _mm512_add_pd(v[2], v[3]);
    pv t3 = _mm512_sub_pd(v[1], v[4]), t4 = _mm512_sub_pd(v[2], v[3]);
    pv x0 = v[0];
    v[0] = _mm512_add_pd(x0, _mm512_add_pd(t1, t2));
    pv a = _mm512_fmadd_pd(C1, t1, _mm512_fmadd_pd(C2, t2, x0));
    pv u = _mm512_setzero_pd(), b;
    if (PLN_LIFT5 && L5) {
        u = _mm512_fnmadd_pd(_mm512_set1_pd(PLN_PHI5), t4, t3);
        b = _mm512_fnmadd_pd(S1, u, _mm512_mul_pd(_mm512_set1_pd(-PLN_KL5), t4));
    } else {
        b = _mm512_fnmadd_pd(S1, t3, _mm512_mul_pd(S2n, t4));
    }
    v[1] = pv_addi(a, b);
    v[4] = pv_subi(a, b);
    pv cc = _mm512_fmadd_pd(C2, t1, _mm512_fmadd_pd(C1, t2, x0));
    pv dd = (PLN_LIFT5 && L5) ? _mm512_mul_pd(S2n, u)
                              : _mm512_fnmadd_pd(S2, t3, _mm512_mul_pd(S1, t4));
    v[2] = pv_addi(cc, dd);
    v[3] = pv_subi(cc, dd);
}

static inline __attribute__((always_inline)) void pv_dft5r(pv *v)
{ pv_dft5r_i(v, 0); }

static inline __attribute__((always_inline)) void pv_dft7r(pv *v)
{
    const pv C1 = _mm512_set1_pd(PLN_D7C1), C2 = _mm512_set1_pd(PLN_D7C2);
    const pv C3 = _mm512_set1_pd(PLN_D7C3);
    const pv S1 = _mm512_set1_pd(PLN_D7S1), S2 = _mm512_set1_pd(PLN_D7S2);
    const pv S3 = _mm512_set1_pd(PLN_D7S3);
    pv t1 = _mm512_add_pd(v[1], v[6]), u1 = _mm512_sub_pd(v[1], v[6]);
    pv t2 = _mm512_add_pd(v[2], v[5]), u2 = _mm512_sub_pd(v[2], v[5]);
    pv t3 = _mm512_add_pd(v[3], v[4]), u3 = _mm512_sub_pd(v[3], v[4]);
    pv x0 = v[0];
    v[0] = _mm512_add_pd(x0, _mm512_add_pd(t1, _mm512_add_pd(t2, t3)));
    pv e1 = _mm512_fmadd_pd(C1, t1, _mm512_fmadd_pd(C2, t2, _mm512_fmadd_pd(C3, t3, x0)));
    pv o1 = _mm512_fmadd_pd(S1, u1, _mm512_fmadd_pd(S2, u2, _mm512_mul_pd(S3, u3)));
    v[1] = pv_subi(e1, o1);
    v[6] = pv_addi(e1, o1);
    pv e2 = _mm512_fmadd_pd(C2, t1, _mm512_fmadd_pd(C3, t2, _mm512_fmadd_pd(C1, t3, x0)));
    pv o2 = _mm512_fnmadd_pd(S1, u3, _mm512_fnmadd_pd(S3, u2, _mm512_mul_pd(S2, u1)));
    v[2] = pv_subi(e2, o2);
    v[5] = pv_addi(e2, o2);
    pv e3 = _mm512_fmadd_pd(C3, t1, _mm512_fmadd_pd(C1, t2, _mm512_fmadd_pd(C2, t3, x0)));
    pv o3 = _mm512_fmadd_pd(S2, u3, _mm512_fnmadd_pd(S1, u2, _mm512_mul_pd(S3, u1)));
    v[3] = pv_subi(e3, o3);
    v[4] = pv_addi(e3, o3);
}

static inline __attribute__((always_inline)) void pv_dft8r(pv *v)
{
    const double Sc = 0.70710678118654752440;
    const pv Sp = _mm512_set1_pd(Sc), Sn = _mm512_set1_pd(-Sc);
    pv a[4], b[4];
    for (int j = 0; j < 4; ++j) {
        a[j] = _mm512_add_pd(v[j], v[j+4]);
        b[j] = _mm512_sub_pd(v[j], v[j+4]);
    }
    b[1] = pv_cmul(b[1], Sp, Sn);
    b[2] = pv_mulmi(b[2]);
    b[3] = pv_cmul(b[3], Sn, Sn);
    pv t0 = _mm512_add_pd(a[0], a[2]), t1 = _mm512_sub_pd(a[0], a[2]);
    pv t2 = _mm512_add_pd(a[1], a[3]), t3 = _mm512_sub_pd(a[1], a[3]);
    v[0] = _mm512_add_pd(t0, t2);
    v[2] = pv_subi(t1, t3);
    v[4] = _mm512_sub_pd(t0, t2);
    v[6] = pv_addi(t1, t3);
    t0 = _mm512_add_pd(b[0], b[2]); t1 = _mm512_sub_pd(b[0], b[2]);
    t2 = _mm512_add_pd(b[1], b[3]); t3 = _mm512_sub_pd(b[1], b[3]);
    v[1] = _mm512_add_pd(t0, t2);
    v[3] = pv_subi(t1, t3);
    v[5] = _mm512_sub_pd(t0, t2);
    v[7] = pv_addi(t1, t3);
}

static inline __attribute__((always_inline)) void pv_dftNr_i(int N, pv *v,
                                                             const int L5)
{
    if (N == 2) pv_dft2r(v);
    else if (N == 3) pv_dft3r(v);
    else if (N == 4) pv_dft4r(v);
    else if (N == 5) pv_dft5r_i(v, L5);
    else if (N == 7) pv_dft7r(v);
    else pv_dft8r(v);
}

static inline __attribute__((always_inline)) void pv_dftNr(int N, pv *v)
{ pv_dftNr_i(N, v, 0); }

/* R, M are compile-time constants through the wrappers below, so every
 * loop fully unrolls and y[] lives in registers (spills only at n = 20). */
static inline __attribute__((always_inline)) void pln_fusedv(
    const int R, const int M, const double *restrict in, ptrdiff_t is,
    double *restrict out, ptrdiff_t os, int w, const double *restrict tw)
{
    for (int c = 0; c < w; c += 4) {
        __mmask8 mk = PLN_MK(w, c);
        pv y[25];
        for (int j1 = 0; j1 < R; ++j1) {         /* child DFT_M on x[j1 + R*t] */
            pv v[8];
            for (int t = 0; t < M; ++t)
                v[t] = pv_ld(mk, in + (ptrdiff_t)(j1 + R*t) * is + 2*c);
            pv_dftNr(M, v);
            for (int k2 = 0; k2 < M; ++k2) y[j1*M + k2] = v[k2];
        }
        for (int k2 = 1; k2 < M; ++k2)           /* twiddle w_n^{j1*k2}, k2-major */
            for (int j1 = 1; j1 < R; ++j1) {
                const double *t = tw + 2*((size_t)(k2-1)*(R-1) + (j1-1));
                y[j1*M + k2] = pv_cmul(y[j1*M + k2],
                                       _mm512_set1_pd(t[0]), _mm512_set1_pd(t[1]));
            }
        for (int k2 = 0; k2 < M; ++k2) {         /* DFT_R across j1, store */
            pv v[8];
            for (int j1 = 0; j1 < R; ++j1) v[j1] = y[j1*M + k2];
            pv_dftNr(R, v);
            for (int k1 = 0; k1 < R; ++k1)
                pv_st(out + (ptrdiff_t)(k1*M + k2) * os + 2*c, mk, v[k1]);
        }
    }
}

#define PLN_FWRAP(R, M) \
static void pln_f_##R##_##M(const double *restrict in, ptrdiff_t is, \
        double *restrict out, ptrdiff_t os, int w, const double *restrict tw) \
{ pln_fusedv(R, M, in, is, out, os, w, tw); }
PLN_FWRAP(2, 2) PLN_FWRAP(2, 3) PLN_FWRAP(2, 4) PLN_FWRAP(2, 5) PLN_FWRAP(2, 7)
PLN_FWRAP(2, 8)
PLN_FWRAP(3, 2) PLN_FWRAP(3, 3) PLN_FWRAP(3, 4) PLN_FWRAP(3, 5) PLN_FWRAP(3, 7)
PLN_FWRAP(3, 8)
PLN_FWRAP(4, 2) PLN_FWRAP(4, 3) PLN_FWRAP(4, 4) PLN_FWRAP(4, 5)
PLN_FWRAP(5, 2) PLN_FWRAP(5, 3) PLN_FWRAP(5, 4) PLN_FWRAP(5, 5)
PLN_FWRAP(7, 2) PLN_FWRAP(7, 3)
PLN_FWRAP(8, 2) PLN_FWRAP(8, 3)
#undef PLN_FWRAP

static void pln_fused_run(int r, int m, const double *in, ptrdiff_t is,
                          double *out, ptrdiff_t os, int w, const double *tw)
{
    switch (r * 16 + m) {
    case 2*16+2: pln_f_2_2(in, is, out, os, w, tw); break;
    case 2*16+3: pln_f_2_3(in, is, out, os, w, tw); break;
    case 2*16+4: pln_f_2_4(in, is, out, os, w, tw); break;
    case 2*16+5: pln_f_2_5(in, is, out, os, w, tw); break;
    case 2*16+7: pln_f_2_7(in, is, out, os, w, tw); break;
    case 2*16+8: pln_f_2_8(in, is, out, os, w, tw); break;
    case 3*16+2: pln_f_3_2(in, is, out, os, w, tw); break;
    case 3*16+3: pln_f_3_3(in, is, out, os, w, tw); break;
    case 3*16+4: pln_f_3_4(in, is, out, os, w, tw); break;
    case 3*16+5: pln_f_3_5(in, is, out, os, w, tw); break;
    case 3*16+7: pln_f_3_7(in, is, out, os, w, tw); break;
    case 3*16+8: pln_f_3_8(in, is, out, os, w, tw); break;
    case 4*16+2: pln_f_4_2(in, is, out, os, w, tw); break;
    case 4*16+3: pln_f_4_3(in, is, out, os, w, tw); break;
    case 4*16+4: pln_f_4_4(in, is, out, os, w, tw); break;
    case 4*16+5: pln_f_4_5(in, is, out, os, w, tw); break;
    case 5*16+2: pln_f_5_2(in, is, out, os, w, tw); break;
    case 5*16+3: pln_f_5_3(in, is, out, os, w, tw); break;
    case 5*16+4: pln_f_5_4(in, is, out, os, w, tw); break;
    case 5*16+5: pln_f_5_5(in, is, out, os, w, tw); break;
    case 7*16+2: pln_f_7_2(in, is, out, os, w, tw); break;
    case 7*16+3: pln_f_7_3(in, is, out, os, w, tw); break;
    case 8*16+2: pln_f_8_2(in, is, out, os, w, tw); break;
    default:     pln_f_8_3(in, is, out, os, w, tw); break;
    }
}

/* ------------------------------------------------------------------ */
/* FUSED GOOD-THOMAS codelets (gen_r7): n = R*M, gcd(R,M) = 1, both      */
/* stages hard leaves.  Ruritanian input map n = (M*n1 + R*n2) mod N     */
/* and output map k = (M*k1 + R*k2) mod N make the whole DFT twiddle-    */
/* FREE:                                                                 */
/*   X[(M k1 + R k2) % N] = DFT_R at freq (M k1 % R) over n1 of          */
/*       [ DFT_M at freq (R k2 % M) of group n1 ],                       */
/* group n1 = rows (M n1 + R n2) % N in n2 order.  R, M are compile-time */
/* through the wrappers, so every index expression folds to an immediate */
/* -- the CRT permutation costs NOTHING here, unlike the generic PFA     */
/* executor's gather/scatter passes (gen_r1's measured 30n penalty).     */
/* vs the fused CT on the same pair: the (R-1)(M-1) twiddle cmuls are    */
/* gone and register pressure DROPS (no twiddle broadcasts).             */
/* gen_pfa_small / gen_batchlane's PFA economics, hosted generically.    */
/* ------------------------------------------------------------------ */
static inline __attribute__((always_inline)) void pln_gtv(
    const int R, const int M, const double *restrict in, ptrdiff_t is,
    double *restrict out, ptrdiff_t os, int w)
{
    const int N = R * M;
    const int L5 = (R == 2 && M == 5);   /* lifted DFT5 only in gt(2,5) */
    for (int c = 0; c < w; c += 4) {
        __mmask8 mk = PLN_MK(w, c);
        pv y[25];
        for (int j1 = 0; j1 < R; ++j1) {
            pv v[8];
            for (int t = 0; t < M; ++t)
                v[t] = pv_ld(mk, in + (ptrdiff_t)((M*j1 + R*t) % N) * is + 2*c);
            pv_dftNr_i(M, v, L5);
            for (int q = 0; q < M; ++q) y[j1*M + q] = v[q];
        }
        for (int k2 = 0; k2 < M; ++k2) {
            const int q2 = (R * k2) % M;
            pv v[8];
            for (int j1 = 0; j1 < R; ++j1) v[j1] = y[j1*M + q2];
            pv_dftNr(R, v);
            for (int k1 = 0; k1 < R; ++k1)
                pv_st(out + (ptrdiff_t)((M*k1 + R*k2) % N) * os + 2*c, mk,
                      v[(M*k1) % R]);
        }
    }
}

#define PLN_GWRAP(R, M) \
static void pln_g_##R##_##M(const double *restrict in, ptrdiff_t is, \
        double *restrict out, ptrdiff_t os, int w) \
{ pln_gtv(R, M, in, is, out, os, w); }
PLN_GWRAP(2, 3) PLN_GWRAP(2, 5) PLN_GWRAP(2, 7)
PLN_GWRAP(3, 4) PLN_GWRAP(3, 5) PLN_GWRAP(3, 7) PLN_GWRAP(3, 8)
PLN_GWRAP(4, 5)
#undef PLN_GWRAP

/* enumeration always emits the PFA node with r = the SMALLER factor, so
 * only canonical (R < M) pairs exist */
static void pln_gt_run(int r, int m, const double *in, ptrdiff_t is,
                       double *out, ptrdiff_t os, int w)
{
    switch (r * 16 + m) {
    case 2*16+3: pln_g_2_3(in, is, out, os, w); break;
    case 2*16+5: pln_g_2_5(in, is, out, os, w); break;
    case 2*16+7: pln_g_2_7(in, is, out, os, w); break;
    case 3*16+4: pln_g_3_4(in, is, out, os, w); break;
    case 3*16+5: pln_g_3_5(in, is, out, os, w); break;
    case 3*16+7: pln_g_3_7(in, is, out, os, w); break;
    case 3*16+8: pln_g_3_8(in, is, out, os, w); break;
    default:     pln_g_4_5(in, is, out, os, w); break;
    }
}

/* ------------------------------------------------------------------ */
/* SPLIT-LANE batch-group codelets (gen_r5).  The batch-lane seam my    */
/* r2-r4 records kept deferring, done generically: 8 volumes of one     */
/* batch are packed site-major SPLIT-complex -- site = {8 re, 8 im} =   */
/* 2 zmm, lane v = volume v -- so every pencil DFT runs full-width      */
/* with broadcast-scalar twiddles.  Consequences, all structural:       */
/*   - cmul = 2 mul + 2 fma, addi/subi = add + sub: ZERO shuffles       */
/*     anywhere (interleaved form pays 1 port-5 permute per cmul and    */
/*     per +-i, and port 5 hosts the second FMA pipe on ICX);           */
/*   - the batch dimension is the vector dimension: no masked tails at  */
/*     any L, and axis 2 runs directly on strided rows -- the per-plane */
/*     transposes (and the P plane) do not exist in this mode;          */
/*   - the map ladder sees 8 sites per zmm instead of 4 (interleaved    */
/*     lanes duplicate |z| per pair): half the rsqrt/rcp work.          */
/* Arithmetic is op-for-op the pv_* form (fmsub/fmadd pairs match       */
/* fmaddsub's per-lane rounding), so outputs are bit-identical to the   */
/* per-volume path on the same tree.  Idea credit: gen_batchlane /      */
/* gen_pfa_small's SoA-8 engines and gen_layout's split-lane fold demo. */
/* ------------------------------------------------------------------ */
typedef struct { pv re, im; } pw;
static inline pw pw_ld(const double *p)
{ pw v; v.re = _mm512_loadu_pd(p); v.im = _mm512_loadu_pd(p + 8); return v; }
static inline void pw_stv(double *p, pw v)
{ _mm512_storeu_pd(p, v.re); _mm512_storeu_pd(p + 8, v.im); }
static inline pw pw_add(pw a, pw b)
{ pw r; r.re = _mm512_add_pd(a.re, b.re); r.im = _mm512_add_pd(a.im, b.im); return r; }
static inline pw pw_sub(pw a, pw b)
{ pw r; r.re = _mm512_sub_pd(a.re, b.re); r.im = _mm512_sub_pd(a.im, b.im); return r; }
static inline pw pw_addi(pw a, pw b)      /* a + i*b */
{ pw r; r.re = _mm512_sub_pd(a.re, b.im); r.im = _mm512_add_pd(a.im, b.re); return r; }
static inline pw pw_subi(pw a, pw b)      /* a - i*b */
{ pw r; r.re = _mm512_add_pd(a.re, b.im); r.im = _mm512_sub_pd(a.im, b.re); return r; }
static inline pw pw_cmulw(pw v, pv wr, pv wi)   /* rounding == pv_cmul per lane */
{
    pw r;
    r.re = _mm512_fmsub_pd(v.re, wr, _mm512_mul_pd(v.im, wi));
    r.im = _mm512_fmadd_pd(v.im, wr, _mm512_mul_pd(v.re, wi));
    return r;
}
static inline pv pw_neg(pv x) { return _mm512_xor_pd(x, _mm512_set1_pd(-0.0)); }

/* store with the graded map fused in:  z = v + c;  z <- z / (1 + |z|).
 * Exactly pln_map_span_w's op sequence (same ladder, same rounding), run
 * at store time on the LAST axis pass so the separate map sweep over the
 * group (state rw + c read) disappears.  gen_batchlane / gen_pfa_small
 * ship the map fused in their final-axis stores; this is that move in the
 * generic split engine. */
static inline __attribute__((always_inline)) void pw_stmap(double *p, pw v,
                                                           const double *c)
{
    const pv half = _mm512_set1_pd(0.5), th = _mm512_set1_pd(1.5);
    const pv one = _mm512_set1_pd(1.0), two = _mm512_set1_pd(2.0);
    const pv tiny = _mm512_set1_pd(1e-300);
    pv re = _mm512_add_pd(v.re, _mm512_loadu_pd(c));
    pv im = _mm512_add_pd(v.im, _mm512_loadu_pd(c + 8));
    pv h = _mm512_add_pd(_mm512_mul_pd(re, re), _mm512_mul_pd(im, im));
    h = _mm512_max_pd(h, tiny);
    pv r = _mm512_rsqrt14_pd(h);
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(_mm512_mul_pd(half, h), r), r, th));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(_mm512_mul_pd(half, h), r), r, th));
    pv d = _mm512_fmadd_pd(h, r, one);
    pv s = _mm512_rcp14_pd(d);
    s = _mm512_mul_pd(s, _mm512_fnmadd_pd(d, s, two));
    s = _mm512_mul_pd(s, _mm512_fnmadd_pd(d, s, two));
    _mm512_storeu_pd(p, _mm512_mul_pd(re, s));
    _mm512_storeu_pd(p + 8, _mm512_mul_pd(im, s));
}

static inline __attribute__((always_inline)) void pw_dft2s(pw *v)
{
    pw a = v[0];
    v[0] = pw_add(a, v[1]);
    v[1] = pw_sub(a, v[1]);
}

static inline __attribute__((always_inline)) void pw_dft3s(pw *v)
{
    const pv C = _mm512_set1_pd(-0.5), S = _mm512_set1_pd(0.86602540378443864676);
    pw t = pw_add(v[1], v[2]), u = pw_sub(v[1], v[2]);
    pw x0 = v[0], vv, su;
    v[0] = pw_add(x0, t);
    vv.re = _mm512_fmadd_pd(C, t.re, x0.re); vv.im = _mm512_fmadd_pd(C, t.im, x0.im);
    su.re = _mm512_mul_pd(S, u.re);          su.im = _mm512_mul_pd(S, u.im);
    v[1] = pw_subi(vv, su);
    v[2] = pw_addi(vv, su);
}

static inline __attribute__((always_inline)) void pw_dft4s(pw *v)
{
    pw t0 = pw_add(v[0], v[2]), t1 = pw_sub(v[0], v[2]);
    pw t2 = pw_add(v[1], v[3]), t3 = pw_sub(v[1], v[3]);
    v[0] = pw_add(t0, t2);
    v[1] = pw_subi(t1, t3);
    v[2] = pw_sub(t0, t2);
    v[3] = pw_addi(t1, t3);
}

static inline __attribute__((always_inline)) void pw_dft5s_i(pw *v, const int L5)
{
    const pv C1 = _mm512_set1_pd(0.30901699437494742410);
    const pv C2 = _mm512_set1_pd(-0.80901699437494742410);
    const pv S1 = _mm512_set1_pd(0.95105651629515357212);
    const pv S2 = _mm512_set1_pd(0.58778525229247312917);
    const pv S2n = _mm512_set1_pd(-0.58778525229247312917);
    pw t1 = pw_add(v[1], v[4]), t2 = pw_add(v[2], v[3]);
    pw t3 = pw_sub(v[1], v[4]), t4 = pw_sub(v[2], v[3]);
    pw x0 = v[0], a, b, cc, dd, u;
    u.re = u.im = _mm512_setzero_pd();
    v[0] = pw_add(x0, pw_add(t1, t2));
    a.re = _mm512_fmadd_pd(C1, t1.re, _mm512_fmadd_pd(C2, t2.re, x0.re));
    a.im = _mm512_fmadd_pd(C1, t1.im, _mm512_fmadd_pd(C2, t2.im, x0.im));
    if (PLN_LIFT5 && L5) {          /* lifted v-pair -- (2,5) GT only */
        const pv PH = _mm512_set1_pd(PLN_PHI5), KLn = _mm512_set1_pd(-PLN_KL5);
        u.re = _mm512_fnmadd_pd(PH, t4.re, t3.re);
        u.im = _mm512_fnmadd_pd(PH, t4.im, t3.im);
        b.re = _mm512_fnmadd_pd(S1, u.re, _mm512_mul_pd(KLn, t4.re));
        b.im = _mm512_fnmadd_pd(S1, u.im, _mm512_mul_pd(KLn, t4.im));
    } else {
        b.re = _mm512_fnmadd_pd(S1, t3.re, _mm512_mul_pd(S2n, t4.re));
        b.im = _mm512_fnmadd_pd(S1, t3.im, _mm512_mul_pd(S2n, t4.im));
    }
    v[1] = pw_addi(a, b);
    v[4] = pw_subi(a, b);
    cc.re = _mm512_fmadd_pd(C2, t1.re, _mm512_fmadd_pd(C1, t2.re, x0.re));
    cc.im = _mm512_fmadd_pd(C2, t1.im, _mm512_fmadd_pd(C1, t2.im, x0.im));
    if (PLN_LIFT5 && L5) {
        dd.re = _mm512_mul_pd(S2n, u.re);
        dd.im = _mm512_mul_pd(S2n, u.im);
    } else {
        dd.re = _mm512_fnmadd_pd(S2, t3.re, _mm512_mul_pd(S1, t4.re));
        dd.im = _mm512_fnmadd_pd(S2, t3.im, _mm512_mul_pd(S1, t4.im));
    }
    v[2] = pw_addi(cc, dd);
    v[3] = pw_subi(cc, dd);
}

static inline __attribute__((always_inline)) void pw_dft5s(pw *v)
{ pw_dft5s_i(v, 0); }

static inline __attribute__((always_inline)) void pw_dft7s(pw *v)
{
    const pv C1 = _mm512_set1_pd(PLN_D7C1), C2 = _mm512_set1_pd(PLN_D7C2);
    const pv C3 = _mm512_set1_pd(PLN_D7C3);
    const pv S1 = _mm512_set1_pd(PLN_D7S1), S2 = _mm512_set1_pd(PLN_D7S2);
    const pv S3 = _mm512_set1_pd(PLN_D7S3);
    pw t1 = pw_add(v[1], v[6]), u1 = pw_sub(v[1], v[6]);
    pw t2 = pw_add(v[2], v[5]), u2 = pw_sub(v[2], v[5]);
    pw t3 = pw_add(v[3], v[4]), u3 = pw_sub(v[3], v[4]);
    pw x0 = v[0], e, o;
    v[0] = pw_add(x0, pw_add(t1, pw_add(t2, t3)));
    e.re = _mm512_fmadd_pd(C1, t1.re, _mm512_fmadd_pd(C2, t2.re, _mm512_fmadd_pd(C3, t3.re, x0.re)));
    e.im = _mm512_fmadd_pd(C1, t1.im, _mm512_fmadd_pd(C2, t2.im, _mm512_fmadd_pd(C3, t3.im, x0.im)));
    o.re = _mm512_fmadd_pd(S1, u1.re, _mm512_fmadd_pd(S2, u2.re, _mm512_mul_pd(S3, u3.re)));
    o.im = _mm512_fmadd_pd(S1, u1.im, _mm512_fmadd_pd(S2, u2.im, _mm512_mul_pd(S3, u3.im)));
    v[1] = pw_subi(e, o);
    v[6] = pw_addi(e, o);
    e.re = _mm512_fmadd_pd(C2, t1.re, _mm512_fmadd_pd(C3, t2.re, _mm512_fmadd_pd(C1, t3.re, x0.re)));
    e.im = _mm512_fmadd_pd(C2, t1.im, _mm512_fmadd_pd(C3, t2.im, _mm512_fmadd_pd(C1, t3.im, x0.im)));
    o.re = _mm512_fnmadd_pd(S1, u3.re, _mm512_fnmadd_pd(S3, u2.re, _mm512_mul_pd(S2, u1.re)));
    o.im = _mm512_fnmadd_pd(S1, u3.im, _mm512_fnmadd_pd(S3, u2.im, _mm512_mul_pd(S2, u1.im)));
    v[2] = pw_subi(e, o);
    v[5] = pw_addi(e, o);
    e.re = _mm512_fmadd_pd(C3, t1.re, _mm512_fmadd_pd(C1, t2.re, _mm512_fmadd_pd(C2, t3.re, x0.re)));
    e.im = _mm512_fmadd_pd(C3, t1.im, _mm512_fmadd_pd(C1, t2.im, _mm512_fmadd_pd(C2, t3.im, x0.im)));
    o.re = _mm512_fmadd_pd(S2, u3.re, _mm512_fnmadd_pd(S1, u2.re, _mm512_mul_pd(S3, u1.re)));
    o.im = _mm512_fmadd_pd(S2, u3.im, _mm512_fnmadd_pd(S1, u2.im, _mm512_mul_pd(S3, u1.im)));
    v[3] = pw_subi(e, o);
    v[4] = pw_addi(e, o);
}

static inline __attribute__((always_inline)) void pw_dft8s(pw *v)
{
    const double Sc = 0.70710678118654752440;
    const pv Sp = _mm512_set1_pd(Sc), Sn = _mm512_set1_pd(-Sc);
    pw a[4], b[4];
    for (int j = 0; j < 4; ++j) {
        a[j] = pw_add(v[j], v[j+4]);
        b[j] = pw_sub(v[j], v[j+4]);
    }
    b[1] = pw_cmulw(b[1], Sp, Sn);
    { pw t = b[2]; b[2].re = t.im; b[2].im = pw_neg(t.re); }   /* * -i */
    b[3] = pw_cmulw(b[3], Sn, Sn);
    pw t0 = pw_add(a[0], a[2]), t1 = pw_sub(a[0], a[2]);
    pw t2 = pw_add(a[1], a[3]), t3 = pw_sub(a[1], a[3]);
    v[0] = pw_add(t0, t2);
    v[2] = pw_subi(t1, t3);
    v[4] = pw_sub(t0, t2);
    v[6] = pw_addi(t1, t3);
    t0 = pw_add(b[0], b[2]); t1 = pw_sub(b[0], b[2]);
    t2 = pw_add(b[1], b[3]); t3 = pw_sub(b[1], b[3]);
    v[1] = pw_add(t0, t2);
    v[3] = pw_subi(t1, t3);
    v[5] = pw_sub(t0, t2);
    v[7] = pw_addi(t1, t3);
}

static inline __attribute__((always_inline)) void pw_dftNs_i(int N, pw *v,
                                                             const int L5)
{
    if (N == 2) pw_dft2s(v);
    else if (N == 3) pw_dft3s(v);
    else if (N == 4) pw_dft4s(v);
    else if (N == 5) pw_dft5s_i(v, L5);
    else if (N == 7) pw_dft7s(v);
    else pw_dft8s(v);
}

static inline __attribute__((always_inline)) void pw_dftNs(int N, pw *v)
{ pw_dftNs_i(N, v, 0); }

/* hard leaf over np pencils: rows R apart (irs/ors doubles), pencils
 * ics/ocs apart on the in/out side (gen_r6: split so the @s4 staged pass
 * can read strided volume pencils and write compact stage pencils, and
 * vice versa; all pre-r6 call sites pass ics == ocs); optional fused
 * twiddle on load, tw[j-1] for row j (row 0 free); optional fused map on
 * store (cb = the c field at the SAME offsets as out -- the split c
 * buffer shares the out geometry).  All loads precede all stores per
 * pencil, so in-place row permutations (the two-pass CT below) are legal. */
static inline __attribute__((always_inline)) void pw_leaf(
    const int R, const double *in, ptrdiff_t irs, double *out, ptrdiff_t ors,
    ptrdiff_t ics, ptrdiff_t ocs, int np, const double *tw, const int HAS,
    const double *cb, const int MAP)
{
    for (int p = 0; p < np; ++p, in += ics, out += ocs, cb += MAP ? ocs : 0) {
        pw v[8];
        for (int j = 0; j < R; ++j) {
            v[j] = pw_ld(in + (ptrdiff_t)j * irs);
            if (HAS && j)
                v[j] = pw_cmulw(v[j], _mm512_set1_pd(tw[2*(j-1)]),
                                      _mm512_set1_pd(tw[2*(j-1)+1]));
        }
        pw_dftNs(R, v);
        for (int k = 0; k < R; ++k) {
            if (MAP) pw_stmap(out + (ptrdiff_t)k * ors, v[k], cb + (ptrdiff_t)k * ors);
            else     pw_stv(out + (ptrdiff_t)k * ors, v[k]);
        }
    }
}

#define PW_LWRAP(R) \
static void pw_l##R(const double *in, ptrdiff_t irs, double *out, ptrdiff_t ors, \
        ptrdiff_t ics, ptrdiff_t ocs, int np, const double *tw, const double *cb) \
{ if (tw)      { if (cb) pw_leaf(R, in, irs, out, ors, ics, ocs, np, tw, 1, cb, 1); \
                 else    pw_leaf(R, in, irs, out, ors, ics, ocs, np, tw, 1, NULL, 0); } \
  else         { if (cb) pw_leaf(R, in, irs, out, ors, ics, ocs, np, NULL, 0, cb, 1); \
                 else    pw_leaf(R, in, irs, out, ors, ics, ocs, np, NULL, 0, NULL, 0); } }
PW_LWRAP(2) PW_LWRAP(3) PW_LWRAP(4) PW_LWRAP(5) PW_LWRAP(7) PW_LWRAP(8)
#undef PW_LWRAP

static void pw_leaf_run(int r, const double *in, ptrdiff_t irs, double *out,
                        ptrdiff_t ors, ptrdiff_t ics, ptrdiff_t ocs, int np,
                        const double *tw, const double *cb)
{
    switch (r) {
    case 2: pw_l2(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 3: pw_l3(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 4: pw_l4(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 5: pw_l5(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 7: pw_l7(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    default: pw_l8(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    }
}

/* fused two-level CT, split form: whole n = R*M DFT per pencil in
 * registers; same schedule and constants as pln_fusedv, so outputs are
 * bit-identical to the interleaved fused codelet on the same data.
 * MAP fuses the graded map into the stores (last-axis pass). */
static inline __attribute__((always_inline)) void pln_fusedw(
    const int R, const int M, const double *in, ptrdiff_t irs,
    double *out, ptrdiff_t ors, ptrdiff_t ics, ptrdiff_t ocs, int np,
    const double *tw, const double *cb, const int MAP)
{
    for (int p = 0; p < np; ++p, in += ics, out += ocs, cb += MAP ? ocs : 0) {
        if (R == 2) {
            /* radix-2 root: no y double-buffer -- peak live drops from
             * 2n + 2M to 2n zmm and (2,5) goes spill-free (asm audit:
             * 20 rsp zmm ops -> 0).  Ops and order identical to the
             * generic branch below, outputs bit-identical. */
            pw y0[8], v[8];
            for (int t = 0; t < M; ++t)
                y0[t] = pw_ld(in + (ptrdiff_t)(2*t) * irs);
            pw_dftNs(M, y0);
            for (int t = 0; t < M; ++t)
                v[t] = pw_ld(in + (ptrdiff_t)(1 + 2*t) * irs);
            pw_dftNs(M, v);
            for (int k2 = 1; k2 < M; ++k2) {
                const double *t = tw + 2*(size_t)(k2-1);
                v[k2] = pw_cmulw(v[k2], _mm512_set1_pd(t[0]), _mm512_set1_pd(t[1]));
            }
            for (int k2 = 0; k2 < M; ++k2) {
                pw sm = pw_add(y0[k2], v[k2]), df = pw_sub(y0[k2], v[k2]);
                if (MAP) {
                    pw_stmap(out + (ptrdiff_t)k2 * ors, sm, cb + (ptrdiff_t)k2 * ors);
                    pw_stmap(out + (ptrdiff_t)(M + k2) * ors, df,
                             cb + (ptrdiff_t)(M + k2) * ors);
                } else {
                    pw_stv(out + (ptrdiff_t)k2 * ors, sm);
                    pw_stv(out + (ptrdiff_t)(M + k2) * ors, df);
                }
            }
            continue;
        }
        pw y[25];
        for (int j1 = 0; j1 < R; ++j1) {
            pw v[8];
            for (int t = 0; t < M; ++t)
                v[t] = pw_ld(in + (ptrdiff_t)(j1 + R*t) * irs);
            pw_dftNs(M, v);
            for (int k2 = 0; k2 < M; ++k2) y[j1*M + k2] = v[k2];
        }
        for (int k2 = 1; k2 < M; ++k2)
            for (int j1 = 1; j1 < R; ++j1) {
                const double *t = tw + 2*((size_t)(k2-1)*(R-1) + (j1-1));
                y[j1*M + k2] = pw_cmulw(y[j1*M + k2],
                                        _mm512_set1_pd(t[0]), _mm512_set1_pd(t[1]));
            }
        for (int k2 = 0; k2 < M; ++k2) {
            pw v[8];
            for (int j1 = 0; j1 < R; ++j1) v[j1] = y[j1*M + k2];
            pw_dftNs(R, v);
            for (int k1 = 0; k1 < R; ++k1) {
                if (MAP) pw_stmap(out + (ptrdiff_t)(k1*M + k2) * ors, v[k1],
                                  cb + (ptrdiff_t)(k1*M + k2) * ors);
                else     pw_stv(out + (ptrdiff_t)(k1*M + k2) * ors, v[k1]);
            }
        }
    }
}

#define PW_FWRAP(R, M) \
static void pw_f_##R##_##M(const double *in, ptrdiff_t irs, double *out, \
        ptrdiff_t ors, ptrdiff_t ics, ptrdiff_t ocs, int np, const double *tw, \
        const double *cb) \
{ if (cb) pln_fusedw(R, M, in, irs, out, ors, ics, ocs, np, tw, cb, 1); \
  else    pln_fusedw(R, M, in, irs, out, ors, ics, ocs, np, tw, NULL, 0); }
PW_FWRAP(2, 2) PW_FWRAP(2, 3) PW_FWRAP(2, 4) PW_FWRAP(2, 5) PW_FWRAP(2, 7)
PW_FWRAP(2, 8)
PW_FWRAP(3, 2) PW_FWRAP(3, 3) PW_FWRAP(3, 4) PW_FWRAP(3, 5) PW_FWRAP(3, 7)
PW_FWRAP(3, 8)
PW_FWRAP(4, 2) PW_FWRAP(4, 3) PW_FWRAP(4, 4) PW_FWRAP(4, 5)
PW_FWRAP(5, 2) PW_FWRAP(5, 3) PW_FWRAP(5, 4) PW_FWRAP(5, 5)
PW_FWRAP(7, 2) PW_FWRAP(7, 3)
PW_FWRAP(8, 2) PW_FWRAP(8, 3)
#undef PW_FWRAP

static void pln_fusedw_run(int r, int m, const double *in, ptrdiff_t irs,
                           double *out, ptrdiff_t ors, ptrdiff_t ics,
                           ptrdiff_t ocs, int np,
                           const double *tw, const double *cb)
{
    switch (r * 16 + m) {
    case 2*16+2: pw_f_2_2(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 2*16+3: pw_f_2_3(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 2*16+4: pw_f_2_4(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 2*16+5: pw_f_2_5(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 2*16+7: pw_f_2_7(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 2*16+8: pw_f_2_8(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 3*16+2: pw_f_3_2(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 3*16+3: pw_f_3_3(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 3*16+4: pw_f_3_4(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 3*16+5: pw_f_3_5(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 3*16+7: pw_f_3_7(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 3*16+8: pw_f_3_8(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 4*16+2: pw_f_4_2(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 4*16+3: pw_f_4_3(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 4*16+4: pw_f_4_4(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 4*16+5: pw_f_4_5(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 5*16+2: pw_f_5_2(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 5*16+3: pw_f_5_3(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 5*16+4: pw_f_5_4(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 5*16+5: pw_f_5_5(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 7*16+2: pw_f_7_2(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 7*16+3: pw_f_7_3(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    case 8*16+2: pw_f_8_2(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    default:     pw_f_8_3(in, irs, out, ors, ics, ocs, np, tw, cb); break;
    }
}

/* fused Good-Thomas, split form (gen_r7): the pln_gtv schedule per pencil
 * on split-complex data -- twiddle-free, shuffle-free, mask-free.  Same
 * in/out pencil-stride split (ics/ocs) and fused-map store option as
 * pln_fusedw, so it serves as an @s1 root and as the child pass of
 * @s2/@s4.  The R == 2 branch mirrors pln_fusedw's: no y double-buffer,
 * peak live 2n zmm instead of 2n + 2M. */
static inline __attribute__((always_inline)) void pln_gtw(
    const int R, const int M, const double *in, ptrdiff_t irs,
    double *out, ptrdiff_t ors, ptrdiff_t ics, ptrdiff_t ocs, int np,
    const double *cb, const int MAP)
{
    const int N = R * M;
    const int L5 = (R == 2 && M == 5);   /* lifted DFT5 only in gt(2,5) */
    for (int p = 0; p < np; ++p, in += ics, out += ocs, cb += MAP ? ocs : 0) {
        if (R == 2) {
            pw y0[8], v[8];
            for (int t = 0; t < M; ++t)               /* group 0: even rows */
                y0[t] = pw_ld(in + (ptrdiff_t)((2*t) % N) * irs);
            pw_dftNs_i(M, y0, L5);
            for (int t = 0; t < M; ++t)               /* group 1: rows M+2t */
                v[t] = pw_ld(in + (ptrdiff_t)((M + 2*t) % N) * irs);
            pw_dftNs_i(M, v, L5);
            for (int k2 = 0; k2 < M; ++k2) {
                const int q2 = (2*k2) % M;            /* M odd: q1 == k1 */
                pw sm = pw_add(y0[q2], v[q2]), df = pw_sub(y0[q2], v[q2]);
                const ptrdiff_t o0 = (ptrdiff_t)((2*k2) % N) * ors;
                const ptrdiff_t o1 = (ptrdiff_t)((M + 2*k2) % N) * ors;
                if (MAP) {
                    pw_stmap(out + o0, sm, cb + o0);
                    pw_stmap(out + o1, df, cb + o1);
                } else {
                    pw_stv(out + o0, sm);
                    pw_stv(out + o1, df);
                }
            }
            continue;
        }
        pw y[25];
        for (int j1 = 0; j1 < R; ++j1) {
            pw v[8];
            for (int t = 0; t < M; ++t)
                v[t] = pw_ld(in + (ptrdiff_t)((M*j1 + R*t) % N) * irs);
            pw_dftNs(M, v);
            for (int q = 0; q < M; ++q) y[j1*M + q] = v[q];
        }
        for (int k2 = 0; k2 < M; ++k2) {
            const int q2 = (R * k2) % M;
            pw v[8];
            for (int j1 = 0; j1 < R; ++j1) v[j1] = y[j1*M + q2];
            pw_dftNs(R, v);
            for (int k1 = 0; k1 < R; ++k1) {
                const ptrdiff_t o = (ptrdiff_t)((M*k1 + R*k2) % N) * ors;
                if (MAP) pw_stmap(out + o, v[(M*k1) % R], cb + o);
                else     pw_stv(out + o, v[(M*k1) % R]);
            }
        }
    }
}

#define PW_GWRAP(R, M) \
static void pw_g_##R##_##M(const double *in, ptrdiff_t irs, double *out, \
        ptrdiff_t ors, ptrdiff_t ics, ptrdiff_t ocs, int np, const double *cb) \
{ if (cb) pln_gtw(R, M, in, irs, out, ors, ics, ocs, np, cb, 1); \
  else    pln_gtw(R, M, in, irs, out, ors, ics, ocs, np, NULL, 0); }
PW_GWRAP(2, 3) PW_GWRAP(2, 5) PW_GWRAP(2, 7)
PW_GWRAP(3, 4) PW_GWRAP(3, 5) PW_GWRAP(3, 7) PW_GWRAP(3, 8)
PW_GWRAP(4, 5)
#undef PW_GWRAP

static void pln_gtw_run(int r, int m, const double *in, ptrdiff_t irs,
                        double *out, ptrdiff_t ors, ptrdiff_t ics,
                        ptrdiff_t ocs, int np, const double *cb)
{
    switch (r * 16 + m) {
    case 2*16+3: pw_g_2_3(in, irs, out, ors, ics, ocs, np, cb); break;
    case 2*16+5: pw_g_2_5(in, irs, out, ors, ics, ocs, np, cb); break;
    case 2*16+7: pw_g_2_7(in, irs, out, ors, ics, ocs, np, cb); break;
    case 3*16+4: pw_g_3_4(in, irs, out, ors, ics, ocs, np, cb); break;
    case 3*16+5: pw_g_3_5(in, irs, out, ors, ics, ocs, np, cb); break;
    case 3*16+7: pw_g_3_7(in, irs, out, ors, ics, ocs, np, cb); break;
    case 3*16+8: pw_g_3_8(in, irs, out, ors, ics, ocs, np, cb); break;
    default:     pw_g_4_5(in, irs, out, ors, ics, ocs, np, cb); break;
    }
}

/* folded odd dense, split form (the pln_fold_applyv algebra per pencil):
 * staging A = x0, s_j, d_j (n groups of 16 doubles, L1-resident), then
 * 4-k-row tiled E/O accumulation with broadcast REAL coefficients.  All
 * pencil loads precede pencil stores -> in place. */
static inline __attribute__((always_inline)) void pln_foldw_i(
    const double *mat, int n, double *base,
    ptrdiff_t rs, ptrdiff_t cs, int np, double *A,
    const double *cb, const int MAP)
{
    const int h = n / 2, hp = PLN_HPAD(h);
    const double *C = mat, *S = mat + (size_t)hp * h;
    for (int p = 0; p < np; ++p, base += cs, cb += MAP ? cs : 0) {
        pw x0 = pw_ld(base);
        pw acc = x0;
        for (int j = 1; j <= h; ++j) {
            pw xj = pw_ld(base + (ptrdiff_t)j * rs);
            pw xn = pw_ld(base + (ptrdiff_t)(n - j) * rs);
            pw s = pw_add(xj, xn);
            pw_stv(A + 16*(size_t)j, s);
            pw_stv(A + 16*(size_t)(h + j), pw_sub(xj, xn));
            acc = pw_add(acc, s);
        }
        if (MAP) pw_stmap(base, acc, cb);        /* X0; all loads done */
        else     pw_stv(base, acc);
        for (int kb = 1; kb <= h; kb += 4) {
            pw er[4], oc[4];
            for (int t = 0; t < 4; ++t) {
                er[t] = x0;
                oc[t].re = oc[t].im = _mm512_setzero_pd();
            }
            for (int j = 1; j <= h; ++j) {
                pw s = pw_ld(A + 16*(size_t)j), d = pw_ld(A + 16*(size_t)(h + j));
                for (int t = 0; t < 4; ++t) {
                    const size_t e = (size_t)(kb - 1 + t) * h + (j - 1);
                    pv cc = _mm512_set1_pd(C[e]), ss = _mm512_set1_pd(S[e]);
                    er[t].re = _mm512_fmadd_pd(cc, s.re, er[t].re);
                    er[t].im = _mm512_fmadd_pd(cc, s.im, er[t].im);
                    oc[t].re = _mm512_fmadd_pd(ss, d.re, oc[t].re);
                    oc[t].im = _mm512_fmadd_pd(ss, d.im, oc[t].im);
                }
            }
            for (int t = 0; t < 4 && kb + t <= h; ++t) {
                const int k = kb + t;
                if (MAP) {
                    pw_stmap(base + (ptrdiff_t)k * rs, pw_subi(er[t], oc[t]),
                             cb + (ptrdiff_t)k * rs);
                    pw_stmap(base + (ptrdiff_t)(n - k) * rs, pw_addi(er[t], oc[t]),
                             cb + (ptrdiff_t)(n - k) * rs);
                } else {
                    pw_stv(base + (ptrdiff_t)k * rs, pw_subi(er[t], oc[t]));
                    pw_stv(base + (ptrdiff_t)(n - k) * rs, pw_addi(er[t], oc[t]));
                }
            }
        }
    }
}

/* the MAP specialization is EXPLICIT (gen_r6): gcc-11 constprop-cloned
 * these two forms by itself until pln_s8_step grew the @s4 branch, then
 * silently stopped -- the generic form keeps both store paths live inside
 * the 16-accumulator E/O tile and cost @s3 at L=31 ~5-9% (interleaved r5
 * pairs 140-142 vs 147-163 us).  Never leave this to the inliner. */
static void pln_foldw_m1(const double *mat, int n, double *base, ptrdiff_t rs,
                         ptrdiff_t cs, int np, double *A, const double *cb)
{ pln_foldw_i(mat, n, base, rs, cs, np, A, cb, 1); }
static void pln_foldw_m0(const double *mat, int n, double *base, ptrdiff_t rs,
                         ptrdiff_t cs, int np, double *A)
{ pln_foldw_i(mat, n, base, rs, cs, np, A, NULL, 0); }
static void pln_foldw(const double *mat, int n, double *base, ptrdiff_t rs,
                      ptrdiff_t cs, int np, double *A, const double *cb)
{
    if (cb) pln_foldw_m1(mat, n, base, rs, cs, np, A, cb);
    else    pln_foldw_m0(mat, n, base, rs, cs, np, A);
}

/* the graded map on split-group data: ns groups of {8 re, 8 im}.  Same
 * ladder as pln_map_span but every zmm lane is a DISTINCT site (the
 * interleaved form duplicates |z| per re/im pair): half the ladder work. */
static __attribute__((unused)) void pln_map_span_w(double *st, const double *cf, size_t ns)
{                                    /* kept: the unfused-ladder reference */
    const pv half = _mm512_set1_pd(0.5), th = _mm512_set1_pd(1.5);
    const pv one = _mm512_set1_pd(1.0), two = _mm512_set1_pd(2.0);
    const pv tiny = _mm512_set1_pd(1e-300);
    for (size_t i = 0; i < ns; ++i) {
        double *p = st + 16*i;
        const double *q = cf + 16*i;
        pv re = _mm512_add_pd(_mm512_loadu_pd(p), _mm512_loadu_pd(q));
        pv im = _mm512_add_pd(_mm512_loadu_pd(p + 8), _mm512_loadu_pd(q + 8));
        pv h = _mm512_add_pd(_mm512_mul_pd(re, re), _mm512_mul_pd(im, im));
        h = _mm512_max_pd(h, tiny);
        pv r = _mm512_rsqrt14_pd(h);
        r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(_mm512_mul_pd(half, h), r), r, th));
        r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(_mm512_mul_pd(half, h), r), r, th));
        pv d = _mm512_fmadd_pd(h, r, one);
        pv s = _mm512_rcp14_pd(d);
        s = _mm512_mul_pd(s, _mm512_fnmadd_pd(d, s, two));
        s = _mm512_mul_pd(s, _mm512_fnmadd_pd(d, s, two));
        _mm512_storeu_pd(p, _mm512_mul_pd(re, s));
        _mm512_storeu_pd(p + 8, _mm512_mul_pd(im, s));
    }
}
#endif /* PLN_SIMD */

#if PLN_SIMD
/* DFT7 leaf entries are NOINLINE wrappers (gen_r7): inlining pln_lv7 into
 * the two leaf dispatchers grew their constprop clones ~20% and cost the
 * 7-free c8(d5) tree at L=40 +1..3% (3/3 RACE=0 pairs) from code layout
 * alone -- the gen_twiddle r5 case-bloat hazard, pinned structurally per
 * the gen_r6 lesson.  Sizes that never contain 7 now execute byte-similar
 * dispatchers; d7 sizes pay one call per node, noise next to the body. */
static __attribute__((noinline)) void pln_lv7_nt(const double *restrict in,
        ptrdiff_t is, double *restrict out, ptrdiff_t os, int w)
{ pln_lv7(in, is, out, os, w, NULL, 0); }
static __attribute__((noinline)) void pln_lv7_tw(const double *restrict in,
        ptrdiff_t is, double *restrict out, ptrdiff_t os, int w,
        const double *restrict tw)
{ pln_lv7(in, is, out, os, w, tw, 1); }
#endif

static void pln_leaf_apply_tw(int r, const double *in, ptrdiff_t is,
                              double *out, ptrdiff_t os, int w, const double *tw)
{
#if PLN_SIMD
    switch (r) {
    case 2: pln_lv2(in, is, out, os, w, tw, 1); break;
    case 3: pln_lv3(in, is, out, os, w, tw, 1); break;
    case 4: pln_lv4(in, is, out, os, w, tw, 1); break;
    case 5: pln_lv5(in, is, out, os, w, tw, 1); break;
    case 7: pln_lv7_tw(in, is, out, os, w, tw); break;
    default: pln_lv8(in, is, out, os, w, tw, 1); break;
    }
#else
    switch (r) {
    case 2: pln_leaf2_tw(in, is, out, os, w, tw); break;
    case 3: pln_leaf3_tw(in, is, out, os, w, tw); break;
    case 4: pln_leaf4_tw(in, is, out, os, w, tw); break;
    case 5: pln_leaf5_tw(in, is, out, os, w, tw); break;
    case 7: pln_leaf7_tw(in, is, out, os, w, tw); break;
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
    case 7: pln_lv7_nt(in, is, out, os, w); break;
    case 8: pln_lv8(in, is, out, os, w, NULL, 0); break;
    default: pln_dense_run(mat, r, in, is, out, os, w); break;
    }
#else
    switch (r) {
    case 2: pln_leaf2(in, is, out, os, w); break;
    case 3: pln_leaf3(in, is, out, os, w); break;
    case 4: pln_leaf4(in, is, out, os, w); break;
    case 5: pln_leaf5(in, is, out, os, w); break;
    case 7: pln_leaf7(in, is, out, os, w); break;
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
    int fus;                     /* CT: register-resident fused codelet (gen_r4) */
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
#if PLN_SIMD
        x->fus = pln_ct_fused(r, t->s1);      /* register-resident: no arena,
                                                 no sub-plan (gen_r4) */
#endif
        own = x->fus ? 0 : 2 * (size_t)tile * t->n;
        if (!x->fus && !pln_leaf_hard(r) && (r & 1) && r >= 11)
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
        if (!x->fus && !pln_leaf_hard(r)) {
            x->fold = (r & 1) && r >= 11;
            x->mat = x->fold ? pln_fold_matrix(r) : pln_dense_matrix(r);
        }
        if (!x->fus)
            x->s1 = pln_xbuild_rec(t->s1, base + own, hi, tile);
        break;
    }
    case PLN_PFA: {
        int n1 = t->r, n2 = t->n / t->r, n = t->n;
#if PLN_SIMD
        x->fus = pln_gt_fused(t->s1, t->s2);  /* register-resident twiddle-free
                                                 GT codelet (gen_r7): no arena,
                                                 no permutation tables, no
                                                 sub-plans */
        if (x->fus) break;
#endif
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
#if PLN_SIMD
        if (x->fus) {                          /* whole node in registers */
            pln_fused_run(r, m, src, ss, dst, ds, w, x->tw);
            break;
        }
#endif
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
#if PLN_SIMD
        if (x->fus) {                          /* fused GT: whole node in registers */
            pln_gt_run(n1, n2, src, ss, dst, ds, w);
            break;
        }
#endif
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

/* host L3 size in bytes (sysfs, cached; fallback = the 24 MB dev node).
 * Used only as the custody DEFAULT -- the race measures the truth. */
static long pln_l3_bytes(void)
{
    static long cached = 0;
    if (cached) return cached;
    long b = 25165824;                        /* 24 MB: a80n0 / Gold 6326 */
    FILE *f = fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r");
    if (f) {
        long v = 0; char u = 0;
        if (fscanf(f, "%ld%c", &v, &u) >= 1 && v > 0)
            b = u == 'M' ? v << 20 : u == 'K' ? v << 10 : v;
        fclose(f);
    }
    return cached = b;
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
    int fusemap;                 /* gen_r6: chain step fuses the map into the
                                    transpose-out exit.  12 < L <= 80, all
                                    same-core interleaved pairs: 40 -6.5%,
                                    50 -4.8%, 64 -10%, 80 -2%, but 100 +8-10%
                                    (3/3 -- the fused exit's blockwise c walk
                                    re-reads the c plane L/4 times and the
                                    plane pair stops being cheap), and
                                    gen_layout r5 measured small L1-resident
                                    L paying dead-lane divides.
                                    GEN_PLANNER_PVFUSE overrides for A/B. */
    int cflush;                  /* gen_r9: c-line custody (ADOPTED from
                                    gen_pfa_large's r5-r7 c-bypass, the PMU
                                    audit's confirmation that L=100 is pure
                                    traffic).  The volume-major chain re-reads
                                    state+c every step; when that working set
                                    (32*L^3 B) busts the ICX 24 MB L3, LRU
                                    thrashes BOTH streams.  clflushopt on each
                                    consumed c line right after the map span
                                    keeps c out of L3 entirely, so the 16 MB
                                    state stays resident across all m steps
                                    and DRAM traffic drops to the c stream.
                                    No arithmetic touched: bit-identical.
                                    gen_r10: the 24 MB-L3 ICX node REFUTED
                                    the theory (custody-on +6..11% at L=100,
                                    3/3 pairs) -- default OFF everywhere;
                                    the create() race's custody PLAYOFF on
                                    the winning pv arm still measures per
                                    host and banks @f0/@f1 through wisdom.
                                    GEN_PLANNER_CFLUSH pins for A/B. */
    int cflush_elig;             /* separate-map path and big enough to care */
    int cflush_pin;              /* env-pinned: playoff and tags must not move it */
    pln_xplan *xp;
    double *P;                   /* one transposed (z,y) plane, pitch x L */
} pln_p3d;

/* gen_r10: the race's TILE PLAYOFF (below) rebuilds the winning pv plan at
 * an alternate row width; this request slot beats the default but never the
 * explicit GEN_PLANNER_TILE env pin (checked by the caller). */
static int pln_tile_req = 0;

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
                                                        25: 70.3 vs 77.5).
                                                        gen_r10: t16 WINS at
                                                        40/50 (-2.5..4%, 3/3
                                                        pairs) and loses at
                                                        32/100 -- per-(host,L)
                                                        truth, so the race now
                                                        holds a tile playoff
                                                        and banks @t16. */
        if (pln_tile_req >= 4) tile = pln_tile_req;
        const char *e = getenv("GEN_PLANNER_TILE");  /* dev A/B override */
        if (e && atoi(e) >= 4) tile = atoi(e);
        p->xp = pln_xplan_build(t, tile);
    }
#if PLN_SIMD
    /* gen_r6 gate: fused exit wins 12 < L <= 80 (z0-outer walk), separate
     * pair-packed map elsewhere.  gen_r10 re-litigated L > 80 with the
     * y0-outer exit (pln_transpose_out_map_yo): it repairs most of the r6
     * loss -- the walk order WAS the bulk of the pathology (node pairs at
     * 100: yo 5146-5589 vs z0 5722-6018) -- but still loses to the separate
     * map pass 3/3 by 0.2-2.2% (uf 5073/5414/5466 vs yo 5146/5424/5589):
     * at 100 the two sequential full-plane sweeps ride the prefetcher at
     * effectively no cost, and the exit's strided P reads + vdivpd beat the
     * traffic they delete.  Default stays unfused at L > 80; PVFUSE=2 keeps
     * the yo exit reachable as a cross-arch race arm. */
    p->fusemap = (L > 12 && L <= 80);
    {
        const char *e = getenv("GEN_PLANNER_PVFUSE");
        if (e && (e[0] >= '0' && e[0] <= '2')) p->fusemap = e[0] - '0';
    }
#else
    p->fusemap = 0;              /* the scalar exit has no fused-map path */
#endif
#if defined(__CLFLUSHOPT__)
    /* custody is ELIGIBLE only where the separate-pass map runs (fusemap==0)
     * and the per-volume chain set is large enough to plausibly contend for
     * L3 (32*L^3 > 16 MB).  gen_r10 NODE VERDICT: custody-on LOSES at the one
     * scored eligible size, L=100 B=1 -- 5565-5612 off vs 5919-6224 on,
     * +6..11%, 3/3 interleaved pairs on a81n2 (ICX 24 MB L3, the exact
     * regime the r9 sysfs heuristic predicted a win) -- the ~2.5K clflushopt
     * uops per plane cost more than the 16 MB read-once c stream's eviction
     * pressure ever did.  DEFAULT is therefore OFF everywhere; eligibility
     * is kept so the race's custody playoff (below) can still measure a host
     * where it wins and bank @f1 through wisdom. */
    p->cflush_elig = (!p->fusemap && 32.0 * (double)L * L * L > 16.0e6);
    p->cflush = 0;
    {
        const char *e = getenv("GEN_PLANNER_CFLUSH");
        if (e && (e[0] == '0' || e[0] == '1')) {
            p->cflush = e[0] - '0';
            p->cflush_pin = 1;
        }
    }
#else
    p->cflush = 0; p->cflush_elig = 0; p->cflush_pin = 0;
#endif
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
#ifndef PLN_TRZST
#define PLN_TRZST 0  /* gen_r10, tried and REFUTED ON ICL: gen_pow2 gen_r9's
                        ZST (store the 4x4 transpose's combine rows as 256-bit
                        halves -- ymm mov + vextractf64x4-to-mem, no p5 uop --
                        instead of 4 vshuff64x2 + zmm stores).  Bit-identical
                        bytes, front-end neutral, -4 p5/block on paper; the
                        node said +2..4% at 32/40 and +3..6% at 100, 3/3
                        interleaved pairs each (e.g. 32: ctl 113.1-116.5 vs
                        zst 117.3-117.9).  My transpose destinations are only
                        16B-aligned (complex-granular strided rows), so half
                        the split 256-bit stores line-cross where the zmm
                        store was one access -- unlike gen_pow2's 64B-aligned
                        custody rows.  =1 kept as a cross-arch race arm only
                        (their SPR pairs measured it a win on THEIR kernel;
                        never enable here without a node win). */
#endif
/* 4x4 complex block transpose, strides in doubles: 4 loads, 4+4 shuffles
 * (PLN_TRZST deletes the second 4 through the store path), 4 full lines
 * stored instead of 16 scalar 16-byte moves */
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
#if PLN_TRZST
    _mm256_storeu_pd(d,              _mm512_castpd512_pd256(a));
    _mm256_storeu_pd(d + 4,          _mm512_castpd512_pd256(b));
    _mm256_storeu_pd(d + drow,       _mm512_extractf64x4_pd(a, 1));
    _mm256_storeu_pd(d + drow + 4,   _mm512_extractf64x4_pd(b, 1));
    _mm256_storeu_pd(d + 2*drow,     _mm512_castpd512_pd256(e));
    _mm256_storeu_pd(d + 2*drow + 4, _mm512_castpd512_pd256(f));
    _mm256_storeu_pd(d + 3*drow,     _mm512_extractf64x4_pd(e, 1));
    _mm256_storeu_pd(d + 3*drow + 4, _mm512_extractf64x4_pd(f, 1));
#else
    _mm512_storeu_pd(d,          _mm512_shuffle_f64x2(a, b, 0x44));
    _mm512_storeu_pd(d + drow,   _mm512_shuffle_f64x2(a, b, 0xEE));
    _mm512_storeu_pd(d + 2*drow, _mm512_shuffle_f64x2(e, f, 0x44));
    _mm512_storeu_pd(d + 3*drow, _mm512_shuffle_f64x2(e, f, 0xEE));
#endif
}

/* masked edge variant (gen_r4): nr x nc block, nr, nc <= 4.  Zero-fill
 * lanes only ever reach rows/columns the masks drop, so tails cost the
 * same shuffles as a full block instead of the old scalar copy loops
 * (36 of 100 elements per transpose at L=10 went through them, twice). */
static inline void pln_tr4x4m(const double *restrict s, ptrdiff_t srow,
                              double *restrict d, ptrdiff_t drow, int nr, int nc)
{
    const __m512i i0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    const __m512i i1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
    const __mmask8 lm = (__mmask8)((1u << (2*nc)) - 1);
    pv r0 = pv_ld(lm, s);
    pv r1 = pv_ld(nr > 1 ? lm : (__mmask8)0, s + srow);
    pv r2 = pv_ld(nr > 2 ? lm : (__mmask8)0, s + 2*srow);
    pv r3 = pv_ld(nr > 3 ? lm : (__mmask8)0, s + 3*srow);
    pv a = _mm512_permutex2var_pd(r0, i0, r1);
    pv b = _mm512_permutex2var_pd(r2, i0, r3);
    pv e = _mm512_permutex2var_pd(r0, i1, r1);
    pv f = _mm512_permutex2var_pd(r2, i1, r3);
    const __mmask8 sm = (__mmask8)((1u << (2*nr)) - 1);
    pv_st(d, sm, _mm512_shuffle_f64x2(a, b, 0x44));
    if (nc > 1) pv_st(d + drow,   sm, _mm512_shuffle_f64x2(a, b, 0xEE));
    if (nc > 2) pv_st(d + 2*drow, sm, _mm512_shuffle_f64x2(e, f, 0x44));
    if (nc > 3) pv_st(d + 3*drow, sm, _mm512_shuffle_f64x2(e, f, 0xEE));
}

/* gen_r11: pair-swap transpose of two mirrored 4x4 complex blocks IN PLACE
 * (both blocks fully loaded before either store, so a and b may be any two
 * disjoint blocks of the same plane).  Building block of the in-place square
 * plane transpose the alternating-layout chain uses. */
static inline void pln_tr4x4_swap(double *a, double *b, ptrdiff_t row)
{
    const __m512i i0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    const __m512i i1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
    pv a0 = _mm512_loadu_pd(a),         a1 = _mm512_loadu_pd(a + row);
    pv a2 = _mm512_loadu_pd(a + 2*row), a3 = _mm512_loadu_pd(a + 3*row);
    pv b0 = _mm512_loadu_pd(b),         b1 = _mm512_loadu_pd(b + row);
    pv b2 = _mm512_loadu_pd(b + 2*row), b3 = _mm512_loadu_pd(b + 3*row);
    pv pa = _mm512_permutex2var_pd(a0, i0, a1), qa = _mm512_permutex2var_pd(a2, i0, a3);
    pv ra = _mm512_permutex2var_pd(a0, i1, a1), sa = _mm512_permutex2var_pd(a2, i1, a3);
    pv pb = _mm512_permutex2var_pd(b0, i0, b1), qb = _mm512_permutex2var_pd(b2, i0, b3);
    pv rb = _mm512_permutex2var_pd(b0, i1, b1), sb = _mm512_permutex2var_pd(b2, i1, b3);
    _mm512_storeu_pd(b,         _mm512_shuffle_f64x2(pa, qa, 0x44));
    _mm512_storeu_pd(b + row,   _mm512_shuffle_f64x2(pa, qa, 0xEE));
    _mm512_storeu_pd(b + 2*row, _mm512_shuffle_f64x2(ra, sa, 0x44));
    _mm512_storeu_pd(b + 3*row, _mm512_shuffle_f64x2(ra, sa, 0xEE));
    _mm512_storeu_pd(a,         _mm512_shuffle_f64x2(pb, qb, 0x44));
    _mm512_storeu_pd(a + row,   _mm512_shuffle_f64x2(pb, qb, 0xEE));
    _mm512_storeu_pd(a + 2*row, _mm512_shuffle_f64x2(rb, sb, 0x44));
    _mm512_storeu_pd(a + 3*row, _mm512_shuffle_f64x2(rb, sb, 0xEE));
}

/* masked pair-swap for the L%4 edges: a is nr x nc at (i,j), b is the
 * mirrored nc x nr block at (j,i); a^T lands at b, b^T at a. */
static inline void pln_tr4x4m_swap(double *a, double *b, ptrdiff_t row,
                                   int nr, int nc)
{
    const __m512i i0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    const __m512i i1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
    const __mmask8 la = (__mmask8)((1u << (2*nc)) - 1);
    const __mmask8 lb = (__mmask8)((1u << (2*nr)) - 1);
    pv a0 = pv_ld(la, a);
    pv a1 = pv_ld(nr > 1 ? la : (__mmask8)0, a + row);
    pv a2 = pv_ld(nr > 2 ? la : (__mmask8)0, a + 2*row);
    pv a3 = pv_ld(nr > 3 ? la : (__mmask8)0, a + 3*row);
    pv b0 = pv_ld(lb, b);
    pv b1 = pv_ld(nc > 1 ? lb : (__mmask8)0, b + row);
    pv b2 = pv_ld(nc > 2 ? lb : (__mmask8)0, b + 2*row);
    pv b3 = pv_ld(nc > 3 ? lb : (__mmask8)0, b + 3*row);
    pv pa = _mm512_permutex2var_pd(a0, i0, a1), qa = _mm512_permutex2var_pd(a2, i0, a3);
    pv ra = _mm512_permutex2var_pd(a0, i1, a1), sa = _mm512_permutex2var_pd(a2, i1, a3);
    pv pb = _mm512_permutex2var_pd(b0, i0, b1), qb = _mm512_permutex2var_pd(b2, i0, b3);
    pv rb = _mm512_permutex2var_pd(b0, i1, b1), sb = _mm512_permutex2var_pd(b2, i1, b3);
    pv_st(b, lb, _mm512_shuffle_f64x2(pa, qa, 0x44));
    if (nc > 1) pv_st(b + row,   lb, _mm512_shuffle_f64x2(pa, qa, 0xEE));
    if (nc > 2) pv_st(b + 2*row, lb, _mm512_shuffle_f64x2(ra, sa, 0x44));
    if (nc > 3) pv_st(b + 3*row, lb, _mm512_shuffle_f64x2(ra, sa, 0xEE));
    pv_st(a, la, _mm512_shuffle_f64x2(pb, qb, 0x44));
    if (nr > 1) pv_st(a + row,   la, _mm512_shuffle_f64x2(pb, qb, 0xEE));
    if (nr > 2) pv_st(a + 2*row, la, _mm512_shuffle_f64x2(rb, sb, 0x44));
    if (nr > 3) pv_st(a + 3*row, la, _mm512_shuffle_f64x2(rb, sb, 0xEE));
}

#endif

/* (y,z) plane, row stride L -> (z,y) plane, row stride pitch */
static void pln_transpose_in(const double *restrict s, double *restrict d,
                             int L, int pitch)
{
#if PLN_SIMD
    for (int y0 = 0; y0 < L; y0 += 4) {
        const int ny = L - y0 < 4 ? L - y0 : 4;
        for (int z0 = 0; z0 < L; z0 += 4) {
            const int nz = L - z0 < 4 ? L - z0 : 4;
            if (ny == 4 && nz == 4)
                pln_tr4x4(s + 2*((size_t)y0*L + z0), 2*(ptrdiff_t)L,
                          d + 2*((size_t)z0*pitch + y0), 2*(ptrdiff_t)pitch);
            else
                pln_tr4x4m(s + 2*((size_t)y0*L + z0), 2*(ptrdiff_t)L,
                           d + 2*((size_t)z0*pitch + y0), 2*(ptrdiff_t)pitch,
                           ny, nz);
        }
    }
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
    for (int z0 = 0; z0 < L; z0 += 4) {
        const int nz = L - z0 < 4 ? L - z0 : 4;
        for (int y0 = 0; y0 < L; y0 += 4) {
            const int ny = L - y0 < 4 ? L - y0 : 4;
            if (nz == 4 && ny == 4)
                pln_tr4x4(s + 2*((size_t)z0*pitch + y0), 2*(ptrdiff_t)pitch,
                          d + 2*((size_t)y0*L + z0), 2*(ptrdiff_t)L);
            else
                pln_tr4x4m(s + 2*((size_t)z0*pitch + y0), 2*(ptrdiff_t)pitch,
                           d + 2*((size_t)y0*L + z0), 2*(ptrdiff_t)L,
                           nz, ny);
        }
    }
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

#if PLN_SIMD
/* transpose-out with the graded map FUSED into the stores (gen_r6): the
 * chain step's final axis-2 values pass through here anyway, so add c and
 * map them while they sit in registers -- the separate per-plane map pass
 * (a full plane read + c read + plane write) disappears.  gen_layout r5's
 * register-exit fusion (their gl_map8 shape: pair-compressed |z|^2, one
 * exact vdivpd per 8 complex -- this exit is shuffle-bound, divider idle).
 * cf follows the OUTPUT (d, row stride L) geometry. */
static void pln_transpose_out_map(const double *restrict s, double *restrict d,
                                  const double *restrict cf, int L, int pitch)
{
    const __m512i i0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    const __m512i i1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
    for (int z0 = 0; z0 < L; z0 += 4) {
        const int nz = L - z0 < 4 ? L - z0 : 4;
        for (int y0 = 0; y0 < L; y0 += 4) {
            const int ny = L - y0 < 4 ? L - y0 : 4;
            const double *sp = s + 2*((size_t)z0*pitch + y0);
            double *dp = d + 2*((size_t)y0*L + z0);
            const double *cp = cf + 2*((size_t)y0*L + z0);
            if (nz == 4 && ny == 4) {
                pv r0 = _mm512_loadu_pd(sp);
                pv r1 = _mm512_loadu_pd(sp + 2*(ptrdiff_t)pitch);
                pv r2 = _mm512_loadu_pd(sp + 4*(ptrdiff_t)pitch);
                pv r3 = _mm512_loadu_pd(sp + 6*(ptrdiff_t)pitch);
                pv a = _mm512_permutex2var_pd(r0, i0, r1);
                pv b = _mm512_permutex2var_pd(r2, i0, r3);
                pv e = _mm512_permutex2var_pd(r0, i1, r1);
                pv f = _mm512_permutex2var_pd(r2, i1, r3);
                pv z0v = _mm512_add_pd(_mm512_shuffle_f64x2(a, b, 0x44),
                                       _mm512_loadu_pd(cp));
                pv z1v = _mm512_add_pd(_mm512_shuffle_f64x2(a, b, 0xEE),
                                       _mm512_loadu_pd(cp + 2*(ptrdiff_t)L));
                pv z2v = _mm512_add_pd(_mm512_shuffle_f64x2(e, f, 0x44),
                                       _mm512_loadu_pd(cp + 4*(ptrdiff_t)L));
                pv z3v = _mm512_add_pd(_mm512_shuffle_f64x2(e, f, 0xEE),
                                       _mm512_loadu_pd(cp + 6*(ptrdiff_t)L));
                pv sa, sb;
                pln_sdup(pln_mapdiv8(pln_mag_pair(z0v, z1v)), &sa, &sb);
                _mm512_storeu_pd(dp,                    _mm512_mul_pd(z0v, sa));
                _mm512_storeu_pd(dp + 2*(ptrdiff_t)L,   _mm512_mul_pd(z1v, sb));
                pln_sdup(pln_mapdiv8(pln_mag_pair(z2v, z3v)), &sa, &sb);
                _mm512_storeu_pd(dp + 4*(ptrdiff_t)L,   _mm512_mul_pd(z2v, sa));
                _mm512_storeu_pd(dp + 6*(ptrdiff_t)L,   _mm512_mul_pd(z3v, sb));
            } else {
                /* edge block: masked transpose first, then map the stored
                 * rows in place (rare -- only L%4 rows/columns; zero-fill
                 * lanes ride the tiny clamp and are never stored) */
                pln_tr4x4m(sp, 2*(ptrdiff_t)pitch, dp, 2*(ptrdiff_t)L, nz, ny);
                const __mmask8 sm = (__mmask8)((1u << (2*nz)) - 1);
                for (int k = 0; k < ny; ++k) {
                    double *o = dp + 2*(size_t)k*L;
                    pv zz = _mm512_add_pd(pv_ld(sm, o),
                                          pv_ld(sm, cp + 2*(size_t)k*L));
                    pv sa, sb;
                    pln_sdup(pln_mapdiv8(pln_mag_pair(zz, zz)), &sa, &sb);
                    (void)sb;
                    pv_st(o, sm, _mm512_mul_pd(zz, sa));
                }
            }
        }
    }
}

/* gen_r10: the same fused exit with the LOOPS SWAPPED (y0 outer, z0 inner).
 * The r6 exit walks z0 outer, which STREAMS the source -- but the source is
 * the P scratch plane, L2-resident by construction -- and STRIDES the two
 * streams that are actually DRAM-backed at L > 80 (the output state rows and
 * c, touched in 64 B chunks at 16L-byte steps; the r6 dead-end's +8..10%
 * loss at 100).  y0-outer visits d and c SEQUENTIALLY (each line exactly
 * once, prefetcher-friendly) and strides only P (cheap L2 hits).  Per-block
 * arithmetic is identical to pln_transpose_out_map -- visit order only, so
 * outputs are bit-identical to it.  Engaged where state+c dwarf L2: L > 80
 * (the fusemap=2 exit); at 12 < L <= 80 the r6 z0-outer exit ships unchanged
 * (its win band was measured in that order; everything is cache-resident
 * there and the A/B at 50 read a wash -- see the r10 record). */
static void pln_transpose_out_map_yo(const double *restrict s,
                                     double *restrict d,
                                     const double *restrict cf,
                                     int L, int pitch)
{
    const __m512i i0 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    const __m512i i1 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
    for (int y0 = 0; y0 < L; y0 += 4) {
        const int ny = L - y0 < 4 ? L - y0 : 4;
        for (int z0 = 0; z0 < L; z0 += 4) {
            const int nz = L - z0 < 4 ? L - z0 : 4;
            const double *sp = s + 2*((size_t)z0*pitch + y0);
            double *dp = d + 2*((size_t)y0*L + z0);
            const double *cp = cf + 2*((size_t)y0*L + z0);
            if (nz == 4 && ny == 4) {
                pv r0 = _mm512_loadu_pd(sp);
                pv r1 = _mm512_loadu_pd(sp + 2*(ptrdiff_t)pitch);
                pv r2 = _mm512_loadu_pd(sp + 4*(ptrdiff_t)pitch);
                pv r3 = _mm512_loadu_pd(sp + 6*(ptrdiff_t)pitch);
                pv a = _mm512_permutex2var_pd(r0, i0, r1);
                pv b = _mm512_permutex2var_pd(r2, i0, r3);
                pv e = _mm512_permutex2var_pd(r0, i1, r1);
                pv f = _mm512_permutex2var_pd(r2, i1, r3);
                pv z0v = _mm512_add_pd(_mm512_shuffle_f64x2(a, b, 0x44),
                                       _mm512_loadu_pd(cp));
                pv z1v = _mm512_add_pd(_mm512_shuffle_f64x2(a, b, 0xEE),
                                       _mm512_loadu_pd(cp + 2*(ptrdiff_t)L));
                pv z2v = _mm512_add_pd(_mm512_shuffle_f64x2(e, f, 0x44),
                                       _mm512_loadu_pd(cp + 4*(ptrdiff_t)L));
                pv z3v = _mm512_add_pd(_mm512_shuffle_f64x2(e, f, 0xEE),
                                       _mm512_loadu_pd(cp + 6*(ptrdiff_t)L));
                pv sa, sb;
                pln_sdup(pln_mapdiv8(pln_mag_pair(z0v, z1v)), &sa, &sb);
                _mm512_storeu_pd(dp,                    _mm512_mul_pd(z0v, sa));
                _mm512_storeu_pd(dp + 2*(ptrdiff_t)L,   _mm512_mul_pd(z1v, sb));
                pln_sdup(pln_mapdiv8(pln_mag_pair(z2v, z3v)), &sa, &sb);
                _mm512_storeu_pd(dp + 4*(ptrdiff_t)L,   _mm512_mul_pd(z2v, sa));
                _mm512_storeu_pd(dp + 6*(ptrdiff_t)L,   _mm512_mul_pd(z3v, sb));
            } else {
                pln_tr4x4m(sp, 2*(ptrdiff_t)pitch, dp, 2*(ptrdiff_t)L, nz, ny);
                const __mmask8 sm = (__mmask8)((1u << (2*nz)) - 1);
                for (int k = 0; k < ny; ++k) {
                    double *o = dp + 2*(size_t)k*L;
                    pv zz = _mm512_add_pd(pv_ld(sm, o),
                                          pv_ld(sm, cp + 2*(size_t)k*L));
                    pv sa, sb;
                    pln_sdup(pln_mapdiv8(pln_mag_pair(zz, zz)), &sa, &sb);
                    (void)sb;
                    pv_st(o, sm, _mm512_mul_pd(zz, sa));
                }
            }
        }
    }
}
#endif

/* axes 1+2 of one x-plane, in place, plane cache-resident.  cf != NULL
 * fuses the graded map (z = out + c; z/(1+|z|)) into the transpose-out
 * stores -- chain-step call sites only. */
static void pln_p3d_plane(pln_p3d *p, double *pl, const double *cf)
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
#if PLN_SIMD
    if (cf) {
        if (p->fusemap == 2)                  /* L > 80: stream d and c (DRAM),
                                                 stride only the L2-hot P */
            pln_transpose_out_map_yo(p->P, pl, cf, L, pitch);
        else
            pln_transpose_out_map(p->P, pl, cf, L, pitch);
        return;
    }
#else
    (void)cf;
#endif
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
        pln_p3d_plane(p, dst + 2 * (size_t)xx * L2, NULL);
}

/* ------------------------------------------------------------------ */
/* the graded map  z <- z / (1 + |z|)  with  z = FFT(state) + c,        */
/* applied to one contiguous span while it is still cache-hot.          */
/* rsqrt14/rcp14 + 2 Newton steps each (~1e-16 rel, campaign standard); */
/* zmm vsqrtpd/vdivpd are unpipelined and ~4x slower here.              */
/* ------------------------------------------------------------------ */
/* gen_r11: the body takes a separate dst so the final chain step can exit
 * straight into the driver's final_out (zero-copy re-home exit, adopted from
 * gen_layout gen_r11).  st/dst deliberately NOT restrict-qualified against
 * each other: the in-place instantiation passes st == dst. */
static inline __attribute__((always_inline)) void pln_map_span_i(
        double *st, const double *restrict cf, double *dst, size_t npts)
{
    size_t i = 0, n = 2 * npts;
#if PLN_SIMD
    /* PAIR-PACKED ladder (gen_r6, adopted from gen_pfa_large r5's
     * map_step_pair): the per-vector form below runs the whole NR ladder
     * on pair-DUPLICATED |z|^2 -- half its lanes compute nothing new.
     * Two vectors' 8 distinct |z|^2 share ONE ladder here; identical
     * per-element operations, so the output is bit-identical. */
    {
        const __m512d half = _mm512_set1_pd(0.5), th = _mm512_set1_pd(1.5);
        const __m512d one = _mm512_set1_pd(1.0), two = _mm512_set1_pd(2.0);
        const __m512d tiny = _mm512_set1_pd(1e-300);
        for (; i + 16 <= n; i += 16) {
            __m512d v0 = _mm512_add_pd(_mm512_loadu_pd(st + i), _mm512_loadu_pd(cf + i));
            __m512d v1 = _mm512_add_pd(_mm512_loadu_pd(st + i + 8), _mm512_loadu_pd(cf + i + 8));
            __m512d h = pln_mag_pair(v0, v1);
            h = _mm512_max_pd(h, tiny);
            __m512d r = _mm512_rsqrt14_pd(h);
            r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(_mm512_mul_pd(half, h), r), r, th));
            r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(_mm512_mul_pd(half, h), r), r, th));
            __m512d d = _mm512_fmadd_pd(h, r, one);
            __m512d s = _mm512_rcp14_pd(d);
            s = _mm512_mul_pd(s, _mm512_fnmadd_pd(d, s, two));
            s = _mm512_mul_pd(s, _mm512_fnmadd_pd(d, s, two));
            __m512d sa, sb;
            pln_sdup(s, &sa, &sb);
            _mm512_storeu_pd(dst + i, _mm512_mul_pd(v0, sa));
            _mm512_storeu_pd(dst + i + 8, _mm512_mul_pd(v1, sb));
        }
    }
#endif
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
        _mm512_storeu_pd(dst + i, _mm512_mul_pd(v, s));
    }
#endif
    for (; i < n; i += 2) {                                       /* exact tail */
        double re = st[i] + cf[i], im = st[i+1] + cf[i+1];
        double sc = 1.0 / (1.0 + sqrt(re*re + im*im));
        dst[i] = re * sc; dst[i+1] = im * sc;
    }
}

static void pln_map_span(double *restrict st, const double *restrict cf, size_t npts)
{
    size_t i = 0, n = 2 * npts;
#if PLN_SIMD
    /* PAIR-PACKED ladder (gen_r6, adopted from gen_pfa_large r5's
     * map_step_pair): the per-vector form below runs the whole NR ladder
     * on pair-DUPLICATED |z|^2 -- half its lanes compute nothing new.
     * Two vectors' 8 distinct |z|^2 share ONE ladder here; identical
     * per-element operations, so the output is bit-identical. */
    {
        const __m512d half = _mm512_set1_pd(0.5), th = _mm512_set1_pd(1.5);
        const __m512d one = _mm512_set1_pd(1.0), two = _mm512_set1_pd(2.0);
        const __m512d tiny = _mm512_set1_pd(1e-300);
        for (; i + 16 <= n; i += 16) {
            __m512d v0 = _mm512_add_pd(_mm512_loadu_pd(st + i), _mm512_loadu_pd(cf + i));
            __m512d v1 = _mm512_add_pd(_mm512_loadu_pd(st + i + 8), _mm512_loadu_pd(cf + i + 8));
            __m512d h = pln_mag_pair(v0, v1);
            h = _mm512_max_pd(h, tiny);
            __m512d r = _mm512_rsqrt14_pd(h);
            r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(_mm512_mul_pd(half, h), r), r, th));
            r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(_mm512_mul_pd(half, h), r), r, th));
            __m512d d = _mm512_fmadd_pd(h, r, one);
            __m512d s = _mm512_rcp14_pd(d);
            s = _mm512_mul_pd(s, _mm512_fnmadd_pd(d, s, two));
            s = _mm512_mul_pd(s, _mm512_fnmadd_pd(d, s, two));
            __m512d sa, sb;
            pln_sdup(s, &sa, &sb);
            _mm512_storeu_pd(st + i, _mm512_mul_pd(v0, sa));
            _mm512_storeu_pd(st + i + 8, _mm512_mul_pd(v1, sb));
        }
    }
#endif
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
        const double *cf = cfield + 2 * (size_t)xx * L2;
        if (p->fusemap) {
            pln_p3d_plane(p, pl, cf);        /* map fused in the exit stores */
        } else {
            pln_p3d_plane(p, pl, NULL);
            pln_map_span(pl, cf, L2);        /* plane still hot */
#if defined(__CLFLUSHOPT__)
            if (p->cflush) { /* gen_r9 custody, default off (r10 refuted) */
                /* c-line custody (gen_r9, from gen_pfa_large's c-bypass):
                 * this plane's c lines are L1-hot right now and will not be
                 * read again until the NEXT step; evicting them here keeps
                 * the read-once c stream from displacing the m-step-resident
                 * state in L3.  Flush is a cache op, not a data op: any
                 * alignment slop only re-fetches a line, never corrupts. */
                for (size_t q = 0; q < 2 * L2; q += 8)
                    _mm_clflushopt((void *)(cf + q));
            }
#endif
        }
    }
}


/* ------------------------------------------------------------------ */
/* SPLIT-GROUP 3-D engine (gen_r5): 8 volumes of a batch, packed        */
/* site-major split-complex (site = {8 re, 8 im}, lane v = volume v).   */
/* Levels:                                                              */
/*   1  fused root      n = r*m <= 25, both hard: one register-resident */
/*      codelet call per pencil, all three axes, in place.              */
/*   2  two-pass CT     n = r*m, r hard, child = hard leaf OR fused CT  */
/*      (covers 27 = c3(c3(d3)), 32 = c4(d8), 40 = c5(d8), ...):        */
/*      pass 1 child DFT_m src -> dst (undoes the decimation), pass 2   */
/*      twiddle-fused leaf_r in place on dst -- each axis ping-pongs    */
/*      the group buffer, 3 swaps per step.                             */
/*   3  folded dense root (odd n >= 11, e.g. d31): single pass per      */
/*      axis, in place, 4 KB pencil staging.                            */
/*   4  staged two-pass CT (gen_r6; same trees as level 2, n > 25):     */
/*      ONE in-place volume sweep per axis through an L1-resident       */
/*      stage of P pencils -- no ping-pong buffer, half the volume      */
/*      sweeps of level 2.                                              */
/* No transposes, no masks, no shuffles exist in this mode; the map     */
/* runs on the whole group with the halved split ladder.  Only raced    */
/* (and only buildable) when batch >= 8; B % 8 volumes take the         */
/* per-volume path.                                                     */
/* ------------------------------------------------------------------ */
#if PLN_SIMD
typedef struct pln_s8 {
    int L, lev;
    int r, m;                    /* root split (lev 1/2); lev 3: r = n */
    int cr, cm;                  /* lev 2 fused child, 0 if child is a leaf */
    int gt;                      /* gen_r7: lev-1 root is a fused GT (no tw) */
    int cgt;                     /* gen_r7: lev-2/4 child is a fused GT */
    int fusemap;                 /* map fused into the axis-2 stores: WINS
                                    once the group (state+c = 256*L^3 B)
                                    leaves L2 (-4% at 15, -8% at 25, -9% at
                                    31, same-core pairs) and LOSES when
                                    L2-resident (+5% at 10: the ladder's ~8
                                    temps push the fused codelet into
                                    spills); identical numerics either way */
    int p4;                      /* @s4: pencils per L1 stage block */
    double *tw, *tw2, *fmat, *stage;
    double *stage4;              /* @s4: n rows x (16*p4 + 8) doubles */
} pln_s8;

static double *pln_s8_tw(int n, int r, int m)   /* k2-major, as the executor's */
{
    double *tw = malloc(2 * (size_t)(m - 1) * (r - 1) * sizeof *tw);
    if (tw)
        for (int k2 = 1; k2 < m; ++k2)
            for (int j1 = 1; j1 < r; ++j1)
                pln_omega((long)j1 * k2, n,
                          &tw[2*((size_t)(k2-1)*(r-1) + (j1-1))],
                          &tw[2*((size_t)(k2-1)*(r-1) + (j1-1)) + 1]);
    return tw;
}

/* lev < 0 lets the tree decide; lev in {1,2,3} demands that level (race
 * arms).  NULL when the tree is not expressible at that level. */
static __attribute__((unused)) pln_s8 *pln_s8_build(int L, const pln_node *t, int lev)
{
    pln_s8 *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    s->L = L;
    s->fusemap = ((size_t)L * L * L > 1728);     /* same-core pairs: 10 loses
                                                    (+5%), 12 wash, 15 wins
                                                    (-4%), 25 (-8%), 31 (-9%):
                                                    fuse strictly above L=12 */
#ifdef PLN_S8_NOFUSEMAP
    s->fusemap = 0;
#endif
#ifdef PLN_S8_FUSEMAP
    s->fusemap = 1;                              /* A/B override */
#endif
    /* gen_r8, tried and REFUTED, kept as a record: running @s1/@s3 axes 0+1
     * fused per z-slab (the @s2/@s4 slab merge applied to levels 1 and 3;
     * bit-identical pass reorder, would have deleted one of the step's
     * three volume sweeps) LOSES 3/3 held-lease pairs at every size where
     * it engages -- 25: 48.4-49.4 vs 42.9-44.4 (+10-12%), 31: 156.2-156.8
     * vs 144.9-146.8 (+7%), 20/15 slight loss.  Mechanism: the baseline
     * axis-0 pass walks pencils CONTIGUOUSLY (pcs=16, L parallel sequential
     * streams -- perfect L2-streamer food), and the slab replaces that with
     * 128*L-byte-strided pencil walks the prefetcher cannot track (ICX
     * stride limit ~2 KB).  @s2's slab win at 27 does not transfer because
     * @s2 had no contiguous-stream pass to lose.  Do not re-derive. */
    if (t->kind == PLN_LEAF && !pln_leaf_hard(t->n) && (t->n & 1) && t->n >= 11
        && (lev < 0 || lev == 3)) {
        s->lev = 3; s->r = t->n;
        s->fmat = pln_fold_matrix(t->n);
        if (!s->fmat || posix_memalign((void **)&s->stage, 64,
                                       16 * (size_t)t->n * sizeof(double)))
            { free(s->fmat); free(s); return NULL; }
        return s;
    }
    if (t->kind == PLN_PFA && pln_gt_fused(t->s1, t->s2) &&
        (lev < 0 || lev == 1)) {
        /* gen_r7: fused-GT root -- one register-resident twiddle-free
         * codelet call per pencil, all three axes, in place (@s1) */
        s->lev = 1; s->gt = 1;
        s->r = t->r; s->m = t->n / t->r;
        return s;
    }
    if (t->kind == PLN_CT && pln_leaf_hard(t->r) && t->s1) {
        const int r = t->r, m = t->n / t->r;
        const pln_node *c = t->s1;
        int child_leaf = (c->kind == PLN_LEAF && pln_leaf_hard(c->n));
        int child_fused = (c->kind == PLN_CT && pln_leaf_hard(c->r) && c->s1 &&
                           c->s1->kind == PLN_LEAF && pln_leaf_hard(c->s1->n) &&
                           c->n <= PLN_FUSEMAX);
        int child_gt = (c->kind == PLN_PFA && pln_gt_fused(c->s1, c->s2));
        if (child_leaf && t->n <= PLN_FUSEMAX && (lev < 0 || lev == 1))
            s->lev = 1;
        else if ((child_leaf || child_fused || child_gt) && lev == 4 &&
                 !(child_leaf && t->n <= PLN_FUSEMAX))
            s->lev = 4;              /* fused-root sizes stay lev 1 territory */
        else if ((child_leaf || child_fused || child_gt) && (lev < 0 || lev == 2))
            s->lev = 2;
        else { free(s); return NULL; }
        s->r = r; s->m = m;
        s->tw = pln_s8_tw(t->n, r, m);
        if (!s->tw) { free(s); return NULL; }
        if (s->lev >= 2 && child_gt) {
            s->cr = c->r; s->cm = c->n / c->r;    /* twiddle-free child */
            s->cgt = 1;
        } else if (s->lev >= 2 && child_fused) {
            s->cr = c->r; s->cm = c->n / c->r;
            s->tw2 = pln_s8_tw(c->n, s->cr, s->cm);
            if (!s->tw2) { free(s->tw); free(s); return NULL; }
        }
        if (s->lev == 4) {
            s->p4 = t->n <= 40 ? 4 : t->n <= 64 ? 2 : 1;
            if (posix_memalign((void **)&s->stage4, 64,
                               (size_t)t->n * (16 * (size_t)s->p4 + 8) *
                               sizeof(double)))
                { free(s->tw); free(s->tw2); free(s); return NULL; }
        }
        return s;
    }
    free(s);
    return NULL;
}

static __attribute__((unused)) void pln_s8_free(pln_s8 *s)
{
    if (!s) return;
    free(s->tw); free(s->tw2); free(s->fmat); free(s->stage);
    free(s->stage4); free(s);
}

/* one two-pass CT over one set of pencils (lev 2), src -> dst; cb != NULL
 * fuses the map into pass 2's stores (final-axis call sites only) */
static void pln_s8_ct2_set(const pln_s8 *s, const double *src, double *dst,
                           ptrdiff_t rs, ptrdiff_t cs, int np, const double *cb)
{
    const int r = s->r, m = s->m;
    for (int j1 = 0; j1 < r; ++j1) {
        const double *in = src + (ptrdiff_t)j1 * rs;
        double *out = dst + (ptrdiff_t)j1 * m * rs;
        if (s->cgt)     pln_gtw_run(s->cr, s->cm, in, rs * r, out, rs, cs, cs,
                                    np, NULL);
        else if (s->cr) pln_fusedw_run(s->cr, s->cm, in, rs * r, out, rs, cs, cs,
                                       np, s->tw2, NULL);
        else            pw_leaf_run(m, in, rs * r, out, rs, cs, cs, np, NULL, NULL);
    }
    for (int k2 = 0; k2 < m; ++k2)
        pw_leaf_run(r, dst + (ptrdiff_t)k2 * rs, (ptrdiff_t)m * rs,
                    dst + (ptrdiff_t)k2 * rs, (ptrdiff_t)m * rs, cs, cs, np,
                    k2 ? s->tw + 2 * (size_t)(k2 - 1) * (r - 1) : NULL,
                    cb ? cb + (ptrdiff_t)k2 * rs : NULL);
}

/* @s4 (gen_r6): one set of np pencils through the two-pass CT as a SINGLE
 * in-place volume sweep.  Blocks of P pencils stage through an L1-resident
 * buffer: pass 1 (child DFT_m, undoing the decimation) reads the volume
 * rows ONCE into the stage, pass 2 (twiddle-fused leaf_r) reads the stage
 * and writes the volume rows ONCE (map fused at the final axis).  vs @s2:
 * no ping-pong buffer, and each axis touches the volume once instead of
 * twice.  Stage rows are padded to an odd number of cache lines
 * (srs = 16P + 8) so pass 2's m*srs row stride never lands 4K-uniform. */
static void pln_s8_ct1_set(const pln_s8 *s, double *v, ptrdiff_t rs,
                           ptrdiff_t pcs, int np, const double *cb)
{
    const int r = s->r, m = s->m, P = s->p4;
    const ptrdiff_t srs = (ptrdiff_t)16 * P + 8;
    double *st = s->stage4;
    for (int p0 = 0; p0 < np; p0 += P) {
        const int pp = np - p0 < P ? np - p0 : P;
        double *base = v + (ptrdiff_t)p0 * pcs;
        const double *cbb = cb ? cb + (ptrdiff_t)p0 * pcs : NULL;
        for (int j1 = 0; j1 < r; ++j1) {
            if (s->cgt)     pln_gtw_run(s->cr, s->cm, base + (ptrdiff_t)j1 * rs,
                                        rs * r, st + (ptrdiff_t)j1 * m * srs, srs,
                                        pcs, 16, pp, NULL);
            else if (s->cr) pln_fusedw_run(s->cr, s->cm, base + (ptrdiff_t)j1 * rs,
                                           rs * r, st + (ptrdiff_t)j1 * m * srs, srs,
                                           pcs, 16, pp, s->tw2, NULL);
            else            pw_leaf_run(m, base + (ptrdiff_t)j1 * rs, rs * r,
                                        st + (ptrdiff_t)j1 * m * srs, srs,
                                        pcs, 16, pp, NULL, NULL);
        }
        for (int k2 = 0; k2 < m; ++k2)
            pw_leaf_run(r, st + (ptrdiff_t)k2 * srs, (ptrdiff_t)m * srs,
                        base + (ptrdiff_t)k2 * rs, (ptrdiff_t)m * rs,
                        16, pcs, pp,
                        k2 ? s->tw + 2 * (size_t)(k2 - 1) * (r - 1) : NULL,
                        cbb ? cbb + (ptrdiff_t)k2 * rs : NULL);
    }
}

/* one full graded chain step (FFT3 + c + map) on a split group.  *pa is
 * the state buffer, *pb the alternate (lev 2 only); pointers swap so the
 * caller always finds the state in *pa. */
static __attribute__((unused)) void pln_s8_step(const pln_s8 *s,
        double **pa, double **pb, const double *Cg)
{
    const int L = s->L;
    const size_t L2 = (size_t)L * L;
    double *S = *pa;
    const double *Cg_fused = s->fusemap ? Cg : NULL;
    if (s->lev == 1 && s->gt) {
        /* fused-GT root (gen_r7): same axis order as the fused-CT @s1 */
        pln_gtw_run(s->r, s->m, S, (ptrdiff_t)16 * L2, S, (ptrdiff_t)16 * L2,
                    16, 16, (int)L2, NULL);
        for (int x = 0; x < L; ++x) {
            double *pl = S + 16 * (size_t)x * L2;
            const double *cl = Cg_fused ? Cg_fused + 16 * (size_t)x * L2 : NULL;
            pln_gtw_run(s->r, s->m, pl, (ptrdiff_t)16 * L, pl, (ptrdiff_t)16 * L,
                        16, 16, L, NULL);
            pln_gtw_run(s->r, s->m, pl, 16, pl, 16,
                        (ptrdiff_t)16 * L, (ptrdiff_t)16 * L, L, cl);
        }
    } else if (s->lev == 1) {
        pln_fusedw_run(s->r, s->m, S, (ptrdiff_t)16 * L2, S, (ptrdiff_t)16 * L2,
                       16, 16, (int)L2, s->tw, NULL);
        for (int x = 0; x < L; ++x) {
            double *pl = S + 16 * (size_t)x * L2;
            const double *cl = Cg_fused ? Cg_fused + 16 * (size_t)x * L2 : NULL;
            pln_fusedw_run(s->r, s->m, pl, (ptrdiff_t)16 * L, pl, (ptrdiff_t)16 * L,
                           16, 16, L, s->tw, NULL);
            pln_fusedw_run(s->r, s->m, pl, 16, pl, 16,
                           (ptrdiff_t)16 * L, (ptrdiff_t)16 * L, L,
                           s->tw, cl);
        }
    } else if (s->lev == 4) {
        /* single in-place sweep per axis through the L1 stage; axes 0+1
         * fused per z-slab (the @s2 slab lesson), axis 2 per x-plane with
         * the map in its stores */
        for (int z = 0; z < L; ++z) {
            pln_s8_ct1_set(s, S + 16 * (ptrdiff_t)z, (ptrdiff_t)16 * L2,
                           (ptrdiff_t)16 * L, L, NULL);
            pln_s8_ct1_set(s, S + 16 * (ptrdiff_t)z, (ptrdiff_t)16 * L,
                           (ptrdiff_t)16 * L2, L, NULL);
        }
        for (int x = 0; x < L; ++x)
            pln_s8_ct1_set(s, S + 16 * (size_t)x * L2, 16, (ptrdiff_t)16 * L, L,
                           Cg_fused ? Cg_fused + 16 * (size_t)x * L2 : NULL);
    } else if (s->lev == 3) {
        pln_foldw(s->fmat, s->r, S, (ptrdiff_t)16 * L2, 16, (int)L2, s->stage, NULL);
        for (int x = 0; x < L; ++x) {
            double *pl = S + 16 * (size_t)x * L2;
            const double *cl = Cg_fused ? Cg_fused + 16 * (size_t)x * L2 : NULL;
            pln_foldw(s->fmat, s->r, pl, (ptrdiff_t)16 * L, 16, L, s->stage, NULL);
            pln_foldw(s->fmat, s->r, pl, 16, (ptrdiff_t)16 * L, L, s->stage, cl);
        }
    } else {
        double *a = *pa, *b = *pb;
        /* two fused volume sweeps instead of three (the lev-2 step was
         * L3-traffic-bound at 27): PASS A runs axes 0 and 1 back to back
         * per z-SLAB (both axes live entirely inside the fixed-z slab,
         * L^2 x 128 B -- L2-hot between the two); PASS B runs axis 2 (+
         * fused map) per x-plane.  Slabs and planes are independent and
         * per-pencil arithmetic is unchanged, so outputs are bit-identical
         * to the separated order. */
        for (int z = 0; z < L; ++z) {
            pln_s8_ct2_set(s, a + 16 * (size_t)z, b + 16 * (size_t)z,
                           (ptrdiff_t)16 * L2, (ptrdiff_t)16 * L, L, NULL);
            pln_s8_ct2_set(s, b + 16 * (size_t)z, a + 16 * (size_t)z,
                           (ptrdiff_t)16 * L, (ptrdiff_t)16 * L2, L, NULL);
        }
        for (int x = 0; x < L; ++x)
            pln_s8_ct2_set(s, a + 16 * (size_t)x * L2, b + 16 * (size_t)x * L2,
                           16, (ptrdiff_t)16 * L, L,
                           Cg_fused ? Cg_fused + 16 * (size_t)x * L2 : NULL);
        *pa = b; *pb = a; S = b;
    }
    if (!Cg_fused) pln_map_span_w(S, Cg, L2 * L);
    (void)S;
}

/* volume-major interleaved (driver layout) <-> site-major split group.
 * Once per chain each way, amortized over the m steps; scalar is fine. */
static __attribute__((unused)) void pln_s8_pack(const double *src, size_t vol, double *G)
{
    for (int v = 0; v < 8; ++v) {
        const double *p = src + 2 * vol * (size_t)v;
        for (size_t i = 0; i < vol; ++i) {
            G[16*i + v]     = p[2*i];
            G[16*i + 8 + v] = p[2*i + 1];
        }
    }
}

static __attribute__((unused)) void pln_s8_unpack(const double *G, size_t vol, double *dst)
{
    for (int v = 0; v < 8; ++v) {
        double *p = dst + 2 * vol * (size_t)v;
        for (size_t i = 0; i < vol; ++i) {
            p[2*i]     = G[16*i + v];
            p[2*i + 1] = G[16*i + 8 + v];
        }
    }
}
#endif /* PLN_SIMD */

/* ================================================================== */
/* standalone fft3d entry (omitted when adopted as a library)          */
/* ================================================================== */
/* ------------------------------------------------------------------ */
/* gen_r11 additions live at the END of the library TU so no          */
/* pre-existing function shifts in .text (emission-order layout       */
/* drift measured +3-5% at L=32 when these sat mid-file).             */
/* ------------------------------------------------------------------ */
#if PLN_SIMD
/* gen_r11: in-place square transpose of one L x L complex plane (row stride
 * exactly L -- the plane lives in the volume, no padded scratch).  Each block
 * pair is touched once: same per-element shuffle cost as pln_transpose_in,
 * and the classic path's SECOND transpose pass (transpose-out: a full plane
 * read + write per plane per step) is the pass this exists to delete. */
static __attribute__((unused)) void pln_transpose_ip(double *pl, int L)
{
    const ptrdiff_t row = 2 * (ptrdiff_t)L;
    const int Lf = L & ~3, e = L & 3;
    int i0;
    for (i0 = 0; i0 < Lf; i0 += 4) {
        double *dg = pl + (size_t)i0 * row + 2*i0;
        pln_tr4x4(dg, row, dg, row);                     /* diagonal, in place */
        for (int j0 = i0 + 4; j0 < Lf; j0 += 4)
            pln_tr4x4_swap(pl + (size_t)i0 * row + 2*j0,
                           pl + (size_t)j0 * row + 2*i0, row);
        if (e)
            pln_tr4x4m_swap(pl + (size_t)i0 * row + 2*Lf,
                            pl + (size_t)Lf * row + 2*i0, row, 4, e);
    }
    if (e) {
        double *dg = pl + (size_t)Lf * row + 2*Lf;       /* trailing diagonal */
        pln_tr4x4m(dg, row, dg, row, e, e);
    }
}
#endif

static __attribute__((unused)) void pln_map_span_to(double *restrict dst,
        double *st, const double *restrict cf, size_t npts)
{
    pln_map_span_i(st, cf, dst, npts);
}


#if PLN_SIMD
/* ------------------------------------------------------------------ */
/* gen_r11: ALTERNATING-LAYOUT one-transpose chain step (L > 80).       */
/* The classic step's plane pass makes FIVE plane round trips (axis 1   */
/* in place, transpose-in to P, axis 2 in P, transpose-out back, map).  */
/* Letting the volume layout alternate (x,y,z) <-> (x,z,y) between      */
/* chain steps needs only ONE in-place square transpose per plane:      */
/* in-plane axis A (rows stride L), pln_transpose_ip, in-plane axis B   */
/* (same strides), map against the EXIT-layout c copy.  The transpose-  */
/* back pass -- a full 2*16*L^2-byte read+write plus 8 shuffles per 16  */
/* complex, per plane, per step -- does not exist here, and the P       */
/* scratch plane leaves the chain's cache footprint.  FFT over y and z  */
/* commute exactly as an algorithm; the summation ORDER changes on      */
/* swapped-entry steps, so chains are NOT bit-identical to the classic  */
/* path (same kernels, same per-step error class -- gated at create()). */
/* Layout bookkeeping, c-copy selection and the final-step canonical    */
/* exit live in the chain driver (pln_chain_pv).                        */
/* ------------------------------------------------------------------ */
static __attribute__((unused)) void pln_p3d_step_alt(pln_p3d *p, const double *src, double *st,
                             const double *cf, double *mdst)
{
    const int L = p->L, T = p->xp->tile;
    const size_t L2 = (size_t)L * L;
    pln_x *root = p->xp->root;
    double *arena = p->xp->arena;
    const ptrdiff_t bs = 2 * (ptrdiff_t)T;

    for (size_t off = 0; off < L2; off += T) {           /* axis 0: src -> st */
        int w = (int)(L2 - off < (size_t)T ? L2 - off : (size_t)T);
        pln_xexec(root, src + 2*off, 2*(ptrdiff_t)L2, st + 2*off,
                  2*(ptrdiff_t)L2, w, arena, bs);
    }
    for (int xx = 0; xx < L; ++xx) {
        double *pl = st + 2 * (size_t)xx * L2;
        for (int off = 0; off < L; off += T) {           /* in-plane axis A */
            int w = L - off < T ? L - off : T;
            pln_xexec(root, pl + 2*off, 2*(ptrdiff_t)L, pl + 2*off,
                      2*(ptrdiff_t)L, w, arena, bs);
        }
        pln_transpose_ip(pl, L);
        for (int off = 0; off < L; off += T) {           /* in-plane axis B */
            int w = L - off < T ? L - off : T;
            pln_xexec(root, pl + 2*off, 2*(ptrdiff_t)L, pl + 2*off,
                      2*(ptrdiff_t)L, w, arena, bs);
        }
        double *md = mdst ? mdst + 2 * (size_t)xx * L2 : pl;
        pln_map_span_to(md, pl, cf + 2 * (size_t)xx * L2, L2);   /* md == pl
                                    is legal: the body carries no st/dst
                                    restrict (zero-copy exit when mdst set) */
    }
}

/* canonical-entry final step (odd m): the classic two-transpose plane pass
 * (stays canonical), with the map exiting to mdst.  Also serves m == 1. */
static __attribute__((unused)) void pln_p3d_step_fin(pln_p3d *p, const double *src, double *st,
                             const double *cf, double *mdst)
{
    const int L = p->L, T = p->xp->tile;
    const size_t L2 = (size_t)L * L;
    pln_x *root = p->xp->root;
    double *arena = p->xp->arena;
    const ptrdiff_t bs = 2 * (ptrdiff_t)T;

    for (size_t off = 0; off < L2; off += T) {
        int w = (int)(L2 - off < (size_t)T ? L2 - off : (size_t)T);
        pln_xexec(root, src + 2*off, 2*(ptrdiff_t)L2, st + 2*off,
                  2*(ptrdiff_t)L2, w, arena, bs);
    }
    for (int xx = 0; xx < L; ++xx) {
        double *pl = st + 2 * (size_t)xx * L2;
        pln_p3d_plane(p, pl, NULL);
        pln_map_span_to(mdst + 2 * (size_t)xx * L2, pl,
                        cf + 2 * (size_t)xx * L2, L2);
    }
}
#endif /* PLN_SIMD */

#ifndef GEN_PLANNER_LIB

#include <time.h>
#include <stdint.h>
#if defined(__linux__)
#include <sys/mman.h>
#endif
#include "../fft3d_api.h"

/* gen_r11, adopted from gen_layout gen_r11: this host runs THP=madvise, so
 * driver posix_memalign buffers get ZERO huge pages -- ~12K page walks per
 * chain step at L=100 (their counters: dtlb walk_active 199M cycles/case).
 * Chain-hot volumes therefore live in MADV_HUGEPAGE arenas; entry (step 1
 * reads x0 directly) and exit (last step's map writes final_out directly)
 * stay zero-copy.  Returns a 2 MiB-aligned pointer; *base/*bsz for munmap. */
static void *pln_huge_alloc(size_t bytes, void **base, size_t *bsz)
{
#if defined(__linux__) && defined(MAP_ANONYMOUS)
    const size_t H = (size_t)2 << 20;
    size_t rb = ((bytes + H - 1) & ~(H - 1)) + H;
    void *m = mmap(NULL, rb, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return NULL;
#ifdef MADV_HUGEPAGE
    madvise(m, rb, MADV_HUGEPAGE);
#endif
    *base = m; *bsz = rb;
    return (void *)(((uintptr_t)m + H - 1) & ~(uintptr_t)(H - 1));
#else
    (void)bytes; (void)base; (void)bsz;
    return NULL;                 /* no THP: the alt path simply stays off */
#endif
}

static void pln_huge_free(void *base, size_t bsz)
{
#if defined(__linux__) && defined(MAP_ANONYMOUS)
    if (base) munmap(base, bsz);
#else
    (void)base; (void)bsz;
#endif
}

/* gen_race adoption (their gen_r2 string-wisdom hook, written for this entry):
 * persist the raced tree name per (host, L) so a second process builds the
 * same plan from a file read -- repeatability across the driver's two
 * processes becomes structural (different trees round differently), and warm
 * create() skips the race.  GEN_RACE_NO_WISDOM/REFRESH pins pass through. */
#define GEN_RACE_LIB_ONLY
#include "gen_race.c"

struct fft3d_plan {
    int L, batch;
    pln_p3d *p3;                 /* per-volume engine: execute(), remainder, fallback */
    int chain_ok;                /* create()-time gate passed: own the chain */
#if PLN_SIMD
    pln_s8 *s8;                  /* split-group engine (gen_r5), NULL unless raced
                                    in at batch >= 8 AND its own chain gate passed */
    double *G, *G2, *Cg;         /* group state / lev-2 alternate / packed c */
    int alt;                     /* gen_r11: alternating-layout one-transpose
                                    chain gated in (L > 80, own 2-step gate) */
    double *stv, *cw;            /* THP-backed chain state volume + (z,y)-
                                    transposed c copy for swapped-exit maps */
    void *hb0, *hb1; size_t hs0, hs1;
#endif
    double _Complex *scratch;    /* one volume, only for the !chain_ok fallback */
    char picked[144];
};

const char *fft3d_name(void) { return "gen_planner"; }
const char *fft3d_description(void)
{
    return "planner layer: L -> {ct,gt-pfa,rader,bluestein,dense} candidate trees + generic "
           "strided-row executor (in-place, fused twiddles, register-resident fused CT/GT "
           "codelets, hard leaves 2/3/4/5/7/8) + volume-resident fused chain, any 2<=L<=128; "
           "alternating-layout one-transpose THP-re-homed chain at L>80; "
           "noise-gated create() race with banked per-host picks (@f custody playoff); "
           "adopt via GEN_PLANNER_LIB include";
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
 *
 * gen_r5, two changes:
 *   1. INTERLEAVED sample-major timing (gen_race r4's finding, itself the
 *      whole panel's r4 lesson): candidate-major timing on this node is
 *      confounded by multi-minute core-state drift -- their receipt is a
 *      wisdom-pinned flip-flop at L=12.  All arms are built and warmed
 *      first, then timed as round-robin ROUNDS, min-of-rounds per arm.
 *   2. At batch >= 8 the split-group engine's expressible levels enter the
 *      race as extra arms (timed on a full 8-volume group, scored /8).
 */
typedef struct {
    const pln_node *t;
    int lev;                     /* 0 = per-volume; 1/2/3 = split-group level */
    char name[112];
    pln_p3d *p3;
#if PLN_SIMD
    pln_s8 *s8;
#endif
    double *ga, *gb;             /* group buffers (state / lev-2 alternate) */
    double best, best2;          /* two smallest trials: best scores, the
                                    best-to-best2 gap is the arm's own noise
                                    estimate (gen_r9 noise gate) */
} pln_arm;

static const char *pln_lev_tag(int lev)
{
    return lev == 1 ? "@s1" : lev == 2 ? "@s2" : lev == 3 ? "@s3"
                            : lev == 4 ? "@s4" : "";
}

static void pln_fill_rand(double *p, size_t n, unsigned long seed, double scale)
{
    unsigned long s = seed;
    for (size_t i = 0; i < n; ++i) {
        s = s * 6364136223846793005UL + 1442695040888963407UL;
        p[i] = scale * ((double)(s >> 12) / (double)(1UL << 52) - 0.5);
    }
}

/* returns the winning arm index and leaves its engine in arm[]; all other
 * arms' engines are freed.  -1 on total failure. */
static int pln_race2(int L, int batch, pln_cand *cand, int k,
                     pln_arm *arm, int maxarm, double **bufs)
{
    int na = 0;
    (void)batch;                 /* only consulted in the AVX-512 build */
    if (k > 6) k = 6;            /* gen_r4's sub-tree diversity widens the pool
                                    and wisdom amortizes the extra trials */
    size_t vol = (size_t)L * L * L;
    for (int i = 0; i < k && na < maxarm; ++i) {
        memset(&arm[na], 0, sizeof arm[na]);
        arm[na].t = cand[i].t; arm[na].lev = 0;
        snprintf(arm[na].name, sizeof arm[na].name, "%s", cand[i].name);
        ++na;
    }
#if PLN_SIMD
    int ns8 = 0;
    if (batch >= 8)
        for (int i = 0; i < k && na < maxarm && ns8 < 6; ++i)
            for (int lev = 1; lev <= 4 && na < maxarm && ns8 < 6; ++lev) {
                pln_s8 *s = pln_s8_build(L, cand[i].t, lev);
                if (!s) continue;
                memset(&arm[na], 0, sizeof arm[na]);
                arm[na].t = cand[i].t; arm[na].lev = lev; arm[na].s8 = s;
                snprintf(arm[na].name, sizeof arm[na].name, "%s%s",
                         cand[i].name, pln_lev_tag(lev));
                ++na; ++ns8;
            }
#endif
    double *a = NULL, *st = NULL, *cf = NULL, *G = NULL, *G2 = NULL, *Cg = NULL;
    int need_g = 0;
    for (int i = 0; i < na; ++i) if (arm[i].lev) need_g = 1;
    if (posix_memalign((void **)&a, 64, 2 * vol * sizeof *a) ||
        posix_memalign((void **)&st, 64, 2 * vol * sizeof *st) ||
        posix_memalign((void **)&cf, 64, 2 * vol * sizeof *cf))
        { free(a); free(st); free(cf); return -1; }
    if (need_g &&
        (posix_memalign((void **)&G, 64, 16 * vol * sizeof *G) ||
         posix_memalign((void **)&G2, 64, 16 * vol * sizeof *G2) ||
         posix_memalign((void **)&Cg, 64, 16 * vol * sizeof *Cg))) {
        free(G); free(G2); free(Cg); G = G2 = Cg = NULL;
        need_g = 0;              /* group buffers unavailable: race pv arms only */
    }
    pln_fill_rand(a, 2 * vol, 12345, 1.0);
    pln_fill_rand(cf, 2 * vol, 99991, 0.1);
    if (need_g) {
        pln_fill_rand(G, 16 * vol, 5551, 1.0);
        pln_fill_rand(Cg, 16 * vol, 77713, 0.1);
    }
    /* build + warm every arm first (each arm's warm is its setup sample),
       then time round-robin */
    for (int i = 0; i < na; ++i) {
        arm[i].best = arm[i].best2 = 1e300;
        if (arm[i].lev == 0) {
            arm[i].p3 = pln_p3d_build(L, arm[i].t);
            if (!arm[i].p3) continue;
            memcpy(st, a, 2 * vol * sizeof *st);
            pln_p3d_step(arm[i].p3, st, cf);
        }
#if PLN_SIMD
        else {
            if (!need_g) { pln_s8_free(arm[i].s8); arm[i].s8 = NULL; continue; }
            arm[i].ga = G; arm[i].gb = G2;
            pln_s8_step(arm[i].s8, &arm[i].ga, &arm[i].gb, Cg);
        }
#endif
    }
    int have[12];
    for (int i = 0; i < na; ++i)
        have[i] = (arm[i].lev == 0) ? (arm[i].p3 != NULL)
#if PLN_SIMD
                                    : (arm[i].s8 != NULL);
#else
                                    : 0;
#endif
    /* one interleaved timing round over the arms with best <= lim (1e300 =
       everyone); tracks each arm's two smallest trials */
#define PLN_RACE_ROUND(lim) do {                                              \
        for (int i = 0; i < na; ++i) {                                        \
            double t0, dt;                                                    \
            if (!have[i] || arm[i].best > (lim)) continue;                    \
            if (arm[i].lev == 0) {                                            \
                t0 = pln_now();                                               \
                pln_p3d_step(arm[i].p3, st, cf);                              \
                dt = pln_now() - t0;                                          \
            } else {                                                          \
                t0 = pln_now();                                               \
                pln_s8_step(arm[i].s8, &arm[i].ga, &arm[i].gb, Cg);           \
                dt = (pln_now() - t0) / 8.0;    /* per volume */              \
            }                                                                 \
            if (dt < arm[i].best) { arm[i].best2 = arm[i].best; arm[i].best = dt; } \
            else if (dt < arm[i].best2) arm[i].best2 = dt;                    \
        }                                                                     \
    } while (0)
#if !PLN_SIMD
    /* no group arms exist in the non-SIMD build; keep the macro compilable */
#define pln_s8_step(s, a, b, c) ((void)0)
#endif
    for (int rep = 0; rep < 3; ++rep)
        PLN_RACE_ROUND(1e300);
    /* 2% simplest-first hysteresis (gen_pfa_large's tuner discipline): arm
       order is pv-by-model-cost then group-by-model-cost, so quiet hosts
       pick stably and the battle-tested path wins ties.
       gen_r9 NOISE GATE (PMU_AUDIT.md avenue 1 -- "re-race, never trust, a
       noisy trial"): if the hysteresis winner and any contender within 10%
       sit closer than the pair's own measured noise (sum of min-to-2nd-min
       spreads, floored at 3%), the min-of-3 verdict is a coin flip that
       wisdom would then FREEZE -- exactly the plan-pick instability that
       cost the L=25 cell 25% in r8's audit.  So: EXTEND the race, contenders
       only, up to 6 extra interleaved rounds, until the gap clears the noise
       or the cap is hit.  At the cap the hysteresis IS the tie-break --
       deterministic model order, not window luck, decides inside the band.
       Wisdom storage stays unconditional (gen_race r8: the driver's second
       process must replay the first's winner or the repeatability cmp
       flags); the gate lives in the SAMPLING, so a stored verdict is either
       a measured beyond-noise winner or the deterministic tie-break. */
    int win = -1;
    for (int xrep = 0; ; ++xrep) {
        double bestt = 1e300, bestmin = 1e300;
        win = -1;
        for (int i = 0; i < na; ++i) {
            if (!have[i]) continue;
            if (arm[i].best < bestt * 0.98) { win = i; bestt = arm[i].best; }
            if (arm[i].best < bestmin) bestmin = arm[i].best;
        }
        if (win < 0 || xrep >= 6) break;
        int ambig = 0;
        for (int j = 0; j < na && !ambig; ++j) {
            if (j == win || !have[j] || arm[j].best > 1.10 * arm[win].best)
                continue;
            double gap = fabs(arm[j].best - arm[win].best) / arm[win].best;
            double nz = 1.0;     /* fewer than 2 trials on either arm: noisy */
            if (arm[win].best2 < 1e300 && arm[j].best2 < 1e300)
                nz = (arm[win].best2 - arm[win].best) / arm[win].best
                   + (arm[j].best2 - arm[j].best) / arm[j].best;
            if (gap < 0.03 || gap < nz) ambig = 1;
        }
        if (!ambig) break;
        PLN_RACE_ROUND(1.10 * bestmin);
    }
#undef PLN_RACE_ROUND
#if !PLN_SIMD
#undef pln_s8_step
#endif
    /* TILE PLAYOFF (gen_r10): the pv row width is a per-(host, L) truth the
     * fixed default cannot know -- node pairs: t16 wins at 40 (-2.5..4%,
     * 3/3) and 50 (-2.6..3%, 3/3), loses at 32 (+3.5..5.9%) and 100
     * (+4..13%).  gen_race's r9 scoring keys measured the same split on its
     * own composition of these trees, which is what pointed here.  When the
     * winner is a per-volume plan (and GEN_PLANNER_TILE does not pin),
     * rebuild the winning tree at the alternate width and race the chain
     * step, same noise gate and 2% keep-default hysteresis as custody; the
     * verdict rides the wisdom value as @t<w>. */
    if (win >= 0 && arm[win].lev == 0 && arm[win].p3 &&
        !getenv("GEN_PLANNER_TILE")) {
        const int deft = arm[win].p3->xp->tile;
        pln_tile_req = deft == 16 ? 32 : 16;
        pln_p3d *alt3 = pln_p3d_build(L, arm[win].t);
        pln_tile_req = 0;
        if (alt3) {
            /* Adoption requires the gate to SETTLE (gap clear of the pair's
             * own trial noise) -- a capped-out noisy playoff keeps the
             * default, so a cold create under contention is deterministic
             * (gen_pfa_large r9's lesson: an in-window comparison at the
             * hysteresis edge is a coin flip, and the first version of this
             * playoff flipped 2-of-5 cold picks at 50 exactly that way).
             * The true t16 margins (2.5-4%) clear a settled quiet window;
             * the scoring race banks the verdict and wisdom pins it. */
            pln_p3d *side[2] = { arm[win].p3, alt3 };   /* 0 = incumbent */
            double tb[2] = { 1e300, 1e300 }, tb2[2] = { 1e300, 1e300 };
            int settled = 0;
            for (int rep = 0; rep < 12; ++rep) {
                for (int s = 0; s < 2; ++s) {
                    double t0 = pln_now();
                    pln_p3d_step(side[s], st, cf);
                    double dt = pln_now() - t0;
                    if (dt < tb[s]) { tb2[s] = tb[s]; tb[s] = dt; }
                    else if (dt < tb2[s]) tb2[s] = dt;
                }
                if (rep >= 3) {
                    double lo = tb[0] < tb[1] ? tb[0] : tb[1];
                    double gap = fabs(tb[1] - tb[0]) / lo;
                    double nz = 1.0;
                    if (tb2[0] < 1e300 && tb2[1] < 1e300)
                        nz = (tb2[0] - tb[0]) / tb[0] + (tb2[1] - tb[1]) / tb[1];
                    if (gap >= 0.02 && gap >= 2.0 * nz) { settled = 1; break; }
                }
            }
            if (settled && tb[1] < tb[0] * 0.98) {
                pln_p3d_free(arm[win].p3);
                arm[win].p3 = alt3;
            } else
                pln_p3d_free(alt3);
        }
    }
    /* CUSTODY PLAYOFF (gen_r9): when the winner is a per-volume plan at a
     * cflush-eligible size, measure custody on/off ON THE WINNING PLAN
     * (same buffers; only the flag flips between interleaved trials) and
     * keep the faster.  Same noise gate as the main race: min-of-3 per
     * side, extended to 9 rounds while the gap is inside the pair's own
     * trial noise, and a 2% hysteresis toward the host-default side so the
     * tie-break is deterministic.  The verdict is banked per host through
     * the wisdom value (@f0/@f1 tag, applied by the caller). */
    if (win >= 0 && arm[win].lev == 0 && arm[win].p3 &&
        arm[win].p3->cflush_elig && !arm[win].p3->cflush_pin) {
        pln_p3d *P3 = arm[win].p3;
        const int def = P3->cflush, alt = !def;
        double tb[2] = { 1e300, 1e300 }, tb2[2] = { 1e300, 1e300 };
        for (int rep = 0; rep < 9; ++rep) {
            for (int s = 0; s < 2; ++s) {
                P3->cflush = s;
                double t0 = pln_now();
                pln_p3d_step(P3, st, cf);
                double dt = pln_now() - t0;
                if (dt < tb[s]) { tb2[s] = tb[s]; tb[s] = dt; }
                else if (dt < tb2[s]) tb2[s] = dt;
            }
            if (rep >= 2) {
                double lo = tb[0] < tb[1] ? tb[0] : tb[1];
                double gap = fabs(tb[1] - tb[0]) / lo;
                double nz = 1.0;
                if (tb2[0] < 1e300 && tb2[1] < 1e300)
                    nz = (tb2[0] - tb[0]) / tb[0] + (tb2[1] - tb[1]) / tb[1];
                if (gap >= 0.03 && gap >= nz) break;
            }
        }
        P3->cflush = (tb[alt] < tb[def] * 0.98) ? alt : def;
    }
    for (int i = 0; i < na; ++i) {
        if (i == win) continue;
        if (arm[i].p3) pln_p3d_free(arm[i].p3);
#if PLN_SIMD
        if (arm[i].s8) pln_s8_free(arm[i].s8);
#endif
    }
    free(a); free(st); free(cf);
    /* group buffers are handed to the caller (the winning plan reuses them) */
    bufs[0] = G; bufs[1] = G2; bufs[2] = Cg;
    return win;
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

    /* Plan-time race ON BY DEFAULT from gen_r2 (deliverable #3 in the brief);
     * interleaved sample-major since gen_r5 (gen_race r4's protocol fix).
     * The wisdom value is the tree name plus an optional split-group level
     * tag ("@s1"/"@s2"/"@s3", batch-lane engine, gen_r5).
     * GEN_PLANNER_RACE=0 restores the deterministic model pick (and skips
     * the wisdom cache entirely, keeping that path bit-repeatable offline). */
    const char *race = getenv("GEN_PLANNER_RACE");
    int do_race = !(race && race[0] == '0');
    char wkey[64];
    int raced = 0, wlev = 0, wcf = -1;   /* wcf: banked custody verdict (@f tag) */
    int wtile = 0;                       /* banked tile-playoff verdict (@t tag) */
    const pln_node *wtree = NULL;
    double *gbufs[3] = { NULL, NULL, NULL };
    /* the winning MODE depends on the batch regime (the group engine only
     * exists at batch >= 8, and its economics differ from per-volume), so
     * the wisdom key carries the regime -- a B=1 create() must never
     * clobber a batched pick or vice versa (gen_r5 bug, caught in dev) */
    snprintf(wkey, sizeof wkey, "gen_planner/tree/L%d%s", L,
             batch >= 8 ? "/g8" : "");
    if (do_race) {
        char wname[GR_NAME_MAX];
        if (gr_wisdom_get_str(wkey, wname, sizeof wname)) {
            /* tags after the tree name, in any order: @sN (group level),
             * @fN (custody), @t<w> (tile playoff, gen_r10).  Unknown tags
             * force a re-race (forward compatibility, unchanged). */
            char *at = strchr(wname, '@');
            if (at) {
                *at = 0;
                for (char *tag = at + 1; tag && wlev >= 0; ) {
                    char *next = strchr(tag, '@');
                    if (next) *next = 0;
                    if (tag[0] == 's' && tag[1] >= '1' && tag[1] <= '4' && !tag[2])
                        wlev = tag[1] - '0';
                    else if (tag[0] == 'f' && (tag[1] == '0' || tag[1] == '1') && !tag[2])
                        wcf = tag[1] - '0';      /* custody verdict, pv pick */
                    else if (tag[0] == 't' && atoi(tag + 1) >= 4 && atoi(tag + 1) <= 64)
                        wtile = atoi(tag + 1);   /* tile verdict, pv pick */
                    else
                        wlev = -1;               /* unknown tag: force re-race */
                    tag = next ? next + 1 : NULL;
                }
            }
            if (wlev >= 0)
                for (int i = 0; i < k; ++i)
                    if (!strcmp(cand[i].name, wname)) {   /* stale names re-race */
                        wtree = cand[i].t;
                        break;
                    }
            if (wtree && wlev > 0 && batch < 8) wtree = NULL;  /* wrong regime */
        }
        if (wtree) {
            int tile_applied = 0;
            if (wtile && wlev == 0 && !getenv("GEN_PLANNER_TILE")) {
                pln_tile_req = wtile;            /* replay the banked tile */
                tile_applied = 1;
            }
            p->p3 = pln_p3d_build(L, wtree);
            pln_tile_req = 0;
#if PLN_SIMD
            if (p->p3 && wlev > 0) {
                p->s8 = pln_s8_build(L, wtree, wlev);
                if (!p->s8) { pln_p3d_free(p->p3); p->p3 = NULL; wtree = NULL; }
            }
#else
            if (wlev > 0) { pln_p3d_free(p->p3); p->p3 = NULL; wtree = NULL; }
#endif
            if (p->p3) {
                if (wcf >= 0 && p->p3->cflush_elig && !p->p3->cflush_pin)
                    p->p3->cflush = wcf;         /* replay the banked verdict */
                if (wlev > 0)
                    snprintf(p->picked, sizeof p->picked, "%s%s", wname,
                             pln_lev_tag(wlev));
                else {
                    int n = snprintf(p->picked, sizeof p->picked, "%s", wname);
                    if (wcf >= 0 && n < (int)sizeof p->picked)
                        n += snprintf(p->picked + n, sizeof p->picked - n,
                                      "@f%d", wcf);
                    if (tile_applied && n < (int)sizeof p->picked)
                        snprintf(p->picked + n, sizeof p->picked - n,
                                 "@t%d", wtile);
                }
            }
        }
        if (!p->p3) {
            pln_arm arm[12];
            int win = pln_race2(L, batch, cand, k, arm, 12, gbufs);
            if (win >= 0) {
                p->p3 = arm[win].p3;             /* NULL when a group arm won */
#if PLN_SIMD
                p->s8 = arm[win].s8;
#endif
                if (!p->p3) p->p3 = pln_p3d_build(L, arm[win].t);
                /* pv pick: the playoff verdicts ride the name so wisdom
                 * replays them exactly (@f0/@f1 custody, @t<w> tile) */
                {
                    int n = snprintf(p->picked, sizeof p->picked, "%s",
                                     arm[win].name);
                    if (p->p3 && arm[win].lev == 0) {
                        if (p->p3->cflush_elig && !p->p3->cflush_pin &&
                            n < (int)sizeof p->picked)
                            n += snprintf(p->picked + n, sizeof p->picked - n,
                                          "@f%d", p->p3->cflush);
                        if (p->p3->xp && p->p3->xp->tile != 32 &&
                            !getenv("GEN_PLANNER_TILE") &&
                            n < (int)sizeof p->picked)
                            snprintf(p->picked + n, sizeof p->picked - n,
                                     "@t%d", p->p3->xp->tile);
                    }
                }
                raced = (p->p3 != NULL);
            }
        }
    }
    if (!p->p3) {
        p->p3 = pln_p3d_build(L, cand[0].t);
        snprintf(p->picked, sizeof p->picked, "%s", cand[0].name);
    }
    free(A);                     /* exec plan owns copies of everything it needs */
    if (!p->p3) {
        free(gbufs[0]); free(gbufs[1]); free(gbufs[2]);
        free(p);
        return NULL;
    }
#if PLN_SIMD
    /* group-chain working buffers: reuse the race's when available */
    if (p->s8) {
        size_t vol = (size_t)L * L * L;
        p->G = gbufs[0]; p->G2 = gbufs[1]; p->Cg = gbufs[2];
        gbufs[0] = gbufs[1] = gbufs[2] = NULL;
        int ok = 1;
        if (!p->G && posix_memalign((void **)&p->G, 64, 16 * vol * sizeof(double)))
            { p->G = NULL; ok = 0; }
        if (ok && !p->G2 && p->s8->lev == 2 &&
            posix_memalign((void **)&p->G2, 64, 16 * vol * sizeof(double)))
            { p->G2 = NULL; ok = 0; }
        if (ok && !p->Cg && posix_memalign((void **)&p->Cg, 64, 16 * vol * sizeof(double)))
            { p->Cg = NULL; ok = 0; }
        if (!ok) {               /* no memory for the group path: drop to pv */
            pln_s8_free(p->s8); p->s8 = NULL;
            free(p->G); free(p->G2); free(p->Cg);
            p->G = p->G2 = p->Cg = NULL;
            char *at = strchr(p->picked, '@');
            if (at) *at = 0;
        }
    }
#endif
    free(gbufs[0]); free(gbufs[1]); free(gbufs[2]);
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
#if PLN_SIMD
    /* gen_r11: gate the ALTERNATING-LAYOUT one-transpose chain (L > 80,
     * separate-map regime only) with the same discipline as the main chain
     * gate: the fin path (odd-m final / m=1) and a full 2-step alt chain
     * (canonical entry, swapped entry, canonical zero-copy exit) are each
     * checked against execute + the exact scalar map.  The alt path swaps
     * the y/z pass ORDER on swapped-entry steps, so it is a different
     * rounding (same kernels) -- never shipped unverified.  Custody (@f1)
     * pins the classic path: the alt step carries no cflush. */
    p->alt = 0;
    if (p->chain_ok && L > 80 && p->p3 && !p->p3->fusemap && !p->p3->cflush) {
        const char *e = getenv("GEN_PLANNER_ALT");
        if (!(e && e[0] == '0')) {
            size_t vol = (size_t)L * L * L;
            const size_t L2 = (size_t)L * L;
            p->stv = pln_huge_alloc(2 * vol * sizeof(double), &p->hb0, &p->hs0);
            p->cw  = pln_huge_alloc(2 * vol * sizeof(double), &p->hb1, &p->hs1);
            double *a = NULL, *rf = NULL, *cf = NULL, *ou = NULL, *z = NULL;
            int mem = p->stv && p->cw &&
                !posix_memalign((void **)&a, 64, 2 * vol * sizeof(double)) &&
                !posix_memalign((void **)&rf, 64, 2 * vol * sizeof(double)) &&
                !posix_memalign((void **)&cf, 64, 2 * vol * sizeof(double)) &&
                !posix_memalign((void **)&ou, 64, 2 * vol * sizeof(double)) &&
                !posix_memalign((void **)&z, 64, 2 * vol * sizeof(double));
            int ok1 = 0, ok2 = 0;
            if (mem) {
                pln_fill_rand(a, 2 * vol, 424242, 1.0);
                pln_fill_rand(cf, 2 * vol, 777, 0.1);
                for (int xx = 0; xx < L; ++xx)
                    pln_transpose_in(cf + 2 * (size_t)xx * L2,
                                     p->cw + 2 * (size_t)xx * L2, L, L);
                /* fin path vs one reference step */
                pln_p3d_step_fin(p->p3, a, p->stv, cf, ou);
                pln_p3d_exec(p->p3, (const double _Complex *)a, (double _Complex *)z);
                for (size_t i = 0; i < 2 * vol; i += 2) {
                    double re = z[i] + cf[i], im = z[i+1] + cf[i+1];
                    double sc = 1.0 / (1.0 + sqrt(re*re + im*im));
                    rf[i] = re * sc; rf[i+1] = im * sc;
                }
                {
                    double num = 0, den = 0;
                    for (size_t i = 0; i < 2 * vol; ++i) {
                        double d = ou[i] - rf[i];
                        num += d * d; den += rf[i] * rf[i];
                    }
                    ok1 = (den > 0 && sqrt(num / den) < 1e-12);
                }
                /* 2-step alt chain vs the two-step reference */
                pln_p3d_step_alt(p->p3, a, p->stv, p->cw, NULL);
                pln_p3d_step_alt(p->p3, p->stv, p->stv, cf, ou);
                pln_p3d_exec(p->p3, (const double _Complex *)rf, (double _Complex *)z);
                for (size_t i = 0; i < 2 * vol; i += 2) {
                    double re = z[i] + cf[i], im = z[i+1] + cf[i+1];
                    double sc = 1.0 / (1.0 + sqrt(re*re + im*im));
                    rf[i] = re * sc; rf[i+1] = im * sc;
                }
                {
                    double num = 0, den = 0;
                    for (size_t i = 0; i < 2 * vol; ++i) {
                        double d = ou[i] - rf[i];
                        num += d * d; den += rf[i] * rf[i];
                    }
                    ok2 = (den > 0 && sqrt(num / den) < 1e-12);
                }
            }
            free(a); free(rf); free(cf); free(ou); free(z);
            p->alt = ok1 && ok2;
            if (!p->alt) {
                pln_huge_free(p->hb0, p->hs0); pln_huge_free(p->hb1, p->hs1);
                p->stv = p->cw = NULL; p->hb0 = p->hb1 = NULL;
            }
        }
    }
#endif
#if PLN_SIMD
    /* the split-group engine gets its OWN two-step gate (same discipline):
     * pack -> 2 group steps -> unpack on 8 random volumes vs the per-volume
     * execute + exact scalar map.  This exercises pack, all three group
     * axis passes, ping-pong parity and the split map end to end.  On any
     * disagreement the group engine is dropped, never shipped. */
    if (p->s8) {
        size_t vol = (size_t)L * L * L;
        double *a = NULL, *rf = NULL, *cf = NULL, *ou = NULL, *z = NULL;
        int ok8 = 0;
        int mem = !posix_memalign((void **)&a, 64, 16 * vol * sizeof(double)) &&
                  !posix_memalign((void **)&rf, 64, 16 * vol * sizeof(double)) &&
                  !posix_memalign((void **)&cf, 64, 16 * vol * sizeof(double)) &&
                  !posix_memalign((void **)&ou, 64, 16 * vol * sizeof(double)) &&
                  !posix_memalign((void **)&z, 64, 2 * vol * sizeof(double));
        if (mem) {
            pln_fill_rand(a, 16 * vol, 987654321, 1.0);
            pln_fill_rand(cf, 16 * vol, 13131, 0.1);
            memcpy(rf, a, 16 * vol * sizeof(double));
            /* group path: 2 chained steps on all 8 volumes */
            pln_s8_pack(a, vol, p->G);
            pln_s8_pack(cf, vol, p->Cg);
            double *ga = p->G, *gb = p->G2;
            pln_s8_step(p->s8, &ga, &gb, p->Cg);
            pln_s8_step(p->s8, &ga, &gb, p->Cg);
            pln_s8_unpack(ga, vol, ou);
            /* reference: per-volume execute + exact scalar map */
            for (int v = 0; v < 8; ++v) {
                double *st = rf + 2 * vol * (size_t)v;
                const double *cv = cf + 2 * vol * (size_t)v;
                for (int step = 0; step < 2; ++step) {
                    pln_p3d_exec(p->p3, (const double _Complex *)st, (double _Complex *)z);
                    for (size_t i = 0; i < 2 * vol; i += 2) {
                        double re = z[i] + cv[i], im = z[i+1] + cv[i+1];
                        double sc = 1.0 / (1.0 + sqrt(re*re + im*im));
                        st[i] = re * sc; st[i+1] = im * sc;
                    }
                }
            }
            double num = 0, den = 0;
            for (size_t i = 0; i < 16 * vol; ++i) {
                double d = ou[i] - rf[i];
                num += d * d; den += rf[i] * rf[i];
            }
            ok8 = (den > 0 && sqrt(num / den) < 1e-12);
        }
        free(a); free(rf); free(cf); free(ou); free(z);
        if (!ok8) {
            pln_s8_free(p->s8); p->s8 = NULL;
            free(p->G); free(p->G2); free(p->Cg);
            p->G = p->G2 = p->Cg = NULL;
            char *at = strchr(p->picked, '@');
            if (at) *at = 0;
            raced = 0;           /* a failed group pick must not be persisted */
        }
    }
#endif
    /* persist the raced pick only after its chain gate passed (gen_powp's
     * wisdom discipline); wisdom hits above skip both race and store */
    if (raced && p->chain_ok)
        gr_wisdom_put_str(wkey, p->picked);
    return p;
}

/* per-volume chain for volumes [b0, b1): the gen_r2 volume-major scheme */
static void pln_chain_pv(fft3d_plan *p, const double _Complex *x0,
                         const double _Complex *c, double _Complex *final_out,
                         int m, int b0, int b1)
{
    const size_t vol = (size_t)p->L * p->L * p->L;
#if PLN_SIMD
    /* gen_r11 alternating-layout chain (L > 80): every step but a canonical-
     * entry final runs ONE in-place plane transpose instead of two, the state
     * lives in a THP arena volume (zero-copy entry from x0 / exit to
     * final_out -- gen_layout gen_r11's re-home shape), and swapped-exit maps
     * read the (z,y)-transposed c copy staged once per volume chain. */
    if (p->alt && m >= 1) {
        const int L = p->L;
        const size_t L2 = (size_t)L * L;
        for (int b = b0; b < b1; ++b) {
            const double *x0b = (const double *)(x0 + (size_t)b * vol);
            const double *cf = (const double *)(c + (size_t)b * vol);
            double *ob = (double *)(final_out + (size_t)b * vol);
            if (m == 1) {
                pln_p3d_step_fin(p->p3, x0b, p->stv, cf, ob);
                continue;
            }
            for (int xx = 0; xx < L; ++xx)
                pln_transpose_in(cf + 2 * (size_t)xx * L2,
                                 p->cw + 2 * (size_t)xx * L2, L, L);
            const double *src = x0b;
            for (int s = 1; s <= m; ++s) {
                const int entry_sw = (s - 1) & 1;   /* swapped-entry step? */
                if (s < m)          /* exit layout flips: c copy must match */
                    pln_p3d_step_alt(p->p3, src, p->stv,
                                     entry_sw ? cf : (const double *)p->cw,
                                     NULL);
                else if (entry_sw)  /* even m: alt step exits canonical */
                    pln_p3d_step_alt(p->p3, src, p->stv, cf, ob);
                else                /* odd m: classic pass stays canonical */
                    pln_p3d_step_fin(p->p3, src, p->stv, cf, ob);
                src = p->stv;
            }
        }
        return;
    }
#endif
    for (int b = b0; b < b1; ++b) {
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

/* Owned graded chain: state <- (FFT3(state) + c) / (1 + |FFT3(state) + c|),
 * m steps.  VOLUME-MAJOR (volumes are independent, c is pointwise): each
 * volume runs its whole m-step chain in place in final_out while it is
 * cache-resident -- the gen_rader / gen_dense_prime chain scheme, generalized.
 * gen_r5: when the race picked the split-group engine, volumes go through
 * it 8 at a time (pack once, m steps on the group, unpack once); the
 * B % 8 remainder takes the per-volume path.  Falls back to execute + exact
 * scalar map if the create() gate failed. */
void fft3d_chain(fft3d_plan *p, const double _Complex *x0,
                 const double _Complex *c, double _Complex *final_out, int m)
{
#if PLN_SIMD
    if (p->s8 && p->batch >= 8) {
        const size_t vol = (size_t)p->L * p->L * p->L;
        const int nb = p->batch - p->batch % 8;
        for (int b = 0; b < nb; b += 8) {
            pln_s8_pack((const double *)(x0 + (size_t)b * vol), vol, p->G);
            pln_s8_pack((const double *)(c + (size_t)b * vol), vol, p->Cg);
            double *ga = p->G, *gb = p->G2;
            for (int s = 0; s < m; ++s)
                pln_s8_step(p->s8, &ga, &gb, p->Cg);
            pln_s8_unpack(ga, vol, (double *)(final_out + (size_t)b * vol));
        }
        pln_chain_pv(p, x0, c, final_out, m, nb, p->batch);
        return;
    }
#endif
    pln_chain_pv(p, x0, c, final_out, m, 0, p->batch);
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
#if PLN_SIMD
    pln_s8_free(p->s8);
    free(p->G); free(p->G2); free(p->Cg);
    pln_huge_free(p->hb0, p->hs0); pln_huge_free(p->hb1, p->hs1);
#endif
    free(p->scratch);
    free(p);
}

#endif /* GEN_PLANNER_LIB */
#endif /* GEN_PLANNER_C_INCLUDED */
