/* =============================================================================
 * GEN_LAYOUT -- the allocation & layout library layer (round gen_r2)
 * =============================================================================
 *
 * LIBRARY LAYER, scored by ADOPTION.  Class owners: adopt with
 *
 *     #define GEN_LAYOUT_LIB_ONLY
 *     #include "gen_layout.c"           // impl/ is the include dir; everything
 *                                       // is `static`, prefix gl_, no symbols leak
 *
 * and take only what you need.  The entry points below the library section are
 * compiled only when this file IS the entry (my demo binary); they never
 * conflict with yours.
 *
 * WHAT IS IN HERE AND WHY (each item is a paid-for lesson, source cited):
 *
 *  1. gl_map_huge / gl_arena_*  -- 2 MiB-aligned THP arenas, MADV_HUGEPAGE,
 *     prefaulted at create() so first-touch faults never land in execute().
 *     [warm 00291a90 prelude_c.py alloc_huge; L17_winograd ice_r6 "2-MB THP
 *      arenas"; LITERATURE 05 SS7: dTLB reach]
 *  2. gl_arena_take            -- every successive buffer lands on a DIFFERENT
 *     mod-4096 phase (4672 B = 73-line rotation; 576 B/step walks all 64 line
 *     phases).  Two page-aligned buffers is the 4K store->load aliasing worst
 *     case.  [warm 00291a90 alloc_huge_st; LITERATURE 08 SS1.8, 10 "4K-aliasing
 *      epidemic"]
 *  3. gl_phase4k / gl_far_phase4k / gl_arena_take_phase -- explicit mod-4096
 *     placement.  The driver owns in/out, so scratch must be placed AGAINST
 *     their phases, at execute() time, re-derived when the buffer pair changes.
 *     [L6_pfa ice: dynamic re-place when driver residues change]
 *  4. gl_alias_pairs4k / gl_best_lineshift4k / gl_walk_min_gap4k -- the
 *     collision MODEL.  A fixed stagger is only as good as the stride analysis
 *     behind it: streams walking different strides mod 4096 can collide on one
 *     specific plane no base offset fixes (L23_rader ice_r5 post-mortem), and
 *     L17_matrixsimd ice_r8 showed the natural 73-line row padding can be
 *     WORSE than 1.25-page padding (weighted collisions 31 vs <=4) -- so
 *     measure the model, don't guess the stagger.  These are the generalized
 *     versions of L17's astab builder.
 *  4b. gl_stream_audit4k / gl_pick_pitch4k -- NEW gen_r2: the one-call forms.
 *     Declare your kernel's concurrent streams, get the full collision report
 *     (assert pairs==0 in create() the way you assert twiddle exactness), and
 *     pick a row/plane pitch by MEASURED score instead of the odd-line rule.
 *     [asked for by gen_dense_prime r1 item 2 (31->32 row padding) and
 *      gen_batchlane r1 item 3 (A/B the inherited bl8 pads)]
 *  5. gl_pad_stride            -- odd-line-count stride rounding (rotates all
 *     64 line phases across rows).  Use 4b when you know the stream geometry.
 *  6. gl_deint8 / gl_int8 / gl_tr8x8 / gl_pack8 / gl_unpack8 -- pencil/lane
 *     SoA converters (AVX-512): interleaved complex <-> split [site][2][8]
 *     blocks, one vpermt2pd per output vector, 8x8 transpose at 24 shuffles /
 *     64 doubles.  For gen_batchlane's 8-vol lanes and any pencil-SoA stage.
 *     [L8_fusedaxes bl8; L13_rader soa8; LITERATURE 04: split layout deletes
 *      permutations from the hot loop]
 *
 * THE DEMO ENTRY (when not GEN_LAYOUT_LIB_ONLY): a generic FOLDED dense
 * row-column DFT, any 2 <= L <= 128, staged through pencil-SoA blocks.
 * gen_r2: adopted gen_dense_prime's conjugate-pair fold (u_j = x_j + x_{L-j},
 * v_j = x_j - x_{L-j}; C = cos-matrix * u, S = sin-matrix * v; X_k = C_k - iS_k,
 * X_{L-k} = C_k + iS_k) -- all constants REAL, ~4x fewer FMAs than the r1
 * unfolded form -- and replaced the load-bound axis-2 row kernel with a
 * transpose-staged 8-pencil block kernel (gl_tr8x8 both ways, broadcast tables
 * reused across 8 pencils).  Still O(L^4)/axis dense class by design: the
 * panel's any-L vectorized floor and the layer's living test bench, not a
 * contender.  Knobs for A/B on the node:
 *     -DGL_DEMO_PLAIN=1     posix_memalign scratch, no THP/stagger/placement
 *     -DGL_DEMO_NOPLACE=1   THP arena + static stagger, no dynamic placement
 *     -DGL_DEMO_COLLIDE=1   adversarial: scratch at out's exact page phase
 * ============================================================================= */

#ifndef GEN_LAYOUT_C_INCLUDED
#define GEN_LAYOUT_C_INCLUDED

#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

enum { GL_LINE = 64, GL_PAGE = 4096 };
#define GL_HUGE (2ul << 20)
#define GL_STAGGER 4672u /* 73 lines; successive phases 576 B apart, gcd(9,64)=1 */

/* ---- 1. huge-page mapping ------------------------------------------------ */
typedef struct {
    char *base;
    size_t len;
    int mmapped;
} gl_map;

static inline void *gl_map_huge(gl_map *m, size_t bytes)
{
    size_t len = (bytes + GL_HUGE - 1) & ~(GL_HUGE - 1);
    void *p = mmap(0, len + GL_HUGE, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p != MAP_FAILED) {
        char *al = (char *)(((uintptr_t)p + GL_HUGE - 1) & ~(uintptr_t)(GL_HUGE - 1));
        size_t head = (size_t)(al - (char *)p);
        if (head) munmap(p, head);
        size_t tail = (size_t)(((char *)p + len + GL_HUGE) - (al + len));
        if (tail) munmap(al + len, tail);
        madvise(al, len, MADV_HUGEPAGE);
        memset(al, 0, len); /* prefault now: page faults belong in create() */
        m->base = al;
        m->len = len;
        m->mmapped = 1;
        return al;
    }
    void *q = NULL;
    if (posix_memalign(&q, GL_LINE, len) != 0 || !q) {
        m->base = NULL;
        return NULL;
    }
    memset(q, 0, len);
    m->base = q;
    m->len = len;
    m->mmapped = 0;
    return q;
}

static inline void gl_unmap(gl_map *m)
{
    if (!m->base) return;
    if (m->mmapped) munmap(m->base, m->len);
    else free(m->base);
    m->base = NULL;
}

/* ---- 2./3. arena with per-buffer phase control --------------------------- */
static inline unsigned gl_phase4k(const void *p)
{
    return (unsigned)((uintptr_t)p & (GL_PAGE - 1));
}

typedef struct {
    gl_map map;
    size_t used;
    unsigned next_phase; /* mod-4096 phase the next take() will get */
} gl_arena;

static inline int gl_arena_init(gl_arena *a, size_t bytes)
{
    a->used = 0;
    a->next_phase = 0;
    return gl_map_huge(&a->map, bytes) ? 0 : -1;
}

static inline void gl_arena_destroy(gl_arena *a) { gl_unmap(&a->map); }

/* Take `bytes` placed so the returned pointer has mod-4096 phase `phase4k`
 * (line-aligned).  Costs at most 4095 B of arena slack per take. */
static inline void *gl_arena_take_phase(gl_arena *a, size_t bytes, unsigned phase4k)
{
    size_t off = (a->used + GL_LINE - 1) & ~(size_t)(GL_LINE - 1);
    unsigned have = gl_phase4k(a->map.base + off);
    off += (phase4k + GL_PAGE - have) & (GL_PAGE - 1);
    if (off + bytes > a->map.len) return NULL;
    a->used = off + bytes;
    return a->map.base + off;
}

/* Take `bytes` on the arena's rotating stagger: consecutive buffers land 576 B
 * apart mod 4096, never sharing a page phase. */
static inline void *gl_arena_take(gl_arena *a, size_t bytes)
{
    void *p = gl_arena_take_phase(a, bytes, a->next_phase & (GL_PAGE - 1));
    a->next_phase = (a->next_phase + GL_STAGGER) & (GL_PAGE - 1);
    return p;
}

/* ---- 4. the mod-4096 collision model ------------------------------------- */
static inline long gl_circdist4k(long x, long y)
{
    long d = (x - y) & (GL_PAGE - 1);
    return d > GL_PAGE / 2 ? GL_PAGE - d : d;
}

/* Line-aligned phase maximizing the minimum circular distance to `n` given
 * phases -- "put the scratch as far mod 4096 from everyone as possible". */
static inline unsigned gl_far_phase4k(const unsigned *ph, int n)
{
    unsigned best = 0;
    long bestmin = -1;
    for (unsigned c = 0; c < GL_PAGE; c += GL_LINE) {
        long mn = GL_PAGE;
        for (int i = 0; i < n; ++i) {
            long d = gl_circdist4k((long)c, (long)(ph[i] & (GL_PAGE - 1)));
            if (d < mn) mn = d;
        }
        if (mn > bestmin) {
            bestmin = mn;
            best = c;
        }
    }
    return best;
}

/* Pairs (i,k) with |(pl + i*sl) - (ps + k*ss)| mod+-4096 < window: the count of
 * load rows of one stream landing inside the 4K-alias window of another
 * stream's store rows.  window 64 B matches ld_blocks_partial.address_alias. */
static inline long gl_alias_pairs4k(long pl, long sl, int nl,
                                    long ps, long ss, int ns, int window)
{
    long hits = 0;
    for (int i = 0; i < nl; ++i)
        for (int k = 0; k < ns; ++k)
            if (gl_circdist4k(pl + i * sl, ps + k * ss) < window) ++hits;
    return hits;
}

typedef struct {
    long phase;  /* mod-4096 phase of row 0 */
    long stride; /* bytes between rows */
    int n;       /* rows live concurrently */
} gl_stream4k;

/* Best 0..maxshift_lines-1 line shift of a stream against fixed streams,
 * minimizing total alias pairs.  Generalizes L17_matrixsimd's astab. */
static inline int gl_best_lineshift4k(long myphase, long mystride, int myn,
                                      const gl_stream4k *fixed, int nfixed,
                                      int window, int maxshift_lines, long *score_out)
{
    int bestj = 0;
    long bests = 0x7fffffffL;
    for (int j = 0; j < maxshift_lines; ++j) {
        long s = 0;
        for (int t = 0; t < nfixed; ++t)
            s += gl_alias_pairs4k(myphase + GL_LINE * (long)j, mystride, myn,
                                  fixed[t].phase, fixed[t].stride, fixed[t].n, window);
        if (s < bests) {
            bests = s;
            bestj = j;
        }
    }
    if (score_out) *score_out = bests;
    return bestj;
}

/* Min circular gap mod 4096 between two walking streams over nsteps: the
 * L23_rader audit -- differing strides CAN collide on one specific step even
 * when the bases are far apart.  Run this at plan time on every (load stream,
 * store stream) pair whose strides differ. */
static inline long gl_walk_min_gap4k(long pa, long sa, long pb, long sb, int nsteps)
{
    long mn = GL_PAGE / 2;
    for (int k = 0; k < nsteps; ++k) {
        long d = gl_circdist4k(pa + k * sa, pb + k * sb);
        if (d < mn) mn = d;
    }
    return mn;
}

/* ---- 4b. one-call stream audit + measured pitch picker (NEW gen_r2) ------- */
typedef struct {
    long pairs;        /* total 4K-alias pairs over all unordered stream pairs */
    int worst_a, worst_b;
    long worst_pairs;  /* the worst single pair's hit count                    */
    long min_walk_gap; /* min circular walk gap over differing-stride pairs    */
    int walk_a, walk_b;
} gl_audit4k;

/* Declare every stream your kernel walks concurrently; returns the total 4K
 * collision-model score (0 is the goal) and, via rep, who collides with whom.
 * Assert ==0 in create() the way gen_twiddle has you assert table exactness. */
static inline long gl_stream_audit4k(const gl_stream4k *s, int n, int window,
                                     gl_audit4k *rep)
{
    gl_audit4k a;
    a.pairs = 0;
    a.worst_a = a.worst_b = a.walk_a = a.walk_b = -1;
    a.worst_pairs = 0;
    a.min_walk_gap = GL_PAGE / 2;
    for (int i = 0; i < n; ++i)
        for (int k = i + 1; k < n; ++k) {
            long h = gl_alias_pairs4k(s[i].phase, s[i].stride, s[i].n,
                                      s[k].phase, s[k].stride, s[k].n, window);
            a.pairs += h;
            if (h > a.worst_pairs) {
                a.worst_pairs = h;
                a.worst_a = i;
                a.worst_b = k;
            }
            if (s[i].stride != s[k].stride) {
                int steps = s[i].n < s[k].n ? s[i].n : s[k].n;
                long g = gl_walk_min_gap4k(s[i].phase, s[i].stride,
                                           s[k].phase, s[k].stride, steps);
                if (g < a.min_walk_gap) {
                    a.min_walk_gap = g;
                    a.walk_a = i;
                    a.walk_b = k;
                }
            }
        }
    if (rep) *rep = a;
    return a.pairs;
}

/* Pick a row/plane pitch by MEASURED collision score, not the odd-line rule:
 * candidates are line-rounded row_bytes + 0..max_extra_lines lines; the score
 * is this stream's alias pairs against the fixed streams.  Ties go to the
 * smallest pitch (memory frugality).  This is the L17 ice_r8 discovery (73-line
 * "natural" padding scored 31 weighted collisions, 80 lines scored <=4) as an
 * API.  gen_dense_prime's 31->32 row padding question is one call. */
static inline size_t gl_pick_pitch4k(size_t row_bytes, int nrows, long base_phase,
                                     const gl_stream4k *fixed, int nfixed,
                                     int window, int max_extra_lines, long *score_out)
{
    size_t base = (row_bytes + GL_LINE - 1) & ~(size_t)(GL_LINE - 1);
    size_t bestp = base;
    long bests = 0x7fffffffL;
    for (int e = 0; e <= max_extra_lines; ++e) {
        size_t pitch = base + (size_t)e * GL_LINE;
        long sc = 0;
        for (int t = 0; t < nfixed; ++t)
            sc += gl_alias_pairs4k(base_phase, (long)pitch, nrows,
                                   fixed[t].phase, fixed[t].stride, fixed[t].n,
                                   window);
        if (sc < bests) {
            bests = sc;
            bestp = pitch;
        }
    }
    if (score_out) *score_out = bests;
    return bestp;
}

/* ---- 5. stride padding ---------------------------------------------------- */
/* Round a row/volume stride up to an ODD number of cache lines: consecutive
 * rows then rotate through all 64 line phases instead of repeating one.
 * When you know both streams' strides, prefer gl_pick_pitch4k: L17 ice_r8
 * measured 73-line padding at 31 weighted collisions vs <=4 for 80 lines. */
static inline size_t gl_pad_stride(size_t bytes)
{
    size_t s = (bytes + GL_LINE - 1) & ~(size_t)(GL_LINE - 1);
    if (((s / GL_LINE) & 1) == 0) s += GL_LINE;
    return s;
}

/* ---- 6. pencil / lane SoA converters (AVX-512) ---------------------------- */
#if defined(__AVX512F__)

/* two interleaved zmm (8 complex) -> split re, im zmm.  1 vpermt2pd each. */
static inline void gl_deint8(__m512d v0, __m512d v1, __m512d *re, __m512d *im)
{
    const __m512i IE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i IO = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    *re = _mm512_permutex2var_pd(v0, IE, v1);
    *im = _mm512_permutex2var_pd(v0, IO, v1);
}

/* split re, im zmm -> two interleaved zmm (8 complex). */
static inline void gl_int8(__m512d re, __m512d im, __m512d *lo, __m512d *hi)
{
    const __m512i IL = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i IH = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    *lo = _mm512_permutex2var_pd(re, IL, im);
    *hi = _mm512_permutex2var_pd(re, IH, im);
}

/* in-register 8x8 double transpose, 24 shuffles (ports: all shuffle). */
static inline void gl_tr8x8(__m512d r[8])
{
    __m512d t0 = _mm512_unpacklo_pd(r[0], r[1]), t1 = _mm512_unpackhi_pd(r[0], r[1]);
    __m512d t2 = _mm512_unpacklo_pd(r[2], r[3]), t3 = _mm512_unpackhi_pd(r[2], r[3]);
    __m512d t4 = _mm512_unpacklo_pd(r[4], r[5]), t5 = _mm512_unpackhi_pd(r[4], r[5]);
    __m512d t6 = _mm512_unpacklo_pd(r[6], r[7]), t7 = _mm512_unpackhi_pd(r[6], r[7]);
    __m512d u0 = _mm512_shuffle_f64x2(t0, t2, 0x88), u1 = _mm512_shuffle_f64x2(t1, t3, 0x88);
    __m512d u2 = _mm512_shuffle_f64x2(t0, t2, 0xdd), u3 = _mm512_shuffle_f64x2(t1, t3, 0xdd);
    __m512d u4 = _mm512_shuffle_f64x2(t4, t6, 0x88), u5 = _mm512_shuffle_f64x2(t5, t7, 0x88);
    __m512d u6 = _mm512_shuffle_f64x2(t4, t6, 0xdd), u7 = _mm512_shuffle_f64x2(t5, t7, 0xdd);
    r[0] = _mm512_shuffle_f64x2(u0, u4, 0x88);
    r[1] = _mm512_shuffle_f64x2(u1, u5, 0x88);
    r[2] = _mm512_shuffle_f64x2(u2, u6, 0x88);
    r[3] = _mm512_shuffle_f64x2(u3, u7, 0x88);
    r[4] = _mm512_shuffle_f64x2(u0, u4, 0xdd);
    r[5] = _mm512_shuffle_f64x2(u1, u5, 0xdd);
    r[6] = _mm512_shuffle_f64x2(u2, u6, 0xdd);
    r[7] = _mm512_shuffle_f64x2(u3, u7, 0xdd);
}
#endif /* __AVX512F__ */

/* 8 interleaved-complex streams (stream t at src + t*stride elements) ->
 * lane-SoA blocks dst[site][2][8]: 8 reals then 8 imags per site, lane =
 * stream.  For gen_batchlane: stream stride = L^3 gives 8-volume lanes; for
 * pencil SoA: stride = the pencil-to-pencil distance. */
static inline void gl_pack8(const double _Complex *src, size_t stride,
                            double *dst, size_t nsites)
{
    size_t s = 0;
#if defined(__AVX512F__)
    for (; s + 4 <= nsites; s += 4) {
        __m512d r[8];
        for (int t = 0; t < 8; ++t)
            r[t] = _mm512_loadu_pd((const double *)(src + t * stride + s));
        gl_tr8x8(r);
        for (int q = 0; q < 8; ++q)
            _mm512_storeu_pd(dst + 16 * s + 8 * q, r[q]);
    }
#endif
    for (; s < nsites; ++s)
        for (int t = 0; t < 8; ++t) {
            dst[16 * s + t] = creal(src[t * stride + s]);
            dst[16 * s + 8 + t] = cimag(src[t * stride + s]);
        }
}

/* exact inverse of gl_pack8 */
static inline void gl_unpack8(const double *src, double _Complex *dst,
                              size_t stride, size_t nsites)
{
    size_t s = 0;
#if defined(__AVX512F__)
    for (; s + 4 <= nsites; s += 4) {
        __m512d r[8];
        for (int q = 0; q < 8; ++q)
            r[q] = _mm512_loadu_pd(src + 16 * s + 8 * q);
        gl_tr8x8(r);
        for (int t = 0; t < 8; ++t)
            _mm512_storeu_pd((double *)(dst + t * stride + s), r[t]);
    }
#endif
    for (; s < nsites; ++s)
        for (int t = 0; t < 8; ++t)
            dst[t * stride + s] = src[16 * s + t] + I * src[16 * s + 8 + t];
}

/* one-shot round-trip audit of the SoA converters; returns 1 on pass */
static inline int gl_selftest(void)
{
    enum { NS = 11, STR = 16 };
    static double _Complex a[8 * STR], b[8 * STR];
    static double soa[16 * NS];
    for (int t = 0; t < 8; ++t)
        for (int s = 0; s < STR; ++s)
            a[t * STR + s] = (double)(t * 100 + s) + I * (double)(t * 100 + s) * 0.5;
    gl_pack8(a, STR, soa, NS);
    for (int s = 0; s < NS; ++s)
        for (int t = 0; t < 8; ++t)
            if (soa[16 * s + t] != (double)(t * 100 + s) ||
                soa[16 * s + 8 + t] != (double)(t * 100 + s) * 0.5)
                return 0;
    memset(b, 0, sizeof b);
    gl_unpack8(soa, b, STR, NS);
    for (int t = 0; t < 8; ++t)
        for (int s = 0; s < NS; ++s)
            if (b[t * STR + s] != a[t * STR + s]) return 0;
    return 1;
}

#endif /* GEN_LAYOUT_C_INCLUDED */

/* =============================================================================
 * DEMO ENTRY: any-L FOLDED dense matrixsimd row-column DFT exercising the layer.
 * Conjugate-pair fold adopted from gen_dense_prime gen_r1 (itself from ice
 * L13_direct / L17 / L23_matrixsimd): real-constant C/S matrices over
 * u_j = x_j + x_{L-j}, v_j = x_j - x_{L-j}; X_k = C_k - iS_k, X_{L-k} = C_k + iS_k.
 * Compiled only when this file is the entry translation unit.
 * ============================================================================= */
#ifndef GEN_LAYOUT_LIB_ONLY

#include <math.h>

#include "../fft3d_api.h"

#ifndef GL_DEMO_PLAIN
#define GL_DEMO_PLAIN 0
#endif
#ifndef GL_DEMO_NOPLACE
#define GL_DEMO_NOPLACE 0
#endif
/* adversarial control: place every scratch at the SAME mod-4096 phase as the
 * driver's out buffer -- the 4K-alias worst case the placement exists to avoid */
#ifndef GL_DEMO_COLLIDE
#define GL_DEMO_COLLIDE 0
#endif

struct fft3d_plan {
    int L, Lp, batch;
    int h;    /* floor((L-1)/2): number of (k, L-k) output pairs / (j, L-j) folds */
    int hu;   /* u rows: h, +1 lone x_{L/2} row when L is even */
    int hs;   /* table row stride (hu padded to 8) */
    int even; /* L % 2 == 0 */
    size_t V;
    gl_arena ar;
    double *Ct; /* [h + 1 + even][hs]: rows k=1..h, then k=0, then k=L/2  */
    double *St; /* [max(h,1)][hs]:    rows k=1..h                         */
    double *pb; /* pencil staging block [Lp][2][8]                        */
    double *ob; /* axis-2 output staging block [Lp][2][8]                 */
    double *ub; /* folded u rows [hu][2][8]                               */
    double *vb; /* folded v rows [h][2][8]                                */
    double _Complex *zt; /* chain: one interleaved z volume                */
    /* scratch slabs own GL_PAGE+GL_LINE of slack; s* are the placed bases */
    double *s1r_slab, *s1i_slab, *s2r_slab, *s2i_slab;
    double *s1r, *s1i, *s2r, *s2i;
    const void *pin, *pout; /* driver buffer pair the placement was derived for */
    void *plain[12];
    int nplain;
};

const char *fft3d_name(void) { return "gen_layout"; }
const char *fft3d_description(void)
{
    return "LIBRARY LAYER (scored by adoption): THP arenas, 4K stagger/collision-model "
           "placement + stream audit & measured pitch picker, pencil SoA pack "
           "(adopt: #define GEN_LAYOUT_LIB_ONLY + #include gen_layout.c); "
           "entry=any-L conjugate-pair-folded dense matrixsimd demo of the layer";
}
int fft3d_supports(int L) { return L >= 2 && L <= 128; }

static void *demo_alloc(fft3d_plan *p, size_t bytes)
{
#if GL_DEMO_PLAIN
    void *q = NULL;
    if (posix_memalign(&q, GL_LINE, bytes) != 0 || !q) return NULL;
    memset(q, 0, bytes);
    p->plain[p->nplain++] = q;
    return q;
#else
    return gl_arena_take(&p->ar, bytes);
#endif
}

/* One folded table row: Crow[j-1] = cos(2*pi*k*j/L), Srow[j-1] = sin(2*pi*k*j/L),
 * j = 1..hu (the lone j = L/2 column included via hu when L is even).  Exact
 * integer reduction + fold into [-pi, pi]; quarter/half turns exact.  gen_twiddle's
 * gen_r1 audit measured this pattern at 0.497 ulp for den <= 128 -- table error
 * is not a limiter at this L range. */
static void dm_fill_row(double *Crow, double *Srow, int k, int L, int hu)
{
    const long double TWO_PI = 6.283185307179586476925286766559L;
    for (int j = 1; j <= hu; ++j) {
        long m = ((long)k * j) % L;
        double c, s;
        if (m == 0) {
            c = 1.0;
            s = 0.0;
        } else if (2 * m == L) {
            c = -1.0;
            s = 0.0;
        } else {
            long mm = (2 * m > L) ? m - L : m;
            long double th = (TWO_PI * (long double)mm) / (long double)L;
            c = (double)cosl(th);
            s = (double)sinl(th);
        }
        Crow[j - 1] = c;
        if (Srow) Srow[j - 1] = s;
    }
}

static void dm_fill_tabs(fft3d_plan *p)
{
    const int L = p->L, h = p->h, hu = p->hu, hs = p->hs;
    for (int k = 1; k <= h; ++k)
        dm_fill_row(p->Ct + (size_t)(k - 1) * hs, p->St + (size_t)(k - 1) * hs,
                    k, L, hu);
    dm_fill_row(p->Ct + (size_t)h * hs, NULL, 0, L, hu); /* k=0: all ones */
    if (p->even)
        dm_fill_row(p->Ct + (size_t)(h + 1) * hs, NULL, L / 2, L, hu);
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L)) return NULL;
    if (!gl_selftest()) return NULL; /* SoA converters must be provably exact */
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->Lp = (L + 7) & ~7;
    p->batch = batch;
    p->V = (size_t)L * L * L;
    p->h = (L - 1) / 2;
    p->even = (L % 2 == 0);
    p->hu = p->h + p->even;
    p->hs = (p->hu + 7) & ~7;

    const size_t ctbytes = (size_t)(p->h + 1 + p->even) * p->hs * sizeof(double);
    const size_t stbytes = (size_t)(p->h > 0 ? p->h : 1) * p->hs * sizeof(double);
    const size_t pbbytes = (size_t)16 * p->Lp * sizeof(double) + 2 * GL_LINE;
    const size_t ubbytes = (size_t)16 * (p->hu > 0 ? p->hu : 1) * sizeof(double) + GL_LINE;
    const size_t vbbytes = (size_t)16 * (p->h > 0 ? p->h : 1) * sizeof(double) + GL_LINE;
    const size_t sbytes = p->V * sizeof(double) + GL_PAGE + 2 * GL_LINE;
    const size_t zbytes = p->V * sizeof(double _Complex) + 2 * GL_LINE;

#if !GL_DEMO_PLAIN
    size_t total = ctbytes + stbytes + 2 * pbbytes + ubbytes + vbbytes +
                   4 * sbytes + zbytes + 24 * GL_PAGE + (64u << 10);
    if (gl_arena_init(&p->ar, total) != 0) {
        free(p);
        return NULL;
    }
#endif
    p->Ct = demo_alloc(p, ctbytes);
    p->St = demo_alloc(p, stbytes);
    p->pb = demo_alloc(p, pbbytes);
    p->ob = demo_alloc(p, pbbytes);
    p->ub = demo_alloc(p, ubbytes);
    p->vb = demo_alloc(p, vbbytes);
    p->s1r_slab = demo_alloc(p, sbytes);
    p->s1i_slab = demo_alloc(p, sbytes);
    p->s2r_slab = demo_alloc(p, sbytes);
    p->s2i_slab = demo_alloc(p, sbytes);
    p->zt = demo_alloc(p, zbytes);
    if (!p->Ct || !p->St || !p->pb || !p->ob || !p->ub || !p->vb || !p->zt ||
        !p->s1r_slab || !p->s1i_slab || !p->s2r_slab || !p->s2i_slab) {
        fft3d_destroy(p);
        return NULL;
    }
    /* until the first execute derives real placement: slab bases (these carry
     * the arena's stagger already, or the plain allocator's phases) */
    p->s1r = p->s1r_slab;
    p->s1i = p->s1i_slab;
    p->s2r = p->s2r_slab;
    p->s2i = p->s2i_slab;
    dm_fill_tabs(p);
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    for (int i = 0; i < p->nplain; ++i) free(p->plain[i]);
#if !GL_DEMO_PLAIN
    gl_arena_destroy(&p->ar);
#endif
    free(p);
}

/* place a slab base at a wanted mod-4096 phase inside its slack */
static double *dm_place(double *slab, unsigned want)
{
    unsigned have = gl_phase4k(slab);
    return (double *)((char *)slab + ((want + GL_PAGE - have) & (GL_PAGE - 1)));
}

/* Dynamic placement (L6_pfa pattern): the driver owns in/out, their phases are
 * only known at execute(); re-derive scratch phases whenever the pair changes.
 * Legal because every scratch is fully rewritten before it is read. */
static void dm_placement(fft3d_plan *p, const void *in, const void *out)
{
#if GL_DEMO_PLAIN || GL_DEMO_NOPLACE
    (void)p;
    (void)in;
    (void)out;
#elif GL_DEMO_COLLIDE
    if (in == p->pin && out == p->pout) return;
    unsigned bad = gl_phase4k(out);
    p->s1r = dm_place(p->s1r_slab, bad);
    p->s1i = dm_place(p->s1i_slab, bad);
    p->s2r = dm_place(p->s2r_slab, bad);
    p->s2i = dm_place(p->s2i_slab, bad);
    p->pin = in;
    p->pout = out;
#else
    if (in == p->pin && out == p->pout) return;
    unsigned ph[8];
    int n = 0;
    ph[n++] = gl_phase4k(in);
    ph[n++] = gl_phase4k(out);
    unsigned vs = (unsigned)((p->V * sizeof(double _Complex)) & (GL_PAGE - 1));
    if (p->batch > 1 && vs) { /* cover the next volume's rotated phases too */
        ph[n] = (ph[0] + vs) & (GL_PAGE - 1);
        ++n;
        ph[n] = (ph[1] + vs) & (GL_PAGE - 1);
        ++n;
    }
    p->s1r = dm_place(p->s1r_slab, gl_far_phase4k(ph, n));
    ph[n++] = gl_phase4k(p->s1r);
    p->s1i = dm_place(p->s1i_slab, gl_far_phase4k(ph, n));
    ph[n++] = gl_phase4k(p->s1i);
    p->s2r = dm_place(p->s2r_slab, gl_far_phase4k(ph, n));
    ph[n++] = gl_phase4k(p->s2r);
    p->s2i = dm_place(p->s2i_slab, gl_far_phase4k(ph, n));
    p->pin = in;
    p->pout = out;
#endif
}

#if defined(__AVX512F__)

/* fold the staged pencil block pb[L][2][8] into u/v rows:
 * ub[j-1] = pb[j] + pb[L-j], vb[j-1] = pb[j] - pb[L-j], j = 1..h;
 * when L is even the lone x_{L/2} row is appended as ub[hu-1] (weights come
 * from the extended cos table column, so it needs no v partner). */
static inline void dm_fold8(const double *pb, int L, int h, int even,
                            double *ub, double *vb)
{
    for (int j = 1; j <= h; ++j) {
        __m512d ar = _mm512_loadu_pd(pb + 16 * j);
        __m512d ai = _mm512_loadu_pd(pb + 16 * j + 8);
        __m512d br = _mm512_loadu_pd(pb + 16 * (L - j));
        __m512d bi = _mm512_loadu_pd(pb + 16 * (L - j) + 8);
        _mm512_storeu_pd(ub + 16 * (j - 1), _mm512_add_pd(ar, br));
        _mm512_storeu_pd(ub + 16 * (j - 1) + 8, _mm512_add_pd(ai, bi));
        _mm512_storeu_pd(vb + 16 * (j - 1), _mm512_sub_pd(ar, br));
        _mm512_storeu_pd(vb + 16 * (j - 1) + 8, _mm512_sub_pd(ai, bi));
    }
    if (even) {
        _mm512_storeu_pd(ub + 16 * h, _mm512_loadu_pd(pb + 16 * (L / 2)));
        _mm512_storeu_pd(ub + 16 * h + 8, _mm512_loadu_pd(pb + 16 * (L / 2) + 8));
    }
}

/* Folded 8-column pencil kernel: from x0 (= pb row 0) and the u/v rows,
 * produce all L output rows into split, strided rows (masked).  4 output
 * pairs per j-sweep = 16 independent FMA chains; 12 loads / 16 FMAs.
 * Also serves axis 2 with dr=ob, di=ob+8, ostride=16, sm=0xFF. */
static inline void dm_kfold8(const fft3d_plan *p, const double *pb,
                             const double *ub, const double *vb,
                             double *dr, double *di, size_t ostride, __mmask8 sm)
{
    const int L = p->L, h = p->h, hu = p->hu, hs = p->hs, even = p->even;
    const __m512d x0r = _mm512_loadu_pd(pb), x0i = _mm512_loadu_pd(pb + 8);

    /* C-only rows: k = 0 (weights all 1) and, when even, k = L/2 (+-1) */
    {
        const double *w = p->Ct + (size_t)h * hs;
        for (int r = 0; r <= even; ++r, w += hs) {
            const int k = r ? L / 2 : 0;
            __m512d ar = x0r, ai = x0i;
            __m512d br = _mm512_setzero_pd(), bi = br;
            int j = 0;
            for (; j + 2 <= hu; j += 2) {
                __m512d c0 = _mm512_set1_pd(w[j]), c1 = _mm512_set1_pd(w[j + 1]);
                ar = _mm512_fmadd_pd(c0, _mm512_loadu_pd(ub + 16 * j), ar);
                ai = _mm512_fmadd_pd(c0, _mm512_loadu_pd(ub + 16 * j + 8), ai);
                br = _mm512_fmadd_pd(c1, _mm512_loadu_pd(ub + 16 * (j + 1)), br);
                bi = _mm512_fmadd_pd(c1, _mm512_loadu_pd(ub + 16 * (j + 1) + 8), bi);
            }
            if (j < hu) {
                __m512d c0 = _mm512_set1_pd(w[j]);
                ar = _mm512_fmadd_pd(c0, _mm512_loadu_pd(ub + 16 * j), ar);
                ai = _mm512_fmadd_pd(c0, _mm512_loadu_pd(ub + 16 * j + 8), ai);
            }
            _mm512_mask_storeu_pd(dr + (size_t)k * ostride, sm, _mm512_add_pd(ar, br));
            _mm512_mask_storeu_pd(di + (size_t)k * ostride, sm, _mm512_add_pd(ai, bi));
        }
    }

    /* paired rows k and L-k, 4 pairs at a time */
    int k = 1;
    for (; k + 4 <= h + 1; k += 4) {
        const double *c0 = p->Ct + (size_t)(k - 1) * hs, *c1 = c0 + hs;
        const double *c2 = c1 + hs, *c3 = c2 + hs;
        const double *s0 = p->St + (size_t)(k - 1) * hs, *s1 = s0 + hs;
        const double *s2 = s1 + hs, *s3 = s2 + hs;
        __m512d cr0 = _mm512_setzero_pd(), ci0 = cr0, sr0 = cr0, si0 = cr0;
        __m512d cr1 = cr0, ci1 = cr0, sr1 = cr0, si1 = cr0;
        __m512d cr2 = cr0, ci2 = cr0, sr2 = cr0, si2 = cr0;
        __m512d cr3 = cr0, ci3 = cr0, sr3 = cr0, si3 = cr0;
        for (int j = 0; j < h; ++j) {
            __m512d ur = _mm512_loadu_pd(ub + 16 * j);
            __m512d ui = _mm512_loadu_pd(ub + 16 * j + 8);
            __m512d wr = _mm512_loadu_pd(vb + 16 * j);
            __m512d wi = _mm512_loadu_pd(vb + 16 * j + 8);
            __m512d w;
            w = _mm512_set1_pd(c0[j]);
            cr0 = _mm512_fmadd_pd(w, ur, cr0);
            ci0 = _mm512_fmadd_pd(w, ui, ci0);
            w = _mm512_set1_pd(s0[j]);
            sr0 = _mm512_fmadd_pd(w, wr, sr0);
            si0 = _mm512_fmadd_pd(w, wi, si0);
            w = _mm512_set1_pd(c1[j]);
            cr1 = _mm512_fmadd_pd(w, ur, cr1);
            ci1 = _mm512_fmadd_pd(w, ui, ci1);
            w = _mm512_set1_pd(s1[j]);
            sr1 = _mm512_fmadd_pd(w, wr, sr1);
            si1 = _mm512_fmadd_pd(w, wi, si1);
            w = _mm512_set1_pd(c2[j]);
            cr2 = _mm512_fmadd_pd(w, ur, cr2);
            ci2 = _mm512_fmadd_pd(w, ui, ci2);
            w = _mm512_set1_pd(s2[j]);
            sr2 = _mm512_fmadd_pd(w, wr, sr2);
            si2 = _mm512_fmadd_pd(w, wi, si2);
            w = _mm512_set1_pd(c3[j]);
            cr3 = _mm512_fmadd_pd(w, ur, cr3);
            ci3 = _mm512_fmadd_pd(w, ui, ci3);
            w = _mm512_set1_pd(s3[j]);
            sr3 = _mm512_fmadd_pd(w, wr, sr3);
            si3 = _mm512_fmadd_pd(w, wi, si3);
        }
        if (even) { /* lone x_{L/2} column feeds C only */
            __m512d ur = _mm512_loadu_pd(ub + 16 * (hu - 1));
            __m512d ui = _mm512_loadu_pd(ub + 16 * (hu - 1) + 8);
            __m512d w;
            w = _mm512_set1_pd(c0[hu - 1]);
            cr0 = _mm512_fmadd_pd(w, ur, cr0);
            ci0 = _mm512_fmadd_pd(w, ui, ci0);
            w = _mm512_set1_pd(c1[hu - 1]);
            cr1 = _mm512_fmadd_pd(w, ur, cr1);
            ci1 = _mm512_fmadd_pd(w, ui, ci1);
            w = _mm512_set1_pd(c2[hu - 1]);
            cr2 = _mm512_fmadd_pd(w, ur, cr2);
            ci2 = _mm512_fmadd_pd(w, ui, ci2);
            w = _mm512_set1_pd(c3[hu - 1]);
            cr3 = _mm512_fmadd_pd(w, ur, cr3);
            ci3 = _mm512_fmadd_pd(w, ui, ci3);
        }
#define DM_COMB(T, KK)                                                              \
    do {                                                                            \
        __m512d t0 = _mm512_add_pd(x0r, cr##T), t1 = _mm512_add_pd(x0i, ci##T);     \
        _mm512_mask_storeu_pd(dr + (size_t)(KK)*ostride, sm,                        \
                              _mm512_add_pd(t0, si##T));                            \
        _mm512_mask_storeu_pd(di + (size_t)(KK)*ostride, sm,                        \
                              _mm512_sub_pd(t1, sr##T));                            \
        _mm512_mask_storeu_pd(dr + (size_t)(L - (KK)) * ostride, sm,                \
                              _mm512_sub_pd(t0, si##T));                            \
        _mm512_mask_storeu_pd(di + (size_t)(L - (KK)) * ostride, sm,                \
                              _mm512_add_pd(t1, sr##T));                            \
    } while (0)
        DM_COMB(0, k + 0);
        DM_COMB(1, k + 1);
        DM_COMB(2, k + 2);
        DM_COMB(3, k + 3);
    }
    for (; k <= h; ++k) { /* pair tail, one k at a time */
        const double *c0 = p->Ct + (size_t)(k - 1) * hs;
        const double *s0 = p->St + (size_t)(k - 1) * hs;
        __m512d cr0 = _mm512_setzero_pd(), ci0 = cr0, sr0 = cr0, si0 = cr0;
        for (int j = 0; j < h; ++j) {
            __m512d w = _mm512_set1_pd(c0[j]);
            cr0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(ub + 16 * j), cr0);
            ci0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(ub + 16 * j + 8), ci0);
            w = _mm512_set1_pd(s0[j]);
            sr0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(vb + 16 * j), sr0);
            si0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(vb + 16 * j + 8), si0);
        }
        if (even) {
            __m512d w = _mm512_set1_pd(c0[hu - 1]);
            cr0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(ub + 16 * (hu - 1)), cr0);
            ci0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(ub + 16 * (hu - 1) + 8), ci0);
        }
        DM_COMB(0, k);
    }
#undef DM_COMB
}

/* axis with contiguous inner index, interleaved source: stage 8 columns of all
 * L rows into the pencil-SoA block (gl_deint8), fold, folded kernel */
static void dm_axis_i2s(const double _Complex *src, double *dr, double *di,
                        fft3d_plan *p, size_t inner)
{
    const int L = p->L;
    double *pb = p->pb;
    for (size_t c = 0; c < inner; c += 8) {
        int cnt = (inner - c < 8) ? (int)(inner - c) : 8;
        __mmask8 sm = (cnt == 8) ? (__mmask8)0xFF : (__mmask8)((1u << cnt) - 1);
        __mmask8 m0 = (2 * cnt >= 8) ? (__mmask8)0xFF : (__mmask8)((1u << (2 * cnt)) - 1);
        __mmask8 m1 = (2 * cnt <= 8) ? (__mmask8)0 : (__mmask8)((1u << (2 * cnt - 8)) - 1);
        for (int j = 0; j < L; ++j) {
            const double *s = (const double *)(src + (size_t)j * inner + c);
            __m512d v0 = _mm512_maskz_loadu_pd(m0, s);
            __m512d v1 = _mm512_maskz_loadu_pd(m1, s + 8);
            __m512d re, im;
            gl_deint8(v0, v1, &re, &im);
            _mm512_storeu_pd(pb + 16 * j, re);
            _mm512_storeu_pd(pb + 16 * j + 8, im);
        }
        dm_fold8(pb, L, p->h, p->even, p->ub, p->vb);
        dm_kfold8(p, pb, p->ub, p->vb, dr + c, di + c, inner, sm);
    }
}

/* same, split source (used per x-plane for the y axis, inner = L) */
static void dm_axis_s2s(const double *sr, const double *si, double *dr, double *di,
                        fft3d_plan *p, size_t inner)
{
    const int L = p->L;
    double *pb = p->pb;
    for (size_t c = 0; c < inner; c += 8) {
        int cnt = (inner - c < 8) ? (int)(inner - c) : 8;
        __mmask8 sm = (cnt == 8) ? (__mmask8)0xFF : (__mmask8)((1u << cnt) - 1);
        for (int j = 0; j < L; ++j) {
            _mm512_storeu_pd(pb + 16 * j,
                             _mm512_maskz_loadu_pd(sm, sr + (size_t)j * inner + c));
            _mm512_storeu_pd(pb + 16 * j + 8,
                             _mm512_maskz_loadu_pd(sm, si + (size_t)j * inner + c));
        }
        dm_fold8(pb, L, p->h, p->even, p->ub, p->vb);
        dm_kfold8(p, pb, p->ub, p->vb, dr + c, di + c, inner, sm);
    }
}

/* fastest axis: 8 contiguous split pencils per block.  Stage via gl_tr8x8
 * (lane = pencil), fold, folded kernel into the ob block, transpose back and
 * interleave to the destination (gl_int8).  Replaces the r1 rowfast form,
 * which was load-bound (4 loads / 4 FMAs) and streamed the whole table per
 * SINGLE pencil; here every table broadcast feeds 8 pencils. */
static void dm_axis_z(const double *sr, const double *si, double _Complex *dst,
                      fft3d_plan *p, size_t nrows)
{
    const int L = p->L;
    double *pb = p->pb, *ob = p->ob;
    for (size_t r0 = 0; r0 < nrows; r0 += 8) {
        int rcnt = (nrows - r0 < 8) ? (int)(nrows - r0) : 8;
        /* stage: pb row j, lane t = pencil r0+t */
        for (int jc = 0; jc < L; jc += 8) {
            int jcnt = (L - jc < 8) ? (L - jc) : 8;
            __mmask8 jm = (jcnt == 8) ? (__mmask8)0xFF : (__mmask8)((1u << jcnt) - 1);
            __m512d re[8], im[8];
            for (int t = 0; t < 8; ++t) {
                if (t < rcnt) {
                    const double *rb = sr + (r0 + t) * (size_t)L + jc;
                    const double *ib = si + (r0 + t) * (size_t)L + jc;
                    re[t] = _mm512_maskz_loadu_pd(jm, rb);
                    im[t] = _mm512_maskz_loadu_pd(jm, ib);
                } else {
                    re[t] = _mm512_setzero_pd();
                    im[t] = _mm512_setzero_pd();
                }
            }
            gl_tr8x8(re);
            gl_tr8x8(im);
            for (int q = 0; q < 8; ++q) {
                _mm512_storeu_pd(pb + 16 * (jc + q), re[q]);
                _mm512_storeu_pd(pb + 16 * (jc + q) + 8, im[q]);
            }
        }
        dm_fold8(pb, L, p->h, p->even, p->ub, p->vb);
        dm_kfold8(p, pb, p->ub, p->vb, ob, ob + 8, 16, (__mmask8)0xFF);
        /* scatter: transpose ob back to pencil-major, interleave, store */
        for (int kc = 0; kc < L; kc += 8) {
            int kcnt = (L - kc < 8) ? (L - kc) : 8;
            __mmask8 m0 = (2 * kcnt >= 8) ? (__mmask8)0xFF
                                          : (__mmask8)((1u << (2 * kcnt)) - 1);
            __mmask8 m1 = (2 * kcnt <= 8) ? (__mmask8)0
                                          : (__mmask8)((1u << (2 * kcnt - 8)) - 1);
            __m512d re[8], im[8];
            for (int q = 0; q < 8; ++q) {
                re[q] = _mm512_loadu_pd(ob + 16 * (kc + q));
                im[q] = _mm512_loadu_pd(ob + 16 * (kc + q) + 8);
            }
            gl_tr8x8(re);
            gl_tr8x8(im);
            for (int t = 0; t < rcnt; ++t) {
                __m512d lo, hi;
                gl_int8(re[t], im[t], &lo, &hi);
                double *o = (double *)(dst + (r0 + t) * (size_t)L + kc);
                _mm512_mask_storeu_pd(o, m0, lo);
                _mm512_mask_storeu_pd(o + 8, m1, hi);
            }
        }
    }
}

#else /* scalar fallback: correctness on non-AVX-512 build hosts only */

/* folded scalar pencil transform via the same tables (h <= 63 at L <= 128) */
static void dm_pencil_sc(const fft3d_plan *p, const double *xr, const double *xi,
                         double *yr, double *yi)
{
    const int L = p->L, h = p->h, hu = p->hu, hs = p->hs, even = p->even;
    double ur[64], ui[64], vr[64], vi[64];
    for (int j = 1; j <= h; ++j) {
        ur[j - 1] = xr[j] + xr[L - j];
        ui[j - 1] = xi[j] + xi[L - j];
        vr[j - 1] = xr[j] - xr[L - j];
        vi[j - 1] = xi[j] - xi[L - j];
    }
    if (even) {
        ur[hu - 1] = xr[L / 2];
        ui[hu - 1] = xi[L / 2];
    }
    const double *w = p->Ct + (size_t)h * hs;
    for (int r = 0; r <= even; ++r, w += hs) {
        int k = r ? L / 2 : 0;
        double ar = xr[0], ai = xi[0];
        for (int j = 0; j < hu; ++j) {
            ar += w[j] * ur[j];
            ai += w[j] * ui[j];
        }
        yr[k] = ar;
        yi[k] = ai;
    }
    for (int k = 1; k <= h; ++k) {
        const double *cw = p->Ct + (size_t)(k - 1) * hs;
        const double *sw = p->St + (size_t)(k - 1) * hs;
        double cr = xr[0], ci = xi[0], sr = 0.0, si = 0.0;
        for (int j = 0; j < h; ++j) {
            cr += cw[j] * ur[j];
            ci += cw[j] * ui[j];
            sr += sw[j] * vr[j];
            si += sw[j] * vi[j];
        }
        if (even) {
            cr += cw[hu - 1] * ur[hu - 1];
            ci += cw[hu - 1] * ui[hu - 1];
        }
        yr[k] = cr + si;
        yi[k] = ci - sr;
        yr[L - k] = cr - si;
        yi[L - k] = ci + sr;
    }
}

static void dm_axis_i2s(const double _Complex *src, double *dr, double *di,
                        fft3d_plan *p, size_t inner)
{
    const int L = p->L;
    double xr[128], xi[128], yr[128], yi[128];
    for (size_t c = 0; c < inner; ++c) {
        for (int j = 0; j < L; ++j) {
            xr[j] = creal(src[(size_t)j * inner + c]);
            xi[j] = cimag(src[(size_t)j * inner + c]);
        }
        dm_pencil_sc(p, xr, xi, yr, yi);
        for (int k = 0; k < L; ++k) {
            dr[(size_t)k * inner + c] = yr[k];
            di[(size_t)k * inner + c] = yi[k];
        }
    }
}

static void dm_axis_s2s(const double *sr, const double *si, double *dr, double *di,
                        fft3d_plan *p, size_t inner)
{
    const int L = p->L;
    double xr[128], xi[128], yr[128], yi[128];
    for (size_t c = 0; c < inner; ++c) {
        for (int j = 0; j < L; ++j) {
            xr[j] = sr[(size_t)j * inner + c];
            xi[j] = si[(size_t)j * inner + c];
        }
        dm_pencil_sc(p, xr, xi, yr, yi);
        for (int k = 0; k < L; ++k) {
            dr[(size_t)k * inner + c] = yr[k];
            di[(size_t)k * inner + c] = yi[k];
        }
    }
}

static void dm_axis_z(const double *sr, const double *si, double _Complex *dst,
                      fft3d_plan *p, size_t nrows)
{
    const int L = p->L;
    double yr[128], yi[128];
    for (size_t r = 0; r < nrows; ++r) {
        dm_pencil_sc(p, sr + r * (size_t)L, si + r * (size_t)L, yr, yi);
        for (int k = 0; k < L; ++k)
            dst[r * (size_t)L + k] = yr[k] + I * yi[k];
    }
}

#endif /* __AVX512F__ */

/* one volume: interleaved src -> interleaved dst through the split scratch */
static void dm_vol_fft(fft3d_plan *p, const double _Complex *src,
                       double _Complex *dst)
{
    const int L = p->L;
    const size_t LL = (size_t)L * L;
    /* axis 0 (x): interleaved in -> split s1, inner = L^2 contiguous */
    dm_axis_i2s(src, p->s1r, p->s1i, p, LL);
    /* axis 1 (y): per x-plane, split s1 -> split s2, inner = L */
    for (int x = 0; x < L; ++x)
        dm_axis_s2s(p->s1r + (size_t)x * LL, p->s1i + (size_t)x * LL,
                    p->s2r + (size_t)x * LL, p->s2i + (size_t)x * LL,
                    p, (size_t)L);
    /* axis 2 (z): 8 contiguous split pencils per block -> interleaved out */
    dm_axis_z(p->s2r, p->s2i, dst, p, LL);
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    dm_placement(p, in, out);
    for (int b = 0; b < p->batch; ++b)
        dm_vol_fft(p, in + (size_t)b * p->V, out + (size_t)b * p->V);
}

/* graded map, one volume: state <- (z + c) / (1 + |z + c|).
 * Campaign-standard ladder (gen_pfa_small / gen_dense_prime s6): pair-compressed
 * |w|^2 via unpacklo/hi, 1e-300 guard, vrsqrt14 + 2 Newton for the sqrt, ONE
 * exact vdivpd per 8 complex (gen_dense_prime measured div beating an rcp
 * ladder on the standalone map pass; the OoO window hides it). */
static void dm_map_vol(const double _Complex *z, const double _Complex *c,
                       double _Complex *o, size_t V)
{
    const double *za = (const double *)z, *ca = (const double *)c;
    double *oa = (double *)o;
    size_t n = 2 * V, i = 0;
#if defined(__AVX512F__)
    const __m512d one = _mm512_set1_pd(1.0), half = _mm512_set1_pd(0.5);
    const __m512d th = _mm512_set1_pd(1.5), tiny = _mm512_set1_pd(1e-300);
    for (; i + 16 <= n; i += 16) {
        __m512d z0 = _mm512_add_pd(_mm512_loadu_pd(za + i), _mm512_loadu_pd(ca + i));
        __m512d z1 = _mm512_add_pd(_mm512_loadu_pd(za + i + 8),
                                   _mm512_loadu_pd(ca + i + 8));
        __m512d lo = _mm512_unpacklo_pd(z0, z1), hi = _mm512_unpackhi_pd(z0, z1);
        __m512d s = _mm512_fmadd_pd(lo, lo, _mm512_fmadd_pd(hi, hi, tiny));
        __m512d r = _mm512_rsqrt14_pd(s);
        __m512d e = _mm512_mul_pd(s, _mm512_mul_pd(r, r));
        r = _mm512_mul_pd(r, _mm512_fnmadd_pd(half, e, th));
        e = _mm512_mul_pd(s, _mm512_mul_pd(r, r));
        r = _mm512_mul_pd(r, _mm512_fnmadd_pd(half, e, th));
        __m512d d = _mm512_fmadd_pd(s, r, one); /* 1 + s*rsqrt(s) = 1 + |w| */
        __m512d rec = _mm512_div_pd(one, d);
        _mm512_storeu_pd(oa + i, _mm512_mul_pd(z0, _mm512_unpacklo_pd(rec, rec)));
        _mm512_storeu_pd(oa + i + 8, _mm512_mul_pd(z1, _mm512_unpackhi_pd(rec, rec)));
    }
#endif
    for (; i < n; i += 2) { /* exact scalar tail (V*2 % 16 != 0 at odd L) */
        double re = za[i] + ca[i], im = za[i + 1] + ca[i + 1];
        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
        oa[i] = re * sc;
        oa[i + 1] = im * sc;
    }
}

/* Own the graded m-step chain (fallback pays a memcpy + a scalar-sqrt map pass
 * per step).  Volume-major (the corpus consensus): each volume runs all m steps
 * while its working set is cache-hot.  state_b lives in final_out; step 1 reads
 * x0 directly, so nothing is ever copied. */
void fft3d_chain(fft3d_plan *p, const double _Complex *x0,
                 const double _Complex *c, double _Complex *final_out, int m)
{
    dm_placement(p, x0, final_out);
    for (int b = 0; b < p->batch; ++b) {
        const double _Complex *cb = c + (size_t)b * p->V;
        double _Complex *sb = final_out + (size_t)b * p->V;
        dm_vol_fft(p, x0 + (size_t)b * p->V, p->zt);
        dm_map_vol(p->zt, cb, sb, p->V);
        for (int s = 1; s < m; ++s) {
            dm_vol_fft(p, sb, p->zt);
            dm_map_vol(p->zt, cb, sb, p->V);
        }
    }
}

#endif /* GEN_LAYOUT_LIB_ONLY */
