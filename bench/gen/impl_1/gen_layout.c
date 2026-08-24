/* =============================================================================
 * GEN_LAYOUT -- the allocation & layout library layer (round gen_r1)
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
 *  5. gl_pad_stride            -- odd-line-count stride rounding (rotates all
 *     64 line phases across rows).  Use 4 when you know the stream geometry.
 *  6. gl_deint8 / gl_int8 / gl_tr8x8 / gl_pack8 / gl_unpack8 -- pencil/lane
 *     SoA converters (AVX-512): interleaved complex <-> split [site][2][8]
 *     blocks, one vpermt2pd per output vector, 8x8 transpose at 24 shuffles /
 *     64 doubles.  For gen_batchlane's 8-vol lanes and any pencil-SoA stage.
 *     [L8_fusedaxes bl8; L13_rader soa8; LITERATURE 04: split layout deletes
 *      permutations from the hot loop]
 *
 * THE DEMO ENTRY (when not GEN_LAYOUT_LIB_ONLY): a generic dense row-column
 * DFT, any 2 <= L <= 128, split-complex broadcast-FMA (the matrixsimd shape),
 * staged through pencil-SoA blocks -- deliberately the same O(L^4) algorithm
 * class as baseline_matrix so the delta IS vectorization + layout.  Knobs for
 * A/B on the node:
 *     -DGL_DEMO_PLAIN=1     posix_memalign scratch, no THP/stagger/placement
 *     -DGL_DEMO_NOPLACE=1   THP arena + static stagger, no dynamic placement
 * Not a contender at any size; it is the layer's living test bench and the
 * panel's any-L vectorized floor.
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

/* ---- 5. stride padding ---------------------------------------------------- */
/* Round a row/volume stride up to an ODD number of cache lines: consecutive
 * rows then rotate through all 64 line phases instead of repeating one.
 * When you know both streams' strides, prefer gl_best_lineshift4k: L17 ice_r8
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
 * DEMO ENTRY: generic dense matrixsimd row-column DFT exercising the layer.
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
    size_t V;
    gl_arena ar;
    double *Wr, *Wi; /* [L][Lp], W symmetric so rows serve both orientations */
    double *pb;      /* pencil-SoA staging block [L][2][8] */
    /* scratch slabs own GL_PAGE+GL_LINE of slack; s* are the placed bases */
    double *s1r_slab, *s1i_slab, *s2r_slab, *s2i_slab;
    double *s1r, *s1i, *s2r, *s2i;
    const void *pin, *pout; /* driver buffer pair the placement was derived for */
    void *plain[8];
    int nplain;
};

const char *fft3d_name(void) { return "gen_layout"; }
const char *fft3d_description(void)
{
    return "LIBRARY LAYER (scored by adoption): THP arenas, 4K stagger/collision-model placement, "
           "pencil SoA pack (adopt: #define GEN_LAYOUT_LIB_ONLY + #include gen_layout.c); "
           "entry=any-L dense matrixsimd demo of the layer";
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

static void dm_fill_w(double *Wr, double *Wi, int L, int Lp)
{
    const long double TWO_PI = 6.283185307179586476925286766559L;
    for (int k = 0; k < L; ++k)
        for (int j = 0; j < L; ++j) {
            long m = ((long)k * (long)j) % L;
            if (2 * m > L) m -= L; /* fold the angle into [-pi, pi] */
            long double th = -(TWO_PI * (long double)m) / (long double)L;
            Wr[(size_t)k * Lp + j] = (double)cosl(th);
            Wi[(size_t)k * Lp + j] = (double)sinl(th);
        }
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

    const size_t wbytes = (size_t)L * p->Lp * sizeof(double);
    const size_t pbbytes = (size_t)16 * p->Lp * sizeof(double) + 2 * GL_LINE;
    const size_t sbytes = p->V * sizeof(double) + GL_PAGE + 2 * GL_LINE;

#if !GL_DEMO_PLAIN
    size_t total = 2 * wbytes + pbbytes + 4 * sbytes + 16 * GL_PAGE + (64u << 10);
    if (gl_arena_init(&p->ar, total) != 0) {
        free(p);
        return NULL;
    }
#endif
    p->Wr = demo_alloc(p, wbytes);
    p->Wi = demo_alloc(p, wbytes);
    p->pb = demo_alloc(p, pbbytes);
    p->s1r_slab = demo_alloc(p, sbytes);
    p->s1i_slab = demo_alloc(p, sbytes);
    p->s2r_slab = demo_alloc(p, sbytes);
    p->s2i_slab = demo_alloc(p, sbytes);
    if (!p->Wr || !p->Wi || !p->pb || !p->s1r_slab || !p->s1i_slab ||
        !p->s2r_slab || !p->s2i_slab) {
        fft3d_destroy(p);
        return NULL;
    }
    /* until the first execute derives real placement: slab bases (these carry
     * the arena's stagger already, or the plain allocator's phases) */
    p->s1r = p->s1r_slab;
    p->s1i = p->s1i_slab;
    p->s2r = p->s2r_slab;
    p->s2i = p->s2i_slab;
    dm_fill_w(p->Wr, p->Wi, L, p->Lp);
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

/* out[k][c] = sum_j W[k][j] * x[j][c] over one pencil-SoA block pb[L][2][8],
 * masked-stored into split, strided output rows.  Split accumulators keep
 * every FMA chain at one FMA per j (latency 4 < issue 8 cycles at KB=4). */
static inline void dm_kmat8(const double *pb, const double *Wr, const double *Wi,
                            int L, int Lp, double *dr, double *di,
                            size_t ostride, __mmask8 sm)
{
    int k = 0;
    for (; k + 4 <= L; k += 4) {
        const double *w0 = Wr + (size_t)(k + 0) * Lp, *u0 = Wi + (size_t)(k + 0) * Lp;
        const double *w1 = Wr + (size_t)(k + 1) * Lp, *u1 = Wi + (size_t)(k + 1) * Lp;
        const double *w2 = Wr + (size_t)(k + 2) * Lp, *u2 = Wi + (size_t)(k + 2) * Lp;
        const double *w3 = Wr + (size_t)(k + 3) * Lp, *u3 = Wi + (size_t)(k + 3) * Lp;
        __m512d r10 = _mm512_setzero_pd(), r20 = r10, i10 = r10, i20 = r10;
        __m512d r11 = r10, r21 = r10, i11 = r10, i21 = r10;
        __m512d r12 = r10, r22 = r10, i12 = r10, i22 = r10;
        __m512d r13 = r10, r23 = r10, i13 = r10, i23 = r10;
        for (int j = 0; j < L; ++j) {
            __m512d xr = _mm512_loadu_pd(pb + 16 * j);
            __m512d xi = _mm512_loadu_pd(pb + 16 * j + 8);
            __m512d w, u;
            w = _mm512_set1_pd(w0[j]);
            u = _mm512_set1_pd(u0[j]);
            r10 = _mm512_fmadd_pd(w, xr, r10);
            r20 = _mm512_fmadd_pd(u, xi, r20);
            i10 = _mm512_fmadd_pd(w, xi, i10);
            i20 = _mm512_fmadd_pd(u, xr, i20);
            w = _mm512_set1_pd(w1[j]);
            u = _mm512_set1_pd(u1[j]);
            r11 = _mm512_fmadd_pd(w, xr, r11);
            r21 = _mm512_fmadd_pd(u, xi, r21);
            i11 = _mm512_fmadd_pd(w, xi, i11);
            i21 = _mm512_fmadd_pd(u, xr, i21);
            w = _mm512_set1_pd(w2[j]);
            u = _mm512_set1_pd(u2[j]);
            r12 = _mm512_fmadd_pd(w, xr, r12);
            r22 = _mm512_fmadd_pd(u, xi, r22);
            i12 = _mm512_fmadd_pd(w, xi, i12);
            i22 = _mm512_fmadd_pd(u, xr, i22);
            w = _mm512_set1_pd(w3[j]);
            u = _mm512_set1_pd(u3[j]);
            r13 = _mm512_fmadd_pd(w, xr, r13);
            r23 = _mm512_fmadd_pd(u, xi, r23);
            i13 = _mm512_fmadd_pd(w, xi, i13);
            i23 = _mm512_fmadd_pd(u, xr, i23);
        }
        _mm512_mask_storeu_pd(dr + (size_t)(k + 0) * ostride, sm, _mm512_sub_pd(r10, r20));
        _mm512_mask_storeu_pd(di + (size_t)(k + 0) * ostride, sm, _mm512_add_pd(i10, i20));
        _mm512_mask_storeu_pd(dr + (size_t)(k + 1) * ostride, sm, _mm512_sub_pd(r11, r21));
        _mm512_mask_storeu_pd(di + (size_t)(k + 1) * ostride, sm, _mm512_add_pd(i11, i21));
        _mm512_mask_storeu_pd(dr + (size_t)(k + 2) * ostride, sm, _mm512_sub_pd(r12, r22));
        _mm512_mask_storeu_pd(di + (size_t)(k + 2) * ostride, sm, _mm512_add_pd(i12, i22));
        _mm512_mask_storeu_pd(dr + (size_t)(k + 3) * ostride, sm, _mm512_sub_pd(r13, r23));
        _mm512_mask_storeu_pd(di + (size_t)(k + 3) * ostride, sm, _mm512_add_pd(i13, i23));
    }
    for (; k < L; ++k) {
        const double *w0 = Wr + (size_t)k * Lp, *u0 = Wi + (size_t)k * Lp;
        __m512d r1 = _mm512_setzero_pd(), r2 = r1, i1 = r1, i2 = r1;
        for (int j = 0; j < L; ++j) {
            __m512d xr = _mm512_loadu_pd(pb + 16 * j);
            __m512d xi = _mm512_loadu_pd(pb + 16 * j + 8);
            __m512d w = _mm512_set1_pd(w0[j]), u = _mm512_set1_pd(u0[j]);
            r1 = _mm512_fmadd_pd(w, xr, r1);
            r2 = _mm512_fmadd_pd(u, xi, r2);
            i1 = _mm512_fmadd_pd(w, xi, i1);
            i2 = _mm512_fmadd_pd(u, xr, i2);
        }
        _mm512_mask_storeu_pd(dr + (size_t)k * ostride, sm, _mm512_sub_pd(r1, r2));
        _mm512_mask_storeu_pd(di + (size_t)k * ostride, sm, _mm512_add_pd(i1, i2));
    }
}

/* axis with contiguous inner index, interleaved source: stage 8 columns of all
 * L rows into the pencil-SoA block (gl_deint8), then one kmat8 per chunk */
static void dm_axis_i2s(const double _Complex *src, double *dr, double *di,
                        const double *Wr, const double *Wi, double *pb,
                        int L, int Lp, size_t inner)
{
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
        dm_kmat8(pb, Wr, Wi, L, Lp, dr + c, di + c, inner, sm);
    }
}

/* same, split source (used per x-plane for the y axis, inner = L) */
static void dm_axis_s2s(const double *sr, const double *si, double *dr, double *di,
                        const double *Wr, const double *Wi, double *pb,
                        int L, int Lp, size_t inner)
{
    for (size_t c = 0; c < inner; c += 8) {
        int cnt = (inner - c < 8) ? (int)(inner - c) : 8;
        __mmask8 sm = (cnt == 8) ? (__mmask8)0xFF : (__mmask8)((1u << cnt) - 1);
        for (int j = 0; j < L; ++j) {
            _mm512_storeu_pd(pb + 16 * j,
                             _mm512_maskz_loadu_pd(sm, sr + (size_t)j * inner + c));
            _mm512_storeu_pd(pb + 16 * j + 8,
                             _mm512_maskz_loadu_pd(sm, si + (size_t)j * inner + c));
        }
        dm_kmat8(pb, Wr, Wi, L, Lp, dr + c, di + c, inner, sm);
    }
}

/* fastest axis, transposed form: out[k] = sum_j x_j * W[j][k] (W symmetric),
 * one split pencil in, one interleaved pencil out (gl_int8 at the stores).
 * j unrolled x2 with independent accumulator sets to break the FMA chain. */
static void dm_rowfast(const double *xr, const double *xi,
                       const double *Wr, const double *Wi,
                       int L, int Lp, double _Complex *dst)
{
    for (int kc = 0; kc < L; kc += 8) {
        int cnt = (L - kc < 8) ? (L - kc) : 8;
        __mmask8 m0 = (2 * cnt >= 8) ? (__mmask8)0xFF : (__mmask8)((1u << (2 * cnt)) - 1);
        __mmask8 m1 = (2 * cnt <= 8) ? (__mmask8)0 : (__mmask8)((1u << (2 * cnt - 8)) - 1);
        __m512d r1a = _mm512_setzero_pd(), r2a = r1a, i1a = r1a, i2a = r1a;
        __m512d r1b = r1a, r2b = r1a, i1b = r1a, i2b = r1a;
        int j = 0;
        for (; j + 2 <= L; j += 2) {
            __m512d w = _mm512_loadu_pd(Wr + (size_t)j * Lp + kc);
            __m512d u = _mm512_loadu_pd(Wi + (size_t)j * Lp + kc);
            __m512d br = _mm512_set1_pd(xr[j]), bi = _mm512_set1_pd(xi[j]);
            r1a = _mm512_fmadd_pd(br, w, r1a);
            r2a = _mm512_fmadd_pd(bi, u, r2a);
            i1a = _mm512_fmadd_pd(br, u, i1a);
            i2a = _mm512_fmadd_pd(bi, w, i2a);
            w = _mm512_loadu_pd(Wr + (size_t)(j + 1) * Lp + kc);
            u = _mm512_loadu_pd(Wi + (size_t)(j + 1) * Lp + kc);
            br = _mm512_set1_pd(xr[j + 1]);
            bi = _mm512_set1_pd(xi[j + 1]);
            r1b = _mm512_fmadd_pd(br, w, r1b);
            r2b = _mm512_fmadd_pd(bi, u, r2b);
            i1b = _mm512_fmadd_pd(br, u, i1b);
            i2b = _mm512_fmadd_pd(bi, w, i2b);
        }
        if (j < L) {
            __m512d w = _mm512_loadu_pd(Wr + (size_t)j * Lp + kc);
            __m512d u = _mm512_loadu_pd(Wi + (size_t)j * Lp + kc);
            __m512d br = _mm512_set1_pd(xr[j]), bi = _mm512_set1_pd(xi[j]);
            r1a = _mm512_fmadd_pd(br, w, r1a);
            r2a = _mm512_fmadd_pd(bi, u, r2a);
            i1a = _mm512_fmadd_pd(br, u, i1a);
            i2a = _mm512_fmadd_pd(bi, w, i2a);
        }
        __m512d re = _mm512_add_pd(_mm512_sub_pd(r1a, r2a), _mm512_sub_pd(r1b, r2b));
        __m512d im = _mm512_add_pd(_mm512_add_pd(i1a, i2a), _mm512_add_pd(i1b, i2b));
        __m512d lo, hi;
        gl_int8(re, im, &lo, &hi);
        double *o = (double *)(dst + kc);
        _mm512_mask_storeu_pd(o, m0, lo);
        _mm512_mask_storeu_pd(o + 8, m1, hi);
    }
}

#else /* scalar fallback: correctness on non-AVX-512 build hosts only */

static void dm_axis_i2s(const double _Complex *src, double *dr, double *di,
                        const double *Wr, const double *Wi, double *pb,
                        int L, int Lp, size_t inner)
{
    (void)pb;
    for (int k = 0; k < L; ++k)
        for (size_t c = 0; c < inner; ++c) {
            double ar = 0.0, ai = 0.0;
            for (int j = 0; j < L; ++j) {
                double w = Wr[(size_t)k * Lp + j], u = Wi[(size_t)k * Lp + j];
                double xr = creal(src[(size_t)j * inner + c]);
                double xi = cimag(src[(size_t)j * inner + c]);
                ar += w * xr - u * xi;
                ai += w * xi + u * xr;
            }
            dr[(size_t)k * inner + c] = ar;
            di[(size_t)k * inner + c] = ai;
        }
}

static void dm_axis_s2s(const double *sr, const double *si, double *dr, double *di,
                        const double *Wr, const double *Wi, double *pb,
                        int L, int Lp, size_t inner)
{
    (void)pb;
    for (int k = 0; k < L; ++k)
        for (size_t c = 0; c < inner; ++c) {
            double ar = 0.0, ai = 0.0;
            for (int j = 0; j < L; ++j) {
                double w = Wr[(size_t)k * Lp + j], u = Wi[(size_t)k * Lp + j];
                double xr = sr[(size_t)j * inner + c], xi = si[(size_t)j * inner + c];
                ar += w * xr - u * xi;
                ai += w * xi + u * xr;
            }
            dr[(size_t)k * inner + c] = ar;
            di[(size_t)k * inner + c] = ai;
        }
}

static void dm_rowfast(const double *xr, const double *xi,
                       const double *Wr, const double *Wi,
                       int L, int Lp, double _Complex *dst)
{
    for (int k = 0; k < L; ++k) {
        double ar = 0.0, ai = 0.0;
        for (int j = 0; j < L; ++j) {
            double w = Wr[(size_t)j * Lp + k], u = Wi[(size_t)j * Lp + k];
            ar += xr[j] * w - xi[j] * u;
            ai += xr[j] * u + xi[j] * w;
        }
        dst[k] = ar + I * ai;
    }
}

#endif /* __AVX512F__ */

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    const int L = p->L, Lp = p->Lp;
    const size_t LL = (size_t)L * L, V = p->V;
    dm_placement(p, in, out);

    for (int b = 0; b < p->batch; ++b) {
        const double _Complex *src = in + (size_t)b * V;
        double _Complex *dst = out + (size_t)b * V;
        /* axis 0 (x): interleaved in -> split s1, inner = L^2 contiguous */
        dm_axis_i2s(src, p->s1r, p->s1i, p->Wr, p->Wi, p->pb, L, Lp, LL);
        /* axis 1 (y): per x-plane, split s1 -> split s2, inner = L */
        for (int x = 0; x < L; ++x)
            dm_axis_s2s(p->s1r + (size_t)x * LL, p->s1i + (size_t)x * LL,
                        p->s2r + (size_t)x * LL, p->s2i + (size_t)x * LL,
                        p->Wr, p->Wi, p->pb, L, Lp, (size_t)L);
        /* axis 2 (z): per pencil, transposed form, split s2 -> interleaved out */
        for (size_t row = 0; row < LL; ++row)
            dm_rowfast(p->s2r + row * L, p->s2i + row * L,
                       p->Wr, p->Wi, L, Lp, dst + row * L);
    }
}

#endif /* GEN_LAYOUT_LIB_ONLY */
