/* =============================================================================
 * GEN_LAYOUT -- the allocation & layout library layer (round gen_r3)
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
 *  4c. gl_alias_drained4k -- NEW gen_r4: the in-flight gate the model was
 *     missing.  gen_dense_prime gen_r3 proved geometry alone over-predicts:
 *     alias pairs only bind while the stores still sit in the store buffer;
 *     enough unrelated uops between drain and loads and the fix is a wash.
 *     Gate your audit score through it before spending a window on a pitch.
 *  6. gl_deint8 / gl_int8 / gl_tr8x8 / gl_pack8 / gl_unpack8 -- pencil/lane
 *     SoA converters (AVX-512): interleaved complex <-> split [site][2][8]
 *     blocks, one vpermt2pd per output vector, 8x8 transpose at 24 shuffles /
 *     64 doubles.  For gen_batchlane's 8-vol lanes and any pencil-SoA stage.
 *     [L8_fusedaxes bl8; L13_rader soa8; LITERATURE 04: split layout deletes
 *      permutations from the hot loop]
 *  6b. gl_tr8x8_c2i -- NEW gen_r4: lane-major split block -> interleaved rows
 *     in ONE fused network (48 shuffles vs 64 for tr8x8 x2 + int8 x8); the
 *     store side of every bl8/soa8-style unpack, port-5 relief on Ice Lake.
 *  1c. gl_thp_bytes -- NEW gen_r11: THP coverage of any range, measured from
 *     /proc/self/smaps.  The scoring node runs THP=madvise on a 5.15 kernel
 *     (no MADV_COLLAPSE), so the driver's posix_memalign buffers are 4K-backed
 *     at every size -- verify a buffer you did not allocate before assuming
 *     TLB reach; re-home chain-hot state into a gl_map_huge arena when it
 *     reports cold (fft3d_chain below is the zero-copy worked example).
 *  7. gl_map8 / gl_map16 -- NEW gen_r5: the graded chain map w/(1+|w|) on
 *     8 (resp. 16) interleaved complex IN REGISTERS: pair-compressed |w|^2,
 *     rsqrt14 + 2 Newton, ONE exact vdivpd per call.  Fuse it at your
 *     engine's exit store and the separate map pass -- and its full state
 *     round trip -- disappears (gen_dense_prime r4 z-phase fusion,
 *     gen_pfa_large r4 ipp, gen_bluestein r4 per-block map all measured the
 *     shape).  gl_map16 is gen_dense_prime r4 item 3 built: one vdivpd per
 *     16 points via the reciprocal-product trick, ~1-2 ulp extra (gate
 *     margin is ~60 ulp/step); raced against gl_map8 in this entry so
 *     adopters get a measured verdict.
 *  6c. gl_tr8x8_ld -- NEW gen_r13 (promoted from the demo's gen_r8
 *     dm_tr8x8_ld): 8x8 double transpose straight from memory via
 *     VINSERTF64X4 (a pure load-port uop on SKX/ICX/SPR): 16 shuffles +
 *     16 loads per block vs gl_tr8x8's 24 + 8.  Full rows only; gate it
 *     off under kernels with < ~16-column sweeps (r8 boundary).
 *  6d. gl_pack8_ld -- NEW gen_r13: gl_pack8 staged through that network,
 *     32 shuffles + 32 loads per 8 sites vs 48 + 16 (-33% port 5).  The
 *     entry pack of a within-volume pencil-lane engine (round-13 B=1
 *     material: 8 z-pencils of ONE volume per lane group, stride = L).
 *  7b. gl_map8s / gl_map4s -- NEW gen_r13: the graded map in SPLIT
 *     (lane-SoA) form, ZERO shuffles, BIT-IDENTICAL per complex to
 *     gl_map8/gl_map4 (asserted in gl_selftest).  A B=1 pencil-lane chain
 *     packs once at step 1, runs every FFT+map step lane-resident, and
 *     unpacks only at the final step -- the map never forces the layout
 *     round trip that made B=1 fallback paths slow.
 *  6e. gl_tr8x8_st / gl_ldi8x8 / gl_sti8x8 / gl_unpack8_st -- NEW gen_r14,
 *     the execute()-reroute kit.  gl_tr8x8_st is gl_tr8x8_ld's store-side
 *     mirror: 16 shuffles + 16 stores per 8x8 block (8 ymm stores + 8
 *     VEXTRACTF64X4-to-memory, which is STORE-PORT-ONLY on ICX/SPR --
 *     llvm-mca icelake-server: zero p5) vs the tr8x8 route's 24 + 8.
 *     gl_ldi8x8/gl_sti8x8 move 8 sites x 8 interleaved streams straight
 *     between driver memory and split site-major REGISTERS (lane = stream,
 *     gl_pack8's block contract): fuse the pack into pass-1 loads and the
 *     unpack into last-pass stores and a single-shot fft3d_execute() stops
 *     paying the pack/unpack memory round trip (~1 us around a ~2 us
 *     transform at L=10 -- gen_batchlane r13).  gl_unpack8_st is the
 *     standalone-unpack form of the same trade (-33% p5), r13's stride
 *     caveat attached.
 *
 * THE DEMO ENTRY (when not GEN_LAYOUT_LIB_ONLY): a generic FOLDED dense
 * row-column DFT, any 2 <= L <= 128, staged through pencil-SoA blocks.
 * gen_r2: adopted gen_dense_prime's conjugate-pair fold (u_j = x_j + x_{L-j},
 * v_j = x_j - x_{L-j}; C = cos-matrix * u, S = sin-matrix * v; X_k = C_k - iS_k,
 * X_{L-k} = C_k + iS_k) -- all constants REAL, ~4x fewer FMAs than the r1
 * unfolded form -- and replaced the load-bound axis-2 row kernel with a
 * transpose-staged 8-pencil block kernel (gl_tr8x8 both ways, broadcast tables
 * reused across 8 pencils).
 * gen_r5: the graded map is fused into the axis-2 exit (gl_map8/gl_map16 on
 * the registers gl_tr8x8_c2i just produced) and the chain runs IN PLACE --
 * the zt volume and the separate map pass are deleted (32 MB/step of DRAM
 * traffic at L=100).
 * gen_r6: even-L second-level fold -- cos(2*pi*k*(L/2-j)/L) = (-1)^k
 * cos(2*pi*k*j/L) (and -(-1)^k for sin), so outputs split by k-parity and
 * j folds again over (j, L/2-j): the kernel j-sweep runs over ~h/2 columns
 * (uo/vo blocks for odd k, ue/ve for even, x_{L/2} absorbed into two base
 * rows).  Halves the kernel FMA count at every even L; odd L unchanged.
 * gen_r7: third-level k-fold at 4 | L -- cos(2*pi*(L/2-k)j/L) =
 * (-1)^j cos(2*pi*k*j/L) (sin: -(-1)^j), and 4|L keeps k and L/2-k in the
 * same parity class, so with the second-fold columns PARITY-SORTED one
 * j-sweep with E/O accumulators yields four outputs (k, L-k, L/2-k, L/2+k):
 * the sweep halves again (quarter of r5).  Plus exit-map packing: kcnt=4
 * tails pair two pencils per gl_map8 call (zero shuffles), kcnt=2 tails
 * pack four (two vinsertf64x4) -- the r5 dead-lane divide residual.
 * gen_r8: (a) insert-load 8x8 transpose (dm_tr8x8_ld) in both axis-2
 * stagings -- VINSERTF64X4 zmm,zmm,m256 is a pure load-port uop on ICX, so
 * the 256-bit transpose stage moves off port 5 (16 shuffles + 16 loads vs
 * gl_tr8x8's 24 + 8); port 5 is the second FMA pipe, so this is kernel
 * relief at every L >= 8.  (b) kcnt=1 exit-map packing (L%8==1): 8
 * one-complex rows share one gl_map8 ladder (pack 6 shuffles, c via masked
 * vbroadcastf64x2 = load-port, exit by 128-bit extract stores).  (c) the
 * full-chunk exit path is a constant-bound unrolled loop so lo/hi stay in
 * registers (the r7 46-spill dm_exit8 diet).
 * gen_r3: (a) axis-1 y-pencils grouped GLOBALLY over (x,z) so vector groups
 * cross plane boundaries (two masked loads/stores) instead of burning dead
 * lanes per plane (L=25 wasted 28% of axis-1 lanes; L=10, 60%); (b) axis 2
 * trails axis 1 plane-by-plane through a 4-plane circular window whose plane
 * pitch is picked by the layer's own gl_alias_pairs4k model, so the second
 * scratch volume never round-trips DRAM; (c) axes 0/1 build the folded u/v
 * rows straight from source loads (no pb staging round trip).
 * Still O(L^4)/axis dense class by design: the panel's any-L vectorized floor
 * and the layer's living test bench, not a contender.  Knobs for A/B on the
 * node:
 *     -DGL_DEMO_PLAIN=1     posix_memalign scratch, no THP/stagger/placement
 *     -DGL_DEMO_NOPLACE=1   THP arena + static stagger, no dynamic placement
 *     -DGL_DEMO_COLLIDE=1   adversarial: scratch at out's exact page phase
 * ============================================================================= */

#ifndef GEN_LAYOUT_C_INCLUDED
#define GEN_LAYOUT_C_INCLUDED

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
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

/* ---- 1c. THP verification (NEW gen_r11: the brief's "smaps!" ask) ---------- */
/* Bytes of [p, p+len) actually backed by transparent huge pages, measured from
 * /proc/self/smaps (AnonHugePages per VMA, clamped to the overlap with the
 * queried range).  Returns -1 if smaps is unreadable.  WHY YOU WANT THIS:
 * a buffer you did not allocate (the driver's in/out/c) is only huge-backed if
 * the host's THP mode says so -- on the scoring node the mode is MADVISE, so
 * plain posix_memalign volumes are 4K-backed no matter their size, and a
 * kernel too old for MADV_COLLAPSE (< 6.1; scoring node is 5.15) cannot fix
 * them in place.  Verify, don't assume: call this at first execute() on the
 * driver's buffers and re-home your chain-hot state into a gl_map_huge arena
 * when it reports cold (the demo's fft3d_chain shows the zero-copy pattern).
 * Cost: one smaps parse, ~100 us -- create()/first-call money, not loop money. */
static inline long long gl_thp_bytes(const void *p, size_t len)
{
    FILE *f = fopen("/proc/self/smaps", "r");
    if (!f) return -1;
    const unsigned long long lo = (unsigned long long)(uintptr_t)p;
    const unsigned long long hi = lo + (unsigned long long)len;
    char line[256];
    unsigned long long vs = 0, ve = 0;
    long long huge = 0;
    int live = 0;
    while (fgets(line, sizeof line, f)) {
        unsigned long long a, b;
        if (line[0] != 'A' && sscanf(line, "%llx-%llx", &a, &b) == 2) {
            vs = a;
            ve = b;
            live = (vs < hi && ve > lo);
        } else if (live && !strncmp(line, "AnonHugePages:", 14)) {
            long long kb = 0;
            if (sscanf(line + 14, "%lld", &kb) == 1) {
                const unsigned long long o0 = vs > lo ? vs : lo;
                const unsigned long long o1 = ve < hi ? ve : hi;
                const long long ov = (long long)(o1 - o0);
                const long long hb = kb * 1024; /* per-VMA total: clamp to overlap */
                huge += hb < ov ? hb : ov;
            }
        }
    }
    fclose(f);
    return huge;
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

/* ---- 4c. the in-flight (drain) gate on the alias model (NEW gen_r4) ------- */
/* gen_dense_prime gen_r3 falsified a pure-geometry reading of the model: their
 * model-clean pitches (4.35 weighted pairs) never beat the model-worst r2
 * layout (240 pairs), because ~450 ops of z-GEMM sit between each block's
 * store drain and the next block's loads -- the stores retire before the
 * aliasing loads issue, so the replays exist but never bind.  gen_rader's
 * engine won 9.5% from the same pitch fix because their chunks run
 * back-to-back.  The missing term: a store can only alias-block a later load
 * while it still occupies the store buffer (Sunny Cove/ICX ~72 entries,
 * SKX/CLX ~56).  Once the line is owned the oldest store retires in ~1-2
 * cycles, so `sbuf` in-flight stores are gone after roughly 2*sbuf unrelated
 * uops.  Gate every gl_alias_pairs4k / gl_stream_audit4k score through this
 * with YOUR kernel's uop count between the store drain and the loads that
 * would alias it; if the gate zeroes the score, skip the pitch fix -- it will
 * measure as a wash or a loss (their table proves it).  First-order model:
 * A/B on the node stays the law. */
enum { GL_SBUF_ICX = 72, GL_SBUF_SKX = 56 };
static inline long gl_alias_drained4k(long pairs, long uops_between, int sbuf_entries)
{
    return (uops_between >= 2L * sbuf_entries) ? 0 : pairs;
}

/* ---- 5. stride padding ---------------------------------------------------- */
/* Round a row/volume stride up to an ODD number of cache lines: consecutive
 * rows then rotate through all 64 line phases instead of repeating one.
 * When you know both streams' strides, prefer gl_pick_pitch4k: L17 ice_r8
 * measured 73-line padding at 31 weighted collisions vs <=4 for 80 lines.
 * WHY it works, revised (gen_pow2 gen_r5): at L=32 their audit shows NO
 * store->load pair at equal addr mod 4K in any phase -- the odd-line pad's
 * value there is L1-SET UNIFORMITY (gcd(stride_lines, 64) = 1 walks all 64 L1
 * sets; a power-of-2 line stride hits 16 sets at 4x depth and loses to
 * conflict pressure from whatever else shares them).  Same rule, second
 * mechanism; both want gcd(stride_lines, 64) = 1. */
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
/* split 8x8 block (re[q]/im[q]: row q = site, lane t = stream/pencil) ->
 * interleaved complex rows per stream: lo[t] = sites 0..3 of stream t as
 * (re,im) pairs, hi[t] = sites 4..7.  48 shuffles for the 16 output zmm --
 * replaces gl_tr8x8(re) + gl_tr8x8(im) + 8x gl_int8 (64 shuffles) whenever a
 * lane-major block leaves for interleaved memory (NEW gen_r4; the axis-2
 * scatter of my demo and the store side of any bl8/soa8 unpack are this). */
static inline void gl_tr8x8_c2i(const __m512d re[8], const __m512d im[8],
                                __m512d lo[8], __m512d hi[8])
{
    __m512d pl[8], ph[8];
    for (int q = 0; q < 8; ++q) {
        pl[q] = _mm512_unpacklo_pd(re[q], im[q]); /* streams 0,2,4,6 */
        ph[q] = _mm512_unpackhi_pd(re[q], im[q]); /* streams 1,3,5,7 */
    }
/* 4x4 transpose of 128-bit lanes: W_j lane k = V_k lane j */
#define GL_T4(V0, V1, V2, V3, W0, W1, W2, W3)                                    \
    do {                                                                         \
        __m512d t0_ = _mm512_shuffle_f64x2(V0, V1, 0x88);                        \
        __m512d t1_ = _mm512_shuffle_f64x2(V2, V3, 0x88);                        \
        __m512d t2_ = _mm512_shuffle_f64x2(V0, V1, 0xdd);                        \
        __m512d t3_ = _mm512_shuffle_f64x2(V2, V3, 0xdd);                        \
        W0 = _mm512_shuffle_f64x2(t0_, t1_, 0x88);                               \
        W2 = _mm512_shuffle_f64x2(t0_, t1_, 0xdd);                               \
        W1 = _mm512_shuffle_f64x2(t2_, t3_, 0x88);                               \
        W3 = _mm512_shuffle_f64x2(t2_, t3_, 0xdd);                               \
    } while (0)
    GL_T4(pl[0], pl[1], pl[2], pl[3], lo[0], lo[2], lo[4], lo[6]);
    GL_T4(pl[4], pl[5], pl[6], pl[7], hi[0], hi[2], hi[4], hi[6]);
    GL_T4(ph[0], ph[1], ph[2], ph[3], lo[1], lo[3], lo[5], lo[7]);
    GL_T4(ph[4], ph[5], ph[6], ph[7], hi[1], hi[3], hi[5], hi[7]);
#undef GL_T4
}

/* ---- 6c. insert-load 8x8 transpose (PROMOTED gen_r13, from the demo's
 * gen_r8 dm_tr8x8_ld -- the round-13 within-volume pencil-lane staging is
 * exactly its shape, so the r8 "promote on first ask" offer is cashed).
 * out[c] lane r = row_r[c], rows `stride` doubles apart, straight from
 * memory.  VINSERTF64X4 zmm,zmm,m256 executes as a PURE LOAD-PORT uop on
 * SKX/ICX/SPR (no shuffle), so the 256-bit stage of the classic network
 * moves off port 5: 16 shuffles + 16 load uops per 8x8 block where
 * gl_tr8x8 spends 24 shuffles + 8 loads.  Port 5 is the second FMA pipe
 * on Ice Lake, so under a kernel this is FMA relief; standalone it lifts
 * the transpose's own p5 bound (48 -> 32 shuffles per 8 complex sites).
 * Two caveats travel with it (r8 record): (a) it reads exactly 64 B per
 * row -- full rows only, no masks, partial tails keep the register
 * network; (b) under a kernel whose j-sweep is shorter than ~16 columns
 * the doubled staging loads bind instead (measured +1% at L=12) -- gate
 * it like the demo's `ldt = (L >= 16)` when it feeds a kernel.  For pure
 * pack/unpack staging (the pencil-lane entry conversion) the trade is
 * p5 32 vs 48 per 8 sites with loads far off the critical path: use it
 * whenever rows are full. */
static inline void gl_tr8x8_ld(const double *b, size_t stride, __m512d out[8])
{
    const __m512i IA = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    const __m512i IB = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
    __m512d w[8];
    for (int r = 0; r < 4; ++r) {
        w[r] = _mm512_insertf64x4(
            _mm512_castpd256_pd512(_mm256_loadu_pd(b + (size_t)r * stride)),
            _mm256_loadu_pd(b + (size_t)(r + 4) * stride), 1);
        w[4 + r] = _mm512_insertf64x4(
            _mm512_castpd256_pd512(_mm256_loadu_pd(b + (size_t)r * stride + 4)),
            _mm256_loadu_pd(b + (size_t)(r + 4) * stride + 4), 1);
    }
    __m512d t0 = _mm512_unpacklo_pd(w[0], w[1]), t1 = _mm512_unpackhi_pd(w[0], w[1]);
    __m512d t2 = _mm512_unpacklo_pd(w[2], w[3]), t3 = _mm512_unpackhi_pd(w[2], w[3]);
    out[0] = _mm512_permutex2var_pd(t0, IA, t2);
    out[1] = _mm512_permutex2var_pd(t1, IA, t3);
    out[2] = _mm512_permutex2var_pd(t0, IB, t2);
    out[3] = _mm512_permutex2var_pd(t1, IB, t3);
    t0 = _mm512_unpacklo_pd(w[4], w[5]);
    t1 = _mm512_unpackhi_pd(w[4], w[5]);
    t2 = _mm512_unpacklo_pd(w[6], w[7]);
    t3 = _mm512_unpackhi_pd(w[6], w[7]);
    out[4] = _mm512_permutex2var_pd(t0, IA, t2);
    out[5] = _mm512_permutex2var_pd(t1, IA, t3);
    out[6] = _mm512_permutex2var_pd(t0, IB, t2);
    out[7] = _mm512_permutex2var_pd(t1, IB, t3);
}

/* ---- 6e. the execute()-reroute kit (NEW gen_r14) --------------------------- */
/* Round 14's seam is fft3d_execute() at B=1: the fast engines run lane-
 * resident (split [site][2][8]) and a SINGLE-SHOT call pays a full pack +
 * unpack MEMORY ROUND TRIP around the transform (gen_batchlane r13: ~1 us of
 * pack/unpack/copy around a ~2 us transform at L=10).  These primitives move
 * data straight between the driver's interleaved buffers and split site-major
 * REGISTERS, so an engine fuses the pack into its first pass's loads and the
 * unpack into its last pass's stores and the round trip disappears.
 *
 * gl_tr8x8_st -- gl_tr8x8_ld's store-side mirror.  v[c] lane r = m[r][c]
 * (i.e. v holds the COLUMNS of an 8x8 double block); stores the rows
 * m[r][0..8) at b + r*stride.  8 unpacks + 8 vpermt2pd build zmm whose
 * 256-bit halves are contiguous row quarters; those leave by 8 plain ymm
 * stores + 8 VEXTRACTF64X4-to-memory -- which is STORE-PORT-ONLY on
 * SKX/ICX/SPR, exactly like the load-side VINSERTF64X4 (llvm-mca
 * icelake-server: p4/p7/p8/p9, zero p5; the exit-side analogue my r8/r10
 * records parked is hereby built).  16 shuffles + 16 store uops per 8x8
 * block where the gl_tr8x8 route spends 24 + 8: -33% port 5 (the second FMA
 * pipe) for +8 store uops (p49 idles at 0.11-0.27/cycle in every measured
 * kernel -- r10 dashboard).  Writes exactly 64 B per row: full rows only,
 * partial tails keep the register network. */
static inline void gl_tr8x8_st(double *b, size_t stride, const __m512d v[8])
{
    const __m512i JA = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
    const __m512i JB = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);
    __m512d t0 = _mm512_unpacklo_pd(v[0], v[1]), t1 = _mm512_unpackhi_pd(v[0], v[1]);
    __m512d t2 = _mm512_unpacklo_pd(v[2], v[3]), t3 = _mm512_unpackhi_pd(v[2], v[3]);
    __m512d t4 = _mm512_unpacklo_pd(v[4], v[5]), t5 = _mm512_unpackhi_pd(v[4], v[5]);
    __m512d t6 = _mm512_unpacklo_pd(v[6], v[7]), t7 = _mm512_unpackhi_pd(v[6], v[7]);
    __m512d ea = _mm512_permutex2var_pd(t0, JA, t2); /* rows 0,2 cols 0:4 */
    __m512d eb = _mm512_permutex2var_pd(t4, JA, t6); /* rows 0,2 cols 4:8 */
    __m512d ec = _mm512_permutex2var_pd(t0, JB, t2); /* rows 4,6 cols 0:4 */
    __m512d ed = _mm512_permutex2var_pd(t4, JB, t6); /* rows 4,6 cols 4:8 */
    __m512d oa = _mm512_permutex2var_pd(t1, JA, t3); /* rows 1,3 cols 0:4 */
    __m512d ob = _mm512_permutex2var_pd(t5, JA, t7); /* rows 1,3 cols 4:8 */
    __m512d oc = _mm512_permutex2var_pd(t1, JB, t3); /* rows 5,7 cols 0:4 */
    __m512d od = _mm512_permutex2var_pd(t5, JB, t7); /* rows 5,7 cols 4:8 */
    _mm256_storeu_pd(b + 0 * stride, _mm512_castpd512_pd256(ea));
    _mm256_storeu_pd(b + 0 * stride + 4, _mm512_castpd512_pd256(eb));
    _mm256_storeu_pd(b + 2 * stride, _mm512_extractf64x4_pd(ea, 1));
    _mm256_storeu_pd(b + 2 * stride + 4, _mm512_extractf64x4_pd(eb, 1));
    _mm256_storeu_pd(b + 4 * stride, _mm512_castpd512_pd256(ec));
    _mm256_storeu_pd(b + 4 * stride + 4, _mm512_castpd512_pd256(ed));
    _mm256_storeu_pd(b + 6 * stride, _mm512_extractf64x4_pd(ec, 1));
    _mm256_storeu_pd(b + 6 * stride + 4, _mm512_extractf64x4_pd(ed, 1));
    _mm256_storeu_pd(b + 1 * stride, _mm512_castpd512_pd256(oa));
    _mm256_storeu_pd(b + 1 * stride + 4, _mm512_castpd512_pd256(ob));
    _mm256_storeu_pd(b + 3 * stride, _mm512_extractf64x4_pd(oa, 1));
    _mm256_storeu_pd(b + 3 * stride + 4, _mm512_extractf64x4_pd(ob, 1));
    _mm256_storeu_pd(b + 5 * stride, _mm512_castpd512_pd256(oc));
    _mm256_storeu_pd(b + 5 * stride + 4, _mm512_castpd512_pd256(od));
    _mm256_storeu_pd(b + 7 * stride, _mm512_extractf64x4_pd(oc, 1));
    _mm256_storeu_pd(b + 7 * stride + 4, _mm512_extractf64x4_pd(od, 1));
}

/* gl_ldi8x8 -- 8 sites x 8 INTERLEAVED complex streams (stream t at
 * src + t*stride elements, gl_pack8's convention) straight into split
 * site-major registers: re[s]/im[s] lane t = stream t, site s -- the
 * register form of one gl_pack8 [site][2][8] block, with the block's 16
 * stores and the engine's 16 reloads DELETED.  Two gl_tr8x8_ld calls (the
 * interleaved row halves ARE 8x8 double blocks whose columns alternate
 * re/im): 32 loads + 32 shuffles, zero stores.  Fuse at pass-1 entry with
 * stride = the pencil pitch (stride-L^2 slot pencils, stride-L y/z pencils:
 * any pitch works, rows are read 64 B at a time).  Reads exactly 8 full
 * sites per stream -- tails keep gl_pack8. */
static inline void gl_ldi8x8(const double _Complex *src, size_t stride,
                             __m512d re[8], __m512d im[8])
{
    const double *b = (const double *)src;
    __m512d q[8];
    gl_tr8x8_ld(b, 2 * stride, q);
    re[0] = q[0]; im[0] = q[1]; re[1] = q[2]; im[1] = q[3];
    re[2] = q[4]; im[2] = q[5]; re[3] = q[6]; im[3] = q[7];
    gl_tr8x8_ld(b + 8, 2 * stride, q);
    re[4] = q[0]; im[4] = q[1]; re[5] = q[2]; im[5] = q[3];
    re[6] = q[4]; im[6] = q[5]; re[7] = q[6]; im[7] = q[7];
}

/* gl_sti8x8 -- exact inverse: split site-major registers -> 8 interleaved
 * streams in memory, via two gl_tr8x8_st networks.  32 shuffles + 32
 * store-port uops per 8x8-site block; the gl_tr8x8_c2i + zmm-store exit
 * spends 48 shuffles + 16 stores for the same bytes, so this is -16 p5 uops
 * per block on top of deleting the unpack pass when fused at the last
 * pass's exit.  NOTE the r10 boundary still holds: a FUSED-MAP exit must
 * materialize interleaved zmm for gl_map8 anyway (use gl_tr8x8_c2i there,
 * or better: map in split form with gl_map8s FIRST, then exit through
 * this). */
static inline void gl_sti8x8(double _Complex *dst, size_t stride,
                             const __m512d re[8], const __m512d im[8])
{
    double *b = (double *)dst;
    __m512d q[8];
    q[0] = re[0]; q[1] = im[0]; q[2] = re[1]; q[3] = im[1];
    q[4] = re[2]; q[5] = im[2]; q[6] = re[3]; q[7] = im[3];
    gl_tr8x8_st(b, 2 * stride, q);
    q[0] = re[4]; q[1] = im[4]; q[2] = re[5]; q[3] = im[5];
    q[4] = re[6]; q[5] = im[6]; q[6] = re[7]; q[7] = im[7];
    gl_tr8x8_st(b + 8, 2 * stride, q);
}

/* ---- 7. the graded chain map, in registers (NEW gen_r5) -------------------- */
/* out = w / (1 + |w|) on 8 interleaved complex held as two zmm (z0 = complex
 * 0..3 as (re,im) pairs, z1 = complex 4..7).  Caller adds c first (w = z + c).
 * Campaign-standard ladder (gen_pfa_small / gen_dense_prime s6): pair-
 * compressed |w|^2 via unpacklo/hi (each lane is one complex's re resp. im,
 * so values never mix across complex), 1e-300 guard, rsqrt14 + 2 Newton,
 * ONE exact vdivpd per 8 complex.  Bit-identical to the r2-r4 standalone map
 * pass this layer shipped.  Fuse at your exit store: the separate map pass
 * and its full state round trip disappear. */
static inline void gl_map8(__m512d z0, __m512d z1, __m512d *o0, __m512d *o1)
{
    const __m512d one = _mm512_set1_pd(1.0), half = _mm512_set1_pd(0.5);
    const __m512d th = _mm512_set1_pd(1.5), tiny = _mm512_set1_pd(1e-300);
    __m512d lo = _mm512_unpacklo_pd(z0, z1), hi = _mm512_unpackhi_pd(z0, z1);
    __m512d s = _mm512_fmadd_pd(lo, lo, _mm512_fmadd_pd(hi, hi, tiny));
    __m512d r = _mm512_rsqrt14_pd(s);
    __m512d e = _mm512_mul_pd(s, _mm512_mul_pd(r, r));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(half, e, th));
    e = _mm512_mul_pd(s, _mm512_mul_pd(r, r));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(half, e, th));
    __m512d d = _mm512_fmadd_pd(s, r, one); /* 1 + s*rsqrt(s) = 1 + |w| */
    __m512d rec = _mm512_div_pd(one, d);
    *o0 = _mm512_mul_pd(z0, _mm512_unpacklo_pd(rec, rec));
    *o1 = _mm512_mul_pd(z1, _mm512_unpackhi_pd(rec, rec));
}

/* 16 complex, ONE vdivpd (gen_dense_prime gen_r4 next-step 3, built):
 * q = 1/(da*db); rec_a = q*db, rec_b = q*da.  Costs 3 extra vmulpd for one
 * saved vdivpd and adds ~1-2 ulp to the reciprocal (per-step budget is
 * 1.5e-14 ~ 60 ulp -- safe).  |w| <= ~L^3 * O(1) in the graded chain, so
 * da*db cannot overflow.  Only pays where the divider is the binder --
 * A/B it in place (knob GL_DEMO_MAP16 here; the verdict is in my record).
 * ADOPTION MAP after gen_r5 (read before racing any map variant blind):
 * gen_powp took exactly the reciprocal-product trick into split-complex
 * (divider ops 25->15 at L=25, -1.6%; their lanes are already distinct so
 * the pair-compress does not apply); gen_twiddle REJECTED pair-packed
 * ladders inside their tr8x8-bound scatter exit (+4 port-5 shuffles cost
 * more than ~11 saved FMA-class ops); gen_pfa_large's map_step_pair wins
 * -8..-14% on STANDALONE sequential map passes.  Boundary: pack/share the
 * ladder where the map is a standalone or FMA-bound pass; in a shuffle-bound
 * (port-5) fused exit use plain gl_map8. */
static inline void gl_map16(__m512d z0, __m512d z1, __m512d z2, __m512d z3,
                            __m512d *o0, __m512d *o1, __m512d *o2, __m512d *o3)
{
    const __m512d one = _mm512_set1_pd(1.0), half = _mm512_set1_pd(0.5);
    const __m512d th = _mm512_set1_pd(1.5), tiny = _mm512_set1_pd(1e-300);
    __m512d la = _mm512_unpacklo_pd(z0, z1), ha = _mm512_unpackhi_pd(z0, z1);
    __m512d lb = _mm512_unpacklo_pd(z2, z3), hb = _mm512_unpackhi_pd(z2, z3);
    __m512d sa = _mm512_fmadd_pd(la, la, _mm512_fmadd_pd(ha, ha, tiny));
    __m512d sb = _mm512_fmadd_pd(lb, lb, _mm512_fmadd_pd(hb, hb, tiny));
    __m512d ra = _mm512_rsqrt14_pd(sa), rb = _mm512_rsqrt14_pd(sb);
    __m512d ea = _mm512_mul_pd(sa, _mm512_mul_pd(ra, ra));
    __m512d eb = _mm512_mul_pd(sb, _mm512_mul_pd(rb, rb));
    ra = _mm512_mul_pd(ra, _mm512_fnmadd_pd(half, ea, th));
    rb = _mm512_mul_pd(rb, _mm512_fnmadd_pd(half, eb, th));
    ea = _mm512_mul_pd(sa, _mm512_mul_pd(ra, ra));
    eb = _mm512_mul_pd(sb, _mm512_mul_pd(rb, rb));
    ra = _mm512_mul_pd(ra, _mm512_fnmadd_pd(half, ea, th));
    rb = _mm512_mul_pd(rb, _mm512_fnmadd_pd(half, eb, th));
    __m512d da = _mm512_fmadd_pd(sa, ra, one), db = _mm512_fmadd_pd(sb, rb, one);
    __m512d q = _mm512_div_pd(one, _mm512_mul_pd(da, db));
    __m512d reca = _mm512_mul_pd(q, db), recb = _mm512_mul_pd(q, da);
    *o0 = _mm512_mul_pd(z0, _mm512_unpacklo_pd(reca, reca));
    *o1 = _mm512_mul_pd(z1, _mm512_unpackhi_pd(reca, reca));
    *o2 = _mm512_mul_pd(z2, _mm512_unpacklo_pd(recb, recb));
    *o3 = _mm512_mul_pd(z3, _mm512_unpackhi_pd(recb, recb));
}

/* ---- 7b. the graded map in SPLIT (lane-SoA) form (NEW gen_r13) ------------- */
/* w/(1+|w|) on 8 complex held as separate re/im zmm -- the natural
 * [site][2][8] lane form of the batch-lane and within-volume pencil-lane
 * engines (the round-13 B=1 material).  Same ladder as gl_map8 with the
 * pair-compress unpacks DELETED: in split form every lane already is one
 * complex, so s = re*re + (im*im + tiny) needs no shuffle and the exact
 * reciprocal applies directly.  ZERO shuffle uops per 8 complex, where
 * gl_map8 spends 4 (2 unpacks in, 2 on the reciprocal) -- those unpacks
 * exist only to serve interleaved data.  Values are BIT-IDENTICAL per
 * complex to gl_map8/gl_map4 (same ops on the same numbers; asserted in
 * gl_selftest, not assumed), so a lane-resident chain that maps with this
 * matches an interleaved exit bit for bit.  THE POINT for a B=1 pencil-lane
 * chain: pack once at step 1, run every FFT+map step entirely in lane form
 * (this map never forces a layout round trip), unpack once at the final
 * step -- the same zero-conversion steady state gen_batchlane runs across
 * volumes, now across chain steps.  Dead lanes in a masked tail are safe
 * arithmetic (tiny guard; garbage lanes at worst produce garbage the masked
 * store discards) but keep inputs finite: a NaN/inf lane stays NaN/inf. */
static inline void gl_map8s(__m512d re, __m512d im, __m512d *ore, __m512d *oim)
{
    const __m512d one = _mm512_set1_pd(1.0), half = _mm512_set1_pd(0.5);
    const __m512d th = _mm512_set1_pd(1.5), tiny = _mm512_set1_pd(1e-300);
    __m512d s = _mm512_fmadd_pd(re, re, _mm512_fmadd_pd(im, im, tiny));
    __m512d r = _mm512_rsqrt14_pd(s);
    __m512d e = _mm512_mul_pd(s, _mm512_mul_pd(r, r));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(half, e, th));
    e = _mm512_mul_pd(s, _mm512_mul_pd(r, r));
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(half, e, th));
    __m512d d = _mm512_fmadd_pd(s, r, one); /* 1 + s*rsqrt(s) = 1 + |w| */
    __m512d rec = _mm512_div_pd(one, d);
    *ore = _mm512_mul_pd(re, rec);
    *oim = _mm512_mul_pd(im, rec);
}
/* ---- 8. 4-lane (ymm) SoA converters + graded map (NEW gen_r9) -------------- */
/* Built to the PMU audit's avenue 4 and gen_planner gen_r8's "half-group G=4
 * for B=4" ask: L=50 runs at B=4 in the suite, where the 8-volume batch-lane
 * layout cannot run at all, and B%8 remainder volumes everywhere else fall
 * back to lane replication.  These are the 8-lane converters' exact 4-lane
 * (ymm) analogues: blocks are [site][2][4] (4 reals then 4 imags per site,
 * lane = stream/volume).  Port character on Ice Lake-SP: 256-bit FP dispatches
 * on ports 0 AND 1 (port 1 idles in every measured kernel -- PMU audit s4),
 * and the in-lane ymm unpacks run p1/p5 where every 512-bit shuffle is p5
 * only, so a ymm remainder/tail path co-issues with zmm main work instead of
 * competing with it.  gl_map4 needs AVX512VL for vrsqrt14/vpermt2 at ymm
 * width (present on ICX/CLX/SPR); everything has a scalar fallback. */
#if defined(__AVX512F__) && defined(__AVX512VL__)

/* two interleaved ymm (4 complex) -> split re, im ymm.  1 vpermt2pd each. */
static inline void gl_deint4(__m256d v0, __m256d v1, __m256d *re, __m256d *im)
{
    const __m256i IE = _mm256_setr_epi64x(0, 2, 4, 6);
    const __m256i IO = _mm256_setr_epi64x(1, 3, 5, 7);
    *re = _mm256_permutex2var_pd(v0, IE, v1);
    *im = _mm256_permutex2var_pd(v0, IO, v1);
}

/* split re, im ymm -> two interleaved ymm (4 complex). */
static inline void gl_int4(__m256d re, __m256d im, __m256d *lo, __m256d *hi)
{
    const __m256i IL = _mm256_setr_epi64x(0, 4, 1, 5);
    const __m256i IH = _mm256_setr_epi64x(2, 6, 3, 7);
    *lo = _mm256_permutex2var_pd(re, IL, im);
    *hi = _mm256_permutex2var_pd(re, IH, im);
}

/* in-register 4x4 double transpose, 8 shuffles (unpacks are p1/p5 at ymm). */
static inline void gl_tr4x4(__m256d r[4])
{
    __m256d t0 = _mm256_unpacklo_pd(r[0], r[1]), t1 = _mm256_unpackhi_pd(r[0], r[1]);
    __m256d t2 = _mm256_unpacklo_pd(r[2], r[3]), t3 = _mm256_unpackhi_pd(r[2], r[3]);
    r[0] = _mm256_permute2f128_pd(t0, t2, 0x20);
    r[1] = _mm256_permute2f128_pd(t1, t3, 0x20);
    r[2] = _mm256_permute2f128_pd(t0, t2, 0x31);
    r[3] = _mm256_permute2f128_pd(t1, t3, 0x31);
}

/* the graded chain map w/(1+|w|) on 4 interleaved complex held as two ymm
 * (z0 = complex 0,1 as (re,im) pairs, z1 = complex 2,3).  Identical ladder to
 * gl_map8 at half width: pair-compressed |w|^2 via unpacklo/hi, 1e-300 guard,
 * vrsqrt14 + 2 Newton, ONE vdivpd (ymm: 8-cycle tput vs zmm's 16 -- per
 * complex the divider cost is unchanged).  VRSQRT14PD's approximation is
 * width-independent on real hardware, so results are bit-identical per
 * complex to gl_map8; gl_selftest() ASSERTS that bit-equality rather than
 * assuming it (if a future part ever disagrees, create() fails loudly and
 * the adopter falls back to gl_map8). */
static inline void gl_map4(__m256d z0, __m256d z1, __m256d *o0, __m256d *o1)
{
    const __m256d one = _mm256_set1_pd(1.0), half = _mm256_set1_pd(0.5);
    const __m256d th = _mm256_set1_pd(1.5), tiny = _mm256_set1_pd(1e-300);
    __m256d lo = _mm256_unpacklo_pd(z0, z1), hi = _mm256_unpackhi_pd(z0, z1);
    __m256d s = _mm256_fmadd_pd(lo, lo, _mm256_fmadd_pd(hi, hi, tiny));
    __m256d r = _mm256_rsqrt14_pd(s);
    __m256d e = _mm256_mul_pd(s, _mm256_mul_pd(r, r));
    r = _mm256_mul_pd(r, _mm256_fnmadd_pd(half, e, th));
    e = _mm256_mul_pd(s, _mm256_mul_pd(r, r));
    r = _mm256_mul_pd(r, _mm256_fnmadd_pd(half, e, th));
    __m256d d = _mm256_fmadd_pd(s, r, one); /* 1 + |w| */
    __m256d rec = _mm256_div_pd(one, d);
    *o0 = _mm256_mul_pd(z0, _mm256_unpacklo_pd(rec, rec));
    *o1 = _mm256_mul_pd(z1, _mm256_unpackhi_pd(rec, rec));
}

/* split-form graded map at ymm width (NEW gen_r13): 4 complex as separate
 * re/im ymm, lane = complex -- the [site][2][4] form of gl_pack4 blocks and
 * the natural tail of a pencil-lane group count that is not a multiple of 8
 * (L=10: 100 pencils = 12 zmm groups + one 4-lane tail).  Zero shuffles;
 * bit-identical per complex to gl_map8s/gl_map8 (vrsqrt14 width-independence
 * asserted in gl_selftest, the gen_r9 doctrine); ymm vdivpd is 8-cycle tput
 * vs zmm's 16, so per-complex divider cost is unchanged, and 256-bit FP
 * dispatches on the otherwise-idle port 1. */
static inline void gl_map4s(__m256d re, __m256d im, __m256d *ore, __m256d *oim)
{
    const __m256d one = _mm256_set1_pd(1.0), half = _mm256_set1_pd(0.5);
    const __m256d th = _mm256_set1_pd(1.5), tiny = _mm256_set1_pd(1e-300);
    __m256d s = _mm256_fmadd_pd(re, re, _mm256_fmadd_pd(im, im, tiny));
    __m256d r = _mm256_rsqrt14_pd(s);
    __m256d e = _mm256_mul_pd(s, _mm256_mul_pd(r, r));
    r = _mm256_mul_pd(r, _mm256_fnmadd_pd(half, e, th));
    e = _mm256_mul_pd(s, _mm256_mul_pd(r, r));
    r = _mm256_mul_pd(r, _mm256_fnmadd_pd(half, e, th));
    __m256d d = _mm256_fmadd_pd(s, r, one); /* 1 + |w| */
    __m256d rec = _mm256_div_pd(one, d);
    *ore = _mm256_mul_pd(re, rec);
    *oim = _mm256_mul_pd(im, rec);
}
#endif /* __AVX512F__ && __AVX512VL__ */
#endif /* __AVX512F__ */

/* 4 interleaved-complex streams (stream t at src + t*stride elements) ->
 * lane-SoA blocks dst[site][2][4]: 4 reals then 4 imags per site, lane =
 * stream.  For gen_batchlane at B=4 (stream stride = L^3 gives 4-volume
 * lanes) and any ymm pencil/tail path; exact 4-lane analogue of gl_pack8. */
static inline void gl_pack4(const double _Complex *src, size_t stride,
                            double *dst, size_t nsites)
{
    size_t s = 0;
#if defined(__AVX512F__) && defined(__AVX512VL__)
    for (; s + 2 <= nsites; s += 2) {
        __m256d v0 = _mm256_loadu_pd((const double *)(src + 0 * stride + s));
        __m256d v1 = _mm256_loadu_pd((const double *)(src + 1 * stride + s));
        __m256d v2 = _mm256_loadu_pd((const double *)(src + 2 * stride + s));
        __m256d v3 = _mm256_loadu_pd((const double *)(src + 3 * stride + s));
        /* vt = [re_s, im_s, re_s+1, im_s+1] of stream t */
        __m256d r01 = _mm256_unpacklo_pd(v0, v1); /* r0s r1s | r0s' r1s' */
        __m256d i01 = _mm256_unpackhi_pd(v0, v1);
        __m256d r23 = _mm256_unpacklo_pd(v2, v3);
        __m256d i23 = _mm256_unpackhi_pd(v2, v3);
        _mm256_storeu_pd(dst + 8 * s, _mm256_permute2f128_pd(r01, r23, 0x20));
        _mm256_storeu_pd(dst + 8 * s + 4, _mm256_permute2f128_pd(i01, i23, 0x20));
        _mm256_storeu_pd(dst + 8 * s + 8, _mm256_permute2f128_pd(r01, r23, 0x31));
        _mm256_storeu_pd(dst + 8 * s + 12, _mm256_permute2f128_pd(i01, i23, 0x31));
    }
#endif
    for (; s < nsites; ++s)
        for (int t = 0; t < 4; ++t) {
            dst[8 * s + t] = creal(src[t * stride + s]);
            dst[8 * s + 4 + t] = cimag(src[t * stride + s]);
        }
}

/* exact inverse of gl_pack4 */
static inline void gl_unpack4(const double *src, double _Complex *dst,
                              size_t stride, size_t nsites)
{
    size_t s = 0;
#if defined(__AVX512F__) && defined(__AVX512VL__)
    for (; s + 2 <= nsites; s += 2) {
        __m256d ra = _mm256_loadu_pd(src + 8 * s);      /* r0 r1 r2 r3 (site s) */
        __m256d ia = _mm256_loadu_pd(src + 8 * s + 4);
        __m256d rb = _mm256_loadu_pd(src + 8 * s + 8);  /* site s+1 */
        __m256d ib = _mm256_loadu_pd(src + 8 * s + 12);
        /* stream-major pairs: [r_t(s), i_t(s), r_t(s+1), i_t(s+1)] */
        __m256d l01 = _mm256_unpacklo_pd(ra, ia); /* r0s i0s | r2s i2s */
        __m256d h01 = _mm256_unpackhi_pd(ra, ia); /* r1s i1s | r3s i3s */
        __m256d l23 = _mm256_unpacklo_pd(rb, ib);
        __m256d h23 = _mm256_unpackhi_pd(rb, ib);
        _mm256_storeu_pd((double *)(dst + 0 * stride + s),
                         _mm256_permute2f128_pd(l01, l23, 0x20));
        _mm256_storeu_pd((double *)(dst + 1 * stride + s),
                         _mm256_permute2f128_pd(h01, h23, 0x20));
        _mm256_storeu_pd((double *)(dst + 2 * stride + s),
                         _mm256_permute2f128_pd(l01, l23, 0x31));
        _mm256_storeu_pd((double *)(dst + 3 * stride + s),
                         _mm256_permute2f128_pd(h01, h23, 0x31));
    }
#endif
    for (; s < nsites; ++s)
        for (int t = 0; t < 4; ++t)
            dst[t * stride + s] = src[8 * s + t] + I * src[8 * s + 4 + t];
}

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

/* gl_pack8 through the insert-load transpose (NEW gen_r13): identical layout
 * contract, full 8-site chunks staged by two gl_tr8x8_ld calls -- 32 shuffles
 * + 32 load uops per 8 sites where gl_pack8's tr8x8 route spends 48 + 16.
 * The pack IS the entry cost of a within-volume pencil-lane engine (round-13
 * material: 8 z-pencils of one volume per lane group, stride = L), and a
 * standalone pack is p5-bound, so this is the -33% p5 form; the sub-8-site
 * tail falls back to gl_pack8 (4-site register network, then scalar).  Full
 * rows only inside the vector chunk -- reads exactly 64 B per stream row. */
static inline void gl_pack8_ld(const double _Complex *src, size_t stride,
                               double *dst, size_t nsites)
{
    size_t s = 0;
#if defined(__AVX512F__)
    const double *b = (const double *)src;
    for (; s + 8 <= nsites; s += 8) {
        __m512d q[8];
        gl_tr8x8_ld(b + 2 * s, 2 * stride, q);
        for (int i = 0; i < 8; ++i)
            _mm512_storeu_pd(dst + 16 * s + 8 * i, q[i]);
        gl_tr8x8_ld(b + 2 * s + 8, 2 * stride, q);
        for (int i = 0; i < 8; ++i)
            _mm512_storeu_pd(dst + 16 * s + 64 + 8 * i, q[i]);
    }
#endif
    if (s < nsites) gl_pack8(src + s, stride, dst + 16 * s, nsites - s);
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

/* gl_unpack8 through the extract-store network (NEW gen_r14): identical
 * layout contract, full 8-site chunks staged by two gl_tr8x8_st calls --
 * 32 shuffles + 32 store uops per 8 sites where gl_unpack8's tr8x8 route
 * spends 48 + 16 (per 8 sites): -33% port 5 for +16 store-port uops, the
 * exact mirror of gl_pack8_ld's trade on the entry side.  The unpack IS the
 * exit cost of a lane-resident engine's plain execute(); the r13 pack8_ld
 * lesson travels: the stride flipped the entry verdict between 160 B and
 * 192 B rows, so MEASURE AT YOUR L before adopting either form -- and
 * remember the real win is fusing the exit into the last pass (gl_sti8x8),
 * not choosing between standalone unpacks.  Sub-8 tails fall back. */
static inline void gl_unpack8_st(const double *src, double _Complex *dst,
                                 size_t stride, size_t nsites)
{
    size_t s = 0;
#if defined(__AVX512F__)
    double *b = (double *)dst;
    for (; s + 8 <= nsites; s += 8) {
        __m512d v[8];
        for (int i = 0; i < 8; ++i)
            v[i] = _mm512_loadu_pd(src + 16 * s + 8 * i);
        gl_tr8x8_st(b + 2 * s, 2 * stride, v);
        for (int i = 0; i < 8; ++i)
            v[i] = _mm512_loadu_pd(src + 16 * s + 64 + 8 * i);
        gl_tr8x8_st(b + 2 * s + 8, 2 * stride, v);
    }
#endif
    if (s < nsites) gl_unpack8(src + 16 * s, dst + s, stride, nsites - s);
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
    { /* gen_r13: gl_pack8_ld must reproduce gl_pack8's exact layout
       * (NS=11 exercises one 8-site insert-load chunk AND the fallback tail) */
        static double soa2[16 * NS];
        memset(soa2, 0, sizeof soa2);
        gl_pack8_ld(a, STR, soa2, NS);
        for (int i = 0; i < 16 * NS; ++i)
            if (soa2[i] != soa[i]) return 0;
    }
    { /* 4-lane converters (gen_r9): pack4 layout + unpack4 round trip */
        static double soa4[8 * NS];
        gl_pack4(a, STR, soa4, NS);
        for (int s = 0; s < NS; ++s)
            for (int t = 0; t < 4; ++t)
                if (soa4[8 * s + t] != (double)(t * 100 + s) ||
                    soa4[8 * s + 4 + t] != (double)(t * 100 + s) * 0.5)
                    return 0;
        memset(b, 0, sizeof b);
        gl_unpack4(soa4, b, STR, NS);
        for (int t = 0; t < 4; ++t)
            for (int s = 0; s < NS; ++s)
                if (b[t * STR + s] != a[t * STR + s]) return 0;
    }
#if defined(__AVX512F__)
    { /* gl_tr8x8_c2i must agree with the tr8x8 + int8 reference path */
        double buf[128], ref[128], out[128];
        __m512d re[8], im[8], lo[8], hi[8];
        for (int q = 0; q < 8; ++q) {
            for (int t = 0; t < 8; ++t) buf[8 * q + t] = (double)(q * 8 + t);
            for (int t = 0; t < 8; ++t) buf[64 + 8 * q + t] = -(double)(q * 8 + t);
            re[q] = _mm512_loadu_pd(buf + 8 * q);
            im[q] = _mm512_loadu_pd(buf + 64 + 8 * q);
        }
        __m512d tr[8], ti[8];
        for (int q = 0; q < 8; ++q) { tr[q] = re[q]; ti[q] = im[q]; }
        gl_tr8x8(tr);
        gl_tr8x8(ti);
        for (int t = 0; t < 8; ++t) {
            __m512d l, h;
            gl_int8(tr[t], ti[t], &l, &h);
            _mm512_storeu_pd(ref + 16 * t, l);
            _mm512_storeu_pd(ref + 16 * t + 8, h);
        }
        gl_tr8x8_c2i(re, im, lo, hi);
        for (int t = 0; t < 8; ++t) {
            _mm512_storeu_pd(out + 16 * t, lo[t]);
            _mm512_storeu_pd(out + 16 * t + 8, hi[t]);
        }
        for (int i = 0; i < 128; ++i)
            if (out[i] != ref[i]) return 0;
    }
    { /* gen_r13: gl_tr8x8_ld must equal gl_tr8x8 on the same 8x8 block
       * (memory rows, stride 16 doubles) */
        double m[8 * 16], t1[64], t2[64];
        for (int r = 0; r < 8; ++r)
            for (int c = 0; c < 16; ++c) m[16 * r + c] = 1000.0 * r + c;
        __m512d v[8], w[8];
        for (int r = 0; r < 8; ++r) v[r] = _mm512_loadu_pd(m + 16 * r);
        gl_tr8x8(v);
        gl_tr8x8_ld(m, 16, w);
        for (int r = 0; r < 8; ++r) {
            _mm512_storeu_pd(t1 + 8 * r, v[r]);
            _mm512_storeu_pd(t2 + 8 * r, w[r]);
        }
        for (int i = 0; i < 64; ++i)
            if (t1[i] != t2[i]) return 0;
        { /* gen_r14: gl_tr8x8_st must invert gl_tr8x8_ld (pure data
           * movement, bit-exact), same block and stride */
            double m2[8 * 16];
            memset(m2, 0, sizeof m2);
            gl_tr8x8_st(m2, 16, w);
            for (int r = 0; r < 8; ++r)
                for (int c = 0; c < 8; ++c)
                    if (m2[16 * r + c] != m[16 * r + c]) return 0;
        }
    }
    { /* gen_r14: register-form entry/exit.  gl_ldi8x8's lanes must equal
       * gl_pack8's [site][2][8] blocks (soa[] above), and gl_sti8x8 must
       * write the original interleaved rows back bit-exactly. */
        __m512d re[8], im[8];
        double lane[8];
        static double _Complex c2[8 * STR];
        gl_ldi8x8(a, STR, re, im);
        for (int s = 0; s < 8; ++s) {
            _mm512_storeu_pd(lane, re[s]);
            for (int t = 0; t < 8; ++t)
                if (lane[t] != soa[16 * s + t]) return 0;
            _mm512_storeu_pd(lane, im[s]);
            for (int t = 0; t < 8; ++t)
                if (lane[t] != soa[16 * s + 8 + t]) return 0;
        }
        memset(c2, 0, sizeof c2);
        gl_sti8x8(c2, STR, re, im);
        for (int t = 0; t < 8; ++t)
            for (int s = 0; s < 8; ++s)
                if (c2[t * STR + s] != a[t * STR + s]) return 0;
    }
    { /* gen_r14: gl_unpack8_st must reproduce gl_unpack8's output exactly
       * (NS=11 exercises one extract-store chunk AND the fallback tail) */
        static double _Complex b2[8 * STR];
        memset(b2, 0, sizeof b2);
        gl_unpack8_st(soa, b2, STR, NS);
        for (int t = 0; t < 8; ++t)
            for (int s = 0; s < NS; ++s)
                if (b2[t * STR + s] != a[t * STR + s]) return 0;
    }
    { /* gl_map8 within 5e-15 of the exact scalar map; gl_map16 within 1e-14
       * of gl_map8 (the reciprocal-product trick's 1-2 ulp) */
        double zin[16], o8[16], o16a[16], o16b[16];
        for (int i = 0; i < 8; ++i) {
            zin[2 * i] = 0.37 * (i + 1) - 1.1;
            zin[2 * i + 1] = -0.53 * (i + 1) + 2.3;
        }
        __m512d a0 = _mm512_loadu_pd(zin), a1 = _mm512_loadu_pd(zin + 8);
        __m512d b0, b1, c0, c1, c2, c3;
        gl_map8(a0, a1, &b0, &b1);
        _mm512_storeu_pd(o8, b0);
        _mm512_storeu_pd(o8 + 8, b1);
        gl_map16(a0, a1, a0, a1, &c0, &c1, &c2, &c3);
        _mm512_storeu_pd(o16a, c0);
        _mm512_storeu_pd(o16a + 8, c1);
        _mm512_storeu_pd(o16b, c2);
        _mm512_storeu_pd(o16b + 8, c3);
        for (int i = 0; i < 16; i += 2) {
            double re = zin[i], im = zin[i + 1];
            double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
            if (fabs(o8[i] - re * sc) > 5e-15 * fabs(re * sc)) return 0;
            if (fabs(o8[i + 1] - im * sc) > 5e-15 * fabs(im * sc)) return 0;
            if (fabs(o16a[i] - o8[i]) > 1e-14 * fabs(o8[i])) return 0;
            if (fabs(o16a[i + 1] - o8[i + 1]) > 1e-14 * fabs(o8[i + 1])) return 0;
            if (fabs(o16b[i] - o8[i]) > 1e-14 * fabs(o8[i])) return 0;
            if (fabs(o16b[i + 1] - o8[i + 1]) > 1e-14 * fabs(o8[i + 1])) return 0;
        }
        { /* gen_r13: gl_map8s (split lane form) must be BIT-IDENTICAL per
           * complex to gl_map8 -- same ladder, unpacks deleted.  o8[] still
           * holds gl_map8's interleaved output. */
            double rr[8], ii[8], mr[8], mi[8];
            for (int i = 0; i < 8; ++i) {
                rr[i] = zin[2 * i];
                ii[i] = zin[2 * i + 1];
            }
            __m512d sre = _mm512_loadu_pd(rr), sim = _mm512_loadu_pd(ii);
            __m512d pre, pim;
            gl_map8s(sre, sim, &pre, &pim);
            _mm512_storeu_pd(mr, pre);
            _mm512_storeu_pd(mi, pim);
            for (int i = 0; i < 8; ++i)
                if (mr[i] != o8[2 * i] || mi[i] != o8[2 * i + 1]) return 0;
#if defined(__AVX512VL__)
            { /* and gl_map4s == gl_map8s at half width (vrsqrt14 width-
               * independence asserted, the gen_r9 doctrine) */
                __m256d r4 = _mm256_loadu_pd(rr), i4 = _mm256_loadu_pd(ii);
                __m256d q0, q1;
                double m4r[4], m4i[4];
                gl_map4s(r4, i4, &q0, &q1);
                _mm256_storeu_pd(m4r, q0);
                _mm256_storeu_pd(m4i, q1);
                for (int i = 0; i < 4; ++i)
                    if (m4r[i] != mr[i] || m4i[i] != mi[i]) return 0;
            }
#endif /* __AVX512VL__ */
        }
#if defined(__AVX512VL__)
        { /* gl_map4 must be BIT-IDENTICAL per complex to gl_map8 (same
           * ladder at ymm width; vrsqrt14's approximation is asserted, not
           * assumed, to be width-independent), and the deint4/int4/tr4x4
           * networks must be exact.  o8[] still holds gl_map8's output. */
            double o4[16];
            __m256d y0 = _mm256_loadu_pd(zin), y1 = _mm256_loadu_pd(zin + 4);
            __m256d y2 = _mm256_loadu_pd(zin + 8), y3 = _mm256_loadu_pd(zin + 12);
            __m256d q0, q1;
            gl_map4(y0, y1, &q0, &q1);
            _mm256_storeu_pd(o4, q0);
            _mm256_storeu_pd(o4 + 4, q1);
            gl_map4(y2, y3, &q0, &q1);
            _mm256_storeu_pd(o4 + 8, q0);
            _mm256_storeu_pd(o4 + 12, q1);
            for (int i = 0; i < 16; ++i)
                if (o4[i] != o8[i]) return 0;
            __m256d re4, im4, lo4, hi4;
            gl_deint4(y0, y1, &re4, &im4);
            gl_int4(re4, im4, &lo4, &hi4);
            double rt[8];
            _mm256_storeu_pd(rt, lo4);
            _mm256_storeu_pd(rt + 4, hi4);
            for (int i = 0; i < 8; ++i)
                if (rt[i] != zin[i]) return 0;
            __m256d tr[4];
            tr[0] = y0;
            tr[1] = y1;
            tr[2] = y2;
            tr[3] = y3;
            gl_tr4x4(tr);
            gl_tr4x4(tr); /* involution: twice = identity */
            double tt[16];
            for (int i = 0; i < 4; ++i) _mm256_storeu_pd(tt + 4 * i, tr[i]);
            for (int i = 0; i < 16; ++i)
                if (tt[i] != zin[i]) return 0;
        }
#endif /* __AVX512VL__ */
    }
#endif
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
/* A/B control: disable the r3 packed-lane + trailing-window y/z sweep (falls
 * back to the r2 per-plane axis-1 + full-volume axis-2 through a full s2) */
#ifndef GL_DEMO_NOFUSE
#define GL_DEMO_NOFUSE 0
#endif
/* A/B controls, gen_r4: disable the non-temporal stores on DRAM-resident
 * volumes / the software prefetch in the fold loads */
#ifndef GL_DEMO_NONT
#define GL_DEMO_NONT 0
#endif
#ifndef GL_DEMO_NOPF
#define GL_DEMO_NOPF 0
#endif
/* A/B controls, gen_r5: NOMAPFUSE reverts the chain to the r4 shape (FFT into
 * a zt volume, separate map pass zt+c -> state); MAP16 swaps the fused exit's
 * gl_map8 for gl_map16 (one vdivpd per 16 complex, gen_dense_prime r4 ask) */
#ifndef GL_DEMO_NOMAPFUSE
#define GL_DEMO_NOMAPFUSE 0
#endif
#ifndef GL_DEMO_MAP16
#define GL_DEMO_MAP16 0
#endif
/* A/B control, gen_r6: disable the even-L second-level fold (k-parity split
 * over j <-> L/2-j; halves the kernel j-sweep) and run the r5 kernel */
#ifndef GL_DEMO_NOEVEN
#define GL_DEMO_NOEVEN 0
#endif
/* A/B control, gen_r7: disable the third-level k-fold (quad kernel, 4 | L;
 * k and L/2-k share one j-sweep) and run the r6 even kernel */
#ifndef GL_DEMO_NOQUAD
#define GL_DEMO_NOQUAD 0
#endif
/* A/B controls, gen_r8: NOLDT disables the insert-load 8x8 transpose in the
 * axis-2 staging (falls back to gl_tr8x8); NOK1 disables the kcnt=1 exit-map
 * packing; NOX8 disables the constant-bound unrolled full-chunk exit path */
#ifndef GL_DEMO_NOLDT
#define GL_DEMO_NOLDT 0
#endif
#ifndef GL_DEMO_NOK1
#define GL_DEMO_NOK1 0
#endif
#ifndef GL_DEMO_NOX8
#define GL_DEMO_NOX8 0
#endif
/* A/B control, gen_r9: revert the kcnt=2 exit-map tail from the ymm gl_map4
 * form (ports 0/1, zero shuffles) to the r7 zmm insert/extract packing */
#ifndef GL_DEMO_NOMAP4
#define GL_DEMO_NOMAP4 0
#endif
/* gen_r12: OPT-IN class-major quad-kernel table rows (see dm_rq2).  Measured
 * a NULL at L=40 and ~-0.7% at L=100 (2 pairs) -- the accumulator sweep is
 * latency-TOLERANT to its L2-resident table stream, so consumption-order
 * relayout buys nothing; default stays the r7-r11 k-order. */
#ifndef GL_DEMO_TORD
#define GL_DEMO_TORD 0
#endif
/* gen_r12 A/B control: force the fold prefetches back to T0 everywhere */
#ifndef GL_DEMO_NOPF1
#define GL_DEMO_NOPF1 0
#endif
/* gen_r10: the m4t boundary as a tunable (the r9 record promised the race
 * layer a per-host flip; this is it).  m4t = (L <= GL_M4T_MAX); 0 disables
 * like GL_DEMO_NOMAP4, large values force the ymm tail everywhere.
 * DEFAULT 0 -- the gen_r10 ICX verdict: the ymm tail loses on Ice Lake at
 * BOTH L=10 (5.16/4.94 vs 5.05/4.91, two interleaved pairs) and L=50
 * (+0.7% forced), where SPR-advisory had it winning at 10.  Score is ICX;
 * SPR hosts want -DGL_M4T_MAX=16 (r9 record). */
#ifndef GL_M4T_MAX
#define GL_M4T_MAX 0
#endif
/* NT stores turn on when one split scratch component exceeds this (bytes):
 * default 4 MiB per component = 8 MiB split pair, i.e. only sizes whose
 * streams cannot live in L2/L3 anyway (L=100 in the suite).  Below it, NT
 * would evict data a later pass reads back from cache. */
#ifndef GL_NT_MIN_BYTES
#define GL_NT_MIN_BYTES (4u << 20)
#endif
/* gen_r11: re-home the chain state (and c) into THP arena volumes when the
 * driver's buffers are measured 4K-backed (gl_thp_bytes on final_out at the
 * first chain call) and the volume is big enough for TLB reach to matter.
 * The state re-home is ZERO-COPY: step 1 already reads x0 directly and the
 * LAST step's axis-2 exit writes final_out directly, so only steps 1..m-1
 * touch the arena volume.  The c re-home costs one volume copy per chain
 * call (read c + write cv once vs m strided re-reads through 4K pages).
 * GL_DEMO_NOREHOME=1 disables both (the A/B arm); GL_DEMO_NOCV=1 keeps the
 * state re-home but reads c in place. */
#ifndef GL_DEMO_NOREHOME
#define GL_DEMO_NOREHOME 0
#endif
#ifndef GL_DEMO_NOCV
#define GL_DEMO_NOCV 0
#endif
#ifndef GL_REHOME_MIN_BYTES
#define GL_REHOME_MIN_BYTES (8u << 20)
#endif

struct fft3d_plan {
    int L, Lp, batch;
    int h;    /* floor((L-1)/2): number of (k, L-k) output pairs / (j, L-j) folds */
    int hu;   /* u rows: h, +1 lone x_{L/2} row when L is even */
    int hs;   /* table row stride (hu padded to 8) */
    int even; /* L % 2 == 0 */
    /* gen_r6, even-L second-level fold (k-parity split over j <-> L/2-j) */
    int evenk; /* the halved kernel is active (even L >= 8, AVX-512 build) */
    int h2;    /* second-fold pairs (j, L/2-j): h/2 */
    int he2;   /* u/v block rows: h2, +1 lone j=L/4 row when 4 | L */
    int hs2;   /* Ct2/St2 row stride (he2 padded to 8) */
    /* gen_r7, third-level k-fold (4 | L): rows k and L/2-k share one j-sweep
     * via the column sign (-1)^j; columns are PARITY-SORTED (odd-j run first)
     * in Ct2/St2 AND the u/v blocks so the sign routing is free */
    int quadk; /* the quad kernel is active (evenk and 4 | L) */
    int no2;   /* length of the odd-j column run (0..no2-1; even j follow) */
    int nt0;  /* NT stores in the axis-0 kernel (rows 64B-aligned, big volume) */
    int nt2;  /* NT stores axis-2 -> dst and in the map (dst rows aligned)     */
    int pf1;  /* fold-source prefetch hint T1 (DRAM-resident source; gen_r12) */
    int ldt;  /* gen_r8: insert-load transpose in the axis-2 staging (L >= 16;
                 below that the kernel sweep is too short to hide the doubled
                 staging loads and the port-5 relief has nothing to relieve) */
    int m4t;  /* gen_r9: kcnt=2 exit-map tail as ymm gl_map4 ladders (ports
                 0/1, zero shuffles).  Pays only where the tail is a large
                 fraction of exit chunks (L <= 16: measured win at L=10; at
                 L=50 the doubled ladder uops lose ~1.5-2%) */
    size_t V;
    size_t wpitch; /* doubles per plane slot of the 4-plane y->z window (0 = old path) */
    gl_arena ar;
    double *Ct; /* [h + 1 + even][hs]: rows k=1..h, then k=0, then k=L/2  */
    double *St; /* [max(h,1)][hs]:    rows k=1..h                         */
    double *Ct2, *St2; /* evenk only, [h + 2][hs2]: rows k=1..h, k=0, k=L/2 */
    double *uob, *vob; /* evenk only: odd-parity folded rows [he2][2][8]  */
    double *ebb;       /* evenk only: base rows e = x0+x_{L/2}, o = x0-x_{L/2} */
    double *pb; /* pencil staging block [Lp][2][8]                        */
    double *ob; /* axis-1/2 output staging block [Lp][2][8]               */
    double *ub; /* folded u rows [hu][2][8]                               */
    double *vb; /* folded v rows [h][2][8]                                */
    double _Complex *zt; /* r4-shape chain A/B only (GL_DEMO_NOMAPFUSE);
                            NULL in the default fused build                 */
    /* gen_r11: THP re-homed chain state / c (arena volumes; NULL = gated off) */
    double _Complex *stv, *cv;
    const void *rh_in, *rh_out; /* driver pair the smaps verdict was taken for */
    int rh_use;                 /* verdict: driver state buffer is 4K-backed   */
    /* scratch slabs own GL_PAGE+GL_LINE of slack; s* are the placed bases */
    double *s1r_slab, *s1i_slab, *s2r_slab, *s2i_slab;
    double *s1r, *s1i, *s2r, *s2i;
    const void *pin, *pout; /* driver buffer pair the placement was derived for */
    void *plain[18];
    int nplain;
};

const char *fft3d_name(void) { return "gen_layout"; }
const char *fft3d_description(void)
{
    return "LIBRARY LAYER (scored by adoption): THP arenas, 4K stagger/collision-model "
           "placement + stream audit & measured pitch picker, pencil SoA pack "
           "(adopt: #define GEN_LAYOUT_LIB_ONLY + #include gen_layout.c); "
           "entry=any-L conjugate-pair-folded dense matrixsimd demo of the layer, "
           "r3: packed cross-plane axis-1 lanes + trailing axis-2 through a "
           "4-plane collision-picked window; r4: fold-load software prefetch "
           "(~L row streams beat the L2 streamer), NT full-line stores on "
           "DRAM-resident volumes, fused transpose+interleave scatter; "
           "r5: graded map fused into the axis-2 exit (gl_map8/gl_map16 "
           "in-register map primitives), chain in place, zt volume deleted; "
           "r6: even-L second-level fold (k-parity split over j<->L/2-j, "
           "halves the kernel j-sweep at every even L); "
           "r7: third-level k-fold at 4|L (rows k and L/2-k share one j-sweep "
           "via the column sign (-1)^j, parity-sorted columns; halves the "
           "sweep again) + exit-map packing for kcnt 4/2 tail chunks; "
           "r8: insert-load 8x8 transpose in the axis-2 staging (VINSERTF64X4 "
           "from memory is a load-port uop: 16 shuffles vs 24, port-5/FMA "
           "relief), kcnt=1 exit-map packing (8 rows one ladder), unrolled "
           "constant-index full-chunk exit (spill diet); "
           "r9: 4-lane (ymm) SoA library family gl_deint4/int4/tr4x4/pack4/"
           "unpack4 + gl_map4 (PMU-audit avenue 4 / gen_planner G=4 ask: "
           "unlocks batch-lane layout at B=4 and B%8 remainder lanes; ymm FP "
           "dispatches on the otherwise-idle port 1), dogfooded in the kcnt=2 "
           "exit-map tail (zero-shuffle ymm ladders, bit-identical to "
           "gl_map8; plan-gated m4t = L<=GL_M4T_MAX, r10 default 0: the ICX "
           "verdict is the zmm packing wins on the scoring host -- SPR "
           "builds want -DGL_M4T_MAX=16); "
           "r10: counter-directed audit round -- PMU dashboard measured "
           "(25/32 run AT the node's ~2.1 uops/cycle dispatch cap, loads "
           "the largest port class; 50/100 traffic-bound), m4t ICX A/B "
           "banked as a deterministic plan gate + race-flippable knob; "
           "r11 (all hands on L=100): gl_thp_bytes smaps THP verification "
           "(the brief's layout ask; finding: THP=madvise on the scoring "
           "node leaves the driver's 32 MB buffers 4K-backed, kernel 5.15 "
           "has no MADV_COLLAPSE) + zero-copy chain-state re-home into the "
           "THP arena (last step exits to final_out directly; c staged once "
           "per call), gated by the measured smaps verdict per buffer pair; "
           "r12: T1 fold-source prefetch on DRAM-resident sources (LFB vs "
           "L2-superqueue; -1.3..2.2% at L=100) + differential-counter "
           "protocol; r13 (B=1 small-L round): pencil-lane kit for adopters "
           "-- gl_tr8x8_ld promoted (insert-load 8x8 transpose, load-port "
           "VINSERTF64X4), gl_pack8_ld (-33% port-5 entry pack, stride=L "
           "gives 8 z-pencils of one volume per lane group), gl_map8s/"
           "gl_map4s (graded map in split lane form, zero shuffles, "
           "bit-identical to gl_map8: a B=1 chain stays lane-resident for "
           "all m steps, packing once and unpacking once); "
           "r14 (execute()-reroute round): register-form entry/exit kit -- "
           "gl_tr8x8_st (extract-store 8x8 transpose: VEXTRACTF64X4 to "
           "memory is store-port-only on ICX/SPR, 16 shuffles + 16 stores "
           "vs 24 + 8), gl_ldi8x8/gl_sti8x8 (8 sites x 8 interleaved "
           "streams <-> split site-major registers, zero memory round "
           "trip: fuse pack into pass-1 loads and unpack into last-pass "
           "stores so single-shot execute() stops paying the B=1 "
           "pack/unpack sandwich), gl_unpack8_st (standalone -33% p5 "
           "unpack)";
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

/* exact (cos, sin)(2*pi*m/L), same reduction as dm_fill_row */
static void dm_cis(long m, int L, double *c, double *s)
{
    const long double TWO_PI = 6.283185307179586476925286766559L;
    m %= L;
    if (m == 0) {
        *c = 1.0;
        *s = 0.0;
    } else if (2 * m == L) {
        *c = -1.0;
        *s = 0.0;
    } else {
        long mm = (2 * m > L) ? m - L : m;
        long double th = (TWO_PI * (long double)mm) / (long double)L;
        *c = (double)cosl(th);
        *s = (double)sinl(th);
    }
}

/* Parity-sorted second-fold column position (gen_r7): natural columns
 * j = 1..h2 land odd-j first (ascending), then even-j; the lone j = L/4
 * column (4 | L) lands at the END of its own parity's run.  Tables and fold
 * stores share this map, so dm_kfold8e (order-blind dot product) is
 * unaffected and dm_kfold8q gets two contiguous (-1)^j sign runs. */
static inline int dm_pos2(int j, int no2)
{
    return (j & 1) ? (j - 1) >> 1 : no2 + (j >> 1) - 1;
}

/* Second-fold tables (evenk): row for output k, columns j = 1..h2 hold
 * cos/sin(2*pi*k*j/L) at parity-sorted positions; the lone j = L/4 column
 * (4 | L) is nonzero only for the parity that consumes it (cos: even k,
 * where it is exactly +-1 or 0; sin: odd k, exactly +-1).  The identity:
 * cos(2*pi*k*(L/2-j)/L) = (-1)^k cos(2*pi*k*j/L) and sin(...) =
 * -(-1)^k sin(...), so even k consume ue = u_j + u_{L/2-j} / ve = v_j -
 * v_{L/2-j} and odd k consume uo / vo -- same table values either way. */
static void dm_fill_row2(const fft3d_plan *p, double *Crow, double *Srow, int k)
{
    const int L = p->L, h2 = p->h2, he2 = p->he2, no2 = p->no2;
    for (int j = 1; j <= h2; ++j)
        dm_cis((long)k * j, L, Crow + dm_pos2(j, no2), Srow + dm_pos2(j, no2));
    if (he2 > h2) {
        const int pos = ((L / 4) & 1) ? no2 - 1 : he2 - 1;
        double c, s;
        dm_cis((long)k * (L / 4), L, &c, &s);
        Crow[pos] = (k % 2 == 0) ? c : 0.0;
        Srow[pos] = (k % 2 != 0) ? s : 0.0;
    }
}

/* gen_r12: quad-kernel table rows in CONSUMPTION order.  dm_kfold8q's two
 * parity classes each read every OTHER row (r = k-1 for k = 1,5,9,.. resp.
 * 2,6,10..): a stride-2*hs2 walk in ~256 B runs that the DCU/L2 stride
 * prefetchers do not track -- measured at L=100 as ~300 L1-miss loads per
 * 8-pencil group (the table walk, re-fetched from L2 every group) with
 * cycle_activity.stalls_l1d_miss at 18% of cycles.  The class-major
 * relayout (even row indices 0..q4-1 first, then odd: each class walks one
 * sequential stream per table) was built and MEASURED A NULL -- L=40 wash,
 * L=100 ~-0.7% over 2 interleaved pairs -- because the 16-accumulator sweep
 * is latency-tolerant: the missing table lines sit in flight behind
 * independent FMA chains and never bind.  stalls_l1d_miss counts
 * correlation, not causation, on accumulator kernels (gen_r12 record).
 * Kept OPT-IN under GL_DEMO_TORD so nobody rebuilds it; rows >= q4 and the
 * base rows at h, h+1 keep identity positions either way, and outputs are
 * bit-identical (fill and reads share this map; quadk plans only). */
static inline int dm_rq2(const fft3d_plan *p, int r)
{
#if GL_DEMO_TORD
    const int q4 = p->L / 4;
    if (!p->quadk || r >= q4) return r;
    return (r & 1) ? (q4 + 1) / 2 + (r >> 1) : (r >> 1);
#else
    (void)p;
    return r;
#endif
}

static void dm_fill_tabs2(fft3d_plan *p)
{
    const int h = p->h, hs2 = p->hs2;
    for (int k = 1; k <= h; ++k)
        dm_fill_row2(p, p->Ct2 + (size_t)dm_rq2(p, k - 1) * hs2,
                     p->St2 + (size_t)dm_rq2(p, k - 1) * hs2, k);
    dm_fill_row2(p, p->Ct2 + (size_t)h * hs2, p->St2 + (size_t)h * hs2, 0);
    dm_fill_row2(p, p->Ct2 + (size_t)(h + 1) * hs2, p->St2 + (size_t)(h + 1) * hs2,
                 p->L / 2);
}

/* Plane pitch (doubles) for the 4-slot y->z window: candidates are the
 * line-rounded plane size + 0..7 lines, scored by the layer's own collision
 * model -- alias pairs of the L live rows (stride 8L bytes) of one slot
 * against the same rows of the other three slots.  The layer, dogfooded. */
#if defined(__AVX512F__) && !GL_DEMO_NOFUSE
static size_t dm_pick_wpitch(int L)
{
    const size_t LL = (size_t)L * L;
    const size_t b0 = (LL * sizeof(double) + GL_LINE - 1) & ~(size_t)(GL_LINE - 1);
    size_t best = b0;
    long bests = 0x7fffffffL;
    for (int e = 0; e < 8; ++e) {
        size_t cand = b0 + (size_t)e * GL_LINE;
        long sc = 0;
        for (int k = 1; k <= 3; ++k)
            sc += gl_alias_pairs4k(0, (long)L * 8, L,
                                   (long)(((size_t)k * cand) & (GL_PAGE - 1)),
                                   (long)L * 8, L, GL_LINE);
        if (sc < bests) {
            bests = sc;
            best = cand;
        }
    }
    return best / sizeof(double);
}
#endif /* __AVX512F__ && !GL_DEMO_NOFUSE */

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
    p->h2 = p->h / 2;
    p->he2 = p->h2 + (L % 4 == 0);
    p->hs2 = (p->he2 + 7) & ~7;
#if defined(__AVX512F__) && !GL_DEMO_NOEVEN
    p->evenk = p->even && L >= 8;
#else
    p->evenk = 0;
#endif
    p->no2 = 0;
    p->quadk = 0;
    if (p->evenk) {
        /* parity-sorted second-fold columns: odd j of 1..h2 first, then even;
         * the lone j = L/4 column joins the run of its own parity (at the
         * run's end).  The fold stores and the Ct2/St2 fill agree on this
         * order, so dm_kfold8e is order-blind and dm_kfold8q gets its two
         * contiguous sign runs for free. */
        p->no2 = (p->h2 + 1) / 2;
        if (p->he2 > p->h2 && ((L / 4) & 1)) p->no2 += 1;
#if !GL_DEMO_NOQUAD
        p->quadk = (L % 4 == 0);
#endif
    }
#if defined(__AVX512F__) && !GL_DEMO_NOFUSE
    p->wpitch = (L >= 8) ? dm_pick_wpitch(L) : 0;
#else
    p->wpitch = 0;
#endif
#if defined(__AVX512F__) && !GL_DEMO_NONT
    /* NT stores only where the destination stream is DRAM-resident by size
     * and every store is a full aligned line: axis-0 kernel rows need
     * L^2 % 8 == 0 (row stride a line multiple), dst pencil rows L % 4 == 0.
     * Below GL_NT_MIN_BYTES a later pass reads the stream back from cache --
     * NT would force that read to DRAM (measured doctrine, see strategy). */
    p->nt0 = p->V * sizeof(double) >= GL_NT_MIN_BYTES &&
             ((size_t)L * L % 8 == 0);
    p->nt2 = p->V * sizeof(double) >= GL_NT_MIN_BYTES && (L % 4 == 0);
#else
    p->nt0 = p->nt2 = 0;
#endif
#if defined(__AVX512F__) && !GL_DEMO_NOPF1
    /* T1 fold prefetch where the fold's source volume is DRAM-resident (the
     * LFB argument at DM_PF2; suite: L=100 only).  Cache-resident sources
     * keep T0 -- there the L1 fill is the whole point (r4: -2.6% at 31). */
    p->pf1 = p->V * sizeof(double) >= GL_NT_MIN_BYTES;
#else
    p->pf1 = 0;
#endif
#if defined(__AVX512F__) && !GL_DEMO_NOLDT
    p->ldt = (L >= 16); /* measured boundary: -6% at 32, -5% at 25, -1.4% at
                           20, +1% at 12 (strategy record gen_r8) */
#else
    p->ldt = 0;
#endif
#if defined(__AVX512F__) && defined(__AVX512VL__) && !GL_DEMO_NOMAP4
    p->m4t = (L <= GL_M4T_MAX); /* ymm tail wins where kcnt=2 chunks are >=
                           1/2 of exits (L=10: -3..4%); loses at L=50
                           (+1.5-2%) -- SPR advisory; ICX verdict gen_r10 */
#else
    p->m4t = 0;
#endif

    const size_t ctbytes = (size_t)(p->h + 1 + p->even) * p->hs * sizeof(double);
    const size_t stbytes = (size_t)(p->h > 0 ? p->h : 1) * p->hs * sizeof(double);
    const size_t c2bytes = p->evenk ? (size_t)(p->h + 2) * p->hs2 * sizeof(double) : 0;
    const size_t uvobytes = p->evenk ? (size_t)16 * p->he2 * sizeof(double) + GL_LINE : 0;
    const size_t ebbytes = p->evenk ? (size_t)32 * sizeof(double) + GL_LINE : 0;
    const size_t pbbytes = (size_t)16 * p->Lp * sizeof(double) + 2 * GL_LINE;
    const size_t ubbytes = (size_t)16 * (p->hu > 0 ? p->hu : 1) * sizeof(double) + GL_LINE;
    const size_t vbbytes = (size_t)16 * (p->h > 0 ? p->h : 1) * sizeof(double) + GL_LINE;
    const size_t sbytes = p->V * sizeof(double) + GL_PAGE + 2 * GL_LINE;
    /* the y->z window needs only 4 plane slots, not a volume */
    const size_t s2bytes = p->wpitch
                               ? 4 * p->wpitch * sizeof(double) + GL_PAGE + 2 * GL_LINE
                               : sbytes;
#if GL_DEMO_NOMAPFUSE
    const size_t zbytes = p->V * sizeof(double _Complex) + 2 * GL_LINE;
#else
    const size_t zbytes = 0; /* fused chain: no zt volume, map at axis-2 exit */
#endif
    (void)zbytes; /* unused only under GL_DEMO_PLAIN && !GL_DEMO_NOMAPFUSE */
    /* gen_r11 re-home gate: volume big enough that state+c 4K pages overrun
     * the STLB (suite: L=100 only).  The smaps verdict on the driver's actual
     * buffers is taken at the first chain call; this only sizes the arena. */
    const size_t vbytes = p->V * sizeof(double _Complex);
    int reh = 0;
#if !GL_DEMO_PLAIN && !GL_DEMO_NOREHOME && !GL_DEMO_NOMAPFUSE
    reh = vbytes >= GL_REHOME_MIN_BYTES;
#endif
    const size_t rsbytes = reh ? vbytes + GL_LINE : 0;
    const size_t rcbytes = (reh && !GL_DEMO_NOCV) ? vbytes + GL_LINE : 0;

#if !GL_DEMO_PLAIN
    size_t total = ctbytes + stbytes + 2 * c2bytes + 2 * uvobytes + ebbytes +
                   2 * pbbytes + ubbytes + vbbytes +
                   2 * sbytes + 2 * s2bytes + zbytes + rsbytes + rcbytes +
                   30 * GL_PAGE + (64u << 10);
    if (gl_arena_init(&p->ar, total) != 0) {
        free(p);
        return NULL;
    }
#endif
    p->Ct = demo_alloc(p, ctbytes);
    p->St = demo_alloc(p, stbytes);
    if (p->evenk) {
        p->Ct2 = demo_alloc(p, c2bytes);
        p->St2 = demo_alloc(p, c2bytes);
        p->uob = demo_alloc(p, uvobytes);
        p->vob = demo_alloc(p, uvobytes);
        p->ebb = demo_alloc(p, ebbytes);
        if (!p->Ct2 || !p->St2 || !p->uob || !p->vob || !p->ebb) {
            fft3d_destroy(p);
            return NULL;
        }
    }
    p->pb = demo_alloc(p, pbbytes);
    p->ob = demo_alloc(p, pbbytes);
    p->ub = demo_alloc(p, ubbytes);
    p->vb = demo_alloc(p, vbbytes);
    p->s1r_slab = demo_alloc(p, sbytes);
    p->s1i_slab = demo_alloc(p, sbytes);
    p->s2r_slab = demo_alloc(p, s2bytes);
    p->s2i_slab = demo_alloc(p, s2bytes);
#if GL_DEMO_NOMAPFUSE
    p->zt = demo_alloc(p, zbytes);
    if (!p->zt) {
        fft3d_destroy(p);
        return NULL;
    }
#else
    p->zt = NULL;
#endif
    if (reh) {
        p->stv = demo_alloc(p, rsbytes);
        if (!p->stv) {
            fft3d_destroy(p);
            return NULL;
        }
        if (rcbytes) {
            p->cv = demo_alloc(p, rcbytes);
            if (!p->cv) {
                fft3d_destroy(p);
                return NULL;
            }
        }
    }
    if (!p->Ct || !p->St || !p->pb || !p->ob || !p->ub || !p->vb ||
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
    if (p->evenk) dm_fill_tabs2(p);
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
static inline double *dm_place(double *slab, unsigned want)
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

/* masked store, or a non-temporal full-line store when the caller asserts
 * nt (only ever with m == 0xFF and a 64B-aligned destination) */
static inline void dm_st(double *d, __mmask8 m, int nt, __m512d v)
{
    if (nt) _mm512_stream_pd(d, v);
    else _mm512_mask_storeu_pd(d, m, v);
}

/* software prefetch for the fold loads: the axis-0/1 folds walk ~L separate
 * row streams per group -- more than the L2 streamer tracks at large L, so
 * the next group's lines are pulled explicitly (2 groups ahead).
 * gen_r12: hint is T0 only while the source volume is cache-resident.  On a
 * DRAM-RESIDENT source (L=100: the state and s1) a T0 prefetch occupies an
 * L1 fill buffer (~12) for the full DRAM latency, so ~200 fold lines per
 * group serialize through the LFB pool at ~9 GB/s -- which is why the r4
 * prefetch measured a WASH at L=100 (it competed with the demand misses for
 * the same LFBs).  T1 fills L2 through the deeper L2 superqueue instead; the
 * demand load then pays one L2 hit that the OoO window hides.  Plan-gated
 * per source size (p->pf1, same size test as nt0); values bit-identical. */
#if !GL_DEMO_NOPF
#define DM_PF(P) _mm_prefetch((const char *)(P), _MM_HINT_T0)
#define DM_PF2(P, F)                                                             \
    do {                                                                         \
        if (F) _mm_prefetch((const char *)(P), _MM_HINT_T1);                     \
        else _mm_prefetch((const char *)(P), _MM_HINT_T0);                       \
    } while (0)
#else
#define DM_PF(P) ((void)0)
#define DM_PF2(P, F) ((void)(F))
#endif

/* 8x8 double transpose straight from memory (gen_r8; PROMOTED to the library
 * as gl_tr8x8_ld in gen_r13 -- the round's within-volume pencil-lane staging
 * is this network, so the r8 offer is cashed).  Identical code; the demo now
 * consumes the library form like any adopter would. */
#define dm_tr8x8_ld gl_tr8x8_ld

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
                             double *dr, double *di, size_t ostride, __mmask8 sm,
                             int nt)
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
            dm_st(dr + (size_t)k * ostride, sm, nt, _mm512_add_pd(ar, br));
            dm_st(di + (size_t)k * ostride, sm, nt, _mm512_add_pd(ai, bi));
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
        dm_st(dr + (size_t)(KK)*ostride, sm, nt, _mm512_add_pd(t0, si##T));         \
        dm_st(di + (size_t)(KK)*ostride, sm, nt, _mm512_sub_pd(t1, sr##T));         \
        dm_st(dr + (size_t)(L - (KK)) * ostride, sm, nt,                            \
              _mm512_sub_pd(t0, si##T));                                            \
        dm_st(di + (size_t)(L - (KK)) * ostride, sm, nt,                            \
              _mm512_add_pd(t1, sr##T));                                            \
    } while (0)
        DM_COMB(0, k + 0);
        DM_COMB(1, k + 1);
        DM_COMB(2, k + 2);
        DM_COMB(3, k + 3);
    }
    for (; k + 2 <= h + 1; k += 2) { /* 2-pair tail (h mod 4 in {2,3}) */
        const double *c0 = p->Ct + (size_t)(k - 1) * hs, *c1 = c0 + hs;
        const double *s0 = p->St + (size_t)(k - 1) * hs, *s1 = s0 + hs;
        __m512d cr0 = _mm512_setzero_pd(), ci0 = cr0, sr0 = cr0, si0 = cr0;
        __m512d cr1 = cr0, ci1 = cr0, sr1 = cr0, si1 = cr0;
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
        }
        if (even) {
            __m512d ur = _mm512_loadu_pd(ub + 16 * (hu - 1));
            __m512d ui = _mm512_loadu_pd(ub + 16 * (hu - 1) + 8);
            __m512d w = _mm512_set1_pd(c0[hu - 1]);
            cr0 = _mm512_fmadd_pd(w, ur, cr0);
            ci0 = _mm512_fmadd_pd(w, ui, ci0);
            w = _mm512_set1_pd(c1[hu - 1]);
            cr1 = _mm512_fmadd_pd(w, ur, cr1);
            ci1 = _mm512_fmadd_pd(w, ui, ci1);
        }
        DM_COMB(0, k);
        DM_COMB(1, k + 1);
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

/* Second-level fold (gen_r6, even L): from the staged pencil block
 * pb[L][2][8] build the four k-parity blocks and the two base rows:
 *   ue[j-1] = u_j + u_{L/2-j}   uo[j-1] = u_j - u_{L/2-j}     (u_j = x_j + x_{L-j})
 *   ve[j-1] = v_j - v_{L/2-j}   vo[j-1] = v_j + v_{L/2-j}     (v_j = x_j - x_{L-j})
 * for j = 1..h2; when 4 | L the lone j = L/4 row is appended (ue/vo carry the
 * real values, uo/ve get finite fillers -- their table coefficients are 0).
 * eb[0..15] = x0 + x_{L/2}  (even-k base), eb[16..31] = x0 - x_{L/2} (odd). */
static inline void dm_fold8e(const fft3d_plan *p, const double *pb, double *eb,
                             double *ue, double *uo, double *ve, double *vo)
{
    const int L = p->L, h2 = p->h2, q = L / 2, no2 = p->no2;
    for (int j = 1; j <= h2; ++j) {
        const size_t o16 = 16 * (size_t)dm_pos2(j, no2);
        __m512d ar = _mm512_loadu_pd(pb + 16 * j);
        __m512d ai = _mm512_loadu_pd(pb + 16 * j + 8);
        __m512d br = _mm512_loadu_pd(pb + 16 * (L - j));
        __m512d bi = _mm512_loadu_pd(pb + 16 * (L - j) + 8);
        __m512d cr = _mm512_loadu_pd(pb + 16 * (q - j));
        __m512d ci = _mm512_loadu_pd(pb + 16 * (q - j) + 8);
        __m512d dr = _mm512_loadu_pd(pb + 16 * (q + j));
        __m512d di = _mm512_loadu_pd(pb + 16 * (q + j) + 8);
        __m512d apr = _mm512_add_pd(ar, br), api = _mm512_add_pd(ai, bi);
        __m512d amr = _mm512_sub_pd(ar, br), ami = _mm512_sub_pd(ai, bi);
        __m512d bpr = _mm512_add_pd(cr, dr), bpi = _mm512_add_pd(ci, di);
        __m512d bmr = _mm512_sub_pd(cr, dr), bmi = _mm512_sub_pd(ci, di);
        _mm512_storeu_pd(ue + o16, _mm512_add_pd(apr, bpr));
        _mm512_storeu_pd(ue + o16 + 8, _mm512_add_pd(api, bpi));
        _mm512_storeu_pd(uo + o16, _mm512_sub_pd(apr, bpr));
        _mm512_storeu_pd(uo + o16 + 8, _mm512_sub_pd(api, bpi));
        _mm512_storeu_pd(ve + o16, _mm512_sub_pd(amr, bmr));
        _mm512_storeu_pd(ve + o16 + 8, _mm512_sub_pd(ami, bmi));
        _mm512_storeu_pd(vo + o16, _mm512_add_pd(amr, bmr));
        _mm512_storeu_pd(vo + o16 + 8, _mm512_add_pd(ami, bmi));
    }
    if (p->he2 > h2) { /* lone j = L/4: pairs with itself under j -> L/2-j */
        const size_t o16 = 16 * (size_t)(((L / 4) & 1) ? no2 - 1 : p->he2 - 1);
        __m512d ar = _mm512_loadu_pd(pb + 16 * (L / 4));
        __m512d ai = _mm512_loadu_pd(pb + 16 * (L / 4) + 8);
        __m512d br = _mm512_loadu_pd(pb + 16 * (q + L / 4));
        __m512d bi = _mm512_loadu_pd(pb + 16 * (q + L / 4) + 8);
        __m512d pr = _mm512_add_pd(ar, br), pi = _mm512_add_pd(ai, bi);
        __m512d mr = _mm512_sub_pd(ar, br), mi = _mm512_sub_pd(ai, bi);
        _mm512_storeu_pd(ue + o16, pr);
        _mm512_storeu_pd(ue + o16 + 8, pi);
        _mm512_storeu_pd(uo + o16, pr); /* coeff 0: finite filler */
        _mm512_storeu_pd(uo + o16 + 8, pi);
        _mm512_storeu_pd(ve + o16, mr); /* coeff 0: finite filler */
        _mm512_storeu_pd(ve + o16 + 8, mi);
        _mm512_storeu_pd(vo + o16, mr);
        _mm512_storeu_pd(vo + o16 + 8, mi);
    }
    {
        __m512d x0r = _mm512_loadu_pd(pb), x0i = _mm512_loadu_pd(pb + 8);
        __m512d xqr = _mm512_loadu_pd(pb + 16 * q);
        __m512d xqi = _mm512_loadu_pd(pb + 16 * q + 8);
        _mm512_storeu_pd(eb, _mm512_add_pd(x0r, xqr));
        _mm512_storeu_pd(eb + 8, _mm512_add_pd(x0i, xqi));
        _mm512_storeu_pd(eb + 16, _mm512_sub_pd(x0r, xqr));
        _mm512_storeu_pd(eb + 24, _mm512_sub_pd(x0i, xqi));
    }
}

/* Even-L folded 8-column pencil kernel (gen_r6): identical contract to
 * dm_kfold8 but the j-sweep runs over he2 ~ h/2 columns.  Output pairs
 * (k, L-k) share k's parity (L even), so each pair binds to one base row and
 * one (u, v) block pair: odd k -> (uo, vo, o-base), even k -> (ue, ve,
 * e-base).  Pairs are taken 4 at a time from k = 1, so the parity pattern
 * (o,e,o,e) is static; the 1-pair tail is always odd (k = 1 mod 4 or 3 mod 4).
 * 16 FMAs / 16 loads per j for 4 pairs -- the loads double per j vs dm_kfold8
 * but j halves, so the sweep is ~2x fewer FMAs at balanced port pressure. */
static inline void dm_kfold8e(const fft3d_plan *p, const double *eb,
                              const double *ue, const double *uo,
                              const double *ve, const double *vo,
                              double *dr, double *di, size_t ostride, __mmask8 sm,
                              int nt)
{
    const int L = p->L, h = p->h, he = p->he2, hs = p->hs2;
    const __m512d ebr = _mm512_loadu_pd(eb), ebi = _mm512_loadu_pd(eb + 8);
    const __m512d obr = _mm512_loadu_pd(eb + 16), obi = _mm512_loadu_pd(eb + 24);

    /* C-only rows: k = 0 (even) and k = L/2 (parity (L/2) & 1) */
    for (int r = 0; r < 2; ++r) {
        const int k = r ? L / 2 : 0;
        const int odd = k & 1;
        const double *w = p->Ct2 + (size_t)(h + r) * hs;
        const double *bu = odd ? uo : ue;
        __m512d ar = odd ? obr : ebr, ai = odd ? obi : ebi;
        __m512d br = _mm512_setzero_pd(), bi = br;
        int j = 0;
        for (; j + 2 <= he; j += 2) {
            __m512d c0 = _mm512_set1_pd(w[j]), c1 = _mm512_set1_pd(w[j + 1]);
            ar = _mm512_fmadd_pd(c0, _mm512_loadu_pd(bu + 16 * j), ar);
            ai = _mm512_fmadd_pd(c0, _mm512_loadu_pd(bu + 16 * j + 8), ai);
            br = _mm512_fmadd_pd(c1, _mm512_loadu_pd(bu + 16 * (j + 1)), br);
            bi = _mm512_fmadd_pd(c1, _mm512_loadu_pd(bu + 16 * (j + 1) + 8), bi);
        }
        if (j < he) {
            __m512d c0 = _mm512_set1_pd(w[j]);
            ar = _mm512_fmadd_pd(c0, _mm512_loadu_pd(bu + 16 * j), ar);
            ai = _mm512_fmadd_pd(c0, _mm512_loadu_pd(bu + 16 * j + 8), ai);
        }
        dm_st(dr + (size_t)k * ostride, sm, nt, _mm512_add_pd(ar, br));
        dm_st(di + (size_t)k * ostride, sm, nt, _mm512_add_pd(ai, bi));
    }

    int k = 1;
    for (; k + 4 <= h + 1; k += 4) { /* parities: odd, even, odd, even */
        const double *c0 = p->Ct2 + (size_t)(k - 1) * hs, *c1 = c0 + hs;
        const double *c2 = c1 + hs, *c3 = c2 + hs;
        const double *s0 = p->St2 + (size_t)(k - 1) * hs, *s1 = s0 + hs;
        const double *s2 = s1 + hs, *s3 = s2 + hs;
        __m512d cr0 = _mm512_setzero_pd(), ci0 = cr0, sr0 = cr0, si0 = cr0;
        __m512d cr1 = cr0, ci1 = cr0, sr1 = cr0, si1 = cr0;
        __m512d cr2 = cr0, ci2 = cr0, sr2 = cr0, si2 = cr0;
        __m512d cr3 = cr0, ci3 = cr0, sr3 = cr0, si3 = cr0;
        for (int j = 0; j < he; ++j) {
            __m512d uer = _mm512_loadu_pd(ue + 16 * j);
            __m512d uei = _mm512_loadu_pd(ue + 16 * j + 8);
            __m512d uor = _mm512_loadu_pd(uo + 16 * j);
            __m512d uoi = _mm512_loadu_pd(uo + 16 * j + 8);
            __m512d ver = _mm512_loadu_pd(ve + 16 * j);
            __m512d vei = _mm512_loadu_pd(ve + 16 * j + 8);
            __m512d vor = _mm512_loadu_pd(vo + 16 * j);
            __m512d voi = _mm512_loadu_pd(vo + 16 * j + 8);
            __m512d w;
            w = _mm512_set1_pd(c0[j]);
            cr0 = _mm512_fmadd_pd(w, uor, cr0);
            ci0 = _mm512_fmadd_pd(w, uoi, ci0);
            w = _mm512_set1_pd(s0[j]);
            sr0 = _mm512_fmadd_pd(w, vor, sr0);
            si0 = _mm512_fmadd_pd(w, voi, si0);
            w = _mm512_set1_pd(c1[j]);
            cr1 = _mm512_fmadd_pd(w, uer, cr1);
            ci1 = _mm512_fmadd_pd(w, uei, ci1);
            w = _mm512_set1_pd(s1[j]);
            sr1 = _mm512_fmadd_pd(w, ver, sr1);
            si1 = _mm512_fmadd_pd(w, vei, si1);
            w = _mm512_set1_pd(c2[j]);
            cr2 = _mm512_fmadd_pd(w, uor, cr2);
            ci2 = _mm512_fmadd_pd(w, uoi, ci2);
            w = _mm512_set1_pd(s2[j]);
            sr2 = _mm512_fmadd_pd(w, vor, sr2);
            si2 = _mm512_fmadd_pd(w, voi, si2);
            w = _mm512_set1_pd(c3[j]);
            cr3 = _mm512_fmadd_pd(w, uer, cr3);
            ci3 = _mm512_fmadd_pd(w, uei, ci3);
            w = _mm512_set1_pd(s3[j]);
            sr3 = _mm512_fmadd_pd(w, ver, sr3);
            si3 = _mm512_fmadd_pd(w, vei, si3);
        }
#define DM_COMBE(T, KK, BR, BI)                                                     \
    do {                                                                            \
        __m512d t0 = _mm512_add_pd(BR, cr##T), t1 = _mm512_add_pd(BI, ci##T);       \
        dm_st(dr + (size_t)(KK)*ostride, sm, nt, _mm512_add_pd(t0, si##T));         \
        dm_st(di + (size_t)(KK)*ostride, sm, nt, _mm512_sub_pd(t1, sr##T));         \
        dm_st(dr + (size_t)(L - (KK)) * ostride, sm, nt,                            \
              _mm512_sub_pd(t0, si##T));                                            \
        dm_st(di + (size_t)(L - (KK)) * ostride, sm, nt,                            \
              _mm512_add_pd(t1, sr##T));                                            \
    } while (0)
        DM_COMBE(0, k + 0, obr, obi);
        DM_COMBE(1, k + 1, ebr, ebi);
        DM_COMBE(2, k + 2, obr, obi);
        DM_COMBE(3, k + 3, ebr, ebi);
    }
    for (; k + 2 <= h + 1; k += 2) { /* 2-pair tail: odd, even */
        const double *c0 = p->Ct2 + (size_t)(k - 1) * hs, *c1 = c0 + hs;
        const double *s0 = p->St2 + (size_t)(k - 1) * hs, *s1 = s0 + hs;
        __m512d cr0 = _mm512_setzero_pd(), ci0 = cr0, sr0 = cr0, si0 = cr0;
        __m512d cr1 = cr0, ci1 = cr0, sr1 = cr0, si1 = cr0;
        for (int j = 0; j < he; ++j) {
            __m512d w;
            w = _mm512_set1_pd(c0[j]);
            cr0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(uo + 16 * j), cr0);
            ci0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(uo + 16 * j + 8), ci0);
            w = _mm512_set1_pd(s0[j]);
            sr0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(vo + 16 * j), sr0);
            si0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(vo + 16 * j + 8), si0);
            w = _mm512_set1_pd(c1[j]);
            cr1 = _mm512_fmadd_pd(w, _mm512_loadu_pd(ue + 16 * j), cr1);
            ci1 = _mm512_fmadd_pd(w, _mm512_loadu_pd(ue + 16 * j + 8), ci1);
            w = _mm512_set1_pd(s1[j]);
            sr1 = _mm512_fmadd_pd(w, _mm512_loadu_pd(ve + 16 * j), sr1);
            si1 = _mm512_fmadd_pd(w, _mm512_loadu_pd(ve + 16 * j + 8), si1);
        }
        DM_COMBE(0, k, obr, obi);
        DM_COMBE(1, k + 1, ebr, ebi);
    }
    for (; k <= h; ++k) { /* 1-pair tail: always odd parity (k = 1 or 3 mod 4) */
        const double *c0 = p->Ct2 + (size_t)(k - 1) * hs;
        const double *s0 = p->St2 + (size_t)(k - 1) * hs;
        __m512d cr0 = _mm512_setzero_pd(), ci0 = cr0, sr0 = cr0, si0 = cr0;
        for (int j = 0; j < he; ++j) {
            __m512d w = _mm512_set1_pd(c0[j]);
            cr0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(uo + 16 * j), cr0);
            ci0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(uo + 16 * j + 8), ci0);
            w = _mm512_set1_pd(s0[j]);
            sr0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(vo + 16 * j), sr0);
            si0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(vo + 16 * j + 8), si0);
        }
        DM_COMBE(0, k, obr, obi);
    }
#undef DM_COMBE
}

/* Third-level k-fold kernel (gen_r7), 4 | L only.  Rows k and L/2-k of the
 * second-fold tables agree up to the COLUMN sign (-1)^j:
 *     cos(2*pi*(L/2-k)*j/L) =  (-1)^j cos(2*pi*k*j/L)
 *     sin(2*pi*(L/2-k)*j/L) = -(-1)^j sin(2*pi*k*j/L)
 * and 4 | L keeps k and L/2-k in the SAME k-parity class (same u/v blocks,
 * same base row).  With the columns parity-sorted (odd-j run 0..no2-1, then
 * even-j; fold and tables agree), one j-sweep per k accumulates E (even-j)
 * and O (odd-j) partial sums and E +- O yields FOUR outputs per sweep:
 * (k, L-k) from E+O and (L/2-k, L/2+k) from E-O -- the r6 kernel needed two
 * sweeps for those.  Two same-class quads run per group: per column 4 block
 * loads + 4 broadcasts against 8 FMAs, the same port balance as r6's 4-pair
 * sweep at HALF the sweep length per output.  k = L/4 pairs with itself
 * (r6-style lone pair over all he2 columns); k = 0 and k = L/2 merge into a
 * table-free E/O add sweep of ue.  Sweep FMAs ~ 4*(L/4)*he2: half of gen_r6,
 * a quarter of gen_r5. */
static inline void dm_kfold8q(const fft3d_plan *p, const double *eb,
                              const double *ue, const double *uo,
                              const double *ve, const double *vo,
                              double *dr, double *di, size_t ostride, __mmask8 sm,
                              int nt)
{
    const int L = p->L, he = p->he2, hs = p->hs2, no = p->no2, q4 = L / 4;
    const __m512d ebr = _mm512_loadu_pd(eb), ebi = _mm512_loadu_pd(eb + 8);
    const __m512d obr = _mm512_loadu_pd(eb + 16), obi = _mm512_loadu_pd(eb + 24);

    { /* k = 0 and k = L/2 (both even class): the all-ones row -- table-free
       * E/O add sweep of ue; X_0 = eb + (E+O), X_{L/2} = eb + (E-O) */
        __m512d er = _mm512_setzero_pd(), ei = er, our = er, oui = er;
        int j = 0;
        for (; j < no; ++j) {
            our = _mm512_add_pd(our, _mm512_loadu_pd(ue + 16 * j));
            oui = _mm512_add_pd(oui, _mm512_loadu_pd(ue + 16 * j + 8));
        }
        for (; j < he; ++j) {
            er = _mm512_add_pd(er, _mm512_loadu_pd(ue + 16 * j));
            ei = _mm512_add_pd(ei, _mm512_loadu_pd(ue + 16 * j + 8));
        }
        dm_st(dr, sm, nt, _mm512_add_pd(ebr, _mm512_add_pd(er, our)));
        dm_st(di, sm, nt, _mm512_add_pd(ebi, _mm512_add_pd(ei, oui)));
        dm_st(dr + (size_t)(L / 2) * ostride, sm, nt,
              _mm512_add_pd(ebr, _mm512_sub_pd(er, our)));
        dm_st(di + (size_t)(L / 2) * ostride, sm, nt,
              _mm512_add_pd(ebi, _mm512_sub_pd(ei, oui)));
    }

/* four outputs from one accumulator set: E+O -> (KK, L-KK); E-O (with the
 * sin sign flipped: S_{L/2-k} = O_s - E_s) -> (L/2-KK, L/2+KK) */
#define DM_COMBQ(Ecr, Eci, Esr, Esi, Ocr, Oci, Osr, Osi, BR, BI, KK)              \
    do {                                                                          \
        __m512d cr_ = _mm512_add_pd(Ecr, Ocr), ci_ = _mm512_add_pd(Eci, Oci);     \
        __m512d sr_ = _mm512_add_pd(Esr, Osr), si_ = _mm512_add_pd(Esi, Osi);     \
        __m512d t0 = _mm512_add_pd(BR, cr_), t1 = _mm512_add_pd(BI, ci_);         \
        dm_st(dr + (size_t)(KK)*ostride, sm, nt, _mm512_add_pd(t0, si_));         \
        dm_st(di + (size_t)(KK)*ostride, sm, nt, _mm512_sub_pd(t1, sr_));         \
        dm_st(dr + (size_t)(L - (KK)) * ostride, sm, nt,                          \
              _mm512_sub_pd(t0, si_));                                            \
        dm_st(di + (size_t)(L - (KK)) * ostride, sm, nt,                          \
              _mm512_add_pd(t1, sr_));                                            \
        cr_ = _mm512_sub_pd(Ecr, Ocr);                                            \
        ci_ = _mm512_sub_pd(Eci, Oci);                                            \
        sr_ = _mm512_sub_pd(Osr, Esr);                                            \
        si_ = _mm512_sub_pd(Osi, Esi);                                            \
        t0 = _mm512_add_pd(BR, cr_);                                              \
        t1 = _mm512_add_pd(BI, ci_);                                              \
        dm_st(dr + (size_t)(L / 2 - (KK)) * ostride, sm, nt,                      \
              _mm512_add_pd(t0, si_));                                            \
        dm_st(di + (size_t)(L / 2 - (KK)) * ostride, sm, nt,                      \
              _mm512_sub_pd(t1, sr_));                                            \
        dm_st(dr + (size_t)(L / 2 + (KK)) * ostride, sm, nt,                      \
              _mm512_sub_pd(t0, si_));                                            \
        dm_st(di + (size_t)(L / 2 + (KK)) * ostride, sm, nt,                      \
              _mm512_add_pd(t1, sr_));                                            \
    } while (0)

    for (int par = 1; par >= 0; --par) { /* odd class (k=1,3,..), then even */
        const double *bu = par ? uo : ue, *bv = par ? vo : ve;
        const __m512d cbr = par ? obr : ebr, cbi = par ? obi : ebi;
        int k = 2 - par;
        for (; k + 2 < q4; k += 4) { /* two same-class quads k and k+2 */
            /* dm_rq2 is identity by default; under GL_DEMO_TORD k-1 and k+1
             * are same-parity neighbors, so the class walk is sequential */
            const double *c0 = p->Ct2 + (size_t)dm_rq2(p, k - 1) * hs;
            const double *s0 = p->St2 + (size_t)dm_rq2(p, k - 1) * hs;
            const double *c1 = p->Ct2 + (size_t)dm_rq2(p, k + 1) * hs;
            const double *s1 = p->St2 + (size_t)dm_rq2(p, k + 1) * hs;
            __m512d aEcr = _mm512_setzero_pd(), aEci = aEcr, aEsr = aEcr,
                    aEsi = aEcr;
            __m512d aOcr = aEcr, aOci = aEcr, aOsr = aEcr, aOsi = aEcr;
            __m512d bEcr = aEcr, bEci = aEcr, bEsr = aEcr, bEsi = aEcr;
            __m512d bOcr = aEcr, bOci = aEcr, bOsr = aEcr, bOsi = aEcr;
            int j = 0;
            for (; j < no; ++j) { /* odd-j run -> O accumulators */
                __m512d ur = _mm512_loadu_pd(bu + 16 * j);
                __m512d ui = _mm512_loadu_pd(bu + 16 * j + 8);
                __m512d wr = _mm512_loadu_pd(bv + 16 * j);
                __m512d wi = _mm512_loadu_pd(bv + 16 * j + 8);
                __m512d w;
                w = _mm512_set1_pd(c0[j]);
                aOcr = _mm512_fmadd_pd(w, ur, aOcr);
                aOci = _mm512_fmadd_pd(w, ui, aOci);
                w = _mm512_set1_pd(s0[j]);
                aOsr = _mm512_fmadd_pd(w, wr, aOsr);
                aOsi = _mm512_fmadd_pd(w, wi, aOsi);
                w = _mm512_set1_pd(c1[j]);
                bOcr = _mm512_fmadd_pd(w, ur, bOcr);
                bOci = _mm512_fmadd_pd(w, ui, bOci);
                w = _mm512_set1_pd(s1[j]);
                bOsr = _mm512_fmadd_pd(w, wr, bOsr);
                bOsi = _mm512_fmadd_pd(w, wi, bOsi);
            }
            for (; j < he; ++j) { /* even-j run -> E accumulators */
                __m512d ur = _mm512_loadu_pd(bu + 16 * j);
                __m512d ui = _mm512_loadu_pd(bu + 16 * j + 8);
                __m512d wr = _mm512_loadu_pd(bv + 16 * j);
                __m512d wi = _mm512_loadu_pd(bv + 16 * j + 8);
                __m512d w;
                w = _mm512_set1_pd(c0[j]);
                aEcr = _mm512_fmadd_pd(w, ur, aEcr);
                aEci = _mm512_fmadd_pd(w, ui, aEci);
                w = _mm512_set1_pd(s0[j]);
                aEsr = _mm512_fmadd_pd(w, wr, aEsr);
                aEsi = _mm512_fmadd_pd(w, wi, aEsi);
                w = _mm512_set1_pd(c1[j]);
                bEcr = _mm512_fmadd_pd(w, ur, bEcr);
                bEci = _mm512_fmadd_pd(w, ui, bEci);
                w = _mm512_set1_pd(s1[j]);
                bEsr = _mm512_fmadd_pd(w, wr, bEsr);
                bEsi = _mm512_fmadd_pd(w, wi, bEsi);
            }
            DM_COMBQ(aEcr, aEci, aEsr, aEsi, aOcr, aOci, aOsr, aOsi, cbr, cbi, k);
            DM_COMBQ(bEcr, bEci, bEsr, bEsi, bOcr, bOci, bOsr, bOsi, cbr, cbi,
                     k + 2);
        }
        for (; k < q4; k += 2) { /* single-quad tail of the class */
            const double *c0 = p->Ct2 + (size_t)dm_rq2(p, k - 1) * hs;
            const double *s0 = p->St2 + (size_t)dm_rq2(p, k - 1) * hs;
            __m512d aEcr = _mm512_setzero_pd(), aEci = aEcr, aEsr = aEcr,
                    aEsi = aEcr;
            __m512d aOcr = aEcr, aOci = aEcr, aOsr = aEcr, aOsi = aEcr;
            int j = 0;
            for (; j < no; ++j) {
                __m512d w = _mm512_set1_pd(c0[j]);
                aOcr = _mm512_fmadd_pd(w, _mm512_loadu_pd(bu + 16 * j), aOcr);
                aOci = _mm512_fmadd_pd(w, _mm512_loadu_pd(bu + 16 * j + 8), aOci);
                w = _mm512_set1_pd(s0[j]);
                aOsr = _mm512_fmadd_pd(w, _mm512_loadu_pd(bv + 16 * j), aOsr);
                aOsi = _mm512_fmadd_pd(w, _mm512_loadu_pd(bv + 16 * j + 8), aOsi);
            }
            for (; j < he; ++j) {
                __m512d w = _mm512_set1_pd(c0[j]);
                aEcr = _mm512_fmadd_pd(w, _mm512_loadu_pd(bu + 16 * j), aEcr);
                aEci = _mm512_fmadd_pd(w, _mm512_loadu_pd(bu + 16 * j + 8), aEci);
                w = _mm512_set1_pd(s0[j]);
                aEsr = _mm512_fmadd_pd(w, _mm512_loadu_pd(bv + 16 * j), aEsr);
                aEsi = _mm512_fmadd_pd(w, _mm512_loadu_pd(bv + 16 * j + 8), aEsi);
            }
            DM_COMBQ(aEcr, aEci, aEsr, aEsi, aOcr, aOci, aOsr, aOsi, cbr, cbi, k);
        }
    }
#undef DM_COMBQ

    { /* lone k = L/4: pairs with itself under k -> L/2-k; r6-style single
       * pair over all he2 columns in its own parity class */
        const int lp = q4 & 1;
        const double *bu = lp ? uo : ue, *bv = lp ? vo : ve;
        const __m512d cbr = lp ? obr : ebr, cbi = lp ? obi : ebi;
        const double *c0 = p->Ct2 + (size_t)dm_rq2(p, q4 - 1) * hs;
        const double *s0 = p->St2 + (size_t)dm_rq2(p, q4 - 1) * hs;
        __m512d cr0 = _mm512_setzero_pd(), ci0 = cr0, sr0 = cr0, si0 = cr0;
        for (int j = 0; j < he; ++j) {
            __m512d w = _mm512_set1_pd(c0[j]);
            cr0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(bu + 16 * j), cr0);
            ci0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(bu + 16 * j + 8), ci0);
            w = _mm512_set1_pd(s0[j]);
            sr0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(bv + 16 * j), sr0);
            si0 = _mm512_fmadd_pd(w, _mm512_loadu_pd(bv + 16 * j + 8), si0);
        }
        __m512d t0 = _mm512_add_pd(cbr, cr0), t1 = _mm512_add_pd(cbi, ci0);
        dm_st(dr + (size_t)q4 * ostride, sm, nt, _mm512_add_pd(t0, si0));
        dm_st(di + (size_t)q4 * ostride, sm, nt, _mm512_sub_pd(t1, sr0));
        dm_st(dr + (size_t)(L - q4) * ostride, sm, nt, _mm512_sub_pd(t0, si0));
        dm_st(di + (size_t)(L - q4) * ostride, sm, nt, _mm512_add_pd(t1, sr0));
    }
}

/* even-L kernel dispatch: quad (4 | L) or the r6 parity kernel */
static inline void dm_keven(const fft3d_plan *p, const double *eb,
                            const double *ue, const double *uo,
                            const double *ve, const double *vo,
                            double *dr, double *di, size_t ostride, __mmask8 sm,
                            int nt)
{
    if (p->quadk)
        dm_kfold8q(p, eb, ue, uo, ve, vo, dr, di, ostride, sm, nt);
    else
        dm_kfold8e(p, eb, ue, uo, ve, vo, dr, di, ostride, sm, nt);
}

/* Axis-2 exit store (gen_r5).  One 8x8-complex chunk sits interleaved in
 * lo/hi (lane t = pencil row0+t, sites kc..kc+7).  cb == NULL: plain store
 * (execute()).  cb != NULL: w = z + c and the graded map w/(1+|w|) run in
 * registers (gl_map8, or gl_map16 pairing two pencils per divide under
 * GL_DEMO_MAP16) and the chain STATE is stored directly -- the separate map
 * pass and the zt volume it read are gone.  Masked-out lanes hold zeros (ob
 * rows >= L are never written and the arena is prefaulted zero), so the
 * ladder cannot trap on garbage; the store masks discard them. */
static inline void dm_exit8(double _Complex *dst, const double _Complex *cb,
                            size_t row0, int rcnt, int kc, int L,
                            __mmask8 m0, __mmask8 m1, int ntlo, int nthi,
                            int m4t, const __m512d lo[8], const __m512d hi[8])
{
    (void)m4t; /* referenced only under __AVX512VL__ */
    if (!cb) {
        for (int t = 0; t < rcnt; ++t) {
            double *o = (double *)(dst + (row0 + t) * (size_t)L + kc);
            dm_st(o, m0, ntlo, lo[t]);
            if (m1) dm_st(o + 8, m1, nthi, hi[t]);
        }
        return;
    }
    int t = 0;
#if !GL_DEMO_MAP16
#if !GL_DEMO_NOX8
    /* Full 8x8 chunk, all 8 rows live -- the dominant case at every L.  A
     * constant-bound loop (gen_r8) so the lo/hi indices are compile-time
     * constants after unrolling and the vectors stay in the registers
     * gl_tr8x8_c2i produced; the r7 spill audit blamed dm_exit8's dynamic
     * indexing for 46 rsp-relative zmm ops. */
    if (m0 == (__mmask8)0xFF && m1 == (__mmask8)0xFF && rcnt == 8) {
        for (int tt = 0; tt < 8; ++tt) {
            const double *ca = (const double *)(cb + (row0 + tt) * (size_t)L + kc);
            double *o = (double *)(dst + (row0 + tt) * (size_t)L + kc);
            __m512d z0 = _mm512_add_pd(lo[tt], _mm512_loadu_pd(ca));
            __m512d z1 = _mm512_add_pd(hi[tt], _mm512_loadu_pd(ca + 8));
            __m512d o0, o1;
            gl_map8(z0, z1, &o0, &o1);
            dm_st(o, (__mmask8)0xFF, ntlo, o0);
            dm_st(o + 8, (__mmask8)0xFF, nthi, o1);
        }
        return;
    }
#endif
    /* Tail-chunk map packing (gen_r7).  gl_map8's pair-compress keeps every
     * complex's result independent of its callmates (r5 gate evidence), so
     * partially-filled exit vectors of ADJACENT PENCILS can share one ladder
     * call bit-identically.  kcnt == 4 (L % 8 == 4: 12, 20, 36, 100...): each
     * row is exactly one full lo[] vector, hi[] dead -- pair two rows per
     * call, zero shuffles (the full 8-double c load is the row's 4 complex).
     * kcnt == 2 (L % 8 == 2: 10, 18, 50...): four rows of 2 complex pack via
     * two vinsertf64x4 and return by 256-bit halves (NT never applies here:
     * ntlo requires kcnt >= 4).  Dead-lane divides on these tails were the
     * r5-documented residual (+33% map divs at L=12, +28% at 25). */
    if (m0 == (__mmask8)0xFF && m1 == 0) {
        for (; t + 2 <= rcnt; t += 2) {
            const double *ca = (const double *)(cb + (row0 + t) * (size_t)L + kc);
            const double *cn =
                (const double *)(cb + (row0 + t + 1) * (size_t)L + kc);
            double *oa = (double *)(dst + (row0 + t) * (size_t)L + kc);
            double *on = (double *)(dst + (row0 + t + 1) * (size_t)L + kc);
            __m512d z0 = _mm512_add_pd(lo[t], _mm512_loadu_pd(ca));
            __m512d z1 = _mm512_add_pd(lo[t + 1], _mm512_loadu_pd(cn));
            __m512d o0, o1;
            gl_map8(z0, z1, &o0, &o1);
            dm_st(oa, m0, ntlo, o0);
            dm_st(on, m0, ntlo, o1);
        }
    } else if (m0 == (__mmask8)0x0F && m1 == 0) {
#if defined(__AVX512VL__) && !GL_DEMO_NOMAP4
        /* gen_r9, PMU-audit avenue 4 (port 1 idles everywhere): the kcnt=2
         * tail as pure ymm.  Each row is 2 complex = one ymm; two rows feed
         * one gl_map4 ladder (bit-identical per complex to gl_map8 -- the
         * selftest asserts it), c loads and stores are plain ymm.  Deletes
         * the r7 form's vinsertf64x4/vextractf64x4 pairs (port 5, the second
         * FMA pipe) and moves the ladder itself to ports 0/1, where port 1
         * is otherwise idle under 512-bit main work.  Plan-gated (p->m4t,
         * L <= 16): the ymm form runs ~2x the ladder uops per complex, so it
         * pays only where these tails are a large share of exit chunks --
         * measured -3..4% at L=10 but +1.5-2% at L=50 (SPR advisory). */
        if (m4t) for (; t + 2 <= rcnt; t += 2) {
            const double *ca = (const double *)(cb + (row0 + t) * (size_t)L + kc);
            const double *cn =
                (const double *)(cb + (row0 + t + 1) * (size_t)L + kc);
            __m256d z0 = _mm256_add_pd(_mm512_castpd512_pd256(lo[t]),
                                       _mm256_loadu_pd(ca));
            __m256d z1 = _mm256_add_pd(_mm512_castpd512_pd256(lo[t + 1]),
                                       _mm256_loadu_pd(cn));
            __m256d o0, o1;
            gl_map4(z0, z1, &o0, &o1);
            _mm256_storeu_pd((double *)(dst + (row0 + t) * (size_t)L + kc), o0);
            _mm256_storeu_pd((double *)(dst + (row0 + t + 1) * (size_t)L + kc),
                             o1);
        }
        else
#endif /* __AVX512VL__ && !GL_DEMO_NOMAP4 */
        for (; t + 4 <= rcnt; t += 4) {
            const double *c0p = (const double *)(cb + (row0 + t) * (size_t)L + kc);
            const double *c1p =
                (const double *)(cb + (row0 + t + 1) * (size_t)L + kc);
            const double *c2p =
                (const double *)(cb + (row0 + t + 2) * (size_t)L + kc);
            const double *c3p =
                (const double *)(cb + (row0 + t + 3) * (size_t)L + kc);
            __m512d z0 =
                _mm512_insertf64x4(lo[t], _mm512_castpd512_pd256(lo[t + 1]), 1);
            __m512d z1 = _mm512_insertf64x4(
                lo[t + 2], _mm512_castpd512_pd256(lo[t + 3]), 1);
            __m512d cz0 = _mm512_insertf64x4(
                _mm512_castpd256_pd512(_mm256_loadu_pd(c0p)),
                _mm256_loadu_pd(c1p), 1);
            __m512d cz1 = _mm512_insertf64x4(
                _mm512_castpd256_pd512(_mm256_loadu_pd(c2p)),
                _mm256_loadu_pd(c3p), 1);
            __m512d o0, o1;
            gl_map8(_mm512_add_pd(z0, cz0), _mm512_add_pd(z1, cz1), &o0, &o1);
            _mm256_storeu_pd((double *)(dst + (row0 + t) * (size_t)L + kc),
                             _mm512_castpd512_pd256(o0));
            _mm256_storeu_pd((double *)(dst + (row0 + t + 1) * (size_t)L + kc),
                             _mm512_extractf64x4_pd(o0, 1));
            _mm256_storeu_pd((double *)(dst + (row0 + t + 2) * (size_t)L + kc),
                             _mm512_castpd512_pd256(o1));
            _mm256_storeu_pd((double *)(dst + (row0 + t + 3) * (size_t)L + kc),
                             _mm512_extractf64x4_pd(o1, 1));
        }
    }
#if !GL_DEMO_NOK1
    /* kcnt == 1 (L % 8 == 1: 25 in-suite; 33, 41, 49... surprise class),
     * gen_r8: EIGHT rows of one complex share ONE ladder call.  Packing is
     * exact at any even-lane placement (the pair-compress never mixes
     * complexes); c enters via masked VBROADCASTF64X2 (a load-port uop, no
     * shuffle) and results leave by 128-bit extract stores.  8 dead-lane
     * divides (7/8 of the ladder work wasted) become 1 full one. */
    else if (m0 == (__mmask8)0x03 && m1 == 0) {
        const __m512i PK = _mm512_setr_epi64(0, 1, 8, 9, 0, 1, 8, 9);
        for (; t + 8 <= rcnt; t += 8) {
            __m512d p01 = _mm512_permutex2var_pd(lo[t], PK, lo[t + 1]);
            __m512d p23 = _mm512_permutex2var_pd(lo[t + 2], PK, lo[t + 3]);
            __m512d p45 = _mm512_permutex2var_pd(lo[t + 4], PK, lo[t + 5]);
            __m512d p67 = _mm512_permutex2var_pd(lo[t + 6], PK, lo[t + 7]);
            __m512d z0 = _mm512_shuffle_f64x2(p01, p23, 0x44);
            __m512d z1 = _mm512_shuffle_f64x2(p45, p67, 0x44);
#define DM_CP(T) ((const double *)(cb + (row0 + (T)) * (size_t)L + kc))
            __m512d cz0 = _mm512_maskz_broadcast_f64x2(
                (__mmask8)0x03, _mm_loadu_pd(DM_CP(t)));
            cz0 = _mm512_mask_broadcast_f64x2(cz0, (__mmask8)0x0C,
                                              _mm_loadu_pd(DM_CP(t + 1)));
            cz0 = _mm512_mask_broadcast_f64x2(cz0, (__mmask8)0x30,
                                              _mm_loadu_pd(DM_CP(t + 2)));
            cz0 = _mm512_mask_broadcast_f64x2(cz0, (__mmask8)0xC0,
                                              _mm_loadu_pd(DM_CP(t + 3)));
            __m512d cz1 = _mm512_maskz_broadcast_f64x2(
                (__mmask8)0x03, _mm_loadu_pd(DM_CP(t + 4)));
            cz1 = _mm512_mask_broadcast_f64x2(cz1, (__mmask8)0x0C,
                                              _mm_loadu_pd(DM_CP(t + 5)));
            cz1 = _mm512_mask_broadcast_f64x2(cz1, (__mmask8)0x30,
                                              _mm_loadu_pd(DM_CP(t + 6)));
            cz1 = _mm512_mask_broadcast_f64x2(cz1, (__mmask8)0xC0,
                                              _mm_loadu_pd(DM_CP(t + 7)));
#undef DM_CP
            __m512d o0, o1;
            gl_map8(_mm512_add_pd(z0, cz0), _mm512_add_pd(z1, cz1), &o0, &o1);
#define DM_OP(T) ((double *)(dst + (row0 + (T)) * (size_t)L + kc))
            _mm_storeu_pd(DM_OP(t), _mm512_castpd512_pd128(o0));
            _mm_storeu_pd(DM_OP(t + 1), _mm512_extractf64x2_pd(o0, 1));
            _mm_storeu_pd(DM_OP(t + 2), _mm512_extractf64x2_pd(o0, 2));
            _mm_storeu_pd(DM_OP(t + 3), _mm512_extractf64x2_pd(o0, 3));
            _mm_storeu_pd(DM_OP(t + 4), _mm512_castpd512_pd128(o1));
            _mm_storeu_pd(DM_OP(t + 5), _mm512_extractf64x2_pd(o1, 1));
            _mm_storeu_pd(DM_OP(t + 6), _mm512_extractf64x2_pd(o1, 2));
            _mm_storeu_pd(DM_OP(t + 7), _mm512_extractf64x2_pd(o1, 3));
#undef DM_OP
        }
    }
#endif
#endif
#if GL_DEMO_MAP16
    for (; t + 2 <= rcnt; t += 2) {
        const double *ca = (const double *)(cb + (row0 + t) * (size_t)L + kc);
        const double *cbn = (const double *)(cb + (row0 + t + 1) * (size_t)L + kc);
        double *oa = (double *)(dst + (row0 + t) * (size_t)L + kc);
        double *on = (double *)(dst + (row0 + t + 1) * (size_t)L + kc);
        __m512d z0 = _mm512_add_pd(lo[t], _mm512_maskz_loadu_pd(m0, ca));
        __m512d z1 = _mm512_add_pd(hi[t], _mm512_maskz_loadu_pd(m1, ca + 8));
        __m512d z2 = _mm512_add_pd(lo[t + 1], _mm512_maskz_loadu_pd(m0, cbn));
        __m512d z3 = _mm512_add_pd(hi[t + 1], _mm512_maskz_loadu_pd(m1, cbn + 8));
        __m512d o0, o1, o2, o3;
        gl_map16(z0, z1, z2, z3, &o0, &o1, &o2, &o3);
        dm_st(oa, m0, ntlo, o0);
        dm_st(oa + 8, m1, nthi, o1);
        dm_st(on, m0, ntlo, o2);
        dm_st(on + 8, m1, nthi, o3);
    }
#endif
    for (; t < rcnt; ++t) {
        const double *ca = (const double *)(cb + (row0 + t) * (size_t)L + kc);
        double *o = (double *)(dst + (row0 + t) * (size_t)L + kc);
        __m512d z0 = _mm512_add_pd(lo[t], _mm512_maskz_loadu_pd(m0, ca));
        __m512d z1 = _mm512_add_pd(hi[t], _mm512_maskz_loadu_pd(m1, ca + 8));
        __m512d o0, o1;
        gl_map8(z0, z1, &o0, &o1);
        dm_st(o, m0, ntlo, o0);
        if (m1) dm_st(o + 8, m1, nthi, o1);
    }
}

/* axis with contiguous inner index, interleaved source.  gen_r3: the folded
 * u/v rows are built straight from the source loads (row j and row L-j
 * deinterleaved, then add/sub) -- the r2 form staged all L rows into pb first
 * and re-read them, one full pb round trip per chunk.  Only row 0 still lands
 * in pb (the kernel reads x0 from there). */
static void dm_axis_i2s(const double _Complex *src, double *dr, double *di,
                        fft3d_plan *p, size_t inner)
{
    const int L = p->L, h = p->h, even = p->even;
    const int pf1 = p->pf1;
    double *pb = p->pb, *ub = p->ub, *vb = p->vb;
    for (size_t c = 0; c < inner; c += 8) {
        int cnt = (inner - c < 8) ? (int)(inner - c) : 8;
        __mmask8 sm = (cnt == 8) ? (__mmask8)0xFF : (__mmask8)((1u << cnt) - 1);
        __mmask8 m0 = (2 * cnt >= 8) ? (__mmask8)0xFF : (__mmask8)((1u << (2 * cnt)) - 1);
        __mmask8 m1 = (2 * cnt <= 8) ? (__mmask8)0 : (__mmask8)((1u << (2 * cnt - 8)) - 1);
#define DM_LDI(J, RE, IM)                                                        \
    do {                                                                         \
        const double *s_ = (const double *)(src + (size_t)(J)*inner + c);        \
        __m512d v0_ = _mm512_maskz_loadu_pd(m0, s_);                             \
        __m512d v1_ = _mm512_maskz_loadu_pd(m1, s_ + 8);                         \
        DM_PF2(s_ + 32, pf1);                                                    \
        DM_PF2(s_ + 40, pf1);                                                    \
        gl_deint8(v0_, v1_, &(RE), &(IM));                                       \
    } while (0)
        __m512d ar, ai, br, bi;
        if (p->evenk) { /* gen_r6: build the four k-parity blocks directly */
            const int q = L / 2, h2 = p->h2, no2 = p->no2;
            double *eb = p->ebb, *uo = p->uob, *vo = p->vob;
            __m512d gr, gi, er, ei;
            DM_LDI(0, ar, ai);
            DM_LDI(q, br, bi);
            _mm512_storeu_pd(eb, _mm512_add_pd(ar, br));
            _mm512_storeu_pd(eb + 8, _mm512_add_pd(ai, bi));
            _mm512_storeu_pd(eb + 16, _mm512_sub_pd(ar, br));
            _mm512_storeu_pd(eb + 24, _mm512_sub_pd(ai, bi));
            for (int j = 1; j <= h2; ++j) {
                const size_t o16 = 16 * (size_t)dm_pos2(j, no2);
                DM_LDI(j, ar, ai);
                DM_LDI(L - j, br, bi);
                DM_LDI(q - j, gr, gi);
                DM_LDI(q + j, er, ei);
                __m512d apr = _mm512_add_pd(ar, br), api = _mm512_add_pd(ai, bi);
                __m512d amr = _mm512_sub_pd(ar, br), ami = _mm512_sub_pd(ai, bi);
                __m512d bpr = _mm512_add_pd(gr, er), bpi = _mm512_add_pd(gi, ei);
                __m512d bmr = _mm512_sub_pd(gr, er), bmi = _mm512_sub_pd(gi, ei);
                _mm512_storeu_pd(ub + o16, _mm512_add_pd(apr, bpr));
                _mm512_storeu_pd(ub + o16 + 8, _mm512_add_pd(api, bpi));
                _mm512_storeu_pd(uo + o16, _mm512_sub_pd(apr, bpr));
                _mm512_storeu_pd(uo + o16 + 8, _mm512_sub_pd(api, bpi));
                _mm512_storeu_pd(vb + o16, _mm512_sub_pd(amr, bmr));
                _mm512_storeu_pd(vb + o16 + 8, _mm512_sub_pd(ami, bmi));
                _mm512_storeu_pd(vo + o16, _mm512_add_pd(amr, bmr));
                _mm512_storeu_pd(vo + o16 + 8, _mm512_add_pd(ami, bmi));
            }
            if (p->he2 > h2) { /* lone j = L/4 */
                const size_t o16 =
                    16 * (size_t)(((L / 4) & 1) ? no2 - 1 : p->he2 - 1);
                DM_LDI(L / 4, ar, ai);
                DM_LDI(q + L / 4, br, bi);
                __m512d pr = _mm512_add_pd(ar, br), pi = _mm512_add_pd(ai, bi);
                __m512d mr = _mm512_sub_pd(ar, br), mi = _mm512_sub_pd(ai, bi);
                _mm512_storeu_pd(ub + o16, pr);
                _mm512_storeu_pd(ub + o16 + 8, pi);
                _mm512_storeu_pd(uo + o16, pr);
                _mm512_storeu_pd(uo + o16 + 8, pi);
                _mm512_storeu_pd(vb + o16, mr);
                _mm512_storeu_pd(vb + o16 + 8, mi);
                _mm512_storeu_pd(vo + o16, mr);
                _mm512_storeu_pd(vo + o16 + 8, mi);
            }
            dm_keven(p, eb, ub, uo, vb, vo, dr + c, di + c, inner, sm,
                     p->nt0 && cnt == 8);
        } else {
            DM_LDI(0, ar, ai);
            _mm512_storeu_pd(pb, ar);
            _mm512_storeu_pd(pb + 8, ai);
            for (int j = 1; j <= h; ++j) {
                DM_LDI(j, ar, ai);
                DM_LDI(L - j, br, bi);
                _mm512_storeu_pd(ub + 16 * (j - 1), _mm512_add_pd(ar, br));
                _mm512_storeu_pd(ub + 16 * (j - 1) + 8, _mm512_add_pd(ai, bi));
                _mm512_storeu_pd(vb + 16 * (j - 1), _mm512_sub_pd(ar, br));
                _mm512_storeu_pd(vb + 16 * (j - 1) + 8, _mm512_sub_pd(ai, bi));
            }
            if (even) {
                DM_LDI(L / 2, ar, ai);
                _mm512_storeu_pd(ub + 16 * h, ar);
                _mm512_storeu_pd(ub + 16 * h + 8, ai);
            }
            dm_kfold8(p, pb, ub, vb, dr + c, di + c, inner, sm,
                      p->nt0 && cnt == 8);
        }
#undef DM_LDI
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
        if (p->evenk) {
            dm_fold8e(p, pb, p->ebb, p->ub, p->uob, p->vb, p->vob);
            dm_keven(p, p->ebb, p->ub, p->uob, p->vb, p->vob,
                     dr + c, di + c, inner, sm, 0);
        } else {
            dm_fold8(pb, L, p->h, p->even, p->ub, p->vb);
            dm_kfold8(p, pb, p->ub, p->vb, dr + c, di + c, inner, sm, 0);
        }
    }
}

/* fastest axis: 8 contiguous split pencils per block.  Stage via gl_tr8x8
 * (lane = pencil), fold, folded kernel into the ob block, transpose back and
 * interleave to the destination (gl_int8).  Replaces the r1 rowfast form,
 * which was load-bound (4 loads / 4 FMAs) and streamed the whole table per
 * SINGLE pencil; here every table broadcast feeds 8 pencils. */
static void dm_axis_z(const double *sr, const double *si, double _Complex *dst,
                      fft3d_plan *p, size_t nrows, const double _Complex *cb,
                      int ntd)
{
    const int L = p->L;
    double *pb = p->pb, *ob = p->ob;
    for (size_t r0 = 0; r0 < nrows; r0 += 8) {
        int rcnt = (nrows - r0 < 8) ? (int)(nrows - r0) : 8;
        /* stage: pb row j, lane t = pencil r0+t */
        for (int jc = 0; jc < L; jc += 8) {
            int jcnt = (L - jc < 8) ? (L - jc) : 8;
            __m512d re[8], im[8];
#if !GL_DEMO_NOLDT
            if (p->ldt && jcnt == 8 && rcnt == 8) { /* insert-load transpose */
                dm_tr8x8_ld(sr + r0 * (size_t)L + jc, (size_t)L, re);
                dm_tr8x8_ld(si + r0 * (size_t)L + jc, (size_t)L, im);
            } else
#endif
            {
                __mmask8 jm =
                    (jcnt == 8) ? (__mmask8)0xFF : (__mmask8)((1u << jcnt) - 1);
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
            }
            for (int q = 0; q < 8; ++q) {
                _mm512_storeu_pd(pb + 16 * (jc + q), re[q]);
                _mm512_storeu_pd(pb + 16 * (jc + q) + 8, im[q]);
            }
        }
        if (p->evenk) {
            dm_fold8e(p, pb, p->ebb, p->ub, p->uob, p->vb, p->vob);
            dm_keven(p, p->ebb, p->ub, p->uob, p->vb, p->vob,
                     ob, ob + 8, 16, (__mmask8)0xFF, 0);
        } else {
            dm_fold8(pb, L, p->h, p->even, p->ub, p->vb);
            dm_kfold8(p, pb, p->ub, p->vb, ob, ob + 8, 16, (__mmask8)0xFF, 0);
        }
        /* scatter: fused transpose+interleave (gl_tr8x8_c2i, gen_r4: 48
         * shuffles for what tr8x8 x2 + int8 x8 did in 64), then store */
        for (int kc = 0; kc < L; kc += 8) {
            int kcnt = (L - kc < 8) ? (L - kc) : 8;
            __mmask8 m0 = (2 * kcnt >= 8) ? (__mmask8)0xFF
                                          : (__mmask8)((1u << (2 * kcnt)) - 1);
            __mmask8 m1 = (2 * kcnt <= 8) ? (__mmask8)0
                                          : (__mmask8)((1u << (2 * kcnt - 8)) - 1);
            const int ntlo = ntd && kcnt >= 4, nthi = ntd && kcnt == 8;
            __m512d re[8], im[8], lo[8], hi[8];
            for (int q = 0; q < 8; ++q) {
                re[q] = _mm512_loadu_pd(ob + 16 * (kc + q));
                im[q] = _mm512_loadu_pd(ob + 16 * (kc + q) + 8);
            }
            gl_tr8x8_c2i(re, im, lo, hi);
            dm_exit8(dst, cb, r0, rcnt, kc, L, m0, m1, ntlo, nthi, p->m4t, lo, hi);
        }
    }
}

/* axis-2 rows [r0, r0+nrows) read from the 4-slot plane window instead of a
 * contiguous volume: row r lives at slot (r/L)&3, offset (r%L)*L.  Blocks of
 * 8 rows may span two window slots; every row pointer is derived per lane.
 * Output rows land contiguously in dst (dst is the real, full volume). */
static void dm_axis_z_win(fft3d_plan *p, size_t r0, size_t nrows,
                          double _Complex *dst, const double _Complex *cb,
                          int ntd)
{
    const int L = p->L;
    const size_t wp = p->wpitch;
    const double *wr = p->s2r, *wi = p->s2i;
    double *pb = p->pb, *ob = p->ob;
    for (size_t rb = r0; rb < r0 + nrows; rb += 8) {
        int rcnt = (r0 + nrows - rb < 8) ? (int)(r0 + nrows - rb) : 8;
        /* insert-load transpose needs 8 stride-L rows in ONE slot: same plane
         * (rows never wrap a plane boundary inside a slot) and a full block */
        const size_t q0 = rb / (size_t)L, z0 = rb - q0 * (size_t)L;
        const int contig = (rcnt == 8) && (z0 + 8 <= (size_t)L);
        const size_t cbase = (q0 & 3) * wp + z0 * (size_t)L;
        (void)contig;
        (void)cbase; /* unused under GL_DEMO_NOLDT */
        for (int jc = 0; jc < L; jc += 8) {
            int jcnt = (L - jc < 8) ? (L - jc) : 8;
            __m512d re[8], im[8];
#if !GL_DEMO_NOLDT
            if (p->ldt && contig && jcnt == 8) {
                dm_tr8x8_ld(wr + cbase + jc, (size_t)L, re);
                dm_tr8x8_ld(wi + cbase + jc, (size_t)L, im);
            } else
#endif
            {
                __mmask8 jm =
                    (jcnt == 8) ? (__mmask8)0xFF : (__mmask8)((1u << jcnt) - 1);
                for (int t = 0; t < 8; ++t) {
                    if (t < rcnt) {
                        size_t r = rb + t, q = r / (size_t)L;
                        size_t off =
                            (q & 3) * wp + (r - q * (size_t)L) * (size_t)L + jc;
                        re[t] = _mm512_maskz_loadu_pd(jm, wr + off);
                        im[t] = _mm512_maskz_loadu_pd(jm, wi + off);
                    } else {
                        re[t] = _mm512_setzero_pd();
                        im[t] = _mm512_setzero_pd();
                    }
                }
                gl_tr8x8(re);
                gl_tr8x8(im);
            }
            for (int q = 0; q < 8; ++q) {
                _mm512_storeu_pd(pb + 16 * (jc + q), re[q]);
                _mm512_storeu_pd(pb + 16 * (jc + q) + 8, im[q]);
            }
        }
        if (p->evenk) {
            dm_fold8e(p, pb, p->ebb, p->ub, p->uob, p->vb, p->vob);
            dm_keven(p, p->ebb, p->ub, p->uob, p->vb, p->vob,
                     ob, ob + 8, 16, (__mmask8)0xFF, 0);
        } else {
            dm_fold8(pb, L, p->h, p->even, p->ub, p->vb);
            dm_kfold8(p, pb, p->ub, p->vb, ob, ob + 8, 16, (__mmask8)0xFF, 0);
        }
        for (int kc = 0; kc < L; kc += 8) {
            int kcnt = (L - kc < 8) ? (L - kc) : 8;
            __mmask8 m0 = (2 * kcnt >= 8) ? (__mmask8)0xFF
                                          : (__mmask8)((1u << (2 * kcnt)) - 1);
            __mmask8 m1 = (2 * kcnt <= 8) ? (__mmask8)0
                                          : (__mmask8)((1u << (2 * kcnt - 8)) - 1);
            const int ntlo = ntd && kcnt >= 4, nthi = ntd && kcnt == 8;
            __m512d re[8], im[8], lo[8], hi[8];
            for (int q = 0; q < 8; ++q) {
                re[q] = _mm512_loadu_pd(ob + 16 * (kc + q));
                im[q] = _mm512_loadu_pd(ob + 16 * (kc + q) + 8);
            }
            gl_tr8x8_c2i(re, im, lo, hi);
            dm_exit8(dst, cb, rb, rcnt, kc, L, m0, m1, ntlo, nthi, p->m4t, lo, hi);
        }
    }
}

/* Fused y then z sweep, L >= 8 (gen_r3).  Axis-1 y-pencils are indexed
 * globally by g = x*L + z and processed 8 per vector group; a group crossing
 * the plane boundary takes lanes 0..cnt-1 from plane x (columns z..L-1) and
 * lanes cnt..7 from plane x+1 (columns 0..7-cnt) with a second masked
 * load/store -- no dead lanes anywhere but the single final tail (odd L^2).
 * Results go into a 4-slot circular plane window (pitch from
 * dm_pick_wpitch); axis-2 trails plane-by-plane on rows already complete, so
 * the window stays cache-hot and the old full s2 volume never exists.  The
 * live span is 3 consecutive planes (writer x,x+1; reader >= x-1), so 4
 * slots cannot collide. */
static void dm_axes_yz(fft3d_plan *p, double _Complex *dst,
                       const double _Complex *cb, int ntd)
{
    const int L = p->L, h = p->h, even = p->even;
    const int pf1 = p->pf1;
    const size_t LL = (size_t)L * L, wp = p->wpitch;
    const double *sr = p->s1r, *si = p->s1i;
    double *wr = p->s2r, *wi = p->s2i;
    double *pb = p->pb, *ob = p->ob, *ub = p->ub, *vb = p->vb;
    size_t g = 0, r2 = 0;
    while (g < LL) {
        const size_t x = g / (size_t)L;
        const int z = (int)(g - x * (size_t)L);
        const size_t rem = LL - g;
        const int tot = rem < 8 ? (int)rem : 8;
        int cnt = L - z;
        if (cnt > tot) cnt = tot;
        const __mmask8 mall = (tot == 8) ? (__mmask8)0xFF : (__mmask8)((1u << tot) - 1);
        const __mmask8 mlo = (cnt == 8) ? (__mmask8)0xFF : (__mmask8)((1u << cnt) - 1);
        const __mmask8 mhi = (__mmask8)(mall & (__mmask8)~mlo);
        const double *b1r = sr + x * LL + z, *b1i = si + x * LL + z;
        const double *b2r = sr + (x + 1) * LL - cnt, *b2i = si + (x + 1) * LL - cnt;
#define DM_LDS(J, RE, IM)                                                        \
    do {                                                                         \
        const size_t o_ = (size_t)(J) * (size_t)L;                               \
        (RE) = _mm512_maskz_loadu_pd(mlo, b1r + o_);                             \
        (IM) = _mm512_maskz_loadu_pd(mlo, b1i + o_);                             \
        DM_PF2(b1r + o_ + 16, pf1);                                              \
        DM_PF2(b1i + o_ + 16, pf1);                                              \
        if (mhi) {                                                               \
            (RE) = _mm512_mask_loadu_pd((RE), mhi, b2r + o_);                    \
            (IM) = _mm512_mask_loadu_pd((IM), mhi, b2i + o_);                    \
        }                                                                        \
    } while (0)
        __m512d ar, ai, br, bi;
        if (p->evenk) { /* gen_r6: build the four k-parity blocks directly */
            const int q = L / 2, h2 = p->h2, no2 = p->no2;
            double *eb = p->ebb, *uo = p->uob, *vo = p->vob;
            __m512d gr, gi, er, ei;
            DM_LDS(0, ar, ai);
            DM_LDS(q, br, bi);
            _mm512_storeu_pd(eb, _mm512_add_pd(ar, br));
            _mm512_storeu_pd(eb + 8, _mm512_add_pd(ai, bi));
            _mm512_storeu_pd(eb + 16, _mm512_sub_pd(ar, br));
            _mm512_storeu_pd(eb + 24, _mm512_sub_pd(ai, bi));
            for (int j = 1; j <= h2; ++j) {
                const size_t o16 = 16 * (size_t)dm_pos2(j, no2);
                DM_LDS(j, ar, ai);
                DM_LDS(L - j, br, bi);
                DM_LDS(q - j, gr, gi);
                DM_LDS(q + j, er, ei);
                __m512d apr = _mm512_add_pd(ar, br), api = _mm512_add_pd(ai, bi);
                __m512d amr = _mm512_sub_pd(ar, br), ami = _mm512_sub_pd(ai, bi);
                __m512d bpr = _mm512_add_pd(gr, er), bpi = _mm512_add_pd(gi, ei);
                __m512d bmr = _mm512_sub_pd(gr, er), bmi = _mm512_sub_pd(gi, ei);
                _mm512_storeu_pd(ub + o16, _mm512_add_pd(apr, bpr));
                _mm512_storeu_pd(ub + o16 + 8, _mm512_add_pd(api, bpi));
                _mm512_storeu_pd(uo + o16, _mm512_sub_pd(apr, bpr));
                _mm512_storeu_pd(uo + o16 + 8, _mm512_sub_pd(api, bpi));
                _mm512_storeu_pd(vb + o16, _mm512_sub_pd(amr, bmr));
                _mm512_storeu_pd(vb + o16 + 8, _mm512_sub_pd(ami, bmi));
                _mm512_storeu_pd(vo + o16, _mm512_add_pd(amr, bmr));
                _mm512_storeu_pd(vo + o16 + 8, _mm512_add_pd(ami, bmi));
            }
            if (p->he2 > h2) { /* lone j = L/4 */
                const size_t o16 =
                    16 * (size_t)(((L / 4) & 1) ? no2 - 1 : p->he2 - 1);
                DM_LDS(L / 4, ar, ai);
                DM_LDS(q + L / 4, br, bi);
                __m512d pr = _mm512_add_pd(ar, br), pi = _mm512_add_pd(ai, bi);
                __m512d mr = _mm512_sub_pd(ar, br), mi = _mm512_sub_pd(ai, bi);
                _mm512_storeu_pd(ub + o16, pr);
                _mm512_storeu_pd(ub + o16 + 8, pi);
                _mm512_storeu_pd(uo + o16, pr);
                _mm512_storeu_pd(uo + o16 + 8, pi);
                _mm512_storeu_pd(vb + o16, mr);
                _mm512_storeu_pd(vb + o16 + 8, mi);
                _mm512_storeu_pd(vo + o16, mr);
                _mm512_storeu_pd(vo + o16 + 8, mi);
            }
        } else {
            DM_LDS(0, ar, ai);
            _mm512_storeu_pd(pb, ar);
            _mm512_storeu_pd(pb + 8, ai);
            for (int j = 1; j <= h; ++j) {
                DM_LDS(j, ar, ai);
                DM_LDS(L - j, br, bi);
                _mm512_storeu_pd(ub + 16 * (j - 1), _mm512_add_pd(ar, br));
                _mm512_storeu_pd(ub + 16 * (j - 1) + 8, _mm512_add_pd(ai, bi));
                _mm512_storeu_pd(vb + 16 * (j - 1), _mm512_sub_pd(ar, br));
                _mm512_storeu_pd(vb + 16 * (j - 1) + 8, _mm512_sub_pd(ai, bi));
            }
            if (even) {
                DM_LDS(L / 2, ar, ai);
                _mm512_storeu_pd(ub + 16 * h, ar);
                _mm512_storeu_pd(ub + 16 * h + 8, ai);
            }
        }
#undef DM_LDS
        double *w1r = wr + (x & 3) * wp + z, *w1i = wi + (x & 3) * wp + z;
        if (!mhi) {
            if (p->evenk)
                dm_keven(p, p->ebb, ub, p->uob, vb, p->vob,
                         w1r, w1i, (size_t)L, mall, 0);
            else
                dm_kfold8(p, pb, ub, vb, w1r, w1i, (size_t)L, mall, 0);
        } else { /* boundary group: kernel into ob, two masked stores per row */
            if (p->evenk)
                dm_keven(p, p->ebb, ub, p->uob, vb, p->vob,
                         ob, ob + 8, 16, (__mmask8)0xFF, 0);
            else
                dm_kfold8(p, pb, ub, vb, ob, ob + 8, 16, (__mmask8)0xFF, 0);
            double *w2r = wr + ((x + 1) & 3) * wp - cnt;
            double *w2i = wi + ((x + 1) & 3) * wp - cnt;
            for (int k = 0; k < L; ++k) {
                __m512d qr = _mm512_loadu_pd(ob + 16 * k);
                __m512d qi = _mm512_loadu_pd(ob + 16 * k + 8);
                _mm512_mask_storeu_pd(w1r + (size_t)k * L, mlo, qr);
                _mm512_mask_storeu_pd(w2r + (size_t)k * L, mhi, qr);
                _mm512_mask_storeu_pd(w1i + (size_t)k * L, mlo, qi);
                _mm512_mask_storeu_pd(w2i + (size_t)k * L, mhi, qi);
            }
        }
        g += tot;
        {
            size_t ready = (g >= LL) ? LL : (g / (size_t)L) * (size_t)L;
            size_t nrun = (ready - r2) & ~(size_t)7;
            if (g >= LL) nrun = ready - r2; /* flush the masked tail */
            if (nrun) {
                dm_axis_z_win(p, r2, nrun, dst, cb, ntd);
                r2 += nrun;
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

/* scalar exit: same fused-map option, exact sqrt (correctness builds only) */
static void dm_axis_z(const double *sr, const double *si, double _Complex *dst,
                      fft3d_plan *p, size_t nrows, const double _Complex *cb)
{
    const int L = p->L;
    double yr[128], yi[128];
    for (size_t r = 0; r < nrows; ++r) {
        dm_pencil_sc(p, sr + r * (size_t)L, si + r * (size_t)L, yr, yi);
        for (int k = 0; k < L; ++k) {
            double re = yr[k], im = yi[k];
            if (cb) {
                re += creal(cb[r * (size_t)L + k]);
                im += cimag(cb[r * (size_t)L + k]);
                const double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                re *= sc;
                im *= sc;
            }
            dst[r * (size_t)L + k] = re + I * im;
        }
    }
}

#endif /* __AVX512F__ */

/* One volume: interleaved src -> interleaved dst through the split scratch.
 * cb == NULL: plain FFT (execute()).  cb != NULL (gen_r5): the graded map
 * (z + cb, then w/(1+|w|)) is fused into the axis-2 exit and dst receives the
 * chain STATE directly -- legal with dst == src because axis 0 fully consumes
 * src into s1 before axis 2 writes dst. */
static void dm_vol_fft(fft3d_plan *p, const double _Complex *src,
                       double _Complex *dst, const double _Complex *cb)
{
    const int L = p->L;
    const size_t LL = (size_t)L * L;
    /* axis 0 (x): interleaved in -> split s1, inner = L^2 contiguous */
    dm_axis_i2s(src, p->s1r, p->s1i, p, LL);
#if defined(__AVX512F__)
    /* NT to dst only when it is provably 64B-aligned (driver owns it) */
    const int ntd = p->nt2 && (((uintptr_t)dst & 63) == 0);
    if (p->nt0) _mm_sfence(); /* NT s1 stores drain before axis 1 reads them */
    if (p->wpitch) { /* L >= 8: packed axis-1 groups + trailing axis-2 window */
        dm_axes_yz(p, dst, cb, ntd);
        if (ntd) _mm_sfence();
        return;
    }
#endif
    /* axis 1 (y): per x-plane, split s1 -> split s2, inner = L */
    for (int x = 0; x < L; ++x)
        dm_axis_s2s(p->s1r + (size_t)x * LL, p->s1i + (size_t)x * LL,
                    p->s2r + (size_t)x * LL, p->s2i + (size_t)x * LL,
                    p, (size_t)L);
    /* axis 2 (z): 8 contiguous split pencils per block -> interleaved out */
#if defined(__AVX512F__)
    dm_axis_z(p->s2r, p->s2i, dst, p, LL, cb, ntd);
    if (ntd) _mm_sfence();
#else
    dm_axis_z(p->s2r, p->s2i, dst, p, LL, cb);
#endif
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    dm_placement(p, in, out);
    for (int b = 0; b < p->batch; ++b)
        dm_vol_fft(p, in + (size_t)b * p->V, out + (size_t)b * p->V, NULL);
}

#if GL_DEMO_NOMAPFUSE
/* r4-shape A/B arm: graded map as a separate pass, state <- (z+c)/(1+|z+c|).
 * The vector body is exactly gl_map8 (see the library, section 7). */
static void dm_map_vol(const double _Complex *z, const double _Complex *c,
                       double _Complex *o, size_t V, int nt)
{
    const double *za = (const double *)z, *ca = (const double *)c;
    double *oa = (double *)o;
    size_t n = 2 * V, i = 0;
    (void)nt;
#if defined(__AVX512F__)
    nt = nt && ((uintptr_t)oa & 63) == 0; /* driver owns o: verify at runtime */
    for (; i + 16 <= n; i += 16) {
        __m512d z0 = _mm512_add_pd(_mm512_loadu_pd(za + i), _mm512_loadu_pd(ca + i));
        __m512d z1 = _mm512_add_pd(_mm512_loadu_pd(za + i + 8),
                                   _mm512_loadu_pd(ca + i + 8));
        __m512d o0, o1;
        gl_map8(z0, z1, &o0, &o1);
        dm_st(oa + i, (__mmask8)0xFF, nt, o0);
        dm_st(oa + i + 8, (__mmask8)0xFF, nt, o1);
    }
    if (nt) _mm_sfence();
#endif
    for (; i < n; i += 2) { /* exact scalar tail (V*2 % 16 != 0 at odd L) */
        double re = za[i] + ca[i], im = za[i + 1] + ca[i + 1];
        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
        oa[i] = re * sc;
        oa[i + 1] = im * sc;
    }
}
#endif /* GL_DEMO_NOMAPFUSE */

/* Own the graded m-step chain (fallback pays a memcpy + a scalar-sqrt map pass
 * per step).  Volume-major (the corpus consensus): each volume runs all m steps
 * while its working set is cache-hot.  state_b lives in final_out; step 1 reads
 * x0 directly, so nothing is ever copied.  gen_r5: the map runs in registers at
 * the axis-2 exit and the state is rewritten IN PLACE -- the zt volume and the
 * separate map pass (a full state round trip per step) are gone.  In-place is
 * legal because axis 0 fully consumes sb into s1 before axis 2 rewrites sb. */
void fft3d_chain(fft3d_plan *p, const double _Complex *x0,
                 const double _Complex *c, double _Complex *final_out, int m)
{
#if !GL_DEMO_NOMAPFUSE
    /* gen_r11: when the driver's state buffer is measured 4K-backed (smaps,
     * once per buffer pair -- on THP=madvise hosts posix_memalign volumes get
     * no huge pages at any size), run steps 1..m-1 with the state in the THP
     * arena volume instead.  Zero extra copies for the state: step 1 reads x0
     * directly (as before) and the LAST step's exit writes final_out directly;
     * c is staged into the arena once per chain call (one volume copy vs m
     * strided re-reads through 4K pages).  Arithmetic, operation order and
     * output are bit-identical to the in-place path -- only addresses change. */
    if (p->stv && m >= 2) {
        if (x0 != p->rh_in || final_out != p->rh_out) {
            const size_t ob = p->V * sizeof(double _Complex) * (size_t)p->batch;
            const long long hb = gl_thp_bytes(final_out, ob);
            p->rh_use = (hb >= 0 && (unsigned long long)hb < ob / 2);
            p->rh_in = x0;
            p->rh_out = final_out;
        }
        if (p->rh_use) {
            const size_t vbytes = p->V * sizeof(double _Complex);
            dm_placement(p, p->stv, p->cv ? (const void *)p->cv : (const void *)c);
            for (int b = 0; b < p->batch; ++b) {
                const double _Complex *cb = c + (size_t)b * p->V;
                double _Complex *sb = final_out + (size_t)b * p->V;
                if (p->cv) {
                    memcpy(p->cv, cb, vbytes);
                    cb = p->cv;
                }
                dm_vol_fft(p, x0 + (size_t)b * p->V, p->stv, cb);
                for (int s = 1; s < m; ++s)
                    dm_vol_fft(p, p->stv, s == m - 1 ? sb : p->stv, cb);
            }
            return;
        }
    }
#endif
    dm_placement(p, x0, final_out);
    for (int b = 0; b < p->batch; ++b) {
        const double _Complex *cb = c + (size_t)b * p->V;
        double _Complex *sb = final_out + (size_t)b * p->V;
#if GL_DEMO_NOMAPFUSE
        dm_vol_fft(p, x0 + (size_t)b * p->V, p->zt, NULL);
        dm_map_vol(p->zt, cb, sb, p->V, p->nt2);
        for (int s = 1; s < m; ++s) {
            dm_vol_fft(p, sb, p->zt, NULL);
            dm_map_vol(p->zt, cb, sb, p->V, p->nt2);
        }
#else
        dm_vol_fft(p, x0 + (size_t)b * p->V, sb, cb);
        for (int s = 1; s < m; ++s)
            dm_vol_fft(p, sb, sb, cb);
#endif
    }
}

#endif /* GEN_LAYOUT_LIB_ONLY */
