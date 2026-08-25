/* gen_dense_prime -- direct dense prime-class entry, round gen_r4.
 *
 * gen_r4: two changes to the L=31 chain engine.
 * (1) LAZY MAP FUSION (map-on-load): the standalone per-step map pass is
 *     gone; the map is applied to each state row AS THE NEXT STEP'S z-phase
 *     LOADS it (fold31zx map variant).  Every eager map fusion on this panel
 *     lost (my r2 plane-map twice, gen_rader's r1/r3 store-side fusions) --
 *     those inject the ladder+divide into a STORE drain and blow the store
 *     buffer.  Load-side fusion is the opposite geometry: no extra in-flight
 *     stores, and the map pass's 2 MB/step of L2 traffic (state re-read +
 *     mapped store) simply disappears; only the last step's map materializes
 *     (one map_volume sweep per chain).  Arithmetic is bit-identical to
 *     map_volume (same 8-point grouping, same op order).
 * (2) zmm z-ROW FOLD: the z-phase's per-row u/v fold was 15 xmm iterations
 *     (90 uops/row); on the padded 32-complex rows it is now 8 zmm loads +
 *     4 lane-reversal shuffles + 8 add/sub + 8 aligned stores (~28 uops),
 *     mask-free, with the junk confined to the never-read j=0 slot.
 *
 * gen_r3: the gen_rader-prescribed 4K-anti-alias pitch was BUILT AND RACED
 * (GDP_PP plane pitch + GDP_BG permuted block order knobs below) and LOST
 * ~0-4% on this engine despite the collision model scoring 240 -> 4.4: the
 * z-transform phase between one block's GEMM drain and the next block's
 * loads retires the in-flight stores, so the replays never bind, while the
 * bigger footprint (0.96 -> 1.07 MB vs 1.25 MB L2) does.  Defaults stay at
 * the r2 layout (992/1); knobs kept raceable.  What DID ship: the c mirror
 * moved into the state arena at page phase +2048 (two separate big
 * aligned_allocs land at the SAME phase, aliasing the map), and the
 * z-combine now fuses straight into the U/V fold (zA/zB stack round-trip
 * deleted: 32 memory ops per row-pair off the z+x pass).
 *
 * gen_r3: class duty widened -- supports() accepts ANY prime p <= 31 (the
 * class), plus the roster composites 10/12/15/20; every non-31 size now
 * runs a VECTORIZED z-pass (k-in-zmm row pairs, duplicated-pair tables,
 * masked half-spectrum stores) instead of the scalar rows of r1/r2.
 *
 * Conjugate-pair FOLDED dense DFT per axis (the L13/L17/L23 winning arithmetic
 * from the ice campaign): for each pencil, u_j = x_j + x_{L-j}, v_j = x_j - x_{L-j},
 * then X_k = C_k - i S_k and X_{L-k} = C_k + i S_k with C_k = x_0 + sum_j u_j cos(2pi jk/L)
 * and S_k = sum_j v_j sin(2pi jk/L) -- all constants REAL, so every multiply on
 * interleaved complex data is a plain real FMA.  ~4h^2 real FMAs per pencil
 * (h = floor(L/2)) instead of the 8h^2 of the unfolded dense matrix.
 *
 * gen_r2: every pass is IN-PLACE SAFE by construction (all src reads of a
 * column block are buffered before any dst store), so the chain step runs
 * z, x, y AND the map fully in place on the state volume: working set =
 * state + c = 953 KB at L=31, L2-resident (borrowed from gen_rader gen_r1,
 * who measured -16 us from exactly this on their engine).  t1 is gone;
 * execute() does z: in -> out, then x/y in place on out.
 *
 * gen_r2: fold31 -- the L=31 x/y contraction specialized at compile time:
 * U/V/Cblk rows padded to 64 doubles (32 complex) so every hot GEMM load is
 * a full aligned 64-byte vector.  The generic path's 496-byte rows made 3/4
 * of all GEMM loads cache-line splits; at 0.5 loads/FMA that made the port
 * 2/3 feed, not the FMA pipes, the binding resource.  All masks are now
 * compile-time (only the last 1/8 store of a tail tile is masked).
 *
 * fft3d_chain: volume-resident chain (each volume runs all m steps while its
 * buffers stay cache-resident -- corpus consensus), map fused as a separate
 * L2-hot pass with the ice s6 arithmetic: pair-compressed |w|^2, rsqrt14 seed
 * + 2 Newton, d = fma(m2, r, 1), ONE vdivpd per 8 points, two mul-outs.
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#include "../fft3d_api.h"

typedef double _Complex cplx;

static const long double PIL = 3.141592653589793238462643383279502884L;

struct fft3d_plan {
    int L, batch;
    int hc;              /* # of C rows/outputs beyond k=0: L/2 (even) or (L-1)/2 (odd) */
    int hs;              /* # of S rows/outputs: (L-1)/2 */
    double *ct;          /* cos table [hc][hc+1]: ct[(j-1)*(hc+1)+k] = cos(2pi jk/L) */
    double *st;          /* sin table [hs][hs+1]: st[(j-1)*(hs+1)+k] = sin(2pi jk/L) */
    double *ctd, *std_;  /* L=31 z-pass duplicated-pair tables, [15][32] doubles each */
    double *U, *V;       /* fold buffers; L=31: 15 rows x 64 doubles, padded+aligned */
    double *Cb;          /* C-block buffer; L=31: 16 rows x 64 doubles */
    cplx *Sb;            /* scalar-path S accumulator */
    cplx *sp, *cp;       /* L=31 chain: PADDED state/c volumes, 31 planes of pitch
                            GDP_PP cplx (rows of 32), one arena, cp at page phase
                            +2048; every hot load/store 64B-aligned */
    double *ctz, *stz;   /* generic vector z-pass (gen_r3): duplicated-pair
                            tables [hc][8*kq] / [hs][8*kq], slots k > kmax
                            zeroed so the FMAs are benign */
    int kq;              /* zmm vectors covering k = 0..hc (4 complex each) */
    int pl;              /* gen_r4 generic padded chain: row pitch in complex
                            ((L+3)&~3, 64B multiple) for 5 <= L <= 30; sp/cp
                            hold L x L x pl volumes, every row 64B-aligned,
                            all fold_pass masks degenerate to full vectors.
                            0 = flat chain (L < 5, or no AVX-512) */
    unsigned char zfm[4];/* first-half store mask per vector */
    unsigned char zsm[4];/* second-half (reversed) store mask per vector */
    int zso2[4];         /* second-half store base offset (doubles) */
    void *abase;         /* L=31: mmap base of the 2 MiB-huge-page arena that
                            holds sp/cp/U/V/Cb/ctd/std_/ct/st at FIXED offsets
                            (identical binary read 124.0-130.4 us/step across
                            process instances from allocation layout luck --
                            the arena makes the layout deterministic and the
                            THP backing makes the L2 set mapping uniform;
                            gl_map_huge/gl_arena lesson, gen_layout) */
    size_t alen;
};

const char *fft3d_name(void) { return "gen_dense_prime"; }
const char *fft3d_description(void)
{
    return "folded dense prime p<=31 (any prime in class supported): 4h^2-FMA "
           "conjugate-pair fold, z-pass fused into the x-contraction with the "
           "z-combine folded straight into U/V (no stack round-trip), fully "
           "in-place L2-resident chain on a padded 31x31x32 state (64B-aligned, "
           "mask-free), register-tiled GEMM, vectorized any-L z-pass, zmm z-row "
           "fold, LAZY map fused into the next step's z-loads (s6 arithmetic, "
           "one vdivpd/8pts; only the last step's map materializes)";
}
int fft3d_supports(int L)
{
    if (L == 10 || L == 12 || L == 15 || L == 20) return 1; /* roster composites */
    if (L < 2 || L > 31) return 0;
    for (int d = 2; d * d <= L; ++d)
        if (L % d == 0) return 0;
    return 1;                                               /* any prime <= 31 */
}

/* L=31 chain-arena geometry.  GDP_PP = plane pitch in complex, GDP_BG =
 * x-pass block-visit stride (need gcd(GDP_BG,31)=1; blocks are independent
 * columns so any order is legal).  The 4K-anti-alias combinations
 * (1108/4 score 4.4, 1004/4 score 8.6 in the gl_alias_pairs4k model vs
 * 240 for 992/1) were raced on the node and did NOT beat the r2 layout --
 * see the strategy record.  Knobs kept for cross-arch re-racing. */
#ifndef GDP_PP
#define GDP_PP 992
#endif
#ifndef GDP_BG
#define GDP_BG 1
#endif
_Static_assert(GDP_PP >= 992 && GDP_PP % 4 == 0, "GDP_PP: >= 31 rows of 32, 64B-aligned");

/* ---------------- tables ---------------- */

static void trig_fill(double *t, int L, int rows, int cols, int want_sin)
{
    for (int j = 1; j <= rows; ++j)
        for (int k = 0; k < cols; ++k) {
            long m = ((long)j * k) % L;
            long double th = 2.0L * PIL * (long double)m / (long double)L;
            t[(size_t)(j - 1) * cols + k] = want_sin ? (double)sinl(th) : (double)cosl(th);
        }
}

static double *trig_table(int L, int rows, int cols, int want_sin)
{
    size_t n = (size_t)rows * cols;
    if (n == 0) n = 8;   /* L=2: hs=0 -> zero sin rows; keep a real allocation */
    double *t = aligned_alloc(64, ((n * sizeof(double)) + 63) & ~(size_t)63);
    if (!t) return NULL;
    trig_fill(t, L, rows, cols, want_sin);
    return t;
}

/* duplicated-pair layout for the L=31 vector z-pass: row j-1 holds
 * (w_{j,0}, w_{j,0}, w_{j,1}, w_{j,1}, ..., w_{j,15}, w_{j,15})  = 32 doubles */
static void trig_fill_dup31(double *t, int want_sin)
{
    for (int j = 1; j <= 15; ++j)
        for (int k = 0; k <= 15; ++k) {
            long m = ((long)j * k) % 31;
            long double th = 2.0L * PIL * (long double)m / 31.0L;
            double w = want_sin ? (double)sinl(th) : (double)cosl(th);
            t[(size_t)(j - 1) * 32 + 2 * k]     = w;
            t[(size_t)(j - 1) * 32 + 2 * k + 1] = w;
        }
}

/* duplicated-pair table for the generic vector z-pass: row j-1 holds
 * (w_{j,0}, w_{j,0}, ..., w_{j,4kq-1}, w_{j,4kq-1}); slots k > kmax are
 * ZERO so the pad-lane FMAs leave their accumulators benign (masked at the
 * stores anyway). */
static double *trig_table_dupz(int L, int rows, int kq, int kmax, int want_sin)
{
    size_t n = (size_t)(rows ? rows : 1) * kq * 8;
    double *t = aligned_alloc(64, n * sizeof(double));
    if (!t) return NULL;
    memset(t, 0, n * sizeof(double));
    for (int j = 1; j <= rows; ++j)
        for (int k = 0; k <= kmax && k < 4 * kq; ++k) {
            long m = ((long)j * k) % L;
            long double th = 2.0L * PIL * (long double)m / (long double)L;
            double w = want_sin ? (double)sinl(th) : (double)cosl(th);
            t[(size_t)(j - 1) * kq * 8 + 2 * k]     = w;
            t[(size_t)(j - 1) * kq * 8 + 2 * k + 1] = w;
        }
    return t;
}

/* 2 MiB-aligned anonymous mapping, MADV_HUGEPAGE, prefaulted so first-touch
 * faults (and, in madvise THP mode, the huge-page promotion) land in
 * create().  Returns the aligned pointer; base and maplen for munmap. */
static void *gdp_huge(size_t bytes, void **base, size_t *maplen)
{
    const size_t TWO_M = (size_t)2 << 20;
    size_t len = bytes + TWO_M;
    void *m = mmap(NULL, len, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return NULL;
    void *a = (void *)(((uintptr_t)m + TWO_M - 1) & ~(uintptr_t)(TWO_M - 1));
#ifdef MADV_HUGEPAGE
    madvise(a, bytes, MADV_HUGEPAGE);
#endif
    memset(a, 0, bytes);
    *base = m;
    *maplen = len;
    return a;
}

/* ---------------- folded contraction of the SLOWEST axis ----------------
 * src is an (L x inner) complex matrix, row stride = inner; dst likewise.
 * Processes inner in blocks of BC (BC must divide inner).
 * IN-PLACE SAFE (src == dst allowed): within a column block, every src read
 * (fold into U/V, x0 copy, even-L middle row) happens before any dst store.
 *
 * AVX-512 form: two register-tiled GEMMs per block, j innermost with the
 * accumulators held in registers (the round-1 axpy form re-loaded and
 * re-stored the accumulator every j: 155 us/pass -> this shape).
 *   C-GEMM:  Cblk[k][*] = x0 + sum_j ct[j][k] * U[j][*]      k = 0..hc
 *   S-GEMM+combine: S in registers, then X_k = C - iS, X_{L-k} = C + iS. */
#ifdef __AVX512F__
static void fold_pass(const cplx *src, cplx *dst,
                      int L, int hc, int hs,
                      const double *restrict ct, const double *restrict st,
                      int inner, int BC,
                      cplx *restrict Ub, cplx *restrict Vb,
                      cplx *restrict Cblk, cplx *restrict Sb_unused)
{
    (void)Sb_unused;
    const int cw = hc + 1, sw = hs + 1, n = 2 * BC;
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    __attribute__((aligned(64))) double x0buf[136]; /* BC <= 64 complex + pad */
    for (int c0 = 0; c0 < inner; c0 += BC) {
        /* fold rows j / L-j into U, V (plain loops; gcc vectorizes these) */
        for (int j = 1; j <= hs; ++j) {
            const double *a = (const double *)(src + (size_t)j * inner + c0);
            const double *b = (const double *)(src + (size_t)(L - j) * inner + c0);
            double *u = (double *)(Ub + (size_t)(j - 1) * BC);
            double *v = (double *)(Vb + (size_t)(j - 1) * BC);
            for (int d = 0; d < n; ++d) {
                u[d] = a[d] + b[d];
                v[d] = a[d] - b[d];
            }
        }
        if (hc > hs)  /* even L: lone middle row j = L/2 */
            memcpy(Ub + (size_t)(hc - 1) * BC, src + (size_t)hc * inner + c0,
                   (size_t)BC * sizeof(cplx));

        /* row 0 copied out so dst may alias src (in-place plane transform):
         * quad k=0 stores dst row 0 before later quads re-read x0 */
        memcpy(x0buf, src + c0, (size_t)BC * sizeof(cplx));
        const double *x0 = x0buf;
        const double *ub = (const double *)Ub;
        const double *vb = (const double *)Vb;
        double *cb = (double *)Cblk;

        /* C-GEMM, k in QUADS x 4-zmm d-tiles: 16 accumulators (independent FMA
         * chains), each broadcast feeds 32 doubles, ~0.5 loads per FMA. */
        for (int k = 0; k <= hc; k += 4) {
            const int k1 = (k + 1 <= hc) ? k + 1 : hc;
            const int k2 = (k + 2 <= hc) ? k + 2 : hc;
            const int k3 = (k + 3 <= hc) ? k + 3 : hc;
            for (int d = 0; d < n; d += 32) {
                const int w = (n - d < 32) ? n - d : 32;
                const __mmask8 m0 = (w >= 8)  ? (__mmask8)0xFF : (__mmask8)((1u << w) - 1);
                const __mmask8 m1 = (w >= 16) ? (__mmask8)0xFF
                                  : (w > 8  ? (__mmask8)((1u << (w - 8))  - 1) : 0);
                const __mmask8 m2 = (w >= 24) ? (__mmask8)0xFF
                                  : (w > 16 ? (__mmask8)((1u << (w - 16)) - 1) : 0);
                const __mmask8 m3 = (w >= 32) ? (__mmask8)0xFF
                                  : (w > 24 ? (__mmask8)((1u << (w - 24)) - 1) : 0);
                __m512d x00 = _mm512_maskz_loadu_pd(m0, x0 + d);
                __m512d x01 = _mm512_maskz_loadu_pd(m1, x0 + d + 8);
                __m512d x02 = _mm512_maskz_loadu_pd(m2, x0 + d + 16);
                __m512d x03 = _mm512_maskz_loadu_pd(m3, x0 + d + 24);
                __m512d a0 = x00, a1 = x01, a2 = x02, a3 = x03;
                __m512d b0 = x00, b1 = x01, b2 = x02, b3 = x03;
                __m512d e0 = x00, e1 = x01, e2 = x02, e3 = x03;
                __m512d f0 = x00, f1 = x01, f2 = x02, f3 = x03;
                for (int j = 1; j <= hc; ++j) {
                    const double *ur = ub + (size_t)(j - 1) * n + d;
                    const double *wr = ct + (size_t)(j - 1) * cw;
                    __m512d u0 = _mm512_maskz_loadu_pd(m0, ur);
                    __m512d u1 = _mm512_maskz_loadu_pd(m1, ur + 8);
                    __m512d u2 = _mm512_maskz_loadu_pd(m2, ur + 16);
                    __m512d u3 = _mm512_maskz_loadu_pd(m3, ur + 24);
                    __m512d wa = _mm512_set1_pd(wr[k]);
                    __m512d wb = _mm512_set1_pd(wr[k1]);
                    __m512d we = _mm512_set1_pd(wr[k2]);
                    __m512d wf = _mm512_set1_pd(wr[k3]);
                    a0 = _mm512_fmadd_pd(u0, wa, a0); a1 = _mm512_fmadd_pd(u1, wa, a1);
                    a2 = _mm512_fmadd_pd(u2, wa, a2); a3 = _mm512_fmadd_pd(u3, wa, a3);
                    b0 = _mm512_fmadd_pd(u0, wb, b0); b1 = _mm512_fmadd_pd(u1, wb, b1);
                    b2 = _mm512_fmadd_pd(u2, wb, b2); b3 = _mm512_fmadd_pd(u3, wb, b3);
                    e0 = _mm512_fmadd_pd(u0, we, e0); e1 = _mm512_fmadd_pd(u1, we, e1);
                    e2 = _mm512_fmadd_pd(u2, we, e2); e3 = _mm512_fmadd_pd(u3, we, e3);
                    f0 = _mm512_fmadd_pd(u0, wf, f0); f1 = _mm512_fmadd_pd(u1, wf, f1);
                    f2 = _mm512_fmadd_pd(u2, wf, f2); f3 = _mm512_fmadd_pd(u3, wf, f3);
                }
                #define GDP_CSTORE(kk, r0, r1, r2, r3) do {                              \
                    double *dd = ((kk) == 0 || 2 * (kk) == L)                            \
                               ? (double *)(dst + (size_t)(kk) * inner + c0)             \
                               : cb + (size_t)(kk) * n;                                  \
                    _mm512_mask_storeu_pd(dd + d,      m0, r0);                          \
                    _mm512_mask_storeu_pd(dd + d + 8,  m1, r1);                          \
                    _mm512_mask_storeu_pd(dd + d + 16, m2, r2);                          \
                    _mm512_mask_storeu_pd(dd + d + 24, m3, r3);                          \
                } while (0)
                GDP_CSTORE(k, a0, a1, a2, a3);
                if (k1 > k)  GDP_CSTORE(k1, b0, b1, b2, b3);
                if (k2 > k1) GDP_CSTORE(k2, e0, e1, e2, e3);
                if (k3 > k2) GDP_CSTORE(k3, f0, f1, f2, f3);
                #undef GDP_CSTORE
            }
        }

        /* S-GEMM + combine, same quad x 4-zmm tile shape */
        for (int k = 1; k <= hs; k += 4) {
            const int k1 = (k + 1 <= hs) ? k + 1 : hs;
            const int k2 = (k + 2 <= hs) ? k + 2 : hs;
            const int k3 = (k + 3 <= hs) ? k + 3 : hs;
            for (int d = 0; d < n; d += 32) {
                const int w = (n - d < 32) ? n - d : 32;
                const __mmask8 m0 = (w >= 8)  ? (__mmask8)0xFF : (__mmask8)((1u << w) - 1);
                const __mmask8 m1 = (w >= 16) ? (__mmask8)0xFF
                                  : (w > 8  ? (__mmask8)((1u << (w - 8))  - 1) : 0);
                const __mmask8 m2 = (w >= 24) ? (__mmask8)0xFF
                                  : (w > 16 ? (__mmask8)((1u << (w - 16)) - 1) : 0);
                const __mmask8 m3 = (w >= 32) ? (__mmask8)0xFF
                                  : (w > 24 ? (__mmask8)((1u << (w - 24)) - 1) : 0);
                __m512d a0 = _mm512_setzero_pd(), a1 = a0, a2 = a0, a3 = a0;
                __m512d b0 = a0, b1 = a0, b2 = a0, b3 = a0;
                __m512d e0 = a0, e1 = a0, e2 = a0, e3 = a0;
                __m512d f0 = a0, f1 = a0, f2 = a0, f3 = a0;
                for (int j = 1; j <= hs; ++j) {
                    const double *vr = vb + (size_t)(j - 1) * n + d;
                    const double *wr = st + (size_t)(j - 1) * sw;
                    __m512d v0 = _mm512_maskz_loadu_pd(m0, vr);
                    __m512d v1 = _mm512_maskz_loadu_pd(m1, vr + 8);
                    __m512d v2 = _mm512_maskz_loadu_pd(m2, vr + 16);
                    __m512d v3 = _mm512_maskz_loadu_pd(m3, vr + 24);
                    __m512d wa = _mm512_set1_pd(wr[k]);
                    __m512d wb = _mm512_set1_pd(wr[k1]);
                    __m512d we = _mm512_set1_pd(wr[k2]);
                    __m512d wf = _mm512_set1_pd(wr[k3]);
                    a0 = _mm512_fmadd_pd(v0, wa, a0); a1 = _mm512_fmadd_pd(v1, wa, a1);
                    a2 = _mm512_fmadd_pd(v2, wa, a2); a3 = _mm512_fmadd_pd(v3, wa, a3);
                    b0 = _mm512_fmadd_pd(v0, wb, b0); b1 = _mm512_fmadd_pd(v1, wb, b1);
                    b2 = _mm512_fmadd_pd(v2, wb, b2); b3 = _mm512_fmadd_pd(v3, wb, b3);
                    e0 = _mm512_fmadd_pd(v0, we, e0); e1 = _mm512_fmadd_pd(v1, we, e1);
                    e2 = _mm512_fmadd_pd(v2, we, e2); e3 = _mm512_fmadd_pd(v3, we, e3);
                    f0 = _mm512_fmadd_pd(v0, wf, f0); f1 = _mm512_fmadd_pd(v1, wf, f1);
                    f2 = _mm512_fmadd_pd(v2, wf, f2); f3 = _mm512_fmadd_pd(v3, wf, f3);
                }
                #define GDP_COMBINE(kk, s0, s1, s2, s3) do {                             \
                    const double *cr = cb + (size_t)(kk) * n;                            \
                    double *ok  = (double *)(dst + (size_t)(kk) * inner + c0);           \
                    double *olk = (double *)(dst + (size_t)(L - (kk)) * inner + c0);     \
                    __m512d c0v = _mm512_loadu_pd(cr + d);                               \
                    __m512d c1v = _mm512_loadu_pd(cr + d + 8);                           \
                    __m512d c2v = _mm512_loadu_pd(cr + d + 16);                          \
                    __m512d c3v = _mm512_loadu_pd(cr + d + 24);                          \
                    __m512d t0 = _mm512_permute_pd(s0, 0x55);                            \
                    __m512d t1 = _mm512_permute_pd(s1, 0x55);                            \
                    __m512d t2 = _mm512_permute_pd(s2, 0x55);                            \
                    __m512d t3 = _mm512_permute_pd(s3, 0x55);                            \
                    _mm512_mask_storeu_pd(ok + d,       m0, _mm512_fmadd_pd(t0, SG, c0v)); \
                    _mm512_mask_storeu_pd(ok + d + 8,   m1, _mm512_fmadd_pd(t1, SG, c1v)); \
                    _mm512_mask_storeu_pd(ok + d + 16,  m2, _mm512_fmadd_pd(t2, SG, c2v)); \
                    _mm512_mask_storeu_pd(ok + d + 24,  m3, _mm512_fmadd_pd(t3, SG, c3v)); \
                    _mm512_mask_storeu_pd(olk + d,      m0, _mm512_fnmadd_pd(t0, SG, c0v)); \
                    _mm512_mask_storeu_pd(olk + d + 8,  m1, _mm512_fnmadd_pd(t1, SG, c1v)); \
                    _mm512_mask_storeu_pd(olk + d + 16, m2, _mm512_fnmadd_pd(t2, SG, c2v)); \
                    _mm512_mask_storeu_pd(olk + d + 24, m3, _mm512_fnmadd_pd(t3, SG, c3v)); \
                } while (0)
                GDP_COMBINE(k, a0, a1, a2, a3);
                if (k1 > k)  GDP_COMBINE(k1, b0, b1, b2, b3);
                if (k2 > k1) GDP_COMBINE(k2, e0, e1, e2, e3);
                if (k3 > k2) GDP_COMBINE(k3, f0, f1, f2, f3);
                #undef GDP_COMBINE
            }
        }
    }
}

/* ---------------- fold31: the L=31 contraction, specialized ----------------
 * BC = 31 columns per block (n = 62 doubles).  U/V/Cblk rows are PADDED to
 * 64 doubles and 64-byte aligned, so every load inside the two GEMMs is a
 * full aligned vector -- no masks, no cache-line splits (the generic path's
 * 496-byte row stride made 3/4 of all GEMM loads split a line).  Padding
 * lanes 62..63 are kept zero by the maskz tail loads in the fold; the only
 * masked ops left are the 0x3F tail STORES to the (misaligned) dst rows.
 * In-place safe exactly like fold_pass. */
static __attribute__((always_inline)) inline
void fold31_core(const cplx *srcv, cplx *dstv, int inner,
                 const double *restrict ct, const double *restrict st,
                 double *restrict U, double *restrict V,
                 double *restrict CB, const int PAD)
{
    const int BW = PAD ? 32 : 31;   /* block width in complex = one row */
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    const double *src = (const double *)srcv;
    double *dst = (double *)dstv;
    const size_t rs = (size_t)inner * 2;   /* volume row stride, doubles */
    __attribute__((aligned(64))) double x0buf[64];
    for (int c0 = 0; c0 < inner; c0 += BW) {
        const double *sb = src + 2 * (size_t)c0;
        double *db = dst + 2 * (size_t)c0;

        /* fold + x0 copy; maskz tail keeps padding lanes 62,63 zero (flat);
         * padded rows load full (their pads are finite junk, masked nowhere
         * downstream because padded dst stores keep the pads) */
        for (int d = 0; d < 64; d += 8) {
            const __mmask8 mm = (PAD || d != 56) ? (__mmask8)0xFF : (__mmask8)0x3F;
            _mm512_store_pd(x0buf + d, _mm512_maskz_loadu_pd(mm, sb + d));
        }
        for (int j = 1; j <= 15; ++j) {
            const double *a = sb + (size_t)j * rs;
            const double *b = sb + (size_t)(31 - j) * rs;
            double *u = U + (size_t)(j - 1) * 64;
            double *v = V + (size_t)(j - 1) * 64;
            for (int d = 0; d < 64; d += 8) {
                const __mmask8 mm = (PAD || d != 56) ? (__mmask8)0xFF : (__mmask8)0x3F;
                __m512d av = _mm512_maskz_loadu_pd(mm, a + d);
                __m512d bv = _mm512_maskz_loadu_pd(mm, b + d);
                _mm512_store_pd(u + d, _mm512_add_pd(av, bv));
                /* V stored PRE-SWAPPED (im,re): swap commutes with the real
                 * GEMM, so the combine's vpermilpd leaves the drain path and
                 * runs here, where port 5 is idle */
                _mm512_store_pd(v + d,
                    _mm512_permute_pd(_mm512_sub_pd(av, bv), 0x55));
            }
        }

        /* C-GEMM: 4 exact k-quads (0..15), 2 d-tiles of 4 zmm, aligned loads */
        for (int k = 0; k <= 15; k += 4) {
            for (int t = 0; t < 2; ++t) {
                const int d = 32 * t;
                __m512d x00 = _mm512_load_pd(x0buf + d);
                __m512d x01 = _mm512_load_pd(x0buf + d + 8);
                __m512d x02 = _mm512_load_pd(x0buf + d + 16);
                __m512d x03 = _mm512_load_pd(x0buf + d + 24);
                __m512d a0 = x00, a1 = x01, a2 = x02, a3 = x03;
                __m512d b0 = x00, b1 = x01, b2 = x02, b3 = x03;
                __m512d e0 = x00, e1 = x01, e2 = x02, e3 = x03;
                __m512d f0 = x00, f1 = x01, f2 = x02, f3 = x03;
#ifdef GDP_UNROLL15
                #pragma GCC unroll 15
#endif
                for (int j = 1; j <= 15; ++j) {
                    const double *ur = U + (size_t)(j - 1) * 64 + d;
                    const double *wr = ct + (size_t)(j - 1) * 16;
                    __m512d u0 = _mm512_load_pd(ur);
                    __m512d u1 = _mm512_load_pd(ur + 8);
                    __m512d u2 = _mm512_load_pd(ur + 16);
                    __m512d u3 = _mm512_load_pd(ur + 24);
                    __m512d wa = _mm512_set1_pd(wr[k]);
                    __m512d wb = _mm512_set1_pd(wr[k + 1]);
                    __m512d we = _mm512_set1_pd(wr[k + 2]);
                    __m512d wf = _mm512_set1_pd(wr[k + 3]);
                    a0 = _mm512_fmadd_pd(u0, wa, a0); a1 = _mm512_fmadd_pd(u1, wa, a1);
                    a2 = _mm512_fmadd_pd(u2, wa, a2); a3 = _mm512_fmadd_pd(u3, wa, a3);
                    b0 = _mm512_fmadd_pd(u0, wb, b0); b1 = _mm512_fmadd_pd(u1, wb, b1);
                    b2 = _mm512_fmadd_pd(u2, wb, b2); b3 = _mm512_fmadd_pd(u3, wb, b3);
                    e0 = _mm512_fmadd_pd(u0, we, e0); e1 = _mm512_fmadd_pd(u1, we, e1);
                    e2 = _mm512_fmadd_pd(u2, we, e2); e3 = _mm512_fmadd_pd(u3, we, e3);
                    f0 = _mm512_fmadd_pd(u0, wf, f0); f1 = _mm512_fmadd_pd(u1, wf, f1);
                    f2 = _mm512_fmadd_pd(u2, wf, f2); f3 = _mm512_fmadd_pd(u3, wf, f3);
                }
                #define GDP31_CB(kk, r0, r1, r2, r3) do {                     \
                    double *cp = CB + (size_t)(kk) * 64 + d;                  \
                    _mm512_store_pd(cp,      r0); _mm512_store_pd(cp + 8,  r1); \
                    _mm512_store_pd(cp + 16, r2); _mm512_store_pd(cp + 24, r3); \
                } while (0)
                if (k == 0) {
                    /* X_0 = C_0 straight to dst row 0 (row 0 already copied out) */
                    if (t == 0) {
                        _mm512_storeu_pd(db,      a0); _mm512_storeu_pd(db + 8,  a1);
                        _mm512_storeu_pd(db + 16, a2); _mm512_storeu_pd(db + 24, a3);
                    } else {
                        _mm512_storeu_pd(db + 32, a0); _mm512_storeu_pd(db + 40, a1);
                        _mm512_storeu_pd(db + 48, a2);
                        if (PAD) _mm512_storeu_pd(db + 56, a3);
                        else _mm512_mask_storeu_pd(db + 56, (__mmask8)0x3F, a3);
                    }
                    GDP31_CB(1, b0, b1, b2, b3);
                    GDP31_CB(2, e0, e1, e2, e3);
                    GDP31_CB(3, f0, f1, f2, f3);
                } else {
                    GDP31_CB(k,     a0, a1, a2, a3);
                    GDP31_CB(k + 1, b0, b1, b2, b3);
                    GDP31_CB(k + 2, e0, e1, e2, e3);
                    GDP31_CB(k + 3, f0, f1, f2, f3);
                }
                #undef GDP31_CB
            }
        }

        /* S-GEMM + combine: k-quads (1..4)(5..8)(9..12)(13,14,15) as a lean triple below */
        for (int k = 1; k <= 9; k += 4) {
            const int k1 = (k + 1 <= 15) ? k + 1 : 15;
            const int k2 = (k + 2 <= 15) ? k + 2 : 15;
            const int k3 = (k + 3 <= 15) ? k + 3 : 15;
            for (int t = 0; t < 2; ++t) {
                const int d = 32 * t;
                __m512d a0 = _mm512_setzero_pd(), a1 = a0, a2 = a0, a3 = a0;
                __m512d b0 = a0, b1 = a0, b2 = a0, b3 = a0;
                __m512d e0 = a0, e1 = a0, e2 = a0, e3 = a0;
                __m512d f0 = a0, f1 = a0, f2 = a0, f3 = a0;
#ifdef GDP_UNROLL15
                #pragma GCC unroll 15
#endif
                for (int j = 1; j <= 15; ++j) {
                    const double *vr = V + (size_t)(j - 1) * 64 + d;
                    const double *wr = st + (size_t)(j - 1) * 16;
                    __m512d v0 = _mm512_load_pd(vr);
                    __m512d v1 = _mm512_load_pd(vr + 8);
                    __m512d v2 = _mm512_load_pd(vr + 16);
                    __m512d v3 = _mm512_load_pd(vr + 24);
                    __m512d wa = _mm512_set1_pd(wr[k]);
                    __m512d wb = _mm512_set1_pd(wr[k1]);
                    __m512d we = _mm512_set1_pd(wr[k2]);
                    __m512d wf = _mm512_set1_pd(wr[k3]);
                    a0 = _mm512_fmadd_pd(v0, wa, a0); a1 = _mm512_fmadd_pd(v1, wa, a1);
                    a2 = _mm512_fmadd_pd(v2, wa, a2); a3 = _mm512_fmadd_pd(v3, wa, a3);
                    b0 = _mm512_fmadd_pd(v0, wb, b0); b1 = _mm512_fmadd_pd(v1, wb, b1);
                    b2 = _mm512_fmadd_pd(v2, wb, b2); b3 = _mm512_fmadd_pd(v3, wb, b3);
                    e0 = _mm512_fmadd_pd(v0, we, e0); e1 = _mm512_fmadd_pd(v1, we, e1);
                    e2 = _mm512_fmadd_pd(v2, we, e2); e3 = _mm512_fmadd_pd(v3, we, e3);
                    f0 = _mm512_fmadd_pd(v0, wf, f0); f1 = _mm512_fmadd_pd(v1, wf, f1);
                    f2 = _mm512_fmadd_pd(v2, wf, f2); f3 = _mm512_fmadd_pd(v3, wf, f3);
                }
                #define GDP31_COMB(kk, s0, s1, s2, s3) do {                              \
                    const double *cr = CB + (size_t)(kk) * 64 + d;                       \
                    double *ok  = db + (size_t)(kk) * rs + d;                            \
                    double *olk = db + (size_t)(31 - (kk)) * rs + d;                     \
                    __m512d c0v = _mm512_load_pd(cr);                                    \
                    __m512d c1v = _mm512_load_pd(cr + 8);                                \
                    __m512d c2v = _mm512_load_pd(cr + 16);                               \
                    __m512d c3v = _mm512_load_pd(cr + 24);                               \
                    __m512d t0 = s0, t1 = s1, t2 = s2, t3 = s3;                      \
                    _mm512_storeu_pd(ok,       _mm512_fmadd_pd(t0, SG, c0v));            \
                    _mm512_storeu_pd(ok + 8,   _mm512_fmadd_pd(t1, SG, c1v));            \
                    _mm512_storeu_pd(ok + 16,  _mm512_fmadd_pd(t2, SG, c2v));            \
                    _mm512_storeu_pd(olk,      _mm512_fnmadd_pd(t0, SG, c0v));           \
                    _mm512_storeu_pd(olk + 8,  _mm512_fnmadd_pd(t1, SG, c1v));           \
                    _mm512_storeu_pd(olk + 16, _mm512_fnmadd_pd(t2, SG, c2v));           \
                    if (PAD || d == 0) {                                                 \
                        _mm512_storeu_pd(ok + 24,  _mm512_fmadd_pd(t3, SG, c3v));        \
                        _mm512_storeu_pd(olk + 24, _mm512_fnmadd_pd(t3, SG, c3v));       \
                    } else {                                                             \
                        _mm512_mask_storeu_pd(ok + 24, (__mmask8)0x3F,                   \
                                              _mm512_fmadd_pd(t3, SG, c3v));             \
                        _mm512_mask_storeu_pd(olk + 24, (__mmask8)0x3F,                  \
                                              _mm512_fnmadd_pd(t3, SG, c3v));            \
                    }                                                                    \
                } while (0)
                GDP31_COMB(k, a0, a1, a2, a3);
                GDP31_COMB(k1, b0, b1, b2, b3);
                GDP31_COMB(k2, e0, e1, e2, e3);
                GDP31_COMB(k3, f0, f1, f2, f3);
                #undef GDP31_COMB
            }
        }

        /* S tail triple k = 13,14,15: 12 accumulators, no duplicate column */
        for (int t = 0; t < 2; ++t) {
            const int d = 32 * t;
            __m512d a0 = _mm512_setzero_pd(), a1 = a0, a2 = a0, a3 = a0;
            __m512d b0 = a0, b1 = a0, b2 = a0, b3 = a0;
            __m512d e0 = a0, e1 = a0, e2 = a0, e3 = a0;
            for (int j = 1; j <= 15; ++j) {
                const double *vr = V + (size_t)(j - 1) * 64 + d;
                const double *wr = st + (size_t)(j - 1) * 16;
                __m512d v0 = _mm512_load_pd(vr);
                __m512d v1 = _mm512_load_pd(vr + 8);
                __m512d v2 = _mm512_load_pd(vr + 16);
                __m512d v3 = _mm512_load_pd(vr + 24);
                __m512d wa = _mm512_set1_pd(wr[13]);
                __m512d wb = _mm512_set1_pd(wr[14]);
                __m512d we = _mm512_set1_pd(wr[15]);
                a0 = _mm512_fmadd_pd(v0, wa, a0); a1 = _mm512_fmadd_pd(v1, wa, a1);
                a2 = _mm512_fmadd_pd(v2, wa, a2); a3 = _mm512_fmadd_pd(v3, wa, a3);
                b0 = _mm512_fmadd_pd(v0, wb, b0); b1 = _mm512_fmadd_pd(v1, wb, b1);
                b2 = _mm512_fmadd_pd(v2, wb, b2); b3 = _mm512_fmadd_pd(v3, wb, b3);
                e0 = _mm512_fmadd_pd(v0, we, e0); e1 = _mm512_fmadd_pd(v1, we, e1);
                e2 = _mm512_fmadd_pd(v2, we, e2); e3 = _mm512_fmadd_pd(v3, we, e3);
            }
            #define GDP31_COMB3(kk, s0, s1, s2, s3) do {                             \
                const double *cr = CB + (size_t)(kk) * 64 + d;                       \
                double *ok  = db + (size_t)(kk) * rs + d;                            \
                double *olk = db + (size_t)(31 - (kk)) * rs + d;                     \
                __m512d c0v = _mm512_load_pd(cr);                                    \
                __m512d c1v = _mm512_load_pd(cr + 8);                                \
                __m512d c2v = _mm512_load_pd(cr + 16);                               \
                __m512d c3v = _mm512_load_pd(cr + 24);                               \
                _mm512_storeu_pd(ok,       _mm512_fmadd_pd(s0, SG, c0v));            \
                _mm512_storeu_pd(ok + 8,   _mm512_fmadd_pd(s1, SG, c1v));            \
                _mm512_storeu_pd(ok + 16,  _mm512_fmadd_pd(s2, SG, c2v));            \
                _mm512_storeu_pd(olk,      _mm512_fnmadd_pd(s0, SG, c0v));           \
                _mm512_storeu_pd(olk + 8,  _mm512_fnmadd_pd(s1, SG, c1v));           \
                _mm512_storeu_pd(olk + 16, _mm512_fnmadd_pd(s2, SG, c2v));           \
                if (PAD || d == 0) {                                                 \
                    _mm512_storeu_pd(ok + 24,  _mm512_fmadd_pd(s3, SG, c3v));        \
                    _mm512_storeu_pd(olk + 24, _mm512_fnmadd_pd(s3, SG, c3v));       \
                } else {                                                             \
                    _mm512_mask_storeu_pd(ok + 24, (__mmask8)0x3F,                   \
                                          _mm512_fmadd_pd(s3, SG, c3v));             \
                    _mm512_mask_storeu_pd(olk + 24, (__mmask8)0x3F,                  \
                                          _mm512_fnmadd_pd(s3, SG, c3v));            \
                }                                                                    \
            } while (0)
            GDP31_COMB3(13, a0, a1, a2, a3);
            GDP31_COMB3(14, b0, b1, b2, b3);
            GDP31_COMB3(15, e0, e1, e2, e3);
            #undef GDP31_COMB3
        }
    }
}
#else
static void fold_pass(const cplx *src, cplx *dst,
                      int L, int hc, int hs,
                      const double *restrict ct, const double *restrict st,
                      int inner, int BC,
                      cplx *restrict Ub, cplx *restrict Vb,
                      cplx *restrict Cb, cplx *restrict Sb)
{
    const int cw = hc + 1, sw = hs + 1;
    cplx x0c[64];   /* row 0 copied out so dst may alias src */
    for (int c0 = 0; c0 < inner; c0 += BC) {
        for (int j = 1; j <= hs; ++j) {
            const double *a = (const double *)(src + (size_t)j * inner + c0);
            const double *b = (const double *)(src + (size_t)(L - j) * inner + c0);
            double *u = (double *)(Ub + (size_t)(j - 1) * BC);
            double *v = (double *)(Vb + (size_t)(j - 1) * BC);
            for (int d = 0; d < 2 * BC; ++d) {
                u[d] = a[d] + b[d];
                v[d] = a[d] - b[d];
            }
        }
        if (hc > hs)
            memcpy(Ub + (size_t)(hc - 1) * BC, src + (size_t)hc * inner + c0,
                   (size_t)BC * sizeof(cplx));

        memcpy(x0c, src + c0, (size_t)BC * sizeof(cplx));
        const double *x0 = (const double *)x0c;
        for (int k = 0; k <= hc; ++k) {
            double *C = (double *)Cb;
            for (int d = 0; d < 2 * BC; ++d) C[d] = x0[d];
            for (int j = 1; j <= hc; ++j) {
                double w = ct[(size_t)(j - 1) * cw + k];
                const double *u = (const double *)(Ub + (size_t)(j - 1) * BC);
                for (int d = 0; d < 2 * BC; ++d) C[d] += w * u[d];
            }
            if (k == 0 || 2 * k == L) {
                memcpy(dst + (size_t)k * inner + c0, Cb, (size_t)BC * sizeof(cplx));
                continue;
            }
            double *S = (double *)Sb;
            for (int d = 0; d < 2 * BC; ++d) S[d] = 0.0;
            for (int j = 1; j <= hs; ++j) {
                double w = st[(size_t)(j - 1) * sw + k];
                const double *v = (const double *)(Vb + (size_t)(j - 1) * BC);
                for (int d = 0; d < 2 * BC; ++d) S[d] += w * v[d];
            }
            double *ok  = (double *)(dst + (size_t)k * inner + c0);
            double *olk = (double *)(dst + (size_t)(L - k) * inner + c0);
            for (int d = 0; d < 2 * BC; d += 2) {
                double cr = C[d], ci = C[d + 1], sr = S[d], si = S[d + 1];
                ok[d]      = cr + si;
                ok[d + 1]  = ci - sr;
                olk[d]     = cr - si;
                olk[d + 1] = ci + sr;
            }
        }
    }
}
#endif


#ifdef __AVX512F__
static void fold31(const cplx *srcv, cplx *dstv, int inner,
                   const double *restrict ct, const double *restrict st,
                   double *restrict U, double *restrict V, double *restrict CB)
{ fold31_core(srcv, dstv, inner, ct, st, U, V, CB, 0); }

static void fold31_p(const cplx *srcv, cplx *dstv, int inner,
                     const double *restrict ct, const double *restrict st,
                     double *restrict U, double *restrict V, double *restrict CB)
{ fold31_core(srcv, dstv, inner, ct, st, U, V, CB, 1); }

/* fold31zx: the x-axis contraction with the z-pass FUSED IN (gen_r2).  The
 * x-pass's column block y0 spans exactly the 31 z-pencils (j, y0, 0..30), so
 * each block z-transforms its rows pairwise into two aligned padded STACK
 * rows and folds U/V straight from there.  The separate z volume sweep --
 * 952 KB of L2 write+read per chain step, every row store a misaligned
 * line-split -- disappears; z outputs only ever feed x0buf/U/V.
 * In-place safe: all volume reads of a block precede its GEMM stores. */
static __attribute__((always_inline)) inline
void zpair31_p(const double *xA, const double *xB, double *yA, double *yB,
               const double *restrict ctd, const double *restrict std,
               const int fulltail);
static __attribute__((always_inline)) inline
void zsingle31_p(const double *x, const double *crow, double *y,
                 const double *restrict ctd, const double *restrict std,
                 const int PAD, const int fulltail);
static __attribute__((always_inline)) inline
void zpair31_uv(const double *xA, const double *xB,
                const double *cA, const double *cB,
                double *restrict u, double *restrict v,
                const double *restrict ctd, const double *restrict std,
                const int PAD);

/* cvv != NULL selects MAP-ON-LOAD (gen_r4): every state row is mapped
 * (w = row + c_row; w/(1+|w|)) inside the z-phase load, using the matching
 * row of the padded c volume.  Only legal with PAD (the chain layout). */
static __attribute__((always_inline)) inline
void fold31zx_core(const cplx *srcv, cplx *dstv, const cplx *cvv,
                   const double *restrict ct, const double *restrict st,
                   const double *restrict ctd, const double *restrict std,
                   double *restrict U, double *restrict V,
                   double *restrict CB, const int PAD)
{
    const int BW = PAD ? 32 : 31;
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    const double *src = (const double *)srcv;
    double *dst = (double *)dstv;
    /* x row stride = plane pitch (padded: GDP_PP, not 31 rows) in doubles */
    const size_t rs = 2 * (size_t)(PAD ? GDP_PP : 961);
    __attribute__((aligned(64))) double x0buf[64];
    /* the fused z drain fills U/V pad lanes 62,63 with finite junk; every
     * downstream consumer masks them out at the final dst store */
    for (int bi = 0; bi < 31; ++bi) {
        /* permuted block order (blocks = independent column sets, any order
         * is legal): consecutive-block column advance GDP_BG*BW*16 B breaks
         * the store->load low-12-bit collisions the sequential order has */
        const int c0 = ((bi * GDP_BG) % 31) * BW;
        const double *sb = src + 2 * (size_t)c0;
        const double *cb2 = cvv ? (const double *)cvv + 2 * (size_t)c0 : NULL;
        double *db = dst + 2 * (size_t)c0;

        for (int j = 1; j <= 15; ++j) {
#ifdef GDP_PREFETCH
            /* next (permuted-order) block's mirrored row pair, 2 lines each;
             * L1 next-line prefetch covers the rest of the 512B rows */
            _mm_prefetch((const char *)(sb + 2 * GDP_BG * BW + (size_t)j * rs), _MM_HINT_T0);
            _mm_prefetch((const char *)(sb + 2 * GDP_BG * BW + (size_t)j * rs) + 256, _MM_HINT_T0);
            _mm_prefetch((const char *)(sb + 2 * GDP_BG * BW + (size_t)(31 - j) * rs), _MM_HINT_T0);
            _mm_prefetch((const char *)(sb + 2 * GDP_BG * BW + (size_t)(31 - j) * rs) + 256, _MM_HINT_T0);
#endif
            zpair31_uv(sb + (size_t)j * rs, sb + (size_t)(31 - j) * rs,
                       cb2 ? cb2 + (size_t)j * rs : NULL,
                       cb2 ? cb2 + (size_t)(31 - j) * rs : NULL,
                       U + (size_t)(j - 1) * 64, V + (size_t)(j - 1) * 64,
                       ctd, std, PAD);
        }
        zsingle31_p(sb, cb2, x0buf, ctd, std, PAD, 1);

        /* C-GEMM: 4 exact k-quads (0..15), 2 d-tiles of 4 zmm, aligned loads */
        for (int k = 0; k <= 15; k += 4) {
            for (int t = 0; t < 2; ++t) {
                const int d = 32 * t;
                __m512d x00 = _mm512_load_pd(x0buf + d);
                __m512d x01 = _mm512_load_pd(x0buf + d + 8);
                __m512d x02 = _mm512_load_pd(x0buf + d + 16);
                __m512d x03 = _mm512_load_pd(x0buf + d + 24);
                __m512d a0 = x00, a1 = x01, a2 = x02, a3 = x03;
                __m512d b0 = x00, b1 = x01, b2 = x02, b3 = x03;
                __m512d e0 = x00, e1 = x01, e2 = x02, e3 = x03;
                __m512d f0 = x00, f1 = x01, f2 = x02, f3 = x03;
                for (int j = 1; j <= 15; ++j) {
                    const double *ur = U + (size_t)(j - 1) * 64 + d;
                    const double *wr = ct + (size_t)(j - 1) * 16;
                    __m512d u0 = _mm512_load_pd(ur);
                    __m512d u1 = _mm512_load_pd(ur + 8);
                    __m512d u2 = _mm512_load_pd(ur + 16);
                    __m512d u3 = _mm512_load_pd(ur + 24);
                    __m512d wa = _mm512_set1_pd(wr[k]);
                    __m512d wb = _mm512_set1_pd(wr[k + 1]);
                    __m512d we = _mm512_set1_pd(wr[k + 2]);
                    __m512d wf = _mm512_set1_pd(wr[k + 3]);
                    a0 = _mm512_fmadd_pd(u0, wa, a0); a1 = _mm512_fmadd_pd(u1, wa, a1);
                    a2 = _mm512_fmadd_pd(u2, wa, a2); a3 = _mm512_fmadd_pd(u3, wa, a3);
                    b0 = _mm512_fmadd_pd(u0, wb, b0); b1 = _mm512_fmadd_pd(u1, wb, b1);
                    b2 = _mm512_fmadd_pd(u2, wb, b2); b3 = _mm512_fmadd_pd(u3, wb, b3);
                    e0 = _mm512_fmadd_pd(u0, we, e0); e1 = _mm512_fmadd_pd(u1, we, e1);
                    e2 = _mm512_fmadd_pd(u2, we, e2); e3 = _mm512_fmadd_pd(u3, we, e3);
                    f0 = _mm512_fmadd_pd(u0, wf, f0); f1 = _mm512_fmadd_pd(u1, wf, f1);
                    f2 = _mm512_fmadd_pd(u2, wf, f2); f3 = _mm512_fmadd_pd(u3, wf, f3);
                }
                #define GDPZX_CB(kk, r0, r1, r2, r3) do {                     \
                    double *cp = CB + (size_t)(kk) * 64 + d;                  \
                    _mm512_store_pd(cp,      r0); _mm512_store_pd(cp + 8,  r1); \
                    _mm512_store_pd(cp + 16, r2); _mm512_store_pd(cp + 24, r3); \
                } while (0)
                if (k == 0) {
                    if (t == 0) {
                        _mm512_storeu_pd(db,      a0); _mm512_storeu_pd(db + 8,  a1);
                        _mm512_storeu_pd(db + 16, a2); _mm512_storeu_pd(db + 24, a3);
                    } else {
                        _mm512_storeu_pd(db + 32, a0); _mm512_storeu_pd(db + 40, a1);
                        _mm512_storeu_pd(db + 48, a2);
                        if (PAD) _mm512_storeu_pd(db + 56, a3);
                        else _mm512_mask_storeu_pd(db + 56, (__mmask8)0x3F, a3);
                    }
                    GDPZX_CB(1, b0, b1, b2, b3);
                    GDPZX_CB(2, e0, e1, e2, e3);
                    GDPZX_CB(3, f0, f1, f2, f3);
                } else {
                    GDPZX_CB(k,     a0, a1, a2, a3);
                    GDPZX_CB(k + 1, b0, b1, b2, b3);
                    GDPZX_CB(k + 2, e0, e1, e2, e3);
                    GDPZX_CB(k + 3, f0, f1, f2, f3);
                }
                #undef GDPZX_CB
            }
        }

        /* S-GEMM + combine: V is pre-swapped, no permutes in the drain */
        for (int k = 1; k <= 9; k += 4) {
            const int k1 = (k + 1 <= 15) ? k + 1 : 15;
            const int k2 = (k + 2 <= 15) ? k + 2 : 15;
            const int k3 = (k + 3 <= 15) ? k + 3 : 15;
            for (int t = 0; t < 2; ++t) {
                const int d = 32 * t;
                __m512d a0 = _mm512_setzero_pd(), a1 = a0, a2 = a0, a3 = a0;
                __m512d b0 = a0, b1 = a0, b2 = a0, b3 = a0;
                __m512d e0 = a0, e1 = a0, e2 = a0, e3 = a0;
                __m512d f0 = a0, f1 = a0, f2 = a0, f3 = a0;
                for (int j = 1; j <= 15; ++j) {
                    const double *vr = V + (size_t)(j - 1) * 64 + d;
                    const double *wr = st + (size_t)(j - 1) * 16;
                    __m512d v0 = _mm512_load_pd(vr);
                    __m512d v1 = _mm512_load_pd(vr + 8);
                    __m512d v2 = _mm512_load_pd(vr + 16);
                    __m512d v3 = _mm512_load_pd(vr + 24);
                    __m512d wa = _mm512_set1_pd(wr[k]);
                    __m512d wb = _mm512_set1_pd(wr[k1]);
                    __m512d we = _mm512_set1_pd(wr[k2]);
                    __m512d wf = _mm512_set1_pd(wr[k3]);
                    a0 = _mm512_fmadd_pd(v0, wa, a0); a1 = _mm512_fmadd_pd(v1, wa, a1);
                    a2 = _mm512_fmadd_pd(v2, wa, a2); a3 = _mm512_fmadd_pd(v3, wa, a3);
                    b0 = _mm512_fmadd_pd(v0, wb, b0); b1 = _mm512_fmadd_pd(v1, wb, b1);
                    b2 = _mm512_fmadd_pd(v2, wb, b2); b3 = _mm512_fmadd_pd(v3, wb, b3);
                    e0 = _mm512_fmadd_pd(v0, we, e0); e1 = _mm512_fmadd_pd(v1, we, e1);
                    e2 = _mm512_fmadd_pd(v2, we, e2); e3 = _mm512_fmadd_pd(v3, we, e3);
                    f0 = _mm512_fmadd_pd(v0, wf, f0); f1 = _mm512_fmadd_pd(v1, wf, f1);
                    f2 = _mm512_fmadd_pd(v2, wf, f2); f3 = _mm512_fmadd_pd(v3, wf, f3);
                }
                #define GDPZX_COMB(kk, s0, s1, s2, s3) do {                              \
                    const double *cr = CB + (size_t)(kk) * 64 + d;                       \
                    double *ok  = db + (size_t)(kk) * rs + d;                            \
                    double *olk = db + (size_t)(31 - (kk)) * rs + d;                     \
                    __m512d c0v = _mm512_load_pd(cr);                                    \
                    __m512d c1v = _mm512_load_pd(cr + 8);                                \
                    __m512d c2v = _mm512_load_pd(cr + 16);                               \
                    __m512d c3v = _mm512_load_pd(cr + 24);                               \
                    _mm512_storeu_pd(ok,       _mm512_fmadd_pd(s0, SG, c0v));            \
                    _mm512_storeu_pd(ok + 8,   _mm512_fmadd_pd(s1, SG, c1v));            \
                    _mm512_storeu_pd(ok + 16,  _mm512_fmadd_pd(s2, SG, c2v));            \
                    _mm512_storeu_pd(olk,      _mm512_fnmadd_pd(s0, SG, c0v));           \
                    _mm512_storeu_pd(olk + 8,  _mm512_fnmadd_pd(s1, SG, c1v));           \
                    _mm512_storeu_pd(olk + 16, _mm512_fnmadd_pd(s2, SG, c2v));           \
                    if (PAD || d == 0) {                                                 \
                        _mm512_storeu_pd(ok + 24,  _mm512_fmadd_pd(s3, SG, c3v));        \
                        _mm512_storeu_pd(olk + 24, _mm512_fnmadd_pd(s3, SG, c3v));       \
                    } else {                                                             \
                        _mm512_mask_storeu_pd(ok + 24, (__mmask8)0x3F,                   \
                                              _mm512_fmadd_pd(s3, SG, c3v));             \
                        _mm512_mask_storeu_pd(olk + 24, (__mmask8)0x3F,                  \
                                              _mm512_fnmadd_pd(s3, SG, c3v));            \
                    }                                                                    \
                } while (0)
                GDPZX_COMB(k, a0, a1, a2, a3);
                GDPZX_COMB(k1, b0, b1, b2, b3);
                GDPZX_COMB(k2, e0, e1, e2, e3);
                GDPZX_COMB(k3, f0, f1, f2, f3);
                #undef GDPZX_COMB
            }
        }

        /* S tail triple k = 13,14,15: 12 accumulators, no duplicate column */
        for (int t = 0; t < 2; ++t) {
            const int d = 32 * t;
            __m512d a0 = _mm512_setzero_pd(), a1 = a0, a2 = a0, a3 = a0;
            __m512d b0 = a0, b1 = a0, b2 = a0, b3 = a0;
            __m512d e0 = a0, e1 = a0, e2 = a0, e3 = a0;
            for (int j = 1; j <= 15; ++j) {
                const double *vr = V + (size_t)(j - 1) * 64 + d;
                const double *wr = st + (size_t)(j - 1) * 16;
                __m512d v0 = _mm512_load_pd(vr);
                __m512d v1 = _mm512_load_pd(vr + 8);
                __m512d v2 = _mm512_load_pd(vr + 16);
                __m512d v3 = _mm512_load_pd(vr + 24);
                __m512d wa = _mm512_set1_pd(wr[13]);
                __m512d wb = _mm512_set1_pd(wr[14]);
                __m512d we = _mm512_set1_pd(wr[15]);
                a0 = _mm512_fmadd_pd(v0, wa, a0); a1 = _mm512_fmadd_pd(v1, wa, a1);
                a2 = _mm512_fmadd_pd(v2, wa, a2); a3 = _mm512_fmadd_pd(v3, wa, a3);
                b0 = _mm512_fmadd_pd(v0, wb, b0); b1 = _mm512_fmadd_pd(v1, wb, b1);
                b2 = _mm512_fmadd_pd(v2, wb, b2); b3 = _mm512_fmadd_pd(v3, wb, b3);
                e0 = _mm512_fmadd_pd(v0, we, e0); e1 = _mm512_fmadd_pd(v1, we, e1);
                e2 = _mm512_fmadd_pd(v2, we, e2); e3 = _mm512_fmadd_pd(v3, we, e3);
            }
            #define GDPZX_COMB3(kk, s0, s1, s2, s3) do {                             \
                const double *cr = CB + (size_t)(kk) * 64 + d;                       \
                double *ok  = db + (size_t)(kk) * rs + d;                            \
                double *olk = db + (size_t)(31 - (kk)) * rs + d;                     \
                __m512d c0v = _mm512_load_pd(cr);                                    \
                __m512d c1v = _mm512_load_pd(cr + 8);                                \
                __m512d c2v = _mm512_load_pd(cr + 16);                               \
                __m512d c3v = _mm512_load_pd(cr + 24);                               \
                _mm512_storeu_pd(ok,       _mm512_fmadd_pd(s0, SG, c0v));            \
                _mm512_storeu_pd(ok + 8,   _mm512_fmadd_pd(s1, SG, c1v));            \
                _mm512_storeu_pd(ok + 16,  _mm512_fmadd_pd(s2, SG, c2v));            \
                _mm512_storeu_pd(olk,      _mm512_fnmadd_pd(s0, SG, c0v));           \
                _mm512_storeu_pd(olk + 8,  _mm512_fnmadd_pd(s1, SG, c1v));           \
                _mm512_storeu_pd(olk + 16, _mm512_fnmadd_pd(s2, SG, c2v));           \
                if (PAD || d == 0) {                                                 \
                    _mm512_storeu_pd(ok + 24,  _mm512_fmadd_pd(s3, SG, c3v));        \
                    _mm512_storeu_pd(olk + 24, _mm512_fnmadd_pd(s3, SG, c3v));       \
                } else {                                                             \
                    _mm512_mask_storeu_pd(ok + 24, (__mmask8)0x3F,                   \
                                          _mm512_fmadd_pd(s3, SG, c3v));             \
                    _mm512_mask_storeu_pd(olk + 24, (__mmask8)0x3F,                  \
                                          _mm512_fnmadd_pd(s3, SG, c3v));            \
                }                                                                    \
            } while (0)
            GDPZX_COMB3(13, a0, a1, a2, a3);
            GDPZX_COMB3(14, b0, b1, b2, b3);
            GDPZX_COMB3(15, e0, e1, e2, e3);
            #undef GDPZX_COMB3
        }
    }
}

static void fold31zx(const cplx *srcv, cplx *dstv,
                     const double *restrict ct, const double *restrict st,
                     const double *restrict ctd, const double *restrict std,
                     double *restrict U, double *restrict V, double *restrict CB)
{ fold31zx_core(srcv, dstv, NULL, ct, st, ctd, std, U, V, CB, 0); }

static void fold31zx_p(const cplx *srcv, cplx *dstv,
                       const double *restrict ct, const double *restrict st,
                       const double *restrict ctd, const double *restrict std,
                       double *restrict U, double *restrict V, double *restrict CB)
{ fold31zx_core(srcv, dstv, NULL, ct, st, ctd, std, U, V, CB, 1); }

/* padded chain z+x pass with the previous step's map applied on load */
static void fold31zx_pm(const cplx *srcv, cplx *dstv, const cplx *cv,
                        const double *restrict ct, const double *restrict st,
                        const double *restrict ctd, const double *restrict std,
                        double *restrict U, double *restrict V, double *restrict CB)
{ fold31zx_core(srcv, dstv, cv, ct, st, ctd, std, U, V, CB, 1); }
#endif

/* ---------------- z-axis pass: contiguous rows ---------------- */

/* generic scalar row transform (used for the small composite sizes; also the
 * non-AVX512 fallback for L=31).  In-place safe: all reads of a row precede
 * its stores. */
static void zpass_generic(const cplx *src, cplx *dst,
                          int L, int hc, int hs,
                          const double *restrict ct, const double *restrict st,
                          size_t nrows)
{
    const int cw = hc + 1, sw = hs + 1;
    cplx u[64], v[64], C[64], S[64];
    for (size_t r = 0; r < nrows; ++r) {
        const cplx *x = src + r * L;
        cplx *y = dst + r * L;
        for (int j = 1; j <= hs; ++j) {
            u[j] = x[j] + x[L - j];
            v[j] = x[j] - x[L - j];
        }
        if (hc > hs) u[hc] = x[hc];
        for (int k = 0; k <= hc; ++k) C[k] = x[0];
        for (int k = 1; k <= hs; ++k) S[k] = 0.0;
        for (int j = 1; j <= hc; ++j) {
            cplx uj = u[j];
            const double *cr = ct + (size_t)(j - 1) * cw;
            for (int k = 1; k <= hc; ++k) C[k] += cr[k] * uj;
            C[0] += uj; /* cos(0)=1 */
        }
        for (int j = 1; j <= hs; ++j) {
            cplx vj = v[j];
            const double *sr = st + (size_t)(j - 1) * sw;
            for (int k = 1; k <= hs; ++k) S[k] += sr[k] * vj;
        }
        y[0] = C[0];
        if (2 * hc == L) y[hc] = C[hc];
        for (int k = 1; k <= hs; ++k) {
            double srr = creal(S[k]), sii = cimag(S[k]);
            cplx mis = sii - srr * I;        /* -i * S */
            y[k]     = C[k] + mis;
            y[L - k] = C[k] - mis;
        }
    }
}

#ifdef __AVX512F__
/* ---------------- generic vector z-pass (gen_r3), any 5 <= L <= 31 -------
 * The zpair31 shape with the k dimension in KQ = ceil((hc+1)/4) zmm and the
 * layout data (store masks, reversed-half offsets) from the plan.  Rows in
 * PAIRS to share the table loads (2*KQ loads + 4 broadcasts feed 4*KQ FMAs
 * per j).  Even L works out of the box: the lone middle row j = L/2 is one
 * extra C-only iteration, and X_{L/2} = C_{L/2} falls out because the sin
 * table's k = L/2 slot is exactly zero.  In-place safe: each row is fully
 * read into stack buffers / accumulators before any store.  Odd row counts
 * reuse the pair kernel with both rows aliased to the last row (all reads
 * precede all stores, so the duplicate store is idempotent). */
#define GDP_ZGEN(KQ)                                                          \
static void zrow_pair_k##KQ(const double *xA, const double *xB,               \
                            double *yA, double *yB, const fft3d_plan *p)      \
{                                                                             \
    const int L = p->L, hs = p->hs, hu = p->hc;                               \
    const double *restrict ctz = p->ctz, *restrict stz = p->stz;              \
    const __m512d SG = _mm512_setr_pd(1.0,-1.0,1.0,-1.0,1.0,-1.0,1.0,-1.0);   \
    __attribute__((aligned(64))) double ua[32], va[32], ub[32], vb[32];       \
    for (int j = 1; j <= hs; ++j) {                                           \
        __m128d a = _mm_loadu_pd(xA + 2*j), b = _mm_loadu_pd(xA + 2*(L-j));   \
        _mm_store_pd(ua + 2*j, _mm_add_pd(a, b));                             \
        _mm_store_pd(va + 2*j, _mm_sub_pd(a, b));                             \
        __m128d c = _mm_loadu_pd(xB + 2*j), e = _mm_loadu_pd(xB + 2*(L-j));   \
        _mm_store_pd(ub + 2*j, _mm_add_pd(c, e));                             \
        _mm_store_pd(vb + 2*j, _mm_sub_pd(c, e));                             \
    }                                                                         \
    if (hu > hs) {   /* even L: lone middle row feeds the +-1 cos row */      \
        ua[2*hu] = xA[2*hu]; ua[2*hu+1] = xA[2*hu+1];                         \
        ub[2*hu] = xB[2*hu]; ub[2*hu+1] = xB[2*hu+1];                         \
    }                                                                         \
    __m512d CA[KQ], CB[KQ], SA[KQ], SB[KQ];                                   \
    {                                                                         \
        __m512d xa0 = _mm512_broadcast_f64x2(_mm_loadu_pd(xA));               \
        __m512d xb0 = _mm512_broadcast_f64x2(_mm_loadu_pd(xB));               \
        for (int q = 0; q < KQ; ++q) {                                        \
            CA[q] = xa0; CB[q] = xb0;                                         \
            SA[q] = _mm512_setzero_pd(); SB[q] = SA[q];                       \
        }                                                                     \
    }                                                                         \
    for (int j = 1; j <= hs; ++j) {                                           \
        const double *cr = ctz + (size_t)(j - 1) * (8 * KQ);                  \
        const double *sr = stz + (size_t)(j - 1) * (8 * KQ);                  \
        __m512d uA = _mm512_broadcast_f64x2(_mm_load_pd(ua + 2*j));           \
        __m512d vA = _mm512_broadcast_f64x2(_mm_load_pd(va + 2*j));           \
        __m512d uB = _mm512_broadcast_f64x2(_mm_load_pd(ub + 2*j));           \
        __m512d vB = _mm512_broadcast_f64x2(_mm_load_pd(vb + 2*j));           \
        for (int q = 0; q < KQ; ++q) {                                        \
            __m512d cw = _mm512_load_pd(cr + 8*q);                            \
            __m512d sw = _mm512_load_pd(sr + 8*q);                            \
            CA[q] = _mm512_fmadd_pd(cw, uA, CA[q]);                           \
            SA[q] = _mm512_fmadd_pd(sw, vA, SA[q]);                           \
            CB[q] = _mm512_fmadd_pd(cw, uB, CB[q]);                           \
            SB[q] = _mm512_fmadd_pd(sw, vB, SB[q]);                           \
        }                                                                     \
    }                                                                         \
    if (hu > hs) {                                                            \
        const double *cr = ctz + (size_t)(hu - 1) * (8 * KQ);                 \
        __m512d uA = _mm512_broadcast_f64x2(_mm_load_pd(ua + 2*hu));          \
        __m512d uB = _mm512_broadcast_f64x2(_mm_load_pd(ub + 2*hu));          \
        for (int q = 0; q < KQ; ++q) {                                        \
            __m512d cw = _mm512_load_pd(cr + 8*q);                            \
            CA[q] = _mm512_fmadd_pd(cw, uA, CA[q]);                           \
            CB[q] = _mm512_fmadd_pd(cw, uB, CB[q]);                           \
        }                                                                     \
    }                                                                         \
    for (int q = 0; q < KQ; ++q) {                                            \
        __m512d TA = _mm512_permute_pd(SA[q], 0x55);                          \
        __m512d TB = _mm512_permute_pd(SB[q], 0x55);                          \
        _mm512_mask_storeu_pd(yA + 8*q, (__mmask8)p->zfm[q],                  \
                              _mm512_fmadd_pd(TA, SG, CA[q]));                \
        _mm512_mask_storeu_pd(yB + 8*q, (__mmask8)p->zfm[q],                  \
                              _mm512_fmadd_pd(TB, SG, CB[q]));                \
        if (p->zsm[q]) {                                                      \
            __m512d hA = _mm512_fnmadd_pd(TA, SG, CA[q]);                     \
            __m512d hB = _mm512_fnmadd_pd(TB, SG, CB[q]);                     \
            _mm512_mask_storeu_pd(yA + p->zso2[q], (__mmask8)p->zsm[q],       \
                                  _mm512_shuffle_f64x2(hA, hA, 0x1B));        \
            _mm512_mask_storeu_pd(yB + p->zso2[q], (__mmask8)p->zsm[q],       \
                                  _mm512_shuffle_f64x2(hB, hB, 0x1B));        \
        }                                                                     \
    }                                                                         \
}
GDP_ZGEN(1)
GDP_ZGEN(2)
GDP_ZGEN(3)
GDP_ZGEN(4)
#undef GDP_ZGEN

static void zpass_vec(const fft3d_plan *p, const cplx *src, cplx *dst,
                      size_t nrows, size_t stride)
{
    void (*krn)(const double *, const double *, double *, double *,
                const fft3d_plan *) =
        p->kq == 1 ? zrow_pair_k1 : p->kq == 2 ? zrow_pair_k2 :
        p->kq == 3 ? zrow_pair_k3 : zrow_pair_k4;
    size_t r = 0;
    for (; r + 2 <= nrows; r += 2)
        krn((const double *)(src + r * stride),
            (const double *)(src + (r + 1) * stride),
            (double *)(dst + r * stride),
            (double *)(dst + (r + 1) * stride), p);
    if (r < nrows)
        krn((const double *)(src + r * stride), (const double *)(src + r * stride),
            (double *)(dst + r * stride), (double *)(dst + r * stride), p);
}

/* L=31 vector row transform: k dimension lives in 4 zmm (16 complex outputs of
 * each half-spectrum), u_j/v_j broadcast as 128-bit pairs, tables pre-duplicated.
 * Rows processed in PAIRS so the 8 table loads per j are shared between two
 * rows' 16 FMAs (single-row form: 10 loads / 8 FMA, pair form: 12 / 16).
 * In-place safe: each row is fully read into stack buffers before any store. */
static __attribute__((always_inline)) inline
void zpair31_p(const double *xA, const double *xB,
               double *yA, double *yB,
               const double *restrict ctd, const double *restrict std,
               const int fulltail)
{
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);

    __attribute__((aligned(64))) double ua[32], va[32], ub2[32], vb2[32];
    for (int j = 1; j <= 15; ++j) {
        __m128d a = _mm_loadu_pd(xA + 2 * j), b = _mm_loadu_pd(xA + 2 * (31 - j));
        _mm_store_pd(ua + 2 * j, _mm_add_pd(a, b));
        _mm_store_pd(va + 2 * j, _mm_sub_pd(a, b));
        __m128d c = _mm_loadu_pd(xB + 2 * j), e = _mm_loadu_pd(xB + 2 * (31 - j));
        _mm_store_pd(ub2 + 2 * j, _mm_add_pd(c, e));
        _mm_store_pd(vb2 + 2 * j, _mm_sub_pd(c, e));
    }
    __m512d xa0 = _mm512_broadcast_f64x2(_mm_loadu_pd(xA));
    __m512d xb0 = _mm512_broadcast_f64x2(_mm_loadu_pd(xB));
    __m512d CA0 = xa0, CA1 = xa0, CA2 = xa0, CA3 = xa0;
    __m512d CB0 = xb0, CB1 = xb0, CB2 = xb0, CB3 = xb0;
    __m512d SA0 = _mm512_setzero_pd(), SA1 = SA0, SA2 = SA0, SA3 = SA0;
    __m512d SB0 = SA0, SB1 = SA0, SB2 = SA0, SB3 = SA0;
    for (int j = 1; j <= 15; ++j) {
        const double *cr = ctd + (size_t)(j - 1) * 32;
        const double *sr = std + (size_t)(j - 1) * 32;
        __m512d c0 = _mm512_load_pd(cr + 0),  c1 = _mm512_load_pd(cr + 8);
        __m512d c2 = _mm512_load_pd(cr + 16), c3 = _mm512_load_pd(cr + 24);
        __m512d s0 = _mm512_load_pd(sr + 0),  s1 = _mm512_load_pd(sr + 8);
        __m512d s2 = _mm512_load_pd(sr + 16), s3 = _mm512_load_pd(sr + 24);
        __m512d uA = _mm512_broadcast_f64x2(_mm_load_pd(ua + 2 * j));
        __m512d vA = _mm512_broadcast_f64x2(_mm_load_pd(va + 2 * j));
        __m512d uB = _mm512_broadcast_f64x2(_mm_load_pd(ub2 + 2 * j));
        __m512d vB = _mm512_broadcast_f64x2(_mm_load_pd(vb2 + 2 * j));
        CA0 = _mm512_fmadd_pd(c0, uA, CA0); CA1 = _mm512_fmadd_pd(c1, uA, CA1);
        CA2 = _mm512_fmadd_pd(c2, uA, CA2); CA3 = _mm512_fmadd_pd(c3, uA, CA3);
        SA0 = _mm512_fmadd_pd(s0, vA, SA0); SA1 = _mm512_fmadd_pd(s1, vA, SA1);
        SA2 = _mm512_fmadd_pd(s2, vA, SA2); SA3 = _mm512_fmadd_pd(s3, vA, SA3);
        CB0 = _mm512_fmadd_pd(c0, uB, CB0); CB1 = _mm512_fmadd_pd(c1, uB, CB1);
        CB2 = _mm512_fmadd_pd(c2, uB, CB2); CB3 = _mm512_fmadd_pd(c3, uB, CB3);
        SB0 = _mm512_fmadd_pd(s0, vB, SB0); SB1 = _mm512_fmadd_pd(s1, vB, SB1);
        SB2 = _mm512_fmadd_pd(s2, vB, SB2); SB3 = _mm512_fmadd_pd(s3, vB, SB3);
    }
    #define GDP_ZSTORE(y, C0, C1, C2, C3, S0, S1, S2, S3) do {                     \
        __m512d T0 = _mm512_permute_pd(S0, 0x55);                                  \
        __m512d T1 = _mm512_permute_pd(S1, 0x55);                                  \
        __m512d T2 = _mm512_permute_pd(S2, 0x55);                                  \
        __m512d T3 = _mm512_permute_pd(S3, 0x55);                                  \
        _mm512_storeu_pd((y) + 0,  _mm512_fmadd_pd(T0, SG, C0));                   \
        _mm512_storeu_pd((y) + 8,  _mm512_fmadd_pd(T1, SG, C1));                   \
        _mm512_storeu_pd((y) + 16, _mm512_fmadd_pd(T2, SG, C2));                   \
        _mm512_storeu_pd((y) + 24, _mm512_fmadd_pd(T3, SG, C3));                   \
        __m512d h0 = _mm512_fnmadd_pd(T0, SG, C0);                                 \
        __m512d h1 = _mm512_fnmadd_pd(T1, SG, C1);                                 \
        __m512d h2 = _mm512_fnmadd_pd(T2, SG, C2);                                 \
        __m512d h3 = _mm512_fnmadd_pd(T3, SG, C3);                                 \
        _mm512_storeu_pd((y) + 32, _mm512_shuffle_f64x2(h3, h3, 0x1B));            \
        _mm512_storeu_pd((y) + 40, _mm512_shuffle_f64x2(h2, h2, 0x1B));            \
        _mm512_storeu_pd((y) + 48, _mm512_shuffle_f64x2(h1, h1, 0x1B));            \
        if (fulltail)                                                              \
            _mm512_storeu_pd((y) + 56, _mm512_shuffle_f64x2(h0, h0, 0x1B));        \
        else                                                                       \
            _mm512_mask_storeu_pd((y) + 56, 0x3F,                                  \
                                  _mm512_shuffle_f64x2(h0, h0, 0x1B));             \
    } while (0)
    GDP_ZSTORE(yA, CA0, CA1, CA2, CA3, SA0, SA1, SA2, SA3);
    GDP_ZSTORE(yB, CB0, CB1, CB2, CB3, SB0, SB1, SB2, SB3);
    #undef GDP_ZSTORE
}

/* z-transform of the mirrored x-row pair (rows j and 31-j) with the x-fold
 * FUSED into the drain (gen_r3): the old path stored both z-spectra zA/zB to
 * stack rows (fulltail) and immediately re-loaded them to build
 * U = zA + zB, V = swap(zA - zB) -- 16 stores + 16 loads per pair through a
 * store->load forwarding chain.  Here the GEMM accumulators are summed/
 * diffed first (fold commutes with the linear combine), then ONE combine
 * feeds each U/V vector directly:
 *   U 1st half = Csum + SG*swap(Ssum),  U 2nd = rev(Csum - SG*swap(Ssum))
 *   V 1st half = swap(Cdif) - SG*Sdif,  V 2nd = rev(swap(Cdif) + SG*Sdif)
 * (swap/rev commute; swap(SG*swap(x)) = -SG*x).  Pad lanes 62,63 of U/V get
 * finite junk exactly like the old fulltail stores -- masked downstream. */
/* gen_r4: zmm fold of one z-row (31 complex + pad) into the pair-slot u/v
 * buffers: u[2j..2j+1] = x_j + x_{31-j}, v[2j..2j+1] = x_j - x_{31-j} for
 * j = 1..15, built from 8 zmm group loads + 4 lane-reversal shuffles + 8
 * add/sub (the old form was 15 xmm iterations, ~90 uops/row).  Slot j = 0
 * receives x_0 + x_31 (pad junk) -- never read by the GEMM (j >= 1).
 * p0 gets x_0 (the C-accumulator seed).  PAD=0 masks the tail group (the
 * flat row has no complex 31; a full load could run past the volume).
 * crow != NULL applies the chain map to the row FIRST (map-on-load), with
 * op order identical to map_volume -- same 8-point groups, same rounding,
 * so fused-chain results are bit-identical to the r3 separate-pass chain. */
static __attribute__((always_inline)) inline
void zrow31_fold(const double *x, const double *crow,
                 double *restrict u, double *restrict v,
                 double *restrict p0, const int PAD)
{
    __m512d g0 = _mm512_loadu_pd(x);      __m512d g1 = _mm512_loadu_pd(x + 8);
    __m512d g2 = _mm512_loadu_pd(x + 16); __m512d g3 = _mm512_loadu_pd(x + 24);
    __m512d g4 = _mm512_loadu_pd(x + 32); __m512d g5 = _mm512_loadu_pd(x + 40);
    __m512d g6 = _mm512_loadu_pd(x + 48);
    __m512d g7 = PAD ? _mm512_loadu_pd(x + 56)
                     : _mm512_maskz_loadu_pd((__mmask8)0x3F, x + 56);
    if (crow) {
        const __m512d ONE  = _mm512_set1_pd(1.0);
        const __m512d TH   = _mm512_set1_pd(1.5);
        const __m512d HALF = _mm512_set1_pd(0.5);
        const __m512d TINY = _mm512_set1_pd(1e-300);
        #define GDP_MAPG(ga, gb, off) do {                                        \
            __m512d w0 = _mm512_add_pd(ga, _mm512_loadu_pd(crow + (off)));        \
            __m512d w1 = _mm512_add_pd(gb, _mm512_loadu_pd(crow + (off) + 8));    \
            __m512d q0 = _mm512_mul_pd(w0, w0), q1 = _mm512_mul_pd(w1, w1);       \
            __m512d m2 = _mm512_add_pd(_mm512_unpacklo_pd(q0, q1),                \
                                       _mm512_unpackhi_pd(q0, q1));               \
            __m512d m2c = _mm512_max_pd(m2, TINY);                                \
            __m512d r = _mm512_rsqrt14_pd(m2c);                                   \
            __m512d hm = _mm512_mul_pd(m2c, HALF);                                \
            r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hm, r), r, TH));  \
            r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hm, r), r, TH));  \
            __m512d dd = _mm512_fmadd_pd(m2c, r, ONE);                            \
            GDP_MAPG_REC;                                                         \
            ga = _mm512_mul_pd(w0, _mm512_unpacklo_pd(rec, rec));                 \
            gb = _mm512_mul_pd(w1, _mm512_unpackhi_pd(rec, rec));                 \
        } while (0)
#ifdef GDP_MAP_RCP
        /* ladder form for the cross-arch/in-situ race: in THIS codelet the
         * divider sits otherwise idle under the z-GEMM, so div is expected
         * to win, but the div-vs-ladder choice is a property of the
         * surrounding code (gen_pfa_small r3 / gen_batchlane r3) -- knob kept */
        const __m512d TWO = _mm512_set1_pd(2.0);
        #define GDP_MAPG_REC                                                      \
            __m512d rec = _mm512_rcp14_pd(dd);                                    \
            rec = _mm512_mul_pd(rec, _mm512_fnmadd_pd(dd, rec, TWO));             \
            rec = _mm512_mul_pd(rec, _mm512_fnmadd_pd(dd, rec, TWO))
#else
        #define GDP_MAPG_REC                                                      \
            __m512d rec = _mm512_div_pd(ONE, dd)
#endif
        GDP_MAPG(g0, g1, 0);
        GDP_MAPG(g2, g3, 16);
        GDP_MAPG(g4, g5, 32);
        GDP_MAPG(g6, g7, 48);
        #undef GDP_MAPG
        #undef GDP_MAPG_REC
    }
    _mm_store_pd(p0, _mm512_castpd512_pd128(g0));
    __m512d r7 = _mm512_shuffle_f64x2(g7, g7, 0x1B);   /* c31,30,29,28 */
    __m512d r6 = _mm512_shuffle_f64x2(g6, g6, 0x1B);   /* c27..24 */
    __m512d r5 = _mm512_shuffle_f64x2(g5, g5, 0x1B);   /* c23..20 */
    __m512d r4 = _mm512_shuffle_f64x2(g4, g4, 0x1B);   /* c19..16 */
    _mm512_store_pd(u + 0,  _mm512_add_pd(g0, r7));    /* j = 0..3   */
    _mm512_store_pd(u + 8,  _mm512_add_pd(g1, r6));    /* j = 4..7   */
    _mm512_store_pd(u + 16, _mm512_add_pd(g2, r5));    /* j = 8..11  */
    _mm512_store_pd(u + 24, _mm512_add_pd(g3, r4));    /* j = 12..15 */
    _mm512_store_pd(v + 0,  _mm512_sub_pd(g0, r7));
    _mm512_store_pd(v + 8,  _mm512_sub_pd(g1, r6));
    _mm512_store_pd(v + 16, _mm512_sub_pd(g2, r5));
    _mm512_store_pd(v + 24, _mm512_sub_pd(g3, r4));
}

static __attribute__((always_inline)) inline
void zpair31_uv(const double *xA, const double *xB,
                const double *cA, const double *cB,
                double *restrict u, double *restrict v,
                const double *restrict ctd, const double *restrict std,
                const int PAD)
{
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);

    __attribute__((aligned(64))) double ua[32], va[32], ub2[32], vb2[32];
    __attribute__((aligned(16))) double pa[2], pb[2];
    zrow31_fold(xA, cA, ua, va, pa, PAD);
    zrow31_fold(xB, cB, ub2, vb2, pb, PAD);
    __m512d xa0 = _mm512_broadcast_f64x2(_mm_load_pd(pa));
    __m512d xb0 = _mm512_broadcast_f64x2(_mm_load_pd(pb));
    __m512d CA0 = xa0, CA1 = xa0, CA2 = xa0, CA3 = xa0;
    __m512d CB0 = xb0, CB1 = xb0, CB2 = xb0, CB3 = xb0;
    __m512d SA0 = _mm512_setzero_pd(), SA1 = SA0, SA2 = SA0, SA3 = SA0;
    __m512d SB0 = SA0, SB1 = SA0, SB2 = SA0, SB3 = SA0;
    for (int j = 1; j <= 15; ++j) {
        const double *cr = ctd + (size_t)(j - 1) * 32;
        const double *sr = std + (size_t)(j - 1) * 32;
        __m512d c0 = _mm512_load_pd(cr + 0),  c1 = _mm512_load_pd(cr + 8);
        __m512d c2 = _mm512_load_pd(cr + 16), c3 = _mm512_load_pd(cr + 24);
        __m512d s0 = _mm512_load_pd(sr + 0),  s1 = _mm512_load_pd(sr + 8);
        __m512d s2 = _mm512_load_pd(sr + 16), s3 = _mm512_load_pd(sr + 24);
        __m512d uA = _mm512_broadcast_f64x2(_mm_load_pd(ua + 2 * j));
        __m512d vA = _mm512_broadcast_f64x2(_mm_load_pd(va + 2 * j));
        __m512d uB = _mm512_broadcast_f64x2(_mm_load_pd(ub2 + 2 * j));
        __m512d vB = _mm512_broadcast_f64x2(_mm_load_pd(vb2 + 2 * j));
        CA0 = _mm512_fmadd_pd(c0, uA, CA0); CA1 = _mm512_fmadd_pd(c1, uA, CA1);
        CA2 = _mm512_fmadd_pd(c2, uA, CA2); CA3 = _mm512_fmadd_pd(c3, uA, CA3);
        SA0 = _mm512_fmadd_pd(s0, vA, SA0); SA1 = _mm512_fmadd_pd(s1, vA, SA1);
        SA2 = _mm512_fmadd_pd(s2, vA, SA2); SA3 = _mm512_fmadd_pd(s3, vA, SA3);
        CB0 = _mm512_fmadd_pd(c0, uB, CB0); CB1 = _mm512_fmadd_pd(c1, uB, CB1);
        CB2 = _mm512_fmadd_pd(c2, uB, CB2); CB3 = _mm512_fmadd_pd(c3, uB, CB3);
        SB0 = _mm512_fmadd_pd(s0, vB, SB0); SB1 = _mm512_fmadd_pd(s1, vB, SB1);
        SB2 = _mm512_fmadd_pd(s2, vB, SB2); SB3 = _mm512_fmadd_pd(s3, vB, SB3);
    }
    #define GDP_UVSTORE(i, CAi, CBi, SAi, SBi) do {                            \
        __m512d Cs = _mm512_add_pd(CAi, CBi);                                  \
        __m512d Cd = _mm512_sub_pd(CAi, CBi);                                  \
        __m512d Ts = _mm512_permute_pd(_mm512_add_pd(SAi, SBi), 0x55);         \
        __m512d Sd = _mm512_sub_pd(SAi, SBi);                                  \
        __m512d Pc = _mm512_permute_pd(Cd, 0x55);                              \
        _mm512_store_pd(u + 8 * (i), _mm512_fmadd_pd(Ts, SG, Cs));             \
        __m512d hu_ = _mm512_fnmadd_pd(Ts, SG, Cs);                            \
        _mm512_store_pd(u + 32 + 8 * (3 - (i)),                                \
                        _mm512_shuffle_f64x2(hu_, hu_, 0x1B));                 \
        _mm512_store_pd(v + 8 * (i), _mm512_fnmadd_pd(Sd, SG, Pc));            \
        __m512d hv_ = _mm512_fmadd_pd(Sd, SG, Pc);                             \
        _mm512_store_pd(v + 32 + 8 * (3 - (i)),                                \
                        _mm512_shuffle_f64x2(hv_, hv_, 0x1B));                 \
    } while (0)
    GDP_UVSTORE(0, CA0, CB0, SA0, SB0);
    GDP_UVSTORE(1, CA1, CB1, SA1, SB1);
    GDP_UVSTORE(2, CA2, CB2, SA2, SB2);
    GDP_UVSTORE(3, CA3, CB3, SA3, SB3);
    #undef GDP_UVSTORE
}

static __attribute__((always_inline)) inline
void zsingle31_p(const double *x, const double *crow, double *y,
                 const double *restrict ctd, const double *restrict std,
                 const int PAD, const int fulltail)
{
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    {
        __attribute__((aligned(64))) double ub[32], vb[32];
        __attribute__((aligned(16))) double p0[2];
        zrow31_fold(x, crow, ub, vb, p0, PAD);
        __m512d x0 = _mm512_broadcast_f64x2(_mm_load_pd(p0));
        __m512d C0 = x0, C1 = x0, C2 = x0, C3 = x0;
        __m512d S0 = _mm512_setzero_pd(), S1 = S0, S2 = S0, S3 = S0;
        for (int j = 1; j <= 15; ++j) {
            __m512d u = _mm512_broadcast_f64x2(_mm_load_pd(ub + 2 * j));
            __m512d v = _mm512_broadcast_f64x2(_mm_load_pd(vb + 2 * j));
            const double *cr = ctd + (size_t)(j - 1) * 32;
            const double *sr = std + (size_t)(j - 1) * 32;
            C0 = _mm512_fmadd_pd(_mm512_load_pd(cr + 0),  u, C0);
            C1 = _mm512_fmadd_pd(_mm512_load_pd(cr + 8),  u, C1);
            C2 = _mm512_fmadd_pd(_mm512_load_pd(cr + 16), u, C2);
            C3 = _mm512_fmadd_pd(_mm512_load_pd(cr + 24), u, C3);
            S0 = _mm512_fmadd_pd(_mm512_load_pd(sr + 0),  v, S0);
            S1 = _mm512_fmadd_pd(_mm512_load_pd(sr + 8),  v, S1);
            S2 = _mm512_fmadd_pd(_mm512_load_pd(sr + 16), v, S2);
            S3 = _mm512_fmadd_pd(_mm512_load_pd(sr + 24), v, S3);
        }
        /* T = swap(re,im) of S; X_k = C + SG*T, X_{31-k} = C - SG*T */
        __m512d T0 = _mm512_permute_pd(S0, 0x55);
        __m512d T1 = _mm512_permute_pd(S1, 0x55);
        __m512d T2 = _mm512_permute_pd(S2, 0x55);
        __m512d T3 = _mm512_permute_pd(S3, 0x55);
        _mm512_storeu_pd(y + 0,  _mm512_fmadd_pd(T0, SG, C0));   /* k = 0..3   */
        _mm512_storeu_pd(y + 8,  _mm512_fmadd_pd(T1, SG, C1));   /* k = 4..7   */
        _mm512_storeu_pd(y + 16, _mm512_fmadd_pd(T2, SG, C2));   /* k = 8..11  */
        _mm512_storeu_pd(y + 24, _mm512_fmadd_pd(T3, SG, C3));   /* k = 12..15 */
        __m512d h0 = _mm512_fnmadd_pd(T0, SG, C0);
        __m512d h1 = _mm512_fnmadd_pd(T1, SG, C1);
        __m512d h2 = _mm512_fnmadd_pd(T2, SG, C2);
        __m512d h3 = _mm512_fnmadd_pd(T3, SG, C3);
        /* reversed second half: y[16..30] = X_{31-k}, k = 15..1 */
        _mm512_storeu_pd(y + 32, _mm512_shuffle_f64x2(h3, h3, 0x1B)); /* k 15..12 */
        _mm512_storeu_pd(y + 40, _mm512_shuffle_f64x2(h2, h2, 0x1B)); /* k 11..8  */
        _mm512_storeu_pd(y + 48, _mm512_shuffle_f64x2(h1, h1, 0x1B)); /* k 7..4   */
        if (fulltail)
            _mm512_storeu_pd(y + 56, _mm512_shuffle_f64x2(h0, h0, 0x1B));
        else
            _mm512_mask_storeu_pd(y + 56, 0x3F,
                                  _mm512_shuffle_f64x2(h0, h0, 0x1B)); /* k 3..1 */
    }
}

static void zpass31(const cplx *src, cplx *dst, size_t nrows,
                    const double *restrict ctd, const double *restrict std)
{
    size_t r = 0;
    for (; r + 2 <= nrows; r += 2)
        zpair31_p((const double *)(src + r * 31), (const double *)(src + r * 31 + 31),
                  (double *)(dst + r * 31), (double *)(dst + r * 31 + 31), ctd, std, 0);
    for (; r < nrows; ++r)
        zsingle31_p((const double *)(src + r * 31), NULL,
                    (double *)(dst + r * 31), ctd, std, 0, 0);
}
#endif

/* ---------------- one volume, forward 3D ---------------- */

/* All three passes are in-place safe, so a single volume buffer suffices:
 * z rows: src -> dst (src == dst allowed), then x and y passes in place on
 * dst.  Chain calls this with src == dst == state; execute with src = in,
 * dst = out. */
static void fft3d_volume(fft3d_plan *p, const cplx *src, cplx *dst)
{
    const int L = p->L;
    const size_t LL = (size_t)L * L;

#ifdef __AVX512F__
    if (L == 31) {
#ifndef GDP_GENERIC31
        fold31zx(src, dst, p->ct, p->st, p->ctd, p->std_, p->U, p->V, p->Cb);
        for (int x = 0; x < 31; ++x)
            fold31(dst + (size_t)x * LL, dst + (size_t)x * LL, 31,
                   p->ct, p->st, p->U, p->V, p->Cb);
#else
        zpass31(src, dst, LL, p->ctd, p->std_);
        fold_pass(dst, dst, L, p->hc, p->hs, p->ct, p->st,
                  (int)LL, L, (cplx *)p->U, (cplx *)p->V, (cplx *)p->Cb, p->Sb);
        for (int x = 0; x < L; ++x)
            fold_pass(dst + (size_t)x * LL, dst + (size_t)x * LL,
                      L, p->hc, p->hs, p->ct, p->st,
                      L, L, (cplx *)p->U, (cplx *)p->V, (cplx *)p->Cb, p->Sb);
#endif
        return;
    }
#endif
#ifdef __AVX512F__
    if (p->ctz)
        zpass_vec(p, src, dst, LL, (size_t)L);
    else
#endif
        zpass_generic(src, dst, L, p->hc, p->hs, p->ct, p->st, LL);

    fold_pass(dst, dst, L, p->hc, p->hs, p->ct, p->st,
              (int)LL, L, (cplx *)p->U, (cplx *)p->V, (cplx *)p->Cb, p->Sb);

    for (int x = 0; x < L; ++x)
        fold_pass(dst + (size_t)x * LL, dst + (size_t)x * LL,
                  L, p->hc, p->hs, p->ct, p->st,
                  L, L, (cplx *)p->U, (cplx *)p->V, (cplx *)p->Cb, p->Sb);
}

void fft3d_execute(fft3d_plan *p, const cplx *in, cplx *out)
{
    const size_t vol = (size_t)p->L * p->L * p->L;
    for (int b = 0; b < p->batch; ++b)
        fft3d_volume(p, in + (size_t)b * vol, out + (size_t)b * vol);
}

/* ---------------- fused map chain ---------------- */

/* state <- (z + c) / (1 + |z + c|); z and o may alias (in-place map) */
static void map_volume(const cplx *z, const cplx *restrict c,
                       cplx *o, size_t npts)
{
    const double *zp = (const double *)z;
    const double *cp = (const double *)c;
    double *op = (double *)o;
    size_t i = 0;
#ifdef __AVX512F__
    const __m512d ONE  = _mm512_set1_pd(1.0);
    const __m512d TH   = _mm512_set1_pd(1.5);
    const __m512d HALF = _mm512_set1_pd(0.5);
    const __m512d TINY = _mm512_set1_pd(1e-300);
    for (; i + 8 <= npts; i += 8) {
        __m512d w0 = _mm512_add_pd(_mm512_loadu_pd(zp + 2 * i),
                                   _mm512_loadu_pd(cp + 2 * i));
        __m512d w1 = _mm512_add_pd(_mm512_loadu_pd(zp + 2 * i + 8),
                                   _mm512_loadu_pd(cp + 2 * i + 8));
        __m512d p0 = _mm512_mul_pd(w0, w0), p1 = _mm512_mul_pd(w1, w1);
        __m512d m2 = _mm512_add_pd(_mm512_unpacklo_pd(p0, p1),
                                   _mm512_unpackhi_pd(p0, p1));
        __m512d m2c = _mm512_max_pd(m2, TINY);
        __m512d r = _mm512_rsqrt14_pd(m2c);
        __m512d hm = _mm512_mul_pd(m2c, HALF);
        r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hm, r), r, TH));
        r = _mm512_mul_pd(r, _mm512_fnmadd_pd(_mm512_mul_pd(hm, r), r, TH));
        __m512d d = _mm512_fmadd_pd(m2c, r, ONE);          /* 1 + |w| */
#ifdef GDP_MAP_RCP
        /* divider-free reciprocal: rcp14 + 2 Newton (r' = r*(2 - d*r)).
         * Raced on the node (gen_r1): LOSES ~1.5% to the exact divide even in
         * this standalone pass -- the one vdivpd per 8 points hides under the
         * ladder's own OoO window (ice L23 s6 ranking confirmed here too). */
        __m512d rec = _mm512_rcp14_pd(d);
        const __m512d TWO = _mm512_set1_pd(2.0);
        rec = _mm512_mul_pd(rec, _mm512_fnmadd_pd(d, rec, TWO));
        rec = _mm512_mul_pd(rec, _mm512_fnmadd_pd(d, rec, TWO));
#else
        __m512d rec = _mm512_div_pd(ONE, d);               /* the one divide */
#endif
        _mm512_storeu_pd(op + 2 * i,     _mm512_mul_pd(w0, _mm512_unpacklo_pd(rec, rec)));
        _mm512_storeu_pd(op + 2 * i + 8, _mm512_mul_pd(w1, _mm512_unpackhi_pd(rec, rec)));
    }
#endif
    for (; i < npts; ++i) {
        double re = zp[2 * i] + cp[2 * i];
        double im = zp[2 * i + 1] + cp[2 * i + 1];
        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
        op[2 * i] = re * sc;
        op[2 * i + 1] = im * sc;
    }
}

void fft3d_chain(fft3d_plan *p, const cplx *x0, const cplx *c,
                 cplx *final_out, int m)
{
    const int L = p->L;
    const size_t LL = (size_t)L * L, vol = LL * L;
    for (int b = 0; b < p->batch; ++b) {
        cplx *stv = final_out + (size_t)b * vol;   /* state lives in the out volume */
        const cplx *cv = c + (size_t)b * vol;
#if defined(__AVX512F__) && !defined(GDP_GENERIC31)
        if (L == 31 && p->sp) {
            /* padded-row chain: state and c live in 31-plane volumes of pitch
             * GDP_PP with 32-complex rows, so every row is 64B-aligned (no
             * masked stores, no line splits) AND no in-flight store/load pair
             * shares low-12 address bits (gen_r3 pitch + block order).  Row
             * pad lanes hold finite junk; the map is a contraction (|out|<1)
             * so they stay bounded forever.  The [992, GDP_PP) plane gap is
             * never read or written after create()'s zeroing.  Strided copies
             * at the volume boundaries amortize over the m steps. */
            const cplx *x0v = x0 + (size_t)b * vol;
            for (int pl = 0; pl < 31; ++pl)
                for (int y = 0; y < 31; ++y) {
                    const size_t rp = (size_t)pl * GDP_PP + (size_t)y * 32;
                    const size_t rf = ((size_t)pl * 31 + y) * 31;
                    memcpy(p->sp + rp, x0v + rf, 31 * sizeof(cplx));
                    memcpy(p->cp + rp, cv + rf, 31 * sizeof(cplx));
                }
            for (int s = 0; s < m; ++s) {
#ifndef GDP_NOZMAPFUSE
                /* gen_r4 LAZY MAP: step s > 0 applies step s-1's map to each
                 * state row as the z-phase loads it (sp holds the UNMAPPED
                 * FFT output between steps); the standalone map pass and its
                 * 2 MB/step of state store + re-read are gone.  Step 0 reads
                 * the raw initial state.  The last step's map materializes
                 * once, below. */
                if (s == 0)
                    fold31zx_p(p->sp, p->sp, p->ct, p->st, p->ctd, p->std_,
                               p->U, p->V, p->Cb);
                else
                    fold31zx_pm(p->sp, p->sp, p->cp, p->ct, p->st,
                                p->ctd, p->std_, p->U, p->V, p->Cb);
#else
                fold31zx_p(p->sp, p->sp, p->ct, p->st, p->ctd, p->std_,
                           p->U, p->V, p->Cb);
#endif
                for (int x = 0; x < 31; ++x) {
                    fold31_p(p->sp + (size_t)x * GDP_PP,
                             p->sp + (size_t)x * GDP_PP,
                             32, p->ct, p->st, p->U, p->V, p->Cb);
#if defined(GDP_NOZMAPFUSE) && defined(GDP_PLANEMAP)
                    map_volume(p->sp + (size_t)x * GDP_PP,
                               p->cp + (size_t)x * GDP_PP,
                               p->sp + (size_t)x * GDP_PP, 992);
#endif
                }
#if defined(GDP_NOZMAPFUSE) && !defined(GDP_PLANEMAP)
                /* separate map pass, per plane (992 = 124 exact zmm groups;
                 * the plane gap is skipped so it stays untouched) */
                for (int x = 0; x < 31; ++x)
                    map_volume(p->sp + (size_t)x * GDP_PP,
                               p->cp + (size_t)x * GDP_PP,
                               p->sp + (size_t)x * GDP_PP, 992);
#endif
            }
#ifndef GDP_NOZMAPFUSE
            /* the final step's map, materialized once per chain */
            for (int x = 0; x < 31; ++x)
                map_volume(p->sp + (size_t)x * GDP_PP,
                           p->cp + (size_t)x * GDP_PP,
                           p->sp + (size_t)x * GDP_PP, 992);
#endif
            for (int pl = 0; pl < 31; ++pl)
                for (int y = 0; y < 31; ++y)
                    memcpy(stv + ((size_t)pl * 31 + y) * 31,
                           p->sp + (size_t)pl * GDP_PP + (size_t)y * 32,
                           31 * sizeof(cplx));
            continue;
        }
        if (p->pl && p->sp) {
            /* gen_r4 generic PADDED chain (any 5 <= L <= 30): rows of PL =
             * (L+3)&~3 complex, so every volume access in all three passes
             * and the map is 64B-aligned and every fold_pass mask is full --
             * the flat chain paid a cache-line split on essentially every
             * row load/store (the r2 fold31 lesson, now for the whole class;
             * the pad columns ride through the GEMMs as bounded junk exactly
             * like the L=31 pads).  The z-pass is the r3 vector kernel with
             * a row stride. */
            const int PL = p->pl;
            const size_t PP = (size_t)L * PL;        /* plane pitch, cplx */
            const cplx *x0v = x0 + (size_t)b * vol;
            for (int pln = 0; pln < L; ++pln)
                for (int y = 0; y < L; ++y) {
                    const size_t rp = (size_t)pln * PP + (size_t)y * PL;
                    const size_t rf = ((size_t)pln * L + y) * L;
                    memcpy(p->sp + rp, x0v + rf, (size_t)L * sizeof(cplx));
                    memcpy(p->cp + rp, cv + rf, (size_t)L * sizeof(cplx));
                }
            for (int s = 0; s < m; ++s) {
                zpass_vec(p, p->sp, p->sp, LL, (size_t)PL);
                fold_pass(p->sp, p->sp, L, p->hc, p->hs, p->ct, p->st,
                          (int)PP, PL, (cplx *)p->U, (cplx *)p->V,
                          (cplx *)p->Cb, p->Sb);
                for (int x = 0; x < L; ++x)
                    fold_pass(p->sp + (size_t)x * PP, p->sp + (size_t)x * PP,
                              L, p->hc, p->hs, p->ct, p->st,
                              PL, PL, (cplx *)p->U, (cplx *)p->V,
                              (cplx *)p->Cb, p->Sb);
                map_volume(p->sp, p->cp, p->sp, (size_t)L * PP);
            }
            for (int pln = 0; pln < L; ++pln)
                for (int y = 0; y < L; ++y)
                    memcpy(stv + ((size_t)pln * L + y) * L,
                           p->sp + (size_t)pln * PP + (size_t)y * PL,
                           (size_t)L * sizeof(cplx));
            continue;
        }
#endif
        memcpy(stv, x0 + (size_t)b * vol, vol * sizeof(cplx));
        for (int s = 0; s < m; ++s) {
            /* everything in place on the state: working set = state + c
             * (953 KB at L=31, L2-resident -- the gen_rader gen_r1 lever).
             * Mapping per y-plane while L1-hot was raced and LOST (152.1 vs
             * 145.7 us/step): the map immediately re-reads chunks the fold
             * just stored at different alignments (store-forward stalls). */
            fft3d_volume(p, stv, stv);
            map_volume(stv, cv, stv, vol);
        }
    }
}

/* ---------------- plan lifecycle ---------------- */

static void *xalloc(size_t bytes)
{
    return aligned_alloc(64, (bytes + 63) & ~(size_t)63);
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L)) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->hc = L / 2;              /* even: L/2 (lone middle), odd: (L-1)/2 */
    p->hs = (L - 1) / 2;
    p->Sb = xalloc((size_t)L * sizeof(cplx));
    int ok = p->Sb != NULL;

    if (L == 31) {
        /* gen_r3: DETERMINISTIC layout -- every buffer the chain touches
         * lives at a FIXED offset in one 2 MiB-huge-page arena (identical
         * binaries read 124.0-130.4 us/step across process instances from
         * allocation/page luck; the arena pins the 4K phases and the THP
         * backing makes the L2 set mapping uniform -- gen_layout's
         * gl_map_huge / gl_arena lesson).  cp sits at page phase +2048 from
         * sp so the map's c loads never share low-12 bits with the state
         * stores in flight; the small buffers stagger by 320 B (5 lines,
         * gcd(5,64)=1 walks the line phases). */
        const size_t vb    = (size_t)31 * GDP_PP * sizeof(cplx);
        const size_t o_cp  = vb + ((2048 - (vb & 4095)) & 4095);
        const size_t o_U   = o_cp  + vb + 320;
        const size_t o_V   = o_U   + 8192 + 320;
        const size_t o_Cb  = o_V   + 8192 + 320;
        const size_t o_ctd = o_Cb  + 8192 + 320;
        const size_t o_std = o_ctd + 15 * 32 * 8 + 320;
        const size_t o_ct  = o_std + 15 * 32 * 8 + 320;
        const size_t o_st  = o_ct  + 15 * 16 * 8 + 320;
        char *a = gdp_huge(o_st + 15 * 16 * 8, &p->abase, &p->alen);
        if (a) {
            p->sp  = (cplx *)a;
            p->cp  = (cplx *)(a + o_cp);
            p->U   = (double *)(a + o_U);
            p->V   = (double *)(a + o_V);
            p->Cb  = (double *)(a + o_Cb);
            p->ctd = (double *)(a + o_ctd);
            p->std_ = (double *)(a + o_std);
            p->ct  = (double *)(a + o_ct);
            p->st  = (double *)(a + o_st);
            trig_fill(p->ct, 31, 15, 16, 0);
            trig_fill(p->st, 31, 15, 16, 1);
            trig_fill_dup31(p->ctd, 0);
            trig_fill_dup31(p->std_, 1);
            if (!ok) { fft3d_destroy(p); return NULL; }
            return p;
        }
        /* mmap unavailable: fall through to the heap layout */
    }

    p->ct = trig_table(L, p->hc, p->hc + 1, 0);
    p->st = trig_table(L, p->hs, p->hs + 1, 1);
    /* one allocation size covers both layouts: generic (rows of BC = L cplx)
     * needs at most hc*L cplx = 3.2 KB (L=20); the L=31 padded layout needs
     * 15 x 64 doubles = 7.7 KB (U/V) and 16 x 64 = 8 KB (Cb, whose generic
     * shape (hc+2)*L cplx = 3.9 KB also fits).  Zeroed so the L=31 padding
     * lanes start (and stay) zero. */
    p->U  = xalloc(8192);
    p->V  = xalloc(8192);
    p->Cb = xalloc(8192);
    ok = ok && p->ct && p->st && p->U && p->V && p->Cb;
    if (ok) {
        memset(p->U, 0, 8192);
        memset(p->V, 0, 8192);
        memset(p->Cb, 0, 8192);
    }
#ifdef __AVX512F__
    if (ok && L >= 5 && L != 31) {
        /* generic vector z-pass tables + store layout (gen_r3) */
        p->kq = (p->hc + 4) / 4;
        p->ctz = trig_table_dupz(L, p->hc, p->kq, p->hc, 0);
        p->stz = trig_table_dupz(L, p->hs, p->kq, p->hs, 1);
        ok = p->ctz && p->stz;
        const int n1 = 2 * (p->hc + 1);          /* first-half doubles */
        for (int q = 0; q < 4; ++q) {
            const int rem = n1 - 8 * q;
            p->zfm[q] = (unsigned char)(rem >= 8 ? 0xFF
                        : rem > 0 ? ((1u << rem) - 1) : 0);
            unsigned m = 0;
            for (int t = 0; t < 4; ++t) {        /* rev lane t holds k = 4q+3-t */
                const int k = 4 * q + 3 - t;
                if (k >= 1 && k <= p->hs) m |= 3u << (2 * t);
            }
            p->zsm[q] = (unsigned char)m;
            p->zso2[q] = m ? 2 * (L - (4 * q + 3)) : 0;
        }
        if (ok) {
            /* gen_r4: padded chain volumes -- rows of (L+3)&~3 complex, one
             * allocation, c mirror at page phase +2048 (the r3 alias lesson).
             * When L is already a multiple of 4 the flat rows are already
             * 64B-aligned and the padded path is pure copy overhead (raced:
             * L=12 +2..9%, L=20 +0.5%) -- flat stays.  Allocation failure
             * falls back to the flat chain, not to a failed plan.  Pads must
             * start finite; c pads stay zero. */
            p->pl = (L % 4) ? ((L + 3) & ~3) : 0;
            const size_t vb = (size_t)L * L * (p->pl ? p->pl : 1) * sizeof(cplx);
            const size_t coff = vb + ((2048 - (vb & 4095)) & 4095);
            if (p->pl) {
                p->sp = xalloc(coff + vb);
                if (p->sp) {
                    p->cp = (cplx *)((char *)p->sp + coff);
                    memset(p->sp, 0, coff + vb);
                } else {
                    p->pl = 0;
                }
            }
        }
    }
#endif
    if (L == 31) {
        /* heap fallback of the arena layout (mmap failed): cp still placed
         * at page phase +2048 from sp inside one allocation */
        p->ctd = xalloc(15 * 32 * 8);
        p->std_ = xalloc(15 * 32 * 8);
        const size_t vb = (size_t)31 * GDP_PP * sizeof(cplx);
        const size_t coff = vb + ((2048 - (vb & 4095)) & 4095);
        p->sp = xalloc(coff + vb);
        p->cp = p->sp ? (cplx *)((char *)p->sp + coff) : NULL;
        ok = ok && p->ctd && p->std_ && p->sp;
        if (ok) {   /* pads must start finite (c pads stay zero forever) */
            trig_fill_dup31(p->ctd, 0);
            trig_fill_dup31(p->std_, 1);
            memset(p->sp, 0, coff + vb);
        }
    }
    if (!ok) { fft3d_destroy(p); return NULL; }
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->Sb);
    free(p->ctz); free(p->stz);
    if (p->abase) {   /* sp/cp/U/V/Cb/ctd/std_/ct/st all live in the arena */
        munmap(p->abase, p->alen);
    } else {
        free(p->ct); free(p->st); free(p->ctd); free(p->std_);
        free(p->U); free(p->V); free(p->Cb);
        free(p->sp);   /* cp lives inside the sp allocation -- not freed */
    }
    free(p);
}
