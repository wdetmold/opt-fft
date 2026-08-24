/* gen_dense_prime -- direct dense prime-class entry, round gen_r1.
 *
 * Conjugate-pair FOLDED dense DFT per axis (the L13/L17/L23 winning arithmetic
 * from the ice campaign): for each pencil, u_j = x_j + x_{L-j}, v_j = x_j - x_{L-j},
 * then X_k = C_k - i S_k and X_{L-k} = C_k + i S_k with C_k = x_0 + sum_j u_j cos(2pi jk/L)
 * and S_k = sum_j v_j sin(2pi jk/L) -- all constants REAL, so every multiply on
 * interleaved complex data is a plain real FMA.  ~4h^2 real FMAs per pencil
 * (h = floor(L/2)) instead of the 8h^2 of the unfolded dense matrix.
 *
 * Pass structure (per volume) needs NO transpose anywhere:
 *   P1: z-axis rows (contiguous) via broadcast row-GEMM   (in-place safe)
 *   P2: x-axis, contract slowest, inner = L^2, register-tiled GEMM (out-of-place)
 *   P3: y-axis, per k0-plane, contract slowest, inner = L (in place)
 * execute: in -> t1 (z), t1 -> out (x), out in place (y);
 * chain:   state in place (z), state -> t1 (x), t1 in place (y), map t1+c -> state,
 * so the whole chain step's working set is state + t1 + c = 3 volumes.
 *
 * fft3d_chain: volume-resident chain (each volume runs all m steps while its
 * buffers stay cache-resident -- corpus consensus), map fused as a separate
 * L2-hot pass with the ice s6 arithmetic: pair-compressed |w|^2, rsqrt14 seed
 * + 2 Newton, d = fma(m2, r, 1), ONE vdivpd per 8 points, two mul-outs.
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

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
    cplx *t1;            /* scratch volume */
    cplx *U, *V;         /* fold buffers  [hc][L], [hs][L] complex */
    cplx *Cb, *Sb;       /* C-block buffer [(hc+1)][L] / scalar-path S accumulator */
};

const char *fft3d_name(void) { return "gen_dense_prime"; }
const char *fft3d_description(void)
{
    return "folded dense prime p<=31: 4h^2-FMA conjugate-pair fold, transpose-free "
           "z-row/x/y passes (AVX-512 row-GEMM at L=31), volume-resident fused chain "
           "(rsqrt14+2NR, one vdivpd per 8 pts)";
}
int fft3d_supports(int L) { return L == 31 || L == 10 || L == 12 || L == 15 || L == 20; }

/* ---------------- tables ---------------- */

static double *trig_table(int L, int rows, int cols, int want_sin)
{
    size_t n = (size_t)rows * cols;
    double *t = aligned_alloc(64, ((n * sizeof(double)) + 63) & ~(size_t)63);
    if (!t) return NULL;
    for (int j = 1; j <= rows; ++j)
        for (int k = 0; k < cols; ++k) {
            long m = ((long)j * k) % L;
            long double th = 2.0L * PIL * (long double)m / (long double)L;
            t[(size_t)(j - 1) * cols + k] = want_sin ? (double)sinl(th) : (double)cosl(th);
        }
    return t;
}

/* duplicated-pair layout for the L=31 vector z-pass: row j-1 holds
 * (w_{j,0}, w_{j,0}, w_{j,1}, w_{j,1}, ..., w_{j,15}, w_{j,15})  = 32 doubles */
static double *trig_table_dup31(int want_sin)
{
    double *t = aligned_alloc(64, 15 * 32 * sizeof(double));
    if (!t) return NULL;
    for (int j = 1; j <= 15; ++j)
        for (int k = 0; k <= 15; ++k) {
            long m = ((long)j * k) % 31;
            long double th = 2.0L * PIL * (long double)m / 31.0L;
            double w = want_sin ? (double)sinl(th) : (double)cosl(th);
            t[(size_t)(j - 1) * 32 + 2 * k]     = w;
            t[(size_t)(j - 1) * 32 + 2 * k + 1] = w;
        }
    return t;
}

/* ---------------- folded contraction of the SLOWEST axis ----------------
 * src is an (L x inner) complex matrix, row stride = inner; dst likewise.
 * Processes inner in blocks of BC (BC must divide inner).
 *
 * AVX-512 form: two register-tiled GEMMs per block, j innermost with the
 * accumulators held in registers (the round-1 axpy form re-loaded and
 * re-stored the accumulator every j: 155 us/pass -> this shape).
 *   C-GEMM:  Cblk[k][*] = x0 + sum_j ct[j][k] * U[j][*]      k = 0..hc
 *   S-GEMM+combine: S in registers, then X_k = C - iS, X_{L-k} = C + iS. */
#ifdef __AVX512F__
static void fold_pass(const cplx *restrict src, cplx *restrict dst,
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
#else
static void fold_pass(const cplx *restrict src, cplx *restrict dst,
                      int L, int hc, int hs,
                      const double *restrict ct, const double *restrict st,
                      int inner, int BC,
                      cplx *restrict Ub, cplx *restrict Vb,
                      cplx *restrict Cb, cplx *restrict Sb)
{
    const int cw = hc + 1, sw = hs + 1;
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

        const double *x0 = (const double *)(src + c0);
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

/* ---------------- z-axis pass: contiguous rows ---------------- */

/* generic scalar row transform (used for the small composite sizes; also the
 * non-AVX512 fallback for L=31) */
static void zpass_generic(const cplx *restrict src, cplx *restrict dst,
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
/* L=31 vector row transform: k dimension lives in 4 zmm (16 complex outputs of
 * each half-spectrum), u_j/v_j broadcast as 128-bit pairs, tables pre-duplicated.
 * Rows processed in PAIRS so the 8 table loads per j are shared between two
 * rows' 16 FMAs (single-row form: 10 loads / 8 FMA, pair form: 12 / 16). */
static void zpass31_pair(const cplx *restrict src, cplx *restrict dst,
                         const double *restrict ctd, const double *restrict std)
{
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    const double *xA = (const double *)src;
    const double *xB = (const double *)(src + 31);
    double *yA = (double *)dst;
    double *yB = (double *)(dst + 31);

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
        _mm512_mask_storeu_pd((y) + 56, 0x3F, _mm512_shuffle_f64x2(h0, h0, 0x1B)); \
    } while (0)
    GDP_ZSTORE(yA, CA0, CA1, CA2, CA3, SA0, SA1, SA2, SA3);
    GDP_ZSTORE(yB, CB0, CB1, CB2, CB3, SB0, SB1, SB2, SB3);
    #undef GDP_ZSTORE
}

static void zpass31(const cplx *restrict src, cplx *restrict dst, size_t nrows,
                    const double *restrict ctd, const double *restrict std)
{
    const __m512d SG = _mm512_setr_pd(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    size_t r = 0;
    for (; r + 2 <= nrows; r += 2)
        zpass31_pair(src + r * 31, dst + r * 31, ctd, std);
    for (; r < nrows; ++r) {
        const double *x = (const double *)(src + r * 31);
        double *y = (double *)(dst + r * 31);

        __attribute__((aligned(64))) double ub[32], vb[32];
        for (int j = 1; j <= 15; ++j) {
            __m128d a = _mm_loadu_pd(x + 2 * j);
            __m128d b = _mm_loadu_pd(x + 2 * (31 - j));
            _mm_store_pd(ub + 2 * j, _mm_add_pd(a, b));
            _mm_store_pd(vb + 2 * j, _mm_sub_pd(a, b));
        }
        __m512d x0 = _mm512_broadcast_f64x2(_mm_loadu_pd(x));
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
        _mm512_mask_storeu_pd(y + 56, 0x3F,
                              _mm512_shuffle_f64x2(h0, h0, 0x1B));    /* k 3..1   */
    }
}
#endif

/* ---------------- one volume, forward 3D ---------------- */

/* P1 z rows: src -> zdst (in-place safe: each row is consumed before stored);
 * P2 contract x: zdst -> xdst (must be distinct buffers);
 * P3 contract y: xdst in place, plane by plane (fold_pass copies row 0 out). */
static void fft3d_volume(fft3d_plan *p, const cplx *restrict src,
                         cplx *zdst, cplx *xdst)
{
    const int L = p->L;
    const size_t LL = (size_t)L * L;

#ifdef __AVX512F__
    if (L == 31)
        zpass31(src, zdst, LL, p->ctd, p->std_);
    else
#endif
        zpass_generic(src, zdst, L, p->hc, p->hs, p->ct, p->st, LL);

    fold_pass(zdst, xdst, L, p->hc, p->hs, p->ct, p->st,
              (int)LL, L, p->U, p->V, p->Cb, p->Sb);

    for (int x = 0; x < L; ++x)
        fold_pass(xdst + (size_t)x * LL, xdst + (size_t)x * LL,
                  L, p->hc, p->hs, p->ct, p->st,
                  L, L, p->U, p->V, p->Cb, p->Sb);
}

void fft3d_execute(fft3d_plan *p, const cplx *in, cplx *out)
{
    const size_t vol = (size_t)p->L * p->L * p->L;
    for (int b = 0; b < p->batch; ++b)
        fft3d_volume(p, in + (size_t)b * vol, p->t1, out + (size_t)b * vol);
}

/* ---------------- fused map chain ---------------- */

/* state <- (z + c) / (1 + |z + c|), z = t1 (FFT output), one volume */
static void map_volume(const cplx *restrict z, const cplx *restrict c,
                       cplx *restrict o, size_t npts)
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
        memcpy(stv, x0 + (size_t)b * vol, vol * sizeof(cplx));
        for (int s = 0; s < m; ++s) {
            /* z = FFT3(state): z-pass in place on the state, x-pass into t1,
             * y-pass in place -- 3-buffer working set (state + t1 + c) */
            fft3d_volume(p, stv, stv, p->t1);
            map_volume(p->t1, cv, stv, vol);
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

    size_t vol = (size_t)L * L * L;
    p->ct = trig_table(L, p->hc, p->hc + 1, 0);
    p->st = trig_table(L, p->hs, p->hs + 1, 1);
    p->t1 = xalloc(vol * sizeof(cplx));
    p->U  = xalloc((size_t)p->hc * L * sizeof(cplx));
    p->V  = xalloc((size_t)p->hs * L * sizeof(cplx));
    /* C-block buffer: rows k=0..hc of BC<=L complex, plus one padding row (the
     * combine's unmasked tail load reads up to 8 doubles past the last row) */
    p->Cb = xalloc((size_t)(p->hc + 2) * L * sizeof(cplx));
    p->Sb = xalloc((size_t)L * sizeof(cplx));
    int ok = p->ct && p->st && p->t1 && p->U && p->V && p->Cb && p->Sb;
    if (L == 31) {
        p->ctd = trig_table_dup31(0);
        p->std_ = trig_table_dup31(1);
        ok = ok && p->ctd && p->std_;
    }
    if (!ok) { fft3d_destroy(p); return NULL; }
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->ct); free(p->st); free(p->ctd); free(p->std_);
    free(p->t1); free(p->U); free(p->V); free(p->Cb); free(p->Sb);
    free(p);
}
