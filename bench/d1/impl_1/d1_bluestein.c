/* d1_bluestein — Bluestein chirp-Z transform, the ANY-length fallback.
 *
 * Structure:
 *   - A power-of-two forward-FFT engine: radix-4 Stockham autosort (natural order in and
 *     out, no bit reversal), with one final radix-2 stage when log2(M) is odd. Twiddles
 *     precomputed per stage at plan time. All butterflies written on explicit re/im
 *     doubles so gcc emits FMA without the C99 __muldc3 NaN slow path.
 *   - L a power of two: the engine runs directly (M = L). The first stage reads `in`,
 *     the ping-pong parity is arranged so the last stage lands in `out`: zero copies.
 *   - Any other L: chirp-Z with M = next_pow2(2L-1).
 *       plan:  a_n = exp(-i pi n^2 / L)   (argument reduced exactly: n^2 mod 2L)
 *              bhat = FFT_M(b) / M with b_0 = 1, b_n = b_{M-n} = conj(a_n)
 *       exec:  y_k = a_k * conj( FFT_M( conj( FFT_M(a*x, 0-pad) * bhat ) ) )_k , k < L
 *     The inverse FFT is done as conj(FFT(conj(.))); both conjugations fold into the
 *     pointwise-multiply and post-chirp loops, so only forward twiddles exist.
 *   - BATCH LANES: for B >= 8, blocks of W=8 vectors are transposed to [position][lane]
 *     layout. The identical Stockham passes then run with every butterfly line W
 *     contiguous complex wide (stage s scaled by W), twiddles broadcast — full zmm
 *     lanes with zero shuffles. Lane kernels are compiled at prefer-vector-width=512
 *     (gcc's -march=native default is 256-bit); the latency-bound scalar paths keep
 *     the default width, which measured faster at B=1.
 *   - fft1d_chain: owns the whole m-step map chain, fusing post-chirp + (z+c)/(1+|z+c|)
 *     into one pass. In the batched chain the state and the c field live in lane layout
 *     for the WHOLE chain: one transpose in, m fused steps, one transpose out.
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../fft1d_api.h"

#define MAXSTAGE 34
#define LANES 8

#if defined(__AVX512F__)
#define VEC512 __attribute__((target("prefer-vector-width=512")))
#else
#define VEC512
#endif

struct fft1d_plan {
    int L, batch, M, is_pow2, nstage;
    int st_n[MAXSTAGE], st_s[MAXSTAGE], st_tw[MAXSTAGE], st_r4[MAXSTAGE];
    int lst_s[MAXSTAGE];     /* lane-path stage strides: st_s * LANES */
    double *tw;              /* concatenated per-stage twiddles (forward sign) */
    double *a;               /* chirp a_n, 2L doubles (chirp path only) */
    double *bhat;            /* FFT_M(kernel)/M, 2M doubles (chirp path only) */
    double *s0, *s1, *s2;    /* scalar scratch, 2M doubles each */
    double *l0, *l1, *l2;    /* lane scratch, 2*LANES*M doubles each (batch>=LANES) */
    double *state;           /* chain state, 2*L*batch doubles */
    double *lc;              /* chain c field in lane layout, 2*L*batch doubles */
};

const char *fft1d_name(void){ return "d1_bluestein"; }
const char *fft1d_description(void){
    return "Bluestein chirp-Z any-L fallback: radix-4 Stockham pow2 engine, 8-wide zmm batch lanes, fused chirp+map chain";
}
int fft1d_supports(int L){ return L >= 1 && L <= (1 << 24); }

static double *amalloc(size_t ndoubles){
    void *p = NULL;
    if (posix_memalign(&p, 64, ndoubles * sizeof(double)) != 0) return NULL;
    return (double *)p;
}

/* ---- Stockham passes (interleaved re/im; x != y). Bodies shared between the
 * default-width build (scalar path) and the 512-bit build (lane path). ---- */

#define PASS4_BODY                                                              \
    for (int p = 0; p < m; ++p) {                                               \
        const double w1r = tw[0], w1i = tw[1], w2r = tw[2],                     \
                     w2i = tw[3], w3r = tw[4], w3i = tw[5];                     \
        tw += 6;                                                                \
        const double *restrict xa = x + 2 * (size_t)s * p;                      \
        const double *restrict xb = xa + 2 * (size_t)s * m;                     \
        const double *restrict xc = xb + 2 * (size_t)s * m;                     \
        const double *restrict xd = xc + 2 * (size_t)s * m;                     \
        double *restrict ya = y + 8 * (size_t)s * p;                            \
        double *restrict yb = ya + 2 * (size_t)s;                               \
        double *restrict yc = ya + 4 * (size_t)s;                               \
        double *restrict yd = ya + 6 * (size_t)s;                               \
        for (int q = 0; q < 2 * s; q += 2) {                                    \
            double ar = xa[q], ai = xa[q + 1];                                  \
            double br = xb[q], bi = xb[q + 1];                                  \
            double cr = xc[q], ci = xc[q + 1];                                  \
            double dr = xd[q], di = xd[q + 1];                                  \
            double apcr = ar + cr, apci = ai + ci;                              \
            double amcr = ar - cr, amci = ai - ci;                              \
            double bpdr = br + dr, bpdi = bi + di;                              \
            double bmdr = br - dr, bmdi = bi - di;                              \
            ya[q] = apcr + bpdr; ya[q + 1] = apci + bpdi;                       \
            double t1r = amcr + bmdi, t1i = amci - bmdr; /* (a-c) - i(b-d) */   \
            double t2r = apcr - bpdr, t2i = apci - bpdi;                        \
            double t3r = amcr - bmdi, t3i = amci + bmdr; /* (a-c) + i(b-d) */   \
            yb[q] = w1r * t1r - w1i * t1i; yb[q + 1] = w1r * t1i + w1i * t1r;   \
            yc[q] = w2r * t2r - w2i * t2i; yc[q + 1] = w2r * t2i + w2i * t2r;   \
            yd[q] = w3r * t3r - w3i * t3i; yd[q + 1] = w3r * t3i + w3i * t3r;   \
        }                                                                       \
    }

/* n == 2, p == 0, twiddle 1: re/im never mix, so plain vector add/sub. */
#define PASS2_BODY                                                              \
    const double *restrict xb = x + 2 * (size_t)s;                              \
    double *restrict yb = y + 2 * (size_t)s;                                    \
    for (int q = 0; q < 2 * s; ++q) {                                           \
        double u = x[q], v = xb[q];                                             \
        y[q] = u + v;                                                           \
        yb[q] = u - v;                                                          \
    }

static void pass4(int m, int s, const double *restrict tw,
                  const double *restrict x, double *restrict y)
{ PASS4_BODY }

static void pass2(int s, const double *restrict x, double *restrict y)
{ PASS2_BODY }

static VEC512 void pass4_z(int m, int s, const double *restrict tw,
                           const double *restrict x, double *restrict y)
{ PASS4_BODY }

static VEC512 void pass2_z(int s, const double *restrict x, double *restrict y)
{ PASS2_BODY }

/* ---- Bluestein-specialized stages ----
 * First stage of the padded forward FFT: M >= 2L-1 puts every input at index >= M/2
 * at zero, so the c and d legs of the radix-4 butterfly vanish; and the b leg is zero
 * for p >= L-m. x holds only the 2L premultiplied doubles — no zero padding exists. */

#define PASS4_FIRST_H0_BODY                                                     \
    int pcut = L - m;                        /* b leg nonzero for p < pcut */   \
    for (int p = 0; p < pcut; ++p) {                                            \
        const double w1r = tw[0], w1i = tw[1], w2r = tw[2],                     \
                     w2i = tw[3], w3r = tw[4], w3i = tw[5];                     \
        tw += 6;                                                                \
        const double *restrict xa = x + 2 * (size_t)W1 * p;                     \
        const double *restrict xb = xa + 2 * (size_t)W1 * m;                    \
        double *restrict ya = y + 8 * (size_t)W1 * p;                           \
        double *restrict yb = ya + 2 * (size_t)W1;                              \
        double *restrict yc = ya + 4 * (size_t)W1;                              \
        double *restrict yd = ya + 6 * (size_t)W1;                              \
        for (int q = 0; q < 2 * W1; q += 2) {                                   \
            double ar = xa[q], ai = xa[q + 1];                                  \
            double br = xb[q], bi = xb[q + 1];                                  \
            ya[q] = ar + br; ya[q + 1] = ai + bi;                               \
            double t1r = ar + bi, t1i = ai - br;      /* a - i b */             \
            double t2r = ar - br, t2i = ai - bi;                                \
            double t3r = ar - bi, t3i = ai + br;      /* a + i b */             \
            yb[q] = w1r * t1r - w1i * t1i; yb[q + 1] = w1r * t1i + w1i * t1r;   \
            yc[q] = w2r * t2r - w2i * t2i; yc[q + 1] = w2r * t2i + w2i * t2r;   \
            yd[q] = w3r * t3r - w3i * t3i; yd[q + 1] = w3r * t3i + w3i * t3r;   \
        }                                                                       \
    }                                                                           \
    for (int p = pcut; p < m; ++p) {         /* only the a leg is nonzero */    \
        const double w1r = tw[0], w1i = tw[1], w2r = tw[2],                     \
                     w2i = tw[3], w3r = tw[4], w3i = tw[5];                     \
        tw += 6;                                                                \
        const double *restrict xa = x + 2 * (size_t)W1 * p;                     \
        double *restrict ya = y + 8 * (size_t)W1 * p;                           \
        double *restrict yb = ya + 2 * (size_t)W1;                              \
        double *restrict yc = ya + 4 * (size_t)W1;                              \
        double *restrict yd = ya + 6 * (size_t)W1;                              \
        for (int q = 0; q < 2 * W1; q += 2) {                                   \
            double ar = xa[q], ai = xa[q + 1];                                  \
            ya[q] = ar; ya[q + 1] = ai;                                         \
            yb[q] = w1r * ar - w1i * ai; yb[q + 1] = w1r * ai + w1i * ar;       \
            yc[q] = w2r * ar - w2i * ai; yc[q + 1] = w2r * ai + w2i * ar;       \
            yd[q] = w3r * ar - w3i * ai; yd[q + 1] = w3r * ai + w3i * ar;       \
        }                                                                       \
    }

static void pass4_first_h0(int m, int L, const double *restrict tw,
                           const double *restrict x, double *restrict y)
{ enum { W1 = 1 }; PASS4_FIRST_H0_BODY }

static VEC512 void pass4_first_h0_z(int m, int L, const double *restrict tw,
                                    const double *restrict x, double *restrict y)
{ enum { W1 = LANES }; PASS4_FIRST_H0_BODY }

/* Last stage, truncated: only outputs k < L are wanted (L <= M/2, so the whole upper
 * half — and for radix-4 the t=2,3 quarters — is discarded before it is computed). */

#define PASS2_LAST_TRUNC_BODY                                                   \
    const double *restrict xb = x + 2 * (size_t)W1 * s;                         \
    for (int q = 0; q < 2 * W1 * L; ++q) y[q] = x[q] + xb[q];

#define PASS4_LAST_TRUNC_BODY                                                   \
    /* m == 1: p == 0, unit twiddles. Quarters t=0 (k < s) and t=1 (k < L-s). */ \
    const double *restrict xa = x;                                              \
    const double *restrict xb = x + 2 * (size_t)W1 * s;                         \
    const double *restrict xc = x + 4 * (size_t)W1 * s;                         \
    const double *restrict xd = x + 6 * (size_t)W1 * s;                         \
    double *restrict y1 = y + 2 * (size_t)W1 * s;                               \
    for (int q = 0; q < 2 * W1 * s; ++q) y[q] = (xa[q] + xc[q]) + (xb[q] + xd[q]); \
    for (int q = 0; q < 2 * W1 * (L - s); q += 2) {                             \
        double amcr = xa[q] - xc[q], amci = xa[q + 1] - xc[q + 1];              \
        double bmdr = xb[q] - xd[q], bmdi = xb[q + 1] - xd[q + 1];              \
        y1[q] = amcr + bmdi;                          /* (a-c) - i(b-d) */      \
        y1[q + 1] = amci - bmdr;                                                \
    }

static void pass2_last_trunc(int s, int L, const double *restrict x, double *restrict y)
{ enum { W1 = 1 }; PASS2_LAST_TRUNC_BODY }

static void pass4_last_trunc(int s, int L, const double *restrict x, double *restrict y)
{ enum { W1 = 1 }; PASS4_LAST_TRUNC_BODY }

static VEC512 void pass2_last_trunc_z(int s, int L, const double *restrict x,
                                      double *restrict y)
{ enum { W1 = LANES }; PASS2_LAST_TRUNC_BODY }

static VEC512 void pass4_last_trunc_z(int s, int L, const double *restrict x,
                                      double *restrict y)
{ enum { W1 = LANES }; PASS4_LAST_TRUNC_BODY }

/* Forward FFT of length p->M over the stage plan with strides ss[] (scalar: st_s,
 * lanes: lst_s; wide selects the 512-bit pass kernels). Result is always bufA for
 * nstage >= 1, src for nstage == 0. src, bufA, bufB must be three distinct buffers;
 * src is read only by stage one. */
static const double *fft_core(const fft1d_plan *p, const int *ss, int wide,
                              const double *src, double *bufA, double *bufB)
{
    int ns = p->nstage;
    if (ns == 0) return src;
    const double *sp = src;
    double *d = (ns & 1) ? bufA : bufB;
    for (int st = 0; st < ns; ++st) {
        if (p->st_r4[st]) {
            if (wide) pass4_z(p->st_n[st] / 4, ss[st], p->tw + p->st_tw[st], sp, d);
            else      pass4(p->st_n[st] / 4, ss[st], p->tw + p->st_tw[st], sp, d);
        } else {
            if (wide) pass2_z(ss[st], sp, d);
            else      pass2(ss[st], sp, d);
        }
        sp = d;
        d = (d == bufA) ? bufB : bufA;
    }
    return sp; /* == bufA */
}

fft1d_plan *fft1d_create(int L, int batch)
{
    if (!fft1d_supports(L) || batch < 1) return NULL;
    fft1d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L; p->batch = batch;
    p->is_pow2 = (L & (L - 1)) == 0;
    int M = 1;
    if (p->is_pow2) M = L;
    else while (M < 2 * L - 1) M <<= 1;
    p->M = M;

    /* stage plan + twiddle sizing */
    int n = M, s = 1, twsize = 0;
    p->nstage = 0;
    while (n >= 4) {
        int st = p->nstage++;
        p->st_n[st] = n; p->st_s[st] = s; p->st_tw[st] = twsize; p->st_r4[st] = 1;
        twsize += 6 * (n / 4);
        s *= 4; n /= 4;
    }
    if (n == 2) {
        int st = p->nstage++;
        p->st_n[st] = 2; p->st_s[st] = s; p->st_tw[st] = 0; p->st_r4[st] = 0;
    }
    for (int st = 0; st < p->nstage; ++st) p->lst_s[st] = p->st_s[st] * LANES;

    p->tw = amalloc(twsize ? (size_t)twsize : 1);
    p->s0 = amalloc(2 * (size_t)M);
    p->s1 = amalloc(2 * (size_t)M);
    p->s2 = amalloc(2 * (size_t)M);
    p->state = amalloc(2 * (size_t)L * batch);
    if (!p->tw || !p->s0 || !p->s1 || !p->s2 || !p->state) { fft1d_destroy(p); return NULL; }
    if (batch >= LANES) {
        p->l0 = amalloc(2 * (size_t)LANES * M);
        p->l1 = amalloc(2 * (size_t)LANES * M);
        p->l2 = amalloc(2 * (size_t)LANES * M);
        p->lc = amalloc(2 * (size_t)L * batch);
        if (!p->l0 || !p->l1 || !p->l2 || !p->lc) { fft1d_destroy(p); return NULL; }
    }

    for (int st = 0; st < p->nstage; ++st) {
        if (!p->st_r4[st]) continue;
        int sn = p->st_n[st], m = sn / 4;
        double *t = p->tw + p->st_tw[st];
        for (int j = 0; j < m; ++j) {
            double ang = -2.0 * M_PI * (double)j / (double)sn;
            t[6 * j + 0] = cos(ang);       t[6 * j + 1] = sin(ang);
            t[6 * j + 2] = cos(2.0 * ang); t[6 * j + 3] = sin(2.0 * ang);
            t[6 * j + 4] = cos(3.0 * ang); t[6 * j + 5] = sin(3.0 * ang);
        }
    }

    if (!p->is_pow2) {
        p->a = amalloc(2 * (size_t)L);
        p->bhat = amalloc(2 * (size_t)M);
        double *btmp = amalloc(2 * (size_t)M);
        if (!p->a || !p->bhat || !btmp) { free(btmp); fft1d_destroy(p); return NULL; }
        /* a_n = exp(-i pi n^2 / L), argument reduced exactly via n^2 mod 2L */
        for (int i = 0; i < L; ++i) {
            long long r = ((long long)i * i) % (2LL * L);
            double ang = -M_PI * (double)r / (double)L;
            p->a[2 * i] = cos(ang); p->a[2 * i + 1] = sin(ang);
        }
        /* kernel b_0 = 1, b_n = b_{M-n} = conj(a_n) = exp(+i pi n^2 / L) */
        memset(btmp, 0, 2 * (size_t)M * sizeof(double));
        btmp[0] = 1.0; btmp[1] = 0.0;
        for (int i = 1; i < L; ++i) {
            double br = p->a[2 * i], bi = -p->a[2 * i + 1];
            btmp[2 * i] = br;           btmp[2 * i + 1] = bi;
            btmp[2 * (M - i)] = br;     btmp[2 * (M - i) + 1] = bi;
        }
        const double *bh = fft_core(p, p->st_s, 0, btmp, p->bhat, p->s0);
        double inv = 1.0 / (double)M;
        for (int i = 0; i < 2 * M; ++i) p->bhat[i] = bh[i] * inv;
        free(btmp);
    }
    return p;
}

/* ================= scalar (per-vector) paths ================= */

/* chirp-Z of one vector: x (2L doubles) -> z in s1 (before post-chirp) */
static const double *bluestein_core(fft1d_plan *p, const double *restrict x)
{
    const int L = p->L, M = p->M;
    const double *restrict a = p->a;
    double *restrict t = p->s2;
    for (int i = 0; i < L; ++i) {
        double xr = x[2 * i], xi = x[2 * i + 1];
        double ar = a[2 * i], ai = a[2 * i + 1];
        t[2 * i] = ar * xr - ai * xi;
        t[2 * i + 1] = ar * xi + ai * xr;
    }
    memset(t + 2 * L, 0, 2 * (size_t)(M - L) * sizeof(double));
    const double *U = fft_core(p, p->st_s, 0, t, p->s0, p->s1);   /* lands in s0 */
    double *restrict u = p->s0;
    const double *restrict bh = p->bhat;
    for (int i = 0; i < M; ++i) {                                 /* W = conj(U * bhat) */
        double ur = U[2 * i], ui = U[2 * i + 1];
        double br = bh[2 * i], bi = bh[2 * i + 1];
        u[2 * i] = ur * br - ui * bi;
        u[2 * i + 1] = -(ur * bi + ui * br);
    }
    return fft_core(p, p->st_s, 0, p->s0, p->s1, p->s2);          /* lands in s1 */
}

/* ================= lane (8 vectors at once) paths =================
 * Lane layout for a block of W=LANES vectors: position i, lane w at complex index
 * i*W + w. Contiguous in w, so each butterfly line is W complex = 2 zmm. */

/* pow2: transpose W vectors of length M in */
static VEC512 void lane_in_pow2(int M, const double *restrict x, double *restrict t)
{
    for (int i = 0; i < M; ++i)
        for (int w = 0; w < LANES; ++w) {
            t[2 * (i * LANES + w)]     = x[2 * ((size_t)w * M + i)];
            t[2 * (i * LANES + w) + 1] = x[2 * ((size_t)w * M + i) + 1];
        }
}

static VEC512 void lane_out_pow2(int M, const double *restrict z, double *restrict y)
{
    for (int i = 0; i < M; ++i)
        for (int w = 0; w < LANES; ++w) {
            y[2 * ((size_t)w * M + i)]     = z[2 * (i * LANES + w)];
            y[2 * ((size_t)w * M + i) + 1] = z[2 * (i * LANES + w) + 1];
        }
}

/* chirp: transpose in with pre-chirp multiply, zero-pad to M */
static VEC512 void lane_in_chirp(const fft1d_plan *p, const double *restrict x,
                                 double *restrict t)
{
    const int L = p->L, M = p->M;
    const double *restrict a = p->a;
    for (int i = 0; i < L; ++i) {
        double ar = a[2 * i], ai = a[2 * i + 1];
        for (int w = 0; w < LANES; ++w) {
            double xr = x[2 * ((size_t)w * L + i)];
            double xi = x[2 * ((size_t)w * L + i) + 1];
            t[2 * (i * LANES + w)]     = ar * xr - ai * xi;
            t[2 * (i * LANES + w) + 1] = ar * xi + ai * xr;
        }
    }
    memset(t + 2 * (size_t)L * LANES, 0, 2 * (size_t)(M - L) * LANES * sizeof(double));
}

/* pre-chirp from lane-layout state into t (used by the batched chain: no transpose) */
static VEC512 void lane_prechirp_state(const fft1d_plan *p, const double *restrict sg,
                                       double *restrict t)
{
    const int L = p->L, M = p->M;
    const double *restrict a = p->a;
    for (int i = 0; i < L; ++i) {
        double ar = a[2 * i], ai = a[2 * i + 1];
        const double *restrict si = sg + 2 * (size_t)i * LANES;
        double *restrict ti = t + 2 * (size_t)i * LANES;
        for (int w = 0; w < 2 * LANES; w += 2) {
            double xr = si[w], xi = si[w + 1];
            ti[w] = ar * xr - ai * xi;
            ti[w + 1] = ar * xi + ai * xr;
        }
    }
    memset(t + 2 * (size_t)L * LANES, 0, 2 * (size_t)(M - L) * LANES * sizeof(double));
}

static VEC512 void lane_conjmult(int M, double *restrict u, const double *restrict bh)
{
    for (int i = 0; i < M; ++i) {                                  /* conj(U * bhat) */
        double br = bh[2 * i], bi = bh[2 * i + 1];
        double *restrict ui = u + 2 * (size_t)i * LANES;
        for (int w = 0; w < 2 * LANES; w += 2) {
            double ur = ui[w], uim = ui[w + 1];
            ui[w] = ur * br - uim * bi;
            ui[w + 1] = -(ur * bi + uim * br);
        }
    }
}

/* post-chirp, transpose out: y = a * conj(z) */
static VEC512 void lane_out_chirp(const fft1d_plan *p, const double *restrict z,
                                  double *restrict y)
{
    const int L = p->L;
    const double *restrict a = p->a;
    for (int i = 0; i < L; ++i) {
        double ar = a[2 * i], ai = a[2 * i + 1];
        for (int w = 0; w < LANES; ++w) {
            double zr = z[2 * (i * LANES + w)];
            double zi = z[2 * (i * LANES + w) + 1];
            y[2 * ((size_t)w * L + i)]     = ar * zr + ai * zi;
            y[2 * ((size_t)w * L + i) + 1] = ai * zr - ar * zi;
        }
    }
}

/* chain step tails: z + c -> map -> state, all in lane layout */
static VEC512 void lane_map_pow2(int n2LW, const double *restrict z,
                                 const double *restrict lcg, double *restrict sg)
{
    for (int i = 0; i < n2LW; i += 2) {
        double re = z[i] + lcg[i];
        double im = z[i + 1] + lcg[i + 1];
        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
        sg[i] = re * sc;
        sg[i + 1] = im * sc;
    }
}

static VEC512 void lane_postchirp_map(const fft1d_plan *p, const double *restrict z,
                                      const double *restrict lcg, double *restrict sg)
{
    const int L = p->L;
    const double *restrict a = p->a;
    for (int i = 0; i < L; ++i) {          /* post-chirp + add c + map, one pass */
        double ar = a[2 * i], ai = a[2 * i + 1];
        const double *restrict zi_ = z + 2 * (size_t)i * LANES;
        const double *restrict ci_ = lcg + 2 * (size_t)i * LANES;
        double *restrict si = sg + 2 * (size_t)i * LANES;
        for (int w = 0; w < 2 * LANES; w += 2) {
            double zr = zi_[w], zim = zi_[w + 1];
            double re = ar * zr + ai * zim + ci_[w];
            double im = ai * zr - ar * zim + ci_[w + 1];
            double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
            si[w] = re * sc;
            si[w + 1] = im * sc;
        }
    }
}

/* chirp lane core: t (lane layout, premultiplied, in l2) -> z in l1 */
static const double *bluestein_core_lane(fft1d_plan *p)
{
    fft_core(p, p->lst_s, 1, p->l2, p->l0, p->l1);                    /* lands in l0 */
    lane_conjmult(p->M, p->l0, p->bhat);
    return fft_core(p, p->lst_s, 1, p->l0, p->l1, p->l2);             /* lands in l1 */
}

void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    const int L = p->L, B = p->batch;
    const double *restrict X = (const double *)in;
    double *restrict Y = (double *)out;
    int b0 = 0;
    if (p->l0 && p->nstage > 0) {
        const int nblk = B / LANES;
        b0 = nblk * LANES;
        for (int g = 0; g < nblk; ++g) {
            const double *xg = X + 2 * (size_t)L * LANES * g;
            double *yg = Y + 2 * (size_t)L * LANES * g;
            if (p->is_pow2) {
                lane_in_pow2(p->M, xg, p->l2);
                const double *z = fft_core(p, p->lst_s, 1, p->l2, p->l0, p->l1);
                lane_out_pow2(p->M, z, yg);
            } else {
                lane_in_chirp(p, xg, p->l2);
                const double *z = bluestein_core_lane(p);
                lane_out_chirp(p, z, yg);
            }
        }
    }
    /* remainder (and the whole batch when B < LANES): per-vector */
    if (p->is_pow2) {
        if (p->nstage == 0) { memcpy(Y, X, 2 * (size_t)L * B * sizeof(double)); return; }
        for (int b = b0; b < B; ++b)
            fft_core(p, p->st_s, 0, X + 2 * (size_t)L * b, Y + 2 * (size_t)L * b, p->s0);
    } else {
        const double *restrict a = p->a;
        for (int b = b0; b < B; ++b) {
            const double *z = bluestein_core(p, X + 2 * (size_t)L * b);
            double *restrict y = Y + 2 * (size_t)L * b;
            for (int k = 0; k < L; ++k) {                      /* y = a * conj(z) */
                double ar = a[2 * k], ai = a[2 * k + 1];
                double zr = z[2 * k], zi = z[2 * k + 1];
                y[2 * k] = ar * zr + ai * zi;
                y[2 * k + 1] = ai * zr - ar * zi;
            }
        }
    }
}

/* fused m-step map chain: state <- (FFT(state) + c) / (1 + |FFT(state) + c|) */
void fft1d_chain(fft1d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    const int L = p->L, B = p->batch;
    const double *restrict C = (const double *)c;
    const double *restrict X0 = (const double *)x0;
    double *restrict OUT = (double *)final_out;

    if (p->l0 && p->nstage > 0 && B % LANES == 0) {
        /* Batched chain entirely in lane layout: transpose state and c in ONCE,
         * run every step fused (FFT lanes + post-chirp + map), transpose out ONCE. */
        const int nblk = B / LANES;
        double *restrict S = p->state;
        double *restrict LC = p->lc;
        for (int g = 0; g < nblk; ++g) {
            lane_in_pow2(L, X0 + 2 * (size_t)L * LANES * g, S + 2 * (size_t)L * LANES * g);
            lane_in_pow2(L, C + 2 * (size_t)L * LANES * g, LC + 2 * (size_t)L * LANES * g);
        }
        for (int step = 0; step < m; ++step) {
            for (int g = 0; g < nblk; ++g) {
                double *sg = S + 2 * (size_t)L * LANES * g;
                const double *lcg = LC + 2 * (size_t)L * LANES * g;
                if (p->is_pow2) {
                    const double *z = fft_core(p, p->lst_s, 1, sg, p->l0, p->l1);
                    lane_map_pow2(2 * L * LANES, z, lcg, sg);
                } else {
                    lane_prechirp_state(p, sg, p->l2);
                    const double *z = bluestein_core_lane(p);
                    lane_postchirp_map(p, z, lcg, sg);
                }
            }
        }
        for (int g = 0; g < nblk; ++g)
            lane_out_pow2(L, S + 2 * (size_t)L * LANES * g, OUT + 2 * (size_t)L * LANES * g);
        return;
    }

    /* scalar chain (B == 1, or batch not lane-divisible) */
    const double *cur = X0;
    for (int step = 0; step < m; ++step) {
        double *dst = (step == m - 1) ? OUT : p->state;
        if (p->is_pow2) {
            for (int b = 0; b < B; ++b) {
                const size_t off = 2 * (size_t)L * b;
                /* stage one consumes cur before dst (possibly == cur) is written */
                const double *z = (p->nstage == 0) ? cur + off
                                : fft_core(p, p->st_s, 0, cur + off, p->s0, p->s1);
                const double *restrict cc = C + off;
                double *restrict d = dst + off;
                for (int k = 0; k < L; ++k) {
                    double re = z[2 * k] + cc[2 * k];
                    double im = z[2 * k + 1] + cc[2 * k + 1];
                    double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                    d[2 * k] = re * sc;
                    d[2 * k + 1] = im * sc;
                }
            }
        } else {
            const double *restrict a = p->a;
            for (int b = 0; b < B; ++b) {
                const size_t off = 2 * (size_t)L * b;
                const double *z = bluestein_core(p, cur + off);
                const double *restrict cc = C + off;
                double *restrict d = dst + off;
                for (int k = 0; k < L; ++k) {              /* post-chirp + map, one pass */
                    double ar = a[2 * k], ai = a[2 * k + 1];
                    double zr = z[2 * k], zi = z[2 * k + 1];
                    double re = ar * zr + ai * zi + cc[2 * k];
                    double im = ai * zr - ar * zi + cc[2 * k + 1];
                    double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                    d[2 * k] = re * sc;
                    d[2 * k + 1] = im * sc;
                }
            }
        }
        cur = dst;
    }
}

void fft1d_destroy(fft1d_plan *p)
{
    if (!p) return;
    free(p->tw); free(p->a); free(p->bhat);
    free(p->s0); free(p->s1); free(p->s2);
    free(p->l0); free(p->l1); free(p->l2);
    free(p->state); free(p->lc);
    free(p);
}
