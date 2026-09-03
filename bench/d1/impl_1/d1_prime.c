/* d1_prime: small-prime 1D DFT (L = 7, 11, 13, 17, 31; graded: 13, 31).
 *
 * Algorithm: symmetric-pair dense DFT ("real-factor" form). For prime L with
 * h = (L-1)/2, fold the input into u_j = x_j + x_{L-j}, v_j = x_j - x_{L-j}
 * (j = 1..h). Then with theta = 2*pi*k*j/L,
 *     X[k]   = x_0 + sum_j ( cos(theta) * u_j )  - i * sum_j ( sin(theta) * v_j )
 *     X[L-k] = x_0 + sum_j ( cos(theta) * u_j )  + i * sum_j ( sin(theta) * v_j )
 *     X[0]   = x_0 + sum_j u_j
 * Every multiply is a REAL coefficient times a complex value: no complex-mult
 * shuffles, pure FMA, and the (L-1)^2 real-multiply count is 4x below the naive
 * dense matvec.
 *
 * All hot kernels are written with GNU vector extensions (v8 = 8 doubles = one
 * zmm) because GCC's auto-vectorizer turned the same code as scalar loops into
 * xmm shuffle soup (measured 5-10x slower). Two data shapes:
 *   - single transform: accumulators vectorized ACROSS k (zero-padded to hp);
 *   - batched chain:    vectorized ACROSS 8 batch lanes (split-complex SoA).
 *
 * Fused chain (fft1d_chain): owns the full m-step map chain
 *     state <- (FFT(state)+c) / (1 + |FFT(state)+c|)
 *   B = 1 : state kept in split re/im v8 rows across all m steps.
 *   B >= 2: 8 chains per lane-block; each block (state+c, ~8 KB at L=31) stays
 *           L1-resident for the WHOLE chain; transposed to SoA once, not per step.
 * The map's 1/(1+sqrt(re^2+im^2)) uses rsqrt14+2 Newton and rcp14+2 Newton on
 * AVX-512 (FMA ports instead of the unpipelined divider); ~1e-16 relative,
 * far inside the chain gate (verified: m=2 one-step gate passes at 8e-16).
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef __AVX512F__
#include <immintrin.h>
#endif
#include "../fft1d_api.h"

#define HMAX 16   /* h <= 15 (L = 31) */
#define LMAX 32

typedef double v8 __attribute__((vector_size(64), aligned(64)));

struct fft1d_plan {
    int L, batch, h, hp;
    double *tc, *ts;   /* [h][hp]: k-contiguous, zero-padded to hp; tc[j][k-1] = cos(2pi k(j+1)/L) */
    double *ck, *sk;   /* [h][h] : j-contiguous; ck[(k-1)*h + (j-1)]  = cos(2pi k j/L) */
};

const char *fft1d_name(void){ return "d1_prime"; }
const char *fft1d_description(void){
    return "symmetric-pair real-coeff dense prime DFT, zmm rows via vector ext; fused SoA map chain (rsqrt14+NR)";
}
int fft1d_supports(int L){ return L == 13 || L == 31 || L == 7 || L == 11 || L == 17; }

fft1d_plan *fft1d_create(int L, int batch){
    if(!fft1d_supports(L)) return NULL;
    fft1d_plan *p = calloc(1, sizeof *p); if(!p) return NULL;
    p->L = L; p->batch = batch;
    /* hp reserves one extra column beyond h: column h holds the k=0 twiddles
     * (cos=1, sin=0) so the fused chain gets X[0] = x0 + sum(u) for free from
     * the same FMA loop. kern1's epilogue never reads lanes >= h, so the extra
     * column is invisible to the plain transform. */
    int h = (L - 1) / 2, hp = (h + 1 + 7) & ~7;
    p->h = h; p->hp = hp;
    if (posix_memalign((void**)&p->tc, 64, (size_t)h*hp*sizeof(double)) ||
        posix_memalign((void**)&p->ts, 64, (size_t)h*hp*sizeof(double)) ||
        posix_memalign((void**)&p->ck, 64, (size_t)h*h*sizeof(double))  ||
        posix_memalign((void**)&p->sk, 64, (size_t)h*h*sizeof(double))) {
        fft1d_destroy(p); return NULL;
    }
    memset(p->tc, 0, (size_t)h*hp*sizeof(double));
    memset(p->ts, 0, (size_t)h*hp*sizeof(double));
    for (int k = 1; k <= h; k++)
        for (int j = 1; j <= h; j++) {
            int r = (k * j) % L;                 /* reduce the angle before sincos */
            int rr = (2 * r > L) ? r - L : r;    /* rr in (-L/2, L/2]: cos even, sin exact */
            double th = 2.0 * M_PI * (double)rr / (double)L;
            double c = cos(th), s = sin(th);
            p->tc[(size_t)(j-1)*p->hp + (k-1)] = c;
            p->ts[(size_t)(j-1)*p->hp + (k-1)] = s;
            p->ck[(size_t)(k-1)*h + (j-1)] = c;
            p->sk[(size_t)(k-1)*h + (j-1)] = s;
        }
    for (int j = 1; j <= h; j++) {
        p->tc[(size_t)(j-1)*p->hp + h] = 1.0;   /* k=0 column: cos = 1 */
        p->ts[(size_t)(j-1)*p->hp + h] = 0.0;   /* k=0 column: sin = 0 */
    }
    return p;
}

void fft1d_destroy(fft1d_plan *p){
    if(!p) return;
    free(p->tc); free(p->ts); free(p->ck); free(p->sk); free(p);
}

/* ---------------- single-transform kernel, k-vectorized over v8 rows ----------------
 * HP8 = hp/8 (1 for L=13, 2 for L=31) is a compile-time constant in the
 * specialized instantiations, so the r-loops fully unroll. */

static inline __attribute__((always_inline)) void
kern1(const double *restrict x, double *restrict y,
      const double *restrict tc, const double *restrict ts,
      const int L, const int h, const int HP8)
{
    double ur[HMAX], ui[HMAX], vr[HMAX], vi[HMAX];
    const double x0r = x[0], x0i = x[1];
    double s0r = x0r, s0i = x0i;
    for (int j = 1; j <= h; j++) {
        double ar = x[2*j],       ai = x[2*j+1];
        double br = x[2*(L-j)],   bi = x[2*(L-j)+1];
        ur[j-1] = ar + br;  ui[j-1] = ai + bi;
        vr[j-1] = ar - br;  vi[j-1] = ai - bi;
        s0r += ur[j-1];     s0i += ui[j-1];
    }
    v8 Pr[2], Pi[2], Rr[2], Si[2];
    for (int r = 0; r < HP8; r++){ Pr[r] = (v8){0}; Pi[r] = (v8){0}; Rr[r] = (v8){0}; Si[r] = (v8){0}; }
    for (int j = 0; j < h; j++) {
        const v8 *restrict cj = (const v8 *)(tc + (size_t)j*8*HP8);
        const v8 *restrict sj = (const v8 *)(ts + (size_t)j*8*HP8);
        const double a = ur[j], b = ui[j], e = vr[j], f = vi[j];
        for (int r = 0; r < HP8; r++) {
            Pr[r] += cj[r]*a;  Pi[r] += cj[r]*b;
            Rr[r] += sj[r]*e;  Si[r] += sj[r]*f;
        }
    }
    y[0] = s0r; y[1] = s0i;
    for (int k = 1; k <= h; k++) {
        const int r = (k-1) >> 3, l = (k-1) & 7;
        double pre = x0r + Pr[r][l], pim = x0i + Pi[r][l];
        double si = Si[r][l], rr = Rr[r][l];
        y[2*k]        = pre + si;
        y[2*k+1]      = pim - rr;
        y[2*(L-k)]    = pre - si;
        y[2*(L-k)+1]  = pim + rr;
    }
}

static void exec_gen (const double *x, double *y, const fft1d_plan *p){ kern1(x,y,p->tc,p->ts,p->L,p->h,p->hp/8); }

#ifdef __AVX512F__
/* Intrinsic execute kernels: the scalar de/re-interleave around kern1 costs more
 * than the FMA core at these sizes, so load/deinterleave/fold and the
 * interleaved store are done with vpermt2pd index permutes instead. The FMA
 * core is the same k-vectorized symmetric-pair loop (X0 via the k=0 column). */

static inline __attribute__((always_inline)) __m512d bcast0(__m512d v){
    return _mm512_broadcastsd_pd(_mm512_castpd512_pd128(v));
}

static void exec13_avx(const double *restrict x, double *restrict y, const fft1d_plan *restrict p)
{
    /* fused deinterleave+fold: forward row x1..x6 and reverse row x12..x7 come
     * straight out of the interleaved loads with one permute each per re/im */
    const __m512i FRE = _mm512_setr_epi64(2,4,6,8,10,12,12,12);    /* x1..x6 re of (z0,z1) */
    const __m512i FIM = _mm512_setr_epi64(3,5,7,9,11,13,13,13);
    const __m512i RRE = _mm512_setr_epi64(14,12,10,8,2,0,0,0);     /* x12..x7 re of (z2,z3) */
    const __m512i RIM = _mm512_setr_epi64(15,13,11,9,3,1,1,1);
    __m512d z0 = _mm512_loadu_pd(x),      z1 = _mm512_loadu_pd(x + 8);
    __m512d z2 = _mm512_loadu_pd(x + 14), z3 = _mm512_loadu_pd(x + 18);
    __m512d fr = _mm512_permutex2var_pd(z0, FRE, z1), fi = _mm512_permutex2var_pd(z0, FIM, z1);
    __m512d rr = _mm512_permutex2var_pd(z2, RRE, z3), ri = _mm512_permutex2var_pd(z2, RIM, z3);
    double ur[8] __attribute__((aligned(64))), ui[8] __attribute__((aligned(64)));
    double vr[8] __attribute__((aligned(64))), vi[8] __attribute__((aligned(64)));
    _mm512_store_pd(ur, _mm512_add_pd(fr, rr));
    _mm512_store_pd(ui, _mm512_add_pd(fi, ri));
    _mm512_store_pd(vr, _mm512_sub_pd(fr, rr));
    _mm512_store_pd(vi, _mm512_sub_pd(fi, ri));
    __m512d x0r = _mm512_set1_pd(x[0]), x0i = _mm512_set1_pd(x[1]);
    /* two accumulator sets halve the FMA dependency depth (6 -> 3) */
    __m512d Pr = _mm512_setzero_pd(), Pi = Pr, Rr = Pr, Si = Pr;
    __m512d Pr2 = Pr, Pi2 = Pr, Rr2 = Pr, Si2 = Pr;
    const double *restrict tc = p->tc, *restrict ts = p->ts;
    for (int j = 0; j < 3; j++) {
        __m512d cj = _mm512_load_pd(tc + 8*j), sj = _mm512_load_pd(ts + 8*j);
        __m512d ck = _mm512_load_pd(tc + 8*(j+3)), sk = _mm512_load_pd(ts + 8*(j+3));
        Pr  = _mm512_fmadd_pd(cj, _mm512_set1_pd(ur[j]), Pr);
        Pi  = _mm512_fmadd_pd(cj, _mm512_set1_pd(ui[j]), Pi);
        Rr  = _mm512_fmadd_pd(sj, _mm512_set1_pd(vr[j]), Rr);
        Si  = _mm512_fmadd_pd(sj, _mm512_set1_pd(vi[j]), Si);
        Pr2 = _mm512_fmadd_pd(ck, _mm512_set1_pd(ur[j+3]), Pr2);
        Pi2 = _mm512_fmadd_pd(ck, _mm512_set1_pd(ui[j+3]), Pi2);
        Rr2 = _mm512_fmadd_pd(sk, _mm512_set1_pd(vr[j+3]), Rr2);
        Si2 = _mm512_fmadd_pd(sk, _mm512_set1_pd(vi[j+3]), Si2);
    }
    Pr = _mm512_add_pd(Pr, Pr2); Pi = _mm512_add_pd(Pi, Pi2);
    Rr = _mm512_add_pd(Rr, Rr2); Si = _mm512_add_pd(Si, Si2);
    __m512d pre = _mm512_add_pd(x0r, Pr), pim = _mm512_add_pd(x0i, Pi);
    __m512d nar = _mm512_add_pd(pre, Si), nai = _mm512_sub_pd(pim, Rr);   /* X[1..6], X0 in lane 6 */
    __m512d nbr = _mm512_sub_pd(pre, Si), nbi = _mm512_add_pd(pim, Rr);   /* X[12..7] in lanes 0..5 */
    /* natural-order re/im rows, then interleave */
    const __m512i Y0IDX = _mm512_setr_epi64(6,0,1,2,3,4,5,13);   /* [X0,X1..X6,X7]: X7 = nbr lane 5 */
    const __m512i Y1IDX = _mm512_setr_epi64(4,3,2,1,0,0,0,0);    /* [X8..X12] = nbr lanes 4..0 */
    __m512d Y0r = _mm512_permutex2var_pd(nar, Y0IDX, nbr), Y0i = _mm512_permutex2var_pd(nai, Y0IDX, nbi);
    __m512d Y1r = _mm512_permutexvar_pd(Y1IDX, nbr),       Y1i = _mm512_permutexvar_pd(Y1IDX, nbi);
    const __m512i ILO = _mm512_setr_epi64(0,8,1,9,2,10,3,11);
    const __m512i IHI = _mm512_setr_epi64(4,12,5,13,6,14,7,15);
    _mm512_storeu_pd(y,      _mm512_permutex2var_pd(Y0r, ILO, Y0i));
    _mm512_storeu_pd(y + 8,  _mm512_permutex2var_pd(Y0r, IHI, Y0i));
    _mm512_storeu_pd(y + 16, _mm512_permutex2var_pd(Y1r, ILO, Y1i));
    _mm512_mask_storeu_pd(y + 24, 0x03, _mm512_permutex2var_pd(Y1r, IHI, Y1i));
}

static void exec31_avx(const double *restrict x, double *restrict y, const fft1d_plan *restrict p)
{
    const __m512i EVEN = _mm512_setr_epi64(0,2,4,6,8,10,12,14);
    const __m512i ODD  = _mm512_setr_epi64(1,3,5,7,9,11,13,15);
    const __m512i T3E  = _mm512_setr_epi64(0,2,4,6,8,10,12,12);   /* x24..x30 from (z6,z7) */
    const __m512i T3O  = _mm512_setr_epi64(1,3,5,7,9,11,13,13);
    __m512d z0 = _mm512_loadu_pd(x),      z1 = _mm512_loadu_pd(x + 8);
    __m512d z2 = _mm512_loadu_pd(x + 16), z3 = _mm512_loadu_pd(x + 24);
    __m512d z4 = _mm512_loadu_pd(x + 32), z5 = _mm512_loadu_pd(x + 40);
    __m512d z6 = _mm512_loadu_pd(x + 48), z7 = _mm512_maskz_loadu_pd(0x3F, x + 56);
    __m512d xr0 = _mm512_permutex2var_pd(z0, EVEN, z1), xi0 = _mm512_permutex2var_pd(z0, ODD, z1);
    __m512d xr1 = _mm512_permutex2var_pd(z2, EVEN, z3), xi1 = _mm512_permutex2var_pd(z2, ODD, z3);
    __m512d xr2 = _mm512_permutex2var_pd(z4, EVEN, z5), xi2 = _mm512_permutex2var_pd(z4, ODD, z5);
    __m512d xr3 = _mm512_permutex2var_pd(z6, T3E,  z7), xi3 = _mm512_permutex2var_pd(z6, T3O, z7);
    /* forward rows x1..x8, x9..x15; reverse rows x30..x23, x22..x16 */
    const __m512i F0 = _mm512_setr_epi64(1,2,3,4,5,6,7,8);
    const __m512i F1 = _mm512_setr_epi64(1,2,3,4,5,6,7,7);
    const __m512i R0 = _mm512_setr_epi64(6,5,4,3,2,1,0,15);       /* (xr3,xr2): x30..x24, x23 */
    const __m512i R1 = _mm512_setr_epi64(6,5,4,3,2,1,0,0);        /* xr2: x22..x16 */
    __m512d f0r = _mm512_permutex2var_pd(xr0, F0, xr1), f0i = _mm512_permutex2var_pd(xi0, F0, xi1);
    __m512d f1r = _mm512_permutexvar_pd(F1, xr1),       f1i = _mm512_permutexvar_pd(F1, xi1);
    __m512d r0r = _mm512_permutex2var_pd(xr3, R0, xr2), r0i = _mm512_permutex2var_pd(xi3, R0, xi2);
    __m512d r1r = _mm512_permutexvar_pd(R1, xr2),       r1i = _mm512_permutexvar_pd(R1, xi2);
    double ur[16] __attribute__((aligned(64))), ui[16] __attribute__((aligned(64)));
    double vr[16] __attribute__((aligned(64))), vi[16] __attribute__((aligned(64)));
    _mm512_store_pd(ur,     _mm512_add_pd(f0r, r0r));  _mm512_store_pd(ur + 8, _mm512_add_pd(f1r, r1r));
    _mm512_store_pd(ui,     _mm512_add_pd(f0i, r0i));  _mm512_store_pd(ui + 8, _mm512_add_pd(f1i, r1i));
    _mm512_store_pd(vr,     _mm512_sub_pd(f0r, r0r));  _mm512_store_pd(vr + 8, _mm512_sub_pd(f1r, r1r));
    _mm512_store_pd(vi,     _mm512_sub_pd(f0i, r0i));  _mm512_store_pd(vi + 8, _mm512_sub_pd(f1i, r1i));
    __m512d x0r = bcast0(xr0), x0i = bcast0(xi0);
    /* second accumulator set on the odd j halves the FMA dependency depth (15 -> 8) */
    __m512d Pr0 = _mm512_setzero_pd(), Pi0 = Pr0, Rr0 = Pr0, Si0 = Pr0;
    __m512d Pr1 = Pr0, Pi1 = Pr0, Rr1 = Pr0, Si1 = Pr0;
    __m512d Qr0 = Pr0, Qi0 = Pr0, Tr0 = Pr0, Ui0 = Pr0;
    __m512d Qr1 = Pr0, Qi1 = Pr0, Tr1 = Pr0, Ui1 = Pr0;
    const double *restrict tc = p->tc, *restrict ts = p->ts;
    for (int j = 0; j < 14; j += 2) {
        __m512d c0 = _mm512_load_pd(tc + 16*j),     s0 = _mm512_load_pd(ts + 16*j);
        __m512d c1 = _mm512_load_pd(tc + 16*j + 8), s1 = _mm512_load_pd(ts + 16*j + 8);
        __m512d a = _mm512_set1_pd(ur[j]), b = _mm512_set1_pd(ui[j]);
        __m512d e = _mm512_set1_pd(vr[j]), f = _mm512_set1_pd(vi[j]);
        Pr0 = _mm512_fmadd_pd(c0, a, Pr0);  Pr1 = _mm512_fmadd_pd(c1, a, Pr1);
        Pi0 = _mm512_fmadd_pd(c0, b, Pi0);  Pi1 = _mm512_fmadd_pd(c1, b, Pi1);
        Rr0 = _mm512_fmadd_pd(s0, e, Rr0);  Rr1 = _mm512_fmadd_pd(s1, e, Rr1);
        Si0 = _mm512_fmadd_pd(s0, f, Si0);  Si1 = _mm512_fmadd_pd(s1, f, Si1);
        __m512d d0 = _mm512_load_pd(tc + 16*(j+1)),     t0 = _mm512_load_pd(ts + 16*(j+1));
        __m512d d1 = _mm512_load_pd(tc + 16*(j+1) + 8), t1 = _mm512_load_pd(ts + 16*(j+1) + 8);
        __m512d a2 = _mm512_set1_pd(ur[j+1]), b2 = _mm512_set1_pd(ui[j+1]);
        __m512d e2 = _mm512_set1_pd(vr[j+1]), f2 = _mm512_set1_pd(vi[j+1]);
        Qr0 = _mm512_fmadd_pd(d0, a2, Qr0);  Qr1 = _mm512_fmadd_pd(d1, a2, Qr1);
        Qi0 = _mm512_fmadd_pd(d0, b2, Qi0);  Qi1 = _mm512_fmadd_pd(d1, b2, Qi1);
        Tr0 = _mm512_fmadd_pd(t0, e2, Tr0);  Tr1 = _mm512_fmadd_pd(t1, e2, Tr1);
        Ui0 = _mm512_fmadd_pd(t0, f2, Ui0);  Ui1 = _mm512_fmadd_pd(t1, f2, Ui1);
    }
    {   /* j = 14 tail */
        __m512d c0 = _mm512_load_pd(tc + 16*14),     s0 = _mm512_load_pd(ts + 16*14);
        __m512d c1 = _mm512_load_pd(tc + 16*14 + 8), s1 = _mm512_load_pd(ts + 16*14 + 8);
        __m512d a = _mm512_set1_pd(ur[14]), b = _mm512_set1_pd(ui[14]);
        __m512d e = _mm512_set1_pd(vr[14]), f = _mm512_set1_pd(vi[14]);
        Pr0 = _mm512_fmadd_pd(c0, a, Pr0);  Pr1 = _mm512_fmadd_pd(c1, a, Pr1);
        Pi0 = _mm512_fmadd_pd(c0, b, Pi0);  Pi1 = _mm512_fmadd_pd(c1, b, Pi1);
        Rr0 = _mm512_fmadd_pd(s0, e, Rr0);  Rr1 = _mm512_fmadd_pd(s1, e, Rr1);
        Si0 = _mm512_fmadd_pd(s0, f, Si0);  Si1 = _mm512_fmadd_pd(s1, f, Si1);
    }
    Pr0 = _mm512_add_pd(Pr0, Qr0);  Pr1 = _mm512_add_pd(Pr1, Qr1);
    Pi0 = _mm512_add_pd(Pi0, Qi0);  Pi1 = _mm512_add_pd(Pi1, Qi1);
    Rr0 = _mm512_add_pd(Rr0, Tr0);  Rr1 = _mm512_add_pd(Rr1, Tr1);
    Si0 = _mm512_add_pd(Si0, Ui0);  Si1 = _mm512_add_pd(Si1, Ui1);
    __m512d pre0 = _mm512_add_pd(x0r, Pr0), pim0 = _mm512_add_pd(x0i, Pi0);
    __m512d pre1 = _mm512_add_pd(x0r, Pr1), pim1 = _mm512_add_pd(x0i, Pi1);
    __m512d na0r = _mm512_add_pd(pre0, Si0), na0i = _mm512_sub_pd(pim0, Rr0);  /* X1..X8 */
    __m512d na1r = _mm512_add_pd(pre1, Si1), na1i = _mm512_sub_pd(pim1, Rr1);  /* X9..X15, X0 lane 7 */
    __m512d nb0r = _mm512_sub_pd(pre0, Si0), nb0i = _mm512_add_pd(pim0, Rr0);  /* X30..X23 */
    __m512d nb1r = _mm512_sub_pd(pre1, Si1), nb1i = _mm512_add_pd(pim1, Rr1);  /* X22..X16 lanes 0..6 */
    /* natural-order rows: Y0=[X0..X7] Y1=[X8..X15] Y2=[X16..X23] Y3=[X24..X30] */
    const __m512i Y01 = _mm512_setr_epi64(7,8,9,10,11,12,13,14);  /* (na1,na0): X0 then X1..X7 / (na0,na1): X8 then X9..X15 */
    const __m512i Y2I = _mm512_setr_epi64(6,5,4,3,2,1,0,15);      /* (nb1,nb0): X16..X22, X23 */
    const __m512i Y3I = _mm512_setr_epi64(6,5,4,3,2,1,0,0);       /* nb0: X24..X30 */
    __m512d Y0r = _mm512_permutex2var_pd(na1r, Y01, na0r), Y0i = _mm512_permutex2var_pd(na1i, Y01, na0i);
    __m512d Y1r = _mm512_permutex2var_pd(na0r, Y01, na1r), Y1i = _mm512_permutex2var_pd(na0i, Y01, na1i);
    __m512d Y2r = _mm512_permutex2var_pd(nb1r, Y2I, nb0r), Y2i = _mm512_permutex2var_pd(nb1i, Y2I, nb0i);
    __m512d Y3r = _mm512_permutexvar_pd(Y3I, nb0r),        Y3i = _mm512_permutexvar_pd(Y3I, nb0i);
    const __m512i ILO = _mm512_setr_epi64(0,8,1,9,2,10,3,11);
    const __m512i IHI = _mm512_setr_epi64(4,12,5,13,6,14,7,15);
    _mm512_storeu_pd(y,      _mm512_permutex2var_pd(Y0r, ILO, Y0i));
    _mm512_storeu_pd(y + 8,  _mm512_permutex2var_pd(Y0r, IHI, Y0i));
    _mm512_storeu_pd(y + 16, _mm512_permutex2var_pd(Y1r, ILO, Y1i));
    _mm512_storeu_pd(y + 24, _mm512_permutex2var_pd(Y1r, IHI, Y1i));
    _mm512_storeu_pd(y + 32, _mm512_permutex2var_pd(Y2r, ILO, Y2i));
    _mm512_storeu_pd(y + 40, _mm512_permutex2var_pd(Y2r, IHI, Y2i));
    _mm512_storeu_pd(y + 48, _mm512_permutex2var_pd(Y3r, ILO, Y3i));
    _mm512_mask_storeu_pd(y + 56, 0x3F, _mm512_permutex2var_pd(Y3r, IHI, Y3i));
}
#endif /* __AVX512F__ */

void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out){
    const int L = p->L, B = p->batch;
    const double *x = (const double *)in;
    double *y = (double *)out;
#ifdef __AVX512F__
    if (L == 13)      { for (int b = 0; b < B; b++) exec13_avx(x + 26*(size_t)b, y + 26*(size_t)b, p); return; }
    else if (L == 31) { for (int b = 0; b < B; b++) exec31_avx(x + 62*(size_t)b, y + 62*(size_t)b, p); return; }
#endif
    for (int b = 0; b < B; b++) exec_gen(x + 2*(size_t)b*L, y + 2*(size_t)b*L, p);
}

/* ---------------- the map:  z -> z / (1 + |z|)  ---------------- */

#ifdef __AVX512F__
/* scale q = 1/(1+sqrt(tr^2+ti^2)) without touching the divider port:
 * rsqrt14 + 2 Newton (error ~1e-16), then rcp14 + 2 Newton. m2 clamped away
 * from 0 so rsqrt's inf never meets a 0 multiply (the clamp shifts a |z| of
 * <1e-150 by an invisible absolute amount). */
static inline __attribute__((always_inline)) v8 map_sc8(v8 trv, v8 tiv){
    __m512d tr = (__m512d)trv, ti = (__m512d)tiv;
    const __m512d half = _mm512_set1_pd(0.5), three_half = _mm512_set1_pd(1.5);
    const __m512d one  = _mm512_set1_pd(1.0), two = _mm512_set1_pd(2.0);
    __m512d m2 = _mm512_fmadd_pd(tr, tr, _mm512_mul_pd(ti, ti));
    m2 = _mm512_max_pd(m2, _mm512_set1_pd(1e-300));
    __m512d r = _mm512_rsqrt14_pd(m2);
    __m512d hm = _mm512_mul_pd(m2, half);
    __m512d t = _mm512_mul_pd(r, r);
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(hm, t, three_half));
    t = _mm512_mul_pd(r, r);
    r = _mm512_mul_pd(r, _mm512_fnmadd_pd(hm, t, three_half));
    __m512d d = _mm512_fmadd_pd(m2, r, one);          /* 1 + sqrt(m2) */
    __m512d q = _mm512_rcp14_pd(d);
    q = _mm512_mul_pd(q, _mm512_fnmadd_pd(d, q, two));
    q = _mm512_mul_pd(q, _mm512_fnmadd_pd(d, q, two));
    return (v8)q;
}
#else
static inline v8 map_sc8(v8 tr, v8 ti){
    v8 q;
    for (int l = 0; l < 8; l++) q[l] = 1.0 / (1.0 + sqrt(tr[l]*tr[l] + ti[l]*ti[l]));
    return q;
}
#endif

/* ---------------- fused chain, B = 1: state lives in A/B row representation ----------------
 * The state is held ACROSS steps as the mapped output rows themselves:
 *   A rows: lane k-1 = state[k] (k=1..h), lane h = state[0]
 *   B rows: lane k-1 = state[L-k]
 * Then next step's fold is just u = A + B, v = A - B on whole registers -- the
 * index reversal x[L-j] is implicit in the A/B pairing, so there is no scatter,
 * no gather, and no reversal permute anywhere in the loop. X[0] = x0 + sum(u)
 * falls out of the same FMA loop via the k=0 table column, and rides lane h of
 * the A rows through the vector map (no scalar sqrt on the critical path). */

static inline __attribute__((always_inline)) void
chain1_body(const int L, const int h, const int HP8, const fft1d_plan *restrict p,
            const double *restrict x0, const double *restrict c,
            double *restrict out, int m)
{
    const int r_h = h >> 3, l_h = h & 7;
    v8 Ar[2], Ai[2], Br[2], Bi[2];
    v8 cAr[2], cAi[2], cBr[2], cBi[2];       /* c rearranged per output pair, padded 0 */
    for (int r = 0; r < HP8; r++){
        Ar[r] = (v8){0}; Ai[r] = (v8){0}; Br[r] = (v8){0}; Bi[r] = (v8){0};
        cAr[r] = (v8){0}; cAi[r] = (v8){0}; cBr[r] = (v8){0}; cBi[r] = (v8){0};
    }
    for (int k = 1; k <= h; k++) {
        const int r = (k-1) >> 3, l = (k-1) & 7;
        Ar[r][l]  = x0[2*k];       Ai[r][l]  = x0[2*k+1];
        Br[r][l]  = x0[2*(L-k)];   Bi[r][l]  = x0[2*(L-k)+1];
        cAr[r][l] = c[2*k];        cAi[r][l] = c[2*k+1];
        cBr[r][l] = c[2*(L-k)];    cBi[r][l] = c[2*(L-k)+1];
    }
    Ar[r_h][l_h]  = x0[0];  Ai[r_h][l_h]  = x0[1];
    cAr[r_h][l_h] = c[0];   cAi[r_h][l_h] = c[1];
    const double *restrict tc = p->tc, *restrict ts = p->ts;

    for (int step = 0; step < m; step++) {
        const double x0r = Ar[r_h][l_h], x0i = Ai[r_h][l_h];
        double ur[HMAX] __attribute__((aligned(64))), ui[HMAX] __attribute__((aligned(64)));
        double vr[HMAX] __attribute__((aligned(64))), vi[HMAX] __attribute__((aligned(64)));
        for (int r = 0; r < HP8; r++) {
            *(v8 *)(ur + 8*r) = Ar[r] + Br[r];
            *(v8 *)(ui + 8*r) = Ai[r] + Bi[r];
            *(v8 *)(vr + 8*r) = Ar[r] - Br[r];
            *(v8 *)(vi + 8*r) = Ai[r] - Bi[r];
        }
        /* two accumulator sets halve the FMA dependency depth; this loop is the
         * serial critical path of the whole chained cell at B=1 */
        v8 Pr[2], Pi[2], Rr[2], Si[2];
        v8 Pr2[2], Pi2[2], Rr2[2], Si2[2];
        for (int r = 0; r < HP8; r++){
            Pr[r] = (v8){0}; Pi[r] = (v8){0}; Rr[r] = (v8){0}; Si[r] = (v8){0};
            Pr2[r] = (v8){0}; Pi2[r] = (v8){0}; Rr2[r] = (v8){0}; Si2[r] = (v8){0};
        }
        /* split only when one row set fits: at HP8=2 the extra 8 accumulators
         * spill (16 state/c rows are already live) and cost more than the
         * shorter chain saves (measured 0.050 -> 0.057 us at L=31) */
        const int hh = (HP8 == 1) ? (h >> 1) : h;
        for (int j = 0; j < hh; j++) {
            const v8 *restrict cj = (const v8 *)(tc + (size_t)j*8*HP8);
            const v8 *restrict sj = (const v8 *)(ts + (size_t)j*8*HP8);
            const double a = ur[j], b = ui[j], e = vr[j], f = vi[j];
            for (int r = 0; r < HP8; r++) {
                Pr[r] += cj[r]*a;  Pi[r] += cj[r]*b;
                Rr[r] += sj[r]*e;  Si[r] += sj[r]*f;
            }
        }
        for (int j = hh; j < h; j++) {
            const v8 *restrict cj = (const v8 *)(tc + (size_t)j*8*HP8);
            const v8 *restrict sj = (const v8 *)(ts + (size_t)j*8*HP8);
            const double a = ur[j], b = ui[j], e = vr[j], f = vi[j];
            for (int r = 0; r < HP8; r++) {
                Pr2[r] += cj[r]*a;  Pi2[r] += cj[r]*b;
                Rr2[r] += sj[r]*e;  Si2[r] += sj[r]*f;
            }
        }
        for (int r = 0; r < HP8; r++) {
            Pr[r] += Pr2[r]; Pi[r] += Pi2[r]; Rr[r] += Rr2[r]; Si[r] += Si2[r];
        }
        for (int r = 0; r < HP8; r++) {
            v8 pre = x0r + Pr[r], pim = x0i + Pi[r];
            v8 nar = pre + Si[r] + cAr[r];
            v8 nai = pim - Rr[r] + cAi[r];
            v8 nbr = pre - Si[r] + cBr[r];
            v8 nbi = pim + Rr[r] + cBi[r];
            v8 qa = map_sc8(nar, nai);
            Ar[r] = nar * qa;  Ai[r] = nai * qa;
            v8 qb = map_sc8(nbr, nbi);
            Br[r] = nbr * qb;  Bi[r] = nbi * qb;
        }
    }
    out[0] = Ar[r_h][l_h]; out[1] = Ai[r_h][l_h];
    for (int k = 1; k <= h; k++) {
        const int r = (k-1) >> 3, l = (k-1) & 7;
        out[2*k]       = Ar[r][l];  out[2*k+1]       = Ai[r][l];
        out[2*(L-k)]   = Br[r][l];  out[2*(L-k)+1]   = Bi[r][l];
    }
}

static void chain1_L13(const fft1d_plan *p, const double *x0, const double *c, double *out, int m){
    chain1_body(13, 6, 1, p, x0, c, out, m);
}
static void chain1_L31(const fft1d_plan *p, const double *x0, const double *c, double *out, int m){
    chain1_body(31, 15, 2, p, x0, c, out, m);
}
static void chain1_gen(const fft1d_plan *p, const double *x0, const double *c, double *out, int m){
    chain1_body(p->L, p->h, p->hp/8, p, x0, c, out, m);
}

/* ---------------- fused chain, batched: 8 chains per lane-block ----------------
 * Each block of 8 batch lanes is transposed to split-complex v8 rows ONCE, runs
 * all m steps L1-resident, and is written back once. Lanes beyond the batch are
 * clamped to the last valid lane (computed redundantly, never stored). */

static inline __attribute__((always_inline)) void
chainblk_body(const int L, const int h, const fft1d_plan *restrict p,
              const double *restrict x0, const double *restrict c,
              double *restrict out, int m, int batch)
{
    const double *restrict ck = p->ck, *restrict sk = p->sk;
    for (int b0 = 0; b0 < batch; b0 += 8) {
        const int lanes = (batch - b0 < 8) ? batch - b0 : 8;
        v8 xr[LMAX], xi[LMAX], cr[LMAX], ci[LMAX];
        for (int l = 0; l < 8; l++) {
            const size_t bb = (size_t)(b0 + (l < lanes ? l : lanes - 1)) * L;
            for (int j = 0; j < L; j++) {
                xr[j][l] = x0[2*(bb+j)];  xi[j][l] = x0[2*(bb+j)+1];
                cr[j][l] = c [2*(bb+j)];  ci[j][l] = c [2*(bb+j)+1];
            }
        }
        for (int step = 0; step < m; step++) {
            v8 ur[HMAX], ui[HMAX], vr[HMAX], vi[HMAX];
            const v8 x0r = xr[0], x0i = xi[0];
            v8 s0r = x0r, s0i = x0i;
            for (int j = 1; j <= h; j++) {
                v8 ar = xr[j],   ai = xi[j];
                v8 br = xr[L-j], bi = xi[L-j];
                ur[j-1] = ar + br;  ui[j-1] = ai + bi;
                vr[j-1] = ar - br;  vi[j-1] = ai - bi;
                s0r += ur[j-1];     s0i += ui[j-1];
            }
            s0r += cr[0]; s0i += ci[0];
            v8 q0 = map_sc8(s0r, s0i);
            xr[0] = s0r * q0; xi[0] = s0i * q0;
            /* k blocked by 3: each u/v row load feeds 12 FMAs instead of 4
             * (the plain k-loop was load-port bound). h=6 and h=15 divide by 3,
             * so the remainder loop below is dead code in the graded sizes. */
            int k = 1;
            for (; k + 2 <= h; k += 3) {
                v8 Pr0=(v8){0}, Pi0=(v8){0}, Rr0=(v8){0}, Si0=(v8){0};
                v8 Pr1=(v8){0}, Pi1=(v8){0}, Rr1=(v8){0}, Si1=(v8){0};
                v8 Pr2=(v8){0}, Pi2=(v8){0}, Rr2=(v8){0}, Si2=(v8){0};
                const double *restrict ck0 = ck + (size_t)(k-1)*h, *restrict sk0 = sk + (size_t)(k-1)*h;
                const double *restrict ck1 = ck0 + h, *restrict sk1 = sk0 + h;
                const double *restrict ck2 = ck1 + h, *restrict sk2 = sk1 + h;
                for (int j = 0; j < h; j++) {
                    const v8 uj = ur[j], wj = ui[j], ej = vr[j], fj = vi[j];
                    Pr0 += ck0[j]*uj;  Pi0 += ck0[j]*wj;  Rr0 += sk0[j]*ej;  Si0 += sk0[j]*fj;
                    Pr1 += ck1[j]*uj;  Pi1 += ck1[j]*wj;  Rr1 += sk1[j]*ej;  Si1 += sk1[j]*fj;
                    Pr2 += ck2[j]*uj;  Pi2 += ck2[j]*wj;  Rr2 += sk2[j]*ej;  Si2 += sk2[j]*fj;
                }
                for (int t = 0; t < 3; t++) {
                    const int kk = k + t;
                    v8 Pr = t==0?Pr0:(t==1?Pr1:Pr2), Pi = t==0?Pi0:(t==1?Pi1:Pi2);
                    v8 Rr = t==0?Rr0:(t==1?Rr1:Rr2), Si = t==0?Si0:(t==1?Si1:Si2);
                    v8 pre = x0r + Pr, pim = x0i + Pi;
                    v8 Arow = pre + Si + cr[kk],   Airow = pim - Rr + ci[kk];
                    v8 Brow = pre - Si + cr[L-kk], Birow = pim + Rr + ci[L-kk];
                    v8 qa = map_sc8(Arow, Airow);
                    xr[kk]   = Arow * qa;  xi[kk]   = Airow * qa;
                    v8 qb = map_sc8(Brow, Birow);
                    xr[L-kk] = Brow * qb;  xi[L-kk] = Birow * qb;
                }
            }
            for (; k <= h; k++) {
                v8 Pr = (v8){0}, Pi = (v8){0}, Rr = (v8){0}, Si = (v8){0};
                const double *restrict ckr = ck + (size_t)(k-1)*h;
                const double *restrict skr = sk + (size_t)(k-1)*h;
                for (int j = 0; j < h; j++) {
                    const double cc = ckr[j], ss = skr[j];
                    Pr += cc*ur[j];  Pi += cc*ui[j];
                    Rr += ss*vr[j];  Si += ss*vi[j];
                }
                v8 pre = x0r + Pr, pim = x0i + Pi;
                v8 Arow = pre + Si + cr[k],   Airow = pim - Rr + ci[k];
                v8 Brow = pre - Si + cr[L-k], Birow = pim + Rr + ci[L-k];
                v8 qa = map_sc8(Arow, Airow);
                xr[k]   = Arow * qa;  xi[k]   = Airow * qa;
                v8 qb = map_sc8(Brow, Birow);
                xr[L-k] = Brow * qb;  xi[L-k] = Birow * qb;
            }
        }
        for (int l = 0; l < lanes; l++) {
            const size_t bb = (size_t)(b0 + l) * L;
            for (int j = 0; j < L; j++){ out[2*(bb+j)] = xr[j][l]; out[2*(bb+j)+1] = xi[j][l]; }
        }
    }
}

static void chainblk_L13(const fft1d_plan *p, const double *x0, const double *c, double *out, int m, int batch){
    chainblk_body(13, 6, p, x0, c, out, m, batch);
}
static void chainblk_L31(const fft1d_plan *p, const double *x0, const double *c, double *out, int m, int batch){
    chainblk_body(31, 15, p, x0, c, out, m, batch);
}
static void chainblk_gen(const fft1d_plan *p, const double *x0, const double *c, double *out, int m, int batch){
    chainblk_body(p->L, p->h, p, x0, c, out, m, batch);
}

void fft1d_chain(fft1d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    const double *xd = (const double *)x0, *cd = (const double *)c;
    double *od = (double *)final_out;
    if (p->batch == 1) {
        if (p->L == 13)      chain1_L13(p, xd, cd, od, m);
        else if (p->L == 31) chain1_L31(p, xd, cd, od, m);
        else                 chain1_gen(p, xd, cd, od, m);
    } else {
        if (p->L == 13)      chainblk_L13(p, xd, cd, od, m, p->batch);
        else if (p->L == 31) chainblk_L31(p, xd, cd, od, m, p->batch);
        else                 chainblk_gen(p, xd, cd, od, m, p->batch);
    }
}
