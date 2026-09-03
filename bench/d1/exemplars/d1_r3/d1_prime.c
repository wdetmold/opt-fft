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

/* dev-only A/B switches (default off = current behavior); set via tryout.sh
 * extra gcc flags, e.g.  ./tryout.sh d1_prime 13 512 -DD1P_NOBAR */
#ifdef D1P_NOBAR          /* drop the "+m" barrier in the batched 13 kernels */
#define BAR13B(...) ((void)0)
#else
#define BAR13B(...) __asm__("" : __VA_ARGS__)
#endif
#ifdef D1P_NOBAR1         /* drop the "+m" barrier in the B=1 13 kernel */
#define BAR13(...) ((void)0)
#else
#define BAR13(...) __asm__("" : __VA_ARGS__)
#endif

typedef double v8 __attribute__((vector_size(64), aligned(64)));

struct fft1d_plan {
    int L, batch, h, hp;
    double *tc, *ts;   /* [h][hp]: k-contiguous, zero-padded to hp; tc[j][k-1] = cos(2pi k(j+1)/L) */
    double *ck, *sk;   /* [h][h] : j-contiguous; ck[(k-1)*h + (j-1)]  = cos(2pi k j/L) */
    double *tp;        /* L=13 pair tables: [j][4][8], j=0..5 = {cpA,cpB,spA,spB}; see exec13p */
};

const char *fft1d_name(void){ return "d1_prime"; }
const char *fft1d_description(void){
    return "symmetric-pair dense prime DFT; interleaved-pair zmm kernels at 13/31 (complex per 128b lane, pair-dup tables); fused SoA map chain (rsqrt14+NR)";
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
    if (L == 13) {
        /* pair tables for the interleaved-pair kernel: each 128-bit lane pair
         * carries one complex output, coefficients duplicated per pair.
         *   cpA lanes (2t,2t+1) = cos(2pi (t+1) j / 13),  t = 0..3  (k = 1..4)
         *   cpB pairs 0,1 = k = 5,6; pair 2 = 1.0 (the k=0 column: X0 = x0 + sum u);
         *         pair 3 = 0
         *   spA/spB likewise with sin (spB pair 2 = 0 so X0 gets no v term) */
        if (posix_memalign((void**)&p->tp, 64, (size_t)6*4*8*sizeof(double))) {
            fft1d_destroy(p); return NULL;
        }
        memset(p->tp, 0, (size_t)6*4*8*sizeof(double));
        for (int j = 1; j <= 6; j++) {
            double *cpA = p->tp + (size_t)(j-1)*32, *cpB = cpA + 8;
            double *spA = cpB + 8,                  *spB = spA + 8;
            for (int k = 1; k <= 6; k++) {
                double c = p->ck[(size_t)(k-1)*h + (j-1)];
                double s = p->sk[(size_t)(k-1)*h + (j-1)];
                if (k <= 4) { cpA[2*(k-1)] = cpA[2*(k-1)+1] = c;
                              spA[2*(k-1)] = s;  spA[2*(k-1)+1] = -s; }
                else        { cpB[2*(k-5)] = cpB[2*(k-5)+1] = c;
                              spB[2*(k-5)] = s;  spB[2*(k-5)+1] = -s; }
            }
            cpB[4] = cpB[5] = 1.0;   /* k=0 column rides pair 2 of the B row */
        }
    }
    if (L == 31) {
        /* pair tables for exec31p: [j][8][8], rows cpA..cpD then spA..spD;
         * row A = k 1..4, B = 5..8, C = 9..12, D = 13..15 + k=0 col in pair 3 */
        if (posix_memalign((void**)&p->tp, 64, (size_t)15*8*8*sizeof(double))) {
            fft1d_destroy(p); return NULL;
        }
        memset(p->tp, 0, (size_t)15*8*8*sizeof(double));
        for (int j = 1; j <= 15; j++) {
            double *base = p->tp + (size_t)(j-1)*64;
            for (int k = 1; k <= 15; k++) {
                double c = p->ck[(size_t)(k-1)*h + (j-1)];
                double s = p->sk[(size_t)(k-1)*h + (j-1)];
                int row = (k-1) >> 2, t = (k-1) & 3;
                base[8*row + 2*t]      = c;  base[8*row + 2*t + 1]      =  c;
                base[8*(row+4) + 2*t]  = s;  base[8*(row+4) + 2*t + 1]  = -s;
            }
            base[8*3 + 6] = base[8*3 + 7] = 1.0;   /* k=0 column: pair 3 of row D */
        }
    }
    return p;
}

void fft1d_destroy(fft1d_plan *p){
    if(!p) return;
    free(p->tc); free(p->ts); free(p->ck); free(p->sk); free(p->tp); free(p);
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
    /* barrier: without it gcc keeps u/v in zmm and turns every set1_pd(ur[j])
     * into a port-5 vpermpd (24 of them = the whole port-5 budget); through
     * memory they become load-port {1to8} FMA operands. Targeted "+m" (not a
     * full "memory" clobber) so the 12 tc/ts row loads can still be hoisted
     * out of the batch loop into registers. */
    BAR13("+m"(ur), "+m"(ui), "+m"(vr), "+m"(vi));
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

/* batched pair variant: TWO transforms interleaved in one body, single
 * accumulator set each (depth 6 hidden by the explicit cross-transform
 * overlap), table rows loaded once per pair */
static inline __attribute__((always_inline)) void
exec13_avx_b2(const double *restrict x, double *restrict y, const fft1d_plan *restrict p)
{
    const __m512i FRE = _mm512_setr_epi64(2,4,6,8,10,12,12,12);
    const __m512i FIM = _mm512_setr_epi64(3,5,7,9,11,13,13,13);
    const __m512i RRE = _mm512_setr_epi64(14,12,10,8,2,0,0,0);
    const __m512i RIM = _mm512_setr_epi64(15,13,11,9,3,1,1,1);
    const double *restrict x2 = x + 26;
    double *restrict y2 = y + 26;
    __m512d z0 = _mm512_loadu_pd(x),       z1 = _mm512_loadu_pd(x + 8);
    __m512d z2 = _mm512_loadu_pd(x + 14),  z3 = _mm512_loadu_pd(x + 18);
    __m512d w0 = _mm512_loadu_pd(x2),      w1 = _mm512_loadu_pd(x2 + 8);
    __m512d w2 = _mm512_loadu_pd(x2 + 14), w3 = _mm512_loadu_pd(x2 + 18);
    __m512d fr  = _mm512_permutex2var_pd(z0, FRE, z1), fi  = _mm512_permutex2var_pd(z0, FIM, z1);
    __m512d rr  = _mm512_permutex2var_pd(z2, RRE, z3), ri  = _mm512_permutex2var_pd(z2, RIM, z3);
    __m512d fr2 = _mm512_permutex2var_pd(w0, FRE, w1), fi2 = _mm512_permutex2var_pd(w0, FIM, w1);
    __m512d rr2 = _mm512_permutex2var_pd(w2, RRE, w3), ri2 = _mm512_permutex2var_pd(w2, RIM, w3);
    double ur[16] __attribute__((aligned(64))), ui[16] __attribute__((aligned(64)));
    double vr[16] __attribute__((aligned(64))), vi[16] __attribute__((aligned(64)));
    _mm512_store_pd(ur,     _mm512_add_pd(fr,  rr));
    _mm512_store_pd(ui,     _mm512_add_pd(fi,  ri));
    _mm512_store_pd(vr,     _mm512_sub_pd(fr,  rr));
    _mm512_store_pd(vi,     _mm512_sub_pd(fi,  ri));
    _mm512_store_pd(ur + 8, _mm512_add_pd(fr2, rr2));
    _mm512_store_pd(ui + 8, _mm512_add_pd(fi2, ri2));
    _mm512_store_pd(vr + 8, _mm512_sub_pd(fr2, rr2));
    _mm512_store_pd(vi + 8, _mm512_sub_pd(fi2, ri2));
    BAR13B("+m"(ur), "+m"(ui), "+m"(vr), "+m"(vi));
    /* hide the pointer so the x0 broadcasts cannot CSE against the z0/w0
     * register loads (that CSE turns them into port-5 vpermpd; from a hidden
     * pointer they are load-port vbroadcastsd m64) */
    const double *xa = x, *xa2 = x2;
    __asm__("" : "+r"(xa), "+r"(xa2));
    __m512d x0r  = _mm512_set1_pd(xa[0]),  x0i  = _mm512_set1_pd(xa[1]);
    __m512d x0r2 = _mm512_set1_pd(xa2[0]), x0i2 = _mm512_set1_pd(xa2[1]);
    __m512d Pr  = _mm512_setzero_pd(), Pi  = Pr, Rr  = Pr, Si  = Pr;
    __m512d Pr2 = Pr, Pi2 = Pr, Rr2 = Pr, Si2 = Pr;
    const double *restrict tc = p->tc, *restrict ts = p->ts;
    for (int j = 0; j < 6; j++) {
        __m512d cj = _mm512_load_pd(tc + 8*j), sj = _mm512_load_pd(ts + 8*j);
        Pr  = _mm512_fmadd_pd(cj, _mm512_set1_pd(ur[j]), Pr);
        Pi  = _mm512_fmadd_pd(cj, _mm512_set1_pd(ui[j]), Pi);
        Rr  = _mm512_fmadd_pd(sj, _mm512_set1_pd(vr[j]), Rr);
        Si  = _mm512_fmadd_pd(sj, _mm512_set1_pd(vi[j]), Si);
        Pr2 = _mm512_fmadd_pd(cj, _mm512_set1_pd(ur[j+8]), Pr2);
        Pi2 = _mm512_fmadd_pd(cj, _mm512_set1_pd(ui[j+8]), Pi2);
        Rr2 = _mm512_fmadd_pd(sj, _mm512_set1_pd(vr[j+8]), Rr2);
        Si2 = _mm512_fmadd_pd(sj, _mm512_set1_pd(vi[j+8]), Si2);
    }
    const __m512i Y0IDX = _mm512_setr_epi64(6,0,1,2,3,4,5,13);
    const __m512i Y1IDX = _mm512_setr_epi64(4,3,2,1,0,0,0,0);
    const __m512i ILO = _mm512_setr_epi64(0,8,1,9,2,10,3,11);
    const __m512i IHI = _mm512_setr_epi64(4,12,5,13,6,14,7,15);
    __m512d pre = _mm512_add_pd(x0r, Pr), pim = _mm512_add_pd(x0i, Pi);
    __m512d nar = _mm512_add_pd(pre, Si), nai = _mm512_sub_pd(pim, Rr);
    __m512d nbr = _mm512_sub_pd(pre, Si), nbi = _mm512_add_pd(pim, Rr);
    __m512d Y0r = _mm512_permutex2var_pd(nar, Y0IDX, nbr), Y0i = _mm512_permutex2var_pd(nai, Y0IDX, nbi);
    __m512d Y1r = _mm512_permutexvar_pd(Y1IDX, nbr),       Y1i = _mm512_permutexvar_pd(Y1IDX, nbi);
    _mm512_storeu_pd(y,      _mm512_permutex2var_pd(Y0r, ILO, Y0i));
    _mm512_storeu_pd(y + 8,  _mm512_permutex2var_pd(Y0r, IHI, Y0i));
    _mm512_storeu_pd(y + 16, _mm512_permutex2var_pd(Y1r, ILO, Y1i));
    _mm512_mask_storeu_pd(y + 24, 0x03, _mm512_permutex2var_pd(Y1r, IHI, Y1i));
    __m512d pre2 = _mm512_add_pd(x0r2, Pr2), pim2 = _mm512_add_pd(x0i2, Pi2);
    __m512d nar2 = _mm512_add_pd(pre2, Si2), nai2 = _mm512_sub_pd(pim2, Rr2);
    __m512d nbr2 = _mm512_sub_pd(pre2, Si2), nbi2 = _mm512_add_pd(pim2, Rr2);
    __m512d Y0r2 = _mm512_permutex2var_pd(nar2, Y0IDX, nbr2), Y0i2 = _mm512_permutex2var_pd(nai2, Y0IDX, nbi2);
    __m512d Y1r2 = _mm512_permutexvar_pd(Y1IDX, nbr2),        Y1i2 = _mm512_permutexvar_pd(Y1IDX, nbi2);
    _mm512_storeu_pd(y2,      _mm512_permutex2var_pd(Y0r2, ILO, Y0i2));
    _mm512_storeu_pd(y2 + 8,  _mm512_permutex2var_pd(Y0r2, IHI, Y0i2));
    _mm512_storeu_pd(y2 + 16, _mm512_permutex2var_pd(Y1r2, ILO, Y1i2));
    _mm512_mask_storeu_pd(y2 + 24, 0x03, _mm512_permutex2var_pd(Y1r2, IHI, Y1i2));
}

/* batched variant: one accumulator set (depth 6 is hidden by cross-transform
 * overlap at B>=8; saves the 4 combine adds and halves accumulator pressure) */
static void exec13_avx_b(const double *restrict x, double *restrict y, const fft1d_plan *restrict p)
{
    const __m512i FRE = _mm512_setr_epi64(2,4,6,8,10,12,12,12);
    const __m512i FIM = _mm512_setr_epi64(3,5,7,9,11,13,13,13);
    const __m512i RRE = _mm512_setr_epi64(14,12,10,8,2,0,0,0);
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
    BAR13B("+m"(ur), "+m"(ui), "+m"(vr), "+m"(vi));
    __m512d x0r = _mm512_set1_pd(x[0]), x0i = _mm512_set1_pd(x[1]);
    __m512d Pr = _mm512_setzero_pd(), Pi = Pr, Rr = Pr, Si = Pr;
    const double *restrict tc = p->tc, *restrict ts = p->ts;
    for (int j = 0; j < 6; j++) {
        __m512d cj = _mm512_load_pd(tc + 8*j), sj = _mm512_load_pd(ts + 8*j);
        Pr  = _mm512_fmadd_pd(cj, _mm512_set1_pd(ur[j]), Pr);
        Pi  = _mm512_fmadd_pd(cj, _mm512_set1_pd(ui[j]), Pi);
        Rr  = _mm512_fmadd_pd(sj, _mm512_set1_pd(vr[j]), Rr);
        Si  = _mm512_fmadd_pd(sj, _mm512_set1_pd(vi[j]), Si);
    }
    __m512d pre = _mm512_add_pd(x0r, Pr), pim = _mm512_add_pd(x0i, Pi);
    __m512d nar = _mm512_add_pd(pre, Si), nai = _mm512_sub_pd(pim, Rr);
    __m512d nbr = _mm512_sub_pd(pre, Si), nbi = _mm512_add_pd(pim, Rr);
    const __m512i Y0IDX = _mm512_setr_epi64(6,0,1,2,3,4,5,13);
    const __m512i Y1IDX = _mm512_setr_epi64(4,3,2,1,0,0,0,0);
    __m512d Y0r = _mm512_permutex2var_pd(nar, Y0IDX, nbr), Y0i = _mm512_permutex2var_pd(nai, Y0IDX, nbi);
    __m512d Y1r = _mm512_permutexvar_pd(Y1IDX, nbr),       Y1i = _mm512_permutexvar_pd(Y1IDX, nbi);
    const __m512i ILO = _mm512_setr_epi64(0,8,1,9,2,10,3,11);
    const __m512i IHI = _mm512_setr_epi64(4,12,5,13,6,14,7,15);
    _mm512_storeu_pd(y,      _mm512_permutex2var_pd(Y0r, ILO, Y0i));
    _mm512_storeu_pd(y + 8,  _mm512_permutex2var_pd(Y0r, IHI, Y0i));
    _mm512_storeu_pd(y + 16, _mm512_permutex2var_pd(Y1r, ILO, Y1i));
    _mm512_mask_storeu_pd(y + 24, 0x03, _mm512_permutex2var_pd(Y1r, IHI, Y1i));
}

/* ---- interleaved-pair L=13 kernel (r3) ----
 * Each 128-bit lane pair carries one complex value, so the natural interleaved
 * layout IS the compute layout: no deinterleave prologue, no re-interleave
 * epilogue, and 12 pair-broadcasts (vshuff64x2) instead of 24 scalar ones.
 * Coefficients are pair-duplicated at plan time (tp); sin is stored (+s,-s) so
 * S accumulates (rr,-si) and a single in-lane swap gives (-si,rr):
 *     na = pre - swap(S) = (pre_re + si, pre_im - rr) = X[k]
 *     nb = pre + swap(S) = X[13-k]
 * A rows: pairs X1..X4.  B rows: pairs X5, X6, X0 (k=0 column), pad. */
#define STEP13P(jj, UW, VW, tt, PA_, PB_, SA_, SB_) do {                      \
    __m512d ub_ = _mm512_shuffle_f64x2(UW, UW, (tt)*0x55);                    \
    __m512d vb_ = _mm512_shuffle_f64x2(VW, VW, (tt)*0x55);                    \
    PA_ = _mm512_fmadd_pd(_mm512_load_pd(tp + 32*(jj) +  0), ub_, PA_);       \
    PB_ = _mm512_fmadd_pd(_mm512_load_pd(tp + 32*(jj) +  8), ub_, PB_);       \
    SA_ = _mm512_fmadd_pd(_mm512_load_pd(tp + 32*(jj) + 16), vb_, SA_);       \
    SB_ = _mm512_fmadd_pd(_mm512_load_pd(tp + 32*(jj) + 24), vb_, SB_);       \
} while (0)

static inline __attribute__((always_inline)) void
exec13p_body(const double *restrict x, double *restrict y,
             const double *restrict tp, const int NSET)
{
    __m512d F1 = _mm512_loadu_pd(x + 2);              /* (x1)(x2)(x3)(x4)   */
    __m512d F2 = _mm512_loadu_pd(x + 10);             /* (x5)(x6)(x7)(x8)   */
    __m512d Z  = _mm512_loadu_pd(x + 18);             /* (x9)(x10)(x11)(x12)*/
    __m512d R1 = _mm512_shuffle_f64x2(Z,  Z,  0x1B);  /* (x12)(x11)(x10)(x9)*/
    __m512d R2 = _mm512_shuffle_f64x2(F2, F2, 0xBB);  /* (x8)(x7)(x8)(x7)   */
    __m512d U1 = _mm512_add_pd(F1, R1), V1 = _mm512_sub_pd(F1, R1);
    __m512d U2 = _mm512_add_pd(F2, R2), V2 = _mm512_sub_pd(F2, R2);
    /* x0 seeds the P accumulators, so no separate pre = x0 + P add at the end */
    __m512d x0p = _mm512_broadcast_f64x2(_mm_loadu_pd(x));
    __m512d PA = x0p, PB = x0p, SA = _mm512_setzero_pd(), SB = SA;
    if (NSET == 2) {   /* two sets halve the FMA depth: the B=1 critical path */
        __m512d PA2 = _mm512_setzero_pd(), PB2 = PA2, SA2 = PA2, SB2 = PA2;
        STEP13P(0, U1, V1, 0, PA,  PB,  SA,  SB );
        STEP13P(1, U1, V1, 1, PA2, PB2, SA2, SB2);
        STEP13P(2, U1, V1, 2, PA,  PB,  SA,  SB );
        STEP13P(3, U1, V1, 3, PA2, PB2, SA2, SB2);
        STEP13P(4, U2, V2, 0, PA,  PB,  SA,  SB );
        STEP13P(5, U2, V2, 1, PA2, PB2, SA2, SB2);
        PA = _mm512_add_pd(PA, PA2); PB = _mm512_add_pd(PB, PB2);
        SA = _mm512_add_pd(SA, SA2); SB = _mm512_add_pd(SB, SB2);
    } else {
        STEP13P(0, U1, V1, 0, PA, PB, SA, SB);
        STEP13P(1, U1, V1, 1, PA, PB, SA, SB);
        STEP13P(2, U1, V1, 2, PA, PB, SA, SB);
        STEP13P(3, U1, V1, 3, PA, PB, SA, SB);
        STEP13P(4, U2, V2, 0, PA, PB, SA, SB);
        STEP13P(5, U2, V2, 1, PA, PB, SA, SB);
    }
    __m512d swA  = _mm512_permute_pd(SA, 0x55), swB = _mm512_permute_pd(SB, 0x55);
    __m512d naA  = _mm512_sub_pd(PA, swA), nbA = _mm512_add_pd(PA, swA);
    __m512d naB  = _mm512_sub_pd(PB, swB), nbB = _mm512_add_pd(PB, swB);
    /* naA = X1..X4, naB = X5,X6,X0,--, nbA = X12..X9, nbB = X8,X7,--,-- */
    const __m512i IDX0 = _mm512_setr_epi64(12,13,0,1,2,3,4,5);      /* X0,X1..X3   */
    const __m512i IDXT = _mm512_setr_epi64(0,1,2,3,10,11,10,11);    /* X5,X6,X7    */
    const __m512i IDX1 = _mm512_setr_epi64(6,7,8,9,10,11,12,13);    /* X4,X5,X6,X7 */
    const __m512i IDX2 = _mm512_setr_epi64(0,1,14,15,12,13,10,11);  /* X8..X11     */
    _mm512_storeu_pd(y,      _mm512_permutex2var_pd(naA, IDX0, naB));
    __m512d t = _mm512_permutex2var_pd(naB, IDXT, nbB);
    _mm512_storeu_pd(y + 8,  _mm512_permutex2var_pd(naA, IDX1, t));
    _mm512_storeu_pd(y + 16, _mm512_permutex2var_pd(nbB, IDX2, nbA));
    _mm512_mask_storeu_pd(y + 24, 0x03, nbA);                       /* X12 */
}

static void exec13p_1(const double *restrict x, double *restrict y, const fft1d_plan *restrict p){
    exec13p_body(x, y, p->tp, 2);
}
static void exec13p_b(const double *restrict x, double *restrict y, const fft1d_plan *restrict p){
    exec13p_body(x, y, p->tp, 1);
}

/* two transforms per body: table rows loaded once per pair, all loads grouped
 * ahead of all stores (keeps load->store distance up across the batch loop) */
#define STEP13P2(jj, UW, VW, UX, VX, tt) do {                                 \
    __m512d c1_ = _mm512_load_pd(tp + 32*(jj) +  0);                          \
    __m512d c2_ = _mm512_load_pd(tp + 32*(jj) +  8);                          \
    __m512d s1_ = _mm512_load_pd(tp + 32*(jj) + 16);                          \
    __m512d s2_ = _mm512_load_pd(tp + 32*(jj) + 24);                          \
    __m512d ub_ = _mm512_shuffle_f64x2(UW, UW, (tt)*0x55);                    \
    __m512d vb_ = _mm512_shuffle_f64x2(VW, VW, (tt)*0x55);                    \
    __m512d uc_ = _mm512_shuffle_f64x2(UX, UX, (tt)*0x55);                    \
    __m512d vc_ = _mm512_shuffle_f64x2(VX, VX, (tt)*0x55);                    \
    PA = _mm512_fmadd_pd(c1_, ub_, PA);  QA = _mm512_fmadd_pd(c1_, uc_, QA);  \
    PB = _mm512_fmadd_pd(c2_, ub_, PB);  QB = _mm512_fmadd_pd(c2_, uc_, QB);  \
    SA = _mm512_fmadd_pd(s1_, vb_, SA);  TA = _mm512_fmadd_pd(s1_, vc_, TA);  \
    SB = _mm512_fmadd_pd(s2_, vb_, SB);  TB = _mm512_fmadd_pd(s2_, vc_, TB);  \
} while (0)

static inline __attribute__((always_inline)) void
exec13p_b2(const double *restrict x, double *restrict y, const double *restrict tp)
{
    const double *restrict x2 = x + 26;
    double *restrict y2 = y + 26;
    __m512d F1 = _mm512_loadu_pd(x + 2),  G1 = _mm512_loadu_pd(x2 + 2);
    __m512d F2 = _mm512_loadu_pd(x + 10), G2 = _mm512_loadu_pd(x2 + 10);
    __m512d Z  = _mm512_loadu_pd(x + 18), W  = _mm512_loadu_pd(x2 + 18);
    __m512d x0p = _mm512_broadcast_f64x2(_mm_loadu_pd(x));
    __m512d x0q = _mm512_broadcast_f64x2(_mm_loadu_pd(x2));
    __m512d R1 = _mm512_shuffle_f64x2(Z,  Z,  0x1B), S1r = _mm512_shuffle_f64x2(W,  W,  0x1B);
    __m512d R2 = _mm512_shuffle_f64x2(F2, F2, 0xBB), S2r = _mm512_shuffle_f64x2(G2, G2, 0xBB);
    __m512d U1 = _mm512_add_pd(F1, R1),  V1 = _mm512_sub_pd(F1, R1);
    __m512d U2 = _mm512_add_pd(F2, R2),  V2 = _mm512_sub_pd(F2, R2);
    __m512d X1 = _mm512_add_pd(G1, S1r), W1 = _mm512_sub_pd(G1, S1r);
    __m512d X2 = _mm512_add_pd(G2, S2r), W2 = _mm512_sub_pd(G2, S2r);
    __m512d PA = x0p, PB = x0p, SA = _mm512_setzero_pd(), SB = SA;
    __m512d QA = x0q, QB = x0q, TA = SA, TB = SA;
    STEP13P2(0, U1, V1, X1, W1, 0);
    STEP13P2(1, U1, V1, X1, W1, 1);
    STEP13P2(2, U1, V1, X1, W1, 2);
    STEP13P2(3, U1, V1, X1, W1, 3);
    STEP13P2(4, U2, V2, X2, W2, 0);
    STEP13P2(5, U2, V2, X2, W2, 1);
    const __m512i IDX0 = _mm512_setr_epi64(12,13,0,1,2,3,4,5);
    const __m512i IDXT = _mm512_setr_epi64(0,1,2,3,10,11,10,11);
    const __m512i IDX1 = _mm512_setr_epi64(6,7,8,9,10,11,12,13);
    const __m512i IDX2 = _mm512_setr_epi64(0,1,14,15,12,13,10,11);
    __m512d swA = _mm512_permute_pd(SA, 0x55), swB = _mm512_permute_pd(SB, 0x55);
    __m512d naA = _mm512_sub_pd(PA, swA), nbA = _mm512_add_pd(PA, swA);
    __m512d naB = _mm512_sub_pd(PB, swB), nbB = _mm512_add_pd(PB, swB);
    __m512d swC = _mm512_permute_pd(TA, 0x55), swD = _mm512_permute_pd(TB, 0x55);
    __m512d naC = _mm512_sub_pd(QA, swC), nbC = _mm512_add_pd(QA, swC);
    __m512d naD = _mm512_sub_pd(QB, swD), nbD = _mm512_add_pd(QB, swD);
    _mm512_storeu_pd(y,       _mm512_permutex2var_pd(naA, IDX0, naB));
    __m512d t  = _mm512_permutex2var_pd(naB, IDXT, nbB);
    _mm512_storeu_pd(y + 8,   _mm512_permutex2var_pd(naA, IDX1, t));
    _mm512_storeu_pd(y + 16,  _mm512_permutex2var_pd(nbB, IDX2, nbA));
    _mm512_mask_storeu_pd(y + 24, 0x03, nbA);
    _mm512_storeu_pd(y2,      _mm512_permutex2var_pd(naC, IDX0, naD));
    __m512d t2 = _mm512_permutex2var_pd(naD, IDXT, nbD);
    _mm512_storeu_pd(y2 + 8,  _mm512_permutex2var_pd(naC, IDX1, t2));
    _mm512_storeu_pd(y2 + 16, _mm512_permutex2var_pd(nbD, IDX2, nbC));
    _mm512_mask_storeu_pd(y2 + 24, 0x03, nbC);
}

/* ---- interleaved-pair L=31 kernel (r3): same scheme as exec13p ----
 * Rows: A=(X1..X4) B=(X5..X8) C=(X9..X12) D=(X13,X14,X15,X0); conjugates
 * nbA=(X30..X27) nbB=(X26..X23) nbC=(X22..X19) nbD=(X18,X17,X16,--).
 * Outputs store straight back interleaved: na rows are already natural order,
 * nb rows need one 0x1B pair-reversal each. Tables tp31[j][8][8], sin (+s,-s). */
#define STEP31P(jj, UW, VW, tt) do {                                          \
    __m512d ub_ = _mm512_shuffle_f64x2(UW, UW, (tt)*0x55);                    \
    __m512d vb_ = _mm512_shuffle_f64x2(VW, VW, (tt)*0x55);                    \
    PA = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) +  0), ub_, PA);         \
    PB = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) +  8), ub_, PB);         \
    PC = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 16), ub_, PC);         \
    PD = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 24), ub_, PD);         \
    SA = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 32), vb_, SA);         \
    SB = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 40), vb_, SB);         \
    SC = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 48), vb_, SC);         \
    SD = _mm512_fmadd_pd(_mm512_load_pd(tq + 64*(jj) + 56), vb_, SD);         \
} while (0)

static inline __attribute__((always_inline)) void
exec31p_body(const double *restrict x, double *restrict y, const double *restrict tq)
{
    __m512d F1 = _mm512_loadu_pd(x + 2);    /* (x1)(x2)(x3)(x4)     */
    __m512d F2 = _mm512_loadu_pd(x + 10);   /* (x5)(x6)(x7)(x8)     */
    __m512d F3 = _mm512_loadu_pd(x + 18);   /* (x9)(x10)(x11)(x12)  */
    __m512d F4 = _mm512_loadu_pd(x + 26);   /* (x13)(x14)(x15)(x16) */
    __m512d Z1 = _mm512_loadu_pd(x + 54);   /* (x27)(x28)(x29)(x30) */
    __m512d Z2 = _mm512_loadu_pd(x + 46);   /* (x23)(x24)(x25)(x26) */
    __m512d Z3 = _mm512_loadu_pd(x + 38);   /* (x19)(x20)(x21)(x22) */
    __m512d Z4 = _mm512_loadu_pd(x + 30);   /* (x15)(x16)(x17)(x18) */
    __m512d R1 = _mm512_shuffle_f64x2(Z1, Z1, 0x1B);   /* (x30)(x29)(x28)(x27) */
    __m512d R2 = _mm512_shuffle_f64x2(Z2, Z2, 0x1B);   /* (x26)(x25)(x24)(x23) */
    __m512d R3 = _mm512_shuffle_f64x2(Z3, Z3, 0x1B);   /* (x22)(x21)(x20)(x19) */
    __m512d R4 = _mm512_shuffle_f64x2(Z4, Z4, 0x1B);   /* (x18)(x17)(x16)(x15) */
    __m512d U1 = _mm512_add_pd(F1, R1), V1 = _mm512_sub_pd(F1, R1);
    __m512d U2 = _mm512_add_pd(F2, R2), V2 = _mm512_sub_pd(F2, R2);
    __m512d U3 = _mm512_add_pd(F3, R3), V3 = _mm512_sub_pd(F3, R3);
    __m512d U4 = _mm512_add_pd(F4, R4), V4 = _mm512_sub_pd(F4, R4);  /* pair 3 junk */
    __m512d x0p = _mm512_broadcast_f64x2(_mm_loadu_pd(x));
    __m512d PA = x0p, PB = x0p, PC = x0p, PD = x0p;
    __m512d SA = _mm512_setzero_pd(), SB = SA, SC = SA, SD = SA;
    STEP31P( 0, U1, V1, 0);  STEP31P( 1, U1, V1, 1);
    STEP31P( 2, U1, V1, 2);  STEP31P( 3, U1, V1, 3);
    STEP31P( 4, U2, V2, 0);  STEP31P( 5, U2, V2, 1);
    STEP31P( 6, U2, V2, 2);  STEP31P( 7, U2, V2, 3);
    STEP31P( 8, U3, V3, 0);  STEP31P( 9, U3, V3, 1);
    STEP31P(10, U3, V3, 2);  STEP31P(11, U3, V3, 3);
    STEP31P(12, U4, V4, 0);  STEP31P(13, U4, V4, 1);
    STEP31P(14, U4, V4, 2);
    __m512d swA = _mm512_permute_pd(SA, 0x55), swB = _mm512_permute_pd(SB, 0x55);
    __m512d swC = _mm512_permute_pd(SC, 0x55), swD = _mm512_permute_pd(SD, 0x55);
    __m512d naA = _mm512_sub_pd(PA, swA), nbA = _mm512_add_pd(PA, swA);
    __m512d naB = _mm512_sub_pd(PB, swB), nbB = _mm512_add_pd(PB, swB);
    __m512d naC = _mm512_sub_pd(PC, swC), nbC = _mm512_add_pd(PC, swC);
    __m512d naD = _mm512_sub_pd(PD, swD), nbD = _mm512_add_pd(PD, swD);
    _mm_storeu_pd(y, _mm512_extractf64x2_pd(naD, 3));            /* X0 */
    _mm512_storeu_pd(y + 2,  naA);                               /* X1..X4   */
    _mm512_storeu_pd(y + 10, naB);                               /* X5..X8   */
    _mm512_storeu_pd(y + 18, naC);                               /* X9..X12  */
    _mm512_mask_storeu_pd(y + 26, 0x3F, naD);                    /* X13..X15 */
    _mm512_mask_storeu_pd(y + 30, 0xFC,
        _mm512_shuffle_f64x2(nbD, nbD, 0x1B));                   /* X16..X18 */
    _mm512_storeu_pd(y + 38, _mm512_shuffle_f64x2(nbC, nbC, 0x1B)); /* X19..X22 */
    _mm512_storeu_pd(y + 46, _mm512_shuffle_f64x2(nbB, nbB, 0x1B)); /* X23..X26 */
    _mm512_storeu_pd(y + 54, _mm512_shuffle_f64x2(nbA, nbA, 0x1B)); /* X27..X30 */
}

static void exec31p(const double *restrict x, double *restrict y, const fft1d_plan *restrict p){
    exec31p_body(x, y, p->tp);
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

/* batched variant of exec31: ONE accumulator set (8 rows). The 2-set version
 * holds 16 accumulators + tables + state and spills (44 vmovapd in the emitted
 * loop); at B>=8 the depth-15 chain is hidden by cross-transform overlap, so
 * the spills are pure loss. */
static inline __attribute__((always_inline)) void
exec31_avx_b(const double *restrict x, double *restrict y, const fft1d_plan *restrict p)
{
    const __m512i EVEN = _mm512_setr_epi64(0,2,4,6,8,10,12,14);
    const __m512i ODD  = _mm512_setr_epi64(1,3,5,7,9,11,13,15);
    const __m512i T3E  = _mm512_setr_epi64(0,2,4,6,8,10,12,12);
    const __m512i T3O  = _mm512_setr_epi64(1,3,5,7,9,11,13,13);
    __m512d z0 = _mm512_loadu_pd(x),      z1 = _mm512_loadu_pd(x + 8);
    __m512d z2 = _mm512_loadu_pd(x + 16), z3 = _mm512_loadu_pd(x + 24);
    __m512d z4 = _mm512_loadu_pd(x + 32), z5 = _mm512_loadu_pd(x + 40);
    __m512d z6 = _mm512_loadu_pd(x + 48), z7 = _mm512_maskz_loadu_pd(0x3F, x + 56);
    __m512d xr0 = _mm512_permutex2var_pd(z0, EVEN, z1), xi0 = _mm512_permutex2var_pd(z0, ODD, z1);
    __m512d xr1 = _mm512_permutex2var_pd(z2, EVEN, z3), xi1 = _mm512_permutex2var_pd(z2, ODD, z3);
    __m512d xr2 = _mm512_permutex2var_pd(z4, EVEN, z5), xi2 = _mm512_permutex2var_pd(z4, ODD, z5);
    __m512d xr3 = _mm512_permutex2var_pd(z6, T3E,  z7), xi3 = _mm512_permutex2var_pd(z6, T3O, z7);
    const __m512i F0 = _mm512_setr_epi64(1,2,3,4,5,6,7,8);
    const __m512i F1 = _mm512_setr_epi64(1,2,3,4,5,6,7,7);
    const __m512i R0 = _mm512_setr_epi64(6,5,4,3,2,1,0,15);
    const __m512i R1 = _mm512_setr_epi64(6,5,4,3,2,1,0,0);
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
    BAR13B("+m"(ur), "+m"(ui), "+m"(vr), "+m"(vi));
    __m512d x0r = bcast0(xr0), x0i = bcast0(xi0);
    __m512d Pr0 = _mm512_setzero_pd(), Pi0 = Pr0, Rr0 = Pr0, Si0 = Pr0;
    __m512d Pr1 = Pr0, Pi1 = Pr0, Rr1 = Pr0, Si1 = Pr0;
    const double *restrict tc = p->tc, *restrict ts = p->ts;
    for (int j = 0; j < 15; j++) {
        __m512d c0 = _mm512_load_pd(tc + 16*j),     s0 = _mm512_load_pd(ts + 16*j);
        __m512d c1 = _mm512_load_pd(tc + 16*j + 8), s1 = _mm512_load_pd(ts + 16*j + 8);
        __m512d a = _mm512_set1_pd(ur[j]), b = _mm512_set1_pd(ui[j]);
        __m512d e = _mm512_set1_pd(vr[j]), f = _mm512_set1_pd(vi[j]);
        Pr0 = _mm512_fmadd_pd(c0, a, Pr0);  Pr1 = _mm512_fmadd_pd(c1, a, Pr1);
        Pi0 = _mm512_fmadd_pd(c0, b, Pi0);  Pi1 = _mm512_fmadd_pd(c1, b, Pi1);
        Rr0 = _mm512_fmadd_pd(s0, e, Rr0);  Rr1 = _mm512_fmadd_pd(s1, e, Rr1);
        Si0 = _mm512_fmadd_pd(s0, f, Si0);  Si1 = _mm512_fmadd_pd(s1, f, Si1);
    }
    __m512d pre0 = _mm512_add_pd(x0r, Pr0), pim0 = _mm512_add_pd(x0i, Pi0);
    __m512d pre1 = _mm512_add_pd(x0r, Pr1), pim1 = _mm512_add_pd(x0i, Pi1);
    __m512d na0r = _mm512_add_pd(pre0, Si0), na0i = _mm512_sub_pd(pim0, Rr0);
    __m512d na1r = _mm512_add_pd(pre1, Si1), na1i = _mm512_sub_pd(pim1, Rr1);
    __m512d nb0r = _mm512_sub_pd(pre0, Si0), nb0i = _mm512_add_pd(pim0, Rr0);
    __m512d nb1r = _mm512_sub_pd(pre1, Si1), nb1i = _mm512_add_pd(pim1, Rr1);
    const __m512i Y01 = _mm512_setr_epi64(7,8,9,10,11,12,13,14);
    const __m512i Y2I = _mm512_setr_epi64(6,5,4,3,2,1,0,15);
    const __m512i Y3I = _mm512_setr_epi64(6,5,4,3,2,1,0,0);
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
#ifdef D1P_R1DISPATCH   /* r1-style: the 2-accumulator-set kernel for every B */
    if (L == 13)      { for (int b = 0; b < B; b++) exec13_avx (x + 26*(size_t)b, y + 26*(size_t)b, p); return; }
    else if (L == 31) { for (int b = 0; b < B; b++) exec31_avx (x + 62*(size_t)b, y + 62*(size_t)b, p); return; }
#endif
#ifndef D1P_OLD13       /* r3 default: interleaved-pair kernel for L=13 */
    if (L == 13)      { if (B == 1) exec13p_1(x, y, p);
                        else if (B < 8) { for (int b = 0; b < B; b++) exec13p_1(x + 26*(size_t)b, y + 26*(size_t)b, p); }
                        else            { int b = 0; const double *tp = p->tp;
                                          for (; b + 2 <= B; b += 2) exec13p_b2(x + 26*(size_t)b, y + 26*(size_t)b, tp);
                                          for (; b < B; b++)         exec13p_b (x + 26*(size_t)b, y + 26*(size_t)b, p); }
                        return; }
    else if (L == 31) {
#ifndef D1P_OLD31B1
        if (B == 1) { exec31p(x, y, p); return; }
#endif
#ifndef D1P_OLD31B
        { const double *tq = p->tp;
          for (int b = 0; b < B; b++) exec31p_body(x + 62*(size_t)b, y + 62*(size_t)b, tq);
          return; }
#endif
    }
#endif
    if (L == 13)      { if (B >= 8) { int b = 0;
                                      for (; b + 2 <= B; b += 2) exec13_avx_b2(x + 26*(size_t)b, y + 26*(size_t)b, p);
                                      for (; b < B; b++)         exec13_avx_b (x + 26*(size_t)b, y + 26*(size_t)b, p); }
                        else        { for (int b = 0; b < B; b++) exec13_avx  (x + 26*(size_t)b, y + 26*(size_t)b, p); } return; }
    else if (L == 31) { if (B >= 8) { for (int b = 0; b < B; b++) exec31_avx_b(x + 62*(size_t)b, y + 62*(size_t)b, p); }
                        else        { for (int b = 0; b < B; b++) exec31_avx  (x + 62*(size_t)b, y + 62*(size_t)b, p); } return; }
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
        /* same barrier as exec13: u/v broadcasts must come from memory
         * (load ports), not vpermpd (port 5) -- see exec13_avx */
#ifndef D1P_NOBARCH
        __asm__("" : : : "memory");
#endif
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
