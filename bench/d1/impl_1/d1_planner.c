/*
 * d1_planner -- LIBRARY LAYER (adoption-scored): 1D factorization -> candidate
 * algorithm chains.
 *
 * The planner in fft1d_create() picks the chain from the factorization of L:
 *   - L fully smooth (all prime factors <= 61)  -> mixed-radix Stockham, one pass
 *     per factor, no bit reversal, natural-order output.
 *   - L prime with L-1 smooth                   -> UNPADDED Rader: the DFT becomes an
 *     (L-1)-point cyclic convolution done with two mixed-radix FFTs (65537 -> 2^16,
 *     1021 -> 1020 = 4*3*5*17).  b-spectrum and the 1/(L-1) normalization are folded
 *     at plan time, so execute pays 2 FFTs + 1 pointwise + 2 permutes.
 *   - anything else                             -> Bluestein chirp-Z, convolution
 *     length chosen by a small cost model over {1,3,5,9,15}*2^k >= 2L-1 (10007 ->
 *     20480 = 5*2^12, not 32768).  Chirp phases are reduced k^2 mod 2L in INTEGERS
 *     before the trig call (survey vein 2: the fp64 trap at k ~ 1e5).
 *
 * Pieces meant to be lifted by other entries (that is what this layer is for):
 *   mr_build/mr_exec   generic out-of-place Stockham engine, sign=-1 fwd / +1 inv
 *                      (inverse UNNORMALIZED), hardcoded radix 2/3/4 kernels plus a
 *                      generic O(r^2) radix for any odd prime r <= 61,
 *                      per-stage twiddles from exact integer phase (j*s mod l*r).
 *   rader_* / blu_*    the two prime chains, both built on mr_*.
 *   choose_conv_len    the padded-convolution size model.
 *
 * Compile-time self test:  gcc -DPLANNER_TEST -O2 d1_planner.c -lm  (checks the whole
 * planner against a naive DFT across smooth/Rader/Bluestein sizes).
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../fft1d_api.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAXF     48   /* max factors of one length (2^48 -- never binding) */
#define GENR_MAX 61   /* largest prime run as a direct mixed-radix stage */

static void *aligned64(size_t bytes)
{
    void *p = NULL;
    if (bytes == 0) bytes = 64;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
}

/* =================== mixed-radix Stockham autosort engine =================== */

typedef struct {
    int n, nf, sign;          /* sign: -1 forward, +1 inverse (unnormalized) */
    int radix[MAXF];
    size_t twoff[MAXF];       /* per-stage offset into tw, in complex elements */
    double *tw;               /* interleaved re,im per-stage twiddle tables */
    double *wmat[MAXF];       /* generic stages: r*r DFT matrix; NULL for 2/3/4 */
} mr_plan;

/* pow2 part: radix-4 chains below R8_THRESH (compute-bound; the r4 kernel is the
 * fastest per point), radix-8 chains above it (memory-bound; fewest full-array
 * sweeps wins), always LED by a radix-4 stage so the scalar m==1 first stage runs
 * in the cheap kernel.  Then odd primes ascending, so generic (slow) radices land
 * late where m is large and their inner loops are long and contiguous.
 * Measured A/B on wallaby: pure-8 plans lose 1.5-2.3x at 1024..20480, win at 65536+.
 * Returns factor count; *leftover = unfactored part. */
#ifndef R8_THRESH
#define R8_THRESH 131072
#endif
static int factorize(int n, int radix[MAXF], int *leftover)
{
    int nf = 0, a = 0;
    const int n0 = n;
    while (n % 2 == 0) { ++a; n /= 2; }
    if (n0 >= R8_THRESH && a >= 3) {
        if (a % 3 == 0) {
            for (int i = 0; i < a / 3 && nf < MAXF; ++i) radix[nf++] = 8;
        } else if (a % 3 == 2) {
            radix[nf++] = 4;
            for (int i = 0; i < (a - 2) / 3 && nf < MAXF; ++i) radix[nf++] = 8;
        } else {                    /* a % 3 == 1, a >= 4: [4, 8..., 4] */
            radix[nf++] = 4;
            for (int i = 0; i < (a - 4) / 3 && nf < MAXF; ++i) radix[nf++] = 8;
            if (nf < MAXF) radix[nf++] = 4;
        }
    } else {
        for (int i = 0; i < a / 2 && nf < MAXF; ++i) radix[nf++] = 4;
        if ((a & 1) && nf < MAXF) radix[nf++] = 2;
    }
    for (int p = 3; p <= GENR_MAX && nf < MAXF; p += 2)
        while (n % p == 0 && nf < MAXF) { radix[nf++] = p; n /= p; }
    *leftover = n;
    return nf;
}

static void mr_free(mr_plan *P)
{
    free(P->tw);
    for (int q = 0; q < P->nf; ++q) free(P->wmat[q]);
    memset(P, 0, sizeof *P);
}

static int mr_build(mr_plan *P, int n, int sign)
{
    memset(P, 0, sizeof *P);
    P->n = n; P->sign = sign;
    int leftover;
    P->nf = factorize(n, P->radix, &leftover);
    if (leftover != 1) return -1;

    size_t tot = 0;
    {
        int l = n;
        for (int q = 0; q < P->nf; ++q) {
            int r = P->radix[q]; l /= r;
            P->twoff[q] = tot;
            tot += (size_t)l * r;
        }
    }
    P->tw = aligned64(2 * tot * sizeof(double));
    if (!P->tw) return -1;

    int l = n;
    for (int q = 0; q < P->nf; ++q) {
        int r = P->radix[q]; l /= r;
        double *t = P->tw + 2 * P->twoff[q];
        long lr = (long)l * r;
        for (int j = 0; j < l; ++j)
            for (int i = 0; i < r; ++i) {
                long ph = ((long)j * i) % lr;      /* exact integer phase */
                double th = 2.0 * M_PI * (double)ph / (double)lr;
                t[2 * ((size_t)j * r + i)]     = cos(th);
                t[2 * ((size_t)j * r + i) + 1] = (sign < 0) ? -sin(th) : sin(th);
            }
        if (r != 2 && r != 3 && r != 4 && r != 8) {
            double *W = aligned64(2 * (size_t)r * r * sizeof(double));
            if (!W) return -1;
            for (int s = 0; s < r; ++s)
                for (int i = 0; i < r; ++i) {
                    int ph = (s * i) % r;
                    double th = 2.0 * M_PI * (double)ph / (double)r;
                    W[2 * ((size_t)s * r + i)]     = cos(th);
                    W[2 * ((size_t)s * r + i) + 1] = (sign < 0) ? -sin(th) : sin(th);
                }
            P->wmat[q] = W;
        }
    }
    return 0;
}

/* One DIF-Stockham stage, radix r, l blocks of r inputs strided l*m, m contiguous:
 *   t_i = src[k + m*(j + l*i)]
 *   u_s = sum_i t_i w_r^{is}            (sign-adjusted)
 *   dst[k + m*(r*j + s)] = u_s * w_{lr}^{js}
 * Natural-order output after all stages (self-sorting), verified in PLANNER_TEST. */

static void pass_r2(int l, int m, const double *restrict s, double *restrict d,
                    const double *restrict tw)
{
    for (int j = 0; j < l; ++j) {
        const double *x0 = s + 2 * (size_t)m * j;
        const double *x1 = s + 2 * (size_t)m * (j + l);
        double *y0 = d + 2 * (size_t)m * (2 * j);
        double *y1 = y0 + 2 * (size_t)m;
        double wr = tw[2 * (2 * j + 1)], wi = tw[2 * (2 * j + 1) + 1];
        for (int k = 0; k < m; ++k) {
            double ar = x0[2*k], ai = x0[2*k+1], br = x1[2*k], bi = x1[2*k+1];
            y0[2*k]   = ar + br;  y0[2*k+1] = ai + bi;
            double ur = ar - br, ui = ai - bi;
            y1[2*k]   = ur * wr - ui * wi;
            y1[2*k+1] = ur * wi + ui * wr;
        }
    }
}

static void pass_r3(int l, int m, const double *restrict s, double *restrict d,
                    const double *restrict tw, double sg)
{
    const double h = 0.5, rt = 0.86602540378443864676 * sg; /* sg*sqrt(3)/2 */
    for (int j = 0; j < l; ++j) {
        const double *x0 = s + 2 * (size_t)m * j;
        const double *x1 = s + 2 * (size_t)m * (j + l);
        const double *x2 = s + 2 * (size_t)m * (j + 2 * l);
        double *y0 = d + 2 * (size_t)m * (3 * j);
        double *y1 = y0 + 2 * (size_t)m;
        double *y2 = y1 + 2 * (size_t)m;
        const double *w = tw + 2 * (size_t)(3 * j);
        double w1r = w[2], w1i = w[3], w2r = w[4], w2i = w[5];
        for (int k = 0; k < m; ++k) {
            double t0r = x0[2*k], t0i = x0[2*k+1];
            double t1r = x1[2*k], t1i = x1[2*k+1];
            double t2r = x2[2*k], t2i = x2[2*k+1];
            double s1r = t1r + t2r, s1i = t1i + t2i;   /* t1+t2 */
            double s2r = t1r - t2r, s2i = t1i - t2i;   /* t1-t2 */
            y0[2*k]   = t0r + s1r;  y0[2*k+1] = t0i + s1i;
            double br = t0r - h * s1r, bi = t0i - h * s1i;
            /* u1 = b + sg*i*(sqrt3/2)*s2 ; u2 = b - sg*i*(sqrt3/2)*s2 */
            double u1r = br - rt * s2i, u1i = bi + rt * s2r;
            double u2r = br + rt * s2i, u2i = bi - rt * s2r;
            y1[2*k]   = u1r * w1r - u1i * w1i;  y1[2*k+1] = u1r * w1i + u1i * w1r;
            y2[2*k]   = u2r * w2r - u2i * w2i;  y2[2*k+1] = u2r * w2i + u2i * w2r;
        }
    }
}

static void pass_r4(int l, int m, const double *restrict s, double *restrict d,
                    const double *restrict tw, double sg)
{
    for (int j = 0; j < l; ++j) {
        const double *x0 = s + 2 * (size_t)m * j;
        const double *x1 = s + 2 * (size_t)m * (j + l);
        const double *x2 = s + 2 * (size_t)m * (j + 2 * l);
        const double *x3 = s + 2 * (size_t)m * (j + 3 * l);
        double *y0 = d + 2 * (size_t)m * (4 * j);
        double *y1 = y0 + 2 * (size_t)m;
        double *y2 = y1 + 2 * (size_t)m;
        double *y3 = y2 + 2 * (size_t)m;
        const double *w = tw + 2 * (size_t)(4 * j);
        double w1r = w[2], w1i = w[3], w2r = w[4], w2i = w[5], w3r = w[6], w3i = w[7];
#define R4_BODY(TWIDDLE)                                                          \
        for (int k = 0; k < m; ++k) {                                             \
            double t0r = x0[2*k], t0i = x0[2*k+1];                                \
            double t1r = x1[2*k], t1i = x1[2*k+1];                                \
            double t2r = x2[2*k], t2i = x2[2*k+1];                                \
            double t3r = x3[2*k], t3i = x3[2*k+1];                                \
            double e0r = t0r + t2r, e0i = t0i + t2i;                              \
            double e1r = t0r - t2r, e1i = t0i - t2i;                              \
            double o0r = t1r + t3r, o0i = t1i + t3i;                              \
            double o1r = t1r - t3r, o1i = t1i - t3i;                              \
            y0[2*k]   = e0r + o0r;  y0[2*k+1] = e0i + o0i;                        \
            /* u1 = e1 + sg*i*o1, u3 = e1 - sg*i*o1, u2 = e0 - o0 */              \
            double u1r = e1r - sg * o1i, u1i = e1i + sg * o1r;                    \
            double u2r = e0r - o0r,      u2i = e0i - o0i;                         \
            double u3r = e1r + sg * o1i, u3i = e1i - sg * o1r;                    \
            TWIDDLE                                                               \
        }
        if (j == 0) {
            R4_BODY(
                y1[2*k]=u1r; y1[2*k+1]=u1i;
                y2[2*k]=u2r; y2[2*k+1]=u2i;
                y3[2*k]=u3r; y3[2*k+1]=u3i;
            )
        } else {
            R4_BODY(
                y1[2*k] = u1r*w1r - u1i*w1i;  y1[2*k+1] = u1r*w1i + u1i*w1r;
                y2[2*k] = u2r*w2r - u2i*w2i;  y2[2*k+1] = u2r*w2i + u2i*w2r;
                y3[2*k] = u3r*w3r - u3i*w3i;  y3[2*k+1] = u3r*w3i + u3i*w3r;
            )
        }
#undef R4_BODY
    }
}

/* Radix-8 = two radix-4 halves (even/odd taps) + w8 combine; c = 1/sqrt(2).
 * j == 0 (unit twiddles -- includes the WHOLE final l==1 stage) skips the
 * twiddle multiplies. */
static void pass_r8(int l, int m, const double *restrict s, double *restrict d,
                    const double *restrict tw, double sg)
{
    const double c = 0.70710678118654752440;
    for (int j = 0; j < l; ++j) {
        const double *restrict x0 = s + 2 * (size_t)m * j;
        const double *restrict x1 = x0 + 2 * (size_t)m * l;
        const double *restrict x2 = x1 + 2 * (size_t)m * l;
        const double *restrict x3 = x2 + 2 * (size_t)m * l;
        const double *restrict x4 = x3 + 2 * (size_t)m * l;
        const double *restrict x5 = x4 + 2 * (size_t)m * l;
        const double *restrict x6 = x5 + 2 * (size_t)m * l;
        const double *restrict x7 = x6 + 2 * (size_t)m * l;
        double *restrict y0 = d + 2 * (size_t)m * (size_t)(8 * j);
        double *restrict y1 = y0 + 2 * (size_t)m;
        double *restrict y2 = y1 + 2 * (size_t)m;
        double *restrict y3 = y2 + 2 * (size_t)m;
        double *restrict y4 = y3 + 2 * (size_t)m;
        double *restrict y5 = y4 + 2 * (size_t)m;
        double *restrict y6 = y5 + 2 * (size_t)m;
        double *restrict y7 = y6 + 2 * (size_t)m;
        const double *w = tw + 2 * (size_t)(8 * j);
#define R8_BODY(TWIDDLE)                                                          \
        for (int k = 0; k < m; ++k) {                                             \
            double t0r=x0[2*k],t0i=x0[2*k+1], t1r=x1[2*k],t1i=x1[2*k+1];          \
            double t2r=x2[2*k],t2i=x2[2*k+1], t3r=x3[2*k],t3i=x3[2*k+1];          \
            double t4r=x4[2*k],t4i=x4[2*k+1], t5r=x5[2*k],t5i=x5[2*k+1];          \
            double t6r=x6[2*k],t6i=x6[2*k+1], t7r=x7[2*k],t7i=x7[2*k+1];          \
            /* even taps t0,t2,t4,t6 -> E ; odd taps t1,t3,t5,t7 -> O (radix-4) */\
            double ae0r=t0r+t4r, ae0i=t0i+t4i, ae1r=t0r-t4r, ae1i=t0i-t4i;        \
            double ao0r=t2r+t6r, ao0i=t2i+t6i, ao1r=t2r-t6r, ao1i=t2i-t6i;        \
            double E0r=ae0r+ao0r, E0i=ae0i+ao0i;                                  \
            double E1r=ae1r-sg*ao1i, E1i=ae1i+sg*ao1r;                            \
            double E2r=ae0r-ao0r, E2i=ae0i-ao0i;                                  \
            double E3r=ae1r+sg*ao1i, E3i=ae1i-sg*ao1r;                            \
            double be0r=t1r+t5r, be0i=t1i+t5i, be1r=t1r-t5r, be1i=t1i-t5i;        \
            double bo0r=t3r+t7r, bo0i=t3i+t7i, bo1r=t3r-t7r, bo1i=t3i-t7i;        \
            double O0r=be0r+bo0r, O0i=be0i+bo0i;                                  \
            double O1r=be1r-sg*bo1i, O1i=be1i+sg*bo1r;                            \
            double O2r=be0r-bo0r, O2i=be0i-bo0i;                                  \
            double O3r=be1r+sg*bo1i, O3i=be1i-sg*bo1r;                            \
            /* a_s = w8^s O_s : w8 = c(1+sg i), w8^2 = sg i, w8^3 = -c(1-sg i) */ \
            double a1r=c*(O1r - sg*O1i), a1i=c*(O1i + sg*O1r);                    \
            double a2r=-sg*O2i,          a2i= sg*O2r;                             \
            double a3r=-c*(O3r + sg*O3i), a3i=-c*(O3i - sg*O3r);                  \
            double u0r=E0r+O0r,u0i=E0i+O0i, u4r=E0r-O0r,u4i=E0i-O0i;              \
            double u1r=E1r+a1r,u1i=E1i+a1i, u5r=E1r-a1r,u5i=E1i-a1i;              \
            double u2r=E2r+a2r,u2i=E2i+a2i, u6r=E2r-a2r,u6i=E2i-a2i;              \
            double u3r=E3r+a3r,u3i=E3i+a3i, u7r=E3r-a3r,u7i=E3i-a3i;              \
            TWIDDLE                                                               \
        }
        if (j == 0) {
            R8_BODY(
                y0[2*k]=u0r; y0[2*k+1]=u0i;  y1[2*k]=u1r; y1[2*k+1]=u1i;
                y2[2*k]=u2r; y2[2*k+1]=u2i;  y3[2*k]=u3r; y3[2*k+1]=u3i;
                y4[2*k]=u4r; y4[2*k+1]=u4i;  y5[2*k]=u5r; y5[2*k+1]=u5i;
                y6[2*k]=u6r; y6[2*k+1]=u6i;  y7[2*k]=u7r; y7[2*k+1]=u7i;
            )
        } else {
            double w1r=w[2],w1i=w[3],w2r=w[4],w2i=w[5],w3r=w[6],w3i=w[7];
            double w4r=w[8],w4i=w[9],w5r=w[10],w5i=w[11],w6r=w[12],w6i=w[13];
            double w7r=w[14],w7i=w[15];
            R8_BODY(
                y0[2*k]=u0r;                       y0[2*k+1]=u0i;
                y1[2*k]=u1r*w1r-u1i*w1i;           y1[2*k+1]=u1r*w1i+u1i*w1r;
                y2[2*k]=u2r*w2r-u2i*w2i;           y2[2*k+1]=u2r*w2i+u2i*w2r;
                y3[2*k]=u3r*w3r-u3i*w3i;           y3[2*k+1]=u3r*w3i+u3i*w3r;
                y4[2*k]=u4r*w4r-u4i*w4i;           y4[2*k+1]=u4r*w4i+u4i*w4r;
                y5[2*k]=u5r*w5r-u5i*w5i;           y5[2*k+1]=u5r*w5i+u5i*w5r;
                y6[2*k]=u6r*w6r-u6i*w6i;           y6[2*k+1]=u6r*w6i+u6i*w6r;
                y7[2*k]=u7r*w7r-u7i*w7i;           y7[2*k+1]=u7r*w7i+u7i*w7r;
            )
        }
#undef R8_BODY
    }
}

/* Generic radix r (odd prime <= 61), O(r^2) per butterfly.
 * m == 1: dense r-point DFT per block, twiddle applied after the sum.
 * m  > 1: fold the twiddle into the r row weights once per (j,s), then stream over the
 *         m-contiguous rows accumulating into the destination row (vectorizable). */
static void pass_gen(int r, int l, int m, const double *restrict s, double *restrict d,
                     const double *restrict tw, const double *restrict W)
{
    if (m == 1) {
        /* dense r-point DFT per block, conjugate-pair split: rows s and r-s of the
         * DFT matrix are conjugates, so the four real products (ac,bd,ad,bc) are
         * shared and the multiply count halves.  s=0 needs no multiplies at all. */
        for (int j = 0; j < l; ++j) {
            double s0r = 0.0, s0i = 0.0;
            for (int i = 0; i < r; ++i) {
                s0r += s[2 * ((size_t)j + (size_t)l * i)];
                s0i += s[2 * ((size_t)j + (size_t)l * i) + 1];
            }
            d[2 * ((size_t)r * j)]     = s0r;   /* tw[j*r+0] == 1 always */
            d[2 * ((size_t)r * j) + 1] = s0i;
            for (int ss = 1; 2 * ss <= r; ++ss) {
                const double *Wrow = W + 2 * (size_t)ss * r;
                double p1 = 0.0, p2 = 0.0, p3 = 0.0, p4 = 0.0;
                for (int i = 0; i < r; ++i) {
                    double a = s[2 * ((size_t)j + (size_t)l * i)];
                    double b = s[2 * ((size_t)j + (size_t)l * i) + 1];
                    double c = Wrow[2*i], e = Wrow[2*i+1];
                    p1 += a * c; p2 += b * e; p3 += a * e; p4 += b * c;
                }
                double u1r = p1 - p2, u1i = p3 + p4;      /* X[ss]   */
                double u2r = p1 + p2, u2i = p4 - p3;      /* X[r-ss] */
                double twr = tw[2 * ((size_t)j * r + ss)];
                double twi = tw[2 * ((size_t)j * r + ss) + 1];
                d[2 * ((size_t)r * j + ss)]     = u1r * twr - u1i * twi;
                d[2 * ((size_t)r * j + ss) + 1] = u1r * twi + u1i * twr;
                int s2 = r - ss;
                twr = tw[2 * ((size_t)j * r + s2)];
                twi = tw[2 * ((size_t)j * r + s2) + 1];
                d[2 * ((size_t)r * j + s2)]     = u2r * twr - u2i * twi;
                d[2 * ((size_t)r * j + s2) + 1] = u2r * twi + u2i * twr;
            }
        }
        return;
    }
    double Wf[2 * GENR_MAX];
    for (int j = 0; j < l; ++j) {
        for (int ss = 0; ss < r; ++ss) {
            double twr = tw[2 * ((size_t)j * r + ss)];
            double twi = tw[2 * ((size_t)j * r + ss) + 1];
            const double *Wrow = W + 2 * (size_t)ss * r;
            for (int i = 0; i < r; ++i) {
                double wr = Wrow[2*i], wi = Wrow[2*i+1];
                Wf[2*i]   = wr * twr - wi * twi;
                Wf[2*i+1] = wr * twi + wi * twr;
            }
            double *y = d + 2 * (size_t)m * ((size_t)r * j + ss);
            const double *x0 = s + 2 * (size_t)m * j;
            double w0r = Wf[0], w0i = Wf[1];
            for (int k = 0; k < m; ++k) {
                y[2*k]   = x0[2*k] * w0r - x0[2*k+1] * w0i;
                y[2*k+1] = x0[2*k] * w0i + x0[2*k+1] * w0r;
            }
            for (int i = 1; i < r; ++i) {
                const double *x = s + 2 * (size_t)m * ((size_t)j + (size_t)l * i);
                double wr = Wf[2*i], wi = Wf[2*i+1];
                for (int k = 0; k < m; ++k) {
                    y[2*k]   += x[2*k] * wr - x[2*k+1] * wi;
                    y[2*k+1] += x[2*k] * wi + x[2*k+1] * wr;
                }
            }
        }
    }
}

/* Out-of-place execute; scratch must hold n complex. in/out/scratch must be distinct
 * (in==out is NOT supported). Result always lands in out. */
static void mr_exec(const mr_plan *P, const double *restrict in, double *restrict out,
                    double *restrict scratch)
{
    int n = P->n;
    if (P->nf == 0) {                    /* n == 1 */
        memcpy(out, in, 2 * (size_t)n * sizeof(double));
        return;
    }
    int l = n, m = 1;
    const double *src = in;
    double sg = (double)P->sign;
    for (int q = 0; q < P->nf; ++q) {
        int r = P->radix[q]; l /= r;
        double *dst = (((P->nf - 1 - q) & 1) == 0) ? out : scratch;
        const double *tw = P->tw + 2 * P->twoff[q];
        switch (r) {
        case 2:  pass_r2(l, m, src, dst, tw);            break;
        case 3:  pass_r3(l, m, src, dst, tw, sg);        break;
        case 4:  pass_r4(l, m, src, dst, tw, sg);        break;
        case 8:  pass_r8(l, m, src, dst, tw, sg);        break;
        default: pass_gen(r, l, m, src, dst, tw, P->wmat[q]); break;
        }
        src = dst; m *= r;
    }
}

/* ============================ number theory bits ============================ */

static long modpow(long b, long e, long mod)
{
    long r = 1; b %= mod;
    while (e) { if (e & 1) r = r * b % mod; b = b * b % mod; e >>= 1; }
    return r;
}

static int is_prime(int n)
{
    if (n < 2) return 0;
    if (n % 2 == 0) return n == 2;
    for (long d = 3; d * d <= n; d += 2) if (n % d == 0) return 0;
    return 1;
}

static int primitive_root(int p)
{
    int pf[16], npf = 0, t = p - 1;
    for (int d = 2; (long)d * d <= t; d = (d == 2) ? 3 : d + 2)
        if (t % d == 0) { pf[npf++] = d; while (t % d == 0) t /= d; }
    if (t > 1) pf[npf++] = t;
    for (int g = 2;; ++g) {
        int ok = 1;
        for (int i = 0; i < npf; ++i)
            if (modpow(g, (p - 1) / pf[i], p) == 1) { ok = 0; break; }
        if (ok) return g;
    }
}

/* Padded-convolution length: smallest cheap smooth M >= need among {1,3,5,9,15}*2^k,
 * ranked by a per-point pass-cost model (radix-4 pipeline is the cheap path). */
static double mr_cost(int n)
{
    int radix[MAXF], lo;
    int nf = factorize(n, radix, &lo);
    if (lo != 1) return 1e300;
    double c = 0.0;
    for (int q = 0; q < nf; ++q) {
        int r = radix[q];
        /* radix-8 saves sweeps only once the working set leaves L2; below that the
         * spill-heavy kernel is slower per point than radix-4 (measured on wallaby) */
        c += (r == 8) ? (n >= 131072 ? 0.75 : 1.35) : (r == 4) ? 1.0 : (r == 2) ? 1.05
           : (r == 3) ? 1.5 : (r == 5) ? 2.1 : 0.45 * r;
    }
    return c * (double)n;
}

static int choose_conv_len(int need)
{
    static const int bases[] = { 1, 3, 5, 9, 15 };
    int best = 0; double bestc = 1e300;
    for (unsigned bi = 0; bi < sizeof bases / sizeof *bases; ++bi) {
        long M = bases[bi];
        while (M < need) M <<= 1;
        if (M > (1L << 28)) continue;
        double c = mr_cost((int)M);
        if (c < bestc) { bestc = c; best = (int)M; }
    }
    return best;
}

/* ===================== across-batch lanes (survey lever #1) =====================
 * On lane-blocked data T[(e*LANEV + v)] (element e of transform v), every Stockham
 * pass is IDENTICAL to the scalar pass with m -> m*LANEV, because the twiddle of a
 * butterfly depends on (j,s) only, never on the position k inside the m-block.  So
 * the same kernels serve both paths; batched execution costs two transposes per
 * group of LANEV vectors and vectorizes even the m==1 stages (small primes go from
 * scalar dense to 8-lane dense).  Gated to transforms whose laned working set stays
 * cache-resident; big transforms keep the per-vector path (one vector in L2 beats
 * eight vectors in DRAM). */
#define LANEV      8
#ifndef LANE_MAX_N
#define LANE_MAX_N 1023   /* measured: lanes win to ~1K (incl Rader's P=1020 conv, where
                           * the generic-17 stage vectorizes over lanes); at 1024+ the
                           * per-vector k-loops are already wide and lanes only add
                           * transpose cost and 8x footprint (1024: 4.1 vs 5.5 us,
                           * 4096: 20 vs 27 us on wallaby core 100) */
#endif

static void mr_exec_lanes(const mr_plan *P, const double *restrict in,
                          double *restrict out, double *restrict scratch)
{
    int n = P->n;
    if (P->nf == 0) {
        memcpy(out, in, 2 * (size_t)n * LANEV * sizeof(double));
        return;
    }
    int l = n, m = 1;
    const double *src = in;
    double sg = (double)P->sign;
    for (int q = 0; q < P->nf; ++q) {
        int r = P->radix[q]; l /= r;
        double *dst = (((P->nf - 1 - q) & 1) == 0) ? out : scratch;
        const double *tw = P->tw + 2 * P->twoff[q];
        int mv = m * LANEV;
        switch (r) {
        case 2:  pass_r2(l, mv, src, dst, tw);            break;
        case 3:  pass_r3(l, mv, src, dst, tw, sg);        break;
        case 4:  pass_r4(l, mv, src, dst, tw, sg);        break;
        case 8:  pass_r8(l, mv, src, dst, tw, sg);        break;
        default: pass_gen(r, l, mv, src, dst, tw, P->wmat[q]); break;
        }
        src = dst; m *= r;
    }
}

/* batch-major x (vector stride vs doubles) -> lane-blocked T, and back */
static void lane_gather(const double *restrict x, size_t vs, int n, double *restrict T)
{
    for (int v = 0; v < LANEV; ++v) {
        const double *xv = x + (size_t)v * vs;
        for (int e = 0; e < n; ++e) {
            T[2 * ((size_t)e * LANEV + v)]     = xv[2*e];
            T[2 * ((size_t)e * LANEV + v) + 1] = xv[2*e + 1];
        }
    }
}

static void lane_scatter(const double *restrict T, size_t vs, int n, double *restrict y)
{
    for (int v = 0; v < LANEV; ++v) {
        double *yv = y + (size_t)v * vs;
        for (int e = 0; e < n; ++e) {
            yv[2*e]     = T[2 * ((size_t)e * LANEV + v)];
            yv[2*e + 1] = T[2 * ((size_t)e * LANEV + v) + 1];
        }
    }
}

/* ============================== the top plan ============================== */

struct fft1d_plan {
    int L, batch;
    int kind;                 /* 0 mixed-radix, 1 Rader, 2 Bluestein */
    mr_plan fwd, inv;         /* inv only built for kinds 1,2 */
    /* Rader (kind 1): P = L-1 */
    int P;
    int *qin, *qout;          /* qin[q] = g^q mod L ; qout[t] = g^{-t} mod L */
    /* Bluestein (kind 2): M = conv length, chirp[k] = exp(-i pi k^2 / L), k < L */
    int M;
    double *chirp;
    double *Bspec;            /* FFT(b)/conv_len, folded normalization (kinds 1,2) */
    double *ba, *bb, *bc;     /* work buffers, conv length complex each */
    int laneN;                /* lane path transform length (0 = lanes disabled) */
    double *la, *lb, *lc;     /* lane buffers, laneN*LANEV complex each */
};

const char *fft1d_name(void) { return "d1_planner"; }
const char *fft1d_description(void)
{
    return "LIBRARY LAYER (adoption-scored): factorization planner -- mixed-radix "
           "Stockham / unpadded Rader (smooth L-1) / smooth-padded Bluestein";
}

int fft1d_supports(int L) { return L >= 1; }

static void spectrum_fold(double *Bspec, const double *bb, int n)
{
    double s = 1.0 / (double)n;
    for (int q = 0; q < n; ++q) { Bspec[2*q] = bb[2*q] * s; Bspec[2*q+1] = bb[2*q+1] * s; }
}

/* enable lanes iff the batch can fill groups and the laned buffers stay cache-sized */
static int lane_setup(fft1d_plan *p, int n)
{
    if (p->batch < LANEV || n > LANE_MAX_N) return 0;
    p->la = aligned64(2 * (size_t)n * LANEV * sizeof(double));
    p->lb = aligned64(2 * (size_t)n * LANEV * sizeof(double));
    p->lc = aligned64(2 * (size_t)n * LANEV * sizeof(double));
    if (!p->la || !p->lb || !p->lc) return -1;
    p->laneN = n;
    return 0;
}

fft1d_plan *fft1d_create(int L, int batch)
{
    if (L < 1 || batch < 1) return NULL;
    fft1d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L; p->batch = batch;

    int radix[MAXF], leftover;
    factorize(L, radix, &leftover);

    if (leftover == 1) {                                   /* fully smooth */
        p->kind = 0;
        if (mr_build(&p->fwd, L, -1) != 0) goto fail;
        p->bc = aligned64(2 * (size_t)L * sizeof(double));
        if (!p->bc) goto fail;
        if (lane_setup(p, L) != 0) goto fail;
        return p;
    }

    if (leftover == L && is_prime(L)) {                    /* prime: try Rader */
        int r2[MAXF], lo2;
        factorize(L - 1, r2, &lo2);
        if (lo2 == 1) {
            p->kind = 1;
            int P = p->P = L - 1;
            if (mr_build(&p->fwd, P, -1) != 0) goto fail;
            if (mr_build(&p->inv, P, +1) != 0) goto fail;
            p->qin  = malloc((size_t)P * sizeof(int));
            p->qout = malloc((size_t)P * sizeof(int));
            p->Bspec = aligned64(2 * (size_t)P * sizeof(double));
            p->ba = aligned64(2 * (size_t)P * sizeof(double));
            p->bb = aligned64(2 * (size_t)P * sizeof(double));
            p->bc = aligned64(2 * (size_t)P * sizeof(double));
            if (!p->qin || !p->qout || !p->Bspec || !p->ba || !p->bb || !p->bc) goto fail;
            long g = primitive_root(L), gi = modpow(g, L - 2, L);
            long a = 1, b = 1;
            for (int q = 0; q < P; ++q) {
                p->qin[q]  = (int)a;
                p->qout[q] = (int)b;
                a = a * g % L;  b = b * gi % L;
            }
            /* b_t = w_L^{qout[t]} = exp(-2 pi i qout[t] / L) */
            for (int t = 0; t < P; ++t) {
                double th = -2.0 * M_PI * (double)p->qout[t] / (double)L;
                p->ba[2*t] = cos(th); p->ba[2*t+1] = sin(th);
            }
            mr_exec(&p->fwd, p->ba, p->bb, p->bc);
            spectrum_fold(p->Bspec, p->bb, P);
            if (lane_setup(p, P) != 0) goto fail;
            return p;
        }
    }

    /* Bluestein: any L (prime with awkward L-1, or composite with a big prime) */
    p->kind = 2;
    {
        int M = p->M = choose_conv_len(2 * L - 1);
        if (M <= 0) goto fail;
        if (mr_build(&p->fwd, M, -1) != 0) goto fail;
        if (mr_build(&p->inv, M, +1) != 0) goto fail;
        p->chirp = aligned64(2 * (size_t)L * sizeof(double));
        p->Bspec = aligned64(2 * (size_t)M * sizeof(double));
        p->ba = aligned64(2 * (size_t)M * sizeof(double));
        p->bb = aligned64(2 * (size_t)M * sizeof(double));
        p->bc = aligned64(2 * (size_t)M * sizeof(double));
        if (!p->chirp || !p->Bspec || !p->ba || !p->bb || !p->bc) goto fail;
        long twoL = 2L * L;
        for (int k = 0; k < L; ++k) {
            long ph = ((long)k * k) % twoL;               /* exact integer k^2 mod 2L */
            double th = -M_PI * (double)ph / (double)L;   /* = -2 pi ph / (2L) */
            p->chirp[2*k] = cos(th); p->chirp[2*k+1] = sin(th);
        }
        /* b_j = conj(chirp_j) for |j| < L, wrapped cyclically into length M */
        memset(p->ba, 0, 2 * (size_t)M * sizeof(double));
        for (int j = 0; j < L; ++j) {
            p->ba[2*j]   = p->chirp[2*j];
            p->ba[2*j+1] = -p->chirp[2*j+1];
            if (j > 0) {
                p->ba[2*(M-j)]   = p->chirp[2*j];
                p->ba[2*(M-j)+1] = -p->chirp[2*j+1];
            }
        }
        mr_exec(&p->fwd, p->ba, p->bb, p->bc);
        spectrum_fold(p->Bspec, p->bb, M);
        if (lane_setup(p, M) != 0) goto fail;
        return p;
    }

fail:
    fft1d_destroy(p);
    return NULL;
}

static void pointwise_mul(double *restrict a, const double *restrict b, int n)
{
    for (int q = 0; q < n; ++q) {
        double ar = a[2*q], ai = a[2*q+1], br = b[2*q], bi = b[2*q+1];
        a[2*q]   = ar * br - ai * bi;
        a[2*q+1] = ar * bi + ai * br;
    }
}

static void rader_exec(fft1d_plan *p, const double *restrict x, double *restrict y)
{
    int P = p->P;
    double sr = x[0], si = x[1];
    for (int q = 0; q < P; ++q) {
        int j = p->qin[q];
        double xr = x[2*j], xi = x[2*j+1];
        p->ba[2*q] = xr; p->ba[2*q+1] = xi;
        sr += xr; si += xi;
    }
    mr_exec(&p->fwd, p->ba, p->bb, p->bc);
    pointwise_mul(p->bb, p->Bspec, P);
    mr_exec(&p->inv, p->bb, p->ba, p->bc);
    double x0r = x[0], x0i = x[1];
    y[0] = sr; y[1] = si;
    for (int t = 0; t < P; ++t) {
        int k = p->qout[t];
        y[2*k]   = x0r + p->ba[2*t];
        y[2*k+1] = x0i + p->ba[2*t+1];
    }
}

static void blu_exec(fft1d_plan *p, const double *restrict x, double *restrict y)
{
    int L = p->L, M = p->M;
    for (int k = 0; k < L; ++k) {
        double cr = p->chirp[2*k], ci = p->chirp[2*k+1];
        p->ba[2*k]   = x[2*k] * cr - x[2*k+1] * ci;
        p->ba[2*k+1] = x[2*k] * ci + x[2*k+1] * cr;
    }
    memset(p->ba + 2 * (size_t)L, 0, 2 * (size_t)(M - L) * sizeof(double));
    mr_exec(&p->fwd, p->ba, p->bb, p->bc);
    pointwise_mul(p->bb, p->Bspec, M);
    mr_exec(&p->inv, p->bb, p->ba, p->bc);
    for (int k = 0; k < L; ++k) {
        double cr = p->chirp[2*k], ci = p->chirp[2*k+1];
        y[2*k]   = p->ba[2*k] * cr - p->ba[2*k+1] * ci;
        y[2*k+1] = p->ba[2*k] * ci + p->ba[2*k+1] * cr;
    }
}

/* laned pointwise: one (br,bi) broadcast over the LANEV lane block */
static void pointwise_mul_lanes(double *restrict a, const double *restrict b, int n)
{
    for (int q = 0; q < n; ++q) {
        double br = b[2*q], bi = b[2*q+1];
        double *aq = a + 2 * (size_t)q * LANEV;
        for (int v = 0; v < LANEV; ++v) {
            double ar = aq[2*v], ai = aq[2*v+1];
            aq[2*v]   = ar * br - ai * bi;
            aq[2*v+1] = ar * bi + ai * br;
        }
    }
}

static void rader_exec_lanes(fft1d_plan *p, const double *restrict x, double *restrict y)
{
    int P = p->P;
    size_t vs = 2 * (size_t)p->L;
    double sr[LANEV], si[LANEV];
    for (int v = 0; v < LANEV; ++v) { sr[v] = x[v*vs]; si[v] = x[v*vs + 1]; }
    for (int q = 0; q < P; ++q) {
        int j = p->qin[q];
        double *T = p->la + 2 * (size_t)q * LANEV;
        for (int v = 0; v < LANEV; ++v) {
            double xr = x[v*vs + 2*j], xi = x[v*vs + 2*j + 1];
            T[2*v] = xr; T[2*v+1] = xi;
            sr[v] += xr; si[v] += xi;
        }
    }
    mr_exec_lanes(&p->fwd, p->la, p->lb, p->lc);
    pointwise_mul_lanes(p->lb, p->Bspec, P);
    mr_exec_lanes(&p->inv, p->lb, p->la, p->lc);
    for (int v = 0; v < LANEV; ++v) { y[v*vs] = sr[v]; y[v*vs + 1] = si[v]; }
    for (int t = 0; t < P; ++t) {
        int k = p->qout[t];
        const double *T = p->la + 2 * (size_t)t * LANEV;
        for (int v = 0; v < LANEV; ++v) {
            y[v*vs + 2*k]     = x[v*vs]     + T[2*v];
            y[v*vs + 2*k + 1] = x[v*vs + 1] + T[2*v+1];
        }
    }
}

static void blu_exec_lanes(fft1d_plan *p, const double *restrict x, double *restrict y)
{
    int L = p->L, M = p->M;
    size_t vs = 2 * (size_t)L;
    for (int k = 0; k < L; ++k) {
        double cr = p->chirp[2*k], ci = p->chirp[2*k+1];
        double *T = p->la + 2 * (size_t)k * LANEV;
        for (int v = 0; v < LANEV; ++v) {
            double xr = x[v*vs + 2*k], xi = x[v*vs + 2*k + 1];
            T[2*v] = xr * cr - xi * ci;
            T[2*v+1] = xr * ci + xi * cr;
        }
    }
    memset(p->la + 2 * (size_t)L * LANEV, 0,
           2 * (size_t)(M - L) * LANEV * sizeof(double));
    mr_exec_lanes(&p->fwd, p->la, p->lb, p->lc);
    pointwise_mul_lanes(p->lb, p->Bspec, M);
    mr_exec_lanes(&p->inv, p->lb, p->la, p->lc);
    for (int k = 0; k < L; ++k) {
        double cr = p->chirp[2*k], ci = p->chirp[2*k+1];
        const double *T = p->la + 2 * (size_t)k * LANEV;
        for (int v = 0; v < LANEV; ++v) {
            y[v*vs + 2*k]     = T[2*v] * cr - T[2*v+1] * ci;
            y[v*vs + 2*k + 1] = T[2*v] * ci + T[2*v+1] * cr;
        }
    }
}

void fft1d_execute(fft1d_plan *p, const double _Complex *cin, double _Complex *cout)
{
    const double *in = (const double *)cin;
    double *out = (double *)cout;
    size_t stride = 2 * (size_t)p->L;
    int b = 0;
    if (p->laneN) {
        for (; b + LANEV <= p->batch; b += LANEV) {
            const double *x = in + (size_t)b * stride;
            double *y = out + (size_t)b * stride;
            switch (p->kind) {
            case 0:
                lane_gather(x, stride, p->L, p->la);
                mr_exec_lanes(&p->fwd, p->la, p->lb, p->lc);
                lane_scatter(p->lb, stride, p->L, y);
                break;
            case 1:  rader_exec_lanes(p, x, y); break;
            default: blu_exec_lanes(p, x, y);   break;
            }
        }
    }
    for (; b < p->batch; ++b) {
        const double *x = in + (size_t)b * stride;
        double *y = out + (size_t)b * stride;
        switch (p->kind) {
        case 0:  mr_exec(&p->fwd, x, y, p->bc); break;
        case 1:  rader_exec(p, x, y);           break;
        default: blu_exec(p, x, y);             break;
        }
    }
}

void fft1d_destroy(fft1d_plan *p)
{
    if (!p) return;
    mr_free(&p->fwd);
    mr_free(&p->inv);
    free(p->qin); free(p->qout);
    free(p->chirp); free(p->Bspec);
    free(p->ba); free(p->bb); free(p->bc);
    free(p->la); free(p->lb); free(p->lc);
    free(p);
}

/* ============================== self test ============================== */
#ifdef PLANNER_TEST
#include <stdio.h>

static void naive_dft(int n, const double *x, double *y)
{
    for (int k = 0; k < n; ++k) {
        double sr = 0, si = 0;
        for (int j = 0; j < n; ++j) {
            long ph = ((long)j * k) % n;
            double th = -2.0 * M_PI * (double)ph / (double)n;
            double wr = cos(th), wi = sin(th);
            sr += x[2*j] * wr - x[2*j+1] * wi;
            si += x[2*j] * wi + x[2*j+1] * wr;
        }
        y[2*k] = sr; y[2*k+1] = si;
    }
}

int main(void)
{
    /* smooth, single-stage primes, Rader (127,1013,1021,2053), Bluestein (1019,2038,
       10007-lite skipped for speed unless PLANNER_TEST_BIG) */
    int sizes[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 24, 31, 32, 36, 45,
                    60, 61, 64, 100, 120, 127, 128, 240, 243, 256, 343, 510, 512,
                    1000, 1013, 1019, 1020, 1021, 1024, 2038, 2048, 2053,
#ifdef PLANNER_TEST_BIG
                    4096, 10007, 16384,
#endif
    };
    srand(12345);
    int nfail = 0;
    for (unsigned si = 0; si < sizeof sizes / sizeof *sizes; ++si) {
        int n = sizes[si], B = 19;   /* 2 lane groups + remainder of 3 */
        double _Complex *in  = malloc((size_t)n * B * sizeof *in);
        double _Complex *out = malloc((size_t)n * B * sizeof *out);
        double *ref = malloc(2 * (size_t)n * sizeof *ref);
        for (int i = 0; i < n * B; ++i)
            in[i] = (2.0 * rand() / RAND_MAX - 1.0) + (2.0 * rand() / RAND_MAX - 1.0) * I;
        fft1d_plan *p = fft1d_create(n, B);
        if (!p) { printf("n=%5d  CREATE FAILED\n", n); ++nfail; continue; }
        fft1d_execute(p, in, out);
        double worst = 0;
        for (int b = 0; b < B; ++b) {
            naive_dft(n, (const double *)(in + (size_t)b * n), ref);
            double e2 = 0, r2 = 0;
            for (int i = 0; i < n; ++i) {
                double dr = creal(out[(size_t)b*n+i]) - ref[2*i];
                double di = cimag(out[(size_t)b*n+i]) - ref[2*i+1];
                e2 += dr*dr + di*di;
                r2 += ref[2*i]*ref[2*i] + ref[2*i+1]*ref[2*i+1];
            }
            double rel = sqrt(e2 / (r2 > 0 ? r2 : 1));
            if (rel > worst) worst = rel;
        }
        const char *k = p->kind == 0 ? "mixed" : p->kind == 1 ? "rader" : "blues";
        printf("n=%5d  %s  relL2=%.3e  %s\n", n, k, worst, worst < 1e-12 ? "ok" : "FAIL");
        if (worst >= 1e-12) ++nfail;
        fft1d_destroy(p);
        free(in); free(out); free(ref);
    }
    printf(nfail ? "== %d FAILURES ==\n" : "== all ok ==\n", nfail);
    return nfail != 0;
}
#endif
