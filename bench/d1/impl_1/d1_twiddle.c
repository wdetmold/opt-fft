/* d1_twiddle -- LIBRARY LAYER (adoption-scored): exact 1D twiddle tables, consumption order.
 *
 * The product of this entry is the ADOPTION BLOCK below: copy it whole into your entry
 * (and say so in your strategy record). It exists because the accuracy vein of the 1D
 * survey converged on one rule -- "inaccurate twiddles are the leading cause of FFT
 * inaccuracy": generate tables in the PLAN stage from exactly-reduced arguments, never
 * from in-loop recurrences (whose error grows O(sqrt N)..O(N^2)).
 *
 * The FFT underneath is the demonstration vehicle, not the point: a mixed-radix
 * (2/3/4/5/8) Stockham autosort FFT (no bit-reversal) whose per-stage tables are laid
 * out in EXACTLY the order the butterfly loops consume them -- the inner loops read the
 * table linearly, no index arithmetic, no strided twiddle loads. It supports smooth
 * L = 2^a 3^b 5^c (graded: 32/60/64/128/1024/4096/16384) and exports a fused-map
 * fft1d_chain that applies the graded map state<-(z+c)/(1+|z+c|) inside the final
 * stage's store loop, saving one full read+write sweep per chain step.
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "../fft1d_api.h"

/* ====================== D1TW ADOPTION BLOCK (v1, round d1_r1) ======================
 * Exact twiddles: w = exp(-2*pi*i * num/den) to ~1 ulp, for ANY int64 num (negative
 * fine), den in [1, 2^59). Two exact reductions before any floating trig:
 *   1. integer:   r = num mod den            (the index never touches fp)
 *   2. quadrant:  q = round(4r/den), s = 4r - q*den   (EXACT in int64)
 * so the trig argument is (pi/2)*(s/den) with |s/den| <= 1/2, i.e. |arg| <= pi/4, and
 * the quadrant factors are exactly 0/+-1 (no sqrt(2)/2 rounding). The argument itself
 * is built from a two-part pi/2 split with an FMA-recovered product error, so the only
 * inexactness left is one rounding of s/den plus libm's ~0.5 ulp sin/cos.
 *
 * d1tw_chirp: the Bluestein chirp exp(-i*pi*k^2/N) with k^2 reduced mod 2N in INTEGERS
 * first -- the survey's explicit fp64 trap at k ~ 1e5 (k^2/N as a double loses the low
 * bits that the periodic reduction needs). d1tw_stage: Stockham stage tables in
 * consumption order (see the pass kernels below for the loop that consumes them).
 * =================================================================================== */
static double _Complex d1tw_cexp(int64_t num, int64_t den)
{
    int64_t r = num % den; if (r < 0) r += den;
    int64_t q = (8*r + den) / (2*den);   /* round(4r/den), in {0..4} */
    int64_t s = 4*r - q*den;             /* |s| <= den/2, exact */
    double  t = (double)s / (double)den; /* one rounding, |t| <= 1/2 */
    static const double PIO2_HI = 1.5707963267948966;      /* 0x1.921fb54442d18p+0 */
    static const double PIO2_LO = 6.123233995736766e-17;   /* pi/2 - PIO2_HI       */
    double h = t * PIO2_HI;
    double l = fma(t, PIO2_HI, -h) + t * PIO2_LO;  /* exact tail of t*(pi/2) */
    double ch = cos(h), sh = sin(h);
    double cd = ch - l * sh, sd = sh + l * ch;     /* cos/sin of the residual angle */
    double cr, si;
    switch ((int)(q & 3)) {              /* rotate by q quarter turns, exactly */
        case 0:  cr =  cd; si =  sd; break;
        case 1:  cr = -sd; si =  cd; break;
        case 2:  cr = -cd; si = -sd; break;
        default: cr =  sd; si = -cd; break;
    }
    return CMPLX(cr, -si);               /* exp(-i*theta) = cos - i sin */
}

/* Bluestein chirp, forward sign: w[k] = exp(-i*pi*k^2/N), k = 0..n-1 (pass n = the
 * padded copy length you need, typically N and then the wrapped tail). k^2 mod 2N in
 * integers BEFORE the trig call. Valid for N < ~3e9 (k^2 fits int64 pre-reduction). */
static void d1tw_chirp(int64_t N, int64_t n, double _Complex *w)
{
    int64_t twoN = 2 * N;
    for (int64_t k = 0; k < n; ++k)
        w[k] = d1tw_cexp((k * k) % twoN, twoN);
}

/* Stockham stage table in CONSUMPTION ORDER for the mixed-radix DIF pass
 *     y[q + s*(r*p + t)] = sum_i x[q + s*(p + m*i)] * W_r^{i t} * W_n^{p t}
 * (n = current sub-length, m = n/r, t = 0..r-1, p = 0..m-1, q = 0..s-1).
 * Entries: for p ascending, t = 1..r-1:  tw[p*(r-1) + t-1] = exp(-2*pi*i * p*t / n).
 * The pass reads this strictly linearly. The LAST stage of a plan has p = 0 only, so
 * its table is identically 1 -- skip it (the last-pass kernels below do). */
static void d1tw_stage(int n, int r, double _Complex *tw)
{
    int m = n / r; size_t idx = 0;
    for (int p = 0; p < m; ++p)
        for (int t = 1; t < r; ++t)
            tw[idx++] = d1tw_cexp((int64_t)p * t, n);
}
/* ==================== end D1TW ADOPTION BLOCK ==================== */

const char *fft1d_name(void) { return "d1_twiddle"; }
const char *fft1d_description(void)
{
    return "LIBRARY LAYER (adoption-scored): exact 1D twiddle tables, consumption order "
           "(d1tw_cexp quadrant-exact, d1tw_chirp integer-reduced, d1tw_stage Stockham "
           "tables); vehicle: mixed-radix 2/3/4/5/8 Stockham, fused-map chain, smooth L";
}

int fft1d_supports(int L)
{
    if (L < 2 || L > (1 << 20)) return 0;
    while (L % 2 == 0) L /= 2;
    while (L % 3 == 0) L /= 3;
    while (L % 5 == 0) L /= 5;
    return L == 1;
}

#define D1TW_MAXF 24
struct fft1d_plan {
    int L, batch, nf;
    int fac[D1TW_MAXF];
    double *tw[D1TW_MAXF];   /* per-stage consumption-order tables (interleaved re,im) */
    double *s0, *s1;         /* ping-pong scratch, L complex each */
};

/* ---- general pass kernels (not the last stage): twiddled, distinct buffers ----
 * All operate on interleaved doubles. tw is read LINEARLY: (r-1) complex per p. */
static void pass2(const double *restrict x, double *restrict y, int m, int s,
                  const double *restrict tw)
{
    for (int p = 0; p < m; ++p) {
        double wr = tw[2*p], wi = tw[2*p+1];
        const double *xa = x + 2*(size_t)s*p, *xb = x + 2*(size_t)s*(p+m);
        double *y0 = y + 2*(size_t)s*(2*p), *y1 = y0 + 2*(size_t)s;
        for (int q = 0; q < s; ++q) {
            double ar = xa[2*q], ai = xa[2*q+1], br = xb[2*q], bi = xb[2*q+1];
            y0[2*q] = ar + br; y0[2*q+1] = ai + bi;
            double dr = ar - br, di = ai - bi;
            y1[2*q] = wr*dr - wi*di; y1[2*q+1] = wr*di + wi*dr;
        }
    }
}

static void pass3(const double *restrict x, double *restrict y, int m, int s,
                  const double *restrict tw)
{
    const double K3 = -0.8660254037844386;  /* sin(-2pi/3) */
    for (int p = 0; p < m; ++p) {
        double w1r = tw[4*p], w1i = tw[4*p+1], w2r = tw[4*p+2], w2i = tw[4*p+3];
        const double *xa = x + 2*(size_t)s*p, *xb = xa + 2*(size_t)s*m, *xc = xb + 2*(size_t)s*m;
        double *y0 = y + 2*(size_t)s*(3*p), *y1 = y0 + 2*(size_t)s, *y2 = y1 + 2*(size_t)s;
        for (int q = 0; q < s; ++q) {
            double ar = xa[2*q], ai = xa[2*q+1];
            double br = xb[2*q], bi = xb[2*q+1];
            double cr = xc[2*q], ci = xc[2*q+1];
            double t1r = br + cr, t1i = bi + ci;
            double ur = ar - 0.5*t1r, ui = ai - 0.5*t1i;
            double vr = K3*(br - cr), vi = K3*(bi - ci);
            y0[2*q] = ar + t1r; y0[2*q+1] = ai + t1i;
            double p1r = ur - vi, p1i = ui + vr;   /* u + i v */
            double p2r = ur + vi, p2i = ui - vr;   /* u - i v */
            y1[2*q] = w1r*p1r - w1i*p1i; y1[2*q+1] = w1r*p1i + w1i*p1r;
            y2[2*q] = w2r*p2r - w2i*p2i; y2[2*q+1] = w2r*p2i + w2i*p2r;
        }
    }
}

static void pass4(const double *restrict x, double *restrict y, int m, int s,
                  const double *restrict tw)
{
    for (int p = 0; p < m; ++p) {
        double w1r = tw[6*p],   w1i = tw[6*p+1];
        double w2r = tw[6*p+2], w2i = tw[6*p+3];
        double w3r = tw[6*p+4], w3i = tw[6*p+5];
        const double *xa = x + 2*(size_t)s*p;
        const double *xb = xa + 2*(size_t)s*m, *xc = xb + 2*(size_t)s*m, *xd = xc + 2*(size_t)s*m;
        double *y0 = y + 2*(size_t)s*(4*p);
        double *y1 = y0 + 2*(size_t)s, *y2 = y1 + 2*(size_t)s, *y3 = y2 + 2*(size_t)s;
        for (int q = 0; q < s; ++q) {
            double ar = xa[2*q], ai = xa[2*q+1], br = xb[2*q], bi = xb[2*q+1];
            double cr = xc[2*q], ci = xc[2*q+1], dr = xd[2*q], di = xd[2*q+1];
            double t0r = ar + cr, t0i = ai + ci, t1r = ar - cr, t1i = ai - ci;
            double t2r = br + dr, t2i = bi + di, t3r = br - dr, t3i = bi - di;
            y0[2*q] = t0r + t2r; y0[2*q+1] = t0i + t2i;
            double p2r = t0r - t2r, p2i = t0i - t2i;
            double p1r = t1r + t3i, p1i = t1i - t3r;   /* t1 - i t3 */
            double p3r = t1r - t3i, p3i = t1i + t3r;   /* t1 + i t3 */
            y1[2*q] = w1r*p1r - w1i*p1i; y1[2*q+1] = w1r*p1i + w1i*p1r;
            y2[2*q] = w2r*p2r - w2i*p2i; y2[2*q+1] = w2r*p2i + w2i*p2r;
            y3[2*q] = w3r*p3r - w3i*p3i; y3[2*q+1] = w3r*p3i + w3i*p3r;
        }
    }
}

static void pass5(const double *restrict x, double *restrict y, int m, int s,
                  const double *restrict tw)
{
    const double C1 = 0.30901699437494745, S1 = 0.9510565162951535;  /* cos,sin 2pi/5 */
    const double C2 = -0.8090169943749475, S2 = 0.5877852522924731;  /* cos,sin 4pi/5 */
    for (int p = 0; p < m; ++p) {
        const double *tp = tw + 8*(size_t)p;
        const double *xa = x + 2*(size_t)s*p;
        const double *xb = xa + 2*(size_t)s*m, *xc = xb + 2*(size_t)s*m;
        const double *xd = xc + 2*(size_t)s*m, *xe = xd + 2*(size_t)s*m;
        double *y0 = y + 2*(size_t)s*(5*p);
        double *y1 = y0 + 2*(size_t)s, *y2 = y1 + 2*(size_t)s;
        double *y3 = y2 + 2*(size_t)s, *y4 = y3 + 2*(size_t)s;
        for (int q = 0; q < s; ++q) {
            double ar = xa[2*q], ai = xa[2*q+1];
            double br = xb[2*q], bi = xb[2*q+1], cr = xc[2*q], ci = xc[2*q+1];
            double dr = xd[2*q], di = xd[2*q+1], er = xe[2*q], ei = xe[2*q+1];
            double t1r = br + er, t1i = bi + ei, t2r = cr + dr, t2i = ci + di;
            double t3r = br - er, t3i = bi - ei, t4r = cr - dr, t4i = ci - di;
            y0[2*q] = ar + t1r + t2r; y0[2*q+1] = ai + t1i + t2i;
            double m1r = ar + C1*t1r + C2*t2r, m1i = ai + C1*t1i + C2*t2i;
            double m2r = ar + C2*t1r + C1*t2r, m2i = ai + C2*t1i + C1*t2i;
            double n1r = S1*t3r + S2*t4r, n1i = S1*t3i + S2*t4i;
            double n2r = S2*t3r - S1*t4r, n2i = S2*t3i - S1*t4i;
            double p1r = m1r + n1i, p1i = m1i - n1r;   /* m1 - i n1 */
            double p4r = m1r - n1i, p4i = m1i + n1r;   /* m1 + i n1 */
            double p2r = m2r + n2i, p2i = m2i - n2r;   /* m2 - i n2 */
            double p3r = m2r - n2i, p3i = m2i + n2r;   /* m2 + i n2 */
            y1[2*q] = tp[0]*p1r - tp[1]*p1i; y1[2*q+1] = tp[0]*p1i + tp[1]*p1r;
            y2[2*q] = tp[2]*p2r - tp[3]*p2i; y2[2*q+1] = tp[2]*p2i + tp[3]*p2r;
            y3[2*q] = tp[4]*p3r - tp[5]*p3i; y3[2*q+1] = tp[4]*p3i + tp[5]*p3r;
            y4[2*q] = tp[6]*p4r - tp[7]*p4i; y4[2*q+1] = tp[6]*p4i + tp[7]*p4r;
        }
    }
}

static void pass8(const double *restrict x, double *restrict y, int m, int s,
                  const double *restrict tw)
{
    const double SQ = 0.7071067811865476;  /* sqrt(2)/2 */
    for (int p = 0; p < m; ++p) {
        const double *tp = tw + 14*(size_t)p;
        const double *x0 = x + 2*(size_t)s*p;
        const size_t st = 2*(size_t)s*m;
        const double *x1 = x0+st, *x2 = x1+st, *x3 = x2+st,
                     *x4 = x3+st, *x5 = x4+st, *x6 = x5+st, *x7 = x6+st;
        double *yy = y + 2*(size_t)s*(8*p);
        const size_t so = 2*(size_t)s;
        for (int q = 0; q < s; ++q) {
            double a0r = x0[2*q], a0i = x0[2*q+1], a1r = x1[2*q], a1i = x1[2*q+1];
            double a2r = x2[2*q], a2i = x2[2*q+1], a3r = x3[2*q], a3i = x3[2*q+1];
            double a4r = x4[2*q], a4i = x4[2*q+1], a5r = x5[2*q], a5i = x5[2*q+1];
            double a6r = x6[2*q], a6i = x6[2*q+1], a7r = x7[2*q], a7i = x7[2*q+1];
            /* DFT4 of evens (a0,a2,a4,a6) */
            double e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
            { double u0r=a0r+a4r,u0i=a0i+a4i,u1r=a0r-a4r,u1i=a0i-a4i;
              double u2r=a2r+a6r,u2i=a2i+a6i,u3r=a2r-a6r,u3i=a2i-a6i;
              e0r=u0r+u2r; e0i=u0i+u2i; e2r=u0r-u2r; e2i=u0i-u2i;
              e1r=u1r+u3i; e1i=u1i-u3r; e3r=u1r-u3i; e3i=u1i+u3r; }
            /* DFT4 of odds (a1,a3,a5,a7) */
            double o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
            { double u0r=a1r+a5r,u0i=a1i+a5i,u1r=a1r-a5r,u1i=a1i-a5i;
              double u2r=a3r+a7r,u2i=a3i+a7i,u3r=a3r-a7r,u3i=a3i-a7i;
              o0r=u0r+u2r; o0i=u0i+u2i; o2r=u0r-u2r; o2i=u0i-u2i;
              o1r=u1r+u3i; o1i=u1i-u3r; o3r=u1r-u3i; o3i=u1i+u3r; }
            /* odd twiddles: w8^1 = SQ(1-i), w8^2 = -i, w8^3 = SQ(-1-i) */
            double b1r = SQ*(o1r + o1i), b1i = SQ*(o1i - o1r);
            double b2r = o2i,            b2i = -o2r;
            double b3r = SQ*(o3i - o3r), b3i = -SQ*(o3r + o3i);
            double p0r=e0r+o0r,p0i=e0i+o0i, p4r=e0r-o0r,p4i=e0i-o0i;
            double p1r=e1r+b1r,p1i=e1i+b1i, p5r=e1r-b1r,p5i=e1i-b1i;
            double p2r=e2r+b2r,p2i=e2i+b2i, p6r=e2r-b2r,p6i=e2i-b2i;
            double p3r=e3r+b3r,p3i=e3i+b3i, p7r=e3r-b3r,p7i=e3i-b3i;
            yy[2*q] = p0r; yy[2*q+1] = p0i;
            double *yt = yy + so;
            yt[2*q] = tp[0]*p1r - tp[1]*p1i;  yt[2*q+1] = tp[0]*p1i + tp[1]*p1r;  yt += so;
            yt[2*q] = tp[2]*p2r - tp[3]*p2i;  yt[2*q+1] = tp[2]*p2i + tp[3]*p2r;  yt += so;
            yt[2*q] = tp[4]*p3r - tp[5]*p3i;  yt[2*q+1] = tp[4]*p3i + tp[5]*p3r;  yt += so;
            yt[2*q] = tp[6]*p4r - tp[7]*p4i;  yt[2*q+1] = tp[6]*p4i + tp[7]*p4r;  yt += so;
            yt[2*q] = tp[8]*p5r - tp[9]*p5i;  yt[2*q+1] = tp[8]*p5i + tp[9]*p5r;  yt += so;
            yt[2*q] = tp[10]*p6r - tp[11]*p6i; yt[2*q+1] = tp[10]*p6i + tp[11]*p6r; yt += so;
            yt[2*q] = tp[12]*p7r - tp[13]*p7i; yt[2*q+1] = tp[12]*p7i + tp[13]*p7r;
        }
    }
}

/* ---- last-stage kernels: p = 0 only (twiddles identically 1, table skipped),
 * optional fused chain map applied at the store:  state = (z+c) / (1+|z+c|).
 * Each kernel is an always_inline body instantiated twice with a CONSTANT domap flag,
 * so the map path and the plain path are both branch-free vectorizable loops (a
 * runtime `if (cm)` per store measurably killed vectorization of the map's sqrt).
 * NO restrict: with a 1-stage plan (L = 2/3/4/5/8) the chain calls these in place,
 * which is safe because each (single) iteration loads everything before storing. */
#define D1TW_ST(o, vr, vi)                                                    \
    do {                                                                      \
        size_t o_ = (size_t)(o);                                              \
        if (domap) {                                                          \
            double zr_ = (vr) + cm[2*o_], zi_ = (vi) + cm[2*o_+1];            \
            double sc_ = 1.0 / (1.0 + sqrt(zr_*zr_ + zi_*zi_));               \
            y[2*o_] = zr_ * sc_; y[2*o_+1] = zi_ * sc_;                       \
        } else { y[2*o_] = (vr); y[2*o_+1] = (vi); }                          \
    } while (0)

#define D1TW_INLINE static inline __attribute__((always_inline)) void

D1TW_INLINE last2_impl(const double *x, double *y, int s, const double *cm, const int domap)
{
    for (int q = 0; q < s; ++q) {
        double ar = x[2*q], ai = x[2*q+1], br = x[2*(q+s)], bi = x[2*(q+s)+1];
        D1TW_ST(q,     ar + br, ai + bi);
        D1TW_ST(q + s, ar - br, ai - bi);
    }
}

D1TW_INLINE last3_impl(const double *x, double *y, int s, const double *cm, const int domap)
{
    const double K3 = -0.8660254037844386;
    for (int q = 0; q < s; ++q) {
        double ar = x[2*q], ai = x[2*q+1];
        double br = x[2*(q+s)], bi = x[2*(q+s)+1];
        double cr = x[2*(q+2*s)], ci = x[2*(q+2*s)+1];
        double t1r = br + cr, t1i = bi + ci;
        double ur = ar - 0.5*t1r, ui = ai - 0.5*t1i;
        double vr = K3*(br - cr), vi = K3*(bi - ci);
        D1TW_ST(q,       ar + t1r, ai + t1i);
        D1TW_ST(q + s,   ur - vi,  ui + vr);
        D1TW_ST(q + 2*s, ur + vi,  ui - vr);
    }
}

D1TW_INLINE last4_impl(const double *x, double *y, int s, const double *cm, const int domap)
{
    for (int q = 0; q < s; ++q) {
        double ar = x[2*q], ai = x[2*q+1];
        double br = x[2*(q+s)], bi = x[2*(q+s)+1];
        double cr = x[2*(q+2*s)], ci = x[2*(q+2*s)+1];
        double dr = x[2*(q+3*s)], di = x[2*(q+3*s)+1];
        double t0r = ar + cr, t0i = ai + ci, t1r = ar - cr, t1i = ai - ci;
        double t2r = br + dr, t2i = bi + di, t3r = br - dr, t3i = bi - di;
        D1TW_ST(q,       t0r + t2r, t0i + t2i);
        D1TW_ST(q + s,   t1r + t3i, t1i - t3r);
        D1TW_ST(q + 2*s, t0r - t2r, t0i - t2i);
        D1TW_ST(q + 3*s, t1r - t3i, t1i + t3r);
    }
}

D1TW_INLINE last5_impl(const double *x, double *y, int s, const double *cm, const int domap)
{
    const double C1 = 0.30901699437494745, S1 = 0.9510565162951535;
    const double C2 = -0.8090169943749475, S2 = 0.5877852522924731;
    for (int q = 0; q < s; ++q) {
        double ar = x[2*q], ai = x[2*q+1];
        double br = x[2*(q+s)], bi = x[2*(q+s)+1];
        double cr = x[2*(q+2*s)], ci = x[2*(q+2*s)+1];
        double dr = x[2*(q+3*s)], di = x[2*(q+3*s)+1];
        double er = x[2*(q+4*s)], ei = x[2*(q+4*s)+1];
        double t1r = br + er, t1i = bi + ei, t2r = cr + dr, t2i = ci + di;
        double t3r = br - er, t3i = bi - ei, t4r = cr - dr, t4i = ci - di;
        double m1r = ar + C1*t1r + C2*t2r, m1i = ai + C1*t1i + C2*t2i;
        double m2r = ar + C2*t1r + C1*t2r, m2i = ai + C2*t1i + C1*t2i;
        double n1r = S1*t3r + S2*t4r, n1i = S1*t3i + S2*t4i;
        double n2r = S2*t3r - S1*t4r, n2i = S2*t3i - S1*t4i;
        D1TW_ST(q,       ar + t1r + t2r, ai + t1i + t2i);
        D1TW_ST(q + s,   m1r + n1i, m1i - n1r);
        D1TW_ST(q + 2*s, m2r + n2i, m2i - n2r);
        D1TW_ST(q + 3*s, m2r - n2i, m2i + n2r);
        D1TW_ST(q + 4*s, m1r - n1i, m1i + n1r);
    }
}

D1TW_INLINE last8_impl(const double *x, double *y, int s, const double *cm, const int domap)
{
    const double SQ = 0.7071067811865476;
    for (int q = 0; q < s; ++q) {
        double a0r = x[2*q], a0i = x[2*q+1];
        double a1r = x[2*(q+s)], a1i = x[2*(q+s)+1];
        double a2r = x[2*(q+2*s)], a2i = x[2*(q+2*s)+1];
        double a3r = x[2*(q+3*s)], a3i = x[2*(q+3*s)+1];
        double a4r = x[2*(q+4*s)], a4i = x[2*(q+4*s)+1];
        double a5r = x[2*(q+5*s)], a5i = x[2*(q+5*s)+1];
        double a6r = x[2*(q+6*s)], a6i = x[2*(q+6*s)+1];
        double a7r = x[2*(q+7*s)], a7i = x[2*(q+7*s)+1];
        double e0r,e0i,e1r,e1i,e2r,e2i,e3r,e3i;
        { double u0r=a0r+a4r,u0i=a0i+a4i,u1r=a0r-a4r,u1i=a0i-a4i;
          double u2r=a2r+a6r,u2i=a2i+a6i,u3r=a2r-a6r,u3i=a2i-a6i;
          e0r=u0r+u2r; e0i=u0i+u2i; e2r=u0r-u2r; e2i=u0i-u2i;
          e1r=u1r+u3i; e1i=u1i-u3r; e3r=u1r-u3i; e3i=u1i+u3r; }
        double o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
        { double u0r=a1r+a5r,u0i=a1i+a5i,u1r=a1r-a5r,u1i=a1i-a5i;
          double u2r=a3r+a7r,u2i=a3i+a7i,u3r=a3r-a7r,u3i=a3i-a7i;
          o0r=u0r+u2r; o0i=u0i+u2i; o2r=u0r-u2r; o2i=u0i-u2i;
          o1r=u1r+u3i; o1i=u1i-u3r; o3r=u1r-u3i; o3i=u1i+u3r; }
        double b1r = SQ*(o1r + o1i), b1i = SQ*(o1i - o1r);
        double b2r = o2i,            b2i = -o2r;
        double b3r = SQ*(o3i - o3r), b3i = -SQ*(o3r + o3i);
        D1TW_ST(q,       e0r + o0r, e0i + o0i);
        D1TW_ST(q + s,   e1r + b1r, e1i + b1i);
        D1TW_ST(q + 2*s, e2r + b2r, e2i + b2i);
        D1TW_ST(q + 3*s, e3r + b3r, e3i + b3i);
        D1TW_ST(q + 4*s, e0r - o0r, e0i - o0i);
        D1TW_ST(q + 5*s, e1r - b1r, e1i - b1i);
        D1TW_ST(q + 6*s, e2r - b2r, e2i - b2i);
        D1TW_ST(q + 7*s, e3r - b3r, e3i - b3i);
    }
}

static void last2p(const double *x, double *y, int s) { last2_impl(x, y, s, NULL, 0); }
static void last2m(const double *x, double *y, int s, const double *cm) { last2_impl(x, y, s, cm, 1); }
static void last3p(const double *x, double *y, int s) { last3_impl(x, y, s, NULL, 0); }
static void last3m(const double *x, double *y, int s, const double *cm) { last3_impl(x, y, s, cm, 1); }
static void last4p(const double *x, double *y, int s) { last4_impl(x, y, s, NULL, 0); }
static void last4m(const double *x, double *y, int s, const double *cm) { last4_impl(x, y, s, cm, 1); }
static void last5p(const double *x, double *y, int s) { last5_impl(x, y, s, NULL, 0); }
static void last5m(const double *x, double *y, int s, const double *cm) { last5_impl(x, y, s, cm, 1); }
static void last8p(const double *x, double *y, int s) { last8_impl(x, y, s, NULL, 0); }
static void last8m(const double *x, double *y, int s, const double *cm) { last8_impl(x, y, s, cm, 1); }

/* One transform: Stockham ping-pong through s0/s1, final stage lands in y.
 * Safe with x == y when nf >= 2 (x is only read in the first pass, y only written
 * in the last) and when nf == 1 (single load-all-then-store-all iteration). */
static void do_fft(const fft1d_plan *pl, const double *x, double *y, const double *cm)
{
    int n = pl->L, s = 1, nf = pl->nf;
    const double *src = x;
    for (int f = 0; f < nf - 1; ++f) {
        int r = pl->fac[f], m = n / r;
        double *dst = (f == 0) ? pl->s0 : (src == pl->s0 ? pl->s1 : pl->s0);
        switch (r) {
            case 2: pass2(src, dst, m, s, pl->tw[f]); break;
            case 3: pass3(src, dst, m, s, pl->tw[f]); break;
            case 4: pass4(src, dst, m, s, pl->tw[f]); break;
            case 5: pass5(src, dst, m, s, pl->tw[f]); break;
            default: pass8(src, dst, m, s, pl->tw[f]); break;
        }
        src = dst; n = m; s *= r;
    }
    if (cm) switch (pl->fac[nf - 1]) {
        case 2: last2m(src, y, s, cm); break;
        case 3: last3m(src, y, s, cm); break;
        case 4: last4m(src, y, s, cm); break;
        case 5: last5m(src, y, s, cm); break;
        default: last8m(src, y, s, cm); break;
    } else switch (pl->fac[nf - 1]) {
        case 2: last2p(src, y, s); break;
        case 3: last3p(src, y, s); break;
        case 4: last4p(src, y, s); break;
        case 5: last5p(src, y, s); break;
        default: last8p(src, y, s); break;
    }
}

fft1d_plan *fft1d_create(int L, int batch)
{
    if (!fft1d_supports(L) || batch < 1) return NULL;
    fft1d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L; p->batch = batch;

    /* pow2 part as 8/4(/2) minimizing stage count, then the 3s, then the 5s
       (so the last, fused-map stage has the largest contiguous q loop) */
    int n = L, v = 0;
    while (n % 2 == 0) { n /= 2; v++; }
    while (v >= 4) { if (v % 3 == 1) { p->fac[p->nf++] = 4; v -= 2; }
                     else            { p->fac[p->nf++] = 8; v -= 3; } }
    if (v == 3) p->fac[p->nf++] = 8;
    else if (v == 2) p->fac[p->nf++] = 4;
    else if (v == 1) p->fac[p->nf++] = 2;
    while (n % 3 == 0) { p->fac[p->nf++] = 3; n /= 3; }
    while (n % 5 == 0) { p->fac[p->nf++] = 5; n /= 5; }

    /* consumption-order tables for every stage but the last (last is all ones) */
    int nc = L;
    for (int f = 0; f < p->nf; ++f) {
        int r = p->fac[f], m = nc / r;
        if (f < p->nf - 1) {
            void *tw = NULL;
            if (posix_memalign(&tw, 64, (size_t)m * (r - 1) * sizeof(double _Complex)))
                { fft1d_destroy(p); return NULL; }
            d1tw_stage(nc, r, tw);
            p->tw[f] = tw;
        }
        nc = m;
    }
    void *a = NULL, *b = NULL;
    if (posix_memalign(&a, 64, (size_t)L * sizeof(double _Complex)) ||
        posix_memalign(&b, 64, (size_t)L * sizeof(double _Complex)))
        { free(a); fft1d_destroy(p); return NULL; }
    p->s0 = a; p->s1 = b;
    return p;
}

void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    const double *x = (const double *)in;
    double *y = (double *)out;
    for (int b = 0; b < p->batch; ++b)
        do_fft(p, x + 2*(size_t)b*p->L, y + 2*(size_t)b*p->L, NULL);
}

/* Fused chain: the graded map is applied inside the final stage's store loop, so a
 * chain step is exactly nf array sweeps instead of the fallback's nf + 1. The state
 * lives in final_out throughout; from step 2 on the transform runs in place
 * (first pass reads state into s0, final pass writes state -- no overlap). */
void fft1d_chain(fft1d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    const double *cd = (const double *)c;
    double *st = (double *)final_out;
    const double *src = (const double *)x0;
    for (int step = 0; step < m; ++step) {
        for (int b = 0; b < p->batch; ++b)
            do_fft(p, src + 2*(size_t)b*p->L, st + 2*(size_t)b*p->L, cd + 2*(size_t)b*p->L);
        src = st;
    }
}

void fft1d_destroy(fft1d_plan *p)
{
    if (!p) return;
    for (int f = 0; f < p->nf; ++f) free(p->tw[f]);
    free(p->s0); free(p->s1); free(p);
}
