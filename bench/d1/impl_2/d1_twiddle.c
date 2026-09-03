/* d1_twiddle -- LIBRARY LAYER (adoption-scored): exact 1D twiddle tables, consumption order.
 *
 * The product of this entry is the ADOPTION BLOCK below: copy it whole into your entry
 * (and say so in your strategy record). It exists because the accuracy vein of the 1D
 * survey converged on one rule -- "inaccurate twiddles are the leading cause of FFT
 * inaccuracy": generate tables in the PLAN stage from exactly-reduced arguments, never
 * from in-loop recurrences (whose error grows O(sqrt N)..O(N^2)).
 *
 * The FFT underneath is the demonstration vehicle, not the point: a mixed-radix
 * (2/3/4/5/8) Stockham autosort FFT (no bit-reversal), AVX-512 4-complex lanes since
 * round d1_r2, whose per-stage tables are laid out in EXACTLY the order and FORMAT the
 * butterfly loops consume them -- linear reads, broadcast-ready pairs, no shuffles to
 * unpack a twiddle. Supports smooth L = 2^a 3^b 5^c (graded: 32/60/64/128/1024/4096/
 * 16384) and exports a fused-map fft1d_chain (state<-(z+c)/(1+|z+c|) inside the final
 * stage's store loop, chain run per-transform so the whole m-step chain stays
 * cache-resident -- borrowed from d1_pow2's round-1 record).
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef __AVX512F__
#include <immintrin.h>
#endif
#include "../fft1d_api.h"

/* ====================== D1TW ADOPTION BLOCK (v2, round d1_r2) ======================
 * Exact twiddles: w = exp(-2*pi*i * num/den) to ~1 ulp, for ANY int64 num (negative
 * fine), den in [1, 2^59). Two exact reductions before any floating trig:
 *   1. integer:   r = num mod den            (the index never touches fp)
 *   2. quadrant:  q = round(4r/den), s = 4r - q*den   (EXACT in int64)
 * so the trig argument is (pi/2)*(s/den) with |s/den| <= 1/2, i.e. |arg| <= pi/4, and
 * the quadrant factors are exactly 0/+-1 (no sqrt(2)/2 rounding). The argument itself
 * is built from a two-part pi/2 split with an FMA-recovered product error, so the only
 * inexactness left is one rounding of s/den plus libm's ~0.5 ulp sin/cos.
 * (Verified in d1_r1 against 150-bit mpmath over 13k adversarial (k,den) pairs:
 * worst component error 1.24 x 2^-53.)
 *
 * d1tw_chirp: the Bluestein chirp exp(-i*pi*k^2/N) with k^2 reduced mod 2N in INTEGERS
 * first -- the survey's explicit fp64 trap at k ~ 1e5. d1tw_stage: Stockham stage
 * tables in consumption order (complex interleaved, v1 format, for scalar loops).
 *
 * NEW in v2 -- broadcast-ready formats for the AVX-512 cmul idiom
 *     u*w = fmadd(permute_pd(u,0x55), wp, mul(u, wr))     [1 shuffle+1 mul+1 fma]
 * where wr = (re,re,...) and wp = (-im,+im,-im,+im,...). The per-radix-4 layout below
 * is byte-identical to what d1_pow2 hand-rolled in round d1_r1; these builders make it
 * for ANY radix from the exact generator, so adopting them is a drop-in.
 *   d1tw_stage_bc(n, r, tw): for the q-vectorized pass (twiddle constant across the
 *     vector). Per p = 0..n/r-1: (r-1) doubles [re w^(p*t), t=1..r-1] then (r-1) pairs
 *     [-im, +im]. Stride 3(r-1) doubles per p. Consume with set1 / broadcast_f64x2.
 *   d1tw_stage_s1bc(n, r, tw): for the s==1 FIRST stage vectorized ACROSS p (4 lanes =
 *     4 consecutive p, per-lane twiddles). Per group g (p0 = 4g), per t = 1..r-1:
 *     8 doubles [re(p0),re(p0),re(p1),re(p1),...] then 8 doubles [-im(p0),+im(p0),...],
 *     i.e. full zmm images, loadu-ready. 16(r-1) doubles per group; the tail group is
 *     zero-padded. Total 16*(r-1)*ceil(m/4) doubles, m = n/r.
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
 * its table is identically 1 -- skip it. */
static void d1tw_stage(int n, int r, double _Complex *tw)
{
    int m = n / r; size_t idx = 0;
    for (int p = 0; p < m; ++p)
        for (int t = 1; t < r; ++t)
            tw[idx++] = d1tw_cexp((int64_t)p * t, n);
}

/* v2: broadcast-pair layout, 3(r-1) doubles per p (see block header). */
static void d1tw_stage_bc(int n, int r, double *tw)
{
    int m = n / r;
    for (int p = 0; p < m; ++p) {
        double *tp = tw + (size_t)3 * (r - 1) * p;
        for (int t = 1; t < r; ++t) {
            double _Complex w = d1tw_cexp((int64_t)p * t, n);
            tp[t - 1]               = creal(w);
            tp[(r - 1) + 2*(t - 1)]     = -cimag(w);
            tp[(r - 1) + 2*(t - 1) + 1] =  cimag(w);
        }
    }
}

/* v2: first-stage (s==1) lane-major layout, groups of 4 p, zero-padded tail. */
static void d1tw_stage_s1bc(int n, int r, double *tw)
{
    int m = n / r, ng = (m + 3) / 4;
    memset(tw, 0, (size_t)16 * (r - 1) * ng * sizeof(double));
    for (int p = 0; p < m; ++p) {
        int g = p / 4, lane = p % 4;
        double *tg = tw + (size_t)16 * (r - 1) * g;
        for (int t = 1; t < r; ++t) {
            double _Complex w = d1tw_cexp((int64_t)p * t, n);
            double *tr = tg + 16 * (t - 1);
            tr[2*lane]     = creal(w);
            tr[2*lane + 1] = creal(w);
            tr[8 + 2*lane]     = -cimag(w);
            tr[8 + 2*lane + 1] =  cimag(w);
        }
    }
}
/* ==================== end D1TW ADOPTION BLOCK ==================== */

const char *fft1d_name(void) { return "d1_twiddle"; }
const char *fft1d_description(void)
{
    return "LIBRARY LAYER (adoption-scored): exact 1D twiddle tables, consumption order "
           "(d1tw_cexp quadrant-exact ~1ulp, d1tw_chirp integer-reduced, d1tw_stage v1 + "
           "v2 broadcast-pair/lane-major AVX-512 formats); vehicle: mixed-radix 2/3/4/5/8 "
           "Stockham, zmm 4-complex lanes, per-transform fused-map chain, smooth L";
}

int fft1d_supports(int L)
{
    if (L < 2 || L > (1 << 20)) return 0;
    while (L % 2 == 0) L /= 2;
    while (L % 3 == 0) L /= 3;
    while (L % 5 == 0) L /= 5;
    return L == 1;
}

/* stage kinds: how the executor runs each stage and which table format it built */
enum { K_SC = 0,     /* scalar pass, v1 complex table */
       K_V,          /* q-vectorized AVX-512 pass, bc table (s % 4 == 0) */
       K_S1V4 };     /* first-stage radix-4 across-p, s1bc table */

#define D1TW_MAXF 24
struct fft1d_plan {
    int L, batch, nf;
    int fac[D1TW_MAXF];
    int kind[D1TW_MAXF];
    double *tw[D1TW_MAXF];   /* per-stage consumption-order tables */
    double *s0, *s1;         /* ping-pong scratch, L complex each */
};

/* ---- scalar pass kernels (fallback: odd s, tiny L; v1 tables, read LINEARLY) ---- */
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

/* ---- scalar last-stage kernels (fallback), map fused at the store.
 * Compile-time domap flag (a runtime branch per store killed vectorization in d1_r1). */
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

/* =================== AVX-512 kernels (new in d1_r2) ===================
 * Interleaved complex, one zmm = 4 complexes. The cmul idiom (1 vpermilpd + mul + fma
 * on (re,re)/( -im,+im) broadcast pairs) and the s==1 across-p first stage with a 4x4
 * complex-lane transpose are BORROWED from d1_pow2's round-d1_r1 implementation; the
 * tables feeding them come from the exact generators above (d1tw_stage_bc/_s1bc). */
#ifdef __AVX512F__

#define PSWAP 0x55  /* vpermilpd immediate: swap re/im inside each 128-bit pair */

static inline __m512d vcmul(__m512d u, __m512d wr, __m512d wp)
{
    return _mm512_fmadd_pd(_mm512_permute_pd(u, PSWAP), wp, _mm512_mul_pd(u, wr));
}

/* first stage, s == 1, radix 4, vectorized across p; masked tail group.
 * y[4p+t] = sum_i x[p + m i] W4^{it} W_n^{pt}; each final store = one p's 4 outputs. */
static void vfirst4(const double *x, double *y, int m, const double *tw)
{
    const __m512d ONE = _mm512_set1_pd(1.0);
    const __m512i idxA = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
    const __m512i idxB = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);
    for (int p = 0; p < m; p += 4) {
        const int g = m - p < 4 ? m - p : 4;
        __m512d a, b, c, d;
        if (g == 4) {
            a = _mm512_loadu_pd(x + 2*p);
            b = _mm512_loadu_pd(x + 2*(p + m));
            c = _mm512_loadu_pd(x + 2*(p + 2*m));
            d = _mm512_loadu_pd(x + 2*(p + 3*m));
        } else {
            __mmask8 k = (__mmask8)((1u << (2*g)) - 1);
            a = _mm512_maskz_loadu_pd(k, x + 2*p);
            b = _mm512_maskz_loadu_pd(k, x + 2*(p + m));
            c = _mm512_maskz_loadu_pd(k, x + 2*(p + 2*m));
            d = _mm512_maskz_loadu_pd(k, x + 2*(p + 3*m));
        }
        __m512d apc = _mm512_add_pd(a, c), amc = _mm512_sub_pd(a, c);
        __m512d bpd = _mm512_add_pd(b, d), bmd = _mm512_sub_pd(b, d);
        __m512d sw = _mm512_permute_pd(bmd, PSWAP);
        __m512d r0 = _mm512_add_pd(apc, bpd);
        __m512d u2 = _mm512_sub_pd(apc, bpd);
        __m512d u1 = _mm512_fmsubadd_pd(amc, ONE, sw);   /* amc - i bmd */
        __m512d u3 = _mm512_fmaddsub_pd(amc, ONE, sw);   /* amc + i bmd */
        const double *t = tw + 12*(size_t)p;   /* 48 dbl per 4-p group (s1bc) */
        __m512d r1 = vcmul(u1, _mm512_loadu_pd(t),      _mm512_loadu_pd(t + 8));
        __m512d r2 = vcmul(u2, _mm512_loadu_pd(t + 16), _mm512_loadu_pd(t + 24));
        __m512d r3 = vcmul(u3, _mm512_loadu_pd(t + 32), _mm512_loadu_pd(t + 40));
        /* 4x4 complex-lane transpose so each store is one p's contiguous outputs */
        __m512d p0 = _mm512_permutex2var_pd(r0, idxA, r1);
        __m512d p1 = _mm512_permutex2var_pd(r0, idxB, r1);
        __m512d p2 = _mm512_permutex2var_pd(r2, idxA, r3);
        __m512d p3 = _mm512_permutex2var_pd(r2, idxB, r3);
        __m512d o0 = _mm512_shuffle_f64x2(p0, p2, 0x44);
        __m512d o1 = _mm512_shuffle_f64x2(p1, p3, 0x44);
        __m512d o2 = _mm512_shuffle_f64x2(p0, p2, 0xEE);
        __m512d o3 = _mm512_shuffle_f64x2(p1, p3, 0xEE);
        double *yy = y + 8*(size_t)p;
        _mm512_storeu_pd(yy, o0);
        if (g > 1) _mm512_storeu_pd(yy + 8,  o1);
        if (g > 2) _mm512_storeu_pd(yy + 16, o2);
        if (g > 3) _mm512_storeu_pd(yy + 24, o3);
    }
}

/* ---- generic q-vectorized twiddled passes, s % 4 == 0, bc tables ---- */
static void vpass2(const double *x, double *y, int m, int s, const double *tw)
{
    const long S = 2L * s;
    for (int p = 0; p < m; ++p) {
        const double *t = tw + 3*(size_t)p;
        const __m512d w1r = _mm512_set1_pd(t[0]);
        const __m512d w1p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 1));
        const double *xa = x + S*p, *xb = xa + S*m;
        double *y0 = y + S*2*p, *y1 = y0 + S;
        for (long q = 0; q < S; q += 8) {
            __m512d a = _mm512_loadu_pd(xa + q), b = _mm512_loadu_pd(xb + q);
            _mm512_storeu_pd(y0 + q, _mm512_add_pd(a, b));
            _mm512_storeu_pd(y1 + q, vcmul(_mm512_sub_pd(a, b), w1r, w1p));
        }
    }
}

static void vpass3(const double *x, double *y, int m, int s, const double *tw)
{
    const long S = 2L * s;
    const __m512d ONE  = _mm512_set1_pd(1.0);
    const __m512d HALF = _mm512_set1_pd(0.5);
    const __m512d K3   = _mm512_set1_pd(-0.8660254037844386);
    for (int p = 0; p < m; ++p) {
        const double *t = tw + 6*(size_t)p;
        const __m512d w1r = _mm512_set1_pd(t[0]);
        const __m512d w2r = _mm512_set1_pd(t[1]);
        const __m512d w1p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 2));
        const __m512d w2p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 4));
        const double *xa = x + S*p, *xb = xa + S*m, *xc = xb + S*m;
        double *y0 = y + S*3*p, *y1 = y0 + S, *y2 = y1 + S;
        for (long q = 0; q < S; q += 8) {
            __m512d a = _mm512_loadu_pd(xa + q);
            __m512d b = _mm512_loadu_pd(xb + q);
            __m512d c = _mm512_loadu_pd(xc + q);
            __m512d t1 = _mm512_add_pd(b, c);
            __m512d u = _mm512_fnmadd_pd(HALF, t1, a);
            __m512d v = _mm512_mul_pd(K3, _mm512_sub_pd(b, c));
            __m512d sw = _mm512_permute_pd(v, PSWAP);
            _mm512_storeu_pd(y0 + q, _mm512_add_pd(a, t1));
            _mm512_storeu_pd(y1 + q, vcmul(_mm512_fmaddsub_pd(u, ONE, sw), w1r, w1p)); /* u+iv */
            _mm512_storeu_pd(y2 + q, vcmul(_mm512_fmsubadd_pd(u, ONE, sw), w2r, w2p)); /* u-iv */
        }
    }
}

static void vpass4(const double *x, double *y, int m, int s, const double *tw)
{
    const long S = 2L * s;
    const __m512d ONE = _mm512_set1_pd(1.0);
    for (int p = 0; p < m; ++p) {
        const double *t = tw + 9*(size_t)p;
        const __m512d w1r = _mm512_set1_pd(t[0]);
        const __m512d w2r = _mm512_set1_pd(t[1]);
        const __m512d w3r = _mm512_set1_pd(t[2]);
        const __m512d w1p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 3));
        const __m512d w2p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 5));
        const __m512d w3p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 7));
        const double *xa = x + S*p, *xb = xa + S*m, *xc = xb + S*m, *xd = xc + S*m;
        double *y0 = y + S*4*p, *y1 = y0 + S, *y2 = y1 + S, *y3 = y2 + S;
        for (long q = 0; q < S; q += 8) {
            __m512d a = _mm512_loadu_pd(xa + q), b = _mm512_loadu_pd(xb + q);
            __m512d c = _mm512_loadu_pd(xc + q), d = _mm512_loadu_pd(xd + q);
            __m512d apc = _mm512_add_pd(a, c), amc = _mm512_sub_pd(a, c);
            __m512d bpd = _mm512_add_pd(b, d), bmd = _mm512_sub_pd(b, d);
            __m512d sw = _mm512_permute_pd(bmd, PSWAP);
            _mm512_storeu_pd(y0 + q, _mm512_add_pd(apc, bpd));
            _mm512_storeu_pd(y1 + q, vcmul(_mm512_fmsubadd_pd(amc, ONE, sw), w1r, w1p));
            _mm512_storeu_pd(y2 + q, vcmul(_mm512_sub_pd(apc, bpd), w2r, w2p));
            _mm512_storeu_pd(y3 + q, vcmul(_mm512_fmaddsub_pd(amc, ONE, sw), w3r, w3p));
        }
    }
}

static void vpass5(const double *x, double *y, int m, int s, const double *tw)
{
    const long S = 2L * s;
    const __m512d ONE = _mm512_set1_pd(1.0);
    const __m512d C1 = _mm512_set1_pd(0.30901699437494745);
    const __m512d S1 = _mm512_set1_pd(0.9510565162951535);
    const __m512d C2 = _mm512_set1_pd(-0.8090169943749475);
    const __m512d S2 = _mm512_set1_pd(0.5877852522924731);
    for (int p = 0; p < m; ++p) {
        const double *t = tw + 12*(size_t)p;
        const __m512d w1r = _mm512_set1_pd(t[0]);
        const __m512d w2r = _mm512_set1_pd(t[1]);
        const __m512d w3r = _mm512_set1_pd(t[2]);
        const __m512d w4r = _mm512_set1_pd(t[3]);
        const __m512d w1p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 4));
        const __m512d w2p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 6));
        const __m512d w3p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 8));
        const __m512d w4p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 10));
        const double *xa = x + S*p, *xb = xa + S*m, *xc = xb + S*m,
                     *xd = xc + S*m, *xe = xd + S*m;
        double *y0 = y + S*5*p, *y1 = y0 + S, *y2 = y1 + S, *y3 = y2 + S, *y4 = y3 + S;
        for (long q = 0; q < S; q += 8) {
            __m512d a = _mm512_loadu_pd(xa + q);
            __m512d b = _mm512_loadu_pd(xb + q), c = _mm512_loadu_pd(xc + q);
            __m512d d = _mm512_loadu_pd(xd + q), e = _mm512_loadu_pd(xe + q);
            __m512d t1 = _mm512_add_pd(b, e), t2 = _mm512_add_pd(c, d);
            __m512d t3 = _mm512_sub_pd(b, e), t4 = _mm512_sub_pd(c, d);
            __m512d m1 = _mm512_fmadd_pd(C2, t2, _mm512_fmadd_pd(C1, t1, a));
            __m512d m2 = _mm512_fmadd_pd(C1, t2, _mm512_fmadd_pd(C2, t1, a));
            __m512d n1 = _mm512_fmadd_pd(S1, t3, _mm512_mul_pd(S2, t4));
            __m512d n2 = _mm512_fnmadd_pd(S1, t4, _mm512_mul_pd(S2, t3));
            __m512d s1 = _mm512_permute_pd(n1, PSWAP);
            __m512d s2 = _mm512_permute_pd(n2, PSWAP);
            _mm512_storeu_pd(y0 + q, _mm512_add_pd(a, _mm512_add_pd(t1, t2)));
            _mm512_storeu_pd(y1 + q, vcmul(_mm512_fmsubadd_pd(m1, ONE, s1), w1r, w1p)); /* m1-in1 */
            _mm512_storeu_pd(y2 + q, vcmul(_mm512_fmsubadd_pd(m2, ONE, s2), w2r, w2p)); /* m2-in2 */
            _mm512_storeu_pd(y3 + q, vcmul(_mm512_fmaddsub_pd(m2, ONE, s2), w3r, w3p)); /* m2+in2 */
            _mm512_storeu_pd(y4 + q, vcmul(_mm512_fmaddsub_pd(m1, ONE, s1), w4r, w4p)); /* m1+in1 */
        }
    }
}

/* radix-8 DIF butterfly on 8 vectors; u1/u3/u5/u7 pick up the w8 factors. */
#define VR8_CONSTS                                                                     \
    const __m512d ONE = _mm512_set1_pd(1.0);                                           \
    const __m512d CQ  = _mm512_set1_pd(0.7071067811865476);                            \
    const __m512d CPN = _mm512_setr_pd(0.7071067811865476, -0.7071067811865476,        \
                                       0.7071067811865476, -0.7071067811865476,        \
                                       0.7071067811865476, -0.7071067811865476,        \
                                       0.7071067811865476, -0.7071067811865476)

#define VR8_BODY(X0, X1, X2, X3, X4, X5, X6, X7)                                       \
    __m512d s0 = _mm512_add_pd(X0, X4), s1 = _mm512_add_pd(X1, X5);                    \
    __m512d s2 = _mm512_add_pd(X2, X6), s3 = _mm512_add_pd(X3, X7);                    \
    __m512d d0 = _mm512_sub_pd(X0, X4), d1 = _mm512_sub_pd(X1, X5);                    \
    __m512d d2 = _mm512_sub_pd(X2, X6), d3 = _mm512_sub_pd(X3, X7);                    \
    __m512d apc = _mm512_add_pd(s0, s2), amc = _mm512_sub_pd(s0, s2);                  \
    __m512d bpd = _mm512_add_pd(s1, s3), bmd = _mm512_sub_pd(s1, s3);                  \
    __m512d swe = _mm512_permute_pd(bmd, PSWAP);                                       \
    __m512d u0 = _mm512_add_pd(apc, bpd);                                              \
    __m512d u4 = _mm512_sub_pd(apc, bpd);                                              \
    __m512d u2 = _mm512_fmsubadd_pd(amc, ONE, swe);                                    \
    __m512d u6 = _mm512_fmaddsub_pd(amc, ONE, swe);                                    \
    /* e1 = (d1 - i d1_swapped...) : d1 * w8^1 = CQ*(d1 - i d1),  e3 = d3 * w8^3 */    \
    __m512d e1 = _mm512_mul_pd(_mm512_fmsubadd_pd(d1, ONE, _mm512_permute_pd(d1, PSWAP)), CQ); \
    __m512d e3 = _mm512_mul_pd(                                                        \
        _mm512_permute_pd(_mm512_fmsubadd_pd(d3, ONE, _mm512_permute_pd(d3, PSWAP)), PSWAP), CPN); \
    __m512d sw2 = _mm512_permute_pd(d2, PSWAP);                                        \
    __m512d apo = _mm512_fmsubadd_pd(d0, ONE, sw2);                                    \
    __m512d amo = _mm512_fmaddsub_pd(d0, ONE, sw2);                                    \
    __m512d bpo = _mm512_add_pd(e1, e3), bmo = _mm512_sub_pd(e1, e3);                  \
    __m512d swo = _mm512_permute_pd(bmo, PSWAP);                                       \
    __m512d u1 = _mm512_add_pd(apo, bpo);                                              \
    __m512d u5 = _mm512_sub_pd(apo, bpo);                                              \
    __m512d u3 = _mm512_fmsubadd_pd(amo, ONE, swo);                                    \
    __m512d u7 = _mm512_fmaddsub_pd(amo, ONE, swo)

static void vpass8(const double *x, double *y, int m, int s, const double *tw)
{
    const long S = 2L * s;
    VR8_CONSTS;
    for (int p = 0; p < m; ++p) {
        const double *t = tw + 21*(size_t)p;
        const __m512d w1r = _mm512_set1_pd(t[0]), w2r = _mm512_set1_pd(t[1]);
        const __m512d w3r = _mm512_set1_pd(t[2]), w4r = _mm512_set1_pd(t[3]);
        const __m512d w5r = _mm512_set1_pd(t[4]), w6r = _mm512_set1_pd(t[5]);
        const __m512d w7r = _mm512_set1_pd(t[6]);
        const __m512d w1p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 7));
        const __m512d w2p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 9));
        const __m512d w3p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 11));
        const __m512d w4p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 13));
        const __m512d w5p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 15));
        const __m512d w6p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 17));
        const __m512d w7p = _mm512_broadcast_f64x2(_mm_loadu_pd(t + 19));
        const double *xa = x + S*p;
        const long M = S*m;
        double *y0 = y + S*8*p;
        for (long q = 0; q < S; q += 8) {
            __m512d a0 = _mm512_loadu_pd(xa + q);
            __m512d a1 = _mm512_loadu_pd(xa + q + M);
            __m512d a2 = _mm512_loadu_pd(xa + q + 2*M);
            __m512d a3 = _mm512_loadu_pd(xa + q + 3*M);
            __m512d a4 = _mm512_loadu_pd(xa + q + 4*M);
            __m512d a5 = _mm512_loadu_pd(xa + q + 5*M);
            __m512d a6 = _mm512_loadu_pd(xa + q + 6*M);
            __m512d a7 = _mm512_loadu_pd(xa + q + 7*M);
            VR8_BODY(a0, a1, a2, a3, a4, a5, a6, a7);
            _mm512_storeu_pd(y0 + q,       u0);
            _mm512_storeu_pd(y0 + q + S,   vcmul(u1, w1r, w1p));
            _mm512_storeu_pd(y0 + q + 2*S, vcmul(u2, w2r, w2p));
            _mm512_storeu_pd(y0 + q + 3*S, vcmul(u3, w3r, w3p));
            _mm512_storeu_pd(y0 + q + 4*S, vcmul(u4, w4r, w4p));
            _mm512_storeu_pd(y0 + q + 5*S, vcmul(u5, w5r, w5p));
            _mm512_storeu_pd(y0 + q + 6*S, vcmul(u6, w6r, w6p));
            _mm512_storeu_pd(y0 + q + 7*S, vcmul(u7, w7r, w7p));
        }
    }
}

/* map store: state = (z+c)/(1+|z+c|). Two builds of the same store:
 *   D1TW_FASTMAP (default): rsqrt14/rcp14 + one Newton step each + an exact-residual
 *     FMA refinement of the sqrt and of the final quotient (~1 ulp each) -- the map
 *     recipe from d1_pow2's d1_r1 record, which passes the chain gate provided the
 *     twiddles are exact (ours are). vsqrtpd/vdivpd zmm are long-latency, poorly
 *     pipelined ops and were ~half the chained step cost at every size.
 *   exact build (-DD1TW_EXACTMAP): vsqrtpd + vdivpd, bit-matches the driver fallback;
 *     the fallback if a scoring-node seed ever fails the gate.
 * n_ is clamped below by 1e-300 so z == 0 cannot produce rsqrt(0) = inf -> NaN
 * (the clamp is far below any admissible |z|^2 and leaves the result z/(1+0) = z). */
#ifndef D1TW_EXACTMAP
#define D1TW_VST(off, v)                                                              \
    do {                                                                              \
        if (domap) {                                                                  \
            __m512d z_ = _mm512_add_pd((v), _mm512_loadu_pd(cm + (off)));             \
            __m512d t_ = _mm512_mul_pd(z_, z_);                                       \
            __m512d n_ = _mm512_max_pd(_mm512_add_pd(t_, _mm512_permute_pd(t_, PSWAP)),\
                                       _mm512_set1_pd(1e-300));                       \
            __m512d y0_ = _mm512_rsqrt14_pd(n_);                                      \
            __m512d h_ = _mm512_mul_pd(_mm512_set1_pd(0.5), y0_);                     \
            __m512d e_ = _mm512_fnmadd_pd(_mm512_mul_pd(n_, y0_), y0_, _mm512_set1_pd(3.0)); \
            __m512d y1_ = _mm512_mul_pd(h_, e_);        /* rsqrt, ~2^-28 */           \
            __m512d s_ = _mm512_mul_pd(n_, y1_);        /* sqrt,  ~2^-28 */           \
            __m512d r_ = _mm512_fmsub_pd(s_, s_, n_);   /* exact residual */          \
            s_ = _mm512_fnmadd_pd(r_, _mm512_mul_pd(_mm512_set1_pd(0.5), y1_), s_);   \
            __m512d d_ = _mm512_add_pd(VONE, s_);                                     \
            __m512d t0_ = _mm512_rcp14_pd(d_);                                        \
            __m512d t1_ = _mm512_mul_pd(t0_, _mm512_fnmadd_pd(d_, t0_, _mm512_set1_pd(2.0))); \
            t1_ = _mm512_mul_pd(t1_, _mm512_fnmadd_pd(d_, t1_, _mm512_set1_pd(2.0))); \
            __m512d q_ = _mm512_mul_pd(z_, t1_);                                      \
            __m512d r2_ = _mm512_fmsub_pd(q_, d_, z_);  /* exact residual */          \
            _mm512_storeu_pd(y + (off), _mm512_fnmadd_pd(r2_, t1_, q_));              \
        } else _mm512_storeu_pd(y + (off), (v));                                      \
    } while (0)
#else
#define D1TW_VST(off, v)                                                              \
    do {                                                                              \
        if (domap) {                                                                  \
            __m512d z_ = _mm512_add_pd((v), _mm512_loadu_pd(cm + (off)));             \
            __m512d t_ = _mm512_mul_pd(z_, z_);                                       \
            __m512d n_ = _mm512_add_pd(t_, _mm512_permute_pd(t_, PSWAP));             \
            __m512d d_ = _mm512_add_pd(VONE, _mm512_sqrt_pd(n_));                     \
            _mm512_storeu_pd(y + (off), _mm512_div_pd(z_, d_));                       \
        } else _mm512_storeu_pd(y + (off), (v));                                      \
    } while (0)
#endif

D1TW_INLINE vlast2_impl(const double *x, double *y, int s, const double *cm, const int domap)
{
    const long S = 2L * s;
    const __m512d VONE = _mm512_set1_pd(1.0);
    for (long q = 0; q < S; q += 8) {
        __m512d a = _mm512_loadu_pd(x + q), b = _mm512_loadu_pd(x + q + S);
        D1TW_VST(q,     _mm512_add_pd(a, b));
        D1TW_VST(q + S, _mm512_sub_pd(a, b));
    }
}

D1TW_INLINE vlast3_impl(const double *x, double *y, int s, const double *cm, const int domap)
{
    const long S = 2L * s;
    const __m512d VONE = _mm512_set1_pd(1.0);
    const __m512d HALF = _mm512_set1_pd(0.5);
    const __m512d K3   = _mm512_set1_pd(-0.8660254037844386);
    for (long q = 0; q < S; q += 8) {
        __m512d a = _mm512_loadu_pd(x + q);
        __m512d b = _mm512_loadu_pd(x + q + S);
        __m512d c = _mm512_loadu_pd(x + q + 2*S);
        __m512d t1 = _mm512_add_pd(b, c);
        __m512d u = _mm512_fnmadd_pd(HALF, t1, a);
        __m512d v = _mm512_mul_pd(K3, _mm512_sub_pd(b, c));
        __m512d sw = _mm512_permute_pd(v, PSWAP);
        D1TW_VST(q,       _mm512_add_pd(a, t1));
        D1TW_VST(q + S,   _mm512_fmaddsub_pd(u, VONE, sw));
        D1TW_VST(q + 2*S, _mm512_fmsubadd_pd(u, VONE, sw));
    }
}

D1TW_INLINE vlast4_impl(const double *x, double *y, int s, const double *cm, const int domap)
{
    const long S = 2L * s;
    const __m512d VONE = _mm512_set1_pd(1.0);
    for (long q = 0; q < S; q += 8) {
        __m512d a = _mm512_loadu_pd(x + q),       b = _mm512_loadu_pd(x + q + S);
        __m512d c = _mm512_loadu_pd(x + q + 2*S), d = _mm512_loadu_pd(x + q + 3*S);
        __m512d apc = _mm512_add_pd(a, c), amc = _mm512_sub_pd(a, c);
        __m512d bpd = _mm512_add_pd(b, d), bmd = _mm512_sub_pd(b, d);
        __m512d sw = _mm512_permute_pd(bmd, PSWAP);
        D1TW_VST(q,       _mm512_add_pd(apc, bpd));
        D1TW_VST(q + S,   _mm512_fmsubadd_pd(amc, VONE, sw));
        D1TW_VST(q + 2*S, _mm512_sub_pd(apc, bpd));
        D1TW_VST(q + 3*S, _mm512_fmaddsub_pd(amc, VONE, sw));
    }
}

D1TW_INLINE vlast5_impl(const double *x, double *y, int s, const double *cm, const int domap)
{
    const long S = 2L * s;
    const __m512d VONE = _mm512_set1_pd(1.0);
    const __m512d C1 = _mm512_set1_pd(0.30901699437494745);
    const __m512d S1 = _mm512_set1_pd(0.9510565162951535);
    const __m512d C2 = _mm512_set1_pd(-0.8090169943749475);
    const __m512d S2 = _mm512_set1_pd(0.5877852522924731);
    for (long q = 0; q < S; q += 8) {
        __m512d a = _mm512_loadu_pd(x + q);
        __m512d b = _mm512_loadu_pd(x + q + S),   c = _mm512_loadu_pd(x + q + 2*S);
        __m512d d = _mm512_loadu_pd(x + q + 3*S), e = _mm512_loadu_pd(x + q + 4*S);
        __m512d t1 = _mm512_add_pd(b, e), t2 = _mm512_add_pd(c, d);
        __m512d t3 = _mm512_sub_pd(b, e), t4 = _mm512_sub_pd(c, d);
        __m512d m1 = _mm512_fmadd_pd(C2, t2, _mm512_fmadd_pd(C1, t1, a));
        __m512d m2 = _mm512_fmadd_pd(C1, t2, _mm512_fmadd_pd(C2, t1, a));
        __m512d n1 = _mm512_fmadd_pd(S1, t3, _mm512_mul_pd(S2, t4));
        __m512d n2 = _mm512_fnmadd_pd(S1, t4, _mm512_mul_pd(S2, t3));
        __m512d s1 = _mm512_permute_pd(n1, PSWAP);
        __m512d s2 = _mm512_permute_pd(n2, PSWAP);
        D1TW_VST(q,       _mm512_add_pd(a, _mm512_add_pd(t1, t2)));
        D1TW_VST(q + S,   _mm512_fmsubadd_pd(m1, VONE, s1));
        D1TW_VST(q + 2*S, _mm512_fmsubadd_pd(m2, VONE, s2));
        D1TW_VST(q + 3*S, _mm512_fmaddsub_pd(m2, VONE, s2));
        D1TW_VST(q + 4*S, _mm512_fmaddsub_pd(m1, VONE, s1));
    }
}

D1TW_INLINE vlast8_impl(const double *x, double *y, int s, const double *cm, const int domap)
{
    const long S = 2L * s;
    const __m512d VONE = _mm512_set1_pd(1.0);
    VR8_CONSTS;
    for (long q = 0; q < S; q += 8) {
        __m512d a0 = _mm512_loadu_pd(x + q);
        __m512d a1 = _mm512_loadu_pd(x + q + S);
        __m512d a2 = _mm512_loadu_pd(x + q + 2*S);
        __m512d a3 = _mm512_loadu_pd(x + q + 3*S);
        __m512d a4 = _mm512_loadu_pd(x + q + 4*S);
        __m512d a5 = _mm512_loadu_pd(x + q + 5*S);
        __m512d a6 = _mm512_loadu_pd(x + q + 6*S);
        __m512d a7 = _mm512_loadu_pd(x + q + 7*S);
        VR8_BODY(a0, a1, a2, a3, a4, a5, a6, a7);
        D1TW_VST(q,       u0);
        D1TW_VST(q + S,   u1);
        D1TW_VST(q + 2*S, u2);
        D1TW_VST(q + 3*S, u3);
        D1TW_VST(q + 4*S, u4);
        D1TW_VST(q + 5*S, u5);
        D1TW_VST(q + 6*S, u6);
        D1TW_VST(q + 7*S, u7);
    }
}

static void vlast2p(const double *x, double *y, int s) { vlast2_impl(x, y, s, NULL, 0); }
static void vlast2m(const double *x, double *y, int s, const double *cm) { vlast2_impl(x, y, s, cm, 1); }
static void vlast3p(const double *x, double *y, int s) { vlast3_impl(x, y, s, NULL, 0); }
static void vlast3m(const double *x, double *y, int s, const double *cm) { vlast3_impl(x, y, s, cm, 1); }
static void vlast4p(const double *x, double *y, int s) { vlast4_impl(x, y, s, NULL, 0); }
static void vlast4m(const double *x, double *y, int s, const double *cm) { vlast4_impl(x, y, s, cm, 1); }
static void vlast5p(const double *x, double *y, int s) { vlast5_impl(x, y, s, NULL, 0); }
static void vlast5m(const double *x, double *y, int s, const double *cm) { vlast5_impl(x, y, s, cm, 1); }
static void vlast8p(const double *x, double *y, int s) { vlast8_impl(x, y, s, NULL, 0); }
static void vlast8m(const double *x, double *y, int s, const double *cm) { vlast8_impl(x, y, s, cm, 1); }

#endif /* __AVX512F__ */

/* One transform: Stockham ping-pong through s0/s1, final stage lands in y.
 * Safe with x == y when nf >= 2 (x only read in the first pass, y only written in the
 * last) and when nf == 1 (each q iteration loads all r values before storing). */
static void do_fft(const fft1d_plan *pl, const double *x, double *y, const double *cm)
{
    int n = pl->L, s = 1, nf = pl->nf;
    const double *src = x;
    for (int f = 0; f < nf - 1; ++f) {
        int r = pl->fac[f], m = n / r;
        double *dst = (f == 0) ? pl->s0 : (src == pl->s0 ? pl->s1 : pl->s0);
#ifdef __AVX512F__
        if (pl->kind[f] == K_S1V4) { vfirst4(src, dst, m, pl->tw[f]); }
        else if (pl->kind[f] == K_V) switch (r) {
            case 2: vpass2(src, dst, m, s, pl->tw[f]); break;
            case 3: vpass3(src, dst, m, s, pl->tw[f]); break;
            case 4: vpass4(src, dst, m, s, pl->tw[f]); break;
            case 5: vpass5(src, dst, m, s, pl->tw[f]); break;
            default: vpass8(src, dst, m, s, pl->tw[f]); break;
        } else
#endif
        switch (r) {
            case 2: pass2(src, dst, m, s, pl->tw[f]); break;
            case 3: pass3(src, dst, m, s, pl->tw[f]); break;
            case 4: pass4(src, dst, m, s, pl->tw[f]); break;
            case 5: pass5(src, dst, m, s, pl->tw[f]); break;
            default: pass8(src, dst, m, s, pl->tw[f]); break;
        }
        src = dst; n = m; s *= r;
    }
#ifdef __AVX512F__
    if (pl->kind[nf - 1] == K_V) {
        if (cm) switch (pl->fac[nf - 1]) {
            case 2: vlast2m(src, y, s, cm); break;
            case 3: vlast3m(src, y, s, cm); break;
            case 4: vlast4m(src, y, s, cm); break;
            case 5: vlast5m(src, y, s, cm); break;
            default: vlast8m(src, y, s, cm); break;
        } else switch (pl->fac[nf - 1]) {
            case 2: vlast2p(src, y, s); break;
            case 3: vlast3p(src, y, s); break;
            case 4: vlast4p(src, y, s); break;
            case 5: vlast5p(src, y, s); break;
            default: vlast8p(src, y, s); break;
        }
        return;
    }
#endif
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

    /* factor schedule: radix-4 FIRST when v >= 2 (its s==1 stage vectorizes across p
       with a 4x4 transpose -- d1_pow2's r1 schedule finding), then radix-4 at small s
       until the remaining pow2 bits divide by 3, then radix-8s, then 3s, then 5s (so
       the last, fused-map stage has the largest contiguous q loop). */
    int n = L, v = 0;
    while (n % 2 == 0) { n /= 2; v++; }
    if (v >= 2) {
        p->fac[p->nf++] = 4;
        int b = v - 2;
        while (b % 3 != 0 && b >= 2) { p->fac[p->nf++] = 4; b -= 2; }
        while (b >= 3)               { p->fac[p->nf++] = 8; b -= 3; }
        if (b == 1) p->fac[p->nf++] = 2;
    } else if (v == 1) p->fac[p->nf++] = 2;
    while (n % 3 == 0) { p->fac[p->nf++] = 3; n /= 3; }
    while (n % 5 == 0) { p->fac[p->nf++] = 5; n /= 5; }

    /* per-stage kind + table in exactly the format that stage's kernel consumes */
    int nc = L, s = 1;
    for (int f = 0; f < p->nf; ++f) {
        int r = p->fac[f], m = nc / r;
        int kind = K_SC;
#ifdef __AVX512F__
        if (f == 0 && r == 4 && p->nf > 1) kind = K_S1V4;
        else if (s % 4 == 0 && s >= 4)     kind = K_V;
#endif
        p->kind[f] = kind;
        if (f < p->nf - 1) {
            size_t nd;                       /* table doubles */
            if (kind == K_S1V4)     nd = (size_t)16 * (r - 1) * ((m + 3) / 4);
            else if (kind == K_V)   nd = (size_t)3 * (r - 1) * m;
            else                    nd = (size_t)2 * (r - 1) * m;
            void *tw = NULL;
            if (posix_memalign(&tw, 64, nd * sizeof(double)))
                { fft1d_destroy(p); return NULL; }
            if (kind == K_S1V4)     d1tw_stage_s1bc(nc, r, tw);
            else if (kind == K_V)   d1tw_stage_bc(nc, r, tw);
            else                    d1tw_stage(nc, r, tw);
            p->tw[f] = tw;
        }
        nc = m; s *= r;
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

/* Fused chain, run PER TRANSFORM (batch outer, steps inner) so the whole m-step chain
 * of one transform stays cache-resident (~4L*16 B) instead of streaming the entire
 * B x L batch through memory every step -- borrowed from d1_pow2's d1_r1 record.
 * The graded map is applied inside the final stage's store loop, so a chain step is
 * exactly nf array sweeps. State lives in final_out; from step 2 on the transform
 * runs in place (first pass reads state into s0, final pass writes state). */
void fft1d_chain(fft1d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    const double *cd = (const double *)c;
    double *st = (double *)final_out;
    const double *xd = (const double *)x0;
    for (int b = 0; b < p->batch; ++b) {
        const size_t off = 2*(size_t)b*p->L;
        const double *src = xd + off;
        for (int step = 0; step < m; ++step) {
            do_fft(p, src, st + off, cd + off);
            src = st + off;
        }
    }
}

void fft1d_destroy(fft1d_plan *p)
{
    if (!p) return;
    for (int f = 0; f < p->nf; ++f) free(p->tw[f]);
    free(p->s0); free(p->s1); free(p);
}
