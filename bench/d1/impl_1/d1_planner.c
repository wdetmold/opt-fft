/* d1_planner -- LIBRARY LAYER (adoption-scored): 1D factorization -> candidate
 * algorithm chains.
 *
 * Round d1_r1: replaces the dense O(L^2) stub with a real planner.
 *
 * Plan = (base codelet, chain of combine radices), chosen at fft1d_create time by
 * factoring L:
 *   L = 2^a 3^b 5^c * P,  P = 1 or an odd prime <= 31.
 * Base codelets: hardcoded 2/3/4/5/8 and a generic table-driven symmetric odd-prime
 * codelet ((p-1)^2 real mults via the cos/sin +-j pairing, vectorized over output k).
 * Combine radices: hardcoded twiddled butterflies 2/3/4/5/8 (DIT: gather r strided
 * values, twiddle, r-point DFT in registers, scatter back in place).
 * The recursion is flattened at plan time into (input digit-reversed offsets for the
 * base pass) + (a list of combine levels, deepest first), so execution is two flat
 * loop nests with no recursion and no bit-reversal pass (natural order in and out).
 *
 * fft1d_chain owns the full m-step map chain: each batch element is chained through
 * all m steps by itself (the map is pointwise and the FFT per-transform, so batch
 * elements are independent) -- one transform of L complex doubles stays L1-resident
 * for the whole chain instead of streaming the whole batch every step. The map
 * (z+c)/(1+|z+c|) is fused into the final-stage stores, saving a full read+write
 * pass per step.
 *
 * All hot-path arithmetic is manual real/imag (never double _Complex products,
 * which gcc lowers to __muldc3 without -ffast-math).
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include "../fft1d_api.h"

#define MAXLEV 12
#define MAXH   16   /* (31-1)/2 = 15 */

typedef struct { int r, m, nblocks; double *tw; } level_t;

struct fft1d_plan {
    int L, batch;
    int base, nb;        /* base codelet size, nb = L/base = stride of base loads */
    int *offs;           /* nb digit-reversed input offsets (complex units) */
    int nlev;
    level_t lev[MAXLEV]; /* combine levels, deepest first */
    int h;               /* (base-1)/2 when base is an odd prime > 5 */
    double *pc, *ps;     /* prime codelet tables, j-major h x h: cos/sin(2pi jk/p) */
    double *small;       /* chain ping buffer: 2*L doubles */
};

const char *fft1d_name(void){ return "d1_planner"; }
const char *fft1d_description(void){
    return "LIBRARY LAYER (adoption-scored): 1D factorization -> candidate algorithm chains "
           "(mixed-radix DIT: codelets 2/3/4/5/8 + symmetric odd-prime<=31, flattened "
           "schedule, fused per-element map chain)";
}

/* ---------------- factorization -> (base, combine chain) ---------------- */

static int plan_factor(int L, int *base, int chain[MAXLEV], int *nchain)
{
    if (L < 2 || L > 4096) return 0;
    int rem = L, a = 0, b = 0, c = 0, P = 1;
    while (rem % 2 == 0) { rem /= 2; a++; }
    while (rem % 3 == 0) { rem /= 3; b++; }
    while (rem % 5 == 0) { rem /= 5; c++; }
    if (rem != 1) {
        static const int okp[] = {7, 11, 13, 17, 19, 23, 29, 31};
        int ok = 0;
        for (unsigned i = 0; i < sizeof okp / sizeof *okp; i++) if (rem == okp[i]) ok = 1;
        if (!ok) return 0;
        P = rem;
    }
    int n = 0, bs;
    if      (P > 1)  bs = P;
    else if (a >= 3) { bs = 8; a -= 3; }
    else if (c >= 1) { bs = 5; c -= 1; }
    else if (a == 2) { bs = 4; a  = 0; }
    else if (b >= 1) { bs = 3; b -= 1; }
    else             { bs = 2; a -= 1; }
    while (a >= 3) { chain[n++] = 8; a -= 3; }
    while (c-- > 0) chain[n++] = 5;
    if (a == 2)    { chain[n++] = 4; a = 0; }
    while (b-- > 0) chain[n++] = 3;
    if (a == 1)     chain[n++] = 2;
    *base = bs; *nchain = n;
    return 1;
}

/* digit-reversed input offsets, emitted in output-block order */
static void emit_offs(const int *chain, int j, int stride, int off, int **pp)
{
    if (j < 0) { *(*pp)++ = off; return; }
    for (int q = 0; q < chain[j]; q++)
        emit_offs(chain, j - 1, stride * chain[j], off + q * stride, pp);
}

/* ---------------- butterfly math (shared by codelets and combines) ------- */

#define K3    0.86602540378443864676372317075294  /* sin(pi/3) */
#define C51   0.30901699437494742410229341718282  /* cos(2pi/5) */
#define C52  (-0.80901699437494742410229341718282) /* cos(4pi/5) */
#define S51   0.95105651629515357211643933337938  /* sin(2pi/5) */
#define S52   0.58778525229247312916870595463907  /* sin(4pi/5) */
#define SQ2H  0.70710678118654752440084436210485  /* 1/sqrt(2) */

#define CMUL(tr, ti, ar, ai, wr, wi) do { \
    double _a = (ar), _b = (ai), _c = (wr), _d = (wi); \
    (tr) = _a*_c - _b*_d; (ti) = _a*_d + _b*_c; } while (0)

#define FFT2_MATH \
    double X0r = a0r + a1r, X0i = a0i + a1i; \
    double X1r = a0r - a1r, X1i = a0i - a1i;

#define FFT3_MATH \
    double ur = a1r + a2r, ui = a1i + a2i, vr = a1r - a2r, vi = a1i - a2i; \
    double X0r = a0r + ur, X0i = a0i + ui; \
    double br_ = a0r - 0.5*ur, bi_ = a0i - 0.5*ui; \
    double X1r = br_ + K3*vi, X1i = bi_ - K3*vr; \
    double X2r = br_ - K3*vi, X2i = bi_ + K3*vr;

#define FFT4_MATH \
    double t0r = a0r + a2r, t0i = a0i + a2i, t1r = a0r - a2r, t1i = a0i - a2i; \
    double t2r = a1r + a3r, t2i = a1i + a3i, t3r = a1r - a3r, t3i = a1i - a3i; \
    double X0r = t0r + t2r, X0i = t0i + t2i, X2r = t0r - t2r, X2i = t0i - t2i; \
    double X1r = t1r + t3i, X1i = t1i - t3r, X3r = t1r - t3i, X3i = t1i + t3r;

#define FFT5_MATH \
    double t1r = a1r + a4r, t1i = a1i + a4i, t3r = a1r - a4r, t3i = a1i - a4i; \
    double t2r = a2r + a3r, t2i = a2i + a3i, t4r = a2r - a3r, t4i = a2i - a3i; \
    double X0r = a0r + t1r + t2r, X0i = a0i + t1i + t2i; \
    double Pr = a0r + C51*t1r + C52*t2r, Pi = a0i + C51*t1i + C52*t2i; \
    double Qr = S51*t3r + S52*t4r,       Qi = S51*t3i + S52*t4i; \
    double Rr = a0r + C52*t1r + C51*t2r, Ri = a0i + C52*t1i + C51*t2i; \
    double Sr = S52*t3r - S51*t4r,       Si = S52*t3i - S51*t4i; \
    double X1r = Pr + Qi, X1i = Pi - Qr, X4r = Pr - Qi, X4i = Pi + Qr; \
    double X2r = Rr + Si, X2i = Ri - Sr, X3r = Rr - Si, X3i = Ri + Sr;

#define FFT8_MATH \
    double s0r = a0r + a4r, s0i = a0i + a4i, d0r = a0r - a4r, d0i = a0i - a4i; \
    double s1r = a1r + a5r, s1i = a1i + a5i, d1r = a1r - a5r, d1i = a1i - a5i; \
    double s2r = a2r + a6r, s2i = a2i + a6i, d2r = a2r - a6r, d2i = a2i - a6i; \
    double s3r = a3r + a7r, s3i = a3i + a7i, d3r = a3r - a7r, d3i = a3i - a7i; \
    double e0r = s0r + s2r, e0i = s0i + s2i, e1r = s0r - s2r, e1i = s0i - s2i; \
    double e2r = s1r + s3r, e2i = s1i + s3i, e3r = s1r - s3r, e3i = s1i - s3i; \
    double X0r = e0r + e2r, X0i = e0i + e2i, X4r = e0r - e2r, X4i = e0i - e2i; \
    double X2r = e1r + e3i, X2i = e1i - e3r, X6r = e1r - e3i, X6i = e1i + e3r; \
    double z1r = SQ2H*(d1r + d1i), z1i = SQ2H*(d1i - d1r); \
    double z2r = d2i, z2i = -d2r; \
    double z3r = SQ2H*(d3i - d3r), z3i = -SQ2H*(d3r + d3i); \
    double f0r = d0r + z2r, f0i = d0i + z2i, f1r = d0r - z2r, f1i = d0i - z2i; \
    double f2r = z1r + z3r, f2i = z1i + z3i, f3r = z1r - z3r, f3i = z1i - z3i; \
    double X1r = f0r + f2r, X1i = f0i + f2i, X5r = f0r - f2r, X5i = f0i - f2i; \
    double X3r = f1r + f3i, X3i = f1i - f3r, X7r = f1r - f3i, X7i = f1i + f3r;

/* stores: plain, and with the chain map (z+c)/(1+|z+c|) fused in.
 * STO / STOM write to codelet output o[]; STY / STYM write in place to combine
 * block y[].  The M variants read the map field cf[] at the same absolute index. */
#define STO(idx, re, im)  do { size_t _o = 2*(size_t)(idx); o[_o] = (re); o[_o+1] = (im); } while (0)
#define STOM(idx, re, im) do { size_t _o = 2*(size_t)(idx); \
    double _zr = (re) + cf[_o], _zi = (im) + cf[_o+1]; \
    double _t = 1.0/(1.0 + sqrt(_zr*_zr + _zi*_zi)); \
    o[_o] = _zr*_t; o[_o+1] = _zi*_t; } while (0)
#define STY(idx, re, im)  do { size_t _o = 2*(size_t)(idx); y[_o] = (re); y[_o+1] = (im); } while (0)
#define STYM(idx, re, im) do { size_t _o = 2*(size_t)(idx); \
    double _zr = (re) + cf[_o], _zi = (im) + cf[_o+1]; \
    double _t = 1.0/(1.0 + sqrt(_zr*_zr + _zi*_zi)); \
    y[_o] = _zr*_t; y[_o+1] = _zi*_t; } while (0)

/* ---------------- base codelets (strided in, contiguous out) ------------- */

static inline void c2(const double *restrict x, int is, double *restrict o, const double *restrict cf)
{
    double a0r = x[0], a0i = x[1], a1r = x[2*(size_t)is], a1i = x[2*(size_t)is+1];
    FFT2_MATH;
    if (!cf) { STO(0,X0r,X0i); STO(1,X1r,X1i); }
    else     { STOM(0,X0r,X0i); STOM(1,X1r,X1i); }
}

static inline void c3(const double *restrict x, int is, double *restrict o, const double *restrict cf)
{
    const size_t s = 2*(size_t)is;
    double a0r = x[0], a0i = x[1], a1r = x[s], a1i = x[s+1], a2r = x[2*s], a2i = x[2*s+1];
    FFT3_MATH;
    if (!cf) { STO(0,X0r,X0i); STO(1,X1r,X1i); STO(2,X2r,X2i); }
    else     { STOM(0,X0r,X0i); STOM(1,X1r,X1i); STOM(2,X2r,X2i); }
}

static inline void c4(const double *restrict x, int is, double *restrict o, const double *restrict cf)
{
    const size_t s = 2*(size_t)is;
    double a0r = x[0], a0i = x[1], a1r = x[s], a1i = x[s+1];
    double a2r = x[2*s], a2i = x[2*s+1], a3r = x[3*s], a3i = x[3*s+1];
    FFT4_MATH;
    if (!cf) { STO(0,X0r,X0i); STO(1,X1r,X1i); STO(2,X2r,X2i); STO(3,X3r,X3i); }
    else     { STOM(0,X0r,X0i); STOM(1,X1r,X1i); STOM(2,X2r,X2i); STOM(3,X3r,X3i); }
}

static inline void c5(const double *restrict x, int is, double *restrict o, const double *restrict cf)
{
    const size_t s = 2*(size_t)is;
    double a0r = x[0], a0i = x[1], a1r = x[s], a1i = x[s+1], a2r = x[2*s], a2i = x[2*s+1];
    double a3r = x[3*s], a3i = x[3*s+1], a4r = x[4*s], a4i = x[4*s+1];
    FFT5_MATH;
    if (!cf) { STO(0,X0r,X0i); STO(1,X1r,X1i); STO(2,X2r,X2i); STO(3,X3r,X3i); STO(4,X4r,X4i); }
    else     { STOM(0,X0r,X0i); STOM(1,X1r,X1i); STOM(2,X2r,X2i); STOM(3,X3r,X3i); STOM(4,X4r,X4i); }
}

static inline void c8(const double *restrict x, int is, double *restrict o, const double *restrict cf)
{
    const size_t s = 2*(size_t)is;
    double a0r = x[0],   a0i = x[1],     a1r = x[s],   a1i = x[s+1];
    double a2r = x[2*s], a2i = x[2*s+1], a3r = x[3*s], a3i = x[3*s+1];
    double a4r = x[4*s], a4i = x[4*s+1], a5r = x[5*s], a5i = x[5*s+1];
    double a6r = x[6*s], a6i = x[6*s+1], a7r = x[7*s], a7i = x[7*s+1];
    FFT8_MATH;
    if (!cf) { STO(0,X0r,X0i); STO(1,X1r,X1i); STO(2,X2r,X2i); STO(3,X3r,X3i);
               STO(4,X4r,X4i); STO(5,X5r,X5i); STO(6,X6r,X6i); STO(7,X7r,X7i); }
    else     { STOM(0,X0r,X0i); STOM(1,X1r,X1i); STOM(2,X2r,X2i); STOM(3,X3r,X3i);
               STOM(4,X4r,X4i); STOM(5,X5r,X5i); STOM(6,X6r,X6i); STOM(7,X7r,X7i); }
}

/* odd prime p (7..31): X[k] = x0 + sum_j [ cos(2pi jk/p) (x_j + x_{p-j})
 *   - i sin(2pi jk/p) (x_j - x_{p-j}) ],  X[p-k] the +i twin.
 * Inner loop runs over k (contiguous tables, broadcast s_j/d_j) so it vectorizes.
 * One instantiation per prime: h and the padded width HP are compile-time constants
 * (a runtime trip count of 6 or 15 left gcc in the scalar epilogue -- 85 ns at L=13).
 * Tables are built with row stride HP, zero-padded, so the k loop is whole vectors. */
#define HPAD(h) (((h) + 7) & ~7)
#define DEF_CPRIME(P)                                                                  \
static void cprime_##P(const double *restrict x, int is, double *restrict o,           \
                       const double *restrict cf,                                      \
                       const double *restrict pc, const double *restrict ps)           \
{                                                                                      \
    enum { p = P, h = (P - 1) / 2, hp = HPAD((P - 1) / 2) };                           \
    const size_t s = 2*(size_t)is;                                                     \
    double sr[h], si[h], dr[h], di[h];                                                 \
    double Ar[hp], Ai[hp], Br[hp], Bi[hp];                                             \
    const double x0r = x[0], x0i = x[1];                                               \
    double sumr = x0r, sumi = x0i;                                                     \
    for (int j = 0; j < h; j++) {                                                      \
        const double *xa = x + s*(size_t)(j + 1);                                      \
        const double *xb = x + s*(size_t)(p - 1 - j);                                  \
        double ar = xa[0], ai = xa[1], br = xb[0], bi = xb[1];                         \
        sr[j] = ar + br; si[j] = ai + bi;                                              \
        dr[j] = ar - br; di[j] = ai - bi;                                              \
        sumr += sr[j]; sumi += si[j];                                                  \
    }                                                                                  \
    for (int k = 0; k < hp; k++) { Ar[k] = x0r; Ai[k] = x0i; Br[k] = 0.0; Bi[k] = 0.0; } \
    for (int j = 0; j < h; j++) {                                                      \
        const double *Cj = pc + (size_t)j*hp;                                          \
        const double *Sj = ps + (size_t)j*hp;                                          \
        double a = sr[j], b = si[j], e = dr[j], f = di[j];                             \
        for (int k = 0; k < hp; k++) {                                                 \
            Ar[k] += Cj[k]*a; Ai[k] += Cj[k]*b;                                        \
            Br[k] += Sj[k]*e; Bi[k] += Sj[k]*f;                                        \
        }                                                                              \
    }                                                                                  \
    if (!cf) {                                                                         \
        STO(0, sumr, sumi);                                                            \
        for (int k = 0; k < h; k++) {                                                  \
            STO(k + 1,     Ar[k] + Bi[k], Ai[k] - Br[k]);                              \
            STO(p - 1 - k, Ar[k] - Bi[k], Ai[k] + Br[k]);                              \
        }                                                                              \
    } else {                                                                           \
        STOM(0, sumr, sumi);                                                           \
        for (int k = 0; k < h; k++) {                                                  \
            STOM(k + 1,     Ar[k] + Bi[k], Ai[k] - Br[k]);                             \
            STOM(p - 1 - k, Ar[k] - Bi[k], Ai[k] + Br[k]);                             \
        }                                                                              \
    }                                                                                  \
}
DEF_CPRIME(7)  DEF_CPRIME(11) DEF_CPRIME(13) DEF_CPRIME(17)
DEF_CPRIME(19) DEF_CPRIME(23) DEF_CPRIME(29) DEF_CPRIME(31)

static void cprime(const fft1d_plan *pl, const double *restrict x, int is,
                   double *restrict o, const double *restrict cf)
{
    switch (pl->base) {
    case 7:  cprime_7 (x, is, o, cf, pl->pc, pl->ps); break;
    case 11: cprime_11(x, is, o, cf, pl->pc, pl->ps); break;
    case 13: cprime_13(x, is, o, cf, pl->pc, pl->ps); break;
    case 17: cprime_17(x, is, o, cf, pl->pc, pl->ps); break;
    case 19: cprime_19(x, is, o, cf, pl->pc, pl->ps); break;
    case 23: cprime_23(x, is, o, cf, pl->pc, pl->ps); break;
    case 29: cprime_29(x, is, o, cf, pl->pc, pl->ps); break;
    case 31: cprime_31(x, is, o, cf, pl->pc, pl->ps); break;
    }
}

/* ---------------- twiddled combine passes (in place on out[]) ------------ */
/* block layout: y = r sub-results of length m; tw = (r-1) blocks of 2m doubles,
 * tw[(q-1)*2m + 2k] = cos(-2pi qk/(rm)), +1 = sin. */

#define COMB2_LOOP(ST) \
    for (int k = 0; k < m; k++) { \
        double a0r = y[2*k], a0i = y[2*k+1], a1r, a1i; \
        CMUL(a1r,a1i, y[2*(m+k)], y[2*(m+k)+1], tw[2*k], tw[2*k+1]); \
        FFT2_MATH; \
        ST(k, X0r, X0i); ST(m+k, X1r, X1i); \
    }
static inline void combine2(double *restrict y, int m, const double *restrict tw, const double *restrict cf)
{
    if (!cf) { COMB2_LOOP(STY) } else { COMB2_LOOP(STYM) }
}

#define COMB3_LOOP(ST) \
    for (int k = 0; k < m; k++) { \
        double a0r = y[2*k], a0i = y[2*k+1], a1r, a1i, a2r, a2i; \
        CMUL(a1r,a1i, y[2*(m+k)],   y[2*(m+k)+1],   tw[2*k],       tw[2*k+1]); \
        CMUL(a2r,a2i, y[2*(2*m+k)], y[2*(2*m+k)+1], tw[2*(m+k)],   tw[2*(m+k)+1]); \
        FFT3_MATH; \
        ST(k, X0r, X0i); ST(m+k, X1r, X1i); ST(2*m+k, X2r, X2i); \
    }
static inline void combine3(double *restrict y, int m, const double *restrict tw, const double *restrict cf)
{
    if (!cf) { COMB3_LOOP(STY) } else { COMB3_LOOP(STYM) }
}

#define COMB4_LOOP(ST) \
    for (int k = 0; k < m; k++) { \
        double a0r = y[2*k], a0i = y[2*k+1], a1r, a1i, a2r, a2i, a3r, a3i; \
        CMUL(a1r,a1i, y[2*(m+k)],   y[2*(m+k)+1],   tw[2*k],         tw[2*k+1]); \
        CMUL(a2r,a2i, y[2*(2*m+k)], y[2*(2*m+k)+1], tw[2*(m+k)],     tw[2*(m+k)+1]); \
        CMUL(a3r,a3i, y[2*(3*m+k)], y[2*(3*m+k)+1], tw[2*(2*m+k)],   tw[2*(2*m+k)+1]); \
        FFT4_MATH; \
        ST(k, X0r, X0i); ST(m+k, X1r, X1i); ST(2*m+k, X2r, X2i); ST(3*m+k, X3r, X3i); \
    }
static inline void combine4(double *restrict y, int m, const double *restrict tw, const double *restrict cf)
{
    if (!cf) { COMB4_LOOP(STY) } else { COMB4_LOOP(STYM) }
}

#define COMB5_LOOP(ST) \
    for (int k = 0; k < m; k++) { \
        double a0r = y[2*k], a0i = y[2*k+1], a1r,a1i,a2r,a2i,a3r,a3i,a4r,a4i; \
        CMUL(a1r,a1i, y[2*(m+k)],   y[2*(m+k)+1],   tw[2*k],         tw[2*k+1]); \
        CMUL(a2r,a2i, y[2*(2*m+k)], y[2*(2*m+k)+1], tw[2*(m+k)],     tw[2*(m+k)+1]); \
        CMUL(a3r,a3i, y[2*(3*m+k)], y[2*(3*m+k)+1], tw[2*(2*m+k)],   tw[2*(2*m+k)+1]); \
        CMUL(a4r,a4i, y[2*(4*m+k)], y[2*(4*m+k)+1], tw[2*(3*m+k)],   tw[2*(3*m+k)+1]); \
        FFT5_MATH; \
        ST(k, X0r, X0i); ST(m+k, X1r, X1i); ST(2*m+k, X2r, X2i); \
        ST(3*m+k, X3r, X3i); ST(4*m+k, X4r, X4i); \
    }
static inline void combine5(double *restrict y, int m, const double *restrict tw, const double *restrict cf)
{
    if (!cf) { COMB5_LOOP(STY) } else { COMB5_LOOP(STYM) }
}

#define COMB8_LOOP(ST) \
    for (int k = 0; k < m; k++) { \
        double a0r = y[2*k], a0i = y[2*k+1]; \
        double a1r,a1i,a2r,a2i,a3r,a3i,a4r,a4i,a5r,a5i,a6r,a6i,a7r,a7i; \
        CMUL(a1r,a1i, y[2*(m+k)],   y[2*(m+k)+1],   tw[2*k],         tw[2*k+1]); \
        CMUL(a2r,a2i, y[2*(2*m+k)], y[2*(2*m+k)+1], tw[2*(m+k)],     tw[2*(m+k)+1]); \
        CMUL(a3r,a3i, y[2*(3*m+k)], y[2*(3*m+k)+1], tw[2*(2*m+k)],   tw[2*(2*m+k)+1]); \
        CMUL(a4r,a4i, y[2*(4*m+k)], y[2*(4*m+k)+1], tw[2*(3*m+k)],   tw[2*(3*m+k)+1]); \
        CMUL(a5r,a5i, y[2*(5*m+k)], y[2*(5*m+k)+1], tw[2*(4*m+k)],   tw[2*(4*m+k)+1]); \
        CMUL(a6r,a6i, y[2*(6*m+k)], y[2*(6*m+k)+1], tw[2*(5*m+k)],   tw[2*(5*m+k)+1]); \
        CMUL(a7r,a7i, y[2*(7*m+k)], y[2*(7*m+k)+1], tw[2*(6*m+k)],   tw[2*(6*m+k)+1]); \
        FFT8_MATH; \
        ST(k, X0r, X0i); ST(m+k, X1r, X1i); ST(2*m+k, X2r, X2i); ST(3*m+k, X3r, X3i); \
        ST(4*m+k, X4r, X4i); ST(5*m+k, X5r, X5i); ST(6*m+k, X6r, X6i); ST(7*m+k, X7r, X7i); \
    }
static inline void combine8(double *restrict y, int m, const double *restrict tw, const double *restrict cf)
{
    if (!cf) { COMB8_LOOP(STY) } else { COMB8_LOOP(STYM) }
}

/* ---------------- execution ---------------------------------------------- */

static void run_base(const fft1d_plan *pl, const double *restrict x, int is,
                     double *restrict o, const double *restrict cf)
{
    switch (pl->base) {
    case 2: c2(x, is, o, cf); break;
    case 3: c3(x, is, o, cf); break;
    case 4: c4(x, is, o, cf); break;
    case 5: c5(x, is, o, cf); break;
    case 8: c8(x, is, o, cf); break;
    default: cprime(pl, x, is, o, cf); break;
    }
}

static void run_combine(int r, double *restrict y, int m, const double *restrict tw,
                        const double *restrict cf)
{
    switch (r) {
    case 2: combine2(y, m, tw, cf); break;
    case 3: combine3(y, m, tw, cf); break;
    case 4: combine4(y, m, tw, cf); break;
    case 5: combine5(y, m, tw, cf); break;
    case 8: combine8(y, m, tw, cf); break;
    }
}

/* ---- specialized paths for the graded sizes: every stride, offset and trip
 * count is a literal, so the inlined codelets/combines unroll and vectorize.
 * The shapes mirror exactly what plan_factor produces for these L. ---- */

static void fft32_fast(const fft1d_plan *pl, const double *restrict x,
                       double *restrict o, const double *restrict cf)
{
    c8(x,     4, o,      NULL);
    c8(x + 2, 4, o + 16, NULL);
    c8(x + 4, 4, o + 32, NULL);
    c8(x + 6, 4, o + 48, NULL);
    combine4(o, 8, pl->lev[0].tw, cf);
}

static void fft64_fast(const fft1d_plan *pl, const double *restrict x,
                       double *restrict o, const double *restrict cf)
{
    for (int q = 0; q < 8; q++)
        c8(x + 2*q, 8, o + 16*q, NULL);
    combine8(o, 8, pl->lev[0].tw, cf);
}

static void fft128_fast(const fft1d_plan *pl, const double *restrict x,
                        double *restrict o, const double *restrict cf)
{
    for (int q = 0; q < 8; q++)                 /* offs: 0,2,..,14 then 1,3,..,15 */
        c8(x + 4*q, 16, o + 16*q, NULL);
    for (int q = 0; q < 8; q++)
        c8(x + 4*q + 2, 16, o + 128 + 16*q, NULL);
    combine8(o,       8, pl->lev[0].tw, NULL);
    combine8(o + 128, 8, pl->lev[0].tw, NULL);
    combine2(o, 64, pl->lev[1].tw, cf);
}

static void fft60_fast(const fft1d_plan *pl, const double *restrict x,
                       double *restrict o, const double *restrict cf)
{
    /* offs: q + 3*q' for q=0..2 (outer), q'=0..3: 0,3,6,9, 1,4,7,10, 2,5,8,11 */
    for (int q = 0; q < 12; q++) {
        static const int offs12[12] = {0,3,6,9, 1,4,7,10, 2,5,8,11};
        c5(x + 2*offs12[q], 12, o + 10*q, NULL);
    }
    combine4(o,      5, pl->lev[0].tw, NULL);
    combine4(o + 40, 5, pl->lev[0].tw, NULL);
    combine4(o + 80, 5, pl->lev[0].tw, NULL);
    combine3(o, 20, pl->lev[1].tw, cf);
}

/* one transform; if cf != NULL the chain map is fused into the final stage */
static void exec_one(const fft1d_plan *pl, const double *restrict in,
                     double *restrict out, const double *restrict cf)
{
    switch (pl->L) {
    case 13: cprime_13(in, 1, out, cf, pl->pc, pl->ps); return;
    case 31: cprime_31(in, 1, out, cf, pl->pc, pl->ps); return;
    case 32: fft32_fast(pl, in, out, cf);  return;
    case 60: fft60_fast(pl, in, out, cf);  return;
    case 64: fft64_fast(pl, in, out, cf);  return;
    case 128: fft128_fast(pl, in, out, cf); return;
    default: break;
    }
    const int nb = pl->nb, base = pl->base;
    const double *cfb = (pl->nlev == 0) ? cf : NULL;
    for (int i = 0; i < nb; i++)
        run_base(pl, in + 2*(size_t)pl->offs[i], nb, out + 2*(size_t)i*base, cfb);
    for (int j = 0; j < pl->nlev; j++) {
        const level_t *lv = &pl->lev[j];
        const double *cfl = (j == pl->nlev - 1) ? cf : NULL;
        const size_t bs = 2*(size_t)lv->r*(size_t)lv->m;
        for (int blk = 0; blk < lv->nblocks; blk++)
            run_combine(lv->r, out + (size_t)blk*bs, lv->m, lv->tw,
                        cfl ? cfl + (size_t)blk*bs : NULL);
    }
}

/* ---------------- API ----------------------------------------------------- */

int fft1d_supports(int L)
{
    int base, chain[MAXLEV], n;
    return plan_factor(L, &base, chain, &n);
}

fft1d_plan *fft1d_create(int L, int batch)
{
    int base, chain[MAXLEV], nlev;
    if (batch < 1 || !plan_factor(L, &base, chain, &nlev)) return NULL;
    fft1d_plan *pl = calloc(1, sizeof *pl);
    if (!pl) return NULL;
    pl->L = L; pl->batch = batch; pl->base = base; pl->nlev = nlev;
    pl->nb = L / base;

    pl->offs = malloc((size_t)pl->nb * sizeof *pl->offs);
    if (!pl->offs) goto fail;
    int *op = pl->offs;
    emit_offs(chain, nlev - 1, 1, 0, &op);

    int m = base;
    for (int j = 0; j < nlev; j++) {
        int r = chain[j], size = r * m;
        pl->lev[j].r = r; pl->lev[j].m = m; pl->lev[j].nblocks = L / size;
        double *tw = malloc((size_t)(r - 1) * 2 * m * sizeof *tw);
        if (!tw) goto fail;
        for (int q = 1; q < r; q++)
            for (int k = 0; k < m; k++) {
                double ang = -2.0 * M_PI * (double)((q * k) % size) / (double)size;
                tw[(size_t)(q - 1)*2*m + 2*k]     = cos(ang);
                tw[(size_t)(q - 1)*2*m + 2*k + 1] = sin(ang);
            }
        pl->lev[j].tw = tw;
        m = size;
    }

    if (base > 5) {          /* odd-prime base: cos/sin tables, rows padded to HP */
        int h = (base - 1) / 2, hp = HPAD(h);
        pl->h = h;
        if (posix_memalign((void **)&pl->pc, 64, (size_t)h * hp * sizeof *pl->pc) ||
            posix_memalign((void **)&pl->ps, 64, (size_t)h * hp * sizeof *pl->ps))
            goto fail;
        for (int j = 1; j <= h; j++)
            for (int k = 1; k <= hp; k++) {
                double cv = 0.0, sv = 0.0;
                if (k <= h) {
                    double ang = 2.0 * M_PI * (double)((j * k) % base) / (double)base;
                    cv = cos(ang); sv = sin(ang);
                }
                pl->pc[(size_t)(j - 1)*hp + (k - 1)] = cv;
                pl->ps[(size_t)(j - 1)*hp + (k - 1)] = sv;
            }
    }

    {
        void *sm = NULL;
        if (posix_memalign(&sm, 64, 2 * (size_t)L * sizeof(double)) != 0) goto fail;
        pl->small = sm;
    }
    return pl;
fail:
    fft1d_destroy(pl);
    return NULL;
}

void fft1d_execute(fft1d_plan *pl, const double _Complex *in, double _Complex *out)
{
    const double *src = (const double *)in;
    double *dst = (double *)out;
    const size_t step = 2 * (size_t)pl->L;
    for (int b = 0; b < pl->batch; b++)
        exec_one(pl, src + (size_t)b*step, dst + (size_t)b*step, NULL);
}

/* own the whole m-step map chain: state <- (FFT(state)+c)/(1+|FFT(state)+c|).
 * Batch elements are independent, so each is chained through all m steps by
 * itself (L1-resident); the map is fused into the final-stage stores.  Buffer
 * parity is arranged so step m lands directly in final_out. */
void fft1d_chain(fft1d_plan *pl, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    if (m < 1) return;
    const size_t step = 2 * (size_t)pl->L;
    for (int b = 0; b < pl->batch; b++) {
        const double *src = (const double *)x0 + (size_t)b*step;
        const double *cb  = (const double *)c  + (size_t)b*step;
        double *FB = (double *)final_out + (size_t)b*step;
        double *A  = pl->small;
        double *dst = (m & 1) ? FB : A;
        for (int s = 0; s < m; s++) {
            exec_one(pl, src, dst, cb);
            src = dst;
            dst = (src == FB) ? A : FB;
        }
    }
}

void fft1d_destroy(fft1d_plan *pl)
{
    if (!pl) return;
    free(pl->offs);
    for (int j = 0; j < pl->nlev; j++) free(pl->lev[j].tw);
    free(pl->pc); free(pl->ps); free(pl->small);
    free(pl);
}
