/* d1_composite -- PFA / mixed-radix coprime split for composite L.
 *
 * L=60 (the graded size, 2^2*3*5): Good-Thomas PFA 3x4x5. The coprime split has NO
 * twiddle factors: input map n=(20n1+15n2+12n3)%60, output map k=(40k1+45k2+36k3)%60
 * turn DFT_60 into a pure 3x4x5 3D DFT (20 DFT-3s, 15 DFT-4s, 12 DFT-5s), all real
 * constants, FMA throughout. Stages 1+2 are fused per n3-plane: each 3x4 sub-plane
 * (12 points) goes DFT3+DFT4 entirely in registers, so the data crosses memory once
 * between the fused 12-point stage and the DFT-5 stage instead of twice.
 *
 * Three codelet forms:
 *   - v2 (SSE, one complex per xmm): B=1 latency cell and batch remainders. addsub +
 *     swap implement the +-i rotations; everything else is FMA on {re,im} pairs.
 *   - 8-lane zmm split SoA (macro): batched cells; 8 transforms in the lanes via 8x8
 *     double transposes, every scalar op is one vector op, zero shuffles inside.
 *   - scalar split SoA (macro): chain remainder, keeps the map auto-vectorizable.
 * fft1d_chain is owned: each group of 8 transforms is transposed ONCE, chained m steps
 * entirely L1-resident (state+c+scratch ~30 KiB), transposed back once. The driver
 * fallback pays the AoS round-trip and a separate map pass every step; we do not.
 *
 * L=12/24/36 keep the dense O(L^2) floor this round (not in cases.txt; correctness only).
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include "../fft1d_api.h"

struct fft1d_plan { int L, batch; double _Complex *w; /* dense fallback only */ };

const char *fft1d_name(void){ return "d1_composite"; }
const char *fft1d_description(void){ return "PFA 3x4x5 Good-Thomas (no twiddles), fused 12-pt stage, zmm batch lanes, L1-resident owned chain"; }
int fft1d_supports(int L){ return L == 12 || L == 24 || L == 36 || L == 60; }

/* ---- Good-Thomas index maps for 60 = 3*4*5, linear index n1*20 + n2*5 + n3 ---- */
static const int PIN[60] = {
    0,12,24,36,48,15,27,39,51,3,30,42,54,6,18,45,57,9,21,33,
    20,32,44,56,8,35,47,59,11,23,50,2,14,26,38,5,17,29,41,53,
    40,52,4,16,28,55,7,19,31,43,10,22,34,46,58,25,37,49,1,13 };
static const int POUT[60] = {
    0,36,12,48,24,45,21,57,33,9,30,6,42,18,54,15,51,27,3,39,
    40,16,52,28,4,25,1,37,13,49,10,46,22,58,34,55,31,7,43,19,
    20,56,32,8,44,5,41,17,53,29,50,26,2,38,14,35,11,47,23,59 };

#define K3  0.86602540378443864676   /* sin(2pi/3) */
#define K5C1  0.30901699437494742410 /* cos(2pi/5) */
#define K5C2 -0.80901699437494742410 /* cos(4pi/5) */
#define K5S1  0.95105651629515357212 /* sin(2pi/5) */
#define K5S2  0.58778525229247312917 /* sin(4pi/5) */

/* ================= split re/im codelet (macro), fused stages 1+2 ================= */
#define DEF_DFT60(FN, VT, IS, OS)                                              \
static void FN(const VT *xr, const VT *xi, VT *yr, VT *yi)                     \
{                                                                              \
    VT tr[60], ti[60];                                                         \
    _Pragma("GCC unroll 5")                                                    \
    for (int n3 = 0; n3 < 5; ++n3) {                                           \
        VT br[12], bi[12];                       /* [n1*4 + n2] */             \
        _Pragma("GCC unroll 4")                                                \
        for (int n2 = 0; n2 < 4; ++n2) {         /* DFT-3 over n1 */           \
            const int u = n2*5 + n3;                                           \
            const int i0 = PIN[u], i1 = PIN[u+20], i2 = PIN[u+40];             \
            VT ar = xr[IS*i0], ai = xi[IS*i0];                                 \
            VT pr = xr[IS*i1], pi = xi[IS*i1];                                 \
            VT qr = xr[IS*i2], qi = xi[IS*i2];                                 \
            VT sr = pr + qr, si = pi + qi;                                     \
            VT dr = pr - qr, di = pi - qi;                                     \
            VT mr = ar - 0.5*sr, mi = ai - 0.5*si;                             \
            br[n2]   = ar + sr;        bi[n2]   = ai + si;                     \
            br[4+n2] = mr + K3*di;     bi[4+n2] = mi - K3*dr;                  \
            br[8+n2] = mr - K3*di;     bi[8+n2] = mi + K3*dr;                  \
        }                                                                      \
        _Pragma("GCC unroll 3")                                                \
        for (int n1 = 0; n1 < 3; ++n1) {         /* DFT-4 over n2 */           \
            VT a0r = br[n1*4],   a0i = bi[n1*4];                               \
            VT a1r = br[n1*4+1], a1i = bi[n1*4+1];                             \
            VT a2r = br[n1*4+2], a2i = bi[n1*4+2];                             \
            VT a3r = br[n1*4+3], a3i = bi[n1*4+3];                             \
            VT e0r = a0r + a2r, e0i = a0i + a2i;                               \
            VT e1r = a0r - a2r, e1i = a0i - a2i;                               \
            VT o0r = a1r + a3r, o0i = a1i + a3i;                               \
            VT o1r = a1r - a3r, o1i = a1i - a3i;                               \
            const int j = n1*20 + n3;                                          \
            tr[j]    = e0r + o0r;  ti[j]    = e0i + o0i;                       \
            tr[j+5]  = e1r + o1i;  ti[j+5]  = e1i - o1r;                       \
            tr[j+10] = e0r - o0r;  ti[j+10] = e0i - o0i;                       \
            tr[j+15] = e1r - o1i;  ti[j+15] = e1i + o1r;                       \
        }                                                                      \
    }                                                                          \
    _Pragma("GCC unroll 12")                                                   \
    for (int q = 0; q < 12; ++q) {               /* DFT-5 over n3 */           \
        const int j = q*5;                                                     \
        VT x0r = tr[j],   x0i = ti[j];                                         \
        VT x1r = tr[j+1], x1i = ti[j+1];                                       \
        VT x2r = tr[j+2], x2i = ti[j+2];                                       \
        VT x3r = tr[j+3], x3i = ti[j+3];                                       \
        VT x4r = tr[j+4], x4i = ti[j+4];                                       \
        VT t1r = x1r + x4r, t1i = x1i + x4i;                                   \
        VT t2r = x2r + x3r, t2i = x2i + x3i;                                   \
        VT t3r = x1r - x4r, t3i = x1i - x4i;                                   \
        VT t4r = x2r - x3r, t4i = x2i - x3i;                                   \
        VT aar = x0r + K5C1*t1r + K5C2*t2r;                                    \
        VT aai = x0i + K5C1*t1i + K5C2*t2i;                                    \
        VT bbr = x0r + K5C2*t1r + K5C1*t2r;                                    \
        VT bbi = x0i + K5C2*t1i + K5C1*t2i;                                    \
        VT ppr = K5S1*t3r + K5S2*t4r, ppi = K5S1*t3i + K5S2*t4i;               \
        VT qqr = K5S2*t3r - K5S1*t4r, qqi = K5S2*t3i - K5S1*t4i;               \
        const int o0 = POUT[j],   o1 = POUT[j+1], o2 = POUT[j+2],              \
                  o3 = POUT[j+3], o4 = POUT[j+4];                              \
        yr[OS*o0] = x0r + t1r + t2r;  yi[OS*o0] = x0i + t1i + t2i;             \
        yr[OS*o1] = aar + ppi;        yi[OS*o1] = aai - ppr;                   \
        yr[OS*o2] = bbr + qqi;        yi[OS*o2] = bbi - qqr;                   \
        yr[OS*o3] = bbr - qqi;        yi[OS*o3] = bbi + qqr;                   \
        yr[OS*o4] = aar - ppi;        yi[OS*o4] = aai + ppr;                   \
    }                                                                          \
}

/* Unfused 3-pass variant for the 8-lane zmm instance: the fused 12-point plane keeps
 * 24+ zmm live and gcc spills ~200 moves; three flat passes stay within the register
 * file and let the t round-trip ride L1 instead of spill slots. */
#define DEF_DFT60_FLAT(FN, VT, IS, OS)                                         \
static void FN(const VT *xr, const VT *xi, VT *yr, VT *yi)                     \
{                                                                              \
    VT tr[60], ti[60];                                                         \
    _Pragma("GCC unroll 20")                                                   \
    for (int u = 0; u < 20; ++u) {                 /* DFT-3 over n1 */         \
        const int i0 = PIN[u], i1 = PIN[u + 20], i2 = PIN[u + 40];             \
        VT ar = xr[IS*i0], ai = xi[IS*i0];                                     \
        VT pr = xr[IS*i1], pi = xi[IS*i1];                                     \
        VT qr = xr[IS*i2], qi = xi[IS*i2];                                     \
        VT sr = pr + qr, si = pi + qi;                                         \
        VT dr = pr - qr, di = pi - qi;                                         \
        VT mr = ar - 0.5*sr, mi = ai - 0.5*si;                                 \
        tr[u]    = ar + sr;      ti[u]    = ai + si;                           \
        tr[u+20] = mr + K3*di;   ti[u+20] = mi - K3*dr;                        \
        tr[u+40] = mr - K3*di;   ti[u+40] = mi + K3*dr;                        \
    }                                                                          \
    _Pragma("GCC unroll 15")                                                   \
    for (int v = 0; v < 15; ++v) {                 /* DFT-4 over n2 */         \
        const int j = (v/5)*20 + (v%5);                                        \
        VT a0r = tr[j],    a0i = ti[j];                                        \
        VT a1r = tr[j+5],  a1i = ti[j+5];                                      \
        VT a2r = tr[j+10], a2i = ti[j+10];                                     \
        VT a3r = tr[j+15], a3i = ti[j+15];                                     \
        VT e0r = a0r + a2r, e0i = a0i + a2i;                                   \
        VT e1r = a0r - a2r, e1i = a0i - a2i;                                   \
        VT o0r = a1r + a3r, o0i = a1i + a3i;                                   \
        VT o1r = a1r - a3r, o1i = a1i - a3i;                                   \
        tr[j]    = e0r + o0r;  ti[j]    = e0i + o0i;                           \
        tr[j+5]  = e1r + o1i;  ti[j+5]  = e1i - o1r;                           \
        tr[j+10] = e0r - o0r;  ti[j+10] = e0i - o0i;                           \
        tr[j+15] = e1r - o1i;  ti[j+15] = e1i + o1r;                           \
    }                                                                          \
    _Pragma("GCC unroll 12")                                                   \
    for (int q = 0; q < 12; ++q) {                 /* DFT-5 over n3 */         \
        const int j = q*5;                                                     \
        VT x0r = tr[j],   x0i = ti[j];                                         \
        VT x1r = tr[j+1], x1i = ti[j+1];                                       \
        VT x2r = tr[j+2], x2i = ti[j+2];                                       \
        VT x3r = tr[j+3], x3i = ti[j+3];                                       \
        VT x4r = tr[j+4], x4i = ti[j+4];                                       \
        VT t1r = x1r + x4r, t1i = x1i + x4i;                                   \
        VT t2r = x2r + x3r, t2i = x2i + x3i;                                   \
        VT t3r = x1r - x4r, t3i = x1i - x4i;                                   \
        VT t4r = x2r - x3r, t4i = x2i - x3i;                                   \
        VT aar = x0r + K5C1*t1r + K5C2*t2r;                                    \
        VT aai = x0i + K5C1*t1i + K5C2*t2i;                                    \
        VT bbr = x0r + K5C2*t1r + K5C1*t2r;                                    \
        VT bbi = x0i + K5C2*t1i + K5C1*t2i;                                    \
        VT ppr = K5S1*t3r + K5S2*t4r, ppi = K5S1*t3i + K5S2*t4i;               \
        VT qqr = K5S2*t3r - K5S1*t4r, qqi = K5S2*t3i - K5S1*t4i;               \
        const int o0 = POUT[j],   o1 = POUT[j+1], o2 = POUT[j+2],              \
                  o3 = POUT[j+3], o4 = POUT[j+4];                              \
        yr[OS*o0] = x0r + t1r + t2r;  yi[OS*o0] = x0i + t1i + t2i;             \
        yr[OS*o1] = aar + ppi;        yi[OS*o1] = aai - ppr;                   \
        yr[OS*o2] = bbr + qqi;        yi[OS*o2] = bbi - qqr;                   \
        yr[OS*o3] = bbr - qqi;        yi[OS*o3] = bbi + qqr;                   \
        yr[OS*o4] = aar - ppi;        yi[OS*o4] = aai + ppr;                   \
    }                                                                          \
}

DEF_DFT60(dft60_sca, double, 1, 1)          /* split scalar (chain remainder) */
DEF_DFT60_FLAT(dft60_soa, __m512d, 1, 1)    /* 8 transforms in the lanes */

/* ================= SSE complex codelet: one {re,im} pair per xmm ================= */
/* +-i rotations via swap + addsub:  m - i*h = addsub(m, -swap(h)),
 *                                   m + i*h = addsub(m,  swap(h)).  */
#define LDC(x, idx)  _mm_loadu_pd((x) + 2*(idx))
#define STC(y, idx, v) _mm_storeu_pd((y) + 2*(idx), (v))
#define SWP(v) _mm_shuffle_pd((v), (v), 1)

static void dft60_v2(const double *x, double *y)
{
    __m128d t[60];
    _Pragma("GCC unroll 5")
    for (int n3 = 0; n3 < 5; ++n3) {
        __m128d b[12];
        _Pragma("GCC unroll 4")
        for (int n2 = 0; n2 < 4; ++n2) {
            const int u = n2*5 + n3;
            __m128d a = LDC(x, PIN[u]);
            __m128d p = LDC(x, PIN[u+20]);
            __m128d q = LDC(x, PIN[u+40]);
            __m128d s = p + q, d = p - q;
            __m128d m = a - 0.5*s;
            __m128d g = K3*SWP(d);
            b[n2]   = a + s;
            b[4+n2] = _mm_addsub_pd(m, -g);
            b[8+n2] = _mm_addsub_pd(m, g);
        }
        _Pragma("GCC unroll 3")
        for (int n1 = 0; n1 < 3; ++n1) {
            __m128d a0 = b[n1*4],   a1 = b[n1*4+1];
            __m128d a2 = b[n1*4+2], a3 = b[n1*4+3];
            __m128d e0 = a0 + a2, e1 = a0 - a2;
            __m128d o0 = a1 + a3, o1 = a1 - a3;
            __m128d so = SWP(o1);
            const int j = n1*20 + n3;
            t[j]    = e0 + o0;
            t[j+5]  = _mm_addsub_pd(e1, -so);
            t[j+10] = e0 - o0;
            t[j+15] = _mm_addsub_pd(e1, so);
        }
    }
    _Pragma("GCC unroll 12")
    for (int q = 0; q < 12; ++q) {
        const int j = q*5;
        __m128d x0 = t[j], x1 = t[j+1], x2 = t[j+2], x3 = t[j+3], x4 = t[j+4];
        __m128d t1 = x1 + x4, t2 = x2 + x3, t3 = x1 - x4, t4 = x2 - x3;
        __m128d aa = x0 + K5C1*t1 + K5C2*t2;
        __m128d bb = x0 + K5C2*t1 + K5C1*t2;
        __m128d pp = K5S1*t3 + K5S2*t4;
        __m128d qq = K5S2*t3 - K5S1*t4;
        __m128d sp = SWP(pp), sq = SWP(qq);
        STC(y, POUT[j],   x0 + t1 + t2);
        STC(y, POUT[j+1], _mm_addsub_pd(aa, -sp));
        STC(y, POUT[j+2], _mm_addsub_pd(bb, -sq));
        STC(y, POUT[j+3], _mm_addsub_pd(bb, sq));
        STC(y, POUT[j+4], _mm_addsub_pd(aa, sp));
    }
}

/* ---- 8x8 double transpose (24 shuffles), used to move 8 AoS rows <-> lane-SoA ---- */
static inline void transpose8(__m512d r[8])
{
    __m512d t[8], u[8];
    t[0]=_mm512_unpacklo_pd(r[0],r[1]); t[1]=_mm512_unpackhi_pd(r[0],r[1]);
    t[2]=_mm512_unpacklo_pd(r[2],r[3]); t[3]=_mm512_unpackhi_pd(r[2],r[3]);
    t[4]=_mm512_unpacklo_pd(r[4],r[5]); t[5]=_mm512_unpackhi_pd(r[4],r[5]);
    t[6]=_mm512_unpacklo_pd(r[6],r[7]); t[7]=_mm512_unpackhi_pd(r[6],r[7]);
    u[0]=_mm512_shuffle_f64x2(t[0],t[2],0x88); u[1]=_mm512_shuffle_f64x2(t[1],t[3],0x88);
    u[2]=_mm512_shuffle_f64x2(t[0],t[2],0xdd); u[3]=_mm512_shuffle_f64x2(t[1],t[3],0xdd);
    u[4]=_mm512_shuffle_f64x2(t[4],t[6],0x88); u[5]=_mm512_shuffle_f64x2(t[5],t[7],0x88);
    u[6]=_mm512_shuffle_f64x2(t[4],t[6],0xdd); u[7]=_mm512_shuffle_f64x2(t[5],t[7],0xdd);
    r[0]=_mm512_shuffle_f64x2(u[0],u[4],0x88);
    r[1]=_mm512_shuffle_f64x2(u[1],u[5],0x88);
    r[2]=_mm512_shuffle_f64x2(u[2],u[6],0x88);
    r[3]=_mm512_shuffle_f64x2(u[3],u[7],0x88);
    r[4]=_mm512_shuffle_f64x2(u[0],u[4],0xdd);
    r[5]=_mm512_shuffle_f64x2(u[1],u[5],0xdd);
    r[6]=_mm512_shuffle_f64x2(u[2],u[6],0xdd);
    r[7]=_mm512_shuffle_f64x2(u[3],u[7],0xdd);
}

/* 8 consecutive AoS transforms (row stride 120 doubles) -> split lane-SoA gre/gim[60]. */
static void group_in(const double _Complex *in, __m512d *gre, __m512d *gim)
{
    const double *b0 = (const double *)in;
    for (int jb = 0; jb < 15; ++jb) {          /* 4 complex points per block */
        __m512d r[8];
        for (int b = 0; b < 8; ++b) r[b] = _mm512_loadu_pd(b0 + (size_t)b*120 + 8*jb);
        transpose8(r);
        gre[4*jb]   = r[0]; gim[4*jb]   = r[1];
        gre[4*jb+1] = r[2]; gim[4*jb+1] = r[3];
        gre[4*jb+2] = r[4]; gim[4*jb+2] = r[5];
        gre[4*jb+3] = r[6]; gim[4*jb+3] = r[7];
    }
}
static void group_out(double _Complex *out, const __m512d *gre, const __m512d *gim)
{
    double *b0 = (double *)out;
    for (int jb = 0; jb < 15; ++jb) {
        __m512d r[8];
        r[0]=gre[4*jb];   r[1]=gim[4*jb];
        r[2]=gre[4*jb+1]; r[3]=gim[4*jb+1];
        r[4]=gre[4*jb+2]; r[5]=gim[4*jb+2];
        r[6]=gre[4*jb+3]; r[7]=gim[4*jb+3];
        transpose8(r);
        for (int b = 0; b < 8; ++b) _mm512_storeu_pd(b0 + (size_t)b*120 + 8*jb, r[b]);
    }
}

/* The graded map step on split arrays: s <- w/(1+|w|), w = y + c. Elementwise, so it is
 * layout-agnostic: n=60 on scalar split arrays, n=480 on a flattened lane-SoA group.
 * Plain sqrt on squares so gcc vectorizes it (-fno-math-errno). */
static void map_split(const double *yr, const double *yi, const double *cr,
                      const double *ci, double *sr, double *si, int n)
{
    for (int i = 0; i < n; ++i) {
        double wr = yr[i] + cr[i], wi = yi[i] + ci[i];
        double sc = 1.0 / (1.0 + sqrt(wr*wr + wi*wi));
        sr[i] = wr * sc;
        si[i] = wi * sc;
    }
}

/* ---- dense floor for the not-yet-specialized sizes (12/24/36) ---- */
static void dense_execute(const fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    int L = p->L;
    for (int b = 0; b < p->batch; ++b) {
        const double _Complex *x = in + (size_t)b*L;
        double _Complex *y = out + (size_t)b*L;
        for (int k = 0; k < L; ++k) {
            double _Complex s = 0;
            const double _Complex *wr = p->w + (size_t)k*L;
            for (int j = 0; j < L; ++j) s += wr[j]*x[j];
            y[k] = s;
        }
    }
}

fft1d_plan *fft1d_create(int L, int batch)
{
    if (!fft1d_supports(L)) return NULL;
    fft1d_plan *p = malloc(sizeof *p); if (!p) return NULL;
    p->L = L; p->batch = batch; p->w = NULL;
    if (L != 60) {
        p->w = malloc((size_t)L*L*sizeof *p->w);
        if (!p->w) { free(p); return NULL; }
        for (int k = 0; k < L; ++k)
            for (int j = 0; j < L; ++j) {
                double ph = -2.0*M_PI*((k*j)%L)/L;
                p->w[(size_t)k*L+j] = cos(ph) + I*sin(ph);
            }
    }
    return p;
}

void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out)
{
    if (p->L != 60) { dense_execute(p, in, out); return; }
    int b = 0;
    for (; b + 8 <= p->batch; b += 8) {
        __m512d gre[60], gim[60], hre[60], him[60];
        group_in(in + (size_t)b*60, gre, gim);
        dft60_soa(gre, gim, hre, him);
        group_out(out + (size_t)b*60, hre, him);
    }
    for (; b < p->batch; ++b)
        dft60_v2((const double *)(in + (size_t)b*60), (double *)(out + (size_t)b*60));
}

/* Owned m-step chain: state <- (FFT(state)+c)/(1+|FFT(state)+c|), repeated m times.
 * Each group of 8 transforms is chained to completion while L1-resident. */
void fft1d_chain(fft1d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    if (p->L != 60) {                          /* dense floor, driver-equivalent */
        int L = p->L;
        size_t count = (size_t)L*p->batch;
        double _Complex *z = malloc(count*sizeof *z);
        double _Complex *s = malloc(count*sizeof *s);
        if (!z || !s) { free(z); free(s); return; }
        memcpy(s, x0, count*sizeof *s);
        for (int st = 0; st < m; ++st) {
            dense_execute(p, s, z);
            for (size_t i = 0; i < count; ++i) {
                double wr = creal(z[i]) + creal(c[i]), wi = cimag(z[i]) + cimag(c[i]);
                double sc = 1.0/(1.0 + sqrt(wr*wr + wi*wi));
                s[i] = wr*sc + I*(wi*sc);
            }
        }
        memcpy(final_out, s, count*sizeof *s);
        free(z); free(s);
        return;
    }
    int b = 0;
    for (; b + 8 <= p->batch; b += 8) {
        __m512d sre[60], sim[60], yre[60], yim[60], cre[60], cim[60];
        group_in(x0 + (size_t)b*60, sre, sim);
        group_in(c  + (size_t)b*60, cre, cim);
        for (int st = 0; st < m; ++st) {
            dft60_soa(sre, sim, yre, yim);
            map_split((const double *)yre, (const double *)yim,
                      (const double *)cre, (const double *)cim,
                      (double *)sre, (double *)sim, 480);
        }
        group_out(final_out + (size_t)b*60, sre, sim);
    }
    for (; b < p->batch; ++b) {                /* B=1 and remainders: split scalar */
        double sr[60], si[60], yr[60], yi[60], cr[60], ci[60];
        const double *xp = (const double *)(x0 + (size_t)b*60);
        const double *cp = (const double *)(c  + (size_t)b*60);
        for (int j = 0; j < 60; ++j) {
            sr[j] = xp[2*j]; si[j] = xp[2*j+1];
            cr[j] = cp[2*j]; ci[j] = cp[2*j+1];
        }
        for (int st = 0; st < m; ++st) {
            dft60_sca(sr, si, yr, yi);
            map_split(yr, yi, cr, ci, sr, si, 60);
        }
        double *op = (double *)(final_out + (size_t)b*60);
        for (int j = 0; j < 60; ++j) { op[2*j] = sr[j]; op[2*j+1] = si[j]; }
    }
}

void fft1d_destroy(fft1d_plan *p){ if (!p) return; free(p->w); free(p); }
