// Batched 3D complex-to-complex FFT + nonlinear iteration; AVX-512; 1 thread.
// State: split re/im planes; slab layout: row r (axis-1) stride SLAB=L*L+8,
// within slab: middle axis * L + fast axis.
// Iteration: passA (axis-1), passB (axis-2), passC (axis-3, via 8x8 transposes,
// fused with +c and z/(1+|z|); output written with axes 2,3 swapped so the
// orientation alternates; c is kept in both orientations).
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef __m512d vd;
#define AI __attribute__((always_inline)) inline
#define VADD _mm512_add_pd
#define VSUB _mm512_sub_pd
#define VMUL _mm512_mul_pd
#define VFMA _mm512_fmadd_pd
#define VFNMA _mm512_fnmadd_pd
#define VSET1 _mm512_set1_pd
#define VLOADU _mm512_loadu_pd
#define VLOADZ _mm512_maskz_loadu_pd
#define VSTOREM _mm512_mask_storeu_pd

static AI vd bcv(const double *p){ return _mm512_set1_pd(*p); }

/* ------------------------------------------------ tables ---------- */
static double C13[6][6], S13[6][6];
static double C17[8][8], S17[8][8];
static double C23[11][11], S23[11][11];
static double W64r[8][8], W64i[8][8];
static double W9r[3][3],  W9i[3][3];
static double C5[2][2],  S5[2][2];

typedef struct { double *sre,*sim,*t1re,*t1im,*t2re,*t2im,*cre,*cim,*cfre,*cfim; } bufs_t;
static bufs_t BUFS[65];

static void trig(long num, long L, double *cr, double *ci){
    long a = num % L; if (a < 0) a += L;
    long double ang = -2.0L * 3.14159265358979323846264338327950288L * (long double)a / (long double)L;
    *cr = (double)cosl(ang);
    *ci = (double)sinl(ang);
}

static void mk_bufs(long L){
    long SLAB = L*L + 8;
    size_t plane = (size_t)(L*SLAB + 16);
    bufs_t *b = &BUFS[L];
    double **fields[10] = {&b->sre,&b->sim,&b->t1re,&b->t1im,&b->t2re,&b->t2im,&b->cre,&b->cim,&b->cfre,&b->cfim};
    for (int i = 0; i < 10; i++){
        *fields[i] = (double*)aligned_alloc(64, sizeof(double)*plane);
        memset(*fields[i], 0, sizeof(double)*plane);
    }
}

void fftmod_init(void){
    for (int k = 1; k <= 6; k++) for (int j = 1; j <= 6; j++){
        trig((long)k*j, 13, &C13[k-1][j-1], &S13[k-1][j-1]);
    }
    for (int k = 1; k <= 8; k++) for (int j = 1; j <= 8; j++){
        trig((long)k*j, 17, &C17[k-1][j-1], &S17[k-1][j-1]);
    }
    for (int k = 1; k <= 11; k++) for (int j = 1; j <= 11; j++){
        trig((long)k*j, 23, &C23[k-1][j-1], &S23[k-1][j-1]);
    }
    for (int k = 1; k <= 2; k++) for (int j = 1; j <= 2; j++){
        trig((long)k*j, 5, &C5[k-1][j-1], &S5[k-1][j-1]);
    }
    for (int b = 0; b < 8; b++) for (int k = 0; k < 8; k++)
        trig((long)b*k, 64, &W64r[b][k], &W64i[b][k]);
    for (int b = 0; b < 3; b++) for (int k = 0; k < 3; k++)
        trig((long)b*k, 9, &W9r[b][k], &W9i[b][k]);
    mk_bufs(6); mk_bufs(8); mk_bufs(13); mk_bufs(17);
    mk_bufs(23); mk_bufs(36); mk_bufs(45); mk_bufs(64);
}

/* ------------------------------------------------ transpose -------- */
static AI void transp8_from(const vd *s, vd r[8]){
    vd t0 = _mm512_unpacklo_pd(s[0], s[1]);
    vd t1 = _mm512_unpackhi_pd(s[0], s[1]);
    vd t2 = _mm512_unpacklo_pd(s[2], s[3]);
    vd t3 = _mm512_unpackhi_pd(s[2], s[3]);
    vd t4 = _mm512_unpacklo_pd(s[4], s[5]);
    vd t5 = _mm512_unpackhi_pd(s[4], s[5]);
    vd t6 = _mm512_unpacklo_pd(s[6], s[7]);
    vd t7 = _mm512_unpackhi_pd(s[6], s[7]);
    vd u0 = _mm512_shuffle_f64x2(t0, t2, 0x88);
    vd u1 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
    vd u2 = _mm512_shuffle_f64x2(t1, t3, 0x88);
    vd u3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
    vd u4 = _mm512_shuffle_f64x2(t4, t6, 0x88);
    vd u5 = _mm512_shuffle_f64x2(t4, t6, 0xDD);
    vd u6 = _mm512_shuffle_f64x2(t5, t7, 0x88);
    vd u7 = _mm512_shuffle_f64x2(t5, t7, 0xDD);
    r[0] = _mm512_shuffle_f64x2(u0, u4, 0x88);
    r[4] = _mm512_shuffle_f64x2(u0, u4, 0xDD);
    r[1] = _mm512_shuffle_f64x2(u2, u6, 0x88);
    r[5] = _mm512_shuffle_f64x2(u2, u6, 0xDD);
    r[2] = _mm512_shuffle_f64x2(u1, u5, 0x88);
    r[6] = _mm512_shuffle_f64x2(u1, u5, 0xDD);
    r[3] = _mm512_shuffle_f64x2(u3, u7, 0x88);
    r[7] = _mm512_shuffle_f64x2(u3, u7, 0xDD);
}

/* -------------------------------------------- nonlinear map -------- */
static AI void map8(vd zr, vd zi, vd *xr, vd *xi){
    vd m = VFMA(zr, zr, VMUL(zi, zi));
    vd y = _mm512_rsqrt14_pd(m);
    vd hm = VMUL(m, VSET1(0.5));
    vd th = VSET1(1.5);
    y = VMUL(y, VFNMA(VMUL(hm, y), y, th));
    y = VMUL(y, VFNMA(VMUL(hm, y), y, th));
    vd s = VMUL(m, y);
    vd d = VADD(VSET1(1.0), s);
    vd r = _mm512_rcp14_pd(d);
    r = VMUL(r, VFNMA(d, r, VSET1(2.0)));
    r = VMUL(r, VFNMA(d, r, VSET1(2.0)));
    __mmask8 k0 = _mm512_cmp_pd_mask(m, VSET1(2.2250738585072014e-308), _CMP_LT_OQ);
    r = _mm512_mask_blend_pd(k0, r, VSET1(1.0));
    *xr = VMUL(zr, r);
    *xi = VMUL(zi, r);
}

/* ======================= FFT cores =========================
   Each core: reads rows via LD (input base pointers pr/pi, row stride rs,
   optional load-mask), writes outputs via OUT (base or_/oi_, row stride os,
   store mask sm; if cq non-null, fuse +c and nonlinear map).
   All strides/flags are compile-time constants at each call site.  */

#define CORE_PROLOG \
    const int MS = masked; (void)MS;

#define LDR(r) (MS ? VLOADZ(lm, pr + (long)(r)*rs) : VLOADU(pr + (long)(r)*rs))
#define LDI(r) (MS ? VLOADZ(lm, pi + (long)(r)*rs) : VLOADU(pi + (long)(r)*rs))

#define OUT(k, vr, vi) do{ \
    if (cq){ \
        vd cr_ = VLOADZ(sm, cq + (long)(k)*os); \
        vd ci_ = VLOADZ(sm, cqi + (long)(k)*os); \
        vd zr_ = VADD(vr, cr_), zi_ = VADD(vi, ci_); \
        vd xr_, xi_; map8(zr_, zi_, &xr_, &xi_); \
        VSTOREM(or_ + (long)(k)*os, sm, xr_); \
        VSTOREM(oi_ + (long)(k)*os, sm, xi_); \
    } else { \
        VSTOREM(or_ + (long)(k)*os, sm, vr); \
        VSTOREM(oi_ + (long)(k)*os, sm, vi); \
    } \
}while(0)

#define CORE_ARGS const double* restrict pr, const double* restrict pi, long rs, __mmask8 lm, int masked, \
                  double* restrict or_, double* restrict oi_, long os, __mmask8 sm, \
                  const double* restrict cq, const double* restrict cqi

/* DFT3 helper macro (on complex vec pairs) */
#define KDFT3(x0r,x0i,x1r,x1i,x2r,x2i, y0r,y0i,y1r,y1i,y2r,y2i) do{ \
    vd tr = VADD(x1r,x2r), ti = VADD(x1i,x2i); \
    vd dr_ = VSUB(x1r,x2r), di_ = VSUB(x1i,x2i); \
    vd ur = VFMA(VSET1(-0.5), tr, x0r); \
    vd ui = VFMA(VSET1(-0.5), ti, x0i); \
    y0r = VADD(x0r, tr); y0i = VADD(x0i, ti); \
    vd s_ = VSET1(0.86602540378443864676372317075293618347); \
    y1r = VFMA(s_, di_, ur);  y1i = VFNMA(s_, dr_, ui); \
    y2r = VFNMA(s_, di_, ur); y2i = VFMA(s_, dr_, ui); \
}while(0)

/* DFT8 on named vars */
#define KDFT8(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i, \
              y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i) do{ \
    vd a0r = VADD(x0r,x4r), a0i = VADD(x0i,x4i); \
    vd a1r = VSUB(x0r,x4r), a1i = VSUB(x0i,x4i); \
    vd a2r = VADD(x2r,x6r), a2i = VADD(x2i,x6i); \
    vd a3r = VSUB(x2r,x6r), a3i = VSUB(x2i,x6i); \
    vd b0r = VADD(x1r,x5r), b0i = VADD(x1i,x5i); \
    vd b1r = VSUB(x1r,x5r), b1i = VSUB(x1i,x5i); \
    vd b2r = VADD(x3r,x7r), b2i = VADD(x3i,x7i); \
    vd b3r = VSUB(x3r,x7r), b3i = VSUB(x3i,x7i); \
    vd e0r = VADD(a0r,a2r), e0i = VADD(a0i,a2i); \
    vd e2r = VSUB(a0r,a2r), e2i = VSUB(a0i,a2i); \
    vd e1r = VADD(a1r,a3i), e1i = VSUB(a1i,a3r); \
    vd e3r = VSUB(a1r,a3i), e3i = VADD(a1i,a3r); \
    vd o0r = VADD(b0r,b2r), o0i = VADD(b0i,b2i); \
    vd o2r = VSUB(b0r,b2r), o2i = VSUB(b0i,b2i); \
    vd o1r = VADD(b1r,b3i), o1i = VSUB(b1i,b3r); \
    vd o3r = VSUB(b1r,b3i), o3i = VADD(b1i,b3r); \
    vd c8_ = VSET1(0.70710678118654752440084436210484903928); \
    vd t1r = VMUL(c8_, VADD(o1r,o1i)); \
    vd t1i = VMUL(c8_, VSUB(o1i,o1r)); \
    vd t3r = VMUL(c8_, VSUB(o3i,o3r)); \
    vd t3i = VMUL(VSET1(-0.70710678118654752440084436210484903928), VADD(o3r,o3i)); \
    y0r = VADD(e0r,o0r); y0i = VADD(e0i,o0i); \
    y4r = VSUB(e0r,o0r); y4i = VSUB(e0i,o0i); \
    y1r = VADD(e1r,t1r); y1i = VADD(e1i,t1i); \
    y5r = VSUB(e1r,t1r); y5i = VSUB(e1i,t1i); \
    y2r = VADD(e2r,o2i); y2i = VSUB(e2i,o2r); \
    y6r = VSUB(e2r,o2i); y6i = VADD(e2i,o2r); \
    y3r = VADD(e3r,t3r); y3i = VADD(e3i,t3i); \
    y7r = VSUB(e3r,t3r); y7i = VSUB(e3i,t3i); \
}while(0)

/* ---- L=6: PFA(3,2) ---- */
static AI void core6(CORE_ARGS){
    CORE_PROLOG
    vd x0r=LDR(0),x0i=LDI(0),x1r=LDR(1),x1i=LDI(1),x2r=LDR(2),x2i=LDI(2);
    vd x3r=LDR(3),x3i=LDI(3),x4r=LDR(4),x4i=LDI(4),x5r=LDR(5),x5i=LDI(5);
    vd g0r,g0i,g1r,g1i,g2r,g2i,h0r,h0i,h1r,h1i,h2r,h2i;
    KDFT3(x0r,x0i,x2r,x2i,x4r,x4i, g0r,g0i,g1r,g1i,g2r,g2i);
    KDFT3(x3r,x3i,x5r,x5i,x1r,x1i, h0r,h0i,h1r,h1i,h2r,h2i);
    OUT(0, VADD(g0r,h0r), VADD(g0i,h0i));
    OUT(3, VSUB(g0r,h0r), VSUB(g0i,h0i));
    OUT(4, VADD(g1r,h1r), VADD(g1i,h1i));
    OUT(1, VSUB(g1r,h1r), VSUB(g1i,h1i));
    OUT(2, VADD(g2r,h2r), VADD(g2i,h2i));
    OUT(5, VSUB(g2r,h2r), VSUB(g2i,h2i));
}

/* ---- L=8 ---- */
static AI void core8(CORE_ARGS){
    CORE_PROLOG
    vd x0r=LDR(0),x0i=LDI(0),x1r=LDR(1),x1i=LDI(1),x2r=LDR(2),x2i=LDI(2),x3r=LDR(3),x3i=LDI(3);
    vd x4r=LDR(4),x4i=LDI(4),x5r=LDR(5),x5i=LDI(5),x6r=LDR(6),x6i=LDI(6),x7r=LDR(7),x7i=LDI(7);
    vd y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i;
    KDFT8(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,
          y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i);
    OUT(0,y0r,y0i); OUT(1,y1r,y1i); OUT(2,y2r,y2i); OUT(3,y3r,y3i);
    OUT(4,y4r,y4i); OUT(5,y5r,y5i); OUT(6,y6r,y6i); OUT(7,y7r,y7i);
}

/* ---- primes via symmetric half-matrix, j-outer / k-inner sweeps ---- */
#define DEF_PRIME_CORE(L, H, CT, ST) \
static AI void core##L(CORE_ARGS){ \
    CORE_PROLOG \
    vd x0r = LDR(0), x0i = LDI(0); \
    vd AR[H+1], AIm[H+1], VRs[H], VIs[H]; \
    AR[0] = x0r; AIm[0] = x0i; \
    _Pragma("GCC unroll 16") \
    for (int k = 1; k <= H; k++){ AR[k] = x0r; AIm[k] = x0i; } \
    _Pragma("GCC unroll 16") \
    for (int j = 0; j < H; j++){ \
        vd ar = LDR(j+1), br = LDR(L-1-j); \
        vd ai = LDI(j+1), bi = LDI(L-1-j); \
        vd ur = VADD(ar, br), ui = VADD(ai, bi); \
        VRs[j] = VSUB(ar, br); VIs[j] = VSUB(ai, bi); \
        AR[0] = VADD(AR[0], ur); AIm[0] = VADD(AIm[0], ui); \
        _Pragma("GCC unroll 16") \
        for (int k = 1; k <= H; k++){ \
            vd cv = bcv(&CT[k-1][j]); \
            AR[k] = VFMA(cv, ur, AR[k]); AIm[k] = VFMA(cv, ui, AIm[k]); \
        } \
    } \
    OUT(0, AR[0], AIm[0]); \
    vd DR[H+1], DIm[H+1]; \
    _Pragma("GCC unroll 16") \
    for (int k = 1; k <= H; k++){ DR[k] = VSET1(0.0); DIm[k] = VSET1(0.0); } \
    _Pragma("GCC unroll 16") \
    for (int j = 0; j < H; j++){ \
        vd vr = VRs[j], vi = VIs[j]; \
        _Pragma("GCC unroll 16") \
        for (int k = 1; k <= H; k++){ \
            vd sv = bcv(&ST[k-1][j]); \
            DR[k] = VFMA(sv, vr, DR[k]); DIm[k] = VFMA(sv, vi, DIm[k]); \
        } \
    } \
    _Pragma("GCC unroll 16") \
    for (int k = 1; k <= H; k++){ \
        OUT(k,   VSUB(AR[k], DIm[k]), VADD(AIm[k], DR[k])); \
        OUT(L-k, VADD(AR[k], DIm[k]), VSUB(AIm[k], DR[k])); \
    } \
}

DEF_PRIME_CORE(13, 6, C13, S13)
DEF_PRIME_CORE(17, 8, C17, S17)
DEF_PRIME_CORE(23, 11, C23, S23)

/* DFT9 on vec arrays xr[9], xi[9] in place (natural->natural k) */
static AI void kdft9(vd *xr, vd *xi){
    vd a0r,a0i,a1r,a1i,a2r,a2i, b0r,b0i,b1r,b1i,b2r,b2i, c0r,c0i,c1r,c1i,c2r,c2i;
    KDFT3(xr[0],xi[0],xr[3],xi[3],xr[6],xi[6], a0r,a0i,a1r,a1i,a2r,a2i);
    KDFT3(xr[1],xi[1],xr[4],xi[4],xr[7],xi[7], b0r,b0i,b1r,b1i,b2r,b2i);
    KDFT3(xr[2],xi[2],xr[5],xi[5],xr[8],xi[8], c0r,c0i,c1r,c1i,c2r,c2i);
    {   vd cr = bcv(&W9r[1][1]), ci = bcv(&W9i[1][1]);
        vd nr = VFNMA(b1i, ci, VMUL(b1r, cr));
        vd ni = VFMA (b1r, ci, VMUL(b1i, cr)); b1r = nr; b1i = ni; }
    {   vd cr = bcv(&W9r[1][2]), ci = bcv(&W9i[1][2]);
        vd nr = VFNMA(b2i, ci, VMUL(b2r, cr));
        vd ni = VFMA (b2r, ci, VMUL(b2i, cr)); b2r = nr; b2i = ni; }
    {   vd cr = bcv(&W9r[2][1]), ci = bcv(&W9i[2][1]);
        vd nr = VFNMA(c1i, ci, VMUL(c1r, cr));
        vd ni = VFMA (c1r, ci, VMUL(c1i, cr)); c1r = nr; c1i = ni; }
    {   vd cr = bcv(&W9r[2][2]), ci = bcv(&W9i[2][2]);
        vd nr = VFNMA(c2i, ci, VMUL(c2r, cr));
        vd ni = VFMA (c2r, ci, VMUL(c2i, cr)); c2r = nr; c2i = ni; }
    KDFT3(a0r,a0i,b0r,b0i,c0r,c0i, xr[0],xi[0],xr[3],xi[3],xr[6],xi[6]);
    KDFT3(a1r,a1i,b1r,b1i,c1r,c1i, xr[1],xi[1],xr[4],xi[4],xr[7],xi[7]);
    KDFT3(a2r,a2i,b2r,b2i,c2r,c2i, xr[2],xi[2],xr[5],xi[5],xr[8],xi[8]);
}

/* ---- L=36: PFA(4,9) ---- */
static AI void core36(CORE_ARGS){
    CORE_PROLOG
    static const int16_t in36[9][4] = {
        {0,9,18,27},{4,13,22,31},{8,17,26,35},{12,21,30,3},{16,25,34,7},
        {20,29,2,11},{24,33,6,15},{28,1,10,19},{32,5,14,23}};
    static const int16_t out36[4][9] = {
        {0,28,20,12,4,32,24,16,8},{9,1,29,21,13,5,33,25,17},
        {18,10,2,30,22,14,6,34,26},{27,19,11,3,31,23,15,7,35}};
    vd HR[4][9], HI[4][9];
    _Pragma("GCC unroll 9")
    for (int n2 = 0; n2 < 9; n2++){
        vd x0r=LDR(in36[n2][0]), x0i=LDI(in36[n2][0]);
        vd x1r=LDR(in36[n2][1]), x1i=LDI(in36[n2][1]);
        vd x2r=LDR(in36[n2][2]), x2i=LDI(in36[n2][2]);
        vd x3r=LDR(in36[n2][3]), x3i=LDI(in36[n2][3]);
        vd t0r=VADD(x0r,x2r), t0i=VADD(x0i,x2i);
        vd t1r=VSUB(x0r,x2r), t1i=VSUB(x0i,x2i);
        vd t2r=VADD(x1r,x3r), t2i=VADD(x1i,x3i);
        vd t3r=VSUB(x1r,x3r), t3i=VSUB(x1i,x3i);
        HR[0][n2]=VADD(t0r,t2r); HI[0][n2]=VADD(t0i,t2i);
        HR[2][n2]=VSUB(t0r,t2r); HI[2][n2]=VSUB(t0i,t2i);
        HR[1][n2]=VADD(t1r,t3i); HI[1][n2]=VSUB(t1i,t3r);
        HR[3][n2]=VSUB(t1r,t3i); HI[3][n2]=VADD(t1i,t3r);
    }
    _Pragma("GCC unroll 4")
    for (int k1 = 0; k1 < 4; k1++){
        kdft9(HR[k1], HI[k1]);
        _Pragma("GCC unroll 9")
        for (int k2 = 0; k2 < 9; k2++)
            OUT(out36[k1][k2], HR[k1][k2], HI[k1][k2]);
    }
}

/* ---- L=45: PFA(9,5) ---- */
static AI void core45(CORE_ARGS){
    CORE_PROLOG
    static const int16_t in45[5][9] = {
        {0,5,10,15,20,25,30,35,40},
        {9,14,19,24,29,34,39,44,4},
        {18,23,28,33,38,43,3,8,13},
        {27,32,37,42,2,7,12,17,22},
        {36,41,1,6,11,16,21,26,31}};
    static const int16_t out45[9][5] = {
        {0,36,27,18,9},{10,1,37,28,19},{20,11,2,38,29},{30,21,12,3,39},
        {40,31,22,13,4},{5,41,32,23,14},{15,6,42,33,24},{25,16,7,43,34},{35,26,17,8,44}};
    vd HR[9][5], HI[9][5];  /* [k1][n2] */
    _Pragma("GCC unroll 5")
    for (int n2 = 0; n2 < 5; n2++){
        vd xr[9], xi[9];
        _Pragma("GCC unroll 9")
        for (int t = 0; t < 9; t++){ xr[t]=LDR(in45[n2][t]); xi[t]=LDI(in45[n2][t]); }
        kdft9(xr, xi);
        _Pragma("GCC unroll 9")
        for (int t = 0; t < 9; t++){ HR[t][n2]=xr[t]; HI[t][n2]=xi[t]; }
    }
    _Pragma("GCC unroll 9")
    for (int k1 = 0; k1 < 9; k1++){
        vd u0r = VADD(HR[k1][1],HR[k1][4]), u0i = VADD(HI[k1][1],HI[k1][4]);
        vd v0r = VSUB(HR[k1][1],HR[k1][4]), v0i = VSUB(HI[k1][1],HI[k1][4]);
        vd u1r = VADD(HR[k1][2],HR[k1][3]), u1i = VADD(HI[k1][2],HI[k1][3]);
        vd v1r = VSUB(HR[k1][2],HR[k1][3]), v1i = VSUB(HI[k1][2],HI[k1][3]);
        vd x0r = HR[k1][0], x0i = HI[k1][0];
        OUT(out45[k1][0], VADD(VADD(u0r,u1r), x0r), VADD(VADD(u0i,u1i), x0i));
        _Pragma("GCC unroll 2")
        for (int k = 1; k <= 2; k++){
            vd c0 = bcv(&C5[k-1][0]), c1 = bcv(&C5[k-1][1]);
            vd s0 = bcv(&S5[k-1][0]), s1 = bcv(&S5[k-1][1]);
            vd ar = VFMA(c1, u1r, VFMA(c0, u0r, x0r));
            vd ai = VFMA(c1, u1i, VFMA(c0, u0i, x0i));
            vd dr = VFMA(s1, v1r, VMUL(s0, v0r));
            vd di = VFMA(s1, v1i, VMUL(s0, v0i));
            OUT(out45[k1][k],   VSUB(ar, di), VADD(ai, dr));
            OUT(out45[k1][5-k], VADD(ar, di), VSUB(ai, dr));
        }
    }
}

/* ---- L=64: CT(8,8) ---- */
static AI void core64(CORE_ARGS){
    CORE_PROLOG
    vd GR[8][8], GI[8][8]; /* [k1][b] */
    for (int b = 0; b < 8; b++){
        vd y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i;
        KDFT8(LDR(b),LDI(b),LDR(8+b),LDI(8+b),LDR(16+b),LDI(16+b),LDR(24+b),LDI(24+b),
              LDR(32+b),LDI(32+b),LDR(40+b),LDI(40+b),LDR(48+b),LDI(48+b),LDR(56+b),LDI(56+b),
              y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i);
        GR[0][b]=y0r; GI[0][b]=y0i;
        vd yr[8] = {y0r,y1r,y2r,y3r,y4r,y5r,y6r,y7r};
        vd yi[8] = {y0i,y1i,y2i,y3i,y4i,y5i,y6i,y7i};
        _Pragma("GCC unroll 8")
        for (int k1 = 1; k1 < 8; k1++){
            vd cr = bcv(&W64r[b][k1]), ci = bcv(&W64i[b][k1]);
            GR[k1][b] = VFNMA(yi[k1], ci, VMUL(yr[k1], cr));
            GI[k1][b] = VFMA (yr[k1], ci, VMUL(yi[k1], cr));
        }
    }
    for (int k1 = 0; k1 < 8; k1++){
        vd y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i;
        KDFT8(GR[k1][0],GI[k1][0],GR[k1][1],GI[k1][1],GR[k1][2],GI[k1][2],GR[k1][3],GI[k1][3],
              GR[k1][4],GI[k1][4],GR[k1][5],GI[k1][5],GR[k1][6],GI[k1][6],GR[k1][7],GI[k1][7],
              y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i);
        OUT(k1,    y0r,y0i); OUT(k1+8,  y1r,y1i);
        OUT(k1+16, y2r,y2i); OUT(k1+24, y3r,y3i);
        OUT(k1+32, y4r,y4i); OUT(k1+40, y5r,y5i);
        OUT(k1+48, y6r,y6i); OUT(k1+56, y7r,y7i);
    }
}

#undef LDR
#undef LDI
#undef OUT

/* ------------------------------------------------ passes ---------- */
#define DEF_PASSES(L) \
static void passA_##L(const double* restrict ir, const double* restrict ii, \
                      double* restrict orr, double* restrict oi){ \
    const long NN = (long)(L)*(L); const long SLAB = NN + 8; \
    long c0 = 0; \
    for (; c0 + 8 <= NN; c0 += 8) \
        core##L(ir + c0, ii + c0, SLAB, 0xFF, 0, orr + c0, oi + c0, SLAB, 0xFF, 0, 0); \
    if (c0 < NN){ \
        __mmask8 lm = (__mmask8)((1u<<(NN-c0))-1u); \
        core##L(ir + c0, ii + c0, SLAB, lm, 1, orr + c0, oi + c0, SLAB, lm, 0, 0); \
    } \
} \
static void passB_##L(const double* restrict ir, const double* restrict ii, \
                      double* restrict orr, double* restrict oi){ \
    const long SLAB = (long)(L)*(L) + 8; \
    for (long x = 0; x < L; x++){ \
        const double *br = ir + x*SLAB, *bi = ii + x*SLAB; \
        double *cr = orr + x*SLAB, *ci = oi + x*SLAB; \
        long z0 = 0; \
        for (; z0 + 8 <= L; z0 += 8) \
            core##L(br + z0, bi + z0, L, 0xFF, 0, cr + z0, ci + z0, L, 0xFF, 0, 0); \
        if (z0 < L){ \
            __mmask8 lm = (__mmask8)((1u<<(L-z0))-1u); \
            core##L(br + z0, bi + z0, L, lm, 1, cr + z0, ci + z0, L, lm, 0, 0); \
        } \
    } \
} \
static void passC_##L(const double* restrict ir, const double* restrict ii, \
                      double* restrict orr, double* restrict oi, \
                      const double* restrict cre, const double* restrict cim){ \
    const long SLAB = (long)(L)*(L) + 8; \
    const __mmask8 kmtail = (__mmask8)((L % 8) ? ((1u<<(L%8))-1u) : 0xFFu); \
    vd KR[(((L)+7)&~7)], KI[(((L)+7)&~7)]; \
    for (long x = 0; x < L; x++){ \
        const double *br = ir + x*SLAB, *bi = ii + x*SLAB; \
        double *dr = orr + x*SLAB, *di = oi + x*SLAB; \
        const double *qr = cre + x*SLAB, *qi = cim + x*SLAB; \
        for (long y0 = 0; y0 < L; y0 += 8){ \
            int ny = (int)((L - y0) < 8 ? (L - y0) : 8); \
            __mmask8 my = (__mmask8)(ny == 8 ? 0xFF : ((1u<<ny)-1u)); \
            for (int jb = 0; jb < L; jb += 8){ \
                __mmask8 km = (jb + 8 <= L) ? (__mmask8)0xFF : kmtail; \
                vd RR[8], SS[8]; \
                for (int t = 0; t < 8; t++){ \
                    if (t < ny){ \
                        RR[t] = VLOADZ(km, br + (y0+t)*(long)L + jb); \
                        SS[t] = VLOADZ(km, bi + (y0+t)*(long)L + jb); \
                    } else { RR[t] = _mm512_setzero_pd(); SS[t] = _mm512_setzero_pd(); } \
                } \
                transp8_from(RR, KR + jb); transp8_from(SS, KI + jb); \
            } \
            core##L((const double*)KR, (const double*)KI, 8, 0xFF, 0, \
                    dr + y0, di + y0, L, my, qr + y0, qi + y0); \
        } \
    } \
}

DEF_PASSES(6)  DEF_PASSES(8)  DEF_PASSES(13) DEF_PASSES(17)
DEF_PASSES(23) DEF_PASSES(36) DEF_PASSES(45) DEF_PASSES(64)

/* ------------------------------------------- conversions ----------- */
static void split_in(long L, const double *x, double *sre, double *sim){
    const long NN = L*L, SLAB = NN + 8;
    const __m512i idxA = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    const __m512i idxB = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    for (long u = 0; u < L; u++){
        const double *src = x + u*NN*2;
        double *dr = sre + u*SLAB, *di = sim + u*SLAB;
        long w = 0;
        for (; w + 8 <= NN; w += 8){
            vd q0 = _mm512_loadu_pd(src + 2*w);
            vd q1 = _mm512_loadu_pd(src + 2*w + 8);
            _mm512_storeu_pd(dr + w, _mm512_permutex2var_pd(q0, idxA, q1));
            _mm512_storeu_pd(di + w, _mm512_permutex2var_pd(q0, idxB, q1));
        }
        for (; w < NN; w++){ dr[w] = src[2*w]; di[w] = src[2*w+1]; }
    }
}

static void flip_copy(long L, const double *src, double *dst){
    const long SLAB = L*L + 8;
    for (long x = 0; x < L; x++){
        const double *s = src + x*SLAB; double *d = dst + x*SLAB;
        for (long y = 0; y < L; y++)
            for (long z = 0; z < L; z++)
                d[z*L + y] = s[y*L + z];
    }
}

static void snap_nat(long L, const double *sre, const double *sim, double *out){
    const long NN = L*L, SLAB = NN + 8;
    const __m512i idxI0 = _mm512_set_epi64(11,3,10,2,9,1,8,0);
    const __m512i idxI1 = _mm512_set_epi64(15,7,14,6,13,5,12,4);
    for (long u = 0; u < L; u++){
        const double *dr = sre + u*SLAB, *di = sim + u*SLAB;
        double *dst = out + u*NN*2;
        long w = 0;
        for (; w + 8 <= NN; w += 8){
            vd a = _mm512_loadu_pd(dr + w);
            vd b = _mm512_loadu_pd(di + w);
            _mm512_storeu_pd(dst + 2*w,     _mm512_permutex2var_pd(a, idxI0, b));
            _mm512_storeu_pd(dst + 2*w + 8, _mm512_permutex2var_pd(a, idxI1, b));
        }
        for (; w < NN; w++){ dst[2*w] = dr[w]; dst[2*w+1] = di[w]; }
    }
}

static void snap_flip(long L, const double *sre, const double *sim, double *out){
    const long SLAB = L*L + 8;
    for (long x = 0; x < L; x++){
        const double *sr = sre + x*SLAB, *si = sim + x*SLAB;
        double *dst = out + x*L*L*2;
        for (long y = 0; y < L; y++)
            for (long z = 0; z < L; z++){
                dst[(y*L+z)*2]   = sr[z*L + y];
                dst[(y*L+z)*2+1] = si[z*L + y];
            }
    }
}

/* ------------------------------------------------ drivers ---------- */
#define DEF_RUN(L) \
void run_##L(long B, long m, const double *x0, const double *cc, double *out1, double *outm){ \
    bufs_t *bf = &BUFS[L]; \
    const long n = (long)L*L*L; \
    for (long b = 0; b < B; b++){ \
        split_in(L, x0 + b*2*n, bf->sre, bf->sim); \
        split_in(L, cc + b*2*n, bf->cre, bf->cim); \
        flip_copy(L, bf->cre, bf->cfre); flip_copy(L, bf->cim, bf->cfim); \
        int flip = 0; \
        for (long it = 0; it < m; it++){ \
            passA_##L(bf->sre, bf->sim, bf->t1re, bf->t1im); \
            passB_##L(bf->t1re, bf->t1im, bf->t2re, bf->t2im); \
            if (!flip) passC_##L(bf->t2re, bf->t2im, bf->sre, bf->sim, bf->cfre, bf->cfim); \
            else       passC_##L(bf->t2re, bf->t2im, bf->sre, bf->sim, bf->cre,  bf->cim); \
            flip ^= 1; \
            if (it == 0){ \
                if (flip) snap_flip(L, bf->sre, bf->sim, out1 + b*2*n); \
                else      snap_nat (L, bf->sre, bf->sim, out1 + b*2*n); \
            } \
        } \
        if (flip) snap_flip(L, bf->sre, bf->sim, outm + b*2*n); \
        else      snap_nat (L, bf->sre, bf->sim, outm + b*2*n); \
    } \
}

DEF_RUN(6)  DEF_RUN(8)  DEF_RUN(13) DEF_RUN(17)
DEF_RUN(23) DEF_RUN(36) DEF_RUN(45) DEF_RUN(64)

/* ---- dev microbenchmarks ---- */
#include <x86intrin.h>
#define BENCH_SWITCH(CALL) \
    switch(L){ \
        case 6: for(long i=0;i<iters;i++){ CALL(6); } break; \
        case 8: for(long i=0;i<iters;i++){ CALL(8); } break; \
        case 13: for(long i=0;i<iters;i++){ CALL(13); } break; \
        case 17: for(long i=0;i<iters;i++){ CALL(17); } break; \
        case 23: for(long i=0;i<iters;i++){ CALL(23); } break; \
        case 36: for(long i=0;i<iters;i++){ CALL(36); } break; \
        case 45: for(long i=0;i<iters;i++){ CALL(45); } break; \
        case 64: for(long i=0;i<iters;i++){ CALL(64); } break; }

double bench_passA(long L, long iters){
    bufs_t *bf = &BUFS[L];
    unsigned long long t0 = __rdtsc();
#define CA(N) passA_##N(bf->sre,bf->sim,bf->t1re,bf->t1im)
    BENCH_SWITCH(CA)
    unsigned long long t1 = __rdtsc();
    return (double)(t1-t0)/iters/(double)(L*L*L);
}
double bench_passB(long L, long iters){
    bufs_t *bf = &BUFS[L];
    unsigned long long t0 = __rdtsc();
#define CB(N) passB_##N(bf->sre,bf->sim,bf->t1re,bf->t1im)
    BENCH_SWITCH(CB)
    unsigned long long t1 = __rdtsc();
    return (double)(t1-t0)/iters/(double)(L*L*L);
}
double bench_passC(long L, long iters){
    bufs_t *bf = &BUFS[L];
    unsigned long long t0 = __rdtsc();
#define CC(N) passC_##N(bf->sre,bf->sim,bf->t1re,bf->t1im,bf->cre,bf->cim)
    BENCH_SWITCH(CC)
    unsigned long long t1 = __rdtsc();
    return (double)(t1-t0)/iters/(double)(L*L*L);
}
void fill_state(long L, const double* x, const double* c){
    split_in(L, x, BUFS[L].sre, BUFS[L].sim);
    split_in(L, c, BUFS[L].cre, BUFS[L].cim);
    split_in(L, x, BUFS[L].t1re, BUFS[L].t1im);
    split_in(L, x, BUFS[L].t2re, BUFS[L].t2im);
    flip_copy(L, BUFS[L].cre, BUFS[L].cfre);
    flip_copy(L, BUFS[L].cim, BUFS[L].cfim);
}
double freq_probe(long iters){
    vd a = VSET1(1.0), b = VSET1(1e-30);
    unsigned long long t0 = __rdtsc();
    for (long i = 0; i < iters; i++){ a = VADD(a,b); __asm__ volatile("" : "+v"(a)); a = VADD(a,b); __asm__ volatile("" : "+v"(a)); a = VADD(a,b); __asm__ volatile("" : "+v"(a)); a = VADD(a,b); __asm__ volatile("" : "+v"(a)); }
    unsigned long long t1 = __rdtsc();
    volatile double sink = ((double*)&a)[0]; (void)sink;
    return (double)(t1-t0)/(double)(iters*4);
}
