// implementation.c -- specialized batched 3D complex FFT + nonlinear map iteration
// Sizes: 6, 8, 13, 17, 23, 36, 45, 64. Single-threaded, AVX-512.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

typedef __m512d VD;
typedef __mmask8 MK;

#define VADD  _mm512_add_pd
#define VSUB  _mm512_sub_pd
#define VMUL  _mm512_mul_pd
#define VFMA  _mm512_fmadd_pd    /* a*b+c */
#define VFNMA _mm512_fnmadd_pd   /* c-a*b */
#define VFMS  _mm512_fmsub_pd    /* a*b-c */
#define BC(x) _mm512_set1_pd(x)
#define LD(p)    _mm512_loadu_pd(p)
#define ST(p,v)  _mm512_storeu_pd((p),(v))
#define MLD(m,p) _mm512_maskz_loadu_pd((m),(p))
#define MST(p,m,v) _mm512_mask_storeu_pd((p),(m),(v))
#define VMAX  _mm512_max_pd
#define ZERO  _mm512_setzero_pd()

// ---------------------------------------------------------------------------
// Tables (filled by init_tables)
// ---------------------------------------------------------------------------
static double g_s3half;        // sqrt(3)/2
static double g_r2half;        // sqrt(2)/2
static double g_c51, g_c52, g_s51, g_s52;   // DFT5: cos/sin 2pi/5, 4pi/5
static double g_w9[8];         // W9^1, W9^2, W9^4 (re,im pairs): [w1r,w1i,w2r,w2i,w4r,w4i]

// folded tables for odd primes: C[k][j] = cos(2pi k j / L), S[k][j] = sin(2pi k j/L), 1<=k,j<=h
static double g_C13[6*6],  g_S13[6*6];
static double g_C17[8*8],  g_S17[8*8];
static double g_C23[11*11], g_S23[11*11];
// z-axis (from-right) vector tables: for j=1..h: lanes k=1..8 (chunk0), k=9.. (chunk1)
// Czv[j][lane] = cos(2pi (k0+lane) j / L)
static double g_Cz13[6][8]   __attribute__((aligned(64))), g_Sz13[6][8]   __attribute__((aligned(64)));
static double g_Cz17[8][8]   __attribute__((aligned(64))), g_Sz17[8][8]   __attribute__((aligned(64)));
static double g_Cz23[2][11][8] __attribute__((aligned(64))), g_Sz23[2][11][8] __attribute__((aligned(64)));
// from-right full W tables for 6 and 8: W[j][k] = exp(-2pi i jk/L), lanes k
static double g_W6r[6][8] __attribute__((aligned(64))), g_W6i[6][8] __attribute__((aligned(64)));
static double g_W8r[8][8] __attribute__((aligned(64))), g_W8i[8][8] __attribute__((aligned(64)));
// folded-even z tables for 6 and 8: Cz[j][k]=cos(2pi jk/L), Sz, ALT[k]=(-1)^k
static double g_Cz6[2][8] __attribute__((aligned(64))), g_Sz6[2][8] __attribute__((aligned(64))), g_ALT6[8] __attribute__((aligned(64)));
static double g_Cz8[3][8] __attribute__((aligned(64))), g_Sz8[3][8] __attribute__((aligned(64))), g_ALT8[8] __attribute__((aligned(64)));
// 64: middle twiddles T[b][t] = W64^{b t} scalars; z-variant vector twiddles per kL over lanes zL
static double g_T64r[8][8], g_T64i[8][8];
static double g_T64zr[8][8] __attribute__((aligned(64))), g_T64zi[8][8] __attribute__((aligned(64)));
// PFA perms
static int g_p36in[4][9];   // input index for (j4, j9)
static int g_p36out[4][9];  // output index for (k4, k9)
static int g_p45in[5][9];   // (j5, j9)
static int g_p45out[5][9];  // (k5, k9)

static int g_inited = 0;

void init_tables(void)
{
    if (g_inited) return;
    g_inited = 1;
    const long double PI = 3.14159265358979323846264338327950288L;
    g_s3half = (double)sqrtl(3.0L)/2.0;
    g_r2half = (double)(sqrtl(2.0L)/2.0L);
    g_c51 = (double)cosl(2.0L*PI/5.0L); g_s51 = (double)sinl(2.0L*PI/5.0L);
    g_c52 = (double)cosl(4.0L*PI/5.0L); g_s52 = (double)sinl(4.0L*PI/5.0L);
    // W9^t = exp(-2pi i t/9)
    g_w9[0] = (double)cosl(2.0L*PI/9.0L);  g_w9[1] = (double)(-sinl(2.0L*PI/9.0L));
    g_w9[2] = (double)cosl(4.0L*PI/9.0L);  g_w9[3] = (double)(-sinl(4.0L*PI/9.0L));
    g_w9[4] = (double)cosl(8.0L*PI/9.0L);  g_w9[5] = (double)(-sinl(8.0L*PI/9.0L));

    // folded prime tables
    struct { int L, h; double *C, *S; } P[3] = {
        {13, 6, g_C13, g_S13}, {17, 8, g_C17, g_S17}, {23, 11, g_C23, g_S23}
    };
    for (int t = 0; t < 3; t++) {
        int L = P[t].L, h = P[t].h;
        for (int k = 1; k <= h; k++)
            for (int j = 1; j <= h; j++) {
                long r = ((long)k*j) % L;
                long double a = 2.0L*PI*(long double)r/(long double)L;
                P[t].C[(k-1)*h + (j-1)] = (double)cosl(a);
                P[t].S[(k-1)*h + (j-1)] = (double)sinl(a);
            }
    }
    // z-variant vector tables (lanes = k)
    for (int j = 1; j <= 6; j++) for (int l = 0; l < 8; l++) {
        int k = 1 + l; if (k > 6) { g_Cz13[j-1][l]=0; g_Sz13[j-1][l]=0; continue; }
        long r = ((long)k*j) % 13; long double a = 2.0L*PI*r/13.0L;
        g_Cz13[j-1][l] = (double)cosl(a); g_Sz13[j-1][l] = (double)sinl(a);
    }
    for (int j = 1; j <= 8; j++) for (int l = 0; l < 8; l++) {
        int k = 1 + l;
        long r = ((long)k*j) % 17; long double a = 2.0L*PI*r/17.0L;
        g_Cz17[j-1][l] = (double)cosl(a); g_Sz17[j-1][l] = (double)sinl(a);
    }
    for (int c = 0; c < 2; c++) for (int j = 1; j <= 11; j++) for (int l = 0; l < 8; l++) {
        int k = 1 + 8*c + l; if (k > 11) { g_Cz23[c][j-1][l]=0; g_Sz23[c][j-1][l]=0; continue; }
        long r = ((long)k*j) % 23; long double a = 2.0L*PI*r/23.0L;
        g_Cz23[c][j-1][l] = (double)cosl(a); g_Sz23[c][j-1][l] = (double)sinl(a);
    }
    // full W tables 6/8 (forward: exp(-i a))
    for (int j = 0; j < 6; j++) for (int l = 0; l < 8; l++) {
        int k = l % 6;  /* lanes 6,7 unused; fill harmlessly */
        long r = ((long)j*k) % 6; long double a = 2.0L*PI*r/6.0L;
        g_W6r[j][l] = (double)cosl(a); g_W6i[j][l] = (double)(-sinl(a));
    }
    for (int j = 0; j < 8; j++) for (int l = 0; l < 8; l++) {
        long r = ((long)j*l) % 8; long double a = 2.0L*PI*r/8.0L;
        g_W8r[j][l] = (double)cosl(a); g_W8i[j][l] = (double)(-sinl(a));
    }
    for (int j = 1; j <= 2; j++) for (int l = 0; l < 8; l++) {
        int k = l % 6;
        long r = ((long)j*k) % 6; long double a = 2.0L*PI*r/6.0L;
        g_Cz6[j-1][l] = (double)cosl(a); g_Sz6[j-1][l] = (double)sinl(a);
    }
    for (int l = 0; l < 8; l++) g_ALT6[l] = ((l%6)&1) ? -1.0 : 1.0;
    for (int j = 1; j <= 3; j++) for (int l = 0; l < 8; l++) {
        long r = ((long)j*l) % 8; long double a = 2.0L*PI*r/8.0L;
        g_Cz8[j-1][l] = (double)cosl(a); g_Sz8[j-1][l] = (double)sinl(a);
    }
    for (int l = 0; l < 8; l++) g_ALT8[l] = (l&1) ? -1.0 : 1.0;
    // 64 twiddles
    for (int b = 0; b < 8; b++) for (int t = 0; t < 8; t++) {
        long r = ((long)b*t) % 64; long double a = 2.0L*PI*r/64.0L;
        g_T64r[b][t] = (double)cosl(a); g_T64i[b][t] = (double)(-sinl(a));
        g_T64zr[t][b] = g_T64r[b][t];  g_T64zi[t][b] = g_T64i[b][t]; // [kL][zL]
    }
    // PFA perms
    for (int j4 = 0; j4 < 4; j4++) for (int j9 = 0; j9 < 9; j9++)
        g_p36in[j4][j9] = (9*j4 + 4*j9) % 36;
    for (int k4 = 0; k4 < 4; k4++) for (int k9 = 0; k9 < 9; k9++)
        g_p36out[k4][k9] = (9*k4 + 28*k9) % 36;
    for (int j5 = 0; j5 < 5; j5++) for (int j9 = 0; j9 < 9; j9++)
        g_p45in[j5][j9] = (5*j9 + 9*j5) % 45;
    for (int k5 = 0; k5 < 5; k5++) for (int k9 = 0; k9 < 9; k9++)
        g_p45out[k5][k9] = (36*k5 + 10*k9) % 45;
}
// ---------------------------------------------------------------------------
// Small DFT macros on (re,im) vector pairs. U = unique suffix for temps.
// ---------------------------------------------------------------------------
#define DFT2V(o0r,o0i,o1r,o1i, a0r,a0i,a1r,a1i) do { \
    VD _t0r=VADD(a0r,a1r), _t0i=VADD(a0i,a1i);       \
    VD _t1r=VSUB(a0r,a1r), _t1i=VSUB(a0i,a1i);       \
    o0r=_t0r; o0i=_t0i; o1r=_t1r; o1i=_t1i; } while(0)

// forward DFT3: needs Vhalf=0.5, Vs3=sqrt(3)/2 in scope
#define DFT3V(o0r,o0i,o1r,o1i,o2r,o2i, x0r,x0i,x1r,x1i,x2r,x2i) do { \
    VD _tr=VADD(x1r,x2r), _ti=VADD(x1i,x2i);   \
    VD _dr=VSUB(x1r,x2r), _di=VSUB(x1i,x2i);   \
    VD _mr=VFNMA(Vhalf,_tr,x0r), _mi=VFNMA(Vhalf,_ti,x0i); \
    VD _sr=VMUL(Vs3,_dr), _si=VMUL(Vs3,_di);   \
    o0r=VADD(x0r,_tr); o0i=VADD(x0i,_ti);      \
    o1r=VADD(_mr,_si); o1i=VSUB(_mi,_sr);      \
    o2r=VSUB(_mr,_si); o2i=VADD(_mi,_sr); } while(0)

// forward DFT4
#define DFT4V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i, x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i) do { \
    VD _t0r=VADD(x0r,x2r), _t0i=VADD(x0i,x2i);  \
    VD _t1r=VSUB(x0r,x2r), _t1i=VSUB(x0i,x2i);  \
    VD _t2r=VADD(x1r,x3r), _t2i=VADD(x1i,x3i);  \
    VD _t3r=VSUB(x1r,x3r), _t3i=VSUB(x1i,x3i);  \
    o0r=VADD(_t0r,_t2r); o0i=VADD(_t0i,_t2i);   \
    o2r=VSUB(_t0r,_t2r); o2i=VSUB(_t0i,_t2i);   \
    o1r=VADD(_t1r,_t3i); o1i=VSUB(_t1i,_t3r);   \
    o3r=VSUB(_t1r,_t3i); o3i=VADD(_t1i,_t3r); } while(0)

// forward DFT5 (folded): needs Vc51,Vc52,Vs51,Vs52
#define DFT5V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i, x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i) do { \
    VD _u1r=VADD(x1r,x4r), _u1i=VADD(x1i,x4i);  \
    VD _v1r=VSUB(x1r,x4r), _v1i=VSUB(x1i,x4i);  \
    VD _u2r=VADD(x2r,x3r), _u2i=VADD(x2i,x3i);  \
    VD _v2r=VSUB(x2r,x3r), _v2i=VSUB(x2i,x3i);  \
    o0r=VADD(x0r,VADD(_u1r,_u2r)); o0i=VADD(x0i,VADD(_u1i,_u2i)); \
    VD _A1r=VFMA(Vc51,_u1r,VFMA(Vc52,_u2r,x0r)); \
    VD _A1i=VFMA(Vc51,_u1i,VFMA(Vc52,_u2i,x0i)); \
    VD _B1r=VFMA(Vs51,_v1r,VMUL(Vs52,_v2r));     \
    VD _B1i=VFMA(Vs51,_v1i,VMUL(Vs52,_v2i));     \
    VD _A2r=VFMA(Vc52,_u1r,VFMA(Vc51,_u2r,x0r)); \
    VD _A2i=VFMA(Vc52,_u1i,VFMA(Vc51,_u2i,x0i)); \
    VD _B2r=VFMS(Vs52,_v1r,VMUL(Vs51,_v2r));     \
    VD _B2i=VFMS(Vs52,_v1i,VMUL(Vs51,_v2i));     \
    o1r=VADD(_A1r,_B1i); o1i=VSUB(_A1i,_B1r);    \
    o4r=VSUB(_A1r,_B1i); o4i=VADD(_A1i,_B1r);    \
    o2r=VADD(_A2r,_B2i); o2i=VSUB(_A2i,_B2r);    \
    o3r=VSUB(_A2r,_B2i); o3i=VADD(_A2i,_B2r); } while(0)

// forward DFT8: needs Vr2 = sqrt(2)/2
#define DFT8V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i, \
              x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i) do { \
    VD _E0r,_E0i,_E1r,_E1i,_E2r,_E2i,_E3r,_E3i; \
    VD _O0r,_O0i,_O1r,_O1i,_O2r,_O2i,_O3r,_O3i; \
    DFT4V(_E0r,_E0i,_E1r,_E1i,_E2r,_E2i,_E3r,_E3i, x0r,x0i,x2r,x2i,x4r,x4i,x6r,x6i); \
    DFT4V(_O0r,_O0i,_O1r,_O1i,_O2r,_O2i,_O3r,_O3i, x1r,x1i,x3r,x3i,x5r,x5i,x7r,x7i); \
    /* T1 = w8 * O1, w8=(s,-s) : re=s*(or+oi), im=s*(oi-or) */ \
    VD _T1r=VMUL(Vr2,VADD(_O1r,_O1i)), _T1i=VMUL(Vr2,VSUB(_O1i,_O1r)); \
    /* T2 = -i*O2: (oi, -or) */ \
    VD _T2r=_O2i, _T2i=_mm512_sub_pd(ZERO,_O2r); \
    /* T3 = w8^3 * O3 = (-s,-s): re=-s*(or-oi)=s*(oi-or), im=-s*(or+oi) */ \
    VD _T3r=VMUL(Vr2,VSUB(_O3i,_O3r)), _T3i=VMUL(Vr2,VSUB(_mm512_sub_pd(ZERO,_O3r),_O3i)); \
    o0r=VADD(_E0r,_O0r); o0i=VADD(_E0i,_O0i);  \
    o4r=VSUB(_E0r,_O0r); o4i=VSUB(_E0i,_O0i);  \
    o1r=VADD(_E1r,_T1r); o1i=VADD(_E1i,_T1i);  \
    o5r=VSUB(_E1r,_T1r); o5i=VSUB(_E1i,_T1i);  \
    o2r=VADD(_E2r,_T2r); o2i=VADD(_E2i,_T2i);  \
    o6r=VSUB(_E2r,_T2r); o6i=VSUB(_E2i,_T2i);  \
    o3r=VADD(_E3r,_T3r); o3i=VADD(_E3i,_T3i);  \
    o7r=VSUB(_E3r,_T3r); o7i=VSUB(_E3i,_T3i); } while(0)

// complex multiply: (or,oi) = (ar,ai)*(br,bi)
#define CMULV(or_,oi_, ar,ai, br,bi) do { \
    VD _pr = VMUL(ar,br), _pi = VMUL(ar,bi); \
    or_ = VFNMA(ai,bi,_pr); oi_ = VFMA(ai,br,_pi); } while(0)

// forward DFT9 via 3x3 CT. Needs Vhalf,Vs3 and Vw91r.. in scope.
// x is 18 VD args x0r..x8i ; o is o0r..o8i
#define DFT9V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i, \
              x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,x8r,x8i) do { \
    VD _A00r,_A00i,_A01r,_A01i,_A02r,_A02i;  \
    VD _A10r,_A10i,_A11r,_A11i,_A12r,_A12i;  \
    VD _A20r,_A20i,_A21r,_A21i,_A22r,_A22i;  \
    DFT3V(_A00r,_A00i,_A01r,_A01i,_A02r,_A02i, x0r,x0i,x3r,x3i,x6r,x6i); \
    DFT3V(_A10r,_A10i,_A11r,_A11i,_A12r,_A12i, x1r,x1i,x4r,x4i,x7r,x7i); \
    DFT3V(_A20r,_A20i,_A21r,_A21i,_A22r,_A22i, x2r,x2i,x5r,x5i,x8r,x8i); \
    VD _B11r,_B11i,_B12r,_B12i,_B21r,_B21i,_B22r,_B22i; \
    CMULV(_B11r,_B11i,_A11r,_A11i,Vw91r,Vw91i);  \
    CMULV(_B12r,_B12i,_A12r,_A12i,Vw92r,Vw92i);  \
    CMULV(_B21r,_B21i,_A21r,_A21i,Vw92r,Vw92i);  \
    CMULV(_B22r,_B22i,_A22r,_A22i,Vw94r,Vw94i);  \
    /* out[k3 + 3q] = DFT3 over j2 of Btilde[.][k3] */ \
    DFT3V(o0r,o0i,o3r,o3i,o6r,o6i, _A00r,_A00i,_A10r,_A10i,_A20r,_A20i); \
    DFT3V(o1r,o1i,o4r,o4i,o7r,o7i, _A01r,_A01i,_B11r,_B11i,_B21r,_B21i); \
    DFT3V(o2r,o2i,o5r,o5i,o8r,o8i, _A02r,_A02i,_B12r,_B12i,_B22r,_B22i); } while(0)
// ---------------------------------------------------------------------------
// 8x8 transpose of doubles (r0..r7 -> o0..o7)
// ---------------------------------------------------------------------------
#define TR8(o0,o1,o2,o3,o4,o5,o6,o7, r0,r1,r2,r3,r4,r5,r6,r7) do { \
    VD _t0=_mm512_unpacklo_pd(r0,r1), _t1=_mm512_unpackhi_pd(r0,r1); \
    VD _t2=_mm512_unpacklo_pd(r2,r3), _t3=_mm512_unpackhi_pd(r2,r3); \
    VD _t4=_mm512_unpacklo_pd(r4,r5), _t5=_mm512_unpackhi_pd(r4,r5); \
    VD _t6=_mm512_unpacklo_pd(r6,r7), _t7=_mm512_unpackhi_pd(r6,r7); \
    VD _u0=_mm512_shuffle_f64x2(_t0,_t2,0x88), _u1=_mm512_shuffle_f64x2(_t1,_t3,0x88); \
    VD _u2=_mm512_shuffle_f64x2(_t0,_t2,0xdd), _u3=_mm512_shuffle_f64x2(_t1,_t3,0xdd); \
    VD _u4=_mm512_shuffle_f64x2(_t4,_t6,0x88), _u5=_mm512_shuffle_f64x2(_t5,_t7,0x88); \
    VD _u6=_mm512_shuffle_f64x2(_t4,_t6,0xdd), _u7=_mm512_shuffle_f64x2(_t5,_t7,0xdd); \
    o0=_mm512_shuffle_f64x2(_u0,_u4,0x88); o4=_mm512_shuffle_f64x2(_u0,_u4,0xdd); \
    o1=_mm512_shuffle_f64x2(_u1,_u5,0x88); o5=_mm512_shuffle_f64x2(_u1,_u5,0xdd); \
    o2=_mm512_shuffle_f64x2(_u2,_u6,0x88); o6=_mm512_shuffle_f64x2(_u2,_u6,0xdd); \
    o3=_mm512_shuffle_f64x2(_u3,_u7,0x88); o7=_mm512_shuffle_f64x2(_u3,_u7,0xdd); } while(0)

// ---------------------------------------------------------------------------
// Elementwise map: given y (=DFT result) vectors and c pointers,
// z = y + c ; out = z / (1 + |z|), stored masked at (pr,pi).
// Needs in scope: Vhalf(0.5), V1p5(1.5), Vone(1.0), Vtiny(2.3e-308)
// ---------------------------------------------------------------------------
#define MAPST(pr, pi, mm, yr, yi, pcr, pci) do { \
    VD _zr = VADD(yr, LD(pcr)), _zi = VADD(yi, LD(pci)); \
    VD _s  = VFMA(_zi,_zi, VMUL(_zr,_zr)); \
    _s = VMAX(_s, Vtiny); \
    VD _t = _mm512_rsqrt14_pd(_s); \
    VD _hs = VMUL(Vhalf,_s); \
    _t = VMUL(_t, VFNMA(_hs, VMUL(_t,_t), V1p5)); \
    _t = VMUL(_t, VFNMA(_hs, VMUL(_t,_t), V1p5)); \
    VD _d = VFMA(_s,_t,Vone); \
    VD _q = _mm512_div_pd(Vone, _d); \
    MST(pr, mm, VMUL(_zr,_q)); MST(pi, mm, VMUL(_zi,_q)); } while(0)

// ---------------------------------------------------------------------------
// KM bodies: generic over LOADR/LOADI(j) and STOREP(k, vr, vi)
// ---------------------------------------------------------------------------
#define KM6_BODY(LOADR, LOADI, STOREP) do { \
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half); \
    VD x0r=LOADR(0),x0i=LOADI(0),x1r=LOADR(1),x1i=LOADI(1),x2r=LOADR(2),x2i=LOADI(2); \
    VD x3r=LOADR(3),x3i=LOADI(3),x4r=LOADR(4),x4i=LOADI(4),x5r=LOADR(5),x5i=LOADI(5); \
    VD A00r,A00i,A01r,A01i,A02r,A02i, A10r,A10i,A11r,A11i,A12r,A12i; \
    DFT3V(A00r,A00i,A01r,A01i,A02r,A02i, x0r,x0i,x2r,x2i,x4r,x4i); \
    DFT3V(A10r,A10i,A11r,A11i,A12r,A12i, x3r,x3i,x5r,x5i,x1r,x1i); \
    STOREP(0, VADD(A00r,A10r), VADD(A00i,A10i)); \
    STOREP(3, VSUB(A00r,A10r), VSUB(A00i,A10i)); \
    STOREP(4, VADD(A01r,A11r), VADD(A01i,A11i)); \
    STOREP(1, VSUB(A01r,A11r), VSUB(A01i,A11i)); \
    STOREP(2, VADD(A02r,A12r), VADD(A02i,A12i)); \
    STOREP(5, VSUB(A02r,A12r), VSUB(A02i,A12i)); } while(0)

#define KM8_BODY(LOADR, LOADI, STOREP) do { \
    const VD Vr2=BC(g_r2half); \
    VD x0r=LOADR(0),x0i=LOADI(0),x1r=LOADR(1),x1i=LOADI(1),x2r=LOADR(2),x2i=LOADI(2),x3r=LOADR(3),x3i=LOADI(3); \
    VD x4r=LOADR(4),x4i=LOADI(4),x5r=LOADR(5),x5i=LOADI(5),x6r=LOADR(6),x6i=LOADI(6),x7r=LOADR(7),x7i=LOADI(7); \
    VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i; \
    DFT8V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i, \
          x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i); \
    STOREP(0,o0r,o0i); STOREP(1,o1r,o1i); STOREP(2,o2r,o2i); STOREP(3,o3r,o3i); \
    STOREP(4,o4r,o4i); STOREP(5,o5r,o5i); STOREP(6,o6r,o6i); STOREP(7,o7r,o7i); } while(0)

static const int P36IN[4][9] = {
 {0,4,8,12,16,20,24,28,32},{9,13,17,21,25,29,33,1,5},
 {18,22,26,30,34,2,6,10,14},{27,31,35,3,7,11,15,19,23}};
static const int P36OUT[4][9] = {
 {0,28,20,12,4,32,24,16,8},{9,1,29,21,13,5,33,25,17},
 {18,10,2,30,22,14,6,34,26},{27,19,11,3,31,23,15,7,35}};

static const int P36IN_S36[4][9] = {
 {0,144,288,432,576,720,864,1008,1152},
 {324,468,612,756,900,1044,1188,36,180},
 {648,792,936,1080,1224,72,216,360,504},
 {972,1116,1260,108,252,396,540,684,828}};
static const int P36OUT_S36[4][9] = {
 {0,1008,720,432,144,1152,864,576,288},
 {324,36,1044,756,468,180,1188,900,612},
 {648,360,72,1080,792,504,216,1224,936},
 {972,684,396,108,1116,828,540,252,1260}};
static const int P36IN_S8[4][9] = {
 {0,32,64,96,128,160,192,224,256},
 {72,104,136,168,200,232,264,8,40},
 {144,176,208,240,272,16,48,80,112},
 {216,248,280,24,56,88,120,152,184}};
static const int P36OUT_S8[4][9] = {
 {0,224,160,96,32,256,192,128,64},
 {72,8,232,168,104,40,264,200,136},
 {144,80,16,240,176,112,48,272,208},
 {216,152,88,24,248,184,120,56,280}};
static const int P36IN_S1296[4][9] = {
 {0,5184,10368,15552,20736,25920,31104,36288,41472},
 {11664,16848,22032,27216,32400,37584,42768,1296,6480},
 {23328,28512,33696,38880,44064,2592,7776,12960,18144},
 {34992,40176,45360,3888,9072,14256,19440,24624,29808}};
static const int P36OUT_S1296[4][9] = {
 {0,36288,25920,15552,5184,41472,31104,20736,10368},
 {11664,1296,37584,27216,16848,6480,42768,32400,22032},
 {23328,12960,2592,38880,28512,18144,7776,44064,33696},
 {34992,24624,14256,3888,40176,29808,19440,9072,45360}};
static const int P45IN_S45[5][9] = {
 {0,225,450,675,900,1125,1350,1575,1800},
 {405,630,855,1080,1305,1530,1755,1980,180},
 {810,1035,1260,1485,1710,1935,135,360,585},
 {1215,1440,1665,1890,90,315,540,765,990},
 {1620,1845,45,270,495,720,945,1170,1395}};
static const int P45OUT_S45[5][9] = {
 {0,450,900,1350,1800,225,675,1125,1575},
 {1620,45,495,945,1395,1845,270,720,1170},
 {1215,1665,90,540,990,1440,1890,315,765},
 {810,1260,1710,135,585,1035,1485,1935,360},
 {405,855,1305,1755,180,630,1080,1530,1980}};
static const int P45IN_S8[5][9] = {
 {0,40,80,120,160,200,240,280,320},
 {72,112,152,192,232,272,312,352,32},
 {144,184,224,264,304,344,24,64,104},
 {216,256,296,336,16,56,96,136,176},
 {288,328,8,48,88,128,168,208,248}};
static const int P45OUT_S8[5][9] = {
 {0,80,160,240,320,40,120,200,280},
 {288,8,88,168,248,328,48,128,208},
 {216,296,16,96,176,256,336,56,136},
 {144,224,304,24,104,184,264,344,64},
 {72,152,232,312,32,112,192,272,352}};
static const int P45IN_S2025[5][9] = {
 {0,10125,20250,30375,40500,50625,60750,70875,81000},
 {18225,28350,38475,48600,58725,68850,78975,89100,8100},
 {36450,46575,56700,66825,76950,87075,6075,16200,26325},
 {54675,64800,74925,85050,4050,14175,24300,34425,44550},
 {72900,83025,2025,12150,22275,32400,42525,52650,62775}};
static const int P45OUT_S2025[5][9] = {
 {0,20250,40500,60750,81000,10125,30375,50625,70875},
 {72900,2025,22275,42525,62775,83025,12150,32400,52650},
 {54675,74925,4050,24300,44550,64800,85050,14175,34425},
 {36450,56700,76950,6075,26325,46575,66825,87075,16200},
 {18225,38475,58725,78975,8100,28350,48600,68850,89100}};

#define KM36_BODY(LOADR, LOADI, STOREP, PIN, POUT) do { \
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half); \
    const VD Vw91r=BC(g_w9[0]),Vw91i=BC(g_w9[1]),Vw92r=BC(g_w9[2]),Vw92i=BC(g_w9[3]),Vw94r=BC(g_w9[4]),Vw94i=BC(g_w9[5]); \
    VD Ar[4][9], Ai[4][9]; \
    for (int j4 = 0; j4 < 4; j4++) { \
        const int *pin = (PIN)[j4]; \
        VD y0r=LOADR(pin[0]),y0i=LOADI(pin[0]),y1r=LOADR(pin[1]),y1i=LOADI(pin[1]),y2r=LOADR(pin[2]),y2i=LOADI(pin[2]); \
        VD y3r=LOADR(pin[3]),y3i=LOADI(pin[3]),y4r=LOADR(pin[4]),y4i=LOADI(pin[4]),y5r=LOADR(pin[5]),y5i=LOADI(pin[5]); \
        VD y6r=LOADR(pin[6]),y6i=LOADI(pin[6]),y7r=LOADR(pin[7]),y7i=LOADI(pin[7]),y8r=LOADR(pin[8]),y8i=LOADI(pin[8]); \
        DFT9V(Ar[j4][0],Ai[j4][0],Ar[j4][1],Ai[j4][1],Ar[j4][2],Ai[j4][2],Ar[j4][3],Ai[j4][3],Ar[j4][4],Ai[j4][4], \
              Ar[j4][5],Ai[j4][5],Ar[j4][6],Ai[j4][6],Ar[j4][7],Ai[j4][7],Ar[j4][8],Ai[j4][8], \
              y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i,y8r,y8i); \
    } \
    for (int k9 = 0; k9 < 9; k9++) { \
        VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i; \
        DFT4V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i, \
              Ar[0][k9],Ai[0][k9],Ar[1][k9],Ai[1][k9],Ar[2][k9],Ai[2][k9],Ar[3][k9],Ai[3][k9]); \
        STOREP((POUT)[0][k9], o0r, o0i); \
        STOREP((POUT)[1][k9], o1r, o1i); \
        STOREP((POUT)[2][k9], o2r, o2i); \
        STOREP((POUT)[3][k9], o3r, o3i); \
    } } while(0)

static const int P45IN[5][9] = {
 {0,5,10,15,20,25,30,35,40},{9,14,19,24,29,34,39,44,4},
 {18,23,28,33,38,43,3,8,13},{27,32,37,42,2,7,12,17,22},{36,41,1,6,11,16,21,26,31}};
static const int P45OUT[5][9] = {
 {0,10,20,30,40,5,15,25,35},{36,1,11,21,31,41,6,16,26},
 {27,37,2,12,22,32,42,7,17},{18,28,38,3,13,23,33,43,8},{9,19,29,39,4,14,24,34,44}};

#define KM45_BODY(LOADR, LOADI, STOREP, PIN, POUT) do { \
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half); \
    const VD Vw91r=BC(g_w9[0]),Vw91i=BC(g_w9[1]),Vw92r=BC(g_w9[2]),Vw92i=BC(g_w9[3]),Vw94r=BC(g_w9[4]),Vw94i=BC(g_w9[5]); \
    const VD Vc51=BC(g_c51),Vc52=BC(g_c52),Vs51=BC(g_s51),Vs52=BC(g_s52); \
    VD Ar[5][9], Ai[5][9]; \
    for (int j5 = 0; j5 < 5; j5++) { \
        const int *pin = (PIN)[j5]; \
        VD y0r=LOADR(pin[0]),y0i=LOADI(pin[0]),y1r=LOADR(pin[1]),y1i=LOADI(pin[1]),y2r=LOADR(pin[2]),y2i=LOADI(pin[2]); \
        VD y3r=LOADR(pin[3]),y3i=LOADI(pin[3]),y4r=LOADR(pin[4]),y4i=LOADI(pin[4]),y5r=LOADR(pin[5]),y5i=LOADI(pin[5]); \
        VD y6r=LOADR(pin[6]),y6i=LOADI(pin[6]),y7r=LOADR(pin[7]),y7i=LOADI(pin[7]),y8r=LOADR(pin[8]),y8i=LOADI(pin[8]); \
        DFT9V(Ar[j5][0],Ai[j5][0],Ar[j5][1],Ai[j5][1],Ar[j5][2],Ai[j5][2],Ar[j5][3],Ai[j5][3],Ar[j5][4],Ai[j5][4], \
              Ar[j5][5],Ai[j5][5],Ar[j5][6],Ai[j5][6],Ar[j5][7],Ai[j5][7],Ar[j5][8],Ai[j5][8], \
              y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i,y8r,y8i); \
    } \
    for (int k9 = 0; k9 < 9; k9++) { \
        VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i; \
        DFT5V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i, \
              Ar[0][k9],Ai[0][k9],Ar[1][k9],Ai[1][k9],Ar[2][k9],Ai[2][k9],Ar[3][k9],Ai[3][k9],Ar[4][k9],Ai[4][k9]); \
        STOREP((POUT)[0][k9], o0r, o0i); \
        STOREP((POUT)[1][k9], o1r, o1i); \
        STOREP((POUT)[2][k9], o2r, o2i); \
        STOREP((POUT)[3][k9], o3r, o3i); \
        STOREP((POUT)[4][k9], o4r, o4i); \
    } } while(0)

#define KM64_BODY(LOADR, LOADI, STOREP) do { \
    const VD Vr2=BC(g_r2half); \
    VD BufR[64], BufI[64]; \
    for (int b = 0; b < 8; b++) { \
        VD x0r=LOADR(b),x0i=LOADI(b),x1r=LOADR(8+b),x1i=LOADI(8+b),x2r=LOADR(16+b),x2i=LOADI(16+b),x3r=LOADR(24+b),x3i=LOADI(24+b); \
        VD x4r=LOADR(32+b),x4i=LOADI(32+b),x5r=LOADR(40+b),x5i=LOADI(40+b),x6r=LOADR(48+b),x6i=LOADI(48+b),x7r=LOADR(56+b),x7i=LOADI(56+b); \
        VD F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i; \
        DFT8V(F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i, \
              x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i); \
        BufR[0*8+b]=F0r; BufI[0*8+b]=F0i; \
        if (b == 0) { \
            BufR[1*8]=F1r; BufI[1*8]=F1i; BufR[2*8]=F2r; BufI[2*8]=F2i; BufR[3*8]=F3r; BufI[3*8]=F3i; \
            BufR[4*8]=F4r; BufI[4*8]=F4i; BufR[5*8]=F5r; BufI[5*8]=F5i; BufR[6*8]=F6r; BufI[6*8]=F6i; BufR[7*8]=F7r; BufI[7*8]=F7i; \
        } else { \
            VD gr,gi; \
            CMULV(gr,gi,F1r,F1i,BC(g_T64r[b][1]),BC(g_T64i[b][1])); BufR[1*8+b]=gr; BufI[1*8+b]=gi; \
            CMULV(gr,gi,F2r,F2i,BC(g_T64r[b][2]),BC(g_T64i[b][2])); BufR[2*8+b]=gr; BufI[2*8+b]=gi; \
            CMULV(gr,gi,F3r,F3i,BC(g_T64r[b][3]),BC(g_T64i[b][3])); BufR[3*8+b]=gr; BufI[3*8+b]=gi; \
            CMULV(gr,gi,F4r,F4i,BC(g_T64r[b][4]),BC(g_T64i[b][4])); BufR[4*8+b]=gr; BufI[4*8+b]=gi; \
            CMULV(gr,gi,F5r,F5i,BC(g_T64r[b][5]),BC(g_T64i[b][5])); BufR[5*8+b]=gr; BufI[5*8+b]=gi; \
            CMULV(gr,gi,F6r,F6i,BC(g_T64r[b][6]),BC(g_T64i[b][6])); BufR[6*8+b]=gr; BufI[6*8+b]=gi; \
            CMULV(gr,gi,F7r,F7i,BC(g_T64r[b][7]),BC(g_T64i[b][7])); BufR[7*8+b]=gr; BufI[7*8+b]=gi; \
        } \
    } \
    for (int t = 0; t < 8; t++) { \
        VD H0r,H0i,H1r,H1i,H2r,H2i,H3r,H3i,H4r,H4i,H5r,H5i,H6r,H6i,H7r,H7i; \
        DFT8V(H0r,H0i,H1r,H1i,H2r,H2i,H3r,H3i,H4r,H4i,H5r,H5i,H6r,H6i,H7r,H7i, \
              BufR[t*8+0],BufI[t*8+0],BufR[t*8+1],BufI[t*8+1],BufR[t*8+2],BufI[t*8+2],BufR[t*8+3],BufI[t*8+3], \
              BufR[t*8+4],BufI[t*8+4],BufR[t*8+5],BufI[t*8+5],BufR[t*8+6],BufI[t*8+6],BufR[t*8+7],BufI[t*8+7]); \
        STOREP(t,    H0r,H0i); STOREP(t+ 8, H1r,H1i); STOREP(t+16, H2r,H2i); STOREP(t+24, H3r,H3i); \
        STOREP(t+32, H4r,H4i); STOREP(t+40, H5r,H5i); STOREP(t+48, H6r,H6i); STOREP(t+56, H7r,H7i); \
    } } while(0)

// Folded odd-prime body with k-pair unrolling for ILP.
#define KMFOLD_BODY(LL, HH, TC, TS, LOADR, LOADI, STOREP) do { \
    VD ur[HH+1], ui[HH+1], vr[HH+1], vi[HH+1]; \
    VD x0r = LOADR(0), x0i = LOADI(0); \
    VD sur0 = x0r, sui0 = x0i, sur1 = ZERO, sui1 = ZERO; \
    _Pragma("GCC unroll 16") \
    for (int j = 1; j <= HH; j++) { \
        VD ar = LOADR(j), ai = LOADI(j); \
        VD br = LOADR(LL-j), bi = LOADI(LL-j); \
        ur[j]=VADD(ar,br); ui[j]=VADD(ai,bi); \
        vr[j]=VSUB(ar,br); vi[j]=VSUB(ai,bi); \
        if (j & 1) { sur0=VADD(sur0,ur[j]); sui0=VADD(sui0,ui[j]); } \
        else       { sur1=VADD(sur1,ur[j]); sui1=VADD(sui1,ui[j]); } \
    } \
    STOREP(0, VADD(sur0,sur1), VADD(sui0,sui1)); \
    _Pragma("GCC unroll 16") \
    for (int k = 1; k + 1 <= HH; k += 2) { \
        VD P1r = x0r, P1i = x0i, Q1r = ZERO, Q1i = ZERO; \
        VD P2r = x0r, P2i = x0i, Q2r = ZERO, Q2i = ZERO; \
        _Pragma("GCC unroll 16") \
        for (int j = 1; j <= HH; j++) { \
            VD c1 = BC(TC[(k-1)*HH + (j-1)]), s1 = BC(TS[(k-1)*HH + (j-1)]); \
            VD c2 = BC(TC[k*HH + (j-1)]),     s2 = BC(TS[k*HH + (j-1)]); \
            VD uurr = ur[j], uuii = ui[j], vvrr = vr[j], vvii = vi[j]; \
            P1r = VFMA(c1, uurr, P1r); P1i = VFMA(c1, uuii, P1i); \
            Q1r = VFMA(s1, vvrr, Q1r); Q1i = VFMA(s1, vvii, Q1i); \
            P2r = VFMA(c2, uurr, P2r); P2i = VFMA(c2, uuii, P2i); \
            Q2r = VFMA(s2, vvrr, Q2r); Q2i = VFMA(s2, vvii, Q2i); \
        } \
        STOREP(k,      VADD(P1r,Q1i), VSUB(P1i,Q1r)); \
        STOREP(LL-k,   VSUB(P1r,Q1i), VADD(P1i,Q1r)); \
        STOREP(k+1,    VADD(P2r,Q2i), VSUB(P2i,Q2r)); \
        STOREP(LL-k-1, VSUB(P2r,Q2i), VADD(P2i,Q2r)); \
    } \
    if (HH & 1) { \
        const int k = HH; \
        VD P1r = x0r, P1i = x0i, Q1r = ZERO, Q1i = ZERO; \
        VD P2r = ZERO, P2i = ZERO, Q2r = ZERO, Q2i = ZERO; \
        _Pragma("GCC unroll 16") \
        for (int j = 1; j + 1 <= HH; j += 2) { \
            VD c1 = BC(TC[(k-1)*HH + (j-1)]), s1 = BC(TS[(k-1)*HH + (j-1)]); \
            VD c2 = BC(TC[(k-1)*HH + j]),     s2 = BC(TS[(k-1)*HH + j]); \
            P1r = VFMA(c1, ur[j], P1r); P1i = VFMA(c1, ui[j], P1i); \
            Q1r = VFMA(s1, vr[j], Q1r); Q1i = VFMA(s1, vi[j], Q1i); \
            P2r = VFMA(c2, ur[j+1], P2r); P2i = VFMA(c2, ui[j+1], P2i); \
            Q2r = VFMA(s2, vr[j+1], Q2r); Q2i = VFMA(s2, vi[j+1], Q2i); \
        } \
        { \
            VD c1 = BC(TC[(k-1)*HH + (HH-1)]), s1 = BC(TS[(k-1)*HH + (HH-1)]); \
            P1r = VFMA(c1, ur[HH], P1r); P1i = VFMA(c1, ui[HH], P1i); \
            Q1r = VFMA(s1, vr[HH], Q1r); Q1i = VFMA(s1, vi[HH], Q1i); \
        } \
        P1r = VADD(P1r,P2r); P1i = VADD(P1i,P2i); \
        Q1r = VADD(Q1r,Q2r); Q1i = VADD(Q1i,Q2i); \
        STOREP(k,    VADD(P1r,Q1i), VSUB(P1i,Q1r)); \
        STOREP(LL-k, VSUB(P1r,Q1i), VADD(P1i,Q1r)); \
    } } while(0)

// Folded odd-prime body processing TWO lane-chunks (base+0, base+8) at once,
// sharing the C/S broadcasts. k-pair x 2-chunk accumulators.
#define KMFOLD2_BODY(LL, HH, TC, TS, LOADR, LOADI, STOREP, STOREP2) do { \
    VD ur[HH+1], ui[HH+1], vr[HH+1], vi[HH+1]; \
    VD u2r[HH+1], u2i[HH+1], v2r[HH+1], v2i[HH+1]; \
    VD x0r = LOADR(0,0), x0i = LOADI(0,0); \
    VD y0r = LOADR(0,8), y0i = LOADI(0,8); \
    VD sur0 = x0r, sui0 = x0i, sur1 = ZERO, sui1 = ZERO; \
    VD tur0 = y0r, tui0 = y0i, tur1 = ZERO, tui1 = ZERO; \
    _Pragma("GCC unroll 16") \
    for (int j = 1; j <= HH; j++) { \
        VD ar = LOADR(j,0), ai = LOADI(j,0); \
        VD br = LOADR(LL-j,0), bi = LOADI(LL-j,0); \
        ur[j]=VADD(ar,br); ui[j]=VADD(ai,bi); \
        vr[j]=VSUB(ar,br); vi[j]=VSUB(ai,bi); \
        VD cr = LOADR(j,8), ci = LOADI(j,8); \
        VD dr = LOADR(LL-j,8), di = LOADI(LL-j,8); \
        u2r[j]=VADD(cr,dr); u2i[j]=VADD(ci,di); \
        v2r[j]=VSUB(cr,dr); v2i[j]=VSUB(ci,di); \
        if (j & 1) { sur0=VADD(sur0,ur[j]); sui0=VADD(sui0,ui[j]); tur0=VADD(tur0,u2r[j]); tui0=VADD(tui0,u2i[j]); } \
        else       { sur1=VADD(sur1,ur[j]); sui1=VADD(sui1,ui[j]); tur1=VADD(tur1,u2r[j]); tui1=VADD(tui1,u2i[j]); } \
    } \
    STOREP(0, VADD(sur0,sur1), VADD(sui0,sui1)); \
    STOREP2(0, VADD(tur0,tur1), VADD(tui0,tui1)); \
    _Pragma("GCC unroll 16") \
    for (int k = 1; k + 1 <= HH; k += 2) { \
        VD P1r = x0r, P1i = x0i, Q1r = ZERO, Q1i = ZERO; \
        VD P2r = x0r, P2i = x0i, Q2r = ZERO, Q2i = ZERO; \
        VD R1r = y0r, R1i = y0i, S1r = ZERO, S1i = ZERO; \
        VD R2r = y0r, R2i = y0i, S2r = ZERO, S2i = ZERO; \
        _Pragma("GCC unroll 16") \
        for (int j = 1; j <= HH; j++) { \
            VD c1 = BC(TC[(k-1)*HH + (j-1)]), s1 = BC(TS[(k-1)*HH + (j-1)]); \
            VD c2 = BC(TC[k*HH + (j-1)]),     s2 = BC(TS[k*HH + (j-1)]); \
            P1r = VFMA(c1, ur[j], P1r); P1i = VFMA(c1, ui[j], P1i); \
            Q1r = VFMA(s1, vr[j], Q1r); Q1i = VFMA(s1, vi[j], Q1i); \
            P2r = VFMA(c2, ur[j], P2r); P2i = VFMA(c2, ui[j], P2i); \
            Q2r = VFMA(s2, vr[j], Q2r); Q2i = VFMA(s2, vi[j], Q2i); \
            R1r = VFMA(c1, u2r[j], R1r); R1i = VFMA(c1, u2i[j], R1i); \
            S1r = VFMA(s1, v2r[j], S1r); S1i = VFMA(s1, v2i[j], S1i); \
            R2r = VFMA(c2, u2r[j], R2r); R2i = VFMA(c2, u2i[j], R2i); \
            S2r = VFMA(s2, v2r[j], S2r); S2i = VFMA(s2, v2i[j], S2i); \
        } \
        STOREP(k,      VADD(P1r,Q1i), VSUB(P1i,Q1r)); \
        STOREP(LL-k,   VSUB(P1r,Q1i), VADD(P1i,Q1r)); \
        STOREP(k+1,    VADD(P2r,Q2i), VSUB(P2i,Q2r)); \
        STOREP(LL-k-1, VSUB(P2r,Q2i), VADD(P2i,Q2r)); \
        STOREP2(k,      VADD(R1r,S1i), VSUB(R1i,S1r)); \
        STOREP2(LL-k,   VSUB(R1r,S1i), VADD(R1i,S1r)); \
        STOREP2(k+1,    VADD(R2r,S2i), VSUB(R2i,S2r)); \
        STOREP2(LL-k-1, VSUB(R2r,S2i), VADD(R2i,S2r)); \
    } \
    if (HH & 1) { \
        const int k = HH; \
        VD P1r = x0r, P1i = x0i, Q1r = ZERO, Q1i = ZERO; \
        VD R1r = y0r, R1i = y0i, S1r = ZERO, S1i = ZERO; \
        _Pragma("GCC unroll 16") \
        for (int j = 1; j <= HH; j++) { \
            VD c1 = BC(TC[(k-1)*HH + (j-1)]), s1 = BC(TS[(k-1)*HH + (j-1)]); \
            P1r = VFMA(c1, ur[j], P1r); P1i = VFMA(c1, ui[j], P1i); \
            Q1r = VFMA(s1, vr[j], Q1r); Q1i = VFMA(s1, vi[j], Q1i); \
            R1r = VFMA(c1, u2r[j], R1r); R1i = VFMA(c1, u2i[j], R1i); \
            S1r = VFMA(s1, v2r[j], S1r); S1i = VFMA(s1, v2i[j], S1i); \
        } \
        STOREP(k,    VADD(P1r,Q1i), VSUB(P1i,Q1r)); \
        STOREP(LL-k, VSUB(P1r,Q1i), VADD(P1i,Q1r)); \
        STOREP2(k,    VADD(R1r,S1i), VSUB(R1i,S1r)); \
        STOREP2(LL-k, VSUB(R1r,S1i), VADD(R1i,S1r)); \
    } } while(0)
// ---------------------------------------------------------------------------
// Kernel instantiations (out-of-place: separate src/dst)
// ---------------------------------------------------------------------------
// ====== L = 6 ======
static inline __attribute__((always_inline)) void km6_y(const double* restrict re, const double* restrict im,
                  double* restrict dre, double* restrict dim, MK m)
{
#define LR_(j) LD(re + 6*(j))
#define LI_(j) LD(im + 6*(j))
#define SP_(k,vr,vi) do { MST(dre + 6*(k), m, vr); MST(dim + 6*(k), m, vi); } while(0)
    KM6_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static inline __attribute__((always_inline)) void km6_x_map(const double* restrict re, const double* restrict im,
                      double* restrict dre, double* restrict dim,
                      const double* restrict cre, const double* restrict cim, MK m)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308);
#define LR_(j) LD(re + 36*(j))
#define LI_(j) LD(im + 36*(j))
#define SP_(k,vr,vi) MAPST(dre + 36*(k), dim + 36*(k), m, vr, vi, cre + 36*(k), cim + 36*(k))
    KM6_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static inline __attribute__((always_inline)) void zpass6(const double* restrict sre, const double* restrict sim,
                   double* restrict dre, double* restrict dim)
{
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half);
    for (int g = 0; g < 5; g++) {
        int nrows = (g < 4) ? 8 : 4;
        const double *pr = sre + 48*g, *pi = sim + 48*g;
        double *qr = dre + 48*g, *qi = dim + 48*g;
        VD r0,r1,r2,r3,r4,r5,r6,r7, t0r,t1r,t2r,t3r,t4r,t5r,t6r,t7r, t0i,t1i,t2i,t3i,t4i,t5i,t6i,t7i;
        r0=LD(pr); r1=LD(pr+6); r2=LD(pr+12); r3=LD(pr+18); r4=LD(pr+24); r5=LD(pr+30);
        r6 = (nrows>6)?LD(pr+36):ZERO; r7=(nrows>7)?LD(pr+42):ZERO;
        TR8(t0r,t1r,t2r,t3r,t4r,t5r,t6r,t7r, r0,r1,r2,r3,r4,r5,r6,r7);
        r0=LD(pi); r1=LD(pi+6); r2=LD(pi+12); r3=LD(pi+18); r4=LD(pi+24); r5=LD(pi+30);
        r6 = (nrows>6)?LD(pi+36):ZERO; r7=(nrows>7)?LD(pi+42):ZERO;
        TR8(t0i,t1i,t2i,t3i,t4i,t5i,t6i,t7i, r0,r1,r2,r3,r4,r5,r6,r7);
        VD A00r,A00i,A01r,A01i,A02r,A02i, A10r,A10i,A11r,A11i,A12r,A12i;
        DFT3V(A00r,A00i,A01r,A01i,A02r,A02i, t0r,t0i,t2r,t2i,t4r,t4i);
        DFT3V(A10r,A10i,A11r,A11i,A12r,A12i, t3r,t3i,t5r,t5i,t1r,t1i);
        VD o0r=VADD(A00r,A10r), o0i=VADD(A00i,A10i);
        VD o3r=VSUB(A00r,A10r), o3i=VSUB(A00i,A10i);
        VD o4r=VADD(A01r,A11r), o4i=VADD(A01i,A11i);
        VD o1r=VSUB(A01r,A11r), o1i=VSUB(A01i,A11i);
        VD o2r=VADD(A02r,A12r), o2i=VADD(A02i,A12i);
        VD o5r=VSUB(A02r,A12r), o5i=VSUB(A02i,A12i);
        TR8(r0,r1,r2,r3,r4,r5,r6,r7, o0r,o1r,o2r,o3r,o4r,o5r,o5r,o5r);
        MST(qr, 0x3F, r0); MST(qr+6, 0x3F, r1); MST(qr+12, 0x3F, r2); MST(qr+18, 0x3F, r3);
        if (nrows > 4) { MST(qr+24, 0x3F, r4); MST(qr+30, 0x3F, r5); MST(qr+36, 0x3F, r6); MST(qr+42, 0x3F, r7); }
        TR8(r0,r1,r2,r3,r4,r5,r6,r7, o0i,o1i,o2i,o3i,o4i,o5i,o5i,o5i);
        MST(qi, 0x3F, r0); MST(qi+6, 0x3F, r1); MST(qi+12, 0x3F, r2); MST(qi+18, 0x3F, r3);
        if (nrows > 4) { MST(qi+24, 0x3F, r4); MST(qi+30, 0x3F, r5); MST(qi+36, 0x3F, r6); MST(qi+42, 0x3F, r7); }
    }
}

// ====== L = 8 ======
static inline __attribute__((always_inline)) void km8_y(const double* restrict re, const double* restrict im,
                  double* restrict dre, double* restrict dim)
{
#define LR_(j) LD(re + 8*(j))
#define LI_(j) LD(im + 8*(j))
#define SP_(k,vr,vi) do { ST(dre + 8*(k), vr); ST(dim + 8*(k), vi); } while(0)
    KM8_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static inline __attribute__((always_inline)) void km8_x_map(const double* restrict re, const double* restrict im,
                      double* restrict dre, double* restrict dim,
                      const double* restrict cre, const double* restrict cim)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
    const MK m = 0xFF;
#define LR_(j) LD(re + 64*(j))
#define LI_(j) LD(im + 64*(j))
#define SP_(k,vr,vi) MAPST(dre + 64*(k), dim + 64*(k), m, vr, vi, cre + 64*(k), cim + 64*(k))
    KM8_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static inline __attribute__((always_inline)) void zpass8(const double* restrict sre, const double* restrict sim,
                   double* restrict dre, double* restrict dim)
{
    const VD Vr2=BC(g_r2half);
    for (int g = 0; g < 8; g++) {
        const double *pr = sre + 64*g, *pi = sim + 64*g;
        double *qr = dre + 64*g, *qi = dim + 64*g;
        VD r0,r1,r2,r3,r4,r5,r6,r7, t0r,t1r,t2r,t3r,t4r,t5r,t6r,t7r, t0i,t1i,t2i,t3i,t4i,t5i,t6i,t7i;
        r0=LD(pr); r1=LD(pr+8); r2=LD(pr+16); r3=LD(pr+24); r4=LD(pr+32); r5=LD(pr+40); r6=LD(pr+48); r7=LD(pr+56);
        TR8(t0r,t1r,t2r,t3r,t4r,t5r,t6r,t7r, r0,r1,r2,r3,r4,r5,r6,r7);
        r0=LD(pi); r1=LD(pi+8); r2=LD(pi+16); r3=LD(pi+24); r4=LD(pi+32); r5=LD(pi+40); r6=LD(pi+48); r7=LD(pi+56);
        TR8(t0i,t1i,t2i,t3i,t4i,t5i,t6i,t7i, r0,r1,r2,r3,r4,r5,r6,r7);
        VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i;
        DFT8V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,
              t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i,t4r,t4i,t5r,t5i,t6r,t6i,t7r,t7i);
        TR8(r0,r1,r2,r3,r4,r5,r6,r7, o0r,o1r,o2r,o3r,o4r,o5r,o6r,o7r);
        ST(qr,r0); ST(qr+8,r1); ST(qr+16,r2); ST(qr+24,r3); ST(qr+32,r4); ST(qr+40,r5); ST(qr+48,r6); ST(qr+56,r7);
        TR8(r0,r1,r2,r3,r4,r5,r6,r7, o0i,o1i,o2i,o3i,o4i,o5i,o6i,o7i);
        ST(qi,r0); ST(qi+8,r1); ST(qi+16,r2); ST(qi+24,r3); ST(qi+32,r4); ST(qi+40,r5); ST(qi+48,r6); ST(qi+56,r7);
    }
}
// ====== odd primes 13 / 17 / 23 ======
static void km13_y(const double* restrict re, const double* restrict im,
                   double* restrict dre, double* restrict dim, MK m)
{
#define LR_(j) LD(re + 13*(j))
#define LI_(j) LD(im + 13*(j))
#define SP_(k,vr,vi) do { MST(dre + 13*(k), m, vr); MST(dim + 13*(k), m, vi); } while(0)
    KMFOLD_BODY(13, 6, g_C13, g_S13, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void km13_x_map(const double* restrict re, const double* restrict im,
                       double* restrict dre, double* restrict dim,
                       const double* restrict cre, const double* restrict cim, MK m)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
#define LR_(j) LD(re + 169*(j))
#define LI_(j) LD(im + 169*(j))
#define SP_(k,vr,vi) MAPST(dre + 169*(k), dim + 169*(k), m, vr, vi, cre + 169*(k), cim + 169*(k))
    KMFOLD_BODY(13, 6, g_C13, g_S13, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void km17_y(const double* restrict re, const double* restrict im,
                   double* restrict dre, double* restrict dim, MK m)
{
#define LR_(j) LD(re + 17*(j))
#define LI_(j) LD(im + 17*(j))
#define SP_(k,vr,vi) do { MST(dre + 17*(k), m, vr); MST(dim + 17*(k), m, vi); } while(0)
    KMFOLD_BODY(17, 8, g_C17, g_S17, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void km17_x_map(const double* restrict re, const double* restrict im,
                       double* restrict dre, double* restrict dim,
                       const double* restrict cre, const double* restrict cim, MK m)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
#define LR_(j) LD(re + 289*(j))
#define LI_(j) LD(im + 289*(j))
#define SP_(k,vr,vi) MAPST(dre + 289*(k), dim + 289*(k), m, vr, vi, cre + 289*(k), cim + 289*(k))
    KMFOLD_BODY(17, 8, g_C17, g_S17, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void km23_y(const double* restrict re, const double* restrict im,
                   double* restrict dre, double* restrict dim, MK m)
{
#define LR_(j) LD(re + 23*(j))
#define LI_(j) LD(im + 23*(j))
#define SP_(k,vr,vi) do { MST(dre + 23*(k), m, vr); MST(dim + 23*(k), m, vi); } while(0)
    KMFOLD_BODY(23, 11, g_C23, g_S23, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void km23_x_map(const double* restrict re, const double* restrict im,
                       double* restrict dre, double* restrict dim,
                       const double* restrict cre, const double* restrict cim, MK m)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
#define LR_(j) LD(re + 529*(j))
#define LI_(j) LD(im + 529*(j))
#define SP_(k,vr,vi) MAPST(dre + 529*(k), dim + 529*(k), m, vr, vi, cre + 529*(k), cim + 529*(k))
    KMFOLD_BODY(23, 11, g_C23, g_S23, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}

#define REVIDX _mm512_set_epi64(0,1,2,3,4,5,6,7)
#define VREV(v) _mm512_permutexvar_pd(REVIDX, v)

// z-axis for 13 (OOP): h=6, vector prep into stack, two rows at a time.
static void zpass13(const double* restrict sre, const double* restrict sim,
                    double* restrict dre, double* restrict dim, int nrows)
{
    VD CzR[6], SzR[6];
    for (int j = 0; j < 6; j++) { CzR[j]=LD(g_Cz13[j]); SzR[j]=LD(g_Sz13[j]); }
    double ub[2][4][8] __attribute__((aligned(64)));
#define ZP13PREP(pr, pi, R) do { \
    VD a_r = LD((pr)+1), a_i = LD((pi)+1); \
    VD b_r = VREV(LD((pr)+5)), b_i = VREV(LD((pi)+5)); \
    ST(ub[R][0], VADD(a_r,b_r)); ST(ub[R][1], VADD(a_i,b_i)); \
    ST(ub[R][2], VSUB(a_r,b_r)); ST(ub[R][3], VSUB(a_i,b_i)); \
} while(0)
#define ZP13ROW(pr, pi, qr, qi, R, X) \
    VD rsr##X = VADD(LD(pr), MLD(0x1F,(pr)+8)), rsi##X = VADD(LD(pi), MLD(0x1F,(pi)+8)); \
    VD Pr##X = BC((pr)[0]), Pi##X = BC((pi)[0]), Qr##X = ZERO, Qi##X = ZERO; \
    _Pragma("GCC unroll 8") \
    for (int j = 0; j < 6; j++) { \
        Pr##X = VFMA(BC(ub[R][0][j]), CzR[j], Pr##X); Pi##X = VFMA(BC(ub[R][1][j]), CzR[j], Pi##X); \
        Qr##X = VFMA(BC(ub[R][2][j]), SzR[j], Qr##X); Qi##X = VFMA(BC(ub[R][3][j]), SzR[j], Qi##X); \
    } \
    (qr)[0] = _mm512_reduce_add_pd(rsr##X); (qi)[0] = _mm512_reduce_add_pd(rsi##X); \
    MST((qr)+1, 0x3F, VADD(Pr##X,Qi##X)); MST((qi)+1, 0x3F, VSUB(Pi##X,Qr##X)); \
    MST((qr)+5, 0xFC, VREV(VSUB(Pr##X,Qi##X))); MST((qi)+5, 0xFC, VREV(VADD(Pi##X,Qr##X)));
    int r = 0;
    for (; r + 1 < nrows; r += 2) {
        const double *prA = sre + 13*r, *piA = sim + 13*r;
        const double *prB = prA + 13,  *piB = piA + 13;
        double *qrA = dre + 13*r, *qiA = dim + 13*r, *qrB = qrA + 13, *qiB = qiA + 13;
        ZP13PREP(prA, piA, 0);
        ZP13PREP(prB, piB, 1);
        ZP13ROW(prA, piA, qrA, qiA, 0, A);
        ZP13ROW(prB, piB, qrB, qiB, 1, B);
    }
    if (r < nrows) {
        const double *prA = sre + 13*r, *piA = sim + 13*r;
        double *qrA = dre + 13*r, *qiA = dim + 13*r;
        ZP13PREP(prA, piA, 0);
        ZP13ROW(prA, piA, qrA, qiA, 0, C);
    }
#undef ZP13PREP
#undef ZP13ROW
}
// z-axis for 17 (OOP): h=8 exact, vector prep, two rows at a time.
static void zpass17(const double* restrict sre, const double* restrict sim,
                    double* restrict dre, double* restrict dim, int nrows)
{
    VD CzR[8], SzR[8];
    for (int j = 0; j < 8; j++) { CzR[j]=LD(g_Cz17[j]); SzR[j]=LD(g_Sz17[j]); }
    double ub[2][4][8] __attribute__((aligned(64)));
#define ZP17PREP(pr, pi, R) do { \
    VD a_r = LD((pr)+1), a_i = LD((pi)+1); \
    VD b_r = VREV(LD((pr)+9)), b_i = VREV(LD((pi)+9)); \
    ST(ub[R][0], VADD(a_r,b_r)); ST(ub[R][1], VADD(a_i,b_i)); \
    ST(ub[R][2], VSUB(a_r,b_r)); ST(ub[R][3], VSUB(a_i,b_i)); \
} while(0)
#define ZP17ROW(pr, pi, qr, qi, R, X) \
    VD rsr##X = VADD(LD(pr), LD((pr)+8)), rsi##X = VADD(LD(pi), LD((pi)+8)); \
    VD Pr##X = BC((pr)[0]), Pi##X = BC((pi)[0]), Qr##X = ZERO, Qi##X = ZERO; \
    _Pragma("GCC unroll 8") \
    for (int j = 0; j < 8; j++) { \
        Pr##X = VFMA(BC(ub[R][0][j]), CzR[j], Pr##X); Pi##X = VFMA(BC(ub[R][1][j]), CzR[j], Pi##X); \
        Qr##X = VFMA(BC(ub[R][2][j]), SzR[j], Qr##X); Qi##X = VFMA(BC(ub[R][3][j]), SzR[j], Qi##X); \
    } \
    double dcr##X = _mm512_reduce_add_pd(rsr##X) + (pr)[16]; \
    double dci##X = _mm512_reduce_add_pd(rsi##X) + (pi)[16]; \
    (qr)[0] = dcr##X; (qi)[0] = dci##X; \
    ST((qr)+1, VADD(Pr##X,Qi##X)); ST((qi)+1, VSUB(Pi##X,Qr##X)); \
    ST((qr)+9, VREV(VSUB(Pr##X,Qi##X))); ST((qi)+9, VREV(VADD(Pi##X,Qr##X)));
    int r = 0;
    for (; r + 1 < nrows; r += 2) {
        const double *prA = sre + 17*r, *piA = sim + 17*r;
        const double *prB = prA + 17,  *piB = piA + 17;
        double *qrA = dre + 17*r, *qiA = dim + 17*r, *qrB = qrA + 17, *qiB = qiA + 17;
        ZP17PREP(prA, piA, 0);
        ZP17PREP(prB, piB, 1);
        ZP17ROW(prA, piA, qrA, qiA, 0, A);
        ZP17ROW(prB, piB, qrB, qiB, 1, B);
    }
    if (r < nrows) {
        const double *prA = sre + 17*r, *piA = sim + 17*r;
        double *qrA = dre + 17*r, *qiA = dim + 17*r;
        ZP17PREP(prA, piA, 0);
        ZP17ROW(prA, piA, qrA, qiA, 0, C);
    }
#undef ZP17PREP
#undef ZP17ROW
}
// z-axis for 23 (OOP): h=11, two k-chunks; vector prep into stack buffers.
static void zpass23(const double* restrict sre, const double* restrict sim,
                    double* restrict dre, double* restrict dim, int nrows)
{
    double ub[4][16] __attribute__((aligned(64)));  /* [ur,ui,vr,vi][j-1] */
#define ZP23PREP(pr, pi) do { \
    VD a_r = LD((pr)+1), a_i = LD((pi)+1); \
    VD b_r = VREV(LD((pr)+15)), b_i = VREV(LD((pi)+15)); \
    ST(ub[0], VADD(a_r,b_r)); ST(ub[1], VADD(a_i,b_i)); \
    ST(ub[2], VSUB(a_r,b_r)); ST(ub[3], VSUB(a_i,b_i)); \
    VD a2r = LD((pr)+9), a2i = LD((pi)+9); \
    VD b2r = VREV(LD((pr)+7)), b2i = VREV(LD((pi)+7)); \
    ST(ub[0]+8, VADD(a2r,b2r)); ST(ub[1]+8, VADD(a2i,b2i)); \
    ST(ub[2]+8, VSUB(a2r,b2r)); ST(ub[3]+8, VSUB(a2i,b2i)); \
} while(0)
    for (int r = 0; r < nrows; r++) {
        const double *pr = sre + 23*r, *pi = sim + 23*r;
        double *qr = dre + 23*r, *qi = dim + 23*r;
        VD rsr = VADD(VADD(LD(pr), LD(pr+8)), MLD(0x7F,pr+16));
        VD rsi = VADD(VADD(LD(pi), LD(pi+8)), MLD(0x7F,pi+16));
        ZP23PREP(pr, pi);
        VD P1r = BC(pr[0]), P1i = BC(pi[0]), Q1r = ZERO, Q1i = ZERO;
        VD P2r = P1r, P2i = P1i, Q2r = ZERO, Q2i = ZERO;
        _Pragma("GCC unroll 16")
        for (int j = 0; j < 11; j++) {
            VD c1 = LD(g_Cz23[0][j]), s1 = LD(g_Sz23[0][j]);
            VD c2 = LD(g_Cz23[1][j]), s2 = LD(g_Sz23[1][j]);
            VD ur = BC(ub[0][j]), uv = BC(ub[1][j]);
            VD wr = BC(ub[2][j]), wv = BC(ub[3][j]);
            P1r = VFMA(ur, c1, P1r); P1i = VFMA(uv, c1, P1i);
            Q1r = VFMA(wr, s1, Q1r); Q1i = VFMA(wv, s1, Q1i);
            P2r = VFMA(ur, c2, P2r); P2i = VFMA(uv, c2, P2i);
            Q2r = VFMA(wr, s2, Q2r); Q2i = VFMA(wv, s2, Q2i);
        }
        qr[0] = _mm512_reduce_add_pd(rsr); qi[0] = _mm512_reduce_add_pd(rsi);
        ST(qr+1,  VADD(P1r,Q1i)); ST(qi+1,  VSUB(P1i,Q1r));
        ST(qr+15, VREV(VSUB(P1r,Q1i))); ST(qi+15, VREV(VADD(P1i,Q1r)));
        MST(qr+9, 0x07, VADD(P2r,Q2i)); MST(qi+9, 0x07, VSUB(P2i,Q2r));
        MST(qr+7, 0xE0, VREV(VSUB(P2r,Q2i))); MST(qi+7, 0xE0, VREV(VADD(P2i,Q2r)));
    }
#undef ZP23PREP
}
// ====== L = 36 ======
static void km36_y(const double* restrict re, const double* restrict im,
                   double* restrict dre, double* restrict dim, MK m)
{
#define LR_(j) LD(re + (j))
#define LI_(j) LD(im + (j))
#define SP_(k,vr,vi) do { MST(dre + (k), m, vr); MST(dim + (k), m, vi); } while(0)
    KM36_BODY(LR_, LI_, SP_, P36IN_S36, P36OUT_S36);
#undef LR_
#undef LI_
#undef SP_
}
static void km36_s(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + (j))
#define LI_(j) LD(im + (j))
#define SP_(k,vr,vi) do { ST(re + (k), vr); ST(im + (k), vi); } while(0)
    KM36_BODY(LR_, LI_, SP_, P36IN_S8, P36OUT_S8);
#undef LR_
#undef LI_
#undef SP_
}

// ====== L = 45 ======
static void km45_y(const double* restrict re, const double* restrict im,
                   double* restrict dre, double* restrict dim, MK m)
{
#define LR_(j) LD(re + (j))
#define LI_(j) LD(im + (j))
#define SP_(k,vr,vi) do { MST(dre + (k), m, vr); MST(dim + (k), m, vi); } while(0)
    KM45_BODY(LR_, LI_, SP_, P45IN_S45, P45OUT_S45);
#undef LR_
#undef LI_
#undef SP_
}
static void km45_s(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + (j))
#define LI_(j) LD(im + (j))
#define SP_(k,vr,vi) do { ST(re + (k), vr); ST(im + (k), vi); } while(0)
    KM45_BODY(LR_, LI_, SP_, P45IN_S8, P45OUT_S8);
#undef LR_
#undef LI_
#undef SP_
}

// ====== L = 64 ======
static void km64_y(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 64*(j))
#define LI_(j) LD(im + 64*(j))
#define SP_(k,vr,vi) do { ST(re + 64*(k), vr); ST(im + 64*(k), vi); } while(0)
    KM64_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}

// z-pass for a whole 64x64 slab: rows of 64, tables hoisted.
static void zslab64(double* restrict re, double* restrict im)
{
    const VD Vr2=BC(g_r2half);
    VD Tzr[8], Tzi[8];
    for (int t = 1; t < 8; t++) { Tzr[t]=LD(g_T64zr[t]); Tzi[t]=LD(g_T64zi[t]); }
    for (int y = 0; y < 64; y++) {
        double *rr = re + 64*y, *ii = im + 64*y;
        VD xr0=LD(rr+0),  xi0=LD(ii+0),  xr1=LD(rr+8),  xi1=LD(ii+8);
        VD xr2=LD(rr+16), xi2=LD(ii+16), xr3=LD(rr+24), xi3=LD(ii+24);
        VD xr4=LD(rr+32), xi4=LD(ii+32), xr5=LD(rr+40), xi5=LD(ii+40);
        VD xr6=LD(rr+48), xi6=LD(ii+48), xr7=LD(rr+56), xi7=LD(ii+56);
        VD F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i;
        DFT8V(F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i,
              xr0,xi0,xr1,xi1,xr2,xi2,xr3,xi3,xr4,xi4,xr5,xi5,xr6,xi6,xr7,xi7);
        VD G1r,G1i,G2r,G2i,G3r,G3i,G4r,G4i,G5r,G5i,G6r,G6i,G7r,G7i;
        CMULV(G1r,G1i,F1r,F1i,Tzr[1],Tzi[1]);
        CMULV(G2r,G2i,F2r,F2i,Tzr[2],Tzi[2]);
        CMULV(G3r,G3i,F3r,F3i,Tzr[3],Tzi[3]);
        CMULV(G4r,G4i,F4r,F4i,Tzr[4],Tzi[4]);
        CMULV(G5r,G5i,F5r,F5i,Tzr[5],Tzi[5]);
        CMULV(G6r,G6i,F6r,F6i,Tzr[6],Tzi[6]);
        CMULV(G7r,G7i,F7r,F7i,Tzr[7],Tzi[7]);
        VD t0r,t1r,t2r,t3r,t4r,t5r,t6r,t7r, t0i,t1i,t2i,t3i,t4i,t5i,t6i,t7i;
        TR8(t0r,t1r,t2r,t3r,t4r,t5r,t6r,t7r, F0r,G1r,G2r,G3r,G4r,G5r,G6r,G7r);
        TR8(t0i,t1i,t2i,t3i,t4i,t5i,t6i,t7i, F0i,G1i,G2i,G3i,G4i,G5i,G6i,G7i);
        VD H0r,H0i,H1r,H1i,H2r,H2i,H3r,H3i,H4r,H4i,H5r,H5i,H6r,H6i,H7r,H7i;
        DFT8V(H0r,H0i,H1r,H1i,H2r,H2i,H3r,H3i,H4r,H4i,H5r,H5i,H6r,H6i,H7r,H7i,
              t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i,t4r,t4i,t5r,t5i,t6r,t6i,t7r,t7i);
        ST(rr+0,H0r);  ST(ii+0,H0i);  ST(rr+8,H1r);  ST(ii+8,H1i);
        ST(rr+16,H2r); ST(ii+16,H2i); ST(rr+24,H3r); ST(ii+24,H3i);
        ST(rr+32,H4r); ST(ii+32,H4i); ST(rr+40,H5r); ST(ii+40,H5i);
        ST(rr+48,H6r); ST(ii+48,H6i); ST(rr+56,H7r); ST(ii+56,H7i);
    }
}

// z-row DFT-64 on one contiguous row (64 complex), six-step with one transpose.
static void zrow64(double* restrict re, double* restrict im)
{
    const VD Vr2=BC(g_r2half);
    VD xr0=LD(re+0),  xi0=LD(im+0),  xr1=LD(re+8),  xi1=LD(im+8);
    VD xr2=LD(re+16), xi2=LD(im+16), xr3=LD(re+24), xi3=LD(im+24);
    VD xr4=LD(re+32), xi4=LD(im+32), xr5=LD(re+40), xi5=LD(im+40);
    VD xr6=LD(re+48), xi6=LD(im+48), xr7=LD(re+56), xi7=LD(im+56);
    VD F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i;
    DFT8V(F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i,
          xr0,xi0,xr1,xi1,xr2,xi2,xr3,xi3,xr4,xi4,xr5,xi5,xr6,xi6,xr7,xi7);
    VD G1r,G1i,G2r,G2i,G3r,G3i,G4r,G4i,G5r,G5i,G6r,G6i,G7r,G7i;
    CMULV(G1r,G1i,F1r,F1i,LD(g_T64zr[1]),LD(g_T64zi[1]));
    CMULV(G2r,G2i,F2r,F2i,LD(g_T64zr[2]),LD(g_T64zi[2]));
    CMULV(G3r,G3i,F3r,F3i,LD(g_T64zr[3]),LD(g_T64zi[3]));
    CMULV(G4r,G4i,F4r,F4i,LD(g_T64zr[4]),LD(g_T64zi[4]));
    CMULV(G5r,G5i,F5r,F5i,LD(g_T64zr[5]),LD(g_T64zi[5]));
    CMULV(G6r,G6i,F6r,F6i,LD(g_T64zr[6]),LD(g_T64zi[6]));
    CMULV(G7r,G7i,F7r,F7i,LD(g_T64zr[7]),LD(g_T64zi[7]));
    VD t0r,t1r,t2r,t3r,t4r,t5r,t6r,t7r, t0i,t1i,t2i,t3i,t4i,t5i,t6i,t7i;
    TR8(t0r,t1r,t2r,t3r,t4r,t5r,t6r,t7r, F0r,G1r,G2r,G3r,G4r,G5r,G6r,G7r);
    TR8(t0i,t1i,t2i,t3i,t4i,t5i,t6i,t7i, F0i,G1i,G2i,G3i,G4i,G5i,G6i,G7i);
    VD H0r,H0i,H1r,H1i,H2r,H2i,H3r,H3i,H4r,H4i,H5r,H5i,H6r,H6i,H7r,H7i;
    DFT8V(H0r,H0i,H1r,H1i,H2r,H2i,H3r,H3i,H4r,H4i,H5r,H5i,H6r,H6i,H7r,H7i,
          t0r,t0i,t1r,t1i,t2r,t2i,t3r,t3i,t4r,t4i,t5r,t5i,t6r,t6i,t7r,t7i);
    ST(re+0,H0r);  ST(im+0,H0i);  ST(re+8,H1r);  ST(im+8,H1i);
    ST(re+16,H2r); ST(im+16,H2i); ST(re+24,H3r); ST(im+24,H3i);
    ST(re+32,H4r); ST(im+32,H4i); ST(re+40,H5r); ST(im+40,H5i);
    ST(re+48,H6r); ST(im+48,H6i); ST(re+56,H7r); ST(im+56,H7i);
}

// ---------------------------------------------------------------------------
// sandwich z-pass for 36/45 (OOP): read src slab, write dst slab
// ---------------------------------------------------------------------------
static void spass_gen(const double* restrict sre, const double* restrict sim,
                      double* restrict dstre, double* restrict dstim, int L,
                      int nzc, const MK* zmask, void (*kern)(double*,double*))
{
    VD TMPr[64+8] __attribute__((aligned(64)));
    VD TMPi[64+8] __attribute__((aligned(64)));
    for (int y0 = 0; y0 < L; y0 += 8) {
        int R = L - y0; if (R > 8) R = 8;
        for (int zc = 0; zc < nzc; zc++) {
            VD r0,r1,r2,r3,r4,r5,r6,r7;
            const double *b0 = sre + (size_t)y0*L + 8*zc;
            r0 =           LD(b0 + 0*L);
            r1 = (R > 1) ? LD(b0 + 1*L) : ZERO;
            r2 = (R > 2) ? LD(b0 + 2*L) : ZERO;
            r3 = (R > 3) ? LD(b0 + 3*L) : ZERO;
            r4 = (R > 4) ? LD(b0 + 4*L) : ZERO;
            r5 = (R > 5) ? LD(b0 + 5*L) : ZERO;
            r6 = (R > 6) ? LD(b0 + 6*L) : ZERO;
            r7 = (R > 7) ? LD(b0 + 7*L) : ZERO;
            VD o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(o0,o1,o2,o3,o4,o5,o6,o7, r0,r1,r2,r3,r4,r5,r6,r7);
            TMPr[8*zc+0]=o0; TMPr[8*zc+1]=o1; TMPr[8*zc+2]=o2; TMPr[8*zc+3]=o3;
            TMPr[8*zc+4]=o4; TMPr[8*zc+5]=o5; TMPr[8*zc+6]=o6; TMPr[8*zc+7]=o7;
            const double *b1 = sim + (size_t)y0*L + 8*zc;
            r0 =           LD(b1 + 0*L);
            r1 = (R > 1) ? LD(b1 + 1*L) : ZERO;
            r2 = (R > 2) ? LD(b1 + 2*L) : ZERO;
            r3 = (R > 3) ? LD(b1 + 3*L) : ZERO;
            r4 = (R > 4) ? LD(b1 + 4*L) : ZERO;
            r5 = (R > 5) ? LD(b1 + 5*L) : ZERO;
            r6 = (R > 6) ? LD(b1 + 6*L) : ZERO;
            r7 = (R > 7) ? LD(b1 + 7*L) : ZERO;
            TR8(o0,o1,o2,o3,o4,o5,o6,o7, r0,r1,r2,r3,r4,r5,r6,r7);
            TMPi[8*zc+0]=o0; TMPi[8*zc+1]=o1; TMPi[8*zc+2]=o2; TMPi[8*zc+3]=o3;
            TMPi[8*zc+4]=o4; TMPi[8*zc+5]=o5; TMPi[8*zc+6]=o6; TMPi[8*zc+7]=o7;
        }
        kern((double*)TMPr, (double*)TMPi);
        for (int zc = 0; zc < nzc; zc++) {
            MK zm = zmask[zc];
            VD o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(o0,o1,o2,o3,o4,o5,o6,o7,
                TMPr[8*zc+0],TMPr[8*zc+1],TMPr[8*zc+2],TMPr[8*zc+3],
                TMPr[8*zc+4],TMPr[8*zc+5],TMPr[8*zc+6],TMPr[8*zc+7]);
            double *b0 = dstre + (size_t)y0*L + 8*zc;
            MST(b0 + 0*L, zm, o0);
            if (R > 1) MST(b0 + 1*L, zm, o1);
            if (R > 2) MST(b0 + 2*L, zm, o2);
            if (R > 3) MST(b0 + 3*L, zm, o3);
            if (R > 4) MST(b0 + 4*L, zm, o4);
            if (R > 5) MST(b0 + 5*L, zm, o5);
            if (R > 6) MST(b0 + 6*L, zm, o6);
            if (R > 7) MST(b0 + 7*L, zm, o7);
            TR8(o0,o1,o2,o3,o4,o5,o6,o7,
                TMPi[8*zc+0],TMPi[8*zc+1],TMPi[8*zc+2],TMPi[8*zc+3],
                TMPi[8*zc+4],TMPi[8*zc+5],TMPi[8*zc+6],TMPi[8*zc+7]);
            double *b1 = dstim + (size_t)y0*L + 8*zc;
            MST(b1 + 0*L, zm, o0);
            if (R > 1) MST(b1 + 1*L, zm, o1);
            if (R > 2) MST(b1 + 2*L, zm, o2);
            if (R > 3) MST(b1 + 3*L, zm, o3);
            if (R > 4) MST(b1 + 4*L, zm, o4);
            if (R > 5) MST(b1 + 5*L, zm, o5);
            if (R > 6) MST(b1 + 6*L, zm, o6);
            if (R > 7) MST(b1 + 7*L, zm, o7);
        }
    }
}

// ---- in-place x+map variants ----
static void km13_x_map_ip(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim, MK m)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
#define LR_(j) LD(re + 169*(j))
#define LI_(j) LD(im + 169*(j))
#define SP_(k,vr,vi) MAPST(re + 169*(k), im + 169*(k), m, vr, vi, cre + 169*(k), cim + 169*(k))
    KMFOLD_BODY(13, 6, g_C13, g_S13, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void km17_x_map_ip(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim, MK m)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
#define LR_(j) LD(re + 289*(j))
#define LI_(j) LD(im + 289*(j))
#define SP_(k,vr,vi) MAPST(re + 289*(k), im + 289*(k), m, vr, vi, cre + 289*(k), cim + 289*(k))
    KMFOLD_BODY(17, 8, g_C17, g_S17, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void km23_x_map_ip(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim, MK m)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
#define LR_(j) LD(re + 529*(j))
#define LI_(j) LD(im + 529*(j))
#define SP_(k,vr,vi) MAPST(re + 529*(k), im + 529*(k), m, vr, vi, cre + 529*(k), cim + 529*(k))
    KMFOLD_BODY(23, 11, g_C23, g_S23, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void km36_x_map_ip(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308);
    const MK m = 0xFF;
    _Pragma("GCC unroll 36")
    for (int pj = 0; pj < 36; pj++) {
        _mm_prefetch((const char*)(re + 1296*pj + 8), _MM_HINT_T0);
        _mm_prefetch((const char*)(im + 1296*pj + 8), _MM_HINT_T0);
        _mm_prefetch((const char*)(cre + 1296*pj + 8), _MM_HINT_T0);
        _mm_prefetch((const char*)(cim + 1296*pj + 8), _MM_HINT_T0);
    }
#define LR_(j) LD(re + (j))
#define LI_(j) LD(im + (j))
#define SP_(k,vr,vi) MAPST(re + (k), im + (k), m, vr, vi, cre + (k), cim + (k))
    KM36_BODY(LR_, LI_, SP_, P36IN_S1296, P36OUT_S1296);
#undef LR_
#undef LI_
#undef SP_
}
static void km45_x_map_ip(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim, MK m)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308);
    _Pragma("GCC unroll 45")
    for (int pj = 0; pj < 45; pj++) {
        _mm_prefetch((const char*)(re + 2025*pj + 8), _MM_HINT_T0);
        _mm_prefetch((const char*)(im + 2025*pj + 8), _MM_HINT_T0);
        _mm_prefetch((const char*)(cre + 2025*pj + 8), _MM_HINT_T0);
        _mm_prefetch((const char*)(cim + 2025*pj + 8), _MM_HINT_T0);
    }
#define LR_(j) LD(re + (j))
#define LI_(j) LD(im + (j))
#define SP_(k,vr,vi) MAPST(re + (k), im + (k), m, vr, vi, cre + (k), cim + (k))
    KM45_BODY(LR_, LI_, SP_, P45IN_S2025, P45OUT_S2025);
#undef LR_
#undef LI_
#undef SP_
}
static double W2re_[12192 + 64] __attribute__((aligned(64)));
static double W2im_[12192 + 64] __attribute__((aligned(64)));
static double Wre_[91152 + 64] __attribute__((aligned(64)));
static double Wim_[91152 + 64] __attribute__((aligned(64)));
// ---------------------------------------------------------------------------
// import / export between interleaved complex128 and SoA
// ---------------------------------------------------------------------------
static void soa_import(double* restrict re, double* restrict im, const double* restrict src, long n)
{
    const __m512i idxe = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    const __m512i idxo = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    long i = 0;
    for (; i + 8 <= n; i += 8) {
        VD v0 = LD(src + 2*i), v1 = LD(src + 2*i + 8);
        ST(re + i, _mm512_permutex2var_pd(v0, idxe, v1));
        ST(im + i, _mm512_permutex2var_pd(v0, idxo, v1));
    }
    for (; i < n; i++) { re[i] = src[2*i]; im[i] = src[2*i+1]; }
}
static void soa_export(const double* restrict re, const double* restrict im, double* restrict dst, long n)
{
    const __m512i idxl = _mm512_set_epi64(11,3,10,2,9,1,8,0);
    const __m512i idxh = _mm512_set_epi64(15,7,14,6,13,5,12,4);
    long i = 0;
    if ((((uintptr_t)dst) & 63) == 0) {
        for (; i + 8 <= n; i += 8) {
            VD r = LD(re + i), m = LD(im + i);
            _mm512_stream_pd(dst + 2*i,     _mm512_permutex2var_pd(r, idxl, m));
            _mm512_stream_pd(dst + 2*i + 8, _mm512_permutex2var_pd(r, idxh, m));
        }
        _mm_sfence();
    } else {
        for (; i + 8 <= n; i += 8) {
            VD r = LD(re + i), m = LD(im + i);
            ST(dst + 2*i,     _mm512_permutex2var_pd(r, idxl, m));
            ST(dst + 2*i + 8, _mm512_permutex2var_pd(r, idxh, m));
        }
    }
    for (; i < n; i++) { dst[2*i] = re[i]; dst[2*i+1] = im[i]; }
}



#define MAPST_NR(pr, pi, mm, yr, yi, pcr, pci) do { \
    VD _zr = VADD(yr, LD(pcr)), _zi = VADD(yi, LD(pci)); \
    VD _s  = VFMA(_zi,_zi, VMUL(_zr,_zr)); \
    _s = VMAX(_s, Vtiny); \
    VD _t = _mm512_rsqrt14_pd(_s); \
    VD _hs = VMUL(Vhalf,_s); \
    _t = VMUL(_t, VFNMA(_hs, VMUL(_t,_t), V1p5)); \
    _t = VMUL(_t, VFNMA(_hs, VMUL(_t,_t), V1p5)); \
    VD _d = VFMA(_s,_t,Vone); \
    VD _q = _mm512_rcp14_pd(_d); \
    _q = VFMA(_q, VFNMA(_d,_q,Vone), _q); \
    _q = VFMA(_q, VFNMA(_d,_q,Vone), _q); \
    MST(pr, mm, VMUL(_zr,_q)); MST(pi, mm, VMUL(_zi,_q)); } while(0)
// ---------------------------------------------------------------------------
// Volume-lane (VL) path for L in {6, 8, 13}: 8 volumes in SIMD lanes.
// Layout: X[i*8 + lane], i = flat element index, lane = volume in group.
// All three axis passes use the same fold/PFA bodies with strides 8, 8L, 8L^2.
// ---------------------------------------------------------------------------
static void vl_import(double* restrict Xre, double* restrict Xim,
                      const double* restrict src, long n, int nl)
{
    // src: nl (<=8) volumes interleaved complex (vol-major), n elements each
    const __m512i idxe = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    const __m512i idxo = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    long i0 = 0;
    for (; i0 + 8 <= n; i0 += 8) {
        VD rr[8], ri[8], o0,o1,o2,o3,o4,o5,o6,o7;
        for (int l = nl; l < 8; l++) { rr[l] = ZERO; ri[l] = ZERO; }
        for (int l = 0; l < nl; l++) {
            VD v0 = LD(src + 2*((size_t)l*n + i0));
            VD v1 = LD(src + 2*((size_t)l*n + i0) + 8);
            rr[l] = _mm512_permutex2var_pd(v0, idxe, v1);
            ri[l] = _mm512_permutex2var_pd(v0, idxo, v1);
        }
        TR8(o0,o1,o2,o3,o4,o5,o6,o7, rr[0],rr[1],rr[2],rr[3],rr[4],rr[5],rr[6],rr[7]);
        ST(Xre+(i0+0)*8,o0); ST(Xre+(i0+1)*8,o1); ST(Xre+(i0+2)*8,o2); ST(Xre+(i0+3)*8,o3);
        ST(Xre+(i0+4)*8,o4); ST(Xre+(i0+5)*8,o5); ST(Xre+(i0+6)*8,o6); ST(Xre+(i0+7)*8,o7);
        TR8(o0,o1,o2,o3,o4,o5,o6,o7, ri[0],ri[1],ri[2],ri[3],ri[4],ri[5],ri[6],ri[7]);
        ST(Xim+(i0+0)*8,o0); ST(Xim+(i0+1)*8,o1); ST(Xim+(i0+2)*8,o2); ST(Xim+(i0+3)*8,o3);
        ST(Xim+(i0+4)*8,o4); ST(Xim+(i0+5)*8,o5); ST(Xim+(i0+6)*8,o6); ST(Xim+(i0+7)*8,o7);
    }
    for (; i0 < n; i0++)
        for (int l = 0; l < 8; l++) {
            Xre[i0*8 + l] = (l < nl) ? src[2*((size_t)l*n + i0)] : 0.0;
            Xim[i0*8 + l] = (l < nl) ? src[2*((size_t)l*n + i0) + 1] : 0.0;
        }
}
static void vl_export(const double* restrict Xre, const double* restrict Xim,
                      double* restrict dst, long n, int nl)
{
    const __m512i idxl = _mm512_set_epi64(11,3,10,2,9,1,8,0);
    const __m512i idxh = _mm512_set_epi64(15,7,14,6,13,5,12,4);
    long i0 = 0;
    for (; i0 + 8 <= n; i0 += 8) {
        VD o0,o1,o2,o3,o4,o5,o6,o7, p0,p1,p2,p3,p4,p5,p6,p7;
        TR8(o0,o1,o2,o3,o4,o5,o6,o7,
            LD(Xre+(i0+0)*8),LD(Xre+(i0+1)*8),LD(Xre+(i0+2)*8),LD(Xre+(i0+3)*8),
            LD(Xre+(i0+4)*8),LD(Xre+(i0+5)*8),LD(Xre+(i0+6)*8),LD(Xre+(i0+7)*8));
        TR8(p0,p1,p2,p3,p4,p5,p6,p7,
            LD(Xim+(i0+0)*8),LD(Xim+(i0+1)*8),LD(Xim+(i0+2)*8),LD(Xim+(i0+3)*8),
            LD(Xim+(i0+4)*8),LD(Xim+(i0+5)*8),LD(Xim+(i0+6)*8),LD(Xim+(i0+7)*8));
#define EXP1(l, vr, vi) do { \
        ST(dst + 2*((size_t)(l)*n + i0),     _mm512_permutex2var_pd(vr, idxl, vi)); \
        ST(dst + 2*((size_t)(l)*n + i0) + 8, _mm512_permutex2var_pd(vr, idxh, vi)); } while(0)
        VD vr_[8] = {o0,o1,o2,o3,o4,o5,o6,o7}, vi_[8] = {p0,p1,p2,p3,p4,p5,p6,p7};
        for (int l = 0; l < nl; l++) EXP1(l, vr_[l], vi_[l]);
#undef EXP1
    }
    for (; i0 < n; i0++)
        for (int l = 0; l < nl; l++) {
            dst[2*((size_t)l*n + i0)]     = Xre[i0*8 + l];
            dst[2*((size_t)l*n + i0) + 1] = Xim[i0*8 + l];
        }
}

static void vlz6(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 8*(j))
#define LI_(j) LD(im + 8*(j))
#define SP_(k,vr,vi) do { ST(re + 8*(k), vr); ST(im + 8*(k), vi); } while(0)
    KM6_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vly6(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 48*(j))
#define LI_(j) LD(im + 48*(j))
#define SP_(k,vr,vi) do { ST(re + 48*(k), vr); ST(im + 48*(k), vi); } while(0)
    KM6_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vlx6_map(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308);
    const MK mf = 0xFF;
#define LR_(j) LD(re + 288*(j))
#define LI_(j) LD(im + 288*(j))
#define SP_(k,vr,vi) MAPST(re + 288*(k), im + 288*(k), mf, vr, vi, cre + 288*(k), cim + 288*(k))
    KM6_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}

static void vlz8(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 8*(j))
#define LI_(j) LD(im + 8*(j))
#define SP_(k,vr,vi) do { ST(re + 8*(k), vr); ST(im + 8*(k), vi); } while(0)
    KM8_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vly8(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 64*(j))
#define LI_(j) LD(im + 64*(j))
#define SP_(k,vr,vi) do { ST(re + 64*(k), vr); ST(im + 64*(k), vi); } while(0)
    KM8_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vlx8_map(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
    const MK mf = 0xFF;
#define LR_(j) LD(re + 512*(j))
#define LI_(j) LD(im + 512*(j))
#define SP_(k,vr,vi) MAPST(re + 512*(k), im + 512*(k), mf, vr, vi, cre + 512*(k), cim + 512*(k))
    KM8_BODY(LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}

static void vlz13(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 8*(j))
#define LI_(j) LD(im + 8*(j))
#define SP_(k,vr,vi) do { ST(re + 8*(k), vr); ST(im + 8*(k), vi); } while(0)
    KMFOLD_BODY(13, 6, g_C13, g_S13, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vly13(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 104*(j))
#define LI_(j) LD(im + 104*(j))
#define SP_(k,vr,vi) do { ST(re + 104*(k), vr); ST(im + 104*(k), vi); } while(0)
    KMFOLD_BODY(13, 6, g_C13, g_S13, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vlx13_map(double* restrict re, double* restrict im,
                      const double* restrict cre, const double* restrict cim)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
    const MK mf = 0xFF;
#define LR_(j) LD(re + 1352*(j))
#define LI_(j) LD(im + 1352*(j))
#define SP_(k,vr,vi) MAPST(re + 1352*(k), im + 1352*(k), mf, vr, vi, cre + 1352*(k), cim + 1352*(k))
    KMFOLD_BODY(13, 6, g_C13, g_S13, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}

static void vliter6(double* restrict Xr, double* restrict Xi,
                    const double* restrict Cr, const double* restrict Ci)
{
    for (int x = 0; x < 6; x++) {
        for (int y = 0; y < 6; y++)
            vlz6(Xr + (x*36 + y*6)*8, Xi + (x*36 + y*6)*8);
        for (int z = 0; z < 6; z++)
            vly6(Xr + (x*36 + z)*8, Xi + (x*36 + z)*8);
    }
    for (int p = 0; p < 36; p++)
        vlx6_map(Xr + 8*p, Xi + 8*p, Cr + 8*p, Ci + 8*p);
}
static void vliter8(double* restrict Xr, double* restrict Xi,
                    const double* restrict Cr, const double* restrict Ci)
{
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++)
            vlz8(Xr + (x*64 + y*8)*8, Xi + (x*64 + y*8)*8);
        for (int z = 0; z < 8; z++)
            vly8(Xr + (x*64 + z)*8, Xi + (x*64 + z)*8);
    }
    for (int p = 0; p < 64; p++)
        vlx8_map(Xr + 8*p, Xi + 8*p, Cr + 8*p, Ci + 8*p);
}
static void vliter13(double* restrict Xr, double* restrict Xi,
                     const double* restrict Cr, const double* restrict Ci)
{
    for (int x = 0; x < 13; x++) {
        for (int y = 0; y < 13; y++)
            vlz13(Xr + (x*169 + y*13)*8, Xi + (x*169 + y*13)*8);
        for (int z = 0; z < 13; z++)
            vly13(Xr + (x*169 + z)*8, Xi + (x*169 + z)*8);
    }
    for (int p = 0; p < 169; p++)
        vlx13_map(Xr + 8*p, Xi + 8*p, Cr + 8*p, Ci + 8*p);
}

static void vlz17(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 8*(j))
#define LI_(j) LD(im + 8*(j))
#define SP_(k,vr,vi) do { ST(re + 8*(k), vr); ST(im + 8*(k), vi); } while(0)
    KMFOLD_BODY(17, 8, g_C17, g_S17, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vly17(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 136*(j))
#define LI_(j) LD(im + 136*(j))
#define SP_(k,vr,vi) do { ST(re + 136*(k), vr); ST(im + 136*(k), vi); } while(0)
    KMFOLD_BODY(17, 8, g_C17, g_S17, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vlx17_map(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
    const MK mf = 0xFF;
#define LR_(j) LD(re + 2312*(j))
#define LI_(j) LD(im + 2312*(j))
#define SP_(k,vr,vi) MAPST(re + 2312*(k), im + 2312*(k), mf, vr, vi, cre + 2312*(k), cim + 2312*(k))
    KMFOLD_BODY(17, 8, g_C17, g_S17, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vliter17(double* restrict Xr, double* restrict Xi,
                      const double* restrict Cr, const double* restrict Ci)
{
    for (int x = 0; x < 17; x++) {
        for (int y = 0; y < 17; y++)
            vlz17(Xr + (x*289 + y*17)*8, Xi + (x*289 + y*17)*8);
        for (int z = 0; z < 17; z++)
            vly17(Xr + (x*289 + z)*8, Xi + (x*289 + z)*8);
    }
    for (int p = 0; p < 289; p++)
        vlx17_map(Xr + 8*p, Xi + 8*p, Cr + 8*p, Ci + 8*p);
}

static void vlz23(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 8*(j))
#define LI_(j) LD(im + 8*(j))
#define SP_(k,vr,vi) do { ST(re + 8*(k), vr); ST(im + 8*(k), vi); } while(0)
    KMFOLD_BODY(23, 11, g_C23, g_S23, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vly23(double* restrict re, double* restrict im)
{
#define LR_(j) LD(re + 184*(j))
#define LI_(j) LD(im + 184*(j))
#define SP_(k,vr,vi) do { ST(re + 184*(k), vr); ST(im + 184*(k), vi); } while(0)
    KMFOLD_BODY(23, 11, g_C23, g_S23, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vlx23_map(double* restrict re, double* restrict im,
                       const double* restrict cre, const double* restrict cim)
{
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
    const MK mf = 0xFF;
#define LR_(j) LD(re + 4232*(j))
#define LI_(j) LD(im + 4232*(j))
#define SP_(k,vr,vi) MAPST(re + 4232*(k), im + 4232*(k), mf, vr, vi, cre + 4232*(k), cim + 4232*(k))
    KMFOLD_BODY(23, 11, g_C23, g_S23, LR_, LI_, SP_);
#undef LR_
#undef LI_
#undef SP_
}
static void vliter23(double* restrict Xr, double* restrict Xi,
                      const double* restrict Cr, const double* restrict Ci)
{
    for (int x = 0; x < 23; x++) {
        for (int y = 0; y < 23; y++)
            vlz23(Xr + (x*529 + y*23)*8, Xi + (x*529 + y*23)*8);
        for (int z = 0; z < 23; z++)
            vly23(Xr + (x*529 + z)*8, Xi + (x*529 + z)*8);
    }
    for (int p = 0; p < 529; p++)
        vlx23_map(Xr + 8*p, Xi + 8*p, Cr + 8*p, Ci + 8*p);
}
// run a group of 8 volumes; x0g/cg point at the first of the 8 volumes
static void vl_chain(int L, long m, const double* x0g, const double* cg,
                     double* out1g, double* outmg, int nl)
{
    long n = (long)L*L*L;
    size_t bytes = (((size_t)n*8*2 + 64) * sizeof(double) + 63) & ~(size_t)63;
    static double *VX = 0, *VC = 0; static size_t vcap = 0;
    if (bytes > vcap) {
        if (VX) { free(VX); free(VC); }
        VX = (double*)aligned_alloc(64, bytes);
        VC = (double*)aligned_alloc(64, bytes);
        vcap = bytes;
    }
    double *Xr = VX, *Xi = VX + n*8, *Cr = VC, *Ci = VC + n*8;
    vl_import(Xr, Xi, x0g, n, nl);
    vl_import(Cr, Ci, cg, n, nl);
    void (*vit)(double*, double*, const double*, const double*) =
        (L == 6) ? vliter6 : (L == 8) ? vliter8 : (L == 13) ? vliter13 :
        (L == 17) ? vliter17 : vliter23;
    for (long it = 0; it < m; it++) {
        vit(Xr, Xi, Cr, Ci);
        if (it == 0) vl_export(Xr, Xi, out1g, n, nl);
    }
    vl_export(Xr, Xi, outmg, n, nl);
}
#pragma GCC push_options
#pragma GCC optimize("O3")
// ---------------------------------------------------------------------------
// L=64 fused pipeline with copy/compute decoupling.
//   x-DFT split as DFT8(a) [stageA w/ twiddle] and DFT8(b) [stageB], x = 8a+b.
//   Per t-step: T(b,t) --copy--> SBT; DFT8_b: SBT->SB; y,z DFTs on SB;
//   c --copy--> SBT; map+DFT8_a+twiddle: SB(+c) -> SB; SB --copy--> T'(t,*).
// ---------------------------------------------------------------------------
#define SBSTRIDE 4104
static double T64re[2*262144 + 16] __attribute__((aligned(64)));
#define T64im (T64re + 262144)
static double SBre[8*SBSTRIDE + 8] __attribute__((aligned(64)));
static double SBim[8*SBSTRIDE + 8] __attribute__((aligned(64)));
static double STre[8*SBSTRIDE + 8] __attribute__((aligned(64)));
static double STim[8*SBSTRIDE + 8] __attribute__((aligned(64)));

// pure streaming copy: n doubles
static inline void pcopy(double* restrict dst, const double* restrict src, long n)
{
    for (long i = 0; i < n; i += 32) {
        VD a = LD(src+i), b = LD(src+i+8), c = LD(src+i+16), d = LD(src+i+24);
        ST(dst+i, a); ST(dst+i+8, b); ST(dst+i+16, c); ST(dst+i+24, d);
    }
}

// stageA from state (prime): read state slabs x=8a+b, DFT8 over a, twiddle -> T
static void stageA_prime64(const double* restrict re, const double* restrict im,
                           double* restrict Tre, double* restrict Tim, int b)
{
    const VD Vr2=BC(g_r2half);
    // stage state slabs into SBT first (pure copy)
    for (int a = 0; a < 8; a++) {
        pcopy(STre + a*SBSTRIDE, re + ((size_t)(8*a+b))*4096, 4096);
        pcopy(STim + a*SBSTRIDE, im + ((size_t)(8*a+b))*4096, 4096);
    }
    double *tr = Tre + (size_t)b*8*4096, *ti = Tim + (size_t)b*8*4096;
    for (long q = 0; q < 4096; q += 8) {
        VD x0r=LD(STre+0*SBSTRIDE+q), x0i=LD(STim+0*SBSTRIDE+q);
        VD x1r=LD(STre+1*SBSTRIDE+q), x1i=LD(STim+1*SBSTRIDE+q);
        VD x2r=LD(STre+2*SBSTRIDE+q), x2i=LD(STim+2*SBSTRIDE+q);
        VD x3r=LD(STre+3*SBSTRIDE+q), x3i=LD(STim+3*SBSTRIDE+q);
        VD x4r=LD(STre+4*SBSTRIDE+q), x4i=LD(STim+4*SBSTRIDE+q);
        VD x5r=LD(STre+5*SBSTRIDE+q), x5i=LD(STim+5*SBSTRIDE+q);
        VD x6r=LD(STre+6*SBSTRIDE+q), x6i=LD(STim+6*SBSTRIDE+q);
        VD x7r=LD(STre+7*SBSTRIDE+q), x7i=LD(STim+7*SBSTRIDE+q);
        VD F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i;
        DFT8V(F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i,
              x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i);
        ST(tr+q, F0r); ST(ti+q, F0i);
        if (b) {
            VD gr,gi;
            CMULV(gr,gi,F1r,F1i,BC(g_T64r[b][1]),BC(g_T64i[b][1])); ST(tr+1*4096+q,gr); ST(ti+1*4096+q,gi);
            CMULV(gr,gi,F2r,F2i,BC(g_T64r[b][2]),BC(g_T64i[b][2])); ST(tr+2*4096+q,gr); ST(ti+2*4096+q,gi);
            CMULV(gr,gi,F3r,F3i,BC(g_T64r[b][3]),BC(g_T64i[b][3])); ST(tr+3*4096+q,gr); ST(ti+3*4096+q,gi);
            CMULV(gr,gi,F4r,F4i,BC(g_T64r[b][4]),BC(g_T64i[b][4])); ST(tr+4*4096+q,gr); ST(ti+4*4096+q,gi);
            CMULV(gr,gi,F5r,F5i,BC(g_T64r[b][5]),BC(g_T64i[b][5])); ST(tr+5*4096+q,gr); ST(ti+5*4096+q,gi);
            CMULV(gr,gi,F6r,F6i,BC(g_T64r[b][6]),BC(g_T64i[b][6])); ST(tr+6*4096+q,gr); ST(ti+6*4096+q,gi);
            CMULV(gr,gi,F7r,F7i,BC(g_T64r[b][7]),BC(g_T64i[b][7])); ST(tr+7*4096+q,gr); ST(ti+7*4096+q,gi);
        } else {
            ST(tr+1*4096+q,F1r); ST(ti+1*4096+q,F1i);
            ST(tr+2*4096+q,F2r); ST(ti+2*4096+q,F2i);
            ST(tr+3*4096+q,F3r); ST(ti+3*4096+q,F3i);
            ST(tr+4*4096+q,F4r); ST(ti+4*4096+q,F4i);
            ST(tr+5*4096+q,F5r); ST(ti+5*4096+q,F5i);
            ST(tr+6*4096+q,F6r); ST(ti+6*4096+q,F6i);
            ST(tr+7*4096+q,F7r); ST(ti+7*4096+q,F7i);
        }
    }
}

// stageB(t): T col -> DFT8 over b -> SB slabs; prefetches c slabs for stageA.
static void stageB64_compute(const double* restrict Tre, const double* restrict Tim,
                             int t, int par,
                             const double* restrict cre, const double* restrict cim)
{
    const VD Vr2=BC(g_r2half);
    // slot(b,t): even: b*8+t ; odd: t*8+b
    const double *p0r, *p0i; long step;
    if (!par) { p0r = Tre + (size_t)t*4096; step = 8*4096; }
    else      { p0r = Tre + (size_t)t*8*4096; step = 4096; }
    p0i = Tim + (p0r - Tre);
    for (long q = 0; q < 4096; q += 8) {
        _Pragma("GCC unroll 8")
        for (int pb = 0; pb < 8; pb++) {
            _mm_prefetch((const char*)(p0r + pb*step + q + 64), _MM_HINT_T0);
            _mm_prefetch((const char*)(p0i + pb*step + q + 64), _MM_HINT_T0);
        }
        VD x0r=LD(p0r+0*step+q), x0i=LD(p0i+0*step+q);
        VD x1r=LD(p0r+1*step+q), x1i=LD(p0i+1*step+q);
        VD x2r=LD(p0r+2*step+q), x2i=LD(p0i+2*step+q);
        VD x3r=LD(p0r+3*step+q), x3i=LD(p0i+3*step+q);
        VD x4r=LD(p0r+4*step+q), x4i=LD(p0i+4*step+q);
        VD x5r=LD(p0r+5*step+q), x5i=LD(p0i+5*step+q);
        VD x6r=LD(p0r+6*step+q), x6i=LD(p0i+6*step+q);
        VD x7r=LD(p0r+7*step+q), x7i=LD(p0i+7*step+q);
        VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i;
        DFT8V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,
              x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i);
        ST(SBre+0*SBSTRIDE+q,o0r); ST(SBim+0*SBSTRIDE+q,o0i);
        ST(SBre+1*SBSTRIDE+q,o1r); ST(SBim+1*SBSTRIDE+q,o1i);
        ST(SBre+2*SBSTRIDE+q,o2r); ST(SBim+2*SBSTRIDE+q,o2i);
        ST(SBre+3*SBSTRIDE+q,o3r); ST(SBim+3*SBSTRIDE+q,o3i);
        ST(SBre+4*SBSTRIDE+q,o4r); ST(SBim+4*SBSTRIDE+q,o4i);
        ST(SBre+5*SBSTRIDE+q,o5r); ST(SBim+5*SBSTRIDE+q,o5i);
        ST(SBre+6*SBSTRIDE+q,o6r); ST(SBim+6*SBSTRIDE+q,o6i);
        ST(SBre+7*SBSTRIDE+q,o7r); ST(SBim+7*SBSTRIDE+q,o7i);
    }
}

// stageA-next: SB + c -> map -> (optional state out, NT) -> DFT8 over a + twiddle
// -> write into T slots for next parity (the slots stageB(t) just freed).
static void stageA64_compute(double* restrict Tre, double* restrict Tim,
                             int t, int par, int noT,
                             const double* restrict cre, const double* restrict cim,
                             double* restrict outre, double* restrict outim)
{
    const VD Vr2=BC(g_r2half);
    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308), Vhalf=BC(0.5);
    double *our = outre ? outre + (size_t)t*4096 : 0, *oui = outim ? outim + (size_t)t*4096 : 0;
    const double *crs = cre + (size_t)t*4096, *cis = cim + (size_t)t*4096;
    // next parity slot(t, tp): nextpar = !par: if nextpar odd -> tp*8+t ; even -> t*8+tp
    double *t0r, *t0i; long tstep;
    if (!par) { t0r = Tre + (size_t)t*4096;   tstep = 8*4096; }  // next parity = odd: slot = tp*8+t
    else      { t0r = Tre + (size_t)t*8*4096; tstep = 4096;  }   // next parity = even: slot = t*8+tp
    t0i = Tim + (t0r - Tre);
    for (long q = 0; q < 4096; q += 8) {
        _Pragma("GCC unroll 8")
        for (int pa = 0; pa < 8; pa++) {
            _mm_prefetch((const char*)(crs + pa*8*4096 + q + 128), _MM_HINT_T0);
            _mm_prefetch((const char*)(cis + pa*8*4096 + q + 128), _MM_HINT_T0);
        }
        VD x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;
#define MAPLD(dstr, dsti, a) do { \
        VD _yr = LD(SBre+(a)*SBSTRIDE+q), _yi = LD(SBim+(a)*SBSTRIDE+q); \
        VD _zr = VADD(_yr, LD(crs+(a)*8*4096+q)), _zi = VADD(_yi, LD(cis+(a)*8*4096+q)); \
        VD _s  = VFMA(_zi,_zi, VMUL(_zr,_zr)); \
        _s = VMAX(_s, Vtiny); \
        VD _t = _mm512_rsqrt14_pd(_s); \
        VD _hs = VMUL(Vhalf,_s); \
        _t = VMUL(_t, VFNMA(_hs, VMUL(_t,_t), V1p5)); \
        _t = VMUL(_t, VFNMA(_hs, VMUL(_t,_t), V1p5)); \
        VD _d = VFMA(_s,_t,Vone); \
        VD _q = _mm512_div_pd(Vone, _d); \
        dstr = VMUL(_zr,_q); dsti = VMUL(_zi,_q); \
    } while(0)
        MAPLD(x0r,x0i,0); MAPLD(x1r,x1i,1); MAPLD(x2r,x2i,2); MAPLD(x3r,x3i,3);
        MAPLD(x4r,x4i,4); MAPLD(x5r,x5i,5); MAPLD(x6r,x6i,6); MAPLD(x7r,x7i,7);
#undef MAPLD
        if (our) {
            _mm512_stream_pd(our+0*8*4096+q, x0r); _mm512_stream_pd(oui+0*8*4096+q, x0i);
            _mm512_stream_pd(our+1*8*4096+q, x1r); _mm512_stream_pd(oui+1*8*4096+q, x1i);
            _mm512_stream_pd(our+2*8*4096+q, x2r); _mm512_stream_pd(oui+2*8*4096+q, x2i);
            _mm512_stream_pd(our+3*8*4096+q, x3r); _mm512_stream_pd(oui+3*8*4096+q, x3i);
            _mm512_stream_pd(our+4*8*4096+q, x4r); _mm512_stream_pd(oui+4*8*4096+q, x4i);
            _mm512_stream_pd(our+5*8*4096+q, x5r); _mm512_stream_pd(oui+5*8*4096+q, x5i);
            _mm512_stream_pd(our+6*8*4096+q, x6r); _mm512_stream_pd(oui+6*8*4096+q, x6i);
            _mm512_stream_pd(our+7*8*4096+q, x7r); _mm512_stream_pd(oui+7*8*4096+q, x7i);
        }
        if (noT) continue;
        VD F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i;
        DFT8V(F0r,F0i,F1r,F1i,F2r,F2i,F3r,F3i,F4r,F4i,F5r,F5i,F6r,F6i,F7r,F7i,
              x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i);
        ST(t0r+0*tstep+q, F0r); ST(t0i+0*tstep+q, F0i);
        if (t) {
            VD gr,gi;
            CMULV(gr,gi,F1r,F1i,BC(g_T64r[t][1]),BC(g_T64i[t][1])); ST(t0r+1*tstep+q,gr); ST(t0i+1*tstep+q,gi);
            CMULV(gr,gi,F2r,F2i,BC(g_T64r[t][2]),BC(g_T64i[t][2])); ST(t0r+2*tstep+q,gr); ST(t0i+2*tstep+q,gi);
            CMULV(gr,gi,F3r,F3i,BC(g_T64r[t][3]),BC(g_T64i[t][3])); ST(t0r+3*tstep+q,gr); ST(t0i+3*tstep+q,gi);
            CMULV(gr,gi,F4r,F4i,BC(g_T64r[t][4]),BC(g_T64i[t][4])); ST(t0r+4*tstep+q,gr); ST(t0i+4*tstep+q,gi);
            CMULV(gr,gi,F5r,F5i,BC(g_T64r[t][5]),BC(g_T64i[t][5])); ST(t0r+5*tstep+q,gr); ST(t0i+5*tstep+q,gi);
            CMULV(gr,gi,F6r,F6i,BC(g_T64r[t][6]),BC(g_T64i[t][6])); ST(t0r+6*tstep+q,gr); ST(t0i+6*tstep+q,gi);
            CMULV(gr,gi,F7r,F7i,BC(g_T64r[t][7]),BC(g_T64i[t][7])); ST(t0r+7*tstep+q,gr); ST(t0i+7*tstep+q,gi);
        } else {
            ST(t0r+1*tstep+q,F1r); ST(t0i+1*tstep+q,F1i);
            ST(t0r+2*tstep+q,F2r); ST(t0i+2*tstep+q,F2i);
            ST(t0r+3*tstep+q,F3r); ST(t0i+3*tstep+q,F3i);
            ST(t0r+4*tstep+q,F4r); ST(t0i+4*tstep+q,F4i);
            ST(t0r+5*tstep+q,F5r); ST(t0i+5*tstep+q,F5i);
            ST(t0r+6*tstep+q,F6r); ST(t0i+6*tstep+q,F6i);
            ST(t0r+7*tstep+q,F7r); ST(t0i+7*tstep+q,F7i);
        }
    }
}

static void chain64(double* restrict re, double* restrict im,
                    const double* restrict cre, const double* restrict cim,
                    long m, double* restrict Tre, double* restrict Tim,
                    double* restrict unused1, double* restrict unused2,
                    double* restrict out1)
{
    (void)unused1; (void)unused2;
    for (int b = 0; b < 8; b++) stageA_prime64(re, im, Tre, Tim, b);
    for (long n = 0; n < m; n++) {
        int last = (n == m-1);
        int first = (n == 0);
        int par = (int)(n & 1);
        for (int t = 0; t < 8; t++) {
            stageB64_compute(Tre, Tim, t, par, cre, cim);
            for (int s = 0; s < 8; s++) {
                double *sr = SBre + (size_t)s*SBSTRIDE, *si = SBim + (size_t)s*SBSTRIDE;
                for (int zc = 0; zc < 8; zc++) km64_y(sr + 8*zc, si + 8*zc);
                zslab64(sr, si);
            }
            stageA64_compute(Tre, Tim, t, par, last, cre, cim,
                             (first || last) ? re : 0, (first || last) ? im : 0);
        }
        if (first || last) _mm_sfence();
        if (first && out1) soa_export(re, im, out1, 262144);
    }
}

#pragma GCC pop_options
// ---------------------------------------------------------------------------
// L=45 and L=36 alternating PFA pipelines (see derivation in comments).
// ---------------------------------------------------------------------------
#define PS45 2040
#define PS36 1304
static double T45re[45*PS45 + 8] __attribute__((aligned(64)));
static double T45im[45*PS45 + 8] __attribute__((aligned(64)));
static double T36re[36*PS36 + 8] __attribute__((aligned(64)));
static double T36im[36*PS36 + 8] __attribute__((aligned(64)));
#define SB45S 2040
#define SB36S 1304

#define PMAP(dstr, dsti, sbr, sbi, crp, cip, q) do { \
    VD _yr = LD((sbr)+(q)), _yi = LD((sbi)+(q)); \
    VD _zr = VADD(_yr, LD((crp)+(q))), _zi = VADD(_yi, LD((cip)+(q))); \
    VD _s  = VFMA(_zi,_zi, VMUL(_zr,_zr)); \
    _s = VMAX(_s, Vtiny); \
    VD _t = _mm512_rsqrt14_pd(_s); \
    VD _hs = VMUL(Vhalf,_s); \
    _t = VMUL(_t, VFNMA(_hs, VMUL(_t,_t), V1p5)); \
    _t = VMUL(_t, VFNMA(_hs, VMUL(_t,_t), V1p5)); \
    VD _d = VFMA(_s,_t,Vone); \
    VD _q = _mm512_div_pd(Vone, _d); \
    dstr = VMUL(_zr,_q); dsti = VMUL(_zi,_q); \
} while(0)
static void prime45(const double* restrict re, const double* restrict im)
{
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half);
    const VD Vw91r=BC(g_w9[0]),Vw91i=BC(g_w9[1]),Vw92r=BC(g_w9[2]),Vw92i=BC(g_w9[3]),Vw94r=BC(g_w9[4]),Vw94i=BC(g_w9[5]);
    for (int j5 = 0; j5 < 5; j5++) {
        const double *s_r[9], *s_i[9]; double *d_r[9], *d_i[9];
        for (int j9 = 0; j9 < 9; j9++) {
            int sl = (5*j9 + 9*j5) % 45;
            s_r[j9] = re + (size_t)sl*2025; s_i[j9] = im + (size_t)sl*2025;
            int slot = ((4*j5) % 5)*9 + j9;
            d_r[j9] = T45re + (size_t)slot*PS45; d_i[j9] = T45im + (size_t)slot*PS45;
        }
        for (long q = 0; q < 2025; q += 8) {
            VD y0r=LD(s_r[0]+q), y0i=LD(s_i[0]+q);
            VD y1r=LD(s_r[1]+q), y1i=LD(s_i[1]+q);
            VD y2r=LD(s_r[2]+q), y2i=LD(s_i[2]+q);
            VD y3r=LD(s_r[3]+q), y3i=LD(s_i[3]+q);
            VD y4r=LD(s_r[4]+q), y4i=LD(s_i[4]+q);
            VD y5r=LD(s_r[5]+q), y5i=LD(s_i[5]+q);
            VD y6r=LD(s_r[6]+q), y6i=LD(s_i[6]+q);
            VD y7r=LD(s_r[7]+q), y7i=LD(s_i[7]+q);
            VD y8r=LD(s_r[8]+q), y8i=LD(s_i[8]+q);
            VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i;
            DFT9V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i,
                  y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i,y8r,y8i);
            ST(d_r[0]+q,o0r); ST(d_i[0]+q,o0i);
            ST(d_r[1]+q,o1r); ST(d_i[1]+q,o1i);
            ST(d_r[2]+q,o2r); ST(d_i[2]+q,o2i);
            ST(d_r[3]+q,o3r); ST(d_i[3]+q,o3i);
            ST(d_r[4]+q,o4r); ST(d_i[4]+q,o4i);
            ST(d_r[5]+q,o5r); ST(d_i[5]+q,o5i);
            ST(d_r[6]+q,o6r); ST(d_i[6]+q,o6i);
            ST(d_r[7]+q,o7r); ST(d_i[7]+q,o7i);
            ST(d_r[8]+q,o8r); ST(d_i[8]+q,o8i);
        }
    }
}
static void yz45(double* restrict sr, double* restrict si)
{
    static const MK zm[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0x1F};
    for (int zc = 0; zc < 5; zc++) km45_y(sr + 8*zc, si + 8*zc, Wre_ + 8*zc, Wim_ + 8*zc, 0xFF);
    km45_y(sr + 40, si + 40, Wre_ + 40, Wim_ + 40, 0x1F);
    spass_gen(Wre_, Wim_, sr, si, 45, 6, zm, km45_s);
}
static void iter45_even(const double* restrict cre, const double* restrict cim, int last, double* restrict outre, double* restrict outim)
{
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half);
    const VD Vc51=BC(g_c51),Vc52=BC(g_c52),Vs51=BC(g_s51),Vs52=BC(g_s52);    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308);
    for (int k9 = 0; k9 < 9; k9++) {
        {
            const double *p_r[5], *p_i[5];
            for (int j5 = 0; j5 < 5; j5++) {
                int slot = ((4*j5) % 5)*9 + k9;
                p_r[j5] = T45re + (size_t)slot*PS45; p_i[j5] = T45im + (size_t)slot*PS45;
            }
            for (long q = 0; q < 2025; q += 8) {
            VD y0r=LD(p_r[0]+q), y0i=LD(p_i[0]+q);
            VD y1r=LD(p_r[1]+q), y1i=LD(p_i[1]+q);
            VD y2r=LD(p_r[2]+q), y2i=LD(p_i[2]+q);
            VD y3r=LD(p_r[3]+q), y3i=LD(p_i[3]+q);
            VD y4r=LD(p_r[4]+q), y4i=LD(p_i[4]+q);
            VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i;
            DFT5V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,
                  y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i);
            ST(SBre+0*SB45S+q,o0r); ST(SBim+0*SB45S+q,o0i);
            ST(SBre+1*SB45S+q,o1r); ST(SBim+1*SB45S+q,o1i);
            ST(SBre+2*SB45S+q,o2r); ST(SBim+2*SB45S+q,o2i);
            ST(SBre+3*SB45S+q,o3r); ST(SBim+3*SB45S+q,o3i);
            ST(SBre+4*SB45S+q,o4r); ST(SBim+4*SB45S+q,o4i);
            }
        }
        for (int k5 = 0; k5 < 5; k5++)
            yz45(SBre + (size_t)k5*SB45S, SBim + (size_t)k5*SB45S);
        {
            const double *c_r[5], *c_i[5], *b_r[5], *b_i[5];
            double *o_r[5], *o_i[5], *t_r[5], *t_i[5];
            for (int i = 0; i < 5; i++) {
                int k5 = (4*i) % 5;   /* input j5'=i comes from SB[k5] */
                int sl = (36*k5 + 10*k9) % 45;
                b_r[i] = SBre + (size_t)k5*SB45S; b_i[i] = SBim + (size_t)k5*SB45S;
                c_r[i] = cre + (size_t)sl*2025;   c_i[i] = cim + (size_t)sl*2025;
                o_r[i] = outre ? outre + (size_t)sl*2025 : 0;
                o_i[i] = outim ? outim + (size_t)sl*2025 : 0;
                int slot = i*9 + k9;  /* output k5'=i at slot k5'*9+k9 */
                t_r[i] = T45re + (size_t)slot*PS45; t_i[i] = T45im + (size_t)slot*PS45;
            }
            for (long q = 0; q < 2025; q += 8) {
                MK mm = (q + 8 <= 2025) ? (MK)0xFF : (MK)((1u << (2025 - q)) - 1);
                (void)mm;
                VD x0r,x0i; PMAP(x0r,x0i,b_r[0],b_i[0],c_r[0],c_i[0],q);
                VD x1r,x1i; PMAP(x1r,x1i,b_r[1],b_i[1],c_r[1],c_i[1],q);
                VD x2r,x2i; PMAP(x2r,x2i,b_r[2],b_i[2],c_r[2],c_i[2],q);
                VD x3r,x3i; PMAP(x3r,x3i,b_r[3],b_i[3],c_r[3],c_i[3],q);
                VD x4r,x4i; PMAP(x4r,x4i,b_r[4],b_i[4],c_r[4],c_i[4],q);
                if (outre) {
                    MST(o_r[0]+q,mm,x0r); MST(o_i[0]+q,mm,x0i);
                    MST(o_r[1]+q,mm,x1r); MST(o_i[1]+q,mm,x1i);
                    MST(o_r[2]+q,mm,x2r); MST(o_i[2]+q,mm,x2i);
                    MST(o_r[3]+q,mm,x3r); MST(o_i[3]+q,mm,x3i);
                    MST(o_r[4]+q,mm,x4r); MST(o_i[4]+q,mm,x4i);
                }
                if (last) continue;
                VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i;
            DFT5V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,
                      x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i);
                ST(t_r[0]+q,o0r); ST(t_i[0]+q,o0i);
                ST(t_r[1]+q,o1r); ST(t_i[1]+q,o1i);
                ST(t_r[2]+q,o2r); ST(t_i[2]+q,o2i);
                ST(t_r[3]+q,o3r); ST(t_i[3]+q,o3i);
                ST(t_r[4]+q,o4r); ST(t_i[4]+q,o4i);
            }
        }
    }
}
static void iter45_odd(const double* restrict cre, const double* restrict cim, int last, double* restrict outre, double* restrict outim)
{
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half);
    const VD Vw91r=BC(g_w9[0]),Vw91i=BC(g_w9[1]),Vw92r=BC(g_w9[2]),Vw92i=BC(g_w9[3]),Vw94r=BC(g_w9[4]),Vw94i=BC(g_w9[5]);    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308);
    for (int k5 = 0; k5 < 5; k5++) {
        {
            const double *p_r[9], *p_i[9];
            for (int j9 = 0; j9 < 9; j9++) {
                int slot = k5*9 + (5*j9) % 9;
                p_r[j9] = T45re + (size_t)slot*PS45; p_i[j9] = T45im + (size_t)slot*PS45;
            }
            for (long q = 0; q < 2025; q += 8) {
            VD y0r=LD(p_r[0]+q), y0i=LD(p_i[0]+q);
            VD y1r=LD(p_r[1]+q), y1i=LD(p_i[1]+q);
            VD y2r=LD(p_r[2]+q), y2i=LD(p_i[2]+q);
            VD y3r=LD(p_r[3]+q), y3i=LD(p_i[3]+q);
            VD y4r=LD(p_r[4]+q), y4i=LD(p_i[4]+q);
            VD y5r=LD(p_r[5]+q), y5i=LD(p_i[5]+q);
            VD y6r=LD(p_r[6]+q), y6i=LD(p_i[6]+q);
            VD y7r=LD(p_r[7]+q), y7i=LD(p_i[7]+q);
            VD y8r=LD(p_r[8]+q), y8i=LD(p_i[8]+q);
            VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i;
            DFT9V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i,
                  y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i,y8r,y8i);
            ST(SBre+0*SB45S+q,o0r); ST(SBim+0*SB45S+q,o0i);
            ST(SBre+1*SB45S+q,o1r); ST(SBim+1*SB45S+q,o1i);
            ST(SBre+2*SB45S+q,o2r); ST(SBim+2*SB45S+q,o2i);
            ST(SBre+3*SB45S+q,o3r); ST(SBim+3*SB45S+q,o3i);
            ST(SBre+4*SB45S+q,o4r); ST(SBim+4*SB45S+q,o4i);
            ST(SBre+5*SB45S+q,o5r); ST(SBim+5*SB45S+q,o5i);
            ST(SBre+6*SB45S+q,o6r); ST(SBim+6*SB45S+q,o6i);
            ST(SBre+7*SB45S+q,o7r); ST(SBim+7*SB45S+q,o7i);
            ST(SBre+8*SB45S+q,o8r); ST(SBim+8*SB45S+q,o8i);
            }
        }
        for (int k9 = 0; k9 < 9; k9++)
            yz45(SBre + (size_t)k9*SB45S, SBim + (size_t)k9*SB45S);
        {
            const double *c_r[9], *c_i[9], *b_r[9], *b_i[9];
            double *o_r[9], *o_i[9], *t_r[9], *t_i[9];
            for (int i = 0; i < 9; i++) {
                int k9 = (5*i) % 9;   /* input j9=i from SB[k9] */
                int sl = (36*k5 + 10*k9) % 45;
                b_r[i] = SBre + (size_t)k9*SB45S; b_i[i] = SBim + (size_t)k9*SB45S;
                c_r[i] = cre + (size_t)sl*2025;   c_i[i] = cim + (size_t)sl*2025;
                o_r[i] = outre ? outre + (size_t)sl*2025 : 0;
                o_i[i] = outim ? outim + (size_t)sl*2025 : 0;
                int slot = k5*9 + i;   /* output k9'=i at slot k5*9+k9' */
                t_r[i] = T45re + (size_t)slot*PS45; t_i[i] = T45im + (size_t)slot*PS45;
            }
            for (long q = 0; q < 2025; q += 8) {
                MK mm = (q + 8 <= 2025) ? (MK)0xFF : (MK)((1u << (2025 - q)) - 1);
                (void)mm;
                VD x0r,x0i; PMAP(x0r,x0i,b_r[0],b_i[0],c_r[0],c_i[0],q);
                VD x1r,x1i; PMAP(x1r,x1i,b_r[1],b_i[1],c_r[1],c_i[1],q);
                VD x2r,x2i; PMAP(x2r,x2i,b_r[2],b_i[2],c_r[2],c_i[2],q);
                VD x3r,x3i; PMAP(x3r,x3i,b_r[3],b_i[3],c_r[3],c_i[3],q);
                VD x4r,x4i; PMAP(x4r,x4i,b_r[4],b_i[4],c_r[4],c_i[4],q);
                VD x5r,x5i; PMAP(x5r,x5i,b_r[5],b_i[5],c_r[5],c_i[5],q);
                VD x6r,x6i; PMAP(x6r,x6i,b_r[6],b_i[6],c_r[6],c_i[6],q);
                VD x7r,x7i; PMAP(x7r,x7i,b_r[7],b_i[7],c_r[7],c_i[7],q);
                VD x8r,x8i; PMAP(x8r,x8i,b_r[8],b_i[8],c_r[8],c_i[8],q);
                if (outre) {
                    MST(o_r[0]+q,mm,x0r); MST(o_i[0]+q,mm,x0i);
                    MST(o_r[1]+q,mm,x1r); MST(o_i[1]+q,mm,x1i);
                    MST(o_r[2]+q,mm,x2r); MST(o_i[2]+q,mm,x2i);
                    MST(o_r[3]+q,mm,x3r); MST(o_i[3]+q,mm,x3i);
                    MST(o_r[4]+q,mm,x4r); MST(o_i[4]+q,mm,x4i);
                    MST(o_r[5]+q,mm,x5r); MST(o_i[5]+q,mm,x5i);
                    MST(o_r[6]+q,mm,x6r); MST(o_i[6]+q,mm,x6i);
                    MST(o_r[7]+q,mm,x7r); MST(o_i[7]+q,mm,x7i);
                    MST(o_r[8]+q,mm,x8r); MST(o_i[8]+q,mm,x8i);
                }
                if (last) continue;
                VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i;
            DFT9V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i,
                      x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,x8r,x8i);
                ST(t_r[0]+q,o0r); ST(t_i[0]+q,o0i);
                ST(t_r[1]+q,o1r); ST(t_i[1]+q,o1i);
                ST(t_r[2]+q,o2r); ST(t_i[2]+q,o2i);
                ST(t_r[3]+q,o3r); ST(t_i[3]+q,o3i);
                ST(t_r[4]+q,o4r); ST(t_i[4]+q,o4i);
                ST(t_r[5]+q,o5r); ST(t_i[5]+q,o5i);
                ST(t_r[6]+q,o6r); ST(t_i[6]+q,o6i);
                ST(t_r[7]+q,o7r); ST(t_i[7]+q,o7i);
                ST(t_r[8]+q,o8r); ST(t_i[8]+q,o8i);
            }
        }
    }
}
static void chain45(double* restrict re, double* restrict im,
                    const double* restrict cre, const double* restrict cim,
                    long m, double* restrict out1)
{
    prime45(re, im);
    for (long n = 0; n < m; n++) {
        int last = (n == m-1), first = (n == 0);
        double *outr = (first || last) ? re : 0, *outi = (first || last) ? im : 0;
        if (!(n & 1)) iter45_even(cre, cim, last, outr, outi);
        else          iter45_odd(cre, cim, last, outr, outi);
        if (first && out1) soa_export(re, im, out1, 91125);
    }
}
static void prime36(const double* restrict re, const double* restrict im)
{
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half);
    const VD Vw91r=BC(g_w9[0]),Vw91i=BC(g_w9[1]),Vw92r=BC(g_w9[2]),Vw92i=BC(g_w9[3]),Vw94r=BC(g_w9[4]),Vw94i=BC(g_w9[5]);
    for (int j4 = 0; j4 < 4; j4++) {
        const double *s_r[9], *s_i[9]; double *d_r[9], *d_i[9];
        for (int j9 = 0; j9 < 9; j9++) {
            int sl = (9*j4 + 4*j9) % 36;
            s_r[j9] = re + (size_t)sl*1296; s_i[j9] = im + (size_t)sl*1296;
            int slot = j4*9 + j9;
            d_r[j9] = T36re + (size_t)slot*PS36; d_i[j9] = T36im + (size_t)slot*PS36;
        }
        for (long q = 0; q < 1296; q += 8) {
            VD y0r=LD(s_r[0]+q), y0i=LD(s_i[0]+q);
            VD y1r=LD(s_r[1]+q), y1i=LD(s_i[1]+q);
            VD y2r=LD(s_r[2]+q), y2i=LD(s_i[2]+q);
            VD y3r=LD(s_r[3]+q), y3i=LD(s_i[3]+q);
            VD y4r=LD(s_r[4]+q), y4i=LD(s_i[4]+q);
            VD y5r=LD(s_r[5]+q), y5i=LD(s_i[5]+q);
            VD y6r=LD(s_r[6]+q), y6i=LD(s_i[6]+q);
            VD y7r=LD(s_r[7]+q), y7i=LD(s_i[7]+q);
            VD y8r=LD(s_r[8]+q), y8i=LD(s_i[8]+q);
            VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i;
            DFT9V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i,
                  y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i,y8r,y8i);
            ST(d_r[0]+q,o0r); ST(d_i[0]+q,o0i);
            ST(d_r[1]+q,o1r); ST(d_i[1]+q,o1i);
            ST(d_r[2]+q,o2r); ST(d_i[2]+q,o2i);
            ST(d_r[3]+q,o3r); ST(d_i[3]+q,o3i);
            ST(d_r[4]+q,o4r); ST(d_i[4]+q,o4i);
            ST(d_r[5]+q,o5r); ST(d_i[5]+q,o5i);
            ST(d_r[6]+q,o6r); ST(d_i[6]+q,o6i);
            ST(d_r[7]+q,o7r); ST(d_i[7]+q,o7i);
            ST(d_r[8]+q,o8r); ST(d_i[8]+q,o8i);
        }
    }
}
static void yz36(double* restrict sr, double* restrict si)
{
    static const MK zm[5] = {0xFF,0xFF,0xFF,0xFF,0x0F};
    for (int zc = 0; zc < 4; zc++) km36_y(sr + 8*zc, si + 8*zc, Wre_ + 8*zc, Wim_ + 8*zc, 0xFF);
    km36_y(sr + 32, si + 32, Wre_ + 32, Wim_ + 32, 0x0F);
    spass_gen(Wre_, Wim_, sr, si, 36, 5, zm, km36_s);
}
static void iter36_even(const double* restrict cre, const double* restrict cim, int last, double* restrict outre, double* restrict outim)
{
    const VD Vhalf=BC(0.5);    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308);
    for (int k9 = 0; k9 < 9; k9++) {
        {
            const double *p_r[4], *p_i[4];
            for (int j4 = 0; j4 < 4; j4++) {
                int slot = j4*9 + k9;
                p_r[j4] = T36re + (size_t)slot*PS36; p_i[j4] = T36im + (size_t)slot*PS36;
            }
            for (long q = 0; q < 1296; q += 8) {
            VD y0r=LD(p_r[0]+q), y0i=LD(p_i[0]+q);
            VD y1r=LD(p_r[1]+q), y1i=LD(p_i[1]+q);
            VD y2r=LD(p_r[2]+q), y2i=LD(p_i[2]+q);
            VD y3r=LD(p_r[3]+q), y3i=LD(p_i[3]+q);
            VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
            DFT4V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,
                  y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i);
            ST(SBre+0*SB36S+q,o0r); ST(SBim+0*SB36S+q,o0i);
            ST(SBre+1*SB36S+q,o1r); ST(SBim+1*SB36S+q,o1i);
            ST(SBre+2*SB36S+q,o2r); ST(SBim+2*SB36S+q,o2i);
            ST(SBre+3*SB36S+q,o3r); ST(SBim+3*SB36S+q,o3i);
            }
        }
        for (int k4 = 0; k4 < 4; k4++)
            yz36(SBre + (size_t)k4*SB36S, SBim + (size_t)k4*SB36S);
        {
            const double *c_r[4], *c_i[4], *b_r[4], *b_i[4];
            double *o_r[4], *o_i[4], *t_r[4], *t_i[4];
            for (int i = 0; i < 4; i++) {
                int k4 = i;  /* j4' = k4 identity */
                int sl = (9*k4 + 28*k9) % 36;
                b_r[i] = SBre + (size_t)k4*SB36S; b_i[i] = SBim + (size_t)k4*SB36S;
                c_r[i] = cre + (size_t)sl*1296;   c_i[i] = cim + (size_t)sl*1296;
                o_r[i] = outre ? outre + (size_t)sl*1296 : 0;
                o_i[i] = outim ? outim + (size_t)sl*1296 : 0;
                int slot = i*9 + k9;  /* slot1(j9',k4'=i) = i*9 + (4*j9')%9, j9'=7k9%9 -> = k9 */
                t_r[i] = T36re + (size_t)slot*PS36; t_i[i] = T36im + (size_t)slot*PS36;
            }
            for (long q = 0; q < 1296; q += 8) {
                VD x0r,x0i; PMAP(x0r,x0i,b_r[0],b_i[0],c_r[0],c_i[0],q);
                VD x1r,x1i; PMAP(x1r,x1i,b_r[1],b_i[1],c_r[1],c_i[1],q);
                VD x2r,x2i; PMAP(x2r,x2i,b_r[2],b_i[2],c_r[2],c_i[2],q);
                VD x3r,x3i; PMAP(x3r,x3i,b_r[3],b_i[3],c_r[3],c_i[3],q);
                if (outre) {
                    ST(o_r[0]+q,x0r); ST(o_i[0]+q,x0i);
                    ST(o_r[1]+q,x1r); ST(o_i[1]+q,x1i);
                    ST(o_r[2]+q,x2r); ST(o_i[2]+q,x2i);
                    ST(o_r[3]+q,x3r); ST(o_i[3]+q,x3i);
                }
                if (last) continue;
                VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i;
            DFT4V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,
                      x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i);
                ST(t_r[0]+q,o0r); ST(t_i[0]+q,o0i);
                ST(t_r[1]+q,o1r); ST(t_i[1]+q,o1i);
                ST(t_r[2]+q,o2r); ST(t_i[2]+q,o2i);
                ST(t_r[3]+q,o3r); ST(t_i[3]+q,o3i);
            }
        }
    }
}
static void iter36_odd(const double* restrict cre, const double* restrict cim, int last, double* restrict outre, double* restrict outim)
{
    const VD Vhalf=BC(0.5), Vs3=BC(g_s3half);
    const VD Vw91r=BC(g_w9[0]),Vw91i=BC(g_w9[1]),Vw92r=BC(g_w9[2]),Vw92i=BC(g_w9[3]),Vw94r=BC(g_w9[4]),Vw94i=BC(g_w9[5]);    const VD Vone=BC(1.0), V1p5=BC(1.5), Vtiny=BC(2.3e-308);
    for (int k4 = 0; k4 < 4; k4++) {
        {
            const double *p_r[9], *p_i[9];
            for (int j9 = 0; j9 < 9; j9++) {
                int slot = k4*9 + (4*j9) % 9;
                p_r[j9] = T36re + (size_t)slot*PS36; p_i[j9] = T36im + (size_t)slot*PS36;
            }
            for (long q = 0; q < 1296; q += 8) {
            VD y0r=LD(p_r[0]+q), y0i=LD(p_i[0]+q);
            VD y1r=LD(p_r[1]+q), y1i=LD(p_i[1]+q);
            VD y2r=LD(p_r[2]+q), y2i=LD(p_i[2]+q);
            VD y3r=LD(p_r[3]+q), y3i=LD(p_i[3]+q);
            VD y4r=LD(p_r[4]+q), y4i=LD(p_i[4]+q);
            VD y5r=LD(p_r[5]+q), y5i=LD(p_i[5]+q);
            VD y6r=LD(p_r[6]+q), y6i=LD(p_i[6]+q);
            VD y7r=LD(p_r[7]+q), y7i=LD(p_i[7]+q);
            VD y8r=LD(p_r[8]+q), y8i=LD(p_i[8]+q);
            VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i;
            DFT9V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i,
                  y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i,y6r,y6i,y7r,y7i,y8r,y8i);
            ST(SBre+0*SB36S+q,o0r); ST(SBim+0*SB36S+q,o0i);
            ST(SBre+1*SB36S+q,o1r); ST(SBim+1*SB36S+q,o1i);
            ST(SBre+2*SB36S+q,o2r); ST(SBim+2*SB36S+q,o2i);
            ST(SBre+3*SB36S+q,o3r); ST(SBim+3*SB36S+q,o3i);
            ST(SBre+4*SB36S+q,o4r); ST(SBim+4*SB36S+q,o4i);
            ST(SBre+5*SB36S+q,o5r); ST(SBim+5*SB36S+q,o5i);
            ST(SBre+6*SB36S+q,o6r); ST(SBim+6*SB36S+q,o6i);
            ST(SBre+7*SB36S+q,o7r); ST(SBim+7*SB36S+q,o7i);
            ST(SBre+8*SB36S+q,o8r); ST(SBim+8*SB36S+q,o8i);
            }
        }
        for (int k9 = 0; k9 < 9; k9++)
            yz36(SBre + (size_t)k9*SB36S, SBim + (size_t)k9*SB36S);
        {
            const double *c_r[9], *c_i[9], *b_r[9], *b_i[9];
            double *o_r[9], *o_i[9], *t_r[9], *t_i[9];
            for (int i = 0; i < 9; i++) {
                int k9 = (4*i) % 9;  /* input j9=i from SB[k9] */
                int sl = (9*k4 + 28*k9) % 36;
                b_r[i] = SBre + (size_t)k9*SB36S; b_i[i] = SBim + (size_t)k9*SB36S;
                c_r[i] = cre + (size_t)sl*1296;   c_i[i] = cim + (size_t)sl*1296;
                o_r[i] = outre ? outre + (size_t)sl*1296 : 0;
                o_i[i] = outim ? outim + (size_t)sl*1296 : 0;
                int slot = k4*9 + i;   /* output k9'=i at slot0 k4*9+k9' */
                t_r[i] = T36re + (size_t)slot*PS36; t_i[i] = T36im + (size_t)slot*PS36;
            }
            for (long q = 0; q < 1296; q += 8) {
                VD x0r,x0i; PMAP(x0r,x0i,b_r[0],b_i[0],c_r[0],c_i[0],q);
                VD x1r,x1i; PMAP(x1r,x1i,b_r[1],b_i[1],c_r[1],c_i[1],q);
                VD x2r,x2i; PMAP(x2r,x2i,b_r[2],b_i[2],c_r[2],c_i[2],q);
                VD x3r,x3i; PMAP(x3r,x3i,b_r[3],b_i[3],c_r[3],c_i[3],q);
                VD x4r,x4i; PMAP(x4r,x4i,b_r[4],b_i[4],c_r[4],c_i[4],q);
                VD x5r,x5i; PMAP(x5r,x5i,b_r[5],b_i[5],c_r[5],c_i[5],q);
                VD x6r,x6i; PMAP(x6r,x6i,b_r[6],b_i[6],c_r[6],c_i[6],q);
                VD x7r,x7i; PMAP(x7r,x7i,b_r[7],b_i[7],c_r[7],c_i[7],q);
                VD x8r,x8i; PMAP(x8r,x8i,b_r[8],b_i[8],c_r[8],c_i[8],q);
                if (outre) {
                    ST(o_r[0]+q,x0r); ST(o_i[0]+q,x0i);
                    ST(o_r[1]+q,x1r); ST(o_i[1]+q,x1i);
                    ST(o_r[2]+q,x2r); ST(o_i[2]+q,x2i);
                    ST(o_r[3]+q,x3r); ST(o_i[3]+q,x3i);
                    ST(o_r[4]+q,x4r); ST(o_i[4]+q,x4i);
                    ST(o_r[5]+q,x5r); ST(o_i[5]+q,x5i);
                    ST(o_r[6]+q,x6r); ST(o_i[6]+q,x6i);
                    ST(o_r[7]+q,x7r); ST(o_i[7]+q,x7i);
                    ST(o_r[8]+q,x8r); ST(o_i[8]+q,x8i);
                }
                if (last) continue;
                VD o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i;
            DFT9V(o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i,o8r,o8i,
                      x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,x8r,x8i);
                ST(t_r[0]+q,o0r); ST(t_i[0]+q,o0i);
                ST(t_r[1]+q,o1r); ST(t_i[1]+q,o1i);
                ST(t_r[2]+q,o2r); ST(t_i[2]+q,o2i);
                ST(t_r[3]+q,o3r); ST(t_i[3]+q,o3i);
                ST(t_r[4]+q,o4r); ST(t_i[4]+q,o4i);
                ST(t_r[5]+q,o5r); ST(t_i[5]+q,o5i);
                ST(t_r[6]+q,o6r); ST(t_i[6]+q,o6i);
                ST(t_r[7]+q,o7r); ST(t_i[7]+q,o7i);
                ST(t_r[8]+q,o8r); ST(t_i[8]+q,o8i);
            }
        }
    }
}
static void chain36(double* restrict re, double* restrict im,
                    const double* restrict cre, const double* restrict cim,
                    long m, double* restrict out1)
{
    prime36(re, im);
    for (long n = 0; n < m; n++) {
        int last = (n == m-1), first = (n == 0);
        double *outr = (first || last) ? re : 0, *outi = (first || last) ? im : 0;
        if (!(n & 1)) iter36_even(cre, cim, last, outr, outi);
        else          iter36_odd(cre, cim, last, outr, outi);
        if (first && out1) soa_export(re, im, out1, 46656);
    }
}
// ---------------------------------------------------------------------------
// Per-size one-iteration drivers: A -> (y) -> W -> (z) -> A -> (x+map) -> B
// ---------------------------------------------------------------------------

static void iter6(double* restrict are, double* restrict aim,
                  double* restrict bre, double* restrict bim,
                  const double* restrict cre, const double* restrict cim)
{
    for (int x = 0; x < 6; x++) km6_y(are + 36*x, aim + 36*x, Wre_ + 36*x, Wim_ + 36*x, 0x3F);
    zpass6(Wre_, Wim_, are, aim);
    static const MK pm[5] = {0xFF,0xFF,0xFF,0xFF,0x0F};
    for (int p = 0; p < 5; p++)
        km6_x_map(are + 8*p, aim + 8*p, bre + 8*p, bim + 8*p, cre + 8*p, cim + 8*p, pm[p]);
}
static void iter8(double* restrict are, double* restrict aim,
                  double* restrict bre, double* restrict bim,
                  const double* restrict cre, const double* restrict cim)
{
    for (int x = 0; x < 8; x++) km8_y(are + 64*x, aim + 64*x, Wre_ + 64*x, Wim_ + 64*x);
    zpass8(Wre_, Wim_, are, aim);
    for (int p = 0; p < 8; p++)
        km8_x_map(are + 8*p, aim + 8*p, bre + 8*p, bim + 8*p, cre + 8*p, cim + 8*p);
}
static void iter13(double* restrict are, double* restrict aim,
                   double* restrict bre, double* restrict bim,
                   const double* restrict cre, const double* restrict cim)
{
    for (int x = 0; x < 13; x++) {
        km13_y(are + 169*x, aim + 169*x, Wre_ + 169*x, Wim_ + 169*x, 0xFF);
        km13_y(are + 169*x + 8, aim + 169*x + 8, Wre_ + 169*x + 8, Wim_ + 169*x + 8, 0x1F);
    }
    zpass13(Wre_, Wim_, are, aim, 169);
    for (int p = 0; p < 21; p++)
        km13_x_map(are + 8*p, aim + 8*p, bre + 8*p, bim + 8*p, cre + 8*p, cim + 8*p, 0xFF);
    km13_x_map(are + 168, aim + 168, bre + 168, bim + 168, cre + 168, cim + 168, 0x01);
}
static void iter17(double* restrict are, double* restrict aim,
                   double* restrict bre, double* restrict bim,
                   const double* restrict cre, const double* restrict cim)
{
    for (int x = 0; x < 17; x++) {
        km17_y(are + 289*x, aim + 289*x, Wre_ + 289*x, Wim_ + 289*x, 0xFF);
        km17_y(are + 289*x + 8, aim + 289*x + 8, Wre_ + 289*x + 8, Wim_ + 289*x + 8, 0xFF);
        km17_y(are + 289*x + 16, aim + 289*x + 16, Wre_ + 289*x + 16, Wim_ + 289*x + 16, 0x01);
    }
    zpass17(Wre_, Wim_, are, aim, 289);
    for (int p = 0; p < 36; p++)
        km17_x_map(are + 8*p, aim + 8*p, bre + 8*p, bim + 8*p, cre + 8*p, cim + 8*p, 0xFF);
    km17_x_map(are + 288, aim + 288, bre + 288, bim + 288, cre + 288, cim + 288, 0x01);
}
static void iter23(double* restrict are, double* restrict aim,
                   double* restrict bre, double* restrict bim,
                   const double* restrict cre, const double* restrict cim)
{
    for (int x = 0; x < 23; x++) {
        km23_y(are + 529*x, aim + 529*x, Wre_ + 529*x, Wim_ + 529*x, 0xFF);
        km23_y(are + 529*x + 8, aim + 529*x + 8, Wre_ + 529*x + 8, Wim_ + 529*x + 8, 0xFF);
        km23_y(are + 529*x + 16, aim + 529*x + 16, Wre_ + 529*x + 16, Wim_ + 529*x + 16, 0x7F);
    }
    zpass23(Wre_, Wim_, are, aim, 529);
    for (int p = 0; p < 66; p++)
        km23_x_map(are + 8*p, aim + 8*p, bre + 8*p, bim + 8*p, cre + 8*p, cim + 8*p, 0xFF);
    km23_x_map(are + 528, aim + 528, bre + 528, bim + 528, cre + 528, cim + 528, 0x01);
}


static void iter13ip(double* restrict are, double* restrict aim,
                   double* restrict bre, double* restrict bim,
                   const double* restrict cre, const double* restrict cim)
{
    (void)bre; (void)bim;
    for (int x = 0; x < 13; x++) {
        km13_y(are + 169*x, aim + 169*x, Wre_ + 169*x, Wim_ + 169*x, 0xFF);
        km13_y(are + 169*x + 8, aim + 169*x + 8, Wre_ + 169*x + 8, Wim_ + 169*x + 8, 0x1F);
    }
    zpass13(Wre_, Wim_, are, aim, 169);
    for (int p = 0; p < 21; p++)
        km13_x_map_ip(are + 8*p, aim + 8*p, cre + 8*p, cim + 8*p, 0xFF);
    km13_x_map_ip(are + 168, aim + 168, cre + 168, cim + 168, 0x01);
}
static void iter17ip(double* restrict are, double* restrict aim,
                   double* restrict bre, double* restrict bim,
                   const double* restrict cre, const double* restrict cim)
{
    (void)bre; (void)bim;
    for (int x = 0; x < 17; x++) {
        km17_y(are + 289*x, aim + 289*x, Wre_ + 289*x, Wim_ + 289*x, 0xFF);
        km17_y(are + 289*x + 8, aim + 289*x + 8, Wre_ + 289*x + 8, Wim_ + 289*x + 8, 0xFF);
        km17_y(are + 289*x + 16, aim + 289*x + 16, Wre_ + 289*x + 16, Wim_ + 289*x + 16, 0x01);
    }
    zpass17(Wre_, Wim_, are, aim, 289);
    for (int p = 0; p < 36; p++)
        km17_x_map_ip(are + 8*p, aim + 8*p, cre + 8*p, cim + 8*p, 0xFF);
    km17_x_map_ip(are + 288, aim + 288, cre + 288, cim + 288, 0x01);
}
static void iter23ip(double* restrict are, double* restrict aim,
                   double* restrict bre, double* restrict bim,
                   const double* restrict cre, const double* restrict cim)
{
    (void)bre; (void)bim;
    for (int x = 0; x < 23; x++) {
        km23_y(are + 529*x, aim + 529*x, Wre_ + 529*x, Wim_ + 529*x, 0xFF);
        km23_y(are + 529*x + 8, aim + 529*x + 8, Wre_ + 529*x + 8, Wim_ + 529*x + 8, 0xFF);
        km23_y(are + 529*x + 16, aim + 529*x + 16, Wre_ + 529*x + 16, Wim_ + 529*x + 16, 0x7F);
    }
    zpass23(Wre_, Wim_, are, aim, 529);
    for (int p = 0; p < 66; p++)
        km23_x_map_ip(are + 8*p, aim + 8*p, cre + 8*p, cim + 8*p, 0xFF);
    km23_x_map_ip(are + 528, aim + 528, cre + 528, cim + 528, 0x01);
}
static void iter36ip(double* restrict are, double* restrict aim,
                   double* restrict bre, double* restrict bim,
                   const double* restrict cre, const double* restrict cim)
{
    (void)bre; (void)bim;
    static const MK zm[5] = {0xFF,0xFF,0xFF,0xFF,0x0F};
    for (int x = 0; x < 36; x++) {
        double *sr = are + 1296*x, *si = aim + 1296*x;
        double *wr = Wre_ + 1296*(x & 1), *wi = Wim_ + 1296*(x & 1);
        for (int zc = 0; zc < 4; zc++) km36_y(sr + 8*zc, si + 8*zc, wr + 8*zc, wi + 8*zc, 0xFF);
        km36_y(sr + 32, si + 32, wr + 32, wi + 32, 0x0F);
        spass_gen(wr, wi, sr, si, 36, 5, zm, km36_s);
    }
    for (int p = 0; p < 162; p++)
        km36_x_map_ip(are + 8*p, aim + 8*p, cre + 8*p, cim + 8*p);
}
static void iter45ip(double* restrict are, double* restrict aim,
                   double* restrict bre, double* restrict bim,
                   const double* restrict cre, const double* restrict cim)
{
    (void)bre; (void)bim;
    static const MK zm[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0x1F};
    for (int x = 0; x < 45; x++) {
        double *sr = are + 2025*x, *si = aim + 2025*x;
        double *wr = Wre_ + 2048*(x & 1), *wi = Wim_ + 2048*(x & 1);
        for (int zc = 0; zc < 5; zc++) km45_y(sr + 8*zc, si + 8*zc, wr + 8*zc, wi + 8*zc, 0xFF);
        km45_y(sr + 40, si + 40, wr + 40, wi + 40, 0x1F);
        spass_gen(wr, wi, sr, si, 45, 6, zm, km45_s);
    }
    for (int p = 0; p < 253; p++)
        km45_x_map_ip(are + 8*p, aim + 8*p, cre + 8*p, cim + 8*p, 0xFF);
    km45_x_map_ip(are + 2024, aim + 2024, cre + 2024, cim + 2024, 0x01);
}


// IP_MODE[L-index]: 1 = in-place x+map (no ping-pong swap)
static int IPMODE(int L){ switch(L){ case 13: case 17: case 23: case 36: case 45: return 1; } return 0; }
// ---------------------------------------------------------------------------
// main entry: run the iteration chain for one size
// ---------------------------------------------------------------------------
void run_chain(int L, long B, long m,
               const double* x0, const double* c,
               double* out1, double* outm)
{
    if (B <= 0) return;
    long n = (long)L*L*L;
    long NP = ((n + 7) & ~7L) + 24;
    long SV = 2*NP + 8;
    size_t bytes = ((size_t)B*SV + 64) * sizeof(double);
    bytes = (bytes + 63) & ~(size_t)63;
    int needB = (L == 6 || L == 8);
    double *stA = (double*)aligned_alloc(64, bytes);
    double *stB = needB ? (double*)aligned_alloc(64, bytes) : 0;
    double *cb  = (double*)aligned_alloc(64, bytes);
    void (*iter)(double*, double*, double*, double*, const double*, const double*) = 0;
    switch (L) {
        case 6:  iter = iter6;  break;
        case 8:  iter = iter8;  break;
        case 13: iter = iter13ip; break;
        case 17: iter = iter17ip; break;
        case 23: iter = iter23ip; break;
        case 36: iter = iter36ip; break;
        case 45: iter = iter45ip; break;
        case 64: break;
        default: free(stA); if (stB) free(stB); free(cb); return;
    }
    long b0 = 0;
    if ((L == 6 || L == 8 || L == 13 || L == 17 || L == 23) && m > 0) {
        int rmin = (L == 6) ? 5 : (L == 13) ? 6 : (L == 23) ? 9 : 7;
        for (; b0 + 8 <= B; b0 += 8)
            vl_chain(L, m, x0 + 2*b0*n, c + 2*b0*n,
                     out1 + 2*b0*n, outm + 2*b0*n, 8);
        if (B - b0 >= rmin) {
            vl_chain(L, m, x0 + 2*b0*n, c + 2*b0*n,
                     out1 + 2*b0*n, outm + 2*b0*n, (int)(B - b0));
            b0 = B;
        }
    }
    for (long b = b0; b < B; b++) {
        double *ar = stA + b*SV, *ai = ar + NP;
        double *br = needB ? stB + b*SV : ar, *bi = needB ? br + NP : ai;
        double *cre = cb + b*SV, *cim = cre + NP;
        soa_import(ar, ai, x0 + 2*b*n, n);
        soa_import(cre, cim, c + 2*b*n, n);
        // zero pads (beyond n up to NP, plus inter-volume gap) to keep dead
        // lanes finite
        for (long i = n; i < NP; i++) { ar[i]=0; ai[i]=0; cre[i]=0; cim[i]=0; br[i]=0; bi[i]=0; }
        for (long i = 0; i < 8; i++) { ai[NP+i]=0; cim[NP+i]=0; bi[NP+i]=0; }
        if (m <= 0) { soa_export(ar, ai, out1 + 2*b*n, n); soa_export(ar, ai, outm + 2*b*n, n); continue; }
        if (L == 64) {
            chain64(ar, ai, cre, cim, m, T64re, T64im, 0, 0, out1 + 2*b*n);
            soa_export(ar, ai, outm + 2*b*n, n);
        } else if (L == 45) {
            chain45(ar, ai, cre, cim, m, out1 + 2*b*n);
            soa_export(ar, ai, outm + 2*b*n, n);
        } else if (L == 36) {
            chain36(ar, ai, cre, cim, m, out1 + 2*b*n);
            soa_export(ar, ai, outm + 2*b*n, n);
        } else {
            int ip = IPMODE(L);
            for (long it = 0; it < m; it++) {
                iter(ar, ai, br, bi, cre, cim);
                if (!ip) { double *t;
                    t = ar; ar = br; br = t;
                    t = ai; ai = bi; bi = t; }
                if (it == 0) soa_export(ar, ai, out1 + 2*b*n, n);
            }
            soa_export(ar, ai, outm + 2*b*n, n);
        }
    }
    free(stA); if (stB) free(stB); free(cb);
}
