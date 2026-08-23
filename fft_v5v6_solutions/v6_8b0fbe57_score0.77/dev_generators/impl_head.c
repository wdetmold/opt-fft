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
