src = open('implementation.c').read()

# offset tables
src = src.replace("// ---------------------------------------------------------------- line FFTs",
"""// PFA index tables (input (9*n1+K*n2)%N, output (A*q1+B*q2)%N)
static long MI36[4][9], MO36[9][4];   // 36 = 4 (x) 9
static long MI45[5][9], MO45[9][5];   // 45 = 5 (x) 9
// ---------------------------------------------------------------- line FFTs""")
src = src.replace("""  fill_prime(C13, S13, 13, 6, PH13);""",
"""  fill_prime(C13, S13, 13, 6, PH13);
  for (int n1 = 0; n1 < 4; n1++)
    for (int n2 = 0; n2 < 9; n2++) MI36[n1][n2] = (9 * n1 + 4 * n2) % 36;
  for (int q2 = 0; q2 < 9; q2++)
    for (int q1 = 0; q1 < 4; q1++) MO36[q2][q1] = (9 * q1 + 28 * q2) % 36;
  for (int n1 = 0; n1 < 5; n1++)
    for (int n2 = 0; n2 < 9; n2++) MI45[n1][n2] = (9 * n1 + 5 * n2) % 45;
  for (int q2 = 0; q2 < 9; q2++)
    for (int q1 = 0; q1 < 5; q1++) MO45[q2][q1] = (36 * q1 + 10 * q2) % 45;""")

# ---- fft36: roll loops
old = src[src.index("// 36 = 4 (x) 9  PFA"):src.index("// 45 = 5 (x) 9  PFA")]
new = '''// 36 = 4 (x) 9  PFA: no twiddles (rolled stage loops)
static inline __attribute__((always_inline)) void
fft36_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim, const double *pf) {
  DEFCONSTS
  for (int n1 = 0; n1 < 4; n1++) {
    const long *mi = MI36[n1];
    if (pf) {
      _Pragma("GCC unroll 9") for (int q = 0; q < 9; q++) {
        PF1(pf + (n1 * 9 + q) * s); PF1(pf + PLANE + (n1 * 9 + q) * s);
      }
    }
    v8 y0r = VL(re + mi[0] * s), y0i = VL(im + mi[0] * s);
    v8 y1r = VL(re + mi[1] * s), y1i = VL(im + mi[1] * s);
    v8 y2r = VL(re + mi[2] * s), y2i = VL(im + mi[2] * s);
    v8 y3r = VL(re + mi[3] * s), y3i = VL(im + mi[3] * s);
    v8 y4r = VL(re + mi[4] * s), y4i = VL(im + mi[4] * s);
    v8 y5r = VL(re + mi[5] * s), y5i = VL(im + mi[5] * s);
    v8 y6r = VL(re + mi[6] * s), y6i = VL(im + mi[6] * s);
    v8 y7r = VL(re + mi[7] * s), y7i = VL(im + mi[7] * s);
    v8 y8r = VL(re + mi[8] * s), y8i = VL(im + mi[8] * s);
    FFT9_CORE(y0, y1, y2, y3, y4, y5, y6, y7, y8);
    // X9[q2] lives in var P[q2], P = {0,3,6,1,4,7,2,5,8}
    double *t1r = T1R + n1 * 8, *t1i = T1I + n1 * 8;
#define ST36(q2, var)  { VS(t1r + (q2) * 32, var##r); VS(t1i + (q2) * 32, var##i); }
    ST36(0, y0) ST36(1, y3) ST36(2, y6) ST36(3, y1) ST36(4, y4)
    ST36(5, y7) ST36(6, y2) ST36(7, y5) ST36(8, y8)
#undef ST36
  }
  for (int q2 = 0; q2 < 9; q2++) {
    const long *mo = MO36[q2];
    if (pf && domap) {
      _Pragma("GCC unroll 4") for (int q = 0; q < 4; q++) {
        PFC(pf + 2 * PLANE + (q2 * 4 + q) * s); PFC(pf + 3 * PLANE + (q2 * 4 + q) * s);
      }
    }
    const double *t1r = T1R + q2 * 32, *t1i = T1I + q2 * 32;
    v8 x0r = VL(t1r + 0 * 8), x0i = VL(t1i + 0 * 8);
    v8 x1r = VL(t1r + 1 * 8), x1i = VL(t1i + 1 * 8);
    v8 x2r = VL(t1r + 2 * 8), x2i = VL(t1i + 2 * 8);
    v8 x3r = VL(t1r + 3 * 8), x3i = VL(t1i + 3 * 8);
    FFT4_CORE(x0, x1, x2, x3);
    STORE_X(re, im, x0r, x0i, mo[0] * s);
    STORE_X(re, im, x1r, x1i, mo[1] * s);
    STORE_X(re, im, x2r, x2i, mo[2] * s);
    STORE_X(re, im, x3r, x3i, mo[3] * s);
  }
}

'''
src = src.replace(old, new)

# ---- fft45: roll loops
old = src[src.index("// 45 = 5 (x) 9  PFA"):src.index("// 64 = 8 x 8")]
new = '''// 45 = 5 (x) 9  PFA (rolled stage loops)
static inline __attribute__((always_inline)) void
fft45_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim, const double *pf) {
  DEFCONSTS
  for (int n1 = 0; n1 < 5; n1++) {
    const long *mi = MI45[n1];
    if (pf) {
      _Pragma("GCC unroll 9") for (int q = 0; q < 9; q++) {
        PF1(pf + (n1 * 9 + q) * s); PF1(pf + PLANE + (n1 * 9 + q) * s);
      }
    }
    v8 y0r = VL(re + mi[0] * s), y0i = VL(im + mi[0] * s);
    v8 y1r = VL(re + mi[1] * s), y1i = VL(im + mi[1] * s);
    v8 y2r = VL(re + mi[2] * s), y2i = VL(im + mi[2] * s);
    v8 y3r = VL(re + mi[3] * s), y3i = VL(im + mi[3] * s);
    v8 y4r = VL(re + mi[4] * s), y4i = VL(im + mi[4] * s);
    v8 y5r = VL(re + mi[5] * s), y5i = VL(im + mi[5] * s);
    v8 y6r = VL(re + mi[6] * s), y6i = VL(im + mi[6] * s);
    v8 y7r = VL(re + mi[7] * s), y7i = VL(im + mi[7] * s);
    v8 y8r = VL(re + mi[8] * s), y8i = VL(im + mi[8] * s);
    FFT9_CORE(y0, y1, y2, y3, y4, y5, y6, y7, y8);
    double *t1r = T1R + n1 * 8, *t1i = T1I + n1 * 8;
#define ST45(q2, var)  { VS(t1r + (q2) * 40, var##r); VS(t1i + (q2) * 40, var##i); }
    ST45(0, y0) ST45(1, y3) ST45(2, y6) ST45(3, y1) ST45(4, y4)
    ST45(5, y7) ST45(6, y2) ST45(7, y5) ST45(8, y8)
#undef ST45
  }
  for (int q2 = 0; q2 < 9; q2++) {
    const long *mo = MO45[q2];
    if (pf && domap) {
      _Pragma("GCC unroll 5") for (int q = 0; q < 5; q++) {
        PFC(pf + 2 * PLANE + (q2 * 5 + q) * s); PFC(pf + 3 * PLANE + (q2 * 5 + q) * s);
      }
    }
    const double *t1r = T1R + q2 * 40, *t1i = T1I + q2 * 40;
    v8 x0r = VL(t1r + 0 * 8), x0i = VL(t1i + 0 * 8);
    v8 x1r = VL(t1r + 1 * 8), x1i = VL(t1i + 1 * 8);
    v8 x2r = VL(t1r + 2 * 8), x2i = VL(t1i + 2 * 8);
    v8 x3r = VL(t1r + 3 * 8), x3i = VL(t1i + 3 * 8);
    v8 x4r = VL(t1r + 4 * 8), x4i = VL(t1i + 4 * 8);
    FFT5_CORE(x0, x1, x2, x3, x4);
    STORE_X(re, im, x0r, x0i, mo[0] * s);
    STORE_X(re, im, x1r, x1i, mo[1] * s);
    STORE_X(re, im, x2r, x2i, mo[2] * s);
    STORE_X(re, im, x3r, x3i, mo[3] * s);
    STORE_X(re, im, x4r, x4i, mo[4] * s);
  }
}

'''
src = src.replace(old, new)
open('implementation.c','w').write(src)
print("ok")
