src = open('implementation.c').read()

# add FFT4_CORE after FFT5_CORE definition
src = src.replace("""// FFT6: outputs in natural order""",
"""#define FFT4_CORE(x0, x1, x2, x3)                              \\
  {                                                            \\
    v8 _t0r = x0##r + x2##r, _t0i = x0##i + x2##i;             \\
    v8 _t1r = x0##r - x2##r, _t1i = x0##i - x2##i;             \\
    v8 _t2r = x1##r + x3##r, _t2i = x1##i + x3##i;             \\
    v8 _t3r = x1##r - x3##r, _t3i = x1##i - x3##i;             \\
    x0##r = _t0r + _t2r; x0##i = _t0i + _t2i;                  \\
    x2##r = _t0r - _t2r; x2##i = _t0i - _t2i;                  \\
    x1##r = _t1r + _t3i; x1##i = _t1i - _t3r;                  \\
    x3##r = _t1r - _t3i; x3##i = _t1i + _t3r;                  \\
  }

// FFT6: outputs in natural order""")

# ---- replace fft36_line with PFA(4,9) ----
start = src.index("// 36 = 6 x 6")
end = src.index("// 45 = 5 x 9")
new36 = '''// 36 = 4 (x) 9  PFA: no twiddles.  input n=(9*n1+4*n2)%36, output k=(9*q1+28*q2)%36
static inline __attribute__((always_inline)) void
fft36_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim, const double *pf) {
  DEFCONSTS
  _Pragma("GCC unroll 4") for (int n1 = 0; n1 < 4; n1++) {
    if (pf) {
      _Pragma("GCC unroll 9") for (int q = 0; q < 9; q++) {
        PF1(pf + (n1 * 9 + q) * s); PF1(pf + PLANE + (n1 * 9 + q) * s);
      }
    }
    v8 y0r = VL(re + ((9 * n1 + 0) % 36) * s), y0i = VL(im + ((9 * n1 + 0) % 36) * s);
    v8 y1r = VL(re + ((9 * n1 + 4) % 36) * s), y1i = VL(im + ((9 * n1 + 4) % 36) * s);
    v8 y2r = VL(re + ((9 * n1 + 8) % 36) * s), y2i = VL(im + ((9 * n1 + 8) % 36) * s);
    v8 y3r = VL(re + ((9 * n1 + 12) % 36) * s), y3i = VL(im + ((9 * n1 + 12) % 36) * s);
    v8 y4r = VL(re + ((9 * n1 + 16) % 36) * s), y4i = VL(im + ((9 * n1 + 16) % 36) * s);
    v8 y5r = VL(re + ((9 * n1 + 20) % 36) * s), y5i = VL(im + ((9 * n1 + 20) % 36) * s);
    v8 y6r = VL(re + ((9 * n1 + 24) % 36) * s), y6i = VL(im + ((9 * n1 + 24) % 36) * s);
    v8 y7r = VL(re + ((9 * n1 + 28) % 36) * s), y7i = VL(im + ((9 * n1 + 28) % 36) * s);
    v8 y8r = VL(re + ((9 * n1 + 32) % 36) * s), y8i = VL(im + ((9 * n1 + 32) % 36) * s);
    FFT9_CORE(y0, y1, y2, y3, y4, y5, y6, y7, y8);
    // X9[q2] lives in var P[q2], P = {0,3,6,1,4,7,2,5,8}
#define ST36(q2, var)                                                     \\
    { VS(T1R + ((q2) * 4 + n1) * 8, var##r); VS(T1I + ((q2) * 4 + n1) * 8, var##i); }
    ST36(0, y0) ST36(1, y3) ST36(2, y6) ST36(3, y1) ST36(4, y4)
    ST36(5, y7) ST36(6, y2) ST36(7, y5) ST36(8, y8)
#undef ST36
  }
  _Pragma("GCC unroll 9") for (int q2 = 0; q2 < 9; q2++) {
    if (pf) {
      _Pragma("GCC unroll 4") for (int q = 0; q < 4; q++) {
        PFC(pf + 2 * PLANE + (q2 * 4 + q) * s); PFC(pf + 3 * PLANE + (q2 * 4 + q) * s);
      }
    }
    v8 x0r = VL(T1R + (q2 * 4 + 0) * 8), x0i = VL(T1I + (q2 * 4 + 0) * 8);
    v8 x1r = VL(T1R + (q2 * 4 + 1) * 8), x1i = VL(T1I + (q2 * 4 + 1) * 8);
    v8 x2r = VL(T1R + (q2 * 4 + 2) * 8), x2i = VL(T1I + (q2 * 4 + 2) * 8);
    v8 x3r = VL(T1R + (q2 * 4 + 3) * 8), x3i = VL(T1I + (q2 * 4 + 3) * 8);
    FFT4_CORE(x0, x1, x2, x3);
    STORE_X(re, im, x0r, x0i, ((9 * 0 + 28 * q2) % 36) * s);
    STORE_X(re, im, x1r, x1i, ((9 * 1 + 28 * q2) % 36) * s);
    STORE_X(re, im, x2r, x2i, ((9 * 2 + 28 * q2) % 36) * s);
    STORE_X(re, im, x3r, x3i, ((9 * 3 + 28 * q2) % 36) * s);
  }
}

'''
src = src[:start] + new36 + src[end:]

# ---- replace fft45_line with PFA(5,9) ----
start = src.index("// 45 = 5 x 9")
end = src.index("// 64 = 8 x 8")
new45 = '''// 45 = 5 (x) 9  PFA: input n=(9*n1+5*n2)%45, output k=(36*q1+10*q2)%45
static inline __attribute__((always_inline)) void
fft45_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim, const double *pf) {
  DEFCONSTS
  _Pragma("GCC unroll 5") for (int n1 = 0; n1 < 5; n1++) {
    if (pf) {
      _Pragma("GCC unroll 9") for (int q = 0; q < 9; q++) {
        PF1(pf + (n1 * 9 + q) * s); PF1(pf + PLANE + (n1 * 9 + q) * s);
      }
    }
    v8 y0r = VL(re + ((9 * n1 + 0) % 45) * s), y0i = VL(im + ((9 * n1 + 0) % 45) * s);
    v8 y1r = VL(re + ((9 * n1 + 5) % 45) * s), y1i = VL(im + ((9 * n1 + 5) % 45) * s);
    v8 y2r = VL(re + ((9 * n1 + 10) % 45) * s), y2i = VL(im + ((9 * n1 + 10) % 45) * s);
    v8 y3r = VL(re + ((9 * n1 + 15) % 45) * s), y3i = VL(im + ((9 * n1 + 15) % 45) * s);
    v8 y4r = VL(re + ((9 * n1 + 20) % 45) * s), y4i = VL(im + ((9 * n1 + 20) % 45) * s);
    v8 y5r = VL(re + ((9 * n1 + 25) % 45) * s), y5i = VL(im + ((9 * n1 + 25) % 45) * s);
    v8 y6r = VL(re + ((9 * n1 + 30) % 45) * s), y6i = VL(im + ((9 * n1 + 30) % 45) * s);
    v8 y7r = VL(re + ((9 * n1 + 35) % 45) * s), y7i = VL(im + ((9 * n1 + 35) % 45) * s);
    v8 y8r = VL(re + ((9 * n1 + 40) % 45) * s), y8i = VL(im + ((9 * n1 + 40) % 45) * s);
    FFT9_CORE(y0, y1, y2, y3, y4, y5, y6, y7, y8);
#define ST45(q2, var)                                                     \\
    { VS(T1R + ((q2) * 5 + n1) * 8, var##r); VS(T1I + ((q2) * 5 + n1) * 8, var##i); }
    ST45(0, y0) ST45(1, y3) ST45(2, y6) ST45(3, y1) ST45(4, y4)
    ST45(5, y7) ST45(6, y2) ST45(7, y5) ST45(8, y8)
#undef ST45
  }
  _Pragma("GCC unroll 9") for (int q2 = 0; q2 < 9; q2++) {
    if (pf) {
      _Pragma("GCC unroll 5") for (int q = 0; q < 5; q++) {
        PFC(pf + 2 * PLANE + (q2 * 5 + q) * s); PFC(pf + 3 * PLANE + (q2 * 5 + q) * s);
      }
    }
    v8 x0r = VL(T1R + (q2 * 5 + 0) * 8), x0i = VL(T1I + (q2 * 5 + 0) * 8);
    v8 x1r = VL(T1R + (q2 * 5 + 1) * 8), x1i = VL(T1I + (q2 * 5 + 1) * 8);
    v8 x2r = VL(T1R + (q2 * 5 + 2) * 8), x2i = VL(T1I + (q2 * 5 + 2) * 8);
    v8 x3r = VL(T1R + (q2 * 5 + 3) * 8), x3i = VL(T1I + (q2 * 5 + 3) * 8);
    v8 x4r = VL(T1R + (q2 * 5 + 4) * 8), x4i = VL(T1I + (q2 * 5 + 4) * 8);
    FFT5_CORE(x0, x1, x2, x3, x4);
    STORE_X(re, im, x0r, x0i, ((36 * 0 + 10 * q2) % 45) * s);
    STORE_X(re, im, x1r, x1i, ((36 * 1 + 10 * q2) % 45) * s);
    STORE_X(re, im, x2r, x2i, ((36 * 2 + 10 * q2) % 45) * s);
    STORE_X(re, im, x3r, x3i, ((36 * 3 + 10 * q2) % 45) * s);
    STORE_X(re, im, x4r, x4i, ((36 * 4 + 10 * q2) % 45) * s);
  }
}

'''
src = src[:start] + new45 + src[end:]
open('implementation.c','w').write(src)
print("ok")
