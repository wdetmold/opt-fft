src = open('implementation.c').read()
old = src[src.index("// 64 = 8 x 8"):src.index("// ---------------------------------------------------------------- engines")]
new = '''// 64 = 8 x 8   (stage loops rolled: front-end friendly)
static inline __attribute__((always_inline)) void
fft64_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim, const double *pf) {
  DEFCONSTS
  // n1 = 0 peeled (no twiddles)
  {
    v8 x0r = VL(re + 0 * s), x0i = VL(im + 0 * s);
    v8 x1r = VL(re + 8 * s), x1i = VL(im + 8 * s);
    v8 x2r = VL(re + 16 * s), x2i = VL(im + 16 * s);
    v8 x3r = VL(re + 24 * s), x3i = VL(im + 24 * s);
    v8 x4r = VL(re + 32 * s), x4i = VL(im + 32 * s);
    v8 x5r = VL(re + 40 * s), x5i = VL(im + 40 * s);
    v8 x6r = VL(re + 48 * s), x6i = VL(im + 48 * s);
    v8 x7r = VL(re + 56 * s), x7i = VL(im + 56 * s);
    if (pf) {
      _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++) {
        PF1(pf + (0 * 8 + q) * s); PF1(pf + PLANE + (0 * 8 + q) * s);
      }
    }
    FFT8_CORE(x0, x1, x2, x3, x4, x5, x6, x7);
    VS(T1R + (0 * 8 + 0) * 8, x0r); VS(T1I + (0 * 8 + 0) * 8, x0i);
    VS(T1R + (1 * 8 + 0) * 8, x1r); VS(T1I + (1 * 8 + 0) * 8, x1i);
    VS(T1R + (2 * 8 + 0) * 8, x2r); VS(T1I + (2 * 8 + 0) * 8, x2i);
    VS(T1R + (3 * 8 + 0) * 8, x3r); VS(T1I + (3 * 8 + 0) * 8, x3i);
    VS(T1R + (4 * 8 + 0) * 8, x4r); VS(T1I + (4 * 8 + 0) * 8, x4i);
    VS(T1R + (5 * 8 + 0) * 8, x5r); VS(T1I + (5 * 8 + 0) * 8, x5i);
    VS(T1R + (6 * 8 + 0) * 8, x6r); VS(T1I + (6 * 8 + 0) * 8, x6i);
    VS(T1R + (7 * 8 + 0) * 8, x7r); VS(T1I + (7 * 8 + 0) * 8, x7i);
  }
  for (int n1 = 1; n1 < 8; n1++) {
    const double *rr = re + n1 * s, *ii = im + n1 * s;
    v8 x0r = VL(rr + 0 * s), x0i = VL(ii + 0 * s);
    v8 x1r = VL(rr + 8 * s), x1i = VL(ii + 8 * s);
    v8 x2r = VL(rr + 16 * s), x2i = VL(ii + 16 * s);
    v8 x3r = VL(rr + 24 * s), x3i = VL(ii + 24 * s);
    v8 x4r = VL(rr + 32 * s), x4i = VL(ii + 32 * s);
    v8 x5r = VL(rr + 40 * s), x5i = VL(ii + 40 * s);
    v8 x6r = VL(rr + 48 * s), x6i = VL(ii + 48 * s);
    v8 x7r = VL(rr + 56 * s), x7i = VL(ii + 56 * s);
    if (pf) {
      _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++) {
        PF1(pf + (n1 * 8 + q) * s); PF1(pf + PLANE + (n1 * 8 + q) * s);
      }
    }
    FFT8_CORE(x0, x1, x2, x3, x4, x5, x6, x7);
    const double *twr = TW64R + n1 * 8, *twi = TW64I + n1 * 8;
    CMULT(x1r, x1i, x1r, x1i, vbc(twr[1]), vbc(twi[1]));
    CMULT(x2r, x2i, x2r, x2i, vbc(twr[2]), vbc(twi[2]));
    CMULT(x3r, x3i, x3r, x3i, vbc(twr[3]), vbc(twi[3]));
    CMULT(x4r, x4i, x4r, x4i, vbc(twr[4]), vbc(twi[4]));
    CMULT(x5r, x5i, x5r, x5i, vbc(twr[5]), vbc(twi[5]));
    CMULT(x6r, x6i, x6r, x6i, vbc(twr[6]), vbc(twi[6]));
    CMULT(x7r, x7i, x7r, x7i, vbc(twr[7]), vbc(twi[7]));
    double *t1r = T1R + n1 * 8, *t1i = T1I + n1 * 8;
    VS(t1r + 0 * 64, x0r); VS(t1i + 0 * 64, x0i);
    VS(t1r + 1 * 64, x1r); VS(t1i + 1 * 64, x1i);
    VS(t1r + 2 * 64, x2r); VS(t1i + 2 * 64, x2i);
    VS(t1r + 3 * 64, x3r); VS(t1i + 3 * 64, x3i);
    VS(t1r + 4 * 64, x4r); VS(t1i + 4 * 64, x4i);
    VS(t1r + 5 * 64, x5r); VS(t1i + 5 * 64, x5i);
    VS(t1r + 6 * 64, x6r); VS(t1i + 6 * 64, x6i);
    VS(t1r + 7 * 64, x7r); VS(t1i + 7 * 64, x7i);
  }
  for (int k2 = 0; k2 < 8; k2++) {
    const double *t1r = T1R + k2 * 64, *t1i = T1I + k2 * 64;
    v8 x0r = VL(t1r + 0 * 8), x0i = VL(t1i + 0 * 8);
    v8 x1r = VL(t1r + 1 * 8), x1i = VL(t1i + 1 * 8);
    v8 x2r = VL(t1r + 2 * 8), x2i = VL(t1i + 2 * 8);
    v8 x3r = VL(t1r + 3 * 8), x3i = VL(t1i + 3 * 8);
    v8 x4r = VL(t1r + 4 * 8), x4i = VL(t1i + 4 * 8);
    v8 x5r = VL(t1r + 5 * 8), x5i = VL(t1i + 5 * 8);
    v8 x6r = VL(t1r + 6 * 8), x6i = VL(t1i + 6 * 8);
    v8 x7r = VL(t1r + 7 * 8), x7i = VL(t1i + 7 * 8);
    if (pf && domap) {
      _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++) {
        PFC(pf + 2 * PLANE + (k2 * 8 + q) * s); PFC(pf + 3 * PLANE + (k2 * 8 + q) * s);
      }
    }
    FFT8_CORE(x0, x1, x2, x3, x4, x5, x6, x7);
    double *ro = re + k2 * s, *io = im + k2 * s;
    const double *cro = cre + k2 * s, *cio = cim + k2 * s;
    (void)cro; (void)cio;
    STORE_X(ro, io, x0r, x0i, 0 * 8 * s - k2 * s + (0 * 8 + k2) * s);
    STORE_X(ro, io, x1r, x1i, (1 * 8 + k2) * s - k2 * s);
    STORE_X(ro, io, x2r, x2i, (2 * 8 + k2) * s - k2 * s);
    STORE_X(ro, io, x3r, x3i, (3 * 8 + k2) * s - k2 * s);
    STORE_X(ro, io, x4r, x4i, (4 * 8 + k2) * s - k2 * s);
    STORE_X(ro, io, x5r, x5i, (5 * 8 + k2) * s - k2 * s);
    STORE_X(ro, io, x6r, x6i, (6 * 8 + k2) * s - k2 * s);
    STORE_X(ro, io, x7r, x7i, (7 * 8 + k2) * s - k2 * s);
  }
}

'''
src = src.replace(old, new)
open('implementation.c','w').write(src)
print("ok")
