src = open('implementation.c').read()

# 1. Add pf parameter to all line fns: change signature pattern
src = src.replace("""fft6_line(double *re, double *im, const long s, const int domap,
          const double *cre, const double *cim) {""",
"""fft6_line(double *re, double *im, const long s, const int domap,
          const double *cre, const double *cim, const double *pf) { (void)pf;""")
src = src.replace("""fft8_line(double *re, double *im, const long s, const int domap,
          const double *cre, const double *cim) {""",
"""fft8_line(double *re, double *im, const long s, const int domap,
          const double *cre, const double *cim, const double *pf) { (void)pf;""")
for L in (13,17,23):
    src = src.replace(f"""fft{L}_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim) {{""",
f"""fft{L}_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim, const double *pf) {{ (void)pf;""")
for L in (36,45,64):
    src = src.replace(f"""fft{L}_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim) {{""",
f"""fft{L}_line(double *re, double *im, const long s, const int domap,
           const double *cre, const double *cim, const double *pf) {{""")

# 2. plane offset constant
src = src.replace("static double *Xre, *Xim, *Cre, *Cim;   // planar volume buffers (max size)",
"""#define PLANE 295424L   // doubles per plane (= 64*4616); buffers are contiguous
static double *Xre, *Xim, *Cre, *Cim;   // planar volume buffers (max size)
#define PF1(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)""")

# 3. insert prefetches in fft36 stage1 (6 iters, 12/iter = 72 + extra) and stage2 (12/iter)
src = src.replace("""    FFT6_CORE(x0, x1, x2, x3, x4, x5);
    if (n1) {
      CMULT(x1r, x1i, x1r, x1i, vbc(TW36R[n1 * 6 + 1]), vbc(TW36I[n1 * 6 + 1]));""",
"""    if (pf) {
      PF1(pf + (n1 * 6 + 0) * s); PF1(pf + (n1 * 6 + 1) * s);
      PF1(pf + (n1 * 6 + 2) * s); PF1(pf + (n1 * 6 + 3) * s);
      PF1(pf + (n1 * 6 + 4) * s); PF1(pf + (n1 * 6 + 5) * s);
      PF1(pf + PLANE + (n1 * 6 + 0) * s); PF1(pf + PLANE + (n1 * 6 + 1) * s);
      PF1(pf + PLANE + (n1 * 6 + 2) * s); PF1(pf + PLANE + (n1 * 6 + 3) * s);
      PF1(pf + PLANE + (n1 * 6 + 4) * s); PF1(pf + PLANE + (n1 * 6 + 5) * s);
    }
    FFT6_CORE(x0, x1, x2, x3, x4, x5);
    if (n1) {
      CMULT(x1r, x1i, x1r, x1i, vbc(TW36R[n1 * 6 + 1]), vbc(TW36I[n1 * 6 + 1]));""")
src = src.replace("""    FFT6_CORE(x0, x1, x2, x3, x4, x5);
    STORE_X(re, im, x0r, x0i, (0 * 6 + k2) * s);""",
"""    if (pf) {
      PF1(pf + 2 * PLANE + (k2 * 6 + 0) * s); PF1(pf + 2 * PLANE + (k2 * 6 + 1) * s);
      PF1(pf + 2 * PLANE + (k2 * 6 + 2) * s); PF1(pf + 2 * PLANE + (k2 * 6 + 3) * s);
      PF1(pf + 2 * PLANE + (k2 * 6 + 4) * s); PF1(pf + 2 * PLANE + (k2 * 6 + 5) * s);
      PF1(pf + 3 * PLANE + (k2 * 6 + 0) * s); PF1(pf + 3 * PLANE + (k2 * 6 + 1) * s);
      PF1(pf + 3 * PLANE + (k2 * 6 + 2) * s); PF1(pf + 3 * PLANE + (k2 * 6 + 3) * s);
      PF1(pf + 3 * PLANE + (k2 * 6 + 4) * s); PF1(pf + 3 * PLANE + (k2 * 6 + 5) * s);
    }
    FFT6_CORE(x0, x1, x2, x3, x4, x5);
    STORE_X(re, im, x0r, x0i, (0 * 6 + k2) * s);""")

# 4. fft45: stage1 (5 iters, 9 elems each -> 18/iter), stage2 (9 iters, 5 elems -> 10/iter)
src = src.replace("""    FFT9_CORE(y0, y1, y2, y3, y4, y5, y6, y7, y8);
    // X[k] is in var P[k], P = {0,3,6,1,4,7,2,5,8}""",
"""    if (pf) {
      _Pragma("GCC unroll 9") for (int q = 0; q < 9; q++) {
        PF1(pf + (n1 * 9 + q) * s); PF1(pf + PLANE + (n1 * 9 + q) * s);
      }
    }
    FFT9_CORE(y0, y1, y2, y3, y4, y5, y6, y7, y8);
    // X[k] is in var P[k], P = {0,3,6,1,4,7,2,5,8}""")
src = src.replace("""    FFT5_CORE(x0, x1, x2, x3, x4);
    STORE_X(re, im, x0r, x0i, (0 * 9 + k2) * s);""",
"""    if (pf) {
      _Pragma("GCC unroll 5") for (int q = 0; q < 5; q++) {
        PF1(pf + 2 * PLANE + (k2 * 5 + q) * s); PF1(pf + 3 * PLANE + (k2 * 5 + q) * s);
      }
    }
    FFT5_CORE(x0, x1, x2, x3, x4);
    STORE_X(re, im, x0r, x0i, (0 * 9 + k2) * s);""")

# 5. fft64: stage1 (8 iters, 8 elems -> 16/iter), stage2 (8 iters -> 16/iter)
src = src.replace("""    FFT8_CORE(x0, x1, x2, x3, x4, x5, x6, x7);
    if (n1) {
      CMULT(x1r, x1i, x1r, x1i, vbc(TW64R[n1 * 8 + 1]), vbc(TW64I[n1 * 8 + 1]));""",
"""    if (pf) {
      _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++) {
        PF1(pf + (n1 * 8 + q) * s); PF1(pf + PLANE + (n1 * 8 + q) * s);
      }
    }
    FFT8_CORE(x0, x1, x2, x3, x4, x5, x6, x7);
    if (n1) {
      CMULT(x1r, x1i, x1r, x1i, vbc(TW64R[n1 * 8 + 1]), vbc(TW64I[n1 * 8 + 1]));""")
src = src.replace("""    FFT8_CORE(x0, x1, x2, x3, x4, x5, x6, x7);
    STORE_X(re, im, x0r, x0i, (0 * 8 + k2) * s);""",
"""    if (pf) {
      _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++) {
        PF1(pf + 2 * PLANE + (k2 * 8 + q) * s); PF1(pf + 3 * PLANE + (k2 * 8 + q) * s);
      }
    }
    FFT8_CORE(x0, x1, x2, x3, x4, x5, x6, x7);
    STORE_X(re, im, x0r, x0i, (0 * 8 + k2) * s);""")

# 6. update call sites in DEF_ENGINE: sweepA pf=0, sweepB computes next pf
src = src.replace("""        FFTLINE(SZA, SZB, 8, 0, 0, 0);""", """        FFTLINE(SZA, SZB, 8, 0, 0, 0, 0);""")
src = src.replace("""      for (long zc = 0; zc < LP_; zc += 8) FFTLINE(br + zc, bi + zc, R_, 0, 0, 0);""",
"""      for (long zc = 0; zc < LP_; zc += 8) FFTLINE(br + zc, bi + zc, R_, 0, 0, 0, 0);""")
src = src.replace("""        long off = base + zc;                                                         \\
        FFTLINE(Xre + off, Xim + off, (long)SS_, 1, Cre + off, Cim + off);            \\""",
"""        long off = base + zc;                                                         \\
        long zc2 = zc + 8, y2 = y;                                                    \\
        if (zc2 >= LP_) { zc2 = 0; y2 = (y + 1 < L_) ? y + 1 : 0; }                   \\
        FFTLINE(Xre + off, Xim + off, (long)SS_, 1, Cre + off, Cim + off,             \\
                Xre + y2 * R_ + zc2);                                                 \\""")

# 7. fix buffer layout comment: ensure planes are PLANE doubles apart
src = src.replace("size_t one = 64l * 4616 * sizeof(double);   // max volume component (L=64 padded)",
                  "size_t one = PLANE * sizeof(double);   // max volume component (L=64 padded)")

# 8. unit test callers signature
for L in (6,8,13,17,23,36,45,64):
    src = src.replace(f"{{ fft{L}_line(r, i, s, d, a, b); }}", f"{{ fft{L}_line(r, i, s, d, a, b, 0); }}")
open('implementation.c','w').write(src)
print("ok")
