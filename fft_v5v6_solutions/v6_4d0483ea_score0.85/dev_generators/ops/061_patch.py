import re
src = open('implementation.c').read()
start = src.index("#define DEF_SWEEPA(LN, L_, LP_, R_, LPS_, SS_, FFTLINE)")
end = src.index("\n\n", start)
old = src[start:end]
new = """// software-pipelined: slice x+1's tiles interleave with slice x's row-axis pass
#define DEF_SWEEPA(LN, L_, LP_, R_, LPS_, SS_, FFTLINE)                               \\
  static inline __attribute__((always_inline)) void tileA_##LN(double *br, double *bi, long y0) { \\
    tile_load(br + y0 * R_, R_, SZA, LP_ / 8);                                        \\
    tile_load(bi + y0 * R_, R_, SZB, LP_ / 8);                                        \\
    FFTLINE(SZA, SZB, 8, 0, 0, 0, 0);                                                 \\
    tile_store(br + y0 * R_, R_, SZA, LP_ / 8, L_);                                   \\
    tile_store(bi + y0 * R_, R_, SZB, LP_ / 8, L_);                                   \\
  }                                                                                   \\
  SWEEPVIS void sweepA_##LN(void) {                                                   \\
    for (long y0 = 0; y0 < LPS_; y0 += 8) tileA_##LN(Xre, Xim, y0);                   \\
    for (long x = 0; x < L_; x++) {                                                   \\
      double *br = Xre + x * (long)SS_, *bi = Xim + x * (long)SS_;                    \\
      double *nr = br + SS_, *ni = bi + SS_;                                          \\
      long k = 0;                                                                     \\
      for (long zc = 0; zc < LP_; zc += 8, k += 8) {                                  \\
        if (x + 1 < L_) tileA_##LN(nr, ni, k);                                        \\
        FFTLINE(br + zc, bi + zc, R_, 0, 0, 0, 0);                                    \\
      }                                                                               \\
    }                                                                                 \\
  }"""
src = src[:start] + new + src[end:]
open('implementation.c','w').write(src)
print("replaced")
