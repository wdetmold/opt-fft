src = open('implementation.c').read()
start = src.index("// software-pipelined: slice x+1's tiles interleave")
end = src.index("\n\n", start)
new = """#define DEF_SWEEPA(LN, L_, LP_, R_, LPS_, SS_, FFTLINE)                               \\
  SWEEPVIS void sweepA_##LN(void) {                                                   \\
    for (long x = 0; x < L_; x++) {                                                   \\
      double *br = Xre + x * (long)SS_, *bi = Xim + x * (long)SS_;                    \\
      for (long y0 = 0; y0 < LPS_; y0 += 8) {                                         \\
        tile_load(br + y0 * R_, R_, SZA, LP_ / 8);                                    \\
        tile_load(bi + y0 * R_, R_, SZB, LP_ / 8);                                    \\
        FFTLINE(SZA, SZB, 8, 0, 0, 0, 0);                                             \\
        tile_store(br + y0 * R_, R_, SZA, LP_ / 8, L_);                               \\
        tile_store(bi + y0 * R_, R_, SZB, LP_ / 8, L_);                               \\
      }                                                                               \\
      for (long zc = 0; zc < LP_; zc += 8) FFTLINE(br + zc, bi + zc, R_, 0, 0, 0, 0); \\
    }                                                                                 \\
  }"""
src = src[:start] + new + src[end:]
open('implementation.c','w').write(src)
print("reverted")
