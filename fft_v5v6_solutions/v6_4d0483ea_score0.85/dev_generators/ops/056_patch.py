src = open('implementation.c').read()

# new engine macro: global row tiling fused with per-slice passY; SS == L*R
src = src.replace("#define DEF_SWEEPB(LN, L_, LP_, R_, LPS_, SS_, FFTLINE)",
"""#define DEF_SWEEPA_G(LN, L_, LP_, R_, MP_, FFTLINE)                                   \\
  SWEEPVIS void sweepA_##LN(void) {                                                   \\
    long t = 0;                                                                       \\
    for (long x = 0; x < L_; x++) {                                                   \\
      long lim = (x + 1) * L_;                                                        \\
      for (; t < lim; t += 8) {                                                       \\
        tile_load(Xre + t * R_, R_, SZA, LP_ / 8);                                    \\
        tile_load(Xim + t * R_, R_, SZB, LP_ / 8);                                    \\
        FFTLINE(SZA, SZB, 8, 0, 0, 0, 0);                                             \\
        tile_store(Xre + t * R_, R_, SZA, LP_ / 8, L_);                               \\
        tile_store(Xim + t * R_, R_, SZB, LP_ / 8, L_);                               \\
      }                                                                               \\
      double *br = Xre + x * (long)(L_ * R_), *bi = Xim + x * (long)(L_ * R_);        \\
      for (long zc = 0; zc < LP_; zc += 8) FFTLINE(br + zc, bi + zc, R_, 0, 0, 0, 0); \\
    }                                                                                 \\
  }

#define DEF_SWEEPB(LN, L_, LP_, R_, LPS_, SS_, FFTLINE)""")

# switch 13,17,23,45 to global tiling; SS becomes L*R
src = src.replace("DEF_SWEEPA(13, 13, 16, 16, 16, 264, fft13_line)\nDEF_SWEEPB(13, 13, 16, 16, 16, 264, fft13_line)",
                  "DEF_SWEEPA_G(13, 13, 16, 16, 176, fft13_line)\nDEF_SWEEPB(13, 13, 16, 16, 16, 208, fft13_line)")
src = src.replace("DEF_SWEEPA(17, 17, 24, 24, 24, 584, fft17_line)\nDEF_SWEEPB(17, 17, 24, 24, 24, 584, fft17_line)",
                  "DEF_SWEEPA_G(17, 17, 24, 24, 296, fft17_line)\nDEF_SWEEPB(17, 17, 24, 24, 24, 408, fft17_line)")
src = src.replace("DEF_SWEEPA(23, 23, 24, 24, 24, 584, fft23_line)\nDEF_SWEEPB(23, 23, 24, 24, 24, 584, fft23_line)",
                  "DEF_SWEEPA_G(23, 23, 24, 24, 536, fft23_line)\nDEF_SWEEPB(23, 23, 24, 24, 24, 552, fft23_line)")
src = src.replace("DEF_SWEEPA(45, 45, 48, 48, 48, 2312, fft45_line)\nDEF_SWEEPB(45, 45, 48, 48, 48, 2312, fft45_line)",
                  "DEF_SWEEPA_G(45, 45, 48, 48, 2032, fft45_line)\nDEF_SWEEPB(45, 45, 48, 48, 48, 2160, fft45_line)")
# driver SS updates
src = src.replace("    RUN_CASE_SOA(13, 13, 16, 264, 8 * (169 + 1))", "    RUN_CASE_SOA(13, 13, 16, 208, 8 * (169 + 1))")
src = src.replace("    RUN_CASE_SOA(17, 17, 24, 584, 8 * (289 + 1))", "    RUN_CASE_SOA(17, 17, 24, 408, 8 * (289 + 1))")
src = src.replace("    RUN_CASE(23, 23, 24, 584)", "    RUN_CASE(23, 23, 24, 552)")
src = src.replace("    RUN_CASE(45, 45, 48, 2312)", "    RUN_CASE(45, 45, 48, 2160)")
open('implementation.c','w').write(src)
print("ok")
