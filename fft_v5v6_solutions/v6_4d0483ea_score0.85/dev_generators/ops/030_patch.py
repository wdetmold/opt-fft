src = open('implementation.c').read()
# 1. gate C prefetches on domap in 36/45/64 codelets
src = src.replace("""    if (pf) {
      _Pragma("GCC unroll 4") for (int q = 0; q < 4; q++) {
        PFC(pf + 2 * PLANE + (q2 * 4 + q) * s); PFC(pf + 3 * PLANE + (q2 * 4 + q) * s);
      }
    }""","""    if (pf && domap) {
      _Pragma("GCC unroll 4") for (int q = 0; q < 4; q++) {
        PFC(pf + 2 * PLANE + (q2 * 4 + q) * s); PFC(pf + 3 * PLANE + (q2 * 4 + q) * s);
      }
    }""")
src = src.replace("""    if (pf) {
      _Pragma("GCC unroll 5") for (int q = 0; q < 5; q++) {
        PFC(pf + 2 * PLANE + (q2 * 5 + q) * s); PFC(pf + 3 * PLANE + (q2 * 5 + q) * s);
      }
    }""","""    if (pf && domap) {
      _Pragma("GCC unroll 5") for (int q = 0; q < 5; q++) {
        PFC(pf + 2 * PLANE + (q2 * 5 + q) * s); PFC(pf + 3 * PLANE + (q2 * 5 + q) * s);
      }
    }""")
src = src.replace("""    if (pf) {
      _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++) {
        PFC(pf + 2 * PLANE + (k2 * 8 + q) * s); PFC(pf + 3 * PLANE + (k2 * 8 + q) * s);
      }
    }
    FFT8_CORE(x0, x1, x2, x3, x4, x5, x6, x7);
    STORE_X(re, im, x0r, x0i, (0 * 8 + k2) * s);""","""    if (pf && domap) {
      _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++) {
        PFC(pf + 2 * PLANE + (k2 * 8 + q) * s); PFC(pf + 3 * PLANE + (k2 * 8 + q) * s);
      }
    }
    FFT8_CORE(x0, x1, x2, x3, x4, x5, x6, x7);
    STORE_X(re, im, x0r, x0i, (0 * 8 + k2) * s);""")
# 2. passY with next-slice prefetch
src = src.replace("""      for (long zc = 0; zc < LP_; zc += 8) FFTLINE(br + zc, bi + zc, R_, 0, 0, 0, 0);    \\""",
"""      for (long zc = 0; zc < LP_; zc += 8)                                             \\
        FFTLINE(br + zc, bi + zc, R_, 0, 0, 0, br + SS_ + zc);                          \\""")
open('implementation.c','w').write(src)
print("ok")
