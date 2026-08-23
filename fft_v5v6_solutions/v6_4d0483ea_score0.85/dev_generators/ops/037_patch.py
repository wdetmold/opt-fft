src = open('implementation.c').read()
src = src.replace("""  _Pragma("GCC unroll 16") for (int j = 1; j <= H; j++) {                         \\
    v8 ar = VL(re + j * s), ai = VL(im + j * s);                                  \\""",
"""  _Pragma("GCC unroll 16") for (int j = 1; j <= H; j++) {                         \\
    if (pf) { PF1(pf + j * s); PF1(pf + PLANE + j * s);                           \\
              PF1(pf + (P - j) * s); PF1(pf + PLANE + (P - j) * s); }             \\
    v8 ar = VL(re + j * s), ai = VL(im + j * s);                                  \\""")
src = src.replace("""    int nk = H - k0 + 1; if (nk > BLK) nk = BLK;                                  \\
    _Pragma("GCC unroll 8") for (int t = 0; t < nk; t++) {                        \\""",
"""    if (pf) {                                                                     \\
      _Pragma("GCC unroll 8") for (int t = 0; t < BLK && k0 + t <= H; t++) {      \\
        PFC(pf + 2 * PLANE + (k0 + t) * s); PFC(pf + 3 * PLANE + (k0 + t) * s);   \\
        PFC(pf + 2 * PLANE + (P - k0 - t) * s); PFC(pf + 3 * PLANE + (P - k0 - t) * s); \\
      }                                                                           \\
    }                                                                             \\
    int nk = H - k0 + 1; if (nk > BLK) nk = BLK;                                  \\
    _Pragma("GCC unroll 8") for (int t = 0; t < nk; t++) {                        \\""")
open('implementation.c','w').write(src)
print("ok")
