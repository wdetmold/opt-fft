src = open('implementation.c').read()
src = src.replace("""  _Pragma("GCC unroll 16") for (int j = 1; j <= H; j++) {                         \\
    if (pf) {                                                                     \\
      PF1(pf + j * s); PF1(pf + PLANE + j * s);                                   \\
      PF1(pf + (P - j) * s); PF1(pf + PLANE + (P - j) * s);                       \\
      PFC(pf + 2 * PLANE + j * s); PFC(pf + 3 * PLANE + j * s);                   \\
      PFC(pf + 2 * PLANE + (P - j) * s); PFC(pf + 3 * PLANE + (P - j) * s);       \\
    }                                                                             \\
    v8 ar = VL(re + j * s), ai = VL(im + j * s);                                  \\""",
"""  _Pragma("GCC unroll 16") for (int j = 1; j <= H; j++) {                         \\
    v8 ar = VL(re + j * s), ai = VL(im + j * s);                                  \\""")
open('implementation.c','w').write(src)
