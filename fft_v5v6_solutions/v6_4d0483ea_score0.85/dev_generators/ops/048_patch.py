src = open('implementation.c').read()
src = src.replace("""  _Pragma("GCC unroll 16") for (int j = 1; j <= H; j++) {                         \\
    v8 ar = VL(re + j * s), ai = VL(im + j * s);                                  \\""",
"""  PRAGMA_UNROLL(JP) for (int j = 1; j <= H; j++) {                                \\
    v8 ar = VL(re + j * s), ai = VL(im + j * s);                                  \\""")
src = src.replace("#define PRIME_LINE(P, H, PHN, CT, ST, BLK, JU)", "#define PRIME_LINE(P, H, PHN, CT, ST, BLK, JU, JP)")
src = src.replace("PRIME_LINE(17, 8, PH17, C17, S17, BLK17, JU17)", "PRIME_LINE(17, 8, PH17, C17, S17, BLK17, JU17, JP17)")
src = src.replace("PRIME_LINE(23, 11, PH23, C23, S23, BLK23, JU23)", "PRIME_LINE(23, 11, PH23, C23, S23, BLK23, JU23, JP23)")
src = src.replace("#ifndef BLK17", "#ifndef JP17\n#define JP17 16\n#endif\n#ifndef JP23\n#define JP23 16\n#endif\n#ifndef BLK17")
src = src.replace("#ifndef UTILE\n#define UTILE 16\n#endif", "#ifndef UTILE\n#define UTILE 1\n#endif")
open('implementation.c','w').write(src)
print("ok")
