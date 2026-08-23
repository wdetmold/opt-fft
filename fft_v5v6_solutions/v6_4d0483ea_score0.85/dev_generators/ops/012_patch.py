src = open('implementation.c').read()
src = src.replace("""// ---------------------------------------------------------------- codelet cores""",
"""#define PSTR(x) #x
#define PRAGMA_UNROLL(n) _Pragma(PSTR(GCC unroll n))
// ---------------------------------------------------------------- codelet cores""")
src = src.replace("#define PRIME_LINE(P, H, PHN, CT, ST, BLK)", "#define PRIME_LINE(P, H, PHN, CT, ST, BLK, JU)")
src = src.replace("""    for (int j = 0; j < H; j++) {                                                 \\
      v8 par = VL(SAR + j * 8), pai = VL(SAI + j * 8);                            \\""",
"""    PRAGMA_UNROLL(JU) for (int j = 0; j < H; j++) {                               \\
      v8 par = VL(SAR + j * 8), pai = VL(SAI + j * 8);                            \\""")
src = src.replace("PRIME_LINE(13, 6, PH13, C13, S13, 3)", "PRIME_LINE(13, 6, PH13, C13, S13, 3, JU13)")
src = src.replace("PRIME_LINE(17, 8, PH17, C17, S17, 4)", "PRIME_LINE(17, 8, PH17, C17, S17, 4, JU17)")
src = src.replace("PRIME_LINE(23, 11, PH23, C23, S23, 4)", "PRIME_LINE(23, 11, PH23, C23, S23, 4, JU23)")
src = src.replace("// ---------------------------------------------------------------- line FFTs",
"""#ifndef JU13
#define JU13 8
#endif
#ifndef JU17
#define JU17 1
#endif
#ifndef JU23
#define JU23 1
#endif
// ---------------------------------------------------------------- line FFTs""")
open('implementation.c','w').write(src)
