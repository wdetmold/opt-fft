src = open('implementation.c').read()
src = src.replace("""  STORE_X(re, im, sumr, sumi, 0);                                                 \\
  for (int k0 = 1; k0 <= H; k0 += BLK) {                                          \\""",
"""  STORE_X(re, im, sumr, sumi, 0);                                                 \\
  _Pragma("GCC unroll 4") for (int k0 = 1; k0 <= H; k0 += BLK) {                  \\""")
open('implementation.c','w').write(src)
