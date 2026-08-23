src = open('implementation.c').read()
src = src.replace("""      long off = y * R64 + zc;
      v8 x0r = VL(STGre + 0 * SS64 + off), x0i = VL(STGim + 0 * SS64 + off);""",
"""      long off = y * R64 + zc;
      long off2 = (zc + 8 < LP64) ? off + 8 : (y + 1) * R64;   // next step
      _Pragma("GCC unroll 8") for (int q = 0; q < 8; q++) {
        __builtin_prefetch(dre + (8 * q + g) * SS64 + off2, 1, 3);
        __builtin_prefetch(dim + (8 * q + g) * SS64 + off2, 1, 3);
      }
      v8 x0r = VL(STGre + 0 * SS64 + off), x0i = VL(STGim + 0 * SS64 + off);""")
open('implementation.c','w').write(src)
