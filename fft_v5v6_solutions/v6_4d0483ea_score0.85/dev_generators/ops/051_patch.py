src = open('implementation.c').read()
old = """  SWEEPVIS void sweepB_##LN(void) {                                                     \\
    for (long y = 0; y < L_; y++) {                                                   \\
      long base = y * R_;                                                             \\
      for (long zc = 0; zc < LP_; zc += 8) {                                          \\
        long off = base + zc;                                                         \\
        long zc2 = zc + 8, y2 = y;                                                    \\
        if (zc2 >= LP_) { zc2 = 0; y2 = (y + 1 < L_) ? y + 1 : 0; }                   \\
        FFTLINE(Xre + off, Xim + off, (long)SS_, 1, Cre + off, Cim + off,             \\
                Xre + y2 * R_ + zc2);                                                 \\
      }                                                                               \\
    }                                                                                 \\
  }"""
new = """  SWEEPVIS void sweepB_##LN(void) {                                                     \\
    _Pragma(PSTR(GCC unroll 1)) for (long zc = 0; zc < LP_; zc += 8) {                \\
      _Pragma(PSTR(GCC unroll 1)) for (long y = 0; y < L_; y++) {                     \\
        long off = y * R_ + zc;                                                       \\
        long y2 = y + 1, zc2 = zc;                                                    \\
        if (y2 >= L_) { y2 = 0; zc2 = (zc + 8 < LP_) ? zc + 8 : 0; }                  \\
        FFTLINE(Xre + off, Xim + off, (long)SS_, 1, Cre + off, Cim + off,             \\
                Xre + y2 * R_ + zc2);                                                 \\
      }                                                                               \\
    }                                                                                 \\
  }"""
import re
assert old in src
open('/tmp/zcouter.patch','w').write('1')
src2 = src.replace(old, new)
open('implementation.c','w').write(src2)
print("patched to zc-outer")
