src = open('implementation.c').read()
old = """      for (long zc = 0; zc < LP_; zc += 8) FFTLINE(br + zc, bi + zc, R_, 0, 0, 0, 0); \\
    }                                                                                 \\
  }"""
new = """      for (long zc = 0; zc < LP_; zc += 8) {                                          \\
        FFTLINE(br + zc, bi + zc, R_, 0, 0, 0, 0);                                    \\
        if (x + 1 < L_) {           /* stage next slice into L2 during compute */     \\
          long chunk = (LPS_ * R_ * 8) / (LP_ / 8);                                   \\
          const char *p0 = (const char *)(br + SS_) + (zc / 8) * chunk;               \\
          const char *p1 = (const char *)(bi + SS_) + (zc / 8) * chunk;               \\
          for (long q = 0; q < chunk; q += 64) {                                      \\
            _mm_prefetch(p0 + q, _MM_HINT_T1);                                        \\
            _mm_prefetch(p1 + q, _MM_HINT_T1);                                        \\
          }                                                                           \\
        }                                                                             \\
      }                                                                               \\
    }                                                                                 \\
  }"""
assert old in src
src = src.replace(old, new)
open('implementation.c','w').write(src)
print("ok")
