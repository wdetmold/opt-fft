src = open('implementation.c').read()
# The edit landed inside DEF_SWEEPA_G (its passY loop). Restore _G's simple loop,
# then apply the prefetch variant inside DEF_SWEEPA only.
g_start = src.index("#define DEF_SWEEPA_G(LN, L_, LP_, R_, MP_, FFTLINE)")
g_end = src.index("#define DEF_SWEEPB(")
gblock = src[g_start:g_end]
new_g = gblock.replace("""      double *br = Xre + x * (long)(L_ * R_), *bi = Xim + x * (long)(L_ * R_);        \\
      for (long zc = 0; zc < LP_; zc += 8) {                                          \\
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
  }""",
"""      double *br = Xre + x * (long)(L_ * R_), *bi = Xim + x * (long)(L_ * R_);        \\
      for (long zc = 0; zc < LP_; zc += 8) FFTLINE(br + zc, bi + zc, R_, 0, 0, 0, 0); \\
    }                                                                                 \\
  }""")
src = src[:g_start] + new_g + src[g_end:]
# now patch DEF_SWEEPA (the one with LPS_/SS_ params)
a_start = src.index("#define DEF_SWEEPA(LN, L_, LP_, R_, LPS_, SS_, FFTLINE)")
a_end = src.index("#define DEF_SWEEPA_G")
ablock = src[a_start:a_end]
new_a = ablock.replace("""      for (long zc = 0; zc < LP_; zc += 8) FFTLINE(br + zc, bi + zc, R_, 0, 0, 0, 0); \\
    }                                                                                 \\
  }""",
"""      for (long zc = 0; zc < LP_; zc += 8) {                                          \\
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
  }""")
src = src[:a_start] + new_a + src[a_end:]
open('implementation.c','w').write(src)
print("ok")
