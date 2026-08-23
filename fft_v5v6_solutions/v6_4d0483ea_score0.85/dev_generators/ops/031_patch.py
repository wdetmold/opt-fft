src = open('implementation.c').read()
src = src.replace("""      for (long zc = 0; zc < LP_; zc += 8)                                             \\
        FFTLINE(br + zc, bi + zc, R_, 0, 0, 0, br + SS_ + zc);                          \\""",
"""      for (long zc = 0; zc < LP_; zc += 8) FFTLINE(br + zc, bi + zc, R_, 0, 0, 0, 0);    \\""")
open('implementation.c','w').write(src)
print("reverted")
