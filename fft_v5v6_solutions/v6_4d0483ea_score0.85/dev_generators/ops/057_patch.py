src = open('implementation.c').read()
src = src.replace("DEF_SWEEPA_G(45, 45, 48, 48, 2032, fft45_line)\nDEF_SWEEPB(45, 45, 48, 48, 48, 2160, fft45_line)",
                  "DEF_SWEEPA(45, 45, 48, 48, 48, 2312, fft45_line)\nDEF_SWEEPB(45, 45, 48, 48, 48, 2312, fft45_line)")
src = src.replace("    RUN_CASE(45, 45, 48, 2160)", "    RUN_CASE(45, 45, 48, 2312)")
open('implementation.c','w').write(src)
print("ok")
