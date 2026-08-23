src = open('implementation.c').read()
src = src.replace("DEF_SWEEPA(36, 36, 40, 40, 40, 1608, fft36_line)\nDEF_SWEEPB(36, 36, 40, 40, 40, 1608, fft36_line)",
                  "DEF_SWEEPA_G(36, 36, 40, 40, 1296, fft36_line)\nDEF_SWEEPB(36, 36, 40, 40, 40, 1440, fft36_line)")
src = src.replace("    RUN_CASE(36, 36, 40, 1608)", "    RUN_CASE(36, 36, 40, 1440)")
open('implementation.c','w').write(src)
