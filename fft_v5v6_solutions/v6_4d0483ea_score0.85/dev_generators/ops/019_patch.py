src = open('implementation.c').read()
src = src.replace("static void phase1_64(long g, double *dre, double *dim) {",
"""#ifdef EXPOSE_SWEEPS
double PH1T, PH2T, SLCT, ST1T;
#include <time.h>
static double now_(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec + 1e-9*ts.tv_nsec; }
#define TMARK(var, expr) { double _t0=now_(); expr; var += now_()-_t0; }
#else
#define TMARK(var, expr) expr;
#endif
static void phase1_64(long g, double *dre, double *dim) {""")
src = src.replace("""  for (long j = 0; j < 8; j++) slice_zy_64(STGre + j * SS64, STGim + j * SS64);
  stage1_64(g, dre, dim);""",
"""  TMARK(SLCT, for (long j = 0; j < 8; j++) slice_zy_64(STGre + j * SS64, STGim + j * SS64))
  TMARK(ST1T, stage1_64(g, dre, dim))""")
src = src.replace("""          for (long g = 0; g < 8; g++) {
            phase2_64(g, Sr, Si);""",
"""          for (long g = 0; g < 8; g++) {
            TMARK(PH2T, phase2_64(g, Sr, Si))""")
open('implementation.c','w').write(src)
