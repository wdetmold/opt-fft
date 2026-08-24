import subprocess, sys, importlib
import gen
importlib.reload(gen)

HARNESS = r'''
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
static double now(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }
CODELET
typedef void (*fnp)(const double*, const double*, ptrdiff_t, double*, double*, ptrdiff_t);
static volatile fnp FP = FNAME;
int main(int argc, char** argv){
  int N = NPOINTS;
  static double ri[70*8] __attribute__((aligned(64)));
  static double ii[70*8] __attribute__((aligned(64)));
  static double ro[70*8] __attribute__((aligned(64)));
  static double io[70*8] __attribute__((aligned(64)));
  for (int i = 0; i < N*8; i++){ ri[i] = 0.001*i - 0.1; ii[i] = 0.0005*i; }
  long R = 2000000;
  double t0 = now();
  fnp f = FP;
  for (long r = 0; r < R; r++){
    f(ri, ii, 8, ro, io, 8);
  }
  double t1 = now();
  double s = 0; for (int i = 0; i < N*8; i++) s += ro[i] + io[i];
  printf("%.3f ns/call  %.4f ns/pt  (s=%g)\n", (t1-t0)/R*1e9, (t1-t0)/R/N/8*1e9, s);
  return 0;
}
'''

def bench(n, text, name):
    src = HARNESS.replace('CODELET', gen.PRELUDE + text).replace('NPOINTS', str(n)).replace('FNAME', name)
    open('mb.c','w').write(src)
    subprocess.run(['gcc','-O3','-march=native','-ffp-contract=fast','mb.c','-o','mb'], check=True)
    out = subprocess.run(['taskset','-c','0','./mb'], capture_output=True, text=True).stdout.strip()
    return out

if __name__ == '__main__':
    for n in (13, 17, 23, 36, 45, 64):
        t = gen.emit_codelet(n, 8)
        print(f"n={n:2d} baseline: {bench(n, t, f'fft{n}_w8')}")
