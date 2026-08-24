import subprocess, gen

HARNESS = r'''
#include <stdio.h>
#include <stdint.h>
#include <time.h>
static double now(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }
CODELET
static void caller(const double* ri, const double* ii, double* ro, double* io){
  FNAME(ri, ii, STRIDE, ro, io, STRIDE);
}
typedef void (*fnp)(const double*, const double*, double*, double*);
static volatile fnp FP = caller;
int main(){
  int N = NPOINTS;
  static double ri[70*8] __attribute__((aligned(64)));
  static double ii[70*8] __attribute__((aligned(64)));
  static double ro[70*8] __attribute__((aligned(64)));
  static double io[70*8] __attribute__((aligned(64)));
  for (int i = 0; i < N*8; i++){ ri[i] = 0.001*i - 0.1; ii[i] = 0.0005*i; }
  long R = 2000000;
  fnp f = FP;
  double t0 = now();
  for (long r = 0; r < R; r++) f(ri, ii, ro, io);
  double t1 = now();
  double s = 0; for (int i = 0; i < N*8; i++) s += ro[i] + io[i];
  printf("%8.3f ns/call %7.4f ns/pt (s=%g)\n", (t1-t0)/R*1e9, (t1-t0)/R/N/8*1e9, s);
}
'''

def bench(n, text, name, label):
    src = HARNESS.replace('CODELET', gen.PRELUDE + text).replace('NPOINTS', str(n)).replace('FNAME', name).replace('STRIDE', '8')
    open('mb.c','w').write(src)
    subprocess.run(['gcc','-O3','-march=native','-ffp-contract=fast','mb.c','-o','mb'], check=True)
    out = subprocess.run(['taskset','-c','0','./mb'], capture_output=True, text=True).stdout.strip()
    print(f"n={n:2d} {label:18s}: {out}")
