import subprocess, gen

H = r'''
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
  static double ri[64*66] __attribute__((aligned(64)));
  static double ii[64*66] __attribute__((aligned(64)));
  static double ro[64*66] __attribute__((aligned(64)));
  static double io[64*66] __attribute__((aligned(64)));
  for (int i = 0; i < N*STRIDE; i++){ ri[i] = 0.001*i - 0.1; ii[i] = 0.0005*i; }
  long R = 1000000;
  fnp f = FP;
  double t0 = now();
  for (long r = 0; r < R; r++) f(ri, ii, ro, io);
  double t1 = now();
  double s = 0; for (int i = 0; i < N*8; i++) s += ro[i] + io[i];
  printf("%8.3f ns/call (s=%g)\n", (t1-t0)/R*1e9, s);
}
'''
def bench(n, text, name, label, stride):
    src = H.replace('CODELET', gen.PRELUDE + text).replace('NPOINTS', str(n)).replace('FNAME', name).replace('STRIDE', str(stride))
    open('mb.c','w').write(src)
    subprocess.run(['gcc','-O3','-march=native','-ffp-contract=fast','-fno-stack-protector','mb.c','-o','mb'], check=True)
    out = subprocess.run(['taskset','-c','0','./mb'], capture_output=True, text=True).stdout.strip()
    print(f"n={n:2d} {label:24s} stride={stride:3d}: {out}")

t59 = gen.emit_codelet_staged(45, 8, ('pfa',5,9))
t95 = gen.emit_codelet_staged(45, 8, ('pfa',9,5))
bench(45, t59, 'fft45_w8', 'staged pfa59', 8)
bench(45, t59, 'fft45_w8', 'staged pfa59', 45)
bench(45, t95, 'fft45_w8', 'staged pfa95', 45)
t64 = gen.emit_codelet_staged(64, 8, ('ct',8,8))
bench(64, t64, 'fft64_w8', 'staged ct88', 8)
bench(64, t64, 'fft64_w8', 'staged ct88', 64)
t36 = gen.emit_codelet_staged(36, 8, ('pfa',4,9))
bench(36, t36, 'fft36_w8', 'staged pfa49', 8)
bench(36, t36, 'fft36_w8', 'staged pfa49', 36)
