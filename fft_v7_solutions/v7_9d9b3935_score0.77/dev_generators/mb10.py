import subprocess, gen

H = r'''
#include <stdio.h>
#include <stdint.h>
#include <time.h>
static double now(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }
CODELET
static void caller(const double* ri, const double* ii, double* ro, double* io){
  FNAME(ri, ii, WIDTH, ro, io, WIDTH);
}
typedef void (*fnp)(const double*, const double*, double*, double*);
static volatile fnp FP = caller;
int main(){
  int N = NPOINTS;
  static double ri[64*16] __attribute__((aligned(64)));
  static double ii[64*16] __attribute__((aligned(64)));
  static double ro[64*16] __attribute__((aligned(64)));
  static double io[64*16] __attribute__((aligned(64)));
  for (int i = 0; i < N*WIDTH; i++){ ri[i] = 0.001*i - 0.1; ii[i] = 0.0005*i; }
  long R = 1000000;
  fnp f = FP;
  double t0 = now();
  for (long r = 0; r < R; r++) f(ri, ii, ro, io);
  double t1 = now();
  double s = 0; for (int i = 0; i < N*WIDTH; i++) s += ro[i] + io[i];
  printf("%8.3f ns/call %8.4f ns/pt-lane (s=%g)\n", (t1-t0)/R*1e9, (t1-t0)/R/N/WIDTH*1e9, s);
}
'''
def bench(n, text, name, label, width):
    src = H.replace('CODELET', gen.PRELUDE + text).replace('NPOINTS', str(n)).replace('FNAME', name).replace('WIDTH', str(width))
    open('mb.c','w').write(src)
    subprocess.run(['gcc','-O3','-march=native','-ffp-contract=fast','-fno-stack-protector','mb.c','-o','mb'], check=True)
    out = subprocess.run(['taskset','-c','0','./mb'], capture_output=True, text=True).stdout.strip()
    print(f"n={n:2d} {label:22s}: {out}")

bench(23, gen.emit_rader(23, 8, maxblock=7), 'fft23_w8', 'rader w8', 8)
bench(23, gen.emit_rader(23, 16, maxblock=7), 'fft23_w16', 'rader w16', 16)
bench(17, gen.emit_rader(17, 8, maxblock=5), 'fft17_w8', 'rader w8', 8)
bench(17, gen.emit_rader(17, 16, maxblock=5), 'fft17_w16', 'rader w16', 16)
bench(13, gen.emit_rader(13, 8, maxblock=5), 'fft13_w8', 'rader w8', 8)
bench(13, gen.emit_rader(13, 16, maxblock=5), 'fft13_w16', 'rader w16', 16)
bench(64, gen.emit_codelet_staged(64, 8, ('ct',8,8)), 'fft64_w8', 'ct88 w8', 8)
bench(64, gen.emit_codelet_staged(64, 16, ('ct',8,8)), 'fft64_w16', 'ct88 w16', 16)
