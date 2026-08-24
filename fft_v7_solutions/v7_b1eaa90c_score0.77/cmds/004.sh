cd /tmp/bench && cat > fma2.c <<'EOF'
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
static double now(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }
#define R(i) r##i = _mm512_fmadd_pd(r##i, b, c)
int main(){
  __m512d r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;
  r0=r1=r2=r3=r4=r5=r6=r7=r8=r9=r10=r11=r12=r13=r14=r15=_mm512_set1_pd(1.0);
  __m512d b=_mm512_set1_pd(1.0000000001), c=_mm512_set1_pd(1e-15);
  long N=1000000000;
  uint64_t t0=rdtscp(); double w0=now();
  for(long i=0;i<N;i+=16){
    R(0);R(1);R(2);R(3);R(4);R(5);R(6);R(7);R(8);R(9);R(10);R(11);R(12);R(13);R(14);R(15);
  }
  uint64_t t1=rdtscp(); double w1=now();
  __m512d s=_mm512_add_pd(r0,r1); s=_mm512_add_pd(s,r2); s=_mm512_add_pd(s,r3);
  s=_mm512_add_pd(s,r4); s=_mm512_add_pd(s,r5);s=_mm512_add_pd(s,r6);s=_mm512_add_pd(s,r7);
  s=_mm512_add_pd(s,r8);s=_mm512_add_pd(s,r9);s=_mm512_add_pd(s,r10);s=_mm512_add_pd(s,r11);
  s=_mm512_add_pd(s,r12);s=_mm512_add_pd(s,r13);s=_mm512_add_pd(s,r14);s=_mm512_add_pd(s,r15);
  printf("FMA512 reg: %.3f tsc-cycles per fma; wall %.3f s -> %.2f GFMA/s (%.1f Gflop/s dp) chk %g\n",
    (double)(t1-t0)/N, w1-w0, N/(w1-w0)/1e9, 16.0*N/(w1-w0)/1e9, s[0]);
  return 0;
}
EOF
gcc -O2 -march=native fma2.c -o fma2 && taskset -c 0 ./fma2 && taskset -c 0 ./fma2