cd /tmp/bench && cat > fma3.c <<'EOF'
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
static double now(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }
#define R(i) r##i = _mm512_fmadd_pd(r##i, b, c)
int main(){
  __m512d r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15;
  volatile double seed[16]; for(int i=0;i<16;i++) seed[i]=1.0+i*1e-6;
  r0=_mm512_set1_pd(seed[0]);r1=_mm512_set1_pd(seed[1]);r2=_mm512_set1_pd(seed[2]);r3=_mm512_set1_pd(seed[3]);
  r4=_mm512_set1_pd(seed[4]);r5=_mm512_set1_pd(seed[5]);r6=_mm512_set1_pd(seed[6]);r7=_mm512_set1_pd(seed[7]);
  r8=_mm512_set1_pd(seed[8]);r9=_mm512_set1_pd(seed[9]);r10=_mm512_set1_pd(seed[10]);r11=_mm512_set1_pd(seed[11]);
  r12=_mm512_set1_pd(seed[12]);r13=_mm512_set1_pd(seed[13]);r14=_mm512_set1_pd(seed[14]);r15=_mm512_set1_pd(seed[15]);
  __m512d b=_mm512_set1_pd(1.0000000001), c=_mm512_set1_pd(1e-15);
  long N=2000000000;
  uint64_t t0=rdtscp(); double w0=now();
  for(long i=0;i<N;i+=16){
    R(0);R(1);R(2);R(3);R(4);R(5);R(6);R(7);R(8);R(9);R(10);R(11);R(12);R(13);R(14);R(15);
  }
  uint64_t t1=rdtscp(); double w1=now();
  __m512d s=_mm512_add_pd(r0,r1); s=_mm512_add_pd(s,r2); s=_mm512_add_pd(s,r3);
  s=_mm512_add_pd(s,r4); s=_mm512_add_pd(s,r5);s=_mm512_add_pd(s,r6);s=_mm512_add_pd(s,r7);
  s=_mm512_add_pd(s,r8);s=_mm512_add_pd(s,r9);s=_mm512_add_pd(s,r10);s=_mm512_add_pd(s,r11);
  s=_mm512_add_pd(s,r12);s=_mm512_add_pd(s,r13);s=_mm512_add_pd(s,r14);s=_mm512_add_pd(s,r15);
  printf("FMA512 reg: %.3f tsc-cycles/fma; wall %.3f s -> %.2f GFMA512/s chk %g\n",
    (double)(t1-t0)/N, w1-w0, N/(w1-w0)/1e9, s[0]);
  // also ymm version for freq comparison
  __m256d q0,q1,q2,q3,q4,q5,q6,q7;
  q0=_mm256_set1_pd(seed[0]);q1=_mm256_set1_pd(seed[1]);q2=_mm256_set1_pd(seed[2]);q3=_mm256_set1_pd(seed[3]);
  q4=_mm256_set1_pd(seed[4]);q5=_mm256_set1_pd(seed[5]);q6=_mm256_set1_pd(seed[6]);q7=_mm256_set1_pd(seed[7]);
  __m256d bb=_mm256_set1_pd(1.0000000001), cc=_mm256_set1_pd(1e-15);
  t0=rdtscp(); w0=now();
  for(long i=0;i<N;i+=8){
    q0=_mm256_fmadd_pd(q0,bb,cc);q1=_mm256_fmadd_pd(q1,bb,cc);q2=_mm256_fmadd_pd(q2,bb,cc);q3=_mm256_fmadd_pd(q3,bb,cc);
    q4=_mm256_fmadd_pd(q4,bb,cc);q5=_mm256_fmadd_pd(q5,bb,cc);q6=_mm256_fmadd_pd(q6,bb,cc);q7=_mm256_fmadd_pd(q7,bb,cc);
  }
  t1=rdtscp(); w1=now();
  __m256d ss=_mm256_add_pd(_mm256_add_pd(q0,q1),_mm256_add_pd(q2,q3));
  ss=_mm256_add_pd(ss,_mm256_add_pd(_mm256_add_pd(q4,q5),_mm256_add_pd(q6,q7)));
  printf("FMA256 reg: %.3f tsc-cycles/fma; wall %.3f -> %.2f GFMA256/s chk %g\n",
    (double)(t1-t0)/N, w1-w0, N/(w1-w0)/1e9, ss[0]);
  return 0;
}
EOF
gcc -O2 -march=native fma3.c -o fma3 && grep -c "vfmadd132pd	%zmm" <(gcc -O2 -march=native -S fma3.c -o /dev/stdout) ; taskset -c 0 ./fma3 && taskset -c 0 ./fma3