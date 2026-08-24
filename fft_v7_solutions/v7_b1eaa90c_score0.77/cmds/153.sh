cd /tmp/bench && cat > mix.c <<'EOF'
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
static double A[8*64] __attribute__((aligned(64)));
static double B[8*64] __attribute__((aligned(64)));
int main(){
  for(int i=0;i<8*64;i++){A[i]=1.0+i*1e-9;B[i]=2.0;}
  __m512d r0=_mm512_set1_pd(1.0),r1=_mm512_set1_pd(1.1),r2=_mm512_set1_pd(1.2),r3=_mm512_set1_pd(1.3);
  __m512d r4=_mm512_set1_pd(1.4),r5=_mm512_set1_pd(1.5),r6=_mm512_set1_pd(1.6),r7=_mm512_set1_pd(1.7);
  __m512d b=_mm512_set1_pd(1.0000001), c=_mm512_set1_pd(1e-12);
  long R=10000000; uint64_t t0,t1;
  // 8 FMA + 4 loads + 2 stores per iteration
  t0=rdtscp();
  for(long i=0;i<R;i++){
    __m512d l0=_mm512_load_pd(A+((i&31)*8)), l1=_mm512_load_pd(A+((i&31)*8+64&511));
    __m512d l2=_mm512_load_pd(A+((i&15)*16)), l3=_mm512_load_pd(A+((i&7)*24));
    r0=_mm512_fmadd_pd(r0,b,l0); r1=_mm512_fmadd_pd(r1,b,l1);
    r2=_mm512_fmadd_pd(r2,b,l2); r3=_mm512_fmadd_pd(r3,b,l3);
    r4=_mm512_fmadd_pd(r4,b,c); r5=_mm512_fmadd_pd(r5,b,c);
    r6=_mm512_fmadd_pd(r6,b,c); r7=_mm512_fmadd_pd(r7,b,c);
    _mm512_store_pd(B+((i&31)*8), r0);
    _mm512_store_pd(B+((i&15)*16+256&511), r1);
  }
  t1=rdtscp();
  printf("mix 8FMA+4L+2S: %.2f cyc/iter (ideal 4.0) -> %.2f 512b-uops/cyc\n",(double)(t1-t0)/R, 14.0*R/(t1-t0));
  double s=0; for(int i=0;i<64;i++)s+=B[i];
  printf("chk %g %g\n", s, r7[0]);
  return 0;
}
EOF
gcc -O2 -march=native mix.c -o mix && taskset -c 0 ./mix