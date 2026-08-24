mkdir -p /tmp/bench && cd /tmp/bench && cat > fma.c <<'EOF'
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
int main(){
  // FMA throughput: 16 independent chains
  __m512d acc[16]; for(int i=0;i<16;i++) acc[i]=_mm512_set1_pd(1.0+i*1e-9);
  __m512d b=_mm512_set1_pd(1.0000001), c=_mm512_set1_pd(1e-12);
  long N=200000000;
  uint64_t t0=rdtscp();
  for(long i=0;i<N;i+=16){
    for(int k=0;k<16;k++) acc[k]=_mm512_fmadd_pd(acc[k],b,c);
  }
  uint64_t t1=rdtscp();
  double s=0; for(int i=0;i<16;i++) s+=acc[i][0];
  printf("FMA512: %.3f cycles per fma (sum %g)\n", (double)(t1-t0)/N, s);

  // div/sqrt throughput
  __m512d d[8]; for(int i=0;i<8;i++) d[i]=_mm512_set1_pd(1.5+i*0.1);
  __m512d one=_mm512_set1_pd(1.000000001);
  N=20000000;
  t0=rdtscp();
  for(long i=0;i<N;i+=8){ for(int k=0;k<8;k++) d[k]=_mm512_div_pd(one,_mm512_add_pd(d[k],one)); }
  t1=rdtscp();
  s=0; for(int i=0;i<8;i++) s+=d[i][0];
  printf("DIV512: %.3f cycles per div (sum %g)\n", (double)(t1-t0)/N, s);

  for(int i=0;i<8;i++) d[i]=_mm512_set1_pd(1.5+i*0.1);
  t0=rdtscp();
  for(long i=0;i<N;i+=8){ for(int k=0;k<8;k++) d[k]=_mm512_sqrt_pd(_mm512_add_pd(d[k],one)); }
  t1=rdtscp();
  s=0; for(int i=0;i<8;i++) s+=d[i][0];
  printf("SQRT512: %.3f cycles per sqrt (sum %g)\n", (double)(t1-t0)/N, s);

  // rcp14 + mul throughput (port check)
  for(int i=0;i<8;i++) d[i]=_mm512_set1_pd(1.5+i*0.1);
  t0=rdtscp();
  for(long i=0;i<N;i+=8){ for(int k=0;k<8;k++) d[k]=_mm512_rcp14_pd(_mm512_add_pd(d[k],one)); }
  t1=rdtscp();
  s=0; for(int i=0;i<8;i++) s+=d[i][0];
  printf("RCP14: %.3f cycles per rcp14+add (sum %g)\n", (double)(t1-t0)/N, s);

  // shuffle (vpermt2pd) throughput
  __m512i idx=_mm512_set_epi64(3,11,2,10,1,9,0,8);
  for(int i=0;i<8;i++) d[i]=_mm512_set1_pd(1.5+i*0.1);
  N=200000000;
  t0=rdtscp();
  for(long i=0;i<N;i+=8){ for(int k=0;k<8;k++) d[k]=_mm512_permutex2var_pd(d[k],idx,one); }
  t1=rdtscp();
  s=0; for(int i=0;i<8;i++) s+=d[i][0];
  printf("PERMT2: %.3f cycles per perm (sum %g)\n", (double)(t1-t0)/N, s);
  return 0;
}
EOF
gcc -O2 -march=native fma.c -o fma && taskset -c 0 ./fma