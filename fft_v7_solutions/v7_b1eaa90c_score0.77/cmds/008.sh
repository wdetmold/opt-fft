cd /tmp/bench && cat > dsq.c <<'EOF'
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
volatile double vs[8]={1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2};
#define INIT __m512d d0=_mm512_set1_pd(vs[0]),d1=_mm512_set1_pd(vs[1]),d2=_mm512_set1_pd(vs[2]),d3=_mm512_set1_pd(vs[3]),d4=_mm512_set1_pd(vs[4]),d5=_mm512_set1_pd(vs[5]),d6=_mm512_set1_pd(vs[6]),d7=_mm512_set1_pd(vs[7]);
#define SUM (d0[0]+d1[0]+d2[0]+d3[0]+d4[0]+d5[0]+d6[0]+d7[0])
int main(){
  long N=80000000; uint64_t t0,t1;
  { INIT; __m512d one=_mm512_set1_pd(1.0000000001);
    t0=rdtscp();
    for(long i=0;i<N;i+=8){ d0=_mm512_div_pd(one,d0);d1=_mm512_div_pd(one,d1);d2=_mm512_div_pd(one,d2);d3=_mm512_div_pd(one,d3);
      d4=_mm512_div_pd(one,d4);d5=_mm512_div_pd(one,d5);d6=_mm512_div_pd(one,d6);d7=_mm512_div_pd(one,d7);}
    t1=rdtscp(); printf("DIV512 tput: %.2f c (chk %g)\n",(double)(t1-t0)/N, SUM); }
  { INIT;
    t0=rdtscp();
    for(long i=0;i<N;i+=8){ d0=_mm512_sqrt_pd(d0);d1=_mm512_sqrt_pd(d1);d2=_mm512_sqrt_pd(d2);d3=_mm512_sqrt_pd(d3);
      d4=_mm512_sqrt_pd(d4);d5=_mm512_sqrt_pd(d5);d6=_mm512_sqrt_pd(d6);d7=_mm512_sqrt_pd(d7);
      d0=_mm512_add_pd(d0,_mm512_set1_pd(1.0)); d1=_mm512_add_pd(d1,_mm512_set1_pd(1.01)); d2=_mm512_add_pd(d2,_mm512_set1_pd(1.02)); d3=_mm512_add_pd(d3,_mm512_set1_pd(1.03));
      d4=_mm512_add_pd(d4,_mm512_set1_pd(1.04)); d5=_mm512_add_pd(d5,_mm512_set1_pd(1.05)); d6=_mm512_add_pd(d6,_mm512_set1_pd(1.06)); d7=_mm512_add_pd(d7,_mm512_set1_pd(1.07)); }
    t1=rdtscp(); printf("SQRT512 tput(+add): %.2f c (chk %g)\n",(double)(t1-t0)/N, SUM); }
  { INIT;
    t0=rdtscp();
    for(long i=0;i<N;i+=8){ d0=_mm512_rcp14_pd(d0);d1=_mm512_rcp14_pd(d1);d2=_mm512_rcp14_pd(d2);d3=_mm512_rcp14_pd(d3);
      d4=_mm512_rcp14_pd(d4);d5=_mm512_rcp14_pd(d5);d6=_mm512_rcp14_pd(d6);d7=_mm512_rcp14_pd(d7);}
    t1=rdtscp(); printf("RCP14 tput: %.2f c (chk %g)\n",(double)(t1-t0)/N, SUM); }
  { INIT; __m512i ix=_mm512_set_epi64(3,11,2,10,1,9,0,8); __m512d e=_mm512_set1_pd(2.0);
    t0=rdtscp();
    for(long i=0;i<N;i+=8){ d0=_mm512_permutex2var_pd(d0,ix,e);d1=_mm512_permutex2var_pd(d1,ix,e);d2=_mm512_permutex2var_pd(d2,ix,e);d3=_mm512_permutex2var_pd(d3,ix,e);
      d4=_mm512_permutex2var_pd(d4,ix,e);d5=_mm512_permutex2var_pd(d5,ix,e);d6=_mm512_permutex2var_pd(d6,ix,e);d7=_mm512_permutex2var_pd(d7,ix,e);}
    t1=rdtscp(); printf("PERMT2 tput: %.2f c (chk %g)\n",(double)(t1-t0)/N, SUM); }
  { INIT; __m512d b=_mm512_set1_pd(1.0000000001), c=_mm512_set1_pd(1e-15);
    t0=rdtscp();
    for(long i=0;i<N;i+=8){ // mixed: 1 sqrt + 1 div + 6 fma
      d0=_mm512_sqrt_pd(d0); d1=_mm512_div_pd(b,d1);
      d2=_mm512_fmadd_pd(d2,b,c);d3=_mm512_fmadd_pd(d3,b,c);d4=_mm512_fmadd_pd(d4,b,c);
      d5=_mm512_fmadd_pd(d5,b,c);d6=_mm512_fmadd_pd(d6,b,c);d7=_mm512_fmadd_pd(d7,b,c);
      d0=_mm512_add_pd(d0,_mm512_set1_pd(0.5)); }
    t1=rdtscp(); printf("MIX sqrt+div+6fma per 8: %.2f c (chk %g)\n",(double)(t1-t0)/N, SUM); }
  return 0;
}
EOF
gcc -O2 -march=native dsq.c -o dsq && taskset -c 0 ./dsq