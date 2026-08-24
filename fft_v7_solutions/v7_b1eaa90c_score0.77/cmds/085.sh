cd /workdir/dev && cat > prof16.c <<'EOF'
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
typedef double V __attribute__((vector_size(64), aligned(64)));
#define VC(x) ((V){(x),(x),(x),(x),(x),(x),(x),(x)})
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
static V ur[12], ui[12], vr[12], vi[12];
static const double tabc[11*11] = {1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11};
static const double tabs[11*11] = {1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11, 1,2,3,4,5,6,7,8,9,10,11};
V outr[22], outi[22];
__attribute__((noinline)) void kloop(void){
  const double *c1 = tabc-1, *s1 = tabs-1;
  const double *c2 = c1+11, *s2 = s1+11;
  const double *c3 = c2+11, *s3 = s2+11;
  const double *c4 = c3+11, *s4 = s3+11;
  V er1=VC(0.1),ei1=VC(0.1),sr1=VC(0.0),si1=VC(0.0);
  V er2=VC(0.2),ei2=VC(0.1),sr2=VC(0.0),si2=VC(0.0);
  V er3=VC(0.3),ei3=VC(0.1),sr3=VC(0.0),si3=VC(0.0);
  V er4=VC(0.4),ei4=VC(0.1),sr4=VC(0.0),si4=VC(0.0);
  V er5=VC(0.5),ei5=VC(0.1),sr5=VC(0.0),si5=VC(0.0);
  V er6=VC(0.6),ei6=VC(0.1),sr6=VC(0.0),si6=VC(0.0);
  V er7=VC(0.7),ei7=VC(0.1),sr7=VC(0.0),si7=VC(0.0);
  V er8=VC(0.8),ei8=VC(0.1),sr8=VC(0.0),si8=VC(0.0);
  V er9=VC(0.9),ei9=VC(0.1),sr9=VC(0.0),si9=VC(0.0);
  V erA=VC(1.0),eiA=VC(0.1),srA=VC(0.0),siA=VC(0.0);
  V erB=VC(1.1),eiB=VC(0.1),srB=VC(0.0),siB=VC(0.0);
  #pragma GCC unroll 1
  for (long j = 1; j <= 11; j++) {
    V u_r = ur[j], u_i = ui[j], v_r = vr[j], v_i = vi[j];
    er1 += c1[j]*u_r; ei1 += c1[j]*u_i; sr1 += s1[j]*v_r; si1 += s1[j]*v_i;
    er2 += c2[j]*u_r; ei2 += c2[j]*u_i; sr2 += s2[j]*v_r; si2 += s2[j]*v_i;
    er3 += c3[j]*u_r; ei3 += c3[j]*u_i; sr3 += s3[j]*v_r; si3 += s3[j]*v_i;
    er4 += c4[j]*u_r; ei4 += c4[j]*u_i; sr4 += s4[j]*v_r; si4 += s4[j]*v_i;
  }
  outr[0]=er1+si1; outi[0]=ei1-sr1; outr[1]=er2+si2; outi[1]=ei2-sr2;
  outr[2]=er3+si3; outi[2]=ei3-sr3; outr[3]=er4+si4; outi[3]=ei4-sr4;
  (void)er5;(void)er6;(void)er7;(void)er8;(void)er9;(void)erA;(void)erB;
}
int main(){
  for(int i=0;i<12;i++) for(int l=0;l<8;l++){ ur[i][l]=0.1; ui[i][l]=0.2; vr[i][l]=0.3; vi[i][l]=0.4; }
  long R=2000000; uint64_t t0,t1;
  kloop();
  t0=rdtscp();
  for(long r=0;r<R;r++) kloop();
  t1=rdtscp();
  printf("kloop (11 iters, 16 FMA/iter): %.1f c/call = %.2f c/iter\n",(double)(t1-t0)/R,(double)(t1-t0)/R/11);
  printf("chk %g\n", outr[0][0]);
  return 0;
}
EOF
gcc -O3 -march=native prof16.c -o prof16 && taskset -c 0 ./prof16