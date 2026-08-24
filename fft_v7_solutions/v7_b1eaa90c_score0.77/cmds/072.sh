cd /workdir/dev && cat > prof13.c <<'EOF'
#include <stdio.h>
#include <stdint.h>
#include "../implementation.c"
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
int main(){
  for(long i=0;i<64*513;i++) for(int l=0;l<8;l++){ sq64_ar[i][l]=0.3*((i*7+l)%13)-1.0; sq64_ai[i][l]=0.2*((i*5+l)%11)-0.8; sq64_br[i][l]=0.1; sq64_bi[i][l]=0.1; sq64_cr[i][l]=0.05; sq64_ci[i][l]=-0.03; }
  long R=30; uint64_t t0,t1;
  // full main sweep
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long g = 0; g < 8; g++) {
      f64_s2(sq64_br + g*8*513, sq64_bi + g*8*513, sq64_cr + g*8*513, sq64_ci + g*8*513, sq64_tr, sq64_ti);
      for (long k2 = 0; k2 < 8; k2++)
        sq64_yz(sq64_tr + k2*513, sq64_ti + k2*513);
      f64_s1(sq64_tr, sq64_ti, sq64_ar + g*513, sq64_ai + g*513, sq64_twr + g*8, sq64_twi + g*8);
    }
  t1=rdtscp(); printf("full sweep: %.3f c/elem\n", (double)(t1-t0)/R/262144);
  // s2 only
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long g = 0; g < 8; g++)
      f64_s2(sq64_br + g*8*513, sq64_bi + g*8*513, sq64_cr + g*8*513, sq64_ci + g*8*513, sq64_tr, sq64_ti);
  t1=rdtscp(); printf("s2 only:    %.3f c/elem\n", (double)(t1-t0)/R/262144);
  // yz only (on T, hot)
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long g = 0; g < 8; g++)
      for (long k2 = 0; k2 < 8; k2++)
        sq64_yz(sq64_tr + k2*513, sq64_ti + k2*513);
  t1=rdtscp(); printf("yz only:    %.3f c/elem\n", (double)(t1-t0)/R/262144);
  // s1 only
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long g = 0; g < 8; g++)
      f64_s1(sq64_tr, sq64_ti, sq64_ar + g*513, sq64_ai + g*513, sq64_twr + g*8, sq64_twi + g*8);
  t1=rdtscp(); printf("s1 only:    %.3f c/elem\n", (double)(t1-t0)/R/262144);
  double s=0; for(long i=0;i<64*513;i++) s+=sq64_ar[i][3]; printf("chk %g\n", s);
  return 0;
}
EOF
gcc -O3 -march=native prof13.c -o prof13 -lm && taskset -c 0 ./prof13