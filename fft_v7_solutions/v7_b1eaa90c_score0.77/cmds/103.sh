cd /workdir/dev && cat > prof18.c <<'EOF'
#include <stdio.h>
#include <stdint.h>
#include "../implementation.c"
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
static V Xr[48*8], Xi[48*8], Yr[64], Yi[64];
int main(){
  for(int i=0;i<48*8;i++) for(int l=0;l<8;l++){ Xr[i][l]=0.1*((i+l)%7)-0.3; Xi[i][l]=0.1*((i*3+l)%5)-0.2; }
  long R=200000; uint64_t t0,t1;
  t0=rdtscp();
  for(long r=0;r<R;r++) f45_cb(Xr+(r&7), Xi+(r&7), Yr, Yi);
  t1=rdtscp();
  printf("f45_cb isolated: %.1f c/call = %.3f c/elem(45x8=360)\n",(double)(t1-t0)/R,(double)(t1-t0)/R/360);
  t0=rdtscp();
  for(long r=0;r<R;r++) f36_cb(Xr+(r&7), Xi+(r&7), Yr, Yi);
  t1=rdtscp();
  printf("f36_cb isolated: %.1f c/call = %.3f c/elem(288)\n",(double)(t1-t0)/R,(double)(t1-t0)/R/288);
  t0=rdtscp();
  for(long r=0;r<R;r++) f64_cb(Xr+(r&7), Xi+(r&7), Yr, Yi);
  t1=rdtscp();
  printf("f64_cb isolated: %.1f c/call = %.3f c/elem(512)\n",(double)(t1-t0)/R,(double)(t1-t0)/R/512);
  t0=rdtscp();
  for(long r=0;r<R;r++){ f64_zline(Xr+(r&7)*8, Xi+(r&7)*8); }
  t1=rdtscp();
  printf("f64_zline: %.1f c/call = %.3f c/elem(64)\n",(double)(t1-t0)/R,(double)(t1-t0)/R/64);
  double s=0; for(int i=0;i<45;i++) s+=Yr[i][2]; printf("chk %g\n", s);
  return 0;
}
EOF
gcc -O3 -march=native prof18.c -o prof18 -lm && taskset -c 0 ./prof18