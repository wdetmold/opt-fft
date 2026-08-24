cd /workdir/dev && cat > prof15.c <<'EOF'
#include <stdio.h>
#include <stdint.h>
#include "../implementation.c"
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
static V Xr[64], Xi[64], Yr[64], Yi[64];
int main(){
  for(int i=0;i<64;i++) for(int l=0;l<8;l++){ Xr[i][l]=0.1*((i+l)%7)-0.3; Xi[i][l]=0.1*((i*3+l)%5)-0.2; }
  long R=500000; uint64_t t0,t1;
  t0=rdtscp();
  for(long r=0;r<R;r++) f23_z(Xr+(r&7), Xi+(r&7), Yr+(r&7), Yi+(r&7));
  t1=rdtscp();
  printf("f23_z isolated: %.1f c/call = %.3f c/elem\n", (double)(t1-t0)/R, (double)(t1-t0)/R/184);
  // dependency-chained version (in-place same line): latency view
  t0=rdtscp();
  for(long r=0;r<R;r++) f23_z(Xr, Xi, Xr, Xi);
  t1=rdtscp();
  printf("f23_z chained:  %.1f c/call\n", (double)(t1-t0)/R);
  double s=0; for(int i=0;i<23;i++) s+=Yr[i][2]+Xr[i][1]; printf("chk %g\n", s);
  return 0;
}
EOF
gcc -O3 -march=native prof15.c -o prof15 -lm && taskset -c 0 ./prof15