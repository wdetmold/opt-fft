cd /workdir/dev && GENCFG='{"pw":{"13":"alt","17":"alt","23":"alt","45":"alt"}}' python3 gen.py && cat > prof14.c <<'EOF'
#include <stdio.h>
#include <stdint.h>
#include "../implementation.c"
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
int main(){
  for(long i=0;i<12167;i++) for(int l=0;l<8;l++){ soa23_xr[i][l]=0.3*((i*7+l)%13)-1.0; soa23_xi[i][l]=0.2*((i*5+l)%11)-0.8; soa23_cr[i][l]=0.05; soa23_ci[i][l]=-0.03; }
  long R=100; uint64_t t0,t1; double el=12167*8;
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long x = 0; x < 23; x++) {
      V *sr = soa23_xr + x*529, *si = soa23_xi + x*529;
      for (long y = 0; y < 23; y++) f23_z(sr + y*23, si + y*23, sr + y*23, si + y*23);
    }
  t1=rdtscp(); printf("z-pass: %.3f c/elem\n", (double)(t1-t0)/R/el);
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long x = 0; x < 23; x++) {
      V *sr = soa23_xr + x*529, *si = soa23_xi + x*529;
      for (long z = 0; z < 23; z++) f23_y(sr + z, si + z, sr + z, si + z);
    }
  t1=rdtscp(); printf("y-pass: %.3f c/elem\n", (double)(t1-t0)/R/el);
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long u = 0; u < 529; u++)
      f23_x(soa23_xr + u, soa23_xi + u, soa23_xr + u, soa23_xi + u, soa23_cr + u, soa23_ci + u);
  t1=rdtscp(); printf("x-pass+pw: %.3f c/elem\n", (double)(t1-t0)/R/el);
  double s=0; for(long i=0;i<12167;i++) s+=soa23_xr[i][3]; printf("chk %g\n", s);
  return 0;
}
EOF
gcc -O3 -march=native prof14.c -o prof14 -lm && taskset -c 0 ./prof14