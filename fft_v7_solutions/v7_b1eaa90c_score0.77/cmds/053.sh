cd /workdir/dev && cat > prof9.c <<'EOF'
#include <stdio.h>
#include <stdint.h>
#include "../implementation.c"
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
int main(){
  for(long i=0;i<64*513;i++) for(int l=0;l<8;l++){ slab64_xr[i][l]=0.3*((i*7+l)%13)-1.0; slab64_xi[i][l]=0.2*((i*5+l)%11)-0.8; }
  long R=30; uint64_t t0,t1;
  // transpose-only loop (like sweep1's transposes, L1-hot buf -> tslab)
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long x = 0; x < 64; x++)
      for (long cc = 0; cc < 8; cc++)
        for (long rb = 0; rb < 8; rb++) {
          V tb[8];
          tr8x8(slab64_br + rb*8, tb);
          for (int q = 0; q < 8; q++) slab64_tsr[(cc*8+q)*8 + rb] = tb[q];
          tr8x8(slab64_bi + rb*8, tb);
          for (int q = 0; q < 8; q++) slab64_tsi[(cc*8+q)*8 + rb] = tb[q];
        }
  t1=rdtscp(); printf("transposes only (2 per elem pair): %.3f cyc/elem\n", (double)(t1-t0)/R/262144);
  // kernels only, no transposes (K1-like: slab -> buf)
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long x = 0; x < 64; x++) {
      V *sr = slab64_xr + x*513, *si = slab64_xi + x*513;
      for (long cc = 0; cc < 8; cc++)
        f64_cb(sr + cc, si + cc, slab64_br, slab64_bi);
    }
  t1=rdtscp(); printf("K1 kernels only (volume from L3): %.3f cyc/elem\n", (double)(t1-t0)/R/262144);
  // K1 kernels only but L2-resident (single slab over and over)
  t0=rdtscp();
  for(long r=0;r<R*64;r++) {
      V *sr = slab64_xr, *si = slab64_xi;
      for (long cc = 0; cc < 8; cc++)
        f64_cb(sr + cc, si + cc, slab64_br, slab64_bi);
  }
  t1=rdtscp(); printf("K1 kernels only (L1-resident slab): %.3f cyc/elem\n", (double)(t1-t0)/R/262144);
  double s=0; for(long i=0;i<64;i++) s+=slab64_br[i][3]; printf("chk %g\n", s);
  return 0;
}
EOF
gcc -O3 -march=native prof9.c -o prof9 -lm && taskset -c 0 ./prof9