cd /workdir/dev && cat > prof8.c <<'EOF'
#include <stdio.h>
#include <stdint.h>
#include "../implementation.c"
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
static V X_r[64*8], X_i[64*8], Y_r[64*8], Y_i[64*8];
int main(){
  for(int i=0;i<64*8;i++) for(int l=0;l<8;l++){ X_r[i][l]=0.1*((i+l)%7)-0.3; X_i[i][l]=0.1*((i*3+l)%5)-0.2; }
  for(long i=0;i<64*513;i++) for(int l=0;l<8;l++){ slab64_xr[i][l]=0.3*((i*7+l)%13)-1.0; slab64_xi[i][l]=0.2*((i*5+l)%11)-0.8; slab64_cer[i][l]=0.05; slab64_cei[i][l]=-0.03; }
  long R=300000; uint64_t t0,t1;
  t0=rdtscp();
  for(long r=0;r<R;r++) f64_cb(slab64_tsr + (r&7), slab64_tsi + (r&7), Y_r, Y_i);
  t1=rdtscp();
  printf("f64_cb indep:  %.1f cyc/call (%.3f c/elem)\n",(double)(t1-t0)/R,(double)(t1-t0)/R/512);
  t0=rdtscp();
  for(long r=0;r<R;r++) f64_k3(slab64_xr + (r&255), slab64_xi + (r&255), Y_r, Y_i, slab64_cer+(r&255), slab64_cei+(r&255));
  t1=rdtscp();
  printf("f64_k3 indep(L2-hot): %.1f cyc/call (%.3f c/elem)\n",(double)(t1-t0)/R,(double)(t1-t0)/R/512);
  // full slab iteration phases
  R=30;
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long x = 0; x < 64; x++) {
      V *sr = slab64_xr + x*513, *si = slab64_xi + x*513;
      for (long cc = 0; cc < 8; cc++) {
        f64_cb(sr + cc, si + cc, slab64_br, slab64_bi);
        for (long rb = 0; rb < 8; rb++) {
          V tb[8];
          tr8x8(slab64_br + rb*8, tb);
          for (int q = 0; q < 8; q++) slab64_tsr[(cc*8+q)*8 + rb] = tb[q];
          tr8x8(slab64_bi + rb*8, tb);
          for (int q = 0; q < 8; q++) slab64_tsi[(cc*8+q)*8 + rb] = tb[q];
        }
      }
      for (long cc = 0; cc < 8; cc++) {
        f64_cb(slab64_tsr + cc, slab64_tsi + cc, slab64_br, slab64_bi);
        for (long rb = 0; rb < 8; rb++) {
          V tb[8];
          tr8x8(slab64_br + rb*8, tb);
          for (int q = 0; q < 8; q++) sr[(cc*8+q)*8 + rb] = tb[q];
          tr8x8(slab64_bi + rb*8, tb);
          for (int q = 0; q < 8; q++) si[(cc*8+q)*8 + rb] = tb[q];
        }
      }
    }
  t1=rdtscp(); printf("sweep1 (K1+K2 w/ 2 transposes): %.3f cyc/elem\n", (double)(t1-t0)/R/262144);
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long o = 0; o < 512; o++)
      f64_k3(slab64_xr + o, slab64_xi + o, slab64_xr + o, slab64_xi + o, slab64_cer + o, slab64_cei + o);
  t1=rdtscp(); printf("K3: %.3f cyc/elem\n", (double)(t1-t0)/R/262144);
  double s=0; for(long i=0;i<64*513;i++) s+=slab64_xr[i][3]; printf("chk %g\n", s);
  return 0;
}
EOF
gcc -O3 -march=native prof8.c -o prof8 -lm && taskset -c 0 ./prof8