cd /workdir/dev && cat > prof3.c <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../implementation.c"
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
static double chk(void){ double s=0; for(long i=0;i<64*513;i++) s+=slab64_xr[i][3]+slab64_xi[i][5]; return s; }
int main(){
  for(long i=0;i<64*513;i++) for(int l=0;l<8;l++){ slab64_xr[i][l]=0.3*((i*7+l)%13)-1.0; slab64_xi[i][l]=0.2*((i*5+l)%11)-0.8; slab64_cer[i][l]=0.05; slab64_cei[i][l]=-0.03; }
  long R=30; uint64_t t0,t1; double c0;
  // K1 loop only (fft + transpose to TSLAB)
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long x = 0; x < 64; x++) {
      V *sr = slab64_xr + x*513, *si = slab64_xi + x*513;
      for (long cc = 0; cc < 8; cc++) {
        fft64_core(sr + cc, si + cc, slab64_br, slab64_bi, 8, 1);
        for (long rb = 0; rb < 8; rb++) {
          V tb[8];
          tr8x8(slab64_br + rb*8, tb);
          for (int q = 0; q < 8; q++) slab64_tsr[(cc*8+q)*8 + rb] = tb[q];
          tr8x8(slab64_bi + rb*8, tb);
          for (int q = 0; q < 8; q++) slab64_tsi[(cc*8+q)*8 + rb] = tb[q];
        }
      }
    }
  t1=rdtscp(); printf("K1 (fft+transpose): %.3f cyc/elem [chk %g]\n", (double)(t1-t0)/R/262144, chk());
  // K2 only
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long x = 0; x < 64; x++) {
      V *sr = slab64_xr + x*513, *si = slab64_xi + x*513;
      for (long cc = 0; cc < 8; cc++)
        fft64_core(slab64_tsr + cc, slab64_tsi + cc, sr + cc, si + cc, 8, 8);
    }
  t1=rdtscp(); printf("K2 (fft strided st): %.3f cyc/elem [chk %g]\n", (double)(t1-t0)/R/262144, chk());
  // K3 fft only (no pointwise): write to kr/ki then copy back plain
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long o = 0; o < 512; o++) {
      fft64_core(slab64_xr + o, slab64_xi + o, slab64_kr, slab64_ki, 513, 1);
      for (long i = 0; i < 64; i++) { long idx=o+i*513; slab64_xr[idx]=slab64_kr[i]*VC(0.11); slab64_xi[idx]=slab64_ki[i]*VC(0.13); }
    }
  t1=rdtscp(); printf("K3 fft+scale only:  %.3f cyc/elem [chk %g]\n", (double)(t1-t0)/R/262144, chk());
  // pointwise pass alone over the volume
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long idx = 0; idx < 64*513; idx++) {
      V zr = slab64_xr[idx] + slab64_cer[idx];
      V zi = slab64_xi[idx] + slab64_cei[idx];
      V f = pw_factor(zr, zi);
      slab64_xr[idx] = zr*f; slab64_xi[idx] = zi*f;
    }
  t1=rdtscp(); printf("pointwise pass:     %.3f cyc/elem [chk %g]\n", (double)(t1-t0)/R/262144/1.0026, chk());
  return 0;
}
EOF
gcc -O3 -march=native prof3.c -o prof3 -lm && taskset -c 0 ./prof3