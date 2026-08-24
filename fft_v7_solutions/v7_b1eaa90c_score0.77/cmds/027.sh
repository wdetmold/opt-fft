cd /workdir/dev && cat > prof1.c <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include "../implementation.c"
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
int main(){
  long L=64, L3=L*L*L;
  double *x0=malloc(2*L3*sizeof(double)), *c=malloc(2*L3*sizeof(double));
  double *o1=malloc(2*L3*sizeof(double)), *om=malloc(2*L3*sizeof(double));
  srand(1); for(long i=0;i<2*L3;i++){ x0[i]=(rand()%1000-500)/250.0; c[i]=(rand()%1000-500)/2500.0; }
  // warm
  run64(x0,c,1,3,o1,om);
  uint64_t t0=rdtscp();
  run64(x0,c,1,23,o1,om);
  uint64_t t1=rdtscp();
  // subtract ingest/extract via m diff
  uint64_t t2=rdtscp();
  run64(x0,c,1,3,o1,om);
  uint64_t t3=rdtscp();
  double per_iter=((double)(t1-t0)-(double)(t3-t2))/20.0;
  printf("L=64 per-iter: %.0f cyc = %.3f cyc/elem = %.3f ns/elem\n", per_iter, per_iter/L3, per_iter/L3/2.6);
  // phase breakdown
  t0=rdtscp();
  for(int r=0;r<20;r++){
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
      for (long cc = 0; cc < 8; cc++)
        fft64_core(slab64_tsr + cc, slab64_tsi + cc, sr + cc, si + cc, 8, 8);
    }
  }
  t1=rdtscp();
  printf("sweep1 (K1+K2): %.3f cyc/elem\n", (double)(t1-t0)/20/L3);
  t0=rdtscp();
  for(int r=0;r<20;r++){
    const V *pcr = slab64_cer, *pci = slab64_cei;
    for (long o = 0; o < 512; o++) {
      fft64_core(slab64_xr + o, slab64_xi + o, slab64_kr, slab64_ki, 513, 1);
      for (long i = 0; i < 64; i++) {
        long idx = o + i*513;
        V zr = slab64_kr[i] + pcr[idx];
        V zi = slab64_ki[i] + pci[idx];
        V f = pw_factor(zr, zi);
        slab64_xr[idx] = zr*f; slab64_xi[idx] = zi*f;
      }
    }
  }
  t1=rdtscp();
  printf("sweep2 (K3+pw): %.3f cyc/elem\n", (double)(t1-t0)/20/L3);
  // K1 only (no transpose)
  t0=rdtscp();
  for(int r=0;r<20;r++)
    for (long x = 0; x < 64; x++) {
      V *sr = slab64_xr + x*513, *si = slab64_xi + x*513;
      for (long cc = 0; cc < 8; cc++)
        fft64_core(sr + cc, si + cc, slab64_br, slab64_bi, 8, 1);
    }
  t1=rdtscp();
  printf("K1 fft only: %.3f cyc/elem\n", (double)(t1-t0)/20/L3);
  return 0;
}
EOF
gcc -O3 -march=native prof1.c -o prof1 -lm && taskset -c 0 ./prof1