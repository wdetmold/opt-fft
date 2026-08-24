cd /workdir/dev && cat > prof2.c <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>
typedef double V __attribute__((vector_size(64), aligned(64)));
#define VC(x) ((V){(x),(x),(x),(x),(x),(x),(x),(x)})
static inline V vrsqrt14(V x){ return (V)_mm512_rsqrt14_pd((__m512d)x); }
static inline V vrcp14(V x){ return (V)_mm512_rcp14_pd((__m512d)x); }
static inline V vmaxv(V a, V b){ return (V)_mm512_max_pd((__m512d)a,(__m512d)b); }
static inline V pw_factor(V zr, V zi){
  V s = zr*zr + zi*zi;
  s = vmaxv(s, VC(2.2250738585072014e-308));
  V r = vrsqrt14(s);
  V h = s * VC(0.5);
  r = r * (VC(1.5) - h*r*r);
  r = r * (VC(1.5) - h*r*r);
  V t = s * r;
  V u = VC(1.0) + t;
  V v = vrcp14(u);
  v = v + v*(VC(1.0) - u*v);
  v = v + v*(VC(1.0) - u*v);
  return v;
}
#include "core64only.h"
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
static V A[64*8], Bv[64*8];
int main(){
  for(int i=0;i<64*8;i++) for(int l=0;l<8;l++){ A[i][l]=0.1*i+0.01*l; Bv[i][l]=0.2; }
  // 1) core throughput: 8 calls per rep (like one chunk batch), in-place on A region stride 8
  long R=20000;
  uint64_t t0=rdtscp();
  for(long r=0;r<R;r++){
    fft64_core(A, A+64*4, A, A+64*4, 4, 4); // stride 4 V to stay in array
  }
  uint64_t t1=rdtscp();
  printf("fft64_core in-place: %.1f cyc/call = %.3f cyc per elem(512)\n", (double)(t1-t0)/R, (double)(t1-t0)/R/512);
  // 2) out-of-place
  t0=rdtscp();
  for(long r=0;r<R;r++) fft64_core(A, A+64*4, Bv, Bv+64*4, 4, 4);
  t1=rdtscp();
  printf("fft64_core oop:      %.1f cyc/call = %.3f cyc per elem\n", (double)(t1-t0)/R, (double)(t1-t0)/R/512);
  // 3) pointwise only
  t0=rdtscp();
  for(long r=0;r<R;r++){
    for(int i=0;i<64;i++){
      V f = pw_factor(A[i], A[i+64*4]);
      Bv[i]=A[i]*f; Bv[i+64]=A[i+64*4]*f;
    }
  }
  t1=rdtscp();
  printf("pointwise 64 vecs:   %.1f cyc = %.3f cyc/elem\n", (double)(t1-t0)/R, (double)(t1-t0)/R/512);
  volatile double sink=Bv[3][2]+A[5][1]; (void)sink;
  return 0;
}
EOF
gcc -O3 -march=native prof2.c -o prof2 -lm && taskset -c 0 ./prof2