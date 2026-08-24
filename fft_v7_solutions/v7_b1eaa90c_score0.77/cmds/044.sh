cd /tmp/bench && cat > bw3.c <<'EOF'
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
#define P2V 513
int main(){
  long nv = 64*P2V; // V units
  double *xr = aligned_alloc(64, nv*64), *xi = aligned_alloc(64, nv*64);
  double *cr = aligned_alloc(64, nv*64), *ci = aligned_alloc(64, nv*64);
  memset(xr,1,nv*64); memset(xi,1,nv*64); memset(cr,1,nv*64); memset(ci,1,nv*64);
  long R=10;
  __m512d s0=_mm512_setzero_pd();
  // K3-shape read: all 4 arrays, 64 rows per col
  uint64_t t0=rdtscp();
  for(long r=0;r<R;r++)
    for(long o=0;o<512;o++){
      __m512d acc=_mm512_setzero_pd();
      for(long i=0;i<64;i++){
        long idx=(o+i*P2V)*8;
        acc=_mm512_add_pd(acc,_mm512_load_pd(xr+idx));
        acc=_mm512_add_pd(acc,_mm512_load_pd(xi+idx));
        acc=_mm512_add_pd(acc,_mm512_load_pd(cr+idx));
        acc=_mm512_add_pd(acc,_mm512_load_pd(ci+idx));
      }
      s0=_mm512_add_pd(s0,acc);
    }
  uint64_t t1=rdtscp();
  double bytes=512.0*64*4*64;
  printf("K3 reads, no PF:   %.2f B/cyc => pass %.0f cyc (%.2f c/elem) chk %g\n", bytes*R/(t1-t0), (double)(t1-t0)/R, (double)(t1-t0)/R/262144, s0[1]);
  for (int pd=8; pd<=32; pd*=2) {
    t0=rdtscp();
    for(long r=0;r<R;r++)
      for(long o=0;o<512;o++){
        if((o&7)==0){
          for(long i=0;i<64;i++){
            long idx=(o+pd+i*P2V)*8;
            __builtin_prefetch(xr+idx,0,2); __builtin_prefetch(xi+idx,0,2);
            __builtin_prefetch(cr+idx,0,2); __builtin_prefetch(ci+idx,0,2);
          }
        }
        __m512d acc=_mm512_setzero_pd();
        for(long i=0;i<64;i++){
          long idx=(o+i*P2V)*8;
          acc=_mm512_add_pd(acc,_mm512_load_pd(xr+idx));
          acc=_mm512_add_pd(acc,_mm512_load_pd(xi+idx));
          acc=_mm512_add_pd(acc,_mm512_load_pd(cr+idx));
          acc=_mm512_add_pd(acc,_mm512_load_pd(ci+idx));
        }
        s0=_mm512_add_pd(s0,acc);
      }
    t1=rdtscp();
    printf("K3 reads, PF d=%2d: %.2f B/cyc => pass %.0f cyc (%.2f c/elem) chk %g\n", pd, bytes*R/(t1-t0), (double)(t1-t0)/R, (double)(t1-t0)/R/262144, s0[1]);
  }
  // sequential read of same total with PF
  t0=rdtscp();
  for(long r=0;r<R;r++){
    __m512d acc=_mm512_setzero_pd();
    for(long i=0;i<nv*8;i+=32){
      __builtin_prefetch(xr+i+512,0,2); __builtin_prefetch(xi+i+512,0,2);
      acc=_mm512_add_pd(acc,_mm512_load_pd(xr+i)); acc=_mm512_add_pd(acc,_mm512_load_pd(xi+i));
      acc=_mm512_add_pd(acc,_mm512_load_pd(xr+i+8)); acc=_mm512_add_pd(acc,_mm512_load_pd(xi+i+8));
      acc=_mm512_add_pd(acc,_mm512_load_pd(xr+i+16)); acc=_mm512_add_pd(acc,_mm512_load_pd(xi+i+16));
      acc=_mm512_add_pd(acc,_mm512_load_pd(xr+i+24)); acc=_mm512_add_pd(acc,_mm512_load_pd(xi+i+24));
    }
    s0=_mm512_add_pd(s0,acc);
  }
  t1=rdtscp();
  printf("seq read 2 arrays + PF: %.2f B/cyc chk %g\n", 2.0*nv*64*R/(t1-t0), s0[2]);
  return 0;
}
EOF
gcc -O3 -march=native bw3.c -o bw3 && taskset -c 0 ./bw3