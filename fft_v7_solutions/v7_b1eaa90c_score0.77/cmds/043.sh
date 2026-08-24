cd /tmp/bench && cat > bw2.c <<'EOF'
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
int main(){
  long sz = 8L<<20; // 8MB
  double *a = aligned_alloc(64, sz), *b = aligned_alloc(64, sz);
  memset(a,1,sz); memset(b,2,sz);
  long n = sz/8; long R=20;
  // pure sequential read (sum)
  uint64_t t0=rdtscp();
  __m512d s0=_mm512_setzero_pd(), s1=_mm512_setzero_pd(),s2=_mm512_setzero_pd(),s3=_mm512_setzero_pd();
  for(long r=0;r<R;r++) for(long i=0;i<n;i+=32){
    s0=_mm512_add_pd(s0,_mm512_load_pd(a+i)); s1=_mm512_add_pd(s1,_mm512_load_pd(a+i+8));
    s2=_mm512_add_pd(s2,_mm512_load_pd(a+i+16)); s3=_mm512_add_pd(s3,_mm512_load_pd(a+i+24));
  }
  uint64_t t1=rdtscp();
  printf("L3 seq read:  %.2f B/cyc (chk %g)\n", (double)sz*R/(t1-t0), s0[0]+s1[1]+s2[2]+s3[3]);
  // pure write
  t0=rdtscp();
  for(long r=0;r<R;r++) for(long i=0;i<n;i+=16){ _mm512_store_pd(b+i,s0); _mm512_store_pd(b+i+8,s1); }
  t1=rdtscp();
  printf("L3 seq write: %.2f B/cyc\n", (double)sz*R/(t1-t0));
  // strided read: stride 32832 B (513 V), 64 streams like K3
  t0=rdtscp();
  long P2V=513;
  for(long r=0;r<R;r++)
    for(long o=0;o<512;o++){
      __m512d acc=_mm512_setzero_pd();
      for(long i=0;i<16;i++) acc=_mm512_add_pd(acc,_mm512_load_pd(a+(o+i*P2V)*8));
      s0=_mm512_add_pd(s0,acc);
    }
  t1=rdtscp();
  printf("K3-like strided read (16 rows/col, 2MB arr): %.2f B/cyc (chk %g)\n", (double)(512*16*64)*R/(t1-t0), s0[2]);
  // same with prefetch 8 ahead issued at o%8==0
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for(long o=0;o<512;o++){
      if((o&7)==0) for(long i=0;i<16;i++) __builtin_prefetch(a+(o+8+i*P2V)*8,0,3);
      __m512d acc=_mm512_setzero_pd();
      for(long i=0;i<16;i++) acc=_mm512_add_pd(acc,_mm512_load_pd(a+(o+i*P2V)*8));
      s0=_mm512_add_pd(s0,acc);
    }
  t1=rdtscp();
  printf("K3-like strided read + PF: %.2f B/cyc (chk %g)\n", (double)(512*16*64)*R/(t1-t0), s0[3]);
  return 0;
}
EOF
gcc -O3 -march=native bw2.c -o bw2 && taskset -c 0 ./bw2