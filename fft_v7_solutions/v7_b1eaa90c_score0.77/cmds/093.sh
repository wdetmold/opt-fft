cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null; cd /tmp/bench && cat > bw4.c <<'EOF'
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
double bench_read(double *a, long n, long R){
  __m512d s0=_mm512_setzero_pd(), s1=_mm512_setzero_pd(),s2=_mm512_setzero_pd(),s3=_mm512_setzero_pd();
  uint64_t t0=rdtscp();
  for(long r=0;r<R;r++) for(long i=0;i<n;i+=32){
    s0=_mm512_add_pd(s0,_mm512_load_pd(a+i)); s1=_mm512_add_pd(s1,_mm512_load_pd(a+i+8));
    s2=_mm512_add_pd(s2,_mm512_load_pd(a+i+16)); s3=_mm512_add_pd(s3,_mm512_load_pd(a+i+24));
  }
  uint64_t t1=rdtscp();
  volatile double sink = s0[0]+s1[1]+s2[2]+s3[3]; (void)sink;
  return (double)n*8*R/(t1-t0);
}
int main(){
  long sz = 16L<<20; long n = sz/8, R=20;
  double *a = aligned_alloc(64, sz); memset(a,1,sz);
  printf("malloc 4K pages: %.2f B/cyc\n", bench_read(a,n,R));
  void *p = mmap(0, sz+(2<<20), PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  uintptr_t q=((uintptr_t)p + (2<<20)-1) & ~((uintptr_t)(2<<20)-1);
  madvise((void*)q, sz, MADV_HUGEPAGE);
  double *b=(double*)q; memset(b,1,sz);
  printf("THP:             %.2f B/cyc\n", bench_read(b,n,R));
  return 0;
}
EOF
gcc -O3 -march=native bw4.c -o bw4 && taskset -c 0 ./bw4