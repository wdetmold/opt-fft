cd /tmp/bench && cat > bw.c <<'EOF'
#include <immintrin.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
int main(){
  for (long sz = 1<<15; sz <= (1L<<28); sz <<= 2) {
    double *a = aligned_alloc(64, sz), *b = aligned_alloc(64, sz);
    memset(a, 1, sz); memset(b, 2, sz);
    long n = sz/8, reps = (1L<<28)/sz < 4 ? 4 : (1L<<28)/sz;
    // read+write: b[i] = a[i]*1.1 (stream copy-scale)
    uint64_t t0=rdtscp();
    for (long r=0;r<reps;r++){
      for (long i=0;i<n;i+=8){
        __m512d v=_mm512_load_pd(a+i); _mm512_store_pd(b+i, _mm512_mul_pd(v,_mm512_set1_pd(1.1)));
      }
    }
    uint64_t t1=rdtscp();
    double cyc=(double)(t1-t0)/reps;
    printf("size %8ld KB: copy-scale %.2f B/cyc (%.1f GB/s @2.6GHz, r+w counted %.1f)\n",
      sz/1024, sz/cyc, sz/cyc*2.6, 2*sz/cyc*2.6);
    free(a); free(b);
  }
  return 0;
}
EOF
gcc -O2 -march=native bw.c -o bw && taskset -c 0 ./bw