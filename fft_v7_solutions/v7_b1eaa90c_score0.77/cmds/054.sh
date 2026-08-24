cd /workdir/dev && cat > prof10.c <<'EOF'
#include <stdio.h>
#include <stdint.h>
#include "../implementation.c"
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
#define PROF(L, P1V, P2V, NB) do { \
  for(long i=0;i<L*P2V;i++) for(int l=0;l<8;l++){ slab##L##_xr[i][l]=0.3*((i*7+l)%13)-1.0; slab##L##_xi[i][l]=0.2*((i*5+l)%11)-0.8; slab##L##_cer[i][l]=0.05; slab##L##_cei[i][l]=-0.03; } \
  long R=200; uint64_t t0,t1; double el=(double)(L)*(L)*(L); \
  t0=rdtscp(); \
  for(long r=0;r<R;r++) \
    for (long x = 0; x < L; x++) { \
      V *sr = slab##L##_xr + x*P2V, *si = slab##L##_xi + x*P2V; \
      for (long cc = 0; cc < NB; cc++) f##L##_cb(sr + cc, si + cc, slab##L##_br, slab##L##_bi); \
    } \
  t1=rdtscp(); printf("L=%d K1 kernels only: %.3f c/elem\n", L, (double)(t1-t0)/R/el); \
  t0=rdtscp(); \
  for(long r=0;r<R;r++) \
    for (long o = 0; o < L*P1V; o++) \
      f##L##_k3(slab##L##_xr + o, slab##L##_xi + o, slab##L##_xr + o, slab##L##_xi + o, slab##L##_cer + o, slab##L##_cei + o); \
  t1=rdtscp(); printf("L=%d K3 full: %.3f c/elem\n", L, (double)(t1-t0)/R/el); \
  t0=rdtscp(); \
  for(long r=0;r<R;r++) slab##L##_iter(1); \
  t1=rdtscp(); printf("L=%d full iter: %.3f c/elem\n", L, (double)(t1-t0)/R/el); \
} while(0)
int main(){
  PROF(17, 3, 51, 3);
  PROF(23, 3, 69, 3);
  PROF(36, 5, 180, 5);
  PROF(45, 6, 270, 6);
  PROF(64, 8, 513, 8);
  return 0;
}
EOF
gcc -O3 -march=native prof10.c -o prof10 -lm && taskset -c 0 ./prof10