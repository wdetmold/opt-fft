cd /workdir/dev && cat > prof5.c <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "../implementation.c"
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
static V Y_r[64*513+17], Y_i[64*513+17];
static double chk(void){ double s=0; for(long i=0;i<64*513;i++) s+=Y_r[i][3]+slab64_xr[i][1]; return s; }
int main(){
  for(long i=0;i<64*513;i++) for(int l=0;l<8;l++){ slab64_xr[i][l]=0.3*((i*7+l)%13)-1.0; slab64_xi[i][l]=0.2*((i*5+l)%11)-0.8; slab64_cer[i][l]=0.05; slab64_cei[i][l]=-0.03; }
  long R=30; uint64_t t0,t1;
  // K3 in-place (current)
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long o = 0; o < 512; o++)
      fft64_colspw(slab64_xr + o, slab64_xi + o, slab64_xr + o, slab64_xi + o, 513, 513, slab64_cer + o, slab64_cei + o);
  t1=rdtscp(); printf("K3 in-place:   %.3f cyc/elem [chk %g]\n", (double)(t1-t0)/R/262144, chk());
  // K3 out-of-place with +9 V offset (576B) destination
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long o = 0; o < 512; o++)
      fft64_colspw(slab64_xr + o, slab64_xi + o, Y_r + 9 + o, Y_i + 9 + o, 513, 513, slab64_cer + o, slab64_cei + o);
  t1=rdtscp(); printf("K3 oop+9:      %.3f cyc/elem [chk %g]\n", (double)(t1-t0)/R/262144, chk());
  // K3 oop aligned same mod 4K as src
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long o = 0; o < 512; o++)
      fft64_colspw(slab64_xr + o, slab64_xi + o, Y_r + o, Y_i + o, 513, 513, slab64_cer + o, slab64_cei + o);
  t1=rdtscp(); printf("K3 oop+0:      %.3f cyc/elem [chk %g]\n", (double)(t1-t0)/R/262144, chk());
  return 0;
}
EOF
gcc -O3 -march=native prof5.c -o prof5 -lm && taskset -c 0 ./prof5