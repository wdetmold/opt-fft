cd /workdir/dev && cat > prof11.c <<'EOF'
#include <stdio.h>
#include <stdint.h>
#include "../implementation.c"
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
int main(){
  for(long i=0;i<36*180;i++) for(int l=0;l<8;l++){ slab36_xr[i][l]=0.3*((i*7+l)%13)-1.0; slab36_xi[i][l]=0.2*((i*5+l)%11)-0.8; slab36_cer[i][l]=0.05; slab36_cei[i][l]=-0.03; }
  long R=20000; uint64_t t0,t1;
  // K3 kernel on L1/L2-hot narrow region (8 columns only)
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long o = 0; o < 8; o++)
      f36_k3(slab36_xr + o, slab36_xi + o, slab36_xr + o, slab36_xi + o, slab36_cer + o, slab36_cei + o);
  t1=rdtscp(); printf("L=36 K3 hot(8 cols): %.3f c/elem-col (%.0f c/call)\n", (double)(t1-t0)/R/8/(36*8), (double)(t1-t0)/R/8);
  // K3 kernel without pw on same (use cb into buf? different shape) - measure pw separately below
  // pure pw loop L1-hot
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long idx = 0; idx < 36*8; idx++) {
      V zr = slab36_xr[idx] + slab36_cer[idx];
      V zi = slab36_xi[idx] + slab36_cei[idx];
      V f = pw_factor(zr, zi);
      slab36_xr[idx] = zr*f; slab36_xi[idx] = zi*f;
    }
  t1=rdtscp(); printf("pw hot: %.3f c/elem\n", (double)(t1-t0)/R/(36*8*8));
  double s=0; for(long i=0;i<36*180;i++) s+=slab36_xr[i][3]; printf("chk %g\n", s);
  return 0;
}
EOF
gcc -O3 -march=native prof11.c -o prof11 -lm && taskset -c 0 ./prof11