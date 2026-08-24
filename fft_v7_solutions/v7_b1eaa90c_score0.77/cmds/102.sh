cd /workdir/dev && cat > prof17.c <<'EOF'
#include <stdio.h>
#include <stdint.h>
#include "../implementation.c"
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
int main(){
  long R; uint64_t t0,t1;
  // --- L=23 SoA phases ---
  for(long i=0;i<12167;i++) for(int l=0;l<8;l++){ soa23_xr[i][l]=0.3*((i*7+l)%13)-1.0; soa23_xi[i][l]=0.2*((i*5+l)%11)-0.8; soa23_cr[i][l]=0.05; soa23_ci[i][l]=-0.03; }
  R=100; double el=12167*8;
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long x = 0; x < 23; x++) {
      V *sr = soa23_xr + x*529, *si = soa23_xi + x*529;
      for (long y = 0; y < 23; y++) f23_z(sr + y*23, si + y*23, sr + y*23, si + y*23);
      for (long z = 0; z < 23; z++) f23_y(sr + z, si + z, sr + z, si + z);
    }
  t1=rdtscp(); printf("23 zy: %.3f c/elem\n", (double)(t1-t0)/R/el);
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long u = 0; u < 529; u++)
      f23_x(soa23_xr + u, soa23_xi + u, soa23_xr + u, soa23_xi + u, soa23_cr + u, soa23_ci + u);
  t1=rdtscp(); printf("23 x+pw: %.3f c/elem\n", (double)(t1-t0)/R/el);
  // --- L=36 sq phases ---
  for(long i=0;i<36*180;i++) for(int l=0;l<8;l++){ sq36_ar[i][l]=0.3*((i*7+l)%13)-1.0; sq36_ai[i][l]=0.2*((i*5+l)%11)-0.8; sq36_br[i][l]=0.1; sq36_bi[i][l]=0.1; sq36_cr[i][l]=0.05; sq36_ci[i][l]=-0.03; }
  R=200; el=46656;
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long g = 0; g < 6; g++) {
      f36_s2(sq36_br + g*6*180, sq36_bi + g*6*180, sq36_cr + g*6*180, sq36_ci + g*6*180, sq36_tr, sq36_ti);
      for (long k2 = 0; k2 < 6; k2++) sq36_yz(sq36_tr + k2*180, sq36_ti + k2*180);
      f36_s1(sq36_tr, sq36_ti, sq36_ar + g*180, sq36_ai + g*180, sq36_twr + g*6, sq36_twi + g*6);
    }
  t1=rdtscp(); printf("36 sweep: %.3f c/elem\n", (double)(t1-t0)/R/el);
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long g = 0; g < 6; g++)
      for (long k2 = 0; k2 < 6; k2++) sq36_yz(sq36_tr + k2*180, sq36_ti + k2*180);
  t1=rdtscp(); printf("36 yz-only: %.3f c/elem\n", (double)(t1-t0)/R/el);
  // --- L=45 slab phases ---
  for(long i=0;i<45*270;i++) for(int l=0;l<8;l++){ slab45_xr[i][l]=0.3*((i*7+l)%13)-1.0; slab45_xi[i][l]=0.2*((i*5+l)%11)-0.8; slab45_cer[i][l]=0.05; slab45_cei[i][l]=-0.03; }
  R=100; el=91125;
  t0=rdtscp();
  for(long r=0;r<R;r++) slab45_iter(1);
  t1=rdtscp(); printf("45 full iter: %.3f c/elem\n", (double)(t1-t0)/R/el);
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long o = 0; o < 45*6; o++)
      f45_k3(slab45_xr + o, slab45_xi + o, slab45_xr + o, slab45_xi + o, slab45_cer + o, slab45_cei + o);
  t1=rdtscp(); printf("45 K3: %.3f c/elem\n", (double)(t1-t0)/R/el);
  // --- L=64 sq phases ---
  for(long i=0;i<64*513;i++) for(int l=0;l<8;l++){ sq64_ar[i][l]=0.3*((i*7+l)%13)-1.0; sq64_ai[i][l]=0.2*((i*5+l)%11)-0.8; sq64_br[i][l]=0.1; sq64_bi[i][l]=0.1; sq64_cr[i][l]=0.05; sq64_ci[i][l]=-0.03; }
  R=30; el=262144;
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long g = 0; g < 8; g++)
      for (long k2 = 0; k2 < 8; k2++) sq64_yz(sq64_tr + k2*513, sq64_ti + k2*513);
  t1=rdtscp(); printf("64 yz-only (new): %.3f c/elem\n", (double)(t1-t0)/R/el);
  double s=0; for(long i=0;i<36*180;i++) s+=sq36_ar[i][3]; printf("chk %g\n", s);
  return 0;
}
EOF
gcc -O3 -march=native prof17.c -o prof17 -lm && taskset -c 0 ./prof17