cd /workdir/dev && cat > prof19.c <<'EOF'
#include <stdio.h>
#include <stdint.h>
#include "../implementation.c"
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
int main(){
  for(long i=0;i<45*270;i++) for(int l=0;l<8;l++){ slab45_xr[i][l]=0.3*((i*7+l)%13)-1.0; slab45_xi[i][l]=0.2*((i*5+l)%11)-0.8; }
  long R=100; uint64_t t0,t1; double el=91125;
  for (long j = 45; j < 48; j++) { slab45_br[j] = VC(0.0); slab45_bi[j] = VC(0.0); slab45_br[48+j]=VC(0.0); slab45_bi[48+j]=VC(0.0);}
  // K1 kernels only
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long x = 0; x < 45; x++) {
      V *sr = slab45_xr + x*270, *si = slab45_xi + x*270;
      for (long cc = 0; cc < 6; cc++) {
        V *br_ = slab45_br + (cc&1)*48, *bi_ = slab45_bi + (cc&1)*48;
        f45_cb(sr + cc, si + cc, br_, bi_);
      }
    }
  t1=rdtscp(); printf("45 K1 kernels: %.3f c/elem\n", (double)(t1-t0)/R/el);
  // K1 + transpose to TSLAB
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long x = 0; x < 45; x++) {
      V *sr = slab45_xr + x*270, *si = slab45_xi + x*270;
      for (long cc = 0; cc < 6; cc++) {
        V *br_ = slab45_br + (cc&1)*48, *bi_ = slab45_bi + (cc&1)*48;
        f45_cb(sr + cc, si + cc, br_, bi_);
        for (long rb = 0; rb < 6; rb++) {
          V tb[8];
          tr8x8(br_ + rb*8, tb);
          for (int q = 0; q < 8; q++) slab45_tsr[(cc*8+q)*6 + rb] = tb[q];
          tr8x8(bi_ + rb*8, tb);
          for (int q = 0; q < 8; q++) slab45_tsi[(cc*8+q)*6 + rb] = tb[q];
        }
      }
    }
  t1=rdtscp(); printf("45 K1+tr: %.3f c/elem\n", (double)(t1-t0)/R/el);
  // K2 (TSLAB->slab with guard)
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long x = 0; x < 45; x++) {
      V *sr = slab45_xr + x*270, *si = slab45_xi + x*270;
      for (long cc = 0; cc < 6; cc++) {
        V *br_ = slab45_br + (cc&1)*48, *bi_ = slab45_bi + (cc&1)*48;
        f45_cb(slab45_tsr + cc, slab45_tsi + cc, br_, bi_);
        int lim = (int)(45 - cc*8); if (lim > 8) lim = 8;
        for (long rb = 0; rb < 6; rb++) {
          V tb[8];
          tr8x8(br_ + rb*8, tb);
          for (int q = 0; q < lim; q++) sr[(cc*8+q)*6 + rb] = tb[q];
          tr8x8(bi_ + rb*8, tb);
          for (int q = 0; q < lim; q++) si[(cc*8+q)*6 + rb] = tb[q];
        }
      }
    }
  t1=rdtscp(); printf("45 K2+tr: %.3f c/elem\n", (double)(t1-t0)/R/el);
  double s=0; for(long i=0;i<45*270;i++) s+=slab45_xr[i][3]; printf("chk %g\n", s);
  return 0;
}
EOF
gcc -O3 -march=native prof19.c -o prof19 -lm && taskset -c 0 ./prof19