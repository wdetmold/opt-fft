cd /workdir/dev && cat > prof20.c <<'EOF'
#include <stdio.h>
#include <stdint.h>
#include "../implementation.c"
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
static V BUF[4096] __attribute__((aligned(4096)));
int main(){
  for(int i=0;i<4096;i++) for(int l=0;l<8;l++) BUF[i][l]=0.1*((i+l)%7)-0.3;
  long R=200000; uint64_t t0,t1;
  // f64_cb: in at BUF, out at BUF + delta, vary delta mod 4096 bytes
  for (int dv = 0; dv < 4; dv++) {
    long deltas[4] = {1024, 1032, 1028, 1039};   // V units: 1024V=64KB (4K-aliased), 1032V=+512B, etc
    long d = deltas[dv];
    t0=rdtscp();
    for(long r=0;r<R;r++)
      f64_cb(BUF + (r&7), BUF + 512 + (r&7), BUF + d, BUF + d + 512, 0, 0); // wait signature (xr,xi,yr,yi)
    t1=rdtscp();
    printf("delta %ld V (%ld B mod 4096 = %ld): %.1f c/call\n", d, d*64, (d*64)%4096, (double)(t1-t0)/R);
  }
  double s=0; for(int i=0;i<64;i++)s+=BUF[2048+i][1];
  printf("chk %g\n", s);
  return 0;
}
EOF
gcc -O3 -march=native prof20.c -o prof20 -lm 2>&1 | head -3