cd /workdir/dev && python3 - <<'EOF'
src=open('../implementation.c').read()
old="""static inline __attribute__((always_inline)) V pw_factor(V zr, V zi){
  V s = zr*zr + zi*zi;
  s = vmaxv(s, VC(2.2250738585072014e-308));
  V r = vrsqrt14(s);
  V h = s * VC(0.5);
  r = r * (VC(1.5) - h*r*r);
  r = r * (VC(1.5) - h*r*r);
  V t = s * r;
  V u = VC(1.0) + t;
  V v = vrcp14(u);
  v = v + v*(VC(1.0) - u*v);
  v = v + v*(VC(1.0) - u*v);
  return v;
}"""
new="""static inline V vsqrtv(V x){ return (V)_mm512_sqrt_pd((__m512d)x); }
static inline __attribute__((always_inline)) V pw_factor(V zr, V zi){
  V s = zr*zr + zi*zi;
  V u = VC(1.0) + vsqrtv(s);
  V v = vrcp14(u);
  v = v + v*(VC(1.0) - u*v);
  v = v + v*(VC(1.0) - u*v);
  return v;
}"""
assert old in src
open('impl_sqrtnr.c','w').write(src.replace(old,new))
EOF
cat > prof12.c <<'EOF'
#include <stdio.h>
#include <stdint.h>
#ifdef SQRTNR
#include "impl_sqrtnr.c"
#else
#include "../implementation.c"
#endif
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
int main(){
  for(long i=0;i<36*180;i++) for(int l=0;l<8;l++){ slab36_xr[i][l]=0.3*((i*7+l)%13)-1.0; slab36_xi[i][l]=0.2*((i*5+l)%11)-0.8; slab36_cer[i][l]=0.05; slab36_cei[i][l]=-0.03; }
  for(long i=0;i<64*513;i++) for(int l=0;l<8;l++){ slab64_xr[i][l]=0.3*((i*7+l)%13)-1.0; slab64_xi[i][l]=0.2*((i*5+l)%11)-0.8; slab64_cer[i][l]=0.05; slab64_cei[i][l]=-0.03; }
  long R=20000; uint64_t t0,t1;
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long o = 0; o < 8; o++)
      f36_k3(slab36_xr + o, slab36_xi + o, slab36_xr + o, slab36_xi + o, slab36_cer + o, slab36_cei + o);
  t1=rdtscp(); printf("L=36 K3 hot: %.3f c/elem\n", (double)(t1-t0)/R/8/(36*8));
  R=300;
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long o = 0; o < 36*5; o++)
      f36_k3(slab36_xr + o, slab36_xi + o, slab36_xr + o, slab36_xi + o, slab36_cer + o, slab36_cei + o);
  t1=rdtscp(); printf("L=36 K3 full: %.3f c/elem\n", (double)(t1-t0)/R/46656);
  R=2000;
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long o = 0; o < 8; o++)
      f64_k3(slab64_xr + o, slab64_xi + o, slab64_xr + o, slab64_xi + o, slab64_cer + o, slab64_cei + o);
  t1=rdtscp(); printf("L=64 K3 hot: %.3f c/elem\n", (double)(t1-t0)/R/8/(64*8));
  R=30;
  t0=rdtscp();
  for(long r=0;r<R;r++)
    for (long o = 0; o < 512; o++)
      f64_k3(slab64_xr + o, slab64_xi + o, slab64_xr + o, slab64_xi + o, slab64_cer + o, slab64_cei + o);
  t1=rdtscp(); printf("L=64 K3 full: %.3f c/elem\n", (double)(t1-t0)/R/262144);
  double s=0; for(long i=0;i<36*180;i++) s+=slab36_xr[i][3]; printf("chk %g\n", s);
  return 0;
}
EOF
gcc -O3 -march=native prof12.c -o prof12a -lm && gcc -O3 -march=native -DSQRTNR prof12.c -o prof12b -lm && echo "== newton ==" && taskset -c 0 ./prof12a && echo "== sqrt+nr ==" && taskset -c 0 ./prof12b