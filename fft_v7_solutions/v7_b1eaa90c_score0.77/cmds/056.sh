cd /workdir/dev && cat > pwtest.c <<'EOF'
#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>
typedef double V __attribute__((vector_size(64), aligned(64)));
#define VC(x) ((V){(x),(x),(x),(x),(x),(x),(x),(x)})
static inline uint64_t rdtscp(){ unsigned a; return __rdtscp(&a); }
static inline V vrsqrt14(V x){ return (V)_mm512_rsqrt14_pd((__m512d)x); }
static inline V vrcp14(V x){ return (V)_mm512_rcp14_pd((__m512d)x); }
static inline V vmaxv(V a, V b){ return (V)_mm512_max_pd((__m512d)a,(__m512d)b); }
static inline V vsqrt(V x){ return (V)_mm512_sqrt_pd((__m512d)x); }
static inline V pw_newton(V zr, V zi){
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
}
static inline V pw_sqrtdiv(V zr, V zi){
  V s = zr*zr + zi*zi;
  V u = VC(1.0) + vsqrt(s);
  return (V)_mm512_div_pd((__m512d)VC(1.0), (__m512d)u);
}
static inline V pw_sqrtnr(V zr, V zi){
  V s = zr*zr + zi*zi;
  V u = VC(1.0) + vsqrt(s);
  V v = vrcp14(u);
  v = v + v*(VC(1.0) - u*v);
  v = v + v*(VC(1.0) - u*v);
  return v;
}
static V Ar[512], Ai[512], Cr[512], Ci[512];
#define BENCH(name, fn) do{ \
  uint64_t t0=rdtscp(); \
  for(long r=0;r<R;r++) for(long i=0;i<512;i++){ \
    V zr=Ar[i]+Cr[i], zi=Ai[i]+Ci[i]; V f=fn(zr,zi); Ar[i]=zr*f; Ai[i]=zi*f; } \
  uint64_t t1=rdtscp(); \
  double s=0; for(int i=0;i<512;i++) s+=Ar[i][3]; \
  printf("%-10s %.3f c/elem (chk %.6g)\n", name, (double)(t1-t0)/R/512/8, s); \
}while(0)
int main(){
  for(int i=0;i<512;i++) for(int l=0;l<8;l++){ Ar[i][l]=0.2*((i+l)%9)-0.8; Ai[i][l]=0.15*((i*3+l)%7)-0.4; Cr[i][l]=0.05; Ci[i][l]=-0.04; }
  long R=30000;
  BENCH("newton", pw_newton);
  BENCH("sqrtdiv", pw_sqrtdiv);
  BENCH("sqrtnr", pw_sqrtnr);
  BENCH("newton2", pw_newton);
  // accuracy check vs exact
  double maxrel=0;
  for(int i=0;i<512;i++) for(int l=0;l<8;l++){
    double zr=Ar[i][l]+0.3, zi=Ai[i][l]-0.2;
    V a=VC(0.0),b=VC(0.0); a[0]=zr; b[0]=zi;
    V f1=pw_newton(a,b), f2=pw_sqrtnr(a,b);
    double fe=1.0/(1.0+__builtin_sqrt(zr*zr+zi*zi));
    double e1=(f1[0]-fe)/fe, e2=(f2[0]-fe)/fe;
    if(e1<0)e1=-e1; if(e2<0)e2=-e2;
    if(e1>maxrel)maxrel=e1; if(e2>maxrel)maxrel=e2;
  }
  printf("max rel err vs libm-chain: %.3e\n", maxrel);
  return 0;
}
EOF
gcc -O3 -march=native pwtest.c -o pwtest -lm && taskset -c 0 ./pwtest