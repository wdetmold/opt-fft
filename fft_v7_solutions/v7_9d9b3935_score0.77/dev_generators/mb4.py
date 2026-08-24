import subprocess, gen

H = r'''
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
static double now(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+1e-9*ts.tv_nsec; }
CODELET
enum { L = 64, XS = 4136 };
int main(){
  size_t n = (size_t)L*XS + 64;
  size_t bytes = 6*n*8 + (4<<20);
  char* raw = mmap(0, bytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  raw = (char*)((((uintptr_t)raw) + (2<<20)-1) & ~(uintptr_t)((2<<20)-1));
  madvise(raw, 6*n*8, MADV_HUGEPAGE);
  memset(raw, 0, 6*n*8);
  double* tr = (double*)raw; double* ti = tr + n; double* sr = ti + n; double* si = sr + n; double* cr = si + n; double* ci = cr + n;
  for (size_t i = 0; i < n; i++){ tr[i] = 0.001*(double)(i%1000) - 0.5; ti[i] = 0.0002*(double)(i%997); cr[i] = 0.01; ci[i] = -0.02; }
  int R = 50; double t0, t1;
  t0 = now();
  for (int r = 0; r < R; r++)
    for (int p = 0; p + 8 <= L*L; p += 8)
      FN(tr + p, ti + p, XS, sr + p, si + p, XS, cr + p, ci + p);
  t1 = now();
  printf("%7.3f ms/pass  %5.2f ns/pt\n", (t1-t0)/R*1e3, (t1-t0)/R/((double)L*L*L)*1e9);
}
'''
def bench(text, name, label):
    src = H.replace('CODELET', gen.PRELUDE + MAPDEF + text).replace('FN', name)
    open('mb.c','w').write(src)
    subprocess.run(['gcc','-O3','-march=native','-ffp-contract=fast','mb.c','-o','mb'], check=True)
    out = subprocess.run(['taskset','-c','0','./mb'], capture_output=True, text=True).stdout.strip()
    print(f"{label:28s}: {out}")

MAPDEF = r'''
#include <immintrin.h>
static inline void map8(V8 zr, V8 zi, V8* mr, V8* mi){
  __m512d R = (__m512d)zr, I = (__m512d)zi;
  __m512d r2 = _mm512_fmadd_pd(I, I, _mm512_mul_pd(R, R));
  __m512d y = _mm512_rsqrt14_pd(r2);
  __m512d h = _mm512_mul_pd(r2, _mm512_set1_pd(0.5));
  __m512d t = _mm512_mul_pd(h, y);
  y = _mm512_mul_pd(y, _mm512_fnmadd_pd(t, y, _mm512_set1_pd(1.5)));
  t = _mm512_mul_pd(h, y);
  y = _mm512_mul_pd(y, _mm512_fnmadd_pd(t, y, _mm512_set1_pd(1.5)));
  __m512d r = _mm512_mul_pd(r2, y);
  __m512d dd = _mm512_add_pd(r, _mm512_set1_pd(1.0));
  __m512d w = _mm512_rcp14_pd(dd);
  w = _mm512_mul_pd(w, _mm512_fnmadd_pd(dd, w, _mm512_set1_pd(2.0)));
  w = _mm512_mul_pd(w, _mm512_fnmadd_pd(dd, w, _mm512_set1_pd(2.0)));
  *mr = (V8)_mm512_mul_pd(R, w);
  *mi = (V8)_mm512_mul_pd(I, w);
}
#define MAPCALL map8
'''

for pf, pfc in ((0,0),(8,0),(8,8),(16,16),(8,16),(24,8)):
    t = gen.emit_codelet_staged(64, 8, ('ct',8,8), mapstore=True, pf=pf, pfc=pfc)
    bench(t, 'fftmap64_w8', f'staged ct88 pf={pf} pfc={pfc}')
