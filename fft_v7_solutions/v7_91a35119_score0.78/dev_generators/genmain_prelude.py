PRELUDE_BASE = r'''
// Auto-generated batched 3D DFT iteration kernels. Single-threaded, AVX-512.
#include <immintrin.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef double v8df __attribute__((vector_size(64), aligned(8)));
typedef double v4df __attribute__((vector_size(32), aligned(8)));
typedef double v2df __attribute__((vector_size(16), aligned(8)));

#define K8(x) ((v8df){x,x,x,x,x,x,x,x})
#define K4(x) ((v4df){x,x,x,x})
#define K2(x) ((v2df){x,x})

static inline v8df maprec8(v8df m){
  __m512d M = (__m512d)m;
  __m512d u = _mm512_rsqrt14_pd(M);
  __m512d h = _mm512_mul_pd(M, _mm512_set1_pd(0.5));
  u = _mm512_mul_pd(u, _mm512_fnmadd_pd(_mm512_mul_pd(u,u), h, _mm512_set1_pd(1.5)));
  u = _mm512_mul_pd(u, _mm512_fnmadd_pd(_mm512_mul_pd(u,u), h, _mm512_set1_pd(1.5)));
  __m512d d = _mm512_add_pd(u, _mm512_set1_pd(1.0));
  __m512d v = _mm512_rcp14_pd(d);
  v = _mm512_mul_pd(v, _mm512_fnmadd_pd(d, v, _mm512_set1_pd(2.0)));
  v = _mm512_mul_pd(v, _mm512_fnmadd_pd(d, v, _mm512_set1_pd(2.0)));
  return (v8df)_mm512_mul_pd(u, v);
}
static inline v4df maprec4(v4df m){
  __m256d M = (__m256d)m;
  __m256d u = _mm256_rsqrt14_pd(M);
  __m256d h = _mm256_mul_pd(M, _mm256_set1_pd(0.5));
  u = _mm256_mul_pd(u, _mm256_fnmadd_pd(_mm256_mul_pd(u,u), h, _mm256_set1_pd(1.5)));
  u = _mm256_mul_pd(u, _mm256_fnmadd_pd(_mm256_mul_pd(u,u), h, _mm256_set1_pd(1.5)));
  __m256d d = _mm256_add_pd(u, _mm256_set1_pd(1.0));
  __m256d v = _mm256_rcp14_pd(d);
  v = _mm256_mul_pd(v, _mm256_fnmadd_pd(d, v, _mm256_set1_pd(2.0)));
  v = _mm256_mul_pd(v, _mm256_fnmadd_pd(d, v, _mm256_set1_pd(2.0)));
  return (v4df)_mm256_mul_pd(u, v);
}
static inline v2df maprec2(v2df m){
  __m128d M = (__m128d)m;
  __m128d u = _mm_rsqrt14_pd(M);
  __m128d h = _mm_mul_pd(M, _mm_set1_pd(0.5));
  u = _mm_mul_pd(u, _mm_fnmadd_pd(_mm_mul_pd(u,u), h, _mm_set1_pd(1.5)));
  u = _mm_mul_pd(u, _mm_fnmadd_pd(_mm_mul_pd(u,u), h, _mm_set1_pd(1.5)));
  __m128d d = _mm_add_pd(u, _mm_set1_pd(1.0));
  __m128d v = _mm_rcp14_pd(d);
  v = _mm_mul_pd(v, _mm_fnmadd_pd(d, v, _mm_set1_pd(2.0)));
  v = _mm_mul_pd(v, _mm_fnmadd_pd(d, v, _mm_set1_pd(2.0)));
  return (v2df)_mm_mul_pd(u, v);
}
static inline double maprec1(double m){ return 1.0/(1.0 + sqrt(m)); }
static inline v8df rsq8(v8df m){
  __m512d M = (__m512d)m;
  __m512d u = _mm512_rsqrt14_pd(M);
  __m512d h = _mm512_mul_pd(M, _mm512_set1_pd(0.5));
  u = _mm512_mul_pd(u, _mm512_fnmadd_pd(_mm512_mul_pd(u,u), h, _mm512_set1_pd(1.5)));
  u = _mm512_mul_pd(u, _mm512_fnmadd_pd(_mm512_mul_pd(u,u), h, _mm512_set1_pd(1.5)));
  return (v8df)u;
}
static inline v8df rpc8(v8df uu){
  __m512d d = _mm512_add_pd((__m512d)uu, _mm512_set1_pd(1.0));
  __m512d v = _mm512_rcp14_pd(d);
  v = _mm512_mul_pd(v, _mm512_fnmadd_pd(d, v, _mm512_set1_pd(2.0)));
  v = _mm512_mul_pd(v, _mm512_fnmadd_pd(d, v, _mm512_set1_pd(2.0)));
  return (v8df)v;
}
static inline v4df rsq4(v4df m){
  __m256d M = (__m256d)m;
  __m256d u = _mm256_rsqrt14_pd(M);
  __m256d h = _mm256_mul_pd(M, _mm256_set1_pd(0.5));
  u = _mm256_mul_pd(u, _mm256_fnmadd_pd(_mm256_mul_pd(u,u), h, _mm256_set1_pd(1.5)));
  u = _mm256_mul_pd(u, _mm256_fnmadd_pd(_mm256_mul_pd(u,u), h, _mm256_set1_pd(1.5)));
  return (v4df)u;
}
static inline v4df rpc4(v4df uu){
  __m256d d = _mm256_add_pd((__m256d)uu, _mm256_set1_pd(1.0));
  __m256d v = _mm256_rcp14_pd(d);
  v = _mm256_mul_pd(v, _mm256_fnmadd_pd(d, v, _mm256_set1_pd(2.0)));
  v = _mm256_mul_pd(v, _mm256_fnmadd_pd(d, v, _mm256_set1_pd(2.0)));
  return (v4df)v;
}
static inline v2df rsq2(v2df m){
  __m128d M = (__m128d)m;
  __m128d u = _mm_rsqrt14_pd(M);
  __m128d h = _mm_mul_pd(M, _mm_set1_pd(0.5));
  u = _mm_mul_pd(u, _mm_fnmadd_pd(_mm_mul_pd(u,u), h, _mm_set1_pd(1.5)));
  u = _mm_mul_pd(u, _mm_fnmadd_pd(_mm_mul_pd(u,u), h, _mm_set1_pd(1.5)));
  return (v2df)u;
}
static inline v2df rpc2(v2df uu){
  __m128d d = _mm_add_pd((__m128d)uu, _mm_set1_pd(1.0));
  __m128d v = _mm_rcp14_pd(d);
  v = _mm_mul_pd(v, _mm_fnmadd_pd(d, v, _mm_set1_pd(2.0)));
  v = _mm_mul_pd(v, _mm_fnmadd_pd(d, v, _mm_set1_pd(2.0)));
  return (v2df)v;
}
static inline double rsq1(double m){ return 1.0/sqrt(m); }
static inline double rpc1(double uu){ return 1.0/(1.0+uu); }


static inline void tr8x8_store(__m512d r0,__m512d r1,__m512d r2,__m512d r3,
                               __m512d r4,__m512d r5,__m512d r6,__m512d r7,
                               double* dst, long P){
  __m512d t0,t1,t2,t3,t4,t5,t6,t7, u0,u1,u2,u3,u4,u5,u6,u7;
  t0=_mm512_unpacklo_pd(r0,r1); t1=_mm512_unpackhi_pd(r0,r1);
  t2=_mm512_unpacklo_pd(r2,r3); t3=_mm512_unpackhi_pd(r2,r3);
  t4=_mm512_unpacklo_pd(r4,r5); t5=_mm512_unpackhi_pd(r4,r5);
  t6=_mm512_unpacklo_pd(r6,r7); t7=_mm512_unpackhi_pd(r6,r7);
  u0=_mm512_shuffle_f64x2(t0,t2,0x88); u1=_mm512_shuffle_f64x2(t1,t3,0x88);
  u2=_mm512_shuffle_f64x2(t0,t2,0xDD); u3=_mm512_shuffle_f64x2(t1,t3,0xDD);
  u4=_mm512_shuffle_f64x2(t4,t6,0x88); u5=_mm512_shuffle_f64x2(t5,t7,0x88);
  u6=_mm512_shuffle_f64x2(t4,t6,0xDD); u7=_mm512_shuffle_f64x2(t5,t7,0xDD);
  _mm512_storeu_pd(dst+0*P, _mm512_shuffle_f64x2(u0,u4,0x88));
  _mm512_storeu_pd(dst+1*P, _mm512_shuffle_f64x2(u1,u5,0x88));
  _mm512_storeu_pd(dst+2*P, _mm512_shuffle_f64x2(u2,u6,0x88));
  _mm512_storeu_pd(dst+3*P, _mm512_shuffle_f64x2(u3,u7,0x88));
  _mm512_storeu_pd(dst+4*P, _mm512_shuffle_f64x2(u0,u4,0xDD));
  _mm512_storeu_pd(dst+5*P, _mm512_shuffle_f64x2(u1,u5,0xDD));
  _mm512_storeu_pd(dst+6*P, _mm512_shuffle_f64x2(u2,u6,0xDD));
  _mm512_storeu_pd(dst+7*P, _mm512_shuffle_f64x2(u3,u7,0xDD));
}
static inline void tr4x4_store(__m256d r0,__m256d r1,__m256d r2,__m256d r3,
                               double* dst, long P){
  __m256d t0=_mm256_unpacklo_pd(r0,r1), t1=_mm256_unpackhi_pd(r0,r1);
  __m256d t2=_mm256_unpacklo_pd(r2,r3), t3=_mm256_unpackhi_pd(r2,r3);
  _mm256_storeu_pd(dst+0*P, _mm256_permute2f128_pd(t0,t2,0x20));
  _mm256_storeu_pd(dst+1*P, _mm256_permute2f128_pd(t1,t3,0x20));
  _mm256_storeu_pd(dst+2*P, _mm256_permute2f128_pd(t0,t2,0x31));
  _mm256_storeu_pd(dst+3*P, _mm256_permute2f128_pd(t1,t3,0x31));
}
static void* xalloc(size_t bytes){
  size_t n = (bytes + 63) & ~(size_t)63;
  void* p = aligned_alloc(64, n);
  memset(p, 0, n);
  return p;
}
'''
