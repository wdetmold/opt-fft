
/* Auto-generated: specialized iterated batched 3D FFT + map.  AVX-512. */
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h>
#include <sys/mman.h>

typedef double vd __attribute__((vector_size(64), aligned(64)));
#define VL(p) (*(const vd *)(p))
#define VS(p, v) (*(vd *)(p) = (v))
#define VK(c) ((vd){(c),(c),(c),(c),(c),(c),(c),(c)})
#define VRSQRT(x) ((vd)_mm512_rsqrt14_pd((__m512d)(x)))
#define VRCP(x)   ((vd)_mm512_rcp14_pd((__m512d)(x)))

static inline void tr8x8_ld(const double *restrict src, long rs, double *restrict dst){
  __m512d r0=_mm512_load_pd(src+0*rs), r1=_mm512_load_pd(src+1*rs),
          r2=_mm512_load_pd(src+2*rs), r3=_mm512_load_pd(src+3*rs),
          r4=_mm512_load_pd(src+4*rs), r5=_mm512_load_pd(src+5*rs),
          r6=_mm512_load_pd(src+6*rs), r7=_mm512_load_pd(src+7*rs);
  __m512d t0=_mm512_unpacklo_pd(r0,r1), t1=_mm512_unpackhi_pd(r0,r1),
          t2=_mm512_unpacklo_pd(r2,r3), t3=_mm512_unpackhi_pd(r2,r3),
          t4=_mm512_unpacklo_pd(r4,r5), t5=_mm512_unpackhi_pd(r4,r5),
          t6=_mm512_unpacklo_pd(r6,r7), t7=_mm512_unpackhi_pd(r6,r7);
  __m512d u0=_mm512_shuffle_f64x2(t0,t2,0x88), u1=_mm512_shuffle_f64x2(t1,t3,0x88),
          u2=_mm512_shuffle_f64x2(t0,t2,0xdd), u3=_mm512_shuffle_f64x2(t1,t3,0xdd),
          u4=_mm512_shuffle_f64x2(t4,t6,0x88), u5=_mm512_shuffle_f64x2(t5,t7,0x88),
          u6=_mm512_shuffle_f64x2(t4,t6,0xdd), u7=_mm512_shuffle_f64x2(t5,t7,0xdd);
  _mm512_store_pd(dst+0*8, _mm512_shuffle_f64x2(u0,u4,0x88));
  _mm512_store_pd(dst+1*8, _mm512_shuffle_f64x2(u1,u5,0x88));
  _mm512_store_pd(dst+2*8, _mm512_shuffle_f64x2(u2,u6,0x88));
  _mm512_store_pd(dst+3*8, _mm512_shuffle_f64x2(u3,u7,0x88));
  _mm512_store_pd(dst+4*8, _mm512_shuffle_f64x2(u0,u4,0xdd));
  _mm512_store_pd(dst+5*8, _mm512_shuffle_f64x2(u1,u5,0xdd));
  _mm512_store_pd(dst+6*8, _mm512_shuffle_f64x2(u2,u6,0xdd));
  _mm512_store_pd(dst+7*8, _mm512_shuffle_f64x2(u3,u7,0xdd));
}
static inline void tr8x8_st(double *restrict dst, long rs, const double *restrict src, int rows){
  __m512d r0=_mm512_load_pd(src+0*8), r1=_mm512_load_pd(src+1*8),
          r2=_mm512_load_pd(src+2*8), r3=_mm512_load_pd(src+3*8),
          r4=_mm512_load_pd(src+4*8), r5=_mm512_load_pd(src+5*8),
          r6=_mm512_load_pd(src+6*8), r7=_mm512_load_pd(src+7*8);
  __m512d t0=_mm512_unpacklo_pd(r0,r1), t1=_mm512_unpackhi_pd(r0,r1),
          t2=_mm512_unpacklo_pd(r2,r3), t3=_mm512_unpackhi_pd(r2,r3),
          t4=_mm512_unpacklo_pd(r4,r5), t5=_mm512_unpackhi_pd(r4,r5),
          t6=_mm512_unpacklo_pd(r6,r7), t7=_mm512_unpackhi_pd(r6,r7);
  __m512d u0=_mm512_shuffle_f64x2(t0,t2,0x88), u1=_mm512_shuffle_f64x2(t1,t3,0x88),
          u2=_mm512_shuffle_f64x2(t0,t2,0xdd), u3=_mm512_shuffle_f64x2(t1,t3,0xdd),
          u4=_mm512_shuffle_f64x2(t4,t6,0x88), u5=_mm512_shuffle_f64x2(t5,t7,0x88),
          u6=_mm512_shuffle_f64x2(t4,t6,0xdd), u7=_mm512_shuffle_f64x2(t5,t7,0xdd);
  __m512d o0=_mm512_shuffle_f64x2(u0,u4,0x88), o1=_mm512_shuffle_f64x2(u1,u5,0x88),
          o2=_mm512_shuffle_f64x2(u2,u6,0x88), o3=_mm512_shuffle_f64x2(u3,u7,0x88),
          o4=_mm512_shuffle_f64x2(u0,u4,0xdd), o5=_mm512_shuffle_f64x2(u1,u5,0xdd),
          o6=_mm512_shuffle_f64x2(u2,u6,0xdd), o7=_mm512_shuffle_f64x2(u3,u7,0xdd);
  switch(rows){
    case 8: _mm512_store_pd(dst+7*rs,o7); /* fallthrough */
    case 7: _mm512_store_pd(dst+6*rs,o6);
    case 6: _mm512_store_pd(dst+5*rs,o5);
    case 5: _mm512_store_pd(dst+4*rs,o4);
    case 4: _mm512_store_pd(dst+3*rs,o3);
    case 3: _mm512_store_pd(dst+2*rs,o2);
    case 2: _mm512_store_pd(dst+1*rs,o1);
    case 1: _mm512_store_pd(dst+0*rs,o0);
  }
}
static inline void deint_row(const double *restrict src, double *restrict dre, double *restrict dim_, int n, int r){
  const __m512i IR = _mm512_set_epi64(14,12,10,8,6,4,2,0);
  const __m512i II = _mm512_set_epi64(15,13,11,9,7,5,3,1);
  int j = 0;
  for(; j + 8 <= n; j += 8){
    __m512d a = _mm512_loadu_pd(src + 2*j), b = _mm512_loadu_pd(src + 2*j + 8);
    _mm512_store_pd(dre + j, _mm512_permutex2var_pd(a, IR, b));
    _mm512_store_pd(dim_ + j, _mm512_permutex2var_pd(a, II, b));
  }
  if(j < n){
    int rem = n - j;
    __mmask8 mlo = (__mmask8)((rem >= 4) ? 0xff : ((1u << (2*rem)) - 1));
    __mmask8 mhi = (__mmask8)((rem <= 4) ? 0 : ((1u << (2*rem - 8)) - 1));
    __mmask8 mo  = (__mmask8)((1u << rem) - 1);
    __m512d a = _mm512_maskz_loadu_pd(mlo, src + 2*j);
    __m512d b = _mm512_maskz_loadu_pd(mhi, src + 2*j + 8);
    _mm512_mask_store_pd(dre + j, mo, _mm512_permutex2var_pd(a, IR, b));
    _mm512_mask_store_pd(dim_ + j, mo, _mm512_permutex2var_pd(a, II, b));
    j = n;
  }
  for(; j < r; j++){ dre[j] = 0.0; dim_[j] = 0.0; }
}
static inline void int_row(double *restrict dst, const double *restrict sre, const double *restrict sim, int n){
  const __m512i ILO = _mm512_set_epi64(11,3,10,2,9,1,8,0);
  const __m512i IHI = _mm512_set_epi64(15,7,14,6,13,5,12,4);
  int j = 0;
  for(; j + 8 <= n; j += 8){
    __m512d re = _mm512_load_pd(sre + j), im = _mm512_load_pd(sim + j);
    _mm512_storeu_pd(dst + 2*j,     _mm512_permutex2var_pd(re, ILO, im));
    _mm512_storeu_pd(dst + 2*j + 8, _mm512_permutex2var_pd(re, IHI, im));
  }
  if(j < n){
    int rem = n - j;
    __mmask8 mlo = (__mmask8)((rem >= 4) ? 0xff : ((1u << (2*rem)) - 1));
    __mmask8 mhi = (__mmask8)((rem <= 4) ? 0 : ((1u << (2*rem - 8)) - 1));
    __m512d re = _mm512_load_pd(sre + j), im = _mm512_load_pd(sim + j);
    _mm512_mask_storeu_pd(dst + 2*j,     mlo, _mm512_permutex2var_pd(re, ILO, im));
    _mm512_mask_storeu_pd(dst + 2*j + 8, mhi, _mm512_permutex2var_pd(re, IHI, im));
  }
}
static double *alloc_buf(size_t doubles){
  size_t bytes = (doubles * 8 + (1u<<21) - 1) & ~(size_t)((1u<<21) - 1);
  void *p = aligned_alloc(1u<<21, bytes);
  madvise(p, bytes, MADV_HUGEPAGE);
  memset(p, 0, bytes);
  return (double *)p;
}

static inline __attribute__((always_inline)) void map1(vd zr, vd zi, vd *or_, vd *oi){
  vd r2 = zr*zr + zi*zi + VK(1e-300);
  vd u = VRSQRT(r2);
  vd h = VK(0.5) * r2;
  u = u * (VK(1.5) - h*u*u);
  u = u * (VK(1.5) - h*u*u);
  vd t = r2*u + VK(1.0);
  vd y = VRCP(t);
  y = y + y*(VK(1.0) - t*y);
  y = y + y*(VK(1.0) - t*y);
  *or_ = zr*y; *oi = zi*y;
}
static void map_rows(double *restrict dr, double *restrict di, long n, long st,
                     const double *restrict cr, const double *restrict ci, long cst,
                     double *restrict sr, double *restrict si){
  if(__builtin_expect(sr != 0, 0)){
    for(long j = 0; j < n; j++){
      vd zr = VL(dr + j*st) + VL(cr + j*cst), zi = VL(di + j*st) + VL(ci + j*cst);
      vd orr, oii; map1(zr, zi, &orr, &oii);
      VS(sr + j*cst, orr); VS(si + j*cst, oii);
      VS(dr + j*st, orr); VS(di + j*st, oii);
    }
  } else {
    for(long j = 0; j < n - 1; j += 2){
      vd zr0 = VL(dr + j*st) + VL(cr + j*cst), zi0 = VL(di + j*st) + VL(ci + j*cst);
      vd zr1 = VL(dr + (j+1)*st) + VL(cr + (j+1)*cst), zi1 = VL(di + (j+1)*st) + VL(ci + (j+1)*cst);
      vd or0, oi0, or1, oi1;
      map1(zr0, zi0, &or0, &oi0); map1(zr1, zi1, &or1, &oi1);
      VS(dr + j*st, or0); VS(di + j*st, oi0);
      VS(dr + (j+1)*st, or1); VS(di + (j+1)*st, oi1);
    }
    if(n & 1){
      long j = n - 1;
      vd zr = VL(dr + j*st) + VL(cr + j*cst), zi = VL(di + j*st) + VL(ci + j*cst);
      vd orr, oii; map1(zr, zi, &orr, &oii);
      VS(dr + j*st, orr); VS(di + j*st, oii);
    }
  }
}
