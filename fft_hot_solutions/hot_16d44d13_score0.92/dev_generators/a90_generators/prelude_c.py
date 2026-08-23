PRELUDE = r'''
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ALIGN64 __attribute__((aligned(64)))
#include <sys/mman.h>
static double* alloc_huge(long bytes){
    long HP = (long)2<<20;
    bytes = (bytes + HP - 1) & ~(HP-1);
    void* p = mmap(0, bytes + HP, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if(p == MAP_FAILED) { void* q = aligned_alloc(64, bytes); memset(q,0,bytes); return (double*)q; }
    char* a = (char*)(((unsigned long)p + HP - 1) & ~((unsigned long)HP-1));
    if(a > (char*)p) munmap(p, a - (char*)p);
    long tail = ((char*)p + bytes + HP) - (a + bytes);
    if(tail > 0) munmap(a + bytes, tail);
    madvise(a, bytes, MADV_HUGEPAGE);
    memset(a, 0, bytes);
    return (double*)a;
}
static long stagger_ctr = 0;
static double* alloc_huge_st(long bytes){
    // distinct sub-page stagger per buffer to avoid cross-buffer set aliasing
    long st = ((stagger_ctr++ % 29) + 1) * 4672;   // 4672 = 73*64 bytes, odd line multiple
    double* p = alloc_huge(bytes + st + 4096);
    return (double*)((char*)p + st);
}

static const __m512d VONE_ = { 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 };
#define VONE VONE_
static const __m512d VHALF_ = { 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5 };
#define VHALF VHALF_

#define FMA_BC(acc, s, mem)  __asm__("vfmadd231pd %2%{1to8%}, %1, %0" : "+v"(acc) : "v"(s), "m"(mem))
#define FNMA_BC(acc, s, mem) __asm__("vfnmadd231pd %2%{1to8%}, %1, %0" : "+v"(acc) : "v"(s), "m"(mem))
#define MUL_BC(dst, s, mem)  __asm__("vmulpd %2%{1to8%}, %1, %0" : "=v"(dst) : "v"(s), "m"(mem))
#define BCAST(dst, mem) __asm__("vbroadcastsd %1, %0" : "=v"(dst) : "m"(mem))
#define BCASTV(dst, mem) dst = _mm512_set1_pd(*(volatile const double*)&(mem))

static inline void map2(__m512d zr, __m512d zi, __m512d* oxr, __m512d* oxi){
    const __m512d TINY = _mm512_set1_pd(1e-30);
    __m512d m  = _mm512_fmadd_pd(zr, zr, _mm512_fmadd_pd(zi, zi, TINY));
    __m512d r0 = _mm512_rsqrt14_pd(m);
    __m512d t  = _mm512_mul_pd(m, r0);
    __m512d hr = _mm512_mul_pd(r0, VHALF);
    __m512d eh = _mm512_fnmadd_pd(t, hr, VHALF);
    __m512d r1 = _mm512_fmadd_pd(r0, eh, r0);
    __m512d mg0= _mm512_mul_pd(m, r1);
    __m512d hr1= _mm512_mul_pd(r1, VHALF);
    __m512d e2 = _mm512_fnmadd_pd(mg0, mg0, m);
    __m512d mag= _mm512_fmadd_pd(e2, hr1, mg0);
    __m512d u  = _mm512_add_pd(VONE, mag);
    __m512d w0 = _mm512_rcp14_pd(u);
    __m512d e3 = _mm512_fnmadd_pd(u, w0, VONE);
    __m512d a  = _mm512_fmadd_pd(w0, e3, w0);
    __m512d ee = _mm512_mul_pd(e3, e3);
    __m512d w2 = _mm512_fmadd_pd(a, ee, a);
    *oxr = _mm512_mul_pd(zr, w2);
    *oxi = _mm512_mul_pd(zi, w2);
}

// map over contiguous element-vec range, layout [e][2][8]; x and c same layout
static inline void map_range(double* x, const double* c, long n){
    long e=0;
    for(; e+2<=n; e+=2){
        __m512d xr0 = _mm512_load_pd(x + e*16);
        __m512d xi0 = _mm512_load_pd(x + e*16 + 8);
        __m512d xr1 = _mm512_load_pd(x + e*16 + 16);
        __m512d xi1 = _mm512_load_pd(x + e*16 + 24);
        __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(c + e*16));
        __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(c + e*16 + 8));
        __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(c + e*16 + 16));
        __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(c + e*16 + 24));
        map2(zr0, zi0, &xr0, &xi0);
        map2(zr1, zi1, &xr1, &xi1);
        _mm512_store_pd(x + e*16, xr0);
        _mm512_store_pd(x + e*16 + 8, xi0);
        _mm512_store_pd(x + e*16 + 16, xr1);
        _mm512_store_pd(x + e*16 + 24, xi1);
    }
    for(; e<n; e++){
        __m512d xr = _mm512_load_pd(x + e*16);
        __m512d xi = _mm512_load_pd(x + e*16 + 8);
        __m512d zr = _mm512_add_pd(xr, _mm512_load_pd(c + e*16));
        __m512d zi = _mm512_add_pd(xi, _mm512_load_pd(c + e*16 + 8));
        map2(zr, zi, &xr, &xi);
        _mm512_store_pd(x + e*16, xr);
        _mm512_store_pd(x + e*16 + 8, xi);
    }
}

// 8x8 double transpose: in r0..r7 -> out o0..o7
#define TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7) do{ \
    __m512d _t0=_mm512_unpacklo_pd(r0,r1), _t1=_mm512_unpackhi_pd(r0,r1); \
    __m512d _t2=_mm512_unpacklo_pd(r2,r3), _t3=_mm512_unpackhi_pd(r2,r3); \
    __m512d _t4=_mm512_unpacklo_pd(r4,r5), _t5=_mm512_unpackhi_pd(r4,r5); \
    __m512d _t6=_mm512_unpacklo_pd(r6,r7), _t7=_mm512_unpackhi_pd(r6,r7); \
    __m512d _u0=_mm512_shuffle_f64x2(_t0,_t2,0x44), _u1=_mm512_shuffle_f64x2(_t4,_t6,0x44); \
    __m512d _u2=_mm512_shuffle_f64x2(_t0,_t2,0xee), _u3=_mm512_shuffle_f64x2(_t4,_t6,0xee); \
    __m512d _u4=_mm512_shuffle_f64x2(_t1,_t3,0x44), _u5=_mm512_shuffle_f64x2(_t5,_t7,0x44); \
    __m512d _u6=_mm512_shuffle_f64x2(_t1,_t3,0xee), _u7=_mm512_shuffle_f64x2(_t5,_t7,0xee); \
    o0=_mm512_shuffle_f64x2(_u0,_u1,0x88); o2=_mm512_shuffle_f64x2(_u0,_u1,0xdd); \
    o4=_mm512_shuffle_f64x2(_u2,_u3,0x88); o6=_mm512_shuffle_f64x2(_u2,_u3,0xdd); \
    o1=_mm512_shuffle_f64x2(_u4,_u5,0x88); o3=_mm512_shuffle_f64x2(_u4,_u5,0xdd); \
    o5=_mm512_shuffle_f64x2(_u6,_u7,0x88); o7=_mm512_shuffle_f64x2(_u6,_u7,0xdd); \
}while(0)

static const __m512i IDX_EVEN_ = {0,2,4,6,8,10,12,14};
static const __m512i IDX_ODD_  = {1,3,5,7,9,11,13,15};
#define DEINT(lo,hi,re,im) do{ re=_mm512_permutex2var_pd(lo, IDX_EVEN_, hi); im=_mm512_permutex2var_pd(lo, IDX_ODD_, hi);}while(0)
static const __m512i IDX_ILO_ = {0,8,1,9,2,10,3,11};
static const __m512i IDX_IHI_ = {4,12,5,13,6,14,7,15};
#define INTER(re,im,lo,hi) do{ lo=_mm512_permutex2var_pd(re, IDX_ILO_, im); hi=_mm512_permutex2var_pd(re, IDX_IHI_, im);}while(0)
'''
