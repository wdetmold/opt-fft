/*
 * Iterated batched 3D complex DFTs for L in {6,8,13,17,23,36,45,64}:
 *   z = FFT3(x) + c ; x <- z / (1 + |z|)     (full fp64 throughout)
 *
 * All transform arithmetic is hand-written here (no FFT libraries of any kind;
 * links only libc/libm). Architecture (evolved from the strongest prior attempt
 * on this problem, with rebuilt kernels and reworked memory behavior):
 *
 *  - sizes 6..23: lane-major SoA batching, 8 volumes per zmm lane group; every
 *    1-D DFT is pure vertical SIMD (no shuffles). PFA 2x3 codelet for 6,
 *    radix-2/4 for 8. Primes 13/17/23 use symmetric (half-length folded) DFTs
 *    rebuilt as register-resident phase-split sweeps: 4 component sweeps
 *    (cos/e, sin/o x re/im), all (p-1)/2 cosine and sine constants held in zmm
 *    registers, j-outer accumulation (~4h^2 reg-reg FMAs per pencil, zero
 *    in-sweep constant loads), partials parked in L1 scratch between sweeps.
 *  - sizes 36/45/64: within-volume split re/im planes, padded row strides
 *    (40/56/72 doubles) against 4K aliasing. PFA 4x9 / 5x9 codelets and
 *    radix-8x8 with exact mod-64 twiddles, staged two-phase through L1 scratch;
 *    8x8 in-register transposes for the contiguous axis, strip-wise plane
 *    processing; the transposed c copy is stored split re/im in
 *    strip-consumption order so the fused map is maskless and sequential;
 *    the pencil pass consumes c from a column-consumption-ordered copy.
 *    Interleaved T1 next-plane prefetch for L=64 (L3-resident volume) and
 *    distance-2 strip prefetch of c inside the map loop.
 *  - one full-data pass per step in steady state: palindromic axis orders let
 *    the slab pass (two in-plane axes) and the pencil pass (plane axis) each
 *    finish step t and immediately pre-transform step t+1 around the map.
 *  - elementwise map z/(1+|z|): rsqrt14 + Newton/Heron for |z| and rcp14 +
 *    2nd/4th-order Newton for the reciprocal (~1 ulp, all fp64); 1e-30 floor
 *    keeps pad lanes out of denormal territory.
 *  - all trig/twiddle constants precomputed in extended precision, baked as hex
 *    literals. Huge-page backed, stagger-offset arenas. Batch remainders below
 *    per-size thresholds run on per-volume (within-volume) fallback paths.
 *
 * Single-threaded, AVX-512 only, deterministic.
 */
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

static const double CC_13[12] ALIGN64 = {0x1.c55a7e00740e9p-1,0x1.22d961ea71119p-1,0x1.edb7debaa3ed8p-4,-0x1.6b1d8b2365da1p-2,-0x1.7f3ccd0032e0cp-1,-0x1.f11f493053d00p-1,0x1.dbe064267c47cp-2,0x1.a55e242a4c3d2p-1,0x1.fc44566966769p-1,0x1.deba72ef20147p-1,0x1.5384d024c2f84p-1,0x1.ea1e54bc48dbfp-3};
static void dft13_v(double* re, double* im, long es){
  double scr[152] __attribute__((aligned(64)));
  {
  const __m512d C0 = _mm512_set1_pd(CC_13[0]);
  const __m512d C1 = _mm512_set1_pd(CC_13[1]);
  const __m512d C2 = _mm512_set1_pd(CC_13[2]);
  const __m512d C3 = _mm512_set1_pd(CC_13[3]);
  const __m512d C4 = _mm512_set1_pd(CC_13[4]);
  const __m512d C5 = _mm512_set1_pd(CC_13[5]);
  __m512d x0r = _mm512_load_pd(re);
  __m512d s0r = x0r;
  __m512d ar1 = x0r;
  __m512d ar2 = x0r;
  __m512d ar3 = x0r;
  __m512d ar4 = x0r;
  __m512d ar5 = x0r;
  __m512d ar6 = x0r;
  { __m512d p = _mm512_load_pd(re + 1*es), q = _mm512_load_pd(re + 12*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 0, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C0, e, ar1);
    ar2 = _mm512_fmadd_pd(C1, e, ar2);
    ar3 = _mm512_fmadd_pd(C2, e, ar3);
    ar4 = _mm512_fmadd_pd(C3, e, ar4);
    ar5 = _mm512_fmadd_pd(C4, e, ar5);
    ar6 = _mm512_fmadd_pd(C5, e, ar6);
  }
  { __m512d p = _mm512_load_pd(re + 2*es), q = _mm512_load_pd(re + 11*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 8, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C1, e, ar1);
    ar2 = _mm512_fmadd_pd(C3, e, ar2);
    ar3 = _mm512_fmadd_pd(C5, e, ar3);
    ar4 = _mm512_fmadd_pd(C4, e, ar4);
    ar5 = _mm512_fmadd_pd(C2, e, ar5);
    ar6 = _mm512_fmadd_pd(C0, e, ar6);
  }
  { __m512d p = _mm512_load_pd(re + 3*es), q = _mm512_load_pd(re + 10*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 16, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C2, e, ar1);
    ar2 = _mm512_fmadd_pd(C5, e, ar2);
    ar3 = _mm512_fmadd_pd(C3, e, ar3);
    ar4 = _mm512_fmadd_pd(C0, e, ar4);
    ar5 = _mm512_fmadd_pd(C1, e, ar5);
    ar6 = _mm512_fmadd_pd(C4, e, ar6);
  }
  { __m512d p = _mm512_load_pd(re + 4*es), q = _mm512_load_pd(re + 9*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 24, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C3, e, ar1);
    ar2 = _mm512_fmadd_pd(C4, e, ar2);
    ar3 = _mm512_fmadd_pd(C0, e, ar3);
    ar4 = _mm512_fmadd_pd(C2, e, ar4);
    ar5 = _mm512_fmadd_pd(C5, e, ar5);
    ar6 = _mm512_fmadd_pd(C1, e, ar6);
  }
  { __m512d p = _mm512_load_pd(re + 5*es), q = _mm512_load_pd(re + 8*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 32, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C4, e, ar1);
    ar2 = _mm512_fmadd_pd(C2, e, ar2);
    ar3 = _mm512_fmadd_pd(C1, e, ar3);
    ar4 = _mm512_fmadd_pd(C5, e, ar4);
    ar5 = _mm512_fmadd_pd(C0, e, ar5);
    ar6 = _mm512_fmadd_pd(C3, e, ar6);
  }
  { __m512d p = _mm512_load_pd(re + 6*es), q = _mm512_load_pd(re + 7*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 40, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C5, e, ar1);
    ar2 = _mm512_fmadd_pd(C0, e, ar2);
    ar3 = _mm512_fmadd_pd(C4, e, ar3);
    ar4 = _mm512_fmadd_pd(C1, e, ar4);
    ar5 = _mm512_fmadd_pd(C3, e, ar5);
    ar6 = _mm512_fmadd_pd(C2, e, ar6);
  }
  _mm512_store_pd(re, s0r);
  _mm512_store_pd(scr + 104, ar1);
  _mm512_store_pd(scr + 112, ar2);
  _mm512_store_pd(scr + 120, ar3);
  _mm512_store_pd(scr + 128, ar4);
  _mm512_store_pd(scr + 136, ar5);
  _mm512_store_pd(scr + 144, ar6);
  }
  {
  const __m512d S0 = _mm512_set1_pd(CC_13[6]);
  const __m512d S1 = _mm512_set1_pd(CC_13[7]);
  const __m512d S2 = _mm512_set1_pd(CC_13[8]);
  const __m512d S3 = _mm512_set1_pd(CC_13[9]);
  const __m512d S4 = _mm512_set1_pd(CC_13[10]);
  const __m512d S5 = _mm512_set1_pd(CC_13[11]);
  __m512d x0i = _mm512_load_pd(im);
  _mm512_store_pd(scr + 96, x0i);
  __m512d s0i = x0i;
  __m512d bi1 = _mm512_setzero_pd();
  __m512d bi2 = _mm512_setzero_pd();
  __m512d bi3 = _mm512_setzero_pd();
  __m512d bi4 = _mm512_setzero_pd();
  __m512d bi5 = _mm512_setzero_pd();
  __m512d bi6 = _mm512_setzero_pd();
  { __m512d p = _mm512_load_pd(im + 1*es), q = _mm512_load_pd(im + 12*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 48, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S0, o, bi1);
    bi2 = _mm512_fmadd_pd(S1, o, bi2);
    bi3 = _mm512_fmadd_pd(S2, o, bi3);
    bi4 = _mm512_fmadd_pd(S3, o, bi4);
    bi5 = _mm512_fmadd_pd(S4, o, bi5);
    bi6 = _mm512_fmadd_pd(S5, o, bi6);
  }
  { __m512d p = _mm512_load_pd(im + 2*es), q = _mm512_load_pd(im + 11*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 56, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S1, o, bi1);
    bi2 = _mm512_fmadd_pd(S3, o, bi2);
    bi3 = _mm512_fmadd_pd(S5, o, bi3);
    bi4 = _mm512_fnmadd_pd(S4, o, bi4);
    bi5 = _mm512_fnmadd_pd(S2, o, bi5);
    bi6 = _mm512_fnmadd_pd(S0, o, bi6);
  }
  { __m512d p = _mm512_load_pd(im + 3*es), q = _mm512_load_pd(im + 10*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 64, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S2, o, bi1);
    bi2 = _mm512_fmadd_pd(S5, o, bi2);
    bi3 = _mm512_fnmadd_pd(S3, o, bi3);
    bi4 = _mm512_fnmadd_pd(S0, o, bi4);
    bi5 = _mm512_fmadd_pd(S1, o, bi5);
    bi6 = _mm512_fmadd_pd(S4, o, bi6);
  }
  { __m512d p = _mm512_load_pd(im + 4*es), q = _mm512_load_pd(im + 9*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 72, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S3, o, bi1);
    bi2 = _mm512_fnmadd_pd(S4, o, bi2);
    bi3 = _mm512_fnmadd_pd(S0, o, bi3);
    bi4 = _mm512_fmadd_pd(S2, o, bi4);
    bi5 = _mm512_fnmadd_pd(S5, o, bi5);
    bi6 = _mm512_fnmadd_pd(S1, o, bi6);
  }
  { __m512d p = _mm512_load_pd(im + 5*es), q = _mm512_load_pd(im + 8*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 80, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S4, o, bi1);
    bi2 = _mm512_fnmadd_pd(S2, o, bi2);
    bi3 = _mm512_fmadd_pd(S1, o, bi3);
    bi4 = _mm512_fnmadd_pd(S5, o, bi4);
    bi5 = _mm512_fnmadd_pd(S0, o, bi5);
    bi6 = _mm512_fmadd_pd(S3, o, bi6);
  }
  { __m512d p = _mm512_load_pd(im + 6*es), q = _mm512_load_pd(im + 7*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 88, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S5, o, bi1);
    bi2 = _mm512_fnmadd_pd(S0, o, bi2);
    bi3 = _mm512_fmadd_pd(S4, o, bi3);
    bi4 = _mm512_fnmadd_pd(S1, o, bi4);
    bi5 = _mm512_fmadd_pd(S3, o, bi5);
    bi6 = _mm512_fnmadd_pd(S2, o, bi6);
  }
  _mm512_store_pd(im, s0i);
  { __m512d a = _mm512_load_pd(scr + 104);
    _mm512_store_pd(re + 1*es, _mm512_add_pd(a, bi1));
    _mm512_store_pd(re + 12*es, _mm512_sub_pd(a, bi1)); }
  { __m512d a = _mm512_load_pd(scr + 112);
    _mm512_store_pd(re + 2*es, _mm512_add_pd(a, bi2));
    _mm512_store_pd(re + 11*es, _mm512_sub_pd(a, bi2)); }
  { __m512d a = _mm512_load_pd(scr + 120);
    _mm512_store_pd(re + 3*es, _mm512_add_pd(a, bi3));
    _mm512_store_pd(re + 10*es, _mm512_sub_pd(a, bi3)); }
  { __m512d a = _mm512_load_pd(scr + 128);
    _mm512_store_pd(re + 4*es, _mm512_add_pd(a, bi4));
    _mm512_store_pd(re + 9*es, _mm512_sub_pd(a, bi4)); }
  { __m512d a = _mm512_load_pd(scr + 136);
    _mm512_store_pd(re + 5*es, _mm512_add_pd(a, bi5));
    _mm512_store_pd(re + 8*es, _mm512_sub_pd(a, bi5)); }
  { __m512d a = _mm512_load_pd(scr + 144);
    _mm512_store_pd(re + 6*es, _mm512_add_pd(a, bi6));
    _mm512_store_pd(re + 7*es, _mm512_sub_pd(a, bi6)); }
  }
  {
  const __m512d C0 = _mm512_set1_pd(CC_13[0]);
  const __m512d C1 = _mm512_set1_pd(CC_13[1]);
  const __m512d C2 = _mm512_set1_pd(CC_13[2]);
  const __m512d C3 = _mm512_set1_pd(CC_13[3]);
  const __m512d C4 = _mm512_set1_pd(CC_13[4]);
  const __m512d C5 = _mm512_set1_pd(CC_13[5]);
  __m512d x0i = _mm512_load_pd(scr + 96);
  __m512d ai1 = x0i;
  __m512d ai2 = x0i;
  __m512d ai3 = x0i;
  __m512d ai4 = x0i;
  __m512d ai5 = x0i;
  __m512d ai6 = x0i;
  { __m512d e = _mm512_load_pd(scr + 48);
    ai1 = _mm512_fmadd_pd(C0, e, ai1);
    ai2 = _mm512_fmadd_pd(C1, e, ai2);
    ai3 = _mm512_fmadd_pd(C2, e, ai3);
    ai4 = _mm512_fmadd_pd(C3, e, ai4);
    ai5 = _mm512_fmadd_pd(C4, e, ai5);
    ai6 = _mm512_fmadd_pd(C5, e, ai6);
  }
  { __m512d e = _mm512_load_pd(scr + 56);
    ai1 = _mm512_fmadd_pd(C1, e, ai1);
    ai2 = _mm512_fmadd_pd(C3, e, ai2);
    ai3 = _mm512_fmadd_pd(C5, e, ai3);
    ai4 = _mm512_fmadd_pd(C4, e, ai4);
    ai5 = _mm512_fmadd_pd(C2, e, ai5);
    ai6 = _mm512_fmadd_pd(C0, e, ai6);
  }
  { __m512d e = _mm512_load_pd(scr + 64);
    ai1 = _mm512_fmadd_pd(C2, e, ai1);
    ai2 = _mm512_fmadd_pd(C5, e, ai2);
    ai3 = _mm512_fmadd_pd(C3, e, ai3);
    ai4 = _mm512_fmadd_pd(C0, e, ai4);
    ai5 = _mm512_fmadd_pd(C1, e, ai5);
    ai6 = _mm512_fmadd_pd(C4, e, ai6);
  }
  { __m512d e = _mm512_load_pd(scr + 72);
    ai1 = _mm512_fmadd_pd(C3, e, ai1);
    ai2 = _mm512_fmadd_pd(C4, e, ai2);
    ai3 = _mm512_fmadd_pd(C0, e, ai3);
    ai4 = _mm512_fmadd_pd(C2, e, ai4);
    ai5 = _mm512_fmadd_pd(C5, e, ai5);
    ai6 = _mm512_fmadd_pd(C1, e, ai6);
  }
  { __m512d e = _mm512_load_pd(scr + 80);
    ai1 = _mm512_fmadd_pd(C4, e, ai1);
    ai2 = _mm512_fmadd_pd(C2, e, ai2);
    ai3 = _mm512_fmadd_pd(C1, e, ai3);
    ai4 = _mm512_fmadd_pd(C5, e, ai4);
    ai5 = _mm512_fmadd_pd(C0, e, ai5);
    ai6 = _mm512_fmadd_pd(C3, e, ai6);
  }
  { __m512d e = _mm512_load_pd(scr + 88);
    ai1 = _mm512_fmadd_pd(C5, e, ai1);
    ai2 = _mm512_fmadd_pd(C0, e, ai2);
    ai3 = _mm512_fmadd_pd(C4, e, ai3);
    ai4 = _mm512_fmadd_pd(C1, e, ai4);
    ai5 = _mm512_fmadd_pd(C3, e, ai5);
    ai6 = _mm512_fmadd_pd(C2, e, ai6);
  }
  _mm512_store_pd(scr + 104, ai1);
  _mm512_store_pd(scr + 112, ai2);
  _mm512_store_pd(scr + 120, ai3);
  _mm512_store_pd(scr + 128, ai4);
  _mm512_store_pd(scr + 136, ai5);
  _mm512_store_pd(scr + 144, ai6);
  }
  {
  const __m512d S0 = _mm512_set1_pd(CC_13[6]);
  const __m512d S1 = _mm512_set1_pd(CC_13[7]);
  const __m512d S2 = _mm512_set1_pd(CC_13[8]);
  const __m512d S3 = _mm512_set1_pd(CC_13[9]);
  const __m512d S4 = _mm512_set1_pd(CC_13[10]);
  const __m512d S5 = _mm512_set1_pd(CC_13[11]);
  __m512d br1 = _mm512_setzero_pd();
  __m512d br2 = _mm512_setzero_pd();
  __m512d br3 = _mm512_setzero_pd();
  __m512d br4 = _mm512_setzero_pd();
  __m512d br5 = _mm512_setzero_pd();
  __m512d br6 = _mm512_setzero_pd();
  { __m512d o = _mm512_load_pd(scr + 0);
    br1 = _mm512_fmadd_pd(S0, o, br1);
    br2 = _mm512_fmadd_pd(S1, o, br2);
    br3 = _mm512_fmadd_pd(S2, o, br3);
    br4 = _mm512_fmadd_pd(S3, o, br4);
    br5 = _mm512_fmadd_pd(S4, o, br5);
    br6 = _mm512_fmadd_pd(S5, o, br6);
  }
  { __m512d o = _mm512_load_pd(scr + 8);
    br1 = _mm512_fmadd_pd(S1, o, br1);
    br2 = _mm512_fmadd_pd(S3, o, br2);
    br3 = _mm512_fmadd_pd(S5, o, br3);
    br4 = _mm512_fnmadd_pd(S4, o, br4);
    br5 = _mm512_fnmadd_pd(S2, o, br5);
    br6 = _mm512_fnmadd_pd(S0, o, br6);
  }
  { __m512d o = _mm512_load_pd(scr + 16);
    br1 = _mm512_fmadd_pd(S2, o, br1);
    br2 = _mm512_fmadd_pd(S5, o, br2);
    br3 = _mm512_fnmadd_pd(S3, o, br3);
    br4 = _mm512_fnmadd_pd(S0, o, br4);
    br5 = _mm512_fmadd_pd(S1, o, br5);
    br6 = _mm512_fmadd_pd(S4, o, br6);
  }
  { __m512d o = _mm512_load_pd(scr + 24);
    br1 = _mm512_fmadd_pd(S3, o, br1);
    br2 = _mm512_fnmadd_pd(S4, o, br2);
    br3 = _mm512_fnmadd_pd(S0, o, br3);
    br4 = _mm512_fmadd_pd(S2, o, br4);
    br5 = _mm512_fnmadd_pd(S5, o, br5);
    br6 = _mm512_fnmadd_pd(S1, o, br6);
  }
  { __m512d o = _mm512_load_pd(scr + 32);
    br1 = _mm512_fmadd_pd(S4, o, br1);
    br2 = _mm512_fnmadd_pd(S2, o, br2);
    br3 = _mm512_fmadd_pd(S1, o, br3);
    br4 = _mm512_fnmadd_pd(S5, o, br4);
    br5 = _mm512_fnmadd_pd(S0, o, br5);
    br6 = _mm512_fmadd_pd(S3, o, br6);
  }
  { __m512d o = _mm512_load_pd(scr + 40);
    br1 = _mm512_fmadd_pd(S5, o, br1);
    br2 = _mm512_fnmadd_pd(S0, o, br2);
    br3 = _mm512_fmadd_pd(S4, o, br3);
    br4 = _mm512_fnmadd_pd(S1, o, br4);
    br5 = _mm512_fmadd_pd(S3, o, br5);
    br6 = _mm512_fnmadd_pd(S2, o, br6);
  }
  { __m512d a = _mm512_load_pd(scr + 104);
    _mm512_store_pd(im + 1*es, _mm512_sub_pd(a, br1));
    _mm512_store_pd(im + 12*es, _mm512_add_pd(a, br1)); }
  { __m512d a = _mm512_load_pd(scr + 112);
    _mm512_store_pd(im + 2*es, _mm512_sub_pd(a, br2));
    _mm512_store_pd(im + 11*es, _mm512_add_pd(a, br2)); }
  { __m512d a = _mm512_load_pd(scr + 120);
    _mm512_store_pd(im + 3*es, _mm512_sub_pd(a, br3));
    _mm512_store_pd(im + 10*es, _mm512_add_pd(a, br3)); }
  { __m512d a = _mm512_load_pd(scr + 128);
    _mm512_store_pd(im + 4*es, _mm512_sub_pd(a, br4));
    _mm512_store_pd(im + 9*es, _mm512_add_pd(a, br4)); }
  { __m512d a = _mm512_load_pd(scr + 136);
    _mm512_store_pd(im + 5*es, _mm512_sub_pd(a, br5));
    _mm512_store_pd(im + 8*es, _mm512_add_pd(a, br5)); }
  { __m512d a = _mm512_load_pd(scr + 144);
    _mm512_store_pd(im + 6*es, _mm512_sub_pd(a, br6));
    _mm512_store_pd(im + 7*es, _mm512_add_pd(a, br6)); }
  }
}
static const double CC_17[16] ALIGN64 = {0x1.dd6d000370991p-1,0x1.7a5f6075d4884p-1,0x1.c86fa2b2883cdp-2,0x1.79ee63259b75ep-4,-0x1.183b1c61f0d01p-2,-0x1.348c86ed5f1bbp-1,-0x1.b34fa910ea3b9p-1,-0x1.f7484007faef3p-1,0x1.71e955d8e7cdcp-2,0x1.58eea2a9d6da3p-1,0x1.ca52d7c9e640bp-1,0x1.fdd0deb564b22p-1,0x1.ec746923c349fp-1,0x1.9895b6c9a05f6p-1,0x1.0d8884363dd80p-1,0x1.7851aacd6c6b4p-3};
static void dft17_v(double* re, double* im, long es){
  double scr[200] __attribute__((aligned(64)));
  {
  const __m512d C0 = _mm512_set1_pd(CC_17[0]);
  const __m512d C1 = _mm512_set1_pd(CC_17[1]);
  const __m512d C2 = _mm512_set1_pd(CC_17[2]);
  const __m512d C3 = _mm512_set1_pd(CC_17[3]);
  const __m512d C4 = _mm512_set1_pd(CC_17[4]);
  const __m512d C5 = _mm512_set1_pd(CC_17[5]);
  const __m512d C6 = _mm512_set1_pd(CC_17[6]);
  const __m512d C7 = _mm512_set1_pd(CC_17[7]);
  __m512d x0r = _mm512_load_pd(re);
  __m512d s0r = x0r;
  __m512d ar1 = x0r;
  __m512d ar2 = x0r;
  __m512d ar3 = x0r;
  __m512d ar4 = x0r;
  __m512d ar5 = x0r;
  __m512d ar6 = x0r;
  __m512d ar7 = x0r;
  __m512d ar8 = x0r;
  { __m512d p = _mm512_load_pd(re + 1*es), q = _mm512_load_pd(re + 16*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 0, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C0, e, ar1);
    ar2 = _mm512_fmadd_pd(C1, e, ar2);
    ar3 = _mm512_fmadd_pd(C2, e, ar3);
    ar4 = _mm512_fmadd_pd(C3, e, ar4);
    ar5 = _mm512_fmadd_pd(C4, e, ar5);
    ar6 = _mm512_fmadd_pd(C5, e, ar6);
    ar7 = _mm512_fmadd_pd(C6, e, ar7);
    ar8 = _mm512_fmadd_pd(C7, e, ar8);
  }
  { __m512d p = _mm512_load_pd(re + 2*es), q = _mm512_load_pd(re + 15*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 8, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C1, e, ar1);
    ar2 = _mm512_fmadd_pd(C3, e, ar2);
    ar3 = _mm512_fmadd_pd(C5, e, ar3);
    ar4 = _mm512_fmadd_pd(C7, e, ar4);
    ar5 = _mm512_fmadd_pd(C6, e, ar5);
    ar6 = _mm512_fmadd_pd(C4, e, ar6);
    ar7 = _mm512_fmadd_pd(C2, e, ar7);
    ar8 = _mm512_fmadd_pd(C0, e, ar8);
  }
  { __m512d p = _mm512_load_pd(re + 3*es), q = _mm512_load_pd(re + 14*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 16, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C2, e, ar1);
    ar2 = _mm512_fmadd_pd(C5, e, ar2);
    ar3 = _mm512_fmadd_pd(C7, e, ar3);
    ar4 = _mm512_fmadd_pd(C4, e, ar4);
    ar5 = _mm512_fmadd_pd(C1, e, ar5);
    ar6 = _mm512_fmadd_pd(C0, e, ar6);
    ar7 = _mm512_fmadd_pd(C3, e, ar7);
    ar8 = _mm512_fmadd_pd(C6, e, ar8);
  }
  { __m512d p = _mm512_load_pd(re + 4*es), q = _mm512_load_pd(re + 13*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 24, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C3, e, ar1);
    ar2 = _mm512_fmadd_pd(C7, e, ar2);
    ar3 = _mm512_fmadd_pd(C4, e, ar3);
    ar4 = _mm512_fmadd_pd(C0, e, ar4);
    ar5 = _mm512_fmadd_pd(C2, e, ar5);
    ar6 = _mm512_fmadd_pd(C6, e, ar6);
    ar7 = _mm512_fmadd_pd(C5, e, ar7);
    ar8 = _mm512_fmadd_pd(C1, e, ar8);
  }
  { __m512d p = _mm512_load_pd(re + 5*es), q = _mm512_load_pd(re + 12*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 32, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C4, e, ar1);
    ar2 = _mm512_fmadd_pd(C6, e, ar2);
    ar3 = _mm512_fmadd_pd(C1, e, ar3);
    ar4 = _mm512_fmadd_pd(C2, e, ar4);
    ar5 = _mm512_fmadd_pd(C7, e, ar5);
    ar6 = _mm512_fmadd_pd(C3, e, ar6);
    ar7 = _mm512_fmadd_pd(C0, e, ar7);
    ar8 = _mm512_fmadd_pd(C5, e, ar8);
  }
  { __m512d p = _mm512_load_pd(re + 6*es), q = _mm512_load_pd(re + 11*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 40, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C5, e, ar1);
    ar2 = _mm512_fmadd_pd(C4, e, ar2);
    ar3 = _mm512_fmadd_pd(C0, e, ar3);
    ar4 = _mm512_fmadd_pd(C6, e, ar4);
    ar5 = _mm512_fmadd_pd(C3, e, ar5);
    ar6 = _mm512_fmadd_pd(C1, e, ar6);
    ar7 = _mm512_fmadd_pd(C7, e, ar7);
    ar8 = _mm512_fmadd_pd(C2, e, ar8);
  }
  { __m512d p = _mm512_load_pd(re + 7*es), q = _mm512_load_pd(re + 10*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 48, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C6, e, ar1);
    ar2 = _mm512_fmadd_pd(C2, e, ar2);
    ar3 = _mm512_fmadd_pd(C3, e, ar3);
    ar4 = _mm512_fmadd_pd(C5, e, ar4);
    ar5 = _mm512_fmadd_pd(C0, e, ar5);
    ar6 = _mm512_fmadd_pd(C7, e, ar6);
    ar7 = _mm512_fmadd_pd(C1, e, ar7);
    ar8 = _mm512_fmadd_pd(C4, e, ar8);
  }
  { __m512d p = _mm512_load_pd(re + 8*es), q = _mm512_load_pd(re + 9*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 56, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C7, e, ar1);
    ar2 = _mm512_fmadd_pd(C0, e, ar2);
    ar3 = _mm512_fmadd_pd(C6, e, ar3);
    ar4 = _mm512_fmadd_pd(C1, e, ar4);
    ar5 = _mm512_fmadd_pd(C5, e, ar5);
    ar6 = _mm512_fmadd_pd(C2, e, ar6);
    ar7 = _mm512_fmadd_pd(C4, e, ar7);
    ar8 = _mm512_fmadd_pd(C3, e, ar8);
  }
  _mm512_store_pd(re, s0r);
  _mm512_store_pd(scr + 136, ar1);
  _mm512_store_pd(scr + 144, ar2);
  _mm512_store_pd(scr + 152, ar3);
  _mm512_store_pd(scr + 160, ar4);
  _mm512_store_pd(scr + 168, ar5);
  _mm512_store_pd(scr + 176, ar6);
  _mm512_store_pd(scr + 184, ar7);
  _mm512_store_pd(scr + 192, ar8);
  }
  {
  const __m512d S0 = _mm512_set1_pd(CC_17[8]);
  const __m512d S1 = _mm512_set1_pd(CC_17[9]);
  const __m512d S2 = _mm512_set1_pd(CC_17[10]);
  const __m512d S3 = _mm512_set1_pd(CC_17[11]);
  const __m512d S4 = _mm512_set1_pd(CC_17[12]);
  const __m512d S5 = _mm512_set1_pd(CC_17[13]);
  const __m512d S6 = _mm512_set1_pd(CC_17[14]);
  const __m512d S7 = _mm512_set1_pd(CC_17[15]);
  __m512d x0i = _mm512_load_pd(im);
  _mm512_store_pd(scr + 128, x0i);
  __m512d s0i = x0i;
  __m512d bi1 = _mm512_setzero_pd();
  __m512d bi2 = _mm512_setzero_pd();
  __m512d bi3 = _mm512_setzero_pd();
  __m512d bi4 = _mm512_setzero_pd();
  __m512d bi5 = _mm512_setzero_pd();
  __m512d bi6 = _mm512_setzero_pd();
  __m512d bi7 = _mm512_setzero_pd();
  __m512d bi8 = _mm512_setzero_pd();
  { __m512d p = _mm512_load_pd(im + 1*es), q = _mm512_load_pd(im + 16*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 64, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S0, o, bi1);
    bi2 = _mm512_fmadd_pd(S1, o, bi2);
    bi3 = _mm512_fmadd_pd(S2, o, bi3);
    bi4 = _mm512_fmadd_pd(S3, o, bi4);
    bi5 = _mm512_fmadd_pd(S4, o, bi5);
    bi6 = _mm512_fmadd_pd(S5, o, bi6);
    bi7 = _mm512_fmadd_pd(S6, o, bi7);
    bi8 = _mm512_fmadd_pd(S7, o, bi8);
  }
  { __m512d p = _mm512_load_pd(im + 2*es), q = _mm512_load_pd(im + 15*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 72, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S1, o, bi1);
    bi2 = _mm512_fmadd_pd(S3, o, bi2);
    bi3 = _mm512_fmadd_pd(S5, o, bi3);
    bi4 = _mm512_fmadd_pd(S7, o, bi4);
    bi5 = _mm512_fnmadd_pd(S6, o, bi5);
    bi6 = _mm512_fnmadd_pd(S4, o, bi6);
    bi7 = _mm512_fnmadd_pd(S2, o, bi7);
    bi8 = _mm512_fnmadd_pd(S0, o, bi8);
  }
  { __m512d p = _mm512_load_pd(im + 3*es), q = _mm512_load_pd(im + 14*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 80, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S2, o, bi1);
    bi2 = _mm512_fmadd_pd(S5, o, bi2);
    bi3 = _mm512_fnmadd_pd(S7, o, bi3);
    bi4 = _mm512_fnmadd_pd(S4, o, bi4);
    bi5 = _mm512_fnmadd_pd(S1, o, bi5);
    bi6 = _mm512_fmadd_pd(S0, o, bi6);
    bi7 = _mm512_fmadd_pd(S3, o, bi7);
    bi8 = _mm512_fmadd_pd(S6, o, bi8);
  }
  { __m512d p = _mm512_load_pd(im + 4*es), q = _mm512_load_pd(im + 13*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 88, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S3, o, bi1);
    bi2 = _mm512_fmadd_pd(S7, o, bi2);
    bi3 = _mm512_fnmadd_pd(S4, o, bi3);
    bi4 = _mm512_fnmadd_pd(S0, o, bi4);
    bi5 = _mm512_fmadd_pd(S2, o, bi5);
    bi6 = _mm512_fmadd_pd(S6, o, bi6);
    bi7 = _mm512_fnmadd_pd(S5, o, bi7);
    bi8 = _mm512_fnmadd_pd(S1, o, bi8);
  }
  { __m512d p = _mm512_load_pd(im + 5*es), q = _mm512_load_pd(im + 12*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 96, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S4, o, bi1);
    bi2 = _mm512_fnmadd_pd(S6, o, bi2);
    bi3 = _mm512_fnmadd_pd(S1, o, bi3);
    bi4 = _mm512_fmadd_pd(S2, o, bi4);
    bi5 = _mm512_fmadd_pd(S7, o, bi5);
    bi6 = _mm512_fnmadd_pd(S3, o, bi6);
    bi7 = _mm512_fmadd_pd(S0, o, bi7);
    bi8 = _mm512_fmadd_pd(S5, o, bi8);
  }
  { __m512d p = _mm512_load_pd(im + 6*es), q = _mm512_load_pd(im + 11*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 104, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S5, o, bi1);
    bi2 = _mm512_fnmadd_pd(S4, o, bi2);
    bi3 = _mm512_fmadd_pd(S0, o, bi3);
    bi4 = _mm512_fmadd_pd(S6, o, bi4);
    bi5 = _mm512_fnmadd_pd(S3, o, bi5);
    bi6 = _mm512_fmadd_pd(S1, o, bi6);
    bi7 = _mm512_fmadd_pd(S7, o, bi7);
    bi8 = _mm512_fnmadd_pd(S2, o, bi8);
  }
  { __m512d p = _mm512_load_pd(im + 7*es), q = _mm512_load_pd(im + 10*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 112, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S6, o, bi1);
    bi2 = _mm512_fnmadd_pd(S2, o, bi2);
    bi3 = _mm512_fmadd_pd(S3, o, bi3);
    bi4 = _mm512_fnmadd_pd(S5, o, bi4);
    bi5 = _mm512_fmadd_pd(S0, o, bi5);
    bi6 = _mm512_fmadd_pd(S7, o, bi6);
    bi7 = _mm512_fnmadd_pd(S1, o, bi7);
    bi8 = _mm512_fmadd_pd(S4, o, bi8);
  }
  { __m512d p = _mm512_load_pd(im + 8*es), q = _mm512_load_pd(im + 9*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 120, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S7, o, bi1);
    bi2 = _mm512_fnmadd_pd(S0, o, bi2);
    bi3 = _mm512_fmadd_pd(S6, o, bi3);
    bi4 = _mm512_fnmadd_pd(S1, o, bi4);
    bi5 = _mm512_fmadd_pd(S5, o, bi5);
    bi6 = _mm512_fnmadd_pd(S2, o, bi6);
    bi7 = _mm512_fmadd_pd(S4, o, bi7);
    bi8 = _mm512_fnmadd_pd(S3, o, bi8);
  }
  _mm512_store_pd(im, s0i);
  { __m512d a = _mm512_load_pd(scr + 136);
    _mm512_store_pd(re + 1*es, _mm512_add_pd(a, bi1));
    _mm512_store_pd(re + 16*es, _mm512_sub_pd(a, bi1)); }
  { __m512d a = _mm512_load_pd(scr + 144);
    _mm512_store_pd(re + 2*es, _mm512_add_pd(a, bi2));
    _mm512_store_pd(re + 15*es, _mm512_sub_pd(a, bi2)); }
  { __m512d a = _mm512_load_pd(scr + 152);
    _mm512_store_pd(re + 3*es, _mm512_add_pd(a, bi3));
    _mm512_store_pd(re + 14*es, _mm512_sub_pd(a, bi3)); }
  { __m512d a = _mm512_load_pd(scr + 160);
    _mm512_store_pd(re + 4*es, _mm512_add_pd(a, bi4));
    _mm512_store_pd(re + 13*es, _mm512_sub_pd(a, bi4)); }
  { __m512d a = _mm512_load_pd(scr + 168);
    _mm512_store_pd(re + 5*es, _mm512_add_pd(a, bi5));
    _mm512_store_pd(re + 12*es, _mm512_sub_pd(a, bi5)); }
  { __m512d a = _mm512_load_pd(scr + 176);
    _mm512_store_pd(re + 6*es, _mm512_add_pd(a, bi6));
    _mm512_store_pd(re + 11*es, _mm512_sub_pd(a, bi6)); }
  { __m512d a = _mm512_load_pd(scr + 184);
    _mm512_store_pd(re + 7*es, _mm512_add_pd(a, bi7));
    _mm512_store_pd(re + 10*es, _mm512_sub_pd(a, bi7)); }
  { __m512d a = _mm512_load_pd(scr + 192);
    _mm512_store_pd(re + 8*es, _mm512_add_pd(a, bi8));
    _mm512_store_pd(re + 9*es, _mm512_sub_pd(a, bi8)); }
  }
  {
  const __m512d C0 = _mm512_set1_pd(CC_17[0]);
  const __m512d C1 = _mm512_set1_pd(CC_17[1]);
  const __m512d C2 = _mm512_set1_pd(CC_17[2]);
  const __m512d C3 = _mm512_set1_pd(CC_17[3]);
  const __m512d C4 = _mm512_set1_pd(CC_17[4]);
  const __m512d C5 = _mm512_set1_pd(CC_17[5]);
  const __m512d C6 = _mm512_set1_pd(CC_17[6]);
  const __m512d C7 = _mm512_set1_pd(CC_17[7]);
  __m512d x0i = _mm512_load_pd(scr + 128);
  __m512d ai1 = x0i;
  __m512d ai2 = x0i;
  __m512d ai3 = x0i;
  __m512d ai4 = x0i;
  __m512d ai5 = x0i;
  __m512d ai6 = x0i;
  __m512d ai7 = x0i;
  __m512d ai8 = x0i;
  { __m512d e = _mm512_load_pd(scr + 64);
    ai1 = _mm512_fmadd_pd(C0, e, ai1);
    ai2 = _mm512_fmadd_pd(C1, e, ai2);
    ai3 = _mm512_fmadd_pd(C2, e, ai3);
    ai4 = _mm512_fmadd_pd(C3, e, ai4);
    ai5 = _mm512_fmadd_pd(C4, e, ai5);
    ai6 = _mm512_fmadd_pd(C5, e, ai6);
    ai7 = _mm512_fmadd_pd(C6, e, ai7);
    ai8 = _mm512_fmadd_pd(C7, e, ai8);
  }
  { __m512d e = _mm512_load_pd(scr + 72);
    ai1 = _mm512_fmadd_pd(C1, e, ai1);
    ai2 = _mm512_fmadd_pd(C3, e, ai2);
    ai3 = _mm512_fmadd_pd(C5, e, ai3);
    ai4 = _mm512_fmadd_pd(C7, e, ai4);
    ai5 = _mm512_fmadd_pd(C6, e, ai5);
    ai6 = _mm512_fmadd_pd(C4, e, ai6);
    ai7 = _mm512_fmadd_pd(C2, e, ai7);
    ai8 = _mm512_fmadd_pd(C0, e, ai8);
  }
  { __m512d e = _mm512_load_pd(scr + 80);
    ai1 = _mm512_fmadd_pd(C2, e, ai1);
    ai2 = _mm512_fmadd_pd(C5, e, ai2);
    ai3 = _mm512_fmadd_pd(C7, e, ai3);
    ai4 = _mm512_fmadd_pd(C4, e, ai4);
    ai5 = _mm512_fmadd_pd(C1, e, ai5);
    ai6 = _mm512_fmadd_pd(C0, e, ai6);
    ai7 = _mm512_fmadd_pd(C3, e, ai7);
    ai8 = _mm512_fmadd_pd(C6, e, ai8);
  }
  { __m512d e = _mm512_load_pd(scr + 88);
    ai1 = _mm512_fmadd_pd(C3, e, ai1);
    ai2 = _mm512_fmadd_pd(C7, e, ai2);
    ai3 = _mm512_fmadd_pd(C4, e, ai3);
    ai4 = _mm512_fmadd_pd(C0, e, ai4);
    ai5 = _mm512_fmadd_pd(C2, e, ai5);
    ai6 = _mm512_fmadd_pd(C6, e, ai6);
    ai7 = _mm512_fmadd_pd(C5, e, ai7);
    ai8 = _mm512_fmadd_pd(C1, e, ai8);
  }
  { __m512d e = _mm512_load_pd(scr + 96);
    ai1 = _mm512_fmadd_pd(C4, e, ai1);
    ai2 = _mm512_fmadd_pd(C6, e, ai2);
    ai3 = _mm512_fmadd_pd(C1, e, ai3);
    ai4 = _mm512_fmadd_pd(C2, e, ai4);
    ai5 = _mm512_fmadd_pd(C7, e, ai5);
    ai6 = _mm512_fmadd_pd(C3, e, ai6);
    ai7 = _mm512_fmadd_pd(C0, e, ai7);
    ai8 = _mm512_fmadd_pd(C5, e, ai8);
  }
  { __m512d e = _mm512_load_pd(scr + 104);
    ai1 = _mm512_fmadd_pd(C5, e, ai1);
    ai2 = _mm512_fmadd_pd(C4, e, ai2);
    ai3 = _mm512_fmadd_pd(C0, e, ai3);
    ai4 = _mm512_fmadd_pd(C6, e, ai4);
    ai5 = _mm512_fmadd_pd(C3, e, ai5);
    ai6 = _mm512_fmadd_pd(C1, e, ai6);
    ai7 = _mm512_fmadd_pd(C7, e, ai7);
    ai8 = _mm512_fmadd_pd(C2, e, ai8);
  }
  { __m512d e = _mm512_load_pd(scr + 112);
    ai1 = _mm512_fmadd_pd(C6, e, ai1);
    ai2 = _mm512_fmadd_pd(C2, e, ai2);
    ai3 = _mm512_fmadd_pd(C3, e, ai3);
    ai4 = _mm512_fmadd_pd(C5, e, ai4);
    ai5 = _mm512_fmadd_pd(C0, e, ai5);
    ai6 = _mm512_fmadd_pd(C7, e, ai6);
    ai7 = _mm512_fmadd_pd(C1, e, ai7);
    ai8 = _mm512_fmadd_pd(C4, e, ai8);
  }
  { __m512d e = _mm512_load_pd(scr + 120);
    ai1 = _mm512_fmadd_pd(C7, e, ai1);
    ai2 = _mm512_fmadd_pd(C0, e, ai2);
    ai3 = _mm512_fmadd_pd(C6, e, ai3);
    ai4 = _mm512_fmadd_pd(C1, e, ai4);
    ai5 = _mm512_fmadd_pd(C5, e, ai5);
    ai6 = _mm512_fmadd_pd(C2, e, ai6);
    ai7 = _mm512_fmadd_pd(C4, e, ai7);
    ai8 = _mm512_fmadd_pd(C3, e, ai8);
  }
  _mm512_store_pd(scr + 136, ai1);
  _mm512_store_pd(scr + 144, ai2);
  _mm512_store_pd(scr + 152, ai3);
  _mm512_store_pd(scr + 160, ai4);
  _mm512_store_pd(scr + 168, ai5);
  _mm512_store_pd(scr + 176, ai6);
  _mm512_store_pd(scr + 184, ai7);
  _mm512_store_pd(scr + 192, ai8);
  }
  {
  const __m512d S0 = _mm512_set1_pd(CC_17[8]);
  const __m512d S1 = _mm512_set1_pd(CC_17[9]);
  const __m512d S2 = _mm512_set1_pd(CC_17[10]);
  const __m512d S3 = _mm512_set1_pd(CC_17[11]);
  const __m512d S4 = _mm512_set1_pd(CC_17[12]);
  const __m512d S5 = _mm512_set1_pd(CC_17[13]);
  const __m512d S6 = _mm512_set1_pd(CC_17[14]);
  const __m512d S7 = _mm512_set1_pd(CC_17[15]);
  __m512d br1 = _mm512_setzero_pd();
  __m512d br2 = _mm512_setzero_pd();
  __m512d br3 = _mm512_setzero_pd();
  __m512d br4 = _mm512_setzero_pd();
  __m512d br5 = _mm512_setzero_pd();
  __m512d br6 = _mm512_setzero_pd();
  __m512d br7 = _mm512_setzero_pd();
  __m512d br8 = _mm512_setzero_pd();
  { __m512d o = _mm512_load_pd(scr + 0);
    br1 = _mm512_fmadd_pd(S0, o, br1);
    br2 = _mm512_fmadd_pd(S1, o, br2);
    br3 = _mm512_fmadd_pd(S2, o, br3);
    br4 = _mm512_fmadd_pd(S3, o, br4);
    br5 = _mm512_fmadd_pd(S4, o, br5);
    br6 = _mm512_fmadd_pd(S5, o, br6);
    br7 = _mm512_fmadd_pd(S6, o, br7);
    br8 = _mm512_fmadd_pd(S7, o, br8);
  }
  { __m512d o = _mm512_load_pd(scr + 8);
    br1 = _mm512_fmadd_pd(S1, o, br1);
    br2 = _mm512_fmadd_pd(S3, o, br2);
    br3 = _mm512_fmadd_pd(S5, o, br3);
    br4 = _mm512_fmadd_pd(S7, o, br4);
    br5 = _mm512_fnmadd_pd(S6, o, br5);
    br6 = _mm512_fnmadd_pd(S4, o, br6);
    br7 = _mm512_fnmadd_pd(S2, o, br7);
    br8 = _mm512_fnmadd_pd(S0, o, br8);
  }
  { __m512d o = _mm512_load_pd(scr + 16);
    br1 = _mm512_fmadd_pd(S2, o, br1);
    br2 = _mm512_fmadd_pd(S5, o, br2);
    br3 = _mm512_fnmadd_pd(S7, o, br3);
    br4 = _mm512_fnmadd_pd(S4, o, br4);
    br5 = _mm512_fnmadd_pd(S1, o, br5);
    br6 = _mm512_fmadd_pd(S0, o, br6);
    br7 = _mm512_fmadd_pd(S3, o, br7);
    br8 = _mm512_fmadd_pd(S6, o, br8);
  }
  { __m512d o = _mm512_load_pd(scr + 24);
    br1 = _mm512_fmadd_pd(S3, o, br1);
    br2 = _mm512_fmadd_pd(S7, o, br2);
    br3 = _mm512_fnmadd_pd(S4, o, br3);
    br4 = _mm512_fnmadd_pd(S0, o, br4);
    br5 = _mm512_fmadd_pd(S2, o, br5);
    br6 = _mm512_fmadd_pd(S6, o, br6);
    br7 = _mm512_fnmadd_pd(S5, o, br7);
    br8 = _mm512_fnmadd_pd(S1, o, br8);
  }
  { __m512d o = _mm512_load_pd(scr + 32);
    br1 = _mm512_fmadd_pd(S4, o, br1);
    br2 = _mm512_fnmadd_pd(S6, o, br2);
    br3 = _mm512_fnmadd_pd(S1, o, br3);
    br4 = _mm512_fmadd_pd(S2, o, br4);
    br5 = _mm512_fmadd_pd(S7, o, br5);
    br6 = _mm512_fnmadd_pd(S3, o, br6);
    br7 = _mm512_fmadd_pd(S0, o, br7);
    br8 = _mm512_fmadd_pd(S5, o, br8);
  }
  { __m512d o = _mm512_load_pd(scr + 40);
    br1 = _mm512_fmadd_pd(S5, o, br1);
    br2 = _mm512_fnmadd_pd(S4, o, br2);
    br3 = _mm512_fmadd_pd(S0, o, br3);
    br4 = _mm512_fmadd_pd(S6, o, br4);
    br5 = _mm512_fnmadd_pd(S3, o, br5);
    br6 = _mm512_fmadd_pd(S1, o, br6);
    br7 = _mm512_fmadd_pd(S7, o, br7);
    br8 = _mm512_fnmadd_pd(S2, o, br8);
  }
  { __m512d o = _mm512_load_pd(scr + 48);
    br1 = _mm512_fmadd_pd(S6, o, br1);
    br2 = _mm512_fnmadd_pd(S2, o, br2);
    br3 = _mm512_fmadd_pd(S3, o, br3);
    br4 = _mm512_fnmadd_pd(S5, o, br4);
    br5 = _mm512_fmadd_pd(S0, o, br5);
    br6 = _mm512_fmadd_pd(S7, o, br6);
    br7 = _mm512_fnmadd_pd(S1, o, br7);
    br8 = _mm512_fmadd_pd(S4, o, br8);
  }
  { __m512d o = _mm512_load_pd(scr + 56);
    br1 = _mm512_fmadd_pd(S7, o, br1);
    br2 = _mm512_fnmadd_pd(S0, o, br2);
    br3 = _mm512_fmadd_pd(S6, o, br3);
    br4 = _mm512_fnmadd_pd(S1, o, br4);
    br5 = _mm512_fmadd_pd(S5, o, br5);
    br6 = _mm512_fnmadd_pd(S2, o, br6);
    br7 = _mm512_fmadd_pd(S4, o, br7);
    br8 = _mm512_fnmadd_pd(S3, o, br8);
  }
  { __m512d a = _mm512_load_pd(scr + 136);
    _mm512_store_pd(im + 1*es, _mm512_sub_pd(a, br1));
    _mm512_store_pd(im + 16*es, _mm512_add_pd(a, br1)); }
  { __m512d a = _mm512_load_pd(scr + 144);
    _mm512_store_pd(im + 2*es, _mm512_sub_pd(a, br2));
    _mm512_store_pd(im + 15*es, _mm512_add_pd(a, br2)); }
  { __m512d a = _mm512_load_pd(scr + 152);
    _mm512_store_pd(im + 3*es, _mm512_sub_pd(a, br3));
    _mm512_store_pd(im + 14*es, _mm512_add_pd(a, br3)); }
  { __m512d a = _mm512_load_pd(scr + 160);
    _mm512_store_pd(im + 4*es, _mm512_sub_pd(a, br4));
    _mm512_store_pd(im + 13*es, _mm512_add_pd(a, br4)); }
  { __m512d a = _mm512_load_pd(scr + 168);
    _mm512_store_pd(im + 5*es, _mm512_sub_pd(a, br5));
    _mm512_store_pd(im + 12*es, _mm512_add_pd(a, br5)); }
  { __m512d a = _mm512_load_pd(scr + 176);
    _mm512_store_pd(im + 6*es, _mm512_sub_pd(a, br6));
    _mm512_store_pd(im + 11*es, _mm512_add_pd(a, br6)); }
  { __m512d a = _mm512_load_pd(scr + 184);
    _mm512_store_pd(im + 7*es, _mm512_sub_pd(a, br7));
    _mm512_store_pd(im + 10*es, _mm512_add_pd(a, br7)); }
  { __m512d a = _mm512_load_pd(scr + 192);
    _mm512_store_pd(im + 8*es, _mm512_sub_pd(a, br8));
    _mm512_store_pd(im + 9*es, _mm512_add_pd(a, br8)); }
  }
}
static const double CC_23[22] ALIGN64 = {0x1.ed037ea3d2dbbp-1,0x1.b57675cf309eep-1,0x1.5d779b07cfef7p-1,0x1.d71b4a0c5a6c8p-2,0x1.a0ad8bd1e2882p-3,-0x1.17855b599f3b9p-4,-0x1.56eaae597c776p-2,-0x1.2742a4a775cfbp-1,-0x1.8d2a07c16d46fp-1,-0x1.d59cb83ef99bcp-1,-0x1.fb3b3035aa6cdp-1,0x1.14459ad2be466p-2,0x1.0a06e851db7cap-1,0x1.763021aaa15dap-1,0x1.c698e42f47b09p-1,0x1.f54a827142577p-1,0x1.fece70dfd3efbp-1,0x1.e270060999288p-1,0x1.a249e0b897ca9p-1,0x1.431df5838f7efp-1,0x1.97f6748e524b2p-2,0x1.16de8a4564f0ap-3};
static void dft23_v(double* re, double* im, long es){
  double scr[272] __attribute__((aligned(64)));
  {
  const __m512d C0 = _mm512_set1_pd(CC_23[0]);
  const __m512d C1 = _mm512_set1_pd(CC_23[1]);
  const __m512d C2 = _mm512_set1_pd(CC_23[2]);
  const __m512d C3 = _mm512_set1_pd(CC_23[3]);
  const __m512d C4 = _mm512_set1_pd(CC_23[4]);
  const __m512d C5 = _mm512_set1_pd(CC_23[5]);
  const __m512d C6 = _mm512_set1_pd(CC_23[6]);
  const __m512d C7 = _mm512_set1_pd(CC_23[7]);
  const __m512d C8 = _mm512_set1_pd(CC_23[8]);
  const __m512d C9 = _mm512_set1_pd(CC_23[9]);
  const __m512d C10 = _mm512_set1_pd(CC_23[10]);
  __m512d x0r = _mm512_load_pd(re);
  __m512d s0r = x0r;
  __m512d ar1 = x0r;
  __m512d ar2 = x0r;
  __m512d ar3 = x0r;
  __m512d ar4 = x0r;
  __m512d ar5 = x0r;
  __m512d ar6 = x0r;
  __m512d ar7 = x0r;
  __m512d ar8 = x0r;
  __m512d ar9 = x0r;
  __m512d ar10 = x0r;
  __m512d ar11 = x0r;
  { __m512d p = _mm512_load_pd(re + 1*es), q = _mm512_load_pd(re + 22*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 0, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C0, e, ar1);
    ar2 = _mm512_fmadd_pd(C1, e, ar2);
    ar3 = _mm512_fmadd_pd(C2, e, ar3);
    ar4 = _mm512_fmadd_pd(C3, e, ar4);
    ar5 = _mm512_fmadd_pd(C4, e, ar5);
    ar6 = _mm512_fmadd_pd(C5, e, ar6);
    ar7 = _mm512_fmadd_pd(C6, e, ar7);
    ar8 = _mm512_fmadd_pd(C7, e, ar8);
    ar9 = _mm512_fmadd_pd(C8, e, ar9);
    ar10 = _mm512_fmadd_pd(C9, e, ar10);
    ar11 = _mm512_fmadd_pd(C10, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 2*es), q = _mm512_load_pd(re + 21*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 8, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C1, e, ar1);
    ar2 = _mm512_fmadd_pd(C3, e, ar2);
    ar3 = _mm512_fmadd_pd(C5, e, ar3);
    ar4 = _mm512_fmadd_pd(C7, e, ar4);
    ar5 = _mm512_fmadd_pd(C9, e, ar5);
    ar6 = _mm512_fmadd_pd(C10, e, ar6);
    ar7 = _mm512_fmadd_pd(C8, e, ar7);
    ar8 = _mm512_fmadd_pd(C6, e, ar8);
    ar9 = _mm512_fmadd_pd(C4, e, ar9);
    ar10 = _mm512_fmadd_pd(C2, e, ar10);
    ar11 = _mm512_fmadd_pd(C0, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 3*es), q = _mm512_load_pd(re + 20*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 16, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C2, e, ar1);
    ar2 = _mm512_fmadd_pd(C5, e, ar2);
    ar3 = _mm512_fmadd_pd(C8, e, ar3);
    ar4 = _mm512_fmadd_pd(C10, e, ar4);
    ar5 = _mm512_fmadd_pd(C7, e, ar5);
    ar6 = _mm512_fmadd_pd(C4, e, ar6);
    ar7 = _mm512_fmadd_pd(C1, e, ar7);
    ar8 = _mm512_fmadd_pd(C0, e, ar8);
    ar9 = _mm512_fmadd_pd(C3, e, ar9);
    ar10 = _mm512_fmadd_pd(C6, e, ar10);
    ar11 = _mm512_fmadd_pd(C9, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 4*es), q = _mm512_load_pd(re + 19*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 24, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C3, e, ar1);
    ar2 = _mm512_fmadd_pd(C7, e, ar2);
    ar3 = _mm512_fmadd_pd(C10, e, ar3);
    ar4 = _mm512_fmadd_pd(C6, e, ar4);
    ar5 = _mm512_fmadd_pd(C2, e, ar5);
    ar6 = _mm512_fmadd_pd(C0, e, ar6);
    ar7 = _mm512_fmadd_pd(C4, e, ar7);
    ar8 = _mm512_fmadd_pd(C8, e, ar8);
    ar9 = _mm512_fmadd_pd(C9, e, ar9);
    ar10 = _mm512_fmadd_pd(C5, e, ar10);
    ar11 = _mm512_fmadd_pd(C1, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 5*es), q = _mm512_load_pd(re + 18*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 32, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C4, e, ar1);
    ar2 = _mm512_fmadd_pd(C9, e, ar2);
    ar3 = _mm512_fmadd_pd(C7, e, ar3);
    ar4 = _mm512_fmadd_pd(C2, e, ar4);
    ar5 = _mm512_fmadd_pd(C1, e, ar5);
    ar6 = _mm512_fmadd_pd(C6, e, ar6);
    ar7 = _mm512_fmadd_pd(C10, e, ar7);
    ar8 = _mm512_fmadd_pd(C5, e, ar8);
    ar9 = _mm512_fmadd_pd(C0, e, ar9);
    ar10 = _mm512_fmadd_pd(C3, e, ar10);
    ar11 = _mm512_fmadd_pd(C8, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 6*es), q = _mm512_load_pd(re + 17*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 40, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C5, e, ar1);
    ar2 = _mm512_fmadd_pd(C10, e, ar2);
    ar3 = _mm512_fmadd_pd(C4, e, ar3);
    ar4 = _mm512_fmadd_pd(C0, e, ar4);
    ar5 = _mm512_fmadd_pd(C6, e, ar5);
    ar6 = _mm512_fmadd_pd(C9, e, ar6);
    ar7 = _mm512_fmadd_pd(C3, e, ar7);
    ar8 = _mm512_fmadd_pd(C1, e, ar8);
    ar9 = _mm512_fmadd_pd(C7, e, ar9);
    ar10 = _mm512_fmadd_pd(C8, e, ar10);
    ar11 = _mm512_fmadd_pd(C2, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 7*es), q = _mm512_load_pd(re + 16*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 48, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C6, e, ar1);
    ar2 = _mm512_fmadd_pd(C8, e, ar2);
    ar3 = _mm512_fmadd_pd(C1, e, ar3);
    ar4 = _mm512_fmadd_pd(C4, e, ar4);
    ar5 = _mm512_fmadd_pd(C10, e, ar5);
    ar6 = _mm512_fmadd_pd(C3, e, ar6);
    ar7 = _mm512_fmadd_pd(C2, e, ar7);
    ar8 = _mm512_fmadd_pd(C9, e, ar8);
    ar9 = _mm512_fmadd_pd(C5, e, ar9);
    ar10 = _mm512_fmadd_pd(C0, e, ar10);
    ar11 = _mm512_fmadd_pd(C7, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 8*es), q = _mm512_load_pd(re + 15*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 56, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C7, e, ar1);
    ar2 = _mm512_fmadd_pd(C6, e, ar2);
    ar3 = _mm512_fmadd_pd(C0, e, ar3);
    ar4 = _mm512_fmadd_pd(C8, e, ar4);
    ar5 = _mm512_fmadd_pd(C5, e, ar5);
    ar6 = _mm512_fmadd_pd(C1, e, ar6);
    ar7 = _mm512_fmadd_pd(C9, e, ar7);
    ar8 = _mm512_fmadd_pd(C4, e, ar8);
    ar9 = _mm512_fmadd_pd(C2, e, ar9);
    ar10 = _mm512_fmadd_pd(C10, e, ar10);
    ar11 = _mm512_fmadd_pd(C3, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 9*es), q = _mm512_load_pd(re + 14*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 64, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C8, e, ar1);
    ar2 = _mm512_fmadd_pd(C4, e, ar2);
    ar3 = _mm512_fmadd_pd(C3, e, ar3);
    ar4 = _mm512_fmadd_pd(C9, e, ar4);
    ar5 = _mm512_fmadd_pd(C0, e, ar5);
    ar6 = _mm512_fmadd_pd(C7, e, ar6);
    ar7 = _mm512_fmadd_pd(C5, e, ar7);
    ar8 = _mm512_fmadd_pd(C2, e, ar8);
    ar9 = _mm512_fmadd_pd(C10, e, ar9);
    ar10 = _mm512_fmadd_pd(C1, e, ar10);
    ar11 = _mm512_fmadd_pd(C6, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 10*es), q = _mm512_load_pd(re + 13*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 72, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C9, e, ar1);
    ar2 = _mm512_fmadd_pd(C2, e, ar2);
    ar3 = _mm512_fmadd_pd(C6, e, ar3);
    ar4 = _mm512_fmadd_pd(C5, e, ar4);
    ar5 = _mm512_fmadd_pd(C3, e, ar5);
    ar6 = _mm512_fmadd_pd(C8, e, ar6);
    ar7 = _mm512_fmadd_pd(C0, e, ar7);
    ar8 = _mm512_fmadd_pd(C10, e, ar8);
    ar9 = _mm512_fmadd_pd(C1, e, ar9);
    ar10 = _mm512_fmadd_pd(C7, e, ar10);
    ar11 = _mm512_fmadd_pd(C4, e, ar11);
  }
  { __m512d p = _mm512_load_pd(re + 11*es), q = _mm512_load_pd(re + 12*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 80, o); s0r = _mm512_add_pd(s0r, e);
    ar1 = _mm512_fmadd_pd(C10, e, ar1);
    ar2 = _mm512_fmadd_pd(C0, e, ar2);
    ar3 = _mm512_fmadd_pd(C9, e, ar3);
    ar4 = _mm512_fmadd_pd(C1, e, ar4);
    ar5 = _mm512_fmadd_pd(C8, e, ar5);
    ar6 = _mm512_fmadd_pd(C2, e, ar6);
    ar7 = _mm512_fmadd_pd(C7, e, ar7);
    ar8 = _mm512_fmadd_pd(C3, e, ar8);
    ar9 = _mm512_fmadd_pd(C6, e, ar9);
    ar10 = _mm512_fmadd_pd(C4, e, ar10);
    ar11 = _mm512_fmadd_pd(C5, e, ar11);
  }
  _mm512_store_pd(re, s0r);
  _mm512_store_pd(scr + 184, ar1);
  _mm512_store_pd(scr + 192, ar2);
  _mm512_store_pd(scr + 200, ar3);
  _mm512_store_pd(scr + 208, ar4);
  _mm512_store_pd(scr + 216, ar5);
  _mm512_store_pd(scr + 224, ar6);
  _mm512_store_pd(scr + 232, ar7);
  _mm512_store_pd(scr + 240, ar8);
  _mm512_store_pd(scr + 248, ar9);
  _mm512_store_pd(scr + 256, ar10);
  _mm512_store_pd(scr + 264, ar11);
  }
  {
  const __m512d S0 = _mm512_set1_pd(CC_23[11]);
  const __m512d S1 = _mm512_set1_pd(CC_23[12]);
  const __m512d S2 = _mm512_set1_pd(CC_23[13]);
  const __m512d S3 = _mm512_set1_pd(CC_23[14]);
  const __m512d S4 = _mm512_set1_pd(CC_23[15]);
  const __m512d S5 = _mm512_set1_pd(CC_23[16]);
  const __m512d S6 = _mm512_set1_pd(CC_23[17]);
  const __m512d S7 = _mm512_set1_pd(CC_23[18]);
  const __m512d S8 = _mm512_set1_pd(CC_23[19]);
  const __m512d S9 = _mm512_set1_pd(CC_23[20]);
  const __m512d S10 = _mm512_set1_pd(CC_23[21]);
  __m512d x0i = _mm512_load_pd(im);
  _mm512_store_pd(scr + 176, x0i);
  __m512d s0i = x0i;
  __m512d bi1 = _mm512_setzero_pd();
  __m512d bi2 = _mm512_setzero_pd();
  __m512d bi3 = _mm512_setzero_pd();
  __m512d bi4 = _mm512_setzero_pd();
  __m512d bi5 = _mm512_setzero_pd();
  __m512d bi6 = _mm512_setzero_pd();
  __m512d bi7 = _mm512_setzero_pd();
  __m512d bi8 = _mm512_setzero_pd();
  __m512d bi9 = _mm512_setzero_pd();
  __m512d bi10 = _mm512_setzero_pd();
  __m512d bi11 = _mm512_setzero_pd();
  { __m512d p = _mm512_load_pd(im + 1*es), q = _mm512_load_pd(im + 22*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 88, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S0, o, bi1);
    bi2 = _mm512_fmadd_pd(S1, o, bi2);
    bi3 = _mm512_fmadd_pd(S2, o, bi3);
    bi4 = _mm512_fmadd_pd(S3, o, bi4);
    bi5 = _mm512_fmadd_pd(S4, o, bi5);
    bi6 = _mm512_fmadd_pd(S5, o, bi6);
    bi7 = _mm512_fmadd_pd(S6, o, bi7);
    bi8 = _mm512_fmadd_pd(S7, o, bi8);
    bi9 = _mm512_fmadd_pd(S8, o, bi9);
    bi10 = _mm512_fmadd_pd(S9, o, bi10);
    bi11 = _mm512_fmadd_pd(S10, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 2*es), q = _mm512_load_pd(im + 21*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 96, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S1, o, bi1);
    bi2 = _mm512_fmadd_pd(S3, o, bi2);
    bi3 = _mm512_fmadd_pd(S5, o, bi3);
    bi4 = _mm512_fmadd_pd(S7, o, bi4);
    bi5 = _mm512_fmadd_pd(S9, o, bi5);
    bi6 = _mm512_fnmadd_pd(S10, o, bi6);
    bi7 = _mm512_fnmadd_pd(S8, o, bi7);
    bi8 = _mm512_fnmadd_pd(S6, o, bi8);
    bi9 = _mm512_fnmadd_pd(S4, o, bi9);
    bi10 = _mm512_fnmadd_pd(S2, o, bi10);
    bi11 = _mm512_fnmadd_pd(S0, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 3*es), q = _mm512_load_pd(im + 20*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 104, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S2, o, bi1);
    bi2 = _mm512_fmadd_pd(S5, o, bi2);
    bi3 = _mm512_fmadd_pd(S8, o, bi3);
    bi4 = _mm512_fnmadd_pd(S10, o, bi4);
    bi5 = _mm512_fnmadd_pd(S7, o, bi5);
    bi6 = _mm512_fnmadd_pd(S4, o, bi6);
    bi7 = _mm512_fnmadd_pd(S1, o, bi7);
    bi8 = _mm512_fmadd_pd(S0, o, bi8);
    bi9 = _mm512_fmadd_pd(S3, o, bi9);
    bi10 = _mm512_fmadd_pd(S6, o, bi10);
    bi11 = _mm512_fmadd_pd(S9, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 4*es), q = _mm512_load_pd(im + 19*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 112, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S3, o, bi1);
    bi2 = _mm512_fmadd_pd(S7, o, bi2);
    bi3 = _mm512_fnmadd_pd(S10, o, bi3);
    bi4 = _mm512_fnmadd_pd(S6, o, bi4);
    bi5 = _mm512_fnmadd_pd(S2, o, bi5);
    bi6 = _mm512_fmadd_pd(S0, o, bi6);
    bi7 = _mm512_fmadd_pd(S4, o, bi7);
    bi8 = _mm512_fmadd_pd(S8, o, bi8);
    bi9 = _mm512_fnmadd_pd(S9, o, bi9);
    bi10 = _mm512_fnmadd_pd(S5, o, bi10);
    bi11 = _mm512_fnmadd_pd(S1, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 5*es), q = _mm512_load_pd(im + 18*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 120, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S4, o, bi1);
    bi2 = _mm512_fmadd_pd(S9, o, bi2);
    bi3 = _mm512_fnmadd_pd(S7, o, bi3);
    bi4 = _mm512_fnmadd_pd(S2, o, bi4);
    bi5 = _mm512_fmadd_pd(S1, o, bi5);
    bi6 = _mm512_fmadd_pd(S6, o, bi6);
    bi7 = _mm512_fnmadd_pd(S10, o, bi7);
    bi8 = _mm512_fnmadd_pd(S5, o, bi8);
    bi9 = _mm512_fnmadd_pd(S0, o, bi9);
    bi10 = _mm512_fmadd_pd(S3, o, bi10);
    bi11 = _mm512_fmadd_pd(S8, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 6*es), q = _mm512_load_pd(im + 17*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 128, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S5, o, bi1);
    bi2 = _mm512_fnmadd_pd(S10, o, bi2);
    bi3 = _mm512_fnmadd_pd(S4, o, bi3);
    bi4 = _mm512_fmadd_pd(S0, o, bi4);
    bi5 = _mm512_fmadd_pd(S6, o, bi5);
    bi6 = _mm512_fnmadd_pd(S9, o, bi6);
    bi7 = _mm512_fnmadd_pd(S3, o, bi7);
    bi8 = _mm512_fmadd_pd(S1, o, bi8);
    bi9 = _mm512_fmadd_pd(S7, o, bi9);
    bi10 = _mm512_fnmadd_pd(S8, o, bi10);
    bi11 = _mm512_fnmadd_pd(S2, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 7*es), q = _mm512_load_pd(im + 16*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 136, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S6, o, bi1);
    bi2 = _mm512_fnmadd_pd(S8, o, bi2);
    bi3 = _mm512_fnmadd_pd(S1, o, bi3);
    bi4 = _mm512_fmadd_pd(S4, o, bi4);
    bi5 = _mm512_fnmadd_pd(S10, o, bi5);
    bi6 = _mm512_fnmadd_pd(S3, o, bi6);
    bi7 = _mm512_fmadd_pd(S2, o, bi7);
    bi8 = _mm512_fmadd_pd(S9, o, bi8);
    bi9 = _mm512_fnmadd_pd(S5, o, bi9);
    bi10 = _mm512_fmadd_pd(S0, o, bi10);
    bi11 = _mm512_fmadd_pd(S7, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 8*es), q = _mm512_load_pd(im + 15*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 144, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S7, o, bi1);
    bi2 = _mm512_fnmadd_pd(S6, o, bi2);
    bi3 = _mm512_fmadd_pd(S0, o, bi3);
    bi4 = _mm512_fmadd_pd(S8, o, bi4);
    bi5 = _mm512_fnmadd_pd(S5, o, bi5);
    bi6 = _mm512_fmadd_pd(S1, o, bi6);
    bi7 = _mm512_fmadd_pd(S9, o, bi7);
    bi8 = _mm512_fnmadd_pd(S4, o, bi8);
    bi9 = _mm512_fmadd_pd(S2, o, bi9);
    bi10 = _mm512_fmadd_pd(S10, o, bi10);
    bi11 = _mm512_fnmadd_pd(S3, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 9*es), q = _mm512_load_pd(im + 14*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 152, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S8, o, bi1);
    bi2 = _mm512_fnmadd_pd(S4, o, bi2);
    bi3 = _mm512_fmadd_pd(S3, o, bi3);
    bi4 = _mm512_fnmadd_pd(S9, o, bi4);
    bi5 = _mm512_fnmadd_pd(S0, o, bi5);
    bi6 = _mm512_fmadd_pd(S7, o, bi6);
    bi7 = _mm512_fnmadd_pd(S5, o, bi7);
    bi8 = _mm512_fmadd_pd(S2, o, bi8);
    bi9 = _mm512_fnmadd_pd(S10, o, bi9);
    bi10 = _mm512_fnmadd_pd(S1, o, bi10);
    bi11 = _mm512_fmadd_pd(S6, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 10*es), q = _mm512_load_pd(im + 13*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 160, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S9, o, bi1);
    bi2 = _mm512_fnmadd_pd(S2, o, bi2);
    bi3 = _mm512_fmadd_pd(S6, o, bi3);
    bi4 = _mm512_fnmadd_pd(S5, o, bi4);
    bi5 = _mm512_fmadd_pd(S3, o, bi5);
    bi6 = _mm512_fnmadd_pd(S8, o, bi6);
    bi7 = _mm512_fmadd_pd(S0, o, bi7);
    bi8 = _mm512_fmadd_pd(S10, o, bi8);
    bi9 = _mm512_fnmadd_pd(S1, o, bi9);
    bi10 = _mm512_fmadd_pd(S7, o, bi10);
    bi11 = _mm512_fnmadd_pd(S4, o, bi11);
  }
  { __m512d p = _mm512_load_pd(im + 11*es), q = _mm512_load_pd(im + 12*es);
    __m512d e = _mm512_add_pd(p,q), o = _mm512_sub_pd(p,q);
    _mm512_store_pd(scr + 168, e); s0i = _mm512_add_pd(s0i, e);
    bi1 = _mm512_fmadd_pd(S10, o, bi1);
    bi2 = _mm512_fnmadd_pd(S0, o, bi2);
    bi3 = _mm512_fmadd_pd(S9, o, bi3);
    bi4 = _mm512_fnmadd_pd(S1, o, bi4);
    bi5 = _mm512_fmadd_pd(S8, o, bi5);
    bi6 = _mm512_fnmadd_pd(S2, o, bi6);
    bi7 = _mm512_fmadd_pd(S7, o, bi7);
    bi8 = _mm512_fnmadd_pd(S3, o, bi8);
    bi9 = _mm512_fmadd_pd(S6, o, bi9);
    bi10 = _mm512_fnmadd_pd(S4, o, bi10);
    bi11 = _mm512_fmadd_pd(S5, o, bi11);
  }
  _mm512_store_pd(im, s0i);
  { __m512d a = _mm512_load_pd(scr + 184);
    _mm512_store_pd(re + 1*es, _mm512_add_pd(a, bi1));
    _mm512_store_pd(re + 22*es, _mm512_sub_pd(a, bi1)); }
  { __m512d a = _mm512_load_pd(scr + 192);
    _mm512_store_pd(re + 2*es, _mm512_add_pd(a, bi2));
    _mm512_store_pd(re + 21*es, _mm512_sub_pd(a, bi2)); }
  { __m512d a = _mm512_load_pd(scr + 200);
    _mm512_store_pd(re + 3*es, _mm512_add_pd(a, bi3));
    _mm512_store_pd(re + 20*es, _mm512_sub_pd(a, bi3)); }
  { __m512d a = _mm512_load_pd(scr + 208);
    _mm512_store_pd(re + 4*es, _mm512_add_pd(a, bi4));
    _mm512_store_pd(re + 19*es, _mm512_sub_pd(a, bi4)); }
  { __m512d a = _mm512_load_pd(scr + 216);
    _mm512_store_pd(re + 5*es, _mm512_add_pd(a, bi5));
    _mm512_store_pd(re + 18*es, _mm512_sub_pd(a, bi5)); }
  { __m512d a = _mm512_load_pd(scr + 224);
    _mm512_store_pd(re + 6*es, _mm512_add_pd(a, bi6));
    _mm512_store_pd(re + 17*es, _mm512_sub_pd(a, bi6)); }
  { __m512d a = _mm512_load_pd(scr + 232);
    _mm512_store_pd(re + 7*es, _mm512_add_pd(a, bi7));
    _mm512_store_pd(re + 16*es, _mm512_sub_pd(a, bi7)); }
  { __m512d a = _mm512_load_pd(scr + 240);
    _mm512_store_pd(re + 8*es, _mm512_add_pd(a, bi8));
    _mm512_store_pd(re + 15*es, _mm512_sub_pd(a, bi8)); }
  { __m512d a = _mm512_load_pd(scr + 248);
    _mm512_store_pd(re + 9*es, _mm512_add_pd(a, bi9));
    _mm512_store_pd(re + 14*es, _mm512_sub_pd(a, bi9)); }
  { __m512d a = _mm512_load_pd(scr + 256);
    _mm512_store_pd(re + 10*es, _mm512_add_pd(a, bi10));
    _mm512_store_pd(re + 13*es, _mm512_sub_pd(a, bi10)); }
  { __m512d a = _mm512_load_pd(scr + 264);
    _mm512_store_pd(re + 11*es, _mm512_add_pd(a, bi11));
    _mm512_store_pd(re + 12*es, _mm512_sub_pd(a, bi11)); }
  }
  {
  const __m512d C0 = _mm512_set1_pd(CC_23[0]);
  const __m512d C1 = _mm512_set1_pd(CC_23[1]);
  const __m512d C2 = _mm512_set1_pd(CC_23[2]);
  const __m512d C3 = _mm512_set1_pd(CC_23[3]);
  const __m512d C4 = _mm512_set1_pd(CC_23[4]);
  const __m512d C5 = _mm512_set1_pd(CC_23[5]);
  const __m512d C6 = _mm512_set1_pd(CC_23[6]);
  const __m512d C7 = _mm512_set1_pd(CC_23[7]);
  const __m512d C8 = _mm512_set1_pd(CC_23[8]);
  const __m512d C9 = _mm512_set1_pd(CC_23[9]);
  const __m512d C10 = _mm512_set1_pd(CC_23[10]);
  __m512d x0i = _mm512_load_pd(scr + 176);
  __m512d ai1 = x0i;
  __m512d ai2 = x0i;
  __m512d ai3 = x0i;
  __m512d ai4 = x0i;
  __m512d ai5 = x0i;
  __m512d ai6 = x0i;
  __m512d ai7 = x0i;
  __m512d ai8 = x0i;
  __m512d ai9 = x0i;
  __m512d ai10 = x0i;
  __m512d ai11 = x0i;
  { __m512d e = _mm512_load_pd(scr + 88);
    ai1 = _mm512_fmadd_pd(C0, e, ai1);
    ai2 = _mm512_fmadd_pd(C1, e, ai2);
    ai3 = _mm512_fmadd_pd(C2, e, ai3);
    ai4 = _mm512_fmadd_pd(C3, e, ai4);
    ai5 = _mm512_fmadd_pd(C4, e, ai5);
    ai6 = _mm512_fmadd_pd(C5, e, ai6);
    ai7 = _mm512_fmadd_pd(C6, e, ai7);
    ai8 = _mm512_fmadd_pd(C7, e, ai8);
    ai9 = _mm512_fmadd_pd(C8, e, ai9);
    ai10 = _mm512_fmadd_pd(C9, e, ai10);
    ai11 = _mm512_fmadd_pd(C10, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 96);
    ai1 = _mm512_fmadd_pd(C1, e, ai1);
    ai2 = _mm512_fmadd_pd(C3, e, ai2);
    ai3 = _mm512_fmadd_pd(C5, e, ai3);
    ai4 = _mm512_fmadd_pd(C7, e, ai4);
    ai5 = _mm512_fmadd_pd(C9, e, ai5);
    ai6 = _mm512_fmadd_pd(C10, e, ai6);
    ai7 = _mm512_fmadd_pd(C8, e, ai7);
    ai8 = _mm512_fmadd_pd(C6, e, ai8);
    ai9 = _mm512_fmadd_pd(C4, e, ai9);
    ai10 = _mm512_fmadd_pd(C2, e, ai10);
    ai11 = _mm512_fmadd_pd(C0, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 104);
    ai1 = _mm512_fmadd_pd(C2, e, ai1);
    ai2 = _mm512_fmadd_pd(C5, e, ai2);
    ai3 = _mm512_fmadd_pd(C8, e, ai3);
    ai4 = _mm512_fmadd_pd(C10, e, ai4);
    ai5 = _mm512_fmadd_pd(C7, e, ai5);
    ai6 = _mm512_fmadd_pd(C4, e, ai6);
    ai7 = _mm512_fmadd_pd(C1, e, ai7);
    ai8 = _mm512_fmadd_pd(C0, e, ai8);
    ai9 = _mm512_fmadd_pd(C3, e, ai9);
    ai10 = _mm512_fmadd_pd(C6, e, ai10);
    ai11 = _mm512_fmadd_pd(C9, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 112);
    ai1 = _mm512_fmadd_pd(C3, e, ai1);
    ai2 = _mm512_fmadd_pd(C7, e, ai2);
    ai3 = _mm512_fmadd_pd(C10, e, ai3);
    ai4 = _mm512_fmadd_pd(C6, e, ai4);
    ai5 = _mm512_fmadd_pd(C2, e, ai5);
    ai6 = _mm512_fmadd_pd(C0, e, ai6);
    ai7 = _mm512_fmadd_pd(C4, e, ai7);
    ai8 = _mm512_fmadd_pd(C8, e, ai8);
    ai9 = _mm512_fmadd_pd(C9, e, ai9);
    ai10 = _mm512_fmadd_pd(C5, e, ai10);
    ai11 = _mm512_fmadd_pd(C1, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 120);
    ai1 = _mm512_fmadd_pd(C4, e, ai1);
    ai2 = _mm512_fmadd_pd(C9, e, ai2);
    ai3 = _mm512_fmadd_pd(C7, e, ai3);
    ai4 = _mm512_fmadd_pd(C2, e, ai4);
    ai5 = _mm512_fmadd_pd(C1, e, ai5);
    ai6 = _mm512_fmadd_pd(C6, e, ai6);
    ai7 = _mm512_fmadd_pd(C10, e, ai7);
    ai8 = _mm512_fmadd_pd(C5, e, ai8);
    ai9 = _mm512_fmadd_pd(C0, e, ai9);
    ai10 = _mm512_fmadd_pd(C3, e, ai10);
    ai11 = _mm512_fmadd_pd(C8, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 128);
    ai1 = _mm512_fmadd_pd(C5, e, ai1);
    ai2 = _mm512_fmadd_pd(C10, e, ai2);
    ai3 = _mm512_fmadd_pd(C4, e, ai3);
    ai4 = _mm512_fmadd_pd(C0, e, ai4);
    ai5 = _mm512_fmadd_pd(C6, e, ai5);
    ai6 = _mm512_fmadd_pd(C9, e, ai6);
    ai7 = _mm512_fmadd_pd(C3, e, ai7);
    ai8 = _mm512_fmadd_pd(C1, e, ai8);
    ai9 = _mm512_fmadd_pd(C7, e, ai9);
    ai10 = _mm512_fmadd_pd(C8, e, ai10);
    ai11 = _mm512_fmadd_pd(C2, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 136);
    ai1 = _mm512_fmadd_pd(C6, e, ai1);
    ai2 = _mm512_fmadd_pd(C8, e, ai2);
    ai3 = _mm512_fmadd_pd(C1, e, ai3);
    ai4 = _mm512_fmadd_pd(C4, e, ai4);
    ai5 = _mm512_fmadd_pd(C10, e, ai5);
    ai6 = _mm512_fmadd_pd(C3, e, ai6);
    ai7 = _mm512_fmadd_pd(C2, e, ai7);
    ai8 = _mm512_fmadd_pd(C9, e, ai8);
    ai9 = _mm512_fmadd_pd(C5, e, ai9);
    ai10 = _mm512_fmadd_pd(C0, e, ai10);
    ai11 = _mm512_fmadd_pd(C7, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 144);
    ai1 = _mm512_fmadd_pd(C7, e, ai1);
    ai2 = _mm512_fmadd_pd(C6, e, ai2);
    ai3 = _mm512_fmadd_pd(C0, e, ai3);
    ai4 = _mm512_fmadd_pd(C8, e, ai4);
    ai5 = _mm512_fmadd_pd(C5, e, ai5);
    ai6 = _mm512_fmadd_pd(C1, e, ai6);
    ai7 = _mm512_fmadd_pd(C9, e, ai7);
    ai8 = _mm512_fmadd_pd(C4, e, ai8);
    ai9 = _mm512_fmadd_pd(C2, e, ai9);
    ai10 = _mm512_fmadd_pd(C10, e, ai10);
    ai11 = _mm512_fmadd_pd(C3, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 152);
    ai1 = _mm512_fmadd_pd(C8, e, ai1);
    ai2 = _mm512_fmadd_pd(C4, e, ai2);
    ai3 = _mm512_fmadd_pd(C3, e, ai3);
    ai4 = _mm512_fmadd_pd(C9, e, ai4);
    ai5 = _mm512_fmadd_pd(C0, e, ai5);
    ai6 = _mm512_fmadd_pd(C7, e, ai6);
    ai7 = _mm512_fmadd_pd(C5, e, ai7);
    ai8 = _mm512_fmadd_pd(C2, e, ai8);
    ai9 = _mm512_fmadd_pd(C10, e, ai9);
    ai10 = _mm512_fmadd_pd(C1, e, ai10);
    ai11 = _mm512_fmadd_pd(C6, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 160);
    ai1 = _mm512_fmadd_pd(C9, e, ai1);
    ai2 = _mm512_fmadd_pd(C2, e, ai2);
    ai3 = _mm512_fmadd_pd(C6, e, ai3);
    ai4 = _mm512_fmadd_pd(C5, e, ai4);
    ai5 = _mm512_fmadd_pd(C3, e, ai5);
    ai6 = _mm512_fmadd_pd(C8, e, ai6);
    ai7 = _mm512_fmadd_pd(C0, e, ai7);
    ai8 = _mm512_fmadd_pd(C10, e, ai8);
    ai9 = _mm512_fmadd_pd(C1, e, ai9);
    ai10 = _mm512_fmadd_pd(C7, e, ai10);
    ai11 = _mm512_fmadd_pd(C4, e, ai11);
  }
  { __m512d e = _mm512_load_pd(scr + 168);
    ai1 = _mm512_fmadd_pd(C10, e, ai1);
    ai2 = _mm512_fmadd_pd(C0, e, ai2);
    ai3 = _mm512_fmadd_pd(C9, e, ai3);
    ai4 = _mm512_fmadd_pd(C1, e, ai4);
    ai5 = _mm512_fmadd_pd(C8, e, ai5);
    ai6 = _mm512_fmadd_pd(C2, e, ai6);
    ai7 = _mm512_fmadd_pd(C7, e, ai7);
    ai8 = _mm512_fmadd_pd(C3, e, ai8);
    ai9 = _mm512_fmadd_pd(C6, e, ai9);
    ai10 = _mm512_fmadd_pd(C4, e, ai10);
    ai11 = _mm512_fmadd_pd(C5, e, ai11);
  }
  _mm512_store_pd(scr + 184, ai1);
  _mm512_store_pd(scr + 192, ai2);
  _mm512_store_pd(scr + 200, ai3);
  _mm512_store_pd(scr + 208, ai4);
  _mm512_store_pd(scr + 216, ai5);
  _mm512_store_pd(scr + 224, ai6);
  _mm512_store_pd(scr + 232, ai7);
  _mm512_store_pd(scr + 240, ai8);
  _mm512_store_pd(scr + 248, ai9);
  _mm512_store_pd(scr + 256, ai10);
  _mm512_store_pd(scr + 264, ai11);
  }
  {
  const __m512d S0 = _mm512_set1_pd(CC_23[11]);
  const __m512d S1 = _mm512_set1_pd(CC_23[12]);
  const __m512d S2 = _mm512_set1_pd(CC_23[13]);
  const __m512d S3 = _mm512_set1_pd(CC_23[14]);
  const __m512d S4 = _mm512_set1_pd(CC_23[15]);
  const __m512d S5 = _mm512_set1_pd(CC_23[16]);
  const __m512d S6 = _mm512_set1_pd(CC_23[17]);
  const __m512d S7 = _mm512_set1_pd(CC_23[18]);
  const __m512d S8 = _mm512_set1_pd(CC_23[19]);
  const __m512d S9 = _mm512_set1_pd(CC_23[20]);
  const __m512d S10 = _mm512_set1_pd(CC_23[21]);
  __m512d br1 = _mm512_setzero_pd();
  __m512d br2 = _mm512_setzero_pd();
  __m512d br3 = _mm512_setzero_pd();
  __m512d br4 = _mm512_setzero_pd();
  __m512d br5 = _mm512_setzero_pd();
  __m512d br6 = _mm512_setzero_pd();
  __m512d br7 = _mm512_setzero_pd();
  __m512d br8 = _mm512_setzero_pd();
  __m512d br9 = _mm512_setzero_pd();
  __m512d br10 = _mm512_setzero_pd();
  __m512d br11 = _mm512_setzero_pd();
  { __m512d o = _mm512_load_pd(scr + 0);
    br1 = _mm512_fmadd_pd(S0, o, br1);
    br2 = _mm512_fmadd_pd(S1, o, br2);
    br3 = _mm512_fmadd_pd(S2, o, br3);
    br4 = _mm512_fmadd_pd(S3, o, br4);
    br5 = _mm512_fmadd_pd(S4, o, br5);
    br6 = _mm512_fmadd_pd(S5, o, br6);
    br7 = _mm512_fmadd_pd(S6, o, br7);
    br8 = _mm512_fmadd_pd(S7, o, br8);
    br9 = _mm512_fmadd_pd(S8, o, br9);
    br10 = _mm512_fmadd_pd(S9, o, br10);
    br11 = _mm512_fmadd_pd(S10, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 8);
    br1 = _mm512_fmadd_pd(S1, o, br1);
    br2 = _mm512_fmadd_pd(S3, o, br2);
    br3 = _mm512_fmadd_pd(S5, o, br3);
    br4 = _mm512_fmadd_pd(S7, o, br4);
    br5 = _mm512_fmadd_pd(S9, o, br5);
    br6 = _mm512_fnmadd_pd(S10, o, br6);
    br7 = _mm512_fnmadd_pd(S8, o, br7);
    br8 = _mm512_fnmadd_pd(S6, o, br8);
    br9 = _mm512_fnmadd_pd(S4, o, br9);
    br10 = _mm512_fnmadd_pd(S2, o, br10);
    br11 = _mm512_fnmadd_pd(S0, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 16);
    br1 = _mm512_fmadd_pd(S2, o, br1);
    br2 = _mm512_fmadd_pd(S5, o, br2);
    br3 = _mm512_fmadd_pd(S8, o, br3);
    br4 = _mm512_fnmadd_pd(S10, o, br4);
    br5 = _mm512_fnmadd_pd(S7, o, br5);
    br6 = _mm512_fnmadd_pd(S4, o, br6);
    br7 = _mm512_fnmadd_pd(S1, o, br7);
    br8 = _mm512_fmadd_pd(S0, o, br8);
    br9 = _mm512_fmadd_pd(S3, o, br9);
    br10 = _mm512_fmadd_pd(S6, o, br10);
    br11 = _mm512_fmadd_pd(S9, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 24);
    br1 = _mm512_fmadd_pd(S3, o, br1);
    br2 = _mm512_fmadd_pd(S7, o, br2);
    br3 = _mm512_fnmadd_pd(S10, o, br3);
    br4 = _mm512_fnmadd_pd(S6, o, br4);
    br5 = _mm512_fnmadd_pd(S2, o, br5);
    br6 = _mm512_fmadd_pd(S0, o, br6);
    br7 = _mm512_fmadd_pd(S4, o, br7);
    br8 = _mm512_fmadd_pd(S8, o, br8);
    br9 = _mm512_fnmadd_pd(S9, o, br9);
    br10 = _mm512_fnmadd_pd(S5, o, br10);
    br11 = _mm512_fnmadd_pd(S1, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 32);
    br1 = _mm512_fmadd_pd(S4, o, br1);
    br2 = _mm512_fmadd_pd(S9, o, br2);
    br3 = _mm512_fnmadd_pd(S7, o, br3);
    br4 = _mm512_fnmadd_pd(S2, o, br4);
    br5 = _mm512_fmadd_pd(S1, o, br5);
    br6 = _mm512_fmadd_pd(S6, o, br6);
    br7 = _mm512_fnmadd_pd(S10, o, br7);
    br8 = _mm512_fnmadd_pd(S5, o, br8);
    br9 = _mm512_fnmadd_pd(S0, o, br9);
    br10 = _mm512_fmadd_pd(S3, o, br10);
    br11 = _mm512_fmadd_pd(S8, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 40);
    br1 = _mm512_fmadd_pd(S5, o, br1);
    br2 = _mm512_fnmadd_pd(S10, o, br2);
    br3 = _mm512_fnmadd_pd(S4, o, br3);
    br4 = _mm512_fmadd_pd(S0, o, br4);
    br5 = _mm512_fmadd_pd(S6, o, br5);
    br6 = _mm512_fnmadd_pd(S9, o, br6);
    br7 = _mm512_fnmadd_pd(S3, o, br7);
    br8 = _mm512_fmadd_pd(S1, o, br8);
    br9 = _mm512_fmadd_pd(S7, o, br9);
    br10 = _mm512_fnmadd_pd(S8, o, br10);
    br11 = _mm512_fnmadd_pd(S2, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 48);
    br1 = _mm512_fmadd_pd(S6, o, br1);
    br2 = _mm512_fnmadd_pd(S8, o, br2);
    br3 = _mm512_fnmadd_pd(S1, o, br3);
    br4 = _mm512_fmadd_pd(S4, o, br4);
    br5 = _mm512_fnmadd_pd(S10, o, br5);
    br6 = _mm512_fnmadd_pd(S3, o, br6);
    br7 = _mm512_fmadd_pd(S2, o, br7);
    br8 = _mm512_fmadd_pd(S9, o, br8);
    br9 = _mm512_fnmadd_pd(S5, o, br9);
    br10 = _mm512_fmadd_pd(S0, o, br10);
    br11 = _mm512_fmadd_pd(S7, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 56);
    br1 = _mm512_fmadd_pd(S7, o, br1);
    br2 = _mm512_fnmadd_pd(S6, o, br2);
    br3 = _mm512_fmadd_pd(S0, o, br3);
    br4 = _mm512_fmadd_pd(S8, o, br4);
    br5 = _mm512_fnmadd_pd(S5, o, br5);
    br6 = _mm512_fmadd_pd(S1, o, br6);
    br7 = _mm512_fmadd_pd(S9, o, br7);
    br8 = _mm512_fnmadd_pd(S4, o, br8);
    br9 = _mm512_fmadd_pd(S2, o, br9);
    br10 = _mm512_fmadd_pd(S10, o, br10);
    br11 = _mm512_fnmadd_pd(S3, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 64);
    br1 = _mm512_fmadd_pd(S8, o, br1);
    br2 = _mm512_fnmadd_pd(S4, o, br2);
    br3 = _mm512_fmadd_pd(S3, o, br3);
    br4 = _mm512_fnmadd_pd(S9, o, br4);
    br5 = _mm512_fnmadd_pd(S0, o, br5);
    br6 = _mm512_fmadd_pd(S7, o, br6);
    br7 = _mm512_fnmadd_pd(S5, o, br7);
    br8 = _mm512_fmadd_pd(S2, o, br8);
    br9 = _mm512_fnmadd_pd(S10, o, br9);
    br10 = _mm512_fnmadd_pd(S1, o, br10);
    br11 = _mm512_fmadd_pd(S6, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 72);
    br1 = _mm512_fmadd_pd(S9, o, br1);
    br2 = _mm512_fnmadd_pd(S2, o, br2);
    br3 = _mm512_fmadd_pd(S6, o, br3);
    br4 = _mm512_fnmadd_pd(S5, o, br4);
    br5 = _mm512_fmadd_pd(S3, o, br5);
    br6 = _mm512_fnmadd_pd(S8, o, br6);
    br7 = _mm512_fmadd_pd(S0, o, br7);
    br8 = _mm512_fmadd_pd(S10, o, br8);
    br9 = _mm512_fnmadd_pd(S1, o, br9);
    br10 = _mm512_fmadd_pd(S7, o, br10);
    br11 = _mm512_fnmadd_pd(S4, o, br11);
  }
  { __m512d o = _mm512_load_pd(scr + 80);
    br1 = _mm512_fmadd_pd(S10, o, br1);
    br2 = _mm512_fnmadd_pd(S0, o, br2);
    br3 = _mm512_fmadd_pd(S9, o, br3);
    br4 = _mm512_fnmadd_pd(S1, o, br4);
    br5 = _mm512_fmadd_pd(S8, o, br5);
    br6 = _mm512_fnmadd_pd(S2, o, br6);
    br7 = _mm512_fmadd_pd(S7, o, br7);
    br8 = _mm512_fnmadd_pd(S3, o, br8);
    br9 = _mm512_fmadd_pd(S6, o, br9);
    br10 = _mm512_fnmadd_pd(S4, o, br10);
    br11 = _mm512_fmadd_pd(S5, o, br11);
  }
  { __m512d a = _mm512_load_pd(scr + 184);
    _mm512_store_pd(im + 1*es, _mm512_sub_pd(a, br1));
    _mm512_store_pd(im + 22*es, _mm512_add_pd(a, br1)); }
  { __m512d a = _mm512_load_pd(scr + 192);
    _mm512_store_pd(im + 2*es, _mm512_sub_pd(a, br2));
    _mm512_store_pd(im + 21*es, _mm512_add_pd(a, br2)); }
  { __m512d a = _mm512_load_pd(scr + 200);
    _mm512_store_pd(im + 3*es, _mm512_sub_pd(a, br3));
    _mm512_store_pd(im + 20*es, _mm512_add_pd(a, br3)); }
  { __m512d a = _mm512_load_pd(scr + 208);
    _mm512_store_pd(im + 4*es, _mm512_sub_pd(a, br4));
    _mm512_store_pd(im + 19*es, _mm512_add_pd(a, br4)); }
  { __m512d a = _mm512_load_pd(scr + 216);
    _mm512_store_pd(im + 5*es, _mm512_sub_pd(a, br5));
    _mm512_store_pd(im + 18*es, _mm512_add_pd(a, br5)); }
  { __m512d a = _mm512_load_pd(scr + 224);
    _mm512_store_pd(im + 6*es, _mm512_sub_pd(a, br6));
    _mm512_store_pd(im + 17*es, _mm512_add_pd(a, br6)); }
  { __m512d a = _mm512_load_pd(scr + 232);
    _mm512_store_pd(im + 7*es, _mm512_sub_pd(a, br7));
    _mm512_store_pd(im + 16*es, _mm512_add_pd(a, br7)); }
  { __m512d a = _mm512_load_pd(scr + 240);
    _mm512_store_pd(im + 8*es, _mm512_sub_pd(a, br8));
    _mm512_store_pd(im + 15*es, _mm512_add_pd(a, br8)); }
  { __m512d a = _mm512_load_pd(scr + 248);
    _mm512_store_pd(im + 9*es, _mm512_sub_pd(a, br9));
    _mm512_store_pd(im + 14*es, _mm512_add_pd(a, br9)); }
  { __m512d a = _mm512_load_pd(scr + 256);
    _mm512_store_pd(im + 10*es, _mm512_sub_pd(a, br10));
    _mm512_store_pd(im + 13*es, _mm512_add_pd(a, br10)); }
  { __m512d a = _mm512_load_pd(scr + 264);
    _mm512_store_pd(im + 11*es, _mm512_sub_pd(a, br11));
    _mm512_store_pd(im + 12*es, _mm512_add_pd(a, br11)); }
  }
}
static const __m512i IDX_EVEN_ = {0,2,4,6,8,10,12,14};
static const __m512i IDX_ODD_  = {1,3,5,7,9,11,13,15};
#define DEINT(lo,hi,re,im) do{ re=_mm512_permutex2var_pd(lo, IDX_EVEN_, hi); im=_mm512_permutex2var_pd(lo, IDX_ODD_, hi);}while(0)
static const __m512i IDX_ILO_ = {0,8,1,9,2,10,3,11};
static const __m512i IDX_IHI_ = {4,12,5,13,6,14,7,15};
#define INTER(re,im,lo,hi) do{ lo=_mm512_permutex2var_pd(re, IDX_ILO_, im); hi=_mm512_permutex2var_pd(re, IDX_IHI_, im);}while(0)
static __attribute__((always_inline)) inline void dft6_v(double* re, double* im, long es){
__m512d t1 = _mm512_load_pd(re + 0*es);
__m512d t2 = _mm512_load_pd(im + 0*es);
__m512d t3 = _mm512_load_pd(re + 2*es);
__m512d t4 = _mm512_load_pd(im + 2*es);
__m512d t5 = _mm512_load_pd(re + 4*es);
__m512d t6 = _mm512_load_pd(im + 4*es);
__m512d t7 = _mm512_load_pd(re + 3*es);
__m512d t8 = _mm512_load_pd(im + 3*es);
__m512d t9 = _mm512_load_pd(re + 5*es);
__m512d t10 = _mm512_load_pd(im + 5*es);
__m512d t11 = _mm512_load_pd(re + 1*es);
__m512d t12 = _mm512_load_pd(im + 1*es);
__m512d t13 = _mm512_add_pd(t1, t7);
__m512d t14 = _mm512_add_pd(t2, t8);
__m512d t15 = _mm512_sub_pd(t1, t7);
__m512d t16 = _mm512_sub_pd(t2, t8);
__m512d t17 = _mm512_add_pd(t3, t9);
__m512d t18 = _mm512_add_pd(t4, t10);
__m512d t19 = _mm512_sub_pd(t3, t9);
__m512d t20 = _mm512_sub_pd(t4, t10);
__m512d t21 = _mm512_add_pd(t5, t11);
__m512d t22 = _mm512_add_pd(t6, t12);
__m512d t23 = _mm512_sub_pd(t5, t11);
__m512d t24 = _mm512_sub_pd(t6, t12);
__m512d t25 = _mm512_add_pd(t17, t21);
__m512d t26 = _mm512_add_pd(t18, t22);
__m512d t27 = _mm512_sub_pd(t17, t21);
__m512d t28 = _mm512_sub_pd(t18, t22);
__m512d t29 = _mm512_add_pd(t13, t25);
__m512d t30 = _mm512_add_pd(t14, t26);
__m512d t31 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t25, t13);
__m512d t32 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t26, t14);
__m512d t33 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t28, t31);
__m512d t34 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t27, t32);
__m512d t35 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t28, t31);
__m512d t36 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t27, t32);
_mm512_store_pd(re + 0*es, t29);
_mm512_store_pd(im + 0*es, t30);
_mm512_store_pd(re + 4*es, t33);
_mm512_store_pd(im + 4*es, t34);
_mm512_store_pd(re + 2*es, t35);
_mm512_store_pd(im + 2*es, t36);
__m512d t37 = _mm512_add_pd(t19, t23);
__m512d t38 = _mm512_add_pd(t20, t24);
__m512d t39 = _mm512_sub_pd(t19, t23);
__m512d t40 = _mm512_sub_pd(t20, t24);
__m512d t41 = _mm512_add_pd(t15, t37);
__m512d t42 = _mm512_add_pd(t16, t38);
__m512d t43 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t37, t15);
__m512d t44 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t38, t16);
__m512d t45 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t40, t43);
__m512d t46 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t39, t44);
__m512d t47 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t40, t43);
__m512d t48 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t39, t44);
_mm512_store_pd(re + 3*es, t41);
_mm512_store_pd(im + 3*es, t42);
_mm512_store_pd(re + 1*es, t45);
_mm512_store_pd(im + 1*es, t46);
_mm512_store_pd(re + 5*es, t47);
_mm512_store_pd(im + 5*es, t48);
}
static __attribute__((always_inline)) inline void dft8_v(double* re, double* im, long es){
__m512d t1 = _mm512_load_pd(re + 0*es);
__m512d t2 = _mm512_load_pd(im + 0*es);
__m512d t3 = _mm512_load_pd(re + 1*es);
__m512d t4 = _mm512_load_pd(im + 1*es);
__m512d t5 = _mm512_load_pd(re + 2*es);
__m512d t6 = _mm512_load_pd(im + 2*es);
__m512d t7 = _mm512_load_pd(re + 3*es);
__m512d t8 = _mm512_load_pd(im + 3*es);
__m512d t9 = _mm512_load_pd(re + 4*es);
__m512d t10 = _mm512_load_pd(im + 4*es);
__m512d t11 = _mm512_load_pd(re + 5*es);
__m512d t12 = _mm512_load_pd(im + 5*es);
__m512d t13 = _mm512_load_pd(re + 6*es);
__m512d t14 = _mm512_load_pd(im + 6*es);
__m512d t15 = _mm512_load_pd(re + 7*es);
__m512d t16 = _mm512_load_pd(im + 7*es);
__m512d t17 = _mm512_add_pd(t1, t9);
__m512d t18 = _mm512_add_pd(t2, t10);
__m512d t19 = _mm512_sub_pd(t1, t9);
__m512d t20 = _mm512_sub_pd(t2, t10);
__m512d t21 = _mm512_add_pd(t5, t13);
__m512d t22 = _mm512_add_pd(t6, t14);
__m512d t23 = _mm512_sub_pd(t5, t13);
__m512d t24 = _mm512_sub_pd(t6, t14);
__m512d t25 = _mm512_add_pd(t17, t21);
__m512d t26 = _mm512_add_pd(t18, t22);
__m512d t27 = _mm512_sub_pd(t17, t21);
__m512d t28 = _mm512_sub_pd(t18, t22);
__m512d t29 = _mm512_add_pd(t19, t24);
__m512d t30 = _mm512_sub_pd(t20, t23);
__m512d t31 = _mm512_sub_pd(t19, t24);
__m512d t32 = _mm512_add_pd(t20, t23);
__m512d t33 = _mm512_add_pd(t3, t11);
__m512d t34 = _mm512_add_pd(t4, t12);
__m512d t35 = _mm512_sub_pd(t3, t11);
__m512d t36 = _mm512_sub_pd(t4, t12);
__m512d t37 = _mm512_add_pd(t7, t15);
__m512d t38 = _mm512_add_pd(t8, t16);
__m512d t39 = _mm512_sub_pd(t7, t15);
__m512d t40 = _mm512_sub_pd(t8, t16);
__m512d t41 = _mm512_add_pd(t33, t37);
__m512d t42 = _mm512_add_pd(t34, t38);
__m512d t43 = _mm512_sub_pd(t33, t37);
__m512d t44 = _mm512_sub_pd(t34, t38);
__m512d t45 = _mm512_add_pd(t35, t40);
__m512d t46 = _mm512_sub_pd(t36, t39);
__m512d t47 = _mm512_sub_pd(t35, t40);
__m512d t48 = _mm512_add_pd(t36, t39);
__m512d t49 = _mm512_add_pd(t25, t41);
__m512d t50 = _mm512_add_pd(t26, t42);
__m512d t51 = _mm512_sub_pd(t25, t41);
__m512d t52 = _mm512_sub_pd(t26, t42);
__m512d t53 = _mm512_add_pd(t45, t46);
__m512d t54 = _mm512_mul_pd(_mm512_set1_pd(0x1.6a09e667f3bcdp-1), t53);
__m512d t55 = _mm512_sub_pd(t46, t45);
__m512d t56 = _mm512_mul_pd(_mm512_set1_pd(0x1.6a09e667f3bcdp-1), t55);
__m512d t57 = _mm512_add_pd(t29, t54);
__m512d t58 = _mm512_add_pd(t30, t56);
__m512d t59 = _mm512_sub_pd(t29, t54);
__m512d t60 = _mm512_sub_pd(t30, t56);
__m512d t61 = _mm512_add_pd(t27, t44);
__m512d t62 = _mm512_sub_pd(t28, t43);
__m512d t63 = _mm512_sub_pd(t27, t44);
__m512d t64 = _mm512_add_pd(t28, t43);
__m512d t65 = _mm512_sub_pd(t48, t47);
__m512d t66 = _mm512_mul_pd(_mm512_set1_pd(0x1.6a09e667f3bcdp-1), t65);
__m512d t67 = _mm512_add_pd(t47, t48);
__m512d t68 = _mm512_mul_pd(_mm512_set1_pd(0x1.6a09e667f3bcdp-1), t67);
__m512d t69 = _mm512_add_pd(t31, t66);
__m512d t70 = _mm512_sub_pd(t32, t68);
__m512d t71 = _mm512_sub_pd(t31, t66);
__m512d t72 = _mm512_add_pd(t32, t68);
_mm512_store_pd(re + 0*es, t49);
_mm512_store_pd(im + 0*es, t50);
_mm512_store_pd(re + 1*es, t57);
_mm512_store_pd(im + 1*es, t58);
_mm512_store_pd(re + 2*es, t61);
_mm512_store_pd(im + 2*es, t62);
_mm512_store_pd(re + 3*es, t69);
_mm512_store_pd(im + 3*es, t70);
_mm512_store_pd(re + 4*es, t51);
_mm512_store_pd(im + 4*es, t52);
_mm512_store_pd(re + 5*es, t59);
_mm512_store_pd(im + 5*es, t60);
_mm512_store_pd(re + 6*es, t63);
_mm512_store_pd(im + 6*es, t64);
_mm512_store_pd(re + 7*es, t71);
_mm512_store_pd(im + 7*es, t72);
}
static const double HT_13[72] ALIGN64 = {0x1.c55a7e00740e9p-1,0x1.dbe064267c47bp-2,0x1.22d961ea71119p-1,0x1.a55e242a4c3d2p-1,0x1.edb7debaa3ed5p-4,0x1.fc44566966769p-1,-0x1.6b1d8b2365d9ep-2,0x1.deba72ef20147p-1,-0x1.7f3ccd0032e0dp-1,0x1.5384d024c2f84p-1,-0x1.f11f493053d00p-1,0x1.ea1e54bc48dbcp-3,0x1.22d961ea71119p-1,0x1.a55e242a4c3d2p-1,-0x1.6b1d8b2365d9ep-2,0x1.deba72ef20147p-1,-0x1.f11f493053d00p-1,0x1.ea1e54bc48dbcp-3,-0x1.7f3ccd0032e0ep-1,-0x1.5384d024c2f82p-1,0x1.edb7debaa3ee3p-4,-0x1.fc44566966769p-1,0x1.c55a7e00740eap-1,-0x1.dbe064267c479p-2,0x1.edb7debaa3ed5p-4,0x1.fc44566966769p-1,-0x1.f11f493053d00p-1,0x1.ea1e54bc48dbcp-3,-0x1.6b1d8b2365da6p-2,-0x1.deba72ef20146p-1,0x1.c55a7e00740eap-1,-0x1.dbe064267c479p-2,0x1.22d961ea71119p-1,0x1.a55e242a4c3d2p-1,-0x1.7f3ccd0032e0dp-1,0x1.5384d024c2f84p-1,-0x1.6b1d8b2365d9ep-2,0x1.deba72ef20147p-1,-0x1.7f3ccd0032e0ep-1,-0x1.5384d024c2f82p-1,0x1.c55a7e00740eap-1,-0x1.dbe064267c479p-2,0x1.edb7debaa3ed5p-4,0x1.fc44566966769p-1,-0x1.f11f493053d01p-1,-0x1.ea1e54bc48db3p-3,0x1.22d961ea71110p-1,-0x1.a55e242a4c3d8p-1,-0x1.7f3ccd0032e0dp-1,0x1.5384d024c2f84p-1,0x1.edb7debaa3ee3p-4,-0x1.fc44566966769p-1,0x1.22d961ea71119p-1,0x1.a55e242a4c3d2p-1,-0x1.f11f493053d01p-1,-0x1.ea1e54bc48db3p-3,0x1.c55a7e00740eap-1,-0x1.dbe064267c479p-2,-0x1.6b1d8b2365d9ep-2,0x1.deba72ef20147p-1,-0x1.f11f493053d00p-1,0x1.ea1e54bc48dbcp-3,0x1.c55a7e00740eap-1,-0x1.dbe064267c479p-2,-0x1.7f3ccd0032e0dp-1,0x1.5384d024c2f84p-1,0x1.22d961ea71110p-1,-0x1.a55e242a4c3d8p-1,-0x1.6b1d8b2365d9ep-2,0x1.deba72ef20147p-1,0x1.edb7debaa3ee3p-4,-0x1.fc44566966769p-1};
static void dftp13_v(double* re, double* im, long es){
__m512d t1 = _mm512_load_pd(re + 0);
__m512d t2 = _mm512_load_pd(im + 0);
__m512d accAr1 = t1, accAi1 = t2, accBr1, accBi1;
__m512d accAr2 = t1, accAi2 = t2, accBr2, accBi2;
__m512d accAr3 = t1, accAi3 = t2, accBr3, accBi3;
__m512d accAr4 = t1, accAi4 = t2, accBr4, accBi4;
__m512d accAr5 = t1, accAi5 = t2, accBr5, accBi5;
__m512d accAr6 = t1, accAi6 = t2, accBr6, accBi6;
__m512d o0r = t1, o0i = t2;
__m512d t3 = _mm512_load_pd(re + 1*es);
__m512d t4 = _mm512_load_pd(im + 1*es);
__m512d t5 = _mm512_load_pd(re + 12*es);
__m512d t6 = _mm512_load_pd(im + 12*es);
__m512d t7 = _mm512_add_pd(t3, t5);
__m512d t8 = _mm512_add_pd(t4, t6);
__m512d t9 = _mm512_sub_pd(t3, t5);
__m512d t10 = _mm512_sub_pd(t4, t6);
o0r = _mm512_add_pd(o0r, t7); o0i = _mm512_add_pd(o0i, t8);
__m512d bc1_1_c, bc1_1_s;
BCASTV(bc1_1_c, HT_13[0]); BCASTV(bc1_1_s, HT_13[1]);
accAr1 = _mm512_fmadd_pd(bc1_1_c, t7, accAr1);
accAi1 = _mm512_fmadd_pd(bc1_1_c, t8, accAi1);
accBr1 = _mm512_mul_pd(bc1_1_s, t10);
accBi1 = _mm512_mul_pd(bc1_1_s, t9);
__m512d bc1_2_c, bc1_2_s;
BCASTV(bc1_2_c, HT_13[2]); BCASTV(bc1_2_s, HT_13[3]);
accAr2 = _mm512_fmadd_pd(bc1_2_c, t7, accAr2);
accAi2 = _mm512_fmadd_pd(bc1_2_c, t8, accAi2);
accBr2 = _mm512_mul_pd(bc1_2_s, t10);
accBi2 = _mm512_mul_pd(bc1_2_s, t9);
__m512d bc1_3_c, bc1_3_s;
BCASTV(bc1_3_c, HT_13[4]); BCASTV(bc1_3_s, HT_13[5]);
accAr3 = _mm512_fmadd_pd(bc1_3_c, t7, accAr3);
accAi3 = _mm512_fmadd_pd(bc1_3_c, t8, accAi3);
accBr3 = _mm512_mul_pd(bc1_3_s, t10);
accBi3 = _mm512_mul_pd(bc1_3_s, t9);
__m512d bc1_4_c, bc1_4_s;
BCASTV(bc1_4_c, HT_13[6]); BCASTV(bc1_4_s, HT_13[7]);
accAr4 = _mm512_fmadd_pd(bc1_4_c, t7, accAr4);
accAi4 = _mm512_fmadd_pd(bc1_4_c, t8, accAi4);
accBr4 = _mm512_mul_pd(bc1_4_s, t10);
accBi4 = _mm512_mul_pd(bc1_4_s, t9);
__m512d bc1_5_c, bc1_5_s;
BCASTV(bc1_5_c, HT_13[8]); BCASTV(bc1_5_s, HT_13[9]);
accAr5 = _mm512_fmadd_pd(bc1_5_c, t7, accAr5);
accAi5 = _mm512_fmadd_pd(bc1_5_c, t8, accAi5);
accBr5 = _mm512_mul_pd(bc1_5_s, t10);
accBi5 = _mm512_mul_pd(bc1_5_s, t9);
__m512d bc1_6_c, bc1_6_s;
BCASTV(bc1_6_c, HT_13[10]); BCASTV(bc1_6_s, HT_13[11]);
accAr6 = _mm512_fmadd_pd(bc1_6_c, t7, accAr6);
accAi6 = _mm512_fmadd_pd(bc1_6_c, t8, accAi6);
accBr6 = _mm512_mul_pd(bc1_6_s, t10);
accBi6 = _mm512_mul_pd(bc1_6_s, t9);
__m512d t11 = _mm512_load_pd(re + 2*es);
__m512d t12 = _mm512_load_pd(im + 2*es);
__m512d t13 = _mm512_load_pd(re + 11*es);
__m512d t14 = _mm512_load_pd(im + 11*es);
__m512d t15 = _mm512_add_pd(t11, t13);
__m512d t16 = _mm512_add_pd(t12, t14);
__m512d t17 = _mm512_sub_pd(t11, t13);
__m512d t18 = _mm512_sub_pd(t12, t14);
o0r = _mm512_add_pd(o0r, t15); o0i = _mm512_add_pd(o0i, t16);
__m512d bc2_1_c, bc2_1_s;
BCASTV(bc2_1_c, HT_13[12]); BCASTV(bc2_1_s, HT_13[13]);
accAr1 = _mm512_fmadd_pd(bc2_1_c, t15, accAr1);
accAi1 = _mm512_fmadd_pd(bc2_1_c, t16, accAi1);
accBr1 = _mm512_fmadd_pd(bc2_1_s, t18, accBr1);
accBi1 = _mm512_fmadd_pd(bc2_1_s, t17, accBi1);
__m512d bc2_2_c, bc2_2_s;
BCASTV(bc2_2_c, HT_13[14]); BCASTV(bc2_2_s, HT_13[15]);
accAr2 = _mm512_fmadd_pd(bc2_2_c, t15, accAr2);
accAi2 = _mm512_fmadd_pd(bc2_2_c, t16, accAi2);
accBr2 = _mm512_fmadd_pd(bc2_2_s, t18, accBr2);
accBi2 = _mm512_fmadd_pd(bc2_2_s, t17, accBi2);
__m512d bc2_3_c, bc2_3_s;
BCASTV(bc2_3_c, HT_13[16]); BCASTV(bc2_3_s, HT_13[17]);
accAr3 = _mm512_fmadd_pd(bc2_3_c, t15, accAr3);
accAi3 = _mm512_fmadd_pd(bc2_3_c, t16, accAi3);
accBr3 = _mm512_fmadd_pd(bc2_3_s, t18, accBr3);
accBi3 = _mm512_fmadd_pd(bc2_3_s, t17, accBi3);
__m512d bc2_4_c, bc2_4_s;
BCASTV(bc2_4_c, HT_13[18]); BCASTV(bc2_4_s, HT_13[19]);
accAr4 = _mm512_fmadd_pd(bc2_4_c, t15, accAr4);
accAi4 = _mm512_fmadd_pd(bc2_4_c, t16, accAi4);
accBr4 = _mm512_fmadd_pd(bc2_4_s, t18, accBr4);
accBi4 = _mm512_fmadd_pd(bc2_4_s, t17, accBi4);
__m512d bc2_5_c, bc2_5_s;
BCASTV(bc2_5_c, HT_13[20]); BCASTV(bc2_5_s, HT_13[21]);
accAr5 = _mm512_fmadd_pd(bc2_5_c, t15, accAr5);
accAi5 = _mm512_fmadd_pd(bc2_5_c, t16, accAi5);
accBr5 = _mm512_fmadd_pd(bc2_5_s, t18, accBr5);
accBi5 = _mm512_fmadd_pd(bc2_5_s, t17, accBi5);
__m512d bc2_6_c, bc2_6_s;
BCASTV(bc2_6_c, HT_13[22]); BCASTV(bc2_6_s, HT_13[23]);
accAr6 = _mm512_fmadd_pd(bc2_6_c, t15, accAr6);
accAi6 = _mm512_fmadd_pd(bc2_6_c, t16, accAi6);
accBr6 = _mm512_fmadd_pd(bc2_6_s, t18, accBr6);
accBi6 = _mm512_fmadd_pd(bc2_6_s, t17, accBi6);
__m512d t19 = _mm512_load_pd(re + 3*es);
__m512d t20 = _mm512_load_pd(im + 3*es);
__m512d t21 = _mm512_load_pd(re + 10*es);
__m512d t22 = _mm512_load_pd(im + 10*es);
__m512d t23 = _mm512_add_pd(t19, t21);
__m512d t24 = _mm512_add_pd(t20, t22);
__m512d t25 = _mm512_sub_pd(t19, t21);
__m512d t26 = _mm512_sub_pd(t20, t22);
o0r = _mm512_add_pd(o0r, t23); o0i = _mm512_add_pd(o0i, t24);
__m512d bc3_1_c, bc3_1_s;
BCASTV(bc3_1_c, HT_13[24]); BCASTV(bc3_1_s, HT_13[25]);
accAr1 = _mm512_fmadd_pd(bc3_1_c, t23, accAr1);
accAi1 = _mm512_fmadd_pd(bc3_1_c, t24, accAi1);
accBr1 = _mm512_fmadd_pd(bc3_1_s, t26, accBr1);
accBi1 = _mm512_fmadd_pd(bc3_1_s, t25, accBi1);
__m512d bc3_2_c, bc3_2_s;
BCASTV(bc3_2_c, HT_13[26]); BCASTV(bc3_2_s, HT_13[27]);
accAr2 = _mm512_fmadd_pd(bc3_2_c, t23, accAr2);
accAi2 = _mm512_fmadd_pd(bc3_2_c, t24, accAi2);
accBr2 = _mm512_fmadd_pd(bc3_2_s, t26, accBr2);
accBi2 = _mm512_fmadd_pd(bc3_2_s, t25, accBi2);
__m512d bc3_3_c, bc3_3_s;
BCASTV(bc3_3_c, HT_13[28]); BCASTV(bc3_3_s, HT_13[29]);
accAr3 = _mm512_fmadd_pd(bc3_3_c, t23, accAr3);
accAi3 = _mm512_fmadd_pd(bc3_3_c, t24, accAi3);
accBr3 = _mm512_fmadd_pd(bc3_3_s, t26, accBr3);
accBi3 = _mm512_fmadd_pd(bc3_3_s, t25, accBi3);
__m512d bc3_4_c, bc3_4_s;
BCASTV(bc3_4_c, HT_13[30]); BCASTV(bc3_4_s, HT_13[31]);
accAr4 = _mm512_fmadd_pd(bc3_4_c, t23, accAr4);
accAi4 = _mm512_fmadd_pd(bc3_4_c, t24, accAi4);
accBr4 = _mm512_fmadd_pd(bc3_4_s, t26, accBr4);
accBi4 = _mm512_fmadd_pd(bc3_4_s, t25, accBi4);
__m512d bc3_5_c, bc3_5_s;
BCASTV(bc3_5_c, HT_13[32]); BCASTV(bc3_5_s, HT_13[33]);
accAr5 = _mm512_fmadd_pd(bc3_5_c, t23, accAr5);
accAi5 = _mm512_fmadd_pd(bc3_5_c, t24, accAi5);
accBr5 = _mm512_fmadd_pd(bc3_5_s, t26, accBr5);
accBi5 = _mm512_fmadd_pd(bc3_5_s, t25, accBi5);
__m512d bc3_6_c, bc3_6_s;
BCASTV(bc3_6_c, HT_13[34]); BCASTV(bc3_6_s, HT_13[35]);
accAr6 = _mm512_fmadd_pd(bc3_6_c, t23, accAr6);
accAi6 = _mm512_fmadd_pd(bc3_6_c, t24, accAi6);
accBr6 = _mm512_fmadd_pd(bc3_6_s, t26, accBr6);
accBi6 = _mm512_fmadd_pd(bc3_6_s, t25, accBi6);
__m512d t27 = _mm512_load_pd(re + 4*es);
__m512d t28 = _mm512_load_pd(im + 4*es);
__m512d t29 = _mm512_load_pd(re + 9*es);
__m512d t30 = _mm512_load_pd(im + 9*es);
__m512d t31 = _mm512_add_pd(t27, t29);
__m512d t32 = _mm512_add_pd(t28, t30);
__m512d t33 = _mm512_sub_pd(t27, t29);
__m512d t34 = _mm512_sub_pd(t28, t30);
o0r = _mm512_add_pd(o0r, t31); o0i = _mm512_add_pd(o0i, t32);
__m512d bc4_1_c, bc4_1_s;
BCASTV(bc4_1_c, HT_13[36]); BCASTV(bc4_1_s, HT_13[37]);
accAr1 = _mm512_fmadd_pd(bc4_1_c, t31, accAr1);
accAi1 = _mm512_fmadd_pd(bc4_1_c, t32, accAi1);
accBr1 = _mm512_fmadd_pd(bc4_1_s, t34, accBr1);
accBi1 = _mm512_fmadd_pd(bc4_1_s, t33, accBi1);
__m512d bc4_2_c, bc4_2_s;
BCASTV(bc4_2_c, HT_13[38]); BCASTV(bc4_2_s, HT_13[39]);
accAr2 = _mm512_fmadd_pd(bc4_2_c, t31, accAr2);
accAi2 = _mm512_fmadd_pd(bc4_2_c, t32, accAi2);
accBr2 = _mm512_fmadd_pd(bc4_2_s, t34, accBr2);
accBi2 = _mm512_fmadd_pd(bc4_2_s, t33, accBi2);
__m512d bc4_3_c, bc4_3_s;
BCASTV(bc4_3_c, HT_13[40]); BCASTV(bc4_3_s, HT_13[41]);
accAr3 = _mm512_fmadd_pd(bc4_3_c, t31, accAr3);
accAi3 = _mm512_fmadd_pd(bc4_3_c, t32, accAi3);
accBr3 = _mm512_fmadd_pd(bc4_3_s, t34, accBr3);
accBi3 = _mm512_fmadd_pd(bc4_3_s, t33, accBi3);
__m512d bc4_4_c, bc4_4_s;
BCASTV(bc4_4_c, HT_13[42]); BCASTV(bc4_4_s, HT_13[43]);
accAr4 = _mm512_fmadd_pd(bc4_4_c, t31, accAr4);
accAi4 = _mm512_fmadd_pd(bc4_4_c, t32, accAi4);
accBr4 = _mm512_fmadd_pd(bc4_4_s, t34, accBr4);
accBi4 = _mm512_fmadd_pd(bc4_4_s, t33, accBi4);
__m512d bc4_5_c, bc4_5_s;
BCASTV(bc4_5_c, HT_13[44]); BCASTV(bc4_5_s, HT_13[45]);
accAr5 = _mm512_fmadd_pd(bc4_5_c, t31, accAr5);
accAi5 = _mm512_fmadd_pd(bc4_5_c, t32, accAi5);
accBr5 = _mm512_fmadd_pd(bc4_5_s, t34, accBr5);
accBi5 = _mm512_fmadd_pd(bc4_5_s, t33, accBi5);
__m512d bc4_6_c, bc4_6_s;
BCASTV(bc4_6_c, HT_13[46]); BCASTV(bc4_6_s, HT_13[47]);
accAr6 = _mm512_fmadd_pd(bc4_6_c, t31, accAr6);
accAi6 = _mm512_fmadd_pd(bc4_6_c, t32, accAi6);
accBr6 = _mm512_fmadd_pd(bc4_6_s, t34, accBr6);
accBi6 = _mm512_fmadd_pd(bc4_6_s, t33, accBi6);
__m512d t35 = _mm512_load_pd(re + 5*es);
__m512d t36 = _mm512_load_pd(im + 5*es);
__m512d t37 = _mm512_load_pd(re + 8*es);
__m512d t38 = _mm512_load_pd(im + 8*es);
__m512d t39 = _mm512_add_pd(t35, t37);
__m512d t40 = _mm512_add_pd(t36, t38);
__m512d t41 = _mm512_sub_pd(t35, t37);
__m512d t42 = _mm512_sub_pd(t36, t38);
o0r = _mm512_add_pd(o0r, t39); o0i = _mm512_add_pd(o0i, t40);
__m512d bc5_1_c, bc5_1_s;
BCASTV(bc5_1_c, HT_13[48]); BCASTV(bc5_1_s, HT_13[49]);
accAr1 = _mm512_fmadd_pd(bc5_1_c, t39, accAr1);
accAi1 = _mm512_fmadd_pd(bc5_1_c, t40, accAi1);
accBr1 = _mm512_fmadd_pd(bc5_1_s, t42, accBr1);
accBi1 = _mm512_fmadd_pd(bc5_1_s, t41, accBi1);
__m512d bc5_2_c, bc5_2_s;
BCASTV(bc5_2_c, HT_13[50]); BCASTV(bc5_2_s, HT_13[51]);
accAr2 = _mm512_fmadd_pd(bc5_2_c, t39, accAr2);
accAi2 = _mm512_fmadd_pd(bc5_2_c, t40, accAi2);
accBr2 = _mm512_fmadd_pd(bc5_2_s, t42, accBr2);
accBi2 = _mm512_fmadd_pd(bc5_2_s, t41, accBi2);
__m512d bc5_3_c, bc5_3_s;
BCASTV(bc5_3_c, HT_13[52]); BCASTV(bc5_3_s, HT_13[53]);
accAr3 = _mm512_fmadd_pd(bc5_3_c, t39, accAr3);
accAi3 = _mm512_fmadd_pd(bc5_3_c, t40, accAi3);
accBr3 = _mm512_fmadd_pd(bc5_3_s, t42, accBr3);
accBi3 = _mm512_fmadd_pd(bc5_3_s, t41, accBi3);
__m512d bc5_4_c, bc5_4_s;
BCASTV(bc5_4_c, HT_13[54]); BCASTV(bc5_4_s, HT_13[55]);
accAr4 = _mm512_fmadd_pd(bc5_4_c, t39, accAr4);
accAi4 = _mm512_fmadd_pd(bc5_4_c, t40, accAi4);
accBr4 = _mm512_fmadd_pd(bc5_4_s, t42, accBr4);
accBi4 = _mm512_fmadd_pd(bc5_4_s, t41, accBi4);
__m512d bc5_5_c, bc5_5_s;
BCASTV(bc5_5_c, HT_13[56]); BCASTV(bc5_5_s, HT_13[57]);
accAr5 = _mm512_fmadd_pd(bc5_5_c, t39, accAr5);
accAi5 = _mm512_fmadd_pd(bc5_5_c, t40, accAi5);
accBr5 = _mm512_fmadd_pd(bc5_5_s, t42, accBr5);
accBi5 = _mm512_fmadd_pd(bc5_5_s, t41, accBi5);
__m512d bc5_6_c, bc5_6_s;
BCASTV(bc5_6_c, HT_13[58]); BCASTV(bc5_6_s, HT_13[59]);
accAr6 = _mm512_fmadd_pd(bc5_6_c, t39, accAr6);
accAi6 = _mm512_fmadd_pd(bc5_6_c, t40, accAi6);
accBr6 = _mm512_fmadd_pd(bc5_6_s, t42, accBr6);
accBi6 = _mm512_fmadd_pd(bc5_6_s, t41, accBi6);
__m512d t43 = _mm512_load_pd(re + 6*es);
__m512d t44 = _mm512_load_pd(im + 6*es);
__m512d t45 = _mm512_load_pd(re + 7*es);
__m512d t46 = _mm512_load_pd(im + 7*es);
__m512d t47 = _mm512_add_pd(t43, t45);
__m512d t48 = _mm512_add_pd(t44, t46);
__m512d t49 = _mm512_sub_pd(t43, t45);
__m512d t50 = _mm512_sub_pd(t44, t46);
o0r = _mm512_add_pd(o0r, t47); o0i = _mm512_add_pd(o0i, t48);
__m512d bc6_1_c, bc6_1_s;
BCASTV(bc6_1_c, HT_13[60]); BCASTV(bc6_1_s, HT_13[61]);
accAr1 = _mm512_fmadd_pd(bc6_1_c, t47, accAr1);
accAi1 = _mm512_fmadd_pd(bc6_1_c, t48, accAi1);
accBr1 = _mm512_fmadd_pd(bc6_1_s, t50, accBr1);
accBi1 = _mm512_fmadd_pd(bc6_1_s, t49, accBi1);
__m512d bc6_2_c, bc6_2_s;
BCASTV(bc6_2_c, HT_13[62]); BCASTV(bc6_2_s, HT_13[63]);
accAr2 = _mm512_fmadd_pd(bc6_2_c, t47, accAr2);
accAi2 = _mm512_fmadd_pd(bc6_2_c, t48, accAi2);
accBr2 = _mm512_fmadd_pd(bc6_2_s, t50, accBr2);
accBi2 = _mm512_fmadd_pd(bc6_2_s, t49, accBi2);
__m512d bc6_3_c, bc6_3_s;
BCASTV(bc6_3_c, HT_13[64]); BCASTV(bc6_3_s, HT_13[65]);
accAr3 = _mm512_fmadd_pd(bc6_3_c, t47, accAr3);
accAi3 = _mm512_fmadd_pd(bc6_3_c, t48, accAi3);
accBr3 = _mm512_fmadd_pd(bc6_3_s, t50, accBr3);
accBi3 = _mm512_fmadd_pd(bc6_3_s, t49, accBi3);
__m512d bc6_4_c, bc6_4_s;
BCASTV(bc6_4_c, HT_13[66]); BCASTV(bc6_4_s, HT_13[67]);
accAr4 = _mm512_fmadd_pd(bc6_4_c, t47, accAr4);
accAi4 = _mm512_fmadd_pd(bc6_4_c, t48, accAi4);
accBr4 = _mm512_fmadd_pd(bc6_4_s, t50, accBr4);
accBi4 = _mm512_fmadd_pd(bc6_4_s, t49, accBi4);
__m512d bc6_5_c, bc6_5_s;
BCASTV(bc6_5_c, HT_13[68]); BCASTV(bc6_5_s, HT_13[69]);
accAr5 = _mm512_fmadd_pd(bc6_5_c, t47, accAr5);
accAi5 = _mm512_fmadd_pd(bc6_5_c, t48, accAi5);
accBr5 = _mm512_fmadd_pd(bc6_5_s, t50, accBr5);
accBi5 = _mm512_fmadd_pd(bc6_5_s, t49, accBi5);
__m512d bc6_6_c, bc6_6_s;
BCASTV(bc6_6_c, HT_13[70]); BCASTV(bc6_6_s, HT_13[71]);
accAr6 = _mm512_fmadd_pd(bc6_6_c, t47, accAr6);
accAi6 = _mm512_fmadd_pd(bc6_6_c, t48, accAi6);
accBr6 = _mm512_fmadd_pd(bc6_6_s, t50, accBr6);
accBi6 = _mm512_fmadd_pd(bc6_6_s, t49, accBi6);
_mm512_store_pd(re, o0r); _mm512_store_pd(im, o0i);
__m512d t51 = _mm512_add_pd(accAr1, accBr1);
__m512d t52 = _mm512_sub_pd(accAi1, accBi1);
__m512d t53 = _mm512_sub_pd(accAr1, accBr1);
__m512d t54 = _mm512_add_pd(accAi1, accBi1);
_mm512_store_pd(re + 1*es, t51);
_mm512_store_pd(im + 1*es, t52);
_mm512_store_pd(re + 12*es, t53);
_mm512_store_pd(im + 12*es, t54);
__m512d t55 = _mm512_add_pd(accAr2, accBr2);
__m512d t56 = _mm512_sub_pd(accAi2, accBi2);
__m512d t57 = _mm512_sub_pd(accAr2, accBr2);
__m512d t58 = _mm512_add_pd(accAi2, accBi2);
_mm512_store_pd(re + 2*es, t55);
_mm512_store_pd(im + 2*es, t56);
_mm512_store_pd(re + 11*es, t57);
_mm512_store_pd(im + 11*es, t58);
__m512d t59 = _mm512_add_pd(accAr3, accBr3);
__m512d t60 = _mm512_sub_pd(accAi3, accBi3);
__m512d t61 = _mm512_sub_pd(accAr3, accBr3);
__m512d t62 = _mm512_add_pd(accAi3, accBi3);
_mm512_store_pd(re + 3*es, t59);
_mm512_store_pd(im + 3*es, t60);
_mm512_store_pd(re + 10*es, t61);
_mm512_store_pd(im + 10*es, t62);
__m512d t63 = _mm512_add_pd(accAr4, accBr4);
__m512d t64 = _mm512_sub_pd(accAi4, accBi4);
__m512d t65 = _mm512_sub_pd(accAr4, accBr4);
__m512d t66 = _mm512_add_pd(accAi4, accBi4);
_mm512_store_pd(re + 4*es, t63);
_mm512_store_pd(im + 4*es, t64);
_mm512_store_pd(re + 9*es, t65);
_mm512_store_pd(im + 9*es, t66);
__m512d t67 = _mm512_add_pd(accAr5, accBr5);
__m512d t68 = _mm512_sub_pd(accAi5, accBi5);
__m512d t69 = _mm512_sub_pd(accAr5, accBr5);
__m512d t70 = _mm512_add_pd(accAi5, accBi5);
_mm512_store_pd(re + 5*es, t67);
_mm512_store_pd(im + 5*es, t68);
_mm512_store_pd(re + 8*es, t69);
_mm512_store_pd(im + 8*es, t70);
__m512d t71 = _mm512_add_pd(accAr6, accBr6);
__m512d t72 = _mm512_sub_pd(accAi6, accBi6);
__m512d t73 = _mm512_sub_pd(accAr6, accBr6);
__m512d t74 = _mm512_add_pd(accAi6, accBi6);
_mm512_store_pd(re + 6*es, t71);
_mm512_store_pd(im + 6*es, t72);
_mm512_store_pd(re + 7*es, t73);
_mm512_store_pd(im + 7*es, t74);
}
static const double HT_17[128] ALIGN64 = {0x1.dd6d000370991p-1,0x1.71e955d8e7cdcp-2,0x1.7a5f6075d4884p-1,0x1.58eea2a9d6da3p-1,0x1.c86fa2b2883cep-2,0x1.ca52d7c9e640bp-1,0x1.79ee63259b75fp-4,0x1.fdd0deb564b22p-1,0x1.7a5f6075d4884p-1,0x1.58eea2a9d6da3p-1,0x1.79ee63259b75fp-4,0x1.fdd0deb564b22p-1,-0x1.348c86ed5f1bap-1,0x1.9895b6c9a05f7p-1,-0x1.f7484007faef3p-1,0x1.7851aacd6c6b5p-3,0x1.c86fa2b2883cep-2,0x1.ca52d7c9e640bp-1,-0x1.348c86ed5f1bap-1,0x1.9895b6c9a05f7p-1,-0x1.f7484007faef3p-1,-0x1.7851aacd6c6acp-3,-0x1.183b1c61f0d05p-2,-0x1.ec746923c349fp-1,0x1.79ee63259b75fp-4,0x1.fdd0deb564b22p-1,-0x1.f7484007faef3p-1,0x1.7851aacd6c6b5p-3,-0x1.183b1c61f0d05p-2,-0x1.ec746923c349fp-1,0x1.dd6d000370991p-1,-0x1.71e955d8e7cdep-2,-0x1.183b1c61f0d01p-2,0x1.ec746923c349fp-1,-0x1.b34fa910ea3b9p-1,-0x1.0d8884363dd80p-1,0x1.7a5f6075d487fp-1,-0x1.58eea2a9d6da8p-1,0x1.c86fa2b2883cep-2,0x1.ca52d7c9e640bp-1,-0x1.348c86ed5f1bap-1,0x1.9895b6c9a05f7p-1,-0x1.183b1c61f0d05p-2,-0x1.ec746923c349fp-1,0x1.dd6d000370991p-1,0x1.71e955d8e7cdcp-2,-0x1.b34fa910ea3b8p-1,0x1.0d8884363dd82p-1,-0x1.b34fa910ea3b8p-1,0x1.0d8884363dd82p-1,0x1.c86fa2b2883c3p-2,-0x1.ca52d7c9e640dp-1,0x1.79ee63259b75fp-4,0x1.fdd0deb564b22p-1,-0x1.348c86ed5f1c2p-1,-0x1.9895b6c9a05f1p-1,-0x1.f7484007faef3p-1,0x1.7851aacd6c6b5p-3,0x1.dd6d000370991p-1,-0x1.71e955d8e7cdep-2,-0x1.b34fa910ea3b8p-1,0x1.0d8884363dd82p-1,0x1.7a5f6075d487fp-1,-0x1.58eea2a9d6da8p-1,-0x1.183b1c61f0d01p-2,0x1.ec746923c349fp-1,-0x1.348c86ed5f1bap-1,0x1.9895b6c9a05f7p-1,-0x1.b34fa910ea3b8p-1,0x1.0d8884363dd82p-1,-0x1.f7484007faef3p-1,0x1.7851aacd6c6b5p-3,-0x1.b34fa910ea3b9p-1,-0x1.0d8884363dd80p-1,-0x1.183b1c61f0d05p-2,-0x1.ec746923c349fp-1,0x1.c86fa2b2883c3p-2,-0x1.ca52d7c9e640dp-1,0x1.dd6d000370991p-1,-0x1.71e955d8e7cdep-2,0x1.7a5f6075d487fp-1,-0x1.58eea2a9d6da8p-1,0x1.dd6d000370991p-1,0x1.71e955d8e7cdcp-2,0x1.79ee63259b75fp-4,0x1.fdd0deb564b22p-1,-0x1.b34fa910ea3b8p-1,0x1.0d8884363dd82p-1,0x1.c86fa2b2883cep-2,0x1.ca52d7c9e640bp-1,-0x1.b34fa910ea3b8p-1,0x1.0d8884363dd82p-1,-0x1.348c86ed5f1c2p-1,-0x1.9895b6c9a05f1p-1,0x1.7a5f6075d487fp-1,-0x1.58eea2a9d6da8p-1,-0x1.f7484007faef3p-1,0x1.7851aacd6c6b5p-3,0x1.79ee63259b77dp-4,-0x1.fdd0deb564b22p-1,0x1.dd6d000370991p-1,0x1.71e955d8e7cdcp-2,-0x1.348c86ed5f1bap-1,0x1.9895b6c9a05f7p-1,0x1.79ee63259b77dp-4,-0x1.fdd0deb564b22p-1,0x1.7a5f6075d4884p-1,0x1.58eea2a9d6da3p-1,-0x1.f7484007faef3p-1,0x1.7851aacd6c6b5p-3,0x1.c86fa2b2883c3p-2,-0x1.ca52d7c9e640dp-1,0x1.dd6d000370991p-1,0x1.71e955d8e7cdcp-2,-0x1.f7484007faef3p-1,0x1.7851aacd6c6b5p-3,0x1.7a5f6075d487fp-1,-0x1.58eea2a9d6da8p-1,-0x1.183b1c61f0d01p-2,0x1.ec746923c349fp-1,-0x1.348c86ed5f1bap-1,0x1.9895b6c9a05f7p-1,0x1.c86fa2b2883c3p-2,-0x1.ca52d7c9e640dp-1,-0x1.183b1c61f0d01p-2,0x1.ec746923c349fp-1,0x1.79ee63259b77dp-4,-0x1.fdd0deb564b22p-1};
static double HS_17[8][32] ALIGN64;
static void dftp17_v(double* re, double* im, long es){
__m512d t1 = _mm512_load_pd(re + 0);
__m512d t2 = _mm512_load_pd(im + 0);
__m512d t3 = _mm512_load_pd(re + 1*es);
__m512d t4 = _mm512_load_pd(im + 1*es);
__m512d t5 = _mm512_load_pd(re + 16*es);
__m512d t6 = _mm512_load_pd(im + 16*es);
__m512d t7 = _mm512_add_pd(t3, t5);
__m512d t8 = _mm512_add_pd(t4, t6);
__m512d t9 = _mm512_sub_pd(t3, t5);
__m512d t10 = _mm512_sub_pd(t4, t6);
_mm512_store_pd(HS_17[0], t7); _mm512_store_pd(HS_17[0]+8, t8);
_mm512_store_pd(HS_17[0]+16, t9); _mm512_store_pd(HS_17[0]+24, t10);
__m512d t11 = _mm512_load_pd(re + 2*es);
__m512d t12 = _mm512_load_pd(im + 2*es);
__m512d t13 = _mm512_load_pd(re + 15*es);
__m512d t14 = _mm512_load_pd(im + 15*es);
__m512d t15 = _mm512_add_pd(t11, t13);
__m512d t16 = _mm512_add_pd(t12, t14);
__m512d t17 = _mm512_sub_pd(t11, t13);
__m512d t18 = _mm512_sub_pd(t12, t14);
_mm512_store_pd(HS_17[1], t15); _mm512_store_pd(HS_17[1]+8, t16);
_mm512_store_pd(HS_17[1]+16, t17); _mm512_store_pd(HS_17[1]+24, t18);
__m512d t19 = _mm512_add_pd(t7, t15);
__m512d t20 = _mm512_add_pd(t8, t16);
__m512d t21 = _mm512_load_pd(re + 3*es);
__m512d t22 = _mm512_load_pd(im + 3*es);
__m512d t23 = _mm512_load_pd(re + 14*es);
__m512d t24 = _mm512_load_pd(im + 14*es);
__m512d t25 = _mm512_add_pd(t21, t23);
__m512d t26 = _mm512_add_pd(t22, t24);
__m512d t27 = _mm512_sub_pd(t21, t23);
__m512d t28 = _mm512_sub_pd(t22, t24);
_mm512_store_pd(HS_17[2], t25); _mm512_store_pd(HS_17[2]+8, t26);
_mm512_store_pd(HS_17[2]+16, t27); _mm512_store_pd(HS_17[2]+24, t28);
__m512d t29 = _mm512_add_pd(t19, t25);
__m512d t30 = _mm512_add_pd(t20, t26);
__m512d t31 = _mm512_load_pd(re + 4*es);
__m512d t32 = _mm512_load_pd(im + 4*es);
__m512d t33 = _mm512_load_pd(re + 13*es);
__m512d t34 = _mm512_load_pd(im + 13*es);
__m512d t35 = _mm512_add_pd(t31, t33);
__m512d t36 = _mm512_add_pd(t32, t34);
__m512d t37 = _mm512_sub_pd(t31, t33);
__m512d t38 = _mm512_sub_pd(t32, t34);
_mm512_store_pd(HS_17[3], t35); _mm512_store_pd(HS_17[3]+8, t36);
_mm512_store_pd(HS_17[3]+16, t37); _mm512_store_pd(HS_17[3]+24, t38);
__m512d t39 = _mm512_add_pd(t29, t35);
__m512d t40 = _mm512_add_pd(t30, t36);
__m512d t41 = _mm512_load_pd(re + 5*es);
__m512d t42 = _mm512_load_pd(im + 5*es);
__m512d t43 = _mm512_load_pd(re + 12*es);
__m512d t44 = _mm512_load_pd(im + 12*es);
__m512d t45 = _mm512_add_pd(t41, t43);
__m512d t46 = _mm512_add_pd(t42, t44);
__m512d t47 = _mm512_sub_pd(t41, t43);
__m512d t48 = _mm512_sub_pd(t42, t44);
_mm512_store_pd(HS_17[4], t45); _mm512_store_pd(HS_17[4]+8, t46);
_mm512_store_pd(HS_17[4]+16, t47); _mm512_store_pd(HS_17[4]+24, t48);
__m512d t49 = _mm512_add_pd(t39, t45);
__m512d t50 = _mm512_add_pd(t40, t46);
__m512d t51 = _mm512_load_pd(re + 6*es);
__m512d t52 = _mm512_load_pd(im + 6*es);
__m512d t53 = _mm512_load_pd(re + 11*es);
__m512d t54 = _mm512_load_pd(im + 11*es);
__m512d t55 = _mm512_add_pd(t51, t53);
__m512d t56 = _mm512_add_pd(t52, t54);
__m512d t57 = _mm512_sub_pd(t51, t53);
__m512d t58 = _mm512_sub_pd(t52, t54);
_mm512_store_pd(HS_17[5], t55); _mm512_store_pd(HS_17[5]+8, t56);
_mm512_store_pd(HS_17[5]+16, t57); _mm512_store_pd(HS_17[5]+24, t58);
__m512d t59 = _mm512_add_pd(t49, t55);
__m512d t60 = _mm512_add_pd(t50, t56);
__m512d t61 = _mm512_load_pd(re + 7*es);
__m512d t62 = _mm512_load_pd(im + 7*es);
__m512d t63 = _mm512_load_pd(re + 10*es);
__m512d t64 = _mm512_load_pd(im + 10*es);
__m512d t65 = _mm512_add_pd(t61, t63);
__m512d t66 = _mm512_add_pd(t62, t64);
__m512d t67 = _mm512_sub_pd(t61, t63);
__m512d t68 = _mm512_sub_pd(t62, t64);
_mm512_store_pd(HS_17[6], t65); _mm512_store_pd(HS_17[6]+8, t66);
_mm512_store_pd(HS_17[6]+16, t67); _mm512_store_pd(HS_17[6]+24, t68);
__m512d t69 = _mm512_add_pd(t59, t65);
__m512d t70 = _mm512_add_pd(t60, t66);
__m512d t71 = _mm512_load_pd(re + 8*es);
__m512d t72 = _mm512_load_pd(im + 8*es);
__m512d t73 = _mm512_load_pd(re + 9*es);
__m512d t74 = _mm512_load_pd(im + 9*es);
__m512d t75 = _mm512_add_pd(t71, t73);
__m512d t76 = _mm512_add_pd(t72, t74);
__m512d t77 = _mm512_sub_pd(t71, t73);
__m512d t78 = _mm512_sub_pd(t72, t74);
_mm512_store_pd(HS_17[7], t75); _mm512_store_pd(HS_17[7]+8, t76);
_mm512_store_pd(HS_17[7]+16, t77); _mm512_store_pd(HS_17[7]+24, t78);
__m512d t79 = _mm512_add_pd(t69, t75);
__m512d t80 = _mm512_add_pd(t70, t76);
__m512d t81 = _mm512_add_pd(t79, t1);
__m512d t82 = _mm512_add_pd(t80, t2);
_mm512_store_pd(re + 0, t81);
_mm512_store_pd(im + 0, t82);
__asm__ volatile("" ::: "memory");
__asm__ volatile("" ::: "memory");
__m512d accAr1 = t1, accAi1 = t2, accBr1, accBi1;
__m512d accAr2 = t1, accAi2 = t2, accBr2, accBi2;
__m512d accAr3 = t1, accAi3 = t2, accBr3, accBi3;
__m512d accAr4 = t1, accAi4 = t2, accBr4, accBi4;
__m512d t83 = _mm512_load_pd(HS_17[0]);
__m512d t84 = _mm512_load_pd(HS_17[0]+8);
__m512d t85 = _mm512_load_pd(HS_17[0]+16);
__m512d t86 = _mm512_load_pd(HS_17[0]+24);
__m512d bc1_1_c, bc1_1_s;
BCASTV(bc1_1_c, HT_17[0]); BCASTV(bc1_1_s, HT_17[1]);
accAr1 = _mm512_fmadd_pd(bc1_1_c, t83, accAr1);
accAi1 = _mm512_fmadd_pd(bc1_1_c, t84, accAi1);
accBr1 = _mm512_mul_pd(bc1_1_s, t86);
accBi1 = _mm512_mul_pd(bc1_1_s, t85);
__m512d bc1_2_c, bc1_2_s;
BCASTV(bc1_2_c, HT_17[2]); BCASTV(bc1_2_s, HT_17[3]);
accAr2 = _mm512_fmadd_pd(bc1_2_c, t83, accAr2);
accAi2 = _mm512_fmadd_pd(bc1_2_c, t84, accAi2);
accBr2 = _mm512_mul_pd(bc1_2_s, t86);
accBi2 = _mm512_mul_pd(bc1_2_s, t85);
__m512d bc1_3_c, bc1_3_s;
BCASTV(bc1_3_c, HT_17[4]); BCASTV(bc1_3_s, HT_17[5]);
accAr3 = _mm512_fmadd_pd(bc1_3_c, t83, accAr3);
accAi3 = _mm512_fmadd_pd(bc1_3_c, t84, accAi3);
accBr3 = _mm512_mul_pd(bc1_3_s, t86);
accBi3 = _mm512_mul_pd(bc1_3_s, t85);
__m512d bc1_4_c, bc1_4_s;
BCASTV(bc1_4_c, HT_17[6]); BCASTV(bc1_4_s, HT_17[7]);
accAr4 = _mm512_fmadd_pd(bc1_4_c, t83, accAr4);
accAi4 = _mm512_fmadd_pd(bc1_4_c, t84, accAi4);
accBr4 = _mm512_mul_pd(bc1_4_s, t86);
accBi4 = _mm512_mul_pd(bc1_4_s, t85);
__m512d t87 = _mm512_load_pd(HS_17[1]);
__m512d t88 = _mm512_load_pd(HS_17[1]+8);
__m512d t89 = _mm512_load_pd(HS_17[1]+16);
__m512d t90 = _mm512_load_pd(HS_17[1]+24);
__m512d bc2_1_c, bc2_1_s;
BCASTV(bc2_1_c, HT_17[8]); BCASTV(bc2_1_s, HT_17[9]);
accAr1 = _mm512_fmadd_pd(bc2_1_c, t87, accAr1);
accAi1 = _mm512_fmadd_pd(bc2_1_c, t88, accAi1);
accBr1 = _mm512_fmadd_pd(bc2_1_s, t90, accBr1);
accBi1 = _mm512_fmadd_pd(bc2_1_s, t89, accBi1);
__m512d bc2_2_c, bc2_2_s;
BCASTV(bc2_2_c, HT_17[10]); BCASTV(bc2_2_s, HT_17[11]);
accAr2 = _mm512_fmadd_pd(bc2_2_c, t87, accAr2);
accAi2 = _mm512_fmadd_pd(bc2_2_c, t88, accAi2);
accBr2 = _mm512_fmadd_pd(bc2_2_s, t90, accBr2);
accBi2 = _mm512_fmadd_pd(bc2_2_s, t89, accBi2);
__m512d bc2_3_c, bc2_3_s;
BCASTV(bc2_3_c, HT_17[12]); BCASTV(bc2_3_s, HT_17[13]);
accAr3 = _mm512_fmadd_pd(bc2_3_c, t87, accAr3);
accAi3 = _mm512_fmadd_pd(bc2_3_c, t88, accAi3);
accBr3 = _mm512_fmadd_pd(bc2_3_s, t90, accBr3);
accBi3 = _mm512_fmadd_pd(bc2_3_s, t89, accBi3);
__m512d bc2_4_c, bc2_4_s;
BCASTV(bc2_4_c, HT_17[14]); BCASTV(bc2_4_s, HT_17[15]);
accAr4 = _mm512_fmadd_pd(bc2_4_c, t87, accAr4);
accAi4 = _mm512_fmadd_pd(bc2_4_c, t88, accAi4);
accBr4 = _mm512_fmadd_pd(bc2_4_s, t90, accBr4);
accBi4 = _mm512_fmadd_pd(bc2_4_s, t89, accBi4);
__m512d t91 = _mm512_load_pd(HS_17[2]);
__m512d t92 = _mm512_load_pd(HS_17[2]+8);
__m512d t93 = _mm512_load_pd(HS_17[2]+16);
__m512d t94 = _mm512_load_pd(HS_17[2]+24);
__m512d bc3_1_c, bc3_1_s;
BCASTV(bc3_1_c, HT_17[16]); BCASTV(bc3_1_s, HT_17[17]);
accAr1 = _mm512_fmadd_pd(bc3_1_c, t91, accAr1);
accAi1 = _mm512_fmadd_pd(bc3_1_c, t92, accAi1);
accBr1 = _mm512_fmadd_pd(bc3_1_s, t94, accBr1);
accBi1 = _mm512_fmadd_pd(bc3_1_s, t93, accBi1);
__m512d bc3_2_c, bc3_2_s;
BCASTV(bc3_2_c, HT_17[18]); BCASTV(bc3_2_s, HT_17[19]);
accAr2 = _mm512_fmadd_pd(bc3_2_c, t91, accAr2);
accAi2 = _mm512_fmadd_pd(bc3_2_c, t92, accAi2);
accBr2 = _mm512_fmadd_pd(bc3_2_s, t94, accBr2);
accBi2 = _mm512_fmadd_pd(bc3_2_s, t93, accBi2);
__m512d bc3_3_c, bc3_3_s;
BCASTV(bc3_3_c, HT_17[20]); BCASTV(bc3_3_s, HT_17[21]);
accAr3 = _mm512_fmadd_pd(bc3_3_c, t91, accAr3);
accAi3 = _mm512_fmadd_pd(bc3_3_c, t92, accAi3);
accBr3 = _mm512_fmadd_pd(bc3_3_s, t94, accBr3);
accBi3 = _mm512_fmadd_pd(bc3_3_s, t93, accBi3);
__m512d bc3_4_c, bc3_4_s;
BCASTV(bc3_4_c, HT_17[22]); BCASTV(bc3_4_s, HT_17[23]);
accAr4 = _mm512_fmadd_pd(bc3_4_c, t91, accAr4);
accAi4 = _mm512_fmadd_pd(bc3_4_c, t92, accAi4);
accBr4 = _mm512_fmadd_pd(bc3_4_s, t94, accBr4);
accBi4 = _mm512_fmadd_pd(bc3_4_s, t93, accBi4);
__m512d t95 = _mm512_load_pd(HS_17[3]);
__m512d t96 = _mm512_load_pd(HS_17[3]+8);
__m512d t97 = _mm512_load_pd(HS_17[3]+16);
__m512d t98 = _mm512_load_pd(HS_17[3]+24);
__m512d bc4_1_c, bc4_1_s;
BCASTV(bc4_1_c, HT_17[24]); BCASTV(bc4_1_s, HT_17[25]);
accAr1 = _mm512_fmadd_pd(bc4_1_c, t95, accAr1);
accAi1 = _mm512_fmadd_pd(bc4_1_c, t96, accAi1);
accBr1 = _mm512_fmadd_pd(bc4_1_s, t98, accBr1);
accBi1 = _mm512_fmadd_pd(bc4_1_s, t97, accBi1);
__m512d bc4_2_c, bc4_2_s;
BCASTV(bc4_2_c, HT_17[26]); BCASTV(bc4_2_s, HT_17[27]);
accAr2 = _mm512_fmadd_pd(bc4_2_c, t95, accAr2);
accAi2 = _mm512_fmadd_pd(bc4_2_c, t96, accAi2);
accBr2 = _mm512_fmadd_pd(bc4_2_s, t98, accBr2);
accBi2 = _mm512_fmadd_pd(bc4_2_s, t97, accBi2);
__m512d bc4_3_c, bc4_3_s;
BCASTV(bc4_3_c, HT_17[28]); BCASTV(bc4_3_s, HT_17[29]);
accAr3 = _mm512_fmadd_pd(bc4_3_c, t95, accAr3);
accAi3 = _mm512_fmadd_pd(bc4_3_c, t96, accAi3);
accBr3 = _mm512_fmadd_pd(bc4_3_s, t98, accBr3);
accBi3 = _mm512_fmadd_pd(bc4_3_s, t97, accBi3);
__m512d bc4_4_c, bc4_4_s;
BCASTV(bc4_4_c, HT_17[30]); BCASTV(bc4_4_s, HT_17[31]);
accAr4 = _mm512_fmadd_pd(bc4_4_c, t95, accAr4);
accAi4 = _mm512_fmadd_pd(bc4_4_c, t96, accAi4);
accBr4 = _mm512_fmadd_pd(bc4_4_s, t98, accBr4);
accBi4 = _mm512_fmadd_pd(bc4_4_s, t97, accBi4);
__m512d t99 = _mm512_load_pd(HS_17[4]);
__m512d t100 = _mm512_load_pd(HS_17[4]+8);
__m512d t101 = _mm512_load_pd(HS_17[4]+16);
__m512d t102 = _mm512_load_pd(HS_17[4]+24);
__m512d bc5_1_c, bc5_1_s;
BCASTV(bc5_1_c, HT_17[32]); BCASTV(bc5_1_s, HT_17[33]);
accAr1 = _mm512_fmadd_pd(bc5_1_c, t99, accAr1);
accAi1 = _mm512_fmadd_pd(bc5_1_c, t100, accAi1);
accBr1 = _mm512_fmadd_pd(bc5_1_s, t102, accBr1);
accBi1 = _mm512_fmadd_pd(bc5_1_s, t101, accBi1);
__m512d bc5_2_c, bc5_2_s;
BCASTV(bc5_2_c, HT_17[34]); BCASTV(bc5_2_s, HT_17[35]);
accAr2 = _mm512_fmadd_pd(bc5_2_c, t99, accAr2);
accAi2 = _mm512_fmadd_pd(bc5_2_c, t100, accAi2);
accBr2 = _mm512_fmadd_pd(bc5_2_s, t102, accBr2);
accBi2 = _mm512_fmadd_pd(bc5_2_s, t101, accBi2);
__m512d bc5_3_c, bc5_3_s;
BCASTV(bc5_3_c, HT_17[36]); BCASTV(bc5_3_s, HT_17[37]);
accAr3 = _mm512_fmadd_pd(bc5_3_c, t99, accAr3);
accAi3 = _mm512_fmadd_pd(bc5_3_c, t100, accAi3);
accBr3 = _mm512_fmadd_pd(bc5_3_s, t102, accBr3);
accBi3 = _mm512_fmadd_pd(bc5_3_s, t101, accBi3);
__m512d bc5_4_c, bc5_4_s;
BCASTV(bc5_4_c, HT_17[38]); BCASTV(bc5_4_s, HT_17[39]);
accAr4 = _mm512_fmadd_pd(bc5_4_c, t99, accAr4);
accAi4 = _mm512_fmadd_pd(bc5_4_c, t100, accAi4);
accBr4 = _mm512_fmadd_pd(bc5_4_s, t102, accBr4);
accBi4 = _mm512_fmadd_pd(bc5_4_s, t101, accBi4);
__m512d t103 = _mm512_load_pd(HS_17[5]);
__m512d t104 = _mm512_load_pd(HS_17[5]+8);
__m512d t105 = _mm512_load_pd(HS_17[5]+16);
__m512d t106 = _mm512_load_pd(HS_17[5]+24);
__m512d bc6_1_c, bc6_1_s;
BCASTV(bc6_1_c, HT_17[40]); BCASTV(bc6_1_s, HT_17[41]);
accAr1 = _mm512_fmadd_pd(bc6_1_c, t103, accAr1);
accAi1 = _mm512_fmadd_pd(bc6_1_c, t104, accAi1);
accBr1 = _mm512_fmadd_pd(bc6_1_s, t106, accBr1);
accBi1 = _mm512_fmadd_pd(bc6_1_s, t105, accBi1);
__m512d bc6_2_c, bc6_2_s;
BCASTV(bc6_2_c, HT_17[42]); BCASTV(bc6_2_s, HT_17[43]);
accAr2 = _mm512_fmadd_pd(bc6_2_c, t103, accAr2);
accAi2 = _mm512_fmadd_pd(bc6_2_c, t104, accAi2);
accBr2 = _mm512_fmadd_pd(bc6_2_s, t106, accBr2);
accBi2 = _mm512_fmadd_pd(bc6_2_s, t105, accBi2);
__m512d bc6_3_c, bc6_3_s;
BCASTV(bc6_3_c, HT_17[44]); BCASTV(bc6_3_s, HT_17[45]);
accAr3 = _mm512_fmadd_pd(bc6_3_c, t103, accAr3);
accAi3 = _mm512_fmadd_pd(bc6_3_c, t104, accAi3);
accBr3 = _mm512_fmadd_pd(bc6_3_s, t106, accBr3);
accBi3 = _mm512_fmadd_pd(bc6_3_s, t105, accBi3);
__m512d bc6_4_c, bc6_4_s;
BCASTV(bc6_4_c, HT_17[46]); BCASTV(bc6_4_s, HT_17[47]);
accAr4 = _mm512_fmadd_pd(bc6_4_c, t103, accAr4);
accAi4 = _mm512_fmadd_pd(bc6_4_c, t104, accAi4);
accBr4 = _mm512_fmadd_pd(bc6_4_s, t106, accBr4);
accBi4 = _mm512_fmadd_pd(bc6_4_s, t105, accBi4);
__m512d t107 = _mm512_load_pd(HS_17[6]);
__m512d t108 = _mm512_load_pd(HS_17[6]+8);
__m512d t109 = _mm512_load_pd(HS_17[6]+16);
__m512d t110 = _mm512_load_pd(HS_17[6]+24);
__m512d bc7_1_c, bc7_1_s;
BCASTV(bc7_1_c, HT_17[48]); BCASTV(bc7_1_s, HT_17[49]);
accAr1 = _mm512_fmadd_pd(bc7_1_c, t107, accAr1);
accAi1 = _mm512_fmadd_pd(bc7_1_c, t108, accAi1);
accBr1 = _mm512_fmadd_pd(bc7_1_s, t110, accBr1);
accBi1 = _mm512_fmadd_pd(bc7_1_s, t109, accBi1);
__m512d bc7_2_c, bc7_2_s;
BCASTV(bc7_2_c, HT_17[50]); BCASTV(bc7_2_s, HT_17[51]);
accAr2 = _mm512_fmadd_pd(bc7_2_c, t107, accAr2);
accAi2 = _mm512_fmadd_pd(bc7_2_c, t108, accAi2);
accBr2 = _mm512_fmadd_pd(bc7_2_s, t110, accBr2);
accBi2 = _mm512_fmadd_pd(bc7_2_s, t109, accBi2);
__m512d bc7_3_c, bc7_3_s;
BCASTV(bc7_3_c, HT_17[52]); BCASTV(bc7_3_s, HT_17[53]);
accAr3 = _mm512_fmadd_pd(bc7_3_c, t107, accAr3);
accAi3 = _mm512_fmadd_pd(bc7_3_c, t108, accAi3);
accBr3 = _mm512_fmadd_pd(bc7_3_s, t110, accBr3);
accBi3 = _mm512_fmadd_pd(bc7_3_s, t109, accBi3);
__m512d bc7_4_c, bc7_4_s;
BCASTV(bc7_4_c, HT_17[54]); BCASTV(bc7_4_s, HT_17[55]);
accAr4 = _mm512_fmadd_pd(bc7_4_c, t107, accAr4);
accAi4 = _mm512_fmadd_pd(bc7_4_c, t108, accAi4);
accBr4 = _mm512_fmadd_pd(bc7_4_s, t110, accBr4);
accBi4 = _mm512_fmadd_pd(bc7_4_s, t109, accBi4);
__m512d t111 = _mm512_load_pd(HS_17[7]);
__m512d t112 = _mm512_load_pd(HS_17[7]+8);
__m512d t113 = _mm512_load_pd(HS_17[7]+16);
__m512d t114 = _mm512_load_pd(HS_17[7]+24);
__m512d bc8_1_c, bc8_1_s;
BCASTV(bc8_1_c, HT_17[56]); BCASTV(bc8_1_s, HT_17[57]);
accAr1 = _mm512_fmadd_pd(bc8_1_c, t111, accAr1);
accAi1 = _mm512_fmadd_pd(bc8_1_c, t112, accAi1);
accBr1 = _mm512_fmadd_pd(bc8_1_s, t114, accBr1);
accBi1 = _mm512_fmadd_pd(bc8_1_s, t113, accBi1);
__m512d bc8_2_c, bc8_2_s;
BCASTV(bc8_2_c, HT_17[58]); BCASTV(bc8_2_s, HT_17[59]);
accAr2 = _mm512_fmadd_pd(bc8_2_c, t111, accAr2);
accAi2 = _mm512_fmadd_pd(bc8_2_c, t112, accAi2);
accBr2 = _mm512_fmadd_pd(bc8_2_s, t114, accBr2);
accBi2 = _mm512_fmadd_pd(bc8_2_s, t113, accBi2);
__m512d bc8_3_c, bc8_3_s;
BCASTV(bc8_3_c, HT_17[60]); BCASTV(bc8_3_s, HT_17[61]);
accAr3 = _mm512_fmadd_pd(bc8_3_c, t111, accAr3);
accAi3 = _mm512_fmadd_pd(bc8_3_c, t112, accAi3);
accBr3 = _mm512_fmadd_pd(bc8_3_s, t114, accBr3);
accBi3 = _mm512_fmadd_pd(bc8_3_s, t113, accBi3);
__m512d bc8_4_c, bc8_4_s;
BCASTV(bc8_4_c, HT_17[62]); BCASTV(bc8_4_s, HT_17[63]);
accAr4 = _mm512_fmadd_pd(bc8_4_c, t111, accAr4);
accAi4 = _mm512_fmadd_pd(bc8_4_c, t112, accAi4);
accBr4 = _mm512_fmadd_pd(bc8_4_s, t114, accBr4);
accBi4 = _mm512_fmadd_pd(bc8_4_s, t113, accBi4);
__m512d t115 = _mm512_add_pd(accAr1, accBr1);
__m512d t116 = _mm512_sub_pd(accAi1, accBi1);
__m512d t117 = _mm512_sub_pd(accAr1, accBr1);
__m512d t118 = _mm512_add_pd(accAi1, accBi1);
_mm512_store_pd(re + 1*es, t115);
_mm512_store_pd(im + 1*es, t116);
_mm512_store_pd(re + 16*es, t117);
_mm512_store_pd(im + 16*es, t118);
__m512d t119 = _mm512_add_pd(accAr2, accBr2);
__m512d t120 = _mm512_sub_pd(accAi2, accBi2);
__m512d t121 = _mm512_sub_pd(accAr2, accBr2);
__m512d t122 = _mm512_add_pd(accAi2, accBi2);
_mm512_store_pd(re + 2*es, t119);
_mm512_store_pd(im + 2*es, t120);
_mm512_store_pd(re + 15*es, t121);
_mm512_store_pd(im + 15*es, t122);
__m512d t123 = _mm512_add_pd(accAr3, accBr3);
__m512d t124 = _mm512_sub_pd(accAi3, accBi3);
__m512d t125 = _mm512_sub_pd(accAr3, accBr3);
__m512d t126 = _mm512_add_pd(accAi3, accBi3);
_mm512_store_pd(re + 3*es, t123);
_mm512_store_pd(im + 3*es, t124);
_mm512_store_pd(re + 14*es, t125);
_mm512_store_pd(im + 14*es, t126);
__m512d t127 = _mm512_add_pd(accAr4, accBr4);
__m512d t128 = _mm512_sub_pd(accAi4, accBi4);
__m512d t129 = _mm512_sub_pd(accAr4, accBr4);
__m512d t130 = _mm512_add_pd(accAi4, accBi4);
_mm512_store_pd(re + 4*es, t127);
_mm512_store_pd(im + 4*es, t128);
_mm512_store_pd(re + 13*es, t129);
_mm512_store_pd(im + 13*es, t130);
__asm__ volatile("" ::: "memory");
__m512d accAr5 = t1, accAi5 = t2, accBr5, accBi5;
__m512d accAr6 = t1, accAi6 = t2, accBr6, accBi6;
__m512d accAr7 = t1, accAi7 = t2, accBr7, accBi7;
__m512d accAr8 = t1, accAi8 = t2, accBr8, accBi8;
__m512d t131 = _mm512_load_pd(HS_17[0]);
__m512d t132 = _mm512_load_pd(HS_17[0]+8);
__m512d t133 = _mm512_load_pd(HS_17[0]+16);
__m512d t134 = _mm512_load_pd(HS_17[0]+24);
__m512d bc1_5_c, bc1_5_s;
BCASTV(bc1_5_c, HT_17[64]); BCASTV(bc1_5_s, HT_17[65]);
accAr5 = _mm512_fmadd_pd(bc1_5_c, t131, accAr5);
accAi5 = _mm512_fmadd_pd(bc1_5_c, t132, accAi5);
accBr5 = _mm512_mul_pd(bc1_5_s, t134);
accBi5 = _mm512_mul_pd(bc1_5_s, t133);
__m512d bc1_6_c, bc1_6_s;
BCASTV(bc1_6_c, HT_17[66]); BCASTV(bc1_6_s, HT_17[67]);
accAr6 = _mm512_fmadd_pd(bc1_6_c, t131, accAr6);
accAi6 = _mm512_fmadd_pd(bc1_6_c, t132, accAi6);
accBr6 = _mm512_mul_pd(bc1_6_s, t134);
accBi6 = _mm512_mul_pd(bc1_6_s, t133);
__m512d bc1_7_c, bc1_7_s;
BCASTV(bc1_7_c, HT_17[68]); BCASTV(bc1_7_s, HT_17[69]);
accAr7 = _mm512_fmadd_pd(bc1_7_c, t131, accAr7);
accAi7 = _mm512_fmadd_pd(bc1_7_c, t132, accAi7);
accBr7 = _mm512_mul_pd(bc1_7_s, t134);
accBi7 = _mm512_mul_pd(bc1_7_s, t133);
__m512d bc1_8_c, bc1_8_s;
BCASTV(bc1_8_c, HT_17[70]); BCASTV(bc1_8_s, HT_17[71]);
accAr8 = _mm512_fmadd_pd(bc1_8_c, t131, accAr8);
accAi8 = _mm512_fmadd_pd(bc1_8_c, t132, accAi8);
accBr8 = _mm512_mul_pd(bc1_8_s, t134);
accBi8 = _mm512_mul_pd(bc1_8_s, t133);
__m512d t135 = _mm512_load_pd(HS_17[1]);
__m512d t136 = _mm512_load_pd(HS_17[1]+8);
__m512d t137 = _mm512_load_pd(HS_17[1]+16);
__m512d t138 = _mm512_load_pd(HS_17[1]+24);
__m512d bc2_5_c, bc2_5_s;
BCASTV(bc2_5_c, HT_17[72]); BCASTV(bc2_5_s, HT_17[73]);
accAr5 = _mm512_fmadd_pd(bc2_5_c, t135, accAr5);
accAi5 = _mm512_fmadd_pd(bc2_5_c, t136, accAi5);
accBr5 = _mm512_fmadd_pd(bc2_5_s, t138, accBr5);
accBi5 = _mm512_fmadd_pd(bc2_5_s, t137, accBi5);
__m512d bc2_6_c, bc2_6_s;
BCASTV(bc2_6_c, HT_17[74]); BCASTV(bc2_6_s, HT_17[75]);
accAr6 = _mm512_fmadd_pd(bc2_6_c, t135, accAr6);
accAi6 = _mm512_fmadd_pd(bc2_6_c, t136, accAi6);
accBr6 = _mm512_fmadd_pd(bc2_6_s, t138, accBr6);
accBi6 = _mm512_fmadd_pd(bc2_6_s, t137, accBi6);
__m512d bc2_7_c, bc2_7_s;
BCASTV(bc2_7_c, HT_17[76]); BCASTV(bc2_7_s, HT_17[77]);
accAr7 = _mm512_fmadd_pd(bc2_7_c, t135, accAr7);
accAi7 = _mm512_fmadd_pd(bc2_7_c, t136, accAi7);
accBr7 = _mm512_fmadd_pd(bc2_7_s, t138, accBr7);
accBi7 = _mm512_fmadd_pd(bc2_7_s, t137, accBi7);
__m512d bc2_8_c, bc2_8_s;
BCASTV(bc2_8_c, HT_17[78]); BCASTV(bc2_8_s, HT_17[79]);
accAr8 = _mm512_fmadd_pd(bc2_8_c, t135, accAr8);
accAi8 = _mm512_fmadd_pd(bc2_8_c, t136, accAi8);
accBr8 = _mm512_fmadd_pd(bc2_8_s, t138, accBr8);
accBi8 = _mm512_fmadd_pd(bc2_8_s, t137, accBi8);
__m512d t139 = _mm512_load_pd(HS_17[2]);
__m512d t140 = _mm512_load_pd(HS_17[2]+8);
__m512d t141 = _mm512_load_pd(HS_17[2]+16);
__m512d t142 = _mm512_load_pd(HS_17[2]+24);
__m512d bc3_5_c, bc3_5_s;
BCASTV(bc3_5_c, HT_17[80]); BCASTV(bc3_5_s, HT_17[81]);
accAr5 = _mm512_fmadd_pd(bc3_5_c, t139, accAr5);
accAi5 = _mm512_fmadd_pd(bc3_5_c, t140, accAi5);
accBr5 = _mm512_fmadd_pd(bc3_5_s, t142, accBr5);
accBi5 = _mm512_fmadd_pd(bc3_5_s, t141, accBi5);
__m512d bc3_6_c, bc3_6_s;
BCASTV(bc3_6_c, HT_17[82]); BCASTV(bc3_6_s, HT_17[83]);
accAr6 = _mm512_fmadd_pd(bc3_6_c, t139, accAr6);
accAi6 = _mm512_fmadd_pd(bc3_6_c, t140, accAi6);
accBr6 = _mm512_fmadd_pd(bc3_6_s, t142, accBr6);
accBi6 = _mm512_fmadd_pd(bc3_6_s, t141, accBi6);
__m512d bc3_7_c, bc3_7_s;
BCASTV(bc3_7_c, HT_17[84]); BCASTV(bc3_7_s, HT_17[85]);
accAr7 = _mm512_fmadd_pd(bc3_7_c, t139, accAr7);
accAi7 = _mm512_fmadd_pd(bc3_7_c, t140, accAi7);
accBr7 = _mm512_fmadd_pd(bc3_7_s, t142, accBr7);
accBi7 = _mm512_fmadd_pd(bc3_7_s, t141, accBi7);
__m512d bc3_8_c, bc3_8_s;
BCASTV(bc3_8_c, HT_17[86]); BCASTV(bc3_8_s, HT_17[87]);
accAr8 = _mm512_fmadd_pd(bc3_8_c, t139, accAr8);
accAi8 = _mm512_fmadd_pd(bc3_8_c, t140, accAi8);
accBr8 = _mm512_fmadd_pd(bc3_8_s, t142, accBr8);
accBi8 = _mm512_fmadd_pd(bc3_8_s, t141, accBi8);
__m512d t143 = _mm512_load_pd(HS_17[3]);
__m512d t144 = _mm512_load_pd(HS_17[3]+8);
__m512d t145 = _mm512_load_pd(HS_17[3]+16);
__m512d t146 = _mm512_load_pd(HS_17[3]+24);
__m512d bc4_5_c, bc4_5_s;
BCASTV(bc4_5_c, HT_17[88]); BCASTV(bc4_5_s, HT_17[89]);
accAr5 = _mm512_fmadd_pd(bc4_5_c, t143, accAr5);
accAi5 = _mm512_fmadd_pd(bc4_5_c, t144, accAi5);
accBr5 = _mm512_fmadd_pd(bc4_5_s, t146, accBr5);
accBi5 = _mm512_fmadd_pd(bc4_5_s, t145, accBi5);
__m512d bc4_6_c, bc4_6_s;
BCASTV(bc4_6_c, HT_17[90]); BCASTV(bc4_6_s, HT_17[91]);
accAr6 = _mm512_fmadd_pd(bc4_6_c, t143, accAr6);
accAi6 = _mm512_fmadd_pd(bc4_6_c, t144, accAi6);
accBr6 = _mm512_fmadd_pd(bc4_6_s, t146, accBr6);
accBi6 = _mm512_fmadd_pd(bc4_6_s, t145, accBi6);
__m512d bc4_7_c, bc4_7_s;
BCASTV(bc4_7_c, HT_17[92]); BCASTV(bc4_7_s, HT_17[93]);
accAr7 = _mm512_fmadd_pd(bc4_7_c, t143, accAr7);
accAi7 = _mm512_fmadd_pd(bc4_7_c, t144, accAi7);
accBr7 = _mm512_fmadd_pd(bc4_7_s, t146, accBr7);
accBi7 = _mm512_fmadd_pd(bc4_7_s, t145, accBi7);
__m512d bc4_8_c, bc4_8_s;
BCASTV(bc4_8_c, HT_17[94]); BCASTV(bc4_8_s, HT_17[95]);
accAr8 = _mm512_fmadd_pd(bc4_8_c, t143, accAr8);
accAi8 = _mm512_fmadd_pd(bc4_8_c, t144, accAi8);
accBr8 = _mm512_fmadd_pd(bc4_8_s, t146, accBr8);
accBi8 = _mm512_fmadd_pd(bc4_8_s, t145, accBi8);
__m512d t147 = _mm512_load_pd(HS_17[4]);
__m512d t148 = _mm512_load_pd(HS_17[4]+8);
__m512d t149 = _mm512_load_pd(HS_17[4]+16);
__m512d t150 = _mm512_load_pd(HS_17[4]+24);
__m512d bc5_5_c, bc5_5_s;
BCASTV(bc5_5_c, HT_17[96]); BCASTV(bc5_5_s, HT_17[97]);
accAr5 = _mm512_fmadd_pd(bc5_5_c, t147, accAr5);
accAi5 = _mm512_fmadd_pd(bc5_5_c, t148, accAi5);
accBr5 = _mm512_fmadd_pd(bc5_5_s, t150, accBr5);
accBi5 = _mm512_fmadd_pd(bc5_5_s, t149, accBi5);
__m512d bc5_6_c, bc5_6_s;
BCASTV(bc5_6_c, HT_17[98]); BCASTV(bc5_6_s, HT_17[99]);
accAr6 = _mm512_fmadd_pd(bc5_6_c, t147, accAr6);
accAi6 = _mm512_fmadd_pd(bc5_6_c, t148, accAi6);
accBr6 = _mm512_fmadd_pd(bc5_6_s, t150, accBr6);
accBi6 = _mm512_fmadd_pd(bc5_6_s, t149, accBi6);
__m512d bc5_7_c, bc5_7_s;
BCASTV(bc5_7_c, HT_17[100]); BCASTV(bc5_7_s, HT_17[101]);
accAr7 = _mm512_fmadd_pd(bc5_7_c, t147, accAr7);
accAi7 = _mm512_fmadd_pd(bc5_7_c, t148, accAi7);
accBr7 = _mm512_fmadd_pd(bc5_7_s, t150, accBr7);
accBi7 = _mm512_fmadd_pd(bc5_7_s, t149, accBi7);
__m512d bc5_8_c, bc5_8_s;
BCASTV(bc5_8_c, HT_17[102]); BCASTV(bc5_8_s, HT_17[103]);
accAr8 = _mm512_fmadd_pd(bc5_8_c, t147, accAr8);
accAi8 = _mm512_fmadd_pd(bc5_8_c, t148, accAi8);
accBr8 = _mm512_fmadd_pd(bc5_8_s, t150, accBr8);
accBi8 = _mm512_fmadd_pd(bc5_8_s, t149, accBi8);
__m512d t151 = _mm512_load_pd(HS_17[5]);
__m512d t152 = _mm512_load_pd(HS_17[5]+8);
__m512d t153 = _mm512_load_pd(HS_17[5]+16);
__m512d t154 = _mm512_load_pd(HS_17[5]+24);
__m512d bc6_5_c, bc6_5_s;
BCASTV(bc6_5_c, HT_17[104]); BCASTV(bc6_5_s, HT_17[105]);
accAr5 = _mm512_fmadd_pd(bc6_5_c, t151, accAr5);
accAi5 = _mm512_fmadd_pd(bc6_5_c, t152, accAi5);
accBr5 = _mm512_fmadd_pd(bc6_5_s, t154, accBr5);
accBi5 = _mm512_fmadd_pd(bc6_5_s, t153, accBi5);
__m512d bc6_6_c, bc6_6_s;
BCASTV(bc6_6_c, HT_17[106]); BCASTV(bc6_6_s, HT_17[107]);
accAr6 = _mm512_fmadd_pd(bc6_6_c, t151, accAr6);
accAi6 = _mm512_fmadd_pd(bc6_6_c, t152, accAi6);
accBr6 = _mm512_fmadd_pd(bc6_6_s, t154, accBr6);
accBi6 = _mm512_fmadd_pd(bc6_6_s, t153, accBi6);
__m512d bc6_7_c, bc6_7_s;
BCASTV(bc6_7_c, HT_17[108]); BCASTV(bc6_7_s, HT_17[109]);
accAr7 = _mm512_fmadd_pd(bc6_7_c, t151, accAr7);
accAi7 = _mm512_fmadd_pd(bc6_7_c, t152, accAi7);
accBr7 = _mm512_fmadd_pd(bc6_7_s, t154, accBr7);
accBi7 = _mm512_fmadd_pd(bc6_7_s, t153, accBi7);
__m512d bc6_8_c, bc6_8_s;
BCASTV(bc6_8_c, HT_17[110]); BCASTV(bc6_8_s, HT_17[111]);
accAr8 = _mm512_fmadd_pd(bc6_8_c, t151, accAr8);
accAi8 = _mm512_fmadd_pd(bc6_8_c, t152, accAi8);
accBr8 = _mm512_fmadd_pd(bc6_8_s, t154, accBr8);
accBi8 = _mm512_fmadd_pd(bc6_8_s, t153, accBi8);
__m512d t155 = _mm512_load_pd(HS_17[6]);
__m512d t156 = _mm512_load_pd(HS_17[6]+8);
__m512d t157 = _mm512_load_pd(HS_17[6]+16);
__m512d t158 = _mm512_load_pd(HS_17[6]+24);
__m512d bc7_5_c, bc7_5_s;
BCASTV(bc7_5_c, HT_17[112]); BCASTV(bc7_5_s, HT_17[113]);
accAr5 = _mm512_fmadd_pd(bc7_5_c, t155, accAr5);
accAi5 = _mm512_fmadd_pd(bc7_5_c, t156, accAi5);
accBr5 = _mm512_fmadd_pd(bc7_5_s, t158, accBr5);
accBi5 = _mm512_fmadd_pd(bc7_5_s, t157, accBi5);
__m512d bc7_6_c, bc7_6_s;
BCASTV(bc7_6_c, HT_17[114]); BCASTV(bc7_6_s, HT_17[115]);
accAr6 = _mm512_fmadd_pd(bc7_6_c, t155, accAr6);
accAi6 = _mm512_fmadd_pd(bc7_6_c, t156, accAi6);
accBr6 = _mm512_fmadd_pd(bc7_6_s, t158, accBr6);
accBi6 = _mm512_fmadd_pd(bc7_6_s, t157, accBi6);
__m512d bc7_7_c, bc7_7_s;
BCASTV(bc7_7_c, HT_17[116]); BCASTV(bc7_7_s, HT_17[117]);
accAr7 = _mm512_fmadd_pd(bc7_7_c, t155, accAr7);
accAi7 = _mm512_fmadd_pd(bc7_7_c, t156, accAi7);
accBr7 = _mm512_fmadd_pd(bc7_7_s, t158, accBr7);
accBi7 = _mm512_fmadd_pd(bc7_7_s, t157, accBi7);
__m512d bc7_8_c, bc7_8_s;
BCASTV(bc7_8_c, HT_17[118]); BCASTV(bc7_8_s, HT_17[119]);
accAr8 = _mm512_fmadd_pd(bc7_8_c, t155, accAr8);
accAi8 = _mm512_fmadd_pd(bc7_8_c, t156, accAi8);
accBr8 = _mm512_fmadd_pd(bc7_8_s, t158, accBr8);
accBi8 = _mm512_fmadd_pd(bc7_8_s, t157, accBi8);
__m512d t159 = _mm512_load_pd(HS_17[7]);
__m512d t160 = _mm512_load_pd(HS_17[7]+8);
__m512d t161 = _mm512_load_pd(HS_17[7]+16);
__m512d t162 = _mm512_load_pd(HS_17[7]+24);
__m512d bc8_5_c, bc8_5_s;
BCASTV(bc8_5_c, HT_17[120]); BCASTV(bc8_5_s, HT_17[121]);
accAr5 = _mm512_fmadd_pd(bc8_5_c, t159, accAr5);
accAi5 = _mm512_fmadd_pd(bc8_5_c, t160, accAi5);
accBr5 = _mm512_fmadd_pd(bc8_5_s, t162, accBr5);
accBi5 = _mm512_fmadd_pd(bc8_5_s, t161, accBi5);
__m512d bc8_6_c, bc8_6_s;
BCASTV(bc8_6_c, HT_17[122]); BCASTV(bc8_6_s, HT_17[123]);
accAr6 = _mm512_fmadd_pd(bc8_6_c, t159, accAr6);
accAi6 = _mm512_fmadd_pd(bc8_6_c, t160, accAi6);
accBr6 = _mm512_fmadd_pd(bc8_6_s, t162, accBr6);
accBi6 = _mm512_fmadd_pd(bc8_6_s, t161, accBi6);
__m512d bc8_7_c, bc8_7_s;
BCASTV(bc8_7_c, HT_17[124]); BCASTV(bc8_7_s, HT_17[125]);
accAr7 = _mm512_fmadd_pd(bc8_7_c, t159, accAr7);
accAi7 = _mm512_fmadd_pd(bc8_7_c, t160, accAi7);
accBr7 = _mm512_fmadd_pd(bc8_7_s, t162, accBr7);
accBi7 = _mm512_fmadd_pd(bc8_7_s, t161, accBi7);
__m512d bc8_8_c, bc8_8_s;
BCASTV(bc8_8_c, HT_17[126]); BCASTV(bc8_8_s, HT_17[127]);
accAr8 = _mm512_fmadd_pd(bc8_8_c, t159, accAr8);
accAi8 = _mm512_fmadd_pd(bc8_8_c, t160, accAi8);
accBr8 = _mm512_fmadd_pd(bc8_8_s, t162, accBr8);
accBi8 = _mm512_fmadd_pd(bc8_8_s, t161, accBi8);
__m512d t163 = _mm512_add_pd(accAr5, accBr5);
__m512d t164 = _mm512_sub_pd(accAi5, accBi5);
__m512d t165 = _mm512_sub_pd(accAr5, accBr5);
__m512d t166 = _mm512_add_pd(accAi5, accBi5);
_mm512_store_pd(re + 5*es, t163);
_mm512_store_pd(im + 5*es, t164);
_mm512_store_pd(re + 12*es, t165);
_mm512_store_pd(im + 12*es, t166);
__m512d t167 = _mm512_add_pd(accAr6, accBr6);
__m512d t168 = _mm512_sub_pd(accAi6, accBi6);
__m512d t169 = _mm512_sub_pd(accAr6, accBr6);
__m512d t170 = _mm512_add_pd(accAi6, accBi6);
_mm512_store_pd(re + 6*es, t167);
_mm512_store_pd(im + 6*es, t168);
_mm512_store_pd(re + 11*es, t169);
_mm512_store_pd(im + 11*es, t170);
__m512d t171 = _mm512_add_pd(accAr7, accBr7);
__m512d t172 = _mm512_sub_pd(accAi7, accBi7);
__m512d t173 = _mm512_sub_pd(accAr7, accBr7);
__m512d t174 = _mm512_add_pd(accAi7, accBi7);
_mm512_store_pd(re + 7*es, t171);
_mm512_store_pd(im + 7*es, t172);
_mm512_store_pd(re + 10*es, t173);
_mm512_store_pd(im + 10*es, t174);
__m512d t175 = _mm512_add_pd(accAr8, accBr8);
__m512d t176 = _mm512_sub_pd(accAi8, accBi8);
__m512d t177 = _mm512_sub_pd(accAr8, accBr8);
__m512d t178 = _mm512_add_pd(accAi8, accBi8);
_mm512_store_pd(re + 8*es, t175);
_mm512_store_pd(im + 8*es, t176);
_mm512_store_pd(re + 9*es, t177);
_mm512_store_pd(im + 9*es, t178);
}
static const double HT_23[242] ALIGN64 = {0x1.ed037ea3d2dbcp-1,0x1.14459ad2be466p-2,0x1.b57675cf309eep-1,0x1.0a06e851db7cap-1,0x1.5d779b07cfef7p-1,0x1.763021aaa15d9p-1,0x1.d71b4a0c5a6c9p-2,0x1.c698e42f47b09p-1,0x1.a0ad8bd1e2881p-3,0x1.f54a827142577p-1,-0x1.17855b599f3b2p-4,0x1.fece70dfd3efbp-1,0x1.b57675cf309eep-1,0x1.0a06e851db7cap-1,0x1.d71b4a0c5a6c9p-2,0x1.c698e42f47b09p-1,-0x1.17855b599f3b2p-4,0x1.fece70dfd3efbp-1,-0x1.2742a4a775cfap-1,0x1.a249e0b897caap-1,-0x1.d59cb83ef99bcp-1,0x1.97f6748e524b1p-2,-0x1.fb3b3035aa6cdp-1,-0x1.16de8a4564f03p-3,0x1.5d779b07cfef7p-1,0x1.763021aaa15d9p-1,-0x1.17855b599f3b2p-4,0x1.fece70dfd3efbp-1,-0x1.8d2a07c16d46ep-1,0x1.431df5838f7f1p-1,-0x1.fb3b3035aa6cdp-1,-0x1.16de8a4564f03p-3,-0x1.2742a4a775cfbp-1,-0x1.a249e0b897ca9p-1,0x1.a0ad8bd1e2871p-3,-0x1.f54a827142578p-1,0x1.d71b4a0c5a6c9p-2,0x1.c698e42f47b09p-1,-0x1.2742a4a775cfap-1,0x1.a249e0b897caap-1,-0x1.fb3b3035aa6cdp-1,-0x1.16de8a4564f03p-3,-0x1.56eaae597c77ap-2,-0x1.e270060999287p-1,0x1.5d779b07cfef8p-1,-0x1.763021aaa15d9p-1,0x1.ed037ea3d2dbcp-1,0x1.14459ad2be466p-2,0x1.a0ad8bd1e2881p-3,0x1.f54a827142577p-1,-0x1.d59cb83ef99bcp-1,0x1.97f6748e524b1p-2,-0x1.2742a4a775cfbp-1,-0x1.a249e0b897ca9p-1,0x1.5d779b07cfef8p-1,-0x1.763021aaa15d9p-1,0x1.b57675cf309eep-1,0x1.0a06e851db7cap-1,-0x1.56eaae597c776p-2,0x1.e270060999288p-1,-0x1.17855b599f3b2p-4,0x1.fece70dfd3efbp-1,-0x1.fb3b3035aa6cdp-1,-0x1.16de8a4564f03p-3,0x1.a0ad8bd1e2871p-3,-0x1.f54a827142578p-1,0x1.ed037ea3d2dbcp-1,0x1.14459ad2be466p-2,-0x1.56eaae597c776p-2,0x1.e270060999288p-1,-0x1.d59cb83ef99bbp-1,-0x1.97f6748e524b4p-2,-0x1.56eaae597c776p-2,0x1.e270060999288p-1,-0x1.8d2a07c16d46fp-1,-0x1.431df5838f7efp-1,0x1.b57675cf309eep-1,-0x1.0a06e851db7cap-1,0x1.a0ad8bd1e2881p-3,0x1.f54a827142577p-1,-0x1.fb3b3035aa6cdp-1,-0x1.16de8a4564f03p-3,0x1.d71b4a0c5a6bep-2,-0x1.c698e42f47b0cp-1,-0x1.2742a4a775cfap-1,0x1.a249e0b897caap-1,-0x1.56eaae597c77ap-2,-0x1.e270060999287p-1,0x1.ed037ea3d2dbcp-1,0x1.14459ad2be466p-2,-0x1.8d2a07c16d46ep-1,0x1.431df5838f7f1p-1,-0x1.17855b599f394p-4,-0x1.fece70dfd3efcp-1,0x1.b57675cf309eep-1,0x1.0a06e851db7cap-1,-0x1.8d2a07c16d46ep-1,0x1.431df5838f7f1p-1,0x1.a0ad8bd1e2871p-3,-0x1.f54a827142578p-1,0x1.d71b4a0c5a6c9p-2,0x1.c698e42f47b09p-1,-0x1.d59cb83ef99bbp-1,-0x1.97f6748e524b4p-2,0x1.ed037ea3d2db9p-1,-0x1.14459ad2be477p-2,-0x1.2742a4a775cfap-1,0x1.a249e0b897caap-1,-0x1.d59cb83ef99bcp-1,0x1.97f6748e524b1p-2,0x1.5d779b07cfef8p-1,-0x1.763021aaa15d9p-1,-0x1.56eaae597c776p-2,0x1.e270060999288p-1,-0x1.17855b599f394p-4,-0x1.fece70dfd3efcp-1,0x1.d71b4a0c5a6c9p-2,0x1.c698e42f47b09p-1,-0x1.8d2a07c16d46fp-1,-0x1.431df5838f7efp-1,-0x1.fb3b3035aa6ccp-1,0x1.16de8a4564f1cp-3,0x1.ed037ea3d2db9p-1,-0x1.14459ad2be477p-2,-0x1.d59cb83ef99bcp-1,0x1.97f6748e524b1p-2,0x1.b57675cf309eep-1,-0x1.0a06e851db7cap-1,-0x1.8d2a07c16d46ep-1,0x1.431df5838f7f1p-1,0x1.5d779b07cfef8p-1,-0x1.763021aaa15d9p-1,-0x1.56eaae597c776p-2,0x1.e270060999288p-1,-0x1.2742a4a775cfap-1,0x1.a249e0b897caap-1,-0x1.8d2a07c16d46ep-1,0x1.431df5838f7f1p-1,-0x1.d59cb83ef99bcp-1,0x1.97f6748e524b1p-2,-0x1.fb3b3035aa6ccp-1,0x1.16de8a4564f1cp-3,-0x1.8d2a07c16d46fp-1,-0x1.431df5838f7efp-1,-0x1.56eaae597c77ap-2,-0x1.e270060999287p-1,0x1.a0ad8bd1e2871p-3,-0x1.f54a827142578p-1,0x1.5d779b07cfef8p-1,-0x1.763021aaa15d9p-1,0x1.ed037ea3d2db9p-1,-0x1.14459ad2be477p-2,0x1.b57675cf309eep-1,-0x1.0a06e851db7cap-1,0x1.ed037ea3d2dbcp-1,0x1.14459ad2be466p-2,0x1.d71b4a0c5a6c9p-2,0x1.c698e42f47b09p-1,-0x1.56eaae597c776p-2,0x1.e270060999288p-1,-0x1.d59cb83ef99bcp-1,0x1.97f6748e524b1p-2,0x1.a0ad8bd1e2881p-3,0x1.f54a827142577p-1,-0x1.8d2a07c16d46ep-1,0x1.431df5838f7f1p-1,-0x1.d59cb83ef99bbp-1,-0x1.97f6748e524b4p-2,-0x1.17855b599f394p-4,-0x1.fece70dfd3efcp-1,0x1.b57675cf309eep-1,-0x1.0a06e851db7cap-1,-0x1.fb3b3035aa6cdp-1,-0x1.16de8a4564f03p-3,-0x1.17855b599f394p-4,-0x1.fece70dfd3efcp-1,0x1.ed037ea3d2db9p-1,-0x1.14459ad2be477p-2,0x1.d71b4a0c5a6c9p-2,0x1.c698e42f47b09p-1,-0x1.8d2a07c16d46ep-1,0x1.431df5838f7f1p-1,0x1.d71b4a0c5a6bep-2,-0x1.c698e42f47b0cp-1,0x1.b57675cf309eep-1,0x1.0a06e851db7cap-1,-0x1.2742a4a775cfap-1,0x1.a249e0b897caap-1,-0x1.8d2a07c16d46fp-1,-0x1.431df5838f7efp-1,0x1.5d779b07cfef8p-1,-0x1.763021aaa15d9p-1,0x1.5d779b07cfef7p-1,0x1.763021aaa15d9p-1,-0x1.d59cb83ef99bcp-1,0x1.97f6748e524b1p-2,-0x1.17855b599f394p-4,-0x1.fece70dfd3efcp-1,0x1.ed037ea3d2dbcp-1,0x1.14459ad2be466p-2,-0x1.2742a4a775cfap-1,0x1.a249e0b897caap-1,-0x1.d59cb83ef99bcp-1,0x1.97f6748e524b1p-2,0x1.a0ad8bd1e2871p-3,-0x1.f54a827142578p-1,0x1.5d779b07cfef7p-1,0x1.763021aaa15d9p-1,-0x1.fb3b3035aa6ccp-1,0x1.16de8a4564f1cp-3,0x1.d71b4a0c5a6bep-2,-0x1.c698e42f47b0cp-1,-0x1.17855b599f394p-4,-0x1.fece70dfd3efcp-1,0x1.5d779b07cfef7p-1,0x1.763021aaa15d9p-1,-0x1.fb3b3035aa6cdp-1,-0x1.16de8a4564f03p-3,0x1.b57675cf309eep-1,-0x1.0a06e851db7cap-1,-0x1.56eaae597c776p-2,0x1.e270060999288p-1,0x1.ed037ea3d2dbcp-1,0x1.14459ad2be466p-2,-0x1.fb3b3035aa6ccp-1,0x1.16de8a4564f1cp-3,0x1.b57675cf309eep-1,-0x1.0a06e851db7cap-1,-0x1.2742a4a775cfap-1,0x1.a249e0b897caap-1,0x1.a0ad8bd1e2871p-3,-0x1.f54a827142578p-1,-0x1.2742a4a775cfap-1,0x1.a249e0b897caap-1,0x1.d71b4a0c5a6bep-2,-0x1.c698e42f47b0cp-1,-0x1.56eaae597c776p-2,0x1.e270060999288p-1,0x1.a0ad8bd1e2871p-3,-0x1.f54a827142578p-1,-0x1.17855b599f3b2p-4,0x1.fece70dfd3efbp-1};
static double HS_23[11][32] ALIGN64;
static void dftp23_v(double* re, double* im, long es){
__m512d t1 = _mm512_load_pd(re + 0);
__m512d t2 = _mm512_load_pd(im + 0);
__m512d t3 = _mm512_load_pd(re + 1*es);
__m512d t4 = _mm512_load_pd(im + 1*es);
__m512d t5 = _mm512_load_pd(re + 22*es);
__m512d t6 = _mm512_load_pd(im + 22*es);
__m512d t7 = _mm512_add_pd(t3, t5);
__m512d t8 = _mm512_add_pd(t4, t6);
__m512d t9 = _mm512_sub_pd(t3, t5);
__m512d t10 = _mm512_sub_pd(t4, t6);
_mm512_store_pd(HS_23[0], t7); _mm512_store_pd(HS_23[0]+8, t8);
_mm512_store_pd(HS_23[0]+16, t9); _mm512_store_pd(HS_23[0]+24, t10);
__m512d t11 = _mm512_load_pd(re + 2*es);
__m512d t12 = _mm512_load_pd(im + 2*es);
__m512d t13 = _mm512_load_pd(re + 21*es);
__m512d t14 = _mm512_load_pd(im + 21*es);
__m512d t15 = _mm512_add_pd(t11, t13);
__m512d t16 = _mm512_add_pd(t12, t14);
__m512d t17 = _mm512_sub_pd(t11, t13);
__m512d t18 = _mm512_sub_pd(t12, t14);
_mm512_store_pd(HS_23[1], t15); _mm512_store_pd(HS_23[1]+8, t16);
_mm512_store_pd(HS_23[1]+16, t17); _mm512_store_pd(HS_23[1]+24, t18);
__m512d t19 = _mm512_add_pd(t7, t15);
__m512d t20 = _mm512_add_pd(t8, t16);
__m512d t21 = _mm512_load_pd(re + 3*es);
__m512d t22 = _mm512_load_pd(im + 3*es);
__m512d t23 = _mm512_load_pd(re + 20*es);
__m512d t24 = _mm512_load_pd(im + 20*es);
__m512d t25 = _mm512_add_pd(t21, t23);
__m512d t26 = _mm512_add_pd(t22, t24);
__m512d t27 = _mm512_sub_pd(t21, t23);
__m512d t28 = _mm512_sub_pd(t22, t24);
_mm512_store_pd(HS_23[2], t25); _mm512_store_pd(HS_23[2]+8, t26);
_mm512_store_pd(HS_23[2]+16, t27); _mm512_store_pd(HS_23[2]+24, t28);
__m512d t29 = _mm512_add_pd(t19, t25);
__m512d t30 = _mm512_add_pd(t20, t26);
__m512d t31 = _mm512_load_pd(re + 4*es);
__m512d t32 = _mm512_load_pd(im + 4*es);
__m512d t33 = _mm512_load_pd(re + 19*es);
__m512d t34 = _mm512_load_pd(im + 19*es);
__m512d t35 = _mm512_add_pd(t31, t33);
__m512d t36 = _mm512_add_pd(t32, t34);
__m512d t37 = _mm512_sub_pd(t31, t33);
__m512d t38 = _mm512_sub_pd(t32, t34);
_mm512_store_pd(HS_23[3], t35); _mm512_store_pd(HS_23[3]+8, t36);
_mm512_store_pd(HS_23[3]+16, t37); _mm512_store_pd(HS_23[3]+24, t38);
__m512d t39 = _mm512_add_pd(t29, t35);
__m512d t40 = _mm512_add_pd(t30, t36);
__m512d t41 = _mm512_load_pd(re + 5*es);
__m512d t42 = _mm512_load_pd(im + 5*es);
__m512d t43 = _mm512_load_pd(re + 18*es);
__m512d t44 = _mm512_load_pd(im + 18*es);
__m512d t45 = _mm512_add_pd(t41, t43);
__m512d t46 = _mm512_add_pd(t42, t44);
__m512d t47 = _mm512_sub_pd(t41, t43);
__m512d t48 = _mm512_sub_pd(t42, t44);
_mm512_store_pd(HS_23[4], t45); _mm512_store_pd(HS_23[4]+8, t46);
_mm512_store_pd(HS_23[4]+16, t47); _mm512_store_pd(HS_23[4]+24, t48);
__m512d t49 = _mm512_add_pd(t39, t45);
__m512d t50 = _mm512_add_pd(t40, t46);
__m512d t51 = _mm512_load_pd(re + 6*es);
__m512d t52 = _mm512_load_pd(im + 6*es);
__m512d t53 = _mm512_load_pd(re + 17*es);
__m512d t54 = _mm512_load_pd(im + 17*es);
__m512d t55 = _mm512_add_pd(t51, t53);
__m512d t56 = _mm512_add_pd(t52, t54);
__m512d t57 = _mm512_sub_pd(t51, t53);
__m512d t58 = _mm512_sub_pd(t52, t54);
_mm512_store_pd(HS_23[5], t55); _mm512_store_pd(HS_23[5]+8, t56);
_mm512_store_pd(HS_23[5]+16, t57); _mm512_store_pd(HS_23[5]+24, t58);
__m512d t59 = _mm512_add_pd(t49, t55);
__m512d t60 = _mm512_add_pd(t50, t56);
__m512d t61 = _mm512_load_pd(re + 7*es);
__m512d t62 = _mm512_load_pd(im + 7*es);
__m512d t63 = _mm512_load_pd(re + 16*es);
__m512d t64 = _mm512_load_pd(im + 16*es);
__m512d t65 = _mm512_add_pd(t61, t63);
__m512d t66 = _mm512_add_pd(t62, t64);
__m512d t67 = _mm512_sub_pd(t61, t63);
__m512d t68 = _mm512_sub_pd(t62, t64);
_mm512_store_pd(HS_23[6], t65); _mm512_store_pd(HS_23[6]+8, t66);
_mm512_store_pd(HS_23[6]+16, t67); _mm512_store_pd(HS_23[6]+24, t68);
__m512d t69 = _mm512_add_pd(t59, t65);
__m512d t70 = _mm512_add_pd(t60, t66);
__m512d t71 = _mm512_load_pd(re + 8*es);
__m512d t72 = _mm512_load_pd(im + 8*es);
__m512d t73 = _mm512_load_pd(re + 15*es);
__m512d t74 = _mm512_load_pd(im + 15*es);
__m512d t75 = _mm512_add_pd(t71, t73);
__m512d t76 = _mm512_add_pd(t72, t74);
__m512d t77 = _mm512_sub_pd(t71, t73);
__m512d t78 = _mm512_sub_pd(t72, t74);
_mm512_store_pd(HS_23[7], t75); _mm512_store_pd(HS_23[7]+8, t76);
_mm512_store_pd(HS_23[7]+16, t77); _mm512_store_pd(HS_23[7]+24, t78);
__m512d t79 = _mm512_add_pd(t69, t75);
__m512d t80 = _mm512_add_pd(t70, t76);
__m512d t81 = _mm512_load_pd(re + 9*es);
__m512d t82 = _mm512_load_pd(im + 9*es);
__m512d t83 = _mm512_load_pd(re + 14*es);
__m512d t84 = _mm512_load_pd(im + 14*es);
__m512d t85 = _mm512_add_pd(t81, t83);
__m512d t86 = _mm512_add_pd(t82, t84);
__m512d t87 = _mm512_sub_pd(t81, t83);
__m512d t88 = _mm512_sub_pd(t82, t84);
_mm512_store_pd(HS_23[8], t85); _mm512_store_pd(HS_23[8]+8, t86);
_mm512_store_pd(HS_23[8]+16, t87); _mm512_store_pd(HS_23[8]+24, t88);
__m512d t89 = _mm512_add_pd(t79, t85);
__m512d t90 = _mm512_add_pd(t80, t86);
__m512d t91 = _mm512_load_pd(re + 10*es);
__m512d t92 = _mm512_load_pd(im + 10*es);
__m512d t93 = _mm512_load_pd(re + 13*es);
__m512d t94 = _mm512_load_pd(im + 13*es);
__m512d t95 = _mm512_add_pd(t91, t93);
__m512d t96 = _mm512_add_pd(t92, t94);
__m512d t97 = _mm512_sub_pd(t91, t93);
__m512d t98 = _mm512_sub_pd(t92, t94);
_mm512_store_pd(HS_23[9], t95); _mm512_store_pd(HS_23[9]+8, t96);
_mm512_store_pd(HS_23[9]+16, t97); _mm512_store_pd(HS_23[9]+24, t98);
__m512d t99 = _mm512_add_pd(t89, t95);
__m512d t100 = _mm512_add_pd(t90, t96);
__m512d t101 = _mm512_load_pd(re + 11*es);
__m512d t102 = _mm512_load_pd(im + 11*es);
__m512d t103 = _mm512_load_pd(re + 12*es);
__m512d t104 = _mm512_load_pd(im + 12*es);
__m512d t105 = _mm512_add_pd(t101, t103);
__m512d t106 = _mm512_add_pd(t102, t104);
__m512d t107 = _mm512_sub_pd(t101, t103);
__m512d t108 = _mm512_sub_pd(t102, t104);
_mm512_store_pd(HS_23[10], t105); _mm512_store_pd(HS_23[10]+8, t106);
_mm512_store_pd(HS_23[10]+16, t107); _mm512_store_pd(HS_23[10]+24, t108);
__m512d t109 = _mm512_add_pd(t99, t105);
__m512d t110 = _mm512_add_pd(t100, t106);
__m512d t111 = _mm512_add_pd(t109, t1);
__m512d t112 = _mm512_add_pd(t110, t2);
_mm512_store_pd(re + 0, t111);
_mm512_store_pd(im + 0, t112);
__asm__ volatile("" ::: "memory");
__asm__ volatile("" ::: "memory");
__m512d accAr1 = t1, accAi1 = t2, accBr1, accBi1;
__m512d accAr2 = t1, accAi2 = t2, accBr2, accBi2;
__m512d accAr3 = t1, accAi3 = t2, accBr3, accBi3;
__m512d accAr4 = t1, accAi4 = t2, accBr4, accBi4;
__m512d accAr5 = t1, accAi5 = t2, accBr5, accBi5;
__m512d accAr6 = t1, accAi6 = t2, accBr6, accBi6;
__m512d t113 = _mm512_load_pd(HS_23[0]);
__m512d t114 = _mm512_load_pd(HS_23[0]+8);
__m512d t115 = _mm512_load_pd(HS_23[0]+16);
__m512d t116 = _mm512_load_pd(HS_23[0]+24);
__m512d bc1_1_c, bc1_1_s;
BCASTV(bc1_1_c, HT_23[0]); BCASTV(bc1_1_s, HT_23[1]);
accAr1 = _mm512_fmadd_pd(bc1_1_c, t113, accAr1);
accAi1 = _mm512_fmadd_pd(bc1_1_c, t114, accAi1);
accBr1 = _mm512_mul_pd(bc1_1_s, t116);
accBi1 = _mm512_mul_pd(bc1_1_s, t115);
__m512d bc1_2_c, bc1_2_s;
BCASTV(bc1_2_c, HT_23[2]); BCASTV(bc1_2_s, HT_23[3]);
accAr2 = _mm512_fmadd_pd(bc1_2_c, t113, accAr2);
accAi2 = _mm512_fmadd_pd(bc1_2_c, t114, accAi2);
accBr2 = _mm512_mul_pd(bc1_2_s, t116);
accBi2 = _mm512_mul_pd(bc1_2_s, t115);
__m512d bc1_3_c, bc1_3_s;
BCASTV(bc1_3_c, HT_23[4]); BCASTV(bc1_3_s, HT_23[5]);
accAr3 = _mm512_fmadd_pd(bc1_3_c, t113, accAr3);
accAi3 = _mm512_fmadd_pd(bc1_3_c, t114, accAi3);
accBr3 = _mm512_mul_pd(bc1_3_s, t116);
accBi3 = _mm512_mul_pd(bc1_3_s, t115);
__m512d bc1_4_c, bc1_4_s;
BCASTV(bc1_4_c, HT_23[6]); BCASTV(bc1_4_s, HT_23[7]);
accAr4 = _mm512_fmadd_pd(bc1_4_c, t113, accAr4);
accAi4 = _mm512_fmadd_pd(bc1_4_c, t114, accAi4);
accBr4 = _mm512_mul_pd(bc1_4_s, t116);
accBi4 = _mm512_mul_pd(bc1_4_s, t115);
__m512d bc1_5_c, bc1_5_s;
BCASTV(bc1_5_c, HT_23[8]); BCASTV(bc1_5_s, HT_23[9]);
accAr5 = _mm512_fmadd_pd(bc1_5_c, t113, accAr5);
accAi5 = _mm512_fmadd_pd(bc1_5_c, t114, accAi5);
accBr5 = _mm512_mul_pd(bc1_5_s, t116);
accBi5 = _mm512_mul_pd(bc1_5_s, t115);
__m512d bc1_6_c, bc1_6_s;
BCASTV(bc1_6_c, HT_23[10]); BCASTV(bc1_6_s, HT_23[11]);
accAr6 = _mm512_fmadd_pd(bc1_6_c, t113, accAr6);
accAi6 = _mm512_fmadd_pd(bc1_6_c, t114, accAi6);
accBr6 = _mm512_mul_pd(bc1_6_s, t116);
accBi6 = _mm512_mul_pd(bc1_6_s, t115);
__m512d t117 = _mm512_load_pd(HS_23[1]);
__m512d t118 = _mm512_load_pd(HS_23[1]+8);
__m512d t119 = _mm512_load_pd(HS_23[1]+16);
__m512d t120 = _mm512_load_pd(HS_23[1]+24);
__m512d bc2_1_c, bc2_1_s;
BCASTV(bc2_1_c, HT_23[12]); BCASTV(bc2_1_s, HT_23[13]);
accAr1 = _mm512_fmadd_pd(bc2_1_c, t117, accAr1);
accAi1 = _mm512_fmadd_pd(bc2_1_c, t118, accAi1);
accBr1 = _mm512_fmadd_pd(bc2_1_s, t120, accBr1);
accBi1 = _mm512_fmadd_pd(bc2_1_s, t119, accBi1);
__m512d bc2_2_c, bc2_2_s;
BCASTV(bc2_2_c, HT_23[14]); BCASTV(bc2_2_s, HT_23[15]);
accAr2 = _mm512_fmadd_pd(bc2_2_c, t117, accAr2);
accAi2 = _mm512_fmadd_pd(bc2_2_c, t118, accAi2);
accBr2 = _mm512_fmadd_pd(bc2_2_s, t120, accBr2);
accBi2 = _mm512_fmadd_pd(bc2_2_s, t119, accBi2);
__m512d bc2_3_c, bc2_3_s;
BCASTV(bc2_3_c, HT_23[16]); BCASTV(bc2_3_s, HT_23[17]);
accAr3 = _mm512_fmadd_pd(bc2_3_c, t117, accAr3);
accAi3 = _mm512_fmadd_pd(bc2_3_c, t118, accAi3);
accBr3 = _mm512_fmadd_pd(bc2_3_s, t120, accBr3);
accBi3 = _mm512_fmadd_pd(bc2_3_s, t119, accBi3);
__m512d bc2_4_c, bc2_4_s;
BCASTV(bc2_4_c, HT_23[18]); BCASTV(bc2_4_s, HT_23[19]);
accAr4 = _mm512_fmadd_pd(bc2_4_c, t117, accAr4);
accAi4 = _mm512_fmadd_pd(bc2_4_c, t118, accAi4);
accBr4 = _mm512_fmadd_pd(bc2_4_s, t120, accBr4);
accBi4 = _mm512_fmadd_pd(bc2_4_s, t119, accBi4);
__m512d bc2_5_c, bc2_5_s;
BCASTV(bc2_5_c, HT_23[20]); BCASTV(bc2_5_s, HT_23[21]);
accAr5 = _mm512_fmadd_pd(bc2_5_c, t117, accAr5);
accAi5 = _mm512_fmadd_pd(bc2_5_c, t118, accAi5);
accBr5 = _mm512_fmadd_pd(bc2_5_s, t120, accBr5);
accBi5 = _mm512_fmadd_pd(bc2_5_s, t119, accBi5);
__m512d bc2_6_c, bc2_6_s;
BCASTV(bc2_6_c, HT_23[22]); BCASTV(bc2_6_s, HT_23[23]);
accAr6 = _mm512_fmadd_pd(bc2_6_c, t117, accAr6);
accAi6 = _mm512_fmadd_pd(bc2_6_c, t118, accAi6);
accBr6 = _mm512_fmadd_pd(bc2_6_s, t120, accBr6);
accBi6 = _mm512_fmadd_pd(bc2_6_s, t119, accBi6);
__m512d t121 = _mm512_load_pd(HS_23[2]);
__m512d t122 = _mm512_load_pd(HS_23[2]+8);
__m512d t123 = _mm512_load_pd(HS_23[2]+16);
__m512d t124 = _mm512_load_pd(HS_23[2]+24);
__m512d bc3_1_c, bc3_1_s;
BCASTV(bc3_1_c, HT_23[24]); BCASTV(bc3_1_s, HT_23[25]);
accAr1 = _mm512_fmadd_pd(bc3_1_c, t121, accAr1);
accAi1 = _mm512_fmadd_pd(bc3_1_c, t122, accAi1);
accBr1 = _mm512_fmadd_pd(bc3_1_s, t124, accBr1);
accBi1 = _mm512_fmadd_pd(bc3_1_s, t123, accBi1);
__m512d bc3_2_c, bc3_2_s;
BCASTV(bc3_2_c, HT_23[26]); BCASTV(bc3_2_s, HT_23[27]);
accAr2 = _mm512_fmadd_pd(bc3_2_c, t121, accAr2);
accAi2 = _mm512_fmadd_pd(bc3_2_c, t122, accAi2);
accBr2 = _mm512_fmadd_pd(bc3_2_s, t124, accBr2);
accBi2 = _mm512_fmadd_pd(bc3_2_s, t123, accBi2);
__m512d bc3_3_c, bc3_3_s;
BCASTV(bc3_3_c, HT_23[28]); BCASTV(bc3_3_s, HT_23[29]);
accAr3 = _mm512_fmadd_pd(bc3_3_c, t121, accAr3);
accAi3 = _mm512_fmadd_pd(bc3_3_c, t122, accAi3);
accBr3 = _mm512_fmadd_pd(bc3_3_s, t124, accBr3);
accBi3 = _mm512_fmadd_pd(bc3_3_s, t123, accBi3);
__m512d bc3_4_c, bc3_4_s;
BCASTV(bc3_4_c, HT_23[30]); BCASTV(bc3_4_s, HT_23[31]);
accAr4 = _mm512_fmadd_pd(bc3_4_c, t121, accAr4);
accAi4 = _mm512_fmadd_pd(bc3_4_c, t122, accAi4);
accBr4 = _mm512_fmadd_pd(bc3_4_s, t124, accBr4);
accBi4 = _mm512_fmadd_pd(bc3_4_s, t123, accBi4);
__m512d bc3_5_c, bc3_5_s;
BCASTV(bc3_5_c, HT_23[32]); BCASTV(bc3_5_s, HT_23[33]);
accAr5 = _mm512_fmadd_pd(bc3_5_c, t121, accAr5);
accAi5 = _mm512_fmadd_pd(bc3_5_c, t122, accAi5);
accBr5 = _mm512_fmadd_pd(bc3_5_s, t124, accBr5);
accBi5 = _mm512_fmadd_pd(bc3_5_s, t123, accBi5);
__m512d bc3_6_c, bc3_6_s;
BCASTV(bc3_6_c, HT_23[34]); BCASTV(bc3_6_s, HT_23[35]);
accAr6 = _mm512_fmadd_pd(bc3_6_c, t121, accAr6);
accAi6 = _mm512_fmadd_pd(bc3_6_c, t122, accAi6);
accBr6 = _mm512_fmadd_pd(bc3_6_s, t124, accBr6);
accBi6 = _mm512_fmadd_pd(bc3_6_s, t123, accBi6);
__m512d t125 = _mm512_load_pd(HS_23[3]);
__m512d t126 = _mm512_load_pd(HS_23[3]+8);
__m512d t127 = _mm512_load_pd(HS_23[3]+16);
__m512d t128 = _mm512_load_pd(HS_23[3]+24);
__m512d bc4_1_c, bc4_1_s;
BCASTV(bc4_1_c, HT_23[36]); BCASTV(bc4_1_s, HT_23[37]);
accAr1 = _mm512_fmadd_pd(bc4_1_c, t125, accAr1);
accAi1 = _mm512_fmadd_pd(bc4_1_c, t126, accAi1);
accBr1 = _mm512_fmadd_pd(bc4_1_s, t128, accBr1);
accBi1 = _mm512_fmadd_pd(bc4_1_s, t127, accBi1);
__m512d bc4_2_c, bc4_2_s;
BCASTV(bc4_2_c, HT_23[38]); BCASTV(bc4_2_s, HT_23[39]);
accAr2 = _mm512_fmadd_pd(bc4_2_c, t125, accAr2);
accAi2 = _mm512_fmadd_pd(bc4_2_c, t126, accAi2);
accBr2 = _mm512_fmadd_pd(bc4_2_s, t128, accBr2);
accBi2 = _mm512_fmadd_pd(bc4_2_s, t127, accBi2);
__m512d bc4_3_c, bc4_3_s;
BCASTV(bc4_3_c, HT_23[40]); BCASTV(bc4_3_s, HT_23[41]);
accAr3 = _mm512_fmadd_pd(bc4_3_c, t125, accAr3);
accAi3 = _mm512_fmadd_pd(bc4_3_c, t126, accAi3);
accBr3 = _mm512_fmadd_pd(bc4_3_s, t128, accBr3);
accBi3 = _mm512_fmadd_pd(bc4_3_s, t127, accBi3);
__m512d bc4_4_c, bc4_4_s;
BCASTV(bc4_4_c, HT_23[42]); BCASTV(bc4_4_s, HT_23[43]);
accAr4 = _mm512_fmadd_pd(bc4_4_c, t125, accAr4);
accAi4 = _mm512_fmadd_pd(bc4_4_c, t126, accAi4);
accBr4 = _mm512_fmadd_pd(bc4_4_s, t128, accBr4);
accBi4 = _mm512_fmadd_pd(bc4_4_s, t127, accBi4);
__m512d bc4_5_c, bc4_5_s;
BCASTV(bc4_5_c, HT_23[44]); BCASTV(bc4_5_s, HT_23[45]);
accAr5 = _mm512_fmadd_pd(bc4_5_c, t125, accAr5);
accAi5 = _mm512_fmadd_pd(bc4_5_c, t126, accAi5);
accBr5 = _mm512_fmadd_pd(bc4_5_s, t128, accBr5);
accBi5 = _mm512_fmadd_pd(bc4_5_s, t127, accBi5);
__m512d bc4_6_c, bc4_6_s;
BCASTV(bc4_6_c, HT_23[46]); BCASTV(bc4_6_s, HT_23[47]);
accAr6 = _mm512_fmadd_pd(bc4_6_c, t125, accAr6);
accAi6 = _mm512_fmadd_pd(bc4_6_c, t126, accAi6);
accBr6 = _mm512_fmadd_pd(bc4_6_s, t128, accBr6);
accBi6 = _mm512_fmadd_pd(bc4_6_s, t127, accBi6);
__m512d t129 = _mm512_load_pd(HS_23[4]);
__m512d t130 = _mm512_load_pd(HS_23[4]+8);
__m512d t131 = _mm512_load_pd(HS_23[4]+16);
__m512d t132 = _mm512_load_pd(HS_23[4]+24);
__m512d bc5_1_c, bc5_1_s;
BCASTV(bc5_1_c, HT_23[48]); BCASTV(bc5_1_s, HT_23[49]);
accAr1 = _mm512_fmadd_pd(bc5_1_c, t129, accAr1);
accAi1 = _mm512_fmadd_pd(bc5_1_c, t130, accAi1);
accBr1 = _mm512_fmadd_pd(bc5_1_s, t132, accBr1);
accBi1 = _mm512_fmadd_pd(bc5_1_s, t131, accBi1);
__m512d bc5_2_c, bc5_2_s;
BCASTV(bc5_2_c, HT_23[50]); BCASTV(bc5_2_s, HT_23[51]);
accAr2 = _mm512_fmadd_pd(bc5_2_c, t129, accAr2);
accAi2 = _mm512_fmadd_pd(bc5_2_c, t130, accAi2);
accBr2 = _mm512_fmadd_pd(bc5_2_s, t132, accBr2);
accBi2 = _mm512_fmadd_pd(bc5_2_s, t131, accBi2);
__m512d bc5_3_c, bc5_3_s;
BCASTV(bc5_3_c, HT_23[52]); BCASTV(bc5_3_s, HT_23[53]);
accAr3 = _mm512_fmadd_pd(bc5_3_c, t129, accAr3);
accAi3 = _mm512_fmadd_pd(bc5_3_c, t130, accAi3);
accBr3 = _mm512_fmadd_pd(bc5_3_s, t132, accBr3);
accBi3 = _mm512_fmadd_pd(bc5_3_s, t131, accBi3);
__m512d bc5_4_c, bc5_4_s;
BCASTV(bc5_4_c, HT_23[54]); BCASTV(bc5_4_s, HT_23[55]);
accAr4 = _mm512_fmadd_pd(bc5_4_c, t129, accAr4);
accAi4 = _mm512_fmadd_pd(bc5_4_c, t130, accAi4);
accBr4 = _mm512_fmadd_pd(bc5_4_s, t132, accBr4);
accBi4 = _mm512_fmadd_pd(bc5_4_s, t131, accBi4);
__m512d bc5_5_c, bc5_5_s;
BCASTV(bc5_5_c, HT_23[56]); BCASTV(bc5_5_s, HT_23[57]);
accAr5 = _mm512_fmadd_pd(bc5_5_c, t129, accAr5);
accAi5 = _mm512_fmadd_pd(bc5_5_c, t130, accAi5);
accBr5 = _mm512_fmadd_pd(bc5_5_s, t132, accBr5);
accBi5 = _mm512_fmadd_pd(bc5_5_s, t131, accBi5);
__m512d bc5_6_c, bc5_6_s;
BCASTV(bc5_6_c, HT_23[58]); BCASTV(bc5_6_s, HT_23[59]);
accAr6 = _mm512_fmadd_pd(bc5_6_c, t129, accAr6);
accAi6 = _mm512_fmadd_pd(bc5_6_c, t130, accAi6);
accBr6 = _mm512_fmadd_pd(bc5_6_s, t132, accBr6);
accBi6 = _mm512_fmadd_pd(bc5_6_s, t131, accBi6);
__m512d t133 = _mm512_load_pd(HS_23[5]);
__m512d t134 = _mm512_load_pd(HS_23[5]+8);
__m512d t135 = _mm512_load_pd(HS_23[5]+16);
__m512d t136 = _mm512_load_pd(HS_23[5]+24);
__m512d bc6_1_c, bc6_1_s;
BCASTV(bc6_1_c, HT_23[60]); BCASTV(bc6_1_s, HT_23[61]);
accAr1 = _mm512_fmadd_pd(bc6_1_c, t133, accAr1);
accAi1 = _mm512_fmadd_pd(bc6_1_c, t134, accAi1);
accBr1 = _mm512_fmadd_pd(bc6_1_s, t136, accBr1);
accBi1 = _mm512_fmadd_pd(bc6_1_s, t135, accBi1);
__m512d bc6_2_c, bc6_2_s;
BCASTV(bc6_2_c, HT_23[62]); BCASTV(bc6_2_s, HT_23[63]);
accAr2 = _mm512_fmadd_pd(bc6_2_c, t133, accAr2);
accAi2 = _mm512_fmadd_pd(bc6_2_c, t134, accAi2);
accBr2 = _mm512_fmadd_pd(bc6_2_s, t136, accBr2);
accBi2 = _mm512_fmadd_pd(bc6_2_s, t135, accBi2);
__m512d bc6_3_c, bc6_3_s;
BCASTV(bc6_3_c, HT_23[64]); BCASTV(bc6_3_s, HT_23[65]);
accAr3 = _mm512_fmadd_pd(bc6_3_c, t133, accAr3);
accAi3 = _mm512_fmadd_pd(bc6_3_c, t134, accAi3);
accBr3 = _mm512_fmadd_pd(bc6_3_s, t136, accBr3);
accBi3 = _mm512_fmadd_pd(bc6_3_s, t135, accBi3);
__m512d bc6_4_c, bc6_4_s;
BCASTV(bc6_4_c, HT_23[66]); BCASTV(bc6_4_s, HT_23[67]);
accAr4 = _mm512_fmadd_pd(bc6_4_c, t133, accAr4);
accAi4 = _mm512_fmadd_pd(bc6_4_c, t134, accAi4);
accBr4 = _mm512_fmadd_pd(bc6_4_s, t136, accBr4);
accBi4 = _mm512_fmadd_pd(bc6_4_s, t135, accBi4);
__m512d bc6_5_c, bc6_5_s;
BCASTV(bc6_5_c, HT_23[68]); BCASTV(bc6_5_s, HT_23[69]);
accAr5 = _mm512_fmadd_pd(bc6_5_c, t133, accAr5);
accAi5 = _mm512_fmadd_pd(bc6_5_c, t134, accAi5);
accBr5 = _mm512_fmadd_pd(bc6_5_s, t136, accBr5);
accBi5 = _mm512_fmadd_pd(bc6_5_s, t135, accBi5);
__m512d bc6_6_c, bc6_6_s;
BCASTV(bc6_6_c, HT_23[70]); BCASTV(bc6_6_s, HT_23[71]);
accAr6 = _mm512_fmadd_pd(bc6_6_c, t133, accAr6);
accAi6 = _mm512_fmadd_pd(bc6_6_c, t134, accAi6);
accBr6 = _mm512_fmadd_pd(bc6_6_s, t136, accBr6);
accBi6 = _mm512_fmadd_pd(bc6_6_s, t135, accBi6);
__m512d t137 = _mm512_load_pd(HS_23[6]);
__m512d t138 = _mm512_load_pd(HS_23[6]+8);
__m512d t139 = _mm512_load_pd(HS_23[6]+16);
__m512d t140 = _mm512_load_pd(HS_23[6]+24);
__m512d bc7_1_c, bc7_1_s;
BCASTV(bc7_1_c, HT_23[72]); BCASTV(bc7_1_s, HT_23[73]);
accAr1 = _mm512_fmadd_pd(bc7_1_c, t137, accAr1);
accAi1 = _mm512_fmadd_pd(bc7_1_c, t138, accAi1);
accBr1 = _mm512_fmadd_pd(bc7_1_s, t140, accBr1);
accBi1 = _mm512_fmadd_pd(bc7_1_s, t139, accBi1);
__m512d bc7_2_c, bc7_2_s;
BCASTV(bc7_2_c, HT_23[74]); BCASTV(bc7_2_s, HT_23[75]);
accAr2 = _mm512_fmadd_pd(bc7_2_c, t137, accAr2);
accAi2 = _mm512_fmadd_pd(bc7_2_c, t138, accAi2);
accBr2 = _mm512_fmadd_pd(bc7_2_s, t140, accBr2);
accBi2 = _mm512_fmadd_pd(bc7_2_s, t139, accBi2);
__m512d bc7_3_c, bc7_3_s;
BCASTV(bc7_3_c, HT_23[76]); BCASTV(bc7_3_s, HT_23[77]);
accAr3 = _mm512_fmadd_pd(bc7_3_c, t137, accAr3);
accAi3 = _mm512_fmadd_pd(bc7_3_c, t138, accAi3);
accBr3 = _mm512_fmadd_pd(bc7_3_s, t140, accBr3);
accBi3 = _mm512_fmadd_pd(bc7_3_s, t139, accBi3);
__m512d bc7_4_c, bc7_4_s;
BCASTV(bc7_4_c, HT_23[78]); BCASTV(bc7_4_s, HT_23[79]);
accAr4 = _mm512_fmadd_pd(bc7_4_c, t137, accAr4);
accAi4 = _mm512_fmadd_pd(bc7_4_c, t138, accAi4);
accBr4 = _mm512_fmadd_pd(bc7_4_s, t140, accBr4);
accBi4 = _mm512_fmadd_pd(bc7_4_s, t139, accBi4);
__m512d bc7_5_c, bc7_5_s;
BCASTV(bc7_5_c, HT_23[80]); BCASTV(bc7_5_s, HT_23[81]);
accAr5 = _mm512_fmadd_pd(bc7_5_c, t137, accAr5);
accAi5 = _mm512_fmadd_pd(bc7_5_c, t138, accAi5);
accBr5 = _mm512_fmadd_pd(bc7_5_s, t140, accBr5);
accBi5 = _mm512_fmadd_pd(bc7_5_s, t139, accBi5);
__m512d bc7_6_c, bc7_6_s;
BCASTV(bc7_6_c, HT_23[82]); BCASTV(bc7_6_s, HT_23[83]);
accAr6 = _mm512_fmadd_pd(bc7_6_c, t137, accAr6);
accAi6 = _mm512_fmadd_pd(bc7_6_c, t138, accAi6);
accBr6 = _mm512_fmadd_pd(bc7_6_s, t140, accBr6);
accBi6 = _mm512_fmadd_pd(bc7_6_s, t139, accBi6);
__m512d t141 = _mm512_load_pd(HS_23[7]);
__m512d t142 = _mm512_load_pd(HS_23[7]+8);
__m512d t143 = _mm512_load_pd(HS_23[7]+16);
__m512d t144 = _mm512_load_pd(HS_23[7]+24);
__m512d bc8_1_c, bc8_1_s;
BCASTV(bc8_1_c, HT_23[84]); BCASTV(bc8_1_s, HT_23[85]);
accAr1 = _mm512_fmadd_pd(bc8_1_c, t141, accAr1);
accAi1 = _mm512_fmadd_pd(bc8_1_c, t142, accAi1);
accBr1 = _mm512_fmadd_pd(bc8_1_s, t144, accBr1);
accBi1 = _mm512_fmadd_pd(bc8_1_s, t143, accBi1);
__m512d bc8_2_c, bc8_2_s;
BCASTV(bc8_2_c, HT_23[86]); BCASTV(bc8_2_s, HT_23[87]);
accAr2 = _mm512_fmadd_pd(bc8_2_c, t141, accAr2);
accAi2 = _mm512_fmadd_pd(bc8_2_c, t142, accAi2);
accBr2 = _mm512_fmadd_pd(bc8_2_s, t144, accBr2);
accBi2 = _mm512_fmadd_pd(bc8_2_s, t143, accBi2);
__m512d bc8_3_c, bc8_3_s;
BCASTV(bc8_3_c, HT_23[88]); BCASTV(bc8_3_s, HT_23[89]);
accAr3 = _mm512_fmadd_pd(bc8_3_c, t141, accAr3);
accAi3 = _mm512_fmadd_pd(bc8_3_c, t142, accAi3);
accBr3 = _mm512_fmadd_pd(bc8_3_s, t144, accBr3);
accBi3 = _mm512_fmadd_pd(bc8_3_s, t143, accBi3);
__m512d bc8_4_c, bc8_4_s;
BCASTV(bc8_4_c, HT_23[90]); BCASTV(bc8_4_s, HT_23[91]);
accAr4 = _mm512_fmadd_pd(bc8_4_c, t141, accAr4);
accAi4 = _mm512_fmadd_pd(bc8_4_c, t142, accAi4);
accBr4 = _mm512_fmadd_pd(bc8_4_s, t144, accBr4);
accBi4 = _mm512_fmadd_pd(bc8_4_s, t143, accBi4);
__m512d bc8_5_c, bc8_5_s;
BCASTV(bc8_5_c, HT_23[92]); BCASTV(bc8_5_s, HT_23[93]);
accAr5 = _mm512_fmadd_pd(bc8_5_c, t141, accAr5);
accAi5 = _mm512_fmadd_pd(bc8_5_c, t142, accAi5);
accBr5 = _mm512_fmadd_pd(bc8_5_s, t144, accBr5);
accBi5 = _mm512_fmadd_pd(bc8_5_s, t143, accBi5);
__m512d bc8_6_c, bc8_6_s;
BCASTV(bc8_6_c, HT_23[94]); BCASTV(bc8_6_s, HT_23[95]);
accAr6 = _mm512_fmadd_pd(bc8_6_c, t141, accAr6);
accAi6 = _mm512_fmadd_pd(bc8_6_c, t142, accAi6);
accBr6 = _mm512_fmadd_pd(bc8_6_s, t144, accBr6);
accBi6 = _mm512_fmadd_pd(bc8_6_s, t143, accBi6);
__m512d t145 = _mm512_load_pd(HS_23[8]);
__m512d t146 = _mm512_load_pd(HS_23[8]+8);
__m512d t147 = _mm512_load_pd(HS_23[8]+16);
__m512d t148 = _mm512_load_pd(HS_23[8]+24);
__m512d bc9_1_c, bc9_1_s;
BCASTV(bc9_1_c, HT_23[96]); BCASTV(bc9_1_s, HT_23[97]);
accAr1 = _mm512_fmadd_pd(bc9_1_c, t145, accAr1);
accAi1 = _mm512_fmadd_pd(bc9_1_c, t146, accAi1);
accBr1 = _mm512_fmadd_pd(bc9_1_s, t148, accBr1);
accBi1 = _mm512_fmadd_pd(bc9_1_s, t147, accBi1);
__m512d bc9_2_c, bc9_2_s;
BCASTV(bc9_2_c, HT_23[98]); BCASTV(bc9_2_s, HT_23[99]);
accAr2 = _mm512_fmadd_pd(bc9_2_c, t145, accAr2);
accAi2 = _mm512_fmadd_pd(bc9_2_c, t146, accAi2);
accBr2 = _mm512_fmadd_pd(bc9_2_s, t148, accBr2);
accBi2 = _mm512_fmadd_pd(bc9_2_s, t147, accBi2);
__m512d bc9_3_c, bc9_3_s;
BCASTV(bc9_3_c, HT_23[100]); BCASTV(bc9_3_s, HT_23[101]);
accAr3 = _mm512_fmadd_pd(bc9_3_c, t145, accAr3);
accAi3 = _mm512_fmadd_pd(bc9_3_c, t146, accAi3);
accBr3 = _mm512_fmadd_pd(bc9_3_s, t148, accBr3);
accBi3 = _mm512_fmadd_pd(bc9_3_s, t147, accBi3);
__m512d bc9_4_c, bc9_4_s;
BCASTV(bc9_4_c, HT_23[102]); BCASTV(bc9_4_s, HT_23[103]);
accAr4 = _mm512_fmadd_pd(bc9_4_c, t145, accAr4);
accAi4 = _mm512_fmadd_pd(bc9_4_c, t146, accAi4);
accBr4 = _mm512_fmadd_pd(bc9_4_s, t148, accBr4);
accBi4 = _mm512_fmadd_pd(bc9_4_s, t147, accBi4);
__m512d bc9_5_c, bc9_5_s;
BCASTV(bc9_5_c, HT_23[104]); BCASTV(bc9_5_s, HT_23[105]);
accAr5 = _mm512_fmadd_pd(bc9_5_c, t145, accAr5);
accAi5 = _mm512_fmadd_pd(bc9_5_c, t146, accAi5);
accBr5 = _mm512_fmadd_pd(bc9_5_s, t148, accBr5);
accBi5 = _mm512_fmadd_pd(bc9_5_s, t147, accBi5);
__m512d bc9_6_c, bc9_6_s;
BCASTV(bc9_6_c, HT_23[106]); BCASTV(bc9_6_s, HT_23[107]);
accAr6 = _mm512_fmadd_pd(bc9_6_c, t145, accAr6);
accAi6 = _mm512_fmadd_pd(bc9_6_c, t146, accAi6);
accBr6 = _mm512_fmadd_pd(bc9_6_s, t148, accBr6);
accBi6 = _mm512_fmadd_pd(bc9_6_s, t147, accBi6);
__m512d t149 = _mm512_load_pd(HS_23[9]);
__m512d t150 = _mm512_load_pd(HS_23[9]+8);
__m512d t151 = _mm512_load_pd(HS_23[9]+16);
__m512d t152 = _mm512_load_pd(HS_23[9]+24);
__m512d bc10_1_c, bc10_1_s;
BCASTV(bc10_1_c, HT_23[108]); BCASTV(bc10_1_s, HT_23[109]);
accAr1 = _mm512_fmadd_pd(bc10_1_c, t149, accAr1);
accAi1 = _mm512_fmadd_pd(bc10_1_c, t150, accAi1);
accBr1 = _mm512_fmadd_pd(bc10_1_s, t152, accBr1);
accBi1 = _mm512_fmadd_pd(bc10_1_s, t151, accBi1);
__m512d bc10_2_c, bc10_2_s;
BCASTV(bc10_2_c, HT_23[110]); BCASTV(bc10_2_s, HT_23[111]);
accAr2 = _mm512_fmadd_pd(bc10_2_c, t149, accAr2);
accAi2 = _mm512_fmadd_pd(bc10_2_c, t150, accAi2);
accBr2 = _mm512_fmadd_pd(bc10_2_s, t152, accBr2);
accBi2 = _mm512_fmadd_pd(bc10_2_s, t151, accBi2);
__m512d bc10_3_c, bc10_3_s;
BCASTV(bc10_3_c, HT_23[112]); BCASTV(bc10_3_s, HT_23[113]);
accAr3 = _mm512_fmadd_pd(bc10_3_c, t149, accAr3);
accAi3 = _mm512_fmadd_pd(bc10_3_c, t150, accAi3);
accBr3 = _mm512_fmadd_pd(bc10_3_s, t152, accBr3);
accBi3 = _mm512_fmadd_pd(bc10_3_s, t151, accBi3);
__m512d bc10_4_c, bc10_4_s;
BCASTV(bc10_4_c, HT_23[114]); BCASTV(bc10_4_s, HT_23[115]);
accAr4 = _mm512_fmadd_pd(bc10_4_c, t149, accAr4);
accAi4 = _mm512_fmadd_pd(bc10_4_c, t150, accAi4);
accBr4 = _mm512_fmadd_pd(bc10_4_s, t152, accBr4);
accBi4 = _mm512_fmadd_pd(bc10_4_s, t151, accBi4);
__m512d bc10_5_c, bc10_5_s;
BCASTV(bc10_5_c, HT_23[116]); BCASTV(bc10_5_s, HT_23[117]);
accAr5 = _mm512_fmadd_pd(bc10_5_c, t149, accAr5);
accAi5 = _mm512_fmadd_pd(bc10_5_c, t150, accAi5);
accBr5 = _mm512_fmadd_pd(bc10_5_s, t152, accBr5);
accBi5 = _mm512_fmadd_pd(bc10_5_s, t151, accBi5);
__m512d bc10_6_c, bc10_6_s;
BCASTV(bc10_6_c, HT_23[118]); BCASTV(bc10_6_s, HT_23[119]);
accAr6 = _mm512_fmadd_pd(bc10_6_c, t149, accAr6);
accAi6 = _mm512_fmadd_pd(bc10_6_c, t150, accAi6);
accBr6 = _mm512_fmadd_pd(bc10_6_s, t152, accBr6);
accBi6 = _mm512_fmadd_pd(bc10_6_s, t151, accBi6);
__m512d t153 = _mm512_load_pd(HS_23[10]);
__m512d t154 = _mm512_load_pd(HS_23[10]+8);
__m512d t155 = _mm512_load_pd(HS_23[10]+16);
__m512d t156 = _mm512_load_pd(HS_23[10]+24);
__m512d bc11_1_c, bc11_1_s;
BCASTV(bc11_1_c, HT_23[120]); BCASTV(bc11_1_s, HT_23[121]);
accAr1 = _mm512_fmadd_pd(bc11_1_c, t153, accAr1);
accAi1 = _mm512_fmadd_pd(bc11_1_c, t154, accAi1);
accBr1 = _mm512_fmadd_pd(bc11_1_s, t156, accBr1);
accBi1 = _mm512_fmadd_pd(bc11_1_s, t155, accBi1);
__m512d bc11_2_c, bc11_2_s;
BCASTV(bc11_2_c, HT_23[122]); BCASTV(bc11_2_s, HT_23[123]);
accAr2 = _mm512_fmadd_pd(bc11_2_c, t153, accAr2);
accAi2 = _mm512_fmadd_pd(bc11_2_c, t154, accAi2);
accBr2 = _mm512_fmadd_pd(bc11_2_s, t156, accBr2);
accBi2 = _mm512_fmadd_pd(bc11_2_s, t155, accBi2);
__m512d bc11_3_c, bc11_3_s;
BCASTV(bc11_3_c, HT_23[124]); BCASTV(bc11_3_s, HT_23[125]);
accAr3 = _mm512_fmadd_pd(bc11_3_c, t153, accAr3);
accAi3 = _mm512_fmadd_pd(bc11_3_c, t154, accAi3);
accBr3 = _mm512_fmadd_pd(bc11_3_s, t156, accBr3);
accBi3 = _mm512_fmadd_pd(bc11_3_s, t155, accBi3);
__m512d bc11_4_c, bc11_4_s;
BCASTV(bc11_4_c, HT_23[126]); BCASTV(bc11_4_s, HT_23[127]);
accAr4 = _mm512_fmadd_pd(bc11_4_c, t153, accAr4);
accAi4 = _mm512_fmadd_pd(bc11_4_c, t154, accAi4);
accBr4 = _mm512_fmadd_pd(bc11_4_s, t156, accBr4);
accBi4 = _mm512_fmadd_pd(bc11_4_s, t155, accBi4);
__m512d bc11_5_c, bc11_5_s;
BCASTV(bc11_5_c, HT_23[128]); BCASTV(bc11_5_s, HT_23[129]);
accAr5 = _mm512_fmadd_pd(bc11_5_c, t153, accAr5);
accAi5 = _mm512_fmadd_pd(bc11_5_c, t154, accAi5);
accBr5 = _mm512_fmadd_pd(bc11_5_s, t156, accBr5);
accBi5 = _mm512_fmadd_pd(bc11_5_s, t155, accBi5);
__m512d bc11_6_c, bc11_6_s;
BCASTV(bc11_6_c, HT_23[130]); BCASTV(bc11_6_s, HT_23[131]);
accAr6 = _mm512_fmadd_pd(bc11_6_c, t153, accAr6);
accAi6 = _mm512_fmadd_pd(bc11_6_c, t154, accAi6);
accBr6 = _mm512_fmadd_pd(bc11_6_s, t156, accBr6);
accBi6 = _mm512_fmadd_pd(bc11_6_s, t155, accBi6);
__m512d t157 = _mm512_add_pd(accAr1, accBr1);
__m512d t158 = _mm512_sub_pd(accAi1, accBi1);
__m512d t159 = _mm512_sub_pd(accAr1, accBr1);
__m512d t160 = _mm512_add_pd(accAi1, accBi1);
_mm512_store_pd(re + 1*es, t157);
_mm512_store_pd(im + 1*es, t158);
_mm512_store_pd(re + 22*es, t159);
_mm512_store_pd(im + 22*es, t160);
__m512d t161 = _mm512_add_pd(accAr2, accBr2);
__m512d t162 = _mm512_sub_pd(accAi2, accBi2);
__m512d t163 = _mm512_sub_pd(accAr2, accBr2);
__m512d t164 = _mm512_add_pd(accAi2, accBi2);
_mm512_store_pd(re + 2*es, t161);
_mm512_store_pd(im + 2*es, t162);
_mm512_store_pd(re + 21*es, t163);
_mm512_store_pd(im + 21*es, t164);
__m512d t165 = _mm512_add_pd(accAr3, accBr3);
__m512d t166 = _mm512_sub_pd(accAi3, accBi3);
__m512d t167 = _mm512_sub_pd(accAr3, accBr3);
__m512d t168 = _mm512_add_pd(accAi3, accBi3);
_mm512_store_pd(re + 3*es, t165);
_mm512_store_pd(im + 3*es, t166);
_mm512_store_pd(re + 20*es, t167);
_mm512_store_pd(im + 20*es, t168);
__m512d t169 = _mm512_add_pd(accAr4, accBr4);
__m512d t170 = _mm512_sub_pd(accAi4, accBi4);
__m512d t171 = _mm512_sub_pd(accAr4, accBr4);
__m512d t172 = _mm512_add_pd(accAi4, accBi4);
_mm512_store_pd(re + 4*es, t169);
_mm512_store_pd(im + 4*es, t170);
_mm512_store_pd(re + 19*es, t171);
_mm512_store_pd(im + 19*es, t172);
__m512d t173 = _mm512_add_pd(accAr5, accBr5);
__m512d t174 = _mm512_sub_pd(accAi5, accBi5);
__m512d t175 = _mm512_sub_pd(accAr5, accBr5);
__m512d t176 = _mm512_add_pd(accAi5, accBi5);
_mm512_store_pd(re + 5*es, t173);
_mm512_store_pd(im + 5*es, t174);
_mm512_store_pd(re + 18*es, t175);
_mm512_store_pd(im + 18*es, t176);
__m512d t177 = _mm512_add_pd(accAr6, accBr6);
__m512d t178 = _mm512_sub_pd(accAi6, accBi6);
__m512d t179 = _mm512_sub_pd(accAr6, accBr6);
__m512d t180 = _mm512_add_pd(accAi6, accBi6);
_mm512_store_pd(re + 6*es, t177);
_mm512_store_pd(im + 6*es, t178);
_mm512_store_pd(re + 17*es, t179);
_mm512_store_pd(im + 17*es, t180);
__asm__ volatile("" ::: "memory");
__m512d accAr7 = t1, accAi7 = t2, accBr7, accBi7;
__m512d accAr8 = t1, accAi8 = t2, accBr8, accBi8;
__m512d accAr9 = t1, accAi9 = t2, accBr9, accBi9;
__m512d accAr10 = t1, accAi10 = t2, accBr10, accBi10;
__m512d accAr11 = t1, accAi11 = t2, accBr11, accBi11;
__m512d t181 = _mm512_load_pd(HS_23[0]);
__m512d t182 = _mm512_load_pd(HS_23[0]+8);
__m512d t183 = _mm512_load_pd(HS_23[0]+16);
__m512d t184 = _mm512_load_pd(HS_23[0]+24);
__m512d bc1_7_c, bc1_7_s;
BCASTV(bc1_7_c, HT_23[132]); BCASTV(bc1_7_s, HT_23[133]);
accAr7 = _mm512_fmadd_pd(bc1_7_c, t181, accAr7);
accAi7 = _mm512_fmadd_pd(bc1_7_c, t182, accAi7);
accBr7 = _mm512_mul_pd(bc1_7_s, t184);
accBi7 = _mm512_mul_pd(bc1_7_s, t183);
__m512d bc1_8_c, bc1_8_s;
BCASTV(bc1_8_c, HT_23[134]); BCASTV(bc1_8_s, HT_23[135]);
accAr8 = _mm512_fmadd_pd(bc1_8_c, t181, accAr8);
accAi8 = _mm512_fmadd_pd(bc1_8_c, t182, accAi8);
accBr8 = _mm512_mul_pd(bc1_8_s, t184);
accBi8 = _mm512_mul_pd(bc1_8_s, t183);
__m512d bc1_9_c, bc1_9_s;
BCASTV(bc1_9_c, HT_23[136]); BCASTV(bc1_9_s, HT_23[137]);
accAr9 = _mm512_fmadd_pd(bc1_9_c, t181, accAr9);
accAi9 = _mm512_fmadd_pd(bc1_9_c, t182, accAi9);
accBr9 = _mm512_mul_pd(bc1_9_s, t184);
accBi9 = _mm512_mul_pd(bc1_9_s, t183);
__m512d bc1_10_c, bc1_10_s;
BCASTV(bc1_10_c, HT_23[138]); BCASTV(bc1_10_s, HT_23[139]);
accAr10 = _mm512_fmadd_pd(bc1_10_c, t181, accAr10);
accAi10 = _mm512_fmadd_pd(bc1_10_c, t182, accAi10);
accBr10 = _mm512_mul_pd(bc1_10_s, t184);
accBi10 = _mm512_mul_pd(bc1_10_s, t183);
__m512d bc1_11_c, bc1_11_s;
BCASTV(bc1_11_c, HT_23[140]); BCASTV(bc1_11_s, HT_23[141]);
accAr11 = _mm512_fmadd_pd(bc1_11_c, t181, accAr11);
accAi11 = _mm512_fmadd_pd(bc1_11_c, t182, accAi11);
accBr11 = _mm512_mul_pd(bc1_11_s, t184);
accBi11 = _mm512_mul_pd(bc1_11_s, t183);
__m512d t185 = _mm512_load_pd(HS_23[1]);
__m512d t186 = _mm512_load_pd(HS_23[1]+8);
__m512d t187 = _mm512_load_pd(HS_23[1]+16);
__m512d t188 = _mm512_load_pd(HS_23[1]+24);
__m512d bc2_7_c, bc2_7_s;
BCASTV(bc2_7_c, HT_23[142]); BCASTV(bc2_7_s, HT_23[143]);
accAr7 = _mm512_fmadd_pd(bc2_7_c, t185, accAr7);
accAi7 = _mm512_fmadd_pd(bc2_7_c, t186, accAi7);
accBr7 = _mm512_fmadd_pd(bc2_7_s, t188, accBr7);
accBi7 = _mm512_fmadd_pd(bc2_7_s, t187, accBi7);
__m512d bc2_8_c, bc2_8_s;
BCASTV(bc2_8_c, HT_23[144]); BCASTV(bc2_8_s, HT_23[145]);
accAr8 = _mm512_fmadd_pd(bc2_8_c, t185, accAr8);
accAi8 = _mm512_fmadd_pd(bc2_8_c, t186, accAi8);
accBr8 = _mm512_fmadd_pd(bc2_8_s, t188, accBr8);
accBi8 = _mm512_fmadd_pd(bc2_8_s, t187, accBi8);
__m512d bc2_9_c, bc2_9_s;
BCASTV(bc2_9_c, HT_23[146]); BCASTV(bc2_9_s, HT_23[147]);
accAr9 = _mm512_fmadd_pd(bc2_9_c, t185, accAr9);
accAi9 = _mm512_fmadd_pd(bc2_9_c, t186, accAi9);
accBr9 = _mm512_fmadd_pd(bc2_9_s, t188, accBr9);
accBi9 = _mm512_fmadd_pd(bc2_9_s, t187, accBi9);
__m512d bc2_10_c, bc2_10_s;
BCASTV(bc2_10_c, HT_23[148]); BCASTV(bc2_10_s, HT_23[149]);
accAr10 = _mm512_fmadd_pd(bc2_10_c, t185, accAr10);
accAi10 = _mm512_fmadd_pd(bc2_10_c, t186, accAi10);
accBr10 = _mm512_fmadd_pd(bc2_10_s, t188, accBr10);
accBi10 = _mm512_fmadd_pd(bc2_10_s, t187, accBi10);
__m512d bc2_11_c, bc2_11_s;
BCASTV(bc2_11_c, HT_23[150]); BCASTV(bc2_11_s, HT_23[151]);
accAr11 = _mm512_fmadd_pd(bc2_11_c, t185, accAr11);
accAi11 = _mm512_fmadd_pd(bc2_11_c, t186, accAi11);
accBr11 = _mm512_fmadd_pd(bc2_11_s, t188, accBr11);
accBi11 = _mm512_fmadd_pd(bc2_11_s, t187, accBi11);
__m512d t189 = _mm512_load_pd(HS_23[2]);
__m512d t190 = _mm512_load_pd(HS_23[2]+8);
__m512d t191 = _mm512_load_pd(HS_23[2]+16);
__m512d t192 = _mm512_load_pd(HS_23[2]+24);
__m512d bc3_7_c, bc3_7_s;
BCASTV(bc3_7_c, HT_23[152]); BCASTV(bc3_7_s, HT_23[153]);
accAr7 = _mm512_fmadd_pd(bc3_7_c, t189, accAr7);
accAi7 = _mm512_fmadd_pd(bc3_7_c, t190, accAi7);
accBr7 = _mm512_fmadd_pd(bc3_7_s, t192, accBr7);
accBi7 = _mm512_fmadd_pd(bc3_7_s, t191, accBi7);
__m512d bc3_8_c, bc3_8_s;
BCASTV(bc3_8_c, HT_23[154]); BCASTV(bc3_8_s, HT_23[155]);
accAr8 = _mm512_fmadd_pd(bc3_8_c, t189, accAr8);
accAi8 = _mm512_fmadd_pd(bc3_8_c, t190, accAi8);
accBr8 = _mm512_fmadd_pd(bc3_8_s, t192, accBr8);
accBi8 = _mm512_fmadd_pd(bc3_8_s, t191, accBi8);
__m512d bc3_9_c, bc3_9_s;
BCASTV(bc3_9_c, HT_23[156]); BCASTV(bc3_9_s, HT_23[157]);
accAr9 = _mm512_fmadd_pd(bc3_9_c, t189, accAr9);
accAi9 = _mm512_fmadd_pd(bc3_9_c, t190, accAi9);
accBr9 = _mm512_fmadd_pd(bc3_9_s, t192, accBr9);
accBi9 = _mm512_fmadd_pd(bc3_9_s, t191, accBi9);
__m512d bc3_10_c, bc3_10_s;
BCASTV(bc3_10_c, HT_23[158]); BCASTV(bc3_10_s, HT_23[159]);
accAr10 = _mm512_fmadd_pd(bc3_10_c, t189, accAr10);
accAi10 = _mm512_fmadd_pd(bc3_10_c, t190, accAi10);
accBr10 = _mm512_fmadd_pd(bc3_10_s, t192, accBr10);
accBi10 = _mm512_fmadd_pd(bc3_10_s, t191, accBi10);
__m512d bc3_11_c, bc3_11_s;
BCASTV(bc3_11_c, HT_23[160]); BCASTV(bc3_11_s, HT_23[161]);
accAr11 = _mm512_fmadd_pd(bc3_11_c, t189, accAr11);
accAi11 = _mm512_fmadd_pd(bc3_11_c, t190, accAi11);
accBr11 = _mm512_fmadd_pd(bc3_11_s, t192, accBr11);
accBi11 = _mm512_fmadd_pd(bc3_11_s, t191, accBi11);
__m512d t193 = _mm512_load_pd(HS_23[3]);
__m512d t194 = _mm512_load_pd(HS_23[3]+8);
__m512d t195 = _mm512_load_pd(HS_23[3]+16);
__m512d t196 = _mm512_load_pd(HS_23[3]+24);
__m512d bc4_7_c, bc4_7_s;
BCASTV(bc4_7_c, HT_23[162]); BCASTV(bc4_7_s, HT_23[163]);
accAr7 = _mm512_fmadd_pd(bc4_7_c, t193, accAr7);
accAi7 = _mm512_fmadd_pd(bc4_7_c, t194, accAi7);
accBr7 = _mm512_fmadd_pd(bc4_7_s, t196, accBr7);
accBi7 = _mm512_fmadd_pd(bc4_7_s, t195, accBi7);
__m512d bc4_8_c, bc4_8_s;
BCASTV(bc4_8_c, HT_23[164]); BCASTV(bc4_8_s, HT_23[165]);
accAr8 = _mm512_fmadd_pd(bc4_8_c, t193, accAr8);
accAi8 = _mm512_fmadd_pd(bc4_8_c, t194, accAi8);
accBr8 = _mm512_fmadd_pd(bc4_8_s, t196, accBr8);
accBi8 = _mm512_fmadd_pd(bc4_8_s, t195, accBi8);
__m512d bc4_9_c, bc4_9_s;
BCASTV(bc4_9_c, HT_23[166]); BCASTV(bc4_9_s, HT_23[167]);
accAr9 = _mm512_fmadd_pd(bc4_9_c, t193, accAr9);
accAi9 = _mm512_fmadd_pd(bc4_9_c, t194, accAi9);
accBr9 = _mm512_fmadd_pd(bc4_9_s, t196, accBr9);
accBi9 = _mm512_fmadd_pd(bc4_9_s, t195, accBi9);
__m512d bc4_10_c, bc4_10_s;
BCASTV(bc4_10_c, HT_23[168]); BCASTV(bc4_10_s, HT_23[169]);
accAr10 = _mm512_fmadd_pd(bc4_10_c, t193, accAr10);
accAi10 = _mm512_fmadd_pd(bc4_10_c, t194, accAi10);
accBr10 = _mm512_fmadd_pd(bc4_10_s, t196, accBr10);
accBi10 = _mm512_fmadd_pd(bc4_10_s, t195, accBi10);
__m512d bc4_11_c, bc4_11_s;
BCASTV(bc4_11_c, HT_23[170]); BCASTV(bc4_11_s, HT_23[171]);
accAr11 = _mm512_fmadd_pd(bc4_11_c, t193, accAr11);
accAi11 = _mm512_fmadd_pd(bc4_11_c, t194, accAi11);
accBr11 = _mm512_fmadd_pd(bc4_11_s, t196, accBr11);
accBi11 = _mm512_fmadd_pd(bc4_11_s, t195, accBi11);
__m512d t197 = _mm512_load_pd(HS_23[4]);
__m512d t198 = _mm512_load_pd(HS_23[4]+8);
__m512d t199 = _mm512_load_pd(HS_23[4]+16);
__m512d t200 = _mm512_load_pd(HS_23[4]+24);
__m512d bc5_7_c, bc5_7_s;
BCASTV(bc5_7_c, HT_23[172]); BCASTV(bc5_7_s, HT_23[173]);
accAr7 = _mm512_fmadd_pd(bc5_7_c, t197, accAr7);
accAi7 = _mm512_fmadd_pd(bc5_7_c, t198, accAi7);
accBr7 = _mm512_fmadd_pd(bc5_7_s, t200, accBr7);
accBi7 = _mm512_fmadd_pd(bc5_7_s, t199, accBi7);
__m512d bc5_8_c, bc5_8_s;
BCASTV(bc5_8_c, HT_23[174]); BCASTV(bc5_8_s, HT_23[175]);
accAr8 = _mm512_fmadd_pd(bc5_8_c, t197, accAr8);
accAi8 = _mm512_fmadd_pd(bc5_8_c, t198, accAi8);
accBr8 = _mm512_fmadd_pd(bc5_8_s, t200, accBr8);
accBi8 = _mm512_fmadd_pd(bc5_8_s, t199, accBi8);
__m512d bc5_9_c, bc5_9_s;
BCASTV(bc5_9_c, HT_23[176]); BCASTV(bc5_9_s, HT_23[177]);
accAr9 = _mm512_fmadd_pd(bc5_9_c, t197, accAr9);
accAi9 = _mm512_fmadd_pd(bc5_9_c, t198, accAi9);
accBr9 = _mm512_fmadd_pd(bc5_9_s, t200, accBr9);
accBi9 = _mm512_fmadd_pd(bc5_9_s, t199, accBi9);
__m512d bc5_10_c, bc5_10_s;
BCASTV(bc5_10_c, HT_23[178]); BCASTV(bc5_10_s, HT_23[179]);
accAr10 = _mm512_fmadd_pd(bc5_10_c, t197, accAr10);
accAi10 = _mm512_fmadd_pd(bc5_10_c, t198, accAi10);
accBr10 = _mm512_fmadd_pd(bc5_10_s, t200, accBr10);
accBi10 = _mm512_fmadd_pd(bc5_10_s, t199, accBi10);
__m512d bc5_11_c, bc5_11_s;
BCASTV(bc5_11_c, HT_23[180]); BCASTV(bc5_11_s, HT_23[181]);
accAr11 = _mm512_fmadd_pd(bc5_11_c, t197, accAr11);
accAi11 = _mm512_fmadd_pd(bc5_11_c, t198, accAi11);
accBr11 = _mm512_fmadd_pd(bc5_11_s, t200, accBr11);
accBi11 = _mm512_fmadd_pd(bc5_11_s, t199, accBi11);
__m512d t201 = _mm512_load_pd(HS_23[5]);
__m512d t202 = _mm512_load_pd(HS_23[5]+8);
__m512d t203 = _mm512_load_pd(HS_23[5]+16);
__m512d t204 = _mm512_load_pd(HS_23[5]+24);
__m512d bc6_7_c, bc6_7_s;
BCASTV(bc6_7_c, HT_23[182]); BCASTV(bc6_7_s, HT_23[183]);
accAr7 = _mm512_fmadd_pd(bc6_7_c, t201, accAr7);
accAi7 = _mm512_fmadd_pd(bc6_7_c, t202, accAi7);
accBr7 = _mm512_fmadd_pd(bc6_7_s, t204, accBr7);
accBi7 = _mm512_fmadd_pd(bc6_7_s, t203, accBi7);
__m512d bc6_8_c, bc6_8_s;
BCASTV(bc6_8_c, HT_23[184]); BCASTV(bc6_8_s, HT_23[185]);
accAr8 = _mm512_fmadd_pd(bc6_8_c, t201, accAr8);
accAi8 = _mm512_fmadd_pd(bc6_8_c, t202, accAi8);
accBr8 = _mm512_fmadd_pd(bc6_8_s, t204, accBr8);
accBi8 = _mm512_fmadd_pd(bc6_8_s, t203, accBi8);
__m512d bc6_9_c, bc6_9_s;
BCASTV(bc6_9_c, HT_23[186]); BCASTV(bc6_9_s, HT_23[187]);
accAr9 = _mm512_fmadd_pd(bc6_9_c, t201, accAr9);
accAi9 = _mm512_fmadd_pd(bc6_9_c, t202, accAi9);
accBr9 = _mm512_fmadd_pd(bc6_9_s, t204, accBr9);
accBi9 = _mm512_fmadd_pd(bc6_9_s, t203, accBi9);
__m512d bc6_10_c, bc6_10_s;
BCASTV(bc6_10_c, HT_23[188]); BCASTV(bc6_10_s, HT_23[189]);
accAr10 = _mm512_fmadd_pd(bc6_10_c, t201, accAr10);
accAi10 = _mm512_fmadd_pd(bc6_10_c, t202, accAi10);
accBr10 = _mm512_fmadd_pd(bc6_10_s, t204, accBr10);
accBi10 = _mm512_fmadd_pd(bc6_10_s, t203, accBi10);
__m512d bc6_11_c, bc6_11_s;
BCASTV(bc6_11_c, HT_23[190]); BCASTV(bc6_11_s, HT_23[191]);
accAr11 = _mm512_fmadd_pd(bc6_11_c, t201, accAr11);
accAi11 = _mm512_fmadd_pd(bc6_11_c, t202, accAi11);
accBr11 = _mm512_fmadd_pd(bc6_11_s, t204, accBr11);
accBi11 = _mm512_fmadd_pd(bc6_11_s, t203, accBi11);
__m512d t205 = _mm512_load_pd(HS_23[6]);
__m512d t206 = _mm512_load_pd(HS_23[6]+8);
__m512d t207 = _mm512_load_pd(HS_23[6]+16);
__m512d t208 = _mm512_load_pd(HS_23[6]+24);
__m512d bc7_7_c, bc7_7_s;
BCASTV(bc7_7_c, HT_23[192]); BCASTV(bc7_7_s, HT_23[193]);
accAr7 = _mm512_fmadd_pd(bc7_7_c, t205, accAr7);
accAi7 = _mm512_fmadd_pd(bc7_7_c, t206, accAi7);
accBr7 = _mm512_fmadd_pd(bc7_7_s, t208, accBr7);
accBi7 = _mm512_fmadd_pd(bc7_7_s, t207, accBi7);
__m512d bc7_8_c, bc7_8_s;
BCASTV(bc7_8_c, HT_23[194]); BCASTV(bc7_8_s, HT_23[195]);
accAr8 = _mm512_fmadd_pd(bc7_8_c, t205, accAr8);
accAi8 = _mm512_fmadd_pd(bc7_8_c, t206, accAi8);
accBr8 = _mm512_fmadd_pd(bc7_8_s, t208, accBr8);
accBi8 = _mm512_fmadd_pd(bc7_8_s, t207, accBi8);
__m512d bc7_9_c, bc7_9_s;
BCASTV(bc7_9_c, HT_23[196]); BCASTV(bc7_9_s, HT_23[197]);
accAr9 = _mm512_fmadd_pd(bc7_9_c, t205, accAr9);
accAi9 = _mm512_fmadd_pd(bc7_9_c, t206, accAi9);
accBr9 = _mm512_fmadd_pd(bc7_9_s, t208, accBr9);
accBi9 = _mm512_fmadd_pd(bc7_9_s, t207, accBi9);
__m512d bc7_10_c, bc7_10_s;
BCASTV(bc7_10_c, HT_23[198]); BCASTV(bc7_10_s, HT_23[199]);
accAr10 = _mm512_fmadd_pd(bc7_10_c, t205, accAr10);
accAi10 = _mm512_fmadd_pd(bc7_10_c, t206, accAi10);
accBr10 = _mm512_fmadd_pd(bc7_10_s, t208, accBr10);
accBi10 = _mm512_fmadd_pd(bc7_10_s, t207, accBi10);
__m512d bc7_11_c, bc7_11_s;
BCASTV(bc7_11_c, HT_23[200]); BCASTV(bc7_11_s, HT_23[201]);
accAr11 = _mm512_fmadd_pd(bc7_11_c, t205, accAr11);
accAi11 = _mm512_fmadd_pd(bc7_11_c, t206, accAi11);
accBr11 = _mm512_fmadd_pd(bc7_11_s, t208, accBr11);
accBi11 = _mm512_fmadd_pd(bc7_11_s, t207, accBi11);
__m512d t209 = _mm512_load_pd(HS_23[7]);
__m512d t210 = _mm512_load_pd(HS_23[7]+8);
__m512d t211 = _mm512_load_pd(HS_23[7]+16);
__m512d t212 = _mm512_load_pd(HS_23[7]+24);
__m512d bc8_7_c, bc8_7_s;
BCASTV(bc8_7_c, HT_23[202]); BCASTV(bc8_7_s, HT_23[203]);
accAr7 = _mm512_fmadd_pd(bc8_7_c, t209, accAr7);
accAi7 = _mm512_fmadd_pd(bc8_7_c, t210, accAi7);
accBr7 = _mm512_fmadd_pd(bc8_7_s, t212, accBr7);
accBi7 = _mm512_fmadd_pd(bc8_7_s, t211, accBi7);
__m512d bc8_8_c, bc8_8_s;
BCASTV(bc8_8_c, HT_23[204]); BCASTV(bc8_8_s, HT_23[205]);
accAr8 = _mm512_fmadd_pd(bc8_8_c, t209, accAr8);
accAi8 = _mm512_fmadd_pd(bc8_8_c, t210, accAi8);
accBr8 = _mm512_fmadd_pd(bc8_8_s, t212, accBr8);
accBi8 = _mm512_fmadd_pd(bc8_8_s, t211, accBi8);
__m512d bc8_9_c, bc8_9_s;
BCASTV(bc8_9_c, HT_23[206]); BCASTV(bc8_9_s, HT_23[207]);
accAr9 = _mm512_fmadd_pd(bc8_9_c, t209, accAr9);
accAi9 = _mm512_fmadd_pd(bc8_9_c, t210, accAi9);
accBr9 = _mm512_fmadd_pd(bc8_9_s, t212, accBr9);
accBi9 = _mm512_fmadd_pd(bc8_9_s, t211, accBi9);
__m512d bc8_10_c, bc8_10_s;
BCASTV(bc8_10_c, HT_23[208]); BCASTV(bc8_10_s, HT_23[209]);
accAr10 = _mm512_fmadd_pd(bc8_10_c, t209, accAr10);
accAi10 = _mm512_fmadd_pd(bc8_10_c, t210, accAi10);
accBr10 = _mm512_fmadd_pd(bc8_10_s, t212, accBr10);
accBi10 = _mm512_fmadd_pd(bc8_10_s, t211, accBi10);
__m512d bc8_11_c, bc8_11_s;
BCASTV(bc8_11_c, HT_23[210]); BCASTV(bc8_11_s, HT_23[211]);
accAr11 = _mm512_fmadd_pd(bc8_11_c, t209, accAr11);
accAi11 = _mm512_fmadd_pd(bc8_11_c, t210, accAi11);
accBr11 = _mm512_fmadd_pd(bc8_11_s, t212, accBr11);
accBi11 = _mm512_fmadd_pd(bc8_11_s, t211, accBi11);
__m512d t213 = _mm512_load_pd(HS_23[8]);
__m512d t214 = _mm512_load_pd(HS_23[8]+8);
__m512d t215 = _mm512_load_pd(HS_23[8]+16);
__m512d t216 = _mm512_load_pd(HS_23[8]+24);
__m512d bc9_7_c, bc9_7_s;
BCASTV(bc9_7_c, HT_23[212]); BCASTV(bc9_7_s, HT_23[213]);
accAr7 = _mm512_fmadd_pd(bc9_7_c, t213, accAr7);
accAi7 = _mm512_fmadd_pd(bc9_7_c, t214, accAi7);
accBr7 = _mm512_fmadd_pd(bc9_7_s, t216, accBr7);
accBi7 = _mm512_fmadd_pd(bc9_7_s, t215, accBi7);
__m512d bc9_8_c, bc9_8_s;
BCASTV(bc9_8_c, HT_23[214]); BCASTV(bc9_8_s, HT_23[215]);
accAr8 = _mm512_fmadd_pd(bc9_8_c, t213, accAr8);
accAi8 = _mm512_fmadd_pd(bc9_8_c, t214, accAi8);
accBr8 = _mm512_fmadd_pd(bc9_8_s, t216, accBr8);
accBi8 = _mm512_fmadd_pd(bc9_8_s, t215, accBi8);
__m512d bc9_9_c, bc9_9_s;
BCASTV(bc9_9_c, HT_23[216]); BCASTV(bc9_9_s, HT_23[217]);
accAr9 = _mm512_fmadd_pd(bc9_9_c, t213, accAr9);
accAi9 = _mm512_fmadd_pd(bc9_9_c, t214, accAi9);
accBr9 = _mm512_fmadd_pd(bc9_9_s, t216, accBr9);
accBi9 = _mm512_fmadd_pd(bc9_9_s, t215, accBi9);
__m512d bc9_10_c, bc9_10_s;
BCASTV(bc9_10_c, HT_23[218]); BCASTV(bc9_10_s, HT_23[219]);
accAr10 = _mm512_fmadd_pd(bc9_10_c, t213, accAr10);
accAi10 = _mm512_fmadd_pd(bc9_10_c, t214, accAi10);
accBr10 = _mm512_fmadd_pd(bc9_10_s, t216, accBr10);
accBi10 = _mm512_fmadd_pd(bc9_10_s, t215, accBi10);
__m512d bc9_11_c, bc9_11_s;
BCASTV(bc9_11_c, HT_23[220]); BCASTV(bc9_11_s, HT_23[221]);
accAr11 = _mm512_fmadd_pd(bc9_11_c, t213, accAr11);
accAi11 = _mm512_fmadd_pd(bc9_11_c, t214, accAi11);
accBr11 = _mm512_fmadd_pd(bc9_11_s, t216, accBr11);
accBi11 = _mm512_fmadd_pd(bc9_11_s, t215, accBi11);
__m512d t217 = _mm512_load_pd(HS_23[9]);
__m512d t218 = _mm512_load_pd(HS_23[9]+8);
__m512d t219 = _mm512_load_pd(HS_23[9]+16);
__m512d t220 = _mm512_load_pd(HS_23[9]+24);
__m512d bc10_7_c, bc10_7_s;
BCASTV(bc10_7_c, HT_23[222]); BCASTV(bc10_7_s, HT_23[223]);
accAr7 = _mm512_fmadd_pd(bc10_7_c, t217, accAr7);
accAi7 = _mm512_fmadd_pd(bc10_7_c, t218, accAi7);
accBr7 = _mm512_fmadd_pd(bc10_7_s, t220, accBr7);
accBi7 = _mm512_fmadd_pd(bc10_7_s, t219, accBi7);
__m512d bc10_8_c, bc10_8_s;
BCASTV(bc10_8_c, HT_23[224]); BCASTV(bc10_8_s, HT_23[225]);
accAr8 = _mm512_fmadd_pd(bc10_8_c, t217, accAr8);
accAi8 = _mm512_fmadd_pd(bc10_8_c, t218, accAi8);
accBr8 = _mm512_fmadd_pd(bc10_8_s, t220, accBr8);
accBi8 = _mm512_fmadd_pd(bc10_8_s, t219, accBi8);
__m512d bc10_9_c, bc10_9_s;
BCASTV(bc10_9_c, HT_23[226]); BCASTV(bc10_9_s, HT_23[227]);
accAr9 = _mm512_fmadd_pd(bc10_9_c, t217, accAr9);
accAi9 = _mm512_fmadd_pd(bc10_9_c, t218, accAi9);
accBr9 = _mm512_fmadd_pd(bc10_9_s, t220, accBr9);
accBi9 = _mm512_fmadd_pd(bc10_9_s, t219, accBi9);
__m512d bc10_10_c, bc10_10_s;
BCASTV(bc10_10_c, HT_23[228]); BCASTV(bc10_10_s, HT_23[229]);
accAr10 = _mm512_fmadd_pd(bc10_10_c, t217, accAr10);
accAi10 = _mm512_fmadd_pd(bc10_10_c, t218, accAi10);
accBr10 = _mm512_fmadd_pd(bc10_10_s, t220, accBr10);
accBi10 = _mm512_fmadd_pd(bc10_10_s, t219, accBi10);
__m512d bc10_11_c, bc10_11_s;
BCASTV(bc10_11_c, HT_23[230]); BCASTV(bc10_11_s, HT_23[231]);
accAr11 = _mm512_fmadd_pd(bc10_11_c, t217, accAr11);
accAi11 = _mm512_fmadd_pd(bc10_11_c, t218, accAi11);
accBr11 = _mm512_fmadd_pd(bc10_11_s, t220, accBr11);
accBi11 = _mm512_fmadd_pd(bc10_11_s, t219, accBi11);
__m512d t221 = _mm512_load_pd(HS_23[10]);
__m512d t222 = _mm512_load_pd(HS_23[10]+8);
__m512d t223 = _mm512_load_pd(HS_23[10]+16);
__m512d t224 = _mm512_load_pd(HS_23[10]+24);
__m512d bc11_7_c, bc11_7_s;
BCASTV(bc11_7_c, HT_23[232]); BCASTV(bc11_7_s, HT_23[233]);
accAr7 = _mm512_fmadd_pd(bc11_7_c, t221, accAr7);
accAi7 = _mm512_fmadd_pd(bc11_7_c, t222, accAi7);
accBr7 = _mm512_fmadd_pd(bc11_7_s, t224, accBr7);
accBi7 = _mm512_fmadd_pd(bc11_7_s, t223, accBi7);
__m512d bc11_8_c, bc11_8_s;
BCASTV(bc11_8_c, HT_23[234]); BCASTV(bc11_8_s, HT_23[235]);
accAr8 = _mm512_fmadd_pd(bc11_8_c, t221, accAr8);
accAi8 = _mm512_fmadd_pd(bc11_8_c, t222, accAi8);
accBr8 = _mm512_fmadd_pd(bc11_8_s, t224, accBr8);
accBi8 = _mm512_fmadd_pd(bc11_8_s, t223, accBi8);
__m512d bc11_9_c, bc11_9_s;
BCASTV(bc11_9_c, HT_23[236]); BCASTV(bc11_9_s, HT_23[237]);
accAr9 = _mm512_fmadd_pd(bc11_9_c, t221, accAr9);
accAi9 = _mm512_fmadd_pd(bc11_9_c, t222, accAi9);
accBr9 = _mm512_fmadd_pd(bc11_9_s, t224, accBr9);
accBi9 = _mm512_fmadd_pd(bc11_9_s, t223, accBi9);
__m512d bc11_10_c, bc11_10_s;
BCASTV(bc11_10_c, HT_23[238]); BCASTV(bc11_10_s, HT_23[239]);
accAr10 = _mm512_fmadd_pd(bc11_10_c, t221, accAr10);
accAi10 = _mm512_fmadd_pd(bc11_10_c, t222, accAi10);
accBr10 = _mm512_fmadd_pd(bc11_10_s, t224, accBr10);
accBi10 = _mm512_fmadd_pd(bc11_10_s, t223, accBi10);
__m512d bc11_11_c, bc11_11_s;
BCASTV(bc11_11_c, HT_23[240]); BCASTV(bc11_11_s, HT_23[241]);
accAr11 = _mm512_fmadd_pd(bc11_11_c, t221, accAr11);
accAi11 = _mm512_fmadd_pd(bc11_11_c, t222, accAi11);
accBr11 = _mm512_fmadd_pd(bc11_11_s, t224, accBr11);
accBi11 = _mm512_fmadd_pd(bc11_11_s, t223, accBi11);
__m512d t225 = _mm512_add_pd(accAr7, accBr7);
__m512d t226 = _mm512_sub_pd(accAi7, accBi7);
__m512d t227 = _mm512_sub_pd(accAr7, accBr7);
__m512d t228 = _mm512_add_pd(accAi7, accBi7);
_mm512_store_pd(re + 7*es, t225);
_mm512_store_pd(im + 7*es, t226);
_mm512_store_pd(re + 16*es, t227);
_mm512_store_pd(im + 16*es, t228);
__m512d t229 = _mm512_add_pd(accAr8, accBr8);
__m512d t230 = _mm512_sub_pd(accAi8, accBi8);
__m512d t231 = _mm512_sub_pd(accAr8, accBr8);
__m512d t232 = _mm512_add_pd(accAi8, accBi8);
_mm512_store_pd(re + 8*es, t229);
_mm512_store_pd(im + 8*es, t230);
_mm512_store_pd(re + 15*es, t231);
_mm512_store_pd(im + 15*es, t232);
__m512d t233 = _mm512_add_pd(accAr9, accBr9);
__m512d t234 = _mm512_sub_pd(accAi9, accBi9);
__m512d t235 = _mm512_sub_pd(accAr9, accBr9);
__m512d t236 = _mm512_add_pd(accAi9, accBi9);
_mm512_store_pd(re + 9*es, t233);
_mm512_store_pd(im + 9*es, t234);
_mm512_store_pd(re + 14*es, t235);
_mm512_store_pd(im + 14*es, t236);
__m512d t237 = _mm512_add_pd(accAr10, accBr10);
__m512d t238 = _mm512_sub_pd(accAi10, accBi10);
__m512d t239 = _mm512_sub_pd(accAr10, accBr10);
__m512d t240 = _mm512_add_pd(accAi10, accBi10);
_mm512_store_pd(re + 10*es, t237);
_mm512_store_pd(im + 10*es, t238);
_mm512_store_pd(re + 13*es, t239);
_mm512_store_pd(im + 13*es, t240);
__m512d t241 = _mm512_add_pd(accAr11, accBr11);
__m512d t242 = _mm512_sub_pd(accAi11, accBi11);
__m512d t243 = _mm512_sub_pd(accAr11, accBr11);
__m512d t244 = _mm512_add_pd(accAi11, accBi11);
_mm512_store_pd(re + 11*es, t241);
_mm512_store_pd(im + 11*es, t242);
_mm512_store_pd(re + 12*es, t243);
_mm512_store_pd(im + 12*es, t244);
}
static const int MK0_13t[2] = {255,255};
static const int MK1_13t[2] = {255,3};
static const int MKJ0_13t[2] = {255,255};
static const int MKJ1_13t[2] = {255,3};
#define MASKS_13t(crow_, pre_, pim_) do{ \
  mapvec_13t((pre_)+0, (pim_)+0, (crow_)+0, 255, 255); \
  mapvec_13t((pre_)+8, (pim_)+8, (crow_)+16, 255, 3); \
}while(0)

// ---------------- family B, L=13 (LPAD=16, LJP=16, CPAD=16, PPS=520) ----------------
static double TS_13t[512] ALIGN64;
static inline void mapvec_13t(double* pre, double* pim, const double* cre, const double* cim){
    __m512d xr = _mm512_load_pd(pre), xi = _mm512_load_pd(pim);
    __m512d zr = _mm512_add_pd(xr, _mm512_load_pd(cre)), zi = _mm512_add_pd(xi, _mm512_load_pd(cim));
    map2(zr, zi, &xr, &xi);
    _mm512_store_pd(pre, xr); _mm512_store_pd(pim, xi);
}

static void S_13t(double* X, const double* Csw, int do_map, int do_next){
    for(int i=0;i<13;i++){
        double* pl = X + (long)i*520;
        for(int kc=0;kc<2;kc++){ dftp13_v(pl + kc*8, pl + 256 + kc*8, 16); }
        for(int jb=0;jb<2;jb++){
            const double* rb = pl + (long)jb*8*16;
            for(int kb=0;kb<2;kb++){
                __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                r0=_mm512_load_pd(rb+0*16+kb*8); r1=_mm512_load_pd(rb+1*16+kb*8);
                r2=_mm512_load_pd(rb+2*16+kb*8); r3=_mm512_load_pd(rb+3*16+kb*8);
                r4=_mm512_load_pd(rb+4*16+kb*8); r5=_mm512_load_pd(rb+5*16+kb*8);
                r6=_mm512_load_pd(rb+6*16+kb*8); r7=_mm512_load_pd(rb+7*16+kb*8);
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                double* tb = &TS_13t[0] + (long)kb*8*16 + jb*8;
                _mm512_store_pd(tb+0*16, o0); _mm512_store_pd(tb+1*16, o1);
                _mm512_store_pd(tb+2*16, o2); _mm512_store_pd(tb+3*16, o3);
                _mm512_store_pd(tb+4*16, o4); _mm512_store_pd(tb+5*16, o5);
                _mm512_store_pd(tb+6*16, o6); _mm512_store_pd(tb+7*16, o7);
                r0=_mm512_load_pd(rb+256+0*16+kb*8); r1=_mm512_load_pd(rb+256+1*16+kb*8);
                r2=_mm512_load_pd(rb+256+2*16+kb*8); r3=_mm512_load_pd(rb+256+3*16+kb*8);
                r4=_mm512_load_pd(rb+256+4*16+kb*8); r5=_mm512_load_pd(rb+256+5*16+kb*8);
                r6=_mm512_load_pd(rb+256+6*16+kb*8); r7=_mm512_load_pd(rb+256+7*16+kb*8);
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                tb += 256;
                _mm512_store_pd(tb+0*16, o0); _mm512_store_pd(tb+1*16, o1);
                _mm512_store_pd(tb+2*16, o2); _mm512_store_pd(tb+3*16, o3);
                _mm512_store_pd(tb+4*16, o4); _mm512_store_pd(tb+5*16, o5);
                _mm512_store_pd(tb+6*16, o6); _mm512_store_pd(tb+7*16, o7);
            }
            dftp13_v(&TS_13t[0] + jb*8, &TS_13t[0] + 256 + jb*8, 16);
            if(do_map){
                const double* cre = Csw + (long)i*520 + (long)jb*104;
                double* tr = &TS_13t[0] + jb*8;
                double* ti = &TS_13t[0] + 256 + jb*8;
                int k=0;
                for(; k+2<=13; k+=2){
                    mapvec_13t(tr + (long)k*16, ti + (long)k*16, cre + k*8, cre + 256 + k*8);
                    mapvec_13t(tr + (long)(k+1)*16, ti + (long)(k+1)*16, cre + k*8 + 8, cre + 256 + k*8 + 8);
                }
                for(; k<13; k++) mapvec_13t(tr + (long)k*16, ti + (long)k*16, cre + k*8, cre + 256 + k*8);
            }
            if(do_next) dftp13_v(&TS_13t[0] + jb*8, &TS_13t[0] + 256 + jb*8, 16);
            {
                double* rb2 = pl + (long)jb*8*16;
                for(int kb=0;kb<2;kb++){
                    __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                    const double* tb = &TS_13t[0] + (long)kb*8*16 + jb*8;
                    r0=_mm512_load_pd(tb+0*16); r1=_mm512_load_pd(tb+1*16);
                    r2=_mm512_load_pd(tb+2*16); r3=_mm512_load_pd(tb+3*16);
                    r4=_mm512_load_pd(tb+4*16); r5=_mm512_load_pd(tb+5*16);
                    r6=_mm512_load_pd(tb+6*16); r7=_mm512_load_pd(tb+7*16);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(rb2+0*16+kb*8, o0); _mm512_store_pd(rb2+1*16+kb*8, o1);
                    _mm512_store_pd(rb2+2*16+kb*8, o2); _mm512_store_pd(rb2+3*16+kb*8, o3);
                    _mm512_store_pd(rb2+4*16+kb*8, o4); _mm512_store_pd(rb2+5*16+kb*8, o5);
                    _mm512_store_pd(rb2+6*16+kb*8, o6); _mm512_store_pd(rb2+7*16+kb*8, o7);
                    const double* tb2 = tb + 256;
                    r0=_mm512_load_pd(tb2+0*16); r1=_mm512_load_pd(tb2+1*16);
                    r2=_mm512_load_pd(tb2+2*16); r3=_mm512_load_pd(tb2+3*16);
                    r4=_mm512_load_pd(tb2+4*16); r5=_mm512_load_pd(tb2+5*16);
                    r6=_mm512_load_pd(tb2+6*16); r7=_mm512_load_pd(tb2+7*16);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(rb2+256+0*16+kb*8, o0); _mm512_store_pd(rb2+256+1*16+kb*8, o1);
                    _mm512_store_pd(rb2+256+2*16+kb*8, o2); _mm512_store_pd(rb2+256+3*16+kb*8, o3);
                    _mm512_store_pd(rb2+256+4*16+kb*8, o4); _mm512_store_pd(rb2+256+5*16+kb*8, o5);
                    _mm512_store_pd(rb2+256+6*16+kb*8, o6); _mm512_store_pd(rb2+256+7*16+kb*8, o7);
                }
            }
        }
        if(do_next){
            for(int kc=0;kc<2;kc++){ dftp13_v(pl + kc*8, pl + 256 + kc*8, 16); }
        }
    }
}
static void mapcol_13t(double* X, const double* C, int j, int kc, int jn, int kcn){
    double* pr = X + (long)j*16 + kc*8;
    double* pi = pr + 256;
    const double* cp = C + (long)j*16 + kc*8;
    for(int i=0;i+2<=13;i+=2){
        __m512d xr0 = _mm512_load_pd(pr + (long)i*520);
        __m512d xi0 = _mm512_load_pd(pi + (long)i*520);
        __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp + (long)i*520));
        __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp + (long)i*520 + 256));
        __m512d xr1 = _mm512_load_pd(pr + (long)(i+1)*520);
        __m512d xi1 = _mm512_load_pd(pi + (long)(i+1)*520);
        __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cp + (long)(i+1)*520));
        __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cp + (long)(i+1)*520 + 256));
        map2(zr0, zi0, &xr0, &xi0);
        map2(zr1, zi1, &xr1, &xi1);
        _mm512_store_pd(pr + (long)i*520, xr0);
        _mm512_store_pd(pi + (long)i*520, xi0);
        _mm512_store_pd(pr + (long)(i+1)*520, xr1);
        _mm512_store_pd(pi + (long)(i+1)*520, xi1);
    }
    { int i = 12;
        __m512d xr0 = _mm512_load_pd(pr + (long)i*520);
        __m512d xi0 = _mm512_load_pd(pi + (long)i*520);
        __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp + (long)i*520));
        __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp + (long)i*520 + 256));
        map2(zr0, zi0, &xr0, &xi0);
        _mm512_store_pd(pr + (long)i*520, xr0);
        _mm512_store_pd(pi + (long)i*520, xi0);
    }
}
static void P_13t(double* X, const double* C, int do_next){
    for(int j=0;j<13;j++){
        for(int kc=0;kc<2;kc++){
            double* pr = X + (long)j*16 + kc*8;
            double* pi = pr + 256;
            int kc2 = kc+1, j2 = j;
            if(kc2 >= 2){ kc2 = 0; j2 = (j+1<13) ? j+1 : 0; }
            dftp13_v(pr, pi, 520);
            mapcol_13t(X, C, j, kc, j2, kc2);
            if(do_next) dftp13_v(pr, pi, 520);
        }
    }
}
static void convin_13t(const double* src, double* X){
    for(int i=0;i<13;i++){
        for(int j=0;j<13;j++){
            const double* row = src + ((long)i*13+j)*26;
            double* pre = X + (long)i*520 + (long)j*16;
            for(int kc=0;kc<2;kc++){
                __m512d lo = _mm512_maskz_loadu_pd((__mmask8)MK0_13t[kc], row + kc*16);
                __m512d hi = _mm512_maskz_loadu_pd((__mmask8)MK1_13t[kc], row + kc*16 + 8);
                __m512d re, im;
                DEINT(lo, hi, re, im);
                _mm512_store_pd(pre + kc*8, re);
                _mm512_store_pd(pre + 256 + kc*8, im);
            }
        }
    }
}
static void convout_13t(const double* X, double* dst){
    for(int i=0;i<13;i++){
        for(int j=0;j<13;j++){
            double* row = dst + ((long)i*13+j)*26;
            const double* pre = X + (long)i*520 + (long)j*16;
            for(int kc=0;kc<2;kc++){
                __m512d re = _mm512_load_pd(pre + kc*8);
                __m512d im = _mm512_load_pd(pre + 256 + kc*8);
                __m512d lo, hi;
                INTER(re, im, lo, hi);
                _mm512_mask_storeu_pd(row + kc*16, (__mmask8)MK0_13t[kc], lo);
                _mm512_mask_storeu_pd(row + kc*16 + 8, (__mmask8)MK1_13t[kc], hi);
            }
        }
    }
}
// build swapped c: csw[i][k][j] = c[i][j][k] (complex), via DEINT + TR8 + INTER on 8x8 tiles
static void buildc_13t(const double* c, double* cnat, double* csw, int build_sw){
    for(int i=0;i<13;i++){
        const double* cp = c + (long)i*338;
        double* npre = cnat + (long)i*520;
        double* spre = csw + (long)i*520;
        for(int jb=0;jb<2;jb++){
            int jn = 13 - jb*8; if(jn>8) jn=8;
            for(int kb=0;kb<2;kb++){
                __m512d RE[8], IM[8];
                for(int r=0;r<jn;r++){
                    const double* row = cp + ((long)(jb*8+r)*13 + kb*8)*2;
                    __m512d lo = _mm512_maskz_loadu_pd((__mmask8)MK0_13t[kb], row);
                    __m512d hi = _mm512_maskz_loadu_pd((__mmask8)MK1_13t[kb], row+8);
                    DEINT(lo, hi, RE[r], IM[r]);
                    _mm512_store_pd(npre + (long)(jb*8+r)*16 + kb*8, RE[r]);
                    _mm512_store_pd(npre + 256 + (long)(jb*8+r)*16 + kb*8, IM[r]);
                }
                if(build_sw){
                    for(int r=jn;r<8;r++){ RE[r]=_mm512_setzero_pd(); IM[r]=_mm512_setzero_pd(); }
                    __m512d o0,o1,o2,o3,o4,o5,o6,o7;
                    TR8(RE[0],RE[1],RE[2],RE[3],RE[4],RE[5],RE[6],RE[7],o0,o1,o2,o3,o4,o5,o6,o7);
                    if(kb*8+0 < 13) _mm512_store_pd(spre + (long)jb*104 + (long)(kb*8+0)*8, o0);
                    if(kb*8+1 < 13) _mm512_store_pd(spre + (long)jb*104 + (long)(kb*8+1)*8, o1);
                    if(kb*8+2 < 13) _mm512_store_pd(spre + (long)jb*104 + (long)(kb*8+2)*8, o2);
                    if(kb*8+3 < 13) _mm512_store_pd(spre + (long)jb*104 + (long)(kb*8+3)*8, o3);
                    if(kb*8+4 < 13) _mm512_store_pd(spre + (long)jb*104 + (long)(kb*8+4)*8, o4);
                    if(kb*8+5 < 13) _mm512_store_pd(spre + (long)jb*104 + (long)(kb*8+5)*8, o5);
                    if(kb*8+6 < 13) _mm512_store_pd(spre + (long)jb*104 + (long)(kb*8+6)*8, o6);
                    if(kb*8+7 < 13) _mm512_store_pd(spre + (long)jb*104 + (long)(kb*8+7)*8, o7);
                    __m512d oo0,oo1,oo2,oo3,oo4,oo5,oo6,oo7;
                    TR8(IM[0],IM[1],IM[2],IM[3],IM[4],IM[5],IM[6],IM[7],oo0,oo1,oo2,oo3,oo4,oo5,oo6,oo7);
                    if(kb*8+0 < 13) _mm512_store_pd(spre + 256 + (long)jb*104 + (long)(kb*8+0)*8, oo0);
                    if(kb*8+1 < 13) _mm512_store_pd(spre + 256 + (long)jb*104 + (long)(kb*8+1)*8, oo1);
                    if(kb*8+2 < 13) _mm512_store_pd(spre + 256 + (long)jb*104 + (long)(kb*8+2)*8, oo2);
                    if(kb*8+3 < 13) _mm512_store_pd(spre + 256 + (long)jb*104 + (long)(kb*8+3)*8, oo3);
                    if(kb*8+4 < 13) _mm512_store_pd(spre + 256 + (long)jb*104 + (long)(kb*8+4)*8, oo4);
                    if(kb*8+5 < 13) _mm512_store_pd(spre + 256 + (long)jb*104 + (long)(kb*8+5)*8, oo5);
                    if(kb*8+6 < 13) _mm512_store_pd(spre + 256 + (long)jb*104 + (long)(kb*8+6)*8, oo6);
                    if(kb*8+7 < 13) _mm512_store_pd(spre + 256 + (long)jb*104 + (long)(kb*8+7)*8, oo7);
                }
            }
        }
    }
}
static double* XV_13t = 0;
static double* CSW_13t = 0;
static double* CNAT_13t = 0;
void run_13t(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if(m < 1) m = 1;
    if(!XV_13t){ XV_13t = alloc_huge_st((long)13*520*8 + 4096); CSW_13t = alloc_huge_st((long)13*520*8 + 4096); CNAT_13t = alloc_huge_st((long)13*520*8 + 4096); }
    for(long v=0; v<B; v++){
        buildc_13t(c + v*4394, CNAT_13t, CSW_13t, m >= 3);
        const double* cx = CNAT_13t;
        convin_13t(x0 + v*4394, XV_13t);
        S_13t(XV_13t, CSW_13t, 0, 0);
        P_13t(XV_13t, cx, 0);
        convout_13t(XV_13t, out1 + v*4394);
        if(m >= 2){
            S_13t(XV_13t, CSW_13t, 0, 0);
            for(long t=2; t<=m; t++){
                if((t & 1) == 0) P_13t(XV_13t, cx, t<m);
                else             S_13t(XV_13t, CSW_13t, 1, t<m);
            }
        }
        convout_13t(XV_13t, outm + v*4394);
    }
}

static const int MK0_17t[3] = {255,255,3};
static const int MK1_17t[3] = {255,255,0};
static const int MKJ0_17t[3] = {255,255,3};
static const int MKJ1_17t[3] = {255,255,0};
#define MASKS_17t(crow_, pre_, pim_) do{ \
  mapvec_17t((pre_)+0, (pim_)+0, (crow_)+0, 255, 255); \
  mapvec_17t((pre_)+8, (pim_)+8, (crow_)+16, 255, 255); \
  mapvec_17t((pre_)+16, (pim_)+16, (crow_)+32, 3, 0); \
}while(0)

// ---------------- family B, L=17 (LPAD=24, LJP=24, CPAD=24, PPS=1160) ----------------
static double TS_17t[1152] ALIGN64;
static inline void mapvec_17t(double* pre, double* pim, const double* cre, const double* cim){
    __m512d xr = _mm512_load_pd(pre), xi = _mm512_load_pd(pim);
    __m512d zr = _mm512_add_pd(xr, _mm512_load_pd(cre)), zi = _mm512_add_pd(xi, _mm512_load_pd(cim));
    map2(zr, zi, &xr, &xi);
    _mm512_store_pd(pre, xr); _mm512_store_pd(pim, xi);
}

static void S_17t(double* X, const double* Csw, int do_map, int do_next){
    for(int i=0;i<17;i++){
        double* pl = X + (long)i*1160;
        for(int kc=0;kc<3;kc++){ dftp17_v(pl + kc*8, pl + 576 + kc*8, 24); }
        for(int jb=0;jb<3;jb++){
            const double* rb = pl + (long)jb*8*24;
            for(int kb=0;kb<3;kb++){
                __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                r0=_mm512_load_pd(rb+0*24+kb*8); r1=_mm512_load_pd(rb+1*24+kb*8);
                r2=_mm512_load_pd(rb+2*24+kb*8); r3=_mm512_load_pd(rb+3*24+kb*8);
                r4=_mm512_load_pd(rb+4*24+kb*8); r5=_mm512_load_pd(rb+5*24+kb*8);
                r6=_mm512_load_pd(rb+6*24+kb*8); r7=_mm512_load_pd(rb+7*24+kb*8);
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                double* tb = &TS_17t[0] + (long)kb*8*24 + jb*8;
                _mm512_store_pd(tb+0*24, o0); _mm512_store_pd(tb+1*24, o1);
                _mm512_store_pd(tb+2*24, o2); _mm512_store_pd(tb+3*24, o3);
                _mm512_store_pd(tb+4*24, o4); _mm512_store_pd(tb+5*24, o5);
                _mm512_store_pd(tb+6*24, o6); _mm512_store_pd(tb+7*24, o7);
                r0=_mm512_load_pd(rb+576+0*24+kb*8); r1=_mm512_load_pd(rb+576+1*24+kb*8);
                r2=_mm512_load_pd(rb+576+2*24+kb*8); r3=_mm512_load_pd(rb+576+3*24+kb*8);
                r4=_mm512_load_pd(rb+576+4*24+kb*8); r5=_mm512_load_pd(rb+576+5*24+kb*8);
                r6=_mm512_load_pd(rb+576+6*24+kb*8); r7=_mm512_load_pd(rb+576+7*24+kb*8);
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                tb += 576;
                _mm512_store_pd(tb+0*24, o0); _mm512_store_pd(tb+1*24, o1);
                _mm512_store_pd(tb+2*24, o2); _mm512_store_pd(tb+3*24, o3);
                _mm512_store_pd(tb+4*24, o4); _mm512_store_pd(tb+5*24, o5);
                _mm512_store_pd(tb+6*24, o6); _mm512_store_pd(tb+7*24, o7);
            }
            dftp17_v(&TS_17t[0] + jb*8, &TS_17t[0] + 576 + jb*8, 24);
            if(do_map){
                const double* cre = Csw + (long)i*1160 + (long)jb*136;
                double* tr = &TS_17t[0] + jb*8;
                double* ti = &TS_17t[0] + 576 + jb*8;
                int k=0;
                for(; k+2<=17; k+=2){
                    mapvec_17t(tr + (long)k*24, ti + (long)k*24, cre + k*8, cre + 576 + k*8);
                    mapvec_17t(tr + (long)(k+1)*24, ti + (long)(k+1)*24, cre + k*8 + 8, cre + 576 + k*8 + 8);
                }
                for(; k<17; k++) mapvec_17t(tr + (long)k*24, ti + (long)k*24, cre + k*8, cre + 576 + k*8);
            }
            if(do_next) dftp17_v(&TS_17t[0] + jb*8, &TS_17t[0] + 576 + jb*8, 24);
            {
                double* rb2 = pl + (long)jb*8*24;
                for(int kb=0;kb<3;kb++){
                    __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                    const double* tb = &TS_17t[0] + (long)kb*8*24 + jb*8;
                    r0=_mm512_load_pd(tb+0*24); r1=_mm512_load_pd(tb+1*24);
                    r2=_mm512_load_pd(tb+2*24); r3=_mm512_load_pd(tb+3*24);
                    r4=_mm512_load_pd(tb+4*24); r5=_mm512_load_pd(tb+5*24);
                    r6=_mm512_load_pd(tb+6*24); r7=_mm512_load_pd(tb+7*24);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(rb2+0*24+kb*8, o0); _mm512_store_pd(rb2+1*24+kb*8, o1);
                    _mm512_store_pd(rb2+2*24+kb*8, o2); _mm512_store_pd(rb2+3*24+kb*8, o3);
                    _mm512_store_pd(rb2+4*24+kb*8, o4); _mm512_store_pd(rb2+5*24+kb*8, o5);
                    _mm512_store_pd(rb2+6*24+kb*8, o6); _mm512_store_pd(rb2+7*24+kb*8, o7);
                    const double* tb2 = tb + 576;
                    r0=_mm512_load_pd(tb2+0*24); r1=_mm512_load_pd(tb2+1*24);
                    r2=_mm512_load_pd(tb2+2*24); r3=_mm512_load_pd(tb2+3*24);
                    r4=_mm512_load_pd(tb2+4*24); r5=_mm512_load_pd(tb2+5*24);
                    r6=_mm512_load_pd(tb2+6*24); r7=_mm512_load_pd(tb2+7*24);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(rb2+576+0*24+kb*8, o0); _mm512_store_pd(rb2+576+1*24+kb*8, o1);
                    _mm512_store_pd(rb2+576+2*24+kb*8, o2); _mm512_store_pd(rb2+576+3*24+kb*8, o3);
                    _mm512_store_pd(rb2+576+4*24+kb*8, o4); _mm512_store_pd(rb2+576+5*24+kb*8, o5);
                    _mm512_store_pd(rb2+576+6*24+kb*8, o6); _mm512_store_pd(rb2+576+7*24+kb*8, o7);
                }
            }
        }
        if(do_next){
            for(int kc=0;kc<3;kc++){ dftp17_v(pl + kc*8, pl + 576 + kc*8, 24); }
        }
    }
}
static void mapcol_17t(double* X, const double* C, int j, int kc, int jn, int kcn){
    double* pr = X + (long)j*24 + kc*8;
    double* pi = pr + 576;
    const double* cp = C + (long)j*24 + kc*8;
    for(int i=0;i+2<=17;i+=2){
        __m512d xr0 = _mm512_load_pd(pr + (long)i*1160);
        __m512d xi0 = _mm512_load_pd(pi + (long)i*1160);
        __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp + (long)i*1160));
        __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp + (long)i*1160 + 576));
        __m512d xr1 = _mm512_load_pd(pr + (long)(i+1)*1160);
        __m512d xi1 = _mm512_load_pd(pi + (long)(i+1)*1160);
        __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cp + (long)(i+1)*1160));
        __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cp + (long)(i+1)*1160 + 576));
        map2(zr0, zi0, &xr0, &xi0);
        map2(zr1, zi1, &xr1, &xi1);
        _mm512_store_pd(pr + (long)i*1160, xr0);
        _mm512_store_pd(pi + (long)i*1160, xi0);
        _mm512_store_pd(pr + (long)(i+1)*1160, xr1);
        _mm512_store_pd(pi + (long)(i+1)*1160, xi1);
    }
    { int i = 16;
        __m512d xr0 = _mm512_load_pd(pr + (long)i*1160);
        __m512d xi0 = _mm512_load_pd(pi + (long)i*1160);
        __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp + (long)i*1160));
        __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp + (long)i*1160 + 576));
        map2(zr0, zi0, &xr0, &xi0);
        _mm512_store_pd(pr + (long)i*1160, xr0);
        _mm512_store_pd(pi + (long)i*1160, xi0);
    }
}
static void P_17t(double* X, const double* C, int do_next){
    for(int j=0;j<17;j++){
        for(int kc=0;kc<3;kc++){
            double* pr = X + (long)j*24 + kc*8;
            double* pi = pr + 576;
            int kc2 = kc+1, j2 = j;
            if(kc2 >= 3){ kc2 = 0; j2 = (j+1<17) ? j+1 : 0; }
            dftp17_v(pr, pi, 1160);
            mapcol_17t(X, C, j, kc, j2, kc2);
            if(do_next) dftp17_v(pr, pi, 1160);
        }
    }
}
static void convin_17t(const double* src, double* X){
    for(int i=0;i<17;i++){
        for(int j=0;j<17;j++){
            const double* row = src + ((long)i*17+j)*34;
            double* pre = X + (long)i*1160 + (long)j*24;
            for(int kc=0;kc<3;kc++){
                __m512d lo = _mm512_maskz_loadu_pd((__mmask8)MK0_17t[kc], row + kc*16);
                __m512d hi = _mm512_maskz_loadu_pd((__mmask8)MK1_17t[kc], row + kc*16 + 8);
                __m512d re, im;
                DEINT(lo, hi, re, im);
                _mm512_store_pd(pre + kc*8, re);
                _mm512_store_pd(pre + 576 + kc*8, im);
            }
        }
    }
}
static void convout_17t(const double* X, double* dst){
    for(int i=0;i<17;i++){
        for(int j=0;j<17;j++){
            double* row = dst + ((long)i*17+j)*34;
            const double* pre = X + (long)i*1160 + (long)j*24;
            for(int kc=0;kc<3;kc++){
                __m512d re = _mm512_load_pd(pre + kc*8);
                __m512d im = _mm512_load_pd(pre + 576 + kc*8);
                __m512d lo, hi;
                INTER(re, im, lo, hi);
                _mm512_mask_storeu_pd(row + kc*16, (__mmask8)MK0_17t[kc], lo);
                _mm512_mask_storeu_pd(row + kc*16 + 8, (__mmask8)MK1_17t[kc], hi);
            }
        }
    }
}
// build swapped c: csw[i][k][j] = c[i][j][k] (complex), via DEINT + TR8 + INTER on 8x8 tiles
static void buildc_17t(const double* c, double* cnat, double* csw, int build_sw){
    for(int i=0;i<17;i++){
        const double* cp = c + (long)i*578;
        double* npre = cnat + (long)i*1160;
        double* spre = csw + (long)i*1160;
        for(int jb=0;jb<3;jb++){
            int jn = 17 - jb*8; if(jn>8) jn=8;
            for(int kb=0;kb<3;kb++){
                __m512d RE[8], IM[8];
                for(int r=0;r<jn;r++){
                    const double* row = cp + ((long)(jb*8+r)*17 + kb*8)*2;
                    __m512d lo = _mm512_maskz_loadu_pd((__mmask8)MK0_17t[kb], row);
                    __m512d hi = _mm512_maskz_loadu_pd((__mmask8)MK1_17t[kb], row+8);
                    DEINT(lo, hi, RE[r], IM[r]);
                    _mm512_store_pd(npre + (long)(jb*8+r)*24 + kb*8, RE[r]);
                    _mm512_store_pd(npre + 576 + (long)(jb*8+r)*24 + kb*8, IM[r]);
                }
                if(build_sw){
                    for(int r=jn;r<8;r++){ RE[r]=_mm512_setzero_pd(); IM[r]=_mm512_setzero_pd(); }
                    __m512d o0,o1,o2,o3,o4,o5,o6,o7;
                    TR8(RE[0],RE[1],RE[2],RE[3],RE[4],RE[5],RE[6],RE[7],o0,o1,o2,o3,o4,o5,o6,o7);
                    if(kb*8+0 < 17) _mm512_store_pd(spre + (long)jb*136 + (long)(kb*8+0)*8, o0);
                    if(kb*8+1 < 17) _mm512_store_pd(spre + (long)jb*136 + (long)(kb*8+1)*8, o1);
                    if(kb*8+2 < 17) _mm512_store_pd(spre + (long)jb*136 + (long)(kb*8+2)*8, o2);
                    if(kb*8+3 < 17) _mm512_store_pd(spre + (long)jb*136 + (long)(kb*8+3)*8, o3);
                    if(kb*8+4 < 17) _mm512_store_pd(spre + (long)jb*136 + (long)(kb*8+4)*8, o4);
                    if(kb*8+5 < 17) _mm512_store_pd(spre + (long)jb*136 + (long)(kb*8+5)*8, o5);
                    if(kb*8+6 < 17) _mm512_store_pd(spre + (long)jb*136 + (long)(kb*8+6)*8, o6);
                    if(kb*8+7 < 17) _mm512_store_pd(spre + (long)jb*136 + (long)(kb*8+7)*8, o7);
                    __m512d oo0,oo1,oo2,oo3,oo4,oo5,oo6,oo7;
                    TR8(IM[0],IM[1],IM[2],IM[3],IM[4],IM[5],IM[6],IM[7],oo0,oo1,oo2,oo3,oo4,oo5,oo6,oo7);
                    if(kb*8+0 < 17) _mm512_store_pd(spre + 576 + (long)jb*136 + (long)(kb*8+0)*8, oo0);
                    if(kb*8+1 < 17) _mm512_store_pd(spre + 576 + (long)jb*136 + (long)(kb*8+1)*8, oo1);
                    if(kb*8+2 < 17) _mm512_store_pd(spre + 576 + (long)jb*136 + (long)(kb*8+2)*8, oo2);
                    if(kb*8+3 < 17) _mm512_store_pd(spre + 576 + (long)jb*136 + (long)(kb*8+3)*8, oo3);
                    if(kb*8+4 < 17) _mm512_store_pd(spre + 576 + (long)jb*136 + (long)(kb*8+4)*8, oo4);
                    if(kb*8+5 < 17) _mm512_store_pd(spre + 576 + (long)jb*136 + (long)(kb*8+5)*8, oo5);
                    if(kb*8+6 < 17) _mm512_store_pd(spre + 576 + (long)jb*136 + (long)(kb*8+6)*8, oo6);
                    if(kb*8+7 < 17) _mm512_store_pd(spre + 576 + (long)jb*136 + (long)(kb*8+7)*8, oo7);
                }
            }
        }
    }
}
static double* XV_17t = 0;
static double* CSW_17t = 0;
static double* CNAT_17t = 0;
void run_17t(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if(m < 1) m = 1;
    if(!XV_17t){ XV_17t = alloc_huge_st((long)17*1160*8 + 4096); CSW_17t = alloc_huge_st((long)17*1160*8 + 4096); CNAT_17t = alloc_huge_st((long)17*1160*8 + 4096); }
    for(long v=0; v<B; v++){
        buildc_17t(c + v*9826, CNAT_17t, CSW_17t, m >= 3);
        const double* cx = CNAT_17t;
        convin_17t(x0 + v*9826, XV_17t);
        S_17t(XV_17t, CSW_17t, 0, 0);
        P_17t(XV_17t, cx, 0);
        convout_17t(XV_17t, out1 + v*9826);
        if(m >= 2){
            S_17t(XV_17t, CSW_17t, 0, 0);
            for(long t=2; t<=m; t++){
                if((t & 1) == 0) P_17t(XV_17t, cx, t<m);
                else             S_17t(XV_17t, CSW_17t, 1, t<m);
            }
        }
        convout_17t(XV_17t, outm + v*9826);
    }
}

static const int MK0_23t[3] = {255,255,255};
static const int MK1_23t[3] = {255,255,63};
static const int MKJ0_23t[3] = {255,255,255};
static const int MKJ1_23t[3] = {255,255,63};
#define MASKS_23t(crow_, pre_, pim_) do{ \
  mapvec_23t((pre_)+0, (pim_)+0, (crow_)+0, 255, 255); \
  mapvec_23t((pre_)+8, (pim_)+8, (crow_)+16, 255, 255); \
  mapvec_23t((pre_)+16, (pim_)+16, (crow_)+32, 255, 63); \
}while(0)

// ---------------- family B, L=23 (LPAD=24, LJP=24, CPAD=24, PPS=1160) ----------------
static double TS_23t[1152] ALIGN64;
static inline void mapvec_23t(double* pre, double* pim, const double* cre, const double* cim){
    __m512d xr = _mm512_load_pd(pre), xi = _mm512_load_pd(pim);
    __m512d zr = _mm512_add_pd(xr, _mm512_load_pd(cre)), zi = _mm512_add_pd(xi, _mm512_load_pd(cim));
    map2(zr, zi, &xr, &xi);
    _mm512_store_pd(pre, xr); _mm512_store_pd(pim, xi);
}

static void S_23t(double* X, const double* Csw, int do_map, int do_next){
    for(int i=0;i<23;i++){
        double* pl = X + (long)i*1160;
        for(int kc=0;kc<3;kc++){ dftp23_v(pl + kc*8, pl + 576 + kc*8, 24); }
        for(int jb=0;jb<3;jb++){
            const double* rb = pl + (long)jb*8*24;
            for(int kb=0;kb<3;kb++){
                __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                r0=_mm512_load_pd(rb+0*24+kb*8); r1=_mm512_load_pd(rb+1*24+kb*8);
                r2=_mm512_load_pd(rb+2*24+kb*8); r3=_mm512_load_pd(rb+3*24+kb*8);
                r4=_mm512_load_pd(rb+4*24+kb*8); r5=_mm512_load_pd(rb+5*24+kb*8);
                r6=_mm512_load_pd(rb+6*24+kb*8); r7=_mm512_load_pd(rb+7*24+kb*8);
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                double* tb = &TS_23t[0] + (long)kb*8*24 + jb*8;
                _mm512_store_pd(tb+0*24, o0); _mm512_store_pd(tb+1*24, o1);
                _mm512_store_pd(tb+2*24, o2); _mm512_store_pd(tb+3*24, o3);
                _mm512_store_pd(tb+4*24, o4); _mm512_store_pd(tb+5*24, o5);
                _mm512_store_pd(tb+6*24, o6); _mm512_store_pd(tb+7*24, o7);
                r0=_mm512_load_pd(rb+576+0*24+kb*8); r1=_mm512_load_pd(rb+576+1*24+kb*8);
                r2=_mm512_load_pd(rb+576+2*24+kb*8); r3=_mm512_load_pd(rb+576+3*24+kb*8);
                r4=_mm512_load_pd(rb+576+4*24+kb*8); r5=_mm512_load_pd(rb+576+5*24+kb*8);
                r6=_mm512_load_pd(rb+576+6*24+kb*8); r7=_mm512_load_pd(rb+576+7*24+kb*8);
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                tb += 576;
                _mm512_store_pd(tb+0*24, o0); _mm512_store_pd(tb+1*24, o1);
                _mm512_store_pd(tb+2*24, o2); _mm512_store_pd(tb+3*24, o3);
                _mm512_store_pd(tb+4*24, o4); _mm512_store_pd(tb+5*24, o5);
                _mm512_store_pd(tb+6*24, o6); _mm512_store_pd(tb+7*24, o7);
            }
            dftp23_v(&TS_23t[0] + jb*8, &TS_23t[0] + 576 + jb*8, 24);
            if(do_map){
                const double* cre = Csw + (long)i*1160 + (long)jb*184;
                double* tr = &TS_23t[0] + jb*8;
                double* ti = &TS_23t[0] + 576 + jb*8;
                int k=0;
                for(; k+2<=23; k+=2){
                    mapvec_23t(tr + (long)k*24, ti + (long)k*24, cre + k*8, cre + 576 + k*8);
                    mapvec_23t(tr + (long)(k+1)*24, ti + (long)(k+1)*24, cre + k*8 + 8, cre + 576 + k*8 + 8);
                }
                for(; k<23; k++) mapvec_23t(tr + (long)k*24, ti + (long)k*24, cre + k*8, cre + 576 + k*8);
            }
            if(do_next) dftp23_v(&TS_23t[0] + jb*8, &TS_23t[0] + 576 + jb*8, 24);
            {
                double* rb2 = pl + (long)jb*8*24;
                for(int kb=0;kb<3;kb++){
                    __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                    const double* tb = &TS_23t[0] + (long)kb*8*24 + jb*8;
                    r0=_mm512_load_pd(tb+0*24); r1=_mm512_load_pd(tb+1*24);
                    r2=_mm512_load_pd(tb+2*24); r3=_mm512_load_pd(tb+3*24);
                    r4=_mm512_load_pd(tb+4*24); r5=_mm512_load_pd(tb+5*24);
                    r6=_mm512_load_pd(tb+6*24); r7=_mm512_load_pd(tb+7*24);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(rb2+0*24+kb*8, o0); _mm512_store_pd(rb2+1*24+kb*8, o1);
                    _mm512_store_pd(rb2+2*24+kb*8, o2); _mm512_store_pd(rb2+3*24+kb*8, o3);
                    _mm512_store_pd(rb2+4*24+kb*8, o4); _mm512_store_pd(rb2+5*24+kb*8, o5);
                    _mm512_store_pd(rb2+6*24+kb*8, o6); _mm512_store_pd(rb2+7*24+kb*8, o7);
                    const double* tb2 = tb + 576;
                    r0=_mm512_load_pd(tb2+0*24); r1=_mm512_load_pd(tb2+1*24);
                    r2=_mm512_load_pd(tb2+2*24); r3=_mm512_load_pd(tb2+3*24);
                    r4=_mm512_load_pd(tb2+4*24); r5=_mm512_load_pd(tb2+5*24);
                    r6=_mm512_load_pd(tb2+6*24); r7=_mm512_load_pd(tb2+7*24);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(rb2+576+0*24+kb*8, o0); _mm512_store_pd(rb2+576+1*24+kb*8, o1);
                    _mm512_store_pd(rb2+576+2*24+kb*8, o2); _mm512_store_pd(rb2+576+3*24+kb*8, o3);
                    _mm512_store_pd(rb2+576+4*24+kb*8, o4); _mm512_store_pd(rb2+576+5*24+kb*8, o5);
                    _mm512_store_pd(rb2+576+6*24+kb*8, o6); _mm512_store_pd(rb2+576+7*24+kb*8, o7);
                }
            }
        }
        if(do_next){
            for(int kc=0;kc<3;kc++){ dftp23_v(pl + kc*8, pl + 576 + kc*8, 24); }
        }
    }
}
static void mapcol_23t(double* X, const double* C, int j, int kc, int jn, int kcn){
    double* pr = X + (long)j*24 + kc*8;
    double* pi = pr + 576;
    const double* cp = C + (long)j*24 + kc*8;
    for(int i=0;i+2<=23;i+=2){
        __m512d xr0 = _mm512_load_pd(pr + (long)i*1160);
        __m512d xi0 = _mm512_load_pd(pi + (long)i*1160);
        __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp + (long)i*1160));
        __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp + (long)i*1160 + 576));
        __m512d xr1 = _mm512_load_pd(pr + (long)(i+1)*1160);
        __m512d xi1 = _mm512_load_pd(pi + (long)(i+1)*1160);
        __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cp + (long)(i+1)*1160));
        __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cp + (long)(i+1)*1160 + 576));
        map2(zr0, zi0, &xr0, &xi0);
        map2(zr1, zi1, &xr1, &xi1);
        _mm512_store_pd(pr + (long)i*1160, xr0);
        _mm512_store_pd(pi + (long)i*1160, xi0);
        _mm512_store_pd(pr + (long)(i+1)*1160, xr1);
        _mm512_store_pd(pi + (long)(i+1)*1160, xi1);
    }
    { int i = 22;
        __m512d xr0 = _mm512_load_pd(pr + (long)i*1160);
        __m512d xi0 = _mm512_load_pd(pi + (long)i*1160);
        __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp + (long)i*1160));
        __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp + (long)i*1160 + 576));
        map2(zr0, zi0, &xr0, &xi0);
        _mm512_store_pd(pr + (long)i*1160, xr0);
        _mm512_store_pd(pi + (long)i*1160, xi0);
    }
}
static void P_23t(double* X, const double* C, int do_next){
    for(int j=0;j<23;j++){
        for(int kc=0;kc<3;kc++){
            double* pr = X + (long)j*24 + kc*8;
            double* pi = pr + 576;
            int kc2 = kc+1, j2 = j;
            if(kc2 >= 3){ kc2 = 0; j2 = (j+1<23) ? j+1 : 0; }
            dftp23_v(pr, pi, 1160);
            mapcol_23t(X, C, j, kc, j2, kc2);
            if(do_next) dftp23_v(pr, pi, 1160);
        }
    }
}
static void convin_23t(const double* src, double* X){
    for(int i=0;i<23;i++){
        for(int j=0;j<23;j++){
            const double* row = src + ((long)i*23+j)*46;
            double* pre = X + (long)i*1160 + (long)j*24;
            for(int kc=0;kc<3;kc++){
                __m512d lo = _mm512_maskz_loadu_pd((__mmask8)MK0_23t[kc], row + kc*16);
                __m512d hi = _mm512_maskz_loadu_pd((__mmask8)MK1_23t[kc], row + kc*16 + 8);
                __m512d re, im;
                DEINT(lo, hi, re, im);
                _mm512_store_pd(pre + kc*8, re);
                _mm512_store_pd(pre + 576 + kc*8, im);
            }
        }
    }
}
static void convout_23t(const double* X, double* dst){
    for(int i=0;i<23;i++){
        for(int j=0;j<23;j++){
            double* row = dst + ((long)i*23+j)*46;
            const double* pre = X + (long)i*1160 + (long)j*24;
            for(int kc=0;kc<3;kc++){
                __m512d re = _mm512_load_pd(pre + kc*8);
                __m512d im = _mm512_load_pd(pre + 576 + kc*8);
                __m512d lo, hi;
                INTER(re, im, lo, hi);
                _mm512_mask_storeu_pd(row + kc*16, (__mmask8)MK0_23t[kc], lo);
                _mm512_mask_storeu_pd(row + kc*16 + 8, (__mmask8)MK1_23t[kc], hi);
            }
        }
    }
}
// build swapped c: csw[i][k][j] = c[i][j][k] (complex), via DEINT + TR8 + INTER on 8x8 tiles
static void buildc_23t(const double* c, double* cnat, double* csw, int build_sw){
    for(int i=0;i<23;i++){
        const double* cp = c + (long)i*1058;
        double* npre = cnat + (long)i*1160;
        double* spre = csw + (long)i*1160;
        for(int jb=0;jb<3;jb++){
            int jn = 23 - jb*8; if(jn>8) jn=8;
            for(int kb=0;kb<3;kb++){
                __m512d RE[8], IM[8];
                for(int r=0;r<jn;r++){
                    const double* row = cp + ((long)(jb*8+r)*23 + kb*8)*2;
                    __m512d lo = _mm512_maskz_loadu_pd((__mmask8)MK0_23t[kb], row);
                    __m512d hi = _mm512_maskz_loadu_pd((__mmask8)MK1_23t[kb], row+8);
                    DEINT(lo, hi, RE[r], IM[r]);
                    _mm512_store_pd(npre + (long)(jb*8+r)*24 + kb*8, RE[r]);
                    _mm512_store_pd(npre + 576 + (long)(jb*8+r)*24 + kb*8, IM[r]);
                }
                if(build_sw){
                    for(int r=jn;r<8;r++){ RE[r]=_mm512_setzero_pd(); IM[r]=_mm512_setzero_pd(); }
                    __m512d o0,o1,o2,o3,o4,o5,o6,o7;
                    TR8(RE[0],RE[1],RE[2],RE[3],RE[4],RE[5],RE[6],RE[7],o0,o1,o2,o3,o4,o5,o6,o7);
                    if(kb*8+0 < 23) _mm512_store_pd(spre + (long)jb*184 + (long)(kb*8+0)*8, o0);
                    if(kb*8+1 < 23) _mm512_store_pd(spre + (long)jb*184 + (long)(kb*8+1)*8, o1);
                    if(kb*8+2 < 23) _mm512_store_pd(spre + (long)jb*184 + (long)(kb*8+2)*8, o2);
                    if(kb*8+3 < 23) _mm512_store_pd(spre + (long)jb*184 + (long)(kb*8+3)*8, o3);
                    if(kb*8+4 < 23) _mm512_store_pd(spre + (long)jb*184 + (long)(kb*8+4)*8, o4);
                    if(kb*8+5 < 23) _mm512_store_pd(spre + (long)jb*184 + (long)(kb*8+5)*8, o5);
                    if(kb*8+6 < 23) _mm512_store_pd(spre + (long)jb*184 + (long)(kb*8+6)*8, o6);
                    if(kb*8+7 < 23) _mm512_store_pd(spre + (long)jb*184 + (long)(kb*8+7)*8, o7);
                    __m512d oo0,oo1,oo2,oo3,oo4,oo5,oo6,oo7;
                    TR8(IM[0],IM[1],IM[2],IM[3],IM[4],IM[5],IM[6],IM[7],oo0,oo1,oo2,oo3,oo4,oo5,oo6,oo7);
                    if(kb*8+0 < 23) _mm512_store_pd(spre + 576 + (long)jb*184 + (long)(kb*8+0)*8, oo0);
                    if(kb*8+1 < 23) _mm512_store_pd(spre + 576 + (long)jb*184 + (long)(kb*8+1)*8, oo1);
                    if(kb*8+2 < 23) _mm512_store_pd(spre + 576 + (long)jb*184 + (long)(kb*8+2)*8, oo2);
                    if(kb*8+3 < 23) _mm512_store_pd(spre + 576 + (long)jb*184 + (long)(kb*8+3)*8, oo3);
                    if(kb*8+4 < 23) _mm512_store_pd(spre + 576 + (long)jb*184 + (long)(kb*8+4)*8, oo4);
                    if(kb*8+5 < 23) _mm512_store_pd(spre + 576 + (long)jb*184 + (long)(kb*8+5)*8, oo5);
                    if(kb*8+6 < 23) _mm512_store_pd(spre + 576 + (long)jb*184 + (long)(kb*8+6)*8, oo6);
                    if(kb*8+7 < 23) _mm512_store_pd(spre + 576 + (long)jb*184 + (long)(kb*8+7)*8, oo7);
                }
            }
        }
    }
}
static double* XV_23t = 0;
static double* CSW_23t = 0;
static double* CNAT_23t = 0;
void run_23t(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if(m < 1) m = 1;
    if(!XV_23t){ XV_23t = alloc_huge_st((long)23*1160*8 + 4096); CSW_23t = alloc_huge_st((long)23*1160*8 + 4096); CNAT_23t = alloc_huge_st((long)23*1160*8 + 4096); }
    for(long v=0; v<B; v++){
        buildc_23t(c + v*24334, CNAT_23t, CSW_23t, m >= 3);
        const double* cx = CNAT_23t;
        convin_23t(x0 + v*24334, XV_23t);
        S_23t(XV_23t, CSW_23t, 0, 0);
        P_23t(XV_23t, cx, 0);
        convout_23t(XV_23t, out1 + v*24334);
        if(m >= 2){
            S_23t(XV_23t, CSW_23t, 0, 0);
            for(long t=2; t<=m; t++){
                if((t & 1) == 0) P_23t(XV_23t, cx, t<m);
                else             S_23t(XV_23t, CSW_23t, 1, t<m);
            }
        }
        convout_23t(XV_23t, outm + v*24334);
    }
}


// ---------------- family A, L=6 (PS=37) ----------------
static void S_6(double* X, const double* C, int do_map, int do_next){
    for(int i=0;i<6;i++){
        double* sl = X + (long)i*592;
        for(int rep=0;;rep++){
            for(int j=0;j<6;j++){ dft6_v(sl + (long)j*96, sl + (long)j*96 + 8, 16);  }
            for(int k=0;k<6;k++) dft6_v(sl + k*16, sl + k*16 + 8, 96);
            if(rep==0){
                if(do_map) map_range(sl, C + (long)i*592, 36);
                if(do_next) continue;
            }
            break;
        }
    }
}
static void P_6(double* X, const double* C, int do_next){
    for(long e=0;e<36;e+=1){
        double* p = X + e*16;
        const double* cp = C + e*16;
        dft6_v(p, p+8, 592);
        for(long ee=0; ee<1; ee++){
        double* p2 = p + ee*16; const double* cp2 = cp + ee*16;
        long t=0;
        for(; t+2<=6; t+=2){
            __m512d xr0 = _mm512_load_pd(p2 + t*592);
            __m512d xi0 = _mm512_load_pd(p2 + t*592 + 8);
            __m512d xr1 = _mm512_load_pd(p2 + t*592 + 592);
            __m512d xi1 = _mm512_load_pd(p2 + t*592 + 592 + 8);
            __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp2 + t*592));
            __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp2 + t*592 + 8));
            __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cp2 + t*592 + 592));
            __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cp2 + t*592 + 592 + 8));
            map2(zr0, zi0, &xr0, &xi0);
            map2(zr1, zi1, &xr1, &xi1);
            _mm512_store_pd(p2 + t*592, xr0);
            _mm512_store_pd(p2 + t*592 + 8, xi0);
            _mm512_store_pd(p2 + t*592 + 592, xr1);
            _mm512_store_pd(p2 + t*592 + 592 + 8, xi1);
        }
        for(; t<6; t++){
            __m512d xr = _mm512_load_pd(p2 + t*592);
            __m512d xi = _mm512_load_pd(p2 + t*592 + 8);
            __m512d zr = _mm512_add_pd(xr, _mm512_load_pd(cp2 + t*592));
            __m512d zi = _mm512_add_pd(xi, _mm512_load_pd(cp2 + t*592 + 8));
            map2(zr, zi, &xr, &xi);
            _mm512_store_pd(p2 + t*592, xr);
            _mm512_store_pd(p2 + t*592 + 8, xi);
        }
        }
        if(do_next) { dft6_v(p, p+8, 592); }
    }
}
static void convin_6(const double* const* src, double* G){
    for(int i=0;i<6;i++){
        double* gp = G + (long)i*592;
        long base = (long)i*36;
        long e=0;
        for(; e+4<=36; e+=4){
            __m512d r0=_mm512_loadu_pd(src[0]+2*(base+e)), r1=_mm512_loadu_pd(src[1]+2*(base+e));
            __m512d r2=_mm512_loadu_pd(src[2]+2*(base+e)), r3=_mm512_loadu_pd(src[3]+2*(base+e));
            __m512d r4=_mm512_loadu_pd(src[4]+2*(base+e)), r5=_mm512_loadu_pd(src[5]+2*(base+e));
            __m512d r6=_mm512_loadu_pd(src[6]+2*(base+e)), r7=_mm512_loadu_pd(src[7]+2*(base+e));
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            _mm512_store_pd(gp+e*16+0,  o0); _mm512_store_pd(gp+e*16+8,  o1);
            _mm512_store_pd(gp+e*16+16, o2); _mm512_store_pd(gp+e*16+24, o3);
            _mm512_store_pd(gp+e*16+32, o4); _mm512_store_pd(gp+e*16+40, o5);
            _mm512_store_pd(gp+e*16+48, o6); _mm512_store_pd(gp+e*16+56, o7);
        }
        for(; e<36; e++){
            for(int v=0;v<8;v++){ gp[e*16+v] = src[v][2*(base+e)]; gp[e*16+8+v] = src[v][2*(base+e)+1]; }
        }
    }
}
static void convout_6(const double* G, double* const* dst, int nv){
    for(int i=0;i<6;i++){
        const double* gp = G + (long)i*592;
        long base = (long)i*36;
        long e=0;
        for(; e+4<=36; e+=4){
            __m512d r0=_mm512_load_pd(gp+e*16+0),  r1=_mm512_load_pd(gp+e*16+8);
            __m512d r2=_mm512_load_pd(gp+e*16+16), r3=_mm512_load_pd(gp+e*16+24);
            __m512d r4=_mm512_load_pd(gp+e*16+32), r5=_mm512_load_pd(gp+e*16+40);
            __m512d r6=_mm512_load_pd(gp+e*16+48), r7=_mm512_load_pd(gp+e*16+56);
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d oo[8] = {o0,o1,o2,o3,o4,o5,o6,o7};
            for(int v=0;v<nv;v++) _mm512_storeu_pd(dst[v]+2*(base+e), oo[v]);
        }
        for(; e<36; e++){
            for(int v=0;v<nv;v++){ dst[v][2*(base+e)] = gp[e*16+v]; dst[v][2*(base+e)+1] = gp[e*16+8+v]; }
        }
    }
}
static double* XG_6 = 0;
static double* CG_6 = 0;
void run_6(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if(m < 1) m = 1;
    if(!XG_6){ XG_6 = alloc_huge_st((long)6*592*8); CG_6 = alloc_huge_st((long)6*592*8); }
    long G = (B + 7) / 8;
    long rem = B % 8;
    
    const double* srcs[8]; double* dsts[8];
    for(long g=0; g<G; g++){
        long v0 = g*8;
        int nv = (int)((B - v0) < 8 ? (B - v0) : 8);
        for(int v=0; v<8; v++){
            long idx = v0 + (v < nv ? v : nv-1);
            srcs[v] = x0 + idx*2*216;
        }
        convin_6(srcs, XG_6);
        for(int v=0; v<8; v++){
            long idx = v0 + (v < nv ? v : nv-1);
            srcs[v] = c + idx*2*216;
        }
        convin_6(srcs, CG_6);
        S_6(XG_6, CG_6, 0, 0);
        P_6(XG_6, CG_6, 0);
        for(int v=0;v<nv;v++) dsts[v] = out1 + (v0+v)*2*216;
        convout_6(XG_6, dsts, nv);
        if(m >= 2){
            S_6(XG_6, CG_6, 0, 0);
            for(long t=2; t<=m; t++){
                if((t & 1) == 0) P_6(XG_6, CG_6, t<m);
                else             S_6(XG_6, CG_6, 1, t<m);
            }
        }
        for(int v=0;v<nv;v++) dsts[v] = outm + (v0+v)*2*216;
        convout_6(XG_6, dsts, nv);
    }
}


// ---------------- family A, L=8 (PS=65) ----------------
static void S_8(double* X, const double* C, int do_map, int do_next){
    for(int i=0;i<8;i++){
        double* sl = X + (long)i*1040;
        for(int rep=0;;rep++){
            for(int j=0;j<8;j++){ dft8_v(sl + (long)j*128, sl + (long)j*128 + 8, 16);  }
            for(int k=0;k<8;k++) dft8_v(sl + k*16, sl + k*16 + 8, 128);
            if(rep==0){
                if(do_map) map_range(sl, C + (long)i*1040, 64);
                if(do_next) continue;
            }
            break;
        }
    }
}
static void P_8(double* X, const double* C, int do_next){
    for(long e=0;e<64;e+=1){
        double* p = X + e*16;
        const double* cp = C + e*16;
        dft8_v(p, p+8, 1040);
        for(long ee=0; ee<1; ee++){
        double* p2 = p + ee*16; const double* cp2 = cp + ee*16;
        long t=0;
        for(; t+2<=8; t+=2){
            __m512d xr0 = _mm512_load_pd(p2 + t*1040);
            __m512d xi0 = _mm512_load_pd(p2 + t*1040 + 8);
            __m512d xr1 = _mm512_load_pd(p2 + t*1040 + 1040);
            __m512d xi1 = _mm512_load_pd(p2 + t*1040 + 1040 + 8);
            __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp2 + t*1040));
            __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp2 + t*1040 + 8));
            __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cp2 + t*1040 + 1040));
            __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cp2 + t*1040 + 1040 + 8));
            map2(zr0, zi0, &xr0, &xi0);
            map2(zr1, zi1, &xr1, &xi1);
            _mm512_store_pd(p2 + t*1040, xr0);
            _mm512_store_pd(p2 + t*1040 + 8, xi0);
            _mm512_store_pd(p2 + t*1040 + 1040, xr1);
            _mm512_store_pd(p2 + t*1040 + 1040 + 8, xi1);
        }
        for(; t<8; t++){
            __m512d xr = _mm512_load_pd(p2 + t*1040);
            __m512d xi = _mm512_load_pd(p2 + t*1040 + 8);
            __m512d zr = _mm512_add_pd(xr, _mm512_load_pd(cp2 + t*1040));
            __m512d zi = _mm512_add_pd(xi, _mm512_load_pd(cp2 + t*1040 + 8));
            map2(zr, zi, &xr, &xi);
            _mm512_store_pd(p2 + t*1040, xr);
            _mm512_store_pd(p2 + t*1040 + 8, xi);
        }
        }
        if(do_next) { dft8_v(p, p+8, 1040); }
    }
}
static void convin_8(const double* const* src, double* G){
    for(int i=0;i<8;i++){
        double* gp = G + (long)i*1040;
        long base = (long)i*64;
        long e=0;
        for(; e+4<=64; e+=4){
            __m512d r0=_mm512_loadu_pd(src[0]+2*(base+e)), r1=_mm512_loadu_pd(src[1]+2*(base+e));
            __m512d r2=_mm512_loadu_pd(src[2]+2*(base+e)), r3=_mm512_loadu_pd(src[3]+2*(base+e));
            __m512d r4=_mm512_loadu_pd(src[4]+2*(base+e)), r5=_mm512_loadu_pd(src[5]+2*(base+e));
            __m512d r6=_mm512_loadu_pd(src[6]+2*(base+e)), r7=_mm512_loadu_pd(src[7]+2*(base+e));
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            _mm512_store_pd(gp+e*16+0,  o0); _mm512_store_pd(gp+e*16+8,  o1);
            _mm512_store_pd(gp+e*16+16, o2); _mm512_store_pd(gp+e*16+24, o3);
            _mm512_store_pd(gp+e*16+32, o4); _mm512_store_pd(gp+e*16+40, o5);
            _mm512_store_pd(gp+e*16+48, o6); _mm512_store_pd(gp+e*16+56, o7);
        }
        for(; e<64; e++){
            for(int v=0;v<8;v++){ gp[e*16+v] = src[v][2*(base+e)]; gp[e*16+8+v] = src[v][2*(base+e)+1]; }
        }
    }
}
static void convout_8(const double* G, double* const* dst, int nv){
    for(int i=0;i<8;i++){
        const double* gp = G + (long)i*1040;
        long base = (long)i*64;
        long e=0;
        for(; e+4<=64; e+=4){
            __m512d r0=_mm512_load_pd(gp+e*16+0),  r1=_mm512_load_pd(gp+e*16+8);
            __m512d r2=_mm512_load_pd(gp+e*16+16), r3=_mm512_load_pd(gp+e*16+24);
            __m512d r4=_mm512_load_pd(gp+e*16+32), r5=_mm512_load_pd(gp+e*16+40);
            __m512d r6=_mm512_load_pd(gp+e*16+48), r7=_mm512_load_pd(gp+e*16+56);
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d oo[8] = {o0,o1,o2,o3,o4,o5,o6,o7};
            for(int v=0;v<nv;v++) _mm512_storeu_pd(dst[v]+2*(base+e), oo[v]);
        }
        for(; e<64; e++){
            for(int v=0;v<nv;v++){ dst[v][2*(base+e)] = gp[e*16+v]; dst[v][2*(base+e)+1] = gp[e*16+8+v]; }
        }
    }
}
static double* XG_8 = 0;
static double* CG_8 = 0;
void run_8(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if(m < 1) m = 1;
    if(!XG_8){ XG_8 = alloc_huge_st((long)8*1040*8); CG_8 = alloc_huge_st((long)8*1040*8); }
    long G = (B + 7) / 8;
    long rem = B % 8;
    
    const double* srcs[8]; double* dsts[8];
    for(long g=0; g<G; g++){
        long v0 = g*8;
        int nv = (int)((B - v0) < 8 ? (B - v0) : 8);
        for(int v=0; v<8; v++){
            long idx = v0 + (v < nv ? v : nv-1);
            srcs[v] = x0 + idx*2*512;
        }
        convin_8(srcs, XG_8);
        for(int v=0; v<8; v++){
            long idx = v0 + (v < nv ? v : nv-1);
            srcs[v] = c + idx*2*512;
        }
        convin_8(srcs, CG_8);
        S_8(XG_8, CG_8, 0, 0);
        P_8(XG_8, CG_8, 0);
        for(int v=0;v<nv;v++) dsts[v] = out1 + (v0+v)*2*512;
        convout_8(XG_8, dsts, nv);
        if(m >= 2){
            S_8(XG_8, CG_8, 0, 0);
            for(long t=2; t<=m; t++){
                if((t & 1) == 0) P_8(XG_8, CG_8, t<m);
                else             S_8(XG_8, CG_8, 1, t<m);
            }
        }
        for(int v=0;v<nv;v++) dsts[v] = outm + (v0+v)*2*512;
        convout_8(XG_8, dsts, nv);
    }
}


// ---------------- family A, L=13 (PS=169) ----------------
static void S_13(double* X, const double* C, int do_map, int do_next){
    for(int i=0;i<13;i++){
        double* sl = X + (long)i*2704;
        for(int rep=0;;rep++){
            for(int j=0;j<13;j++){ dftp13_v(sl + (long)j*208, sl + (long)j*208 + 8, 16);  }
            for(int k=0;k<13;k++) dftp13_v(sl + k*16, sl + k*16 + 8, 208);
            if(rep==0){
                if(do_map) map_range(sl, C + (long)i*2704, 169);
                if(do_next) continue;
            }
            break;
        }
    }
}
static void P_13(double* X, const double* C, int do_next){
    for(long e=0;e<169;e+=1){
        double* p = X + e*16;
        const double* cp = C + e*208;
        dftp13_v(p, p+8, 2704);
        for(long ee=0; ee<1; ee++){
        double* p2 = p + ee*16; const double* cp2 = cp + ee*16;
        long t=0;
        for(; t+2<=13; t+=2){
            __m512d xr0 = _mm512_load_pd(p2 + t*2704);
            __m512d xi0 = _mm512_load_pd(p2 + t*2704 + 8);
            __m512d xr1 = _mm512_load_pd(p2 + t*2704 + 2704);
            __m512d xi1 = _mm512_load_pd(p2 + t*2704 + 2704 + 8);
            __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp2 + t*16));
            __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp2 + t*16 + 8));
            __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cp2 + t*16 + 16));
            __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cp2 + t*16 + 24));
            map2(zr0, zi0, &xr0, &xi0);
            map2(zr1, zi1, &xr1, &xi1);
            _mm512_store_pd(p2 + t*2704, xr0);
            _mm512_store_pd(p2 + t*2704 + 8, xi0);
            _mm512_store_pd(p2 + t*2704 + 2704, xr1);
            _mm512_store_pd(p2 + t*2704 + 2704 + 8, xi1);
        }
        for(; t<13; t++){
            __m512d xr = _mm512_load_pd(p2 + t*2704);
            __m512d xi = _mm512_load_pd(p2 + t*2704 + 8);
            __m512d zr = _mm512_add_pd(xr, _mm512_load_pd(cp2 + t*16));
            __m512d zi = _mm512_add_pd(xi, _mm512_load_pd(cp2 + t*16 + 8));
            map2(zr, zi, &xr, &xi);
            _mm512_store_pd(p2 + t*2704, xr);
            _mm512_store_pd(p2 + t*2704 + 8, xi);
        }
        }
        if(do_next) { dftp13_v(p, p+8, 2704); }
    }
}
static void convin_13(const double* const* src, double* G){
    for(int i=0;i<13;i++){
        double* gp = G + (long)i*2704;
        long base = (long)i*169;
        long e=0;
        for(; e+4<=169; e+=4){
            __m512d r0=_mm512_loadu_pd(src[0]+2*(base+e)), r1=_mm512_loadu_pd(src[1]+2*(base+e));
            __m512d r2=_mm512_loadu_pd(src[2]+2*(base+e)), r3=_mm512_loadu_pd(src[3]+2*(base+e));
            __m512d r4=_mm512_loadu_pd(src[4]+2*(base+e)), r5=_mm512_loadu_pd(src[5]+2*(base+e));
            __m512d r6=_mm512_loadu_pd(src[6]+2*(base+e)), r7=_mm512_loadu_pd(src[7]+2*(base+e));
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            _mm512_store_pd(gp+e*16+0,  o0); _mm512_store_pd(gp+e*16+8,  o1);
            _mm512_store_pd(gp+e*16+16, o2); _mm512_store_pd(gp+e*16+24, o3);
            _mm512_store_pd(gp+e*16+32, o4); _mm512_store_pd(gp+e*16+40, o5);
            _mm512_store_pd(gp+e*16+48, o6); _mm512_store_pd(gp+e*16+56, o7);
        }
        for(; e<169; e++){
            for(int v=0;v<8;v++){ gp[e*16+v] = src[v][2*(base+e)]; gp[e*16+8+v] = src[v][2*(base+e)+1]; }
        }
    }
}
static void convout_13(const double* G, double* const* dst, int nv){
    for(int i=0;i<13;i++){
        const double* gp = G + (long)i*2704;
        long base = (long)i*169;
        long e=0;
        for(; e+4<=169; e+=4){
            __m512d r0=_mm512_load_pd(gp+e*16+0),  r1=_mm512_load_pd(gp+e*16+8);
            __m512d r2=_mm512_load_pd(gp+e*16+16), r3=_mm512_load_pd(gp+e*16+24);
            __m512d r4=_mm512_load_pd(gp+e*16+32), r5=_mm512_load_pd(gp+e*16+40);
            __m512d r6=_mm512_load_pd(gp+e*16+48), r7=_mm512_load_pd(gp+e*16+56);
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d oo[8] = {o0,o1,o2,o3,o4,o5,o6,o7};
            for(int v=0;v<nv;v++) _mm512_storeu_pd(dst[v]+2*(base+e), oo[v]);
        }
        for(; e<169; e++){
            for(int v=0;v<nv;v++){ dst[v][2*(base+e)] = gp[e*16+v]; dst[v][2*(base+e)+1] = gp[e*16+8+v]; }
        }
    }
}
static double* XG_13 = 0;
static double* CP_13 = 0;
static double* CG_13 = 0;
void run_13(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if(m < 1) m = 1;
    if(!XG_13){ XG_13 = alloc_huge_st((long)13*2704*8); CG_13 = alloc_huge_st((long)13*2704*8); CP_13 = alloc_huge_st((long)13*2704*8); }
    long G = (B + 7) / 8;
    long rem = B % 8;
    if(rem && rem < 6){ long Bm = B - rem; run_13t(x0 + Bm*2*2197, c + Bm*2*2197, out1 + Bm*2*2197, outm + Bm*2*2197, rem, m); B = Bm; G = B/8; if(!B) return; }
    const double* srcs[8]; double* dsts[8];
    for(long g=0; g<G; g++){
        long v0 = g*8;
        int nv = (int)((B - v0) < 8 ? (B - v0) : 8);
        for(int v=0; v<8; v++){
            long idx = v0 + (v < nv ? v : nv-1);
            srcs[v] = x0 + idx*2*2197;
        }
        convin_13(srcs, XG_13);
        for(int v=0; v<8; v++){
            long idx = v0 + (v < nv ? v : nv-1);
            srcs[v] = c + idx*2*2197;
        }
        convin_13(srcs, CG_13);
        {   // column-order copy of c for the P pass (sequential map loads)
            for(int i=0;i<13;i++){
                const double* s = CG_13 + (long)i*2704;
                double* d = CP_13 + (long)i*16;
                for(long e=0;e<169;e++){
                    _mm512_store_pd(d + e*208, _mm512_load_pd(s + e*16));
                    _mm512_store_pd(d + e*208 + 8, _mm512_load_pd(s + e*16 + 8));
                }
            }
        }
        S_13(XG_13, CG_13, 0, 0);
        P_13(XG_13, CP_13, 0);
        for(int v=0;v<nv;v++) dsts[v] = out1 + (v0+v)*2*2197;
        convout_13(XG_13, dsts, nv);
        if(m >= 2){
            S_13(XG_13, CG_13, 0, 0);
            for(long t=2; t<=m; t++){
                if((t & 1) == 0) P_13(XG_13, CP_13, t<m);
                else             S_13(XG_13, CG_13, 1, t<m);
            }
        }
        for(int v=0;v<nv;v++) dsts[v] = outm + (v0+v)*2*2197;
        convout_13(XG_13, dsts, nv);
    }
}


// ---------------- family A, L=17 (PS=289) ----------------
static void S_17(double* X, const double* C, int do_map, int do_next){
    for(int i=0;i<17;i++){
        double* sl = X + (long)i*4624;
        for(int rep=0;;rep++){
            for(int j=0;j<17;j++){ dftp17_v(sl + (long)j*272, sl + (long)j*272 + 8, 16);  }
            for(int k=0;k<17;k++) dftp17_v(sl + k*16, sl + k*16 + 8, 272);
            if(rep==0){
                if(do_map) map_range(sl, C + (long)i*4624, 289);
                if(do_next) continue;
            }
            break;
        }
    }
}
static void P_17(double* X, const double* C, int do_next){
    for(long e=0;e<289;e+=1){
        double* p = X + e*16;
        const double* cp = C + e*272;
        dftp17_v(p, p+8, 4624);
        for(long ee=0; ee<1; ee++){
        double* p2 = p + ee*16; const double* cp2 = cp + ee*16;
        long t=0;
        for(; t+2<=17; t+=2){
            __m512d xr0 = _mm512_load_pd(p2 + t*4624);
            __m512d xi0 = _mm512_load_pd(p2 + t*4624 + 8);
            __m512d xr1 = _mm512_load_pd(p2 + t*4624 + 4624);
            __m512d xi1 = _mm512_load_pd(p2 + t*4624 + 4624 + 8);
            __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp2 + t*16));
            __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp2 + t*16 + 8));
            __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cp2 + t*16 + 16));
            __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cp2 + t*16 + 24));
            map2(zr0, zi0, &xr0, &xi0);
            map2(zr1, zi1, &xr1, &xi1);
            _mm512_store_pd(p2 + t*4624, xr0);
            _mm512_store_pd(p2 + t*4624 + 8, xi0);
            _mm512_store_pd(p2 + t*4624 + 4624, xr1);
            _mm512_store_pd(p2 + t*4624 + 4624 + 8, xi1);
        }
        for(; t<17; t++){
            __m512d xr = _mm512_load_pd(p2 + t*4624);
            __m512d xi = _mm512_load_pd(p2 + t*4624 + 8);
            __m512d zr = _mm512_add_pd(xr, _mm512_load_pd(cp2 + t*16));
            __m512d zi = _mm512_add_pd(xi, _mm512_load_pd(cp2 + t*16 + 8));
            map2(zr, zi, &xr, &xi);
            _mm512_store_pd(p2 + t*4624, xr);
            _mm512_store_pd(p2 + t*4624 + 8, xi);
        }
        }
        if(do_next) { dftp17_v(p, p+8, 4624); }
    }
}
static void convin_17(const double* const* src, double* G){
    for(int i=0;i<17;i++){
        double* gp = G + (long)i*4624;
        long base = (long)i*289;
        long e=0;
        for(; e+4<=289; e+=4){
            __m512d r0=_mm512_loadu_pd(src[0]+2*(base+e)), r1=_mm512_loadu_pd(src[1]+2*(base+e));
            __m512d r2=_mm512_loadu_pd(src[2]+2*(base+e)), r3=_mm512_loadu_pd(src[3]+2*(base+e));
            __m512d r4=_mm512_loadu_pd(src[4]+2*(base+e)), r5=_mm512_loadu_pd(src[5]+2*(base+e));
            __m512d r6=_mm512_loadu_pd(src[6]+2*(base+e)), r7=_mm512_loadu_pd(src[7]+2*(base+e));
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            _mm512_store_pd(gp+e*16+0,  o0); _mm512_store_pd(gp+e*16+8,  o1);
            _mm512_store_pd(gp+e*16+16, o2); _mm512_store_pd(gp+e*16+24, o3);
            _mm512_store_pd(gp+e*16+32, o4); _mm512_store_pd(gp+e*16+40, o5);
            _mm512_store_pd(gp+e*16+48, o6); _mm512_store_pd(gp+e*16+56, o7);
        }
        for(; e<289; e++){
            for(int v=0;v<8;v++){ gp[e*16+v] = src[v][2*(base+e)]; gp[e*16+8+v] = src[v][2*(base+e)+1]; }
        }
    }
}
static void convout_17(const double* G, double* const* dst, int nv){
    for(int i=0;i<17;i++){
        const double* gp = G + (long)i*4624;
        long base = (long)i*289;
        long e=0;
        for(; e+4<=289; e+=4){
            __m512d r0=_mm512_load_pd(gp+e*16+0),  r1=_mm512_load_pd(gp+e*16+8);
            __m512d r2=_mm512_load_pd(gp+e*16+16), r3=_mm512_load_pd(gp+e*16+24);
            __m512d r4=_mm512_load_pd(gp+e*16+32), r5=_mm512_load_pd(gp+e*16+40);
            __m512d r6=_mm512_load_pd(gp+e*16+48), r7=_mm512_load_pd(gp+e*16+56);
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d oo[8] = {o0,o1,o2,o3,o4,o5,o6,o7};
            for(int v=0;v<nv;v++) _mm512_storeu_pd(dst[v]+2*(base+e), oo[v]);
        }
        for(; e<289; e++){
            for(int v=0;v<nv;v++){ dst[v][2*(base+e)] = gp[e*16+v]; dst[v][2*(base+e)+1] = gp[e*16+8+v]; }
        }
    }
}
static double* XG_17 = 0;
static double* CP_17 = 0;
static double* CG_17 = 0;
void run_17(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if(m < 1) m = 1;
    if(!XG_17){ XG_17 = alloc_huge_st((long)17*4624*8); CG_17 = alloc_huge_st((long)17*4624*8); CP_17 = alloc_huge_st((long)17*4624*8); }
    long G = (B + 7) / 8;
    long rem = B % 8;
    if(rem && rem < 5){ long Bm = B - rem; run_17t(x0 + Bm*2*4913, c + Bm*2*4913, out1 + Bm*2*4913, outm + Bm*2*4913, rem, m); B = Bm; G = B/8; if(!B) return; }
    const double* srcs[8]; double* dsts[8];
    for(long g=0; g<G; g++){
        long v0 = g*8;
        int nv = (int)((B - v0) < 8 ? (B - v0) : 8);
        for(int v=0; v<8; v++){
            long idx = v0 + (v < nv ? v : nv-1);
            srcs[v] = x0 + idx*2*4913;
        }
        convin_17(srcs, XG_17);
        for(int v=0; v<8; v++){
            long idx = v0 + (v < nv ? v : nv-1);
            srcs[v] = c + idx*2*4913;
        }
        convin_17(srcs, CG_17);
        {   // column-order copy of c for the P pass (sequential map loads)
            for(int i=0;i<17;i++){
                const double* s = CG_17 + (long)i*4624;
                double* d = CP_17 + (long)i*16;
                for(long e=0;e<289;e++){
                    _mm512_store_pd(d + e*272, _mm512_load_pd(s + e*16));
                    _mm512_store_pd(d + e*272 + 8, _mm512_load_pd(s + e*16 + 8));
                }
            }
        }
        S_17(XG_17, CG_17, 0, 0);
        P_17(XG_17, CP_17, 0);
        for(int v=0;v<nv;v++) dsts[v] = out1 + (v0+v)*2*4913;
        convout_17(XG_17, dsts, nv);
        if(m >= 2){
            S_17(XG_17, CG_17, 0, 0);
            for(long t=2; t<=m; t++){
                if((t & 1) == 0) P_17(XG_17, CP_17, t<m);
                else             S_17(XG_17, CG_17, 1, t<m);
            }
        }
        for(int v=0;v<nv;v++) dsts[v] = outm + (v0+v)*2*4913;
        convout_17(XG_17, dsts, nv);
    }
}


// ---------------- family A, L=23 (PS=529) ----------------
static void S_23(double* X, const double* C, int do_map, int do_next){
    for(int i=0;i<23;i++){
        double* sl = X + (long)i*8464;
        for(int rep=0;;rep++){
            for(int j=0;j<23;j++){ dftp23_v(sl + (long)j*368, sl + (long)j*368 + 8, 16);  }
            for(int k=0;k<23;k++) dftp23_v(sl + k*16, sl + k*16 + 8, 368);
            if(rep==0){
                if(do_map) map_range(sl, C + (long)i*8464, 529);
                if(do_next) continue;
            }
            break;
        }
    }
}
static void P_23(double* X, const double* C, int do_next){
    for(long e=0;e<529;e+=1){
        double* p = X + e*16;
        const double* cp = C + e*368;
        dftp23_v(p, p+8, 8464);
        for(long ee=0; ee<1; ee++){
        double* p2 = p + ee*16; const double* cp2 = cp + ee*16;
        long t=0;
        for(; t+2<=23; t+=2){
            __m512d xr0 = _mm512_load_pd(p2 + t*8464);
            __m512d xi0 = _mm512_load_pd(p2 + t*8464 + 8);
            __m512d xr1 = _mm512_load_pd(p2 + t*8464 + 8464);
            __m512d xi1 = _mm512_load_pd(p2 + t*8464 + 8464 + 8);
            __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp2 + t*16));
            __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp2 + t*16 + 8));
            __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cp2 + t*16 + 16));
            __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cp2 + t*16 + 24));
            map2(zr0, zi0, &xr0, &xi0);
            map2(zr1, zi1, &xr1, &xi1);
            _mm512_store_pd(p2 + t*8464, xr0);
            _mm512_store_pd(p2 + t*8464 + 8, xi0);
            _mm512_store_pd(p2 + t*8464 + 8464, xr1);
            _mm512_store_pd(p2 + t*8464 + 8464 + 8, xi1);
        }
        for(; t<23; t++){
            __m512d xr = _mm512_load_pd(p2 + t*8464);
            __m512d xi = _mm512_load_pd(p2 + t*8464 + 8);
            __m512d zr = _mm512_add_pd(xr, _mm512_load_pd(cp2 + t*16));
            __m512d zi = _mm512_add_pd(xi, _mm512_load_pd(cp2 + t*16 + 8));
            map2(zr, zi, &xr, &xi);
            _mm512_store_pd(p2 + t*8464, xr);
            _mm512_store_pd(p2 + t*8464 + 8, xi);
        }
        }
        if(do_next) { dftp23_v(p, p+8, 8464); }
    }
}
static void convin_23(const double* const* src, double* G){
    for(int i=0;i<23;i++){
        double* gp = G + (long)i*8464;
        long base = (long)i*529;
        long e=0;
        for(; e+4<=529; e+=4){
            __m512d r0=_mm512_loadu_pd(src[0]+2*(base+e)), r1=_mm512_loadu_pd(src[1]+2*(base+e));
            __m512d r2=_mm512_loadu_pd(src[2]+2*(base+e)), r3=_mm512_loadu_pd(src[3]+2*(base+e));
            __m512d r4=_mm512_loadu_pd(src[4]+2*(base+e)), r5=_mm512_loadu_pd(src[5]+2*(base+e));
            __m512d r6=_mm512_loadu_pd(src[6]+2*(base+e)), r7=_mm512_loadu_pd(src[7]+2*(base+e));
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            _mm512_store_pd(gp+e*16+0,  o0); _mm512_store_pd(gp+e*16+8,  o1);
            _mm512_store_pd(gp+e*16+16, o2); _mm512_store_pd(gp+e*16+24, o3);
            _mm512_store_pd(gp+e*16+32, o4); _mm512_store_pd(gp+e*16+40, o5);
            _mm512_store_pd(gp+e*16+48, o6); _mm512_store_pd(gp+e*16+56, o7);
        }
        for(; e<529; e++){
            for(int v=0;v<8;v++){ gp[e*16+v] = src[v][2*(base+e)]; gp[e*16+8+v] = src[v][2*(base+e)+1]; }
        }
    }
}
static void convout_23(const double* G, double* const* dst, int nv){
    for(int i=0;i<23;i++){
        const double* gp = G + (long)i*8464;
        long base = (long)i*529;
        long e=0;
        for(; e+4<=529; e+=4){
            __m512d r0=_mm512_load_pd(gp+e*16+0),  r1=_mm512_load_pd(gp+e*16+8);
            __m512d r2=_mm512_load_pd(gp+e*16+16), r3=_mm512_load_pd(gp+e*16+24);
            __m512d r4=_mm512_load_pd(gp+e*16+32), r5=_mm512_load_pd(gp+e*16+40);
            __m512d r6=_mm512_load_pd(gp+e*16+48), r7=_mm512_load_pd(gp+e*16+56);
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d oo[8] = {o0,o1,o2,o3,o4,o5,o6,o7};
            for(int v=0;v<nv;v++) _mm512_storeu_pd(dst[v]+2*(base+e), oo[v]);
        }
        for(; e<529; e++){
            for(int v=0;v<nv;v++){ dst[v][2*(base+e)] = gp[e*16+v]; dst[v][2*(base+e)+1] = gp[e*16+8+v]; }
        }
    }
}
static double* XG_23 = 0;
static double* CP_23 = 0;
static double* CG_23 = 0;
void run_23(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if(m < 1) m = 1;
    if(!XG_23){ XG_23 = alloc_huge_st((long)23*8464*8); CG_23 = alloc_huge_st((long)23*8464*8); CP_23 = alloc_huge_st((long)23*8464*8); }
    long G = (B + 7) / 8;
    long rem = B % 8;
    if(rem && rem < 8){ long Bm = B - rem; run_23t(x0 + Bm*2*12167, c + Bm*2*12167, out1 + Bm*2*12167, outm + Bm*2*12167, rem, m); B = Bm; G = B/8; if(!B) return; }
    const double* srcs[8]; double* dsts[8];
    for(long g=0; g<G; g++){
        long v0 = g*8;
        int nv = (int)((B - v0) < 8 ? (B - v0) : 8);
        for(int v=0; v<8; v++){
            long idx = v0 + (v < nv ? v : nv-1);
            srcs[v] = x0 + idx*2*12167;
        }
        convin_23(srcs, XG_23);
        for(int v=0; v<8; v++){
            long idx = v0 + (v < nv ? v : nv-1);
            srcs[v] = c + idx*2*12167;
        }
        convin_23(srcs, CG_23);
        {   // column-order copy of c for the P pass (sequential map loads)
            for(int i=0;i<23;i++){
                const double* s = CG_23 + (long)i*8464;
                double* d = CP_23 + (long)i*16;
                for(long e=0;e<529;e++){
                    _mm512_store_pd(d + e*368, _mm512_load_pd(s + e*16));
                    _mm512_store_pd(d + e*368 + 8, _mm512_load_pd(s + e*16 + 8));
                }
            }
        }
        S_23(XG_23, CG_23, 0, 0);
        P_23(XG_23, CP_23, 0);
        for(int v=0;v<nv;v++) dsts[v] = out1 + (v0+v)*2*12167;
        convout_23(XG_23, dsts, nv);
        if(m >= 2){
            S_23(XG_23, CG_23, 0, 0);
            for(long t=2; t<=m; t++){
                if((t & 1) == 0) P_23(XG_23, CP_23, t<m);
                else             S_23(XG_23, CG_23, 1, t<m);
            }
        }
        for(int v=0;v<nv;v++) dsts[v] = outm + (v0+v)*2*12167;
        convout_23(XG_23, dsts, nv);
    }
}
static double SC_36[36][16] ALIGN64;
static const int IN_36[9][4] = {{0,9,18,27},{4,13,22,31},{8,17,26,35},{12,21,30,3},{16,25,34,7},{20,29,2,11},{24,33,6,15},{28,1,10,19},{32,5,14,23}};
static const int OUT_36[4][9] = {{0,28,20,12,4,32,24,16,8},{9,1,29,21,13,5,33,25,17},{18,10,2,30,22,14,6,34,26},{27,19,11,3,31,23,15,7,35}};
static void dft36_v(double* re, double* im, long es){
for(int n2=0;n2<9;n2++){
const int* off = IN_36[n2];
__m512d t1 = _mm512_load_pd(re + (long)off[0]*es);
__m512d t2 = _mm512_load_pd(im + (long)off[0]*es);
__m512d t3 = _mm512_load_pd(re + (long)off[1]*es);
__m512d t4 = _mm512_load_pd(im + (long)off[1]*es);
__m512d t5 = _mm512_load_pd(re + (long)off[2]*es);
__m512d t6 = _mm512_load_pd(im + (long)off[2]*es);
__m512d t7 = _mm512_load_pd(re + (long)off[3]*es);
__m512d t8 = _mm512_load_pd(im + (long)off[3]*es);
__m512d t9 = _mm512_add_pd(t1, t5);
__m512d t10 = _mm512_add_pd(t2, t6);
__m512d t11 = _mm512_sub_pd(t1, t5);
__m512d t12 = _mm512_sub_pd(t2, t6);
__m512d t13 = _mm512_add_pd(t3, t7);
__m512d t14 = _mm512_add_pd(t4, t8);
__m512d t15 = _mm512_sub_pd(t3, t7);
__m512d t16 = _mm512_sub_pd(t4, t8);
__m512d t17 = _mm512_add_pd(t9, t13);
__m512d t18 = _mm512_add_pd(t10, t14);
__m512d t19 = _mm512_sub_pd(t9, t13);
__m512d t20 = _mm512_sub_pd(t10, t14);
__m512d t21 = _mm512_add_pd(t11, t16);
__m512d t22 = _mm512_sub_pd(t12, t15);
__m512d t23 = _mm512_sub_pd(t11, t16);
__m512d t24 = _mm512_add_pd(t12, t15);
_mm512_store_pd(SC_36[0*9+n2], t17); _mm512_store_pd(SC_36[0*9+n2]+8, t18);
_mm512_store_pd(SC_36[1*9+n2], t21); _mm512_store_pd(SC_36[1*9+n2]+8, t22);
_mm512_store_pd(SC_36[2*9+n2], t19); _mm512_store_pd(SC_36[2*9+n2]+8, t20);
_mm512_store_pd(SC_36[3*9+n2], t23); _mm512_store_pd(SC_36[3*9+n2]+8, t24);
}
for(int k1=0;k1<4;k1++){
const int* off = OUT_36[k1];
__m512d t25 = _mm512_load_pd(SC_36[k1*9+0]);
__m512d t26 = _mm512_load_pd(SC_36[k1*9+0]+8);
__m512d t27 = _mm512_load_pd(SC_36[k1*9+1]);
__m512d t28 = _mm512_load_pd(SC_36[k1*9+1]+8);
__m512d t29 = _mm512_load_pd(SC_36[k1*9+2]);
__m512d t30 = _mm512_load_pd(SC_36[k1*9+2]+8);
__m512d t31 = _mm512_load_pd(SC_36[k1*9+3]);
__m512d t32 = _mm512_load_pd(SC_36[k1*9+3]+8);
__m512d t33 = _mm512_load_pd(SC_36[k1*9+4]);
__m512d t34 = _mm512_load_pd(SC_36[k1*9+4]+8);
__m512d t35 = _mm512_load_pd(SC_36[k1*9+5]);
__m512d t36 = _mm512_load_pd(SC_36[k1*9+5]+8);
__m512d t37 = _mm512_load_pd(SC_36[k1*9+6]);
__m512d t38 = _mm512_load_pd(SC_36[k1*9+6]+8);
__m512d t39 = _mm512_load_pd(SC_36[k1*9+7]);
__m512d t40 = _mm512_load_pd(SC_36[k1*9+7]+8);
__m512d t41 = _mm512_load_pd(SC_36[k1*9+8]);
__m512d t42 = _mm512_load_pd(SC_36[k1*9+8]+8);
__m512d t43 = _mm512_add_pd(t31, t37);
__m512d t44 = _mm512_add_pd(t32, t38);
__m512d t45 = _mm512_sub_pd(t31, t37);
__m512d t46 = _mm512_sub_pd(t32, t38);
__m512d t47 = _mm512_add_pd(t25, t43);
__m512d t48 = _mm512_add_pd(t26, t44);
__m512d t49 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t43, t25);
__m512d t50 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t44, t26);
__m512d t51 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t46, t49);
__m512d t52 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t45, t50);
__m512d t53 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t46, t49);
__m512d t54 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t45, t50);
__m512d t55 = _mm512_add_pd(t33, t39);
__m512d t56 = _mm512_add_pd(t34, t40);
__m512d t57 = _mm512_sub_pd(t33, t39);
__m512d t58 = _mm512_sub_pd(t34, t40);
__m512d t59 = _mm512_add_pd(t27, t55);
__m512d t60 = _mm512_add_pd(t28, t56);
__m512d t61 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t55, t27);
__m512d t62 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t56, t28);
__m512d t63 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t58, t61);
__m512d t64 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t57, t62);
__m512d t65 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t58, t61);
__m512d t66 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t57, t62);
__m512d t67 = _mm512_add_pd(t35, t41);
__m512d t68 = _mm512_add_pd(t36, t42);
__m512d t69 = _mm512_sub_pd(t35, t41);
__m512d t70 = _mm512_sub_pd(t36, t42);
__m512d t71 = _mm512_add_pd(t29, t67);
__m512d t72 = _mm512_add_pd(t30, t68);
__m512d t73 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t67, t29);
__m512d t74 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t68, t30);
__m512d t75 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t70, t73);
__m512d t76 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t69, t74);
__m512d t77 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t70, t73);
__m512d t78 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t69, t74);
__m512d t79 = _mm512_mul_pd(t63, _mm512_set1_pd(0x1.8836fa2cf5039p-1));
__m512d t80 = _mm512_fnmadd_pd(t64, _mm512_set1_pd(-0x1.491b7523c161cp-1), t79);
__m512d t81 = _mm512_mul_pd(t64, _mm512_set1_pd(0x1.8836fa2cf5039p-1));
__m512d t82 = _mm512_fmadd_pd(t63, _mm512_set1_pd(-0x1.491b7523c161cp-1), t81);
__m512d t83 = _mm512_mul_pd(t65, _mm512_set1_pd(0x1.63a1a7e0b738cp-3));
__m512d t84 = _mm512_fnmadd_pd(t66, _mm512_set1_pd(-0x1.f838b8c811c17p-1), t83);
__m512d t85 = _mm512_mul_pd(t66, _mm512_set1_pd(0x1.63a1a7e0b738cp-3));
__m512d t86 = _mm512_fmadd_pd(t65, _mm512_set1_pd(-0x1.f838b8c811c17p-1), t85);
__m512d t87 = _mm512_mul_pd(t75, _mm512_set1_pd(0x1.63a1a7e0b738cp-3));
__m512d t88 = _mm512_fnmadd_pd(t76, _mm512_set1_pd(-0x1.f838b8c811c17p-1), t87);
__m512d t89 = _mm512_mul_pd(t76, _mm512_set1_pd(0x1.63a1a7e0b738cp-3));
__m512d t90 = _mm512_fmadd_pd(t75, _mm512_set1_pd(-0x1.f838b8c811c17p-1), t89);
__m512d t91 = _mm512_mul_pd(t77, _mm512_set1_pd(-0x1.e11f642522d1bp-1));
__m512d t92 = _mm512_fnmadd_pd(t78, _mm512_set1_pd(-0x1.5e3a8748a0bf8p-2), t91);
__m512d t93 = _mm512_mul_pd(t78, _mm512_set1_pd(-0x1.e11f642522d1bp-1));
__m512d t94 = _mm512_fmadd_pd(t77, _mm512_set1_pd(-0x1.5e3a8748a0bf8p-2), t93);
__m512d t95 = _mm512_add_pd(t59, t71);
__m512d t96 = _mm512_add_pd(t60, t72);
__m512d t97 = _mm512_sub_pd(t59, t71);
__m512d t98 = _mm512_sub_pd(t60, t72);
__m512d t99 = _mm512_add_pd(t47, t95);
__m512d t100 = _mm512_add_pd(t48, t96);
__m512d t101 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t95, t47);
__m512d t102 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t96, t48);
__m512d t103 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t98, t101);
__m512d t104 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t97, t102);
__m512d t105 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t98, t101);
__m512d t106 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t97, t102);
__m512d t107 = _mm512_add_pd(t80, t88);
__m512d t108 = _mm512_add_pd(t82, t90);
__m512d t109 = _mm512_sub_pd(t80, t88);
__m512d t110 = _mm512_sub_pd(t82, t90);
__m512d t111 = _mm512_add_pd(t51, t107);
__m512d t112 = _mm512_add_pd(t52, t108);
__m512d t113 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t107, t51);
__m512d t114 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t108, t52);
__m512d t115 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t110, t113);
__m512d t116 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t109, t114);
__m512d t117 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t110, t113);
__m512d t118 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t109, t114);
__m512d t119 = _mm512_add_pd(t84, t92);
__m512d t120 = _mm512_add_pd(t86, t94);
__m512d t121 = _mm512_sub_pd(t84, t92);
__m512d t122 = _mm512_sub_pd(t86, t94);
__m512d t123 = _mm512_add_pd(t53, t119);
__m512d t124 = _mm512_add_pd(t54, t120);
__m512d t125 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t119, t53);
__m512d t126 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t120, t54);
__m512d t127 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t122, t125);
__m512d t128 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t121, t126);
__m512d t129 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t122, t125);
__m512d t130 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t121, t126);
_mm512_store_pd(re + (long)off[0]*es, t99); _mm512_store_pd(im + (long)off[0]*es, t100);
_mm512_store_pd(re + (long)off[1]*es, t111); _mm512_store_pd(im + (long)off[1]*es, t112);
_mm512_store_pd(re + (long)off[2]*es, t123); _mm512_store_pd(im + (long)off[2]*es, t124);
_mm512_store_pd(re + (long)off[3]*es, t103); _mm512_store_pd(im + (long)off[3]*es, t104);
_mm512_store_pd(re + (long)off[4]*es, t115); _mm512_store_pd(im + (long)off[4]*es, t116);
_mm512_store_pd(re + (long)off[5]*es, t127); _mm512_store_pd(im + (long)off[5]*es, t128);
_mm512_store_pd(re + (long)off[6]*es, t105); _mm512_store_pd(im + (long)off[6]*es, t106);
_mm512_store_pd(re + (long)off[7]*es, t117); _mm512_store_pd(im + (long)off[7]*es, t118);
_mm512_store_pd(re + (long)off[8]*es, t129); _mm512_store_pd(im + (long)off[8]*es, t130);
}
}
static double SC_45[45][16] ALIGN64;
static const int IN_45[9][5] = {{0,9,18,27,36},{5,14,23,32,41},{10,19,28,37,1},{15,24,33,42,6},{20,29,38,2,11},{25,34,43,7,16},{30,39,3,12,21},{35,44,8,17,26},{40,4,13,22,31}};
static const int OUT_45[5][9] = {{0,10,20,30,40,5,15,25,35},{36,1,11,21,31,41,6,16,26},{27,37,2,12,22,32,42,7,17},{18,28,38,3,13,23,33,43,8},{9,19,29,39,4,14,24,34,44}};
static void dft45_v(double* re, double* im, long es){
for(int n2=0;n2<9;n2++){
const int* off = IN_45[n2];
__m512d t1 = _mm512_load_pd(re + (long)off[0]*es);
__m512d t2 = _mm512_load_pd(im + (long)off[0]*es);
__m512d t3 = _mm512_load_pd(re + (long)off[1]*es);
__m512d t4 = _mm512_load_pd(im + (long)off[1]*es);
__m512d t5 = _mm512_load_pd(re + (long)off[2]*es);
__m512d t6 = _mm512_load_pd(im + (long)off[2]*es);
__m512d t7 = _mm512_load_pd(re + (long)off[3]*es);
__m512d t8 = _mm512_load_pd(im + (long)off[3]*es);
__m512d t9 = _mm512_load_pd(re + (long)off[4]*es);
__m512d t10 = _mm512_load_pd(im + (long)off[4]*es);
__m512d t11 = _mm512_add_pd(t3, t9);
__m512d t12 = _mm512_add_pd(t4, t10);
__m512d t13 = _mm512_add_pd(t5, t7);
__m512d t14 = _mm512_add_pd(t6, t8);
__m512d t15 = _mm512_sub_pd(t3, t9);
__m512d t16 = _mm512_sub_pd(t4, t10);
__m512d t17 = _mm512_sub_pd(t5, t7);
__m512d t18 = _mm512_sub_pd(t6, t8);
__m512d t19 = _mm512_add_pd(t11, t13);
__m512d t20 = _mm512_add_pd(t12, t14);
__m512d t21 = _mm512_add_pd(t1, t19);
__m512d t22 = _mm512_add_pd(t2, t20);
__m512d t23 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.3c6ef372fe950p-2), t11, t1);
__m512d t24 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.9e3779b97f4a7p-1), t13, t23);
__m512d t25 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.3c6ef372fe950p-2), t12, t2);
__m512d t26 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.9e3779b97f4a7p-1), t14, t25);
__m512d t27 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.9e3779b97f4a7p-1), t11, t1);
__m512d t28 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.3c6ef372fe950p-2), t13, t27);
__m512d t29 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.9e3779b97f4a7p-1), t12, t2);
__m512d t30 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.3c6ef372fe950p-2), t14, t29);
__m512d t31 = _mm512_mul_pd(_mm512_set1_pd(-0x1.e6f0e134454ffp-1), t15);
__m512d t32 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.2cf2304755a5fp-1), t17, t31);
__m512d t33 = _mm512_mul_pd(_mm512_set1_pd(-0x1.e6f0e134454ffp-1), t16);
__m512d t34 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.2cf2304755a5fp-1), t18, t33);
__m512d t35 = _mm512_mul_pd(_mm512_set1_pd(-0x1.2cf2304755a5fp-1), t15);
__m512d t36 = _mm512_fnmadd_pd(_mm512_set1_pd(-0x1.e6f0e134454ffp-1), t17, t35);
__m512d t37 = _mm512_mul_pd(_mm512_set1_pd(-0x1.2cf2304755a5fp-1), t16);
__m512d t38 = _mm512_fnmadd_pd(_mm512_set1_pd(-0x1.e6f0e134454ffp-1), t18, t37);
__m512d t39 = _mm512_sub_pd(t24, t34);
__m512d t40 = _mm512_add_pd(t26, t32);
__m512d t41 = _mm512_add_pd(t24, t34);
__m512d t42 = _mm512_sub_pd(t26, t32);
__m512d t43 = _mm512_sub_pd(t28, t38);
__m512d t44 = _mm512_add_pd(t30, t36);
__m512d t45 = _mm512_add_pd(t28, t38);
__m512d t46 = _mm512_sub_pd(t30, t36);
_mm512_store_pd(SC_45[0*9+n2], t21); _mm512_store_pd(SC_45[0*9+n2]+8, t22);
_mm512_store_pd(SC_45[1*9+n2], t39); _mm512_store_pd(SC_45[1*9+n2]+8, t40);
_mm512_store_pd(SC_45[2*9+n2], t43); _mm512_store_pd(SC_45[2*9+n2]+8, t44);
_mm512_store_pd(SC_45[3*9+n2], t45); _mm512_store_pd(SC_45[3*9+n2]+8, t46);
_mm512_store_pd(SC_45[4*9+n2], t41); _mm512_store_pd(SC_45[4*9+n2]+8, t42);
}
for(int k1=0;k1<5;k1++){
const int* off = OUT_45[k1];
__m512d t47 = _mm512_load_pd(SC_45[k1*9+0]);
__m512d t48 = _mm512_load_pd(SC_45[k1*9+0]+8);
__m512d t49 = _mm512_load_pd(SC_45[k1*9+1]);
__m512d t50 = _mm512_load_pd(SC_45[k1*9+1]+8);
__m512d t51 = _mm512_load_pd(SC_45[k1*9+2]);
__m512d t52 = _mm512_load_pd(SC_45[k1*9+2]+8);
__m512d t53 = _mm512_load_pd(SC_45[k1*9+3]);
__m512d t54 = _mm512_load_pd(SC_45[k1*9+3]+8);
__m512d t55 = _mm512_load_pd(SC_45[k1*9+4]);
__m512d t56 = _mm512_load_pd(SC_45[k1*9+4]+8);
__m512d t57 = _mm512_load_pd(SC_45[k1*9+5]);
__m512d t58 = _mm512_load_pd(SC_45[k1*9+5]+8);
__m512d t59 = _mm512_load_pd(SC_45[k1*9+6]);
__m512d t60 = _mm512_load_pd(SC_45[k1*9+6]+8);
__m512d t61 = _mm512_load_pd(SC_45[k1*9+7]);
__m512d t62 = _mm512_load_pd(SC_45[k1*9+7]+8);
__m512d t63 = _mm512_load_pd(SC_45[k1*9+8]);
__m512d t64 = _mm512_load_pd(SC_45[k1*9+8]+8);
__m512d t65 = _mm512_add_pd(t53, t59);
__m512d t66 = _mm512_add_pd(t54, t60);
__m512d t67 = _mm512_sub_pd(t53, t59);
__m512d t68 = _mm512_sub_pd(t54, t60);
__m512d t69 = _mm512_add_pd(t47, t65);
__m512d t70 = _mm512_add_pd(t48, t66);
__m512d t71 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t65, t47);
__m512d t72 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t66, t48);
__m512d t73 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t68, t71);
__m512d t74 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t67, t72);
__m512d t75 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t68, t71);
__m512d t76 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t67, t72);
__m512d t77 = _mm512_add_pd(t55, t61);
__m512d t78 = _mm512_add_pd(t56, t62);
__m512d t79 = _mm512_sub_pd(t55, t61);
__m512d t80 = _mm512_sub_pd(t56, t62);
__m512d t81 = _mm512_add_pd(t49, t77);
__m512d t82 = _mm512_add_pd(t50, t78);
__m512d t83 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t77, t49);
__m512d t84 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t78, t50);
__m512d t85 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t80, t83);
__m512d t86 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t79, t84);
__m512d t87 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t80, t83);
__m512d t88 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t79, t84);
__m512d t89 = _mm512_add_pd(t57, t63);
__m512d t90 = _mm512_add_pd(t58, t64);
__m512d t91 = _mm512_sub_pd(t57, t63);
__m512d t92 = _mm512_sub_pd(t58, t64);
__m512d t93 = _mm512_add_pd(t51, t89);
__m512d t94 = _mm512_add_pd(t52, t90);
__m512d t95 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t89, t51);
__m512d t96 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t90, t52);
__m512d t97 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t92, t95);
__m512d t98 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t91, t96);
__m512d t99 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t92, t95);
__m512d t100 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t91, t96);
__m512d t101 = _mm512_mul_pd(t85, _mm512_set1_pd(0x1.8836fa2cf5039p-1));
__m512d t102 = _mm512_fnmadd_pd(t86, _mm512_set1_pd(-0x1.491b7523c161cp-1), t101);
__m512d t103 = _mm512_mul_pd(t86, _mm512_set1_pd(0x1.8836fa2cf5039p-1));
__m512d t104 = _mm512_fmadd_pd(t85, _mm512_set1_pd(-0x1.491b7523c161cp-1), t103);
__m512d t105 = _mm512_mul_pd(t87, _mm512_set1_pd(0x1.63a1a7e0b738cp-3));
__m512d t106 = _mm512_fnmadd_pd(t88, _mm512_set1_pd(-0x1.f838b8c811c17p-1), t105);
__m512d t107 = _mm512_mul_pd(t88, _mm512_set1_pd(0x1.63a1a7e0b738cp-3));
__m512d t108 = _mm512_fmadd_pd(t87, _mm512_set1_pd(-0x1.f838b8c811c17p-1), t107);
__m512d t109 = _mm512_mul_pd(t97, _mm512_set1_pd(0x1.63a1a7e0b738cp-3));
__m512d t110 = _mm512_fnmadd_pd(t98, _mm512_set1_pd(-0x1.f838b8c811c17p-1), t109);
__m512d t111 = _mm512_mul_pd(t98, _mm512_set1_pd(0x1.63a1a7e0b738cp-3));
__m512d t112 = _mm512_fmadd_pd(t97, _mm512_set1_pd(-0x1.f838b8c811c17p-1), t111);
__m512d t113 = _mm512_mul_pd(t99, _mm512_set1_pd(-0x1.e11f642522d1bp-1));
__m512d t114 = _mm512_fnmadd_pd(t100, _mm512_set1_pd(-0x1.5e3a8748a0bf8p-2), t113);
__m512d t115 = _mm512_mul_pd(t100, _mm512_set1_pd(-0x1.e11f642522d1bp-1));
__m512d t116 = _mm512_fmadd_pd(t99, _mm512_set1_pd(-0x1.5e3a8748a0bf8p-2), t115);
__m512d t117 = _mm512_add_pd(t81, t93);
__m512d t118 = _mm512_add_pd(t82, t94);
__m512d t119 = _mm512_sub_pd(t81, t93);
__m512d t120 = _mm512_sub_pd(t82, t94);
__m512d t121 = _mm512_add_pd(t69, t117);
__m512d t122 = _mm512_add_pd(t70, t118);
__m512d t123 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t117, t69);
__m512d t124 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t118, t70);
__m512d t125 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t120, t123);
__m512d t126 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t119, t124);
__m512d t127 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t120, t123);
__m512d t128 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t119, t124);
__m512d t129 = _mm512_add_pd(t102, t110);
__m512d t130 = _mm512_add_pd(t104, t112);
__m512d t131 = _mm512_sub_pd(t102, t110);
__m512d t132 = _mm512_sub_pd(t104, t112);
__m512d t133 = _mm512_add_pd(t73, t129);
__m512d t134 = _mm512_add_pd(t74, t130);
__m512d t135 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t129, t73);
__m512d t136 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t130, t74);
__m512d t137 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t132, t135);
__m512d t138 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t131, t136);
__m512d t139 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t132, t135);
__m512d t140 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t131, t136);
__m512d t141 = _mm512_add_pd(t106, t114);
__m512d t142 = _mm512_add_pd(t108, t116);
__m512d t143 = _mm512_sub_pd(t106, t114);
__m512d t144 = _mm512_sub_pd(t108, t116);
__m512d t145 = _mm512_add_pd(t75, t141);
__m512d t146 = _mm512_add_pd(t76, t142);
__m512d t147 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t141, t75);
__m512d t148 = _mm512_fmadd_pd(_mm512_set1_pd(-0x1.0000000000000p-1), t142, t76);
__m512d t149 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t144, t147);
__m512d t150 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t143, t148);
__m512d t151 = _mm512_fnmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t144, t147);
__m512d t152 = _mm512_fmadd_pd(_mm512_set1_pd(0x1.bb67ae8584caap-1), t143, t148);
_mm512_store_pd(re + (long)off[0]*es, t121); _mm512_store_pd(im + (long)off[0]*es, t122);
_mm512_store_pd(re + (long)off[1]*es, t133); _mm512_store_pd(im + (long)off[1]*es, t134);
_mm512_store_pd(re + (long)off[2]*es, t145); _mm512_store_pd(im + (long)off[2]*es, t146);
_mm512_store_pd(re + (long)off[3]*es, t125); _mm512_store_pd(im + (long)off[3]*es, t126);
_mm512_store_pd(re + (long)off[4]*es, t137); _mm512_store_pd(im + (long)off[4]*es, t138);
_mm512_store_pd(re + (long)off[5]*es, t149); _mm512_store_pd(im + (long)off[5]*es, t150);
_mm512_store_pd(re + (long)off[6]*es, t127); _mm512_store_pd(im + (long)off[6]*es, t128);
_mm512_store_pd(re + (long)off[7]*es, t139); _mm512_store_pd(im + (long)off[7]*es, t140);
_mm512_store_pd(re + (long)off[8]*es, t151); _mm512_store_pd(im + (long)off[8]*es, t152);
}
}
static double SC_64[64][16] ALIGN64;
static const double TW_64[128] ALIGN64 = {0x1.0000000000000p+0,-0x0.0p+0,0x1.0000000000000p+0,-0x0.0p+0,0x1.0000000000000p+0,-0x0.0p+0,0x1.0000000000000p+0,-0x0.0p+0,0x1.0000000000000p+0,-0x0.0p+0,0x1.0000000000000p+0,-0x0.0p+0,0x1.0000000000000p+0,-0x0.0p+0,0x1.0000000000000p+0,-0x0.0p+0,0x1.0000000000000p+0,-0x0.0p+0,0x1.fd88da3d12526p-1,-0x1.917a6bc29b42cp-4,0x1.f6297cff75cb0p-1,-0x1.8f8b83c69a60ap-3,0x1.e9f4156c62ddap-1,-0x1.294062ed59f05p-2,0x1.d906bcf328d46p-1,-0x1.87de2a6aea963p-2,0x1.c38b2f180bdb1p-1,-0x1.e2b5d3806f63bp-2,0x1.a9b66290ea1a3p-1,-0x1.1c73b39ae68c8p-1,0x1.8bc806b151741p-1,-0x1.44cf325091dd6p-1,0x1.0000000000000p+0,-0x0.0p+0,0x1.f6297cff75cb0p-1,-0x1.8f8b83c69a60ap-3,0x1.d906bcf328d46p-1,-0x1.87de2a6aea963p-2,0x1.a9b66290ea1a3p-1,-0x1.1c73b39ae68c8p-1,0x1.6a09e667f3bcdp-1,-0x1.6a09e667f3bccp-1,0x1.1c73b39ae68c9p-1,-0x1.a9b66290ea1a3p-1,0x1.87de2a6aea964p-2,-0x1.d906bcf328d46p-1,0x1.8f8b83c69a60dp-3,-0x1.f6297cff75cb0p-1,0x1.0000000000000p+0,-0x0.0p+0,0x1.e9f4156c62ddap-1,-0x1.294062ed59f05p-2,0x1.a9b66290ea1a3p-1,-0x1.1c73b39ae68c8p-1,0x1.44cf325091dd6p-1,-0x1.8bc806b151741p-1,0x1.87de2a6aea964p-2,-0x1.d906bcf328d46p-1,0x1.917a6bc29b438p-4,-0x1.fd88da3d12525p-1,-0x1.8f8b83c69a608p-3,-0x1.f6297cff75cb0p-1,-0x1.e2b5d3806f63cp-2,-0x1.c38b2f180bdb1p-1,0x1.0000000000000p+0,-0x0.0p+0,0x1.d906bcf328d46p-1,-0x1.87de2a6aea963p-2,0x1.6a09e667f3bcdp-1,-0x1.6a09e667f3bccp-1,0x1.87de2a6aea964p-2,-0x1.d906bcf328d46p-1,0x1.1a62633145c07p-54,-0x1.0000000000000p+0,-0x1.87de2a6aea962p-2,-0x1.d906bcf328d46p-1,-0x1.6a09e667f3bccp-1,-0x1.6a09e667f3bcdp-1,-0x1.d906bcf328d46p-1,-0x1.87de2a6aea965p-2,0x1.0000000000000p+0,-0x0.0p+0,0x1.c38b2f180bdb1p-1,-0x1.e2b5d3806f63bp-2,0x1.1c73b39ae68c9p-1,-0x1.a9b66290ea1a3p-1,0x1.917a6bc29b438p-4,-0x1.fd88da3d12525p-1,-0x1.87de2a6aea962p-2,-0x1.d906bcf328d46p-1,-0x1.8bc806b151741p-1,-0x1.44cf325091dd6p-1,-0x1.f6297cff75cb0p-1,-0x1.8f8b83c69a617p-3,-0x1.e9f4156c62ddbp-1,0x1.294062ed59f01p-2,0x1.0000000000000p+0,-0x0.0p+0,0x1.a9b66290ea1a3p-1,-0x1.1c73b39ae68c8p-1,0x1.87de2a6aea964p-2,-0x1.d906bcf328d46p-1,-0x1.8f8b83c69a608p-3,-0x1.f6297cff75cb0p-1,-0x1.6a09e667f3bccp-1,-0x1.6a09e667f3bcdp-1,-0x1.f6297cff75cb0p-1,-0x1.8f8b83c69a617p-3,-0x1.d906bcf328d47p-1,0x1.87de2a6aea961p-2,-0x1.1c73b39ae68c8p-1,0x1.a9b66290ea1a3p-1,0x1.0000000000000p+0,-0x0.0p+0,0x1.8bc806b151741p-1,-0x1.44cf325091dd6p-1,0x1.8f8b83c69a60dp-3,-0x1.f6297cff75cb0p-1,-0x1.e2b5d3806f63cp-2,-0x1.c38b2f180bdb1p-1,-0x1.d906bcf328d46p-1,-0x1.87de2a6aea965p-2,-0x1.e9f4156c62ddbp-1,0x1.294062ed59f01p-2,-0x1.1c73b39ae68c8p-1,0x1.a9b66290ea1a3p-1,0x1.917a6bc29b407p-4,0x1.fd88da3d12526p-1};
static void dft64_v(double* re, double* im, long es){
for(int r=0;r<8;r++){
const double* twp = TW_64 + r*16;
double* rb = re + (long)r*es; double* ib = im + (long)r*es;
__m512d t1 = _mm512_load_pd(rb + (long)0*es);
__m512d t2 = _mm512_load_pd(ib + (long)0*es);
__m512d t3 = _mm512_load_pd(rb + (long)8*es);
__m512d t4 = _mm512_load_pd(ib + (long)8*es);
__m512d t5 = _mm512_load_pd(rb + (long)16*es);
__m512d t6 = _mm512_load_pd(ib + (long)16*es);
__m512d t7 = _mm512_load_pd(rb + (long)24*es);
__m512d t8 = _mm512_load_pd(ib + (long)24*es);
__m512d t9 = _mm512_load_pd(rb + (long)32*es);
__m512d t10 = _mm512_load_pd(ib + (long)32*es);
__m512d t11 = _mm512_load_pd(rb + (long)40*es);
__m512d t12 = _mm512_load_pd(ib + (long)40*es);
__m512d t13 = _mm512_load_pd(rb + (long)48*es);
__m512d t14 = _mm512_load_pd(ib + (long)48*es);
__m512d t15 = _mm512_load_pd(rb + (long)56*es);
__m512d t16 = _mm512_load_pd(ib + (long)56*es);
__m512d t17 = _mm512_add_pd(t1, t9);
__m512d t18 = _mm512_add_pd(t2, t10);
__m512d t19 = _mm512_sub_pd(t1, t9);
__m512d t20 = _mm512_sub_pd(t2, t10);
__m512d t21 = _mm512_add_pd(t5, t13);
__m512d t22 = _mm512_add_pd(t6, t14);
__m512d t23 = _mm512_sub_pd(t5, t13);
__m512d t24 = _mm512_sub_pd(t6, t14);
__m512d t25 = _mm512_add_pd(t17, t21);
__m512d t26 = _mm512_add_pd(t18, t22);
__m512d t27 = _mm512_sub_pd(t17, t21);
__m512d t28 = _mm512_sub_pd(t18, t22);
__m512d t29 = _mm512_add_pd(t19, t24);
__m512d t30 = _mm512_sub_pd(t20, t23);
__m512d t31 = _mm512_sub_pd(t19, t24);
__m512d t32 = _mm512_add_pd(t20, t23);
__m512d t33 = _mm512_add_pd(t3, t11);
__m512d t34 = _mm512_add_pd(t4, t12);
__m512d t35 = _mm512_sub_pd(t3, t11);
__m512d t36 = _mm512_sub_pd(t4, t12);
__m512d t37 = _mm512_add_pd(t7, t15);
__m512d t38 = _mm512_add_pd(t8, t16);
__m512d t39 = _mm512_sub_pd(t7, t15);
__m512d t40 = _mm512_sub_pd(t8, t16);
__m512d t41 = _mm512_add_pd(t33, t37);
__m512d t42 = _mm512_add_pd(t34, t38);
__m512d t43 = _mm512_sub_pd(t33, t37);
__m512d t44 = _mm512_sub_pd(t34, t38);
__m512d t45 = _mm512_add_pd(t35, t40);
__m512d t46 = _mm512_sub_pd(t36, t39);
__m512d t47 = _mm512_sub_pd(t35, t40);
__m512d t48 = _mm512_add_pd(t36, t39);
__m512d t49 = _mm512_add_pd(t25, t41);
__m512d t50 = _mm512_add_pd(t26, t42);
__m512d t51 = _mm512_sub_pd(t25, t41);
__m512d t52 = _mm512_sub_pd(t26, t42);
__m512d t53 = _mm512_add_pd(t45, t46);
__m512d t54 = _mm512_mul_pd(_mm512_set1_pd(0x1.6a09e667f3bcdp-1), t53);
__m512d t55 = _mm512_sub_pd(t46, t45);
__m512d t56 = _mm512_mul_pd(_mm512_set1_pd(0x1.6a09e667f3bcdp-1), t55);
__m512d t57 = _mm512_add_pd(t29, t54);
__m512d t58 = _mm512_add_pd(t30, t56);
__m512d t59 = _mm512_sub_pd(t29, t54);
__m512d t60 = _mm512_sub_pd(t30, t56);
__m512d t61 = _mm512_add_pd(t27, t44);
__m512d t62 = _mm512_sub_pd(t28, t43);
__m512d t63 = _mm512_sub_pd(t27, t44);
__m512d t64 = _mm512_add_pd(t28, t43);
__m512d t65 = _mm512_sub_pd(t48, t47);
__m512d t66 = _mm512_mul_pd(_mm512_set1_pd(0x1.6a09e667f3bcdp-1), t65);
__m512d t67 = _mm512_add_pd(t47, t48);
__m512d t68 = _mm512_mul_pd(_mm512_set1_pd(0x1.6a09e667f3bcdp-1), t67);
__m512d t69 = _mm512_add_pd(t31, t66);
__m512d t70 = _mm512_sub_pd(t32, t68);
__m512d t71 = _mm512_sub_pd(t31, t66);
__m512d t72 = _mm512_add_pd(t32, t68);
_mm512_store_pd(&SC_64[0][0] + (0*8+r)*16, t49); _mm512_store_pd(&SC_64[0][0] + (0*8+r)*16+8, t50);
__m512d tvr1, tvi1, twr1, twi1;
BCASTV(twr1, twp[2]); BCASTV(twi1, twp[3]);
tvr1 = _mm512_fnmadd_pd(t58, twi1, _mm512_mul_pd(t57, twr1));
tvi1 = _mm512_fmadd_pd(t57, twi1, _mm512_mul_pd(t58, twr1));
_mm512_store_pd(&SC_64[0][0] + (1*8+r)*16, tvr1); _mm512_store_pd(&SC_64[0][0] + (1*8+r)*16+8, tvi1);
__m512d tvr2, tvi2, twr2, twi2;
BCASTV(twr2, twp[4]); BCASTV(twi2, twp[5]);
tvr2 = _mm512_fnmadd_pd(t62, twi2, _mm512_mul_pd(t61, twr2));
tvi2 = _mm512_fmadd_pd(t61, twi2, _mm512_mul_pd(t62, twr2));
_mm512_store_pd(&SC_64[0][0] + (2*8+r)*16, tvr2); _mm512_store_pd(&SC_64[0][0] + (2*8+r)*16+8, tvi2);
__m512d tvr3, tvi3, twr3, twi3;
BCASTV(twr3, twp[6]); BCASTV(twi3, twp[7]);
tvr3 = _mm512_fnmadd_pd(t70, twi3, _mm512_mul_pd(t69, twr3));
tvi3 = _mm512_fmadd_pd(t69, twi3, _mm512_mul_pd(t70, twr3));
_mm512_store_pd(&SC_64[0][0] + (3*8+r)*16, tvr3); _mm512_store_pd(&SC_64[0][0] + (3*8+r)*16+8, tvi3);
__m512d tvr4, tvi4, twr4, twi4;
BCASTV(twr4, twp[8]); BCASTV(twi4, twp[9]);
tvr4 = _mm512_fnmadd_pd(t52, twi4, _mm512_mul_pd(t51, twr4));
tvi4 = _mm512_fmadd_pd(t51, twi4, _mm512_mul_pd(t52, twr4));
_mm512_store_pd(&SC_64[0][0] + (4*8+r)*16, tvr4); _mm512_store_pd(&SC_64[0][0] + (4*8+r)*16+8, tvi4);
__m512d tvr5, tvi5, twr5, twi5;
BCASTV(twr5, twp[10]); BCASTV(twi5, twp[11]);
tvr5 = _mm512_fnmadd_pd(t60, twi5, _mm512_mul_pd(t59, twr5));
tvi5 = _mm512_fmadd_pd(t59, twi5, _mm512_mul_pd(t60, twr5));
_mm512_store_pd(&SC_64[0][0] + (5*8+r)*16, tvr5); _mm512_store_pd(&SC_64[0][0] + (5*8+r)*16+8, tvi5);
__m512d tvr6, tvi6, twr6, twi6;
BCASTV(twr6, twp[12]); BCASTV(twi6, twp[13]);
tvr6 = _mm512_fnmadd_pd(t64, twi6, _mm512_mul_pd(t63, twr6));
tvi6 = _mm512_fmadd_pd(t63, twi6, _mm512_mul_pd(t64, twr6));
_mm512_store_pd(&SC_64[0][0] + (6*8+r)*16, tvr6); _mm512_store_pd(&SC_64[0][0] + (6*8+r)*16+8, tvi6);
__m512d tvr7, tvi7, twr7, twi7;
BCASTV(twr7, twp[14]); BCASTV(twi7, twp[15]);
tvr7 = _mm512_fnmadd_pd(t72, twi7, _mm512_mul_pd(t71, twr7));
tvi7 = _mm512_fmadd_pd(t71, twi7, _mm512_mul_pd(t72, twr7));
_mm512_store_pd(&SC_64[0][0] + (7*8+r)*16, tvr7); _mm512_store_pd(&SC_64[0][0] + (7*8+r)*16+8, tvi7);
}
for(int d=0;d<8;d++){
double* rb = re + (long)d*es; double* ib = im + (long)d*es;
__m512d t73 = _mm512_load_pd(&SC_64[0][0] + (d*8+0)*16);
__m512d t74 = _mm512_load_pd(&SC_64[0][0] + (d*8+0)*16+8);
__m512d t75 = _mm512_load_pd(&SC_64[0][0] + (d*8+1)*16);
__m512d t76 = _mm512_load_pd(&SC_64[0][0] + (d*8+1)*16+8);
__m512d t77 = _mm512_load_pd(&SC_64[0][0] + (d*8+2)*16);
__m512d t78 = _mm512_load_pd(&SC_64[0][0] + (d*8+2)*16+8);
__m512d t79 = _mm512_load_pd(&SC_64[0][0] + (d*8+3)*16);
__m512d t80 = _mm512_load_pd(&SC_64[0][0] + (d*8+3)*16+8);
__m512d t81 = _mm512_load_pd(&SC_64[0][0] + (d*8+4)*16);
__m512d t82 = _mm512_load_pd(&SC_64[0][0] + (d*8+4)*16+8);
__m512d t83 = _mm512_load_pd(&SC_64[0][0] + (d*8+5)*16);
__m512d t84 = _mm512_load_pd(&SC_64[0][0] + (d*8+5)*16+8);
__m512d t85 = _mm512_load_pd(&SC_64[0][0] + (d*8+6)*16);
__m512d t86 = _mm512_load_pd(&SC_64[0][0] + (d*8+6)*16+8);
__m512d t87 = _mm512_load_pd(&SC_64[0][0] + (d*8+7)*16);
__m512d t88 = _mm512_load_pd(&SC_64[0][0] + (d*8+7)*16+8);
__m512d t89 = _mm512_add_pd(t73, t81);
__m512d t90 = _mm512_add_pd(t74, t82);
__m512d t91 = _mm512_sub_pd(t73, t81);
__m512d t92 = _mm512_sub_pd(t74, t82);
__m512d t93 = _mm512_add_pd(t77, t85);
__m512d t94 = _mm512_add_pd(t78, t86);
__m512d t95 = _mm512_sub_pd(t77, t85);
__m512d t96 = _mm512_sub_pd(t78, t86);
__m512d t97 = _mm512_add_pd(t89, t93);
__m512d t98 = _mm512_add_pd(t90, t94);
__m512d t99 = _mm512_sub_pd(t89, t93);
__m512d t100 = _mm512_sub_pd(t90, t94);
__m512d t101 = _mm512_add_pd(t91, t96);
__m512d t102 = _mm512_sub_pd(t92, t95);
__m512d t103 = _mm512_sub_pd(t91, t96);
__m512d t104 = _mm512_add_pd(t92, t95);
__m512d t105 = _mm512_add_pd(t75, t83);
__m512d t106 = _mm512_add_pd(t76, t84);
__m512d t107 = _mm512_sub_pd(t75, t83);
__m512d t108 = _mm512_sub_pd(t76, t84);
__m512d t109 = _mm512_add_pd(t79, t87);
__m512d t110 = _mm512_add_pd(t80, t88);
__m512d t111 = _mm512_sub_pd(t79, t87);
__m512d t112 = _mm512_sub_pd(t80, t88);
__m512d t113 = _mm512_add_pd(t105, t109);
__m512d t114 = _mm512_add_pd(t106, t110);
__m512d t115 = _mm512_sub_pd(t105, t109);
__m512d t116 = _mm512_sub_pd(t106, t110);
__m512d t117 = _mm512_add_pd(t107, t112);
__m512d t118 = _mm512_sub_pd(t108, t111);
__m512d t119 = _mm512_sub_pd(t107, t112);
__m512d t120 = _mm512_add_pd(t108, t111);
__m512d t121 = _mm512_add_pd(t97, t113);
__m512d t122 = _mm512_add_pd(t98, t114);
__m512d t123 = _mm512_sub_pd(t97, t113);
__m512d t124 = _mm512_sub_pd(t98, t114);
__m512d t125 = _mm512_add_pd(t117, t118);
__m512d t126 = _mm512_mul_pd(_mm512_set1_pd(0x1.6a09e667f3bcdp-1), t125);
__m512d t127 = _mm512_sub_pd(t118, t117);
__m512d t128 = _mm512_mul_pd(_mm512_set1_pd(0x1.6a09e667f3bcdp-1), t127);
__m512d t129 = _mm512_add_pd(t101, t126);
__m512d t130 = _mm512_add_pd(t102, t128);
__m512d t131 = _mm512_sub_pd(t101, t126);
__m512d t132 = _mm512_sub_pd(t102, t128);
__m512d t133 = _mm512_add_pd(t99, t116);
__m512d t134 = _mm512_sub_pd(t100, t115);
__m512d t135 = _mm512_sub_pd(t99, t116);
__m512d t136 = _mm512_add_pd(t100, t115);
__m512d t137 = _mm512_sub_pd(t120, t119);
__m512d t138 = _mm512_mul_pd(_mm512_set1_pd(0x1.6a09e667f3bcdp-1), t137);
__m512d t139 = _mm512_add_pd(t119, t120);
__m512d t140 = _mm512_mul_pd(_mm512_set1_pd(0x1.6a09e667f3bcdp-1), t139);
__m512d t141 = _mm512_add_pd(t103, t138);
__m512d t142 = _mm512_sub_pd(t104, t140);
__m512d t143 = _mm512_sub_pd(t103, t138);
__m512d t144 = _mm512_add_pd(t104, t140);
_mm512_store_pd(rb + (long)0*es, t121); _mm512_store_pd(ib + (long)0*es, t122);
_mm512_store_pd(rb + (long)8*es, t129); _mm512_store_pd(ib + (long)8*es, t130);
_mm512_store_pd(rb + (long)16*es, t133); _mm512_store_pd(ib + (long)16*es, t134);
_mm512_store_pd(rb + (long)24*es, t141); _mm512_store_pd(ib + (long)24*es, t142);
_mm512_store_pd(rb + (long)32*es, t123); _mm512_store_pd(ib + (long)32*es, t124);
_mm512_store_pd(rb + (long)40*es, t131); _mm512_store_pd(ib + (long)40*es, t132);
_mm512_store_pd(rb + (long)48*es, t135); _mm512_store_pd(ib + (long)48*es, t136);
_mm512_store_pd(rb + (long)56*es, t143); _mm512_store_pd(ib + (long)56*es, t144);
}
}
static const int MK0_36[5] = {255,255,255,255,255};
static const int MK1_36[5] = {255,255,255,255,0};
static const int MKJ0_36[5] = {255,255,255,255,255};
static const int MKJ1_36[5] = {255,255,255,255,0};
#define MASKS_36(crow_, pre_, pim_) do{ \
  mapvec_36((pre_)+0, (pim_)+0, (crow_)+0, 255, 255); \
  mapvec_36((pre_)+8, (pim_)+8, (crow_)+16, 255, 255); \
  mapvec_36((pre_)+16, (pim_)+16, (crow_)+32, 255, 255); \
  mapvec_36((pre_)+24, (pim_)+24, (crow_)+48, 255, 255); \
  mapvec_36((pre_)+32, (pim_)+32, (crow_)+64, 255, 0); \
}while(0)

// ---------------- family B, L=36 (LPAD=40, LJP=40, CPAD=40, PPS=3208) ----------------
static double TS_36[3200] ALIGN64;
static inline void mapvec_36(double* pre, double* pim, const double* cre, const double* cim){
    __m512d xr = _mm512_load_pd(pre), xi = _mm512_load_pd(pim);
    __m512d zr = _mm512_add_pd(xr, _mm512_load_pd(cre)), zi = _mm512_add_pd(xi, _mm512_load_pd(cim));
    map2(zr, zi, &xr, &xi);
    _mm512_store_pd(pre, xr); _mm512_store_pd(pim, xi);
}

static void S_36(double* X, const double* Csw, int do_map, int do_next){
    for(int i=0;i<36;i++){
        double* pl = X + (long)i*3208;
        const char* npl = (const char*)(X + (long)(i+1 < 36 ? i+1 : 0)*3208);
        const char* ncw = (const char*)(Csw + (long)i*3208);
        (void)npl; (void)ncw;
        for(int kc=0;kc<5;kc++){ dft36_v(pl + kc*8, pl + 1600 + kc*8, 40); }
        for(int jb=0;jb<5;jb++){
            const double* rb = pl + (long)jb*8*40;
            for(int kb=0;kb<5;kb++){
                __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                r0=_mm512_load_pd(rb+0*40+kb*8); r1=_mm512_load_pd(rb+1*40+kb*8);
                r2=_mm512_load_pd(rb+2*40+kb*8); r3=_mm512_load_pd(rb+3*40+kb*8);
                r4=_mm512_load_pd(rb+4*40+kb*8); r5=_mm512_load_pd(rb+5*40+kb*8);
                r6=_mm512_load_pd(rb+6*40+kb*8); r7=_mm512_load_pd(rb+7*40+kb*8);
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                double* tb = &TS_36[0] + (long)kb*8*40 + jb*8;
                _mm512_store_pd(tb+0*40, o0); _mm512_store_pd(tb+1*40, o1);
                _mm512_store_pd(tb+2*40, o2); _mm512_store_pd(tb+3*40, o3);
                _mm512_store_pd(tb+4*40, o4); _mm512_store_pd(tb+5*40, o5);
                _mm512_store_pd(tb+6*40, o6); _mm512_store_pd(tb+7*40, o7);
                r0=_mm512_load_pd(rb+1600+0*40+kb*8); r1=_mm512_load_pd(rb+1600+1*40+kb*8);
                r2=_mm512_load_pd(rb+1600+2*40+kb*8); r3=_mm512_load_pd(rb+1600+3*40+kb*8);
                r4=_mm512_load_pd(rb+1600+4*40+kb*8); r5=_mm512_load_pd(rb+1600+5*40+kb*8);
                r6=_mm512_load_pd(rb+1600+6*40+kb*8); r7=_mm512_load_pd(rb+1600+7*40+kb*8);
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                tb += 1600;
                _mm512_store_pd(tb+0*40, o0); _mm512_store_pd(tb+1*40, o1);
                _mm512_store_pd(tb+2*40, o2); _mm512_store_pd(tb+3*40, o3);
                _mm512_store_pd(tb+4*40, o4); _mm512_store_pd(tb+5*40, o5);
                _mm512_store_pd(tb+6*40, o6); _mm512_store_pd(tb+7*40, o7);
            }
            dft36_v(&TS_36[0] + jb*8, &TS_36[0] + 1600 + jb*8, 40);
            if(do_map){
                const double* cre = Csw + (long)i*3208 + (long)jb*288;
                long jb2 = jb + 2;
                const char* cnx = (jb2 < 5) ? (const char*)(cre + 2*288)
                                  : (const char*)(Csw + (long)(i+1 < 36 ? i+1 : 0)*3208 + (long)(jb2-5)*288);
                double* tr = &TS_36[0] + jb*8;
                double* ti = &TS_36[0] + 1600 + jb*8;
                for(int k=0; k+2<=36; k+=2){
                    _mm_prefetch(cnx + k*64, _MM_HINT_T0);
                    _mm_prefetch(cnx + 1600*8 + k*64, _MM_HINT_T0);
                    _mm_prefetch(cnx + k*64 + 64, _MM_HINT_T0);
                    _mm_prefetch(cnx + 1600*8 + k*64 + 64, _MM_HINT_T0);
                    __m512d xr0 = _mm512_load_pd(tr + (long)k*40), xi0 = _mm512_load_pd(ti + (long)k*40);
                    __m512d xr1 = _mm512_load_pd(tr + (long)(k+1)*40), xi1 = _mm512_load_pd(ti + (long)(k+1)*40);
                    __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cre + k*8));
                    __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cre + 1600 + k*8));
                    __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cre + k*8 + 8));
                    __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cre + 1600 + k*8 + 8));
                    map2(zr0, zi0, &xr0, &xi0);
                    map2(zr1, zi1, &xr1, &xi1);
                    _mm512_store_pd(tr + (long)k*40, xr0); _mm512_store_pd(ti + (long)k*40, xi0);
                    _mm512_store_pd(tr + (long)(k+1)*40, xr1); _mm512_store_pd(ti + (long)(k+1)*40, xi1);
                }
            }
            if(do_next) dft36_v(&TS_36[0] + jb*8, &TS_36[0] + 1600 + jb*8, 40);
            {
                double* rb2 = pl + (long)jb*8*40;
                for(int kb=0;kb<5;kb++){
                    __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                    const double* tb = &TS_36[0] + (long)kb*8*40 + jb*8;
                    r0=_mm512_load_pd(tb+0*40); r1=_mm512_load_pd(tb+1*40);
                    r2=_mm512_load_pd(tb+2*40); r3=_mm512_load_pd(tb+3*40);
                    r4=_mm512_load_pd(tb+4*40); r5=_mm512_load_pd(tb+5*40);
                    r6=_mm512_load_pd(tb+6*40); r7=_mm512_load_pd(tb+7*40);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(rb2+0*40+kb*8, o0); _mm512_store_pd(rb2+1*40+kb*8, o1);
                    _mm512_store_pd(rb2+2*40+kb*8, o2); _mm512_store_pd(rb2+3*40+kb*8, o3);
                    _mm512_store_pd(rb2+4*40+kb*8, o4); _mm512_store_pd(rb2+5*40+kb*8, o5);
                    _mm512_store_pd(rb2+6*40+kb*8, o6); _mm512_store_pd(rb2+7*40+kb*8, o7);
                    const double* tb2 = tb + 1600;
                    r0=_mm512_load_pd(tb2+0*40); r1=_mm512_load_pd(tb2+1*40);
                    r2=_mm512_load_pd(tb2+2*40); r3=_mm512_load_pd(tb2+3*40);
                    r4=_mm512_load_pd(tb2+4*40); r5=_mm512_load_pd(tb2+5*40);
                    r6=_mm512_load_pd(tb2+6*40); r7=_mm512_load_pd(tb2+7*40);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(rb2+1600+0*40+kb*8, o0); _mm512_store_pd(rb2+1600+1*40+kb*8, o1);
                    _mm512_store_pd(rb2+1600+2*40+kb*8, o2); _mm512_store_pd(rb2+1600+3*40+kb*8, o3);
                    _mm512_store_pd(rb2+1600+4*40+kb*8, o4); _mm512_store_pd(rb2+1600+5*40+kb*8, o5);
                    _mm512_store_pd(rb2+1600+6*40+kb*8, o6); _mm512_store_pd(rb2+1600+7*40+kb*8, o7);
                }
            }
        }
        if(do_next){
            for(int kc=0;kc<5;kc++){ dft36_v(pl + kc*8, pl + 1600 + kc*8, 40); }
        }
    }
}
static void mapcol_36(double* X, const double* C, int j, int kc, int jn, int kcn){
    double* pr = X + (long)j*40 + kc*8;
    double* pi = pr + 1600;
    const double* cp = C + ((long)j*5 + kc)*576;
    const char* npr = (const char*)(X + (long)jn*40 + kcn*8);
    for(int i=0;i+2<=36;i+=2){
        _mm_prefetch(npr + (long)i*3208*8, _MM_HINT_T0);
        _mm_prefetch(npr + (long)i*3208*8 + 1600*8, _MM_HINT_T0);
        _mm_prefetch(npr + (long)(i+1)*3208*8, _MM_HINT_T0);
        _mm_prefetch(npr + (long)(i+1)*3208*8 + 1600*8, _MM_HINT_T0);
        __m512d xr0 = _mm512_load_pd(pr + (long)i*3208);
        __m512d xi0 = _mm512_load_pd(pi + (long)i*3208);
        __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp + (long)i*16));
        __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp + (long)i*16 + 8));
        __m512d xr1 = _mm512_load_pd(pr + (long)(i+1)*3208);
        __m512d xi1 = _mm512_load_pd(pi + (long)(i+1)*3208);
        __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cp + (long)(i+1)*16));
        __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cp + (long)(i+1)*16 + 8));
        map2(zr0, zi0, &xr0, &xi0);
        map2(zr1, zi1, &xr1, &xi1);
        _mm512_store_pd(pr + (long)i*3208, xr0);
        _mm512_store_pd(pi + (long)i*3208, xi0);
        _mm512_store_pd(pr + (long)(i+1)*3208, xr1);
        _mm512_store_pd(pi + (long)(i+1)*3208, xi1);
    }
}
static void P_36(double* X, const double* C, int do_next){
    for(int j=0;j<36;j++){
        for(int kc=0;kc<5;kc++){
            double* pr = X + (long)j*40 + kc*8;
            double* pi = pr + 1600;
            int kc2 = kc+1, j2 = j;
            if(kc2 >= 5){ kc2 = 0; j2 = (j+1<36) ? j+1 : 0; }
            dft36_v(pr, pi, 3208);
            mapcol_36(X, C, j, kc, j2, kc2);
            if(do_next) dft36_v(pr, pi, 3208);
        }
    }
}
static void convin_36(const double* src, double* X){
    for(int i=0;i<36;i++){
        for(int j=0;j<36;j++){
            const double* row = src + ((long)i*36+j)*72;
            double* pre = X + (long)i*3208 + (long)j*40;
            for(int kc=0;kc<5;kc++){
                __m512d lo = _mm512_maskz_loadu_pd((__mmask8)MK0_36[kc], row + kc*16);
                __m512d hi = _mm512_maskz_loadu_pd((__mmask8)MK1_36[kc], row + kc*16 + 8);
                __m512d re, im;
                DEINT(lo, hi, re, im);
                _mm512_store_pd(pre + kc*8, re);
                _mm512_store_pd(pre + 1600 + kc*8, im);
            }
        }
    }
}
static void convout_36(const double* X, double* dst){
    for(int i=0;i<36;i++){
        for(int j=0;j<36;j++){
            double* row = dst + ((long)i*36+j)*72;
            const double* pre = X + (long)i*3208 + (long)j*40;
            for(int kc=0;kc<5;kc++){
                __m512d re = _mm512_load_pd(pre + kc*8);
                __m512d im = _mm512_load_pd(pre + 1600 + kc*8);
                __m512d lo, hi;
                INTER(re, im, lo, hi);
                _mm512_mask_storeu_pd(row + kc*16, (__mmask8)MK0_36[kc], lo);
                _mm512_mask_storeu_pd(row + kc*16 + 8, (__mmask8)MK1_36[kc], hi);
            }
        }
    }
}
// build swapped c: csw[i][k][j] = c[i][j][k] (complex), via DEINT + TR8 + INTER on 8x8 tiles
static void buildc_36(const double* c, double* cnat, double* csw, int build_sw){
    for(int i=0;i<36;i++){
        const double* cp = c + (long)i*2592;
        
        double* spre = csw + (long)i*3208;
        for(int jb=0;jb<5;jb++){
            int jn = 36 - jb*8; if(jn>8) jn=8;
            for(int kb=0;kb<5;kb++){
                __m512d RE[8], IM[8];
                for(int r=0;r<jn;r++){
                    const double* row = cp + ((long)(jb*8+r)*36 + kb*8)*2;
                    __m512d lo = _mm512_maskz_loadu_pd((__mmask8)MK0_36[kb], row);
                    __m512d hi = _mm512_maskz_loadu_pd((__mmask8)MK1_36[kb], row+8);
                    DEINT(lo, hi, RE[r], IM[r]);
                    { long e = (long)(jb*8+r)*5 + kb;
                       _mm512_store_pd(cnat + e*576 + (long)i*16, RE[r]);
                       _mm512_store_pd(cnat + e*576 + (long)i*16 + 8, IM[r]); }
                }
                if(build_sw){
                    for(int r=jn;r<8;r++){ RE[r]=_mm512_setzero_pd(); IM[r]=_mm512_setzero_pd(); }
                    __m512d o0,o1,o2,o3,o4,o5,o6,o7;
                    TR8(RE[0],RE[1],RE[2],RE[3],RE[4],RE[5],RE[6],RE[7],o0,o1,o2,o3,o4,o5,o6,o7);
                    if(kb*8+0 < 36) _mm512_store_pd(spre + (long)jb*288 + (long)(kb*8+0)*8, o0);
                    if(kb*8+1 < 36) _mm512_store_pd(spre + (long)jb*288 + (long)(kb*8+1)*8, o1);
                    if(kb*8+2 < 36) _mm512_store_pd(spre + (long)jb*288 + (long)(kb*8+2)*8, o2);
                    if(kb*8+3 < 36) _mm512_store_pd(spre + (long)jb*288 + (long)(kb*8+3)*8, o3);
                    if(kb*8+4 < 36) _mm512_store_pd(spre + (long)jb*288 + (long)(kb*8+4)*8, o4);
                    if(kb*8+5 < 36) _mm512_store_pd(spre + (long)jb*288 + (long)(kb*8+5)*8, o5);
                    if(kb*8+6 < 36) _mm512_store_pd(spre + (long)jb*288 + (long)(kb*8+6)*8, o6);
                    if(kb*8+7 < 36) _mm512_store_pd(spre + (long)jb*288 + (long)(kb*8+7)*8, o7);
                    TR8(IM[0],IM[1],IM[2],IM[3],IM[4],IM[5],IM[6],IM[7],o0,o1,o2,o3,o4,o5,o6,o7);
                    if(kb*8+0 < 36) _mm512_store_pd(spre + 1600 + (long)jb*288 + (long)(kb*8+0)*8, o0);
                    if(kb*8+1 < 36) _mm512_store_pd(spre + 1600 + (long)jb*288 + (long)(kb*8+1)*8, o1);
                    if(kb*8+2 < 36) _mm512_store_pd(spre + 1600 + (long)jb*288 + (long)(kb*8+2)*8, o2);
                    if(kb*8+3 < 36) _mm512_store_pd(spre + 1600 + (long)jb*288 + (long)(kb*8+3)*8, o3);
                    if(kb*8+4 < 36) _mm512_store_pd(spre + 1600 + (long)jb*288 + (long)(kb*8+4)*8, o4);
                    if(kb*8+5 < 36) _mm512_store_pd(spre + 1600 + (long)jb*288 + (long)(kb*8+5)*8, o5);
                    if(kb*8+6 < 36) _mm512_store_pd(spre + 1600 + (long)jb*288 + (long)(kb*8+6)*8, o6);
                    if(kb*8+7 < 36) _mm512_store_pd(spre + 1600 + (long)jb*288 + (long)(kb*8+7)*8, o7);
                }
            }
        }
    }
}
static double* XV_36 = 0;
static double* CSW_36 = 0;
static double* CNAT_36 = 0;
void run_36(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if(m < 1) m = 1;
    if(!XV_36){ XV_36 = alloc_huge_st((long)36*3208*8 + 4096); CSW_36 = alloc_huge_st((long)36*3208*8 + 4096); CNAT_36 = alloc_huge_st((long)36*3208*8 + 4096); }
    for(long v=0; v<B; v++){
        buildc_36(c + v*93312, CNAT_36, CSW_36, m >= 3);
        const double* cx = CNAT_36;
        convin_36(x0 + v*93312, XV_36);
        S_36(XV_36, CSW_36, 0, 0);
        P_36(XV_36, cx, 0);
        convout_36(XV_36, out1 + v*93312);
        if(m >= 2){
            S_36(XV_36, CSW_36, 0, 0);
            for(long t=2; t<=m; t++){
                if((t & 1) == 0) P_36(XV_36, cx, t<m);
                else             S_36(XV_36, CSW_36, 1, t<m);
            }
        }
        convout_36(XV_36, outm + v*93312);
    }
}

static const int MK0_45[6] = {255,255,255,255,255,255};
static const int MK1_45[6] = {255,255,255,255,255,3};
static const int MKJ0_45[6] = {255,255,255,255,255,255};
static const int MKJ1_45[6] = {255,255,255,255,255,3};
#define MASKS_45(crow_, pre_, pim_) do{ \
  mapvec_45((pre_)+0, (pim_)+0, (crow_)+0, 255, 255); \
  mapvec_45((pre_)+8, (pim_)+8, (crow_)+16, 255, 255); \
  mapvec_45((pre_)+16, (pim_)+16, (crow_)+32, 255, 255); \
  mapvec_45((pre_)+24, (pim_)+24, (crow_)+48, 255, 255); \
  mapvec_45((pre_)+32, (pim_)+32, (crow_)+64, 255, 255); \
  mapvec_45((pre_)+40, (pim_)+40, (crow_)+80, 255, 3); \
}while(0)

// ---------------- family B, L=45 (LPAD=56, LJP=48, CPAD=56, PPS=5384) ----------------
static double TS_45[5376] ALIGN64;
static inline void mapvec_45(double* pre, double* pim, const double* cre, const double* cim){
    __m512d xr = _mm512_load_pd(pre), xi = _mm512_load_pd(pim);
    __m512d zr = _mm512_add_pd(xr, _mm512_load_pd(cre)), zi = _mm512_add_pd(xi, _mm512_load_pd(cim));
    map2(zr, zi, &xr, &xi);
    _mm512_store_pd(pre, xr); _mm512_store_pd(pim, xi);
}

static void S_45(double* X, const double* Csw, int do_map, int do_next){
    for(int i=0;i<45;i++){
        double* pl = X + (long)i*5384;
        const char* npl = (const char*)(X + (long)(i+1 < 45 ? i+1 : 0)*5384);
        const char* ncw = (const char*)(Csw + (long)i*5384);
        (void)npl; (void)ncw;
        for(int kc=0;kc<6;kc++){ dft45_v(pl + kc*8, pl + 2688 + kc*8, 56); }
        for(int jb=0;jb<6;jb++){
            const double* rb = pl + (long)jb*8*56;
            for(int kb=0;kb<6;kb++){
                __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                r0=_mm512_load_pd(rb+0*56+kb*8); r1=_mm512_load_pd(rb+1*56+kb*8);
                r2=_mm512_load_pd(rb+2*56+kb*8); r3=_mm512_load_pd(rb+3*56+kb*8);
                r4=_mm512_load_pd(rb+4*56+kb*8); r5=_mm512_load_pd(rb+5*56+kb*8);
                r6=_mm512_load_pd(rb+6*56+kb*8); r7=_mm512_load_pd(rb+7*56+kb*8);
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                double* tb = &TS_45[0] + (long)kb*8*56 + jb*8;
                _mm512_store_pd(tb+0*56, o0); _mm512_store_pd(tb+1*56, o1);
                _mm512_store_pd(tb+2*56, o2); _mm512_store_pd(tb+3*56, o3);
                _mm512_store_pd(tb+4*56, o4); _mm512_store_pd(tb+5*56, o5);
                _mm512_store_pd(tb+6*56, o6); _mm512_store_pd(tb+7*56, o7);
                r0=_mm512_load_pd(rb+2688+0*56+kb*8); r1=_mm512_load_pd(rb+2688+1*56+kb*8);
                r2=_mm512_load_pd(rb+2688+2*56+kb*8); r3=_mm512_load_pd(rb+2688+3*56+kb*8);
                r4=_mm512_load_pd(rb+2688+4*56+kb*8); r5=_mm512_load_pd(rb+2688+5*56+kb*8);
                r6=_mm512_load_pd(rb+2688+6*56+kb*8); r7=_mm512_load_pd(rb+2688+7*56+kb*8);
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                tb += 2688;
                _mm512_store_pd(tb+0*56, o0); _mm512_store_pd(tb+1*56, o1);
                _mm512_store_pd(tb+2*56, o2); _mm512_store_pd(tb+3*56, o3);
                _mm512_store_pd(tb+4*56, o4); _mm512_store_pd(tb+5*56, o5);
                _mm512_store_pd(tb+6*56, o6); _mm512_store_pd(tb+7*56, o7);
            }
            dft45_v(&TS_45[0] + jb*8, &TS_45[0] + 2688 + jb*8, 56);
            if(do_map){
                const double* cre = Csw + (long)i*5384 + (long)jb*360;
                long jb2 = jb + 2;
                const char* cnx = (jb2 < 6) ? (const char*)(cre + 2*360)
                                  : (const char*)(Csw + (long)(i+1 < 45 ? i+1 : 0)*5384 + (long)(jb2-6)*360);
                double* tr = &TS_45[0] + jb*8;
                double* ti = &TS_45[0] + 2688 + jb*8;
                for(int k=0; k+2<=45; k+=2){
                    _mm_prefetch(cnx + k*64, _MM_HINT_T0);
                    _mm_prefetch(cnx + 2688*8 + k*64, _MM_HINT_T0);
                    _mm_prefetch(cnx + k*64 + 64, _MM_HINT_T0);
                    _mm_prefetch(cnx + 2688*8 + k*64 + 64, _MM_HINT_T0);
                    __m512d xr0 = _mm512_load_pd(tr + (long)k*56), xi0 = _mm512_load_pd(ti + (long)k*56);
                    __m512d xr1 = _mm512_load_pd(tr + (long)(k+1)*56), xi1 = _mm512_load_pd(ti + (long)(k+1)*56);
                    __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cre + k*8));
                    __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cre + 2688 + k*8));
                    __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cre + k*8 + 8));
                    __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cre + 2688 + k*8 + 8));
                    map2(zr0, zi0, &xr0, &xi0);
                    map2(zr1, zi1, &xr1, &xi1);
                    _mm512_store_pd(tr + (long)k*56, xr0); _mm512_store_pd(ti + (long)k*56, xi0);
                    _mm512_store_pd(tr + (long)(k+1)*56, xr1); _mm512_store_pd(ti + (long)(k+1)*56, xi1);
                }
                { int k = 44;
                    __m512d xr0 = _mm512_load_pd(tr + (long)k*56), xi0 = _mm512_load_pd(ti + (long)k*56);
                    __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cre + k*8));
                    __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cre + 2688 + k*8));
                    map2(zr0, zi0, &xr0, &xi0);
                    _mm512_store_pd(tr + (long)k*56, xr0); _mm512_store_pd(ti + (long)k*56, xi0);
                }
            }
            if(do_next) dft45_v(&TS_45[0] + jb*8, &TS_45[0] + 2688 + jb*8, 56);
            {
                double* rb2 = pl + (long)jb*8*56;
                for(int kb=0;kb<6;kb++){
                    __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                    const double* tb = &TS_45[0] + (long)kb*8*56 + jb*8;
                    r0=_mm512_load_pd(tb+0*56); r1=_mm512_load_pd(tb+1*56);
                    r2=_mm512_load_pd(tb+2*56); r3=_mm512_load_pd(tb+3*56);
                    r4=_mm512_load_pd(tb+4*56); r5=_mm512_load_pd(tb+5*56);
                    r6=_mm512_load_pd(tb+6*56); r7=_mm512_load_pd(tb+7*56);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(rb2+0*56+kb*8, o0); _mm512_store_pd(rb2+1*56+kb*8, o1);
                    _mm512_store_pd(rb2+2*56+kb*8, o2); _mm512_store_pd(rb2+3*56+kb*8, o3);
                    _mm512_store_pd(rb2+4*56+kb*8, o4); _mm512_store_pd(rb2+5*56+kb*8, o5);
                    _mm512_store_pd(rb2+6*56+kb*8, o6); _mm512_store_pd(rb2+7*56+kb*8, o7);
                    const double* tb2 = tb + 2688;
                    r0=_mm512_load_pd(tb2+0*56); r1=_mm512_load_pd(tb2+1*56);
                    r2=_mm512_load_pd(tb2+2*56); r3=_mm512_load_pd(tb2+3*56);
                    r4=_mm512_load_pd(tb2+4*56); r5=_mm512_load_pd(tb2+5*56);
                    r6=_mm512_load_pd(tb2+6*56); r7=_mm512_load_pd(tb2+7*56);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(rb2+2688+0*56+kb*8, o0); _mm512_store_pd(rb2+2688+1*56+kb*8, o1);
                    _mm512_store_pd(rb2+2688+2*56+kb*8, o2); _mm512_store_pd(rb2+2688+3*56+kb*8, o3);
                    _mm512_store_pd(rb2+2688+4*56+kb*8, o4); _mm512_store_pd(rb2+2688+5*56+kb*8, o5);
                    _mm512_store_pd(rb2+2688+6*56+kb*8, o6); _mm512_store_pd(rb2+2688+7*56+kb*8, o7);
                }
            }
        }
        if(do_next){
            for(int kc=0;kc<6;kc++){ dft45_v(pl + kc*8, pl + 2688 + kc*8, 56); }
        }
    }
}
static void mapcol_45(double* X, const double* C, int j, int kc, int jn, int kcn){
    double* pr = X + (long)j*56 + kc*8;
    double* pi = pr + 2688;
    const double* cp = C + ((long)j*6 + kc)*720;
    const char* npr = (const char*)(X + (long)jn*56 + kcn*8);
    for(int i=0;i+2<=45;i+=2){
        _mm_prefetch(npr + (long)i*5384*8, _MM_HINT_T0);
        _mm_prefetch(npr + (long)i*5384*8 + 2688*8, _MM_HINT_T0);
        _mm_prefetch(npr + (long)(i+1)*5384*8, _MM_HINT_T0);
        _mm_prefetch(npr + (long)(i+1)*5384*8 + 2688*8, _MM_HINT_T0);
        __m512d xr0 = _mm512_load_pd(pr + (long)i*5384);
        __m512d xi0 = _mm512_load_pd(pi + (long)i*5384);
        __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp + (long)i*16));
        __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp + (long)i*16 + 8));
        __m512d xr1 = _mm512_load_pd(pr + (long)(i+1)*5384);
        __m512d xi1 = _mm512_load_pd(pi + (long)(i+1)*5384);
        __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cp + (long)(i+1)*16));
        __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cp + (long)(i+1)*16 + 8));
        map2(zr0, zi0, &xr0, &xi0);
        map2(zr1, zi1, &xr1, &xi1);
        _mm512_store_pd(pr + (long)i*5384, xr0);
        _mm512_store_pd(pi + (long)i*5384, xi0);
        _mm512_store_pd(pr + (long)(i+1)*5384, xr1);
        _mm512_store_pd(pi + (long)(i+1)*5384, xi1);
    }
    { int i = 44;
        __m512d xr0 = _mm512_load_pd(pr + (long)i*5384);
        __m512d xi0 = _mm512_load_pd(pi + (long)i*5384);
        __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp + (long)i*16));
        __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp + (long)i*16 + 8));
        map2(zr0, zi0, &xr0, &xi0);
        _mm512_store_pd(pr + (long)i*5384, xr0);
        _mm512_store_pd(pi + (long)i*5384, xi0);
    }
}
static void P_45(double* X, const double* C, int do_next){
    for(int j=0;j<45;j++){
        for(int kc=0;kc<6;kc++){
            double* pr = X + (long)j*56 + kc*8;
            double* pi = pr + 2688;
            int kc2 = kc+1, j2 = j;
            if(kc2 >= 6){ kc2 = 0; j2 = (j+1<45) ? j+1 : 0; }
            dft45_v(pr, pi, 5384);
            mapcol_45(X, C, j, kc, j2, kc2);
            if(do_next) dft45_v(pr, pi, 5384);
        }
    }
}
static void convin_45(const double* src, double* X){
    for(int i=0;i<45;i++){
        for(int j=0;j<45;j++){
            const double* row = src + ((long)i*45+j)*90;
            double* pre = X + (long)i*5384 + (long)j*56;
            for(int kc=0;kc<6;kc++){
                __m512d lo = _mm512_maskz_loadu_pd((__mmask8)MK0_45[kc], row + kc*16);
                __m512d hi = _mm512_maskz_loadu_pd((__mmask8)MK1_45[kc], row + kc*16 + 8);
                __m512d re, im;
                DEINT(lo, hi, re, im);
                _mm512_store_pd(pre + kc*8, re);
                _mm512_store_pd(pre + 2688 + kc*8, im);
            }
        }
    }
}
static void convout_45(const double* X, double* dst){
    for(int i=0;i<45;i++){
        for(int j=0;j<45;j++){
            double* row = dst + ((long)i*45+j)*90;
            const double* pre = X + (long)i*5384 + (long)j*56;
            for(int kc=0;kc<6;kc++){
                __m512d re = _mm512_load_pd(pre + kc*8);
                __m512d im = _mm512_load_pd(pre + 2688 + kc*8);
                __m512d lo, hi;
                INTER(re, im, lo, hi);
                _mm512_mask_storeu_pd(row + kc*16, (__mmask8)MK0_45[kc], lo);
                _mm512_mask_storeu_pd(row + kc*16 + 8, (__mmask8)MK1_45[kc], hi);
            }
        }
    }
}
// build swapped c: csw[i][k][j] = c[i][j][k] (complex), via DEINT + TR8 + INTER on 8x8 tiles
static void buildc_45(const double* c, double* cnat, double* csw, int build_sw){
    for(int i=0;i<45;i++){
        const double* cp = c + (long)i*4050;
        
        double* spre = csw + (long)i*5384;
        for(int jb=0;jb<6;jb++){
            int jn = 45 - jb*8; if(jn>8) jn=8;
            for(int kb=0;kb<6;kb++){
                __m512d RE[8], IM[8];
                for(int r=0;r<jn;r++){
                    const double* row = cp + ((long)(jb*8+r)*45 + kb*8)*2;
                    __m512d lo = _mm512_maskz_loadu_pd((__mmask8)MK0_45[kb], row);
                    __m512d hi = _mm512_maskz_loadu_pd((__mmask8)MK1_45[kb], row+8);
                    DEINT(lo, hi, RE[r], IM[r]);
                    { long e = (long)(jb*8+r)*6 + kb;
                       _mm512_store_pd(cnat + e*720 + (long)i*16, RE[r]);
                       _mm512_store_pd(cnat + e*720 + (long)i*16 + 8, IM[r]); }
                }
                if(build_sw){
                    for(int r=jn;r<8;r++){ RE[r]=_mm512_setzero_pd(); IM[r]=_mm512_setzero_pd(); }
                    __m512d o0,o1,o2,o3,o4,o5,o6,o7;
                    TR8(RE[0],RE[1],RE[2],RE[3],RE[4],RE[5],RE[6],RE[7],o0,o1,o2,o3,o4,o5,o6,o7);
                    if(kb*8+0 < 45) _mm512_store_pd(spre + (long)jb*360 + (long)(kb*8+0)*8, o0);
                    if(kb*8+1 < 45) _mm512_store_pd(spre + (long)jb*360 + (long)(kb*8+1)*8, o1);
                    if(kb*8+2 < 45) _mm512_store_pd(spre + (long)jb*360 + (long)(kb*8+2)*8, o2);
                    if(kb*8+3 < 45) _mm512_store_pd(spre + (long)jb*360 + (long)(kb*8+3)*8, o3);
                    if(kb*8+4 < 45) _mm512_store_pd(spre + (long)jb*360 + (long)(kb*8+4)*8, o4);
                    if(kb*8+5 < 45) _mm512_store_pd(spre + (long)jb*360 + (long)(kb*8+5)*8, o5);
                    if(kb*8+6 < 45) _mm512_store_pd(spre + (long)jb*360 + (long)(kb*8+6)*8, o6);
                    if(kb*8+7 < 45) _mm512_store_pd(spre + (long)jb*360 + (long)(kb*8+7)*8, o7);
                    TR8(IM[0],IM[1],IM[2],IM[3],IM[4],IM[5],IM[6],IM[7],o0,o1,o2,o3,o4,o5,o6,o7);
                    if(kb*8+0 < 45) _mm512_store_pd(spre + 2688 + (long)jb*360 + (long)(kb*8+0)*8, o0);
                    if(kb*8+1 < 45) _mm512_store_pd(spre + 2688 + (long)jb*360 + (long)(kb*8+1)*8, o1);
                    if(kb*8+2 < 45) _mm512_store_pd(spre + 2688 + (long)jb*360 + (long)(kb*8+2)*8, o2);
                    if(kb*8+3 < 45) _mm512_store_pd(spre + 2688 + (long)jb*360 + (long)(kb*8+3)*8, o3);
                    if(kb*8+4 < 45) _mm512_store_pd(spre + 2688 + (long)jb*360 + (long)(kb*8+4)*8, o4);
                    if(kb*8+5 < 45) _mm512_store_pd(spre + 2688 + (long)jb*360 + (long)(kb*8+5)*8, o5);
                    if(kb*8+6 < 45) _mm512_store_pd(spre + 2688 + (long)jb*360 + (long)(kb*8+6)*8, o6);
                    if(kb*8+7 < 45) _mm512_store_pd(spre + 2688 + (long)jb*360 + (long)(kb*8+7)*8, o7);
                }
            }
        }
    }
}
static double* XV_45 = 0;
static double* CSW_45 = 0;
static double* CNAT_45 = 0;
void run_45(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if(m < 1) m = 1;
    if(!XV_45){ XV_45 = alloc_huge_st((long)45*5384*8 + 4096); CSW_45 = alloc_huge_st((long)45*5384*8 + 4096); CNAT_45 = alloc_huge_st((long)45*5384*8 + 4096); }
    for(long v=0; v<B; v++){
        buildc_45(c + v*182250, CNAT_45, CSW_45, m >= 3);
        const double* cx = CNAT_45;
        convin_45(x0 + v*182250, XV_45);
        S_45(XV_45, CSW_45, 0, 0);
        P_45(XV_45, cx, 0);
        convout_45(XV_45, out1 + v*182250);
        if(m >= 2){
            S_45(XV_45, CSW_45, 0, 0);
            for(long t=2; t<=m; t++){
                if((t & 1) == 0) P_45(XV_45, cx, t<m);
                else             S_45(XV_45, CSW_45, 1, t<m);
            }
        }
        convout_45(XV_45, outm + v*182250);
    }
}

static const int MK0_64[8] = {255,255,255,255,255,255,255,255};
static const int MK1_64[8] = {255,255,255,255,255,255,255,255};
static const int MKJ0_64[8] = {255,255,255,255,255,255,255,255};
static const int MKJ1_64[8] = {255,255,255,255,255,255,255,255};
#define MASKS_64(crow_, pre_, pim_) do{ \
  mapvec_64((pre_)+0, (pim_)+0, (crow_)+0, 255, 255); \
  mapvec_64((pre_)+8, (pim_)+8, (crow_)+16, 255, 255); \
  mapvec_64((pre_)+16, (pim_)+16, (crow_)+32, 255, 255); \
  mapvec_64((pre_)+24, (pim_)+24, (crow_)+48, 255, 255); \
  mapvec_64((pre_)+32, (pim_)+32, (crow_)+64, 255, 255); \
  mapvec_64((pre_)+40, (pim_)+40, (crow_)+80, 255, 255); \
  mapvec_64((pre_)+48, (pim_)+48, (crow_)+96, 255, 255); \
  mapvec_64((pre_)+56, (pim_)+56, (crow_)+112, 255, 255); \
}while(0)

// ---------------- family B, L=64 (LPAD=72, LJP=64, CPAD=72, PPS=9224) ----------------
static double TS_64[9216] ALIGN64;
static inline void mapvec_64(double* pre, double* pim, const double* cre, const double* cim){
    __m512d xr = _mm512_load_pd(pre), xi = _mm512_load_pd(pim);
    __m512d zr = _mm512_add_pd(xr, _mm512_load_pd(cre)), zi = _mm512_add_pd(xi, _mm512_load_pd(cim));
    map2(zr, zi, &xr, &xi);
    _mm512_store_pd(pre, xr); _mm512_store_pd(pim, xi);
}

static void S_64(double* X, const double* Csw, int do_map, int do_next){
    for(int i=0;i<64;i++){
        double* pl = X + (long)i*9224;
        const char* npl = (const char*)(X + (long)(i+1 < 64 ? i+1 : 0)*9224);
        const char* ncw = (const char*)(Csw + (long)i*9224);
        (void)npl; (void)ncw;
        for(int kc=0;kc<8;kc++){ dft64_v(pl + kc*8, pl + 4608 + kc*8, 72); }
        for(int jb=0;jb<8;jb++){
            const double* rb = pl + (long)jb*8*72;
            for(int kb=0;kb<8;kb++){
                __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                r0=_mm512_load_pd(rb+0*72+kb*8); r1=_mm512_load_pd(rb+1*72+kb*8);
                r2=_mm512_load_pd(rb+2*72+kb*8); r3=_mm512_load_pd(rb+3*72+kb*8);
                r4=_mm512_load_pd(rb+4*72+kb*8); r5=_mm512_load_pd(rb+5*72+kb*8);
                r6=_mm512_load_pd(rb+6*72+kb*8); r7=_mm512_load_pd(rb+7*72+kb*8);
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                double* tb = &TS_64[0] + (long)kb*8*72 + jb*8;
                _mm512_store_pd(tb+0*72, o0); _mm512_store_pd(tb+1*72, o1);
                _mm512_store_pd(tb+2*72, o2); _mm512_store_pd(tb+3*72, o3);
                _mm512_store_pd(tb+4*72, o4); _mm512_store_pd(tb+5*72, o5);
                _mm512_store_pd(tb+6*72, o6); _mm512_store_pd(tb+7*72, o7);
                r0=_mm512_load_pd(rb+4608+0*72+kb*8); r1=_mm512_load_pd(rb+4608+1*72+kb*8);
                r2=_mm512_load_pd(rb+4608+2*72+kb*8); r3=_mm512_load_pd(rb+4608+3*72+kb*8);
                r4=_mm512_load_pd(rb+4608+4*72+kb*8); r5=_mm512_load_pd(rb+4608+5*72+kb*8);
                r6=_mm512_load_pd(rb+4608+6*72+kb*8); r7=_mm512_load_pd(rb+4608+7*72+kb*8);
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                tb += 4608;
                _mm512_store_pd(tb+0*72, o0); _mm512_store_pd(tb+1*72, o1);
                _mm512_store_pd(tb+2*72, o2); _mm512_store_pd(tb+3*72, o3);
                _mm512_store_pd(tb+4*72, o4); _mm512_store_pd(tb+5*72, o5);
                _mm512_store_pd(tb+6*72, o6); _mm512_store_pd(tb+7*72, o7);
            }
            dft64_v(&TS_64[0] + jb*8, &TS_64[0] + 4608 + jb*8, 72);
            { const char* q = npl + (long)jb*9288;
               for(long b=0;b<9288;b+=64) _mm_prefetch(q+b, _MM_HINT_T1); }
            if(do_map){
                const double* cre = Csw + (long)i*9224 + (long)jb*512;
                long jb2 = jb + 2;
                const char* cnx = (jb2 < 8) ? (const char*)(cre + 2*512)
                                  : (const char*)(Csw + (long)(i+1 < 64 ? i+1 : 0)*9224 + (long)(jb2-8)*512);
                double* tr = &TS_64[0] + jb*8;
                double* ti = &TS_64[0] + 4608 + jb*8;
                for(int k=0; k+2<=64; k+=2){
                    _mm_prefetch(cnx + k*64, _MM_HINT_T0);
                    _mm_prefetch(cnx + 4608*8 + k*64, _MM_HINT_T0);
                    _mm_prefetch(cnx + k*64 + 64, _MM_HINT_T0);
                    _mm_prefetch(cnx + 4608*8 + k*64 + 64, _MM_HINT_T0);
                    __m512d xr0 = _mm512_load_pd(tr + (long)k*72), xi0 = _mm512_load_pd(ti + (long)k*72);
                    __m512d xr1 = _mm512_load_pd(tr + (long)(k+1)*72), xi1 = _mm512_load_pd(ti + (long)(k+1)*72);
                    __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cre + k*8));
                    __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cre + 4608 + k*8));
                    __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cre + k*8 + 8));
                    __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cre + 4608 + k*8 + 8));
                    map2(zr0, zi0, &xr0, &xi0);
                    map2(zr1, zi1, &xr1, &xi1);
                    _mm512_store_pd(tr + (long)k*72, xr0); _mm512_store_pd(ti + (long)k*72, xi0);
                    _mm512_store_pd(tr + (long)(k+1)*72, xr1); _mm512_store_pd(ti + (long)(k+1)*72, xi1);
                }
            }
            if(do_next) dft64_v(&TS_64[0] + jb*8, &TS_64[0] + 4608 + jb*8, 72);
            {
                double* rb2 = pl + (long)jb*8*72;
                for(int kb=0;kb<8;kb++){
                    __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                    const double* tb = &TS_64[0] + (long)kb*8*72 + jb*8;
                    r0=_mm512_load_pd(tb+0*72); r1=_mm512_load_pd(tb+1*72);
                    r2=_mm512_load_pd(tb+2*72); r3=_mm512_load_pd(tb+3*72);
                    r4=_mm512_load_pd(tb+4*72); r5=_mm512_load_pd(tb+5*72);
                    r6=_mm512_load_pd(tb+6*72); r7=_mm512_load_pd(tb+7*72);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(rb2+0*72+kb*8, o0); _mm512_store_pd(rb2+1*72+kb*8, o1);
                    _mm512_store_pd(rb2+2*72+kb*8, o2); _mm512_store_pd(rb2+3*72+kb*8, o3);
                    _mm512_store_pd(rb2+4*72+kb*8, o4); _mm512_store_pd(rb2+5*72+kb*8, o5);
                    _mm512_store_pd(rb2+6*72+kb*8, o6); _mm512_store_pd(rb2+7*72+kb*8, o7);
                    const double* tb2 = tb + 4608;
                    r0=_mm512_load_pd(tb2+0*72); r1=_mm512_load_pd(tb2+1*72);
                    r2=_mm512_load_pd(tb2+2*72); r3=_mm512_load_pd(tb2+3*72);
                    r4=_mm512_load_pd(tb2+4*72); r5=_mm512_load_pd(tb2+5*72);
                    r6=_mm512_load_pd(tb2+6*72); r7=_mm512_load_pd(tb2+7*72);
                    TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                    _mm512_store_pd(rb2+4608+0*72+kb*8, o0); _mm512_store_pd(rb2+4608+1*72+kb*8, o1);
                    _mm512_store_pd(rb2+4608+2*72+kb*8, o2); _mm512_store_pd(rb2+4608+3*72+kb*8, o3);
                    _mm512_store_pd(rb2+4608+4*72+kb*8, o4); _mm512_store_pd(rb2+4608+5*72+kb*8, o5);
                    _mm512_store_pd(rb2+4608+6*72+kb*8, o6); _mm512_store_pd(rb2+4608+7*72+kb*8, o7);
                }
            }
        }
        if(do_next){
            for(int kc=0;kc<8;kc++){ dft64_v(pl + kc*8, pl + 4608 + kc*8, 72); }
        }
    }
}
static void mapcol_64(double* X, const double* C, int j, int kc, int jn, int kcn){
    double* pr = X + (long)j*72 + kc*8;
    double* pi = pr + 4608;
    const double* cp = C + ((long)j*8 + kc)*1024;
    const char* npr = (const char*)(X + (long)jn*72 + kcn*8);
    for(int i=0;i+2<=64;i+=2){
        _mm_prefetch(npr + (long)i*9224*8, _MM_HINT_T0);
        _mm_prefetch(npr + (long)i*9224*8 + 4608*8, _MM_HINT_T0);
        _mm_prefetch(npr + (long)(i+1)*9224*8, _MM_HINT_T0);
        _mm_prefetch(npr + (long)(i+1)*9224*8 + 4608*8, _MM_HINT_T0);
        __m512d xr0 = _mm512_load_pd(pr + (long)i*9224);
        __m512d xi0 = _mm512_load_pd(pi + (long)i*9224);
        __m512d zr0 = _mm512_add_pd(xr0, _mm512_load_pd(cp + (long)i*16));
        __m512d zi0 = _mm512_add_pd(xi0, _mm512_load_pd(cp + (long)i*16 + 8));
        __m512d xr1 = _mm512_load_pd(pr + (long)(i+1)*9224);
        __m512d xi1 = _mm512_load_pd(pi + (long)(i+1)*9224);
        __m512d zr1 = _mm512_add_pd(xr1, _mm512_load_pd(cp + (long)(i+1)*16));
        __m512d zi1 = _mm512_add_pd(xi1, _mm512_load_pd(cp + (long)(i+1)*16 + 8));
        map2(zr0, zi0, &xr0, &xi0);
        map2(zr1, zi1, &xr1, &xi1);
        _mm512_store_pd(pr + (long)i*9224, xr0);
        _mm512_store_pd(pi + (long)i*9224, xi0);
        _mm512_store_pd(pr + (long)(i+1)*9224, xr1);
        _mm512_store_pd(pi + (long)(i+1)*9224, xi1);
    }
}
static void P_64(double* X, const double* C, int do_next){
    for(int j=0;j<64;j++){
        for(int kc=0;kc<8;kc++){
            double* pr = X + (long)j*72 + kc*8;
            double* pi = pr + 4608;
            int kc2 = kc+1, j2 = j;
            if(kc2 >= 8){ kc2 = 0; j2 = (j+1<64) ? j+1 : 0; }
            dft64_v(pr, pi, 9224);
            mapcol_64(X, C, j, kc, j2, kc2);
            if(do_next) dft64_v(pr, pi, 9224);
        }
    }
}
static void convin_64(const double* src, double* X){
    for(int i=0;i<64;i++){
        for(int j=0;j<64;j++){
            const double* row = src + ((long)i*64+j)*128;
            double* pre = X + (long)i*9224 + (long)j*72;
            for(int kc=0;kc<8;kc++){
                __m512d lo = _mm512_maskz_loadu_pd((__mmask8)MK0_64[kc], row + kc*16);
                __m512d hi = _mm512_maskz_loadu_pd((__mmask8)MK1_64[kc], row + kc*16 + 8);
                __m512d re, im;
                DEINT(lo, hi, re, im);
                _mm512_store_pd(pre + kc*8, re);
                _mm512_store_pd(pre + 4608 + kc*8, im);
            }
        }
    }
}
static void convout_64(const double* X, double* dst){
    for(int i=0;i<64;i++){
        for(int j=0;j<64;j++){
            double* row = dst + ((long)i*64+j)*128;
            const double* pre = X + (long)i*9224 + (long)j*72;
            for(int kc=0;kc<8;kc++){
                __m512d re = _mm512_load_pd(pre + kc*8);
                __m512d im = _mm512_load_pd(pre + 4608 + kc*8);
                __m512d lo, hi;
                INTER(re, im, lo, hi);
                _mm512_mask_storeu_pd(row + kc*16, (__mmask8)MK0_64[kc], lo);
                _mm512_mask_storeu_pd(row + kc*16 + 8, (__mmask8)MK1_64[kc], hi);
            }
        }
    }
}
// build swapped c: csw[i][k][j] = c[i][j][k] (complex), via DEINT + TR8 + INTER on 8x8 tiles
static void buildc_64(const double* c, double* cnat, double* csw, int build_sw){
    for(int i=0;i<64;i++){
        const double* cp = c + (long)i*8192;
        
        double* spre = csw + (long)i*9224;
        for(int jb=0;jb<8;jb++){
            int jn = 64 - jb*8; if(jn>8) jn=8;
            for(int kb=0;kb<8;kb++){
                __m512d RE[8], IM[8];
                for(int r=0;r<jn;r++){
                    const double* row = cp + ((long)(jb*8+r)*64 + kb*8)*2;
                    __m512d lo = _mm512_maskz_loadu_pd((__mmask8)MK0_64[kb], row);
                    __m512d hi = _mm512_maskz_loadu_pd((__mmask8)MK1_64[kb], row+8);
                    DEINT(lo, hi, RE[r], IM[r]);
                    { long e = (long)(jb*8+r)*8 + kb;
                       _mm512_store_pd(cnat + e*1024 + (long)i*16, RE[r]);
                       _mm512_store_pd(cnat + e*1024 + (long)i*16 + 8, IM[r]); }
                }
                if(build_sw){
                    for(int r=jn;r<8;r++){ RE[r]=_mm512_setzero_pd(); IM[r]=_mm512_setzero_pd(); }
                    __m512d o0,o1,o2,o3,o4,o5,o6,o7;
                    TR8(RE[0],RE[1],RE[2],RE[3],RE[4],RE[5],RE[6],RE[7],o0,o1,o2,o3,o4,o5,o6,o7);
                    if(kb*8+0 < 64) _mm512_store_pd(spre + (long)jb*512 + (long)(kb*8+0)*8, o0);
                    if(kb*8+1 < 64) _mm512_store_pd(spre + (long)jb*512 + (long)(kb*8+1)*8, o1);
                    if(kb*8+2 < 64) _mm512_store_pd(spre + (long)jb*512 + (long)(kb*8+2)*8, o2);
                    if(kb*8+3 < 64) _mm512_store_pd(spre + (long)jb*512 + (long)(kb*8+3)*8, o3);
                    if(kb*8+4 < 64) _mm512_store_pd(spre + (long)jb*512 + (long)(kb*8+4)*8, o4);
                    if(kb*8+5 < 64) _mm512_store_pd(spre + (long)jb*512 + (long)(kb*8+5)*8, o5);
                    if(kb*8+6 < 64) _mm512_store_pd(spre + (long)jb*512 + (long)(kb*8+6)*8, o6);
                    if(kb*8+7 < 64) _mm512_store_pd(spre + (long)jb*512 + (long)(kb*8+7)*8, o7);
                    TR8(IM[0],IM[1],IM[2],IM[3],IM[4],IM[5],IM[6],IM[7],o0,o1,o2,o3,o4,o5,o6,o7);
                    if(kb*8+0 < 64) _mm512_store_pd(spre + 4608 + (long)jb*512 + (long)(kb*8+0)*8, o0);
                    if(kb*8+1 < 64) _mm512_store_pd(spre + 4608 + (long)jb*512 + (long)(kb*8+1)*8, o1);
                    if(kb*8+2 < 64) _mm512_store_pd(spre + 4608 + (long)jb*512 + (long)(kb*8+2)*8, o2);
                    if(kb*8+3 < 64) _mm512_store_pd(spre + 4608 + (long)jb*512 + (long)(kb*8+3)*8, o3);
                    if(kb*8+4 < 64) _mm512_store_pd(spre + 4608 + (long)jb*512 + (long)(kb*8+4)*8, o4);
                    if(kb*8+5 < 64) _mm512_store_pd(spre + 4608 + (long)jb*512 + (long)(kb*8+5)*8, o5);
                    if(kb*8+6 < 64) _mm512_store_pd(spre + 4608 + (long)jb*512 + (long)(kb*8+6)*8, o6);
                    if(kb*8+7 < 64) _mm512_store_pd(spre + 4608 + (long)jb*512 + (long)(kb*8+7)*8, o7);
                }
            }
        }
    }
}
static double* XV_64 = 0;
static double* CSW_64 = 0;
static double* CNAT_64 = 0;
void run_64(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if(m < 1) m = 1;
    if(!XV_64){ XV_64 = alloc_huge_st((long)64*9224*8 + 4096); CSW_64 = alloc_huge_st((long)64*9224*8 + 4096); CNAT_64 = alloc_huge_st((long)64*9224*8 + 4096); }
    for(long v=0; v<B; v++){
        buildc_64(c + v*524288, CNAT_64, CSW_64, m >= 3);
        const double* cx = CNAT_64;
        convin_64(x0 + v*524288, XV_64);
        S_64(XV_64, CSW_64, 0, 0);
        P_64(XV_64, cx, 0);
        convout_64(XV_64, out1 + v*524288);
        if(m >= 2){
            S_64(XV_64, CSW_64, 0, 0);
            for(long t=2; t<=m; t++){
                if((t & 1) == 0) P_64(XV_64, cx, t<m);
                else             S_64(XV_64, CSW_64, 1, t<m);
            }
        }
        convout_64(XV_64, outm + v*524288);
    }
}
