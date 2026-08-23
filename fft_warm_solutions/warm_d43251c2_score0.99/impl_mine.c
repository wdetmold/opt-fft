#define PFHINT _MM_HINT_T0
#define USEASM_FLAG 0
#define PFCOMP 0
#define MAPZB_FLAG 0
#define XFIRST_FLAG 0

#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define ALIGN64 __attribute__((aligned(64)))
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
    long st = ((stagger_ctr++ % 29) + 1) * 4672;
    double* p = alloc_huge(bytes + st + 8192);
    return (double*)((char*)p + st);
}

static const __m512d VONE_ = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
#define VONE VONE_
static const __m512d VHALF_ = {0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5};
#define VHALF VHALF_

/* map z/(1+|z|): zr,zi -> xr,xi. ~1ulp via rsqrt14+NR+Heron and rcp14+NR2 */
static inline void maphw(__m512d zr, __m512d zi, __m512d* oxr, __m512d* oxi){
    const __m512d TINY = _mm512_set1_pd(1e-30);
    __m512d m  = _mm512_fmadd_pd(zr, zr, _mm512_fmadd_pd(zi, zi, TINY));
    __m512d s  = _mm512_sqrt_pd(m);
    __m512d u  = _mm512_add_pd(VONE, s);
    __m512d w  = _mm512_div_pd(VONE, u);
    *oxr = _mm512_mul_pd(zr, w);
    *oxi = _mm512_mul_pd(zi, w);
}
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

/* 8x8 transpose */
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
} while(0)

/* in-place +c+map over n contiguous element-vecs, hw/NR alternating */
static void __attribute__((noinline)) mapslab(double* restrict X, const double* restrict C, long n){
    long e = 0;
    for(; e+2 <= n; e += 2){
        __m512d zr0 = _mm512_add_pd(_mm512_load_pd(X+e*16),    _mm512_load_pd(C+e*16));
        __m512d zi0 = _mm512_add_pd(_mm512_load_pd(X+e*16+8),  _mm512_load_pd(C+e*16+8));
        __m512d zr1 = _mm512_add_pd(_mm512_load_pd(X+e*16+16), _mm512_load_pd(C+e*16+16));
        __m512d zi1 = _mm512_add_pd(_mm512_load_pd(X+e*16+24), _mm512_load_pd(C+e*16+24));
        maphw(zr0, zi0, &zr0, &zi0);
        map2(zr1, zi1, &zr1, &zi1);
        _mm512_store_pd(X+e*16,    zr0); _mm512_store_pd(X+e*16+8,  zi0);
        _mm512_store_pd(X+e*16+16, zr1); _mm512_store_pd(X+e*16+24, zi1);
    }
    for(; e < n; e++){
        __m512d zr = _mm512_add_pd(_mm512_load_pd(X+e*16),   _mm512_load_pd(C+e*16));
        __m512d zi = _mm512_add_pd(_mm512_load_pd(X+e*16+8), _mm512_load_pd(C+e*16+8));
        map2(zr, zi, &zr, &zi);
        _mm512_store_pd(X+e*16, zr); _mm512_store_pd(X+e*16+8, zi);
    }
}

static void dft6(double* restrict X, long es){
    const long s = es*16;
    {
    const __m512d k0 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k1 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    __m512d t1 = _mm512_load_pd(X+0*s);
    __m512d t2 = _mm512_load_pd(X+0*s+8);
    __m512d t3 = _mm512_load_pd(X+2*s);
    __m512d t4 = _mm512_load_pd(X+2*s+8);
    __m512d t5 = _mm512_load_pd(X+4*s);
    __m512d t6 = _mm512_load_pd(X+4*s+8);
    __m512d t7 = _mm512_add_pd(t3, t5);
    __m512d t8 = _mm512_add_pd(t4, t6);
    __m512d t9 = _mm512_sub_pd(t3, t5);
    __m512d t10 = _mm512_sub_pd(t4, t6);
    __m512d t11 = _mm512_add_pd(t1, t7);
    __m512d t12 = _mm512_add_pd(t2, t8);
    __m512d t13 = _mm512_fmadd_pd(k0, t7, t1);
    __m512d t14 = _mm512_fmadd_pd(k0, t8, t2);
    __m512d t15 = _mm512_mul_pd(k1, t9);
    __m512d t16 = _mm512_mul_pd(k1, t10);
    __m512d t17 = _mm512_add_pd(t13, t16);
    __m512d t18 = _mm512_sub_pd(t14, t15);
    __m512d t19 = _mm512_sub_pd(t13, t16);
    __m512d t20 = _mm512_add_pd(t14, t15);
    __m512d t21 = _mm512_load_pd(X+3*s);
    __m512d t22 = _mm512_load_pd(X+3*s+8);
    __m512d t23 = _mm512_load_pd(X+5*s);
    __m512d t24 = _mm512_load_pd(X+5*s+8);
    __m512d t25 = _mm512_load_pd(X+1*s);
    __m512d t26 = _mm512_load_pd(X+1*s+8);
    __m512d t27 = _mm512_add_pd(t23, t25);
    __m512d t28 = _mm512_add_pd(t24, t26);
    __m512d t29 = _mm512_sub_pd(t23, t25);
    __m512d t30 = _mm512_sub_pd(t24, t26);
    __m512d t31 = _mm512_add_pd(t21, t27);
    __m512d t32 = _mm512_add_pd(t22, t28);
    __m512d t33 = _mm512_fmadd_pd(k0, t27, t21);
    __m512d t34 = _mm512_fmadd_pd(k0, t28, t22);
    __m512d t35 = _mm512_mul_pd(k1, t29);
    __m512d t36 = _mm512_mul_pd(k1, t30);
    __m512d t37 = _mm512_add_pd(t33, t36);
    __m512d t38 = _mm512_sub_pd(t34, t35);
    __m512d t39 = _mm512_sub_pd(t33, t36);
    __m512d t40 = _mm512_add_pd(t34, t35);
    __m512d t41 = _mm512_add_pd(t11, t31);
    __m512d t42 = _mm512_add_pd(t12, t32);
    __m512d t43 = _mm512_sub_pd(t11, t31);
    __m512d t44 = _mm512_sub_pd(t12, t32);
    __m512d t45 = _mm512_add_pd(t17, t37);
    __m512d t46 = _mm512_add_pd(t18, t38);
    __m512d t47 = _mm512_sub_pd(t17, t37);
    __m512d t48 = _mm512_sub_pd(t18, t38);
    __m512d t49 = _mm512_add_pd(t19, t39);
    __m512d t50 = _mm512_add_pd(t20, t40);
    __m512d t51 = _mm512_sub_pd(t19, t39);
    __m512d t52 = _mm512_sub_pd(t20, t40);
    _mm512_store_pd(X+0*s, t41);
    _mm512_store_pd(X+0*s+8, t42);
    _mm512_store_pd(X+3*s, t43);
    _mm512_store_pd(X+3*s+8, t44);
    _mm512_store_pd(X+4*s, t45);
    _mm512_store_pd(X+4*s+8, t46);
    _mm512_store_pd(X+1*s, t47);
    _mm512_store_pd(X+1*s+8, t48);
    _mm512_store_pd(X+2*s, t49);
    _mm512_store_pd(X+2*s+8, t50);
    _mm512_store_pd(X+5*s, t51);
    _mm512_store_pd(X+5*s+8, t52);
    }
}
static void dft6m(double* restrict X, long es, const double* restrict C){
    const long s = es*16;
    {
    const __m512d k0 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k1 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    __m512d t1 = _mm512_load_pd(X+0*s);
    __m512d t2 = _mm512_load_pd(X+0*s+8);
    __m512d t3 = _mm512_load_pd(X+2*s);
    __m512d t4 = _mm512_load_pd(X+2*s+8);
    __m512d t5 = _mm512_load_pd(X+4*s);
    __m512d t6 = _mm512_load_pd(X+4*s+8);
    __m512d t7 = _mm512_add_pd(t3, t5);
    __m512d t8 = _mm512_add_pd(t4, t6);
    __m512d t9 = _mm512_sub_pd(t3, t5);
    __m512d t10 = _mm512_sub_pd(t4, t6);
    __m512d t11 = _mm512_add_pd(t1, t7);
    __m512d t12 = _mm512_add_pd(t2, t8);
    __m512d t13 = _mm512_fmadd_pd(k0, t7, t1);
    __m512d t14 = _mm512_fmadd_pd(k0, t8, t2);
    __m512d t15 = _mm512_mul_pd(k1, t9);
    __m512d t16 = _mm512_mul_pd(k1, t10);
    __m512d t17 = _mm512_add_pd(t13, t16);
    __m512d t18 = _mm512_sub_pd(t14, t15);
    __m512d t19 = _mm512_sub_pd(t13, t16);
    __m512d t20 = _mm512_add_pd(t14, t15);
    __m512d t21 = _mm512_load_pd(X+3*s);
    __m512d t22 = _mm512_load_pd(X+3*s+8);
    __m512d t23 = _mm512_load_pd(X+5*s);
    __m512d t24 = _mm512_load_pd(X+5*s+8);
    __m512d t25 = _mm512_load_pd(X+1*s);
    __m512d t26 = _mm512_load_pd(X+1*s+8);
    __m512d t27 = _mm512_add_pd(t23, t25);
    __m512d t28 = _mm512_add_pd(t24, t26);
    __m512d t29 = _mm512_sub_pd(t23, t25);
    __m512d t30 = _mm512_sub_pd(t24, t26);
    __m512d t31 = _mm512_add_pd(t21, t27);
    __m512d t32 = _mm512_add_pd(t22, t28);
    __m512d t33 = _mm512_fmadd_pd(k0, t27, t21);
    __m512d t34 = _mm512_fmadd_pd(k0, t28, t22);
    __m512d t35 = _mm512_mul_pd(k1, t29);
    __m512d t36 = _mm512_mul_pd(k1, t30);
    __m512d t37 = _mm512_add_pd(t33, t36);
    __m512d t38 = _mm512_sub_pd(t34, t35);
    __m512d t39 = _mm512_sub_pd(t33, t36);
    __m512d t40 = _mm512_add_pd(t34, t35);
    __m512d t41 = _mm512_add_pd(t11, t31);
    __m512d t42 = _mm512_add_pd(t12, t32);
    __m512d t43 = _mm512_sub_pd(t11, t31);
    __m512d t44 = _mm512_sub_pd(t12, t32);
    __m512d t45 = _mm512_add_pd(t17, t37);
    __m512d t46 = _mm512_add_pd(t18, t38);
    __m512d t47 = _mm512_sub_pd(t17, t37);
    __m512d t48 = _mm512_sub_pd(t18, t38);
    __m512d t49 = _mm512_add_pd(t19, t39);
    __m512d t50 = _mm512_add_pd(t20, t40);
    __m512d t51 = _mm512_sub_pd(t19, t39);
    __m512d t52 = _mm512_sub_pd(t20, t40);
    { __m512d zr = _mm512_add_pd(t41, _mm512_load_pd(C+0*16));
      __m512d zi = _mm512_add_pd(t42, _mm512_load_pd(C+0*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+0*s, zr); _mm512_store_pd(X+0*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t43, _mm512_load_pd(C+3*16));
      __m512d zi = _mm512_add_pd(t44, _mm512_load_pd(C+3*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+3*s, zr); _mm512_store_pd(X+3*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t45, _mm512_load_pd(C+4*16));
      __m512d zi = _mm512_add_pd(t46, _mm512_load_pd(C+4*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+4*s, zr); _mm512_store_pd(X+4*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t47, _mm512_load_pd(C+1*16));
      __m512d zi = _mm512_add_pd(t48, _mm512_load_pd(C+1*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+1*s, zr); _mm512_store_pd(X+1*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t49, _mm512_load_pd(C+2*16));
      __m512d zi = _mm512_add_pd(t50, _mm512_load_pd(C+2*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+2*s, zr); _mm512_store_pd(X+2*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t51, _mm512_load_pd(C+5*16));
      __m512d zi = _mm512_add_pd(t52, _mm512_load_pd(C+5*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+5*s, zr); _mm512_store_pd(X+5*s+8, zi); }
    }
}
#define PFIN_6 0

static void __attribute__((noinline)) dft6_one(double* X, long es){ dft6(X, es); }
#if PFIN_6
static void __attribute__((noinline)) dft6_onesq(double* X){ dft6pfz(X, 1); }
static void __attribute__((noinline)) dft6_onem(double* X, long es, const double* Ct){ dft6m(X, es, Ct); }
#elif 0
static void __attribute__((noinline)) dft6_onem_unused(double* X, long es, const double* Ct){ dft6mpf(X, es, Ct); }
#else
#if 0
static void __attribute__((noinline)) dft6_onesq(double* X){ dft6sq(X, 1); }
#else
static void __attribute__((noinline)) dft6_onesq(double* X){ dft6(X, 1); }
#endif
static void __attribute__((noinline)) dft6_onem(double* X, long es, const double* Ct){ dft6m(X, es, Ct); }
#endif
static void dft6_sweep_zy(double* restrict X){
#if PFCOMP
    const long PB = (36*2 + 6 - 1)/6;   /* 128B-blocks of next plane per y-codelet */
    for(long x=0; x<6; x++){
        double* P = X + x*37*16;
        const char* nxt = (const char*)(P + 37*16);
        for(long y=0; y<6; y++) dft6_one(P + y*6*16, 1);
        for(long z=0; z<6; z++){
            if(x+1 < 6){
                const char* q = nxt + z*PB*128;
                for(long l=0; l<PB; l++) _mm_prefetch(q + l*128, _MM_HINT_T0);
            }
            dft6_one(P + z*16, 6);
        }
    }
#else
    for(long x=0; x<6; x++){
        double* P = X + x*37*16;
        for(long y=0; y<6; y++) dft6_onesq(P + y*6*16);
        for(long z=0; z<6; z++) dft6_one(P + z*16, 6);
    }
#endif
}
static void dft6_sweep_x_map(double* restrict X, const double* restrict Ct){
    for(long p=0; p<36; p++) dft6_onem(X + p*16, 37, Ct + p*6*16);
}
static void dft6_sweep_x_plain(double* restrict X){
    for(long p=0; p<36; p++) dft6_one(X + p*16, 37);
}
static void dft6_sweep_zy_ms(double* restrict X, const double* restrict C){
    for(long x=0; x<6; x++){
        double* P = X + x*37*16;
        for(long y=0; y<6; y++) dft6_onesq(P + y*6*16);
        for(long z=0; z<6; z++) dft6_one(P + z*16, 6);
        if(x) mapslab(X + (x-1)*37*16, C + (x-1)*37*16, 36);
    }
    mapslab(X + (6-1)*37*16, C + (6-1)*37*16, 36);
}
/* plane-wise ingest/output (padded plane stride 37) */
static void ingest_6(const double* const* src, double* G){
    for(long x=0; x<6; x++){
        const long base = x*36;
        double* Gp = G + x*37*16;
        for(long e=0; e<36; e+=4){
            __m512d r0=_mm512_loadu_pd(src[0]+2*(base+e)), r1=_mm512_loadu_pd(src[1]+2*(base+e));
            __m512d r2=_mm512_loadu_pd(src[2]+2*(base+e)), r3=_mm512_loadu_pd(src[3]+2*(base+e));
            __m512d r4=_mm512_loadu_pd(src[4]+2*(base+e)), r5=_mm512_loadu_pd(src[5]+2*(base+e));
            __m512d r6=_mm512_loadu_pd(src[6]+2*(base+e)), r7=_mm512_loadu_pd(src[7]+2*(base+e));
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            _mm512_store_pd(Gp+e*16,    o0); _mm512_store_pd(Gp+e*16+8,  o1);
            _mm512_store_pd(Gp+e*16+16, o2); _mm512_store_pd(Gp+e*16+24, o3);
            _mm512_store_pd(Gp+e*16+32, o4); _mm512_store_pd(Gp+e*16+40, o5);
            _mm512_store_pd(Gp+e*16+48, o6); _mm512_store_pd(Gp+e*16+56, o7);
        }
#if 0 > 0
        {
            const long e = 36;
            const __mmask8 mk = (__mmask8)((1u<<(2*0))-1u);
            __m512d r0=_mm512_maskz_loadu_pd(mk, src[0]+2*(base+e)), r1=_mm512_maskz_loadu_pd(mk, src[1]+2*(base+e));
            __m512d r2=_mm512_maskz_loadu_pd(mk, src[2]+2*(base+e)), r3=_mm512_maskz_loadu_pd(mk, src[3]+2*(base+e));
            __m512d r4=_mm512_maskz_loadu_pd(mk, src[4]+2*(base+e)), r5=_mm512_maskz_loadu_pd(mk, src[5]+2*(base+e));
            __m512d r6=_mm512_maskz_loadu_pd(mk, src[6]+2*(base+e)), r7=_mm512_maskz_loadu_pd(mk, src[7]+2*(base+e));
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d A[8]; A[0]=o0;A[1]=o1;A[2]=o2;A[3]=o3;A[4]=o4;A[5]=o5;A[6]=o6;A[7]=o7;
            for(int q=0;q<2*0;q++) _mm512_store_pd(Gp+e*16+q*8, A[q]);
        }
#endif
    }
}
static void output_6(const double* G, double* const* dst, int nv){
    for(long x=0; x<6; x++){
        const long base = x*36;
        const double* Gp = G + x*37*16;
        for(long e=0; e<36; e+=4){
            __m512d i0=_mm512_load_pd(Gp+e*16),    i1=_mm512_load_pd(Gp+e*16+8);
            __m512d i2=_mm512_load_pd(Gp+e*16+16), i3=_mm512_load_pd(Gp+e*16+24);
            __m512d i4=_mm512_load_pd(Gp+e*16+32), i5=_mm512_load_pd(Gp+e*16+40);
            __m512d i6=_mm512_load_pd(Gp+e*16+48), i7=_mm512_load_pd(Gp+e*16+56);
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(i0,i1,i2,i3,i4,i5,i6,i7,o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
            for(int v=0; v<nv; v++) _mm512_storeu_pd(dst[v]+2*(base+e), *O[v]);
        }
#if 0 > 0
        {
            const long e = 36;
            const __mmask8 mk = (__mmask8)((1u<<(2*0))-1u);
            __m512d A[8];
            for(int q=0;q<2*0;q++) A[q] = _mm512_load_pd(Gp+e*16+q*8);
            for(int q=2*0;q<8;q++) A[q] = _mm512_setzero_pd();
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
            for(int v=0; v<nv; v++) _mm512_mask_storeu_pd(dst[v]+2*(base+e), mk, *O[v]);
        }
#endif
    }
}


static double* Xg_6 = 0;
static double* Cg_6 = 0;
void hot_6(long n){
    if(!Xg_6){ Xg_6 = alloc_huge_st(6*37*16*8); Cg_6 = alloc_huge_st(216*16*8); }
    for(long i=0;i<6*37*16;i++) Xg_6[i] = 0.5 + 1e-6*(i%97);
    for(long r=0;r<n;r++){
        double* P = Xg_6;
        for(long y=0; y<6; y++) dft6_one(P + y*6*16, 1);
        for(long z=0; z<6; z++) dft6_one(P + z*16, 6);
        if((r&1)==1) for(long i=0;i<36*16;i++) Xg_6[i] = 0.5 + 1e-6*(i%97);
    }
}
void hot2_6(long which){
    if(!Xg_6){ Xg_6 = alloc_huge_st((216+64*6)*16*8); Cg_6 = alloc_huge_st(216*16*8); }
    double* P = Xg_6;
    if(which==99){ for(long i=0;i<36*16;i++) P[i] = 0.5 + 1e-6*(i%97); return; }
    if(which==0 || which==2) for(long y=0; y<6; y++) dft6_one(P + y*6*16, 1);
    if(which==1 || which==2) for(long z=0; z<6; z++) dft6_one(P + z*16, 6);
}
void bsweep_6(long which, long n){
    if(!Xg_6){ Xg_6 = alloc_huge_st(6*37*16*8); Cg_6 = alloc_huge_st(216*16*8); }
    for(long i=0;i<6*37*16;i++) Xg_6[i] = 0.5 + 1e-6*(i%97);
    for(long i=0;i<216*16;i++) Cg_6[i] = 0.01;
    for(long r=0;r<n;r++){
        if(which==0) dft6_sweep_zy(Xg_6);
        else if(which==2) dft6_sweep_x_map(Xg_6, Cg_6);
        if((r&3)==3) for(long i=0;i<6*37*16;i+=997) Xg_6[i] = 0.5;
    }
}
void diag_6(long which, long n){
    if(!Xg_6){ Xg_6 = alloc_huge_st(6*37*16*8); Cg_6 = alloc_huge_st(6*37*16*8); }
    for(long i=0;i<6*37*16;i++){ Xg_6[i] = 0.5 + 1e-6*(i%97); Cg_6[i] = 0.01; }
    for(long r=0;r<n;r++){
        if(which==0){ for(long x=0;x<6;x++) mapslab(Xg_6 + x*37*16, Cg_6 + x*37*16, 36); }
        else if(which==1) dft6_sweep_zy(Xg_6);
        else dft6_sweep_x_plain(Xg_6);
        if((r&1)==1) for(long i=0;i<6*37*16;i+=997) Xg_6[i] = 0.5;
    }
}
void run_6(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    const long NE = 216;
    if(!Xg_6){ Xg_6 = alloc_huge_st(6*37*16*8); Cg_6 = alloc_huge_st(6*37*16*8); }
    double* X = Xg_6; double* Ct = Cg_6;
    for(long g0=0; g0<B; g0+=8){
        int nv = (int)((B - g0) < 8 ? (B - g0) : 8);
        const double* src[8]; const double* csrc[8];
        double* d1[8]; double* dm[8];
        for(int v=0; v<8; v++){
            int vv = v < nv ? v : 0;
            src[v] = x0 + (g0+vv)*2*NE; csrc[v] = c + (g0+vv)*2*NE;
            if(v<nv){ d1[v] = out1 + (g0+v)*2*NE; dm[v] = outm + (g0+v)*2*NE; }
        }
#if XFIRST_FLAG
        ingest_6(csrc, Ct);   /* c in padded group layout */
        ingest_6(src, X);
        for(long t=0; t<m; t++){
            dft6_sweep_x_plain(X);
            dft6_sweep_zy_ms(X, Ct);
            if(t==0 && m>1) output_6(X, d1, nv);
        }
#else
        ingest_6(csrc, X);
        for(long p=0; p<36; p++)
            for(long k=0; k<6; k++){
                _mm512_store_pd(Ct + (p*6+k)*16,     _mm512_load_pd(X + (k*37+p)*16));
                _mm512_store_pd(Ct + (p*6+k)*16 + 8, _mm512_load_pd(X + (k*37+p)*16 + 8));
            }
        ingest_6(src, X);
        for(long t=0; t<m; t++){
            dft6_sweep_zy(X);
            dft6_sweep_x_map(X, Ct);
            if(t==0 && m>1) output_6(X, d1, nv);
        }
#endif
        output_6(X, dm, nv);
        if(m==1) output_6(X, d1, nv);
    }
}

static void dft8(double* restrict X, long es){
    const long s = es*16;
    {
    const __m512d k0 = _mm512_set1_pd(0x1.6a09e667f3bcdp-1);
    __m512d r0 = _mm512_load_pd(X+0*s);
    __m512d i0 = _mm512_load_pd(X+0*s+8);
    __m512d r1 = _mm512_load_pd(X+1*s);
    __m512d i1 = _mm512_load_pd(X+1*s+8);
    __m512d r2 = _mm512_load_pd(X+2*s);
    __m512d i2 = _mm512_load_pd(X+2*s+8);
    __m512d r3 = _mm512_load_pd(X+3*s);
    __m512d i3 = _mm512_load_pd(X+3*s+8);
    __m512d r4 = _mm512_load_pd(X+4*s);
    __m512d i4 = _mm512_load_pd(X+4*s+8);
    __m512d r5 = _mm512_load_pd(X+5*s);
    __m512d i5 = _mm512_load_pd(X+5*s+8);
    __m512d r6 = _mm512_load_pd(X+6*s);
    __m512d i6 = _mm512_load_pd(X+6*s+8);
    __m512d r7 = _mm512_load_pd(X+7*s);
    __m512d i7 = _mm512_load_pd(X+7*s+8);
    __m512d t1 = _mm512_add_pd(r0, r4);
    __m512d t2 = _mm512_add_pd(i0, i4);
    __m512d t3 = _mm512_add_pd(r1, r5);
    __m512d t4 = _mm512_add_pd(i1, i5);
    __m512d t5 = _mm512_add_pd(r2, r6);
    __m512d t6 = _mm512_add_pd(i2, i6);
    __m512d t7 = _mm512_add_pd(r3, r7);
    __m512d t8 = _mm512_add_pd(i3, i7);
    __m512d t9 = _mm512_sub_pd(r0, r4);
    __m512d t10 = _mm512_sub_pd(i0, i4);
    __m512d t11 = _mm512_sub_pd(r1, r5);
    __m512d t12 = _mm512_sub_pd(i1, i5);
    __m512d t13 = _mm512_sub_pd(r2, r6);
    __m512d t14 = _mm512_sub_pd(i2, i6);
    __m512d t15 = _mm512_sub_pd(r3, r7);
    __m512d t16 = _mm512_sub_pd(i3, i7);
    __m512d t17 = _mm512_add_pd(t11, t12);
    __m512d t18 = _mm512_mul_pd(t17, k0);
    __m512d t19 = _mm512_sub_pd(t12, t11);
    __m512d t20 = _mm512_mul_pd(t19, k0);
    __m512d t21 = _mm512_setzero_pd();
    __m512d t22 = _mm512_sub_pd(t21, t13);
    __m512d t23 = _mm512_sub_pd(t16, t15);
    __m512d t24 = _mm512_mul_pd(t23, k0);
    __m512d t25 = _mm512_add_pd(t15, t16);
    __m512d t26 = _mm512_sub_pd(t21, t25);
    __m512d t27 = _mm512_mul_pd(t26, k0);
    __m512d t28 = _mm512_add_pd(t1, t5);
    __m512d t29 = _mm512_add_pd(t2, t6);
    __m512d t30 = _mm512_sub_pd(t1, t5);
    __m512d t31 = _mm512_sub_pd(t2, t6);
    __m512d t32 = _mm512_add_pd(t3, t7);
    __m512d t33 = _mm512_add_pd(t4, t8);
    __m512d t34 = _mm512_sub_pd(t3, t7);
    __m512d t35 = _mm512_sub_pd(t4, t8);
    __m512d t36 = _mm512_add_pd(t28, t32);
    __m512d t37 = _mm512_add_pd(t29, t33);
    __m512d t38 = _mm512_sub_pd(t28, t32);
    __m512d t39 = _mm512_sub_pd(t29, t33);
    __m512d t40 = _mm512_add_pd(t30, t35);
    __m512d t41 = _mm512_sub_pd(t31, t34);
    __m512d t42 = _mm512_sub_pd(t30, t35);
    __m512d t43 = _mm512_add_pd(t31, t34);
    __m512d t44 = _mm512_add_pd(t9, t14);
    __m512d t45 = _mm512_add_pd(t10, t22);
    __m512d t46 = _mm512_sub_pd(t9, t14);
    __m512d t47 = _mm512_sub_pd(t10, t22);
    __m512d t48 = _mm512_add_pd(t18, t24);
    __m512d t49 = _mm512_add_pd(t20, t27);
    __m512d t50 = _mm512_sub_pd(t18, t24);
    __m512d t51 = _mm512_sub_pd(t20, t27);
    __m512d t52 = _mm512_add_pd(t44, t48);
    __m512d t53 = _mm512_add_pd(t45, t49);
    __m512d t54 = _mm512_sub_pd(t44, t48);
    __m512d t55 = _mm512_sub_pd(t45, t49);
    __m512d t56 = _mm512_add_pd(t46, t51);
    __m512d t57 = _mm512_sub_pd(t47, t50);
    __m512d t58 = _mm512_sub_pd(t46, t51);
    __m512d t59 = _mm512_add_pd(t47, t50);
    _mm512_store_pd(X+0*s, t36);
    _mm512_store_pd(X+0*s+8, t37);
    _mm512_store_pd(X+1*s, t52);
    _mm512_store_pd(X+1*s+8, t53);
    _mm512_store_pd(X+2*s, t40);
    _mm512_store_pd(X+2*s+8, t41);
    _mm512_store_pd(X+3*s, t56);
    _mm512_store_pd(X+3*s+8, t57);
    _mm512_store_pd(X+4*s, t38);
    _mm512_store_pd(X+4*s+8, t39);
    _mm512_store_pd(X+5*s, t54);
    _mm512_store_pd(X+5*s+8, t55);
    _mm512_store_pd(X+6*s, t42);
    _mm512_store_pd(X+6*s+8, t43);
    _mm512_store_pd(X+7*s, t58);
    _mm512_store_pd(X+7*s+8, t59);
    }
}
static void dft8m(double* restrict X, long es, const double* restrict C){
    const long s = es*16;
    {
    const __m512d k0 = _mm512_set1_pd(0x1.6a09e667f3bcdp-1);
    __m512d r0 = _mm512_load_pd(X+0*s);
    __m512d i0 = _mm512_load_pd(X+0*s+8);
    __m512d r1 = _mm512_load_pd(X+1*s);
    __m512d i1 = _mm512_load_pd(X+1*s+8);
    __m512d r2 = _mm512_load_pd(X+2*s);
    __m512d i2 = _mm512_load_pd(X+2*s+8);
    __m512d r3 = _mm512_load_pd(X+3*s);
    __m512d i3 = _mm512_load_pd(X+3*s+8);
    __m512d r4 = _mm512_load_pd(X+4*s);
    __m512d i4 = _mm512_load_pd(X+4*s+8);
    __m512d r5 = _mm512_load_pd(X+5*s);
    __m512d i5 = _mm512_load_pd(X+5*s+8);
    __m512d r6 = _mm512_load_pd(X+6*s);
    __m512d i6 = _mm512_load_pd(X+6*s+8);
    __m512d r7 = _mm512_load_pd(X+7*s);
    __m512d i7 = _mm512_load_pd(X+7*s+8);
    __m512d t1 = _mm512_add_pd(r0, r4);
    __m512d t2 = _mm512_add_pd(i0, i4);
    __m512d t3 = _mm512_add_pd(r1, r5);
    __m512d t4 = _mm512_add_pd(i1, i5);
    __m512d t5 = _mm512_add_pd(r2, r6);
    __m512d t6 = _mm512_add_pd(i2, i6);
    __m512d t7 = _mm512_add_pd(r3, r7);
    __m512d t8 = _mm512_add_pd(i3, i7);
    __m512d t9 = _mm512_sub_pd(r0, r4);
    __m512d t10 = _mm512_sub_pd(i0, i4);
    __m512d t11 = _mm512_sub_pd(r1, r5);
    __m512d t12 = _mm512_sub_pd(i1, i5);
    __m512d t13 = _mm512_sub_pd(r2, r6);
    __m512d t14 = _mm512_sub_pd(i2, i6);
    __m512d t15 = _mm512_sub_pd(r3, r7);
    __m512d t16 = _mm512_sub_pd(i3, i7);
    __m512d t17 = _mm512_add_pd(t11, t12);
    __m512d t18 = _mm512_mul_pd(t17, k0);
    __m512d t19 = _mm512_sub_pd(t12, t11);
    __m512d t20 = _mm512_mul_pd(t19, k0);
    __m512d t21 = _mm512_setzero_pd();
    __m512d t22 = _mm512_sub_pd(t21, t13);
    __m512d t23 = _mm512_sub_pd(t16, t15);
    __m512d t24 = _mm512_mul_pd(t23, k0);
    __m512d t25 = _mm512_add_pd(t15, t16);
    __m512d t26 = _mm512_sub_pd(t21, t25);
    __m512d t27 = _mm512_mul_pd(t26, k0);
    __m512d t28 = _mm512_add_pd(t1, t5);
    __m512d t29 = _mm512_add_pd(t2, t6);
    __m512d t30 = _mm512_sub_pd(t1, t5);
    __m512d t31 = _mm512_sub_pd(t2, t6);
    __m512d t32 = _mm512_add_pd(t3, t7);
    __m512d t33 = _mm512_add_pd(t4, t8);
    __m512d t34 = _mm512_sub_pd(t3, t7);
    __m512d t35 = _mm512_sub_pd(t4, t8);
    __m512d t36 = _mm512_add_pd(t28, t32);
    __m512d t37 = _mm512_add_pd(t29, t33);
    __m512d t38 = _mm512_sub_pd(t28, t32);
    __m512d t39 = _mm512_sub_pd(t29, t33);
    __m512d t40 = _mm512_add_pd(t30, t35);
    __m512d t41 = _mm512_sub_pd(t31, t34);
    __m512d t42 = _mm512_sub_pd(t30, t35);
    __m512d t43 = _mm512_add_pd(t31, t34);
    __m512d t44 = _mm512_add_pd(t9, t14);
    __m512d t45 = _mm512_add_pd(t10, t22);
    __m512d t46 = _mm512_sub_pd(t9, t14);
    __m512d t47 = _mm512_sub_pd(t10, t22);
    __m512d t48 = _mm512_add_pd(t18, t24);
    __m512d t49 = _mm512_add_pd(t20, t27);
    __m512d t50 = _mm512_sub_pd(t18, t24);
    __m512d t51 = _mm512_sub_pd(t20, t27);
    __m512d t52 = _mm512_add_pd(t44, t48);
    __m512d t53 = _mm512_add_pd(t45, t49);
    __m512d t54 = _mm512_sub_pd(t44, t48);
    __m512d t55 = _mm512_sub_pd(t45, t49);
    __m512d t56 = _mm512_add_pd(t46, t51);
    __m512d t57 = _mm512_sub_pd(t47, t50);
    __m512d t58 = _mm512_sub_pd(t46, t51);
    __m512d t59 = _mm512_add_pd(t47, t50);
    { __m512d zr = _mm512_add_pd(t36, _mm512_load_pd(C+0*16));
      __m512d zi = _mm512_add_pd(t37, _mm512_load_pd(C+0*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+0*s, zr); _mm512_store_pd(X+0*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t52, _mm512_load_pd(C+1*16));
      __m512d zi = _mm512_add_pd(t53, _mm512_load_pd(C+1*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+1*s, zr); _mm512_store_pd(X+1*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t40, _mm512_load_pd(C+2*16));
      __m512d zi = _mm512_add_pd(t41, _mm512_load_pd(C+2*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+2*s, zr); _mm512_store_pd(X+2*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t56, _mm512_load_pd(C+3*16));
      __m512d zi = _mm512_add_pd(t57, _mm512_load_pd(C+3*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+3*s, zr); _mm512_store_pd(X+3*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t38, _mm512_load_pd(C+4*16));
      __m512d zi = _mm512_add_pd(t39, _mm512_load_pd(C+4*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+4*s, zr); _mm512_store_pd(X+4*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t54, _mm512_load_pd(C+5*16));
      __m512d zi = _mm512_add_pd(t55, _mm512_load_pd(C+5*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+5*s, zr); _mm512_store_pd(X+5*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t42, _mm512_load_pd(C+6*16));
      __m512d zi = _mm512_add_pd(t43, _mm512_load_pd(C+6*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+6*s, zr); _mm512_store_pd(X+6*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t58, _mm512_load_pd(C+7*16));
      __m512d zi = _mm512_add_pd(t59, _mm512_load_pd(C+7*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+7*s, zr); _mm512_store_pd(X+7*s+8, zi); }
    }
}
#define PFIN_8 0

static void __attribute__((noinline)) dft8_one(double* X, long es){ dft8(X, es); }
#if PFIN_8
static void __attribute__((noinline)) dft8_onesq(double* X){ dft8pfz(X, 1); }
static void __attribute__((noinline)) dft8_onem(double* X, long es, const double* Ct){ dft8m(X, es, Ct); }
#elif 0
static void __attribute__((noinline)) dft8_onem_unused(double* X, long es, const double* Ct){ dft8mpf(X, es, Ct); }
#else
#if 0
static void __attribute__((noinline)) dft8_onesq(double* X){ dft8sq(X, 1); }
#else
static void __attribute__((noinline)) dft8_onesq(double* X){ dft8(X, 1); }
#endif
static void __attribute__((noinline)) dft8_onem(double* X, long es, const double* Ct){ dft8m(X, es, Ct); }
#endif
static void dft8_sweep_zy(double* restrict X){
#if PFCOMP
    const long PB = (64*2 + 8 - 1)/8;   /* 128B-blocks of next plane per y-codelet */
    for(long x=0; x<8; x++){
        double* P = X + x*65*16;
        const char* nxt = (const char*)(P + 65*16);
        for(long y=0; y<8; y++) dft8_one(P + y*8*16, 1);
        for(long z=0; z<8; z++){
            if(x+1 < 8){
                const char* q = nxt + z*PB*128;
                for(long l=0; l<PB; l++) _mm_prefetch(q + l*128, _MM_HINT_T0);
            }
            dft8_one(P + z*16, 8);
        }
    }
#else
    for(long x=0; x<8; x++){
        double* P = X + x*65*16;
        for(long y=0; y<8; y++) dft8_onesq(P + y*8*16);
        for(long z=0; z<8; z++) dft8_one(P + z*16, 8);
    }
#endif
}
static void dft8_sweep_x_map(double* restrict X, const double* restrict Ct){
    for(long p=0; p<64; p++) dft8_onem(X + p*16, 65, Ct + p*8*16);
}
static void dft8_sweep_x_plain(double* restrict X){
    for(long p=0; p<64; p++) dft8_one(X + p*16, 65);
}
static void dft8_sweep_zy_ms(double* restrict X, const double* restrict C){
    for(long x=0; x<8; x++){
        double* P = X + x*65*16;
        for(long y=0; y<8; y++) dft8_onesq(P + y*8*16);
        for(long z=0; z<8; z++) dft8_one(P + z*16, 8);
        if(x) mapslab(X + (x-1)*65*16, C + (x-1)*65*16, 64);
    }
    mapslab(X + (8-1)*65*16, C + (8-1)*65*16, 64);
}
/* plane-wise ingest/output (padded plane stride 65) */
static void ingest_8(const double* const* src, double* G){
    for(long x=0; x<8; x++){
        const long base = x*64;
        double* Gp = G + x*65*16;
        for(long e=0; e<64; e+=4){
            __m512d r0=_mm512_loadu_pd(src[0]+2*(base+e)), r1=_mm512_loadu_pd(src[1]+2*(base+e));
            __m512d r2=_mm512_loadu_pd(src[2]+2*(base+e)), r3=_mm512_loadu_pd(src[3]+2*(base+e));
            __m512d r4=_mm512_loadu_pd(src[4]+2*(base+e)), r5=_mm512_loadu_pd(src[5]+2*(base+e));
            __m512d r6=_mm512_loadu_pd(src[6]+2*(base+e)), r7=_mm512_loadu_pd(src[7]+2*(base+e));
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            _mm512_store_pd(Gp+e*16,    o0); _mm512_store_pd(Gp+e*16+8,  o1);
            _mm512_store_pd(Gp+e*16+16, o2); _mm512_store_pd(Gp+e*16+24, o3);
            _mm512_store_pd(Gp+e*16+32, o4); _mm512_store_pd(Gp+e*16+40, o5);
            _mm512_store_pd(Gp+e*16+48, o6); _mm512_store_pd(Gp+e*16+56, o7);
        }
#if 0 > 0
        {
            const long e = 64;
            const __mmask8 mk = (__mmask8)((1u<<(2*0))-1u);
            __m512d r0=_mm512_maskz_loadu_pd(mk, src[0]+2*(base+e)), r1=_mm512_maskz_loadu_pd(mk, src[1]+2*(base+e));
            __m512d r2=_mm512_maskz_loadu_pd(mk, src[2]+2*(base+e)), r3=_mm512_maskz_loadu_pd(mk, src[3]+2*(base+e));
            __m512d r4=_mm512_maskz_loadu_pd(mk, src[4]+2*(base+e)), r5=_mm512_maskz_loadu_pd(mk, src[5]+2*(base+e));
            __m512d r6=_mm512_maskz_loadu_pd(mk, src[6]+2*(base+e)), r7=_mm512_maskz_loadu_pd(mk, src[7]+2*(base+e));
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d A[8]; A[0]=o0;A[1]=o1;A[2]=o2;A[3]=o3;A[4]=o4;A[5]=o5;A[6]=o6;A[7]=o7;
            for(int q=0;q<2*0;q++) _mm512_store_pd(Gp+e*16+q*8, A[q]);
        }
#endif
    }
}
static void output_8(const double* G, double* const* dst, int nv){
    for(long x=0; x<8; x++){
        const long base = x*64;
        const double* Gp = G + x*65*16;
        for(long e=0; e<64; e+=4){
            __m512d i0=_mm512_load_pd(Gp+e*16),    i1=_mm512_load_pd(Gp+e*16+8);
            __m512d i2=_mm512_load_pd(Gp+e*16+16), i3=_mm512_load_pd(Gp+e*16+24);
            __m512d i4=_mm512_load_pd(Gp+e*16+32), i5=_mm512_load_pd(Gp+e*16+40);
            __m512d i6=_mm512_load_pd(Gp+e*16+48), i7=_mm512_load_pd(Gp+e*16+56);
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(i0,i1,i2,i3,i4,i5,i6,i7,o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
            for(int v=0; v<nv; v++) _mm512_storeu_pd(dst[v]+2*(base+e), *O[v]);
        }
#if 0 > 0
        {
            const long e = 64;
            const __mmask8 mk = (__mmask8)((1u<<(2*0))-1u);
            __m512d A[8];
            for(int q=0;q<2*0;q++) A[q] = _mm512_load_pd(Gp+e*16+q*8);
            for(int q=2*0;q<8;q++) A[q] = _mm512_setzero_pd();
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
            for(int v=0; v<nv; v++) _mm512_mask_storeu_pd(dst[v]+2*(base+e), mk, *O[v]);
        }
#endif
    }
}


static double* Xg_8 = 0;
static double* Cg_8 = 0;
void hot_8(long n){
    if(!Xg_8){ Xg_8 = alloc_huge_st(8*65*16*8); Cg_8 = alloc_huge_st(512*16*8); }
    for(long i=0;i<8*65*16;i++) Xg_8[i] = 0.5 + 1e-6*(i%97);
    for(long r=0;r<n;r++){
        double* P = Xg_8;
        for(long y=0; y<8; y++) dft8_one(P + y*8*16, 1);
        for(long z=0; z<8; z++) dft8_one(P + z*16, 8);
        if((r&1)==1) for(long i=0;i<64*16;i++) Xg_8[i] = 0.5 + 1e-6*(i%97);
    }
}
void hot2_8(long which){
    if(!Xg_8){ Xg_8 = alloc_huge_st((512+64*8)*16*8); Cg_8 = alloc_huge_st(512*16*8); }
    double* P = Xg_8;
    if(which==99){ for(long i=0;i<64*16;i++) P[i] = 0.5 + 1e-6*(i%97); return; }
    if(which==0 || which==2) for(long y=0; y<8; y++) dft8_one(P + y*8*16, 1);
    if(which==1 || which==2) for(long z=0; z<8; z++) dft8_one(P + z*16, 8);
}
void bsweep_8(long which, long n){
    if(!Xg_8){ Xg_8 = alloc_huge_st(8*65*16*8); Cg_8 = alloc_huge_st(512*16*8); }
    for(long i=0;i<8*65*16;i++) Xg_8[i] = 0.5 + 1e-6*(i%97);
    for(long i=0;i<512*16;i++) Cg_8[i] = 0.01;
    for(long r=0;r<n;r++){
        if(which==0) dft8_sweep_zy(Xg_8);
        else if(which==2) dft8_sweep_x_map(Xg_8, Cg_8);
        if((r&3)==3) for(long i=0;i<8*65*16;i+=997) Xg_8[i] = 0.5;
    }
}
void diag_8(long which, long n){
    if(!Xg_8){ Xg_8 = alloc_huge_st(8*65*16*8); Cg_8 = alloc_huge_st(8*65*16*8); }
    for(long i=0;i<8*65*16;i++){ Xg_8[i] = 0.5 + 1e-6*(i%97); Cg_8[i] = 0.01; }
    for(long r=0;r<n;r++){
        if(which==0){ for(long x=0;x<8;x++) mapslab(Xg_8 + x*65*16, Cg_8 + x*65*16, 64); }
        else if(which==1) dft8_sweep_zy(Xg_8);
        else dft8_sweep_x_plain(Xg_8);
        if((r&1)==1) for(long i=0;i<8*65*16;i+=997) Xg_8[i] = 0.5;
    }
}
void run_8(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    const long NE = 512;
    if(!Xg_8){ Xg_8 = alloc_huge_st(8*65*16*8); Cg_8 = alloc_huge_st(8*65*16*8); }
    double* X = Xg_8; double* Ct = Cg_8;
    for(long g0=0; g0<B; g0+=8){
        int nv = (int)((B - g0) < 8 ? (B - g0) : 8);
        const double* src[8]; const double* csrc[8];
        double* d1[8]; double* dm[8];
        for(int v=0; v<8; v++){
            int vv = v < nv ? v : 0;
            src[v] = x0 + (g0+vv)*2*NE; csrc[v] = c + (g0+vv)*2*NE;
            if(v<nv){ d1[v] = out1 + (g0+v)*2*NE; dm[v] = outm + (g0+v)*2*NE; }
        }
#if XFIRST_FLAG
        ingest_8(csrc, Ct);   /* c in padded group layout */
        ingest_8(src, X);
        for(long t=0; t<m; t++){
            dft8_sweep_x_plain(X);
            dft8_sweep_zy_ms(X, Ct);
            if(t==0 && m>1) output_8(X, d1, nv);
        }
#else
        ingest_8(csrc, X);
        for(long p=0; p<64; p++)
            for(long k=0; k<8; k++){
                _mm512_store_pd(Ct + (p*8+k)*16,     _mm512_load_pd(X + (k*65+p)*16));
                _mm512_store_pd(Ct + (p*8+k)*16 + 8, _mm512_load_pd(X + (k*65+p)*16 + 8));
            }
        ingest_8(src, X);
        for(long t=0; t<m; t++){
            dft8_sweep_zy(X);
            dft8_sweep_x_map(X, Ct);
            if(t==0 && m>1) output_8(X, d1, nv);
        }
#endif
        output_8(X, dm, nv);
        if(m==1) output_8(X, d1, nv);
    }
}

static void dft36(double* restrict X, long es){
    const long s = es*16;
    double SCR[576] ALIGN64;
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+0*s);
    __m512d i0 = _mm512_load_pd(X+0*s+8);
    __m512d r1 = _mm512_load_pd(X+4*s);
    __m512d i1 = _mm512_load_pd(X+4*s+8);
    __m512d r2 = _mm512_load_pd(X+8*s);
    __m512d i2 = _mm512_load_pd(X+8*s+8);
    __m512d r3 = _mm512_load_pd(X+12*s);
    __m512d i3 = _mm512_load_pd(X+12*s+8);
    __m512d r4 = _mm512_load_pd(X+16*s);
    __m512d i4 = _mm512_load_pd(X+16*s+8);
    __m512d r5 = _mm512_load_pd(X+20*s);
    __m512d i5 = _mm512_load_pd(X+20*s+8);
    __m512d r6 = _mm512_load_pd(X+24*s);
    __m512d i6 = _mm512_load_pd(X+24*s+8);
    __m512d r7 = _mm512_load_pd(X+28*s);
    __m512d i7 = _mm512_load_pd(X+28*s+8);
    __m512d r8 = _mm512_load_pd(X+32*s);
    __m512d i8 = _mm512_load_pd(X+32*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+0, t23);
    _mm512_store_pd(SCR+8, t24);
    _mm512_store_pd(SCR+64, t41);
    _mm512_store_pd(SCR+72, t42);
    _mm512_store_pd(SCR+128, t61);
    _mm512_store_pd(SCR+136, t62);
    _mm512_store_pd(SCR+192, t81);
    _mm512_store_pd(SCR+200, t82);
    _mm512_store_pd(SCR+256, t101);
    _mm512_store_pd(SCR+264, t102);
    _mm512_store_pd(SCR+320, t103);
    _mm512_store_pd(SCR+328, t104);
    _mm512_store_pd(SCR+384, t83);
    _mm512_store_pd(SCR+392, t84);
    _mm512_store_pd(SCR+448, t63);
    _mm512_store_pd(SCR+456, t64);
    _mm512_store_pd(SCR+512, t43);
    _mm512_store_pd(SCR+520, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+9*s);
    __m512d i0 = _mm512_load_pd(X+9*s+8);
    __m512d r1 = _mm512_load_pd(X+13*s);
    __m512d i1 = _mm512_load_pd(X+13*s+8);
    __m512d r2 = _mm512_load_pd(X+17*s);
    __m512d i2 = _mm512_load_pd(X+17*s+8);
    __m512d r3 = _mm512_load_pd(X+21*s);
    __m512d i3 = _mm512_load_pd(X+21*s+8);
    __m512d r4 = _mm512_load_pd(X+25*s);
    __m512d i4 = _mm512_load_pd(X+25*s+8);
    __m512d r5 = _mm512_load_pd(X+29*s);
    __m512d i5 = _mm512_load_pd(X+29*s+8);
    __m512d r6 = _mm512_load_pd(X+33*s);
    __m512d i6 = _mm512_load_pd(X+33*s+8);
    __m512d r7 = _mm512_load_pd(X+1*s);
    __m512d i7 = _mm512_load_pd(X+1*s+8);
    __m512d r8 = _mm512_load_pd(X+5*s);
    __m512d i8 = _mm512_load_pd(X+5*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+16, t23);
    _mm512_store_pd(SCR+24, t24);
    _mm512_store_pd(SCR+80, t41);
    _mm512_store_pd(SCR+88, t42);
    _mm512_store_pd(SCR+144, t61);
    _mm512_store_pd(SCR+152, t62);
    _mm512_store_pd(SCR+208, t81);
    _mm512_store_pd(SCR+216, t82);
    _mm512_store_pd(SCR+272, t101);
    _mm512_store_pd(SCR+280, t102);
    _mm512_store_pd(SCR+336, t103);
    _mm512_store_pd(SCR+344, t104);
    _mm512_store_pd(SCR+400, t83);
    _mm512_store_pd(SCR+408, t84);
    _mm512_store_pd(SCR+464, t63);
    _mm512_store_pd(SCR+472, t64);
    _mm512_store_pd(SCR+528, t43);
    _mm512_store_pd(SCR+536, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+18*s);
    __m512d i0 = _mm512_load_pd(X+18*s+8);
    __m512d r1 = _mm512_load_pd(X+22*s);
    __m512d i1 = _mm512_load_pd(X+22*s+8);
    __m512d r2 = _mm512_load_pd(X+26*s);
    __m512d i2 = _mm512_load_pd(X+26*s+8);
    __m512d r3 = _mm512_load_pd(X+30*s);
    __m512d i3 = _mm512_load_pd(X+30*s+8);
    __m512d r4 = _mm512_load_pd(X+34*s);
    __m512d i4 = _mm512_load_pd(X+34*s+8);
    __m512d r5 = _mm512_load_pd(X+2*s);
    __m512d i5 = _mm512_load_pd(X+2*s+8);
    __m512d r6 = _mm512_load_pd(X+6*s);
    __m512d i6 = _mm512_load_pd(X+6*s+8);
    __m512d r7 = _mm512_load_pd(X+10*s);
    __m512d i7 = _mm512_load_pd(X+10*s+8);
    __m512d r8 = _mm512_load_pd(X+14*s);
    __m512d i8 = _mm512_load_pd(X+14*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+32, t23);
    _mm512_store_pd(SCR+40, t24);
    _mm512_store_pd(SCR+96, t41);
    _mm512_store_pd(SCR+104, t42);
    _mm512_store_pd(SCR+160, t61);
    _mm512_store_pd(SCR+168, t62);
    _mm512_store_pd(SCR+224, t81);
    _mm512_store_pd(SCR+232, t82);
    _mm512_store_pd(SCR+288, t101);
    _mm512_store_pd(SCR+296, t102);
    _mm512_store_pd(SCR+352, t103);
    _mm512_store_pd(SCR+360, t104);
    _mm512_store_pd(SCR+416, t83);
    _mm512_store_pd(SCR+424, t84);
    _mm512_store_pd(SCR+480, t63);
    _mm512_store_pd(SCR+488, t64);
    _mm512_store_pd(SCR+544, t43);
    _mm512_store_pd(SCR+552, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+27*s);
    __m512d i0 = _mm512_load_pd(X+27*s+8);
    __m512d r1 = _mm512_load_pd(X+31*s);
    __m512d i1 = _mm512_load_pd(X+31*s+8);
    __m512d r2 = _mm512_load_pd(X+35*s);
    __m512d i2 = _mm512_load_pd(X+35*s+8);
    __m512d r3 = _mm512_load_pd(X+3*s);
    __m512d i3 = _mm512_load_pd(X+3*s+8);
    __m512d r4 = _mm512_load_pd(X+7*s);
    __m512d i4 = _mm512_load_pd(X+7*s+8);
    __m512d r5 = _mm512_load_pd(X+11*s);
    __m512d i5 = _mm512_load_pd(X+11*s+8);
    __m512d r6 = _mm512_load_pd(X+15*s);
    __m512d i6 = _mm512_load_pd(X+15*s+8);
    __m512d r7 = _mm512_load_pd(X+19*s);
    __m512d i7 = _mm512_load_pd(X+19*s+8);
    __m512d r8 = _mm512_load_pd(X+23*s);
    __m512d i8 = _mm512_load_pd(X+23*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+48, t23);
    _mm512_store_pd(SCR+56, t24);
    _mm512_store_pd(SCR+112, t41);
    _mm512_store_pd(SCR+120, t42);
    _mm512_store_pd(SCR+176, t61);
    _mm512_store_pd(SCR+184, t62);
    _mm512_store_pd(SCR+240, t81);
    _mm512_store_pd(SCR+248, t82);
    _mm512_store_pd(SCR+304, t101);
    _mm512_store_pd(SCR+312, t102);
    _mm512_store_pd(SCR+368, t103);
    _mm512_store_pd(SCR+376, t104);
    _mm512_store_pd(SCR+432, t83);
    _mm512_store_pd(SCR+440, t84);
    _mm512_store_pd(SCR+496, t63);
    _mm512_store_pd(SCR+504, t64);
    _mm512_store_pd(SCR+560, t43);
    _mm512_store_pd(SCR+568, t44);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+0);
    __m512d i0 = _mm512_load_pd(SCR+8);
    __m512d r1 = _mm512_load_pd(SCR+16);
    __m512d i1 = _mm512_load_pd(SCR+24);
    __m512d r2 = _mm512_load_pd(SCR+32);
    __m512d i2 = _mm512_load_pd(SCR+40);
    __m512d r3 = _mm512_load_pd(SCR+48);
    __m512d i3 = _mm512_load_pd(SCR+56);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+0*s, t9);
    _mm512_store_pd(X+0*s+8, t10);
    _mm512_store_pd(X+9*s, t13);
    _mm512_store_pd(X+9*s+8, t14);
    _mm512_store_pd(X+18*s, t11);
    _mm512_store_pd(X+18*s+8, t12);
    _mm512_store_pd(X+27*s, t15);
    _mm512_store_pd(X+27*s+8, t16);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+64);
    __m512d i0 = _mm512_load_pd(SCR+72);
    __m512d r1 = _mm512_load_pd(SCR+80);
    __m512d i1 = _mm512_load_pd(SCR+88);
    __m512d r2 = _mm512_load_pd(SCR+96);
    __m512d i2 = _mm512_load_pd(SCR+104);
    __m512d r3 = _mm512_load_pd(SCR+112);
    __m512d i3 = _mm512_load_pd(SCR+120);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+28*s, t9);
    _mm512_store_pd(X+28*s+8, t10);
    _mm512_store_pd(X+1*s, t13);
    _mm512_store_pd(X+1*s+8, t14);
    _mm512_store_pd(X+10*s, t11);
    _mm512_store_pd(X+10*s+8, t12);
    _mm512_store_pd(X+19*s, t15);
    _mm512_store_pd(X+19*s+8, t16);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+128);
    __m512d i0 = _mm512_load_pd(SCR+136);
    __m512d r1 = _mm512_load_pd(SCR+144);
    __m512d i1 = _mm512_load_pd(SCR+152);
    __m512d r2 = _mm512_load_pd(SCR+160);
    __m512d i2 = _mm512_load_pd(SCR+168);
    __m512d r3 = _mm512_load_pd(SCR+176);
    __m512d i3 = _mm512_load_pd(SCR+184);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+20*s, t9);
    _mm512_store_pd(X+20*s+8, t10);
    _mm512_store_pd(X+29*s, t13);
    _mm512_store_pd(X+29*s+8, t14);
    _mm512_store_pd(X+2*s, t11);
    _mm512_store_pd(X+2*s+8, t12);
    _mm512_store_pd(X+11*s, t15);
    _mm512_store_pd(X+11*s+8, t16);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+192);
    __m512d i0 = _mm512_load_pd(SCR+200);
    __m512d r1 = _mm512_load_pd(SCR+208);
    __m512d i1 = _mm512_load_pd(SCR+216);
    __m512d r2 = _mm512_load_pd(SCR+224);
    __m512d i2 = _mm512_load_pd(SCR+232);
    __m512d r3 = _mm512_load_pd(SCR+240);
    __m512d i3 = _mm512_load_pd(SCR+248);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+12*s, t9);
    _mm512_store_pd(X+12*s+8, t10);
    _mm512_store_pd(X+21*s, t13);
    _mm512_store_pd(X+21*s+8, t14);
    _mm512_store_pd(X+30*s, t11);
    _mm512_store_pd(X+30*s+8, t12);
    _mm512_store_pd(X+3*s, t15);
    _mm512_store_pd(X+3*s+8, t16);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+256);
    __m512d i0 = _mm512_load_pd(SCR+264);
    __m512d r1 = _mm512_load_pd(SCR+272);
    __m512d i1 = _mm512_load_pd(SCR+280);
    __m512d r2 = _mm512_load_pd(SCR+288);
    __m512d i2 = _mm512_load_pd(SCR+296);
    __m512d r3 = _mm512_load_pd(SCR+304);
    __m512d i3 = _mm512_load_pd(SCR+312);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+4*s, t9);
    _mm512_store_pd(X+4*s+8, t10);
    _mm512_store_pd(X+13*s, t13);
    _mm512_store_pd(X+13*s+8, t14);
    _mm512_store_pd(X+22*s, t11);
    _mm512_store_pd(X+22*s+8, t12);
    _mm512_store_pd(X+31*s, t15);
    _mm512_store_pd(X+31*s+8, t16);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+320);
    __m512d i0 = _mm512_load_pd(SCR+328);
    __m512d r1 = _mm512_load_pd(SCR+336);
    __m512d i1 = _mm512_load_pd(SCR+344);
    __m512d r2 = _mm512_load_pd(SCR+352);
    __m512d i2 = _mm512_load_pd(SCR+360);
    __m512d r3 = _mm512_load_pd(SCR+368);
    __m512d i3 = _mm512_load_pd(SCR+376);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+32*s, t9);
    _mm512_store_pd(X+32*s+8, t10);
    _mm512_store_pd(X+5*s, t13);
    _mm512_store_pd(X+5*s+8, t14);
    _mm512_store_pd(X+14*s, t11);
    _mm512_store_pd(X+14*s+8, t12);
    _mm512_store_pd(X+23*s, t15);
    _mm512_store_pd(X+23*s+8, t16);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+384);
    __m512d i0 = _mm512_load_pd(SCR+392);
    __m512d r1 = _mm512_load_pd(SCR+400);
    __m512d i1 = _mm512_load_pd(SCR+408);
    __m512d r2 = _mm512_load_pd(SCR+416);
    __m512d i2 = _mm512_load_pd(SCR+424);
    __m512d r3 = _mm512_load_pd(SCR+432);
    __m512d i3 = _mm512_load_pd(SCR+440);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+24*s, t9);
    _mm512_store_pd(X+24*s+8, t10);
    _mm512_store_pd(X+33*s, t13);
    _mm512_store_pd(X+33*s+8, t14);
    _mm512_store_pd(X+6*s, t11);
    _mm512_store_pd(X+6*s+8, t12);
    _mm512_store_pd(X+15*s, t15);
    _mm512_store_pd(X+15*s+8, t16);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+448);
    __m512d i0 = _mm512_load_pd(SCR+456);
    __m512d r1 = _mm512_load_pd(SCR+464);
    __m512d i1 = _mm512_load_pd(SCR+472);
    __m512d r2 = _mm512_load_pd(SCR+480);
    __m512d i2 = _mm512_load_pd(SCR+488);
    __m512d r3 = _mm512_load_pd(SCR+496);
    __m512d i3 = _mm512_load_pd(SCR+504);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+16*s, t9);
    _mm512_store_pd(X+16*s+8, t10);
    _mm512_store_pd(X+25*s, t13);
    _mm512_store_pd(X+25*s+8, t14);
    _mm512_store_pd(X+34*s, t11);
    _mm512_store_pd(X+34*s+8, t12);
    _mm512_store_pd(X+7*s, t15);
    _mm512_store_pd(X+7*s+8, t16);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+512);
    __m512d i0 = _mm512_load_pd(SCR+520);
    __m512d r1 = _mm512_load_pd(SCR+528);
    __m512d i1 = _mm512_load_pd(SCR+536);
    __m512d r2 = _mm512_load_pd(SCR+544);
    __m512d i2 = _mm512_load_pd(SCR+552);
    __m512d r3 = _mm512_load_pd(SCR+560);
    __m512d i3 = _mm512_load_pd(SCR+568);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+8*s, t9);
    _mm512_store_pd(X+8*s+8, t10);
    _mm512_store_pd(X+17*s, t13);
    _mm512_store_pd(X+17*s+8, t14);
    _mm512_store_pd(X+26*s, t11);
    _mm512_store_pd(X+26*s+8, t12);
    _mm512_store_pd(X+35*s, t15);
    _mm512_store_pd(X+35*s+8, t16);
    }
}
static void dft36sq(double* restrict X, long es){
    const long s = es*16;
    double SCR[576] ALIGN64;
    double SQ[576] ALIGN64;
    for(long q=0; q<576; q+=16){
        _mm512_store_pd(SQ+q,   _mm512_load_pd(X+q));
        _mm512_store_pd(SQ+q+8, _mm512_load_pd(X+q+8));
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(SQ+0*16);
    __m512d i0 = _mm512_load_pd(SQ+0*16+8);
    __m512d r1 = _mm512_load_pd(SQ+4*16);
    __m512d i1 = _mm512_load_pd(SQ+4*16+8);
    __m512d r2 = _mm512_load_pd(SQ+8*16);
    __m512d i2 = _mm512_load_pd(SQ+8*16+8);
    __m512d r3 = _mm512_load_pd(SQ+12*16);
    __m512d i3 = _mm512_load_pd(SQ+12*16+8);
    __m512d r4 = _mm512_load_pd(SQ+16*16);
    __m512d i4 = _mm512_load_pd(SQ+16*16+8);
    __m512d r5 = _mm512_load_pd(SQ+20*16);
    __m512d i5 = _mm512_load_pd(SQ+20*16+8);
    __m512d r6 = _mm512_load_pd(SQ+24*16);
    __m512d i6 = _mm512_load_pd(SQ+24*16+8);
    __m512d r7 = _mm512_load_pd(SQ+28*16);
    __m512d i7 = _mm512_load_pd(SQ+28*16+8);
    __m512d r8 = _mm512_load_pd(SQ+32*16);
    __m512d i8 = _mm512_load_pd(SQ+32*16+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+0, t23);
    _mm512_store_pd(SCR+8, t24);
    _mm512_store_pd(SCR+64, t41);
    _mm512_store_pd(SCR+72, t42);
    _mm512_store_pd(SCR+128, t61);
    _mm512_store_pd(SCR+136, t62);
    _mm512_store_pd(SCR+192, t81);
    _mm512_store_pd(SCR+200, t82);
    _mm512_store_pd(SCR+256, t101);
    _mm512_store_pd(SCR+264, t102);
    _mm512_store_pd(SCR+320, t103);
    _mm512_store_pd(SCR+328, t104);
    _mm512_store_pd(SCR+384, t83);
    _mm512_store_pd(SCR+392, t84);
    _mm512_store_pd(SCR+448, t63);
    _mm512_store_pd(SCR+456, t64);
    _mm512_store_pd(SCR+512, t43);
    _mm512_store_pd(SCR+520, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(SQ+9*16);
    __m512d i0 = _mm512_load_pd(SQ+9*16+8);
    __m512d r1 = _mm512_load_pd(SQ+13*16);
    __m512d i1 = _mm512_load_pd(SQ+13*16+8);
    __m512d r2 = _mm512_load_pd(SQ+17*16);
    __m512d i2 = _mm512_load_pd(SQ+17*16+8);
    __m512d r3 = _mm512_load_pd(SQ+21*16);
    __m512d i3 = _mm512_load_pd(SQ+21*16+8);
    __m512d r4 = _mm512_load_pd(SQ+25*16);
    __m512d i4 = _mm512_load_pd(SQ+25*16+8);
    __m512d r5 = _mm512_load_pd(SQ+29*16);
    __m512d i5 = _mm512_load_pd(SQ+29*16+8);
    __m512d r6 = _mm512_load_pd(SQ+33*16);
    __m512d i6 = _mm512_load_pd(SQ+33*16+8);
    __m512d r7 = _mm512_load_pd(SQ+1*16);
    __m512d i7 = _mm512_load_pd(SQ+1*16+8);
    __m512d r8 = _mm512_load_pd(SQ+5*16);
    __m512d i8 = _mm512_load_pd(SQ+5*16+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+16, t23);
    _mm512_store_pd(SCR+24, t24);
    _mm512_store_pd(SCR+80, t41);
    _mm512_store_pd(SCR+88, t42);
    _mm512_store_pd(SCR+144, t61);
    _mm512_store_pd(SCR+152, t62);
    _mm512_store_pd(SCR+208, t81);
    _mm512_store_pd(SCR+216, t82);
    _mm512_store_pd(SCR+272, t101);
    _mm512_store_pd(SCR+280, t102);
    _mm512_store_pd(SCR+336, t103);
    _mm512_store_pd(SCR+344, t104);
    _mm512_store_pd(SCR+400, t83);
    _mm512_store_pd(SCR+408, t84);
    _mm512_store_pd(SCR+464, t63);
    _mm512_store_pd(SCR+472, t64);
    _mm512_store_pd(SCR+528, t43);
    _mm512_store_pd(SCR+536, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(SQ+18*16);
    __m512d i0 = _mm512_load_pd(SQ+18*16+8);
    __m512d r1 = _mm512_load_pd(SQ+22*16);
    __m512d i1 = _mm512_load_pd(SQ+22*16+8);
    __m512d r2 = _mm512_load_pd(SQ+26*16);
    __m512d i2 = _mm512_load_pd(SQ+26*16+8);
    __m512d r3 = _mm512_load_pd(SQ+30*16);
    __m512d i3 = _mm512_load_pd(SQ+30*16+8);
    __m512d r4 = _mm512_load_pd(SQ+34*16);
    __m512d i4 = _mm512_load_pd(SQ+34*16+8);
    __m512d r5 = _mm512_load_pd(SQ+2*16);
    __m512d i5 = _mm512_load_pd(SQ+2*16+8);
    __m512d r6 = _mm512_load_pd(SQ+6*16);
    __m512d i6 = _mm512_load_pd(SQ+6*16+8);
    __m512d r7 = _mm512_load_pd(SQ+10*16);
    __m512d i7 = _mm512_load_pd(SQ+10*16+8);
    __m512d r8 = _mm512_load_pd(SQ+14*16);
    __m512d i8 = _mm512_load_pd(SQ+14*16+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+32, t23);
    _mm512_store_pd(SCR+40, t24);
    _mm512_store_pd(SCR+96, t41);
    _mm512_store_pd(SCR+104, t42);
    _mm512_store_pd(SCR+160, t61);
    _mm512_store_pd(SCR+168, t62);
    _mm512_store_pd(SCR+224, t81);
    _mm512_store_pd(SCR+232, t82);
    _mm512_store_pd(SCR+288, t101);
    _mm512_store_pd(SCR+296, t102);
    _mm512_store_pd(SCR+352, t103);
    _mm512_store_pd(SCR+360, t104);
    _mm512_store_pd(SCR+416, t83);
    _mm512_store_pd(SCR+424, t84);
    _mm512_store_pd(SCR+480, t63);
    _mm512_store_pd(SCR+488, t64);
    _mm512_store_pd(SCR+544, t43);
    _mm512_store_pd(SCR+552, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(SQ+27*16);
    __m512d i0 = _mm512_load_pd(SQ+27*16+8);
    __m512d r1 = _mm512_load_pd(SQ+31*16);
    __m512d i1 = _mm512_load_pd(SQ+31*16+8);
    __m512d r2 = _mm512_load_pd(SQ+35*16);
    __m512d i2 = _mm512_load_pd(SQ+35*16+8);
    __m512d r3 = _mm512_load_pd(SQ+3*16);
    __m512d i3 = _mm512_load_pd(SQ+3*16+8);
    __m512d r4 = _mm512_load_pd(SQ+7*16);
    __m512d i4 = _mm512_load_pd(SQ+7*16+8);
    __m512d r5 = _mm512_load_pd(SQ+11*16);
    __m512d i5 = _mm512_load_pd(SQ+11*16+8);
    __m512d r6 = _mm512_load_pd(SQ+15*16);
    __m512d i6 = _mm512_load_pd(SQ+15*16+8);
    __m512d r7 = _mm512_load_pd(SQ+19*16);
    __m512d i7 = _mm512_load_pd(SQ+19*16+8);
    __m512d r8 = _mm512_load_pd(SQ+23*16);
    __m512d i8 = _mm512_load_pd(SQ+23*16+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+48, t23);
    _mm512_store_pd(SCR+56, t24);
    _mm512_store_pd(SCR+112, t41);
    _mm512_store_pd(SCR+120, t42);
    _mm512_store_pd(SCR+176, t61);
    _mm512_store_pd(SCR+184, t62);
    _mm512_store_pd(SCR+240, t81);
    _mm512_store_pd(SCR+248, t82);
    _mm512_store_pd(SCR+304, t101);
    _mm512_store_pd(SCR+312, t102);
    _mm512_store_pd(SCR+368, t103);
    _mm512_store_pd(SCR+376, t104);
    _mm512_store_pd(SCR+432, t83);
    _mm512_store_pd(SCR+440, t84);
    _mm512_store_pd(SCR+496, t63);
    _mm512_store_pd(SCR+504, t64);
    _mm512_store_pd(SCR+560, t43);
    _mm512_store_pd(SCR+568, t44);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+0);
    __m512d i0 = _mm512_load_pd(SCR+8);
    __m512d r1 = _mm512_load_pd(SCR+16);
    __m512d i1 = _mm512_load_pd(SCR+24);
    __m512d r2 = _mm512_load_pd(SCR+32);
    __m512d i2 = _mm512_load_pd(SCR+40);
    __m512d r3 = _mm512_load_pd(SCR+48);
    __m512d i3 = _mm512_load_pd(SCR+56);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+0*s, t9);
    _mm512_store_pd(X+0*s+8, t10);
    _mm512_store_pd(X+9*s, t13);
    _mm512_store_pd(X+9*s+8, t14);
    _mm512_store_pd(X+18*s, t11);
    _mm512_store_pd(X+18*s+8, t12);
    _mm512_store_pd(X+27*s, t15);
    _mm512_store_pd(X+27*s+8, t16);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+64);
    __m512d i0 = _mm512_load_pd(SCR+72);
    __m512d r1 = _mm512_load_pd(SCR+80);
    __m512d i1 = _mm512_load_pd(SCR+88);
    __m512d r2 = _mm512_load_pd(SCR+96);
    __m512d i2 = _mm512_load_pd(SCR+104);
    __m512d r3 = _mm512_load_pd(SCR+112);
    __m512d i3 = _mm512_load_pd(SCR+120);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+28*s, t9);
    _mm512_store_pd(X+28*s+8, t10);
    _mm512_store_pd(X+1*s, t13);
    _mm512_store_pd(X+1*s+8, t14);
    _mm512_store_pd(X+10*s, t11);
    _mm512_store_pd(X+10*s+8, t12);
    _mm512_store_pd(X+19*s, t15);
    _mm512_store_pd(X+19*s+8, t16);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+128);
    __m512d i0 = _mm512_load_pd(SCR+136);
    __m512d r1 = _mm512_load_pd(SCR+144);
    __m512d i1 = _mm512_load_pd(SCR+152);
    __m512d r2 = _mm512_load_pd(SCR+160);
    __m512d i2 = _mm512_load_pd(SCR+168);
    __m512d r3 = _mm512_load_pd(SCR+176);
    __m512d i3 = _mm512_load_pd(SCR+184);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+20*s, t9);
    _mm512_store_pd(X+20*s+8, t10);
    _mm512_store_pd(X+29*s, t13);
    _mm512_store_pd(X+29*s+8, t14);
    _mm512_store_pd(X+2*s, t11);
    _mm512_store_pd(X+2*s+8, t12);
    _mm512_store_pd(X+11*s, t15);
    _mm512_store_pd(X+11*s+8, t16);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+192);
    __m512d i0 = _mm512_load_pd(SCR+200);
    __m512d r1 = _mm512_load_pd(SCR+208);
    __m512d i1 = _mm512_load_pd(SCR+216);
    __m512d r2 = _mm512_load_pd(SCR+224);
    __m512d i2 = _mm512_load_pd(SCR+232);
    __m512d r3 = _mm512_load_pd(SCR+240);
    __m512d i3 = _mm512_load_pd(SCR+248);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+12*s, t9);
    _mm512_store_pd(X+12*s+8, t10);
    _mm512_store_pd(X+21*s, t13);
    _mm512_store_pd(X+21*s+8, t14);
    _mm512_store_pd(X+30*s, t11);
    _mm512_store_pd(X+30*s+8, t12);
    _mm512_store_pd(X+3*s, t15);
    _mm512_store_pd(X+3*s+8, t16);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+256);
    __m512d i0 = _mm512_load_pd(SCR+264);
    __m512d r1 = _mm512_load_pd(SCR+272);
    __m512d i1 = _mm512_load_pd(SCR+280);
    __m512d r2 = _mm512_load_pd(SCR+288);
    __m512d i2 = _mm512_load_pd(SCR+296);
    __m512d r3 = _mm512_load_pd(SCR+304);
    __m512d i3 = _mm512_load_pd(SCR+312);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+4*s, t9);
    _mm512_store_pd(X+4*s+8, t10);
    _mm512_store_pd(X+13*s, t13);
    _mm512_store_pd(X+13*s+8, t14);
    _mm512_store_pd(X+22*s, t11);
    _mm512_store_pd(X+22*s+8, t12);
    _mm512_store_pd(X+31*s, t15);
    _mm512_store_pd(X+31*s+8, t16);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+320);
    __m512d i0 = _mm512_load_pd(SCR+328);
    __m512d r1 = _mm512_load_pd(SCR+336);
    __m512d i1 = _mm512_load_pd(SCR+344);
    __m512d r2 = _mm512_load_pd(SCR+352);
    __m512d i2 = _mm512_load_pd(SCR+360);
    __m512d r3 = _mm512_load_pd(SCR+368);
    __m512d i3 = _mm512_load_pd(SCR+376);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+32*s, t9);
    _mm512_store_pd(X+32*s+8, t10);
    _mm512_store_pd(X+5*s, t13);
    _mm512_store_pd(X+5*s+8, t14);
    _mm512_store_pd(X+14*s, t11);
    _mm512_store_pd(X+14*s+8, t12);
    _mm512_store_pd(X+23*s, t15);
    _mm512_store_pd(X+23*s+8, t16);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+384);
    __m512d i0 = _mm512_load_pd(SCR+392);
    __m512d r1 = _mm512_load_pd(SCR+400);
    __m512d i1 = _mm512_load_pd(SCR+408);
    __m512d r2 = _mm512_load_pd(SCR+416);
    __m512d i2 = _mm512_load_pd(SCR+424);
    __m512d r3 = _mm512_load_pd(SCR+432);
    __m512d i3 = _mm512_load_pd(SCR+440);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+24*s, t9);
    _mm512_store_pd(X+24*s+8, t10);
    _mm512_store_pd(X+33*s, t13);
    _mm512_store_pd(X+33*s+8, t14);
    _mm512_store_pd(X+6*s, t11);
    _mm512_store_pd(X+6*s+8, t12);
    _mm512_store_pd(X+15*s, t15);
    _mm512_store_pd(X+15*s+8, t16);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+448);
    __m512d i0 = _mm512_load_pd(SCR+456);
    __m512d r1 = _mm512_load_pd(SCR+464);
    __m512d i1 = _mm512_load_pd(SCR+472);
    __m512d r2 = _mm512_load_pd(SCR+480);
    __m512d i2 = _mm512_load_pd(SCR+488);
    __m512d r3 = _mm512_load_pd(SCR+496);
    __m512d i3 = _mm512_load_pd(SCR+504);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+16*s, t9);
    _mm512_store_pd(X+16*s+8, t10);
    _mm512_store_pd(X+25*s, t13);
    _mm512_store_pd(X+25*s+8, t14);
    _mm512_store_pd(X+34*s, t11);
    _mm512_store_pd(X+34*s+8, t12);
    _mm512_store_pd(X+7*s, t15);
    _mm512_store_pd(X+7*s+8, t16);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+512);
    __m512d i0 = _mm512_load_pd(SCR+520);
    __m512d r1 = _mm512_load_pd(SCR+528);
    __m512d i1 = _mm512_load_pd(SCR+536);
    __m512d r2 = _mm512_load_pd(SCR+544);
    __m512d i2 = _mm512_load_pd(SCR+552);
    __m512d r3 = _mm512_load_pd(SCR+560);
    __m512d i3 = _mm512_load_pd(SCR+568);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    _mm512_store_pd(X+8*s, t9);
    _mm512_store_pd(X+8*s+8, t10);
    _mm512_store_pd(X+17*s, t13);
    _mm512_store_pd(X+17*s+8, t14);
    _mm512_store_pd(X+26*s, t11);
    _mm512_store_pd(X+26*s+8, t12);
    _mm512_store_pd(X+35*s, t15);
    _mm512_store_pd(X+35*s+8, t16);
    }
}
static void dft36m(double* restrict X, long es, const double* restrict C){
    const long s = es*16;
    double SCR[576] ALIGN64;
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+0*s);
    __m512d i0 = _mm512_load_pd(X+0*s+8);
    __m512d r1 = _mm512_load_pd(X+4*s);
    __m512d i1 = _mm512_load_pd(X+4*s+8);
    __m512d r2 = _mm512_load_pd(X+8*s);
    __m512d i2 = _mm512_load_pd(X+8*s+8);
    __m512d r3 = _mm512_load_pd(X+12*s);
    __m512d i3 = _mm512_load_pd(X+12*s+8);
    __m512d r4 = _mm512_load_pd(X+16*s);
    __m512d i4 = _mm512_load_pd(X+16*s+8);
    __m512d r5 = _mm512_load_pd(X+20*s);
    __m512d i5 = _mm512_load_pd(X+20*s+8);
    __m512d r6 = _mm512_load_pd(X+24*s);
    __m512d i6 = _mm512_load_pd(X+24*s+8);
    __m512d r7 = _mm512_load_pd(X+28*s);
    __m512d i7 = _mm512_load_pd(X+28*s+8);
    __m512d r8 = _mm512_load_pd(X+32*s);
    __m512d i8 = _mm512_load_pd(X+32*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+0, t23);
    _mm512_store_pd(SCR+8, t24);
    _mm512_store_pd(SCR+64, t41);
    _mm512_store_pd(SCR+72, t42);
    _mm512_store_pd(SCR+128, t61);
    _mm512_store_pd(SCR+136, t62);
    _mm512_store_pd(SCR+192, t81);
    _mm512_store_pd(SCR+200, t82);
    _mm512_store_pd(SCR+256, t101);
    _mm512_store_pd(SCR+264, t102);
    _mm512_store_pd(SCR+320, t103);
    _mm512_store_pd(SCR+328, t104);
    _mm512_store_pd(SCR+384, t83);
    _mm512_store_pd(SCR+392, t84);
    _mm512_store_pd(SCR+448, t63);
    _mm512_store_pd(SCR+456, t64);
    _mm512_store_pd(SCR+512, t43);
    _mm512_store_pd(SCR+520, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+9*s);
    __m512d i0 = _mm512_load_pd(X+9*s+8);
    __m512d r1 = _mm512_load_pd(X+13*s);
    __m512d i1 = _mm512_load_pd(X+13*s+8);
    __m512d r2 = _mm512_load_pd(X+17*s);
    __m512d i2 = _mm512_load_pd(X+17*s+8);
    __m512d r3 = _mm512_load_pd(X+21*s);
    __m512d i3 = _mm512_load_pd(X+21*s+8);
    __m512d r4 = _mm512_load_pd(X+25*s);
    __m512d i4 = _mm512_load_pd(X+25*s+8);
    __m512d r5 = _mm512_load_pd(X+29*s);
    __m512d i5 = _mm512_load_pd(X+29*s+8);
    __m512d r6 = _mm512_load_pd(X+33*s);
    __m512d i6 = _mm512_load_pd(X+33*s+8);
    __m512d r7 = _mm512_load_pd(X+1*s);
    __m512d i7 = _mm512_load_pd(X+1*s+8);
    __m512d r8 = _mm512_load_pd(X+5*s);
    __m512d i8 = _mm512_load_pd(X+5*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+16, t23);
    _mm512_store_pd(SCR+24, t24);
    _mm512_store_pd(SCR+80, t41);
    _mm512_store_pd(SCR+88, t42);
    _mm512_store_pd(SCR+144, t61);
    _mm512_store_pd(SCR+152, t62);
    _mm512_store_pd(SCR+208, t81);
    _mm512_store_pd(SCR+216, t82);
    _mm512_store_pd(SCR+272, t101);
    _mm512_store_pd(SCR+280, t102);
    _mm512_store_pd(SCR+336, t103);
    _mm512_store_pd(SCR+344, t104);
    _mm512_store_pd(SCR+400, t83);
    _mm512_store_pd(SCR+408, t84);
    _mm512_store_pd(SCR+464, t63);
    _mm512_store_pd(SCR+472, t64);
    _mm512_store_pd(SCR+528, t43);
    _mm512_store_pd(SCR+536, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+18*s);
    __m512d i0 = _mm512_load_pd(X+18*s+8);
    __m512d r1 = _mm512_load_pd(X+22*s);
    __m512d i1 = _mm512_load_pd(X+22*s+8);
    __m512d r2 = _mm512_load_pd(X+26*s);
    __m512d i2 = _mm512_load_pd(X+26*s+8);
    __m512d r3 = _mm512_load_pd(X+30*s);
    __m512d i3 = _mm512_load_pd(X+30*s+8);
    __m512d r4 = _mm512_load_pd(X+34*s);
    __m512d i4 = _mm512_load_pd(X+34*s+8);
    __m512d r5 = _mm512_load_pd(X+2*s);
    __m512d i5 = _mm512_load_pd(X+2*s+8);
    __m512d r6 = _mm512_load_pd(X+6*s);
    __m512d i6 = _mm512_load_pd(X+6*s+8);
    __m512d r7 = _mm512_load_pd(X+10*s);
    __m512d i7 = _mm512_load_pd(X+10*s+8);
    __m512d r8 = _mm512_load_pd(X+14*s);
    __m512d i8 = _mm512_load_pd(X+14*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+32, t23);
    _mm512_store_pd(SCR+40, t24);
    _mm512_store_pd(SCR+96, t41);
    _mm512_store_pd(SCR+104, t42);
    _mm512_store_pd(SCR+160, t61);
    _mm512_store_pd(SCR+168, t62);
    _mm512_store_pd(SCR+224, t81);
    _mm512_store_pd(SCR+232, t82);
    _mm512_store_pd(SCR+288, t101);
    _mm512_store_pd(SCR+296, t102);
    _mm512_store_pd(SCR+352, t103);
    _mm512_store_pd(SCR+360, t104);
    _mm512_store_pd(SCR+416, t83);
    _mm512_store_pd(SCR+424, t84);
    _mm512_store_pd(SCR+480, t63);
    _mm512_store_pd(SCR+488, t64);
    _mm512_store_pd(SCR+544, t43);
    _mm512_store_pd(SCR+552, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+27*s);
    __m512d i0 = _mm512_load_pd(X+27*s+8);
    __m512d r1 = _mm512_load_pd(X+31*s);
    __m512d i1 = _mm512_load_pd(X+31*s+8);
    __m512d r2 = _mm512_load_pd(X+35*s);
    __m512d i2 = _mm512_load_pd(X+35*s+8);
    __m512d r3 = _mm512_load_pd(X+3*s);
    __m512d i3 = _mm512_load_pd(X+3*s+8);
    __m512d r4 = _mm512_load_pd(X+7*s);
    __m512d i4 = _mm512_load_pd(X+7*s+8);
    __m512d r5 = _mm512_load_pd(X+11*s);
    __m512d i5 = _mm512_load_pd(X+11*s+8);
    __m512d r6 = _mm512_load_pd(X+15*s);
    __m512d i6 = _mm512_load_pd(X+15*s+8);
    __m512d r7 = _mm512_load_pd(X+19*s);
    __m512d i7 = _mm512_load_pd(X+19*s+8);
    __m512d r8 = _mm512_load_pd(X+23*s);
    __m512d i8 = _mm512_load_pd(X+23*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+48, t23);
    _mm512_store_pd(SCR+56, t24);
    _mm512_store_pd(SCR+112, t41);
    _mm512_store_pd(SCR+120, t42);
    _mm512_store_pd(SCR+176, t61);
    _mm512_store_pd(SCR+184, t62);
    _mm512_store_pd(SCR+240, t81);
    _mm512_store_pd(SCR+248, t82);
    _mm512_store_pd(SCR+304, t101);
    _mm512_store_pd(SCR+312, t102);
    _mm512_store_pd(SCR+368, t103);
    _mm512_store_pd(SCR+376, t104);
    _mm512_store_pd(SCR+432, t83);
    _mm512_store_pd(SCR+440, t84);
    _mm512_store_pd(SCR+496, t63);
    _mm512_store_pd(SCR+504, t64);
    _mm512_store_pd(SCR+560, t43);
    _mm512_store_pd(SCR+568, t44);
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+0);
    __m512d i0 = _mm512_load_pd(SCR+8);
    __m512d r1 = _mm512_load_pd(SCR+16);
    __m512d i1 = _mm512_load_pd(SCR+24);
    __m512d r2 = _mm512_load_pd(SCR+32);
    __m512d i2 = _mm512_load_pd(SCR+40);
    __m512d r3 = _mm512_load_pd(SCR+48);
    __m512d i3 = _mm512_load_pd(SCR+56);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    { __m512d zr = _mm512_add_pd(t9, _mm512_load_pd(C+0*16));
      __m512d zi = _mm512_add_pd(t10, _mm512_load_pd(C+0*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+0*s, zr); _mm512_store_pd(X+0*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t13, _mm512_load_pd(C+9*16));
      __m512d zi = _mm512_add_pd(t14, _mm512_load_pd(C+9*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+9*s, zr); _mm512_store_pd(X+9*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+18*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+18*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+18*s, zr); _mm512_store_pd(X+18*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t15, _mm512_load_pd(C+27*16));
      __m512d zi = _mm512_add_pd(t16, _mm512_load_pd(C+27*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+27*s, zr); _mm512_store_pd(X+27*s+8, zi); }
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+64);
    __m512d i0 = _mm512_load_pd(SCR+72);
    __m512d r1 = _mm512_load_pd(SCR+80);
    __m512d i1 = _mm512_load_pd(SCR+88);
    __m512d r2 = _mm512_load_pd(SCR+96);
    __m512d i2 = _mm512_load_pd(SCR+104);
    __m512d r3 = _mm512_load_pd(SCR+112);
    __m512d i3 = _mm512_load_pd(SCR+120);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    { __m512d zr = _mm512_add_pd(t9, _mm512_load_pd(C+28*16));
      __m512d zi = _mm512_add_pd(t10, _mm512_load_pd(C+28*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+28*s, zr); _mm512_store_pd(X+28*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t13, _mm512_load_pd(C+1*16));
      __m512d zi = _mm512_add_pd(t14, _mm512_load_pd(C+1*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+1*s, zr); _mm512_store_pd(X+1*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+10*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+10*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+10*s, zr); _mm512_store_pd(X+10*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t15, _mm512_load_pd(C+19*16));
      __m512d zi = _mm512_add_pd(t16, _mm512_load_pd(C+19*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+19*s, zr); _mm512_store_pd(X+19*s+8, zi); }
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+128);
    __m512d i0 = _mm512_load_pd(SCR+136);
    __m512d r1 = _mm512_load_pd(SCR+144);
    __m512d i1 = _mm512_load_pd(SCR+152);
    __m512d r2 = _mm512_load_pd(SCR+160);
    __m512d i2 = _mm512_load_pd(SCR+168);
    __m512d r3 = _mm512_load_pd(SCR+176);
    __m512d i3 = _mm512_load_pd(SCR+184);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    { __m512d zr = _mm512_add_pd(t9, _mm512_load_pd(C+20*16));
      __m512d zi = _mm512_add_pd(t10, _mm512_load_pd(C+20*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+20*s, zr); _mm512_store_pd(X+20*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t13, _mm512_load_pd(C+29*16));
      __m512d zi = _mm512_add_pd(t14, _mm512_load_pd(C+29*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+29*s, zr); _mm512_store_pd(X+29*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+2*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+2*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+2*s, zr); _mm512_store_pd(X+2*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t15, _mm512_load_pd(C+11*16));
      __m512d zi = _mm512_add_pd(t16, _mm512_load_pd(C+11*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+11*s, zr); _mm512_store_pd(X+11*s+8, zi); }
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+192);
    __m512d i0 = _mm512_load_pd(SCR+200);
    __m512d r1 = _mm512_load_pd(SCR+208);
    __m512d i1 = _mm512_load_pd(SCR+216);
    __m512d r2 = _mm512_load_pd(SCR+224);
    __m512d i2 = _mm512_load_pd(SCR+232);
    __m512d r3 = _mm512_load_pd(SCR+240);
    __m512d i3 = _mm512_load_pd(SCR+248);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    { __m512d zr = _mm512_add_pd(t9, _mm512_load_pd(C+12*16));
      __m512d zi = _mm512_add_pd(t10, _mm512_load_pd(C+12*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+12*s, zr); _mm512_store_pd(X+12*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t13, _mm512_load_pd(C+21*16));
      __m512d zi = _mm512_add_pd(t14, _mm512_load_pd(C+21*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+21*s, zr); _mm512_store_pd(X+21*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+30*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+30*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+30*s, zr); _mm512_store_pd(X+30*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t15, _mm512_load_pd(C+3*16));
      __m512d zi = _mm512_add_pd(t16, _mm512_load_pd(C+3*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+3*s, zr); _mm512_store_pd(X+3*s+8, zi); }
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+256);
    __m512d i0 = _mm512_load_pd(SCR+264);
    __m512d r1 = _mm512_load_pd(SCR+272);
    __m512d i1 = _mm512_load_pd(SCR+280);
    __m512d r2 = _mm512_load_pd(SCR+288);
    __m512d i2 = _mm512_load_pd(SCR+296);
    __m512d r3 = _mm512_load_pd(SCR+304);
    __m512d i3 = _mm512_load_pd(SCR+312);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    { __m512d zr = _mm512_add_pd(t9, _mm512_load_pd(C+4*16));
      __m512d zi = _mm512_add_pd(t10, _mm512_load_pd(C+4*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+4*s, zr); _mm512_store_pd(X+4*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t13, _mm512_load_pd(C+13*16));
      __m512d zi = _mm512_add_pd(t14, _mm512_load_pd(C+13*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+13*s, zr); _mm512_store_pd(X+13*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+22*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+22*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+22*s, zr); _mm512_store_pd(X+22*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t15, _mm512_load_pd(C+31*16));
      __m512d zi = _mm512_add_pd(t16, _mm512_load_pd(C+31*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+31*s, zr); _mm512_store_pd(X+31*s+8, zi); }
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+320);
    __m512d i0 = _mm512_load_pd(SCR+328);
    __m512d r1 = _mm512_load_pd(SCR+336);
    __m512d i1 = _mm512_load_pd(SCR+344);
    __m512d r2 = _mm512_load_pd(SCR+352);
    __m512d i2 = _mm512_load_pd(SCR+360);
    __m512d r3 = _mm512_load_pd(SCR+368);
    __m512d i3 = _mm512_load_pd(SCR+376);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    { __m512d zr = _mm512_add_pd(t9, _mm512_load_pd(C+32*16));
      __m512d zi = _mm512_add_pd(t10, _mm512_load_pd(C+32*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+32*s, zr); _mm512_store_pd(X+32*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t13, _mm512_load_pd(C+5*16));
      __m512d zi = _mm512_add_pd(t14, _mm512_load_pd(C+5*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+5*s, zr); _mm512_store_pd(X+5*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+14*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+14*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+14*s, zr); _mm512_store_pd(X+14*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t15, _mm512_load_pd(C+23*16));
      __m512d zi = _mm512_add_pd(t16, _mm512_load_pd(C+23*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+23*s, zr); _mm512_store_pd(X+23*s+8, zi); }
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+384);
    __m512d i0 = _mm512_load_pd(SCR+392);
    __m512d r1 = _mm512_load_pd(SCR+400);
    __m512d i1 = _mm512_load_pd(SCR+408);
    __m512d r2 = _mm512_load_pd(SCR+416);
    __m512d i2 = _mm512_load_pd(SCR+424);
    __m512d r3 = _mm512_load_pd(SCR+432);
    __m512d i3 = _mm512_load_pd(SCR+440);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    { __m512d zr = _mm512_add_pd(t9, _mm512_load_pd(C+24*16));
      __m512d zi = _mm512_add_pd(t10, _mm512_load_pd(C+24*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+24*s, zr); _mm512_store_pd(X+24*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t13, _mm512_load_pd(C+33*16));
      __m512d zi = _mm512_add_pd(t14, _mm512_load_pd(C+33*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+33*s, zr); _mm512_store_pd(X+33*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+6*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+6*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+6*s, zr); _mm512_store_pd(X+6*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t15, _mm512_load_pd(C+15*16));
      __m512d zi = _mm512_add_pd(t16, _mm512_load_pd(C+15*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+15*s, zr); _mm512_store_pd(X+15*s+8, zi); }
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+448);
    __m512d i0 = _mm512_load_pd(SCR+456);
    __m512d r1 = _mm512_load_pd(SCR+464);
    __m512d i1 = _mm512_load_pd(SCR+472);
    __m512d r2 = _mm512_load_pd(SCR+480);
    __m512d i2 = _mm512_load_pd(SCR+488);
    __m512d r3 = _mm512_load_pd(SCR+496);
    __m512d i3 = _mm512_load_pd(SCR+504);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    { __m512d zr = _mm512_add_pd(t9, _mm512_load_pd(C+16*16));
      __m512d zi = _mm512_add_pd(t10, _mm512_load_pd(C+16*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+16*s, zr); _mm512_store_pd(X+16*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t13, _mm512_load_pd(C+25*16));
      __m512d zi = _mm512_add_pd(t14, _mm512_load_pd(C+25*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+25*s, zr); _mm512_store_pd(X+25*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+34*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+34*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+34*s, zr); _mm512_store_pd(X+34*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t15, _mm512_load_pd(C+7*16));
      __m512d zi = _mm512_add_pd(t16, _mm512_load_pd(C+7*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+7*s, zr); _mm512_store_pd(X+7*s+8, zi); }
    }
    {
    __m512d r0 = _mm512_load_pd(SCR+512);
    __m512d i0 = _mm512_load_pd(SCR+520);
    __m512d r1 = _mm512_load_pd(SCR+528);
    __m512d i1 = _mm512_load_pd(SCR+536);
    __m512d r2 = _mm512_load_pd(SCR+544);
    __m512d i2 = _mm512_load_pd(SCR+552);
    __m512d r3 = _mm512_load_pd(SCR+560);
    __m512d i3 = _mm512_load_pd(SCR+568);
    __m512d t1 = _mm512_add_pd(r0, r2);
    __m512d t2 = _mm512_add_pd(i0, i2);
    __m512d t3 = _mm512_sub_pd(r0, r2);
    __m512d t4 = _mm512_sub_pd(i0, i2);
    __m512d t5 = _mm512_add_pd(r1, r3);
    __m512d t6 = _mm512_add_pd(i1, i3);
    __m512d t7 = _mm512_sub_pd(r1, r3);
    __m512d t8 = _mm512_sub_pd(i1, i3);
    __m512d t9 = _mm512_add_pd(t1, t5);
    __m512d t10 = _mm512_add_pd(t2, t6);
    __m512d t11 = _mm512_sub_pd(t1, t5);
    __m512d t12 = _mm512_sub_pd(t2, t6);
    __m512d t13 = _mm512_add_pd(t3, t8);
    __m512d t14 = _mm512_sub_pd(t4, t7);
    __m512d t15 = _mm512_sub_pd(t3, t8);
    __m512d t16 = _mm512_add_pd(t4, t7);
    { __m512d zr = _mm512_add_pd(t9, _mm512_load_pd(C+8*16));
      __m512d zi = _mm512_add_pd(t10, _mm512_load_pd(C+8*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+8*s, zr); _mm512_store_pd(X+8*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t13, _mm512_load_pd(C+17*16));
      __m512d zi = _mm512_add_pd(t14, _mm512_load_pd(C+17*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+17*s, zr); _mm512_store_pd(X+17*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+26*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+26*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+26*s, zr); _mm512_store_pd(X+26*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t15, _mm512_load_pd(C+35*16));
      __m512d zi = _mm512_add_pd(t16, _mm512_load_pd(C+35*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+35*s, zr); _mm512_store_pd(X+35*s+8, zi); }
    }
}
#define PFIN_36 0

static void __attribute__((noinline)) dft36_one(double* X, long es){ dft36(X, es); }
#if PFIN_36
static void __attribute__((noinline)) dft36_onesq(double* X){ dft36pfz(X, 1); }
static void __attribute__((noinline)) dft36_onem(double* X, long es, const double* Ct){ dft36m(X, es, Ct); }
#elif 0
static void __attribute__((noinline)) dft36_onem_unused(double* X, long es, const double* Ct){ dft36mpf(X, es, Ct); }
#else
#if 1
static void __attribute__((noinline)) dft36_onesq(double* X){ dft36sq(X, 1); }
#else
static void __attribute__((noinline)) dft36_onesq(double* X){ dft36(X, 1); }
#endif
static void __attribute__((noinline)) dft36_onem(double* X, long es, const double* Ct){ dft36m(X, es, Ct); }
#endif
static void dft36_sweep_zy(double* restrict X){
#if PFCOMP
    const long PB = (1296*2 + 36 - 1)/36;   /* 128B-blocks of next plane per y-codelet */
    for(long x=0; x<36; x++){
        double* P = X + x*1297*16;
        const char* nxt = (const char*)(P + 1297*16);
        for(long y=0; y<36; y++) dft36_one(P + y*36*16, 1);
        for(long z=0; z<36; z++){
            if(x+1 < 36){
                const char* q = nxt + z*PB*128;
                for(long l=0; l<PB; l++) _mm_prefetch(q + l*128, _MM_HINT_T0);
            }
            dft36_one(P + z*16, 36);
        }
    }
#else
    for(long x=0; x<36; x++){
        double* P = X + x*1297*16;
        for(long y=0; y<36; y++) dft36_onesq(P + y*36*16);
        for(long z=0; z<36; z++) dft36_one(P + z*16, 36);
    }
#endif
}
static void dft36_sweep_x_map(double* restrict X, const double* restrict Ct){
    for(long p=0; p<1296; p++) dft36_onem(X + p*16, 1297, Ct + p*36*16);
}
static void dft36_sweep_x_plain(double* restrict X){
    for(long p=0; p<1296; p++) dft36_one(X + p*16, 1297);
}
static void dft36_sweep_zy_ms(double* restrict X, const double* restrict C){
    for(long x=0; x<36; x++){
        double* P = X + x*1297*16;
        for(long y=0; y<36; y++) dft36_onesq(P + y*36*16);
        for(long z=0; z<36; z++) dft36_one(P + z*16, 36);
        if(x) mapslab(X + (x-1)*1297*16, C + (x-1)*1297*16, 1296);
    }
    mapslab(X + (36-1)*1297*16, C + (36-1)*1297*16, 1296);
}
/* plane-wise ingest/output (padded plane stride 1297) */
static void ingest_36(const double* const* src, double* G){
    for(long x=0; x<36; x++){
        const long base = x*1296;
        double* Gp = G + x*1297*16;
        for(long e=0; e<1296; e+=4){
            __m512d r0=_mm512_loadu_pd(src[0]+2*(base+e)), r1=_mm512_loadu_pd(src[1]+2*(base+e));
            __m512d r2=_mm512_loadu_pd(src[2]+2*(base+e)), r3=_mm512_loadu_pd(src[3]+2*(base+e));
            __m512d r4=_mm512_loadu_pd(src[4]+2*(base+e)), r5=_mm512_loadu_pd(src[5]+2*(base+e));
            __m512d r6=_mm512_loadu_pd(src[6]+2*(base+e)), r7=_mm512_loadu_pd(src[7]+2*(base+e));
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            _mm512_store_pd(Gp+e*16,    o0); _mm512_store_pd(Gp+e*16+8,  o1);
            _mm512_store_pd(Gp+e*16+16, o2); _mm512_store_pd(Gp+e*16+24, o3);
            _mm512_store_pd(Gp+e*16+32, o4); _mm512_store_pd(Gp+e*16+40, o5);
            _mm512_store_pd(Gp+e*16+48, o6); _mm512_store_pd(Gp+e*16+56, o7);
        }
#if 0 > 0
        {
            const long e = 1296;
            const __mmask8 mk = (__mmask8)((1u<<(2*0))-1u);
            __m512d r0=_mm512_maskz_loadu_pd(mk, src[0]+2*(base+e)), r1=_mm512_maskz_loadu_pd(mk, src[1]+2*(base+e));
            __m512d r2=_mm512_maskz_loadu_pd(mk, src[2]+2*(base+e)), r3=_mm512_maskz_loadu_pd(mk, src[3]+2*(base+e));
            __m512d r4=_mm512_maskz_loadu_pd(mk, src[4]+2*(base+e)), r5=_mm512_maskz_loadu_pd(mk, src[5]+2*(base+e));
            __m512d r6=_mm512_maskz_loadu_pd(mk, src[6]+2*(base+e)), r7=_mm512_maskz_loadu_pd(mk, src[7]+2*(base+e));
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d A[8]; A[0]=o0;A[1]=o1;A[2]=o2;A[3]=o3;A[4]=o4;A[5]=o5;A[6]=o6;A[7]=o7;
            for(int q=0;q<2*0;q++) _mm512_store_pd(Gp+e*16+q*8, A[q]);
        }
#endif
    }
}
static void output_36(const double* G, double* const* dst, int nv){
    for(long x=0; x<36; x++){
        const long base = x*1296;
        const double* Gp = G + x*1297*16;
        for(long e=0; e<1296; e+=4){
            __m512d i0=_mm512_load_pd(Gp+e*16),    i1=_mm512_load_pd(Gp+e*16+8);
            __m512d i2=_mm512_load_pd(Gp+e*16+16), i3=_mm512_load_pd(Gp+e*16+24);
            __m512d i4=_mm512_load_pd(Gp+e*16+32), i5=_mm512_load_pd(Gp+e*16+40);
            __m512d i6=_mm512_load_pd(Gp+e*16+48), i7=_mm512_load_pd(Gp+e*16+56);
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(i0,i1,i2,i3,i4,i5,i6,i7,o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
            for(int v=0; v<nv; v++) _mm512_storeu_pd(dst[v]+2*(base+e), *O[v]);
        }
#if 0 > 0
        {
            const long e = 1296;
            const __mmask8 mk = (__mmask8)((1u<<(2*0))-1u);
            __m512d A[8];
            for(int q=0;q<2*0;q++) A[q] = _mm512_load_pd(Gp+e*16+q*8);
            for(int q=2*0;q<8;q++) A[q] = _mm512_setzero_pd();
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
            for(int v=0; v<nv; v++) _mm512_mask_storeu_pd(dst[v]+2*(base+e), mk, *O[v]);
        }
#endif
    }
}


static double* Xg_36 = 0;
static double* Cg_36 = 0;
void hot_36(long n){
    if(!Xg_36){ Xg_36 = alloc_huge_st(36*1297*16*8); Cg_36 = alloc_huge_st(46656*16*8); }
    for(long i=0;i<36*1297*16;i++) Xg_36[i] = 0.5 + 1e-6*(i%97);
    for(long r=0;r<n;r++){
        double* P = Xg_36;
        for(long y=0; y<36; y++) dft36_one(P + y*36*16, 1);
        for(long z=0; z<36; z++) dft36_one(P + z*16, 36);
        if((r&1)==1) for(long i=0;i<1296*16;i++) Xg_36[i] = 0.5 + 1e-6*(i%97);
    }
}
void hot2_36(long which){
    if(!Xg_36){ Xg_36 = alloc_huge_st((46656+64*36)*16*8); Cg_36 = alloc_huge_st(46656*16*8); }
    double* P = Xg_36;
    if(which==99){ for(long i=0;i<1296*16;i++) P[i] = 0.5 + 1e-6*(i%97); return; }
    if(which==0 || which==2) for(long y=0; y<36; y++) dft36_one(P + y*36*16, 1);
    if(which==1 || which==2) for(long z=0; z<36; z++) dft36_one(P + z*16, 36);
}
void bsweep_36(long which, long n){
    if(!Xg_36){ Xg_36 = alloc_huge_st(36*1297*16*8); Cg_36 = alloc_huge_st(46656*16*8); }
    for(long i=0;i<36*1297*16;i++) Xg_36[i] = 0.5 + 1e-6*(i%97);
    for(long i=0;i<46656*16;i++) Cg_36[i] = 0.01;
    for(long r=0;r<n;r++){
        if(which==0) dft36_sweep_zy(Xg_36);
        else if(which==2) dft36_sweep_x_map(Xg_36, Cg_36);
        if((r&3)==3) for(long i=0;i<36*1297*16;i+=997) Xg_36[i] = 0.5;
    }
}
void diag_36(long which, long n){
    if(!Xg_36){ Xg_36 = alloc_huge_st(36*1297*16*8); Cg_36 = alloc_huge_st(36*1297*16*8); }
    for(long i=0;i<36*1297*16;i++){ Xg_36[i] = 0.5 + 1e-6*(i%97); Cg_36[i] = 0.01; }
    for(long r=0;r<n;r++){
        if(which==0){ for(long x=0;x<36;x++) mapslab(Xg_36 + x*1297*16, Cg_36 + x*1297*16, 1296); }
        else if(which==1) dft36_sweep_zy(Xg_36);
        else dft36_sweep_x_plain(Xg_36);
        if((r&1)==1) for(long i=0;i<36*1297*16;i+=997) Xg_36[i] = 0.5;
    }
}
void run_36(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    const long NE = 46656;
    if(!Xg_36){ Xg_36 = alloc_huge_st(36*1297*16*8); Cg_36 = alloc_huge_st(36*1297*16*8); }
    double* X = Xg_36; double* Ct = Cg_36;
    for(long g0=0; g0<B; g0+=8){
        int nv = (int)((B - g0) < 8 ? (B - g0) : 8);
        const double* src[8]; const double* csrc[8];
        double* d1[8]; double* dm[8];
        for(int v=0; v<8; v++){
            int vv = v < nv ? v : 0;
            src[v] = x0 + (g0+vv)*2*NE; csrc[v] = c + (g0+vv)*2*NE;
            if(v<nv){ d1[v] = out1 + (g0+v)*2*NE; dm[v] = outm + (g0+v)*2*NE; }
        }
#if XFIRST_FLAG
        ingest_36(csrc, Ct);   /* c in padded group layout */
        ingest_36(src, X);
        for(long t=0; t<m; t++){
            dft36_sweep_x_plain(X);
            dft36_sweep_zy_ms(X, Ct);
            if(t==0 && m>1) output_36(X, d1, nv);
        }
#else
        ingest_36(csrc, X);
        for(long p=0; p<1296; p++)
            for(long k=0; k<36; k++){
                _mm512_store_pd(Ct + (p*36+k)*16,     _mm512_load_pd(X + (k*1297+p)*16));
                _mm512_store_pd(Ct + (p*36+k)*16 + 8, _mm512_load_pd(X + (k*1297+p)*16 + 8));
            }
        ingest_36(src, X);
        for(long t=0; t<m; t++){
            dft36_sweep_zy(X);
            dft36_sweep_x_map(X, Ct);
            if(t==0 && m>1) output_36(X, d1, nv);
        }
#endif
        output_36(X, dm, nv);
        if(m==1) output_36(X, d1, nv);
    }
}

static void dft45(double* restrict X, long es){
    const long s = es*16;
    double SCR[720] ALIGN64;
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+0*s);
    __m512d i0 = _mm512_load_pd(X+0*s+8);
    __m512d r1 = _mm512_load_pd(X+5*s);
    __m512d i1 = _mm512_load_pd(X+5*s+8);
    __m512d r2 = _mm512_load_pd(X+10*s);
    __m512d i2 = _mm512_load_pd(X+10*s+8);
    __m512d r3 = _mm512_load_pd(X+15*s);
    __m512d i3 = _mm512_load_pd(X+15*s+8);
    __m512d r4 = _mm512_load_pd(X+20*s);
    __m512d i4 = _mm512_load_pd(X+20*s+8);
    __m512d r5 = _mm512_load_pd(X+25*s);
    __m512d i5 = _mm512_load_pd(X+25*s+8);
    __m512d r6 = _mm512_load_pd(X+30*s);
    __m512d i6 = _mm512_load_pd(X+30*s+8);
    __m512d r7 = _mm512_load_pd(X+35*s);
    __m512d i7 = _mm512_load_pd(X+35*s+8);
    __m512d r8 = _mm512_load_pd(X+40*s);
    __m512d i8 = _mm512_load_pd(X+40*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+0, t23);
    _mm512_store_pd(SCR+8, t24);
    _mm512_store_pd(SCR+80, t41);
    _mm512_store_pd(SCR+88, t42);
    _mm512_store_pd(SCR+160, t61);
    _mm512_store_pd(SCR+168, t62);
    _mm512_store_pd(SCR+240, t81);
    _mm512_store_pd(SCR+248, t82);
    _mm512_store_pd(SCR+320, t101);
    _mm512_store_pd(SCR+328, t102);
    _mm512_store_pd(SCR+400, t103);
    _mm512_store_pd(SCR+408, t104);
    _mm512_store_pd(SCR+480, t83);
    _mm512_store_pd(SCR+488, t84);
    _mm512_store_pd(SCR+560, t63);
    _mm512_store_pd(SCR+568, t64);
    _mm512_store_pd(SCR+640, t43);
    _mm512_store_pd(SCR+648, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+9*s);
    __m512d i0 = _mm512_load_pd(X+9*s+8);
    __m512d r1 = _mm512_load_pd(X+14*s);
    __m512d i1 = _mm512_load_pd(X+14*s+8);
    __m512d r2 = _mm512_load_pd(X+19*s);
    __m512d i2 = _mm512_load_pd(X+19*s+8);
    __m512d r3 = _mm512_load_pd(X+24*s);
    __m512d i3 = _mm512_load_pd(X+24*s+8);
    __m512d r4 = _mm512_load_pd(X+29*s);
    __m512d i4 = _mm512_load_pd(X+29*s+8);
    __m512d r5 = _mm512_load_pd(X+34*s);
    __m512d i5 = _mm512_load_pd(X+34*s+8);
    __m512d r6 = _mm512_load_pd(X+39*s);
    __m512d i6 = _mm512_load_pd(X+39*s+8);
    __m512d r7 = _mm512_load_pd(X+44*s);
    __m512d i7 = _mm512_load_pd(X+44*s+8);
    __m512d r8 = _mm512_load_pd(X+4*s);
    __m512d i8 = _mm512_load_pd(X+4*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+16, t23);
    _mm512_store_pd(SCR+24, t24);
    _mm512_store_pd(SCR+96, t41);
    _mm512_store_pd(SCR+104, t42);
    _mm512_store_pd(SCR+176, t61);
    _mm512_store_pd(SCR+184, t62);
    _mm512_store_pd(SCR+256, t81);
    _mm512_store_pd(SCR+264, t82);
    _mm512_store_pd(SCR+336, t101);
    _mm512_store_pd(SCR+344, t102);
    _mm512_store_pd(SCR+416, t103);
    _mm512_store_pd(SCR+424, t104);
    _mm512_store_pd(SCR+496, t83);
    _mm512_store_pd(SCR+504, t84);
    _mm512_store_pd(SCR+576, t63);
    _mm512_store_pd(SCR+584, t64);
    _mm512_store_pd(SCR+656, t43);
    _mm512_store_pd(SCR+664, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+18*s);
    __m512d i0 = _mm512_load_pd(X+18*s+8);
    __m512d r1 = _mm512_load_pd(X+23*s);
    __m512d i1 = _mm512_load_pd(X+23*s+8);
    __m512d r2 = _mm512_load_pd(X+28*s);
    __m512d i2 = _mm512_load_pd(X+28*s+8);
    __m512d r3 = _mm512_load_pd(X+33*s);
    __m512d i3 = _mm512_load_pd(X+33*s+8);
    __m512d r4 = _mm512_load_pd(X+38*s);
    __m512d i4 = _mm512_load_pd(X+38*s+8);
    __m512d r5 = _mm512_load_pd(X+43*s);
    __m512d i5 = _mm512_load_pd(X+43*s+8);
    __m512d r6 = _mm512_load_pd(X+3*s);
    __m512d i6 = _mm512_load_pd(X+3*s+8);
    __m512d r7 = _mm512_load_pd(X+8*s);
    __m512d i7 = _mm512_load_pd(X+8*s+8);
    __m512d r8 = _mm512_load_pd(X+13*s);
    __m512d i8 = _mm512_load_pd(X+13*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+32, t23);
    _mm512_store_pd(SCR+40, t24);
    _mm512_store_pd(SCR+112, t41);
    _mm512_store_pd(SCR+120, t42);
    _mm512_store_pd(SCR+192, t61);
    _mm512_store_pd(SCR+200, t62);
    _mm512_store_pd(SCR+272, t81);
    _mm512_store_pd(SCR+280, t82);
    _mm512_store_pd(SCR+352, t101);
    _mm512_store_pd(SCR+360, t102);
    _mm512_store_pd(SCR+432, t103);
    _mm512_store_pd(SCR+440, t104);
    _mm512_store_pd(SCR+512, t83);
    _mm512_store_pd(SCR+520, t84);
    _mm512_store_pd(SCR+592, t63);
    _mm512_store_pd(SCR+600, t64);
    _mm512_store_pd(SCR+672, t43);
    _mm512_store_pd(SCR+680, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+27*s);
    __m512d i0 = _mm512_load_pd(X+27*s+8);
    __m512d r1 = _mm512_load_pd(X+32*s);
    __m512d i1 = _mm512_load_pd(X+32*s+8);
    __m512d r2 = _mm512_load_pd(X+37*s);
    __m512d i2 = _mm512_load_pd(X+37*s+8);
    __m512d r3 = _mm512_load_pd(X+42*s);
    __m512d i3 = _mm512_load_pd(X+42*s+8);
    __m512d r4 = _mm512_load_pd(X+2*s);
    __m512d i4 = _mm512_load_pd(X+2*s+8);
    __m512d r5 = _mm512_load_pd(X+7*s);
    __m512d i5 = _mm512_load_pd(X+7*s+8);
    __m512d r6 = _mm512_load_pd(X+12*s);
    __m512d i6 = _mm512_load_pd(X+12*s+8);
    __m512d r7 = _mm512_load_pd(X+17*s);
    __m512d i7 = _mm512_load_pd(X+17*s+8);
    __m512d r8 = _mm512_load_pd(X+22*s);
    __m512d i8 = _mm512_load_pd(X+22*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+48, t23);
    _mm512_store_pd(SCR+56, t24);
    _mm512_store_pd(SCR+128, t41);
    _mm512_store_pd(SCR+136, t42);
    _mm512_store_pd(SCR+208, t61);
    _mm512_store_pd(SCR+216, t62);
    _mm512_store_pd(SCR+288, t81);
    _mm512_store_pd(SCR+296, t82);
    _mm512_store_pd(SCR+368, t101);
    _mm512_store_pd(SCR+376, t102);
    _mm512_store_pd(SCR+448, t103);
    _mm512_store_pd(SCR+456, t104);
    _mm512_store_pd(SCR+528, t83);
    _mm512_store_pd(SCR+536, t84);
    _mm512_store_pd(SCR+608, t63);
    _mm512_store_pd(SCR+616, t64);
    _mm512_store_pd(SCR+688, t43);
    _mm512_store_pd(SCR+696, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+36*s);
    __m512d i0 = _mm512_load_pd(X+36*s+8);
    __m512d r1 = _mm512_load_pd(X+41*s);
    __m512d i1 = _mm512_load_pd(X+41*s+8);
    __m512d r2 = _mm512_load_pd(X+1*s);
    __m512d i2 = _mm512_load_pd(X+1*s+8);
    __m512d r3 = _mm512_load_pd(X+6*s);
    __m512d i3 = _mm512_load_pd(X+6*s+8);
    __m512d r4 = _mm512_load_pd(X+11*s);
    __m512d i4 = _mm512_load_pd(X+11*s+8);
    __m512d r5 = _mm512_load_pd(X+16*s);
    __m512d i5 = _mm512_load_pd(X+16*s+8);
    __m512d r6 = _mm512_load_pd(X+21*s);
    __m512d i6 = _mm512_load_pd(X+21*s+8);
    __m512d r7 = _mm512_load_pd(X+26*s);
    __m512d i7 = _mm512_load_pd(X+26*s+8);
    __m512d r8 = _mm512_load_pd(X+31*s);
    __m512d i8 = _mm512_load_pd(X+31*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+64, t23);
    _mm512_store_pd(SCR+72, t24);
    _mm512_store_pd(SCR+144, t41);
    _mm512_store_pd(SCR+152, t42);
    _mm512_store_pd(SCR+224, t61);
    _mm512_store_pd(SCR+232, t62);
    _mm512_store_pd(SCR+304, t81);
    _mm512_store_pd(SCR+312, t82);
    _mm512_store_pd(SCR+384, t101);
    _mm512_store_pd(SCR+392, t102);
    _mm512_store_pd(SCR+464, t103);
    _mm512_store_pd(SCR+472, t104);
    _mm512_store_pd(SCR+544, t83);
    _mm512_store_pd(SCR+552, t84);
    _mm512_store_pd(SCR+624, t63);
    _mm512_store_pd(SCR+632, t64);
    _mm512_store_pd(SCR+704, t43);
    _mm512_store_pd(SCR+712, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+0);
    __m512d i0 = _mm512_load_pd(SCR+8);
    __m512d r1 = _mm512_load_pd(SCR+16);
    __m512d i1 = _mm512_load_pd(SCR+24);
    __m512d r2 = _mm512_load_pd(SCR+32);
    __m512d i2 = _mm512_load_pd(SCR+40);
    __m512d r3 = _mm512_load_pd(SCR+48);
    __m512d i3 = _mm512_load_pd(SCR+56);
    __m512d r4 = _mm512_load_pd(SCR+64);
    __m512d i4 = _mm512_load_pd(SCR+72);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+0*s, t11);
    _mm512_store_pd(X+0*s+8, t12);
    _mm512_store_pd(X+36*s, t21);
    _mm512_store_pd(X+36*s+8, t22);
    _mm512_store_pd(X+27*s, t33);
    _mm512_store_pd(X+27*s+8, t34);
    _mm512_store_pd(X+18*s, t35);
    _mm512_store_pd(X+18*s+8, t36);
    _mm512_store_pd(X+9*s, t23);
    _mm512_store_pd(X+9*s+8, t24);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+80);
    __m512d i0 = _mm512_load_pd(SCR+88);
    __m512d r1 = _mm512_load_pd(SCR+96);
    __m512d i1 = _mm512_load_pd(SCR+104);
    __m512d r2 = _mm512_load_pd(SCR+112);
    __m512d i2 = _mm512_load_pd(SCR+120);
    __m512d r3 = _mm512_load_pd(SCR+128);
    __m512d i3 = _mm512_load_pd(SCR+136);
    __m512d r4 = _mm512_load_pd(SCR+144);
    __m512d i4 = _mm512_load_pd(SCR+152);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+10*s, t11);
    _mm512_store_pd(X+10*s+8, t12);
    _mm512_store_pd(X+1*s, t21);
    _mm512_store_pd(X+1*s+8, t22);
    _mm512_store_pd(X+37*s, t33);
    _mm512_store_pd(X+37*s+8, t34);
    _mm512_store_pd(X+28*s, t35);
    _mm512_store_pd(X+28*s+8, t36);
    _mm512_store_pd(X+19*s, t23);
    _mm512_store_pd(X+19*s+8, t24);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+160);
    __m512d i0 = _mm512_load_pd(SCR+168);
    __m512d r1 = _mm512_load_pd(SCR+176);
    __m512d i1 = _mm512_load_pd(SCR+184);
    __m512d r2 = _mm512_load_pd(SCR+192);
    __m512d i2 = _mm512_load_pd(SCR+200);
    __m512d r3 = _mm512_load_pd(SCR+208);
    __m512d i3 = _mm512_load_pd(SCR+216);
    __m512d r4 = _mm512_load_pd(SCR+224);
    __m512d i4 = _mm512_load_pd(SCR+232);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+20*s, t11);
    _mm512_store_pd(X+20*s+8, t12);
    _mm512_store_pd(X+11*s, t21);
    _mm512_store_pd(X+11*s+8, t22);
    _mm512_store_pd(X+2*s, t33);
    _mm512_store_pd(X+2*s+8, t34);
    _mm512_store_pd(X+38*s, t35);
    _mm512_store_pd(X+38*s+8, t36);
    _mm512_store_pd(X+29*s, t23);
    _mm512_store_pd(X+29*s+8, t24);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+240);
    __m512d i0 = _mm512_load_pd(SCR+248);
    __m512d r1 = _mm512_load_pd(SCR+256);
    __m512d i1 = _mm512_load_pd(SCR+264);
    __m512d r2 = _mm512_load_pd(SCR+272);
    __m512d i2 = _mm512_load_pd(SCR+280);
    __m512d r3 = _mm512_load_pd(SCR+288);
    __m512d i3 = _mm512_load_pd(SCR+296);
    __m512d r4 = _mm512_load_pd(SCR+304);
    __m512d i4 = _mm512_load_pd(SCR+312);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+30*s, t11);
    _mm512_store_pd(X+30*s+8, t12);
    _mm512_store_pd(X+21*s, t21);
    _mm512_store_pd(X+21*s+8, t22);
    _mm512_store_pd(X+12*s, t33);
    _mm512_store_pd(X+12*s+8, t34);
    _mm512_store_pd(X+3*s, t35);
    _mm512_store_pd(X+3*s+8, t36);
    _mm512_store_pd(X+39*s, t23);
    _mm512_store_pd(X+39*s+8, t24);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+320);
    __m512d i0 = _mm512_load_pd(SCR+328);
    __m512d r1 = _mm512_load_pd(SCR+336);
    __m512d i1 = _mm512_load_pd(SCR+344);
    __m512d r2 = _mm512_load_pd(SCR+352);
    __m512d i2 = _mm512_load_pd(SCR+360);
    __m512d r3 = _mm512_load_pd(SCR+368);
    __m512d i3 = _mm512_load_pd(SCR+376);
    __m512d r4 = _mm512_load_pd(SCR+384);
    __m512d i4 = _mm512_load_pd(SCR+392);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+40*s, t11);
    _mm512_store_pd(X+40*s+8, t12);
    _mm512_store_pd(X+31*s, t21);
    _mm512_store_pd(X+31*s+8, t22);
    _mm512_store_pd(X+22*s, t33);
    _mm512_store_pd(X+22*s+8, t34);
    _mm512_store_pd(X+13*s, t35);
    _mm512_store_pd(X+13*s+8, t36);
    _mm512_store_pd(X+4*s, t23);
    _mm512_store_pd(X+4*s+8, t24);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+400);
    __m512d i0 = _mm512_load_pd(SCR+408);
    __m512d r1 = _mm512_load_pd(SCR+416);
    __m512d i1 = _mm512_load_pd(SCR+424);
    __m512d r2 = _mm512_load_pd(SCR+432);
    __m512d i2 = _mm512_load_pd(SCR+440);
    __m512d r3 = _mm512_load_pd(SCR+448);
    __m512d i3 = _mm512_load_pd(SCR+456);
    __m512d r4 = _mm512_load_pd(SCR+464);
    __m512d i4 = _mm512_load_pd(SCR+472);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+5*s, t11);
    _mm512_store_pd(X+5*s+8, t12);
    _mm512_store_pd(X+41*s, t21);
    _mm512_store_pd(X+41*s+8, t22);
    _mm512_store_pd(X+32*s, t33);
    _mm512_store_pd(X+32*s+8, t34);
    _mm512_store_pd(X+23*s, t35);
    _mm512_store_pd(X+23*s+8, t36);
    _mm512_store_pd(X+14*s, t23);
    _mm512_store_pd(X+14*s+8, t24);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+480);
    __m512d i0 = _mm512_load_pd(SCR+488);
    __m512d r1 = _mm512_load_pd(SCR+496);
    __m512d i1 = _mm512_load_pd(SCR+504);
    __m512d r2 = _mm512_load_pd(SCR+512);
    __m512d i2 = _mm512_load_pd(SCR+520);
    __m512d r3 = _mm512_load_pd(SCR+528);
    __m512d i3 = _mm512_load_pd(SCR+536);
    __m512d r4 = _mm512_load_pd(SCR+544);
    __m512d i4 = _mm512_load_pd(SCR+552);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+15*s, t11);
    _mm512_store_pd(X+15*s+8, t12);
    _mm512_store_pd(X+6*s, t21);
    _mm512_store_pd(X+6*s+8, t22);
    _mm512_store_pd(X+42*s, t33);
    _mm512_store_pd(X+42*s+8, t34);
    _mm512_store_pd(X+33*s, t35);
    _mm512_store_pd(X+33*s+8, t36);
    _mm512_store_pd(X+24*s, t23);
    _mm512_store_pd(X+24*s+8, t24);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+560);
    __m512d i0 = _mm512_load_pd(SCR+568);
    __m512d r1 = _mm512_load_pd(SCR+576);
    __m512d i1 = _mm512_load_pd(SCR+584);
    __m512d r2 = _mm512_load_pd(SCR+592);
    __m512d i2 = _mm512_load_pd(SCR+600);
    __m512d r3 = _mm512_load_pd(SCR+608);
    __m512d i3 = _mm512_load_pd(SCR+616);
    __m512d r4 = _mm512_load_pd(SCR+624);
    __m512d i4 = _mm512_load_pd(SCR+632);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+25*s, t11);
    _mm512_store_pd(X+25*s+8, t12);
    _mm512_store_pd(X+16*s, t21);
    _mm512_store_pd(X+16*s+8, t22);
    _mm512_store_pd(X+7*s, t33);
    _mm512_store_pd(X+7*s+8, t34);
    _mm512_store_pd(X+43*s, t35);
    _mm512_store_pd(X+43*s+8, t36);
    _mm512_store_pd(X+34*s, t23);
    _mm512_store_pd(X+34*s+8, t24);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+640);
    __m512d i0 = _mm512_load_pd(SCR+648);
    __m512d r1 = _mm512_load_pd(SCR+656);
    __m512d i1 = _mm512_load_pd(SCR+664);
    __m512d r2 = _mm512_load_pd(SCR+672);
    __m512d i2 = _mm512_load_pd(SCR+680);
    __m512d r3 = _mm512_load_pd(SCR+688);
    __m512d i3 = _mm512_load_pd(SCR+696);
    __m512d r4 = _mm512_load_pd(SCR+704);
    __m512d i4 = _mm512_load_pd(SCR+712);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+35*s, t11);
    _mm512_store_pd(X+35*s+8, t12);
    _mm512_store_pd(X+26*s, t21);
    _mm512_store_pd(X+26*s+8, t22);
    _mm512_store_pd(X+17*s, t33);
    _mm512_store_pd(X+17*s+8, t34);
    _mm512_store_pd(X+8*s, t35);
    _mm512_store_pd(X+8*s+8, t36);
    _mm512_store_pd(X+44*s, t23);
    _mm512_store_pd(X+44*s+8, t24);
    }
}
static void dft45sq(double* restrict X, long es){
    const long s = es*16;
    double SCR[720] ALIGN64;
    double SQ[720] ALIGN64;
    for(long q=0; q<720; q+=16){
        _mm512_store_pd(SQ+q,   _mm512_load_pd(X+q));
        _mm512_store_pd(SQ+q+8, _mm512_load_pd(X+q+8));
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(SQ+0*16);
    __m512d i0 = _mm512_load_pd(SQ+0*16+8);
    __m512d r1 = _mm512_load_pd(SQ+5*16);
    __m512d i1 = _mm512_load_pd(SQ+5*16+8);
    __m512d r2 = _mm512_load_pd(SQ+10*16);
    __m512d i2 = _mm512_load_pd(SQ+10*16+8);
    __m512d r3 = _mm512_load_pd(SQ+15*16);
    __m512d i3 = _mm512_load_pd(SQ+15*16+8);
    __m512d r4 = _mm512_load_pd(SQ+20*16);
    __m512d i4 = _mm512_load_pd(SQ+20*16+8);
    __m512d r5 = _mm512_load_pd(SQ+25*16);
    __m512d i5 = _mm512_load_pd(SQ+25*16+8);
    __m512d r6 = _mm512_load_pd(SQ+30*16);
    __m512d i6 = _mm512_load_pd(SQ+30*16+8);
    __m512d r7 = _mm512_load_pd(SQ+35*16);
    __m512d i7 = _mm512_load_pd(SQ+35*16+8);
    __m512d r8 = _mm512_load_pd(SQ+40*16);
    __m512d i8 = _mm512_load_pd(SQ+40*16+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+0, t23);
    _mm512_store_pd(SCR+8, t24);
    _mm512_store_pd(SCR+80, t41);
    _mm512_store_pd(SCR+88, t42);
    _mm512_store_pd(SCR+160, t61);
    _mm512_store_pd(SCR+168, t62);
    _mm512_store_pd(SCR+240, t81);
    _mm512_store_pd(SCR+248, t82);
    _mm512_store_pd(SCR+320, t101);
    _mm512_store_pd(SCR+328, t102);
    _mm512_store_pd(SCR+400, t103);
    _mm512_store_pd(SCR+408, t104);
    _mm512_store_pd(SCR+480, t83);
    _mm512_store_pd(SCR+488, t84);
    _mm512_store_pd(SCR+560, t63);
    _mm512_store_pd(SCR+568, t64);
    _mm512_store_pd(SCR+640, t43);
    _mm512_store_pd(SCR+648, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(SQ+9*16);
    __m512d i0 = _mm512_load_pd(SQ+9*16+8);
    __m512d r1 = _mm512_load_pd(SQ+14*16);
    __m512d i1 = _mm512_load_pd(SQ+14*16+8);
    __m512d r2 = _mm512_load_pd(SQ+19*16);
    __m512d i2 = _mm512_load_pd(SQ+19*16+8);
    __m512d r3 = _mm512_load_pd(SQ+24*16);
    __m512d i3 = _mm512_load_pd(SQ+24*16+8);
    __m512d r4 = _mm512_load_pd(SQ+29*16);
    __m512d i4 = _mm512_load_pd(SQ+29*16+8);
    __m512d r5 = _mm512_load_pd(SQ+34*16);
    __m512d i5 = _mm512_load_pd(SQ+34*16+8);
    __m512d r6 = _mm512_load_pd(SQ+39*16);
    __m512d i6 = _mm512_load_pd(SQ+39*16+8);
    __m512d r7 = _mm512_load_pd(SQ+44*16);
    __m512d i7 = _mm512_load_pd(SQ+44*16+8);
    __m512d r8 = _mm512_load_pd(SQ+4*16);
    __m512d i8 = _mm512_load_pd(SQ+4*16+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+16, t23);
    _mm512_store_pd(SCR+24, t24);
    _mm512_store_pd(SCR+96, t41);
    _mm512_store_pd(SCR+104, t42);
    _mm512_store_pd(SCR+176, t61);
    _mm512_store_pd(SCR+184, t62);
    _mm512_store_pd(SCR+256, t81);
    _mm512_store_pd(SCR+264, t82);
    _mm512_store_pd(SCR+336, t101);
    _mm512_store_pd(SCR+344, t102);
    _mm512_store_pd(SCR+416, t103);
    _mm512_store_pd(SCR+424, t104);
    _mm512_store_pd(SCR+496, t83);
    _mm512_store_pd(SCR+504, t84);
    _mm512_store_pd(SCR+576, t63);
    _mm512_store_pd(SCR+584, t64);
    _mm512_store_pd(SCR+656, t43);
    _mm512_store_pd(SCR+664, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(SQ+18*16);
    __m512d i0 = _mm512_load_pd(SQ+18*16+8);
    __m512d r1 = _mm512_load_pd(SQ+23*16);
    __m512d i1 = _mm512_load_pd(SQ+23*16+8);
    __m512d r2 = _mm512_load_pd(SQ+28*16);
    __m512d i2 = _mm512_load_pd(SQ+28*16+8);
    __m512d r3 = _mm512_load_pd(SQ+33*16);
    __m512d i3 = _mm512_load_pd(SQ+33*16+8);
    __m512d r4 = _mm512_load_pd(SQ+38*16);
    __m512d i4 = _mm512_load_pd(SQ+38*16+8);
    __m512d r5 = _mm512_load_pd(SQ+43*16);
    __m512d i5 = _mm512_load_pd(SQ+43*16+8);
    __m512d r6 = _mm512_load_pd(SQ+3*16);
    __m512d i6 = _mm512_load_pd(SQ+3*16+8);
    __m512d r7 = _mm512_load_pd(SQ+8*16);
    __m512d i7 = _mm512_load_pd(SQ+8*16+8);
    __m512d r8 = _mm512_load_pd(SQ+13*16);
    __m512d i8 = _mm512_load_pd(SQ+13*16+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+32, t23);
    _mm512_store_pd(SCR+40, t24);
    _mm512_store_pd(SCR+112, t41);
    _mm512_store_pd(SCR+120, t42);
    _mm512_store_pd(SCR+192, t61);
    _mm512_store_pd(SCR+200, t62);
    _mm512_store_pd(SCR+272, t81);
    _mm512_store_pd(SCR+280, t82);
    _mm512_store_pd(SCR+352, t101);
    _mm512_store_pd(SCR+360, t102);
    _mm512_store_pd(SCR+432, t103);
    _mm512_store_pd(SCR+440, t104);
    _mm512_store_pd(SCR+512, t83);
    _mm512_store_pd(SCR+520, t84);
    _mm512_store_pd(SCR+592, t63);
    _mm512_store_pd(SCR+600, t64);
    _mm512_store_pd(SCR+672, t43);
    _mm512_store_pd(SCR+680, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(SQ+27*16);
    __m512d i0 = _mm512_load_pd(SQ+27*16+8);
    __m512d r1 = _mm512_load_pd(SQ+32*16);
    __m512d i1 = _mm512_load_pd(SQ+32*16+8);
    __m512d r2 = _mm512_load_pd(SQ+37*16);
    __m512d i2 = _mm512_load_pd(SQ+37*16+8);
    __m512d r3 = _mm512_load_pd(SQ+42*16);
    __m512d i3 = _mm512_load_pd(SQ+42*16+8);
    __m512d r4 = _mm512_load_pd(SQ+2*16);
    __m512d i4 = _mm512_load_pd(SQ+2*16+8);
    __m512d r5 = _mm512_load_pd(SQ+7*16);
    __m512d i5 = _mm512_load_pd(SQ+7*16+8);
    __m512d r6 = _mm512_load_pd(SQ+12*16);
    __m512d i6 = _mm512_load_pd(SQ+12*16+8);
    __m512d r7 = _mm512_load_pd(SQ+17*16);
    __m512d i7 = _mm512_load_pd(SQ+17*16+8);
    __m512d r8 = _mm512_load_pd(SQ+22*16);
    __m512d i8 = _mm512_load_pd(SQ+22*16+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+48, t23);
    _mm512_store_pd(SCR+56, t24);
    _mm512_store_pd(SCR+128, t41);
    _mm512_store_pd(SCR+136, t42);
    _mm512_store_pd(SCR+208, t61);
    _mm512_store_pd(SCR+216, t62);
    _mm512_store_pd(SCR+288, t81);
    _mm512_store_pd(SCR+296, t82);
    _mm512_store_pd(SCR+368, t101);
    _mm512_store_pd(SCR+376, t102);
    _mm512_store_pd(SCR+448, t103);
    _mm512_store_pd(SCR+456, t104);
    _mm512_store_pd(SCR+528, t83);
    _mm512_store_pd(SCR+536, t84);
    _mm512_store_pd(SCR+608, t63);
    _mm512_store_pd(SCR+616, t64);
    _mm512_store_pd(SCR+688, t43);
    _mm512_store_pd(SCR+696, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(SQ+36*16);
    __m512d i0 = _mm512_load_pd(SQ+36*16+8);
    __m512d r1 = _mm512_load_pd(SQ+41*16);
    __m512d i1 = _mm512_load_pd(SQ+41*16+8);
    __m512d r2 = _mm512_load_pd(SQ+1*16);
    __m512d i2 = _mm512_load_pd(SQ+1*16+8);
    __m512d r3 = _mm512_load_pd(SQ+6*16);
    __m512d i3 = _mm512_load_pd(SQ+6*16+8);
    __m512d r4 = _mm512_load_pd(SQ+11*16);
    __m512d i4 = _mm512_load_pd(SQ+11*16+8);
    __m512d r5 = _mm512_load_pd(SQ+16*16);
    __m512d i5 = _mm512_load_pd(SQ+16*16+8);
    __m512d r6 = _mm512_load_pd(SQ+21*16);
    __m512d i6 = _mm512_load_pd(SQ+21*16+8);
    __m512d r7 = _mm512_load_pd(SQ+26*16);
    __m512d i7 = _mm512_load_pd(SQ+26*16+8);
    __m512d r8 = _mm512_load_pd(SQ+31*16);
    __m512d i8 = _mm512_load_pd(SQ+31*16+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+64, t23);
    _mm512_store_pd(SCR+72, t24);
    _mm512_store_pd(SCR+144, t41);
    _mm512_store_pd(SCR+152, t42);
    _mm512_store_pd(SCR+224, t61);
    _mm512_store_pd(SCR+232, t62);
    _mm512_store_pd(SCR+304, t81);
    _mm512_store_pd(SCR+312, t82);
    _mm512_store_pd(SCR+384, t101);
    _mm512_store_pd(SCR+392, t102);
    _mm512_store_pd(SCR+464, t103);
    _mm512_store_pd(SCR+472, t104);
    _mm512_store_pd(SCR+544, t83);
    _mm512_store_pd(SCR+552, t84);
    _mm512_store_pd(SCR+624, t63);
    _mm512_store_pd(SCR+632, t64);
    _mm512_store_pd(SCR+704, t43);
    _mm512_store_pd(SCR+712, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+0);
    __m512d i0 = _mm512_load_pd(SCR+8);
    __m512d r1 = _mm512_load_pd(SCR+16);
    __m512d i1 = _mm512_load_pd(SCR+24);
    __m512d r2 = _mm512_load_pd(SCR+32);
    __m512d i2 = _mm512_load_pd(SCR+40);
    __m512d r3 = _mm512_load_pd(SCR+48);
    __m512d i3 = _mm512_load_pd(SCR+56);
    __m512d r4 = _mm512_load_pd(SCR+64);
    __m512d i4 = _mm512_load_pd(SCR+72);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+0*s, t11);
    _mm512_store_pd(X+0*s+8, t12);
    _mm512_store_pd(X+36*s, t21);
    _mm512_store_pd(X+36*s+8, t22);
    _mm512_store_pd(X+27*s, t33);
    _mm512_store_pd(X+27*s+8, t34);
    _mm512_store_pd(X+18*s, t35);
    _mm512_store_pd(X+18*s+8, t36);
    _mm512_store_pd(X+9*s, t23);
    _mm512_store_pd(X+9*s+8, t24);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+80);
    __m512d i0 = _mm512_load_pd(SCR+88);
    __m512d r1 = _mm512_load_pd(SCR+96);
    __m512d i1 = _mm512_load_pd(SCR+104);
    __m512d r2 = _mm512_load_pd(SCR+112);
    __m512d i2 = _mm512_load_pd(SCR+120);
    __m512d r3 = _mm512_load_pd(SCR+128);
    __m512d i3 = _mm512_load_pd(SCR+136);
    __m512d r4 = _mm512_load_pd(SCR+144);
    __m512d i4 = _mm512_load_pd(SCR+152);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+10*s, t11);
    _mm512_store_pd(X+10*s+8, t12);
    _mm512_store_pd(X+1*s, t21);
    _mm512_store_pd(X+1*s+8, t22);
    _mm512_store_pd(X+37*s, t33);
    _mm512_store_pd(X+37*s+8, t34);
    _mm512_store_pd(X+28*s, t35);
    _mm512_store_pd(X+28*s+8, t36);
    _mm512_store_pd(X+19*s, t23);
    _mm512_store_pd(X+19*s+8, t24);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+160);
    __m512d i0 = _mm512_load_pd(SCR+168);
    __m512d r1 = _mm512_load_pd(SCR+176);
    __m512d i1 = _mm512_load_pd(SCR+184);
    __m512d r2 = _mm512_load_pd(SCR+192);
    __m512d i2 = _mm512_load_pd(SCR+200);
    __m512d r3 = _mm512_load_pd(SCR+208);
    __m512d i3 = _mm512_load_pd(SCR+216);
    __m512d r4 = _mm512_load_pd(SCR+224);
    __m512d i4 = _mm512_load_pd(SCR+232);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+20*s, t11);
    _mm512_store_pd(X+20*s+8, t12);
    _mm512_store_pd(X+11*s, t21);
    _mm512_store_pd(X+11*s+8, t22);
    _mm512_store_pd(X+2*s, t33);
    _mm512_store_pd(X+2*s+8, t34);
    _mm512_store_pd(X+38*s, t35);
    _mm512_store_pd(X+38*s+8, t36);
    _mm512_store_pd(X+29*s, t23);
    _mm512_store_pd(X+29*s+8, t24);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+240);
    __m512d i0 = _mm512_load_pd(SCR+248);
    __m512d r1 = _mm512_load_pd(SCR+256);
    __m512d i1 = _mm512_load_pd(SCR+264);
    __m512d r2 = _mm512_load_pd(SCR+272);
    __m512d i2 = _mm512_load_pd(SCR+280);
    __m512d r3 = _mm512_load_pd(SCR+288);
    __m512d i3 = _mm512_load_pd(SCR+296);
    __m512d r4 = _mm512_load_pd(SCR+304);
    __m512d i4 = _mm512_load_pd(SCR+312);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+30*s, t11);
    _mm512_store_pd(X+30*s+8, t12);
    _mm512_store_pd(X+21*s, t21);
    _mm512_store_pd(X+21*s+8, t22);
    _mm512_store_pd(X+12*s, t33);
    _mm512_store_pd(X+12*s+8, t34);
    _mm512_store_pd(X+3*s, t35);
    _mm512_store_pd(X+3*s+8, t36);
    _mm512_store_pd(X+39*s, t23);
    _mm512_store_pd(X+39*s+8, t24);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+320);
    __m512d i0 = _mm512_load_pd(SCR+328);
    __m512d r1 = _mm512_load_pd(SCR+336);
    __m512d i1 = _mm512_load_pd(SCR+344);
    __m512d r2 = _mm512_load_pd(SCR+352);
    __m512d i2 = _mm512_load_pd(SCR+360);
    __m512d r3 = _mm512_load_pd(SCR+368);
    __m512d i3 = _mm512_load_pd(SCR+376);
    __m512d r4 = _mm512_load_pd(SCR+384);
    __m512d i4 = _mm512_load_pd(SCR+392);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+40*s, t11);
    _mm512_store_pd(X+40*s+8, t12);
    _mm512_store_pd(X+31*s, t21);
    _mm512_store_pd(X+31*s+8, t22);
    _mm512_store_pd(X+22*s, t33);
    _mm512_store_pd(X+22*s+8, t34);
    _mm512_store_pd(X+13*s, t35);
    _mm512_store_pd(X+13*s+8, t36);
    _mm512_store_pd(X+4*s, t23);
    _mm512_store_pd(X+4*s+8, t24);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+400);
    __m512d i0 = _mm512_load_pd(SCR+408);
    __m512d r1 = _mm512_load_pd(SCR+416);
    __m512d i1 = _mm512_load_pd(SCR+424);
    __m512d r2 = _mm512_load_pd(SCR+432);
    __m512d i2 = _mm512_load_pd(SCR+440);
    __m512d r3 = _mm512_load_pd(SCR+448);
    __m512d i3 = _mm512_load_pd(SCR+456);
    __m512d r4 = _mm512_load_pd(SCR+464);
    __m512d i4 = _mm512_load_pd(SCR+472);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+5*s, t11);
    _mm512_store_pd(X+5*s+8, t12);
    _mm512_store_pd(X+41*s, t21);
    _mm512_store_pd(X+41*s+8, t22);
    _mm512_store_pd(X+32*s, t33);
    _mm512_store_pd(X+32*s+8, t34);
    _mm512_store_pd(X+23*s, t35);
    _mm512_store_pd(X+23*s+8, t36);
    _mm512_store_pd(X+14*s, t23);
    _mm512_store_pd(X+14*s+8, t24);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+480);
    __m512d i0 = _mm512_load_pd(SCR+488);
    __m512d r1 = _mm512_load_pd(SCR+496);
    __m512d i1 = _mm512_load_pd(SCR+504);
    __m512d r2 = _mm512_load_pd(SCR+512);
    __m512d i2 = _mm512_load_pd(SCR+520);
    __m512d r3 = _mm512_load_pd(SCR+528);
    __m512d i3 = _mm512_load_pd(SCR+536);
    __m512d r4 = _mm512_load_pd(SCR+544);
    __m512d i4 = _mm512_load_pd(SCR+552);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+15*s, t11);
    _mm512_store_pd(X+15*s+8, t12);
    _mm512_store_pd(X+6*s, t21);
    _mm512_store_pd(X+6*s+8, t22);
    _mm512_store_pd(X+42*s, t33);
    _mm512_store_pd(X+42*s+8, t34);
    _mm512_store_pd(X+33*s, t35);
    _mm512_store_pd(X+33*s+8, t36);
    _mm512_store_pd(X+24*s, t23);
    _mm512_store_pd(X+24*s+8, t24);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+560);
    __m512d i0 = _mm512_load_pd(SCR+568);
    __m512d r1 = _mm512_load_pd(SCR+576);
    __m512d i1 = _mm512_load_pd(SCR+584);
    __m512d r2 = _mm512_load_pd(SCR+592);
    __m512d i2 = _mm512_load_pd(SCR+600);
    __m512d r3 = _mm512_load_pd(SCR+608);
    __m512d i3 = _mm512_load_pd(SCR+616);
    __m512d r4 = _mm512_load_pd(SCR+624);
    __m512d i4 = _mm512_load_pd(SCR+632);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+25*s, t11);
    _mm512_store_pd(X+25*s+8, t12);
    _mm512_store_pd(X+16*s, t21);
    _mm512_store_pd(X+16*s+8, t22);
    _mm512_store_pd(X+7*s, t33);
    _mm512_store_pd(X+7*s+8, t34);
    _mm512_store_pd(X+43*s, t35);
    _mm512_store_pd(X+43*s+8, t36);
    _mm512_store_pd(X+34*s, t23);
    _mm512_store_pd(X+34*s+8, t24);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+640);
    __m512d i0 = _mm512_load_pd(SCR+648);
    __m512d r1 = _mm512_load_pd(SCR+656);
    __m512d i1 = _mm512_load_pd(SCR+664);
    __m512d r2 = _mm512_load_pd(SCR+672);
    __m512d i2 = _mm512_load_pd(SCR+680);
    __m512d r3 = _mm512_load_pd(SCR+688);
    __m512d i3 = _mm512_load_pd(SCR+696);
    __m512d r4 = _mm512_load_pd(SCR+704);
    __m512d i4 = _mm512_load_pd(SCR+712);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    _mm512_store_pd(X+35*s, t11);
    _mm512_store_pd(X+35*s+8, t12);
    _mm512_store_pd(X+26*s, t21);
    _mm512_store_pd(X+26*s+8, t22);
    _mm512_store_pd(X+17*s, t33);
    _mm512_store_pd(X+17*s+8, t34);
    _mm512_store_pd(X+8*s, t35);
    _mm512_store_pd(X+8*s+8, t36);
    _mm512_store_pd(X+44*s, t23);
    _mm512_store_pd(X+44*s+8, t24);
    }
}
static void dft45m(double* restrict X, long es, const double* restrict C){
    const long s = es*16;
    double SCR[720] ALIGN64;
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+0*s);
    __m512d i0 = _mm512_load_pd(X+0*s+8);
    __m512d r1 = _mm512_load_pd(X+5*s);
    __m512d i1 = _mm512_load_pd(X+5*s+8);
    __m512d r2 = _mm512_load_pd(X+10*s);
    __m512d i2 = _mm512_load_pd(X+10*s+8);
    __m512d r3 = _mm512_load_pd(X+15*s);
    __m512d i3 = _mm512_load_pd(X+15*s+8);
    __m512d r4 = _mm512_load_pd(X+20*s);
    __m512d i4 = _mm512_load_pd(X+20*s+8);
    __m512d r5 = _mm512_load_pd(X+25*s);
    __m512d i5 = _mm512_load_pd(X+25*s+8);
    __m512d r6 = _mm512_load_pd(X+30*s);
    __m512d i6 = _mm512_load_pd(X+30*s+8);
    __m512d r7 = _mm512_load_pd(X+35*s);
    __m512d i7 = _mm512_load_pd(X+35*s+8);
    __m512d r8 = _mm512_load_pd(X+40*s);
    __m512d i8 = _mm512_load_pd(X+40*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+0, t23);
    _mm512_store_pd(SCR+8, t24);
    _mm512_store_pd(SCR+80, t41);
    _mm512_store_pd(SCR+88, t42);
    _mm512_store_pd(SCR+160, t61);
    _mm512_store_pd(SCR+168, t62);
    _mm512_store_pd(SCR+240, t81);
    _mm512_store_pd(SCR+248, t82);
    _mm512_store_pd(SCR+320, t101);
    _mm512_store_pd(SCR+328, t102);
    _mm512_store_pd(SCR+400, t103);
    _mm512_store_pd(SCR+408, t104);
    _mm512_store_pd(SCR+480, t83);
    _mm512_store_pd(SCR+488, t84);
    _mm512_store_pd(SCR+560, t63);
    _mm512_store_pd(SCR+568, t64);
    _mm512_store_pd(SCR+640, t43);
    _mm512_store_pd(SCR+648, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+9*s);
    __m512d i0 = _mm512_load_pd(X+9*s+8);
    __m512d r1 = _mm512_load_pd(X+14*s);
    __m512d i1 = _mm512_load_pd(X+14*s+8);
    __m512d r2 = _mm512_load_pd(X+19*s);
    __m512d i2 = _mm512_load_pd(X+19*s+8);
    __m512d r3 = _mm512_load_pd(X+24*s);
    __m512d i3 = _mm512_load_pd(X+24*s+8);
    __m512d r4 = _mm512_load_pd(X+29*s);
    __m512d i4 = _mm512_load_pd(X+29*s+8);
    __m512d r5 = _mm512_load_pd(X+34*s);
    __m512d i5 = _mm512_load_pd(X+34*s+8);
    __m512d r6 = _mm512_load_pd(X+39*s);
    __m512d i6 = _mm512_load_pd(X+39*s+8);
    __m512d r7 = _mm512_load_pd(X+44*s);
    __m512d i7 = _mm512_load_pd(X+44*s+8);
    __m512d r8 = _mm512_load_pd(X+4*s);
    __m512d i8 = _mm512_load_pd(X+4*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+16, t23);
    _mm512_store_pd(SCR+24, t24);
    _mm512_store_pd(SCR+96, t41);
    _mm512_store_pd(SCR+104, t42);
    _mm512_store_pd(SCR+176, t61);
    _mm512_store_pd(SCR+184, t62);
    _mm512_store_pd(SCR+256, t81);
    _mm512_store_pd(SCR+264, t82);
    _mm512_store_pd(SCR+336, t101);
    _mm512_store_pd(SCR+344, t102);
    _mm512_store_pd(SCR+416, t103);
    _mm512_store_pd(SCR+424, t104);
    _mm512_store_pd(SCR+496, t83);
    _mm512_store_pd(SCR+504, t84);
    _mm512_store_pd(SCR+576, t63);
    _mm512_store_pd(SCR+584, t64);
    _mm512_store_pd(SCR+656, t43);
    _mm512_store_pd(SCR+664, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+18*s);
    __m512d i0 = _mm512_load_pd(X+18*s+8);
    __m512d r1 = _mm512_load_pd(X+23*s);
    __m512d i1 = _mm512_load_pd(X+23*s+8);
    __m512d r2 = _mm512_load_pd(X+28*s);
    __m512d i2 = _mm512_load_pd(X+28*s+8);
    __m512d r3 = _mm512_load_pd(X+33*s);
    __m512d i3 = _mm512_load_pd(X+33*s+8);
    __m512d r4 = _mm512_load_pd(X+38*s);
    __m512d i4 = _mm512_load_pd(X+38*s+8);
    __m512d r5 = _mm512_load_pd(X+43*s);
    __m512d i5 = _mm512_load_pd(X+43*s+8);
    __m512d r6 = _mm512_load_pd(X+3*s);
    __m512d i6 = _mm512_load_pd(X+3*s+8);
    __m512d r7 = _mm512_load_pd(X+8*s);
    __m512d i7 = _mm512_load_pd(X+8*s+8);
    __m512d r8 = _mm512_load_pd(X+13*s);
    __m512d i8 = _mm512_load_pd(X+13*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+32, t23);
    _mm512_store_pd(SCR+40, t24);
    _mm512_store_pd(SCR+112, t41);
    _mm512_store_pd(SCR+120, t42);
    _mm512_store_pd(SCR+192, t61);
    _mm512_store_pd(SCR+200, t62);
    _mm512_store_pd(SCR+272, t81);
    _mm512_store_pd(SCR+280, t82);
    _mm512_store_pd(SCR+352, t101);
    _mm512_store_pd(SCR+360, t102);
    _mm512_store_pd(SCR+432, t103);
    _mm512_store_pd(SCR+440, t104);
    _mm512_store_pd(SCR+512, t83);
    _mm512_store_pd(SCR+520, t84);
    _mm512_store_pd(SCR+592, t63);
    _mm512_store_pd(SCR+600, t64);
    _mm512_store_pd(SCR+672, t43);
    _mm512_store_pd(SCR+680, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+27*s);
    __m512d i0 = _mm512_load_pd(X+27*s+8);
    __m512d r1 = _mm512_load_pd(X+32*s);
    __m512d i1 = _mm512_load_pd(X+32*s+8);
    __m512d r2 = _mm512_load_pd(X+37*s);
    __m512d i2 = _mm512_load_pd(X+37*s+8);
    __m512d r3 = _mm512_load_pd(X+42*s);
    __m512d i3 = _mm512_load_pd(X+42*s+8);
    __m512d r4 = _mm512_load_pd(X+2*s);
    __m512d i4 = _mm512_load_pd(X+2*s+8);
    __m512d r5 = _mm512_load_pd(X+7*s);
    __m512d i5 = _mm512_load_pd(X+7*s+8);
    __m512d r6 = _mm512_load_pd(X+12*s);
    __m512d i6 = _mm512_load_pd(X+12*s+8);
    __m512d r7 = _mm512_load_pd(X+17*s);
    __m512d i7 = _mm512_load_pd(X+17*s+8);
    __m512d r8 = _mm512_load_pd(X+22*s);
    __m512d i8 = _mm512_load_pd(X+22*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+48, t23);
    _mm512_store_pd(SCR+56, t24);
    _mm512_store_pd(SCR+128, t41);
    _mm512_store_pd(SCR+136, t42);
    _mm512_store_pd(SCR+208, t61);
    _mm512_store_pd(SCR+216, t62);
    _mm512_store_pd(SCR+288, t81);
    _mm512_store_pd(SCR+296, t82);
    _mm512_store_pd(SCR+368, t101);
    _mm512_store_pd(SCR+376, t102);
    _mm512_store_pd(SCR+448, t103);
    _mm512_store_pd(SCR+456, t104);
    _mm512_store_pd(SCR+528, t83);
    _mm512_store_pd(SCR+536, t84);
    _mm512_store_pd(SCR+608, t63);
    _mm512_store_pd(SCR+616, t64);
    _mm512_store_pd(SCR+688, t43);
    _mm512_store_pd(SCR+696, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.8836fa2cf5039p-1);
    const __m512d k1 = _mm512_set1_pd(0x1.63a1a7e0b738cp-3);
    const __m512d k2 = _mm512_set1_pd(-0x1.ffffffffffffdp-2);
    const __m512d k3 = _mm512_set1_pd(-0x1.e11f642522d1bp-1);
    const __m512d k4 = _mm512_set1_pd(0x1.491b7523c161cp-1);
    const __m512d k5 = _mm512_set1_pd(0x1.f838b8c811c17p-1);
    const __m512d k6 = _mm512_set1_pd(0x1.bb67ae8584cabp-1);
    const __m512d k7 = _mm512_set1_pd(0x1.5e3a8748a0bf8p-2);
    const __m512d k8 = _mm512_set1_pd(0x1.0000000000000p+0);
    const __m512d k9 = _mm512_set1_pd(0x0.0p+0);
    __m512d r0 = _mm512_load_pd(X+36*s);
    __m512d i0 = _mm512_load_pd(X+36*s+8);
    __m512d r1 = _mm512_load_pd(X+41*s);
    __m512d i1 = _mm512_load_pd(X+41*s+8);
    __m512d r2 = _mm512_load_pd(X+1*s);
    __m512d i2 = _mm512_load_pd(X+1*s+8);
    __m512d r3 = _mm512_load_pd(X+6*s);
    __m512d i3 = _mm512_load_pd(X+6*s+8);
    __m512d r4 = _mm512_load_pd(X+11*s);
    __m512d i4 = _mm512_load_pd(X+11*s+8);
    __m512d r5 = _mm512_load_pd(X+16*s);
    __m512d i5 = _mm512_load_pd(X+16*s+8);
    __m512d r6 = _mm512_load_pd(X+21*s);
    __m512d i6 = _mm512_load_pd(X+21*s+8);
    __m512d r7 = _mm512_load_pd(X+26*s);
    __m512d i7 = _mm512_load_pd(X+26*s+8);
    __m512d r8 = _mm512_load_pd(X+31*s);
    __m512d i8 = _mm512_load_pd(X+31*s+8);
    __m512d t1 = _mm512_add_pd(r1, r8);
    __m512d t2 = _mm512_add_pd(i1, i8);
    __m512d t3 = _mm512_sub_pd(r1, r8);
    __m512d t4 = _mm512_sub_pd(i1, i8);
    __m512d t5 = _mm512_add_pd(r2, r7);
    __m512d t6 = _mm512_add_pd(i2, i7);
    __m512d t7 = _mm512_sub_pd(r2, r7);
    __m512d t8 = _mm512_sub_pd(i2, i7);
    __m512d t9 = _mm512_add_pd(r3, r6);
    __m512d t10 = _mm512_add_pd(i3, i6);
    __m512d t11 = _mm512_sub_pd(r3, r6);
    __m512d t12 = _mm512_sub_pd(i3, i6);
    __m512d t13 = _mm512_add_pd(r4, r5);
    __m512d t14 = _mm512_add_pd(i4, i5);
    __m512d t15 = _mm512_sub_pd(r4, r5);
    __m512d t16 = _mm512_sub_pd(i4, i5);
    __m512d t17 = _mm512_add_pd(r0, t1);
    __m512d t18 = _mm512_add_pd(i0, t2);
    __m512d t19 = _mm512_add_pd(t17, t5);
    __m512d t20 = _mm512_add_pd(t18, t6);
    __m512d t21 = _mm512_add_pd(t19, t9);
    __m512d t22 = _mm512_add_pd(t20, t10);
    __m512d t23 = _mm512_add_pd(t21, t13);
    __m512d t24 = _mm512_add_pd(t22, t14);
    __m512d t25 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k1, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k1, t6, t26);
    __m512d t29 = _mm512_fmadd_pd(k2, t9, t27);
    __m512d t30 = _mm512_fmadd_pd(k2, t10, t28);
    __m512d t31 = _mm512_fmadd_pd(k3, t13, t29);
    __m512d t32 = _mm512_fmadd_pd(k3, t14, t30);
    __m512d t33 = _mm512_mul_pd(k4, t3);
    __m512d t34 = _mm512_mul_pd(k4, t4);
    __m512d t35 = _mm512_fmadd_pd(k5, t7, t33);
    __m512d t36 = _mm512_fmadd_pd(k5, t8, t34);
    __m512d t37 = _mm512_fmadd_pd(k6, t11, t35);
    __m512d t38 = _mm512_fmadd_pd(k6, t12, t36);
    __m512d t39 = _mm512_fmadd_pd(k7, t15, t37);
    __m512d t40 = _mm512_fmadd_pd(k7, t16, t38);
    __m512d t41 = _mm512_add_pd(t31, t40);
    __m512d t42 = _mm512_sub_pd(t32, t39);
    __m512d t43 = _mm512_sub_pd(t31, t40);
    __m512d t44 = _mm512_add_pd(t32, t39);
    __m512d t45 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t46 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t47 = _mm512_fmadd_pd(k3, t5, t45);
    __m512d t48 = _mm512_fmadd_pd(k3, t6, t46);
    __m512d t49 = _mm512_fmadd_pd(k2, t9, t47);
    __m512d t50 = _mm512_fmadd_pd(k2, t10, t48);
    __m512d t51 = _mm512_fmadd_pd(k0, t13, t49);
    __m512d t52 = _mm512_fmadd_pd(k0, t14, t50);
    __m512d t53 = _mm512_mul_pd(k5, t3);
    __m512d t54 = _mm512_mul_pd(k5, t4);
    __m512d t55 = _mm512_fmadd_pd(k7, t7, t53);
    __m512d t56 = _mm512_fmadd_pd(k7, t8, t54);
    __m512d t57 = _mm512_fnmadd_pd(k6, t11, t55);
    __m512d t58 = _mm512_fnmadd_pd(k6, t12, t56);
    __m512d t59 = _mm512_fnmadd_pd(k4, t15, t57);
    __m512d t60 = _mm512_fnmadd_pd(k4, t16, t58);
    __m512d t61 = _mm512_add_pd(t51, t60);
    __m512d t62 = _mm512_sub_pd(t52, t59);
    __m512d t63 = _mm512_sub_pd(t51, t60);
    __m512d t64 = _mm512_add_pd(t52, t59);
    __m512d t65 = _mm512_fmadd_pd(k2, t1, r0);
    __m512d t66 = _mm512_fmadd_pd(k2, t2, i0);
    __m512d t67 = _mm512_fmadd_pd(k2, t5, t65);
    __m512d t68 = _mm512_fmadd_pd(k2, t6, t66);
    __m512d t69 = _mm512_fmadd_pd(k8, t9, t67);
    __m512d t70 = _mm512_fmadd_pd(k8, t10, t68);
    __m512d t71 = _mm512_fmadd_pd(k2, t13, t69);
    __m512d t72 = _mm512_fmadd_pd(k2, t14, t70);
    __m512d t73 = _mm512_mul_pd(k6, t3);
    __m512d t74 = _mm512_mul_pd(k6, t4);
    __m512d t75 = _mm512_fnmadd_pd(k6, t7, t73);
    __m512d t76 = _mm512_fnmadd_pd(k6, t8, t74);
    __m512d t77 = _mm512_fmadd_pd(k9, t11, t75);
    __m512d t78 = _mm512_fmadd_pd(k9, t12, t76);
    __m512d t79 = _mm512_fmadd_pd(k6, t15, t77);
    __m512d t80 = _mm512_fmadd_pd(k6, t16, t78);
    __m512d t81 = _mm512_add_pd(t71, t80);
    __m512d t82 = _mm512_sub_pd(t72, t79);
    __m512d t83 = _mm512_sub_pd(t71, t80);
    __m512d t84 = _mm512_add_pd(t72, t79);
    __m512d t85 = _mm512_fmadd_pd(k3, t1, r0);
    __m512d t86 = _mm512_fmadd_pd(k3, t2, i0);
    __m512d t87 = _mm512_fmadd_pd(k0, t5, t85);
    __m512d t88 = _mm512_fmadd_pd(k0, t6, t86);
    __m512d t89 = _mm512_fmadd_pd(k2, t9, t87);
    __m512d t90 = _mm512_fmadd_pd(k2, t10, t88);
    __m512d t91 = _mm512_fmadd_pd(k1, t13, t89);
    __m512d t92 = _mm512_fmadd_pd(k1, t14, t90);
    __m512d t93 = _mm512_mul_pd(k7, t3);
    __m512d t94 = _mm512_mul_pd(k7, t4);
    __m512d t95 = _mm512_fnmadd_pd(k4, t7, t93);
    __m512d t96 = _mm512_fnmadd_pd(k4, t8, t94);
    __m512d t97 = _mm512_fmadd_pd(k6, t11, t95);
    __m512d t98 = _mm512_fmadd_pd(k6, t12, t96);
    __m512d t99 = _mm512_fnmadd_pd(k5, t15, t97);
    __m512d t100 = _mm512_fnmadd_pd(k5, t16, t98);
    __m512d t101 = _mm512_add_pd(t91, t100);
    __m512d t102 = _mm512_sub_pd(t92, t99);
    __m512d t103 = _mm512_sub_pd(t91, t100);
    __m512d t104 = _mm512_add_pd(t92, t99);
    _mm512_store_pd(SCR+64, t23);
    _mm512_store_pd(SCR+72, t24);
    _mm512_store_pd(SCR+144, t41);
    _mm512_store_pd(SCR+152, t42);
    _mm512_store_pd(SCR+224, t61);
    _mm512_store_pd(SCR+232, t62);
    _mm512_store_pd(SCR+304, t81);
    _mm512_store_pd(SCR+312, t82);
    _mm512_store_pd(SCR+384, t101);
    _mm512_store_pd(SCR+392, t102);
    _mm512_store_pd(SCR+464, t103);
    _mm512_store_pd(SCR+472, t104);
    _mm512_store_pd(SCR+544, t83);
    _mm512_store_pd(SCR+552, t84);
    _mm512_store_pd(SCR+624, t63);
    _mm512_store_pd(SCR+632, t64);
    _mm512_store_pd(SCR+704, t43);
    _mm512_store_pd(SCR+712, t44);
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+0);
    __m512d i0 = _mm512_load_pd(SCR+8);
    __m512d r1 = _mm512_load_pd(SCR+16);
    __m512d i1 = _mm512_load_pd(SCR+24);
    __m512d r2 = _mm512_load_pd(SCR+32);
    __m512d i2 = _mm512_load_pd(SCR+40);
    __m512d r3 = _mm512_load_pd(SCR+48);
    __m512d i3 = _mm512_load_pd(SCR+56);
    __m512d r4 = _mm512_load_pd(SCR+64);
    __m512d i4 = _mm512_load_pd(SCR+72);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+0*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+0*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+0*s, zr); _mm512_store_pd(X+0*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t21, _mm512_load_pd(C+36*16));
      __m512d zi = _mm512_add_pd(t22, _mm512_load_pd(C+36*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+36*s, zr); _mm512_store_pd(X+36*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t33, _mm512_load_pd(C+27*16));
      __m512d zi = _mm512_add_pd(t34, _mm512_load_pd(C+27*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+27*s, zr); _mm512_store_pd(X+27*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t35, _mm512_load_pd(C+18*16));
      __m512d zi = _mm512_add_pd(t36, _mm512_load_pd(C+18*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+18*s, zr); _mm512_store_pd(X+18*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t23, _mm512_load_pd(C+9*16));
      __m512d zi = _mm512_add_pd(t24, _mm512_load_pd(C+9*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+9*s, zr); _mm512_store_pd(X+9*s+8, zi); }
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+80);
    __m512d i0 = _mm512_load_pd(SCR+88);
    __m512d r1 = _mm512_load_pd(SCR+96);
    __m512d i1 = _mm512_load_pd(SCR+104);
    __m512d r2 = _mm512_load_pd(SCR+112);
    __m512d i2 = _mm512_load_pd(SCR+120);
    __m512d r3 = _mm512_load_pd(SCR+128);
    __m512d i3 = _mm512_load_pd(SCR+136);
    __m512d r4 = _mm512_load_pd(SCR+144);
    __m512d i4 = _mm512_load_pd(SCR+152);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+10*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+10*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+10*s, zr); _mm512_store_pd(X+10*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t21, _mm512_load_pd(C+1*16));
      __m512d zi = _mm512_add_pd(t22, _mm512_load_pd(C+1*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+1*s, zr); _mm512_store_pd(X+1*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t33, _mm512_load_pd(C+37*16));
      __m512d zi = _mm512_add_pd(t34, _mm512_load_pd(C+37*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+37*s, zr); _mm512_store_pd(X+37*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t35, _mm512_load_pd(C+28*16));
      __m512d zi = _mm512_add_pd(t36, _mm512_load_pd(C+28*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+28*s, zr); _mm512_store_pd(X+28*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t23, _mm512_load_pd(C+19*16));
      __m512d zi = _mm512_add_pd(t24, _mm512_load_pd(C+19*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+19*s, zr); _mm512_store_pd(X+19*s+8, zi); }
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+160);
    __m512d i0 = _mm512_load_pd(SCR+168);
    __m512d r1 = _mm512_load_pd(SCR+176);
    __m512d i1 = _mm512_load_pd(SCR+184);
    __m512d r2 = _mm512_load_pd(SCR+192);
    __m512d i2 = _mm512_load_pd(SCR+200);
    __m512d r3 = _mm512_load_pd(SCR+208);
    __m512d i3 = _mm512_load_pd(SCR+216);
    __m512d r4 = _mm512_load_pd(SCR+224);
    __m512d i4 = _mm512_load_pd(SCR+232);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+20*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+20*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+20*s, zr); _mm512_store_pd(X+20*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t21, _mm512_load_pd(C+11*16));
      __m512d zi = _mm512_add_pd(t22, _mm512_load_pd(C+11*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+11*s, zr); _mm512_store_pd(X+11*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t33, _mm512_load_pd(C+2*16));
      __m512d zi = _mm512_add_pd(t34, _mm512_load_pd(C+2*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+2*s, zr); _mm512_store_pd(X+2*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t35, _mm512_load_pd(C+38*16));
      __m512d zi = _mm512_add_pd(t36, _mm512_load_pd(C+38*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+38*s, zr); _mm512_store_pd(X+38*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t23, _mm512_load_pd(C+29*16));
      __m512d zi = _mm512_add_pd(t24, _mm512_load_pd(C+29*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+29*s, zr); _mm512_store_pd(X+29*s+8, zi); }
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+240);
    __m512d i0 = _mm512_load_pd(SCR+248);
    __m512d r1 = _mm512_load_pd(SCR+256);
    __m512d i1 = _mm512_load_pd(SCR+264);
    __m512d r2 = _mm512_load_pd(SCR+272);
    __m512d i2 = _mm512_load_pd(SCR+280);
    __m512d r3 = _mm512_load_pd(SCR+288);
    __m512d i3 = _mm512_load_pd(SCR+296);
    __m512d r4 = _mm512_load_pd(SCR+304);
    __m512d i4 = _mm512_load_pd(SCR+312);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+30*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+30*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+30*s, zr); _mm512_store_pd(X+30*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t21, _mm512_load_pd(C+21*16));
      __m512d zi = _mm512_add_pd(t22, _mm512_load_pd(C+21*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+21*s, zr); _mm512_store_pd(X+21*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t33, _mm512_load_pd(C+12*16));
      __m512d zi = _mm512_add_pd(t34, _mm512_load_pd(C+12*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+12*s, zr); _mm512_store_pd(X+12*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t35, _mm512_load_pd(C+3*16));
      __m512d zi = _mm512_add_pd(t36, _mm512_load_pd(C+3*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+3*s, zr); _mm512_store_pd(X+3*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t23, _mm512_load_pd(C+39*16));
      __m512d zi = _mm512_add_pd(t24, _mm512_load_pd(C+39*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+39*s, zr); _mm512_store_pd(X+39*s+8, zi); }
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+320);
    __m512d i0 = _mm512_load_pd(SCR+328);
    __m512d r1 = _mm512_load_pd(SCR+336);
    __m512d i1 = _mm512_load_pd(SCR+344);
    __m512d r2 = _mm512_load_pd(SCR+352);
    __m512d i2 = _mm512_load_pd(SCR+360);
    __m512d r3 = _mm512_load_pd(SCR+368);
    __m512d i3 = _mm512_load_pd(SCR+376);
    __m512d r4 = _mm512_load_pd(SCR+384);
    __m512d i4 = _mm512_load_pd(SCR+392);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+40*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+40*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+40*s, zr); _mm512_store_pd(X+40*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t21, _mm512_load_pd(C+31*16));
      __m512d zi = _mm512_add_pd(t22, _mm512_load_pd(C+31*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+31*s, zr); _mm512_store_pd(X+31*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t33, _mm512_load_pd(C+22*16));
      __m512d zi = _mm512_add_pd(t34, _mm512_load_pd(C+22*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+22*s, zr); _mm512_store_pd(X+22*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t35, _mm512_load_pd(C+13*16));
      __m512d zi = _mm512_add_pd(t36, _mm512_load_pd(C+13*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+13*s, zr); _mm512_store_pd(X+13*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t23, _mm512_load_pd(C+4*16));
      __m512d zi = _mm512_add_pd(t24, _mm512_load_pd(C+4*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+4*s, zr); _mm512_store_pd(X+4*s+8, zi); }
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+400);
    __m512d i0 = _mm512_load_pd(SCR+408);
    __m512d r1 = _mm512_load_pd(SCR+416);
    __m512d i1 = _mm512_load_pd(SCR+424);
    __m512d r2 = _mm512_load_pd(SCR+432);
    __m512d i2 = _mm512_load_pd(SCR+440);
    __m512d r3 = _mm512_load_pd(SCR+448);
    __m512d i3 = _mm512_load_pd(SCR+456);
    __m512d r4 = _mm512_load_pd(SCR+464);
    __m512d i4 = _mm512_load_pd(SCR+472);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+5*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+5*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+5*s, zr); _mm512_store_pd(X+5*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t21, _mm512_load_pd(C+41*16));
      __m512d zi = _mm512_add_pd(t22, _mm512_load_pd(C+41*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+41*s, zr); _mm512_store_pd(X+41*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t33, _mm512_load_pd(C+32*16));
      __m512d zi = _mm512_add_pd(t34, _mm512_load_pd(C+32*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+32*s, zr); _mm512_store_pd(X+32*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t35, _mm512_load_pd(C+23*16));
      __m512d zi = _mm512_add_pd(t36, _mm512_load_pd(C+23*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+23*s, zr); _mm512_store_pd(X+23*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t23, _mm512_load_pd(C+14*16));
      __m512d zi = _mm512_add_pd(t24, _mm512_load_pd(C+14*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+14*s, zr); _mm512_store_pd(X+14*s+8, zi); }
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+480);
    __m512d i0 = _mm512_load_pd(SCR+488);
    __m512d r1 = _mm512_load_pd(SCR+496);
    __m512d i1 = _mm512_load_pd(SCR+504);
    __m512d r2 = _mm512_load_pd(SCR+512);
    __m512d i2 = _mm512_load_pd(SCR+520);
    __m512d r3 = _mm512_load_pd(SCR+528);
    __m512d i3 = _mm512_load_pd(SCR+536);
    __m512d r4 = _mm512_load_pd(SCR+544);
    __m512d i4 = _mm512_load_pd(SCR+552);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+15*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+15*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+15*s, zr); _mm512_store_pd(X+15*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t21, _mm512_load_pd(C+6*16));
      __m512d zi = _mm512_add_pd(t22, _mm512_load_pd(C+6*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+6*s, zr); _mm512_store_pd(X+6*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t33, _mm512_load_pd(C+42*16));
      __m512d zi = _mm512_add_pd(t34, _mm512_load_pd(C+42*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+42*s, zr); _mm512_store_pd(X+42*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t35, _mm512_load_pd(C+33*16));
      __m512d zi = _mm512_add_pd(t36, _mm512_load_pd(C+33*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+33*s, zr); _mm512_store_pd(X+33*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t23, _mm512_load_pd(C+24*16));
      __m512d zi = _mm512_add_pd(t24, _mm512_load_pd(C+24*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+24*s, zr); _mm512_store_pd(X+24*s+8, zi); }
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+560);
    __m512d i0 = _mm512_load_pd(SCR+568);
    __m512d r1 = _mm512_load_pd(SCR+576);
    __m512d i1 = _mm512_load_pd(SCR+584);
    __m512d r2 = _mm512_load_pd(SCR+592);
    __m512d i2 = _mm512_load_pd(SCR+600);
    __m512d r3 = _mm512_load_pd(SCR+608);
    __m512d i3 = _mm512_load_pd(SCR+616);
    __m512d r4 = _mm512_load_pd(SCR+624);
    __m512d i4 = _mm512_load_pd(SCR+632);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+25*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+25*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+25*s, zr); _mm512_store_pd(X+25*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t21, _mm512_load_pd(C+16*16));
      __m512d zi = _mm512_add_pd(t22, _mm512_load_pd(C+16*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+16*s, zr); _mm512_store_pd(X+16*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t33, _mm512_load_pd(C+7*16));
      __m512d zi = _mm512_add_pd(t34, _mm512_load_pd(C+7*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+7*s, zr); _mm512_store_pd(X+7*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t35, _mm512_load_pd(C+43*16));
      __m512d zi = _mm512_add_pd(t36, _mm512_load_pd(C+43*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+43*s, zr); _mm512_store_pd(X+43*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t23, _mm512_load_pd(C+34*16));
      __m512d zi = _mm512_add_pd(t24, _mm512_load_pd(C+34*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+34*s, zr); _mm512_store_pd(X+34*s+8, zi); }
    }
    {
    const __m512d k0 = _mm512_set1_pd(0x1.3c6ef372fe950p-2);
    const __m512d k1 = _mm512_set1_pd(-0x1.9e3779b97f4a7p-1);
    const __m512d k2 = _mm512_set1_pd(0x1.e6f0e134454ffp-1);
    const __m512d k3 = _mm512_set1_pd(0x1.2cf2304755a5fp-1);
    __m512d r0 = _mm512_load_pd(SCR+640);
    __m512d i0 = _mm512_load_pd(SCR+648);
    __m512d r1 = _mm512_load_pd(SCR+656);
    __m512d i1 = _mm512_load_pd(SCR+664);
    __m512d r2 = _mm512_load_pd(SCR+672);
    __m512d i2 = _mm512_load_pd(SCR+680);
    __m512d r3 = _mm512_load_pd(SCR+688);
    __m512d i3 = _mm512_load_pd(SCR+696);
    __m512d r4 = _mm512_load_pd(SCR+704);
    __m512d i4 = _mm512_load_pd(SCR+712);
    __m512d t1 = _mm512_add_pd(r1, r4);
    __m512d t2 = _mm512_add_pd(i1, i4);
    __m512d t3 = _mm512_sub_pd(r1, r4);
    __m512d t4 = _mm512_sub_pd(i1, i4);
    __m512d t5 = _mm512_add_pd(r2, r3);
    __m512d t6 = _mm512_add_pd(i2, i3);
    __m512d t7 = _mm512_sub_pd(r2, r3);
    __m512d t8 = _mm512_sub_pd(i2, i3);
    __m512d t9 = _mm512_add_pd(r0, t1);
    __m512d t10 = _mm512_add_pd(i0, t2);
    __m512d t11 = _mm512_add_pd(t9, t5);
    __m512d t12 = _mm512_add_pd(t10, t6);
    __m512d t13 = _mm512_fmadd_pd(k0, t1, r0);
    __m512d t14 = _mm512_fmadd_pd(k0, t2, i0);
    __m512d t15 = _mm512_fmadd_pd(k1, t5, t13);
    __m512d t16 = _mm512_fmadd_pd(k1, t6, t14);
    __m512d t17 = _mm512_mul_pd(k2, t3);
    __m512d t18 = _mm512_mul_pd(k2, t4);
    __m512d t19 = _mm512_fmadd_pd(k3, t7, t17);
    __m512d t20 = _mm512_fmadd_pd(k3, t8, t18);
    __m512d t21 = _mm512_add_pd(t15, t20);
    __m512d t22 = _mm512_sub_pd(t16, t19);
    __m512d t23 = _mm512_sub_pd(t15, t20);
    __m512d t24 = _mm512_add_pd(t16, t19);
    __m512d t25 = _mm512_fmadd_pd(k1, t1, r0);
    __m512d t26 = _mm512_fmadd_pd(k1, t2, i0);
    __m512d t27 = _mm512_fmadd_pd(k0, t5, t25);
    __m512d t28 = _mm512_fmadd_pd(k0, t6, t26);
    __m512d t29 = _mm512_mul_pd(k3, t3);
    __m512d t30 = _mm512_mul_pd(k3, t4);
    __m512d t31 = _mm512_fnmadd_pd(k2, t7, t29);
    __m512d t32 = _mm512_fnmadd_pd(k2, t8, t30);
    __m512d t33 = _mm512_add_pd(t27, t32);
    __m512d t34 = _mm512_sub_pd(t28, t31);
    __m512d t35 = _mm512_sub_pd(t27, t32);
    __m512d t36 = _mm512_add_pd(t28, t31);
    { __m512d zr = _mm512_add_pd(t11, _mm512_load_pd(C+35*16));
      __m512d zi = _mm512_add_pd(t12, _mm512_load_pd(C+35*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+35*s, zr); _mm512_store_pd(X+35*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t21, _mm512_load_pd(C+26*16));
      __m512d zi = _mm512_add_pd(t22, _mm512_load_pd(C+26*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+26*s, zr); _mm512_store_pd(X+26*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t33, _mm512_load_pd(C+17*16));
      __m512d zi = _mm512_add_pd(t34, _mm512_load_pd(C+17*16+8));
      map2(zr, zi, &zr, &zi);
      _mm512_store_pd(X+17*s, zr); _mm512_store_pd(X+17*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t35, _mm512_load_pd(C+8*16));
      __m512d zi = _mm512_add_pd(t36, _mm512_load_pd(C+8*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+8*s, zr); _mm512_store_pd(X+8*s+8, zi); }
    { __m512d zr = _mm512_add_pd(t23, _mm512_load_pd(C+44*16));
      __m512d zi = _mm512_add_pd(t24, _mm512_load_pd(C+44*16+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X+44*s, zr); _mm512_store_pd(X+44*s+8, zi); }
    }
}
#define PFIN_45 0

static void __attribute__((noinline)) dft45_one(double* X, long es){ dft45(X, es); }
#if PFIN_45
static void __attribute__((noinline)) dft45_onesq(double* X){ dft45pfz(X, 1); }
static void __attribute__((noinline)) dft45_onem(double* X, long es, const double* Ct){ dft45m(X, es, Ct); }
#elif 0
static void __attribute__((noinline)) dft45_onem_unused(double* X, long es, const double* Ct){ dft45mpf(X, es, Ct); }
#else
#if 1
static void __attribute__((noinline)) dft45_onesq(double* X){ dft45sq(X, 1); }
#else
static void __attribute__((noinline)) dft45_onesq(double* X){ dft45(X, 1); }
#endif
static void __attribute__((noinline)) dft45_onem(double* X, long es, const double* Ct){ dft45m(X, es, Ct); }
#endif
static void dft45_sweep_zy(double* restrict X){
#if PFCOMP
    const long PB = (2025*2 + 45 - 1)/45;   /* 128B-blocks of next plane per y-codelet */
    for(long x=0; x<45; x++){
        double* P = X + x*2025*16;
        const char* nxt = (const char*)(P + 2025*16);
        for(long y=0; y<45; y++) dft45_one(P + y*45*16, 1);
        for(long z=0; z<45; z++){
            if(x+1 < 45){
                const char* q = nxt + z*PB*128;
                for(long l=0; l<PB; l++) _mm_prefetch(q + l*128, _MM_HINT_T0);
            }
            dft45_one(P + z*16, 45);
        }
    }
#else
    for(long x=0; x<45; x++){
        double* P = X + x*2025*16;
        for(long y=0; y<45; y++) dft45_onesq(P + y*45*16);
        for(long z=0; z<45; z++) dft45_one(P + z*16, 45);
    }
#endif
}
static void dft45_sweep_x_map(double* restrict X, const double* restrict Ct){
    for(long p=0; p<2025; p++) dft45_onem(X + p*16, 2025, Ct + p*45*16);
}
static void dft45_sweep_x_plain(double* restrict X){
    for(long p=0; p<2025; p++) dft45_one(X + p*16, 2025);
}
static void dft45_sweep_zy_ms(double* restrict X, const double* restrict C){
    for(long x=0; x<45; x++){
        double* P = X + x*2025*16;
        for(long y=0; y<45; y++) dft45_onesq(P + y*45*16);
        for(long z=0; z<45; z++) dft45_one(P + z*16, 45);
        if(x) mapslab(X + (x-1)*2025*16, C + (x-1)*2025*16, 2025);
    }
    mapslab(X + (45-1)*2025*16, C + (45-1)*2025*16, 2025);
}
/* plane-wise ingest/output (padded plane stride 2025) */
static void ingest_45(const double* const* src, double* G){
    for(long x=0; x<45; x++){
        const long base = x*2025;
        double* Gp = G + x*2025*16;
        for(long e=0; e<2024; e+=4){
            __m512d r0=_mm512_loadu_pd(src[0]+2*(base+e)), r1=_mm512_loadu_pd(src[1]+2*(base+e));
            __m512d r2=_mm512_loadu_pd(src[2]+2*(base+e)), r3=_mm512_loadu_pd(src[3]+2*(base+e));
            __m512d r4=_mm512_loadu_pd(src[4]+2*(base+e)), r5=_mm512_loadu_pd(src[5]+2*(base+e));
            __m512d r6=_mm512_loadu_pd(src[6]+2*(base+e)), r7=_mm512_loadu_pd(src[7]+2*(base+e));
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            _mm512_store_pd(Gp+e*16,    o0); _mm512_store_pd(Gp+e*16+8,  o1);
            _mm512_store_pd(Gp+e*16+16, o2); _mm512_store_pd(Gp+e*16+24, o3);
            _mm512_store_pd(Gp+e*16+32, o4); _mm512_store_pd(Gp+e*16+40, o5);
            _mm512_store_pd(Gp+e*16+48, o6); _mm512_store_pd(Gp+e*16+56, o7);
        }
#if 1 > 0
        {
            const long e = 2024;
            const __mmask8 mk = (__mmask8)((1u<<(2*1))-1u);
            __m512d r0=_mm512_maskz_loadu_pd(mk, src[0]+2*(base+e)), r1=_mm512_maskz_loadu_pd(mk, src[1]+2*(base+e));
            __m512d r2=_mm512_maskz_loadu_pd(mk, src[2]+2*(base+e)), r3=_mm512_maskz_loadu_pd(mk, src[3]+2*(base+e));
            __m512d r4=_mm512_maskz_loadu_pd(mk, src[4]+2*(base+e)), r5=_mm512_maskz_loadu_pd(mk, src[5]+2*(base+e));
            __m512d r6=_mm512_maskz_loadu_pd(mk, src[6]+2*(base+e)), r7=_mm512_maskz_loadu_pd(mk, src[7]+2*(base+e));
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d A[8]; A[0]=o0;A[1]=o1;A[2]=o2;A[3]=o3;A[4]=o4;A[5]=o5;A[6]=o6;A[7]=o7;
            for(int q=0;q<2*1;q++) _mm512_store_pd(Gp+e*16+q*8, A[q]);
        }
#endif
    }
}
static void output_45(const double* G, double* const* dst, int nv){
    for(long x=0; x<45; x++){
        const long base = x*2025;
        const double* Gp = G + x*2025*16;
        for(long e=0; e<2024; e+=4){
            __m512d i0=_mm512_load_pd(Gp+e*16),    i1=_mm512_load_pd(Gp+e*16+8);
            __m512d i2=_mm512_load_pd(Gp+e*16+16), i3=_mm512_load_pd(Gp+e*16+24);
            __m512d i4=_mm512_load_pd(Gp+e*16+32), i5=_mm512_load_pd(Gp+e*16+40);
            __m512d i6=_mm512_load_pd(Gp+e*16+48), i7=_mm512_load_pd(Gp+e*16+56);
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(i0,i1,i2,i3,i4,i5,i6,i7,o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
            for(int v=0; v<nv; v++) _mm512_storeu_pd(dst[v]+2*(base+e), *O[v]);
        }
#if 1 > 0
        {
            const long e = 2024;
            const __mmask8 mk = (__mmask8)((1u<<(2*1))-1u);
            __m512d A[8];
            for(int q=0;q<2*1;q++) A[q] = _mm512_load_pd(Gp+e*16+q*8);
            for(int q=2*1;q<8;q++) A[q] = _mm512_setzero_pd();
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            TR8(A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],o0,o1,o2,o3,o4,o5,o6,o7);
            __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
            for(int v=0; v<nv; v++) _mm512_mask_storeu_pd(dst[v]+2*(base+e), mk, *O[v]);
        }
#endif
    }
}


static double* Xg_45 = 0;
static double* Cg_45 = 0;
void hot_45(long n){
    if(!Xg_45){ Xg_45 = alloc_huge_st(45*2025*16*8); Cg_45 = alloc_huge_st(91125*16*8); }
    for(long i=0;i<45*2025*16;i++) Xg_45[i] = 0.5 + 1e-6*(i%97);
    for(long r=0;r<n;r++){
        double* P = Xg_45;
        for(long y=0; y<45; y++) dft45_one(P + y*45*16, 1);
        for(long z=0; z<45; z++) dft45_one(P + z*16, 45);
        if((r&1)==1) for(long i=0;i<2025*16;i++) Xg_45[i] = 0.5 + 1e-6*(i%97);
    }
}
void hot2_45(long which){
    if(!Xg_45){ Xg_45 = alloc_huge_st((91125+64*45)*16*8); Cg_45 = alloc_huge_st(91125*16*8); }
    double* P = Xg_45;
    if(which==99){ for(long i=0;i<2025*16;i++) P[i] = 0.5 + 1e-6*(i%97); return; }
    if(which==0 || which==2) for(long y=0; y<45; y++) dft45_one(P + y*45*16, 1);
    if(which==1 || which==2) for(long z=0; z<45; z++) dft45_one(P + z*16, 45);
}
void bsweep_45(long which, long n){
    if(!Xg_45){ Xg_45 = alloc_huge_st(45*2025*16*8); Cg_45 = alloc_huge_st(91125*16*8); }
    for(long i=0;i<45*2025*16;i++) Xg_45[i] = 0.5 + 1e-6*(i%97);
    for(long i=0;i<91125*16;i++) Cg_45[i] = 0.01;
    for(long r=0;r<n;r++){
        if(which==0) dft45_sweep_zy(Xg_45);
        else if(which==2) dft45_sweep_x_map(Xg_45, Cg_45);
        if((r&3)==3) for(long i=0;i<45*2025*16;i+=997) Xg_45[i] = 0.5;
    }
}
void diag_45(long which, long n){
    if(!Xg_45){ Xg_45 = alloc_huge_st(45*2025*16*8); Cg_45 = alloc_huge_st(45*2025*16*8); }
    for(long i=0;i<45*2025*16;i++){ Xg_45[i] = 0.5 + 1e-6*(i%97); Cg_45[i] = 0.01; }
    for(long r=0;r<n;r++){
        if(which==0){ for(long x=0;x<45;x++) mapslab(Xg_45 + x*2025*16, Cg_45 + x*2025*16, 2025); }
        else if(which==1) dft45_sweep_zy(Xg_45);
        else dft45_sweep_x_plain(Xg_45);
        if((r&1)==1) for(long i=0;i<45*2025*16;i+=997) Xg_45[i] = 0.5;
    }
}
void run_45(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    const long NE = 91125;
    if(!Xg_45){ Xg_45 = alloc_huge_st(45*2025*16*8); Cg_45 = alloc_huge_st(45*2025*16*8); }
    double* X = Xg_45; double* Ct = Cg_45;
    for(long g0=0; g0<B; g0+=8){
        int nv = (int)((B - g0) < 8 ? (B - g0) : 8);
        const double* src[8]; const double* csrc[8];
        double* d1[8]; double* dm[8];
        for(int v=0; v<8; v++){
            int vv = v < nv ? v : 0;
            src[v] = x0 + (g0+vv)*2*NE; csrc[v] = c + (g0+vv)*2*NE;
            if(v<nv){ d1[v] = out1 + (g0+v)*2*NE; dm[v] = outm + (g0+v)*2*NE; }
        }
#if XFIRST_FLAG
        ingest_45(csrc, Ct);   /* c in padded group layout */
        ingest_45(src, X);
        for(long t=0; t<m; t++){
            dft45_sweep_x_plain(X);
            dft45_sweep_zy_ms(X, Ct);
            if(t==0 && m>1) output_45(X, d1, nv);
        }
#else
        ingest_45(csrc, X);
        for(long p=0; p<2025; p++)
            for(long k=0; k<45; k++){
                _mm512_store_pd(Ct + (p*45+k)*16,     _mm512_load_pd(X + (k*2025+p)*16));
                _mm512_store_pd(Ct + (p*45+k)*16 + 8, _mm512_load_pd(X + (k*2025+p)*16 + 8));
            }
        ingest_45(src, X);
        for(long t=0; t<m; t++){
            dft45_sweep_zy(X);
            dft45_sweep_x_map(X, Ct);
            if(t==0 && m>1) output_45(X, d1, nv);
        }
#endif
        output_45(X, dm, nv);
        if(m==1) output_45(X, d1, nv);
    }
}

static __attribute__((always_inline)) inline void dft13(double* restrict X, long es, int dopf){
    const long s = es*16;
    double AB[48*8] ALIGN64;
    double Escr[12*8] ALIGN64;
    __m512d x0r = _mm512_load_pd(X);
    if(dopf){ _mm_prefetch((const char*)(X+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+216), _MM_HINT_T0); }
    __m512d x0i = _mm512_load_pd(X+8);
    { __m512d c1 = _mm512_set1_pd(0x1.c55a7e00740e9p-1), c2 = _mm512_set1_pd(0x1.22d961ea71119p-1), c3 = _mm512_set1_pd(0x1.edb7debaa3ed5p-4), c4 = _mm512_set1_pd(-0x1.6b1d8b2365d9ep-2), c5 = _mm512_set1_pd(-0x1.7f3ccd0032e0dp-1), c6 = _mm512_set1_pd(-0x1.f11f493053d00p-1);
    __m512d e1r = x0r, e1i = x0i;
    __m512d e2r = x0r, e2i = x0i;
    __m512d e3r = x0r, e3i = x0i;
    __m512d e4r = x0r, e4i = x0i;
    __m512d e5r = x0r, e5i = x0i;
    __m512d e6r = x0r, e6i = x0i;
    __m512d sr = x0r, si = x0i;
    { __m512d pr = _mm512_load_pd(X+1*s), qr = _mm512_load_pd(X+12*s);
      if(dopf){ _mm_prefetch((const char*)(X+1*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+1*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+12*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+12*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+1*s+8), qi = _mm512_load_pd(X+12*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+0, ur);    _mm512_store_pd(AB+8, ui);
      _mm512_store_pd(AB+16, vr); _mm512_store_pd(AB+24, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c1, ur, e1r); e1i = _mm512_fmadd_pd(c1, ui, e1i);
      e2r = _mm512_fmadd_pd(c2, ur, e2r); e2i = _mm512_fmadd_pd(c2, ui, e2i);
      e3r = _mm512_fmadd_pd(c3, ur, e3r); e3i = _mm512_fmadd_pd(c3, ui, e3i);
      e4r = _mm512_fmadd_pd(c4, ur, e4r); e4i = _mm512_fmadd_pd(c4, ui, e4i);
      e5r = _mm512_fmadd_pd(c5, ur, e5r); e5i = _mm512_fmadd_pd(c5, ui, e5i);
      e6r = _mm512_fmadd_pd(c6, ur, e6r); e6i = _mm512_fmadd_pd(c6, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+2*s), qr = _mm512_load_pd(X+11*s);
      if(dopf){ _mm_prefetch((const char*)(X+2*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+2*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+11*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+11*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+2*s+8), qi = _mm512_load_pd(X+11*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+32, ur);    _mm512_store_pd(AB+40, ui);
      _mm512_store_pd(AB+48, vr); _mm512_store_pd(AB+56, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c2, ur, e1r); e1i = _mm512_fmadd_pd(c2, ui, e1i);
      e2r = _mm512_fmadd_pd(c4, ur, e2r); e2i = _mm512_fmadd_pd(c4, ui, e2i);
      e3r = _mm512_fmadd_pd(c6, ur, e3r); e3i = _mm512_fmadd_pd(c6, ui, e3i);
      e4r = _mm512_fmadd_pd(c5, ur, e4r); e4i = _mm512_fmadd_pd(c5, ui, e4i);
      e5r = _mm512_fmadd_pd(c3, ur, e5r); e5i = _mm512_fmadd_pd(c3, ui, e5i);
      e6r = _mm512_fmadd_pd(c1, ur, e6r); e6i = _mm512_fmadd_pd(c1, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+3*s), qr = _mm512_load_pd(X+10*s);
      if(dopf){ _mm_prefetch((const char*)(X+3*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+3*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+10*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+10*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+3*s+8), qi = _mm512_load_pd(X+10*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+64, ur);    _mm512_store_pd(AB+72, ui);
      _mm512_store_pd(AB+80, vr); _mm512_store_pd(AB+88, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c3, ur, e1r); e1i = _mm512_fmadd_pd(c3, ui, e1i);
      e2r = _mm512_fmadd_pd(c6, ur, e2r); e2i = _mm512_fmadd_pd(c6, ui, e2i);
      e3r = _mm512_fmadd_pd(c4, ur, e3r); e3i = _mm512_fmadd_pd(c4, ui, e3i);
      e4r = _mm512_fmadd_pd(c1, ur, e4r); e4i = _mm512_fmadd_pd(c1, ui, e4i);
      e5r = _mm512_fmadd_pd(c2, ur, e5r); e5i = _mm512_fmadd_pd(c2, ui, e5i);
      e6r = _mm512_fmadd_pd(c5, ur, e6r); e6i = _mm512_fmadd_pd(c5, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+4*s), qr = _mm512_load_pd(X+9*s);
      if(dopf){ _mm_prefetch((const char*)(X+4*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+4*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+9*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+9*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+4*s+8), qi = _mm512_load_pd(X+9*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+96, ur);    _mm512_store_pd(AB+104, ui);
      _mm512_store_pd(AB+112, vr); _mm512_store_pd(AB+120, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c4, ur, e1r); e1i = _mm512_fmadd_pd(c4, ui, e1i);
      e2r = _mm512_fmadd_pd(c5, ur, e2r); e2i = _mm512_fmadd_pd(c5, ui, e2i);
      e3r = _mm512_fmadd_pd(c1, ur, e3r); e3i = _mm512_fmadd_pd(c1, ui, e3i);
      e4r = _mm512_fmadd_pd(c3, ur, e4r); e4i = _mm512_fmadd_pd(c3, ui, e4i);
      e5r = _mm512_fmadd_pd(c6, ur, e5r); e5i = _mm512_fmadd_pd(c6, ui, e5i);
      e6r = _mm512_fmadd_pd(c2, ur, e6r); e6i = _mm512_fmadd_pd(c2, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+5*s), qr = _mm512_load_pd(X+8*s);
      if(dopf){ _mm_prefetch((const char*)(X+5*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+5*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+8*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+8*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+5*s+8), qi = _mm512_load_pd(X+8*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+128, ur);    _mm512_store_pd(AB+136, ui);
      _mm512_store_pd(AB+144, vr); _mm512_store_pd(AB+152, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c5, ur, e1r); e1i = _mm512_fmadd_pd(c5, ui, e1i);
      e2r = _mm512_fmadd_pd(c3, ur, e2r); e2i = _mm512_fmadd_pd(c3, ui, e2i);
      e3r = _mm512_fmadd_pd(c2, ur, e3r); e3i = _mm512_fmadd_pd(c2, ui, e3i);
      e4r = _mm512_fmadd_pd(c6, ur, e4r); e4i = _mm512_fmadd_pd(c6, ui, e4i);
      e5r = _mm512_fmadd_pd(c1, ur, e5r); e5i = _mm512_fmadd_pd(c1, ui, e5i);
      e6r = _mm512_fmadd_pd(c4, ur, e6r); e6i = _mm512_fmadd_pd(c4, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+6*s), qr = _mm512_load_pd(X+7*s);
      if(dopf){ _mm_prefetch((const char*)(X+6*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+6*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+7*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+7*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+6*s+8), qi = _mm512_load_pd(X+7*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+160, ur);    _mm512_store_pd(AB+168, ui);
      _mm512_store_pd(AB+176, vr); _mm512_store_pd(AB+184, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c6, ur, e1r); e1i = _mm512_fmadd_pd(c6, ui, e1i);
      e2r = _mm512_fmadd_pd(c1, ur, e2r); e2i = _mm512_fmadd_pd(c1, ui, e2i);
      e3r = _mm512_fmadd_pd(c5, ur, e3r); e3i = _mm512_fmadd_pd(c5, ui, e3i);
      e4r = _mm512_fmadd_pd(c2, ur, e4r); e4i = _mm512_fmadd_pd(c2, ui, e4i);
      e5r = _mm512_fmadd_pd(c4, ur, e5r); e5i = _mm512_fmadd_pd(c4, ui, e5i);
      e6r = _mm512_fmadd_pd(c3, ur, e6r); e6i = _mm512_fmadd_pd(c3, ui, e6i);
    }
    _mm512_store_pd(X, sr); _mm512_store_pd(X+8, si);
    _mm512_store_pd(Escr+0, e1r); _mm512_store_pd(Escr+8, e1i);
    _mm512_store_pd(Escr+16, e2r); _mm512_store_pd(Escr+24, e2i);
    _mm512_store_pd(Escr+32, e3r); _mm512_store_pd(Escr+40, e3i);
    _mm512_store_pd(Escr+48, e4r); _mm512_store_pd(Escr+56, e4i);
    _mm512_store_pd(Escr+64, e5r); _mm512_store_pd(Escr+72, e5i);
    _mm512_store_pd(Escr+80, e6r); _mm512_store_pd(Escr+88, e6i);
    }
    { __m512d s1 = _mm512_set1_pd(0x1.dbe064267c47bp-2), s2 = _mm512_set1_pd(0x1.a55e242a4c3d2p-1), s3 = _mm512_set1_pd(0x1.fc44566966769p-1), s4 = _mm512_set1_pd(0x1.deba72ef20147p-1), s5 = _mm512_set1_pd(0x1.5384d024c2f84p-1), s6 = _mm512_set1_pd(0x1.ea1e54bc48dbcp-3);
    __m512d o1r = _mm512_setzero_pd(), o1i = _mm512_setzero_pd();
    __m512d o2r = _mm512_setzero_pd(), o2i = _mm512_setzero_pd();
    __m512d o3r = _mm512_setzero_pd(), o3i = _mm512_setzero_pd();
    __m512d o4r = _mm512_setzero_pd(), o4i = _mm512_setzero_pd();
    __m512d o5r = _mm512_setzero_pd(), o5i = _mm512_setzero_pd();
    __m512d o6r = _mm512_setzero_pd(), o6i = _mm512_setzero_pd();
    { __m512d vr = _mm512_load_pd(AB+16);
      __m512d vi = _mm512_load_pd(AB+24);
      o1r = _mm512_fmadd_pd(s1, vr, o1r); o1i = _mm512_fmadd_pd(s1, vi, o1i);
      o2r = _mm512_fmadd_pd(s2, vr, o2r); o2i = _mm512_fmadd_pd(s2, vi, o2i);
      o3r = _mm512_fmadd_pd(s3, vr, o3r); o3i = _mm512_fmadd_pd(s3, vi, o3i);
      o4r = _mm512_fmadd_pd(s4, vr, o4r); o4i = _mm512_fmadd_pd(s4, vi, o4i);
      o5r = _mm512_fmadd_pd(s5, vr, o5r); o5i = _mm512_fmadd_pd(s5, vi, o5i);
      o6r = _mm512_fmadd_pd(s6, vr, o6r); o6i = _mm512_fmadd_pd(s6, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+48);
      __m512d vi = _mm512_load_pd(AB+56);
      o1r = _mm512_fmadd_pd(s2, vr, o1r); o1i = _mm512_fmadd_pd(s2, vi, o1i);
      o2r = _mm512_fmadd_pd(s4, vr, o2r); o2i = _mm512_fmadd_pd(s4, vi, o2i);
      o3r = _mm512_fmadd_pd(s6, vr, o3r); o3i = _mm512_fmadd_pd(s6, vi, o3i);
      o4r = _mm512_fnmadd_pd(s5, vr, o4r); o4i = _mm512_fnmadd_pd(s5, vi, o4i);
      o5r = _mm512_fnmadd_pd(s3, vr, o5r); o5i = _mm512_fnmadd_pd(s3, vi, o5i);
      o6r = _mm512_fnmadd_pd(s1, vr, o6r); o6i = _mm512_fnmadd_pd(s1, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+80);
      __m512d vi = _mm512_load_pd(AB+88);
      o1r = _mm512_fmadd_pd(s3, vr, o1r); o1i = _mm512_fmadd_pd(s3, vi, o1i);
      o2r = _mm512_fmadd_pd(s6, vr, o2r); o2i = _mm512_fmadd_pd(s6, vi, o2i);
      o3r = _mm512_fnmadd_pd(s4, vr, o3r); o3i = _mm512_fnmadd_pd(s4, vi, o3i);
      o4r = _mm512_fnmadd_pd(s1, vr, o4r); o4i = _mm512_fnmadd_pd(s1, vi, o4i);
      o5r = _mm512_fmadd_pd(s2, vr, o5r); o5i = _mm512_fmadd_pd(s2, vi, o5i);
      o6r = _mm512_fmadd_pd(s5, vr, o6r); o6i = _mm512_fmadd_pd(s5, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+112);
      __m512d vi = _mm512_load_pd(AB+120);
      o1r = _mm512_fmadd_pd(s4, vr, o1r); o1i = _mm512_fmadd_pd(s4, vi, o1i);
      o2r = _mm512_fnmadd_pd(s5, vr, o2r); o2i = _mm512_fnmadd_pd(s5, vi, o2i);
      o3r = _mm512_fnmadd_pd(s1, vr, o3r); o3i = _mm512_fnmadd_pd(s1, vi, o3i);
      o4r = _mm512_fmadd_pd(s3, vr, o4r); o4i = _mm512_fmadd_pd(s3, vi, o4i);
      o5r = _mm512_fnmadd_pd(s6, vr, o5r); o5i = _mm512_fnmadd_pd(s6, vi, o5i);
      o6r = _mm512_fnmadd_pd(s2, vr, o6r); o6i = _mm512_fnmadd_pd(s2, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+144);
      __m512d vi = _mm512_load_pd(AB+152);
      o1r = _mm512_fmadd_pd(s5, vr, o1r); o1i = _mm512_fmadd_pd(s5, vi, o1i);
      o2r = _mm512_fnmadd_pd(s3, vr, o2r); o2i = _mm512_fnmadd_pd(s3, vi, o2i);
      o3r = _mm512_fmadd_pd(s2, vr, o3r); o3i = _mm512_fmadd_pd(s2, vi, o3i);
      o4r = _mm512_fnmadd_pd(s6, vr, o4r); o4i = _mm512_fnmadd_pd(s6, vi, o4i);
      o5r = _mm512_fnmadd_pd(s1, vr, o5r); o5i = _mm512_fnmadd_pd(s1, vi, o5i);
      o6r = _mm512_fmadd_pd(s4, vr, o6r); o6i = _mm512_fmadd_pd(s4, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+176);
      __m512d vi = _mm512_load_pd(AB+184);
      o1r = _mm512_fmadd_pd(s6, vr, o1r); o1i = _mm512_fmadd_pd(s6, vi, o1i);
      o2r = _mm512_fnmadd_pd(s1, vr, o2r); o2i = _mm512_fnmadd_pd(s1, vi, o2i);
      o3r = _mm512_fmadd_pd(s5, vr, o3r); o3i = _mm512_fmadd_pd(s5, vi, o3i);
      o4r = _mm512_fnmadd_pd(s2, vr, o4r); o4i = _mm512_fnmadd_pd(s2, vi, o4i);
      o5r = _mm512_fmadd_pd(s4, vr, o5r); o5i = _mm512_fmadd_pd(s4, vi, o5i);
      o6r = _mm512_fnmadd_pd(s3, vr, o6r); o6i = _mm512_fnmadd_pd(s3, vi, o6i);
    }
    { __m512d er = _mm512_load_pd(Escr+0), ei = _mm512_load_pd(Escr+8);
      _mm512_store_pd(X+1*s,   _mm512_add_pd(er, o1i));
      _mm512_store_pd(X+1*s+8, _mm512_sub_pd(ei, o1r));
      _mm512_store_pd(X+12*s,   _mm512_sub_pd(er, o1i));
      _mm512_store_pd(X+12*s+8, _mm512_add_pd(ei, o1r)); }
    { __m512d er = _mm512_load_pd(Escr+16), ei = _mm512_load_pd(Escr+24);
      _mm512_store_pd(X+2*s,   _mm512_add_pd(er, o2i));
      _mm512_store_pd(X+2*s+8, _mm512_sub_pd(ei, o2r));
      _mm512_store_pd(X+11*s,   _mm512_sub_pd(er, o2i));
      _mm512_store_pd(X+11*s+8, _mm512_add_pd(ei, o2r)); }
    { __m512d er = _mm512_load_pd(Escr+32), ei = _mm512_load_pd(Escr+40);
      _mm512_store_pd(X+3*s,   _mm512_add_pd(er, o3i));
      _mm512_store_pd(X+3*s+8, _mm512_sub_pd(ei, o3r));
      _mm512_store_pd(X+10*s,   _mm512_sub_pd(er, o3i));
      _mm512_store_pd(X+10*s+8, _mm512_add_pd(ei, o3r)); }
    { __m512d er = _mm512_load_pd(Escr+48), ei = _mm512_load_pd(Escr+56);
      _mm512_store_pd(X+4*s,   _mm512_add_pd(er, o4i));
      _mm512_store_pd(X+4*s+8, _mm512_sub_pd(ei, o4r));
      _mm512_store_pd(X+9*s,   _mm512_sub_pd(er, o4i));
      _mm512_store_pd(X+9*s+8, _mm512_add_pd(ei, o4r)); }
    { __m512d er = _mm512_load_pd(Escr+64), ei = _mm512_load_pd(Escr+72);
      _mm512_store_pd(X+5*s,   _mm512_add_pd(er, o5i));
      _mm512_store_pd(X+5*s+8, _mm512_sub_pd(ei, o5r));
      _mm512_store_pd(X+8*s,   _mm512_sub_pd(er, o5i));
      _mm512_store_pd(X+8*s+8, _mm512_add_pd(ei, o5r)); }
    { __m512d er = _mm512_load_pd(Escr+80), ei = _mm512_load_pd(Escr+88);
      _mm512_store_pd(X+6*s,   _mm512_add_pd(er, o6i));
      _mm512_store_pd(X+6*s+8, _mm512_sub_pd(ei, o6r));
      _mm512_store_pd(X+7*s,   _mm512_sub_pd(er, o6i));
      _mm512_store_pd(X+7*s+8, _mm512_add_pd(ei, o6r)); }
    }
}
static __attribute__((always_inline)) inline void dft13zm(double* restrict X, long es, int dopf){
    const long s = es*16;
    double AB[48*8] ALIGN64;
    double Escr[12*8] ALIGN64;
    __m512d x0r = _mm512_load_pd(X);
    if(dopf){ _mm_prefetch((const char*)(X+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+216), _MM_HINT_T0); }
    __m512d x0i = _mm512_load_pd(X+8);
    map2(x0r, x0i, &x0r, &x0i);
    { __m512d c1 = _mm512_set1_pd(0x1.c55a7e00740e9p-1), c2 = _mm512_set1_pd(0x1.22d961ea71119p-1), c3 = _mm512_set1_pd(0x1.edb7debaa3ed5p-4), c4 = _mm512_set1_pd(-0x1.6b1d8b2365d9ep-2), c5 = _mm512_set1_pd(-0x1.7f3ccd0032e0dp-1), c6 = _mm512_set1_pd(-0x1.f11f493053d00p-1);
    __m512d e1r = x0r, e1i = x0i;
    __m512d e2r = x0r, e2i = x0i;
    __m512d e3r = x0r, e3i = x0i;
    __m512d e4r = x0r, e4i = x0i;
    __m512d e5r = x0r, e5i = x0i;
    __m512d e6r = x0r, e6i = x0i;
    __m512d sr = x0r, si = x0i;
    { __m512d pr = _mm512_load_pd(X+1*s), qr = _mm512_load_pd(X+12*s);
      if(dopf){ _mm_prefetch((const char*)(X+1*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+1*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+12*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+12*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+1*s+8), qi = _mm512_load_pd(X+12*s+8);
      map2(pr, pi, &pr, &pi); map2(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+0, ur);    _mm512_store_pd(AB+8, ui);
      _mm512_store_pd(AB+16, vr); _mm512_store_pd(AB+24, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c1, ur, e1r); e1i = _mm512_fmadd_pd(c1, ui, e1i);
      e2r = _mm512_fmadd_pd(c2, ur, e2r); e2i = _mm512_fmadd_pd(c2, ui, e2i);
      e3r = _mm512_fmadd_pd(c3, ur, e3r); e3i = _mm512_fmadd_pd(c3, ui, e3i);
      e4r = _mm512_fmadd_pd(c4, ur, e4r); e4i = _mm512_fmadd_pd(c4, ui, e4i);
      e5r = _mm512_fmadd_pd(c5, ur, e5r); e5i = _mm512_fmadd_pd(c5, ui, e5i);
      e6r = _mm512_fmadd_pd(c6, ur, e6r); e6i = _mm512_fmadd_pd(c6, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+2*s), qr = _mm512_load_pd(X+11*s);
      if(dopf){ _mm_prefetch((const char*)(X+2*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+2*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+11*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+11*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+2*s+8), qi = _mm512_load_pd(X+11*s+8);
      map2(pr, pi, &pr, &pi); map2(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+32, ur);    _mm512_store_pd(AB+40, ui);
      _mm512_store_pd(AB+48, vr); _mm512_store_pd(AB+56, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c2, ur, e1r); e1i = _mm512_fmadd_pd(c2, ui, e1i);
      e2r = _mm512_fmadd_pd(c4, ur, e2r); e2i = _mm512_fmadd_pd(c4, ui, e2i);
      e3r = _mm512_fmadd_pd(c6, ur, e3r); e3i = _mm512_fmadd_pd(c6, ui, e3i);
      e4r = _mm512_fmadd_pd(c5, ur, e4r); e4i = _mm512_fmadd_pd(c5, ui, e4i);
      e5r = _mm512_fmadd_pd(c3, ur, e5r); e5i = _mm512_fmadd_pd(c3, ui, e5i);
      e6r = _mm512_fmadd_pd(c1, ur, e6r); e6i = _mm512_fmadd_pd(c1, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+3*s), qr = _mm512_load_pd(X+10*s);
      if(dopf){ _mm_prefetch((const char*)(X+3*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+3*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+10*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+10*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+3*s+8), qi = _mm512_load_pd(X+10*s+8);
      map2(pr, pi, &pr, &pi); map2(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+64, ur);    _mm512_store_pd(AB+72, ui);
      _mm512_store_pd(AB+80, vr); _mm512_store_pd(AB+88, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c3, ur, e1r); e1i = _mm512_fmadd_pd(c3, ui, e1i);
      e2r = _mm512_fmadd_pd(c6, ur, e2r); e2i = _mm512_fmadd_pd(c6, ui, e2i);
      e3r = _mm512_fmadd_pd(c4, ur, e3r); e3i = _mm512_fmadd_pd(c4, ui, e3i);
      e4r = _mm512_fmadd_pd(c1, ur, e4r); e4i = _mm512_fmadd_pd(c1, ui, e4i);
      e5r = _mm512_fmadd_pd(c2, ur, e5r); e5i = _mm512_fmadd_pd(c2, ui, e5i);
      e6r = _mm512_fmadd_pd(c5, ur, e6r); e6i = _mm512_fmadd_pd(c5, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+4*s), qr = _mm512_load_pd(X+9*s);
      if(dopf){ _mm_prefetch((const char*)(X+4*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+4*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+9*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+9*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+4*s+8), qi = _mm512_load_pd(X+9*s+8);
      map2(pr, pi, &pr, &pi); map2(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+96, ur);    _mm512_store_pd(AB+104, ui);
      _mm512_store_pd(AB+112, vr); _mm512_store_pd(AB+120, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c4, ur, e1r); e1i = _mm512_fmadd_pd(c4, ui, e1i);
      e2r = _mm512_fmadd_pd(c5, ur, e2r); e2i = _mm512_fmadd_pd(c5, ui, e2i);
      e3r = _mm512_fmadd_pd(c1, ur, e3r); e3i = _mm512_fmadd_pd(c1, ui, e3i);
      e4r = _mm512_fmadd_pd(c3, ur, e4r); e4i = _mm512_fmadd_pd(c3, ui, e4i);
      e5r = _mm512_fmadd_pd(c6, ur, e5r); e5i = _mm512_fmadd_pd(c6, ui, e5i);
      e6r = _mm512_fmadd_pd(c2, ur, e6r); e6i = _mm512_fmadd_pd(c2, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+5*s), qr = _mm512_load_pd(X+8*s);
      if(dopf){ _mm_prefetch((const char*)(X+5*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+5*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+8*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+8*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+5*s+8), qi = _mm512_load_pd(X+8*s+8);
      map2(pr, pi, &pr, &pi); map2(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+128, ur);    _mm512_store_pd(AB+136, ui);
      _mm512_store_pd(AB+144, vr); _mm512_store_pd(AB+152, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c5, ur, e1r); e1i = _mm512_fmadd_pd(c5, ui, e1i);
      e2r = _mm512_fmadd_pd(c3, ur, e2r); e2i = _mm512_fmadd_pd(c3, ui, e2i);
      e3r = _mm512_fmadd_pd(c2, ur, e3r); e3i = _mm512_fmadd_pd(c2, ui, e3i);
      e4r = _mm512_fmadd_pd(c6, ur, e4r); e4i = _mm512_fmadd_pd(c6, ui, e4i);
      e5r = _mm512_fmadd_pd(c1, ur, e5r); e5i = _mm512_fmadd_pd(c1, ui, e5i);
      e6r = _mm512_fmadd_pd(c4, ur, e6r); e6i = _mm512_fmadd_pd(c4, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+6*s), qr = _mm512_load_pd(X+7*s);
      if(dopf){ _mm_prefetch((const char*)(X+6*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+6*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+7*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+7*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+6*s+8), qi = _mm512_load_pd(X+7*s+8);
      map2(pr, pi, &pr, &pi); map2(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+160, ur);    _mm512_store_pd(AB+168, ui);
      _mm512_store_pd(AB+176, vr); _mm512_store_pd(AB+184, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c6, ur, e1r); e1i = _mm512_fmadd_pd(c6, ui, e1i);
      e2r = _mm512_fmadd_pd(c1, ur, e2r); e2i = _mm512_fmadd_pd(c1, ui, e2i);
      e3r = _mm512_fmadd_pd(c5, ur, e3r); e3i = _mm512_fmadd_pd(c5, ui, e3i);
      e4r = _mm512_fmadd_pd(c2, ur, e4r); e4i = _mm512_fmadd_pd(c2, ui, e4i);
      e5r = _mm512_fmadd_pd(c4, ur, e5r); e5i = _mm512_fmadd_pd(c4, ui, e5i);
      e6r = _mm512_fmadd_pd(c3, ur, e6r); e6i = _mm512_fmadd_pd(c3, ui, e6i);
    }
    _mm512_store_pd(X, sr); _mm512_store_pd(X+8, si);
    _mm512_store_pd(Escr+0, e1r); _mm512_store_pd(Escr+8, e1i);
    _mm512_store_pd(Escr+16, e2r); _mm512_store_pd(Escr+24, e2i);
    _mm512_store_pd(Escr+32, e3r); _mm512_store_pd(Escr+40, e3i);
    _mm512_store_pd(Escr+48, e4r); _mm512_store_pd(Escr+56, e4i);
    _mm512_store_pd(Escr+64, e5r); _mm512_store_pd(Escr+72, e5i);
    _mm512_store_pd(Escr+80, e6r); _mm512_store_pd(Escr+88, e6i);
    }
    { __m512d s1 = _mm512_set1_pd(0x1.dbe064267c47bp-2), s2 = _mm512_set1_pd(0x1.a55e242a4c3d2p-1), s3 = _mm512_set1_pd(0x1.fc44566966769p-1), s4 = _mm512_set1_pd(0x1.deba72ef20147p-1), s5 = _mm512_set1_pd(0x1.5384d024c2f84p-1), s6 = _mm512_set1_pd(0x1.ea1e54bc48dbcp-3);
    __m512d o1r = _mm512_setzero_pd(), o1i = _mm512_setzero_pd();
    __m512d o2r = _mm512_setzero_pd(), o2i = _mm512_setzero_pd();
    __m512d o3r = _mm512_setzero_pd(), o3i = _mm512_setzero_pd();
    __m512d o4r = _mm512_setzero_pd(), o4i = _mm512_setzero_pd();
    __m512d o5r = _mm512_setzero_pd(), o5i = _mm512_setzero_pd();
    __m512d o6r = _mm512_setzero_pd(), o6i = _mm512_setzero_pd();
    { __m512d vr = _mm512_load_pd(AB+16);
      __m512d vi = _mm512_load_pd(AB+24);
      o1r = _mm512_fmadd_pd(s1, vr, o1r); o1i = _mm512_fmadd_pd(s1, vi, o1i);
      o2r = _mm512_fmadd_pd(s2, vr, o2r); o2i = _mm512_fmadd_pd(s2, vi, o2i);
      o3r = _mm512_fmadd_pd(s3, vr, o3r); o3i = _mm512_fmadd_pd(s3, vi, o3i);
      o4r = _mm512_fmadd_pd(s4, vr, o4r); o4i = _mm512_fmadd_pd(s4, vi, o4i);
      o5r = _mm512_fmadd_pd(s5, vr, o5r); o5i = _mm512_fmadd_pd(s5, vi, o5i);
      o6r = _mm512_fmadd_pd(s6, vr, o6r); o6i = _mm512_fmadd_pd(s6, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+48);
      __m512d vi = _mm512_load_pd(AB+56);
      o1r = _mm512_fmadd_pd(s2, vr, o1r); o1i = _mm512_fmadd_pd(s2, vi, o1i);
      o2r = _mm512_fmadd_pd(s4, vr, o2r); o2i = _mm512_fmadd_pd(s4, vi, o2i);
      o3r = _mm512_fmadd_pd(s6, vr, o3r); o3i = _mm512_fmadd_pd(s6, vi, o3i);
      o4r = _mm512_fnmadd_pd(s5, vr, o4r); o4i = _mm512_fnmadd_pd(s5, vi, o4i);
      o5r = _mm512_fnmadd_pd(s3, vr, o5r); o5i = _mm512_fnmadd_pd(s3, vi, o5i);
      o6r = _mm512_fnmadd_pd(s1, vr, o6r); o6i = _mm512_fnmadd_pd(s1, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+80);
      __m512d vi = _mm512_load_pd(AB+88);
      o1r = _mm512_fmadd_pd(s3, vr, o1r); o1i = _mm512_fmadd_pd(s3, vi, o1i);
      o2r = _mm512_fmadd_pd(s6, vr, o2r); o2i = _mm512_fmadd_pd(s6, vi, o2i);
      o3r = _mm512_fnmadd_pd(s4, vr, o3r); o3i = _mm512_fnmadd_pd(s4, vi, o3i);
      o4r = _mm512_fnmadd_pd(s1, vr, o4r); o4i = _mm512_fnmadd_pd(s1, vi, o4i);
      o5r = _mm512_fmadd_pd(s2, vr, o5r); o5i = _mm512_fmadd_pd(s2, vi, o5i);
      o6r = _mm512_fmadd_pd(s5, vr, o6r); o6i = _mm512_fmadd_pd(s5, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+112);
      __m512d vi = _mm512_load_pd(AB+120);
      o1r = _mm512_fmadd_pd(s4, vr, o1r); o1i = _mm512_fmadd_pd(s4, vi, o1i);
      o2r = _mm512_fnmadd_pd(s5, vr, o2r); o2i = _mm512_fnmadd_pd(s5, vi, o2i);
      o3r = _mm512_fnmadd_pd(s1, vr, o3r); o3i = _mm512_fnmadd_pd(s1, vi, o3i);
      o4r = _mm512_fmadd_pd(s3, vr, o4r); o4i = _mm512_fmadd_pd(s3, vi, o4i);
      o5r = _mm512_fnmadd_pd(s6, vr, o5r); o5i = _mm512_fnmadd_pd(s6, vi, o5i);
      o6r = _mm512_fnmadd_pd(s2, vr, o6r); o6i = _mm512_fnmadd_pd(s2, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+144);
      __m512d vi = _mm512_load_pd(AB+152);
      o1r = _mm512_fmadd_pd(s5, vr, o1r); o1i = _mm512_fmadd_pd(s5, vi, o1i);
      o2r = _mm512_fnmadd_pd(s3, vr, o2r); o2i = _mm512_fnmadd_pd(s3, vi, o2i);
      o3r = _mm512_fmadd_pd(s2, vr, o3r); o3i = _mm512_fmadd_pd(s2, vi, o3i);
      o4r = _mm512_fnmadd_pd(s6, vr, o4r); o4i = _mm512_fnmadd_pd(s6, vi, o4i);
      o5r = _mm512_fnmadd_pd(s1, vr, o5r); o5i = _mm512_fnmadd_pd(s1, vi, o5i);
      o6r = _mm512_fmadd_pd(s4, vr, o6r); o6i = _mm512_fmadd_pd(s4, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+176);
      __m512d vi = _mm512_load_pd(AB+184);
      o1r = _mm512_fmadd_pd(s6, vr, o1r); o1i = _mm512_fmadd_pd(s6, vi, o1i);
      o2r = _mm512_fnmadd_pd(s1, vr, o2r); o2i = _mm512_fnmadd_pd(s1, vi, o2i);
      o3r = _mm512_fmadd_pd(s5, vr, o3r); o3i = _mm512_fmadd_pd(s5, vi, o3i);
      o4r = _mm512_fnmadd_pd(s2, vr, o4r); o4i = _mm512_fnmadd_pd(s2, vi, o4i);
      o5r = _mm512_fmadd_pd(s4, vr, o5r); o5i = _mm512_fmadd_pd(s4, vi, o5i);
      o6r = _mm512_fnmadd_pd(s3, vr, o6r); o6i = _mm512_fnmadd_pd(s3, vi, o6i);
    }
    { __m512d er = _mm512_load_pd(Escr+0), ei = _mm512_load_pd(Escr+8);
      _mm512_store_pd(X+1*s,   _mm512_add_pd(er, o1i));
      _mm512_store_pd(X+1*s+8, _mm512_sub_pd(ei, o1r));
      _mm512_store_pd(X+12*s,   _mm512_sub_pd(er, o1i));
      _mm512_store_pd(X+12*s+8, _mm512_add_pd(ei, o1r)); }
    { __m512d er = _mm512_load_pd(Escr+16), ei = _mm512_load_pd(Escr+24);
      _mm512_store_pd(X+2*s,   _mm512_add_pd(er, o2i));
      _mm512_store_pd(X+2*s+8, _mm512_sub_pd(ei, o2r));
      _mm512_store_pd(X+11*s,   _mm512_sub_pd(er, o2i));
      _mm512_store_pd(X+11*s+8, _mm512_add_pd(ei, o2r)); }
    { __m512d er = _mm512_load_pd(Escr+32), ei = _mm512_load_pd(Escr+40);
      _mm512_store_pd(X+3*s,   _mm512_add_pd(er, o3i));
      _mm512_store_pd(X+3*s+8, _mm512_sub_pd(ei, o3r));
      _mm512_store_pd(X+10*s,   _mm512_sub_pd(er, o3i));
      _mm512_store_pd(X+10*s+8, _mm512_add_pd(ei, o3r)); }
    { __m512d er = _mm512_load_pd(Escr+48), ei = _mm512_load_pd(Escr+56);
      _mm512_store_pd(X+4*s,   _mm512_add_pd(er, o4i));
      _mm512_store_pd(X+4*s+8, _mm512_sub_pd(ei, o4r));
      _mm512_store_pd(X+9*s,   _mm512_sub_pd(er, o4i));
      _mm512_store_pd(X+9*s+8, _mm512_add_pd(ei, o4r)); }
    { __m512d er = _mm512_load_pd(Escr+64), ei = _mm512_load_pd(Escr+72);
      _mm512_store_pd(X+5*s,   _mm512_add_pd(er, o5i));
      _mm512_store_pd(X+5*s+8, _mm512_sub_pd(ei, o5r));
      _mm512_store_pd(X+8*s,   _mm512_sub_pd(er, o5i));
      _mm512_store_pd(X+8*s+8, _mm512_add_pd(ei, o5r)); }
    { __m512d er = _mm512_load_pd(Escr+80), ei = _mm512_load_pd(Escr+88);
      _mm512_store_pd(X+6*s,   _mm512_add_pd(er, o6i));
      _mm512_store_pd(X+6*s+8, _mm512_sub_pd(ei, o6r));
      _mm512_store_pd(X+7*s,   _mm512_sub_pd(er, o6i));
      _mm512_store_pd(X+7*s+8, _mm512_add_pd(ei, o6r)); }
    }
}
static __attribute__((always_inline)) inline void dft13m(double* restrict X, long es, int dopf, const double* restrict C, long ces){
    const long s = es*16;
    const long cs = ces*16;
    double AB[48*8] ALIGN64;
    double Escr[12*8] ALIGN64;
    __m512d x0r = _mm512_load_pd(X);
    if(dopf){ _mm_prefetch((const char*)(X+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+216), _MM_HINT_T0); }
    __m512d x0i = _mm512_load_pd(X+8);
    { __m512d c1 = _mm512_set1_pd(0x1.c55a7e00740e9p-1), c2 = _mm512_set1_pd(0x1.22d961ea71119p-1), c3 = _mm512_set1_pd(0x1.edb7debaa3ed5p-4), c4 = _mm512_set1_pd(-0x1.6b1d8b2365d9ep-2), c5 = _mm512_set1_pd(-0x1.7f3ccd0032e0dp-1), c6 = _mm512_set1_pd(-0x1.f11f493053d00p-1);
    __m512d e1r = x0r, e1i = x0i;
    __m512d e2r = x0r, e2i = x0i;
    __m512d e3r = x0r, e3i = x0i;
    __m512d e4r = x0r, e4i = x0i;
    __m512d e5r = x0r, e5i = x0i;
    __m512d e6r = x0r, e6i = x0i;
    __m512d sr = x0r, si = x0i;
    { __m512d pr = _mm512_load_pd(X+1*s), qr = _mm512_load_pd(X+12*s);
      if(dopf){ _mm_prefetch((const char*)(X+1*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+1*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+12*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+12*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+1*s+8), qi = _mm512_load_pd(X+12*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+0, ur);    _mm512_store_pd(AB+8, ui);
      _mm512_store_pd(AB+16, vr); _mm512_store_pd(AB+24, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c1, ur, e1r); e1i = _mm512_fmadd_pd(c1, ui, e1i);
      e2r = _mm512_fmadd_pd(c2, ur, e2r); e2i = _mm512_fmadd_pd(c2, ui, e2i);
      e3r = _mm512_fmadd_pd(c3, ur, e3r); e3i = _mm512_fmadd_pd(c3, ui, e3i);
      e4r = _mm512_fmadd_pd(c4, ur, e4r); e4i = _mm512_fmadd_pd(c4, ui, e4i);
      e5r = _mm512_fmadd_pd(c5, ur, e5r); e5i = _mm512_fmadd_pd(c5, ui, e5i);
      e6r = _mm512_fmadd_pd(c6, ur, e6r); e6i = _mm512_fmadd_pd(c6, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+2*s), qr = _mm512_load_pd(X+11*s);
      if(dopf){ _mm_prefetch((const char*)(X+2*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+2*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+11*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+11*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+2*s+8), qi = _mm512_load_pd(X+11*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+32, ur);    _mm512_store_pd(AB+40, ui);
      _mm512_store_pd(AB+48, vr); _mm512_store_pd(AB+56, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c2, ur, e1r); e1i = _mm512_fmadd_pd(c2, ui, e1i);
      e2r = _mm512_fmadd_pd(c4, ur, e2r); e2i = _mm512_fmadd_pd(c4, ui, e2i);
      e3r = _mm512_fmadd_pd(c6, ur, e3r); e3i = _mm512_fmadd_pd(c6, ui, e3i);
      e4r = _mm512_fmadd_pd(c5, ur, e4r); e4i = _mm512_fmadd_pd(c5, ui, e4i);
      e5r = _mm512_fmadd_pd(c3, ur, e5r); e5i = _mm512_fmadd_pd(c3, ui, e5i);
      e6r = _mm512_fmadd_pd(c1, ur, e6r); e6i = _mm512_fmadd_pd(c1, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+3*s), qr = _mm512_load_pd(X+10*s);
      if(dopf){ _mm_prefetch((const char*)(X+3*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+3*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+10*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+10*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+3*s+8), qi = _mm512_load_pd(X+10*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+64, ur);    _mm512_store_pd(AB+72, ui);
      _mm512_store_pd(AB+80, vr); _mm512_store_pd(AB+88, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c3, ur, e1r); e1i = _mm512_fmadd_pd(c3, ui, e1i);
      e2r = _mm512_fmadd_pd(c6, ur, e2r); e2i = _mm512_fmadd_pd(c6, ui, e2i);
      e3r = _mm512_fmadd_pd(c4, ur, e3r); e3i = _mm512_fmadd_pd(c4, ui, e3i);
      e4r = _mm512_fmadd_pd(c1, ur, e4r); e4i = _mm512_fmadd_pd(c1, ui, e4i);
      e5r = _mm512_fmadd_pd(c2, ur, e5r); e5i = _mm512_fmadd_pd(c2, ui, e5i);
      e6r = _mm512_fmadd_pd(c5, ur, e6r); e6i = _mm512_fmadd_pd(c5, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+4*s), qr = _mm512_load_pd(X+9*s);
      if(dopf){ _mm_prefetch((const char*)(X+4*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+4*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+9*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+9*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+4*s+8), qi = _mm512_load_pd(X+9*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+96, ur);    _mm512_store_pd(AB+104, ui);
      _mm512_store_pd(AB+112, vr); _mm512_store_pd(AB+120, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c4, ur, e1r); e1i = _mm512_fmadd_pd(c4, ui, e1i);
      e2r = _mm512_fmadd_pd(c5, ur, e2r); e2i = _mm512_fmadd_pd(c5, ui, e2i);
      e3r = _mm512_fmadd_pd(c1, ur, e3r); e3i = _mm512_fmadd_pd(c1, ui, e3i);
      e4r = _mm512_fmadd_pd(c3, ur, e4r); e4i = _mm512_fmadd_pd(c3, ui, e4i);
      e5r = _mm512_fmadd_pd(c6, ur, e5r); e5i = _mm512_fmadd_pd(c6, ui, e5i);
      e6r = _mm512_fmadd_pd(c2, ur, e6r); e6i = _mm512_fmadd_pd(c2, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+5*s), qr = _mm512_load_pd(X+8*s);
      if(dopf){ _mm_prefetch((const char*)(X+5*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+5*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+8*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+8*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+5*s+8), qi = _mm512_load_pd(X+8*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+128, ur);    _mm512_store_pd(AB+136, ui);
      _mm512_store_pd(AB+144, vr); _mm512_store_pd(AB+152, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c5, ur, e1r); e1i = _mm512_fmadd_pd(c5, ui, e1i);
      e2r = _mm512_fmadd_pd(c3, ur, e2r); e2i = _mm512_fmadd_pd(c3, ui, e2i);
      e3r = _mm512_fmadd_pd(c2, ur, e3r); e3i = _mm512_fmadd_pd(c2, ui, e3i);
      e4r = _mm512_fmadd_pd(c6, ur, e4r); e4i = _mm512_fmadd_pd(c6, ui, e4i);
      e5r = _mm512_fmadd_pd(c1, ur, e5r); e5i = _mm512_fmadd_pd(c1, ui, e5i);
      e6r = _mm512_fmadd_pd(c4, ur, e6r); e6i = _mm512_fmadd_pd(c4, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+6*s), qr = _mm512_load_pd(X+7*s);
      if(dopf){ _mm_prefetch((const char*)(X+6*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+6*s+208+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+7*s+208), _MM_HINT_T0); _mm_prefetch((const char*)(X+7*s+208+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+6*s+8), qi = _mm512_load_pd(X+7*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+160, ur);    _mm512_store_pd(AB+168, ui);
      _mm512_store_pd(AB+176, vr); _mm512_store_pd(AB+184, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c6, ur, e1r); e1i = _mm512_fmadd_pd(c6, ui, e1i);
      e2r = _mm512_fmadd_pd(c1, ur, e2r); e2i = _mm512_fmadd_pd(c1, ui, e2i);
      e3r = _mm512_fmadd_pd(c5, ur, e3r); e3i = _mm512_fmadd_pd(c5, ui, e3i);
      e4r = _mm512_fmadd_pd(c2, ur, e4r); e4i = _mm512_fmadd_pd(c2, ui, e4i);
      e5r = _mm512_fmadd_pd(c4, ur, e5r); e5i = _mm512_fmadd_pd(c4, ui, e5i);
      e6r = _mm512_fmadd_pd(c3, ur, e6r); e6i = _mm512_fmadd_pd(c3, ui, e6i);
    }
    { __m512d zr = _mm512_add_pd(sr, _mm512_load_pd(C)), zi = _mm512_add_pd(si, _mm512_load_pd(C+8));
      _mm512_store_pd(X, zr); _mm512_store_pd(X+8, zi); }
    _mm512_store_pd(Escr+0, e1r); _mm512_store_pd(Escr+8, e1i);
    _mm512_store_pd(Escr+16, e2r); _mm512_store_pd(Escr+24, e2i);
    _mm512_store_pd(Escr+32, e3r); _mm512_store_pd(Escr+40, e3i);
    _mm512_store_pd(Escr+48, e4r); _mm512_store_pd(Escr+56, e4i);
    _mm512_store_pd(Escr+64, e5r); _mm512_store_pd(Escr+72, e5i);
    _mm512_store_pd(Escr+80, e6r); _mm512_store_pd(Escr+88, e6i);
    }
    { __m512d s1 = _mm512_set1_pd(0x1.dbe064267c47bp-2), s2 = _mm512_set1_pd(0x1.a55e242a4c3d2p-1), s3 = _mm512_set1_pd(0x1.fc44566966769p-1), s4 = _mm512_set1_pd(0x1.deba72ef20147p-1), s5 = _mm512_set1_pd(0x1.5384d024c2f84p-1), s6 = _mm512_set1_pd(0x1.ea1e54bc48dbcp-3);
    __m512d o1r = _mm512_setzero_pd(), o1i = _mm512_setzero_pd();
    __m512d o2r = _mm512_setzero_pd(), o2i = _mm512_setzero_pd();
    __m512d o3r = _mm512_setzero_pd(), o3i = _mm512_setzero_pd();
    __m512d o4r = _mm512_setzero_pd(), o4i = _mm512_setzero_pd();
    __m512d o5r = _mm512_setzero_pd(), o5i = _mm512_setzero_pd();
    __m512d o6r = _mm512_setzero_pd(), o6i = _mm512_setzero_pd();
    { __m512d vr = _mm512_load_pd(AB+16);
      __m512d vi = _mm512_load_pd(AB+24);
      o1r = _mm512_fmadd_pd(s1, vr, o1r); o1i = _mm512_fmadd_pd(s1, vi, o1i);
      o2r = _mm512_fmadd_pd(s2, vr, o2r); o2i = _mm512_fmadd_pd(s2, vi, o2i);
      o3r = _mm512_fmadd_pd(s3, vr, o3r); o3i = _mm512_fmadd_pd(s3, vi, o3i);
      o4r = _mm512_fmadd_pd(s4, vr, o4r); o4i = _mm512_fmadd_pd(s4, vi, o4i);
      o5r = _mm512_fmadd_pd(s5, vr, o5r); o5i = _mm512_fmadd_pd(s5, vi, o5i);
      o6r = _mm512_fmadd_pd(s6, vr, o6r); o6i = _mm512_fmadd_pd(s6, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+48);
      __m512d vi = _mm512_load_pd(AB+56);
      o1r = _mm512_fmadd_pd(s2, vr, o1r); o1i = _mm512_fmadd_pd(s2, vi, o1i);
      o2r = _mm512_fmadd_pd(s4, vr, o2r); o2i = _mm512_fmadd_pd(s4, vi, o2i);
      o3r = _mm512_fmadd_pd(s6, vr, o3r); o3i = _mm512_fmadd_pd(s6, vi, o3i);
      o4r = _mm512_fnmadd_pd(s5, vr, o4r); o4i = _mm512_fnmadd_pd(s5, vi, o4i);
      o5r = _mm512_fnmadd_pd(s3, vr, o5r); o5i = _mm512_fnmadd_pd(s3, vi, o5i);
      o6r = _mm512_fnmadd_pd(s1, vr, o6r); o6i = _mm512_fnmadd_pd(s1, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+80);
      __m512d vi = _mm512_load_pd(AB+88);
      o1r = _mm512_fmadd_pd(s3, vr, o1r); o1i = _mm512_fmadd_pd(s3, vi, o1i);
      o2r = _mm512_fmadd_pd(s6, vr, o2r); o2i = _mm512_fmadd_pd(s6, vi, o2i);
      o3r = _mm512_fnmadd_pd(s4, vr, o3r); o3i = _mm512_fnmadd_pd(s4, vi, o3i);
      o4r = _mm512_fnmadd_pd(s1, vr, o4r); o4i = _mm512_fnmadd_pd(s1, vi, o4i);
      o5r = _mm512_fmadd_pd(s2, vr, o5r); o5i = _mm512_fmadd_pd(s2, vi, o5i);
      o6r = _mm512_fmadd_pd(s5, vr, o6r); o6i = _mm512_fmadd_pd(s5, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+112);
      __m512d vi = _mm512_load_pd(AB+120);
      o1r = _mm512_fmadd_pd(s4, vr, o1r); o1i = _mm512_fmadd_pd(s4, vi, o1i);
      o2r = _mm512_fnmadd_pd(s5, vr, o2r); o2i = _mm512_fnmadd_pd(s5, vi, o2i);
      o3r = _mm512_fnmadd_pd(s1, vr, o3r); o3i = _mm512_fnmadd_pd(s1, vi, o3i);
      o4r = _mm512_fmadd_pd(s3, vr, o4r); o4i = _mm512_fmadd_pd(s3, vi, o4i);
      o5r = _mm512_fnmadd_pd(s6, vr, o5r); o5i = _mm512_fnmadd_pd(s6, vi, o5i);
      o6r = _mm512_fnmadd_pd(s2, vr, o6r); o6i = _mm512_fnmadd_pd(s2, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+144);
      __m512d vi = _mm512_load_pd(AB+152);
      o1r = _mm512_fmadd_pd(s5, vr, o1r); o1i = _mm512_fmadd_pd(s5, vi, o1i);
      o2r = _mm512_fnmadd_pd(s3, vr, o2r); o2i = _mm512_fnmadd_pd(s3, vi, o2i);
      o3r = _mm512_fmadd_pd(s2, vr, o3r); o3i = _mm512_fmadd_pd(s2, vi, o3i);
      o4r = _mm512_fnmadd_pd(s6, vr, o4r); o4i = _mm512_fnmadd_pd(s6, vi, o4i);
      o5r = _mm512_fnmadd_pd(s1, vr, o5r); o5i = _mm512_fnmadd_pd(s1, vi, o5i);
      o6r = _mm512_fmadd_pd(s4, vr, o6r); o6i = _mm512_fmadd_pd(s4, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+176);
      __m512d vi = _mm512_load_pd(AB+184);
      o1r = _mm512_fmadd_pd(s6, vr, o1r); o1i = _mm512_fmadd_pd(s6, vi, o1i);
      o2r = _mm512_fnmadd_pd(s1, vr, o2r); o2i = _mm512_fnmadd_pd(s1, vi, o2i);
      o3r = _mm512_fmadd_pd(s5, vr, o3r); o3i = _mm512_fmadd_pd(s5, vi, o3i);
      o4r = _mm512_fnmadd_pd(s2, vr, o4r); o4i = _mm512_fnmadd_pd(s2, vi, o4i);
      o5r = _mm512_fmadd_pd(s4, vr, o5r); o5i = _mm512_fmadd_pd(s4, vi, o5i);
      o6r = _mm512_fnmadd_pd(s3, vr, o6r); o6i = _mm512_fnmadd_pd(s3, vi, o6i);
    }
    { __m512d er = _mm512_load_pd(Escr+0), ei = _mm512_load_pd(Escr+8);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o1i), _mm512_load_pd(C+1*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o1r), _mm512_load_pd(C+1*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o1i), _mm512_load_pd(C+12*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o1r), _mm512_load_pd(C+12*cs+8));
      _mm512_store_pd(X+1*s, zr1);   _mm512_store_pd(X+1*s+8, zi1);
      _mm512_store_pd(X+12*s, zr2); _mm512_store_pd(X+12*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+16), ei = _mm512_load_pd(Escr+24);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o2i), _mm512_load_pd(C+2*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o2r), _mm512_load_pd(C+2*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o2i), _mm512_load_pd(C+11*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o2r), _mm512_load_pd(C+11*cs+8));
      _mm512_store_pd(X+2*s, zr1);   _mm512_store_pd(X+2*s+8, zi1);
      _mm512_store_pd(X+11*s, zr2); _mm512_store_pd(X+11*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+32), ei = _mm512_load_pd(Escr+40);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o3i), _mm512_load_pd(C+3*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o3r), _mm512_load_pd(C+3*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o3i), _mm512_load_pd(C+10*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o3r), _mm512_load_pd(C+10*cs+8));
      _mm512_store_pd(X+3*s, zr1);   _mm512_store_pd(X+3*s+8, zi1);
      _mm512_store_pd(X+10*s, zr2); _mm512_store_pd(X+10*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+48), ei = _mm512_load_pd(Escr+56);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o4i), _mm512_load_pd(C+4*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o4r), _mm512_load_pd(C+4*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o4i), _mm512_load_pd(C+9*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o4r), _mm512_load_pd(C+9*cs+8));
      _mm512_store_pd(X+4*s, zr1);   _mm512_store_pd(X+4*s+8, zi1);
      _mm512_store_pd(X+9*s, zr2); _mm512_store_pd(X+9*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+64), ei = _mm512_load_pd(Escr+72);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o5i), _mm512_load_pd(C+5*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o5r), _mm512_load_pd(C+5*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o5i), _mm512_load_pd(C+8*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o5r), _mm512_load_pd(C+8*cs+8));
      _mm512_store_pd(X+5*s, zr1);   _mm512_store_pd(X+5*s+8, zi1);
      _mm512_store_pd(X+8*s, zr2); _mm512_store_pd(X+8*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+80), ei = _mm512_load_pd(Escr+88);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o6i), _mm512_load_pd(C+6*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o6r), _mm512_load_pd(C+6*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o6i), _mm512_load_pd(C+7*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o6r), _mm512_load_pd(C+7*cs+8));
      _mm512_store_pd(X+6*s, zr1);   _mm512_store_pd(X+6*s+8, zi1);
      _mm512_store_pd(X+7*s, zr2); _mm512_store_pd(X+7*s+8, zi2); }
    }
}
#define MAPX_13 0
#define PXF_13 0
#define PFPRIME_13 0

static void __attribute__((noinline)) __attribute__((optimize("schedule-insns,sched-pressure"))) dft13_one(double* X, long es){ dft13(X, es, 0); }
static void __attribute__((noinline)) __attribute__((optimize("schedule-insns,sched-pressure"))) dft13_onez(double* X){ dft13(X, 1, PFPRIME_13); }
static void __attribute__((noinline)) __attribute__((optimize("schedule-insns,sched-pressure"))) dft13_onezm(double* X){ dft13zm(X, 1, PFPRIME_13); }
static void __attribute__((noinline)) __attribute__((optimize("schedule-insns,sched-pressure"))) dft13_onem(double* X, long es, const double* Ct){ dft13m(X, es, 0, Ct, 1); }
static void dft13_sweep_zy(double* restrict X){
    for(long x=0; x<13; x++){
        double* P = X + x*169*16;
        for(long y=0; y<13; y++) dft13_onez(P + y*13*16);
        for(long z=0; z<13; z++) dft13_one(P + z*16, 13);
    }
}
#if MAPZB_FLAG
static void __attribute__((noinline)) mapblk_13(double* restrict P){
    for(long e=0; e<13; e+=1){
        __m512d zr = _mm512_load_pd(P + e*16);
        __m512d zi = _mm512_load_pd(P + e*16 + 8);
        if(e & 1){ map2(zr, zi, &zr, &zi); } else { maphw(zr, zi, &zr, &zi); }
        _mm512_store_pd(P + e*16, zr);
        _mm512_store_pd(P + e*16 + 8, zi);
    }
}
static void dft13_sweep_zym(double* restrict X){
    for(long x=0; x<13; x++){
        double* P = X + x*169*16;
        for(long y=0; y<13; y++){ mapblk_13(P + y*13*16); dft13_one(P + y*13*16, 1); }
        for(long z=0; z<13; z++) dft13_one(P + z*16, 13);
    }
}
#else
static void dft13_sweep_zym(double* restrict X){
    for(long x=0; x<13; x++){
        double* P = X + x*169*16;
        for(long y=0; y<13; y++) dft13_onezm(P + y*13*16);
        for(long z=0; z<13; z++) dft13_one(P + z*16, 13);
    }
}
#endif
static void dft13_sweep_x_map(double* restrict X, const double* restrict Ct){
    for(long p=0; p<169; p++) dft13_onem(X + p*16, 169, Ct + p*13*16);
}
static void dft13_sweep_x_plain(double* restrict X){
    for(long p=0; p<169; p++) dft13_one(X + p*16, 169);
}
static void dft13_sweep_zy_ms(double* restrict X, const double* restrict C){
    for(long x=0; x<13; x++){
        double* P = X + x*169*16;
        for(long y=0; y<13; y++) dft13_one(P + y*13*16, 1);
        for(long z=0; z<13; z++) dft13_one(P + z*16, 13);
        if(x) mapslab(X + (x-1)*169*16, C + (x-1)*169*16, 169);
    }
    mapslab(X + (13-1)*169*16, C + (13-1)*169*16, 169);
}


/* ingest: 8 volumes AoS complex -> group SoA [e][2][8] */
static void ingest_13(const double* const* src, double* G){
    for(long e=0; e<2196; e+=4){
        __m512d r0=_mm512_loadu_pd(src[0]+2*e), r1=_mm512_loadu_pd(src[1]+2*e);
        __m512d r2=_mm512_loadu_pd(src[2]+2*e), r3=_mm512_loadu_pd(src[3]+2*e);
        __m512d r4=_mm512_loadu_pd(src[4]+2*e), r5=_mm512_loadu_pd(src[5]+2*e);
        __m512d r6=_mm512_loadu_pd(src[6]+2*e), r7=_mm512_loadu_pd(src[7]+2*e);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        _mm512_store_pd(G+e*16,    o0); _mm512_store_pd(G+e*16+8,  o1);
        _mm512_store_pd(G+e*16+16, o2); _mm512_store_pd(G+e*16+24, o3);
        _mm512_store_pd(G+e*16+32, o4); _mm512_store_pd(G+e*16+40, o5);
        _mm512_store_pd(G+e*16+48, o6); _mm512_store_pd(G+e*16+56, o7);
    }
    { /* tail of 1 elements */
        const long e = 2196;
        const __mmask8 mk = (__mmask8)((1u<<(2*1))-1u);
        __m512d r0=_mm512_maskz_loadu_pd(mk, src[0]+2*e), r1=_mm512_maskz_loadu_pd(mk, src[1]+2*e);
        __m512d r2=_mm512_maskz_loadu_pd(mk, src[2]+2*e), r3=_mm512_maskz_loadu_pd(mk, src[3]+2*e);
        __m512d r4=_mm512_maskz_loadu_pd(mk, src[4]+2*e), r5=_mm512_maskz_loadu_pd(mk, src[5]+2*e);
        __m512d r6=_mm512_maskz_loadu_pd(mk, src[6]+2*e), r7=_mm512_maskz_loadu_pd(mk, src[7]+2*e);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        __m512d A[8]; A[0]=o0;A[1]=o1;A[2]=o2;A[3]=o3;A[4]=o4;A[5]=o5;A[6]=o6;A[7]=o7;
        for(int q=0;q<2*1;q++) _mm512_store_pd(G+e*16+q*8, A[q]);
    }
}
/* output with map: group SoA (unmapped z) -> nv AoS volumes */
static void output_13(const double* G, double* const* dst, int nv){
    for(long e=0; e<2196; e+=4){
        __m512d i0=_mm512_load_pd(G+e*16),    i1=_mm512_load_pd(G+e*16+8);
        __m512d i2=_mm512_load_pd(G+e*16+16), i3=_mm512_load_pd(G+e*16+24);
        __m512d i4=_mm512_load_pd(G+e*16+32), i5=_mm512_load_pd(G+e*16+40);
        __m512d i6=_mm512_load_pd(G+e*16+48), i7=_mm512_load_pd(G+e*16+56);
        map2(i0,i1,&i0,&i1); map2(i2,i3,&i2,&i3); map2(i4,i5,&i4,&i5); map2(i6,i7,&i6,&i7);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(i0,i1,i2,i3,i4,i5,i6,i7,o0,o1,o2,o3,o4,o5,o6,o7);
        __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
        for(int v=0; v<nv; v++) _mm512_storeu_pd(dst[v]+2*e, *O[v]);
    }
    { /* tail */
        const long e = 2196;
        const __mmask8 mk = (__mmask8)((1u<<(2*1))-1u);
        __m512d A[8];
        for(int q=0;q<2*1;q++) A[q] = _mm512_load_pd(G+e*16+q*8);
        for(int q=0;q<2*1;q+=2) map2(A[q],A[q+1],&A[q],&A[q+1]);
        for(int q=2*1;q<8;q++) A[q] = _mm512_setzero_pd();
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],o0,o1,o2,o3,o4,o5,o6,o7);
        __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
        for(int v=0; v<nv; v++) _mm512_mask_storeu_pd(dst[v]+2*e, mk, *O[v]);
    }
}


static double* Xg_13 = 0;
static double* Cg_13 = 0;
void hot2_13(long which){
    if(!Xg_13){ Xg_13 = alloc_huge_st((2197+64*13)*16*8); Cg_13 = alloc_huge_st(2197*16*8); }
    double* P = Xg_13;
    if(which==99){ for(long i=0;i<169*16;i++) P[i] = 0.5 + 1e-6*(i%97); return; }
    if(which==0 || which==2) for(long y=0; y<13; y++) dft13_one(P + y*13*16, 1);
    if(which==1 || which==2) for(long z=0; z<13; z++) dft13_one(P + z*16, 13);
}
void bsweep_13(long which, long n){
    if(!Xg_13){ Xg_13 = alloc_huge_st(2197*16*8); Cg_13 = alloc_huge_st(2197*16*8); }
    for(long i=0;i<2197*16;i++){ Xg_13[i] = 0.5 + 1e-6*(i%97); Cg_13[i] = 0.01; }
    for(long r=0;r<n;r++){
        if(which==0) dft13_sweep_zy(Xg_13);
        else if(which==1) dft13_sweep_zym(Xg_13);
        else if(which==2) dft13_sweep_x_map(Xg_13, Cg_13);
#if USEASM_FLAG
#endif
        if((r&7)==7) for(long i=0;i<2197*16;i+=997) Xg_13[i] = 0.5;
    }
}
void run_13(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    const long NE = 2197;
    if(!Xg_13){ Xg_13 = alloc_huge_st(NE*16*8); Cg_13 = alloc_huge_st(NE*16*8); }
    double* X = Xg_13; double* Ct = Cg_13;
    for(long g0=0; g0<B; g0+=8){
        int nv = (int)((B - g0) < 8 ? (B - g0) : 8);
        const double* src[8]; const double* csrc[8];
        double* d1[8]; double* dm[8];
        for(int v=0; v<8; v++){
            int vv = v < nv ? v : 0;
            src[v] = x0 + (g0+vv)*2*NE; csrc[v] = c + (g0+vv)*2*NE;
            if(v<nv){ d1[v] = out1 + (g0+v)*2*NE; dm[v] = outm + (g0+v)*2*NE; }
        }
        /* ingest c (consumption order for MAPX/deferred paths; plain for PXF) */
#if PXF_13
        ingest_13(csrc, Ct);
#else
        ingest_13(csrc, X);
        for(long p=0; p<169; p++)
            for(long k=0; k<13; k++){
                _mm512_store_pd(Ct + (p*13+k)*16,     _mm512_load_pd(X + (k*169+p)*16));
                _mm512_store_pd(Ct + (p*13+k)*16 + 8, _mm512_load_pd(X + (k*169+p)*16 + 8));
            }
#endif
        ingest_13(src, X);
        for(long t=0; t<m; t++){
#if PXF_13
            dft13_sweep_x_plain(X);
            dft13_sweep_zy_ms(X, Ct);
#elif USEASM_FLAG
            dft13_sweep_zy_asm(X, t>0);
            dft13_sweep_x_asm(X, Ct);
#else
#if MAPX_13
            dft13_sweep_zy(X);
            dft13_sweep_x_map(X, Ct);
#else
            if(t==0) dft13_sweep_zy(X); else dft13_sweep_zym(X);
            dft13_sweep_x_map(X, Ct);
#endif
#endif
            if(t==0 && m>1) output_13(X, d1, nv);
        }
        output_13(X, dm, nv);
        if(m==1) output_13(X, d1, nv);
    }
}

static __attribute__((always_inline)) inline void dft17(double* restrict X, long es, int dopf){
    const long s = es*16;
    double AB[64*8] ALIGN64;
    double Escr[16*8] ALIGN64;
    __m512d x0r = _mm512_load_pd(X);
    if(dopf){ _mm_prefetch((const char*)(X+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+280), _MM_HINT_T0); }
    __m512d x0i = _mm512_load_pd(X+8);
    { __m512d c1 = _mm512_set1_pd(0x1.dd6d000370991p-1), c2 = _mm512_set1_pd(0x1.7a5f6075d4884p-1), c3 = _mm512_set1_pd(0x1.c86fa2b2883cep-2), c4 = _mm512_set1_pd(0x1.79ee63259b75fp-4), c5 = _mm512_set1_pd(-0x1.183b1c61f0d01p-2), c6 = _mm512_set1_pd(-0x1.348c86ed5f1bap-1), c7 = _mm512_set1_pd(-0x1.b34fa910ea3b8p-1), c8 = _mm512_set1_pd(-0x1.f7484007faef3p-1);
    __m512d e1r = x0r, e1i = x0i;
    __m512d e2r = x0r, e2i = x0i;
    __m512d e3r = x0r, e3i = x0i;
    __m512d e4r = x0r, e4i = x0i;
    __m512d sr = x0r, si = x0i;
    { __m512d pr = _mm512_load_pd(X+1*s), qr = _mm512_load_pd(X+16*s);
      if(dopf){ _mm_prefetch((const char*)(X+1*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+1*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+16*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+16*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+1*s+8), qi = _mm512_load_pd(X+16*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+0, ur);    _mm512_store_pd(AB+8, ui);
      _mm512_store_pd(AB+16, vr); _mm512_store_pd(AB+24, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c1, ur, e1r); e1i = _mm512_fmadd_pd(c1, ui, e1i);
      e2r = _mm512_fmadd_pd(c2, ur, e2r); e2i = _mm512_fmadd_pd(c2, ui, e2i);
      e3r = _mm512_fmadd_pd(c3, ur, e3r); e3i = _mm512_fmadd_pd(c3, ui, e3i);
      e4r = _mm512_fmadd_pd(c4, ur, e4r); e4i = _mm512_fmadd_pd(c4, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+2*s), qr = _mm512_load_pd(X+15*s);
      if(dopf){ _mm_prefetch((const char*)(X+2*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+2*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+15*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+15*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+2*s+8), qi = _mm512_load_pd(X+15*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+32, ur);    _mm512_store_pd(AB+40, ui);
      _mm512_store_pd(AB+48, vr); _mm512_store_pd(AB+56, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c2, ur, e1r); e1i = _mm512_fmadd_pd(c2, ui, e1i);
      e2r = _mm512_fmadd_pd(c4, ur, e2r); e2i = _mm512_fmadd_pd(c4, ui, e2i);
      e3r = _mm512_fmadd_pd(c6, ur, e3r); e3i = _mm512_fmadd_pd(c6, ui, e3i);
      e4r = _mm512_fmadd_pd(c8, ur, e4r); e4i = _mm512_fmadd_pd(c8, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+3*s), qr = _mm512_load_pd(X+14*s);
      if(dopf){ _mm_prefetch((const char*)(X+3*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+3*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+14*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+14*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+3*s+8), qi = _mm512_load_pd(X+14*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+64, ur);    _mm512_store_pd(AB+72, ui);
      _mm512_store_pd(AB+80, vr); _mm512_store_pd(AB+88, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c3, ur, e1r); e1i = _mm512_fmadd_pd(c3, ui, e1i);
      e2r = _mm512_fmadd_pd(c6, ur, e2r); e2i = _mm512_fmadd_pd(c6, ui, e2i);
      e3r = _mm512_fmadd_pd(c8, ur, e3r); e3i = _mm512_fmadd_pd(c8, ui, e3i);
      e4r = _mm512_fmadd_pd(c5, ur, e4r); e4i = _mm512_fmadd_pd(c5, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+4*s), qr = _mm512_load_pd(X+13*s);
      if(dopf){ _mm_prefetch((const char*)(X+4*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+4*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+13*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+13*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+4*s+8), qi = _mm512_load_pd(X+13*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+96, ur);    _mm512_store_pd(AB+104, ui);
      _mm512_store_pd(AB+112, vr); _mm512_store_pd(AB+120, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c4, ur, e1r); e1i = _mm512_fmadd_pd(c4, ui, e1i);
      e2r = _mm512_fmadd_pd(c8, ur, e2r); e2i = _mm512_fmadd_pd(c8, ui, e2i);
      e3r = _mm512_fmadd_pd(c5, ur, e3r); e3i = _mm512_fmadd_pd(c5, ui, e3i);
      e4r = _mm512_fmadd_pd(c1, ur, e4r); e4i = _mm512_fmadd_pd(c1, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+5*s), qr = _mm512_load_pd(X+12*s);
      if(dopf){ _mm_prefetch((const char*)(X+5*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+5*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+12*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+12*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+5*s+8), qi = _mm512_load_pd(X+12*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+128, ur);    _mm512_store_pd(AB+136, ui);
      _mm512_store_pd(AB+144, vr); _mm512_store_pd(AB+152, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c5, ur, e1r); e1i = _mm512_fmadd_pd(c5, ui, e1i);
      e2r = _mm512_fmadd_pd(c7, ur, e2r); e2i = _mm512_fmadd_pd(c7, ui, e2i);
      e3r = _mm512_fmadd_pd(c2, ur, e3r); e3i = _mm512_fmadd_pd(c2, ui, e3i);
      e4r = _mm512_fmadd_pd(c3, ur, e4r); e4i = _mm512_fmadd_pd(c3, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+6*s), qr = _mm512_load_pd(X+11*s);
      if(dopf){ _mm_prefetch((const char*)(X+6*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+6*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+11*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+11*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+6*s+8), qi = _mm512_load_pd(X+11*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+160, ur);    _mm512_store_pd(AB+168, ui);
      _mm512_store_pd(AB+176, vr); _mm512_store_pd(AB+184, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c6, ur, e1r); e1i = _mm512_fmadd_pd(c6, ui, e1i);
      e2r = _mm512_fmadd_pd(c5, ur, e2r); e2i = _mm512_fmadd_pd(c5, ui, e2i);
      e3r = _mm512_fmadd_pd(c1, ur, e3r); e3i = _mm512_fmadd_pd(c1, ui, e3i);
      e4r = _mm512_fmadd_pd(c7, ur, e4r); e4i = _mm512_fmadd_pd(c7, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+7*s), qr = _mm512_load_pd(X+10*s);
      if(dopf){ _mm_prefetch((const char*)(X+7*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+7*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+10*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+10*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+7*s+8), qi = _mm512_load_pd(X+10*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+192, ur);    _mm512_store_pd(AB+200, ui);
      _mm512_store_pd(AB+208, vr); _mm512_store_pd(AB+216, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c7, ur, e1r); e1i = _mm512_fmadd_pd(c7, ui, e1i);
      e2r = _mm512_fmadd_pd(c3, ur, e2r); e2i = _mm512_fmadd_pd(c3, ui, e2i);
      e3r = _mm512_fmadd_pd(c4, ur, e3r); e3i = _mm512_fmadd_pd(c4, ui, e3i);
      e4r = _mm512_fmadd_pd(c6, ur, e4r); e4i = _mm512_fmadd_pd(c6, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+8*s), qr = _mm512_load_pd(X+9*s);
      if(dopf){ _mm_prefetch((const char*)(X+8*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+8*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+9*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+9*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+8*s+8), qi = _mm512_load_pd(X+9*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+224, ur);    _mm512_store_pd(AB+232, ui);
      _mm512_store_pd(AB+240, vr); _mm512_store_pd(AB+248, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c8, ur, e1r); e1i = _mm512_fmadd_pd(c8, ui, e1i);
      e2r = _mm512_fmadd_pd(c1, ur, e2r); e2i = _mm512_fmadd_pd(c1, ui, e2i);
      e3r = _mm512_fmadd_pd(c7, ur, e3r); e3i = _mm512_fmadd_pd(c7, ui, e3i);
      e4r = _mm512_fmadd_pd(c2, ur, e4r); e4i = _mm512_fmadd_pd(c2, ui, e4i);
    }
    _mm512_store_pd(X, sr); _mm512_store_pd(X+8, si);
    _mm512_store_pd(Escr+0, e1r); _mm512_store_pd(Escr+8, e1i);
    _mm512_store_pd(Escr+16, e2r); _mm512_store_pd(Escr+24, e2i);
    _mm512_store_pd(Escr+32, e3r); _mm512_store_pd(Escr+40, e3i);
    _mm512_store_pd(Escr+48, e4r); _mm512_store_pd(Escr+56, e4i);
    }
    { __m512d c1 = _mm512_set1_pd(0x1.dd6d000370991p-1), c2 = _mm512_set1_pd(0x1.7a5f6075d4884p-1), c3 = _mm512_set1_pd(0x1.c86fa2b2883cep-2), c4 = _mm512_set1_pd(0x1.79ee63259b75fp-4), c5 = _mm512_set1_pd(-0x1.183b1c61f0d01p-2), c6 = _mm512_set1_pd(-0x1.348c86ed5f1bap-1), c7 = _mm512_set1_pd(-0x1.b34fa910ea3b8p-1), c8 = _mm512_set1_pd(-0x1.f7484007faef3p-1);
    __m512d e5r = x0r, e5i = x0i;
    __m512d e6r = x0r, e6i = x0i;
    __m512d e7r = x0r, e7i = x0i;
    __m512d e8r = x0r, e8i = x0i;
    { __m512d ur = _mm512_load_pd(AB+0);
      __m512d ui = _mm512_load_pd(AB+8);
      e5r = _mm512_fmadd_pd(c5, ur, e5r); e5i = _mm512_fmadd_pd(c5, ui, e5i);
      e6r = _mm512_fmadd_pd(c6, ur, e6r); e6i = _mm512_fmadd_pd(c6, ui, e6i);
      e7r = _mm512_fmadd_pd(c7, ur, e7r); e7i = _mm512_fmadd_pd(c7, ui, e7i);
      e8r = _mm512_fmadd_pd(c8, ur, e8r); e8i = _mm512_fmadd_pd(c8, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+32);
      __m512d ui = _mm512_load_pd(AB+40);
      e5r = _mm512_fmadd_pd(c7, ur, e5r); e5i = _mm512_fmadd_pd(c7, ui, e5i);
      e6r = _mm512_fmadd_pd(c5, ur, e6r); e6i = _mm512_fmadd_pd(c5, ui, e6i);
      e7r = _mm512_fmadd_pd(c3, ur, e7r); e7i = _mm512_fmadd_pd(c3, ui, e7i);
      e8r = _mm512_fmadd_pd(c1, ur, e8r); e8i = _mm512_fmadd_pd(c1, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+64);
      __m512d ui = _mm512_load_pd(AB+72);
      e5r = _mm512_fmadd_pd(c2, ur, e5r); e5i = _mm512_fmadd_pd(c2, ui, e5i);
      e6r = _mm512_fmadd_pd(c1, ur, e6r); e6i = _mm512_fmadd_pd(c1, ui, e6i);
      e7r = _mm512_fmadd_pd(c4, ur, e7r); e7i = _mm512_fmadd_pd(c4, ui, e7i);
      e8r = _mm512_fmadd_pd(c7, ur, e8r); e8i = _mm512_fmadd_pd(c7, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+96);
      __m512d ui = _mm512_load_pd(AB+104);
      e5r = _mm512_fmadd_pd(c3, ur, e5r); e5i = _mm512_fmadd_pd(c3, ui, e5i);
      e6r = _mm512_fmadd_pd(c7, ur, e6r); e6i = _mm512_fmadd_pd(c7, ui, e6i);
      e7r = _mm512_fmadd_pd(c6, ur, e7r); e7i = _mm512_fmadd_pd(c6, ui, e7i);
      e8r = _mm512_fmadd_pd(c2, ur, e8r); e8i = _mm512_fmadd_pd(c2, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+128);
      __m512d ui = _mm512_load_pd(AB+136);
      e5r = _mm512_fmadd_pd(c8, ur, e5r); e5i = _mm512_fmadd_pd(c8, ui, e5i);
      e6r = _mm512_fmadd_pd(c4, ur, e6r); e6i = _mm512_fmadd_pd(c4, ui, e6i);
      e7r = _mm512_fmadd_pd(c1, ur, e7r); e7i = _mm512_fmadd_pd(c1, ui, e7i);
      e8r = _mm512_fmadd_pd(c6, ur, e8r); e8i = _mm512_fmadd_pd(c6, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+160);
      __m512d ui = _mm512_load_pd(AB+168);
      e5r = _mm512_fmadd_pd(c4, ur, e5r); e5i = _mm512_fmadd_pd(c4, ui, e5i);
      e6r = _mm512_fmadd_pd(c2, ur, e6r); e6i = _mm512_fmadd_pd(c2, ui, e6i);
      e7r = _mm512_fmadd_pd(c8, ur, e7r); e7i = _mm512_fmadd_pd(c8, ui, e7i);
      e8r = _mm512_fmadd_pd(c3, ur, e8r); e8i = _mm512_fmadd_pd(c3, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+192);
      __m512d ui = _mm512_load_pd(AB+200);
      e5r = _mm512_fmadd_pd(c1, ur, e5r); e5i = _mm512_fmadd_pd(c1, ui, e5i);
      e6r = _mm512_fmadd_pd(c8, ur, e6r); e6i = _mm512_fmadd_pd(c8, ui, e6i);
      e7r = _mm512_fmadd_pd(c2, ur, e7r); e7i = _mm512_fmadd_pd(c2, ui, e7i);
      e8r = _mm512_fmadd_pd(c5, ur, e8r); e8i = _mm512_fmadd_pd(c5, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+224);
      __m512d ui = _mm512_load_pd(AB+232);
      e5r = _mm512_fmadd_pd(c6, ur, e5r); e5i = _mm512_fmadd_pd(c6, ui, e5i);
      e6r = _mm512_fmadd_pd(c3, ur, e6r); e6i = _mm512_fmadd_pd(c3, ui, e6i);
      e7r = _mm512_fmadd_pd(c5, ur, e7r); e7i = _mm512_fmadd_pd(c5, ui, e7i);
      e8r = _mm512_fmadd_pd(c4, ur, e8r); e8i = _mm512_fmadd_pd(c4, ui, e8i);
    }
    _mm512_store_pd(Escr+64, e5r); _mm512_store_pd(Escr+72, e5i);
    _mm512_store_pd(Escr+80, e6r); _mm512_store_pd(Escr+88, e6i);
    _mm512_store_pd(Escr+96, e7r); _mm512_store_pd(Escr+104, e7i);
    _mm512_store_pd(Escr+112, e8r); _mm512_store_pd(Escr+120, e8i);
    }
    { __m512d s1 = _mm512_set1_pd(0x1.71e955d8e7cdcp-2), s2 = _mm512_set1_pd(0x1.58eea2a9d6da3p-1), s3 = _mm512_set1_pd(0x1.ca52d7c9e640bp-1), s4 = _mm512_set1_pd(0x1.fdd0deb564b22p-1), s5 = _mm512_set1_pd(0x1.ec746923c349fp-1), s6 = _mm512_set1_pd(0x1.9895b6c9a05f7p-1), s7 = _mm512_set1_pd(0x1.0d8884363dd82p-1), s8 = _mm512_set1_pd(0x1.7851aacd6c6b5p-3);
    __m512d o1r = _mm512_setzero_pd(), o1i = _mm512_setzero_pd();
    __m512d o2r = _mm512_setzero_pd(), o2i = _mm512_setzero_pd();
    __m512d o3r = _mm512_setzero_pd(), o3i = _mm512_setzero_pd();
    __m512d o4r = _mm512_setzero_pd(), o4i = _mm512_setzero_pd();
    { __m512d vr = _mm512_load_pd(AB+16);
      __m512d vi = _mm512_load_pd(AB+24);
      o1r = _mm512_fmadd_pd(s1, vr, o1r); o1i = _mm512_fmadd_pd(s1, vi, o1i);
      o2r = _mm512_fmadd_pd(s2, vr, o2r); o2i = _mm512_fmadd_pd(s2, vi, o2i);
      o3r = _mm512_fmadd_pd(s3, vr, o3r); o3i = _mm512_fmadd_pd(s3, vi, o3i);
      o4r = _mm512_fmadd_pd(s4, vr, o4r); o4i = _mm512_fmadd_pd(s4, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+48);
      __m512d vi = _mm512_load_pd(AB+56);
      o1r = _mm512_fmadd_pd(s2, vr, o1r); o1i = _mm512_fmadd_pd(s2, vi, o1i);
      o2r = _mm512_fmadd_pd(s4, vr, o2r); o2i = _mm512_fmadd_pd(s4, vi, o2i);
      o3r = _mm512_fmadd_pd(s6, vr, o3r); o3i = _mm512_fmadd_pd(s6, vi, o3i);
      o4r = _mm512_fmadd_pd(s8, vr, o4r); o4i = _mm512_fmadd_pd(s8, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+80);
      __m512d vi = _mm512_load_pd(AB+88);
      o1r = _mm512_fmadd_pd(s3, vr, o1r); o1i = _mm512_fmadd_pd(s3, vi, o1i);
      o2r = _mm512_fmadd_pd(s6, vr, o2r); o2i = _mm512_fmadd_pd(s6, vi, o2i);
      o3r = _mm512_fnmadd_pd(s8, vr, o3r); o3i = _mm512_fnmadd_pd(s8, vi, o3i);
      o4r = _mm512_fnmadd_pd(s5, vr, o4r); o4i = _mm512_fnmadd_pd(s5, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+112);
      __m512d vi = _mm512_load_pd(AB+120);
      o1r = _mm512_fmadd_pd(s4, vr, o1r); o1i = _mm512_fmadd_pd(s4, vi, o1i);
      o2r = _mm512_fmadd_pd(s8, vr, o2r); o2i = _mm512_fmadd_pd(s8, vi, o2i);
      o3r = _mm512_fnmadd_pd(s5, vr, o3r); o3i = _mm512_fnmadd_pd(s5, vi, o3i);
      o4r = _mm512_fnmadd_pd(s1, vr, o4r); o4i = _mm512_fnmadd_pd(s1, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+144);
      __m512d vi = _mm512_load_pd(AB+152);
      o1r = _mm512_fmadd_pd(s5, vr, o1r); o1i = _mm512_fmadd_pd(s5, vi, o1i);
      o2r = _mm512_fnmadd_pd(s7, vr, o2r); o2i = _mm512_fnmadd_pd(s7, vi, o2i);
      o3r = _mm512_fnmadd_pd(s2, vr, o3r); o3i = _mm512_fnmadd_pd(s2, vi, o3i);
      o4r = _mm512_fmadd_pd(s3, vr, o4r); o4i = _mm512_fmadd_pd(s3, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+176);
      __m512d vi = _mm512_load_pd(AB+184);
      o1r = _mm512_fmadd_pd(s6, vr, o1r); o1i = _mm512_fmadd_pd(s6, vi, o1i);
      o2r = _mm512_fnmadd_pd(s5, vr, o2r); o2i = _mm512_fnmadd_pd(s5, vi, o2i);
      o3r = _mm512_fmadd_pd(s1, vr, o3r); o3i = _mm512_fmadd_pd(s1, vi, o3i);
      o4r = _mm512_fmadd_pd(s7, vr, o4r); o4i = _mm512_fmadd_pd(s7, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+208);
      __m512d vi = _mm512_load_pd(AB+216);
      o1r = _mm512_fmadd_pd(s7, vr, o1r); o1i = _mm512_fmadd_pd(s7, vi, o1i);
      o2r = _mm512_fnmadd_pd(s3, vr, o2r); o2i = _mm512_fnmadd_pd(s3, vi, o2i);
      o3r = _mm512_fmadd_pd(s4, vr, o3r); o3i = _mm512_fmadd_pd(s4, vi, o3i);
      o4r = _mm512_fnmadd_pd(s6, vr, o4r); o4i = _mm512_fnmadd_pd(s6, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+240);
      __m512d vi = _mm512_load_pd(AB+248);
      o1r = _mm512_fmadd_pd(s8, vr, o1r); o1i = _mm512_fmadd_pd(s8, vi, o1i);
      o2r = _mm512_fnmadd_pd(s1, vr, o2r); o2i = _mm512_fnmadd_pd(s1, vi, o2i);
      o3r = _mm512_fmadd_pd(s7, vr, o3r); o3i = _mm512_fmadd_pd(s7, vi, o3i);
      o4r = _mm512_fnmadd_pd(s2, vr, o4r); o4i = _mm512_fnmadd_pd(s2, vi, o4i);
    }
    { __m512d er = _mm512_load_pd(Escr+0), ei = _mm512_load_pd(Escr+8);
      _mm512_store_pd(X+1*s,   _mm512_add_pd(er, o1i));
      _mm512_store_pd(X+1*s+8, _mm512_sub_pd(ei, o1r));
      _mm512_store_pd(X+16*s,   _mm512_sub_pd(er, o1i));
      _mm512_store_pd(X+16*s+8, _mm512_add_pd(ei, o1r)); }
    { __m512d er = _mm512_load_pd(Escr+16), ei = _mm512_load_pd(Escr+24);
      _mm512_store_pd(X+2*s,   _mm512_add_pd(er, o2i));
      _mm512_store_pd(X+2*s+8, _mm512_sub_pd(ei, o2r));
      _mm512_store_pd(X+15*s,   _mm512_sub_pd(er, o2i));
      _mm512_store_pd(X+15*s+8, _mm512_add_pd(ei, o2r)); }
    { __m512d er = _mm512_load_pd(Escr+32), ei = _mm512_load_pd(Escr+40);
      _mm512_store_pd(X+3*s,   _mm512_add_pd(er, o3i));
      _mm512_store_pd(X+3*s+8, _mm512_sub_pd(ei, o3r));
      _mm512_store_pd(X+14*s,   _mm512_sub_pd(er, o3i));
      _mm512_store_pd(X+14*s+8, _mm512_add_pd(ei, o3r)); }
    { __m512d er = _mm512_load_pd(Escr+48), ei = _mm512_load_pd(Escr+56);
      _mm512_store_pd(X+4*s,   _mm512_add_pd(er, o4i));
      _mm512_store_pd(X+4*s+8, _mm512_sub_pd(ei, o4r));
      _mm512_store_pd(X+13*s,   _mm512_sub_pd(er, o4i));
      _mm512_store_pd(X+13*s+8, _mm512_add_pd(ei, o4r)); }
    }
    { __m512d s1 = _mm512_set1_pd(0x1.71e955d8e7cdcp-2), s2 = _mm512_set1_pd(0x1.58eea2a9d6da3p-1), s3 = _mm512_set1_pd(0x1.ca52d7c9e640bp-1), s4 = _mm512_set1_pd(0x1.fdd0deb564b22p-1), s5 = _mm512_set1_pd(0x1.ec746923c349fp-1), s6 = _mm512_set1_pd(0x1.9895b6c9a05f7p-1), s7 = _mm512_set1_pd(0x1.0d8884363dd82p-1), s8 = _mm512_set1_pd(0x1.7851aacd6c6b5p-3);
    __m512d o5r = _mm512_setzero_pd(), o5i = _mm512_setzero_pd();
    __m512d o6r = _mm512_setzero_pd(), o6i = _mm512_setzero_pd();
    __m512d o7r = _mm512_setzero_pd(), o7i = _mm512_setzero_pd();
    __m512d o8r = _mm512_setzero_pd(), o8i = _mm512_setzero_pd();
    { __m512d vr = _mm512_load_pd(AB+16);
      __m512d vi = _mm512_load_pd(AB+24);
      o5r = _mm512_fmadd_pd(s5, vr, o5r); o5i = _mm512_fmadd_pd(s5, vi, o5i);
      o6r = _mm512_fmadd_pd(s6, vr, o6r); o6i = _mm512_fmadd_pd(s6, vi, o6i);
      o7r = _mm512_fmadd_pd(s7, vr, o7r); o7i = _mm512_fmadd_pd(s7, vi, o7i);
      o8r = _mm512_fmadd_pd(s8, vr, o8r); o8i = _mm512_fmadd_pd(s8, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+48);
      __m512d vi = _mm512_load_pd(AB+56);
      o5r = _mm512_fnmadd_pd(s7, vr, o5r); o5i = _mm512_fnmadd_pd(s7, vi, o5i);
      o6r = _mm512_fnmadd_pd(s5, vr, o6r); o6i = _mm512_fnmadd_pd(s5, vi, o6i);
      o7r = _mm512_fnmadd_pd(s3, vr, o7r); o7i = _mm512_fnmadd_pd(s3, vi, o7i);
      o8r = _mm512_fnmadd_pd(s1, vr, o8r); o8i = _mm512_fnmadd_pd(s1, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+80);
      __m512d vi = _mm512_load_pd(AB+88);
      o5r = _mm512_fnmadd_pd(s2, vr, o5r); o5i = _mm512_fnmadd_pd(s2, vi, o5i);
      o6r = _mm512_fmadd_pd(s1, vr, o6r); o6i = _mm512_fmadd_pd(s1, vi, o6i);
      o7r = _mm512_fmadd_pd(s4, vr, o7r); o7i = _mm512_fmadd_pd(s4, vi, o7i);
      o8r = _mm512_fmadd_pd(s7, vr, o8r); o8i = _mm512_fmadd_pd(s7, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+112);
      __m512d vi = _mm512_load_pd(AB+120);
      o5r = _mm512_fmadd_pd(s3, vr, o5r); o5i = _mm512_fmadd_pd(s3, vi, o5i);
      o6r = _mm512_fmadd_pd(s7, vr, o6r); o6i = _mm512_fmadd_pd(s7, vi, o6i);
      o7r = _mm512_fnmadd_pd(s6, vr, o7r); o7i = _mm512_fnmadd_pd(s6, vi, o7i);
      o8r = _mm512_fnmadd_pd(s2, vr, o8r); o8i = _mm512_fnmadd_pd(s2, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+144);
      __m512d vi = _mm512_load_pd(AB+152);
      o5r = _mm512_fmadd_pd(s8, vr, o5r); o5i = _mm512_fmadd_pd(s8, vi, o5i);
      o6r = _mm512_fnmadd_pd(s4, vr, o6r); o6i = _mm512_fnmadd_pd(s4, vi, o6i);
      o7r = _mm512_fmadd_pd(s1, vr, o7r); o7i = _mm512_fmadd_pd(s1, vi, o7i);
      o8r = _mm512_fmadd_pd(s6, vr, o8r); o8i = _mm512_fmadd_pd(s6, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+176);
      __m512d vi = _mm512_load_pd(AB+184);
      o5r = _mm512_fnmadd_pd(s4, vr, o5r); o5i = _mm512_fnmadd_pd(s4, vi, o5i);
      o6r = _mm512_fmadd_pd(s2, vr, o6r); o6i = _mm512_fmadd_pd(s2, vi, o6i);
      o7r = _mm512_fmadd_pd(s8, vr, o7r); o7i = _mm512_fmadd_pd(s8, vi, o7i);
      o8r = _mm512_fnmadd_pd(s3, vr, o8r); o8i = _mm512_fnmadd_pd(s3, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+208);
      __m512d vi = _mm512_load_pd(AB+216);
      o5r = _mm512_fmadd_pd(s1, vr, o5r); o5i = _mm512_fmadd_pd(s1, vi, o5i);
      o6r = _mm512_fmadd_pd(s8, vr, o6r); o6i = _mm512_fmadd_pd(s8, vi, o6i);
      o7r = _mm512_fnmadd_pd(s2, vr, o7r); o7i = _mm512_fnmadd_pd(s2, vi, o7i);
      o8r = _mm512_fmadd_pd(s5, vr, o8r); o8i = _mm512_fmadd_pd(s5, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+240);
      __m512d vi = _mm512_load_pd(AB+248);
      o5r = _mm512_fmadd_pd(s6, vr, o5r); o5i = _mm512_fmadd_pd(s6, vi, o5i);
      o6r = _mm512_fnmadd_pd(s3, vr, o6r); o6i = _mm512_fnmadd_pd(s3, vi, o6i);
      o7r = _mm512_fmadd_pd(s5, vr, o7r); o7i = _mm512_fmadd_pd(s5, vi, o7i);
      o8r = _mm512_fnmadd_pd(s4, vr, o8r); o8i = _mm512_fnmadd_pd(s4, vi, o8i);
    }
    { __m512d er = _mm512_load_pd(Escr+64), ei = _mm512_load_pd(Escr+72);
      _mm512_store_pd(X+5*s,   _mm512_add_pd(er, o5i));
      _mm512_store_pd(X+5*s+8, _mm512_sub_pd(ei, o5r));
      _mm512_store_pd(X+12*s,   _mm512_sub_pd(er, o5i));
      _mm512_store_pd(X+12*s+8, _mm512_add_pd(ei, o5r)); }
    { __m512d er = _mm512_load_pd(Escr+80), ei = _mm512_load_pd(Escr+88);
      _mm512_store_pd(X+6*s,   _mm512_add_pd(er, o6i));
      _mm512_store_pd(X+6*s+8, _mm512_sub_pd(ei, o6r));
      _mm512_store_pd(X+11*s,   _mm512_sub_pd(er, o6i));
      _mm512_store_pd(X+11*s+8, _mm512_add_pd(ei, o6r)); }
    { __m512d er = _mm512_load_pd(Escr+96), ei = _mm512_load_pd(Escr+104);
      _mm512_store_pd(X+7*s,   _mm512_add_pd(er, o7i));
      _mm512_store_pd(X+7*s+8, _mm512_sub_pd(ei, o7r));
      _mm512_store_pd(X+10*s,   _mm512_sub_pd(er, o7i));
      _mm512_store_pd(X+10*s+8, _mm512_add_pd(ei, o7r)); }
    { __m512d er = _mm512_load_pd(Escr+112), ei = _mm512_load_pd(Escr+120);
      _mm512_store_pd(X+8*s,   _mm512_add_pd(er, o8i));
      _mm512_store_pd(X+8*s+8, _mm512_sub_pd(ei, o8r));
      _mm512_store_pd(X+9*s,   _mm512_sub_pd(er, o8i));
      _mm512_store_pd(X+9*s+8, _mm512_add_pd(ei, o8r)); }
    }
}
static __attribute__((always_inline)) inline void dft17zm(double* restrict X, long es, int dopf){
    const long s = es*16;
    double AB[64*8] ALIGN64;
    double Escr[16*8] ALIGN64;
    __m512d x0r = _mm512_load_pd(X);
    if(dopf){ _mm_prefetch((const char*)(X+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+280), _MM_HINT_T0); }
    __m512d x0i = _mm512_load_pd(X+8);
    maphw(x0r, x0i, &x0r, &x0i);
    { __m512d c1 = _mm512_set1_pd(0x1.dd6d000370991p-1), c2 = _mm512_set1_pd(0x1.7a5f6075d4884p-1), c3 = _mm512_set1_pd(0x1.c86fa2b2883cep-2), c4 = _mm512_set1_pd(0x1.79ee63259b75fp-4), c5 = _mm512_set1_pd(-0x1.183b1c61f0d01p-2), c6 = _mm512_set1_pd(-0x1.348c86ed5f1bap-1), c7 = _mm512_set1_pd(-0x1.b34fa910ea3b8p-1), c8 = _mm512_set1_pd(-0x1.f7484007faef3p-1);
    __m512d e1r = x0r, e1i = x0i;
    __m512d e2r = x0r, e2i = x0i;
    __m512d e3r = x0r, e3i = x0i;
    __m512d e4r = x0r, e4i = x0i;
    __m512d sr = x0r, si = x0i;
    { __m512d pr = _mm512_load_pd(X+1*s), qr = _mm512_load_pd(X+16*s);
      if(dopf){ _mm_prefetch((const char*)(X+1*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+1*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+16*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+16*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+1*s+8), qi = _mm512_load_pd(X+16*s+8);
      map2(pr, pi, &pr, &pi); maphw(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+0, ur);    _mm512_store_pd(AB+8, ui);
      _mm512_store_pd(AB+16, vr); _mm512_store_pd(AB+24, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c1, ur, e1r); e1i = _mm512_fmadd_pd(c1, ui, e1i);
      e2r = _mm512_fmadd_pd(c2, ur, e2r); e2i = _mm512_fmadd_pd(c2, ui, e2i);
      e3r = _mm512_fmadd_pd(c3, ur, e3r); e3i = _mm512_fmadd_pd(c3, ui, e3i);
      e4r = _mm512_fmadd_pd(c4, ur, e4r); e4i = _mm512_fmadd_pd(c4, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+2*s), qr = _mm512_load_pd(X+15*s);
      if(dopf){ _mm_prefetch((const char*)(X+2*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+2*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+15*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+15*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+2*s+8), qi = _mm512_load_pd(X+15*s+8);
      maphw(pr, pi, &pr, &pi); map2(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+32, ur);    _mm512_store_pd(AB+40, ui);
      _mm512_store_pd(AB+48, vr); _mm512_store_pd(AB+56, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c2, ur, e1r); e1i = _mm512_fmadd_pd(c2, ui, e1i);
      e2r = _mm512_fmadd_pd(c4, ur, e2r); e2i = _mm512_fmadd_pd(c4, ui, e2i);
      e3r = _mm512_fmadd_pd(c6, ur, e3r); e3i = _mm512_fmadd_pd(c6, ui, e3i);
      e4r = _mm512_fmadd_pd(c8, ur, e4r); e4i = _mm512_fmadd_pd(c8, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+3*s), qr = _mm512_load_pd(X+14*s);
      if(dopf){ _mm_prefetch((const char*)(X+3*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+3*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+14*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+14*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+3*s+8), qi = _mm512_load_pd(X+14*s+8);
      map2(pr, pi, &pr, &pi); maphw(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+64, ur);    _mm512_store_pd(AB+72, ui);
      _mm512_store_pd(AB+80, vr); _mm512_store_pd(AB+88, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c3, ur, e1r); e1i = _mm512_fmadd_pd(c3, ui, e1i);
      e2r = _mm512_fmadd_pd(c6, ur, e2r); e2i = _mm512_fmadd_pd(c6, ui, e2i);
      e3r = _mm512_fmadd_pd(c8, ur, e3r); e3i = _mm512_fmadd_pd(c8, ui, e3i);
      e4r = _mm512_fmadd_pd(c5, ur, e4r); e4i = _mm512_fmadd_pd(c5, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+4*s), qr = _mm512_load_pd(X+13*s);
      if(dopf){ _mm_prefetch((const char*)(X+4*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+4*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+13*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+13*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+4*s+8), qi = _mm512_load_pd(X+13*s+8);
      maphw(pr, pi, &pr, &pi); map2(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+96, ur);    _mm512_store_pd(AB+104, ui);
      _mm512_store_pd(AB+112, vr); _mm512_store_pd(AB+120, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c4, ur, e1r); e1i = _mm512_fmadd_pd(c4, ui, e1i);
      e2r = _mm512_fmadd_pd(c8, ur, e2r); e2i = _mm512_fmadd_pd(c8, ui, e2i);
      e3r = _mm512_fmadd_pd(c5, ur, e3r); e3i = _mm512_fmadd_pd(c5, ui, e3i);
      e4r = _mm512_fmadd_pd(c1, ur, e4r); e4i = _mm512_fmadd_pd(c1, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+5*s), qr = _mm512_load_pd(X+12*s);
      if(dopf){ _mm_prefetch((const char*)(X+5*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+5*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+12*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+12*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+5*s+8), qi = _mm512_load_pd(X+12*s+8);
      map2(pr, pi, &pr, &pi); maphw(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+128, ur);    _mm512_store_pd(AB+136, ui);
      _mm512_store_pd(AB+144, vr); _mm512_store_pd(AB+152, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c5, ur, e1r); e1i = _mm512_fmadd_pd(c5, ui, e1i);
      e2r = _mm512_fmadd_pd(c7, ur, e2r); e2i = _mm512_fmadd_pd(c7, ui, e2i);
      e3r = _mm512_fmadd_pd(c2, ur, e3r); e3i = _mm512_fmadd_pd(c2, ui, e3i);
      e4r = _mm512_fmadd_pd(c3, ur, e4r); e4i = _mm512_fmadd_pd(c3, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+6*s), qr = _mm512_load_pd(X+11*s);
      if(dopf){ _mm_prefetch((const char*)(X+6*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+6*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+11*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+11*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+6*s+8), qi = _mm512_load_pd(X+11*s+8);
      maphw(pr, pi, &pr, &pi); map2(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+160, ur);    _mm512_store_pd(AB+168, ui);
      _mm512_store_pd(AB+176, vr); _mm512_store_pd(AB+184, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c6, ur, e1r); e1i = _mm512_fmadd_pd(c6, ui, e1i);
      e2r = _mm512_fmadd_pd(c5, ur, e2r); e2i = _mm512_fmadd_pd(c5, ui, e2i);
      e3r = _mm512_fmadd_pd(c1, ur, e3r); e3i = _mm512_fmadd_pd(c1, ui, e3i);
      e4r = _mm512_fmadd_pd(c7, ur, e4r); e4i = _mm512_fmadd_pd(c7, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+7*s), qr = _mm512_load_pd(X+10*s);
      if(dopf){ _mm_prefetch((const char*)(X+7*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+7*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+10*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+10*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+7*s+8), qi = _mm512_load_pd(X+10*s+8);
      map2(pr, pi, &pr, &pi); maphw(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+192, ur);    _mm512_store_pd(AB+200, ui);
      _mm512_store_pd(AB+208, vr); _mm512_store_pd(AB+216, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c7, ur, e1r); e1i = _mm512_fmadd_pd(c7, ui, e1i);
      e2r = _mm512_fmadd_pd(c3, ur, e2r); e2i = _mm512_fmadd_pd(c3, ui, e2i);
      e3r = _mm512_fmadd_pd(c4, ur, e3r); e3i = _mm512_fmadd_pd(c4, ui, e3i);
      e4r = _mm512_fmadd_pd(c6, ur, e4r); e4i = _mm512_fmadd_pd(c6, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+8*s), qr = _mm512_load_pd(X+9*s);
      if(dopf){ _mm_prefetch((const char*)(X+8*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+8*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+9*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+9*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+8*s+8), qi = _mm512_load_pd(X+9*s+8);
      maphw(pr, pi, &pr, &pi); map2(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+224, ur);    _mm512_store_pd(AB+232, ui);
      _mm512_store_pd(AB+240, vr); _mm512_store_pd(AB+248, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c8, ur, e1r); e1i = _mm512_fmadd_pd(c8, ui, e1i);
      e2r = _mm512_fmadd_pd(c1, ur, e2r); e2i = _mm512_fmadd_pd(c1, ui, e2i);
      e3r = _mm512_fmadd_pd(c7, ur, e3r); e3i = _mm512_fmadd_pd(c7, ui, e3i);
      e4r = _mm512_fmadd_pd(c2, ur, e4r); e4i = _mm512_fmadd_pd(c2, ui, e4i);
    }
    _mm512_store_pd(X, sr); _mm512_store_pd(X+8, si);
    _mm512_store_pd(Escr+0, e1r); _mm512_store_pd(Escr+8, e1i);
    _mm512_store_pd(Escr+16, e2r); _mm512_store_pd(Escr+24, e2i);
    _mm512_store_pd(Escr+32, e3r); _mm512_store_pd(Escr+40, e3i);
    _mm512_store_pd(Escr+48, e4r); _mm512_store_pd(Escr+56, e4i);
    }
    { __m512d c1 = _mm512_set1_pd(0x1.dd6d000370991p-1), c2 = _mm512_set1_pd(0x1.7a5f6075d4884p-1), c3 = _mm512_set1_pd(0x1.c86fa2b2883cep-2), c4 = _mm512_set1_pd(0x1.79ee63259b75fp-4), c5 = _mm512_set1_pd(-0x1.183b1c61f0d01p-2), c6 = _mm512_set1_pd(-0x1.348c86ed5f1bap-1), c7 = _mm512_set1_pd(-0x1.b34fa910ea3b8p-1), c8 = _mm512_set1_pd(-0x1.f7484007faef3p-1);
    __m512d e5r = x0r, e5i = x0i;
    __m512d e6r = x0r, e6i = x0i;
    __m512d e7r = x0r, e7i = x0i;
    __m512d e8r = x0r, e8i = x0i;
    { __m512d ur = _mm512_load_pd(AB+0);
      __m512d ui = _mm512_load_pd(AB+8);
      e5r = _mm512_fmadd_pd(c5, ur, e5r); e5i = _mm512_fmadd_pd(c5, ui, e5i);
      e6r = _mm512_fmadd_pd(c6, ur, e6r); e6i = _mm512_fmadd_pd(c6, ui, e6i);
      e7r = _mm512_fmadd_pd(c7, ur, e7r); e7i = _mm512_fmadd_pd(c7, ui, e7i);
      e8r = _mm512_fmadd_pd(c8, ur, e8r); e8i = _mm512_fmadd_pd(c8, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+32);
      __m512d ui = _mm512_load_pd(AB+40);
      e5r = _mm512_fmadd_pd(c7, ur, e5r); e5i = _mm512_fmadd_pd(c7, ui, e5i);
      e6r = _mm512_fmadd_pd(c5, ur, e6r); e6i = _mm512_fmadd_pd(c5, ui, e6i);
      e7r = _mm512_fmadd_pd(c3, ur, e7r); e7i = _mm512_fmadd_pd(c3, ui, e7i);
      e8r = _mm512_fmadd_pd(c1, ur, e8r); e8i = _mm512_fmadd_pd(c1, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+64);
      __m512d ui = _mm512_load_pd(AB+72);
      e5r = _mm512_fmadd_pd(c2, ur, e5r); e5i = _mm512_fmadd_pd(c2, ui, e5i);
      e6r = _mm512_fmadd_pd(c1, ur, e6r); e6i = _mm512_fmadd_pd(c1, ui, e6i);
      e7r = _mm512_fmadd_pd(c4, ur, e7r); e7i = _mm512_fmadd_pd(c4, ui, e7i);
      e8r = _mm512_fmadd_pd(c7, ur, e8r); e8i = _mm512_fmadd_pd(c7, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+96);
      __m512d ui = _mm512_load_pd(AB+104);
      e5r = _mm512_fmadd_pd(c3, ur, e5r); e5i = _mm512_fmadd_pd(c3, ui, e5i);
      e6r = _mm512_fmadd_pd(c7, ur, e6r); e6i = _mm512_fmadd_pd(c7, ui, e6i);
      e7r = _mm512_fmadd_pd(c6, ur, e7r); e7i = _mm512_fmadd_pd(c6, ui, e7i);
      e8r = _mm512_fmadd_pd(c2, ur, e8r); e8i = _mm512_fmadd_pd(c2, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+128);
      __m512d ui = _mm512_load_pd(AB+136);
      e5r = _mm512_fmadd_pd(c8, ur, e5r); e5i = _mm512_fmadd_pd(c8, ui, e5i);
      e6r = _mm512_fmadd_pd(c4, ur, e6r); e6i = _mm512_fmadd_pd(c4, ui, e6i);
      e7r = _mm512_fmadd_pd(c1, ur, e7r); e7i = _mm512_fmadd_pd(c1, ui, e7i);
      e8r = _mm512_fmadd_pd(c6, ur, e8r); e8i = _mm512_fmadd_pd(c6, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+160);
      __m512d ui = _mm512_load_pd(AB+168);
      e5r = _mm512_fmadd_pd(c4, ur, e5r); e5i = _mm512_fmadd_pd(c4, ui, e5i);
      e6r = _mm512_fmadd_pd(c2, ur, e6r); e6i = _mm512_fmadd_pd(c2, ui, e6i);
      e7r = _mm512_fmadd_pd(c8, ur, e7r); e7i = _mm512_fmadd_pd(c8, ui, e7i);
      e8r = _mm512_fmadd_pd(c3, ur, e8r); e8i = _mm512_fmadd_pd(c3, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+192);
      __m512d ui = _mm512_load_pd(AB+200);
      e5r = _mm512_fmadd_pd(c1, ur, e5r); e5i = _mm512_fmadd_pd(c1, ui, e5i);
      e6r = _mm512_fmadd_pd(c8, ur, e6r); e6i = _mm512_fmadd_pd(c8, ui, e6i);
      e7r = _mm512_fmadd_pd(c2, ur, e7r); e7i = _mm512_fmadd_pd(c2, ui, e7i);
      e8r = _mm512_fmadd_pd(c5, ur, e8r); e8i = _mm512_fmadd_pd(c5, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+224);
      __m512d ui = _mm512_load_pd(AB+232);
      e5r = _mm512_fmadd_pd(c6, ur, e5r); e5i = _mm512_fmadd_pd(c6, ui, e5i);
      e6r = _mm512_fmadd_pd(c3, ur, e6r); e6i = _mm512_fmadd_pd(c3, ui, e6i);
      e7r = _mm512_fmadd_pd(c5, ur, e7r); e7i = _mm512_fmadd_pd(c5, ui, e7i);
      e8r = _mm512_fmadd_pd(c4, ur, e8r); e8i = _mm512_fmadd_pd(c4, ui, e8i);
    }
    _mm512_store_pd(Escr+64, e5r); _mm512_store_pd(Escr+72, e5i);
    _mm512_store_pd(Escr+80, e6r); _mm512_store_pd(Escr+88, e6i);
    _mm512_store_pd(Escr+96, e7r); _mm512_store_pd(Escr+104, e7i);
    _mm512_store_pd(Escr+112, e8r); _mm512_store_pd(Escr+120, e8i);
    }
    { __m512d s1 = _mm512_set1_pd(0x1.71e955d8e7cdcp-2), s2 = _mm512_set1_pd(0x1.58eea2a9d6da3p-1), s3 = _mm512_set1_pd(0x1.ca52d7c9e640bp-1), s4 = _mm512_set1_pd(0x1.fdd0deb564b22p-1), s5 = _mm512_set1_pd(0x1.ec746923c349fp-1), s6 = _mm512_set1_pd(0x1.9895b6c9a05f7p-1), s7 = _mm512_set1_pd(0x1.0d8884363dd82p-1), s8 = _mm512_set1_pd(0x1.7851aacd6c6b5p-3);
    __m512d o1r = _mm512_setzero_pd(), o1i = _mm512_setzero_pd();
    __m512d o2r = _mm512_setzero_pd(), o2i = _mm512_setzero_pd();
    __m512d o3r = _mm512_setzero_pd(), o3i = _mm512_setzero_pd();
    __m512d o4r = _mm512_setzero_pd(), o4i = _mm512_setzero_pd();
    { __m512d vr = _mm512_load_pd(AB+16);
      __m512d vi = _mm512_load_pd(AB+24);
      o1r = _mm512_fmadd_pd(s1, vr, o1r); o1i = _mm512_fmadd_pd(s1, vi, o1i);
      o2r = _mm512_fmadd_pd(s2, vr, o2r); o2i = _mm512_fmadd_pd(s2, vi, o2i);
      o3r = _mm512_fmadd_pd(s3, vr, o3r); o3i = _mm512_fmadd_pd(s3, vi, o3i);
      o4r = _mm512_fmadd_pd(s4, vr, o4r); o4i = _mm512_fmadd_pd(s4, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+48);
      __m512d vi = _mm512_load_pd(AB+56);
      o1r = _mm512_fmadd_pd(s2, vr, o1r); o1i = _mm512_fmadd_pd(s2, vi, o1i);
      o2r = _mm512_fmadd_pd(s4, vr, o2r); o2i = _mm512_fmadd_pd(s4, vi, o2i);
      o3r = _mm512_fmadd_pd(s6, vr, o3r); o3i = _mm512_fmadd_pd(s6, vi, o3i);
      o4r = _mm512_fmadd_pd(s8, vr, o4r); o4i = _mm512_fmadd_pd(s8, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+80);
      __m512d vi = _mm512_load_pd(AB+88);
      o1r = _mm512_fmadd_pd(s3, vr, o1r); o1i = _mm512_fmadd_pd(s3, vi, o1i);
      o2r = _mm512_fmadd_pd(s6, vr, o2r); o2i = _mm512_fmadd_pd(s6, vi, o2i);
      o3r = _mm512_fnmadd_pd(s8, vr, o3r); o3i = _mm512_fnmadd_pd(s8, vi, o3i);
      o4r = _mm512_fnmadd_pd(s5, vr, o4r); o4i = _mm512_fnmadd_pd(s5, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+112);
      __m512d vi = _mm512_load_pd(AB+120);
      o1r = _mm512_fmadd_pd(s4, vr, o1r); o1i = _mm512_fmadd_pd(s4, vi, o1i);
      o2r = _mm512_fmadd_pd(s8, vr, o2r); o2i = _mm512_fmadd_pd(s8, vi, o2i);
      o3r = _mm512_fnmadd_pd(s5, vr, o3r); o3i = _mm512_fnmadd_pd(s5, vi, o3i);
      o4r = _mm512_fnmadd_pd(s1, vr, o4r); o4i = _mm512_fnmadd_pd(s1, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+144);
      __m512d vi = _mm512_load_pd(AB+152);
      o1r = _mm512_fmadd_pd(s5, vr, o1r); o1i = _mm512_fmadd_pd(s5, vi, o1i);
      o2r = _mm512_fnmadd_pd(s7, vr, o2r); o2i = _mm512_fnmadd_pd(s7, vi, o2i);
      o3r = _mm512_fnmadd_pd(s2, vr, o3r); o3i = _mm512_fnmadd_pd(s2, vi, o3i);
      o4r = _mm512_fmadd_pd(s3, vr, o4r); o4i = _mm512_fmadd_pd(s3, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+176);
      __m512d vi = _mm512_load_pd(AB+184);
      o1r = _mm512_fmadd_pd(s6, vr, o1r); o1i = _mm512_fmadd_pd(s6, vi, o1i);
      o2r = _mm512_fnmadd_pd(s5, vr, o2r); o2i = _mm512_fnmadd_pd(s5, vi, o2i);
      o3r = _mm512_fmadd_pd(s1, vr, o3r); o3i = _mm512_fmadd_pd(s1, vi, o3i);
      o4r = _mm512_fmadd_pd(s7, vr, o4r); o4i = _mm512_fmadd_pd(s7, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+208);
      __m512d vi = _mm512_load_pd(AB+216);
      o1r = _mm512_fmadd_pd(s7, vr, o1r); o1i = _mm512_fmadd_pd(s7, vi, o1i);
      o2r = _mm512_fnmadd_pd(s3, vr, o2r); o2i = _mm512_fnmadd_pd(s3, vi, o2i);
      o3r = _mm512_fmadd_pd(s4, vr, o3r); o3i = _mm512_fmadd_pd(s4, vi, o3i);
      o4r = _mm512_fnmadd_pd(s6, vr, o4r); o4i = _mm512_fnmadd_pd(s6, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+240);
      __m512d vi = _mm512_load_pd(AB+248);
      o1r = _mm512_fmadd_pd(s8, vr, o1r); o1i = _mm512_fmadd_pd(s8, vi, o1i);
      o2r = _mm512_fnmadd_pd(s1, vr, o2r); o2i = _mm512_fnmadd_pd(s1, vi, o2i);
      o3r = _mm512_fmadd_pd(s7, vr, o3r); o3i = _mm512_fmadd_pd(s7, vi, o3i);
      o4r = _mm512_fnmadd_pd(s2, vr, o4r); o4i = _mm512_fnmadd_pd(s2, vi, o4i);
    }
    { __m512d er = _mm512_load_pd(Escr+0), ei = _mm512_load_pd(Escr+8);
      _mm512_store_pd(X+1*s,   _mm512_add_pd(er, o1i));
      _mm512_store_pd(X+1*s+8, _mm512_sub_pd(ei, o1r));
      _mm512_store_pd(X+16*s,   _mm512_sub_pd(er, o1i));
      _mm512_store_pd(X+16*s+8, _mm512_add_pd(ei, o1r)); }
    { __m512d er = _mm512_load_pd(Escr+16), ei = _mm512_load_pd(Escr+24);
      _mm512_store_pd(X+2*s,   _mm512_add_pd(er, o2i));
      _mm512_store_pd(X+2*s+8, _mm512_sub_pd(ei, o2r));
      _mm512_store_pd(X+15*s,   _mm512_sub_pd(er, o2i));
      _mm512_store_pd(X+15*s+8, _mm512_add_pd(ei, o2r)); }
    { __m512d er = _mm512_load_pd(Escr+32), ei = _mm512_load_pd(Escr+40);
      _mm512_store_pd(X+3*s,   _mm512_add_pd(er, o3i));
      _mm512_store_pd(X+3*s+8, _mm512_sub_pd(ei, o3r));
      _mm512_store_pd(X+14*s,   _mm512_sub_pd(er, o3i));
      _mm512_store_pd(X+14*s+8, _mm512_add_pd(ei, o3r)); }
    { __m512d er = _mm512_load_pd(Escr+48), ei = _mm512_load_pd(Escr+56);
      _mm512_store_pd(X+4*s,   _mm512_add_pd(er, o4i));
      _mm512_store_pd(X+4*s+8, _mm512_sub_pd(ei, o4r));
      _mm512_store_pd(X+13*s,   _mm512_sub_pd(er, o4i));
      _mm512_store_pd(X+13*s+8, _mm512_add_pd(ei, o4r)); }
    }
    { __m512d s1 = _mm512_set1_pd(0x1.71e955d8e7cdcp-2), s2 = _mm512_set1_pd(0x1.58eea2a9d6da3p-1), s3 = _mm512_set1_pd(0x1.ca52d7c9e640bp-1), s4 = _mm512_set1_pd(0x1.fdd0deb564b22p-1), s5 = _mm512_set1_pd(0x1.ec746923c349fp-1), s6 = _mm512_set1_pd(0x1.9895b6c9a05f7p-1), s7 = _mm512_set1_pd(0x1.0d8884363dd82p-1), s8 = _mm512_set1_pd(0x1.7851aacd6c6b5p-3);
    __m512d o5r = _mm512_setzero_pd(), o5i = _mm512_setzero_pd();
    __m512d o6r = _mm512_setzero_pd(), o6i = _mm512_setzero_pd();
    __m512d o7r = _mm512_setzero_pd(), o7i = _mm512_setzero_pd();
    __m512d o8r = _mm512_setzero_pd(), o8i = _mm512_setzero_pd();
    { __m512d vr = _mm512_load_pd(AB+16);
      __m512d vi = _mm512_load_pd(AB+24);
      o5r = _mm512_fmadd_pd(s5, vr, o5r); o5i = _mm512_fmadd_pd(s5, vi, o5i);
      o6r = _mm512_fmadd_pd(s6, vr, o6r); o6i = _mm512_fmadd_pd(s6, vi, o6i);
      o7r = _mm512_fmadd_pd(s7, vr, o7r); o7i = _mm512_fmadd_pd(s7, vi, o7i);
      o8r = _mm512_fmadd_pd(s8, vr, o8r); o8i = _mm512_fmadd_pd(s8, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+48);
      __m512d vi = _mm512_load_pd(AB+56);
      o5r = _mm512_fnmadd_pd(s7, vr, o5r); o5i = _mm512_fnmadd_pd(s7, vi, o5i);
      o6r = _mm512_fnmadd_pd(s5, vr, o6r); o6i = _mm512_fnmadd_pd(s5, vi, o6i);
      o7r = _mm512_fnmadd_pd(s3, vr, o7r); o7i = _mm512_fnmadd_pd(s3, vi, o7i);
      o8r = _mm512_fnmadd_pd(s1, vr, o8r); o8i = _mm512_fnmadd_pd(s1, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+80);
      __m512d vi = _mm512_load_pd(AB+88);
      o5r = _mm512_fnmadd_pd(s2, vr, o5r); o5i = _mm512_fnmadd_pd(s2, vi, o5i);
      o6r = _mm512_fmadd_pd(s1, vr, o6r); o6i = _mm512_fmadd_pd(s1, vi, o6i);
      o7r = _mm512_fmadd_pd(s4, vr, o7r); o7i = _mm512_fmadd_pd(s4, vi, o7i);
      o8r = _mm512_fmadd_pd(s7, vr, o8r); o8i = _mm512_fmadd_pd(s7, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+112);
      __m512d vi = _mm512_load_pd(AB+120);
      o5r = _mm512_fmadd_pd(s3, vr, o5r); o5i = _mm512_fmadd_pd(s3, vi, o5i);
      o6r = _mm512_fmadd_pd(s7, vr, o6r); o6i = _mm512_fmadd_pd(s7, vi, o6i);
      o7r = _mm512_fnmadd_pd(s6, vr, o7r); o7i = _mm512_fnmadd_pd(s6, vi, o7i);
      o8r = _mm512_fnmadd_pd(s2, vr, o8r); o8i = _mm512_fnmadd_pd(s2, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+144);
      __m512d vi = _mm512_load_pd(AB+152);
      o5r = _mm512_fmadd_pd(s8, vr, o5r); o5i = _mm512_fmadd_pd(s8, vi, o5i);
      o6r = _mm512_fnmadd_pd(s4, vr, o6r); o6i = _mm512_fnmadd_pd(s4, vi, o6i);
      o7r = _mm512_fmadd_pd(s1, vr, o7r); o7i = _mm512_fmadd_pd(s1, vi, o7i);
      o8r = _mm512_fmadd_pd(s6, vr, o8r); o8i = _mm512_fmadd_pd(s6, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+176);
      __m512d vi = _mm512_load_pd(AB+184);
      o5r = _mm512_fnmadd_pd(s4, vr, o5r); o5i = _mm512_fnmadd_pd(s4, vi, o5i);
      o6r = _mm512_fmadd_pd(s2, vr, o6r); o6i = _mm512_fmadd_pd(s2, vi, o6i);
      o7r = _mm512_fmadd_pd(s8, vr, o7r); o7i = _mm512_fmadd_pd(s8, vi, o7i);
      o8r = _mm512_fnmadd_pd(s3, vr, o8r); o8i = _mm512_fnmadd_pd(s3, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+208);
      __m512d vi = _mm512_load_pd(AB+216);
      o5r = _mm512_fmadd_pd(s1, vr, o5r); o5i = _mm512_fmadd_pd(s1, vi, o5i);
      o6r = _mm512_fmadd_pd(s8, vr, o6r); o6i = _mm512_fmadd_pd(s8, vi, o6i);
      o7r = _mm512_fnmadd_pd(s2, vr, o7r); o7i = _mm512_fnmadd_pd(s2, vi, o7i);
      o8r = _mm512_fmadd_pd(s5, vr, o8r); o8i = _mm512_fmadd_pd(s5, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+240);
      __m512d vi = _mm512_load_pd(AB+248);
      o5r = _mm512_fmadd_pd(s6, vr, o5r); o5i = _mm512_fmadd_pd(s6, vi, o5i);
      o6r = _mm512_fnmadd_pd(s3, vr, o6r); o6i = _mm512_fnmadd_pd(s3, vi, o6i);
      o7r = _mm512_fmadd_pd(s5, vr, o7r); o7i = _mm512_fmadd_pd(s5, vi, o7i);
      o8r = _mm512_fnmadd_pd(s4, vr, o8r); o8i = _mm512_fnmadd_pd(s4, vi, o8i);
    }
    { __m512d er = _mm512_load_pd(Escr+64), ei = _mm512_load_pd(Escr+72);
      _mm512_store_pd(X+5*s,   _mm512_add_pd(er, o5i));
      _mm512_store_pd(X+5*s+8, _mm512_sub_pd(ei, o5r));
      _mm512_store_pd(X+12*s,   _mm512_sub_pd(er, o5i));
      _mm512_store_pd(X+12*s+8, _mm512_add_pd(ei, o5r)); }
    { __m512d er = _mm512_load_pd(Escr+80), ei = _mm512_load_pd(Escr+88);
      _mm512_store_pd(X+6*s,   _mm512_add_pd(er, o6i));
      _mm512_store_pd(X+6*s+8, _mm512_sub_pd(ei, o6r));
      _mm512_store_pd(X+11*s,   _mm512_sub_pd(er, o6i));
      _mm512_store_pd(X+11*s+8, _mm512_add_pd(ei, o6r)); }
    { __m512d er = _mm512_load_pd(Escr+96), ei = _mm512_load_pd(Escr+104);
      _mm512_store_pd(X+7*s,   _mm512_add_pd(er, o7i));
      _mm512_store_pd(X+7*s+8, _mm512_sub_pd(ei, o7r));
      _mm512_store_pd(X+10*s,   _mm512_sub_pd(er, o7i));
      _mm512_store_pd(X+10*s+8, _mm512_add_pd(ei, o7r)); }
    { __m512d er = _mm512_load_pd(Escr+112), ei = _mm512_load_pd(Escr+120);
      _mm512_store_pd(X+8*s,   _mm512_add_pd(er, o8i));
      _mm512_store_pd(X+8*s+8, _mm512_sub_pd(ei, o8r));
      _mm512_store_pd(X+9*s,   _mm512_sub_pd(er, o8i));
      _mm512_store_pd(X+9*s+8, _mm512_add_pd(ei, o8r)); }
    }
}
static __attribute__((always_inline)) inline void dft17m(double* restrict X, long es, int dopf, const double* restrict C, long ces){
    const long s = es*16;
    const long cs = ces*16;
    double AB[64*8] ALIGN64;
    double Escr[16*8] ALIGN64;
    __m512d x0r = _mm512_load_pd(X);
    if(dopf){ _mm_prefetch((const char*)(X+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+280), _MM_HINT_T0); }
    __m512d x0i = _mm512_load_pd(X+8);
    { __m512d c1 = _mm512_set1_pd(0x1.dd6d000370991p-1), c2 = _mm512_set1_pd(0x1.7a5f6075d4884p-1), c3 = _mm512_set1_pd(0x1.c86fa2b2883cep-2), c4 = _mm512_set1_pd(0x1.79ee63259b75fp-4), c5 = _mm512_set1_pd(-0x1.183b1c61f0d01p-2), c6 = _mm512_set1_pd(-0x1.348c86ed5f1bap-1), c7 = _mm512_set1_pd(-0x1.b34fa910ea3b8p-1), c8 = _mm512_set1_pd(-0x1.f7484007faef3p-1);
    __m512d e1r = x0r, e1i = x0i;
    __m512d e2r = x0r, e2i = x0i;
    __m512d e3r = x0r, e3i = x0i;
    __m512d e4r = x0r, e4i = x0i;
    __m512d sr = x0r, si = x0i;
    { __m512d pr = _mm512_load_pd(X+1*s), qr = _mm512_load_pd(X+16*s);
      if(dopf){ _mm_prefetch((const char*)(X+1*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+1*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+16*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+16*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+1*s+8), qi = _mm512_load_pd(X+16*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+0, ur);    _mm512_store_pd(AB+8, ui);
      _mm512_store_pd(AB+16, vr); _mm512_store_pd(AB+24, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c1, ur, e1r); e1i = _mm512_fmadd_pd(c1, ui, e1i);
      e2r = _mm512_fmadd_pd(c2, ur, e2r); e2i = _mm512_fmadd_pd(c2, ui, e2i);
      e3r = _mm512_fmadd_pd(c3, ur, e3r); e3i = _mm512_fmadd_pd(c3, ui, e3i);
      e4r = _mm512_fmadd_pd(c4, ur, e4r); e4i = _mm512_fmadd_pd(c4, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+2*s), qr = _mm512_load_pd(X+15*s);
      if(dopf){ _mm_prefetch((const char*)(X+2*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+2*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+15*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+15*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+2*s+8), qi = _mm512_load_pd(X+15*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+32, ur);    _mm512_store_pd(AB+40, ui);
      _mm512_store_pd(AB+48, vr); _mm512_store_pd(AB+56, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c2, ur, e1r); e1i = _mm512_fmadd_pd(c2, ui, e1i);
      e2r = _mm512_fmadd_pd(c4, ur, e2r); e2i = _mm512_fmadd_pd(c4, ui, e2i);
      e3r = _mm512_fmadd_pd(c6, ur, e3r); e3i = _mm512_fmadd_pd(c6, ui, e3i);
      e4r = _mm512_fmadd_pd(c8, ur, e4r); e4i = _mm512_fmadd_pd(c8, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+3*s), qr = _mm512_load_pd(X+14*s);
      if(dopf){ _mm_prefetch((const char*)(X+3*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+3*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+14*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+14*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+3*s+8), qi = _mm512_load_pd(X+14*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+64, ur);    _mm512_store_pd(AB+72, ui);
      _mm512_store_pd(AB+80, vr); _mm512_store_pd(AB+88, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c3, ur, e1r); e1i = _mm512_fmadd_pd(c3, ui, e1i);
      e2r = _mm512_fmadd_pd(c6, ur, e2r); e2i = _mm512_fmadd_pd(c6, ui, e2i);
      e3r = _mm512_fmadd_pd(c8, ur, e3r); e3i = _mm512_fmadd_pd(c8, ui, e3i);
      e4r = _mm512_fmadd_pd(c5, ur, e4r); e4i = _mm512_fmadd_pd(c5, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+4*s), qr = _mm512_load_pd(X+13*s);
      if(dopf){ _mm_prefetch((const char*)(X+4*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+4*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+13*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+13*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+4*s+8), qi = _mm512_load_pd(X+13*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+96, ur);    _mm512_store_pd(AB+104, ui);
      _mm512_store_pd(AB+112, vr); _mm512_store_pd(AB+120, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c4, ur, e1r); e1i = _mm512_fmadd_pd(c4, ui, e1i);
      e2r = _mm512_fmadd_pd(c8, ur, e2r); e2i = _mm512_fmadd_pd(c8, ui, e2i);
      e3r = _mm512_fmadd_pd(c5, ur, e3r); e3i = _mm512_fmadd_pd(c5, ui, e3i);
      e4r = _mm512_fmadd_pd(c1, ur, e4r); e4i = _mm512_fmadd_pd(c1, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+5*s), qr = _mm512_load_pd(X+12*s);
      if(dopf){ _mm_prefetch((const char*)(X+5*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+5*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+12*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+12*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+5*s+8), qi = _mm512_load_pd(X+12*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+128, ur);    _mm512_store_pd(AB+136, ui);
      _mm512_store_pd(AB+144, vr); _mm512_store_pd(AB+152, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c5, ur, e1r); e1i = _mm512_fmadd_pd(c5, ui, e1i);
      e2r = _mm512_fmadd_pd(c7, ur, e2r); e2i = _mm512_fmadd_pd(c7, ui, e2i);
      e3r = _mm512_fmadd_pd(c2, ur, e3r); e3i = _mm512_fmadd_pd(c2, ui, e3i);
      e4r = _mm512_fmadd_pd(c3, ur, e4r); e4i = _mm512_fmadd_pd(c3, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+6*s), qr = _mm512_load_pd(X+11*s);
      if(dopf){ _mm_prefetch((const char*)(X+6*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+6*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+11*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+11*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+6*s+8), qi = _mm512_load_pd(X+11*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+160, ur);    _mm512_store_pd(AB+168, ui);
      _mm512_store_pd(AB+176, vr); _mm512_store_pd(AB+184, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c6, ur, e1r); e1i = _mm512_fmadd_pd(c6, ui, e1i);
      e2r = _mm512_fmadd_pd(c5, ur, e2r); e2i = _mm512_fmadd_pd(c5, ui, e2i);
      e3r = _mm512_fmadd_pd(c1, ur, e3r); e3i = _mm512_fmadd_pd(c1, ui, e3i);
      e4r = _mm512_fmadd_pd(c7, ur, e4r); e4i = _mm512_fmadd_pd(c7, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+7*s), qr = _mm512_load_pd(X+10*s);
      if(dopf){ _mm_prefetch((const char*)(X+7*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+7*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+10*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+10*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+7*s+8), qi = _mm512_load_pd(X+10*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+192, ur);    _mm512_store_pd(AB+200, ui);
      _mm512_store_pd(AB+208, vr); _mm512_store_pd(AB+216, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c7, ur, e1r); e1i = _mm512_fmadd_pd(c7, ui, e1i);
      e2r = _mm512_fmadd_pd(c3, ur, e2r); e2i = _mm512_fmadd_pd(c3, ui, e2i);
      e3r = _mm512_fmadd_pd(c4, ur, e3r); e3i = _mm512_fmadd_pd(c4, ui, e3i);
      e4r = _mm512_fmadd_pd(c6, ur, e4r); e4i = _mm512_fmadd_pd(c6, ui, e4i);
    }
    { __m512d pr = _mm512_load_pd(X+8*s), qr = _mm512_load_pd(X+9*s);
      if(dopf){ _mm_prefetch((const char*)(X+8*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+8*s+272+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+9*s+272), _MM_HINT_T0); _mm_prefetch((const char*)(X+9*s+272+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+8*s+8), qi = _mm512_load_pd(X+9*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+224, ur);    _mm512_store_pd(AB+232, ui);
      _mm512_store_pd(AB+240, vr); _mm512_store_pd(AB+248, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c8, ur, e1r); e1i = _mm512_fmadd_pd(c8, ui, e1i);
      e2r = _mm512_fmadd_pd(c1, ur, e2r); e2i = _mm512_fmadd_pd(c1, ui, e2i);
      e3r = _mm512_fmadd_pd(c7, ur, e3r); e3i = _mm512_fmadd_pd(c7, ui, e3i);
      e4r = _mm512_fmadd_pd(c2, ur, e4r); e4i = _mm512_fmadd_pd(c2, ui, e4i);
    }
    { __m512d zr = _mm512_add_pd(sr, _mm512_load_pd(C)), zi = _mm512_add_pd(si, _mm512_load_pd(C+8));
      maphw(zr, zi, &zr, &zi);
      _mm512_store_pd(X, zr); _mm512_store_pd(X+8, zi); }
    _mm512_store_pd(Escr+0, e1r); _mm512_store_pd(Escr+8, e1i);
    _mm512_store_pd(Escr+16, e2r); _mm512_store_pd(Escr+24, e2i);
    _mm512_store_pd(Escr+32, e3r); _mm512_store_pd(Escr+40, e3i);
    _mm512_store_pd(Escr+48, e4r); _mm512_store_pd(Escr+56, e4i);
    }
    { __m512d c1 = _mm512_set1_pd(0x1.dd6d000370991p-1), c2 = _mm512_set1_pd(0x1.7a5f6075d4884p-1), c3 = _mm512_set1_pd(0x1.c86fa2b2883cep-2), c4 = _mm512_set1_pd(0x1.79ee63259b75fp-4), c5 = _mm512_set1_pd(-0x1.183b1c61f0d01p-2), c6 = _mm512_set1_pd(-0x1.348c86ed5f1bap-1), c7 = _mm512_set1_pd(-0x1.b34fa910ea3b8p-1), c8 = _mm512_set1_pd(-0x1.f7484007faef3p-1);
    __m512d e5r = x0r, e5i = x0i;
    __m512d e6r = x0r, e6i = x0i;
    __m512d e7r = x0r, e7i = x0i;
    __m512d e8r = x0r, e8i = x0i;
    { __m512d ur = _mm512_load_pd(AB+0);
      __m512d ui = _mm512_load_pd(AB+8);
      e5r = _mm512_fmadd_pd(c5, ur, e5r); e5i = _mm512_fmadd_pd(c5, ui, e5i);
      e6r = _mm512_fmadd_pd(c6, ur, e6r); e6i = _mm512_fmadd_pd(c6, ui, e6i);
      e7r = _mm512_fmadd_pd(c7, ur, e7r); e7i = _mm512_fmadd_pd(c7, ui, e7i);
      e8r = _mm512_fmadd_pd(c8, ur, e8r); e8i = _mm512_fmadd_pd(c8, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+32);
      __m512d ui = _mm512_load_pd(AB+40);
      e5r = _mm512_fmadd_pd(c7, ur, e5r); e5i = _mm512_fmadd_pd(c7, ui, e5i);
      e6r = _mm512_fmadd_pd(c5, ur, e6r); e6i = _mm512_fmadd_pd(c5, ui, e6i);
      e7r = _mm512_fmadd_pd(c3, ur, e7r); e7i = _mm512_fmadd_pd(c3, ui, e7i);
      e8r = _mm512_fmadd_pd(c1, ur, e8r); e8i = _mm512_fmadd_pd(c1, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+64);
      __m512d ui = _mm512_load_pd(AB+72);
      e5r = _mm512_fmadd_pd(c2, ur, e5r); e5i = _mm512_fmadd_pd(c2, ui, e5i);
      e6r = _mm512_fmadd_pd(c1, ur, e6r); e6i = _mm512_fmadd_pd(c1, ui, e6i);
      e7r = _mm512_fmadd_pd(c4, ur, e7r); e7i = _mm512_fmadd_pd(c4, ui, e7i);
      e8r = _mm512_fmadd_pd(c7, ur, e8r); e8i = _mm512_fmadd_pd(c7, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+96);
      __m512d ui = _mm512_load_pd(AB+104);
      e5r = _mm512_fmadd_pd(c3, ur, e5r); e5i = _mm512_fmadd_pd(c3, ui, e5i);
      e6r = _mm512_fmadd_pd(c7, ur, e6r); e6i = _mm512_fmadd_pd(c7, ui, e6i);
      e7r = _mm512_fmadd_pd(c6, ur, e7r); e7i = _mm512_fmadd_pd(c6, ui, e7i);
      e8r = _mm512_fmadd_pd(c2, ur, e8r); e8i = _mm512_fmadd_pd(c2, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+128);
      __m512d ui = _mm512_load_pd(AB+136);
      e5r = _mm512_fmadd_pd(c8, ur, e5r); e5i = _mm512_fmadd_pd(c8, ui, e5i);
      e6r = _mm512_fmadd_pd(c4, ur, e6r); e6i = _mm512_fmadd_pd(c4, ui, e6i);
      e7r = _mm512_fmadd_pd(c1, ur, e7r); e7i = _mm512_fmadd_pd(c1, ui, e7i);
      e8r = _mm512_fmadd_pd(c6, ur, e8r); e8i = _mm512_fmadd_pd(c6, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+160);
      __m512d ui = _mm512_load_pd(AB+168);
      e5r = _mm512_fmadd_pd(c4, ur, e5r); e5i = _mm512_fmadd_pd(c4, ui, e5i);
      e6r = _mm512_fmadd_pd(c2, ur, e6r); e6i = _mm512_fmadd_pd(c2, ui, e6i);
      e7r = _mm512_fmadd_pd(c8, ur, e7r); e7i = _mm512_fmadd_pd(c8, ui, e7i);
      e8r = _mm512_fmadd_pd(c3, ur, e8r); e8i = _mm512_fmadd_pd(c3, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+192);
      __m512d ui = _mm512_load_pd(AB+200);
      e5r = _mm512_fmadd_pd(c1, ur, e5r); e5i = _mm512_fmadd_pd(c1, ui, e5i);
      e6r = _mm512_fmadd_pd(c8, ur, e6r); e6i = _mm512_fmadd_pd(c8, ui, e6i);
      e7r = _mm512_fmadd_pd(c2, ur, e7r); e7i = _mm512_fmadd_pd(c2, ui, e7i);
      e8r = _mm512_fmadd_pd(c5, ur, e8r); e8i = _mm512_fmadd_pd(c5, ui, e8i);
    }
    { __m512d ur = _mm512_load_pd(AB+224);
      __m512d ui = _mm512_load_pd(AB+232);
      e5r = _mm512_fmadd_pd(c6, ur, e5r); e5i = _mm512_fmadd_pd(c6, ui, e5i);
      e6r = _mm512_fmadd_pd(c3, ur, e6r); e6i = _mm512_fmadd_pd(c3, ui, e6i);
      e7r = _mm512_fmadd_pd(c5, ur, e7r); e7i = _mm512_fmadd_pd(c5, ui, e7i);
      e8r = _mm512_fmadd_pd(c4, ur, e8r); e8i = _mm512_fmadd_pd(c4, ui, e8i);
    }
    _mm512_store_pd(Escr+64, e5r); _mm512_store_pd(Escr+72, e5i);
    _mm512_store_pd(Escr+80, e6r); _mm512_store_pd(Escr+88, e6i);
    _mm512_store_pd(Escr+96, e7r); _mm512_store_pd(Escr+104, e7i);
    _mm512_store_pd(Escr+112, e8r); _mm512_store_pd(Escr+120, e8i);
    }
    { __m512d s1 = _mm512_set1_pd(0x1.71e955d8e7cdcp-2), s2 = _mm512_set1_pd(0x1.58eea2a9d6da3p-1), s3 = _mm512_set1_pd(0x1.ca52d7c9e640bp-1), s4 = _mm512_set1_pd(0x1.fdd0deb564b22p-1), s5 = _mm512_set1_pd(0x1.ec746923c349fp-1), s6 = _mm512_set1_pd(0x1.9895b6c9a05f7p-1), s7 = _mm512_set1_pd(0x1.0d8884363dd82p-1), s8 = _mm512_set1_pd(0x1.7851aacd6c6b5p-3);
    __m512d o1r = _mm512_setzero_pd(), o1i = _mm512_setzero_pd();
    __m512d o2r = _mm512_setzero_pd(), o2i = _mm512_setzero_pd();
    __m512d o3r = _mm512_setzero_pd(), o3i = _mm512_setzero_pd();
    __m512d o4r = _mm512_setzero_pd(), o4i = _mm512_setzero_pd();
    { __m512d vr = _mm512_load_pd(AB+16);
      __m512d vi = _mm512_load_pd(AB+24);
      o1r = _mm512_fmadd_pd(s1, vr, o1r); o1i = _mm512_fmadd_pd(s1, vi, o1i);
      o2r = _mm512_fmadd_pd(s2, vr, o2r); o2i = _mm512_fmadd_pd(s2, vi, o2i);
      o3r = _mm512_fmadd_pd(s3, vr, o3r); o3i = _mm512_fmadd_pd(s3, vi, o3i);
      o4r = _mm512_fmadd_pd(s4, vr, o4r); o4i = _mm512_fmadd_pd(s4, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+48);
      __m512d vi = _mm512_load_pd(AB+56);
      o1r = _mm512_fmadd_pd(s2, vr, o1r); o1i = _mm512_fmadd_pd(s2, vi, o1i);
      o2r = _mm512_fmadd_pd(s4, vr, o2r); o2i = _mm512_fmadd_pd(s4, vi, o2i);
      o3r = _mm512_fmadd_pd(s6, vr, o3r); o3i = _mm512_fmadd_pd(s6, vi, o3i);
      o4r = _mm512_fmadd_pd(s8, vr, o4r); o4i = _mm512_fmadd_pd(s8, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+80);
      __m512d vi = _mm512_load_pd(AB+88);
      o1r = _mm512_fmadd_pd(s3, vr, o1r); o1i = _mm512_fmadd_pd(s3, vi, o1i);
      o2r = _mm512_fmadd_pd(s6, vr, o2r); o2i = _mm512_fmadd_pd(s6, vi, o2i);
      o3r = _mm512_fnmadd_pd(s8, vr, o3r); o3i = _mm512_fnmadd_pd(s8, vi, o3i);
      o4r = _mm512_fnmadd_pd(s5, vr, o4r); o4i = _mm512_fnmadd_pd(s5, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+112);
      __m512d vi = _mm512_load_pd(AB+120);
      o1r = _mm512_fmadd_pd(s4, vr, o1r); o1i = _mm512_fmadd_pd(s4, vi, o1i);
      o2r = _mm512_fmadd_pd(s8, vr, o2r); o2i = _mm512_fmadd_pd(s8, vi, o2i);
      o3r = _mm512_fnmadd_pd(s5, vr, o3r); o3i = _mm512_fnmadd_pd(s5, vi, o3i);
      o4r = _mm512_fnmadd_pd(s1, vr, o4r); o4i = _mm512_fnmadd_pd(s1, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+144);
      __m512d vi = _mm512_load_pd(AB+152);
      o1r = _mm512_fmadd_pd(s5, vr, o1r); o1i = _mm512_fmadd_pd(s5, vi, o1i);
      o2r = _mm512_fnmadd_pd(s7, vr, o2r); o2i = _mm512_fnmadd_pd(s7, vi, o2i);
      o3r = _mm512_fnmadd_pd(s2, vr, o3r); o3i = _mm512_fnmadd_pd(s2, vi, o3i);
      o4r = _mm512_fmadd_pd(s3, vr, o4r); o4i = _mm512_fmadd_pd(s3, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+176);
      __m512d vi = _mm512_load_pd(AB+184);
      o1r = _mm512_fmadd_pd(s6, vr, o1r); o1i = _mm512_fmadd_pd(s6, vi, o1i);
      o2r = _mm512_fnmadd_pd(s5, vr, o2r); o2i = _mm512_fnmadd_pd(s5, vi, o2i);
      o3r = _mm512_fmadd_pd(s1, vr, o3r); o3i = _mm512_fmadd_pd(s1, vi, o3i);
      o4r = _mm512_fmadd_pd(s7, vr, o4r); o4i = _mm512_fmadd_pd(s7, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+208);
      __m512d vi = _mm512_load_pd(AB+216);
      o1r = _mm512_fmadd_pd(s7, vr, o1r); o1i = _mm512_fmadd_pd(s7, vi, o1i);
      o2r = _mm512_fnmadd_pd(s3, vr, o2r); o2i = _mm512_fnmadd_pd(s3, vi, o2i);
      o3r = _mm512_fmadd_pd(s4, vr, o3r); o3i = _mm512_fmadd_pd(s4, vi, o3i);
      o4r = _mm512_fnmadd_pd(s6, vr, o4r); o4i = _mm512_fnmadd_pd(s6, vi, o4i);
    }
    { __m512d vr = _mm512_load_pd(AB+240);
      __m512d vi = _mm512_load_pd(AB+248);
      o1r = _mm512_fmadd_pd(s8, vr, o1r); o1i = _mm512_fmadd_pd(s8, vi, o1i);
      o2r = _mm512_fnmadd_pd(s1, vr, o2r); o2i = _mm512_fnmadd_pd(s1, vi, o2i);
      o3r = _mm512_fmadd_pd(s7, vr, o3r); o3i = _mm512_fmadd_pd(s7, vi, o3i);
      o4r = _mm512_fnmadd_pd(s2, vr, o4r); o4i = _mm512_fnmadd_pd(s2, vi, o4i);
    }
    { __m512d er = _mm512_load_pd(Escr+0), ei = _mm512_load_pd(Escr+8);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o1i), _mm512_load_pd(C+1*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o1r), _mm512_load_pd(C+1*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o1i), _mm512_load_pd(C+16*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o1r), _mm512_load_pd(C+16*cs+8));
      map2(zr1, zi1, &zr1, &zi1); maphw(zr2, zi2, &zr2, &zi2);
      _mm512_store_pd(X+1*s, zr1);   _mm512_store_pd(X+1*s+8, zi1);
      _mm512_store_pd(X+16*s, zr2); _mm512_store_pd(X+16*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+16), ei = _mm512_load_pd(Escr+24);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o2i), _mm512_load_pd(C+2*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o2r), _mm512_load_pd(C+2*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o2i), _mm512_load_pd(C+15*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o2r), _mm512_load_pd(C+15*cs+8));
      maphw(zr1, zi1, &zr1, &zi1); map2(zr2, zi2, &zr2, &zi2);
      _mm512_store_pd(X+2*s, zr1);   _mm512_store_pd(X+2*s+8, zi1);
      _mm512_store_pd(X+15*s, zr2); _mm512_store_pd(X+15*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+32), ei = _mm512_load_pd(Escr+40);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o3i), _mm512_load_pd(C+3*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o3r), _mm512_load_pd(C+3*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o3i), _mm512_load_pd(C+14*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o3r), _mm512_load_pd(C+14*cs+8));
      map2(zr1, zi1, &zr1, &zi1); maphw(zr2, zi2, &zr2, &zi2);
      _mm512_store_pd(X+3*s, zr1);   _mm512_store_pd(X+3*s+8, zi1);
      _mm512_store_pd(X+14*s, zr2); _mm512_store_pd(X+14*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+48), ei = _mm512_load_pd(Escr+56);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o4i), _mm512_load_pd(C+4*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o4r), _mm512_load_pd(C+4*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o4i), _mm512_load_pd(C+13*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o4r), _mm512_load_pd(C+13*cs+8));
      maphw(zr1, zi1, &zr1, &zi1); map2(zr2, zi2, &zr2, &zi2);
      _mm512_store_pd(X+4*s, zr1);   _mm512_store_pd(X+4*s+8, zi1);
      _mm512_store_pd(X+13*s, zr2); _mm512_store_pd(X+13*s+8, zi2); }
    }
    { __m512d s1 = _mm512_set1_pd(0x1.71e955d8e7cdcp-2), s2 = _mm512_set1_pd(0x1.58eea2a9d6da3p-1), s3 = _mm512_set1_pd(0x1.ca52d7c9e640bp-1), s4 = _mm512_set1_pd(0x1.fdd0deb564b22p-1), s5 = _mm512_set1_pd(0x1.ec746923c349fp-1), s6 = _mm512_set1_pd(0x1.9895b6c9a05f7p-1), s7 = _mm512_set1_pd(0x1.0d8884363dd82p-1), s8 = _mm512_set1_pd(0x1.7851aacd6c6b5p-3);
    __m512d o5r = _mm512_setzero_pd(), o5i = _mm512_setzero_pd();
    __m512d o6r = _mm512_setzero_pd(), o6i = _mm512_setzero_pd();
    __m512d o7r = _mm512_setzero_pd(), o7i = _mm512_setzero_pd();
    __m512d o8r = _mm512_setzero_pd(), o8i = _mm512_setzero_pd();
    { __m512d vr = _mm512_load_pd(AB+16);
      __m512d vi = _mm512_load_pd(AB+24);
      o5r = _mm512_fmadd_pd(s5, vr, o5r); o5i = _mm512_fmadd_pd(s5, vi, o5i);
      o6r = _mm512_fmadd_pd(s6, vr, o6r); o6i = _mm512_fmadd_pd(s6, vi, o6i);
      o7r = _mm512_fmadd_pd(s7, vr, o7r); o7i = _mm512_fmadd_pd(s7, vi, o7i);
      o8r = _mm512_fmadd_pd(s8, vr, o8r); o8i = _mm512_fmadd_pd(s8, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+48);
      __m512d vi = _mm512_load_pd(AB+56);
      o5r = _mm512_fnmadd_pd(s7, vr, o5r); o5i = _mm512_fnmadd_pd(s7, vi, o5i);
      o6r = _mm512_fnmadd_pd(s5, vr, o6r); o6i = _mm512_fnmadd_pd(s5, vi, o6i);
      o7r = _mm512_fnmadd_pd(s3, vr, o7r); o7i = _mm512_fnmadd_pd(s3, vi, o7i);
      o8r = _mm512_fnmadd_pd(s1, vr, o8r); o8i = _mm512_fnmadd_pd(s1, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+80);
      __m512d vi = _mm512_load_pd(AB+88);
      o5r = _mm512_fnmadd_pd(s2, vr, o5r); o5i = _mm512_fnmadd_pd(s2, vi, o5i);
      o6r = _mm512_fmadd_pd(s1, vr, o6r); o6i = _mm512_fmadd_pd(s1, vi, o6i);
      o7r = _mm512_fmadd_pd(s4, vr, o7r); o7i = _mm512_fmadd_pd(s4, vi, o7i);
      o8r = _mm512_fmadd_pd(s7, vr, o8r); o8i = _mm512_fmadd_pd(s7, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+112);
      __m512d vi = _mm512_load_pd(AB+120);
      o5r = _mm512_fmadd_pd(s3, vr, o5r); o5i = _mm512_fmadd_pd(s3, vi, o5i);
      o6r = _mm512_fmadd_pd(s7, vr, o6r); o6i = _mm512_fmadd_pd(s7, vi, o6i);
      o7r = _mm512_fnmadd_pd(s6, vr, o7r); o7i = _mm512_fnmadd_pd(s6, vi, o7i);
      o8r = _mm512_fnmadd_pd(s2, vr, o8r); o8i = _mm512_fnmadd_pd(s2, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+144);
      __m512d vi = _mm512_load_pd(AB+152);
      o5r = _mm512_fmadd_pd(s8, vr, o5r); o5i = _mm512_fmadd_pd(s8, vi, o5i);
      o6r = _mm512_fnmadd_pd(s4, vr, o6r); o6i = _mm512_fnmadd_pd(s4, vi, o6i);
      o7r = _mm512_fmadd_pd(s1, vr, o7r); o7i = _mm512_fmadd_pd(s1, vi, o7i);
      o8r = _mm512_fmadd_pd(s6, vr, o8r); o8i = _mm512_fmadd_pd(s6, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+176);
      __m512d vi = _mm512_load_pd(AB+184);
      o5r = _mm512_fnmadd_pd(s4, vr, o5r); o5i = _mm512_fnmadd_pd(s4, vi, o5i);
      o6r = _mm512_fmadd_pd(s2, vr, o6r); o6i = _mm512_fmadd_pd(s2, vi, o6i);
      o7r = _mm512_fmadd_pd(s8, vr, o7r); o7i = _mm512_fmadd_pd(s8, vi, o7i);
      o8r = _mm512_fnmadd_pd(s3, vr, o8r); o8i = _mm512_fnmadd_pd(s3, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+208);
      __m512d vi = _mm512_load_pd(AB+216);
      o5r = _mm512_fmadd_pd(s1, vr, o5r); o5i = _mm512_fmadd_pd(s1, vi, o5i);
      o6r = _mm512_fmadd_pd(s8, vr, o6r); o6i = _mm512_fmadd_pd(s8, vi, o6i);
      o7r = _mm512_fnmadd_pd(s2, vr, o7r); o7i = _mm512_fnmadd_pd(s2, vi, o7i);
      o8r = _mm512_fmadd_pd(s5, vr, o8r); o8i = _mm512_fmadd_pd(s5, vi, o8i);
    }
    { __m512d vr = _mm512_load_pd(AB+240);
      __m512d vi = _mm512_load_pd(AB+248);
      o5r = _mm512_fmadd_pd(s6, vr, o5r); o5i = _mm512_fmadd_pd(s6, vi, o5i);
      o6r = _mm512_fnmadd_pd(s3, vr, o6r); o6i = _mm512_fnmadd_pd(s3, vi, o6i);
      o7r = _mm512_fmadd_pd(s5, vr, o7r); o7i = _mm512_fmadd_pd(s5, vi, o7i);
      o8r = _mm512_fnmadd_pd(s4, vr, o8r); o8i = _mm512_fnmadd_pd(s4, vi, o8i);
    }
    { __m512d er = _mm512_load_pd(Escr+64), ei = _mm512_load_pd(Escr+72);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o5i), _mm512_load_pd(C+5*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o5r), _mm512_load_pd(C+5*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o5i), _mm512_load_pd(C+12*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o5r), _mm512_load_pd(C+12*cs+8));
      map2(zr1, zi1, &zr1, &zi1); maphw(zr2, zi2, &zr2, &zi2);
      _mm512_store_pd(X+5*s, zr1);   _mm512_store_pd(X+5*s+8, zi1);
      _mm512_store_pd(X+12*s, zr2); _mm512_store_pd(X+12*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+80), ei = _mm512_load_pd(Escr+88);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o6i), _mm512_load_pd(C+6*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o6r), _mm512_load_pd(C+6*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o6i), _mm512_load_pd(C+11*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o6r), _mm512_load_pd(C+11*cs+8));
      maphw(zr1, zi1, &zr1, &zi1); map2(zr2, zi2, &zr2, &zi2);
      _mm512_store_pd(X+6*s, zr1);   _mm512_store_pd(X+6*s+8, zi1);
      _mm512_store_pd(X+11*s, zr2); _mm512_store_pd(X+11*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+96), ei = _mm512_load_pd(Escr+104);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o7i), _mm512_load_pd(C+7*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o7r), _mm512_load_pd(C+7*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o7i), _mm512_load_pd(C+10*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o7r), _mm512_load_pd(C+10*cs+8));
      map2(zr1, zi1, &zr1, &zi1); maphw(zr2, zi2, &zr2, &zi2);
      _mm512_store_pd(X+7*s, zr1);   _mm512_store_pd(X+7*s+8, zi1);
      _mm512_store_pd(X+10*s, zr2); _mm512_store_pd(X+10*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+112), ei = _mm512_load_pd(Escr+120);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o8i), _mm512_load_pd(C+8*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o8r), _mm512_load_pd(C+8*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o8i), _mm512_load_pd(C+9*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o8r), _mm512_load_pd(C+9*cs+8));
      maphw(zr1, zi1, &zr1, &zi1); map2(zr2, zi2, &zr2, &zi2);
      _mm512_store_pd(X+8*s, zr1);   _mm512_store_pd(X+8*s+8, zi1);
      _mm512_store_pd(X+9*s, zr2); _mm512_store_pd(X+9*s+8, zi2); }
    }
}
#define MAPX_17 1
#define PXF_17 0
#define PFPRIME_17 0

static void __attribute__((noinline))  dft17_one(double* X, long es){ dft17(X, es, 0); }
static void __attribute__((noinline))  dft17_onez(double* X){ dft17(X, 1, PFPRIME_17); }
static void __attribute__((noinline))  dft17_onezm(double* X){ dft17zm(X, 1, PFPRIME_17); }
static void __attribute__((noinline))  dft17_onem(double* X, long es, const double* Ct){ dft17m(X, es, 0, Ct, 1); }
static void dft17_sweep_zy(double* restrict X){
    for(long x=0; x<17; x++){
        double* P = X + x*289*16;
        for(long y=0; y<17; y++) dft17_onez(P + y*17*16);
        for(long z=0; z<17; z++) dft17_one(P + z*16, 17);
    }
}
#if MAPZB_FLAG
static void __attribute__((noinline)) mapblk_17(double* restrict P){
    for(long e=0; e<17; e+=1){
        __m512d zr = _mm512_load_pd(P + e*16);
        __m512d zi = _mm512_load_pd(P + e*16 + 8);
        if(e & 1){ map2(zr, zi, &zr, &zi); } else { maphw(zr, zi, &zr, &zi); }
        _mm512_store_pd(P + e*16, zr);
        _mm512_store_pd(P + e*16 + 8, zi);
    }
}
static void dft17_sweep_zym(double* restrict X){
    for(long x=0; x<17; x++){
        double* P = X + x*289*16;
        for(long y=0; y<17; y++){ mapblk_17(P + y*17*16); dft17_one(P + y*17*16, 1); }
        for(long z=0; z<17; z++) dft17_one(P + z*16, 17);
    }
}
#else
static void dft17_sweep_zym(double* restrict X){
    for(long x=0; x<17; x++){
        double* P = X + x*289*16;
        for(long y=0; y<17; y++) dft17_onezm(P + y*17*16);
        for(long z=0; z<17; z++) dft17_one(P + z*16, 17);
    }
}
#endif
static void dft17_sweep_x_map(double* restrict X, const double* restrict Ct){
    for(long p=0; p<289; p++) dft17_onem(X + p*16, 289, Ct + p*17*16);
}
static void dft17_sweep_x_plain(double* restrict X){
    for(long p=0; p<289; p++) dft17_one(X + p*16, 289);
}
static void dft17_sweep_zy_ms(double* restrict X, const double* restrict C){
    for(long x=0; x<17; x++){
        double* P = X + x*289*16;
        for(long y=0; y<17; y++) dft17_one(P + y*17*16, 1);
        for(long z=0; z<17; z++) dft17_one(P + z*16, 17);
        if(x) mapslab(X + (x-1)*289*16, C + (x-1)*289*16, 289);
    }
    mapslab(X + (17-1)*289*16, C + (17-1)*289*16, 289);
}


/* ingest: 8 volumes AoS complex -> group SoA [e][2][8] */
static void ingest_17(const double* const* src, double* G){
    for(long e=0; e<4912; e+=4){
        __m512d r0=_mm512_loadu_pd(src[0]+2*e), r1=_mm512_loadu_pd(src[1]+2*e);
        __m512d r2=_mm512_loadu_pd(src[2]+2*e), r3=_mm512_loadu_pd(src[3]+2*e);
        __m512d r4=_mm512_loadu_pd(src[4]+2*e), r5=_mm512_loadu_pd(src[5]+2*e);
        __m512d r6=_mm512_loadu_pd(src[6]+2*e), r7=_mm512_loadu_pd(src[7]+2*e);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        _mm512_store_pd(G+e*16,    o0); _mm512_store_pd(G+e*16+8,  o1);
        _mm512_store_pd(G+e*16+16, o2); _mm512_store_pd(G+e*16+24, o3);
        _mm512_store_pd(G+e*16+32, o4); _mm512_store_pd(G+e*16+40, o5);
        _mm512_store_pd(G+e*16+48, o6); _mm512_store_pd(G+e*16+56, o7);
    }
    { /* tail of 1 elements */
        const long e = 4912;
        const __mmask8 mk = (__mmask8)((1u<<(2*1))-1u);
        __m512d r0=_mm512_maskz_loadu_pd(mk, src[0]+2*e), r1=_mm512_maskz_loadu_pd(mk, src[1]+2*e);
        __m512d r2=_mm512_maskz_loadu_pd(mk, src[2]+2*e), r3=_mm512_maskz_loadu_pd(mk, src[3]+2*e);
        __m512d r4=_mm512_maskz_loadu_pd(mk, src[4]+2*e), r5=_mm512_maskz_loadu_pd(mk, src[5]+2*e);
        __m512d r6=_mm512_maskz_loadu_pd(mk, src[6]+2*e), r7=_mm512_maskz_loadu_pd(mk, src[7]+2*e);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        __m512d A[8]; A[0]=o0;A[1]=o1;A[2]=o2;A[3]=o3;A[4]=o4;A[5]=o5;A[6]=o6;A[7]=o7;
        for(int q=0;q<2*1;q++) _mm512_store_pd(G+e*16+q*8, A[q]);
    }
}
/* output with map: group SoA (unmapped z) -> nv AoS volumes */
static void output_17(const double* G, double* const* dst, int nv){
    for(long e=0; e<4912; e+=4){
        __m512d i0=_mm512_load_pd(G+e*16),    i1=_mm512_load_pd(G+e*16+8);
        __m512d i2=_mm512_load_pd(G+e*16+16), i3=_mm512_load_pd(G+e*16+24);
        __m512d i4=_mm512_load_pd(G+e*16+32), i5=_mm512_load_pd(G+e*16+40);
        __m512d i6=_mm512_load_pd(G+e*16+48), i7=_mm512_load_pd(G+e*16+56);

        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(i0,i1,i2,i3,i4,i5,i6,i7,o0,o1,o2,o3,o4,o5,o6,o7);
        __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
        for(int v=0; v<nv; v++) _mm512_storeu_pd(dst[v]+2*e, *O[v]);
    }
    { /* tail */
        const long e = 4912;
        const __mmask8 mk = (__mmask8)((1u<<(2*1))-1u);
        __m512d A[8];
        for(int q=0;q<2*1;q++) A[q] = _mm512_load_pd(G+e*16+q*8);

        for(int q=2*1;q<8;q++) A[q] = _mm512_setzero_pd();
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],o0,o1,o2,o3,o4,o5,o6,o7);
        __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
        for(int v=0; v<nv; v++) _mm512_mask_storeu_pd(dst[v]+2*e, mk, *O[v]);
    }
}


static double* Xg_17 = 0;
static double* Cg_17 = 0;
void hot2_17(long which){
    if(!Xg_17){ Xg_17 = alloc_huge_st((4913+64*17)*16*8); Cg_17 = alloc_huge_st(4913*16*8); }
    double* P = Xg_17;
    if(which==99){ for(long i=0;i<289*16;i++) P[i] = 0.5 + 1e-6*(i%97); return; }
    if(which==0 || which==2) for(long y=0; y<17; y++) dft17_one(P + y*17*16, 1);
    if(which==1 || which==2) for(long z=0; z<17; z++) dft17_one(P + z*16, 17);
}
void bsweep_17(long which, long n){
    if(!Xg_17){ Xg_17 = alloc_huge_st(4913*16*8); Cg_17 = alloc_huge_st(4913*16*8); }
    for(long i=0;i<4913*16;i++){ Xg_17[i] = 0.5 + 1e-6*(i%97); Cg_17[i] = 0.01; }
    for(long r=0;r<n;r++){
        if(which==0) dft17_sweep_zy(Xg_17);
        else if(which==1) dft17_sweep_zym(Xg_17);
        else if(which==2) dft17_sweep_x_map(Xg_17, Cg_17);
#if USEASM_FLAG
#endif
        if((r&7)==7) for(long i=0;i<4913*16;i+=997) Xg_17[i] = 0.5;
    }
}
void run_17(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    const long NE = 4913;
    if(!Xg_17){ Xg_17 = alloc_huge_st(NE*16*8); Cg_17 = alloc_huge_st(NE*16*8); }
    double* X = Xg_17; double* Ct = Cg_17;
    for(long g0=0; g0<B; g0+=8){
        int nv = (int)((B - g0) < 8 ? (B - g0) : 8);
        const double* src[8]; const double* csrc[8];
        double* d1[8]; double* dm[8];
        for(int v=0; v<8; v++){
            int vv = v < nv ? v : 0;
            src[v] = x0 + (g0+vv)*2*NE; csrc[v] = c + (g0+vv)*2*NE;
            if(v<nv){ d1[v] = out1 + (g0+v)*2*NE; dm[v] = outm + (g0+v)*2*NE; }
        }
        /* ingest c (consumption order for MAPX/deferred paths; plain for PXF) */
#if PXF_17
        ingest_17(csrc, Ct);
#else
        ingest_17(csrc, X);
        for(long p=0; p<289; p++)
            for(long k=0; k<17; k++){
                _mm512_store_pd(Ct + (p*17+k)*16,     _mm512_load_pd(X + (k*289+p)*16));
                _mm512_store_pd(Ct + (p*17+k)*16 + 8, _mm512_load_pd(X + (k*289+p)*16 + 8));
            }
#endif
        ingest_17(src, X);
        for(long t=0; t<m; t++){
#if PXF_17
            dft17_sweep_x_plain(X);
            dft17_sweep_zy_ms(X, Ct);
#elif USEASM_FLAG
            dft17_sweep_zy_asm(X, t>0);
            dft17_sweep_x_asm(X, Ct);
#else
#if MAPX_17
            dft17_sweep_zy(X);
            dft17_sweep_x_map(X, Ct);
#else
            if(t==0) dft17_sweep_zy(X); else dft17_sweep_zym(X);
            dft17_sweep_x_map(X, Ct);
#endif
#endif
            if(t==0 && m>1) output_17(X, d1, nv);
        }
        output_17(X, dm, nv);
        if(m==1) output_17(X, d1, nv);
    }
}

static __attribute__((always_inline)) inline void dft23(double* restrict X, long es, int dopf){
    const long s = es*16;
    double AB[88*8] ALIGN64;
    double Escr[22*8] ALIGN64;
    __m512d x0r = _mm512_load_pd(X);
    if(dopf){ _mm_prefetch((const char*)(X+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+376), _MM_HINT_T0); }
    __m512d x0i = _mm512_load_pd(X+8);
    { __m512d c1 = _mm512_set1_pd(0x1.ed037ea3d2dbcp-1), c2 = _mm512_set1_pd(0x1.b57675cf309eep-1), c3 = _mm512_set1_pd(0x1.5d779b07cfef7p-1), c4 = _mm512_set1_pd(0x1.d71b4a0c5a6c9p-2), c5 = _mm512_set1_pd(0x1.a0ad8bd1e2881p-3), c6 = _mm512_set1_pd(-0x1.17855b599f3b2p-4), c7 = _mm512_set1_pd(-0x1.56eaae597c776p-2), c8 = _mm512_set1_pd(-0x1.2742a4a775cfap-1), c9 = _mm512_set1_pd(-0x1.8d2a07c16d46ep-1), c10 = _mm512_set1_pd(-0x1.d59cb83ef99bcp-1), c11 = _mm512_set1_pd(-0x1.fb3b3035aa6ccp-1);
    __m512d e1r = x0r, e1i = x0i;
    __m512d e2r = x0r, e2i = x0i;
    __m512d e3r = x0r, e3i = x0i;
    __m512d e4r = x0r, e4i = x0i;
    __m512d e5r = x0r, e5i = x0i;
    __m512d e6r = x0r, e6i = x0i;
    __m512d sr = x0r, si = x0i;
    { __m512d pr = _mm512_load_pd(X+1*s), qr = _mm512_load_pd(X+22*s);
      if(dopf){ _mm_prefetch((const char*)(X+1*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+1*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+22*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+22*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+1*s+8), qi = _mm512_load_pd(X+22*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+0, ur);    _mm512_store_pd(AB+8, ui);
      _mm512_store_pd(AB+16, vr); _mm512_store_pd(AB+24, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c1, ur, e1r); e1i = _mm512_fmadd_pd(c1, ui, e1i);
      e2r = _mm512_fmadd_pd(c2, ur, e2r); e2i = _mm512_fmadd_pd(c2, ui, e2i);
      e3r = _mm512_fmadd_pd(c3, ur, e3r); e3i = _mm512_fmadd_pd(c3, ui, e3i);
      e4r = _mm512_fmadd_pd(c4, ur, e4r); e4i = _mm512_fmadd_pd(c4, ui, e4i);
      e5r = _mm512_fmadd_pd(c5, ur, e5r); e5i = _mm512_fmadd_pd(c5, ui, e5i);
      e6r = _mm512_fmadd_pd(c6, ur, e6r); e6i = _mm512_fmadd_pd(c6, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+2*s), qr = _mm512_load_pd(X+21*s);
      if(dopf){ _mm_prefetch((const char*)(X+2*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+2*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+21*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+21*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+2*s+8), qi = _mm512_load_pd(X+21*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+32, ur);    _mm512_store_pd(AB+40, ui);
      _mm512_store_pd(AB+48, vr); _mm512_store_pd(AB+56, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c2, ur, e1r); e1i = _mm512_fmadd_pd(c2, ui, e1i);
      e2r = _mm512_fmadd_pd(c4, ur, e2r); e2i = _mm512_fmadd_pd(c4, ui, e2i);
      e3r = _mm512_fmadd_pd(c6, ur, e3r); e3i = _mm512_fmadd_pd(c6, ui, e3i);
      e4r = _mm512_fmadd_pd(c8, ur, e4r); e4i = _mm512_fmadd_pd(c8, ui, e4i);
      e5r = _mm512_fmadd_pd(c10, ur, e5r); e5i = _mm512_fmadd_pd(c10, ui, e5i);
      e6r = _mm512_fmadd_pd(c11, ur, e6r); e6i = _mm512_fmadd_pd(c11, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+3*s), qr = _mm512_load_pd(X+20*s);
      if(dopf){ _mm_prefetch((const char*)(X+3*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+3*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+20*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+20*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+3*s+8), qi = _mm512_load_pd(X+20*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+64, ur);    _mm512_store_pd(AB+72, ui);
      _mm512_store_pd(AB+80, vr); _mm512_store_pd(AB+88, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c3, ur, e1r); e1i = _mm512_fmadd_pd(c3, ui, e1i);
      e2r = _mm512_fmadd_pd(c6, ur, e2r); e2i = _mm512_fmadd_pd(c6, ui, e2i);
      e3r = _mm512_fmadd_pd(c9, ur, e3r); e3i = _mm512_fmadd_pd(c9, ui, e3i);
      e4r = _mm512_fmadd_pd(c11, ur, e4r); e4i = _mm512_fmadd_pd(c11, ui, e4i);
      e5r = _mm512_fmadd_pd(c8, ur, e5r); e5i = _mm512_fmadd_pd(c8, ui, e5i);
      e6r = _mm512_fmadd_pd(c5, ur, e6r); e6i = _mm512_fmadd_pd(c5, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+4*s), qr = _mm512_load_pd(X+19*s);
      if(dopf){ _mm_prefetch((const char*)(X+4*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+4*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+19*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+19*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+4*s+8), qi = _mm512_load_pd(X+19*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+96, ur);    _mm512_store_pd(AB+104, ui);
      _mm512_store_pd(AB+112, vr); _mm512_store_pd(AB+120, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c4, ur, e1r); e1i = _mm512_fmadd_pd(c4, ui, e1i);
      e2r = _mm512_fmadd_pd(c8, ur, e2r); e2i = _mm512_fmadd_pd(c8, ui, e2i);
      e3r = _mm512_fmadd_pd(c11, ur, e3r); e3i = _mm512_fmadd_pd(c11, ui, e3i);
      e4r = _mm512_fmadd_pd(c7, ur, e4r); e4i = _mm512_fmadd_pd(c7, ui, e4i);
      e5r = _mm512_fmadd_pd(c3, ur, e5r); e5i = _mm512_fmadd_pd(c3, ui, e5i);
      e6r = _mm512_fmadd_pd(c1, ur, e6r); e6i = _mm512_fmadd_pd(c1, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+5*s), qr = _mm512_load_pd(X+18*s);
      if(dopf){ _mm_prefetch((const char*)(X+5*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+5*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+18*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+18*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+5*s+8), qi = _mm512_load_pd(X+18*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+128, ur);    _mm512_store_pd(AB+136, ui);
      _mm512_store_pd(AB+144, vr); _mm512_store_pd(AB+152, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c5, ur, e1r); e1i = _mm512_fmadd_pd(c5, ui, e1i);
      e2r = _mm512_fmadd_pd(c10, ur, e2r); e2i = _mm512_fmadd_pd(c10, ui, e2i);
      e3r = _mm512_fmadd_pd(c8, ur, e3r); e3i = _mm512_fmadd_pd(c8, ui, e3i);
      e4r = _mm512_fmadd_pd(c3, ur, e4r); e4i = _mm512_fmadd_pd(c3, ui, e4i);
      e5r = _mm512_fmadd_pd(c2, ur, e5r); e5i = _mm512_fmadd_pd(c2, ui, e5i);
      e6r = _mm512_fmadd_pd(c7, ur, e6r); e6i = _mm512_fmadd_pd(c7, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+6*s), qr = _mm512_load_pd(X+17*s);
      if(dopf){ _mm_prefetch((const char*)(X+6*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+6*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+17*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+17*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+6*s+8), qi = _mm512_load_pd(X+17*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+160, ur);    _mm512_store_pd(AB+168, ui);
      _mm512_store_pd(AB+176, vr); _mm512_store_pd(AB+184, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c6, ur, e1r); e1i = _mm512_fmadd_pd(c6, ui, e1i);
      e2r = _mm512_fmadd_pd(c11, ur, e2r); e2i = _mm512_fmadd_pd(c11, ui, e2i);
      e3r = _mm512_fmadd_pd(c5, ur, e3r); e3i = _mm512_fmadd_pd(c5, ui, e3i);
      e4r = _mm512_fmadd_pd(c1, ur, e4r); e4i = _mm512_fmadd_pd(c1, ui, e4i);
      e5r = _mm512_fmadd_pd(c7, ur, e5r); e5i = _mm512_fmadd_pd(c7, ui, e5i);
      e6r = _mm512_fmadd_pd(c10, ur, e6r); e6i = _mm512_fmadd_pd(c10, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+7*s), qr = _mm512_load_pd(X+16*s);
      if(dopf){ _mm_prefetch((const char*)(X+7*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+7*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+16*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+16*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+7*s+8), qi = _mm512_load_pd(X+16*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+192, ur);    _mm512_store_pd(AB+200, ui);
      _mm512_store_pd(AB+208, vr); _mm512_store_pd(AB+216, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c7, ur, e1r); e1i = _mm512_fmadd_pd(c7, ui, e1i);
      e2r = _mm512_fmadd_pd(c9, ur, e2r); e2i = _mm512_fmadd_pd(c9, ui, e2i);
      e3r = _mm512_fmadd_pd(c2, ur, e3r); e3i = _mm512_fmadd_pd(c2, ui, e3i);
      e4r = _mm512_fmadd_pd(c5, ur, e4r); e4i = _mm512_fmadd_pd(c5, ui, e4i);
      e5r = _mm512_fmadd_pd(c11, ur, e5r); e5i = _mm512_fmadd_pd(c11, ui, e5i);
      e6r = _mm512_fmadd_pd(c4, ur, e6r); e6i = _mm512_fmadd_pd(c4, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+8*s), qr = _mm512_load_pd(X+15*s);
      if(dopf){ _mm_prefetch((const char*)(X+8*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+8*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+15*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+15*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+8*s+8), qi = _mm512_load_pd(X+15*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+224, ur);    _mm512_store_pd(AB+232, ui);
      _mm512_store_pd(AB+240, vr); _mm512_store_pd(AB+248, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c8, ur, e1r); e1i = _mm512_fmadd_pd(c8, ui, e1i);
      e2r = _mm512_fmadd_pd(c7, ur, e2r); e2i = _mm512_fmadd_pd(c7, ui, e2i);
      e3r = _mm512_fmadd_pd(c1, ur, e3r); e3i = _mm512_fmadd_pd(c1, ui, e3i);
      e4r = _mm512_fmadd_pd(c9, ur, e4r); e4i = _mm512_fmadd_pd(c9, ui, e4i);
      e5r = _mm512_fmadd_pd(c6, ur, e5r); e5i = _mm512_fmadd_pd(c6, ui, e5i);
      e6r = _mm512_fmadd_pd(c2, ur, e6r); e6i = _mm512_fmadd_pd(c2, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+9*s), qr = _mm512_load_pd(X+14*s);
      if(dopf){ _mm_prefetch((const char*)(X+9*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+9*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+14*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+14*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+9*s+8), qi = _mm512_load_pd(X+14*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+256, ur);    _mm512_store_pd(AB+264, ui);
      _mm512_store_pd(AB+272, vr); _mm512_store_pd(AB+280, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c9, ur, e1r); e1i = _mm512_fmadd_pd(c9, ui, e1i);
      e2r = _mm512_fmadd_pd(c5, ur, e2r); e2i = _mm512_fmadd_pd(c5, ui, e2i);
      e3r = _mm512_fmadd_pd(c4, ur, e3r); e3i = _mm512_fmadd_pd(c4, ui, e3i);
      e4r = _mm512_fmadd_pd(c10, ur, e4r); e4i = _mm512_fmadd_pd(c10, ui, e4i);
      e5r = _mm512_fmadd_pd(c1, ur, e5r); e5i = _mm512_fmadd_pd(c1, ui, e5i);
      e6r = _mm512_fmadd_pd(c8, ur, e6r); e6i = _mm512_fmadd_pd(c8, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+10*s), qr = _mm512_load_pd(X+13*s);
      if(dopf){ _mm_prefetch((const char*)(X+10*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+10*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+13*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+13*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+10*s+8), qi = _mm512_load_pd(X+13*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+288, ur);    _mm512_store_pd(AB+296, ui);
      _mm512_store_pd(AB+304, vr); _mm512_store_pd(AB+312, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c10, ur, e1r); e1i = _mm512_fmadd_pd(c10, ui, e1i);
      e2r = _mm512_fmadd_pd(c3, ur, e2r); e2i = _mm512_fmadd_pd(c3, ui, e2i);
      e3r = _mm512_fmadd_pd(c7, ur, e3r); e3i = _mm512_fmadd_pd(c7, ui, e3i);
      e4r = _mm512_fmadd_pd(c6, ur, e4r); e4i = _mm512_fmadd_pd(c6, ui, e4i);
      e5r = _mm512_fmadd_pd(c4, ur, e5r); e5i = _mm512_fmadd_pd(c4, ui, e5i);
      e6r = _mm512_fmadd_pd(c9, ur, e6r); e6i = _mm512_fmadd_pd(c9, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+11*s), qr = _mm512_load_pd(X+12*s);
      if(dopf){ _mm_prefetch((const char*)(X+11*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+11*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+12*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+12*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+11*s+8), qi = _mm512_load_pd(X+12*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+320, ur);    _mm512_store_pd(AB+328, ui);
      _mm512_store_pd(AB+336, vr); _mm512_store_pd(AB+344, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c11, ur, e1r); e1i = _mm512_fmadd_pd(c11, ui, e1i);
      e2r = _mm512_fmadd_pd(c1, ur, e2r); e2i = _mm512_fmadd_pd(c1, ui, e2i);
      e3r = _mm512_fmadd_pd(c10, ur, e3r); e3i = _mm512_fmadd_pd(c10, ui, e3i);
      e4r = _mm512_fmadd_pd(c2, ur, e4r); e4i = _mm512_fmadd_pd(c2, ui, e4i);
      e5r = _mm512_fmadd_pd(c9, ur, e5r); e5i = _mm512_fmadd_pd(c9, ui, e5i);
      e6r = _mm512_fmadd_pd(c3, ur, e6r); e6i = _mm512_fmadd_pd(c3, ui, e6i);
    }
    _mm512_store_pd(X, sr); _mm512_store_pd(X+8, si);
    _mm512_store_pd(Escr+0, e1r); _mm512_store_pd(Escr+8, e1i);
    _mm512_store_pd(Escr+16, e2r); _mm512_store_pd(Escr+24, e2i);
    _mm512_store_pd(Escr+32, e3r); _mm512_store_pd(Escr+40, e3i);
    _mm512_store_pd(Escr+48, e4r); _mm512_store_pd(Escr+56, e4i);
    _mm512_store_pd(Escr+64, e5r); _mm512_store_pd(Escr+72, e5i);
    _mm512_store_pd(Escr+80, e6r); _mm512_store_pd(Escr+88, e6i);
    }
    { __m512d c1 = _mm512_set1_pd(0x1.ed037ea3d2dbcp-1), c2 = _mm512_set1_pd(0x1.b57675cf309eep-1), c3 = _mm512_set1_pd(0x1.5d779b07cfef7p-1), c4 = _mm512_set1_pd(0x1.d71b4a0c5a6c9p-2), c5 = _mm512_set1_pd(0x1.a0ad8bd1e2881p-3), c6 = _mm512_set1_pd(-0x1.17855b599f3b2p-4), c7 = _mm512_set1_pd(-0x1.56eaae597c776p-2), c8 = _mm512_set1_pd(-0x1.2742a4a775cfap-1), c9 = _mm512_set1_pd(-0x1.8d2a07c16d46ep-1), c10 = _mm512_set1_pd(-0x1.d59cb83ef99bcp-1), c11 = _mm512_set1_pd(-0x1.fb3b3035aa6ccp-1);
    __m512d e7r = x0r, e7i = x0i;
    __m512d e8r = x0r, e8i = x0i;
    __m512d e9r = x0r, e9i = x0i;
    __m512d e10r = x0r, e10i = x0i;
    __m512d e11r = x0r, e11i = x0i;
    { __m512d ur = _mm512_load_pd(AB+0);
      __m512d ui = _mm512_load_pd(AB+8);
      e7r = _mm512_fmadd_pd(c7, ur, e7r); e7i = _mm512_fmadd_pd(c7, ui, e7i);
      e8r = _mm512_fmadd_pd(c8, ur, e8r); e8i = _mm512_fmadd_pd(c8, ui, e8i);
      e9r = _mm512_fmadd_pd(c9, ur, e9r); e9i = _mm512_fmadd_pd(c9, ui, e9i);
      e10r = _mm512_fmadd_pd(c10, ur, e10r); e10i = _mm512_fmadd_pd(c10, ui, e10i);
      e11r = _mm512_fmadd_pd(c11, ur, e11r); e11i = _mm512_fmadd_pd(c11, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+32);
      __m512d ui = _mm512_load_pd(AB+40);
      e7r = _mm512_fmadd_pd(c9, ur, e7r); e7i = _mm512_fmadd_pd(c9, ui, e7i);
      e8r = _mm512_fmadd_pd(c7, ur, e8r); e8i = _mm512_fmadd_pd(c7, ui, e8i);
      e9r = _mm512_fmadd_pd(c5, ur, e9r); e9i = _mm512_fmadd_pd(c5, ui, e9i);
      e10r = _mm512_fmadd_pd(c3, ur, e10r); e10i = _mm512_fmadd_pd(c3, ui, e10i);
      e11r = _mm512_fmadd_pd(c1, ur, e11r); e11i = _mm512_fmadd_pd(c1, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+64);
      __m512d ui = _mm512_load_pd(AB+72);
      e7r = _mm512_fmadd_pd(c2, ur, e7r); e7i = _mm512_fmadd_pd(c2, ui, e7i);
      e8r = _mm512_fmadd_pd(c1, ur, e8r); e8i = _mm512_fmadd_pd(c1, ui, e8i);
      e9r = _mm512_fmadd_pd(c4, ur, e9r); e9i = _mm512_fmadd_pd(c4, ui, e9i);
      e10r = _mm512_fmadd_pd(c7, ur, e10r); e10i = _mm512_fmadd_pd(c7, ui, e10i);
      e11r = _mm512_fmadd_pd(c10, ur, e11r); e11i = _mm512_fmadd_pd(c10, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+96);
      __m512d ui = _mm512_load_pd(AB+104);
      e7r = _mm512_fmadd_pd(c5, ur, e7r); e7i = _mm512_fmadd_pd(c5, ui, e7i);
      e8r = _mm512_fmadd_pd(c9, ur, e8r); e8i = _mm512_fmadd_pd(c9, ui, e8i);
      e9r = _mm512_fmadd_pd(c10, ur, e9r); e9i = _mm512_fmadd_pd(c10, ui, e9i);
      e10r = _mm512_fmadd_pd(c6, ur, e10r); e10i = _mm512_fmadd_pd(c6, ui, e10i);
      e11r = _mm512_fmadd_pd(c2, ur, e11r); e11i = _mm512_fmadd_pd(c2, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+128);
      __m512d ui = _mm512_load_pd(AB+136);
      e7r = _mm512_fmadd_pd(c11, ur, e7r); e7i = _mm512_fmadd_pd(c11, ui, e7i);
      e8r = _mm512_fmadd_pd(c6, ur, e8r); e8i = _mm512_fmadd_pd(c6, ui, e8i);
      e9r = _mm512_fmadd_pd(c1, ur, e9r); e9i = _mm512_fmadd_pd(c1, ui, e9i);
      e10r = _mm512_fmadd_pd(c4, ur, e10r); e10i = _mm512_fmadd_pd(c4, ui, e10i);
      e11r = _mm512_fmadd_pd(c9, ur, e11r); e11i = _mm512_fmadd_pd(c9, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+160);
      __m512d ui = _mm512_load_pd(AB+168);
      e7r = _mm512_fmadd_pd(c4, ur, e7r); e7i = _mm512_fmadd_pd(c4, ui, e7i);
      e8r = _mm512_fmadd_pd(c2, ur, e8r); e8i = _mm512_fmadd_pd(c2, ui, e8i);
      e9r = _mm512_fmadd_pd(c8, ur, e9r); e9i = _mm512_fmadd_pd(c8, ui, e9i);
      e10r = _mm512_fmadd_pd(c9, ur, e10r); e10i = _mm512_fmadd_pd(c9, ui, e10i);
      e11r = _mm512_fmadd_pd(c3, ur, e11r); e11i = _mm512_fmadd_pd(c3, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+192);
      __m512d ui = _mm512_load_pd(AB+200);
      e7r = _mm512_fmadd_pd(c3, ur, e7r); e7i = _mm512_fmadd_pd(c3, ui, e7i);
      e8r = _mm512_fmadd_pd(c10, ur, e8r); e8i = _mm512_fmadd_pd(c10, ui, e8i);
      e9r = _mm512_fmadd_pd(c6, ur, e9r); e9i = _mm512_fmadd_pd(c6, ui, e9i);
      e10r = _mm512_fmadd_pd(c1, ur, e10r); e10i = _mm512_fmadd_pd(c1, ui, e10i);
      e11r = _mm512_fmadd_pd(c8, ur, e11r); e11i = _mm512_fmadd_pd(c8, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+224);
      __m512d ui = _mm512_load_pd(AB+232);
      e7r = _mm512_fmadd_pd(c10, ur, e7r); e7i = _mm512_fmadd_pd(c10, ui, e7i);
      e8r = _mm512_fmadd_pd(c5, ur, e8r); e8i = _mm512_fmadd_pd(c5, ui, e8i);
      e9r = _mm512_fmadd_pd(c3, ur, e9r); e9i = _mm512_fmadd_pd(c3, ui, e9i);
      e10r = _mm512_fmadd_pd(c11, ur, e10r); e10i = _mm512_fmadd_pd(c11, ui, e10i);
      e11r = _mm512_fmadd_pd(c4, ur, e11r); e11i = _mm512_fmadd_pd(c4, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+256);
      __m512d ui = _mm512_load_pd(AB+264);
      e7r = _mm512_fmadd_pd(c6, ur, e7r); e7i = _mm512_fmadd_pd(c6, ui, e7i);
      e8r = _mm512_fmadd_pd(c3, ur, e8r); e8i = _mm512_fmadd_pd(c3, ui, e8i);
      e9r = _mm512_fmadd_pd(c11, ur, e9r); e9i = _mm512_fmadd_pd(c11, ui, e9i);
      e10r = _mm512_fmadd_pd(c2, ur, e10r); e10i = _mm512_fmadd_pd(c2, ui, e10i);
      e11r = _mm512_fmadd_pd(c7, ur, e11r); e11i = _mm512_fmadd_pd(c7, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+288);
      __m512d ui = _mm512_load_pd(AB+296);
      e7r = _mm512_fmadd_pd(c1, ur, e7r); e7i = _mm512_fmadd_pd(c1, ui, e7i);
      e8r = _mm512_fmadd_pd(c11, ur, e8r); e8i = _mm512_fmadd_pd(c11, ui, e8i);
      e9r = _mm512_fmadd_pd(c2, ur, e9r); e9i = _mm512_fmadd_pd(c2, ui, e9i);
      e10r = _mm512_fmadd_pd(c8, ur, e10r); e10i = _mm512_fmadd_pd(c8, ui, e10i);
      e11r = _mm512_fmadd_pd(c5, ur, e11r); e11i = _mm512_fmadd_pd(c5, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+320);
      __m512d ui = _mm512_load_pd(AB+328);
      e7r = _mm512_fmadd_pd(c8, ur, e7r); e7i = _mm512_fmadd_pd(c8, ui, e7i);
      e8r = _mm512_fmadd_pd(c4, ur, e8r); e8i = _mm512_fmadd_pd(c4, ui, e8i);
      e9r = _mm512_fmadd_pd(c7, ur, e9r); e9i = _mm512_fmadd_pd(c7, ui, e9i);
      e10r = _mm512_fmadd_pd(c5, ur, e10r); e10i = _mm512_fmadd_pd(c5, ui, e10i);
      e11r = _mm512_fmadd_pd(c6, ur, e11r); e11i = _mm512_fmadd_pd(c6, ui, e11i);
    }
    _mm512_store_pd(Escr+96, e7r); _mm512_store_pd(Escr+104, e7i);
    _mm512_store_pd(Escr+112, e8r); _mm512_store_pd(Escr+120, e8i);
    _mm512_store_pd(Escr+128, e9r); _mm512_store_pd(Escr+136, e9i);
    _mm512_store_pd(Escr+144, e10r); _mm512_store_pd(Escr+152, e10i);
    _mm512_store_pd(Escr+160, e11r); _mm512_store_pd(Escr+168, e11i);
    }
    { __m512d s1 = _mm512_set1_pd(0x1.14459ad2be466p-2), s2 = _mm512_set1_pd(0x1.0a06e851db7cap-1), s3 = _mm512_set1_pd(0x1.763021aaa15d9p-1), s4 = _mm512_set1_pd(0x1.c698e42f47b09p-1), s5 = _mm512_set1_pd(0x1.f54a827142577p-1), s6 = _mm512_set1_pd(0x1.fece70dfd3efbp-1), s7 = _mm512_set1_pd(0x1.e270060999288p-1), s8 = _mm512_set1_pd(0x1.a249e0b897caap-1), s9 = _mm512_set1_pd(0x1.431df5838f7f1p-1), s10 = _mm512_set1_pd(0x1.97f6748e524b1p-2), s11 = _mm512_set1_pd(0x1.16de8a4564f1cp-3);
    __m512d o1r = _mm512_setzero_pd(), o1i = _mm512_setzero_pd();
    __m512d o2r = _mm512_setzero_pd(), o2i = _mm512_setzero_pd();
    __m512d o3r = _mm512_setzero_pd(), o3i = _mm512_setzero_pd();
    __m512d o4r = _mm512_setzero_pd(), o4i = _mm512_setzero_pd();
    __m512d o5r = _mm512_setzero_pd(), o5i = _mm512_setzero_pd();
    __m512d o6r = _mm512_setzero_pd(), o6i = _mm512_setzero_pd();
    { __m512d vr = _mm512_load_pd(AB+16);
      __m512d vi = _mm512_load_pd(AB+24);
      o1r = _mm512_fmadd_pd(s1, vr, o1r); o1i = _mm512_fmadd_pd(s1, vi, o1i);
      o2r = _mm512_fmadd_pd(s2, vr, o2r); o2i = _mm512_fmadd_pd(s2, vi, o2i);
      o3r = _mm512_fmadd_pd(s3, vr, o3r); o3i = _mm512_fmadd_pd(s3, vi, o3i);
      o4r = _mm512_fmadd_pd(s4, vr, o4r); o4i = _mm512_fmadd_pd(s4, vi, o4i);
      o5r = _mm512_fmadd_pd(s5, vr, o5r); o5i = _mm512_fmadd_pd(s5, vi, o5i);
      o6r = _mm512_fmadd_pd(s6, vr, o6r); o6i = _mm512_fmadd_pd(s6, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+48);
      __m512d vi = _mm512_load_pd(AB+56);
      o1r = _mm512_fmadd_pd(s2, vr, o1r); o1i = _mm512_fmadd_pd(s2, vi, o1i);
      o2r = _mm512_fmadd_pd(s4, vr, o2r); o2i = _mm512_fmadd_pd(s4, vi, o2i);
      o3r = _mm512_fmadd_pd(s6, vr, o3r); o3i = _mm512_fmadd_pd(s6, vi, o3i);
      o4r = _mm512_fmadd_pd(s8, vr, o4r); o4i = _mm512_fmadd_pd(s8, vi, o4i);
      o5r = _mm512_fmadd_pd(s10, vr, o5r); o5i = _mm512_fmadd_pd(s10, vi, o5i);
      o6r = _mm512_fnmadd_pd(s11, vr, o6r); o6i = _mm512_fnmadd_pd(s11, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+80);
      __m512d vi = _mm512_load_pd(AB+88);
      o1r = _mm512_fmadd_pd(s3, vr, o1r); o1i = _mm512_fmadd_pd(s3, vi, o1i);
      o2r = _mm512_fmadd_pd(s6, vr, o2r); o2i = _mm512_fmadd_pd(s6, vi, o2i);
      o3r = _mm512_fmadd_pd(s9, vr, o3r); o3i = _mm512_fmadd_pd(s9, vi, o3i);
      o4r = _mm512_fnmadd_pd(s11, vr, o4r); o4i = _mm512_fnmadd_pd(s11, vi, o4i);
      o5r = _mm512_fnmadd_pd(s8, vr, o5r); o5i = _mm512_fnmadd_pd(s8, vi, o5i);
      o6r = _mm512_fnmadd_pd(s5, vr, o6r); o6i = _mm512_fnmadd_pd(s5, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+112);
      __m512d vi = _mm512_load_pd(AB+120);
      o1r = _mm512_fmadd_pd(s4, vr, o1r); o1i = _mm512_fmadd_pd(s4, vi, o1i);
      o2r = _mm512_fmadd_pd(s8, vr, o2r); o2i = _mm512_fmadd_pd(s8, vi, o2i);
      o3r = _mm512_fnmadd_pd(s11, vr, o3r); o3i = _mm512_fnmadd_pd(s11, vi, o3i);
      o4r = _mm512_fnmadd_pd(s7, vr, o4r); o4i = _mm512_fnmadd_pd(s7, vi, o4i);
      o5r = _mm512_fnmadd_pd(s3, vr, o5r); o5i = _mm512_fnmadd_pd(s3, vi, o5i);
      o6r = _mm512_fmadd_pd(s1, vr, o6r); o6i = _mm512_fmadd_pd(s1, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+144);
      __m512d vi = _mm512_load_pd(AB+152);
      o1r = _mm512_fmadd_pd(s5, vr, o1r); o1i = _mm512_fmadd_pd(s5, vi, o1i);
      o2r = _mm512_fmadd_pd(s10, vr, o2r); o2i = _mm512_fmadd_pd(s10, vi, o2i);
      o3r = _mm512_fnmadd_pd(s8, vr, o3r); o3i = _mm512_fnmadd_pd(s8, vi, o3i);
      o4r = _mm512_fnmadd_pd(s3, vr, o4r); o4i = _mm512_fnmadd_pd(s3, vi, o4i);
      o5r = _mm512_fmadd_pd(s2, vr, o5r); o5i = _mm512_fmadd_pd(s2, vi, o5i);
      o6r = _mm512_fmadd_pd(s7, vr, o6r); o6i = _mm512_fmadd_pd(s7, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+176);
      __m512d vi = _mm512_load_pd(AB+184);
      o1r = _mm512_fmadd_pd(s6, vr, o1r); o1i = _mm512_fmadd_pd(s6, vi, o1i);
      o2r = _mm512_fnmadd_pd(s11, vr, o2r); o2i = _mm512_fnmadd_pd(s11, vi, o2i);
      o3r = _mm512_fnmadd_pd(s5, vr, o3r); o3i = _mm512_fnmadd_pd(s5, vi, o3i);
      o4r = _mm512_fmadd_pd(s1, vr, o4r); o4i = _mm512_fmadd_pd(s1, vi, o4i);
      o5r = _mm512_fmadd_pd(s7, vr, o5r); o5i = _mm512_fmadd_pd(s7, vi, o5i);
      o6r = _mm512_fnmadd_pd(s10, vr, o6r); o6i = _mm512_fnmadd_pd(s10, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+208);
      __m512d vi = _mm512_load_pd(AB+216);
      o1r = _mm512_fmadd_pd(s7, vr, o1r); o1i = _mm512_fmadd_pd(s7, vi, o1i);
      o2r = _mm512_fnmadd_pd(s9, vr, o2r); o2i = _mm512_fnmadd_pd(s9, vi, o2i);
      o3r = _mm512_fnmadd_pd(s2, vr, o3r); o3i = _mm512_fnmadd_pd(s2, vi, o3i);
      o4r = _mm512_fmadd_pd(s5, vr, o4r); o4i = _mm512_fmadd_pd(s5, vi, o4i);
      o5r = _mm512_fnmadd_pd(s11, vr, o5r); o5i = _mm512_fnmadd_pd(s11, vi, o5i);
      o6r = _mm512_fnmadd_pd(s4, vr, o6r); o6i = _mm512_fnmadd_pd(s4, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+240);
      __m512d vi = _mm512_load_pd(AB+248);
      o1r = _mm512_fmadd_pd(s8, vr, o1r); o1i = _mm512_fmadd_pd(s8, vi, o1i);
      o2r = _mm512_fnmadd_pd(s7, vr, o2r); o2i = _mm512_fnmadd_pd(s7, vi, o2i);
      o3r = _mm512_fmadd_pd(s1, vr, o3r); o3i = _mm512_fmadd_pd(s1, vi, o3i);
      o4r = _mm512_fmadd_pd(s9, vr, o4r); o4i = _mm512_fmadd_pd(s9, vi, o4i);
      o5r = _mm512_fnmadd_pd(s6, vr, o5r); o5i = _mm512_fnmadd_pd(s6, vi, o5i);
      o6r = _mm512_fmadd_pd(s2, vr, o6r); o6i = _mm512_fmadd_pd(s2, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+272);
      __m512d vi = _mm512_load_pd(AB+280);
      o1r = _mm512_fmadd_pd(s9, vr, o1r); o1i = _mm512_fmadd_pd(s9, vi, o1i);
      o2r = _mm512_fnmadd_pd(s5, vr, o2r); o2i = _mm512_fnmadd_pd(s5, vi, o2i);
      o3r = _mm512_fmadd_pd(s4, vr, o3r); o3i = _mm512_fmadd_pd(s4, vi, o3i);
      o4r = _mm512_fnmadd_pd(s10, vr, o4r); o4i = _mm512_fnmadd_pd(s10, vi, o4i);
      o5r = _mm512_fnmadd_pd(s1, vr, o5r); o5i = _mm512_fnmadd_pd(s1, vi, o5i);
      o6r = _mm512_fmadd_pd(s8, vr, o6r); o6i = _mm512_fmadd_pd(s8, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+304);
      __m512d vi = _mm512_load_pd(AB+312);
      o1r = _mm512_fmadd_pd(s10, vr, o1r); o1i = _mm512_fmadd_pd(s10, vi, o1i);
      o2r = _mm512_fnmadd_pd(s3, vr, o2r); o2i = _mm512_fnmadd_pd(s3, vi, o2i);
      o3r = _mm512_fmadd_pd(s7, vr, o3r); o3i = _mm512_fmadd_pd(s7, vi, o3i);
      o4r = _mm512_fnmadd_pd(s6, vr, o4r); o4i = _mm512_fnmadd_pd(s6, vi, o4i);
      o5r = _mm512_fmadd_pd(s4, vr, o5r); o5i = _mm512_fmadd_pd(s4, vi, o5i);
      o6r = _mm512_fnmadd_pd(s9, vr, o6r); o6i = _mm512_fnmadd_pd(s9, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+336);
      __m512d vi = _mm512_load_pd(AB+344);
      o1r = _mm512_fmadd_pd(s11, vr, o1r); o1i = _mm512_fmadd_pd(s11, vi, o1i);
      o2r = _mm512_fnmadd_pd(s1, vr, o2r); o2i = _mm512_fnmadd_pd(s1, vi, o2i);
      o3r = _mm512_fmadd_pd(s10, vr, o3r); o3i = _mm512_fmadd_pd(s10, vi, o3i);
      o4r = _mm512_fnmadd_pd(s2, vr, o4r); o4i = _mm512_fnmadd_pd(s2, vi, o4i);
      o5r = _mm512_fmadd_pd(s9, vr, o5r); o5i = _mm512_fmadd_pd(s9, vi, o5i);
      o6r = _mm512_fnmadd_pd(s3, vr, o6r); o6i = _mm512_fnmadd_pd(s3, vi, o6i);
    }
    { __m512d er = _mm512_load_pd(Escr+0), ei = _mm512_load_pd(Escr+8);
      _mm512_store_pd(X+1*s,   _mm512_add_pd(er, o1i));
      _mm512_store_pd(X+1*s+8, _mm512_sub_pd(ei, o1r));
      _mm512_store_pd(X+22*s,   _mm512_sub_pd(er, o1i));
      _mm512_store_pd(X+22*s+8, _mm512_add_pd(ei, o1r)); }
    { __m512d er = _mm512_load_pd(Escr+16), ei = _mm512_load_pd(Escr+24);
      _mm512_store_pd(X+2*s,   _mm512_add_pd(er, o2i));
      _mm512_store_pd(X+2*s+8, _mm512_sub_pd(ei, o2r));
      _mm512_store_pd(X+21*s,   _mm512_sub_pd(er, o2i));
      _mm512_store_pd(X+21*s+8, _mm512_add_pd(ei, o2r)); }
    { __m512d er = _mm512_load_pd(Escr+32), ei = _mm512_load_pd(Escr+40);
      _mm512_store_pd(X+3*s,   _mm512_add_pd(er, o3i));
      _mm512_store_pd(X+3*s+8, _mm512_sub_pd(ei, o3r));
      _mm512_store_pd(X+20*s,   _mm512_sub_pd(er, o3i));
      _mm512_store_pd(X+20*s+8, _mm512_add_pd(ei, o3r)); }
    { __m512d er = _mm512_load_pd(Escr+48), ei = _mm512_load_pd(Escr+56);
      _mm512_store_pd(X+4*s,   _mm512_add_pd(er, o4i));
      _mm512_store_pd(X+4*s+8, _mm512_sub_pd(ei, o4r));
      _mm512_store_pd(X+19*s,   _mm512_sub_pd(er, o4i));
      _mm512_store_pd(X+19*s+8, _mm512_add_pd(ei, o4r)); }
    { __m512d er = _mm512_load_pd(Escr+64), ei = _mm512_load_pd(Escr+72);
      _mm512_store_pd(X+5*s,   _mm512_add_pd(er, o5i));
      _mm512_store_pd(X+5*s+8, _mm512_sub_pd(ei, o5r));
      _mm512_store_pd(X+18*s,   _mm512_sub_pd(er, o5i));
      _mm512_store_pd(X+18*s+8, _mm512_add_pd(ei, o5r)); }
    { __m512d er = _mm512_load_pd(Escr+80), ei = _mm512_load_pd(Escr+88);
      _mm512_store_pd(X+6*s,   _mm512_add_pd(er, o6i));
      _mm512_store_pd(X+6*s+8, _mm512_sub_pd(ei, o6r));
      _mm512_store_pd(X+17*s,   _mm512_sub_pd(er, o6i));
      _mm512_store_pd(X+17*s+8, _mm512_add_pd(ei, o6r)); }
    }
    { __m512d s1 = _mm512_set1_pd(0x1.14459ad2be466p-2), s2 = _mm512_set1_pd(0x1.0a06e851db7cap-1), s3 = _mm512_set1_pd(0x1.763021aaa15d9p-1), s4 = _mm512_set1_pd(0x1.c698e42f47b09p-1), s5 = _mm512_set1_pd(0x1.f54a827142577p-1), s6 = _mm512_set1_pd(0x1.fece70dfd3efbp-1), s7 = _mm512_set1_pd(0x1.e270060999288p-1), s8 = _mm512_set1_pd(0x1.a249e0b897caap-1), s9 = _mm512_set1_pd(0x1.431df5838f7f1p-1), s10 = _mm512_set1_pd(0x1.97f6748e524b1p-2), s11 = _mm512_set1_pd(0x1.16de8a4564f1cp-3);
    __m512d o7r = _mm512_setzero_pd(), o7i = _mm512_setzero_pd();
    __m512d o8r = _mm512_setzero_pd(), o8i = _mm512_setzero_pd();
    __m512d o9r = _mm512_setzero_pd(), o9i = _mm512_setzero_pd();
    __m512d o10r = _mm512_setzero_pd(), o10i = _mm512_setzero_pd();
    __m512d o11r = _mm512_setzero_pd(), o11i = _mm512_setzero_pd();
    { __m512d vr = _mm512_load_pd(AB+16);
      __m512d vi = _mm512_load_pd(AB+24);
      o7r = _mm512_fmadd_pd(s7, vr, o7r); o7i = _mm512_fmadd_pd(s7, vi, o7i);
      o8r = _mm512_fmadd_pd(s8, vr, o8r); o8i = _mm512_fmadd_pd(s8, vi, o8i);
      o9r = _mm512_fmadd_pd(s9, vr, o9r); o9i = _mm512_fmadd_pd(s9, vi, o9i);
      o10r = _mm512_fmadd_pd(s10, vr, o10r); o10i = _mm512_fmadd_pd(s10, vi, o10i);
      o11r = _mm512_fmadd_pd(s11, vr, o11r); o11i = _mm512_fmadd_pd(s11, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+48);
      __m512d vi = _mm512_load_pd(AB+56);
      o7r = _mm512_fnmadd_pd(s9, vr, o7r); o7i = _mm512_fnmadd_pd(s9, vi, o7i);
      o8r = _mm512_fnmadd_pd(s7, vr, o8r); o8i = _mm512_fnmadd_pd(s7, vi, o8i);
      o9r = _mm512_fnmadd_pd(s5, vr, o9r); o9i = _mm512_fnmadd_pd(s5, vi, o9i);
      o10r = _mm512_fnmadd_pd(s3, vr, o10r); o10i = _mm512_fnmadd_pd(s3, vi, o10i);
      o11r = _mm512_fnmadd_pd(s1, vr, o11r); o11i = _mm512_fnmadd_pd(s1, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+80);
      __m512d vi = _mm512_load_pd(AB+88);
      o7r = _mm512_fnmadd_pd(s2, vr, o7r); o7i = _mm512_fnmadd_pd(s2, vi, o7i);
      o8r = _mm512_fmadd_pd(s1, vr, o8r); o8i = _mm512_fmadd_pd(s1, vi, o8i);
      o9r = _mm512_fmadd_pd(s4, vr, o9r); o9i = _mm512_fmadd_pd(s4, vi, o9i);
      o10r = _mm512_fmadd_pd(s7, vr, o10r); o10i = _mm512_fmadd_pd(s7, vi, o10i);
      o11r = _mm512_fmadd_pd(s10, vr, o11r); o11i = _mm512_fmadd_pd(s10, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+112);
      __m512d vi = _mm512_load_pd(AB+120);
      o7r = _mm512_fmadd_pd(s5, vr, o7r); o7i = _mm512_fmadd_pd(s5, vi, o7i);
      o8r = _mm512_fmadd_pd(s9, vr, o8r); o8i = _mm512_fmadd_pd(s9, vi, o8i);
      o9r = _mm512_fnmadd_pd(s10, vr, o9r); o9i = _mm512_fnmadd_pd(s10, vi, o9i);
      o10r = _mm512_fnmadd_pd(s6, vr, o10r); o10i = _mm512_fnmadd_pd(s6, vi, o10i);
      o11r = _mm512_fnmadd_pd(s2, vr, o11r); o11i = _mm512_fnmadd_pd(s2, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+144);
      __m512d vi = _mm512_load_pd(AB+152);
      o7r = _mm512_fnmadd_pd(s11, vr, o7r); o7i = _mm512_fnmadd_pd(s11, vi, o7i);
      o8r = _mm512_fnmadd_pd(s6, vr, o8r); o8i = _mm512_fnmadd_pd(s6, vi, o8i);
      o9r = _mm512_fnmadd_pd(s1, vr, o9r); o9i = _mm512_fnmadd_pd(s1, vi, o9i);
      o10r = _mm512_fmadd_pd(s4, vr, o10r); o10i = _mm512_fmadd_pd(s4, vi, o10i);
      o11r = _mm512_fmadd_pd(s9, vr, o11r); o11i = _mm512_fmadd_pd(s9, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+176);
      __m512d vi = _mm512_load_pd(AB+184);
      o7r = _mm512_fnmadd_pd(s4, vr, o7r); o7i = _mm512_fnmadd_pd(s4, vi, o7i);
      o8r = _mm512_fmadd_pd(s2, vr, o8r); o8i = _mm512_fmadd_pd(s2, vi, o8i);
      o9r = _mm512_fmadd_pd(s8, vr, o9r); o9i = _mm512_fmadd_pd(s8, vi, o9i);
      o10r = _mm512_fnmadd_pd(s9, vr, o10r); o10i = _mm512_fnmadd_pd(s9, vi, o10i);
      o11r = _mm512_fnmadd_pd(s3, vr, o11r); o11i = _mm512_fnmadd_pd(s3, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+208);
      __m512d vi = _mm512_load_pd(AB+216);
      o7r = _mm512_fmadd_pd(s3, vr, o7r); o7i = _mm512_fmadd_pd(s3, vi, o7i);
      o8r = _mm512_fmadd_pd(s10, vr, o8r); o8i = _mm512_fmadd_pd(s10, vi, o8i);
      o9r = _mm512_fnmadd_pd(s6, vr, o9r); o9i = _mm512_fnmadd_pd(s6, vi, o9i);
      o10r = _mm512_fmadd_pd(s1, vr, o10r); o10i = _mm512_fmadd_pd(s1, vi, o10i);
      o11r = _mm512_fmadd_pd(s8, vr, o11r); o11i = _mm512_fmadd_pd(s8, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+240);
      __m512d vi = _mm512_load_pd(AB+248);
      o7r = _mm512_fmadd_pd(s10, vr, o7r); o7i = _mm512_fmadd_pd(s10, vi, o7i);
      o8r = _mm512_fnmadd_pd(s5, vr, o8r); o8i = _mm512_fnmadd_pd(s5, vi, o8i);
      o9r = _mm512_fmadd_pd(s3, vr, o9r); o9i = _mm512_fmadd_pd(s3, vi, o9i);
      o10r = _mm512_fmadd_pd(s11, vr, o10r); o10i = _mm512_fmadd_pd(s11, vi, o10i);
      o11r = _mm512_fnmadd_pd(s4, vr, o11r); o11i = _mm512_fnmadd_pd(s4, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+272);
      __m512d vi = _mm512_load_pd(AB+280);
      o7r = _mm512_fnmadd_pd(s6, vr, o7r); o7i = _mm512_fnmadd_pd(s6, vi, o7i);
      o8r = _mm512_fmadd_pd(s3, vr, o8r); o8i = _mm512_fmadd_pd(s3, vi, o8i);
      o9r = _mm512_fnmadd_pd(s11, vr, o9r); o9i = _mm512_fnmadd_pd(s11, vi, o9i);
      o10r = _mm512_fnmadd_pd(s2, vr, o10r); o10i = _mm512_fnmadd_pd(s2, vi, o10i);
      o11r = _mm512_fmadd_pd(s7, vr, o11r); o11i = _mm512_fmadd_pd(s7, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+304);
      __m512d vi = _mm512_load_pd(AB+312);
      o7r = _mm512_fmadd_pd(s1, vr, o7r); o7i = _mm512_fmadd_pd(s1, vi, o7i);
      o8r = _mm512_fmadd_pd(s11, vr, o8r); o8i = _mm512_fmadd_pd(s11, vi, o8i);
      o9r = _mm512_fnmadd_pd(s2, vr, o9r); o9i = _mm512_fnmadd_pd(s2, vi, o9i);
      o10r = _mm512_fmadd_pd(s8, vr, o10r); o10i = _mm512_fmadd_pd(s8, vi, o10i);
      o11r = _mm512_fnmadd_pd(s5, vr, o11r); o11i = _mm512_fnmadd_pd(s5, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+336);
      __m512d vi = _mm512_load_pd(AB+344);
      o7r = _mm512_fmadd_pd(s8, vr, o7r); o7i = _mm512_fmadd_pd(s8, vi, o7i);
      o8r = _mm512_fnmadd_pd(s4, vr, o8r); o8i = _mm512_fnmadd_pd(s4, vi, o8i);
      o9r = _mm512_fmadd_pd(s7, vr, o9r); o9i = _mm512_fmadd_pd(s7, vi, o9i);
      o10r = _mm512_fnmadd_pd(s5, vr, o10r); o10i = _mm512_fnmadd_pd(s5, vi, o10i);
      o11r = _mm512_fmadd_pd(s6, vr, o11r); o11i = _mm512_fmadd_pd(s6, vi, o11i);
    }
    { __m512d er = _mm512_load_pd(Escr+96), ei = _mm512_load_pd(Escr+104);
      _mm512_store_pd(X+7*s,   _mm512_add_pd(er, o7i));
      _mm512_store_pd(X+7*s+8, _mm512_sub_pd(ei, o7r));
      _mm512_store_pd(X+16*s,   _mm512_sub_pd(er, o7i));
      _mm512_store_pd(X+16*s+8, _mm512_add_pd(ei, o7r)); }
    { __m512d er = _mm512_load_pd(Escr+112), ei = _mm512_load_pd(Escr+120);
      _mm512_store_pd(X+8*s,   _mm512_add_pd(er, o8i));
      _mm512_store_pd(X+8*s+8, _mm512_sub_pd(ei, o8r));
      _mm512_store_pd(X+15*s,   _mm512_sub_pd(er, o8i));
      _mm512_store_pd(X+15*s+8, _mm512_add_pd(ei, o8r)); }
    { __m512d er = _mm512_load_pd(Escr+128), ei = _mm512_load_pd(Escr+136);
      _mm512_store_pd(X+9*s,   _mm512_add_pd(er, o9i));
      _mm512_store_pd(X+9*s+8, _mm512_sub_pd(ei, o9r));
      _mm512_store_pd(X+14*s,   _mm512_sub_pd(er, o9i));
      _mm512_store_pd(X+14*s+8, _mm512_add_pd(ei, o9r)); }
    { __m512d er = _mm512_load_pd(Escr+144), ei = _mm512_load_pd(Escr+152);
      _mm512_store_pd(X+10*s,   _mm512_add_pd(er, o10i));
      _mm512_store_pd(X+10*s+8, _mm512_sub_pd(ei, o10r));
      _mm512_store_pd(X+13*s,   _mm512_sub_pd(er, o10i));
      _mm512_store_pd(X+13*s+8, _mm512_add_pd(ei, o10r)); }
    { __m512d er = _mm512_load_pd(Escr+160), ei = _mm512_load_pd(Escr+168);
      _mm512_store_pd(X+11*s,   _mm512_add_pd(er, o11i));
      _mm512_store_pd(X+11*s+8, _mm512_sub_pd(ei, o11r));
      _mm512_store_pd(X+12*s,   _mm512_sub_pd(er, o11i));
      _mm512_store_pd(X+12*s+8, _mm512_add_pd(ei, o11r)); }
    }
}
static __attribute__((always_inline)) inline void dft23zm(double* restrict X, long es, int dopf){
    const long s = es*16;
    double AB[88*8] ALIGN64;
    double Escr[22*8] ALIGN64;
    __m512d x0r = _mm512_load_pd(X);
    if(dopf){ _mm_prefetch((const char*)(X+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+376), _MM_HINT_T0); }
    __m512d x0i = _mm512_load_pd(X+8);
    maphw(x0r, x0i, &x0r, &x0i);
    { __m512d c1 = _mm512_set1_pd(0x1.ed037ea3d2dbcp-1), c2 = _mm512_set1_pd(0x1.b57675cf309eep-1), c3 = _mm512_set1_pd(0x1.5d779b07cfef7p-1), c4 = _mm512_set1_pd(0x1.d71b4a0c5a6c9p-2), c5 = _mm512_set1_pd(0x1.a0ad8bd1e2881p-3), c6 = _mm512_set1_pd(-0x1.17855b599f3b2p-4), c7 = _mm512_set1_pd(-0x1.56eaae597c776p-2), c8 = _mm512_set1_pd(-0x1.2742a4a775cfap-1), c9 = _mm512_set1_pd(-0x1.8d2a07c16d46ep-1), c10 = _mm512_set1_pd(-0x1.d59cb83ef99bcp-1), c11 = _mm512_set1_pd(-0x1.fb3b3035aa6ccp-1);
    __m512d e1r = x0r, e1i = x0i;
    __m512d e2r = x0r, e2i = x0i;
    __m512d e3r = x0r, e3i = x0i;
    __m512d e4r = x0r, e4i = x0i;
    __m512d e5r = x0r, e5i = x0i;
    __m512d e6r = x0r, e6i = x0i;
    __m512d sr = x0r, si = x0i;
    { __m512d pr = _mm512_load_pd(X+1*s), qr = _mm512_load_pd(X+22*s);
      if(dopf){ _mm_prefetch((const char*)(X+1*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+1*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+22*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+22*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+1*s+8), qi = _mm512_load_pd(X+22*s+8);
      map2(pr, pi, &pr, &pi); maphw(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+0, ur);    _mm512_store_pd(AB+8, ui);
      _mm512_store_pd(AB+16, vr); _mm512_store_pd(AB+24, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c1, ur, e1r); e1i = _mm512_fmadd_pd(c1, ui, e1i);
      e2r = _mm512_fmadd_pd(c2, ur, e2r); e2i = _mm512_fmadd_pd(c2, ui, e2i);
      e3r = _mm512_fmadd_pd(c3, ur, e3r); e3i = _mm512_fmadd_pd(c3, ui, e3i);
      e4r = _mm512_fmadd_pd(c4, ur, e4r); e4i = _mm512_fmadd_pd(c4, ui, e4i);
      e5r = _mm512_fmadd_pd(c5, ur, e5r); e5i = _mm512_fmadd_pd(c5, ui, e5i);
      e6r = _mm512_fmadd_pd(c6, ur, e6r); e6i = _mm512_fmadd_pd(c6, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+2*s), qr = _mm512_load_pd(X+21*s);
      if(dopf){ _mm_prefetch((const char*)(X+2*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+2*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+21*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+21*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+2*s+8), qi = _mm512_load_pd(X+21*s+8);
      maphw(pr, pi, &pr, &pi); map2(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+32, ur);    _mm512_store_pd(AB+40, ui);
      _mm512_store_pd(AB+48, vr); _mm512_store_pd(AB+56, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c2, ur, e1r); e1i = _mm512_fmadd_pd(c2, ui, e1i);
      e2r = _mm512_fmadd_pd(c4, ur, e2r); e2i = _mm512_fmadd_pd(c4, ui, e2i);
      e3r = _mm512_fmadd_pd(c6, ur, e3r); e3i = _mm512_fmadd_pd(c6, ui, e3i);
      e4r = _mm512_fmadd_pd(c8, ur, e4r); e4i = _mm512_fmadd_pd(c8, ui, e4i);
      e5r = _mm512_fmadd_pd(c10, ur, e5r); e5i = _mm512_fmadd_pd(c10, ui, e5i);
      e6r = _mm512_fmadd_pd(c11, ur, e6r); e6i = _mm512_fmadd_pd(c11, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+3*s), qr = _mm512_load_pd(X+20*s);
      if(dopf){ _mm_prefetch((const char*)(X+3*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+3*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+20*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+20*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+3*s+8), qi = _mm512_load_pd(X+20*s+8);
      map2(pr, pi, &pr, &pi); maphw(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+64, ur);    _mm512_store_pd(AB+72, ui);
      _mm512_store_pd(AB+80, vr); _mm512_store_pd(AB+88, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c3, ur, e1r); e1i = _mm512_fmadd_pd(c3, ui, e1i);
      e2r = _mm512_fmadd_pd(c6, ur, e2r); e2i = _mm512_fmadd_pd(c6, ui, e2i);
      e3r = _mm512_fmadd_pd(c9, ur, e3r); e3i = _mm512_fmadd_pd(c9, ui, e3i);
      e4r = _mm512_fmadd_pd(c11, ur, e4r); e4i = _mm512_fmadd_pd(c11, ui, e4i);
      e5r = _mm512_fmadd_pd(c8, ur, e5r); e5i = _mm512_fmadd_pd(c8, ui, e5i);
      e6r = _mm512_fmadd_pd(c5, ur, e6r); e6i = _mm512_fmadd_pd(c5, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+4*s), qr = _mm512_load_pd(X+19*s);
      if(dopf){ _mm_prefetch((const char*)(X+4*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+4*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+19*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+19*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+4*s+8), qi = _mm512_load_pd(X+19*s+8);
      maphw(pr, pi, &pr, &pi); map2(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+96, ur);    _mm512_store_pd(AB+104, ui);
      _mm512_store_pd(AB+112, vr); _mm512_store_pd(AB+120, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c4, ur, e1r); e1i = _mm512_fmadd_pd(c4, ui, e1i);
      e2r = _mm512_fmadd_pd(c8, ur, e2r); e2i = _mm512_fmadd_pd(c8, ui, e2i);
      e3r = _mm512_fmadd_pd(c11, ur, e3r); e3i = _mm512_fmadd_pd(c11, ui, e3i);
      e4r = _mm512_fmadd_pd(c7, ur, e4r); e4i = _mm512_fmadd_pd(c7, ui, e4i);
      e5r = _mm512_fmadd_pd(c3, ur, e5r); e5i = _mm512_fmadd_pd(c3, ui, e5i);
      e6r = _mm512_fmadd_pd(c1, ur, e6r); e6i = _mm512_fmadd_pd(c1, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+5*s), qr = _mm512_load_pd(X+18*s);
      if(dopf){ _mm_prefetch((const char*)(X+5*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+5*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+18*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+18*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+5*s+8), qi = _mm512_load_pd(X+18*s+8);
      map2(pr, pi, &pr, &pi); maphw(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+128, ur);    _mm512_store_pd(AB+136, ui);
      _mm512_store_pd(AB+144, vr); _mm512_store_pd(AB+152, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c5, ur, e1r); e1i = _mm512_fmadd_pd(c5, ui, e1i);
      e2r = _mm512_fmadd_pd(c10, ur, e2r); e2i = _mm512_fmadd_pd(c10, ui, e2i);
      e3r = _mm512_fmadd_pd(c8, ur, e3r); e3i = _mm512_fmadd_pd(c8, ui, e3i);
      e4r = _mm512_fmadd_pd(c3, ur, e4r); e4i = _mm512_fmadd_pd(c3, ui, e4i);
      e5r = _mm512_fmadd_pd(c2, ur, e5r); e5i = _mm512_fmadd_pd(c2, ui, e5i);
      e6r = _mm512_fmadd_pd(c7, ur, e6r); e6i = _mm512_fmadd_pd(c7, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+6*s), qr = _mm512_load_pd(X+17*s);
      if(dopf){ _mm_prefetch((const char*)(X+6*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+6*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+17*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+17*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+6*s+8), qi = _mm512_load_pd(X+17*s+8);
      maphw(pr, pi, &pr, &pi); map2(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+160, ur);    _mm512_store_pd(AB+168, ui);
      _mm512_store_pd(AB+176, vr); _mm512_store_pd(AB+184, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c6, ur, e1r); e1i = _mm512_fmadd_pd(c6, ui, e1i);
      e2r = _mm512_fmadd_pd(c11, ur, e2r); e2i = _mm512_fmadd_pd(c11, ui, e2i);
      e3r = _mm512_fmadd_pd(c5, ur, e3r); e3i = _mm512_fmadd_pd(c5, ui, e3i);
      e4r = _mm512_fmadd_pd(c1, ur, e4r); e4i = _mm512_fmadd_pd(c1, ui, e4i);
      e5r = _mm512_fmadd_pd(c7, ur, e5r); e5i = _mm512_fmadd_pd(c7, ui, e5i);
      e6r = _mm512_fmadd_pd(c10, ur, e6r); e6i = _mm512_fmadd_pd(c10, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+7*s), qr = _mm512_load_pd(X+16*s);
      if(dopf){ _mm_prefetch((const char*)(X+7*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+7*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+16*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+16*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+7*s+8), qi = _mm512_load_pd(X+16*s+8);
      map2(pr, pi, &pr, &pi); maphw(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+192, ur);    _mm512_store_pd(AB+200, ui);
      _mm512_store_pd(AB+208, vr); _mm512_store_pd(AB+216, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c7, ur, e1r); e1i = _mm512_fmadd_pd(c7, ui, e1i);
      e2r = _mm512_fmadd_pd(c9, ur, e2r); e2i = _mm512_fmadd_pd(c9, ui, e2i);
      e3r = _mm512_fmadd_pd(c2, ur, e3r); e3i = _mm512_fmadd_pd(c2, ui, e3i);
      e4r = _mm512_fmadd_pd(c5, ur, e4r); e4i = _mm512_fmadd_pd(c5, ui, e4i);
      e5r = _mm512_fmadd_pd(c11, ur, e5r); e5i = _mm512_fmadd_pd(c11, ui, e5i);
      e6r = _mm512_fmadd_pd(c4, ur, e6r); e6i = _mm512_fmadd_pd(c4, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+8*s), qr = _mm512_load_pd(X+15*s);
      if(dopf){ _mm_prefetch((const char*)(X+8*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+8*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+15*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+15*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+8*s+8), qi = _mm512_load_pd(X+15*s+8);
      maphw(pr, pi, &pr, &pi); map2(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+224, ur);    _mm512_store_pd(AB+232, ui);
      _mm512_store_pd(AB+240, vr); _mm512_store_pd(AB+248, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c8, ur, e1r); e1i = _mm512_fmadd_pd(c8, ui, e1i);
      e2r = _mm512_fmadd_pd(c7, ur, e2r); e2i = _mm512_fmadd_pd(c7, ui, e2i);
      e3r = _mm512_fmadd_pd(c1, ur, e3r); e3i = _mm512_fmadd_pd(c1, ui, e3i);
      e4r = _mm512_fmadd_pd(c9, ur, e4r); e4i = _mm512_fmadd_pd(c9, ui, e4i);
      e5r = _mm512_fmadd_pd(c6, ur, e5r); e5i = _mm512_fmadd_pd(c6, ui, e5i);
      e6r = _mm512_fmadd_pd(c2, ur, e6r); e6i = _mm512_fmadd_pd(c2, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+9*s), qr = _mm512_load_pd(X+14*s);
      if(dopf){ _mm_prefetch((const char*)(X+9*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+9*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+14*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+14*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+9*s+8), qi = _mm512_load_pd(X+14*s+8);
      map2(pr, pi, &pr, &pi); maphw(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+256, ur);    _mm512_store_pd(AB+264, ui);
      _mm512_store_pd(AB+272, vr); _mm512_store_pd(AB+280, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c9, ur, e1r); e1i = _mm512_fmadd_pd(c9, ui, e1i);
      e2r = _mm512_fmadd_pd(c5, ur, e2r); e2i = _mm512_fmadd_pd(c5, ui, e2i);
      e3r = _mm512_fmadd_pd(c4, ur, e3r); e3i = _mm512_fmadd_pd(c4, ui, e3i);
      e4r = _mm512_fmadd_pd(c10, ur, e4r); e4i = _mm512_fmadd_pd(c10, ui, e4i);
      e5r = _mm512_fmadd_pd(c1, ur, e5r); e5i = _mm512_fmadd_pd(c1, ui, e5i);
      e6r = _mm512_fmadd_pd(c8, ur, e6r); e6i = _mm512_fmadd_pd(c8, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+10*s), qr = _mm512_load_pd(X+13*s);
      if(dopf){ _mm_prefetch((const char*)(X+10*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+10*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+13*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+13*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+10*s+8), qi = _mm512_load_pd(X+13*s+8);
      maphw(pr, pi, &pr, &pi); map2(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+288, ur);    _mm512_store_pd(AB+296, ui);
      _mm512_store_pd(AB+304, vr); _mm512_store_pd(AB+312, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c10, ur, e1r); e1i = _mm512_fmadd_pd(c10, ui, e1i);
      e2r = _mm512_fmadd_pd(c3, ur, e2r); e2i = _mm512_fmadd_pd(c3, ui, e2i);
      e3r = _mm512_fmadd_pd(c7, ur, e3r); e3i = _mm512_fmadd_pd(c7, ui, e3i);
      e4r = _mm512_fmadd_pd(c6, ur, e4r); e4i = _mm512_fmadd_pd(c6, ui, e4i);
      e5r = _mm512_fmadd_pd(c4, ur, e5r); e5i = _mm512_fmadd_pd(c4, ui, e5i);
      e6r = _mm512_fmadd_pd(c9, ur, e6r); e6i = _mm512_fmadd_pd(c9, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+11*s), qr = _mm512_load_pd(X+12*s);
      if(dopf){ _mm_prefetch((const char*)(X+11*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+11*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+12*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+12*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+11*s+8), qi = _mm512_load_pd(X+12*s+8);
      map2(pr, pi, &pr, &pi); maphw(qr, qi, &qr, &qi);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+320, ur);    _mm512_store_pd(AB+328, ui);
      _mm512_store_pd(AB+336, vr); _mm512_store_pd(AB+344, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c11, ur, e1r); e1i = _mm512_fmadd_pd(c11, ui, e1i);
      e2r = _mm512_fmadd_pd(c1, ur, e2r); e2i = _mm512_fmadd_pd(c1, ui, e2i);
      e3r = _mm512_fmadd_pd(c10, ur, e3r); e3i = _mm512_fmadd_pd(c10, ui, e3i);
      e4r = _mm512_fmadd_pd(c2, ur, e4r); e4i = _mm512_fmadd_pd(c2, ui, e4i);
      e5r = _mm512_fmadd_pd(c9, ur, e5r); e5i = _mm512_fmadd_pd(c9, ui, e5i);
      e6r = _mm512_fmadd_pd(c3, ur, e6r); e6i = _mm512_fmadd_pd(c3, ui, e6i);
    }
    _mm512_store_pd(X, sr); _mm512_store_pd(X+8, si);
    _mm512_store_pd(Escr+0, e1r); _mm512_store_pd(Escr+8, e1i);
    _mm512_store_pd(Escr+16, e2r); _mm512_store_pd(Escr+24, e2i);
    _mm512_store_pd(Escr+32, e3r); _mm512_store_pd(Escr+40, e3i);
    _mm512_store_pd(Escr+48, e4r); _mm512_store_pd(Escr+56, e4i);
    _mm512_store_pd(Escr+64, e5r); _mm512_store_pd(Escr+72, e5i);
    _mm512_store_pd(Escr+80, e6r); _mm512_store_pd(Escr+88, e6i);
    }
    { __m512d c1 = _mm512_set1_pd(0x1.ed037ea3d2dbcp-1), c2 = _mm512_set1_pd(0x1.b57675cf309eep-1), c3 = _mm512_set1_pd(0x1.5d779b07cfef7p-1), c4 = _mm512_set1_pd(0x1.d71b4a0c5a6c9p-2), c5 = _mm512_set1_pd(0x1.a0ad8bd1e2881p-3), c6 = _mm512_set1_pd(-0x1.17855b599f3b2p-4), c7 = _mm512_set1_pd(-0x1.56eaae597c776p-2), c8 = _mm512_set1_pd(-0x1.2742a4a775cfap-1), c9 = _mm512_set1_pd(-0x1.8d2a07c16d46ep-1), c10 = _mm512_set1_pd(-0x1.d59cb83ef99bcp-1), c11 = _mm512_set1_pd(-0x1.fb3b3035aa6ccp-1);
    __m512d e7r = x0r, e7i = x0i;
    __m512d e8r = x0r, e8i = x0i;
    __m512d e9r = x0r, e9i = x0i;
    __m512d e10r = x0r, e10i = x0i;
    __m512d e11r = x0r, e11i = x0i;
    { __m512d ur = _mm512_load_pd(AB+0);
      __m512d ui = _mm512_load_pd(AB+8);
      e7r = _mm512_fmadd_pd(c7, ur, e7r); e7i = _mm512_fmadd_pd(c7, ui, e7i);
      e8r = _mm512_fmadd_pd(c8, ur, e8r); e8i = _mm512_fmadd_pd(c8, ui, e8i);
      e9r = _mm512_fmadd_pd(c9, ur, e9r); e9i = _mm512_fmadd_pd(c9, ui, e9i);
      e10r = _mm512_fmadd_pd(c10, ur, e10r); e10i = _mm512_fmadd_pd(c10, ui, e10i);
      e11r = _mm512_fmadd_pd(c11, ur, e11r); e11i = _mm512_fmadd_pd(c11, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+32);
      __m512d ui = _mm512_load_pd(AB+40);
      e7r = _mm512_fmadd_pd(c9, ur, e7r); e7i = _mm512_fmadd_pd(c9, ui, e7i);
      e8r = _mm512_fmadd_pd(c7, ur, e8r); e8i = _mm512_fmadd_pd(c7, ui, e8i);
      e9r = _mm512_fmadd_pd(c5, ur, e9r); e9i = _mm512_fmadd_pd(c5, ui, e9i);
      e10r = _mm512_fmadd_pd(c3, ur, e10r); e10i = _mm512_fmadd_pd(c3, ui, e10i);
      e11r = _mm512_fmadd_pd(c1, ur, e11r); e11i = _mm512_fmadd_pd(c1, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+64);
      __m512d ui = _mm512_load_pd(AB+72);
      e7r = _mm512_fmadd_pd(c2, ur, e7r); e7i = _mm512_fmadd_pd(c2, ui, e7i);
      e8r = _mm512_fmadd_pd(c1, ur, e8r); e8i = _mm512_fmadd_pd(c1, ui, e8i);
      e9r = _mm512_fmadd_pd(c4, ur, e9r); e9i = _mm512_fmadd_pd(c4, ui, e9i);
      e10r = _mm512_fmadd_pd(c7, ur, e10r); e10i = _mm512_fmadd_pd(c7, ui, e10i);
      e11r = _mm512_fmadd_pd(c10, ur, e11r); e11i = _mm512_fmadd_pd(c10, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+96);
      __m512d ui = _mm512_load_pd(AB+104);
      e7r = _mm512_fmadd_pd(c5, ur, e7r); e7i = _mm512_fmadd_pd(c5, ui, e7i);
      e8r = _mm512_fmadd_pd(c9, ur, e8r); e8i = _mm512_fmadd_pd(c9, ui, e8i);
      e9r = _mm512_fmadd_pd(c10, ur, e9r); e9i = _mm512_fmadd_pd(c10, ui, e9i);
      e10r = _mm512_fmadd_pd(c6, ur, e10r); e10i = _mm512_fmadd_pd(c6, ui, e10i);
      e11r = _mm512_fmadd_pd(c2, ur, e11r); e11i = _mm512_fmadd_pd(c2, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+128);
      __m512d ui = _mm512_load_pd(AB+136);
      e7r = _mm512_fmadd_pd(c11, ur, e7r); e7i = _mm512_fmadd_pd(c11, ui, e7i);
      e8r = _mm512_fmadd_pd(c6, ur, e8r); e8i = _mm512_fmadd_pd(c6, ui, e8i);
      e9r = _mm512_fmadd_pd(c1, ur, e9r); e9i = _mm512_fmadd_pd(c1, ui, e9i);
      e10r = _mm512_fmadd_pd(c4, ur, e10r); e10i = _mm512_fmadd_pd(c4, ui, e10i);
      e11r = _mm512_fmadd_pd(c9, ur, e11r); e11i = _mm512_fmadd_pd(c9, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+160);
      __m512d ui = _mm512_load_pd(AB+168);
      e7r = _mm512_fmadd_pd(c4, ur, e7r); e7i = _mm512_fmadd_pd(c4, ui, e7i);
      e8r = _mm512_fmadd_pd(c2, ur, e8r); e8i = _mm512_fmadd_pd(c2, ui, e8i);
      e9r = _mm512_fmadd_pd(c8, ur, e9r); e9i = _mm512_fmadd_pd(c8, ui, e9i);
      e10r = _mm512_fmadd_pd(c9, ur, e10r); e10i = _mm512_fmadd_pd(c9, ui, e10i);
      e11r = _mm512_fmadd_pd(c3, ur, e11r); e11i = _mm512_fmadd_pd(c3, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+192);
      __m512d ui = _mm512_load_pd(AB+200);
      e7r = _mm512_fmadd_pd(c3, ur, e7r); e7i = _mm512_fmadd_pd(c3, ui, e7i);
      e8r = _mm512_fmadd_pd(c10, ur, e8r); e8i = _mm512_fmadd_pd(c10, ui, e8i);
      e9r = _mm512_fmadd_pd(c6, ur, e9r); e9i = _mm512_fmadd_pd(c6, ui, e9i);
      e10r = _mm512_fmadd_pd(c1, ur, e10r); e10i = _mm512_fmadd_pd(c1, ui, e10i);
      e11r = _mm512_fmadd_pd(c8, ur, e11r); e11i = _mm512_fmadd_pd(c8, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+224);
      __m512d ui = _mm512_load_pd(AB+232);
      e7r = _mm512_fmadd_pd(c10, ur, e7r); e7i = _mm512_fmadd_pd(c10, ui, e7i);
      e8r = _mm512_fmadd_pd(c5, ur, e8r); e8i = _mm512_fmadd_pd(c5, ui, e8i);
      e9r = _mm512_fmadd_pd(c3, ur, e9r); e9i = _mm512_fmadd_pd(c3, ui, e9i);
      e10r = _mm512_fmadd_pd(c11, ur, e10r); e10i = _mm512_fmadd_pd(c11, ui, e10i);
      e11r = _mm512_fmadd_pd(c4, ur, e11r); e11i = _mm512_fmadd_pd(c4, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+256);
      __m512d ui = _mm512_load_pd(AB+264);
      e7r = _mm512_fmadd_pd(c6, ur, e7r); e7i = _mm512_fmadd_pd(c6, ui, e7i);
      e8r = _mm512_fmadd_pd(c3, ur, e8r); e8i = _mm512_fmadd_pd(c3, ui, e8i);
      e9r = _mm512_fmadd_pd(c11, ur, e9r); e9i = _mm512_fmadd_pd(c11, ui, e9i);
      e10r = _mm512_fmadd_pd(c2, ur, e10r); e10i = _mm512_fmadd_pd(c2, ui, e10i);
      e11r = _mm512_fmadd_pd(c7, ur, e11r); e11i = _mm512_fmadd_pd(c7, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+288);
      __m512d ui = _mm512_load_pd(AB+296);
      e7r = _mm512_fmadd_pd(c1, ur, e7r); e7i = _mm512_fmadd_pd(c1, ui, e7i);
      e8r = _mm512_fmadd_pd(c11, ur, e8r); e8i = _mm512_fmadd_pd(c11, ui, e8i);
      e9r = _mm512_fmadd_pd(c2, ur, e9r); e9i = _mm512_fmadd_pd(c2, ui, e9i);
      e10r = _mm512_fmadd_pd(c8, ur, e10r); e10i = _mm512_fmadd_pd(c8, ui, e10i);
      e11r = _mm512_fmadd_pd(c5, ur, e11r); e11i = _mm512_fmadd_pd(c5, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+320);
      __m512d ui = _mm512_load_pd(AB+328);
      e7r = _mm512_fmadd_pd(c8, ur, e7r); e7i = _mm512_fmadd_pd(c8, ui, e7i);
      e8r = _mm512_fmadd_pd(c4, ur, e8r); e8i = _mm512_fmadd_pd(c4, ui, e8i);
      e9r = _mm512_fmadd_pd(c7, ur, e9r); e9i = _mm512_fmadd_pd(c7, ui, e9i);
      e10r = _mm512_fmadd_pd(c5, ur, e10r); e10i = _mm512_fmadd_pd(c5, ui, e10i);
      e11r = _mm512_fmadd_pd(c6, ur, e11r); e11i = _mm512_fmadd_pd(c6, ui, e11i);
    }
    _mm512_store_pd(Escr+96, e7r); _mm512_store_pd(Escr+104, e7i);
    _mm512_store_pd(Escr+112, e8r); _mm512_store_pd(Escr+120, e8i);
    _mm512_store_pd(Escr+128, e9r); _mm512_store_pd(Escr+136, e9i);
    _mm512_store_pd(Escr+144, e10r); _mm512_store_pd(Escr+152, e10i);
    _mm512_store_pd(Escr+160, e11r); _mm512_store_pd(Escr+168, e11i);
    }
    { __m512d s1 = _mm512_set1_pd(0x1.14459ad2be466p-2), s2 = _mm512_set1_pd(0x1.0a06e851db7cap-1), s3 = _mm512_set1_pd(0x1.763021aaa15d9p-1), s4 = _mm512_set1_pd(0x1.c698e42f47b09p-1), s5 = _mm512_set1_pd(0x1.f54a827142577p-1), s6 = _mm512_set1_pd(0x1.fece70dfd3efbp-1), s7 = _mm512_set1_pd(0x1.e270060999288p-1), s8 = _mm512_set1_pd(0x1.a249e0b897caap-1), s9 = _mm512_set1_pd(0x1.431df5838f7f1p-1), s10 = _mm512_set1_pd(0x1.97f6748e524b1p-2), s11 = _mm512_set1_pd(0x1.16de8a4564f1cp-3);
    __m512d o1r = _mm512_setzero_pd(), o1i = _mm512_setzero_pd();
    __m512d o2r = _mm512_setzero_pd(), o2i = _mm512_setzero_pd();
    __m512d o3r = _mm512_setzero_pd(), o3i = _mm512_setzero_pd();
    __m512d o4r = _mm512_setzero_pd(), o4i = _mm512_setzero_pd();
    __m512d o5r = _mm512_setzero_pd(), o5i = _mm512_setzero_pd();
    __m512d o6r = _mm512_setzero_pd(), o6i = _mm512_setzero_pd();
    { __m512d vr = _mm512_load_pd(AB+16);
      __m512d vi = _mm512_load_pd(AB+24);
      o1r = _mm512_fmadd_pd(s1, vr, o1r); o1i = _mm512_fmadd_pd(s1, vi, o1i);
      o2r = _mm512_fmadd_pd(s2, vr, o2r); o2i = _mm512_fmadd_pd(s2, vi, o2i);
      o3r = _mm512_fmadd_pd(s3, vr, o3r); o3i = _mm512_fmadd_pd(s3, vi, o3i);
      o4r = _mm512_fmadd_pd(s4, vr, o4r); o4i = _mm512_fmadd_pd(s4, vi, o4i);
      o5r = _mm512_fmadd_pd(s5, vr, o5r); o5i = _mm512_fmadd_pd(s5, vi, o5i);
      o6r = _mm512_fmadd_pd(s6, vr, o6r); o6i = _mm512_fmadd_pd(s6, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+48);
      __m512d vi = _mm512_load_pd(AB+56);
      o1r = _mm512_fmadd_pd(s2, vr, o1r); o1i = _mm512_fmadd_pd(s2, vi, o1i);
      o2r = _mm512_fmadd_pd(s4, vr, o2r); o2i = _mm512_fmadd_pd(s4, vi, o2i);
      o3r = _mm512_fmadd_pd(s6, vr, o3r); o3i = _mm512_fmadd_pd(s6, vi, o3i);
      o4r = _mm512_fmadd_pd(s8, vr, o4r); o4i = _mm512_fmadd_pd(s8, vi, o4i);
      o5r = _mm512_fmadd_pd(s10, vr, o5r); o5i = _mm512_fmadd_pd(s10, vi, o5i);
      o6r = _mm512_fnmadd_pd(s11, vr, o6r); o6i = _mm512_fnmadd_pd(s11, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+80);
      __m512d vi = _mm512_load_pd(AB+88);
      o1r = _mm512_fmadd_pd(s3, vr, o1r); o1i = _mm512_fmadd_pd(s3, vi, o1i);
      o2r = _mm512_fmadd_pd(s6, vr, o2r); o2i = _mm512_fmadd_pd(s6, vi, o2i);
      o3r = _mm512_fmadd_pd(s9, vr, o3r); o3i = _mm512_fmadd_pd(s9, vi, o3i);
      o4r = _mm512_fnmadd_pd(s11, vr, o4r); o4i = _mm512_fnmadd_pd(s11, vi, o4i);
      o5r = _mm512_fnmadd_pd(s8, vr, o5r); o5i = _mm512_fnmadd_pd(s8, vi, o5i);
      o6r = _mm512_fnmadd_pd(s5, vr, o6r); o6i = _mm512_fnmadd_pd(s5, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+112);
      __m512d vi = _mm512_load_pd(AB+120);
      o1r = _mm512_fmadd_pd(s4, vr, o1r); o1i = _mm512_fmadd_pd(s4, vi, o1i);
      o2r = _mm512_fmadd_pd(s8, vr, o2r); o2i = _mm512_fmadd_pd(s8, vi, o2i);
      o3r = _mm512_fnmadd_pd(s11, vr, o3r); o3i = _mm512_fnmadd_pd(s11, vi, o3i);
      o4r = _mm512_fnmadd_pd(s7, vr, o4r); o4i = _mm512_fnmadd_pd(s7, vi, o4i);
      o5r = _mm512_fnmadd_pd(s3, vr, o5r); o5i = _mm512_fnmadd_pd(s3, vi, o5i);
      o6r = _mm512_fmadd_pd(s1, vr, o6r); o6i = _mm512_fmadd_pd(s1, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+144);
      __m512d vi = _mm512_load_pd(AB+152);
      o1r = _mm512_fmadd_pd(s5, vr, o1r); o1i = _mm512_fmadd_pd(s5, vi, o1i);
      o2r = _mm512_fmadd_pd(s10, vr, o2r); o2i = _mm512_fmadd_pd(s10, vi, o2i);
      o3r = _mm512_fnmadd_pd(s8, vr, o3r); o3i = _mm512_fnmadd_pd(s8, vi, o3i);
      o4r = _mm512_fnmadd_pd(s3, vr, o4r); o4i = _mm512_fnmadd_pd(s3, vi, o4i);
      o5r = _mm512_fmadd_pd(s2, vr, o5r); o5i = _mm512_fmadd_pd(s2, vi, o5i);
      o6r = _mm512_fmadd_pd(s7, vr, o6r); o6i = _mm512_fmadd_pd(s7, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+176);
      __m512d vi = _mm512_load_pd(AB+184);
      o1r = _mm512_fmadd_pd(s6, vr, o1r); o1i = _mm512_fmadd_pd(s6, vi, o1i);
      o2r = _mm512_fnmadd_pd(s11, vr, o2r); o2i = _mm512_fnmadd_pd(s11, vi, o2i);
      o3r = _mm512_fnmadd_pd(s5, vr, o3r); o3i = _mm512_fnmadd_pd(s5, vi, o3i);
      o4r = _mm512_fmadd_pd(s1, vr, o4r); o4i = _mm512_fmadd_pd(s1, vi, o4i);
      o5r = _mm512_fmadd_pd(s7, vr, o5r); o5i = _mm512_fmadd_pd(s7, vi, o5i);
      o6r = _mm512_fnmadd_pd(s10, vr, o6r); o6i = _mm512_fnmadd_pd(s10, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+208);
      __m512d vi = _mm512_load_pd(AB+216);
      o1r = _mm512_fmadd_pd(s7, vr, o1r); o1i = _mm512_fmadd_pd(s7, vi, o1i);
      o2r = _mm512_fnmadd_pd(s9, vr, o2r); o2i = _mm512_fnmadd_pd(s9, vi, o2i);
      o3r = _mm512_fnmadd_pd(s2, vr, o3r); o3i = _mm512_fnmadd_pd(s2, vi, o3i);
      o4r = _mm512_fmadd_pd(s5, vr, o4r); o4i = _mm512_fmadd_pd(s5, vi, o4i);
      o5r = _mm512_fnmadd_pd(s11, vr, o5r); o5i = _mm512_fnmadd_pd(s11, vi, o5i);
      o6r = _mm512_fnmadd_pd(s4, vr, o6r); o6i = _mm512_fnmadd_pd(s4, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+240);
      __m512d vi = _mm512_load_pd(AB+248);
      o1r = _mm512_fmadd_pd(s8, vr, o1r); o1i = _mm512_fmadd_pd(s8, vi, o1i);
      o2r = _mm512_fnmadd_pd(s7, vr, o2r); o2i = _mm512_fnmadd_pd(s7, vi, o2i);
      o3r = _mm512_fmadd_pd(s1, vr, o3r); o3i = _mm512_fmadd_pd(s1, vi, o3i);
      o4r = _mm512_fmadd_pd(s9, vr, o4r); o4i = _mm512_fmadd_pd(s9, vi, o4i);
      o5r = _mm512_fnmadd_pd(s6, vr, o5r); o5i = _mm512_fnmadd_pd(s6, vi, o5i);
      o6r = _mm512_fmadd_pd(s2, vr, o6r); o6i = _mm512_fmadd_pd(s2, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+272);
      __m512d vi = _mm512_load_pd(AB+280);
      o1r = _mm512_fmadd_pd(s9, vr, o1r); o1i = _mm512_fmadd_pd(s9, vi, o1i);
      o2r = _mm512_fnmadd_pd(s5, vr, o2r); o2i = _mm512_fnmadd_pd(s5, vi, o2i);
      o3r = _mm512_fmadd_pd(s4, vr, o3r); o3i = _mm512_fmadd_pd(s4, vi, o3i);
      o4r = _mm512_fnmadd_pd(s10, vr, o4r); o4i = _mm512_fnmadd_pd(s10, vi, o4i);
      o5r = _mm512_fnmadd_pd(s1, vr, o5r); o5i = _mm512_fnmadd_pd(s1, vi, o5i);
      o6r = _mm512_fmadd_pd(s8, vr, o6r); o6i = _mm512_fmadd_pd(s8, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+304);
      __m512d vi = _mm512_load_pd(AB+312);
      o1r = _mm512_fmadd_pd(s10, vr, o1r); o1i = _mm512_fmadd_pd(s10, vi, o1i);
      o2r = _mm512_fnmadd_pd(s3, vr, o2r); o2i = _mm512_fnmadd_pd(s3, vi, o2i);
      o3r = _mm512_fmadd_pd(s7, vr, o3r); o3i = _mm512_fmadd_pd(s7, vi, o3i);
      o4r = _mm512_fnmadd_pd(s6, vr, o4r); o4i = _mm512_fnmadd_pd(s6, vi, o4i);
      o5r = _mm512_fmadd_pd(s4, vr, o5r); o5i = _mm512_fmadd_pd(s4, vi, o5i);
      o6r = _mm512_fnmadd_pd(s9, vr, o6r); o6i = _mm512_fnmadd_pd(s9, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+336);
      __m512d vi = _mm512_load_pd(AB+344);
      o1r = _mm512_fmadd_pd(s11, vr, o1r); o1i = _mm512_fmadd_pd(s11, vi, o1i);
      o2r = _mm512_fnmadd_pd(s1, vr, o2r); o2i = _mm512_fnmadd_pd(s1, vi, o2i);
      o3r = _mm512_fmadd_pd(s10, vr, o3r); o3i = _mm512_fmadd_pd(s10, vi, o3i);
      o4r = _mm512_fnmadd_pd(s2, vr, o4r); o4i = _mm512_fnmadd_pd(s2, vi, o4i);
      o5r = _mm512_fmadd_pd(s9, vr, o5r); o5i = _mm512_fmadd_pd(s9, vi, o5i);
      o6r = _mm512_fnmadd_pd(s3, vr, o6r); o6i = _mm512_fnmadd_pd(s3, vi, o6i);
    }
    { __m512d er = _mm512_load_pd(Escr+0), ei = _mm512_load_pd(Escr+8);
      _mm512_store_pd(X+1*s,   _mm512_add_pd(er, o1i));
      _mm512_store_pd(X+1*s+8, _mm512_sub_pd(ei, o1r));
      _mm512_store_pd(X+22*s,   _mm512_sub_pd(er, o1i));
      _mm512_store_pd(X+22*s+8, _mm512_add_pd(ei, o1r)); }
    { __m512d er = _mm512_load_pd(Escr+16), ei = _mm512_load_pd(Escr+24);
      _mm512_store_pd(X+2*s,   _mm512_add_pd(er, o2i));
      _mm512_store_pd(X+2*s+8, _mm512_sub_pd(ei, o2r));
      _mm512_store_pd(X+21*s,   _mm512_sub_pd(er, o2i));
      _mm512_store_pd(X+21*s+8, _mm512_add_pd(ei, o2r)); }
    { __m512d er = _mm512_load_pd(Escr+32), ei = _mm512_load_pd(Escr+40);
      _mm512_store_pd(X+3*s,   _mm512_add_pd(er, o3i));
      _mm512_store_pd(X+3*s+8, _mm512_sub_pd(ei, o3r));
      _mm512_store_pd(X+20*s,   _mm512_sub_pd(er, o3i));
      _mm512_store_pd(X+20*s+8, _mm512_add_pd(ei, o3r)); }
    { __m512d er = _mm512_load_pd(Escr+48), ei = _mm512_load_pd(Escr+56);
      _mm512_store_pd(X+4*s,   _mm512_add_pd(er, o4i));
      _mm512_store_pd(X+4*s+8, _mm512_sub_pd(ei, o4r));
      _mm512_store_pd(X+19*s,   _mm512_sub_pd(er, o4i));
      _mm512_store_pd(X+19*s+8, _mm512_add_pd(ei, o4r)); }
    { __m512d er = _mm512_load_pd(Escr+64), ei = _mm512_load_pd(Escr+72);
      _mm512_store_pd(X+5*s,   _mm512_add_pd(er, o5i));
      _mm512_store_pd(X+5*s+8, _mm512_sub_pd(ei, o5r));
      _mm512_store_pd(X+18*s,   _mm512_sub_pd(er, o5i));
      _mm512_store_pd(X+18*s+8, _mm512_add_pd(ei, o5r)); }
    { __m512d er = _mm512_load_pd(Escr+80), ei = _mm512_load_pd(Escr+88);
      _mm512_store_pd(X+6*s,   _mm512_add_pd(er, o6i));
      _mm512_store_pd(X+6*s+8, _mm512_sub_pd(ei, o6r));
      _mm512_store_pd(X+17*s,   _mm512_sub_pd(er, o6i));
      _mm512_store_pd(X+17*s+8, _mm512_add_pd(ei, o6r)); }
    }
    { __m512d s1 = _mm512_set1_pd(0x1.14459ad2be466p-2), s2 = _mm512_set1_pd(0x1.0a06e851db7cap-1), s3 = _mm512_set1_pd(0x1.763021aaa15d9p-1), s4 = _mm512_set1_pd(0x1.c698e42f47b09p-1), s5 = _mm512_set1_pd(0x1.f54a827142577p-1), s6 = _mm512_set1_pd(0x1.fece70dfd3efbp-1), s7 = _mm512_set1_pd(0x1.e270060999288p-1), s8 = _mm512_set1_pd(0x1.a249e0b897caap-1), s9 = _mm512_set1_pd(0x1.431df5838f7f1p-1), s10 = _mm512_set1_pd(0x1.97f6748e524b1p-2), s11 = _mm512_set1_pd(0x1.16de8a4564f1cp-3);
    __m512d o7r = _mm512_setzero_pd(), o7i = _mm512_setzero_pd();
    __m512d o8r = _mm512_setzero_pd(), o8i = _mm512_setzero_pd();
    __m512d o9r = _mm512_setzero_pd(), o9i = _mm512_setzero_pd();
    __m512d o10r = _mm512_setzero_pd(), o10i = _mm512_setzero_pd();
    __m512d o11r = _mm512_setzero_pd(), o11i = _mm512_setzero_pd();
    { __m512d vr = _mm512_load_pd(AB+16);
      __m512d vi = _mm512_load_pd(AB+24);
      o7r = _mm512_fmadd_pd(s7, vr, o7r); o7i = _mm512_fmadd_pd(s7, vi, o7i);
      o8r = _mm512_fmadd_pd(s8, vr, o8r); o8i = _mm512_fmadd_pd(s8, vi, o8i);
      o9r = _mm512_fmadd_pd(s9, vr, o9r); o9i = _mm512_fmadd_pd(s9, vi, o9i);
      o10r = _mm512_fmadd_pd(s10, vr, o10r); o10i = _mm512_fmadd_pd(s10, vi, o10i);
      o11r = _mm512_fmadd_pd(s11, vr, o11r); o11i = _mm512_fmadd_pd(s11, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+48);
      __m512d vi = _mm512_load_pd(AB+56);
      o7r = _mm512_fnmadd_pd(s9, vr, o7r); o7i = _mm512_fnmadd_pd(s9, vi, o7i);
      o8r = _mm512_fnmadd_pd(s7, vr, o8r); o8i = _mm512_fnmadd_pd(s7, vi, o8i);
      o9r = _mm512_fnmadd_pd(s5, vr, o9r); o9i = _mm512_fnmadd_pd(s5, vi, o9i);
      o10r = _mm512_fnmadd_pd(s3, vr, o10r); o10i = _mm512_fnmadd_pd(s3, vi, o10i);
      o11r = _mm512_fnmadd_pd(s1, vr, o11r); o11i = _mm512_fnmadd_pd(s1, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+80);
      __m512d vi = _mm512_load_pd(AB+88);
      o7r = _mm512_fnmadd_pd(s2, vr, o7r); o7i = _mm512_fnmadd_pd(s2, vi, o7i);
      o8r = _mm512_fmadd_pd(s1, vr, o8r); o8i = _mm512_fmadd_pd(s1, vi, o8i);
      o9r = _mm512_fmadd_pd(s4, vr, o9r); o9i = _mm512_fmadd_pd(s4, vi, o9i);
      o10r = _mm512_fmadd_pd(s7, vr, o10r); o10i = _mm512_fmadd_pd(s7, vi, o10i);
      o11r = _mm512_fmadd_pd(s10, vr, o11r); o11i = _mm512_fmadd_pd(s10, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+112);
      __m512d vi = _mm512_load_pd(AB+120);
      o7r = _mm512_fmadd_pd(s5, vr, o7r); o7i = _mm512_fmadd_pd(s5, vi, o7i);
      o8r = _mm512_fmadd_pd(s9, vr, o8r); o8i = _mm512_fmadd_pd(s9, vi, o8i);
      o9r = _mm512_fnmadd_pd(s10, vr, o9r); o9i = _mm512_fnmadd_pd(s10, vi, o9i);
      o10r = _mm512_fnmadd_pd(s6, vr, o10r); o10i = _mm512_fnmadd_pd(s6, vi, o10i);
      o11r = _mm512_fnmadd_pd(s2, vr, o11r); o11i = _mm512_fnmadd_pd(s2, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+144);
      __m512d vi = _mm512_load_pd(AB+152);
      o7r = _mm512_fnmadd_pd(s11, vr, o7r); o7i = _mm512_fnmadd_pd(s11, vi, o7i);
      o8r = _mm512_fnmadd_pd(s6, vr, o8r); o8i = _mm512_fnmadd_pd(s6, vi, o8i);
      o9r = _mm512_fnmadd_pd(s1, vr, o9r); o9i = _mm512_fnmadd_pd(s1, vi, o9i);
      o10r = _mm512_fmadd_pd(s4, vr, o10r); o10i = _mm512_fmadd_pd(s4, vi, o10i);
      o11r = _mm512_fmadd_pd(s9, vr, o11r); o11i = _mm512_fmadd_pd(s9, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+176);
      __m512d vi = _mm512_load_pd(AB+184);
      o7r = _mm512_fnmadd_pd(s4, vr, o7r); o7i = _mm512_fnmadd_pd(s4, vi, o7i);
      o8r = _mm512_fmadd_pd(s2, vr, o8r); o8i = _mm512_fmadd_pd(s2, vi, o8i);
      o9r = _mm512_fmadd_pd(s8, vr, o9r); o9i = _mm512_fmadd_pd(s8, vi, o9i);
      o10r = _mm512_fnmadd_pd(s9, vr, o10r); o10i = _mm512_fnmadd_pd(s9, vi, o10i);
      o11r = _mm512_fnmadd_pd(s3, vr, o11r); o11i = _mm512_fnmadd_pd(s3, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+208);
      __m512d vi = _mm512_load_pd(AB+216);
      o7r = _mm512_fmadd_pd(s3, vr, o7r); o7i = _mm512_fmadd_pd(s3, vi, o7i);
      o8r = _mm512_fmadd_pd(s10, vr, o8r); o8i = _mm512_fmadd_pd(s10, vi, o8i);
      o9r = _mm512_fnmadd_pd(s6, vr, o9r); o9i = _mm512_fnmadd_pd(s6, vi, o9i);
      o10r = _mm512_fmadd_pd(s1, vr, o10r); o10i = _mm512_fmadd_pd(s1, vi, o10i);
      o11r = _mm512_fmadd_pd(s8, vr, o11r); o11i = _mm512_fmadd_pd(s8, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+240);
      __m512d vi = _mm512_load_pd(AB+248);
      o7r = _mm512_fmadd_pd(s10, vr, o7r); o7i = _mm512_fmadd_pd(s10, vi, o7i);
      o8r = _mm512_fnmadd_pd(s5, vr, o8r); o8i = _mm512_fnmadd_pd(s5, vi, o8i);
      o9r = _mm512_fmadd_pd(s3, vr, o9r); o9i = _mm512_fmadd_pd(s3, vi, o9i);
      o10r = _mm512_fmadd_pd(s11, vr, o10r); o10i = _mm512_fmadd_pd(s11, vi, o10i);
      o11r = _mm512_fnmadd_pd(s4, vr, o11r); o11i = _mm512_fnmadd_pd(s4, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+272);
      __m512d vi = _mm512_load_pd(AB+280);
      o7r = _mm512_fnmadd_pd(s6, vr, o7r); o7i = _mm512_fnmadd_pd(s6, vi, o7i);
      o8r = _mm512_fmadd_pd(s3, vr, o8r); o8i = _mm512_fmadd_pd(s3, vi, o8i);
      o9r = _mm512_fnmadd_pd(s11, vr, o9r); o9i = _mm512_fnmadd_pd(s11, vi, o9i);
      o10r = _mm512_fnmadd_pd(s2, vr, o10r); o10i = _mm512_fnmadd_pd(s2, vi, o10i);
      o11r = _mm512_fmadd_pd(s7, vr, o11r); o11i = _mm512_fmadd_pd(s7, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+304);
      __m512d vi = _mm512_load_pd(AB+312);
      o7r = _mm512_fmadd_pd(s1, vr, o7r); o7i = _mm512_fmadd_pd(s1, vi, o7i);
      o8r = _mm512_fmadd_pd(s11, vr, o8r); o8i = _mm512_fmadd_pd(s11, vi, o8i);
      o9r = _mm512_fnmadd_pd(s2, vr, o9r); o9i = _mm512_fnmadd_pd(s2, vi, o9i);
      o10r = _mm512_fmadd_pd(s8, vr, o10r); o10i = _mm512_fmadd_pd(s8, vi, o10i);
      o11r = _mm512_fnmadd_pd(s5, vr, o11r); o11i = _mm512_fnmadd_pd(s5, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+336);
      __m512d vi = _mm512_load_pd(AB+344);
      o7r = _mm512_fmadd_pd(s8, vr, o7r); o7i = _mm512_fmadd_pd(s8, vi, o7i);
      o8r = _mm512_fnmadd_pd(s4, vr, o8r); o8i = _mm512_fnmadd_pd(s4, vi, o8i);
      o9r = _mm512_fmadd_pd(s7, vr, o9r); o9i = _mm512_fmadd_pd(s7, vi, o9i);
      o10r = _mm512_fnmadd_pd(s5, vr, o10r); o10i = _mm512_fnmadd_pd(s5, vi, o10i);
      o11r = _mm512_fmadd_pd(s6, vr, o11r); o11i = _mm512_fmadd_pd(s6, vi, o11i);
    }
    { __m512d er = _mm512_load_pd(Escr+96), ei = _mm512_load_pd(Escr+104);
      _mm512_store_pd(X+7*s,   _mm512_add_pd(er, o7i));
      _mm512_store_pd(X+7*s+8, _mm512_sub_pd(ei, o7r));
      _mm512_store_pd(X+16*s,   _mm512_sub_pd(er, o7i));
      _mm512_store_pd(X+16*s+8, _mm512_add_pd(ei, o7r)); }
    { __m512d er = _mm512_load_pd(Escr+112), ei = _mm512_load_pd(Escr+120);
      _mm512_store_pd(X+8*s,   _mm512_add_pd(er, o8i));
      _mm512_store_pd(X+8*s+8, _mm512_sub_pd(ei, o8r));
      _mm512_store_pd(X+15*s,   _mm512_sub_pd(er, o8i));
      _mm512_store_pd(X+15*s+8, _mm512_add_pd(ei, o8r)); }
    { __m512d er = _mm512_load_pd(Escr+128), ei = _mm512_load_pd(Escr+136);
      _mm512_store_pd(X+9*s,   _mm512_add_pd(er, o9i));
      _mm512_store_pd(X+9*s+8, _mm512_sub_pd(ei, o9r));
      _mm512_store_pd(X+14*s,   _mm512_sub_pd(er, o9i));
      _mm512_store_pd(X+14*s+8, _mm512_add_pd(ei, o9r)); }
    { __m512d er = _mm512_load_pd(Escr+144), ei = _mm512_load_pd(Escr+152);
      _mm512_store_pd(X+10*s,   _mm512_add_pd(er, o10i));
      _mm512_store_pd(X+10*s+8, _mm512_sub_pd(ei, o10r));
      _mm512_store_pd(X+13*s,   _mm512_sub_pd(er, o10i));
      _mm512_store_pd(X+13*s+8, _mm512_add_pd(ei, o10r)); }
    { __m512d er = _mm512_load_pd(Escr+160), ei = _mm512_load_pd(Escr+168);
      _mm512_store_pd(X+11*s,   _mm512_add_pd(er, o11i));
      _mm512_store_pd(X+11*s+8, _mm512_sub_pd(ei, o11r));
      _mm512_store_pd(X+12*s,   _mm512_sub_pd(er, o11i));
      _mm512_store_pd(X+12*s+8, _mm512_add_pd(ei, o11r)); }
    }
}
static __attribute__((always_inline)) inline void dft23m(double* restrict X, long es, int dopf, const double* restrict C, long ces){
    const long s = es*16;
    const long cs = ces*16;
    double AB[88*8] ALIGN64;
    double Escr[22*8] ALIGN64;
    __m512d x0r = _mm512_load_pd(X);
    if(dopf){ _mm_prefetch((const char*)(X+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+376), _MM_HINT_T0); }
    __m512d x0i = _mm512_load_pd(X+8);
    { __m512d c1 = _mm512_set1_pd(0x1.ed037ea3d2dbcp-1), c2 = _mm512_set1_pd(0x1.b57675cf309eep-1), c3 = _mm512_set1_pd(0x1.5d779b07cfef7p-1), c4 = _mm512_set1_pd(0x1.d71b4a0c5a6c9p-2), c5 = _mm512_set1_pd(0x1.a0ad8bd1e2881p-3), c6 = _mm512_set1_pd(-0x1.17855b599f3b2p-4), c7 = _mm512_set1_pd(-0x1.56eaae597c776p-2), c8 = _mm512_set1_pd(-0x1.2742a4a775cfap-1), c9 = _mm512_set1_pd(-0x1.8d2a07c16d46ep-1), c10 = _mm512_set1_pd(-0x1.d59cb83ef99bcp-1), c11 = _mm512_set1_pd(-0x1.fb3b3035aa6ccp-1);
    __m512d e1r = x0r, e1i = x0i;
    __m512d e2r = x0r, e2i = x0i;
    __m512d e3r = x0r, e3i = x0i;
    __m512d e4r = x0r, e4i = x0i;
    __m512d e5r = x0r, e5i = x0i;
    __m512d e6r = x0r, e6i = x0i;
    __m512d sr = x0r, si = x0i;
    { __m512d pr = _mm512_load_pd(X+1*s), qr = _mm512_load_pd(X+22*s);
      if(dopf){ _mm_prefetch((const char*)(X+1*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+1*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+22*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+22*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+1*s+8), qi = _mm512_load_pd(X+22*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+0, ur);    _mm512_store_pd(AB+8, ui);
      _mm512_store_pd(AB+16, vr); _mm512_store_pd(AB+24, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c1, ur, e1r); e1i = _mm512_fmadd_pd(c1, ui, e1i);
      e2r = _mm512_fmadd_pd(c2, ur, e2r); e2i = _mm512_fmadd_pd(c2, ui, e2i);
      e3r = _mm512_fmadd_pd(c3, ur, e3r); e3i = _mm512_fmadd_pd(c3, ui, e3i);
      e4r = _mm512_fmadd_pd(c4, ur, e4r); e4i = _mm512_fmadd_pd(c4, ui, e4i);
      e5r = _mm512_fmadd_pd(c5, ur, e5r); e5i = _mm512_fmadd_pd(c5, ui, e5i);
      e6r = _mm512_fmadd_pd(c6, ur, e6r); e6i = _mm512_fmadd_pd(c6, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+2*s), qr = _mm512_load_pd(X+21*s);
      if(dopf){ _mm_prefetch((const char*)(X+2*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+2*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+21*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+21*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+2*s+8), qi = _mm512_load_pd(X+21*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+32, ur);    _mm512_store_pd(AB+40, ui);
      _mm512_store_pd(AB+48, vr); _mm512_store_pd(AB+56, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c2, ur, e1r); e1i = _mm512_fmadd_pd(c2, ui, e1i);
      e2r = _mm512_fmadd_pd(c4, ur, e2r); e2i = _mm512_fmadd_pd(c4, ui, e2i);
      e3r = _mm512_fmadd_pd(c6, ur, e3r); e3i = _mm512_fmadd_pd(c6, ui, e3i);
      e4r = _mm512_fmadd_pd(c8, ur, e4r); e4i = _mm512_fmadd_pd(c8, ui, e4i);
      e5r = _mm512_fmadd_pd(c10, ur, e5r); e5i = _mm512_fmadd_pd(c10, ui, e5i);
      e6r = _mm512_fmadd_pd(c11, ur, e6r); e6i = _mm512_fmadd_pd(c11, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+3*s), qr = _mm512_load_pd(X+20*s);
      if(dopf){ _mm_prefetch((const char*)(X+3*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+3*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+20*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+20*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+3*s+8), qi = _mm512_load_pd(X+20*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+64, ur);    _mm512_store_pd(AB+72, ui);
      _mm512_store_pd(AB+80, vr); _mm512_store_pd(AB+88, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c3, ur, e1r); e1i = _mm512_fmadd_pd(c3, ui, e1i);
      e2r = _mm512_fmadd_pd(c6, ur, e2r); e2i = _mm512_fmadd_pd(c6, ui, e2i);
      e3r = _mm512_fmadd_pd(c9, ur, e3r); e3i = _mm512_fmadd_pd(c9, ui, e3i);
      e4r = _mm512_fmadd_pd(c11, ur, e4r); e4i = _mm512_fmadd_pd(c11, ui, e4i);
      e5r = _mm512_fmadd_pd(c8, ur, e5r); e5i = _mm512_fmadd_pd(c8, ui, e5i);
      e6r = _mm512_fmadd_pd(c5, ur, e6r); e6i = _mm512_fmadd_pd(c5, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+4*s), qr = _mm512_load_pd(X+19*s);
      if(dopf){ _mm_prefetch((const char*)(X+4*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+4*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+19*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+19*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+4*s+8), qi = _mm512_load_pd(X+19*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+96, ur);    _mm512_store_pd(AB+104, ui);
      _mm512_store_pd(AB+112, vr); _mm512_store_pd(AB+120, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c4, ur, e1r); e1i = _mm512_fmadd_pd(c4, ui, e1i);
      e2r = _mm512_fmadd_pd(c8, ur, e2r); e2i = _mm512_fmadd_pd(c8, ui, e2i);
      e3r = _mm512_fmadd_pd(c11, ur, e3r); e3i = _mm512_fmadd_pd(c11, ui, e3i);
      e4r = _mm512_fmadd_pd(c7, ur, e4r); e4i = _mm512_fmadd_pd(c7, ui, e4i);
      e5r = _mm512_fmadd_pd(c3, ur, e5r); e5i = _mm512_fmadd_pd(c3, ui, e5i);
      e6r = _mm512_fmadd_pd(c1, ur, e6r); e6i = _mm512_fmadd_pd(c1, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+5*s), qr = _mm512_load_pd(X+18*s);
      if(dopf){ _mm_prefetch((const char*)(X+5*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+5*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+18*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+18*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+5*s+8), qi = _mm512_load_pd(X+18*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+128, ur);    _mm512_store_pd(AB+136, ui);
      _mm512_store_pd(AB+144, vr); _mm512_store_pd(AB+152, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c5, ur, e1r); e1i = _mm512_fmadd_pd(c5, ui, e1i);
      e2r = _mm512_fmadd_pd(c10, ur, e2r); e2i = _mm512_fmadd_pd(c10, ui, e2i);
      e3r = _mm512_fmadd_pd(c8, ur, e3r); e3i = _mm512_fmadd_pd(c8, ui, e3i);
      e4r = _mm512_fmadd_pd(c3, ur, e4r); e4i = _mm512_fmadd_pd(c3, ui, e4i);
      e5r = _mm512_fmadd_pd(c2, ur, e5r); e5i = _mm512_fmadd_pd(c2, ui, e5i);
      e6r = _mm512_fmadd_pd(c7, ur, e6r); e6i = _mm512_fmadd_pd(c7, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+6*s), qr = _mm512_load_pd(X+17*s);
      if(dopf){ _mm_prefetch((const char*)(X+6*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+6*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+17*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+17*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+6*s+8), qi = _mm512_load_pd(X+17*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+160, ur);    _mm512_store_pd(AB+168, ui);
      _mm512_store_pd(AB+176, vr); _mm512_store_pd(AB+184, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c6, ur, e1r); e1i = _mm512_fmadd_pd(c6, ui, e1i);
      e2r = _mm512_fmadd_pd(c11, ur, e2r); e2i = _mm512_fmadd_pd(c11, ui, e2i);
      e3r = _mm512_fmadd_pd(c5, ur, e3r); e3i = _mm512_fmadd_pd(c5, ui, e3i);
      e4r = _mm512_fmadd_pd(c1, ur, e4r); e4i = _mm512_fmadd_pd(c1, ui, e4i);
      e5r = _mm512_fmadd_pd(c7, ur, e5r); e5i = _mm512_fmadd_pd(c7, ui, e5i);
      e6r = _mm512_fmadd_pd(c10, ur, e6r); e6i = _mm512_fmadd_pd(c10, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+7*s), qr = _mm512_load_pd(X+16*s);
      if(dopf){ _mm_prefetch((const char*)(X+7*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+7*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+16*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+16*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+7*s+8), qi = _mm512_load_pd(X+16*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+192, ur);    _mm512_store_pd(AB+200, ui);
      _mm512_store_pd(AB+208, vr); _mm512_store_pd(AB+216, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c7, ur, e1r); e1i = _mm512_fmadd_pd(c7, ui, e1i);
      e2r = _mm512_fmadd_pd(c9, ur, e2r); e2i = _mm512_fmadd_pd(c9, ui, e2i);
      e3r = _mm512_fmadd_pd(c2, ur, e3r); e3i = _mm512_fmadd_pd(c2, ui, e3i);
      e4r = _mm512_fmadd_pd(c5, ur, e4r); e4i = _mm512_fmadd_pd(c5, ui, e4i);
      e5r = _mm512_fmadd_pd(c11, ur, e5r); e5i = _mm512_fmadd_pd(c11, ui, e5i);
      e6r = _mm512_fmadd_pd(c4, ur, e6r); e6i = _mm512_fmadd_pd(c4, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+8*s), qr = _mm512_load_pd(X+15*s);
      if(dopf){ _mm_prefetch((const char*)(X+8*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+8*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+15*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+15*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+8*s+8), qi = _mm512_load_pd(X+15*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+224, ur);    _mm512_store_pd(AB+232, ui);
      _mm512_store_pd(AB+240, vr); _mm512_store_pd(AB+248, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c8, ur, e1r); e1i = _mm512_fmadd_pd(c8, ui, e1i);
      e2r = _mm512_fmadd_pd(c7, ur, e2r); e2i = _mm512_fmadd_pd(c7, ui, e2i);
      e3r = _mm512_fmadd_pd(c1, ur, e3r); e3i = _mm512_fmadd_pd(c1, ui, e3i);
      e4r = _mm512_fmadd_pd(c9, ur, e4r); e4i = _mm512_fmadd_pd(c9, ui, e4i);
      e5r = _mm512_fmadd_pd(c6, ur, e5r); e5i = _mm512_fmadd_pd(c6, ui, e5i);
      e6r = _mm512_fmadd_pd(c2, ur, e6r); e6i = _mm512_fmadd_pd(c2, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+9*s), qr = _mm512_load_pd(X+14*s);
      if(dopf){ _mm_prefetch((const char*)(X+9*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+9*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+14*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+14*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+9*s+8), qi = _mm512_load_pd(X+14*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+256, ur);    _mm512_store_pd(AB+264, ui);
      _mm512_store_pd(AB+272, vr); _mm512_store_pd(AB+280, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c9, ur, e1r); e1i = _mm512_fmadd_pd(c9, ui, e1i);
      e2r = _mm512_fmadd_pd(c5, ur, e2r); e2i = _mm512_fmadd_pd(c5, ui, e2i);
      e3r = _mm512_fmadd_pd(c4, ur, e3r); e3i = _mm512_fmadd_pd(c4, ui, e3i);
      e4r = _mm512_fmadd_pd(c10, ur, e4r); e4i = _mm512_fmadd_pd(c10, ui, e4i);
      e5r = _mm512_fmadd_pd(c1, ur, e5r); e5i = _mm512_fmadd_pd(c1, ui, e5i);
      e6r = _mm512_fmadd_pd(c8, ur, e6r); e6i = _mm512_fmadd_pd(c8, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+10*s), qr = _mm512_load_pd(X+13*s);
      if(dopf){ _mm_prefetch((const char*)(X+10*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+10*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+13*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+13*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+10*s+8), qi = _mm512_load_pd(X+13*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+288, ur);    _mm512_store_pd(AB+296, ui);
      _mm512_store_pd(AB+304, vr); _mm512_store_pd(AB+312, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c10, ur, e1r); e1i = _mm512_fmadd_pd(c10, ui, e1i);
      e2r = _mm512_fmadd_pd(c3, ur, e2r); e2i = _mm512_fmadd_pd(c3, ui, e2i);
      e3r = _mm512_fmadd_pd(c7, ur, e3r); e3i = _mm512_fmadd_pd(c7, ui, e3i);
      e4r = _mm512_fmadd_pd(c6, ur, e4r); e4i = _mm512_fmadd_pd(c6, ui, e4i);
      e5r = _mm512_fmadd_pd(c4, ur, e5r); e5i = _mm512_fmadd_pd(c4, ui, e5i);
      e6r = _mm512_fmadd_pd(c9, ur, e6r); e6i = _mm512_fmadd_pd(c9, ui, e6i);
    }
    { __m512d pr = _mm512_load_pd(X+11*s), qr = _mm512_load_pd(X+12*s);
      if(dopf){ _mm_prefetch((const char*)(X+11*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+11*s+368+8), _MM_HINT_T0);
                _mm_prefetch((const char*)(X+12*s+368), _MM_HINT_T0); _mm_prefetch((const char*)(X+12*s+368+8), _MM_HINT_T0); }
      __m512d pi = _mm512_load_pd(X+11*s+8), qi = _mm512_load_pd(X+12*s+8);
      __m512d ur = _mm512_add_pd(pr,qr), ui = _mm512_add_pd(pi,qi);
      __m512d vr = _mm512_sub_pd(pr,qr), vi = _mm512_sub_pd(pi,qi);
      _mm512_store_pd(AB+320, ur);    _mm512_store_pd(AB+328, ui);
      _mm512_store_pd(AB+336, vr); _mm512_store_pd(AB+344, vi);
      sr = _mm512_add_pd(sr, ur); si = _mm512_add_pd(si, ui);
      e1r = _mm512_fmadd_pd(c11, ur, e1r); e1i = _mm512_fmadd_pd(c11, ui, e1i);
      e2r = _mm512_fmadd_pd(c1, ur, e2r); e2i = _mm512_fmadd_pd(c1, ui, e2i);
      e3r = _mm512_fmadd_pd(c10, ur, e3r); e3i = _mm512_fmadd_pd(c10, ui, e3i);
      e4r = _mm512_fmadd_pd(c2, ur, e4r); e4i = _mm512_fmadd_pd(c2, ui, e4i);
      e5r = _mm512_fmadd_pd(c9, ur, e5r); e5i = _mm512_fmadd_pd(c9, ui, e5i);
      e6r = _mm512_fmadd_pd(c3, ur, e6r); e6i = _mm512_fmadd_pd(c3, ui, e6i);
    }
    { __m512d zr = _mm512_add_pd(sr, _mm512_load_pd(C)), zi = _mm512_add_pd(si, _mm512_load_pd(C+8));
      _mm512_store_pd(X, zr); _mm512_store_pd(X+8, zi); }
    _mm512_store_pd(Escr+0, e1r); _mm512_store_pd(Escr+8, e1i);
    _mm512_store_pd(Escr+16, e2r); _mm512_store_pd(Escr+24, e2i);
    _mm512_store_pd(Escr+32, e3r); _mm512_store_pd(Escr+40, e3i);
    _mm512_store_pd(Escr+48, e4r); _mm512_store_pd(Escr+56, e4i);
    _mm512_store_pd(Escr+64, e5r); _mm512_store_pd(Escr+72, e5i);
    _mm512_store_pd(Escr+80, e6r); _mm512_store_pd(Escr+88, e6i);
    }
    { __m512d c1 = _mm512_set1_pd(0x1.ed037ea3d2dbcp-1), c2 = _mm512_set1_pd(0x1.b57675cf309eep-1), c3 = _mm512_set1_pd(0x1.5d779b07cfef7p-1), c4 = _mm512_set1_pd(0x1.d71b4a0c5a6c9p-2), c5 = _mm512_set1_pd(0x1.a0ad8bd1e2881p-3), c6 = _mm512_set1_pd(-0x1.17855b599f3b2p-4), c7 = _mm512_set1_pd(-0x1.56eaae597c776p-2), c8 = _mm512_set1_pd(-0x1.2742a4a775cfap-1), c9 = _mm512_set1_pd(-0x1.8d2a07c16d46ep-1), c10 = _mm512_set1_pd(-0x1.d59cb83ef99bcp-1), c11 = _mm512_set1_pd(-0x1.fb3b3035aa6ccp-1);
    __m512d e7r = x0r, e7i = x0i;
    __m512d e8r = x0r, e8i = x0i;
    __m512d e9r = x0r, e9i = x0i;
    __m512d e10r = x0r, e10i = x0i;
    __m512d e11r = x0r, e11i = x0i;
    { __m512d ur = _mm512_load_pd(AB+0);
      __m512d ui = _mm512_load_pd(AB+8);
      e7r = _mm512_fmadd_pd(c7, ur, e7r); e7i = _mm512_fmadd_pd(c7, ui, e7i);
      e8r = _mm512_fmadd_pd(c8, ur, e8r); e8i = _mm512_fmadd_pd(c8, ui, e8i);
      e9r = _mm512_fmadd_pd(c9, ur, e9r); e9i = _mm512_fmadd_pd(c9, ui, e9i);
      e10r = _mm512_fmadd_pd(c10, ur, e10r); e10i = _mm512_fmadd_pd(c10, ui, e10i);
      e11r = _mm512_fmadd_pd(c11, ur, e11r); e11i = _mm512_fmadd_pd(c11, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+32);
      __m512d ui = _mm512_load_pd(AB+40);
      e7r = _mm512_fmadd_pd(c9, ur, e7r); e7i = _mm512_fmadd_pd(c9, ui, e7i);
      e8r = _mm512_fmadd_pd(c7, ur, e8r); e8i = _mm512_fmadd_pd(c7, ui, e8i);
      e9r = _mm512_fmadd_pd(c5, ur, e9r); e9i = _mm512_fmadd_pd(c5, ui, e9i);
      e10r = _mm512_fmadd_pd(c3, ur, e10r); e10i = _mm512_fmadd_pd(c3, ui, e10i);
      e11r = _mm512_fmadd_pd(c1, ur, e11r); e11i = _mm512_fmadd_pd(c1, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+64);
      __m512d ui = _mm512_load_pd(AB+72);
      e7r = _mm512_fmadd_pd(c2, ur, e7r); e7i = _mm512_fmadd_pd(c2, ui, e7i);
      e8r = _mm512_fmadd_pd(c1, ur, e8r); e8i = _mm512_fmadd_pd(c1, ui, e8i);
      e9r = _mm512_fmadd_pd(c4, ur, e9r); e9i = _mm512_fmadd_pd(c4, ui, e9i);
      e10r = _mm512_fmadd_pd(c7, ur, e10r); e10i = _mm512_fmadd_pd(c7, ui, e10i);
      e11r = _mm512_fmadd_pd(c10, ur, e11r); e11i = _mm512_fmadd_pd(c10, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+96);
      __m512d ui = _mm512_load_pd(AB+104);
      e7r = _mm512_fmadd_pd(c5, ur, e7r); e7i = _mm512_fmadd_pd(c5, ui, e7i);
      e8r = _mm512_fmadd_pd(c9, ur, e8r); e8i = _mm512_fmadd_pd(c9, ui, e8i);
      e9r = _mm512_fmadd_pd(c10, ur, e9r); e9i = _mm512_fmadd_pd(c10, ui, e9i);
      e10r = _mm512_fmadd_pd(c6, ur, e10r); e10i = _mm512_fmadd_pd(c6, ui, e10i);
      e11r = _mm512_fmadd_pd(c2, ur, e11r); e11i = _mm512_fmadd_pd(c2, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+128);
      __m512d ui = _mm512_load_pd(AB+136);
      e7r = _mm512_fmadd_pd(c11, ur, e7r); e7i = _mm512_fmadd_pd(c11, ui, e7i);
      e8r = _mm512_fmadd_pd(c6, ur, e8r); e8i = _mm512_fmadd_pd(c6, ui, e8i);
      e9r = _mm512_fmadd_pd(c1, ur, e9r); e9i = _mm512_fmadd_pd(c1, ui, e9i);
      e10r = _mm512_fmadd_pd(c4, ur, e10r); e10i = _mm512_fmadd_pd(c4, ui, e10i);
      e11r = _mm512_fmadd_pd(c9, ur, e11r); e11i = _mm512_fmadd_pd(c9, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+160);
      __m512d ui = _mm512_load_pd(AB+168);
      e7r = _mm512_fmadd_pd(c4, ur, e7r); e7i = _mm512_fmadd_pd(c4, ui, e7i);
      e8r = _mm512_fmadd_pd(c2, ur, e8r); e8i = _mm512_fmadd_pd(c2, ui, e8i);
      e9r = _mm512_fmadd_pd(c8, ur, e9r); e9i = _mm512_fmadd_pd(c8, ui, e9i);
      e10r = _mm512_fmadd_pd(c9, ur, e10r); e10i = _mm512_fmadd_pd(c9, ui, e10i);
      e11r = _mm512_fmadd_pd(c3, ur, e11r); e11i = _mm512_fmadd_pd(c3, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+192);
      __m512d ui = _mm512_load_pd(AB+200);
      e7r = _mm512_fmadd_pd(c3, ur, e7r); e7i = _mm512_fmadd_pd(c3, ui, e7i);
      e8r = _mm512_fmadd_pd(c10, ur, e8r); e8i = _mm512_fmadd_pd(c10, ui, e8i);
      e9r = _mm512_fmadd_pd(c6, ur, e9r); e9i = _mm512_fmadd_pd(c6, ui, e9i);
      e10r = _mm512_fmadd_pd(c1, ur, e10r); e10i = _mm512_fmadd_pd(c1, ui, e10i);
      e11r = _mm512_fmadd_pd(c8, ur, e11r); e11i = _mm512_fmadd_pd(c8, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+224);
      __m512d ui = _mm512_load_pd(AB+232);
      e7r = _mm512_fmadd_pd(c10, ur, e7r); e7i = _mm512_fmadd_pd(c10, ui, e7i);
      e8r = _mm512_fmadd_pd(c5, ur, e8r); e8i = _mm512_fmadd_pd(c5, ui, e8i);
      e9r = _mm512_fmadd_pd(c3, ur, e9r); e9i = _mm512_fmadd_pd(c3, ui, e9i);
      e10r = _mm512_fmadd_pd(c11, ur, e10r); e10i = _mm512_fmadd_pd(c11, ui, e10i);
      e11r = _mm512_fmadd_pd(c4, ur, e11r); e11i = _mm512_fmadd_pd(c4, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+256);
      __m512d ui = _mm512_load_pd(AB+264);
      e7r = _mm512_fmadd_pd(c6, ur, e7r); e7i = _mm512_fmadd_pd(c6, ui, e7i);
      e8r = _mm512_fmadd_pd(c3, ur, e8r); e8i = _mm512_fmadd_pd(c3, ui, e8i);
      e9r = _mm512_fmadd_pd(c11, ur, e9r); e9i = _mm512_fmadd_pd(c11, ui, e9i);
      e10r = _mm512_fmadd_pd(c2, ur, e10r); e10i = _mm512_fmadd_pd(c2, ui, e10i);
      e11r = _mm512_fmadd_pd(c7, ur, e11r); e11i = _mm512_fmadd_pd(c7, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+288);
      __m512d ui = _mm512_load_pd(AB+296);
      e7r = _mm512_fmadd_pd(c1, ur, e7r); e7i = _mm512_fmadd_pd(c1, ui, e7i);
      e8r = _mm512_fmadd_pd(c11, ur, e8r); e8i = _mm512_fmadd_pd(c11, ui, e8i);
      e9r = _mm512_fmadd_pd(c2, ur, e9r); e9i = _mm512_fmadd_pd(c2, ui, e9i);
      e10r = _mm512_fmadd_pd(c8, ur, e10r); e10i = _mm512_fmadd_pd(c8, ui, e10i);
      e11r = _mm512_fmadd_pd(c5, ur, e11r); e11i = _mm512_fmadd_pd(c5, ui, e11i);
    }
    { __m512d ur = _mm512_load_pd(AB+320);
      __m512d ui = _mm512_load_pd(AB+328);
      e7r = _mm512_fmadd_pd(c8, ur, e7r); e7i = _mm512_fmadd_pd(c8, ui, e7i);
      e8r = _mm512_fmadd_pd(c4, ur, e8r); e8i = _mm512_fmadd_pd(c4, ui, e8i);
      e9r = _mm512_fmadd_pd(c7, ur, e9r); e9i = _mm512_fmadd_pd(c7, ui, e9i);
      e10r = _mm512_fmadd_pd(c5, ur, e10r); e10i = _mm512_fmadd_pd(c5, ui, e10i);
      e11r = _mm512_fmadd_pd(c6, ur, e11r); e11i = _mm512_fmadd_pd(c6, ui, e11i);
    }
    _mm512_store_pd(Escr+96, e7r); _mm512_store_pd(Escr+104, e7i);
    _mm512_store_pd(Escr+112, e8r); _mm512_store_pd(Escr+120, e8i);
    _mm512_store_pd(Escr+128, e9r); _mm512_store_pd(Escr+136, e9i);
    _mm512_store_pd(Escr+144, e10r); _mm512_store_pd(Escr+152, e10i);
    _mm512_store_pd(Escr+160, e11r); _mm512_store_pd(Escr+168, e11i);
    }
    { __m512d s1 = _mm512_set1_pd(0x1.14459ad2be466p-2), s2 = _mm512_set1_pd(0x1.0a06e851db7cap-1), s3 = _mm512_set1_pd(0x1.763021aaa15d9p-1), s4 = _mm512_set1_pd(0x1.c698e42f47b09p-1), s5 = _mm512_set1_pd(0x1.f54a827142577p-1), s6 = _mm512_set1_pd(0x1.fece70dfd3efbp-1), s7 = _mm512_set1_pd(0x1.e270060999288p-1), s8 = _mm512_set1_pd(0x1.a249e0b897caap-1), s9 = _mm512_set1_pd(0x1.431df5838f7f1p-1), s10 = _mm512_set1_pd(0x1.97f6748e524b1p-2), s11 = _mm512_set1_pd(0x1.16de8a4564f1cp-3);
    __m512d o1r = _mm512_setzero_pd(), o1i = _mm512_setzero_pd();
    __m512d o2r = _mm512_setzero_pd(), o2i = _mm512_setzero_pd();
    __m512d o3r = _mm512_setzero_pd(), o3i = _mm512_setzero_pd();
    __m512d o4r = _mm512_setzero_pd(), o4i = _mm512_setzero_pd();
    __m512d o5r = _mm512_setzero_pd(), o5i = _mm512_setzero_pd();
    __m512d o6r = _mm512_setzero_pd(), o6i = _mm512_setzero_pd();
    { __m512d vr = _mm512_load_pd(AB+16);
      __m512d vi = _mm512_load_pd(AB+24);
      o1r = _mm512_fmadd_pd(s1, vr, o1r); o1i = _mm512_fmadd_pd(s1, vi, o1i);
      o2r = _mm512_fmadd_pd(s2, vr, o2r); o2i = _mm512_fmadd_pd(s2, vi, o2i);
      o3r = _mm512_fmadd_pd(s3, vr, o3r); o3i = _mm512_fmadd_pd(s3, vi, o3i);
      o4r = _mm512_fmadd_pd(s4, vr, o4r); o4i = _mm512_fmadd_pd(s4, vi, o4i);
      o5r = _mm512_fmadd_pd(s5, vr, o5r); o5i = _mm512_fmadd_pd(s5, vi, o5i);
      o6r = _mm512_fmadd_pd(s6, vr, o6r); o6i = _mm512_fmadd_pd(s6, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+48);
      __m512d vi = _mm512_load_pd(AB+56);
      o1r = _mm512_fmadd_pd(s2, vr, o1r); o1i = _mm512_fmadd_pd(s2, vi, o1i);
      o2r = _mm512_fmadd_pd(s4, vr, o2r); o2i = _mm512_fmadd_pd(s4, vi, o2i);
      o3r = _mm512_fmadd_pd(s6, vr, o3r); o3i = _mm512_fmadd_pd(s6, vi, o3i);
      o4r = _mm512_fmadd_pd(s8, vr, o4r); o4i = _mm512_fmadd_pd(s8, vi, o4i);
      o5r = _mm512_fmadd_pd(s10, vr, o5r); o5i = _mm512_fmadd_pd(s10, vi, o5i);
      o6r = _mm512_fnmadd_pd(s11, vr, o6r); o6i = _mm512_fnmadd_pd(s11, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+80);
      __m512d vi = _mm512_load_pd(AB+88);
      o1r = _mm512_fmadd_pd(s3, vr, o1r); o1i = _mm512_fmadd_pd(s3, vi, o1i);
      o2r = _mm512_fmadd_pd(s6, vr, o2r); o2i = _mm512_fmadd_pd(s6, vi, o2i);
      o3r = _mm512_fmadd_pd(s9, vr, o3r); o3i = _mm512_fmadd_pd(s9, vi, o3i);
      o4r = _mm512_fnmadd_pd(s11, vr, o4r); o4i = _mm512_fnmadd_pd(s11, vi, o4i);
      o5r = _mm512_fnmadd_pd(s8, vr, o5r); o5i = _mm512_fnmadd_pd(s8, vi, o5i);
      o6r = _mm512_fnmadd_pd(s5, vr, o6r); o6i = _mm512_fnmadd_pd(s5, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+112);
      __m512d vi = _mm512_load_pd(AB+120);
      o1r = _mm512_fmadd_pd(s4, vr, o1r); o1i = _mm512_fmadd_pd(s4, vi, o1i);
      o2r = _mm512_fmadd_pd(s8, vr, o2r); o2i = _mm512_fmadd_pd(s8, vi, o2i);
      o3r = _mm512_fnmadd_pd(s11, vr, o3r); o3i = _mm512_fnmadd_pd(s11, vi, o3i);
      o4r = _mm512_fnmadd_pd(s7, vr, o4r); o4i = _mm512_fnmadd_pd(s7, vi, o4i);
      o5r = _mm512_fnmadd_pd(s3, vr, o5r); o5i = _mm512_fnmadd_pd(s3, vi, o5i);
      o6r = _mm512_fmadd_pd(s1, vr, o6r); o6i = _mm512_fmadd_pd(s1, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+144);
      __m512d vi = _mm512_load_pd(AB+152);
      o1r = _mm512_fmadd_pd(s5, vr, o1r); o1i = _mm512_fmadd_pd(s5, vi, o1i);
      o2r = _mm512_fmadd_pd(s10, vr, o2r); o2i = _mm512_fmadd_pd(s10, vi, o2i);
      o3r = _mm512_fnmadd_pd(s8, vr, o3r); o3i = _mm512_fnmadd_pd(s8, vi, o3i);
      o4r = _mm512_fnmadd_pd(s3, vr, o4r); o4i = _mm512_fnmadd_pd(s3, vi, o4i);
      o5r = _mm512_fmadd_pd(s2, vr, o5r); o5i = _mm512_fmadd_pd(s2, vi, o5i);
      o6r = _mm512_fmadd_pd(s7, vr, o6r); o6i = _mm512_fmadd_pd(s7, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+176);
      __m512d vi = _mm512_load_pd(AB+184);
      o1r = _mm512_fmadd_pd(s6, vr, o1r); o1i = _mm512_fmadd_pd(s6, vi, o1i);
      o2r = _mm512_fnmadd_pd(s11, vr, o2r); o2i = _mm512_fnmadd_pd(s11, vi, o2i);
      o3r = _mm512_fnmadd_pd(s5, vr, o3r); o3i = _mm512_fnmadd_pd(s5, vi, o3i);
      o4r = _mm512_fmadd_pd(s1, vr, o4r); o4i = _mm512_fmadd_pd(s1, vi, o4i);
      o5r = _mm512_fmadd_pd(s7, vr, o5r); o5i = _mm512_fmadd_pd(s7, vi, o5i);
      o6r = _mm512_fnmadd_pd(s10, vr, o6r); o6i = _mm512_fnmadd_pd(s10, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+208);
      __m512d vi = _mm512_load_pd(AB+216);
      o1r = _mm512_fmadd_pd(s7, vr, o1r); o1i = _mm512_fmadd_pd(s7, vi, o1i);
      o2r = _mm512_fnmadd_pd(s9, vr, o2r); o2i = _mm512_fnmadd_pd(s9, vi, o2i);
      o3r = _mm512_fnmadd_pd(s2, vr, o3r); o3i = _mm512_fnmadd_pd(s2, vi, o3i);
      o4r = _mm512_fmadd_pd(s5, vr, o4r); o4i = _mm512_fmadd_pd(s5, vi, o4i);
      o5r = _mm512_fnmadd_pd(s11, vr, o5r); o5i = _mm512_fnmadd_pd(s11, vi, o5i);
      o6r = _mm512_fnmadd_pd(s4, vr, o6r); o6i = _mm512_fnmadd_pd(s4, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+240);
      __m512d vi = _mm512_load_pd(AB+248);
      o1r = _mm512_fmadd_pd(s8, vr, o1r); o1i = _mm512_fmadd_pd(s8, vi, o1i);
      o2r = _mm512_fnmadd_pd(s7, vr, o2r); o2i = _mm512_fnmadd_pd(s7, vi, o2i);
      o3r = _mm512_fmadd_pd(s1, vr, o3r); o3i = _mm512_fmadd_pd(s1, vi, o3i);
      o4r = _mm512_fmadd_pd(s9, vr, o4r); o4i = _mm512_fmadd_pd(s9, vi, o4i);
      o5r = _mm512_fnmadd_pd(s6, vr, o5r); o5i = _mm512_fnmadd_pd(s6, vi, o5i);
      o6r = _mm512_fmadd_pd(s2, vr, o6r); o6i = _mm512_fmadd_pd(s2, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+272);
      __m512d vi = _mm512_load_pd(AB+280);
      o1r = _mm512_fmadd_pd(s9, vr, o1r); o1i = _mm512_fmadd_pd(s9, vi, o1i);
      o2r = _mm512_fnmadd_pd(s5, vr, o2r); o2i = _mm512_fnmadd_pd(s5, vi, o2i);
      o3r = _mm512_fmadd_pd(s4, vr, o3r); o3i = _mm512_fmadd_pd(s4, vi, o3i);
      o4r = _mm512_fnmadd_pd(s10, vr, o4r); o4i = _mm512_fnmadd_pd(s10, vi, o4i);
      o5r = _mm512_fnmadd_pd(s1, vr, o5r); o5i = _mm512_fnmadd_pd(s1, vi, o5i);
      o6r = _mm512_fmadd_pd(s8, vr, o6r); o6i = _mm512_fmadd_pd(s8, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+304);
      __m512d vi = _mm512_load_pd(AB+312);
      o1r = _mm512_fmadd_pd(s10, vr, o1r); o1i = _mm512_fmadd_pd(s10, vi, o1i);
      o2r = _mm512_fnmadd_pd(s3, vr, o2r); o2i = _mm512_fnmadd_pd(s3, vi, o2i);
      o3r = _mm512_fmadd_pd(s7, vr, o3r); o3i = _mm512_fmadd_pd(s7, vi, o3i);
      o4r = _mm512_fnmadd_pd(s6, vr, o4r); o4i = _mm512_fnmadd_pd(s6, vi, o4i);
      o5r = _mm512_fmadd_pd(s4, vr, o5r); o5i = _mm512_fmadd_pd(s4, vi, o5i);
      o6r = _mm512_fnmadd_pd(s9, vr, o6r); o6i = _mm512_fnmadd_pd(s9, vi, o6i);
    }
    { __m512d vr = _mm512_load_pd(AB+336);
      __m512d vi = _mm512_load_pd(AB+344);
      o1r = _mm512_fmadd_pd(s11, vr, o1r); o1i = _mm512_fmadd_pd(s11, vi, o1i);
      o2r = _mm512_fnmadd_pd(s1, vr, o2r); o2i = _mm512_fnmadd_pd(s1, vi, o2i);
      o3r = _mm512_fmadd_pd(s10, vr, o3r); o3i = _mm512_fmadd_pd(s10, vi, o3i);
      o4r = _mm512_fnmadd_pd(s2, vr, o4r); o4i = _mm512_fnmadd_pd(s2, vi, o4i);
      o5r = _mm512_fmadd_pd(s9, vr, o5r); o5i = _mm512_fmadd_pd(s9, vi, o5i);
      o6r = _mm512_fnmadd_pd(s3, vr, o6r); o6i = _mm512_fnmadd_pd(s3, vi, o6i);
    }
    { __m512d er = _mm512_load_pd(Escr+0), ei = _mm512_load_pd(Escr+8);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o1i), _mm512_load_pd(C+1*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o1r), _mm512_load_pd(C+1*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o1i), _mm512_load_pd(C+22*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o1r), _mm512_load_pd(C+22*cs+8));
      _mm512_store_pd(X+1*s, zr1);   _mm512_store_pd(X+1*s+8, zi1);
      _mm512_store_pd(X+22*s, zr2); _mm512_store_pd(X+22*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+16), ei = _mm512_load_pd(Escr+24);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o2i), _mm512_load_pd(C+2*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o2r), _mm512_load_pd(C+2*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o2i), _mm512_load_pd(C+21*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o2r), _mm512_load_pd(C+21*cs+8));
      _mm512_store_pd(X+2*s, zr1);   _mm512_store_pd(X+2*s+8, zi1);
      _mm512_store_pd(X+21*s, zr2); _mm512_store_pd(X+21*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+32), ei = _mm512_load_pd(Escr+40);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o3i), _mm512_load_pd(C+3*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o3r), _mm512_load_pd(C+3*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o3i), _mm512_load_pd(C+20*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o3r), _mm512_load_pd(C+20*cs+8));
      _mm512_store_pd(X+3*s, zr1);   _mm512_store_pd(X+3*s+8, zi1);
      _mm512_store_pd(X+20*s, zr2); _mm512_store_pd(X+20*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+48), ei = _mm512_load_pd(Escr+56);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o4i), _mm512_load_pd(C+4*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o4r), _mm512_load_pd(C+4*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o4i), _mm512_load_pd(C+19*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o4r), _mm512_load_pd(C+19*cs+8));
      _mm512_store_pd(X+4*s, zr1);   _mm512_store_pd(X+4*s+8, zi1);
      _mm512_store_pd(X+19*s, zr2); _mm512_store_pd(X+19*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+64), ei = _mm512_load_pd(Escr+72);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o5i), _mm512_load_pd(C+5*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o5r), _mm512_load_pd(C+5*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o5i), _mm512_load_pd(C+18*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o5r), _mm512_load_pd(C+18*cs+8));
      _mm512_store_pd(X+5*s, zr1);   _mm512_store_pd(X+5*s+8, zi1);
      _mm512_store_pd(X+18*s, zr2); _mm512_store_pd(X+18*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+80), ei = _mm512_load_pd(Escr+88);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o6i), _mm512_load_pd(C+6*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o6r), _mm512_load_pd(C+6*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o6i), _mm512_load_pd(C+17*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o6r), _mm512_load_pd(C+17*cs+8));
      _mm512_store_pd(X+6*s, zr1);   _mm512_store_pd(X+6*s+8, zi1);
      _mm512_store_pd(X+17*s, zr2); _mm512_store_pd(X+17*s+8, zi2); }
    }
    { __m512d s1 = _mm512_set1_pd(0x1.14459ad2be466p-2), s2 = _mm512_set1_pd(0x1.0a06e851db7cap-1), s3 = _mm512_set1_pd(0x1.763021aaa15d9p-1), s4 = _mm512_set1_pd(0x1.c698e42f47b09p-1), s5 = _mm512_set1_pd(0x1.f54a827142577p-1), s6 = _mm512_set1_pd(0x1.fece70dfd3efbp-1), s7 = _mm512_set1_pd(0x1.e270060999288p-1), s8 = _mm512_set1_pd(0x1.a249e0b897caap-1), s9 = _mm512_set1_pd(0x1.431df5838f7f1p-1), s10 = _mm512_set1_pd(0x1.97f6748e524b1p-2), s11 = _mm512_set1_pd(0x1.16de8a4564f1cp-3);
    __m512d o7r = _mm512_setzero_pd(), o7i = _mm512_setzero_pd();
    __m512d o8r = _mm512_setzero_pd(), o8i = _mm512_setzero_pd();
    __m512d o9r = _mm512_setzero_pd(), o9i = _mm512_setzero_pd();
    __m512d o10r = _mm512_setzero_pd(), o10i = _mm512_setzero_pd();
    __m512d o11r = _mm512_setzero_pd(), o11i = _mm512_setzero_pd();
    { __m512d vr = _mm512_load_pd(AB+16);
      __m512d vi = _mm512_load_pd(AB+24);
      o7r = _mm512_fmadd_pd(s7, vr, o7r); o7i = _mm512_fmadd_pd(s7, vi, o7i);
      o8r = _mm512_fmadd_pd(s8, vr, o8r); o8i = _mm512_fmadd_pd(s8, vi, o8i);
      o9r = _mm512_fmadd_pd(s9, vr, o9r); o9i = _mm512_fmadd_pd(s9, vi, o9i);
      o10r = _mm512_fmadd_pd(s10, vr, o10r); o10i = _mm512_fmadd_pd(s10, vi, o10i);
      o11r = _mm512_fmadd_pd(s11, vr, o11r); o11i = _mm512_fmadd_pd(s11, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+48);
      __m512d vi = _mm512_load_pd(AB+56);
      o7r = _mm512_fnmadd_pd(s9, vr, o7r); o7i = _mm512_fnmadd_pd(s9, vi, o7i);
      o8r = _mm512_fnmadd_pd(s7, vr, o8r); o8i = _mm512_fnmadd_pd(s7, vi, o8i);
      o9r = _mm512_fnmadd_pd(s5, vr, o9r); o9i = _mm512_fnmadd_pd(s5, vi, o9i);
      o10r = _mm512_fnmadd_pd(s3, vr, o10r); o10i = _mm512_fnmadd_pd(s3, vi, o10i);
      o11r = _mm512_fnmadd_pd(s1, vr, o11r); o11i = _mm512_fnmadd_pd(s1, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+80);
      __m512d vi = _mm512_load_pd(AB+88);
      o7r = _mm512_fnmadd_pd(s2, vr, o7r); o7i = _mm512_fnmadd_pd(s2, vi, o7i);
      o8r = _mm512_fmadd_pd(s1, vr, o8r); o8i = _mm512_fmadd_pd(s1, vi, o8i);
      o9r = _mm512_fmadd_pd(s4, vr, o9r); o9i = _mm512_fmadd_pd(s4, vi, o9i);
      o10r = _mm512_fmadd_pd(s7, vr, o10r); o10i = _mm512_fmadd_pd(s7, vi, o10i);
      o11r = _mm512_fmadd_pd(s10, vr, o11r); o11i = _mm512_fmadd_pd(s10, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+112);
      __m512d vi = _mm512_load_pd(AB+120);
      o7r = _mm512_fmadd_pd(s5, vr, o7r); o7i = _mm512_fmadd_pd(s5, vi, o7i);
      o8r = _mm512_fmadd_pd(s9, vr, o8r); o8i = _mm512_fmadd_pd(s9, vi, o8i);
      o9r = _mm512_fnmadd_pd(s10, vr, o9r); o9i = _mm512_fnmadd_pd(s10, vi, o9i);
      o10r = _mm512_fnmadd_pd(s6, vr, o10r); o10i = _mm512_fnmadd_pd(s6, vi, o10i);
      o11r = _mm512_fnmadd_pd(s2, vr, o11r); o11i = _mm512_fnmadd_pd(s2, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+144);
      __m512d vi = _mm512_load_pd(AB+152);
      o7r = _mm512_fnmadd_pd(s11, vr, o7r); o7i = _mm512_fnmadd_pd(s11, vi, o7i);
      o8r = _mm512_fnmadd_pd(s6, vr, o8r); o8i = _mm512_fnmadd_pd(s6, vi, o8i);
      o9r = _mm512_fnmadd_pd(s1, vr, o9r); o9i = _mm512_fnmadd_pd(s1, vi, o9i);
      o10r = _mm512_fmadd_pd(s4, vr, o10r); o10i = _mm512_fmadd_pd(s4, vi, o10i);
      o11r = _mm512_fmadd_pd(s9, vr, o11r); o11i = _mm512_fmadd_pd(s9, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+176);
      __m512d vi = _mm512_load_pd(AB+184);
      o7r = _mm512_fnmadd_pd(s4, vr, o7r); o7i = _mm512_fnmadd_pd(s4, vi, o7i);
      o8r = _mm512_fmadd_pd(s2, vr, o8r); o8i = _mm512_fmadd_pd(s2, vi, o8i);
      o9r = _mm512_fmadd_pd(s8, vr, o9r); o9i = _mm512_fmadd_pd(s8, vi, o9i);
      o10r = _mm512_fnmadd_pd(s9, vr, o10r); o10i = _mm512_fnmadd_pd(s9, vi, o10i);
      o11r = _mm512_fnmadd_pd(s3, vr, o11r); o11i = _mm512_fnmadd_pd(s3, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+208);
      __m512d vi = _mm512_load_pd(AB+216);
      o7r = _mm512_fmadd_pd(s3, vr, o7r); o7i = _mm512_fmadd_pd(s3, vi, o7i);
      o8r = _mm512_fmadd_pd(s10, vr, o8r); o8i = _mm512_fmadd_pd(s10, vi, o8i);
      o9r = _mm512_fnmadd_pd(s6, vr, o9r); o9i = _mm512_fnmadd_pd(s6, vi, o9i);
      o10r = _mm512_fmadd_pd(s1, vr, o10r); o10i = _mm512_fmadd_pd(s1, vi, o10i);
      o11r = _mm512_fmadd_pd(s8, vr, o11r); o11i = _mm512_fmadd_pd(s8, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+240);
      __m512d vi = _mm512_load_pd(AB+248);
      o7r = _mm512_fmadd_pd(s10, vr, o7r); o7i = _mm512_fmadd_pd(s10, vi, o7i);
      o8r = _mm512_fnmadd_pd(s5, vr, o8r); o8i = _mm512_fnmadd_pd(s5, vi, o8i);
      o9r = _mm512_fmadd_pd(s3, vr, o9r); o9i = _mm512_fmadd_pd(s3, vi, o9i);
      o10r = _mm512_fmadd_pd(s11, vr, o10r); o10i = _mm512_fmadd_pd(s11, vi, o10i);
      o11r = _mm512_fnmadd_pd(s4, vr, o11r); o11i = _mm512_fnmadd_pd(s4, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+272);
      __m512d vi = _mm512_load_pd(AB+280);
      o7r = _mm512_fnmadd_pd(s6, vr, o7r); o7i = _mm512_fnmadd_pd(s6, vi, o7i);
      o8r = _mm512_fmadd_pd(s3, vr, o8r); o8i = _mm512_fmadd_pd(s3, vi, o8i);
      o9r = _mm512_fnmadd_pd(s11, vr, o9r); o9i = _mm512_fnmadd_pd(s11, vi, o9i);
      o10r = _mm512_fnmadd_pd(s2, vr, o10r); o10i = _mm512_fnmadd_pd(s2, vi, o10i);
      o11r = _mm512_fmadd_pd(s7, vr, o11r); o11i = _mm512_fmadd_pd(s7, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+304);
      __m512d vi = _mm512_load_pd(AB+312);
      o7r = _mm512_fmadd_pd(s1, vr, o7r); o7i = _mm512_fmadd_pd(s1, vi, o7i);
      o8r = _mm512_fmadd_pd(s11, vr, o8r); o8i = _mm512_fmadd_pd(s11, vi, o8i);
      o9r = _mm512_fnmadd_pd(s2, vr, o9r); o9i = _mm512_fnmadd_pd(s2, vi, o9i);
      o10r = _mm512_fmadd_pd(s8, vr, o10r); o10i = _mm512_fmadd_pd(s8, vi, o10i);
      o11r = _mm512_fnmadd_pd(s5, vr, o11r); o11i = _mm512_fnmadd_pd(s5, vi, o11i);
    }
    { __m512d vr = _mm512_load_pd(AB+336);
      __m512d vi = _mm512_load_pd(AB+344);
      o7r = _mm512_fmadd_pd(s8, vr, o7r); o7i = _mm512_fmadd_pd(s8, vi, o7i);
      o8r = _mm512_fnmadd_pd(s4, vr, o8r); o8i = _mm512_fnmadd_pd(s4, vi, o8i);
      o9r = _mm512_fmadd_pd(s7, vr, o9r); o9i = _mm512_fmadd_pd(s7, vi, o9i);
      o10r = _mm512_fnmadd_pd(s5, vr, o10r); o10i = _mm512_fnmadd_pd(s5, vi, o10i);
      o11r = _mm512_fmadd_pd(s6, vr, o11r); o11i = _mm512_fmadd_pd(s6, vi, o11i);
    }
    { __m512d er = _mm512_load_pd(Escr+96), ei = _mm512_load_pd(Escr+104);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o7i), _mm512_load_pd(C+7*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o7r), _mm512_load_pd(C+7*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o7i), _mm512_load_pd(C+16*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o7r), _mm512_load_pd(C+16*cs+8));
      _mm512_store_pd(X+7*s, zr1);   _mm512_store_pd(X+7*s+8, zi1);
      _mm512_store_pd(X+16*s, zr2); _mm512_store_pd(X+16*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+112), ei = _mm512_load_pd(Escr+120);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o8i), _mm512_load_pd(C+8*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o8r), _mm512_load_pd(C+8*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o8i), _mm512_load_pd(C+15*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o8r), _mm512_load_pd(C+15*cs+8));
      _mm512_store_pd(X+8*s, zr1);   _mm512_store_pd(X+8*s+8, zi1);
      _mm512_store_pd(X+15*s, zr2); _mm512_store_pd(X+15*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+128), ei = _mm512_load_pd(Escr+136);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o9i), _mm512_load_pd(C+9*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o9r), _mm512_load_pd(C+9*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o9i), _mm512_load_pd(C+14*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o9r), _mm512_load_pd(C+14*cs+8));
      _mm512_store_pd(X+9*s, zr1);   _mm512_store_pd(X+9*s+8, zi1);
      _mm512_store_pd(X+14*s, zr2); _mm512_store_pd(X+14*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+144), ei = _mm512_load_pd(Escr+152);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o10i), _mm512_load_pd(C+10*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o10r), _mm512_load_pd(C+10*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o10i), _mm512_load_pd(C+13*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o10r), _mm512_load_pd(C+13*cs+8));
      _mm512_store_pd(X+10*s, zr1);   _mm512_store_pd(X+10*s+8, zi1);
      _mm512_store_pd(X+13*s, zr2); _mm512_store_pd(X+13*s+8, zi2); }
    { __m512d er = _mm512_load_pd(Escr+160), ei = _mm512_load_pd(Escr+168);
      __m512d zr1 = _mm512_add_pd(_mm512_add_pd(er, o11i), _mm512_load_pd(C+11*cs));
      __m512d zi1 = _mm512_add_pd(_mm512_sub_pd(ei, o11r), _mm512_load_pd(C+11*cs+8));
      __m512d zr2 = _mm512_add_pd(_mm512_sub_pd(er, o11i), _mm512_load_pd(C+12*cs));
      __m512d zi2 = _mm512_add_pd(_mm512_add_pd(ei, o11r), _mm512_load_pd(C+12*cs+8));
      _mm512_store_pd(X+11*s, zr1);   _mm512_store_pd(X+11*s+8, zi1);
      _mm512_store_pd(X+12*s, zr2); _mm512_store_pd(X+12*s+8, zi2); }
    }
}
#define MAPX_23 0
#define PXF_23 0
#define PFPRIME_23 0

static void __attribute__((noinline))  dft23_one(double* X, long es){ dft23(X, es, 0); }
static void __attribute__((noinline))  dft23_onez(double* X){ dft23(X, 1, PFPRIME_23); }
static void __attribute__((noinline))  dft23_onezm(double* X){ dft23zm(X, 1, PFPRIME_23); }
static void __attribute__((noinline))  dft23_onem(double* X, long es, const double* Ct){ dft23m(X, es, 0, Ct, 1); }
static void dft23_sweep_zy(double* restrict X){
    for(long x=0; x<23; x++){
        double* P = X + x*529*16;
        for(long y=0; y<23; y++) dft23_onez(P + y*23*16);
        for(long z=0; z<23; z++) dft23_one(P + z*16, 23);
    }
}
#if MAPZB_FLAG
static void __attribute__((noinline)) mapblk_23(double* restrict P){
    for(long e=0; e<23; e+=1){
        __m512d zr = _mm512_load_pd(P + e*16);
        __m512d zi = _mm512_load_pd(P + e*16 + 8);
        if(e & 1){ map2(zr, zi, &zr, &zi); } else { maphw(zr, zi, &zr, &zi); }
        _mm512_store_pd(P + e*16, zr);
        _mm512_store_pd(P + e*16 + 8, zi);
    }
}
static void dft23_sweep_zym(double* restrict X){
    for(long x=0; x<23; x++){
        double* P = X + x*529*16;
        for(long y=0; y<23; y++){ mapblk_23(P + y*23*16); dft23_one(P + y*23*16, 1); }
        for(long z=0; z<23; z++) dft23_one(P + z*16, 23);
    }
}
#else
static void dft23_sweep_zym(double* restrict X){
    for(long x=0; x<23; x++){
        double* P = X + x*529*16;
        for(long y=0; y<23; y++) dft23_onezm(P + y*23*16);
        for(long z=0; z<23; z++) dft23_one(P + z*16, 23);
    }
}
#endif
static void dft23_sweep_x_map(double* restrict X, const double* restrict Ct){
    for(long p=0; p<529; p++) dft23_onem(X + p*16, 529, Ct + p*23*16);
}
static void dft23_sweep_x_plain(double* restrict X){
    for(long p=0; p<529; p++) dft23_one(X + p*16, 529);
}
static void dft23_sweep_zy_ms(double* restrict X, const double* restrict C){
    for(long x=0; x<23; x++){
        double* P = X + x*529*16;
        for(long y=0; y<23; y++) dft23_one(P + y*23*16, 1);
        for(long z=0; z<23; z++) dft23_one(P + z*16, 23);
        if(x) mapslab(X + (x-1)*529*16, C + (x-1)*529*16, 529);
    }
    mapslab(X + (23-1)*529*16, C + (23-1)*529*16, 529);
}


/* ingest: 8 volumes AoS complex -> group SoA [e][2][8] */
static void ingest_23(const double* const* src, double* G){
    for(long e=0; e<12164; e+=4){
        __m512d r0=_mm512_loadu_pd(src[0]+2*e), r1=_mm512_loadu_pd(src[1]+2*e);
        __m512d r2=_mm512_loadu_pd(src[2]+2*e), r3=_mm512_loadu_pd(src[3]+2*e);
        __m512d r4=_mm512_loadu_pd(src[4]+2*e), r5=_mm512_loadu_pd(src[5]+2*e);
        __m512d r6=_mm512_loadu_pd(src[6]+2*e), r7=_mm512_loadu_pd(src[7]+2*e);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        _mm512_store_pd(G+e*16,    o0); _mm512_store_pd(G+e*16+8,  o1);
        _mm512_store_pd(G+e*16+16, o2); _mm512_store_pd(G+e*16+24, o3);
        _mm512_store_pd(G+e*16+32, o4); _mm512_store_pd(G+e*16+40, o5);
        _mm512_store_pd(G+e*16+48, o6); _mm512_store_pd(G+e*16+56, o7);
    }
    { /* tail of 3 elements */
        const long e = 12164;
        const __mmask8 mk = (__mmask8)((1u<<(2*3))-1u);
        __m512d r0=_mm512_maskz_loadu_pd(mk, src[0]+2*e), r1=_mm512_maskz_loadu_pd(mk, src[1]+2*e);
        __m512d r2=_mm512_maskz_loadu_pd(mk, src[2]+2*e), r3=_mm512_maskz_loadu_pd(mk, src[3]+2*e);
        __m512d r4=_mm512_maskz_loadu_pd(mk, src[4]+2*e), r5=_mm512_maskz_loadu_pd(mk, src[5]+2*e);
        __m512d r6=_mm512_maskz_loadu_pd(mk, src[6]+2*e), r7=_mm512_maskz_loadu_pd(mk, src[7]+2*e);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        __m512d A[8]; A[0]=o0;A[1]=o1;A[2]=o2;A[3]=o3;A[4]=o4;A[5]=o5;A[6]=o6;A[7]=o7;
        for(int q=0;q<2*3;q++) _mm512_store_pd(G+e*16+q*8, A[q]);
    }
}
/* output with map: group SoA (unmapped z) -> nv AoS volumes */
static void output_23(const double* G, double* const* dst, int nv){
    for(long e=0; e<12164; e+=4){
        __m512d i0=_mm512_load_pd(G+e*16),    i1=_mm512_load_pd(G+e*16+8);
        __m512d i2=_mm512_load_pd(G+e*16+16), i3=_mm512_load_pd(G+e*16+24);
        __m512d i4=_mm512_load_pd(G+e*16+32), i5=_mm512_load_pd(G+e*16+40);
        __m512d i6=_mm512_load_pd(G+e*16+48), i7=_mm512_load_pd(G+e*16+56);
        map2(i0,i1,&i0,&i1); map2(i2,i3,&i2,&i3); map2(i4,i5,&i4,&i5); map2(i6,i7,&i6,&i7);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(i0,i1,i2,i3,i4,i5,i6,i7,o0,o1,o2,o3,o4,o5,o6,o7);
        __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
        for(int v=0; v<nv; v++) _mm512_storeu_pd(dst[v]+2*e, *O[v]);
    }
    { /* tail */
        const long e = 12164;
        const __mmask8 mk = (__mmask8)((1u<<(2*3))-1u);
        __m512d A[8];
        for(int q=0;q<2*3;q++) A[q] = _mm512_load_pd(G+e*16+q*8);
        for(int q=0;q<2*3;q+=2) map2(A[q],A[q+1],&A[q],&A[q+1]);
        for(int q=2*3;q<8;q++) A[q] = _mm512_setzero_pd();
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7],o0,o1,o2,o3,o4,o5,o6,o7);
        __m512d* O[8] = {&o0,&o1,&o2,&o3,&o4,&o5,&o6,&o7};
        for(int v=0; v<nv; v++) _mm512_mask_storeu_pd(dst[v]+2*e, mk, *O[v]);
    }
}


static double* Xg_23 = 0;
static double* Cg_23 = 0;
void hot2_23(long which){
    if(!Xg_23){ Xg_23 = alloc_huge_st((12167+64*23)*16*8); Cg_23 = alloc_huge_st(12167*16*8); }
    double* P = Xg_23;
    if(which==99){ for(long i=0;i<529*16;i++) P[i] = 0.5 + 1e-6*(i%97); return; }
    if(which==0 || which==2) for(long y=0; y<23; y++) dft23_one(P + y*23*16, 1);
    if(which==1 || which==2) for(long z=0; z<23; z++) dft23_one(P + z*16, 23);
}
void bsweep_23(long which, long n){
    if(!Xg_23){ Xg_23 = alloc_huge_st(12167*16*8); Cg_23 = alloc_huge_st(12167*16*8); }
    for(long i=0;i<12167*16;i++){ Xg_23[i] = 0.5 + 1e-6*(i%97); Cg_23[i] = 0.01; }
    for(long r=0;r<n;r++){
        if(which==0) dft23_sweep_zy(Xg_23);
        else if(which==1) dft23_sweep_zym(Xg_23);
        else if(which==2) dft23_sweep_x_map(Xg_23, Cg_23);
#if USEASM_FLAG
#endif
        if((r&7)==7) for(long i=0;i<12167*16;i+=997) Xg_23[i] = 0.5;
    }
}
void run_23(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    const long NE = 12167;
    if(!Xg_23){ Xg_23 = alloc_huge_st(NE*16*8); Cg_23 = alloc_huge_st(NE*16*8); }
    double* X = Xg_23; double* Ct = Cg_23;
    for(long g0=0; g0<B; g0+=8){
        int nv = (int)((B - g0) < 8 ? (B - g0) : 8);
        const double* src[8]; const double* csrc[8];
        double* d1[8]; double* dm[8];
        for(int v=0; v<8; v++){
            int vv = v < nv ? v : 0;
            src[v] = x0 + (g0+vv)*2*NE; csrc[v] = c + (g0+vv)*2*NE;
            if(v<nv){ d1[v] = out1 + (g0+v)*2*NE; dm[v] = outm + (g0+v)*2*NE; }
        }
        /* ingest c (consumption order for MAPX/deferred paths; plain for PXF) */
#if PXF_23
        ingest_23(csrc, Ct);
#else
        ingest_23(csrc, X);
        for(long p=0; p<529; p++)
            for(long k=0; k<23; k++){
                _mm512_store_pd(Ct + (p*23+k)*16,     _mm512_load_pd(X + (k*529+p)*16));
                _mm512_store_pd(Ct + (p*23+k)*16 + 8, _mm512_load_pd(X + (k*529+p)*16 + 8));
            }
#endif
        ingest_23(src, X);
        for(long t=0; t<m; t++){
#if PXF_23
            dft23_sweep_x_plain(X);
            dft23_sweep_zy_ms(X, Ct);
#elif USEASM_FLAG
            dft23_sweep_zy_asm(X, t>0);
            dft23_sweep_x_asm(X, Ct);
#else
#if MAPX_23
            dft23_sweep_zy(X);
            dft23_sweep_x_map(X, Ct);
#else
            if(t==0) dft23_sweep_zy(X); else dft23_sweep_zym(X);
            dft23_sweep_x_map(X, Ct);
#endif
#endif
            if(t==0 && m>1) output_23(X, d1, nv);
        }
        output_23(X, dm, nv);
        if(m==1) output_23(X, d1, nv);
    }
}
