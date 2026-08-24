
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#define ALIGN64 __attribute__((aligned(64)))
static double* alloc_arena(long bytes){
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
static __m512d V_ONE, V_HALF, V_TINY;
__attribute__((constructor)) static void init_consts(void){
    V_ONE = _mm512_set1_pd(1.0); V_HALF = _mm512_set1_pd(0.5); V_TINY = _mm512_set1_pd(1e-30);
}
#define TR8_UNUSED(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7) do{ \
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


#define BCV(dst, mem) dst = _mm512_set1_pd(*(volatile const double*)&(mem))
// 17-VOP map: z=(xr+c, xi+c); out = z/(1+|z|)
#define MAPST(xr_, xi_, dst, off, cbase, coff) do{ \
    __m512d zr = _mm512_add_pd(xr_, _mm512_load_pd((cbase)+(coff))); \
    __m512d zi = _mm512_add_pd(xi_, _mm512_load_pd((cbase)+(coff)+8)); \
    __m512d mm = _mm512_fmadd_pd(zr, zr, _mm512_fmadd_pd(zi, zi, V_TINY)); \
    __m512d r0 = _mm512_rsqrt14_pd(mm); \
    __m512d mg0= _mm512_mul_pd(mm, r0); \
    __m512d t_ = _mm512_mul_pd(mg0, r0); \
    __m512d e_ = _mm512_fnmadd_pd(t_, V_HALF, V_15); \
    __m512d mg1= _mm512_mul_pd(mg0, e_); \
    __m512d r1 = _mm512_mul_pd(r0, e_); \
    __m512d e3 = _mm512_fnmadd_pd(mg1, mg1, mm); \
    __m512d hr = _mm512_mul_pd(r1, V_HALF); \
    __m512d u  = _mm512_add_pd(V_ONE, mg1); \
    u = _mm512_fmadd_pd(e3, hr, u); \
    __m512d w0 = _mm512_rcp14_pd(u); \
    __m512d e4 = _mm512_fnmadd_pd(u, w0, V_ONE); \
    __m512d w1 = _mm512_fmadd_pd(w0, e4, w0); \
    __m512d ee = _mm512_mul_pd(e4, e4); \
    __m512d w2 = _mm512_fmadd_pd(w1, ee, w1); \
    _mm512_store_pd((dst)+(off),   _mm512_mul_pd(zr, w2)); \
    _mm512_store_pd((dst)+(off)+8, _mm512_mul_pd(zi, w2)); \
}while(0)


static __m512i IDX_RE, IDX_IM;
static __m512d V_15;
__attribute__((constructor)) static void init_idx2(void){
    IDX_RE = _mm512_set_epi64(14,12,10,8,6,4,2,0);
    IDX_IM = _mm512_set_epi64(15,13,11,9,7,5,3,1);
    V_15 = _mm512_set1_pd(1.5);
}
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


static void conv_in_13(const double* const* src, double* restrict G){
    for(long e=0; e+4<=2197; e+=4){
        __m512d r0=_mm512_loadu_pd(src[0]+2*e), r1=_mm512_loadu_pd(src[1]+2*e);
        __m512d r2=_mm512_loadu_pd(src[2]+2*e), r3=_mm512_loadu_pd(src[3]+2*e);
        __m512d r4=_mm512_loadu_pd(src[4]+2*e), r5=_mm512_loadu_pd(src[5]+2*e);
        __m512d r6=_mm512_loadu_pd(src[6]+2*e), r7=_mm512_loadu_pd(src[7]+2*e);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        double* g = G + e*16;
        _mm512_store_pd(g,o0); _mm512_store_pd(g+8,o1);
        _mm512_store_pd(g+16,o2); _mm512_store_pd(g+24,o3);
        _mm512_store_pd(g+32,o4); _mm512_store_pd(g+40,o5);
        _mm512_store_pd(g+48,o6); _mm512_store_pd(g+56,o7);
    }
    for(long e=2196; e<2197; e++)
        for(int v=0; v<8; v++){ G[e*16+v] = src[v][2*e]; G[e*16+8+v] = src[v][2*e+1]; }
}
static void conv_out_13(const double* restrict G, double* const* dst){
    for(long e=0; e+4<=2197; e+=4){
        const double* g = G + e*16;
        __m512d r0=_mm512_load_pd(g),    r1=_mm512_load_pd(g+8);
        __m512d r2=_mm512_load_pd(g+16), r3=_mm512_load_pd(g+24);
        __m512d r4=_mm512_load_pd(g+32), r5=_mm512_load_pd(g+40);
        __m512d r6=_mm512_load_pd(g+48), r7=_mm512_load_pd(g+56);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        _mm512_storeu_pd(dst[0]+2*e,o0); _mm512_storeu_pd(dst[1]+2*e,o1);
        _mm512_storeu_pd(dst[2]+2*e,o2); _mm512_storeu_pd(dst[3]+2*e,o3);
        _mm512_storeu_pd(dst[4]+2*e,o4); _mm512_storeu_pd(dst[5]+2*e,o5);
        _mm512_storeu_pd(dst[6]+2*e,o6); _mm512_storeu_pd(dst[7]+2*e,o7);
    }
    for(long e=2196; e<2197; e++)
        for(int v=0; v<8; v++){ dst[v][2*e] = G[e*16+v]; dst[v][2*e+1] = G[e*16+8+v]; }
}

static double* G_13 = 0;
static double* G2_13 = 0;
static double* CP_13 = 0;
static double* CT_13 = 0;
static const double KC_13[6] ALIGN64 = { 0x1.c55a7e00740e9p-1, 0x1.22d961ea71119p-1, 0x1.edb7debaa3ed5p-4, -0x1.6b1d8b2365d9ep-2, -0x1.7f3ccd0032e0dp-1, -0x1.f11f493053d00p-1 };
static const double KS_13[6] ALIGN64 = { 0x1.dbe064267c47bp-2, 0x1.a55e242a4c3d2p-1, 0x1.fc44566966769p-1, 0x1.deba72ef20147p-1, 0x1.5384d024c2f84p-1, 0x1.ea1e54bc48dbcp-3 };

static void __attribute__((noinline)) dz_13(double* restrict x){
    double dscr[12*8] ALIGN64;
    double pscr[14*8] ALIGN64;
    double sscr[12*8] ALIGN64;
    {
    __m512d kc1; BCV(kc1, KC_13[0]);
    __m512d kc2; BCV(kc2, KC_13[1]);
    __m512d kc3; BCV(kc3, KC_13[2]);
    __m512d kc4; BCV(kc4, KC_13[3]);
    __m512d kc5; BCV(kc5, KC_13[4]);
    __m512d kc6; BCV(kc6, KC_13[5]);
    __m512d u0r = _mm512_load_pd(x), u0i = _mm512_load_pd(x+8);
    __m512d pr1 = u0r, pi1 = u0i;
    __m512d pr2 = u0r, pi2 = u0i;
    __m512d pr3 = u0r, pi3 = u0i;
    __m512d pr4 = u0r, pi4 = u0i;
    __m512d pr5 = u0r, pi5 = u0i;
    __m512d pr6 = u0r, pi6 = u0i;
    {
    __m512d ar = _mm512_load_pd(x+16), ai = _mm512_load_pd(x+16+8);
    __m512d br = _mm512_load_pd(x+192), bi = _mm512_load_pd(x+192+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+0, dr); _mm512_store_pd(dscr+8, di);
    _mm512_store_pd(sscr+0, sr); _mm512_store_pd(sscr+8, si);
    pr1 = _mm512_fmadd_pd(kc1, sr, pr1); pi1 = _mm512_fmadd_pd(kc1, si, pi1);
    pr2 = _mm512_fmadd_pd(kc2, sr, pr2); pi2 = _mm512_fmadd_pd(kc2, si, pi2);
    pr3 = _mm512_fmadd_pd(kc3, sr, pr3); pi3 = _mm512_fmadd_pd(kc3, si, pi3);
    pr4 = _mm512_fmadd_pd(kc4, sr, pr4); pi4 = _mm512_fmadd_pd(kc4, si, pi4);
    pr5 = _mm512_fmadd_pd(kc5, sr, pr5); pi5 = _mm512_fmadd_pd(kc5, si, pi5);
    pr6 = _mm512_fmadd_pd(kc6, sr, pr6); pi6 = _mm512_fmadd_pd(kc6, si, pi6);
    }
    {
    __m512d ar = _mm512_load_pd(x+32), ai = _mm512_load_pd(x+32+8);
    __m512d br = _mm512_load_pd(x+176), bi = _mm512_load_pd(x+176+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+16, dr); _mm512_store_pd(dscr+24, di);
    _mm512_store_pd(sscr+16, sr); _mm512_store_pd(sscr+24, si);
    pr1 = _mm512_fmadd_pd(kc2, sr, pr1); pi1 = _mm512_fmadd_pd(kc2, si, pi1);
    pr2 = _mm512_fmadd_pd(kc4, sr, pr2); pi2 = _mm512_fmadd_pd(kc4, si, pi2);
    pr3 = _mm512_fmadd_pd(kc6, sr, pr3); pi3 = _mm512_fmadd_pd(kc6, si, pi3);
    pr4 = _mm512_fmadd_pd(kc5, sr, pr4); pi4 = _mm512_fmadd_pd(kc5, si, pi4);
    pr5 = _mm512_fmadd_pd(kc3, sr, pr5); pi5 = _mm512_fmadd_pd(kc3, si, pi5);
    pr6 = _mm512_fmadd_pd(kc1, sr, pr6); pi6 = _mm512_fmadd_pd(kc1, si, pi6);
    }
    {
    __m512d ar = _mm512_load_pd(x+48), ai = _mm512_load_pd(x+48+8);
    __m512d br = _mm512_load_pd(x+160), bi = _mm512_load_pd(x+160+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+32, dr); _mm512_store_pd(dscr+40, di);
    _mm512_store_pd(sscr+32, sr); _mm512_store_pd(sscr+40, si);
    pr1 = _mm512_fmadd_pd(kc3, sr, pr1); pi1 = _mm512_fmadd_pd(kc3, si, pi1);
    pr2 = _mm512_fmadd_pd(kc6, sr, pr2); pi2 = _mm512_fmadd_pd(kc6, si, pi2);
    pr3 = _mm512_fmadd_pd(kc4, sr, pr3); pi3 = _mm512_fmadd_pd(kc4, si, pi3);
    pr4 = _mm512_fmadd_pd(kc1, sr, pr4); pi4 = _mm512_fmadd_pd(kc1, si, pi4);
    pr5 = _mm512_fmadd_pd(kc2, sr, pr5); pi5 = _mm512_fmadd_pd(kc2, si, pi5);
    pr6 = _mm512_fmadd_pd(kc5, sr, pr6); pi6 = _mm512_fmadd_pd(kc5, si, pi6);
    }
    {
    __m512d ar = _mm512_load_pd(x+64), ai = _mm512_load_pd(x+64+8);
    __m512d br = _mm512_load_pd(x+144), bi = _mm512_load_pd(x+144+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+48, dr); _mm512_store_pd(dscr+56, di);
    _mm512_store_pd(sscr+48, sr); _mm512_store_pd(sscr+56, si);
    pr1 = _mm512_fmadd_pd(kc4, sr, pr1); pi1 = _mm512_fmadd_pd(kc4, si, pi1);
    pr2 = _mm512_fmadd_pd(kc5, sr, pr2); pi2 = _mm512_fmadd_pd(kc5, si, pi2);
    pr3 = _mm512_fmadd_pd(kc1, sr, pr3); pi3 = _mm512_fmadd_pd(kc1, si, pi3);
    pr4 = _mm512_fmadd_pd(kc3, sr, pr4); pi4 = _mm512_fmadd_pd(kc3, si, pi4);
    pr5 = _mm512_fmadd_pd(kc6, sr, pr5); pi5 = _mm512_fmadd_pd(kc6, si, pi5);
    pr6 = _mm512_fmadd_pd(kc2, sr, pr6); pi6 = _mm512_fmadd_pd(kc2, si, pi6);
    }
    {
    __m512d ar = _mm512_load_pd(x+80), ai = _mm512_load_pd(x+80+8);
    __m512d br = _mm512_load_pd(x+128), bi = _mm512_load_pd(x+128+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+64, dr); _mm512_store_pd(dscr+72, di);
    _mm512_store_pd(sscr+64, sr); _mm512_store_pd(sscr+72, si);
    pr1 = _mm512_fmadd_pd(kc5, sr, pr1); pi1 = _mm512_fmadd_pd(kc5, si, pi1);
    pr2 = _mm512_fmadd_pd(kc3, sr, pr2); pi2 = _mm512_fmadd_pd(kc3, si, pi2);
    pr3 = _mm512_fmadd_pd(kc2, sr, pr3); pi3 = _mm512_fmadd_pd(kc2, si, pi3);
    pr4 = _mm512_fmadd_pd(kc6, sr, pr4); pi4 = _mm512_fmadd_pd(kc6, si, pi4);
    pr5 = _mm512_fmadd_pd(kc1, sr, pr5); pi5 = _mm512_fmadd_pd(kc1, si, pi5);
    pr6 = _mm512_fmadd_pd(kc4, sr, pr6); pi6 = _mm512_fmadd_pd(kc4, si, pi6);
    }
    {
    __m512d ar = _mm512_load_pd(x+96), ai = _mm512_load_pd(x+96+8);
    __m512d br = _mm512_load_pd(x+112), bi = _mm512_load_pd(x+112+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+80, dr); _mm512_store_pd(dscr+88, di);
    _mm512_store_pd(sscr+80, sr); _mm512_store_pd(sscr+88, si);
    pr1 = _mm512_fmadd_pd(kc6, sr, pr1); pi1 = _mm512_fmadd_pd(kc6, si, pi1);
    pr2 = _mm512_fmadd_pd(kc1, sr, pr2); pi2 = _mm512_fmadd_pd(kc1, si, pi2);
    pr3 = _mm512_fmadd_pd(kc5, sr, pr3); pi3 = _mm512_fmadd_pd(kc5, si, pi3);
    pr4 = _mm512_fmadd_pd(kc2, sr, pr4); pi4 = _mm512_fmadd_pd(kc2, si, pi4);
    pr5 = _mm512_fmadd_pd(kc4, sr, pr5); pi5 = _mm512_fmadd_pd(kc4, si, pi5);
    pr6 = _mm512_fmadd_pd(kc3, sr, pr6); pi6 = _mm512_fmadd_pd(kc3, si, pi6);
    }
    _mm512_store_pd(pscr+0, pr1); _mm512_store_pd(pscr+8, pi1);
    _mm512_store_pd(pscr+16, pr2); _mm512_store_pd(pscr+24, pi2);
    _mm512_store_pd(pscr+32, pr3); _mm512_store_pd(pscr+40, pi3);
    _mm512_store_pd(pscr+48, pr4); _mm512_store_pd(pscr+56, pi4);
    _mm512_store_pd(pscr+64, pr5); _mm512_store_pd(pscr+72, pi5);
    _mm512_store_pd(pscr+80, pr6); _mm512_store_pd(pscr+88, pi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d x0r = _mm512_load_pd(x), x0i = _mm512_load_pd(x+8);
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+0)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+8));
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+16)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+24));
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+32)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+40));
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+48)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+56));
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+64)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+72));
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+80)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+88));
    _mm512_store_pd(pscr+96, x0r); _mm512_store_pd(pscr+104, x0i);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_13[0]);
    __m512d ks2; BCV(ks2, KS_13[1]);
    __m512d ks3; BCV(ks3, KS_13[2]);
    __m512d ks4; BCV(ks4, KS_13[3]);
    __m512d ks5; BCV(ks5, KS_13[4]);
    __m512d ks6; BCV(ks6, KS_13[5]);
    __m512d qr1, qi1;
    __m512d qr2, qi2;
    __m512d qr3, qi3;
    __m512d qr4, qi4;
    __m512d qr5, qi5;
    __m512d qr6, qi6;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr1 = _mm512_mul_pd(ks1, di); qi1 = _mm512_mul_pd(ks1, dr);
    qr2 = _mm512_mul_pd(ks2, di); qi2 = _mm512_mul_pd(ks2, dr);
    qr3 = _mm512_mul_pd(ks3, di); qi3 = _mm512_mul_pd(ks3, dr);
    qr4 = _mm512_mul_pd(ks4, di); qi4 = _mm512_mul_pd(ks4, dr);
    qr5 = _mm512_mul_pd(ks5, di); qi5 = _mm512_mul_pd(ks5, dr);
    qr6 = _mm512_mul_pd(ks6, di); qi6 = _mm512_mul_pd(ks6, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr1 = _mm512_fmadd_pd(ks2, di, qr1); qi1 = _mm512_fmadd_pd(ks2, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks4, di, qr2); qi2 = _mm512_fmadd_pd(ks4, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks6, di, qr3); qi3 = _mm512_fmadd_pd(ks6, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks5, di, qr4); qi4 = _mm512_fnmadd_pd(ks5, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks3, di, qr5); qi5 = _mm512_fnmadd_pd(ks3, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks1, di, qr6); qi6 = _mm512_fnmadd_pd(ks1, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr1 = _mm512_fmadd_pd(ks3, di, qr1); qi1 = _mm512_fmadd_pd(ks3, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks6, di, qr2); qi2 = _mm512_fmadd_pd(ks6, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks4, di, qr3); qi3 = _mm512_fnmadd_pd(ks4, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks1, di, qr4); qi4 = _mm512_fnmadd_pd(ks1, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks2, di, qr5); qi5 = _mm512_fmadd_pd(ks2, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks5, di, qr6); qi6 = _mm512_fmadd_pd(ks5, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr1 = _mm512_fmadd_pd(ks4, di, qr1); qi1 = _mm512_fmadd_pd(ks4, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks5, di, qr2); qi2 = _mm512_fnmadd_pd(ks5, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks1, di, qr3); qi3 = _mm512_fnmadd_pd(ks1, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks3, di, qr4); qi4 = _mm512_fmadd_pd(ks3, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks6, di, qr5); qi5 = _mm512_fnmadd_pd(ks6, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks2, di, qr6); qi6 = _mm512_fnmadd_pd(ks2, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr1 = _mm512_fmadd_pd(ks5, di, qr1); qi1 = _mm512_fmadd_pd(ks5, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks3, di, qr2); qi2 = _mm512_fnmadd_pd(ks3, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks2, di, qr3); qi3 = _mm512_fmadd_pd(ks2, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks6, di, qr4); qi4 = _mm512_fnmadd_pd(ks6, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks1, di, qr5); qi5 = _mm512_fnmadd_pd(ks1, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks4, di, qr6); qi6 = _mm512_fmadd_pd(ks4, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr1 = _mm512_fmadd_pd(ks6, di, qr1); qi1 = _mm512_fmadd_pd(ks6, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks1, di, qr2); qi2 = _mm512_fnmadd_pd(ks1, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks5, di, qr3); qi3 = _mm512_fmadd_pd(ks5, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks2, di, qr4); qi4 = _mm512_fnmadd_pd(ks2, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks4, di, qr5); qi5 = _mm512_fmadd_pd(ks4, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks3, di, qr6); qi6 = _mm512_fnmadd_pd(ks3, dr, qi6);
    }
    __m512d x0r = _mm512_load_pd(pscr+96), x0i = _mm512_load_pd(pscr+104);
    _mm512_store_pd(x, x0r); _mm512_store_pd(x+8, x0i);
    {
    __m512d Pr = _mm512_load_pd(pscr+0), Pi = _mm512_load_pd(pscr+8);
    __m512d xr = _mm512_add_pd(Pr, qr1), xi = _mm512_sub_pd(Pi, qi1);
    __m512d yr = _mm512_sub_pd(Pr, qr1), yi = _mm512_add_pd(Pi, qi1);
    _mm512_store_pd(x+16, xr); _mm512_store_pd(x+16+8, xi);
    _mm512_store_pd(x+192, yr); _mm512_store_pd(x+192+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+16), Pi = _mm512_load_pd(pscr+24);
    __m512d xr = _mm512_add_pd(Pr, qr2), xi = _mm512_sub_pd(Pi, qi2);
    __m512d yr = _mm512_sub_pd(Pr, qr2), yi = _mm512_add_pd(Pi, qi2);
    _mm512_store_pd(x+32, xr); _mm512_store_pd(x+32+8, xi);
    _mm512_store_pd(x+176, yr); _mm512_store_pd(x+176+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+32), Pi = _mm512_load_pd(pscr+40);
    __m512d xr = _mm512_add_pd(Pr, qr3), xi = _mm512_sub_pd(Pi, qi3);
    __m512d yr = _mm512_sub_pd(Pr, qr3), yi = _mm512_add_pd(Pi, qi3);
    _mm512_store_pd(x+48, xr); _mm512_store_pd(x+48+8, xi);
    _mm512_store_pd(x+160, yr); _mm512_store_pd(x+160+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+48), Pi = _mm512_load_pd(pscr+56);
    __m512d xr = _mm512_add_pd(Pr, qr4), xi = _mm512_sub_pd(Pi, qi4);
    __m512d yr = _mm512_sub_pd(Pr, qr4), yi = _mm512_add_pd(Pi, qi4);
    _mm512_store_pd(x+64, xr); _mm512_store_pd(x+64+8, xi);
    _mm512_store_pd(x+144, yr); _mm512_store_pd(x+144+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+64), Pi = _mm512_load_pd(pscr+72);
    __m512d xr = _mm512_add_pd(Pr, qr5), xi = _mm512_sub_pd(Pi, qi5);
    __m512d yr = _mm512_sub_pd(Pr, qr5), yi = _mm512_add_pd(Pi, qi5);
    _mm512_store_pd(x+80, xr); _mm512_store_pd(x+80+8, xi);
    _mm512_store_pd(x+128, yr); _mm512_store_pd(x+128+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+80), Pi = _mm512_load_pd(pscr+88);
    __m512d xr = _mm512_add_pd(Pr, qr6), xi = _mm512_sub_pd(Pi, qi6);
    __m512d yr = _mm512_sub_pd(Pr, qr6), yi = _mm512_add_pd(Pi, qi6);
    _mm512_store_pd(x+96, xr); _mm512_store_pd(x+96+8, xi);
    _mm512_store_pd(x+112, yr); _mm512_store_pd(x+112+8, yi);
    }
    }
}
static void __attribute__((noinline)) dy_13(double* restrict x, double* restrict d){
    double dscr[12*8] ALIGN64;
    double pscr[14*8] ALIGN64;
    double sscr[12*8] ALIGN64;
    {
    __m512d kc1; BCV(kc1, KC_13[0]);
    __m512d kc2; BCV(kc2, KC_13[1]);
    __m512d kc3; BCV(kc3, KC_13[2]);
    __m512d kc4; BCV(kc4, KC_13[3]);
    __m512d kc5; BCV(kc5, KC_13[4]);
    __m512d kc6; BCV(kc6, KC_13[5]);
    __m512d u0r = _mm512_load_pd(x), u0i = _mm512_load_pd(x+8);
    __m512d pr1 = u0r, pi1 = u0i;
    __m512d pr2 = u0r, pi2 = u0i;
    __m512d pr3 = u0r, pi3 = u0i;
    __m512d pr4 = u0r, pi4 = u0i;
    __m512d pr5 = u0r, pi5 = u0i;
    __m512d pr6 = u0r, pi6 = u0i;
    {
    __m512d ar = _mm512_load_pd(x+208), ai = _mm512_load_pd(x+208+8);
    __m512d br = _mm512_load_pd(x+2496), bi = _mm512_load_pd(x+2496+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+0, dr); _mm512_store_pd(dscr+8, di);
    _mm512_store_pd(sscr+0, sr); _mm512_store_pd(sscr+8, si);
    pr1 = _mm512_fmadd_pd(kc1, sr, pr1); pi1 = _mm512_fmadd_pd(kc1, si, pi1);
    pr2 = _mm512_fmadd_pd(kc2, sr, pr2); pi2 = _mm512_fmadd_pd(kc2, si, pi2);
    pr3 = _mm512_fmadd_pd(kc3, sr, pr3); pi3 = _mm512_fmadd_pd(kc3, si, pi3);
    pr4 = _mm512_fmadd_pd(kc4, sr, pr4); pi4 = _mm512_fmadd_pd(kc4, si, pi4);
    pr5 = _mm512_fmadd_pd(kc5, sr, pr5); pi5 = _mm512_fmadd_pd(kc5, si, pi5);
    pr6 = _mm512_fmadd_pd(kc6, sr, pr6); pi6 = _mm512_fmadd_pd(kc6, si, pi6);
    }
    {
    __m512d ar = _mm512_load_pd(x+416), ai = _mm512_load_pd(x+416+8);
    __m512d br = _mm512_load_pd(x+2288), bi = _mm512_load_pd(x+2288+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+16, dr); _mm512_store_pd(dscr+24, di);
    _mm512_store_pd(sscr+16, sr); _mm512_store_pd(sscr+24, si);
    pr1 = _mm512_fmadd_pd(kc2, sr, pr1); pi1 = _mm512_fmadd_pd(kc2, si, pi1);
    pr2 = _mm512_fmadd_pd(kc4, sr, pr2); pi2 = _mm512_fmadd_pd(kc4, si, pi2);
    pr3 = _mm512_fmadd_pd(kc6, sr, pr3); pi3 = _mm512_fmadd_pd(kc6, si, pi3);
    pr4 = _mm512_fmadd_pd(kc5, sr, pr4); pi4 = _mm512_fmadd_pd(kc5, si, pi4);
    pr5 = _mm512_fmadd_pd(kc3, sr, pr5); pi5 = _mm512_fmadd_pd(kc3, si, pi5);
    pr6 = _mm512_fmadd_pd(kc1, sr, pr6); pi6 = _mm512_fmadd_pd(kc1, si, pi6);
    }
    {
    __m512d ar = _mm512_load_pd(x+624), ai = _mm512_load_pd(x+624+8);
    __m512d br = _mm512_load_pd(x+2080), bi = _mm512_load_pd(x+2080+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+32, dr); _mm512_store_pd(dscr+40, di);
    _mm512_store_pd(sscr+32, sr); _mm512_store_pd(sscr+40, si);
    pr1 = _mm512_fmadd_pd(kc3, sr, pr1); pi1 = _mm512_fmadd_pd(kc3, si, pi1);
    pr2 = _mm512_fmadd_pd(kc6, sr, pr2); pi2 = _mm512_fmadd_pd(kc6, si, pi2);
    pr3 = _mm512_fmadd_pd(kc4, sr, pr3); pi3 = _mm512_fmadd_pd(kc4, si, pi3);
    pr4 = _mm512_fmadd_pd(kc1, sr, pr4); pi4 = _mm512_fmadd_pd(kc1, si, pi4);
    pr5 = _mm512_fmadd_pd(kc2, sr, pr5); pi5 = _mm512_fmadd_pd(kc2, si, pi5);
    pr6 = _mm512_fmadd_pd(kc5, sr, pr6); pi6 = _mm512_fmadd_pd(kc5, si, pi6);
    }
    {
    __m512d ar = _mm512_load_pd(x+832), ai = _mm512_load_pd(x+832+8);
    __m512d br = _mm512_load_pd(x+1872), bi = _mm512_load_pd(x+1872+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+48, dr); _mm512_store_pd(dscr+56, di);
    _mm512_store_pd(sscr+48, sr); _mm512_store_pd(sscr+56, si);
    pr1 = _mm512_fmadd_pd(kc4, sr, pr1); pi1 = _mm512_fmadd_pd(kc4, si, pi1);
    pr2 = _mm512_fmadd_pd(kc5, sr, pr2); pi2 = _mm512_fmadd_pd(kc5, si, pi2);
    pr3 = _mm512_fmadd_pd(kc1, sr, pr3); pi3 = _mm512_fmadd_pd(kc1, si, pi3);
    pr4 = _mm512_fmadd_pd(kc3, sr, pr4); pi4 = _mm512_fmadd_pd(kc3, si, pi4);
    pr5 = _mm512_fmadd_pd(kc6, sr, pr5); pi5 = _mm512_fmadd_pd(kc6, si, pi5);
    pr6 = _mm512_fmadd_pd(kc2, sr, pr6); pi6 = _mm512_fmadd_pd(kc2, si, pi6);
    }
    {
    __m512d ar = _mm512_load_pd(x+1040), ai = _mm512_load_pd(x+1040+8);
    __m512d br = _mm512_load_pd(x+1664), bi = _mm512_load_pd(x+1664+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+64, dr); _mm512_store_pd(dscr+72, di);
    _mm512_store_pd(sscr+64, sr); _mm512_store_pd(sscr+72, si);
    pr1 = _mm512_fmadd_pd(kc5, sr, pr1); pi1 = _mm512_fmadd_pd(kc5, si, pi1);
    pr2 = _mm512_fmadd_pd(kc3, sr, pr2); pi2 = _mm512_fmadd_pd(kc3, si, pi2);
    pr3 = _mm512_fmadd_pd(kc2, sr, pr3); pi3 = _mm512_fmadd_pd(kc2, si, pi3);
    pr4 = _mm512_fmadd_pd(kc6, sr, pr4); pi4 = _mm512_fmadd_pd(kc6, si, pi4);
    pr5 = _mm512_fmadd_pd(kc1, sr, pr5); pi5 = _mm512_fmadd_pd(kc1, si, pi5);
    pr6 = _mm512_fmadd_pd(kc4, sr, pr6); pi6 = _mm512_fmadd_pd(kc4, si, pi6);
    }
    {
    __m512d ar = _mm512_load_pd(x+1248), ai = _mm512_load_pd(x+1248+8);
    __m512d br = _mm512_load_pd(x+1456), bi = _mm512_load_pd(x+1456+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+80, dr); _mm512_store_pd(dscr+88, di);
    _mm512_store_pd(sscr+80, sr); _mm512_store_pd(sscr+88, si);
    pr1 = _mm512_fmadd_pd(kc6, sr, pr1); pi1 = _mm512_fmadd_pd(kc6, si, pi1);
    pr2 = _mm512_fmadd_pd(kc1, sr, pr2); pi2 = _mm512_fmadd_pd(kc1, si, pi2);
    pr3 = _mm512_fmadd_pd(kc5, sr, pr3); pi3 = _mm512_fmadd_pd(kc5, si, pi3);
    pr4 = _mm512_fmadd_pd(kc2, sr, pr4); pi4 = _mm512_fmadd_pd(kc2, si, pi4);
    pr5 = _mm512_fmadd_pd(kc4, sr, pr5); pi5 = _mm512_fmadd_pd(kc4, si, pi5);
    pr6 = _mm512_fmadd_pd(kc3, sr, pr6); pi6 = _mm512_fmadd_pd(kc3, si, pi6);
    }
    _mm512_store_pd(pscr+0, pr1); _mm512_store_pd(pscr+8, pi1);
    _mm512_store_pd(pscr+16, pr2); _mm512_store_pd(pscr+24, pi2);
    _mm512_store_pd(pscr+32, pr3); _mm512_store_pd(pscr+40, pi3);
    _mm512_store_pd(pscr+48, pr4); _mm512_store_pd(pscr+56, pi4);
    _mm512_store_pd(pscr+64, pr5); _mm512_store_pd(pscr+72, pi5);
    _mm512_store_pd(pscr+80, pr6); _mm512_store_pd(pscr+88, pi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d x0r = _mm512_load_pd(x), x0i = _mm512_load_pd(x+8);
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+0)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+8));
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+16)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+24));
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+32)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+40));
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+48)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+56));
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+64)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+72));
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+80)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+88));
    _mm512_store_pd(pscr+96, x0r); _mm512_store_pd(pscr+104, x0i);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_13[0]);
    __m512d ks2; BCV(ks2, KS_13[1]);
    __m512d ks3; BCV(ks3, KS_13[2]);
    __m512d ks4; BCV(ks4, KS_13[3]);
    __m512d ks5; BCV(ks5, KS_13[4]);
    __m512d ks6; BCV(ks6, KS_13[5]);
    __m512d qr1, qi1;
    __m512d qr2, qi2;
    __m512d qr3, qi3;
    __m512d qr4, qi4;
    __m512d qr5, qi5;
    __m512d qr6, qi6;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr1 = _mm512_mul_pd(ks1, di); qi1 = _mm512_mul_pd(ks1, dr);
    qr2 = _mm512_mul_pd(ks2, di); qi2 = _mm512_mul_pd(ks2, dr);
    qr3 = _mm512_mul_pd(ks3, di); qi3 = _mm512_mul_pd(ks3, dr);
    qr4 = _mm512_mul_pd(ks4, di); qi4 = _mm512_mul_pd(ks4, dr);
    qr5 = _mm512_mul_pd(ks5, di); qi5 = _mm512_mul_pd(ks5, dr);
    qr6 = _mm512_mul_pd(ks6, di); qi6 = _mm512_mul_pd(ks6, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr1 = _mm512_fmadd_pd(ks2, di, qr1); qi1 = _mm512_fmadd_pd(ks2, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks4, di, qr2); qi2 = _mm512_fmadd_pd(ks4, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks6, di, qr3); qi3 = _mm512_fmadd_pd(ks6, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks5, di, qr4); qi4 = _mm512_fnmadd_pd(ks5, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks3, di, qr5); qi5 = _mm512_fnmadd_pd(ks3, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks1, di, qr6); qi6 = _mm512_fnmadd_pd(ks1, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr1 = _mm512_fmadd_pd(ks3, di, qr1); qi1 = _mm512_fmadd_pd(ks3, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks6, di, qr2); qi2 = _mm512_fmadd_pd(ks6, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks4, di, qr3); qi3 = _mm512_fnmadd_pd(ks4, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks1, di, qr4); qi4 = _mm512_fnmadd_pd(ks1, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks2, di, qr5); qi5 = _mm512_fmadd_pd(ks2, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks5, di, qr6); qi6 = _mm512_fmadd_pd(ks5, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr1 = _mm512_fmadd_pd(ks4, di, qr1); qi1 = _mm512_fmadd_pd(ks4, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks5, di, qr2); qi2 = _mm512_fnmadd_pd(ks5, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks1, di, qr3); qi3 = _mm512_fnmadd_pd(ks1, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks3, di, qr4); qi4 = _mm512_fmadd_pd(ks3, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks6, di, qr5); qi5 = _mm512_fnmadd_pd(ks6, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks2, di, qr6); qi6 = _mm512_fnmadd_pd(ks2, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr1 = _mm512_fmadd_pd(ks5, di, qr1); qi1 = _mm512_fmadd_pd(ks5, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks3, di, qr2); qi2 = _mm512_fnmadd_pd(ks3, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks2, di, qr3); qi3 = _mm512_fmadd_pd(ks2, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks6, di, qr4); qi4 = _mm512_fnmadd_pd(ks6, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks1, di, qr5); qi5 = _mm512_fnmadd_pd(ks1, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks4, di, qr6); qi6 = _mm512_fmadd_pd(ks4, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr1 = _mm512_fmadd_pd(ks6, di, qr1); qi1 = _mm512_fmadd_pd(ks6, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks1, di, qr2); qi2 = _mm512_fnmadd_pd(ks1, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks5, di, qr3); qi3 = _mm512_fmadd_pd(ks5, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks2, di, qr4); qi4 = _mm512_fnmadd_pd(ks2, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks4, di, qr5); qi5 = _mm512_fmadd_pd(ks4, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks3, di, qr6); qi6 = _mm512_fnmadd_pd(ks3, dr, qi6);
    }
    __m512d x0r = _mm512_load_pd(pscr+96), x0i = _mm512_load_pd(pscr+104);
    _mm512_store_pd(d, x0r); _mm512_store_pd(d+8, x0i);
    {
    __m512d Pr = _mm512_load_pd(pscr+0), Pi = _mm512_load_pd(pscr+8);
    __m512d xr = _mm512_add_pd(Pr, qr1), xi = _mm512_sub_pd(Pi, qi1);
    __m512d yr = _mm512_sub_pd(Pr, qr1), yi = _mm512_add_pd(Pi, qi1);
    _mm512_store_pd(d+2704, xr); _mm512_store_pd(d+2704+8, xi);
    _mm512_store_pd(d+32448, yr); _mm512_store_pd(d+32448+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+16), Pi = _mm512_load_pd(pscr+24);
    __m512d xr = _mm512_add_pd(Pr, qr2), xi = _mm512_sub_pd(Pi, qi2);
    __m512d yr = _mm512_sub_pd(Pr, qr2), yi = _mm512_add_pd(Pi, qi2);
    _mm512_store_pd(d+5408, xr); _mm512_store_pd(d+5408+8, xi);
    _mm512_store_pd(d+29744, yr); _mm512_store_pd(d+29744+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+32), Pi = _mm512_load_pd(pscr+40);
    __m512d xr = _mm512_add_pd(Pr, qr3), xi = _mm512_sub_pd(Pi, qi3);
    __m512d yr = _mm512_sub_pd(Pr, qr3), yi = _mm512_add_pd(Pi, qi3);
    _mm512_store_pd(d+8112, xr); _mm512_store_pd(d+8112+8, xi);
    _mm512_store_pd(d+27040, yr); _mm512_store_pd(d+27040+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+48), Pi = _mm512_load_pd(pscr+56);
    __m512d xr = _mm512_add_pd(Pr, qr4), xi = _mm512_sub_pd(Pi, qi4);
    __m512d yr = _mm512_sub_pd(Pr, qr4), yi = _mm512_add_pd(Pi, qi4);
    _mm512_store_pd(d+10816, xr); _mm512_store_pd(d+10816+8, xi);
    _mm512_store_pd(d+24336, yr); _mm512_store_pd(d+24336+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+64), Pi = _mm512_load_pd(pscr+72);
    __m512d xr = _mm512_add_pd(Pr, qr5), xi = _mm512_sub_pd(Pi, qi5);
    __m512d yr = _mm512_sub_pd(Pr, qr5), yi = _mm512_add_pd(Pi, qi5);
    _mm512_store_pd(d+13520, xr); _mm512_store_pd(d+13520+8, xi);
    _mm512_store_pd(d+21632, yr); _mm512_store_pd(d+21632+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+80), Pi = _mm512_load_pd(pscr+88);
    __m512d xr = _mm512_add_pd(Pr, qr6), xi = _mm512_sub_pd(Pi, qi6);
    __m512d yr = _mm512_sub_pd(Pr, qr6), yi = _mm512_add_pd(Pi, qi6);
    _mm512_store_pd(d+16224, xr); _mm512_store_pd(d+16224+8, xi);
    _mm512_store_pd(d+18928, yr); _mm512_store_pd(d+18928+8, yi);
    }
    }
}
static void __attribute__((noinline)) dx_13(double* restrict x, double* restrict d, const double* restrict cb){
    double dscr[12*8] ALIGN64;
    double pscr[14*8] ALIGN64;
    double sscr[12*8] ALIGN64;
    {
    __m512d kc1; BCV(kc1, KC_13[0]);
    __m512d kc2; BCV(kc2, KC_13[1]);
    __m512d kc3; BCV(kc3, KC_13[2]);
    __m512d kc4; BCV(kc4, KC_13[3]);
    __m512d kc5; BCV(kc5, KC_13[4]);
    __m512d kc6; BCV(kc6, KC_13[5]);
    __m512d u0r = _mm512_load_pd(x), u0i = _mm512_load_pd(x+8);
    __m512d pr1 = u0r, pi1 = u0i;
    __m512d pr2 = u0r, pi2 = u0i;
    __m512d pr3 = u0r, pi3 = u0i;
    __m512d pr4 = u0r, pi4 = u0i;
    __m512d pr5 = u0r, pi5 = u0i;
    __m512d pr6 = u0r, pi6 = u0i;
    {
    __m512d ar = _mm512_load_pd(x+16), ai = _mm512_load_pd(x+16+8);
    __m512d br = _mm512_load_pd(x+192), bi = _mm512_load_pd(x+192+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+0, dr); _mm512_store_pd(dscr+8, di);
    _mm512_store_pd(sscr+0, sr); _mm512_store_pd(sscr+8, si);
    pr1 = _mm512_fmadd_pd(kc1, sr, pr1); pi1 = _mm512_fmadd_pd(kc1, si, pi1);
    pr2 = _mm512_fmadd_pd(kc2, sr, pr2); pi2 = _mm512_fmadd_pd(kc2, si, pi2);
    pr3 = _mm512_fmadd_pd(kc3, sr, pr3); pi3 = _mm512_fmadd_pd(kc3, si, pi3);
    pr4 = _mm512_fmadd_pd(kc4, sr, pr4); pi4 = _mm512_fmadd_pd(kc4, si, pi4);
    pr5 = _mm512_fmadd_pd(kc5, sr, pr5); pi5 = _mm512_fmadd_pd(kc5, si, pi5);
    pr6 = _mm512_fmadd_pd(kc6, sr, pr6); pi6 = _mm512_fmadd_pd(kc6, si, pi6);
    }
    {
    __m512d ar = _mm512_load_pd(x+32), ai = _mm512_load_pd(x+32+8);
    __m512d br = _mm512_load_pd(x+176), bi = _mm512_load_pd(x+176+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+16, dr); _mm512_store_pd(dscr+24, di);
    _mm512_store_pd(sscr+16, sr); _mm512_store_pd(sscr+24, si);
    pr1 = _mm512_fmadd_pd(kc2, sr, pr1); pi1 = _mm512_fmadd_pd(kc2, si, pi1);
    pr2 = _mm512_fmadd_pd(kc4, sr, pr2); pi2 = _mm512_fmadd_pd(kc4, si, pi2);
    pr3 = _mm512_fmadd_pd(kc6, sr, pr3); pi3 = _mm512_fmadd_pd(kc6, si, pi3);
    pr4 = _mm512_fmadd_pd(kc5, sr, pr4); pi4 = _mm512_fmadd_pd(kc5, si, pi4);
    pr5 = _mm512_fmadd_pd(kc3, sr, pr5); pi5 = _mm512_fmadd_pd(kc3, si, pi5);
    pr6 = _mm512_fmadd_pd(kc1, sr, pr6); pi6 = _mm512_fmadd_pd(kc1, si, pi6);
    }
    {
    __m512d ar = _mm512_load_pd(x+48), ai = _mm512_load_pd(x+48+8);
    __m512d br = _mm512_load_pd(x+160), bi = _mm512_load_pd(x+160+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+32, dr); _mm512_store_pd(dscr+40, di);
    _mm512_store_pd(sscr+32, sr); _mm512_store_pd(sscr+40, si);
    pr1 = _mm512_fmadd_pd(kc3, sr, pr1); pi1 = _mm512_fmadd_pd(kc3, si, pi1);
    pr2 = _mm512_fmadd_pd(kc6, sr, pr2); pi2 = _mm512_fmadd_pd(kc6, si, pi2);
    pr3 = _mm512_fmadd_pd(kc4, sr, pr3); pi3 = _mm512_fmadd_pd(kc4, si, pi3);
    pr4 = _mm512_fmadd_pd(kc1, sr, pr4); pi4 = _mm512_fmadd_pd(kc1, si, pi4);
    pr5 = _mm512_fmadd_pd(kc2, sr, pr5); pi5 = _mm512_fmadd_pd(kc2, si, pi5);
    pr6 = _mm512_fmadd_pd(kc5, sr, pr6); pi6 = _mm512_fmadd_pd(kc5, si, pi6);
    }
    {
    __m512d ar = _mm512_load_pd(x+64), ai = _mm512_load_pd(x+64+8);
    __m512d br = _mm512_load_pd(x+144), bi = _mm512_load_pd(x+144+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+48, dr); _mm512_store_pd(dscr+56, di);
    _mm512_store_pd(sscr+48, sr); _mm512_store_pd(sscr+56, si);
    pr1 = _mm512_fmadd_pd(kc4, sr, pr1); pi1 = _mm512_fmadd_pd(kc4, si, pi1);
    pr2 = _mm512_fmadd_pd(kc5, sr, pr2); pi2 = _mm512_fmadd_pd(kc5, si, pi2);
    pr3 = _mm512_fmadd_pd(kc1, sr, pr3); pi3 = _mm512_fmadd_pd(kc1, si, pi3);
    pr4 = _mm512_fmadd_pd(kc3, sr, pr4); pi4 = _mm512_fmadd_pd(kc3, si, pi4);
    pr5 = _mm512_fmadd_pd(kc6, sr, pr5); pi5 = _mm512_fmadd_pd(kc6, si, pi5);
    pr6 = _mm512_fmadd_pd(kc2, sr, pr6); pi6 = _mm512_fmadd_pd(kc2, si, pi6);
    }
    {
    __m512d ar = _mm512_load_pd(x+80), ai = _mm512_load_pd(x+80+8);
    __m512d br = _mm512_load_pd(x+128), bi = _mm512_load_pd(x+128+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+64, dr); _mm512_store_pd(dscr+72, di);
    _mm512_store_pd(sscr+64, sr); _mm512_store_pd(sscr+72, si);
    pr1 = _mm512_fmadd_pd(kc5, sr, pr1); pi1 = _mm512_fmadd_pd(kc5, si, pi1);
    pr2 = _mm512_fmadd_pd(kc3, sr, pr2); pi2 = _mm512_fmadd_pd(kc3, si, pi2);
    pr3 = _mm512_fmadd_pd(kc2, sr, pr3); pi3 = _mm512_fmadd_pd(kc2, si, pi3);
    pr4 = _mm512_fmadd_pd(kc6, sr, pr4); pi4 = _mm512_fmadd_pd(kc6, si, pi4);
    pr5 = _mm512_fmadd_pd(kc1, sr, pr5); pi5 = _mm512_fmadd_pd(kc1, si, pi5);
    pr6 = _mm512_fmadd_pd(kc4, sr, pr6); pi6 = _mm512_fmadd_pd(kc4, si, pi6);
    }
    {
    __m512d ar = _mm512_load_pd(x+96), ai = _mm512_load_pd(x+96+8);
    __m512d br = _mm512_load_pd(x+112), bi = _mm512_load_pd(x+112+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(dscr+80, dr); _mm512_store_pd(dscr+88, di);
    _mm512_store_pd(sscr+80, sr); _mm512_store_pd(sscr+88, si);
    pr1 = _mm512_fmadd_pd(kc6, sr, pr1); pi1 = _mm512_fmadd_pd(kc6, si, pi1);
    pr2 = _mm512_fmadd_pd(kc1, sr, pr2); pi2 = _mm512_fmadd_pd(kc1, si, pi2);
    pr3 = _mm512_fmadd_pd(kc5, sr, pr3); pi3 = _mm512_fmadd_pd(kc5, si, pi3);
    pr4 = _mm512_fmadd_pd(kc2, sr, pr4); pi4 = _mm512_fmadd_pd(kc2, si, pi4);
    pr5 = _mm512_fmadd_pd(kc4, sr, pr5); pi5 = _mm512_fmadd_pd(kc4, si, pi5);
    pr6 = _mm512_fmadd_pd(kc3, sr, pr6); pi6 = _mm512_fmadd_pd(kc3, si, pi6);
    }
    _mm512_store_pd(pscr+0, pr1); _mm512_store_pd(pscr+8, pi1);
    _mm512_store_pd(pscr+16, pr2); _mm512_store_pd(pscr+24, pi2);
    _mm512_store_pd(pscr+32, pr3); _mm512_store_pd(pscr+40, pi3);
    _mm512_store_pd(pscr+48, pr4); _mm512_store_pd(pscr+56, pi4);
    _mm512_store_pd(pscr+64, pr5); _mm512_store_pd(pscr+72, pi5);
    _mm512_store_pd(pscr+80, pr6); _mm512_store_pd(pscr+88, pi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d x0r = _mm512_load_pd(x), x0i = _mm512_load_pd(x+8);
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+0)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+8));
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+16)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+24));
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+32)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+40));
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+48)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+56));
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+64)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+72));
    x0r = _mm512_add_pd(x0r, _mm512_load_pd(sscr+80)); x0i = _mm512_add_pd(x0i, _mm512_load_pd(sscr+88));
    _mm512_store_pd(pscr+96, x0r); _mm512_store_pd(pscr+104, x0i);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_13[0]);
    __m512d ks2; BCV(ks2, KS_13[1]);
    __m512d ks3; BCV(ks3, KS_13[2]);
    __m512d ks4; BCV(ks4, KS_13[3]);
    __m512d ks5; BCV(ks5, KS_13[4]);
    __m512d ks6; BCV(ks6, KS_13[5]);
    __m512d qr1, qi1;
    __m512d qr2, qi2;
    __m512d qr3, qi3;
    __m512d qr4, qi4;
    __m512d qr5, qi5;
    __m512d qr6, qi6;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr1 = _mm512_mul_pd(ks1, di); qi1 = _mm512_mul_pd(ks1, dr);
    qr2 = _mm512_mul_pd(ks2, di); qi2 = _mm512_mul_pd(ks2, dr);
    qr3 = _mm512_mul_pd(ks3, di); qi3 = _mm512_mul_pd(ks3, dr);
    qr4 = _mm512_mul_pd(ks4, di); qi4 = _mm512_mul_pd(ks4, dr);
    qr5 = _mm512_mul_pd(ks5, di); qi5 = _mm512_mul_pd(ks5, dr);
    qr6 = _mm512_mul_pd(ks6, di); qi6 = _mm512_mul_pd(ks6, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr1 = _mm512_fmadd_pd(ks2, di, qr1); qi1 = _mm512_fmadd_pd(ks2, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks4, di, qr2); qi2 = _mm512_fmadd_pd(ks4, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks6, di, qr3); qi3 = _mm512_fmadd_pd(ks6, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks5, di, qr4); qi4 = _mm512_fnmadd_pd(ks5, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks3, di, qr5); qi5 = _mm512_fnmadd_pd(ks3, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks1, di, qr6); qi6 = _mm512_fnmadd_pd(ks1, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr1 = _mm512_fmadd_pd(ks3, di, qr1); qi1 = _mm512_fmadd_pd(ks3, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks6, di, qr2); qi2 = _mm512_fmadd_pd(ks6, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks4, di, qr3); qi3 = _mm512_fnmadd_pd(ks4, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks1, di, qr4); qi4 = _mm512_fnmadd_pd(ks1, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks2, di, qr5); qi5 = _mm512_fmadd_pd(ks2, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks5, di, qr6); qi6 = _mm512_fmadd_pd(ks5, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr1 = _mm512_fmadd_pd(ks4, di, qr1); qi1 = _mm512_fmadd_pd(ks4, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks5, di, qr2); qi2 = _mm512_fnmadd_pd(ks5, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks1, di, qr3); qi3 = _mm512_fnmadd_pd(ks1, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks3, di, qr4); qi4 = _mm512_fmadd_pd(ks3, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks6, di, qr5); qi5 = _mm512_fnmadd_pd(ks6, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks2, di, qr6); qi6 = _mm512_fnmadd_pd(ks2, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr1 = _mm512_fmadd_pd(ks5, di, qr1); qi1 = _mm512_fmadd_pd(ks5, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks3, di, qr2); qi2 = _mm512_fnmadd_pd(ks3, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks2, di, qr3); qi3 = _mm512_fmadd_pd(ks2, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks6, di, qr4); qi4 = _mm512_fnmadd_pd(ks6, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks1, di, qr5); qi5 = _mm512_fnmadd_pd(ks1, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks4, di, qr6); qi6 = _mm512_fmadd_pd(ks4, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr1 = _mm512_fmadd_pd(ks6, di, qr1); qi1 = _mm512_fmadd_pd(ks6, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks1, di, qr2); qi2 = _mm512_fnmadd_pd(ks1, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks5, di, qr3); qi3 = _mm512_fmadd_pd(ks5, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks2, di, qr4); qi4 = _mm512_fnmadd_pd(ks2, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks4, di, qr5); qi5 = _mm512_fmadd_pd(ks4, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks3, di, qr6); qi6 = _mm512_fnmadd_pd(ks3, dr, qi6);
    }
    __m512d x0r = _mm512_load_pd(pscr+96), x0i = _mm512_load_pd(pscr+104);
    MAPST(x0r, x0i, d, 0, cb, 0);
    {
    __m512d Pr = _mm512_load_pd(pscr+0), Pi = _mm512_load_pd(pscr+8);
    __m512d xr = _mm512_add_pd(Pr, qr1), xi = _mm512_sub_pd(Pi, qi1);
    __m512d yr = _mm512_sub_pd(Pr, qr1), yi = _mm512_add_pd(Pi, qi1);
    MAPST(xr, xi, d, 2704, cb, 16);
    MAPST(yr, yi, d, 32448, cb, 192);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+16), Pi = _mm512_load_pd(pscr+24);
    __m512d xr = _mm512_add_pd(Pr, qr2), xi = _mm512_sub_pd(Pi, qi2);
    __m512d yr = _mm512_sub_pd(Pr, qr2), yi = _mm512_add_pd(Pi, qi2);
    MAPST(xr, xi, d, 5408, cb, 32);
    MAPST(yr, yi, d, 29744, cb, 176);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+32), Pi = _mm512_load_pd(pscr+40);
    __m512d xr = _mm512_add_pd(Pr, qr3), xi = _mm512_sub_pd(Pi, qi3);
    __m512d yr = _mm512_sub_pd(Pr, qr3), yi = _mm512_add_pd(Pi, qi3);
    MAPST(xr, xi, d, 8112, cb, 48);
    MAPST(yr, yi, d, 27040, cb, 160);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+48), Pi = _mm512_load_pd(pscr+56);
    __m512d xr = _mm512_add_pd(Pr, qr4), xi = _mm512_sub_pd(Pi, qi4);
    __m512d yr = _mm512_sub_pd(Pr, qr4), yi = _mm512_add_pd(Pi, qi4);
    MAPST(xr, xi, d, 10816, cb, 64);
    MAPST(yr, yi, d, 24336, cb, 144);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+64), Pi = _mm512_load_pd(pscr+72);
    __m512d xr = _mm512_add_pd(Pr, qr5), xi = _mm512_sub_pd(Pi, qi5);
    __m512d yr = _mm512_sub_pd(Pr, qr5), yi = _mm512_add_pd(Pi, qi5);
    MAPST(xr, xi, d, 13520, cb, 80);
    MAPST(yr, yi, d, 21632, cb, 128);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+80), Pi = _mm512_load_pd(pscr+88);
    __m512d xr = _mm512_add_pd(Pr, qr6), xi = _mm512_sub_pd(Pi, qi6);
    __m512d yr = _mm512_sub_pd(Pr, qr6), yi = _mm512_add_pd(Pi, qi6);
    MAPST(xr, xi, d, 16224, cb, 96);
    MAPST(yr, yi, d, 18928, cb, 112);
    }
    }
}

static void step2_13(double* restrict G, double* restrict G2, const double* restrict CP){
    for(int x=0; x<13; x++){
        double* pl = G + (long)x*169*16;
        for(int y=0; y<13; y++) dz_13(pl + (long)y*13*16);
        for(int z=0; z<13; z++) dy_13(pl + (long)z*16, G2 + ((long)z*13 + x)*16);
    }
    for(int e=0; e<169; e++)
        dx_13(G2 + (long)e*13*16, G + (long)e*16, CP + (long)e*13*16);
}


void run2_13(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if(!G_13){ G_13 = alloc_arena(2197*16*8); G2_13 = alloc_arena(2197*16*8 + 4096) + 256; CP_13 = alloc_arena(2197*16*8 + 65536) + 128; CT_13 = alloc_arena(2197*16*8); }
    long G8 = B/8;
    for(long g=0; g<G8; g++){
        const double* sx[8]; const double* sc[8]; double* d1[8]; double* dm[8];
        for(int v=0; v<8; v++){
            long off = (g*8+v)*(long)2197*2;
            sx[v] = x0+off; sc[v] = c+off; d1[v] = out1+off; dm[v] = outm+off;
        }
        conv_in_13(sx, G_13);
        conv_in_13(sc, CT_13);
        for(long e=0; e<169; e++)
            for(int j=0; j<13; j++)
                memcpy(CP_13 + (e*(long)13 + j)*16, CT_13 + ((long)j*169 + e)*16, 128);
        for(long t=0; t<m; t++){
            step2_13(G_13, G2_13, CP_13);
            if(t==0 && m>1) conv_out_13(G_13, d1);
        }
        conv_out_13(G_13, dm);
        if(m==1) for(int v=0; v<8; v++) memcpy(d1[v], dm[v], 2197*16);
    }
}


static void conv_in_17(const double* const* src, double* restrict G){
    for(long e=0; e+4<=4913; e+=4){
        __m512d r0=_mm512_loadu_pd(src[0]+2*e), r1=_mm512_loadu_pd(src[1]+2*e);
        __m512d r2=_mm512_loadu_pd(src[2]+2*e), r3=_mm512_loadu_pd(src[3]+2*e);
        __m512d r4=_mm512_loadu_pd(src[4]+2*e), r5=_mm512_loadu_pd(src[5]+2*e);
        __m512d r6=_mm512_loadu_pd(src[6]+2*e), r7=_mm512_loadu_pd(src[7]+2*e);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        double* g = G + e*16;
        _mm512_store_pd(g,o0); _mm512_store_pd(g+8,o1);
        _mm512_store_pd(g+16,o2); _mm512_store_pd(g+24,o3);
        _mm512_store_pd(g+32,o4); _mm512_store_pd(g+40,o5);
        _mm512_store_pd(g+48,o6); _mm512_store_pd(g+56,o7);
    }
    for(long e=4912; e<4913; e++)
        for(int v=0; v<8; v++){ G[e*16+v] = src[v][2*e]; G[e*16+8+v] = src[v][2*e+1]; }
}
static void conv_out_17(const double* restrict G, double* const* dst){
    for(long e=0; e+4<=4913; e+=4){
        const double* g = G + e*16;
        __m512d r0=_mm512_load_pd(g),    r1=_mm512_load_pd(g+8);
        __m512d r2=_mm512_load_pd(g+16), r3=_mm512_load_pd(g+24);
        __m512d r4=_mm512_load_pd(g+32), r5=_mm512_load_pd(g+40);
        __m512d r6=_mm512_load_pd(g+48), r7=_mm512_load_pd(g+56);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        _mm512_storeu_pd(dst[0]+2*e,o0); _mm512_storeu_pd(dst[1]+2*e,o1);
        _mm512_storeu_pd(dst[2]+2*e,o2); _mm512_storeu_pd(dst[3]+2*e,o3);
        _mm512_storeu_pd(dst[4]+2*e,o4); _mm512_storeu_pd(dst[5]+2*e,o5);
        _mm512_storeu_pd(dst[6]+2*e,o6); _mm512_storeu_pd(dst[7]+2*e,o7);
    }
    for(long e=4912; e<4913; e++)
        for(int v=0; v<8; v++){ dst[v][2*e] = G[e*16+v]; dst[v][2*e+1] = G[e*16+8+v]; }
}

static double* G_17 = 0;
static double* G2_17 = 0;
static double* CP_17 = 0;
static double* CT_17 = 0;
static const double KC_17[8] ALIGN64 = { 0x1.dd6d000370991p-1, 0x1.7a5f6075d4884p-1, 0x1.c86fa2b2883cep-2, 0x1.79ee63259b75fp-4, -0x1.183b1c61f0d01p-2, -0x1.348c86ed5f1bap-1, -0x1.b34fa910ea3b8p-1, -0x1.f7484007faef3p-1 };
static const double KS_17[8] ALIGN64 = { 0x1.71e955d8e7cdcp-2, 0x1.58eea2a9d6da3p-1, 0x1.ca52d7c9e640bp-1, 0x1.fdd0deb564b22p-1, 0x1.ec746923c349fp-1, 0x1.9895b6c9a05f7p-1, 0x1.0d8884363dd82p-1, 0x1.7851aacd6c6b5p-3 };

static void __attribute__((noinline)) dz_17(double* restrict x){
    double sscr[16*8] ALIGN64;
    double dscr[16*8] ALIGN64;
    double pscr[16*8+16] ALIGN64;
    double qscr[16*8] ALIGN64;
    {
    __m512d u0r = _mm512_load_pd(x), u0i = _mm512_load_pd(x+8);
    __m512d x0r = u0r, x0i = u0i;
    {
    __m512d ar = _mm512_load_pd(x+16), ai = _mm512_load_pd(x+16+8);
    __m512d br = _mm512_load_pd(x+256), bi = _mm512_load_pd(x+256+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+0, sr); _mm512_store_pd(sscr+8, si);
    _mm512_store_pd(dscr+0, dr); _mm512_store_pd(dscr+8, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+32), ai = _mm512_load_pd(x+32+8);
    __m512d br = _mm512_load_pd(x+240), bi = _mm512_load_pd(x+240+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+16, sr); _mm512_store_pd(sscr+24, si);
    _mm512_store_pd(dscr+16, dr); _mm512_store_pd(dscr+24, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+48), ai = _mm512_load_pd(x+48+8);
    __m512d br = _mm512_load_pd(x+224), bi = _mm512_load_pd(x+224+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+32, sr); _mm512_store_pd(sscr+40, si);
    _mm512_store_pd(dscr+32, dr); _mm512_store_pd(dscr+40, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+64), ai = _mm512_load_pd(x+64+8);
    __m512d br = _mm512_load_pd(x+208), bi = _mm512_load_pd(x+208+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+48, sr); _mm512_store_pd(sscr+56, si);
    _mm512_store_pd(dscr+48, dr); _mm512_store_pd(dscr+56, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+80), ai = _mm512_load_pd(x+80+8);
    __m512d br = _mm512_load_pd(x+192), bi = _mm512_load_pd(x+192+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+64, sr); _mm512_store_pd(sscr+72, si);
    _mm512_store_pd(dscr+64, dr); _mm512_store_pd(dscr+72, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+96), ai = _mm512_load_pd(x+96+8);
    __m512d br = _mm512_load_pd(x+176), bi = _mm512_load_pd(x+176+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+80, sr); _mm512_store_pd(sscr+88, si);
    _mm512_store_pd(dscr+80, dr); _mm512_store_pd(dscr+88, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+112), ai = _mm512_load_pd(x+112+8);
    __m512d br = _mm512_load_pd(x+160), bi = _mm512_load_pd(x+160+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+96, sr); _mm512_store_pd(sscr+104, si);
    _mm512_store_pd(dscr+96, dr); _mm512_store_pd(dscr+104, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+128), ai = _mm512_load_pd(x+128+8);
    __m512d br = _mm512_load_pd(x+144), bi = _mm512_load_pd(x+144+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+112, sr); _mm512_store_pd(sscr+120, si);
    _mm512_store_pd(dscr+112, dr); _mm512_store_pd(dscr+120, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    _mm512_store_pd(x, x0r); _mm512_store_pd(x+8, x0i);
    __m512d u0r_s = u0r, u0i_s = u0i;
    _mm512_store_pd(pscr, u0r_s); _mm512_store_pd(pscr+8, u0i_s);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d u0r = _mm512_load_pd(x+0*0), u0i;
    u0r = _mm512_load_pd(pscr); u0i = _mm512_load_pd(pscr+8);
    __m512d kc1; BCV(kc1, KC_17[0]);
    __m512d kc2; BCV(kc2, KC_17[1]);
    __m512d kc3; BCV(kc3, KC_17[2]);
    __m512d kc4; BCV(kc4, KC_17[3]);
    __m512d kc5; BCV(kc5, KC_17[4]);
    __m512d kc6; BCV(kc6, KC_17[5]);
    __m512d kc7; BCV(kc7, KC_17[6]);
    __m512d kc8; BCV(kc8, KC_17[7]);
    __m512d pr1 = u0r, pi1 = u0i;
    __m512d pr2 = u0r, pi2 = u0i;
    __m512d pr3 = u0r, pi3 = u0i;
    __m512d pr4 = u0r, pi4 = u0i;
    __m512d pr5 = u0r, pi5 = u0i;
    __m512d pr6 = u0r, pi6 = u0i;
    {
    __m512d sr = _mm512_load_pd(sscr+0), si = _mm512_load_pd(sscr+8);
    pr1 = _mm512_fmadd_pd(kc1, sr, pr1); pi1 = _mm512_fmadd_pd(kc1, si, pi1);
    pr2 = _mm512_fmadd_pd(kc2, sr, pr2); pi2 = _mm512_fmadd_pd(kc2, si, pi2);
    pr3 = _mm512_fmadd_pd(kc3, sr, pr3); pi3 = _mm512_fmadd_pd(kc3, si, pi3);
    pr4 = _mm512_fmadd_pd(kc4, sr, pr4); pi4 = _mm512_fmadd_pd(kc4, si, pi4);
    pr5 = _mm512_fmadd_pd(kc5, sr, pr5); pi5 = _mm512_fmadd_pd(kc5, si, pi5);
    pr6 = _mm512_fmadd_pd(kc6, sr, pr6); pi6 = _mm512_fmadd_pd(kc6, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+16), si = _mm512_load_pd(sscr+24);
    pr1 = _mm512_fmadd_pd(kc2, sr, pr1); pi1 = _mm512_fmadd_pd(kc2, si, pi1);
    pr2 = _mm512_fmadd_pd(kc4, sr, pr2); pi2 = _mm512_fmadd_pd(kc4, si, pi2);
    pr3 = _mm512_fmadd_pd(kc6, sr, pr3); pi3 = _mm512_fmadd_pd(kc6, si, pi3);
    pr4 = _mm512_fmadd_pd(kc8, sr, pr4); pi4 = _mm512_fmadd_pd(kc8, si, pi4);
    pr5 = _mm512_fmadd_pd(kc7, sr, pr5); pi5 = _mm512_fmadd_pd(kc7, si, pi5);
    pr6 = _mm512_fmadd_pd(kc5, sr, pr6); pi6 = _mm512_fmadd_pd(kc5, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+32), si = _mm512_load_pd(sscr+40);
    pr1 = _mm512_fmadd_pd(kc3, sr, pr1); pi1 = _mm512_fmadd_pd(kc3, si, pi1);
    pr2 = _mm512_fmadd_pd(kc6, sr, pr2); pi2 = _mm512_fmadd_pd(kc6, si, pi2);
    pr3 = _mm512_fmadd_pd(kc8, sr, pr3); pi3 = _mm512_fmadd_pd(kc8, si, pi3);
    pr4 = _mm512_fmadd_pd(kc5, sr, pr4); pi4 = _mm512_fmadd_pd(kc5, si, pi4);
    pr5 = _mm512_fmadd_pd(kc2, sr, pr5); pi5 = _mm512_fmadd_pd(kc2, si, pi5);
    pr6 = _mm512_fmadd_pd(kc1, sr, pr6); pi6 = _mm512_fmadd_pd(kc1, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+48), si = _mm512_load_pd(sscr+56);
    pr1 = _mm512_fmadd_pd(kc4, sr, pr1); pi1 = _mm512_fmadd_pd(kc4, si, pi1);
    pr2 = _mm512_fmadd_pd(kc8, sr, pr2); pi2 = _mm512_fmadd_pd(kc8, si, pi2);
    pr3 = _mm512_fmadd_pd(kc5, sr, pr3); pi3 = _mm512_fmadd_pd(kc5, si, pi3);
    pr4 = _mm512_fmadd_pd(kc1, sr, pr4); pi4 = _mm512_fmadd_pd(kc1, si, pi4);
    pr5 = _mm512_fmadd_pd(kc3, sr, pr5); pi5 = _mm512_fmadd_pd(kc3, si, pi5);
    pr6 = _mm512_fmadd_pd(kc7, sr, pr6); pi6 = _mm512_fmadd_pd(kc7, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+64), si = _mm512_load_pd(sscr+72);
    pr1 = _mm512_fmadd_pd(kc5, sr, pr1); pi1 = _mm512_fmadd_pd(kc5, si, pi1);
    pr2 = _mm512_fmadd_pd(kc7, sr, pr2); pi2 = _mm512_fmadd_pd(kc7, si, pi2);
    pr3 = _mm512_fmadd_pd(kc2, sr, pr3); pi3 = _mm512_fmadd_pd(kc2, si, pi3);
    pr4 = _mm512_fmadd_pd(kc3, sr, pr4); pi4 = _mm512_fmadd_pd(kc3, si, pi4);
    pr5 = _mm512_fmadd_pd(kc8, sr, pr5); pi5 = _mm512_fmadd_pd(kc8, si, pi5);
    pr6 = _mm512_fmadd_pd(kc4, sr, pr6); pi6 = _mm512_fmadd_pd(kc4, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+80), si = _mm512_load_pd(sscr+88);
    pr1 = _mm512_fmadd_pd(kc6, sr, pr1); pi1 = _mm512_fmadd_pd(kc6, si, pi1);
    pr2 = _mm512_fmadd_pd(kc5, sr, pr2); pi2 = _mm512_fmadd_pd(kc5, si, pi2);
    pr3 = _mm512_fmadd_pd(kc1, sr, pr3); pi3 = _mm512_fmadd_pd(kc1, si, pi3);
    pr4 = _mm512_fmadd_pd(kc7, sr, pr4); pi4 = _mm512_fmadd_pd(kc7, si, pi4);
    pr5 = _mm512_fmadd_pd(kc4, sr, pr5); pi5 = _mm512_fmadd_pd(kc4, si, pi5);
    pr6 = _mm512_fmadd_pd(kc2, sr, pr6); pi6 = _mm512_fmadd_pd(kc2, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+96), si = _mm512_load_pd(sscr+104);
    pr1 = _mm512_fmadd_pd(kc7, sr, pr1); pi1 = _mm512_fmadd_pd(kc7, si, pi1);
    pr2 = _mm512_fmadd_pd(kc3, sr, pr2); pi2 = _mm512_fmadd_pd(kc3, si, pi2);
    pr3 = _mm512_fmadd_pd(kc4, sr, pr3); pi3 = _mm512_fmadd_pd(kc4, si, pi3);
    pr4 = _mm512_fmadd_pd(kc6, sr, pr4); pi4 = _mm512_fmadd_pd(kc6, si, pi4);
    pr5 = _mm512_fmadd_pd(kc1, sr, pr5); pi5 = _mm512_fmadd_pd(kc1, si, pi5);
    pr6 = _mm512_fmadd_pd(kc8, sr, pr6); pi6 = _mm512_fmadd_pd(kc8, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+112), si = _mm512_load_pd(sscr+120);
    pr1 = _mm512_fmadd_pd(kc8, sr, pr1); pi1 = _mm512_fmadd_pd(kc8, si, pi1);
    pr2 = _mm512_fmadd_pd(kc1, sr, pr2); pi2 = _mm512_fmadd_pd(kc1, si, pi2);
    pr3 = _mm512_fmadd_pd(kc7, sr, pr3); pi3 = _mm512_fmadd_pd(kc7, si, pi3);
    pr4 = _mm512_fmadd_pd(kc2, sr, pr4); pi4 = _mm512_fmadd_pd(kc2, si, pi4);
    pr5 = _mm512_fmadd_pd(kc6, sr, pr5); pi5 = _mm512_fmadd_pd(kc6, si, pi5);
    pr6 = _mm512_fmadd_pd(kc3, sr, pr6); pi6 = _mm512_fmadd_pd(kc3, si, pi6);
    }
    _mm512_store_pd(pscr+16, pr1); _mm512_store_pd(pscr+24, pi1);
    _mm512_store_pd(pscr+32, pr2); _mm512_store_pd(pscr+40, pi2);
    _mm512_store_pd(pscr+48, pr3); _mm512_store_pd(pscr+56, pi3);
    _mm512_store_pd(pscr+64, pr4); _mm512_store_pd(pscr+72, pi4);
    _mm512_store_pd(pscr+80, pr5); _mm512_store_pd(pscr+88, pi5);
    _mm512_store_pd(pscr+96, pr6); _mm512_store_pd(pscr+104, pi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d u0r = _mm512_load_pd(x+0*0), u0i;
    u0r = _mm512_load_pd(pscr); u0i = _mm512_load_pd(pscr+8);
    __m512d kc1; BCV(kc1, KC_17[0]);
    __m512d kc2; BCV(kc2, KC_17[1]);
    __m512d kc3; BCV(kc3, KC_17[2]);
    __m512d kc4; BCV(kc4, KC_17[3]);
    __m512d kc5; BCV(kc5, KC_17[4]);
    __m512d kc6; BCV(kc6, KC_17[5]);
    __m512d kc7; BCV(kc7, KC_17[6]);
    __m512d kc8; BCV(kc8, KC_17[7]);
    __m512d pr7 = u0r, pi7 = u0i;
    __m512d pr8 = u0r, pi8 = u0i;
    {
    __m512d sr = _mm512_load_pd(sscr+0), si = _mm512_load_pd(sscr+8);
    pr7 = _mm512_fmadd_pd(kc7, sr, pr7); pi7 = _mm512_fmadd_pd(kc7, si, pi7);
    pr8 = _mm512_fmadd_pd(kc8, sr, pr8); pi8 = _mm512_fmadd_pd(kc8, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+16), si = _mm512_load_pd(sscr+24);
    pr7 = _mm512_fmadd_pd(kc3, sr, pr7); pi7 = _mm512_fmadd_pd(kc3, si, pi7);
    pr8 = _mm512_fmadd_pd(kc1, sr, pr8); pi8 = _mm512_fmadd_pd(kc1, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+32), si = _mm512_load_pd(sscr+40);
    pr7 = _mm512_fmadd_pd(kc4, sr, pr7); pi7 = _mm512_fmadd_pd(kc4, si, pi7);
    pr8 = _mm512_fmadd_pd(kc7, sr, pr8); pi8 = _mm512_fmadd_pd(kc7, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+48), si = _mm512_load_pd(sscr+56);
    pr7 = _mm512_fmadd_pd(kc6, sr, pr7); pi7 = _mm512_fmadd_pd(kc6, si, pi7);
    pr8 = _mm512_fmadd_pd(kc2, sr, pr8); pi8 = _mm512_fmadd_pd(kc2, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+64), si = _mm512_load_pd(sscr+72);
    pr7 = _mm512_fmadd_pd(kc1, sr, pr7); pi7 = _mm512_fmadd_pd(kc1, si, pi7);
    pr8 = _mm512_fmadd_pd(kc6, sr, pr8); pi8 = _mm512_fmadd_pd(kc6, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+80), si = _mm512_load_pd(sscr+88);
    pr7 = _mm512_fmadd_pd(kc8, sr, pr7); pi7 = _mm512_fmadd_pd(kc8, si, pi7);
    pr8 = _mm512_fmadd_pd(kc3, sr, pr8); pi8 = _mm512_fmadd_pd(kc3, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+96), si = _mm512_load_pd(sscr+104);
    pr7 = _mm512_fmadd_pd(kc2, sr, pr7); pi7 = _mm512_fmadd_pd(kc2, si, pi7);
    pr8 = _mm512_fmadd_pd(kc5, sr, pr8); pi8 = _mm512_fmadd_pd(kc5, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+112), si = _mm512_load_pd(sscr+120);
    pr7 = _mm512_fmadd_pd(kc5, sr, pr7); pi7 = _mm512_fmadd_pd(kc5, si, pi7);
    pr8 = _mm512_fmadd_pd(kc4, sr, pr8); pi8 = _mm512_fmadd_pd(kc4, si, pi8);
    }
    _mm512_store_pd(pscr+112, pr7); _mm512_store_pd(pscr+120, pi7);
    _mm512_store_pd(pscr+128, pr8); _mm512_store_pd(pscr+136, pi8);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_17[0]);
    __m512d ks2; BCV(ks2, KS_17[1]);
    __m512d ks3; BCV(ks3, KS_17[2]);
    __m512d ks4; BCV(ks4, KS_17[3]);
    __m512d ks5; BCV(ks5, KS_17[4]);
    __m512d ks6; BCV(ks6, KS_17[5]);
    __m512d ks7; BCV(ks7, KS_17[6]);
    __m512d ks8; BCV(ks8, KS_17[7]);
    __m512d qr1, qi1;
    __m512d qr2, qi2;
    __m512d qr3, qi3;
    __m512d qr4, qi4;
    __m512d qr5, qi5;
    __m512d qr6, qi6;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr1 = _mm512_mul_pd(ks1, di); qi1 = _mm512_mul_pd(ks1, dr);
    qr2 = _mm512_mul_pd(ks2, di); qi2 = _mm512_mul_pd(ks2, dr);
    qr3 = _mm512_mul_pd(ks3, di); qi3 = _mm512_mul_pd(ks3, dr);
    qr4 = _mm512_mul_pd(ks4, di); qi4 = _mm512_mul_pd(ks4, dr);
    qr5 = _mm512_mul_pd(ks5, di); qi5 = _mm512_mul_pd(ks5, dr);
    qr6 = _mm512_mul_pd(ks6, di); qi6 = _mm512_mul_pd(ks6, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr1 = _mm512_fmadd_pd(ks2, di, qr1); qi1 = _mm512_fmadd_pd(ks2, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks4, di, qr2); qi2 = _mm512_fmadd_pd(ks4, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks6, di, qr3); qi3 = _mm512_fmadd_pd(ks6, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks8, di, qr4); qi4 = _mm512_fmadd_pd(ks8, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks7, di, qr5); qi5 = _mm512_fnmadd_pd(ks7, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks5, di, qr6); qi6 = _mm512_fnmadd_pd(ks5, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr1 = _mm512_fmadd_pd(ks3, di, qr1); qi1 = _mm512_fmadd_pd(ks3, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks6, di, qr2); qi2 = _mm512_fmadd_pd(ks6, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks8, di, qr3); qi3 = _mm512_fnmadd_pd(ks8, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks5, di, qr4); qi4 = _mm512_fnmadd_pd(ks5, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks2, di, qr5); qi5 = _mm512_fnmadd_pd(ks2, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks1, di, qr6); qi6 = _mm512_fmadd_pd(ks1, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr1 = _mm512_fmadd_pd(ks4, di, qr1); qi1 = _mm512_fmadd_pd(ks4, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks8, di, qr2); qi2 = _mm512_fmadd_pd(ks8, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks5, di, qr3); qi3 = _mm512_fnmadd_pd(ks5, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks1, di, qr4); qi4 = _mm512_fnmadd_pd(ks1, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks3, di, qr5); qi5 = _mm512_fmadd_pd(ks3, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks7, di, qr6); qi6 = _mm512_fmadd_pd(ks7, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr1 = _mm512_fmadd_pd(ks5, di, qr1); qi1 = _mm512_fmadd_pd(ks5, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks7, di, qr2); qi2 = _mm512_fnmadd_pd(ks7, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks2, di, qr3); qi3 = _mm512_fnmadd_pd(ks2, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks3, di, qr4); qi4 = _mm512_fmadd_pd(ks3, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks8, di, qr5); qi5 = _mm512_fmadd_pd(ks8, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks4, di, qr6); qi6 = _mm512_fnmadd_pd(ks4, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr1 = _mm512_fmadd_pd(ks6, di, qr1); qi1 = _mm512_fmadd_pd(ks6, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks5, di, qr2); qi2 = _mm512_fnmadd_pd(ks5, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks1, di, qr3); qi3 = _mm512_fmadd_pd(ks1, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks7, di, qr4); qi4 = _mm512_fmadd_pd(ks7, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks4, di, qr5); qi5 = _mm512_fnmadd_pd(ks4, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks2, di, qr6); qi6 = _mm512_fmadd_pd(ks2, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+96), di = _mm512_load_pd(dscr+104);
    qr1 = _mm512_fmadd_pd(ks7, di, qr1); qi1 = _mm512_fmadd_pd(ks7, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks3, di, qr2); qi2 = _mm512_fnmadd_pd(ks3, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks4, di, qr3); qi3 = _mm512_fmadd_pd(ks4, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks6, di, qr4); qi4 = _mm512_fnmadd_pd(ks6, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks1, di, qr5); qi5 = _mm512_fmadd_pd(ks1, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks8, di, qr6); qi6 = _mm512_fmadd_pd(ks8, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+112), di = _mm512_load_pd(dscr+120);
    qr1 = _mm512_fmadd_pd(ks8, di, qr1); qi1 = _mm512_fmadd_pd(ks8, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks1, di, qr2); qi2 = _mm512_fnmadd_pd(ks1, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks7, di, qr3); qi3 = _mm512_fmadd_pd(ks7, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks2, di, qr4); qi4 = _mm512_fnmadd_pd(ks2, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks6, di, qr5); qi5 = _mm512_fmadd_pd(ks6, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks3, di, qr6); qi6 = _mm512_fnmadd_pd(ks3, dr, qi6);
    }
    _mm512_store_pd(qscr+0, qr1); _mm512_store_pd(qscr+8, qi1);
    _mm512_store_pd(qscr+16, qr2); _mm512_store_pd(qscr+24, qi2);
    _mm512_store_pd(qscr+32, qr3); _mm512_store_pd(qscr+40, qi3);
    _mm512_store_pd(qscr+48, qr4); _mm512_store_pd(qscr+56, qi4);
    _mm512_store_pd(qscr+64, qr5); _mm512_store_pd(qscr+72, qi5);
    _mm512_store_pd(qscr+80, qr6); _mm512_store_pd(qscr+88, qi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_17[0]);
    __m512d ks2; BCV(ks2, KS_17[1]);
    __m512d ks3; BCV(ks3, KS_17[2]);
    __m512d ks4; BCV(ks4, KS_17[3]);
    __m512d ks5; BCV(ks5, KS_17[4]);
    __m512d ks6; BCV(ks6, KS_17[5]);
    __m512d ks7; BCV(ks7, KS_17[6]);
    __m512d ks8; BCV(ks8, KS_17[7]);
    __m512d qr7, qi7;
    __m512d qr8, qi8;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr7 = _mm512_mul_pd(ks7, di); qi7 = _mm512_mul_pd(ks7, dr);
    qr8 = _mm512_mul_pd(ks8, di); qi8 = _mm512_mul_pd(ks8, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr7 = _mm512_fnmadd_pd(ks3, di, qr7); qi7 = _mm512_fnmadd_pd(ks3, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks1, di, qr8); qi8 = _mm512_fnmadd_pd(ks1, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr7 = _mm512_fmadd_pd(ks4, di, qr7); qi7 = _mm512_fmadd_pd(ks4, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks7, di, qr8); qi8 = _mm512_fmadd_pd(ks7, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr7 = _mm512_fnmadd_pd(ks6, di, qr7); qi7 = _mm512_fnmadd_pd(ks6, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks2, di, qr8); qi8 = _mm512_fnmadd_pd(ks2, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr7 = _mm512_fmadd_pd(ks1, di, qr7); qi7 = _mm512_fmadd_pd(ks1, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks6, di, qr8); qi8 = _mm512_fmadd_pd(ks6, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr7 = _mm512_fmadd_pd(ks8, di, qr7); qi7 = _mm512_fmadd_pd(ks8, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks3, di, qr8); qi8 = _mm512_fnmadd_pd(ks3, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+96), di = _mm512_load_pd(dscr+104);
    qr7 = _mm512_fnmadd_pd(ks2, di, qr7); qi7 = _mm512_fnmadd_pd(ks2, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks5, di, qr8); qi8 = _mm512_fmadd_pd(ks5, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+112), di = _mm512_load_pd(dscr+120);
    qr7 = _mm512_fmadd_pd(ks5, di, qr7); qi7 = _mm512_fmadd_pd(ks5, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks4, di, qr8); qi8 = _mm512_fnmadd_pd(ks4, dr, qi8);
    }
    _mm512_store_pd(qscr+96, qr7); _mm512_store_pd(qscr+104, qi7);
    _mm512_store_pd(qscr+112, qr8); _mm512_store_pd(qscr+120, qi8);
    }
    __asm__ volatile("" ::: "memory");
    {
    {
    __m512d Pr = _mm512_load_pd(pscr+16), Pi = _mm512_load_pd(pscr+24);
    __m512d Qr = _mm512_load_pd(qscr+0), Qi = _mm512_load_pd(qscr+8);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+16, xr); _mm512_store_pd(x+16+8, xi);
    _mm512_store_pd(x+256, yr); _mm512_store_pd(x+256+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+32), Pi = _mm512_load_pd(pscr+40);
    __m512d Qr = _mm512_load_pd(qscr+16), Qi = _mm512_load_pd(qscr+24);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+32, xr); _mm512_store_pd(x+32+8, xi);
    _mm512_store_pd(x+240, yr); _mm512_store_pd(x+240+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+48), Pi = _mm512_load_pd(pscr+56);
    __m512d Qr = _mm512_load_pd(qscr+32), Qi = _mm512_load_pd(qscr+40);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+48, xr); _mm512_store_pd(x+48+8, xi);
    _mm512_store_pd(x+224, yr); _mm512_store_pd(x+224+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+64), Pi = _mm512_load_pd(pscr+72);
    __m512d Qr = _mm512_load_pd(qscr+48), Qi = _mm512_load_pd(qscr+56);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+64, xr); _mm512_store_pd(x+64+8, xi);
    _mm512_store_pd(x+208, yr); _mm512_store_pd(x+208+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+80), Pi = _mm512_load_pd(pscr+88);
    __m512d Qr = _mm512_load_pd(qscr+64), Qi = _mm512_load_pd(qscr+72);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+80, xr); _mm512_store_pd(x+80+8, xi);
    _mm512_store_pd(x+192, yr); _mm512_store_pd(x+192+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+96), Pi = _mm512_load_pd(pscr+104);
    __m512d Qr = _mm512_load_pd(qscr+80), Qi = _mm512_load_pd(qscr+88);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+96, xr); _mm512_store_pd(x+96+8, xi);
    _mm512_store_pd(x+176, yr); _mm512_store_pd(x+176+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+112), Pi = _mm512_load_pd(pscr+120);
    __m512d Qr = _mm512_load_pd(qscr+96), Qi = _mm512_load_pd(qscr+104);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+112, xr); _mm512_store_pd(x+112+8, xi);
    _mm512_store_pd(x+160, yr); _mm512_store_pd(x+160+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+128), Pi = _mm512_load_pd(pscr+136);
    __m512d Qr = _mm512_load_pd(qscr+112), Qi = _mm512_load_pd(qscr+120);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+128, xr); _mm512_store_pd(x+128+8, xi);
    _mm512_store_pd(x+144, yr); _mm512_store_pd(x+144+8, yi);
    }
    }
}
static void __attribute__((noinline)) dy_17(double* restrict x){
    double sscr[16*8] ALIGN64;
    double dscr[16*8] ALIGN64;
    double pscr[16*8+16] ALIGN64;
    double qscr[16*8] ALIGN64;
    {
    __m512d u0r = _mm512_load_pd(x), u0i = _mm512_load_pd(x+8);
    __m512d x0r = u0r, x0i = u0i;
    {
    __m512d ar = _mm512_load_pd(x+272), ai = _mm512_load_pd(x+272+8);
    __m512d br = _mm512_load_pd(x+4352), bi = _mm512_load_pd(x+4352+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+0, sr); _mm512_store_pd(sscr+8, si);
    _mm512_store_pd(dscr+0, dr); _mm512_store_pd(dscr+8, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+544), ai = _mm512_load_pd(x+544+8);
    __m512d br = _mm512_load_pd(x+4080), bi = _mm512_load_pd(x+4080+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+16, sr); _mm512_store_pd(sscr+24, si);
    _mm512_store_pd(dscr+16, dr); _mm512_store_pd(dscr+24, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+816), ai = _mm512_load_pd(x+816+8);
    __m512d br = _mm512_load_pd(x+3808), bi = _mm512_load_pd(x+3808+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+32, sr); _mm512_store_pd(sscr+40, si);
    _mm512_store_pd(dscr+32, dr); _mm512_store_pd(dscr+40, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+1088), ai = _mm512_load_pd(x+1088+8);
    __m512d br = _mm512_load_pd(x+3536), bi = _mm512_load_pd(x+3536+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+48, sr); _mm512_store_pd(sscr+56, si);
    _mm512_store_pd(dscr+48, dr); _mm512_store_pd(dscr+56, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+1360), ai = _mm512_load_pd(x+1360+8);
    __m512d br = _mm512_load_pd(x+3264), bi = _mm512_load_pd(x+3264+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+64, sr); _mm512_store_pd(sscr+72, si);
    _mm512_store_pd(dscr+64, dr); _mm512_store_pd(dscr+72, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+1632), ai = _mm512_load_pd(x+1632+8);
    __m512d br = _mm512_load_pd(x+2992), bi = _mm512_load_pd(x+2992+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+80, sr); _mm512_store_pd(sscr+88, si);
    _mm512_store_pd(dscr+80, dr); _mm512_store_pd(dscr+88, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+1904), ai = _mm512_load_pd(x+1904+8);
    __m512d br = _mm512_load_pd(x+2720), bi = _mm512_load_pd(x+2720+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+96, sr); _mm512_store_pd(sscr+104, si);
    _mm512_store_pd(dscr+96, dr); _mm512_store_pd(dscr+104, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+2176), ai = _mm512_load_pd(x+2176+8);
    __m512d br = _mm512_load_pd(x+2448), bi = _mm512_load_pd(x+2448+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+112, sr); _mm512_store_pd(sscr+120, si);
    _mm512_store_pd(dscr+112, dr); _mm512_store_pd(dscr+120, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    _mm512_store_pd(x, x0r); _mm512_store_pd(x+8, x0i);
    __m512d u0r_s = u0r, u0i_s = u0i;
    _mm512_store_pd(pscr, u0r_s); _mm512_store_pd(pscr+8, u0i_s);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d u0r = _mm512_load_pd(x+0*0), u0i;
    u0r = _mm512_load_pd(pscr); u0i = _mm512_load_pd(pscr+8);
    __m512d kc1; BCV(kc1, KC_17[0]);
    __m512d kc2; BCV(kc2, KC_17[1]);
    __m512d kc3; BCV(kc3, KC_17[2]);
    __m512d kc4; BCV(kc4, KC_17[3]);
    __m512d kc5; BCV(kc5, KC_17[4]);
    __m512d kc6; BCV(kc6, KC_17[5]);
    __m512d kc7; BCV(kc7, KC_17[6]);
    __m512d kc8; BCV(kc8, KC_17[7]);
    __m512d pr1 = u0r, pi1 = u0i;
    __m512d pr2 = u0r, pi2 = u0i;
    __m512d pr3 = u0r, pi3 = u0i;
    __m512d pr4 = u0r, pi4 = u0i;
    __m512d pr5 = u0r, pi5 = u0i;
    __m512d pr6 = u0r, pi6 = u0i;
    {
    __m512d sr = _mm512_load_pd(sscr+0), si = _mm512_load_pd(sscr+8);
    pr1 = _mm512_fmadd_pd(kc1, sr, pr1); pi1 = _mm512_fmadd_pd(kc1, si, pi1);
    pr2 = _mm512_fmadd_pd(kc2, sr, pr2); pi2 = _mm512_fmadd_pd(kc2, si, pi2);
    pr3 = _mm512_fmadd_pd(kc3, sr, pr3); pi3 = _mm512_fmadd_pd(kc3, si, pi3);
    pr4 = _mm512_fmadd_pd(kc4, sr, pr4); pi4 = _mm512_fmadd_pd(kc4, si, pi4);
    pr5 = _mm512_fmadd_pd(kc5, sr, pr5); pi5 = _mm512_fmadd_pd(kc5, si, pi5);
    pr6 = _mm512_fmadd_pd(kc6, sr, pr6); pi6 = _mm512_fmadd_pd(kc6, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+16), si = _mm512_load_pd(sscr+24);
    pr1 = _mm512_fmadd_pd(kc2, sr, pr1); pi1 = _mm512_fmadd_pd(kc2, si, pi1);
    pr2 = _mm512_fmadd_pd(kc4, sr, pr2); pi2 = _mm512_fmadd_pd(kc4, si, pi2);
    pr3 = _mm512_fmadd_pd(kc6, sr, pr3); pi3 = _mm512_fmadd_pd(kc6, si, pi3);
    pr4 = _mm512_fmadd_pd(kc8, sr, pr4); pi4 = _mm512_fmadd_pd(kc8, si, pi4);
    pr5 = _mm512_fmadd_pd(kc7, sr, pr5); pi5 = _mm512_fmadd_pd(kc7, si, pi5);
    pr6 = _mm512_fmadd_pd(kc5, sr, pr6); pi6 = _mm512_fmadd_pd(kc5, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+32), si = _mm512_load_pd(sscr+40);
    pr1 = _mm512_fmadd_pd(kc3, sr, pr1); pi1 = _mm512_fmadd_pd(kc3, si, pi1);
    pr2 = _mm512_fmadd_pd(kc6, sr, pr2); pi2 = _mm512_fmadd_pd(kc6, si, pi2);
    pr3 = _mm512_fmadd_pd(kc8, sr, pr3); pi3 = _mm512_fmadd_pd(kc8, si, pi3);
    pr4 = _mm512_fmadd_pd(kc5, sr, pr4); pi4 = _mm512_fmadd_pd(kc5, si, pi4);
    pr5 = _mm512_fmadd_pd(kc2, sr, pr5); pi5 = _mm512_fmadd_pd(kc2, si, pi5);
    pr6 = _mm512_fmadd_pd(kc1, sr, pr6); pi6 = _mm512_fmadd_pd(kc1, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+48), si = _mm512_load_pd(sscr+56);
    pr1 = _mm512_fmadd_pd(kc4, sr, pr1); pi1 = _mm512_fmadd_pd(kc4, si, pi1);
    pr2 = _mm512_fmadd_pd(kc8, sr, pr2); pi2 = _mm512_fmadd_pd(kc8, si, pi2);
    pr3 = _mm512_fmadd_pd(kc5, sr, pr3); pi3 = _mm512_fmadd_pd(kc5, si, pi3);
    pr4 = _mm512_fmadd_pd(kc1, sr, pr4); pi4 = _mm512_fmadd_pd(kc1, si, pi4);
    pr5 = _mm512_fmadd_pd(kc3, sr, pr5); pi5 = _mm512_fmadd_pd(kc3, si, pi5);
    pr6 = _mm512_fmadd_pd(kc7, sr, pr6); pi6 = _mm512_fmadd_pd(kc7, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+64), si = _mm512_load_pd(sscr+72);
    pr1 = _mm512_fmadd_pd(kc5, sr, pr1); pi1 = _mm512_fmadd_pd(kc5, si, pi1);
    pr2 = _mm512_fmadd_pd(kc7, sr, pr2); pi2 = _mm512_fmadd_pd(kc7, si, pi2);
    pr3 = _mm512_fmadd_pd(kc2, sr, pr3); pi3 = _mm512_fmadd_pd(kc2, si, pi3);
    pr4 = _mm512_fmadd_pd(kc3, sr, pr4); pi4 = _mm512_fmadd_pd(kc3, si, pi4);
    pr5 = _mm512_fmadd_pd(kc8, sr, pr5); pi5 = _mm512_fmadd_pd(kc8, si, pi5);
    pr6 = _mm512_fmadd_pd(kc4, sr, pr6); pi6 = _mm512_fmadd_pd(kc4, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+80), si = _mm512_load_pd(sscr+88);
    pr1 = _mm512_fmadd_pd(kc6, sr, pr1); pi1 = _mm512_fmadd_pd(kc6, si, pi1);
    pr2 = _mm512_fmadd_pd(kc5, sr, pr2); pi2 = _mm512_fmadd_pd(kc5, si, pi2);
    pr3 = _mm512_fmadd_pd(kc1, sr, pr3); pi3 = _mm512_fmadd_pd(kc1, si, pi3);
    pr4 = _mm512_fmadd_pd(kc7, sr, pr4); pi4 = _mm512_fmadd_pd(kc7, si, pi4);
    pr5 = _mm512_fmadd_pd(kc4, sr, pr5); pi5 = _mm512_fmadd_pd(kc4, si, pi5);
    pr6 = _mm512_fmadd_pd(kc2, sr, pr6); pi6 = _mm512_fmadd_pd(kc2, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+96), si = _mm512_load_pd(sscr+104);
    pr1 = _mm512_fmadd_pd(kc7, sr, pr1); pi1 = _mm512_fmadd_pd(kc7, si, pi1);
    pr2 = _mm512_fmadd_pd(kc3, sr, pr2); pi2 = _mm512_fmadd_pd(kc3, si, pi2);
    pr3 = _mm512_fmadd_pd(kc4, sr, pr3); pi3 = _mm512_fmadd_pd(kc4, si, pi3);
    pr4 = _mm512_fmadd_pd(kc6, sr, pr4); pi4 = _mm512_fmadd_pd(kc6, si, pi4);
    pr5 = _mm512_fmadd_pd(kc1, sr, pr5); pi5 = _mm512_fmadd_pd(kc1, si, pi5);
    pr6 = _mm512_fmadd_pd(kc8, sr, pr6); pi6 = _mm512_fmadd_pd(kc8, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+112), si = _mm512_load_pd(sscr+120);
    pr1 = _mm512_fmadd_pd(kc8, sr, pr1); pi1 = _mm512_fmadd_pd(kc8, si, pi1);
    pr2 = _mm512_fmadd_pd(kc1, sr, pr2); pi2 = _mm512_fmadd_pd(kc1, si, pi2);
    pr3 = _mm512_fmadd_pd(kc7, sr, pr3); pi3 = _mm512_fmadd_pd(kc7, si, pi3);
    pr4 = _mm512_fmadd_pd(kc2, sr, pr4); pi4 = _mm512_fmadd_pd(kc2, si, pi4);
    pr5 = _mm512_fmadd_pd(kc6, sr, pr5); pi5 = _mm512_fmadd_pd(kc6, si, pi5);
    pr6 = _mm512_fmadd_pd(kc3, sr, pr6); pi6 = _mm512_fmadd_pd(kc3, si, pi6);
    }
    _mm512_store_pd(pscr+16, pr1); _mm512_store_pd(pscr+24, pi1);
    _mm512_store_pd(pscr+32, pr2); _mm512_store_pd(pscr+40, pi2);
    _mm512_store_pd(pscr+48, pr3); _mm512_store_pd(pscr+56, pi3);
    _mm512_store_pd(pscr+64, pr4); _mm512_store_pd(pscr+72, pi4);
    _mm512_store_pd(pscr+80, pr5); _mm512_store_pd(pscr+88, pi5);
    _mm512_store_pd(pscr+96, pr6); _mm512_store_pd(pscr+104, pi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d u0r = _mm512_load_pd(x+0*0), u0i;
    u0r = _mm512_load_pd(pscr); u0i = _mm512_load_pd(pscr+8);
    __m512d kc1; BCV(kc1, KC_17[0]);
    __m512d kc2; BCV(kc2, KC_17[1]);
    __m512d kc3; BCV(kc3, KC_17[2]);
    __m512d kc4; BCV(kc4, KC_17[3]);
    __m512d kc5; BCV(kc5, KC_17[4]);
    __m512d kc6; BCV(kc6, KC_17[5]);
    __m512d kc7; BCV(kc7, KC_17[6]);
    __m512d kc8; BCV(kc8, KC_17[7]);
    __m512d pr7 = u0r, pi7 = u0i;
    __m512d pr8 = u0r, pi8 = u0i;
    {
    __m512d sr = _mm512_load_pd(sscr+0), si = _mm512_load_pd(sscr+8);
    pr7 = _mm512_fmadd_pd(kc7, sr, pr7); pi7 = _mm512_fmadd_pd(kc7, si, pi7);
    pr8 = _mm512_fmadd_pd(kc8, sr, pr8); pi8 = _mm512_fmadd_pd(kc8, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+16), si = _mm512_load_pd(sscr+24);
    pr7 = _mm512_fmadd_pd(kc3, sr, pr7); pi7 = _mm512_fmadd_pd(kc3, si, pi7);
    pr8 = _mm512_fmadd_pd(kc1, sr, pr8); pi8 = _mm512_fmadd_pd(kc1, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+32), si = _mm512_load_pd(sscr+40);
    pr7 = _mm512_fmadd_pd(kc4, sr, pr7); pi7 = _mm512_fmadd_pd(kc4, si, pi7);
    pr8 = _mm512_fmadd_pd(kc7, sr, pr8); pi8 = _mm512_fmadd_pd(kc7, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+48), si = _mm512_load_pd(sscr+56);
    pr7 = _mm512_fmadd_pd(kc6, sr, pr7); pi7 = _mm512_fmadd_pd(kc6, si, pi7);
    pr8 = _mm512_fmadd_pd(kc2, sr, pr8); pi8 = _mm512_fmadd_pd(kc2, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+64), si = _mm512_load_pd(sscr+72);
    pr7 = _mm512_fmadd_pd(kc1, sr, pr7); pi7 = _mm512_fmadd_pd(kc1, si, pi7);
    pr8 = _mm512_fmadd_pd(kc6, sr, pr8); pi8 = _mm512_fmadd_pd(kc6, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+80), si = _mm512_load_pd(sscr+88);
    pr7 = _mm512_fmadd_pd(kc8, sr, pr7); pi7 = _mm512_fmadd_pd(kc8, si, pi7);
    pr8 = _mm512_fmadd_pd(kc3, sr, pr8); pi8 = _mm512_fmadd_pd(kc3, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+96), si = _mm512_load_pd(sscr+104);
    pr7 = _mm512_fmadd_pd(kc2, sr, pr7); pi7 = _mm512_fmadd_pd(kc2, si, pi7);
    pr8 = _mm512_fmadd_pd(kc5, sr, pr8); pi8 = _mm512_fmadd_pd(kc5, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+112), si = _mm512_load_pd(sscr+120);
    pr7 = _mm512_fmadd_pd(kc5, sr, pr7); pi7 = _mm512_fmadd_pd(kc5, si, pi7);
    pr8 = _mm512_fmadd_pd(kc4, sr, pr8); pi8 = _mm512_fmadd_pd(kc4, si, pi8);
    }
    _mm512_store_pd(pscr+112, pr7); _mm512_store_pd(pscr+120, pi7);
    _mm512_store_pd(pscr+128, pr8); _mm512_store_pd(pscr+136, pi8);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_17[0]);
    __m512d ks2; BCV(ks2, KS_17[1]);
    __m512d ks3; BCV(ks3, KS_17[2]);
    __m512d ks4; BCV(ks4, KS_17[3]);
    __m512d ks5; BCV(ks5, KS_17[4]);
    __m512d ks6; BCV(ks6, KS_17[5]);
    __m512d ks7; BCV(ks7, KS_17[6]);
    __m512d ks8; BCV(ks8, KS_17[7]);
    __m512d qr1, qi1;
    __m512d qr2, qi2;
    __m512d qr3, qi3;
    __m512d qr4, qi4;
    __m512d qr5, qi5;
    __m512d qr6, qi6;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr1 = _mm512_mul_pd(ks1, di); qi1 = _mm512_mul_pd(ks1, dr);
    qr2 = _mm512_mul_pd(ks2, di); qi2 = _mm512_mul_pd(ks2, dr);
    qr3 = _mm512_mul_pd(ks3, di); qi3 = _mm512_mul_pd(ks3, dr);
    qr4 = _mm512_mul_pd(ks4, di); qi4 = _mm512_mul_pd(ks4, dr);
    qr5 = _mm512_mul_pd(ks5, di); qi5 = _mm512_mul_pd(ks5, dr);
    qr6 = _mm512_mul_pd(ks6, di); qi6 = _mm512_mul_pd(ks6, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr1 = _mm512_fmadd_pd(ks2, di, qr1); qi1 = _mm512_fmadd_pd(ks2, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks4, di, qr2); qi2 = _mm512_fmadd_pd(ks4, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks6, di, qr3); qi3 = _mm512_fmadd_pd(ks6, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks8, di, qr4); qi4 = _mm512_fmadd_pd(ks8, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks7, di, qr5); qi5 = _mm512_fnmadd_pd(ks7, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks5, di, qr6); qi6 = _mm512_fnmadd_pd(ks5, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr1 = _mm512_fmadd_pd(ks3, di, qr1); qi1 = _mm512_fmadd_pd(ks3, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks6, di, qr2); qi2 = _mm512_fmadd_pd(ks6, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks8, di, qr3); qi3 = _mm512_fnmadd_pd(ks8, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks5, di, qr4); qi4 = _mm512_fnmadd_pd(ks5, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks2, di, qr5); qi5 = _mm512_fnmadd_pd(ks2, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks1, di, qr6); qi6 = _mm512_fmadd_pd(ks1, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr1 = _mm512_fmadd_pd(ks4, di, qr1); qi1 = _mm512_fmadd_pd(ks4, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks8, di, qr2); qi2 = _mm512_fmadd_pd(ks8, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks5, di, qr3); qi3 = _mm512_fnmadd_pd(ks5, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks1, di, qr4); qi4 = _mm512_fnmadd_pd(ks1, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks3, di, qr5); qi5 = _mm512_fmadd_pd(ks3, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks7, di, qr6); qi6 = _mm512_fmadd_pd(ks7, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr1 = _mm512_fmadd_pd(ks5, di, qr1); qi1 = _mm512_fmadd_pd(ks5, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks7, di, qr2); qi2 = _mm512_fnmadd_pd(ks7, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks2, di, qr3); qi3 = _mm512_fnmadd_pd(ks2, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks3, di, qr4); qi4 = _mm512_fmadd_pd(ks3, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks8, di, qr5); qi5 = _mm512_fmadd_pd(ks8, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks4, di, qr6); qi6 = _mm512_fnmadd_pd(ks4, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr1 = _mm512_fmadd_pd(ks6, di, qr1); qi1 = _mm512_fmadd_pd(ks6, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks5, di, qr2); qi2 = _mm512_fnmadd_pd(ks5, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks1, di, qr3); qi3 = _mm512_fmadd_pd(ks1, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks7, di, qr4); qi4 = _mm512_fmadd_pd(ks7, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks4, di, qr5); qi5 = _mm512_fnmadd_pd(ks4, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks2, di, qr6); qi6 = _mm512_fmadd_pd(ks2, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+96), di = _mm512_load_pd(dscr+104);
    qr1 = _mm512_fmadd_pd(ks7, di, qr1); qi1 = _mm512_fmadd_pd(ks7, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks3, di, qr2); qi2 = _mm512_fnmadd_pd(ks3, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks4, di, qr3); qi3 = _mm512_fmadd_pd(ks4, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks6, di, qr4); qi4 = _mm512_fnmadd_pd(ks6, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks1, di, qr5); qi5 = _mm512_fmadd_pd(ks1, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks8, di, qr6); qi6 = _mm512_fmadd_pd(ks8, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+112), di = _mm512_load_pd(dscr+120);
    qr1 = _mm512_fmadd_pd(ks8, di, qr1); qi1 = _mm512_fmadd_pd(ks8, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks1, di, qr2); qi2 = _mm512_fnmadd_pd(ks1, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks7, di, qr3); qi3 = _mm512_fmadd_pd(ks7, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks2, di, qr4); qi4 = _mm512_fnmadd_pd(ks2, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks6, di, qr5); qi5 = _mm512_fmadd_pd(ks6, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks3, di, qr6); qi6 = _mm512_fnmadd_pd(ks3, dr, qi6);
    }
    _mm512_store_pd(qscr+0, qr1); _mm512_store_pd(qscr+8, qi1);
    _mm512_store_pd(qscr+16, qr2); _mm512_store_pd(qscr+24, qi2);
    _mm512_store_pd(qscr+32, qr3); _mm512_store_pd(qscr+40, qi3);
    _mm512_store_pd(qscr+48, qr4); _mm512_store_pd(qscr+56, qi4);
    _mm512_store_pd(qscr+64, qr5); _mm512_store_pd(qscr+72, qi5);
    _mm512_store_pd(qscr+80, qr6); _mm512_store_pd(qscr+88, qi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_17[0]);
    __m512d ks2; BCV(ks2, KS_17[1]);
    __m512d ks3; BCV(ks3, KS_17[2]);
    __m512d ks4; BCV(ks4, KS_17[3]);
    __m512d ks5; BCV(ks5, KS_17[4]);
    __m512d ks6; BCV(ks6, KS_17[5]);
    __m512d ks7; BCV(ks7, KS_17[6]);
    __m512d ks8; BCV(ks8, KS_17[7]);
    __m512d qr7, qi7;
    __m512d qr8, qi8;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr7 = _mm512_mul_pd(ks7, di); qi7 = _mm512_mul_pd(ks7, dr);
    qr8 = _mm512_mul_pd(ks8, di); qi8 = _mm512_mul_pd(ks8, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr7 = _mm512_fnmadd_pd(ks3, di, qr7); qi7 = _mm512_fnmadd_pd(ks3, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks1, di, qr8); qi8 = _mm512_fnmadd_pd(ks1, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr7 = _mm512_fmadd_pd(ks4, di, qr7); qi7 = _mm512_fmadd_pd(ks4, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks7, di, qr8); qi8 = _mm512_fmadd_pd(ks7, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr7 = _mm512_fnmadd_pd(ks6, di, qr7); qi7 = _mm512_fnmadd_pd(ks6, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks2, di, qr8); qi8 = _mm512_fnmadd_pd(ks2, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr7 = _mm512_fmadd_pd(ks1, di, qr7); qi7 = _mm512_fmadd_pd(ks1, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks6, di, qr8); qi8 = _mm512_fmadd_pd(ks6, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr7 = _mm512_fmadd_pd(ks8, di, qr7); qi7 = _mm512_fmadd_pd(ks8, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks3, di, qr8); qi8 = _mm512_fnmadd_pd(ks3, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+96), di = _mm512_load_pd(dscr+104);
    qr7 = _mm512_fnmadd_pd(ks2, di, qr7); qi7 = _mm512_fnmadd_pd(ks2, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks5, di, qr8); qi8 = _mm512_fmadd_pd(ks5, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+112), di = _mm512_load_pd(dscr+120);
    qr7 = _mm512_fmadd_pd(ks5, di, qr7); qi7 = _mm512_fmadd_pd(ks5, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks4, di, qr8); qi8 = _mm512_fnmadd_pd(ks4, dr, qi8);
    }
    _mm512_store_pd(qscr+96, qr7); _mm512_store_pd(qscr+104, qi7);
    _mm512_store_pd(qscr+112, qr8); _mm512_store_pd(qscr+120, qi8);
    }
    __asm__ volatile("" ::: "memory");
    {
    {
    __m512d Pr = _mm512_load_pd(pscr+16), Pi = _mm512_load_pd(pscr+24);
    __m512d Qr = _mm512_load_pd(qscr+0), Qi = _mm512_load_pd(qscr+8);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+272, xr); _mm512_store_pd(x+272+8, xi);
    _mm512_store_pd(x+4352, yr); _mm512_store_pd(x+4352+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+32), Pi = _mm512_load_pd(pscr+40);
    __m512d Qr = _mm512_load_pd(qscr+16), Qi = _mm512_load_pd(qscr+24);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+544, xr); _mm512_store_pd(x+544+8, xi);
    _mm512_store_pd(x+4080, yr); _mm512_store_pd(x+4080+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+48), Pi = _mm512_load_pd(pscr+56);
    __m512d Qr = _mm512_load_pd(qscr+32), Qi = _mm512_load_pd(qscr+40);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+816, xr); _mm512_store_pd(x+816+8, xi);
    _mm512_store_pd(x+3808, yr); _mm512_store_pd(x+3808+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+64), Pi = _mm512_load_pd(pscr+72);
    __m512d Qr = _mm512_load_pd(qscr+48), Qi = _mm512_load_pd(qscr+56);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+1088, xr); _mm512_store_pd(x+1088+8, xi);
    _mm512_store_pd(x+3536, yr); _mm512_store_pd(x+3536+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+80), Pi = _mm512_load_pd(pscr+88);
    __m512d Qr = _mm512_load_pd(qscr+64), Qi = _mm512_load_pd(qscr+72);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+1360, xr); _mm512_store_pd(x+1360+8, xi);
    _mm512_store_pd(x+3264, yr); _mm512_store_pd(x+3264+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+96), Pi = _mm512_load_pd(pscr+104);
    __m512d Qr = _mm512_load_pd(qscr+80), Qi = _mm512_load_pd(qscr+88);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+1632, xr); _mm512_store_pd(x+1632+8, xi);
    _mm512_store_pd(x+2992, yr); _mm512_store_pd(x+2992+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+112), Pi = _mm512_load_pd(pscr+120);
    __m512d Qr = _mm512_load_pd(qscr+96), Qi = _mm512_load_pd(qscr+104);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+1904, xr); _mm512_store_pd(x+1904+8, xi);
    _mm512_store_pd(x+2720, yr); _mm512_store_pd(x+2720+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+128), Pi = _mm512_load_pd(pscr+136);
    __m512d Qr = _mm512_load_pd(qscr+112), Qi = _mm512_load_pd(qscr+120);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+2176, xr); _mm512_store_pd(x+2176+8, xi);
    _mm512_store_pd(x+2448, yr); _mm512_store_pd(x+2448+8, yi);
    }
    }
}
static void __attribute__((noinline)) dx_17(double* restrict x, const double* restrict cb){
    double sscr[16*8] ALIGN64;
    double dscr[16*8] ALIGN64;
    double pscr[16*8+16] ALIGN64;
    double qscr[16*8] ALIGN64;
    {
    __m512d u0r = _mm512_load_pd(x), u0i = _mm512_load_pd(x+8);
    __m512d x0r = u0r, x0i = u0i;
    {
    __m512d ar = _mm512_load_pd(x+4624), ai = _mm512_load_pd(x+4624+8);
    __m512d br = _mm512_load_pd(x+73984), bi = _mm512_load_pd(x+73984+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+0, sr); _mm512_store_pd(sscr+8, si);
    _mm512_store_pd(dscr+0, dr); _mm512_store_pd(dscr+8, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+9248), ai = _mm512_load_pd(x+9248+8);
    __m512d br = _mm512_load_pd(x+69360), bi = _mm512_load_pd(x+69360+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+16, sr); _mm512_store_pd(sscr+24, si);
    _mm512_store_pd(dscr+16, dr); _mm512_store_pd(dscr+24, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+13872), ai = _mm512_load_pd(x+13872+8);
    __m512d br = _mm512_load_pd(x+64736), bi = _mm512_load_pd(x+64736+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+32, sr); _mm512_store_pd(sscr+40, si);
    _mm512_store_pd(dscr+32, dr); _mm512_store_pd(dscr+40, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+18496), ai = _mm512_load_pd(x+18496+8);
    __m512d br = _mm512_load_pd(x+60112), bi = _mm512_load_pd(x+60112+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+48, sr); _mm512_store_pd(sscr+56, si);
    _mm512_store_pd(dscr+48, dr); _mm512_store_pd(dscr+56, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+23120), ai = _mm512_load_pd(x+23120+8);
    __m512d br = _mm512_load_pd(x+55488), bi = _mm512_load_pd(x+55488+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+64, sr); _mm512_store_pd(sscr+72, si);
    _mm512_store_pd(dscr+64, dr); _mm512_store_pd(dscr+72, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+27744), ai = _mm512_load_pd(x+27744+8);
    __m512d br = _mm512_load_pd(x+50864), bi = _mm512_load_pd(x+50864+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+80, sr); _mm512_store_pd(sscr+88, si);
    _mm512_store_pd(dscr+80, dr); _mm512_store_pd(dscr+88, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+32368), ai = _mm512_load_pd(x+32368+8);
    __m512d br = _mm512_load_pd(x+46240), bi = _mm512_load_pd(x+46240+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+96, sr); _mm512_store_pd(sscr+104, si);
    _mm512_store_pd(dscr+96, dr); _mm512_store_pd(dscr+104, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+36992), ai = _mm512_load_pd(x+36992+8);
    __m512d br = _mm512_load_pd(x+41616), bi = _mm512_load_pd(x+41616+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+112, sr); _mm512_store_pd(sscr+120, si);
    _mm512_store_pd(dscr+112, dr); _mm512_store_pd(dscr+120, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    MAPST(x0r, x0i, x, 0, cb, 0);
    __m512d u0r_s = u0r, u0i_s = u0i;
    _mm512_store_pd(pscr, u0r_s); _mm512_store_pd(pscr+8, u0i_s);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d u0r = _mm512_load_pd(x+0*0), u0i;
    u0r = _mm512_load_pd(pscr); u0i = _mm512_load_pd(pscr+8);
    __m512d kc1; BCV(kc1, KC_17[0]);
    __m512d kc2; BCV(kc2, KC_17[1]);
    __m512d kc3; BCV(kc3, KC_17[2]);
    __m512d kc4; BCV(kc4, KC_17[3]);
    __m512d kc5; BCV(kc5, KC_17[4]);
    __m512d kc6; BCV(kc6, KC_17[5]);
    __m512d kc7; BCV(kc7, KC_17[6]);
    __m512d kc8; BCV(kc8, KC_17[7]);
    __m512d pr1 = u0r, pi1 = u0i;
    __m512d pr2 = u0r, pi2 = u0i;
    __m512d pr3 = u0r, pi3 = u0i;
    __m512d pr4 = u0r, pi4 = u0i;
    __m512d pr5 = u0r, pi5 = u0i;
    __m512d pr6 = u0r, pi6 = u0i;
    {
    __m512d sr = _mm512_load_pd(sscr+0), si = _mm512_load_pd(sscr+8);
    pr1 = _mm512_fmadd_pd(kc1, sr, pr1); pi1 = _mm512_fmadd_pd(kc1, si, pi1);
    pr2 = _mm512_fmadd_pd(kc2, sr, pr2); pi2 = _mm512_fmadd_pd(kc2, si, pi2);
    pr3 = _mm512_fmadd_pd(kc3, sr, pr3); pi3 = _mm512_fmadd_pd(kc3, si, pi3);
    pr4 = _mm512_fmadd_pd(kc4, sr, pr4); pi4 = _mm512_fmadd_pd(kc4, si, pi4);
    pr5 = _mm512_fmadd_pd(kc5, sr, pr5); pi5 = _mm512_fmadd_pd(kc5, si, pi5);
    pr6 = _mm512_fmadd_pd(kc6, sr, pr6); pi6 = _mm512_fmadd_pd(kc6, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+16), si = _mm512_load_pd(sscr+24);
    pr1 = _mm512_fmadd_pd(kc2, sr, pr1); pi1 = _mm512_fmadd_pd(kc2, si, pi1);
    pr2 = _mm512_fmadd_pd(kc4, sr, pr2); pi2 = _mm512_fmadd_pd(kc4, si, pi2);
    pr3 = _mm512_fmadd_pd(kc6, sr, pr3); pi3 = _mm512_fmadd_pd(kc6, si, pi3);
    pr4 = _mm512_fmadd_pd(kc8, sr, pr4); pi4 = _mm512_fmadd_pd(kc8, si, pi4);
    pr5 = _mm512_fmadd_pd(kc7, sr, pr5); pi5 = _mm512_fmadd_pd(kc7, si, pi5);
    pr6 = _mm512_fmadd_pd(kc5, sr, pr6); pi6 = _mm512_fmadd_pd(kc5, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+32), si = _mm512_load_pd(sscr+40);
    pr1 = _mm512_fmadd_pd(kc3, sr, pr1); pi1 = _mm512_fmadd_pd(kc3, si, pi1);
    pr2 = _mm512_fmadd_pd(kc6, sr, pr2); pi2 = _mm512_fmadd_pd(kc6, si, pi2);
    pr3 = _mm512_fmadd_pd(kc8, sr, pr3); pi3 = _mm512_fmadd_pd(kc8, si, pi3);
    pr4 = _mm512_fmadd_pd(kc5, sr, pr4); pi4 = _mm512_fmadd_pd(kc5, si, pi4);
    pr5 = _mm512_fmadd_pd(kc2, sr, pr5); pi5 = _mm512_fmadd_pd(kc2, si, pi5);
    pr6 = _mm512_fmadd_pd(kc1, sr, pr6); pi6 = _mm512_fmadd_pd(kc1, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+48), si = _mm512_load_pd(sscr+56);
    pr1 = _mm512_fmadd_pd(kc4, sr, pr1); pi1 = _mm512_fmadd_pd(kc4, si, pi1);
    pr2 = _mm512_fmadd_pd(kc8, sr, pr2); pi2 = _mm512_fmadd_pd(kc8, si, pi2);
    pr3 = _mm512_fmadd_pd(kc5, sr, pr3); pi3 = _mm512_fmadd_pd(kc5, si, pi3);
    pr4 = _mm512_fmadd_pd(kc1, sr, pr4); pi4 = _mm512_fmadd_pd(kc1, si, pi4);
    pr5 = _mm512_fmadd_pd(kc3, sr, pr5); pi5 = _mm512_fmadd_pd(kc3, si, pi5);
    pr6 = _mm512_fmadd_pd(kc7, sr, pr6); pi6 = _mm512_fmadd_pd(kc7, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+64), si = _mm512_load_pd(sscr+72);
    pr1 = _mm512_fmadd_pd(kc5, sr, pr1); pi1 = _mm512_fmadd_pd(kc5, si, pi1);
    pr2 = _mm512_fmadd_pd(kc7, sr, pr2); pi2 = _mm512_fmadd_pd(kc7, si, pi2);
    pr3 = _mm512_fmadd_pd(kc2, sr, pr3); pi3 = _mm512_fmadd_pd(kc2, si, pi3);
    pr4 = _mm512_fmadd_pd(kc3, sr, pr4); pi4 = _mm512_fmadd_pd(kc3, si, pi4);
    pr5 = _mm512_fmadd_pd(kc8, sr, pr5); pi5 = _mm512_fmadd_pd(kc8, si, pi5);
    pr6 = _mm512_fmadd_pd(kc4, sr, pr6); pi6 = _mm512_fmadd_pd(kc4, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+80), si = _mm512_load_pd(sscr+88);
    pr1 = _mm512_fmadd_pd(kc6, sr, pr1); pi1 = _mm512_fmadd_pd(kc6, si, pi1);
    pr2 = _mm512_fmadd_pd(kc5, sr, pr2); pi2 = _mm512_fmadd_pd(kc5, si, pi2);
    pr3 = _mm512_fmadd_pd(kc1, sr, pr3); pi3 = _mm512_fmadd_pd(kc1, si, pi3);
    pr4 = _mm512_fmadd_pd(kc7, sr, pr4); pi4 = _mm512_fmadd_pd(kc7, si, pi4);
    pr5 = _mm512_fmadd_pd(kc4, sr, pr5); pi5 = _mm512_fmadd_pd(kc4, si, pi5);
    pr6 = _mm512_fmadd_pd(kc2, sr, pr6); pi6 = _mm512_fmadd_pd(kc2, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+96), si = _mm512_load_pd(sscr+104);
    pr1 = _mm512_fmadd_pd(kc7, sr, pr1); pi1 = _mm512_fmadd_pd(kc7, si, pi1);
    pr2 = _mm512_fmadd_pd(kc3, sr, pr2); pi2 = _mm512_fmadd_pd(kc3, si, pi2);
    pr3 = _mm512_fmadd_pd(kc4, sr, pr3); pi3 = _mm512_fmadd_pd(kc4, si, pi3);
    pr4 = _mm512_fmadd_pd(kc6, sr, pr4); pi4 = _mm512_fmadd_pd(kc6, si, pi4);
    pr5 = _mm512_fmadd_pd(kc1, sr, pr5); pi5 = _mm512_fmadd_pd(kc1, si, pi5);
    pr6 = _mm512_fmadd_pd(kc8, sr, pr6); pi6 = _mm512_fmadd_pd(kc8, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+112), si = _mm512_load_pd(sscr+120);
    pr1 = _mm512_fmadd_pd(kc8, sr, pr1); pi1 = _mm512_fmadd_pd(kc8, si, pi1);
    pr2 = _mm512_fmadd_pd(kc1, sr, pr2); pi2 = _mm512_fmadd_pd(kc1, si, pi2);
    pr3 = _mm512_fmadd_pd(kc7, sr, pr3); pi3 = _mm512_fmadd_pd(kc7, si, pi3);
    pr4 = _mm512_fmadd_pd(kc2, sr, pr4); pi4 = _mm512_fmadd_pd(kc2, si, pi4);
    pr5 = _mm512_fmadd_pd(kc6, sr, pr5); pi5 = _mm512_fmadd_pd(kc6, si, pi5);
    pr6 = _mm512_fmadd_pd(kc3, sr, pr6); pi6 = _mm512_fmadd_pd(kc3, si, pi6);
    }
    _mm512_store_pd(pscr+16, pr1); _mm512_store_pd(pscr+24, pi1);
    _mm512_store_pd(pscr+32, pr2); _mm512_store_pd(pscr+40, pi2);
    _mm512_store_pd(pscr+48, pr3); _mm512_store_pd(pscr+56, pi3);
    _mm512_store_pd(pscr+64, pr4); _mm512_store_pd(pscr+72, pi4);
    _mm512_store_pd(pscr+80, pr5); _mm512_store_pd(pscr+88, pi5);
    _mm512_store_pd(pscr+96, pr6); _mm512_store_pd(pscr+104, pi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d u0r = _mm512_load_pd(x+0*0), u0i;
    u0r = _mm512_load_pd(pscr); u0i = _mm512_load_pd(pscr+8);
    __m512d kc1; BCV(kc1, KC_17[0]);
    __m512d kc2; BCV(kc2, KC_17[1]);
    __m512d kc3; BCV(kc3, KC_17[2]);
    __m512d kc4; BCV(kc4, KC_17[3]);
    __m512d kc5; BCV(kc5, KC_17[4]);
    __m512d kc6; BCV(kc6, KC_17[5]);
    __m512d kc7; BCV(kc7, KC_17[6]);
    __m512d kc8; BCV(kc8, KC_17[7]);
    __m512d pr7 = u0r, pi7 = u0i;
    __m512d pr8 = u0r, pi8 = u0i;
    {
    __m512d sr = _mm512_load_pd(sscr+0), si = _mm512_load_pd(sscr+8);
    pr7 = _mm512_fmadd_pd(kc7, sr, pr7); pi7 = _mm512_fmadd_pd(kc7, si, pi7);
    pr8 = _mm512_fmadd_pd(kc8, sr, pr8); pi8 = _mm512_fmadd_pd(kc8, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+16), si = _mm512_load_pd(sscr+24);
    pr7 = _mm512_fmadd_pd(kc3, sr, pr7); pi7 = _mm512_fmadd_pd(kc3, si, pi7);
    pr8 = _mm512_fmadd_pd(kc1, sr, pr8); pi8 = _mm512_fmadd_pd(kc1, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+32), si = _mm512_load_pd(sscr+40);
    pr7 = _mm512_fmadd_pd(kc4, sr, pr7); pi7 = _mm512_fmadd_pd(kc4, si, pi7);
    pr8 = _mm512_fmadd_pd(kc7, sr, pr8); pi8 = _mm512_fmadd_pd(kc7, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+48), si = _mm512_load_pd(sscr+56);
    pr7 = _mm512_fmadd_pd(kc6, sr, pr7); pi7 = _mm512_fmadd_pd(kc6, si, pi7);
    pr8 = _mm512_fmadd_pd(kc2, sr, pr8); pi8 = _mm512_fmadd_pd(kc2, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+64), si = _mm512_load_pd(sscr+72);
    pr7 = _mm512_fmadd_pd(kc1, sr, pr7); pi7 = _mm512_fmadd_pd(kc1, si, pi7);
    pr8 = _mm512_fmadd_pd(kc6, sr, pr8); pi8 = _mm512_fmadd_pd(kc6, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+80), si = _mm512_load_pd(sscr+88);
    pr7 = _mm512_fmadd_pd(kc8, sr, pr7); pi7 = _mm512_fmadd_pd(kc8, si, pi7);
    pr8 = _mm512_fmadd_pd(kc3, sr, pr8); pi8 = _mm512_fmadd_pd(kc3, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+96), si = _mm512_load_pd(sscr+104);
    pr7 = _mm512_fmadd_pd(kc2, sr, pr7); pi7 = _mm512_fmadd_pd(kc2, si, pi7);
    pr8 = _mm512_fmadd_pd(kc5, sr, pr8); pi8 = _mm512_fmadd_pd(kc5, si, pi8);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+112), si = _mm512_load_pd(sscr+120);
    pr7 = _mm512_fmadd_pd(kc5, sr, pr7); pi7 = _mm512_fmadd_pd(kc5, si, pi7);
    pr8 = _mm512_fmadd_pd(kc4, sr, pr8); pi8 = _mm512_fmadd_pd(kc4, si, pi8);
    }
    _mm512_store_pd(pscr+112, pr7); _mm512_store_pd(pscr+120, pi7);
    _mm512_store_pd(pscr+128, pr8); _mm512_store_pd(pscr+136, pi8);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_17[0]);
    __m512d ks2; BCV(ks2, KS_17[1]);
    __m512d ks3; BCV(ks3, KS_17[2]);
    __m512d ks4; BCV(ks4, KS_17[3]);
    __m512d ks5; BCV(ks5, KS_17[4]);
    __m512d ks6; BCV(ks6, KS_17[5]);
    __m512d ks7; BCV(ks7, KS_17[6]);
    __m512d ks8; BCV(ks8, KS_17[7]);
    __m512d qr1, qi1;
    __m512d qr2, qi2;
    __m512d qr3, qi3;
    __m512d qr4, qi4;
    __m512d qr5, qi5;
    __m512d qr6, qi6;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr1 = _mm512_mul_pd(ks1, di); qi1 = _mm512_mul_pd(ks1, dr);
    qr2 = _mm512_mul_pd(ks2, di); qi2 = _mm512_mul_pd(ks2, dr);
    qr3 = _mm512_mul_pd(ks3, di); qi3 = _mm512_mul_pd(ks3, dr);
    qr4 = _mm512_mul_pd(ks4, di); qi4 = _mm512_mul_pd(ks4, dr);
    qr5 = _mm512_mul_pd(ks5, di); qi5 = _mm512_mul_pd(ks5, dr);
    qr6 = _mm512_mul_pd(ks6, di); qi6 = _mm512_mul_pd(ks6, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr1 = _mm512_fmadd_pd(ks2, di, qr1); qi1 = _mm512_fmadd_pd(ks2, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks4, di, qr2); qi2 = _mm512_fmadd_pd(ks4, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks6, di, qr3); qi3 = _mm512_fmadd_pd(ks6, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks8, di, qr4); qi4 = _mm512_fmadd_pd(ks8, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks7, di, qr5); qi5 = _mm512_fnmadd_pd(ks7, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks5, di, qr6); qi6 = _mm512_fnmadd_pd(ks5, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr1 = _mm512_fmadd_pd(ks3, di, qr1); qi1 = _mm512_fmadd_pd(ks3, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks6, di, qr2); qi2 = _mm512_fmadd_pd(ks6, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks8, di, qr3); qi3 = _mm512_fnmadd_pd(ks8, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks5, di, qr4); qi4 = _mm512_fnmadd_pd(ks5, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks2, di, qr5); qi5 = _mm512_fnmadd_pd(ks2, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks1, di, qr6); qi6 = _mm512_fmadd_pd(ks1, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr1 = _mm512_fmadd_pd(ks4, di, qr1); qi1 = _mm512_fmadd_pd(ks4, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks8, di, qr2); qi2 = _mm512_fmadd_pd(ks8, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks5, di, qr3); qi3 = _mm512_fnmadd_pd(ks5, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks1, di, qr4); qi4 = _mm512_fnmadd_pd(ks1, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks3, di, qr5); qi5 = _mm512_fmadd_pd(ks3, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks7, di, qr6); qi6 = _mm512_fmadd_pd(ks7, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr1 = _mm512_fmadd_pd(ks5, di, qr1); qi1 = _mm512_fmadd_pd(ks5, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks7, di, qr2); qi2 = _mm512_fnmadd_pd(ks7, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks2, di, qr3); qi3 = _mm512_fnmadd_pd(ks2, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks3, di, qr4); qi4 = _mm512_fmadd_pd(ks3, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks8, di, qr5); qi5 = _mm512_fmadd_pd(ks8, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks4, di, qr6); qi6 = _mm512_fnmadd_pd(ks4, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr1 = _mm512_fmadd_pd(ks6, di, qr1); qi1 = _mm512_fmadd_pd(ks6, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks5, di, qr2); qi2 = _mm512_fnmadd_pd(ks5, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks1, di, qr3); qi3 = _mm512_fmadd_pd(ks1, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks7, di, qr4); qi4 = _mm512_fmadd_pd(ks7, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks4, di, qr5); qi5 = _mm512_fnmadd_pd(ks4, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks2, di, qr6); qi6 = _mm512_fmadd_pd(ks2, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+96), di = _mm512_load_pd(dscr+104);
    qr1 = _mm512_fmadd_pd(ks7, di, qr1); qi1 = _mm512_fmadd_pd(ks7, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks3, di, qr2); qi2 = _mm512_fnmadd_pd(ks3, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks4, di, qr3); qi3 = _mm512_fmadd_pd(ks4, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks6, di, qr4); qi4 = _mm512_fnmadd_pd(ks6, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks1, di, qr5); qi5 = _mm512_fmadd_pd(ks1, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks8, di, qr6); qi6 = _mm512_fmadd_pd(ks8, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+112), di = _mm512_load_pd(dscr+120);
    qr1 = _mm512_fmadd_pd(ks8, di, qr1); qi1 = _mm512_fmadd_pd(ks8, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks1, di, qr2); qi2 = _mm512_fnmadd_pd(ks1, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks7, di, qr3); qi3 = _mm512_fmadd_pd(ks7, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks2, di, qr4); qi4 = _mm512_fnmadd_pd(ks2, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks6, di, qr5); qi5 = _mm512_fmadd_pd(ks6, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks3, di, qr6); qi6 = _mm512_fnmadd_pd(ks3, dr, qi6);
    }
    _mm512_store_pd(qscr+0, qr1); _mm512_store_pd(qscr+8, qi1);
    _mm512_store_pd(qscr+16, qr2); _mm512_store_pd(qscr+24, qi2);
    _mm512_store_pd(qscr+32, qr3); _mm512_store_pd(qscr+40, qi3);
    _mm512_store_pd(qscr+48, qr4); _mm512_store_pd(qscr+56, qi4);
    _mm512_store_pd(qscr+64, qr5); _mm512_store_pd(qscr+72, qi5);
    _mm512_store_pd(qscr+80, qr6); _mm512_store_pd(qscr+88, qi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_17[0]);
    __m512d ks2; BCV(ks2, KS_17[1]);
    __m512d ks3; BCV(ks3, KS_17[2]);
    __m512d ks4; BCV(ks4, KS_17[3]);
    __m512d ks5; BCV(ks5, KS_17[4]);
    __m512d ks6; BCV(ks6, KS_17[5]);
    __m512d ks7; BCV(ks7, KS_17[6]);
    __m512d ks8; BCV(ks8, KS_17[7]);
    __m512d qr7, qi7;
    __m512d qr8, qi8;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr7 = _mm512_mul_pd(ks7, di); qi7 = _mm512_mul_pd(ks7, dr);
    qr8 = _mm512_mul_pd(ks8, di); qi8 = _mm512_mul_pd(ks8, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr7 = _mm512_fnmadd_pd(ks3, di, qr7); qi7 = _mm512_fnmadd_pd(ks3, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks1, di, qr8); qi8 = _mm512_fnmadd_pd(ks1, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr7 = _mm512_fmadd_pd(ks4, di, qr7); qi7 = _mm512_fmadd_pd(ks4, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks7, di, qr8); qi8 = _mm512_fmadd_pd(ks7, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr7 = _mm512_fnmadd_pd(ks6, di, qr7); qi7 = _mm512_fnmadd_pd(ks6, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks2, di, qr8); qi8 = _mm512_fnmadd_pd(ks2, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr7 = _mm512_fmadd_pd(ks1, di, qr7); qi7 = _mm512_fmadd_pd(ks1, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks6, di, qr8); qi8 = _mm512_fmadd_pd(ks6, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr7 = _mm512_fmadd_pd(ks8, di, qr7); qi7 = _mm512_fmadd_pd(ks8, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks3, di, qr8); qi8 = _mm512_fnmadd_pd(ks3, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+96), di = _mm512_load_pd(dscr+104);
    qr7 = _mm512_fnmadd_pd(ks2, di, qr7); qi7 = _mm512_fnmadd_pd(ks2, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks5, di, qr8); qi8 = _mm512_fmadd_pd(ks5, dr, qi8);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+112), di = _mm512_load_pd(dscr+120);
    qr7 = _mm512_fmadd_pd(ks5, di, qr7); qi7 = _mm512_fmadd_pd(ks5, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks4, di, qr8); qi8 = _mm512_fnmadd_pd(ks4, dr, qi8);
    }
    _mm512_store_pd(qscr+96, qr7); _mm512_store_pd(qscr+104, qi7);
    _mm512_store_pd(qscr+112, qr8); _mm512_store_pd(qscr+120, qi8);
    }
    __asm__ volatile("" ::: "memory");
    {
    {
    __m512d Pr = _mm512_load_pd(pscr+16), Pi = _mm512_load_pd(pscr+24);
    __m512d Qr = _mm512_load_pd(qscr+0), Qi = _mm512_load_pd(qscr+8);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 4624, cb, 16);
    MAPST(yr, yi, x, 73984, cb, 256);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+32), Pi = _mm512_load_pd(pscr+40);
    __m512d Qr = _mm512_load_pd(qscr+16), Qi = _mm512_load_pd(qscr+24);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 9248, cb, 32);
    MAPST(yr, yi, x, 69360, cb, 240);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+48), Pi = _mm512_load_pd(pscr+56);
    __m512d Qr = _mm512_load_pd(qscr+32), Qi = _mm512_load_pd(qscr+40);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 13872, cb, 48);
    MAPST(yr, yi, x, 64736, cb, 224);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+64), Pi = _mm512_load_pd(pscr+72);
    __m512d Qr = _mm512_load_pd(qscr+48), Qi = _mm512_load_pd(qscr+56);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 18496, cb, 64);
    MAPST(yr, yi, x, 60112, cb, 208);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+80), Pi = _mm512_load_pd(pscr+88);
    __m512d Qr = _mm512_load_pd(qscr+64), Qi = _mm512_load_pd(qscr+72);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 23120, cb, 80);
    MAPST(yr, yi, x, 55488, cb, 192);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+96), Pi = _mm512_load_pd(pscr+104);
    __m512d Qr = _mm512_load_pd(qscr+80), Qi = _mm512_load_pd(qscr+88);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 27744, cb, 96);
    MAPST(yr, yi, x, 50864, cb, 176);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+112), Pi = _mm512_load_pd(pscr+120);
    __m512d Qr = _mm512_load_pd(qscr+96), Qi = _mm512_load_pd(qscr+104);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 32368, cb, 112);
    MAPST(yr, yi, x, 46240, cb, 160);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+128), Pi = _mm512_load_pd(pscr+136);
    __m512d Qr = _mm512_load_pd(qscr+112), Qi = _mm512_load_pd(qscr+120);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 36992, cb, 128);
    MAPST(yr, yi, x, 41616, cb, 144);
    }
    }
}

static void step2_17(double* restrict G, double* restrict G2, const double* restrict CP){
    (void)G2;
    for(int x=0; x<17; x++){
        double* pl = G + (long)x*289*16;
        for(int y=0; y<17; y++) dz_17(pl + (long)y*17*16);
        for(int z=0; z<17; z++) dy_17(pl + (long)z*16);
    }
    for(int e=0; e<289; e++)
        dx_17(G + (long)e*16, CP + (long)e*17*16);
}


void run2_17(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if(!G_17){ G_17 = alloc_arena(4913*16*8); G2_17 = alloc_arena(4913*16*8 + 4096) + 256; CP_17 = alloc_arena(4913*16*8 + 65536) + 128; CT_17 = alloc_arena(4913*16*8); }
    long G8 = B/8;
    for(long g=0; g<G8; g++){
        const double* sx[8]; const double* sc[8]; double* d1[8]; double* dm[8];
        for(int v=0; v<8; v++){
            long off = (g*8+v)*(long)4913*2;
            sx[v] = x0+off; sc[v] = c+off; d1[v] = out1+off; dm[v] = outm+off;
        }
        conv_in_17(sx, G_17);
        conv_in_17(sc, CT_17);
        for(long e=0; e<289; e++)
            for(int j=0; j<17; j++)
                memcpy(CP_17 + (e*(long)17 + j)*16, CT_17 + ((long)j*289 + e)*16, 128);
        for(long t=0; t<m; t++){
            step2_17(G_17, G2_17, CP_17);
            if(t==0 && m>1) conv_out_17(G_17, d1);
        }
        conv_out_17(G_17, dm);
        if(m==1) for(int v=0; v<8; v++) memcpy(d1[v], dm[v], 4913*16);
    }
}


static void conv_in_23(const double* const* src, double* restrict G){
    for(long e=0; e+4<=12167; e+=4){
        __m512d r0=_mm512_loadu_pd(src[0]+2*e), r1=_mm512_loadu_pd(src[1]+2*e);
        __m512d r2=_mm512_loadu_pd(src[2]+2*e), r3=_mm512_loadu_pd(src[3]+2*e);
        __m512d r4=_mm512_loadu_pd(src[4]+2*e), r5=_mm512_loadu_pd(src[5]+2*e);
        __m512d r6=_mm512_loadu_pd(src[6]+2*e), r7=_mm512_loadu_pd(src[7]+2*e);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        double* g = G + e*16;
        _mm512_store_pd(g,o0); _mm512_store_pd(g+8,o1);
        _mm512_store_pd(g+16,o2); _mm512_store_pd(g+24,o3);
        _mm512_store_pd(g+32,o4); _mm512_store_pd(g+40,o5);
        _mm512_store_pd(g+48,o6); _mm512_store_pd(g+56,o7);
    }
    for(long e=12164; e<12167; e++)
        for(int v=0; v<8; v++){ G[e*16+v] = src[v][2*e]; G[e*16+8+v] = src[v][2*e+1]; }
}
static void conv_out_23(const double* restrict G, double* const* dst){
    for(long e=0; e+4<=12167; e+=4){
        const double* g = G + e*16;
        __m512d r0=_mm512_load_pd(g),    r1=_mm512_load_pd(g+8);
        __m512d r2=_mm512_load_pd(g+16), r3=_mm512_load_pd(g+24);
        __m512d r4=_mm512_load_pd(g+32), r5=_mm512_load_pd(g+40);
        __m512d r6=_mm512_load_pd(g+48), r7=_mm512_load_pd(g+56);
        __m512d o0,o1,o2,o3,o4,o5,o6,o7;
        TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
        _mm512_storeu_pd(dst[0]+2*e,o0); _mm512_storeu_pd(dst[1]+2*e,o1);
        _mm512_storeu_pd(dst[2]+2*e,o2); _mm512_storeu_pd(dst[3]+2*e,o3);
        _mm512_storeu_pd(dst[4]+2*e,o4); _mm512_storeu_pd(dst[5]+2*e,o5);
        _mm512_storeu_pd(dst[6]+2*e,o6); _mm512_storeu_pd(dst[7]+2*e,o7);
    }
    for(long e=12164; e<12167; e++)
        for(int v=0; v<8; v++){ dst[v][2*e] = G[e*16+v]; dst[v][2*e+1] = G[e*16+8+v]; }
}

static double* G_23 = 0;
static double* G2_23 = 0;
static double* CP_23 = 0;
static double* CT_23 = 0;
static const double KC_23[11] ALIGN64 = { 0x1.ed037ea3d2dbcp-1, 0x1.b57675cf309eep-1, 0x1.5d779b07cfef7p-1, 0x1.d71b4a0c5a6c9p-2, 0x1.a0ad8bd1e2881p-3, -0x1.17855b599f3b2p-4, -0x1.56eaae597c776p-2, -0x1.2742a4a775cfap-1, -0x1.8d2a07c16d46ep-1, -0x1.d59cb83ef99bcp-1, -0x1.fb3b3035aa6ccp-1 };
static const double KS_23[11] ALIGN64 = { 0x1.14459ad2be466p-2, 0x1.0a06e851db7cap-1, 0x1.763021aaa15d9p-1, 0x1.c698e42f47b09p-1, 0x1.f54a827142577p-1, 0x1.fece70dfd3efbp-1, 0x1.e270060999288p-1, 0x1.a249e0b897caap-1, 0x1.431df5838f7f1p-1, 0x1.97f6748e524b1p-2, 0x1.16de8a4564f1cp-3 };

static void __attribute__((noinline)) dz_23(double* restrict x){
    double sscr[22*8] ALIGN64;
    double dscr[22*8] ALIGN64;
    double pscr[22*8+16] ALIGN64;
    double qscr[22*8] ALIGN64;
    {
    __m512d u0r = _mm512_load_pd(x), u0i = _mm512_load_pd(x+8);
    __m512d x0r = u0r, x0i = u0i;
    {
    __m512d ar = _mm512_load_pd(x+16), ai = _mm512_load_pd(x+16+8);
    __m512d br = _mm512_load_pd(x+352), bi = _mm512_load_pd(x+352+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+0, sr); _mm512_store_pd(sscr+8, si);
    _mm512_store_pd(dscr+0, dr); _mm512_store_pd(dscr+8, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+32), ai = _mm512_load_pd(x+32+8);
    __m512d br = _mm512_load_pd(x+336), bi = _mm512_load_pd(x+336+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+16, sr); _mm512_store_pd(sscr+24, si);
    _mm512_store_pd(dscr+16, dr); _mm512_store_pd(dscr+24, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+48), ai = _mm512_load_pd(x+48+8);
    __m512d br = _mm512_load_pd(x+320), bi = _mm512_load_pd(x+320+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+32, sr); _mm512_store_pd(sscr+40, si);
    _mm512_store_pd(dscr+32, dr); _mm512_store_pd(dscr+40, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+64), ai = _mm512_load_pd(x+64+8);
    __m512d br = _mm512_load_pd(x+304), bi = _mm512_load_pd(x+304+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+48, sr); _mm512_store_pd(sscr+56, si);
    _mm512_store_pd(dscr+48, dr); _mm512_store_pd(dscr+56, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+80), ai = _mm512_load_pd(x+80+8);
    __m512d br = _mm512_load_pd(x+288), bi = _mm512_load_pd(x+288+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+64, sr); _mm512_store_pd(sscr+72, si);
    _mm512_store_pd(dscr+64, dr); _mm512_store_pd(dscr+72, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+96), ai = _mm512_load_pd(x+96+8);
    __m512d br = _mm512_load_pd(x+272), bi = _mm512_load_pd(x+272+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+80, sr); _mm512_store_pd(sscr+88, si);
    _mm512_store_pd(dscr+80, dr); _mm512_store_pd(dscr+88, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+112), ai = _mm512_load_pd(x+112+8);
    __m512d br = _mm512_load_pd(x+256), bi = _mm512_load_pd(x+256+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+96, sr); _mm512_store_pd(sscr+104, si);
    _mm512_store_pd(dscr+96, dr); _mm512_store_pd(dscr+104, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+128), ai = _mm512_load_pd(x+128+8);
    __m512d br = _mm512_load_pd(x+240), bi = _mm512_load_pd(x+240+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+112, sr); _mm512_store_pd(sscr+120, si);
    _mm512_store_pd(dscr+112, dr); _mm512_store_pd(dscr+120, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+144), ai = _mm512_load_pd(x+144+8);
    __m512d br = _mm512_load_pd(x+224), bi = _mm512_load_pd(x+224+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+128, sr); _mm512_store_pd(sscr+136, si);
    _mm512_store_pd(dscr+128, dr); _mm512_store_pd(dscr+136, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+160), ai = _mm512_load_pd(x+160+8);
    __m512d br = _mm512_load_pd(x+208), bi = _mm512_load_pd(x+208+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+144, sr); _mm512_store_pd(sscr+152, si);
    _mm512_store_pd(dscr+144, dr); _mm512_store_pd(dscr+152, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+176), ai = _mm512_load_pd(x+176+8);
    __m512d br = _mm512_load_pd(x+192), bi = _mm512_load_pd(x+192+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+160, sr); _mm512_store_pd(sscr+168, si);
    _mm512_store_pd(dscr+160, dr); _mm512_store_pd(dscr+168, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    _mm512_store_pd(x, x0r); _mm512_store_pd(x+8, x0i);
    __m512d u0r_s = u0r, u0i_s = u0i;
    _mm512_store_pd(pscr, u0r_s); _mm512_store_pd(pscr+8, u0i_s);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d u0r = _mm512_load_pd(x+0*0), u0i;
    u0r = _mm512_load_pd(pscr); u0i = _mm512_load_pd(pscr+8);
    __m512d kc1; BCV(kc1, KC_23[0]);
    __m512d kc2; BCV(kc2, KC_23[1]);
    __m512d kc3; BCV(kc3, KC_23[2]);
    __m512d kc4; BCV(kc4, KC_23[3]);
    __m512d kc5; BCV(kc5, KC_23[4]);
    __m512d kc6; BCV(kc6, KC_23[5]);
    __m512d kc7; BCV(kc7, KC_23[6]);
    __m512d kc8; BCV(kc8, KC_23[7]);
    __m512d kc9; BCV(kc9, KC_23[8]);
    __m512d kc10; BCV(kc10, KC_23[9]);
    __m512d kc11; BCV(kc11, KC_23[10]);
    __m512d pr1 = u0r, pi1 = u0i;
    __m512d pr2 = u0r, pi2 = u0i;
    __m512d pr3 = u0r, pi3 = u0i;
    __m512d pr4 = u0r, pi4 = u0i;
    __m512d pr5 = u0r, pi5 = u0i;
    __m512d pr6 = u0r, pi6 = u0i;
    {
    __m512d sr = _mm512_load_pd(sscr+0), si = _mm512_load_pd(sscr+8);
    pr1 = _mm512_fmadd_pd(kc1, sr, pr1); pi1 = _mm512_fmadd_pd(kc1, si, pi1);
    pr2 = _mm512_fmadd_pd(kc2, sr, pr2); pi2 = _mm512_fmadd_pd(kc2, si, pi2);
    pr3 = _mm512_fmadd_pd(kc3, sr, pr3); pi3 = _mm512_fmadd_pd(kc3, si, pi3);
    pr4 = _mm512_fmadd_pd(kc4, sr, pr4); pi4 = _mm512_fmadd_pd(kc4, si, pi4);
    pr5 = _mm512_fmadd_pd(kc5, sr, pr5); pi5 = _mm512_fmadd_pd(kc5, si, pi5);
    pr6 = _mm512_fmadd_pd(kc6, sr, pr6); pi6 = _mm512_fmadd_pd(kc6, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+16), si = _mm512_load_pd(sscr+24);
    pr1 = _mm512_fmadd_pd(kc2, sr, pr1); pi1 = _mm512_fmadd_pd(kc2, si, pi1);
    pr2 = _mm512_fmadd_pd(kc4, sr, pr2); pi2 = _mm512_fmadd_pd(kc4, si, pi2);
    pr3 = _mm512_fmadd_pd(kc6, sr, pr3); pi3 = _mm512_fmadd_pd(kc6, si, pi3);
    pr4 = _mm512_fmadd_pd(kc8, sr, pr4); pi4 = _mm512_fmadd_pd(kc8, si, pi4);
    pr5 = _mm512_fmadd_pd(kc10, sr, pr5); pi5 = _mm512_fmadd_pd(kc10, si, pi5);
    pr6 = _mm512_fmadd_pd(kc11, sr, pr6); pi6 = _mm512_fmadd_pd(kc11, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+32), si = _mm512_load_pd(sscr+40);
    pr1 = _mm512_fmadd_pd(kc3, sr, pr1); pi1 = _mm512_fmadd_pd(kc3, si, pi1);
    pr2 = _mm512_fmadd_pd(kc6, sr, pr2); pi2 = _mm512_fmadd_pd(kc6, si, pi2);
    pr3 = _mm512_fmadd_pd(kc9, sr, pr3); pi3 = _mm512_fmadd_pd(kc9, si, pi3);
    pr4 = _mm512_fmadd_pd(kc11, sr, pr4); pi4 = _mm512_fmadd_pd(kc11, si, pi4);
    pr5 = _mm512_fmadd_pd(kc8, sr, pr5); pi5 = _mm512_fmadd_pd(kc8, si, pi5);
    pr6 = _mm512_fmadd_pd(kc5, sr, pr6); pi6 = _mm512_fmadd_pd(kc5, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+48), si = _mm512_load_pd(sscr+56);
    pr1 = _mm512_fmadd_pd(kc4, sr, pr1); pi1 = _mm512_fmadd_pd(kc4, si, pi1);
    pr2 = _mm512_fmadd_pd(kc8, sr, pr2); pi2 = _mm512_fmadd_pd(kc8, si, pi2);
    pr3 = _mm512_fmadd_pd(kc11, sr, pr3); pi3 = _mm512_fmadd_pd(kc11, si, pi3);
    pr4 = _mm512_fmadd_pd(kc7, sr, pr4); pi4 = _mm512_fmadd_pd(kc7, si, pi4);
    pr5 = _mm512_fmadd_pd(kc3, sr, pr5); pi5 = _mm512_fmadd_pd(kc3, si, pi5);
    pr6 = _mm512_fmadd_pd(kc1, sr, pr6); pi6 = _mm512_fmadd_pd(kc1, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+64), si = _mm512_load_pd(sscr+72);
    pr1 = _mm512_fmadd_pd(kc5, sr, pr1); pi1 = _mm512_fmadd_pd(kc5, si, pi1);
    pr2 = _mm512_fmadd_pd(kc10, sr, pr2); pi2 = _mm512_fmadd_pd(kc10, si, pi2);
    pr3 = _mm512_fmadd_pd(kc8, sr, pr3); pi3 = _mm512_fmadd_pd(kc8, si, pi3);
    pr4 = _mm512_fmadd_pd(kc3, sr, pr4); pi4 = _mm512_fmadd_pd(kc3, si, pi4);
    pr5 = _mm512_fmadd_pd(kc2, sr, pr5); pi5 = _mm512_fmadd_pd(kc2, si, pi5);
    pr6 = _mm512_fmadd_pd(kc7, sr, pr6); pi6 = _mm512_fmadd_pd(kc7, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+80), si = _mm512_load_pd(sscr+88);
    pr1 = _mm512_fmadd_pd(kc6, sr, pr1); pi1 = _mm512_fmadd_pd(kc6, si, pi1);
    pr2 = _mm512_fmadd_pd(kc11, sr, pr2); pi2 = _mm512_fmadd_pd(kc11, si, pi2);
    pr3 = _mm512_fmadd_pd(kc5, sr, pr3); pi3 = _mm512_fmadd_pd(kc5, si, pi3);
    pr4 = _mm512_fmadd_pd(kc1, sr, pr4); pi4 = _mm512_fmadd_pd(kc1, si, pi4);
    pr5 = _mm512_fmadd_pd(kc7, sr, pr5); pi5 = _mm512_fmadd_pd(kc7, si, pi5);
    pr6 = _mm512_fmadd_pd(kc10, sr, pr6); pi6 = _mm512_fmadd_pd(kc10, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+96), si = _mm512_load_pd(sscr+104);
    pr1 = _mm512_fmadd_pd(kc7, sr, pr1); pi1 = _mm512_fmadd_pd(kc7, si, pi1);
    pr2 = _mm512_fmadd_pd(kc9, sr, pr2); pi2 = _mm512_fmadd_pd(kc9, si, pi2);
    pr3 = _mm512_fmadd_pd(kc2, sr, pr3); pi3 = _mm512_fmadd_pd(kc2, si, pi3);
    pr4 = _mm512_fmadd_pd(kc5, sr, pr4); pi4 = _mm512_fmadd_pd(kc5, si, pi4);
    pr5 = _mm512_fmadd_pd(kc11, sr, pr5); pi5 = _mm512_fmadd_pd(kc11, si, pi5);
    pr6 = _mm512_fmadd_pd(kc4, sr, pr6); pi6 = _mm512_fmadd_pd(kc4, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+112), si = _mm512_load_pd(sscr+120);
    pr1 = _mm512_fmadd_pd(kc8, sr, pr1); pi1 = _mm512_fmadd_pd(kc8, si, pi1);
    pr2 = _mm512_fmadd_pd(kc7, sr, pr2); pi2 = _mm512_fmadd_pd(kc7, si, pi2);
    pr3 = _mm512_fmadd_pd(kc1, sr, pr3); pi3 = _mm512_fmadd_pd(kc1, si, pi3);
    pr4 = _mm512_fmadd_pd(kc9, sr, pr4); pi4 = _mm512_fmadd_pd(kc9, si, pi4);
    pr5 = _mm512_fmadd_pd(kc6, sr, pr5); pi5 = _mm512_fmadd_pd(kc6, si, pi5);
    pr6 = _mm512_fmadd_pd(kc2, sr, pr6); pi6 = _mm512_fmadd_pd(kc2, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+128), si = _mm512_load_pd(sscr+136);
    pr1 = _mm512_fmadd_pd(kc9, sr, pr1); pi1 = _mm512_fmadd_pd(kc9, si, pi1);
    pr2 = _mm512_fmadd_pd(kc5, sr, pr2); pi2 = _mm512_fmadd_pd(kc5, si, pi2);
    pr3 = _mm512_fmadd_pd(kc4, sr, pr3); pi3 = _mm512_fmadd_pd(kc4, si, pi3);
    pr4 = _mm512_fmadd_pd(kc10, sr, pr4); pi4 = _mm512_fmadd_pd(kc10, si, pi4);
    pr5 = _mm512_fmadd_pd(kc1, sr, pr5); pi5 = _mm512_fmadd_pd(kc1, si, pi5);
    pr6 = _mm512_fmadd_pd(kc8, sr, pr6); pi6 = _mm512_fmadd_pd(kc8, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+144), si = _mm512_load_pd(sscr+152);
    pr1 = _mm512_fmadd_pd(kc10, sr, pr1); pi1 = _mm512_fmadd_pd(kc10, si, pi1);
    pr2 = _mm512_fmadd_pd(kc3, sr, pr2); pi2 = _mm512_fmadd_pd(kc3, si, pi2);
    pr3 = _mm512_fmadd_pd(kc7, sr, pr3); pi3 = _mm512_fmadd_pd(kc7, si, pi3);
    pr4 = _mm512_fmadd_pd(kc6, sr, pr4); pi4 = _mm512_fmadd_pd(kc6, si, pi4);
    pr5 = _mm512_fmadd_pd(kc4, sr, pr5); pi5 = _mm512_fmadd_pd(kc4, si, pi5);
    pr6 = _mm512_fmadd_pd(kc9, sr, pr6); pi6 = _mm512_fmadd_pd(kc9, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+160), si = _mm512_load_pd(sscr+168);
    pr1 = _mm512_fmadd_pd(kc11, sr, pr1); pi1 = _mm512_fmadd_pd(kc11, si, pi1);
    pr2 = _mm512_fmadd_pd(kc1, sr, pr2); pi2 = _mm512_fmadd_pd(kc1, si, pi2);
    pr3 = _mm512_fmadd_pd(kc10, sr, pr3); pi3 = _mm512_fmadd_pd(kc10, si, pi3);
    pr4 = _mm512_fmadd_pd(kc2, sr, pr4); pi4 = _mm512_fmadd_pd(kc2, si, pi4);
    pr5 = _mm512_fmadd_pd(kc9, sr, pr5); pi5 = _mm512_fmadd_pd(kc9, si, pi5);
    pr6 = _mm512_fmadd_pd(kc3, sr, pr6); pi6 = _mm512_fmadd_pd(kc3, si, pi6);
    }
    _mm512_store_pd(pscr+16, pr1); _mm512_store_pd(pscr+24, pi1);
    _mm512_store_pd(pscr+32, pr2); _mm512_store_pd(pscr+40, pi2);
    _mm512_store_pd(pscr+48, pr3); _mm512_store_pd(pscr+56, pi3);
    _mm512_store_pd(pscr+64, pr4); _mm512_store_pd(pscr+72, pi4);
    _mm512_store_pd(pscr+80, pr5); _mm512_store_pd(pscr+88, pi5);
    _mm512_store_pd(pscr+96, pr6); _mm512_store_pd(pscr+104, pi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d u0r = _mm512_load_pd(x+0*0), u0i;
    u0r = _mm512_load_pd(pscr); u0i = _mm512_load_pd(pscr+8);
    __m512d kc1; BCV(kc1, KC_23[0]);
    __m512d kc2; BCV(kc2, KC_23[1]);
    __m512d kc3; BCV(kc3, KC_23[2]);
    __m512d kc4; BCV(kc4, KC_23[3]);
    __m512d kc5; BCV(kc5, KC_23[4]);
    __m512d kc6; BCV(kc6, KC_23[5]);
    __m512d kc7; BCV(kc7, KC_23[6]);
    __m512d kc8; BCV(kc8, KC_23[7]);
    __m512d kc9; BCV(kc9, KC_23[8]);
    __m512d kc10; BCV(kc10, KC_23[9]);
    __m512d kc11; BCV(kc11, KC_23[10]);
    __m512d pr7 = u0r, pi7 = u0i;
    __m512d pr8 = u0r, pi8 = u0i;
    __m512d pr9 = u0r, pi9 = u0i;
    __m512d pr10 = u0r, pi10 = u0i;
    __m512d pr11 = u0r, pi11 = u0i;
    {
    __m512d sr = _mm512_load_pd(sscr+0), si = _mm512_load_pd(sscr+8);
    pr7 = _mm512_fmadd_pd(kc7, sr, pr7); pi7 = _mm512_fmadd_pd(kc7, si, pi7);
    pr8 = _mm512_fmadd_pd(kc8, sr, pr8); pi8 = _mm512_fmadd_pd(kc8, si, pi8);
    pr9 = _mm512_fmadd_pd(kc9, sr, pr9); pi9 = _mm512_fmadd_pd(kc9, si, pi9);
    pr10 = _mm512_fmadd_pd(kc10, sr, pr10); pi10 = _mm512_fmadd_pd(kc10, si, pi10);
    pr11 = _mm512_fmadd_pd(kc11, sr, pr11); pi11 = _mm512_fmadd_pd(kc11, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+16), si = _mm512_load_pd(sscr+24);
    pr7 = _mm512_fmadd_pd(kc9, sr, pr7); pi7 = _mm512_fmadd_pd(kc9, si, pi7);
    pr8 = _mm512_fmadd_pd(kc7, sr, pr8); pi8 = _mm512_fmadd_pd(kc7, si, pi8);
    pr9 = _mm512_fmadd_pd(kc5, sr, pr9); pi9 = _mm512_fmadd_pd(kc5, si, pi9);
    pr10 = _mm512_fmadd_pd(kc3, sr, pr10); pi10 = _mm512_fmadd_pd(kc3, si, pi10);
    pr11 = _mm512_fmadd_pd(kc1, sr, pr11); pi11 = _mm512_fmadd_pd(kc1, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+32), si = _mm512_load_pd(sscr+40);
    pr7 = _mm512_fmadd_pd(kc2, sr, pr7); pi7 = _mm512_fmadd_pd(kc2, si, pi7);
    pr8 = _mm512_fmadd_pd(kc1, sr, pr8); pi8 = _mm512_fmadd_pd(kc1, si, pi8);
    pr9 = _mm512_fmadd_pd(kc4, sr, pr9); pi9 = _mm512_fmadd_pd(kc4, si, pi9);
    pr10 = _mm512_fmadd_pd(kc7, sr, pr10); pi10 = _mm512_fmadd_pd(kc7, si, pi10);
    pr11 = _mm512_fmadd_pd(kc10, sr, pr11); pi11 = _mm512_fmadd_pd(kc10, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+48), si = _mm512_load_pd(sscr+56);
    pr7 = _mm512_fmadd_pd(kc5, sr, pr7); pi7 = _mm512_fmadd_pd(kc5, si, pi7);
    pr8 = _mm512_fmadd_pd(kc9, sr, pr8); pi8 = _mm512_fmadd_pd(kc9, si, pi8);
    pr9 = _mm512_fmadd_pd(kc10, sr, pr9); pi9 = _mm512_fmadd_pd(kc10, si, pi9);
    pr10 = _mm512_fmadd_pd(kc6, sr, pr10); pi10 = _mm512_fmadd_pd(kc6, si, pi10);
    pr11 = _mm512_fmadd_pd(kc2, sr, pr11); pi11 = _mm512_fmadd_pd(kc2, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+64), si = _mm512_load_pd(sscr+72);
    pr7 = _mm512_fmadd_pd(kc11, sr, pr7); pi7 = _mm512_fmadd_pd(kc11, si, pi7);
    pr8 = _mm512_fmadd_pd(kc6, sr, pr8); pi8 = _mm512_fmadd_pd(kc6, si, pi8);
    pr9 = _mm512_fmadd_pd(kc1, sr, pr9); pi9 = _mm512_fmadd_pd(kc1, si, pi9);
    pr10 = _mm512_fmadd_pd(kc4, sr, pr10); pi10 = _mm512_fmadd_pd(kc4, si, pi10);
    pr11 = _mm512_fmadd_pd(kc9, sr, pr11); pi11 = _mm512_fmadd_pd(kc9, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+80), si = _mm512_load_pd(sscr+88);
    pr7 = _mm512_fmadd_pd(kc4, sr, pr7); pi7 = _mm512_fmadd_pd(kc4, si, pi7);
    pr8 = _mm512_fmadd_pd(kc2, sr, pr8); pi8 = _mm512_fmadd_pd(kc2, si, pi8);
    pr9 = _mm512_fmadd_pd(kc8, sr, pr9); pi9 = _mm512_fmadd_pd(kc8, si, pi9);
    pr10 = _mm512_fmadd_pd(kc9, sr, pr10); pi10 = _mm512_fmadd_pd(kc9, si, pi10);
    pr11 = _mm512_fmadd_pd(kc3, sr, pr11); pi11 = _mm512_fmadd_pd(kc3, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+96), si = _mm512_load_pd(sscr+104);
    pr7 = _mm512_fmadd_pd(kc3, sr, pr7); pi7 = _mm512_fmadd_pd(kc3, si, pi7);
    pr8 = _mm512_fmadd_pd(kc10, sr, pr8); pi8 = _mm512_fmadd_pd(kc10, si, pi8);
    pr9 = _mm512_fmadd_pd(kc6, sr, pr9); pi9 = _mm512_fmadd_pd(kc6, si, pi9);
    pr10 = _mm512_fmadd_pd(kc1, sr, pr10); pi10 = _mm512_fmadd_pd(kc1, si, pi10);
    pr11 = _mm512_fmadd_pd(kc8, sr, pr11); pi11 = _mm512_fmadd_pd(kc8, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+112), si = _mm512_load_pd(sscr+120);
    pr7 = _mm512_fmadd_pd(kc10, sr, pr7); pi7 = _mm512_fmadd_pd(kc10, si, pi7);
    pr8 = _mm512_fmadd_pd(kc5, sr, pr8); pi8 = _mm512_fmadd_pd(kc5, si, pi8);
    pr9 = _mm512_fmadd_pd(kc3, sr, pr9); pi9 = _mm512_fmadd_pd(kc3, si, pi9);
    pr10 = _mm512_fmadd_pd(kc11, sr, pr10); pi10 = _mm512_fmadd_pd(kc11, si, pi10);
    pr11 = _mm512_fmadd_pd(kc4, sr, pr11); pi11 = _mm512_fmadd_pd(kc4, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+128), si = _mm512_load_pd(sscr+136);
    pr7 = _mm512_fmadd_pd(kc6, sr, pr7); pi7 = _mm512_fmadd_pd(kc6, si, pi7);
    pr8 = _mm512_fmadd_pd(kc3, sr, pr8); pi8 = _mm512_fmadd_pd(kc3, si, pi8);
    pr9 = _mm512_fmadd_pd(kc11, sr, pr9); pi9 = _mm512_fmadd_pd(kc11, si, pi9);
    pr10 = _mm512_fmadd_pd(kc2, sr, pr10); pi10 = _mm512_fmadd_pd(kc2, si, pi10);
    pr11 = _mm512_fmadd_pd(kc7, sr, pr11); pi11 = _mm512_fmadd_pd(kc7, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+144), si = _mm512_load_pd(sscr+152);
    pr7 = _mm512_fmadd_pd(kc1, sr, pr7); pi7 = _mm512_fmadd_pd(kc1, si, pi7);
    pr8 = _mm512_fmadd_pd(kc11, sr, pr8); pi8 = _mm512_fmadd_pd(kc11, si, pi8);
    pr9 = _mm512_fmadd_pd(kc2, sr, pr9); pi9 = _mm512_fmadd_pd(kc2, si, pi9);
    pr10 = _mm512_fmadd_pd(kc8, sr, pr10); pi10 = _mm512_fmadd_pd(kc8, si, pi10);
    pr11 = _mm512_fmadd_pd(kc5, sr, pr11); pi11 = _mm512_fmadd_pd(kc5, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+160), si = _mm512_load_pd(sscr+168);
    pr7 = _mm512_fmadd_pd(kc8, sr, pr7); pi7 = _mm512_fmadd_pd(kc8, si, pi7);
    pr8 = _mm512_fmadd_pd(kc4, sr, pr8); pi8 = _mm512_fmadd_pd(kc4, si, pi8);
    pr9 = _mm512_fmadd_pd(kc7, sr, pr9); pi9 = _mm512_fmadd_pd(kc7, si, pi9);
    pr10 = _mm512_fmadd_pd(kc5, sr, pr10); pi10 = _mm512_fmadd_pd(kc5, si, pi10);
    pr11 = _mm512_fmadd_pd(kc6, sr, pr11); pi11 = _mm512_fmadd_pd(kc6, si, pi11);
    }
    _mm512_store_pd(pscr+112, pr7); _mm512_store_pd(pscr+120, pi7);
    _mm512_store_pd(pscr+128, pr8); _mm512_store_pd(pscr+136, pi8);
    _mm512_store_pd(pscr+144, pr9); _mm512_store_pd(pscr+152, pi9);
    _mm512_store_pd(pscr+160, pr10); _mm512_store_pd(pscr+168, pi10);
    _mm512_store_pd(pscr+176, pr11); _mm512_store_pd(pscr+184, pi11);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_23[0]);
    __m512d ks2; BCV(ks2, KS_23[1]);
    __m512d ks3; BCV(ks3, KS_23[2]);
    __m512d ks4; BCV(ks4, KS_23[3]);
    __m512d ks5; BCV(ks5, KS_23[4]);
    __m512d ks6; BCV(ks6, KS_23[5]);
    __m512d ks7; BCV(ks7, KS_23[6]);
    __m512d ks8; BCV(ks8, KS_23[7]);
    __m512d ks9; BCV(ks9, KS_23[8]);
    __m512d ks10; BCV(ks10, KS_23[9]);
    __m512d ks11; BCV(ks11, KS_23[10]);
    __m512d qr1, qi1;
    __m512d qr2, qi2;
    __m512d qr3, qi3;
    __m512d qr4, qi4;
    __m512d qr5, qi5;
    __m512d qr6, qi6;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr1 = _mm512_mul_pd(ks1, di); qi1 = _mm512_mul_pd(ks1, dr);
    qr2 = _mm512_mul_pd(ks2, di); qi2 = _mm512_mul_pd(ks2, dr);
    qr3 = _mm512_mul_pd(ks3, di); qi3 = _mm512_mul_pd(ks3, dr);
    qr4 = _mm512_mul_pd(ks4, di); qi4 = _mm512_mul_pd(ks4, dr);
    qr5 = _mm512_mul_pd(ks5, di); qi5 = _mm512_mul_pd(ks5, dr);
    qr6 = _mm512_mul_pd(ks6, di); qi6 = _mm512_mul_pd(ks6, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr1 = _mm512_fmadd_pd(ks2, di, qr1); qi1 = _mm512_fmadd_pd(ks2, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks4, di, qr2); qi2 = _mm512_fmadd_pd(ks4, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks6, di, qr3); qi3 = _mm512_fmadd_pd(ks6, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks8, di, qr4); qi4 = _mm512_fmadd_pd(ks8, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks10, di, qr5); qi5 = _mm512_fmadd_pd(ks10, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks11, di, qr6); qi6 = _mm512_fnmadd_pd(ks11, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr1 = _mm512_fmadd_pd(ks3, di, qr1); qi1 = _mm512_fmadd_pd(ks3, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks6, di, qr2); qi2 = _mm512_fmadd_pd(ks6, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks9, di, qr3); qi3 = _mm512_fmadd_pd(ks9, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks11, di, qr4); qi4 = _mm512_fnmadd_pd(ks11, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks8, di, qr5); qi5 = _mm512_fnmadd_pd(ks8, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks5, di, qr6); qi6 = _mm512_fnmadd_pd(ks5, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr1 = _mm512_fmadd_pd(ks4, di, qr1); qi1 = _mm512_fmadd_pd(ks4, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks8, di, qr2); qi2 = _mm512_fmadd_pd(ks8, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks11, di, qr3); qi3 = _mm512_fnmadd_pd(ks11, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks7, di, qr4); qi4 = _mm512_fnmadd_pd(ks7, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks3, di, qr5); qi5 = _mm512_fnmadd_pd(ks3, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks1, di, qr6); qi6 = _mm512_fmadd_pd(ks1, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr1 = _mm512_fmadd_pd(ks5, di, qr1); qi1 = _mm512_fmadd_pd(ks5, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks10, di, qr2); qi2 = _mm512_fmadd_pd(ks10, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks8, di, qr3); qi3 = _mm512_fnmadd_pd(ks8, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks3, di, qr4); qi4 = _mm512_fnmadd_pd(ks3, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks2, di, qr5); qi5 = _mm512_fmadd_pd(ks2, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks7, di, qr6); qi6 = _mm512_fmadd_pd(ks7, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr1 = _mm512_fmadd_pd(ks6, di, qr1); qi1 = _mm512_fmadd_pd(ks6, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks11, di, qr2); qi2 = _mm512_fnmadd_pd(ks11, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks5, di, qr3); qi3 = _mm512_fnmadd_pd(ks5, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks1, di, qr4); qi4 = _mm512_fmadd_pd(ks1, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks7, di, qr5); qi5 = _mm512_fmadd_pd(ks7, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks10, di, qr6); qi6 = _mm512_fnmadd_pd(ks10, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+96), di = _mm512_load_pd(dscr+104);
    qr1 = _mm512_fmadd_pd(ks7, di, qr1); qi1 = _mm512_fmadd_pd(ks7, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks9, di, qr2); qi2 = _mm512_fnmadd_pd(ks9, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks2, di, qr3); qi3 = _mm512_fnmadd_pd(ks2, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks5, di, qr4); qi4 = _mm512_fmadd_pd(ks5, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks11, di, qr5); qi5 = _mm512_fnmadd_pd(ks11, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks4, di, qr6); qi6 = _mm512_fnmadd_pd(ks4, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+112), di = _mm512_load_pd(dscr+120);
    qr1 = _mm512_fmadd_pd(ks8, di, qr1); qi1 = _mm512_fmadd_pd(ks8, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks7, di, qr2); qi2 = _mm512_fnmadd_pd(ks7, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks1, di, qr3); qi3 = _mm512_fmadd_pd(ks1, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks9, di, qr4); qi4 = _mm512_fmadd_pd(ks9, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks6, di, qr5); qi5 = _mm512_fnmadd_pd(ks6, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks2, di, qr6); qi6 = _mm512_fmadd_pd(ks2, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+128), di = _mm512_load_pd(dscr+136);
    qr1 = _mm512_fmadd_pd(ks9, di, qr1); qi1 = _mm512_fmadd_pd(ks9, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks5, di, qr2); qi2 = _mm512_fnmadd_pd(ks5, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks4, di, qr3); qi3 = _mm512_fmadd_pd(ks4, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks10, di, qr4); qi4 = _mm512_fnmadd_pd(ks10, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks1, di, qr5); qi5 = _mm512_fnmadd_pd(ks1, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks8, di, qr6); qi6 = _mm512_fmadd_pd(ks8, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+144), di = _mm512_load_pd(dscr+152);
    qr1 = _mm512_fmadd_pd(ks10, di, qr1); qi1 = _mm512_fmadd_pd(ks10, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks3, di, qr2); qi2 = _mm512_fnmadd_pd(ks3, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks7, di, qr3); qi3 = _mm512_fmadd_pd(ks7, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks6, di, qr4); qi4 = _mm512_fnmadd_pd(ks6, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks4, di, qr5); qi5 = _mm512_fmadd_pd(ks4, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks9, di, qr6); qi6 = _mm512_fnmadd_pd(ks9, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+160), di = _mm512_load_pd(dscr+168);
    qr1 = _mm512_fmadd_pd(ks11, di, qr1); qi1 = _mm512_fmadd_pd(ks11, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks1, di, qr2); qi2 = _mm512_fnmadd_pd(ks1, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks10, di, qr3); qi3 = _mm512_fmadd_pd(ks10, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks2, di, qr4); qi4 = _mm512_fnmadd_pd(ks2, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks9, di, qr5); qi5 = _mm512_fmadd_pd(ks9, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks3, di, qr6); qi6 = _mm512_fnmadd_pd(ks3, dr, qi6);
    }
    _mm512_store_pd(qscr+0, qr1); _mm512_store_pd(qscr+8, qi1);
    _mm512_store_pd(qscr+16, qr2); _mm512_store_pd(qscr+24, qi2);
    _mm512_store_pd(qscr+32, qr3); _mm512_store_pd(qscr+40, qi3);
    _mm512_store_pd(qscr+48, qr4); _mm512_store_pd(qscr+56, qi4);
    _mm512_store_pd(qscr+64, qr5); _mm512_store_pd(qscr+72, qi5);
    _mm512_store_pd(qscr+80, qr6); _mm512_store_pd(qscr+88, qi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_23[0]);
    __m512d ks2; BCV(ks2, KS_23[1]);
    __m512d ks3; BCV(ks3, KS_23[2]);
    __m512d ks4; BCV(ks4, KS_23[3]);
    __m512d ks5; BCV(ks5, KS_23[4]);
    __m512d ks6; BCV(ks6, KS_23[5]);
    __m512d ks7; BCV(ks7, KS_23[6]);
    __m512d ks8; BCV(ks8, KS_23[7]);
    __m512d ks9; BCV(ks9, KS_23[8]);
    __m512d ks10; BCV(ks10, KS_23[9]);
    __m512d ks11; BCV(ks11, KS_23[10]);
    __m512d qr7, qi7;
    __m512d qr8, qi8;
    __m512d qr9, qi9;
    __m512d qr10, qi10;
    __m512d qr11, qi11;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr7 = _mm512_mul_pd(ks7, di); qi7 = _mm512_mul_pd(ks7, dr);
    qr8 = _mm512_mul_pd(ks8, di); qi8 = _mm512_mul_pd(ks8, dr);
    qr9 = _mm512_mul_pd(ks9, di); qi9 = _mm512_mul_pd(ks9, dr);
    qr10 = _mm512_mul_pd(ks10, di); qi10 = _mm512_mul_pd(ks10, dr);
    qr11 = _mm512_mul_pd(ks11, di); qi11 = _mm512_mul_pd(ks11, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr7 = _mm512_fnmadd_pd(ks9, di, qr7); qi7 = _mm512_fnmadd_pd(ks9, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks7, di, qr8); qi8 = _mm512_fnmadd_pd(ks7, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks5, di, qr9); qi9 = _mm512_fnmadd_pd(ks5, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks3, di, qr10); qi10 = _mm512_fnmadd_pd(ks3, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks1, di, qr11); qi11 = _mm512_fnmadd_pd(ks1, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr7 = _mm512_fnmadd_pd(ks2, di, qr7); qi7 = _mm512_fnmadd_pd(ks2, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks1, di, qr8); qi8 = _mm512_fmadd_pd(ks1, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks4, di, qr9); qi9 = _mm512_fmadd_pd(ks4, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks7, di, qr10); qi10 = _mm512_fmadd_pd(ks7, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks10, di, qr11); qi11 = _mm512_fmadd_pd(ks10, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr7 = _mm512_fmadd_pd(ks5, di, qr7); qi7 = _mm512_fmadd_pd(ks5, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks9, di, qr8); qi8 = _mm512_fmadd_pd(ks9, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks10, di, qr9); qi9 = _mm512_fnmadd_pd(ks10, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks6, di, qr10); qi10 = _mm512_fnmadd_pd(ks6, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks2, di, qr11); qi11 = _mm512_fnmadd_pd(ks2, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr7 = _mm512_fnmadd_pd(ks11, di, qr7); qi7 = _mm512_fnmadd_pd(ks11, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks6, di, qr8); qi8 = _mm512_fnmadd_pd(ks6, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks1, di, qr9); qi9 = _mm512_fnmadd_pd(ks1, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks4, di, qr10); qi10 = _mm512_fmadd_pd(ks4, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks9, di, qr11); qi11 = _mm512_fmadd_pd(ks9, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr7 = _mm512_fnmadd_pd(ks4, di, qr7); qi7 = _mm512_fnmadd_pd(ks4, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks2, di, qr8); qi8 = _mm512_fmadd_pd(ks2, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks8, di, qr9); qi9 = _mm512_fmadd_pd(ks8, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks9, di, qr10); qi10 = _mm512_fnmadd_pd(ks9, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks3, di, qr11); qi11 = _mm512_fnmadd_pd(ks3, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+96), di = _mm512_load_pd(dscr+104);
    qr7 = _mm512_fmadd_pd(ks3, di, qr7); qi7 = _mm512_fmadd_pd(ks3, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks10, di, qr8); qi8 = _mm512_fmadd_pd(ks10, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks6, di, qr9); qi9 = _mm512_fnmadd_pd(ks6, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks1, di, qr10); qi10 = _mm512_fmadd_pd(ks1, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks8, di, qr11); qi11 = _mm512_fmadd_pd(ks8, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+112), di = _mm512_load_pd(dscr+120);
    qr7 = _mm512_fmadd_pd(ks10, di, qr7); qi7 = _mm512_fmadd_pd(ks10, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks5, di, qr8); qi8 = _mm512_fnmadd_pd(ks5, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks3, di, qr9); qi9 = _mm512_fmadd_pd(ks3, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks11, di, qr10); qi10 = _mm512_fmadd_pd(ks11, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks4, di, qr11); qi11 = _mm512_fnmadd_pd(ks4, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+128), di = _mm512_load_pd(dscr+136);
    qr7 = _mm512_fnmadd_pd(ks6, di, qr7); qi7 = _mm512_fnmadd_pd(ks6, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks3, di, qr8); qi8 = _mm512_fmadd_pd(ks3, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks11, di, qr9); qi9 = _mm512_fnmadd_pd(ks11, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks2, di, qr10); qi10 = _mm512_fnmadd_pd(ks2, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks7, di, qr11); qi11 = _mm512_fmadd_pd(ks7, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+144), di = _mm512_load_pd(dscr+152);
    qr7 = _mm512_fmadd_pd(ks1, di, qr7); qi7 = _mm512_fmadd_pd(ks1, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks11, di, qr8); qi8 = _mm512_fmadd_pd(ks11, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks2, di, qr9); qi9 = _mm512_fnmadd_pd(ks2, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks8, di, qr10); qi10 = _mm512_fmadd_pd(ks8, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks5, di, qr11); qi11 = _mm512_fnmadd_pd(ks5, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+160), di = _mm512_load_pd(dscr+168);
    qr7 = _mm512_fmadd_pd(ks8, di, qr7); qi7 = _mm512_fmadd_pd(ks8, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks4, di, qr8); qi8 = _mm512_fnmadd_pd(ks4, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks7, di, qr9); qi9 = _mm512_fmadd_pd(ks7, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks5, di, qr10); qi10 = _mm512_fnmadd_pd(ks5, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks6, di, qr11); qi11 = _mm512_fmadd_pd(ks6, dr, qi11);
    }
    _mm512_store_pd(qscr+96, qr7); _mm512_store_pd(qscr+104, qi7);
    _mm512_store_pd(qscr+112, qr8); _mm512_store_pd(qscr+120, qi8);
    _mm512_store_pd(qscr+128, qr9); _mm512_store_pd(qscr+136, qi9);
    _mm512_store_pd(qscr+144, qr10); _mm512_store_pd(qscr+152, qi10);
    _mm512_store_pd(qscr+160, qr11); _mm512_store_pd(qscr+168, qi11);
    }
    __asm__ volatile("" ::: "memory");
    {
    {
    __m512d Pr = _mm512_load_pd(pscr+16), Pi = _mm512_load_pd(pscr+24);
    __m512d Qr = _mm512_load_pd(qscr+0), Qi = _mm512_load_pd(qscr+8);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+16, xr); _mm512_store_pd(x+16+8, xi);
    _mm512_store_pd(x+352, yr); _mm512_store_pd(x+352+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+32), Pi = _mm512_load_pd(pscr+40);
    __m512d Qr = _mm512_load_pd(qscr+16), Qi = _mm512_load_pd(qscr+24);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+32, xr); _mm512_store_pd(x+32+8, xi);
    _mm512_store_pd(x+336, yr); _mm512_store_pd(x+336+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+48), Pi = _mm512_load_pd(pscr+56);
    __m512d Qr = _mm512_load_pd(qscr+32), Qi = _mm512_load_pd(qscr+40);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+48, xr); _mm512_store_pd(x+48+8, xi);
    _mm512_store_pd(x+320, yr); _mm512_store_pd(x+320+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+64), Pi = _mm512_load_pd(pscr+72);
    __m512d Qr = _mm512_load_pd(qscr+48), Qi = _mm512_load_pd(qscr+56);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+64, xr); _mm512_store_pd(x+64+8, xi);
    _mm512_store_pd(x+304, yr); _mm512_store_pd(x+304+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+80), Pi = _mm512_load_pd(pscr+88);
    __m512d Qr = _mm512_load_pd(qscr+64), Qi = _mm512_load_pd(qscr+72);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+80, xr); _mm512_store_pd(x+80+8, xi);
    _mm512_store_pd(x+288, yr); _mm512_store_pd(x+288+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+96), Pi = _mm512_load_pd(pscr+104);
    __m512d Qr = _mm512_load_pd(qscr+80), Qi = _mm512_load_pd(qscr+88);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+96, xr); _mm512_store_pd(x+96+8, xi);
    _mm512_store_pd(x+272, yr); _mm512_store_pd(x+272+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+112), Pi = _mm512_load_pd(pscr+120);
    __m512d Qr = _mm512_load_pd(qscr+96), Qi = _mm512_load_pd(qscr+104);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+112, xr); _mm512_store_pd(x+112+8, xi);
    _mm512_store_pd(x+256, yr); _mm512_store_pd(x+256+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+128), Pi = _mm512_load_pd(pscr+136);
    __m512d Qr = _mm512_load_pd(qscr+112), Qi = _mm512_load_pd(qscr+120);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+128, xr); _mm512_store_pd(x+128+8, xi);
    _mm512_store_pd(x+240, yr); _mm512_store_pd(x+240+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+144), Pi = _mm512_load_pd(pscr+152);
    __m512d Qr = _mm512_load_pd(qscr+128), Qi = _mm512_load_pd(qscr+136);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+144, xr); _mm512_store_pd(x+144+8, xi);
    _mm512_store_pd(x+224, yr); _mm512_store_pd(x+224+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+160), Pi = _mm512_load_pd(pscr+168);
    __m512d Qr = _mm512_load_pd(qscr+144), Qi = _mm512_load_pd(qscr+152);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+160, xr); _mm512_store_pd(x+160+8, xi);
    _mm512_store_pd(x+208, yr); _mm512_store_pd(x+208+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+176), Pi = _mm512_load_pd(pscr+184);
    __m512d Qr = _mm512_load_pd(qscr+160), Qi = _mm512_load_pd(qscr+168);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+176, xr); _mm512_store_pd(x+176+8, xi);
    _mm512_store_pd(x+192, yr); _mm512_store_pd(x+192+8, yi);
    }
    }
}
static void __attribute__((noinline)) dy_23(double* restrict x){
    double sscr[22*8] ALIGN64;
    double dscr[22*8] ALIGN64;
    double pscr[22*8+16] ALIGN64;
    double qscr[22*8] ALIGN64;
    {
    __m512d u0r = _mm512_load_pd(x), u0i = _mm512_load_pd(x+8);
    __m512d x0r = u0r, x0i = u0i;
    {
    __m512d ar = _mm512_load_pd(x+368), ai = _mm512_load_pd(x+368+8);
    __m512d br = _mm512_load_pd(x+8096), bi = _mm512_load_pd(x+8096+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+0, sr); _mm512_store_pd(sscr+8, si);
    _mm512_store_pd(dscr+0, dr); _mm512_store_pd(dscr+8, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+736), ai = _mm512_load_pd(x+736+8);
    __m512d br = _mm512_load_pd(x+7728), bi = _mm512_load_pd(x+7728+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+16, sr); _mm512_store_pd(sscr+24, si);
    _mm512_store_pd(dscr+16, dr); _mm512_store_pd(dscr+24, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+1104), ai = _mm512_load_pd(x+1104+8);
    __m512d br = _mm512_load_pd(x+7360), bi = _mm512_load_pd(x+7360+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+32, sr); _mm512_store_pd(sscr+40, si);
    _mm512_store_pd(dscr+32, dr); _mm512_store_pd(dscr+40, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+1472), ai = _mm512_load_pd(x+1472+8);
    __m512d br = _mm512_load_pd(x+6992), bi = _mm512_load_pd(x+6992+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+48, sr); _mm512_store_pd(sscr+56, si);
    _mm512_store_pd(dscr+48, dr); _mm512_store_pd(dscr+56, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+1840), ai = _mm512_load_pd(x+1840+8);
    __m512d br = _mm512_load_pd(x+6624), bi = _mm512_load_pd(x+6624+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+64, sr); _mm512_store_pd(sscr+72, si);
    _mm512_store_pd(dscr+64, dr); _mm512_store_pd(dscr+72, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+2208), ai = _mm512_load_pd(x+2208+8);
    __m512d br = _mm512_load_pd(x+6256), bi = _mm512_load_pd(x+6256+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+80, sr); _mm512_store_pd(sscr+88, si);
    _mm512_store_pd(dscr+80, dr); _mm512_store_pd(dscr+88, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+2576), ai = _mm512_load_pd(x+2576+8);
    __m512d br = _mm512_load_pd(x+5888), bi = _mm512_load_pd(x+5888+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+96, sr); _mm512_store_pd(sscr+104, si);
    _mm512_store_pd(dscr+96, dr); _mm512_store_pd(dscr+104, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+2944), ai = _mm512_load_pd(x+2944+8);
    __m512d br = _mm512_load_pd(x+5520), bi = _mm512_load_pd(x+5520+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+112, sr); _mm512_store_pd(sscr+120, si);
    _mm512_store_pd(dscr+112, dr); _mm512_store_pd(dscr+120, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+3312), ai = _mm512_load_pd(x+3312+8);
    __m512d br = _mm512_load_pd(x+5152), bi = _mm512_load_pd(x+5152+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+128, sr); _mm512_store_pd(sscr+136, si);
    _mm512_store_pd(dscr+128, dr); _mm512_store_pd(dscr+136, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+3680), ai = _mm512_load_pd(x+3680+8);
    __m512d br = _mm512_load_pd(x+4784), bi = _mm512_load_pd(x+4784+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+144, sr); _mm512_store_pd(sscr+152, si);
    _mm512_store_pd(dscr+144, dr); _mm512_store_pd(dscr+152, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+4048), ai = _mm512_load_pd(x+4048+8);
    __m512d br = _mm512_load_pd(x+4416), bi = _mm512_load_pd(x+4416+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+160, sr); _mm512_store_pd(sscr+168, si);
    _mm512_store_pd(dscr+160, dr); _mm512_store_pd(dscr+168, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    _mm512_store_pd(x, x0r); _mm512_store_pd(x+8, x0i);
    __m512d u0r_s = u0r, u0i_s = u0i;
    _mm512_store_pd(pscr, u0r_s); _mm512_store_pd(pscr+8, u0i_s);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d u0r = _mm512_load_pd(x+0*0), u0i;
    u0r = _mm512_load_pd(pscr); u0i = _mm512_load_pd(pscr+8);
    __m512d kc1; BCV(kc1, KC_23[0]);
    __m512d kc2; BCV(kc2, KC_23[1]);
    __m512d kc3; BCV(kc3, KC_23[2]);
    __m512d kc4; BCV(kc4, KC_23[3]);
    __m512d kc5; BCV(kc5, KC_23[4]);
    __m512d kc6; BCV(kc6, KC_23[5]);
    __m512d kc7; BCV(kc7, KC_23[6]);
    __m512d kc8; BCV(kc8, KC_23[7]);
    __m512d kc9; BCV(kc9, KC_23[8]);
    __m512d kc10; BCV(kc10, KC_23[9]);
    __m512d kc11; BCV(kc11, KC_23[10]);
    __m512d pr1 = u0r, pi1 = u0i;
    __m512d pr2 = u0r, pi2 = u0i;
    __m512d pr3 = u0r, pi3 = u0i;
    __m512d pr4 = u0r, pi4 = u0i;
    __m512d pr5 = u0r, pi5 = u0i;
    __m512d pr6 = u0r, pi6 = u0i;
    {
    __m512d sr = _mm512_load_pd(sscr+0), si = _mm512_load_pd(sscr+8);
    pr1 = _mm512_fmadd_pd(kc1, sr, pr1); pi1 = _mm512_fmadd_pd(kc1, si, pi1);
    pr2 = _mm512_fmadd_pd(kc2, sr, pr2); pi2 = _mm512_fmadd_pd(kc2, si, pi2);
    pr3 = _mm512_fmadd_pd(kc3, sr, pr3); pi3 = _mm512_fmadd_pd(kc3, si, pi3);
    pr4 = _mm512_fmadd_pd(kc4, sr, pr4); pi4 = _mm512_fmadd_pd(kc4, si, pi4);
    pr5 = _mm512_fmadd_pd(kc5, sr, pr5); pi5 = _mm512_fmadd_pd(kc5, si, pi5);
    pr6 = _mm512_fmadd_pd(kc6, sr, pr6); pi6 = _mm512_fmadd_pd(kc6, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+16), si = _mm512_load_pd(sscr+24);
    pr1 = _mm512_fmadd_pd(kc2, sr, pr1); pi1 = _mm512_fmadd_pd(kc2, si, pi1);
    pr2 = _mm512_fmadd_pd(kc4, sr, pr2); pi2 = _mm512_fmadd_pd(kc4, si, pi2);
    pr3 = _mm512_fmadd_pd(kc6, sr, pr3); pi3 = _mm512_fmadd_pd(kc6, si, pi3);
    pr4 = _mm512_fmadd_pd(kc8, sr, pr4); pi4 = _mm512_fmadd_pd(kc8, si, pi4);
    pr5 = _mm512_fmadd_pd(kc10, sr, pr5); pi5 = _mm512_fmadd_pd(kc10, si, pi5);
    pr6 = _mm512_fmadd_pd(kc11, sr, pr6); pi6 = _mm512_fmadd_pd(kc11, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+32), si = _mm512_load_pd(sscr+40);
    pr1 = _mm512_fmadd_pd(kc3, sr, pr1); pi1 = _mm512_fmadd_pd(kc3, si, pi1);
    pr2 = _mm512_fmadd_pd(kc6, sr, pr2); pi2 = _mm512_fmadd_pd(kc6, si, pi2);
    pr3 = _mm512_fmadd_pd(kc9, sr, pr3); pi3 = _mm512_fmadd_pd(kc9, si, pi3);
    pr4 = _mm512_fmadd_pd(kc11, sr, pr4); pi4 = _mm512_fmadd_pd(kc11, si, pi4);
    pr5 = _mm512_fmadd_pd(kc8, sr, pr5); pi5 = _mm512_fmadd_pd(kc8, si, pi5);
    pr6 = _mm512_fmadd_pd(kc5, sr, pr6); pi6 = _mm512_fmadd_pd(kc5, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+48), si = _mm512_load_pd(sscr+56);
    pr1 = _mm512_fmadd_pd(kc4, sr, pr1); pi1 = _mm512_fmadd_pd(kc4, si, pi1);
    pr2 = _mm512_fmadd_pd(kc8, sr, pr2); pi2 = _mm512_fmadd_pd(kc8, si, pi2);
    pr3 = _mm512_fmadd_pd(kc11, sr, pr3); pi3 = _mm512_fmadd_pd(kc11, si, pi3);
    pr4 = _mm512_fmadd_pd(kc7, sr, pr4); pi4 = _mm512_fmadd_pd(kc7, si, pi4);
    pr5 = _mm512_fmadd_pd(kc3, sr, pr5); pi5 = _mm512_fmadd_pd(kc3, si, pi5);
    pr6 = _mm512_fmadd_pd(kc1, sr, pr6); pi6 = _mm512_fmadd_pd(kc1, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+64), si = _mm512_load_pd(sscr+72);
    pr1 = _mm512_fmadd_pd(kc5, sr, pr1); pi1 = _mm512_fmadd_pd(kc5, si, pi1);
    pr2 = _mm512_fmadd_pd(kc10, sr, pr2); pi2 = _mm512_fmadd_pd(kc10, si, pi2);
    pr3 = _mm512_fmadd_pd(kc8, sr, pr3); pi3 = _mm512_fmadd_pd(kc8, si, pi3);
    pr4 = _mm512_fmadd_pd(kc3, sr, pr4); pi4 = _mm512_fmadd_pd(kc3, si, pi4);
    pr5 = _mm512_fmadd_pd(kc2, sr, pr5); pi5 = _mm512_fmadd_pd(kc2, si, pi5);
    pr6 = _mm512_fmadd_pd(kc7, sr, pr6); pi6 = _mm512_fmadd_pd(kc7, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+80), si = _mm512_load_pd(sscr+88);
    pr1 = _mm512_fmadd_pd(kc6, sr, pr1); pi1 = _mm512_fmadd_pd(kc6, si, pi1);
    pr2 = _mm512_fmadd_pd(kc11, sr, pr2); pi2 = _mm512_fmadd_pd(kc11, si, pi2);
    pr3 = _mm512_fmadd_pd(kc5, sr, pr3); pi3 = _mm512_fmadd_pd(kc5, si, pi3);
    pr4 = _mm512_fmadd_pd(kc1, sr, pr4); pi4 = _mm512_fmadd_pd(kc1, si, pi4);
    pr5 = _mm512_fmadd_pd(kc7, sr, pr5); pi5 = _mm512_fmadd_pd(kc7, si, pi5);
    pr6 = _mm512_fmadd_pd(kc10, sr, pr6); pi6 = _mm512_fmadd_pd(kc10, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+96), si = _mm512_load_pd(sscr+104);
    pr1 = _mm512_fmadd_pd(kc7, sr, pr1); pi1 = _mm512_fmadd_pd(kc7, si, pi1);
    pr2 = _mm512_fmadd_pd(kc9, sr, pr2); pi2 = _mm512_fmadd_pd(kc9, si, pi2);
    pr3 = _mm512_fmadd_pd(kc2, sr, pr3); pi3 = _mm512_fmadd_pd(kc2, si, pi3);
    pr4 = _mm512_fmadd_pd(kc5, sr, pr4); pi4 = _mm512_fmadd_pd(kc5, si, pi4);
    pr5 = _mm512_fmadd_pd(kc11, sr, pr5); pi5 = _mm512_fmadd_pd(kc11, si, pi5);
    pr6 = _mm512_fmadd_pd(kc4, sr, pr6); pi6 = _mm512_fmadd_pd(kc4, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+112), si = _mm512_load_pd(sscr+120);
    pr1 = _mm512_fmadd_pd(kc8, sr, pr1); pi1 = _mm512_fmadd_pd(kc8, si, pi1);
    pr2 = _mm512_fmadd_pd(kc7, sr, pr2); pi2 = _mm512_fmadd_pd(kc7, si, pi2);
    pr3 = _mm512_fmadd_pd(kc1, sr, pr3); pi3 = _mm512_fmadd_pd(kc1, si, pi3);
    pr4 = _mm512_fmadd_pd(kc9, sr, pr4); pi4 = _mm512_fmadd_pd(kc9, si, pi4);
    pr5 = _mm512_fmadd_pd(kc6, sr, pr5); pi5 = _mm512_fmadd_pd(kc6, si, pi5);
    pr6 = _mm512_fmadd_pd(kc2, sr, pr6); pi6 = _mm512_fmadd_pd(kc2, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+128), si = _mm512_load_pd(sscr+136);
    pr1 = _mm512_fmadd_pd(kc9, sr, pr1); pi1 = _mm512_fmadd_pd(kc9, si, pi1);
    pr2 = _mm512_fmadd_pd(kc5, sr, pr2); pi2 = _mm512_fmadd_pd(kc5, si, pi2);
    pr3 = _mm512_fmadd_pd(kc4, sr, pr3); pi3 = _mm512_fmadd_pd(kc4, si, pi3);
    pr4 = _mm512_fmadd_pd(kc10, sr, pr4); pi4 = _mm512_fmadd_pd(kc10, si, pi4);
    pr5 = _mm512_fmadd_pd(kc1, sr, pr5); pi5 = _mm512_fmadd_pd(kc1, si, pi5);
    pr6 = _mm512_fmadd_pd(kc8, sr, pr6); pi6 = _mm512_fmadd_pd(kc8, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+144), si = _mm512_load_pd(sscr+152);
    pr1 = _mm512_fmadd_pd(kc10, sr, pr1); pi1 = _mm512_fmadd_pd(kc10, si, pi1);
    pr2 = _mm512_fmadd_pd(kc3, sr, pr2); pi2 = _mm512_fmadd_pd(kc3, si, pi2);
    pr3 = _mm512_fmadd_pd(kc7, sr, pr3); pi3 = _mm512_fmadd_pd(kc7, si, pi3);
    pr4 = _mm512_fmadd_pd(kc6, sr, pr4); pi4 = _mm512_fmadd_pd(kc6, si, pi4);
    pr5 = _mm512_fmadd_pd(kc4, sr, pr5); pi5 = _mm512_fmadd_pd(kc4, si, pi5);
    pr6 = _mm512_fmadd_pd(kc9, sr, pr6); pi6 = _mm512_fmadd_pd(kc9, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+160), si = _mm512_load_pd(sscr+168);
    pr1 = _mm512_fmadd_pd(kc11, sr, pr1); pi1 = _mm512_fmadd_pd(kc11, si, pi1);
    pr2 = _mm512_fmadd_pd(kc1, sr, pr2); pi2 = _mm512_fmadd_pd(kc1, si, pi2);
    pr3 = _mm512_fmadd_pd(kc10, sr, pr3); pi3 = _mm512_fmadd_pd(kc10, si, pi3);
    pr4 = _mm512_fmadd_pd(kc2, sr, pr4); pi4 = _mm512_fmadd_pd(kc2, si, pi4);
    pr5 = _mm512_fmadd_pd(kc9, sr, pr5); pi5 = _mm512_fmadd_pd(kc9, si, pi5);
    pr6 = _mm512_fmadd_pd(kc3, sr, pr6); pi6 = _mm512_fmadd_pd(kc3, si, pi6);
    }
    _mm512_store_pd(pscr+16, pr1); _mm512_store_pd(pscr+24, pi1);
    _mm512_store_pd(pscr+32, pr2); _mm512_store_pd(pscr+40, pi2);
    _mm512_store_pd(pscr+48, pr3); _mm512_store_pd(pscr+56, pi3);
    _mm512_store_pd(pscr+64, pr4); _mm512_store_pd(pscr+72, pi4);
    _mm512_store_pd(pscr+80, pr5); _mm512_store_pd(pscr+88, pi5);
    _mm512_store_pd(pscr+96, pr6); _mm512_store_pd(pscr+104, pi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d u0r = _mm512_load_pd(x+0*0), u0i;
    u0r = _mm512_load_pd(pscr); u0i = _mm512_load_pd(pscr+8);
    __m512d kc1; BCV(kc1, KC_23[0]);
    __m512d kc2; BCV(kc2, KC_23[1]);
    __m512d kc3; BCV(kc3, KC_23[2]);
    __m512d kc4; BCV(kc4, KC_23[3]);
    __m512d kc5; BCV(kc5, KC_23[4]);
    __m512d kc6; BCV(kc6, KC_23[5]);
    __m512d kc7; BCV(kc7, KC_23[6]);
    __m512d kc8; BCV(kc8, KC_23[7]);
    __m512d kc9; BCV(kc9, KC_23[8]);
    __m512d kc10; BCV(kc10, KC_23[9]);
    __m512d kc11; BCV(kc11, KC_23[10]);
    __m512d pr7 = u0r, pi7 = u0i;
    __m512d pr8 = u0r, pi8 = u0i;
    __m512d pr9 = u0r, pi9 = u0i;
    __m512d pr10 = u0r, pi10 = u0i;
    __m512d pr11 = u0r, pi11 = u0i;
    {
    __m512d sr = _mm512_load_pd(sscr+0), si = _mm512_load_pd(sscr+8);
    pr7 = _mm512_fmadd_pd(kc7, sr, pr7); pi7 = _mm512_fmadd_pd(kc7, si, pi7);
    pr8 = _mm512_fmadd_pd(kc8, sr, pr8); pi8 = _mm512_fmadd_pd(kc8, si, pi8);
    pr9 = _mm512_fmadd_pd(kc9, sr, pr9); pi9 = _mm512_fmadd_pd(kc9, si, pi9);
    pr10 = _mm512_fmadd_pd(kc10, sr, pr10); pi10 = _mm512_fmadd_pd(kc10, si, pi10);
    pr11 = _mm512_fmadd_pd(kc11, sr, pr11); pi11 = _mm512_fmadd_pd(kc11, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+16), si = _mm512_load_pd(sscr+24);
    pr7 = _mm512_fmadd_pd(kc9, sr, pr7); pi7 = _mm512_fmadd_pd(kc9, si, pi7);
    pr8 = _mm512_fmadd_pd(kc7, sr, pr8); pi8 = _mm512_fmadd_pd(kc7, si, pi8);
    pr9 = _mm512_fmadd_pd(kc5, sr, pr9); pi9 = _mm512_fmadd_pd(kc5, si, pi9);
    pr10 = _mm512_fmadd_pd(kc3, sr, pr10); pi10 = _mm512_fmadd_pd(kc3, si, pi10);
    pr11 = _mm512_fmadd_pd(kc1, sr, pr11); pi11 = _mm512_fmadd_pd(kc1, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+32), si = _mm512_load_pd(sscr+40);
    pr7 = _mm512_fmadd_pd(kc2, sr, pr7); pi7 = _mm512_fmadd_pd(kc2, si, pi7);
    pr8 = _mm512_fmadd_pd(kc1, sr, pr8); pi8 = _mm512_fmadd_pd(kc1, si, pi8);
    pr9 = _mm512_fmadd_pd(kc4, sr, pr9); pi9 = _mm512_fmadd_pd(kc4, si, pi9);
    pr10 = _mm512_fmadd_pd(kc7, sr, pr10); pi10 = _mm512_fmadd_pd(kc7, si, pi10);
    pr11 = _mm512_fmadd_pd(kc10, sr, pr11); pi11 = _mm512_fmadd_pd(kc10, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+48), si = _mm512_load_pd(sscr+56);
    pr7 = _mm512_fmadd_pd(kc5, sr, pr7); pi7 = _mm512_fmadd_pd(kc5, si, pi7);
    pr8 = _mm512_fmadd_pd(kc9, sr, pr8); pi8 = _mm512_fmadd_pd(kc9, si, pi8);
    pr9 = _mm512_fmadd_pd(kc10, sr, pr9); pi9 = _mm512_fmadd_pd(kc10, si, pi9);
    pr10 = _mm512_fmadd_pd(kc6, sr, pr10); pi10 = _mm512_fmadd_pd(kc6, si, pi10);
    pr11 = _mm512_fmadd_pd(kc2, sr, pr11); pi11 = _mm512_fmadd_pd(kc2, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+64), si = _mm512_load_pd(sscr+72);
    pr7 = _mm512_fmadd_pd(kc11, sr, pr7); pi7 = _mm512_fmadd_pd(kc11, si, pi7);
    pr8 = _mm512_fmadd_pd(kc6, sr, pr8); pi8 = _mm512_fmadd_pd(kc6, si, pi8);
    pr9 = _mm512_fmadd_pd(kc1, sr, pr9); pi9 = _mm512_fmadd_pd(kc1, si, pi9);
    pr10 = _mm512_fmadd_pd(kc4, sr, pr10); pi10 = _mm512_fmadd_pd(kc4, si, pi10);
    pr11 = _mm512_fmadd_pd(kc9, sr, pr11); pi11 = _mm512_fmadd_pd(kc9, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+80), si = _mm512_load_pd(sscr+88);
    pr7 = _mm512_fmadd_pd(kc4, sr, pr7); pi7 = _mm512_fmadd_pd(kc4, si, pi7);
    pr8 = _mm512_fmadd_pd(kc2, sr, pr8); pi8 = _mm512_fmadd_pd(kc2, si, pi8);
    pr9 = _mm512_fmadd_pd(kc8, sr, pr9); pi9 = _mm512_fmadd_pd(kc8, si, pi9);
    pr10 = _mm512_fmadd_pd(kc9, sr, pr10); pi10 = _mm512_fmadd_pd(kc9, si, pi10);
    pr11 = _mm512_fmadd_pd(kc3, sr, pr11); pi11 = _mm512_fmadd_pd(kc3, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+96), si = _mm512_load_pd(sscr+104);
    pr7 = _mm512_fmadd_pd(kc3, sr, pr7); pi7 = _mm512_fmadd_pd(kc3, si, pi7);
    pr8 = _mm512_fmadd_pd(kc10, sr, pr8); pi8 = _mm512_fmadd_pd(kc10, si, pi8);
    pr9 = _mm512_fmadd_pd(kc6, sr, pr9); pi9 = _mm512_fmadd_pd(kc6, si, pi9);
    pr10 = _mm512_fmadd_pd(kc1, sr, pr10); pi10 = _mm512_fmadd_pd(kc1, si, pi10);
    pr11 = _mm512_fmadd_pd(kc8, sr, pr11); pi11 = _mm512_fmadd_pd(kc8, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+112), si = _mm512_load_pd(sscr+120);
    pr7 = _mm512_fmadd_pd(kc10, sr, pr7); pi7 = _mm512_fmadd_pd(kc10, si, pi7);
    pr8 = _mm512_fmadd_pd(kc5, sr, pr8); pi8 = _mm512_fmadd_pd(kc5, si, pi8);
    pr9 = _mm512_fmadd_pd(kc3, sr, pr9); pi9 = _mm512_fmadd_pd(kc3, si, pi9);
    pr10 = _mm512_fmadd_pd(kc11, sr, pr10); pi10 = _mm512_fmadd_pd(kc11, si, pi10);
    pr11 = _mm512_fmadd_pd(kc4, sr, pr11); pi11 = _mm512_fmadd_pd(kc4, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+128), si = _mm512_load_pd(sscr+136);
    pr7 = _mm512_fmadd_pd(kc6, sr, pr7); pi7 = _mm512_fmadd_pd(kc6, si, pi7);
    pr8 = _mm512_fmadd_pd(kc3, sr, pr8); pi8 = _mm512_fmadd_pd(kc3, si, pi8);
    pr9 = _mm512_fmadd_pd(kc11, sr, pr9); pi9 = _mm512_fmadd_pd(kc11, si, pi9);
    pr10 = _mm512_fmadd_pd(kc2, sr, pr10); pi10 = _mm512_fmadd_pd(kc2, si, pi10);
    pr11 = _mm512_fmadd_pd(kc7, sr, pr11); pi11 = _mm512_fmadd_pd(kc7, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+144), si = _mm512_load_pd(sscr+152);
    pr7 = _mm512_fmadd_pd(kc1, sr, pr7); pi7 = _mm512_fmadd_pd(kc1, si, pi7);
    pr8 = _mm512_fmadd_pd(kc11, sr, pr8); pi8 = _mm512_fmadd_pd(kc11, si, pi8);
    pr9 = _mm512_fmadd_pd(kc2, sr, pr9); pi9 = _mm512_fmadd_pd(kc2, si, pi9);
    pr10 = _mm512_fmadd_pd(kc8, sr, pr10); pi10 = _mm512_fmadd_pd(kc8, si, pi10);
    pr11 = _mm512_fmadd_pd(kc5, sr, pr11); pi11 = _mm512_fmadd_pd(kc5, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+160), si = _mm512_load_pd(sscr+168);
    pr7 = _mm512_fmadd_pd(kc8, sr, pr7); pi7 = _mm512_fmadd_pd(kc8, si, pi7);
    pr8 = _mm512_fmadd_pd(kc4, sr, pr8); pi8 = _mm512_fmadd_pd(kc4, si, pi8);
    pr9 = _mm512_fmadd_pd(kc7, sr, pr9); pi9 = _mm512_fmadd_pd(kc7, si, pi9);
    pr10 = _mm512_fmadd_pd(kc5, sr, pr10); pi10 = _mm512_fmadd_pd(kc5, si, pi10);
    pr11 = _mm512_fmadd_pd(kc6, sr, pr11); pi11 = _mm512_fmadd_pd(kc6, si, pi11);
    }
    _mm512_store_pd(pscr+112, pr7); _mm512_store_pd(pscr+120, pi7);
    _mm512_store_pd(pscr+128, pr8); _mm512_store_pd(pscr+136, pi8);
    _mm512_store_pd(pscr+144, pr9); _mm512_store_pd(pscr+152, pi9);
    _mm512_store_pd(pscr+160, pr10); _mm512_store_pd(pscr+168, pi10);
    _mm512_store_pd(pscr+176, pr11); _mm512_store_pd(pscr+184, pi11);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_23[0]);
    __m512d ks2; BCV(ks2, KS_23[1]);
    __m512d ks3; BCV(ks3, KS_23[2]);
    __m512d ks4; BCV(ks4, KS_23[3]);
    __m512d ks5; BCV(ks5, KS_23[4]);
    __m512d ks6; BCV(ks6, KS_23[5]);
    __m512d ks7; BCV(ks7, KS_23[6]);
    __m512d ks8; BCV(ks8, KS_23[7]);
    __m512d ks9; BCV(ks9, KS_23[8]);
    __m512d ks10; BCV(ks10, KS_23[9]);
    __m512d ks11; BCV(ks11, KS_23[10]);
    __m512d qr1, qi1;
    __m512d qr2, qi2;
    __m512d qr3, qi3;
    __m512d qr4, qi4;
    __m512d qr5, qi5;
    __m512d qr6, qi6;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr1 = _mm512_mul_pd(ks1, di); qi1 = _mm512_mul_pd(ks1, dr);
    qr2 = _mm512_mul_pd(ks2, di); qi2 = _mm512_mul_pd(ks2, dr);
    qr3 = _mm512_mul_pd(ks3, di); qi3 = _mm512_mul_pd(ks3, dr);
    qr4 = _mm512_mul_pd(ks4, di); qi4 = _mm512_mul_pd(ks4, dr);
    qr5 = _mm512_mul_pd(ks5, di); qi5 = _mm512_mul_pd(ks5, dr);
    qr6 = _mm512_mul_pd(ks6, di); qi6 = _mm512_mul_pd(ks6, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr1 = _mm512_fmadd_pd(ks2, di, qr1); qi1 = _mm512_fmadd_pd(ks2, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks4, di, qr2); qi2 = _mm512_fmadd_pd(ks4, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks6, di, qr3); qi3 = _mm512_fmadd_pd(ks6, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks8, di, qr4); qi4 = _mm512_fmadd_pd(ks8, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks10, di, qr5); qi5 = _mm512_fmadd_pd(ks10, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks11, di, qr6); qi6 = _mm512_fnmadd_pd(ks11, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr1 = _mm512_fmadd_pd(ks3, di, qr1); qi1 = _mm512_fmadd_pd(ks3, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks6, di, qr2); qi2 = _mm512_fmadd_pd(ks6, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks9, di, qr3); qi3 = _mm512_fmadd_pd(ks9, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks11, di, qr4); qi4 = _mm512_fnmadd_pd(ks11, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks8, di, qr5); qi5 = _mm512_fnmadd_pd(ks8, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks5, di, qr6); qi6 = _mm512_fnmadd_pd(ks5, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr1 = _mm512_fmadd_pd(ks4, di, qr1); qi1 = _mm512_fmadd_pd(ks4, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks8, di, qr2); qi2 = _mm512_fmadd_pd(ks8, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks11, di, qr3); qi3 = _mm512_fnmadd_pd(ks11, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks7, di, qr4); qi4 = _mm512_fnmadd_pd(ks7, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks3, di, qr5); qi5 = _mm512_fnmadd_pd(ks3, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks1, di, qr6); qi6 = _mm512_fmadd_pd(ks1, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr1 = _mm512_fmadd_pd(ks5, di, qr1); qi1 = _mm512_fmadd_pd(ks5, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks10, di, qr2); qi2 = _mm512_fmadd_pd(ks10, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks8, di, qr3); qi3 = _mm512_fnmadd_pd(ks8, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks3, di, qr4); qi4 = _mm512_fnmadd_pd(ks3, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks2, di, qr5); qi5 = _mm512_fmadd_pd(ks2, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks7, di, qr6); qi6 = _mm512_fmadd_pd(ks7, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr1 = _mm512_fmadd_pd(ks6, di, qr1); qi1 = _mm512_fmadd_pd(ks6, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks11, di, qr2); qi2 = _mm512_fnmadd_pd(ks11, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks5, di, qr3); qi3 = _mm512_fnmadd_pd(ks5, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks1, di, qr4); qi4 = _mm512_fmadd_pd(ks1, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks7, di, qr5); qi5 = _mm512_fmadd_pd(ks7, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks10, di, qr6); qi6 = _mm512_fnmadd_pd(ks10, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+96), di = _mm512_load_pd(dscr+104);
    qr1 = _mm512_fmadd_pd(ks7, di, qr1); qi1 = _mm512_fmadd_pd(ks7, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks9, di, qr2); qi2 = _mm512_fnmadd_pd(ks9, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks2, di, qr3); qi3 = _mm512_fnmadd_pd(ks2, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks5, di, qr4); qi4 = _mm512_fmadd_pd(ks5, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks11, di, qr5); qi5 = _mm512_fnmadd_pd(ks11, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks4, di, qr6); qi6 = _mm512_fnmadd_pd(ks4, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+112), di = _mm512_load_pd(dscr+120);
    qr1 = _mm512_fmadd_pd(ks8, di, qr1); qi1 = _mm512_fmadd_pd(ks8, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks7, di, qr2); qi2 = _mm512_fnmadd_pd(ks7, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks1, di, qr3); qi3 = _mm512_fmadd_pd(ks1, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks9, di, qr4); qi4 = _mm512_fmadd_pd(ks9, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks6, di, qr5); qi5 = _mm512_fnmadd_pd(ks6, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks2, di, qr6); qi6 = _mm512_fmadd_pd(ks2, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+128), di = _mm512_load_pd(dscr+136);
    qr1 = _mm512_fmadd_pd(ks9, di, qr1); qi1 = _mm512_fmadd_pd(ks9, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks5, di, qr2); qi2 = _mm512_fnmadd_pd(ks5, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks4, di, qr3); qi3 = _mm512_fmadd_pd(ks4, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks10, di, qr4); qi4 = _mm512_fnmadd_pd(ks10, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks1, di, qr5); qi5 = _mm512_fnmadd_pd(ks1, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks8, di, qr6); qi6 = _mm512_fmadd_pd(ks8, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+144), di = _mm512_load_pd(dscr+152);
    qr1 = _mm512_fmadd_pd(ks10, di, qr1); qi1 = _mm512_fmadd_pd(ks10, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks3, di, qr2); qi2 = _mm512_fnmadd_pd(ks3, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks7, di, qr3); qi3 = _mm512_fmadd_pd(ks7, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks6, di, qr4); qi4 = _mm512_fnmadd_pd(ks6, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks4, di, qr5); qi5 = _mm512_fmadd_pd(ks4, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks9, di, qr6); qi6 = _mm512_fnmadd_pd(ks9, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+160), di = _mm512_load_pd(dscr+168);
    qr1 = _mm512_fmadd_pd(ks11, di, qr1); qi1 = _mm512_fmadd_pd(ks11, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks1, di, qr2); qi2 = _mm512_fnmadd_pd(ks1, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks10, di, qr3); qi3 = _mm512_fmadd_pd(ks10, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks2, di, qr4); qi4 = _mm512_fnmadd_pd(ks2, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks9, di, qr5); qi5 = _mm512_fmadd_pd(ks9, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks3, di, qr6); qi6 = _mm512_fnmadd_pd(ks3, dr, qi6);
    }
    _mm512_store_pd(qscr+0, qr1); _mm512_store_pd(qscr+8, qi1);
    _mm512_store_pd(qscr+16, qr2); _mm512_store_pd(qscr+24, qi2);
    _mm512_store_pd(qscr+32, qr3); _mm512_store_pd(qscr+40, qi3);
    _mm512_store_pd(qscr+48, qr4); _mm512_store_pd(qscr+56, qi4);
    _mm512_store_pd(qscr+64, qr5); _mm512_store_pd(qscr+72, qi5);
    _mm512_store_pd(qscr+80, qr6); _mm512_store_pd(qscr+88, qi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_23[0]);
    __m512d ks2; BCV(ks2, KS_23[1]);
    __m512d ks3; BCV(ks3, KS_23[2]);
    __m512d ks4; BCV(ks4, KS_23[3]);
    __m512d ks5; BCV(ks5, KS_23[4]);
    __m512d ks6; BCV(ks6, KS_23[5]);
    __m512d ks7; BCV(ks7, KS_23[6]);
    __m512d ks8; BCV(ks8, KS_23[7]);
    __m512d ks9; BCV(ks9, KS_23[8]);
    __m512d ks10; BCV(ks10, KS_23[9]);
    __m512d ks11; BCV(ks11, KS_23[10]);
    __m512d qr7, qi7;
    __m512d qr8, qi8;
    __m512d qr9, qi9;
    __m512d qr10, qi10;
    __m512d qr11, qi11;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr7 = _mm512_mul_pd(ks7, di); qi7 = _mm512_mul_pd(ks7, dr);
    qr8 = _mm512_mul_pd(ks8, di); qi8 = _mm512_mul_pd(ks8, dr);
    qr9 = _mm512_mul_pd(ks9, di); qi9 = _mm512_mul_pd(ks9, dr);
    qr10 = _mm512_mul_pd(ks10, di); qi10 = _mm512_mul_pd(ks10, dr);
    qr11 = _mm512_mul_pd(ks11, di); qi11 = _mm512_mul_pd(ks11, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr7 = _mm512_fnmadd_pd(ks9, di, qr7); qi7 = _mm512_fnmadd_pd(ks9, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks7, di, qr8); qi8 = _mm512_fnmadd_pd(ks7, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks5, di, qr9); qi9 = _mm512_fnmadd_pd(ks5, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks3, di, qr10); qi10 = _mm512_fnmadd_pd(ks3, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks1, di, qr11); qi11 = _mm512_fnmadd_pd(ks1, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr7 = _mm512_fnmadd_pd(ks2, di, qr7); qi7 = _mm512_fnmadd_pd(ks2, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks1, di, qr8); qi8 = _mm512_fmadd_pd(ks1, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks4, di, qr9); qi9 = _mm512_fmadd_pd(ks4, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks7, di, qr10); qi10 = _mm512_fmadd_pd(ks7, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks10, di, qr11); qi11 = _mm512_fmadd_pd(ks10, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr7 = _mm512_fmadd_pd(ks5, di, qr7); qi7 = _mm512_fmadd_pd(ks5, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks9, di, qr8); qi8 = _mm512_fmadd_pd(ks9, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks10, di, qr9); qi9 = _mm512_fnmadd_pd(ks10, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks6, di, qr10); qi10 = _mm512_fnmadd_pd(ks6, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks2, di, qr11); qi11 = _mm512_fnmadd_pd(ks2, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr7 = _mm512_fnmadd_pd(ks11, di, qr7); qi7 = _mm512_fnmadd_pd(ks11, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks6, di, qr8); qi8 = _mm512_fnmadd_pd(ks6, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks1, di, qr9); qi9 = _mm512_fnmadd_pd(ks1, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks4, di, qr10); qi10 = _mm512_fmadd_pd(ks4, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks9, di, qr11); qi11 = _mm512_fmadd_pd(ks9, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr7 = _mm512_fnmadd_pd(ks4, di, qr7); qi7 = _mm512_fnmadd_pd(ks4, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks2, di, qr8); qi8 = _mm512_fmadd_pd(ks2, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks8, di, qr9); qi9 = _mm512_fmadd_pd(ks8, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks9, di, qr10); qi10 = _mm512_fnmadd_pd(ks9, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks3, di, qr11); qi11 = _mm512_fnmadd_pd(ks3, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+96), di = _mm512_load_pd(dscr+104);
    qr7 = _mm512_fmadd_pd(ks3, di, qr7); qi7 = _mm512_fmadd_pd(ks3, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks10, di, qr8); qi8 = _mm512_fmadd_pd(ks10, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks6, di, qr9); qi9 = _mm512_fnmadd_pd(ks6, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks1, di, qr10); qi10 = _mm512_fmadd_pd(ks1, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks8, di, qr11); qi11 = _mm512_fmadd_pd(ks8, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+112), di = _mm512_load_pd(dscr+120);
    qr7 = _mm512_fmadd_pd(ks10, di, qr7); qi7 = _mm512_fmadd_pd(ks10, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks5, di, qr8); qi8 = _mm512_fnmadd_pd(ks5, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks3, di, qr9); qi9 = _mm512_fmadd_pd(ks3, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks11, di, qr10); qi10 = _mm512_fmadd_pd(ks11, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks4, di, qr11); qi11 = _mm512_fnmadd_pd(ks4, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+128), di = _mm512_load_pd(dscr+136);
    qr7 = _mm512_fnmadd_pd(ks6, di, qr7); qi7 = _mm512_fnmadd_pd(ks6, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks3, di, qr8); qi8 = _mm512_fmadd_pd(ks3, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks11, di, qr9); qi9 = _mm512_fnmadd_pd(ks11, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks2, di, qr10); qi10 = _mm512_fnmadd_pd(ks2, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks7, di, qr11); qi11 = _mm512_fmadd_pd(ks7, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+144), di = _mm512_load_pd(dscr+152);
    qr7 = _mm512_fmadd_pd(ks1, di, qr7); qi7 = _mm512_fmadd_pd(ks1, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks11, di, qr8); qi8 = _mm512_fmadd_pd(ks11, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks2, di, qr9); qi9 = _mm512_fnmadd_pd(ks2, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks8, di, qr10); qi10 = _mm512_fmadd_pd(ks8, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks5, di, qr11); qi11 = _mm512_fnmadd_pd(ks5, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+160), di = _mm512_load_pd(dscr+168);
    qr7 = _mm512_fmadd_pd(ks8, di, qr7); qi7 = _mm512_fmadd_pd(ks8, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks4, di, qr8); qi8 = _mm512_fnmadd_pd(ks4, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks7, di, qr9); qi9 = _mm512_fmadd_pd(ks7, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks5, di, qr10); qi10 = _mm512_fnmadd_pd(ks5, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks6, di, qr11); qi11 = _mm512_fmadd_pd(ks6, dr, qi11);
    }
    _mm512_store_pd(qscr+96, qr7); _mm512_store_pd(qscr+104, qi7);
    _mm512_store_pd(qscr+112, qr8); _mm512_store_pd(qscr+120, qi8);
    _mm512_store_pd(qscr+128, qr9); _mm512_store_pd(qscr+136, qi9);
    _mm512_store_pd(qscr+144, qr10); _mm512_store_pd(qscr+152, qi10);
    _mm512_store_pd(qscr+160, qr11); _mm512_store_pd(qscr+168, qi11);
    }
    __asm__ volatile("" ::: "memory");
    {
    {
    __m512d Pr = _mm512_load_pd(pscr+16), Pi = _mm512_load_pd(pscr+24);
    __m512d Qr = _mm512_load_pd(qscr+0), Qi = _mm512_load_pd(qscr+8);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+368, xr); _mm512_store_pd(x+368+8, xi);
    _mm512_store_pd(x+8096, yr); _mm512_store_pd(x+8096+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+32), Pi = _mm512_load_pd(pscr+40);
    __m512d Qr = _mm512_load_pd(qscr+16), Qi = _mm512_load_pd(qscr+24);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+736, xr); _mm512_store_pd(x+736+8, xi);
    _mm512_store_pd(x+7728, yr); _mm512_store_pd(x+7728+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+48), Pi = _mm512_load_pd(pscr+56);
    __m512d Qr = _mm512_load_pd(qscr+32), Qi = _mm512_load_pd(qscr+40);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+1104, xr); _mm512_store_pd(x+1104+8, xi);
    _mm512_store_pd(x+7360, yr); _mm512_store_pd(x+7360+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+64), Pi = _mm512_load_pd(pscr+72);
    __m512d Qr = _mm512_load_pd(qscr+48), Qi = _mm512_load_pd(qscr+56);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+1472, xr); _mm512_store_pd(x+1472+8, xi);
    _mm512_store_pd(x+6992, yr); _mm512_store_pd(x+6992+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+80), Pi = _mm512_load_pd(pscr+88);
    __m512d Qr = _mm512_load_pd(qscr+64), Qi = _mm512_load_pd(qscr+72);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+1840, xr); _mm512_store_pd(x+1840+8, xi);
    _mm512_store_pd(x+6624, yr); _mm512_store_pd(x+6624+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+96), Pi = _mm512_load_pd(pscr+104);
    __m512d Qr = _mm512_load_pd(qscr+80), Qi = _mm512_load_pd(qscr+88);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+2208, xr); _mm512_store_pd(x+2208+8, xi);
    _mm512_store_pd(x+6256, yr); _mm512_store_pd(x+6256+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+112), Pi = _mm512_load_pd(pscr+120);
    __m512d Qr = _mm512_load_pd(qscr+96), Qi = _mm512_load_pd(qscr+104);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+2576, xr); _mm512_store_pd(x+2576+8, xi);
    _mm512_store_pd(x+5888, yr); _mm512_store_pd(x+5888+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+128), Pi = _mm512_load_pd(pscr+136);
    __m512d Qr = _mm512_load_pd(qscr+112), Qi = _mm512_load_pd(qscr+120);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+2944, xr); _mm512_store_pd(x+2944+8, xi);
    _mm512_store_pd(x+5520, yr); _mm512_store_pd(x+5520+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+144), Pi = _mm512_load_pd(pscr+152);
    __m512d Qr = _mm512_load_pd(qscr+128), Qi = _mm512_load_pd(qscr+136);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+3312, xr); _mm512_store_pd(x+3312+8, xi);
    _mm512_store_pd(x+5152, yr); _mm512_store_pd(x+5152+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+160), Pi = _mm512_load_pd(pscr+168);
    __m512d Qr = _mm512_load_pd(qscr+144), Qi = _mm512_load_pd(qscr+152);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+3680, xr); _mm512_store_pd(x+3680+8, xi);
    _mm512_store_pd(x+4784, yr); _mm512_store_pd(x+4784+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+176), Pi = _mm512_load_pd(pscr+184);
    __m512d Qr = _mm512_load_pd(qscr+160), Qi = _mm512_load_pd(qscr+168);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+4048, xr); _mm512_store_pd(x+4048+8, xi);
    _mm512_store_pd(x+4416, yr); _mm512_store_pd(x+4416+8, yi);
    }
    }
}
static void __attribute__((noinline)) dx_23(double* restrict x, const double* restrict cb){
    double sscr[22*8] ALIGN64;
    double dscr[22*8] ALIGN64;
    double pscr[22*8+16] ALIGN64;
    double qscr[22*8] ALIGN64;
    {
    __m512d u0r = _mm512_load_pd(x), u0i = _mm512_load_pd(x+8);
    __m512d x0r = u0r, x0i = u0i;
    {
    __m512d ar = _mm512_load_pd(x+8464), ai = _mm512_load_pd(x+8464+8);
    __m512d br = _mm512_load_pd(x+186208), bi = _mm512_load_pd(x+186208+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+0, sr); _mm512_store_pd(sscr+8, si);
    _mm512_store_pd(dscr+0, dr); _mm512_store_pd(dscr+8, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+16928), ai = _mm512_load_pd(x+16928+8);
    __m512d br = _mm512_load_pd(x+177744), bi = _mm512_load_pd(x+177744+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+16, sr); _mm512_store_pd(sscr+24, si);
    _mm512_store_pd(dscr+16, dr); _mm512_store_pd(dscr+24, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+25392), ai = _mm512_load_pd(x+25392+8);
    __m512d br = _mm512_load_pd(x+169280), bi = _mm512_load_pd(x+169280+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+32, sr); _mm512_store_pd(sscr+40, si);
    _mm512_store_pd(dscr+32, dr); _mm512_store_pd(dscr+40, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+33856), ai = _mm512_load_pd(x+33856+8);
    __m512d br = _mm512_load_pd(x+160816), bi = _mm512_load_pd(x+160816+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+48, sr); _mm512_store_pd(sscr+56, si);
    _mm512_store_pd(dscr+48, dr); _mm512_store_pd(dscr+56, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+42320), ai = _mm512_load_pd(x+42320+8);
    __m512d br = _mm512_load_pd(x+152352), bi = _mm512_load_pd(x+152352+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+64, sr); _mm512_store_pd(sscr+72, si);
    _mm512_store_pd(dscr+64, dr); _mm512_store_pd(dscr+72, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+50784), ai = _mm512_load_pd(x+50784+8);
    __m512d br = _mm512_load_pd(x+143888), bi = _mm512_load_pd(x+143888+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+80, sr); _mm512_store_pd(sscr+88, si);
    _mm512_store_pd(dscr+80, dr); _mm512_store_pd(dscr+88, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+59248), ai = _mm512_load_pd(x+59248+8);
    __m512d br = _mm512_load_pd(x+135424), bi = _mm512_load_pd(x+135424+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+96, sr); _mm512_store_pd(sscr+104, si);
    _mm512_store_pd(dscr+96, dr); _mm512_store_pd(dscr+104, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+67712), ai = _mm512_load_pd(x+67712+8);
    __m512d br = _mm512_load_pd(x+126960), bi = _mm512_load_pd(x+126960+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+112, sr); _mm512_store_pd(sscr+120, si);
    _mm512_store_pd(dscr+112, dr); _mm512_store_pd(dscr+120, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+76176), ai = _mm512_load_pd(x+76176+8);
    __m512d br = _mm512_load_pd(x+118496), bi = _mm512_load_pd(x+118496+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+128, sr); _mm512_store_pd(sscr+136, si);
    _mm512_store_pd(dscr+128, dr); _mm512_store_pd(dscr+136, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+84640), ai = _mm512_load_pd(x+84640+8);
    __m512d br = _mm512_load_pd(x+110032), bi = _mm512_load_pd(x+110032+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+144, sr); _mm512_store_pd(sscr+152, si);
    _mm512_store_pd(dscr+144, dr); _mm512_store_pd(dscr+152, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+93104), ai = _mm512_load_pd(x+93104+8);
    __m512d br = _mm512_load_pd(x+101568), bi = _mm512_load_pd(x+101568+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+160, sr); _mm512_store_pd(sscr+168, si);
    _mm512_store_pd(dscr+160, dr); _mm512_store_pd(dscr+168, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    MAPST(x0r, x0i, x, 0, cb, 0);
    __m512d u0r_s = u0r, u0i_s = u0i;
    _mm512_store_pd(pscr, u0r_s); _mm512_store_pd(pscr+8, u0i_s);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d u0r = _mm512_load_pd(x+0*0), u0i;
    u0r = _mm512_load_pd(pscr); u0i = _mm512_load_pd(pscr+8);
    __m512d kc1; BCV(kc1, KC_23[0]);
    __m512d kc2; BCV(kc2, KC_23[1]);
    __m512d kc3; BCV(kc3, KC_23[2]);
    __m512d kc4; BCV(kc4, KC_23[3]);
    __m512d kc5; BCV(kc5, KC_23[4]);
    __m512d kc6; BCV(kc6, KC_23[5]);
    __m512d kc7; BCV(kc7, KC_23[6]);
    __m512d kc8; BCV(kc8, KC_23[7]);
    __m512d kc9; BCV(kc9, KC_23[8]);
    __m512d kc10; BCV(kc10, KC_23[9]);
    __m512d kc11; BCV(kc11, KC_23[10]);
    __m512d pr1 = u0r, pi1 = u0i;
    __m512d pr2 = u0r, pi2 = u0i;
    __m512d pr3 = u0r, pi3 = u0i;
    __m512d pr4 = u0r, pi4 = u0i;
    __m512d pr5 = u0r, pi5 = u0i;
    __m512d pr6 = u0r, pi6 = u0i;
    {
    __m512d sr = _mm512_load_pd(sscr+0), si = _mm512_load_pd(sscr+8);
    pr1 = _mm512_fmadd_pd(kc1, sr, pr1); pi1 = _mm512_fmadd_pd(kc1, si, pi1);
    pr2 = _mm512_fmadd_pd(kc2, sr, pr2); pi2 = _mm512_fmadd_pd(kc2, si, pi2);
    pr3 = _mm512_fmadd_pd(kc3, sr, pr3); pi3 = _mm512_fmadd_pd(kc3, si, pi3);
    pr4 = _mm512_fmadd_pd(kc4, sr, pr4); pi4 = _mm512_fmadd_pd(kc4, si, pi4);
    pr5 = _mm512_fmadd_pd(kc5, sr, pr5); pi5 = _mm512_fmadd_pd(kc5, si, pi5);
    pr6 = _mm512_fmadd_pd(kc6, sr, pr6); pi6 = _mm512_fmadd_pd(kc6, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+16), si = _mm512_load_pd(sscr+24);
    pr1 = _mm512_fmadd_pd(kc2, sr, pr1); pi1 = _mm512_fmadd_pd(kc2, si, pi1);
    pr2 = _mm512_fmadd_pd(kc4, sr, pr2); pi2 = _mm512_fmadd_pd(kc4, si, pi2);
    pr3 = _mm512_fmadd_pd(kc6, sr, pr3); pi3 = _mm512_fmadd_pd(kc6, si, pi3);
    pr4 = _mm512_fmadd_pd(kc8, sr, pr4); pi4 = _mm512_fmadd_pd(kc8, si, pi4);
    pr5 = _mm512_fmadd_pd(kc10, sr, pr5); pi5 = _mm512_fmadd_pd(kc10, si, pi5);
    pr6 = _mm512_fmadd_pd(kc11, sr, pr6); pi6 = _mm512_fmadd_pd(kc11, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+32), si = _mm512_load_pd(sscr+40);
    pr1 = _mm512_fmadd_pd(kc3, sr, pr1); pi1 = _mm512_fmadd_pd(kc3, si, pi1);
    pr2 = _mm512_fmadd_pd(kc6, sr, pr2); pi2 = _mm512_fmadd_pd(kc6, si, pi2);
    pr3 = _mm512_fmadd_pd(kc9, sr, pr3); pi3 = _mm512_fmadd_pd(kc9, si, pi3);
    pr4 = _mm512_fmadd_pd(kc11, sr, pr4); pi4 = _mm512_fmadd_pd(kc11, si, pi4);
    pr5 = _mm512_fmadd_pd(kc8, sr, pr5); pi5 = _mm512_fmadd_pd(kc8, si, pi5);
    pr6 = _mm512_fmadd_pd(kc5, sr, pr6); pi6 = _mm512_fmadd_pd(kc5, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+48), si = _mm512_load_pd(sscr+56);
    pr1 = _mm512_fmadd_pd(kc4, sr, pr1); pi1 = _mm512_fmadd_pd(kc4, si, pi1);
    pr2 = _mm512_fmadd_pd(kc8, sr, pr2); pi2 = _mm512_fmadd_pd(kc8, si, pi2);
    pr3 = _mm512_fmadd_pd(kc11, sr, pr3); pi3 = _mm512_fmadd_pd(kc11, si, pi3);
    pr4 = _mm512_fmadd_pd(kc7, sr, pr4); pi4 = _mm512_fmadd_pd(kc7, si, pi4);
    pr5 = _mm512_fmadd_pd(kc3, sr, pr5); pi5 = _mm512_fmadd_pd(kc3, si, pi5);
    pr6 = _mm512_fmadd_pd(kc1, sr, pr6); pi6 = _mm512_fmadd_pd(kc1, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+64), si = _mm512_load_pd(sscr+72);
    pr1 = _mm512_fmadd_pd(kc5, sr, pr1); pi1 = _mm512_fmadd_pd(kc5, si, pi1);
    pr2 = _mm512_fmadd_pd(kc10, sr, pr2); pi2 = _mm512_fmadd_pd(kc10, si, pi2);
    pr3 = _mm512_fmadd_pd(kc8, sr, pr3); pi3 = _mm512_fmadd_pd(kc8, si, pi3);
    pr4 = _mm512_fmadd_pd(kc3, sr, pr4); pi4 = _mm512_fmadd_pd(kc3, si, pi4);
    pr5 = _mm512_fmadd_pd(kc2, sr, pr5); pi5 = _mm512_fmadd_pd(kc2, si, pi5);
    pr6 = _mm512_fmadd_pd(kc7, sr, pr6); pi6 = _mm512_fmadd_pd(kc7, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+80), si = _mm512_load_pd(sscr+88);
    pr1 = _mm512_fmadd_pd(kc6, sr, pr1); pi1 = _mm512_fmadd_pd(kc6, si, pi1);
    pr2 = _mm512_fmadd_pd(kc11, sr, pr2); pi2 = _mm512_fmadd_pd(kc11, si, pi2);
    pr3 = _mm512_fmadd_pd(kc5, sr, pr3); pi3 = _mm512_fmadd_pd(kc5, si, pi3);
    pr4 = _mm512_fmadd_pd(kc1, sr, pr4); pi4 = _mm512_fmadd_pd(kc1, si, pi4);
    pr5 = _mm512_fmadd_pd(kc7, sr, pr5); pi5 = _mm512_fmadd_pd(kc7, si, pi5);
    pr6 = _mm512_fmadd_pd(kc10, sr, pr6); pi6 = _mm512_fmadd_pd(kc10, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+96), si = _mm512_load_pd(sscr+104);
    pr1 = _mm512_fmadd_pd(kc7, sr, pr1); pi1 = _mm512_fmadd_pd(kc7, si, pi1);
    pr2 = _mm512_fmadd_pd(kc9, sr, pr2); pi2 = _mm512_fmadd_pd(kc9, si, pi2);
    pr3 = _mm512_fmadd_pd(kc2, sr, pr3); pi3 = _mm512_fmadd_pd(kc2, si, pi3);
    pr4 = _mm512_fmadd_pd(kc5, sr, pr4); pi4 = _mm512_fmadd_pd(kc5, si, pi4);
    pr5 = _mm512_fmadd_pd(kc11, sr, pr5); pi5 = _mm512_fmadd_pd(kc11, si, pi5);
    pr6 = _mm512_fmadd_pd(kc4, sr, pr6); pi6 = _mm512_fmadd_pd(kc4, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+112), si = _mm512_load_pd(sscr+120);
    pr1 = _mm512_fmadd_pd(kc8, sr, pr1); pi1 = _mm512_fmadd_pd(kc8, si, pi1);
    pr2 = _mm512_fmadd_pd(kc7, sr, pr2); pi2 = _mm512_fmadd_pd(kc7, si, pi2);
    pr3 = _mm512_fmadd_pd(kc1, sr, pr3); pi3 = _mm512_fmadd_pd(kc1, si, pi3);
    pr4 = _mm512_fmadd_pd(kc9, sr, pr4); pi4 = _mm512_fmadd_pd(kc9, si, pi4);
    pr5 = _mm512_fmadd_pd(kc6, sr, pr5); pi5 = _mm512_fmadd_pd(kc6, si, pi5);
    pr6 = _mm512_fmadd_pd(kc2, sr, pr6); pi6 = _mm512_fmadd_pd(kc2, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+128), si = _mm512_load_pd(sscr+136);
    pr1 = _mm512_fmadd_pd(kc9, sr, pr1); pi1 = _mm512_fmadd_pd(kc9, si, pi1);
    pr2 = _mm512_fmadd_pd(kc5, sr, pr2); pi2 = _mm512_fmadd_pd(kc5, si, pi2);
    pr3 = _mm512_fmadd_pd(kc4, sr, pr3); pi3 = _mm512_fmadd_pd(kc4, si, pi3);
    pr4 = _mm512_fmadd_pd(kc10, sr, pr4); pi4 = _mm512_fmadd_pd(kc10, si, pi4);
    pr5 = _mm512_fmadd_pd(kc1, sr, pr5); pi5 = _mm512_fmadd_pd(kc1, si, pi5);
    pr6 = _mm512_fmadd_pd(kc8, sr, pr6); pi6 = _mm512_fmadd_pd(kc8, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+144), si = _mm512_load_pd(sscr+152);
    pr1 = _mm512_fmadd_pd(kc10, sr, pr1); pi1 = _mm512_fmadd_pd(kc10, si, pi1);
    pr2 = _mm512_fmadd_pd(kc3, sr, pr2); pi2 = _mm512_fmadd_pd(kc3, si, pi2);
    pr3 = _mm512_fmadd_pd(kc7, sr, pr3); pi3 = _mm512_fmadd_pd(kc7, si, pi3);
    pr4 = _mm512_fmadd_pd(kc6, sr, pr4); pi4 = _mm512_fmadd_pd(kc6, si, pi4);
    pr5 = _mm512_fmadd_pd(kc4, sr, pr5); pi5 = _mm512_fmadd_pd(kc4, si, pi5);
    pr6 = _mm512_fmadd_pd(kc9, sr, pr6); pi6 = _mm512_fmadd_pd(kc9, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+160), si = _mm512_load_pd(sscr+168);
    pr1 = _mm512_fmadd_pd(kc11, sr, pr1); pi1 = _mm512_fmadd_pd(kc11, si, pi1);
    pr2 = _mm512_fmadd_pd(kc1, sr, pr2); pi2 = _mm512_fmadd_pd(kc1, si, pi2);
    pr3 = _mm512_fmadd_pd(kc10, sr, pr3); pi3 = _mm512_fmadd_pd(kc10, si, pi3);
    pr4 = _mm512_fmadd_pd(kc2, sr, pr4); pi4 = _mm512_fmadd_pd(kc2, si, pi4);
    pr5 = _mm512_fmadd_pd(kc9, sr, pr5); pi5 = _mm512_fmadd_pd(kc9, si, pi5);
    pr6 = _mm512_fmadd_pd(kc3, sr, pr6); pi6 = _mm512_fmadd_pd(kc3, si, pi6);
    }
    _mm512_store_pd(pscr+16, pr1); _mm512_store_pd(pscr+24, pi1);
    _mm512_store_pd(pscr+32, pr2); _mm512_store_pd(pscr+40, pi2);
    _mm512_store_pd(pscr+48, pr3); _mm512_store_pd(pscr+56, pi3);
    _mm512_store_pd(pscr+64, pr4); _mm512_store_pd(pscr+72, pi4);
    _mm512_store_pd(pscr+80, pr5); _mm512_store_pd(pscr+88, pi5);
    _mm512_store_pd(pscr+96, pr6); _mm512_store_pd(pscr+104, pi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d u0r = _mm512_load_pd(x+0*0), u0i;
    u0r = _mm512_load_pd(pscr); u0i = _mm512_load_pd(pscr+8);
    __m512d kc1; BCV(kc1, KC_23[0]);
    __m512d kc2; BCV(kc2, KC_23[1]);
    __m512d kc3; BCV(kc3, KC_23[2]);
    __m512d kc4; BCV(kc4, KC_23[3]);
    __m512d kc5; BCV(kc5, KC_23[4]);
    __m512d kc6; BCV(kc6, KC_23[5]);
    __m512d kc7; BCV(kc7, KC_23[6]);
    __m512d kc8; BCV(kc8, KC_23[7]);
    __m512d kc9; BCV(kc9, KC_23[8]);
    __m512d kc10; BCV(kc10, KC_23[9]);
    __m512d kc11; BCV(kc11, KC_23[10]);
    __m512d pr7 = u0r, pi7 = u0i;
    __m512d pr8 = u0r, pi8 = u0i;
    __m512d pr9 = u0r, pi9 = u0i;
    __m512d pr10 = u0r, pi10 = u0i;
    __m512d pr11 = u0r, pi11 = u0i;
    {
    __m512d sr = _mm512_load_pd(sscr+0), si = _mm512_load_pd(sscr+8);
    pr7 = _mm512_fmadd_pd(kc7, sr, pr7); pi7 = _mm512_fmadd_pd(kc7, si, pi7);
    pr8 = _mm512_fmadd_pd(kc8, sr, pr8); pi8 = _mm512_fmadd_pd(kc8, si, pi8);
    pr9 = _mm512_fmadd_pd(kc9, sr, pr9); pi9 = _mm512_fmadd_pd(kc9, si, pi9);
    pr10 = _mm512_fmadd_pd(kc10, sr, pr10); pi10 = _mm512_fmadd_pd(kc10, si, pi10);
    pr11 = _mm512_fmadd_pd(kc11, sr, pr11); pi11 = _mm512_fmadd_pd(kc11, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+16), si = _mm512_load_pd(sscr+24);
    pr7 = _mm512_fmadd_pd(kc9, sr, pr7); pi7 = _mm512_fmadd_pd(kc9, si, pi7);
    pr8 = _mm512_fmadd_pd(kc7, sr, pr8); pi8 = _mm512_fmadd_pd(kc7, si, pi8);
    pr9 = _mm512_fmadd_pd(kc5, sr, pr9); pi9 = _mm512_fmadd_pd(kc5, si, pi9);
    pr10 = _mm512_fmadd_pd(kc3, sr, pr10); pi10 = _mm512_fmadd_pd(kc3, si, pi10);
    pr11 = _mm512_fmadd_pd(kc1, sr, pr11); pi11 = _mm512_fmadd_pd(kc1, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+32), si = _mm512_load_pd(sscr+40);
    pr7 = _mm512_fmadd_pd(kc2, sr, pr7); pi7 = _mm512_fmadd_pd(kc2, si, pi7);
    pr8 = _mm512_fmadd_pd(kc1, sr, pr8); pi8 = _mm512_fmadd_pd(kc1, si, pi8);
    pr9 = _mm512_fmadd_pd(kc4, sr, pr9); pi9 = _mm512_fmadd_pd(kc4, si, pi9);
    pr10 = _mm512_fmadd_pd(kc7, sr, pr10); pi10 = _mm512_fmadd_pd(kc7, si, pi10);
    pr11 = _mm512_fmadd_pd(kc10, sr, pr11); pi11 = _mm512_fmadd_pd(kc10, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+48), si = _mm512_load_pd(sscr+56);
    pr7 = _mm512_fmadd_pd(kc5, sr, pr7); pi7 = _mm512_fmadd_pd(kc5, si, pi7);
    pr8 = _mm512_fmadd_pd(kc9, sr, pr8); pi8 = _mm512_fmadd_pd(kc9, si, pi8);
    pr9 = _mm512_fmadd_pd(kc10, sr, pr9); pi9 = _mm512_fmadd_pd(kc10, si, pi9);
    pr10 = _mm512_fmadd_pd(kc6, sr, pr10); pi10 = _mm512_fmadd_pd(kc6, si, pi10);
    pr11 = _mm512_fmadd_pd(kc2, sr, pr11); pi11 = _mm512_fmadd_pd(kc2, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+64), si = _mm512_load_pd(sscr+72);
    pr7 = _mm512_fmadd_pd(kc11, sr, pr7); pi7 = _mm512_fmadd_pd(kc11, si, pi7);
    pr8 = _mm512_fmadd_pd(kc6, sr, pr8); pi8 = _mm512_fmadd_pd(kc6, si, pi8);
    pr9 = _mm512_fmadd_pd(kc1, sr, pr9); pi9 = _mm512_fmadd_pd(kc1, si, pi9);
    pr10 = _mm512_fmadd_pd(kc4, sr, pr10); pi10 = _mm512_fmadd_pd(kc4, si, pi10);
    pr11 = _mm512_fmadd_pd(kc9, sr, pr11); pi11 = _mm512_fmadd_pd(kc9, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+80), si = _mm512_load_pd(sscr+88);
    pr7 = _mm512_fmadd_pd(kc4, sr, pr7); pi7 = _mm512_fmadd_pd(kc4, si, pi7);
    pr8 = _mm512_fmadd_pd(kc2, sr, pr8); pi8 = _mm512_fmadd_pd(kc2, si, pi8);
    pr9 = _mm512_fmadd_pd(kc8, sr, pr9); pi9 = _mm512_fmadd_pd(kc8, si, pi9);
    pr10 = _mm512_fmadd_pd(kc9, sr, pr10); pi10 = _mm512_fmadd_pd(kc9, si, pi10);
    pr11 = _mm512_fmadd_pd(kc3, sr, pr11); pi11 = _mm512_fmadd_pd(kc3, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+96), si = _mm512_load_pd(sscr+104);
    pr7 = _mm512_fmadd_pd(kc3, sr, pr7); pi7 = _mm512_fmadd_pd(kc3, si, pi7);
    pr8 = _mm512_fmadd_pd(kc10, sr, pr8); pi8 = _mm512_fmadd_pd(kc10, si, pi8);
    pr9 = _mm512_fmadd_pd(kc6, sr, pr9); pi9 = _mm512_fmadd_pd(kc6, si, pi9);
    pr10 = _mm512_fmadd_pd(kc1, sr, pr10); pi10 = _mm512_fmadd_pd(kc1, si, pi10);
    pr11 = _mm512_fmadd_pd(kc8, sr, pr11); pi11 = _mm512_fmadd_pd(kc8, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+112), si = _mm512_load_pd(sscr+120);
    pr7 = _mm512_fmadd_pd(kc10, sr, pr7); pi7 = _mm512_fmadd_pd(kc10, si, pi7);
    pr8 = _mm512_fmadd_pd(kc5, sr, pr8); pi8 = _mm512_fmadd_pd(kc5, si, pi8);
    pr9 = _mm512_fmadd_pd(kc3, sr, pr9); pi9 = _mm512_fmadd_pd(kc3, si, pi9);
    pr10 = _mm512_fmadd_pd(kc11, sr, pr10); pi10 = _mm512_fmadd_pd(kc11, si, pi10);
    pr11 = _mm512_fmadd_pd(kc4, sr, pr11); pi11 = _mm512_fmadd_pd(kc4, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+128), si = _mm512_load_pd(sscr+136);
    pr7 = _mm512_fmadd_pd(kc6, sr, pr7); pi7 = _mm512_fmadd_pd(kc6, si, pi7);
    pr8 = _mm512_fmadd_pd(kc3, sr, pr8); pi8 = _mm512_fmadd_pd(kc3, si, pi8);
    pr9 = _mm512_fmadd_pd(kc11, sr, pr9); pi9 = _mm512_fmadd_pd(kc11, si, pi9);
    pr10 = _mm512_fmadd_pd(kc2, sr, pr10); pi10 = _mm512_fmadd_pd(kc2, si, pi10);
    pr11 = _mm512_fmadd_pd(kc7, sr, pr11); pi11 = _mm512_fmadd_pd(kc7, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+144), si = _mm512_load_pd(sscr+152);
    pr7 = _mm512_fmadd_pd(kc1, sr, pr7); pi7 = _mm512_fmadd_pd(kc1, si, pi7);
    pr8 = _mm512_fmadd_pd(kc11, sr, pr8); pi8 = _mm512_fmadd_pd(kc11, si, pi8);
    pr9 = _mm512_fmadd_pd(kc2, sr, pr9); pi9 = _mm512_fmadd_pd(kc2, si, pi9);
    pr10 = _mm512_fmadd_pd(kc8, sr, pr10); pi10 = _mm512_fmadd_pd(kc8, si, pi10);
    pr11 = _mm512_fmadd_pd(kc5, sr, pr11); pi11 = _mm512_fmadd_pd(kc5, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+160), si = _mm512_load_pd(sscr+168);
    pr7 = _mm512_fmadd_pd(kc8, sr, pr7); pi7 = _mm512_fmadd_pd(kc8, si, pi7);
    pr8 = _mm512_fmadd_pd(kc4, sr, pr8); pi8 = _mm512_fmadd_pd(kc4, si, pi8);
    pr9 = _mm512_fmadd_pd(kc7, sr, pr9); pi9 = _mm512_fmadd_pd(kc7, si, pi9);
    pr10 = _mm512_fmadd_pd(kc5, sr, pr10); pi10 = _mm512_fmadd_pd(kc5, si, pi10);
    pr11 = _mm512_fmadd_pd(kc6, sr, pr11); pi11 = _mm512_fmadd_pd(kc6, si, pi11);
    }
    _mm512_store_pd(pscr+112, pr7); _mm512_store_pd(pscr+120, pi7);
    _mm512_store_pd(pscr+128, pr8); _mm512_store_pd(pscr+136, pi8);
    _mm512_store_pd(pscr+144, pr9); _mm512_store_pd(pscr+152, pi9);
    _mm512_store_pd(pscr+160, pr10); _mm512_store_pd(pscr+168, pi10);
    _mm512_store_pd(pscr+176, pr11); _mm512_store_pd(pscr+184, pi11);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_23[0]);
    __m512d ks2; BCV(ks2, KS_23[1]);
    __m512d ks3; BCV(ks3, KS_23[2]);
    __m512d ks4; BCV(ks4, KS_23[3]);
    __m512d ks5; BCV(ks5, KS_23[4]);
    __m512d ks6; BCV(ks6, KS_23[5]);
    __m512d ks7; BCV(ks7, KS_23[6]);
    __m512d ks8; BCV(ks8, KS_23[7]);
    __m512d ks9; BCV(ks9, KS_23[8]);
    __m512d ks10; BCV(ks10, KS_23[9]);
    __m512d ks11; BCV(ks11, KS_23[10]);
    __m512d qr1, qi1;
    __m512d qr2, qi2;
    __m512d qr3, qi3;
    __m512d qr4, qi4;
    __m512d qr5, qi5;
    __m512d qr6, qi6;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr1 = _mm512_mul_pd(ks1, di); qi1 = _mm512_mul_pd(ks1, dr);
    qr2 = _mm512_mul_pd(ks2, di); qi2 = _mm512_mul_pd(ks2, dr);
    qr3 = _mm512_mul_pd(ks3, di); qi3 = _mm512_mul_pd(ks3, dr);
    qr4 = _mm512_mul_pd(ks4, di); qi4 = _mm512_mul_pd(ks4, dr);
    qr5 = _mm512_mul_pd(ks5, di); qi5 = _mm512_mul_pd(ks5, dr);
    qr6 = _mm512_mul_pd(ks6, di); qi6 = _mm512_mul_pd(ks6, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr1 = _mm512_fmadd_pd(ks2, di, qr1); qi1 = _mm512_fmadd_pd(ks2, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks4, di, qr2); qi2 = _mm512_fmadd_pd(ks4, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks6, di, qr3); qi3 = _mm512_fmadd_pd(ks6, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks8, di, qr4); qi4 = _mm512_fmadd_pd(ks8, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks10, di, qr5); qi5 = _mm512_fmadd_pd(ks10, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks11, di, qr6); qi6 = _mm512_fnmadd_pd(ks11, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr1 = _mm512_fmadd_pd(ks3, di, qr1); qi1 = _mm512_fmadd_pd(ks3, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks6, di, qr2); qi2 = _mm512_fmadd_pd(ks6, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks9, di, qr3); qi3 = _mm512_fmadd_pd(ks9, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks11, di, qr4); qi4 = _mm512_fnmadd_pd(ks11, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks8, di, qr5); qi5 = _mm512_fnmadd_pd(ks8, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks5, di, qr6); qi6 = _mm512_fnmadd_pd(ks5, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr1 = _mm512_fmadd_pd(ks4, di, qr1); qi1 = _mm512_fmadd_pd(ks4, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks8, di, qr2); qi2 = _mm512_fmadd_pd(ks8, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks11, di, qr3); qi3 = _mm512_fnmadd_pd(ks11, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks7, di, qr4); qi4 = _mm512_fnmadd_pd(ks7, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks3, di, qr5); qi5 = _mm512_fnmadd_pd(ks3, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks1, di, qr6); qi6 = _mm512_fmadd_pd(ks1, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr1 = _mm512_fmadd_pd(ks5, di, qr1); qi1 = _mm512_fmadd_pd(ks5, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks10, di, qr2); qi2 = _mm512_fmadd_pd(ks10, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks8, di, qr3); qi3 = _mm512_fnmadd_pd(ks8, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks3, di, qr4); qi4 = _mm512_fnmadd_pd(ks3, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks2, di, qr5); qi5 = _mm512_fmadd_pd(ks2, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks7, di, qr6); qi6 = _mm512_fmadd_pd(ks7, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr1 = _mm512_fmadd_pd(ks6, di, qr1); qi1 = _mm512_fmadd_pd(ks6, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks11, di, qr2); qi2 = _mm512_fnmadd_pd(ks11, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks5, di, qr3); qi3 = _mm512_fnmadd_pd(ks5, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks1, di, qr4); qi4 = _mm512_fmadd_pd(ks1, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks7, di, qr5); qi5 = _mm512_fmadd_pd(ks7, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks10, di, qr6); qi6 = _mm512_fnmadd_pd(ks10, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+96), di = _mm512_load_pd(dscr+104);
    qr1 = _mm512_fmadd_pd(ks7, di, qr1); qi1 = _mm512_fmadd_pd(ks7, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks9, di, qr2); qi2 = _mm512_fnmadd_pd(ks9, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks2, di, qr3); qi3 = _mm512_fnmadd_pd(ks2, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks5, di, qr4); qi4 = _mm512_fmadd_pd(ks5, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks11, di, qr5); qi5 = _mm512_fnmadd_pd(ks11, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks4, di, qr6); qi6 = _mm512_fnmadd_pd(ks4, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+112), di = _mm512_load_pd(dscr+120);
    qr1 = _mm512_fmadd_pd(ks8, di, qr1); qi1 = _mm512_fmadd_pd(ks8, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks7, di, qr2); qi2 = _mm512_fnmadd_pd(ks7, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks1, di, qr3); qi3 = _mm512_fmadd_pd(ks1, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks9, di, qr4); qi4 = _mm512_fmadd_pd(ks9, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks6, di, qr5); qi5 = _mm512_fnmadd_pd(ks6, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks2, di, qr6); qi6 = _mm512_fmadd_pd(ks2, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+128), di = _mm512_load_pd(dscr+136);
    qr1 = _mm512_fmadd_pd(ks9, di, qr1); qi1 = _mm512_fmadd_pd(ks9, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks5, di, qr2); qi2 = _mm512_fnmadd_pd(ks5, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks4, di, qr3); qi3 = _mm512_fmadd_pd(ks4, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks10, di, qr4); qi4 = _mm512_fnmadd_pd(ks10, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks1, di, qr5); qi5 = _mm512_fnmadd_pd(ks1, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks8, di, qr6); qi6 = _mm512_fmadd_pd(ks8, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+144), di = _mm512_load_pd(dscr+152);
    qr1 = _mm512_fmadd_pd(ks10, di, qr1); qi1 = _mm512_fmadd_pd(ks10, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks3, di, qr2); qi2 = _mm512_fnmadd_pd(ks3, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks7, di, qr3); qi3 = _mm512_fmadd_pd(ks7, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks6, di, qr4); qi4 = _mm512_fnmadd_pd(ks6, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks4, di, qr5); qi5 = _mm512_fmadd_pd(ks4, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks9, di, qr6); qi6 = _mm512_fnmadd_pd(ks9, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+160), di = _mm512_load_pd(dscr+168);
    qr1 = _mm512_fmadd_pd(ks11, di, qr1); qi1 = _mm512_fmadd_pd(ks11, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks1, di, qr2); qi2 = _mm512_fnmadd_pd(ks1, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks10, di, qr3); qi3 = _mm512_fmadd_pd(ks10, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks2, di, qr4); qi4 = _mm512_fnmadd_pd(ks2, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks9, di, qr5); qi5 = _mm512_fmadd_pd(ks9, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks3, di, qr6); qi6 = _mm512_fnmadd_pd(ks3, dr, qi6);
    }
    _mm512_store_pd(qscr+0, qr1); _mm512_store_pd(qscr+8, qi1);
    _mm512_store_pd(qscr+16, qr2); _mm512_store_pd(qscr+24, qi2);
    _mm512_store_pd(qscr+32, qr3); _mm512_store_pd(qscr+40, qi3);
    _mm512_store_pd(qscr+48, qr4); _mm512_store_pd(qscr+56, qi4);
    _mm512_store_pd(qscr+64, qr5); _mm512_store_pd(qscr+72, qi5);
    _mm512_store_pd(qscr+80, qr6); _mm512_store_pd(qscr+88, qi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_23[0]);
    __m512d ks2; BCV(ks2, KS_23[1]);
    __m512d ks3; BCV(ks3, KS_23[2]);
    __m512d ks4; BCV(ks4, KS_23[3]);
    __m512d ks5; BCV(ks5, KS_23[4]);
    __m512d ks6; BCV(ks6, KS_23[5]);
    __m512d ks7; BCV(ks7, KS_23[6]);
    __m512d ks8; BCV(ks8, KS_23[7]);
    __m512d ks9; BCV(ks9, KS_23[8]);
    __m512d ks10; BCV(ks10, KS_23[9]);
    __m512d ks11; BCV(ks11, KS_23[10]);
    __m512d qr7, qi7;
    __m512d qr8, qi8;
    __m512d qr9, qi9;
    __m512d qr10, qi10;
    __m512d qr11, qi11;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr7 = _mm512_mul_pd(ks7, di); qi7 = _mm512_mul_pd(ks7, dr);
    qr8 = _mm512_mul_pd(ks8, di); qi8 = _mm512_mul_pd(ks8, dr);
    qr9 = _mm512_mul_pd(ks9, di); qi9 = _mm512_mul_pd(ks9, dr);
    qr10 = _mm512_mul_pd(ks10, di); qi10 = _mm512_mul_pd(ks10, dr);
    qr11 = _mm512_mul_pd(ks11, di); qi11 = _mm512_mul_pd(ks11, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr7 = _mm512_fnmadd_pd(ks9, di, qr7); qi7 = _mm512_fnmadd_pd(ks9, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks7, di, qr8); qi8 = _mm512_fnmadd_pd(ks7, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks5, di, qr9); qi9 = _mm512_fnmadd_pd(ks5, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks3, di, qr10); qi10 = _mm512_fnmadd_pd(ks3, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks1, di, qr11); qi11 = _mm512_fnmadd_pd(ks1, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr7 = _mm512_fnmadd_pd(ks2, di, qr7); qi7 = _mm512_fnmadd_pd(ks2, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks1, di, qr8); qi8 = _mm512_fmadd_pd(ks1, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks4, di, qr9); qi9 = _mm512_fmadd_pd(ks4, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks7, di, qr10); qi10 = _mm512_fmadd_pd(ks7, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks10, di, qr11); qi11 = _mm512_fmadd_pd(ks10, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr7 = _mm512_fmadd_pd(ks5, di, qr7); qi7 = _mm512_fmadd_pd(ks5, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks9, di, qr8); qi8 = _mm512_fmadd_pd(ks9, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks10, di, qr9); qi9 = _mm512_fnmadd_pd(ks10, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks6, di, qr10); qi10 = _mm512_fnmadd_pd(ks6, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks2, di, qr11); qi11 = _mm512_fnmadd_pd(ks2, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr7 = _mm512_fnmadd_pd(ks11, di, qr7); qi7 = _mm512_fnmadd_pd(ks11, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks6, di, qr8); qi8 = _mm512_fnmadd_pd(ks6, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks1, di, qr9); qi9 = _mm512_fnmadd_pd(ks1, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks4, di, qr10); qi10 = _mm512_fmadd_pd(ks4, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks9, di, qr11); qi11 = _mm512_fmadd_pd(ks9, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr7 = _mm512_fnmadd_pd(ks4, di, qr7); qi7 = _mm512_fnmadd_pd(ks4, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks2, di, qr8); qi8 = _mm512_fmadd_pd(ks2, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks8, di, qr9); qi9 = _mm512_fmadd_pd(ks8, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks9, di, qr10); qi10 = _mm512_fnmadd_pd(ks9, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks3, di, qr11); qi11 = _mm512_fnmadd_pd(ks3, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+96), di = _mm512_load_pd(dscr+104);
    qr7 = _mm512_fmadd_pd(ks3, di, qr7); qi7 = _mm512_fmadd_pd(ks3, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks10, di, qr8); qi8 = _mm512_fmadd_pd(ks10, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks6, di, qr9); qi9 = _mm512_fnmadd_pd(ks6, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks1, di, qr10); qi10 = _mm512_fmadd_pd(ks1, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks8, di, qr11); qi11 = _mm512_fmadd_pd(ks8, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+112), di = _mm512_load_pd(dscr+120);
    qr7 = _mm512_fmadd_pd(ks10, di, qr7); qi7 = _mm512_fmadd_pd(ks10, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks5, di, qr8); qi8 = _mm512_fnmadd_pd(ks5, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks3, di, qr9); qi9 = _mm512_fmadd_pd(ks3, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks11, di, qr10); qi10 = _mm512_fmadd_pd(ks11, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks4, di, qr11); qi11 = _mm512_fnmadd_pd(ks4, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+128), di = _mm512_load_pd(dscr+136);
    qr7 = _mm512_fnmadd_pd(ks6, di, qr7); qi7 = _mm512_fnmadd_pd(ks6, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks3, di, qr8); qi8 = _mm512_fmadd_pd(ks3, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks11, di, qr9); qi9 = _mm512_fnmadd_pd(ks11, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks2, di, qr10); qi10 = _mm512_fnmadd_pd(ks2, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks7, di, qr11); qi11 = _mm512_fmadd_pd(ks7, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+144), di = _mm512_load_pd(dscr+152);
    qr7 = _mm512_fmadd_pd(ks1, di, qr7); qi7 = _mm512_fmadd_pd(ks1, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks11, di, qr8); qi8 = _mm512_fmadd_pd(ks11, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks2, di, qr9); qi9 = _mm512_fnmadd_pd(ks2, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks8, di, qr10); qi10 = _mm512_fmadd_pd(ks8, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks5, di, qr11); qi11 = _mm512_fnmadd_pd(ks5, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+160), di = _mm512_load_pd(dscr+168);
    qr7 = _mm512_fmadd_pd(ks8, di, qr7); qi7 = _mm512_fmadd_pd(ks8, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks4, di, qr8); qi8 = _mm512_fnmadd_pd(ks4, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks7, di, qr9); qi9 = _mm512_fmadd_pd(ks7, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks5, di, qr10); qi10 = _mm512_fnmadd_pd(ks5, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks6, di, qr11); qi11 = _mm512_fmadd_pd(ks6, dr, qi11);
    }
    _mm512_store_pd(qscr+96, qr7); _mm512_store_pd(qscr+104, qi7);
    _mm512_store_pd(qscr+112, qr8); _mm512_store_pd(qscr+120, qi8);
    _mm512_store_pd(qscr+128, qr9); _mm512_store_pd(qscr+136, qi9);
    _mm512_store_pd(qscr+144, qr10); _mm512_store_pd(qscr+152, qi10);
    _mm512_store_pd(qscr+160, qr11); _mm512_store_pd(qscr+168, qi11);
    }
    __asm__ volatile("" ::: "memory");
    {
    {
    __m512d Pr = _mm512_load_pd(pscr+16), Pi = _mm512_load_pd(pscr+24);
    __m512d Qr = _mm512_load_pd(qscr+0), Qi = _mm512_load_pd(qscr+8);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 8464, cb, 16);
    MAPST(yr, yi, x, 186208, cb, 352);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+32), Pi = _mm512_load_pd(pscr+40);
    __m512d Qr = _mm512_load_pd(qscr+16), Qi = _mm512_load_pd(qscr+24);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 16928, cb, 32);
    MAPST(yr, yi, x, 177744, cb, 336);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+48), Pi = _mm512_load_pd(pscr+56);
    __m512d Qr = _mm512_load_pd(qscr+32), Qi = _mm512_load_pd(qscr+40);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 25392, cb, 48);
    MAPST(yr, yi, x, 169280, cb, 320);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+64), Pi = _mm512_load_pd(pscr+72);
    __m512d Qr = _mm512_load_pd(qscr+48), Qi = _mm512_load_pd(qscr+56);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 33856, cb, 64);
    MAPST(yr, yi, x, 160816, cb, 304);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+80), Pi = _mm512_load_pd(pscr+88);
    __m512d Qr = _mm512_load_pd(qscr+64), Qi = _mm512_load_pd(qscr+72);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 42320, cb, 80);
    MAPST(yr, yi, x, 152352, cb, 288);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+96), Pi = _mm512_load_pd(pscr+104);
    __m512d Qr = _mm512_load_pd(qscr+80), Qi = _mm512_load_pd(qscr+88);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 50784, cb, 96);
    MAPST(yr, yi, x, 143888, cb, 272);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+112), Pi = _mm512_load_pd(pscr+120);
    __m512d Qr = _mm512_load_pd(qscr+96), Qi = _mm512_load_pd(qscr+104);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 59248, cb, 112);
    MAPST(yr, yi, x, 135424, cb, 256);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+128), Pi = _mm512_load_pd(pscr+136);
    __m512d Qr = _mm512_load_pd(qscr+112), Qi = _mm512_load_pd(qscr+120);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 67712, cb, 128);
    MAPST(yr, yi, x, 126960, cb, 240);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+144), Pi = _mm512_load_pd(pscr+152);
    __m512d Qr = _mm512_load_pd(qscr+128), Qi = _mm512_load_pd(qscr+136);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 76176, cb, 144);
    MAPST(yr, yi, x, 118496, cb, 224);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+160), Pi = _mm512_load_pd(pscr+168);
    __m512d Qr = _mm512_load_pd(qscr+144), Qi = _mm512_load_pd(qscr+152);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 84640, cb, 160);
    MAPST(yr, yi, x, 110032, cb, 208);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+176), Pi = _mm512_load_pd(pscr+184);
    __m512d Qr = _mm512_load_pd(qscr+160), Qi = _mm512_load_pd(qscr+168);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 93104, cb, 176);
    MAPST(yr, yi, x, 101568, cb, 192);
    }
    }
}

static void step2_23(double* restrict G, double* restrict G2, const double* restrict CP){
    (void)G2;
    for(int x=0; x<23; x++){
        double* pl = G + (long)x*529*16;
        for(int y=0; y<23; y++) dz_23(pl + (long)y*23*16);
        for(int z=0; z<23; z++) dy_23(pl + (long)z*16);
    }
    for(int e=0; e<529; e++)
        dx_23(G + (long)e*16, CP + (long)e*23*16);
}


void run2_23(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if(!G_23){ G_23 = alloc_arena(12167*16*8); G2_23 = alloc_arena(12167*16*8 + 4096) + 256; CP_23 = alloc_arena(12167*16*8 + 65536) + 128; CT_23 = alloc_arena(12167*16*8); }
    long G8 = B/8;
    for(long g=0; g<G8; g++){
        const double* sx[8]; const double* sc[8]; double* d1[8]; double* dm[8];
        for(int v=0; v<8; v++){
            long off = (g*8+v)*(long)12167*2;
            sx[v] = x0+off; sc[v] = c+off; d1[v] = out1+off; dm[v] = outm+off;
        }
        conv_in_23(sx, G_23);
        conv_in_23(sc, CT_23);
        for(long e=0; e<529; e++)
            for(int j=0; j<23; j++)
                memcpy(CP_23 + (e*(long)23 + j)*16, CT_23 + ((long)j*529 + e)*16, 128);
        for(long t=0; t<m; t++){
            step2_23(G_23, G2_23, CP_23);
            if(t==0 && m>1) conv_out_23(G_23, d1);
        }
        conv_out_23(G_23, dm);
        if(m==1) for(int v=0; v<8; v++) memcpy(d1[v], dm[v], 12167*16);
    }
}

static void __attribute__((noinline)) wz_23(double* restrict x){
    double sscr[22*8] ALIGN64;
    double dscr[22*8] ALIGN64;
    double pscr[22*8+16] ALIGN64;
    double qscr[22*8] ALIGN64;
    {
    __m512d u0r = _mm512_load_pd(x), u0i = _mm512_load_pd(x+8);
    __m512d x0r = u0r, x0i = u0i;
    {
    __m512d ar = _mm512_load_pd(x+48), ai = _mm512_load_pd(x+48+8);
    __m512d br = _mm512_load_pd(x+1056), bi = _mm512_load_pd(x+1056+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+0, sr); _mm512_store_pd(sscr+8, si);
    _mm512_store_pd(dscr+0, dr); _mm512_store_pd(dscr+8, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+96), ai = _mm512_load_pd(x+96+8);
    __m512d br = _mm512_load_pd(x+1008), bi = _mm512_load_pd(x+1008+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+16, sr); _mm512_store_pd(sscr+24, si);
    _mm512_store_pd(dscr+16, dr); _mm512_store_pd(dscr+24, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+144), ai = _mm512_load_pd(x+144+8);
    __m512d br = _mm512_load_pd(x+960), bi = _mm512_load_pd(x+960+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+32, sr); _mm512_store_pd(sscr+40, si);
    _mm512_store_pd(dscr+32, dr); _mm512_store_pd(dscr+40, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+192), ai = _mm512_load_pd(x+192+8);
    __m512d br = _mm512_load_pd(x+912), bi = _mm512_load_pd(x+912+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+48, sr); _mm512_store_pd(sscr+56, si);
    _mm512_store_pd(dscr+48, dr); _mm512_store_pd(dscr+56, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+240), ai = _mm512_load_pd(x+240+8);
    __m512d br = _mm512_load_pd(x+864), bi = _mm512_load_pd(x+864+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+64, sr); _mm512_store_pd(sscr+72, si);
    _mm512_store_pd(dscr+64, dr); _mm512_store_pd(dscr+72, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+288), ai = _mm512_load_pd(x+288+8);
    __m512d br = _mm512_load_pd(x+816), bi = _mm512_load_pd(x+816+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+80, sr); _mm512_store_pd(sscr+88, si);
    _mm512_store_pd(dscr+80, dr); _mm512_store_pd(dscr+88, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+336), ai = _mm512_load_pd(x+336+8);
    __m512d br = _mm512_load_pd(x+768), bi = _mm512_load_pd(x+768+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+96, sr); _mm512_store_pd(sscr+104, si);
    _mm512_store_pd(dscr+96, dr); _mm512_store_pd(dscr+104, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+384), ai = _mm512_load_pd(x+384+8);
    __m512d br = _mm512_load_pd(x+720), bi = _mm512_load_pd(x+720+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+112, sr); _mm512_store_pd(sscr+120, si);
    _mm512_store_pd(dscr+112, dr); _mm512_store_pd(dscr+120, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+432), ai = _mm512_load_pd(x+432+8);
    __m512d br = _mm512_load_pd(x+672), bi = _mm512_load_pd(x+672+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+128, sr); _mm512_store_pd(sscr+136, si);
    _mm512_store_pd(dscr+128, dr); _mm512_store_pd(dscr+136, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+480), ai = _mm512_load_pd(x+480+8);
    __m512d br = _mm512_load_pd(x+624), bi = _mm512_load_pd(x+624+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+144, sr); _mm512_store_pd(sscr+152, si);
    _mm512_store_pd(dscr+144, dr); _mm512_store_pd(dscr+152, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+528), ai = _mm512_load_pd(x+528+8);
    __m512d br = _mm512_load_pd(x+576), bi = _mm512_load_pd(x+576+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+160, sr); _mm512_store_pd(sscr+168, si);
    _mm512_store_pd(dscr+160, dr); _mm512_store_pd(dscr+168, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    _mm512_store_pd(x, x0r); _mm512_store_pd(x+8, x0i);
    __m512d u0r_s = u0r, u0i_s = u0i;
    _mm512_store_pd(pscr, u0r_s); _mm512_store_pd(pscr+8, u0i_s);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d u0r = _mm512_load_pd(x+0*0), u0i;
    u0r = _mm512_load_pd(pscr); u0i = _mm512_load_pd(pscr+8);
    __m512d kc1; BCV(kc1, KC_23[0]);
    __m512d kc2; BCV(kc2, KC_23[1]);
    __m512d kc3; BCV(kc3, KC_23[2]);
    __m512d kc4; BCV(kc4, KC_23[3]);
    __m512d kc5; BCV(kc5, KC_23[4]);
    __m512d kc6; BCV(kc6, KC_23[5]);
    __m512d kc7; BCV(kc7, KC_23[6]);
    __m512d kc8; BCV(kc8, KC_23[7]);
    __m512d kc9; BCV(kc9, KC_23[8]);
    __m512d kc10; BCV(kc10, KC_23[9]);
    __m512d kc11; BCV(kc11, KC_23[10]);
    __m512d pr1 = u0r, pi1 = u0i;
    __m512d pr2 = u0r, pi2 = u0i;
    __m512d pr3 = u0r, pi3 = u0i;
    __m512d pr4 = u0r, pi4 = u0i;
    __m512d pr5 = u0r, pi5 = u0i;
    __m512d pr6 = u0r, pi6 = u0i;
    {
    __m512d sr = _mm512_load_pd(sscr+0), si = _mm512_load_pd(sscr+8);
    pr1 = _mm512_fmadd_pd(kc1, sr, pr1); pi1 = _mm512_fmadd_pd(kc1, si, pi1);
    pr2 = _mm512_fmadd_pd(kc2, sr, pr2); pi2 = _mm512_fmadd_pd(kc2, si, pi2);
    pr3 = _mm512_fmadd_pd(kc3, sr, pr3); pi3 = _mm512_fmadd_pd(kc3, si, pi3);
    pr4 = _mm512_fmadd_pd(kc4, sr, pr4); pi4 = _mm512_fmadd_pd(kc4, si, pi4);
    pr5 = _mm512_fmadd_pd(kc5, sr, pr5); pi5 = _mm512_fmadd_pd(kc5, si, pi5);
    pr6 = _mm512_fmadd_pd(kc6, sr, pr6); pi6 = _mm512_fmadd_pd(kc6, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+16), si = _mm512_load_pd(sscr+24);
    pr1 = _mm512_fmadd_pd(kc2, sr, pr1); pi1 = _mm512_fmadd_pd(kc2, si, pi1);
    pr2 = _mm512_fmadd_pd(kc4, sr, pr2); pi2 = _mm512_fmadd_pd(kc4, si, pi2);
    pr3 = _mm512_fmadd_pd(kc6, sr, pr3); pi3 = _mm512_fmadd_pd(kc6, si, pi3);
    pr4 = _mm512_fmadd_pd(kc8, sr, pr4); pi4 = _mm512_fmadd_pd(kc8, si, pi4);
    pr5 = _mm512_fmadd_pd(kc10, sr, pr5); pi5 = _mm512_fmadd_pd(kc10, si, pi5);
    pr6 = _mm512_fmadd_pd(kc11, sr, pr6); pi6 = _mm512_fmadd_pd(kc11, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+32), si = _mm512_load_pd(sscr+40);
    pr1 = _mm512_fmadd_pd(kc3, sr, pr1); pi1 = _mm512_fmadd_pd(kc3, si, pi1);
    pr2 = _mm512_fmadd_pd(kc6, sr, pr2); pi2 = _mm512_fmadd_pd(kc6, si, pi2);
    pr3 = _mm512_fmadd_pd(kc9, sr, pr3); pi3 = _mm512_fmadd_pd(kc9, si, pi3);
    pr4 = _mm512_fmadd_pd(kc11, sr, pr4); pi4 = _mm512_fmadd_pd(kc11, si, pi4);
    pr5 = _mm512_fmadd_pd(kc8, sr, pr5); pi5 = _mm512_fmadd_pd(kc8, si, pi5);
    pr6 = _mm512_fmadd_pd(kc5, sr, pr6); pi6 = _mm512_fmadd_pd(kc5, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+48), si = _mm512_load_pd(sscr+56);
    pr1 = _mm512_fmadd_pd(kc4, sr, pr1); pi1 = _mm512_fmadd_pd(kc4, si, pi1);
    pr2 = _mm512_fmadd_pd(kc8, sr, pr2); pi2 = _mm512_fmadd_pd(kc8, si, pi2);
    pr3 = _mm512_fmadd_pd(kc11, sr, pr3); pi3 = _mm512_fmadd_pd(kc11, si, pi3);
    pr4 = _mm512_fmadd_pd(kc7, sr, pr4); pi4 = _mm512_fmadd_pd(kc7, si, pi4);
    pr5 = _mm512_fmadd_pd(kc3, sr, pr5); pi5 = _mm512_fmadd_pd(kc3, si, pi5);
    pr6 = _mm512_fmadd_pd(kc1, sr, pr6); pi6 = _mm512_fmadd_pd(kc1, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+64), si = _mm512_load_pd(sscr+72);
    pr1 = _mm512_fmadd_pd(kc5, sr, pr1); pi1 = _mm512_fmadd_pd(kc5, si, pi1);
    pr2 = _mm512_fmadd_pd(kc10, sr, pr2); pi2 = _mm512_fmadd_pd(kc10, si, pi2);
    pr3 = _mm512_fmadd_pd(kc8, sr, pr3); pi3 = _mm512_fmadd_pd(kc8, si, pi3);
    pr4 = _mm512_fmadd_pd(kc3, sr, pr4); pi4 = _mm512_fmadd_pd(kc3, si, pi4);
    pr5 = _mm512_fmadd_pd(kc2, sr, pr5); pi5 = _mm512_fmadd_pd(kc2, si, pi5);
    pr6 = _mm512_fmadd_pd(kc7, sr, pr6); pi6 = _mm512_fmadd_pd(kc7, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+80), si = _mm512_load_pd(sscr+88);
    pr1 = _mm512_fmadd_pd(kc6, sr, pr1); pi1 = _mm512_fmadd_pd(kc6, si, pi1);
    pr2 = _mm512_fmadd_pd(kc11, sr, pr2); pi2 = _mm512_fmadd_pd(kc11, si, pi2);
    pr3 = _mm512_fmadd_pd(kc5, sr, pr3); pi3 = _mm512_fmadd_pd(kc5, si, pi3);
    pr4 = _mm512_fmadd_pd(kc1, sr, pr4); pi4 = _mm512_fmadd_pd(kc1, si, pi4);
    pr5 = _mm512_fmadd_pd(kc7, sr, pr5); pi5 = _mm512_fmadd_pd(kc7, si, pi5);
    pr6 = _mm512_fmadd_pd(kc10, sr, pr6); pi6 = _mm512_fmadd_pd(kc10, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+96), si = _mm512_load_pd(sscr+104);
    pr1 = _mm512_fmadd_pd(kc7, sr, pr1); pi1 = _mm512_fmadd_pd(kc7, si, pi1);
    pr2 = _mm512_fmadd_pd(kc9, sr, pr2); pi2 = _mm512_fmadd_pd(kc9, si, pi2);
    pr3 = _mm512_fmadd_pd(kc2, sr, pr3); pi3 = _mm512_fmadd_pd(kc2, si, pi3);
    pr4 = _mm512_fmadd_pd(kc5, sr, pr4); pi4 = _mm512_fmadd_pd(kc5, si, pi4);
    pr5 = _mm512_fmadd_pd(kc11, sr, pr5); pi5 = _mm512_fmadd_pd(kc11, si, pi5);
    pr6 = _mm512_fmadd_pd(kc4, sr, pr6); pi6 = _mm512_fmadd_pd(kc4, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+112), si = _mm512_load_pd(sscr+120);
    pr1 = _mm512_fmadd_pd(kc8, sr, pr1); pi1 = _mm512_fmadd_pd(kc8, si, pi1);
    pr2 = _mm512_fmadd_pd(kc7, sr, pr2); pi2 = _mm512_fmadd_pd(kc7, si, pi2);
    pr3 = _mm512_fmadd_pd(kc1, sr, pr3); pi3 = _mm512_fmadd_pd(kc1, si, pi3);
    pr4 = _mm512_fmadd_pd(kc9, sr, pr4); pi4 = _mm512_fmadd_pd(kc9, si, pi4);
    pr5 = _mm512_fmadd_pd(kc6, sr, pr5); pi5 = _mm512_fmadd_pd(kc6, si, pi5);
    pr6 = _mm512_fmadd_pd(kc2, sr, pr6); pi6 = _mm512_fmadd_pd(kc2, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+128), si = _mm512_load_pd(sscr+136);
    pr1 = _mm512_fmadd_pd(kc9, sr, pr1); pi1 = _mm512_fmadd_pd(kc9, si, pi1);
    pr2 = _mm512_fmadd_pd(kc5, sr, pr2); pi2 = _mm512_fmadd_pd(kc5, si, pi2);
    pr3 = _mm512_fmadd_pd(kc4, sr, pr3); pi3 = _mm512_fmadd_pd(kc4, si, pi3);
    pr4 = _mm512_fmadd_pd(kc10, sr, pr4); pi4 = _mm512_fmadd_pd(kc10, si, pi4);
    pr5 = _mm512_fmadd_pd(kc1, sr, pr5); pi5 = _mm512_fmadd_pd(kc1, si, pi5);
    pr6 = _mm512_fmadd_pd(kc8, sr, pr6); pi6 = _mm512_fmadd_pd(kc8, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+144), si = _mm512_load_pd(sscr+152);
    pr1 = _mm512_fmadd_pd(kc10, sr, pr1); pi1 = _mm512_fmadd_pd(kc10, si, pi1);
    pr2 = _mm512_fmadd_pd(kc3, sr, pr2); pi2 = _mm512_fmadd_pd(kc3, si, pi2);
    pr3 = _mm512_fmadd_pd(kc7, sr, pr3); pi3 = _mm512_fmadd_pd(kc7, si, pi3);
    pr4 = _mm512_fmadd_pd(kc6, sr, pr4); pi4 = _mm512_fmadd_pd(kc6, si, pi4);
    pr5 = _mm512_fmadd_pd(kc4, sr, pr5); pi5 = _mm512_fmadd_pd(kc4, si, pi5);
    pr6 = _mm512_fmadd_pd(kc9, sr, pr6); pi6 = _mm512_fmadd_pd(kc9, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+160), si = _mm512_load_pd(sscr+168);
    pr1 = _mm512_fmadd_pd(kc11, sr, pr1); pi1 = _mm512_fmadd_pd(kc11, si, pi1);
    pr2 = _mm512_fmadd_pd(kc1, sr, pr2); pi2 = _mm512_fmadd_pd(kc1, si, pi2);
    pr3 = _mm512_fmadd_pd(kc10, sr, pr3); pi3 = _mm512_fmadd_pd(kc10, si, pi3);
    pr4 = _mm512_fmadd_pd(kc2, sr, pr4); pi4 = _mm512_fmadd_pd(kc2, si, pi4);
    pr5 = _mm512_fmadd_pd(kc9, sr, pr5); pi5 = _mm512_fmadd_pd(kc9, si, pi5);
    pr6 = _mm512_fmadd_pd(kc3, sr, pr6); pi6 = _mm512_fmadd_pd(kc3, si, pi6);
    }
    _mm512_store_pd(pscr+16, pr1); _mm512_store_pd(pscr+24, pi1);
    _mm512_store_pd(pscr+32, pr2); _mm512_store_pd(pscr+40, pi2);
    _mm512_store_pd(pscr+48, pr3); _mm512_store_pd(pscr+56, pi3);
    _mm512_store_pd(pscr+64, pr4); _mm512_store_pd(pscr+72, pi4);
    _mm512_store_pd(pscr+80, pr5); _mm512_store_pd(pscr+88, pi5);
    _mm512_store_pd(pscr+96, pr6); _mm512_store_pd(pscr+104, pi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d u0r = _mm512_load_pd(x+0*0), u0i;
    u0r = _mm512_load_pd(pscr); u0i = _mm512_load_pd(pscr+8);
    __m512d kc1; BCV(kc1, KC_23[0]);
    __m512d kc2; BCV(kc2, KC_23[1]);
    __m512d kc3; BCV(kc3, KC_23[2]);
    __m512d kc4; BCV(kc4, KC_23[3]);
    __m512d kc5; BCV(kc5, KC_23[4]);
    __m512d kc6; BCV(kc6, KC_23[5]);
    __m512d kc7; BCV(kc7, KC_23[6]);
    __m512d kc8; BCV(kc8, KC_23[7]);
    __m512d kc9; BCV(kc9, KC_23[8]);
    __m512d kc10; BCV(kc10, KC_23[9]);
    __m512d kc11; BCV(kc11, KC_23[10]);
    __m512d pr7 = u0r, pi7 = u0i;
    __m512d pr8 = u0r, pi8 = u0i;
    __m512d pr9 = u0r, pi9 = u0i;
    __m512d pr10 = u0r, pi10 = u0i;
    __m512d pr11 = u0r, pi11 = u0i;
    {
    __m512d sr = _mm512_load_pd(sscr+0), si = _mm512_load_pd(sscr+8);
    pr7 = _mm512_fmadd_pd(kc7, sr, pr7); pi7 = _mm512_fmadd_pd(kc7, si, pi7);
    pr8 = _mm512_fmadd_pd(kc8, sr, pr8); pi8 = _mm512_fmadd_pd(kc8, si, pi8);
    pr9 = _mm512_fmadd_pd(kc9, sr, pr9); pi9 = _mm512_fmadd_pd(kc9, si, pi9);
    pr10 = _mm512_fmadd_pd(kc10, sr, pr10); pi10 = _mm512_fmadd_pd(kc10, si, pi10);
    pr11 = _mm512_fmadd_pd(kc11, sr, pr11); pi11 = _mm512_fmadd_pd(kc11, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+16), si = _mm512_load_pd(sscr+24);
    pr7 = _mm512_fmadd_pd(kc9, sr, pr7); pi7 = _mm512_fmadd_pd(kc9, si, pi7);
    pr8 = _mm512_fmadd_pd(kc7, sr, pr8); pi8 = _mm512_fmadd_pd(kc7, si, pi8);
    pr9 = _mm512_fmadd_pd(kc5, sr, pr9); pi9 = _mm512_fmadd_pd(kc5, si, pi9);
    pr10 = _mm512_fmadd_pd(kc3, sr, pr10); pi10 = _mm512_fmadd_pd(kc3, si, pi10);
    pr11 = _mm512_fmadd_pd(kc1, sr, pr11); pi11 = _mm512_fmadd_pd(kc1, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+32), si = _mm512_load_pd(sscr+40);
    pr7 = _mm512_fmadd_pd(kc2, sr, pr7); pi7 = _mm512_fmadd_pd(kc2, si, pi7);
    pr8 = _mm512_fmadd_pd(kc1, sr, pr8); pi8 = _mm512_fmadd_pd(kc1, si, pi8);
    pr9 = _mm512_fmadd_pd(kc4, sr, pr9); pi9 = _mm512_fmadd_pd(kc4, si, pi9);
    pr10 = _mm512_fmadd_pd(kc7, sr, pr10); pi10 = _mm512_fmadd_pd(kc7, si, pi10);
    pr11 = _mm512_fmadd_pd(kc10, sr, pr11); pi11 = _mm512_fmadd_pd(kc10, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+48), si = _mm512_load_pd(sscr+56);
    pr7 = _mm512_fmadd_pd(kc5, sr, pr7); pi7 = _mm512_fmadd_pd(kc5, si, pi7);
    pr8 = _mm512_fmadd_pd(kc9, sr, pr8); pi8 = _mm512_fmadd_pd(kc9, si, pi8);
    pr9 = _mm512_fmadd_pd(kc10, sr, pr9); pi9 = _mm512_fmadd_pd(kc10, si, pi9);
    pr10 = _mm512_fmadd_pd(kc6, sr, pr10); pi10 = _mm512_fmadd_pd(kc6, si, pi10);
    pr11 = _mm512_fmadd_pd(kc2, sr, pr11); pi11 = _mm512_fmadd_pd(kc2, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+64), si = _mm512_load_pd(sscr+72);
    pr7 = _mm512_fmadd_pd(kc11, sr, pr7); pi7 = _mm512_fmadd_pd(kc11, si, pi7);
    pr8 = _mm512_fmadd_pd(kc6, sr, pr8); pi8 = _mm512_fmadd_pd(kc6, si, pi8);
    pr9 = _mm512_fmadd_pd(kc1, sr, pr9); pi9 = _mm512_fmadd_pd(kc1, si, pi9);
    pr10 = _mm512_fmadd_pd(kc4, sr, pr10); pi10 = _mm512_fmadd_pd(kc4, si, pi10);
    pr11 = _mm512_fmadd_pd(kc9, sr, pr11); pi11 = _mm512_fmadd_pd(kc9, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+80), si = _mm512_load_pd(sscr+88);
    pr7 = _mm512_fmadd_pd(kc4, sr, pr7); pi7 = _mm512_fmadd_pd(kc4, si, pi7);
    pr8 = _mm512_fmadd_pd(kc2, sr, pr8); pi8 = _mm512_fmadd_pd(kc2, si, pi8);
    pr9 = _mm512_fmadd_pd(kc8, sr, pr9); pi9 = _mm512_fmadd_pd(kc8, si, pi9);
    pr10 = _mm512_fmadd_pd(kc9, sr, pr10); pi10 = _mm512_fmadd_pd(kc9, si, pi10);
    pr11 = _mm512_fmadd_pd(kc3, sr, pr11); pi11 = _mm512_fmadd_pd(kc3, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+96), si = _mm512_load_pd(sscr+104);
    pr7 = _mm512_fmadd_pd(kc3, sr, pr7); pi7 = _mm512_fmadd_pd(kc3, si, pi7);
    pr8 = _mm512_fmadd_pd(kc10, sr, pr8); pi8 = _mm512_fmadd_pd(kc10, si, pi8);
    pr9 = _mm512_fmadd_pd(kc6, sr, pr9); pi9 = _mm512_fmadd_pd(kc6, si, pi9);
    pr10 = _mm512_fmadd_pd(kc1, sr, pr10); pi10 = _mm512_fmadd_pd(kc1, si, pi10);
    pr11 = _mm512_fmadd_pd(kc8, sr, pr11); pi11 = _mm512_fmadd_pd(kc8, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+112), si = _mm512_load_pd(sscr+120);
    pr7 = _mm512_fmadd_pd(kc10, sr, pr7); pi7 = _mm512_fmadd_pd(kc10, si, pi7);
    pr8 = _mm512_fmadd_pd(kc5, sr, pr8); pi8 = _mm512_fmadd_pd(kc5, si, pi8);
    pr9 = _mm512_fmadd_pd(kc3, sr, pr9); pi9 = _mm512_fmadd_pd(kc3, si, pi9);
    pr10 = _mm512_fmadd_pd(kc11, sr, pr10); pi10 = _mm512_fmadd_pd(kc11, si, pi10);
    pr11 = _mm512_fmadd_pd(kc4, sr, pr11); pi11 = _mm512_fmadd_pd(kc4, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+128), si = _mm512_load_pd(sscr+136);
    pr7 = _mm512_fmadd_pd(kc6, sr, pr7); pi7 = _mm512_fmadd_pd(kc6, si, pi7);
    pr8 = _mm512_fmadd_pd(kc3, sr, pr8); pi8 = _mm512_fmadd_pd(kc3, si, pi8);
    pr9 = _mm512_fmadd_pd(kc11, sr, pr9); pi9 = _mm512_fmadd_pd(kc11, si, pi9);
    pr10 = _mm512_fmadd_pd(kc2, sr, pr10); pi10 = _mm512_fmadd_pd(kc2, si, pi10);
    pr11 = _mm512_fmadd_pd(kc7, sr, pr11); pi11 = _mm512_fmadd_pd(kc7, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+144), si = _mm512_load_pd(sscr+152);
    pr7 = _mm512_fmadd_pd(kc1, sr, pr7); pi7 = _mm512_fmadd_pd(kc1, si, pi7);
    pr8 = _mm512_fmadd_pd(kc11, sr, pr8); pi8 = _mm512_fmadd_pd(kc11, si, pi8);
    pr9 = _mm512_fmadd_pd(kc2, sr, pr9); pi9 = _mm512_fmadd_pd(kc2, si, pi9);
    pr10 = _mm512_fmadd_pd(kc8, sr, pr10); pi10 = _mm512_fmadd_pd(kc8, si, pi10);
    pr11 = _mm512_fmadd_pd(kc5, sr, pr11); pi11 = _mm512_fmadd_pd(kc5, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+160), si = _mm512_load_pd(sscr+168);
    pr7 = _mm512_fmadd_pd(kc8, sr, pr7); pi7 = _mm512_fmadd_pd(kc8, si, pi7);
    pr8 = _mm512_fmadd_pd(kc4, sr, pr8); pi8 = _mm512_fmadd_pd(kc4, si, pi8);
    pr9 = _mm512_fmadd_pd(kc7, sr, pr9); pi9 = _mm512_fmadd_pd(kc7, si, pi9);
    pr10 = _mm512_fmadd_pd(kc5, sr, pr10); pi10 = _mm512_fmadd_pd(kc5, si, pi10);
    pr11 = _mm512_fmadd_pd(kc6, sr, pr11); pi11 = _mm512_fmadd_pd(kc6, si, pi11);
    }
    _mm512_store_pd(pscr+112, pr7); _mm512_store_pd(pscr+120, pi7);
    _mm512_store_pd(pscr+128, pr8); _mm512_store_pd(pscr+136, pi8);
    _mm512_store_pd(pscr+144, pr9); _mm512_store_pd(pscr+152, pi9);
    _mm512_store_pd(pscr+160, pr10); _mm512_store_pd(pscr+168, pi10);
    _mm512_store_pd(pscr+176, pr11); _mm512_store_pd(pscr+184, pi11);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_23[0]);
    __m512d ks2; BCV(ks2, KS_23[1]);
    __m512d ks3; BCV(ks3, KS_23[2]);
    __m512d ks4; BCV(ks4, KS_23[3]);
    __m512d ks5; BCV(ks5, KS_23[4]);
    __m512d ks6; BCV(ks6, KS_23[5]);
    __m512d ks7; BCV(ks7, KS_23[6]);
    __m512d ks8; BCV(ks8, KS_23[7]);
    __m512d ks9; BCV(ks9, KS_23[8]);
    __m512d ks10; BCV(ks10, KS_23[9]);
    __m512d ks11; BCV(ks11, KS_23[10]);
    __m512d qr1, qi1;
    __m512d qr2, qi2;
    __m512d qr3, qi3;
    __m512d qr4, qi4;
    __m512d qr5, qi5;
    __m512d qr6, qi6;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr1 = _mm512_mul_pd(ks1, di); qi1 = _mm512_mul_pd(ks1, dr);
    qr2 = _mm512_mul_pd(ks2, di); qi2 = _mm512_mul_pd(ks2, dr);
    qr3 = _mm512_mul_pd(ks3, di); qi3 = _mm512_mul_pd(ks3, dr);
    qr4 = _mm512_mul_pd(ks4, di); qi4 = _mm512_mul_pd(ks4, dr);
    qr5 = _mm512_mul_pd(ks5, di); qi5 = _mm512_mul_pd(ks5, dr);
    qr6 = _mm512_mul_pd(ks6, di); qi6 = _mm512_mul_pd(ks6, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr1 = _mm512_fmadd_pd(ks2, di, qr1); qi1 = _mm512_fmadd_pd(ks2, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks4, di, qr2); qi2 = _mm512_fmadd_pd(ks4, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks6, di, qr3); qi3 = _mm512_fmadd_pd(ks6, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks8, di, qr4); qi4 = _mm512_fmadd_pd(ks8, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks10, di, qr5); qi5 = _mm512_fmadd_pd(ks10, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks11, di, qr6); qi6 = _mm512_fnmadd_pd(ks11, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr1 = _mm512_fmadd_pd(ks3, di, qr1); qi1 = _mm512_fmadd_pd(ks3, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks6, di, qr2); qi2 = _mm512_fmadd_pd(ks6, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks9, di, qr3); qi3 = _mm512_fmadd_pd(ks9, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks11, di, qr4); qi4 = _mm512_fnmadd_pd(ks11, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks8, di, qr5); qi5 = _mm512_fnmadd_pd(ks8, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks5, di, qr6); qi6 = _mm512_fnmadd_pd(ks5, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr1 = _mm512_fmadd_pd(ks4, di, qr1); qi1 = _mm512_fmadd_pd(ks4, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks8, di, qr2); qi2 = _mm512_fmadd_pd(ks8, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks11, di, qr3); qi3 = _mm512_fnmadd_pd(ks11, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks7, di, qr4); qi4 = _mm512_fnmadd_pd(ks7, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks3, di, qr5); qi5 = _mm512_fnmadd_pd(ks3, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks1, di, qr6); qi6 = _mm512_fmadd_pd(ks1, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr1 = _mm512_fmadd_pd(ks5, di, qr1); qi1 = _mm512_fmadd_pd(ks5, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks10, di, qr2); qi2 = _mm512_fmadd_pd(ks10, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks8, di, qr3); qi3 = _mm512_fnmadd_pd(ks8, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks3, di, qr4); qi4 = _mm512_fnmadd_pd(ks3, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks2, di, qr5); qi5 = _mm512_fmadd_pd(ks2, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks7, di, qr6); qi6 = _mm512_fmadd_pd(ks7, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr1 = _mm512_fmadd_pd(ks6, di, qr1); qi1 = _mm512_fmadd_pd(ks6, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks11, di, qr2); qi2 = _mm512_fnmadd_pd(ks11, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks5, di, qr3); qi3 = _mm512_fnmadd_pd(ks5, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks1, di, qr4); qi4 = _mm512_fmadd_pd(ks1, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks7, di, qr5); qi5 = _mm512_fmadd_pd(ks7, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks10, di, qr6); qi6 = _mm512_fnmadd_pd(ks10, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+96), di = _mm512_load_pd(dscr+104);
    qr1 = _mm512_fmadd_pd(ks7, di, qr1); qi1 = _mm512_fmadd_pd(ks7, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks9, di, qr2); qi2 = _mm512_fnmadd_pd(ks9, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks2, di, qr3); qi3 = _mm512_fnmadd_pd(ks2, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks5, di, qr4); qi4 = _mm512_fmadd_pd(ks5, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks11, di, qr5); qi5 = _mm512_fnmadd_pd(ks11, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks4, di, qr6); qi6 = _mm512_fnmadd_pd(ks4, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+112), di = _mm512_load_pd(dscr+120);
    qr1 = _mm512_fmadd_pd(ks8, di, qr1); qi1 = _mm512_fmadd_pd(ks8, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks7, di, qr2); qi2 = _mm512_fnmadd_pd(ks7, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks1, di, qr3); qi3 = _mm512_fmadd_pd(ks1, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks9, di, qr4); qi4 = _mm512_fmadd_pd(ks9, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks6, di, qr5); qi5 = _mm512_fnmadd_pd(ks6, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks2, di, qr6); qi6 = _mm512_fmadd_pd(ks2, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+128), di = _mm512_load_pd(dscr+136);
    qr1 = _mm512_fmadd_pd(ks9, di, qr1); qi1 = _mm512_fmadd_pd(ks9, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks5, di, qr2); qi2 = _mm512_fnmadd_pd(ks5, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks4, di, qr3); qi3 = _mm512_fmadd_pd(ks4, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks10, di, qr4); qi4 = _mm512_fnmadd_pd(ks10, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks1, di, qr5); qi5 = _mm512_fnmadd_pd(ks1, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks8, di, qr6); qi6 = _mm512_fmadd_pd(ks8, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+144), di = _mm512_load_pd(dscr+152);
    qr1 = _mm512_fmadd_pd(ks10, di, qr1); qi1 = _mm512_fmadd_pd(ks10, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks3, di, qr2); qi2 = _mm512_fnmadd_pd(ks3, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks7, di, qr3); qi3 = _mm512_fmadd_pd(ks7, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks6, di, qr4); qi4 = _mm512_fnmadd_pd(ks6, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks4, di, qr5); qi5 = _mm512_fmadd_pd(ks4, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks9, di, qr6); qi6 = _mm512_fnmadd_pd(ks9, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+160), di = _mm512_load_pd(dscr+168);
    qr1 = _mm512_fmadd_pd(ks11, di, qr1); qi1 = _mm512_fmadd_pd(ks11, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks1, di, qr2); qi2 = _mm512_fnmadd_pd(ks1, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks10, di, qr3); qi3 = _mm512_fmadd_pd(ks10, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks2, di, qr4); qi4 = _mm512_fnmadd_pd(ks2, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks9, di, qr5); qi5 = _mm512_fmadd_pd(ks9, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks3, di, qr6); qi6 = _mm512_fnmadd_pd(ks3, dr, qi6);
    }
    _mm512_store_pd(qscr+0, qr1); _mm512_store_pd(qscr+8, qi1);
    _mm512_store_pd(qscr+16, qr2); _mm512_store_pd(qscr+24, qi2);
    _mm512_store_pd(qscr+32, qr3); _mm512_store_pd(qscr+40, qi3);
    _mm512_store_pd(qscr+48, qr4); _mm512_store_pd(qscr+56, qi4);
    _mm512_store_pd(qscr+64, qr5); _mm512_store_pd(qscr+72, qi5);
    _mm512_store_pd(qscr+80, qr6); _mm512_store_pd(qscr+88, qi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_23[0]);
    __m512d ks2; BCV(ks2, KS_23[1]);
    __m512d ks3; BCV(ks3, KS_23[2]);
    __m512d ks4; BCV(ks4, KS_23[3]);
    __m512d ks5; BCV(ks5, KS_23[4]);
    __m512d ks6; BCV(ks6, KS_23[5]);
    __m512d ks7; BCV(ks7, KS_23[6]);
    __m512d ks8; BCV(ks8, KS_23[7]);
    __m512d ks9; BCV(ks9, KS_23[8]);
    __m512d ks10; BCV(ks10, KS_23[9]);
    __m512d ks11; BCV(ks11, KS_23[10]);
    __m512d qr7, qi7;
    __m512d qr8, qi8;
    __m512d qr9, qi9;
    __m512d qr10, qi10;
    __m512d qr11, qi11;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr7 = _mm512_mul_pd(ks7, di); qi7 = _mm512_mul_pd(ks7, dr);
    qr8 = _mm512_mul_pd(ks8, di); qi8 = _mm512_mul_pd(ks8, dr);
    qr9 = _mm512_mul_pd(ks9, di); qi9 = _mm512_mul_pd(ks9, dr);
    qr10 = _mm512_mul_pd(ks10, di); qi10 = _mm512_mul_pd(ks10, dr);
    qr11 = _mm512_mul_pd(ks11, di); qi11 = _mm512_mul_pd(ks11, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr7 = _mm512_fnmadd_pd(ks9, di, qr7); qi7 = _mm512_fnmadd_pd(ks9, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks7, di, qr8); qi8 = _mm512_fnmadd_pd(ks7, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks5, di, qr9); qi9 = _mm512_fnmadd_pd(ks5, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks3, di, qr10); qi10 = _mm512_fnmadd_pd(ks3, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks1, di, qr11); qi11 = _mm512_fnmadd_pd(ks1, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr7 = _mm512_fnmadd_pd(ks2, di, qr7); qi7 = _mm512_fnmadd_pd(ks2, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks1, di, qr8); qi8 = _mm512_fmadd_pd(ks1, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks4, di, qr9); qi9 = _mm512_fmadd_pd(ks4, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks7, di, qr10); qi10 = _mm512_fmadd_pd(ks7, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks10, di, qr11); qi11 = _mm512_fmadd_pd(ks10, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr7 = _mm512_fmadd_pd(ks5, di, qr7); qi7 = _mm512_fmadd_pd(ks5, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks9, di, qr8); qi8 = _mm512_fmadd_pd(ks9, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks10, di, qr9); qi9 = _mm512_fnmadd_pd(ks10, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks6, di, qr10); qi10 = _mm512_fnmadd_pd(ks6, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks2, di, qr11); qi11 = _mm512_fnmadd_pd(ks2, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr7 = _mm512_fnmadd_pd(ks11, di, qr7); qi7 = _mm512_fnmadd_pd(ks11, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks6, di, qr8); qi8 = _mm512_fnmadd_pd(ks6, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks1, di, qr9); qi9 = _mm512_fnmadd_pd(ks1, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks4, di, qr10); qi10 = _mm512_fmadd_pd(ks4, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks9, di, qr11); qi11 = _mm512_fmadd_pd(ks9, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr7 = _mm512_fnmadd_pd(ks4, di, qr7); qi7 = _mm512_fnmadd_pd(ks4, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks2, di, qr8); qi8 = _mm512_fmadd_pd(ks2, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks8, di, qr9); qi9 = _mm512_fmadd_pd(ks8, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks9, di, qr10); qi10 = _mm512_fnmadd_pd(ks9, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks3, di, qr11); qi11 = _mm512_fnmadd_pd(ks3, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+96), di = _mm512_load_pd(dscr+104);
    qr7 = _mm512_fmadd_pd(ks3, di, qr7); qi7 = _mm512_fmadd_pd(ks3, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks10, di, qr8); qi8 = _mm512_fmadd_pd(ks10, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks6, di, qr9); qi9 = _mm512_fnmadd_pd(ks6, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks1, di, qr10); qi10 = _mm512_fmadd_pd(ks1, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks8, di, qr11); qi11 = _mm512_fmadd_pd(ks8, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+112), di = _mm512_load_pd(dscr+120);
    qr7 = _mm512_fmadd_pd(ks10, di, qr7); qi7 = _mm512_fmadd_pd(ks10, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks5, di, qr8); qi8 = _mm512_fnmadd_pd(ks5, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks3, di, qr9); qi9 = _mm512_fmadd_pd(ks3, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks11, di, qr10); qi10 = _mm512_fmadd_pd(ks11, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks4, di, qr11); qi11 = _mm512_fnmadd_pd(ks4, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+128), di = _mm512_load_pd(dscr+136);
    qr7 = _mm512_fnmadd_pd(ks6, di, qr7); qi7 = _mm512_fnmadd_pd(ks6, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks3, di, qr8); qi8 = _mm512_fmadd_pd(ks3, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks11, di, qr9); qi9 = _mm512_fnmadd_pd(ks11, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks2, di, qr10); qi10 = _mm512_fnmadd_pd(ks2, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks7, di, qr11); qi11 = _mm512_fmadd_pd(ks7, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+144), di = _mm512_load_pd(dscr+152);
    qr7 = _mm512_fmadd_pd(ks1, di, qr7); qi7 = _mm512_fmadd_pd(ks1, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks11, di, qr8); qi8 = _mm512_fmadd_pd(ks11, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks2, di, qr9); qi9 = _mm512_fnmadd_pd(ks2, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks8, di, qr10); qi10 = _mm512_fmadd_pd(ks8, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks5, di, qr11); qi11 = _mm512_fnmadd_pd(ks5, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+160), di = _mm512_load_pd(dscr+168);
    qr7 = _mm512_fmadd_pd(ks8, di, qr7); qi7 = _mm512_fmadd_pd(ks8, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks4, di, qr8); qi8 = _mm512_fnmadd_pd(ks4, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks7, di, qr9); qi9 = _mm512_fmadd_pd(ks7, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks5, di, qr10); qi10 = _mm512_fnmadd_pd(ks5, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks6, di, qr11); qi11 = _mm512_fmadd_pd(ks6, dr, qi11);
    }
    _mm512_store_pd(qscr+96, qr7); _mm512_store_pd(qscr+104, qi7);
    _mm512_store_pd(qscr+112, qr8); _mm512_store_pd(qscr+120, qi8);
    _mm512_store_pd(qscr+128, qr9); _mm512_store_pd(qscr+136, qi9);
    _mm512_store_pd(qscr+144, qr10); _mm512_store_pd(qscr+152, qi10);
    _mm512_store_pd(qscr+160, qr11); _mm512_store_pd(qscr+168, qi11);
    }
    __asm__ volatile("" ::: "memory");
    {
    {
    __m512d Pr = _mm512_load_pd(pscr+16), Pi = _mm512_load_pd(pscr+24);
    __m512d Qr = _mm512_load_pd(qscr+0), Qi = _mm512_load_pd(qscr+8);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+48, xr); _mm512_store_pd(x+48+8, xi);
    _mm512_store_pd(x+1056, yr); _mm512_store_pd(x+1056+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+32), Pi = _mm512_load_pd(pscr+40);
    __m512d Qr = _mm512_load_pd(qscr+16), Qi = _mm512_load_pd(qscr+24);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+96, xr); _mm512_store_pd(x+96+8, xi);
    _mm512_store_pd(x+1008, yr); _mm512_store_pd(x+1008+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+48), Pi = _mm512_load_pd(pscr+56);
    __m512d Qr = _mm512_load_pd(qscr+32), Qi = _mm512_load_pd(qscr+40);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+144, xr); _mm512_store_pd(x+144+8, xi);
    _mm512_store_pd(x+960, yr); _mm512_store_pd(x+960+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+64), Pi = _mm512_load_pd(pscr+72);
    __m512d Qr = _mm512_load_pd(qscr+48), Qi = _mm512_load_pd(qscr+56);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+192, xr); _mm512_store_pd(x+192+8, xi);
    _mm512_store_pd(x+912, yr); _mm512_store_pd(x+912+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+80), Pi = _mm512_load_pd(pscr+88);
    __m512d Qr = _mm512_load_pd(qscr+64), Qi = _mm512_load_pd(qscr+72);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+240, xr); _mm512_store_pd(x+240+8, xi);
    _mm512_store_pd(x+864, yr); _mm512_store_pd(x+864+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+96), Pi = _mm512_load_pd(pscr+104);
    __m512d Qr = _mm512_load_pd(qscr+80), Qi = _mm512_load_pd(qscr+88);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+288, xr); _mm512_store_pd(x+288+8, xi);
    _mm512_store_pd(x+816, yr); _mm512_store_pd(x+816+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+112), Pi = _mm512_load_pd(pscr+120);
    __m512d Qr = _mm512_load_pd(qscr+96), Qi = _mm512_load_pd(qscr+104);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+336, xr); _mm512_store_pd(x+336+8, xi);
    _mm512_store_pd(x+768, yr); _mm512_store_pd(x+768+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+128), Pi = _mm512_load_pd(pscr+136);
    __m512d Qr = _mm512_load_pd(qscr+112), Qi = _mm512_load_pd(qscr+120);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+384, xr); _mm512_store_pd(x+384+8, xi);
    _mm512_store_pd(x+720, yr); _mm512_store_pd(x+720+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+144), Pi = _mm512_load_pd(pscr+152);
    __m512d Qr = _mm512_load_pd(qscr+128), Qi = _mm512_load_pd(qscr+136);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+432, xr); _mm512_store_pd(x+432+8, xi);
    _mm512_store_pd(x+672, yr); _mm512_store_pd(x+672+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+160), Pi = _mm512_load_pd(pscr+168);
    __m512d Qr = _mm512_load_pd(qscr+144), Qi = _mm512_load_pd(qscr+152);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+480, xr); _mm512_store_pd(x+480+8, xi);
    _mm512_store_pd(x+624, yr); _mm512_store_pd(x+624+8, yi);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+176), Pi = _mm512_load_pd(pscr+184);
    __m512d Qr = _mm512_load_pd(qscr+160), Qi = _mm512_load_pd(qscr+168);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    _mm512_store_pd(x+528, xr); _mm512_store_pd(x+528+8, xi);
    _mm512_store_pd(x+576, yr); _mm512_store_pd(x+576+8, yi);
    }
    }
}
static void __attribute__((noinline)) wx_23(double* restrict x, const double* restrict cb){
    double sscr[22*8] ALIGN64;
    double dscr[22*8] ALIGN64;
    double pscr[22*8+16] ALIGN64;
    double qscr[22*8] ALIGN64;
    {
    __m512d u0r = _mm512_load_pd(x), u0i = _mm512_load_pd(x+8);
    __m512d x0r = u0r, x0i = u0i;
    {
    __m512d ar = _mm512_load_pd(x+1160), ai = _mm512_load_pd(x+1160+8);
    __m512d br = _mm512_load_pd(x+25520), bi = _mm512_load_pd(x+25520+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+0, sr); _mm512_store_pd(sscr+8, si);
    _mm512_store_pd(dscr+0, dr); _mm512_store_pd(dscr+8, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+2320), ai = _mm512_load_pd(x+2320+8);
    __m512d br = _mm512_load_pd(x+24360), bi = _mm512_load_pd(x+24360+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+16, sr); _mm512_store_pd(sscr+24, si);
    _mm512_store_pd(dscr+16, dr); _mm512_store_pd(dscr+24, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+3480), ai = _mm512_load_pd(x+3480+8);
    __m512d br = _mm512_load_pd(x+23200), bi = _mm512_load_pd(x+23200+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+32, sr); _mm512_store_pd(sscr+40, si);
    _mm512_store_pd(dscr+32, dr); _mm512_store_pd(dscr+40, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+4640), ai = _mm512_load_pd(x+4640+8);
    __m512d br = _mm512_load_pd(x+22040), bi = _mm512_load_pd(x+22040+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+48, sr); _mm512_store_pd(sscr+56, si);
    _mm512_store_pd(dscr+48, dr); _mm512_store_pd(dscr+56, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+5800), ai = _mm512_load_pd(x+5800+8);
    __m512d br = _mm512_load_pd(x+20880), bi = _mm512_load_pd(x+20880+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+64, sr); _mm512_store_pd(sscr+72, si);
    _mm512_store_pd(dscr+64, dr); _mm512_store_pd(dscr+72, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+6960), ai = _mm512_load_pd(x+6960+8);
    __m512d br = _mm512_load_pd(x+19720), bi = _mm512_load_pd(x+19720+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+80, sr); _mm512_store_pd(sscr+88, si);
    _mm512_store_pd(dscr+80, dr); _mm512_store_pd(dscr+88, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+8120), ai = _mm512_load_pd(x+8120+8);
    __m512d br = _mm512_load_pd(x+18560), bi = _mm512_load_pd(x+18560+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+96, sr); _mm512_store_pd(sscr+104, si);
    _mm512_store_pd(dscr+96, dr); _mm512_store_pd(dscr+104, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+9280), ai = _mm512_load_pd(x+9280+8);
    __m512d br = _mm512_load_pd(x+17400), bi = _mm512_load_pd(x+17400+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+112, sr); _mm512_store_pd(sscr+120, si);
    _mm512_store_pd(dscr+112, dr); _mm512_store_pd(dscr+120, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+10440), ai = _mm512_load_pd(x+10440+8);
    __m512d br = _mm512_load_pd(x+16240), bi = _mm512_load_pd(x+16240+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+128, sr); _mm512_store_pd(sscr+136, si);
    _mm512_store_pd(dscr+128, dr); _mm512_store_pd(dscr+136, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+11600), ai = _mm512_load_pd(x+11600+8);
    __m512d br = _mm512_load_pd(x+15080), bi = _mm512_load_pd(x+15080+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+144, sr); _mm512_store_pd(sscr+152, si);
    _mm512_store_pd(dscr+144, dr); _mm512_store_pd(dscr+152, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    {
    __m512d ar = _mm512_load_pd(x+12760), ai = _mm512_load_pd(x+12760+8);
    __m512d br = _mm512_load_pd(x+13920), bi = _mm512_load_pd(x+13920+8);
    __m512d sr = _mm512_add_pd(ar,br), si = _mm512_add_pd(ai,bi);
    __m512d dr = _mm512_sub_pd(ar,br), di = _mm512_sub_pd(ai,bi);
    _mm512_store_pd(sscr+160, sr); _mm512_store_pd(sscr+168, si);
    _mm512_store_pd(dscr+160, dr); _mm512_store_pd(dscr+168, di);
    x0r = _mm512_add_pd(x0r, sr); x0i = _mm512_add_pd(x0i, si);
    }
    MAPST(x0r, x0i, x, 0, cb, 0);
    __m512d u0r_s = u0r, u0i_s = u0i;
    _mm512_store_pd(pscr, u0r_s); _mm512_store_pd(pscr+8, u0i_s);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d u0r = _mm512_load_pd(x+0*0), u0i;
    u0r = _mm512_load_pd(pscr); u0i = _mm512_load_pd(pscr+8);
    __m512d kc1; BCV(kc1, KC_23[0]);
    __m512d kc2; BCV(kc2, KC_23[1]);
    __m512d kc3; BCV(kc3, KC_23[2]);
    __m512d kc4; BCV(kc4, KC_23[3]);
    __m512d kc5; BCV(kc5, KC_23[4]);
    __m512d kc6; BCV(kc6, KC_23[5]);
    __m512d kc7; BCV(kc7, KC_23[6]);
    __m512d kc8; BCV(kc8, KC_23[7]);
    __m512d kc9; BCV(kc9, KC_23[8]);
    __m512d kc10; BCV(kc10, KC_23[9]);
    __m512d kc11; BCV(kc11, KC_23[10]);
    __m512d pr1 = u0r, pi1 = u0i;
    __m512d pr2 = u0r, pi2 = u0i;
    __m512d pr3 = u0r, pi3 = u0i;
    __m512d pr4 = u0r, pi4 = u0i;
    __m512d pr5 = u0r, pi5 = u0i;
    __m512d pr6 = u0r, pi6 = u0i;
    {
    __m512d sr = _mm512_load_pd(sscr+0), si = _mm512_load_pd(sscr+8);
    pr1 = _mm512_fmadd_pd(kc1, sr, pr1); pi1 = _mm512_fmadd_pd(kc1, si, pi1);
    pr2 = _mm512_fmadd_pd(kc2, sr, pr2); pi2 = _mm512_fmadd_pd(kc2, si, pi2);
    pr3 = _mm512_fmadd_pd(kc3, sr, pr3); pi3 = _mm512_fmadd_pd(kc3, si, pi3);
    pr4 = _mm512_fmadd_pd(kc4, sr, pr4); pi4 = _mm512_fmadd_pd(kc4, si, pi4);
    pr5 = _mm512_fmadd_pd(kc5, sr, pr5); pi5 = _mm512_fmadd_pd(kc5, si, pi5);
    pr6 = _mm512_fmadd_pd(kc6, sr, pr6); pi6 = _mm512_fmadd_pd(kc6, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+16), si = _mm512_load_pd(sscr+24);
    pr1 = _mm512_fmadd_pd(kc2, sr, pr1); pi1 = _mm512_fmadd_pd(kc2, si, pi1);
    pr2 = _mm512_fmadd_pd(kc4, sr, pr2); pi2 = _mm512_fmadd_pd(kc4, si, pi2);
    pr3 = _mm512_fmadd_pd(kc6, sr, pr3); pi3 = _mm512_fmadd_pd(kc6, si, pi3);
    pr4 = _mm512_fmadd_pd(kc8, sr, pr4); pi4 = _mm512_fmadd_pd(kc8, si, pi4);
    pr5 = _mm512_fmadd_pd(kc10, sr, pr5); pi5 = _mm512_fmadd_pd(kc10, si, pi5);
    pr6 = _mm512_fmadd_pd(kc11, sr, pr6); pi6 = _mm512_fmadd_pd(kc11, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+32), si = _mm512_load_pd(sscr+40);
    pr1 = _mm512_fmadd_pd(kc3, sr, pr1); pi1 = _mm512_fmadd_pd(kc3, si, pi1);
    pr2 = _mm512_fmadd_pd(kc6, sr, pr2); pi2 = _mm512_fmadd_pd(kc6, si, pi2);
    pr3 = _mm512_fmadd_pd(kc9, sr, pr3); pi3 = _mm512_fmadd_pd(kc9, si, pi3);
    pr4 = _mm512_fmadd_pd(kc11, sr, pr4); pi4 = _mm512_fmadd_pd(kc11, si, pi4);
    pr5 = _mm512_fmadd_pd(kc8, sr, pr5); pi5 = _mm512_fmadd_pd(kc8, si, pi5);
    pr6 = _mm512_fmadd_pd(kc5, sr, pr6); pi6 = _mm512_fmadd_pd(kc5, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+48), si = _mm512_load_pd(sscr+56);
    pr1 = _mm512_fmadd_pd(kc4, sr, pr1); pi1 = _mm512_fmadd_pd(kc4, si, pi1);
    pr2 = _mm512_fmadd_pd(kc8, sr, pr2); pi2 = _mm512_fmadd_pd(kc8, si, pi2);
    pr3 = _mm512_fmadd_pd(kc11, sr, pr3); pi3 = _mm512_fmadd_pd(kc11, si, pi3);
    pr4 = _mm512_fmadd_pd(kc7, sr, pr4); pi4 = _mm512_fmadd_pd(kc7, si, pi4);
    pr5 = _mm512_fmadd_pd(kc3, sr, pr5); pi5 = _mm512_fmadd_pd(kc3, si, pi5);
    pr6 = _mm512_fmadd_pd(kc1, sr, pr6); pi6 = _mm512_fmadd_pd(kc1, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+64), si = _mm512_load_pd(sscr+72);
    pr1 = _mm512_fmadd_pd(kc5, sr, pr1); pi1 = _mm512_fmadd_pd(kc5, si, pi1);
    pr2 = _mm512_fmadd_pd(kc10, sr, pr2); pi2 = _mm512_fmadd_pd(kc10, si, pi2);
    pr3 = _mm512_fmadd_pd(kc8, sr, pr3); pi3 = _mm512_fmadd_pd(kc8, si, pi3);
    pr4 = _mm512_fmadd_pd(kc3, sr, pr4); pi4 = _mm512_fmadd_pd(kc3, si, pi4);
    pr5 = _mm512_fmadd_pd(kc2, sr, pr5); pi5 = _mm512_fmadd_pd(kc2, si, pi5);
    pr6 = _mm512_fmadd_pd(kc7, sr, pr6); pi6 = _mm512_fmadd_pd(kc7, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+80), si = _mm512_load_pd(sscr+88);
    pr1 = _mm512_fmadd_pd(kc6, sr, pr1); pi1 = _mm512_fmadd_pd(kc6, si, pi1);
    pr2 = _mm512_fmadd_pd(kc11, sr, pr2); pi2 = _mm512_fmadd_pd(kc11, si, pi2);
    pr3 = _mm512_fmadd_pd(kc5, sr, pr3); pi3 = _mm512_fmadd_pd(kc5, si, pi3);
    pr4 = _mm512_fmadd_pd(kc1, sr, pr4); pi4 = _mm512_fmadd_pd(kc1, si, pi4);
    pr5 = _mm512_fmadd_pd(kc7, sr, pr5); pi5 = _mm512_fmadd_pd(kc7, si, pi5);
    pr6 = _mm512_fmadd_pd(kc10, sr, pr6); pi6 = _mm512_fmadd_pd(kc10, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+96), si = _mm512_load_pd(sscr+104);
    pr1 = _mm512_fmadd_pd(kc7, sr, pr1); pi1 = _mm512_fmadd_pd(kc7, si, pi1);
    pr2 = _mm512_fmadd_pd(kc9, sr, pr2); pi2 = _mm512_fmadd_pd(kc9, si, pi2);
    pr3 = _mm512_fmadd_pd(kc2, sr, pr3); pi3 = _mm512_fmadd_pd(kc2, si, pi3);
    pr4 = _mm512_fmadd_pd(kc5, sr, pr4); pi4 = _mm512_fmadd_pd(kc5, si, pi4);
    pr5 = _mm512_fmadd_pd(kc11, sr, pr5); pi5 = _mm512_fmadd_pd(kc11, si, pi5);
    pr6 = _mm512_fmadd_pd(kc4, sr, pr6); pi6 = _mm512_fmadd_pd(kc4, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+112), si = _mm512_load_pd(sscr+120);
    pr1 = _mm512_fmadd_pd(kc8, sr, pr1); pi1 = _mm512_fmadd_pd(kc8, si, pi1);
    pr2 = _mm512_fmadd_pd(kc7, sr, pr2); pi2 = _mm512_fmadd_pd(kc7, si, pi2);
    pr3 = _mm512_fmadd_pd(kc1, sr, pr3); pi3 = _mm512_fmadd_pd(kc1, si, pi3);
    pr4 = _mm512_fmadd_pd(kc9, sr, pr4); pi4 = _mm512_fmadd_pd(kc9, si, pi4);
    pr5 = _mm512_fmadd_pd(kc6, sr, pr5); pi5 = _mm512_fmadd_pd(kc6, si, pi5);
    pr6 = _mm512_fmadd_pd(kc2, sr, pr6); pi6 = _mm512_fmadd_pd(kc2, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+128), si = _mm512_load_pd(sscr+136);
    pr1 = _mm512_fmadd_pd(kc9, sr, pr1); pi1 = _mm512_fmadd_pd(kc9, si, pi1);
    pr2 = _mm512_fmadd_pd(kc5, sr, pr2); pi2 = _mm512_fmadd_pd(kc5, si, pi2);
    pr3 = _mm512_fmadd_pd(kc4, sr, pr3); pi3 = _mm512_fmadd_pd(kc4, si, pi3);
    pr4 = _mm512_fmadd_pd(kc10, sr, pr4); pi4 = _mm512_fmadd_pd(kc10, si, pi4);
    pr5 = _mm512_fmadd_pd(kc1, sr, pr5); pi5 = _mm512_fmadd_pd(kc1, si, pi5);
    pr6 = _mm512_fmadd_pd(kc8, sr, pr6); pi6 = _mm512_fmadd_pd(kc8, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+144), si = _mm512_load_pd(sscr+152);
    pr1 = _mm512_fmadd_pd(kc10, sr, pr1); pi1 = _mm512_fmadd_pd(kc10, si, pi1);
    pr2 = _mm512_fmadd_pd(kc3, sr, pr2); pi2 = _mm512_fmadd_pd(kc3, si, pi2);
    pr3 = _mm512_fmadd_pd(kc7, sr, pr3); pi3 = _mm512_fmadd_pd(kc7, si, pi3);
    pr4 = _mm512_fmadd_pd(kc6, sr, pr4); pi4 = _mm512_fmadd_pd(kc6, si, pi4);
    pr5 = _mm512_fmadd_pd(kc4, sr, pr5); pi5 = _mm512_fmadd_pd(kc4, si, pi5);
    pr6 = _mm512_fmadd_pd(kc9, sr, pr6); pi6 = _mm512_fmadd_pd(kc9, si, pi6);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+160), si = _mm512_load_pd(sscr+168);
    pr1 = _mm512_fmadd_pd(kc11, sr, pr1); pi1 = _mm512_fmadd_pd(kc11, si, pi1);
    pr2 = _mm512_fmadd_pd(kc1, sr, pr2); pi2 = _mm512_fmadd_pd(kc1, si, pi2);
    pr3 = _mm512_fmadd_pd(kc10, sr, pr3); pi3 = _mm512_fmadd_pd(kc10, si, pi3);
    pr4 = _mm512_fmadd_pd(kc2, sr, pr4); pi4 = _mm512_fmadd_pd(kc2, si, pi4);
    pr5 = _mm512_fmadd_pd(kc9, sr, pr5); pi5 = _mm512_fmadd_pd(kc9, si, pi5);
    pr6 = _mm512_fmadd_pd(kc3, sr, pr6); pi6 = _mm512_fmadd_pd(kc3, si, pi6);
    }
    _mm512_store_pd(pscr+16, pr1); _mm512_store_pd(pscr+24, pi1);
    _mm512_store_pd(pscr+32, pr2); _mm512_store_pd(pscr+40, pi2);
    _mm512_store_pd(pscr+48, pr3); _mm512_store_pd(pscr+56, pi3);
    _mm512_store_pd(pscr+64, pr4); _mm512_store_pd(pscr+72, pi4);
    _mm512_store_pd(pscr+80, pr5); _mm512_store_pd(pscr+88, pi5);
    _mm512_store_pd(pscr+96, pr6); _mm512_store_pd(pscr+104, pi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d u0r = _mm512_load_pd(x+0*0), u0i;
    u0r = _mm512_load_pd(pscr); u0i = _mm512_load_pd(pscr+8);
    __m512d kc1; BCV(kc1, KC_23[0]);
    __m512d kc2; BCV(kc2, KC_23[1]);
    __m512d kc3; BCV(kc3, KC_23[2]);
    __m512d kc4; BCV(kc4, KC_23[3]);
    __m512d kc5; BCV(kc5, KC_23[4]);
    __m512d kc6; BCV(kc6, KC_23[5]);
    __m512d kc7; BCV(kc7, KC_23[6]);
    __m512d kc8; BCV(kc8, KC_23[7]);
    __m512d kc9; BCV(kc9, KC_23[8]);
    __m512d kc10; BCV(kc10, KC_23[9]);
    __m512d kc11; BCV(kc11, KC_23[10]);
    __m512d pr7 = u0r, pi7 = u0i;
    __m512d pr8 = u0r, pi8 = u0i;
    __m512d pr9 = u0r, pi9 = u0i;
    __m512d pr10 = u0r, pi10 = u0i;
    __m512d pr11 = u0r, pi11 = u0i;
    {
    __m512d sr = _mm512_load_pd(sscr+0), si = _mm512_load_pd(sscr+8);
    pr7 = _mm512_fmadd_pd(kc7, sr, pr7); pi7 = _mm512_fmadd_pd(kc7, si, pi7);
    pr8 = _mm512_fmadd_pd(kc8, sr, pr8); pi8 = _mm512_fmadd_pd(kc8, si, pi8);
    pr9 = _mm512_fmadd_pd(kc9, sr, pr9); pi9 = _mm512_fmadd_pd(kc9, si, pi9);
    pr10 = _mm512_fmadd_pd(kc10, sr, pr10); pi10 = _mm512_fmadd_pd(kc10, si, pi10);
    pr11 = _mm512_fmadd_pd(kc11, sr, pr11); pi11 = _mm512_fmadd_pd(kc11, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+16), si = _mm512_load_pd(sscr+24);
    pr7 = _mm512_fmadd_pd(kc9, sr, pr7); pi7 = _mm512_fmadd_pd(kc9, si, pi7);
    pr8 = _mm512_fmadd_pd(kc7, sr, pr8); pi8 = _mm512_fmadd_pd(kc7, si, pi8);
    pr9 = _mm512_fmadd_pd(kc5, sr, pr9); pi9 = _mm512_fmadd_pd(kc5, si, pi9);
    pr10 = _mm512_fmadd_pd(kc3, sr, pr10); pi10 = _mm512_fmadd_pd(kc3, si, pi10);
    pr11 = _mm512_fmadd_pd(kc1, sr, pr11); pi11 = _mm512_fmadd_pd(kc1, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+32), si = _mm512_load_pd(sscr+40);
    pr7 = _mm512_fmadd_pd(kc2, sr, pr7); pi7 = _mm512_fmadd_pd(kc2, si, pi7);
    pr8 = _mm512_fmadd_pd(kc1, sr, pr8); pi8 = _mm512_fmadd_pd(kc1, si, pi8);
    pr9 = _mm512_fmadd_pd(kc4, sr, pr9); pi9 = _mm512_fmadd_pd(kc4, si, pi9);
    pr10 = _mm512_fmadd_pd(kc7, sr, pr10); pi10 = _mm512_fmadd_pd(kc7, si, pi10);
    pr11 = _mm512_fmadd_pd(kc10, sr, pr11); pi11 = _mm512_fmadd_pd(kc10, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+48), si = _mm512_load_pd(sscr+56);
    pr7 = _mm512_fmadd_pd(kc5, sr, pr7); pi7 = _mm512_fmadd_pd(kc5, si, pi7);
    pr8 = _mm512_fmadd_pd(kc9, sr, pr8); pi8 = _mm512_fmadd_pd(kc9, si, pi8);
    pr9 = _mm512_fmadd_pd(kc10, sr, pr9); pi9 = _mm512_fmadd_pd(kc10, si, pi9);
    pr10 = _mm512_fmadd_pd(kc6, sr, pr10); pi10 = _mm512_fmadd_pd(kc6, si, pi10);
    pr11 = _mm512_fmadd_pd(kc2, sr, pr11); pi11 = _mm512_fmadd_pd(kc2, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+64), si = _mm512_load_pd(sscr+72);
    pr7 = _mm512_fmadd_pd(kc11, sr, pr7); pi7 = _mm512_fmadd_pd(kc11, si, pi7);
    pr8 = _mm512_fmadd_pd(kc6, sr, pr8); pi8 = _mm512_fmadd_pd(kc6, si, pi8);
    pr9 = _mm512_fmadd_pd(kc1, sr, pr9); pi9 = _mm512_fmadd_pd(kc1, si, pi9);
    pr10 = _mm512_fmadd_pd(kc4, sr, pr10); pi10 = _mm512_fmadd_pd(kc4, si, pi10);
    pr11 = _mm512_fmadd_pd(kc9, sr, pr11); pi11 = _mm512_fmadd_pd(kc9, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+80), si = _mm512_load_pd(sscr+88);
    pr7 = _mm512_fmadd_pd(kc4, sr, pr7); pi7 = _mm512_fmadd_pd(kc4, si, pi7);
    pr8 = _mm512_fmadd_pd(kc2, sr, pr8); pi8 = _mm512_fmadd_pd(kc2, si, pi8);
    pr9 = _mm512_fmadd_pd(kc8, sr, pr9); pi9 = _mm512_fmadd_pd(kc8, si, pi9);
    pr10 = _mm512_fmadd_pd(kc9, sr, pr10); pi10 = _mm512_fmadd_pd(kc9, si, pi10);
    pr11 = _mm512_fmadd_pd(kc3, sr, pr11); pi11 = _mm512_fmadd_pd(kc3, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+96), si = _mm512_load_pd(sscr+104);
    pr7 = _mm512_fmadd_pd(kc3, sr, pr7); pi7 = _mm512_fmadd_pd(kc3, si, pi7);
    pr8 = _mm512_fmadd_pd(kc10, sr, pr8); pi8 = _mm512_fmadd_pd(kc10, si, pi8);
    pr9 = _mm512_fmadd_pd(kc6, sr, pr9); pi9 = _mm512_fmadd_pd(kc6, si, pi9);
    pr10 = _mm512_fmadd_pd(kc1, sr, pr10); pi10 = _mm512_fmadd_pd(kc1, si, pi10);
    pr11 = _mm512_fmadd_pd(kc8, sr, pr11); pi11 = _mm512_fmadd_pd(kc8, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+112), si = _mm512_load_pd(sscr+120);
    pr7 = _mm512_fmadd_pd(kc10, sr, pr7); pi7 = _mm512_fmadd_pd(kc10, si, pi7);
    pr8 = _mm512_fmadd_pd(kc5, sr, pr8); pi8 = _mm512_fmadd_pd(kc5, si, pi8);
    pr9 = _mm512_fmadd_pd(kc3, sr, pr9); pi9 = _mm512_fmadd_pd(kc3, si, pi9);
    pr10 = _mm512_fmadd_pd(kc11, sr, pr10); pi10 = _mm512_fmadd_pd(kc11, si, pi10);
    pr11 = _mm512_fmadd_pd(kc4, sr, pr11); pi11 = _mm512_fmadd_pd(kc4, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+128), si = _mm512_load_pd(sscr+136);
    pr7 = _mm512_fmadd_pd(kc6, sr, pr7); pi7 = _mm512_fmadd_pd(kc6, si, pi7);
    pr8 = _mm512_fmadd_pd(kc3, sr, pr8); pi8 = _mm512_fmadd_pd(kc3, si, pi8);
    pr9 = _mm512_fmadd_pd(kc11, sr, pr9); pi9 = _mm512_fmadd_pd(kc11, si, pi9);
    pr10 = _mm512_fmadd_pd(kc2, sr, pr10); pi10 = _mm512_fmadd_pd(kc2, si, pi10);
    pr11 = _mm512_fmadd_pd(kc7, sr, pr11); pi11 = _mm512_fmadd_pd(kc7, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+144), si = _mm512_load_pd(sscr+152);
    pr7 = _mm512_fmadd_pd(kc1, sr, pr7); pi7 = _mm512_fmadd_pd(kc1, si, pi7);
    pr8 = _mm512_fmadd_pd(kc11, sr, pr8); pi8 = _mm512_fmadd_pd(kc11, si, pi8);
    pr9 = _mm512_fmadd_pd(kc2, sr, pr9); pi9 = _mm512_fmadd_pd(kc2, si, pi9);
    pr10 = _mm512_fmadd_pd(kc8, sr, pr10); pi10 = _mm512_fmadd_pd(kc8, si, pi10);
    pr11 = _mm512_fmadd_pd(kc5, sr, pr11); pi11 = _mm512_fmadd_pd(kc5, si, pi11);
    }
    {
    __m512d sr = _mm512_load_pd(sscr+160), si = _mm512_load_pd(sscr+168);
    pr7 = _mm512_fmadd_pd(kc8, sr, pr7); pi7 = _mm512_fmadd_pd(kc8, si, pi7);
    pr8 = _mm512_fmadd_pd(kc4, sr, pr8); pi8 = _mm512_fmadd_pd(kc4, si, pi8);
    pr9 = _mm512_fmadd_pd(kc7, sr, pr9); pi9 = _mm512_fmadd_pd(kc7, si, pi9);
    pr10 = _mm512_fmadd_pd(kc5, sr, pr10); pi10 = _mm512_fmadd_pd(kc5, si, pi10);
    pr11 = _mm512_fmadd_pd(kc6, sr, pr11); pi11 = _mm512_fmadd_pd(kc6, si, pi11);
    }
    _mm512_store_pd(pscr+112, pr7); _mm512_store_pd(pscr+120, pi7);
    _mm512_store_pd(pscr+128, pr8); _mm512_store_pd(pscr+136, pi8);
    _mm512_store_pd(pscr+144, pr9); _mm512_store_pd(pscr+152, pi9);
    _mm512_store_pd(pscr+160, pr10); _mm512_store_pd(pscr+168, pi10);
    _mm512_store_pd(pscr+176, pr11); _mm512_store_pd(pscr+184, pi11);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_23[0]);
    __m512d ks2; BCV(ks2, KS_23[1]);
    __m512d ks3; BCV(ks3, KS_23[2]);
    __m512d ks4; BCV(ks4, KS_23[3]);
    __m512d ks5; BCV(ks5, KS_23[4]);
    __m512d ks6; BCV(ks6, KS_23[5]);
    __m512d ks7; BCV(ks7, KS_23[6]);
    __m512d ks8; BCV(ks8, KS_23[7]);
    __m512d ks9; BCV(ks9, KS_23[8]);
    __m512d ks10; BCV(ks10, KS_23[9]);
    __m512d ks11; BCV(ks11, KS_23[10]);
    __m512d qr1, qi1;
    __m512d qr2, qi2;
    __m512d qr3, qi3;
    __m512d qr4, qi4;
    __m512d qr5, qi5;
    __m512d qr6, qi6;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr1 = _mm512_mul_pd(ks1, di); qi1 = _mm512_mul_pd(ks1, dr);
    qr2 = _mm512_mul_pd(ks2, di); qi2 = _mm512_mul_pd(ks2, dr);
    qr3 = _mm512_mul_pd(ks3, di); qi3 = _mm512_mul_pd(ks3, dr);
    qr4 = _mm512_mul_pd(ks4, di); qi4 = _mm512_mul_pd(ks4, dr);
    qr5 = _mm512_mul_pd(ks5, di); qi5 = _mm512_mul_pd(ks5, dr);
    qr6 = _mm512_mul_pd(ks6, di); qi6 = _mm512_mul_pd(ks6, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr1 = _mm512_fmadd_pd(ks2, di, qr1); qi1 = _mm512_fmadd_pd(ks2, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks4, di, qr2); qi2 = _mm512_fmadd_pd(ks4, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks6, di, qr3); qi3 = _mm512_fmadd_pd(ks6, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks8, di, qr4); qi4 = _mm512_fmadd_pd(ks8, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks10, di, qr5); qi5 = _mm512_fmadd_pd(ks10, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks11, di, qr6); qi6 = _mm512_fnmadd_pd(ks11, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr1 = _mm512_fmadd_pd(ks3, di, qr1); qi1 = _mm512_fmadd_pd(ks3, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks6, di, qr2); qi2 = _mm512_fmadd_pd(ks6, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks9, di, qr3); qi3 = _mm512_fmadd_pd(ks9, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks11, di, qr4); qi4 = _mm512_fnmadd_pd(ks11, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks8, di, qr5); qi5 = _mm512_fnmadd_pd(ks8, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks5, di, qr6); qi6 = _mm512_fnmadd_pd(ks5, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr1 = _mm512_fmadd_pd(ks4, di, qr1); qi1 = _mm512_fmadd_pd(ks4, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks8, di, qr2); qi2 = _mm512_fmadd_pd(ks8, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks11, di, qr3); qi3 = _mm512_fnmadd_pd(ks11, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks7, di, qr4); qi4 = _mm512_fnmadd_pd(ks7, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks3, di, qr5); qi5 = _mm512_fnmadd_pd(ks3, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks1, di, qr6); qi6 = _mm512_fmadd_pd(ks1, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr1 = _mm512_fmadd_pd(ks5, di, qr1); qi1 = _mm512_fmadd_pd(ks5, dr, qi1);
    qr2 = _mm512_fmadd_pd(ks10, di, qr2); qi2 = _mm512_fmadd_pd(ks10, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks8, di, qr3); qi3 = _mm512_fnmadd_pd(ks8, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks3, di, qr4); qi4 = _mm512_fnmadd_pd(ks3, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks2, di, qr5); qi5 = _mm512_fmadd_pd(ks2, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks7, di, qr6); qi6 = _mm512_fmadd_pd(ks7, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr1 = _mm512_fmadd_pd(ks6, di, qr1); qi1 = _mm512_fmadd_pd(ks6, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks11, di, qr2); qi2 = _mm512_fnmadd_pd(ks11, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks5, di, qr3); qi3 = _mm512_fnmadd_pd(ks5, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks1, di, qr4); qi4 = _mm512_fmadd_pd(ks1, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks7, di, qr5); qi5 = _mm512_fmadd_pd(ks7, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks10, di, qr6); qi6 = _mm512_fnmadd_pd(ks10, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+96), di = _mm512_load_pd(dscr+104);
    qr1 = _mm512_fmadd_pd(ks7, di, qr1); qi1 = _mm512_fmadd_pd(ks7, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks9, di, qr2); qi2 = _mm512_fnmadd_pd(ks9, dr, qi2);
    qr3 = _mm512_fnmadd_pd(ks2, di, qr3); qi3 = _mm512_fnmadd_pd(ks2, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks5, di, qr4); qi4 = _mm512_fmadd_pd(ks5, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks11, di, qr5); qi5 = _mm512_fnmadd_pd(ks11, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks4, di, qr6); qi6 = _mm512_fnmadd_pd(ks4, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+112), di = _mm512_load_pd(dscr+120);
    qr1 = _mm512_fmadd_pd(ks8, di, qr1); qi1 = _mm512_fmadd_pd(ks8, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks7, di, qr2); qi2 = _mm512_fnmadd_pd(ks7, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks1, di, qr3); qi3 = _mm512_fmadd_pd(ks1, dr, qi3);
    qr4 = _mm512_fmadd_pd(ks9, di, qr4); qi4 = _mm512_fmadd_pd(ks9, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks6, di, qr5); qi5 = _mm512_fnmadd_pd(ks6, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks2, di, qr6); qi6 = _mm512_fmadd_pd(ks2, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+128), di = _mm512_load_pd(dscr+136);
    qr1 = _mm512_fmadd_pd(ks9, di, qr1); qi1 = _mm512_fmadd_pd(ks9, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks5, di, qr2); qi2 = _mm512_fnmadd_pd(ks5, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks4, di, qr3); qi3 = _mm512_fmadd_pd(ks4, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks10, di, qr4); qi4 = _mm512_fnmadd_pd(ks10, dr, qi4);
    qr5 = _mm512_fnmadd_pd(ks1, di, qr5); qi5 = _mm512_fnmadd_pd(ks1, dr, qi5);
    qr6 = _mm512_fmadd_pd(ks8, di, qr6); qi6 = _mm512_fmadd_pd(ks8, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+144), di = _mm512_load_pd(dscr+152);
    qr1 = _mm512_fmadd_pd(ks10, di, qr1); qi1 = _mm512_fmadd_pd(ks10, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks3, di, qr2); qi2 = _mm512_fnmadd_pd(ks3, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks7, di, qr3); qi3 = _mm512_fmadd_pd(ks7, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks6, di, qr4); qi4 = _mm512_fnmadd_pd(ks6, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks4, di, qr5); qi5 = _mm512_fmadd_pd(ks4, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks9, di, qr6); qi6 = _mm512_fnmadd_pd(ks9, dr, qi6);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+160), di = _mm512_load_pd(dscr+168);
    qr1 = _mm512_fmadd_pd(ks11, di, qr1); qi1 = _mm512_fmadd_pd(ks11, dr, qi1);
    qr2 = _mm512_fnmadd_pd(ks1, di, qr2); qi2 = _mm512_fnmadd_pd(ks1, dr, qi2);
    qr3 = _mm512_fmadd_pd(ks10, di, qr3); qi3 = _mm512_fmadd_pd(ks10, dr, qi3);
    qr4 = _mm512_fnmadd_pd(ks2, di, qr4); qi4 = _mm512_fnmadd_pd(ks2, dr, qi4);
    qr5 = _mm512_fmadd_pd(ks9, di, qr5); qi5 = _mm512_fmadd_pd(ks9, dr, qi5);
    qr6 = _mm512_fnmadd_pd(ks3, di, qr6); qi6 = _mm512_fnmadd_pd(ks3, dr, qi6);
    }
    _mm512_store_pd(qscr+0, qr1); _mm512_store_pd(qscr+8, qi1);
    _mm512_store_pd(qscr+16, qr2); _mm512_store_pd(qscr+24, qi2);
    _mm512_store_pd(qscr+32, qr3); _mm512_store_pd(qscr+40, qi3);
    _mm512_store_pd(qscr+48, qr4); _mm512_store_pd(qscr+56, qi4);
    _mm512_store_pd(qscr+64, qr5); _mm512_store_pd(qscr+72, qi5);
    _mm512_store_pd(qscr+80, qr6); _mm512_store_pd(qscr+88, qi6);
    }
    __asm__ volatile("" ::: "memory");
    {
    __m512d ks1; BCV(ks1, KS_23[0]);
    __m512d ks2; BCV(ks2, KS_23[1]);
    __m512d ks3; BCV(ks3, KS_23[2]);
    __m512d ks4; BCV(ks4, KS_23[3]);
    __m512d ks5; BCV(ks5, KS_23[4]);
    __m512d ks6; BCV(ks6, KS_23[5]);
    __m512d ks7; BCV(ks7, KS_23[6]);
    __m512d ks8; BCV(ks8, KS_23[7]);
    __m512d ks9; BCV(ks9, KS_23[8]);
    __m512d ks10; BCV(ks10, KS_23[9]);
    __m512d ks11; BCV(ks11, KS_23[10]);
    __m512d qr7, qi7;
    __m512d qr8, qi8;
    __m512d qr9, qi9;
    __m512d qr10, qi10;
    __m512d qr11, qi11;
    {
    __m512d dr = _mm512_load_pd(dscr+0), di = _mm512_load_pd(dscr+8);
    qr7 = _mm512_mul_pd(ks7, di); qi7 = _mm512_mul_pd(ks7, dr);
    qr8 = _mm512_mul_pd(ks8, di); qi8 = _mm512_mul_pd(ks8, dr);
    qr9 = _mm512_mul_pd(ks9, di); qi9 = _mm512_mul_pd(ks9, dr);
    qr10 = _mm512_mul_pd(ks10, di); qi10 = _mm512_mul_pd(ks10, dr);
    qr11 = _mm512_mul_pd(ks11, di); qi11 = _mm512_mul_pd(ks11, dr);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+16), di = _mm512_load_pd(dscr+24);
    qr7 = _mm512_fnmadd_pd(ks9, di, qr7); qi7 = _mm512_fnmadd_pd(ks9, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks7, di, qr8); qi8 = _mm512_fnmadd_pd(ks7, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks5, di, qr9); qi9 = _mm512_fnmadd_pd(ks5, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks3, di, qr10); qi10 = _mm512_fnmadd_pd(ks3, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks1, di, qr11); qi11 = _mm512_fnmadd_pd(ks1, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+32), di = _mm512_load_pd(dscr+40);
    qr7 = _mm512_fnmadd_pd(ks2, di, qr7); qi7 = _mm512_fnmadd_pd(ks2, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks1, di, qr8); qi8 = _mm512_fmadd_pd(ks1, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks4, di, qr9); qi9 = _mm512_fmadd_pd(ks4, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks7, di, qr10); qi10 = _mm512_fmadd_pd(ks7, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks10, di, qr11); qi11 = _mm512_fmadd_pd(ks10, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+48), di = _mm512_load_pd(dscr+56);
    qr7 = _mm512_fmadd_pd(ks5, di, qr7); qi7 = _mm512_fmadd_pd(ks5, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks9, di, qr8); qi8 = _mm512_fmadd_pd(ks9, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks10, di, qr9); qi9 = _mm512_fnmadd_pd(ks10, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks6, di, qr10); qi10 = _mm512_fnmadd_pd(ks6, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks2, di, qr11); qi11 = _mm512_fnmadd_pd(ks2, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+64), di = _mm512_load_pd(dscr+72);
    qr7 = _mm512_fnmadd_pd(ks11, di, qr7); qi7 = _mm512_fnmadd_pd(ks11, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks6, di, qr8); qi8 = _mm512_fnmadd_pd(ks6, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks1, di, qr9); qi9 = _mm512_fnmadd_pd(ks1, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks4, di, qr10); qi10 = _mm512_fmadd_pd(ks4, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks9, di, qr11); qi11 = _mm512_fmadd_pd(ks9, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+80), di = _mm512_load_pd(dscr+88);
    qr7 = _mm512_fnmadd_pd(ks4, di, qr7); qi7 = _mm512_fnmadd_pd(ks4, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks2, di, qr8); qi8 = _mm512_fmadd_pd(ks2, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks8, di, qr9); qi9 = _mm512_fmadd_pd(ks8, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks9, di, qr10); qi10 = _mm512_fnmadd_pd(ks9, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks3, di, qr11); qi11 = _mm512_fnmadd_pd(ks3, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+96), di = _mm512_load_pd(dscr+104);
    qr7 = _mm512_fmadd_pd(ks3, di, qr7); qi7 = _mm512_fmadd_pd(ks3, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks10, di, qr8); qi8 = _mm512_fmadd_pd(ks10, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks6, di, qr9); qi9 = _mm512_fnmadd_pd(ks6, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks1, di, qr10); qi10 = _mm512_fmadd_pd(ks1, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks8, di, qr11); qi11 = _mm512_fmadd_pd(ks8, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+112), di = _mm512_load_pd(dscr+120);
    qr7 = _mm512_fmadd_pd(ks10, di, qr7); qi7 = _mm512_fmadd_pd(ks10, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks5, di, qr8); qi8 = _mm512_fnmadd_pd(ks5, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks3, di, qr9); qi9 = _mm512_fmadd_pd(ks3, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks11, di, qr10); qi10 = _mm512_fmadd_pd(ks11, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks4, di, qr11); qi11 = _mm512_fnmadd_pd(ks4, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+128), di = _mm512_load_pd(dscr+136);
    qr7 = _mm512_fnmadd_pd(ks6, di, qr7); qi7 = _mm512_fnmadd_pd(ks6, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks3, di, qr8); qi8 = _mm512_fmadd_pd(ks3, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks11, di, qr9); qi9 = _mm512_fnmadd_pd(ks11, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks2, di, qr10); qi10 = _mm512_fnmadd_pd(ks2, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks7, di, qr11); qi11 = _mm512_fmadd_pd(ks7, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+144), di = _mm512_load_pd(dscr+152);
    qr7 = _mm512_fmadd_pd(ks1, di, qr7); qi7 = _mm512_fmadd_pd(ks1, dr, qi7);
    qr8 = _mm512_fmadd_pd(ks11, di, qr8); qi8 = _mm512_fmadd_pd(ks11, dr, qi8);
    qr9 = _mm512_fnmadd_pd(ks2, di, qr9); qi9 = _mm512_fnmadd_pd(ks2, dr, qi9);
    qr10 = _mm512_fmadd_pd(ks8, di, qr10); qi10 = _mm512_fmadd_pd(ks8, dr, qi10);
    qr11 = _mm512_fnmadd_pd(ks5, di, qr11); qi11 = _mm512_fnmadd_pd(ks5, dr, qi11);
    }
    {
    __m512d dr = _mm512_load_pd(dscr+160), di = _mm512_load_pd(dscr+168);
    qr7 = _mm512_fmadd_pd(ks8, di, qr7); qi7 = _mm512_fmadd_pd(ks8, dr, qi7);
    qr8 = _mm512_fnmadd_pd(ks4, di, qr8); qi8 = _mm512_fnmadd_pd(ks4, dr, qi8);
    qr9 = _mm512_fmadd_pd(ks7, di, qr9); qi9 = _mm512_fmadd_pd(ks7, dr, qi9);
    qr10 = _mm512_fnmadd_pd(ks5, di, qr10); qi10 = _mm512_fnmadd_pd(ks5, dr, qi10);
    qr11 = _mm512_fmadd_pd(ks6, di, qr11); qi11 = _mm512_fmadd_pd(ks6, dr, qi11);
    }
    _mm512_store_pd(qscr+96, qr7); _mm512_store_pd(qscr+104, qi7);
    _mm512_store_pd(qscr+112, qr8); _mm512_store_pd(qscr+120, qi8);
    _mm512_store_pd(qscr+128, qr9); _mm512_store_pd(qscr+136, qi9);
    _mm512_store_pd(qscr+144, qr10); _mm512_store_pd(qscr+152, qi10);
    _mm512_store_pd(qscr+160, qr11); _mm512_store_pd(qscr+168, qi11);
    }
    __asm__ volatile("" ::: "memory");
    {
    {
    __m512d Pr = _mm512_load_pd(pscr+16), Pi = _mm512_load_pd(pscr+24);
    __m512d Qr = _mm512_load_pd(qscr+0), Qi = _mm512_load_pd(qscr+8);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 1160, cb, 1160);
    MAPST(yr, yi, x, 25520, cb, 25520);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+32), Pi = _mm512_load_pd(pscr+40);
    __m512d Qr = _mm512_load_pd(qscr+16), Qi = _mm512_load_pd(qscr+24);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 2320, cb, 2320);
    MAPST(yr, yi, x, 24360, cb, 24360);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+48), Pi = _mm512_load_pd(pscr+56);
    __m512d Qr = _mm512_load_pd(qscr+32), Qi = _mm512_load_pd(qscr+40);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 3480, cb, 3480);
    MAPST(yr, yi, x, 23200, cb, 23200);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+64), Pi = _mm512_load_pd(pscr+72);
    __m512d Qr = _mm512_load_pd(qscr+48), Qi = _mm512_load_pd(qscr+56);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 4640, cb, 4640);
    MAPST(yr, yi, x, 22040, cb, 22040);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+80), Pi = _mm512_load_pd(pscr+88);
    __m512d Qr = _mm512_load_pd(qscr+64), Qi = _mm512_load_pd(qscr+72);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 5800, cb, 5800);
    MAPST(yr, yi, x, 20880, cb, 20880);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+96), Pi = _mm512_load_pd(pscr+104);
    __m512d Qr = _mm512_load_pd(qscr+80), Qi = _mm512_load_pd(qscr+88);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 6960, cb, 6960);
    MAPST(yr, yi, x, 19720, cb, 19720);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+112), Pi = _mm512_load_pd(pscr+120);
    __m512d Qr = _mm512_load_pd(qscr+96), Qi = _mm512_load_pd(qscr+104);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 8120, cb, 8120);
    MAPST(yr, yi, x, 18560, cb, 18560);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+128), Pi = _mm512_load_pd(pscr+136);
    __m512d Qr = _mm512_load_pd(qscr+112), Qi = _mm512_load_pd(qscr+120);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 9280, cb, 9280);
    MAPST(yr, yi, x, 17400, cb, 17400);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+144), Pi = _mm512_load_pd(pscr+152);
    __m512d Qr = _mm512_load_pd(qscr+128), Qi = _mm512_load_pd(qscr+136);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 10440, cb, 10440);
    MAPST(yr, yi, x, 16240, cb, 16240);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+160), Pi = _mm512_load_pd(pscr+168);
    __m512d Qr = _mm512_load_pd(qscr+144), Qi = _mm512_load_pd(qscr+152);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 11600, cb, 11600);
    MAPST(yr, yi, x, 15080, cb, 15080);
    }
    {
    __m512d Pr = _mm512_load_pd(pscr+176), Pi = _mm512_load_pd(pscr+184);
    __m512d Qr = _mm512_load_pd(qscr+160), Qi = _mm512_load_pd(qscr+168);
    __m512d xr = _mm512_add_pd(Pr, Qr), xi = _mm512_sub_pd(Pi, Qi);
    __m512d yr = _mm512_sub_pd(Pr, Qr), yi = _mm512_add_pd(Pi, Qi);
    MAPST(xr, xi, x, 12760, cb, 12760);
    MAPST(yr, yi, x, 13920, cb, 13920);
    }
    }
}

static void wtr_23(double* restrict P){
    for(int a=0;a<3;a++){
        {
            double* ta = P + (long)8*a*48 + a*16;
            for(int half=0; half<2; half++){
                double* pa = ta + 8*half;
                __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                r0=_mm512_load_pd(pa); r1=_mm512_load_pd(pa+48); r2=_mm512_load_pd(pa+2*48); r3=_mm512_load_pd(pa+3*48);
                r4=_mm512_load_pd(pa+4*48); r5=_mm512_load_pd(pa+5*48); r6=_mm512_load_pd(pa+6*48); r7=_mm512_load_pd(pa+7*48);
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                _mm512_store_pd(pa,o0); _mm512_store_pd(pa+48,o1); _mm512_store_pd(pa+2*48,o2); _mm512_store_pd(pa+3*48,o3);
                _mm512_store_pd(pa+4*48,o4); _mm512_store_pd(pa+5*48,o5); _mm512_store_pd(pa+6*48,o6); _mm512_store_pd(pa+7*48,o7);
            }
        }
        for(int b=a+1;b<3;b++){
            double* ta = P + (long)8*a*48 + b*16;
            double* tb = P + (long)8*b*48 + a*16;
            for(int half=0; half<2; half++){
                double* pa = ta + 8*half; double* pb = tb + 8*half;
                __m512d r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7;
                __m512d q0,q1,q2,q3,q4,q5,q6,q7,s0,s1,s2,s3,s4,s5,s6,s7;
                r0=_mm512_load_pd(pa); r1=_mm512_load_pd(pa+48); r2=_mm512_load_pd(pa+2*48); r3=_mm512_load_pd(pa+3*48);
                r4=_mm512_load_pd(pa+4*48); r5=_mm512_load_pd(pa+5*48); r6=_mm512_load_pd(pa+6*48); r7=_mm512_load_pd(pa+7*48);
                q0=_mm512_load_pd(pb); q1=_mm512_load_pd(pb+48); q2=_mm512_load_pd(pb+2*48); q3=_mm512_load_pd(pb+3*48);
                q4=_mm512_load_pd(pb+4*48); q5=_mm512_load_pd(pb+5*48); q6=_mm512_load_pd(pb+6*48); q7=_mm512_load_pd(pb+7*48);
                TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
                TR8(q0,q1,q2,q3,q4,q5,q6,q7,s0,s1,s2,s3,s4,s5,s6,s7);
                _mm512_store_pd(pb,o0); _mm512_store_pd(pb+48,o1); _mm512_store_pd(pb+2*48,o2); _mm512_store_pd(pb+3*48,o3);
                _mm512_store_pd(pb+4*48,o4); _mm512_store_pd(pb+5*48,o5); _mm512_store_pd(pb+6*48,o6); _mm512_store_pd(pb+7*48,o7);
                _mm512_store_pd(pa,s0); _mm512_store_pd(pa+48,s1); _mm512_store_pd(pa+2*48,s2); _mm512_store_pd(pa+3*48,s3);
                _mm512_store_pd(pa+4*48,s4); _mm512_store_pd(pa+5*48,s5); _mm512_store_pd(pa+6*48,s6); _mm512_store_pd(pa+7*48,s7);
            }
        }
    }
}
static void wstep_23(double* restrict V, const double* restrict CP){
    for(int x=0;x<23;x++){
        double* P = V + (long)x*1160;
        for(int g=0;g<3;g++) wz_23(P + g*16);
        wtr_23(P);
        for(int g=0;g<3;g++) wz_23(P + g*16);
    }
    for(int r=0;r<23;r++)
        for(int g=0;g<3;g++)
            wx_23(V + (long)r*48 + g*16, CP + (long)r*48 + g*16);
}
static void wconv_in_23(const double* restrict src, double* restrict V){
    for(int x=0;x<23;x++){
        for(int y=0;y<23;y++){
            const double* sp = src + ((long)x*23+y)*23*2;
            double* d = V + (long)x*1160 + (long)y*48;
            int z=0;
            for(; z+8<=23; z+=8){
                __m512d a = _mm512_loadu_pd(sp + 2*z);
                __m512d b = _mm512_loadu_pd(sp + 2*z + 8);
                __m512d re = _mm512_permutex2var_pd(a, IDX_RE, b);
                __m512d im = _mm512_permutex2var_pd(a, IDX_IM, b);
                _mm512_store_pd(d + (z/8)*16, re);
                _mm512_store_pd(d + (z/8)*16 + 8, im);
            }
            {
                double tre[8]={0,0,0,0,0,0,0,0}, tim[8]={0,0,0,0,0,0,0,0};
                for(int t=0; z+t<23; t++){ tre[t]=sp[2*(z+t)]; tim[t]=sp[2*(z+t)+1]; }
                _mm512_store_pd(d + (z/8)*16, _mm512_loadu_pd(tre));
                _mm512_store_pd(d + (z/8)*16 + 8, _mm512_loadu_pd(tim));
            }
        }
        for(int y=23; y<24; y++) memset(V + (long)x*1160 + (long)y*48, 0, 48*8);
        memset(V + (long)x*1160 + (long)24*48, 0, (1160-24*48)*8);
    }
}
static void wconv_out_23(const double* restrict V, double* restrict dst, int parity){
    for(int x=0;x<23;x++){
        const double* P = V + (long)x*1160;
        if(parity==0){
            for(int y=0;y<23;y++){
                double* d = dst + ((long)x*23+y)*23*2;
                const double* pp = P + (long)y*48;
                for(int z=0; z<23; z++){ d[2*z] = pp[(z/8)*16 + (z%8)]; d[2*z+1] = pp[(z/8)*16 + 8 + (z%8)]; }
            }
        } else {
            for(int z=0;z<23;z++){
                const double* pp = P + (long)z*48;
                for(int y=0;y<23;y++){
                    double* d = dst + (((long)x*23+y)*23 + z)*2;
                    d[0] = pp[(y/8)*16 + (y%8)];
                    d[1] = pp[(y/8)*16 + 8 + (y%8)];
                }
            }
        }
    }
}
static double* WV_23 = 0; static double* WC0_23 = 0; static double* WC1_23 = 0;
void run3_23(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if(!WV_23){ WV_23 = alloc_arena(26680*8+4096); WC0_23 = alloc_arena(26680*8+4096)+64; WC1_23 = alloc_arena(26680*8+4096)+128; }
    for(long b=0;b<B;b++){
        long off = b*(long)12167*2;
        wconv_in_23(x0 + off, WV_23);
        wconv_in_23(c + off, WC0_23);
        memcpy(WC1_23, WC0_23, (long)26680*8);
        for(int x=0;x<23;x++) wtr_23(WC1_23 + (long)x*1160);
        for(long t=0;t<m;t++){
            wstep_23(WV_23, (t%2==0)? WC1_23 : WC0_23);
            if(t==0 && m>1) wconv_out_23(WV_23, out1 + off, 1);
        }
        wconv_out_23(WV_23, outm + off, (int)(m%2));
        if(m==1) memcpy(out1 + off, outm + off, (long)12167*16);
    }
}
