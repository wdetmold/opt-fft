// L=64 engine: iterated z = FFT3(x)+c ; x <- z/(1+|z|)
// Storage: per volume: planes i=0..63 (digit-swapped x), rows y=0..63 (ds y),
// row = 8 slots of 16 doubles [8 re | 8 im], z digit-swapped across (slot,lane):
//   slot s lane l holds z index 8*l+s  (i.e. a zmm load of slot s = z's {8l+s}).
// All three axes stored in digit-swap order; conv in/out applies it.
#include <immintrin.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <sys/mman.h>

#define RS64 144
#define PS64 9232
#define ALIGN64 __attribute__((aligned(64)))

static double *X64, *CS64, *CP64;   // state, c slab-layout, c pencil-layout
static double SCRA[64*16] ALIGN64;  // stage scratch
static double SCRB[64*16] ALIGN64;

// twiddle tables: w64^(j2*k1), k1=1..7, per j2=1..7 : [j2][k1][c,s] ; cos/sin of -2pi*j2*k1/64
static double TW64[8][7][2] ALIGN64;

static const double S8C = 0.70710678118654752440084436210484903929; // sqrt(2)/2

// ---------------- DFT8 macro (forward, w8 = e^{-2pi i/8}) ----------------
// in: x0r..x7r / x0i..x7i ; out: y0..y7 (may alias new vars)
#define DFT8(S, xr0,xi0,xr1,xi1,xr2,xi2,xr3,xi3,xr4,xi4,xr5,xi5,xr6,xi6,xr7,xi7, \
                yr0,yi0,yr1,yi1,yr2,yi2,yr3,yi3,yr4,yi4,yr5,yi5,yr6,yi6,yr7,yi7) do{ \
    __m512d t0r=_mm512_add_pd(xr0,xr4), t0i=_mm512_add_pd(xi0,xi4); \
    __m512d t1r=_mm512_sub_pd(xr0,xr4), t1i=_mm512_sub_pd(xi0,xi4); \
    __m512d t2r=_mm512_add_pd(xr2,xr6), t2i=_mm512_add_pd(xi2,xi6); \
    __m512d t3r=_mm512_sub_pd(xr2,xr6), t3i=_mm512_sub_pd(xi2,xi6); \
    __m512d u0r=_mm512_add_pd(xr1,xr5), u0i=_mm512_add_pd(xi1,xi5); \
    __m512d u1r=_mm512_sub_pd(xr1,xr5), u1i=_mm512_sub_pd(xi1,xi5); \
    __m512d u2r=_mm512_add_pd(xr3,xr7), u2i=_mm512_add_pd(xi3,xi7); \
    __m512d u3r=_mm512_sub_pd(xr3,xr7), u3i=_mm512_sub_pd(xi3,xi7); \
    __m512d E0r=_mm512_add_pd(t0r,t2r), E0i=_mm512_add_pd(t0i,t2i); \
    __m512d E2r=_mm512_sub_pd(t0r,t2r), E2i=_mm512_sub_pd(t0i,t2i); \
    __m512d E1r=_mm512_add_pd(t1r,t3i), E1i=_mm512_sub_pd(t1i,t3r); \
    __m512d E3r=_mm512_sub_pd(t1r,t3i), E3i=_mm512_add_pd(t1i,t3r); \
    __m512d O0r=_mm512_add_pd(u0r,u2r), O0i=_mm512_add_pd(u0i,u2i); \
    __m512d O2r=_mm512_sub_pd(u0r,u2r), O2i=_mm512_sub_pd(u0i,u2i); \
    __m512d O1r=_mm512_add_pd(u1r,u3i), O1i=_mm512_sub_pd(u1i,u3r); \
    __m512d O3r=_mm512_sub_pd(u1r,u3i), O3i=_mm512_add_pd(u1i,u3r); \
    yr0=_mm512_add_pd(E0r,O0r); yi0=_mm512_add_pd(E0i,O0i); \
    yr4=_mm512_sub_pd(E0r,O0r); yi4=_mm512_sub_pd(E0i,O0i); \
    yr2=_mm512_add_pd(E2r,O2i); yi2=_mm512_sub_pd(E2i,O2r); \
    yr6=_mm512_sub_pd(E2r,O2i); yi6=_mm512_add_pd(E2i,O2r); \
    { __m512d p1r=_mm512_add_pd(O1r,O1i), p1i=_mm512_sub_pd(O1i,O1r); \
      yr1=_mm512_fmadd_pd(S,p1r,E1r); yi1=_mm512_fmadd_pd(S,p1i,E1i); \
      yr5=_mm512_fnmadd_pd(S,p1r,E1r); yi5=_mm512_fnmadd_pd(S,p1i,E1i); } \
    { __m512d p3r=_mm512_sub_pd(O3i,O3r), p3i=_mm512_add_pd(O3r,O3i); \
      yr3=_mm512_fmadd_pd(S,p3r,E3r); yi3=_mm512_fnmadd_pd(S,p3i,E3i); \
      yr7=_mm512_fnmadd_pd(S,p3r,E3r); yi7=_mm512_fmadd_pd(S,p3i,E3i); } \
}while(0)

// 8x8 transpose of doubles in r0..r7 (in place via temps)
#define TR8(i0,i1,i2,i3,i4,i5,i6,i7) do{ \
    __m512d _t0=_mm512_unpacklo_pd(i0,i1), _t1=_mm512_unpackhi_pd(i0,i1); \
    __m512d _t2=_mm512_unpacklo_pd(i2,i3), _t3=_mm512_unpackhi_pd(i2,i3); \
    __m512d _t4=_mm512_unpacklo_pd(i4,i5), _t5=_mm512_unpackhi_pd(i4,i5); \
    __m512d _t6=_mm512_unpacklo_pd(i6,i7), _t7=_mm512_unpackhi_pd(i6,i7); \
    __m512d _u0=_mm512_shuffle_f64x2(_t0,_t2,0x44), _u1=_mm512_shuffle_f64x2(_t4,_t6,0x44); \
    __m512d _u2=_mm512_shuffle_f64x2(_t0,_t2,0xee), _u3=_mm512_shuffle_f64x2(_t4,_t6,0xee); \
    __m512d _u4=_mm512_shuffle_f64x2(_t1,_t3,0x44), _u5=_mm512_shuffle_f64x2(_t5,_t7,0x44); \
    __m512d _u6=_mm512_shuffle_f64x2(_t1,_t3,0xee), _u7=_mm512_shuffle_f64x2(_t5,_t7,0xee); \
    i0=_mm512_shuffle_f64x2(_u0,_u1,0x88); i2=_mm512_shuffle_f64x2(_u0,_u1,0xdd); \
    i4=_mm512_shuffle_f64x2(_u2,_u3,0x88); i6=_mm512_shuffle_f64x2(_u2,_u3,0xdd); \
    i1=_mm512_shuffle_f64x2(_u4,_u5,0x88); i3=_mm512_shuffle_f64x2(_u4,_u5,0xdd); \
    i5=_mm512_shuffle_f64x2(_u6,_u7,0x88); i7=_mm512_shuffle_f64x2(_u6,_u7,0xdd); \
}while(0)

// complex twiddle multiply by (c,s): (r,i) -> (r*c - i*s, i*c + r*s) with broadcast operands
#define CTW(rr, ii, cval, sval) do{ \
    __m512d _c=_mm512_set1_pd(cval), _s=_mm512_set1_pd(sval); \
    __m512d _tr=_mm512_mul_pd(rr,_c); \
    __m512d _ti=_mm512_mul_pd(ii,_c); \
    _tr=_mm512_fnmadd_pd(_s,ii,_tr); \
    _ti=_mm512_fmadd_pd(_s,rr,_ti); \
    rr=_tr; ii=_ti; \
}while(0)

// ---- map: x = z/(1+|z|), z=(zr,zi); full double precision ----
static const double MAPK[8] ALIGN64 = {1e-30, 1.0, 0.5, 0,0,0,0,0};
#if MAPV == 3
#define MAP2(zr, zi) do{ \
    __m512d TINY=_mm512_set1_pd(1e-30), VONE=_mm512_set1_pd(1.0); \
    __m512d m  = _mm512_fmadd_pd(zr, zr, _mm512_fmadd_pd(zi, zi, TINY)); \
    __m512d u  = _mm512_add_pd(VONE, _mm512_sqrt_pd(m)); \
    __m512d w0 = _mm512_rcp14_pd(u); \
    __m512d e3 = _mm512_fnmadd_pd(u, w0, VONE); \
    __m512d a  = _mm512_fmadd_pd(w0, e3, w0); \
    __m512d ee = _mm512_mul_pd(e3, e3); \
    __m512d w2 = _mm512_fmadd_pd(a, ee, a); \
    zr = _mm512_mul_pd(zr, w2); \
    zi = _mm512_mul_pd(zi, w2); \
}while(0)
#elif MAPV == 2
#define MAP2(zr, zi) do{ \
    __m512d TINY=_mm512_set1_pd(1e-30), VONE=_mm512_set1_pd(1.0); \
    __m512d m  = _mm512_fmadd_pd(zr, zr, _mm512_fmadd_pd(zi, zi, TINY)); \
    __m512d w = _mm512_div_pd(VONE, _mm512_add_pd(VONE, _mm512_sqrt_pd(m))); \
    zr = _mm512_mul_pd(zr, w); \
    zi = _mm512_mul_pd(zi, w); \
}while(0)
#elif MAPV == 1
#define MAP2(zr, zi) do{ \
    __m512d TINY=_mm512_set1_pd(1e-30), VONE=_mm512_set1_pd(1.0), VHALF=_mm512_set1_pd(0.5); \
    __m512d m  = _mm512_fmadd_pd(zr, zr, _mm512_fmadd_pd(zi, zi, TINY)); \
    __m512d r0 = _mm512_rsqrt14_pd(m); \
    __m512d t  = _mm512_mul_pd(m, r0); \
    __m512d hr = _mm512_mul_pd(r0, VHALF); \
    __m512d eh = _mm512_fnmadd_pd(t, hr, VHALF); \
    __m512d r1 = _mm512_fmadd_pd(r0, eh, r0); \
    __m512d mg0= _mm512_mul_pd(m, r1); \
    __m512d hr1= _mm512_mul_pd(r1, VHALF); \
    __m512d e2 = _mm512_fnmadd_pd(mg0, mg0, m); \
    __m512d mag= _mm512_fmadd_pd(e2, hr1, mg0); \
    __m512d u  = _mm512_add_pd(VONE, mag); \
    __m512d w0 = _mm512_rcp14_pd(u); \
    __m512d e3 = _mm512_fnmadd_pd(u, w0, VONE); \
    __m512d a  = _mm512_fmadd_pd(w0, e3, w0); \
    __m512d ee = _mm512_mul_pd(e3, e3); \
    __m512d w2 = _mm512_fmadd_pd(a, ee, a); \
    zr = _mm512_mul_pd(zr, w2); \
    zi = _mm512_mul_pd(zi, w2); \
}while(0)
#else
#define MAP2(zr, zi) do{ \
    __m512d TINY=_mm512_set1_pd(1e-30), VONE=_mm512_set1_pd(1.0), VHALF=_mm512_set1_pd(0.5), V15=_mm512_set1_pd(1.5); \
    __m512d m  = _mm512_fmadd_pd(zr, zr, _mm512_fmadd_pd(zi, zi, TINY)); \
    __m512d r0 = _mm512_cvtps_pd(_mm256_rsqrt_ps(_mm512_cvtpd_ps(m))); \
    __m512d h  = _mm512_mul_pd(m, VHALF); \
    __m512d e1 = _mm512_fnmadd_pd(_mm512_mul_pd(r0,r0), h, V15); \
    __m512d r1 = _mm512_mul_pd(r0, e1); \
    __m512d e2 = _mm512_fnmadd_pd(_mm512_mul_pd(r1,r1), h, V15); \
    __m512d r2 = _mm512_mul_pd(r1, e2); \
    __m512d mg0= _mm512_mul_pd(m, r2); \
    __m512d er = _mm512_fnmadd_pd(mg0, mg0, m); \
    __m512d mag= _mm512_fmadd_pd(er, _mm512_mul_pd(r2,VHALF), mg0); \
    __m512d u  = _mm512_add_pd(VONE, mag); \
    __m512d w0 = _mm512_cvtps_pd(_mm256_rcp_ps(_mm512_cvtpd_ps(u))); \
    __m512d e3 = _mm512_fnmadd_pd(u, w0, VONE); \
    __m512d a  = _mm512_fmadd_pd(w0, e3, w0); \
    __m512d w2 = _mm512_fmadd_pd(a, _mm512_mul_pd(e3,e3), a); \
    zr = _mm512_mul_pd(zr, w2); \
    zi = _mm512_mul_pd(zi, w2); \
}while(0)
#endif

#define LOAD16(base, st, pr, pi) \
    pr##0=_mm512_load_pd((base)+0*(st)); pi##0=_mm512_load_pd((base)+0*(st)+8); \
    pr##1=_mm512_load_pd((base)+1*(st)); pi##1=_mm512_load_pd((base)+1*(st)+8); \
    pr##2=_mm512_load_pd((base)+2*(st)); pi##2=_mm512_load_pd((base)+2*(st)+8); \
    pr##3=_mm512_load_pd((base)+3*(st)); pi##3=_mm512_load_pd((base)+3*(st)+8); \
    pr##4=_mm512_load_pd((base)+4*(st)); pi##4=_mm512_load_pd((base)+4*(st)+8); \
    pr##5=_mm512_load_pd((base)+5*(st)); pi##5=_mm512_load_pd((base)+5*(st)+8); \
    pr##6=_mm512_load_pd((base)+6*(st)); pi##6=_mm512_load_pd((base)+6*(st)+8); \
    pr##7=_mm512_load_pd((base)+7*(st)); pi##7=_mm512_load_pd((base)+7*(st)+8);

#define STORE16(base, st, pr, pi) \
    _mm512_store_pd((base)+0*(st), pr##0); _mm512_store_pd((base)+0*(st)+8, pi##0); \
    _mm512_store_pd((base)+1*(st), pr##1); _mm512_store_pd((base)+1*(st)+8, pi##1); \
    _mm512_store_pd((base)+2*(st), pr##2); _mm512_store_pd((base)+2*(st)+8, pi##2); \
    _mm512_store_pd((base)+3*(st), pr##3); _mm512_store_pd((base)+3*(st)+8, pi##3); \
    _mm512_store_pd((base)+4*(st), pr##4); _mm512_store_pd((base)+4*(st)+8, pi##4); \
    _mm512_store_pd((base)+5*(st), pr##5); _mm512_store_pd((base)+5*(st)+8, pi##5); \
    _mm512_store_pd((base)+6*(st), pr##6); _mm512_store_pd((base)+6*(st)+8, pi##6); \
    _mm512_store_pd((base)+7*(st), pr##7); _mm512_store_pd((base)+7*(st)+8, pi##7);

#define DECL16(pr,pi) __m512d pr##0,pi##0,pr##1,pi##1,pr##2,pi##2,pr##3,pi##3,pr##4,pi##4,pr##5,pi##5,pr##6,pi##6,pr##7,pi##7;

// twiddle by TW64[g][k-1] applied to regs a1..a7 (a0 untouched)
#define TW7(g, ar, ai) do{ const double (*_tw)[2] = TW64[g]; \
    CTW(ar##1, ai##1, _tw[0][0], _tw[0][1]); \
    CTW(ar##2, ai##2, _tw[1][0], _tw[1][1]); \
    CTW(ar##3, ai##3, _tw[2][0], _tw[2][1]); \
    CTW(ar##4, ai##4, _tw[3][0], _tw[3][1]); \
    CTW(ar##5, ai##5, _tw[4][0], _tw[4][1]); \
    CTW(ar##6, ai##6, _tw[5][0], _tw[5][1]); \
    CTW(ar##7, ai##7, _tw[6][0], _tw[6][1]); \
}while(0)

// ============================ column codelets =============================
// generic strided column: 64 slots at stride st (doubles). Used for X (st=PS64)
// and Y (st=RS64) axes.
// S1: groups j2=0..7: read slots 8*j2..8*j2+7 -> DFT8 -> twiddle(j2) -> SCRA[k1*8+j2]
// S2: groups k1=0..7: read SCRA[k1*8+..] -> DFT8 -> write slots 8*k1..8*k1+7

__attribute__((noinline))
static void col_s1(const double* src, long st){
    const __m512d S=_mm512_set1_pd(S8C);
    for(int g=0; g<8; g++){
        const double* p = src + (long)g*8*st;
        DECL16(xr,xi)
        LOAD16(p, st, xr, xi)
        DECL16(yr,yi)
        DFT8(S, xr0,xi0,xr1,xi1,xr2,xi2,xr3,xi3,xr4,xi4,xr5,xi5,xr6,xi6,xr7,xi7,
                yr0,yi0,yr1,yi1,yr2,yi2,yr3,yi3,yr4,yi4,yr5,yi5,yr6,yi6,yr7,yi7);
        if(g){ TW7(g, yr, yi); }
        double* q = SCRA + g*16;
        _mm512_store_pd(q+0*128, yr0); _mm512_store_pd(q+0*128+8, yi0);
        _mm512_store_pd(q+1*128, yr1); _mm512_store_pd(q+1*128+8, yi1);
        _mm512_store_pd(q+2*128, yr2); _mm512_store_pd(q+2*128+8, yi2);
        _mm512_store_pd(q+3*128, yr3); _mm512_store_pd(q+3*128+8, yi3);
        _mm512_store_pd(q+4*128, yr4); _mm512_store_pd(q+4*128+8, yi4);
        _mm512_store_pd(q+5*128, yr5); _mm512_store_pd(q+5*128+8, yi5);
        _mm512_store_pd(q+6*128, yr6); _mm512_store_pd(q+6*128+8, yi6);
        _mm512_store_pd(q+7*128, yr7); _mm512_store_pd(q+7*128+8, yi7);
    }
}

__attribute__((noinline))
static void col_s1_pf(const double* src, long st, const char* pf, long pfst){
    const __m512d S=_mm512_set1_pd(S8C);
    for(int g=0; g<8; g++){
        const double* p = src + (long)g*8*st;
        { const char* q = pf + (long)g*8*pfst;
          _mm_prefetch(q, _MM_HINT_T0); _mm_prefetch(q+64, _MM_HINT_T0);
          _mm_prefetch(q+pfst, _MM_HINT_T0); _mm_prefetch(q+pfst+64, _MM_HINT_T0);
          _mm_prefetch(q+2*pfst, _MM_HINT_T0); _mm_prefetch(q+2*pfst+64, _MM_HINT_T0);
          _mm_prefetch(q+3*pfst, _MM_HINT_T0); _mm_prefetch(q+3*pfst+64, _MM_HINT_T0);
          _mm_prefetch(q+4*pfst, _MM_HINT_T0); _mm_prefetch(q+4*pfst+64, _MM_HINT_T0);
          _mm_prefetch(q+5*pfst, _MM_HINT_T0); _mm_prefetch(q+5*pfst+64, _MM_HINT_T0);
          _mm_prefetch(q+6*pfst, _MM_HINT_T0); _mm_prefetch(q+6*pfst+64, _MM_HINT_T0);
          _mm_prefetch(q+7*pfst, _MM_HINT_T0); _mm_prefetch(q+7*pfst+64, _MM_HINT_T0); }
        DECL16(xr,xi)
        LOAD16(p, st, xr, xi)
        DECL16(yr,yi)
        DFT8(S, xr0,xi0,xr1,xi1,xr2,xi2,xr3,xi3,xr4,xi4,xr5,xi5,xr6,xi6,xr7,xi7,
                yr0,yi0,yr1,yi1,yr2,yi2,yr3,yi3,yr4,yi4,yr5,yi5,yr6,yi6,yr7,yi7);
        if(g){ TW7(g, yr, yi); }
        double* q = SCRA + g*16;
        _mm512_store_pd(q+0*128, yr0); _mm512_store_pd(q+0*128+8, yi0);
        _mm512_store_pd(q+1*128, yr1); _mm512_store_pd(q+1*128+8, yi1);
        _mm512_store_pd(q+2*128, yr2); _mm512_store_pd(q+2*128+8, yi2);
        _mm512_store_pd(q+3*128, yr3); _mm512_store_pd(q+3*128+8, yi3);
        _mm512_store_pd(q+4*128, yr4); _mm512_store_pd(q+4*128+8, yi4);
        _mm512_store_pd(q+5*128, yr5); _mm512_store_pd(q+5*128+8, yi5);
        _mm512_store_pd(q+6*128, yr6); _mm512_store_pd(q+6*128+8, yi6);
        _mm512_store_pd(q+7*128, yr7); _mm512_store_pd(q+7*128+8, yi7);
    }
}
// S2 plain: write to dst slots
__attribute__((noinline))
static void col_s2(double* dst, long st){
    const __m512d S=_mm512_set1_pd(S8C);
    for(int k1=0; k1<8; k1++){
        const double* p = SCRA + k1*128;
        DECL16(xr,xi)
        LOAD16(p, 16, xr, xi)
        DECL16(yr,yi)
        DFT8(S, xr0,xi0,xr1,xi1,xr2,xi2,xr3,xi3,xr4,xi4,xr5,xi5,xr6,xi6,xr7,xi7,
                yr0,yi0,yr1,yi1,yr2,yi2,yr3,yi3,yr4,yi4,yr5,yi5,yr6,yi6,yr7,yi7);
        double* q = dst + (long)k1*8*st;
        STORE16(q, st, yr, yi)
    }
}
// S2 with +c,map, then write plain to dst slots. c: 16-double blocks in k1-major
// order: c + k1*128 + k2*16.
__attribute__((noinline))
static void col_s2_map(double* dst, long st, const double* c){
    const __m512d S=_mm512_set1_pd(S8C);
    for(int k1=0; k1<8; k1++){
        const double* p = SCRA + k1*128;
        DECL16(xr,xi)
        LOAD16(p, 16, xr, xi)
        DECL16(yr,yi)
        DFT8(S, xr0,xi0,xr1,xi1,xr2,xi2,xr3,xi3,xr4,xi4,xr5,xi5,xr6,xi6,xr7,xi7,
                yr0,yi0,yr1,yi1,yr2,yi2,yr3,yi3,yr4,yi4,yr5,yi5,yr6,yi6,yr7,yi7);
        const double* cc = c + k1*128;
        yr0=_mm512_add_pd(yr0,_mm512_load_pd(cc+0));   yi0=_mm512_add_pd(yi0,_mm512_load_pd(cc+8));
        yr1=_mm512_add_pd(yr1,_mm512_load_pd(cc+16));  yi1=_mm512_add_pd(yi1,_mm512_load_pd(cc+24));
        yr2=_mm512_add_pd(yr2,_mm512_load_pd(cc+32));  yi2=_mm512_add_pd(yi2,_mm512_load_pd(cc+40));
        yr3=_mm512_add_pd(yr3,_mm512_load_pd(cc+48));  yi3=_mm512_add_pd(yi3,_mm512_load_pd(cc+56));
        yr4=_mm512_add_pd(yr4,_mm512_load_pd(cc+64));  yi4=_mm512_add_pd(yi4,_mm512_load_pd(cc+72));
        yr5=_mm512_add_pd(yr5,_mm512_load_pd(cc+80));  yi5=_mm512_add_pd(yi5,_mm512_load_pd(cc+88));
        yr6=_mm512_add_pd(yr6,_mm512_load_pd(cc+96));  yi6=_mm512_add_pd(yi6,_mm512_load_pd(cc+104));
        yr7=_mm512_add_pd(yr7,_mm512_load_pd(cc+112)); yi7=_mm512_add_pd(yi7,_mm512_load_pd(cc+120));
        MAP2(yr0,yi0); MAP2(yr1,yi1); MAP2(yr2,yi2); MAP2(yr3,yi3);
        MAP2(yr4,yi4); MAP2(yr5,yi5); MAP2(yr6,yi6); MAP2(yr7,yi7);
        double* q = dst + (long)k1*8*st;
        STORE16(q, st, yr, yi)
    }
}
// S2 with +c,map, then fused next-step S1 (group j2==k1): DFT8 + twiddle -> SCRB
__attribute__((noinline))
static void col_s2_map_s1(const double* c, const char* pfc){
    const __m512d S=_mm512_set1_pd(S8C);
    for(int k1=0; k1<8; k1++){
        { const char* q2 = pfc + (long)k1*1024;
          _mm_prefetch(q2, _MM_HINT_T1); _mm_prefetch(q2+64, _MM_HINT_T1); _mm_prefetch(q2+128, _MM_HINT_T1);
          _mm_prefetch(q2+192, _MM_HINT_T1); _mm_prefetch(q2+256, _MM_HINT_T1); _mm_prefetch(q2+320, _MM_HINT_T1);
          _mm_prefetch(q2+384, _MM_HINT_T1); _mm_prefetch(q2+448, _MM_HINT_T1); _mm_prefetch(q2+512, _MM_HINT_T1);
          _mm_prefetch(q2+576, _MM_HINT_T1); _mm_prefetch(q2+640, _MM_HINT_T1); _mm_prefetch(q2+704, _MM_HINT_T1);
          _mm_prefetch(q2+768, _MM_HINT_T1); _mm_prefetch(q2+832, _MM_HINT_T1); _mm_prefetch(q2+896, _MM_HINT_T1);
          _mm_prefetch(q2+960, _MM_HINT_T1); }
        const double* p = SCRA + k1*128;
        DECL16(xr,xi)
        LOAD16(p, 16, xr, xi)
        DECL16(yr,yi)
        DFT8(S, xr0,xi0,xr1,xi1,xr2,xi2,xr3,xi3,xr4,xi4,xr5,xi5,xr6,xi6,xr7,xi7,
                yr0,yi0,yr1,yi1,yr2,yi2,yr3,yi3,yr4,yi4,yr5,yi5,yr6,yi6,yr7,yi7);
        const double* cc = c + k1*128;
        yr0=_mm512_add_pd(yr0,_mm512_load_pd(cc+0));   yi0=_mm512_add_pd(yi0,_mm512_load_pd(cc+8));
        yr1=_mm512_add_pd(yr1,_mm512_load_pd(cc+16));  yi1=_mm512_add_pd(yi1,_mm512_load_pd(cc+24));
        yr2=_mm512_add_pd(yr2,_mm512_load_pd(cc+32));  yi2=_mm512_add_pd(yi2,_mm512_load_pd(cc+40));
        yr3=_mm512_add_pd(yr3,_mm512_load_pd(cc+48));  yi3=_mm512_add_pd(yi3,_mm512_load_pd(cc+56));
        yr4=_mm512_add_pd(yr4,_mm512_load_pd(cc+64));  yi4=_mm512_add_pd(yi4,_mm512_load_pd(cc+72));
        yr5=_mm512_add_pd(yr5,_mm512_load_pd(cc+80));  yi5=_mm512_add_pd(yi5,_mm512_load_pd(cc+88));
        yr6=_mm512_add_pd(yr6,_mm512_load_pd(cc+96));  yi6=_mm512_add_pd(yi6,_mm512_load_pd(cc+104));
        yr7=_mm512_add_pd(yr7,_mm512_load_pd(cc+112)); yi7=_mm512_add_pd(yi7,_mm512_load_pd(cc+120));
        MAP2(yr0,yi0); MAP2(yr1,yi1); MAP2(yr2,yi2); MAP2(yr3,yi3);
        MAP2(yr4,yi4); MAP2(yr5,yi5); MAP2(yr6,yi6); MAP2(yr7,yi7);
        // next step S1, group j2=k1: values yr/yi are inputs {j = k1 + 8*k2}, DFT8 over k2
        DECL16(zr,zi)
        DFT8(S, yr0,yi0,yr1,yi1,yr2,yi2,yr3,yi3,yr4,yi4,yr5,yi5,yr6,yi6,yr7,yi7,
                zr0,zi0,zr1,zi1,zr2,zi2,zr3,zi3,zr4,zi4,zr5,zi5,zr6,zi6,zr7,zi7);
        if(k1){ TW7(k1, zr, zi); }
        double* q = SCRB + k1*16;
        _mm512_store_pd(q+0*128, zr0); _mm512_store_pd(q+0*128+8, zi0);
        _mm512_store_pd(q+1*128, zr1); _mm512_store_pd(q+1*128+8, zi1);
        _mm512_store_pd(q+2*128, zr2); _mm512_store_pd(q+2*128+8, zi2);
        _mm512_store_pd(q+3*128, zr3); _mm512_store_pd(q+3*128+8, zi3);
        _mm512_store_pd(q+4*128, zr4); _mm512_store_pd(q+4*128+8, zi4);
        _mm512_store_pd(q+5*128, zr5); _mm512_store_pd(q+5*128+8, zi5);
        _mm512_store_pd(q+6*128, zr6); _mm512_store_pd(q+6*128+8, zi6);
        _mm512_store_pd(q+7*128, zr7); _mm512_store_pd(q+7*128+8, zi7);
    }
}
// S2 from SCRB
__attribute__((noinline))
static void col_s2b(double* dst, long st){
    const __m512d S=_mm512_set1_pd(S8C);
    for(int k1=0; k1<8; k1++){
        const double* p = SCRB + k1*128;
        DECL16(xr,xi)
        LOAD16(p, 16, xr, xi)
        DECL16(yr,yi)
        DFT8(S, xr0,xi0,xr1,xi1,xr2,xi2,xr3,xi3,xr4,xi4,xr5,xi5,xr6,xi6,xr7,xi7,
                yr0,yi0,yr1,yi1,yr2,yi2,yr3,yi3,yr4,yi4,yr5,yi5,yr6,yi6,yr7,yi7);
        double* q = dst + (long)k1*8*st;
        STORE16(q, st, yr, yi)
    }
}

// ---------------- z axis (row-block of 8 rows, TR8-fused) ----------------
// S1z: for tile v=0..7: load slot v of 8 rows (base rb + r*RS64 + v*16),
// transpose re/im tiles -> regs hold z-group {8*j1+v} across y-lanes,
// DFT8, twiddle(v), store SCRA[k1*8+v].
__attribute__((noinline))
static void zcol_s1(const double* rb){
    const __m512d S=_mm512_set1_pd(S8C);
    for(int v=0; v<8; v++){
        const double* p = rb + v*16;
        DECL16(xr,xi)
        LOAD16(p, RS64, xr, xi)
        TR8(xr0,xr1,xr2,xr3,xr4,xr5,xr6,xr7);
        TR8(xi0,xi1,xi2,xi3,xi4,xi5,xi6,xi7);
        DECL16(yr,yi)
        DFT8(S, xr0,xi0,xr1,xi1,xr2,xi2,xr3,xi3,xr4,xi4,xr5,xi5,xr6,xi6,xr7,xi7,
                yr0,yi0,yr1,yi1,yr2,yi2,yr3,yi3,yr4,yi4,yr5,yi5,yr6,yi6,yr7,yi7);
        if(v){ TW7(v, yr, yi); }
        double* q = SCRA + v*16;
        _mm512_store_pd(q+0*128, yr0); _mm512_store_pd(q+0*128+8, yi0);
        _mm512_store_pd(q+1*128, yr1); _mm512_store_pd(q+1*128+8, yi1);
        _mm512_store_pd(q+2*128, yr2); _mm512_store_pd(q+2*128+8, yi2);
        _mm512_store_pd(q+3*128, yr3); _mm512_store_pd(q+3*128+8, yi3);
        _mm512_store_pd(q+4*128, yr4); _mm512_store_pd(q+4*128+8, yi4);
        _mm512_store_pd(q+5*128, yr5); _mm512_store_pd(q+5*128+8, yi5);
        _mm512_store_pd(q+6*128, yr6); _mm512_store_pd(q+6*128+8, yi6);
        _mm512_store_pd(q+7*128, yr7); _mm512_store_pd(q+7*128+8, yi7);
    }
}
// S2z from SCRB (post fused-map stage): DFT8, transpose back, store slot k1 of 8 rows
__attribute__((noinline))
static void zcol_s2b(double* rb){
    const __m512d S=_mm512_set1_pd(S8C);
    for(int k1=0; k1<8; k1++){
        const double* p = SCRB + k1*128;
        DECL16(xr,xi)
        LOAD16(p, 16, xr, xi)
        DECL16(yr,yi)
        DFT8(S, xr0,xi0,xr1,xi1,xr2,xi2,xr3,xi3,xr4,xi4,xr5,xi5,xr6,xi6,xr7,xi7,
                yr0,yi0,yr1,yi1,yr2,yi2,yr3,yi3,yr4,yi4,yr5,yi5,yr6,yi6,yr7,yi7);
        TR8(yr0,yr1,yr2,yr3,yr4,yr5,yr6,yr7);
        TR8(yi0,yi1,yi2,yi3,yi4,yi5,yi6,yi7);
        double* q = rb + k1*16;
        STORE16(q, RS64, yr, yi)
    }
}
// S2z plain from SCRA (no map): for SB0 / final-z...
__attribute__((noinline))
static void zcol_s2a(double* rb){
    const __m512d S=_mm512_set1_pd(S8C);
    for(int k1=0; k1<8; k1++){
        const double* p = SCRA + k1*128;
        DECL16(xr,xi)
        LOAD16(p, 16, xr, xi)
        DECL16(yr,yi)
        DFT8(S, xr0,xi0,xr1,xi1,xr2,xi2,xr3,xi3,xr4,xi4,xr5,xi5,xr6,xi6,xr7,xi7,
                yr0,yi0,yr1,yi1,yr2,yi2,yr3,yi3,yr4,yi4,yr5,yi5,yr6,yi6,yr7,yi7);
        TR8(yr0,yr1,yr2,yr3,yr4,yr5,yr6,yr7);
        TR8(yi0,yi1,yi2,yi3,yi4,yi5,yi6,yi7);
        double* q = rb + k1*16;
        STORE16(q, RS64, yr, yi)
    }
}
// S2z with +c, map, store plain rows (final step in SB)
__attribute__((noinline))
static void zcol_s2a_map(double* rb, const double* c){
    const __m512d S=_mm512_set1_pd(S8C);
    for(int k1=0; k1<8; k1++){
        const double* p = SCRA + k1*128;
        DECL16(xr,xi)
        LOAD16(p, 16, xr, xi)
        DECL16(yr,yi)
        DFT8(S, xr0,xi0,xr1,xi1,xr2,xi2,xr3,xi3,xr4,xi4,xr5,xi5,xr6,xi6,xr7,xi7,
                yr0,yi0,yr1,yi1,yr2,yi2,yr3,yi3,yr4,yi4,yr5,yi5,yr6,yi6,yr7,yi7);
        const double* cc = c + k1*128;
        yr0=_mm512_add_pd(yr0,_mm512_load_pd(cc+0));   yi0=_mm512_add_pd(yi0,_mm512_load_pd(cc+8));
        yr1=_mm512_add_pd(yr1,_mm512_load_pd(cc+16));  yi1=_mm512_add_pd(yi1,_mm512_load_pd(cc+24));
        yr2=_mm512_add_pd(yr2,_mm512_load_pd(cc+32));  yi2=_mm512_add_pd(yi2,_mm512_load_pd(cc+40));
        yr3=_mm512_add_pd(yr3,_mm512_load_pd(cc+48));  yi3=_mm512_add_pd(yi3,_mm512_load_pd(cc+56));
        yr4=_mm512_add_pd(yr4,_mm512_load_pd(cc+64));  yi4=_mm512_add_pd(yi4,_mm512_load_pd(cc+72));
        yr5=_mm512_add_pd(yr5,_mm512_load_pd(cc+80));  yi5=_mm512_add_pd(yi5,_mm512_load_pd(cc+88));
        yr6=_mm512_add_pd(yr6,_mm512_load_pd(cc+96));  yi6=_mm512_add_pd(yi6,_mm512_load_pd(cc+104));
        yr7=_mm512_add_pd(yr7,_mm512_load_pd(cc+112)); yi7=_mm512_add_pd(yi7,_mm512_load_pd(cc+120));
        MAP2(yr0,yi0); MAP2(yr1,yi1); MAP2(yr2,yi2); MAP2(yr3,yi3);
        MAP2(yr4,yi4); MAP2(yr5,yi5); MAP2(yr6,yi6); MAP2(yr7,yi7);
        TR8(yr0,yr1,yr2,yr3,yr4,yr5,yr6,yr7);
        TR8(yi0,yi1,yi2,yi3,yi4,yi5,yi6,yi7);
        double* q = rb + k1*16;
        STORE16(q, RS64, yr, yi)
    }
}
// S2z +c, map, fused next S1 -> SCRB (steady state)
__attribute__((noinline))
static void zcol_s2a_map_s1(const double* c, const char* pf1, const char* pf2){
    const __m512d S=_mm512_set1_pd(S8C);
    for(int k1=0; k1<8; k1++){
        { const char* q1 = pf1 + (long)k1*1152; 
          _mm_prefetch(q1, _MM_HINT_T1); _mm_prefetch(q1+64, _MM_HINT_T1); _mm_prefetch(q1+128, _MM_HINT_T1);
          _mm_prefetch(q1+192, _MM_HINT_T1); _mm_prefetch(q1+256, _MM_HINT_T1); _mm_prefetch(q1+320, _MM_HINT_T1);
          _mm_prefetch(q1+384, _MM_HINT_T1); _mm_prefetch(q1+448, _MM_HINT_T1); _mm_prefetch(q1+512, _MM_HINT_T1);
          _mm_prefetch(q1+576, _MM_HINT_T1); _mm_prefetch(q1+640, _MM_HINT_T1); _mm_prefetch(q1+704, _MM_HINT_T1);
          _mm_prefetch(q1+768, _MM_HINT_T1); _mm_prefetch(q1+832, _MM_HINT_T1); _mm_prefetch(q1+896, _MM_HINT_T1);
          _mm_prefetch(q1+960, _MM_HINT_T1); _mm_prefetch(q1+1024, _MM_HINT_T1); _mm_prefetch(q1+1088, _MM_HINT_T1);
          const char* q2 = pf2 + (long)k1*1024;
          _mm_prefetch(q2, _MM_HINT_T1); _mm_prefetch(q2+64, _MM_HINT_T1); _mm_prefetch(q2+128, _MM_HINT_T1);
          _mm_prefetch(q2+192, _MM_HINT_T1); _mm_prefetch(q2+256, _MM_HINT_T1); _mm_prefetch(q2+320, _MM_HINT_T1);
          _mm_prefetch(q2+384, _MM_HINT_T1); _mm_prefetch(q2+448, _MM_HINT_T1); _mm_prefetch(q2+512, _MM_HINT_T1);
          _mm_prefetch(q2+576, _MM_HINT_T1); _mm_prefetch(q2+640, _MM_HINT_T1); _mm_prefetch(q2+704, _MM_HINT_T1);
          _mm_prefetch(q2+768, _MM_HINT_T1); _mm_prefetch(q2+832, _MM_HINT_T1); _mm_prefetch(q2+896, _MM_HINT_T1);
          _mm_prefetch(q2+960, _MM_HINT_T1); }
        const double* p = SCRA + k1*128;
        DECL16(xr,xi)
        LOAD16(p, 16, xr, xi)
        DECL16(yr,yi)
        DFT8(S, xr0,xi0,xr1,xi1,xr2,xi2,xr3,xi3,xr4,xi4,xr5,xi5,xr6,xi6,xr7,xi7,
                yr0,yi0,yr1,yi1,yr2,yi2,yr3,yi3,yr4,yi4,yr5,yi5,yr6,yi6,yr7,yi7);
        const double* cc = c + k1*128;
        yr0=_mm512_add_pd(yr0,_mm512_load_pd(cc+0));   yi0=_mm512_add_pd(yi0,_mm512_load_pd(cc+8));
        yr1=_mm512_add_pd(yr1,_mm512_load_pd(cc+16));  yi1=_mm512_add_pd(yi1,_mm512_load_pd(cc+24));
        yr2=_mm512_add_pd(yr2,_mm512_load_pd(cc+32));  yi2=_mm512_add_pd(yi2,_mm512_load_pd(cc+40));
        yr3=_mm512_add_pd(yr3,_mm512_load_pd(cc+48));  yi3=_mm512_add_pd(yi3,_mm512_load_pd(cc+56));
        yr4=_mm512_add_pd(yr4,_mm512_load_pd(cc+64));  yi4=_mm512_add_pd(yi4,_mm512_load_pd(cc+72));
        yr5=_mm512_add_pd(yr5,_mm512_load_pd(cc+80));  yi5=_mm512_add_pd(yi5,_mm512_load_pd(cc+88));
        yr6=_mm512_add_pd(yr6,_mm512_load_pd(cc+96));  yi6=_mm512_add_pd(yi6,_mm512_load_pd(cc+104));
        yr7=_mm512_add_pd(yr7,_mm512_load_pd(cc+112)); yi7=_mm512_add_pd(yi7,_mm512_load_pd(cc+120));
        MAP2(yr0,yi0); MAP2(yr1,yi1); MAP2(yr2,yi2); MAP2(yr3,yi3);
        MAP2(yr4,yi4); MAP2(yr5,yi5); MAP2(yr6,yi6); MAP2(yr7,yi7);
        DECL16(zr,zi)
        DFT8(S, yr0,yi0,yr1,yi1,yr2,yi2,yr3,yi3,yr4,yi4,yr5,yi5,yr6,yi6,yr7,yi7,
                zr0,zi0,zr1,zi1,zr2,zi2,zr3,zi3,zr4,zi4,zr5,zi5,zr6,zi6,zr7,zi7);
        if(k1){ TW7(k1, zr, zi); }
        double* q = SCRB + k1*16;
        _mm512_store_pd(q+0*128, zr0); _mm512_store_pd(q+0*128+8, zi0);
        _mm512_store_pd(q+1*128, zr1); _mm512_store_pd(q+1*128+8, zi1);
        _mm512_store_pd(q+2*128, zr2); _mm512_store_pd(q+2*128+8, zi2);
        _mm512_store_pd(q+3*128, zr3); _mm512_store_pd(q+3*128+8, zi3);
        _mm512_store_pd(q+4*128, zr4); _mm512_store_pd(q+4*128+8, zi4);
        _mm512_store_pd(q+5*128, zr5); _mm512_store_pd(q+5*128+8, zi5);
        _mm512_store_pd(q+6*128, zr6); _mm512_store_pd(q+6*128+8, zi6);
        _mm512_store_pd(q+7*128, zr7); _mm512_store_pd(q+7*128+8, zi7);
    }
}


// map pass over a full plane, c in natural (X64) layout
__attribute__((noinline))
static void map_plane(double* pl, const double* cpl){
    for(int y=0;y<64;y++){
        double* r = pl + (long)y*RS64;
        const double* cr = cpl + (long)y*RS64;
        for(int v=0;v<8;v+=2){
            __m512d zr0=_mm512_add_pd(_mm512_load_pd(r+v*16),    _mm512_load_pd(cr+v*16));
            __m512d zi0=_mm512_add_pd(_mm512_load_pd(r+v*16+8),  _mm512_load_pd(cr+v*16+8));
            __m512d zr1=_mm512_add_pd(_mm512_load_pd(r+v*16+16), _mm512_load_pd(cr+v*16+16));
            __m512d zi1=_mm512_add_pd(_mm512_load_pd(r+v*16+24), _mm512_load_pd(cr+v*16+24));
            MAP2(zr0,zi0); MAP2(zr1,zi1);
            _mm512_store_pd(r+v*16,    zr0); _mm512_store_pd(r+v*16+8,  zi0);
            _mm512_store_pd(r+v*16+16, zr1); _mm512_store_pd(r+v*16+24, zi1);
        }
    }
}
// ============================== sweeps =====================================
// SB modes: 0 = FyFz plain (init); 1 = steady (Fy,Fz,+c,map,Fz',Fy'); 2 = final (Fy,Fz,+c,map)
static unsigned long long TC[8];
static inline unsigned long long rdt0(void){ unsigned a,d,c; __asm__ volatile("rdtscp":"=a"(a),"=d"(d),"=c"(c)); return ((unsigned long long)d<<32)|a; }
static void SB64(int mode){
    for(int i=0;i<64;i++){
        double* pl = X64 + (long)i*PS64;
        unsigned long long t0=rdt0();
        for(int v=0; v<8; v++){ col_s1(pl + v*16, RS64); col_s2(pl + v*16, RS64); }
        unsigned long long t1=rdt0(); TC[4]+=t1-t0;
        for(int yb=0; yb<8; yb++){
            double* rb = pl + (long)yb*8*RS64;
            zcol_s1(rb);
            if(mode==0){ zcol_s2a(rb); }
            else if(mode==2){ zcol_s2a_map(rb, CS64 + ((long)i*8+yb)*1024); }
            else { int ip = (i+1)&63;
                   zcol_s2a_map_s1(CS64 + ((long)i*8+yb)*1024,
                        (const char*)(X64 + (long)ip*PS64 + (long)yb*8*RS64),
                        (const char*)(CS64 + ((long)ip*8+yb)*1024)); zcol_s2b(rb); }
        }
        unsigned long long t2=rdt0(); TC[5]+=t2-t1;
        if(mode==1){
            for(int v=0; v<8; v++){ col_s1(pl + v*16, RS64); col_s2(pl + v*16, RS64); }
        }
        TC[6]+=rdt0()-t2;
    }
}
// PB modes: 0 = Fx plain; 1 = Fx,+c,map plain (first/final); 2 = steady (Fx,+c,map,Fx')
static void PB64(int mode){
    for(int y=0;y<64;y++){
        for(int v=0;v<8;v++){
            double* p = X64 + (long)y*RS64 + v*16;
            // prefetch the pencil 2 ahead (v+2, wrapping to next y)
            long e = (long)y*8 + v + 2;
            const char* pf = (const char*)(X64 + (e/8)*RS64 + (e%8)*16);
            col_s1_pf(p, PS64, pf, PS64*8);
            if(mode==0){ col_s2(p, PS64); }
            else if(mode==1){ col_s2_map(p, PS64, CP64 + ((long)y*8+v)*1024); }
            else { col_s2_map_s1(CP64 + ((long)y*8+v)*1024, (const char*)(CP64 + ((long)y*8+v+2)*1024)); col_s2b(p, PS64); }
        }
    }
}

// =========================== conversions ==================================
// digit swap
static inline int ds(int t){ return ((t&7)<<3) | (t>>3); }
// convin: natural complex volume (in[((x*64+y)*64+z)*2]) -> X64 layout
static void convin64(const double* in, double* X){
    for(int i=0;i<64;i++){
        int xs = ds(i);
        for(int y=0;y<64;y++){
            int ys = ds(y);
            const double* s = in + (((long)xs*64+ys)*64)*2;
            double* d = X + (long)i*PS64 + (long)y*RS64;
            // slot s lane l <- z = 8l+s : gather via 8x8 transpose of (l,s)
            __m512d a0,a1,a2,a3,a4,a5,a6,a7, b0,b1,b2,b3,b4,b5,b6,b7;
            // load z-major pairs: row l: z=8l..8l+7 -> re in a_l, im in b_l
            for(int l=0;l<8;l++){
                __m512d lo=_mm512_loadu_pd(s + l*16), hi=_mm512_loadu_pd(s + l*16 + 8);
                // deinterleave re/im
                __m512i idxe=_mm512_set_epi64(14,12,10,8,6,4,2,0);
                __m512i idxo=_mm512_set_epi64(15,13,11,9,7,5,3,1);
                __m512d re=_mm512_permutex2var_pd(lo, idxe, hi);
                __m512d im=_mm512_permutex2var_pd(lo, idxo, hi);
                switch(l){
                case 0: a0=re;b0=im;break; case 1: a1=re;b1=im;break;
                case 2: a2=re;b2=im;break; case 3: a3=re;b3=im;break;
                case 4: a4=re;b4=im;break; case 5: a5=re;b5=im;break;
                case 6: a6=re;b6=im;break; default: a7=re;b7=im;break;}
            }
            TR8(a0,a1,a2,a3,a4,a5,a6,a7);
            TR8(b0,b1,b2,b3,b4,b5,b6,b7);
            _mm512_store_pd(d+0*16, a0); _mm512_store_pd(d+0*16+8, b0);
            _mm512_store_pd(d+1*16, a1); _mm512_store_pd(d+1*16+8, b1);
            _mm512_store_pd(d+2*16, a2); _mm512_store_pd(d+2*16+8, b2);
            _mm512_store_pd(d+3*16, a3); _mm512_store_pd(d+3*16+8, b3);
            _mm512_store_pd(d+4*16, a4); _mm512_store_pd(d+4*16+8, b4);
            _mm512_store_pd(d+5*16, a5); _mm512_store_pd(d+5*16+8, b5);
            _mm512_store_pd(d+6*16, a6); _mm512_store_pd(d+6*16+8, b6);
            _mm512_store_pd(d+7*16, a7); _mm512_store_pd(d+7*16+8, b7);
        }
    }
}
// convout: X64 layout -> natural complex volume
static void convout64(const double* X, double* out){
    for(int i=0;i<64;i++){
        int xs = ds(i);
        for(int y=0;y<64;y++){
            int ys = ds(y);
            double* dnat = out + (((long)xs*64+ys)*64)*2;
            const double* dsrc = X + (long)i*PS64 + (long)y*RS64;
            __m512d a0,a1,a2,a3,a4,a5,a6,a7, b0,b1,b2,b3,b4,b5,b6,b7;
            a0=_mm512_load_pd(dsrc+0*16); b0=_mm512_load_pd(dsrc+0*16+8);
            a1=_mm512_load_pd(dsrc+1*16); b1=_mm512_load_pd(dsrc+1*16+8);
            a2=_mm512_load_pd(dsrc+2*16); b2=_mm512_load_pd(dsrc+2*16+8);
            a3=_mm512_load_pd(dsrc+3*16); b3=_mm512_load_pd(dsrc+3*16+8);
            a4=_mm512_load_pd(dsrc+4*16); b4=_mm512_load_pd(dsrc+4*16+8);
            a5=_mm512_load_pd(dsrc+5*16); b5=_mm512_load_pd(dsrc+5*16+8);
            a6=_mm512_load_pd(dsrc+6*16); b6=_mm512_load_pd(dsrc+6*16+8);
            a7=_mm512_load_pd(dsrc+7*16); b7=_mm512_load_pd(dsrc+7*16+8);
            TR8(a0,a1,a2,a3,a4,a5,a6,a7);
            TR8(b0,b1,b2,b3,b4,b5,b6,b7);
            // row l: z = 8l..8l+7: re=a_l, im=b_l; interleave
            __m512i idl=_mm512_set_epi64(11,3,10,2,9,1,8,0);
            __m512i idh=_mm512_set_epi64(15,7,14,6,13,5,12,4);
            #define PUTROW(l, re, im) { \
                __m512d lo=_mm512_permutex2var_pd(re, idl, im); \
                __m512d hi=_mm512_permutex2var_pd(re, idh, im); \
                _mm512_storeu_pd(dnat + l*16, lo); _mm512_storeu_pd(dnat + l*16 + 8, hi); }
            PUTROW(0,a0,b0) PUTROW(1,a1,b1) PUTROW(2,a2,b2) PUTROW(3,a3,b3)
            PUTROW(4,a4,b4) PUTROW(5,a5,b5) PUTROW(6,a6,b6) PUTROW(7,a7,b7)
            #undef PUTROW
        }
    }
}
// c_slab: block (i, yb, k1) zmm k2 lane l = c[ds(i)][8l+yb][k1+8k2]
static void build_cs64(const double* c, double* CS){
    for(int i=0;i<64;i++){
        int xs=ds(i);
        for(int yb=0;yb<8;yb++){
            double* blk = CS + ((long)i*8+yb)*1024;
            for(int k1=0;k1<8;k1++){
                for(int k2=0;k2<8;k2++){
                    for(int l=0;l<8;l++){
                        const double* src = c + (((long)xs*64 + (8*l+yb))*64 + (k1+8*k2))*2;
                        blk[k1*128 + k2*16 + l] = src[0];
                        blk[k1*128 + k2*16 + 8 + l] = src[1];
                    }
                }
            }
        }
    }
}
// c_penc: block (y, v, k1) zmm k2 lane l = c[k1+8k2][ds(y)][8l+v]
static void build_cp64(const double* c, double* CP){
    for(int y=0;y<64;y++){
        int ys=ds(y);
        for(int v=0;v<8;v++){
            double* blk = CP + ((long)y*8+v)*1024;
            for(int k1=0;k1<8;k1++){
                for(int k2=0;k2<8;k2++){
                    for(int l=0;l<8;l++){
                        const double* src = c + (((long)(k1+8*k2)*64 + ys)*64 + (8*l+v))*2;
                        blk[k1*128 + k2*16 + l] = src[0];
                        blk[k1*128 + k2*16 + 8 + l] = src[1];
                    }
                }
            }
        }
    }
}

static void* alloc_big(long bytes){
    void* p = mmap(0, (bytes+2097151)&~2097151L, PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    madvise(p, (bytes+2097151)&~2097151L, MADV_HUGEPAGE);
    memset(p, 0, bytes);
    return p;
}

static void init64(void){
    if(X64) return;
    X64 = alloc_big((long)64*PS64*8);
    CS64 = alloc_big((long)64*8*1024*8);
    CP64 = alloc_big((long)64*8*1024*8);
    for(int g=0; g<8; g++)
        for(int k=1;k<8;k++){
            long double ang = -2.0L*3.14159265358979323846264338327950288L*((g*k)%64)/64.0L;
            TW64[g][k-1][0] = (double)cosl(ang);
            TW64[g][k-1][1] = (double)sinl(ang);
        }
}

void run_64(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    init64();
    if(m<1) m=1;
    for(long b=0;b<B;b++){
        convin64(x0 + b*2*262144, X64);
        build_cs64(c + b*2*262144, CS64);
        build_cp64(c + b*2*262144, CP64);
        SB64(0);               // X = FyFz X0
        PB64(1);               // X = X1 plain
        convout64(X64, out1 + b*2*262144);
        if(m==1){ memcpy(outm + b*2*262144, out1 + b*2*262144, 262144*16); continue; }
        PB64(0);               // X = Fx X1
        for(long t=2;t<=m;t++){
            if((t&1)==0){ SB64(t==m ? 2 : 1); }
            else        { PB64(t==m ? 1 : 2); }
        }
        convout64(X64, outm + b*2*262144);
    }
}

// ---- test helpers ----
void t_roundtrip(const double* in, double* out){
    init64();
    convin64(in, X64);
    convout64(X64, out);
}
void t_fft3(const double* in, double* out){
    init64();
    convin64(in, X64);
    SB64(0);
    PB64(0);
    convout64(X64, out);
}
void t_zonly(const double* in, double* out){
    init64();
    convin64(in, X64);
    for(int i=0;i<64;i++){
        double* pl = X64 + (long)i*PS64;
        for(int yb=0; yb<8; yb++){
            double* rb = pl + (long)yb*8*RS64;
            zcol_s1(rb);
            zcol_s2a(rb);
        }
    }
    convout64(X64, out);
}
void t_yonly(const double* in, double* out){
    init64();
    convin64(in, X64);
    for(int i=0;i<64;i++){
        double* pl = X64 + (long)i*PS64;
        for(int v=0; v<8; v++){ col_s1(pl + v*16, RS64); col_s2(pl + v*16, RS64); }
    }
    convout64(X64, out);
}
void t_xonly(const double* in, double* out){
    init64();
    convin64(in, X64);
    PB64(0);
    convout64(X64, out);
}

static inline unsigned long long rdt(void){ return rdt0(); }
void get_tc(unsigned long long* o){ for(int i=0;i<8;i++){o[i]=TC[i]; TC[i]=0;} }
void run_64p(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    init64();
    if(m<1) m=1;
    for(long b=0;b<B;b++){
        convin64(x0 + b*2*262144, X64);
        build_cs64(c + b*2*262144, CS64);
        build_cp64(c + b*2*262144, CP64);
        SB64(0);
        PB64(1);
        convout64(X64, out1 + b*2*262144);
        if(m==1){ memcpy(outm + b*2*262144, out1 + b*2*262144, 262144*16); continue; }
        PB64(0);
        for(long t=2;t<=m;t++){
            unsigned long long t0=rdt();
            if((t&1)==0){ SB64(t==m ? 2 : 1); TC[0]+=rdt()-t0; TC[1]++; }
            else        { PB64(t==m ? 1 : 2); TC[2]+=rdt()-t0; TC[3]++; }
        }
        convout64(X64, outm + b*2*262144);
    }
}


