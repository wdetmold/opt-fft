// PFA engine for L = 45 (5x9) and 36 (4x9): iterated z = FFT3(x)+c ; x <- z/(1+|z|)
// Storage: per-axis positions in PFA input order: position q holds value index
// VAL(q) = (N2IN*n1 + N1IN*n2) % L  with q = n2*N1 + n1   (N1-contig groups)
// S1: DFT_N1 over contiguous groups -> scratch [k1*N2 + n2]
// S2: DFT_N2 over scratch groups -> output positions OT[k1][k2] (table)
// map fused at start of next step's S1 (c prepermuted to position order).
#include <immintrin.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <sys/mman.h>

#define ALIGN64 __attribute__((aligned(64)))

static inline unsigned long long rdt0(void){ unsigned a,d,c; __asm__ volatile("rdtscp":"=a"(a),"=d"(d),"=c"(c)); return ((unsigned long long)d<<32)|a; }
static unsigned long long TCP[16];
void get_tcp(unsigned long long* o){ for(int i=0;i<16;i++){o[i]=TCP[i]; TCP[i]=0;} }

static void* alloc_big2(long bytes){
    void* p = mmap(0, (bytes+2097151)&~2097151L, PROT_READ|PROT_WRITE,
                   MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    madvise(p, (bytes+2097151)&~2097151L, MADV_HUGEPAGE);
    memset(p, 0, bytes);
    return p;
}

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

// ---- DFT3 on complex reg pairs (forward): y = F3 x ----
#define C3  (-0.5)
#define S3  (-0.86602540378443864676372317075293618347)  /* -sqrt(3)/2 (forward) */
#define DFT3(x0r,x0i,x1r,x1i,x2r,x2i, y0r,y0i,y1r,y1i,y2r,y2i) do{ \
    __m512d _tr=_mm512_add_pd(x1r,x2r), _ti=_mm512_add_pd(x1i,x2i); \
    __m512d _dr=_mm512_sub_pd(x1r,x2r), _di=_mm512_sub_pd(x1i,x2i); \
    __m512d _mr=_mm512_fmadd_pd(_mm512_set1_pd(C3), _tr, x0r); \
    __m512d _mi=_mm512_fmadd_pd(_mm512_set1_pd(C3), _ti, x0i); \
    y0r=_mm512_add_pd(x0r,_tr); y0i=_mm512_add_pd(x0i,_ti); \
    y1r=_mm512_fnmadd_pd(_mm512_set1_pd(S3), _di, _mr); \
    y1i=_mm512_fmadd_pd(_mm512_set1_pd(S3), _dr, _mi); \
    y2r=_mm512_fmadd_pd(_mm512_set1_pd(S3), _di, _mr); \
    y2i=_mm512_fnmadd_pd(_mm512_set1_pd(S3), _dr, _mi); \
}while(0)
/* check: X1 = x0 + w x1 + w^2 x2, w=e^{-2pi i/3} = -1/2 - i s, s=sqrt3/2.
   X1 = (x0 -(x1+x2)/2) - i s (x1 - x2)  => X1r = mr + s*di', X1i = mi - s*dr'
   with s=+sqrt3/2: X1r = mr + s*_di, X1i = mi - s*_dr.
   S3 = -s above, so y1r = mr - S3... careful: y1r = fnmadd(S3,_di,_mr) = mr - S3*di = mr + s*di  OK
   y1i = fmadd(S3,_dr,_mi) = mi + S3*dr = mi - s*dr  OK ; y2 mirrored. */

// twiddle w9^t: (r,i) -> (r c - i s, i c + r s), c=cos(-2pi t/9) etc (baked below)
static double TW9[5][2] ALIGN64; // for t=1..4: [t-1] = {cos,sin} of -2pi t/9... (5th unused)
#define CTW2(rr, ii, cval, sval) do{ \
    __m512d _c=_mm512_set1_pd(cval), _s=_mm512_set1_pd(sval); \
    __m512d _tr=_mm512_mul_pd(rr,_c); \
    __m512d _ti=_mm512_mul_pd(ii,_c); \
    _tr=_mm512_fnmadd_pd(_s,ii,_tr); \
    _ti=_mm512_fmadd_pd(_s,rr,_ti); \
    rr=_tr; ii=_ti; \
}while(0)

// DFT9 as 3x3 CT: x indexed 0..8 (position = value index j), out y0..y8 (value index k)
// j = 3a+b ; sub_b = DFT3 over a; twiddle w9^{q b}; out[3t+q] = DFT3_b over b.
#define DFT9(xr,xi,yr,yi) do{ \
    __m512d s0r,s0i,s1r,s1i,s2r,s2i, t0r,t0i,t1r,t1i,t2r,t2i, u0r,u0i,u1r,u1i,u2r,u2i; \
    DFT3(xr##0,xi##0,xr##3,xi##3,xr##6,xi##6, s0r,s0i,s1r,s1i,s2r,s2i); \
    DFT3(xr##1,xi##1,xr##4,xi##4,xr##7,xi##7, t0r,t0i,t1r,t1i,t2r,t2i); \
    DFT3(xr##2,xi##2,xr##5,xi##5,xr##8,xi##8, u0r,u0i,u1r,u1i,u2r,u2i); \
    /* q=1 row needs b-twiddles w9^{1*b}: t1 *= w9^1, u1 *= w9^2 ; q=2: t2 *= w9^2, u2 *= w9^4 */ \
    CTW2(t1r,t1i, TW9[0][0], TW9[0][1]); \
    CTW2(u1r,u1i, TW9[1][0], TW9[1][1]); \
    CTW2(t2r,t2i, TW9[1][0], TW9[1][1]); \
    CTW2(u2r,u2i, TW9[3][0], TW9[3][1]); \
    DFT3(s0r,s0i,t0r,t0i,u0r,u0i, yr##0,yi##0,yr##3,yi##3,yr##6,yi##6); \
    DFT3(s1r,s1i,t1r,t1i,u1r,u1i, yr##1,yi##1,yr##4,yi##4,yr##7,yi##7); \
    DFT3(s2r,s2i,t2r,t2i,u2r,u2i, yr##2,yi##2,yr##5,yi##5,yr##8,yi##8); \
}while(0)

// DFT5 (forward)
#define C51 (0.30901699437494742410229341718281905886)
#define C52 (-0.80901699437494742410229341718281905886)
#define S51 (0.95105651629515357211643933337938214340)
#define S52 (0.58778525229247312916870595463907276860)
#define DFT5(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i, y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i) do{ \
    __m512d u1r=_mm512_add_pd(x1r,x4r), u1i=_mm512_add_pd(x1i,x4i); \
    __m512d u2r=_mm512_add_pd(x2r,x3r), u2i=_mm512_add_pd(x2i,x3i); \
    __m512d v1r=_mm512_sub_pd(x1r,x4r), v1i=_mm512_sub_pd(x1i,x4i); \
    __m512d v2r=_mm512_sub_pd(x2r,x3r), v2i=_mm512_sub_pd(x2i,x3i); \
    y0r=_mm512_add_pd(x0r,_mm512_add_pd(u1r,u2r)); y0i=_mm512_add_pd(x0i,_mm512_add_pd(u1i,u2i)); \
    __m512d a1r=_mm512_fmadd_pd(_mm512_set1_pd(C51),u1r,_mm512_fmadd_pd(_mm512_set1_pd(C52),u2r,x0r)); \
    __m512d a1i=_mm512_fmadd_pd(_mm512_set1_pd(C51),u1i,_mm512_fmadd_pd(_mm512_set1_pd(C52),u2i,x0i)); \
    __m512d a2r=_mm512_fmadd_pd(_mm512_set1_pd(C52),u1r,_mm512_fmadd_pd(_mm512_set1_pd(C51),u2r,x0r)); \
    __m512d a2i=_mm512_fmadd_pd(_mm512_set1_pd(C52),u1i,_mm512_fmadd_pd(_mm512_set1_pd(C51),u2i,x0i)); \
    __m512d b1r=_mm512_fmadd_pd(_mm512_set1_pd(S51),v1r,_mm512_mul_pd(_mm512_set1_pd(S52),v2r)); \
    __m512d b1i=_mm512_fmadd_pd(_mm512_set1_pd(S51),v1i,_mm512_mul_pd(_mm512_set1_pd(S52),v2i)); \
    __m512d b2r=_mm512_fmsub_pd(_mm512_set1_pd(S52),v1r,_mm512_mul_pd(_mm512_set1_pd(S51),v2r)); \
    __m512d b2i=_mm512_fmsub_pd(_mm512_set1_pd(S52),v1i,_mm512_mul_pd(_mm512_set1_pd(S51),v2i)); \
    y1r=_mm512_add_pd(a1r,b1i); y1i=_mm512_sub_pd(a1i,b1r); \
    y4r=_mm512_sub_pd(a1r,b1i); y4i=_mm512_add_pd(a1i,b1r); \
    y2r=_mm512_add_pd(a2r,b2i); y2i=_mm512_sub_pd(a2i,b2r); \
    y3r=_mm512_sub_pd(a2r,b2i); y3i=_mm512_add_pd(a2i,b2r); \
}while(0)

// DFT4 (forward): X0=x0+x1+x2+x3 ... X1 = (x0-x2) - i(x1-x3), X2 = x0-x1+x2-x3, X3 = (x0-x2)+i(x1-x3)
#define DFT4(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i, y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i) do{ \
    __m512d t0r=_mm512_add_pd(x0r,x2r), t0i=_mm512_add_pd(x0i,x2i); \
    __m512d t1r=_mm512_sub_pd(x0r,x2r), t1i=_mm512_sub_pd(x0i,x2i); \
    __m512d t2r=_mm512_add_pd(x1r,x3r), t2i=_mm512_add_pd(x1i,x3i); \
    __m512d t3r=_mm512_sub_pd(x1r,x3r), t3i=_mm512_sub_pd(x1i,x3i); \
    y0r=_mm512_add_pd(t0r,t2r); y0i=_mm512_add_pd(t0i,t2i); \
    y2r=_mm512_sub_pd(t0r,t2r); y2i=_mm512_sub_pd(t0i,t2i); \
    y1r=_mm512_add_pd(t1r,t3i); y1i=_mm512_sub_pd(t1i,t3r); \
    y3r=_mm512_sub_pd(t1r,t3i); y3i=_mm512_add_pd(t1i,t3r); \
}while(0)

static void init_tw9(void){
    for(int t=1;t<=4;t++){
        long double ang = -2.0L*3.14159265358979323846264338327950288L*t/9.0L;
        TW9[t-1][0]=(double)cosl(ang); TW9[t-1][1]=(double)sinl(ang);
    }
}

// =================== instantiate L=45 ===================
#define LL 45
#define N1 5
#define N2 9
#define NSLOT 6
#define YP 48              /* padded rows */
#define RS 96
#define PS 4624
#define SUF(x) x##_45
#include "pfa_impl.h"
#undef LL
#undef N1
#undef N2
#undef NSLOT
#undef YP
#undef RS
#undef PS
#undef SUF

// =================== instantiate L=36 ===================
#define LL 36
#define N1 4
#define N2 9
#define NSLOT 5
#define YP 40
#define RS 80
#define PS 3216
#define SUF(x) x##_36
#include "pfa_impl.h"
#undef LL
#undef N1
#undef N2
#undef NSLOT
#undef YP
#undef RS
#undef PS
#undef SUF
