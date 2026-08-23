// ============================================================================
// Iterated batched 3D complex-to-complex FFTs for L in {6,8,13,17,23,36,45,64}
// with the elementwise map  x <- z/(1+|z|),  z = FFT3(x) + c,  iterated m_L
// times per volume. Forward unnormalized DFT, C-order volumes.
//
// ALL transform arithmetic in this file is hand-written (no FFT libraries):
//   - 6  = Good-Thomas PFA(2,3) of hand-coded DFT2/DFT3 kernels
//   - 8  = hand-coded split DFT8 (DIT radix-2 with trivial twiddles)
//   - 13,17,23 = direct symmetric (cos/sin folded) prime DFT, FMA-dominated,
//                4-way blocked over output pairs (k, L-k)
//   - 36 = Good-Thomas PFA(4,9), DFT9 = 3x3 Cooley-Tukey of DFT3s
//   - 45 = Good-Thomas PFA(5,9)
//   - 64 = 8x8 Cooley-Tukey of two hand-coded DFT8 layers + 49 twiddles
//
// Strategy: vertical SIMD (8-wide AVX-512 doubles), split re/im storage in
// 128-byte blocks of 8 complex values. Two batching schemes:
//   BL ("batch-lane"): SIMD lanes = 8 independent volumes. Zero shuffle
//       overhead, zero padding waste; used for small L when the batch B
//       is large enough. Working set = one 8-volume group, L3-resident.
//   PV ("per-volume"): SIMD lanes = 8 consecutive z samples (z chunks padded
//       to 8); the z-axis pass uses in-register 8x8 transposes (strips of 8
//       rows, with an overlapped final strip when L%8 != 0 so no y padding
//       is needed). Used for large L (36,45,64) and for small batches.
// Each step does:  sweep1 = x-axis line FFTs;  sweep2 = per-x-plane y-axis
// and z-axis FFTs followed by the fused "+c and map" pass (plane-sequential,
// cache-friendly). Order of axis transforms is irrelevant mathematically.
//
// The map z/(1+|z|) uses rsqrt14 + 2 Newton steps for sqrt and rcp14 + 2
// Newton steps for the reciprocal (both good to ~1e-16 relative, verified
// against the long-double reference; one-step block error ~9e-16 rel L2).
// Twiddle/cos/sin tables are computed once at import in 80-bit long double
// with exact mod-L argument reduction.
//
// Single-threaded. No OpenMP, no pthreads, no external compute libraries.
// ============================================================================
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <immintrin.h>

typedef double v8 __attribute__((vector_size(64), aligned(64)));
typedef double v8u __attribute__((vector_size(64), aligned(8)));
#define AI static inline __attribute__((always_inline))

AI v8 splat(double x){ return (v8)_mm512_set1_pd(x); }
#define VZERO ((v8){0,0,0,0,0,0,0,0})

// ---------------------------------------------------------------- tables
static double tab13[6*6*2];    // [k-1][j-1][{cos,sin}] of 2*pi*k*j/13
static double tab17[8*8*2];
static double tab23[11*11*2];
static double tw64[8*8*2];     // W64^{j1*k1} = exp(-2*pi*i*j1*k1/64) [re,im]
static double K3;              // sqrt(3)/2
static double K8;              // sqrt(2)/2
static double C51,C52,S51,S52; // cos/sin 2pi/5, 4pi/5
static double W91r,W91i,W92r,W92i,W94r,W94i; // W9^1,2,4

static void fill_prime_tab(double *tab, int L){
    int h=(L-1)/2;
    long double PI = acosl(-1.0L);
    for(int k=1;k<=h;k++) for(int j=1;j<=h;j++){
        long double ang = 2.0L*PI*(long double)((k*j)%L)/(long double)L;
        tab[((k-1)*h+(j-1))*2+0] = (double)cosl(ang);
        tab[((k-1)*h+(j-1))*2+1] = (double)sinl(ang);
    }
}
void init_tables(void){
    long double PI = acosl(-1.0L);
    fill_prime_tab(tab13,13);
    fill_prime_tab(tab17,17);
    fill_prime_tab(tab23,23);
    for(int j1=0;j1<8;j1++) for(int k1=0;k1<8;k1++){
        long double ang = -2.0L*PI*(long double)((j1*k1)%64)/64.0L;
        tw64[(j1*8+k1)*2+0]=(double)cosl(ang);
        tw64[(j1*8+k1)*2+1]=(double)sinl(ang);
    }
    K3=(double)(sqrtl(3.0L)/2.0L);
    K8=(double)(sqrtl(2.0L)/2.0L);
    C51=(double)cosl(2.0L*PI/5.0L); S51=(double)sinl(2.0L*PI/5.0L);
    C52=(double)cosl(4.0L*PI/5.0L); S52=(double)sinl(4.0L*PI/5.0L);
    W91r=(double)cosl(-2.0L*PI/9.0L);  W91i=(double)sinl(-2.0L*PI/9.0L);
    W92r=(double)cosl(-4.0L*PI/9.0L);  W92i=(double)sinl(-4.0L*PI/9.0L);
    W94r=(double)cosl(-8.0L*PI/9.0L);  W94i=(double)sinl(-8.0L*PI/9.0L);
}

// ---------------------------------------------------------------- map
// z <- z/(1+|z|) elementwise on an 8-wide block, z has been loaded in (zr,zi)
AI void vmap(v8 *zr, v8 *zi){
    v8 s = *zr * *zr + *zi * *zi;
    s = (v8)_mm512_max_pd((__m512d)s, (__m512d)splat(1e-300));
    v8 r = (v8)_mm512_rsqrt14_pd((__m512d)s);
    v8 hs = s*splat(0.5);
    r = r*(splat(1.5) - hs*r*r);
    r = r*(splat(1.5) - hs*r*r);
    v8 u = splat(1.0) + s*r;
    v8 t = (v8)_mm512_rcp14_pd((__m512d)u);
    t = t*(splat(2.0) - u*t);
    t = t*(splat(2.0) - u*t);
    *zr *= t; *zi *= t;
}
// 4-way ILP map over n blocks: p <- (p+c)/(1+|p+c|)
AI void map_blocks(v8 *restrict p, const v8 *restrict c, long n){
    long i=0;
    for(; i+4<=n; i+=4){
        v8 zr0=p[2*i+0]+c[2*i+0], zi0=p[2*i+1]+c[2*i+1];
        v8 zr1=p[2*i+2]+c[2*i+2], zi1=p[2*i+3]+c[2*i+3];
        v8 zr2=p[2*i+4]+c[2*i+4], zi2=p[2*i+5]+c[2*i+5];
        v8 zr3=p[2*i+6]+c[2*i+6], zi3=p[2*i+7]+c[2*i+7];
        v8 s0=zr0*zr0+zi0*zi0, s1=zr1*zr1+zi1*zi1, s2=zr2*zr2+zi2*zi2, s3=zr3*zr3+zi3*zi3;
        s0=(v8)_mm512_max_pd((__m512d)s0,(__m512d)splat(1e-300));
        s1=(v8)_mm512_max_pd((__m512d)s1,(__m512d)splat(1e-300));
        s2=(v8)_mm512_max_pd((__m512d)s2,(__m512d)splat(1e-300));
        s3=(v8)_mm512_max_pd((__m512d)s3,(__m512d)splat(1e-300));
        v8 r0=(v8)_mm512_rsqrt14_pd((__m512d)s0), r1=(v8)_mm512_rsqrt14_pd((__m512d)s1);
        v8 r2=(v8)_mm512_rsqrt14_pd((__m512d)s2), r3=(v8)_mm512_rsqrt14_pd((__m512d)s3);
        v8 h0=s0*splat(0.5), h1=s1*splat(0.5), h2=s2*splat(0.5), h3=s3*splat(0.5);
        r0=r0*(splat(1.5)-h0*r0*r0); r1=r1*(splat(1.5)-h1*r1*r1);
        r2=r2*(splat(1.5)-h2*r2*r2); r3=r3*(splat(1.5)-h3*r3*r3);
        r0=r0*(splat(1.5)-h0*r0*r0); r1=r1*(splat(1.5)-h1*r1*r1);
        r2=r2*(splat(1.5)-h2*r2*r2); r3=r3*(splat(1.5)-h3*r3*r3);
        v8 u0=splat(1.0)+s0*r0, u1=splat(1.0)+s1*r1, u2=splat(1.0)+s2*r2, u3=splat(1.0)+s3*r3;
        v8 t0=(v8)_mm512_rcp14_pd((__m512d)u0), t1=(v8)_mm512_rcp14_pd((__m512d)u1);
        v8 t2=(v8)_mm512_rcp14_pd((__m512d)u2), t3=(v8)_mm512_rcp14_pd((__m512d)u3);
        t0=t0*(splat(2.0)-u0*t0); t1=t1*(splat(2.0)-u1*t1);
        t2=t2*(splat(2.0)-u2*t2); t3=t3*(splat(2.0)-u3*t3);
        t0=t0*(splat(2.0)-u0*t0); t1=t1*(splat(2.0)-u1*t1);
        t2=t2*(splat(2.0)-u2*t2); t3=t3*(splat(2.0)-u3*t3);
        p[2*i+0]=zr0*t0; p[2*i+1]=zi0*t0;
        p[2*i+2]=zr1*t1; p[2*i+3]=zi1*t1;
        p[2*i+4]=zr2*t2; p[2*i+5]=zi2*t2;
        p[2*i+6]=zr3*t3; p[2*i+7]=zi3*t3;
    }
    for(; i<n; i++){
        v8 zr=p[2*i]+c[2*i], zi=p[2*i+1]+c[2*i+1];
        vmap(&zr,&zi);
        p[2*i]=zr; p[2*i+1]=zi;
    }
}

// ---------------------------------------------------------------- small DFT building blocks (on v8 lvalues)
#define CMUL(or_,oi_, ar,ai, br,bi) { v8 _t=(ar); (or_)=_t*(br)-(ai)*(bi); (oi_)=_t*(bi)+(ai)*(br); }

#define DFT2(y0r,y0i,y1r,y1i, a0r,a0i,a1r,a1i) { \
    v8 _ar=(a0r),_ai=(a0i),_br=(a1r),_bi=(a1i); \
    (y0r)=_ar+_br; (y0i)=_ai+_bi; (y1r)=_ar-_br; (y1i)=_ai-_bi; }

#define DFT3(y0r,y0i,y1r,y1i,y2r,y2i, x0r,x0i,x1r,x1i,x2r,x2i) { \
    v8 _ur=(x1r)+(x2r), _ui=(x1i)+(x2i); \
    v8 _vr=(x1r)-(x2r), _vi=(x1i)-(x2i); \
    v8 _mr=(x0r)-splat(0.5)*_ur, _mi=(x0i)-splat(0.5)*_ui; \
    v8 _wr=splat(K3)*_vr, _wi=splat(K3)*_vi; \
    (y0r)=(x0r)+_ur; (y0i)=(x0i)+_ui; \
    (y1r)=_mr+_wi; (y1i)=_mi-_wr; \
    (y2r)=_mr-_wi; (y2i)=_mi+_wr; }

#define DFT4(y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i, a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i) { \
    v8 _t0r=(a0r)+(a2r), _t0i=(a0i)+(a2i); \
    v8 _t1r=(a0r)-(a2r), _t1i=(a0i)-(a2i); \
    v8 _t2r=(a1r)+(a3r), _t2i=(a1i)+(a3i); \
    v8 _t3r=(a1r)-(a3r), _t3i=(a1i)-(a3i); \
    (y0r)=_t0r+_t2r; (y0i)=_t0i+_t2i; \
    (y2r)=_t0r-_t2r; (y2i)=_t0i-_t2i; \
    (y1r)=_t1r+_t3i; (y1i)=_t1i-_t3r; \
    (y3r)=_t1r-_t3i; (y3i)=_t1i+_t3r; }

#define DFT5(y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i, x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i) { \
    v8 _s1r=(x1r)+(x4r), _s1i=(x1i)+(x4i), _d1r=(x1r)-(x4r), _d1i=(x1i)-(x4i); \
    v8 _s2r=(x2r)+(x3r), _s2i=(x2i)+(x3i), _d2r=(x2r)-(x3r), _d2i=(x2i)-(x3i); \
    v8 _x0r=(x0r), _x0i=(x0i); \
    (y0r)=_x0r+_s1r+_s2r; (y0i)=_x0i+_s1i+_s2i; \
    { v8 _ar=_x0r+splat(C51)*_s1r+splat(C52)*_s2r, _ai=_x0i+splat(C51)*_s1i+splat(C52)*_s2i; \
      v8 _br=splat(S51)*_d1r+splat(S52)*_d2r,      _bi=splat(S51)*_d1i+splat(S52)*_d2i; \
      (y1r)=_ar+_bi; (y1i)=_ai-_br; (y4r)=_ar-_bi; (y4i)=_ai+_br; } \
    { v8 _ar=_x0r+splat(C52)*_s1r+splat(C51)*_s2r, _ai=_x0i+splat(C52)*_s1i+splat(C51)*_s2i; \
      v8 _br=splat(S52)*_d1r-splat(S51)*_d2r,      _bi=splat(S52)*_d1i-splat(S51)*_d2i; \
      (y2r)=_ar+_bi; (y2i)=_ai-_br; (y3r)=_ar-_bi; (y3i)=_ai+_br; } }

// 8-point DFT on arrays xr[8], xi[8] (in-place, natural order in and out)
AI void dft8(v8 *xr, v8 *xi){
    v8 a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i;
    v8 b0r,b0i,b1r,b1i,b2r,b2i,b3r,b3i;
    a0r=xr[0]+xr[4]; a0i=xi[0]+xi[4]; b0r=xr[0]-xr[4]; b0i=xi[0]-xi[4];
    a1r=xr[1]+xr[5]; a1i=xi[1]+xi[5]; b1r=xr[1]-xr[5]; b1i=xi[1]-xi[5];
    a2r=xr[2]+xr[6]; a2i=xi[2]+xi[6]; b2r=xr[2]-xr[6]; b2i=xi[2]-xi[6];
    a3r=xr[3]+xr[7]; a3i=xi[3]+xi[7]; b3r=xr[3]-xr[7]; b3i=xi[3]-xi[7];
    // even: DFT4(a0,a1,a2,a3) -> X0 X2 X4 X6
    DFT4(xr[0],xi[0],xr[2],xi[2],xr[4],xi[4],xr[6],xi[6],
         a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i);
    // odd: twiddle then DFT4 -> X1 X3 X5 X7
    v8 t1r=splat(K8)*(b1r+b1i), t1i=splat(K8)*(b1i-b1r);
    v8 t2r=b2i, t2i=-b2r;
    v8 t3r=splat(K8)*(b3i-b3r), t3i=-splat(K8)*(b3r+b3i);
    DFT4(xr[1],xi[1],xr[3],xi[3],xr[5],xi[5],xr[7],xi[7],
         b0r,b0i,t1r,t1i,t2r,t2i,t3r,t3i);
}

// 9-point DFT on arrays r[9], i[9] (in-place, natural order)
AI void dft9(v8 *r, v8 *i){
    v8 A0r[3],A0i[3],A1r[3],A1i[3],A2r[3],A2i[3];
    DFT3(A0r[0],A0i[0],A0r[1],A0i[1],A0r[2],A0i[2], r[0],i[0],r[3],i[3],r[6],i[6]);
    DFT3(A1r[0],A1i[0],A1r[1],A1i[1],A1r[2],A1i[2], r[1],i[1],r[4],i[4],r[7],i[7]);
    DFT3(A2r[0],A2i[0],A2r[1],A2i[1],A2r[2],A2i[2], r[2],i[2],r[5],i[5],r[8],i[8]);
    v8 tr,ti;
    CMUL(tr,ti, A1r[1],A1i[1], splat(W91r),splat(W91i)); A1r[1]=tr; A1i[1]=ti;
    CMUL(tr,ti, A1r[2],A1i[2], splat(W92r),splat(W92i)); A1r[2]=tr; A1i[2]=ti;
    CMUL(tr,ti, A2r[1],A2i[1], splat(W92r),splat(W92i)); A2r[1]=tr; A2i[1]=ti;
    CMUL(tr,ti, A2r[2],A2i[2], splat(W94r),splat(W94i)); A2r[2]=tr; A2i[2]=ti;
    // outer DFT3 over j1 at each k1; X[k1+3*k2]
    DFT3(r[0],i[0],r[3],i[3],r[6],i[6], A0r[0],A0i[0],A1r[0],A1i[0],A2r[0],A2i[0]);
    DFT3(r[1],i[1],r[4],i[4],r[7],i[7], A0r[1],A0i[1],A1r[1],A1i[1],A2r[1],A2i[1]);
    DFT3(r[2],i[2],r[5],i[5],r[8],i[8], A0r[2],A0i[2],A1r[2],A1i[2],A2r[2],A2i[2]);
}

// ---------------------------------------------------------------- full-size line kernels
// data: element t at p[2*st*t] (re) and p[2*st*t+1] (im); in-place.

AI void k6(v8 *restrict p, const long st){
    // PFA(2,3): n[n1][n2] = {{0,2,4},{3,5,1}}, k[k1][k2] = {{0,4,2},{3,1,5}}
    v8 e0r,e0i,e1r,e1i,e2r,e2i, o0r,o0i,o1r,o1i,o2r,o2i;
    DFT2(e0r,e0i,o0r,o0i, p[0],p[1],p[2*st*3],p[2*st*3+1]);
    DFT2(e1r,e1i,o1r,o1i, p[2*st*2],p[2*st*2+1],p[2*st*5],p[2*st*5+1]);
    DFT2(e2r,e2i,o2r,o2i, p[2*st*4],p[2*st*4+1],p[2*st*1],p[2*st*1+1]);
    DFT3(p[0],p[1],p[2*st*4],p[2*st*4+1],p[2*st*2],p[2*st*2+1], e0r,e0i,e1r,e1i,e2r,e2i);
    DFT3(p[2*st*3],p[2*st*3+1],p[2*st*1],p[2*st*1+1],p[2*st*5],p[2*st*5+1], o0r,o0i,o1r,o1i,o2r,o2i);
}

AI void k8(v8 *restrict p, const long st){
    v8 r[8], i[8];
    #pragma GCC unroll 8
    for(int t=0;t<8;t++){ r[t]=p[2*st*t]; i[t]=p[2*st*t+1]; }
    dft8(r,i);
    #pragma GCC unroll 8
    for(int t=0;t<8;t++){ p[2*st*t]=r[t]; p[2*st*t+1]=i[t]; }
}

// direct symmetric prime kernel, 4-way k-blocked (FMA-bound)
#define PK_ACC4(HH,T0,T1,T2,T3) \
        _Pragma("GCC unroll 16") \
        for(int j=1;j<=HH;j++){ \
            v8 _sr=sr[j], _si=si[j], _dr=dr[j], _di=di[j]; \
            v8 c0=splat(T0[2*j-2]), s0=splat(T0[2*j-1]); \
            v8 c1=splat(T1[2*j-2]), s1=splat(T1[2*j-1]); \
            v8 c2=splat(T2[2*j-2]), s2=splat(T2[2*j-1]); \
            v8 c3=splat(T3[2*j-2]), s3=splat(T3[2*j-1]); \
            ur0+=c0*_sr; ui0+=c0*_si; vr0+=s0*_dr; vi0+=s0*_di; \
            ur1+=c1*_sr; ui1+=c1*_si; vr1+=s1*_dr; vi1+=s1*_di; \
            ur2+=c2*_sr; ui2+=c2*_si; vr2+=s2*_dr; vi2+=s2*_di; \
            ur3+=c3*_sr; ui3+=c3*_si; vr3+=s3*_dr; vi3+=s3*_di; \
        }
#define PK_OUT(LL,K, UR,UI,VR,VI) \
        p[2*st*(K)]=UR+VI;        p[2*st*(K)+1]=UI-VR; \
        p[2*st*(LL-(K))]=UR-VI;   p[2*st*(LL-(K))+1]=UI+VR;

#define PRIME_KERNEL(NAME, LL, HH, TABLE) \
AI void NAME(v8 *restrict p, const long st){ \
    v8 sr[HH+1], si[HH+1], dr[HH+1], di[HH+1]; \
    const v8 x0r=p[0], x0i=p[1]; \
    v8 o0r=x0r, o0i=x0i; \
    _Pragma("GCC unroll 16") \
    for(int j=1;j<=HH;j++){ \
        v8 ar=p[2*st*j], ai=p[2*st*j+1]; \
        v8 br=p[2*st*(LL-j)], bi=p[2*st*(LL-j)+1]; \
        sr[j]=ar+br; si[j]=ai+bi; dr[j]=ar-br; di[j]=ai-bi; \
        o0r+=sr[j]; o0i+=si[j]; \
    } \
    p[0]=o0r; p[1]=o0i; \
    int k=1; \
    for(; k+3<=HH; k+=4){ \
        const double *t0 = TABLE + (k-1)*HH*2 - 0; \
        const double *t1 = t0 + HH*2; \
        const double *t2 = t1 + HH*2; \
        const double *t3 = t2 + HH*2; \
        v8 ur0=x0r, ui0=x0i, vr0=VZERO, vi0=VZERO; \
        v8 ur1=x0r, ui1=x0i, vr1=VZERO, vi1=VZERO; \
        v8 ur2=x0r, ui2=x0i, vr2=VZERO, vi2=VZERO; \
        v8 ur3=x0r, ui3=x0i, vr3=VZERO, vi3=VZERO; \
        PK_ACC4(HH,t0,t1,t2,t3) \
        PK_OUT(LL,k,   ur0,ui0,vr0,vi0) \
        PK_OUT(LL,k+1, ur1,ui1,vr1,vi1) \
        PK_OUT(LL,k+2, ur2,ui2,vr2,vi2) \
        PK_OUT(LL,k+3, ur3,ui3,vr3,vi3) \
    } \
    for(; k+2<=HH; k+=3){ \
        const double *t0 = TABLE + (k-1)*HH*2; \
        const double *t1 = t0 + HH*2; \
        const double *t2 = t1 + HH*2; \
        v8 ur0=x0r, ui0=x0i, vr0=VZERO, vi0=VZERO; \
        v8 ur1=x0r, ui1=x0i, vr1=VZERO, vi1=VZERO; \
        v8 ur2=x0r, ui2=x0i, vr2=VZERO, vi2=VZERO; \
        _Pragma("GCC unroll 16") \
        for(int j=1;j<=HH;j++){ \
            v8 _sr=sr[j], _si=si[j], _dr=dr[j], _di=di[j]; \
            v8 c0=splat(t0[2*j-2]), s0=splat(t0[2*j-1]); \
            v8 c1=splat(t1[2*j-2]), s1=splat(t1[2*j-1]); \
            v8 c2=splat(t2[2*j-2]), s2=splat(t2[2*j-1]); \
            ur0+=c0*_sr; ui0+=c0*_si; vr0+=s0*_dr; vi0+=s0*_di; \
            ur1+=c1*_sr; ui1+=c1*_si; vr1+=s1*_dr; vi1+=s1*_di; \
            ur2+=c2*_sr; ui2+=c2*_si; vr2+=s2*_dr; vi2+=s2*_di; \
        } \
        PK_OUT(LL,k,   ur0,ui0,vr0,vi0) \
        PK_OUT(LL,k+1, ur1,ui1,vr1,vi1) \
        PK_OUT(LL,k+2, ur2,ui2,vr2,vi2) \
    } \
    for(; k+1<=HH; k+=2){ \
        const double *t0 = TABLE + (k-1)*HH*2; \
        const double *t1 = t0 + HH*2; \
        v8 ur0=x0r, ui0=x0i, vr0=VZERO, vi0=VZERO; \
        v8 ur1=x0r, ui1=x0i, vr1=VZERO, vi1=VZERO; \
        _Pragma("GCC unroll 16") \
        for(int j=1;j<=HH;j++){ \
            v8 _sr=sr[j], _si=si[j], _dr=dr[j], _di=di[j]; \
            v8 c0=splat(t0[2*j-2]), s0=splat(t0[2*j-1]); \
            v8 c1=splat(t1[2*j-2]), s1=splat(t1[2*j-1]); \
            ur0+=c0*_sr; ui0+=c0*_si; vr0+=s0*_dr; vi0+=s0*_di; \
            ur1+=c1*_sr; ui1+=c1*_si; vr1+=s1*_dr; vi1+=s1*_di; \
        } \
        PK_OUT(LL,k,   ur0,ui0,vr0,vi0) \
        PK_OUT(LL,k+1, ur1,ui1,vr1,vi1) \
    } \
    for(; k<=HH; k++){ \
        const double *t0 = TABLE + (k-1)*HH*2; \
        v8 ur=x0r, ui=x0i, vr=VZERO, vi=VZERO; \
        _Pragma("GCC unroll 16") \
        for(int j=1;j<=HH;j++){ \
            v8 c=splat(t0[2*j-2]), s=splat(t0[2*j-1]); \
            ur+=c*sr[j]; ui+=c*si[j]; vr+=s*dr[j]; vi+=s*di[j]; \
        } \
        PK_OUT(LL,k, ur,ui,vr,vi) \
    } \
}

PRIME_KERNEL(k13, 13, 6, tab13)
PRIME_KERNEL(k17, 17, 8, tab17)
PRIME_KERNEL(k23, 23, 11, tab23)

// 36 = PFA(4,9)
static const int IN36[4][9]={{0,4,8,12,16,20,24,28,32},{9,13,17,21,25,29,33,1,5},{18,22,26,30,34,2,6,10,14},{27,31,35,3,7,11,15,19,23}};
static const int K36T[4][9]={{0,28,20,12,4,32,24,16,8},{9,1,29,21,13,5,33,25,17},{18,10,2,30,22,14,6,34,26},{27,19,11,3,31,23,15,7,35}};
#define K36GEN(NAME, XCOPY) \
AI void NAME(v8 *restrict p, const long st){ \
    v8 Xb[XCOPY?72:1]; \
    const v8 *restrict X; \
    if(XCOPY){ \
        _Pragma("GCC unroll 36") \
        for(int j=0;j<36;j++){ Xb[2*j]=p[2*st*j]; Xb[2*j+1]=p[2*st*j+1]; } \
        X=Xb; \
    } else X=p; \
    v8 Tr[4][9], Ti[4][9]; \
    _Pragma("GCC unroll 9") \
    for(int n2=0;n2<9;n2++){ \
        v8 x0r=X[2*IN36[0][n2]], x0i=X[2*IN36[0][n2]+1]; \
        v8 x1r=X[2*IN36[1][n2]], x1i=X[2*IN36[1][n2]+1]; \
        v8 x2r=X[2*IN36[2][n2]], x2i=X[2*IN36[2][n2]+1]; \
        v8 x3r=X[2*IN36[3][n2]], x3i=X[2*IN36[3][n2]+1]; \
        DFT4(Tr[0][n2],Ti[0][n2],Tr[1][n2],Ti[1][n2],Tr[2][n2],Ti[2][n2],Tr[3][n2],Ti[3][n2], \
             x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i); \
    } \
    _Pragma("GCC unroll 4") \
    for(int k1=0;k1<4;k1++){ \
        dft9(Tr[k1], Ti[k1]); \
        _Pragma("GCC unroll 9") \
        for(int k2=0;k2<9;k2++){ \
            p[2*st*K36T[k1][k2]]   = Tr[k1][k2]; \
            p[2*st*K36T[k1][k2]+1] = Ti[k1][k2]; \
        } \
    } \
}
K36GEN(k36, 1)
K36GEN(k36d, 0)
// 45 = PFA(5,9)
static const int IN45[5][9]={{0,5,10,15,20,25,30,35,40},{9,14,19,24,29,34,39,44,4},{18,23,28,33,38,43,3,8,13},{27,32,37,42,2,7,12,17,22},{36,41,1,6,11,16,21,26,31}};
static const int K45T[5][9]={{0,10,20,30,40,5,15,25,35},{36,1,11,21,31,41,6,16,26},{27,37,2,12,22,32,42,7,17},{18,28,38,3,13,23,33,43,8},{9,19,29,39,4,14,24,34,44}};
#define K45GEN(NAME, XCOPY) \
AI void NAME(v8 *restrict p, const long st){ \
    v8 Xb[XCOPY?90:1]; \
    const v8 *restrict X; \
    if(XCOPY){ \
        _Pragma("GCC unroll 45") \
        for(int j=0;j<45;j++){ Xb[2*j]=p[2*st*j]; Xb[2*j+1]=p[2*st*j+1]; } \
        X=Xb; \
    } else X=p; \
    v8 Tr[5][9], Ti[5][9]; \
    _Pragma("GCC unroll 9") \
    for(int n2=0;n2<9;n2++){ \
        v8 x0r=X[2*IN45[0][n2]], x0i=X[2*IN45[0][n2]+1]; \
        v8 x1r=X[2*IN45[1][n2]], x1i=X[2*IN45[1][n2]+1]; \
        v8 x2r=X[2*IN45[2][n2]], x2i=X[2*IN45[2][n2]+1]; \
        v8 x3r=X[2*IN45[3][n2]], x3i=X[2*IN45[3][n2]+1]; \
        v8 x4r=X[2*IN45[4][n2]], x4i=X[2*IN45[4][n2]+1]; \
        DFT5(Tr[0][n2],Ti[0][n2],Tr[1][n2],Ti[1][n2],Tr[2][n2],Ti[2][n2],Tr[3][n2],Ti[3][n2],Tr[4][n2],Ti[4][n2], \
             x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i); \
    } \
    _Pragma("GCC unroll 5") \
    for(int k1=0;k1<5;k1++){ \
        dft9(Tr[k1], Ti[k1]); \
        _Pragma("GCC unroll 9") \
        for(int k2=0;k2<9;k2++){ \
            p[2*st*K45T[k1][k2]]   = Tr[k1][k2]; \
            p[2*st*K45T[k1][k2]+1] = Ti[k1][k2]; \
        } \
    } \
}
K45GEN(k45, 1)
K45GEN(k45d, 0)
// 64 = 8 x 8 Cooley-Tukey. K64GEN(name, 1) copies inputs up-front (better for
// strided/missing loads); K64GEN(name, 0) reads the (dense, L1-hot) input directly.
#define K64GEN(NAME, XCOPY) \
AI void NAME(v8 *restrict p, const long st){ \
    v8 Xb[XCOPY?128:1]; \
    const v8 *restrict X; \
    if(XCOPY){ \
        _Pragma("GCC unroll 64") \
        for(int j=0;j<64;j++){ Xb[2*j]=p[2*st*j]; Xb[2*j+1]=p[2*st*j+1]; } \
        X=Xb; \
    } else X=p; \
    v8 T[128]; \
    _Pragma("GCC unroll 8") \
    for(int j1=0;j1<8;j1++){ \
        v8 r[8], i[8]; \
        _Pragma("GCC unroll 8") \
        for(int t=0;t<8;t++){ r[t]=X[(8*t+j1)*2]; i[t]=X[(8*t+j1)*2+1]; } \
        dft8(r,i); \
        if(j1==0){ \
            _Pragma("GCC unroll 8") \
            for(int k1=0;k1<8;k1++){ T[(k1*8)*2]=r[k1]; T[(k1*8)*2+1]=i[k1]; } \
        }else{ \
            T[(0*8+j1)*2]=r[0]; T[(0*8+j1)*2+1]=i[0]; \
            _Pragma("GCC unroll 8") \
            for(int k1=1;k1<8;k1++){ \
                v8 wr=splat(tw64[(j1*8+k1)*2]), wi=splat(tw64[(j1*8+k1)*2+1]); \
                v8 tr,ti; CMUL(tr,ti, r[k1],i[k1], wr,wi); \
                T[(k1*8+j1)*2]=tr; T[(k1*8+j1)*2+1]=ti; \
            } \
        } \
    } \
    _Pragma("GCC unroll 8") \
    for(int k1=0;k1<8;k1++){ \
        v8 r[8], i[8]; \
        _Pragma("GCC unroll 8") \
        for(int t=0;t<8;t++){ r[t]=T[(k1*8+t)*2]; i[t]=T[(k1*8+t)*2+1]; } \
        dft8(r,i); \
        _Pragma("GCC unroll 8") \
        for(int k2=0;k2<8;k2++){ p[2*st*(k1+8*k2)]=r[k2]; p[2*st*(k1+8*k2)+1]=i[k2]; } \
    } \
}
K64GEN(k64, 1)
K64GEN(k64d, 0)
// ---------------------------------------------------------------- 8x8 transpose of v8 rows (in-place)
AI void tr8(v8 *a){
    v8 u0=__builtin_shufflevector(a[0],a[1],0,8,2,10,4,12,6,14);
    v8 u1=__builtin_shufflevector(a[0],a[1],1,9,3,11,5,13,7,15);
    v8 u2=__builtin_shufflevector(a[2],a[3],0,8,2,10,4,12,6,14);
    v8 u3=__builtin_shufflevector(a[2],a[3],1,9,3,11,5,13,7,15);
    v8 u4=__builtin_shufflevector(a[4],a[5],0,8,2,10,4,12,6,14);
    v8 u5=__builtin_shufflevector(a[4],a[5],1,9,3,11,5,13,7,15);
    v8 u6=__builtin_shufflevector(a[6],a[7],0,8,2,10,4,12,6,14);
    v8 u7=__builtin_shufflevector(a[6],a[7],1,9,3,11,5,13,7,15);
    v8 w0=__builtin_shufflevector(u0,u2,0,1,8,9,4,5,12,13);
    v8 w1=__builtin_shufflevector(u1,u3,0,1,8,9,4,5,12,13);
    v8 w2=__builtin_shufflevector(u0,u2,2,3,10,11,6,7,14,15);
    v8 w3=__builtin_shufflevector(u1,u3,2,3,10,11,6,7,14,15);
    v8 w4=__builtin_shufflevector(u4,u6,0,1,8,9,4,5,12,13);
    v8 w5=__builtin_shufflevector(u5,u7,0,1,8,9,4,5,12,13);
    v8 w6=__builtin_shufflevector(u4,u6,2,3,10,11,6,7,14,15);
    v8 w7=__builtin_shufflevector(u5,u7,2,3,10,11,6,7,14,15);
    a[0]=__builtin_shufflevector(w0,w4,0,1,2,3,8,9,10,11);
    a[4]=__builtin_shufflevector(w0,w4,4,5,6,7,12,13,14,15);
    a[1]=__builtin_shufflevector(w1,w5,0,1,2,3,8,9,10,11);
    a[5]=__builtin_shufflevector(w1,w5,4,5,6,7,12,13,14,15);
    a[2]=__builtin_shufflevector(w2,w6,0,1,2,3,8,9,10,11);
    a[6]=__builtin_shufflevector(w2,w6,4,5,6,7,12,13,14,15);
    a[3]=__builtin_shufflevector(w3,w7,0,1,2,3,8,9,10,11);
    a[7]=__builtin_shufflevector(w3,w7,4,5,6,7,12,13,14,15);
}
// ---------------------------------------------------------------- buffers
static v8 *SBUF, *CBUF;               // big shared scratch (one unit: group or volume) + c
#define MAXBLK (64*512)               // largest unit in blocks (L=64 volume)
static void *xalloc(size_t bytes){
    size_t sz=(bytes+2097151)&~(size_t)2097151;
    void *p=mmap(0, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if(p==MAP_FAILED){ p=aligned_alloc(64,sz); }
    else madvise(p, sz, MADV_HUGEPAGE);
    memset(p,0,sz);
    return p;
}
void init_buffers(void){
    SBUF=(v8*)xalloc((size_t)MAXBLK*128);
    CBUF=(v8*)((char*)xalloc((size_t)MAXBLK*128 + 65536) + 2176+4096);  /* decorrelate sets vs SBUF */
}

// ---------------------------------------------------------------- interleave helpers
AI void deint(const double *s, v8 *re, v8 *im){ // 8 complex -> split
    v8 v0=*(const v8u*)s, v1=*(const v8u*)(s+8);
    *re=__builtin_shufflevector(v0,v1,0,2,4,6,8,10,12,14);
    *im=__builtin_shufflevector(v0,v1,1,3,5,7,9,11,13,15);
}
AI void ileav(double *d, v8 re, v8 im){ // split -> 8 complex
    *(v8u*)d     = __builtin_shufflevector(re,im,0,8,1,9,2,10,3,11);
    *(v8u*)(d+8) = __builtin_shufflevector(re,im,4,12,5,13,6,14,7,15);
}

// ---------------------------------------------------------------- BL (batch-lane): lanes = 8 volumes
#define GEN_BL(SFX, LL, KFN) \
static void bl_step_##SFX(v8 *restrict S, const v8 *restrict C){ \
    const long L2=(long)LL*LL; \
    for(long yz=0; yz<L2; yz++) KFN(S + yz*2, L2);      /* x lines */ \
    for(long x=0;x<LL;x++){ \
        v8 *pl = S + x*L2*2; \
        const v8 *cpl = C + x*L2*2; \
        for(long y=0;y<LL;y++) KFN(pl + y*LL*2, 1);     /* z lines */ \
        for(long z=0;z<LL;z++) KFN(pl + z*2, LL);       /* y lines */ \
        map_blocks(pl, cpl, L2); \
    } \
}
GEN_BL(6, 6, k6)
GEN_BL(8, 8, k8)
GEN_BL(13, 13, k13)
GEN_BL(17, 17, k17)
GEN_BL(23, 23, k23)

static void bl_in(const double *restrict x0, long B, long g, v8 *restrict S, long L3){
    long e8=L3&~7L;
    for(long e=0;e<e8;e+=8){
        v8 R[8], I[8];
        for(int v=0;v<8;v++){
            long vol=g*8+v;
            if(vol<B) deint(x0+vol*L3*2+e*2, &R[v], &I[v]);
            else { R[v]=VZERO; I[v]=VZERO; }
        }
        tr8(R); tr8(I);
        for(int t=0;t<8;t++){ S[(e+t)*2]=R[t]; S[(e+t)*2+1]=I[t]; }
    }
    for(long e=e8;e<L3;e++){
        double *d=(double*)(S+e*2);
        for(int v=0;v<8;v++){
            long vol=g*8+v;
            if(vol<B){ d[v]=x0[vol*L3*2+2*e]; d[8+v]=x0[vol*L3*2+2*e+1]; }
            else { d[v]=0.0; d[8+v]=0.0; }
        }
    }
}
static void bl_out(double *restrict o, long B, long g, const v8 *restrict S, long L3){
    long e8=L3&~7L, nv = B-g*8; if(nv>8) nv=8;
    for(long e=0;e<e8;e+=8){
        v8 R[8], I[8];
        for(int t=0;t<8;t++){ R[t]=S[(e+t)*2]; I[t]=S[(e+t)*2+1]; }
        tr8(R); tr8(I);
        for(int v=0;v<nv;v++) ileav(o+(g*8+v)*L3*2+e*2, R[v], I[v]);
    }
    for(long e=e8;e<L3;e++){
        const double *d=(const double*)(S+e*2);
        for(int v=0;v<nv;v++){ o[(g*8+v)*L3*2+2*e]=d[v]; o[(g*8+v)*L3*2+2*e+1]=d[8+v]; }
    }
}
static void bl_run(long LL, long B, long m, const double *x0, const double *c,
                   double *o1, double *of, void(*step)(v8*,const v8*)){
    long L3=LL*LL*LL, G=(B+7)/8;
    for(long g=0; g<G; g++){
        bl_in(x0,B,g,SBUF,L3);
        bl_in(c,B,g,CBUF,L3);
        for(long s=0;s<m;s++){
            step(SBUF,CBUF);
            if(s==0 && m>1) bl_out(o1,B,g,SBUF,L3);
        }
        bl_out(of,B,g,SBUF,L3);
    }
    if(m==1) memcpy(o1, of, (size_t)B*L3*16);
}

// ---------------------------------------------------------------- PV (per-volume): lanes = z chunks
#define GEN_PV(SFX, LL, RSY, RB, KFN, KFND, PF) \
static void pv_step_##SFX(v8 *restrict V, const v8 *restrict C){ \
    const long PB=(long)RSY*RB, NSTR=((LL)+7)/8; \
    for(long y=0;y<LL;y++) \
        for(long zc=0; zc<RB; zc++) KFN(V + (y*RB+zc)*2, PB);   /* x lines */ \
    for(long x=0;x<LL;x++){ \
        v8 *pl = V + x*PB*2; \
        const v8 *cpl = C + x*PB*2; \
        for(long zc=0; zc<RB; zc++) KFN(pl + zc*2, RB);         /* y lines */ \
        for(long s=0; s<NSTR; s++){                              /* z lines */ \
            const long y0 = (s<NSTR-1)? 8*s : (long)RSY-8; \
            const long skip = (s<NSTR-1)? 0 : 8*NSTR-(long)RSY; \
            v8 buf[RB*8*2]; \
            for(long t=0;t<RB;t++){ \
                v8 ar[8], ai[8]; \
                for(int e=0;e<8;e++){ ar[e]=pl[((y0+e)*RB+t)*2]; ai[e]=pl[((y0+e)*RB+t)*2+1]; } \
                tr8(ar); tr8(ai); \
                for(int e=0;e<8;e++){ buf[(t*8+e)*2]=ar[e]; buf[(t*8+e)*2+1]=ai[e]; } \
            } \
            KFND(buf, 1); \
            for(long t=0;t<RB;t++){ \
                v8 ar[8], ai[8]; \
                for(int e=0;e<8;e++){ ar[e]=buf[(t*8+e)*2]; ai[e]=buf[(t*8+e)*2+1]; } \
                tr8(ar); tr8(ai); \
                for(int e=skip;e<8;e++){ pl[((y0+e)*RB+t)*2]=ar[e]; pl[((y0+e)*RB+t)*2+1]=ai[e]; } \
            } \
        } \
        map_blocks(pl, cpl, (long)LL*RB);                        /* +c, map */ \
    } \
}
GEN_PV(6,  6,  8, 1, k6,  k6,   0)
GEN_PV(8,  8,  8, 1, k8,  k8,   0)
GEN_PV(13, 13, 13, 2, k13, k13, 0)
GEN_PV(17, 17, 17, 3, k17, k17, 0)
GEN_PV(23, 23, 23, 3, k23, k23, 0)
GEN_PV(36, 36, 36, 5, k36, k36d, 0)
GEN_PV(45, 45, 45, 6, k45, k45d, 0)
GEN_PV(64, 64, 64, 8, k64, k64d, 0)

static void pv_in(const double *restrict src, v8 *restrict V, long LL, long RB, long PB){
    long full=LL/8, tail=LL%8;
    if(LL<8) memset(V, 0, (size_t)LL*PB*128);  /* L=6: pad rows too */
    for(long x=0;x<LL;x++) for(long y=0;y<LL;y++){
        const double *s = src + ((x*LL+y)*LL)*2;
        v8 *row = V + (x*PB + y*RB)*2;
        for(long zc=0; zc<full; zc++) deint(s+zc*16, &row[zc*2], &row[zc*2+1]);
        if(tail){
            double *rr=(double*)(row+full*2);
            for(long t=0;t<tail;t++){ rr[t]=s[(full*8+t)*2]; rr[8+t]=s[(full*8+t)*2+1]; }
            for(long t=tail;t<8;t++){ rr[t]=0.0; rr[8+t]=0.0; }   /* zero z-pad lanes */
        }
    }
}
static void pv_out(double *restrict dst, const v8 *restrict V, long LL, long RB, long PB){
    long full=LL/8, tail=LL%8;
    for(long x=0;x<LL;x++) for(long y=0;y<LL;y++){
        double *t_ = dst + ((x*LL+y)*LL)*2;
        const v8 *row = V + (x*PB + y*RB)*2;
        for(long zc=0; zc<full; zc++) ileav(t_+zc*16, row[zc*2], row[zc*2+1]);
        if(tail){
            const double *rr=(const double*)(row+full*2);
            for(long tt=0;tt<tail;tt++){ t_[(full*8+tt)*2]=rr[tt]; t_[(full*8+tt)*2+1]=rr[8+tt]; }
        }
    }
}
static void pv_run(long LL, long RB, long PB, long B, long m, const double *x0, const double *c,
                   double *o1, double *of, void(*step)(v8*,const v8*)){
    long L3=LL*LL*LL;
    for(long v=0; v<B; v++){
        pv_in(x0+v*L3*2, SBUF, LL, RB, PB);
        pv_in(c+v*L3*2, CBUF, LL, RB, PB);
        for(long s=0;s<m;s++){
            step(SBUF,CBUF);
            if(s==0 && m>1) pv_out(o1+v*L3*2, SBUF, LL, RB, PB);
        }
        pv_out(of+v*L3*2, SBUF, LL, RB, PB);
    }
    if(m==1) memcpy(o1, of, (size_t)B*L3*16);
}

// ---------------------------------------------------------------- API
static int FORCE_SCHEME = 0;  // 0=auto, 1=BL, 2=PV  (bench hook)
void set_scheme(int s){ FORCE_SCHEME=s; }

typedef void (*stepfn)(v8*,const v8*);
void run_size(long L, long B, long m, const double *x0, const double *c,
              double *out_one, double *out_final){
    if(B<=0 || m<=0) return;
    stepfn bl=0, pv=0; long RB=0, PB=0, remt=0;
    switch(L){
        case 6:  bl=bl_step_6;  pv=pv_step_6;  RB=1; PB=8;   remt=4; break;
        case 8:  bl=bl_step_8;  pv=pv_step_8;  RB=1; PB=8;   remt=6; break;
        case 13: bl=bl_step_13; pv=pv_step_13; RB=2; PB=26;  remt=5; break;
        case 17: bl=bl_step_17; pv=pv_step_17; RB=3; PB=51;  remt=4; break;
        case 23: bl=bl_step_23; pv=pv_step_23; RB=3; PB=69;  remt=7; break;
        case 36: pv=pv_step_36; RB=5; PB=180; break;
        case 45: pv=pv_step_45; RB=6; PB=270; break;
        case 64: pv=pv_step_64; RB=8; PB=512; break;
    }
    if(!bl){ pv_run(L,RB,PB,B,m,x0,c,out_one,out_final,pv); return; }
    if(FORCE_SCHEME==1){ bl_run(L,B,m,x0,c,out_one,out_final,bl); return; }
    if(FORCE_SCHEME==2){ pv_run(L,RB,PB,B,m,x0,c,out_one,out_final,pv); return; }
    long L3=L*L*L, Gfull=B/8, r=B%8;
    if(r==0){ bl_run(L,B,m,x0,c,out_one,out_final,bl); return; }
    if(r<=remt){
        if(Gfull>0) bl_run(L,Gfull*8,m,x0,c,out_one,out_final,bl);
        long off=Gfull*8*L3*2;
        pv_run(L,RB,PB,r,m,x0+off,c+off,out_one+off,out_final+off,pv);
    }else{
        bl_run(L,B,m,x0,c,out_one,out_final,bl);
    }
}
void setup(void){ init_tables(); init_buffers(); }
