// Iterated batched 3D complex FFT + nonlinear map, specialized for
// L in {6,8,13,17,23,36,45,64}. Single-threaded, AVX-512 vertical SIMD.
// All transforms hand-written (no FFT libraries).
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>
#include <immintrin.h>

typedef double v8 __attribute__((vector_size(64), aligned(64)));
typedef double v8u __attribute__((vector_size(64), aligned(8)));
#define AI static inline __attribute__((always_inline))

AI v8 splat(double x){ return (v8){x,x,x,x,x,x,x,x}; }
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
    v8 mag = s*r;                 // = sqrt(s) to ~1ulp... refine below
    // one Heron touch-up for accuracy: mag = 0.5*(mag + s/mag) -- skip (r is 2xNR)
    v8 u = splat(1.0) + mag;
    v8 t = (v8)_mm512_rcp14_pd((__m512d)u);
    t = t*(splat(2.0) - u*t);
    t = t*(splat(2.0) - u*t);
    *zr *= t; *zi *= t;
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
    for(int t=0;t<8;t++){ r[t]=p[2*st*t]; i[t]=p[2*st*t+1]; }
    dft8(r,i);
    for(int t=0;t<8;t++){ p[2*st*t]=r[t]; p[2*st*t+1]=i[t]; }
}

// direct symmetric prime kernel
#define PRIME_KERNEL(NAME, LL, HH, TABLE) \
AI void NAME(v8 *restrict p, const long st){ \
    v8 sr[HH+1], si[HH+1], dr[HH+1], di[HH+1]; \
    const v8 x0r=p[0], x0i=p[1]; \
    v8 o0r=x0r, o0i=x0i; \
    for(int j=1;j<=HH;j++){ \
        v8 ar=p[2*st*j], ai=p[2*st*j+1]; \
        v8 br=p[2*st*(LL-j)], bi=p[2*st*(LL-j)+1]; \
        sr[j]=ar+br; si[j]=ai+bi; dr[j]=ar-br; di[j]=ai-bi; \
        o0r+=sr[j]; o0i+=si[j]; \
    } \
    p[0]=o0r; p[1]=o0i; \
    int k=1; \
    for(; k+1<=HH; k+=2){ \
        const double *t0 = TABLE + (k-1)*HH*2; \
        const double *t1 = TABLE + k*HH*2; \
        v8 ur0=x0r, ui0=x0i, vr0=VZERO, vi0=VZERO; \
        v8 ur1=x0r, ui1=x0i, vr1=VZERO, vi1=VZERO; \
        for(int j=1;j<=HH;j++){ \
            v8 _sr=sr[j], _si=si[j], _dr=dr[j], _di=di[j]; \
            v8 c0=splat(t0[2*(j-1)]), s0=splat(t0[2*(j-1)+1]); \
            v8 c1=splat(t1[2*(j-1)]), s1=splat(t1[2*(j-1)+1]); \
            ur0+=c0*_sr; ui0+=c0*_si; vr0+=s0*_dr; vi0+=s0*_di; \
            ur1+=c1*_sr; ui1+=c1*_si; vr1+=s1*_dr; vi1+=s1*_di; \
        } \
        p[2*st*k]=ur0+vi0;        p[2*st*k+1]=ui0-vr0; \
        p[2*st*(LL-k)]=ur0-vi0;   p[2*st*(LL-k)+1]=ui0+vr0; \
        p[2*st*(k+1)]=ur1+vi1;      p[2*st*(k+1)+1]=ui1-vr1; \
        p[2*st*(LL-k-1)]=ur1-vi1;   p[2*st*(LL-k-1)+1]=ui1+vr1; \
    } \
    for(; k<=HH; k++){ \
        const double *t0 = TABLE + (k-1)*HH*2; \
        v8 ur=x0r, ui=x0i, vr=VZERO, vi=VZERO; \
        for(int j=1;j<=HH;j++){ \
            v8 c=splat(t0[2*(j-1)]), s=splat(t0[2*(j-1)+1]); \
            ur+=c*sr[j]; ui+=c*si[j]; vr+=s*dr[j]; vi+=s*di[j]; \
        } \
        p[2*st*k]=ur+vi;        p[2*st*k+1]=ui-vr; \
        p[2*st*(LL-k)]=ur-vi;   p[2*st*(LL-k)+1]=ui+vr; \
    } \
}

PRIME_KERNEL(k13, 13, 6, tab13)
PRIME_KERNEL(k17, 17, 8, tab17)
PRIME_KERNEL(k23, 23, 11, tab23)

// 36 = PFA(4,9)
static const int IN36[4][9]={{0,4,8,12,16,20,24,28,32},{9,13,17,21,25,29,33,1,5},{18,22,26,30,34,2,6,10,14},{27,31,35,3,7,11,15,19,23}};
static const int K36T[4][9]={{0,28,20,12,4,32,24,16,8},{9,1,29,21,13,5,33,25,17},{18,10,2,30,22,14,6,34,26},{27,19,11,3,31,23,15,7,35}};
AI void k36(v8 *restrict p, const long st){
    v8 Tr[4][9], Ti[4][9];
    for(int n2=0;n2<9;n2++){
        v8 x0r=p[2*st*IN36[0][n2]], x0i=p[2*st*IN36[0][n2]+1];
        v8 x1r=p[2*st*IN36[1][n2]], x1i=p[2*st*IN36[1][n2]+1];
        v8 x2r=p[2*st*IN36[2][n2]], x2i=p[2*st*IN36[2][n2]+1];
        v8 x3r=p[2*st*IN36[3][n2]], x3i=p[2*st*IN36[3][n2]+1];
        DFT4(Tr[0][n2],Ti[0][n2],Tr[1][n2],Ti[1][n2],Tr[2][n2],Ti[2][n2],Tr[3][n2],Ti[3][n2],
             x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i);
    }
    for(int k1=0;k1<4;k1++){
        dft9(Tr[k1], Ti[k1]);
        for(int k2=0;k2<9;k2++){
            p[2*st*K36T[k1][k2]]   = Tr[k1][k2];
            p[2*st*K36T[k1][k2]+1] = Ti[k1][k2];
        }
    }
}

// 45 = PFA(5,9)
static const int IN45[5][9]={{0,5,10,15,20,25,30,35,40},{9,14,19,24,29,34,39,44,4},{18,23,28,33,38,43,3,8,13},{27,32,37,42,2,7,12,17,22},{36,41,1,6,11,16,21,26,31}};
static const int K45T[5][9]={{0,10,20,30,40,5,15,25,35},{36,1,11,21,31,41,6,16,26},{27,37,2,12,22,32,42,7,17},{18,28,38,3,13,23,33,43,8},{9,19,29,39,4,14,24,34,44}};
AI void k45(v8 *restrict p, const long st){
    v8 Tr[5][9], Ti[5][9];
    for(int n2=0;n2<9;n2++){
        v8 x0r=p[2*st*IN45[0][n2]], x0i=p[2*st*IN45[0][n2]+1];
        v8 x1r=p[2*st*IN45[1][n2]], x1i=p[2*st*IN45[1][n2]+1];
        v8 x2r=p[2*st*IN45[2][n2]], x2i=p[2*st*IN45[2][n2]+1];
        v8 x3r=p[2*st*IN45[3][n2]], x3i=p[2*st*IN45[3][n2]+1];
        v8 x4r=p[2*st*IN45[4][n2]], x4i=p[2*st*IN45[4][n2]+1];
        DFT5(Tr[0][n2],Ti[0][n2],Tr[1][n2],Ti[1][n2],Tr[2][n2],Ti[2][n2],Tr[3][n2],Ti[3][n2],Tr[4][n2],Ti[4][n2],
             x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i);
    }
    for(int k1=0;k1<5;k1++){
        dft9(Tr[k1], Ti[k1]);
        for(int k2=0;k2<9;k2++){
            p[2*st*K45T[k1][k2]]   = Tr[k1][k2];
            p[2*st*K45T[k1][k2]+1] = Ti[k1][k2];
        }
    }
}

// 64 = 8 x 8 Cooley-Tukey
AI void k64(v8 *restrict p, const long st){
    v8 T[128]; // [k1][j1] pairs at (k1*8+j1)*2
    for(int j1=0;j1<8;j1++){
        v8 r[8], i[8];
        for(int t=0;t<8;t++){ r[t]=p[2*st*(j1+8*t)]; i[t]=p[2*st*(j1+8*t)+1]; }
        dft8(r,i);
        if(j1==0){
            for(int k1=0;k1<8;k1++){ T[(k1*8)*2]=r[k1]; T[(k1*8)*2+1]=i[k1]; }
        }else{
            T[(0*8+j1)*2]=r[0]; T[(0*8+j1)*2+1]=i[0];
            for(int k1=1;k1<8;k1++){
                v8 wr=splat(tw64[(j1*8+k1)*2]), wi=splat(tw64[(j1*8+k1)*2+1]);
                v8 tr,ti; CMUL(tr,ti, r[k1],i[k1], wr,wi);
                T[(k1*8+j1)*2]=tr; T[(k1*8+j1)*2+1]=ti;
            }
        }
    }
    for(int k1=0;k1<8;k1++){
        v8 r[8], i[8];
        for(int t=0;t<8;t++){ r[t]=T[(k1*8+t)*2]; i[t]=T[(k1*8+t)*2+1]; }
        dft8(r,i);
        for(int k2=0;k2<8;k2++){ p[2*st*(k1+8*k2)]=r[k2]; p[2*st*(k1+8*k2)+1]=i[k2]; }
    }
}
