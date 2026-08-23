// Iterated batched 3D complex FFTs for fixed cube sizes {6,8,13,17,23,36,45,64}.
// Single-threaded AVX-512 (Ice Lake), split re/im planes, all DFT arithmetic
// hand-written (radix codelets for 6/8/36/45/64, symmetric half-matrix kernels
// for the primes 13/17/23; PFA index maps for 36=4x9 and 45=9x5; 64=8x8
// Cooley-Tukey). The elementwise map x <- z/(1+|z|), z = FFT3+c, is fused into
// the last pass (hardware sqrt + rcp14 with two Newton steps).
//
// Two execution modes, chosen per size/batch at runtime:
//  - per-volume (36,45,64 and small batches): iteration = pass ZY (per x-slab:
//    z-axis DFT via 8x8 register transposes in place, then y-axis DFT,
//    slab stays cache-hot) + pass X (x-axis DFT fused with +c and the map;
//    c is pre-arranged per volume in streaming order). Volumes are iterated
//    one at a time so the m iterations run cache-resident.
//  - grouped (6,8,13,17,23 when B>=8, incl. padded tail groups for B%8>=5):
//    8 volumes ride in the 8 SIMD lanes; all three axis passes use contiguous
//    vector loads, broadcast twiddles, no transposes and no masked tails.
//
// Precomputation (plan): trig tables via long double, PFA maps, aligned
// scratch arenas with stagger to avoid cache-set aliasing.

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/mman.h>

#define AI  static __attribute__((always_inline)) inline
#define NI  static __attribute__((noinline))

typedef __m512d V;
typedef struct { V re, im; } CV;

#define VB(x)      _mm512_set1_pd(x)
#define VLD(p)     _mm512_loadu_pd(p)
#define VST(p,v)   _mm512_storeu_pd((p),(v))
#define VSTM(p,m,v) _mm512_mask_storeu_pd((p),(m),(v))
AI V vadd(V a, V b){ return _mm512_add_pd(a,b); }
AI V vsub(V a, V b){ return _mm512_sub_pd(a,b); }
AI V vmul(V a, V b){ return _mm512_mul_pd(a,b); }
AI V vfma(V a, V b, V c){ return _mm512_fmadd_pd(a,b,c); }
AI V vfnma(V a, V b, V c){ return _mm512_fnmadd_pd(a,b,c); }

AI CV cvadd(CV a, CV b){ CV r; r.re=vadd(a.re,b.re); r.im=vadd(a.im,b.im); return r; }
AI CV cvsub(CV a, CV b){ CV r; r.re=vsub(a.re,b.re); r.im=vsub(a.im,b.im); return r; }
AI CV cvaddi(CV a, CV b){ CV r; r.re=vsub(a.re,b.im); r.im=vadd(a.im,b.re); return r; }
AI CV cvsubi(CV a, CV b){ CV r; r.re=vadd(a.re,b.im); r.im=vsub(a.im,b.re); return r; }
AI CV ldc(const double* re, const double* im, ptrdiff_t off){ CV r; r.re=VLD(re+off); r.im=VLD(im+off); return r; }

// ---------------------------------------------------------------------------
static double W3S, W8C, S5Q, S5_1, S5_2;
static double TW9r[3], TW9i[3];
static double C13[7], S13[7];
static double C17[9], S17[9];
static double C23[12], S23[12];
static double TW64r[64], TW64i[64];
static int    MAPJ36[36], MAPK36[36];
static int    MAPJ45[45], MAPK45[45];

static double *ARE,*AIM,*BRE,*BIM,*CRE2,*CIM2;
#define PLANE_DBL ((size_t)(64*64+16)*64)
static const long long IDX_RE_[8]={0,2,4,6,8,10,12,14};
static const long long IDX_IM_[8]={1,3,5,7,9,11,13,15};
static const long long IDX_LO_[8]={0,8,1,9,2,10,3,11};
static const long long IDX_HI_[8]={4,12,5,13,6,14,7,15};

// ---------------------------------------------------------------------------
AI void dft3c(CV* o0, CV* o1, CV* o2, CV x0, CV x1, CV x2){
  CV t1=cvadd(x1,x2), t2=cvsub(x1,x2);
  *o0=cvadd(x0,t1);
  CV m; m.re=vfnma(VB(0.5),t1.re,x0.re); m.im=vfnma(VB(0.5),t1.im,x0.im);
  CV n; n.re=vmul(VB(W3S),t2.re); n.im=vmul(VB(W3S),t2.im);
  *o1=cvsubi(m,n);
  *o2=cvaddi(m,n);
}
AI void dft4c(CV* o0, CV* o1, CV* o2, CV* o3, CV x0, CV x1, CV x2, CV x3){
  CV t0=cvadd(x0,x2), t1=cvsub(x0,x2), t2=cvadd(x1,x3), t3=cvsub(x1,x3);
  *o0=cvadd(t0,t2); *o2=cvsub(t0,t2);
  *o1=cvsubi(t1,t3); *o3=cvaddi(t1,t3);
}
AI void dft5c(CV* o0, CV* o1, CV* o2, CV* o3, CV* o4, CV x0, CV x1, CV x2, CV x3, CV x4){
  CV t1=cvadd(x1,x4), t3=cvsub(x1,x4);
  CV t2=cvadd(x2,x3), t4=cvsub(x2,x3);
  CV t5=cvadd(t1,t2);
  *o0=cvadd(x0,t5);
  CV bq; bq.re=vfnma(VB(0.25),t5.re,x0.re); bq.im=vfnma(VB(0.25),t5.im,x0.im);
  CV cc; cc.re=vmul(VB(S5Q),vsub(t1.re,t2.re)); cc.im=vmul(VB(S5Q),vsub(t1.im,t2.im));
  CV s1=cvadd(bq,cc), s2=cvsub(bq,cc);
  CV w1, w2;
  w1.re=vfma(VB(S5_2),t4.re,vmul(VB(S5_1),t3.re));
  w1.im=vfma(VB(S5_2),t4.im,vmul(VB(S5_1),t3.im));
  w2.re=vfnma(VB(S5_1),t4.re,vmul(VB(S5_2),t3.re));
  w2.im=vfnma(VB(S5_1),t4.im,vmul(VB(S5_2),t3.im));
  *o1=cvsubi(s1,w1); *o4=cvaddi(s1,w1);
  *o2=cvsubi(s2,w2); *o3=cvaddi(s2,w2);
}
AI void dft8c(CV* o, CV x0, CV x1, CV x2, CV x3, CV x4, CV x5, CV x6, CV x7){
  CV t0=cvadd(x0,x4), u0=cvsub(x0,x4);
  CV t1=cvadd(x1,x5), u1=cvsub(x1,x5);
  CV t2=cvadd(x2,x6), u2=cvsub(x2,x6);
  CV t3=cvadd(x3,x7), u3=cvsub(x3,x7);
  CV p0=cvadd(t0,t2), p1=cvsub(t0,t2), p2=cvadd(t1,t3), p3=cvsub(t1,t3);
  o[0]=cvadd(p0,p2); o[4]=cvsub(p0,p2);
  o[2]=cvsubi(p1,p3); o[6]=cvaddi(p1,p3);
  CV v1, v3;
  v1.re=vmul(VB(W8C), vadd(u1.re,u1.im)); v1.im=vmul(VB(W8C), vsub(u1.im,u1.re));
  v3.re=vmul(VB(W8C), vsub(u3.im,u3.re)); v3.im=vsub(_mm512_setzero_pd(), vmul(VB(W8C), vadd(u3.re,u3.im)));
  CV q0=cvsubi(u0,u2), q1=cvaddi(u0,u2), q2=cvadd(v1,v3), q3=cvsub(v1,v3);
  o[1]=cvadd(q0,q2); o[5]=cvsub(q0,q2);
  o[3]=cvsubi(q1,q3); o[7]=cvaddi(q1,q3);
}
AI void dft9c(CV* o, const CV* x){
  CV a0,a1,a2,b0,b1,b2,c0,c1,c2;
  dft3c(&a0,&a1,&a2, x[0],x[3],x[6]);
  dft3c(&b0,&b1,&b2, x[1],x[4],x[7]);
  dft3c(&c0,&c1,&c2, x[2],x[5],x[8]);
  { V wr=VB(TW9r[0]), wi=VB(TW9i[0]); CV t;
    t.re=vfnma(wi,b1.im,vmul(wr,b1.re)); t.im=vfma(wi,b1.re,vmul(wr,b1.im)); b1=t; }
  { V wr=VB(TW9r[1]), wi=VB(TW9i[1]); CV t;
    t.re=vfnma(wi,b2.im,vmul(wr,b2.re)); t.im=vfma(wi,b2.re,vmul(wr,b2.im)); b2=t;
    t.re=vfnma(wi,c1.im,vmul(wr,c1.re)); t.im=vfma(wi,c1.re,vmul(wr,c1.im)); c1=t; }
  { V wr=VB(TW9r[2]), wi=VB(TW9i[2]); CV t;
    t.re=vfnma(wi,c2.im,vmul(wr,c2.re)); t.im=vfma(wi,c2.re,vmul(wr,c2.im)); c2=t; }
  dft3c(&o[0],&o[3],&o[6], a0,b0,c0);
  dft3c(&o[1],&o[4],&o[7], a1,b1,c1);
  dft3c(&o[2],&o[5],&o[8], a2,b2,c2);
}

// 8x8 transpose
AI void tr8(V* r){
  V t0=_mm512_unpacklo_pd(r[0],r[1]), t1=_mm512_unpackhi_pd(r[0],r[1]);
  V t2=_mm512_unpacklo_pd(r[2],r[3]), t3=_mm512_unpackhi_pd(r[2],r[3]);
  V t4=_mm512_unpacklo_pd(r[4],r[5]), t5=_mm512_unpackhi_pd(r[4],r[5]);
  V t6=_mm512_unpacklo_pd(r[6],r[7]), t7=_mm512_unpackhi_pd(r[6],r[7]);
  V u0=_mm512_shuffle_f64x2(t0,t2,0x88), u1=_mm512_shuffle_f64x2(t0,t2,0xDD);
  V u2=_mm512_shuffle_f64x2(t1,t3,0x88), u3=_mm512_shuffle_f64x2(t1,t3,0xDD);
  V u4=_mm512_shuffle_f64x2(t4,t6,0x88), u5=_mm512_shuffle_f64x2(t4,t6,0xDD);
  V u6=_mm512_shuffle_f64x2(t5,t7,0x88), u7=_mm512_shuffle_f64x2(t5,t7,0xDD);
  r[0]=_mm512_shuffle_f64x2(u0,u4,0x88); r[4]=_mm512_shuffle_f64x2(u0,u4,0xDD);
  r[1]=_mm512_shuffle_f64x2(u2,u6,0x88); r[5]=_mm512_shuffle_f64x2(u2,u6,0xDD);
  r[2]=_mm512_shuffle_f64x2(u1,u5,0x88); r[6]=_mm512_shuffle_f64x2(u1,u5,0xDD);
  r[3]=_mm512_shuffle_f64x2(u3,u7,0x88); r[7]=_mm512_shuffle_f64x2(u3,u7,0xDD);
}

// elementwise map z <- z/(1+|z|)
AI void mapop(V* zr, V* zi){
  V m2=vfma(*zi,*zi,vmul(*zr,*zr));
  V s=_mm512_sqrt_pd(m2);
  V d=vadd(VB(1.0),s);
  V w=_mm512_rcp14_pd(d);
  w=vmul(w,vfnma(d,w,VB(2.0)));
  w=vmul(w,vfnma(d,w,VB(2.0)));
  *zr=vmul(*zr,w);
  *zi=vmul(*zi,w);
}

// ---------------------------------------------------------------------------
// CORE bodies: load at (re,im)+j*S, emit STOREK(k, CV-value)

#define CORE6(S, STOREK) do{ \
  CV x0=ldc(re,im,0), x1=ldc(re,im,S), x2=ldc(re,im,2*(ptrdiff_t)(S)), x3=ldc(re,im,3*(ptrdiff_t)(S)), x4=ldc(re,im,4*(ptrdiff_t)(S)), x5=ldc(re,im,5*(ptrdiff_t)(S)); \
  CV a0=cvadd(x0,x3), d0=cvsub(x0,x3); \
  CV a1=cvadd(x2,x5), d1=cvsub(x2,x5); \
  CV a2=cvadd(x4,x1), d2=cvsub(x4,x1); \
  CV e0,e1,e2,f0,f1,f2; \
  dft3c(&e0,&e1,&e2, a0,a1,a2); \
  dft3c(&f0,&f1,&f2, d0,d1,d2); \
  STOREK(0,e0); STOREK(4,e1); STOREK(2,e2); \
  STOREK(3,f0); STOREK(1,f1); STOREK(5,f2); \
}while(0)

#define CORE8(S, STOREK) do{ \
  CV oo[8]; \
  dft8c(oo, ldc(re,im,0),ldc(re,im,S),ldc(re,im,2*(ptrdiff_t)(S)),ldc(re,im,3*(ptrdiff_t)(S)), \
            ldc(re,im,4*(ptrdiff_t)(S)),ldc(re,im,5*(ptrdiff_t)(S)),ldc(re,im,6*(ptrdiff_t)(S)),ldc(re,im,7*(ptrdiff_t)(S))); \
  _Pragma("GCC unroll 8") \
  for (int k_=0;k_<8;k_++) STOREK(k_,oo[k_]); \
}while(0)

// primes: C-sweeps accumulate cos part into Cb (with x0 added);
// T-sweeps accumulate sin part and store combined outputs.
#define PRIME_SWEEP_C(P,H,CT,k_lo,k_hi,S) do{ \
  V c[H]; \
  _Pragma("GCC unroll 16") for (int e=0;e<H;e++) c[e]=VB(CT[e+1]); \
  V ar[(k_hi)-(k_lo)], ai[(k_hi)-(k_lo)]; \
  _Pragma("GCC unroll 16") for (int q=0;q<(k_hi)-(k_lo);q++){ ar[q]=x0.re; ai[q]=x0.im; } \
  _Pragma("GCC unroll 16") \
  for (int j=1;j<=H;j++){ \
    CV a=ldc(re,im,(ptrdiff_t)j*(S)), b=ldc(re,im,(ptrdiff_t)((P)-j)*(S)); \
    V sr=vadd(a.re,b.re), si=vadd(a.im,b.im); \
    if ((k_lo)==0){ sumr=vadd(sumr,sr); sumi=vadd(sumi,si); } \
    _Pragma("GCC unroll 16") \
    for (int q=0;q<(k_hi)-(k_lo);q++){ int k=(k_lo)+q+1; \
      int e=(j*k)%(P); int idx=(e<=(H))?e:((P)-e); \
      ar[q]=vfma(c[idx-1],sr,ar[q]); ai[q]=vfma(c[idx-1],si,ai[q]); } \
  } \
  _Pragma("GCC unroll 16") \
  for (int q=0;q<(k_hi)-(k_lo);q++){ VST(Cb+16*((k_lo)+q),ar[q]); VST(Cb+16*((k_lo)+q)+8,ai[q]); } \
}while(0)

#define PRIME_SWEEP_T(P,H,ST,k_lo,k_hi,S,STOREK) do{ \
  V s_[H]; \
  _Pragma("GCC unroll 16") for (int e=0;e<H;e++) s_[e]=VB(ST[e+1]); \
  V tr[(k_hi)-(k_lo)], ti[(k_hi)-(k_lo)]; \
  _Pragma("GCC unroll 16") for (int q=0;q<(k_hi)-(k_lo);q++){ tr[q]=_mm512_setzero_pd(); ti[q]=tr[q]; } \
  _Pragma("GCC unroll 16") \
  for (int j=1;j<=H;j++){ \
    CV a=ldc(re,im,(ptrdiff_t)j*(S)), b=ldc(re,im,(ptrdiff_t)((P)-j)*(S)); \
    V dr=vsub(a.re,b.re), di=vsub(a.im,b.im); \
    _Pragma("GCC unroll 16") \
    for (int q=0;q<(k_hi)-(k_lo);q++){ int k=(k_lo)+q+1; \
      int e=(j*k)%(P); int idx=(e<=(H))?e:((P)-e); int sg=(e<=(H))?1:-1; \
      if (sg>0){ tr[q]=vfma(s_[idx-1],dr,tr[q]); ti[q]=vfma(s_[idx-1],di,ti[q]); } \
      else     { tr[q]=vfnma(s_[idx-1],dr,tr[q]); ti[q]=vfnma(s_[idx-1],di,ti[q]); } } \
  } \
  _Pragma("GCC unroll 16") \
  for (int q=0;q<(k_hi)-(k_lo);q++){ int k=(k_lo)+q+1; \
    V cr_=VLD(Cb+16*(k-1)), ci_=VLD(Cb+16*(k-1)+8); \
    CV e1_, e2_; \
    e1_.re=vadd(cr_,ti[q]);     e1_.im=vsub(ci_,tr[q]); \
    e2_.re=vsub(cr_,ti[q]);     e2_.im=vadd(ci_,tr[q]); \
    STOREK(k,e1_); STOREK((P)-k,e2_); } \
}while(0)

#define COREPRIME(P,H,CT,ST,KSPL,S,STOREK) do{ \
  double Cb[16*(H)] __attribute__((aligned(64))); \
  CV x0=ldc(re,im,0); \
  V sumr=x0.re, sumi=x0.im; \
  PRIME_SWEEP_C(P,H,CT,0,KSPL,S); \
  PRIME_SWEEP_C(P,H,CT,KSPL,H,S); \
  { CV s0; s0.re=sumr; s0.im=sumi; STOREK(0,s0); } \
  PRIME_SWEEP_T(P,H,ST,0,KSPL,S,STOREK); \
  PRIME_SWEEP_T(P,H,ST,KSPL,H,S,STOREK); \
}while(0)

#define CORE13(S, STOREK) COREPRIME(13,6,C13,S13,6,S,STOREK)
#define CORE17(S, STOREK) COREPRIME(17,8,C17,S17,4,S,STOREK)
#define CORE23(S, STOREK) COREPRIME(23,11,C23,S23,6,S,STOREK)

#define CORE36(S, STOREK) do{ \
  CV U[36]; \
  _Pragma("GCC unroll 1") \
  for (int j1=0;j1<4;j1++){ \
    CV x[9]; \
    _Pragma("GCC unroll 9") \
    for (int j2=0;j2<9;j2++) x[j2]=ldc(re,im,(ptrdiff_t)MAPJ36[j1*9+j2]*(S)); \
    dft9c(&U[j1*9], x); \
  } \
  _Pragma("GCC unroll 1") \
  for (int k9=0;k9<9;k9++){ \
    CV y0,y1,y2,y3; \
    dft4c(&y0,&y1,&y2,&y3, U[0*9+k9],U[1*9+k9],U[2*9+k9],U[3*9+k9]); \
    STOREK(MAPK36[0*9+k9],y0); STOREK(MAPK36[1*9+k9],y1); \
    STOREK(MAPK36[2*9+k9],y2); STOREK(MAPK36[3*9+k9],y3); \
  } \
}while(0)

#define CORE45(S, STOREK) do{ \
  CV U[45]; \
  _Pragma("GCC unroll 1") \
  for (int j5=0;j5<5;j5++){ \
    CV x[9], t[9]; \
    _Pragma("GCC unroll 9") \
    for (int j9=0;j9<9;j9++) x[j9]=ldc(re,im,(ptrdiff_t)MAPJ45[j5*9+j9]*(S)); \
    dft9c(t, x); \
    _Pragma("GCC unroll 9") \
    for (int k9=0;k9<9;k9++) U[k9*5+j5]=t[k9]; \
  } \
  _Pragma("GCC unroll 1") \
  for (int k9=0;k9<9;k9++){ \
    CV y0,y1,y2,y3,y4; \
    dft5c(&y0,&y1,&y2,&y3,&y4, U[k9*5+0],U[k9*5+1],U[k9*5+2],U[k9*5+3],U[k9*5+4]); \
    STOREK(MAPK45[k9*5+0],y0); STOREK(MAPK45[k9*5+1],y1); STOREK(MAPK45[k9*5+2],y2); \
    STOREK(MAPK45[k9*5+3],y3); STOREK(MAPK45[k9*5+4],y4); \
  } \
}while(0)

#define CORE64(S, STOREK) do{ \
  CV T[64]; \
  _Pragma("GCC unroll 1") \
  for (int j2=0;j2<8;j2++){ \
    const double* rr=re+(ptrdiff_t)j2*(S); const double* ii=im+(ptrdiff_t)j2*(S); \
    CV u[8]; \
    dft8c(u, ldc(rr,ii,0),ldc(rr,ii,8*(ptrdiff_t)(S)),ldc(rr,ii,16*(ptrdiff_t)(S)),ldc(rr,ii,24*(ptrdiff_t)(S)), \
             ldc(rr,ii,32*(ptrdiff_t)(S)),ldc(rr,ii,40*(ptrdiff_t)(S)),ldc(rr,ii,48*(ptrdiff_t)(S)),ldc(rr,ii,56*(ptrdiff_t)(S))); \
    T[0*8+j2]=u[0]; \
    _Pragma("GCC unroll 7") \
    for (int k1=1;k1<8;k1++){ \
      V wr=VB(TW64r[j2*8+k1]), wi=VB(TW64i[j2*8+k1]); \
      CV t; t.re=vfnma(wi,u[k1].im,vmul(wr,u[k1].re)); t.im=vfma(wi,u[k1].re,vmul(wr,u[k1].im)); \
      T[k1*8+j2]=t; \
    } \
  } \
  _Pragma("GCC unroll 1") \
  for (int k1=0;k1<8;k1++){ \
    CV v[8]; \
    dft8c(v, T[k1*8+0],T[k1*8+1],T[k1*8+2],T[k1*8+3],T[k1*8+4],T[k1*8+5],T[k1*8+6],T[k1*8+7]); \
    _Pragma("GCC unroll 8") \
    for (int k2=0;k2<8;k2++) STOREK(k1+8*k2,v[k2]); \
  } \
}while(0)

// ---------------------------------------------------------------------------
// store hooks
#define SK_BUF(k,v) do{ o[k]=(v); }while(0)
// instances get SS (store stride), qr/qi (store base), m (mask) in scope
#define SK_STR(k,v) do{ CV t_=(v); VSTM(qr+(ptrdiff_t)(k)*SS, m, t_.re); VSTM(qi+(ptrdiff_t)(k)*SS, m, t_.im); }while(0)
#define SK_MAP(k,v) do{ CV t_=(v); \
  t_.re=vadd(t_.re, VLD(cr+(ptrdiff_t)(k)*CSS)); \
  t_.im=vadd(t_.im, VLD(ci+(ptrdiff_t)(k)*CSS)); \
  mapop(&t_.re,&t_.im); \
  VSTM(qr+(ptrdiff_t)(k)*SS, m, t_.re); VSTM(qi+(ptrdiff_t)(k)*SS, m, t_.im); }while(0)

#define DEF_CODELETS(LN) \
NI void dftz_##LN(const double* restrict re, const double* restrict im, CV* restrict o){ \
  CORE##LN(8, SK_BUF); \
} \
NI void dfty_##LN(const double* restrict re0, const double* restrict im0, \
                  double* restrict qr0, double* restrict qi0, long ncol, const __mmask8 mtail){ \
  enum { SS = LN }; \
  for (long i_=0;i_<ncol;i_++){ \
    const double* re=re0+8*i_; const double* im=im0+8*i_; \
    double* qr=qr0+8*i_; double* qi=qi0+8*i_; \
    const __mmask8 m=(i_==ncol-1)?mtail:(__mmask8)0xFFu; \
    CORE##LN(LN, SK_STR); \
  } \
} \
NI void dftx_##LN(const double* restrict re0, const double* restrict im0, \
                  double* restrict qr0, double* restrict qi0, long ncol, const __mmask8 mtail){ \
  enum { SS = LN*LN }; \
  for (long i_=0;i_<ncol;i_++){ \
    const double* re=re0+8*i_; const double* im=im0+8*i_; \
    double* qr=qr0+8*i_; double* qi=qi0+8*i_; \
    const __mmask8 m=(i_==ncol-1)?mtail:(__mmask8)0xFFu; \
    CORE##LN(LN*LN, SK_STR); \
  } \
} \
NI void dftm_##LN(const double* restrict re0, const double* restrict im0, \
                  double* restrict qr0, double* restrict qi0, \
                  const double* restrict cr0, const double* restrict ci0, \
                  long ncol, const __mmask8 mtail){ \
  enum { SS = LN*LN, CSS = 8 }; \
  for (long i_=0;i_<ncol;i_++){ \
    const double* re=re0+8*i_; const double* im=im0+8*i_; \
    double* qr=qr0+8*i_; double* qi=qi0+8*i_; \
    const double* cr=cr0+(long)LN*8*i_; const double* ci=ci0+(long)LN*8*i_; \
    const __mmask8 m=(i_==ncol-1)?mtail:(__mmask8)0xFFu; \
    CORE##LN(LN*LN, SK_MAP); \
  } \
}

#define DEF_CODELETS_G(LN) \
NI void dftgz_##LN(const double* restrict re0, const double* restrict im0, \
                  double* restrict qr0, double* restrict qi0, long ncol){ \
  enum { SS = 8 }; \
  const __mmask8 m=(__mmask8)0xFFu; \
  for (long i_=0;i_<ncol;i_++){ \
    const double* re=re0+(long)LN*8*i_; const double* im=im0+(long)LN*8*i_; \
    double* qr=qr0+(long)LN*8*i_; double* qi=qi0+(long)LN*8*i_; \
    CORE##LN(8, SK_STR); \
  } \
} \
NI void dftgy_##LN(const double* restrict re0, const double* restrict im0, \
                  double* restrict qr0, double* restrict qi0, long ncol){ \
  enum { SS = 8*LN }; \
  const __mmask8 m=(__mmask8)0xFFu; \
  for (long i_=0;i_<ncol;i_++){ \
    const double* re=re0+8*i_; const double* im=im0+8*i_; \
    double* qr=qr0+8*i_; double* qi=qi0+8*i_; \
    CORE##LN(8*LN, SK_STR); \
  } \
} \
NI void dftgm_##LN(const double* restrict re0, const double* restrict im0, \
                  double* restrict qr0, double* restrict qi0, \
                  const double* restrict cr0, const double* restrict ci0, long ncol){ \
  enum { SS = 8*LN*LN, CSS = 8 }; \
  const __mmask8 m=(__mmask8)0xFFu; \
  for (long i_=0;i_<ncol;i_++){ \
    const double* re=re0+8*i_; const double* im=im0+8*i_; \
    double* qr=qr0+8*i_; double* qi=qi0+8*i_; \
    const double* cr=cr0+(long)LN*8*i_; const double* ci=ci0+(long)LN*8*i_; \
    CORE##LN(8*LN*LN, SK_MAP); \
  } \
}

DEF_CODELETS(6)
DEF_CODELETS(8)
DEF_CODELETS(13)
DEF_CODELETS(17)
DEF_CODELETS(23)
DEF_CODELETS(36)
DEF_CODELETS(45)
DEF_CODELETS(64)

DEF_CODELETS_G(6)
DEF_CODELETS_G(8)
DEF_CODELETS_G(13)
DEF_CODELETS_G(17)
DEF_CODELETS_G(23)
DEF_CODELETS_G(36)
DEF_CODELETS_G(45)

// grouped pass drivers: layout el(b,x,y,z) at ((x*L+y)*L+z)*8+b, b=0..7 lanes
static double SLBR[45*45*8+64] __attribute__((aligned(64)));
static double SLBI[45*45*8+64] __attribute__((aligned(64)));

#define DEF_GPASS(LN) \
NI void gpass_##LN(double* restrict ar, double* restrict ai, \
                   double* restrict br, double* restrict bi, \
                   const double* restrict cr, const double* restrict ci, const int domap){ \
  const int L=LN; \
  /* per x-slab: z axis (stride 8) a->slab buffer, then y axis (stride 8L) buffer->a */ \
  for (int x=0;x<L;x++){ \
    const size_t sb=(size_t)x*L*L*8; \
    dftgz_##LN(ar+sb, ai+sb, SLBR, SLBI, L); \
    dftgy_##LN(SLBR, SLBI, ar+sb, ai+sb, L); \
  } \
  /* x axis (stride 8L^2) + map in place; (test path without map goes a->b) */ \
  if (domap) dftgmi_##LN(ar, ai, cr, ci, (long)L*L); \
  else dftgx_##LN(ar, ai, br, bi, (long)L*L); \
}

// in-place grouped x-sweep with map: copy column to stack, then transform+map+store
#define DEF_CODELETS_GMI(LN) \
NI void dftgmi_##LN(double* restrict qr0, double* restrict qi0, \
                    const double* restrict cr0, const double* restrict ci0, long ncol){ \
  enum { SS = 8*LN*LN, CSS = 8 }; \
  const __mmask8 m=(__mmask8)0xFFu; \
  for (long i_=0;i_<ncol;i_++){ \
    double* qr=qr0+8*i_; double* qi=qi0+8*i_; \
    const double* cr=cr0+(long)LN*8*i_; const double* ci=ci0+(long)LN*8*i_; \
    double Xb[16*LN] __attribute__((aligned(64))); \
    _Pragma("GCC unroll 8") \
    for (int j_=0;j_<LN;j_++){ VST(Xb+16*j_, VLD(qr+(ptrdiff_t)j_*SS)); VST(Xb+16*j_+8, VLD(qi+(ptrdiff_t)j_*SS)); } \
    const double* re=Xb; const double* im=Xb+8; \
    CORE##LN(16, SK_MAP); \
  } \
}
// non-prime: core loads all inputs before storing -> safe in place without copy
NI void dftgmi_6(double* restrict qr0, double* restrict qi0,
                 const double* restrict cr0, const double* restrict ci0, long ncol){
  enum { SS = 8*6*6, CSS = 8 };
  const __mmask8 m=(__mmask8)0xFFu;
  for (long i_=0;i_<ncol;i_++){
    double* qr=qr0+8*i_; double* qi=qi0+8*i_;
    const double* cr=cr0+(long)6*8*i_; const double* ci=ci0+(long)6*8*i_;
    const double* re=qr; const double* im=qi;
    CORE6(8*6*6, SK_MAP);
  }
}
DEF_CODELETS_GMI(8)
DEF_CODELETS_GMI(13)
DEF_CODELETS_GMI(17)
DEF_CODELETS_GMI(23)
DEF_CODELETS_GMI(36)
DEF_CODELETS_GMI(45)

#define DEF_CODELETS_GX(LN) \
NI void dftgx_##LN(const double* restrict re0, const double* restrict im0, \
                  double* restrict qr0, double* restrict qi0, long ncol){ \
  enum { SS = 8*LN*LN }; \
  const __mmask8 m=(__mmask8)0xFFu; \
  for (long i_=0;i_<ncol;i_++){ \
    const double* re=re0+8*i_; const double* im=im0+8*i_; \
    double* qr=qr0+8*i_; double* qi=qi0+8*i_; \
    CORE##LN(8*LN*LN, SK_STR); \
  } \
}
DEF_CODELETS_GX(6)
DEF_CODELETS_GX(8)
DEF_CODELETS_GX(13)
DEF_CODELETS_GX(17)
DEF_CODELETS_GX(23)
DEF_CODELETS_GX(36)
DEF_CODELETS_GX(45)

DEF_GPASS(6)
DEF_GPASS(8)
DEF_GPASS(13)
DEF_GPASS(17)
DEF_GPASS(23)
DEF_GPASS(36)
DEF_GPASS(45)

// ---------------------------------------------------------------------------
// pass drivers

// generic ZY pass: z-axis in place on A (via transposes), then y-axis A->B
#define DEF_PASSZY(LN) \
NI void passzy_##LN(double* restrict ar, double* restrict ai, double* restrict br, double* restrict bi){ \
  const int L=LN; const int NB=(LN+7)/8; \
  const __mmask8 TM=(__mmask8)((LN%8)?((1u<<(LN%8))-1u):0xFFu); \
  for (int x=0;x<L;x++){ \
    double* sr=ar+(size_t)x*L*L; double* si=ai+(size_t)x*L*L; \
    for (int y0=0;y0<L;y0+=8){ \
      int nr=L-y0; if (nr>8) nr=8; \
      double INr[8*LN] __attribute__((aligned(64))), INi[8*LN] __attribute__((aligned(64))); \
      CV o[LN]; \
      _Pragma("GCC unroll 8") \
      for (int kb=0;kb<NB;kb++){ \
        V R[8], I[8]; \
        _Pragma("GCC unroll 8") \
        for (int l=0;l<8;l++){ int row=y0+l; R[l]=VLD(sr+(size_t)row*L+kb*8); I[l]=VLD(si+(size_t)row*L+kb*8); } \
        tr8(R); tr8(I); \
        _Pragma("GCC unroll 8") \
        for (int t=0;t<8;t++){ int j=kb*8+t; if (j<L){ VST(INr+8*j,R[t]); VST(INi+8*j,I[t]); } } \
      } \
      dftz_##LN(INr, INi, o); \
      _Pragma("GCC unroll 8") \
      for (int kb=0;kb<NB;kb++){ \
        const int kw=(L-kb*8>=8)?8:(L-kb*8); \
        const __mmask8 km=(kw==8)?0xFFu:((1u<<kw)-1u); \
        V R[8], I[8]; \
        _Pragma("GCC unroll 8") \
        for (int t=0;t<8;t++){ int u=kb*8+(t<kw?t:kw-1); R[t]=o[u].re; I[t]=o[u].im; } \
        tr8(R); tr8(I); \
        _Pragma("GCC unroll 8") \
        for (int l=0;l<8;l++){ if (l<nr){ VSTM(sr+(size_t)(y0+l)*L+kb*8,km,R[l]); VSTM(si+(size_t)(y0+l)*L+kb*8,km,I[l]); } } \
      } \
    } \
    double* qr0=br+(size_t)x*L*L; double* qi0=bi+(size_t)x*L*L; \
    dfty_##LN(sr, si, qr0, qi0, NB, TM); \
  } \
}

// fused ZY for L=6 and L=8: one slab fully in registers, A->B
NI void passzy_f8(double* restrict ar, double* restrict ai, double* restrict br, double* restrict bi){
  for (int x=0;x<8;x++){
    const double* sr=ar+(size_t)x*64; const double* si=ai+(size_t)x*64;
    double* qr=br+(size_t)x*64; double* qi=bi+(size_t)x*64;
    V R[8], I[8];
    _Pragma("GCC unroll 8")
    for (int l=0;l<8;l++){ R[l]=VLD(sr+8*l); I[l]=VLD(si+8*l); }
    tr8(R); tr8(I);                       // lanes = y, element = z
    CV t[8], u[8];
    _Pragma("GCC unroll 8")
    for (int j=0;j<8;j++){ t[j].re=R[j]; t[j].im=I[j]; }
    dft8c(u, t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]);   // z DFT
    _Pragma("GCC unroll 8")
    for (int j=0;j<8;j++){ R[j]=u[j].re; I[j]=u[j].im; }
    tr8(R); tr8(I);                       // lanes = kz, element = y
    _Pragma("GCC unroll 8")
    for (int j=0;j<8;j++){ t[j].re=R[j]; t[j].im=I[j]; }
    dft8c(u, t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]);   // y DFT
    _Pragma("GCC unroll 8")
    for (int k=0;k<8;k++){ VST(qr+8*k,u[k].re); VST(qi+8*k,u[k].im); }
  }
}
NI void passzy_f6(double* restrict ar, double* restrict ai, double* restrict br, double* restrict bi){
  const __mmask8 TM=0x3F;
  for (int x=0;x<6;x++){
    const double* sr=ar+(size_t)x*36; const double* si=ai+(size_t)x*36;
    double* qr=br+(size_t)x*36; double* qi=bi+(size_t)x*36;
    V R[8], I[8];
    _Pragma("GCC unroll 8")
    for (int l=0;l<8;l++){ R[l]=VLD(sr+6*l); I[l]=VLD(si+6*l); }
    tr8(R); tr8(I);                       // lanes = y(0..5 valid), element = z
    CV t[6], u0,u1,u2,u3,u4,u5;
    _Pragma("GCC unroll 6")
    for (int j=0;j<6;j++){ t[j].re=R[j]; t[j].im=I[j]; }
    {
      CV a0=cvadd(t[0],t[3]), d0=cvsub(t[0],t[3]);
      CV a1=cvadd(t[2],t[5]), d1=cvsub(t[2],t[5]);
      CV a2=cvadd(t[4],t[1]), d2=cvsub(t[4],t[1]);
      dft3c(&u0,&u4,&u2, a0,a1,a2);
      dft3c(&u3,&u1,&u5, d0,d1,d2);
    }
    R[0]=u0.re; R[1]=u1.re; R[2]=u2.re; R[3]=u3.re; R[4]=u4.re; R[5]=u5.re; R[6]=u5.re; R[7]=u5.re;
    I[0]=u0.im; I[1]=u1.im; I[2]=u2.im; I[3]=u3.im; I[4]=u4.im; I[5]=u5.im; I[6]=u5.im; I[7]=u5.im;
    tr8(R); tr8(I);                       // lanes = kz(0..5 valid), element = y
    _Pragma("GCC unroll 6")
    for (int j=0;j<6;j++){ t[j].re=R[j]; t[j].im=I[j]; }
    {
      CV a0=cvadd(t[0],t[3]), d0=cvsub(t[0],t[3]);
      CV a1=cvadd(t[2],t[5]), d1=cvsub(t[2],t[5]);
      CV a2=cvadd(t[4],t[1]), d2=cvsub(t[4],t[1]);
      dft3c(&u0,&u4,&u2, a0,a1,a2);
      dft3c(&u3,&u1,&u5, d0,d1,d2);
    }
    VSTM(qr+0 ,TM,u0.re); VSTM(qi+0 ,TM,u0.im);
    VSTM(qr+6 ,TM,u1.re); VSTM(qi+6 ,TM,u1.im);
    VSTM(qr+12,TM,u2.re); VSTM(qi+12,TM,u2.im);
    VSTM(qr+18,TM,u3.re); VSTM(qi+18,TM,u3.im);
    VSTM(qr+24,TM,u4.re); VSTM(qi+24,TM,u4.im);
    VSTM(qr+30,TM,u5.re); VSTM(qi+30,TM,u5.im);
  }
}

#define DEF_PASSX(LN) \
NI void passx_##LN(const double* restrict br, const double* restrict bi, \
                   double* restrict arr, double* restrict aii, \
                   const double* restrict cxr, const double* restrict cxi, const int domap){ \
  const int L=LN; const int NB=(LN+7)/8; \
  const __mmask8 TM=(__mmask8)((LN%8)?((1u<<(LN%8))-1u):0xFFu); \
  for (int y=0;y<L;y++){ \
    const size_t base=(size_t)y*L; \
    const size_t cbase=(size_t)y*NB*L*8; \
    if (domap) dftm_##LN(br+base, bi+base, arr+base, aii+base, cxr+cbase, cxi+cbase, NB, TM); \
    else       dftx_##LN(br+base, bi+base, arr+base, aii+base, NB, TM); \
  } \
}

DEF_PASSZY(13)
DEF_PASSZY(17)
DEF_PASSZY(23)
DEF_PASSZY(36)
DEF_PASSZY(45)
DEF_PASSZY(64)
DEF_PASSX(6)
NI void passx_8(const double* restrict br, const double* restrict bi,
                double* restrict arr, double* restrict aii,
                const double* restrict cxr, const double* restrict cxi, const int domap){
  if (domap) dftm_8(br, bi, arr, aii, cxr, cxi, 8, (__mmask8)0xFFu);
  else       dftx_8(br, bi, arr, aii, 8, (__mmask8)0xFFu);
}
DEF_PASSX(13)
DEF_PASSX(17)
DEF_PASSX(23)
DEF_PASSX(36)
DEF_PASSX(45)
DEF_PASSX(64)

static void passzy_6(double* ar, double* ai, double* br, double* bi){ passzy_f6(ar,ai,br,bi); }
static void passzy_8(double* ar, double* ai, double* br, double* bi){ passzy_f8(ar,ai,br,bi); }

// ---------------------------------------------------------------------------
static void deint(const double* restrict src, double* restrict dre, double* restrict dim_, long n){
  __m512i ire=_mm512_loadu_si512(IDX_RE_), iim=_mm512_loadu_si512(IDX_IM_);
  long i=0;
  for (; i+8<=n; i+=8){
    V a=VLD(src+2*i), b=VLD(src+2*i+8);
    VST(dre+i,_mm512_permutex2var_pd(a,ire,b));
    VST(dim_+i,_mm512_permutex2var_pd(a,iim,b));
  }
  for (; i<n; i++){ dre[i]=src[2*i]; dim_[i]=src[2*i+1]; }
}
static void intl(const double* restrict sre, const double* restrict sim, double* restrict dst, long n){
  __m512i ilo=_mm512_loadu_si512(IDX_LO_), ihi=_mm512_loadu_si512(IDX_HI_);
  long i=0;
  if (((uintptr_t)dst & 63)==0){
    for (; i+8<=n; i+=8){
      V a=VLD(sre+i), b=VLD(sim+i);
      _mm512_stream_pd(dst+2*i,  _mm512_permutex2var_pd(a,ilo,b));
      _mm512_stream_pd(dst+2*i+8,_mm512_permutex2var_pd(a,ihi,b));
    }
    _mm_sfence();
  } else {
    for (; i+8<=n; i+=8){
      V a=VLD(sre+i), b=VLD(sim+i);
      VST(dst+2*i,  _mm512_permutex2var_pd(a,ilo,b));
      VST(dst+2*i+8,_mm512_permutex2var_pd(a,ihi,b));
    }
  }
  for (; i<n; i++){ dst[2*i]=sre[i]; dst[2*i+1]=sim[i]; }
}

// group layout conversion: 8 volumes (complex interleaved, each n points)
// -> grouped split planes g[(idx)*8 + lane]
static void tovolsg_n(const double* restrict x, long n, int nv, double* restrict gre, double* restrict gim){
  __m512i ire=_mm512_loadu_si512(IDX_RE_), iim=_mm512_loadu_si512(IDX_IM_);
  const double* src[8];
  for (int l=0;l<8;l++) src[l]=x+(size_t)((l<nv)?l:(nv-1))*2*n;
  long i=0;
  for (; i+8<=n; i+=8){
    V R[8], I[8];
    _Pragma("GCC unroll 8")
    for (int l=0;l<8;l++){
      V a=VLD(src[l]+2*i), b=VLD(src[l]+2*i+8);
      R[l]=_mm512_permutex2var_pd(a,ire,b);
      I[l]=_mm512_permutex2var_pd(a,iim,b);
    }
    tr8(R); tr8(I);
    _Pragma("GCC unroll 8")
    for (int t=0;t<8;t++){ VST(gre+(i+t)*8,R[t]); VST(gim+(i+t)*8,I[t]); }
  }
  for (; i<n; i++)
    for (int l=0;l<8;l++){ gre[i*8+l]=src[l][2*i]; gim[i*8+l]=src[l][2*i+1]; }
}
static void tovolsg(const double* restrict x, long n, double* restrict gre, double* restrict gim){
  tovolsg_n(x,n,8,gre,gim);
}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Waggressive-loop-optimizations"
static void fromvolsg_n(const double* restrict gre, const double* restrict gim, long n, int nv, double* restrict y){
  __m512i ilo=_mm512_loadu_si512(IDX_LO_), ihi=_mm512_loadu_si512(IDX_HI_);
  long i=0;
  for (; i+8<=n; i+=8){
    V R[8], I[8];
    _Pragma("GCC unroll 8")
    for (int t=0;t<8;t++){ R[t]=VLD(gre+(i+t)*8); I[t]=VLD(gim+(i+t)*8); }
    tr8(R); tr8(I);
    for (int l=0;l<nv;l++){
      VST(y+(size_t)l*2*n+2*i,  _mm512_permutex2var_pd(R[l],ilo,I[l]));
      VST(y+(size_t)l*2*n+2*i+8,_mm512_permutex2var_pd(R[l],ihi,I[l]));
    }
  }
  for (; i<n; i++){
    const double* gr=gre+(size_t)i*8;
    const double* gi=gim+(size_t)i*8;
    for (int l=0;l<nv;l++){
      double* yp=y+(size_t)l*2*(size_t)n+2*(size_t)i;
      yp[0]=gr[l]; yp[1]=gi[l];
    }
  }
}
#pragma GCC diagnostic pop
static void fromvolsg(const double* restrict gre, const double* restrict gim, long n, double* restrict y){
  __m512i ilo=_mm512_loadu_si512(IDX_LO_), ihi=_mm512_loadu_si512(IDX_HI_);
  long i=0;
  if ((((uintptr_t)y & 63)==0) && ((n & 3)==0)){
    for (; i+8<=n; i+=8){
      V R[8], I[8];
      _Pragma("GCC unroll 8")
      for (int t=0;t<8;t++){ R[t]=VLD(gre+(i+t)*8); I[t]=VLD(gim+(i+t)*8); }
      tr8(R); tr8(I);
      _Pragma("GCC unroll 8")
      for (int l=0;l<8;l++){
        _mm512_stream_pd(y+(size_t)l*2*n+2*i,  _mm512_permutex2var_pd(R[l],ilo,I[l]));
        _mm512_stream_pd(y+(size_t)l*2*n+2*i+8,_mm512_permutex2var_pd(R[l],ihi,I[l]));
      }
    }
    _mm_sfence();
  } else {
    for (; i+8<=n; i+=8){
      V R[8], I[8];
      _Pragma("GCC unroll 8")
      for (int t=0;t<8;t++){ R[t]=VLD(gre+(i+t)*8); I[t]=VLD(gim+(i+t)*8); }
      tr8(R); tr8(I);
      _Pragma("GCC unroll 8")
      for (int l=0;l<8;l++){
        VST(y+(size_t)l*2*n+2*i,  _mm512_permutex2var_pd(R[l],ilo,I[l]));
        VST(y+(size_t)l*2*n+2*i+8,_mm512_permutex2var_pd(R[l],ihi,I[l]));
      }
    }
  }
  for (; i<n; i++)
    for (int l=0;l<8;l++){ y[(size_t)l*2*n+2*i]=gre[i*8+l]; y[(size_t)l*2*n+2*i+1]=gim[i*8+l]; }
}

static double *GAR,*GAI,*GBR,*GBI,*GCR,*GCI,*CXR,*CXI,*GXR,*GXI;

// grouped c in x-sweep order: gcx[(q*L+k)*8+l] = gc[(k*L*L+q)*8+l]
#define DEF_BUILDGCX(LN) \
static void buildgcx_##LN(const double* restrict gcr, const double* restrict gci, \
                          double* restrict gxr, double* restrict gxi){ \
  const long L=LN, L2=(long)LN*LN; \
  for (long q=0;q<L2;q++){ \
    double* pr=gxr+q*L*8; double* pi=gxi+q*L*8; \
    const double* sr=gcr+q*8; const double* si=gci+q*8; \
    for (long k=0;k<L;k++){ VST(pr+k*8, VLD(sr+k*L2*8)); VST(pi+k*8, VLD(si+k*L2*8)); } \
  } \
}
DEF_BUILDGCX(6)
DEF_BUILDGCX(8)
DEF_BUILDGCX(13)
DEF_BUILDGCX(17)
DEF_BUILDGCX(23)
DEF_BUILDGCX(36)
DEF_BUILDGCX(45)

// build x-pass-ordered c planes directly from interleaved complex c:
// cx[((y*NB+zb)*L + k)*8 + l] = c[k*L*L + y*L + zb*8 + l]
#define DEF_BUILDCX(LN) \
static void buildcx_##LN(const double* restrict c, \
                         double* restrict cxr, double* restrict cxi){ \
  const int L=LN; const int NB=(LN+7)/8; \
  __m512i ire=_mm512_loadu_si512(IDX_RE_), iim=_mm512_loadu_si512(IDX_IM_); \
  for (int y=0;y<L;y++) for (int zb=0;zb<NB;zb++){ \
    const size_t base=(size_t)y*L+zb*8; \
    const int kw=(L-zb*8>=8)?8:(L-zb*8); \
    const __mmask8 mA=(__mmask8)((kw>=4)?0xFFu:((1u<<(2*kw))-1u)); \
    const __mmask8 mB=(__mmask8)((kw>=8)?0xFFu:((kw>4)?((1u<<(2*(kw-4)))-1u):0u)); \
    double* pr=cxr+((size_t)y*NB+zb)*L*8; \
    double* pi=cxi+((size_t)y*NB+zb)*L*8; \
    for (int k=0;k<L;k++){ \
      size_t q=2*((size_t)k*L*L+base); \
      V a=_mm512_maskz_loadu_pd(mA,c+q), b=_mm512_maskz_loadu_pd(mB,c+q+8); \
      VST(pr+(size_t)k*8, _mm512_permutex2var_pd(a,ire,b)); \
      VST(pi+(size_t)k*8, _mm512_permutex2var_pd(a,iim,b)); \
    } \
  } \
}
DEF_BUILDCX(6)
DEF_BUILDCX(8)
DEF_BUILDCX(13)
DEF_BUILDCX(17)
DEF_BUILDCX(23)
DEF_BUILDCX(36)
DEF_BUILDCX(45)
DEF_BUILDCX(64)
#define GPLANE_DBL ((size_t)(45*45*45+16)*8)

#define DEF_GRUN(LN) \
static void grun_##LN(long G, int lastnv, long m, const double* restrict x, const double* restrict c, \
                      double* restrict one, double* restrict fin){ \
  const long n=(long)LN*LN*LN; \
  for (long g=0;g<G;g++){ \
    const int nv=(g==G-1)?lastnv:8; \
    tovolsg_n(x+(size_t)16*n*g, n, nv, GAR, GAI); \
    tovolsg_n(c+(size_t)16*n*g, n, nv, GCR, GCI); \
    buildgcx_##LN(GCR,GCI,GXR,GXI); \
    for (long it=1;it<=m;it++){ \
      gpass_##LN(GAR,GAI,GBR,GBI,GXR,GXI,1); \
      if (it==1) fromvolsg_n(GAR,GAI,n,nv,one+(size_t)16*n*g); \
    } \
    fromvolsg_n(GAR,GAI,n,nv,fin+(size_t)16*n*g); \
  } \
}
DEF_GRUN(6)
DEF_GRUN(8)
DEF_GRUN(13)
DEF_GRUN(17)
DEF_GRUN(23)
DEF_GRUN(36)
DEF_GRUN(45)
static void grun_64(long G, int lastnv, long m, const double* restrict x, const double* restrict c, double* restrict one, double* restrict fin){ (void)G;(void)lastnv;(void)m;(void)x;(void)c;(void)one;(void)fin; }

// ---------------------------------------------------------------------------
static int USE_GROUPED[65];

#define DEF_RUN(LN, XR, XI) \
static void run_##LN(long B, long m, const double* restrict x, const double* restrict c, \
                     double* restrict one, double* restrict fin){ \
  const long n=(long)LN*LN*LN; \
  if (m<1) m=1; \
  long b0=0; \
  if (LN<=45 && USE_GROUPED[LN]){ \
    long G=B/8; int r=(int)(B%8); \
    if (r>=5){ grun_##LN(G+1,r,m,x,c,one,fin); b0=B; } \
    else if (G>0){ grun_##LN(G,8,m,x,c,one,fin); b0=8*G; } \
  } \
  for (long b=b0;b<B;b++){ \
    deint(x+2*n*b, ARE, AIM, n); \
    buildcx_##LN(c+2*n*b,CXR,CXI); \
    for (long it=1;it<=m;it++){ \
      passzy_##LN(ARE,AIM,XR,XI); \
      passx_##LN(XR,XI,ARE,AIM,CXR,CXI,1); \
      if (it==1) intl(ARE,AIM,one+2*n*b,n); \
    } \
    intl(ARE,AIM,fin+2*n*b,n); \
  } \
}

DEF_RUN(6, BRE, BIM)
DEF_RUN(8, BRE, BIM)
DEF_RUN(13, BRE, BIM)
DEF_RUN(17, BRE, BIM)
DEF_RUN(23, BRE, BIM)
DEF_RUN(36, ARE, AIM)
DEF_RUN(45, ARE, AIM)
DEF_RUN(64, ARE, AIM)

// ---------------------------------------------------------------------------
void plan(void){
  const long double PI=acosl(-1.0L);
  W3S =(double)(sqrtl(3.0L)/2.0L);
  W8C =(double)(sqrtl(2.0L)/2.0L);
  S5Q =(double)(sqrtl(5.0L)/4.0L);
  S5_1=(double)sinl(2.0L*PI/5.0L);
  S5_2=(double)sinl(4.0L*PI/5.0L);
  {
    int e[3]={1,2,4};
    for (int i=0;i<3;i++){ TW9r[i]=(double)cosl(2.0L*PI*e[i]/9.0L); TW9i[i]=(double)(-sinl(2.0L*PI*e[i]/9.0L)); }
  }
  for (int e=0;e<=6;e++){ C13[e]=(double)cosl(2.0L*PI*e/13.0L); S13[e]=(double)sinl(2.0L*PI*e/13.0L); }
  for (int e=0;e<=8;e++){ C17[e]=(double)cosl(2.0L*PI*e/17.0L); S17[e]=(double)sinl(2.0L*PI*e/17.0L); }
  for (int e=0;e<=11;e++){ C23[e]=(double)cosl(2.0L*PI*e/23.0L); S23[e]=(double)sinl(2.0L*PI*e/23.0L); }
  for (int j2=0;j2<8;j2++) for (int k1=0;k1<8;k1++){
    int e=(j2*k1)%64; TW64r[j2*8+k1]=(double)cosl(2.0L*PI*e/64.0L); TW64i[j2*8+k1]=(double)(-sinl(2.0L*PI*e/64.0L));
  }
  for (int j1=0;j1<4;j1++) for (int j2=0;j2<9;j2++) MAPJ36[j1*9+j2]=(9*j1+4*j2)%36;
  for (int k4=0;k4<4;k4++) for (int k9=0;k9<9;k9++) MAPK36[k4*9+k9]=(9*k4+28*k9)%36;
  for (int j5=0;j5<5;j5++) for (int j9=0;j9<9;j9++) MAPJ45[j5*9+j9]=(5*j9+9*j5)%45;
  for (int k9=0;k9<9;k9++) for (int k5=0;k5<5;k5++) MAPK45[k9*5+k5]=(10*k9+36*k5)%45;

  for (int i=0;i<65;i++) USE_GROUPED[i]=0;
  USE_GROUPED[6]=1; USE_GROUPED[8]=1; USE_GROUPED[13]=1; USE_GROUPED[17]=1; USE_GROUPED[23]=1;
  if (!ARE){
    static int hoff_=0;
    #define HALLOC(sz) (double*)((char*)aligned_alloc(4096, (sz)+65536) + (hoff_=(hoff_+1)%8)*(4096+320))
    size_t bytes=PLANE_DBL*sizeof(double);
    ARE=HALLOC(bytes); AIM=HALLOC(bytes);
    BRE=HALLOC(bytes); BIM=HALLOC(bytes);
    CRE2=HALLOC(bytes); CIM2=HALLOC(bytes);
    memset(ARE,0,bytes); memset(AIM,0,bytes); memset(BRE,0,bytes);
    memset(BIM,0,bytes); memset(CRE2,0,bytes); memset(CIM2,0,bytes);
    size_t gb=GPLANE_DBL*sizeof(double);
    GAR=HALLOC(gb); GAI=HALLOC(gb);
    GBR=HALLOC(gb); GBI=HALLOC(gb);
    GCR=HALLOC(gb); GCI=HALLOC(gb);
    GXR=HALLOC(gb); GXI=HALLOC(gb);
    memset(GAR,0,gb); memset(GAI,0,gb); memset(GBR,0,gb);
    memset(GBI,0,gb); memset(GCR,0,gb); memset(GCI,0,gb);
    memset(GXR,0,gb); memset(GXI,0,gb);
    CXR=HALLOC(bytes); CXI=HALLOC(bytes);
    memset(CXR,0,bytes); memset(CXI,0,bytes);
    madvise(ARE,bytes,MADV_HUGEPAGE); madvise(AIM,bytes,MADV_HUGEPAGE);
    madvise(BRE,bytes,MADV_HUGEPAGE); madvise(BIM,bytes,MADV_HUGEPAGE);
    madvise(CRE2,bytes,MADV_HUGEPAGE); madvise(CIM2,bytes,MADV_HUGEPAGE);
    madvise(CXR,bytes,MADV_HUGEPAGE); madvise(CXI,bytes,MADV_HUGEPAGE);
    madvise(GAR,gb,MADV_HUGEPAGE); madvise(GAI,gb,MADV_HUGEPAGE);
    madvise(GBR,gb,MADV_HUGEPAGE); madvise(GBI,gb,MADV_HUGEPAGE);
    madvise(GCR,gb,MADV_HUGEPAGE); madvise(GCI,gb,MADV_HUGEPAGE);
  }
}

void set_grouped(long L, long v){ if (L>=0 && L<65) USE_GROUPED[L]=(int)v; }

void run(long L, long B, long m, const double* x, const double* c, double* one, double* fin){
  switch(L){
    case 6:  run_6 (B,m,x,c,one,fin); break;
    case 8:  run_8 (B,m,x,c,one,fin); break;
    case 13: run_13(B,m,x,c,one,fin); break;
    case 17: run_17(B,m,x,c,one,fin); break;
    case 23: run_23(B,m,x,c,one,fin); break;
    case 36: run_36(B,m,x,c,one,fin); break;
    case 45: run_45(B,m,x,c,one,fin); break;
    case 64: run_64(B,m,x,c,one,fin); break;
  }
}

void fft3_once(long L, const double* x, double* y){
  long n=L*L*L;
  deint(x, ARE, AIM, n);
  switch(L){
#define CASE1(LN) case LN: passzy_##LN(ARE,AIM,BRE,BIM); passx_##LN(BRE,BIM,ARE,AIM,0,0,0); break;
    CASE1(6) CASE1(8) CASE1(13) CASE1(17) CASE1(23) CASE1(36) CASE1(45) CASE1(64)
#undef CASE1
  }
  intl(ARE,AIM,y,n);
}
