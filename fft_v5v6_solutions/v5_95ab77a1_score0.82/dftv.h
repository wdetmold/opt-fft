// Hand-form in-place vector DFT networks on register arrays.
// R[i], I[i] are __m512d (8 lanes). All twiddles exact-rounded constants.
#include <immintrin.h>

#define VADD _mm512_add_pd
#define VSUB _mm512_sub_pd
#define VMUL _mm512_mul_pd
#define VFMA _mm512_fmadd_pd
#define VFMS _mm512_fmsub_pd
#define VFNMA _mm512_fnmadd_pd
#define VSET _mm512_set1_pd

// ---- DFT2 ----
static inline void dft2v(__m512d* R, __m512d* I){
  __m512d ar=R[0], ai=I[0];
  R[0]=VADD(ar,R[1]); I[0]=VADD(ai,I[1]);
  R[1]=VSUB(ar,R[1]); I[1]=VSUB(ai,I[1]);
}

// ---- DFT3 ----
#define S3V 0x1.bb67ae8584caap-1   /* sin(2pi/3) */
static inline void dft3v(__m512d* R, __m512d* I){
  __m512d t1r=VADD(R[1],R[2]), t1i=VADD(I[1],I[2]);
  __m512d t2r=VSUB(R[1],R[2]), t2i=VSUB(I[1],I[2]);
  __m512d half=VSET(0.5), s3=VSET(S3V);
  __m512d mr=VFNMA(t1r,half,R[0]), mi=VFNMA(t1i,half,I[0]);
  R[0]=VADD(R[0],t1r); I[0]=VADD(I[0],t1i);
  R[1]=VFMA(t2i,s3,mr);  I[1]=VFNMA(t2r,s3,mi);
  R[2]=VFNMA(t2i,s3,mr); I[2]=VFMA(t2r,s3,mi);
}

// ---- DFT4 ----
static inline void dft4v(__m512d* R, __m512d* I){
  __m512d t0r=VADD(R[0],R[2]), t0i=VADD(I[0],I[2]);
  __m512d t1r=VSUB(R[0],R[2]), t1i=VSUB(I[0],I[2]);
  __m512d t2r=VADD(R[1],R[3]), t2i=VADD(I[1],I[3]);
  __m512d t3r=VSUB(R[1],R[3]), t3i=VSUB(I[1],I[3]);
  R[0]=VADD(t0r,t2r); I[0]=VADD(t0i,t2i);
  R[2]=VSUB(t0r,t2r); I[2]=VSUB(t0i,t2i);
  R[1]=VADD(t1r,t3i); I[1]=VSUB(t1i,t3r);
  R[3]=VSUB(t1r,t3i); I[3]=VADD(t1i,t3r);
}

// ---- DFT5 (symmetric) ----
#define C51 0x1.3c6ef372fe950p-2    /* cos(2pi/5) */
#define C52 -0x1.9e3779b97f4a8p-1  /* cos(4pi/5) */
#define S51 0x1.e6f0e134454ffp-1   /* sin(2pi/5) */
#define S52 0x1.2cf2304755a5ep-1   /* sin(4pi/5) */
static inline void dft5v(__m512d* R, __m512d* I){
  __m512d u1r=VADD(R[1],R[4]), u1i=VADD(I[1],I[4]);
  __m512d v1r=VSUB(R[1],R[4]), v1i=VSUB(I[1],I[4]);
  __m512d u2r=VADD(R[2],R[3]), u2i=VADD(I[2],I[3]);
  __m512d v2r=VSUB(R[2],R[3]), v2i=VSUB(I[2],I[3]);
  __m512d x0r=R[0], x0i=I[0];
  __m512d c51=VSET(C51), c52=VSET(C52), s51=VSET(S51), s52=VSET(S52);
  R[0]=VADD(x0r,VADD(u1r,u2r)); I[0]=VADD(x0i,VADD(u1i,u2i));
  __m512d p1r=VFMA(u2r,c52,VFMA(u1r,c51,x0r));
  __m512d p1i=VFMA(u2i,c52,VFMA(u1i,c51,x0i));
  __m512d b1r=VFMA(v2r,s52,VMUL(v1r,s51));
  __m512d b1i=VFMA(v2i,s52,VMUL(v1i,s51));
  __m512d p2r=VFMA(u2r,c51,VFMA(u1r,c52,x0r));
  __m512d p2i=VFMA(u2i,c51,VFMA(u1i,c52,x0i));
  __m512d b2r=VFNMA(v2r,s51,VMUL(v1r,s52));
  __m512d b2i=VFNMA(v2i,s51,VMUL(v1i,s52));
  R[1]=VADD(p1r,b1i); I[1]=VSUB(p1i,b1r);
  R[4]=VSUB(p1r,b1i); I[4]=VADD(p1i,b1r);
  R[2]=VADD(p2r,b2i); I[2]=VSUB(p2i,b2r);
  R[3]=VSUB(p2r,b2i); I[3]=VADD(p2i,b2r);
}

// ---- DFT8 (DIT: 2 x DFT4 + twiddles) ----
#define RT2 0x1.6a09e667f3bcdp-1   /* sqrt(2)/2 */
static inline void dft8v(__m512d* R, __m512d* I){
  __m512d er[4], ei[4], or_[4], oi[4];
  er[0]=R[0]; er[1]=R[2]; er[2]=R[4]; er[3]=R[6];
  ei[0]=I[0]; ei[1]=I[2]; ei[2]=I[4]; ei[3]=I[6];
  or_[0]=R[1]; or_[1]=R[3]; or_[2]=R[5]; or_[3]=R[7];
  oi[0]=I[1]; oi[1]=I[3]; oi[2]=I[5]; oi[3]=I[7];
  dft4v(er, ei);
  dft4v(or_, oi);
  __m512d rt2=VSET(RT2);
  // W8^1 * O1 = c*(or+oi) + i*c*(oi-or)
  __m512d t1r=VMUL(rt2,VADD(or_[1],oi[1]));
  __m512d t1i=VMUL(rt2,VSUB(oi[1],or_[1]));
  // W8^2 * O2 = oi - i*or
  __m512d t2r=oi[2];
  __m512d t2i=_mm512_xor_pd(or_[2],VSET(-0.0));
  // W8^3 * O3 = c*(oi-or) - i*c*(or+oi)
  __m512d t3r=VMUL(rt2,VSUB(oi[3],or_[3]));
  __m512d t3i=_mm512_xor_pd(VMUL(rt2,VADD(or_[3],oi[3])),VSET(-0.0));
  R[0]=VADD(er[0],or_[0]); I[0]=VADD(ei[0],oi[0]);
  R[4]=VSUB(er[0],or_[0]); I[4]=VSUB(ei[0],oi[0]);
  R[1]=VADD(er[1],t1r);    I[1]=VADD(ei[1],t1i);
  R[5]=VSUB(er[1],t1r);    I[5]=VSUB(ei[1],t1i);
  R[2]=VADD(er[2],t2r);    I[2]=VADD(ei[2],t2i);
  R[6]=VSUB(er[2],t2r);    I[6]=VSUB(ei[2],t2i);
  R[3]=VADD(er[3],t3r);    I[3]=VADD(ei[3],t3i);
  R[7]=VSUB(er[3],t3r);    I[7]=VSUB(ei[3],t3i);
}

// ---- DFT9 (CT 3x3): X[k1+3*k2] ----
#define W91R 0x1.8836fa2cf5039p-1  /* cos(2pi/9) */
#define W91I -0x1.491b7523c161dp-1 /* -sin(2pi/9) */
#define W92R 0x1.63a1a7e0b738ap-3  /* cos(4pi/9) */
#define W92I -0x1.f838b8c811c17p-1
#define W94R -0x1.e11f642522d1cp-1 /* cos(8pi/9) */
#define W94I -0x1.5e3a8748a0bf5p-2
static inline void dft9v(__m512d* R, __m512d* I){
  __m512d ar[3], ai[3], br[3], bi[3], cr[3], ci[3];
  ar[0]=R[0]; ar[1]=R[3]; ar[2]=R[6]; ai[0]=I[0]; ai[1]=I[3]; ai[2]=I[6];
  br[0]=R[1]; br[1]=R[4]; br[2]=R[7]; bi[0]=I[1]; bi[1]=I[4]; bi[2]=I[7];
  cr[0]=R[2]; cr[1]=R[5]; cr[2]=R[8]; ci[0]=I[2]; ci[1]=I[5]; ci[2]=I[8];
  dft3v(ar, ai);  // j2=0
  dft3v(br, bi);  // j2=1 -> twiddle W9^{k1}
  dft3v(cr, ci);  // j2=2 -> twiddle W9^{2k1}
  // b[1] *= W9^1; b[2] *= W9^2; c[1] *= W9^2; c[2] *= W9^4
  {
    __m512d wr=VSET(W91R), wi=VSET(W91I);
    __m512d nr=VFMS(br[1],wr,VMUL(bi[1],wi));
    bi[1]=VFMA(br[1],wi,VMUL(bi[1],wr)); br[1]=nr;
  }
  {
    __m512d wr=VSET(W92R), wi=VSET(W92I);
    __m512d nr=VFMS(br[2],wr,VMUL(bi[2],wi));
    bi[2]=VFMA(br[2],wi,VMUL(bi[2],wr)); br[2]=nr;
    nr=VFMS(cr[1],wr,VMUL(ci[1],wi));
    ci[1]=VFMA(cr[1],wi,VMUL(ci[1],wr)); cr[1]=nr;
  }
  {
    __m512d wr=VSET(W94R), wi=VSET(W94I);
    __m512d nr=VFMS(cr[2],wr,VMUL(ci[2],wi));
    ci[2]=VFMA(cr[2],wi,VMUL(ci[2],wr)); cr[2]=nr;
  }
  // second stage: for each k1, DFT3 over (a[k1], b[k1], c[k1]) -> X[k1+3*k2]
  for (int k1=0;k1<3;k1++){
    __m512d rr[3], ii[3];
    rr[0]=ar[k1]; rr[1]=br[k1]; rr[2]=cr[k1];
    ii[0]=ai[k1]; ii[1]=bi[k1]; ii[2]=ci[k1];
    dft3v(rr, ii);
    R[k1]=rr[0];   I[k1]=ii[0];
    R[k1+3]=rr[1]; I[k1+3]=ii[1];
    R[k1+6]=rr[2]; I[k1+6]=ii[2];
  }
}
