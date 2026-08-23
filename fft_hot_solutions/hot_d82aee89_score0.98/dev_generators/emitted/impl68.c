
#include <immintrin.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#define ALIGN64 __attribute__((aligned(64)))
typedef __m512d V;
#define VL(p)      _mm512_load_pd(p)
#define VLU(p)     _mm512_loadu_pd(p)
#define VS(p,v)    _mm512_store_pd(p,v)
#define VSU(p,v)   _mm512_storeu_pd(p,v)
#define VADD(a,b)  _mm512_add_pd(a,b)
#define VSUB(a,b)  _mm512_sub_pd(a,b)
#define VMUL(a,b)  _mm512_mul_pd(a,b)
#define VFMA(a,b,c)  _mm512_fmadd_pd(a,b,c)
#define VFMS(a,b,c)  _mm512_fmsub_pd(a,b,c)
#define VFNMA(a,b,c) _mm512_fnmadd_pd(a,b,c)
#define VSET1(x)   _mm512_set1_pd(x)
#define VRSQRT(x)  _mm512_rsqrt14_pd(x)
#define VRCP(x)    _mm512_rcp14_pd(x)


static const long long IDXR_[8] ALIGN64 = {0,2,4,6,8,10,12,14};
static const long long IDXI_[8] ALIGN64 = {1,3,5,7,9,11,13,15};
static const long long IDXLO_[8] ALIGN64 = {0,8,1,9,2,10,3,11};
static const long long IDXHI_[8] ALIGN64 = {4,12,5,13,6,14,7,15};
#define PERM2(a,idx,b) _mm512_permutex2var_pd(a, _mm512_load_si512((const void*)idx), b)
#define PERM2Z(m,a,idx,b) _mm512_maskz_permutex2var_pd(m, a, _mm512_load_si512((const void*)idx), b)

// ---- map: given zr,zi (post +c), produce zr*q, zi*q with q = 1/(1+sqrt(zr^2+zi^2))
// rsqrt14 seed + 2 Newton for rsqrt; m = s*r; rcp14 seed + 2 Newton for 1/(1+m).
// all-FMA, ~1ulp.  s floored at tiny to avoid rsqrt(0)=inf -> NaN (pad lanes).
#define MAP2(zr, zi) do { \
    V s_ = VFMA(zr, zr, _mm512_mul_pd(zi, zi)); \
    V r_ = VRSQRT(s_); \
    V h_ = VMUL(s_, VSET1(0.5)); \
    V e_ = VFNMA(VMUL(r_, r_), h_, VSET1(0.5)); \
    r_ = VFMA(r_, e_, r_); \
    e_ = VFNMA(VMUL(r_, r_), h_, VSET1(0.5)); \
    r_ = VFMA(r_, e_, r_); \
    V m_ = VMUL(s_, r_); \
    V u_ = VADD(VSET1(1.0), m_); \
    V q_ = VRCP(u_); \
    V t_ = VFNMA(u_, q_, VSET1(1.0)); \
    q_ = VFMA(q_, t_, q_); \
    t_ = VFNMA(u_, q_, VSET1(1.0)); \
    q_ = VFMA(q_, t_, q_); \
    zr = VMUL(zr, q_); \
    zi = VMUL(zi, q_); \
} while(0)


#define TR8(r0,r1,r2,r3,r4,r5,r6,r7) do { \
    V u0=_mm512_unpacklo_pd(r0,r1), u1=_mm512_unpackhi_pd(r0,r1); \
    V u2=_mm512_unpacklo_pd(r2,r3), u3=_mm512_unpackhi_pd(r2,r3); \
    V u4=_mm512_unpacklo_pd(r4,r5), u5=_mm512_unpackhi_pd(r4,r5); \
    V u6=_mm512_unpacklo_pd(r6,r7), u7=_mm512_unpackhi_pd(r6,r7); \
    V s0=_mm512_shuffle_f64x2(u0,u2,0x88), s1=_mm512_shuffle_f64x2(u1,u3,0x88); \
    V s2=_mm512_shuffle_f64x2(u0,u2,0xdd), s3=_mm512_shuffle_f64x2(u1,u3,0xdd); \
    V s4=_mm512_shuffle_f64x2(u4,u6,0x88), s5=_mm512_shuffle_f64x2(u5,u7,0x88); \
    V s6=_mm512_shuffle_f64x2(u4,u6,0xdd), s7=_mm512_shuffle_f64x2(u5,u7,0xdd); \
    r0=_mm512_shuffle_f64x2(s0,s4,0x88); r4=_mm512_shuffle_f64x2(s0,s4,0xdd); \
    r1=_mm512_shuffle_f64x2(s1,s5,0x88); r5=_mm512_shuffle_f64x2(s1,s5,0xdd); \
    r2=_mm512_shuffle_f64x2(s2,s6,0x88); r6=_mm512_shuffle_f64x2(s2,s6,0xdd); \
    r3=_mm512_shuffle_f64x2(s3,s7,0x88); r7=_mm512_shuffle_f64x2(s3,s7,0xdd); \
} while(0)


// convert 8 sites x 8 vols from natural (per-vol interleaved) to SoA split
static inline __attribute__((always_inline)) void soa_in8(const double* xv, long vstride, long site, double* RE, double* IM){
    V r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7;
#define LD1(v) { V a=VLU(xv + v*vstride + 2*site), b=VLU(xv + v*vstride + 2*site + 8);     r##v = PERM2(a, IDXR_, b); i##v = PERM2(a, IDXI_, b); }
    LD1(0) LD1(1) LD1(2) LD1(3) LD1(4) LD1(5) LD1(6) LD1(7)
#undef LD1
    TR8(r0,r1,r2,r3,r4,r5,r6,r7);
    TR8(i0,i1,i2,i3,i4,i5,i6,i7);
    VS(RE + site*8, r0); VS(RE + site*8+8, r1); VS(RE + site*8+16, r2); VS(RE + site*8+24, r3);
    VS(RE + site*8+32, r4); VS(RE + site*8+40, r5); VS(RE + site*8+48, r6); VS(RE + site*8+56, r7);
    VS(IM + site*8, i0); VS(IM + site*8+8, i1); VS(IM + site*8+16, i2); VS(IM + site*8+24, i3);
    VS(IM + site*8+32, i4); VS(IM + site*8+40, i5); VS(IM + site*8+48, i6); VS(IM + site*8+56, i7);
}
// same but with zero-fill for vols >= nv (tail groups)
static void soa_in8_nv(const double* xv, long vstride, long site, double* RE, double* IM, int nv){
    V r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7;
#define LD1(v) if (nv > v) { V a=VLU(xv + v*vstride + 2*site), b=VLU(xv + v*vstride + 2*site + 8);     r##v = PERM2(a, IDXR_, b); i##v = PERM2(a, IDXI_, b); } else { r##v = _mm512_setzero_pd(); i##v = _mm512_setzero_pd(); }
    LD1(0) LD1(1) LD1(2) LD1(3) LD1(4) LD1(5) LD1(6) LD1(7)
#undef LD1
    TR8(r0,r1,r2,r3,r4,r5,r6,r7);
    TR8(i0,i1,i2,i3,i4,i5,i6,i7);
    VS(RE + site*8, r0); VS(RE + site*8+8, r1); VS(RE + site*8+16, r2); VS(RE + site*8+24, r3);
    VS(RE + site*8+32, r4); VS(RE + site*8+40, r5); VS(RE + site*8+48, r6); VS(RE + site*8+56, r7);
    VS(IM + site*8, i0); VS(IM + site*8+8, i1); VS(IM + site*8+16, i2); VS(IM + site*8+24, i3);
    VS(IM + site*8+32, i4); VS(IM + site*8+40, i5); VS(IM + site*8+48, i6); VS(IM + site*8+56, i7);
}
// SoA -> natural (snapshot), 8 sites x nv vols
static void soa_out8(const double* RE, const double* IM, long site, double* ov, long vstride, int nv){
    V r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7;
    r0=VL(RE+site*8); r1=VL(RE+site*8+8); r2=VL(RE+site*8+16); r3=VL(RE+site*8+24);
    r4=VL(RE+site*8+32); r5=VL(RE+site*8+40); r6=VL(RE+site*8+48); r7=VL(RE+site*8+56);
    i0=VL(IM+site*8); i1=VL(IM+site*8+8); i2=VL(IM+site*8+16); i3=VL(IM+site*8+24);
    i4=VL(IM+site*8+32); i5=VL(IM+site*8+40); i6=VL(IM+site*8+48); i7=VL(IM+site*8+56);
    TR8(r0,r1,r2,r3,r4,r5,r6,r7);
    TR8(i0,i1,i2,i3,i4,i5,i6,i7);
#define ST1(v) if (nv > v) { VSU(ov + v*vstride + 2*site, PERM2(r##v, IDXLO_, i##v));     VSU(ov + v*vstride + 2*site + 8, PERM2(r##v, IDXHI_, i##v)); }
    ST1(0) ST1(1) ST1(2) ST1(3) ST1(4) ST1(5) ST1(6) ST1(7)
#undef ST1
}

// ============ L=6 SoA-8 v2 (XS=288) ============
static double S6RE[1728] ALIGN64;
static double S6IM[1728] ALIGN64;
static double C6RE[1728] ALIGN64;
static double C6IM[1728] ALIGN64;
static __attribute__((noinline)) void p6_zz(double* PR, double* PI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        {
        V x0r = VL(pr + 0), x0i = VL(pi + 0);
        V x1r = VL(pr + 8), x1i = VL(pi + 8);
        V x2r = VL(pr + 16), x2i = VL(pi + 16);
        V x3r = VL(pr + 24), x3i = VL(pi + 24);
        V x4r = VL(pr + 32), x4i = VL(pi + 32);
        V x5r = VL(pr + 40), x5i = VL(pi + 40);
        V a00r = VADD(x0r, x3r), a00i = VADD(x0i, x3i);
        V a01r = VSUB(x0r, x3r), a01i = VSUB(x0i, x3i);
        V a10r = VADD(x2r, x5r), a10i = VADD(x2i, x5i);
        V a11r = VSUB(x2r, x5r), a11i = VSUB(x2i, x5i);
        V a20r = VADD(x4r, x1r), a20i = VADD(x4i, x1i);
        V a21r = VSUB(x4r, x1r), a21i = VSUB(x4i, x1i);
        V o00r, o00i, o01r, o01i, o02r, o02i;
        { V ur=VADD(a10r,a20r), ui=VADD(a10i,a20i);
          V vr=VSUB(a10r,a20r), vi=VSUB(a10i,a20i);
          V sr=VFMA(ur,VSET1(-0.5),a00r), si=VFMA(ui,VSET1(-0.5),a00i);
          o00r=VADD(a00r,ur); o00i=VADD(a00i,ui);
          o01r=VFMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o01i=VFNMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
          o02r=VFNMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o02i=VFMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
        }
        V o10r, o10i, o11r, o11i, o12r, o12i;
        { V ur=VADD(a11r,a21r), ui=VADD(a11i,a21i);
          V vr=VSUB(a11r,a21r), vi=VSUB(a11i,a21i);
          V sr=VFMA(ur,VSET1(-0.5),a01r), si=VFMA(ui,VSET1(-0.5),a01i);
          o10r=VADD(a01r,ur); o10i=VADD(a01i,ui);
          o11r=VFMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o11i=VFNMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
          o12r=VFNMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o12i=VFMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
        }
        VS(pr + 0, o00r); VS(pi + 0, o00i);
        VS(pr + 32, o01r); VS(pi + 32, o01i);
        VS(pr + 16, o02r); VS(pi + 16, o02i);
        VS(pr + 24, o10r); VS(pi + 24, o10i);
        VS(pr + 8, o11r); VS(pi + 8, o11i);
        VS(pr + 40, o12r); VS(pi + 40, o12i);
        }
    }
}
static __attribute__((noinline)) void p6_yy(double* PR, double* PI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        {
        V x0r = VL(pr + 0), x0i = VL(pi + 0);
        V x1r = VL(pr + 48), x1i = VL(pi + 48);
        V x2r = VL(pr + 96), x2i = VL(pi + 96);
        V x3r = VL(pr + 144), x3i = VL(pi + 144);
        V x4r = VL(pr + 192), x4i = VL(pi + 192);
        V x5r = VL(pr + 240), x5i = VL(pi + 240);
        V a00r = VADD(x0r, x3r), a00i = VADD(x0i, x3i);
        V a01r = VSUB(x0r, x3r), a01i = VSUB(x0i, x3i);
        V a10r = VADD(x2r, x5r), a10i = VADD(x2i, x5i);
        V a11r = VSUB(x2r, x5r), a11i = VSUB(x2i, x5i);
        V a20r = VADD(x4r, x1r), a20i = VADD(x4i, x1i);
        V a21r = VSUB(x4r, x1r), a21i = VSUB(x4i, x1i);
        V o00r, o00i, o01r, o01i, o02r, o02i;
        { V ur=VADD(a10r,a20r), ui=VADD(a10i,a20i);
          V vr=VSUB(a10r,a20r), vi=VSUB(a10i,a20i);
          V sr=VFMA(ur,VSET1(-0.5),a00r), si=VFMA(ui,VSET1(-0.5),a00i);
          o00r=VADD(a00r,ur); o00i=VADD(a00i,ui);
          o01r=VFMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o01i=VFNMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
          o02r=VFNMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o02i=VFMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
        }
        V o10r, o10i, o11r, o11i, o12r, o12i;
        { V ur=VADD(a11r,a21r), ui=VADD(a11i,a21i);
          V vr=VSUB(a11r,a21r), vi=VSUB(a11i,a21i);
          V sr=VFMA(ur,VSET1(-0.5),a01r), si=VFMA(ui,VSET1(-0.5),a01i);
          o10r=VADD(a01r,ur); o10i=VADD(a01i,ui);
          o11r=VFMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o11i=VFNMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
          o12r=VFNMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o12i=VFMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
        }
        VS(pr + 0, o00r); VS(pi + 0, o00i);
        VS(pr + 192, o01r); VS(pi + 192, o01i);
        VS(pr + 96, o02r); VS(pi + 96, o02i);
        VS(pr + 144, o10r); VS(pi + 144, o10i);
        VS(pr + 48, o11r); VS(pi + 48, o11i);
        VS(pr + 240, o12r); VS(pi + 240, o12i);
        }
    }
}
static __attribute__((noinline)) void p6_xx(double* PR, double* PI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        {
        V x0r = VL(pr + 0), x0i = VL(pi + 0);
        V x1r = VL(pr + 288), x1i = VL(pi + 288);
        V x2r = VL(pr + 576), x2i = VL(pi + 576);
        V x3r = VL(pr + 864), x3i = VL(pi + 864);
        V x4r = VL(pr + 1152), x4i = VL(pi + 1152);
        V x5r = VL(pr + 1440), x5i = VL(pi + 1440);
        V a00r = VADD(x0r, x3r), a00i = VADD(x0i, x3i);
        V a01r = VSUB(x0r, x3r), a01i = VSUB(x0i, x3i);
        V a10r = VADD(x2r, x5r), a10i = VADD(x2i, x5i);
        V a11r = VSUB(x2r, x5r), a11i = VSUB(x2i, x5i);
        V a20r = VADD(x4r, x1r), a20i = VADD(x4i, x1i);
        V a21r = VSUB(x4r, x1r), a21i = VSUB(x4i, x1i);
        V o00r, o00i, o01r, o01i, o02r, o02i;
        { V ur=VADD(a10r,a20r), ui=VADD(a10i,a20i);
          V vr=VSUB(a10r,a20r), vi=VSUB(a10i,a20i);
          V sr=VFMA(ur,VSET1(-0.5),a00r), si=VFMA(ui,VSET1(-0.5),a00i);
          o00r=VADD(a00r,ur); o00i=VADD(a00i,ui);
          o01r=VFMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o01i=VFNMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
          o02r=VFNMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o02i=VFMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
        }
        V o10r, o10i, o11r, o11i, o12r, o12i;
        { V ur=VADD(a11r,a21r), ui=VADD(a11i,a21i);
          V vr=VSUB(a11r,a21r), vi=VSUB(a11i,a21i);
          V sr=VFMA(ur,VSET1(-0.5),a01r), si=VFMA(ui,VSET1(-0.5),a01i);
          o10r=VADD(a01r,ur); o10i=VADD(a01i,ui);
          o11r=VFMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o11i=VFNMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
          o12r=VFNMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o12i=VFMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
        }
        VS(pr + 0, o00r); VS(pi + 0, o00i);
        VS(pr + 1152, o01r); VS(pi + 1152, o01i);
        VS(pr + 576, o02r); VS(pi + 576, o02i);
        VS(pr + 864, o10r); VS(pi + 864, o10i);
        VS(pr + 288, o11r); VS(pi + 288, o11i);
        VS(pr + 1440, o12r); VS(pi + 1440, o12i);
        }
    }
}
static __attribute__((noinline)) void p6_xxm(double* PR, double* PI, const double* CR, const double* CI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        const double* cr = CR + q_*pstep; const double* ci = CI + q_*pstep;
        {
        V x0r = VL(pr + 0), x0i = VL(pi + 0);
        V x1r = VL(pr + 288), x1i = VL(pi + 288);
        V x2r = VL(pr + 576), x2i = VL(pi + 576);
        V x3r = VL(pr + 864), x3i = VL(pi + 864);
        V x4r = VL(pr + 1152), x4i = VL(pi + 1152);
        V x5r = VL(pr + 1440), x5i = VL(pi + 1440);
        V a00r = VADD(x0r, x3r), a00i = VADD(x0i, x3i);
        V a01r = VSUB(x0r, x3r), a01i = VSUB(x0i, x3i);
        V a10r = VADD(x2r, x5r), a10i = VADD(x2i, x5i);
        V a11r = VSUB(x2r, x5r), a11i = VSUB(x2i, x5i);
        V a20r = VADD(x4r, x1r), a20i = VADD(x4i, x1i);
        V a21r = VSUB(x4r, x1r), a21i = VSUB(x4i, x1i);
        V o00r, o00i, o01r, o01i, o02r, o02i;
        { V ur=VADD(a10r,a20r), ui=VADD(a10i,a20i);
          V vr=VSUB(a10r,a20r), vi=VSUB(a10i,a20i);
          V sr=VFMA(ur,VSET1(-0.5),a00r), si=VFMA(ui,VSET1(-0.5),a00i);
          o00r=VADD(a00r,ur); o00i=VADD(a00i,ui);
          o01r=VFMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o01i=VFNMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
          o02r=VFNMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o02i=VFMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
        }
        V o10r, o10i, o11r, o11i, o12r, o12i;
        { V ur=VADD(a11r,a21r), ui=VADD(a11i,a21i);
          V vr=VSUB(a11r,a21r), vi=VSUB(a11i,a21i);
          V sr=VFMA(ur,VSET1(-0.5),a01r), si=VFMA(ui,VSET1(-0.5),a01i);
          o10r=VADD(a01r,ur); o10i=VADD(a01i,ui);
          o11r=VFMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o11i=VFNMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
          o12r=VFNMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o12i=VFMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
        }
        { V zr_ = VADD(o00r, VL(cr + 0)), zi_ = VADD(o00i, VL(ci + 0));
          MAP2(zr_, zi_);
          VS(pr + 0, zr_); VS(pi + 0, zi_); }
        { V zr_ = VADD(o01r, VL(cr + 1152)), zi_ = VADD(o01i, VL(ci + 1152));
          MAP2(zr_, zi_);
          VS(pr + 1152, zr_); VS(pi + 1152, zi_); }
        { V zr_ = VADD(o02r, VL(cr + 576)), zi_ = VADD(o02i, VL(ci + 576));
          MAP2(zr_, zi_);
          VS(pr + 576, zr_); VS(pi + 576, zi_); }
        { V zr_ = VADD(o10r, VL(cr + 864)), zi_ = VADD(o10i, VL(ci + 864));
          MAP2(zr_, zi_);
          VS(pr + 864, zr_); VS(pi + 864, zi_); }
        { V zr_ = VADD(o11r, VL(cr + 288)), zi_ = VADD(o11i, VL(ci + 288));
          MAP2(zr_, zi_);
          VS(pr + 288, zr_); VS(pi + 288, zi_); }
        { V zr_ = VADD(o12r, VL(cr + 1440)), zi_ = VADD(o12i, VL(ci + 1440));
          MAP2(zr_, zi_);
          VS(pr + 1440, zr_); VS(pi + 1440, zi_); }
        }
    }
}
static __attribute__((noinline)) void p6_yym(double* PR, double* PI, const double* CR, const double* CI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        const double* cr = CR + q_*pstep; const double* ci = CI + q_*pstep;
        {
        V x0r = VL(pr + 0), x0i = VL(pi + 0);
        V x1r = VL(pr + 48), x1i = VL(pi + 48);
        V x2r = VL(pr + 96), x2i = VL(pi + 96);
        V x3r = VL(pr + 144), x3i = VL(pi + 144);
        V x4r = VL(pr + 192), x4i = VL(pi + 192);
        V x5r = VL(pr + 240), x5i = VL(pi + 240);
        V a00r = VADD(x0r, x3r), a00i = VADD(x0i, x3i);
        V a01r = VSUB(x0r, x3r), a01i = VSUB(x0i, x3i);
        V a10r = VADD(x2r, x5r), a10i = VADD(x2i, x5i);
        V a11r = VSUB(x2r, x5r), a11i = VSUB(x2i, x5i);
        V a20r = VADD(x4r, x1r), a20i = VADD(x4i, x1i);
        V a21r = VSUB(x4r, x1r), a21i = VSUB(x4i, x1i);
        V o00r, o00i, o01r, o01i, o02r, o02i;
        { V ur=VADD(a10r,a20r), ui=VADD(a10i,a20i);
          V vr=VSUB(a10r,a20r), vi=VSUB(a10i,a20i);
          V sr=VFMA(ur,VSET1(-0.5),a00r), si=VFMA(ui,VSET1(-0.5),a00i);
          o00r=VADD(a00r,ur); o00i=VADD(a00i,ui);
          o01r=VFMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o01i=VFNMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
          o02r=VFNMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o02i=VFMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
        }
        V o10r, o10i, o11r, o11i, o12r, o12i;
        { V ur=VADD(a11r,a21r), ui=VADD(a11i,a21i);
          V vr=VSUB(a11r,a21r), vi=VSUB(a11i,a21i);
          V sr=VFMA(ur,VSET1(-0.5),a01r), si=VFMA(ui,VSET1(-0.5),a01i);
          o10r=VADD(a01r,ur); o10i=VADD(a01i,ui);
          o11r=VFMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o11i=VFNMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
          o12r=VFNMA(vi,VSET1(0x1.bb67ae8584caap-1),sr);
          o12i=VFMA(vr,VSET1(0x1.bb67ae8584caap-1),si);
        }
        { V zr_ = VADD(o00r, VL(cr + 0)), zi_ = VADD(o00i, VL(ci + 0));
          MAP2(zr_, zi_);
          VS(pr + 0, zr_); VS(pi + 0, zi_); }
        { V zr_ = VADD(o01r, VL(cr + 192)), zi_ = VADD(o01i, VL(ci + 192));
          MAP2(zr_, zi_);
          VS(pr + 192, zr_); VS(pi + 192, zi_); }
        { V zr_ = VADD(o02r, VL(cr + 96)), zi_ = VADD(o02i, VL(ci + 96));
          MAP2(zr_, zi_);
          VS(pr + 96, zr_); VS(pi + 96, zi_); }
        { V zr_ = VADD(o10r, VL(cr + 144)), zi_ = VADD(o10i, VL(ci + 144));
          MAP2(zr_, zi_);
          VS(pr + 144, zr_); VS(pi + 144, zi_); }
        { V zr_ = VADD(o11r, VL(cr + 48)), zi_ = VADD(o11i, VL(ci + 48));
          MAP2(zr_, zi_);
          VS(pr + 48, zr_); VS(pi + 48, zi_); }
        { V zr_ = VADD(o12r, VL(cr + 240)), zi_ = VADD(o12i, VL(ci + 240));
          MAP2(zr_, zi_);
          VS(pr + 240, zr_); VS(pi + 240, zi_); }
        }
    }
}

static void sw6_SyX(int y0, int pre){
    long b0 = (long)y0*6*8;
    p6_xxm(S6RE + b0, S6IM + b0, C6RE + b0, C6IM + b0, 6, 8);
    if (pre) {
        p6_zz(S6RE + b0, S6IM + b0, 6, 288);
        p6_xx(S6RE + b0, S6IM + b0, 6, 8);
    }
}
static void sw6_PxY(int x0, int pre){
    long b0 = (long)x0*288;
    p6_yym(S6RE + b0, S6IM + b0, C6RE + b0, C6IM + b0, 6, 8);
    if (pre) {
        p6_zz(S6RE + b0, S6IM + b0, 6, (long)6*8);
        p6_yy(S6RE + b0, S6IM + b0, 6, 8);
    }
}
static void prz_6(void){ for (int x = 0; x < 6; x++) p6_zz(S6RE + (long)x*288, S6IM + (long)x*288, 6, (long)6*8); }
void run_6v2(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if (m < 1) m = 1;
    long vs = 2*(long)216;
    for (long g0 = 0; g0 < B; g0 += 8) {
        int nv = (B - g0) < 8 ? (int)(B - g0) : 8;
        const double* xg = x0 + g0*vs;
        const double* cg = c + g0*vs;
        for (int x = 0; x < 6; x++) {
            long sb = (long)x*6*6;
            double* sre = S6RE + (long)x*288; double* sim = S6IM + (long)x*288;
            double* cre = C6RE + (long)x*288; double* cim = C6IM + (long)x*288;
            if (nv == 8) {
                for (long s = 0; s < 32; s += 8) {
                    soa_in8(xg + 2*sb, vs, s, sre, sim);
                    soa_in8(cg + 2*sb, vs, s, cre, cim);
                }
            } else {
                for (long s = 0; s < 32; s += 8) {
                    soa_in8_nv(xg + 2*sb, vs, s, sre, sim, nv);
                    soa_in8_nv(cg + 2*sb, vs, s, cre, cim, nv);
                }
            }
            for (long s = 32; s < 6*6; s++) for (int v = 0; v < 8; v++) {
                sre[s*8+v] = v < nv ? xg[v*vs + 2*(sb+s)] : 0.0;
                sim[s*8+v] = v < nv ? xg[v*vs + 2*(sb+s) + 1] : 0.0;
                cre[s*8+v] = v < nv ? cg[v*vs + 2*(sb+s)] : 0.0;
                cim[s*8+v] = v < nv ? cg[v*vs + 2*(sb+s) + 1] : 0.0;
            }
        }
        for (int x = 0; x < 6; x++) {
            p6_zz(S6RE + (long)x*288, S6IM + (long)x*288, 6, (long)6*8);
            p6_yy(S6RE + (long)x*288, S6IM + (long)x*288, 6, 8);
        }
        for (long t = 1; t <= m; t++) {
            int snap = (t == 1) || (t == m);
            int pre = (t < m);
            int dopre = pre && !snap;
            if (t & 1) { for (int y0 = 0; y0 < 6; y0++) sw6_SyX(y0, dopre); }
            else       { for (int xp = 0; xp < 6; xp++) sw6_PxY(xp, dopre); }
            if (snap) {
                for (int x = 0; x < 6; x++) {
                    long sb = (long)x*6*6;
                    double* sre = S6RE + (long)x*288; double* sim = S6IM + (long)x*288;
                    if (t == 1) {
                        for (long s = 0; s < 32; s += 8) soa_out8(sre, sim, s, out1 + g0*vs + 2*sb, vs, nv);
                        for (long s = 32; s < 6*6; s++) for (int v = 0; v < nv; v++) {
                            out1[g0*vs + v*vs + 2*(sb+s)] = sre[s*8+v]; out1[g0*vs + v*vs + 2*(sb+s)+1] = sim[s*8+v]; }
                    }
                    if (t == m) {
                        for (long s = 0; s < 32; s += 8) soa_out8(sre, sim, s, outm + g0*vs + 2*sb, vs, nv);
                        for (long s = 32; s < 6*6; s++) for (int v = 0; v < nv; v++) {
                            outm[g0*vs + v*vs + 2*(sb+s)] = sre[s*8+v]; outm[g0*vs + v*vs + 2*(sb+s)+1] = sim[s*8+v]; }
                    }
                }
            }
            if (pre && !dopre) {
                prz_6();
                if (t & 1) {
                    for (int y0 = 0; y0 < 6; y0++)
                        p6_xx(S6RE + (long)y0*6*8, S6IM + (long)y0*6*8, 6, 8);
                } else {
                    for (int x0 = 0; x0 < 6; x0++)
                        p6_yy(S6RE + (long)x0*288, S6IM + (long)x0*288, 6, 8);
                }
            }
        }
    }
}

// ============ L=8 SoA-8 v2 (XS=576) ============
static double S8RE[4608] ALIGN64;
static double S8IM[4608] ALIGN64;
static double C8RE[4608] ALIGN64;
static double C8IM[4608] ALIGN64;
static __attribute__((noinline)) void p8_zz(double* PR, double* PI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        {
        V x0r = VL(pr + 0), x0i = VL(pi + 0);
        V x1r = VL(pr + 8), x1i = VL(pi + 8);
        V x2r = VL(pr + 16), x2i = VL(pi + 16);
        V x3r = VL(pr + 24), x3i = VL(pi + 24);
        V x4r = VL(pr + 32), x4i = VL(pi + 32);
        V x5r = VL(pr + 40), x5i = VL(pi + 40);
        V x6r = VL(pr + 48), x6i = VL(pi + 48);
        V x7r = VL(pr + 56), x7i = VL(pi + 56);
        V s0r = VADD(x0r, x4r), s0i = VADD(x0i, x4i);
        V d0r = VSUB(x0r, x4r), d0i = VSUB(x0i, x4i);
        V s1r = VADD(x1r, x5r), s1i = VADD(x1i, x5i);
        V d1r = VSUB(x1r, x5r), d1i = VSUB(x1i, x5i);
        V s2r = VADD(x2r, x6r), s2i = VADD(x2i, x6i);
        V d2r = VSUB(x2r, x6r), d2i = VSUB(x2i, x6i);
        V s3r = VADD(x3r, x7r), s3i = VADD(x3i, x7i);
        V d3r = VSUB(x3r, x7r), d3i = VSUB(x3i, x7i);
        V c_ = VSET1(0x1.6a09e667f3bcdp-1);
        { V tr = VMUL(VADD(d1r, d1i), c_); V ti = VMUL(VSUB(d1i, d1r), c_); d1r = tr; d1i = ti; }
        { V tr = d2i; V ti = _mm512_sub_pd(_mm512_setzero_pd(), d2r); d2r = tr; d2i = ti; }
        { V tr = VMUL(VSUB(d3i, d3r), c_); V ti = VMUL(VADD(d3r, d3i), _mm512_sub_pd(_mm512_setzero_pd(), c_)); d3r = tr; d3i = ti; }
        { V t0r=VADD(s0r,s2r), t0i=VADD(s0i,s2i);
          V t1r=VSUB(s0r,s2r), t1i=VSUB(s0i,s2i);
          V t2r=VADD(s1r,s3r), t2i=VADD(s1i,s3i);
          V t3r=VSUB(s1r,s3r), t3i=VSUB(s1i,s3i);
          V y0r=VADD(t0r,t2r), y0i=VADD(t0i,t2i);
          V y2r=VSUB(t0r,t2r), y2i=VSUB(t0i,t2i);
          V y1r=VADD(t1r,t3i), y1i=VSUB(t1i,t3r);
          V y3r=VSUB(t1r,t3i), y3i=VADD(t1i,t3r);
        VS(pr + 0, y0r); VS(pi + 0, y0i);
        VS(pr + 16, y1r); VS(pi + 16, y1i);
        VS(pr + 32, y2r); VS(pi + 32, y2i);
        VS(pr + 48, y3r); VS(pi + 48, y3i);
        }
        { V t0r=VADD(d0r,d2r), t0i=VADD(d0i,d2i);
          V t1r=VSUB(d0r,d2r), t1i=VSUB(d0i,d2i);
          V t2r=VADD(d1r,d3r), t2i=VADD(d1i,d3i);
          V t3r=VSUB(d1r,d3r), t3i=VSUB(d1i,d3i);
          V y0r=VADD(t0r,t2r), y0i=VADD(t0i,t2i);
          V y2r=VSUB(t0r,t2r), y2i=VSUB(t0i,t2i);
          V y1r=VADD(t1r,t3i), y1i=VSUB(t1i,t3r);
          V y3r=VSUB(t1r,t3i), y3i=VADD(t1i,t3r);
        VS(pr + 8, y0r); VS(pi + 8, y0i);
        VS(pr + 24, y1r); VS(pi + 24, y1i);
        VS(pr + 40, y2r); VS(pi + 40, y2i);
        VS(pr + 56, y3r); VS(pi + 56, y3i);
        }
        }
    }
}
static __attribute__((noinline)) void p8_yy(double* PR, double* PI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        {
        V x0r = VL(pr + 0), x0i = VL(pi + 0);
        V x1r = VL(pr + 64), x1i = VL(pi + 64);
        V x2r = VL(pr + 128), x2i = VL(pi + 128);
        V x3r = VL(pr + 192), x3i = VL(pi + 192);
        V x4r = VL(pr + 256), x4i = VL(pi + 256);
        V x5r = VL(pr + 320), x5i = VL(pi + 320);
        V x6r = VL(pr + 384), x6i = VL(pi + 384);
        V x7r = VL(pr + 448), x7i = VL(pi + 448);
        V s0r = VADD(x0r, x4r), s0i = VADD(x0i, x4i);
        V d0r = VSUB(x0r, x4r), d0i = VSUB(x0i, x4i);
        V s1r = VADD(x1r, x5r), s1i = VADD(x1i, x5i);
        V d1r = VSUB(x1r, x5r), d1i = VSUB(x1i, x5i);
        V s2r = VADD(x2r, x6r), s2i = VADD(x2i, x6i);
        V d2r = VSUB(x2r, x6r), d2i = VSUB(x2i, x6i);
        V s3r = VADD(x3r, x7r), s3i = VADD(x3i, x7i);
        V d3r = VSUB(x3r, x7r), d3i = VSUB(x3i, x7i);
        V c_ = VSET1(0x1.6a09e667f3bcdp-1);
        { V tr = VMUL(VADD(d1r, d1i), c_); V ti = VMUL(VSUB(d1i, d1r), c_); d1r = tr; d1i = ti; }
        { V tr = d2i; V ti = _mm512_sub_pd(_mm512_setzero_pd(), d2r); d2r = tr; d2i = ti; }
        { V tr = VMUL(VSUB(d3i, d3r), c_); V ti = VMUL(VADD(d3r, d3i), _mm512_sub_pd(_mm512_setzero_pd(), c_)); d3r = tr; d3i = ti; }
        { V t0r=VADD(s0r,s2r), t0i=VADD(s0i,s2i);
          V t1r=VSUB(s0r,s2r), t1i=VSUB(s0i,s2i);
          V t2r=VADD(s1r,s3r), t2i=VADD(s1i,s3i);
          V t3r=VSUB(s1r,s3r), t3i=VSUB(s1i,s3i);
          V y0r=VADD(t0r,t2r), y0i=VADD(t0i,t2i);
          V y2r=VSUB(t0r,t2r), y2i=VSUB(t0i,t2i);
          V y1r=VADD(t1r,t3i), y1i=VSUB(t1i,t3r);
          V y3r=VSUB(t1r,t3i), y3i=VADD(t1i,t3r);
        VS(pr + 0, y0r); VS(pi + 0, y0i);
        VS(pr + 128, y1r); VS(pi + 128, y1i);
        VS(pr + 256, y2r); VS(pi + 256, y2i);
        VS(pr + 384, y3r); VS(pi + 384, y3i);
        }
        { V t0r=VADD(d0r,d2r), t0i=VADD(d0i,d2i);
          V t1r=VSUB(d0r,d2r), t1i=VSUB(d0i,d2i);
          V t2r=VADD(d1r,d3r), t2i=VADD(d1i,d3i);
          V t3r=VSUB(d1r,d3r), t3i=VSUB(d1i,d3i);
          V y0r=VADD(t0r,t2r), y0i=VADD(t0i,t2i);
          V y2r=VSUB(t0r,t2r), y2i=VSUB(t0i,t2i);
          V y1r=VADD(t1r,t3i), y1i=VSUB(t1i,t3r);
          V y3r=VSUB(t1r,t3i), y3i=VADD(t1i,t3r);
        VS(pr + 64, y0r); VS(pi + 64, y0i);
        VS(pr + 192, y1r); VS(pi + 192, y1i);
        VS(pr + 320, y2r); VS(pi + 320, y2i);
        VS(pr + 448, y3r); VS(pi + 448, y3i);
        }
        }
    }
}
static __attribute__((noinline)) void p8_xx(double* PR, double* PI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        {
        V x0r = VL(pr + 0), x0i = VL(pi + 0);
        V x1r = VL(pr + 576), x1i = VL(pi + 576);
        V x2r = VL(pr + 1152), x2i = VL(pi + 1152);
        V x3r = VL(pr + 1728), x3i = VL(pi + 1728);
        V x4r = VL(pr + 2304), x4i = VL(pi + 2304);
        V x5r = VL(pr + 2880), x5i = VL(pi + 2880);
        V x6r = VL(pr + 3456), x6i = VL(pi + 3456);
        V x7r = VL(pr + 4032), x7i = VL(pi + 4032);
        V s0r = VADD(x0r, x4r), s0i = VADD(x0i, x4i);
        V d0r = VSUB(x0r, x4r), d0i = VSUB(x0i, x4i);
        V s1r = VADD(x1r, x5r), s1i = VADD(x1i, x5i);
        V d1r = VSUB(x1r, x5r), d1i = VSUB(x1i, x5i);
        V s2r = VADD(x2r, x6r), s2i = VADD(x2i, x6i);
        V d2r = VSUB(x2r, x6r), d2i = VSUB(x2i, x6i);
        V s3r = VADD(x3r, x7r), s3i = VADD(x3i, x7i);
        V d3r = VSUB(x3r, x7r), d3i = VSUB(x3i, x7i);
        V c_ = VSET1(0x1.6a09e667f3bcdp-1);
        { V tr = VMUL(VADD(d1r, d1i), c_); V ti = VMUL(VSUB(d1i, d1r), c_); d1r = tr; d1i = ti; }
        { V tr = d2i; V ti = _mm512_sub_pd(_mm512_setzero_pd(), d2r); d2r = tr; d2i = ti; }
        { V tr = VMUL(VSUB(d3i, d3r), c_); V ti = VMUL(VADD(d3r, d3i), _mm512_sub_pd(_mm512_setzero_pd(), c_)); d3r = tr; d3i = ti; }
        { V t0r=VADD(s0r,s2r), t0i=VADD(s0i,s2i);
          V t1r=VSUB(s0r,s2r), t1i=VSUB(s0i,s2i);
          V t2r=VADD(s1r,s3r), t2i=VADD(s1i,s3i);
          V t3r=VSUB(s1r,s3r), t3i=VSUB(s1i,s3i);
          V y0r=VADD(t0r,t2r), y0i=VADD(t0i,t2i);
          V y2r=VSUB(t0r,t2r), y2i=VSUB(t0i,t2i);
          V y1r=VADD(t1r,t3i), y1i=VSUB(t1i,t3r);
          V y3r=VSUB(t1r,t3i), y3i=VADD(t1i,t3r);
        VS(pr + 0, y0r); VS(pi + 0, y0i);
        VS(pr + 1152, y1r); VS(pi + 1152, y1i);
        VS(pr + 2304, y2r); VS(pi + 2304, y2i);
        VS(pr + 3456, y3r); VS(pi + 3456, y3i);
        }
        { V t0r=VADD(d0r,d2r), t0i=VADD(d0i,d2i);
          V t1r=VSUB(d0r,d2r), t1i=VSUB(d0i,d2i);
          V t2r=VADD(d1r,d3r), t2i=VADD(d1i,d3i);
          V t3r=VSUB(d1r,d3r), t3i=VSUB(d1i,d3i);
          V y0r=VADD(t0r,t2r), y0i=VADD(t0i,t2i);
          V y2r=VSUB(t0r,t2r), y2i=VSUB(t0i,t2i);
          V y1r=VADD(t1r,t3i), y1i=VSUB(t1i,t3r);
          V y3r=VSUB(t1r,t3i), y3i=VADD(t1i,t3r);
        VS(pr + 576, y0r); VS(pi + 576, y0i);
        VS(pr + 1728, y1r); VS(pi + 1728, y1i);
        VS(pr + 2880, y2r); VS(pi + 2880, y2i);
        VS(pr + 4032, y3r); VS(pi + 4032, y3i);
        }
        }
    }
}
static __attribute__((noinline)) void p8_xxm(double* PR, double* PI, const double* CR, const double* CI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        const double* cr = CR + q_*pstep; const double* ci = CI + q_*pstep;
        {
        V x0r = VL(pr + 0), x0i = VL(pi + 0);
        V x1r = VL(pr + 576), x1i = VL(pi + 576);
        V x2r = VL(pr + 1152), x2i = VL(pi + 1152);
        V x3r = VL(pr + 1728), x3i = VL(pi + 1728);
        V x4r = VL(pr + 2304), x4i = VL(pi + 2304);
        V x5r = VL(pr + 2880), x5i = VL(pi + 2880);
        V x6r = VL(pr + 3456), x6i = VL(pi + 3456);
        V x7r = VL(pr + 4032), x7i = VL(pi + 4032);
        V s0r = VADD(x0r, x4r), s0i = VADD(x0i, x4i);
        V d0r = VSUB(x0r, x4r), d0i = VSUB(x0i, x4i);
        V s1r = VADD(x1r, x5r), s1i = VADD(x1i, x5i);
        V d1r = VSUB(x1r, x5r), d1i = VSUB(x1i, x5i);
        V s2r = VADD(x2r, x6r), s2i = VADD(x2i, x6i);
        V d2r = VSUB(x2r, x6r), d2i = VSUB(x2i, x6i);
        V s3r = VADD(x3r, x7r), s3i = VADD(x3i, x7i);
        V d3r = VSUB(x3r, x7r), d3i = VSUB(x3i, x7i);
        V c_ = VSET1(0x1.6a09e667f3bcdp-1);
        { V tr = VMUL(VADD(d1r, d1i), c_); V ti = VMUL(VSUB(d1i, d1r), c_); d1r = tr; d1i = ti; }
        { V tr = d2i; V ti = _mm512_sub_pd(_mm512_setzero_pd(), d2r); d2r = tr; d2i = ti; }
        { V tr = VMUL(VSUB(d3i, d3r), c_); V ti = VMUL(VADD(d3r, d3i), _mm512_sub_pd(_mm512_setzero_pd(), c_)); d3r = tr; d3i = ti; }
        { V t0r=VADD(s0r,s2r), t0i=VADD(s0i,s2i);
          V t1r=VSUB(s0r,s2r), t1i=VSUB(s0i,s2i);
          V t2r=VADD(s1r,s3r), t2i=VADD(s1i,s3i);
          V t3r=VSUB(s1r,s3r), t3i=VSUB(s1i,s3i);
          V y0r=VADD(t0r,t2r), y0i=VADD(t0i,t2i);
          V y2r=VSUB(t0r,t2r), y2i=VSUB(t0i,t2i);
          V y1r=VADD(t1r,t3i), y1i=VSUB(t1i,t3r);
          V y3r=VSUB(t1r,t3i), y3i=VADD(t1i,t3r);
        { V zr_ = VADD(y0r, VL(cr + 0)), zi_ = VADD(y0i, VL(ci + 0));
          MAP2(zr_, zi_);
          VS(pr + 0, zr_); VS(pi + 0, zi_); }
        { V zr_ = VADD(y1r, VL(cr + 1152)), zi_ = VADD(y1i, VL(ci + 1152));
          MAP2(zr_, zi_);
          VS(pr + 1152, zr_); VS(pi + 1152, zi_); }
        { V zr_ = VADD(y2r, VL(cr + 2304)), zi_ = VADD(y2i, VL(ci + 2304));
          MAP2(zr_, zi_);
          VS(pr + 2304, zr_); VS(pi + 2304, zi_); }
        { V zr_ = VADD(y3r, VL(cr + 3456)), zi_ = VADD(y3i, VL(ci + 3456));
          MAP2(zr_, zi_);
          VS(pr + 3456, zr_); VS(pi + 3456, zi_); }
        }
        { V t0r=VADD(d0r,d2r), t0i=VADD(d0i,d2i);
          V t1r=VSUB(d0r,d2r), t1i=VSUB(d0i,d2i);
          V t2r=VADD(d1r,d3r), t2i=VADD(d1i,d3i);
          V t3r=VSUB(d1r,d3r), t3i=VSUB(d1i,d3i);
          V y0r=VADD(t0r,t2r), y0i=VADD(t0i,t2i);
          V y2r=VSUB(t0r,t2r), y2i=VSUB(t0i,t2i);
          V y1r=VADD(t1r,t3i), y1i=VSUB(t1i,t3r);
          V y3r=VSUB(t1r,t3i), y3i=VADD(t1i,t3r);
        { V zr_ = VADD(y0r, VL(cr + 576)), zi_ = VADD(y0i, VL(ci + 576));
          MAP2(zr_, zi_);
          VS(pr + 576, zr_); VS(pi + 576, zi_); }
        { V zr_ = VADD(y1r, VL(cr + 1728)), zi_ = VADD(y1i, VL(ci + 1728));
          MAP2(zr_, zi_);
          VS(pr + 1728, zr_); VS(pi + 1728, zi_); }
        { V zr_ = VADD(y2r, VL(cr + 2880)), zi_ = VADD(y2i, VL(ci + 2880));
          MAP2(zr_, zi_);
          VS(pr + 2880, zr_); VS(pi + 2880, zi_); }
        { V zr_ = VADD(y3r, VL(cr + 4032)), zi_ = VADD(y3i, VL(ci + 4032));
          MAP2(zr_, zi_);
          VS(pr + 4032, zr_); VS(pi + 4032, zi_); }
        }
        }
    }
}
static __attribute__((noinline)) void p8_yym(double* PR, double* PI, const double* CR, const double* CI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        const double* cr = CR + q_*pstep; const double* ci = CI + q_*pstep;
        {
        V x0r = VL(pr + 0), x0i = VL(pi + 0);
        V x1r = VL(pr + 64), x1i = VL(pi + 64);
        V x2r = VL(pr + 128), x2i = VL(pi + 128);
        V x3r = VL(pr + 192), x3i = VL(pi + 192);
        V x4r = VL(pr + 256), x4i = VL(pi + 256);
        V x5r = VL(pr + 320), x5i = VL(pi + 320);
        V x6r = VL(pr + 384), x6i = VL(pi + 384);
        V x7r = VL(pr + 448), x7i = VL(pi + 448);
        V s0r = VADD(x0r, x4r), s0i = VADD(x0i, x4i);
        V d0r = VSUB(x0r, x4r), d0i = VSUB(x0i, x4i);
        V s1r = VADD(x1r, x5r), s1i = VADD(x1i, x5i);
        V d1r = VSUB(x1r, x5r), d1i = VSUB(x1i, x5i);
        V s2r = VADD(x2r, x6r), s2i = VADD(x2i, x6i);
        V d2r = VSUB(x2r, x6r), d2i = VSUB(x2i, x6i);
        V s3r = VADD(x3r, x7r), s3i = VADD(x3i, x7i);
        V d3r = VSUB(x3r, x7r), d3i = VSUB(x3i, x7i);
        V c_ = VSET1(0x1.6a09e667f3bcdp-1);
        { V tr = VMUL(VADD(d1r, d1i), c_); V ti = VMUL(VSUB(d1i, d1r), c_); d1r = tr; d1i = ti; }
        { V tr = d2i; V ti = _mm512_sub_pd(_mm512_setzero_pd(), d2r); d2r = tr; d2i = ti; }
        { V tr = VMUL(VSUB(d3i, d3r), c_); V ti = VMUL(VADD(d3r, d3i), _mm512_sub_pd(_mm512_setzero_pd(), c_)); d3r = tr; d3i = ti; }
        { V t0r=VADD(s0r,s2r), t0i=VADD(s0i,s2i);
          V t1r=VSUB(s0r,s2r), t1i=VSUB(s0i,s2i);
          V t2r=VADD(s1r,s3r), t2i=VADD(s1i,s3i);
          V t3r=VSUB(s1r,s3r), t3i=VSUB(s1i,s3i);
          V y0r=VADD(t0r,t2r), y0i=VADD(t0i,t2i);
          V y2r=VSUB(t0r,t2r), y2i=VSUB(t0i,t2i);
          V y1r=VADD(t1r,t3i), y1i=VSUB(t1i,t3r);
          V y3r=VSUB(t1r,t3i), y3i=VADD(t1i,t3r);
        { V zr_ = VADD(y0r, VL(cr + 0)), zi_ = VADD(y0i, VL(ci + 0));
          MAP2(zr_, zi_);
          VS(pr + 0, zr_); VS(pi + 0, zi_); }
        { V zr_ = VADD(y1r, VL(cr + 128)), zi_ = VADD(y1i, VL(ci + 128));
          MAP2(zr_, zi_);
          VS(pr + 128, zr_); VS(pi + 128, zi_); }
        { V zr_ = VADD(y2r, VL(cr + 256)), zi_ = VADD(y2i, VL(ci + 256));
          MAP2(zr_, zi_);
          VS(pr + 256, zr_); VS(pi + 256, zi_); }
        { V zr_ = VADD(y3r, VL(cr + 384)), zi_ = VADD(y3i, VL(ci + 384));
          MAP2(zr_, zi_);
          VS(pr + 384, zr_); VS(pi + 384, zi_); }
        }
        { V t0r=VADD(d0r,d2r), t0i=VADD(d0i,d2i);
          V t1r=VSUB(d0r,d2r), t1i=VSUB(d0i,d2i);
          V t2r=VADD(d1r,d3r), t2i=VADD(d1i,d3i);
          V t3r=VSUB(d1r,d3r), t3i=VSUB(d1i,d3i);
          V y0r=VADD(t0r,t2r), y0i=VADD(t0i,t2i);
          V y2r=VSUB(t0r,t2r), y2i=VSUB(t0i,t2i);
          V y1r=VADD(t1r,t3i), y1i=VSUB(t1i,t3r);
          V y3r=VSUB(t1r,t3i), y3i=VADD(t1i,t3r);
        { V zr_ = VADD(y0r, VL(cr + 64)), zi_ = VADD(y0i, VL(ci + 64));
          MAP2(zr_, zi_);
          VS(pr + 64, zr_); VS(pi + 64, zi_); }
        { V zr_ = VADD(y1r, VL(cr + 192)), zi_ = VADD(y1i, VL(ci + 192));
          MAP2(zr_, zi_);
          VS(pr + 192, zr_); VS(pi + 192, zi_); }
        { V zr_ = VADD(y2r, VL(cr + 320)), zi_ = VADD(y2i, VL(ci + 320));
          MAP2(zr_, zi_);
          VS(pr + 320, zr_); VS(pi + 320, zi_); }
        { V zr_ = VADD(y3r, VL(cr + 448)), zi_ = VADD(y3i, VL(ci + 448));
          MAP2(zr_, zi_);
          VS(pr + 448, zr_); VS(pi + 448, zi_); }
        }
        }
    }
}

static void sw8_SyX(int y0, int pre){
    long b0 = (long)y0*8*8;
    p8_xxm(S8RE + b0, S8IM + b0, C8RE + b0, C8IM + b0, 8, 8);
    if (pre) {
        p8_zz(S8RE + b0, S8IM + b0, 8, 576);
        p8_xx(S8RE + b0, S8IM + b0, 8, 8);
    }
}
static void sw8_PxY(int x0, int pre){
    long b0 = (long)x0*576;
    p8_yym(S8RE + b0, S8IM + b0, C8RE + b0, C8IM + b0, 8, 8);
    if (pre) {
        p8_zz(S8RE + b0, S8IM + b0, 8, (long)8*8);
        p8_yy(S8RE + b0, S8IM + b0, 8, 8);
    }
}
static void prz_8(void){ for (int x = 0; x < 8; x++) p8_zz(S8RE + (long)x*576, S8IM + (long)x*576, 8, (long)8*8); }
void run_8v2(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if (m < 1) m = 1;
    long vs = 2*(long)512;
    for (long g0 = 0; g0 < B; g0 += 8) {
        int nv = (B - g0) < 8 ? (int)(B - g0) : 8;
        const double* xg = x0 + g0*vs;
        const double* cg = c + g0*vs;
        for (int x = 0; x < 8; x++) {
            long sb = (long)x*8*8;
            double* sre = S8RE + (long)x*576; double* sim = S8IM + (long)x*576;
            double* cre = C8RE + (long)x*576; double* cim = C8IM + (long)x*576;
            if (nv == 8) {
                for (long s = 0; s < 64; s += 8) {
                    soa_in8(xg + 2*sb, vs, s, sre, sim);
                    soa_in8(cg + 2*sb, vs, s, cre, cim);
                }
            } else {
                for (long s = 0; s < 64; s += 8) {
                    soa_in8_nv(xg + 2*sb, vs, s, sre, sim, nv);
                    soa_in8_nv(cg + 2*sb, vs, s, cre, cim, nv);
                }
            }
            for (long s = 64; s < 8*8; s++) for (int v = 0; v < 8; v++) {
                sre[s*8+v] = v < nv ? xg[v*vs + 2*(sb+s)] : 0.0;
                sim[s*8+v] = v < nv ? xg[v*vs + 2*(sb+s) + 1] : 0.0;
                cre[s*8+v] = v < nv ? cg[v*vs + 2*(sb+s)] : 0.0;
                cim[s*8+v] = v < nv ? cg[v*vs + 2*(sb+s) + 1] : 0.0;
            }
        }
        for (int x = 0; x < 8; x++) {
            p8_zz(S8RE + (long)x*576, S8IM + (long)x*576, 8, (long)8*8);
            p8_yy(S8RE + (long)x*576, S8IM + (long)x*576, 8, 8);
        }
        for (long t = 1; t <= m; t++) {
            int snap = (t == 1) || (t == m);
            int pre = (t < m);
            int dopre = pre && !snap;
            if (t & 1) { for (int y0 = 0; y0 < 8; y0++) sw8_SyX(y0, dopre); }
            else       { for (int xp = 0; xp < 8; xp++) sw8_PxY(xp, dopre); }
            if (snap) {
                for (int x = 0; x < 8; x++) {
                    long sb = (long)x*8*8;
                    double* sre = S8RE + (long)x*576; double* sim = S8IM + (long)x*576;
                    if (t == 1) {
                        for (long s = 0; s < 64; s += 8) soa_out8(sre, sim, s, out1 + g0*vs + 2*sb, vs, nv);
                        for (long s = 64; s < 8*8; s++) for (int v = 0; v < nv; v++) {
                            out1[g0*vs + v*vs + 2*(sb+s)] = sre[s*8+v]; out1[g0*vs + v*vs + 2*(sb+s)+1] = sim[s*8+v]; }
                    }
                    if (t == m) {
                        for (long s = 0; s < 64; s += 8) soa_out8(sre, sim, s, outm + g0*vs + 2*sb, vs, nv);
                        for (long s = 64; s < 8*8; s++) for (int v = 0; v < nv; v++) {
                            outm[g0*vs + v*vs + 2*(sb+s)] = sre[s*8+v]; outm[g0*vs + v*vs + 2*(sb+s)+1] = sim[s*8+v]; }
                    }
                }
            }
            if (pre && !dopre) {
                prz_8();
                if (t & 1) {
                    for (int y0 = 0; y0 < 8; y0++)
                        p8_xx(S8RE + (long)y0*8*8, S8IM + (long)y0*8*8, 8, 8);
                } else {
                    for (int x0 = 0; x0 < 8; x0++)
                        p8_yy(S8RE + (long)x0*576, S8IM + (long)x0*576, 8, 8);
                }
            }
        }
    }
}

