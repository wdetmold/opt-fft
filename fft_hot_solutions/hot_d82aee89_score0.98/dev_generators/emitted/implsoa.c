
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

static double PSCR[400] ALIGN64;
static double USCR[400] ALIGN64;

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

// ============ L=13 SoA-8 ============
static double S13RE[17576] ALIGN64;
static double S13IM[17576] ALIGN64;
static double C13RE[17576] ALIGN64;
static double C13IM[17576] ALIGN64;
static const double CT13[6] ALIGN64 = { 0x1.c55a7e00740e9p-1, 0x1.22d961ea71119p-1, 0x1.edb7debaa3ed5p-4, -0x1.6b1d8b2365d9ep-2, -0x1.7f3ccd0032e0dp-1, -0x1.f11f493053d00p-1 };
static const double ST13[6] ALIGN64 = { 0x1.dbe064267c47bp-2, 0x1.a55e242a4c3d2p-1, 0x1.fc44566966769p-1, 0x1.deba72ef20147p-1, 0x1.5384d024c2f84p-1, 0x1.ea1e54bc48dbcp-3 };
static __attribute__((noinline)) void p13_zz(double* PR, double* PI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        {
        V S1 = VSET1(0x1.dbe064267c47bp-2);
        V S2 = VSET1(0x1.a55e242a4c3d2p-1);
        V S3 = VSET1(0x1.fc44566966769p-1);
        V S4 = VSET1(0x1.deba72ef20147p-1);
        V S5 = VSET1(0x1.5384d024c2f84p-1);
        V S6 = VSET1(0x1.ea1e54bc48dbcp-3);
        V D1 = _mm512_setzero_pd(), E1 = _mm512_setzero_pd();
        V D2 = _mm512_setzero_pd(), E2 = _mm512_setzero_pd();
        V D3 = _mm512_setzero_pd(), E3 = _mm512_setzero_pd();
        V D4 = _mm512_setzero_pd(), E4 = _mm512_setzero_pd();
        V D5 = _mm512_setzero_pd(), E5 = _mm512_setzero_pd();
        V D6 = _mm512_setzero_pd(), E6 = _mm512_setzero_pd();
        { V ar = VL(pr + 8), ai = VL(pi + 8);
          V br = VL(pr + 96), bi = VL(pi + 96);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 0, VADD(ar,br)); VS(USCR + 8, VADD(ai,bi));
          D1 = VFMA(vi, S1, D1); E1 = VFMA(vr, S1, E1);
          D2 = VFMA(vi, S2, D2); E2 = VFMA(vr, S2, E2);
          D3 = VFMA(vi, S3, D3); E3 = VFMA(vr, S3, E3);
          D4 = VFMA(vi, S4, D4); E4 = VFMA(vr, S4, E4);
          D5 = VFMA(vi, S5, D5); E5 = VFMA(vr, S5, E5);
          D6 = VFMA(vi, S6, D6); E6 = VFMA(vr, S6, E6);
        }
        { V ar = VL(pr + 16), ai = VL(pi + 16);
          V br = VL(pr + 88), bi = VL(pi + 88);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 16, VADD(ar,br)); VS(USCR + 24, VADD(ai,bi));
          D1 = VFMA(vi, S2, D1); E1 = VFMA(vr, S2, E1);
          D2 = VFMA(vi, S4, D2); E2 = VFMA(vr, S4, E2);
          D3 = VFMA(vi, S6, D3); E3 = VFMA(vr, S6, E3);
          D4 = VFNMA(vi, S5, D4); E4 = VFNMA(vr, S5, E4);
          D5 = VFNMA(vi, S3, D5); E5 = VFNMA(vr, S3, E5);
          D6 = VFNMA(vi, S1, D6); E6 = VFNMA(vr, S1, E6);
        }
        { V ar = VL(pr + 24), ai = VL(pi + 24);
          V br = VL(pr + 80), bi = VL(pi + 80);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 32, VADD(ar,br)); VS(USCR + 40, VADD(ai,bi));
          D1 = VFMA(vi, S3, D1); E1 = VFMA(vr, S3, E1);
          D2 = VFMA(vi, S6, D2); E2 = VFMA(vr, S6, E2);
          D3 = VFNMA(vi, S4, D3); E3 = VFNMA(vr, S4, E3);
          D4 = VFNMA(vi, S1, D4); E4 = VFNMA(vr, S1, E4);
          D5 = VFMA(vi, S2, D5); E5 = VFMA(vr, S2, E5);
          D6 = VFMA(vi, S5, D6); E6 = VFMA(vr, S5, E6);
        }
        { V ar = VL(pr + 32), ai = VL(pi + 32);
          V br = VL(pr + 72), bi = VL(pi + 72);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 48, VADD(ar,br)); VS(USCR + 56, VADD(ai,bi));
          D1 = VFMA(vi, S4, D1); E1 = VFMA(vr, S4, E1);
          D2 = VFNMA(vi, S5, D2); E2 = VFNMA(vr, S5, E2);
          D3 = VFNMA(vi, S1, D3); E3 = VFNMA(vr, S1, E3);
          D4 = VFMA(vi, S3, D4); E4 = VFMA(vr, S3, E4);
          D5 = VFNMA(vi, S6, D5); E5 = VFNMA(vr, S6, E5);
          D6 = VFNMA(vi, S2, D6); E6 = VFNMA(vr, S2, E6);
        }
        { V ar = VL(pr + 40), ai = VL(pi + 40);
          V br = VL(pr + 64), bi = VL(pi + 64);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 64, VADD(ar,br)); VS(USCR + 72, VADD(ai,bi));
          D1 = VFMA(vi, S5, D1); E1 = VFMA(vr, S5, E1);
          D2 = VFNMA(vi, S3, D2); E2 = VFNMA(vr, S3, E2);
          D3 = VFMA(vi, S2, D3); E3 = VFMA(vr, S2, E3);
          D4 = VFNMA(vi, S6, D4); E4 = VFNMA(vr, S6, E4);
          D5 = VFNMA(vi, S1, D5); E5 = VFNMA(vr, S1, E5);
          D6 = VFMA(vi, S4, D6); E6 = VFMA(vr, S4, E6);
        }
        { V ar = VL(pr + 48), ai = VL(pi + 48);
          V br = VL(pr + 56), bi = VL(pi + 56);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 80, VADD(ar,br)); VS(USCR + 88, VADD(ai,bi));
          D1 = VFMA(vi, S6, D1); E1 = VFMA(vr, S6, E1);
          D2 = VFNMA(vi, S1, D2); E2 = VFNMA(vr, S1, E2);
          D3 = VFMA(vi, S5, D3); E3 = VFMA(vr, S5, E3);
          D4 = VFNMA(vi, S2, D4); E4 = VFNMA(vr, S2, E4);
          D5 = VFMA(vi, S4, D5); E5 = VFMA(vr, S4, E5);
          D6 = VFNMA(vi, S3, D6); E6 = VFNMA(vr, S3, E6);
        }
        VS(PSCR + 0, D1); VS(PSCR + 8, E1);
        VS(PSCR + 16, D2); VS(PSCR + 24, E2);
        VS(PSCR + 32, D3); VS(PSCR + 40, E3);
        VS(PSCR + 48, D4); VS(PSCR + 56, E4);
        VS(PSCR + 64, D5); VS(PSCR + 72, E5);
        VS(PSCR + 80, D6); VS(PSCR + 88, E6);
        }
        VS(USCR + 384, VL(pr)); VS(USCR + 392, VL(pi));  // save x0 (row 0 gets overwritten)
        {
        V C1 = VSET1(0x1.c55a7e00740e9p-1);
        V C2 = VSET1(0x1.22d961ea71119p-1);
        V C3 = VSET1(0x1.edb7debaa3ed5p-4);
        V C4 = VSET1(-0x1.6b1d8b2365d9ep-2);
        V C5 = VSET1(-0x1.7f3ccd0032e0dp-1);
        V C6 = VSET1(-0x1.f11f493053d00p-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A1 = x0r, B1 = x0i;
        V A2 = x0r, B2 = x0i;
        V A3 = x0r, B3 = x0i;
        V A4 = x0r, B4 = x0i;
        V A5 = x0r, B5 = x0i;
        V A6 = x0r, B6 = x0i;
        V s0r = x0r, s0i = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A1 = VFMA(ur, C1, A1); B1 = VFMA(ui, C1, B1);
          A2 = VFMA(ur, C2, A2); B2 = VFMA(ui, C2, B2);
          A3 = VFMA(ur, C3, A3); B3 = VFMA(ui, C3, B3);
          A4 = VFMA(ur, C4, A4); B4 = VFMA(ui, C4, B4);
          A5 = VFMA(ur, C5, A5); B5 = VFMA(ui, C5, B5);
          A6 = VFMA(ur, C6, A6); B6 = VFMA(ui, C6, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A1 = VFMA(ur, C2, A1); B1 = VFMA(ui, C2, B1);
          A2 = VFMA(ur, C4, A2); B2 = VFMA(ui, C4, B2);
          A3 = VFMA(ur, C6, A3); B3 = VFMA(ui, C6, B3);
          A4 = VFMA(ur, C5, A4); B4 = VFMA(ui, C5, B4);
          A5 = VFMA(ur, C3, A5); B5 = VFMA(ui, C3, B5);
          A6 = VFMA(ur, C1, A6); B6 = VFMA(ui, C1, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A1 = VFMA(ur, C3, A1); B1 = VFMA(ui, C3, B1);
          A2 = VFMA(ur, C6, A2); B2 = VFMA(ui, C6, B2);
          A3 = VFMA(ur, C4, A3); B3 = VFMA(ui, C4, B3);
          A4 = VFMA(ur, C1, A4); B4 = VFMA(ui, C1, B4);
          A5 = VFMA(ur, C2, A5); B5 = VFMA(ui, C2, B5);
          A6 = VFMA(ur, C5, A6); B6 = VFMA(ui, C5, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A1 = VFMA(ur, C4, A1); B1 = VFMA(ui, C4, B1);
          A2 = VFMA(ur, C5, A2); B2 = VFMA(ui, C5, B2);
          A3 = VFMA(ur, C1, A3); B3 = VFMA(ui, C1, B3);
          A4 = VFMA(ur, C3, A4); B4 = VFMA(ui, C3, B4);
          A5 = VFMA(ur, C6, A5); B5 = VFMA(ui, C6, B5);
          A6 = VFMA(ur, C2, A6); B6 = VFMA(ui, C2, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A1 = VFMA(ur, C5, A1); B1 = VFMA(ui, C5, B1);
          A2 = VFMA(ur, C3, A2); B2 = VFMA(ui, C3, B2);
          A3 = VFMA(ur, C2, A3); B3 = VFMA(ui, C2, B3);
          A4 = VFMA(ur, C6, A4); B4 = VFMA(ui, C6, B4);
          A5 = VFMA(ur, C1, A5); B5 = VFMA(ui, C1, B5);
          A6 = VFMA(ur, C4, A6); B6 = VFMA(ui, C4, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A1 = VFMA(ur, C6, A1); B1 = VFMA(ui, C6, B1);
          A2 = VFMA(ur, C1, A2); B2 = VFMA(ui, C1, B2);
          A3 = VFMA(ur, C5, A3); B3 = VFMA(ui, C5, B3);
          A4 = VFMA(ur, C2, A4); B4 = VFMA(ui, C2, B4);
          A5 = VFMA(ur, C4, A5); B5 = VFMA(ui, C4, B5);
          A6 = VFMA(ur, C3, A6); B6 = VFMA(ui, C3, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        VS(pr + 0, s0r); VS(pi + 0, s0i);
        { V Dk = VL(PSCR + 0), Ek = VL(PSCR + 8);
          V Xr = VADD(A1, Dk), Xi = VSUB(B1, Ek);
          V Yr = VSUB(A1, Dk), Yi = VADD(B1, Ek);
        VS(pr + 8, Xr); VS(pi + 8, Xi);
        VS(pr + 96, Yr); VS(pi + 96, Yi);
        }
        { V Dk = VL(PSCR + 16), Ek = VL(PSCR + 24);
          V Xr = VADD(A2, Dk), Xi = VSUB(B2, Ek);
          V Yr = VSUB(A2, Dk), Yi = VADD(B2, Ek);
        VS(pr + 16, Xr); VS(pi + 16, Xi);
        VS(pr + 88, Yr); VS(pi + 88, Yi);
        }
        { V Dk = VL(PSCR + 32), Ek = VL(PSCR + 40);
          V Xr = VADD(A3, Dk), Xi = VSUB(B3, Ek);
          V Yr = VSUB(A3, Dk), Yi = VADD(B3, Ek);
        VS(pr + 24, Xr); VS(pi + 24, Xi);
        VS(pr + 80, Yr); VS(pi + 80, Yi);
        }
        { V Dk = VL(PSCR + 48), Ek = VL(PSCR + 56);
          V Xr = VADD(A4, Dk), Xi = VSUB(B4, Ek);
          V Yr = VSUB(A4, Dk), Yi = VADD(B4, Ek);
        VS(pr + 32, Xr); VS(pi + 32, Xi);
        VS(pr + 72, Yr); VS(pi + 72, Yi);
        }
        { V Dk = VL(PSCR + 64), Ek = VL(PSCR + 72);
          V Xr = VADD(A5, Dk), Xi = VSUB(B5, Ek);
          V Yr = VSUB(A5, Dk), Yi = VADD(B5, Ek);
        VS(pr + 40, Xr); VS(pi + 40, Xi);
        VS(pr + 64, Yr); VS(pi + 64, Yi);
        }
        { V Dk = VL(PSCR + 80), Ek = VL(PSCR + 88);
          V Xr = VADD(A6, Dk), Xi = VSUB(B6, Ek);
          V Yr = VSUB(A6, Dk), Yi = VADD(B6, Ek);
        VS(pr + 48, Xr); VS(pi + 48, Xi);
        VS(pr + 56, Yr); VS(pi + 56, Yi);
        }
        }
    }
}
static __attribute__((noinline)) void p13_yy(double* PR, double* PI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        {
        V S1 = VSET1(0x1.dbe064267c47bp-2);
        V S2 = VSET1(0x1.a55e242a4c3d2p-1);
        V S3 = VSET1(0x1.fc44566966769p-1);
        V S4 = VSET1(0x1.deba72ef20147p-1);
        V S5 = VSET1(0x1.5384d024c2f84p-1);
        V S6 = VSET1(0x1.ea1e54bc48dbcp-3);
        V D1 = _mm512_setzero_pd(), E1 = _mm512_setzero_pd();
        V D2 = _mm512_setzero_pd(), E2 = _mm512_setzero_pd();
        V D3 = _mm512_setzero_pd(), E3 = _mm512_setzero_pd();
        V D4 = _mm512_setzero_pd(), E4 = _mm512_setzero_pd();
        V D5 = _mm512_setzero_pd(), E5 = _mm512_setzero_pd();
        V D6 = _mm512_setzero_pd(), E6 = _mm512_setzero_pd();
        { V ar = VL(pr + 104), ai = VL(pi + 104);
          V br = VL(pr + 1248), bi = VL(pi + 1248);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 0, VADD(ar,br)); VS(USCR + 8, VADD(ai,bi));
          D1 = VFMA(vi, S1, D1); E1 = VFMA(vr, S1, E1);
          D2 = VFMA(vi, S2, D2); E2 = VFMA(vr, S2, E2);
          D3 = VFMA(vi, S3, D3); E3 = VFMA(vr, S3, E3);
          D4 = VFMA(vi, S4, D4); E4 = VFMA(vr, S4, E4);
          D5 = VFMA(vi, S5, D5); E5 = VFMA(vr, S5, E5);
          D6 = VFMA(vi, S6, D6); E6 = VFMA(vr, S6, E6);
        }
        { V ar = VL(pr + 208), ai = VL(pi + 208);
          V br = VL(pr + 1144), bi = VL(pi + 1144);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 16, VADD(ar,br)); VS(USCR + 24, VADD(ai,bi));
          D1 = VFMA(vi, S2, D1); E1 = VFMA(vr, S2, E1);
          D2 = VFMA(vi, S4, D2); E2 = VFMA(vr, S4, E2);
          D3 = VFMA(vi, S6, D3); E3 = VFMA(vr, S6, E3);
          D4 = VFNMA(vi, S5, D4); E4 = VFNMA(vr, S5, E4);
          D5 = VFNMA(vi, S3, D5); E5 = VFNMA(vr, S3, E5);
          D6 = VFNMA(vi, S1, D6); E6 = VFNMA(vr, S1, E6);
        }
        { V ar = VL(pr + 312), ai = VL(pi + 312);
          V br = VL(pr + 1040), bi = VL(pi + 1040);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 32, VADD(ar,br)); VS(USCR + 40, VADD(ai,bi));
          D1 = VFMA(vi, S3, D1); E1 = VFMA(vr, S3, E1);
          D2 = VFMA(vi, S6, D2); E2 = VFMA(vr, S6, E2);
          D3 = VFNMA(vi, S4, D3); E3 = VFNMA(vr, S4, E3);
          D4 = VFNMA(vi, S1, D4); E4 = VFNMA(vr, S1, E4);
          D5 = VFMA(vi, S2, D5); E5 = VFMA(vr, S2, E5);
          D6 = VFMA(vi, S5, D6); E6 = VFMA(vr, S5, E6);
        }
        { V ar = VL(pr + 416), ai = VL(pi + 416);
          V br = VL(pr + 936), bi = VL(pi + 936);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 48, VADD(ar,br)); VS(USCR + 56, VADD(ai,bi));
          D1 = VFMA(vi, S4, D1); E1 = VFMA(vr, S4, E1);
          D2 = VFNMA(vi, S5, D2); E2 = VFNMA(vr, S5, E2);
          D3 = VFNMA(vi, S1, D3); E3 = VFNMA(vr, S1, E3);
          D4 = VFMA(vi, S3, D4); E4 = VFMA(vr, S3, E4);
          D5 = VFNMA(vi, S6, D5); E5 = VFNMA(vr, S6, E5);
          D6 = VFNMA(vi, S2, D6); E6 = VFNMA(vr, S2, E6);
        }
        { V ar = VL(pr + 520), ai = VL(pi + 520);
          V br = VL(pr + 832), bi = VL(pi + 832);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 64, VADD(ar,br)); VS(USCR + 72, VADD(ai,bi));
          D1 = VFMA(vi, S5, D1); E1 = VFMA(vr, S5, E1);
          D2 = VFNMA(vi, S3, D2); E2 = VFNMA(vr, S3, E2);
          D3 = VFMA(vi, S2, D3); E3 = VFMA(vr, S2, E3);
          D4 = VFNMA(vi, S6, D4); E4 = VFNMA(vr, S6, E4);
          D5 = VFNMA(vi, S1, D5); E5 = VFNMA(vr, S1, E5);
          D6 = VFMA(vi, S4, D6); E6 = VFMA(vr, S4, E6);
        }
        { V ar = VL(pr + 624), ai = VL(pi + 624);
          V br = VL(pr + 728), bi = VL(pi + 728);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 80, VADD(ar,br)); VS(USCR + 88, VADD(ai,bi));
          D1 = VFMA(vi, S6, D1); E1 = VFMA(vr, S6, E1);
          D2 = VFNMA(vi, S1, D2); E2 = VFNMA(vr, S1, E2);
          D3 = VFMA(vi, S5, D3); E3 = VFMA(vr, S5, E3);
          D4 = VFNMA(vi, S2, D4); E4 = VFNMA(vr, S2, E4);
          D5 = VFMA(vi, S4, D5); E5 = VFMA(vr, S4, E5);
          D6 = VFNMA(vi, S3, D6); E6 = VFNMA(vr, S3, E6);
        }
        VS(PSCR + 0, D1); VS(PSCR + 8, E1);
        VS(PSCR + 16, D2); VS(PSCR + 24, E2);
        VS(PSCR + 32, D3); VS(PSCR + 40, E3);
        VS(PSCR + 48, D4); VS(PSCR + 56, E4);
        VS(PSCR + 64, D5); VS(PSCR + 72, E5);
        VS(PSCR + 80, D6); VS(PSCR + 88, E6);
        }
        VS(USCR + 384, VL(pr)); VS(USCR + 392, VL(pi));  // save x0 (row 0 gets overwritten)
        {
        V C1 = VSET1(0x1.c55a7e00740e9p-1);
        V C2 = VSET1(0x1.22d961ea71119p-1);
        V C3 = VSET1(0x1.edb7debaa3ed5p-4);
        V C4 = VSET1(-0x1.6b1d8b2365d9ep-2);
        V C5 = VSET1(-0x1.7f3ccd0032e0dp-1);
        V C6 = VSET1(-0x1.f11f493053d00p-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A1 = x0r, B1 = x0i;
        V A2 = x0r, B2 = x0i;
        V A3 = x0r, B3 = x0i;
        V A4 = x0r, B4 = x0i;
        V A5 = x0r, B5 = x0i;
        V A6 = x0r, B6 = x0i;
        V s0r = x0r, s0i = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A1 = VFMA(ur, C1, A1); B1 = VFMA(ui, C1, B1);
          A2 = VFMA(ur, C2, A2); B2 = VFMA(ui, C2, B2);
          A3 = VFMA(ur, C3, A3); B3 = VFMA(ui, C3, B3);
          A4 = VFMA(ur, C4, A4); B4 = VFMA(ui, C4, B4);
          A5 = VFMA(ur, C5, A5); B5 = VFMA(ui, C5, B5);
          A6 = VFMA(ur, C6, A6); B6 = VFMA(ui, C6, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A1 = VFMA(ur, C2, A1); B1 = VFMA(ui, C2, B1);
          A2 = VFMA(ur, C4, A2); B2 = VFMA(ui, C4, B2);
          A3 = VFMA(ur, C6, A3); B3 = VFMA(ui, C6, B3);
          A4 = VFMA(ur, C5, A4); B4 = VFMA(ui, C5, B4);
          A5 = VFMA(ur, C3, A5); B5 = VFMA(ui, C3, B5);
          A6 = VFMA(ur, C1, A6); B6 = VFMA(ui, C1, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A1 = VFMA(ur, C3, A1); B1 = VFMA(ui, C3, B1);
          A2 = VFMA(ur, C6, A2); B2 = VFMA(ui, C6, B2);
          A3 = VFMA(ur, C4, A3); B3 = VFMA(ui, C4, B3);
          A4 = VFMA(ur, C1, A4); B4 = VFMA(ui, C1, B4);
          A5 = VFMA(ur, C2, A5); B5 = VFMA(ui, C2, B5);
          A6 = VFMA(ur, C5, A6); B6 = VFMA(ui, C5, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A1 = VFMA(ur, C4, A1); B1 = VFMA(ui, C4, B1);
          A2 = VFMA(ur, C5, A2); B2 = VFMA(ui, C5, B2);
          A3 = VFMA(ur, C1, A3); B3 = VFMA(ui, C1, B3);
          A4 = VFMA(ur, C3, A4); B4 = VFMA(ui, C3, B4);
          A5 = VFMA(ur, C6, A5); B5 = VFMA(ui, C6, B5);
          A6 = VFMA(ur, C2, A6); B6 = VFMA(ui, C2, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A1 = VFMA(ur, C5, A1); B1 = VFMA(ui, C5, B1);
          A2 = VFMA(ur, C3, A2); B2 = VFMA(ui, C3, B2);
          A3 = VFMA(ur, C2, A3); B3 = VFMA(ui, C2, B3);
          A4 = VFMA(ur, C6, A4); B4 = VFMA(ui, C6, B4);
          A5 = VFMA(ur, C1, A5); B5 = VFMA(ui, C1, B5);
          A6 = VFMA(ur, C4, A6); B6 = VFMA(ui, C4, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A1 = VFMA(ur, C6, A1); B1 = VFMA(ui, C6, B1);
          A2 = VFMA(ur, C1, A2); B2 = VFMA(ui, C1, B2);
          A3 = VFMA(ur, C5, A3); B3 = VFMA(ui, C5, B3);
          A4 = VFMA(ur, C2, A4); B4 = VFMA(ui, C2, B4);
          A5 = VFMA(ur, C4, A5); B5 = VFMA(ui, C4, B5);
          A6 = VFMA(ur, C3, A6); B6 = VFMA(ui, C3, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        VS(pr + 0, s0r); VS(pi + 0, s0i);
        { V Dk = VL(PSCR + 0), Ek = VL(PSCR + 8);
          V Xr = VADD(A1, Dk), Xi = VSUB(B1, Ek);
          V Yr = VSUB(A1, Dk), Yi = VADD(B1, Ek);
        VS(pr + 104, Xr); VS(pi + 104, Xi);
        VS(pr + 1248, Yr); VS(pi + 1248, Yi);
        }
        { V Dk = VL(PSCR + 16), Ek = VL(PSCR + 24);
          V Xr = VADD(A2, Dk), Xi = VSUB(B2, Ek);
          V Yr = VSUB(A2, Dk), Yi = VADD(B2, Ek);
        VS(pr + 208, Xr); VS(pi + 208, Xi);
        VS(pr + 1144, Yr); VS(pi + 1144, Yi);
        }
        { V Dk = VL(PSCR + 32), Ek = VL(PSCR + 40);
          V Xr = VADD(A3, Dk), Xi = VSUB(B3, Ek);
          V Yr = VSUB(A3, Dk), Yi = VADD(B3, Ek);
        VS(pr + 312, Xr); VS(pi + 312, Xi);
        VS(pr + 1040, Yr); VS(pi + 1040, Yi);
        }
        { V Dk = VL(PSCR + 48), Ek = VL(PSCR + 56);
          V Xr = VADD(A4, Dk), Xi = VSUB(B4, Ek);
          V Yr = VSUB(A4, Dk), Yi = VADD(B4, Ek);
        VS(pr + 416, Xr); VS(pi + 416, Xi);
        VS(pr + 936, Yr); VS(pi + 936, Yi);
        }
        { V Dk = VL(PSCR + 64), Ek = VL(PSCR + 72);
          V Xr = VADD(A5, Dk), Xi = VSUB(B5, Ek);
          V Yr = VSUB(A5, Dk), Yi = VADD(B5, Ek);
        VS(pr + 520, Xr); VS(pi + 520, Xi);
        VS(pr + 832, Yr); VS(pi + 832, Yi);
        }
        { V Dk = VL(PSCR + 80), Ek = VL(PSCR + 88);
          V Xr = VADD(A6, Dk), Xi = VSUB(B6, Ek);
          V Yr = VSUB(A6, Dk), Yi = VADD(B6, Ek);
        VS(pr + 624, Xr); VS(pi + 624, Xi);
        VS(pr + 728, Yr); VS(pi + 728, Yi);
        }
        }
    }
}
static __attribute__((noinline)) void p13_xx(double* PR, double* PI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        {
        V S1 = VSET1(0x1.dbe064267c47bp-2);
        V S2 = VSET1(0x1.a55e242a4c3d2p-1);
        V S3 = VSET1(0x1.fc44566966769p-1);
        V S4 = VSET1(0x1.deba72ef20147p-1);
        V S5 = VSET1(0x1.5384d024c2f84p-1);
        V S6 = VSET1(0x1.ea1e54bc48dbcp-3);
        V D1 = _mm512_setzero_pd(), E1 = _mm512_setzero_pd();
        V D2 = _mm512_setzero_pd(), E2 = _mm512_setzero_pd();
        V D3 = _mm512_setzero_pd(), E3 = _mm512_setzero_pd();
        V D4 = _mm512_setzero_pd(), E4 = _mm512_setzero_pd();
        V D5 = _mm512_setzero_pd(), E5 = _mm512_setzero_pd();
        V D6 = _mm512_setzero_pd(), E6 = _mm512_setzero_pd();
        { V ar = VL(pr + 1352), ai = VL(pi + 1352);
          V br = VL(pr + 16224), bi = VL(pi + 16224);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 0, VADD(ar,br)); VS(USCR + 8, VADD(ai,bi));
          D1 = VFMA(vi, S1, D1); E1 = VFMA(vr, S1, E1);
          D2 = VFMA(vi, S2, D2); E2 = VFMA(vr, S2, E2);
          D3 = VFMA(vi, S3, D3); E3 = VFMA(vr, S3, E3);
          D4 = VFMA(vi, S4, D4); E4 = VFMA(vr, S4, E4);
          D5 = VFMA(vi, S5, D5); E5 = VFMA(vr, S5, E5);
          D6 = VFMA(vi, S6, D6); E6 = VFMA(vr, S6, E6);
        }
        { V ar = VL(pr + 2704), ai = VL(pi + 2704);
          V br = VL(pr + 14872), bi = VL(pi + 14872);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 16, VADD(ar,br)); VS(USCR + 24, VADD(ai,bi));
          D1 = VFMA(vi, S2, D1); E1 = VFMA(vr, S2, E1);
          D2 = VFMA(vi, S4, D2); E2 = VFMA(vr, S4, E2);
          D3 = VFMA(vi, S6, D3); E3 = VFMA(vr, S6, E3);
          D4 = VFNMA(vi, S5, D4); E4 = VFNMA(vr, S5, E4);
          D5 = VFNMA(vi, S3, D5); E5 = VFNMA(vr, S3, E5);
          D6 = VFNMA(vi, S1, D6); E6 = VFNMA(vr, S1, E6);
        }
        { V ar = VL(pr + 4056), ai = VL(pi + 4056);
          V br = VL(pr + 13520), bi = VL(pi + 13520);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 32, VADD(ar,br)); VS(USCR + 40, VADD(ai,bi));
          D1 = VFMA(vi, S3, D1); E1 = VFMA(vr, S3, E1);
          D2 = VFMA(vi, S6, D2); E2 = VFMA(vr, S6, E2);
          D3 = VFNMA(vi, S4, D3); E3 = VFNMA(vr, S4, E3);
          D4 = VFNMA(vi, S1, D4); E4 = VFNMA(vr, S1, E4);
          D5 = VFMA(vi, S2, D5); E5 = VFMA(vr, S2, E5);
          D6 = VFMA(vi, S5, D6); E6 = VFMA(vr, S5, E6);
        }
        { V ar = VL(pr + 5408), ai = VL(pi + 5408);
          V br = VL(pr + 12168), bi = VL(pi + 12168);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 48, VADD(ar,br)); VS(USCR + 56, VADD(ai,bi));
          D1 = VFMA(vi, S4, D1); E1 = VFMA(vr, S4, E1);
          D2 = VFNMA(vi, S5, D2); E2 = VFNMA(vr, S5, E2);
          D3 = VFNMA(vi, S1, D3); E3 = VFNMA(vr, S1, E3);
          D4 = VFMA(vi, S3, D4); E4 = VFMA(vr, S3, E4);
          D5 = VFNMA(vi, S6, D5); E5 = VFNMA(vr, S6, E5);
          D6 = VFNMA(vi, S2, D6); E6 = VFNMA(vr, S2, E6);
        }
        { V ar = VL(pr + 6760), ai = VL(pi + 6760);
          V br = VL(pr + 10816), bi = VL(pi + 10816);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 64, VADD(ar,br)); VS(USCR + 72, VADD(ai,bi));
          D1 = VFMA(vi, S5, D1); E1 = VFMA(vr, S5, E1);
          D2 = VFNMA(vi, S3, D2); E2 = VFNMA(vr, S3, E2);
          D3 = VFMA(vi, S2, D3); E3 = VFMA(vr, S2, E3);
          D4 = VFNMA(vi, S6, D4); E4 = VFNMA(vr, S6, E4);
          D5 = VFNMA(vi, S1, D5); E5 = VFNMA(vr, S1, E5);
          D6 = VFMA(vi, S4, D6); E6 = VFMA(vr, S4, E6);
        }
        { V ar = VL(pr + 8112), ai = VL(pi + 8112);
          V br = VL(pr + 9464), bi = VL(pi + 9464);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 80, VADD(ar,br)); VS(USCR + 88, VADD(ai,bi));
          D1 = VFMA(vi, S6, D1); E1 = VFMA(vr, S6, E1);
          D2 = VFNMA(vi, S1, D2); E2 = VFNMA(vr, S1, E2);
          D3 = VFMA(vi, S5, D3); E3 = VFMA(vr, S5, E3);
          D4 = VFNMA(vi, S2, D4); E4 = VFNMA(vr, S2, E4);
          D5 = VFMA(vi, S4, D5); E5 = VFMA(vr, S4, E5);
          D6 = VFNMA(vi, S3, D6); E6 = VFNMA(vr, S3, E6);
        }
        VS(PSCR + 0, D1); VS(PSCR + 8, E1);
        VS(PSCR + 16, D2); VS(PSCR + 24, E2);
        VS(PSCR + 32, D3); VS(PSCR + 40, E3);
        VS(PSCR + 48, D4); VS(PSCR + 56, E4);
        VS(PSCR + 64, D5); VS(PSCR + 72, E5);
        VS(PSCR + 80, D6); VS(PSCR + 88, E6);
        }
        VS(USCR + 384, VL(pr)); VS(USCR + 392, VL(pi));  // save x0 (row 0 gets overwritten)
        {
        V C1 = VSET1(0x1.c55a7e00740e9p-1);
        V C2 = VSET1(0x1.22d961ea71119p-1);
        V C3 = VSET1(0x1.edb7debaa3ed5p-4);
        V C4 = VSET1(-0x1.6b1d8b2365d9ep-2);
        V C5 = VSET1(-0x1.7f3ccd0032e0dp-1);
        V C6 = VSET1(-0x1.f11f493053d00p-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A1 = x0r, B1 = x0i;
        V A2 = x0r, B2 = x0i;
        V A3 = x0r, B3 = x0i;
        V A4 = x0r, B4 = x0i;
        V A5 = x0r, B5 = x0i;
        V A6 = x0r, B6 = x0i;
        V s0r = x0r, s0i = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A1 = VFMA(ur, C1, A1); B1 = VFMA(ui, C1, B1);
          A2 = VFMA(ur, C2, A2); B2 = VFMA(ui, C2, B2);
          A3 = VFMA(ur, C3, A3); B3 = VFMA(ui, C3, B3);
          A4 = VFMA(ur, C4, A4); B4 = VFMA(ui, C4, B4);
          A5 = VFMA(ur, C5, A5); B5 = VFMA(ui, C5, B5);
          A6 = VFMA(ur, C6, A6); B6 = VFMA(ui, C6, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A1 = VFMA(ur, C2, A1); B1 = VFMA(ui, C2, B1);
          A2 = VFMA(ur, C4, A2); B2 = VFMA(ui, C4, B2);
          A3 = VFMA(ur, C6, A3); B3 = VFMA(ui, C6, B3);
          A4 = VFMA(ur, C5, A4); B4 = VFMA(ui, C5, B4);
          A5 = VFMA(ur, C3, A5); B5 = VFMA(ui, C3, B5);
          A6 = VFMA(ur, C1, A6); B6 = VFMA(ui, C1, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A1 = VFMA(ur, C3, A1); B1 = VFMA(ui, C3, B1);
          A2 = VFMA(ur, C6, A2); B2 = VFMA(ui, C6, B2);
          A3 = VFMA(ur, C4, A3); B3 = VFMA(ui, C4, B3);
          A4 = VFMA(ur, C1, A4); B4 = VFMA(ui, C1, B4);
          A5 = VFMA(ur, C2, A5); B5 = VFMA(ui, C2, B5);
          A6 = VFMA(ur, C5, A6); B6 = VFMA(ui, C5, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A1 = VFMA(ur, C4, A1); B1 = VFMA(ui, C4, B1);
          A2 = VFMA(ur, C5, A2); B2 = VFMA(ui, C5, B2);
          A3 = VFMA(ur, C1, A3); B3 = VFMA(ui, C1, B3);
          A4 = VFMA(ur, C3, A4); B4 = VFMA(ui, C3, B4);
          A5 = VFMA(ur, C6, A5); B5 = VFMA(ui, C6, B5);
          A6 = VFMA(ur, C2, A6); B6 = VFMA(ui, C2, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A1 = VFMA(ur, C5, A1); B1 = VFMA(ui, C5, B1);
          A2 = VFMA(ur, C3, A2); B2 = VFMA(ui, C3, B2);
          A3 = VFMA(ur, C2, A3); B3 = VFMA(ui, C2, B3);
          A4 = VFMA(ur, C6, A4); B4 = VFMA(ui, C6, B4);
          A5 = VFMA(ur, C1, A5); B5 = VFMA(ui, C1, B5);
          A6 = VFMA(ur, C4, A6); B6 = VFMA(ui, C4, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A1 = VFMA(ur, C6, A1); B1 = VFMA(ui, C6, B1);
          A2 = VFMA(ur, C1, A2); B2 = VFMA(ui, C1, B2);
          A3 = VFMA(ur, C5, A3); B3 = VFMA(ui, C5, B3);
          A4 = VFMA(ur, C2, A4); B4 = VFMA(ui, C2, B4);
          A5 = VFMA(ur, C4, A5); B5 = VFMA(ui, C4, B5);
          A6 = VFMA(ur, C3, A6); B6 = VFMA(ui, C3, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        VS(pr + 0, s0r); VS(pi + 0, s0i);
        { V Dk = VL(PSCR + 0), Ek = VL(PSCR + 8);
          V Xr = VADD(A1, Dk), Xi = VSUB(B1, Ek);
          V Yr = VSUB(A1, Dk), Yi = VADD(B1, Ek);
        VS(pr + 1352, Xr); VS(pi + 1352, Xi);
        VS(pr + 16224, Yr); VS(pi + 16224, Yi);
        }
        { V Dk = VL(PSCR + 16), Ek = VL(PSCR + 24);
          V Xr = VADD(A2, Dk), Xi = VSUB(B2, Ek);
          V Yr = VSUB(A2, Dk), Yi = VADD(B2, Ek);
        VS(pr + 2704, Xr); VS(pi + 2704, Xi);
        VS(pr + 14872, Yr); VS(pi + 14872, Yi);
        }
        { V Dk = VL(PSCR + 32), Ek = VL(PSCR + 40);
          V Xr = VADD(A3, Dk), Xi = VSUB(B3, Ek);
          V Yr = VSUB(A3, Dk), Yi = VADD(B3, Ek);
        VS(pr + 4056, Xr); VS(pi + 4056, Xi);
        VS(pr + 13520, Yr); VS(pi + 13520, Yi);
        }
        { V Dk = VL(PSCR + 48), Ek = VL(PSCR + 56);
          V Xr = VADD(A4, Dk), Xi = VSUB(B4, Ek);
          V Yr = VSUB(A4, Dk), Yi = VADD(B4, Ek);
        VS(pr + 5408, Xr); VS(pi + 5408, Xi);
        VS(pr + 12168, Yr); VS(pi + 12168, Yi);
        }
        { V Dk = VL(PSCR + 64), Ek = VL(PSCR + 72);
          V Xr = VADD(A5, Dk), Xi = VSUB(B5, Ek);
          V Yr = VSUB(A5, Dk), Yi = VADD(B5, Ek);
        VS(pr + 6760, Xr); VS(pi + 6760, Xi);
        VS(pr + 10816, Yr); VS(pi + 10816, Yi);
        }
        { V Dk = VL(PSCR + 80), Ek = VL(PSCR + 88);
          V Xr = VADD(A6, Dk), Xi = VSUB(B6, Ek);
          V Yr = VSUB(A6, Dk), Yi = VADD(B6, Ek);
        VS(pr + 8112, Xr); VS(pi + 8112, Xi);
        VS(pr + 9464, Yr); VS(pi + 9464, Yi);
        }
        }
    }
}
static __attribute__((noinline)) void p13_xxm(double* PR, double* PI, const double* CR, const double* CI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        const double* cr = CR + q_*pstep; const double* ci = CI + q_*pstep;
        {
        V S1 = VSET1(0x1.dbe064267c47bp-2);
        V S2 = VSET1(0x1.a55e242a4c3d2p-1);
        V S3 = VSET1(0x1.fc44566966769p-1);
        V S4 = VSET1(0x1.deba72ef20147p-1);
        V S5 = VSET1(0x1.5384d024c2f84p-1);
        V S6 = VSET1(0x1.ea1e54bc48dbcp-3);
        V D1 = _mm512_setzero_pd(), E1 = _mm512_setzero_pd();
        V D2 = _mm512_setzero_pd(), E2 = _mm512_setzero_pd();
        V D3 = _mm512_setzero_pd(), E3 = _mm512_setzero_pd();
        V D4 = _mm512_setzero_pd(), E4 = _mm512_setzero_pd();
        V D5 = _mm512_setzero_pd(), E5 = _mm512_setzero_pd();
        V D6 = _mm512_setzero_pd(), E6 = _mm512_setzero_pd();
        { V ar = VL(pr + 1352), ai = VL(pi + 1352);
          V br = VL(pr + 16224), bi = VL(pi + 16224);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 0, VADD(ar,br)); VS(USCR + 8, VADD(ai,bi));
          D1 = VFMA(vi, S1, D1); E1 = VFMA(vr, S1, E1);
          D2 = VFMA(vi, S2, D2); E2 = VFMA(vr, S2, E2);
          D3 = VFMA(vi, S3, D3); E3 = VFMA(vr, S3, E3);
          D4 = VFMA(vi, S4, D4); E4 = VFMA(vr, S4, E4);
          D5 = VFMA(vi, S5, D5); E5 = VFMA(vr, S5, E5);
          D6 = VFMA(vi, S6, D6); E6 = VFMA(vr, S6, E6);
        }
        { V ar = VL(pr + 2704), ai = VL(pi + 2704);
          V br = VL(pr + 14872), bi = VL(pi + 14872);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 16, VADD(ar,br)); VS(USCR + 24, VADD(ai,bi));
          D1 = VFMA(vi, S2, D1); E1 = VFMA(vr, S2, E1);
          D2 = VFMA(vi, S4, D2); E2 = VFMA(vr, S4, E2);
          D3 = VFMA(vi, S6, D3); E3 = VFMA(vr, S6, E3);
          D4 = VFNMA(vi, S5, D4); E4 = VFNMA(vr, S5, E4);
          D5 = VFNMA(vi, S3, D5); E5 = VFNMA(vr, S3, E5);
          D6 = VFNMA(vi, S1, D6); E6 = VFNMA(vr, S1, E6);
        }
        { V ar = VL(pr + 4056), ai = VL(pi + 4056);
          V br = VL(pr + 13520), bi = VL(pi + 13520);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 32, VADD(ar,br)); VS(USCR + 40, VADD(ai,bi));
          D1 = VFMA(vi, S3, D1); E1 = VFMA(vr, S3, E1);
          D2 = VFMA(vi, S6, D2); E2 = VFMA(vr, S6, E2);
          D3 = VFNMA(vi, S4, D3); E3 = VFNMA(vr, S4, E3);
          D4 = VFNMA(vi, S1, D4); E4 = VFNMA(vr, S1, E4);
          D5 = VFMA(vi, S2, D5); E5 = VFMA(vr, S2, E5);
          D6 = VFMA(vi, S5, D6); E6 = VFMA(vr, S5, E6);
        }
        { V ar = VL(pr + 5408), ai = VL(pi + 5408);
          V br = VL(pr + 12168), bi = VL(pi + 12168);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 48, VADD(ar,br)); VS(USCR + 56, VADD(ai,bi));
          D1 = VFMA(vi, S4, D1); E1 = VFMA(vr, S4, E1);
          D2 = VFNMA(vi, S5, D2); E2 = VFNMA(vr, S5, E2);
          D3 = VFNMA(vi, S1, D3); E3 = VFNMA(vr, S1, E3);
          D4 = VFMA(vi, S3, D4); E4 = VFMA(vr, S3, E4);
          D5 = VFNMA(vi, S6, D5); E5 = VFNMA(vr, S6, E5);
          D6 = VFNMA(vi, S2, D6); E6 = VFNMA(vr, S2, E6);
        }
        { V ar = VL(pr + 6760), ai = VL(pi + 6760);
          V br = VL(pr + 10816), bi = VL(pi + 10816);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 64, VADD(ar,br)); VS(USCR + 72, VADD(ai,bi));
          D1 = VFMA(vi, S5, D1); E1 = VFMA(vr, S5, E1);
          D2 = VFNMA(vi, S3, D2); E2 = VFNMA(vr, S3, E2);
          D3 = VFMA(vi, S2, D3); E3 = VFMA(vr, S2, E3);
          D4 = VFNMA(vi, S6, D4); E4 = VFNMA(vr, S6, E4);
          D5 = VFNMA(vi, S1, D5); E5 = VFNMA(vr, S1, E5);
          D6 = VFMA(vi, S4, D6); E6 = VFMA(vr, S4, E6);
        }
        { V ar = VL(pr + 8112), ai = VL(pi + 8112);
          V br = VL(pr + 9464), bi = VL(pi + 9464);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 80, VADD(ar,br)); VS(USCR + 88, VADD(ai,bi));
          D1 = VFMA(vi, S6, D1); E1 = VFMA(vr, S6, E1);
          D2 = VFNMA(vi, S1, D2); E2 = VFNMA(vr, S1, E2);
          D3 = VFMA(vi, S5, D3); E3 = VFMA(vr, S5, E3);
          D4 = VFNMA(vi, S2, D4); E4 = VFNMA(vr, S2, E4);
          D5 = VFMA(vi, S4, D5); E5 = VFMA(vr, S4, E5);
          D6 = VFNMA(vi, S3, D6); E6 = VFNMA(vr, S3, E6);
        }
        VS(PSCR + 0, D1); VS(PSCR + 8, E1);
        VS(PSCR + 16, D2); VS(PSCR + 24, E2);
        VS(PSCR + 32, D3); VS(PSCR + 40, E3);
        VS(PSCR + 48, D4); VS(PSCR + 56, E4);
        VS(PSCR + 64, D5); VS(PSCR + 72, E5);
        VS(PSCR + 80, D6); VS(PSCR + 88, E6);
        }
        VS(USCR + 384, VL(pr)); VS(USCR + 392, VL(pi));  // save x0 (row 0 gets overwritten)
        {
        V C1 = VSET1(0x1.c55a7e00740e9p-1);
        V C2 = VSET1(0x1.22d961ea71119p-1);
        V C3 = VSET1(0x1.edb7debaa3ed5p-4);
        V C4 = VSET1(-0x1.6b1d8b2365d9ep-2);
        V C5 = VSET1(-0x1.7f3ccd0032e0dp-1);
        V C6 = VSET1(-0x1.f11f493053d00p-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A1 = x0r, B1 = x0i;
        V A2 = x0r, B2 = x0i;
        V A3 = x0r, B3 = x0i;
        V A4 = x0r, B4 = x0i;
        V A5 = x0r, B5 = x0i;
        V A6 = x0r, B6 = x0i;
        V s0r = x0r, s0i = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A1 = VFMA(ur, C1, A1); B1 = VFMA(ui, C1, B1);
          A2 = VFMA(ur, C2, A2); B2 = VFMA(ui, C2, B2);
          A3 = VFMA(ur, C3, A3); B3 = VFMA(ui, C3, B3);
          A4 = VFMA(ur, C4, A4); B4 = VFMA(ui, C4, B4);
          A5 = VFMA(ur, C5, A5); B5 = VFMA(ui, C5, B5);
          A6 = VFMA(ur, C6, A6); B6 = VFMA(ui, C6, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A1 = VFMA(ur, C2, A1); B1 = VFMA(ui, C2, B1);
          A2 = VFMA(ur, C4, A2); B2 = VFMA(ui, C4, B2);
          A3 = VFMA(ur, C6, A3); B3 = VFMA(ui, C6, B3);
          A4 = VFMA(ur, C5, A4); B4 = VFMA(ui, C5, B4);
          A5 = VFMA(ur, C3, A5); B5 = VFMA(ui, C3, B5);
          A6 = VFMA(ur, C1, A6); B6 = VFMA(ui, C1, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A1 = VFMA(ur, C3, A1); B1 = VFMA(ui, C3, B1);
          A2 = VFMA(ur, C6, A2); B2 = VFMA(ui, C6, B2);
          A3 = VFMA(ur, C4, A3); B3 = VFMA(ui, C4, B3);
          A4 = VFMA(ur, C1, A4); B4 = VFMA(ui, C1, B4);
          A5 = VFMA(ur, C2, A5); B5 = VFMA(ui, C2, B5);
          A6 = VFMA(ur, C5, A6); B6 = VFMA(ui, C5, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A1 = VFMA(ur, C4, A1); B1 = VFMA(ui, C4, B1);
          A2 = VFMA(ur, C5, A2); B2 = VFMA(ui, C5, B2);
          A3 = VFMA(ur, C1, A3); B3 = VFMA(ui, C1, B3);
          A4 = VFMA(ur, C3, A4); B4 = VFMA(ui, C3, B4);
          A5 = VFMA(ur, C6, A5); B5 = VFMA(ui, C6, B5);
          A6 = VFMA(ur, C2, A6); B6 = VFMA(ui, C2, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A1 = VFMA(ur, C5, A1); B1 = VFMA(ui, C5, B1);
          A2 = VFMA(ur, C3, A2); B2 = VFMA(ui, C3, B2);
          A3 = VFMA(ur, C2, A3); B3 = VFMA(ui, C2, B3);
          A4 = VFMA(ur, C6, A4); B4 = VFMA(ui, C6, B4);
          A5 = VFMA(ur, C1, A5); B5 = VFMA(ui, C1, B5);
          A6 = VFMA(ur, C4, A6); B6 = VFMA(ui, C4, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A1 = VFMA(ur, C6, A1); B1 = VFMA(ui, C6, B1);
          A2 = VFMA(ur, C1, A2); B2 = VFMA(ui, C1, B2);
          A3 = VFMA(ur, C5, A3); B3 = VFMA(ui, C5, B3);
          A4 = VFMA(ur, C2, A4); B4 = VFMA(ui, C2, B4);
          A5 = VFMA(ur, C4, A5); B5 = VFMA(ui, C4, B5);
          A6 = VFMA(ur, C3, A6); B6 = VFMA(ui, C3, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V zr_ = VADD(s0r, VL(cr + 0)), zi_ = VADD(s0i, VL(ci + 0));
          MAP2(zr_, zi_);
          VS(pr + 0, zr_); VS(pi + 0, zi_); }
        { V Dk = VL(PSCR + 0), Ek = VL(PSCR + 8);
          V Xr = VADD(A1, Dk), Xi = VSUB(B1, Ek);
          V Yr = VSUB(A1, Dk), Yi = VADD(B1, Ek);
        { V zr_ = VADD(Xr, VL(cr + 1352)), zi_ = VADD(Xi, VL(ci + 1352));
          MAP2(zr_, zi_);
          VS(pr + 1352, zr_); VS(pi + 1352, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 16224)), zi_ = VADD(Yi, VL(ci + 16224));
          MAP2(zr_, zi_);
          VS(pr + 16224, zr_); VS(pi + 16224, zi_); }
        }
        { V Dk = VL(PSCR + 16), Ek = VL(PSCR + 24);
          V Xr = VADD(A2, Dk), Xi = VSUB(B2, Ek);
          V Yr = VSUB(A2, Dk), Yi = VADD(B2, Ek);
        { V zr_ = VADD(Xr, VL(cr + 2704)), zi_ = VADD(Xi, VL(ci + 2704));
          MAP2(zr_, zi_);
          VS(pr + 2704, zr_); VS(pi + 2704, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 14872)), zi_ = VADD(Yi, VL(ci + 14872));
          MAP2(zr_, zi_);
          VS(pr + 14872, zr_); VS(pi + 14872, zi_); }
        }
        { V Dk = VL(PSCR + 32), Ek = VL(PSCR + 40);
          V Xr = VADD(A3, Dk), Xi = VSUB(B3, Ek);
          V Yr = VSUB(A3, Dk), Yi = VADD(B3, Ek);
        { V zr_ = VADD(Xr, VL(cr + 4056)), zi_ = VADD(Xi, VL(ci + 4056));
          MAP2(zr_, zi_);
          VS(pr + 4056, zr_); VS(pi + 4056, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 13520)), zi_ = VADD(Yi, VL(ci + 13520));
          MAP2(zr_, zi_);
          VS(pr + 13520, zr_); VS(pi + 13520, zi_); }
        }
        { V Dk = VL(PSCR + 48), Ek = VL(PSCR + 56);
          V Xr = VADD(A4, Dk), Xi = VSUB(B4, Ek);
          V Yr = VSUB(A4, Dk), Yi = VADD(B4, Ek);
        { V zr_ = VADD(Xr, VL(cr + 5408)), zi_ = VADD(Xi, VL(ci + 5408));
          MAP2(zr_, zi_);
          VS(pr + 5408, zr_); VS(pi + 5408, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 12168)), zi_ = VADD(Yi, VL(ci + 12168));
          MAP2(zr_, zi_);
          VS(pr + 12168, zr_); VS(pi + 12168, zi_); }
        }
        { V Dk = VL(PSCR + 64), Ek = VL(PSCR + 72);
          V Xr = VADD(A5, Dk), Xi = VSUB(B5, Ek);
          V Yr = VSUB(A5, Dk), Yi = VADD(B5, Ek);
        { V zr_ = VADD(Xr, VL(cr + 6760)), zi_ = VADD(Xi, VL(ci + 6760));
          MAP2(zr_, zi_);
          VS(pr + 6760, zr_); VS(pi + 6760, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 10816)), zi_ = VADD(Yi, VL(ci + 10816));
          MAP2(zr_, zi_);
          VS(pr + 10816, zr_); VS(pi + 10816, zi_); }
        }
        { V Dk = VL(PSCR + 80), Ek = VL(PSCR + 88);
          V Xr = VADD(A6, Dk), Xi = VSUB(B6, Ek);
          V Yr = VSUB(A6, Dk), Yi = VADD(B6, Ek);
        { V zr_ = VADD(Xr, VL(cr + 8112)), zi_ = VADD(Xi, VL(ci + 8112));
          MAP2(zr_, zi_);
          VS(pr + 8112, zr_); VS(pi + 8112, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 9464)), zi_ = VADD(Yi, VL(ci + 9464));
          MAP2(zr_, zi_);
          VS(pr + 9464, zr_); VS(pi + 9464, zi_); }
        }
        }
    }
}
static __attribute__((noinline)) void p13_yym(double* PR, double* PI, const double* CR, const double* CI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        const double* cr = CR + q_*pstep; const double* ci = CI + q_*pstep;
        {
        V S1 = VSET1(0x1.dbe064267c47bp-2);
        V S2 = VSET1(0x1.a55e242a4c3d2p-1);
        V S3 = VSET1(0x1.fc44566966769p-1);
        V S4 = VSET1(0x1.deba72ef20147p-1);
        V S5 = VSET1(0x1.5384d024c2f84p-1);
        V S6 = VSET1(0x1.ea1e54bc48dbcp-3);
        V D1 = _mm512_setzero_pd(), E1 = _mm512_setzero_pd();
        V D2 = _mm512_setzero_pd(), E2 = _mm512_setzero_pd();
        V D3 = _mm512_setzero_pd(), E3 = _mm512_setzero_pd();
        V D4 = _mm512_setzero_pd(), E4 = _mm512_setzero_pd();
        V D5 = _mm512_setzero_pd(), E5 = _mm512_setzero_pd();
        V D6 = _mm512_setzero_pd(), E6 = _mm512_setzero_pd();
        { V ar = VL(pr + 104), ai = VL(pi + 104);
          V br = VL(pr + 1248), bi = VL(pi + 1248);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 0, VADD(ar,br)); VS(USCR + 8, VADD(ai,bi));
          D1 = VFMA(vi, S1, D1); E1 = VFMA(vr, S1, E1);
          D2 = VFMA(vi, S2, D2); E2 = VFMA(vr, S2, E2);
          D3 = VFMA(vi, S3, D3); E3 = VFMA(vr, S3, E3);
          D4 = VFMA(vi, S4, D4); E4 = VFMA(vr, S4, E4);
          D5 = VFMA(vi, S5, D5); E5 = VFMA(vr, S5, E5);
          D6 = VFMA(vi, S6, D6); E6 = VFMA(vr, S6, E6);
        }
        { V ar = VL(pr + 208), ai = VL(pi + 208);
          V br = VL(pr + 1144), bi = VL(pi + 1144);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 16, VADD(ar,br)); VS(USCR + 24, VADD(ai,bi));
          D1 = VFMA(vi, S2, D1); E1 = VFMA(vr, S2, E1);
          D2 = VFMA(vi, S4, D2); E2 = VFMA(vr, S4, E2);
          D3 = VFMA(vi, S6, D3); E3 = VFMA(vr, S6, E3);
          D4 = VFNMA(vi, S5, D4); E4 = VFNMA(vr, S5, E4);
          D5 = VFNMA(vi, S3, D5); E5 = VFNMA(vr, S3, E5);
          D6 = VFNMA(vi, S1, D6); E6 = VFNMA(vr, S1, E6);
        }
        { V ar = VL(pr + 312), ai = VL(pi + 312);
          V br = VL(pr + 1040), bi = VL(pi + 1040);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 32, VADD(ar,br)); VS(USCR + 40, VADD(ai,bi));
          D1 = VFMA(vi, S3, D1); E1 = VFMA(vr, S3, E1);
          D2 = VFMA(vi, S6, D2); E2 = VFMA(vr, S6, E2);
          D3 = VFNMA(vi, S4, D3); E3 = VFNMA(vr, S4, E3);
          D4 = VFNMA(vi, S1, D4); E4 = VFNMA(vr, S1, E4);
          D5 = VFMA(vi, S2, D5); E5 = VFMA(vr, S2, E5);
          D6 = VFMA(vi, S5, D6); E6 = VFMA(vr, S5, E6);
        }
        { V ar = VL(pr + 416), ai = VL(pi + 416);
          V br = VL(pr + 936), bi = VL(pi + 936);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 48, VADD(ar,br)); VS(USCR + 56, VADD(ai,bi));
          D1 = VFMA(vi, S4, D1); E1 = VFMA(vr, S4, E1);
          D2 = VFNMA(vi, S5, D2); E2 = VFNMA(vr, S5, E2);
          D3 = VFNMA(vi, S1, D3); E3 = VFNMA(vr, S1, E3);
          D4 = VFMA(vi, S3, D4); E4 = VFMA(vr, S3, E4);
          D5 = VFNMA(vi, S6, D5); E5 = VFNMA(vr, S6, E5);
          D6 = VFNMA(vi, S2, D6); E6 = VFNMA(vr, S2, E6);
        }
        { V ar = VL(pr + 520), ai = VL(pi + 520);
          V br = VL(pr + 832), bi = VL(pi + 832);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 64, VADD(ar,br)); VS(USCR + 72, VADD(ai,bi));
          D1 = VFMA(vi, S5, D1); E1 = VFMA(vr, S5, E1);
          D2 = VFNMA(vi, S3, D2); E2 = VFNMA(vr, S3, E2);
          D3 = VFMA(vi, S2, D3); E3 = VFMA(vr, S2, E3);
          D4 = VFNMA(vi, S6, D4); E4 = VFNMA(vr, S6, E4);
          D5 = VFNMA(vi, S1, D5); E5 = VFNMA(vr, S1, E5);
          D6 = VFMA(vi, S4, D6); E6 = VFMA(vr, S4, E6);
        }
        { V ar = VL(pr + 624), ai = VL(pi + 624);
          V br = VL(pr + 728), bi = VL(pi + 728);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 80, VADD(ar,br)); VS(USCR + 88, VADD(ai,bi));
          D1 = VFMA(vi, S6, D1); E1 = VFMA(vr, S6, E1);
          D2 = VFNMA(vi, S1, D2); E2 = VFNMA(vr, S1, E2);
          D3 = VFMA(vi, S5, D3); E3 = VFMA(vr, S5, E3);
          D4 = VFNMA(vi, S2, D4); E4 = VFNMA(vr, S2, E4);
          D5 = VFMA(vi, S4, D5); E5 = VFMA(vr, S4, E5);
          D6 = VFNMA(vi, S3, D6); E6 = VFNMA(vr, S3, E6);
        }
        VS(PSCR + 0, D1); VS(PSCR + 8, E1);
        VS(PSCR + 16, D2); VS(PSCR + 24, E2);
        VS(PSCR + 32, D3); VS(PSCR + 40, E3);
        VS(PSCR + 48, D4); VS(PSCR + 56, E4);
        VS(PSCR + 64, D5); VS(PSCR + 72, E5);
        VS(PSCR + 80, D6); VS(PSCR + 88, E6);
        }
        VS(USCR + 384, VL(pr)); VS(USCR + 392, VL(pi));  // save x0 (row 0 gets overwritten)
        {
        V C1 = VSET1(0x1.c55a7e00740e9p-1);
        V C2 = VSET1(0x1.22d961ea71119p-1);
        V C3 = VSET1(0x1.edb7debaa3ed5p-4);
        V C4 = VSET1(-0x1.6b1d8b2365d9ep-2);
        V C5 = VSET1(-0x1.7f3ccd0032e0dp-1);
        V C6 = VSET1(-0x1.f11f493053d00p-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A1 = x0r, B1 = x0i;
        V A2 = x0r, B2 = x0i;
        V A3 = x0r, B3 = x0i;
        V A4 = x0r, B4 = x0i;
        V A5 = x0r, B5 = x0i;
        V A6 = x0r, B6 = x0i;
        V s0r = x0r, s0i = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A1 = VFMA(ur, C1, A1); B1 = VFMA(ui, C1, B1);
          A2 = VFMA(ur, C2, A2); B2 = VFMA(ui, C2, B2);
          A3 = VFMA(ur, C3, A3); B3 = VFMA(ui, C3, B3);
          A4 = VFMA(ur, C4, A4); B4 = VFMA(ui, C4, B4);
          A5 = VFMA(ur, C5, A5); B5 = VFMA(ui, C5, B5);
          A6 = VFMA(ur, C6, A6); B6 = VFMA(ui, C6, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A1 = VFMA(ur, C2, A1); B1 = VFMA(ui, C2, B1);
          A2 = VFMA(ur, C4, A2); B2 = VFMA(ui, C4, B2);
          A3 = VFMA(ur, C6, A3); B3 = VFMA(ui, C6, B3);
          A4 = VFMA(ur, C5, A4); B4 = VFMA(ui, C5, B4);
          A5 = VFMA(ur, C3, A5); B5 = VFMA(ui, C3, B5);
          A6 = VFMA(ur, C1, A6); B6 = VFMA(ui, C1, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A1 = VFMA(ur, C3, A1); B1 = VFMA(ui, C3, B1);
          A2 = VFMA(ur, C6, A2); B2 = VFMA(ui, C6, B2);
          A3 = VFMA(ur, C4, A3); B3 = VFMA(ui, C4, B3);
          A4 = VFMA(ur, C1, A4); B4 = VFMA(ui, C1, B4);
          A5 = VFMA(ur, C2, A5); B5 = VFMA(ui, C2, B5);
          A6 = VFMA(ur, C5, A6); B6 = VFMA(ui, C5, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A1 = VFMA(ur, C4, A1); B1 = VFMA(ui, C4, B1);
          A2 = VFMA(ur, C5, A2); B2 = VFMA(ui, C5, B2);
          A3 = VFMA(ur, C1, A3); B3 = VFMA(ui, C1, B3);
          A4 = VFMA(ur, C3, A4); B4 = VFMA(ui, C3, B4);
          A5 = VFMA(ur, C6, A5); B5 = VFMA(ui, C6, B5);
          A6 = VFMA(ur, C2, A6); B6 = VFMA(ui, C2, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A1 = VFMA(ur, C5, A1); B1 = VFMA(ui, C5, B1);
          A2 = VFMA(ur, C3, A2); B2 = VFMA(ui, C3, B2);
          A3 = VFMA(ur, C2, A3); B3 = VFMA(ui, C2, B3);
          A4 = VFMA(ur, C6, A4); B4 = VFMA(ui, C6, B4);
          A5 = VFMA(ur, C1, A5); B5 = VFMA(ui, C1, B5);
          A6 = VFMA(ur, C4, A6); B6 = VFMA(ui, C4, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A1 = VFMA(ur, C6, A1); B1 = VFMA(ui, C6, B1);
          A2 = VFMA(ur, C1, A2); B2 = VFMA(ui, C1, B2);
          A3 = VFMA(ur, C5, A3); B3 = VFMA(ui, C5, B3);
          A4 = VFMA(ur, C2, A4); B4 = VFMA(ui, C2, B4);
          A5 = VFMA(ur, C4, A5); B5 = VFMA(ui, C4, B5);
          A6 = VFMA(ur, C3, A6); B6 = VFMA(ui, C3, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V zr_ = VADD(s0r, VL(cr + 0)), zi_ = VADD(s0i, VL(ci + 0));
          MAP2(zr_, zi_);
          VS(pr + 0, zr_); VS(pi + 0, zi_); }
        { V Dk = VL(PSCR + 0), Ek = VL(PSCR + 8);
          V Xr = VADD(A1, Dk), Xi = VSUB(B1, Ek);
          V Yr = VSUB(A1, Dk), Yi = VADD(B1, Ek);
        { V zr_ = VADD(Xr, VL(cr + 104)), zi_ = VADD(Xi, VL(ci + 104));
          MAP2(zr_, zi_);
          VS(pr + 104, zr_); VS(pi + 104, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 1248)), zi_ = VADD(Yi, VL(ci + 1248));
          MAP2(zr_, zi_);
          VS(pr + 1248, zr_); VS(pi + 1248, zi_); }
        }
        { V Dk = VL(PSCR + 16), Ek = VL(PSCR + 24);
          V Xr = VADD(A2, Dk), Xi = VSUB(B2, Ek);
          V Yr = VSUB(A2, Dk), Yi = VADD(B2, Ek);
        { V zr_ = VADD(Xr, VL(cr + 208)), zi_ = VADD(Xi, VL(ci + 208));
          MAP2(zr_, zi_);
          VS(pr + 208, zr_); VS(pi + 208, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 1144)), zi_ = VADD(Yi, VL(ci + 1144));
          MAP2(zr_, zi_);
          VS(pr + 1144, zr_); VS(pi + 1144, zi_); }
        }
        { V Dk = VL(PSCR + 32), Ek = VL(PSCR + 40);
          V Xr = VADD(A3, Dk), Xi = VSUB(B3, Ek);
          V Yr = VSUB(A3, Dk), Yi = VADD(B3, Ek);
        { V zr_ = VADD(Xr, VL(cr + 312)), zi_ = VADD(Xi, VL(ci + 312));
          MAP2(zr_, zi_);
          VS(pr + 312, zr_); VS(pi + 312, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 1040)), zi_ = VADD(Yi, VL(ci + 1040));
          MAP2(zr_, zi_);
          VS(pr + 1040, zr_); VS(pi + 1040, zi_); }
        }
        { V Dk = VL(PSCR + 48), Ek = VL(PSCR + 56);
          V Xr = VADD(A4, Dk), Xi = VSUB(B4, Ek);
          V Yr = VSUB(A4, Dk), Yi = VADD(B4, Ek);
        { V zr_ = VADD(Xr, VL(cr + 416)), zi_ = VADD(Xi, VL(ci + 416));
          MAP2(zr_, zi_);
          VS(pr + 416, zr_); VS(pi + 416, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 936)), zi_ = VADD(Yi, VL(ci + 936));
          MAP2(zr_, zi_);
          VS(pr + 936, zr_); VS(pi + 936, zi_); }
        }
        { V Dk = VL(PSCR + 64), Ek = VL(PSCR + 72);
          V Xr = VADD(A5, Dk), Xi = VSUB(B5, Ek);
          V Yr = VSUB(A5, Dk), Yi = VADD(B5, Ek);
        { V zr_ = VADD(Xr, VL(cr + 520)), zi_ = VADD(Xi, VL(ci + 520));
          MAP2(zr_, zi_);
          VS(pr + 520, zr_); VS(pi + 520, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 832)), zi_ = VADD(Yi, VL(ci + 832));
          MAP2(zr_, zi_);
          VS(pr + 832, zr_); VS(pi + 832, zi_); }
        }
        { V Dk = VL(PSCR + 80), Ek = VL(PSCR + 88);
          V Xr = VADD(A6, Dk), Xi = VSUB(B6, Ek);
          V Yr = VSUB(A6, Dk), Yi = VADD(B6, Ek);
        { V zr_ = VADD(Xr, VL(cr + 624)), zi_ = VADD(Xi, VL(ci + 624));
          MAP2(zr_, zi_);
          VS(pr + 624, zr_); VS(pi + 624, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 728)), zi_ = VADD(Yi, VL(ci + 728));
          MAP2(zr_, zi_);
          VS(pr + 728, zr_); VS(pi + 728, zi_); }
        }
        }
    }
}

static void sw13_SyX(int y0, int pre){
    long b0 = (long)y0*13*8;
    p13_xxm(S13RE + b0, S13IM + b0, C13RE + b0, C13IM + b0, 13, 8);
    if (pre) {
        p13_zz(S13RE + (long)y0*13*8, S13IM + (long)y0*13*8, 13, (long)13*13*8);
        p13_xx(S13RE + b0, S13IM + b0, 13, 8);
    }
}
static void sw13_PxY(int x0, int pre){
    long b0 = (long)x0*13*13*8;
    p13_yym(S13RE + b0, S13IM + b0, C13RE + b0, C13IM + b0, 13, 8);
    if (pre) {
        p13_zz(S13RE + b0, S13IM + b0, 13, (long)13*8);
        p13_yy(S13RE + b0, S13IM + b0, 13, 8);
    }
}
void run_13(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if (m < 1) m = 1;
    long vs = 2*(long)2197;
    for (long g0 = 0; g0 < B; g0 += 8) {
        int nv = (B - g0) < 8 ? (int)(B - g0) : 8;
        const double* xg = x0 + g0*vs;
        const double* cg = c + g0*vs;
        long nfull = (2197/8)*8;
        if (nv == 8) {
            for (long s = 0; s < nfull; s += 8) { soa_in8(xg, vs, s, S13RE, S13IM); soa_in8(cg, vs, s, C13RE, C13IM); }
        } else {
            for (long s = 0; s < nfull; s += 8) { soa_in8_nv(xg, vs, s, S13RE, S13IM, nv); soa_in8_nv(cg, vs, s, C13RE, C13IM, nv); }
        }
        for (long s = nfull; s < 2197; s++) {   // tail sites scalar
            for (int v = 0; v < 8; v++) {
                S13RE[s*8+v] = v < nv ? xg[v*vs + 2*s] : 0.0;
                S13IM[s*8+v] = v < nv ? xg[v*vs + 2*s + 1] : 0.0;
                C13RE[s*8+v] = v < nv ? cg[v*vs + 2*s] : 0.0;
                C13IM[s*8+v] = v < nv ? cg[v*vs + 2*s + 1] : 0.0;
            }
        }
        // prologue: Z,Y per plane x
        for (int x = 0; x < 13; x++) {
            long b0 = (long)x*13*13*8;
            p13_zz(S13RE + b0, S13IM + b0, 13, (long)13*8);
            p13_yy(S13RE + b0, S13IM + b0, 13, 8);
        }
        for (long t = 1; t <= m; t++) {
            int snap = (t == 1) || (t == m);
            int pre = (t < m);
            int dopre = pre && !snap;
            if (t & 1) { for (int y0 = 0; y0 < 13; y0++) sw13_SyX(y0, dopre); }
            else       { for (int xp = 0; xp < 13; xp++) sw13_PxY(xp, dopre); }
            if (t == 1) {
                for (long s = 0; s < nfull; s += 8) soa_out8(S13RE, S13IM, s, out1 + g0*vs, vs, nv);
                for (long s = nfull; s < 2197; s++) for (int v = 0; v < nv; v++) {
                    out1[g0*vs + v*vs + 2*s] = S13RE[s*8+v]; out1[g0*vs + v*vs + 2*s+1] = S13IM[s*8+v]; }
            }
            if (t == m) {
                for (long s = 0; s < nfull; s += 8) soa_out8(S13RE, S13IM, s, outm + g0*vs, vs, nv);
                for (long s = nfull; s < 2197; s++) for (int v = 0; v < nv; v++) {
                    outm[g0*vs + v*vs + 2*s] = S13RE[s*8+v]; outm[g0*vs + v*vs + 2*s+1] = S13IM[s*8+v]; }
            }
            if (pre && !dopre) {
                p13_zz(S13RE, S13IM, 13*13, (long)13*8);
                if (t & 1) {
                    for (int y0 = 0; y0 < 13; y0++)
                        p13_xx(S13RE + (long)y0*13*8, S13IM + (long)y0*13*8, 13, 8);
                } else {
                    for (int x0 = 0; x0 < 13; x0++)
                        p13_yy(S13RE + (long)x0*13*13*8, S13IM + (long)x0*13*13*8, 13, 8);
                }
            }
        }
    }
}

// ============ L=17 SoA-8 ============
static double S17RE[39304] ALIGN64;
static double S17IM[39304] ALIGN64;
static double C17RE[39304] ALIGN64;
static double C17IM[39304] ALIGN64;
static const double CT17[8] ALIGN64 = { 0x1.dd6d000370991p-1, 0x1.7a5f6075d4884p-1, 0x1.c86fa2b2883cep-2, 0x1.79ee63259b75fp-4, -0x1.183b1c61f0d01p-2, -0x1.348c86ed5f1bap-1, -0x1.b34fa910ea3b8p-1, -0x1.f7484007faef3p-1 };
static const double ST17[8] ALIGN64 = { 0x1.71e955d8e7cdcp-2, 0x1.58eea2a9d6da3p-1, 0x1.ca52d7c9e640bp-1, 0x1.fdd0deb564b22p-1, 0x1.ec746923c349fp-1, 0x1.9895b6c9a05f7p-1, 0x1.0d8884363dd82p-1, 0x1.7851aacd6c6b5p-3 };
static __attribute__((noinline)) void p17_zz(double* PR, double* PI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        {
        V S1 = VSET1(0x1.71e955d8e7cdcp-2);
        V S2 = VSET1(0x1.58eea2a9d6da3p-1);
        V S3 = VSET1(0x1.ca52d7c9e640bp-1);
        V S4 = VSET1(0x1.fdd0deb564b22p-1);
        V S5 = VSET1(0x1.ec746923c349fp-1);
        V S6 = VSET1(0x1.9895b6c9a05f7p-1);
        V S7 = VSET1(0x1.0d8884363dd82p-1);
        V S8 = VSET1(0x1.7851aacd6c6b5p-3);
        V D1 = _mm512_setzero_pd(), E1 = _mm512_setzero_pd();
        V D2 = _mm512_setzero_pd(), E2 = _mm512_setzero_pd();
        V D3 = _mm512_setzero_pd(), E3 = _mm512_setzero_pd();
        V D4 = _mm512_setzero_pd(), E4 = _mm512_setzero_pd();
        V D5 = _mm512_setzero_pd(), E5 = _mm512_setzero_pd();
        V D6 = _mm512_setzero_pd(), E6 = _mm512_setzero_pd();
        V D7 = _mm512_setzero_pd(), E7 = _mm512_setzero_pd();
        V D8 = _mm512_setzero_pd(), E8 = _mm512_setzero_pd();
        { V ar = VL(pr + 8), ai = VL(pi + 8);
          V br = VL(pr + 128), bi = VL(pi + 128);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 0, VADD(ar,br)); VS(USCR + 8, VADD(ai,bi));
          D1 = VFMA(vi, S1, D1); E1 = VFMA(vr, S1, E1);
          D2 = VFMA(vi, S2, D2); E2 = VFMA(vr, S2, E2);
          D3 = VFMA(vi, S3, D3); E3 = VFMA(vr, S3, E3);
          D4 = VFMA(vi, S4, D4); E4 = VFMA(vr, S4, E4);
          D5 = VFMA(vi, S5, D5); E5 = VFMA(vr, S5, E5);
          D6 = VFMA(vi, S6, D6); E6 = VFMA(vr, S6, E6);
          D7 = VFMA(vi, S7, D7); E7 = VFMA(vr, S7, E7);
          D8 = VFMA(vi, S8, D8); E8 = VFMA(vr, S8, E8);
        }
        { V ar = VL(pr + 16), ai = VL(pi + 16);
          V br = VL(pr + 120), bi = VL(pi + 120);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 16, VADD(ar,br)); VS(USCR + 24, VADD(ai,bi));
          D1 = VFMA(vi, S2, D1); E1 = VFMA(vr, S2, E1);
          D2 = VFMA(vi, S4, D2); E2 = VFMA(vr, S4, E2);
          D3 = VFMA(vi, S6, D3); E3 = VFMA(vr, S6, E3);
          D4 = VFMA(vi, S8, D4); E4 = VFMA(vr, S8, E4);
          D5 = VFNMA(vi, S7, D5); E5 = VFNMA(vr, S7, E5);
          D6 = VFNMA(vi, S5, D6); E6 = VFNMA(vr, S5, E6);
          D7 = VFNMA(vi, S3, D7); E7 = VFNMA(vr, S3, E7);
          D8 = VFNMA(vi, S1, D8); E8 = VFNMA(vr, S1, E8);
        }
        { V ar = VL(pr + 24), ai = VL(pi + 24);
          V br = VL(pr + 112), bi = VL(pi + 112);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 32, VADD(ar,br)); VS(USCR + 40, VADD(ai,bi));
          D1 = VFMA(vi, S3, D1); E1 = VFMA(vr, S3, E1);
          D2 = VFMA(vi, S6, D2); E2 = VFMA(vr, S6, E2);
          D3 = VFNMA(vi, S8, D3); E3 = VFNMA(vr, S8, E3);
          D4 = VFNMA(vi, S5, D4); E4 = VFNMA(vr, S5, E4);
          D5 = VFNMA(vi, S2, D5); E5 = VFNMA(vr, S2, E5);
          D6 = VFMA(vi, S1, D6); E6 = VFMA(vr, S1, E6);
          D7 = VFMA(vi, S4, D7); E7 = VFMA(vr, S4, E7);
          D8 = VFMA(vi, S7, D8); E8 = VFMA(vr, S7, E8);
        }
        { V ar = VL(pr + 32), ai = VL(pi + 32);
          V br = VL(pr + 104), bi = VL(pi + 104);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 48, VADD(ar,br)); VS(USCR + 56, VADD(ai,bi));
          D1 = VFMA(vi, S4, D1); E1 = VFMA(vr, S4, E1);
          D2 = VFMA(vi, S8, D2); E2 = VFMA(vr, S8, E2);
          D3 = VFNMA(vi, S5, D3); E3 = VFNMA(vr, S5, E3);
          D4 = VFNMA(vi, S1, D4); E4 = VFNMA(vr, S1, E4);
          D5 = VFMA(vi, S3, D5); E5 = VFMA(vr, S3, E5);
          D6 = VFMA(vi, S7, D6); E6 = VFMA(vr, S7, E6);
          D7 = VFNMA(vi, S6, D7); E7 = VFNMA(vr, S6, E7);
          D8 = VFNMA(vi, S2, D8); E8 = VFNMA(vr, S2, E8);
        }
        { V ar = VL(pr + 40), ai = VL(pi + 40);
          V br = VL(pr + 96), bi = VL(pi + 96);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 64, VADD(ar,br)); VS(USCR + 72, VADD(ai,bi));
          D1 = VFMA(vi, S5, D1); E1 = VFMA(vr, S5, E1);
          D2 = VFNMA(vi, S7, D2); E2 = VFNMA(vr, S7, E2);
          D3 = VFNMA(vi, S2, D3); E3 = VFNMA(vr, S2, E3);
          D4 = VFMA(vi, S3, D4); E4 = VFMA(vr, S3, E4);
          D5 = VFMA(vi, S8, D5); E5 = VFMA(vr, S8, E5);
          D6 = VFNMA(vi, S4, D6); E6 = VFNMA(vr, S4, E6);
          D7 = VFMA(vi, S1, D7); E7 = VFMA(vr, S1, E7);
          D8 = VFMA(vi, S6, D8); E8 = VFMA(vr, S6, E8);
        }
        { V ar = VL(pr + 48), ai = VL(pi + 48);
          V br = VL(pr + 88), bi = VL(pi + 88);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 80, VADD(ar,br)); VS(USCR + 88, VADD(ai,bi));
          D1 = VFMA(vi, S6, D1); E1 = VFMA(vr, S6, E1);
          D2 = VFNMA(vi, S5, D2); E2 = VFNMA(vr, S5, E2);
          D3 = VFMA(vi, S1, D3); E3 = VFMA(vr, S1, E3);
          D4 = VFMA(vi, S7, D4); E4 = VFMA(vr, S7, E4);
          D5 = VFNMA(vi, S4, D5); E5 = VFNMA(vr, S4, E5);
          D6 = VFMA(vi, S2, D6); E6 = VFMA(vr, S2, E6);
          D7 = VFMA(vi, S8, D7); E7 = VFMA(vr, S8, E7);
          D8 = VFNMA(vi, S3, D8); E8 = VFNMA(vr, S3, E8);
        }
        { V ar = VL(pr + 56), ai = VL(pi + 56);
          V br = VL(pr + 80), bi = VL(pi + 80);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 96, VADD(ar,br)); VS(USCR + 104, VADD(ai,bi));
          D1 = VFMA(vi, S7, D1); E1 = VFMA(vr, S7, E1);
          D2 = VFNMA(vi, S3, D2); E2 = VFNMA(vr, S3, E2);
          D3 = VFMA(vi, S4, D3); E3 = VFMA(vr, S4, E3);
          D4 = VFNMA(vi, S6, D4); E4 = VFNMA(vr, S6, E4);
          D5 = VFMA(vi, S1, D5); E5 = VFMA(vr, S1, E5);
          D6 = VFMA(vi, S8, D6); E6 = VFMA(vr, S8, E6);
          D7 = VFNMA(vi, S2, D7); E7 = VFNMA(vr, S2, E7);
          D8 = VFMA(vi, S5, D8); E8 = VFMA(vr, S5, E8);
        }
        { V ar = VL(pr + 64), ai = VL(pi + 64);
          V br = VL(pr + 72), bi = VL(pi + 72);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 112, VADD(ar,br)); VS(USCR + 120, VADD(ai,bi));
          D1 = VFMA(vi, S8, D1); E1 = VFMA(vr, S8, E1);
          D2 = VFNMA(vi, S1, D2); E2 = VFNMA(vr, S1, E2);
          D3 = VFMA(vi, S7, D3); E3 = VFMA(vr, S7, E3);
          D4 = VFNMA(vi, S2, D4); E4 = VFNMA(vr, S2, E4);
          D5 = VFMA(vi, S6, D5); E5 = VFMA(vr, S6, E5);
          D6 = VFNMA(vi, S3, D6); E6 = VFNMA(vr, S3, E6);
          D7 = VFMA(vi, S5, D7); E7 = VFMA(vr, S5, E7);
          D8 = VFNMA(vi, S4, D8); E8 = VFNMA(vr, S4, E8);
        }
        VS(PSCR + 0, D1); VS(PSCR + 8, E1);
        VS(PSCR + 16, D2); VS(PSCR + 24, E2);
        VS(PSCR + 32, D3); VS(PSCR + 40, E3);
        VS(PSCR + 48, D4); VS(PSCR + 56, E4);
        VS(PSCR + 64, D5); VS(PSCR + 72, E5);
        VS(PSCR + 80, D6); VS(PSCR + 88, E6);
        VS(PSCR + 96, D7); VS(PSCR + 104, E7);
        VS(PSCR + 112, D8); VS(PSCR + 120, E8);
        }
        VS(USCR + 384, VL(pr)); VS(USCR + 392, VL(pi));  // save x0 (row 0 gets overwritten)
        {
        V C1 = VSET1(0x1.dd6d000370991p-1);
        V C2 = VSET1(0x1.7a5f6075d4884p-1);
        V C3 = VSET1(0x1.c86fa2b2883cep-2);
        V C4 = VSET1(0x1.79ee63259b75fp-4);
        V C5 = VSET1(-0x1.183b1c61f0d01p-2);
        V C6 = VSET1(-0x1.348c86ed5f1bap-1);
        V C7 = VSET1(-0x1.b34fa910ea3b8p-1);
        V C8 = VSET1(-0x1.f7484007faef3p-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A1 = x0r, B1 = x0i;
        V A2 = x0r, B2 = x0i;
        V A3 = x0r, B3 = x0i;
        V A4 = x0r, B4 = x0i;
        V A5 = x0r, B5 = x0i;
        V A6 = x0r, B6 = x0i;
        V A7 = x0r, B7 = x0i;
        V A8 = x0r, B8 = x0i;
        V s0r = x0r, s0i = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A1 = VFMA(ur, C1, A1); B1 = VFMA(ui, C1, B1);
          A2 = VFMA(ur, C2, A2); B2 = VFMA(ui, C2, B2);
          A3 = VFMA(ur, C3, A3); B3 = VFMA(ui, C3, B3);
          A4 = VFMA(ur, C4, A4); B4 = VFMA(ui, C4, B4);
          A5 = VFMA(ur, C5, A5); B5 = VFMA(ui, C5, B5);
          A6 = VFMA(ur, C6, A6); B6 = VFMA(ui, C6, B6);
          A7 = VFMA(ur, C7, A7); B7 = VFMA(ui, C7, B7);
          A8 = VFMA(ur, C8, A8); B8 = VFMA(ui, C8, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A1 = VFMA(ur, C2, A1); B1 = VFMA(ui, C2, B1);
          A2 = VFMA(ur, C4, A2); B2 = VFMA(ui, C4, B2);
          A3 = VFMA(ur, C6, A3); B3 = VFMA(ui, C6, B3);
          A4 = VFMA(ur, C8, A4); B4 = VFMA(ui, C8, B4);
          A5 = VFMA(ur, C7, A5); B5 = VFMA(ui, C7, B5);
          A6 = VFMA(ur, C5, A6); B6 = VFMA(ui, C5, B6);
          A7 = VFMA(ur, C3, A7); B7 = VFMA(ui, C3, B7);
          A8 = VFMA(ur, C1, A8); B8 = VFMA(ui, C1, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A1 = VFMA(ur, C3, A1); B1 = VFMA(ui, C3, B1);
          A2 = VFMA(ur, C6, A2); B2 = VFMA(ui, C6, B2);
          A3 = VFMA(ur, C8, A3); B3 = VFMA(ui, C8, B3);
          A4 = VFMA(ur, C5, A4); B4 = VFMA(ui, C5, B4);
          A5 = VFMA(ur, C2, A5); B5 = VFMA(ui, C2, B5);
          A6 = VFMA(ur, C1, A6); B6 = VFMA(ui, C1, B6);
          A7 = VFMA(ur, C4, A7); B7 = VFMA(ui, C4, B7);
          A8 = VFMA(ur, C7, A8); B8 = VFMA(ui, C7, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A1 = VFMA(ur, C4, A1); B1 = VFMA(ui, C4, B1);
          A2 = VFMA(ur, C8, A2); B2 = VFMA(ui, C8, B2);
          A3 = VFMA(ur, C5, A3); B3 = VFMA(ui, C5, B3);
          A4 = VFMA(ur, C1, A4); B4 = VFMA(ui, C1, B4);
          A5 = VFMA(ur, C3, A5); B5 = VFMA(ui, C3, B5);
          A6 = VFMA(ur, C7, A6); B6 = VFMA(ui, C7, B6);
          A7 = VFMA(ur, C6, A7); B7 = VFMA(ui, C6, B7);
          A8 = VFMA(ur, C2, A8); B8 = VFMA(ui, C2, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A1 = VFMA(ur, C5, A1); B1 = VFMA(ui, C5, B1);
          A2 = VFMA(ur, C7, A2); B2 = VFMA(ui, C7, B2);
          A3 = VFMA(ur, C2, A3); B3 = VFMA(ui, C2, B3);
          A4 = VFMA(ur, C3, A4); B4 = VFMA(ui, C3, B4);
          A5 = VFMA(ur, C8, A5); B5 = VFMA(ui, C8, B5);
          A6 = VFMA(ur, C4, A6); B6 = VFMA(ui, C4, B6);
          A7 = VFMA(ur, C1, A7); B7 = VFMA(ui, C1, B7);
          A8 = VFMA(ur, C6, A8); B8 = VFMA(ui, C6, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A1 = VFMA(ur, C6, A1); B1 = VFMA(ui, C6, B1);
          A2 = VFMA(ur, C5, A2); B2 = VFMA(ui, C5, B2);
          A3 = VFMA(ur, C1, A3); B3 = VFMA(ui, C1, B3);
          A4 = VFMA(ur, C7, A4); B4 = VFMA(ui, C7, B4);
          A5 = VFMA(ur, C4, A5); B5 = VFMA(ui, C4, B5);
          A6 = VFMA(ur, C2, A6); B6 = VFMA(ui, C2, B6);
          A7 = VFMA(ur, C8, A7); B7 = VFMA(ui, C8, B7);
          A8 = VFMA(ur, C3, A8); B8 = VFMA(ui, C3, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 96), ui = VL(USCR + 104);
          A1 = VFMA(ur, C7, A1); B1 = VFMA(ui, C7, B1);
          A2 = VFMA(ur, C3, A2); B2 = VFMA(ui, C3, B2);
          A3 = VFMA(ur, C4, A3); B3 = VFMA(ui, C4, B3);
          A4 = VFMA(ur, C6, A4); B4 = VFMA(ui, C6, B4);
          A5 = VFMA(ur, C1, A5); B5 = VFMA(ui, C1, B5);
          A6 = VFMA(ur, C8, A6); B6 = VFMA(ui, C8, B6);
          A7 = VFMA(ur, C2, A7); B7 = VFMA(ui, C2, B7);
          A8 = VFMA(ur, C5, A8); B8 = VFMA(ui, C5, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 112), ui = VL(USCR + 120);
          A1 = VFMA(ur, C8, A1); B1 = VFMA(ui, C8, B1);
          A2 = VFMA(ur, C1, A2); B2 = VFMA(ui, C1, B2);
          A3 = VFMA(ur, C7, A3); B3 = VFMA(ui, C7, B3);
          A4 = VFMA(ur, C2, A4); B4 = VFMA(ui, C2, B4);
          A5 = VFMA(ur, C6, A5); B5 = VFMA(ui, C6, B5);
          A6 = VFMA(ur, C3, A6); B6 = VFMA(ui, C3, B6);
          A7 = VFMA(ur, C5, A7); B7 = VFMA(ui, C5, B7);
          A8 = VFMA(ur, C4, A8); B8 = VFMA(ui, C4, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        VS(pr + 0, s0r); VS(pi + 0, s0i);
        { V Dk = VL(PSCR + 0), Ek = VL(PSCR + 8);
          V Xr = VADD(A1, Dk), Xi = VSUB(B1, Ek);
          V Yr = VSUB(A1, Dk), Yi = VADD(B1, Ek);
        VS(pr + 8, Xr); VS(pi + 8, Xi);
        VS(pr + 128, Yr); VS(pi + 128, Yi);
        }
        { V Dk = VL(PSCR + 16), Ek = VL(PSCR + 24);
          V Xr = VADD(A2, Dk), Xi = VSUB(B2, Ek);
          V Yr = VSUB(A2, Dk), Yi = VADD(B2, Ek);
        VS(pr + 16, Xr); VS(pi + 16, Xi);
        VS(pr + 120, Yr); VS(pi + 120, Yi);
        }
        { V Dk = VL(PSCR + 32), Ek = VL(PSCR + 40);
          V Xr = VADD(A3, Dk), Xi = VSUB(B3, Ek);
          V Yr = VSUB(A3, Dk), Yi = VADD(B3, Ek);
        VS(pr + 24, Xr); VS(pi + 24, Xi);
        VS(pr + 112, Yr); VS(pi + 112, Yi);
        }
        { V Dk = VL(PSCR + 48), Ek = VL(PSCR + 56);
          V Xr = VADD(A4, Dk), Xi = VSUB(B4, Ek);
          V Yr = VSUB(A4, Dk), Yi = VADD(B4, Ek);
        VS(pr + 32, Xr); VS(pi + 32, Xi);
        VS(pr + 104, Yr); VS(pi + 104, Yi);
        }
        { V Dk = VL(PSCR + 64), Ek = VL(PSCR + 72);
          V Xr = VADD(A5, Dk), Xi = VSUB(B5, Ek);
          V Yr = VSUB(A5, Dk), Yi = VADD(B5, Ek);
        VS(pr + 40, Xr); VS(pi + 40, Xi);
        VS(pr + 96, Yr); VS(pi + 96, Yi);
        }
        { V Dk = VL(PSCR + 80), Ek = VL(PSCR + 88);
          V Xr = VADD(A6, Dk), Xi = VSUB(B6, Ek);
          V Yr = VSUB(A6, Dk), Yi = VADD(B6, Ek);
        VS(pr + 48, Xr); VS(pi + 48, Xi);
        VS(pr + 88, Yr); VS(pi + 88, Yi);
        }
        { V Dk = VL(PSCR + 96), Ek = VL(PSCR + 104);
          V Xr = VADD(A7, Dk), Xi = VSUB(B7, Ek);
          V Yr = VSUB(A7, Dk), Yi = VADD(B7, Ek);
        VS(pr + 56, Xr); VS(pi + 56, Xi);
        VS(pr + 80, Yr); VS(pi + 80, Yi);
        }
        { V Dk = VL(PSCR + 112), Ek = VL(PSCR + 120);
          V Xr = VADD(A8, Dk), Xi = VSUB(B8, Ek);
          V Yr = VSUB(A8, Dk), Yi = VADD(B8, Ek);
        VS(pr + 64, Xr); VS(pi + 64, Xi);
        VS(pr + 72, Yr); VS(pi + 72, Yi);
        }
        }
    }
}
static __attribute__((noinline)) void p17_yy(double* PR, double* PI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        {
        V S1 = VSET1(0x1.71e955d8e7cdcp-2);
        V S2 = VSET1(0x1.58eea2a9d6da3p-1);
        V S3 = VSET1(0x1.ca52d7c9e640bp-1);
        V S4 = VSET1(0x1.fdd0deb564b22p-1);
        V S5 = VSET1(0x1.ec746923c349fp-1);
        V S6 = VSET1(0x1.9895b6c9a05f7p-1);
        V S7 = VSET1(0x1.0d8884363dd82p-1);
        V S8 = VSET1(0x1.7851aacd6c6b5p-3);
        V D1 = _mm512_setzero_pd(), E1 = _mm512_setzero_pd();
        V D2 = _mm512_setzero_pd(), E2 = _mm512_setzero_pd();
        V D3 = _mm512_setzero_pd(), E3 = _mm512_setzero_pd();
        V D4 = _mm512_setzero_pd(), E4 = _mm512_setzero_pd();
        V D5 = _mm512_setzero_pd(), E5 = _mm512_setzero_pd();
        V D6 = _mm512_setzero_pd(), E6 = _mm512_setzero_pd();
        V D7 = _mm512_setzero_pd(), E7 = _mm512_setzero_pd();
        V D8 = _mm512_setzero_pd(), E8 = _mm512_setzero_pd();
        { V ar = VL(pr + 136), ai = VL(pi + 136);
          V br = VL(pr + 2176), bi = VL(pi + 2176);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 0, VADD(ar,br)); VS(USCR + 8, VADD(ai,bi));
          D1 = VFMA(vi, S1, D1); E1 = VFMA(vr, S1, E1);
          D2 = VFMA(vi, S2, D2); E2 = VFMA(vr, S2, E2);
          D3 = VFMA(vi, S3, D3); E3 = VFMA(vr, S3, E3);
          D4 = VFMA(vi, S4, D4); E4 = VFMA(vr, S4, E4);
          D5 = VFMA(vi, S5, D5); E5 = VFMA(vr, S5, E5);
          D6 = VFMA(vi, S6, D6); E6 = VFMA(vr, S6, E6);
          D7 = VFMA(vi, S7, D7); E7 = VFMA(vr, S7, E7);
          D8 = VFMA(vi, S8, D8); E8 = VFMA(vr, S8, E8);
        }
        { V ar = VL(pr + 272), ai = VL(pi + 272);
          V br = VL(pr + 2040), bi = VL(pi + 2040);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 16, VADD(ar,br)); VS(USCR + 24, VADD(ai,bi));
          D1 = VFMA(vi, S2, D1); E1 = VFMA(vr, S2, E1);
          D2 = VFMA(vi, S4, D2); E2 = VFMA(vr, S4, E2);
          D3 = VFMA(vi, S6, D3); E3 = VFMA(vr, S6, E3);
          D4 = VFMA(vi, S8, D4); E4 = VFMA(vr, S8, E4);
          D5 = VFNMA(vi, S7, D5); E5 = VFNMA(vr, S7, E5);
          D6 = VFNMA(vi, S5, D6); E6 = VFNMA(vr, S5, E6);
          D7 = VFNMA(vi, S3, D7); E7 = VFNMA(vr, S3, E7);
          D8 = VFNMA(vi, S1, D8); E8 = VFNMA(vr, S1, E8);
        }
        { V ar = VL(pr + 408), ai = VL(pi + 408);
          V br = VL(pr + 1904), bi = VL(pi + 1904);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 32, VADD(ar,br)); VS(USCR + 40, VADD(ai,bi));
          D1 = VFMA(vi, S3, D1); E1 = VFMA(vr, S3, E1);
          D2 = VFMA(vi, S6, D2); E2 = VFMA(vr, S6, E2);
          D3 = VFNMA(vi, S8, D3); E3 = VFNMA(vr, S8, E3);
          D4 = VFNMA(vi, S5, D4); E4 = VFNMA(vr, S5, E4);
          D5 = VFNMA(vi, S2, D5); E5 = VFNMA(vr, S2, E5);
          D6 = VFMA(vi, S1, D6); E6 = VFMA(vr, S1, E6);
          D7 = VFMA(vi, S4, D7); E7 = VFMA(vr, S4, E7);
          D8 = VFMA(vi, S7, D8); E8 = VFMA(vr, S7, E8);
        }
        { V ar = VL(pr + 544), ai = VL(pi + 544);
          V br = VL(pr + 1768), bi = VL(pi + 1768);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 48, VADD(ar,br)); VS(USCR + 56, VADD(ai,bi));
          D1 = VFMA(vi, S4, D1); E1 = VFMA(vr, S4, E1);
          D2 = VFMA(vi, S8, D2); E2 = VFMA(vr, S8, E2);
          D3 = VFNMA(vi, S5, D3); E3 = VFNMA(vr, S5, E3);
          D4 = VFNMA(vi, S1, D4); E4 = VFNMA(vr, S1, E4);
          D5 = VFMA(vi, S3, D5); E5 = VFMA(vr, S3, E5);
          D6 = VFMA(vi, S7, D6); E6 = VFMA(vr, S7, E6);
          D7 = VFNMA(vi, S6, D7); E7 = VFNMA(vr, S6, E7);
          D8 = VFNMA(vi, S2, D8); E8 = VFNMA(vr, S2, E8);
        }
        { V ar = VL(pr + 680), ai = VL(pi + 680);
          V br = VL(pr + 1632), bi = VL(pi + 1632);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 64, VADD(ar,br)); VS(USCR + 72, VADD(ai,bi));
          D1 = VFMA(vi, S5, D1); E1 = VFMA(vr, S5, E1);
          D2 = VFNMA(vi, S7, D2); E2 = VFNMA(vr, S7, E2);
          D3 = VFNMA(vi, S2, D3); E3 = VFNMA(vr, S2, E3);
          D4 = VFMA(vi, S3, D4); E4 = VFMA(vr, S3, E4);
          D5 = VFMA(vi, S8, D5); E5 = VFMA(vr, S8, E5);
          D6 = VFNMA(vi, S4, D6); E6 = VFNMA(vr, S4, E6);
          D7 = VFMA(vi, S1, D7); E7 = VFMA(vr, S1, E7);
          D8 = VFMA(vi, S6, D8); E8 = VFMA(vr, S6, E8);
        }
        { V ar = VL(pr + 816), ai = VL(pi + 816);
          V br = VL(pr + 1496), bi = VL(pi + 1496);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 80, VADD(ar,br)); VS(USCR + 88, VADD(ai,bi));
          D1 = VFMA(vi, S6, D1); E1 = VFMA(vr, S6, E1);
          D2 = VFNMA(vi, S5, D2); E2 = VFNMA(vr, S5, E2);
          D3 = VFMA(vi, S1, D3); E3 = VFMA(vr, S1, E3);
          D4 = VFMA(vi, S7, D4); E4 = VFMA(vr, S7, E4);
          D5 = VFNMA(vi, S4, D5); E5 = VFNMA(vr, S4, E5);
          D6 = VFMA(vi, S2, D6); E6 = VFMA(vr, S2, E6);
          D7 = VFMA(vi, S8, D7); E7 = VFMA(vr, S8, E7);
          D8 = VFNMA(vi, S3, D8); E8 = VFNMA(vr, S3, E8);
        }
        { V ar = VL(pr + 952), ai = VL(pi + 952);
          V br = VL(pr + 1360), bi = VL(pi + 1360);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 96, VADD(ar,br)); VS(USCR + 104, VADD(ai,bi));
          D1 = VFMA(vi, S7, D1); E1 = VFMA(vr, S7, E1);
          D2 = VFNMA(vi, S3, D2); E2 = VFNMA(vr, S3, E2);
          D3 = VFMA(vi, S4, D3); E3 = VFMA(vr, S4, E3);
          D4 = VFNMA(vi, S6, D4); E4 = VFNMA(vr, S6, E4);
          D5 = VFMA(vi, S1, D5); E5 = VFMA(vr, S1, E5);
          D6 = VFMA(vi, S8, D6); E6 = VFMA(vr, S8, E6);
          D7 = VFNMA(vi, S2, D7); E7 = VFNMA(vr, S2, E7);
          D8 = VFMA(vi, S5, D8); E8 = VFMA(vr, S5, E8);
        }
        { V ar = VL(pr + 1088), ai = VL(pi + 1088);
          V br = VL(pr + 1224), bi = VL(pi + 1224);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 112, VADD(ar,br)); VS(USCR + 120, VADD(ai,bi));
          D1 = VFMA(vi, S8, D1); E1 = VFMA(vr, S8, E1);
          D2 = VFNMA(vi, S1, D2); E2 = VFNMA(vr, S1, E2);
          D3 = VFMA(vi, S7, D3); E3 = VFMA(vr, S7, E3);
          D4 = VFNMA(vi, S2, D4); E4 = VFNMA(vr, S2, E4);
          D5 = VFMA(vi, S6, D5); E5 = VFMA(vr, S6, E5);
          D6 = VFNMA(vi, S3, D6); E6 = VFNMA(vr, S3, E6);
          D7 = VFMA(vi, S5, D7); E7 = VFMA(vr, S5, E7);
          D8 = VFNMA(vi, S4, D8); E8 = VFNMA(vr, S4, E8);
        }
        VS(PSCR + 0, D1); VS(PSCR + 8, E1);
        VS(PSCR + 16, D2); VS(PSCR + 24, E2);
        VS(PSCR + 32, D3); VS(PSCR + 40, E3);
        VS(PSCR + 48, D4); VS(PSCR + 56, E4);
        VS(PSCR + 64, D5); VS(PSCR + 72, E5);
        VS(PSCR + 80, D6); VS(PSCR + 88, E6);
        VS(PSCR + 96, D7); VS(PSCR + 104, E7);
        VS(PSCR + 112, D8); VS(PSCR + 120, E8);
        }
        VS(USCR + 384, VL(pr)); VS(USCR + 392, VL(pi));  // save x0 (row 0 gets overwritten)
        {
        V C1 = VSET1(0x1.dd6d000370991p-1);
        V C2 = VSET1(0x1.7a5f6075d4884p-1);
        V C3 = VSET1(0x1.c86fa2b2883cep-2);
        V C4 = VSET1(0x1.79ee63259b75fp-4);
        V C5 = VSET1(-0x1.183b1c61f0d01p-2);
        V C6 = VSET1(-0x1.348c86ed5f1bap-1);
        V C7 = VSET1(-0x1.b34fa910ea3b8p-1);
        V C8 = VSET1(-0x1.f7484007faef3p-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A1 = x0r, B1 = x0i;
        V A2 = x0r, B2 = x0i;
        V A3 = x0r, B3 = x0i;
        V A4 = x0r, B4 = x0i;
        V A5 = x0r, B5 = x0i;
        V A6 = x0r, B6 = x0i;
        V A7 = x0r, B7 = x0i;
        V A8 = x0r, B8 = x0i;
        V s0r = x0r, s0i = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A1 = VFMA(ur, C1, A1); B1 = VFMA(ui, C1, B1);
          A2 = VFMA(ur, C2, A2); B2 = VFMA(ui, C2, B2);
          A3 = VFMA(ur, C3, A3); B3 = VFMA(ui, C3, B3);
          A4 = VFMA(ur, C4, A4); B4 = VFMA(ui, C4, B4);
          A5 = VFMA(ur, C5, A5); B5 = VFMA(ui, C5, B5);
          A6 = VFMA(ur, C6, A6); B6 = VFMA(ui, C6, B6);
          A7 = VFMA(ur, C7, A7); B7 = VFMA(ui, C7, B7);
          A8 = VFMA(ur, C8, A8); B8 = VFMA(ui, C8, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A1 = VFMA(ur, C2, A1); B1 = VFMA(ui, C2, B1);
          A2 = VFMA(ur, C4, A2); B2 = VFMA(ui, C4, B2);
          A3 = VFMA(ur, C6, A3); B3 = VFMA(ui, C6, B3);
          A4 = VFMA(ur, C8, A4); B4 = VFMA(ui, C8, B4);
          A5 = VFMA(ur, C7, A5); B5 = VFMA(ui, C7, B5);
          A6 = VFMA(ur, C5, A6); B6 = VFMA(ui, C5, B6);
          A7 = VFMA(ur, C3, A7); B7 = VFMA(ui, C3, B7);
          A8 = VFMA(ur, C1, A8); B8 = VFMA(ui, C1, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A1 = VFMA(ur, C3, A1); B1 = VFMA(ui, C3, B1);
          A2 = VFMA(ur, C6, A2); B2 = VFMA(ui, C6, B2);
          A3 = VFMA(ur, C8, A3); B3 = VFMA(ui, C8, B3);
          A4 = VFMA(ur, C5, A4); B4 = VFMA(ui, C5, B4);
          A5 = VFMA(ur, C2, A5); B5 = VFMA(ui, C2, B5);
          A6 = VFMA(ur, C1, A6); B6 = VFMA(ui, C1, B6);
          A7 = VFMA(ur, C4, A7); B7 = VFMA(ui, C4, B7);
          A8 = VFMA(ur, C7, A8); B8 = VFMA(ui, C7, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A1 = VFMA(ur, C4, A1); B1 = VFMA(ui, C4, B1);
          A2 = VFMA(ur, C8, A2); B2 = VFMA(ui, C8, B2);
          A3 = VFMA(ur, C5, A3); B3 = VFMA(ui, C5, B3);
          A4 = VFMA(ur, C1, A4); B4 = VFMA(ui, C1, B4);
          A5 = VFMA(ur, C3, A5); B5 = VFMA(ui, C3, B5);
          A6 = VFMA(ur, C7, A6); B6 = VFMA(ui, C7, B6);
          A7 = VFMA(ur, C6, A7); B7 = VFMA(ui, C6, B7);
          A8 = VFMA(ur, C2, A8); B8 = VFMA(ui, C2, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A1 = VFMA(ur, C5, A1); B1 = VFMA(ui, C5, B1);
          A2 = VFMA(ur, C7, A2); B2 = VFMA(ui, C7, B2);
          A3 = VFMA(ur, C2, A3); B3 = VFMA(ui, C2, B3);
          A4 = VFMA(ur, C3, A4); B4 = VFMA(ui, C3, B4);
          A5 = VFMA(ur, C8, A5); B5 = VFMA(ui, C8, B5);
          A6 = VFMA(ur, C4, A6); B6 = VFMA(ui, C4, B6);
          A7 = VFMA(ur, C1, A7); B7 = VFMA(ui, C1, B7);
          A8 = VFMA(ur, C6, A8); B8 = VFMA(ui, C6, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A1 = VFMA(ur, C6, A1); B1 = VFMA(ui, C6, B1);
          A2 = VFMA(ur, C5, A2); B2 = VFMA(ui, C5, B2);
          A3 = VFMA(ur, C1, A3); B3 = VFMA(ui, C1, B3);
          A4 = VFMA(ur, C7, A4); B4 = VFMA(ui, C7, B4);
          A5 = VFMA(ur, C4, A5); B5 = VFMA(ui, C4, B5);
          A6 = VFMA(ur, C2, A6); B6 = VFMA(ui, C2, B6);
          A7 = VFMA(ur, C8, A7); B7 = VFMA(ui, C8, B7);
          A8 = VFMA(ur, C3, A8); B8 = VFMA(ui, C3, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 96), ui = VL(USCR + 104);
          A1 = VFMA(ur, C7, A1); B1 = VFMA(ui, C7, B1);
          A2 = VFMA(ur, C3, A2); B2 = VFMA(ui, C3, B2);
          A3 = VFMA(ur, C4, A3); B3 = VFMA(ui, C4, B3);
          A4 = VFMA(ur, C6, A4); B4 = VFMA(ui, C6, B4);
          A5 = VFMA(ur, C1, A5); B5 = VFMA(ui, C1, B5);
          A6 = VFMA(ur, C8, A6); B6 = VFMA(ui, C8, B6);
          A7 = VFMA(ur, C2, A7); B7 = VFMA(ui, C2, B7);
          A8 = VFMA(ur, C5, A8); B8 = VFMA(ui, C5, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 112), ui = VL(USCR + 120);
          A1 = VFMA(ur, C8, A1); B1 = VFMA(ui, C8, B1);
          A2 = VFMA(ur, C1, A2); B2 = VFMA(ui, C1, B2);
          A3 = VFMA(ur, C7, A3); B3 = VFMA(ui, C7, B3);
          A4 = VFMA(ur, C2, A4); B4 = VFMA(ui, C2, B4);
          A5 = VFMA(ur, C6, A5); B5 = VFMA(ui, C6, B5);
          A6 = VFMA(ur, C3, A6); B6 = VFMA(ui, C3, B6);
          A7 = VFMA(ur, C5, A7); B7 = VFMA(ui, C5, B7);
          A8 = VFMA(ur, C4, A8); B8 = VFMA(ui, C4, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        VS(pr + 0, s0r); VS(pi + 0, s0i);
        { V Dk = VL(PSCR + 0), Ek = VL(PSCR + 8);
          V Xr = VADD(A1, Dk), Xi = VSUB(B1, Ek);
          V Yr = VSUB(A1, Dk), Yi = VADD(B1, Ek);
        VS(pr + 136, Xr); VS(pi + 136, Xi);
        VS(pr + 2176, Yr); VS(pi + 2176, Yi);
        }
        { V Dk = VL(PSCR + 16), Ek = VL(PSCR + 24);
          V Xr = VADD(A2, Dk), Xi = VSUB(B2, Ek);
          V Yr = VSUB(A2, Dk), Yi = VADD(B2, Ek);
        VS(pr + 272, Xr); VS(pi + 272, Xi);
        VS(pr + 2040, Yr); VS(pi + 2040, Yi);
        }
        { V Dk = VL(PSCR + 32), Ek = VL(PSCR + 40);
          V Xr = VADD(A3, Dk), Xi = VSUB(B3, Ek);
          V Yr = VSUB(A3, Dk), Yi = VADD(B3, Ek);
        VS(pr + 408, Xr); VS(pi + 408, Xi);
        VS(pr + 1904, Yr); VS(pi + 1904, Yi);
        }
        { V Dk = VL(PSCR + 48), Ek = VL(PSCR + 56);
          V Xr = VADD(A4, Dk), Xi = VSUB(B4, Ek);
          V Yr = VSUB(A4, Dk), Yi = VADD(B4, Ek);
        VS(pr + 544, Xr); VS(pi + 544, Xi);
        VS(pr + 1768, Yr); VS(pi + 1768, Yi);
        }
        { V Dk = VL(PSCR + 64), Ek = VL(PSCR + 72);
          V Xr = VADD(A5, Dk), Xi = VSUB(B5, Ek);
          V Yr = VSUB(A5, Dk), Yi = VADD(B5, Ek);
        VS(pr + 680, Xr); VS(pi + 680, Xi);
        VS(pr + 1632, Yr); VS(pi + 1632, Yi);
        }
        { V Dk = VL(PSCR + 80), Ek = VL(PSCR + 88);
          V Xr = VADD(A6, Dk), Xi = VSUB(B6, Ek);
          V Yr = VSUB(A6, Dk), Yi = VADD(B6, Ek);
        VS(pr + 816, Xr); VS(pi + 816, Xi);
        VS(pr + 1496, Yr); VS(pi + 1496, Yi);
        }
        { V Dk = VL(PSCR + 96), Ek = VL(PSCR + 104);
          V Xr = VADD(A7, Dk), Xi = VSUB(B7, Ek);
          V Yr = VSUB(A7, Dk), Yi = VADD(B7, Ek);
        VS(pr + 952, Xr); VS(pi + 952, Xi);
        VS(pr + 1360, Yr); VS(pi + 1360, Yi);
        }
        { V Dk = VL(PSCR + 112), Ek = VL(PSCR + 120);
          V Xr = VADD(A8, Dk), Xi = VSUB(B8, Ek);
          V Yr = VSUB(A8, Dk), Yi = VADD(B8, Ek);
        VS(pr + 1088, Xr); VS(pi + 1088, Xi);
        VS(pr + 1224, Yr); VS(pi + 1224, Yi);
        }
        }
    }
}
static __attribute__((noinline)) void p17_xx(double* PR, double* PI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        {
        V S1 = VSET1(0x1.71e955d8e7cdcp-2);
        V S2 = VSET1(0x1.58eea2a9d6da3p-1);
        V S3 = VSET1(0x1.ca52d7c9e640bp-1);
        V S4 = VSET1(0x1.fdd0deb564b22p-1);
        V S5 = VSET1(0x1.ec746923c349fp-1);
        V S6 = VSET1(0x1.9895b6c9a05f7p-1);
        V S7 = VSET1(0x1.0d8884363dd82p-1);
        V S8 = VSET1(0x1.7851aacd6c6b5p-3);
        V D1 = _mm512_setzero_pd(), E1 = _mm512_setzero_pd();
        V D2 = _mm512_setzero_pd(), E2 = _mm512_setzero_pd();
        V D3 = _mm512_setzero_pd(), E3 = _mm512_setzero_pd();
        V D4 = _mm512_setzero_pd(), E4 = _mm512_setzero_pd();
        V D5 = _mm512_setzero_pd(), E5 = _mm512_setzero_pd();
        V D6 = _mm512_setzero_pd(), E6 = _mm512_setzero_pd();
        V D7 = _mm512_setzero_pd(), E7 = _mm512_setzero_pd();
        V D8 = _mm512_setzero_pd(), E8 = _mm512_setzero_pd();
        { V ar = VL(pr + 2312), ai = VL(pi + 2312);
          V br = VL(pr + 36992), bi = VL(pi + 36992);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 0, VADD(ar,br)); VS(USCR + 8, VADD(ai,bi));
          D1 = VFMA(vi, S1, D1); E1 = VFMA(vr, S1, E1);
          D2 = VFMA(vi, S2, D2); E2 = VFMA(vr, S2, E2);
          D3 = VFMA(vi, S3, D3); E3 = VFMA(vr, S3, E3);
          D4 = VFMA(vi, S4, D4); E4 = VFMA(vr, S4, E4);
          D5 = VFMA(vi, S5, D5); E5 = VFMA(vr, S5, E5);
          D6 = VFMA(vi, S6, D6); E6 = VFMA(vr, S6, E6);
          D7 = VFMA(vi, S7, D7); E7 = VFMA(vr, S7, E7);
          D8 = VFMA(vi, S8, D8); E8 = VFMA(vr, S8, E8);
        }
        { V ar = VL(pr + 4624), ai = VL(pi + 4624);
          V br = VL(pr + 34680), bi = VL(pi + 34680);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 16, VADD(ar,br)); VS(USCR + 24, VADD(ai,bi));
          D1 = VFMA(vi, S2, D1); E1 = VFMA(vr, S2, E1);
          D2 = VFMA(vi, S4, D2); E2 = VFMA(vr, S4, E2);
          D3 = VFMA(vi, S6, D3); E3 = VFMA(vr, S6, E3);
          D4 = VFMA(vi, S8, D4); E4 = VFMA(vr, S8, E4);
          D5 = VFNMA(vi, S7, D5); E5 = VFNMA(vr, S7, E5);
          D6 = VFNMA(vi, S5, D6); E6 = VFNMA(vr, S5, E6);
          D7 = VFNMA(vi, S3, D7); E7 = VFNMA(vr, S3, E7);
          D8 = VFNMA(vi, S1, D8); E8 = VFNMA(vr, S1, E8);
        }
        { V ar = VL(pr + 6936), ai = VL(pi + 6936);
          V br = VL(pr + 32368), bi = VL(pi + 32368);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 32, VADD(ar,br)); VS(USCR + 40, VADD(ai,bi));
          D1 = VFMA(vi, S3, D1); E1 = VFMA(vr, S3, E1);
          D2 = VFMA(vi, S6, D2); E2 = VFMA(vr, S6, E2);
          D3 = VFNMA(vi, S8, D3); E3 = VFNMA(vr, S8, E3);
          D4 = VFNMA(vi, S5, D4); E4 = VFNMA(vr, S5, E4);
          D5 = VFNMA(vi, S2, D5); E5 = VFNMA(vr, S2, E5);
          D6 = VFMA(vi, S1, D6); E6 = VFMA(vr, S1, E6);
          D7 = VFMA(vi, S4, D7); E7 = VFMA(vr, S4, E7);
          D8 = VFMA(vi, S7, D8); E8 = VFMA(vr, S7, E8);
        }
        { V ar = VL(pr + 9248), ai = VL(pi + 9248);
          V br = VL(pr + 30056), bi = VL(pi + 30056);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 48, VADD(ar,br)); VS(USCR + 56, VADD(ai,bi));
          D1 = VFMA(vi, S4, D1); E1 = VFMA(vr, S4, E1);
          D2 = VFMA(vi, S8, D2); E2 = VFMA(vr, S8, E2);
          D3 = VFNMA(vi, S5, D3); E3 = VFNMA(vr, S5, E3);
          D4 = VFNMA(vi, S1, D4); E4 = VFNMA(vr, S1, E4);
          D5 = VFMA(vi, S3, D5); E5 = VFMA(vr, S3, E5);
          D6 = VFMA(vi, S7, D6); E6 = VFMA(vr, S7, E6);
          D7 = VFNMA(vi, S6, D7); E7 = VFNMA(vr, S6, E7);
          D8 = VFNMA(vi, S2, D8); E8 = VFNMA(vr, S2, E8);
        }
        { V ar = VL(pr + 11560), ai = VL(pi + 11560);
          V br = VL(pr + 27744), bi = VL(pi + 27744);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 64, VADD(ar,br)); VS(USCR + 72, VADD(ai,bi));
          D1 = VFMA(vi, S5, D1); E1 = VFMA(vr, S5, E1);
          D2 = VFNMA(vi, S7, D2); E2 = VFNMA(vr, S7, E2);
          D3 = VFNMA(vi, S2, D3); E3 = VFNMA(vr, S2, E3);
          D4 = VFMA(vi, S3, D4); E4 = VFMA(vr, S3, E4);
          D5 = VFMA(vi, S8, D5); E5 = VFMA(vr, S8, E5);
          D6 = VFNMA(vi, S4, D6); E6 = VFNMA(vr, S4, E6);
          D7 = VFMA(vi, S1, D7); E7 = VFMA(vr, S1, E7);
          D8 = VFMA(vi, S6, D8); E8 = VFMA(vr, S6, E8);
        }
        { V ar = VL(pr + 13872), ai = VL(pi + 13872);
          V br = VL(pr + 25432), bi = VL(pi + 25432);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 80, VADD(ar,br)); VS(USCR + 88, VADD(ai,bi));
          D1 = VFMA(vi, S6, D1); E1 = VFMA(vr, S6, E1);
          D2 = VFNMA(vi, S5, D2); E2 = VFNMA(vr, S5, E2);
          D3 = VFMA(vi, S1, D3); E3 = VFMA(vr, S1, E3);
          D4 = VFMA(vi, S7, D4); E4 = VFMA(vr, S7, E4);
          D5 = VFNMA(vi, S4, D5); E5 = VFNMA(vr, S4, E5);
          D6 = VFMA(vi, S2, D6); E6 = VFMA(vr, S2, E6);
          D7 = VFMA(vi, S8, D7); E7 = VFMA(vr, S8, E7);
          D8 = VFNMA(vi, S3, D8); E8 = VFNMA(vr, S3, E8);
        }
        { V ar = VL(pr + 16184), ai = VL(pi + 16184);
          V br = VL(pr + 23120), bi = VL(pi + 23120);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 96, VADD(ar,br)); VS(USCR + 104, VADD(ai,bi));
          D1 = VFMA(vi, S7, D1); E1 = VFMA(vr, S7, E1);
          D2 = VFNMA(vi, S3, D2); E2 = VFNMA(vr, S3, E2);
          D3 = VFMA(vi, S4, D3); E3 = VFMA(vr, S4, E3);
          D4 = VFNMA(vi, S6, D4); E4 = VFNMA(vr, S6, E4);
          D5 = VFMA(vi, S1, D5); E5 = VFMA(vr, S1, E5);
          D6 = VFMA(vi, S8, D6); E6 = VFMA(vr, S8, E6);
          D7 = VFNMA(vi, S2, D7); E7 = VFNMA(vr, S2, E7);
          D8 = VFMA(vi, S5, D8); E8 = VFMA(vr, S5, E8);
        }
        { V ar = VL(pr + 18496), ai = VL(pi + 18496);
          V br = VL(pr + 20808), bi = VL(pi + 20808);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 112, VADD(ar,br)); VS(USCR + 120, VADD(ai,bi));
          D1 = VFMA(vi, S8, D1); E1 = VFMA(vr, S8, E1);
          D2 = VFNMA(vi, S1, D2); E2 = VFNMA(vr, S1, E2);
          D3 = VFMA(vi, S7, D3); E3 = VFMA(vr, S7, E3);
          D4 = VFNMA(vi, S2, D4); E4 = VFNMA(vr, S2, E4);
          D5 = VFMA(vi, S6, D5); E5 = VFMA(vr, S6, E5);
          D6 = VFNMA(vi, S3, D6); E6 = VFNMA(vr, S3, E6);
          D7 = VFMA(vi, S5, D7); E7 = VFMA(vr, S5, E7);
          D8 = VFNMA(vi, S4, D8); E8 = VFNMA(vr, S4, E8);
        }
        VS(PSCR + 0, D1); VS(PSCR + 8, E1);
        VS(PSCR + 16, D2); VS(PSCR + 24, E2);
        VS(PSCR + 32, D3); VS(PSCR + 40, E3);
        VS(PSCR + 48, D4); VS(PSCR + 56, E4);
        VS(PSCR + 64, D5); VS(PSCR + 72, E5);
        VS(PSCR + 80, D6); VS(PSCR + 88, E6);
        VS(PSCR + 96, D7); VS(PSCR + 104, E7);
        VS(PSCR + 112, D8); VS(PSCR + 120, E8);
        }
        VS(USCR + 384, VL(pr)); VS(USCR + 392, VL(pi));  // save x0 (row 0 gets overwritten)
        {
        V C1 = VSET1(0x1.dd6d000370991p-1);
        V C2 = VSET1(0x1.7a5f6075d4884p-1);
        V C3 = VSET1(0x1.c86fa2b2883cep-2);
        V C4 = VSET1(0x1.79ee63259b75fp-4);
        V C5 = VSET1(-0x1.183b1c61f0d01p-2);
        V C6 = VSET1(-0x1.348c86ed5f1bap-1);
        V C7 = VSET1(-0x1.b34fa910ea3b8p-1);
        V C8 = VSET1(-0x1.f7484007faef3p-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A1 = x0r, B1 = x0i;
        V A2 = x0r, B2 = x0i;
        V A3 = x0r, B3 = x0i;
        V A4 = x0r, B4 = x0i;
        V A5 = x0r, B5 = x0i;
        V A6 = x0r, B6 = x0i;
        V A7 = x0r, B7 = x0i;
        V A8 = x0r, B8 = x0i;
        V s0r = x0r, s0i = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A1 = VFMA(ur, C1, A1); B1 = VFMA(ui, C1, B1);
          A2 = VFMA(ur, C2, A2); B2 = VFMA(ui, C2, B2);
          A3 = VFMA(ur, C3, A3); B3 = VFMA(ui, C3, B3);
          A4 = VFMA(ur, C4, A4); B4 = VFMA(ui, C4, B4);
          A5 = VFMA(ur, C5, A5); B5 = VFMA(ui, C5, B5);
          A6 = VFMA(ur, C6, A6); B6 = VFMA(ui, C6, B6);
          A7 = VFMA(ur, C7, A7); B7 = VFMA(ui, C7, B7);
          A8 = VFMA(ur, C8, A8); B8 = VFMA(ui, C8, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A1 = VFMA(ur, C2, A1); B1 = VFMA(ui, C2, B1);
          A2 = VFMA(ur, C4, A2); B2 = VFMA(ui, C4, B2);
          A3 = VFMA(ur, C6, A3); B3 = VFMA(ui, C6, B3);
          A4 = VFMA(ur, C8, A4); B4 = VFMA(ui, C8, B4);
          A5 = VFMA(ur, C7, A5); B5 = VFMA(ui, C7, B5);
          A6 = VFMA(ur, C5, A6); B6 = VFMA(ui, C5, B6);
          A7 = VFMA(ur, C3, A7); B7 = VFMA(ui, C3, B7);
          A8 = VFMA(ur, C1, A8); B8 = VFMA(ui, C1, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A1 = VFMA(ur, C3, A1); B1 = VFMA(ui, C3, B1);
          A2 = VFMA(ur, C6, A2); B2 = VFMA(ui, C6, B2);
          A3 = VFMA(ur, C8, A3); B3 = VFMA(ui, C8, B3);
          A4 = VFMA(ur, C5, A4); B4 = VFMA(ui, C5, B4);
          A5 = VFMA(ur, C2, A5); B5 = VFMA(ui, C2, B5);
          A6 = VFMA(ur, C1, A6); B6 = VFMA(ui, C1, B6);
          A7 = VFMA(ur, C4, A7); B7 = VFMA(ui, C4, B7);
          A8 = VFMA(ur, C7, A8); B8 = VFMA(ui, C7, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A1 = VFMA(ur, C4, A1); B1 = VFMA(ui, C4, B1);
          A2 = VFMA(ur, C8, A2); B2 = VFMA(ui, C8, B2);
          A3 = VFMA(ur, C5, A3); B3 = VFMA(ui, C5, B3);
          A4 = VFMA(ur, C1, A4); B4 = VFMA(ui, C1, B4);
          A5 = VFMA(ur, C3, A5); B5 = VFMA(ui, C3, B5);
          A6 = VFMA(ur, C7, A6); B6 = VFMA(ui, C7, B6);
          A7 = VFMA(ur, C6, A7); B7 = VFMA(ui, C6, B7);
          A8 = VFMA(ur, C2, A8); B8 = VFMA(ui, C2, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A1 = VFMA(ur, C5, A1); B1 = VFMA(ui, C5, B1);
          A2 = VFMA(ur, C7, A2); B2 = VFMA(ui, C7, B2);
          A3 = VFMA(ur, C2, A3); B3 = VFMA(ui, C2, B3);
          A4 = VFMA(ur, C3, A4); B4 = VFMA(ui, C3, B4);
          A5 = VFMA(ur, C8, A5); B5 = VFMA(ui, C8, B5);
          A6 = VFMA(ur, C4, A6); B6 = VFMA(ui, C4, B6);
          A7 = VFMA(ur, C1, A7); B7 = VFMA(ui, C1, B7);
          A8 = VFMA(ur, C6, A8); B8 = VFMA(ui, C6, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A1 = VFMA(ur, C6, A1); B1 = VFMA(ui, C6, B1);
          A2 = VFMA(ur, C5, A2); B2 = VFMA(ui, C5, B2);
          A3 = VFMA(ur, C1, A3); B3 = VFMA(ui, C1, B3);
          A4 = VFMA(ur, C7, A4); B4 = VFMA(ui, C7, B4);
          A5 = VFMA(ur, C4, A5); B5 = VFMA(ui, C4, B5);
          A6 = VFMA(ur, C2, A6); B6 = VFMA(ui, C2, B6);
          A7 = VFMA(ur, C8, A7); B7 = VFMA(ui, C8, B7);
          A8 = VFMA(ur, C3, A8); B8 = VFMA(ui, C3, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 96), ui = VL(USCR + 104);
          A1 = VFMA(ur, C7, A1); B1 = VFMA(ui, C7, B1);
          A2 = VFMA(ur, C3, A2); B2 = VFMA(ui, C3, B2);
          A3 = VFMA(ur, C4, A3); B3 = VFMA(ui, C4, B3);
          A4 = VFMA(ur, C6, A4); B4 = VFMA(ui, C6, B4);
          A5 = VFMA(ur, C1, A5); B5 = VFMA(ui, C1, B5);
          A6 = VFMA(ur, C8, A6); B6 = VFMA(ui, C8, B6);
          A7 = VFMA(ur, C2, A7); B7 = VFMA(ui, C2, B7);
          A8 = VFMA(ur, C5, A8); B8 = VFMA(ui, C5, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 112), ui = VL(USCR + 120);
          A1 = VFMA(ur, C8, A1); B1 = VFMA(ui, C8, B1);
          A2 = VFMA(ur, C1, A2); B2 = VFMA(ui, C1, B2);
          A3 = VFMA(ur, C7, A3); B3 = VFMA(ui, C7, B3);
          A4 = VFMA(ur, C2, A4); B4 = VFMA(ui, C2, B4);
          A5 = VFMA(ur, C6, A5); B5 = VFMA(ui, C6, B5);
          A6 = VFMA(ur, C3, A6); B6 = VFMA(ui, C3, B6);
          A7 = VFMA(ur, C5, A7); B7 = VFMA(ui, C5, B7);
          A8 = VFMA(ur, C4, A8); B8 = VFMA(ui, C4, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        VS(pr + 0, s0r); VS(pi + 0, s0i);
        { V Dk = VL(PSCR + 0), Ek = VL(PSCR + 8);
          V Xr = VADD(A1, Dk), Xi = VSUB(B1, Ek);
          V Yr = VSUB(A1, Dk), Yi = VADD(B1, Ek);
        VS(pr + 2312, Xr); VS(pi + 2312, Xi);
        VS(pr + 36992, Yr); VS(pi + 36992, Yi);
        }
        { V Dk = VL(PSCR + 16), Ek = VL(PSCR + 24);
          V Xr = VADD(A2, Dk), Xi = VSUB(B2, Ek);
          V Yr = VSUB(A2, Dk), Yi = VADD(B2, Ek);
        VS(pr + 4624, Xr); VS(pi + 4624, Xi);
        VS(pr + 34680, Yr); VS(pi + 34680, Yi);
        }
        { V Dk = VL(PSCR + 32), Ek = VL(PSCR + 40);
          V Xr = VADD(A3, Dk), Xi = VSUB(B3, Ek);
          V Yr = VSUB(A3, Dk), Yi = VADD(B3, Ek);
        VS(pr + 6936, Xr); VS(pi + 6936, Xi);
        VS(pr + 32368, Yr); VS(pi + 32368, Yi);
        }
        { V Dk = VL(PSCR + 48), Ek = VL(PSCR + 56);
          V Xr = VADD(A4, Dk), Xi = VSUB(B4, Ek);
          V Yr = VSUB(A4, Dk), Yi = VADD(B4, Ek);
        VS(pr + 9248, Xr); VS(pi + 9248, Xi);
        VS(pr + 30056, Yr); VS(pi + 30056, Yi);
        }
        { V Dk = VL(PSCR + 64), Ek = VL(PSCR + 72);
          V Xr = VADD(A5, Dk), Xi = VSUB(B5, Ek);
          V Yr = VSUB(A5, Dk), Yi = VADD(B5, Ek);
        VS(pr + 11560, Xr); VS(pi + 11560, Xi);
        VS(pr + 27744, Yr); VS(pi + 27744, Yi);
        }
        { V Dk = VL(PSCR + 80), Ek = VL(PSCR + 88);
          V Xr = VADD(A6, Dk), Xi = VSUB(B6, Ek);
          V Yr = VSUB(A6, Dk), Yi = VADD(B6, Ek);
        VS(pr + 13872, Xr); VS(pi + 13872, Xi);
        VS(pr + 25432, Yr); VS(pi + 25432, Yi);
        }
        { V Dk = VL(PSCR + 96), Ek = VL(PSCR + 104);
          V Xr = VADD(A7, Dk), Xi = VSUB(B7, Ek);
          V Yr = VSUB(A7, Dk), Yi = VADD(B7, Ek);
        VS(pr + 16184, Xr); VS(pi + 16184, Xi);
        VS(pr + 23120, Yr); VS(pi + 23120, Yi);
        }
        { V Dk = VL(PSCR + 112), Ek = VL(PSCR + 120);
          V Xr = VADD(A8, Dk), Xi = VSUB(B8, Ek);
          V Yr = VSUB(A8, Dk), Yi = VADD(B8, Ek);
        VS(pr + 18496, Xr); VS(pi + 18496, Xi);
        VS(pr + 20808, Yr); VS(pi + 20808, Yi);
        }
        }
    }
}
static __attribute__((noinline)) void p17_xxm(double* PR, double* PI, const double* CR, const double* CI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        const double* cr = CR + q_*pstep; const double* ci = CI + q_*pstep;
        {
        V S1 = VSET1(0x1.71e955d8e7cdcp-2);
        V S2 = VSET1(0x1.58eea2a9d6da3p-1);
        V S3 = VSET1(0x1.ca52d7c9e640bp-1);
        V S4 = VSET1(0x1.fdd0deb564b22p-1);
        V S5 = VSET1(0x1.ec746923c349fp-1);
        V S6 = VSET1(0x1.9895b6c9a05f7p-1);
        V S7 = VSET1(0x1.0d8884363dd82p-1);
        V S8 = VSET1(0x1.7851aacd6c6b5p-3);
        V D1 = _mm512_setzero_pd(), E1 = _mm512_setzero_pd();
        V D2 = _mm512_setzero_pd(), E2 = _mm512_setzero_pd();
        V D3 = _mm512_setzero_pd(), E3 = _mm512_setzero_pd();
        V D4 = _mm512_setzero_pd(), E4 = _mm512_setzero_pd();
        V D5 = _mm512_setzero_pd(), E5 = _mm512_setzero_pd();
        V D6 = _mm512_setzero_pd(), E6 = _mm512_setzero_pd();
        V D7 = _mm512_setzero_pd(), E7 = _mm512_setzero_pd();
        V D8 = _mm512_setzero_pd(), E8 = _mm512_setzero_pd();
        { V ar = VL(pr + 2312), ai = VL(pi + 2312);
          V br = VL(pr + 36992), bi = VL(pi + 36992);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 0, VADD(ar,br)); VS(USCR + 8, VADD(ai,bi));
          D1 = VFMA(vi, S1, D1); E1 = VFMA(vr, S1, E1);
          D2 = VFMA(vi, S2, D2); E2 = VFMA(vr, S2, E2);
          D3 = VFMA(vi, S3, D3); E3 = VFMA(vr, S3, E3);
          D4 = VFMA(vi, S4, D4); E4 = VFMA(vr, S4, E4);
          D5 = VFMA(vi, S5, D5); E5 = VFMA(vr, S5, E5);
          D6 = VFMA(vi, S6, D6); E6 = VFMA(vr, S6, E6);
          D7 = VFMA(vi, S7, D7); E7 = VFMA(vr, S7, E7);
          D8 = VFMA(vi, S8, D8); E8 = VFMA(vr, S8, E8);
        }
        { V ar = VL(pr + 4624), ai = VL(pi + 4624);
          V br = VL(pr + 34680), bi = VL(pi + 34680);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 16, VADD(ar,br)); VS(USCR + 24, VADD(ai,bi));
          D1 = VFMA(vi, S2, D1); E1 = VFMA(vr, S2, E1);
          D2 = VFMA(vi, S4, D2); E2 = VFMA(vr, S4, E2);
          D3 = VFMA(vi, S6, D3); E3 = VFMA(vr, S6, E3);
          D4 = VFMA(vi, S8, D4); E4 = VFMA(vr, S8, E4);
          D5 = VFNMA(vi, S7, D5); E5 = VFNMA(vr, S7, E5);
          D6 = VFNMA(vi, S5, D6); E6 = VFNMA(vr, S5, E6);
          D7 = VFNMA(vi, S3, D7); E7 = VFNMA(vr, S3, E7);
          D8 = VFNMA(vi, S1, D8); E8 = VFNMA(vr, S1, E8);
        }
        { V ar = VL(pr + 6936), ai = VL(pi + 6936);
          V br = VL(pr + 32368), bi = VL(pi + 32368);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 32, VADD(ar,br)); VS(USCR + 40, VADD(ai,bi));
          D1 = VFMA(vi, S3, D1); E1 = VFMA(vr, S3, E1);
          D2 = VFMA(vi, S6, D2); E2 = VFMA(vr, S6, E2);
          D3 = VFNMA(vi, S8, D3); E3 = VFNMA(vr, S8, E3);
          D4 = VFNMA(vi, S5, D4); E4 = VFNMA(vr, S5, E4);
          D5 = VFNMA(vi, S2, D5); E5 = VFNMA(vr, S2, E5);
          D6 = VFMA(vi, S1, D6); E6 = VFMA(vr, S1, E6);
          D7 = VFMA(vi, S4, D7); E7 = VFMA(vr, S4, E7);
          D8 = VFMA(vi, S7, D8); E8 = VFMA(vr, S7, E8);
        }
        { V ar = VL(pr + 9248), ai = VL(pi + 9248);
          V br = VL(pr + 30056), bi = VL(pi + 30056);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 48, VADD(ar,br)); VS(USCR + 56, VADD(ai,bi));
          D1 = VFMA(vi, S4, D1); E1 = VFMA(vr, S4, E1);
          D2 = VFMA(vi, S8, D2); E2 = VFMA(vr, S8, E2);
          D3 = VFNMA(vi, S5, D3); E3 = VFNMA(vr, S5, E3);
          D4 = VFNMA(vi, S1, D4); E4 = VFNMA(vr, S1, E4);
          D5 = VFMA(vi, S3, D5); E5 = VFMA(vr, S3, E5);
          D6 = VFMA(vi, S7, D6); E6 = VFMA(vr, S7, E6);
          D7 = VFNMA(vi, S6, D7); E7 = VFNMA(vr, S6, E7);
          D8 = VFNMA(vi, S2, D8); E8 = VFNMA(vr, S2, E8);
        }
        { V ar = VL(pr + 11560), ai = VL(pi + 11560);
          V br = VL(pr + 27744), bi = VL(pi + 27744);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 64, VADD(ar,br)); VS(USCR + 72, VADD(ai,bi));
          D1 = VFMA(vi, S5, D1); E1 = VFMA(vr, S5, E1);
          D2 = VFNMA(vi, S7, D2); E2 = VFNMA(vr, S7, E2);
          D3 = VFNMA(vi, S2, D3); E3 = VFNMA(vr, S2, E3);
          D4 = VFMA(vi, S3, D4); E4 = VFMA(vr, S3, E4);
          D5 = VFMA(vi, S8, D5); E5 = VFMA(vr, S8, E5);
          D6 = VFNMA(vi, S4, D6); E6 = VFNMA(vr, S4, E6);
          D7 = VFMA(vi, S1, D7); E7 = VFMA(vr, S1, E7);
          D8 = VFMA(vi, S6, D8); E8 = VFMA(vr, S6, E8);
        }
        { V ar = VL(pr + 13872), ai = VL(pi + 13872);
          V br = VL(pr + 25432), bi = VL(pi + 25432);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 80, VADD(ar,br)); VS(USCR + 88, VADD(ai,bi));
          D1 = VFMA(vi, S6, D1); E1 = VFMA(vr, S6, E1);
          D2 = VFNMA(vi, S5, D2); E2 = VFNMA(vr, S5, E2);
          D3 = VFMA(vi, S1, D3); E3 = VFMA(vr, S1, E3);
          D4 = VFMA(vi, S7, D4); E4 = VFMA(vr, S7, E4);
          D5 = VFNMA(vi, S4, D5); E5 = VFNMA(vr, S4, E5);
          D6 = VFMA(vi, S2, D6); E6 = VFMA(vr, S2, E6);
          D7 = VFMA(vi, S8, D7); E7 = VFMA(vr, S8, E7);
          D8 = VFNMA(vi, S3, D8); E8 = VFNMA(vr, S3, E8);
        }
        { V ar = VL(pr + 16184), ai = VL(pi + 16184);
          V br = VL(pr + 23120), bi = VL(pi + 23120);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 96, VADD(ar,br)); VS(USCR + 104, VADD(ai,bi));
          D1 = VFMA(vi, S7, D1); E1 = VFMA(vr, S7, E1);
          D2 = VFNMA(vi, S3, D2); E2 = VFNMA(vr, S3, E2);
          D3 = VFMA(vi, S4, D3); E3 = VFMA(vr, S4, E3);
          D4 = VFNMA(vi, S6, D4); E4 = VFNMA(vr, S6, E4);
          D5 = VFMA(vi, S1, D5); E5 = VFMA(vr, S1, E5);
          D6 = VFMA(vi, S8, D6); E6 = VFMA(vr, S8, E6);
          D7 = VFNMA(vi, S2, D7); E7 = VFNMA(vr, S2, E7);
          D8 = VFMA(vi, S5, D8); E8 = VFMA(vr, S5, E8);
        }
        { V ar = VL(pr + 18496), ai = VL(pi + 18496);
          V br = VL(pr + 20808), bi = VL(pi + 20808);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 112, VADD(ar,br)); VS(USCR + 120, VADD(ai,bi));
          D1 = VFMA(vi, S8, D1); E1 = VFMA(vr, S8, E1);
          D2 = VFNMA(vi, S1, D2); E2 = VFNMA(vr, S1, E2);
          D3 = VFMA(vi, S7, D3); E3 = VFMA(vr, S7, E3);
          D4 = VFNMA(vi, S2, D4); E4 = VFNMA(vr, S2, E4);
          D5 = VFMA(vi, S6, D5); E5 = VFMA(vr, S6, E5);
          D6 = VFNMA(vi, S3, D6); E6 = VFNMA(vr, S3, E6);
          D7 = VFMA(vi, S5, D7); E7 = VFMA(vr, S5, E7);
          D8 = VFNMA(vi, S4, D8); E8 = VFNMA(vr, S4, E8);
        }
        VS(PSCR + 0, D1); VS(PSCR + 8, E1);
        VS(PSCR + 16, D2); VS(PSCR + 24, E2);
        VS(PSCR + 32, D3); VS(PSCR + 40, E3);
        VS(PSCR + 48, D4); VS(PSCR + 56, E4);
        VS(PSCR + 64, D5); VS(PSCR + 72, E5);
        VS(PSCR + 80, D6); VS(PSCR + 88, E6);
        VS(PSCR + 96, D7); VS(PSCR + 104, E7);
        VS(PSCR + 112, D8); VS(PSCR + 120, E8);
        }
        VS(USCR + 384, VL(pr)); VS(USCR + 392, VL(pi));  // save x0 (row 0 gets overwritten)
        {
        V C1 = VSET1(0x1.dd6d000370991p-1);
        V C2 = VSET1(0x1.7a5f6075d4884p-1);
        V C3 = VSET1(0x1.c86fa2b2883cep-2);
        V C4 = VSET1(0x1.79ee63259b75fp-4);
        V C5 = VSET1(-0x1.183b1c61f0d01p-2);
        V C6 = VSET1(-0x1.348c86ed5f1bap-1);
        V C7 = VSET1(-0x1.b34fa910ea3b8p-1);
        V C8 = VSET1(-0x1.f7484007faef3p-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A1 = x0r, B1 = x0i;
        V A2 = x0r, B2 = x0i;
        V A3 = x0r, B3 = x0i;
        V A4 = x0r, B4 = x0i;
        V A5 = x0r, B5 = x0i;
        V A6 = x0r, B6 = x0i;
        V A7 = x0r, B7 = x0i;
        V A8 = x0r, B8 = x0i;
        V s0r = x0r, s0i = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A1 = VFMA(ur, C1, A1); B1 = VFMA(ui, C1, B1);
          A2 = VFMA(ur, C2, A2); B2 = VFMA(ui, C2, B2);
          A3 = VFMA(ur, C3, A3); B3 = VFMA(ui, C3, B3);
          A4 = VFMA(ur, C4, A4); B4 = VFMA(ui, C4, B4);
          A5 = VFMA(ur, C5, A5); B5 = VFMA(ui, C5, B5);
          A6 = VFMA(ur, C6, A6); B6 = VFMA(ui, C6, B6);
          A7 = VFMA(ur, C7, A7); B7 = VFMA(ui, C7, B7);
          A8 = VFMA(ur, C8, A8); B8 = VFMA(ui, C8, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A1 = VFMA(ur, C2, A1); B1 = VFMA(ui, C2, B1);
          A2 = VFMA(ur, C4, A2); B2 = VFMA(ui, C4, B2);
          A3 = VFMA(ur, C6, A3); B3 = VFMA(ui, C6, B3);
          A4 = VFMA(ur, C8, A4); B4 = VFMA(ui, C8, B4);
          A5 = VFMA(ur, C7, A5); B5 = VFMA(ui, C7, B5);
          A6 = VFMA(ur, C5, A6); B6 = VFMA(ui, C5, B6);
          A7 = VFMA(ur, C3, A7); B7 = VFMA(ui, C3, B7);
          A8 = VFMA(ur, C1, A8); B8 = VFMA(ui, C1, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A1 = VFMA(ur, C3, A1); B1 = VFMA(ui, C3, B1);
          A2 = VFMA(ur, C6, A2); B2 = VFMA(ui, C6, B2);
          A3 = VFMA(ur, C8, A3); B3 = VFMA(ui, C8, B3);
          A4 = VFMA(ur, C5, A4); B4 = VFMA(ui, C5, B4);
          A5 = VFMA(ur, C2, A5); B5 = VFMA(ui, C2, B5);
          A6 = VFMA(ur, C1, A6); B6 = VFMA(ui, C1, B6);
          A7 = VFMA(ur, C4, A7); B7 = VFMA(ui, C4, B7);
          A8 = VFMA(ur, C7, A8); B8 = VFMA(ui, C7, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A1 = VFMA(ur, C4, A1); B1 = VFMA(ui, C4, B1);
          A2 = VFMA(ur, C8, A2); B2 = VFMA(ui, C8, B2);
          A3 = VFMA(ur, C5, A3); B3 = VFMA(ui, C5, B3);
          A4 = VFMA(ur, C1, A4); B4 = VFMA(ui, C1, B4);
          A5 = VFMA(ur, C3, A5); B5 = VFMA(ui, C3, B5);
          A6 = VFMA(ur, C7, A6); B6 = VFMA(ui, C7, B6);
          A7 = VFMA(ur, C6, A7); B7 = VFMA(ui, C6, B7);
          A8 = VFMA(ur, C2, A8); B8 = VFMA(ui, C2, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A1 = VFMA(ur, C5, A1); B1 = VFMA(ui, C5, B1);
          A2 = VFMA(ur, C7, A2); B2 = VFMA(ui, C7, B2);
          A3 = VFMA(ur, C2, A3); B3 = VFMA(ui, C2, B3);
          A4 = VFMA(ur, C3, A4); B4 = VFMA(ui, C3, B4);
          A5 = VFMA(ur, C8, A5); B5 = VFMA(ui, C8, B5);
          A6 = VFMA(ur, C4, A6); B6 = VFMA(ui, C4, B6);
          A7 = VFMA(ur, C1, A7); B7 = VFMA(ui, C1, B7);
          A8 = VFMA(ur, C6, A8); B8 = VFMA(ui, C6, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A1 = VFMA(ur, C6, A1); B1 = VFMA(ui, C6, B1);
          A2 = VFMA(ur, C5, A2); B2 = VFMA(ui, C5, B2);
          A3 = VFMA(ur, C1, A3); B3 = VFMA(ui, C1, B3);
          A4 = VFMA(ur, C7, A4); B4 = VFMA(ui, C7, B4);
          A5 = VFMA(ur, C4, A5); B5 = VFMA(ui, C4, B5);
          A6 = VFMA(ur, C2, A6); B6 = VFMA(ui, C2, B6);
          A7 = VFMA(ur, C8, A7); B7 = VFMA(ui, C8, B7);
          A8 = VFMA(ur, C3, A8); B8 = VFMA(ui, C3, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 96), ui = VL(USCR + 104);
          A1 = VFMA(ur, C7, A1); B1 = VFMA(ui, C7, B1);
          A2 = VFMA(ur, C3, A2); B2 = VFMA(ui, C3, B2);
          A3 = VFMA(ur, C4, A3); B3 = VFMA(ui, C4, B3);
          A4 = VFMA(ur, C6, A4); B4 = VFMA(ui, C6, B4);
          A5 = VFMA(ur, C1, A5); B5 = VFMA(ui, C1, B5);
          A6 = VFMA(ur, C8, A6); B6 = VFMA(ui, C8, B6);
          A7 = VFMA(ur, C2, A7); B7 = VFMA(ui, C2, B7);
          A8 = VFMA(ur, C5, A8); B8 = VFMA(ui, C5, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 112), ui = VL(USCR + 120);
          A1 = VFMA(ur, C8, A1); B1 = VFMA(ui, C8, B1);
          A2 = VFMA(ur, C1, A2); B2 = VFMA(ui, C1, B2);
          A3 = VFMA(ur, C7, A3); B3 = VFMA(ui, C7, B3);
          A4 = VFMA(ur, C2, A4); B4 = VFMA(ui, C2, B4);
          A5 = VFMA(ur, C6, A5); B5 = VFMA(ui, C6, B5);
          A6 = VFMA(ur, C3, A6); B6 = VFMA(ui, C3, B6);
          A7 = VFMA(ur, C5, A7); B7 = VFMA(ui, C5, B7);
          A8 = VFMA(ur, C4, A8); B8 = VFMA(ui, C4, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V zr_ = VADD(s0r, VL(cr + 0)), zi_ = VADD(s0i, VL(ci + 0));
          MAP2(zr_, zi_);
          VS(pr + 0, zr_); VS(pi + 0, zi_); }
        { V Dk = VL(PSCR + 0), Ek = VL(PSCR + 8);
          V Xr = VADD(A1, Dk), Xi = VSUB(B1, Ek);
          V Yr = VSUB(A1, Dk), Yi = VADD(B1, Ek);
        { V zr_ = VADD(Xr, VL(cr + 2312)), zi_ = VADD(Xi, VL(ci + 2312));
          MAP2(zr_, zi_);
          VS(pr + 2312, zr_); VS(pi + 2312, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 36992)), zi_ = VADD(Yi, VL(ci + 36992));
          MAP2(zr_, zi_);
          VS(pr + 36992, zr_); VS(pi + 36992, zi_); }
        }
        { V Dk = VL(PSCR + 16), Ek = VL(PSCR + 24);
          V Xr = VADD(A2, Dk), Xi = VSUB(B2, Ek);
          V Yr = VSUB(A2, Dk), Yi = VADD(B2, Ek);
        { V zr_ = VADD(Xr, VL(cr + 4624)), zi_ = VADD(Xi, VL(ci + 4624));
          MAP2(zr_, zi_);
          VS(pr + 4624, zr_); VS(pi + 4624, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 34680)), zi_ = VADD(Yi, VL(ci + 34680));
          MAP2(zr_, zi_);
          VS(pr + 34680, zr_); VS(pi + 34680, zi_); }
        }
        { V Dk = VL(PSCR + 32), Ek = VL(PSCR + 40);
          V Xr = VADD(A3, Dk), Xi = VSUB(B3, Ek);
          V Yr = VSUB(A3, Dk), Yi = VADD(B3, Ek);
        { V zr_ = VADD(Xr, VL(cr + 6936)), zi_ = VADD(Xi, VL(ci + 6936));
          MAP2(zr_, zi_);
          VS(pr + 6936, zr_); VS(pi + 6936, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 32368)), zi_ = VADD(Yi, VL(ci + 32368));
          MAP2(zr_, zi_);
          VS(pr + 32368, zr_); VS(pi + 32368, zi_); }
        }
        { V Dk = VL(PSCR + 48), Ek = VL(PSCR + 56);
          V Xr = VADD(A4, Dk), Xi = VSUB(B4, Ek);
          V Yr = VSUB(A4, Dk), Yi = VADD(B4, Ek);
        { V zr_ = VADD(Xr, VL(cr + 9248)), zi_ = VADD(Xi, VL(ci + 9248));
          MAP2(zr_, zi_);
          VS(pr + 9248, zr_); VS(pi + 9248, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 30056)), zi_ = VADD(Yi, VL(ci + 30056));
          MAP2(zr_, zi_);
          VS(pr + 30056, zr_); VS(pi + 30056, zi_); }
        }
        { V Dk = VL(PSCR + 64), Ek = VL(PSCR + 72);
          V Xr = VADD(A5, Dk), Xi = VSUB(B5, Ek);
          V Yr = VSUB(A5, Dk), Yi = VADD(B5, Ek);
        { V zr_ = VADD(Xr, VL(cr + 11560)), zi_ = VADD(Xi, VL(ci + 11560));
          MAP2(zr_, zi_);
          VS(pr + 11560, zr_); VS(pi + 11560, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 27744)), zi_ = VADD(Yi, VL(ci + 27744));
          MAP2(zr_, zi_);
          VS(pr + 27744, zr_); VS(pi + 27744, zi_); }
        }
        { V Dk = VL(PSCR + 80), Ek = VL(PSCR + 88);
          V Xr = VADD(A6, Dk), Xi = VSUB(B6, Ek);
          V Yr = VSUB(A6, Dk), Yi = VADD(B6, Ek);
        { V zr_ = VADD(Xr, VL(cr + 13872)), zi_ = VADD(Xi, VL(ci + 13872));
          MAP2(zr_, zi_);
          VS(pr + 13872, zr_); VS(pi + 13872, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 25432)), zi_ = VADD(Yi, VL(ci + 25432));
          MAP2(zr_, zi_);
          VS(pr + 25432, zr_); VS(pi + 25432, zi_); }
        }
        { V Dk = VL(PSCR + 96), Ek = VL(PSCR + 104);
          V Xr = VADD(A7, Dk), Xi = VSUB(B7, Ek);
          V Yr = VSUB(A7, Dk), Yi = VADD(B7, Ek);
        { V zr_ = VADD(Xr, VL(cr + 16184)), zi_ = VADD(Xi, VL(ci + 16184));
          MAP2(zr_, zi_);
          VS(pr + 16184, zr_); VS(pi + 16184, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 23120)), zi_ = VADD(Yi, VL(ci + 23120));
          MAP2(zr_, zi_);
          VS(pr + 23120, zr_); VS(pi + 23120, zi_); }
        }
        { V Dk = VL(PSCR + 112), Ek = VL(PSCR + 120);
          V Xr = VADD(A8, Dk), Xi = VSUB(B8, Ek);
          V Yr = VSUB(A8, Dk), Yi = VADD(B8, Ek);
        { V zr_ = VADD(Xr, VL(cr + 18496)), zi_ = VADD(Xi, VL(ci + 18496));
          MAP2(zr_, zi_);
          VS(pr + 18496, zr_); VS(pi + 18496, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 20808)), zi_ = VADD(Yi, VL(ci + 20808));
          MAP2(zr_, zi_);
          VS(pr + 20808, zr_); VS(pi + 20808, zi_); }
        }
        }
    }
}
static __attribute__((noinline)) void p17_yym(double* PR, double* PI, const double* CR, const double* CI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        const double* cr = CR + q_*pstep; const double* ci = CI + q_*pstep;
        {
        V S1 = VSET1(0x1.71e955d8e7cdcp-2);
        V S2 = VSET1(0x1.58eea2a9d6da3p-1);
        V S3 = VSET1(0x1.ca52d7c9e640bp-1);
        V S4 = VSET1(0x1.fdd0deb564b22p-1);
        V S5 = VSET1(0x1.ec746923c349fp-1);
        V S6 = VSET1(0x1.9895b6c9a05f7p-1);
        V S7 = VSET1(0x1.0d8884363dd82p-1);
        V S8 = VSET1(0x1.7851aacd6c6b5p-3);
        V D1 = _mm512_setzero_pd(), E1 = _mm512_setzero_pd();
        V D2 = _mm512_setzero_pd(), E2 = _mm512_setzero_pd();
        V D3 = _mm512_setzero_pd(), E3 = _mm512_setzero_pd();
        V D4 = _mm512_setzero_pd(), E4 = _mm512_setzero_pd();
        V D5 = _mm512_setzero_pd(), E5 = _mm512_setzero_pd();
        V D6 = _mm512_setzero_pd(), E6 = _mm512_setzero_pd();
        V D7 = _mm512_setzero_pd(), E7 = _mm512_setzero_pd();
        V D8 = _mm512_setzero_pd(), E8 = _mm512_setzero_pd();
        { V ar = VL(pr + 136), ai = VL(pi + 136);
          V br = VL(pr + 2176), bi = VL(pi + 2176);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 0, VADD(ar,br)); VS(USCR + 8, VADD(ai,bi));
          D1 = VFMA(vi, S1, D1); E1 = VFMA(vr, S1, E1);
          D2 = VFMA(vi, S2, D2); E2 = VFMA(vr, S2, E2);
          D3 = VFMA(vi, S3, D3); E3 = VFMA(vr, S3, E3);
          D4 = VFMA(vi, S4, D4); E4 = VFMA(vr, S4, E4);
          D5 = VFMA(vi, S5, D5); E5 = VFMA(vr, S5, E5);
          D6 = VFMA(vi, S6, D6); E6 = VFMA(vr, S6, E6);
          D7 = VFMA(vi, S7, D7); E7 = VFMA(vr, S7, E7);
          D8 = VFMA(vi, S8, D8); E8 = VFMA(vr, S8, E8);
        }
        { V ar = VL(pr + 272), ai = VL(pi + 272);
          V br = VL(pr + 2040), bi = VL(pi + 2040);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 16, VADD(ar,br)); VS(USCR + 24, VADD(ai,bi));
          D1 = VFMA(vi, S2, D1); E1 = VFMA(vr, S2, E1);
          D2 = VFMA(vi, S4, D2); E2 = VFMA(vr, S4, E2);
          D3 = VFMA(vi, S6, D3); E3 = VFMA(vr, S6, E3);
          D4 = VFMA(vi, S8, D4); E4 = VFMA(vr, S8, E4);
          D5 = VFNMA(vi, S7, D5); E5 = VFNMA(vr, S7, E5);
          D6 = VFNMA(vi, S5, D6); E6 = VFNMA(vr, S5, E6);
          D7 = VFNMA(vi, S3, D7); E7 = VFNMA(vr, S3, E7);
          D8 = VFNMA(vi, S1, D8); E8 = VFNMA(vr, S1, E8);
        }
        { V ar = VL(pr + 408), ai = VL(pi + 408);
          V br = VL(pr + 1904), bi = VL(pi + 1904);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 32, VADD(ar,br)); VS(USCR + 40, VADD(ai,bi));
          D1 = VFMA(vi, S3, D1); E1 = VFMA(vr, S3, E1);
          D2 = VFMA(vi, S6, D2); E2 = VFMA(vr, S6, E2);
          D3 = VFNMA(vi, S8, D3); E3 = VFNMA(vr, S8, E3);
          D4 = VFNMA(vi, S5, D4); E4 = VFNMA(vr, S5, E4);
          D5 = VFNMA(vi, S2, D5); E5 = VFNMA(vr, S2, E5);
          D6 = VFMA(vi, S1, D6); E6 = VFMA(vr, S1, E6);
          D7 = VFMA(vi, S4, D7); E7 = VFMA(vr, S4, E7);
          D8 = VFMA(vi, S7, D8); E8 = VFMA(vr, S7, E8);
        }
        { V ar = VL(pr + 544), ai = VL(pi + 544);
          V br = VL(pr + 1768), bi = VL(pi + 1768);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 48, VADD(ar,br)); VS(USCR + 56, VADD(ai,bi));
          D1 = VFMA(vi, S4, D1); E1 = VFMA(vr, S4, E1);
          D2 = VFMA(vi, S8, D2); E2 = VFMA(vr, S8, E2);
          D3 = VFNMA(vi, S5, D3); E3 = VFNMA(vr, S5, E3);
          D4 = VFNMA(vi, S1, D4); E4 = VFNMA(vr, S1, E4);
          D5 = VFMA(vi, S3, D5); E5 = VFMA(vr, S3, E5);
          D6 = VFMA(vi, S7, D6); E6 = VFMA(vr, S7, E6);
          D7 = VFNMA(vi, S6, D7); E7 = VFNMA(vr, S6, E7);
          D8 = VFNMA(vi, S2, D8); E8 = VFNMA(vr, S2, E8);
        }
        { V ar = VL(pr + 680), ai = VL(pi + 680);
          V br = VL(pr + 1632), bi = VL(pi + 1632);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 64, VADD(ar,br)); VS(USCR + 72, VADD(ai,bi));
          D1 = VFMA(vi, S5, D1); E1 = VFMA(vr, S5, E1);
          D2 = VFNMA(vi, S7, D2); E2 = VFNMA(vr, S7, E2);
          D3 = VFNMA(vi, S2, D3); E3 = VFNMA(vr, S2, E3);
          D4 = VFMA(vi, S3, D4); E4 = VFMA(vr, S3, E4);
          D5 = VFMA(vi, S8, D5); E5 = VFMA(vr, S8, E5);
          D6 = VFNMA(vi, S4, D6); E6 = VFNMA(vr, S4, E6);
          D7 = VFMA(vi, S1, D7); E7 = VFMA(vr, S1, E7);
          D8 = VFMA(vi, S6, D8); E8 = VFMA(vr, S6, E8);
        }
        { V ar = VL(pr + 816), ai = VL(pi + 816);
          V br = VL(pr + 1496), bi = VL(pi + 1496);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 80, VADD(ar,br)); VS(USCR + 88, VADD(ai,bi));
          D1 = VFMA(vi, S6, D1); E1 = VFMA(vr, S6, E1);
          D2 = VFNMA(vi, S5, D2); E2 = VFNMA(vr, S5, E2);
          D3 = VFMA(vi, S1, D3); E3 = VFMA(vr, S1, E3);
          D4 = VFMA(vi, S7, D4); E4 = VFMA(vr, S7, E4);
          D5 = VFNMA(vi, S4, D5); E5 = VFNMA(vr, S4, E5);
          D6 = VFMA(vi, S2, D6); E6 = VFMA(vr, S2, E6);
          D7 = VFMA(vi, S8, D7); E7 = VFMA(vr, S8, E7);
          D8 = VFNMA(vi, S3, D8); E8 = VFNMA(vr, S3, E8);
        }
        { V ar = VL(pr + 952), ai = VL(pi + 952);
          V br = VL(pr + 1360), bi = VL(pi + 1360);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 96, VADD(ar,br)); VS(USCR + 104, VADD(ai,bi));
          D1 = VFMA(vi, S7, D1); E1 = VFMA(vr, S7, E1);
          D2 = VFNMA(vi, S3, D2); E2 = VFNMA(vr, S3, E2);
          D3 = VFMA(vi, S4, D3); E3 = VFMA(vr, S4, E3);
          D4 = VFNMA(vi, S6, D4); E4 = VFNMA(vr, S6, E4);
          D5 = VFMA(vi, S1, D5); E5 = VFMA(vr, S1, E5);
          D6 = VFMA(vi, S8, D6); E6 = VFMA(vr, S8, E6);
          D7 = VFNMA(vi, S2, D7); E7 = VFNMA(vr, S2, E7);
          D8 = VFMA(vi, S5, D8); E8 = VFMA(vr, S5, E8);
        }
        { V ar = VL(pr + 1088), ai = VL(pi + 1088);
          V br = VL(pr + 1224), bi = VL(pi + 1224);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 112, VADD(ar,br)); VS(USCR + 120, VADD(ai,bi));
          D1 = VFMA(vi, S8, D1); E1 = VFMA(vr, S8, E1);
          D2 = VFNMA(vi, S1, D2); E2 = VFNMA(vr, S1, E2);
          D3 = VFMA(vi, S7, D3); E3 = VFMA(vr, S7, E3);
          D4 = VFNMA(vi, S2, D4); E4 = VFNMA(vr, S2, E4);
          D5 = VFMA(vi, S6, D5); E5 = VFMA(vr, S6, E5);
          D6 = VFNMA(vi, S3, D6); E6 = VFNMA(vr, S3, E6);
          D7 = VFMA(vi, S5, D7); E7 = VFMA(vr, S5, E7);
          D8 = VFNMA(vi, S4, D8); E8 = VFNMA(vr, S4, E8);
        }
        VS(PSCR + 0, D1); VS(PSCR + 8, E1);
        VS(PSCR + 16, D2); VS(PSCR + 24, E2);
        VS(PSCR + 32, D3); VS(PSCR + 40, E3);
        VS(PSCR + 48, D4); VS(PSCR + 56, E4);
        VS(PSCR + 64, D5); VS(PSCR + 72, E5);
        VS(PSCR + 80, D6); VS(PSCR + 88, E6);
        VS(PSCR + 96, D7); VS(PSCR + 104, E7);
        VS(PSCR + 112, D8); VS(PSCR + 120, E8);
        }
        VS(USCR + 384, VL(pr)); VS(USCR + 392, VL(pi));  // save x0 (row 0 gets overwritten)
        {
        V C1 = VSET1(0x1.dd6d000370991p-1);
        V C2 = VSET1(0x1.7a5f6075d4884p-1);
        V C3 = VSET1(0x1.c86fa2b2883cep-2);
        V C4 = VSET1(0x1.79ee63259b75fp-4);
        V C5 = VSET1(-0x1.183b1c61f0d01p-2);
        V C6 = VSET1(-0x1.348c86ed5f1bap-1);
        V C7 = VSET1(-0x1.b34fa910ea3b8p-1);
        V C8 = VSET1(-0x1.f7484007faef3p-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A1 = x0r, B1 = x0i;
        V A2 = x0r, B2 = x0i;
        V A3 = x0r, B3 = x0i;
        V A4 = x0r, B4 = x0i;
        V A5 = x0r, B5 = x0i;
        V A6 = x0r, B6 = x0i;
        V A7 = x0r, B7 = x0i;
        V A8 = x0r, B8 = x0i;
        V s0r = x0r, s0i = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A1 = VFMA(ur, C1, A1); B1 = VFMA(ui, C1, B1);
          A2 = VFMA(ur, C2, A2); B2 = VFMA(ui, C2, B2);
          A3 = VFMA(ur, C3, A3); B3 = VFMA(ui, C3, B3);
          A4 = VFMA(ur, C4, A4); B4 = VFMA(ui, C4, B4);
          A5 = VFMA(ur, C5, A5); B5 = VFMA(ui, C5, B5);
          A6 = VFMA(ur, C6, A6); B6 = VFMA(ui, C6, B6);
          A7 = VFMA(ur, C7, A7); B7 = VFMA(ui, C7, B7);
          A8 = VFMA(ur, C8, A8); B8 = VFMA(ui, C8, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A1 = VFMA(ur, C2, A1); B1 = VFMA(ui, C2, B1);
          A2 = VFMA(ur, C4, A2); B2 = VFMA(ui, C4, B2);
          A3 = VFMA(ur, C6, A3); B3 = VFMA(ui, C6, B3);
          A4 = VFMA(ur, C8, A4); B4 = VFMA(ui, C8, B4);
          A5 = VFMA(ur, C7, A5); B5 = VFMA(ui, C7, B5);
          A6 = VFMA(ur, C5, A6); B6 = VFMA(ui, C5, B6);
          A7 = VFMA(ur, C3, A7); B7 = VFMA(ui, C3, B7);
          A8 = VFMA(ur, C1, A8); B8 = VFMA(ui, C1, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A1 = VFMA(ur, C3, A1); B1 = VFMA(ui, C3, B1);
          A2 = VFMA(ur, C6, A2); B2 = VFMA(ui, C6, B2);
          A3 = VFMA(ur, C8, A3); B3 = VFMA(ui, C8, B3);
          A4 = VFMA(ur, C5, A4); B4 = VFMA(ui, C5, B4);
          A5 = VFMA(ur, C2, A5); B5 = VFMA(ui, C2, B5);
          A6 = VFMA(ur, C1, A6); B6 = VFMA(ui, C1, B6);
          A7 = VFMA(ur, C4, A7); B7 = VFMA(ui, C4, B7);
          A8 = VFMA(ur, C7, A8); B8 = VFMA(ui, C7, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A1 = VFMA(ur, C4, A1); B1 = VFMA(ui, C4, B1);
          A2 = VFMA(ur, C8, A2); B2 = VFMA(ui, C8, B2);
          A3 = VFMA(ur, C5, A3); B3 = VFMA(ui, C5, B3);
          A4 = VFMA(ur, C1, A4); B4 = VFMA(ui, C1, B4);
          A5 = VFMA(ur, C3, A5); B5 = VFMA(ui, C3, B5);
          A6 = VFMA(ur, C7, A6); B6 = VFMA(ui, C7, B6);
          A7 = VFMA(ur, C6, A7); B7 = VFMA(ui, C6, B7);
          A8 = VFMA(ur, C2, A8); B8 = VFMA(ui, C2, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A1 = VFMA(ur, C5, A1); B1 = VFMA(ui, C5, B1);
          A2 = VFMA(ur, C7, A2); B2 = VFMA(ui, C7, B2);
          A3 = VFMA(ur, C2, A3); B3 = VFMA(ui, C2, B3);
          A4 = VFMA(ur, C3, A4); B4 = VFMA(ui, C3, B4);
          A5 = VFMA(ur, C8, A5); B5 = VFMA(ui, C8, B5);
          A6 = VFMA(ur, C4, A6); B6 = VFMA(ui, C4, B6);
          A7 = VFMA(ur, C1, A7); B7 = VFMA(ui, C1, B7);
          A8 = VFMA(ur, C6, A8); B8 = VFMA(ui, C6, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A1 = VFMA(ur, C6, A1); B1 = VFMA(ui, C6, B1);
          A2 = VFMA(ur, C5, A2); B2 = VFMA(ui, C5, B2);
          A3 = VFMA(ur, C1, A3); B3 = VFMA(ui, C1, B3);
          A4 = VFMA(ur, C7, A4); B4 = VFMA(ui, C7, B4);
          A5 = VFMA(ur, C4, A5); B5 = VFMA(ui, C4, B5);
          A6 = VFMA(ur, C2, A6); B6 = VFMA(ui, C2, B6);
          A7 = VFMA(ur, C8, A7); B7 = VFMA(ui, C8, B7);
          A8 = VFMA(ur, C3, A8); B8 = VFMA(ui, C3, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 96), ui = VL(USCR + 104);
          A1 = VFMA(ur, C7, A1); B1 = VFMA(ui, C7, B1);
          A2 = VFMA(ur, C3, A2); B2 = VFMA(ui, C3, B2);
          A3 = VFMA(ur, C4, A3); B3 = VFMA(ui, C4, B3);
          A4 = VFMA(ur, C6, A4); B4 = VFMA(ui, C6, B4);
          A5 = VFMA(ur, C1, A5); B5 = VFMA(ui, C1, B5);
          A6 = VFMA(ur, C8, A6); B6 = VFMA(ui, C8, B6);
          A7 = VFMA(ur, C2, A7); B7 = VFMA(ui, C2, B7);
          A8 = VFMA(ur, C5, A8); B8 = VFMA(ui, C5, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 112), ui = VL(USCR + 120);
          A1 = VFMA(ur, C8, A1); B1 = VFMA(ui, C8, B1);
          A2 = VFMA(ur, C1, A2); B2 = VFMA(ui, C1, B2);
          A3 = VFMA(ur, C7, A3); B3 = VFMA(ui, C7, B3);
          A4 = VFMA(ur, C2, A4); B4 = VFMA(ui, C2, B4);
          A5 = VFMA(ur, C6, A5); B5 = VFMA(ui, C6, B5);
          A6 = VFMA(ur, C3, A6); B6 = VFMA(ui, C3, B6);
          A7 = VFMA(ur, C5, A7); B7 = VFMA(ui, C5, B7);
          A8 = VFMA(ur, C4, A8); B8 = VFMA(ui, C4, B8);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V zr_ = VADD(s0r, VL(cr + 0)), zi_ = VADD(s0i, VL(ci + 0));
          MAP2(zr_, zi_);
          VS(pr + 0, zr_); VS(pi + 0, zi_); }
        { V Dk = VL(PSCR + 0), Ek = VL(PSCR + 8);
          V Xr = VADD(A1, Dk), Xi = VSUB(B1, Ek);
          V Yr = VSUB(A1, Dk), Yi = VADD(B1, Ek);
        { V zr_ = VADD(Xr, VL(cr + 136)), zi_ = VADD(Xi, VL(ci + 136));
          MAP2(zr_, zi_);
          VS(pr + 136, zr_); VS(pi + 136, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 2176)), zi_ = VADD(Yi, VL(ci + 2176));
          MAP2(zr_, zi_);
          VS(pr + 2176, zr_); VS(pi + 2176, zi_); }
        }
        { V Dk = VL(PSCR + 16), Ek = VL(PSCR + 24);
          V Xr = VADD(A2, Dk), Xi = VSUB(B2, Ek);
          V Yr = VSUB(A2, Dk), Yi = VADD(B2, Ek);
        { V zr_ = VADD(Xr, VL(cr + 272)), zi_ = VADD(Xi, VL(ci + 272));
          MAP2(zr_, zi_);
          VS(pr + 272, zr_); VS(pi + 272, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 2040)), zi_ = VADD(Yi, VL(ci + 2040));
          MAP2(zr_, zi_);
          VS(pr + 2040, zr_); VS(pi + 2040, zi_); }
        }
        { V Dk = VL(PSCR + 32), Ek = VL(PSCR + 40);
          V Xr = VADD(A3, Dk), Xi = VSUB(B3, Ek);
          V Yr = VSUB(A3, Dk), Yi = VADD(B3, Ek);
        { V zr_ = VADD(Xr, VL(cr + 408)), zi_ = VADD(Xi, VL(ci + 408));
          MAP2(zr_, zi_);
          VS(pr + 408, zr_); VS(pi + 408, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 1904)), zi_ = VADD(Yi, VL(ci + 1904));
          MAP2(zr_, zi_);
          VS(pr + 1904, zr_); VS(pi + 1904, zi_); }
        }
        { V Dk = VL(PSCR + 48), Ek = VL(PSCR + 56);
          V Xr = VADD(A4, Dk), Xi = VSUB(B4, Ek);
          V Yr = VSUB(A4, Dk), Yi = VADD(B4, Ek);
        { V zr_ = VADD(Xr, VL(cr + 544)), zi_ = VADD(Xi, VL(ci + 544));
          MAP2(zr_, zi_);
          VS(pr + 544, zr_); VS(pi + 544, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 1768)), zi_ = VADD(Yi, VL(ci + 1768));
          MAP2(zr_, zi_);
          VS(pr + 1768, zr_); VS(pi + 1768, zi_); }
        }
        { V Dk = VL(PSCR + 64), Ek = VL(PSCR + 72);
          V Xr = VADD(A5, Dk), Xi = VSUB(B5, Ek);
          V Yr = VSUB(A5, Dk), Yi = VADD(B5, Ek);
        { V zr_ = VADD(Xr, VL(cr + 680)), zi_ = VADD(Xi, VL(ci + 680));
          MAP2(zr_, zi_);
          VS(pr + 680, zr_); VS(pi + 680, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 1632)), zi_ = VADD(Yi, VL(ci + 1632));
          MAP2(zr_, zi_);
          VS(pr + 1632, zr_); VS(pi + 1632, zi_); }
        }
        { V Dk = VL(PSCR + 80), Ek = VL(PSCR + 88);
          V Xr = VADD(A6, Dk), Xi = VSUB(B6, Ek);
          V Yr = VSUB(A6, Dk), Yi = VADD(B6, Ek);
        { V zr_ = VADD(Xr, VL(cr + 816)), zi_ = VADD(Xi, VL(ci + 816));
          MAP2(zr_, zi_);
          VS(pr + 816, zr_); VS(pi + 816, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 1496)), zi_ = VADD(Yi, VL(ci + 1496));
          MAP2(zr_, zi_);
          VS(pr + 1496, zr_); VS(pi + 1496, zi_); }
        }
        { V Dk = VL(PSCR + 96), Ek = VL(PSCR + 104);
          V Xr = VADD(A7, Dk), Xi = VSUB(B7, Ek);
          V Yr = VSUB(A7, Dk), Yi = VADD(B7, Ek);
        { V zr_ = VADD(Xr, VL(cr + 952)), zi_ = VADD(Xi, VL(ci + 952));
          MAP2(zr_, zi_);
          VS(pr + 952, zr_); VS(pi + 952, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 1360)), zi_ = VADD(Yi, VL(ci + 1360));
          MAP2(zr_, zi_);
          VS(pr + 1360, zr_); VS(pi + 1360, zi_); }
        }
        { V Dk = VL(PSCR + 112), Ek = VL(PSCR + 120);
          V Xr = VADD(A8, Dk), Xi = VSUB(B8, Ek);
          V Yr = VSUB(A8, Dk), Yi = VADD(B8, Ek);
        { V zr_ = VADD(Xr, VL(cr + 1088)), zi_ = VADD(Xi, VL(ci + 1088));
          MAP2(zr_, zi_);
          VS(pr + 1088, zr_); VS(pi + 1088, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 1224)), zi_ = VADD(Yi, VL(ci + 1224));
          MAP2(zr_, zi_);
          VS(pr + 1224, zr_); VS(pi + 1224, zi_); }
        }
        }
    }
}

static void sw17_SyX(int y0, int pre){
    long b0 = (long)y0*17*8;
    p17_xxm(S17RE + b0, S17IM + b0, C17RE + b0, C17IM + b0, 17, 8);
    if (pre) {
        p17_zz(S17RE + (long)y0*17*8, S17IM + (long)y0*17*8, 17, (long)17*17*8);
        p17_xx(S17RE + b0, S17IM + b0, 17, 8);
    }
}
static void sw17_PxY(int x0, int pre){
    long b0 = (long)x0*17*17*8;
    p17_yym(S17RE + b0, S17IM + b0, C17RE + b0, C17IM + b0, 17, 8);
    if (pre) {
        p17_zz(S17RE + b0, S17IM + b0, 17, (long)17*8);
        p17_yy(S17RE + b0, S17IM + b0, 17, 8);
    }
}
void run_17(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if (m < 1) m = 1;
    long vs = 2*(long)4913;
    for (long g0 = 0; g0 < B; g0 += 8) {
        int nv = (B - g0) < 8 ? (int)(B - g0) : 8;
        const double* xg = x0 + g0*vs;
        const double* cg = c + g0*vs;
        long nfull = (4913/8)*8;
        if (nv == 8) {
            for (long s = 0; s < nfull; s += 8) { soa_in8(xg, vs, s, S17RE, S17IM); soa_in8(cg, vs, s, C17RE, C17IM); }
        } else {
            for (long s = 0; s < nfull; s += 8) { soa_in8_nv(xg, vs, s, S17RE, S17IM, nv); soa_in8_nv(cg, vs, s, C17RE, C17IM, nv); }
        }
        for (long s = nfull; s < 4913; s++) {   // tail sites scalar
            for (int v = 0; v < 8; v++) {
                S17RE[s*8+v] = v < nv ? xg[v*vs + 2*s] : 0.0;
                S17IM[s*8+v] = v < nv ? xg[v*vs + 2*s + 1] : 0.0;
                C17RE[s*8+v] = v < nv ? cg[v*vs + 2*s] : 0.0;
                C17IM[s*8+v] = v < nv ? cg[v*vs + 2*s + 1] : 0.0;
            }
        }
        // prologue: Z,Y per plane x
        for (int x = 0; x < 17; x++) {
            long b0 = (long)x*17*17*8;
            p17_zz(S17RE + b0, S17IM + b0, 17, (long)17*8);
            p17_yy(S17RE + b0, S17IM + b0, 17, 8);
        }
        for (long t = 1; t <= m; t++) {
            int snap = (t == 1) || (t == m);
            int pre = (t < m);
            int dopre = pre && !snap;
            if (t & 1) { for (int y0 = 0; y0 < 17; y0++) sw17_SyX(y0, dopre); }
            else       { for (int xp = 0; xp < 17; xp++) sw17_PxY(xp, dopre); }
            if (t == 1) {
                for (long s = 0; s < nfull; s += 8) soa_out8(S17RE, S17IM, s, out1 + g0*vs, vs, nv);
                for (long s = nfull; s < 4913; s++) for (int v = 0; v < nv; v++) {
                    out1[g0*vs + v*vs + 2*s] = S17RE[s*8+v]; out1[g0*vs + v*vs + 2*s+1] = S17IM[s*8+v]; }
            }
            if (t == m) {
                for (long s = 0; s < nfull; s += 8) soa_out8(S17RE, S17IM, s, outm + g0*vs, vs, nv);
                for (long s = nfull; s < 4913; s++) for (int v = 0; v < nv; v++) {
                    outm[g0*vs + v*vs + 2*s] = S17RE[s*8+v]; outm[g0*vs + v*vs + 2*s+1] = S17IM[s*8+v]; }
            }
            if (pre && !dopre) {
                p17_zz(S17RE, S17IM, 17*17, (long)17*8);
                if (t & 1) {
                    for (int y0 = 0; y0 < 17; y0++)
                        p17_xx(S17RE + (long)y0*17*8, S17IM + (long)y0*17*8, 17, 8);
                } else {
                    for (int x0 = 0; x0 < 17; x0++)
                        p17_yy(S17RE + (long)x0*17*17*8, S17IM + (long)x0*17*17*8, 17, 8);
                }
            }
        }
    }
}

// ============ L=23 SoA-8 ============
static double S23RE[97336] ALIGN64;
static double S23IM[97336] ALIGN64;
static double C23RE[97336] ALIGN64;
static double C23IM[97336] ALIGN64;
static const double CT23[11] ALIGN64 = { 0x1.ed037ea3d2dbcp-1, 0x1.b57675cf309eep-1, 0x1.5d779b07cfef7p-1, 0x1.d71b4a0c5a6c9p-2, 0x1.a0ad8bd1e2881p-3, -0x1.17855b599f3b2p-4, -0x1.56eaae597c776p-2, -0x1.2742a4a775cfap-1, -0x1.8d2a07c16d46ep-1, -0x1.d59cb83ef99bcp-1, -0x1.fb3b3035aa6ccp-1 };
static const double ST23[11] ALIGN64 = { 0x1.14459ad2be466p-2, 0x1.0a06e851db7cap-1, 0x1.763021aaa15d9p-1, 0x1.c698e42f47b09p-1, 0x1.f54a827142577p-1, 0x1.fece70dfd3efbp-1, 0x1.e270060999288p-1, 0x1.a249e0b897caap-1, 0x1.431df5838f7f1p-1, 0x1.97f6748e524b1p-2, 0x1.16de8a4564f1cp-3 };
static __attribute__((noinline)) void p23_zz(double* PR, double* PI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        {
        V S1 = VSET1(0x1.14459ad2be466p-2);
        V S2 = VSET1(0x1.0a06e851db7cap-1);
        V S3 = VSET1(0x1.763021aaa15d9p-1);
        V S4 = VSET1(0x1.c698e42f47b09p-1);
        V S5 = VSET1(0x1.f54a827142577p-1);
        V S6 = VSET1(0x1.fece70dfd3efbp-1);
        V S7 = VSET1(0x1.e270060999288p-1);
        V S8 = VSET1(0x1.a249e0b897caap-1);
        V S9 = VSET1(0x1.431df5838f7f1p-1);
        V S10 = VSET1(0x1.97f6748e524b1p-2);
        V S11 = VSET1(0x1.16de8a4564f1cp-3);
        V D1 = _mm512_setzero_pd(), E1 = _mm512_setzero_pd();
        V D2 = _mm512_setzero_pd(), E2 = _mm512_setzero_pd();
        V D3 = _mm512_setzero_pd(), E3 = _mm512_setzero_pd();
        V D4 = _mm512_setzero_pd(), E4 = _mm512_setzero_pd();
        V D5 = _mm512_setzero_pd(), E5 = _mm512_setzero_pd();
        V D6 = _mm512_setzero_pd(), E6 = _mm512_setzero_pd();
        { V ar = VL(pr + 8), ai = VL(pi + 8);
          V br = VL(pr + 176), bi = VL(pi + 176);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 0, VADD(ar,br)); VS(USCR + 8, VADD(ai,bi));
          D1 = VFMA(vi, S1, D1); E1 = VFMA(vr, S1, E1);
          D2 = VFMA(vi, S2, D2); E2 = VFMA(vr, S2, E2);
          D3 = VFMA(vi, S3, D3); E3 = VFMA(vr, S3, E3);
          D4 = VFMA(vi, S4, D4); E4 = VFMA(vr, S4, E4);
          D5 = VFMA(vi, S5, D5); E5 = VFMA(vr, S5, E5);
          D6 = VFMA(vi, S6, D6); E6 = VFMA(vr, S6, E6);
        }
        { V ar = VL(pr + 16), ai = VL(pi + 16);
          V br = VL(pr + 168), bi = VL(pi + 168);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 16, VADD(ar,br)); VS(USCR + 24, VADD(ai,bi));
          D1 = VFMA(vi, S2, D1); E1 = VFMA(vr, S2, E1);
          D2 = VFMA(vi, S4, D2); E2 = VFMA(vr, S4, E2);
          D3 = VFMA(vi, S6, D3); E3 = VFMA(vr, S6, E3);
          D4 = VFMA(vi, S8, D4); E4 = VFMA(vr, S8, E4);
          D5 = VFMA(vi, S10, D5); E5 = VFMA(vr, S10, E5);
          D6 = VFNMA(vi, S11, D6); E6 = VFNMA(vr, S11, E6);
        }
        { V ar = VL(pr + 24), ai = VL(pi + 24);
          V br = VL(pr + 160), bi = VL(pi + 160);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 32, VADD(ar,br)); VS(USCR + 40, VADD(ai,bi));
          D1 = VFMA(vi, S3, D1); E1 = VFMA(vr, S3, E1);
          D2 = VFMA(vi, S6, D2); E2 = VFMA(vr, S6, E2);
          D3 = VFMA(vi, S9, D3); E3 = VFMA(vr, S9, E3);
          D4 = VFNMA(vi, S11, D4); E4 = VFNMA(vr, S11, E4);
          D5 = VFNMA(vi, S8, D5); E5 = VFNMA(vr, S8, E5);
          D6 = VFNMA(vi, S5, D6); E6 = VFNMA(vr, S5, E6);
        }
        { V ar = VL(pr + 32), ai = VL(pi + 32);
          V br = VL(pr + 152), bi = VL(pi + 152);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 48, VADD(ar,br)); VS(USCR + 56, VADD(ai,bi));
          D1 = VFMA(vi, S4, D1); E1 = VFMA(vr, S4, E1);
          D2 = VFMA(vi, S8, D2); E2 = VFMA(vr, S8, E2);
          D3 = VFNMA(vi, S11, D3); E3 = VFNMA(vr, S11, E3);
          D4 = VFNMA(vi, S7, D4); E4 = VFNMA(vr, S7, E4);
          D5 = VFNMA(vi, S3, D5); E5 = VFNMA(vr, S3, E5);
          D6 = VFMA(vi, S1, D6); E6 = VFMA(vr, S1, E6);
        }
        { V ar = VL(pr + 40), ai = VL(pi + 40);
          V br = VL(pr + 144), bi = VL(pi + 144);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 64, VADD(ar,br)); VS(USCR + 72, VADD(ai,bi));
          D1 = VFMA(vi, S5, D1); E1 = VFMA(vr, S5, E1);
          D2 = VFMA(vi, S10, D2); E2 = VFMA(vr, S10, E2);
          D3 = VFNMA(vi, S8, D3); E3 = VFNMA(vr, S8, E3);
          D4 = VFNMA(vi, S3, D4); E4 = VFNMA(vr, S3, E4);
          D5 = VFMA(vi, S2, D5); E5 = VFMA(vr, S2, E5);
          D6 = VFMA(vi, S7, D6); E6 = VFMA(vr, S7, E6);
        }
        { V ar = VL(pr + 48), ai = VL(pi + 48);
          V br = VL(pr + 136), bi = VL(pi + 136);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 80, VADD(ar,br)); VS(USCR + 88, VADD(ai,bi));
          D1 = VFMA(vi, S6, D1); E1 = VFMA(vr, S6, E1);
          D2 = VFNMA(vi, S11, D2); E2 = VFNMA(vr, S11, E2);
          D3 = VFNMA(vi, S5, D3); E3 = VFNMA(vr, S5, E3);
          D4 = VFMA(vi, S1, D4); E4 = VFMA(vr, S1, E4);
          D5 = VFMA(vi, S7, D5); E5 = VFMA(vr, S7, E5);
          D6 = VFNMA(vi, S10, D6); E6 = VFNMA(vr, S10, E6);
        }
        { V ar = VL(pr + 56), ai = VL(pi + 56);
          V br = VL(pr + 128), bi = VL(pi + 128);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 96, VADD(ar,br)); VS(USCR + 104, VADD(ai,bi));
          D1 = VFMA(vi, S7, D1); E1 = VFMA(vr, S7, E1);
          D2 = VFNMA(vi, S9, D2); E2 = VFNMA(vr, S9, E2);
          D3 = VFNMA(vi, S2, D3); E3 = VFNMA(vr, S2, E3);
          D4 = VFMA(vi, S5, D4); E4 = VFMA(vr, S5, E4);
          D5 = VFNMA(vi, S11, D5); E5 = VFNMA(vr, S11, E5);
          D6 = VFNMA(vi, S4, D6); E6 = VFNMA(vr, S4, E6);
        }
        { V ar = VL(pr + 64), ai = VL(pi + 64);
          V br = VL(pr + 120), bi = VL(pi + 120);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 112, VADD(ar,br)); VS(USCR + 120, VADD(ai,bi));
          D1 = VFMA(vi, S8, D1); E1 = VFMA(vr, S8, E1);
          D2 = VFNMA(vi, S7, D2); E2 = VFNMA(vr, S7, E2);
          D3 = VFMA(vi, S1, D3); E3 = VFMA(vr, S1, E3);
          D4 = VFMA(vi, S9, D4); E4 = VFMA(vr, S9, E4);
          D5 = VFNMA(vi, S6, D5); E5 = VFNMA(vr, S6, E5);
          D6 = VFMA(vi, S2, D6); E6 = VFMA(vr, S2, E6);
        }
        { V ar = VL(pr + 72), ai = VL(pi + 72);
          V br = VL(pr + 112), bi = VL(pi + 112);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 128, VADD(ar,br)); VS(USCR + 136, VADD(ai,bi));
          D1 = VFMA(vi, S9, D1); E1 = VFMA(vr, S9, E1);
          D2 = VFNMA(vi, S5, D2); E2 = VFNMA(vr, S5, E2);
          D3 = VFMA(vi, S4, D3); E3 = VFMA(vr, S4, E3);
          D4 = VFNMA(vi, S10, D4); E4 = VFNMA(vr, S10, E4);
          D5 = VFNMA(vi, S1, D5); E5 = VFNMA(vr, S1, E5);
          D6 = VFMA(vi, S8, D6); E6 = VFMA(vr, S8, E6);
        }
        { V ar = VL(pr + 80), ai = VL(pi + 80);
          V br = VL(pr + 104), bi = VL(pi + 104);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 144, VADD(ar,br)); VS(USCR + 152, VADD(ai,bi));
          D1 = VFMA(vi, S10, D1); E1 = VFMA(vr, S10, E1);
          D2 = VFNMA(vi, S3, D2); E2 = VFNMA(vr, S3, E2);
          D3 = VFMA(vi, S7, D3); E3 = VFMA(vr, S7, E3);
          D4 = VFNMA(vi, S6, D4); E4 = VFNMA(vr, S6, E4);
          D5 = VFMA(vi, S4, D5); E5 = VFMA(vr, S4, E5);
          D6 = VFNMA(vi, S9, D6); E6 = VFNMA(vr, S9, E6);
        }
        { V ar = VL(pr + 88), ai = VL(pi + 88);
          V br = VL(pr + 96), bi = VL(pi + 96);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 160, VADD(ar,br)); VS(USCR + 168, VADD(ai,bi));
          D1 = VFMA(vi, S11, D1); E1 = VFMA(vr, S11, E1);
          D2 = VFNMA(vi, S1, D2); E2 = VFNMA(vr, S1, E2);
          D3 = VFMA(vi, S10, D3); E3 = VFMA(vr, S10, E3);
          D4 = VFNMA(vi, S2, D4); E4 = VFNMA(vr, S2, E4);
          D5 = VFMA(vi, S9, D5); E5 = VFMA(vr, S9, E5);
          D6 = VFNMA(vi, S3, D6); E6 = VFNMA(vr, S3, E6);
        }
        VS(PSCR + 0, D1); VS(PSCR + 8, E1);
        VS(PSCR + 16, D2); VS(PSCR + 24, E2);
        VS(PSCR + 32, D3); VS(PSCR + 40, E3);
        VS(PSCR + 48, D4); VS(PSCR + 56, E4);
        VS(PSCR + 64, D5); VS(PSCR + 72, E5);
        VS(PSCR + 80, D6); VS(PSCR + 88, E6);
        }
        {
        V S1 = VSET1(0x1.14459ad2be466p-2);
        V S2 = VSET1(0x1.0a06e851db7cap-1);
        V S3 = VSET1(0x1.763021aaa15d9p-1);
        V S4 = VSET1(0x1.c698e42f47b09p-1);
        V S5 = VSET1(0x1.f54a827142577p-1);
        V S6 = VSET1(0x1.fece70dfd3efbp-1);
        V S7 = VSET1(0x1.e270060999288p-1);
        V S8 = VSET1(0x1.a249e0b897caap-1);
        V S9 = VSET1(0x1.431df5838f7f1p-1);
        V S10 = VSET1(0x1.97f6748e524b1p-2);
        V S11 = VSET1(0x1.16de8a4564f1cp-3);
        V D7 = _mm512_setzero_pd(), E7 = _mm512_setzero_pd();
        V D8 = _mm512_setzero_pd(), E8 = _mm512_setzero_pd();
        V D9 = _mm512_setzero_pd(), E9 = _mm512_setzero_pd();
        V D10 = _mm512_setzero_pd(), E10 = _mm512_setzero_pd();
        V D11 = _mm512_setzero_pd(), E11 = _mm512_setzero_pd();
        { V ar = VL(pr + 8), ai = VL(pi + 8);
          V br = VL(pr + 176), bi = VL(pi + 176);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S7, D7); E7 = VFMA(vr, S7, E7);
          D8 = VFMA(vi, S8, D8); E8 = VFMA(vr, S8, E8);
          D9 = VFMA(vi, S9, D9); E9 = VFMA(vr, S9, E9);
          D10 = VFMA(vi, S10, D10); E10 = VFMA(vr, S10, E10);
          D11 = VFMA(vi, S11, D11); E11 = VFMA(vr, S11, E11);
        }
        { V ar = VL(pr + 16), ai = VL(pi + 16);
          V br = VL(pr + 168), bi = VL(pi + 168);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S9, D7); E7 = VFNMA(vr, S9, E7);
          D8 = VFNMA(vi, S7, D8); E8 = VFNMA(vr, S7, E8);
          D9 = VFNMA(vi, S5, D9); E9 = VFNMA(vr, S5, E9);
          D10 = VFNMA(vi, S3, D10); E10 = VFNMA(vr, S3, E10);
          D11 = VFNMA(vi, S1, D11); E11 = VFNMA(vr, S1, E11);
        }
        { V ar = VL(pr + 24), ai = VL(pi + 24);
          V br = VL(pr + 160), bi = VL(pi + 160);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S2, D7); E7 = VFNMA(vr, S2, E7);
          D8 = VFMA(vi, S1, D8); E8 = VFMA(vr, S1, E8);
          D9 = VFMA(vi, S4, D9); E9 = VFMA(vr, S4, E9);
          D10 = VFMA(vi, S7, D10); E10 = VFMA(vr, S7, E10);
          D11 = VFMA(vi, S10, D11); E11 = VFMA(vr, S10, E11);
        }
        { V ar = VL(pr + 32), ai = VL(pi + 32);
          V br = VL(pr + 152), bi = VL(pi + 152);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S5, D7); E7 = VFMA(vr, S5, E7);
          D8 = VFMA(vi, S9, D8); E8 = VFMA(vr, S9, E8);
          D9 = VFNMA(vi, S10, D9); E9 = VFNMA(vr, S10, E9);
          D10 = VFNMA(vi, S6, D10); E10 = VFNMA(vr, S6, E10);
          D11 = VFNMA(vi, S2, D11); E11 = VFNMA(vr, S2, E11);
        }
        { V ar = VL(pr + 40), ai = VL(pi + 40);
          V br = VL(pr + 144), bi = VL(pi + 144);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S11, D7); E7 = VFNMA(vr, S11, E7);
          D8 = VFNMA(vi, S6, D8); E8 = VFNMA(vr, S6, E8);
          D9 = VFNMA(vi, S1, D9); E9 = VFNMA(vr, S1, E9);
          D10 = VFMA(vi, S4, D10); E10 = VFMA(vr, S4, E10);
          D11 = VFMA(vi, S9, D11); E11 = VFMA(vr, S9, E11);
        }
        { V ar = VL(pr + 48), ai = VL(pi + 48);
          V br = VL(pr + 136), bi = VL(pi + 136);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S4, D7); E7 = VFNMA(vr, S4, E7);
          D8 = VFMA(vi, S2, D8); E8 = VFMA(vr, S2, E8);
          D9 = VFMA(vi, S8, D9); E9 = VFMA(vr, S8, E9);
          D10 = VFNMA(vi, S9, D10); E10 = VFNMA(vr, S9, E10);
          D11 = VFNMA(vi, S3, D11); E11 = VFNMA(vr, S3, E11);
        }
        { V ar = VL(pr + 56), ai = VL(pi + 56);
          V br = VL(pr + 128), bi = VL(pi + 128);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S3, D7); E7 = VFMA(vr, S3, E7);
          D8 = VFMA(vi, S10, D8); E8 = VFMA(vr, S10, E8);
          D9 = VFNMA(vi, S6, D9); E9 = VFNMA(vr, S6, E9);
          D10 = VFMA(vi, S1, D10); E10 = VFMA(vr, S1, E10);
          D11 = VFMA(vi, S8, D11); E11 = VFMA(vr, S8, E11);
        }
        { V ar = VL(pr + 64), ai = VL(pi + 64);
          V br = VL(pr + 120), bi = VL(pi + 120);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S10, D7); E7 = VFMA(vr, S10, E7);
          D8 = VFNMA(vi, S5, D8); E8 = VFNMA(vr, S5, E8);
          D9 = VFMA(vi, S3, D9); E9 = VFMA(vr, S3, E9);
          D10 = VFMA(vi, S11, D10); E10 = VFMA(vr, S11, E10);
          D11 = VFNMA(vi, S4, D11); E11 = VFNMA(vr, S4, E11);
        }
        { V ar = VL(pr + 72), ai = VL(pi + 72);
          V br = VL(pr + 112), bi = VL(pi + 112);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S6, D7); E7 = VFNMA(vr, S6, E7);
          D8 = VFMA(vi, S3, D8); E8 = VFMA(vr, S3, E8);
          D9 = VFNMA(vi, S11, D9); E9 = VFNMA(vr, S11, E9);
          D10 = VFNMA(vi, S2, D10); E10 = VFNMA(vr, S2, E10);
          D11 = VFMA(vi, S7, D11); E11 = VFMA(vr, S7, E11);
        }
        { V ar = VL(pr + 80), ai = VL(pi + 80);
          V br = VL(pr + 104), bi = VL(pi + 104);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S1, D7); E7 = VFMA(vr, S1, E7);
          D8 = VFMA(vi, S11, D8); E8 = VFMA(vr, S11, E8);
          D9 = VFNMA(vi, S2, D9); E9 = VFNMA(vr, S2, E9);
          D10 = VFMA(vi, S8, D10); E10 = VFMA(vr, S8, E10);
          D11 = VFNMA(vi, S5, D11); E11 = VFNMA(vr, S5, E11);
        }
        { V ar = VL(pr + 88), ai = VL(pi + 88);
          V br = VL(pr + 96), bi = VL(pi + 96);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S8, D7); E7 = VFMA(vr, S8, E7);
          D8 = VFNMA(vi, S4, D8); E8 = VFNMA(vr, S4, E8);
          D9 = VFMA(vi, S7, D9); E9 = VFMA(vr, S7, E9);
          D10 = VFNMA(vi, S5, D10); E10 = VFNMA(vr, S5, E10);
          D11 = VFMA(vi, S6, D11); E11 = VFMA(vr, S6, E11);
        }
        VS(PSCR + 96, D7); VS(PSCR + 104, E7);
        VS(PSCR + 112, D8); VS(PSCR + 120, E8);
        VS(PSCR + 128, D9); VS(PSCR + 136, E9);
        VS(PSCR + 144, D10); VS(PSCR + 152, E10);
        VS(PSCR + 160, D11); VS(PSCR + 168, E11);
        }
        VS(USCR + 384, VL(pr)); VS(USCR + 392, VL(pi));  // save x0 (row 0 gets overwritten)
        {
        V C1 = VSET1(0x1.ed037ea3d2dbcp-1);
        V C2 = VSET1(0x1.b57675cf309eep-1);
        V C3 = VSET1(0x1.5d779b07cfef7p-1);
        V C4 = VSET1(0x1.d71b4a0c5a6c9p-2);
        V C5 = VSET1(0x1.a0ad8bd1e2881p-3);
        V C6 = VSET1(-0x1.17855b599f3b2p-4);
        V C7 = VSET1(-0x1.56eaae597c776p-2);
        V C8 = VSET1(-0x1.2742a4a775cfap-1);
        V C9 = VSET1(-0x1.8d2a07c16d46ep-1);
        V C10 = VSET1(-0x1.d59cb83ef99bcp-1);
        V C11 = VSET1(-0x1.fb3b3035aa6ccp-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A1 = x0r, B1 = x0i;
        V A2 = x0r, B2 = x0i;
        V A3 = x0r, B3 = x0i;
        V A4 = x0r, B4 = x0i;
        V A5 = x0r, B5 = x0i;
        V A6 = x0r, B6 = x0i;
        V s0r = x0r, s0i = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A1 = VFMA(ur, C1, A1); B1 = VFMA(ui, C1, B1);
          A2 = VFMA(ur, C2, A2); B2 = VFMA(ui, C2, B2);
          A3 = VFMA(ur, C3, A3); B3 = VFMA(ui, C3, B3);
          A4 = VFMA(ur, C4, A4); B4 = VFMA(ui, C4, B4);
          A5 = VFMA(ur, C5, A5); B5 = VFMA(ui, C5, B5);
          A6 = VFMA(ur, C6, A6); B6 = VFMA(ui, C6, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A1 = VFMA(ur, C2, A1); B1 = VFMA(ui, C2, B1);
          A2 = VFMA(ur, C4, A2); B2 = VFMA(ui, C4, B2);
          A3 = VFMA(ur, C6, A3); B3 = VFMA(ui, C6, B3);
          A4 = VFMA(ur, C8, A4); B4 = VFMA(ui, C8, B4);
          A5 = VFMA(ur, C10, A5); B5 = VFMA(ui, C10, B5);
          A6 = VFMA(ur, C11, A6); B6 = VFMA(ui, C11, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A1 = VFMA(ur, C3, A1); B1 = VFMA(ui, C3, B1);
          A2 = VFMA(ur, C6, A2); B2 = VFMA(ui, C6, B2);
          A3 = VFMA(ur, C9, A3); B3 = VFMA(ui, C9, B3);
          A4 = VFMA(ur, C11, A4); B4 = VFMA(ui, C11, B4);
          A5 = VFMA(ur, C8, A5); B5 = VFMA(ui, C8, B5);
          A6 = VFMA(ur, C5, A6); B6 = VFMA(ui, C5, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A1 = VFMA(ur, C4, A1); B1 = VFMA(ui, C4, B1);
          A2 = VFMA(ur, C8, A2); B2 = VFMA(ui, C8, B2);
          A3 = VFMA(ur, C11, A3); B3 = VFMA(ui, C11, B3);
          A4 = VFMA(ur, C7, A4); B4 = VFMA(ui, C7, B4);
          A5 = VFMA(ur, C3, A5); B5 = VFMA(ui, C3, B5);
          A6 = VFMA(ur, C1, A6); B6 = VFMA(ui, C1, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A1 = VFMA(ur, C5, A1); B1 = VFMA(ui, C5, B1);
          A2 = VFMA(ur, C10, A2); B2 = VFMA(ui, C10, B2);
          A3 = VFMA(ur, C8, A3); B3 = VFMA(ui, C8, B3);
          A4 = VFMA(ur, C3, A4); B4 = VFMA(ui, C3, B4);
          A5 = VFMA(ur, C2, A5); B5 = VFMA(ui, C2, B5);
          A6 = VFMA(ur, C7, A6); B6 = VFMA(ui, C7, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A1 = VFMA(ur, C6, A1); B1 = VFMA(ui, C6, B1);
          A2 = VFMA(ur, C11, A2); B2 = VFMA(ui, C11, B2);
          A3 = VFMA(ur, C5, A3); B3 = VFMA(ui, C5, B3);
          A4 = VFMA(ur, C1, A4); B4 = VFMA(ui, C1, B4);
          A5 = VFMA(ur, C7, A5); B5 = VFMA(ui, C7, B5);
          A6 = VFMA(ur, C10, A6); B6 = VFMA(ui, C10, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 96), ui = VL(USCR + 104);
          A1 = VFMA(ur, C7, A1); B1 = VFMA(ui, C7, B1);
          A2 = VFMA(ur, C9, A2); B2 = VFMA(ui, C9, B2);
          A3 = VFMA(ur, C2, A3); B3 = VFMA(ui, C2, B3);
          A4 = VFMA(ur, C5, A4); B4 = VFMA(ui, C5, B4);
          A5 = VFMA(ur, C11, A5); B5 = VFMA(ui, C11, B5);
          A6 = VFMA(ur, C4, A6); B6 = VFMA(ui, C4, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 112), ui = VL(USCR + 120);
          A1 = VFMA(ur, C8, A1); B1 = VFMA(ui, C8, B1);
          A2 = VFMA(ur, C7, A2); B2 = VFMA(ui, C7, B2);
          A3 = VFMA(ur, C1, A3); B3 = VFMA(ui, C1, B3);
          A4 = VFMA(ur, C9, A4); B4 = VFMA(ui, C9, B4);
          A5 = VFMA(ur, C6, A5); B5 = VFMA(ui, C6, B5);
          A6 = VFMA(ur, C2, A6); B6 = VFMA(ui, C2, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 128), ui = VL(USCR + 136);
          A1 = VFMA(ur, C9, A1); B1 = VFMA(ui, C9, B1);
          A2 = VFMA(ur, C5, A2); B2 = VFMA(ui, C5, B2);
          A3 = VFMA(ur, C4, A3); B3 = VFMA(ui, C4, B3);
          A4 = VFMA(ur, C10, A4); B4 = VFMA(ui, C10, B4);
          A5 = VFMA(ur, C1, A5); B5 = VFMA(ui, C1, B5);
          A6 = VFMA(ur, C8, A6); B6 = VFMA(ui, C8, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 144), ui = VL(USCR + 152);
          A1 = VFMA(ur, C10, A1); B1 = VFMA(ui, C10, B1);
          A2 = VFMA(ur, C3, A2); B2 = VFMA(ui, C3, B2);
          A3 = VFMA(ur, C7, A3); B3 = VFMA(ui, C7, B3);
          A4 = VFMA(ur, C6, A4); B4 = VFMA(ui, C6, B4);
          A5 = VFMA(ur, C4, A5); B5 = VFMA(ui, C4, B5);
          A6 = VFMA(ur, C9, A6); B6 = VFMA(ui, C9, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 160), ui = VL(USCR + 168);
          A1 = VFMA(ur, C11, A1); B1 = VFMA(ui, C11, B1);
          A2 = VFMA(ur, C1, A2); B2 = VFMA(ui, C1, B2);
          A3 = VFMA(ur, C10, A3); B3 = VFMA(ui, C10, B3);
          A4 = VFMA(ur, C2, A4); B4 = VFMA(ui, C2, B4);
          A5 = VFMA(ur, C9, A5); B5 = VFMA(ui, C9, B5);
          A6 = VFMA(ur, C3, A6); B6 = VFMA(ui, C3, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        VS(pr + 0, s0r); VS(pi + 0, s0i);
        { V Dk = VL(PSCR + 0), Ek = VL(PSCR + 8);
          V Xr = VADD(A1, Dk), Xi = VSUB(B1, Ek);
          V Yr = VSUB(A1, Dk), Yi = VADD(B1, Ek);
        VS(pr + 8, Xr); VS(pi + 8, Xi);
        VS(pr + 176, Yr); VS(pi + 176, Yi);
        }
        { V Dk = VL(PSCR + 16), Ek = VL(PSCR + 24);
          V Xr = VADD(A2, Dk), Xi = VSUB(B2, Ek);
          V Yr = VSUB(A2, Dk), Yi = VADD(B2, Ek);
        VS(pr + 16, Xr); VS(pi + 16, Xi);
        VS(pr + 168, Yr); VS(pi + 168, Yi);
        }
        { V Dk = VL(PSCR + 32), Ek = VL(PSCR + 40);
          V Xr = VADD(A3, Dk), Xi = VSUB(B3, Ek);
          V Yr = VSUB(A3, Dk), Yi = VADD(B3, Ek);
        VS(pr + 24, Xr); VS(pi + 24, Xi);
        VS(pr + 160, Yr); VS(pi + 160, Yi);
        }
        { V Dk = VL(PSCR + 48), Ek = VL(PSCR + 56);
          V Xr = VADD(A4, Dk), Xi = VSUB(B4, Ek);
          V Yr = VSUB(A4, Dk), Yi = VADD(B4, Ek);
        VS(pr + 32, Xr); VS(pi + 32, Xi);
        VS(pr + 152, Yr); VS(pi + 152, Yi);
        }
        { V Dk = VL(PSCR + 64), Ek = VL(PSCR + 72);
          V Xr = VADD(A5, Dk), Xi = VSUB(B5, Ek);
          V Yr = VSUB(A5, Dk), Yi = VADD(B5, Ek);
        VS(pr + 40, Xr); VS(pi + 40, Xi);
        VS(pr + 144, Yr); VS(pi + 144, Yi);
        }
        { V Dk = VL(PSCR + 80), Ek = VL(PSCR + 88);
          V Xr = VADD(A6, Dk), Xi = VSUB(B6, Ek);
          V Yr = VSUB(A6, Dk), Yi = VADD(B6, Ek);
        VS(pr + 48, Xr); VS(pi + 48, Xi);
        VS(pr + 136, Yr); VS(pi + 136, Yi);
        }
        }
        {
        V C1 = VSET1(0x1.ed037ea3d2dbcp-1);
        V C2 = VSET1(0x1.b57675cf309eep-1);
        V C3 = VSET1(0x1.5d779b07cfef7p-1);
        V C4 = VSET1(0x1.d71b4a0c5a6c9p-2);
        V C5 = VSET1(0x1.a0ad8bd1e2881p-3);
        V C6 = VSET1(-0x1.17855b599f3b2p-4);
        V C7 = VSET1(-0x1.56eaae597c776p-2);
        V C8 = VSET1(-0x1.2742a4a775cfap-1);
        V C9 = VSET1(-0x1.8d2a07c16d46ep-1);
        V C10 = VSET1(-0x1.d59cb83ef99bcp-1);
        V C11 = VSET1(-0x1.fb3b3035aa6ccp-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A7 = x0r, B7 = x0i;
        V A8 = x0r, B8 = x0i;
        V A9 = x0r, B9 = x0i;
        V A10 = x0r, B10 = x0i;
        V A11 = x0r, B11 = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A7 = VFMA(ur, C7, A7); B7 = VFMA(ui, C7, B7);
          A8 = VFMA(ur, C8, A8); B8 = VFMA(ui, C8, B8);
          A9 = VFMA(ur, C9, A9); B9 = VFMA(ui, C9, B9);
          A10 = VFMA(ur, C10, A10); B10 = VFMA(ui, C10, B10);
          A11 = VFMA(ur, C11, A11); B11 = VFMA(ui, C11, B11);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A7 = VFMA(ur, C9, A7); B7 = VFMA(ui, C9, B7);
          A8 = VFMA(ur, C7, A8); B8 = VFMA(ui, C7, B8);
          A9 = VFMA(ur, C5, A9); B9 = VFMA(ui, C5, B9);
          A10 = VFMA(ur, C3, A10); B10 = VFMA(ui, C3, B10);
          A11 = VFMA(ur, C1, A11); B11 = VFMA(ui, C1, B11);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A7 = VFMA(ur, C2, A7); B7 = VFMA(ui, C2, B7);
          A8 = VFMA(ur, C1, A8); B8 = VFMA(ui, C1, B8);
          A9 = VFMA(ur, C4, A9); B9 = VFMA(ui, C4, B9);
          A10 = VFMA(ur, C7, A10); B10 = VFMA(ui, C7, B10);
          A11 = VFMA(ur, C10, A11); B11 = VFMA(ui, C10, B11);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A7 = VFMA(ur, C5, A7); B7 = VFMA(ui, C5, B7);
          A8 = VFMA(ur, C9, A8); B8 = VFMA(ui, C9, B8);
          A9 = VFMA(ur, C10, A9); B9 = VFMA(ui, C10, B9);
          A10 = VFMA(ur, C6, A10); B10 = VFMA(ui, C6, B10);
          A11 = VFMA(ur, C2, A11); B11 = VFMA(ui, C2, B11);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A7 = VFMA(ur, C11, A7); B7 = VFMA(ui, C11, B7);
          A8 = VFMA(ur, C6, A8); B8 = VFMA(ui, C6, B8);
          A9 = VFMA(ur, C1, A9); B9 = VFMA(ui, C1, B9);
          A10 = VFMA(ur, C4, A10); B10 = VFMA(ui, C4, B10);
          A11 = VFMA(ur, C9, A11); B11 = VFMA(ui, C9, B11);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A7 = VFMA(ur, C4, A7); B7 = VFMA(ui, C4, B7);
          A8 = VFMA(ur, C2, A8); B8 = VFMA(ui, C2, B8);
          A9 = VFMA(ur, C8, A9); B9 = VFMA(ui, C8, B9);
          A10 = VFMA(ur, C9, A10); B10 = VFMA(ui, C9, B10);
          A11 = VFMA(ur, C3, A11); B11 = VFMA(ui, C3, B11);
        }
        { V ur = VL(USCR + 96), ui = VL(USCR + 104);
          A7 = VFMA(ur, C3, A7); B7 = VFMA(ui, C3, B7);
          A8 = VFMA(ur, C10, A8); B8 = VFMA(ui, C10, B8);
          A9 = VFMA(ur, C6, A9); B9 = VFMA(ui, C6, B9);
          A10 = VFMA(ur, C1, A10); B10 = VFMA(ui, C1, B10);
          A11 = VFMA(ur, C8, A11); B11 = VFMA(ui, C8, B11);
        }
        { V ur = VL(USCR + 112), ui = VL(USCR + 120);
          A7 = VFMA(ur, C10, A7); B7 = VFMA(ui, C10, B7);
          A8 = VFMA(ur, C5, A8); B8 = VFMA(ui, C5, B8);
          A9 = VFMA(ur, C3, A9); B9 = VFMA(ui, C3, B9);
          A10 = VFMA(ur, C11, A10); B10 = VFMA(ui, C11, B10);
          A11 = VFMA(ur, C4, A11); B11 = VFMA(ui, C4, B11);
        }
        { V ur = VL(USCR + 128), ui = VL(USCR + 136);
          A7 = VFMA(ur, C6, A7); B7 = VFMA(ui, C6, B7);
          A8 = VFMA(ur, C3, A8); B8 = VFMA(ui, C3, B8);
          A9 = VFMA(ur, C11, A9); B9 = VFMA(ui, C11, B9);
          A10 = VFMA(ur, C2, A10); B10 = VFMA(ui, C2, B10);
          A11 = VFMA(ur, C7, A11); B11 = VFMA(ui, C7, B11);
        }
        { V ur = VL(USCR + 144), ui = VL(USCR + 152);
          A7 = VFMA(ur, C1, A7); B7 = VFMA(ui, C1, B7);
          A8 = VFMA(ur, C11, A8); B8 = VFMA(ui, C11, B8);
          A9 = VFMA(ur, C2, A9); B9 = VFMA(ui, C2, B9);
          A10 = VFMA(ur, C8, A10); B10 = VFMA(ui, C8, B10);
          A11 = VFMA(ur, C5, A11); B11 = VFMA(ui, C5, B11);
        }
        { V ur = VL(USCR + 160), ui = VL(USCR + 168);
          A7 = VFMA(ur, C8, A7); B7 = VFMA(ui, C8, B7);
          A8 = VFMA(ur, C4, A8); B8 = VFMA(ui, C4, B8);
          A9 = VFMA(ur, C7, A9); B9 = VFMA(ui, C7, B9);
          A10 = VFMA(ur, C5, A10); B10 = VFMA(ui, C5, B10);
          A11 = VFMA(ur, C6, A11); B11 = VFMA(ui, C6, B11);
        }
        { V Dk = VL(PSCR + 96), Ek = VL(PSCR + 104);
          V Xr = VADD(A7, Dk), Xi = VSUB(B7, Ek);
          V Yr = VSUB(A7, Dk), Yi = VADD(B7, Ek);
        VS(pr + 56, Xr); VS(pi + 56, Xi);
        VS(pr + 128, Yr); VS(pi + 128, Yi);
        }
        { V Dk = VL(PSCR + 112), Ek = VL(PSCR + 120);
          V Xr = VADD(A8, Dk), Xi = VSUB(B8, Ek);
          V Yr = VSUB(A8, Dk), Yi = VADD(B8, Ek);
        VS(pr + 64, Xr); VS(pi + 64, Xi);
        VS(pr + 120, Yr); VS(pi + 120, Yi);
        }
        { V Dk = VL(PSCR + 128), Ek = VL(PSCR + 136);
          V Xr = VADD(A9, Dk), Xi = VSUB(B9, Ek);
          V Yr = VSUB(A9, Dk), Yi = VADD(B9, Ek);
        VS(pr + 72, Xr); VS(pi + 72, Xi);
        VS(pr + 112, Yr); VS(pi + 112, Yi);
        }
        { V Dk = VL(PSCR + 144), Ek = VL(PSCR + 152);
          V Xr = VADD(A10, Dk), Xi = VSUB(B10, Ek);
          V Yr = VSUB(A10, Dk), Yi = VADD(B10, Ek);
        VS(pr + 80, Xr); VS(pi + 80, Xi);
        VS(pr + 104, Yr); VS(pi + 104, Yi);
        }
        { V Dk = VL(PSCR + 160), Ek = VL(PSCR + 168);
          V Xr = VADD(A11, Dk), Xi = VSUB(B11, Ek);
          V Yr = VSUB(A11, Dk), Yi = VADD(B11, Ek);
        VS(pr + 88, Xr); VS(pi + 88, Xi);
        VS(pr + 96, Yr); VS(pi + 96, Yi);
        }
        }
    }
}
static __attribute__((noinline)) void p23_yy(double* PR, double* PI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        {
        V S1 = VSET1(0x1.14459ad2be466p-2);
        V S2 = VSET1(0x1.0a06e851db7cap-1);
        V S3 = VSET1(0x1.763021aaa15d9p-1);
        V S4 = VSET1(0x1.c698e42f47b09p-1);
        V S5 = VSET1(0x1.f54a827142577p-1);
        V S6 = VSET1(0x1.fece70dfd3efbp-1);
        V S7 = VSET1(0x1.e270060999288p-1);
        V S8 = VSET1(0x1.a249e0b897caap-1);
        V S9 = VSET1(0x1.431df5838f7f1p-1);
        V S10 = VSET1(0x1.97f6748e524b1p-2);
        V S11 = VSET1(0x1.16de8a4564f1cp-3);
        V D1 = _mm512_setzero_pd(), E1 = _mm512_setzero_pd();
        V D2 = _mm512_setzero_pd(), E2 = _mm512_setzero_pd();
        V D3 = _mm512_setzero_pd(), E3 = _mm512_setzero_pd();
        V D4 = _mm512_setzero_pd(), E4 = _mm512_setzero_pd();
        V D5 = _mm512_setzero_pd(), E5 = _mm512_setzero_pd();
        V D6 = _mm512_setzero_pd(), E6 = _mm512_setzero_pd();
        { V ar = VL(pr + 184), ai = VL(pi + 184);
          V br = VL(pr + 4048), bi = VL(pi + 4048);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 0, VADD(ar,br)); VS(USCR + 8, VADD(ai,bi));
          D1 = VFMA(vi, S1, D1); E1 = VFMA(vr, S1, E1);
          D2 = VFMA(vi, S2, D2); E2 = VFMA(vr, S2, E2);
          D3 = VFMA(vi, S3, D3); E3 = VFMA(vr, S3, E3);
          D4 = VFMA(vi, S4, D4); E4 = VFMA(vr, S4, E4);
          D5 = VFMA(vi, S5, D5); E5 = VFMA(vr, S5, E5);
          D6 = VFMA(vi, S6, D6); E6 = VFMA(vr, S6, E6);
        }
        { V ar = VL(pr + 368), ai = VL(pi + 368);
          V br = VL(pr + 3864), bi = VL(pi + 3864);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 16, VADD(ar,br)); VS(USCR + 24, VADD(ai,bi));
          D1 = VFMA(vi, S2, D1); E1 = VFMA(vr, S2, E1);
          D2 = VFMA(vi, S4, D2); E2 = VFMA(vr, S4, E2);
          D3 = VFMA(vi, S6, D3); E3 = VFMA(vr, S6, E3);
          D4 = VFMA(vi, S8, D4); E4 = VFMA(vr, S8, E4);
          D5 = VFMA(vi, S10, D5); E5 = VFMA(vr, S10, E5);
          D6 = VFNMA(vi, S11, D6); E6 = VFNMA(vr, S11, E6);
        }
        { V ar = VL(pr + 552), ai = VL(pi + 552);
          V br = VL(pr + 3680), bi = VL(pi + 3680);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 32, VADD(ar,br)); VS(USCR + 40, VADD(ai,bi));
          D1 = VFMA(vi, S3, D1); E1 = VFMA(vr, S3, E1);
          D2 = VFMA(vi, S6, D2); E2 = VFMA(vr, S6, E2);
          D3 = VFMA(vi, S9, D3); E3 = VFMA(vr, S9, E3);
          D4 = VFNMA(vi, S11, D4); E4 = VFNMA(vr, S11, E4);
          D5 = VFNMA(vi, S8, D5); E5 = VFNMA(vr, S8, E5);
          D6 = VFNMA(vi, S5, D6); E6 = VFNMA(vr, S5, E6);
        }
        { V ar = VL(pr + 736), ai = VL(pi + 736);
          V br = VL(pr + 3496), bi = VL(pi + 3496);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 48, VADD(ar,br)); VS(USCR + 56, VADD(ai,bi));
          D1 = VFMA(vi, S4, D1); E1 = VFMA(vr, S4, E1);
          D2 = VFMA(vi, S8, D2); E2 = VFMA(vr, S8, E2);
          D3 = VFNMA(vi, S11, D3); E3 = VFNMA(vr, S11, E3);
          D4 = VFNMA(vi, S7, D4); E4 = VFNMA(vr, S7, E4);
          D5 = VFNMA(vi, S3, D5); E5 = VFNMA(vr, S3, E5);
          D6 = VFMA(vi, S1, D6); E6 = VFMA(vr, S1, E6);
        }
        { V ar = VL(pr + 920), ai = VL(pi + 920);
          V br = VL(pr + 3312), bi = VL(pi + 3312);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 64, VADD(ar,br)); VS(USCR + 72, VADD(ai,bi));
          D1 = VFMA(vi, S5, D1); E1 = VFMA(vr, S5, E1);
          D2 = VFMA(vi, S10, D2); E2 = VFMA(vr, S10, E2);
          D3 = VFNMA(vi, S8, D3); E3 = VFNMA(vr, S8, E3);
          D4 = VFNMA(vi, S3, D4); E4 = VFNMA(vr, S3, E4);
          D5 = VFMA(vi, S2, D5); E5 = VFMA(vr, S2, E5);
          D6 = VFMA(vi, S7, D6); E6 = VFMA(vr, S7, E6);
        }
        { V ar = VL(pr + 1104), ai = VL(pi + 1104);
          V br = VL(pr + 3128), bi = VL(pi + 3128);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 80, VADD(ar,br)); VS(USCR + 88, VADD(ai,bi));
          D1 = VFMA(vi, S6, D1); E1 = VFMA(vr, S6, E1);
          D2 = VFNMA(vi, S11, D2); E2 = VFNMA(vr, S11, E2);
          D3 = VFNMA(vi, S5, D3); E3 = VFNMA(vr, S5, E3);
          D4 = VFMA(vi, S1, D4); E4 = VFMA(vr, S1, E4);
          D5 = VFMA(vi, S7, D5); E5 = VFMA(vr, S7, E5);
          D6 = VFNMA(vi, S10, D6); E6 = VFNMA(vr, S10, E6);
        }
        { V ar = VL(pr + 1288), ai = VL(pi + 1288);
          V br = VL(pr + 2944), bi = VL(pi + 2944);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 96, VADD(ar,br)); VS(USCR + 104, VADD(ai,bi));
          D1 = VFMA(vi, S7, D1); E1 = VFMA(vr, S7, E1);
          D2 = VFNMA(vi, S9, D2); E2 = VFNMA(vr, S9, E2);
          D3 = VFNMA(vi, S2, D3); E3 = VFNMA(vr, S2, E3);
          D4 = VFMA(vi, S5, D4); E4 = VFMA(vr, S5, E4);
          D5 = VFNMA(vi, S11, D5); E5 = VFNMA(vr, S11, E5);
          D6 = VFNMA(vi, S4, D6); E6 = VFNMA(vr, S4, E6);
        }
        { V ar = VL(pr + 1472), ai = VL(pi + 1472);
          V br = VL(pr + 2760), bi = VL(pi + 2760);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 112, VADD(ar,br)); VS(USCR + 120, VADD(ai,bi));
          D1 = VFMA(vi, S8, D1); E1 = VFMA(vr, S8, E1);
          D2 = VFNMA(vi, S7, D2); E2 = VFNMA(vr, S7, E2);
          D3 = VFMA(vi, S1, D3); E3 = VFMA(vr, S1, E3);
          D4 = VFMA(vi, S9, D4); E4 = VFMA(vr, S9, E4);
          D5 = VFNMA(vi, S6, D5); E5 = VFNMA(vr, S6, E5);
          D6 = VFMA(vi, S2, D6); E6 = VFMA(vr, S2, E6);
        }
        { V ar = VL(pr + 1656), ai = VL(pi + 1656);
          V br = VL(pr + 2576), bi = VL(pi + 2576);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 128, VADD(ar,br)); VS(USCR + 136, VADD(ai,bi));
          D1 = VFMA(vi, S9, D1); E1 = VFMA(vr, S9, E1);
          D2 = VFNMA(vi, S5, D2); E2 = VFNMA(vr, S5, E2);
          D3 = VFMA(vi, S4, D3); E3 = VFMA(vr, S4, E3);
          D4 = VFNMA(vi, S10, D4); E4 = VFNMA(vr, S10, E4);
          D5 = VFNMA(vi, S1, D5); E5 = VFNMA(vr, S1, E5);
          D6 = VFMA(vi, S8, D6); E6 = VFMA(vr, S8, E6);
        }
        { V ar = VL(pr + 1840), ai = VL(pi + 1840);
          V br = VL(pr + 2392), bi = VL(pi + 2392);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 144, VADD(ar,br)); VS(USCR + 152, VADD(ai,bi));
          D1 = VFMA(vi, S10, D1); E1 = VFMA(vr, S10, E1);
          D2 = VFNMA(vi, S3, D2); E2 = VFNMA(vr, S3, E2);
          D3 = VFMA(vi, S7, D3); E3 = VFMA(vr, S7, E3);
          D4 = VFNMA(vi, S6, D4); E4 = VFNMA(vr, S6, E4);
          D5 = VFMA(vi, S4, D5); E5 = VFMA(vr, S4, E5);
          D6 = VFNMA(vi, S9, D6); E6 = VFNMA(vr, S9, E6);
        }
        { V ar = VL(pr + 2024), ai = VL(pi + 2024);
          V br = VL(pr + 2208), bi = VL(pi + 2208);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 160, VADD(ar,br)); VS(USCR + 168, VADD(ai,bi));
          D1 = VFMA(vi, S11, D1); E1 = VFMA(vr, S11, E1);
          D2 = VFNMA(vi, S1, D2); E2 = VFNMA(vr, S1, E2);
          D3 = VFMA(vi, S10, D3); E3 = VFMA(vr, S10, E3);
          D4 = VFNMA(vi, S2, D4); E4 = VFNMA(vr, S2, E4);
          D5 = VFMA(vi, S9, D5); E5 = VFMA(vr, S9, E5);
          D6 = VFNMA(vi, S3, D6); E6 = VFNMA(vr, S3, E6);
        }
        VS(PSCR + 0, D1); VS(PSCR + 8, E1);
        VS(PSCR + 16, D2); VS(PSCR + 24, E2);
        VS(PSCR + 32, D3); VS(PSCR + 40, E3);
        VS(PSCR + 48, D4); VS(PSCR + 56, E4);
        VS(PSCR + 64, D5); VS(PSCR + 72, E5);
        VS(PSCR + 80, D6); VS(PSCR + 88, E6);
        }
        {
        V S1 = VSET1(0x1.14459ad2be466p-2);
        V S2 = VSET1(0x1.0a06e851db7cap-1);
        V S3 = VSET1(0x1.763021aaa15d9p-1);
        V S4 = VSET1(0x1.c698e42f47b09p-1);
        V S5 = VSET1(0x1.f54a827142577p-1);
        V S6 = VSET1(0x1.fece70dfd3efbp-1);
        V S7 = VSET1(0x1.e270060999288p-1);
        V S8 = VSET1(0x1.a249e0b897caap-1);
        V S9 = VSET1(0x1.431df5838f7f1p-1);
        V S10 = VSET1(0x1.97f6748e524b1p-2);
        V S11 = VSET1(0x1.16de8a4564f1cp-3);
        V D7 = _mm512_setzero_pd(), E7 = _mm512_setzero_pd();
        V D8 = _mm512_setzero_pd(), E8 = _mm512_setzero_pd();
        V D9 = _mm512_setzero_pd(), E9 = _mm512_setzero_pd();
        V D10 = _mm512_setzero_pd(), E10 = _mm512_setzero_pd();
        V D11 = _mm512_setzero_pd(), E11 = _mm512_setzero_pd();
        { V ar = VL(pr + 184), ai = VL(pi + 184);
          V br = VL(pr + 4048), bi = VL(pi + 4048);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S7, D7); E7 = VFMA(vr, S7, E7);
          D8 = VFMA(vi, S8, D8); E8 = VFMA(vr, S8, E8);
          D9 = VFMA(vi, S9, D9); E9 = VFMA(vr, S9, E9);
          D10 = VFMA(vi, S10, D10); E10 = VFMA(vr, S10, E10);
          D11 = VFMA(vi, S11, D11); E11 = VFMA(vr, S11, E11);
        }
        { V ar = VL(pr + 368), ai = VL(pi + 368);
          V br = VL(pr + 3864), bi = VL(pi + 3864);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S9, D7); E7 = VFNMA(vr, S9, E7);
          D8 = VFNMA(vi, S7, D8); E8 = VFNMA(vr, S7, E8);
          D9 = VFNMA(vi, S5, D9); E9 = VFNMA(vr, S5, E9);
          D10 = VFNMA(vi, S3, D10); E10 = VFNMA(vr, S3, E10);
          D11 = VFNMA(vi, S1, D11); E11 = VFNMA(vr, S1, E11);
        }
        { V ar = VL(pr + 552), ai = VL(pi + 552);
          V br = VL(pr + 3680), bi = VL(pi + 3680);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S2, D7); E7 = VFNMA(vr, S2, E7);
          D8 = VFMA(vi, S1, D8); E8 = VFMA(vr, S1, E8);
          D9 = VFMA(vi, S4, D9); E9 = VFMA(vr, S4, E9);
          D10 = VFMA(vi, S7, D10); E10 = VFMA(vr, S7, E10);
          D11 = VFMA(vi, S10, D11); E11 = VFMA(vr, S10, E11);
        }
        { V ar = VL(pr + 736), ai = VL(pi + 736);
          V br = VL(pr + 3496), bi = VL(pi + 3496);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S5, D7); E7 = VFMA(vr, S5, E7);
          D8 = VFMA(vi, S9, D8); E8 = VFMA(vr, S9, E8);
          D9 = VFNMA(vi, S10, D9); E9 = VFNMA(vr, S10, E9);
          D10 = VFNMA(vi, S6, D10); E10 = VFNMA(vr, S6, E10);
          D11 = VFNMA(vi, S2, D11); E11 = VFNMA(vr, S2, E11);
        }
        { V ar = VL(pr + 920), ai = VL(pi + 920);
          V br = VL(pr + 3312), bi = VL(pi + 3312);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S11, D7); E7 = VFNMA(vr, S11, E7);
          D8 = VFNMA(vi, S6, D8); E8 = VFNMA(vr, S6, E8);
          D9 = VFNMA(vi, S1, D9); E9 = VFNMA(vr, S1, E9);
          D10 = VFMA(vi, S4, D10); E10 = VFMA(vr, S4, E10);
          D11 = VFMA(vi, S9, D11); E11 = VFMA(vr, S9, E11);
        }
        { V ar = VL(pr + 1104), ai = VL(pi + 1104);
          V br = VL(pr + 3128), bi = VL(pi + 3128);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S4, D7); E7 = VFNMA(vr, S4, E7);
          D8 = VFMA(vi, S2, D8); E8 = VFMA(vr, S2, E8);
          D9 = VFMA(vi, S8, D9); E9 = VFMA(vr, S8, E9);
          D10 = VFNMA(vi, S9, D10); E10 = VFNMA(vr, S9, E10);
          D11 = VFNMA(vi, S3, D11); E11 = VFNMA(vr, S3, E11);
        }
        { V ar = VL(pr + 1288), ai = VL(pi + 1288);
          V br = VL(pr + 2944), bi = VL(pi + 2944);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S3, D7); E7 = VFMA(vr, S3, E7);
          D8 = VFMA(vi, S10, D8); E8 = VFMA(vr, S10, E8);
          D9 = VFNMA(vi, S6, D9); E9 = VFNMA(vr, S6, E9);
          D10 = VFMA(vi, S1, D10); E10 = VFMA(vr, S1, E10);
          D11 = VFMA(vi, S8, D11); E11 = VFMA(vr, S8, E11);
        }
        { V ar = VL(pr + 1472), ai = VL(pi + 1472);
          V br = VL(pr + 2760), bi = VL(pi + 2760);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S10, D7); E7 = VFMA(vr, S10, E7);
          D8 = VFNMA(vi, S5, D8); E8 = VFNMA(vr, S5, E8);
          D9 = VFMA(vi, S3, D9); E9 = VFMA(vr, S3, E9);
          D10 = VFMA(vi, S11, D10); E10 = VFMA(vr, S11, E10);
          D11 = VFNMA(vi, S4, D11); E11 = VFNMA(vr, S4, E11);
        }
        { V ar = VL(pr + 1656), ai = VL(pi + 1656);
          V br = VL(pr + 2576), bi = VL(pi + 2576);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S6, D7); E7 = VFNMA(vr, S6, E7);
          D8 = VFMA(vi, S3, D8); E8 = VFMA(vr, S3, E8);
          D9 = VFNMA(vi, S11, D9); E9 = VFNMA(vr, S11, E9);
          D10 = VFNMA(vi, S2, D10); E10 = VFNMA(vr, S2, E10);
          D11 = VFMA(vi, S7, D11); E11 = VFMA(vr, S7, E11);
        }
        { V ar = VL(pr + 1840), ai = VL(pi + 1840);
          V br = VL(pr + 2392), bi = VL(pi + 2392);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S1, D7); E7 = VFMA(vr, S1, E7);
          D8 = VFMA(vi, S11, D8); E8 = VFMA(vr, S11, E8);
          D9 = VFNMA(vi, S2, D9); E9 = VFNMA(vr, S2, E9);
          D10 = VFMA(vi, S8, D10); E10 = VFMA(vr, S8, E10);
          D11 = VFNMA(vi, S5, D11); E11 = VFNMA(vr, S5, E11);
        }
        { V ar = VL(pr + 2024), ai = VL(pi + 2024);
          V br = VL(pr + 2208), bi = VL(pi + 2208);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S8, D7); E7 = VFMA(vr, S8, E7);
          D8 = VFNMA(vi, S4, D8); E8 = VFNMA(vr, S4, E8);
          D9 = VFMA(vi, S7, D9); E9 = VFMA(vr, S7, E9);
          D10 = VFNMA(vi, S5, D10); E10 = VFNMA(vr, S5, E10);
          D11 = VFMA(vi, S6, D11); E11 = VFMA(vr, S6, E11);
        }
        VS(PSCR + 96, D7); VS(PSCR + 104, E7);
        VS(PSCR + 112, D8); VS(PSCR + 120, E8);
        VS(PSCR + 128, D9); VS(PSCR + 136, E9);
        VS(PSCR + 144, D10); VS(PSCR + 152, E10);
        VS(PSCR + 160, D11); VS(PSCR + 168, E11);
        }
        VS(USCR + 384, VL(pr)); VS(USCR + 392, VL(pi));  // save x0 (row 0 gets overwritten)
        {
        V C1 = VSET1(0x1.ed037ea3d2dbcp-1);
        V C2 = VSET1(0x1.b57675cf309eep-1);
        V C3 = VSET1(0x1.5d779b07cfef7p-1);
        V C4 = VSET1(0x1.d71b4a0c5a6c9p-2);
        V C5 = VSET1(0x1.a0ad8bd1e2881p-3);
        V C6 = VSET1(-0x1.17855b599f3b2p-4);
        V C7 = VSET1(-0x1.56eaae597c776p-2);
        V C8 = VSET1(-0x1.2742a4a775cfap-1);
        V C9 = VSET1(-0x1.8d2a07c16d46ep-1);
        V C10 = VSET1(-0x1.d59cb83ef99bcp-1);
        V C11 = VSET1(-0x1.fb3b3035aa6ccp-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A1 = x0r, B1 = x0i;
        V A2 = x0r, B2 = x0i;
        V A3 = x0r, B3 = x0i;
        V A4 = x0r, B4 = x0i;
        V A5 = x0r, B5 = x0i;
        V A6 = x0r, B6 = x0i;
        V s0r = x0r, s0i = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A1 = VFMA(ur, C1, A1); B1 = VFMA(ui, C1, B1);
          A2 = VFMA(ur, C2, A2); B2 = VFMA(ui, C2, B2);
          A3 = VFMA(ur, C3, A3); B3 = VFMA(ui, C3, B3);
          A4 = VFMA(ur, C4, A4); B4 = VFMA(ui, C4, B4);
          A5 = VFMA(ur, C5, A5); B5 = VFMA(ui, C5, B5);
          A6 = VFMA(ur, C6, A6); B6 = VFMA(ui, C6, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A1 = VFMA(ur, C2, A1); B1 = VFMA(ui, C2, B1);
          A2 = VFMA(ur, C4, A2); B2 = VFMA(ui, C4, B2);
          A3 = VFMA(ur, C6, A3); B3 = VFMA(ui, C6, B3);
          A4 = VFMA(ur, C8, A4); B4 = VFMA(ui, C8, B4);
          A5 = VFMA(ur, C10, A5); B5 = VFMA(ui, C10, B5);
          A6 = VFMA(ur, C11, A6); B6 = VFMA(ui, C11, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A1 = VFMA(ur, C3, A1); B1 = VFMA(ui, C3, B1);
          A2 = VFMA(ur, C6, A2); B2 = VFMA(ui, C6, B2);
          A3 = VFMA(ur, C9, A3); B3 = VFMA(ui, C9, B3);
          A4 = VFMA(ur, C11, A4); B4 = VFMA(ui, C11, B4);
          A5 = VFMA(ur, C8, A5); B5 = VFMA(ui, C8, B5);
          A6 = VFMA(ur, C5, A6); B6 = VFMA(ui, C5, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A1 = VFMA(ur, C4, A1); B1 = VFMA(ui, C4, B1);
          A2 = VFMA(ur, C8, A2); B2 = VFMA(ui, C8, B2);
          A3 = VFMA(ur, C11, A3); B3 = VFMA(ui, C11, B3);
          A4 = VFMA(ur, C7, A4); B4 = VFMA(ui, C7, B4);
          A5 = VFMA(ur, C3, A5); B5 = VFMA(ui, C3, B5);
          A6 = VFMA(ur, C1, A6); B6 = VFMA(ui, C1, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A1 = VFMA(ur, C5, A1); B1 = VFMA(ui, C5, B1);
          A2 = VFMA(ur, C10, A2); B2 = VFMA(ui, C10, B2);
          A3 = VFMA(ur, C8, A3); B3 = VFMA(ui, C8, B3);
          A4 = VFMA(ur, C3, A4); B4 = VFMA(ui, C3, B4);
          A5 = VFMA(ur, C2, A5); B5 = VFMA(ui, C2, B5);
          A6 = VFMA(ur, C7, A6); B6 = VFMA(ui, C7, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A1 = VFMA(ur, C6, A1); B1 = VFMA(ui, C6, B1);
          A2 = VFMA(ur, C11, A2); B2 = VFMA(ui, C11, B2);
          A3 = VFMA(ur, C5, A3); B3 = VFMA(ui, C5, B3);
          A4 = VFMA(ur, C1, A4); B4 = VFMA(ui, C1, B4);
          A5 = VFMA(ur, C7, A5); B5 = VFMA(ui, C7, B5);
          A6 = VFMA(ur, C10, A6); B6 = VFMA(ui, C10, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 96), ui = VL(USCR + 104);
          A1 = VFMA(ur, C7, A1); B1 = VFMA(ui, C7, B1);
          A2 = VFMA(ur, C9, A2); B2 = VFMA(ui, C9, B2);
          A3 = VFMA(ur, C2, A3); B3 = VFMA(ui, C2, B3);
          A4 = VFMA(ur, C5, A4); B4 = VFMA(ui, C5, B4);
          A5 = VFMA(ur, C11, A5); B5 = VFMA(ui, C11, B5);
          A6 = VFMA(ur, C4, A6); B6 = VFMA(ui, C4, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 112), ui = VL(USCR + 120);
          A1 = VFMA(ur, C8, A1); B1 = VFMA(ui, C8, B1);
          A2 = VFMA(ur, C7, A2); B2 = VFMA(ui, C7, B2);
          A3 = VFMA(ur, C1, A3); B3 = VFMA(ui, C1, B3);
          A4 = VFMA(ur, C9, A4); B4 = VFMA(ui, C9, B4);
          A5 = VFMA(ur, C6, A5); B5 = VFMA(ui, C6, B5);
          A6 = VFMA(ur, C2, A6); B6 = VFMA(ui, C2, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 128), ui = VL(USCR + 136);
          A1 = VFMA(ur, C9, A1); B1 = VFMA(ui, C9, B1);
          A2 = VFMA(ur, C5, A2); B2 = VFMA(ui, C5, B2);
          A3 = VFMA(ur, C4, A3); B3 = VFMA(ui, C4, B3);
          A4 = VFMA(ur, C10, A4); B4 = VFMA(ui, C10, B4);
          A5 = VFMA(ur, C1, A5); B5 = VFMA(ui, C1, B5);
          A6 = VFMA(ur, C8, A6); B6 = VFMA(ui, C8, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 144), ui = VL(USCR + 152);
          A1 = VFMA(ur, C10, A1); B1 = VFMA(ui, C10, B1);
          A2 = VFMA(ur, C3, A2); B2 = VFMA(ui, C3, B2);
          A3 = VFMA(ur, C7, A3); B3 = VFMA(ui, C7, B3);
          A4 = VFMA(ur, C6, A4); B4 = VFMA(ui, C6, B4);
          A5 = VFMA(ur, C4, A5); B5 = VFMA(ui, C4, B5);
          A6 = VFMA(ur, C9, A6); B6 = VFMA(ui, C9, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 160), ui = VL(USCR + 168);
          A1 = VFMA(ur, C11, A1); B1 = VFMA(ui, C11, B1);
          A2 = VFMA(ur, C1, A2); B2 = VFMA(ui, C1, B2);
          A3 = VFMA(ur, C10, A3); B3 = VFMA(ui, C10, B3);
          A4 = VFMA(ur, C2, A4); B4 = VFMA(ui, C2, B4);
          A5 = VFMA(ur, C9, A5); B5 = VFMA(ui, C9, B5);
          A6 = VFMA(ur, C3, A6); B6 = VFMA(ui, C3, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        VS(pr + 0, s0r); VS(pi + 0, s0i);
        { V Dk = VL(PSCR + 0), Ek = VL(PSCR + 8);
          V Xr = VADD(A1, Dk), Xi = VSUB(B1, Ek);
          V Yr = VSUB(A1, Dk), Yi = VADD(B1, Ek);
        VS(pr + 184, Xr); VS(pi + 184, Xi);
        VS(pr + 4048, Yr); VS(pi + 4048, Yi);
        }
        { V Dk = VL(PSCR + 16), Ek = VL(PSCR + 24);
          V Xr = VADD(A2, Dk), Xi = VSUB(B2, Ek);
          V Yr = VSUB(A2, Dk), Yi = VADD(B2, Ek);
        VS(pr + 368, Xr); VS(pi + 368, Xi);
        VS(pr + 3864, Yr); VS(pi + 3864, Yi);
        }
        { V Dk = VL(PSCR + 32), Ek = VL(PSCR + 40);
          V Xr = VADD(A3, Dk), Xi = VSUB(B3, Ek);
          V Yr = VSUB(A3, Dk), Yi = VADD(B3, Ek);
        VS(pr + 552, Xr); VS(pi + 552, Xi);
        VS(pr + 3680, Yr); VS(pi + 3680, Yi);
        }
        { V Dk = VL(PSCR + 48), Ek = VL(PSCR + 56);
          V Xr = VADD(A4, Dk), Xi = VSUB(B4, Ek);
          V Yr = VSUB(A4, Dk), Yi = VADD(B4, Ek);
        VS(pr + 736, Xr); VS(pi + 736, Xi);
        VS(pr + 3496, Yr); VS(pi + 3496, Yi);
        }
        { V Dk = VL(PSCR + 64), Ek = VL(PSCR + 72);
          V Xr = VADD(A5, Dk), Xi = VSUB(B5, Ek);
          V Yr = VSUB(A5, Dk), Yi = VADD(B5, Ek);
        VS(pr + 920, Xr); VS(pi + 920, Xi);
        VS(pr + 3312, Yr); VS(pi + 3312, Yi);
        }
        { V Dk = VL(PSCR + 80), Ek = VL(PSCR + 88);
          V Xr = VADD(A6, Dk), Xi = VSUB(B6, Ek);
          V Yr = VSUB(A6, Dk), Yi = VADD(B6, Ek);
        VS(pr + 1104, Xr); VS(pi + 1104, Xi);
        VS(pr + 3128, Yr); VS(pi + 3128, Yi);
        }
        }
        {
        V C1 = VSET1(0x1.ed037ea3d2dbcp-1);
        V C2 = VSET1(0x1.b57675cf309eep-1);
        V C3 = VSET1(0x1.5d779b07cfef7p-1);
        V C4 = VSET1(0x1.d71b4a0c5a6c9p-2);
        V C5 = VSET1(0x1.a0ad8bd1e2881p-3);
        V C6 = VSET1(-0x1.17855b599f3b2p-4);
        V C7 = VSET1(-0x1.56eaae597c776p-2);
        V C8 = VSET1(-0x1.2742a4a775cfap-1);
        V C9 = VSET1(-0x1.8d2a07c16d46ep-1);
        V C10 = VSET1(-0x1.d59cb83ef99bcp-1);
        V C11 = VSET1(-0x1.fb3b3035aa6ccp-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A7 = x0r, B7 = x0i;
        V A8 = x0r, B8 = x0i;
        V A9 = x0r, B9 = x0i;
        V A10 = x0r, B10 = x0i;
        V A11 = x0r, B11 = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A7 = VFMA(ur, C7, A7); B7 = VFMA(ui, C7, B7);
          A8 = VFMA(ur, C8, A8); B8 = VFMA(ui, C8, B8);
          A9 = VFMA(ur, C9, A9); B9 = VFMA(ui, C9, B9);
          A10 = VFMA(ur, C10, A10); B10 = VFMA(ui, C10, B10);
          A11 = VFMA(ur, C11, A11); B11 = VFMA(ui, C11, B11);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A7 = VFMA(ur, C9, A7); B7 = VFMA(ui, C9, B7);
          A8 = VFMA(ur, C7, A8); B8 = VFMA(ui, C7, B8);
          A9 = VFMA(ur, C5, A9); B9 = VFMA(ui, C5, B9);
          A10 = VFMA(ur, C3, A10); B10 = VFMA(ui, C3, B10);
          A11 = VFMA(ur, C1, A11); B11 = VFMA(ui, C1, B11);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A7 = VFMA(ur, C2, A7); B7 = VFMA(ui, C2, B7);
          A8 = VFMA(ur, C1, A8); B8 = VFMA(ui, C1, B8);
          A9 = VFMA(ur, C4, A9); B9 = VFMA(ui, C4, B9);
          A10 = VFMA(ur, C7, A10); B10 = VFMA(ui, C7, B10);
          A11 = VFMA(ur, C10, A11); B11 = VFMA(ui, C10, B11);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A7 = VFMA(ur, C5, A7); B7 = VFMA(ui, C5, B7);
          A8 = VFMA(ur, C9, A8); B8 = VFMA(ui, C9, B8);
          A9 = VFMA(ur, C10, A9); B9 = VFMA(ui, C10, B9);
          A10 = VFMA(ur, C6, A10); B10 = VFMA(ui, C6, B10);
          A11 = VFMA(ur, C2, A11); B11 = VFMA(ui, C2, B11);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A7 = VFMA(ur, C11, A7); B7 = VFMA(ui, C11, B7);
          A8 = VFMA(ur, C6, A8); B8 = VFMA(ui, C6, B8);
          A9 = VFMA(ur, C1, A9); B9 = VFMA(ui, C1, B9);
          A10 = VFMA(ur, C4, A10); B10 = VFMA(ui, C4, B10);
          A11 = VFMA(ur, C9, A11); B11 = VFMA(ui, C9, B11);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A7 = VFMA(ur, C4, A7); B7 = VFMA(ui, C4, B7);
          A8 = VFMA(ur, C2, A8); B8 = VFMA(ui, C2, B8);
          A9 = VFMA(ur, C8, A9); B9 = VFMA(ui, C8, B9);
          A10 = VFMA(ur, C9, A10); B10 = VFMA(ui, C9, B10);
          A11 = VFMA(ur, C3, A11); B11 = VFMA(ui, C3, B11);
        }
        { V ur = VL(USCR + 96), ui = VL(USCR + 104);
          A7 = VFMA(ur, C3, A7); B7 = VFMA(ui, C3, B7);
          A8 = VFMA(ur, C10, A8); B8 = VFMA(ui, C10, B8);
          A9 = VFMA(ur, C6, A9); B9 = VFMA(ui, C6, B9);
          A10 = VFMA(ur, C1, A10); B10 = VFMA(ui, C1, B10);
          A11 = VFMA(ur, C8, A11); B11 = VFMA(ui, C8, B11);
        }
        { V ur = VL(USCR + 112), ui = VL(USCR + 120);
          A7 = VFMA(ur, C10, A7); B7 = VFMA(ui, C10, B7);
          A8 = VFMA(ur, C5, A8); B8 = VFMA(ui, C5, B8);
          A9 = VFMA(ur, C3, A9); B9 = VFMA(ui, C3, B9);
          A10 = VFMA(ur, C11, A10); B10 = VFMA(ui, C11, B10);
          A11 = VFMA(ur, C4, A11); B11 = VFMA(ui, C4, B11);
        }
        { V ur = VL(USCR + 128), ui = VL(USCR + 136);
          A7 = VFMA(ur, C6, A7); B7 = VFMA(ui, C6, B7);
          A8 = VFMA(ur, C3, A8); B8 = VFMA(ui, C3, B8);
          A9 = VFMA(ur, C11, A9); B9 = VFMA(ui, C11, B9);
          A10 = VFMA(ur, C2, A10); B10 = VFMA(ui, C2, B10);
          A11 = VFMA(ur, C7, A11); B11 = VFMA(ui, C7, B11);
        }
        { V ur = VL(USCR + 144), ui = VL(USCR + 152);
          A7 = VFMA(ur, C1, A7); B7 = VFMA(ui, C1, B7);
          A8 = VFMA(ur, C11, A8); B8 = VFMA(ui, C11, B8);
          A9 = VFMA(ur, C2, A9); B9 = VFMA(ui, C2, B9);
          A10 = VFMA(ur, C8, A10); B10 = VFMA(ui, C8, B10);
          A11 = VFMA(ur, C5, A11); B11 = VFMA(ui, C5, B11);
        }
        { V ur = VL(USCR + 160), ui = VL(USCR + 168);
          A7 = VFMA(ur, C8, A7); B7 = VFMA(ui, C8, B7);
          A8 = VFMA(ur, C4, A8); B8 = VFMA(ui, C4, B8);
          A9 = VFMA(ur, C7, A9); B9 = VFMA(ui, C7, B9);
          A10 = VFMA(ur, C5, A10); B10 = VFMA(ui, C5, B10);
          A11 = VFMA(ur, C6, A11); B11 = VFMA(ui, C6, B11);
        }
        { V Dk = VL(PSCR + 96), Ek = VL(PSCR + 104);
          V Xr = VADD(A7, Dk), Xi = VSUB(B7, Ek);
          V Yr = VSUB(A7, Dk), Yi = VADD(B7, Ek);
        VS(pr + 1288, Xr); VS(pi + 1288, Xi);
        VS(pr + 2944, Yr); VS(pi + 2944, Yi);
        }
        { V Dk = VL(PSCR + 112), Ek = VL(PSCR + 120);
          V Xr = VADD(A8, Dk), Xi = VSUB(B8, Ek);
          V Yr = VSUB(A8, Dk), Yi = VADD(B8, Ek);
        VS(pr + 1472, Xr); VS(pi + 1472, Xi);
        VS(pr + 2760, Yr); VS(pi + 2760, Yi);
        }
        { V Dk = VL(PSCR + 128), Ek = VL(PSCR + 136);
          V Xr = VADD(A9, Dk), Xi = VSUB(B9, Ek);
          V Yr = VSUB(A9, Dk), Yi = VADD(B9, Ek);
        VS(pr + 1656, Xr); VS(pi + 1656, Xi);
        VS(pr + 2576, Yr); VS(pi + 2576, Yi);
        }
        { V Dk = VL(PSCR + 144), Ek = VL(PSCR + 152);
          V Xr = VADD(A10, Dk), Xi = VSUB(B10, Ek);
          V Yr = VSUB(A10, Dk), Yi = VADD(B10, Ek);
        VS(pr + 1840, Xr); VS(pi + 1840, Xi);
        VS(pr + 2392, Yr); VS(pi + 2392, Yi);
        }
        { V Dk = VL(PSCR + 160), Ek = VL(PSCR + 168);
          V Xr = VADD(A11, Dk), Xi = VSUB(B11, Ek);
          V Yr = VSUB(A11, Dk), Yi = VADD(B11, Ek);
        VS(pr + 2024, Xr); VS(pi + 2024, Xi);
        VS(pr + 2208, Yr); VS(pi + 2208, Yi);
        }
        }
    }
}
static __attribute__((noinline)) void p23_xx(double* PR, double* PI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        {
        V S1 = VSET1(0x1.14459ad2be466p-2);
        V S2 = VSET1(0x1.0a06e851db7cap-1);
        V S3 = VSET1(0x1.763021aaa15d9p-1);
        V S4 = VSET1(0x1.c698e42f47b09p-1);
        V S5 = VSET1(0x1.f54a827142577p-1);
        V S6 = VSET1(0x1.fece70dfd3efbp-1);
        V S7 = VSET1(0x1.e270060999288p-1);
        V S8 = VSET1(0x1.a249e0b897caap-1);
        V S9 = VSET1(0x1.431df5838f7f1p-1);
        V S10 = VSET1(0x1.97f6748e524b1p-2);
        V S11 = VSET1(0x1.16de8a4564f1cp-3);
        V D1 = _mm512_setzero_pd(), E1 = _mm512_setzero_pd();
        V D2 = _mm512_setzero_pd(), E2 = _mm512_setzero_pd();
        V D3 = _mm512_setzero_pd(), E3 = _mm512_setzero_pd();
        V D4 = _mm512_setzero_pd(), E4 = _mm512_setzero_pd();
        V D5 = _mm512_setzero_pd(), E5 = _mm512_setzero_pd();
        V D6 = _mm512_setzero_pd(), E6 = _mm512_setzero_pd();
        { V ar = VL(pr + 4232), ai = VL(pi + 4232);
          V br = VL(pr + 93104), bi = VL(pi + 93104);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 0, VADD(ar,br)); VS(USCR + 8, VADD(ai,bi));
          D1 = VFMA(vi, S1, D1); E1 = VFMA(vr, S1, E1);
          D2 = VFMA(vi, S2, D2); E2 = VFMA(vr, S2, E2);
          D3 = VFMA(vi, S3, D3); E3 = VFMA(vr, S3, E3);
          D4 = VFMA(vi, S4, D4); E4 = VFMA(vr, S4, E4);
          D5 = VFMA(vi, S5, D5); E5 = VFMA(vr, S5, E5);
          D6 = VFMA(vi, S6, D6); E6 = VFMA(vr, S6, E6);
        }
        { V ar = VL(pr + 8464), ai = VL(pi + 8464);
          V br = VL(pr + 88872), bi = VL(pi + 88872);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 16, VADD(ar,br)); VS(USCR + 24, VADD(ai,bi));
          D1 = VFMA(vi, S2, D1); E1 = VFMA(vr, S2, E1);
          D2 = VFMA(vi, S4, D2); E2 = VFMA(vr, S4, E2);
          D3 = VFMA(vi, S6, D3); E3 = VFMA(vr, S6, E3);
          D4 = VFMA(vi, S8, D4); E4 = VFMA(vr, S8, E4);
          D5 = VFMA(vi, S10, D5); E5 = VFMA(vr, S10, E5);
          D6 = VFNMA(vi, S11, D6); E6 = VFNMA(vr, S11, E6);
        }
        { V ar = VL(pr + 12696), ai = VL(pi + 12696);
          V br = VL(pr + 84640), bi = VL(pi + 84640);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 32, VADD(ar,br)); VS(USCR + 40, VADD(ai,bi));
          D1 = VFMA(vi, S3, D1); E1 = VFMA(vr, S3, E1);
          D2 = VFMA(vi, S6, D2); E2 = VFMA(vr, S6, E2);
          D3 = VFMA(vi, S9, D3); E3 = VFMA(vr, S9, E3);
          D4 = VFNMA(vi, S11, D4); E4 = VFNMA(vr, S11, E4);
          D5 = VFNMA(vi, S8, D5); E5 = VFNMA(vr, S8, E5);
          D6 = VFNMA(vi, S5, D6); E6 = VFNMA(vr, S5, E6);
        }
        { V ar = VL(pr + 16928), ai = VL(pi + 16928);
          V br = VL(pr + 80408), bi = VL(pi + 80408);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 48, VADD(ar,br)); VS(USCR + 56, VADD(ai,bi));
          D1 = VFMA(vi, S4, D1); E1 = VFMA(vr, S4, E1);
          D2 = VFMA(vi, S8, D2); E2 = VFMA(vr, S8, E2);
          D3 = VFNMA(vi, S11, D3); E3 = VFNMA(vr, S11, E3);
          D4 = VFNMA(vi, S7, D4); E4 = VFNMA(vr, S7, E4);
          D5 = VFNMA(vi, S3, D5); E5 = VFNMA(vr, S3, E5);
          D6 = VFMA(vi, S1, D6); E6 = VFMA(vr, S1, E6);
        }
        { V ar = VL(pr + 21160), ai = VL(pi + 21160);
          V br = VL(pr + 76176), bi = VL(pi + 76176);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 64, VADD(ar,br)); VS(USCR + 72, VADD(ai,bi));
          D1 = VFMA(vi, S5, D1); E1 = VFMA(vr, S5, E1);
          D2 = VFMA(vi, S10, D2); E2 = VFMA(vr, S10, E2);
          D3 = VFNMA(vi, S8, D3); E3 = VFNMA(vr, S8, E3);
          D4 = VFNMA(vi, S3, D4); E4 = VFNMA(vr, S3, E4);
          D5 = VFMA(vi, S2, D5); E5 = VFMA(vr, S2, E5);
          D6 = VFMA(vi, S7, D6); E6 = VFMA(vr, S7, E6);
        }
        { V ar = VL(pr + 25392), ai = VL(pi + 25392);
          V br = VL(pr + 71944), bi = VL(pi + 71944);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 80, VADD(ar,br)); VS(USCR + 88, VADD(ai,bi));
          D1 = VFMA(vi, S6, D1); E1 = VFMA(vr, S6, E1);
          D2 = VFNMA(vi, S11, D2); E2 = VFNMA(vr, S11, E2);
          D3 = VFNMA(vi, S5, D3); E3 = VFNMA(vr, S5, E3);
          D4 = VFMA(vi, S1, D4); E4 = VFMA(vr, S1, E4);
          D5 = VFMA(vi, S7, D5); E5 = VFMA(vr, S7, E5);
          D6 = VFNMA(vi, S10, D6); E6 = VFNMA(vr, S10, E6);
        }
        { V ar = VL(pr + 29624), ai = VL(pi + 29624);
          V br = VL(pr + 67712), bi = VL(pi + 67712);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 96, VADD(ar,br)); VS(USCR + 104, VADD(ai,bi));
          D1 = VFMA(vi, S7, D1); E1 = VFMA(vr, S7, E1);
          D2 = VFNMA(vi, S9, D2); E2 = VFNMA(vr, S9, E2);
          D3 = VFNMA(vi, S2, D3); E3 = VFNMA(vr, S2, E3);
          D4 = VFMA(vi, S5, D4); E4 = VFMA(vr, S5, E4);
          D5 = VFNMA(vi, S11, D5); E5 = VFNMA(vr, S11, E5);
          D6 = VFNMA(vi, S4, D6); E6 = VFNMA(vr, S4, E6);
        }
        { V ar = VL(pr + 33856), ai = VL(pi + 33856);
          V br = VL(pr + 63480), bi = VL(pi + 63480);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 112, VADD(ar,br)); VS(USCR + 120, VADD(ai,bi));
          D1 = VFMA(vi, S8, D1); E1 = VFMA(vr, S8, E1);
          D2 = VFNMA(vi, S7, D2); E2 = VFNMA(vr, S7, E2);
          D3 = VFMA(vi, S1, D3); E3 = VFMA(vr, S1, E3);
          D4 = VFMA(vi, S9, D4); E4 = VFMA(vr, S9, E4);
          D5 = VFNMA(vi, S6, D5); E5 = VFNMA(vr, S6, E5);
          D6 = VFMA(vi, S2, D6); E6 = VFMA(vr, S2, E6);
        }
        { V ar = VL(pr + 38088), ai = VL(pi + 38088);
          V br = VL(pr + 59248), bi = VL(pi + 59248);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 128, VADD(ar,br)); VS(USCR + 136, VADD(ai,bi));
          D1 = VFMA(vi, S9, D1); E1 = VFMA(vr, S9, E1);
          D2 = VFNMA(vi, S5, D2); E2 = VFNMA(vr, S5, E2);
          D3 = VFMA(vi, S4, D3); E3 = VFMA(vr, S4, E3);
          D4 = VFNMA(vi, S10, D4); E4 = VFNMA(vr, S10, E4);
          D5 = VFNMA(vi, S1, D5); E5 = VFNMA(vr, S1, E5);
          D6 = VFMA(vi, S8, D6); E6 = VFMA(vr, S8, E6);
        }
        { V ar = VL(pr + 42320), ai = VL(pi + 42320);
          V br = VL(pr + 55016), bi = VL(pi + 55016);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 144, VADD(ar,br)); VS(USCR + 152, VADD(ai,bi));
          D1 = VFMA(vi, S10, D1); E1 = VFMA(vr, S10, E1);
          D2 = VFNMA(vi, S3, D2); E2 = VFNMA(vr, S3, E2);
          D3 = VFMA(vi, S7, D3); E3 = VFMA(vr, S7, E3);
          D4 = VFNMA(vi, S6, D4); E4 = VFNMA(vr, S6, E4);
          D5 = VFMA(vi, S4, D5); E5 = VFMA(vr, S4, E5);
          D6 = VFNMA(vi, S9, D6); E6 = VFNMA(vr, S9, E6);
        }
        { V ar = VL(pr + 46552), ai = VL(pi + 46552);
          V br = VL(pr + 50784), bi = VL(pi + 50784);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 160, VADD(ar,br)); VS(USCR + 168, VADD(ai,bi));
          D1 = VFMA(vi, S11, D1); E1 = VFMA(vr, S11, E1);
          D2 = VFNMA(vi, S1, D2); E2 = VFNMA(vr, S1, E2);
          D3 = VFMA(vi, S10, D3); E3 = VFMA(vr, S10, E3);
          D4 = VFNMA(vi, S2, D4); E4 = VFNMA(vr, S2, E4);
          D5 = VFMA(vi, S9, D5); E5 = VFMA(vr, S9, E5);
          D6 = VFNMA(vi, S3, D6); E6 = VFNMA(vr, S3, E6);
        }
        VS(PSCR + 0, D1); VS(PSCR + 8, E1);
        VS(PSCR + 16, D2); VS(PSCR + 24, E2);
        VS(PSCR + 32, D3); VS(PSCR + 40, E3);
        VS(PSCR + 48, D4); VS(PSCR + 56, E4);
        VS(PSCR + 64, D5); VS(PSCR + 72, E5);
        VS(PSCR + 80, D6); VS(PSCR + 88, E6);
        }
        {
        V S1 = VSET1(0x1.14459ad2be466p-2);
        V S2 = VSET1(0x1.0a06e851db7cap-1);
        V S3 = VSET1(0x1.763021aaa15d9p-1);
        V S4 = VSET1(0x1.c698e42f47b09p-1);
        V S5 = VSET1(0x1.f54a827142577p-1);
        V S6 = VSET1(0x1.fece70dfd3efbp-1);
        V S7 = VSET1(0x1.e270060999288p-1);
        V S8 = VSET1(0x1.a249e0b897caap-1);
        V S9 = VSET1(0x1.431df5838f7f1p-1);
        V S10 = VSET1(0x1.97f6748e524b1p-2);
        V S11 = VSET1(0x1.16de8a4564f1cp-3);
        V D7 = _mm512_setzero_pd(), E7 = _mm512_setzero_pd();
        V D8 = _mm512_setzero_pd(), E8 = _mm512_setzero_pd();
        V D9 = _mm512_setzero_pd(), E9 = _mm512_setzero_pd();
        V D10 = _mm512_setzero_pd(), E10 = _mm512_setzero_pd();
        V D11 = _mm512_setzero_pd(), E11 = _mm512_setzero_pd();
        { V ar = VL(pr + 4232), ai = VL(pi + 4232);
          V br = VL(pr + 93104), bi = VL(pi + 93104);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S7, D7); E7 = VFMA(vr, S7, E7);
          D8 = VFMA(vi, S8, D8); E8 = VFMA(vr, S8, E8);
          D9 = VFMA(vi, S9, D9); E9 = VFMA(vr, S9, E9);
          D10 = VFMA(vi, S10, D10); E10 = VFMA(vr, S10, E10);
          D11 = VFMA(vi, S11, D11); E11 = VFMA(vr, S11, E11);
        }
        { V ar = VL(pr + 8464), ai = VL(pi + 8464);
          V br = VL(pr + 88872), bi = VL(pi + 88872);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S9, D7); E7 = VFNMA(vr, S9, E7);
          D8 = VFNMA(vi, S7, D8); E8 = VFNMA(vr, S7, E8);
          D9 = VFNMA(vi, S5, D9); E9 = VFNMA(vr, S5, E9);
          D10 = VFNMA(vi, S3, D10); E10 = VFNMA(vr, S3, E10);
          D11 = VFNMA(vi, S1, D11); E11 = VFNMA(vr, S1, E11);
        }
        { V ar = VL(pr + 12696), ai = VL(pi + 12696);
          V br = VL(pr + 84640), bi = VL(pi + 84640);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S2, D7); E7 = VFNMA(vr, S2, E7);
          D8 = VFMA(vi, S1, D8); E8 = VFMA(vr, S1, E8);
          D9 = VFMA(vi, S4, D9); E9 = VFMA(vr, S4, E9);
          D10 = VFMA(vi, S7, D10); E10 = VFMA(vr, S7, E10);
          D11 = VFMA(vi, S10, D11); E11 = VFMA(vr, S10, E11);
        }
        { V ar = VL(pr + 16928), ai = VL(pi + 16928);
          V br = VL(pr + 80408), bi = VL(pi + 80408);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S5, D7); E7 = VFMA(vr, S5, E7);
          D8 = VFMA(vi, S9, D8); E8 = VFMA(vr, S9, E8);
          D9 = VFNMA(vi, S10, D9); E9 = VFNMA(vr, S10, E9);
          D10 = VFNMA(vi, S6, D10); E10 = VFNMA(vr, S6, E10);
          D11 = VFNMA(vi, S2, D11); E11 = VFNMA(vr, S2, E11);
        }
        { V ar = VL(pr + 21160), ai = VL(pi + 21160);
          V br = VL(pr + 76176), bi = VL(pi + 76176);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S11, D7); E7 = VFNMA(vr, S11, E7);
          D8 = VFNMA(vi, S6, D8); E8 = VFNMA(vr, S6, E8);
          D9 = VFNMA(vi, S1, D9); E9 = VFNMA(vr, S1, E9);
          D10 = VFMA(vi, S4, D10); E10 = VFMA(vr, S4, E10);
          D11 = VFMA(vi, S9, D11); E11 = VFMA(vr, S9, E11);
        }
        { V ar = VL(pr + 25392), ai = VL(pi + 25392);
          V br = VL(pr + 71944), bi = VL(pi + 71944);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S4, D7); E7 = VFNMA(vr, S4, E7);
          D8 = VFMA(vi, S2, D8); E8 = VFMA(vr, S2, E8);
          D9 = VFMA(vi, S8, D9); E9 = VFMA(vr, S8, E9);
          D10 = VFNMA(vi, S9, D10); E10 = VFNMA(vr, S9, E10);
          D11 = VFNMA(vi, S3, D11); E11 = VFNMA(vr, S3, E11);
        }
        { V ar = VL(pr + 29624), ai = VL(pi + 29624);
          V br = VL(pr + 67712), bi = VL(pi + 67712);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S3, D7); E7 = VFMA(vr, S3, E7);
          D8 = VFMA(vi, S10, D8); E8 = VFMA(vr, S10, E8);
          D9 = VFNMA(vi, S6, D9); E9 = VFNMA(vr, S6, E9);
          D10 = VFMA(vi, S1, D10); E10 = VFMA(vr, S1, E10);
          D11 = VFMA(vi, S8, D11); E11 = VFMA(vr, S8, E11);
        }
        { V ar = VL(pr + 33856), ai = VL(pi + 33856);
          V br = VL(pr + 63480), bi = VL(pi + 63480);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S10, D7); E7 = VFMA(vr, S10, E7);
          D8 = VFNMA(vi, S5, D8); E8 = VFNMA(vr, S5, E8);
          D9 = VFMA(vi, S3, D9); E9 = VFMA(vr, S3, E9);
          D10 = VFMA(vi, S11, D10); E10 = VFMA(vr, S11, E10);
          D11 = VFNMA(vi, S4, D11); E11 = VFNMA(vr, S4, E11);
        }
        { V ar = VL(pr + 38088), ai = VL(pi + 38088);
          V br = VL(pr + 59248), bi = VL(pi + 59248);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S6, D7); E7 = VFNMA(vr, S6, E7);
          D8 = VFMA(vi, S3, D8); E8 = VFMA(vr, S3, E8);
          D9 = VFNMA(vi, S11, D9); E9 = VFNMA(vr, S11, E9);
          D10 = VFNMA(vi, S2, D10); E10 = VFNMA(vr, S2, E10);
          D11 = VFMA(vi, S7, D11); E11 = VFMA(vr, S7, E11);
        }
        { V ar = VL(pr + 42320), ai = VL(pi + 42320);
          V br = VL(pr + 55016), bi = VL(pi + 55016);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S1, D7); E7 = VFMA(vr, S1, E7);
          D8 = VFMA(vi, S11, D8); E8 = VFMA(vr, S11, E8);
          D9 = VFNMA(vi, S2, D9); E9 = VFNMA(vr, S2, E9);
          D10 = VFMA(vi, S8, D10); E10 = VFMA(vr, S8, E10);
          D11 = VFNMA(vi, S5, D11); E11 = VFNMA(vr, S5, E11);
        }
        { V ar = VL(pr + 46552), ai = VL(pi + 46552);
          V br = VL(pr + 50784), bi = VL(pi + 50784);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S8, D7); E7 = VFMA(vr, S8, E7);
          D8 = VFNMA(vi, S4, D8); E8 = VFNMA(vr, S4, E8);
          D9 = VFMA(vi, S7, D9); E9 = VFMA(vr, S7, E9);
          D10 = VFNMA(vi, S5, D10); E10 = VFNMA(vr, S5, E10);
          D11 = VFMA(vi, S6, D11); E11 = VFMA(vr, S6, E11);
        }
        VS(PSCR + 96, D7); VS(PSCR + 104, E7);
        VS(PSCR + 112, D8); VS(PSCR + 120, E8);
        VS(PSCR + 128, D9); VS(PSCR + 136, E9);
        VS(PSCR + 144, D10); VS(PSCR + 152, E10);
        VS(PSCR + 160, D11); VS(PSCR + 168, E11);
        }
        VS(USCR + 384, VL(pr)); VS(USCR + 392, VL(pi));  // save x0 (row 0 gets overwritten)
        {
        V C1 = VSET1(0x1.ed037ea3d2dbcp-1);
        V C2 = VSET1(0x1.b57675cf309eep-1);
        V C3 = VSET1(0x1.5d779b07cfef7p-1);
        V C4 = VSET1(0x1.d71b4a0c5a6c9p-2);
        V C5 = VSET1(0x1.a0ad8bd1e2881p-3);
        V C6 = VSET1(-0x1.17855b599f3b2p-4);
        V C7 = VSET1(-0x1.56eaae597c776p-2);
        V C8 = VSET1(-0x1.2742a4a775cfap-1);
        V C9 = VSET1(-0x1.8d2a07c16d46ep-1);
        V C10 = VSET1(-0x1.d59cb83ef99bcp-1);
        V C11 = VSET1(-0x1.fb3b3035aa6ccp-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A1 = x0r, B1 = x0i;
        V A2 = x0r, B2 = x0i;
        V A3 = x0r, B3 = x0i;
        V A4 = x0r, B4 = x0i;
        V A5 = x0r, B5 = x0i;
        V A6 = x0r, B6 = x0i;
        V s0r = x0r, s0i = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A1 = VFMA(ur, C1, A1); B1 = VFMA(ui, C1, B1);
          A2 = VFMA(ur, C2, A2); B2 = VFMA(ui, C2, B2);
          A3 = VFMA(ur, C3, A3); B3 = VFMA(ui, C3, B3);
          A4 = VFMA(ur, C4, A4); B4 = VFMA(ui, C4, B4);
          A5 = VFMA(ur, C5, A5); B5 = VFMA(ui, C5, B5);
          A6 = VFMA(ur, C6, A6); B6 = VFMA(ui, C6, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A1 = VFMA(ur, C2, A1); B1 = VFMA(ui, C2, B1);
          A2 = VFMA(ur, C4, A2); B2 = VFMA(ui, C4, B2);
          A3 = VFMA(ur, C6, A3); B3 = VFMA(ui, C6, B3);
          A4 = VFMA(ur, C8, A4); B4 = VFMA(ui, C8, B4);
          A5 = VFMA(ur, C10, A5); B5 = VFMA(ui, C10, B5);
          A6 = VFMA(ur, C11, A6); B6 = VFMA(ui, C11, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A1 = VFMA(ur, C3, A1); B1 = VFMA(ui, C3, B1);
          A2 = VFMA(ur, C6, A2); B2 = VFMA(ui, C6, B2);
          A3 = VFMA(ur, C9, A3); B3 = VFMA(ui, C9, B3);
          A4 = VFMA(ur, C11, A4); B4 = VFMA(ui, C11, B4);
          A5 = VFMA(ur, C8, A5); B5 = VFMA(ui, C8, B5);
          A6 = VFMA(ur, C5, A6); B6 = VFMA(ui, C5, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A1 = VFMA(ur, C4, A1); B1 = VFMA(ui, C4, B1);
          A2 = VFMA(ur, C8, A2); B2 = VFMA(ui, C8, B2);
          A3 = VFMA(ur, C11, A3); B3 = VFMA(ui, C11, B3);
          A4 = VFMA(ur, C7, A4); B4 = VFMA(ui, C7, B4);
          A5 = VFMA(ur, C3, A5); B5 = VFMA(ui, C3, B5);
          A6 = VFMA(ur, C1, A6); B6 = VFMA(ui, C1, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A1 = VFMA(ur, C5, A1); B1 = VFMA(ui, C5, B1);
          A2 = VFMA(ur, C10, A2); B2 = VFMA(ui, C10, B2);
          A3 = VFMA(ur, C8, A3); B3 = VFMA(ui, C8, B3);
          A4 = VFMA(ur, C3, A4); B4 = VFMA(ui, C3, B4);
          A5 = VFMA(ur, C2, A5); B5 = VFMA(ui, C2, B5);
          A6 = VFMA(ur, C7, A6); B6 = VFMA(ui, C7, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A1 = VFMA(ur, C6, A1); B1 = VFMA(ui, C6, B1);
          A2 = VFMA(ur, C11, A2); B2 = VFMA(ui, C11, B2);
          A3 = VFMA(ur, C5, A3); B3 = VFMA(ui, C5, B3);
          A4 = VFMA(ur, C1, A4); B4 = VFMA(ui, C1, B4);
          A5 = VFMA(ur, C7, A5); B5 = VFMA(ui, C7, B5);
          A6 = VFMA(ur, C10, A6); B6 = VFMA(ui, C10, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 96), ui = VL(USCR + 104);
          A1 = VFMA(ur, C7, A1); B1 = VFMA(ui, C7, B1);
          A2 = VFMA(ur, C9, A2); B2 = VFMA(ui, C9, B2);
          A3 = VFMA(ur, C2, A3); B3 = VFMA(ui, C2, B3);
          A4 = VFMA(ur, C5, A4); B4 = VFMA(ui, C5, B4);
          A5 = VFMA(ur, C11, A5); B5 = VFMA(ui, C11, B5);
          A6 = VFMA(ur, C4, A6); B6 = VFMA(ui, C4, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 112), ui = VL(USCR + 120);
          A1 = VFMA(ur, C8, A1); B1 = VFMA(ui, C8, B1);
          A2 = VFMA(ur, C7, A2); B2 = VFMA(ui, C7, B2);
          A3 = VFMA(ur, C1, A3); B3 = VFMA(ui, C1, B3);
          A4 = VFMA(ur, C9, A4); B4 = VFMA(ui, C9, B4);
          A5 = VFMA(ur, C6, A5); B5 = VFMA(ui, C6, B5);
          A6 = VFMA(ur, C2, A6); B6 = VFMA(ui, C2, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 128), ui = VL(USCR + 136);
          A1 = VFMA(ur, C9, A1); B1 = VFMA(ui, C9, B1);
          A2 = VFMA(ur, C5, A2); B2 = VFMA(ui, C5, B2);
          A3 = VFMA(ur, C4, A3); B3 = VFMA(ui, C4, B3);
          A4 = VFMA(ur, C10, A4); B4 = VFMA(ui, C10, B4);
          A5 = VFMA(ur, C1, A5); B5 = VFMA(ui, C1, B5);
          A6 = VFMA(ur, C8, A6); B6 = VFMA(ui, C8, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 144), ui = VL(USCR + 152);
          A1 = VFMA(ur, C10, A1); B1 = VFMA(ui, C10, B1);
          A2 = VFMA(ur, C3, A2); B2 = VFMA(ui, C3, B2);
          A3 = VFMA(ur, C7, A3); B3 = VFMA(ui, C7, B3);
          A4 = VFMA(ur, C6, A4); B4 = VFMA(ui, C6, B4);
          A5 = VFMA(ur, C4, A5); B5 = VFMA(ui, C4, B5);
          A6 = VFMA(ur, C9, A6); B6 = VFMA(ui, C9, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 160), ui = VL(USCR + 168);
          A1 = VFMA(ur, C11, A1); B1 = VFMA(ui, C11, B1);
          A2 = VFMA(ur, C1, A2); B2 = VFMA(ui, C1, B2);
          A3 = VFMA(ur, C10, A3); B3 = VFMA(ui, C10, B3);
          A4 = VFMA(ur, C2, A4); B4 = VFMA(ui, C2, B4);
          A5 = VFMA(ur, C9, A5); B5 = VFMA(ui, C9, B5);
          A6 = VFMA(ur, C3, A6); B6 = VFMA(ui, C3, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        VS(pr + 0, s0r); VS(pi + 0, s0i);
        { V Dk = VL(PSCR + 0), Ek = VL(PSCR + 8);
          V Xr = VADD(A1, Dk), Xi = VSUB(B1, Ek);
          V Yr = VSUB(A1, Dk), Yi = VADD(B1, Ek);
        VS(pr + 4232, Xr); VS(pi + 4232, Xi);
        VS(pr + 93104, Yr); VS(pi + 93104, Yi);
        }
        { V Dk = VL(PSCR + 16), Ek = VL(PSCR + 24);
          V Xr = VADD(A2, Dk), Xi = VSUB(B2, Ek);
          V Yr = VSUB(A2, Dk), Yi = VADD(B2, Ek);
        VS(pr + 8464, Xr); VS(pi + 8464, Xi);
        VS(pr + 88872, Yr); VS(pi + 88872, Yi);
        }
        { V Dk = VL(PSCR + 32), Ek = VL(PSCR + 40);
          V Xr = VADD(A3, Dk), Xi = VSUB(B3, Ek);
          V Yr = VSUB(A3, Dk), Yi = VADD(B3, Ek);
        VS(pr + 12696, Xr); VS(pi + 12696, Xi);
        VS(pr + 84640, Yr); VS(pi + 84640, Yi);
        }
        { V Dk = VL(PSCR + 48), Ek = VL(PSCR + 56);
          V Xr = VADD(A4, Dk), Xi = VSUB(B4, Ek);
          V Yr = VSUB(A4, Dk), Yi = VADD(B4, Ek);
        VS(pr + 16928, Xr); VS(pi + 16928, Xi);
        VS(pr + 80408, Yr); VS(pi + 80408, Yi);
        }
        { V Dk = VL(PSCR + 64), Ek = VL(PSCR + 72);
          V Xr = VADD(A5, Dk), Xi = VSUB(B5, Ek);
          V Yr = VSUB(A5, Dk), Yi = VADD(B5, Ek);
        VS(pr + 21160, Xr); VS(pi + 21160, Xi);
        VS(pr + 76176, Yr); VS(pi + 76176, Yi);
        }
        { V Dk = VL(PSCR + 80), Ek = VL(PSCR + 88);
          V Xr = VADD(A6, Dk), Xi = VSUB(B6, Ek);
          V Yr = VSUB(A6, Dk), Yi = VADD(B6, Ek);
        VS(pr + 25392, Xr); VS(pi + 25392, Xi);
        VS(pr + 71944, Yr); VS(pi + 71944, Yi);
        }
        }
        {
        V C1 = VSET1(0x1.ed037ea3d2dbcp-1);
        V C2 = VSET1(0x1.b57675cf309eep-1);
        V C3 = VSET1(0x1.5d779b07cfef7p-1);
        V C4 = VSET1(0x1.d71b4a0c5a6c9p-2);
        V C5 = VSET1(0x1.a0ad8bd1e2881p-3);
        V C6 = VSET1(-0x1.17855b599f3b2p-4);
        V C7 = VSET1(-0x1.56eaae597c776p-2);
        V C8 = VSET1(-0x1.2742a4a775cfap-1);
        V C9 = VSET1(-0x1.8d2a07c16d46ep-1);
        V C10 = VSET1(-0x1.d59cb83ef99bcp-1);
        V C11 = VSET1(-0x1.fb3b3035aa6ccp-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A7 = x0r, B7 = x0i;
        V A8 = x0r, B8 = x0i;
        V A9 = x0r, B9 = x0i;
        V A10 = x0r, B10 = x0i;
        V A11 = x0r, B11 = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A7 = VFMA(ur, C7, A7); B7 = VFMA(ui, C7, B7);
          A8 = VFMA(ur, C8, A8); B8 = VFMA(ui, C8, B8);
          A9 = VFMA(ur, C9, A9); B9 = VFMA(ui, C9, B9);
          A10 = VFMA(ur, C10, A10); B10 = VFMA(ui, C10, B10);
          A11 = VFMA(ur, C11, A11); B11 = VFMA(ui, C11, B11);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A7 = VFMA(ur, C9, A7); B7 = VFMA(ui, C9, B7);
          A8 = VFMA(ur, C7, A8); B8 = VFMA(ui, C7, B8);
          A9 = VFMA(ur, C5, A9); B9 = VFMA(ui, C5, B9);
          A10 = VFMA(ur, C3, A10); B10 = VFMA(ui, C3, B10);
          A11 = VFMA(ur, C1, A11); B11 = VFMA(ui, C1, B11);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A7 = VFMA(ur, C2, A7); B7 = VFMA(ui, C2, B7);
          A8 = VFMA(ur, C1, A8); B8 = VFMA(ui, C1, B8);
          A9 = VFMA(ur, C4, A9); B9 = VFMA(ui, C4, B9);
          A10 = VFMA(ur, C7, A10); B10 = VFMA(ui, C7, B10);
          A11 = VFMA(ur, C10, A11); B11 = VFMA(ui, C10, B11);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A7 = VFMA(ur, C5, A7); B7 = VFMA(ui, C5, B7);
          A8 = VFMA(ur, C9, A8); B8 = VFMA(ui, C9, B8);
          A9 = VFMA(ur, C10, A9); B9 = VFMA(ui, C10, B9);
          A10 = VFMA(ur, C6, A10); B10 = VFMA(ui, C6, B10);
          A11 = VFMA(ur, C2, A11); B11 = VFMA(ui, C2, B11);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A7 = VFMA(ur, C11, A7); B7 = VFMA(ui, C11, B7);
          A8 = VFMA(ur, C6, A8); B8 = VFMA(ui, C6, B8);
          A9 = VFMA(ur, C1, A9); B9 = VFMA(ui, C1, B9);
          A10 = VFMA(ur, C4, A10); B10 = VFMA(ui, C4, B10);
          A11 = VFMA(ur, C9, A11); B11 = VFMA(ui, C9, B11);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A7 = VFMA(ur, C4, A7); B7 = VFMA(ui, C4, B7);
          A8 = VFMA(ur, C2, A8); B8 = VFMA(ui, C2, B8);
          A9 = VFMA(ur, C8, A9); B9 = VFMA(ui, C8, B9);
          A10 = VFMA(ur, C9, A10); B10 = VFMA(ui, C9, B10);
          A11 = VFMA(ur, C3, A11); B11 = VFMA(ui, C3, B11);
        }
        { V ur = VL(USCR + 96), ui = VL(USCR + 104);
          A7 = VFMA(ur, C3, A7); B7 = VFMA(ui, C3, B7);
          A8 = VFMA(ur, C10, A8); B8 = VFMA(ui, C10, B8);
          A9 = VFMA(ur, C6, A9); B9 = VFMA(ui, C6, B9);
          A10 = VFMA(ur, C1, A10); B10 = VFMA(ui, C1, B10);
          A11 = VFMA(ur, C8, A11); B11 = VFMA(ui, C8, B11);
        }
        { V ur = VL(USCR + 112), ui = VL(USCR + 120);
          A7 = VFMA(ur, C10, A7); B7 = VFMA(ui, C10, B7);
          A8 = VFMA(ur, C5, A8); B8 = VFMA(ui, C5, B8);
          A9 = VFMA(ur, C3, A9); B9 = VFMA(ui, C3, B9);
          A10 = VFMA(ur, C11, A10); B10 = VFMA(ui, C11, B10);
          A11 = VFMA(ur, C4, A11); B11 = VFMA(ui, C4, B11);
        }
        { V ur = VL(USCR + 128), ui = VL(USCR + 136);
          A7 = VFMA(ur, C6, A7); B7 = VFMA(ui, C6, B7);
          A8 = VFMA(ur, C3, A8); B8 = VFMA(ui, C3, B8);
          A9 = VFMA(ur, C11, A9); B9 = VFMA(ui, C11, B9);
          A10 = VFMA(ur, C2, A10); B10 = VFMA(ui, C2, B10);
          A11 = VFMA(ur, C7, A11); B11 = VFMA(ui, C7, B11);
        }
        { V ur = VL(USCR + 144), ui = VL(USCR + 152);
          A7 = VFMA(ur, C1, A7); B7 = VFMA(ui, C1, B7);
          A8 = VFMA(ur, C11, A8); B8 = VFMA(ui, C11, B8);
          A9 = VFMA(ur, C2, A9); B9 = VFMA(ui, C2, B9);
          A10 = VFMA(ur, C8, A10); B10 = VFMA(ui, C8, B10);
          A11 = VFMA(ur, C5, A11); B11 = VFMA(ui, C5, B11);
        }
        { V ur = VL(USCR + 160), ui = VL(USCR + 168);
          A7 = VFMA(ur, C8, A7); B7 = VFMA(ui, C8, B7);
          A8 = VFMA(ur, C4, A8); B8 = VFMA(ui, C4, B8);
          A9 = VFMA(ur, C7, A9); B9 = VFMA(ui, C7, B9);
          A10 = VFMA(ur, C5, A10); B10 = VFMA(ui, C5, B10);
          A11 = VFMA(ur, C6, A11); B11 = VFMA(ui, C6, B11);
        }
        { V Dk = VL(PSCR + 96), Ek = VL(PSCR + 104);
          V Xr = VADD(A7, Dk), Xi = VSUB(B7, Ek);
          V Yr = VSUB(A7, Dk), Yi = VADD(B7, Ek);
        VS(pr + 29624, Xr); VS(pi + 29624, Xi);
        VS(pr + 67712, Yr); VS(pi + 67712, Yi);
        }
        { V Dk = VL(PSCR + 112), Ek = VL(PSCR + 120);
          V Xr = VADD(A8, Dk), Xi = VSUB(B8, Ek);
          V Yr = VSUB(A8, Dk), Yi = VADD(B8, Ek);
        VS(pr + 33856, Xr); VS(pi + 33856, Xi);
        VS(pr + 63480, Yr); VS(pi + 63480, Yi);
        }
        { V Dk = VL(PSCR + 128), Ek = VL(PSCR + 136);
          V Xr = VADD(A9, Dk), Xi = VSUB(B9, Ek);
          V Yr = VSUB(A9, Dk), Yi = VADD(B9, Ek);
        VS(pr + 38088, Xr); VS(pi + 38088, Xi);
        VS(pr + 59248, Yr); VS(pi + 59248, Yi);
        }
        { V Dk = VL(PSCR + 144), Ek = VL(PSCR + 152);
          V Xr = VADD(A10, Dk), Xi = VSUB(B10, Ek);
          V Yr = VSUB(A10, Dk), Yi = VADD(B10, Ek);
        VS(pr + 42320, Xr); VS(pi + 42320, Xi);
        VS(pr + 55016, Yr); VS(pi + 55016, Yi);
        }
        { V Dk = VL(PSCR + 160), Ek = VL(PSCR + 168);
          V Xr = VADD(A11, Dk), Xi = VSUB(B11, Ek);
          V Yr = VSUB(A11, Dk), Yi = VADD(B11, Ek);
        VS(pr + 46552, Xr); VS(pi + 46552, Xi);
        VS(pr + 50784, Yr); VS(pi + 50784, Yi);
        }
        }
    }
}
static __attribute__((noinline)) void p23_xxm(double* PR, double* PI, const double* CR, const double* CI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        const double* cr = CR + q_*pstep; const double* ci = CI + q_*pstep;
        {
        V S1 = VSET1(0x1.14459ad2be466p-2);
        V S2 = VSET1(0x1.0a06e851db7cap-1);
        V S3 = VSET1(0x1.763021aaa15d9p-1);
        V S4 = VSET1(0x1.c698e42f47b09p-1);
        V S5 = VSET1(0x1.f54a827142577p-1);
        V S6 = VSET1(0x1.fece70dfd3efbp-1);
        V S7 = VSET1(0x1.e270060999288p-1);
        V S8 = VSET1(0x1.a249e0b897caap-1);
        V S9 = VSET1(0x1.431df5838f7f1p-1);
        V S10 = VSET1(0x1.97f6748e524b1p-2);
        V S11 = VSET1(0x1.16de8a4564f1cp-3);
        V D1 = _mm512_setzero_pd(), E1 = _mm512_setzero_pd();
        V D2 = _mm512_setzero_pd(), E2 = _mm512_setzero_pd();
        V D3 = _mm512_setzero_pd(), E3 = _mm512_setzero_pd();
        V D4 = _mm512_setzero_pd(), E4 = _mm512_setzero_pd();
        V D5 = _mm512_setzero_pd(), E5 = _mm512_setzero_pd();
        V D6 = _mm512_setzero_pd(), E6 = _mm512_setzero_pd();
        { V ar = VL(pr + 4232), ai = VL(pi + 4232);
          V br = VL(pr + 93104), bi = VL(pi + 93104);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 0, VADD(ar,br)); VS(USCR + 8, VADD(ai,bi));
          D1 = VFMA(vi, S1, D1); E1 = VFMA(vr, S1, E1);
          D2 = VFMA(vi, S2, D2); E2 = VFMA(vr, S2, E2);
          D3 = VFMA(vi, S3, D3); E3 = VFMA(vr, S3, E3);
          D4 = VFMA(vi, S4, D4); E4 = VFMA(vr, S4, E4);
          D5 = VFMA(vi, S5, D5); E5 = VFMA(vr, S5, E5);
          D6 = VFMA(vi, S6, D6); E6 = VFMA(vr, S6, E6);
        }
        { V ar = VL(pr + 8464), ai = VL(pi + 8464);
          V br = VL(pr + 88872), bi = VL(pi + 88872);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 16, VADD(ar,br)); VS(USCR + 24, VADD(ai,bi));
          D1 = VFMA(vi, S2, D1); E1 = VFMA(vr, S2, E1);
          D2 = VFMA(vi, S4, D2); E2 = VFMA(vr, S4, E2);
          D3 = VFMA(vi, S6, D3); E3 = VFMA(vr, S6, E3);
          D4 = VFMA(vi, S8, D4); E4 = VFMA(vr, S8, E4);
          D5 = VFMA(vi, S10, D5); E5 = VFMA(vr, S10, E5);
          D6 = VFNMA(vi, S11, D6); E6 = VFNMA(vr, S11, E6);
        }
        { V ar = VL(pr + 12696), ai = VL(pi + 12696);
          V br = VL(pr + 84640), bi = VL(pi + 84640);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 32, VADD(ar,br)); VS(USCR + 40, VADD(ai,bi));
          D1 = VFMA(vi, S3, D1); E1 = VFMA(vr, S3, E1);
          D2 = VFMA(vi, S6, D2); E2 = VFMA(vr, S6, E2);
          D3 = VFMA(vi, S9, D3); E3 = VFMA(vr, S9, E3);
          D4 = VFNMA(vi, S11, D4); E4 = VFNMA(vr, S11, E4);
          D5 = VFNMA(vi, S8, D5); E5 = VFNMA(vr, S8, E5);
          D6 = VFNMA(vi, S5, D6); E6 = VFNMA(vr, S5, E6);
        }
        { V ar = VL(pr + 16928), ai = VL(pi + 16928);
          V br = VL(pr + 80408), bi = VL(pi + 80408);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 48, VADD(ar,br)); VS(USCR + 56, VADD(ai,bi));
          D1 = VFMA(vi, S4, D1); E1 = VFMA(vr, S4, E1);
          D2 = VFMA(vi, S8, D2); E2 = VFMA(vr, S8, E2);
          D3 = VFNMA(vi, S11, D3); E3 = VFNMA(vr, S11, E3);
          D4 = VFNMA(vi, S7, D4); E4 = VFNMA(vr, S7, E4);
          D5 = VFNMA(vi, S3, D5); E5 = VFNMA(vr, S3, E5);
          D6 = VFMA(vi, S1, D6); E6 = VFMA(vr, S1, E6);
        }
        { V ar = VL(pr + 21160), ai = VL(pi + 21160);
          V br = VL(pr + 76176), bi = VL(pi + 76176);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 64, VADD(ar,br)); VS(USCR + 72, VADD(ai,bi));
          D1 = VFMA(vi, S5, D1); E1 = VFMA(vr, S5, E1);
          D2 = VFMA(vi, S10, D2); E2 = VFMA(vr, S10, E2);
          D3 = VFNMA(vi, S8, D3); E3 = VFNMA(vr, S8, E3);
          D4 = VFNMA(vi, S3, D4); E4 = VFNMA(vr, S3, E4);
          D5 = VFMA(vi, S2, D5); E5 = VFMA(vr, S2, E5);
          D6 = VFMA(vi, S7, D6); E6 = VFMA(vr, S7, E6);
        }
        { V ar = VL(pr + 25392), ai = VL(pi + 25392);
          V br = VL(pr + 71944), bi = VL(pi + 71944);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 80, VADD(ar,br)); VS(USCR + 88, VADD(ai,bi));
          D1 = VFMA(vi, S6, D1); E1 = VFMA(vr, S6, E1);
          D2 = VFNMA(vi, S11, D2); E2 = VFNMA(vr, S11, E2);
          D3 = VFNMA(vi, S5, D3); E3 = VFNMA(vr, S5, E3);
          D4 = VFMA(vi, S1, D4); E4 = VFMA(vr, S1, E4);
          D5 = VFMA(vi, S7, D5); E5 = VFMA(vr, S7, E5);
          D6 = VFNMA(vi, S10, D6); E6 = VFNMA(vr, S10, E6);
        }
        { V ar = VL(pr + 29624), ai = VL(pi + 29624);
          V br = VL(pr + 67712), bi = VL(pi + 67712);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 96, VADD(ar,br)); VS(USCR + 104, VADD(ai,bi));
          D1 = VFMA(vi, S7, D1); E1 = VFMA(vr, S7, E1);
          D2 = VFNMA(vi, S9, D2); E2 = VFNMA(vr, S9, E2);
          D3 = VFNMA(vi, S2, D3); E3 = VFNMA(vr, S2, E3);
          D4 = VFMA(vi, S5, D4); E4 = VFMA(vr, S5, E4);
          D5 = VFNMA(vi, S11, D5); E5 = VFNMA(vr, S11, E5);
          D6 = VFNMA(vi, S4, D6); E6 = VFNMA(vr, S4, E6);
        }
        { V ar = VL(pr + 33856), ai = VL(pi + 33856);
          V br = VL(pr + 63480), bi = VL(pi + 63480);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 112, VADD(ar,br)); VS(USCR + 120, VADD(ai,bi));
          D1 = VFMA(vi, S8, D1); E1 = VFMA(vr, S8, E1);
          D2 = VFNMA(vi, S7, D2); E2 = VFNMA(vr, S7, E2);
          D3 = VFMA(vi, S1, D3); E3 = VFMA(vr, S1, E3);
          D4 = VFMA(vi, S9, D4); E4 = VFMA(vr, S9, E4);
          D5 = VFNMA(vi, S6, D5); E5 = VFNMA(vr, S6, E5);
          D6 = VFMA(vi, S2, D6); E6 = VFMA(vr, S2, E6);
        }
        { V ar = VL(pr + 38088), ai = VL(pi + 38088);
          V br = VL(pr + 59248), bi = VL(pi + 59248);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 128, VADD(ar,br)); VS(USCR + 136, VADD(ai,bi));
          D1 = VFMA(vi, S9, D1); E1 = VFMA(vr, S9, E1);
          D2 = VFNMA(vi, S5, D2); E2 = VFNMA(vr, S5, E2);
          D3 = VFMA(vi, S4, D3); E3 = VFMA(vr, S4, E3);
          D4 = VFNMA(vi, S10, D4); E4 = VFNMA(vr, S10, E4);
          D5 = VFNMA(vi, S1, D5); E5 = VFNMA(vr, S1, E5);
          D6 = VFMA(vi, S8, D6); E6 = VFMA(vr, S8, E6);
        }
        { V ar = VL(pr + 42320), ai = VL(pi + 42320);
          V br = VL(pr + 55016), bi = VL(pi + 55016);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 144, VADD(ar,br)); VS(USCR + 152, VADD(ai,bi));
          D1 = VFMA(vi, S10, D1); E1 = VFMA(vr, S10, E1);
          D2 = VFNMA(vi, S3, D2); E2 = VFNMA(vr, S3, E2);
          D3 = VFMA(vi, S7, D3); E3 = VFMA(vr, S7, E3);
          D4 = VFNMA(vi, S6, D4); E4 = VFNMA(vr, S6, E4);
          D5 = VFMA(vi, S4, D5); E5 = VFMA(vr, S4, E5);
          D6 = VFNMA(vi, S9, D6); E6 = VFNMA(vr, S9, E6);
        }
        { V ar = VL(pr + 46552), ai = VL(pi + 46552);
          V br = VL(pr + 50784), bi = VL(pi + 50784);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 160, VADD(ar,br)); VS(USCR + 168, VADD(ai,bi));
          D1 = VFMA(vi, S11, D1); E1 = VFMA(vr, S11, E1);
          D2 = VFNMA(vi, S1, D2); E2 = VFNMA(vr, S1, E2);
          D3 = VFMA(vi, S10, D3); E3 = VFMA(vr, S10, E3);
          D4 = VFNMA(vi, S2, D4); E4 = VFNMA(vr, S2, E4);
          D5 = VFMA(vi, S9, D5); E5 = VFMA(vr, S9, E5);
          D6 = VFNMA(vi, S3, D6); E6 = VFNMA(vr, S3, E6);
        }
        VS(PSCR + 0, D1); VS(PSCR + 8, E1);
        VS(PSCR + 16, D2); VS(PSCR + 24, E2);
        VS(PSCR + 32, D3); VS(PSCR + 40, E3);
        VS(PSCR + 48, D4); VS(PSCR + 56, E4);
        VS(PSCR + 64, D5); VS(PSCR + 72, E5);
        VS(PSCR + 80, D6); VS(PSCR + 88, E6);
        }
        {
        V S1 = VSET1(0x1.14459ad2be466p-2);
        V S2 = VSET1(0x1.0a06e851db7cap-1);
        V S3 = VSET1(0x1.763021aaa15d9p-1);
        V S4 = VSET1(0x1.c698e42f47b09p-1);
        V S5 = VSET1(0x1.f54a827142577p-1);
        V S6 = VSET1(0x1.fece70dfd3efbp-1);
        V S7 = VSET1(0x1.e270060999288p-1);
        V S8 = VSET1(0x1.a249e0b897caap-1);
        V S9 = VSET1(0x1.431df5838f7f1p-1);
        V S10 = VSET1(0x1.97f6748e524b1p-2);
        V S11 = VSET1(0x1.16de8a4564f1cp-3);
        V D7 = _mm512_setzero_pd(), E7 = _mm512_setzero_pd();
        V D8 = _mm512_setzero_pd(), E8 = _mm512_setzero_pd();
        V D9 = _mm512_setzero_pd(), E9 = _mm512_setzero_pd();
        V D10 = _mm512_setzero_pd(), E10 = _mm512_setzero_pd();
        V D11 = _mm512_setzero_pd(), E11 = _mm512_setzero_pd();
        { V ar = VL(pr + 4232), ai = VL(pi + 4232);
          V br = VL(pr + 93104), bi = VL(pi + 93104);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S7, D7); E7 = VFMA(vr, S7, E7);
          D8 = VFMA(vi, S8, D8); E8 = VFMA(vr, S8, E8);
          D9 = VFMA(vi, S9, D9); E9 = VFMA(vr, S9, E9);
          D10 = VFMA(vi, S10, D10); E10 = VFMA(vr, S10, E10);
          D11 = VFMA(vi, S11, D11); E11 = VFMA(vr, S11, E11);
        }
        { V ar = VL(pr + 8464), ai = VL(pi + 8464);
          V br = VL(pr + 88872), bi = VL(pi + 88872);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S9, D7); E7 = VFNMA(vr, S9, E7);
          D8 = VFNMA(vi, S7, D8); E8 = VFNMA(vr, S7, E8);
          D9 = VFNMA(vi, S5, D9); E9 = VFNMA(vr, S5, E9);
          D10 = VFNMA(vi, S3, D10); E10 = VFNMA(vr, S3, E10);
          D11 = VFNMA(vi, S1, D11); E11 = VFNMA(vr, S1, E11);
        }
        { V ar = VL(pr + 12696), ai = VL(pi + 12696);
          V br = VL(pr + 84640), bi = VL(pi + 84640);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S2, D7); E7 = VFNMA(vr, S2, E7);
          D8 = VFMA(vi, S1, D8); E8 = VFMA(vr, S1, E8);
          D9 = VFMA(vi, S4, D9); E9 = VFMA(vr, S4, E9);
          D10 = VFMA(vi, S7, D10); E10 = VFMA(vr, S7, E10);
          D11 = VFMA(vi, S10, D11); E11 = VFMA(vr, S10, E11);
        }
        { V ar = VL(pr + 16928), ai = VL(pi + 16928);
          V br = VL(pr + 80408), bi = VL(pi + 80408);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S5, D7); E7 = VFMA(vr, S5, E7);
          D8 = VFMA(vi, S9, D8); E8 = VFMA(vr, S9, E8);
          D9 = VFNMA(vi, S10, D9); E9 = VFNMA(vr, S10, E9);
          D10 = VFNMA(vi, S6, D10); E10 = VFNMA(vr, S6, E10);
          D11 = VFNMA(vi, S2, D11); E11 = VFNMA(vr, S2, E11);
        }
        { V ar = VL(pr + 21160), ai = VL(pi + 21160);
          V br = VL(pr + 76176), bi = VL(pi + 76176);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S11, D7); E7 = VFNMA(vr, S11, E7);
          D8 = VFNMA(vi, S6, D8); E8 = VFNMA(vr, S6, E8);
          D9 = VFNMA(vi, S1, D9); E9 = VFNMA(vr, S1, E9);
          D10 = VFMA(vi, S4, D10); E10 = VFMA(vr, S4, E10);
          D11 = VFMA(vi, S9, D11); E11 = VFMA(vr, S9, E11);
        }
        { V ar = VL(pr + 25392), ai = VL(pi + 25392);
          V br = VL(pr + 71944), bi = VL(pi + 71944);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S4, D7); E7 = VFNMA(vr, S4, E7);
          D8 = VFMA(vi, S2, D8); E8 = VFMA(vr, S2, E8);
          D9 = VFMA(vi, S8, D9); E9 = VFMA(vr, S8, E9);
          D10 = VFNMA(vi, S9, D10); E10 = VFNMA(vr, S9, E10);
          D11 = VFNMA(vi, S3, D11); E11 = VFNMA(vr, S3, E11);
        }
        { V ar = VL(pr + 29624), ai = VL(pi + 29624);
          V br = VL(pr + 67712), bi = VL(pi + 67712);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S3, D7); E7 = VFMA(vr, S3, E7);
          D8 = VFMA(vi, S10, D8); E8 = VFMA(vr, S10, E8);
          D9 = VFNMA(vi, S6, D9); E9 = VFNMA(vr, S6, E9);
          D10 = VFMA(vi, S1, D10); E10 = VFMA(vr, S1, E10);
          D11 = VFMA(vi, S8, D11); E11 = VFMA(vr, S8, E11);
        }
        { V ar = VL(pr + 33856), ai = VL(pi + 33856);
          V br = VL(pr + 63480), bi = VL(pi + 63480);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S10, D7); E7 = VFMA(vr, S10, E7);
          D8 = VFNMA(vi, S5, D8); E8 = VFNMA(vr, S5, E8);
          D9 = VFMA(vi, S3, D9); E9 = VFMA(vr, S3, E9);
          D10 = VFMA(vi, S11, D10); E10 = VFMA(vr, S11, E10);
          D11 = VFNMA(vi, S4, D11); E11 = VFNMA(vr, S4, E11);
        }
        { V ar = VL(pr + 38088), ai = VL(pi + 38088);
          V br = VL(pr + 59248), bi = VL(pi + 59248);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S6, D7); E7 = VFNMA(vr, S6, E7);
          D8 = VFMA(vi, S3, D8); E8 = VFMA(vr, S3, E8);
          D9 = VFNMA(vi, S11, D9); E9 = VFNMA(vr, S11, E9);
          D10 = VFNMA(vi, S2, D10); E10 = VFNMA(vr, S2, E10);
          D11 = VFMA(vi, S7, D11); E11 = VFMA(vr, S7, E11);
        }
        { V ar = VL(pr + 42320), ai = VL(pi + 42320);
          V br = VL(pr + 55016), bi = VL(pi + 55016);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S1, D7); E7 = VFMA(vr, S1, E7);
          D8 = VFMA(vi, S11, D8); E8 = VFMA(vr, S11, E8);
          D9 = VFNMA(vi, S2, D9); E9 = VFNMA(vr, S2, E9);
          D10 = VFMA(vi, S8, D10); E10 = VFMA(vr, S8, E10);
          D11 = VFNMA(vi, S5, D11); E11 = VFNMA(vr, S5, E11);
        }
        { V ar = VL(pr + 46552), ai = VL(pi + 46552);
          V br = VL(pr + 50784), bi = VL(pi + 50784);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S8, D7); E7 = VFMA(vr, S8, E7);
          D8 = VFNMA(vi, S4, D8); E8 = VFNMA(vr, S4, E8);
          D9 = VFMA(vi, S7, D9); E9 = VFMA(vr, S7, E9);
          D10 = VFNMA(vi, S5, D10); E10 = VFNMA(vr, S5, E10);
          D11 = VFMA(vi, S6, D11); E11 = VFMA(vr, S6, E11);
        }
        VS(PSCR + 96, D7); VS(PSCR + 104, E7);
        VS(PSCR + 112, D8); VS(PSCR + 120, E8);
        VS(PSCR + 128, D9); VS(PSCR + 136, E9);
        VS(PSCR + 144, D10); VS(PSCR + 152, E10);
        VS(PSCR + 160, D11); VS(PSCR + 168, E11);
        }
        VS(USCR + 384, VL(pr)); VS(USCR + 392, VL(pi));  // save x0 (row 0 gets overwritten)
        {
        V C1 = VSET1(0x1.ed037ea3d2dbcp-1);
        V C2 = VSET1(0x1.b57675cf309eep-1);
        V C3 = VSET1(0x1.5d779b07cfef7p-1);
        V C4 = VSET1(0x1.d71b4a0c5a6c9p-2);
        V C5 = VSET1(0x1.a0ad8bd1e2881p-3);
        V C6 = VSET1(-0x1.17855b599f3b2p-4);
        V C7 = VSET1(-0x1.56eaae597c776p-2);
        V C8 = VSET1(-0x1.2742a4a775cfap-1);
        V C9 = VSET1(-0x1.8d2a07c16d46ep-1);
        V C10 = VSET1(-0x1.d59cb83ef99bcp-1);
        V C11 = VSET1(-0x1.fb3b3035aa6ccp-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A1 = x0r, B1 = x0i;
        V A2 = x0r, B2 = x0i;
        V A3 = x0r, B3 = x0i;
        V A4 = x0r, B4 = x0i;
        V A5 = x0r, B5 = x0i;
        V A6 = x0r, B6 = x0i;
        V s0r = x0r, s0i = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A1 = VFMA(ur, C1, A1); B1 = VFMA(ui, C1, B1);
          A2 = VFMA(ur, C2, A2); B2 = VFMA(ui, C2, B2);
          A3 = VFMA(ur, C3, A3); B3 = VFMA(ui, C3, B3);
          A4 = VFMA(ur, C4, A4); B4 = VFMA(ui, C4, B4);
          A5 = VFMA(ur, C5, A5); B5 = VFMA(ui, C5, B5);
          A6 = VFMA(ur, C6, A6); B6 = VFMA(ui, C6, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A1 = VFMA(ur, C2, A1); B1 = VFMA(ui, C2, B1);
          A2 = VFMA(ur, C4, A2); B2 = VFMA(ui, C4, B2);
          A3 = VFMA(ur, C6, A3); B3 = VFMA(ui, C6, B3);
          A4 = VFMA(ur, C8, A4); B4 = VFMA(ui, C8, B4);
          A5 = VFMA(ur, C10, A5); B5 = VFMA(ui, C10, B5);
          A6 = VFMA(ur, C11, A6); B6 = VFMA(ui, C11, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A1 = VFMA(ur, C3, A1); B1 = VFMA(ui, C3, B1);
          A2 = VFMA(ur, C6, A2); B2 = VFMA(ui, C6, B2);
          A3 = VFMA(ur, C9, A3); B3 = VFMA(ui, C9, B3);
          A4 = VFMA(ur, C11, A4); B4 = VFMA(ui, C11, B4);
          A5 = VFMA(ur, C8, A5); B5 = VFMA(ui, C8, B5);
          A6 = VFMA(ur, C5, A6); B6 = VFMA(ui, C5, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A1 = VFMA(ur, C4, A1); B1 = VFMA(ui, C4, B1);
          A2 = VFMA(ur, C8, A2); B2 = VFMA(ui, C8, B2);
          A3 = VFMA(ur, C11, A3); B3 = VFMA(ui, C11, B3);
          A4 = VFMA(ur, C7, A4); B4 = VFMA(ui, C7, B4);
          A5 = VFMA(ur, C3, A5); B5 = VFMA(ui, C3, B5);
          A6 = VFMA(ur, C1, A6); B6 = VFMA(ui, C1, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A1 = VFMA(ur, C5, A1); B1 = VFMA(ui, C5, B1);
          A2 = VFMA(ur, C10, A2); B2 = VFMA(ui, C10, B2);
          A3 = VFMA(ur, C8, A3); B3 = VFMA(ui, C8, B3);
          A4 = VFMA(ur, C3, A4); B4 = VFMA(ui, C3, B4);
          A5 = VFMA(ur, C2, A5); B5 = VFMA(ui, C2, B5);
          A6 = VFMA(ur, C7, A6); B6 = VFMA(ui, C7, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A1 = VFMA(ur, C6, A1); B1 = VFMA(ui, C6, B1);
          A2 = VFMA(ur, C11, A2); B2 = VFMA(ui, C11, B2);
          A3 = VFMA(ur, C5, A3); B3 = VFMA(ui, C5, B3);
          A4 = VFMA(ur, C1, A4); B4 = VFMA(ui, C1, B4);
          A5 = VFMA(ur, C7, A5); B5 = VFMA(ui, C7, B5);
          A6 = VFMA(ur, C10, A6); B6 = VFMA(ui, C10, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 96), ui = VL(USCR + 104);
          A1 = VFMA(ur, C7, A1); B1 = VFMA(ui, C7, B1);
          A2 = VFMA(ur, C9, A2); B2 = VFMA(ui, C9, B2);
          A3 = VFMA(ur, C2, A3); B3 = VFMA(ui, C2, B3);
          A4 = VFMA(ur, C5, A4); B4 = VFMA(ui, C5, B4);
          A5 = VFMA(ur, C11, A5); B5 = VFMA(ui, C11, B5);
          A6 = VFMA(ur, C4, A6); B6 = VFMA(ui, C4, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 112), ui = VL(USCR + 120);
          A1 = VFMA(ur, C8, A1); B1 = VFMA(ui, C8, B1);
          A2 = VFMA(ur, C7, A2); B2 = VFMA(ui, C7, B2);
          A3 = VFMA(ur, C1, A3); B3 = VFMA(ui, C1, B3);
          A4 = VFMA(ur, C9, A4); B4 = VFMA(ui, C9, B4);
          A5 = VFMA(ur, C6, A5); B5 = VFMA(ui, C6, B5);
          A6 = VFMA(ur, C2, A6); B6 = VFMA(ui, C2, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 128), ui = VL(USCR + 136);
          A1 = VFMA(ur, C9, A1); B1 = VFMA(ui, C9, B1);
          A2 = VFMA(ur, C5, A2); B2 = VFMA(ui, C5, B2);
          A3 = VFMA(ur, C4, A3); B3 = VFMA(ui, C4, B3);
          A4 = VFMA(ur, C10, A4); B4 = VFMA(ui, C10, B4);
          A5 = VFMA(ur, C1, A5); B5 = VFMA(ui, C1, B5);
          A6 = VFMA(ur, C8, A6); B6 = VFMA(ui, C8, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 144), ui = VL(USCR + 152);
          A1 = VFMA(ur, C10, A1); B1 = VFMA(ui, C10, B1);
          A2 = VFMA(ur, C3, A2); B2 = VFMA(ui, C3, B2);
          A3 = VFMA(ur, C7, A3); B3 = VFMA(ui, C7, B3);
          A4 = VFMA(ur, C6, A4); B4 = VFMA(ui, C6, B4);
          A5 = VFMA(ur, C4, A5); B5 = VFMA(ui, C4, B5);
          A6 = VFMA(ur, C9, A6); B6 = VFMA(ui, C9, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 160), ui = VL(USCR + 168);
          A1 = VFMA(ur, C11, A1); B1 = VFMA(ui, C11, B1);
          A2 = VFMA(ur, C1, A2); B2 = VFMA(ui, C1, B2);
          A3 = VFMA(ur, C10, A3); B3 = VFMA(ui, C10, B3);
          A4 = VFMA(ur, C2, A4); B4 = VFMA(ui, C2, B4);
          A5 = VFMA(ur, C9, A5); B5 = VFMA(ui, C9, B5);
          A6 = VFMA(ur, C3, A6); B6 = VFMA(ui, C3, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V zr_ = VADD(s0r, VL(cr + 0)), zi_ = VADD(s0i, VL(ci + 0));
          MAP2(zr_, zi_);
          VS(pr + 0, zr_); VS(pi + 0, zi_); }
        { V Dk = VL(PSCR + 0), Ek = VL(PSCR + 8);
          V Xr = VADD(A1, Dk), Xi = VSUB(B1, Ek);
          V Yr = VSUB(A1, Dk), Yi = VADD(B1, Ek);
        { V zr_ = VADD(Xr, VL(cr + 4232)), zi_ = VADD(Xi, VL(ci + 4232));
          MAP2(zr_, zi_);
          VS(pr + 4232, zr_); VS(pi + 4232, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 93104)), zi_ = VADD(Yi, VL(ci + 93104));
          MAP2(zr_, zi_);
          VS(pr + 93104, zr_); VS(pi + 93104, zi_); }
        }
        { V Dk = VL(PSCR + 16), Ek = VL(PSCR + 24);
          V Xr = VADD(A2, Dk), Xi = VSUB(B2, Ek);
          V Yr = VSUB(A2, Dk), Yi = VADD(B2, Ek);
        { V zr_ = VADD(Xr, VL(cr + 8464)), zi_ = VADD(Xi, VL(ci + 8464));
          MAP2(zr_, zi_);
          VS(pr + 8464, zr_); VS(pi + 8464, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 88872)), zi_ = VADD(Yi, VL(ci + 88872));
          MAP2(zr_, zi_);
          VS(pr + 88872, zr_); VS(pi + 88872, zi_); }
        }
        { V Dk = VL(PSCR + 32), Ek = VL(PSCR + 40);
          V Xr = VADD(A3, Dk), Xi = VSUB(B3, Ek);
          V Yr = VSUB(A3, Dk), Yi = VADD(B3, Ek);
        { V zr_ = VADD(Xr, VL(cr + 12696)), zi_ = VADD(Xi, VL(ci + 12696));
          MAP2(zr_, zi_);
          VS(pr + 12696, zr_); VS(pi + 12696, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 84640)), zi_ = VADD(Yi, VL(ci + 84640));
          MAP2(zr_, zi_);
          VS(pr + 84640, zr_); VS(pi + 84640, zi_); }
        }
        { V Dk = VL(PSCR + 48), Ek = VL(PSCR + 56);
          V Xr = VADD(A4, Dk), Xi = VSUB(B4, Ek);
          V Yr = VSUB(A4, Dk), Yi = VADD(B4, Ek);
        { V zr_ = VADD(Xr, VL(cr + 16928)), zi_ = VADD(Xi, VL(ci + 16928));
          MAP2(zr_, zi_);
          VS(pr + 16928, zr_); VS(pi + 16928, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 80408)), zi_ = VADD(Yi, VL(ci + 80408));
          MAP2(zr_, zi_);
          VS(pr + 80408, zr_); VS(pi + 80408, zi_); }
        }
        { V Dk = VL(PSCR + 64), Ek = VL(PSCR + 72);
          V Xr = VADD(A5, Dk), Xi = VSUB(B5, Ek);
          V Yr = VSUB(A5, Dk), Yi = VADD(B5, Ek);
        { V zr_ = VADD(Xr, VL(cr + 21160)), zi_ = VADD(Xi, VL(ci + 21160));
          MAP2(zr_, zi_);
          VS(pr + 21160, zr_); VS(pi + 21160, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 76176)), zi_ = VADD(Yi, VL(ci + 76176));
          MAP2(zr_, zi_);
          VS(pr + 76176, zr_); VS(pi + 76176, zi_); }
        }
        { V Dk = VL(PSCR + 80), Ek = VL(PSCR + 88);
          V Xr = VADD(A6, Dk), Xi = VSUB(B6, Ek);
          V Yr = VSUB(A6, Dk), Yi = VADD(B6, Ek);
        { V zr_ = VADD(Xr, VL(cr + 25392)), zi_ = VADD(Xi, VL(ci + 25392));
          MAP2(zr_, zi_);
          VS(pr + 25392, zr_); VS(pi + 25392, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 71944)), zi_ = VADD(Yi, VL(ci + 71944));
          MAP2(zr_, zi_);
          VS(pr + 71944, zr_); VS(pi + 71944, zi_); }
        }
        }
        {
        V C1 = VSET1(0x1.ed037ea3d2dbcp-1);
        V C2 = VSET1(0x1.b57675cf309eep-1);
        V C3 = VSET1(0x1.5d779b07cfef7p-1);
        V C4 = VSET1(0x1.d71b4a0c5a6c9p-2);
        V C5 = VSET1(0x1.a0ad8bd1e2881p-3);
        V C6 = VSET1(-0x1.17855b599f3b2p-4);
        V C7 = VSET1(-0x1.56eaae597c776p-2);
        V C8 = VSET1(-0x1.2742a4a775cfap-1);
        V C9 = VSET1(-0x1.8d2a07c16d46ep-1);
        V C10 = VSET1(-0x1.d59cb83ef99bcp-1);
        V C11 = VSET1(-0x1.fb3b3035aa6ccp-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A7 = x0r, B7 = x0i;
        V A8 = x0r, B8 = x0i;
        V A9 = x0r, B9 = x0i;
        V A10 = x0r, B10 = x0i;
        V A11 = x0r, B11 = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A7 = VFMA(ur, C7, A7); B7 = VFMA(ui, C7, B7);
          A8 = VFMA(ur, C8, A8); B8 = VFMA(ui, C8, B8);
          A9 = VFMA(ur, C9, A9); B9 = VFMA(ui, C9, B9);
          A10 = VFMA(ur, C10, A10); B10 = VFMA(ui, C10, B10);
          A11 = VFMA(ur, C11, A11); B11 = VFMA(ui, C11, B11);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A7 = VFMA(ur, C9, A7); B7 = VFMA(ui, C9, B7);
          A8 = VFMA(ur, C7, A8); B8 = VFMA(ui, C7, B8);
          A9 = VFMA(ur, C5, A9); B9 = VFMA(ui, C5, B9);
          A10 = VFMA(ur, C3, A10); B10 = VFMA(ui, C3, B10);
          A11 = VFMA(ur, C1, A11); B11 = VFMA(ui, C1, B11);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A7 = VFMA(ur, C2, A7); B7 = VFMA(ui, C2, B7);
          A8 = VFMA(ur, C1, A8); B8 = VFMA(ui, C1, B8);
          A9 = VFMA(ur, C4, A9); B9 = VFMA(ui, C4, B9);
          A10 = VFMA(ur, C7, A10); B10 = VFMA(ui, C7, B10);
          A11 = VFMA(ur, C10, A11); B11 = VFMA(ui, C10, B11);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A7 = VFMA(ur, C5, A7); B7 = VFMA(ui, C5, B7);
          A8 = VFMA(ur, C9, A8); B8 = VFMA(ui, C9, B8);
          A9 = VFMA(ur, C10, A9); B9 = VFMA(ui, C10, B9);
          A10 = VFMA(ur, C6, A10); B10 = VFMA(ui, C6, B10);
          A11 = VFMA(ur, C2, A11); B11 = VFMA(ui, C2, B11);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A7 = VFMA(ur, C11, A7); B7 = VFMA(ui, C11, B7);
          A8 = VFMA(ur, C6, A8); B8 = VFMA(ui, C6, B8);
          A9 = VFMA(ur, C1, A9); B9 = VFMA(ui, C1, B9);
          A10 = VFMA(ur, C4, A10); B10 = VFMA(ui, C4, B10);
          A11 = VFMA(ur, C9, A11); B11 = VFMA(ui, C9, B11);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A7 = VFMA(ur, C4, A7); B7 = VFMA(ui, C4, B7);
          A8 = VFMA(ur, C2, A8); B8 = VFMA(ui, C2, B8);
          A9 = VFMA(ur, C8, A9); B9 = VFMA(ui, C8, B9);
          A10 = VFMA(ur, C9, A10); B10 = VFMA(ui, C9, B10);
          A11 = VFMA(ur, C3, A11); B11 = VFMA(ui, C3, B11);
        }
        { V ur = VL(USCR + 96), ui = VL(USCR + 104);
          A7 = VFMA(ur, C3, A7); B7 = VFMA(ui, C3, B7);
          A8 = VFMA(ur, C10, A8); B8 = VFMA(ui, C10, B8);
          A9 = VFMA(ur, C6, A9); B9 = VFMA(ui, C6, B9);
          A10 = VFMA(ur, C1, A10); B10 = VFMA(ui, C1, B10);
          A11 = VFMA(ur, C8, A11); B11 = VFMA(ui, C8, B11);
        }
        { V ur = VL(USCR + 112), ui = VL(USCR + 120);
          A7 = VFMA(ur, C10, A7); B7 = VFMA(ui, C10, B7);
          A8 = VFMA(ur, C5, A8); B8 = VFMA(ui, C5, B8);
          A9 = VFMA(ur, C3, A9); B9 = VFMA(ui, C3, B9);
          A10 = VFMA(ur, C11, A10); B10 = VFMA(ui, C11, B10);
          A11 = VFMA(ur, C4, A11); B11 = VFMA(ui, C4, B11);
        }
        { V ur = VL(USCR + 128), ui = VL(USCR + 136);
          A7 = VFMA(ur, C6, A7); B7 = VFMA(ui, C6, B7);
          A8 = VFMA(ur, C3, A8); B8 = VFMA(ui, C3, B8);
          A9 = VFMA(ur, C11, A9); B9 = VFMA(ui, C11, B9);
          A10 = VFMA(ur, C2, A10); B10 = VFMA(ui, C2, B10);
          A11 = VFMA(ur, C7, A11); B11 = VFMA(ui, C7, B11);
        }
        { V ur = VL(USCR + 144), ui = VL(USCR + 152);
          A7 = VFMA(ur, C1, A7); B7 = VFMA(ui, C1, B7);
          A8 = VFMA(ur, C11, A8); B8 = VFMA(ui, C11, B8);
          A9 = VFMA(ur, C2, A9); B9 = VFMA(ui, C2, B9);
          A10 = VFMA(ur, C8, A10); B10 = VFMA(ui, C8, B10);
          A11 = VFMA(ur, C5, A11); B11 = VFMA(ui, C5, B11);
        }
        { V ur = VL(USCR + 160), ui = VL(USCR + 168);
          A7 = VFMA(ur, C8, A7); B7 = VFMA(ui, C8, B7);
          A8 = VFMA(ur, C4, A8); B8 = VFMA(ui, C4, B8);
          A9 = VFMA(ur, C7, A9); B9 = VFMA(ui, C7, B9);
          A10 = VFMA(ur, C5, A10); B10 = VFMA(ui, C5, B10);
          A11 = VFMA(ur, C6, A11); B11 = VFMA(ui, C6, B11);
        }
        { V Dk = VL(PSCR + 96), Ek = VL(PSCR + 104);
          V Xr = VADD(A7, Dk), Xi = VSUB(B7, Ek);
          V Yr = VSUB(A7, Dk), Yi = VADD(B7, Ek);
        { V zr_ = VADD(Xr, VL(cr + 29624)), zi_ = VADD(Xi, VL(ci + 29624));
          MAP2(zr_, zi_);
          VS(pr + 29624, zr_); VS(pi + 29624, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 67712)), zi_ = VADD(Yi, VL(ci + 67712));
          MAP2(zr_, zi_);
          VS(pr + 67712, zr_); VS(pi + 67712, zi_); }
        }
        { V Dk = VL(PSCR + 112), Ek = VL(PSCR + 120);
          V Xr = VADD(A8, Dk), Xi = VSUB(B8, Ek);
          V Yr = VSUB(A8, Dk), Yi = VADD(B8, Ek);
        { V zr_ = VADD(Xr, VL(cr + 33856)), zi_ = VADD(Xi, VL(ci + 33856));
          MAP2(zr_, zi_);
          VS(pr + 33856, zr_); VS(pi + 33856, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 63480)), zi_ = VADD(Yi, VL(ci + 63480));
          MAP2(zr_, zi_);
          VS(pr + 63480, zr_); VS(pi + 63480, zi_); }
        }
        { V Dk = VL(PSCR + 128), Ek = VL(PSCR + 136);
          V Xr = VADD(A9, Dk), Xi = VSUB(B9, Ek);
          V Yr = VSUB(A9, Dk), Yi = VADD(B9, Ek);
        { V zr_ = VADD(Xr, VL(cr + 38088)), zi_ = VADD(Xi, VL(ci + 38088));
          MAP2(zr_, zi_);
          VS(pr + 38088, zr_); VS(pi + 38088, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 59248)), zi_ = VADD(Yi, VL(ci + 59248));
          MAP2(zr_, zi_);
          VS(pr + 59248, zr_); VS(pi + 59248, zi_); }
        }
        { V Dk = VL(PSCR + 144), Ek = VL(PSCR + 152);
          V Xr = VADD(A10, Dk), Xi = VSUB(B10, Ek);
          V Yr = VSUB(A10, Dk), Yi = VADD(B10, Ek);
        { V zr_ = VADD(Xr, VL(cr + 42320)), zi_ = VADD(Xi, VL(ci + 42320));
          MAP2(zr_, zi_);
          VS(pr + 42320, zr_); VS(pi + 42320, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 55016)), zi_ = VADD(Yi, VL(ci + 55016));
          MAP2(zr_, zi_);
          VS(pr + 55016, zr_); VS(pi + 55016, zi_); }
        }
        { V Dk = VL(PSCR + 160), Ek = VL(PSCR + 168);
          V Xr = VADD(A11, Dk), Xi = VSUB(B11, Ek);
          V Yr = VSUB(A11, Dk), Yi = VADD(B11, Ek);
        { V zr_ = VADD(Xr, VL(cr + 46552)), zi_ = VADD(Xi, VL(ci + 46552));
          MAP2(zr_, zi_);
          VS(pr + 46552, zr_); VS(pi + 46552, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 50784)), zi_ = VADD(Yi, VL(ci + 50784));
          MAP2(zr_, zi_);
          VS(pr + 50784, zr_); VS(pi + 50784, zi_); }
        }
        }
    }
}
static __attribute__((noinline)) void p23_yym(double* PR, double* PI, const double* CR, const double* CI, long n, long pstep){
    #pragma GCC unroll 1
    for (long q_ = 0; q_ < n; q_++) {
        double* pr = PR + q_*pstep; double* pi = PI + q_*pstep;
        const double* cr = CR + q_*pstep; const double* ci = CI + q_*pstep;
        {
        V S1 = VSET1(0x1.14459ad2be466p-2);
        V S2 = VSET1(0x1.0a06e851db7cap-1);
        V S3 = VSET1(0x1.763021aaa15d9p-1);
        V S4 = VSET1(0x1.c698e42f47b09p-1);
        V S5 = VSET1(0x1.f54a827142577p-1);
        V S6 = VSET1(0x1.fece70dfd3efbp-1);
        V S7 = VSET1(0x1.e270060999288p-1);
        V S8 = VSET1(0x1.a249e0b897caap-1);
        V S9 = VSET1(0x1.431df5838f7f1p-1);
        V S10 = VSET1(0x1.97f6748e524b1p-2);
        V S11 = VSET1(0x1.16de8a4564f1cp-3);
        V D1 = _mm512_setzero_pd(), E1 = _mm512_setzero_pd();
        V D2 = _mm512_setzero_pd(), E2 = _mm512_setzero_pd();
        V D3 = _mm512_setzero_pd(), E3 = _mm512_setzero_pd();
        V D4 = _mm512_setzero_pd(), E4 = _mm512_setzero_pd();
        V D5 = _mm512_setzero_pd(), E5 = _mm512_setzero_pd();
        V D6 = _mm512_setzero_pd(), E6 = _mm512_setzero_pd();
        { V ar = VL(pr + 184), ai = VL(pi + 184);
          V br = VL(pr + 4048), bi = VL(pi + 4048);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 0, VADD(ar,br)); VS(USCR + 8, VADD(ai,bi));
          D1 = VFMA(vi, S1, D1); E1 = VFMA(vr, S1, E1);
          D2 = VFMA(vi, S2, D2); E2 = VFMA(vr, S2, E2);
          D3 = VFMA(vi, S3, D3); E3 = VFMA(vr, S3, E3);
          D4 = VFMA(vi, S4, D4); E4 = VFMA(vr, S4, E4);
          D5 = VFMA(vi, S5, D5); E5 = VFMA(vr, S5, E5);
          D6 = VFMA(vi, S6, D6); E6 = VFMA(vr, S6, E6);
        }
        { V ar = VL(pr + 368), ai = VL(pi + 368);
          V br = VL(pr + 3864), bi = VL(pi + 3864);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 16, VADD(ar,br)); VS(USCR + 24, VADD(ai,bi));
          D1 = VFMA(vi, S2, D1); E1 = VFMA(vr, S2, E1);
          D2 = VFMA(vi, S4, D2); E2 = VFMA(vr, S4, E2);
          D3 = VFMA(vi, S6, D3); E3 = VFMA(vr, S6, E3);
          D4 = VFMA(vi, S8, D4); E4 = VFMA(vr, S8, E4);
          D5 = VFMA(vi, S10, D5); E5 = VFMA(vr, S10, E5);
          D6 = VFNMA(vi, S11, D6); E6 = VFNMA(vr, S11, E6);
        }
        { V ar = VL(pr + 552), ai = VL(pi + 552);
          V br = VL(pr + 3680), bi = VL(pi + 3680);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 32, VADD(ar,br)); VS(USCR + 40, VADD(ai,bi));
          D1 = VFMA(vi, S3, D1); E1 = VFMA(vr, S3, E1);
          D2 = VFMA(vi, S6, D2); E2 = VFMA(vr, S6, E2);
          D3 = VFMA(vi, S9, D3); E3 = VFMA(vr, S9, E3);
          D4 = VFNMA(vi, S11, D4); E4 = VFNMA(vr, S11, E4);
          D5 = VFNMA(vi, S8, D5); E5 = VFNMA(vr, S8, E5);
          D6 = VFNMA(vi, S5, D6); E6 = VFNMA(vr, S5, E6);
        }
        { V ar = VL(pr + 736), ai = VL(pi + 736);
          V br = VL(pr + 3496), bi = VL(pi + 3496);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 48, VADD(ar,br)); VS(USCR + 56, VADD(ai,bi));
          D1 = VFMA(vi, S4, D1); E1 = VFMA(vr, S4, E1);
          D2 = VFMA(vi, S8, D2); E2 = VFMA(vr, S8, E2);
          D3 = VFNMA(vi, S11, D3); E3 = VFNMA(vr, S11, E3);
          D4 = VFNMA(vi, S7, D4); E4 = VFNMA(vr, S7, E4);
          D5 = VFNMA(vi, S3, D5); E5 = VFNMA(vr, S3, E5);
          D6 = VFMA(vi, S1, D6); E6 = VFMA(vr, S1, E6);
        }
        { V ar = VL(pr + 920), ai = VL(pi + 920);
          V br = VL(pr + 3312), bi = VL(pi + 3312);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 64, VADD(ar,br)); VS(USCR + 72, VADD(ai,bi));
          D1 = VFMA(vi, S5, D1); E1 = VFMA(vr, S5, E1);
          D2 = VFMA(vi, S10, D2); E2 = VFMA(vr, S10, E2);
          D3 = VFNMA(vi, S8, D3); E3 = VFNMA(vr, S8, E3);
          D4 = VFNMA(vi, S3, D4); E4 = VFNMA(vr, S3, E4);
          D5 = VFMA(vi, S2, D5); E5 = VFMA(vr, S2, E5);
          D6 = VFMA(vi, S7, D6); E6 = VFMA(vr, S7, E6);
        }
        { V ar = VL(pr + 1104), ai = VL(pi + 1104);
          V br = VL(pr + 3128), bi = VL(pi + 3128);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 80, VADD(ar,br)); VS(USCR + 88, VADD(ai,bi));
          D1 = VFMA(vi, S6, D1); E1 = VFMA(vr, S6, E1);
          D2 = VFNMA(vi, S11, D2); E2 = VFNMA(vr, S11, E2);
          D3 = VFNMA(vi, S5, D3); E3 = VFNMA(vr, S5, E3);
          D4 = VFMA(vi, S1, D4); E4 = VFMA(vr, S1, E4);
          D5 = VFMA(vi, S7, D5); E5 = VFMA(vr, S7, E5);
          D6 = VFNMA(vi, S10, D6); E6 = VFNMA(vr, S10, E6);
        }
        { V ar = VL(pr + 1288), ai = VL(pi + 1288);
          V br = VL(pr + 2944), bi = VL(pi + 2944);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 96, VADD(ar,br)); VS(USCR + 104, VADD(ai,bi));
          D1 = VFMA(vi, S7, D1); E1 = VFMA(vr, S7, E1);
          D2 = VFNMA(vi, S9, D2); E2 = VFNMA(vr, S9, E2);
          D3 = VFNMA(vi, S2, D3); E3 = VFNMA(vr, S2, E3);
          D4 = VFMA(vi, S5, D4); E4 = VFMA(vr, S5, E4);
          D5 = VFNMA(vi, S11, D5); E5 = VFNMA(vr, S11, E5);
          D6 = VFNMA(vi, S4, D6); E6 = VFNMA(vr, S4, E6);
        }
        { V ar = VL(pr + 1472), ai = VL(pi + 1472);
          V br = VL(pr + 2760), bi = VL(pi + 2760);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 112, VADD(ar,br)); VS(USCR + 120, VADD(ai,bi));
          D1 = VFMA(vi, S8, D1); E1 = VFMA(vr, S8, E1);
          D2 = VFNMA(vi, S7, D2); E2 = VFNMA(vr, S7, E2);
          D3 = VFMA(vi, S1, D3); E3 = VFMA(vr, S1, E3);
          D4 = VFMA(vi, S9, D4); E4 = VFMA(vr, S9, E4);
          D5 = VFNMA(vi, S6, D5); E5 = VFNMA(vr, S6, E5);
          D6 = VFMA(vi, S2, D6); E6 = VFMA(vr, S2, E6);
        }
        { V ar = VL(pr + 1656), ai = VL(pi + 1656);
          V br = VL(pr + 2576), bi = VL(pi + 2576);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 128, VADD(ar,br)); VS(USCR + 136, VADD(ai,bi));
          D1 = VFMA(vi, S9, D1); E1 = VFMA(vr, S9, E1);
          D2 = VFNMA(vi, S5, D2); E2 = VFNMA(vr, S5, E2);
          D3 = VFMA(vi, S4, D3); E3 = VFMA(vr, S4, E3);
          D4 = VFNMA(vi, S10, D4); E4 = VFNMA(vr, S10, E4);
          D5 = VFNMA(vi, S1, D5); E5 = VFNMA(vr, S1, E5);
          D6 = VFMA(vi, S8, D6); E6 = VFMA(vr, S8, E6);
        }
        { V ar = VL(pr + 1840), ai = VL(pi + 1840);
          V br = VL(pr + 2392), bi = VL(pi + 2392);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 144, VADD(ar,br)); VS(USCR + 152, VADD(ai,bi));
          D1 = VFMA(vi, S10, D1); E1 = VFMA(vr, S10, E1);
          D2 = VFNMA(vi, S3, D2); E2 = VFNMA(vr, S3, E2);
          D3 = VFMA(vi, S7, D3); E3 = VFMA(vr, S7, E3);
          D4 = VFNMA(vi, S6, D4); E4 = VFNMA(vr, S6, E4);
          D5 = VFMA(vi, S4, D5); E5 = VFMA(vr, S4, E5);
          D6 = VFNMA(vi, S9, D6); E6 = VFNMA(vr, S9, E6);
        }
        { V ar = VL(pr + 2024), ai = VL(pi + 2024);
          V br = VL(pr + 2208), bi = VL(pi + 2208);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          VS(USCR + 160, VADD(ar,br)); VS(USCR + 168, VADD(ai,bi));
          D1 = VFMA(vi, S11, D1); E1 = VFMA(vr, S11, E1);
          D2 = VFNMA(vi, S1, D2); E2 = VFNMA(vr, S1, E2);
          D3 = VFMA(vi, S10, D3); E3 = VFMA(vr, S10, E3);
          D4 = VFNMA(vi, S2, D4); E4 = VFNMA(vr, S2, E4);
          D5 = VFMA(vi, S9, D5); E5 = VFMA(vr, S9, E5);
          D6 = VFNMA(vi, S3, D6); E6 = VFNMA(vr, S3, E6);
        }
        VS(PSCR + 0, D1); VS(PSCR + 8, E1);
        VS(PSCR + 16, D2); VS(PSCR + 24, E2);
        VS(PSCR + 32, D3); VS(PSCR + 40, E3);
        VS(PSCR + 48, D4); VS(PSCR + 56, E4);
        VS(PSCR + 64, D5); VS(PSCR + 72, E5);
        VS(PSCR + 80, D6); VS(PSCR + 88, E6);
        }
        {
        V S1 = VSET1(0x1.14459ad2be466p-2);
        V S2 = VSET1(0x1.0a06e851db7cap-1);
        V S3 = VSET1(0x1.763021aaa15d9p-1);
        V S4 = VSET1(0x1.c698e42f47b09p-1);
        V S5 = VSET1(0x1.f54a827142577p-1);
        V S6 = VSET1(0x1.fece70dfd3efbp-1);
        V S7 = VSET1(0x1.e270060999288p-1);
        V S8 = VSET1(0x1.a249e0b897caap-1);
        V S9 = VSET1(0x1.431df5838f7f1p-1);
        V S10 = VSET1(0x1.97f6748e524b1p-2);
        V S11 = VSET1(0x1.16de8a4564f1cp-3);
        V D7 = _mm512_setzero_pd(), E7 = _mm512_setzero_pd();
        V D8 = _mm512_setzero_pd(), E8 = _mm512_setzero_pd();
        V D9 = _mm512_setzero_pd(), E9 = _mm512_setzero_pd();
        V D10 = _mm512_setzero_pd(), E10 = _mm512_setzero_pd();
        V D11 = _mm512_setzero_pd(), E11 = _mm512_setzero_pd();
        { V ar = VL(pr + 184), ai = VL(pi + 184);
          V br = VL(pr + 4048), bi = VL(pi + 4048);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S7, D7); E7 = VFMA(vr, S7, E7);
          D8 = VFMA(vi, S8, D8); E8 = VFMA(vr, S8, E8);
          D9 = VFMA(vi, S9, D9); E9 = VFMA(vr, S9, E9);
          D10 = VFMA(vi, S10, D10); E10 = VFMA(vr, S10, E10);
          D11 = VFMA(vi, S11, D11); E11 = VFMA(vr, S11, E11);
        }
        { V ar = VL(pr + 368), ai = VL(pi + 368);
          V br = VL(pr + 3864), bi = VL(pi + 3864);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S9, D7); E7 = VFNMA(vr, S9, E7);
          D8 = VFNMA(vi, S7, D8); E8 = VFNMA(vr, S7, E8);
          D9 = VFNMA(vi, S5, D9); E9 = VFNMA(vr, S5, E9);
          D10 = VFNMA(vi, S3, D10); E10 = VFNMA(vr, S3, E10);
          D11 = VFNMA(vi, S1, D11); E11 = VFNMA(vr, S1, E11);
        }
        { V ar = VL(pr + 552), ai = VL(pi + 552);
          V br = VL(pr + 3680), bi = VL(pi + 3680);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S2, D7); E7 = VFNMA(vr, S2, E7);
          D8 = VFMA(vi, S1, D8); E8 = VFMA(vr, S1, E8);
          D9 = VFMA(vi, S4, D9); E9 = VFMA(vr, S4, E9);
          D10 = VFMA(vi, S7, D10); E10 = VFMA(vr, S7, E10);
          D11 = VFMA(vi, S10, D11); E11 = VFMA(vr, S10, E11);
        }
        { V ar = VL(pr + 736), ai = VL(pi + 736);
          V br = VL(pr + 3496), bi = VL(pi + 3496);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S5, D7); E7 = VFMA(vr, S5, E7);
          D8 = VFMA(vi, S9, D8); E8 = VFMA(vr, S9, E8);
          D9 = VFNMA(vi, S10, D9); E9 = VFNMA(vr, S10, E9);
          D10 = VFNMA(vi, S6, D10); E10 = VFNMA(vr, S6, E10);
          D11 = VFNMA(vi, S2, D11); E11 = VFNMA(vr, S2, E11);
        }
        { V ar = VL(pr + 920), ai = VL(pi + 920);
          V br = VL(pr + 3312), bi = VL(pi + 3312);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S11, D7); E7 = VFNMA(vr, S11, E7);
          D8 = VFNMA(vi, S6, D8); E8 = VFNMA(vr, S6, E8);
          D9 = VFNMA(vi, S1, D9); E9 = VFNMA(vr, S1, E9);
          D10 = VFMA(vi, S4, D10); E10 = VFMA(vr, S4, E10);
          D11 = VFMA(vi, S9, D11); E11 = VFMA(vr, S9, E11);
        }
        { V ar = VL(pr + 1104), ai = VL(pi + 1104);
          V br = VL(pr + 3128), bi = VL(pi + 3128);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S4, D7); E7 = VFNMA(vr, S4, E7);
          D8 = VFMA(vi, S2, D8); E8 = VFMA(vr, S2, E8);
          D9 = VFMA(vi, S8, D9); E9 = VFMA(vr, S8, E9);
          D10 = VFNMA(vi, S9, D10); E10 = VFNMA(vr, S9, E10);
          D11 = VFNMA(vi, S3, D11); E11 = VFNMA(vr, S3, E11);
        }
        { V ar = VL(pr + 1288), ai = VL(pi + 1288);
          V br = VL(pr + 2944), bi = VL(pi + 2944);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S3, D7); E7 = VFMA(vr, S3, E7);
          D8 = VFMA(vi, S10, D8); E8 = VFMA(vr, S10, E8);
          D9 = VFNMA(vi, S6, D9); E9 = VFNMA(vr, S6, E9);
          D10 = VFMA(vi, S1, D10); E10 = VFMA(vr, S1, E10);
          D11 = VFMA(vi, S8, D11); E11 = VFMA(vr, S8, E11);
        }
        { V ar = VL(pr + 1472), ai = VL(pi + 1472);
          V br = VL(pr + 2760), bi = VL(pi + 2760);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S10, D7); E7 = VFMA(vr, S10, E7);
          D8 = VFNMA(vi, S5, D8); E8 = VFNMA(vr, S5, E8);
          D9 = VFMA(vi, S3, D9); E9 = VFMA(vr, S3, E9);
          D10 = VFMA(vi, S11, D10); E10 = VFMA(vr, S11, E10);
          D11 = VFNMA(vi, S4, D11); E11 = VFNMA(vr, S4, E11);
        }
        { V ar = VL(pr + 1656), ai = VL(pi + 1656);
          V br = VL(pr + 2576), bi = VL(pi + 2576);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFNMA(vi, S6, D7); E7 = VFNMA(vr, S6, E7);
          D8 = VFMA(vi, S3, D8); E8 = VFMA(vr, S3, E8);
          D9 = VFNMA(vi, S11, D9); E9 = VFNMA(vr, S11, E9);
          D10 = VFNMA(vi, S2, D10); E10 = VFNMA(vr, S2, E10);
          D11 = VFMA(vi, S7, D11); E11 = VFMA(vr, S7, E11);
        }
        { V ar = VL(pr + 1840), ai = VL(pi + 1840);
          V br = VL(pr + 2392), bi = VL(pi + 2392);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S1, D7); E7 = VFMA(vr, S1, E7);
          D8 = VFMA(vi, S11, D8); E8 = VFMA(vr, S11, E8);
          D9 = VFNMA(vi, S2, D9); E9 = VFNMA(vr, S2, E9);
          D10 = VFMA(vi, S8, D10); E10 = VFMA(vr, S8, E10);
          D11 = VFNMA(vi, S5, D11); E11 = VFNMA(vr, S5, E11);
        }
        { V ar = VL(pr + 2024), ai = VL(pi + 2024);
          V br = VL(pr + 2208), bi = VL(pi + 2208);
          V vr = VSUB(ar,br), vi = VSUB(ai,bi);
          D7 = VFMA(vi, S8, D7); E7 = VFMA(vr, S8, E7);
          D8 = VFNMA(vi, S4, D8); E8 = VFNMA(vr, S4, E8);
          D9 = VFMA(vi, S7, D9); E9 = VFMA(vr, S7, E9);
          D10 = VFNMA(vi, S5, D10); E10 = VFNMA(vr, S5, E10);
          D11 = VFMA(vi, S6, D11); E11 = VFMA(vr, S6, E11);
        }
        VS(PSCR + 96, D7); VS(PSCR + 104, E7);
        VS(PSCR + 112, D8); VS(PSCR + 120, E8);
        VS(PSCR + 128, D9); VS(PSCR + 136, E9);
        VS(PSCR + 144, D10); VS(PSCR + 152, E10);
        VS(PSCR + 160, D11); VS(PSCR + 168, E11);
        }
        VS(USCR + 384, VL(pr)); VS(USCR + 392, VL(pi));  // save x0 (row 0 gets overwritten)
        {
        V C1 = VSET1(0x1.ed037ea3d2dbcp-1);
        V C2 = VSET1(0x1.b57675cf309eep-1);
        V C3 = VSET1(0x1.5d779b07cfef7p-1);
        V C4 = VSET1(0x1.d71b4a0c5a6c9p-2);
        V C5 = VSET1(0x1.a0ad8bd1e2881p-3);
        V C6 = VSET1(-0x1.17855b599f3b2p-4);
        V C7 = VSET1(-0x1.56eaae597c776p-2);
        V C8 = VSET1(-0x1.2742a4a775cfap-1);
        V C9 = VSET1(-0x1.8d2a07c16d46ep-1);
        V C10 = VSET1(-0x1.d59cb83ef99bcp-1);
        V C11 = VSET1(-0x1.fb3b3035aa6ccp-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A1 = x0r, B1 = x0i;
        V A2 = x0r, B2 = x0i;
        V A3 = x0r, B3 = x0i;
        V A4 = x0r, B4 = x0i;
        V A5 = x0r, B5 = x0i;
        V A6 = x0r, B6 = x0i;
        V s0r = x0r, s0i = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A1 = VFMA(ur, C1, A1); B1 = VFMA(ui, C1, B1);
          A2 = VFMA(ur, C2, A2); B2 = VFMA(ui, C2, B2);
          A3 = VFMA(ur, C3, A3); B3 = VFMA(ui, C3, B3);
          A4 = VFMA(ur, C4, A4); B4 = VFMA(ui, C4, B4);
          A5 = VFMA(ur, C5, A5); B5 = VFMA(ui, C5, B5);
          A6 = VFMA(ur, C6, A6); B6 = VFMA(ui, C6, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A1 = VFMA(ur, C2, A1); B1 = VFMA(ui, C2, B1);
          A2 = VFMA(ur, C4, A2); B2 = VFMA(ui, C4, B2);
          A3 = VFMA(ur, C6, A3); B3 = VFMA(ui, C6, B3);
          A4 = VFMA(ur, C8, A4); B4 = VFMA(ui, C8, B4);
          A5 = VFMA(ur, C10, A5); B5 = VFMA(ui, C10, B5);
          A6 = VFMA(ur, C11, A6); B6 = VFMA(ui, C11, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A1 = VFMA(ur, C3, A1); B1 = VFMA(ui, C3, B1);
          A2 = VFMA(ur, C6, A2); B2 = VFMA(ui, C6, B2);
          A3 = VFMA(ur, C9, A3); B3 = VFMA(ui, C9, B3);
          A4 = VFMA(ur, C11, A4); B4 = VFMA(ui, C11, B4);
          A5 = VFMA(ur, C8, A5); B5 = VFMA(ui, C8, B5);
          A6 = VFMA(ur, C5, A6); B6 = VFMA(ui, C5, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A1 = VFMA(ur, C4, A1); B1 = VFMA(ui, C4, B1);
          A2 = VFMA(ur, C8, A2); B2 = VFMA(ui, C8, B2);
          A3 = VFMA(ur, C11, A3); B3 = VFMA(ui, C11, B3);
          A4 = VFMA(ur, C7, A4); B4 = VFMA(ui, C7, B4);
          A5 = VFMA(ur, C3, A5); B5 = VFMA(ui, C3, B5);
          A6 = VFMA(ur, C1, A6); B6 = VFMA(ui, C1, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A1 = VFMA(ur, C5, A1); B1 = VFMA(ui, C5, B1);
          A2 = VFMA(ur, C10, A2); B2 = VFMA(ui, C10, B2);
          A3 = VFMA(ur, C8, A3); B3 = VFMA(ui, C8, B3);
          A4 = VFMA(ur, C3, A4); B4 = VFMA(ui, C3, B4);
          A5 = VFMA(ur, C2, A5); B5 = VFMA(ui, C2, B5);
          A6 = VFMA(ur, C7, A6); B6 = VFMA(ui, C7, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A1 = VFMA(ur, C6, A1); B1 = VFMA(ui, C6, B1);
          A2 = VFMA(ur, C11, A2); B2 = VFMA(ui, C11, B2);
          A3 = VFMA(ur, C5, A3); B3 = VFMA(ui, C5, B3);
          A4 = VFMA(ur, C1, A4); B4 = VFMA(ui, C1, B4);
          A5 = VFMA(ur, C7, A5); B5 = VFMA(ui, C7, B5);
          A6 = VFMA(ur, C10, A6); B6 = VFMA(ui, C10, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 96), ui = VL(USCR + 104);
          A1 = VFMA(ur, C7, A1); B1 = VFMA(ui, C7, B1);
          A2 = VFMA(ur, C9, A2); B2 = VFMA(ui, C9, B2);
          A3 = VFMA(ur, C2, A3); B3 = VFMA(ui, C2, B3);
          A4 = VFMA(ur, C5, A4); B4 = VFMA(ui, C5, B4);
          A5 = VFMA(ur, C11, A5); B5 = VFMA(ui, C11, B5);
          A6 = VFMA(ur, C4, A6); B6 = VFMA(ui, C4, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 112), ui = VL(USCR + 120);
          A1 = VFMA(ur, C8, A1); B1 = VFMA(ui, C8, B1);
          A2 = VFMA(ur, C7, A2); B2 = VFMA(ui, C7, B2);
          A3 = VFMA(ur, C1, A3); B3 = VFMA(ui, C1, B3);
          A4 = VFMA(ur, C9, A4); B4 = VFMA(ui, C9, B4);
          A5 = VFMA(ur, C6, A5); B5 = VFMA(ui, C6, B5);
          A6 = VFMA(ur, C2, A6); B6 = VFMA(ui, C2, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 128), ui = VL(USCR + 136);
          A1 = VFMA(ur, C9, A1); B1 = VFMA(ui, C9, B1);
          A2 = VFMA(ur, C5, A2); B2 = VFMA(ui, C5, B2);
          A3 = VFMA(ur, C4, A3); B3 = VFMA(ui, C4, B3);
          A4 = VFMA(ur, C10, A4); B4 = VFMA(ui, C10, B4);
          A5 = VFMA(ur, C1, A5); B5 = VFMA(ui, C1, B5);
          A6 = VFMA(ur, C8, A6); B6 = VFMA(ui, C8, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 144), ui = VL(USCR + 152);
          A1 = VFMA(ur, C10, A1); B1 = VFMA(ui, C10, B1);
          A2 = VFMA(ur, C3, A2); B2 = VFMA(ui, C3, B2);
          A3 = VFMA(ur, C7, A3); B3 = VFMA(ui, C7, B3);
          A4 = VFMA(ur, C6, A4); B4 = VFMA(ui, C6, B4);
          A5 = VFMA(ur, C4, A5); B5 = VFMA(ui, C4, B5);
          A6 = VFMA(ur, C9, A6); B6 = VFMA(ui, C9, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V ur = VL(USCR + 160), ui = VL(USCR + 168);
          A1 = VFMA(ur, C11, A1); B1 = VFMA(ui, C11, B1);
          A2 = VFMA(ur, C1, A2); B2 = VFMA(ui, C1, B2);
          A3 = VFMA(ur, C10, A3); B3 = VFMA(ui, C10, B3);
          A4 = VFMA(ur, C2, A4); B4 = VFMA(ui, C2, B4);
          A5 = VFMA(ur, C9, A5); B5 = VFMA(ui, C9, B5);
          A6 = VFMA(ur, C3, A6); B6 = VFMA(ui, C3, B6);
          s0r = VADD(s0r, ur); s0i = VADD(s0i, ui);
        }
        { V zr_ = VADD(s0r, VL(cr + 0)), zi_ = VADD(s0i, VL(ci + 0));
          MAP2(zr_, zi_);
          VS(pr + 0, zr_); VS(pi + 0, zi_); }
        { V Dk = VL(PSCR + 0), Ek = VL(PSCR + 8);
          V Xr = VADD(A1, Dk), Xi = VSUB(B1, Ek);
          V Yr = VSUB(A1, Dk), Yi = VADD(B1, Ek);
        { V zr_ = VADD(Xr, VL(cr + 184)), zi_ = VADD(Xi, VL(ci + 184));
          MAP2(zr_, zi_);
          VS(pr + 184, zr_); VS(pi + 184, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 4048)), zi_ = VADD(Yi, VL(ci + 4048));
          MAP2(zr_, zi_);
          VS(pr + 4048, zr_); VS(pi + 4048, zi_); }
        }
        { V Dk = VL(PSCR + 16), Ek = VL(PSCR + 24);
          V Xr = VADD(A2, Dk), Xi = VSUB(B2, Ek);
          V Yr = VSUB(A2, Dk), Yi = VADD(B2, Ek);
        { V zr_ = VADD(Xr, VL(cr + 368)), zi_ = VADD(Xi, VL(ci + 368));
          MAP2(zr_, zi_);
          VS(pr + 368, zr_); VS(pi + 368, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 3864)), zi_ = VADD(Yi, VL(ci + 3864));
          MAP2(zr_, zi_);
          VS(pr + 3864, zr_); VS(pi + 3864, zi_); }
        }
        { V Dk = VL(PSCR + 32), Ek = VL(PSCR + 40);
          V Xr = VADD(A3, Dk), Xi = VSUB(B3, Ek);
          V Yr = VSUB(A3, Dk), Yi = VADD(B3, Ek);
        { V zr_ = VADD(Xr, VL(cr + 552)), zi_ = VADD(Xi, VL(ci + 552));
          MAP2(zr_, zi_);
          VS(pr + 552, zr_); VS(pi + 552, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 3680)), zi_ = VADD(Yi, VL(ci + 3680));
          MAP2(zr_, zi_);
          VS(pr + 3680, zr_); VS(pi + 3680, zi_); }
        }
        { V Dk = VL(PSCR + 48), Ek = VL(PSCR + 56);
          V Xr = VADD(A4, Dk), Xi = VSUB(B4, Ek);
          V Yr = VSUB(A4, Dk), Yi = VADD(B4, Ek);
        { V zr_ = VADD(Xr, VL(cr + 736)), zi_ = VADD(Xi, VL(ci + 736));
          MAP2(zr_, zi_);
          VS(pr + 736, zr_); VS(pi + 736, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 3496)), zi_ = VADD(Yi, VL(ci + 3496));
          MAP2(zr_, zi_);
          VS(pr + 3496, zr_); VS(pi + 3496, zi_); }
        }
        { V Dk = VL(PSCR + 64), Ek = VL(PSCR + 72);
          V Xr = VADD(A5, Dk), Xi = VSUB(B5, Ek);
          V Yr = VSUB(A5, Dk), Yi = VADD(B5, Ek);
        { V zr_ = VADD(Xr, VL(cr + 920)), zi_ = VADD(Xi, VL(ci + 920));
          MAP2(zr_, zi_);
          VS(pr + 920, zr_); VS(pi + 920, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 3312)), zi_ = VADD(Yi, VL(ci + 3312));
          MAP2(zr_, zi_);
          VS(pr + 3312, zr_); VS(pi + 3312, zi_); }
        }
        { V Dk = VL(PSCR + 80), Ek = VL(PSCR + 88);
          V Xr = VADD(A6, Dk), Xi = VSUB(B6, Ek);
          V Yr = VSUB(A6, Dk), Yi = VADD(B6, Ek);
        { V zr_ = VADD(Xr, VL(cr + 1104)), zi_ = VADD(Xi, VL(ci + 1104));
          MAP2(zr_, zi_);
          VS(pr + 1104, zr_); VS(pi + 1104, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 3128)), zi_ = VADD(Yi, VL(ci + 3128));
          MAP2(zr_, zi_);
          VS(pr + 3128, zr_); VS(pi + 3128, zi_); }
        }
        }
        {
        V C1 = VSET1(0x1.ed037ea3d2dbcp-1);
        V C2 = VSET1(0x1.b57675cf309eep-1);
        V C3 = VSET1(0x1.5d779b07cfef7p-1);
        V C4 = VSET1(0x1.d71b4a0c5a6c9p-2);
        V C5 = VSET1(0x1.a0ad8bd1e2881p-3);
        V C6 = VSET1(-0x1.17855b599f3b2p-4);
        V C7 = VSET1(-0x1.56eaae597c776p-2);
        V C8 = VSET1(-0x1.2742a4a775cfap-1);
        V C9 = VSET1(-0x1.8d2a07c16d46ep-1);
        V C10 = VSET1(-0x1.d59cb83ef99bcp-1);
        V C11 = VSET1(-0x1.fb3b3035aa6ccp-1);
        V x0r = VL(USCR + 384), x0i = VL(USCR + 392);
        V A7 = x0r, B7 = x0i;
        V A8 = x0r, B8 = x0i;
        V A9 = x0r, B9 = x0i;
        V A10 = x0r, B10 = x0i;
        V A11 = x0r, B11 = x0i;
        { V ur = VL(USCR + 0), ui = VL(USCR + 8);
          A7 = VFMA(ur, C7, A7); B7 = VFMA(ui, C7, B7);
          A8 = VFMA(ur, C8, A8); B8 = VFMA(ui, C8, B8);
          A9 = VFMA(ur, C9, A9); B9 = VFMA(ui, C9, B9);
          A10 = VFMA(ur, C10, A10); B10 = VFMA(ui, C10, B10);
          A11 = VFMA(ur, C11, A11); B11 = VFMA(ui, C11, B11);
        }
        { V ur = VL(USCR + 16), ui = VL(USCR + 24);
          A7 = VFMA(ur, C9, A7); B7 = VFMA(ui, C9, B7);
          A8 = VFMA(ur, C7, A8); B8 = VFMA(ui, C7, B8);
          A9 = VFMA(ur, C5, A9); B9 = VFMA(ui, C5, B9);
          A10 = VFMA(ur, C3, A10); B10 = VFMA(ui, C3, B10);
          A11 = VFMA(ur, C1, A11); B11 = VFMA(ui, C1, B11);
        }
        { V ur = VL(USCR + 32), ui = VL(USCR + 40);
          A7 = VFMA(ur, C2, A7); B7 = VFMA(ui, C2, B7);
          A8 = VFMA(ur, C1, A8); B8 = VFMA(ui, C1, B8);
          A9 = VFMA(ur, C4, A9); B9 = VFMA(ui, C4, B9);
          A10 = VFMA(ur, C7, A10); B10 = VFMA(ui, C7, B10);
          A11 = VFMA(ur, C10, A11); B11 = VFMA(ui, C10, B11);
        }
        { V ur = VL(USCR + 48), ui = VL(USCR + 56);
          A7 = VFMA(ur, C5, A7); B7 = VFMA(ui, C5, B7);
          A8 = VFMA(ur, C9, A8); B8 = VFMA(ui, C9, B8);
          A9 = VFMA(ur, C10, A9); B9 = VFMA(ui, C10, B9);
          A10 = VFMA(ur, C6, A10); B10 = VFMA(ui, C6, B10);
          A11 = VFMA(ur, C2, A11); B11 = VFMA(ui, C2, B11);
        }
        { V ur = VL(USCR + 64), ui = VL(USCR + 72);
          A7 = VFMA(ur, C11, A7); B7 = VFMA(ui, C11, B7);
          A8 = VFMA(ur, C6, A8); B8 = VFMA(ui, C6, B8);
          A9 = VFMA(ur, C1, A9); B9 = VFMA(ui, C1, B9);
          A10 = VFMA(ur, C4, A10); B10 = VFMA(ui, C4, B10);
          A11 = VFMA(ur, C9, A11); B11 = VFMA(ui, C9, B11);
        }
        { V ur = VL(USCR + 80), ui = VL(USCR + 88);
          A7 = VFMA(ur, C4, A7); B7 = VFMA(ui, C4, B7);
          A8 = VFMA(ur, C2, A8); B8 = VFMA(ui, C2, B8);
          A9 = VFMA(ur, C8, A9); B9 = VFMA(ui, C8, B9);
          A10 = VFMA(ur, C9, A10); B10 = VFMA(ui, C9, B10);
          A11 = VFMA(ur, C3, A11); B11 = VFMA(ui, C3, B11);
        }
        { V ur = VL(USCR + 96), ui = VL(USCR + 104);
          A7 = VFMA(ur, C3, A7); B7 = VFMA(ui, C3, B7);
          A8 = VFMA(ur, C10, A8); B8 = VFMA(ui, C10, B8);
          A9 = VFMA(ur, C6, A9); B9 = VFMA(ui, C6, B9);
          A10 = VFMA(ur, C1, A10); B10 = VFMA(ui, C1, B10);
          A11 = VFMA(ur, C8, A11); B11 = VFMA(ui, C8, B11);
        }
        { V ur = VL(USCR + 112), ui = VL(USCR + 120);
          A7 = VFMA(ur, C10, A7); B7 = VFMA(ui, C10, B7);
          A8 = VFMA(ur, C5, A8); B8 = VFMA(ui, C5, B8);
          A9 = VFMA(ur, C3, A9); B9 = VFMA(ui, C3, B9);
          A10 = VFMA(ur, C11, A10); B10 = VFMA(ui, C11, B10);
          A11 = VFMA(ur, C4, A11); B11 = VFMA(ui, C4, B11);
        }
        { V ur = VL(USCR + 128), ui = VL(USCR + 136);
          A7 = VFMA(ur, C6, A7); B7 = VFMA(ui, C6, B7);
          A8 = VFMA(ur, C3, A8); B8 = VFMA(ui, C3, B8);
          A9 = VFMA(ur, C11, A9); B9 = VFMA(ui, C11, B9);
          A10 = VFMA(ur, C2, A10); B10 = VFMA(ui, C2, B10);
          A11 = VFMA(ur, C7, A11); B11 = VFMA(ui, C7, B11);
        }
        { V ur = VL(USCR + 144), ui = VL(USCR + 152);
          A7 = VFMA(ur, C1, A7); B7 = VFMA(ui, C1, B7);
          A8 = VFMA(ur, C11, A8); B8 = VFMA(ui, C11, B8);
          A9 = VFMA(ur, C2, A9); B9 = VFMA(ui, C2, B9);
          A10 = VFMA(ur, C8, A10); B10 = VFMA(ui, C8, B10);
          A11 = VFMA(ur, C5, A11); B11 = VFMA(ui, C5, B11);
        }
        { V ur = VL(USCR + 160), ui = VL(USCR + 168);
          A7 = VFMA(ur, C8, A7); B7 = VFMA(ui, C8, B7);
          A8 = VFMA(ur, C4, A8); B8 = VFMA(ui, C4, B8);
          A9 = VFMA(ur, C7, A9); B9 = VFMA(ui, C7, B9);
          A10 = VFMA(ur, C5, A10); B10 = VFMA(ui, C5, B10);
          A11 = VFMA(ur, C6, A11); B11 = VFMA(ui, C6, B11);
        }
        { V Dk = VL(PSCR + 96), Ek = VL(PSCR + 104);
          V Xr = VADD(A7, Dk), Xi = VSUB(B7, Ek);
          V Yr = VSUB(A7, Dk), Yi = VADD(B7, Ek);
        { V zr_ = VADD(Xr, VL(cr + 1288)), zi_ = VADD(Xi, VL(ci + 1288));
          MAP2(zr_, zi_);
          VS(pr + 1288, zr_); VS(pi + 1288, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 2944)), zi_ = VADD(Yi, VL(ci + 2944));
          MAP2(zr_, zi_);
          VS(pr + 2944, zr_); VS(pi + 2944, zi_); }
        }
        { V Dk = VL(PSCR + 112), Ek = VL(PSCR + 120);
          V Xr = VADD(A8, Dk), Xi = VSUB(B8, Ek);
          V Yr = VSUB(A8, Dk), Yi = VADD(B8, Ek);
        { V zr_ = VADD(Xr, VL(cr + 1472)), zi_ = VADD(Xi, VL(ci + 1472));
          MAP2(zr_, zi_);
          VS(pr + 1472, zr_); VS(pi + 1472, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 2760)), zi_ = VADD(Yi, VL(ci + 2760));
          MAP2(zr_, zi_);
          VS(pr + 2760, zr_); VS(pi + 2760, zi_); }
        }
        { V Dk = VL(PSCR + 128), Ek = VL(PSCR + 136);
          V Xr = VADD(A9, Dk), Xi = VSUB(B9, Ek);
          V Yr = VSUB(A9, Dk), Yi = VADD(B9, Ek);
        { V zr_ = VADD(Xr, VL(cr + 1656)), zi_ = VADD(Xi, VL(ci + 1656));
          MAP2(zr_, zi_);
          VS(pr + 1656, zr_); VS(pi + 1656, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 2576)), zi_ = VADD(Yi, VL(ci + 2576));
          MAP2(zr_, zi_);
          VS(pr + 2576, zr_); VS(pi + 2576, zi_); }
        }
        { V Dk = VL(PSCR + 144), Ek = VL(PSCR + 152);
          V Xr = VADD(A10, Dk), Xi = VSUB(B10, Ek);
          V Yr = VSUB(A10, Dk), Yi = VADD(B10, Ek);
        { V zr_ = VADD(Xr, VL(cr + 1840)), zi_ = VADD(Xi, VL(ci + 1840));
          MAP2(zr_, zi_);
          VS(pr + 1840, zr_); VS(pi + 1840, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 2392)), zi_ = VADD(Yi, VL(ci + 2392));
          MAP2(zr_, zi_);
          VS(pr + 2392, zr_); VS(pi + 2392, zi_); }
        }
        { V Dk = VL(PSCR + 160), Ek = VL(PSCR + 168);
          V Xr = VADD(A11, Dk), Xi = VSUB(B11, Ek);
          V Yr = VSUB(A11, Dk), Yi = VADD(B11, Ek);
        { V zr_ = VADD(Xr, VL(cr + 2024)), zi_ = VADD(Xi, VL(ci + 2024));
          MAP2(zr_, zi_);
          VS(pr + 2024, zr_); VS(pi + 2024, zi_); }
        { V zr_ = VADD(Yr, VL(cr + 2208)), zi_ = VADD(Yi, VL(ci + 2208));
          MAP2(zr_, zi_);
          VS(pr + 2208, zr_); VS(pi + 2208, zi_); }
        }
        }
    }
}

static void sw23_SyX(int y0, int pre){
    long b0 = (long)y0*23*8;
    p23_xxm(S23RE + b0, S23IM + b0, C23RE + b0, C23IM + b0, 23, 8);
    if (pre) {
        p23_zz(S23RE + (long)y0*23*8, S23IM + (long)y0*23*8, 23, (long)23*23*8);
        p23_xx(S23RE + b0, S23IM + b0, 23, 8);
    }
}
static void sw23_PxY(int x0, int pre){
    long b0 = (long)x0*23*23*8;
    p23_yym(S23RE + b0, S23IM + b0, C23RE + b0, C23IM + b0, 23, 8);
    if (pre) {
        p23_zz(S23RE + b0, S23IM + b0, 23, (long)23*8);
        p23_yy(S23RE + b0, S23IM + b0, 23, 8);
    }
}
void run_23(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if (m < 1) m = 1;
    long vs = 2*(long)12167;
    for (long g0 = 0; g0 < B; g0 += 8) {
        int nv = (B - g0) < 8 ? (int)(B - g0) : 8;
        const double* xg = x0 + g0*vs;
        const double* cg = c + g0*vs;
        long nfull = (12167/8)*8;
        if (nv == 8) {
            for (long s = 0; s < nfull; s += 8) { soa_in8(xg, vs, s, S23RE, S23IM); soa_in8(cg, vs, s, C23RE, C23IM); }
        } else {
            for (long s = 0; s < nfull; s += 8) { soa_in8_nv(xg, vs, s, S23RE, S23IM, nv); soa_in8_nv(cg, vs, s, C23RE, C23IM, nv); }
        }
        for (long s = nfull; s < 12167; s++) {   // tail sites scalar
            for (int v = 0; v < 8; v++) {
                S23RE[s*8+v] = v < nv ? xg[v*vs + 2*s] : 0.0;
                S23IM[s*8+v] = v < nv ? xg[v*vs + 2*s + 1] : 0.0;
                C23RE[s*8+v] = v < nv ? cg[v*vs + 2*s] : 0.0;
                C23IM[s*8+v] = v < nv ? cg[v*vs + 2*s + 1] : 0.0;
            }
        }
        // prologue: Z,Y per plane x
        for (int x = 0; x < 23; x++) {
            long b0 = (long)x*23*23*8;
            p23_zz(S23RE + b0, S23IM + b0, 23, (long)23*8);
            p23_yy(S23RE + b0, S23IM + b0, 23, 8);
        }
        for (long t = 1; t <= m; t++) {
            int snap = (t == 1) || (t == m);
            int pre = (t < m);
            int dopre = pre && !snap;
            if (t & 1) { for (int y0 = 0; y0 < 23; y0++) sw23_SyX(y0, dopre); }
            else       { for (int xp = 0; xp < 23; xp++) sw23_PxY(xp, dopre); }
            if (t == 1) {
                for (long s = 0; s < nfull; s += 8) soa_out8(S23RE, S23IM, s, out1 + g0*vs, vs, nv);
                for (long s = nfull; s < 12167; s++) for (int v = 0; v < nv; v++) {
                    out1[g0*vs + v*vs + 2*s] = S23RE[s*8+v]; out1[g0*vs + v*vs + 2*s+1] = S23IM[s*8+v]; }
            }
            if (t == m) {
                for (long s = 0; s < nfull; s += 8) soa_out8(S23RE, S23IM, s, outm + g0*vs, vs, nv);
                for (long s = nfull; s < 12167; s++) for (int v = 0; v < nv; v++) {
                    outm[g0*vs + v*vs + 2*s] = S23RE[s*8+v]; outm[g0*vs + v*vs + 2*s+1] = S23IM[s*8+v]; }
            }
            if (pre && !dopre) {
                p23_zz(S23RE, S23IM, 23*23, (long)23*8);
                if (t & 1) {
                    for (int y0 = 0; y0 < 23; y0++)
                        p23_xx(S23RE + (long)y0*23*8, S23IM + (long)y0*23*8, 23, 8);
                } else {
                    for (int x0 = 0; x0 < 23; x0++)
                        p23_yy(S23RE + (long)x0*23*23*8, S23IM + (long)x0*23*23*8, 23, 8);
                }
            }
        }
    }
}

// ============ L=6 SoA-8 ============
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
        p6_zz(S6RE + (long)y0*6*8, S6IM + (long)y0*6*8, 6, (long)6*6*8);
        p6_xx(S6RE + b0, S6IM + b0, 6, 8);
    }
}
static void sw6_PxY(int x0, int pre){
    long b0 = (long)x0*6*6*8;
    p6_yym(S6RE + b0, S6IM + b0, C6RE + b0, C6IM + b0, 6, 8);
    if (pre) {
        p6_zz(S6RE + b0, S6IM + b0, 6, (long)6*8);
        p6_yy(S6RE + b0, S6IM + b0, 6, 8);
    }
}
void run_6(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if (m < 1) m = 1;
    long vs = 2*(long)216;
    for (long g0 = 0; g0 < B; g0 += 8) {
        int nv = (B - g0) < 8 ? (int)(B - g0) : 8;
        const double* xg = x0 + g0*vs;
        const double* cg = c + g0*vs;
        long nfull = (216/8)*8;
        if (nv == 8) {
            for (long s = 0; s < nfull; s += 8) { soa_in8(xg, vs, s, S6RE, S6IM); soa_in8(cg, vs, s, C6RE, C6IM); }
        } else {
            for (long s = 0; s < nfull; s += 8) { soa_in8_nv(xg, vs, s, S6RE, S6IM, nv); soa_in8_nv(cg, vs, s, C6RE, C6IM, nv); }
        }
        for (long s = nfull; s < 216; s++) {   // tail sites scalar
            for (int v = 0; v < 8; v++) {
                S6RE[s*8+v] = v < nv ? xg[v*vs + 2*s] : 0.0;
                S6IM[s*8+v] = v < nv ? xg[v*vs + 2*s + 1] : 0.0;
                C6RE[s*8+v] = v < nv ? cg[v*vs + 2*s] : 0.0;
                C6IM[s*8+v] = v < nv ? cg[v*vs + 2*s + 1] : 0.0;
            }
        }
        // prologue: Z,Y per plane x
        for (int x = 0; x < 6; x++) {
            long b0 = (long)x*6*6*8;
            p6_zz(S6RE + b0, S6IM + b0, 6, (long)6*8);
            p6_yy(S6RE + b0, S6IM + b0, 6, 8);
        }
        for (long t = 1; t <= m; t++) {
            int snap = (t == 1) || (t == m);
            int pre = (t < m);
            int dopre = pre && !snap;
            if (t & 1) { for (int y0 = 0; y0 < 6; y0++) sw6_SyX(y0, dopre); }
            else       { for (int xp = 0; xp < 6; xp++) sw6_PxY(xp, dopre); }
            if (t == 1) {
                for (long s = 0; s < nfull; s += 8) soa_out8(S6RE, S6IM, s, out1 + g0*vs, vs, nv);
                for (long s = nfull; s < 216; s++) for (int v = 0; v < nv; v++) {
                    out1[g0*vs + v*vs + 2*s] = S6RE[s*8+v]; out1[g0*vs + v*vs + 2*s+1] = S6IM[s*8+v]; }
            }
            if (t == m) {
                for (long s = 0; s < nfull; s += 8) soa_out8(S6RE, S6IM, s, outm + g0*vs, vs, nv);
                for (long s = nfull; s < 216; s++) for (int v = 0; v < nv; v++) {
                    outm[g0*vs + v*vs + 2*s] = S6RE[s*8+v]; outm[g0*vs + v*vs + 2*s+1] = S6IM[s*8+v]; }
            }
            if (pre && !dopre) {
                p6_zz(S6RE, S6IM, 6*6, (long)6*8);
                if (t & 1) {
                    for (int y0 = 0; y0 < 6; y0++)
                        p6_xx(S6RE + (long)y0*6*8, S6IM + (long)y0*6*8, 6, 8);
                } else {
                    for (int x0 = 0; x0 < 6; x0++)
                        p6_yy(S6RE + (long)x0*6*6*8, S6IM + (long)x0*6*6*8, 6, 8);
                }
            }
        }
    }
}

// ============ L=8 SoA-8 ============
static double S8RE[4096] ALIGN64;
static double S8IM[4096] ALIGN64;
static double C8RE[4096] ALIGN64;
static double C8IM[4096] ALIGN64;
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
        V x1r = VL(pr + 512), x1i = VL(pi + 512);
        V x2r = VL(pr + 1024), x2i = VL(pi + 1024);
        V x3r = VL(pr + 1536), x3i = VL(pi + 1536);
        V x4r = VL(pr + 2048), x4i = VL(pi + 2048);
        V x5r = VL(pr + 2560), x5i = VL(pi + 2560);
        V x6r = VL(pr + 3072), x6i = VL(pi + 3072);
        V x7r = VL(pr + 3584), x7i = VL(pi + 3584);
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
        VS(pr + 1024, y1r); VS(pi + 1024, y1i);
        VS(pr + 2048, y2r); VS(pi + 2048, y2i);
        VS(pr + 3072, y3r); VS(pi + 3072, y3i);
        }
        { V t0r=VADD(d0r,d2r), t0i=VADD(d0i,d2i);
          V t1r=VSUB(d0r,d2r), t1i=VSUB(d0i,d2i);
          V t2r=VADD(d1r,d3r), t2i=VADD(d1i,d3i);
          V t3r=VSUB(d1r,d3r), t3i=VSUB(d1i,d3i);
          V y0r=VADD(t0r,t2r), y0i=VADD(t0i,t2i);
          V y2r=VSUB(t0r,t2r), y2i=VSUB(t0i,t2i);
          V y1r=VADD(t1r,t3i), y1i=VSUB(t1i,t3r);
          V y3r=VSUB(t1r,t3i), y3i=VADD(t1i,t3r);
        VS(pr + 512, y0r); VS(pi + 512, y0i);
        VS(pr + 1536, y1r); VS(pi + 1536, y1i);
        VS(pr + 2560, y2r); VS(pi + 2560, y2i);
        VS(pr + 3584, y3r); VS(pi + 3584, y3i);
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
        V x1r = VL(pr + 512), x1i = VL(pi + 512);
        V x2r = VL(pr + 1024), x2i = VL(pi + 1024);
        V x3r = VL(pr + 1536), x3i = VL(pi + 1536);
        V x4r = VL(pr + 2048), x4i = VL(pi + 2048);
        V x5r = VL(pr + 2560), x5i = VL(pi + 2560);
        V x6r = VL(pr + 3072), x6i = VL(pi + 3072);
        V x7r = VL(pr + 3584), x7i = VL(pi + 3584);
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
        { V zr_ = VADD(y1r, VL(cr + 1024)), zi_ = VADD(y1i, VL(ci + 1024));
          MAP2(zr_, zi_);
          VS(pr + 1024, zr_); VS(pi + 1024, zi_); }
        { V zr_ = VADD(y2r, VL(cr + 2048)), zi_ = VADD(y2i, VL(ci + 2048));
          MAP2(zr_, zi_);
          VS(pr + 2048, zr_); VS(pi + 2048, zi_); }
        { V zr_ = VADD(y3r, VL(cr + 3072)), zi_ = VADD(y3i, VL(ci + 3072));
          MAP2(zr_, zi_);
          VS(pr + 3072, zr_); VS(pi + 3072, zi_); }
        }
        { V t0r=VADD(d0r,d2r), t0i=VADD(d0i,d2i);
          V t1r=VSUB(d0r,d2r), t1i=VSUB(d0i,d2i);
          V t2r=VADD(d1r,d3r), t2i=VADD(d1i,d3i);
          V t3r=VSUB(d1r,d3r), t3i=VSUB(d1i,d3i);
          V y0r=VADD(t0r,t2r), y0i=VADD(t0i,t2i);
          V y2r=VSUB(t0r,t2r), y2i=VSUB(t0i,t2i);
          V y1r=VADD(t1r,t3i), y1i=VSUB(t1i,t3r);
          V y3r=VSUB(t1r,t3i), y3i=VADD(t1i,t3r);
        { V zr_ = VADD(y0r, VL(cr + 512)), zi_ = VADD(y0i, VL(ci + 512));
          MAP2(zr_, zi_);
          VS(pr + 512, zr_); VS(pi + 512, zi_); }
        { V zr_ = VADD(y1r, VL(cr + 1536)), zi_ = VADD(y1i, VL(ci + 1536));
          MAP2(zr_, zi_);
          VS(pr + 1536, zr_); VS(pi + 1536, zi_); }
        { V zr_ = VADD(y2r, VL(cr + 2560)), zi_ = VADD(y2i, VL(ci + 2560));
          MAP2(zr_, zi_);
          VS(pr + 2560, zr_); VS(pi + 2560, zi_); }
        { V zr_ = VADD(y3r, VL(cr + 3584)), zi_ = VADD(y3i, VL(ci + 3584));
          MAP2(zr_, zi_);
          VS(pr + 3584, zr_); VS(pi + 3584, zi_); }
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
        p8_zz(S8RE + (long)y0*8*8, S8IM + (long)y0*8*8, 8, (long)8*8*8);
        p8_xx(S8RE + b0, S8IM + b0, 8, 8);
    }
}
static void sw8_PxY(int x0, int pre){
    long b0 = (long)x0*8*8*8;
    p8_yym(S8RE + b0, S8IM + b0, C8RE + b0, C8IM + b0, 8, 8);
    if (pre) {
        p8_zz(S8RE + b0, S8IM + b0, 8, (long)8*8);
        p8_yy(S8RE + b0, S8IM + b0, 8, 8);
    }
}
void run_8(const double* x0, const double* c, double* out1, double* outm, long B, long m){
    if (m < 1) m = 1;
    long vs = 2*(long)512;
    for (long g0 = 0; g0 < B; g0 += 8) {
        int nv = (B - g0) < 8 ? (int)(B - g0) : 8;
        const double* xg = x0 + g0*vs;
        const double* cg = c + g0*vs;
        long nfull = (512/8)*8;
        if (nv == 8) {
            for (long s = 0; s < nfull; s += 8) { soa_in8(xg, vs, s, S8RE, S8IM); soa_in8(cg, vs, s, C8RE, C8IM); }
        } else {
            for (long s = 0; s < nfull; s += 8) { soa_in8_nv(xg, vs, s, S8RE, S8IM, nv); soa_in8_nv(cg, vs, s, C8RE, C8IM, nv); }
        }
        for (long s = nfull; s < 512; s++) {   // tail sites scalar
            for (int v = 0; v < 8; v++) {
                S8RE[s*8+v] = v < nv ? xg[v*vs + 2*s] : 0.0;
                S8IM[s*8+v] = v < nv ? xg[v*vs + 2*s + 1] : 0.0;
                C8RE[s*8+v] = v < nv ? cg[v*vs + 2*s] : 0.0;
                C8IM[s*8+v] = v < nv ? cg[v*vs + 2*s + 1] : 0.0;
            }
        }
        // prologue: Z,Y per plane x
        for (int x = 0; x < 8; x++) {
            long b0 = (long)x*8*8*8;
            p8_zz(S8RE + b0, S8IM + b0, 8, (long)8*8);
            p8_yy(S8RE + b0, S8IM + b0, 8, 8);
        }
        for (long t = 1; t <= m; t++) {
            int snap = (t == 1) || (t == m);
            int pre = (t < m);
            int dopre = pre && !snap;
            if (t & 1) { for (int y0 = 0; y0 < 8; y0++) sw8_SyX(y0, dopre); }
            else       { for (int xp = 0; xp < 8; xp++) sw8_PxY(xp, dopre); }
            if (t == 1) {
                for (long s = 0; s < nfull; s += 8) soa_out8(S8RE, S8IM, s, out1 + g0*vs, vs, nv);
                for (long s = nfull; s < 512; s++) for (int v = 0; v < nv; v++) {
                    out1[g0*vs + v*vs + 2*s] = S8RE[s*8+v]; out1[g0*vs + v*vs + 2*s+1] = S8IM[s*8+v]; }
            }
            if (t == m) {
                for (long s = 0; s < nfull; s += 8) soa_out8(S8RE, S8IM, s, outm + g0*vs, vs, nv);
                for (long s = nfull; s < 512; s++) for (int v = 0; v < nv; v++) {
                    outm[g0*vs + v*vs + 2*s] = S8RE[s*8+v]; outm[g0*vs + v*vs + 2*s+1] = S8IM[s*8+v]; }
            }
            if (pre && !dopre) {
                p8_zz(S8RE, S8IM, 8*8, (long)8*8);
                if (t & 1) {
                    for (int y0 = 0; y0 < 8; y0++)
                        p8_xx(S8RE + (long)y0*8*8, S8IM + (long)y0*8*8, 8, 8);
                } else {
                    for (int x0 = 0; x0 < 8; x0++)
                        p8_yy(S8RE + (long)x0*8*8*8, S8IM + (long)x0*8*8*8, 8, 8);
                }
            }
        }
    }
}

