// ============================================================================
// Iterated batched 3D complex-to-complex FFT engine for L in {6,8,13,17,23,36,45,64}
//
// All DFT arithmetic in this file is original, hand-written code (AVX-512
// intrinsics): small-size codelets (PFA/Cooley-Tukey/direct symmetric prime
// transforms) specialized per size. No FFT library of any kind is called or
// linked; libm (sinl/cosl) is used only at plan/init time to build twiddle
// tables. Single-threaded throughout; no OpenMP/pthreads.
//
// Per step (per volume): z = FFT3(x) + c; x <- z/(1+|z|).
//   The 3D transform runs as three 1D lane-vectorized passes; the elementwise
//   map is fused into the last pass. Two engines:
//   - per-volume engine: split re/im "column panel" layout, padded to 8;
//     pass1 = x-axis (cross-slab), fused pass2+3 = in-slab axes, the
//     contiguous axis handled via 8x8 in-register transposes; output plane
//     written transposed (parity flips each step; c kept in both layouts).
//   - octet engine (L<=17, batches of 8 volumes): lanes = 8 volumes,
//     position-major layout, no transposes/padding, all passes in-place.
//   |z| and 1/(1+|z|) via rsqrt14/rcp14 + Newton iterations to full double
//   precision (optionally hardware divide), guarded against q=0.
// ============================================================================
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <sys/mman.h>

#define AI static inline __attribute__((always_inline))
typedef __m512d V;
#define VLD(p)      _mm512_load_pd(p)
#define VLDU(p)     _mm512_loadu_pd(p)
#define VST(p,v)    _mm512_store_pd((p),(v))
#define VSTU(p,v)   _mm512_storeu_pd((p),(v))
#define VADD        _mm512_add_pd
#define VSUB        _mm512_sub_pd
#define VMUL        _mm512_mul_pd
#define VFMA        _mm512_fmadd_pd
#define VFNMA       _mm512_fnmadd_pd
#define VFMS        _mm512_fmsub_pd
#define VSET1       _mm512_set1_pd

static void* big_alloc2(size_t bytes, size_t stagger){
  bytes += stagger;
  size_t sz = (bytes + (2UL<<20) - 1) & ~((2UL<<20)-1);
  void* p = mmap(0, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) { void* q = aligned_alloc(64, bytes); if(q) memset(q,0,bytes); return (char*)q + stagger; }
  madvise(p, sz, MADV_HUGEPAGE);
  memset(p, 0, sz);
  return (char*)p + stagger;
}
static void* big_alloc(size_t bytes){ return big_alloc2(bytes, 0); }

// ----------------------------------------------------------------------------
// elementwise map: given (re,im) after FFT and c (cre,cim):  w = z + c ; out = w/(1+|w|)
static int g_pf = 2, g_pf23 = 0, g_inplace = 1, g_raw = 0, g_ip2 = 1, g_f23c = 0;
static int g_mapmix = 2;      // of every 4 rows, this many use the hw divider
// s = 1/(1+sqrt(q)) = t/(1+t) with t = rsqrt(q)
AI void map8m(V* pre, V* pim, V cre, V cim, int use_div){
  V r = VADD(*pre, cre), i = VADD(*pim, cim);
  V q = VFMA(r, r, VFMA(i, i, VSET1(1e-300)));
  V t = _mm512_rsqrt14_pd(q);
  V hq = VMUL(q, VSET1(0.5));
  V u = VMUL(hq, t); V w = VFNMA(u, t, VSET1(1.5)); t = VMUL(t, w);
  u = VMUL(hq, t); w = VFNMA(u, t, VSET1(1.5)); t = VMUL(t, w);
  V m = VMUL(q, t);               // ~= sqrt(q)
  V d = VADD(m, VSET1(1.0));
  V s;
  if (use_div){
    s = _mm512_div_pd(VSET1(1.0), d);
  } else {
    V y = _mm512_rcp14_pd(d);
    V e = VFNMA(d, y, VSET1(1.0)); y = VFMA(y, e, y);
    e = VFNMA(d, y, VSET1(1.0)); y = VFMA(y, e, y);
    s = y;
  }
  *pre = VMUL(r, s); *pim = VMUL(i, s);
}
AI void map8(V* pre, V* pim, V cre, V cim){ map8m(pre, pim, cre, cim, 0); }
// map without the +c (state already holds z = FFT3(x)+c)
AI void map8nc(V* pre, V* pim, int use_div){
  V r = *pre, i = *pim;
  V q = VFMA(r, r, VFMA(i, i, VSET1(1e-300)));
  V t = _mm512_rsqrt14_pd(q);
  V hq = VMUL(q, VSET1(0.5));
  V u = VMUL(hq, t); V w = VFNMA(u, t, VSET1(1.5)); t = VMUL(t, w);
  u = VMUL(hq, t); w = VFNMA(u, t, VSET1(1.5)); t = VMUL(t, w);
  V m = VMUL(q, t);
  V d = VADD(m, VSET1(1.0));
  V s;
  if (use_div){
    s = _mm512_div_pd(VSET1(1.0), d);
  } else {
    V y = _mm512_rcp14_pd(d);
    V e = VFNMA(d, y, VSET1(1.0)); y = VFMA(y, e, y);
    e = VFNMA(d, y, VSET1(1.0)); y = VFMA(y, e, y);
    s = y;
  }
  *pre = VMUL(r, s); *pim = VMUL(i, s);
}

// ----------------------------------------------------------------------------
// 8x8 double transpose (zmm)
#define TRANSP8(r0,r1,r2,r3,r4,r5,r6,r7) do{                          \
  V _t0=_mm512_unpacklo_pd(r0,r1), _t1=_mm512_unpackhi_pd(r0,r1);     \
  V _t2=_mm512_unpacklo_pd(r2,r3), _t3=_mm512_unpackhi_pd(r2,r3);     \
  V _t4=_mm512_unpacklo_pd(r4,r5), _t5=_mm512_unpackhi_pd(r4,r5);     \
  V _t6=_mm512_unpacklo_pd(r6,r7), _t7=_mm512_unpackhi_pd(r6,r7);     \
  V _u0=_mm512_shuffle_f64x2(_t0,_t2,0x88), _u1=_mm512_shuffle_f64x2(_t1,_t3,0x88); \
  V _u2=_mm512_shuffle_f64x2(_t0,_t2,0xdd), _u3=_mm512_shuffle_f64x2(_t1,_t3,0xdd); \
  V _u4=_mm512_shuffle_f64x2(_t4,_t6,0x88), _u5=_mm512_shuffle_f64x2(_t5,_t7,0x88); \
  V _u6=_mm512_shuffle_f64x2(_t4,_t6,0xdd), _u7=_mm512_shuffle_f64x2(_t5,_t7,0xdd); \
  r0=_mm512_shuffle_f64x2(_u0,_u4,0x88); r4=_mm512_shuffle_f64x2(_u0,_u4,0xdd);     \
  r1=_mm512_shuffle_f64x2(_u1,_u5,0x88); r5=_mm512_shuffle_f64x2(_u1,_u5,0xdd);     \
  r2=_mm512_shuffle_f64x2(_u2,_u6,0x88); r6=_mm512_shuffle_f64x2(_u2,_u6,0xdd);     \
  r3=_mm512_shuffle_f64x2(_u3,_u7,0x88); r7=_mm512_shuffle_f64x2(_u3,_u7,0xdd);     \
}while(0)

static const long long IDXE_[8] = {0,2,4,6,8,10,12,14};
static const long long IDXO_[8] = {1,3,5,7,9,11,13,15};
static const long long ILA_[8]  = {0,8,1,9,2,10,3,11};
static const long long ILB_[8]  = {4,12,5,13,6,14,7,15};
#define IDXE _mm512_load_si512((const void*)IDXE_)
#define IDXO _mm512_load_si512((const void*)IDXO_)
#define ILA  _mm512_load_si512((const void*)ILA_)
#define ILB  _mm512_load_si512((const void*)ILB_)

// ----------------------------------------------------------------------------
// generic validation codelet: naive matvec with DFT matrix (any L)
static double* g_W[65];   // per L: [k*L+j]*2 = {cos, -sin} (forward DFT W^{kj})

static void generic_col(int L, const double* sre, const double* sim, long ss,
                        double* dre, double* dim, long ds){
  const double* W = g_W[L];
  for (int k = 0; k < L; k++){
    V ar = VSET1(0.0), ai = VSET1(0.0);
    for (int j = 0; j < L; j++){
      V xr = VLDU(sre + (long)j*ss), xi = VLDU(sim + (long)j*ss);
      V c = VSET1(W[(k*L+j)*2]), s = VSET1(W[(k*L+j)*2+1]); // W = c + i s (s = -sin)
      ar = VFMA(xr, c, ar); ar = VFNMA(xi, s, ar);
      ai = VFMA(xi, c, ai); ai = VFMA(xr, s, ai);
    }
    VSTU(dre + (long)k*ds, ar); VSTU(dim + (long)k*ds, ai);
  }
}

// ----------------------------------------------------------------------------
// per-size engine state
typedef struct {
  int L, P;
  long PLANE;   // P*P doubles (one component)
  long SLAB;    // 2*PLANE
  long VOL;     // L*SLAB
  double *A, *B, *c0, *c1;
} eng_t;
static eng_t g_eng[65];

static double SCR1RE[72*8] __attribute__((aligned(64)));
static double SCR1IM[72*8] __attribute__((aligned(64)));
static double SCR2RE[72*8] __attribute__((aligned(64)));
static double SCR2IM[72*8] __attribute__((aligned(64)));
static double MIDSLAB_RAW[2*64*64+8+88] __attribute__((aligned(4096)));
#define MIDSLAB (MIDSLAB_RAW + 40)
static double COLRE[72*8] __attribute__((aligned(64)));
static double COLIM[72*8] __attribute__((aligned(64)));


// ========================= generic pass drivers ==============================
// plane storage: column panels: offset(r,c) = (c/8)*P*8 + r*8 + (c%8)
static void gpass1(eng_t* e, const double* src, double* dst){
  int L = e->L, P = e->P; long SLAB = e->SLAB, PLANE = e->PLANE;
  for (int p = 0; p < P/8; p++)
    for (int k = 0; k < L; k++){
      long o = (long)p*P*16 + k*8;
      generic_col(L, src+o, src+PLANE+o, SLAB, dst+o, dst+PLANE+o, SLAB);
    }
}
static void gpass2(eng_t* e, const double* src, double* dst){
  int L = e->L, P = e->P; long SLAB = e->SLAB, PLANE = e->PLANE;
  for (int x = 0; x < L; x++){
    const double* sb = src + (long)x*SLAB; double* db = dst + (long)x*SLAB;
    for (int p = 0; p < P/8; p++){
      long o = (long)p*P*16;
      generic_col(L, sb+o, sb+PLANE+o, 8, db+o, db+PLANE+o, 8);
    }
  }
}
static void gpass3(eng_t* e, const double* src, double* dst, const double* cb_){
  int L = e->L, P = e->P; long SLAB = e->SLAB, PLANE = e->PLANE;
  int nstr = P/8;
  for (int x = 0; x < L; x++){
    const double* sb = src + (long)x*SLAB;
    double* db = dst + (long)x*SLAB;
    const double* cc = cb_ + (long)x*SLAB;
    for (int t = 0; t < nstr; t++){
      for (int p = 0; p < P/8; p++){
        const double* q = sb + (long)p*P*16 + (long)t*64;
        V r0=VLD(q),r1=VLD(q+8),r2=VLD(q+16),r3=VLD(q+24),r4=VLD(q+32),r5=VLD(q+40),r6=VLD(q+48),r7=VLD(q+56);
        TRANSP8(r0,r1,r2,r3,r4,r5,r6,r7);
        VST(SCR1RE+(8*p+0)*8,r0); VST(SCR1RE+(8*p+1)*8,r1); VST(SCR1RE+(8*p+2)*8,r2); VST(SCR1RE+(8*p+3)*8,r3);
        VST(SCR1RE+(8*p+4)*8,r4); VST(SCR1RE+(8*p+5)*8,r5); VST(SCR1RE+(8*p+6)*8,r6); VST(SCR1RE+(8*p+7)*8,r7);
        q += PLANE;
        r0=VLD(q);r1=VLD(q+8);r2=VLD(q+16);r3=VLD(q+24);r4=VLD(q+32);r5=VLD(q+40);r6=VLD(q+48);r7=VLD(q+56);
        TRANSP8(r0,r1,r2,r3,r4,r5,r6,r7);
        VST(SCR1IM+(8*p+0)*8,r0); VST(SCR1IM+(8*p+1)*8,r1); VST(SCR1IM+(8*p+2)*8,r2); VST(SCR1IM+(8*p+3)*8,r3);
        VST(SCR1IM+(8*p+4)*8,r4); VST(SCR1IM+(8*p+5)*8,r5); VST(SCR1IM+(8*p+6)*8,r6); VST(SCR1IM+(8*p+7)*8,r7);
      }
      generic_col(L, SCR1RE, SCR1IM, 8, SCR2RE, SCR2IM, 8);
      for (int k = 0; k < L; k++){
        V re = VLD(SCR2RE + k*8), im = VLD(SCR2IM + k*8);
        long o = (long)t*P*16 + k*8;
        V cre = VLD(cc + o), cim = VLD(cc + PLANE + o);
        map8(&re, &im, cre, cim);
        VST(db + o, re); VST(db + PLANE + o, im);
      }
    }
  }
}

// ========================= fast codelets =====================================
// All codelets: read rows sre/sim + j*ss, write rows dre/dim + k*ds, natural order.
// Strides are compile-time constants at every call site (always_inline).

// trig constants (filled at init with long-double accuracy)
static double K_C2;                   // 1/sqrt(2)
static double K_S3;                   // sqrt(3)/2
static double K_C51,K_C52,K_S51,K_S52; // cos/sin 2pi/5, 4pi/5
static double K_C91,K_S91,K_C92,K_S92,K_C94,K_S94; // W9^{1,2,4}
static double TW64C[8][8], TW64S[8][8];  // W64^{j2*k1}
static double DS13C[7][7], DS13S[7][7];  // [j][k] j,k in 1..6
static double DS17C[9][9], DS17S[9][9];
static double DS23C[12][12], DS23S[12][12];

// complex butterfly helpers on register pairs
#define CADD(or_,oi_,ar,ai,br,bi) { or_ = VADD(ar,br); oi_ = VADD(ai,bi); }
#define CSUB(or_,oi_,ar,ai,br,bi) { or_ = VSUB(ar,br); oi_ = VSUB(ai,bi); }
// y = x * (c - i s)
#define CMULF(yr,yi,xr,xi,c,s) { V _cc=VSET1(c), _ss=VSET1(s); \
  yr = VFMA(_ss, xi, VMUL(_cc, xr)); yi = VFNMA(_ss, xr, VMUL(_cc, xi)); }

// ---- F2/F3/F4/F5 on register pairs (outputs may alias inputs) ----
#define F3K(x0r,x0i,x1r,x1i,x2r,x2i, X0r,X0i,X1r,X1i,X2r,X2i) { \
  V _tr = VADD(x1r,x2r), _ti = VADD(x1i,x2i);                   \
  V _dr = VSUB(x1r,x2r), _di = VSUB(x1i,x2i);                   \
  V _mr = VFNMA(_tr, VSET1(0.5), x0r), _mi = VFNMA(_ti, VSET1(0.5), x0i); \
  X0r = VADD(x0r,_tr); X0i = VADD(x0i,_ti);                     \
  V _s3 = VSET1(K_S3);                                          \
  X1r = VFMA(_di,_s3,_mr); X1i = VFNMA(_dr,_s3,_mi);            \
  X2r = VFNMA(_di,_s3,_mr); X2i = VFMA(_dr,_s3,_mi);            \
}
#define F4K(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i, X0r,X0i,X1r,X1i,X2r,X2i,X3r,X3i) { \
  V _t0r=VADD(x0r,x2r), _t0i=VADD(x0i,x2i), _t1r=VSUB(x0r,x2r), _t1i=VSUB(x0i,x2i); \
  V _t2r=VADD(x1r,x3r), _t2i=VADD(x1i,x3i), _t3r=VSUB(x1r,x3r), _t3i=VSUB(x1i,x3i); \
  X0r=VADD(_t0r,_t2r); X0i=VADD(_t0i,_t2i); X2r=VSUB(_t0r,_t2r); X2i=VSUB(_t0i,_t2i); \
  X1r=VADD(_t1r,_t3i); X1i=VSUB(_t1i,_t3r); X3r=VSUB(_t1r,_t3i); X3i=VADD(_t1i,_t3r); \
}
#define F5K(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i, X0r,X0i,X1r,X1i,X2r,X2i,X3r,X3i,X4r,X4i) { \
  V _u1r=VADD(x1r,x4r), _u1i=VADD(x1i,x4i), _v1r=VSUB(x1r,x4r), _v1i=VSUB(x1i,x4i); \
  V _u2r=VADD(x2r,x3r), _u2i=VADD(x2i,x3i), _v2r=VSUB(x2r,x3r), _v2i=VSUB(x2i,x3i); \
  V _c51=VSET1(K_C51), _c52=VSET1(K_C52), _s51=VSET1(K_S51), _s52=VSET1(K_S52); \
  V _Ar = VFMA(_c52,_u2r, VFMA(_c51,_u1r, x0r)), _Ai = VFMA(_c52,_u2i, VFMA(_c51,_u1i, x0i)); \
  V _Br = VFMA(_c51,_u2r, VFMA(_c52,_u1r, x0r)), _Bi = VFMA(_c51,_u2i, VFMA(_c52,_u1i, x0i)); \
  V _Sr = VFMA(_s52,_v2r, VMUL(_s51,_v1r)), _Si = VFMA(_s52,_v2i, VMUL(_s51,_v1i)); \
  V _Tr = VFNMA(_s51,_v2r, VMUL(_s52,_v1r)), _Ti = VFNMA(_s51,_v2i, VMUL(_s52,_v1i)); \
  X0r = VADD(x0r, VADD(_u1r,_u2r)); X0i = VADD(x0i, VADD(_u1i,_u2i)); \
  X1r = VADD(_Ar,_Si); X1i = VSUB(_Ai,_Sr); X4r = VSUB(_Ar,_Si); X4i = VADD(_Ai,_Sr); \
  X2r = VADD(_Br,_Ti); X2i = VSUB(_Bi,_Tr); X3r = VSUB(_Br,_Ti); X3i = VADD(_Bi,_Tr); \
}
// F8 on 8 register pairs, natural in/out
#define F8K(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i, \
            X0r,X0i,X1r,X1i,X2r,X2i,X3r,X3i,X4r,X4i,X5r,X5i,X6r,X6i,X7r,X7i) { \
  V _a0r=VADD(x0r,x4r), _a0i=VADD(x0i,x4i), _b0r=VSUB(x0r,x4r), _b0i=VSUB(x0i,x4i); \
  V _a1r=VADD(x1r,x5r), _a1i=VADD(x1i,x5i), _b1r=VSUB(x1r,x5r), _b1i=VSUB(x1i,x5i); \
  V _a2r=VADD(x2r,x6r), _a2i=VADD(x2i,x6i), _b2r=VSUB(x2r,x6r), _b2i=VSUB(x2i,x6i); \
  V _a3r=VADD(x3r,x7r), _a3i=VADD(x3i,x7i), _b3r=VSUB(x3r,x7r), _b3i=VSUB(x3i,x7i); \
  /* evens: F4(a0..a3) */ \
  V _t0r=VADD(_a0r,_a2r), _t0i=VADD(_a0i,_a2i), _t1r=VSUB(_a0r,_a2r), _t1i=VSUB(_a0i,_a2i); \
  V _t2r=VADD(_a1r,_a3r), _t2i=VADD(_a1i,_a3i), _t3r=VSUB(_a1r,_a3r), _t3i=VSUB(_a1i,_a3i); \
  X0r=VADD(_t0r,_t2r); X0i=VADD(_t0i,_t2i); X4r=VSUB(_t0r,_t2r); X4i=VSUB(_t0i,_t2i); \
  X2r=VADD(_t1r,_t3i); X2i=VSUB(_t1i,_t3r); X6r=VSUB(_t1r,_t3i); X6i=VADD(_t1i,_t3r); \
  /* odds: twiddle then F4 */ \
  V _C2 = VSET1(K_C2); \
  V _y1r=VMUL(VADD(_b1r,_b1i),_C2), _y1i=VMUL(VSUB(_b1i,_b1r),_C2); \
  V _y3r=VMUL(VSUB(_b3i,_b3r),_C2), _y3i=VMUL(VADD(_b3r,_b3i),VSET1(K_NC2)); \
  V _s0r=VADD(_b0r,_b2i), _s0i=VSUB(_b0i,_b2r); /* b0 + (-i)b2 */ \
  V _s1r=VSUB(_b0r,_b2i), _s1i=VADD(_b0i,_b2r); \
  V _s2r=VADD(_y1r,_y3r), _s2i=VADD(_y1i,_y3i); \
  V _s3r=VSUB(_y1r,_y3r), _s3i=VSUB(_y1i,_y3i); \
  X1r=VADD(_s0r,_s2r); X1i=VADD(_s0i,_s2i); X5r=VSUB(_s0r,_s2r); X5i=VSUB(_s0i,_s2i); \
  X3r=VADD(_s1r,_s3i); X3i=VSUB(_s1i,_s3r); X7r=VSUB(_s1r,_s3i); X7i=VADD(_s1i,_s3r); \
}
static double K_NC2; // -1/sqrt(2)

#define LDROW(vr,vi,j) V vr = VLDU(sre + (long)(j)*ss), vi = VLDU(sim + (long)(j)*ss)
#define STROW(k,vr,vi) { VSTU(dre + (long)(k)*ds, vr); VSTU(dim + (long)(k)*ds, vi); }

// ---- L=6: PFA 2x3 ----
AI void cod6(const double* sre, const double* sim, long ss, double* dre, double* dim, long ds){
  LDROW(x0r,x0i,0); LDROW(x3r,x3i,3); LDROW(x2r,x2i,2); LDROW(x5r,x5i,5); LDROW(x4r,x4i,4); LDROW(x1r,x1i,1);
  // F2 over n1 per n2: rows {0,3},{2,5},{4,1}
  V a0r=VADD(x0r,x3r), a0i=VADD(x0i,x3i), b0r=VSUB(x0r,x3r), b0i=VSUB(x0i,x3i);
  V a1r=VADD(x2r,x5r), a1i=VADD(x2i,x5i), b1r=VSUB(x2r,x5r), b1i=VSUB(x2i,x5i);
  V a2r=VADD(x4r,x1r), a2i=VADD(x4i,x1i), b2r=VSUB(x4r,x1r), b2i=VSUB(x4i,x1i);
  V A0r,A0i,A1r,A1i,A2r,A2i, B0r,B0i,B1r,B1i,B2r,B2i;
  F3K(a0r,a0i,a1r,a1i,a2r,a2i, A0r,A0i,A1r,A1i,A2r,A2i);   // k1=0 -> rows {0,4,2}
  F3K(b0r,b0i,b1r,b1i,b2r,b2i, B0r,B0i,B1r,B1i,B2r,B2i);   // k1=1 -> rows {3,1,5}
  STROW(0,A0r,A0i); STROW(4,A1r,A1i); STROW(2,A2r,A2i);
  STROW(3,B0r,B0i); STROW(1,B1r,B1i); STROW(5,B2r,B2i);
}

// ---- L=8 ----
AI void cod8(const double* sre, const double* sim, long ss, double* dre, double* dim, long ds){
  LDROW(x0r,x0i,0); LDROW(x1r,x1i,1); LDROW(x2r,x2i,2); LDROW(x3r,x3i,3);
  LDROW(x4r,x4i,4); LDROW(x5r,x5i,5); LDROW(x6r,x6i,6); LDROW(x7r,x7i,7);
  V X0r,X0i,X1r,X1i,X2r,X2i,X3r,X3i,X4r,X4i,X5r,X5i,X6r,X6i,X7r,X7i;
  F8K(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,
      X0r,X0i,X1r,X1i,X2r,X2i,X3r,X3i,X4r,X4i,X5r,X5i,X6r,X6i,X7r,X7i);
  STROW(0,X0r,X0i); STROW(1,X1r,X1i); STROW(2,X2r,X2i); STROW(3,X3r,X3i);
  STROW(4,X4r,X4i); STROW(5,X5r,X5i); STROW(6,X6r,X6i); STROW(7,X7r,X7i);
}

// ---- direct symmetric prime codelets (13/17/23) ----
#define DS_ACC(H, CT, ST, KLIST_N, K0,K1,K2,K3,K4,K5, PFX, WITH_X0SUM) \
  V PFX##cu0r=x0r, PFX##cu0i=x0i, PFX##sv0r=VSET1(0.0), PFX##sv0i=VSET1(0.0); \
  V PFX##cu1r=x0r, PFX##cu1i=x0i, PFX##sv1r=VSET1(0.0), PFX##sv1i=VSET1(0.0); \
  V PFX##cu2r=x0r, PFX##cu2i=x0i, PFX##sv2r=VSET1(0.0), PFX##sv2i=VSET1(0.0); \
  V PFX##cu3r=x0r, PFX##cu3i=x0i, PFX##sv3r=VSET1(0.0), PFX##sv3i=VSET1(0.0); \
  V PFX##cu4r=x0r, PFX##cu4i=x0i, PFX##sv4r=VSET1(0.0), PFX##sv4i=VSET1(0.0); \
  V PFX##cu5r=x0r, PFX##cu5i=x0i, PFX##sv5r=VSET1(0.0), PFX##sv5i=VSET1(0.0); \
  (void)PFX##cu4r; (void)PFX##cu4i; (void)PFX##sv4r; (void)PFX##sv4i; \
  (void)PFX##cu5r; (void)PFX##cu5i; (void)PFX##sv5r; (void)PFX##sv5i; \
  for (int j = 1; j <= H; j++){ \
    V ur = VADD(VLDU(sre+(long)j*ss), VLDU(sre+(long)(LL0-j)*ss)); \
    V vr = VSUB(VLDU(sre+(long)j*ss), VLDU(sre+(long)(LL0-j)*ss)); \
    V ui = VADD(VLDU(sim+(long)j*ss), VLDU(sim+(long)(LL0-j)*ss)); \
    V vi = VSUB(VLDU(sim+(long)j*ss), VLDU(sim+(long)(LL0-j)*ss)); \
    if (WITH_X0SUM){ t0r = VADD(t0r, ur); t0i = VADD(t0i, ui); } \
    V c0=VSET1(CT[j][K0]), s0=VSET1(ST[j][K0]); \
    PFX##cu0r=VFMA(c0,ur,PFX##cu0r); PFX##cu0i=VFMA(c0,ui,PFX##cu0i); PFX##sv0r=VFMA(s0,vr,PFX##sv0r); PFX##sv0i=VFMA(s0,vi,PFX##sv0i); \
    if (KLIST_N > 1){ V c1=VSET1(CT[j][K1]), s1=VSET1(ST[j][K1]); \
      PFX##cu1r=VFMA(c1,ur,PFX##cu1r); PFX##cu1i=VFMA(c1,ui,PFX##cu1i); PFX##sv1r=VFMA(s1,vr,PFX##sv1r); PFX##sv1i=VFMA(s1,vi,PFX##sv1i);} \
    if (KLIST_N > 2){ V c2=VSET1(CT[j][K2]), s2=VSET1(ST[j][K2]); \
      PFX##cu2r=VFMA(c2,ur,PFX##cu2r); PFX##cu2i=VFMA(c2,ui,PFX##cu2i); PFX##sv2r=VFMA(s2,vr,PFX##sv2r); PFX##sv2i=VFMA(s2,vi,PFX##sv2i);} \
    if (KLIST_N > 3){ V c3=VSET1(CT[j][K3]), s3=VSET1(ST[j][K3]); \
      PFX##cu3r=VFMA(c3,ur,PFX##cu3r); PFX##cu3i=VFMA(c3,ui,PFX##cu3i); PFX##sv3r=VFMA(s3,vr,PFX##sv3r); PFX##sv3i=VFMA(s3,vi,PFX##sv3i);} \
    if (KLIST_N > 4){ V c4=VSET1(CT[j][K4]), s4=VSET1(ST[j][K4]); \
      PFX##cu4r=VFMA(c4,ur,PFX##cu4r); PFX##cu4i=VFMA(c4,ui,PFX##cu4i); PFX##sv4r=VFMA(s4,vr,PFX##sv4r); PFX##sv4i=VFMA(s4,vi,PFX##sv4i);} \
    if (KLIST_N > 5){ V c5=VSET1(CT[j][K5]), s5=VSET1(ST[j][K5]); \
      PFX##cu5r=VFMA(c5,ur,PFX##cu5r); PFX##cu5i=VFMA(c5,ui,PFX##cu5i); PFX##sv5r=VFMA(s5,vr,PFX##sv5r); PFX##sv5i=VFMA(s5,vi,PFX##sv5i);} \
  }
#define DS_OUT(KLIST_N, K0,K1,K2,K3,K4,K5, PFX) { \
  STROW(K0, VADD(PFX##cu0r,PFX##sv0i), VSUB(PFX##cu0i,PFX##sv0r)); STROW(LL0-K0, VSUB(PFX##cu0r,PFX##sv0i), VADD(PFX##cu0i,PFX##sv0r)); \
  if (KLIST_N > 1){ STROW(K1, VADD(PFX##cu1r,PFX##sv1i), VSUB(PFX##cu1i,PFX##sv1r)); STROW(LL0-K1, VSUB(PFX##cu1r,PFX##sv1i), VADD(PFX##cu1i,PFX##sv1r)); } \
  if (KLIST_N > 2){ STROW(K2, VADD(PFX##cu2r,PFX##sv2i), VSUB(PFX##cu2i,PFX##sv2r)); STROW(LL0-K2, VSUB(PFX##cu2r,PFX##sv2i), VADD(PFX##cu2i,PFX##sv2r)); } \
  if (KLIST_N > 3){ STROW(K3, VADD(PFX##cu3r,PFX##sv3i), VSUB(PFX##cu3i,PFX##sv3r)); STROW(LL0-K3, VSUB(PFX##cu3r,PFX##sv3i), VADD(PFX##cu3i,PFX##sv3r)); } \
  if (KLIST_N > 4){ STROW(K4, VADD(PFX##cu4r,PFX##sv4i), VSUB(PFX##cu4i,PFX##sv4r)); STROW(LL0-K4, VSUB(PFX##cu4r,PFX##sv4i), VADD(PFX##cu4i,PFX##sv4r)); } \
  if (KLIST_N > 5){ STROW(K5, VADD(PFX##cu5r,PFX##sv5i), VSUB(PFX##cu5i,PFX##sv5r)); STROW(LL0-K5, VSUB(PFX##cu5r,PFX##sv5i), VADD(PFX##cu5i,PFX##sv5r)); } \
}
#define DS_GROUP(H, CT, ST, KLIST_N, K0,K1,K2,K3,K4,K5, WITH_X0SUM) { \
  DS_ACC(H, CT, ST, KLIST_N, K0,K1,K2,K3,K4,K5, g_, WITH_X0SUM) \
  DS_OUT(KLIST_N, K0,K1,K2,K3,K4,K5, g_) \
}

#define LL0 13
AI void cod13(const double* sre, const double* sim, long ss, double* dre, double* dim, long ds){
  V x0r = VLDU(sre), x0i = VLDU(sim);
  V t0r = x0r, t0i = x0i;
  DS_ACC(6, DS13C, DS13S, 6, 1,2,3,4,5,6, a_, 1)
  STROW(0, t0r, t0i);
  DS_OUT(6, 1,2,3,4,5,6, a_)
}
#undef LL0
#define LL0 17
AI void cod17(const double* sre, const double* sim, long ss, double* dre, double* dim, long ds){
  V x0r = VLDU(sre), x0i = VLDU(sim);
  V t0r = x0r, t0i = x0i;
  DS_ACC(8, DS17C, DS17S, 4, 1,2,3,4,0,0, a_, 1)
  DS_ACC(8, DS17C, DS17S, 4, 5,6,7,8,0,0, b_, 0)
  STROW(0, t0r, t0i);
  DS_OUT(4, 1,2,3,4,0,0, a_)
  DS_OUT(4, 5,6,7,8,0,0, b_)
}
#undef LL0
#define LL0 23
AI void cod23(const double* sre, const double* sim, long ss, double* dre, double* dim, long ds){
  V x0r = VLDU(sre), x0i = VLDU(sim);
  V t0r = x0r, t0i = x0i;
  DS_GROUP(11, DS23C, DS23S, 6, 1,2,3,4,5,6, 1);
  DS_GROUP(11, DS23C, DS23S, 5, 7,8,9,10,11,0, 0);
  STROW(0, t0r, t0i);
}
#undef LL0

// ---- F9 on register pairs (CT 3x3) ----
// inputs x0..x8 (register pairs), outputs Y0..Y8, natural order both sides
#define F9K(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,x8r,x8i, \
            Y0r,Y0i,Y1r,Y1i,Y2r,Y2i,Y3r,Y3i,Y4r,Y4i,Y5r,Y5i,Y6r,Y6i,Y7r,Y7i,Y8r,Y8i) { \
  V A0r,A0i,A1r,A1i,A2r,A2i, B0r,B0i,B1r,B1i,B2r,B2i, C0r,C0i,C1r,C1i,C2r,C2i; \
  F3K(x0r,x0i,x3r,x3i,x6r,x6i, A0r,A0i,A1r,A1i,A2r,A2i); \
  F3K(x1r,x1i,x4r,x4i,x7r,x7i, B0r,B0i,B1r,B1i,B2r,B2i); \
  F3K(x2r,x2i,x5r,x5i,x8r,x8i, C0r,C0i,C1r,C1i,C2r,C2i); \
  V w1r,w1i,w2r,w2i,w3r,w3i,w4r,w4i; \
  CMULF(w1r,w1i,B1r,B1i,K_C91,K_S91); CMULF(w2r,w2i,C1r,C1i,K_C92,K_S92); \
  CMULF(w3r,w3i,B2r,B2i,K_C92,K_S92); CMULF(w4r,w4i,C2r,C2i,K_C94,K_S94); \
  F3K(A0r,A0i,B0r,B0i,C0r,C0i, Y0r,Y0i,Y3r,Y3i,Y6r,Y6i); \
  F3K(A1r,A1i,w1r,w1i,w2r,w2i, Y1r,Y1i,Y4r,Y4i,Y7r,Y7i); \
  F3K(A2r,A2i,w3r,w3i,w4r,w4i, Y2r,Y2i,Y5r,Y5i,Y8r,Y8i); \
}

// ---- L=36: PFA 4x9 ----
static const int IN36[9][4] = {{0,9,18,27},{4,13,22,31},{8,17,26,35},{12,21,30,3},{16,25,34,7},{20,29,2,11},{24,33,6,15},{28,1,10,19},{32,5,14,23}};
static const int OUT36[4][9] = {{0,28,20,12,4,32,24,16,8},{9,1,29,21,13,5,33,25,17},{18,10,2,30,22,14,6,34,26},{27,19,11,3,31,23,15,7,35}};
AI void cod36(const double* sre, const double* sim, long ss, double* dre, double* dim, long ds){
  for (int n2 = 0; n2 < 9; n2++){
    const int* q = IN36[n2];
    V a0r=VLDU(sre+(long)q[0]*ss), a0i=VLDU(sim+(long)q[0]*ss);
    V a1r=VLDU(sre+(long)q[1]*ss), a1i=VLDU(sim+(long)q[1]*ss);
    V a2r=VLDU(sre+(long)q[2]*ss), a2i=VLDU(sim+(long)q[2]*ss);
    V a3r=VLDU(sre+(long)q[3]*ss), a3i=VLDU(sim+(long)q[3]*ss);
    V Y0r,Y0i,Y1r,Y1i,Y2r,Y2i,Y3r,Y3i;
    F4K(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i, Y0r,Y0i,Y1r,Y1i,Y2r,Y2i,Y3r,Y3i);
    STROW(4*n2+0,Y0r,Y0i); STROW(4*n2+1,Y1r,Y1i); STROW(4*n2+2,Y2r,Y2i); STROW(4*n2+3,Y3r,Y3i);
  }
  for (int k1 = 0; k1 < 4; k1++){
    const int* q = OUT36[k1];
    V x0r=VLDU(dre+(long)(k1   )*ds), x0i=VLDU(dim+(long)(k1   )*ds);
    V x1r=VLDU(dre+(long)(k1+ 4)*ds), x1i=VLDU(dim+(long)(k1+ 4)*ds);
    V x2r=VLDU(dre+(long)(k1+ 8)*ds), x2i=VLDU(dim+(long)(k1+ 8)*ds);
    V x3r=VLDU(dre+(long)(k1+12)*ds), x3i=VLDU(dim+(long)(k1+12)*ds);
    V x4r=VLDU(dre+(long)(k1+16)*ds), x4i=VLDU(dim+(long)(k1+16)*ds);
    V x5r=VLDU(dre+(long)(k1+20)*ds), x5i=VLDU(dim+(long)(k1+20)*ds);
    V x6r=VLDU(dre+(long)(k1+24)*ds), x6i=VLDU(dim+(long)(k1+24)*ds);
    V x7r=VLDU(dre+(long)(k1+28)*ds), x7i=VLDU(dim+(long)(k1+28)*ds);
    V x8r=VLDU(dre+(long)(k1+32)*ds), x8i=VLDU(dim+(long)(k1+32)*ds);
    V Y0r,Y0i,Y1r,Y1i,Y2r,Y2i,Y3r,Y3i,Y4r,Y4i,Y5r,Y5i,Y6r,Y6i,Y7r,Y7i,Y8r,Y8i;
    F9K(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,x8r,x8i,
        Y0r,Y0i,Y1r,Y1i,Y2r,Y2i,Y3r,Y3i,Y4r,Y4i,Y5r,Y5i,Y6r,Y6i,Y7r,Y7i,Y8r,Y8i);
    VSTU(dre+(long)q[0]*ds,Y0r); VSTU(dim+(long)q[0]*ds,Y0i);
    VSTU(dre+(long)q[1]*ds,Y1r); VSTU(dim+(long)q[1]*ds,Y1i);
    VSTU(dre+(long)q[2]*ds,Y2r); VSTU(dim+(long)q[2]*ds,Y2i);
    VSTU(dre+(long)q[3]*ds,Y3r); VSTU(dim+(long)q[3]*ds,Y3i);
    VSTU(dre+(long)q[4]*ds,Y4r); VSTU(dim+(long)q[4]*ds,Y4i);
    VSTU(dre+(long)q[5]*ds,Y5r); VSTU(dim+(long)q[5]*ds,Y5i);
    VSTU(dre+(long)q[6]*ds,Y6r); VSTU(dim+(long)q[6]*ds,Y6i);
    VSTU(dre+(long)q[7]*ds,Y7r); VSTU(dim+(long)q[7]*ds,Y7i);
    VSTU(dre+(long)q[8]*ds,Y8r); VSTU(dim+(long)q[8]*ds,Y8i);
  }
}

// ---- L=45: PFA 5x9 ----
static const int IN45[9][5] = {{0,9,18,27,36},{5,14,23,32,41},{10,19,28,37,1},{15,24,33,42,6},{20,29,38,2,11},{25,34,43,7,16},{30,39,3,12,21},{35,44,8,17,26},{40,4,13,22,31}};
static const int OUT45[5][9] = {{0,10,20,30,40,5,15,25,35},{36,1,11,21,31,41,6,16,26},{27,37,2,12,22,32,42,7,17},{18,28,38,3,13,23,33,43,8},{9,19,29,39,4,14,24,34,44}};
AI void cod45(const double* sre, const double* sim, long ss, double* dre, double* dim, long ds){
  for (int n2 = 0; n2 < 9; n2++){
    const int* q = IN45[n2];
    V a0r=VLDU(sre+(long)q[0]*ss), a0i=VLDU(sim+(long)q[0]*ss);
    V a1r=VLDU(sre+(long)q[1]*ss), a1i=VLDU(sim+(long)q[1]*ss);
    V a2r=VLDU(sre+(long)q[2]*ss), a2i=VLDU(sim+(long)q[2]*ss);
    V a3r=VLDU(sre+(long)q[3]*ss), a3i=VLDU(sim+(long)q[3]*ss);
    V a4r=VLDU(sre+(long)q[4]*ss), a4i=VLDU(sim+(long)q[4]*ss);
    V Y0r,Y0i,Y1r,Y1i,Y2r,Y2i,Y3r,Y3i,Y4r,Y4i;
    F5K(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i,a4r,a4i, Y0r,Y0i,Y1r,Y1i,Y2r,Y2i,Y3r,Y3i,Y4r,Y4i);
    STROW(5*n2+0,Y0r,Y0i); STROW(5*n2+1,Y1r,Y1i); STROW(5*n2+2,Y2r,Y2i); STROW(5*n2+3,Y3r,Y3i); STROW(5*n2+4,Y4r,Y4i);
  }
  for (int k1 = 0; k1 < 5; k1++){
    const int* q = OUT45[k1];
    V x0r=VLDU(dre+(long)(k1   )*ds), x0i=VLDU(dim+(long)(k1   )*ds);
    V x1r=VLDU(dre+(long)(k1+ 5)*ds), x1i=VLDU(dim+(long)(k1+ 5)*ds);
    V x2r=VLDU(dre+(long)(k1+10)*ds), x2i=VLDU(dim+(long)(k1+10)*ds);
    V x3r=VLDU(dre+(long)(k1+15)*ds), x3i=VLDU(dim+(long)(k1+15)*ds);
    V x4r=VLDU(dre+(long)(k1+20)*ds), x4i=VLDU(dim+(long)(k1+20)*ds);
    V x5r=VLDU(dre+(long)(k1+25)*ds), x5i=VLDU(dim+(long)(k1+25)*ds);
    V x6r=VLDU(dre+(long)(k1+30)*ds), x6i=VLDU(dim+(long)(k1+30)*ds);
    V x7r=VLDU(dre+(long)(k1+35)*ds), x7i=VLDU(dim+(long)(k1+35)*ds);
    V x8r=VLDU(dre+(long)(k1+40)*ds), x8i=VLDU(dim+(long)(k1+40)*ds);
    V Y0r,Y0i,Y1r,Y1i,Y2r,Y2i,Y3r,Y3i,Y4r,Y4i,Y5r,Y5i,Y6r,Y6i,Y7r,Y7i,Y8r,Y8i;
    F9K(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,x8r,x8i,
        Y0r,Y0i,Y1r,Y1i,Y2r,Y2i,Y3r,Y3i,Y4r,Y4i,Y5r,Y5i,Y6r,Y6i,Y7r,Y7i,Y8r,Y8i);
    VSTU(dre+(long)q[0]*ds,Y0r); VSTU(dim+(long)q[0]*ds,Y0i);
    VSTU(dre+(long)q[1]*ds,Y1r); VSTU(dim+(long)q[1]*ds,Y1i);
    VSTU(dre+(long)q[2]*ds,Y2r); VSTU(dim+(long)q[2]*ds,Y2i);
    VSTU(dre+(long)q[3]*ds,Y3r); VSTU(dim+(long)q[3]*ds,Y3i);
    VSTU(dre+(long)q[4]*ds,Y4r); VSTU(dim+(long)q[4]*ds,Y4i);
    VSTU(dre+(long)q[5]*ds,Y5r); VSTU(dim+(long)q[5]*ds,Y5i);
    VSTU(dre+(long)q[6]*ds,Y6r); VSTU(dim+(long)q[6]*ds,Y6i);
    VSTU(dre+(long)q[7]*ds,Y7r); VSTU(dim+(long)q[7]*ds,Y7i);
    VSTU(dre+(long)q[8]*ds,Y8r); VSTU(dim+(long)q[8]*ds,Y8i);
  }
}

// ---- L=64: CT 8x8 ----
AI void cod64(const double* sre, const double* sim, long ss, double* dre, double* dim, long ds){
  // stage A: for j2: F8 over rows {8*j1+j2}, twiddle W64^{j2*k1}, store rows 8*j2+k1
  for (int j2 = 0; j2 < 8; j2++){
    V x0r=VLDU(sre+(long)(j2   )*ss), x0i=VLDU(sim+(long)(j2   )*ss);
    V x1r=VLDU(sre+(long)(j2+ 8)*ss), x1i=VLDU(sim+(long)(j2+ 8)*ss);
    V x2r=VLDU(sre+(long)(j2+16)*ss), x2i=VLDU(sim+(long)(j2+16)*ss);
    V x3r=VLDU(sre+(long)(j2+24)*ss), x3i=VLDU(sim+(long)(j2+24)*ss);
    V x4r=VLDU(sre+(long)(j2+32)*ss), x4i=VLDU(sim+(long)(j2+32)*ss);
    V x5r=VLDU(sre+(long)(j2+40)*ss), x5i=VLDU(sim+(long)(j2+40)*ss);
    V x6r=VLDU(sre+(long)(j2+48)*ss), x6i=VLDU(sim+(long)(j2+48)*ss);
    V x7r=VLDU(sre+(long)(j2+56)*ss), x7i=VLDU(sim+(long)(j2+56)*ss);
    V X0r,X0i,X1r,X1i,X2r,X2i,X3r,X3i,X4r,X4i,X5r,X5i,X6r,X6i,X7r,X7i;
    F8K(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,
        X0r,X0i,X1r,X1i,X2r,X2i,X3r,X3i,X4r,X4i,X5r,X5i,X6r,X6i,X7r,X7i);
    double* wr = dre + (long)(8*j2)*ds; double* wi = dim + (long)(8*j2)*ds;
    if (j2 == 0){
      VSTU(wr,X0r); VSTU(wi,X0i);
      VSTU(wr+ds,X1r); VSTU(wi+ds,X1i); VSTU(wr+2*ds,X2r); VSTU(wi+2*ds,X2i);
      VSTU(wr+3*ds,X3r); VSTU(wi+3*ds,X3i); VSTU(wr+4*ds,X4r); VSTU(wi+4*ds,X4i);
      VSTU(wr+5*ds,X5r); VSTU(wi+5*ds,X5i); VSTU(wr+6*ds,X6r); VSTU(wi+6*ds,X6i);
      VSTU(wr+7*ds,X7r); VSTU(wi+7*ds,X7i);
    } else {
      VSTU(wr,X0r); VSTU(wi,X0i);
      V tr, ti;
      CMULF(tr,ti,X1r,X1i,TW64C[j2][1],TW64S[j2][1]); VSTU(wr+ds,tr); VSTU(wi+ds,ti);
      CMULF(tr,ti,X2r,X2i,TW64C[j2][2],TW64S[j2][2]); VSTU(wr+2*ds,tr); VSTU(wi+2*ds,ti);
      CMULF(tr,ti,X3r,X3i,TW64C[j2][3],TW64S[j2][3]); VSTU(wr+3*ds,tr); VSTU(wi+3*ds,ti);
      CMULF(tr,ti,X4r,X4i,TW64C[j2][4],TW64S[j2][4]); VSTU(wr+4*ds,tr); VSTU(wi+4*ds,ti);
      CMULF(tr,ti,X5r,X5i,TW64C[j2][5],TW64S[j2][5]); VSTU(wr+5*ds,tr); VSTU(wi+5*ds,ti);
      CMULF(tr,ti,X6r,X6i,TW64C[j2][6],TW64S[j2][6]); VSTU(wr+6*ds,tr); VSTU(wi+6*ds,ti);
      CMULF(tr,ti,X7r,X7i,TW64C[j2][7],TW64S[j2][7]); VSTU(wr+7*ds,tr); VSTU(wi+7*ds,ti);
    }
  }
  // stage B: for k1: F8 over dst rows {8*j2+k1}, write X[k1+8*k2] at row k1+8*k2
  for (int k1 = 0; k1 < 8; k1++){
    V x0r=VLDU(dre+(long)(k1   )*ds), x0i=VLDU(dim+(long)(k1   )*ds);
    V x1r=VLDU(dre+(long)(k1+ 8)*ds), x1i=VLDU(dim+(long)(k1+ 8)*ds);
    V x2r=VLDU(dre+(long)(k1+16)*ds), x2i=VLDU(dim+(long)(k1+16)*ds);
    V x3r=VLDU(dre+(long)(k1+24)*ds), x3i=VLDU(dim+(long)(k1+24)*ds);
    V x4r=VLDU(dre+(long)(k1+32)*ds), x4i=VLDU(dim+(long)(k1+32)*ds);
    V x5r=VLDU(dre+(long)(k1+40)*ds), x5i=VLDU(dim+(long)(k1+40)*ds);
    V x6r=VLDU(dre+(long)(k1+48)*ds), x6i=VLDU(dim+(long)(k1+48)*ds);
    V x7r=VLDU(dre+(long)(k1+56)*ds), x7i=VLDU(dim+(long)(k1+56)*ds);
    V X0r,X0i,X1r,X1i,X2r,X2i,X3r,X3i,X4r,X4i,X5r,X5i,X6r,X6i,X7r,X7i;
    F8K(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,
        X0r,X0i,X1r,X1i,X2r,X2i,X3r,X3i,X4r,X4i,X5r,X5i,X6r,X6i,X7r,X7i);
    VSTU(dre+(long)(k1   )*ds,X0r); VSTU(dim+(long)(k1   )*ds,X0i);
    VSTU(dre+(long)(k1+ 8)*ds,X1r); VSTU(dim+(long)(k1+ 8)*ds,X1i);
    VSTU(dre+(long)(k1+16)*ds,X2r); VSTU(dim+(long)(k1+16)*ds,X2i);
    VSTU(dre+(long)(k1+24)*ds,X3r); VSTU(dim+(long)(k1+24)*ds,X3i);
    VSTU(dre+(long)(k1+32)*ds,X4r); VSTU(dim+(long)(k1+32)*ds,X4i);
    VSTU(dre+(long)(k1+40)*ds,X5r); VSTU(dim+(long)(k1+40)*ds,X5i);
    VSTU(dre+(long)(k1+48)*ds,X6r); VSTU(dim+(long)(k1+48)*ds,X6i);
    VSTU(dre+(long)(k1+56)*ds,X7r); VSTU(dim+(long)(k1+56)*ds,X7i);
  }
}

// ---- L=64 variant: stage B fused with +c and map ----
AI void cod64c(const double* sre, const double* sim, long ss,
               double* tre, double* tim,
               double* dre, double* dim,
               const double* cre, const double* cim){
  const long ds = 8, ts = 8;
  // stage A: for j2: F8 over rows {8*j1+j2}, twiddle W64^{j2*k1}, store rows 8*j2+k1
  for (int j2 = 0; j2 < 8; j2++){
    V x0r=VLDU(sre+(long)(j2   )*ss), x0i=VLDU(sim+(long)(j2   )*ss);
    V x1r=VLDU(sre+(long)(j2+ 8)*ss), x1i=VLDU(sim+(long)(j2+ 8)*ss);
    V x2r=VLDU(sre+(long)(j2+16)*ss), x2i=VLDU(sim+(long)(j2+16)*ss);
    V x3r=VLDU(sre+(long)(j2+24)*ss), x3i=VLDU(sim+(long)(j2+24)*ss);
    V x4r=VLDU(sre+(long)(j2+32)*ss), x4i=VLDU(sim+(long)(j2+32)*ss);
    V x5r=VLDU(sre+(long)(j2+40)*ss), x5i=VLDU(sim+(long)(j2+40)*ss);
    V x6r=VLDU(sre+(long)(j2+48)*ss), x6i=VLDU(sim+(long)(j2+48)*ss);
    V x7r=VLDU(sre+(long)(j2+56)*ss), x7i=VLDU(sim+(long)(j2+56)*ss);
    V X0r,X0i,X1r,X1i,X2r,X2i,X3r,X3i,X4r,X4i,X5r,X5i,X6r,X6i,X7r,X7i;
    F8K(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,
        X0r,X0i,X1r,X1i,X2r,X2i,X3r,X3i,X4r,X4i,X5r,X5i,X6r,X6i,X7r,X7i);
    double* wr = tre + (long)(8*j2)*ts; double* wi = tim + (long)(8*j2)*ts;
    if (j2 == 0){
      VSTU(wr,X0r); VSTU(wi,X0i);
      VSTU(wr+ds,X1r); VSTU(wi+ds,X1i); VSTU(wr+2*ds,X2r); VSTU(wi+2*ds,X2i);
      VSTU(wr+3*ds,X3r); VSTU(wi+3*ds,X3i); VSTU(wr+4*ds,X4r); VSTU(wi+4*ds,X4i);
      VSTU(wr+5*ds,X5r); VSTU(wi+5*ds,X5i); VSTU(wr+6*ds,X6r); VSTU(wi+6*ds,X6i);
      VSTU(wr+7*ds,X7r); VSTU(wi+7*ds,X7i);
    } else {
      VSTU(wr,X0r); VSTU(wi,X0i);
      V tr, ti;
      CMULF(tr,ti,X1r,X1i,TW64C[j2][1],TW64S[j2][1]); VSTU(wr+ds,tr); VSTU(wi+ds,ti);
      CMULF(tr,ti,X2r,X2i,TW64C[j2][2],TW64S[j2][2]); VSTU(wr+2*ds,tr); VSTU(wi+2*ds,ti);
      CMULF(tr,ti,X3r,X3i,TW64C[j2][3],TW64S[j2][3]); VSTU(wr+3*ds,tr); VSTU(wi+3*ds,ti);
      CMULF(tr,ti,X4r,X4i,TW64C[j2][4],TW64S[j2][4]); VSTU(wr+4*ds,tr); VSTU(wi+4*ds,ti);
      CMULF(tr,ti,X5r,X5i,TW64C[j2][5],TW64S[j2][5]); VSTU(wr+5*ds,tr); VSTU(wi+5*ds,ti);
      CMULF(tr,ti,X6r,X6i,TW64C[j2][6],TW64S[j2][6]); VSTU(wr+6*ds,tr); VSTU(wi+6*ds,ti);
      CMULF(tr,ti,X7r,X7i,TW64C[j2][7],TW64S[j2][7]); VSTU(wr+7*ds,tr); VSTU(wi+7*ds,ti);
    }
  }
  // stage B: for k1: F8 over dst rows {8*j2+k1}, write X[k1+8*k2] at row k1+8*k2
  for (int k1 = 0; k1 < 8; k1++){
    V x0r=VLDU(tre+(long)(k1   )*ts), x0i=VLDU(tim+(long)(k1   )*ts);
    V x1r=VLDU(tre+(long)(k1+ 8)*ds), x1i=VLDU(tim+(long)(k1+ 8)*ds);
    V x2r=VLDU(tre+(long)(k1+16)*ds), x2i=VLDU(tim+(long)(k1+16)*ds);
    V x3r=VLDU(tre+(long)(k1+24)*ds), x3i=VLDU(tim+(long)(k1+24)*ds);
    V x4r=VLDU(tre+(long)(k1+32)*ds), x4i=VLDU(tim+(long)(k1+32)*ds);
    V x5r=VLDU(tre+(long)(k1+40)*ds), x5i=VLDU(tim+(long)(k1+40)*ds);
    V x6r=VLDU(tre+(long)(k1+48)*ds), x6i=VLDU(tim+(long)(k1+48)*ds);
    V x7r=VLDU(tre+(long)(k1+56)*ds), x7i=VLDU(tim+(long)(k1+56)*ds);
    V X0r,X0i,X1r,X1i,X2r,X2i,X3r,X3i,X4r,X4i,X5r,X5i,X6r,X6i,X7r,X7i;
    F8K(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,
        X0r,X0i,X1r,X1i,X2r,X2i,X3r,X3i,X4r,X4i,X5r,X5i,X6r,X6i,X7r,X7i);
#define MST64(OFF, XR, XI) { \
      V _cr = VLDU(cre+(long)(OFF)*ds), _ci = VLDU(cim+(long)(OFF)*ds); \
      map8m(&XR, &XI, _cr, _ci, ((OFF) & 3) < g_mapmix); \
      VSTU(dre+(long)(OFF)*ds, XR); VSTU(dim+(long)(OFF)*ds, XI); }
    MST64(k1,    X0r, X0i); MST64(k1+ 8, X1r, X1i); MST64(k1+16, X2r, X2i); MST64(k1+24, X3r, X3i);
    MST64(k1+32, X4r, X4i); MST64(k1+40, X5r, X5i); MST64(k1+48, X6r, X6i); MST64(k1+56, X7r, X7i);
#undef MST64
  }
}


// ========================= fast pass drivers =================================
#define DEF_PASSES(SFX, LL, PP, CODELET)                                        \
static void pass1_##SFX(const double* src, double* dst){                        \
  const long PLANE = (long)PP*8, SLAB = 2L*PP*PP + 8;                           \
  for (int p = 0; p < PP/8; p++)                                                \
    for (int k = 0; k < LL; k++){                                               \
      long o = (long)p*PP*16 + k*8;                                             \
      if (PP >= 64 && g_pf){                                                    \
        long dist = (g_pf >= 3) ? 16 : 8;                                       \
        const double* pf = src + o + dist;                                      \
        double* pfd = dst + o + dist;                                           \
        for (int r = 0; r < LL; r++){                                           \
          _mm_prefetch((const char*)(pf + (long)r*SLAB), _MM_HINT_T0);          \
          _mm_prefetch((const char*)(pf + PLANE + (long)r*SLAB), _MM_HINT_T0);  \
          if (g_pf == 2 || g_pf == 4){                                          \
            __builtin_prefetch(pfd + (long)r*SLAB, 1, 3);                       \
            __builtin_prefetch(pfd + PLANE + (long)r*SLAB, 1, 3);               \
          }                                                                     \
        }                                                                       \
      }                                                                         \
      CODELET(src+o, src+PLANE+o, SLAB, dst+o, dst+PLANE+o, SLAB);              \
    }                                                                           \
}                                                                               \
static void fused23_##SFX(const double* p1out, double* mid, double* dst, const double* restrict cvol){ \
  const long PLANE = (long)PP*8, SLAB = 2L*PP*PP + 8;                           \
  (void)mid;                                                                    \
  for (int x = 0; x < LL; x++){                                                 \
    { const double* sb = p1out + (long)x*SLAB; double* db = MIDSLAB;            \
      const char* cpf = (const char*)(cvol + (long)x*SLAB);                     \
      const long nlines = (g_pf23 && PP >= 64) ? (SLAB*8/64) : 0;                             \
      const long per = (nlines*8)/(PP > 0 ? PP : 1) + 1;                        \
      long pfi = 0;                                                             \
      for (int p = 0; p < PP/8; p++){                                           \
        long o = (long)p*PP*16;                                                 \
        if (g_pf23 == 2){ long lim = pfi + per; if (lim > nlines) lim = nlines; \
          for (; pfi < lim; pfi++) _mm_prefetch(cpf + pfi*64, _MM_HINT_T0); }   \
        CODELET(sb+o, sb+PLANE+o, 8, db+o, db+PLANE+o, 8);                      \
      }                                                                         \
    }                                                                           \
    const double* sb = MIDSLAB;                                                 \
    double* db = dst + (long)x*SLAB;                                            \
    const double* cc = cvol + (long)x*SLAB;                                     \
    const char* npf = (const char*)(p1out + (long)(x+1)*SLAB);                  \
    const long nl3 = (g_pf23 && PP >= 64 && x+1 < LL) ? (SLAB*8/64) : 0;                    \
    const long per3 = (nl3*8)/(PP > 0 ? PP : 1) + 1;                            \
    long pfi3 = 0;                                                              \
    for (int t = 0; t < PP/8; t++){                                             \
      if (g_pf23 == 2){ long lim = pfi3 + per3; if (lim > nl3) lim = nl3;       \
        for (; pfi3 < lim; pfi3++) _mm_prefetch(npf + pfi3*64, _MM_HINT_T0); }  \
      for (int p = 0; p < PP/8; p++){                                           \
        const double* q = sb + (long)p*PP*16 + (long)t*64;                      \
        V r0=VLD(q),r1=VLD(q+8),r2=VLD(q+16),r3=VLD(q+24),r4=VLD(q+32),r5=VLD(q+40),r6=VLD(q+48),r7=VLD(q+56); \
        TRANSP8(r0,r1,r2,r3,r4,r5,r6,r7);                                       \
        VST(SCR1RE+(8*p+0)*8,r0); VST(SCR1RE+(8*p+1)*8,r1); VST(SCR1RE+(8*p+2)*8,r2); VST(SCR1RE+(8*p+3)*8,r3); \
        VST(SCR1RE+(8*p+4)*8,r4); VST(SCR1RE+(8*p+5)*8,r5); VST(SCR1RE+(8*p+6)*8,r6); VST(SCR1RE+(8*p+7)*8,r7); \
        const double* qq = q + (long)PP*8;                                      \
        r0=VLD(qq);r1=VLD(qq+8);r2=VLD(qq+16);r3=VLD(qq+24);r4=VLD(qq+32);r5=VLD(qq+40);r6=VLD(qq+48);r7=VLD(qq+56); \
        TRANSP8(r0,r1,r2,r3,r4,r5,r6,r7);                                       \
        VST(SCR1IM+(8*p+0)*8,r0); VST(SCR1IM+(8*p+1)*8,r1); VST(SCR1IM+(8*p+2)*8,r2); VST(SCR1IM+(8*p+3)*8,r3); \
        VST(SCR1IM+(8*p+4)*8,r4); VST(SCR1IM+(8*p+5)*8,r5); VST(SCR1IM+(8*p+6)*8,r6); VST(SCR1IM+(8*p+7)*8,r7); \
      }                                                                         \
      CODELET(SCR1RE, SCR1IM, 8, SCR2RE, SCR2IM, 8);                            \
      {                                                                         \
        long o = (long)t*PP*16;                                                 \
        long onx = (long)(t+1)*PP*16;                                           \
        int dopf = (PP >= 64) && g_pf23 && (t+1 < PP/8);                        \
        for (int k = 0; k < LL; k++){                                           \
          if (dopf){                                                            \
            _mm_prefetch((const char*)(cc + onx + k*8), _MM_HINT_T0);           \
            _mm_prefetch((const char*)(cc + PLANE + onx + k*8), _MM_HINT_T0);   \
          }                                                                     \
          V re = VLD(SCR2RE + k*8), im = VLD(SCR2IM + k*8);                     \
          V cre = VLD(cc + o + k*8), cim = VLD(cc + PLANE + o + k*8);           \
          map8m(&re, &im, cre, cim, (k & 3) < g_mapmix);                        \
          VST(db + o + k*8, re); VST(db + PLANE + o + k*8, im);                 \
        }                                                                       \
      }                                                                         \
    }                                                                           \
  }                                                                             \
}

DEF_PASSES(s6, 6, 8, cod6)
DEF_PASSES(s8, 8, 8, cod8)
DEF_PASSES(s13, 13, 16, cod13)
DEF_PASSES(s17, 17, 24, cod17)
DEF_PASSES(s23, 23, 24, cod23)
DEF_PASSES(s36, 36, 40, cod36)
DEF_PASSES(s45, 45, 48, cod45)
DEF_PASSES(s64, 64, 64, cod64)


// ========================= octet engine (vectorize across 8 volumes) =========
// layout: position-major, pos = ((x*L+y)*L+z), data[pos*16 + 0..7] = RE lanes, +8..15 = IM lanes
#define STROWM(k,vr,vi) { \
  V _cr = VLDU(cre + (long)(k)*ds), _ci = VLDU(cim + (long)(k)*ds); \
  V _vr = vr, _vi = vi; map8m(&_vr, &_vi, _cr, _ci, ((k) & 3) < g_mapmix); \
  VSTU(dre + (long)(k)*ds, _vr); VSTU(dim + (long)(k)*ds, _vi); }

AI void cod6m(const double* sre, const double* sim, long ss, double* dre, double* dim, long ds,
              const double* cre, const double* cim){
  LDROW(x0r,x0i,0); LDROW(x3r,x3i,3); LDROW(x2r,x2i,2); LDROW(x5r,x5i,5); LDROW(x4r,x4i,4); LDROW(x1r,x1i,1);
  V a0r=VADD(x0r,x3r), a0i=VADD(x0i,x3i), b0r=VSUB(x0r,x3r), b0i=VSUB(x0i,x3i);
  V a1r=VADD(x2r,x5r), a1i=VADD(x2i,x5i), b1r=VSUB(x2r,x5r), b1i=VSUB(x2i,x5i);
  V a2r=VADD(x4r,x1r), a2i=VADD(x4i,x1i), b2r=VSUB(x4r,x1r), b2i=VSUB(x4i,x1i);
  V A0r,A0i,A1r,A1i,A2r,A2i, B0r,B0i,B1r,B1i,B2r,B2i;
  F3K(a0r,a0i,a1r,a1i,a2r,a2i, A0r,A0i,A1r,A1i,A2r,A2i);
  F3K(b0r,b0i,b1r,b1i,b2r,b2i, B0r,B0i,B1r,B1i,B2r,B2i);
  STROWM(0,A0r,A0i); STROWM(4,A1r,A1i); STROWM(2,A2r,A2i);
  STROWM(3,B0r,B0i); STROWM(1,B1r,B1i); STROWM(5,B2r,B2i);
}
AI void cod8m(const double* sre, const double* sim, long ss, double* dre, double* dim, long ds,
              const double* cre, const double* cim){
  LDROW(x0r,x0i,0); LDROW(x1r,x1i,1); LDROW(x2r,x2i,2); LDROW(x3r,x3i,3);
  LDROW(x4r,x4i,4); LDROW(x5r,x5i,5); LDROW(x6r,x6i,6); LDROW(x7r,x7i,7);
  V X0r,X0i,X1r,X1i,X2r,X2i,X3r,X3i,X4r,X4i,X5r,X5i,X6r,X6i,X7r,X7i;
  F8K(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,
      X0r,X0i,X1r,X1i,X2r,X2i,X3r,X3i,X4r,X4i,X5r,X5i,X6r,X6i,X7r,X7i);
  STROWM(0,X0r,X0i); STROWM(1,X1r,X1i); STROWM(2,X2r,X2i); STROWM(3,X3r,X3i);
  STROWM(4,X4r,X4i); STROWM(5,X5r,X5i); STROWM(6,X6r,X6i); STROWM(7,X7r,X7i);
}
#define LL0 13
AI void cod13m(const double* sre, const double* sim, long ss, double* dre, double* dim, long ds,
               const double* cre, const double* cim){
  V x0r = VLDU(sre), x0i = VLDU(sim);
  V t0r = x0r, t0i = x0i;
  DS_ACC(6, DS13C, DS13S, 6, 1,2,3,4,5,6, a_, 1)
#undef STROW
#define STROW(k,vr,vi) STROWM(k,vr,vi)
  STROW(0, t0r, t0i);
  DS_OUT(6, 1,2,3,4,5,6, a_)
#undef STROW
#define STROW(k,vr,vi) { VSTU(dre + (long)(k)*ds, vr); VSTU(dim + (long)(k)*ds, vi); }
}
#undef LL0

#define LL0 17
AI void cod17m(const double* sre, const double* sim, long ss, double* dre, double* dim, long ds,
               const double* cre, const double* cim){
  V x0r = VLDU(sre), x0i = VLDU(sim);
  V t0r = x0r, t0i = x0i;
  DS_ACC(8, DS17C, DS17S, 4, 1,2,3,4,0,0, a_, 1)
  DS_ACC(8, DS17C, DS17S, 4, 5,6,7,8,0,0, b_, 0)
#undef STROW
#define STROW(k,vr,vi) STROWM(k,vr,vi)
  STROW(0, t0r, t0i);
  DS_OUT(4, 1,2,3,4,0,0, a_)
  DS_OUT(4, 5,6,7,8,0,0, b_)
#undef STROW
#define STROW(k,vr,vi) { VSTU(dre + (long)(k)*ds, vr); VSTU(dim + (long)(k)*ds, vi); }
}
#undef LL0

#define DEF_OCT_IP(SFX, LL, CODELET, CODELETM)                                  \
static void opass1_##SFX(const double* src, double* dst){                        \
  (void)src;                                                                    \
  const long OSLAB = (long)LL*LL*16 + 8;                                        \
  double* a = dst;                                                              \
  for (long q = 0; q < (long)LL*LL; q++){                                       \
    long o = q*16;                                                              \
    CODELET(a+o, a+o+8, OSLAB, a+o, a+o+8, OSLAB);                              \
  }                                                                             \
}                                                                               \
static void opass23f_##SFX(double* restrict a, const double* restrict cc){      \
  const long OSLAB = (long)LL*LL*16 + 8;                                        \
  for (int x = 0; x < LL; x++){                                                 \
    double* sb = a + (long)x*OSLAB;                                             \
    const double* cb = cc + (long)x*OSLAB;                                      \
    for (int z = 0; z < LL; z++){                                               \
      long o = (long)z*16;                                                      \
      CODELET(sb+o, sb+o+8, (long)LL*16, sb+o, sb+o+8, (long)LL*16);            \
    }                                                                           \
    for (int y = 0; y < LL; y++){                                               \
      long o = (long)y*LL*16;                                                   \
      CODELETM(sb+o, sb+o+8, 16, sb+o, sb+o+8, 16, cb+o, cb+o+8);               \
    }                                                                           \
  }                                                                             \
}                                                                               \
static void opass23s_##SFX(double* restrict a, const double* restrict cc){      \
  const long OSLAB = (long)LL*LL*16 + 8;                                        \
  for (int x = 0; x < LL; x++){                                                 \
    double* sb = a + (long)x*OSLAB;                                             \
    for (int z = 0; z < LL; z++){                                               \
      long o = (long)z*16;                                                      \
      CODELET(sb+o, sb+o+8, (long)LL*16, sb+o, sb+o+8, (long)LL*16);            \
    }                                                                           \
  }                                                                             \
  for (int x = 0; x < LL; x++){                                                 \
    double* sb = a + (long)x*OSLAB;                                             \
    const double* cb = cc + (long)x*OSLAB;                                      \
    for (int y = 0; y < LL; y++){                                               \
      long o = (long)y*LL*16;                                                   \
      CODELETM(sb+o, sb+o+8, 16, sb+o, sb+o+8, 16, cb+o, cb+o+8);               \
    }                                                                           \
  }                                                                             \
}

DEF_OCT_IP(o6, 6, cod6, cod6m)
DEF_OCT_IP(o8, 8, cod8, cod8m)
DEF_OCT_IP(o13, 13, cod13, cod13m)
DEF_OCT_IP(o17, 17, cod17, cod17m)

typedef void (*opass12_f)(const double*, double*);
typedef void (*opass23_f)(double*, const double*);
typedef struct { opass12_f p1; opass23_f p23; } opasses_t;
static opasses_t g_opasses[65];

static double *OCT_A, *OCT_B, *OCT_C;   // 13^3*16 doubles each

// 8 interleaved-complex volumes -> octet layout
static void conv_in_oct(int L, const double* const* srcs, int nval, double* dst){
  long n2 = (long)L*L;              // positions per slab
  long OSLAB = n2*16 + 8;
  for (int x = 0; x < L; x++){
    long base = (long)x*n2;         // position index base
    for (long pb = 0; pb < n2; pb += 8){
      int rem = (int)(n2 - pb); if (rem > 8) rem = 8;
      __mmask8 mlo = (__mmask8)((rem>=4)?0xff:((1u<<(2*rem))-1));
      __mmask8 mhi = (__mmask8)((rem>4)?((1u<<(2*(rem-4)))-1):0);
      V re[8], im[8];
      for (int v = 0; v < 8; v++){
        const double* s = srcs[v < nval ? v : nval-1] + 2*(base + pb);
        V a = _mm512_maskz_loadu_pd(mlo, s), b = _mm512_maskz_loadu_pd(mhi, s+8);
        re[v] = _mm512_permutex2var_pd(a, IDXE, b);
        im[v] = _mm512_permutex2var_pd(a, IDXO, b);
      }
      TRANSP8(re[0],re[1],re[2],re[3],re[4],re[5],re[6],re[7]);
      TRANSP8(im[0],im[1],im[2],im[3],im[4],im[5],im[6],im[7]);
      double* d = dst + (long)x*OSLAB + pb*16;
      for (int j = 0; j < rem; j++){ VST(d + j*16, re[j]); VST(d + j*16 + 8, im[j]); }
    }
  }
}
// octet layout -> interleaved complex volumes (nval lanes)
static void conv_out_oct(int L, const double* src, double* const* dsts, int nval){
  long n2 = (long)L*L;
  long OSLAB = n2*16 + 8;
  for (int x = 0; x < L; x++){
    long base = (long)x*n2;
    for (long pb = 0; pb < n2; pb += 8){
      int rem = (int)(n2 - pb); if (rem > 8) rem = 8;
      const double* s = src + (long)x*OSLAB + pb*16;
      V re[8], im[8];
      for (int j = 0; j < 8; j++){ re[j] = (j < rem) ? VLD(s + j*16) : VSET1(0.0); im[j] = (j < rem) ? VLD(s + j*16 + 8) : VSET1(0.0); }
      TRANSP8(re[0],re[1],re[2],re[3],re[4],re[5],re[6],re[7]);
      TRANSP8(im[0],im[1],im[2],im[3],im[4],im[5],im[6],im[7]);
      __mmask8 mlo = (__mmask8)((rem>=4)?0xff:((1u<<(2*rem))-1));
      __mmask8 mhi = (__mmask8)((rem>4)?((1u<<(2*(rem-4)))-1):0);
      for (int v = 0; v < nval; v++){
        double* d = dsts[v] + 2*(base + pb);
        _mm512_mask_storeu_pd(d,   mlo, _mm512_permutex2var_pd(re[v], ILA, im[v]));
        _mm512_mask_storeu_pd(d+8, mhi, _mm512_permutex2var_pd(re[v], ILB, im[v]));
      }
    }
  }
}

static void run_octet(int L, int bstart, int bcount, int m, const double* x0, const double* c,
                      double* one, double* fin){
  long n2 = 2L*L*L*L;
  opasses_t* ps = &g_opasses[L];
  for (int g = bstart; g < bstart + bcount; g += 8){
    int nval = bstart + bcount - g; if (nval > 8) nval = 8;
    const double* srcs[8]; double* douts[8]; double* fouts[8];
    for (int v = 0; v < 8; v++){
      int b = g + (v < nval ? v : nval-1);
      srcs[v] = x0 + (long)b*n2;
      douts[v] = one + (long)b*n2;
      fouts[v] = fin + (long)b*n2;
    }
    conv_in_oct(L, srcs, nval, OCT_A);
    for (int v = 0; v < 8; v++) srcs[v] = c + (long)(g + (v < nval ? v : nval-1))*n2;
    conv_in_oct(L, srcs, nval, OCT_C);
    for (int k = 1; k <= m; k++){
      ps->p1(OCT_A, OCT_A);
      ps->p23(OCT_A, OCT_C);
      if (k == 1) conv_out_oct(L, OCT_A, douts, nval);
    }
    conv_out_oct(L, OCT_A, fouts, nval);
  }
}

// raw-state variants: state between steps holds z = FFT3(x)+c; the map is
// applied when pass1 of the next step loads rows (BW-bound phase, FMA idle).
#define DEF_RAW(SFX, LL, PP, CODELET)                                           \
static void pass1m_##SFX(const double* src, double* dst){                       \
  const long PLANE = (long)PP*8, SLAB = 2L*PP*PP + 8;                           \
  for (int p = 0; p < PP/8; p++)                                                \
    for (int k = 0; k < LL; k++){                                               \
      long o = (long)p*PP*16 + k*8;                                             \
      if (PP >= 64 && g_pf){                                                    \
        const double* pf = src + o + 8;                                         \
        double* pfd = dst + o + 8;                                              \
        for (int r = 0; r < LL; r++){                                           \
          _mm_prefetch((const char*)(pf + (long)r*SLAB), _MM_HINT_T0);          \
          _mm_prefetch((const char*)(pf + PLANE + (long)r*SLAB), _MM_HINT_T0);  \
          __builtin_prefetch(pfd + (long)r*SLAB, 1, 3);                         \
          __builtin_prefetch(pfd + PLANE + (long)r*SLAB, 1, 3);                 \
        }                                                                       \
      }                                                                         \
      for (int r = 0; r < LL; r++){                                             \
        V re = VLDU(src + o + (long)r*SLAB), im = VLDU(src + PLANE + o + (long)r*SLAB); \
        map8nc(&re, &im, (r & 3) < g_mapmix);                                   \
        VST(COLRE + r*8, re); VST(COLIM + r*8, im);                             \
      }                                                                         \
      CODELET(COLRE, COLIM, 8, dst+o, dst+PLANE+o, SLAB);                       \
    }                                                                           \
}                                                                               \
static void fused23r_##SFX(const double* p1out, double* mid, double* dst, const double* restrict cvol){ \
  const long PLANE = (long)PP*8, SLAB = 2L*PP*PP + 8;                           \
  (void)mid;                                                                    \
  for (int x = 0; x < LL; x++){                                                 \
    { const double* sb = p1out + (long)x*SLAB; double* db = MIDSLAB;            \
      for (int p = 0; p < PP/8; p++){                                           \
        long o = (long)p*PP*16;                                                 \
        CODELET(sb+o, sb+PLANE+o, 8, db+o, db+PLANE+o, 8);                      \
      }                                                                         \
    }                                                                           \
    const double* sb = MIDSLAB;                                                 \
    double* db = dst + (long)x*SLAB;                                            \
    const double* cc = cvol + (long)x*SLAB;                                     \
    for (int t = 0; t < PP/8; t++){                                             \
      for (int p = 0; p < PP/8; p++){                                           \
        const double* q = sb + (long)p*PP*16 + (long)t*64;                      \
        V r0=VLD(q),r1=VLD(q+8),r2=VLD(q+16),r3=VLD(q+24),r4=VLD(q+32),r5=VLD(q+40),r6=VLD(q+48),r7=VLD(q+56); \
        TRANSP8(r0,r1,r2,r3,r4,r5,r6,r7);                                       \
        VST(SCR1RE+(8*p+0)*8,r0); VST(SCR1RE+(8*p+1)*8,r1); VST(SCR1RE+(8*p+2)*8,r2); VST(SCR1RE+(8*p+3)*8,r3); \
        VST(SCR1RE+(8*p+4)*8,r4); VST(SCR1RE+(8*p+5)*8,r5); VST(SCR1RE+(8*p+6)*8,r6); VST(SCR1RE+(8*p+7)*8,r7); \
        const double* qq = q + (long)PP*8;                                      \
        r0=VLD(qq);r1=VLD(qq+8);r2=VLD(qq+16);r3=VLD(qq+24);r4=VLD(qq+32);r5=VLD(qq+40);r6=VLD(qq+48);r7=VLD(qq+56); \
        TRANSP8(r0,r1,r2,r3,r4,r5,r6,r7);                                       \
        VST(SCR1IM+(8*p+0)*8,r0); VST(SCR1IM+(8*p+1)*8,r1); VST(SCR1IM+(8*p+2)*8,r2); VST(SCR1IM+(8*p+3)*8,r3); \
        VST(SCR1IM+(8*p+4)*8,r4); VST(SCR1IM+(8*p+5)*8,r5); VST(SCR1IM+(8*p+6)*8,r6); VST(SCR1IM+(8*p+7)*8,r7); \
      }                                                                         \
      CODELET(SCR1RE, SCR1IM, 8, SCR2RE, SCR2IM, 8);                            \
      {                                                                         \
        long o = (long)t*PP*16;                                                 \
        for (int k = 0; k < LL; k++){                                           \
          V re = VLD(SCR2RE + k*8), im = VLD(SCR2IM + k*8);                     \
          V cre = VLD(cc + o + k*8), cim = VLD(cc + PLANE + o + k*8);           \
          VST(db + o + k*8, VADD(re, cre)); VST(db + PLANE + o + k*8, VADD(im, cim)); \
        }                                                                       \
      }                                                                         \
    }                                                                           \
  }                                                                             \
}
DEF_RAW(s36, 36, 40, cod36)
DEF_RAW(s45, 45, 48, cod45)
DEF_RAW(s64, 64, 64, cod64)

// in-place pass1 for multi-stage codelets: rows staged through a column scratch
#define DEF_P1IP(SFX, LL, PP, CODELET)                                          \
static void pass1ip_##SFX(double* a){                                           \
  const long PLANE = (long)PP*8, SLAB = 2L*PP*PP + 8;                           \
  for (int p = 0; p < PP/8; p++)                                                \
    for (int k = 0; k < LL; k++){                                               \
      long o = (long)p*PP*16 + k*8;                                             \
      if (PP >= 64 && g_pf){                                                    \
        const double* pf = a + o + 8;                                           \
        for (int r = 0; r < LL; r++){                                           \
          __builtin_prefetch(pf + (long)r*SLAB, 1, 3);                          \
          __builtin_prefetch(pf + PLANE + (long)r*SLAB, 1, 3);                  \
        }                                                                       \
      }                                                                         \
      for (int r = 0; r < LL; r++){                                             \
        VST(COLRE + r*8, VLDU(a + o + (long)r*SLAB));                           \
        VST(COLIM + r*8, VLDU(a + PLANE + o + (long)r*SLAB));                   \
      }                                                                         \
      CODELET(COLRE, COLIM, 8, a+o, a+PLANE+o, SLAB);                           \
    }                                                                           \
}
#define DEF_P1IPM(SFX, LL, PP, CODELET)                                         \
static void pass1ipm_##SFX(double* a){                                          \
  const long PLANE = (long)PP*8, SLAB = 2L*PP*PP + 8;                           \
  for (int p = 0; p < PP/8; p++)                                                \
    for (int k = 0; k < LL; k++){                                               \
      long o = (long)p*PP*16 + k*8;                                             \
      if (PP >= 64 && g_pf){                                                    \
        const double* pf = a + o + 8;                                           \
        for (int r = 0; r < LL; r++){                                           \
          __builtin_prefetch(pf + (long)r*SLAB, 1, 3);                          \
          __builtin_prefetch(pf + PLANE + (long)r*SLAB, 1, 3);                  \
        }                                                                       \
      }                                                                         \
      for (int r = 0; r < LL; r++){                                             \
        V re = VLDU(a + o + (long)r*SLAB), im = VLDU(a + PLANE + o + (long)r*SLAB); \
        map8nc(&re, &im, (r & 3) < g_mapmix);                                   \
        VST(COLRE + r*8, re); VST(COLIM + r*8, im);                             \
      }                                                                         \
      CODELET(COLRE, COLIM, 8, a+o, a+PLANE+o, SLAB);                           \
    }                                                                           \
}
DEF_P1IPM(s36, 36, 40, cod36)
DEF_P1IPM(s45, 45, 48, cod45)
DEF_P1IPM(s64, 64, 64, cod64)
static void* g_p1ipm_tbl_init_marker;
DEF_P1IP(s23, 23, 24, cod23)
DEF_P1IP(s36, 36, 40, cod36)
DEF_P1IP(s45, 45, 48, cod45)
DEF_P1IP(s64, 64, 64, cod64)
typedef void (*pass1ip_f)(double*);
static pass1ip_f g_p1ip[65];
static pass1ip_f g_p1ipm[65];

// fused23 for L=64 with cod64c (stage-B-fused +c/map), no separate map loop
static void fused23c_s64(const double* p1out, double* mid, double* dst, const double* restrict cvol){
  enum { PP = 64 };
  const long PLANE = (long)PP*8, SLAB = 2L*PP*PP + 8;
  (void)mid;
  for (int x = 0; x < 64; x++){
    { const double* sb = p1out + (long)x*SLAB; double* db = MIDSLAB;
      for (int p = 0; p < PP/8; p++){
        long o = (long)p*PP*16;
        cod64(sb+o, sb+PLANE+o, 8, db+o, db+PLANE+o, 8);
      }
    }
    const double* sb = MIDSLAB;
    double* db = dst + (long)x*SLAB;
    const double* cc = cvol + (long)x*SLAB;
    for (int t = 0; t < PP/8; t++){
      for (int p = 0; p < PP/8; p++){
        const double* q = sb + (long)p*PP*16 + (long)t*64;
        V r0=VLD(q),r1=VLD(q+8),r2=VLD(q+16),r3=VLD(q+24),r4=VLD(q+32),r5=VLD(q+40),r6=VLD(q+48),r7=VLD(q+56);
        TRANSP8(r0,r1,r2,r3,r4,r5,r6,r7);
        VST(SCR1RE+(8*p+0)*8,r0); VST(SCR1RE+(8*p+1)*8,r1); VST(SCR1RE+(8*p+2)*8,r2); VST(SCR1RE+(8*p+3)*8,r3);
        VST(SCR1RE+(8*p+4)*8,r4); VST(SCR1RE+(8*p+5)*8,r5); VST(SCR1RE+(8*p+6)*8,r6); VST(SCR1RE+(8*p+7)*8,r7);
        const double* qq = q + (long)PP*8;
        r0=VLD(qq);r1=VLD(qq+8);r2=VLD(qq+16);r3=VLD(qq+24);r4=VLD(qq+32);r5=VLD(qq+40);r6=VLD(qq+48);r7=VLD(qq+56);
        TRANSP8(r0,r1,r2,r3,r4,r5,r6,r7);
        VST(SCR1IM+(8*p+0)*8,r0); VST(SCR1IM+(8*p+1)*8,r1); VST(SCR1IM+(8*p+2)*8,r2); VST(SCR1IM+(8*p+3)*8,r3);
        VST(SCR1IM+(8*p+4)*8,r4); VST(SCR1IM+(8*p+5)*8,r5); VST(SCR1IM+(8*p+6)*8,r6); VST(SCR1IM+(8*p+7)*8,r7);
      }
      long o = (long)t*PP*16;
      cod64c(SCR1RE, SCR1IM, 8, SCR2RE, SCR2IM, db+o, db+PLANE+o, cc+o, cc+PLANE+o);
    }
  }
}

typedef void (*pass12_f)(const double*, double*);
typedef void (*fused23_f)(const double*, double*, double*, const double*);
typedef struct { pass12_f p1, p1m; fused23_f p23, p23r; } passes_t;
static passes_t g_passes[65];

// ========================= conversions =======================================

// interleaved complex row of L values -> split re/im rows (cols 0..L-1)
AI void row_deint(const double* src, double* dre, double* dim, int L){
  int c = 0;
  for (; c + 8 <= L; c += 8){
    V a = VLDU(src + 2*c), b = VLDU(src + 2*c + 8);
    VSTU(dre + c, _mm512_permutex2var_pd(a, IDXE, b));
    VSTU(dim + c, _mm512_permutex2var_pd(a, IDXO, b));
  }
  int rem = L - c;
  if (rem){
    __mmask8 mlo = (__mmask8)((rem>=4)?0xff:((1u<<(2*rem))-1));
    __mmask8 mhi = (__mmask8)((rem>4)?((1u<<(2*(rem-4)))-1):0);
    V a = _mm512_maskz_loadu_pd(mlo, src + 2*c);
    V b = _mm512_maskz_loadu_pd(mhi, src + 2*c + 8);
    __mmask8 mm = (__mmask8)((1u<<rem)-1);
    _mm512_mask_storeu_pd(dre + c, mm, _mm512_permutex2var_pd(a, IDXE, b));
    _mm512_mask_storeu_pd(dim + c, mm, _mm512_permutex2var_pd(a, IDXO, b));
  }
}
AI void row_int(const double* sre, const double* sim, double* dst, int L){
  int c = 0;
  for (; c + 8 <= L; c += 8){
    V re = VLDU(sre + c), im = VLDU(sim + c);
    VSTU(dst + 2*c,     _mm512_permutex2var_pd(re, ILA, im));
    VSTU(dst + 2*c + 8, _mm512_permutex2var_pd(re, ILB, im));
  }
  int rem = L - c;
  if (rem){
    V re = VLDU(sre + c), im = VLDU(sim + c);   // over-read ok (pads exist)
    __mmask8 mlo = (__mmask8)((rem>=4)?0xff:((1u<<(2*rem))-1));
    __mmask8 mhi = (__mmask8)((rem>4)?((1u<<(2*(rem-4)))-1):0);
    _mm512_mask_storeu_pd(dst + 2*c,     mlo, _mm512_permutex2var_pd(re, ILA, im));
    _mm512_mask_storeu_pd(dst + 2*c + 8, mhi, _mm512_permutex2var_pd(re, ILB, im));
  }
}

// x0 (interleaved natural) -> layout0 volume (panel layout)
static void conv_in(eng_t* e, const double* src, double* dst){
  int L = e->L, P = e->P; long PLANE = e->PLANE, SLAB = e->SLAB;
  for (int x = 0; x < L; x++)
    for (int y = 0; y < L; y++){
      const double* s = src + 2L*((long)(x*L + y)*L);
      double* dre = dst + (long)x*SLAB + (long)y*8;
      double* dim = dre + PLANE;
      int c = 0;
      for (; c + 8 <= L; c += 8){
        V a = VLDU(s + 2*c), b2 = VLDU(s + 2*c + 8);
        VSTU(dre + (long)(c>>3)*P*16, _mm512_permutex2var_pd(a, IDXE, b2));
        VSTU(dim + (long)(c>>3)*P*16, _mm512_permutex2var_pd(a, IDXO, b2));
      }
      int rem = L - c;
      if (rem){
        __mmask8 mlo = (__mmask8)((rem>=4)?0xff:((1u<<(2*rem))-1));
        __mmask8 mhi = (__mmask8)((rem>4)?((1u<<(2*(rem-4)))-1):0);
        V a = _mm512_maskz_loadu_pd(mlo, s + 2*c);
        V b2 = _mm512_maskz_loadu_pd(mhi, s + 2*c + 8);
        __mmask8 mm = (__mmask8)((1u<<rem)-1);
        _mm512_mask_storeu_pd(dre + (long)(c>>3)*P*16, mm, _mm512_permutex2var_pd(a, IDXE, b2));
        _mm512_mask_storeu_pd(dim + (long)(c>>3)*P*16, mm, _mm512_permutex2var_pd(a, IDXO, b2));
      }
    }
}
// layout0 volume -> transposed-plane copy (layout1), panel layout both sides
static void conv_T(eng_t* e, const double* src, double* dst){
  int L = e->L, P = e->P; long PLANE = e->PLANE, SLAB = e->SLAB;
  for (int x = 0; x < L; x++){
    for (int comp = 0; comp < 2; comp++){
      const double* sp = src + (long)x*SLAB + comp*PLANE;
      double* dp = dst + (long)x*SLAB + comp*PLANE;
      for (int rb = 0; rb < P; rb += 8)
        for (int cb = 0; cb < P; cb += 8){
          const double* q = sp + (long)(cb>>3)*P*16 + (long)rb*8;
          V r0=VLD(q),r1=VLD(q+8),r2=VLD(q+16),r3=VLD(q+24),r4=VLD(q+32),r5=VLD(q+40),r6=VLD(q+48),r7=VLD(q+56);
          TRANSP8(r0,r1,r2,r3,r4,r5,r6,r7);
          double* w = dp + (long)(rb>>3)*P*16 + (long)cb*8;
          VST(w,r0); VST(w+8,r1); VST(w+16,r2); VST(w+24,r3); VST(w+32,r4); VST(w+40,r5); VST(w+48,r6); VST(w+56,r7);
        }
    }
  }
}
// state volume (parity, raw z) -> apply map, then interleave to natural
static void conv_out_mapped(eng_t* e, const double* A, int parity, double* dst){
  int L = e->L, P = e->P; long PLANE = e->PLANE, SLAB = e->SLAB;
  if (parity == 0){
    for (int x = 0; x < L; x++)
      for (int y = 0; y < L; y++){
        const double* sre = A + (long)x*SLAB + (long)y*8;
        const double* sim = sre + PLANE;
        double* d = dst + 2L*((long)(x*L+y)*L);
        int c = 0;
        for (; c + 8 <= L; c += 8){
          V re = VLDU(sre + (long)(c>>3)*P*16), im = VLDU(sim + (long)(c>>3)*P*16);
          map8nc(&re, &im, 1);
          VSTU(d + 2*c,     _mm512_permutex2var_pd(re, ILA, im));
          VSTU(d + 2*c + 8, _mm512_permutex2var_pd(re, ILB, im));
        }
        int rem = L - c;
        if (rem){
          V re = VLDU(sre + (long)(c>>3)*P*16), im = VLDU(sim + (long)(c>>3)*P*16);
          map8nc(&re, &im, 1);
          __mmask8 mlo = (__mmask8)((rem>=4)?0xff:((1u<<(2*rem))-1));
          __mmask8 mhi = (__mmask8)((rem>4)?((1u<<(2*(rem-4)))-1):0);
          _mm512_mask_storeu_pd(d + 2*c,     mlo, _mm512_permutex2var_pd(re, ILA, im));
          _mm512_mask_storeu_pd(d + 2*c + 8, mhi, _mm512_permutex2var_pd(re, ILB, im));
        }
      }
  } else {
    for (int x = 0; x < L; x++){
      const double* sre = A + (long)x*SLAB;
      const double* sim = sre + PLANE;
      for (int yb = 0; yb < L; yb += 8){
        for (int zb = 0; zb < P; zb += 8){
          if (zb >= L) break;
          const double* q = sre + (long)(yb>>3)*P*16 + (long)zb*8;
          V a0=VLD(q),a1=VLD(q+8),a2=VLD(q+16),a3=VLD(q+24),a4=VLD(q+32),a5=VLD(q+40),a6=VLD(q+48),a7=VLD(q+56);
          TRANSP8(a0,a1,a2,a3,a4,a5,a6,a7);
          const double* qi = sim + (long)(yb>>3)*P*16 + (long)zb*8;
          V b0=VLD(qi),b1=VLD(qi+8),b2=VLD(qi+16),b3=VLD(qi+24),b4=VLD(qi+32),b5=VLD(qi+40),b6=VLD(qi+48),b7=VLD(qi+56);
          TRANSP8(b0,b1,b2,b3,b4,b5,b6,b7);
          int zrem = L - zb; if (zrem > 8) zrem = 8;
          __mmask8 mlo = (__mmask8)((zrem>=4)?0xff:((1u<<(2*zrem))-1));
          __mmask8 mhi = (__mmask8)((zrem>4)?((1u<<(2*(zrem-4)))-1):0);
          V ar[8] = {a0,a1,a2,a3,a4,a5,a6,a7}, br[8] = {b0,b1,b2,b3,b4,b5,b6,b7};
          int ymax = L - yb; if (ymax > 8) ymax = 8;
          for (int v = 0; v < ymax; v++){
            V re = ar[v], im = br[v];
            map8nc(&re, &im, 1);
            double* w = dst + 2L*(((long)x*L + (yb+v))*L + zb);
            _mm512_mask_storeu_pd(w,     mlo, _mm512_permutex2var_pd(re, ILA, im));
            _mm512_mask_storeu_pd(w + 8, mhi, _mm512_permutex2var_pd(re, ILB, im));
          }
        }
      }
    }
  }
}
// state volume (parity) -> interleaved natural
static void conv_out(eng_t* e, const double* A, int parity, double* dst){
  int L = e->L, P = e->P; long PLANE = e->PLANE, SLAB = e->SLAB;
  if (parity == 0){
    for (int x = 0; x < L; x++)
      for (int y = 0; y < L; y++){
        const double* sre = A + (long)x*SLAB + (long)y*8;
        const double* sim = sre + PLANE;
        double* d = dst + 2L*((long)(x*L+y)*L);
        int c = 0;
        for (; c + 8 <= L; c += 8){
          V re = VLDU(sre + (long)(c>>3)*P*16), im = VLDU(sim + (long)(c>>3)*P*16);
          VSTU(d + 2*c,     _mm512_permutex2var_pd(re, ILA, im));
          VSTU(d + 2*c + 8, _mm512_permutex2var_pd(re, ILB, im));
        }
        int rem = L - c;
        if (rem){
          V re = VLDU(sre + (long)(c>>3)*P*16), im = VLDU(sim + (long)(c>>3)*P*16);
          __mmask8 mlo = (__mmask8)((rem>=4)?0xff:((1u<<(2*rem))-1));
          __mmask8 mhi = (__mmask8)((rem>4)?((1u<<(2*(rem-4)))-1):0);
          _mm512_mask_storeu_pd(d + 2*c,     mlo, _mm512_permutex2var_pd(re, ILA, im));
          _mm512_mask_storeu_pd(d + 2*c + 8, mhi, _mm512_permutex2var_pd(re, ILB, im));
        }
      }
  } else {
    // plane rows = z, cols = y; need dst[x][y][z]
    for (int x = 0; x < L; x++){
      const double* sre = A + (long)x*SLAB;
      const double* sim = sre + PLANE;
      for (int yb = 0; yb < L; yb += 8){
        for (int zb = 0; zb < P; zb += 8){
          if (zb >= L) break;
          const double* q = sre + (long)(yb>>3)*P*16 + (long)zb*8;
          V a0=VLD(q),a1=VLD(q+8),a2=VLD(q+16),a3=VLD(q+24),a4=VLD(q+32),a5=VLD(q+40),a6=VLD(q+48),a7=VLD(q+56);
          TRANSP8(a0,a1,a2,a3,a4,a5,a6,a7);
          const double* qi = sim + (long)(yb>>3)*P*16 + (long)zb*8;
          V b0=VLD(qi),b1=VLD(qi+8),b2=VLD(qi+16),b3=VLD(qi+24),b4=VLD(qi+32),b5=VLD(qi+40),b6=VLD(qi+48),b7=VLD(qi+56);
          TRANSP8(b0,b1,b2,b3,b4,b5,b6,b7);
          int zrem = L - zb; if (zrem > 8) zrem = 8;
          __mmask8 mlo = (__mmask8)((zrem>=4)?0xff:((1u<<(2*zrem))-1));
          __mmask8 mhi = (__mmask8)((zrem>4)?((1u<<(2*(zrem-4)))-1):0);
          V ar[8] = {a0,a1,a2,a3,a4,a5,a6,a7}, br[8] = {b0,b1,b2,b3,b4,b5,b6,b7};
          int ymax = L - yb; if (ymax > 8) ymax = 8;
          for (int v = 0; v < ymax; v++){
            double* w = dst + 2L*(((long)x*L + (yb+v))*L + zb);
            _mm512_mask_storeu_pd(w,     mlo, _mm512_permutex2var_pd(ar[v], ILA, br[v]));
            _mm512_mask_storeu_pd(w + 8, mhi, _mm512_permutex2var_pd(ar[v], ILB, br[v]));
          }
        }
      }
    }
  }
}

// ========================= top-level ==========================================
static int g_use_generic = 0;
static void run_eng_range(eng_t* e, int bstart, int bcount, int m, const double* x0, const double* c,
                    double* one, double* fin){
  long n2 = 2L*e->L*e->L*e->L;
  passes_t* ps = &g_passes[e->L];
  for (int b = bstart; b < bstart + bcount; b++){
    conv_in(e, x0 + b*n2, e->A);
    conv_in(e, c + b*n2, e->c0);
    conv_T(e, e->c0, e->c1);
    int s = 0;
    for (int k = 1; k <= m; k++){
      if (__builtin_expect(g_use_generic, 0)){
        gpass1(e, e->A, e->B);
        gpass2(e, e->B, e->A);
        gpass3(e, e->A, e->B, s ? e->c0 : e->c1);
      } else if (e->L <= 17 && g_inplace){
        ps->p1(e->A, e->A);
        ps->p23(e->A, e->B, e->A, s ? e->c0 : e->c1);
        s ^= 1;
        if (k == 1) conv_out(e, e->A, s, one + b*n2);
        continue;
      } else if (g_p1ipm[e->L] && g_ip2 && g_raw){
        if (k == 1) g_p1ip[e->L](e->A); else g_p1ipm[e->L](e->A);
        ps->p23r(e->A, e->B, e->A, s ? e->c0 : e->c1);
        s ^= 1;
        if (k == 1) conv_out_mapped(e, e->A, s, one + b*n2);
        continue;
      } else if (g_p1ip[e->L] && g_ip2){
        g_p1ip[e->L](e->A);
        ps->p23(e->A, e->B, e->A, s ? e->c0 : e->c1);
        s ^= 1;
        if (k == 1) conv_out(e, e->A, s, one + b*n2);
        continue;
      } else if (ps->p1m && g_raw){
        // raw-state mode: state holds z = FFT3(x)+c; map applied on next load
        if (k == 1) ps->p1(e->A, e->B); else ps->p1m(e->A, e->B);
        ps->p23r(e->B, e->A, e->B, s ? e->c0 : e->c1);
        double* t = e->A; e->A = e->B; e->B = t;
        s ^= 1;
        if (k == 1) conv_out_mapped(e, e->A, s, one + b*n2);
        continue;
      } else {
        ps->p1(e->A, e->B);
        ps->p23(e->B, e->A, e->B, s ? e->c0 : e->c1);
      }
      double* t = e->A; e->A = e->B; e->B = t;
      s ^= 1;
      if (k == 1) conv_out(e, e->A, s, one + b*n2);
    }
    if ((ps->p1m || (g_p1ipm[e->L] && g_ip2)) && g_raw) conv_out_mapped(e, e->A, s, fin + b*n2);
    else conv_out(e, e->A, s, fin + b*n2);
  }
}
static void run_eng(eng_t* e, int B, int m, const double* x0, const double* c,
                    double* one, double* fin){
  run_eng_range(e, 0, B, m, x0, c, one, fin);
}

// ========================= init & API ========================================
static const int SIZES[8] = {6,8,13,17,23,36,45,64};
static const int PADS[8]  = {8,8,16,24,24,40,48,64};

#define PIL 3.14159265358979323846264338327950288L
static void fill_consts(void){
  K_C2 = (double)(1.0L/sqrtl(2.0L)); K_NC2 = -K_C2;
  K_S3 = (double)(sqrtl(3.0L)/2.0L);
  K_C51 = (double)cosl(2.0L*PIL/5.0L); K_S51 = (double)sinl(2.0L*PIL/5.0L);
  K_C52 = (double)cosl(4.0L*PIL/5.0L); K_S52 = (double)sinl(4.0L*PIL/5.0L);
  K_C91 = (double)cosl(2.0L*PIL/9.0L); K_S91 = (double)sinl(2.0L*PIL/9.0L);
  K_C92 = (double)cosl(4.0L*PIL/9.0L); K_S92 = (double)sinl(4.0L*PIL/9.0L);
  K_C94 = (double)cosl(8.0L*PIL/9.0L); K_S94 = (double)sinl(8.0L*PIL/9.0L);
  for (int j2 = 0; j2 < 8; j2++)
    for (int k1 = 0; k1 < 8; k1++){
      long double a = 2.0L*PIL*((j2*k1) % 64)/64.0L;
      TW64C[j2][k1] = (double)cosl(a); TW64S[j2][k1] = (double)sinl(a);
    }
  for (int j = 1; j <= 6; j++) for (int k = 1; k <= 6; k++){
    long double a = 2.0L*PIL*((j*k)%13)/13.0L;
    DS13C[j][k] = (double)cosl(a); DS13S[j][k] = (double)sinl(a);
  }
  for (int j = 1; j <= 8; j++) for (int k = 1; k <= 8; k++){
    long double a = 2.0L*PIL*((j*k)%17)/17.0L;
    DS17C[j][k] = (double)cosl(a); DS17S[j][k] = (double)sinl(a);
  }
  for (int j = 1; j <= 11; j++) for (int k = 1; k <= 11; k++){
    long double a = 2.0L*PIL*((j*k)%23)/23.0L;
    DS23C[j][k] = (double)cosl(a); DS23S[j][k] = (double)sinl(a);
  }
  g_opasses[6] = (opasses_t){opass1_o6, opass23f_o6};
  g_opasses[8] = (opasses_t){opass1_o8, opass23f_o8};
  g_opasses[13] = (opasses_t){opass1_o13, opass23s_o13};
  g_opasses[17] = (opasses_t){opass1_o17, opass23s_o17};
  OCT_A = (double*)big_alloc2(17L*(17*17*16+8)*8, 0);
  OCT_B = (double*)big_alloc2(17L*(17*17*16+8)*8, 2112);
  OCT_C = (double*)big_alloc2(17L*(17*17*16+8)*8, 1088);
  g_passes[6]  = (passes_t){pass1_s6, 0, fused23_s6, 0};
  g_passes[8]  = (passes_t){pass1_s8, 0, fused23_s8, 0};
  g_passes[13] = (passes_t){pass1_s13, 0, fused23_s13, 0};
  g_passes[17] = (passes_t){pass1_s17, 0, fused23_s17, 0};
  g_passes[23] = (passes_t){pass1_s23, 0, fused23_s23, 0};
  g_p1ip[23] = pass1ip_s23; g_p1ip[36] = pass1ip_s36; g_p1ip[45] = pass1ip_s45; g_p1ip[64] = pass1ip_s64;
  g_p1ipm[36] = pass1ipm_s36; g_p1ipm[45] = pass1ipm_s45; g_p1ipm[64] = pass1ipm_s64;
  g_passes[36] = (passes_t){pass1_s36, pass1m_s36, fused23_s36, fused23r_s36};
  g_passes[45] = (passes_t){pass1_s45, pass1m_s45, fused23_s45, fused23r_s45};
  g_passes[64] = (passes_t){pass1_s64, pass1m_s64, g_f23c ? fused23c_s64 : fused23_s64, fused23r_s64};
}

void impl_init(void){
  fill_consts();
  for (int i = 0; i < 8; i++){
    int L = SIZES[i], P = PADS[i];
    eng_t* e = &g_eng[L];
    e->L = L; e->P = P;
    e->PLANE = (long)P*8; e->SLAB = 2L*P*P + 8; e->VOL = (long)L*e->SLAB;
    e->A  = (double*)big_alloc2(e->VOL*8, 0);
    e->B  = (double*)big_alloc2(e->VOL*8, 2112);
    e->c0 = (double*)big_alloc2(e->VOL*8, 1088);
    e->c1 = (double*)big_alloc2(e->VOL*8, 3136);
    // generic DFT matrix
    double* W = (double*)big_alloc((long)L*L*2*8);
    for (int k = 0; k < L; k++)
      for (int j = 0; j < L; j++){
        long kj = ((long)k*j) % L;
        long double ang = -2.0L*3.14159265358979323846264338327950288L*(long double)kj/(long double)L;
        W[(k*L+j)*2]   = (double)cosl(ang);
        W[(k*L+j)*2+1] = (double)sinl(ang);
      }
    g_W[L] = W;
  }
}

void impl_set_exact_map(int on){ (void)on; }

#include <stdio.h>
#include <x86intrin.h>
#define BENCH_COD(NAME, CODFN) \
void NAME(int iters){ \
  unsigned aux; uint64_t t0, t1; \
  for (long i = 0; i < 72*8; i++){ SCR1RE[i] = (i%13)*0.01; SCR1IM[i] = (i%7)*0.02; } \
  t0 = __rdtscp(&aux); \
  for (int it = 0; it < iters; it++){ CODFN(SCR1RE, SCR1IM, 8, SCR2RE, SCR2IM, 8); __asm__ volatile("" ::: "memory"); } \
  t1 = __rdtscp(&aux); \
  printf("%s: %.1f cyc/chunk\n", #CODFN, (double)(t1-t0)/iters); fflush(stdout); \
}
BENCH_COD(bcod6, cod6) BENCH_COD(bcod8, cod8) BENCH_COD(bcod13, cod13)
BENCH_COD(bcod17, cod17) BENCH_COD(bcod23, cod23) BENCH_COD(bcod36, cod36)
BENCH_COD(bcod45, cod45) BENCH_COD(bcod64, cod64)
void impl_bench_oct(int L, int iters){
  opasses_t* ps = &g_opasses[L];
  long n = (long)L*L*L*16;
  for (long i = 0; i < n; i++) OCT_A[i] = (i % 97) * 0.01;
  unsigned aux; uint64_t t0, t1;
  double per = 1.0/iters;
  t0 = __rdtscp(&aux);
  for (int it = 0; it < iters; it++) ps->p1(OCT_A, OCT_A);
  t1 = __rdtscp(&aux);
  printf("L=%d opass1: %10.0f cyc\n", L, (t1-t0)*per);
  t0 = __rdtscp(&aux);
  for (int it = 0; it < iters; it++) ps->p23(OCT_A, OCT_C);
  t1 = __rdtscp(&aux);
  printf("L=%d opass23: %10.0f cyc\n", L, (t1-t0)*per);
  fflush(stdout);
}
void impl_bench_ip(int L, int iters){
  eng_t* e = &g_eng[L];
  passes_t* ps = &g_passes[L];
  for (long i = 0; i < e->VOL; i++) e->A[i] = (i % 97) * 0.01;
  unsigned aux; uint64_t t0, t1;
  double per = 1.0/iters;
  t0 = __rdtscp(&aux);
  for (int it = 0; it < iters; it++) g_p1ip[L](e->A);
  t1 = __rdtscp(&aux);
  printf("L=%d p1ip: %10.0f cyc\n", L, (t1-t0)*per);
  t0 = __rdtscp(&aux);
  for (int it = 0; it < iters; it++) ps->p23(e->A, e->B, e->A, e->c0);
  t1 = __rdtscp(&aux);
  printf("L=%d f23ip: %10.0f cyc\n", L, (t1-t0)*per);
  fflush(stdout);
}
void impl_bench(int L, int iters){
  eng_t* e = &g_eng[L];
  passes_t* ps = &g_passes[L];
  // fill A with some data
  for (long i = 0; i < e->VOL; i++) e->A[i] = (i % 97) * 0.01;
  unsigned aux; uint64_t t0, t1;
  double per = 1.0/iters;
  t0 = __rdtscp(&aux);
  for (int it = 0; it < iters; it++) ps->p1(e->A, e->B);
  t1 = __rdtscp(&aux);
  printf("L=%d pass1: %10.0f cyc\n", L, (t1-t0)*per);
  t0 = __rdtscp(&aux);
  for (int it = 0; it < iters; it++) ps->p23(e->B, e->A, e->B, e->c0);
  t1 = __rdtscp(&aux);
  printf("L=%d fused23: %10.0f cyc\n", L, (t1-t0)*per);
  fflush(stdout);
}
void impl_set_generic(int on){ g_use_generic = on; }
void impl_set_pf(int a, int b){ g_pf = a; g_pf23 = b; }
void impl_set_mapmix(int v){ g_mapmix = v; }
void impl_set_inplace(int v){ g_inplace = v; }
void impl_set_raw(int v){ g_raw = v; }
void impl_set_ip2(int v){ g_ip2 = v; }
void impl_set_f23c(int v){ g_f23c = v; g_passes[64].p23 = v ? fused23c_s64 : fused23_s64; }


int impl_run(int L, int B, int m, const double* x0, const double* c,
             double* one, double* fin){
  eng_t* e = &g_eng[L];
  if (!e->L) return -1;
  if (m < 1) m = 1;
  if (L <= 17 && !g_use_generic){
    int thresh = (L == 6) ? 5 : (L == 8 ? 9 : 6);   // min remainder worth a padded octet
    int noct = (B/8)*8;
    int rem = B - noct;
    if (rem >= thresh) { noct = B; rem = 0; }
    if (noct > 0) run_octet(L, 0, noct, m, x0, c, one, fin);
    if (rem > 0) run_eng_range(e, noct, rem, m, x0, c, one, fin);
  } else {
    run_eng(e, B, m, x0, c, one, fin);
  }
  return 0;
}
