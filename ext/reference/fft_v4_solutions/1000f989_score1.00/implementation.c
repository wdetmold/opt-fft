// Iterated batched 3D complex FFT for L in {6,8,13,17,23,36,45,64}.
// Lane-interleaved SoA layout: slot(x,y,z) = { vec re, vec im }, vec = 8 or 4 doubles (volumes).
// Per iteration: sweep1 (y-planes): lazy map rows + FFT_z + FFT_x; sweep2 (x-planes): FFT_y.
// Buffer holds RAW FFT3 between iterations; map z/(1+|z|) applied lazily next iteration.
#ifndef TPASS
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <sys/mman.h>

#define AI __attribute__((always_inline)) static inline
#define NOIN __attribute__((noinline))
#define CAT_(a,b) a##b
#define CAT(a,b) CAT_(a,b)

// padded strides (in slots)
#define PX6 37
#define PY6 6
#define PX8 70
#define PY8 8
#define PX13 174
#define PY13 13
#define PX17 290
#define PY17 17
#define PX23 531
#define PY23 23
#define PX36 1299
#define PY36 36
#define PX45 2027
#define PY45 45
#define PX64 4161
#define PY64 65

static double W9r[5], W9i[5];
static double W64r[64], W64i[64];
static double C13t[7][8], S13t[7][8];    // transposed: [j][k]
static double C17t[9][9], S17t[9][9];
static double C23t[12][12], S23t[12][12];

static void init_zsplit(void);
static void* g_buf = 0;
static void* g_cbuf = 0;

static void ensure_buffers(void){
  if(!g_buf){
    size_t sz = (size_t)64*PX64*128;
    sz = (sz + (1<<21)) & ~((size_t)(1<<21)-1);
    g_buf  = aligned_alloc(1<<21, sz);
    g_cbuf = aligned_alloc(1<<21, sz);
    madvise(g_buf, sz, MADV_HUGEPAGE);
    madvise(g_cbuf, sz, MADV_HUGEPAGE);
    memset(g_buf, 0, sz); memset(g_cbuf, 0, sz);
  }
}

void init_tables(void){
  const long double PI = 3.14159265358979323846264338327950288L;
  for(int t=0;t<64;t++){ W64r[t]=(double)cosl(-2*PI*t/64); W64i[t]=(double)sinl(-2*PI*t/64); }
  for(int t=0;t<5;t++){ W9r[t]=(double)cosl(-2*PI*t/9); W9i[t]=(double)sinl(-2*PI*t/9); }
  for(int k=0;k<7;k++) for(int j=0;j<7;j++){ C13t[j][k]=(double)cosl(2*PI*((long)k*j%13)/13); S13t[j][k]=(double)sinl(2*PI*((long)k*j%13)/13); }
  for(int k=0;k<9;k++) for(int j=0;j<9;j++){ C17t[j][k]=(double)cosl(2*PI*((long)k*j%17)/17); S17t[j][k]=(double)sinl(2*PI*((long)k*j%17)/17); }
  for(int k=0;k<12;k++) for(int j=0;j<12;j++){ C23t[j][k]=(double)cosl(2*PI*((long)k*j%23)/23); S23t[j][k]=(double)sinl(2*PI*((long)k*j%23)/23); }
  init_zsplit();
  ensure_buffers();
}

// ======== instantiate templated code for widths 8 and 4 ========
#define TPASS 1
#define VW 8
#include __FILE__
#undef VW
#define VW 4
#include __FILE__
#undef VW
#define VW 2
#include __FILE__
#undef VW
#undef TPASS


// =================== L=64 within-volume z-split (w8 lanes over z-octants) ===================
// Form A: slot(x,y,zh), lane l holds z = zh + 8*l  (natural)
// Form B: slot(x,y,q),  lane l holds z = brev(l) + 8*q
#define PXZ 583
#define PYZ 9
static __m512d ZTWr[8], ZTWi[8];      // [i].lane[l] = w64^{i*brev(l)}
static __m512d T4Dr, T4Di;            // DIF dist-4 post-tw (sign-folded): lanes 4..7 = -w8^{l-4}
static __m512d T2Dr, T2Di;            // DIF dist-2 post-tw: lanes {2,3,6,7} = -[1,-i]
static __m512d T2Tr, T2Ti;            // DIT dist-2 pre-tw: lanes {2,3,6,7} = [1,-i], others identity
static __m512d T4Tr, T4Ti;            // DIT dist-4 pre-tw: lanes 4..7 = w8^{0..3}, others identity
static const int BREV8[8] = {0,4,2,6,1,5,3,7};

AI __m512d zswap4(__m512d x){ return _mm512_shuffle_f64x2(x, x, 0x4E); }
AI __m512d zswap2(__m512d x){ return _mm512_shuffle_f64x2(x, x, 0xB1); }
AI __m512d zswap1(__m512d x){ return _mm512_permute_pd(x, 0x55); }

// DIF cross-lane DFT8 (natural lanes in, bit-reversed out), on (re,im)
AI void xl_dif8(__m512d* xr, __m512d* xi){
  __m512d r=*xr, i=*xi;
  // stage dist4, post-tw on lanes 4-7 (sign folded into T4D)
  { __m512d sr=_mm512_add_pd(r, zswap4(r)), si=_mm512_add_pd(i, zswap4(i));
    __m512d dr=_mm512_sub_pd(r, zswap4(r)), di=_mm512_sub_pd(i, zswap4(i));
    __m512d tr=_mm512_fnmadd_pd(di, T4Di, _mm512_mul_pd(dr, T4Dr));
    __m512d ti=_mm512_fmadd_pd (dr, T4Di, _mm512_mul_pd(di, T4Dr));
    r=_mm512_mask_blend_pd(0xF0, sr, tr); i=_mm512_mask_blend_pd(0xF0, si, ti); }
  // stage dist2, post-tw on lanes {2,3,6,7}
  { __m512d sr=_mm512_add_pd(r, zswap2(r)), si=_mm512_add_pd(i, zswap2(i));
    __m512d dr=_mm512_sub_pd(r, zswap2(r)), di=_mm512_sub_pd(i, zswap2(i));
    __m512d tr=_mm512_fnmadd_pd(di, T2Di, _mm512_mul_pd(dr, T2Dr));
    __m512d ti=_mm512_fmadd_pd (dr, T2Di, _mm512_mul_pd(di, T2Dr));
    r=_mm512_mask_blend_pd(0xCC, sr, tr); i=_mm512_mask_blend_pd(0xCC, si, ti); }
  // stage dist1, no tw: odd lanes = even - odd
  { __m512d sr=_mm512_add_pd(r, zswap1(r)), si=_mm512_add_pd(i, zswap1(i));
    r=_mm512_mask_sub_pd(sr, 0xAA, zswap1(r), r);
    i=_mm512_mask_sub_pd(si, 0xAA, zswap1(i), i); }
  *xr=r; *xi=i;
}
// DIT cross-lane DFT8 (bit-reversed lanes in, natural out)
AI void xl_dit8(__m512d* xr, __m512d* xi){
  __m512d r=*xr, i=*xi;
  // stage dist1, no tw
  { __m512d sr=_mm512_add_pd(r, zswap1(r)), si=_mm512_add_pd(i, zswap1(i));
    __m512d nr=_mm512_mask_sub_pd(sr, 0xAA, zswap1(r), r);
    __m512d ni=_mm512_mask_sub_pd(si, 0xAA, zswap1(i), i);
    r=nr; i=ni; }
  // stage dist2: pre-tw lanes {2,3,6,7} by [1,-i]
  { __m512d tr=_mm512_fnmadd_pd(i, T2Ti, _mm512_mul_pd(r, T2Tr));
    __m512d ti=_mm512_fmadd_pd (r, T2Ti, _mm512_mul_pd(i, T2Tr));
    __m512d sr=_mm512_add_pd(tr, zswap2(tr)), si=_mm512_add_pd(ti, zswap2(ti));
    r=_mm512_mask_sub_pd(sr, 0xCC, zswap2(tr), tr);
    i=_mm512_mask_sub_pd(si, 0xCC, zswap2(ti), ti); }
  // stage dist4: pre-tw lanes 4-7 by w8^{0..3}
  { __m512d tr=_mm512_fnmadd_pd(i, T4Ti, _mm512_mul_pd(r, T4Tr));
    __m512d ti=_mm512_fmadd_pd (r, T4Ti, _mm512_mul_pd(i, T4Tr));
    __m512d sr=_mm512_add_pd(tr, zswap4(tr)), si=_mm512_add_pd(ti, zswap4(ti));
    r=_mm512_mask_sub_pd(sr, 0xF0, zswap4(tr), tr);
    i=_mm512_mask_sub_pd(si, 0xF0, zswap4(ti), ti); }
  *xr=r; *xi=i;
}

// z-pass A->B on one row of 8 slots (stride 1); optional fused map on load
AI void zpassAB_m(cpx_w8* restrict row, const int domap, const cpx_w8* restrict crow){
  cplx_w8 g[8];
  _Pragma("GCC unroll 8")
  for(int zh=0; zh<8; zh++){
    cplx_w8 in = LD_w8(row+zh);
    if(domap) in = mapc_w8(in, crow+zh);
    __m512d r=(__m512d)in.r, i=(__m512d)in.i;
    xl_dif8(&r, &i);
    if(zh){
      __m512d tr=_mm512_fnmadd_pd(i, ZTWi[zh], _mm512_mul_pd(r, ZTWr[zh]));
      __m512d ti=_mm512_fmadd_pd (r, ZTWi[zh], _mm512_mul_pd(i, ZTWr[zh]));
      r=tr; i=ti;
    }
    g[zh].r=(vecd_w8)r; g[zh].i=(vecd_w8)i;
  }
  DFT8V_w8(g);
  _Pragma("GCC unroll 8")
  for(int q=0; q<8; q++){ row[q].re=g[q].r; row[q].im=g[q].i; }
}
// z-pass B->A
AI void zpassBA_m(cpx_w8* restrict row, const int domap, const cpx_w8* restrict crow){
  cplx_w8 g[8];
  _Pragma("GCC unroll 8")
  for(int s=0; s<8; s++){ cplx_w8 in = LD_w8(row+s); if(domap) in = mapc_w8(in, crow+s); g[s]=in; }
  DFT8V_w8(g);
  _Pragma("GCC unroll 8")
  for(int k1=0; k1<8; k1++){
    __m512d r=(__m512d)g[k1].r, i=(__m512d)g[k1].i;
    if(k1){
      __m512d tr=_mm512_fnmadd_pd(i, ZTWi[k1], _mm512_mul_pd(r, ZTWr[k1]));
      __m512d ti=_mm512_fmadd_pd (r, ZTWi[k1], _mm512_mul_pd(i, ZTWr[k1]));
      r=tr; i=ti;
    }
    xl_dit8(&r, &i);
    row[k1].re=(vecd_w8)r; row[k1].im=(vecd_w8)i;
  }
}

// 8x8 transpose of (slot,lane) for one row; rows fed in brev order gives form-B c from form-A c.
AI void tr8x8_brev(const __m512d in[8], __m512d out[8]){
  __m512d t0=_mm512_unpacklo_pd(in[0],in[1]), t1=_mm512_unpackhi_pd(in[0],in[1]);
  __m512d t2=_mm512_unpacklo_pd(in[2],in[3]), t3=_mm512_unpackhi_pd(in[2],in[3]);
  __m512d t4=_mm512_unpacklo_pd(in[4],in[5]), t5=_mm512_unpackhi_pd(in[4],in[5]);
  __m512d t6=_mm512_unpacklo_pd(in[6],in[7]), t7=_mm512_unpackhi_pd(in[6],in[7]);
  __m512d m0=_mm512_shuffle_f64x2(t0,t2,0x88), m1=_mm512_shuffle_f64x2(t0,t2,0xDD);
  __m512d m2=_mm512_shuffle_f64x2(t1,t3,0x88), m3=_mm512_shuffle_f64x2(t1,t3,0xDD);
  __m512d m4=_mm512_shuffle_f64x2(t4,t6,0x88), m5=_mm512_shuffle_f64x2(t4,t6,0xDD);
  __m512d m6=_mm512_shuffle_f64x2(t5,t7,0x88), m7=_mm512_shuffle_f64x2(t5,t7,0xDD);
  out[0]=_mm512_shuffle_f64x2(m0,m4,0x88);
  out[1]=_mm512_shuffle_f64x2(m2,m6,0x88);
  out[2]=_mm512_shuffle_f64x2(m1,m5,0x88);
  out[3]=_mm512_shuffle_f64x2(m3,m7,0x88);
  out[4]=_mm512_shuffle_f64x2(m0,m4,0xDD);
  out[5]=_mm512_shuffle_f64x2(m2,m6,0xDD);
  out[6]=_mm512_shuffle_f64x2(m1,m5,0xDD);
  out[7]=_mm512_shuffle_f64x2(m3,m7,0xDD);
}
// NOTE: matches tr_fwd network: out[c] = lanes r of in[r] at column c.
// map a row of 8 slots in place, with c given in form A but data in form B:
// cB[q].lane[l] = cA[brev(l)].lane[q]
AI void map_row_trB(cpx_w8* restrict p, const cpx_w8* restrict cArow){
  __m512d inr[8], ini[8], cr[8], ci[8];
  _Pragma("GCC unroll 8")
  for(int l=0;l<8;l++){ int s=BREV8[l]; inr[l]=(__m512d)cArow[s].re; ini[l]=(__m512d)cArow[s].im; }
  tr8x8_brev(inr, cr);
  tr8x8_brev(ini, ci);
  _Pragma("GCC unroll 8")
  for(int q=0;q<8;q++){
    cpx_w8 cb; cb.re=(vecd_w8)cr[q]; cb.im=(vecd_w8)ci[q];
    cplx_w8 v = mapc_w8(LD_w8(p+q), &cb);
    ST_w8(p+q, v);
  }
}

// one zsplit iteration. dir: 0 = A->B (zAB), 1 = B->A. premap: 0 none, else map with given c.
NOIN void iter_z64(cpx_w8* restrict b, const cpx_w8* restrict cf, const int premap, const int dir){
  for(int y=0;y<64;y++){
    cpx_w8* pl = b + (long)y*PYZ;
    const cpx_w8* cl = cf + (long)y*PYZ;
    if(premap){ if(dir==0) map_row_w8(pl, cl, 8); else map_row_trB(pl, cl); }
    for(int x=0;x<64;x++){
      if(x+3<64){
        const char* q=(const char*)(pl + (long)(x+3)*PXZ);
        for(int t=0;t<16;t++) __builtin_prefetch(q + t*64, 0, 3);
        if(premap){ const char* q2=(const char*)(cl + (long)(x+3)*PXZ); for(int t=0;t<16;t++) __builtin_prefetch(q2 + t*64, 0, 3); }
      }
      if(premap && x+1<64){
        if(dir==0) map_row_w8(pl + (long)(x+1)*PXZ, cl + (long)(x+1)*PXZ, 8);
        else       map_row_trB(pl + (long)(x+1)*PXZ, cl + (long)(x+1)*PXZ);
      }
      if(dir==0) zpassAB_m(pl + (long)x*PXZ, 0, 0);
      else       zpassBA_m(pl + (long)x*PXZ, 0, 0);
    }
    for(int zh=0; zh<8; zh++) dft64_w8(pl + zh, PXZ);
  }
  for(int x=0;x<64;x++){
    cpx_w8* pl = b + (long)x*PXZ;
    for(int zh=0; zh<8; zh++) dft64_w8(pl + zh, PYZ);
  }
}

// ---------------- zsplit io ----------------
// interleave natural complex volume -> form A buffer
NOIN void z64_interleave_A(const double* restrict src, cpx_w8* restrict buf){
  const double* base[8];
  for(int x=0;x<64;x++) for(int y=0;y<64;y++){
    cpx_w8* row = buf + (long)x*PXZ + (long)y*PYZ;
    long e0 = ((long)x*64+y)*64;
    for(int l=0;l<8;l++) base[l] = src + (e0 + 8*l)*2;
    tr_fwd_w8(base, 0, row);
    tr_fwd_w8(base, 8, row+4);
  }
}
// interleave natural complex volume -> form B buffer (for cB)
static __m512d ZIDX0, ZIDX1; // permute indices
NOIN void z64_interleave_B(const double* restrict src, cpx_w8* restrict buf){
  for(int x=0;x<64;x++) for(int y=0;y<64;y++){
    cpx_w8* row = buf + (long)x*PXZ + (long)y*PYZ;
    long e0 = ((long)x*64+y)*64;
    for(int q=0;q<8;q++){
      __m512d a = _mm512_loadu_pd(src + (e0+8*q)*2);      // r0 i0 r1 i1 r2 i2 r3 i3
      __m512d bb = _mm512_loadu_pd(src + (e0+8*q)*2 + 8); // r4 i4 ...
      // lane l: re = elem brev(l): idx in [a|b] doubles = 2*brev(l) ; im = +1
      __m512d re = _mm512_permutex2var_pd(a, (__m512i)ZIDX0, bb);
      __m512d im = _mm512_permutex2var_pd(a, (__m512i)ZIDX1, bb);
      row[q].re=(vecd_w8)re; row[q].im=(vecd_w8)im;
    }
  }
}
// materialize form A (optionally map) -> natural complex
NOIN void z64_out_A(const cpx_w8* restrict buf, const cpx_w8* restrict cA, double* restrict dst, int domap){
  double* base[8];
  for(int x=0;x<64;x++) for(int y=0;y<64;y++){
    const cpx_w8* row = buf + (long)x*PXZ + (long)y*PYZ;
    const cpx_w8* crow = cA + (long)x*PXZ + (long)y*PYZ;
    long e0 = ((long)x*64+y)*64;
    for(int l=0;l<8;l++) base[l] = dst + (e0 + 8*l)*2;
    cplx_w8 s[4];
    for(int t=0;t<4;t++) s[t] = domap ? mapc_w8(LD_w8(row+t), crow+t) : LD_w8(row+t);
    tr_bwd_w8(s, base, 0, 8);
    for(int t=0;t<4;t++) s[t] = domap ? mapc_w8(LD_w8(row+4+t), crow+4+t) : LD_w8(row+4+t);
    tr_bwd_w8(s, base, 8, 8);
  }
}
static __m512d ZODX0, ZODX1;
NOIN void z64_out_B(const cpx_w8* restrict buf, const cpx_w8* restrict cA, double* restrict dst, int domap){
  for(int x=0;x<64;x++) for(int y=0;y<64;y++){
    const cpx_w8* row = buf + (long)x*PXZ + (long)y*PYZ;
    const cpx_w8* crow = cA + (long)x*PXZ + (long)y*PYZ;
    long e0 = ((long)x*64+y)*64;
    __m512d inr[8], ini[8], cr[8], ci[8];
    for(int l=0;l<8;l++){ int s=BREV8[l]; inr[l]=(__m512d)crow[s].re; ini[l]=(__m512d)crow[s].im; }
    tr8x8_brev(inr, cr);
    tr8x8_brev(ini, ci);
    for(int q=0;q<8;q++){
      cpx_w8 cb; cb.re=(vecd_w8)cr[q]; cb.im=(vecd_w8)ci[q];
      cplx_w8 s = domap ? mapc_w8(LD_w8(row+q), &cb) : LD_w8(row+q);
      __m512d lo = _mm512_permutex2var_pd((__m512d)s.r, (__m512i)ZODX0, (__m512d)s.i);
      __m512d hi = _mm512_permutex2var_pd((__m512d)s.r, (__m512i)ZODX1, (__m512d)s.i);
      _mm512_storeu_pd(dst + (e0+8*q)*2, lo);
      _mm512_storeu_pd(dst + (e0+8*q)*2 + 8, hi);
    }
  }
}

static void init_zsplit(void){
  const long double PI = 3.14159265358979323846264338327950288L;
  double tmpr[8], tmpi[8];
  for(int i=0;i<8;i++){
    for(int l=0;l<8;l++){
      long e = (long)i*BREV8[l];
      tmpr[l]=(double)cosl(-2*PI*e/64); tmpi[l]=(double)sinl(-2*PI*e/64);
    }
    ZTWr[i]=_mm512_loadu_pd(tmpr); ZTWi[i]=_mm512_loadu_pd(tmpi);
  }
  // T4D: lanes 4-7 = -w8^{l-4}; others (1,0)
  for(int l=0;l<8;l++){ tmpr[l]=1.0; tmpi[l]=0.0; }
  for(int l=4;l<8;l++){ tmpr[l]=-(double)cosl(-2*PI*(l-4)/8); tmpi[l]=-(double)sinl(-2*PI*(l-4)/8); }
  T4Dr=_mm512_loadu_pd(tmpr); T4Di=_mm512_loadu_pd(tmpi);
  // T2D: lanes {2,6} = -1, {3,7} = -(-i) = +i ; others (1,0)
  for(int l=0;l<8;l++){ tmpr[l]=1.0; tmpi[l]=0.0; }
  tmpr[2]=-1.0; tmpi[2]=0.0; tmpr[6]=-1.0; tmpi[6]=0.0;
  tmpr[3]=0.0;  tmpi[3]=1.0; tmpr[7]=0.0;  tmpi[7]=1.0;
  T2Dr=_mm512_loadu_pd(tmpr); T2Di=_mm512_loadu_pd(tmpi);
  // T2T (DIT pre-tw): lanes {3,7} = -i; others 1
  for(int l=0;l<8;l++){ tmpr[l]=1.0; tmpi[l]=0.0; }
  tmpr[3]=0.0; tmpi[3]=-1.0; tmpr[7]=0.0; tmpi[7]=-1.0;
  T2Tr=_mm512_loadu_pd(tmpr); T2Ti=_mm512_loadu_pd(tmpi);
  // T4T: lanes 4-7 = w8^{0..3}; others 1
  for(int l=0;l<8;l++){ tmpr[l]=1.0; tmpi[l]=0.0; }
  for(int l=4;l<8;l++){ tmpr[l]=(double)cosl(-2*PI*(l-4)/8); tmpi[l]=(double)sinl(-2*PI*(l-4)/8); }
  T4Tr=_mm512_loadu_pd(tmpr); T4Ti=_mm512_loadu_pd(tmpi);
  // permute indices
  long long i0[8], i1[8];
  for(int l=0;l<8;l++){ i0[l]=2*BREV8[l]; i1[l]=2*BREV8[l]+1; }
  ZIDX0=_mm512_loadu_pd((double*)i0); ZIDX1=_mm512_loadu_pd((double*)i1);
  for(int j=0;j<4;j++){ i0[2*j]=BREV8[j]; i0[2*j+1]=8+BREV8[j]; i1[2*j]=BREV8[4+j]; i1[2*j+1]=8+BREV8[4+j]; }
  ZODX0=_mm512_loadu_pd((double*)i0); ZODX1=_mm512_loadu_pd((double*)i1);
}

// run L=64 with zsplit layout: one volume per group
NOIN void run64_zsplit(long B, long m, const double* x0, const double* cs, double* out1, double* outm){
  const long vol3 = 64L*64*64;
  cpx_w8* buf = (cpx_w8*)g_buf;
  cpx_w8* cA  = (cpx_w8*)g_cbuf;
  for(long v=0; v<B; v++){
    const double* xg = x0 + v*vol3*2;
    const double* cg = cs + v*vol3*2;
    z64_interleave_A(xg, buf);
    z64_interleave_A(cg, cA);
    for(long it=1; it<=m; it++){
      int dir = (it & 1) ? 0 : 1;              // odd iter: A->B
      iter_z64(buf, cA, it>1, dir);
      if(it==1) z64_out_B(buf, cA, out1 + v*vol3*2, 1);
    }
    if(m==1){ memcpy(outm + v*vol3*2, out1 + v*vol3*2, (size_t)vol3*16); }
    else if(m & 1) z64_out_B(buf, cA, outm + v*vol3*2, 1);
    else           z64_out_A(buf, cA, outm + v*vol3*2, 1);
  }
}

// ======================= top-level dispatch ==========================
// per-size full-group width (tunable)
static int FW[65];
static void set_policy(void){
  FW[6]=8; FW[8]=8; FW[13]=8; FW[17]=8; FW[23]=8; FW[36]=8; FW[45]=8; FW[64]=1;
}
void set_fw(int L, int w){ FW[L]=w; }

// SCHEME: 0 = buffer holds mapped state (snapshot_out); 1 = raw + lazy map (materialize_out)
#define SCHEME_6 0
#define SCHEME_8 0
#define SCHEME_13 0
#define SCHEME_17 0
#define SCHEME_23 0
#define SCHEME_36 1
#define SCHEME_45 1
#define SCHEME_64 1

#define RUN_GROUP(LV, PX, PY, W) \
  { \
    interleave_in_w##W(xg, g_buf, LV, PX, PY, lanes, vol3); \
    interleave_in_w##W(cg, g_cbuf, LV, PX, PY, lanes, vol3); \
    for(long it=1; it<=m; it++){ \
      iterw_##LV##_w##W(g_buf, g_cbuf, it>1); \
      if(it==1){ \
        if(SCHEME_##LV) materialize_out_w##W(g_buf, g_cbuf, out1 + done*vol3*2, LV, PX, PY, lanes, vol3); \
        else snapshot_out_w##W(g_buf, out1 + done*vol3*2, LV, PX, PY, lanes, vol3); \
      } \
    } \
    if(m==1) memcpy(outm + done*vol3*2, out1 + done*vol3*2, (size_t)lanes*vol3*16); \
    else if(SCHEME_##LV) materialize_out_w##W(g_buf, g_cbuf, outm + done*vol3*2, LV, PX, PY, lanes, vol3); \
    else snapshot_out_w##W(g_buf, outm + done*vol3*2, LV, PX, PY, lanes, vol3); \
  }

#define RUN_CASE(LV, PX, PY) \
  case LV: { \
    const long vol3=(long)LV*LV*LV; \
    long done=0; \
    while(done < B){ \
      long rem = B - done; \
      const double* xg = x0 + done*vol3*2; \
      const double* cg = cs + done*vol3*2; \
      int w = FW[LV]; \
      while(w > 2 && rem <= w/2) w >>= 1; \
      if(w == 8){ \
        int lanes = (int)(rem < 8 ? rem : 8); \
        RUN_GROUP(LV, PX, PY, 8) \
        done += lanes; \
      } else if(w == 4){ \
        int lanes = (int)(rem < 4 ? rem : 4); \
        RUN_GROUP(LV, PX, PY, 4) \
        done += lanes; \
      } else { \
        int lanes = (int)(rem < 2 ? rem : 2); \
        RUN_GROUP(LV, PX, PY, 2) \
        done += lanes; \
      } \
    } \
  } break;

void run(int L, long B, long m, const double* x0, const double* cs, double* out1, double* outm){
  ensure_buffers();
  if(m < 1) m = 1;
  if(B < 1) return;
  if(!FW[6]) set_policy();
  if(L==64 && FW[64]==1){ run64_zsplit(B, m, x0, cs, out1, outm); return; }
  switch(L){
    RUN_CASE(6,  PX6,  PY6)
    RUN_CASE(8,  PX8,  PY8)
    RUN_CASE(13, PX13, PY13)
    RUN_CASE(17, PX17, PY17)
    RUN_CASE(23, PX23, PY23)
    RUN_CASE(36, PX36, PY36)
    RUN_CASE(45, PX45, PY45)
    RUN_CASE(64, PX64, PY64)
  }
}

#else  /* ==================== TEMPLATED SECTION (uses VW) ==================== */

#define FN(name) CAT(name, CAT(_w, VW))
#define vN  FN(vecd)
#define cpN FN(cpx)
#define CN  FN(cplx)

typedef double vN __attribute__((vector_size(VW*8), aligned(VW*8)));
typedef struct { vN re, im; } cpN;
typedef struct { vN r, i; } CN;

#if VW == 8
#define BCAST(x) ((vN){x,x,x,x,x,x,x,x})
#define VRSQRT14(x) ((vN)_mm512_rsqrt14_pd((__m512d)(x)))
#elif VW == 4
#define BCAST(x) ((vN){x,x,x,x})
#define VRSQRT14(x) ((vN)_mm256_rsqrt14_pd((__m256d)(x)))
#else
#define BCAST(x) ((vN){x,x})
#define VRSQRT14(x) ((vN)_mm_rsqrt14_pd((__m128d)(x)))
#endif

#define bc    FN(bc)
#define LD    FN(LD)
#define ST    FN(ST)
#define cadd  FN(cadd)
#define csub  FN(csub)
#define ctw   FN(ctw)
#define cmi   FN(cmi)
#define mapc  FN(mapc)
#define PUTM  FN(PUTM)
#define map_row FN(map_row)
#define DFT2  FN(DFT2)
#define DFT3  FN(DFT3)
#define DFT4  FN(DFT4)
#define DFT5  FN(DFT5)
#define DFT8V FN(DFT8V)
#define DFT9V FN(DFT9V)
#define dft6  FN(dft6)
#define dft8  FN(dft8)
#define dft13 FN(dft13)
#define dft17 FN(dft17)
#define dft23 FN(dft23)
#define dft36 FN(dft36)
#define dft45 FN(dft45)
#define dft64 FN(dft64)

AI vN bc(double x){ return BCAST(x); }
AI CN LD(const cpN* p){ CN a; a.r = p->re; a.i = p->im; return a; }
AI void ST(cpN* p, CN a){ p->re = a.r; p->im = a.i; }
AI CN cadd(CN a, CN b){ CN c; c.r=a.r+b.r; c.i=a.i+b.i; return c; }
AI CN csub(CN a, CN b){ CN c; c.r=a.r-b.r; c.i=a.i-b.i; return c; }
AI CN ctw(CN a, double tr, double ti){
  CN c;
  c.r = a.r*bc(tr) - a.i*bc(ti);
  c.i = a.r*bc(ti) + a.i*bc(tr);
  return c;
}
AI CN cmi(CN a){ CN c; c.r = a.i; c.i = -a.r; return c; }

AI CN mapc(CN X, const cpN* cp){
  vN zr = X.r + cp->re, zi = X.i + cp->im;
  vN r  = zr*zr + zi*zi + bc(1e-300);
  vN q = VRSQRT14(r);
  vN hr = bc(0.5)*r;
  q = q*(bc(1.5) - hr*q*q);
  q = q*(bc(1.5) - hr*q*q);
  vN w = bc(1.0) + r*q;
  vN p = bc(1.0)/w;
  CN o; o.r = zr*p; o.i = zi*p; return o;
}

AI void PUTM(cpN* p, CN v, const int domap, const cpN* cp){
  if(domap) v = mapc(v, cp);
  ST(p, v);
}

AI void map_row(cpN* restrict p, const cpN* restrict cp, const int n){
  for(int i=0;i<n;i++){
    CN v = mapc(LD(p+i), cp+i);
    ST(p+i, v);
  }
}

AI void DFT2(CN* a, CN* b){ CN t=*a; *a = cadd(t,*b); *b = csub(t,*b); }

AI void DFT3(CN* x0, CN* x1, CN* x2){
  const double T = 0.86602540378443864676372317075293618347;
  vN sr = x1->r + x2->r, si = x1->i + x2->i;
  vN dr = x1->r - x2->r, di = x1->i - x2->i;
  vN ur = x0->r - bc(0.5)*sr, ui = x0->i - bc(0.5)*si;
  CN X0; X0.r = x0->r + sr; X0.i = x0->i + si;
  CN X1; X1.r = ur + bc(T)*di; X1.i = ui - bc(T)*dr;
  CN X2; X2.r = ur - bc(T)*di; X2.i = ui + bc(T)*dr;
  *x0 = X0; *x1 = X1; *x2 = X2;
}

AI void DFT4(CN* x0, CN* x1, CN* x2, CN* x3){
  CN t0 = cadd(*x0, *x2), t1 = csub(*x0, *x2);
  CN t2 = cadd(*x1, *x3), t3 = csub(*x1, *x3);
  *x0 = cadd(t0, t2); *x2 = csub(t0, t2);
  CN mt3 = cmi(t3);
  *x1 = cadd(t1, mt3); *x3 = csub(t1, mt3);
}

AI void DFT5(CN* x0, CN* x1, CN* x2, CN* x3, CN* x4){
  const double C1 = 0.30901699437494742410229341718281905886;
  const double C2 = -0.80901699437494742410229341718281905886;
  const double S1 = 0.95105651629515357211643933337938214340;
  const double S2 = 0.58778525229247312916870595463907276860;
  vN a1r = x1->r + x4->r, a1i = x1->i + x4->i;
  vN b1r = x1->r - x4->r, b1i = x1->i - x4->i;
  vN a2r = x2->r + x3->r, a2i = x2->i + x3->i;
  vN b2r = x2->r - x3->r, b2i = x2->i - x3->i;
  CN X0; X0.r = x0->r + a1r + a2r; X0.i = x0->i + a1i + a2i;
  vN cr = x0->r + bc(C1)*a1r + bc(C2)*a2r;
  vN ci = x0->i + bc(C1)*a1i + bc(C2)*a2i;
  vN sr = bc(S1)*b1r + bc(S2)*b2r;
  vN si = bc(S1)*b1i + bc(S2)*b2i;
  CN X1, X4;
  X1.r = cr + si; X1.i = ci - sr;
  X4.r = cr - si; X4.i = ci + sr;
  cr = x0->r + bc(C2)*a1r + bc(C1)*a2r;
  ci = x0->i + bc(C2)*a1i + bc(C1)*a2i;
  sr = bc(S2)*b1r - bc(S1)*b2r;
  si = bc(S2)*b1i - bc(S1)*b2i;
  CN X2, X3;
  X2.r = cr + si; X2.i = ci - sr;
  X3.r = cr - si; X3.i = ci + sr;
  *x0 = X0; *x1 = X1; *x2 = X2; *x3 = X3; *x4 = X4;
}

AI void DFT8V(CN x[8]){
  CN e0=x[0], e1=x[2], e2=x[4], e3=x[6];
  CN o0=x[1], o1=x[3], o2=x[5], o3=x[7];
  DFT4(&e0,&e1,&e2,&e3);
  DFT4(&o0,&o1,&o2,&o3);
  const double Chalf = 0.70710678118654752440084436210484903928;
  x[0] = cadd(e0,o0); x[4] = csub(e0,o0);
  vN t = o1.r + o1.i, u = o1.i - o1.r;
  x[1].r = e1.r + bc(Chalf)*t; x[1].i = e1.i + bc(Chalf)*u;
  x[5].r = e1.r - bc(Chalf)*t; x[5].i = e1.i - bc(Chalf)*u;
  x[2].r = e2.r + o2.i; x[2].i = e2.i - o2.r;
  x[6].r = e2.r - o2.i; x[6].i = e2.i + o2.r;
  t = o3.i - o3.r; u = o3.r + o3.i;
  x[3].r = e3.r + bc(Chalf)*t; x[3].i = e3.i - bc(Chalf)*u;
  x[7].r = e3.r - bc(Chalf)*t; x[7].i = e3.i + bc(Chalf)*u;
}

AI void DFT9V(CN x[9]){
  CN g00=x[0], g01=x[3], g02=x[6];
  CN g10=x[1], g11=x[4], g12=x[7];
  CN g20=x[2], g21=x[5], g22=x[8];
  DFT3(&g00,&g01,&g02);
  DFT3(&g10,&g11,&g12);
  DFT3(&g20,&g21,&g22);
  DFT3(&g00,&g10,&g20);
  x[0]=g00; x[3]=g10; x[6]=g20;
  CN t1 = ctw(g11, W9r[1], W9i[1]);
  CN t2 = ctw(g21, W9r[2], W9i[2]);
  DFT3(&g01,&t1,&t2);
  x[1]=g01; x[4]=t1; x[7]=t2;
  t1 = ctw(g12, W9r[2], W9i[2]);
  t2 = ctw(g22, W9r[4], W9i[4]);
  DFT3(&g02,&t1,&t2);
  x[2]=g02; x[5]=t1; x[8]=t2;
}

AI void dft6(cpN* restrict p, const long S, const int domap, const cpN* restrict cp){
  CN e0 = LD(p+0*S), e1 = LD(p+2*S), e2 = LD(p+4*S);
  CN o0 = LD(p+3*S), o1 = LD(p+5*S), o2 = LD(p+1*S);
  DFT2(&e0,&o0); DFT2(&e1,&o1); DFT2(&e2,&o2);
  DFT3(&e0,&e1,&e2);
  DFT3(&o0,&o1,&o2);
  PUTM(p+0*S,e0,domap,cp+0*S); PUTM(p+4*S,e1,domap,cp+4*S); PUTM(p+2*S,e2,domap,cp+2*S);
  PUTM(p+3*S,o0,domap,cp+3*S); PUTM(p+1*S,o1,domap,cp+1*S); PUTM(p+5*S,o2,domap,cp+5*S);
}

AI void dft8(cpN* restrict p, const long S, const int domap, const cpN* restrict cp){
  CN x[8];
  _Pragma("GCC unroll 8")
  for(int j=0;j<8;j++) x[j]=LD(p+j*S);
  DFT8V(x);
  _Pragma("GCC unroll 8")
  for(int k=0;k<8;k++) PUTM(p+k*S,x[k],domap,cp+(long)k*S);
}

// primes: symmetric direct, j-outer matvec with k-indexed register accumulators
#define GEN_PRIME(NAME, P, H, CTj, STj) \
AI void NAME(cpN* restrict p, const long S, const int domap, const cpN* restrict cp){ \
  vN br[H], bi[H]; \
  vN CR[H+1], CI[H+1]; \
  { CN x0 = LD(p); \
    _Pragma("GCC unroll 16") \
    for(int k=0;k<=H;k++){ CR[k]=x0.r; CI[k]=x0.i; } } \
  _Pragma("GCC unroll 16") \
  for(int j=1;j<=H;j++){ \
    CN u = LD(p+j*S), v = LD(p+(P-j)*S); \
    vN arj = u.r+v.r, aij = u.i+v.i; \
    br[j-1] = u.r-v.r; bi[j-1] = u.i-v.i; \
    CR[0] += arj; CI[0] += aij; \
    _Pragma("GCC unroll 16") \
    for(int k=1;k<=H;k++){ \
      vN wc = bc(CTj[j][k]); \
      CR[k] += wc*arj; CI[k] += wc*aij; \
    } \
  } \
  PUTM(p, (CN){CR[0], CI[0]}, domap, cp); \
  vN SR[H+1], SI[H+1]; \
  { vN b0r = br[0], b0i = bi[0]; \
    _Pragma("GCC unroll 16") \
    for(int k=1;k<=H;k++){ \
      vN ws = bc(STj[1][k]); \
      SR[k] = ws*b0r; SI[k] = ws*b0i; \
    } } \
  _Pragma("GCC unroll 16") \
  for(int j=2;j<=H;j++){ \
    vN brj = br[j-1], bij = bi[j-1]; \
    _Pragma("GCC unroll 16") \
    for(int k=1;k<=H;k++){ \
      vN ws = bc(STj[j][k]); \
      SR[k] += ws*brj; SI[k] += ws*bij; \
    } \
  } \
  _Pragma("GCC unroll 16") \
  for(int k=1;k<=H;k++){ \
    CN Xk, Xpk; \
    Xk.r  = CR[k] + SI[k]; Xk.i  = CI[k] - SR[k]; \
    Xpk.r = CR[k] - SI[k]; Xpk.i = CI[k] + SR[k]; \
    PUTM(p+(long)k*S, Xk, domap, cp+(long)k*S); \
    PUTM(p+(long)(P-k)*S, Xpk, domap, cp+(long)(P-k)*S); \
  } \
}
GEN_PRIME(dft13, 13, 6, C13t, S13t)
GEN_PRIME(dft17, 17, 8, C17t, S17t)
GEN_PRIME(dft23, 23, 11, C23t, S23t)
#undef GEN_PRIME

AI void dft36(cpN* restrict p, const long S, const int domap, const cpN* restrict cp){
  CN buf[36];
  _Pragma("GCC unroll 4")
  for(int n2=0;n2<4;n2++){
    CN x[9];
    _Pragma("GCC unroll 9")
    for(int n1=0;n1<9;n1++){ int n=(4*n1+9*n2)%36; x[n1] = LD(p + n*S); }
    DFT9V(x);
    _Pragma("GCC unroll 9")
    for(int k9=0;k9<9;k9++) buf[k9*4+n2] = x[k9];
  }
  _Pragma("GCC unroll 9")
  for(int k9=0;k9<9;k9++){
    CN* b = buf + k9*4;
    DFT4(&b[0],&b[1],&b[2],&b[3]);
    _Pragma("GCC unroll 4")
    for(int k4=0;k4<4;k4++){
      int k = (28*k9+9*k4)%36;
      PUTM(p+k*S, b[k4], domap, cp+(long)k*S);
    }
  }
}

AI void dft45(cpN* restrict p, const long S, const int domap, const cpN* restrict cp){
  CN buf[45];
  _Pragma("GCC unroll 5")
  for(int n2=0;n2<5;n2++){
    CN x[9];
    _Pragma("GCC unroll 9")
    for(int n1=0;n1<9;n1++){ int n=(5*n1+9*n2)%45; x[n1] = LD(p + n*S); }
    DFT9V(x);
    _Pragma("GCC unroll 9")
    for(int k9=0;k9<9;k9++) buf[k9*5+n2] = x[k9];
  }
  _Pragma("GCC unroll 9")
  for(int k9=0;k9<9;k9++){
    CN* b = buf + k9*5;
    DFT5(&b[0],&b[1],&b[2],&b[3],&b[4]);
    _Pragma("GCC unroll 5")
    for(int k5=0;k5<5;k5++){
      int k = (10*k9+36*k5)%45;
      PUTM(p+k*S, b[k5], domap, cp+(long)k*S);
    }
  }
}

AI void dft64(cpN* restrict p, const long S){
  CN buf[64];
  _Pragma("GCC unroll 8")
  for(int r=0;r<8;r++){
    CN x[8];
    _Pragma("GCC unroll 8")
    for(int t=0;t<8;t++){ int n=r+8*t; x[t] = LD(p + n*S); }
    DFT8V(x);
    _Pragma("GCC unroll 8")
    for(int km=0;km<8;km++) buf[km*8+r] = x[km];
  }
  _Pragma("GCC unroll 8")
  for(int km=0;km<8;km++){
    CN x[8];
    x[0] = buf[km*8+0];
    _Pragma("GCC unroll 8")
    for(int r=1;r<8;r++){
      int e = (r*km)&63;
      x[r] = (e==0) ? buf[km*8+r] : ctw(buf[km*8+r], W64r[e], W64i[e]);
    }
    DFT8V(x);
    _Pragma("GCC unroll 8")
    for(int q=0;q<8;q++){
      int k = km + 8*q;
      ST(p+k*S, x[q]);
    }
  }
}

#define PFROW(base, n) { const char* _q = (const char*)(base); for(int _i=0;_i<(int)((n)*sizeof(cpN)/64);_i++) __builtin_prefetch(_q + _i*64, 0, 3); }

// Scheme A (small L, cache-resident): buffer holds MAPPED state; map fused at store of last pass (y).
#define GEN_DRIVER_A(LV, DFT1, PX, PY) \
NOIN void FN(iter_##LV)(cpN* restrict b, const cpN* restrict c, const int premap){ \
  (void)premap; \
  for(int y=0;y<LV;y++){ \
    cpN* pl = b + (long)y*PY; \
    for(int x=0;x<LV;x++) DFT1(pl + (long)x*PX, 1, 0, 0); \
    for(int z=0;z<LV;z++) DFT1(pl + z, PX, 0, 0); \
  } \
  for(int x=0;x<LV;x++){ \
    cpN* pl = b + (long)x*PX; \
    const cpN* cl = c + (long)x*PX; \
    for(int z=0;z<LV;z++) DFT1(pl + z, PY, 1, cl + z); \
  } \
}

// Scheme B (large L): buffer holds RAW fft3; lazy map_row pipelined in z-phase.
#define GEN_DRIVER_B(LV, DFT1, PX, PY, DOPF) \
NOIN void FN(iter_##LV)(cpN* restrict b, const cpN* restrict c, const int premap){ \
  for(int y=0;y<LV;y++){ \
    cpN* pl = b + (long)y*PY; \
    const cpN* cl = c + (long)y*PY; \
    if(premap) map_row(pl, cl, LV); \
    for(int x=0;x<LV;x++){ \
      if(DOPF && x+2<LV){ const char* _q=(const char*)(pl + (long)(x+2)*PX); const char* _q2=(const char*)(cl + (long)(x+2)*PX); \
        for(int _t=0;_t<16;_t++){ __builtin_prefetch(_q+_t*64,0,3); if(premap) __builtin_prefetch(_q2+_t*64,0,3); } } \
      if(premap && x+1<LV) map_row(pl + (long)(x+1)*PX, cl + (long)(x+1)*PX, LV); \
      DFT1(pl + (long)x*PX, 1); \
    } \
    for(int z=0;z<LV;z++) DFT1(pl + z, PX); \
  } \
  for(int x=0;x<LV;x++){ \
    cpN* pl = b + (long)x*PX; \
    for(int z=0;z<LV;z++){ \
      if(DOPF && z+8<LV) { for(int yy=0; yy<LV; yy++){ const char* q=(const char*)(pl + z+8 + (long)yy*PY); __builtin_prefetch(q, 0, 3); if(sizeof(cpN)>64) __builtin_prefetch(q+64, 0, 3);} } \
      DFT1(pl + z, PY); \
    } \
  } \
}

GEN_DRIVER_A(6,  dft6,  PX6,  PY6)
GEN_DRIVER_A(8,  dft8,  PX8,  PY8)
GEN_DRIVER_A(13, dft13, PX13, PY13)
GEN_DRIVER_A(17, dft17, PX17, PY17)
GEN_DRIVER_A(23, dft23, PX23, PY23)
#define DFT36P(p,S) dft36(p,S,0,0)
#define DFT45P(p,S) dft45(p,S,0,0)
GEN_DRIVER_B(36, DFT36P, PX36, PY36, 1)
GEN_DRIVER_B(45, DFT45P, PX45, PY45, 1)

GEN_DRIVER_B(64, dft64, PX64, PY64, 1)
#undef GEN_DRIVER_A
#undef GEN_DRIVER_B


// ---- width-specific transpose helpers for IO ----
#if VW == 8
// transpose 8 rows (zmm each, from 8 volumes: 4 complex each) into 4 slots (re,im)
AI void FN(tr_fwd)(const double* base[8], long off, cpN* slot0){
  __m512d r0=_mm512_loadu_pd(base[0]+off), r1=_mm512_loadu_pd(base[1]+off),
          r2=_mm512_loadu_pd(base[2]+off), r3=_mm512_loadu_pd(base[3]+off),
          r4=_mm512_loadu_pd(base[4]+off), r5=_mm512_loadu_pd(base[5]+off),
          r6=_mm512_loadu_pd(base[6]+off), r7=_mm512_loadu_pd(base[7]+off);
  __m512d t0=_mm512_unpacklo_pd(r0,r1), t1=_mm512_unpackhi_pd(r0,r1);
  __m512d t2=_mm512_unpacklo_pd(r2,r3), t3=_mm512_unpackhi_pd(r2,r3);
  __m512d t4=_mm512_unpacklo_pd(r4,r5), t5=_mm512_unpackhi_pd(r4,r5);
  __m512d t6=_mm512_unpacklo_pd(r6,r7), t7=_mm512_unpackhi_pd(r6,r7);
  __m512d m0=_mm512_shuffle_f64x2(t0,t2,0x88), m1=_mm512_shuffle_f64x2(t0,t2,0xDD);
  __m512d m2=_mm512_shuffle_f64x2(t1,t3,0x88), m3=_mm512_shuffle_f64x2(t1,t3,0xDD);
  __m512d m4=_mm512_shuffle_f64x2(t4,t6,0x88), m5=_mm512_shuffle_f64x2(t4,t6,0xDD);
  __m512d m6=_mm512_shuffle_f64x2(t5,t7,0x88), m7=_mm512_shuffle_f64x2(t5,t7,0xDD);
  slot0[0].re=(vN)_mm512_shuffle_f64x2(m0,m4,0x88);
  slot0[0].im=(vN)_mm512_shuffle_f64x2(m2,m6,0x88);
  slot0[1].re=(vN)_mm512_shuffle_f64x2(m1,m5,0x88);
  slot0[1].im=(vN)_mm512_shuffle_f64x2(m3,m7,0x88);
  slot0[2].re=(vN)_mm512_shuffle_f64x2(m0,m4,0xDD);
  slot0[2].im=(vN)_mm512_shuffle_f64x2(m2,m6,0xDD);
  slot0[3].re=(vN)_mm512_shuffle_f64x2(m1,m5,0xDD);
  slot0[3].im=(vN)_mm512_shuffle_f64x2(m3,m7,0xDD);
}
AI void FN(tr_bwd)(CN s[4], double* base[8], long off, int lanes){
  __m512d r0=(__m512d)s[0].r, r1=(__m512d)s[0].i, r2=(__m512d)s[1].r, r3=(__m512d)s[1].i;
  __m512d r4=(__m512d)s[2].r, r5=(__m512d)s[2].i, r6=(__m512d)s[3].r, r7=(__m512d)s[3].i;
  __m512d t0=_mm512_unpacklo_pd(r0,r1), t1=_mm512_unpackhi_pd(r0,r1);
  __m512d t2=_mm512_unpacklo_pd(r2,r3), t3=_mm512_unpackhi_pd(r2,r3);
  __m512d t4=_mm512_unpacklo_pd(r4,r5), t5=_mm512_unpackhi_pd(r4,r5);
  __m512d t6=_mm512_unpacklo_pd(r6,r7), t7=_mm512_unpackhi_pd(r6,r7);
  __m512d m0=_mm512_shuffle_f64x2(t0,t2,0x88), m1=_mm512_shuffle_f64x2(t0,t2,0xDD);
  __m512d m2=_mm512_shuffle_f64x2(t1,t3,0x88), m3=_mm512_shuffle_f64x2(t1,t3,0xDD);
  __m512d m4=_mm512_shuffle_f64x2(t4,t6,0x88), m5=_mm512_shuffle_f64x2(t4,t6,0xDD);
  __m512d m6=_mm512_shuffle_f64x2(t5,t7,0x88), m7=_mm512_shuffle_f64x2(t5,t7,0xDD);
  __m512d o0=_mm512_shuffle_f64x2(m0,m4,0x88);
  __m512d o1=_mm512_shuffle_f64x2(m2,m6,0x88);
  __m512d o2=_mm512_shuffle_f64x2(m1,m5,0x88);
  __m512d o3=_mm512_shuffle_f64x2(m3,m7,0x88);
  __m512d o4=_mm512_shuffle_f64x2(m0,m4,0xDD);
  __m512d o5=_mm512_shuffle_f64x2(m2,m6,0xDD);
  __m512d o6=_mm512_shuffle_f64x2(m1,m5,0xDD);
  __m512d o7=_mm512_shuffle_f64x2(m3,m7,0xDD);
  switch(lanes){
    default: _mm512_storeu_pd(base[7]+off, o7); /* fallthrough */
    case 7: _mm512_storeu_pd(base[6]+off, o6);
    case 6: _mm512_storeu_pd(base[5]+off, o5);
    case 5: _mm512_storeu_pd(base[4]+off, o4);
    case 4: _mm512_storeu_pd(base[3]+off, o3);
    case 3: _mm512_storeu_pd(base[2]+off, o2);
    case 2: _mm512_storeu_pd(base[1]+off, o1);
    case 1: _mm512_storeu_pd(base[0]+off, o0);
  }
}
#define TRZ 4
#elif VW == 4
AI void FN(tr_fwd)(const double* base[4], long off, cpN* slot0){
  __m256d r0=_mm256_loadu_pd(base[0]+off), r1=_mm256_loadu_pd(base[1]+off),
          r2=_mm256_loadu_pd(base[2]+off), r3=_mm256_loadu_pd(base[3]+off);
  __m256d t0=_mm256_unpacklo_pd(r0,r1), t1=_mm256_unpackhi_pd(r0,r1);
  __m256d t2=_mm256_unpacklo_pd(r2,r3), t3=_mm256_unpackhi_pd(r2,r3);
  slot0[0].re=(vN)_mm256_permute2f128_pd(t0,t2,0x20);
  slot0[0].im=(vN)_mm256_permute2f128_pd(t1,t3,0x20);
  slot0[1].re=(vN)_mm256_permute2f128_pd(t0,t2,0x31);
  slot0[1].im=(vN)_mm256_permute2f128_pd(t1,t3,0x31);
}
AI void FN(tr_bwd)(CN s[2], double* base[4], long off, int lanes){
  __m256d r0=(__m256d)s[0].r, r1=(__m256d)s[0].i, r2=(__m256d)s[1].r, r3=(__m256d)s[1].i;
  __m256d t0=_mm256_unpacklo_pd(r0,r1), t1=_mm256_unpackhi_pd(r0,r1);
  __m256d t2=_mm256_unpacklo_pd(r2,r3), t3=_mm256_unpackhi_pd(r2,r3);
  __m256d o0=_mm256_permute2f128_pd(t0,t2,0x20);
  __m256d o1=_mm256_permute2f128_pd(t1,t3,0x20);
  __m256d o2=_mm256_permute2f128_pd(t0,t2,0x31);
  __m256d o3=_mm256_permute2f128_pd(t1,t3,0x31);
  switch(lanes){
    default: _mm256_storeu_pd(base[3]+off, o3);
    case 3: _mm256_storeu_pd(base[2]+off, o2);
    case 2: _mm256_storeu_pd(base[1]+off, o1);
    case 1: _mm256_storeu_pd(base[0]+off, o0);
  }
}
#define TRZ 2
#else
AI void FN(tr_fwd)(const double* base[2], long off, cpN* slot0){
  __m128d r0=_mm_loadu_pd(base[0]+off), r1=_mm_loadu_pd(base[1]+off);
  slot0[0].re=(vN)_mm_unpacklo_pd(r0,r1);
  slot0[0].im=(vN)_mm_unpackhi_pd(r0,r1);
}
AI void FN(tr_bwd)(CN s[1], double* base[2], long off, int lanes){
  __m128d o0=_mm_unpacklo_pd((__m128d)s[0].r,(__m128d)s[0].i);
  __m128d o1=_mm_unpackhi_pd((__m128d)s[0].r,(__m128d)s[0].i);
  if(lanes>1) _mm_storeu_pd(base[1]+off, o1);
  _mm_storeu_pd(base[0]+off, o0);
}
#define TRZ 1
#endif

static double FN(zero_page)[VW*2*TRZ];

NOIN void FN(interleave_in)(const double* restrict src, void* restrict bufv, int L, long PX, long PY, int lanes, long vol3){
  cpN* buf = (cpN*)bufv;
  const double* base[VW];
  for(int v=0;v<VW;v++) base[v] = (v<lanes) ? (src + (long)v*vol3*2) : 0;
  const int NZ = (L/TRZ)*TRZ;
  for(int x=0;x<L;x++) for(int y=0;y<L;y++){
    cpN* row = buf + (long)x*PX + (long)y*PY;
    long e0 = ((long)x*L+y)*L;
    if(lanes==VW){
      for(int z=0;z<NZ;z+=TRZ) FN(tr_fwd)(base, (e0+z)*2, row+z);
      for(int z=NZ;z<L;z++){
        cpN s;
        for(int v=0;v<VW;v++){ const double* q = base[v] + (e0+z)*2; s.re[v]=q[0]; s.im[v]=q[1]; }
        row[z]=s;
      }
    } else {
      for(int z=0;z<L;z++){
        cpN s;
        for(int v=0;v<VW;v++){
          if(v<lanes){ const double* q = base[v] + (e0+z)*2; s.re[v]=q[0]; s.im[v]=q[1]; }
          else { s.re[v]=0.0; s.im[v]=0.0; }
        }
        row[z]=s;
      }
    }
  }
}

NOIN void FN(snapshot_out)(const void* restrict bufv, double* restrict dst, int L, long PX, long PY, int lanes, long vol3){
  const cpN* buf = (const cpN*)bufv;
  double* base[VW];
  for(int v=0;v<VW;v++) base[v] = (v<lanes) ? (dst + (long)v*vol3*2) : FN(zero_page);
  const int NZ = (L/TRZ)*TRZ;
  for(int x=0;x<L;x++) for(int y=0;y<L;y++){
    const cpN* row = buf + (long)x*PX + (long)y*PY;
    long e0 = ((long)x*L+y)*L;
    for(int z=0;z<NZ;z+=TRZ){
      CN s[TRZ];
      _Pragma("GCC unroll 4")
      for(int t=0;t<TRZ;t++) s[t] = LD(row+z+t);
      if(lanes==VW) FN(tr_bwd)(s, base, (e0+z)*2, lanes);
      else {
        for(int t=0;t<TRZ;t++)
          for(int v=0;v<lanes;v++){ double* q = dst + ((long)v*vol3 + e0+z+t)*2; q[0]=s[t].r[v]; q[1]=s[t].i[v]; }
      }
    }
    for(int z=NZ;z<L;z++){
      CN sv = LD(row+z);
      for(int v=0;v<lanes;v++){
        double* q = dst + ((long)v*vol3 + e0 + z)*2;
        q[0]=sv.r[v]; q[1]=sv.i[v];
      }
    }
  }
}

NOIN void FN(materialize_out)(const void* restrict bufv, const void* restrict cbufv, double* restrict dst, int L, long PX, long PY, int lanes, long vol3){
  const cpN* buf = (const cpN*)bufv;
  const cpN* cbuf = (const cpN*)cbufv;
  double* base[VW];
  for(int v=0;v<VW;v++) base[v] = (v<lanes) ? (dst + (long)v*vol3*2) : FN(zero_page);
  const int NZ = (L/TRZ)*TRZ;
  for(int x=0;x<L;x++) for(int y=0;y<L;y++){
    const cpN* row = buf + (long)x*PX + (long)y*PY;
    const cpN* crow = cbuf + (long)x*PX + (long)y*PY;
    long e0 = ((long)x*L+y)*L;
    for(int z=0;z<NZ;z+=TRZ){
      CN s[TRZ];
      _Pragma("GCC unroll 4")
      for(int t=0;t<TRZ;t++) s[t] = mapc(LD(row+z+t), crow+z+t);
      if(lanes==VW) FN(tr_bwd)(s, base, (e0+z)*2, lanes);
      else {
        for(int t=0;t<TRZ;t++)
          for(int v=0;v<lanes;v++){ double* q = dst + ((long)v*vol3 + e0+z+t)*2; q[0]=s[t].r[v]; q[1]=s[t].i[v]; }
      }
    }
    for(int z=NZ;z<L;z++){
      CN sv = mapc(LD(row+z), crow+z);
      for(int v=0;v<lanes;v++){
        double* q = dst + ((long)v*vol3 + e0 + z)*2;
        q[0]=sv.r[v]; q[1]=sv.i[v];
      }
    }
  }
}

// wrappers with typed buffer pointers for the dispatcher
#define DEFINE_ITER_WRAP(LV) \
NOIN void FN(iterw_##LV)(void* b, const void* c, int premap){ FN(iter_##LV)((cpN*)b, (const cpN*)c, premap); }
DEFINE_ITER_WRAP(6)
DEFINE_ITER_WRAP(8)
DEFINE_ITER_WRAP(13)
DEFINE_ITER_WRAP(17)
DEFINE_ITER_WRAP(23)
DEFINE_ITER_WRAP(36)
DEFINE_ITER_WRAP(45)
DEFINE_ITER_WRAP(64)
#undef DEFINE_ITER_WRAP

#undef bc
#undef LD
#undef ST
#undef cadd
#undef csub
#undef ctw
#undef cmi
#undef mapc
#undef PUTM
#undef map_row
#undef DFT2
#undef DFT3
#undef DFT4
#undef DFT5
#undef DFT8V
#undef DFT9V
#undef dft6
#undef dft8
#undef dft13
#undef dft17
#undef dft23
#undef dft36
#undef dft45
#undef dft64
#undef PFROW
#undef TRZ
#undef BCAST
#undef VRSQRT14
#undef FN
#undef vN
#undef cpN
#undef CN

#endif
