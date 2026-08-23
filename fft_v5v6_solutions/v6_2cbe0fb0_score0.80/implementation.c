// ============================================================================
// Iterated batched 3D complex FFTs, specialized for L in {6,8,13,17,23,36,45,64}
//
// All transform arithmetic is implemented here from scratch (no FFT libraries):
//   - L=6:  Cooley-Tukey 2x3 codelet
//   - L=8:  radix-2 DFT8 codelet (DFT4-based)
//   - L=13/17/23: symmetric half-matmul DFT (even/odd folding, cos/sin tables
//                 computed at init in long double; pure FMA accumulation)
//   - L=36: Cooley-Tukey 6x6 with twiddles
//   - L=45: prime-factor (PFA) 9x5, DFT9 = 3x3 CT, DFT5 codelet, no twiddles
//   - L=64: Cooley-Tukey 8x8 with twiddles; the z-axis uses a fused codelet
//           with one in-register 8x8 transpose between the two radix-8 stages
//
// Data layout: split re/im planes of SIMD vectors (AVX-512, 8 doubles).
//   Two execution paths chosen at runtime:
//   - batch-interleaved (8 volumes in SIMD lanes) for L<=36 when B>=8
//   - per-volume with padded rows + 8x8 transposed z-tiles otherwise
// Iteration structure keeps each volume/group cache-resident across all m
// steps; pass order per step is (Y, X, Z) so that at step boundaries the
// contiguous z-pass completes the previous FFT, the nonlinear map
// x = z/(1+|z|) (computed with rsqrt14/rcp14 + Newton iterations to full
// double precision; no divider ops) is applied plane-locally, and outputs
// for steps 1 and m are captured on the fly. 'c' is read directly from the
// caller's interleaved array with on-the-fly masked deinterleaving.
// Single-threaded throughout.
// ============================================================================
// Specialized batched 3D complex FFT + nonlinear map iteration
// for L in {6,8,13,17,23,36,45,64}. AVX-512, single thread.
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <sys/mman.h>

typedef double v8 __attribute__((vector_size(64),aligned(64)));
#define AI __attribute__((always_inline)) inline

// ---------------- constants / tables ----------------
#define CS3 0.8660254037844386467637231707529362
#define CR2 0.7071067811865475244008443621048490
#define C51 0.3090169943749474241022934171828191
#define C52 (-0.8090169943749474241022934171828191)
#define S51 0.9510565162951535721164393333793821
#define S52 0.5877852522924731291687059546390728

static double C13T[6][6],  S13T[6][6];
static double C17T[8][8],  S17T[8][8];
static double C23T[11][11],S23T[11][11];
static double TW36R[6][6], TW36I[6][6];
static double TW64R[8][8], TW64I[8][8];
static v8 VTW64R[8], VTW64I[8]; // lane b: w64^{a*b} for vector row a
static double W9R[5], W9I[5];

static void fill_prime(int L, int h, double*Ct, double*St, int ld){
    for (int k=1;k<=h;k++) for (int j=1;j<=h;j++){
        long mm = ((long)k*j) % L;
        long double ang = 2.0L*3.14159265358979323846264338327950288L*(long double)mm/(long double)L;
        Ct[(k-1)*ld + (j-1)] = (double)cosl(ang);
        St[(k-1)*ld + (j-1)] = (double)sinl(ang);
    }
}
static void fill_tw(int L, int n, double*Tr, double*Ti){
    for (int a=0;a<n;a++) for (int b=0;b<n;b++){
        long mm = ((long)a*b) % L;
        long double ang = 2.0L*3.14159265358979323846264338327950288L*(long double)mm/(long double)L;
        Tr[a*n+b] = (double)cosl(ang);
        Ti[a*n+b] = -(double)sinl(ang);
    }
}
void ns_init(void){
    fill_prime(13, 6,  &C13T[0][0], &S13T[0][0], 6);
    fill_prime(17, 8,  &C17T[0][0], &S17T[0][0], 8);
    fill_prime(23, 11, &C23T[0][0], &S23T[0][0], 11);
    fill_tw(36, 6, &TW36R[0][0], &TW36I[0][0]);
    fill_tw(64, 8, &TW64R[0][0], &TW64I[0][0]);
    for (int a=0;a<8;a++) for (int b=0;b<8;b++){ VTW64R[a][b]=TW64R[a][b]; VTW64I[a][b]=TW64I[a][b]; }
    for (int e=1;e<=4;e++){
        if (e==3) continue;
        long double ang = 2.0L*3.14159265358979323846264338327950288L*(long double)e/9.0L;
        W9R[e] = (double)cosl(ang); W9I[e] = -(double)sinl(ang);
    }
}

// ---------------- small DFT building blocks (arrays of v8, arbitrary index exprs) ----------------
#define MDFT3(Xr,Xi,Yr,Yi, i0,i1,i2, o0,o1,o2) do{ \
  v8 _tr=Xr[i1]+Xr[i2], _ti=Xi[i1]+Xi[i2]; \
  v8 _ur=Xr[i1]-Xr[i2], _ui=Xi[i1]-Xi[i2]; \
  v8 _x0r=Xr[i0], _x0i=Xi[i0]; \
  v8 _mr=_x0r-0.5*_tr, _mi=_x0i-0.5*_ti; \
  v8 _vr=CS3*_ur, _vi=CS3*_ui; \
  Yr[o0]=_x0r+_tr; Yi[o0]=_x0i+_ti; \
  Yr[o1]=_mr+_vi; Yi[o1]=_mi-_vr; \
  Yr[o2]=_mr-_vi; Yi[o2]=_mi+_vr; \
}while(0)

#define MDFT4(Xr,Xi,Yr,Yi, i0,i1,i2,i3, o0,o1,o2,o3) do{ \
  v8 _t0r=Xr[i0]+Xr[i2], _t0i=Xi[i0]+Xi[i2]; \
  v8 _t1r=Xr[i0]-Xr[i2], _t1i=Xi[i0]-Xi[i2]; \
  v8 _t2r=Xr[i1]+Xr[i3], _t2i=Xi[i1]+Xi[i3]; \
  v8 _t3r=Xr[i1]-Xr[i3], _t3i=Xi[i1]-Xi[i3]; \
  Yr[o0]=_t0r+_t2r; Yi[o0]=_t0i+_t2i; \
  Yr[o2]=_t0r-_t2r; Yi[o2]=_t0i-_t2i; \
  Yr[o1]=_t1r+_t3i; Yi[o1]=_t1i-_t3r; \
  Yr[o3]=_t1r-_t3i; Yi[o3]=_t1i+_t3r; \
}while(0)

#define MDFT5(Xr,Xi,Yr,Yi, i0,i1,i2,i3,i4, o0,o1,o2,o3,o4) do{ \
  v8 _t1r=Xr[i1]+Xr[i4], _t1i=Xi[i1]+Xi[i4]; \
  v8 _t2r=Xr[i2]+Xr[i3], _t2i=Xi[i2]+Xi[i3]; \
  v8 _t3r=Xr[i1]-Xr[i4], _t3i=Xi[i1]-Xi[i4]; \
  v8 _t4r=Xr[i2]-Xr[i3], _t4i=Xi[i2]-Xi[i3]; \
  v8 _x0r=Xr[i0], _x0i=Xi[i0]; \
  Yr[o0]=_x0r+_t1r+_t2r; Yi[o0]=_x0i+_t1i+_t2i; \
  v8 _a1r=_x0r+C51*_t1r+C52*_t2r, _a1i=_x0i+C51*_t1i+C52*_t2i; \
  v8 _a2r=_x0r+C52*_t1r+C51*_t2r, _a2i=_x0i+C52*_t1i+C51*_t2i; \
  v8 _b1r=S51*_t3r+S52*_t4r, _b1i=S51*_t3i+S52*_t4i; \
  v8 _b2r=S52*_t3r-S51*_t4r, _b2i=S52*_t3i-S51*_t4i; \
  Yr[o1]=_a1r+_b1i; Yi[o1]=_a1i-_b1r; \
  Yr[o4]=_a1r-_b1i; Yi[o4]=_a1i+_b1r; \
  Yr[o2]=_a2r+_b2i; Yi[o2]=_a2i-_b2r; \
  Yr[o3]=_a2r-_b2i; Yi[o3]=_a2i+_b2r; \
}while(0)

#define MDFT6(Xr,Xi,Yr,Yi, i0,i1,i2,i3,i4,i5, o0,o1,o2,o3,o4,o5) do{ \
  v8 _Er[3],_Ei[3],_Or[3],_Oi[3]; \
  MDFT3(Xr,Xi,_Er,_Ei, i0,i2,i4, 0,1,2); \
  MDFT3(Xr,Xi,_Or,_Oi, i1,i3,i5, 0,1,2); \
  Yr[o0]=_Er[0]+_Or[0]; Yi[o0]=_Ei[0]+_Oi[0]; \
  Yr[o3]=_Er[0]-_Or[0]; Yi[o3]=_Ei[0]-_Oi[0]; \
  v8 _p1r=0.5*_Or[1]+CS3*_Oi[1], _p1i=0.5*_Oi[1]-CS3*_Or[1]; \
  Yr[o1]=_Er[1]+_p1r; Yi[o1]=_Ei[1]+_p1i; \
  Yr[o4]=_Er[1]-_p1r; Yi[o4]=_Ei[1]-_p1i; \
  v8 _p2r=CS3*_Oi[2]-0.5*_Or[2], _p2i=-0.5*_Oi[2]-CS3*_Or[2]; \
  Yr[o2]=_Er[2]+_p2r; Yi[o2]=_Ei[2]+_p2i; \
  Yr[o5]=_Er[2]-_p2r; Yi[o5]=_Ei[2]-_p2i; \
}while(0)

#define MDFT8(Xr,Xi,Yr,Yi, i0,i1,i2,i3,i4,i5,i6,i7, o0,o1,o2,o3,o4,o5,o6,o7) do{ \
  v8 _Er[4],_Ei[4],_Or[4],_Oi[4]; \
  MDFT4(Xr,Xi,_Er,_Ei, i0,i2,i4,i6, 0,1,2,3); \
  MDFT4(Xr,Xi,_Or,_Oi, i1,i3,i5,i7, 0,1,2,3); \
  Yr[o0]=_Er[0]+_Or[0]; Yi[o0]=_Ei[0]+_Oi[0]; \
  Yr[o4]=_Er[0]-_Or[0]; Yi[o4]=_Ei[0]-_Oi[0]; \
  v8 _p1r=CR2*(_Or[1]+_Oi[1]), _p1i=CR2*(_Oi[1]-_Or[1]); \
  Yr[o1]=_Er[1]+_p1r; Yi[o1]=_Ei[1]+_p1i; \
  Yr[o5]=_Er[1]-_p1r; Yi[o5]=_Ei[1]-_p1i; \
  Yr[o2]=_Er[2]+_Oi[2]; Yi[o2]=_Ei[2]-_Or[2]; \
  Yr[o6]=_Er[2]-_Oi[2]; Yi[o6]=_Ei[2]+_Or[2]; \
  v8 _p3r=CR2*(_Oi[3]-_Or[3]), _p3i=-CR2*(_Or[3]+_Oi[3]); \
  Yr[o3]=_Er[3]+_p3r; Yi[o3]=_Ei[3]+_p3i; \
  Yr[o7]=_Er[3]-_p3r; Yi[o7]=_Ei[3]-_p3i; \
}while(0)

#define CMULW(Sr,Si,idx,wr,wi) do{ v8 _cr=Sr[idx], _ci=Si[idx]; \
  Sr[idx]=_cr*(wr)-_ci*(wi); Si[idx]=_cr*(wi)+_ci*(wr); }while(0)

#define MDFT9(Xr,Xi,Yr,Yi, i0,i1,i2,i3,i4,i5,i6,i7,i8, o0,o1,o2,o3,o4,o5,o6,o7,o8) do{ \
  v8 _Sr[9],_Si[9]; \
  MDFT3(Xr,Xi,_Sr,_Si, i0,i3,i6, 0,3,6); \
  MDFT3(Xr,Xi,_Sr,_Si, i1,i4,i7, 1,4,7); \
  MDFT3(Xr,Xi,_Sr,_Si, i2,i5,i8, 2,5,8); \
  CMULW(_Sr,_Si,4,W9R[1],W9I[1]); \
  CMULW(_Sr,_Si,5,W9R[2],W9I[2]); \
  CMULW(_Sr,_Si,7,W9R[2],W9I[2]); \
  CMULW(_Sr,_Si,8,W9R[4],W9I[4]); \
  MDFT3(_Sr,_Si,Yr,Yi, 0,1,2, o0,o3,o6); \
  MDFT3(_Sr,_Si,Yr,Yi, 3,4,5, o1,o4,o7); \
  MDFT3(_Sr,_Si,Yr,Yi, 6,7,8, o2,o5,o8); \
}while(0)

// ---------------- pointwise map ----------------
#define PWCORE(xr_out,xi_out, zr_,zi_) \
  v8 _zr=(zr_), _zi=(zi_); \
  v8 _m2=_zr*_zr+(_zi*_zi+1e-300); \
  v8 _u=(v8)_mm512_rsqrt14_pd((__m512d)_m2); \
  v8 _h=0.5*_m2; \
  _u=_u*(1.5-_h*_u*_u); _u=_u*(1.5-_h*_u*_u); \
  v8 _d=1.0+_m2*_u; \
  v8 _r=(v8)_mm512_rcp14_pd((__m512d)_d); \
  _r=_r*(2.0-_d*_r); _r=_r*(2.0-_d*_r); \
  v8 xr_out=_zr*_r, xi_out=_zi*_r;

#define PWSTORE(rp,ip,vidx, yr_,yi_, cpr,cpi) do{ \
  long _vx=(vidx); \
  PWCORE(_xr,_xi, (yr_)+cpr[_vx], (yi_)+cpi[_vx]) \
  rp[_vx]=_xr; ip[_vx]=_xi; \
}while(0)

// ---------------- line codelets: v8* with stride sv (in vecs) ----------------
#define LINE_PROLOGUE(NAME) \
static AI void NAME##_impl(v8*restrict re, v8*restrict im, \
                           const v8*restrict cr, const v8*restrict ci, \
                           const long sv, const int PW, const int PF)

#define PFNEXT(p,idx) _mm_prefetch((const char*)((p)+(idx))+64, _MM_HINT_T0)

#define LINE_WRAPPERS(NAME) \
static void NAME(v8*restrict re, v8*restrict im, long sv){ NAME##_impl(re,im,0,0,sv,0,0); } \
static void NAME##_pf(v8*restrict re, v8*restrict im, long sv){ NAME##_impl(re,im,0,0,sv,0,1); } \
static void NAME##_zpw(v8*restrict re, v8*restrict im, const v8*restrict cr, const v8*restrict ci, long sv){ NAME##_impl(re,im,cr,ci,sv,1,0); }

LINE_PROLOGUE(line6){
    v8 yr[6],yi[6];
    if (PF) for (int n=0;n<6;n++){ PFNEXT(re,n*sv); PFNEXT(im,n*sv); }
    if (PW){
        v8 xr[6],xi[6];
        for (int n=0;n<6;n++){
            PWCORE(ur,ui, re[n*sv]+cr[n*sv], im[n*sv]+ci[n*sv])
            xr[n]=ur; xi[n]=ui;
        }
        MDFT6(xr,xi,yr,yi, 0,1,2,3,4,5, 0,1,2,3,4,5);
    } else {
        MDFT6(re,im,yr,yi, 0,sv,2*sv,3*sv,4*sv,5*sv, 0,1,2,3,4,5);
    }
    for (int n=0;n<6;n++){ re[n*sv]=yr[n]; im[n*sv]=yi[n]; }
}
LINE_WRAPPERS(line6)

LINE_PROLOGUE(line8){
    v8 yr[8],yi[8];
    if (PF) for (int n=0;n<8;n++){ PFNEXT(re,n*sv); PFNEXT(im,n*sv); }
    if (PW){
        v8 xr[8],xi[8];
        for (int n=0;n<8;n++){
            PWCORE(ur,ui, re[n*sv]+cr[n*sv], im[n*sv]+ci[n*sv])
            xr[n]=ur; xi[n]=ui;
        }
        MDFT8(xr,xi,yr,yi, 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7);
    } else {
        MDFT8(re,im,yr,yi, 0,sv,2*sv,3*sv,4*sv,5*sv,6*sv,7*sv, 0,1,2,3,4,5,6,7);
    }
    for (int n=0;n<8;n++){ re[n*sv]=yr[n]; im[n*sv]=yi[n]; }
}
LINE_WRAPPERS(line8)

#define PRIME_LINE(NAME, LSZ, H, CT, ST) \
LINE_PROLOGUE(NAME){ \
    v8 er[H],ei[H],odr[H],odi[H]; \
    if (PF) for (int n=0;n<LSZ;n++){ PFNEXT(re,n*sv); PFNEXT(im,n*sv); } \
    v8 x0r, x0i; \
    if (PW){ PWCORE(ur,ui, re[0]+cr[0], im[0]+ci[0]) x0r=ur; x0i=ui; } \
    else { x0r=re[0]; x0i=im[0]; } \
    for (int j=1;j<=H;j++){ \
        v8 ar,ai,br,bi; \
        if (PW){ \
            { PWCORE(ur,ui, re[j*sv]+cr[j*sv], im[j*sv]+ci[j*sv]) ar=ur; ai=ui; } \
            { PWCORE(ur,ui, re[(LSZ-j)*sv]+cr[(LSZ-j)*sv], im[(LSZ-j)*sv]+ci[(LSZ-j)*sv]) br=ur; bi=ui; } \
        } else { \
            ar=re[j*sv]; ai=im[j*sv]; br=re[(LSZ-j)*sv]; bi=im[(LSZ-j)*sv]; \
        } \
        er[j-1]=ar+br; ei[j-1]=ai+bi; odr[j-1]=ar-br; odi[j-1]=ai-bi; \
    } \
    v8 X0r=x0r, X0i=x0i; \
    for (int j=0;j<H;j++){ X0r+=er[j]; X0i+=ei[j]; } \
    re[0]=X0r; im[0]=X0i; \
    for (int k=1;k<=H;k++){ \
        v8 ar=x0r, ai=x0i; \
        v8 br=ST[k-1][0]*odr[0], bi=ST[k-1][0]*odi[0]; \
        ar+=CT[k-1][0]*er[0]; ai+=CT[k-1][0]*ei[0]; \
        for (int j=1;j<H;j++){ \
            double cc=CT[k-1][j], ss=ST[k-1][j]; \
            ar+=cc*er[j]; ai+=cc*ei[j]; \
            br+=ss*odr[j]; bi+=ss*odi[j]; \
        } \
        re[k*sv]=ar+bi; im[k*sv]=ai-br; \
        re[(LSZ-k)*sv]=ar-bi; im[(LSZ-k)*sv]=ai+br; \
    } \
} \
LINE_WRAPPERS(NAME)


// two-phase register-blocked prime codelet (for larger H): e/o block in regs, partial accums in scratch
#define PRIME_LINE2(NAME, LSZ, H, JB, CT, ST) \
LINE_PROLOGUE(NAME){ \
    v8 x0r, x0i; \
    if (PF) for (int n=0;n<LSZ;n++){ PFNEXT(re,n*sv); PFNEXT(im,n*sv); } \
    if (PW){ PWCORE(ur,ui, re[0]+cr[0], im[0]+ci[0]) x0r=ur; x0i=ui; } \
    else { x0r=re[0]; x0i=im[0]; } \
    v8 Ar[H],Ai[H],Br[H],Bi[H]; \
    v8 sumr=x0r, sumi=x0i; \
    { \
        v8 er[JB],ei[JB],odr[JB],odi[JB]; \
        _Pragma("GCC unroll 16") \
        for (int j=1;j<=JB;j++){ \
            v8 ar,ai,br,bi; \
            if (PW){ \
                { PWCORE(ur,ui, re[j*sv]+cr[j*sv], im[j*sv]+ci[j*sv]) ar=ur; ai=ui; } \
                { PWCORE(ur,ui, re[(LSZ-j)*sv]+cr[(LSZ-j)*sv], im[(LSZ-j)*sv]+ci[(LSZ-j)*sv]) br=ur; bi=ui; } \
            } else { ar=re[j*sv]; ai=im[j*sv]; br=re[(LSZ-j)*sv]; bi=im[(LSZ-j)*sv]; } \
            er[j-1]=ar+br; ei[j-1]=ai+bi; odr[j-1]=ar-br; odi[j-1]=ai-bi; \
            sumr+=er[j-1]; sumi+=ei[j-1]; \
        } \
        for (int k=1;k<=H;k++){ \
            v8 ar=x0r, ai=x0i; \
            v8 br=ST[k-1][0]*odr[0], bi=ST[k-1][0]*odi[0]; \
            ar+=CT[k-1][0]*er[0]; ai+=CT[k-1][0]*ei[0]; \
            _Pragma("GCC unroll 16") \
            for (int j=1;j<JB;j++){ \
                ar+=CT[k-1][j]*er[j]; ai+=CT[k-1][j]*ei[j]; \
                br+=ST[k-1][j]*odr[j]; bi+=ST[k-1][j]*odi[j]; \
            } \
            Ar[k-1]=ar; Ai[k-1]=ai; Br[k-1]=br; Bi[k-1]=bi; \
        } \
    } \
    { \
        v8 er[H-JB],ei[H-JB],odr[H-JB],odi[H-JB]; \
        _Pragma("GCC unroll 16") \
        for (int j=JB+1;j<=H;j++){ \
            v8 ar,ai,br,bi; \
            if (PW){ \
                { PWCORE(ur,ui, re[j*sv]+cr[j*sv], im[j*sv]+ci[j*sv]) ar=ur; ai=ui; } \
                { PWCORE(ur,ui, re[(LSZ-j)*sv]+cr[(LSZ-j)*sv], im[(LSZ-j)*sv]+ci[(LSZ-j)*sv]) br=ur; bi=ui; } \
            } else { ar=re[j*sv]; ai=im[j*sv]; br=re[(LSZ-j)*sv]; bi=im[(LSZ-j)*sv]; } \
            er[j-JB-1]=ar+br; ei[j-JB-1]=ai+bi; odr[j-JB-1]=ar-br; odi[j-JB-1]=ai-bi; \
            sumr+=er[j-JB-1]; sumi+=ei[j-JB-1]; \
        } \
        re[0]=sumr; im[0]=sumi; \
        for (int k=1;k<=H;k++){ \
            v8 ar=Ar[k-1], ai=Ai[k-1], br=Br[k-1], bi=Bi[k-1]; \
            _Pragma("GCC unroll 16") \
            for (int j=JB;j<H;j++){ \
                ar+=CT[k-1][j]*er[j-JB]; ai+=CT[k-1][j]*ei[j-JB]; \
                br+=ST[k-1][j]*odr[j-JB]; bi+=ST[k-1][j]*odi[j-JB]; \
            } \
            re[k*sv]=ar+bi; im[k*sv]=ai-br; \
            re[(LSZ-k)*sv]=ar-bi; im[(LSZ-k)*sv]=ai+br; \
        } \
    } \
} \
LINE_WRAPPERS(NAME)

PRIME_LINE(line13, 13, 6,  C13T, S13T)
PRIME_LINE(line17, 17, 8,  C17T, S17T)
PRIME_LINE2(line23, 23, 11, 6, C23T, S23T)

LINE_PROLOGUE(line36){
    v8 Tr[36],Ti[36];
    if (PF) for (int n=0;n<36;n++){ PFNEXT(re,n*sv); PFNEXT(im,n*sv); }
    v8 pxr[36],pxi[36];
    if (PW){
        for (int n=0;n<36;n++){
            PWCORE(ur,ui, re[n*sv]+cr[n*sv], im[n*sv]+ci[n*sv])
            pxr[n]=ur; pxi[n]=ui;
        }
    }
    for (int b=0;b<6;b++){
        v8 gr[6],gi[6];
        if (PW) MDFT6(pxr,pxi,gr,gi, b,6+b,12+b,18+b,24+b,30+b, 0,1,2,3,4,5);
        else MDFT6(re,im,gr,gi, b*sv,(6+b)*sv,(12+b)*sv,(18+b)*sv,(24+b)*sv,(30+b)*sv, 0,1,2,3,4,5);
        Tr[b]=gr[0]; Ti[b]=gi[0];
        for (int a=1;a<6;a++){
            if (b==0){ Tr[a*6]=gr[a]; Ti[a*6]=gi[a]; }
            else {
                double wr=TW36R[a][b], wi=TW36I[a][b];
                Tr[a*6+b]=gr[a]*wr-gi[a]*wi; Ti[a*6+b]=gr[a]*wi+gi[a]*wr;
            }
        }
    }
    for (int a=0;a<6;a++){
        v8 yr[6],yi[6];
        MDFT6(Tr,Ti,yr,yi, a*6+0,a*6+1,a*6+2,a*6+3,a*6+4,a*6+5, 0,1,2,3,4,5);
        for (int bb=0;bb<6;bb++){
            long o=(long)(6*bb+a)*sv;
            re[o]=yr[bb]; im[o]=yi[bb];
        }
    }
}
LINE_WRAPPERS(line36)

static const int IN45[5][9] = {
 {0,5,10,15,20,25,30,35,40},
 {9,14,19,24,29,34,39,44,4},
 {18,23,28,33,38,43,3,8,13},
 {27,32,37,42,2,7,12,17,22},
 {36,41,1,6,11,16,21,26,31}};
static const int OUT45[9][5] = {
 {0,36,27,18,9},{10,1,37,28,19},{20,11,2,38,29},{30,21,12,3,39},
 {40,31,22,13,4},{5,41,32,23,14},{15,6,42,33,24},{25,16,7,43,34},
 {35,26,17,8,44}};

LINE_PROLOGUE(line45){
    v8 Gr[45],Gi[45];
    if (PF) for (int n=0;n<45;n++){ PFNEXT(re,n*sv); PFNEXT(im,n*sv); }
    v8 pxr[45],pxi[45];
    if (PW){
        for (int n=0;n<45;n++){
            PWCORE(ur,ui, re[n*sv]+cr[n*sv], im[n*sv]+ci[n*sv])
            pxr[n]=ur; pxi[n]=ui;
        }
    }
    for (int b=0;b<5;b++){
        if (PW) MDFT9(pxr,pxi,Gr,Gi,
          IN45[b][0],IN45[b][1],IN45[b][2],IN45[b][3],IN45[b][4],
          IN45[b][5],IN45[b][6],IN45[b][7],IN45[b][8],
          b, 5+b, 10+b, 15+b, 20+b, 25+b, 30+b, 35+b, 40+b);
        else MDFT9(re,im,Gr,Gi,
          IN45[b][0]*sv,IN45[b][1]*sv,IN45[b][2]*sv,IN45[b][3]*sv,IN45[b][4]*sv,
          IN45[b][5]*sv,IN45[b][6]*sv,IN45[b][7]*sv,IN45[b][8]*sv,
          b, 5+b, 10+b, 15+b, 20+b, 25+b, 30+b, 35+b, 40+b);
    }
    for (int a=0;a<9;a++){
        v8 yr[5],yi[5];
        MDFT5(Gr,Gi,yr,yi, a*5+0,a*5+1,a*5+2,a*5+3,a*5+4, 0,1,2,3,4);
        for (int bb=0;bb<5;bb++){
            long o=(long)OUT45[a][bb]*sv;
            re[o]=yr[bb]; im[o]=yi[bb];
        }
    }
}
LINE_WRAPPERS(line45)

LINE_PROLOGUE(line64){
    v8 Tr[64],Ti[64];
    if (PF) for (int n=0;n<64;n++){ PFNEXT(re,n*sv); PFNEXT(im,n*sv); }
    v8 pxr[64],pxi[64];
    if (PW){
        for (int n=0;n<64;n++){
            PWCORE(ur,ui, re[n*sv]+cr[n*sv], im[n*sv]+ci[n*sv])
            pxr[n]=ur; pxi[n]=ui;
        }
    }
    for (int b=0;b<8;b++){
        v8 gr[8],gi[8];
        if (PW) MDFT8(pxr,pxi,gr,gi, b,8+b,16+b,24+b,32+b,40+b,48+b,56+b, 0,1,2,3,4,5,6,7);
        else MDFT8(re,im,gr,gi, b*sv,(8+b)*sv,(16+b)*sv,(24+b)*sv,(32+b)*sv,(40+b)*sv,(48+b)*sv,(56+b)*sv,
              0,1,2,3,4,5,6,7);
        Tr[b]=gr[0]; Ti[b]=gi[0];
        for (int a=1;a<8;a++){
            if (b==0){ Tr[a*8]=gr[a]; Ti[a*8]=gi[a]; }
            else {
                double wr=TW64R[a][b], wi=TW64I[a][b];
                Tr[a*8+b]=gr[a]*wr-gi[a]*wi; Ti[a*8+b]=gr[a]*wi+gi[a]*wr;
            }
        }
    }
    for (int a=0;a<8;a++){
        v8 yr[8],yi[8];
        MDFT8(Tr,Ti,yr,yi, a*8+0,a*8+1,a*8+2,a*8+3,a*8+4,a*8+5,a*8+6,a*8+7, 0,1,2,3,4,5,6,7);
        for (int bb=0;bb<8;bb++){
            long o=(long)(8*bb+a)*sv;
            re[o]=yr[bb]; im[o]=yi[bb];
        }
    }
}
LINE_WRAPPERS(line64)

// ---------------- special fused z-codelet for L=64 (one contiguous row) ----------------
#define TRANSPOSE8(r0,r1,r2,r3,r4,r5,r6,r7) do{ \
  __m512d t0,t1,t2,t3,t4,t5,t6,t7; \
  t0=_mm512_unpacklo_pd((__m512d)r0,(__m512d)r1); t1=_mm512_unpackhi_pd((__m512d)r0,(__m512d)r1); \
  t2=_mm512_unpacklo_pd((__m512d)r2,(__m512d)r3); t3=_mm512_unpackhi_pd((__m512d)r2,(__m512d)r3); \
  t4=_mm512_unpacklo_pd((__m512d)r4,(__m512d)r5); t5=_mm512_unpackhi_pd((__m512d)r4,(__m512d)r5); \
  t6=_mm512_unpacklo_pd((__m512d)r6,(__m512d)r7); t7=_mm512_unpackhi_pd((__m512d)r6,(__m512d)r7); \
  __m512i idlo=_mm512_setr_epi64(0,1,8,9,4,5,12,13); \
  __m512i idhi=_mm512_setr_epi64(2,3,10,11,6,7,14,15); \
  __m512d u0=_mm512_permutex2var_pd(t0,idlo,t2); \
  __m512d u1=_mm512_permutex2var_pd(t1,idlo,t3); \
  __m512d u2=_mm512_permutex2var_pd(t0,idhi,t2); \
  __m512d u3=_mm512_permutex2var_pd(t1,idhi,t3); \
  __m512d u4=_mm512_permutex2var_pd(t4,idlo,t6); \
  __m512d u5=_mm512_permutex2var_pd(t5,idlo,t7); \
  __m512d u6=_mm512_permutex2var_pd(t4,idhi,t6); \
  __m512d u7=_mm512_permutex2var_pd(t5,idhi,t7); \
  r0=(v8)_mm512_shuffle_f64x2(u0,u4,0x44); \
  r1=(v8)_mm512_shuffle_f64x2(u1,u5,0x44); \
  r2=(v8)_mm512_shuffle_f64x2(u2,u6,0x44); \
  r3=(v8)_mm512_shuffle_f64x2(u3,u7,0x44); \
  r4=(v8)_mm512_shuffle_f64x2(u0,u4,0xEE); \
  r5=(v8)_mm512_shuffle_f64x2(u1,u5,0xEE); \
  r6=(v8)_mm512_shuffle_f64x2(u2,u6,0xEE); \
  r7=(v8)_mm512_shuffle_f64x2(u3,u7,0xEE); \
}while(0)

static AI void zline64_impl(v8*restrict re, v8*restrict im,
                            const double*restrict crow, double*restrict cap, const int PW){
    v8 ar[8], ai[8];
    if (PW){
        const __m512i idre = _mm512_setr_epi64(0,2,4,6,8,10,12,14);
        const __m512i idim = _mm512_setr_epi64(1,3,5,7,9,11,13,15);
        const __m512i idlo = _mm512_setr_epi64(0,8,1,9,2,10,3,11);
        const __m512i idhi = _mm512_setr_epi64(4,12,5,13,6,14,7,15);
        for (int a=0;a<8;a++){
            _mm_prefetch((const char*)(crow+16*a)+2048, _MM_HINT_T0);
            _mm_prefetch((const char*)(crow+16*a)+2112, _MM_HINT_T0);
            _mm_prefetch((const char*)(re+a)+1024, _MM_HINT_T0);
            _mm_prefetch((const char*)(im+a)+1024, _MM_HINT_T0);
            __m512d ca=_mm512_loadu_pd(crow + 16*a);
            __m512d cb=_mm512_loadu_pd(crow + 16*a + 8);
            v8 car=(v8)_mm512_permutex2var_pd(ca, idre, cb);
            v8 cai=(v8)_mm512_permutex2var_pd(ca, idim, cb);
            PWCORE(xr,xi, re[a]+car, im[a]+cai)
            ar[a]=xr; ai[a]=xi;
            if (cap){
                _mm512_storeu_pd(cap+16*a,   _mm512_permutex2var_pd((__m512d)xr, idlo, (__m512d)xi));
                _mm512_storeu_pd(cap+16*a+8, _mm512_permutex2var_pd((__m512d)xr, idhi, (__m512d)xi));
            }
        }
    } else {
        for (int a=0;a<8;a++){ ar[a]=re[a]; ai[a]=im[a]; }
    }
    v8 Sr[8],Si[8];
    MDFT8(ar,ai,Sr,Si, 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7);
    for (int a=1;a<8;a++){
        v8 tr=Sr[a], ti=Si[a];
        Sr[a]=tr*VTW64R[a]-ti*VTW64I[a];
        Si[a]=tr*VTW64I[a]+ti*VTW64R[a];
    }
    TRANSPOSE8(Sr[0],Sr[1],Sr[2],Sr[3],Sr[4],Sr[5],Sr[6],Sr[7]);
    TRANSPOSE8(Si[0],Si[1],Si[2],Si[3],Si[4],Si[5],Si[6],Si[7]);
    MDFT8(Sr,Si,re,im, 0,1,2,3,4,5,6,7, 0,1,2,3,4,5,6,7);
}
static void zline64(v8*restrict re, v8*restrict im){ zline64_impl(re,im,0,0,0); }
static void zline64_pwc(v8*restrict re, v8*restrict im, const double*restrict crow){ zline64_impl(re,im,crow,0,1); }
static void zline64_pwc_cap(v8*restrict re, v8*restrict im, const double*restrict crow, double*restrict cap){ zline64_impl(re,im,crow,cap,1); }

// generic tile transpose for z-pass (other sizes): plane row stride R1 doubles
static AI void tile_in2(const double*restrict pr, const double*restrict pi,
                        const double*restrict cr, const double*restrict ci,
                        v8*restrict trr, v8*restrict tri, const int R1, const long rowbase, const int PW){
    for (int cb=0; cb<R1; cb+=8){
        v8 r[8], s[8];
        for (int q=0;q<8;q++){
            long off = rowbase + (long)q*R1 + cb;
            _mm_prefetch((const char*)(pr+off+8*(long)R1), _MM_HINT_T0);
            _mm_prefetch((const char*)(pi+off+8*(long)R1), _MM_HINT_T0);
            v8 xr=*(const v8*)(pr+off), xi=*(const v8*)(pi+off);
            if (PW){
                PWCORE(yr,yi, xr+*(const v8*)(cr+off), xi+*(const v8*)(ci+off))
                r[q]=yr; s[q]=yi;
            } else { r[q]=xr; s[q]=xi; }
        }
        TRANSPOSE8(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7]);
        TRANSPOSE8(s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7]);
        for (int q=0;q<8;q++){ trr[cb+q]=r[q]; tri[cb+q]=s[q]; }
    }
}
static AI void tile_out(double*restrict p, const v8*restrict tr, const int R1, const long rowbase){
    for (int cb=0; cb<R1; cb+=8){
        v8 r0=tr[cb+0], r1=tr[cb+1], r2=tr[cb+2], r3=tr[cb+3];
        v8 r4=tr[cb+4], r5=tr[cb+5], r6=tr[cb+6], r7=tr[cb+7];
        TRANSPOSE8(r0,r1,r2,r3,r4,r5,r6,r7);
        *(v8*)(p+rowbase+0*R1+cb)=r0; *(v8*)(p+rowbase+1*R1+cb)=r1;
        *(v8*)(p+rowbase+2*R1+cb)=r2; *(v8*)(p+rowbase+3*R1+cb)=r3;
        *(v8*)(p+rowbase+4*R1+cb)=r4; *(v8*)(p+rowbase+5*R1+cb)=r5;
        *(v8*)(p+rowbase+6*R1+cb)=r6; *(v8*)(p+rowbase+7*R1+cb)=r7;
    }
}

#define CALL_LINE(L, re,im,s) do{ switch(L){ \
  case 6: line6(re,im,s); break; case 8: line8(re,im,s); break; \
  case 13: line13(re,im,s); break; case 17: line17(re,im,s); break; \
  case 23: line23(re,im,s); break; case 36: line36(re,im,s); break; \
  case 45: line45(re,im,s); break; default: line64(re,im,s); } }while(0)

#define CALL_LINE_ZPW(L, re,im,cr,ci,s) do{ switch(L){ \
  case 6: line6_zpw(re,im,cr,ci,s); break; case 8: line8_zpw(re,im,cr,ci,s); break; \
  case 13: line13_zpw(re,im,cr,ci,s); break; case 17: line17_zpw(re,im,cr,ci,s); break; \
  case 23: line23_zpw(re,im,cr,ci,s); break; case 36: line36_zpw(re,im,cr,ci,s); break; \
  case 45: line45_zpw(re,im,cr,ci,s); break; default: line64_zpw(re,im,cr,ci,s); } }while(0)

#define CALL_LINE_PF(L, re,im,s) do{ switch(L){ \
  case 6: line6_pf(re,im,s); break; case 8: line8_pf(re,im,s); break; \
  case 13: line13_pf(re,im,s); break; case 17: line17_pf(re,im,s); break; \
  case 23: line23_pf(re,im,s); break; case 36: line36_pf(re,im,s); break; \
  case 45: line45_pf(re,im,s); break; default: line64_pf(re,im,s); } }while(0)

static AI void row_deint(const double*restrict src, double*restrict dre, double*restrict dim, const int L){
    const __m512i idre = _mm512_setr_epi64(0,2,4,6,8,10,12,14);
    const __m512i idim = _mm512_setr_epi64(1,3,5,7,9,11,13,15);
    int z=0;
    for (; z+8<=L; z+=8){
        __m512d a=_mm512_loadu_pd(src+2*z), b=_mm512_loadu_pd(src+2*z+8);
        _mm512_store_pd(dre+z, _mm512_permutex2var_pd(a, idre, b));
        _mm512_store_pd(dim+z, _mm512_permutex2var_pd(a, idim, b));
    }
    for (; z<L; ++z){ dre[z]=src[2*z]; dim[z]=src[2*z+1]; }
}
static AI void row_int(double*restrict dst, const double*restrict sre, const double*restrict sim, const int L){
    const __m512i idlo = _mm512_setr_epi64(0,8,1,9,2,10,3,11);
    const __m512i idhi = _mm512_setr_epi64(4,12,5,13,6,14,7,15);
    int z=0;
    for (; z+8<=L; z+=8){
        __m512d a=_mm512_load_pd(sre+z), b=_mm512_load_pd(sim+z);
        _mm512_storeu_pd(dst+2*z,   _mm512_permutex2var_pd(a, idlo, b));
        _mm512_storeu_pd(dst+2*z+8, _mm512_permutex2var_pd(a, idhi, b));
    }
    for (; z<L; ++z){ dst[2*z]=sre[z]; dst[2*z+1]=sim[z]; }
}


// deinterleave one vec (8 complex) of c from interleaved source, masked (zeros beyond)
static AI void deint_c(const double*restrict csrc, long k, __mmask8 ma, __mmask8 mb,
                       v8*restrict car, v8*restrict cai){
    const __m512i idre = _mm512_setr_epi64(0,2,4,6,8,10,12,14);
    const __m512i idim = _mm512_setr_epi64(1,3,5,7,9,11,13,15);
    __m512d a=_mm512_maskz_loadu_pd(ma, csrc + 16*k);
    __m512d b=_mm512_maskz_loadu_pd(mb, csrc + 16*k + 8);
    *car=(v8)_mm512_permutex2var_pd(a, idre, b);
    *cai=(v8)_mm512_permutex2var_pd(a, idim, b);
}

// ---------------- per-volume sweeps (pass order per step: Z, X, Y) ----------------
static AI void pv_plane_z(const int L, const int R1, double*restrict pr, double*restrict pi){
    v8 trr[64] __attribute__((aligned(64)));
    v8 tri[64] __attribute__((aligned(64)));
    if (L==64){
        for (int r=0;r<64;r++)
            zline64((v8*)(pr+(long)r*R1), (v8*)(pi+(long)r*R1));
    } else {
        for (int t=0; t<R1; t+=8){
            long rowbase = (long)t*R1;
            tile_in2(pr,pi,0,0, trr,tri, R1, rowbase, 0);
            CALL_LINE(L, trr, tri, 1);
            tile_out(pr, trr, R1, rowbase);
            tile_out(pi, tri, R1, rowbase);
        }
    }
}
static AI void pv_plane_y(const int L, const int R1, double*restrict pr, double*restrict pi){
    const int R1v = R1/8;
    for (int v=0; v<R1v; v++)
        CALL_LINE(L, (v8*)pr+v, (v8*)pi+v, R1v);
}
static AI void pv_plane_pw(const int L, const int R1, double*restrict pr, double*restrict pi,
                           const double*restrict cplane /* interleaved, L*L complex */){
    const int NV = (L+7)/8;        // vecs per row (data only)
    const int TV = L - (NV-1)*8;   // valid complex in last vec (1..8)
    const __mmask8 mA = (__mmask8)((2*TV>=8)? 0xFF : ((1u<<(2*TV))-1));
    const __mmask8 mB = (__mmask8)((2*TV<=8)? 0 : ((1u<<(2*TV-8))-1));
    for (int y=0;y<L;y++){
        v8 *ar=(v8*)(pr+(long)y*R1), *ai=(v8*)(pi+(long)y*R1);
        const double* crow = cplane + 2*(long)y*L;
        for (int v=0; v<NV; v++){
            v8 car, cai;
            deint_c(crow, v, (v==NV-1)?mA:0xFF, (v==NV-1)?mB:0xFF, &car, &cai);
            PWCORE(xr,xi, ar[v]+car, ai[v]+cai)
            ar[v]=xr; ai[v]=xi;
        }
    }
}
static AI void pv_sweepX(const int L, const int R1, const long P, double*restrict br, double*restrict bi){
    const long Pv = P/8; const int R1v = R1/8;
    for (int y=0; y<L; ++y){
        long basev = (long)y*R1v;
        for (int v=0; v<R1v; v++)
            CALL_LINE_PF(L, (v8*)br+basev+v, (v8*)bi+basev+v, Pv);
    }
}
// capture plane (post-pw state) into interleaved complex output
static AI void pv_capture_plane(const int L, const int R1, double*restrict dst, int x,
                                const double*restrict pr, const double*restrict pi){
    for (int y=0;y<L;y++){
        double* d = dst + 2*((long)(x*L+y)*L);
        row_int(d, pr+(long)y*R1, pi+(long)y*R1, L);
    }
}

// ---------------- conversions ----------------
// ---------------- conversions ----------------
static AI void conv_in_plane(const int L, const int R1,
                       const double*restrict srcplane, double*restrict pr, double*restrict pi){
    for (int y=0;y<L;y++){
        row_deint(srcplane + 2*(long)y*L, pr+(long)y*R1, pi+(long)y*R1, L);
    }
}
static AI void conv_out_pw(const int L, const int R1, const long P,
                        double*restrict dst, double*restrict dst2,
                        const double*restrict br, const double*restrict bi,
                        const double*restrict csrc /* interleaved */){
    const int NV = (L+7)/8;
    const int TV = L - (NV-1)*8;
    const __mmask8 mA = (__mmask8)((2*TV>=8)? 0xFF : ((1u<<(2*TV))-1));
    const __mmask8 mB = (__mmask8)((2*TV<=8)? 0 : ((1u<<(2*TV-8))-1));
    const __m512i idlo = _mm512_setr_epi64(0,8,1,9,2,10,3,11);
    const __m512i idhi = _mm512_setr_epi64(4,12,5,13,6,14,7,15);
    for (int x=0;x<L;x++) for (int y=0;y<L;y++){
        long roff = (long)(x*L+y)*L;
        double* d = dst + 2*roff;
        double* d2 = dst2 ? dst2 + 2*roff : 0;
        const double* crow = csrc + 2*roff;
        long off = (long)x*P + (long)y*R1;
        const v8* sr = (const v8*)(br+off); const v8* si = (const v8*)(bi+off);
        for (int v=0; v<NV; v++){
            int tv = (v==NV-1)? TV : 8;
            v8 car, cai;
            deint_c(crow, v, (v==NV-1)?mA:0xFF, (v==NV-1)?mB:0xFF, &car, &cai);
            PWCORE(xr,xi, sr[v]+car, si[v]+cai)
            __m512d lo=_mm512_permutex2var_pd((__m512d)xr, idlo, (__m512d)xi);
            __m512d hi=_mm512_permutex2var_pd((__m512d)xr, idhi, (__m512d)xi);
            _mm512_mask_storeu_pd(d+16*v, (v==NV-1)?mA:(__mmask8)0xFF, lo);
            if (2*tv > 8) _mm512_mask_storeu_pd(d+16*v+8, (v==NV-1)?mB:(__mmask8)0xFF, hi);
            if (d2){
                _mm512_mask_storeu_pd(d2+16*v, (v==NV-1)?mA:(__mmask8)0xFF, lo);
                if (2*tv > 8) _mm512_mask_storeu_pd(d2+16*v+8, (v==NV-1)?mB:(__mmask8)0xFF, hi);
            }
        }
    }
}


// ---------------- batch-interleaved path (8 volumes in lanes) ----------------
// layout: element (x,y,z) at vec index x*PL2 + y*L + z, PL2 = L*L+1 (conflict padding)
static AI void conv_in_il(const int L, const double*restrict src, const long vs,
                          v8*restrict XR, v8*restrict XI, const int nv){
    const long L2 = (long)L*L, PL2 = L2+1;
    for (int x=0;x<L;x++){
        const double* sp = src + 2*(long)x*L2;
        v8 *xr = XR + (long)x*PL2, *xi = XI + (long)x*PL2;
        long e=0;
        for (; e+8<=L2; e+=8){
            v8 A[8], Bv[8];
            for (int w=0;w<8;w++){
                long wo = (w<nv)? (long)w*vs : 0;
                A[w] =(v8)_mm512_loadu_pd(sp + wo + 2*e);
                Bv[w]=(v8)_mm512_loadu_pd(sp + wo + 2*e + 8);
            }
            TRANSPOSE8(A[0],A[1],A[2],A[3],A[4],A[5],A[6],A[7]);
            TRANSPOSE8(Bv[0],Bv[1],Bv[2],Bv[3],Bv[4],Bv[5],Bv[6],Bv[7]);
            xr[e+0]=A[0]; xi[e+0]=A[1]; xr[e+1]=A[2]; xi[e+1]=A[3];
            xr[e+2]=A[4]; xi[e+2]=A[5]; xr[e+3]=A[6]; xi[e+3]=A[7];
            xr[e+4]=Bv[0]; xi[e+4]=Bv[1]; xr[e+5]=Bv[2]; xi[e+5]=Bv[3];
            xr[e+6]=Bv[4]; xi[e+6]=Bv[5]; xr[e+7]=Bv[6]; xi[e+7]=Bv[7];
        }
        for (; e<L2; ++e){
            for (int w=0;w<8;w++){
                long wo = (w<nv)? (long)w*vs : 0;
                xr[e][w]=sp[wo+2*e]; xi[e][w]=sp[wo+2*e+1];
            }
        }
    }
}
static AI void conv_out_il_pw(const int L, double*restrict dst, const long vs,
                              const v8*restrict XR, const v8*restrict XI,
                              const v8*restrict CR, const v8*restrict CI, const int nv){
    const long L2 = (long)L*L, PL2 = L2+1;
    for (int x=0;x<L;x++){
        double* dp = dst + 2*(long)x*L2;
        const v8 *xr = XR + (long)x*PL2, *xi = XI + (long)x*PL2;
        const v8 *kr = CR + (long)x*PL2, *ki = CI + (long)x*PL2;
        long e=0;
        for (; e+8<=L2; e+=8){
            v8 R[8], I[8];
            for (int q=0;q<8;q++){
                PWCORE(ur,ui, xr[e+q]+kr[e+q], xi[e+q]+ki[e+q])
                R[q]=ur; I[q]=ui;
            }
            v8 A0=R[0],A1=I[0],A2=R[1],A3=I[1],A4=R[2],A5=I[2],A6=R[3],A7=I[3];
            TRANSPOSE8(A0,A1,A2,A3,A4,A5,A6,A7);
            v8 B0=R[4],B1=I[4],B2=R[5],B3=I[5],B4=R[6],B5=I[6],B6=R[7],B7=I[7];
            TRANSPOSE8(B0,B1,B2,B3,B4,B5,B6,B7);
            if (0<nv){ _mm512_storeu_pd(dp + 0*vs + 2*e, (__m512d)A0); _mm512_storeu_pd(dp + 0*vs + 2*e + 8, (__m512d)B0); }
            if (1<nv){ _mm512_storeu_pd(dp + 1*vs + 2*e, (__m512d)A1); _mm512_storeu_pd(dp + 1*vs + 2*e + 8, (__m512d)B1); }
            if (2<nv){ _mm512_storeu_pd(dp + 2*vs + 2*e, (__m512d)A2); _mm512_storeu_pd(dp + 2*vs + 2*e + 8, (__m512d)B2); }
            if (3<nv){ _mm512_storeu_pd(dp + 3*vs + 2*e, (__m512d)A3); _mm512_storeu_pd(dp + 3*vs + 2*e + 8, (__m512d)B3); }
            if (4<nv){ _mm512_storeu_pd(dp + 4*vs + 2*e, (__m512d)A4); _mm512_storeu_pd(dp + 4*vs + 2*e + 8, (__m512d)B4); }
            if (5<nv){ _mm512_storeu_pd(dp + 5*vs + 2*e, (__m512d)A5); _mm512_storeu_pd(dp + 5*vs + 2*e + 8, (__m512d)B5); }
            if (6<nv){ _mm512_storeu_pd(dp + 6*vs + 2*e, (__m512d)A6); _mm512_storeu_pd(dp + 6*vs + 2*e + 8, (__m512d)B6); }
            if (7<nv){ _mm512_storeu_pd(dp + 7*vs + 2*e, (__m512d)A7); _mm512_storeu_pd(dp + 7*vs + 2*e + 8, (__m512d)B7); }
        }
        for (; e<L2; ++e){
            PWCORE(ur,ui, xr[e]+kr[e], xi[e]+ki[e])
            for (int w=0;w<nv;w++){
                dp[(long)w*vs+2*e]   = ur[w];
                dp[(long)w*vs+2*e+1] = ui[w];
            }
        }
    }
}

static AI void conv_out_il_pw_direct(const int L, double*restrict dst, double*restrict dst2, const long vs,
                              const v8*restrict XR, const v8*restrict XI, const double*restrict csrc, const int nv){
    const long L2 = (long)L*L, PL2 = L2+1;
    for (int x=0;x<L;x++){
        double* dp = dst + 2*(long)x*L2;
        double* dq = dst2 ? dst2 + 2*(long)x*L2 : 0;
        const double* cp = csrc + 2*(long)x*L2;
        const v8 *xr = XR + (long)x*PL2, *xi = XI + (long)x*PL2;
        long e=0;
        for (; e+8<=L2; e+=8){
            v8 CA[8], CB[8];
            for (int w=0;w<8;w++){
                long wo = (w<nv)? (long)w*vs : 0;
                CA[w]=(v8)_mm512_loadu_pd(cp + wo + 2*e);
                CB[w]=(v8)_mm512_loadu_pd(cp + wo + 2*e + 8);
            }
            TRANSPOSE8(CA[0],CA[1],CA[2],CA[3],CA[4],CA[5],CA[6],CA[7]);
            TRANSPOSE8(CB[0],CB[1],CB[2],CB[3],CB[4],CB[5],CB[6],CB[7]);
            v8 R[8], I[8];
            for (int q=0;q<4;q++){
                PWCORE(ur,ui, xr[e+q]+CA[2*q], xi[e+q]+CA[2*q+1])
                R[q]=ur; I[q]=ui;
            }
            for (int q=4;q<8;q++){
                PWCORE(ur,ui, xr[e+q]+CB[2*(q-4)], xi[e+q]+CB[2*(q-4)+1])
                R[q]=ur; I[q]=ui;
            }
            v8 A0=R[0],A1=I[0],A2=R[1],A3=I[1],A4=R[2],A5=I[2],A6=R[3],A7=I[3];
            TRANSPOSE8(A0,A1,A2,A3,A4,A5,A6,A7);
            v8 B0=R[4],B1=I[4],B2=R[5],B3=I[5],B4=R[6],B5=I[6],B6=R[7],B7=I[7];
            TRANSPOSE8(B0,B1,B2,B3,B4,B5,B6,B7);
            if (0<nv){ _mm512_storeu_pd(dp + 0*vs + 2*e, (__m512d)A0); _mm512_storeu_pd(dp + 0*vs + 2*e + 8, (__m512d)B0); }
            if (1<nv){ _mm512_storeu_pd(dp + 1*vs + 2*e, (__m512d)A1); _mm512_storeu_pd(dp + 1*vs + 2*e + 8, (__m512d)B1); }
            if (2<nv){ _mm512_storeu_pd(dp + 2*vs + 2*e, (__m512d)A2); _mm512_storeu_pd(dp + 2*vs + 2*e + 8, (__m512d)B2); }
            if (3<nv){ _mm512_storeu_pd(dp + 3*vs + 2*e, (__m512d)A3); _mm512_storeu_pd(dp + 3*vs + 2*e + 8, (__m512d)B3); }
            if (4<nv){ _mm512_storeu_pd(dp + 4*vs + 2*e, (__m512d)A4); _mm512_storeu_pd(dp + 4*vs + 2*e + 8, (__m512d)B4); }
            if (5<nv){ _mm512_storeu_pd(dp + 5*vs + 2*e, (__m512d)A5); _mm512_storeu_pd(dp + 5*vs + 2*e + 8, (__m512d)B5); }
            if (6<nv){ _mm512_storeu_pd(dp + 6*vs + 2*e, (__m512d)A6); _mm512_storeu_pd(dp + 6*vs + 2*e + 8, (__m512d)B6); }
            if (7<nv){ _mm512_storeu_pd(dp + 7*vs + 2*e, (__m512d)A7); _mm512_storeu_pd(dp + 7*vs + 2*e + 8, (__m512d)B7); }
            if (dq){
                if (0<nv){ _mm512_storeu_pd(dq + 0*vs + 2*e, (__m512d)A0); _mm512_storeu_pd(dq + 0*vs + 2*e + 8, (__m512d)B0); }
                if (1<nv){ _mm512_storeu_pd(dq + 1*vs + 2*e, (__m512d)A1); _mm512_storeu_pd(dq + 1*vs + 2*e + 8, (__m512d)B1); }
                if (2<nv){ _mm512_storeu_pd(dq + 2*vs + 2*e, (__m512d)A2); _mm512_storeu_pd(dq + 2*vs + 2*e + 8, (__m512d)B2); }
                if (3<nv){ _mm512_storeu_pd(dq + 3*vs + 2*e, (__m512d)A3); _mm512_storeu_pd(dq + 3*vs + 2*e + 8, (__m512d)B3); }
                if (4<nv){ _mm512_storeu_pd(dq + 4*vs + 2*e, (__m512d)A4); _mm512_storeu_pd(dq + 4*vs + 2*e + 8, (__m512d)B4); }
                if (5<nv){ _mm512_storeu_pd(dq + 5*vs + 2*e, (__m512d)A5); _mm512_storeu_pd(dq + 5*vs + 2*e + 8, (__m512d)B5); }
                if (6<nv){ _mm512_storeu_pd(dq + 6*vs + 2*e, (__m512d)A6); _mm512_storeu_pd(dq + 6*vs + 2*e + 8, (__m512d)B6); }
                if (7<nv){ _mm512_storeu_pd(dq + 7*vs + 2*e, (__m512d)A7); _mm512_storeu_pd(dq + 7*vs + 2*e + 8, (__m512d)B7); }
            }
        }
        for (; e<L2; ++e){
            v8 car, cai;
            for (int w=0;w<8;w++){ long wo=(w<nv)?(long)w*vs:0; car[w]=cp[wo+2*e]; cai[w]=cp[wo+2*e+1]; }
            PWCORE(ur,ui, xr[e]+car, xi[e]+cai)
            for (int w=0;w<nv;w++){
                dp[(long)w*vs+2*e] = ur[w]; dp[(long)w*vs+2*e+1] = ui[w];
                if (dq){ dq[(long)w*vs+2*e] = ur[w]; dq[(long)w*vs+2*e+1] = ui[w]; }
            }
        }
    }
}

static AI void il_plane_pw(const long L2, v8*restrict pr, v8*restrict pi,
                           const v8*restrict qr, const v8*restrict qi){
    long i=0;
    for (;i+4<=L2;i+=4){
        _mm_prefetch((const char*)(qr+i)+2048, _MM_HINT_T0);
        _mm_prefetch((const char*)(qi+i)+2048, _MM_HINT_T0);
        _mm_prefetch((const char*)(qr+i)+2112, _MM_HINT_T0);
        _mm_prefetch((const char*)(qi+i)+2112, _MM_HINT_T0);
        { PWCORE(xr,xi, pr[i]+qr[i], pi[i]+qi[i]) pr[i]=xr; pi[i]=xi; }
        { PWCORE(xr,xi, pr[i+1]+qr[i+1], pi[i+1]+qi[i+1]) pr[i+1]=xr; pi[i+1]=xi; }
        { PWCORE(xr,xi, pr[i+2]+qr[i+2], pi[i+2]+qi[i+2]) pr[i+2]=xr; pi[i+2]=xi; }
        { PWCORE(xr,xi, pr[i+3]+qr[i+3], pi[i+3]+qi[i+3]) pr[i+3]=xr; pi[i+3]=xi; }
    }
    for (;i<L2;i++){ PWCORE(xr,xi, pr[i]+qr[i], pi[i]+qi[i]) pr[i]=xr; pi[i]=xi; }
}
static AI void il_capture_plane(const int L, double*restrict dst, const long vs, const int x,
                                const v8*restrict pr, const v8*restrict pi, const int nv){
    const long L2=(long)L*L;
    double* dp = dst + 2*(long)x*L2;
    long e=0;
    for (; e+8<=L2; e+=8){
        v8 A0=pr[e+0],A1=pi[e+0],A2=pr[e+1],A3=pi[e+1],A4=pr[e+2],A5=pi[e+2],A6=pr[e+3],A7=pi[e+3];
        TRANSPOSE8(A0,A1,A2,A3,A4,A5,A6,A7);
        v8 B0=pr[e+4],B1=pi[e+4],B2=pr[e+5],B3=pi[e+5],B4=pr[e+6],B5=pi[e+6],B6=pr[e+7],B7=pi[e+7];
        TRANSPOSE8(B0,B1,B2,B3,B4,B5,B6,B7);
        if (0<nv){ _mm512_storeu_pd(dp + 0*vs + 2*e, (__m512d)A0); _mm512_storeu_pd(dp + 0*vs + 2*e + 8, (__m512d)B0); }
        if (1<nv){ _mm512_storeu_pd(dp + 1*vs + 2*e, (__m512d)A1); _mm512_storeu_pd(dp + 1*vs + 2*e + 8, (__m512d)B1); }
        if (2<nv){ _mm512_storeu_pd(dp + 2*vs + 2*e, (__m512d)A2); _mm512_storeu_pd(dp + 2*vs + 2*e + 8, (__m512d)B2); }
        if (3<nv){ _mm512_storeu_pd(dp + 3*vs + 2*e, (__m512d)A3); _mm512_storeu_pd(dp + 3*vs + 2*e + 8, (__m512d)B3); }
        if (4<nv){ _mm512_storeu_pd(dp + 4*vs + 2*e, (__m512d)A4); _mm512_storeu_pd(dp + 4*vs + 2*e + 8, (__m512d)B4); }
        if (5<nv){ _mm512_storeu_pd(dp + 5*vs + 2*e, (__m512d)A5); _mm512_storeu_pd(dp + 5*vs + 2*e + 8, (__m512d)B5); }
        if (6<nv){ _mm512_storeu_pd(dp + 6*vs + 2*e, (__m512d)A6); _mm512_storeu_pd(dp + 6*vs + 2*e + 8, (__m512d)B6); }
        if (7<nv){ _mm512_storeu_pd(dp + 7*vs + 2*e, (__m512d)A7); _mm512_storeu_pd(dp + 7*vs + 2*e + 8, (__m512d)B7); }
    }
    for (; e<L2; ++e){
        for (int w=0;w<nv;w++){
            dp[(long)w*vs+2*e]   = pr[e][w];
            dp[(long)w*vs+2*e+1] = pi[e][w];
        }
    }
}
static AI void il_plane_z(const int L, v8*restrict pr, v8*restrict pi){
    for (int y=0; y<L; ++y)
        CALL_LINE(L, pr+(long)y*L, pi+(long)y*L, 1);
}
static AI void il_plane_y(const int L, v8*restrict pr, v8*restrict pi){
    for (int z=0; z<L; ++z)
        CALL_LINE(L, pr+z, pi+z, L);
}
static AI void il_sweepX(const int L, v8*restrict XR, v8*restrict XI){
    const long L2=(long)L*L, PL2=L2+1;
    for (long yz=0; yz<L2; ++yz)
        CALL_LINE_PF(L, XR+yz, XI+yz, PL2);
}

// ---------------- buffers ----------------
static double *BUF[8][4];
static long BUFSZ[8];
static v8 *ILBUF[8][4];
static long ILBUFSZ[8];
static const int SIZES[8] = {6,8,13,17,23,36,45,64};
static int size_index(int L){ for(int i=0;i<8;i++) if (SIZES[i]==L) return i; return -1; }

static double* big_alloc(size_t bytes){
    size_t sz = (bytes + (2UL<<20) - 1) & ~((2UL<<20)-1);
    void* p = mmap(0, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { p = aligned_alloc(64, bytes); if (p) memset(p,0,bytes); return (double*)p; }
    madvise(p, sz, MADV_HUGEPAGE);
    memset(p, 0, sz);
    return (double*)p;
}

static void get_buffers(int si, int L, int R1, long P, double**br,double**bi){
    long need = P*(long)L + 16;
    if (L==64) need = (72L*64+8)*64 + 16;
    if (BUFSZ[si] < need){
        for (int q=0;q<2;q++) BUF[si][q] = big_alloc((size_t)need*8);
        BUFSZ[si]=need;
    }
    *br=BUF[si][0]; *bi=BUF[si][1];
}
static void get_il_buffers(int si, long L3, v8**xr, v8**xi, v8**cr, v8**ci){
    if (ILBUFSZ[si] < L3){
        for (int q=0;q<4;q++) ILBUF[si][q] = (v8*)big_alloc((size_t)L3*64);
        ILBUFSZ[si]=L3;
    }
    *xr=ILBUF[si][0]; *xi=ILBUF[si][1]; *cr=ILBUF[si][2]; *ci=ILBUF[si][3];
}

// ---------------- top-level ----------------
static AI void run_generic(const int L, const long B, const long m,
                           const double*restrict x0, const double*restrict c,
                           double*restrict o1, double*restrict om){
    const int R1 = (L+7)&~7;
    const long P = (long)R1*R1 + 8;
    double *br,*bi;
    get_buffers(size_index(L), L, R1, P, &br,&bi);
    const long vs = 2L*L*L*L;
    long v0 = 0;
    {   // interleaved path; final partial group (nv>=RMIN) runs with duplicated lanes
        int RMIN;
        switch(L){ case 6: RMIN=4; break; case 13: RMIN=6; break; case 17: RMIN=5; break;
                   default: RMIN=8; }
        if (L <= 36){
        const long L3 = (long)L*((long)L*L+1) + 8;
        const long L2=(long)L*L, PL2=L2+1;
        v8 *XR=0,*XI=0,*CR=0,*CI=0;
        while (B - v0 >= ((8<RMIN)?RMIN:( (B-v0)>=8 ? 8 : RMIN ))){
            long rem = B - v0;
            int nv;
            if (rem >= 8) nv = 8;
            else if (rem >= RMIN) nv = (int)rem;
            else break;
            if (!XR) get_il_buffers(size_index(L), L3, &XR,&XI,&CR,&CI);
            conv_in_il(L, x0+v0*vs, vs, XR, XI, nv);
            if (m==1){
                for (int x=0;x<L;x++) il_plane_y(L, XR+(long)x*PL2, XI+(long)x*PL2);
                il_sweepX(L, XR, XI);
                for (int x=0;x<L;x++) il_plane_z(L, XR+(long)x*PL2, XI+(long)x*PL2);
                conv_out_il_pw_direct(L, o1+v0*vs, om+v0*vs, vs, XR, XI, c+v0*vs, nv);
                v0 += nv;
                continue;
            }
            conv_in_il(L, c+v0*vs, vs, CR, CI, nv);
            for (int x=0;x<L;x++) il_plane_y(L, XR+(long)x*PL2, XI+(long)x*PL2);
            il_sweepX(L, XR, XI);
            for (long t=1; t<m; ++t){
                for (int x=0;x<L;x++){
                    v8 *pr=XR+(long)x*PL2, *pi=XI+(long)x*PL2;
                    const v8 *qr=CR+(long)x*PL2, *qi=CI+(long)x*PL2;
                    il_plane_z(L, pr,pi);
                    il_plane_pw(L2, pr,pi, qr,qi);
                    if (t==1) il_capture_plane(L, o1+v0*vs, vs, x, pr,pi, nv);
                    il_plane_y(L, pr,pi);
                }
                il_sweepX(L, XR, XI);
            }
            for (int x=0;x<L;x++){
                v8 *pr=XR+(long)x*PL2, *pi=XI+(long)x*PL2;
                const v8 *qr=CR+(long)x*PL2, *qi=CI+(long)x*PL2;
                il_plane_z(L, pr,pi);
                il_plane_pw(L2, pr,pi, qr,qi);
                if (m==1) il_capture_plane(L, o1+v0*vs, vs, x, pr,pi, nv);
                il_capture_plane(L, om+v0*vs, vs, x, pr,pi, nv);
            }
            v0 += nv;
        }
        }
    }
    if (L==64){
        const long RS = 72;                 // padded row stride (breaks 512B L1-set aliasing)
        const long P64 = RS*64 + 8;
        const long Pv64 = P64/8;
        for (long v=v0; v<B; ++v){
            const double* cv = c+v*vs;
            for (long t=0; t<m; ++t){
                for (int x=0;x<L;x++){
                    double *pr=br+(long)x*P64, *pi=bi+(long)x*P64;
                    if (t==0) conv_in_plane(L,RS, x0+v*vs+2*(long)x*L*L, pr,pi);
                    if (t>0){
                        const double* cp = cv + 2*(long)x*L*L;
                        double* capp = (t==1) ? o1+v*vs+2*(long)x*L*L : 0;
                        for (int r=0;r<64;r++){
                            if (capp) zline64_pwc_cap((v8*)(pr+(long)r*RS), (v8*)(pi+(long)r*RS), cp + 2*(long)r*L, capp + 2*(long)r*L);
                            else zline64_pwc((v8*)(pr+(long)r*RS), (v8*)(pi+(long)r*RS), cp + 2*(long)r*L);
                        }
                    } else {
                        for (int r=0;r<64;r++)
                            zline64((v8*)(pr+(long)r*RS), (v8*)(pi+(long)r*RS));
                    }
                    for (int w=0; w<8; w++)
                        line64((v8*)pr+w, (v8*)pi+w, RS/8);
                }
                for (int y=0; y<64; ++y){
                    long basev = (long)y*(RS/8);
                    for (int w=0; w<8; w++)
                        line64_pf((v8*)br+basev+w, (v8*)bi+basev+w, Pv64);
                }
            }
            if (m==1) conv_out_pw(L,RS,P64, o1+v*vs, om+v*vs, br,bi, cv);
            else conv_out_pw(L,RS,P64, om+v*vs, 0, br,bi, cv);
        }
        return;
    }
    for (long v=v0; v<B; ++v){
        const double* cv = c+v*vs;
        for (int x=0;x<L;x++){
            double *pr=br+(long)x*P, *pi=bi+(long)x*P;
            conv_in_plane(L,R1, x0+v*vs+2*(long)x*L*L, pr,pi);
            pv_plane_y(L,R1, pr,pi);
        }
        pv_sweepX(L,R1,P, br,bi);
        for (long t=1; t<m; ++t){
            for (int x=0;x<L;x++){
                double *pr=br+(long)x*P, *pi=bi+(long)x*P;
                pv_plane_z(L,R1, pr,pi);
                pv_plane_pw(L,R1, pr,pi, cv + 2*(long)x*L*L);
                if (t==1) pv_capture_plane(L,R1, o1+v*vs, x, pr,pi);
                pv_plane_y(L,R1, pr,pi);
            }
            pv_sweepX(L,R1,P, br,bi);
        }
        for (int x=0;x<L;x++){
            double *pr=br+(long)x*P, *pi=bi+(long)x*P;
            pv_plane_z(L,R1, pr,pi);
            pv_plane_pw(L,R1, pr,pi, cv + 2*(long)x*L*L);
            if (m==1) pv_capture_plane(L,R1, o1+v*vs, x, pr,pi);
            pv_capture_plane(L,R1, om+v*vs, x, pr,pi);
        }
    }
}

void ns_run(int L, long B, long m, const double* x0, const double* c, double* o1, double* om){
    switch(L){
        case 6:  run_generic(6,B,m,x0,c,o1,om); break;
        case 8:  run_generic(8,B,m,x0,c,o1,om); break;
        case 13: run_generic(13,B,m,x0,c,o1,om); break;
        case 17: run_generic(17,B,m,x0,c,o1,om); break;
        case 23: run_generic(23,B,m,x0,c,o1,om); break;
        case 36: run_generic(36,B,m,x0,c,o1,om); break;
        case 45: run_generic(45,B,m,x0,c,o1,om); break;
        case 64: run_generic(64,B,m,x0,c,o1,om); break;
    }
}
