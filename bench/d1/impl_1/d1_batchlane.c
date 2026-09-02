/* d1_batchlane: SoA 8-lane-per-zmm batch-lane engine.
 *
 * Every scalar operation of the underlying algorithm becomes exactly one AVX-512
 * vector instruction across 8 transforms of the batch (split complex, zero shuffles
 * inside the kernels; the only shuffles are the 8x8 transposes at the boundary).
 *
 * Kernels: conjugate-symmetric dense DFT for primes 13/31 (4 FMA per (j,k) pair),
 * recursive radix-4(+2) for 32/64/128, twiddle-free PFA 3x4x5 for 60.
 * fft1d_chain keeps the state SoA-resident across all m steps (no per-step
 * transposes) and maps in-register with NR-refined rsqrt/rcp (no divider).
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "../fft1d_api.h"

const char *fft1d_name(void){ return "d1_batchlane"; }
const char *fft1d_description(void){
    return "SoA 8-lane/zmm batch engine: conj-sym dense 13/31, radix-4 pow2, PFA-60; "
           "SoA-resident fused chain (rsqrt14+NR map)";
}
int fft1d_supports(int L){
    return L==13 || L==31 || L==32 || L==60 || L==64 || L==128;
}

#if defined(__AVX512F__)

#include <immintrin.h>

typedef __m512d VD;
#define LDA(p)      _mm512_load_pd(p)
#define STA(p,v)    _mm512_store_pd((p),(v))
#define SET1(x)     _mm512_set1_pd(x)
#define FMA(a,b,c)  _mm512_fmadd_pd((a),(b),(c))   /* a*b+c  */
#define FNMA(a,b,c) _mm512_fnmadd_pd((a),(b),(c))  /* c-a*b  */

struct fft1d_plan {
    int L, batch;
    int H;                 /* (L-1)/2 for primes */
    double *ct, *st;       /* prime tables, [H][H] each */
    double *tw2;           /* pow2 twiddles, all levels flattened */
    long twoff[8];
    int pfa_in[60];        /* cube slot g*5+q reads natural element pfa_in[..] */
    int pfa_out[60];       /* natural output k lives in cube slot pfa_out[k]   */
    double *mem;           /* one aligned slab, partitioned below */
    double *ar,*ai,*br,*bi;      /* kernel in / out (L vectors each) */
    double *sr,*si,*cr,*ci;      /* chain state and c field          */
    double *er,*ei,*odr,*odi;    /* prime even/odd scratch           */
};

/* ---------------- 8x8 double transpose (24 shuffles) ---------------- */
static inline void tr8(VD r[8]){
    VD t0=_mm512_unpacklo_pd(r[0],r[1]), t1=_mm512_unpackhi_pd(r[0],r[1]);
    VD t2=_mm512_unpacklo_pd(r[2],r[3]), t3=_mm512_unpackhi_pd(r[2],r[3]);
    VD t4=_mm512_unpacklo_pd(r[4],r[5]), t5=_mm512_unpackhi_pd(r[4],r[5]);
    VD t6=_mm512_unpacklo_pd(r[6],r[7]), t7=_mm512_unpackhi_pd(r[6],r[7]);
    VD u0=_mm512_shuffle_f64x2(t0,t2,0x88), u1=_mm512_shuffle_f64x2(t1,t3,0x88);
    VD u2=_mm512_shuffle_f64x2(t0,t2,0xDD), u3=_mm512_shuffle_f64x2(t1,t3,0xDD);
    VD u4=_mm512_shuffle_f64x2(t4,t6,0x88), u5=_mm512_shuffle_f64x2(t5,t7,0x88);
    VD u6=_mm512_shuffle_f64x2(t4,t6,0xDD), u7=_mm512_shuffle_f64x2(t5,t7,0xDD);
    r[0]=_mm512_shuffle_f64x2(u0,u4,0x88); r[4]=_mm512_shuffle_f64x2(u0,u4,0xDD);
    r[1]=_mm512_shuffle_f64x2(u1,u5,0x88); r[5]=_mm512_shuffle_f64x2(u1,u5,0xDD);
    r[2]=_mm512_shuffle_f64x2(u2,u6,0x88); r[6]=_mm512_shuffle_f64x2(u2,u6,0xDD);
    r[3]=_mm512_shuffle_f64x2(u3,u7,0x88); r[7]=_mm512_shuffle_f64x2(u3,u7,0xDD);
}

/* interleaved rows (nrows <= 8 valid) -> split SoA vectors xr[L], xi[L] */
static void load_group(const double *row0, long rowstride, int nrows, int L,
                       double *restrict xr, double *restrict xi){
    const int nd = 2*L;
    for(int c0=0; c0<nd; c0+=8){
        int nc = nd-c0; if(nc>8) nc=8;
        VD r[8];
        if(nc==8){
            for(int q=0;q<8;q++)
                r[q] = (q<nrows) ? _mm512_loadu_pd(row0+(long)q*rowstride+c0)
                                 : _mm512_setzero_pd();
        }else{
            __mmask8 mk = (__mmask8)((1u<<nc)-1u);
            for(int q=0;q<8;q++)
                r[q] = (q<nrows) ? _mm512_maskz_loadu_pd(mk, row0+(long)q*rowstride+c0)
                                 : _mm512_setzero_pd();
        }
        tr8(r);
        for(int c=0;c<nc;c++){
            int col=c0+c, e=col>>1;
            STA(((col&1)?xi:xr)+8*(long)e, r[c]);
        }
    }
}

/* split SoA vectors (element j taken from slot perm[j] if perm) -> interleaved rows */
static void store_group(double *row0, long rowstride, int nrows, int L,
                        const int *perm,
                        const double *restrict yr, const double *restrict yi){
    const int nd = 2*L;
    for(int c0=0; c0<nd; c0+=8){
        int nc = nd-c0; if(nc>8) nc=8;
        VD r[8];
        for(int c=0;c<8;c++){
            if(c<nc){
                int col=c0+c, e=col>>1, s = perm?perm[e]:e;
                r[c] = LDA(((col&1)?yi:yr)+8*(long)s);
            }else r[c]=_mm512_setzero_pd();
        }
        tr8(r);
        if(nc==8){
            for(int q=0;q<nrows;q++)
                _mm512_storeu_pd(row0+(long)q*rowstride+c0, r[q]);
        }else{
            __mmask8 mk = (__mmask8)((1u<<nc)-1u);
            for(int q=0;q<nrows;q++)
                _mm512_mask_storeu_pd(row0+(long)q*rowstride+c0, mk, r[q]);
        }
    }
}

/* ---------------- primes 13 / 31: conjugate-symmetric dense ----------------
 * y_k = x0 + sum_j [ e_j cos(2pi jk/P) - i o_j sin(2pi jk/P) ],  e/o = x_j +- x_{P-j}
 * 4 FMAs per (j,k) with 4 accumulators; k blocked by 3 (H = 6 and 15 both divide). */
static inline void kern_prime_g(const int P, const int H, const fft1d_plan *p,
                                const double *restrict xr, const double *restrict xi,
                                double *restrict yr, double *restrict yi){
    double *er=p->er, *ei=p->ei, *odr=p->odr, *odi=p->odi;
    VD s0r=LDA(xr), s0i=LDA(xi);
    VD accr=s0r, acci=s0i;
    for(int j=1;j<=H;j++){
        VD a=LDA(xr+8*j), b=LDA(xr+8*(long)(P-j));
        VD c=LDA(xi+8*j), d=LDA(xi+8*(long)(P-j));
        VD e1=a+b, e2=c+d;
        STA(er+8*j,e1); STA(ei+8*j,e2);
        STA(odr+8*j,a-b); STA(odi+8*j,c-d);
        accr=accr+e1; acci=acci+e2;
    }
    STA(yr,accr); STA(yi,acci);
    const double *ct=p->ct, *st=p->st;
    for(int k=1;k<=H;k+=3){
        VD A0r=s0r,A0i=s0i,B0r=_mm512_setzero_pd(),B0i=_mm512_setzero_pd();
        VD A1r=s0r,A1i=s0i,B1r=_mm512_setzero_pd(),B1i=_mm512_setzero_pd();
        VD A2r=s0r,A2i=s0i,B2r=_mm512_setzero_pd(),B2i=_mm512_setzero_pd();
        const double *c0=ct+(size_t)(k-1)*H-1, *s0=st+(size_t)(k-1)*H-1;
        const double *c1=c0+H, *s1=s0+H, *c2=c1+H, *s2=s1+H;
        for(int j=1;j<=H;j++){
            VD e_r=LDA(er+8*j), e_i=LDA(ei+8*j);
            VD o_r=LDA(odr+8*j), o_i=LDA(odi+8*j);
            A0r=FMA(e_r,SET1(c0[j]),A0r); A0i=FMA(e_i,SET1(c0[j]),A0i);
            B0r=FMA(o_i,SET1(s0[j]),B0r); B0i=FMA(o_r,SET1(s0[j]),B0i);
            A1r=FMA(e_r,SET1(c1[j]),A1r); A1i=FMA(e_i,SET1(c1[j]),A1i);
            B1r=FMA(o_i,SET1(s1[j]),B1r); B1i=FMA(o_r,SET1(s1[j]),B1i);
            A2r=FMA(e_r,SET1(c2[j]),A2r); A2i=FMA(e_i,SET1(c2[j]),A2i);
            B2r=FMA(o_i,SET1(s2[j]),B2r); B2i=FMA(o_r,SET1(s2[j]),B2i);
        }
        STA(yr+8*(long)k,      A0r+B0r); STA(yi+8*(long)k,      A0i-B0i);
        STA(yr+8*(long)(P-k),  A0r-B0r); STA(yi+8*(long)(P-k),  A0i+B0i);
        STA(yr+8*(long)(k+1),  A1r+B1r); STA(yi+8*(long)(k+1),  A1i-B1i);
        STA(yr+8*(long)(P-k-1),A1r-B1r); STA(yi+8*(long)(P-k-1),A1i+B1i);
        STA(yr+8*(long)(k+2),  A2r+B2r); STA(yi+8*(long)(k+2),  A2i-B2i);
        STA(yr+8*(long)(P-k-2),A2r-B2r); STA(yi+8*(long)(P-k-2),A2i+B2i);
    }
}
static void kern_13(const fft1d_plan *p,const double *xr,const double *xi,double *yr,double *yi)
{ kern_prime_g(13,6,p,xr,xi,yr,yi); }
static void kern_31(const fft1d_plan *p,const double *xr,const double *xi,double *yr,double *yi)
{ kern_prime_g(31,15,p,xr,xi,yr,yi); }

/* ---------------- pow2: recursive radix-4, base 2/4 ---------------- */
#define DFT4_CORE(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i,R0,I0,R1,I1,R2,I2,R3,I3) do{ \
    VD t0r=(a0r)+(a2r), t0i=(a0i)+(a2i), t1r=(a0r)-(a2r), t1i=(a0i)-(a2i);      \
    VD t2r=(a1r)+(a3r), t2i=(a1i)+(a3i), t3r=(a1r)-(a3r), t3i=(a1i)-(a3i);      \
    R0=t0r+t2r; I0=t0i+t2i; R2=t0r-t2r; I2=t0i-t2i;                             \
    R1=t1r+t3i; I1=t1i-t3r; R3=t1r-t3i; I3=t1i+t3r; }while(0)

static inline void dft4_ip(double *restrict br, double *restrict bi,
                           long i0,long i1,long i2,long i3){
    VD a0r=LDA(br+8*i0), a0i=LDA(bi+8*i0);
    VD a1r=LDA(br+8*i1), a1i=LDA(bi+8*i1);
    VD a2r=LDA(br+8*i2), a2i=LDA(bi+8*i2);
    VD a3r=LDA(br+8*i3), a3i=LDA(bi+8*i3);
    VD R0,I0,R1,I1,R2,I2,R3,I3;
    DFT4_CORE(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i,R0,I0,R1,I1,R2,I2,R3,I3);
    STA(br+8*i0,R0); STA(bi+8*i0,I0); STA(br+8*i1,R1); STA(bi+8*i1,I1);
    STA(br+8*i2,R2); STA(bi+8*i2,I2); STA(br+8*i3,R3); STA(bi+8*i3,I3);
}

static void rec4(const fft1d_plan *p, int N, int lvl,
                 const double *restrict xr, const double *restrict xi, long s,
                 double *restrict yr, double *restrict yi){
    if(N==2){
        VD a=LDA(xr), b=LDA(xr+8*s), c=LDA(xi), d=LDA(xi+8*s);
        STA(yr,a+b); STA(yr+8,a-b); STA(yi,c+d); STA(yi+8,c-d);
        return;
    }
    if(N==4){
        VD a0r=LDA(xr),      a0i=LDA(xi);
        VD a1r=LDA(xr+8*s),  a1i=LDA(xi+8*s);
        VD a2r=LDA(xr+16*s), a2i=LDA(xi+16*s);
        VD a3r=LDA(xr+24*s), a3i=LDA(xi+24*s);
        VD R0,I0,R1,I1,R2,I2,R3,I3;
        DFT4_CORE(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i,R0,I0,R1,I1,R2,I2,R3,I3);
        STA(yr,R0); STA(yi,I0); STA(yr+8,R1); STA(yi+8,I1);
        STA(yr+16,R2); STA(yi+16,I2); STA(yr+24,R3); STA(yi+24,I3);
        return;
    }
    const int M = N/4;
    for(int q=0;q<4;q++)
        rec4(p, M, lvl+1, xr+8*s*q, xi+8*s*q, 4*s, yr+8*(long)M*q, yi+8*(long)M*q);
    dft4_ip(yr, yi, 0, M, 2L*M, 3L*M);
    const double *tw = p->tw2 + p->twoff[lvl];
    for(int t=1;t<M;t++){
        const double *w = tw + (size_t)(t-1)*6;
        long i0=t, i1=M+t, i2=2L*M+t, i3=3L*M+t;
        VD a0r=LDA(yr+8*i0), a0i=LDA(yi+8*i0);
        VD x1r=LDA(yr+8*i1), x1i=LDA(yi+8*i1);
        VD x2r=LDA(yr+8*i2), x2i=LDA(yi+8*i2);
        VD x3r=LDA(yr+8*i3), x3i=LDA(yi+8*i3);
        VD c1=SET1(w[0]), s1=SET1(w[1]), c2v=SET1(w[2]), s2v=SET1(w[3]),
           c3=SET1(w[4]), s3=SET1(w[5]);
        VD a1r=FMA(x1i,s1,x1r*c1),  a1i=FNMA(x1r,s1,x1i*c1);
        VD a2r=FMA(x2i,s2v,x2r*c2v),a2i=FNMA(x2r,s2v,x2i*c2v);
        VD a3r=FMA(x3i,s3,x3r*c3),  a3i=FNMA(x3r,s3,x3i*c3);
        VD R0,I0,R1,I1,R2,I2,R3,I3;
        DFT4_CORE(a0r,a0i,a1r,a1i,a2r,a2i,a3r,a3i,R0,I0,R1,I1,R2,I2,R3,I3);
        STA(yr+8*i0,R0); STA(yi+8*i0,I0); STA(yr+8*i1,R1); STA(yi+8*i1,I1);
        STA(yr+8*i2,R2); STA(yi+8*i2,I2); STA(yr+8*i3,R3); STA(yi+8*i3,I3);
    }
}
/* ---------------- PFA 60 = 3 x 4 x 5, twiddle-free ---------------- */
static const double C51 =  0.30901699437494742410;  /* cos 72  */
static const double C52 = -0.80901699437494742410;  /* cos 144 */
static const double S51 =  0.95105651629515357212;  /* sin 72  */
static const double S52 =  0.58778525229247312917;  /* sin 144 */
static const double K3  =  0.86602540378443864676;  /* sqrt(3)/2 */

static inline void dft5(const double *restrict xr, const double *restrict xi,
                        const int *ix, double *restrict yr, double *restrict yi){
    VD x0r=LDA(xr+8*(long)ix[0]), x0i=LDA(xi+8*(long)ix[0]);
    VD x1r=LDA(xr+8*(long)ix[1]), x1i=LDA(xi+8*(long)ix[1]);
    VD x2r=LDA(xr+8*(long)ix[2]), x2i=LDA(xi+8*(long)ix[2]);
    VD x3r=LDA(xr+8*(long)ix[3]), x3i=LDA(xi+8*(long)ix[3]);
    VD x4r=LDA(xr+8*(long)ix[4]), x4i=LDA(xi+8*(long)ix[4]);
    VD e1r=x1r+x4r, e1i=x1i+x4i, o1r=x1r-x4r, o1i=x1i-x4i;
    VD e2r=x2r+x3r, e2i=x2i+x3i, o2r=x2r-x3r, o2i=x2i-x3i;
    STA(yr, x0r+e1r+e2r); STA(yi, x0i+e1i+e2i);
    VD vc1=SET1(C51), vc2=SET1(C52), vs1=SET1(S51), vs2=SET1(S52);
    VD Ar=FMA(e1r,vc1,FMA(e2r,vc2,x0r)), Ai=FMA(e1i,vc1,FMA(e2i,vc2,x0i));
    VD Br=FMA(o1i,vs1,o2i*vs2),          Bi=FMA(o1r,vs1,o2r*vs2);
    STA(yr+8,  Ar+Br); STA(yi+8,  Ai-Bi);
    STA(yr+32, Ar-Br); STA(yi+32, Ai+Bi);
    VD Cr=FMA(e1r,vc2,FMA(e2r,vc1,x0r)), Ci=FMA(e1i,vc2,FMA(e2i,vc1,x0i));
    VD Dr=FNMA(o2i,vs1,o1i*vs2),         Di=FNMA(o2r,vs1,o1r*vs2);
    STA(yr+16, Cr+Dr); STA(yi+16, Ci-Di);
    STA(yr+24, Cr-Dr); STA(yi+24, Ci+Di);
}

static inline void dft3_ip(double *restrict br, double *restrict bi,
                           long i0,long i1,long i2){
    VD x0r=LDA(br+8*i0), x0i=LDA(bi+8*i0);
    VD x1r=LDA(br+8*i1), x1i=LDA(bi+8*i1);
    VD x2r=LDA(br+8*i2), x2i=LDA(bi+8*i2);
    VD e_r=x1r+x2r, e_i=x1i+x2i, o_r=x1r-x2r, o_i=x1i-x2i;
    STA(br+8*i0, x0r+e_r); STA(bi+8*i0, x0i+e_i);
    VD half=SET1(0.5), k3=SET1(K3);
    VD ur=FNMA(e_r,half,x0r), ui=FNMA(e_i,half,x0i);
    VD g=o_i*k3, h=o_r*k3;
    STA(br+8*i1, ur+g); STA(bi+8*i1, ui-h);
    STA(br+8*i2, ur-g); STA(bi+8*i2, ui+h);
}

static void kern_60(const fft1d_plan *p, const double *restrict xr,
                    const double *restrict xi, double *restrict br, double *restrict bi){
    for(int g=0; g<12; g++)
        dft5(xr, xi, p->pfa_in + g*5, br+8*(long)(g*5), bi+8*(long)(g*5));
    for(int a1=0;a1<3;a1++)
        for(int k3=0;k3<5;k3++){
            long b0 = a1*20 + k3;
            dft4_ip(br,bi, b0, b0+5, b0+10, b0+15);
        }
    for(int t=0;t<20;t++) dft3_ip(br,bi, t, t+20, t+40);
}

/* ---------------- dispatch ---------------- */
static void run_kernel(const fft1d_plan *p, const double *xr, const double *xi,
                       double *yr, double *yi){
    switch(p->L){
    case 13: kern_13(p,xr,xi,yr,yi); break;
    case 31: kern_31(p,xr,xi,yr,yi); break;
    case 60: kern_60(p,xr,xi,yr,yi); break;
    default: rec4(p, p->L, 0, xr, xi, 1, yr, yi); break;
    }
}

void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out){
    const int L=p->L, B=p->batch;
    const long rs = 2L*(long)L;
    const double *ind=(const double*)in;
    double *outd=(double*)out;
    const int *perm = (L==60) ? p->pfa_out : NULL;
    for(int b0=0;b0<B;b0+=8){
        int nv = B-b0; if(nv>8) nv=8;
        load_group(ind + (long)b0*rs, rs, nv, L, p->ar, p->ai);
        run_kernel(p, p->ar, p->ai, p->br, p->bi);
        store_group(outd + (long)b0*rs, rs, nv, L, perm, p->br, p->bi);
    }
}

/* map: s <- z/(1+|z|), z = f + c. rsqrt14/rcp14 + 2 Newton steps each: ~1ulp,
 * no divider port. m2 clamped so |z|=0 cannot make 0*inf. */
static inline void map8(VD fr, VD fi, VD c_r, VD c_i, VD *outr, VD *outi){
    VD re=fr+c_r, im=fi+c_i;
    VD m2=FMA(re,re,im*im);
    m2=_mm512_max_pd(m2,SET1(1e-300));
    VD y=_mm512_rsqrt14_pd(m2);
    VD half=SET1(0.5), th=SET1(1.5), one=SET1(1.0), two=SET1(2.0);
    VD hm2=m2*half;
    y = y*(th - hm2*y*y);
    y = y*(th - hm2*y*y);
    VD mag = m2*y;
    VD d = one + mag;
    VD r = _mm512_rcp14_pd(d);
    r = r*(two - d*r);
    r = r*(two - d*r);
    *outr = re*r; *outi = im*r;
}

void fft1d_chain(fft1d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m){
    const int L=p->L, B=p->batch;
    const long rs = 2L*(long)L;
    const double *x0d=(const double*)x0, *cd=(const double*)c;
    double *od=(double*)final_out;
    const int *perm = (L==60) ? p->pfa_out : NULL;
    for(int b0=0;b0<B;b0+=8){
        int nv = B-b0; if(nv>8) nv=8;
        load_group(x0d + (long)b0*rs, rs, nv, L, p->sr, p->si);
        load_group(cd  + (long)b0*rs, rs, nv, L, p->cr, p->ci);
        for(int s=0;s<m;s++){
            run_kernel(p, p->sr, p->si, p->br, p->bi);
            for(int j=0;j<L;j++){
                long q = perm ? perm[j] : j;
                VD zr=LDA(p->br+8*q), zi=LDA(p->bi+8*q);
                VD or_, oi_;
                map8(zr, zi, LDA(p->cr+8*(long)j), LDA(p->ci+8*(long)j), &or_, &oi_);
                STA(p->sr+8*(long)j, or_); STA(p->si+8*(long)j, oi_);
            }
        }
        store_group(od + (long)b0*rs, rs, nv, L, NULL, p->sr, p->si);
    }
}

/* ---------------- plan setup ---------------- */
static const long double PI_L = 3.14159265358979323846264338327950288L;

fft1d_plan *fft1d_create(int L, int batch){
    if(!fft1d_supports(L)) return NULL;
    fft1d_plan *p = calloc(1, sizeof *p);
    if(!p) return NULL;
    p->L=L; p->batch=batch;

    /* one slab: 12 arrays of L vectors (8 doubles each) */
    size_t vecs = 12u*(size_t)L;
    void *mem=NULL;
    if(posix_memalign(&mem, 64, vecs*8*sizeof(double))){ free(p); return NULL; }
    p->mem = mem;
    double *q = p->mem;
    p->ar=q; q+=8L*L; p->ai=q; q+=8L*L; p->br=q; q+=8L*L; p->bi=q; q+=8L*L;
    p->sr=q; q+=8L*L; p->si=q; q+=8L*L; p->cr=q; q+=8L*L; p->ci=q; q+=8L*L;
    p->er=q; q+=8L*L; p->ei=q; q+=8L*L; p->odr=q; q+=8L*L; p->odi=q;

    if(L==13 || L==31){
        int H=(L-1)/2; p->H=H;
        p->ct=malloc(2u*(size_t)H*H*sizeof(double));
        if(!p->ct){ free(p->mem); free(p); return NULL; }
        p->st=p->ct+(size_t)H*H;
        for(int k=1;k<=H;k++) for(int j=1;j<=H;j++){
            long double ph = 2.0L*PI_L*(long double)((j*k)%L)/(long double)L;
            p->ct[(size_t)(k-1)*H+j-1]=(double)cosl(ph);
            p->st[(size_t)(k-1)*H+j-1]=(double)sinl(ph);
        }
    }else if(L==60){
        for(int a1=0;a1<3;a1++) for(int a2=0;a2<4;a2++) for(int a3=0;a3<5;a3++)
            p->pfa_in[(a1*4+a2)*5+a3] = (20*a1 + 15*a2 + 12*a3) % 60;
        for(int k=0;k<60;k++)
            p->pfa_out[k] = (k%3)*20 + (k%4)*5 + (k%5);
    }else{ /* 32 / 64 / 128 */
        long total=0; int nl=0;
        for(int N=L; N>4; N/=4){ p->twoff[nl++]=total; total += ((long)N/4-1)*6; }
        p->tw2=malloc((size_t)total*sizeof(double));
        if(!p->tw2){ free(p->mem); free(p); return NULL; }
        nl=0;
        for(int N=L; N>4; N/=4){
            double *tw = p->tw2 + p->twoff[nl++];
            int M=N/4;
            for(int t=1;t<M;t++)
                for(int pp=1;pp<=3;pp++){
                    long double ph = 2.0L*PI_L*(long double)((long)pp*t % N)/(long double)N;
                    tw[(size_t)(t-1)*6 + (pp-1)*2    ] = (double)cosl(ph);
                    tw[(size_t)(t-1)*6 + (pp-1)*2 + 1] = (double)sinl(ph);
                }
        }
    }
    return p;
}

void fft1d_destroy(fft1d_plan *p){
    if(!p) return;
    free(p->ct); free(p->tw2); free(p->mem); free(p);
}

#else  /* no AVX-512: correct scalar fallback so the file builds anywhere */

struct fft1d_plan { int L, batch; double _Complex *w; };
fft1d_plan *fft1d_create(int L, int batch){
    if(!fft1d_supports(L)) return NULL;
    fft1d_plan *p = malloc(sizeof *p); if(!p) return NULL; p->L=L; p->batch=batch;
    p->w = malloc((size_t)L*L*sizeof *p->w); if(!p->w){ free(p); return NULL; }
    for(int k=0;k<L;k++) for(int j=0;j<L;j++){
        double ph=-2.0*M_PI*((k*j)%L)/L; p->w[(size_t)k*L+j]=cos(ph)+I*sin(ph);
    }
    return p;
}
void fft1d_execute(fft1d_plan *p, const double _Complex *in, double _Complex *out){
    int L=p->L;
    for(int b=0;b<p->batch;b++){
        const double _Complex *x=in+(size_t)b*L; double _Complex *y=out+(size_t)b*L;
        for(int k=0;k<L;k++){
            double _Complex s=0; const double _Complex *wr=p->w+(size_t)k*L;
            for(int j=0;j<L;j++) s+=wr[j]*x[j];
            y[k]=s;
        }
    }
}
void fft1d_destroy(fft1d_plan *p){ if(!p) return; free(p->w); free(p); }

#endif
