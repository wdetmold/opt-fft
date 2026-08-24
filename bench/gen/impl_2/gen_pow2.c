/* gen_pow2: the 2^k-axes class entry.  L=32 is the scored size (B=8, m=250
 * graded map chain); 16/64/128 are accepted unscored and served by a generic
 * scalar radix-2 path until the custody engine is generalized over G = L/8.
 *
 * The L=32 fast path is the ice campaign's winning L64_blocked structure
 * (rounds ice_r5..r8) re-derived for G=4 slots:
 *   - CUSTODY LAYOUT: the chain state lives split-complex at slot
 *     x*XS + y*KS + g*16 doubles (re at +0, im at +8, lanes = 8 ADJACENT z,
 *     z = 8g + l).  KS = 72 doubles (9 lines) and XS = 32*KS + 8 (289 lines)
 *     are both odd line counts -- the standing 4K-aliasing proofing.
 *     One volume is 578 KB; state + custody c = 1.16 MB, essentially
 *     L2-RESIDENT on Ice Lake, and each volume iterates through all m chain
 *     steps while cache-hot (corpus 10 3 / 1000f989 residency directive).
 *   - FUSED MAP (gen_r2): the graded map state <- (z+c)/(1+|z+c|) is fused
 *     into each step's x-pass stores (vfft32m); MAP8V is the ice_r8 rsqrt14 +
 *     2-quadratic-Newton sqrt with ONE vdivpd for 1/(1+sqrt) -- the divider
 *     runs beside the saturated FMA ports and the fusion deletes the old
 *     map prepass's 1.16 MB/step L2 round trip.  Step m stays raw z for the
 *     epilogue; a plan-b prepass form is kept under GP2_PREMAP=1.
 *   - Z-LINE PAIR: 32 = 4 slots x 8 lanes does not fill an 8x8 transpose, so
 *     TWO adjacent y-rows share one TR8: per row DFT4 over slots, lane
 *     twiddle W32^{l*k2}, stack both rows' 4 vectors, TR8, one DFT8 over the
 *     former lane axis, and two vshuff64x2 per output slot put X[k] back at
 *     slot k/8 lane k%8 -- custody form in, custody form out, no bit
 *     reversal, stable across steps.
 *   - Y/X PASSES: vertical split-complex 32-point FFT (4 x DFT8 strided,
 *     twiddle W32^{b*k2}, 8 x DFT4), two register-array passes per line.
 *
 * Attribution: DFT8S / TR8 / CTWS / CTWV / MAP8V / DEIN / ILV and the whole
 * custody+lazy-map architecture are lifted from bench/ice L64_blocked
 * (ice_r8 exemplar), itself descended from rival pipeline 1000f989 and
 * L64_radix8 -- see strategies/gen_pow2.md for the full lineage. */
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../fft3d_api.h"

#ifdef __AVX512F__
#include <immintrin.h>
#endif

const char *fft3d_name(void) { return "gen_pow2"; }
const char *fft3d_description(void)
{
    return "2^k axes: L32 custody split-complex chain (4x8 z-pair TR8, lazy exact map),"
           " 16/64/128 generic radix-2";
}
int fft3d_supports(int L) { return L == 32 || L == 16 || L == 64 || L == 128; }

/* ---------------------------------------------------------------- plan --- */
struct fft3d_plan {
    int L, batch, fast;
    /* generic scalar path (16/64/128, and any L if AVX-512 is absent) */
    double *twc, *tws;      /* W_L^j, j < L */
    int *brev;
    double *gre, *gim;      /* one line of gather scratch */
    double *gstate, *gz;    /* chain ping buffers, batch*V complex each */
    /* fast path: custody buffers, one volume each */
    double *S, *C;
    void *blk;              /* single allocation backing S and C */
};

/* ============================= generic scalar path ======================= */

static void gfft_line(const fft3d_plan *p, double *base, long cstride)
{
    const int L = p->L;
    double *re = p->gre, *im = p->gim;
    for (int i = 0; i < L; ++i) {
        long j = p->brev[i];
        re[i] = base[2 * j * cstride];
        im[i] = base[2 * j * cstride + 1];
    }
    for (int len = 2; len <= L; len <<= 1) {
        int half = len >> 1, step = L / len;
        for (int i = 0; i < L; i += len)
            for (int j = 0; j < half; ++j) {
                double wr = p->twc[j * step], wi = p->tws[j * step];
                double vr = re[i + j + half], vi = im[i + j + half];
                double tr = vr * wr - vi * wi, ti = vr * wi + vi * wr;
                double ur = re[i + j], ui = im[i + j];
                re[i + j] = ur + tr;        im[i + j] = ui + ti;
                re[i + j + half] = ur - tr; im[i + j + half] = ui - ti;
            }
    }
    for (int i = 0; i < L; ++i) {
        base[2 * i * cstride] = re[i];
        base[2 * i * cstride + 1] = im[i];
    }
}

/* in-place forward 3D FFT of one interleaved volume */
static void gfft_vol(const fft3d_plan *p, double *vol)
{
    const long L = p->L, LL = L * L, V = LL * L;
    for (long r = 0; r < V; r += L) gfft_line(p, vol + 2 * r, 1);
    for (long x = 0; x < L; ++x)
        for (long z = 0; z < L; ++z) gfft_line(p, vol + 2 * (x * LL + z), L);
    for (long y = 0; y < L; ++y)
        for (long z = 0; z < L; ++z) gfft_line(p, vol + 2 * (y * L + z), LL);
}

static void generic_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    const size_t V = (size_t)p->L * p->L * p->L;
    memcpy(out, in, (size_t)p->batch * V * sizeof *out);
    for (int b = 0; b < p->batch; ++b) gfft_vol(p, (double *)out + (size_t)b * V * 2);
}

static void generic_chain(fft3d_plan *p, const double _Complex *x0,
                          const double _Complex *c, double _Complex *out, int m)
{
    const size_t V = (size_t)p->L * p->L * p->L;
    const size_t cnt = (size_t)p->batch * V, bytes = cnt * sizeof(double _Complex);
    const double *cd = (const double *)c;
    memcpy(p->gstate, x0, bytes);
    for (int s = 0; s < m; ++s) {
        memcpy(p->gz, p->gstate, bytes);
        for (int b = 0; b < p->batch; ++b) gfft_vol(p, p->gz + (size_t)b * V * 2);
        for (size_t i = 0; i < cnt; ++i) {
            double wr = p->gz[2 * i] + cd[2 * i];
            double wi = p->gz[2 * i + 1] + cd[2 * i + 1];
            double sc = 1.0 / (1.0 + sqrt(wr * wr + wi * wi));
            p->gstate[2 * i] = wr * sc;
            p->gstate[2 * i + 1] = wi * sc;
        }
    }
    memcpy(out, p->gstate, bytes);
}

/* ============================ L=32 fast path ============================= */
#ifdef __AVX512F__

typedef double    v8d __attribute__((vector_size(64)));
typedef long long v8i __attribute__((vector_size(64)));

#ifdef __clang__
# define VSH8(a,b,...) __builtin_shufflevector((v8d)(a),(v8d)(b), __VA_ARGS__)
#else
# define VSH8(a,b,...) __builtin_shuffle((v8d)(a),(v8d)(b),(v8i){__VA_ARGS__})
#endif
#define V8FMA(a,b,c)  ((v8d)_mm512_fmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
#define V8FNMA(a,b,c) ((v8d)_mm512_fnmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
#define VSPL8(x)      ((v8d){(x),(x),(x),(x),(x),(x),(x),(x)})
#define CS8V          VSPL8(0.70710678118654752440084436210485)

#define DEIN_RE(A,B) VSH8(A,B, 0,2,4,6,8,10,12,14)
#define DEIN_IM(A,B) VSH8(A,B, 1,3,5,7,9,11,13,15)
#define ILV_LO(R,I)  VSH8(R,I, 0,8,1,9,2,10,3,11)
#define ILV_HI(R,I)  VSH8(R,I, 4,12,5,13,6,14,7,15)

/* the graded map, split form, EXACT tier.  GP2_MAPDIV selects the tail:
 *   0 (ice_r8 MAPDIV=3): rsqrt14 + 2 quadratic Newtons, rcp14 + 2 quadratic
 *     Newtons; ~2-4 ulp/point, no divider, 16 FMA-port ops + 2 seeds per 8
 *     complex points.
 *   1 (SHIPS; gen_dense_prime r1 shape): same rsqrt ladder for sqrt(m2), then
 *     ONE vdivpd for 1/(1+sqrt).  12 FMA-port ops + 1 seed + 1 divider op; in
 *     the FMA-saturated x-pass (map placement GP2_PREMAP=0) the Ice Lake
 *     divider (~8 cyc/zmm) runs in parallel with the FMA ports, where the rcp
 *     ladder's 6 extra ops queue on them.  As a standalone prepass the two
 *     forms raced a WASH (that pass is L2-traffic-bound, not port-bound); the
 *     div form only pays combined with x-pass fusion.
 * 1e-300 bias makes m2 = 0 safe and is invisible (den = 1 exactly either way
 * below 1e-287). */
#ifndef GP2_MAPDIV
# define GP2_MAPDIV 1
#endif
#if GP2_MAPDIV
#define MAP8V(wr, wi, rr, ii) do {                                            \
    v8d m2_ = V8FMA((wr), (wr), V8FMA((wi), (wi), VSPL8(1e-300)));            \
    v8d q_  = (v8d)_mm512_rsqrt14_pd((__m512d)m2_);                           \
    v8d h_  = VSPL8(0.5) * m2_;                                               \
    q_ = q_ * V8FNMA(h_ * q_, q_, VSPL8(1.5));                                \
    q_ = q_ * V8FNMA(h_ * q_, q_, VSPL8(1.5));                                \
    v8d den_ = V8FMA(m2_, q_, VSPL8(1.0));                                    \
    v8d s_ = (v8d)_mm512_div_pd((__m512d)VSPL8(1.0), (__m512d)den_);          \
    (rr) = (wr) * s_; (ii) = (wi) * s_;                                       \
} while (0)
#else
#define MAP8V(wr, wi, rr, ii) do {                                            \
    v8d m2_ = V8FMA((wr), (wr), V8FMA((wi), (wi), VSPL8(1e-300)));            \
    v8d q_  = (v8d)_mm512_rsqrt14_pd((__m512d)m2_);                           \
    v8d h_  = VSPL8(0.5) * m2_;                                               \
    q_ = q_ * V8FNMA(h_ * q_, q_, VSPL8(1.5));                                \
    q_ = q_ * V8FNMA(h_ * q_, q_, VSPL8(1.5));                                \
    v8d den_ = V8FMA(m2_, q_, VSPL8(1.0));                                    \
    v8d s_ = (v8d)_mm512_rcp14_pd((__m512d)den_);                             \
    v8d e_ = V8FNMA(den_, s_, VSPL8(1.0));                                    \
    s_ = V8FMA(s_, e_, s_);                                                   \
    e_ = V8FNMA(den_, s_, VSPL8(1.0));                                        \
    s_ = V8FMA(s_, e_, s_);                                                   \
    (rr) = (wr) * s_; (ii) = (wi) * s_;                                       \
} while (0)
#endif

/* forward split-complex DFT8 on arrays of 8 (re,im) vector pairs; natural
 * order both sides; all reads before writes, so Y may alias X.  44 add/sub +
 * 8 FMA.  Lineage: ice L8_batchsimd via L64_radix8 / L64_blocked. */
#define DFT8S(XR, XI, YR, YI) do {                                            \
    v8d a0r_=(XR)[0]+(XR)[4], a0i_=(XI)[0]+(XI)[4];                           \
    v8d a1r_=(XR)[0]-(XR)[4], a1i_=(XI)[0]-(XI)[4];                           \
    v8d a2r_=(XR)[2]+(XR)[6], a2i_=(XI)[2]+(XI)[6];                           \
    v8d a3r_=(XR)[2]-(XR)[6], a3i_=(XI)[2]-(XI)[6];                           \
    v8d b0r_=(XR)[1]+(XR)[5], b0i_=(XI)[1]+(XI)[5];                           \
    v8d b1r_=(XR)[1]-(XR)[5], b1i_=(XI)[1]-(XI)[5];                           \
    v8d b2r_=(XR)[3]+(XR)[7], b2i_=(XI)[3]+(XI)[7];                           \
    v8d b3r_=(XR)[3]-(XR)[7], b3i_=(XI)[3]-(XI)[7];                           \
    v8d E0r_=a0r_+a2r_, E0i_=a0i_+a2i_, E2r_=a0r_-a2r_, E2i_=a0i_-a2i_;       \
    v8d E1r_=a1r_+a3i_, E1i_=a1i_-a3r_, E3r_=a1r_-a3i_, E3i_=a1i_+a3r_;       \
    v8d O0r_=b0r_+b2r_, O0i_=b0i_+b2i_, O2r_=b0r_-b2r_, O2i_=b0i_-b2i_;       \
    v8d O1r_=b1r_+b3i_, O1i_=b1i_-b3r_, O3r_=b1r_-b3i_, O3i_=b1i_+b3r_;       \
    v8d s1_=O1r_+O1i_, d1_=O1i_-O1r_, s3_=O3i_-O3r_, d3_=O3r_+O3i_;           \
    (YR)[0]=E0r_+O0r_; (YI)[0]=E0i_+O0i_;                                     \
    (YR)[4]=E0r_-O0r_; (YI)[4]=E0i_-O0i_;                                     \
    (YR)[2]=E2r_+O2i_; (YI)[2]=E2i_-O2r_;                                     \
    (YR)[6]=E2r_-O2i_; (YI)[6]=E2i_+O2r_;                                     \
    (YR)[1]=V8FMA (s1_,CS8V,E1r_); (YI)[1]=V8FMA (d1_,CS8V,E1i_);             \
    (YR)[5]=V8FNMA(s1_,CS8V,E1r_); (YI)[5]=V8FNMA(d1_,CS8V,E1i_);             \
    (YR)[3]=V8FMA (s3_,CS8V,E3r_); (YI)[3]=V8FNMA(d3_,CS8V,E3i_);             \
    (YR)[7]=V8FNMA(s3_,CS8V,E3r_); (YI)[7]=V8FMA (d3_,CS8V,E3i_);             \
} while (0)

/* forward split-complex DFT4, arrays of 4; all reads before writes. */
#define DFT4S(XR, XI, YR, YI) do {                                            \
    v8d t0r_=(XR)[0]+(XR)[2], t0i_=(XI)[0]+(XI)[2];                           \
    v8d t1r_=(XR)[0]-(XR)[2], t1i_=(XI)[0]-(XI)[2];                           \
    v8d t2r_=(XR)[1]+(XR)[3], t2i_=(XI)[1]+(XI)[3];                           \
    v8d t3r_=(XR)[1]-(XR)[3], t3i_=(XI)[1]-(XI)[3];                           \
    (YR)[0]=t0r_+t2r_; (YI)[0]=t0i_+t2i_;                                     \
    (YR)[2]=t0r_-t2r_; (YI)[2]=t0i_-t2i_;                                     \
    (YR)[1]=t1r_+t3i_; (YI)[1]=t1i_-t3r_;                                     \
    (YR)[3]=t1r_-t3i_; (YI)[3]=t1i_+t3r_;                                     \
} while (0)

/* (RR,II) *= (c + i*s): broadcast scalar / vector-table twiddle */
#define CTWS(RR, II, c, s) do {                                               \
    v8d cr_ = VSPL8(c), ci_ = VSPL8(s), t0_ = (RR);                           \
    (RR) = V8FNMA((II), ci_, t0_ * cr_);                                      \
    (II) = V8FMA ((II), cr_, t0_ * ci_);                                      \
} while (0)
#define CTWV(RR, II, TR, TI) do {                                             \
    v8d tr_ = (TR), ti_ = (TI), t0_ = (RR);                                   \
    (RR) = V8FNMA((II), ti_, t0_ * tr_);                                      \
    (II) = V8FMA ((II), tr_, t0_ * ti_);                                      \
} while (0)

/* 8x8 transpose of one v8d[8] block, 24 two-source shuffles */
#define TR8(V, T) do {                                                        \
    v8d w0_=VSH8((V)[0],(V)[1], 0,8,2,10,4,12,6,14);                          \
    v8d w1_=VSH8((V)[0],(V)[1], 1,9,3,11,5,13,7,15);                          \
    v8d w2_=VSH8((V)[2],(V)[3], 0,8,2,10,4,12,6,14);                          \
    v8d w3_=VSH8((V)[2],(V)[3], 1,9,3,11,5,13,7,15);                          \
    v8d w4_=VSH8((V)[4],(V)[5], 0,8,2,10,4,12,6,14);                          \
    v8d w5_=VSH8((V)[4],(V)[5], 1,9,3,11,5,13,7,15);                          \
    v8d w6_=VSH8((V)[6],(V)[7], 0,8,2,10,4,12,6,14);                          \
    v8d w7_=VSH8((V)[6],(V)[7], 1,9,3,11,5,13,7,15);                          \
    v8d x0_=VSH8(w0_,w2_, 0,1,8,9,4,5,12,13), x1_=VSH8(w0_,w2_, 2,3,10,11,6,7,14,15); \
    v8d x2_=VSH8(w1_,w3_, 0,1,8,9,4,5,12,13), x3_=VSH8(w1_,w3_, 2,3,10,11,6,7,14,15); \
    v8d x4_=VSH8(w4_,w6_, 0,1,8,9,4,5,12,13), x5_=VSH8(w4_,w6_, 2,3,10,11,6,7,14,15); \
    v8d x6_=VSH8(w5_,w7_, 0,1,8,9,4,5,12,13), x7_=VSH8(w5_,w7_, 2,3,10,11,6,7,14,15); \
    (T)[0]=VSH8(x0_,x4_, 0,1,2,3,8,9,10,11); (T)[4]=VSH8(x0_,x4_, 4,5,6,7,12,13,14,15); \
    (T)[2]=VSH8(x1_,x5_, 0,1,2,3,8,9,10,11); (T)[6]=VSH8(x1_,x5_, 4,5,6,7,12,13,14,15); \
    (T)[1]=VSH8(x2_,x6_, 0,1,2,3,8,9,10,11); (T)[5]=VSH8(x2_,x6_, 4,5,6,7,12,13,14,15); \
    (T)[3]=VSH8(x3_,x7_, 0,1,2,3,8,9,10,11); (T)[7]=VSH8(x3_,x7_, 4,5,6,7,12,13,14,15); \
} while (0)

/* custody strides (doubles): slot (x, y, g) = x*XS + y*KS + g*16, z = 8g+l.
 * Row 72 doubles = 9 lines, plane 32*72+8 = 2312 doubles = 289 lines; both
 * odd line counts. One volume = 32*XS doubles = 578 KB. */
#define KS 72
#define XS (32 * KS + 8)
#define VOLD ((size_t)32 * XS)

/* prefetch the z-row phase's next y-pair (state, and c when mapping).  Raced
 * on the node: OFF 66.6 vs ON 68.8 us/step-vol -- the working set is
 * L2-resident and the prefetches are pure issue overhead (the ice_r5 lesson
 * reconfirmed at this scale).  Default OFF; -DGP2_PF=1 re-arms it. */
#ifndef GP2_PF
# define GP2_PF 0
#endif
/* T0-prefetch the map prepass's c rows (the one L2-cold stream in sweep A) */
#ifndef GP2_PFC
# define GP2_PFC 0
#endif
/* T0-prefetch the x-pass's own-column c rows during pass 1 (xmap mode) */
#ifndef GP2_PFXC
# define GP2_PFXC 0
#endif
/* skewed z/y plane pipeline (z of plane x interleaved with y of plane x-1):
 * the z-pairs are port-5-bound (1 shuffle/point), the y-lines FMA-bound; the
 * codelet-granularity interleave gives the OOO scheduler both profiles in
 * every window.  Raced on the node (gen_r2): 59.4/59.4/59.8 vs 60.3/61.1
 * us/step-vol, bit-identical output.  SHIPS =1. */
#ifndef GP2_ZYIL
# define GP2_ZYIL 1
#endif
/* map placement: 1 = separate per-plane prepass at step start; 0 (SHIPS from
 * gen_r2) = map fused into the previous step's x-pass stores (vfft32m).
 * With the rcp-ladder map the two raced a wash in gen_r1 (63.6/64.4 vs
 * 63.4/64.9): the fusion deletes the prepass's 1.16 MB/step L2 round trip
 * but its 20 FMA ops clog the FMA-saturated x-pass.  With GP2_MAPDIV=1 the
 * 1/den moves to the otherwise-idle divider and the fusion wins outright:
 * 60.5-61.5 vs 63.6-65.4 us/step-vol (gen_r2, interleaved same-day pairs). */
#ifndef GP2_PREMAP
# define GP2_PREMAP 0
#endif

/* z-line twiddles W32^{l*k2}, lane l, k2 = 1..3; and scalar W32^j table */
static double wz32r[4][8] __attribute__((aligned(64)));
static double wz32i[4][8] __attribute__((aligned(64)));
static double tw32c[32], tw32s[32];

static void fill_fast_tables(void)
{
    const long double PI2 = 6.283185307179586476925286766559005768L;
    for (int j = 0; j < 32; ++j) {
        long double a = -PI2 * (long double)j / 32.0L;
        tw32c[j] = (double)cosl(a);
        tw32s[j] = (double)sinl(a);
    }
    for (int k2 = 0; k2 < 4; ++k2)
        for (int l = 0; l < 8; ++l) {
            long double a = -PI2 * (long double)((l * k2) % 32) / 32.0L;
            wz32r[k2][l] = (double)cosl(a);
            wz32i[k2][l] = (double)sinl(a);
        }
}

/* Two custody rows (y, y+1) of one x-plane: optional lazy map, then the
 * 32-point z-line of each, entirely in registers.
 *   per row: DFT4 over slots (k2 = k mod 4), lane twiddle W32^{l*k2}; stack
 *   both rows' 4 vectors, TR8 (lane axis <-> array axis), DFT8 over the
 *   former lanes (k1 = k div 4); X[8m+j] lands in lane (j mod 4) of
 *   O[2m + (j>=4)], so slot m re-forms with one vshuff64x2 per component. */
static inline __attribute__((always_inline))
void zpair(double *restrict rA, double *restrict rB)
{
    v8d Mr[8], Mi[8];
#pragma GCC unroll 4
    for (int g = 0; g < 4; ++g) {
        Mr[g] = *(const v8d *)(rA + 16 * g);
        Mi[g] = *(const v8d *)(rA + 16 * g + 8);
        Mr[4 + g] = *(const v8d *)(rB + 16 * g);
        Mi[4 + g] = *(const v8d *)(rB + 16 * g + 8);
    }
    DFT4S(Mr, Mi, Mr, Mi);
    DFT4S(Mr + 4, Mi + 4, Mr + 4, Mi + 4);
#pragma GCC unroll 3
    for (int k2 = 1; k2 < 4; ++k2) {
        v8d tr = *(const v8d *)wz32r[k2], ti = *(const v8d *)wz32i[k2];
        CTWV(Mr[k2], Mi[k2], tr, ti);
        CTWV(Mr[4 + k2], Mi[4 + k2], tr, ti);
    }
    v8d Tr[8], Ti[8], Or[8], Oi[8];
    TR8(Mr, Tr);
    TR8(Mi, Ti);
    DFT8S(Tr, Ti, Or, Oi);
#pragma GCC unroll 4
    for (int m = 0; m < 4; ++m) {
        *(v8d *)(rA + 16 * m)     = VSH8(Or[2 * m], Or[2 * m + 1], 0,1,2,3,8,9,10,11);
        *(v8d *)(rA + 16 * m + 8) = VSH8(Oi[2 * m], Oi[2 * m + 1], 0,1,2,3,8,9,10,11);
        *(v8d *)(rB + 16 * m)     = VSH8(Or[2 * m], Or[2 * m + 1], 4,5,6,7,12,13,14,15);
        *(v8d *)(rB + 16 * m + 8) = VSH8(Oi[2 * m], Oi[2 * m + 1], 4,5,6,7,12,13,14,15);
    }
}

/* vertical split-complex 32-point FFT over elements base + n*st doubles
 * (each element one (re,im) vector pair), in place, natural order:
 *   X[k2+8k1] = DFT4_b( W32^{b*k2} * DFT8_a( x[4a+b] ) ),  via a 4-KB
 * register-array line buffer (the FFT64S shape, ice lineage). */
static inline __attribute__((always_inline))
void vfft32(double *restrict base, const long st)
{
    v8d Hr[32], Hi[32];
#pragma GCC unroll 4
    for (int b = 0; b < 4; ++b) {
        v8d xr[8], xi[8], yr[8], yi[8];
#pragma GCC unroll 8
        for (int a = 0; a < 8; ++a) {
            const double *pp = base + (long)(4 * a + b) * st;
            xr[a] = *(const v8d *)pp;
            xi[a] = *(const v8d *)(pp + 8);
        }
        DFT8S(xr, xi, yr, yi);
#pragma GCC unroll 8
        for (int k2 = 0; k2 < 8; ++k2) { Hr[4 * k2 + b] = yr[k2]; Hi[4 * k2 + b] = yi[k2]; }
    }
#pragma GCC unroll 8
    for (int k2 = 0; k2 < 8; ++k2) {
        v8d tr[4], ti[4], zr[4], zi[4];
#pragma GCC unroll 4
        for (int b = 0; b < 4; ++b) { tr[b] = Hr[4 * k2 + b]; ti[b] = Hi[4 * k2 + b]; }
        if (k2) {
#pragma GCC unroll 3
            for (int b = 1; b < 4; ++b)
                CTWS(tr[b], ti[b], tw32c[b * k2], tw32s[b * k2]);
        }
        DFT4S(tr, ti, zr, zi);
#pragma GCC unroll 4
        for (int k1 = 0; k1 < 4; ++k1) {
            double *q = base + (long)(k2 + 8 * k1) * st;
            *(v8d *)q = zr[k1];
            *(v8d *)(q + 8) = zi[k1];
        }
    }
}

/* compile-time phase profiler (rdtsc, printed at destroy): -DGP2_PROF=1.
 * The shipped build (GP2_PROF=0) compiles all of it away. */
#ifndef GP2_PROF
# define GP2_PROF 0
#endif
#if GP2_PROF
static unsigned long long prof_m, prof_z, prof_y, prof_x, prof_steps;
# define PROF_T(v) unsigned long long v = __rdtsc()
# define PROF_ACC(acc, a, b) (acc += (b) - (a))
#else
# define PROF_T(v) do { } while (0)
# define PROF_ACC(acc, a, b) do { } while (0)
#endif

/* vfft32 with the NEXT step's lazy map fused into the pass-2 stores
 * (output-side fusion): after the x-line completes, each output element is
 * raw z of this step, so state <- (z+c)/(1+|z+c|) can be applied right here,
 * deleting the separate map prepass's state load/store round trip.  Used for
 * steps 1..m-1; step m stays raw for the epilogue.  c is read at the same
 * custody addresses as the outputs. */
static inline __attribute__((always_inline))
void vfft32m(double *restrict base, const double *restrict cb, const long st)
{
    v8d Hr[32], Hi[32];
#pragma GCC unroll 4
    for (int b = 0; b < 4; ++b) {
        v8d xr[8], xi[8], yr[8], yi[8];
#pragma GCC unroll 8
        for (int a = 0; a < 8; ++a) {
            const double *pp = base + (long)(4 * a + b) * st;
#if GP2_PFXC
            /* this column's c is consumed in pass 2, ~300 cyc from now; its
             * XS stride defeats the L2 streamer and S+C rides the L2
             * capacity edge, so hoist the (possibly L3) fill into pass 1's
             * idle load-port slots */
            __builtin_prefetch(cb + (long)(4 * a + b) * st, 0, 3);
            __builtin_prefetch(cb + (long)(4 * a + b) * st + 8, 0, 3);
#endif
            xr[a] = *(const v8d *)pp;
            xi[a] = *(const v8d *)(pp + 8);
        }
        DFT8S(xr, xi, yr, yi);
#pragma GCC unroll 8
        for (int k2 = 0; k2 < 8; ++k2) { Hr[4 * k2 + b] = yr[k2]; Hi[4 * k2 + b] = yi[k2]; }
    }
#pragma GCC unroll 8
    for (int k2 = 0; k2 < 8; ++k2) {
        v8d tr[4], ti[4], zr[4], zi[4];
#pragma GCC unroll 4
        for (int b = 0; b < 4; ++b) { tr[b] = Hr[4 * k2 + b]; ti[b] = Hi[4 * k2 + b]; }
        if (k2) {
#pragma GCC unroll 3
            for (int b = 1; b < 4; ++b)
                CTWS(tr[b], ti[b], tw32c[b * k2], tw32s[b * k2]);
        }
        DFT4S(tr, ti, zr, zi);
#pragma GCC unroll 4
        for (int k1 = 0; k1 < 4; ++k1) {
            const long n = k2 + 8 * k1;
            double *q = base + n * st;
            const double *cq = cb + n * st;
            v8d wr = zr[k1] + *(const v8d *)cq;
            v8d wi = zi[k1] + *(const v8d *)(cq + 8);
            v8d rr, ii;
            MAP8V(wr, wi, rr, ii);
            *(v8d *)q = rr;
            *(v8d *)(q + 8) = ii;
        }
    }
}

/* one full FFT step on the custody volume: per x-plane the (mapped) z-rows
 * then the plane's y-lines; then the x-lines blocked per y-row (the 32
 * touched rows -- one per plane -- stay L1-hot across the 4 slots). */
/* mode: 0 = plain FFT step (execute, and the chain's last step);
 *       1 = map PREPASS then FFT (per-plane in-place lazy map, raced control);
 *       2 = plain z/y passes, map fused into the x-pass stores (ships). */
static inline __attribute__((always_inline))
void fft_step(double *restrict S, const double *restrict C, const int mode)
{
#if GP2_ZYIL
    /* skewed z/y pipeline: plane x's z-pairs (port-5-heavy) interleaved at
     * codelet granularity with plane x-1's y-lines (FMA-heavy), so the OOO
     * window sees complementary port profiles at every codelet boundary.
     * Bit-identical: each plane's y still runs strictly after its z. */
    for (int x = 0; x <= 32; ++x) {
        double *pz = S + (size_t)x * XS;
        double *py = S + (size_t)(x ? x - 1 : 0) * XS;
        if (mode == 1 && x < 32) {
            const double *cr0 = C + (size_t)x * XS;
#pragma GCC unroll 2
            for (int y = 0; y < 32; ++y) {
                double *r = pz + (size_t)y * KS;
                const double *cr = cr0 + (size_t)y * KS;
#pragma GCC unroll 4
                for (int g = 0; g < 4; ++g) {
                    v8d wr = *(const v8d *)(r + 16 * g) + *(const v8d *)(cr + 16 * g);
                    v8d wi = *(const v8d *)(r + 16 * g + 8) + *(const v8d *)(cr + 16 * g + 8);
                    v8d rr, ii;
                    MAP8V(wr, wi, rr, ii);
                    *(v8d *)(r + 16 * g) = rr;
                    *(v8d *)(r + 16 * g + 8) = ii;
                }
            }
        }
        PROF_T(t0_);
#pragma GCC unroll 1
        for (int q = 0; q < 4; ++q) {
            if (x < 32)
#pragma GCC unroll 4
                for (int y = 8 * q; y < 8 * q + 8; y += 2)
                    zpair(pz + (size_t)y * KS, pz + (size_t)(y + 1) * KS);
            if (x > 0)
                vfft32(py + 16 * q, KS);
        }
        PROF_T(t1_);
        PROF_ACC(prof_z, t0_, t1_);
    }
#else
    for (int x = 0; x < 32; ++x) {
        double *pp = S + (size_t)x * XS;
        const double *cp = C + (size_t)x * XS;
        PROF_T(tm_);
        /* lazy map as its own per-plane prepass, in place in custody: a tiny
         * loop body gives the OOO core 128 independent ladders where the
         * register-fused form spilled (80 stores/pair, measured); the plane
         * (18 KB state + 18 KB c) stays L1-hot for the z-lines right below. */
        if (mode == 1) {
#pragma GCC unroll 2
            for (int y = 0; y < 32; ++y) {
                double *r = pp + (size_t)y * KS;
                const double *cr = cp + (size_t)y * KS;
#if GP2_PFC
                if (y + 2 < 32) {
#pragma GCC unroll 8
                    for (int q = 0; q < 8; ++q)
                        __builtin_prefetch(cr + 2 * KS + 8 * q, 0, 3);
                }
#endif
#pragma GCC unroll 4
                for (int g = 0; g < 4; ++g) {
                    v8d wr = *(const v8d *)(r + 16 * g) + *(const v8d *)(cr + 16 * g);
                    v8d wi = *(const v8d *)(r + 16 * g + 8) + *(const v8d *)(cr + 16 * g + 8);
                    v8d rr, ii;
                    MAP8V(wr, wi, rr, ii);
                    *(v8d *)(r + 16 * g) = rr;
                    *(v8d *)(r + 16 * g + 8) = ii;
                }
            }
        }
        PROF_T(t0_);
        PROF_ACC(prof_m, tm_, t0_);
        for (int y = 0; y < 32; y += 2) {
#if GP2_PF
            if (y + 4 < 32) {
#pragma GCC unroll 18
                for (int q = 0; q < 18; ++q)
                    __builtin_prefetch(pp + (size_t)(y + 4) * KS + 8 * q, 0, 3);
            }
#endif
            zpair(pp + (size_t)y * KS, pp + (size_t)(y + 1) * KS);
        }
        PROF_T(t1_);
        PROF_ACC(prof_z, t0_, t1_);
#pragma GCC unroll 4
        for (int g = 0; g < 4; ++g)
            vfft32(pp + 16 * g, KS);
        PROF_T(t2_);
        PROF_ACC(prof_y, t1_, t2_);
    }
#endif /* GP2_ZYIL */
    PROF_T(t3_);
    if (mode == 2) {
        for (int y = 0; y < 32; ++y)
#pragma GCC unroll 4
            for (int g = 0; g < 4; ++g)
                vfft32m(S + (size_t)y * KS + 16 * g, C + (size_t)y * KS + 16 * g, XS);
    } else {
        for (int y = 0; y < 32; ++y)
#pragma GCC unroll 4
            for (int g = 0; g < 4; ++g)
                vfft32(S + (size_t)y * KS + 16 * g, XS);
    }
    PROF_T(t4_);
    PROF_ACC(prof_x, t3_, t4_);
#if GP2_PROF
    ++prof_steps;
#endif
}

/* pre-RA scheduling on the step bodies (gen_batchlane SCHED15 / gen_powp
 * 25-family trick: pays only on spill-bound bodies; vfft32's 64-vector line
 * buffer qualifies on paper).  Raced via -DGP2_SCHED=1. */
#ifndef GP2_SCHED
# define GP2_SCHED 0
#endif
#if GP2_SCHED
# define SCHED_ATTR __attribute__((optimize("schedule-insns", "sched-pressure")))
#else
# define SCHED_ATTR
#endif

static SCHED_ATTR void fft_step_plain(double *restrict S, const double *restrict C)
{ fft_step(S, C, 0); }
static __attribute__((unused)) SCHED_ATTR void fft_step_premap(double *restrict S, const double *restrict C)
{ fft_step(S, C, 1); }
static __attribute__((unused)) SCHED_ATTR void fft_step_xmap(double *restrict S, const double *restrict C)
{ fft_step(S, C, 2); }

/* natural interleaved volume -> custody (pure deinterleave, slot g holds 8
 * adjacent z, so there is no transpose anywhere in the conversion) */
static void nat_to_cust(const double *restrict nat, double *restrict S)
{
    for (int x = 0; x < 32; ++x)
        for (int y = 0; y < 32; ++y) {
            const double *r = nat + ((size_t)x * 32 + y) * 64;
            double *q = S + (size_t)x * XS + (size_t)y * KS;
#pragma GCC unroll 4
            for (int g = 0; g < 4; ++g) {
                v8d A = *(const v8d *)(r + 16 * g), B = *(const v8d *)(r + 16 * g + 8);
                *(v8d *)(q + 16 * g) = DEIN_RE(A, B);
                *(v8d *)(q + 16 * g + 8) = DEIN_IM(A, B);
            }
        }
}

/* custody -> natural interleaved (execute path, no map) */
static void cust_to_nat(const double *restrict S, double *restrict nat)
{
    for (int x = 0; x < 32; ++x)
        for (int y = 0; y < 32; ++y) {
            const double *q = S + (size_t)x * XS + (size_t)y * KS;
            double *r = nat + ((size_t)x * 32 + y) * 64;
#pragma GCC unroll 4
            for (int g = 0; g < 4; ++g) {
                v8d re = *(const v8d *)(q + 16 * g), im = *(const v8d *)(q + 16 * g + 8);
                *(v8d *)(r + 16 * g) = ILV_LO(re, im);
                *(v8d *)(r + 16 * g + 8) = ILV_HI(re, im);
            }
        }
}

/* chain epilogue: map the final step's raw z (custody) with custody c and
 * interleave-store the driver's final_out volume */
static void cust_map_to_nat(const double *restrict S, const double *restrict C,
                            double *restrict nat)
{
    for (int x = 0; x < 32; ++x)
        for (int y = 0; y < 32; ++y) {
            const double *q = S + (size_t)x * XS + (size_t)y * KS;
            const double *cq = C + (size_t)x * XS + (size_t)y * KS;
            double *r = nat + ((size_t)x * 32 + y) * 64;
#pragma GCC unroll 4
            for (int g = 0; g < 4; ++g) {
                v8d wr = *(const v8d *)(q + 16 * g) + *(const v8d *)(cq + 16 * g);
                v8d wi = *(const v8d *)(q + 16 * g + 8) + *(const v8d *)(cq + 16 * g + 8);
                v8d rr, ii;
                MAP8V(wr, wi, rr, ii);
                *(v8d *)(r + 16 * g) = ILV_LO(rr, ii);
                *(v8d *)(r + 16 * g + 8) = ILV_HI(rr, ii);
            }
        }
}

#endif /* __AVX512F__ */

/* ================================ API ==================================== */

fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L) || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;

#ifdef __AVX512F__
    if (L == 32) {
        fill_fast_tables();
        /* one block: custody state + custody c + prefetch-overrun guard */
        size_t bytes = (2 * VOLD + 1024) * sizeof(double);
        if (posix_memalign(&p->blk, 64, bytes) != 0) { free(p); return NULL; }
        p->S = (double *)p->blk;
        p->C = p->S + VOLD;
        p->fast = 1;
        return p;
    }
#endif

    /* generic scalar path */
    const size_t V = (size_t)L * L * L;
    p->twc = malloc((size_t)L * sizeof(double));
    p->tws = malloc((size_t)L * sizeof(double));
    p->brev = malloc((size_t)L * sizeof(int));
    p->gre = malloc((size_t)L * sizeof(double));
    p->gim = malloc((size_t)L * sizeof(double));
    p->gstate = malloc((size_t)batch * V * 2 * sizeof(double));
    p->gz = malloc((size_t)batch * V * 2 * sizeof(double));
    if (!p->twc || !p->tws || !p->brev || !p->gre || !p->gim || !p->gstate || !p->gz) {
        fft3d_destroy(p);
        return NULL;
    }
    const long double PI2 = 6.283185307179586476925286766559005768L;
    for (int j = 0; j < L; ++j) {
        long double a = -PI2 * (long double)j / (long double)L;
        p->twc[j] = (double)cosl(a);
        p->tws[j] = (double)sinl(a);
    }
    int lg = 0;
    while ((1 << lg) < L) ++lg;
    for (int i = 0; i < L; ++i) {
        int r = 0;
        for (int b = 0; b < lg; ++b) r |= ((i >> b) & 1) << (lg - 1 - b);
        p->brev[i] = r;
    }
    return p;
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
#ifdef __AVX512F__
    if (p->fast) {
        const size_t vd = (size_t)32 * 32 * 32 * 2;
        for (int b = 0; b < p->batch; ++b) {
            nat_to_cust((const double *)in + (size_t)b * vd, p->S);
            fft_step_plain(p->S, p->S);
            cust_to_nat(p->S, (double *)out + (size_t)b * vd);
        }
        return;
    }
#endif
    generic_execute(p, in, out);
}

/* the graded fused chain: state <- (FFT3(state) + c) / (1 + |...|), m steps.
 * Fast path: per VOLUME, the state lives in custody form for the whole chain
 * (1.16 MB with c, ~L2-resident); steps 1..m-1 apply the map in their x-pass
 * stores, step m stays raw z and the epilogue maps it into final_out. */
void fft3d_chain(fft3d_plan *p, const double _Complex *x0, const double _Complex *c,
                 double _Complex *final_out, int m)
{
    const size_t V = (size_t)p->L * p->L * p->L;
    if (m <= 0) {
        memcpy(final_out, x0, (size_t)p->batch * V * sizeof *final_out);
        return;
    }
#ifdef __AVX512F__
    if (p->fast) {
        const size_t vd = V * 2;
        for (int b = 0; b < p->batch; ++b) {
            nat_to_cust((const double *)x0 + (size_t)b * vd, p->S);
            nat_to_cust((const double *)c + (size_t)b * vd, p->C);
#if GP2_PREMAP
            /* raced control: separate per-plane map prepass at each step start */
            fft_step_plain(p->S, p->C);
            for (int s = 1; s < m; ++s) fft_step_premap(p->S, p->C);
#else
            /* ships: steps 1..m-1 fuse the map into their x-pass stores, so
             * the state entering each step is already mapped; step m stays
             * raw z for the epilogue.  Bit-identical to the prepass scheme
             * (same MAP8V on the same values). */
            for (int s = 1; s < m; ++s) fft_step_xmap(p->S, p->C);
            fft_step_plain(p->S, p->C);
#endif
            cust_map_to_nat(p->S, p->C, (double *)final_out + (size_t)b * vd);
        }
        return;
    }
#endif
    generic_chain(p, x0, c, final_out, m);
}

void fft3d_destroy(fft3d_plan *p)
{
#if defined(__AVX512F__) && GP2_PROF
    if (prof_steps) {
        fprintf(stderr, "gp2 prof: steps=%llu  map=%.0f cyc/step  z=%.0f  y=%.0f  x=%.0f\n",
                prof_steps, (double)prof_m / prof_steps, (double)prof_z / prof_steps,
                (double)prof_y / prof_steps, (double)prof_x / prof_steps);
    }
#endif
    if (!p) return;
    free(p->blk);
    free(p->twc); free(p->tws); free(p->brev);
    free(p->gre); free(p->gim);
    free(p->gstate); free(p->gz);
    free(p);
}
