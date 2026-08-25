/* gen_pow2: the 2^k-axes class entry.  L=32 is the scored size (B=8, m=250
 * graded map chain); from gen_r4 the custody engine is GENERALIZED over
 * G = L/8: L=16 (G=2, 4 rows/TR8) and L=64 (G=8, the ice L64_blocked z-line,
 * L3 regime) run the same architecture; 2/4/8/128 stay on the generic
 * scalar radix-2 path (128 is outside round 6's 14..127 draw).  From gen_r6
 * the L=16 step gets the z/y port-profile skew (GP16_ZYIL) and the
 * dual-select FMA twiddle fold in vfft16i's pass 2 (GP16_FTW): -3.5%.
 * The custody c volume is stored X-FASTEST from gen_r4 (GP2_CT), so the
 * x-pass map fusion reads c as one contiguous stream per column.
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
 *     From gen_r5 the pass-2 twiddles are dual-select FMA-folded (GP2_FTW,
 *     literature 11 Tier 1): 420 -> 390 FMA-port ops/line, wall-clock wash
 *     on ICL (the passes are not port-bound), kept for accuracy headroom
 *     and as a cross-arch race axis.
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
#ifdef __linux__
#include <sys/mman.h>
#endif

#include "../fft3d_api.h"

#ifdef __AVX512F__
#include <immintrin.h>
#endif

const char *fft3d_name(void) { return "gen_pow2"; }
const char *fft3d_description(void)
{
    return "2^k axes: custody split-complex chain engine over G=L/8 (L=16/32/64:"
           " TR8 z-codelets, x-fastest c, lazy exact map, DSB-resident bodies,"
           " dual-select FMA-folded twiddles), other 2^k in 2..128 generic radix-2";
}
/* round-3 rule: take ANY 2^k the driver asks for (2..128).  32 is the scored
 * acceptance size; the rest run the generic path until the custody engine is
 * generalized over G = L/8. */
int fft3d_supports(int L) { return L >= 2 && L <= 128 && (L & (L - 1)) == 0; }

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
#define V8FMS(a,b,c)  ((v8d)_mm512_fmsub_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
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
 * odd line counts. One volume = 32*XS doubles = 578 KB.
 * GP2_KS=64 (gen_r5 experiment): drop the row pad entirely -- S falls
 * 578 -> 526 KB, S+C 1.11 -> 1.06 MB against the 1.25 MB L2.  Re-derived
 * hazard audit for THIS engine's access mix (not the ice one the odd-line
 * rule came from): plane stays 257 lines (odd, L2-set-uniform); y-pass
 * loads at 8-line stride hit 16 L1 sets x 4 lines < 12-way; no phase has
 * a store->load pair at equal addr mod 4K (z/y real deps forward full
 * width; cross-column offsets are +128B). */
#ifndef GP2_KS
# define GP2_KS 72
#endif
#define KS GP2_KS
#define XS (32 * KS + 8)
#define VOLD ((size_t)32 * XS)

/* c-TRANSPOSE (gen_r4): the custody c volume stored X-FASTEST, slot (y, g, x)
 * = y*CYS + g*CGS + x*16 doubles (re +0, im +8).  Only the x-pass (vfft32m)
 * and the epilogue read c, and both consume it along x -- in this layout each
 * x-pass column's 32 c-slots are ONE contiguous 4-KB stream instead of 32
 * loads at XS = 18.5 KB stride (which defeats the L2 streamer; the r2
 * GP2_PFXC prefetch attack on the same problem lost 4% to issue overhead --
 * a layout fix has no issue cost).  Same values at every read, so the chain
 * output is bit-identical.  CGS = 32*16+8 = 520 doubles (65 lines, odd). */
#ifndef GP2_CT
# define GP2_CT 1
#endif
#define CGS 520
#define CYS (4 * CGS)
#define CVOLD ((size_t)32 * CYS)

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
/* x-pass COLUMN SKEW (gen_r3): software-pipeline the x-pass at codelet
 * granularity -- column n+1's pass-1 b-group codelets run between column n's
 * pass-2 k2-group codelets, through ping-pong H buffers.  Kills the
 * pass-1/pass-2 join bubble (pass 2 cannot start until its own column's last
 * b-group stores; ~the r2 record's #1 slack item) and overlaps pass 1's
 * XS-strided L2 loads with pass 2's FMA+divider work.  Same per-column
 * arithmetic and order => bit-identical output.  Register pressure is one
 * codelet's worth (the H buffers already live in stack), unlike the
 * register-fused vfft32x2 that died in r1 (+15%).  RACED OUT in gen_r3:
 * 63.0 vs 58.1 us, x-phase 107.6K vs 89.0K cyc -- the OOO core already
 * covers the join, and the ping-pong H buffers double the stack traffic.
 * Kept compilable as the raced control; default OFF. */
#ifndef GP2_XSK
# define GP2_XSK 0
#endif
/* x-pass loop body size (gen_r3, SHIPS =1): 0 = r2 shape (g-loop unrolled 4
 * -> ~4 columns of code per iteration, ~5K uops, streams from the legacy
 * decoder at ~16 code bytes/cyc of 7-byte EVEX instructions), 1 = ONE column
 * per iteration so the ~1.3K-uop body fits the ~2.3K-uop DSB and the front
 * end feeds 6 uops/cyc.  The x-pass measured neither FMA-port- nor
 * latency-bound (793 cyc/col vs ~434 port floor; the XSK cross-column skew
 * was a wash), which pointed at the front end: =1 cut the x-phase 101.9K ->
 * 92.0K cyc same-window, bit-identical.  2 = fully rolled codelet loops
 * (~400-uop body): overshoots, loop overhead + runtime twiddle loads cost
 * +7K cyc (raced control). */
#ifndef GP2_XU
# define GP2_XU 1
#endif
/* same front-end diet for the z/y skewed loop (SHIPS =1): keep the 4-zpair
 * sub-loop rolled (one zpair body per iteration) so the q-body drops from
 * ~2K uops (over the DSB) to ~1.3K (resident).  Small but consistent:
 * z-phase 77.0-77.7K vs 77.4-82.0K cyc across four windows. */
#ifndef GP2_ZU1
# define GP2_ZU1 1
#endif
#if GP2_ZU1
# define ZPAIR_UNROLL _Pragma("GCC unroll 1")
#else
# define ZPAIR_UNROLL _Pragma("GCC unroll 4")
#endif
/* FINE z/y interleave (gen_r3 experiment): inside each q-group, alternate
 * single zpairs (shuffle-heavy) with single y-line vp1 codelets (FMA-heavy)
 * through the split vp1_b/vp2_k2 helpers, instead of a ~1000-uop zpair block
 * followed by a ~1000-uop vfft32 block -- the ~350-entry ROB cannot span a
 * whole block, so mid-vfft32 the shuffle port goes dark.  Same per-line
 * arithmetic order => bit-identical.  RACED OUT in gen_r3: 58.97 vs 57.69
 * us clean-window, z-phase 79.0K vs 77.7K cyc -- the OOO core already covers
 * the coarse boundary (same lesson as GP2_XSK on the x-pass), and the rolled
 * zpair+vp1 loop adds overhead.  Default OFF, kept as the raced control. */
#ifndef GP2_ZYF
# define GP2_ZYF 0
#endif

/* z-line twiddles W32^{l*k2}, lane l, k2 = 1..3; and scalar W32^j table */
static double wz32r[4][8] __attribute__((aligned(64)));
static double wz32i[4][8] __attribute__((aligned(64)));
static double tw32c[32], tw32s[32];

/* GP2_FTW=1 (SHIPS, gen_r5): dual-select FMA-folded pass-2 twiddles
 * (literature 11 Tier 1: Bergach arXiv:2604.00567 / Linzer-Feig, "2^k
 * codelets first" -- never before validated in performant code).  The
 * twiddle product x = W32^j * h is factored x = f * m with m computed by
 * TWO FMAs from a stored ratio, and the scale f folded into the following
 * DFT4 stage-1 butterfly (2 more FMAs replacing 4 add/sub), so
 * twiddle+butterfly drops from 8 ops to 6 per (b,k2) site.  Dual-select
 * keeps every stored ratio <= 1: c-form (|cos|>=|sin|) stores t = s/c and
 * f = c, computing m = (hr - t*hi, hi + t*hr); s-form stores u = c/s and
 * f = s, computing m = (u*hr - hi, hr + u*hi).  j = 8 (w = -i, the b=2
 * site of k2=4) needs no multiply at all: 4 pure adds.  Per 32-point line
 * the pass-2 op count falls 212 -> 182 and the line total 420 -> 390
 * (-7.1%), in BOTH the y- and x-passes.  Tables from long double, ratio
 * rounded once.  0 = the r4 CTWS+DFT4S arithmetic (raced control). */
#ifndef GP2_FTW
# define GP2_FTW 1
#endif
/* |cos| >= |sin| for angle -2*pi*j/32 */
#define FW_CFORM(J) ((((J) % 16) <= 4) || (((J) % 16) >= 12))
static double fwt32[32], fwf32[32];

static void fill_fast_tables(void)
{
    const long double PI2 = 6.283185307179586476925286766559005768L;
    for (int j = 0; j < 32; ++j) {
        long double a = -PI2 * (long double)j / 32.0L;
        tw32c[j] = (double)cosl(a);
        tw32s[j] = (double)sinl(a);
        long double c = cosl(a), s = sinl(a);
        if (FW_CFORM(j)) { fwt32[j] = (c == 0.0L) ? 0.0 : (double)(s / c); fwf32[j] = (double)c; }
        else             { fwt32[j] = (double)(c / s); fwf32[j] = (double)s; }
    }
    for (int k2 = 0; k2 < 4; ++k2)
        for (int l = 0; l < 8; ++l) {
            long double a = -PI2 * (long double)((l * k2) % 32) / 32.0L;
            wz32r[k2][l] = (double)cosl(a);
            wz32i[k2][l] = (double)sinl(a);
        }
}

/* m = the ratio-form factor of W32^j * h (x = fwf32[j] * m); j a
 * compile-time constant in every unrolled caller, so the form branch and
 * the j==8 special in twdft4 fold away. */
#define FWM(MR, MI, HR, HI, J) do {                                          \
    v8d tj_ = VSPL8(fwt32[J]);                                               \
    if (FW_CFORM(J)) {                                                       \
        (MR) = V8FNMA(tj_, (HI), (HR));                                      \
        (MI) = V8FMA (tj_, (HR), (HI));                                      \
    } else {                                                                 \
        (MR) = V8FMS (tj_, (HR), (HI));                                      \
        (MI) = V8FMA (tj_, (HI), (HR));                                      \
    }                                                                        \
} while (0)

/* pass-2 twiddle + DFT4 with the scales folded in: replaces
 * (3x CTWS; DFT4S) for k2 != 0.  All reads before writes; Z may alias T. */
static inline __attribute__((always_inline))
void twdft4(const v8d *restrict tr, const v8d *restrict ti,
            v8d *restrict zr, v8d *restrict zi, const int k2)
{
    if (k2 == 0) {
        DFT4S(tr, ti, zr, zi);
        return;
    }
    const int j1 = k2, j2 = 2 * k2, j3 = 3 * k2;
    v8d m1r, m1i, m3r, m3i;
    FWM(m1r, m1i, tr[1], ti[1], j1);
    FWM(m3r, m3i, tr[3], ti[3], j3);
    const v8d f1 = VSPL8(fwf32[j1]), f3 = VSPL8(fwf32[j3]);
    v8d t0r, t0i, t1r, t1i;
    if (j2 == 8) {           /* W32^8 = -i: x2 = (h2i, -h2r), pure adds */
        t0r = tr[0] + ti[2]; t0i = ti[0] - tr[2];
        t1r = tr[0] - ti[2]; t1i = ti[0] + tr[2];
    } else {
        v8d m2r, m2i;
        FWM(m2r, m2i, tr[2], ti[2], j2);
        const v8d f2 = VSPL8(fwf32[j2]);
        t0r = V8FMA (f2, m2r, tr[0]); t0i = V8FMA (f2, m2i, ti[0]);
        t1r = V8FNMA(f2, m2r, tr[0]); t1i = V8FNMA(f2, m2i, ti[0]);
    }
    const v8d p3r = f3 * m3r, p3i = f3 * m3i;
    const v8d t2r = V8FMA(f1, m1r, p3r), t2i = V8FMA(f1, m1i, p3i);
    const v8d t3r = V8FMS(f1, m1r, p3r), t3i = V8FMS(f1, m1i, p3i);
    zr[0] = t0r + t2r; zi[0] = t0i + t2i;
    zr[2] = t0r - t2r; zi[2] = t0i - t2i;
    zr[1] = t1r + t3i; zi[1] = t1i - t3r;
    zr[3] = t1r - t3i; zi[3] = t1i + t3r;
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
#if GP2_FTW
        twdft4(tr, ti, zr, zi, k2);
#else
        if (k2) {
#pragma GCC unroll 3
            for (int b = 1; b < 4; ++b)
                CTWS(tr[b], ti[b], tw32c[b * k2], tw32s[b * k2]);
        }
        DFT4S(tr, ti, zr, zi);
#endif
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
void vfft32m(double *restrict base, const double *restrict cb, const long st,
             const long cst)
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
            __builtin_prefetch(cb + (long)(4 * a + b) * cst, 0, 3);
            __builtin_prefetch(cb + (long)(4 * a + b) * cst + 8, 0, 3);
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
#if GP2_FTW
        twdft4(tr, ti, zr, zi, k2);
#else
        if (k2) {
#pragma GCC unroll 3
            for (int b = 1; b < 4; ++b)
                CTWS(tr[b], ti[b], tw32c[b * k2], tw32s[b * k2]);
        }
        DFT4S(tr, ti, zr, zi);
#endif
#pragma GCC unroll 4
        for (int k1 = 0; k1 < 4; ++k1) {
            const long n = k2 + 8 * k1;
            double *q = base + n * st;
            const double *cq = cb + n * cst;
            v8d wr = zr[k1] + *(const v8d *)cq;
            v8d wi = zi[k1] + *(const v8d *)(cq + 8);
            v8d rr, ii;
            MAP8V(wr, wi, rr, ii);
            *(v8d *)q = rr;
            *(v8d *)(q + 8) = ii;
        }
    }
}

/* one b-group of vfft32 pass 1: 8 strided loads, DFT8, park in H (k2-major).
 * Split out of vfft32 so the x-pass column skew can interleave it with
 * another column's pass-2 codelets. */
static inline __attribute__((always_inline))
void vp1_b(const double *restrict base, const long st,
           v8d *restrict Hr, v8d *restrict Hi, const int b)
{
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

/* one k2-group of vfft32 pass 2: twiddle, DFT4, store (with the lazy map
 * fused into the stores when domap -- compile-time constant after inlining) */
static inline __attribute__((always_inline))
void vp2_k2(double *restrict base, const double *restrict cb, const long st,
            const long cst, const v8d *restrict Hr, const v8d *restrict Hi,
            const int k2, const int domap)
{
    v8d tr[4], ti[4], zr[4], zi[4];
#pragma GCC unroll 4
    for (int b = 0; b < 4; ++b) { tr[b] = Hr[4 * k2 + b]; ti[b] = Hi[4 * k2 + b]; }
#if GP2_FTW
    twdft4(tr, ti, zr, zi, k2);
#else
    if (k2) {
#pragma GCC unroll 3
        for (int b = 1; b < 4; ++b)
            CTWS(tr[b], ti[b], tw32c[b * k2], tw32s[b * k2]);
    }
    DFT4S(tr, ti, zr, zi);
#endif
#pragma GCC unroll 4
    for (int k1 = 0; k1 < 4; ++k1) {
        const long n = k2 + 8 * k1;
        double *q = base + n * st;
        if (domap) {
            const double *cq = cb + n * cst;
            v8d wr = zr[k1] + *(const v8d *)cq;
            v8d wi = zi[k1] + *(const v8d *)(cq + 8);
            v8d rr, ii;
            MAP8V(wr, wi, rr, ii);
            *(v8d *)q = rr;
            *(v8d *)(q + 8) = ii;
        } else {
            *(v8d *)q = zr[k1];
            *(v8d *)(q + 8) = zi[k1];
        }
    }
}

/* fully-ROLLED vfft32(m): pass 1 and pass 2 as 4- and 8-iteration loops over
 * the split codelets, so one column's static code is ~400 uops instead of
 * ~1300 -- the deepest front-end diet (GP2_XU=2).  Runtime k2 makes the
 * twiddles scalar table loads and the k2=0 skip a well-predicted branch. */
static inline __attribute__((always_inline))
void vfft32r(double *restrict base, const double *restrict cb, const long st,
             const long cst, const int domap)
{
    v8d Hr[32], Hi[32];
#pragma GCC unroll 1
    for (int b = 0; b < 4; ++b)
        vp1_b(base, st, Hr, Hi, b);
#pragma GCC unroll 1
    for (int k2 = 0; k2 < 8; ++k2)
        vp2_k2(base, cb, st, cst, Hr, Hi, k2, domap);
}

/* the skewed x-pass: 128 columns cc = (y<<2)|g, base S + y*KS + 16*g, stride
 * XS.  Column cc's 4 pass-1 codelets are interleaved with column cc-1's 8
 * pass-2 codelets through ping-pong H buffers; per-column arithmetic order is
 * unchanged, so the output is bit-identical to the vfft32/vfft32m loop. */
/* c address of column cc = (y<<2)|g in the shipping layout (CT: x-fastest) */
#if GP2_CT
# define CCOL(C, cc) ((C) + (size_t)((cc) >> 2) * CYS + (size_t)((cc) & 3) * CGS)
# define CST 16
#else
# define CCOL(C, cc) ((C) + (size_t)((cc) >> 2) * KS + 16 * ((cc) & 3))
# define CST XS
#endif

static inline __attribute__((always_inline))
void xpass_skew(double *restrict S, const double *restrict C, const int domap)
{
    v8d HR[2][32], HI[2][32];
#pragma GCC unroll 4
    for (int b = 0; b < 4; ++b)
        vp1_b(S, XS, HR[0], HI[0], b);
#pragma GCC unroll 1
    for (int cc = 1; cc <= 128; ++cc) {
        const int cur = (cc - 1) & 1, nxt = cc & 1;
        double *pc = S + (size_t)((cc - 1) >> 2) * KS + 16 * ((cc - 1) & 3);
        const double *cq = CCOL(C, cc - 1);
        const double *pn = S + (size_t)(cc >> 2) * KS + 16 * (cc & 3);
        if (cc < 128) {
#pragma GCC unroll 4
            for (int q = 0; q < 4; ++q) {
                vp1_b(pn, XS, HR[nxt], HI[nxt], q);
                vp2_k2(pc, cq, XS, CST, HR[cur], HI[cur], 2 * q, domap);
                vp2_k2(pc, cq, XS, CST, HR[cur], HI[cur], 2 * q + 1, domap);
            }
        } else {
#pragma GCC unroll 8
            for (int k2 = 0; k2 < 8; ++k2)
                vp2_k2(pc, cq, XS, CST, HR[cur], HI[cur], k2, domap);
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
#pragma GCC unroll 2
            for (int y = 0; y < 32; ++y) {
                double *r = pz + (size_t)y * KS;
#pragma GCC unroll 4
                for (int g = 0; g < 4; ++g) {
#if GP2_CT
                    const double *cr = C + (size_t)y * CYS + (size_t)g * CGS + (size_t)x * 16;
#else
                    const double *cr = C + (size_t)x * XS + (size_t)y * KS + 16 * g;
#endif
                    v8d wr = *(const v8d *)(r + 16 * g) + *(const v8d *)cr;
                    v8d wi = *(const v8d *)(r + 16 * g + 8) + *(const v8d *)(cr + 8);
                    v8d rr, ii;
                    MAP8V(wr, wi, rr, ii);
                    *(v8d *)(r + 16 * g) = rr;
                    *(v8d *)(r + 16 * g + 8) = ii;
                }
            }
        }
        PROF_T(t0_);
#if GP2_ZYF
#pragma GCC unroll 1
        for (int q = 0; q < 4; ++q) {
            if (x > 0 && x < 32) {
                v8d Hr[32], Hi[32];
                double *yb = py + 16 * q;
#pragma GCC unroll 1
                for (int i = 0; i < 4; ++i) {
                    zpair(pz + (size_t)(8 * q + 2 * i) * KS,
                          pz + (size_t)(8 * q + 2 * i + 1) * KS);
                    vp1_b(yb, KS, Hr, Hi, i);
                }
#pragma GCC unroll 8
                for (int k2 = 0; k2 < 8; ++k2)
                    vp2_k2(yb, yb, KS, KS, Hr, Hi, k2, 0);
            } else if (x < 32) {
ZPAIR_UNROLL
                for (int y = 8 * q; y < 8 * q + 8; y += 2)
                    zpair(pz + (size_t)y * KS, pz + (size_t)(y + 1) * KS);
            } else {
                vfft32(py + 16 * q, KS);
            }
        }
#else
#pragma GCC unroll 1
        for (int q = 0; q < 4; ++q) {
            if (x < 32)
ZPAIR_UNROLL
                for (int y = 8 * q; y < 8 * q + 8; y += 2)
                    zpair(pz + (size_t)y * KS, pz + (size_t)(y + 1) * KS);
            if (x > 0)
                vfft32(py + 16 * q, KS);
        }
#endif
        PROF_T(t1_);
        PROF_ACC(prof_z, t0_, t1_);
    }
#else
    for (int x = 0; x < 32; ++x) {
        double *pp = S + (size_t)x * XS;
        PROF_T(tm_);
        /* lazy map as its own per-plane prepass, in place in custody: a tiny
         * loop body gives the OOO core 128 independent ladders where the
         * register-fused form spilled (80 stores/pair, measured); the plane
         * (18 KB state + 18 KB c) stays L1-hot for the z-lines right below. */
        if (mode == 1) {
#pragma GCC unroll 2
            for (int y = 0; y < 32; ++y) {
                double *r = pp + (size_t)y * KS;
#pragma GCC unroll 4
                for (int g = 0; g < 4; ++g) {
#if GP2_CT
                    const double *cr = C + (size_t)y * CYS + (size_t)g * CGS + (size_t)x * 16;
#else
                    const double *cr = C + (size_t)x * XS + (size_t)y * KS + 16 * g;
#endif
                    v8d wr = *(const v8d *)(r + 16 * g) + *(const v8d *)cr;
                    v8d wi = *(const v8d *)(r + 16 * g + 8) + *(const v8d *)(cr + 8);
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
#if GP2_XSK
    if (mode == 2)
        xpass_skew(S, C, 1);
    else
        xpass_skew(S, C, 0);
#elif GP2_XU == 2
    if (mode == 2) {
#pragma GCC unroll 1
        for (int cc = 0; cc < 128; ++cc)
            vfft32r(S + (size_t)(cc >> 2) * KS + 16 * (cc & 3),
                    CCOL(C, cc), XS, CST, 1);
    } else {
#pragma GCC unroll 1
        for (int cc = 0; cc < 128; ++cc)
            vfft32r(S + (size_t)(cc >> 2) * KS + 16 * (cc & 3),
                    CCOL(C, cc), XS, CST, 0);
    }
#elif GP2_XU
    if (mode == 2) {
#pragma GCC unroll 1
        for (int cc = 0; cc < 128; ++cc)
            vfft32m(S + (size_t)(cc >> 2) * KS + 16 * (cc & 3),
                    CCOL(C, cc), XS, CST);
    } else {
#pragma GCC unroll 1
        for (int cc = 0; cc < 128; ++cc)
            vfft32(S + (size_t)(cc >> 2) * KS + 16 * (cc & 3), XS);
    }
#else
    if (mode == 2) {
        for (int y = 0; y < 32; ++y)
#pragma GCC unroll 4
            for (int g = 0; g < 4; ++g)
                vfft32m(S + (size_t)y * KS + 16 * g, CCOL(C, 4 * y + g), XS, CST);
    } else {
        for (int y = 0; y < 32; ++y)
#pragma GCC unroll 4
            for (int g = 0; g < 4; ++g)
                vfft32(S + (size_t)y * KS + 16 * g, XS);
    }
#endif
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

/* natural interleaved c volume -> x-fastest custody (GP2_CT layout): the
 * same deinterleave, stores scattered at x*16 within each (y,g) column run.
 * One-time per chain volume; the m-step x-pass reads it sequentially. */
static void nat_to_cust_c(const double *restrict nat, double *restrict C)
{
    for (int x = 0; x < 32; ++x)
        for (int y = 0; y < 32; ++y) {
            const double *r = nat + ((size_t)x * 32 + y) * 64;
            double *q = C + (size_t)y * CYS + (size_t)x * 16;
#pragma GCC unroll 4
            for (int g = 0; g < 4; ++g) {
                v8d A = *(const v8d *)(r + 16 * g), B = *(const v8d *)(r + 16 * g + 8);
                *(v8d *)(q + (size_t)g * CGS) = DEIN_RE(A, B);
                *(v8d *)(q + (size_t)g * CGS + 8) = DEIN_IM(A, B);
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
            double *r = nat + ((size_t)x * 32 + y) * 64;
#pragma GCC unroll 4
            for (int g = 0; g < 4; ++g) {
#if GP2_CT
                const double *cq = C + (size_t)y * CYS + (size_t)g * CGS + (size_t)x * 16;
#else
                const double *cq = C + (size_t)x * XS + (size_t)y * KS + 16 * g;
#endif
                v8d wr = *(const v8d *)(q + 16 * g) + *(const v8d *)cq;
                v8d wi = *(const v8d *)(q + 16 * g + 8) + *(const v8d *)(cq + 8);
                v8d rr, ii;
                MAP8V(wr, wi, rr, ii);
                *(v8d *)(r + 16 * g) = ILV_LO(rr, ii);
                *(v8d *)(r + 16 * g + 8) = ILV_HI(rr, ii);
            }
        }
}

/* =============== generalized custody engine (gen_r4): L=64, L=16 ==========
 * The same architecture as the L=32 path, re-derived per G = L/8:
 *   G=8 (L=64): one custody row fills the 8x8 transpose -- the ice
 *     L64_blocked st=3 z-line verbatim (DFT8 over slots, lane twiddle
 *     W64^{l*k2}, TR8 pair, DFT8 over lanes, direct slot store).  Volume
 *     4.46 MB, S+C ~8.7 MB: the L3 REGIME (per-volume chain stays
 *     L3-resident; per-step traffic ~13 MB of L3, the ice lessons apply).
 *   G=2 (L=16): FOUR rows share one TR8 (DFT2 over slots, lane twiddle
 *     W16^l, stack 4x2 vectors, TR8, DFT8 over lanes, 3-shuffle granule
 *     scatter per output slot).  Volume 81 KB: everything L2-resident.
 * Vertical pass generalizes vfft32's shape: X[k2+8*k1] =
 *   DFTG_b( W_L^{b*k2} * DFT8_a( x[G*a+b] ) ), two passes through an
 * H[8G] register-array line buffer.  DSB rule from r3 applied: loops rolled
 * so no body exceeds ~1.3K uops.  c is stored x-fastest (GP2_CT) here too. */
#define K64 136
#define X64 (64 * K64 + 8)                 /* 8712 doubles = 1089 lines, odd */
#define VOLD64 ((size_t)64 * X64)
#define CGS64 (64 * 16 + 8)                /* 1032 doubles = 129 lines, odd  */
#define CYS64 ((size_t)8 * CGS64)
#define CVOLD64 ((size_t)64 * CYS64)

#define K16 40
#define X16 (16 * K16 + 8)                 /* 648 doubles = 81 lines, odd    */
#define VOLD16 ((size_t)16 * X16)
#define CGS16 (16 * 16 + 8)                /* 264 doubles = 33 lines, odd    */
#define CYS16 ((size_t)2 * CGS16)
#define CVOLD16 ((size_t)16 * CYS16)

/* gen_r6 L=16 knobs.
 * GP16_ZYIL: the L=32 z/y port-profile skew (GP2_ZYIL, +3.5% there)
 * instantiated at G=2 -- plane x's zquads (shuffle-heavy: 96 port-5 ops per
 * 64 points, vs the zpair's 64) interleaved at codelet granularity with
 * plane x-1's y-lines (FMA-heavy).  Bit-identical: each plane's y still runs
 * strictly after its own z.
 * GP16_FTW: the r5 dual-select FMA twiddle fold (literature 11 Tier 1) for
 * vfft16i's pass-2 DFT2 sites: x = W16^k2 * h factored f*m, m by two FMAs
 * from a stored ratio <= 1, f folded into the DFT2 add/sub (2 FMAs); the
 * k2 = 4 site (W16^4 = -i) is 4 pure adds.  Pass-2 twiddle+butterfly
 * 60 -> 44 ops per 16-point line, in BOTH the y- and x-passes. */
#ifndef GP16_ZYIL
# define GP16_ZYIL 1
#endif
#ifndef GP16_FTW
# define GP16_FTW 1
#endif

static double wz64r[8][8] __attribute__((aligned(64)));
static double wz64i[8][8] __attribute__((aligned(64)));
static double tw64c[64], tw64s[64];
static double wz16r[8] __attribute__((aligned(64)));
static double wz16i[8] __attribute__((aligned(64)));
static double tw16c[16], tw16s[16];
/* dual-select ratio/scale tables for the L=16 pass-2 fold (GP16_FTW);
 * |cos| >= |sin| for angle -2*pi*j/16 */
#define FW16_CFORM(J) ((((J) % 8) <= 2) || (((J) % 8) >= 6))
static double fwt16[16], fwf16[16];

static void fill_tables_64(void)
{
    const long double PI2 = 6.283185307179586476925286766559005768L;
    for (int j = 0; j < 64; ++j) {
        long double a = -PI2 * (long double)j / 64.0L;
        tw64c[j] = (double)cosl(a);
        tw64s[j] = (double)sinl(a);
    }
    for (int k2 = 0; k2 < 8; ++k2)
        for (int l = 0; l < 8; ++l) {
            long double a = -PI2 * (long double)((l * k2) % 64) / 64.0L;
            wz64r[k2][l] = (double)cosl(a);
            wz64i[k2][l] = (double)sinl(a);
        }
}

static void fill_tables_16(void)
{
    const long double PI2 = 6.283185307179586476925286766559005768L;
    for (int j = 0; j < 16; ++j) {
        long double a = -PI2 * (long double)j / 16.0L;
        tw16c[j] = (double)cosl(a);
        tw16s[j] = (double)sinl(a);
        long double c = cosl(a), s = sinl(a);
        if (FW16_CFORM(j)) { fwt16[j] = (c == 0.0L) ? 0.0 : (double)(s / c); fwf16[j] = (double)c; }
        else               { fwt16[j] = (double)(c / s); fwf16[j] = (double)s; }
    }
    for (int l = 0; l < 8; ++l) {
        long double a = -PI2 * (long double)l / 16.0L;
        wz16r[l] = (double)cosl(a);
        wz16i[l] = (double)sinl(a);
    }
}

/* one L=64 custody z-row, in place: DFT8 over slots (k2 = k mod 8), lane
 * twiddle W64^{l*k2}, TR8 pair, DFT8 over former lanes (k1 = k div 8);
 * X[8*k1 + l'] lands directly at slot k1 -- no re-form shuffles. */
static inline __attribute__((always_inline))
void zrow64(double *restrict r)
{
    v8d Mr[8], Mi[8];
#pragma GCC unroll 8
    for (int g = 0; g < 8; ++g) {
        Mr[g] = *(const v8d *)(r + 16 * g);
        Mi[g] = *(const v8d *)(r + 16 * g + 8);
    }
    DFT8S(Mr, Mi, Mr, Mi);
#pragma GCC unroll 7
    for (int k2 = 1; k2 < 8; ++k2)
        CTWV(Mr[k2], Mi[k2], *(const v8d *)wz64r[k2], *(const v8d *)wz64i[k2]);
    v8d Tr[8], Ti[8], Or[8], Oi[8];
    TR8(Mr, Tr);
    TR8(Mi, Ti);
    DFT8S(Tr, Ti, Or, Oi);
#pragma GCC unroll 8
    for (int g = 0; g < 8; ++g) {
        *(v8d *)(r + 16 * g) = Or[g];
        *(v8d *)(r + 16 * g + 8) = Oi[g];
    }
}

/* four L=16 custody z-rows sharing one TR8: per row DFT2 over the 2 slots,
 * lane twiddle W16^l on the odd branch, stack 4x2 vectors (j = 2r + k2),
 * TR8 pair, DFT8 over lanes (k1); X_r[k2+2*k1] sits in O[k1] lane (2r+k2),
 * so each output slot re-forms from 128-bit granules of 4 O vectors:
 * 2 gather shuffles + 1 merge per slot component. */
static inline __attribute__((always_inline))
void zquad16(double *restrict r0, double *restrict r1,
             double *restrict r2, double *restrict r3)
{
    v8d Ar[8], Ai[8];
    const v8d twr = *(const v8d *)wz16r, twi = *(const v8d *)wz16i;
#define ZQ_LOAD(r, p) do {                                                    \
        v8d s0r_ = *(const v8d *)(p),        s0i_ = *(const v8d *)((p) + 8);  \
        v8d s1r_ = *(const v8d *)((p) + 16), s1i_ = *(const v8d *)((p) + 24); \
        Ar[2*(r)]   = s0r_ + s1r_; Ai[2*(r)]   = s0i_ + s1i_;                 \
        Ar[2*(r)+1] = s0r_ - s1r_; Ai[2*(r)+1] = s0i_ - s1i_;                 \
        CTWV(Ar[2*(r)+1], Ai[2*(r)+1], twr, twi);                             \
    } while (0)
    ZQ_LOAD(0, r0); ZQ_LOAD(1, r1); ZQ_LOAD(2, r2); ZQ_LOAD(3, r3);
#undef ZQ_LOAD
    v8d Tr[8], Ti[8], Or[8], Oi[8];
    TR8(Ar, Tr);
    TR8(Ai, Ti);
    DFT8S(Tr, Ti, Or, Oi);
#define ZQ_SCAT(r, p) do {                                                    \
        v8d a0_ = VSH8(Or[0], Or[1], 2*(r),2*(r)+1, 8+2*(r),9+2*(r), 0,0,0,0);\
        v8d a1_ = VSH8(Or[2], Or[3], 2*(r),2*(r)+1, 8+2*(r),9+2*(r), 0,0,0,0);\
        v8d b0_ = VSH8(Oi[0], Oi[1], 2*(r),2*(r)+1, 8+2*(r),9+2*(r), 0,0,0,0);\
        v8d b1_ = VSH8(Oi[2], Oi[3], 2*(r),2*(r)+1, 8+2*(r),9+2*(r), 0,0,0,0);\
        v8d c0_ = VSH8(Or[4], Or[5], 2*(r),2*(r)+1, 8+2*(r),9+2*(r), 0,0,0,0);\
        v8d c1_ = VSH8(Or[6], Or[7], 2*(r),2*(r)+1, 8+2*(r),9+2*(r), 0,0,0,0);\
        v8d d0_ = VSH8(Oi[4], Oi[5], 2*(r),2*(r)+1, 8+2*(r),9+2*(r), 0,0,0,0);\
        v8d d1_ = VSH8(Oi[6], Oi[7], 2*(r),2*(r)+1, 8+2*(r),9+2*(r), 0,0,0,0);\
        *(v8d *)(p)      = VSH8(a0_, a1_, 0,1,2,3, 8,9,10,11);                \
        *(v8d *)((p)+8)  = VSH8(b0_, b1_, 0,1,2,3, 8,9,10,11);                \
        *(v8d *)((p)+16) = VSH8(c0_, c1_, 0,1,2,3, 8,9,10,11);                \
        *(v8d *)((p)+24) = VSH8(d0_, d1_, 0,1,2,3, 8,9,10,11);                \
    } while (0)
    ZQ_SCAT(0, r0); ZQ_SCAT(1, r1); ZQ_SCAT(2, r2); ZQ_SCAT(3, r3);
#undef ZQ_SCAT
}

/* vertical split-complex 64-point FFT, elements at base + n*st doubles:
 * X[k2+8k1] = DFT8_b( W64^{b*k2} * DFT8_a( x[8a+b] ) ), H[64] line buffer.
 * domap fuses the graded map into the stores (c at cb + n*16, x-fastest). */
static inline __attribute__((always_inline))
void vfft64i(double *restrict base, const double *restrict cb, const long st,
             const int domap, const int pfn)
{
    v8d Hr[64], Hi[64];
#pragma GCC unroll 1
    for (int b = 0; b < 8; ++b) {
        v8d xr[8], xi[8], yr[8], yi[8];
#pragma GCC unroll 8
        for (int a = 0; a < 8; ++a) {
            const double *pp = base + (long)(8 * a + b) * st;
            if (pfn) {
                /* next-column T0 on every load (ice L64_radix8's +12% hint,
                 * sc_pass23 SC_LDX form): the +16-double line pair is column
                 * g+1 of this row, consumed one vfft64i call from now --
                 * paced at consumption rate, one column of lead */
                __builtin_prefetch(pp + 16, 0, 3);
                __builtin_prefetch(pp + 24, 0, 3);
            }
            xr[a] = *(const v8d *)pp;
            xi[a] = *(const v8d *)(pp + 8);
        }
        DFT8S(xr, xi, yr, yi);
#pragma GCC unroll 8
        for (int k2 = 0; k2 < 8; ++k2) { Hr[8 * k2 + b] = yr[k2]; Hi[8 * k2 + b] = yi[k2]; }
    }
#pragma GCC unroll 1
    for (int k2 = 0; k2 < 8; ++k2) {
        v8d tr[8], ti[8], zr[8], zi[8];
#pragma GCC unroll 8
        for (int b = 0; b < 8; ++b) { tr[b] = Hr[8 * k2 + b]; ti[b] = Hi[8 * k2 + b]; }
        if (k2) {
#pragma GCC unroll 7
            for (int b = 1; b < 8; ++b)
                CTWS(tr[b], ti[b], tw64c[b * k2], tw64s[b * k2]);
        }
        DFT8S(tr, ti, zr, zi);
#pragma GCC unroll 8
        for (int k1 = 0; k1 < 8; ++k1) {
            const long n = k2 + 8 * k1;
            double *q = base + n * st;
            if (domap) {
                const double *cq = cb + n * 16;
                v8d wr = zr[k1] + *(const v8d *)cq;
                v8d wi = zi[k1] + *(const v8d *)(cq + 8);
                v8d rr, ii;
                MAP8V(wr, wi, rr, ii);
                *(v8d *)q = rr;
                *(v8d *)(q + 8) = ii;
            } else {
                *(v8d *)q = zr[k1];
                *(v8d *)(q + 8) = zi[k1];
            }
        }
    }
}

/* vertical split-complex 16-point FFT:
 * X[k2+8k1] = DFT2_b( W16^{b*k2} * DFT8_a( x[2a+b] ) ), H[16] buffer. */
static inline __attribute__((always_inline))
void vfft16i(double *restrict base, const double *restrict cb, const long st,
             const int domap)
{
    v8d Hr[16], Hi[16];
#pragma GCC unroll 2
    for (int b = 0; b < 2; ++b) {
        v8d xr[8], xi[8], yr[8], yi[8];
#pragma GCC unroll 8
        for (int a = 0; a < 8; ++a) {
            const double *pp = base + (long)(2 * a + b) * st;
            xr[a] = *(const v8d *)pp;
            xi[a] = *(const v8d *)(pp + 8);
        }
        DFT8S(xr, xi, yr, yi);
#pragma GCC unroll 8
        for (int k2 = 0; k2 < 8; ++k2) { Hr[2 * k2 + b] = yr[k2]; Hi[2 * k2 + b] = yi[k2]; }
    }
#pragma GCC unroll 8
    for (int k2 = 0; k2 < 8; ++k2) {
        v8d y0r = Hr[2 * k2], y0i = Hi[2 * k2];
        v8d y1r = Hr[2 * k2 + 1], y1i = Hi[2 * k2 + 1];
        v8d zr[2], zi[2];
#if GP16_FTW
        if (k2 == 0) {
            zr[0] = y0r + y1r; zr[1] = y0r - y1r;
            zi[0] = y0i + y1i; zi[1] = y0i - y1i;
        } else if (k2 == 4) {   /* W16^4 = -i: x1 = (h1i, -h1r), pure adds */
            zr[0] = y0r + y1i; zi[0] = y0i - y1r;
            zr[1] = y0r - y1i; zi[1] = y0i + y1r;
        } else {
            v8d mr, mi;
            v8d tj = VSPL8(fwt16[k2]);
            if (FW16_CFORM(k2)) {
                mr = V8FNMA(tj, y1i, y1r);
                mi = V8FMA (tj, y1r, y1i);
            } else {
                mr = V8FMS (tj, y1r, y1i);
                mi = V8FMA (tj, y1i, y1r);
            }
            const v8d f = VSPL8(fwf16[k2]);
            zr[0] = V8FMA (f, mr, y0r); zi[0] = V8FMA (f, mi, y0i);
            zr[1] = V8FNMA(f, mr, y0r); zi[1] = V8FNMA(f, mi, y0i);
        }
#else
        if (k2) CTWS(y1r, y1i, tw16c[k2], tw16s[k2]);
        zr[0] = y0r + y1r; zr[1] = y0r - y1r;
        zi[0] = y0i + y1i; zi[1] = y0i - y1i;
#endif
#pragma GCC unroll 2
        for (int k1 = 0; k1 < 2; ++k1) {
            const long n = k2 + 8 * k1;
            double *q = base + n * st;
            if (domap) {
                const double *cq = cb + n * 16;
                v8d wr = zr[k1] + *(const v8d *)cq;
                v8d wi = zi[k1] + *(const v8d *)(cq + 8);
                v8d rr, ii;
                MAP8V(wr, wi, rr, ii);
                *(v8d *)q = rr;
                *(v8d *)(q + 8) = ii;
            } else {
                *(v8d *)q = zr[k1];
                *(v8d *)(q + 8) = zi[k1];
            }
        }
    }
}

/* x-pass prefetch for L=64 -- the one regime (4.46 MB/volume, L3-resident)
 * where software prefetch has ever won in this lineage.  0 = none;
 * 1 = slab burst (128 T0/column covering row y+1 across all planes): RACED
 * OUT on the node, 721-753 vs 679 us -- too bursty, issue overhead again;
 * 2 = next-column T0 fused into each pass-1 load, the ice L64_radix8
 * sc_pass23 form -- a WASH on the node (677.5/677.7 vs 677.4/679.2),
 * default 0 by simplest-wins.  That is prefetch loss/wash number seven for
 * this lineage, now including its one previously-winning regime. */
#ifndef GP64_PFS
# define GP64_PFS 0
#endif

/* one L=64 chain step, z/y skewed exactly like the L=32 fft_step */
static inline __attribute__((always_inline))
void fft_step64(double *restrict S, const double *restrict C, const int mode)
{
    for (int x = 0; x <= 64; ++x) {
        double *pz = S + (size_t)x * X64;
        double *py = S + (size_t)(x ? x - 1 : 0) * X64;
        if (mode == 1 && x < 64) {
#pragma GCC unroll 1
            for (int y = 0; y < 64; ++y) {
                double *r = pz + (size_t)y * K64;
#pragma GCC unroll 8
                for (int g = 0; g < 8; ++g) {
                    const double *cr = C + (size_t)y * CYS64 + (size_t)g * CGS64 + (size_t)x * 16;
                    v8d wr = *(const v8d *)(r + 16 * g) + *(const v8d *)cr;
                    v8d wi = *(const v8d *)(r + 16 * g + 8) + *(const v8d *)(cr + 8);
                    v8d rr, ii;
                    MAP8V(wr, wi, rr, ii);
                    *(v8d *)(r + 16 * g) = rr;
                    *(v8d *)(r + 16 * g + 8) = ii;
                }
            }
        }
        PROF_T(t0_);
#pragma GCC unroll 1
        for (int q = 0; q < 8; ++q) {
            if (x < 64)
#pragma GCC unroll 1
                for (int y = 8 * q; y < 8 * q + 8; ++y)
                    zrow64(pz + (size_t)y * K64);
            if (x > 0)
                vfft64i(py + 16 * q, py, K64, 0, 0);
        }
        PROF_T(t1_);
        PROF_ACC(prof_z, t0_, t1_);
    }
    PROF_T(t3_);
#pragma GCC unroll 1
    for (int cc = 0; cc < 512; ++cc) {
        const int y = cc >> 3, g = cc & 7;
#if GP64_PFS == 1
        if (y < 63) {
            const double *sl = S + (size_t)(8 * g) * X64 + (size_t)(y + 1) * K64;
#pragma GCC unroll 8
            for (int xp = 0; xp < 8; ++xp)
#pragma GCC unroll 16
                for (int q = 0; q < 16; ++q)
                    __builtin_prefetch(sl + (size_t)xp * X64 + 8 * q, 0, 3);
        }
#endif
        double *col = S + (size_t)y * K64 + 16 * g;
        const double *ccol = C + (size_t)y * CYS64 + (size_t)g * CGS64;
        vfft64i(col, ccol, X64, mode == 2, GP64_PFS == 2);
    }
    PROF_T(t4_);
    PROF_ACC(prof_x, t3_, t4_);
#if GP2_PROF
    ++prof_steps;
#endif
}

/* one L=16 chain step: z then y per plane (81-KB volume, everything
 * L1/L2-hot), then the x-pass with optional fused map.  From gen_r6 the
 * z/y phases run SKEWED like the L=32 step (GP16_ZYIL): plane x's zquads
 * (96 port-5 shuffles per 64 points) interleave at codelet granularity
 * with plane x-1's y-lines (FMA-heavy) -- bit-identical output, each
 * plane's y still strictly after its own z. */
static inline __attribute__((always_inline))
void fft_step16(double *restrict S, const double *restrict C, const int mode)
{
#if GP16_ZYIL
    for (int x = 0; x <= 16; ++x) {
        double *pz = S + (size_t)x * X16;
        double *py = S + (size_t)(x ? x - 1 : 0) * X16;
        if (mode == 1 && x < 16) {
#pragma GCC unroll 1
            for (int y = 0; y < 16; ++y) {
                double *r = pz + (size_t)y * K16;
#pragma GCC unroll 2
                for (int g = 0; g < 2; ++g) {
                    const double *cr = C + (size_t)y * CYS16 + (size_t)g * CGS16 + (size_t)x * 16;
                    v8d wr = *(const v8d *)(r + 16 * g) + *(const v8d *)cr;
                    v8d wi = *(const v8d *)(r + 16 * g + 8) + *(const v8d *)(cr + 8);
                    v8d rr, ii;
                    MAP8V(wr, wi, rr, ii);
                    *(v8d *)(r + 16 * g) = rr;
                    *(v8d *)(r + 16 * g + 8) = ii;
                }
            }
        }
#pragma GCC unroll 1
        for (int q = 0; q < 2; ++q) {
            if (x < 16) {
#pragma GCC unroll 1
                for (int h = 0; h < 2; ++h) {
                    const int r = 8 * q + 4 * h;
                    zquad16(pz + (size_t)r * K16,       pz + (size_t)(r + 1) * K16,
                            pz + (size_t)(r + 2) * K16, pz + (size_t)(r + 3) * K16);
                }
            }
            if (x > 0)
                vfft16i(py + 16 * q, py, K16, 0);
        }
    }
#else
    for (int x = 0; x < 16; ++x) {
        double *pp = S + (size_t)x * X16;
        if (mode == 1) {
#pragma GCC unroll 1
            for (int y = 0; y < 16; ++y) {
                double *r = pp + (size_t)y * K16;
#pragma GCC unroll 2
                for (int g = 0; g < 2; ++g) {
                    const double *cr = C + (size_t)y * CYS16 + (size_t)g * CGS16 + (size_t)x * 16;
                    v8d wr = *(const v8d *)(r + 16 * g) + *(const v8d *)cr;
                    v8d wi = *(const v8d *)(r + 16 * g + 8) + *(const v8d *)(cr + 8);
                    v8d rr, ii;
                    MAP8V(wr, wi, rr, ii);
                    *(v8d *)(r + 16 * g) = rr;
                    *(v8d *)(r + 16 * g + 8) = ii;
                }
            }
        }
#pragma GCC unroll 1
        for (int q = 0; q < 4; ++q)
            zquad16(pp + (size_t)(4 * q) * K16,     pp + (size_t)(4 * q + 1) * K16,
                    pp + (size_t)(4 * q + 2) * K16, pp + (size_t)(4 * q + 3) * K16);
#pragma GCC unroll 2
        for (int g = 0; g < 2; ++g)
            vfft16i(pp + 16 * g, pp, K16, 0);
    }
#endif /* GP16_ZYIL */
#pragma GCC unroll 1
    for (int cc = 0; cc < 32; ++cc) {
        const int y = cc >> 1, g = cc & 1;
        vfft16i(S + (size_t)y * K16 + 16 * g,
                C + (size_t)y * CYS16 + (size_t)g * CGS16, X16, mode == 2);
    }
}

static SCHED_ATTR void fft_step64_plain(double *restrict S, const double *restrict C)
{ fft_step64(S, C, 0); }
static __attribute__((unused)) SCHED_ATTR void fft_step64_premap(double *restrict S, const double *restrict C)
{ fft_step64(S, C, 1); }
static SCHED_ATTR void fft_step64_xmap(double *restrict S, const double *restrict C)
{ fft_step64(S, C, 2); }
static SCHED_ATTR void fft_step16_plain(double *restrict S, const double *restrict C)
{ fft_step16(S, C, 0); }
static __attribute__((unused)) SCHED_ATTR void fft_step16_premap(double *restrict S, const double *restrict C)
{ fft_step16(S, C, 1); }
static SCHED_ATTR void fft_step16_xmap(double *restrict S, const double *restrict C)
{ fft_step16(S, C, 2); }

/* natural <-> custody conversions, generic over G (chain-end cold paths) */
static void nat_to_cust_g(const double *restrict nat, double *restrict S,
                          const int L, const int G, const size_t ks, const size_t xs)
{
    for (int x = 0; x < L; ++x)
        for (int y = 0; y < L; ++y) {
            const double *r = nat + ((size_t)x * L + y) * (size_t)L * 2;
            double *q = S + (size_t)x * xs + (size_t)y * ks;
            for (int g = 0; g < G; ++g) {
                v8d A = *(const v8d *)(r + 16 * g), B = *(const v8d *)(r + 16 * g + 8);
                *(v8d *)(q + 16 * g) = DEIN_RE(A, B);
                *(v8d *)(q + 16 * g + 8) = DEIN_IM(A, B);
            }
        }
}

static void nat_to_cust_c_g(const double *restrict nat, double *restrict C,
                            const int L, const int G, const size_t cys, const size_t cgs)
{
    for (int x = 0; x < L; ++x)
        for (int y = 0; y < L; ++y) {
            const double *r = nat + ((size_t)x * L + y) * (size_t)L * 2;
            double *q = C + (size_t)y * cys + (size_t)x * 16;
            for (int g = 0; g < G; ++g) {
                v8d A = *(const v8d *)(r + 16 * g), B = *(const v8d *)(r + 16 * g + 8);
                *(v8d *)(q + (size_t)g * cgs) = DEIN_RE(A, B);
                *(v8d *)(q + (size_t)g * cgs + 8) = DEIN_IM(A, B);
            }
        }
}

static void cust_to_nat_g(const double *restrict S, double *restrict nat,
                          const int L, const int G, const size_t ks, const size_t xs)
{
    for (int x = 0; x < L; ++x)
        for (int y = 0; y < L; ++y) {
            const double *q = S + (size_t)x * xs + (size_t)y * ks;
            double *r = nat + ((size_t)x * L + y) * (size_t)L * 2;
            for (int g = 0; g < G; ++g) {
                v8d re = *(const v8d *)(q + 16 * g), im = *(const v8d *)(q + 16 * g + 8);
                *(v8d *)(r + 16 * g) = ILV_LO(re, im);
                *(v8d *)(r + 16 * g + 8) = ILV_HI(re, im);
            }
        }
}

static void cust_map_to_nat_g(const double *restrict S, const double *restrict C,
                              double *restrict nat, const int L, const int G,
                              const size_t ks, const size_t xs,
                              const size_t cys, const size_t cgs)
{
    for (int x = 0; x < L; ++x)
        for (int y = 0; y < L; ++y) {
            const double *q = S + (size_t)x * xs + (size_t)y * ks;
            double *r = nat + ((size_t)x * L + y) * (size_t)L * 2;
            for (int g = 0; g < G; ++g) {
                const double *cq = C + (size_t)y * cys + (size_t)g * cgs + (size_t)x * 16;
                v8d wr = *(const v8d *)(q + 16 * g) + *(const v8d *)cq;
                v8d wi = *(const v8d *)(q + 16 * g + 8) + *(const v8d *)(cq + 8);
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
    if (L == 32 || L == 64 || L == 16) {
        size_t sd, cd;
        if (L == 32) { fill_fast_tables(); sd = VOLD;   cd = GP2_CT ? CVOLD : VOLD; }
        else if (L == 64) { fill_tables_64(); sd = VOLD64; cd = CVOLD64; }
        else { fill_tables_16(); sd = VOLD16; cd = CVOLD16; }
        /* one block: custody state + custody c + prefetch-overrun guard.
         * GP64_HP=1 (control, RACED OUT): 2-MB-aligned + MADV_HUGEPAGE +
         * touch for the L=64 block (the ice_r7 hugepage move).  On this node
         * it LOSES ~1.7% (688-691 vs 677.7 x3 us, THP confirmed [madvise]):
         * the phase is L3-bandwidth-bound, not TLB-bound, and 2-MB frames
         * flatten the 4-KB page-color scatter the odd-line padding relies on. */
#ifndef GP64_HP
# define GP64_HP 0
#endif
        size_t bytes = (sd + cd + 1024) * sizeof(double);
        size_t align = (GP64_HP && L == 64) ? (size_t)2 << 20 : 64;
        if (posix_memalign(&p->blk, align, bytes) != 0) { free(p); return NULL; }
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        if (GP64_HP && L == 64) {
            madvise(p->blk, bytes, MADV_HUGEPAGE);
            memset(p->blk, 0, bytes);
        }
#endif
        p->S = (double *)p->blk;
        p->C = p->S + sd;
        p->fast = L;
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
    if (p->fast == 32) {
        const size_t vd = (size_t)32 * 32 * 32 * 2;
        for (int b = 0; b < p->batch; ++b) {
            nat_to_cust((const double *)in + (size_t)b * vd, p->S);
            fft_step_plain(p->S, p->S);
            cust_to_nat(p->S, (double *)out + (size_t)b * vd);
        }
        return;
    }
    if (p->fast == 64) {
        const size_t vd = (size_t)64 * 64 * 64 * 2;
        for (int b = 0; b < p->batch; ++b) {
            nat_to_cust_g((const double *)in + (size_t)b * vd, p->S, 64, 8, K64, X64);
            fft_step64_plain(p->S, p->S);
            cust_to_nat_g(p->S, (double *)out + (size_t)b * vd, 64, 8, K64, X64);
        }
        return;
    }
    if (p->fast == 16) {
        const size_t vd = (size_t)16 * 16 * 16 * 2;
        for (int b = 0; b < p->batch; ++b) {
            nat_to_cust_g((const double *)in + (size_t)b * vd, p->S, 16, 2, K16, X16);
            fft_step16_plain(p->S, p->S);
            cust_to_nat_g(p->S, (double *)out + (size_t)b * vd, 16, 2, K16, X16);
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
    if (p->fast == 32) {
        const size_t vd = V * 2;
        for (int b = 0; b < p->batch; ++b) {
            nat_to_cust((const double *)x0 + (size_t)b * vd, p->S);
#if GP2_CT
            nat_to_cust_c((const double *)c + (size_t)b * vd, p->C);
#else
            nat_to_cust((const double *)c + (size_t)b * vd, p->C);
#endif
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
    if (p->fast == 64 || p->fast == 16) {
        const size_t vd = V * 2;
        const int L = p->fast, G = L / 8;
        const size_t ks = (L == 64) ? K64 : K16, xs = (L == 64) ? X64 : X16;
        const size_t cys = (L == 64) ? CYS64 : CYS16, cgs = (L == 64) ? CGS64 : CGS16;
        for (int b = 0; b < p->batch; ++b) {
            nat_to_cust_g((const double *)x0 + (size_t)b * vd, p->S, L, G, ks, xs);
            nat_to_cust_c_g((const double *)c + (size_t)b * vd, p->C, L, G, cys, cgs);
#if GP2_PREMAP
            if (L == 64) {
                fft_step64_plain(p->S, p->C);
                for (int s = 1; s < m; ++s) fft_step64_premap(p->S, p->C);
            } else {
                fft_step16_plain(p->S, p->C);
                for (int s = 1; s < m; ++s) fft_step16_premap(p->S, p->C);
            }
#else
            if (L == 64) {
                for (int s = 1; s < m; ++s) fft_step64_xmap(p->S, p->C);
                fft_step64_plain(p->S, p->C);
            } else {
                for (int s = 1; s < m; ++s) fft_step16_xmap(p->S, p->C);
                fft_step16_plain(p->S, p->C);
            }
#endif
            cust_map_to_nat_g(p->S, p->C, (double *)final_out + (size_t)b * vd,
                              L, G, ks, xs, cys, cgs);
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
