/* gen_pow2: the 2^k-axes class entry.  L=32 is the scored size (B=8, m=250
 * graded map chain); from gen_r4 the custody engine is GENERALIZED over
 * G = L/8: L=16 (G=2, 4 rows/TR8) and L=64 (G=8, the ice L64_blocked z-line,
 * L3 regime) run the same architecture; from gen_r8 so does L=128 (G=16,
 * DFT16-over-slots z-row, DRAM regime); 2/4/8 stay on the generic
 * scalar radix-2 path.  From gen_r6
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
#include <stdint.h>
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
    return "2^k axes: custody split-complex chain engine over G=L/8 (L=16/32/64/128:"
           " TR8 z-codelets, x-fastest c, lazy exact map, DSB-resident bodies,"
           " dual-select FMA-folded twiddles; r8 adds the G=16 DRAM-regime engine;"
           " r9 re-forms z-codelet output slots via 256-bit extract-to-memory stores;"
           " r11 fuses the L=128 chain step into ONE tile-resident sweep"
           " (x-stage-2+map+z+y+x-stage-1, tile-order c): -61% DRAM reads, -14% wall;"
           " r12 fuses the next step's z-rows into stage-2's L1-hot row completion"
           " (-8% wall at 128) and flips the fused sweep ON at 64 (-11%);"
           " r13 fuses the custody conversions into the EXECUTE path's z-loads"
           " and x-stores (GP2_XFE: the benchFFT B=1 single-call lever, two"
           " custody-volume round trips deleted per call);"
           " r14 reroutes execute() at 64/128 through the r11/r12 TILED"
           " one-sweep structure (GP64_XT/GP128_XT: z-nat+y+x-stage-1 per"
           " group, x-stage-2 emits straight to natural per 16-plane tile --"
           " the in-place x-pass write and the cust_to_nat read both deleted),"
           " NT zmm emit for 64-B-aligned callers (GP128_NTE), and"
           " alignment-safe LDU/STU on ALL caller buffers (benchFFT's"
           " malloc+16 buffers segfaulted every r13 fused path)),"
           " other 2^k in 2..8 generic radix-2";
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
/* gen_r14: explicitly UNALIGNED vector accesses for CALLER buffers.  A plain
 * *(v8d *) deref licenses gcc to assume 64-B alignment; our own driver
 * 64-aligns everything so it never showed, but benchFFT's bench_malloc
 * returns malloc+16 (16-B aligned) and the r13 execute-fusion paths
 * SEGFAULTED under it (found by an alignment-offset harness this round).
 * Natural-side loads/stores must go through these; custody S/C stays *(v8d*)
 * (we allocate it, 64-B aligned by construction). */
typedef double    v8du __attribute__((vector_size(64), aligned(8)));
#define LDU(p)      ((v8d)*(const v8du *)(p))
#define STU(p, v)   (*(v8du *)(p) = (v8du)(v))
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
/* GP2_PFX1 (gen_r10): T0-prefetch the NEXT column's 32 state slots (64
 * lines), spread one slot per pass-1 load iteration of the CURRENT column.
 * Motivation is the r8 OSACA/measured attribution of the x-pass residual:
 * ~211 cyc/column is the column's ~12 KB of L2 state traffic running
 * UNOVERLAPPED, because the ~350-entry ROB cannot span a 1.3K-uop column
 * body to reach column n+1's pass-1 loads -- and the XS = 18.5 KB stride
 * defeats the L2 streamer, while the DCU IP-stride prefetcher sees the
 * per-PC stride change every 4th column (g wraps to the next y-row).
 * Software prefetch is the ROB-independent form of exactly that overlap;
 * unlike the seven raced-out prefetch attacks in this record (all of which
 * targeted latency the OOO core already hid), this one targets transfer
 * time the r8 audit measured as NOT hidden.  64 T0s per ~1300-uop column
 * (+5% issue, spread across pass 1's half-idle load ports).
 * RACED OUT in gen_r10 on the node (a81n2, quiet windows, sd<=0.2%):
 * +0.3-0.6% in 3/3 clean B=8 pairs and +0.8% at B=1 -- prefetch loss #8,
 * and the definitive one: even full next-column coverage of the
 * model-attributed unhidden transfer LOSES, so the DCU IP-stride prefetcher
 * evidently already covers the 3-of-4 within-row column strides and the
 * residual is store-eviction/fill contention no software prefetch can cut.
 * The software-prefetch book on this engine is closed.  Default 0
 * (bit-identical output; =0 builds byte-identical to the r9 ship). */
#ifndef GP2_PFX1
# define GP2_PFX1 0
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
/* GP2_ZST (gen_r9): zpair slot RE-FORM VIA 256-BIT EXTRACT-TO-MEMORY STORES.
 * The r8 OSACA audit put the fused z/y q-group floor at 539 cyc with port 5
 * carrying ALL 256 shuffles (4 zpairs x 64) plus its FMA share; the r9 PMU
 * brief's avenue 4 says port pressure off p0/p5 is the remaining currency.
 * The re-form's 16 vshuff64x2 + 16 512-bit stores per pair only ever move
 * ALIGNED 256-bit halves (rA gets the low halves of an O pair, rB the high
 * halves), so the same bytes can be written as 32 x 256-bit stores instead:
 * vmovupd ymm for the low halves, vextractf64x4-to-memory for the high --
 * both are pure store-port ops on ICL/SPR (no p5 uop), and at 2 x 256-bit
 * stores/cyc the store bandwidth cost is NEUTRAL vs 16 x 512-bit.  Front end
 * neutral too (32 fused store uops replace 16 shuffle + 16 store uops).
 * Net: -16 p5 uops per zpair, q-group port floor 539 -> 507 (-6%).
 * Bit-identical output (same doubles to the same addresses).  The following
 * y-pass reads these rows one full plane later (GP2_ZYIL), so the split
 * stores are long committed to L1 -- no store-forward-fail exposure. */
#ifndef GP2_ZST
# define GP2_ZST 1
#endif
/* same trick for the L=16 zquad granule scatter (ZQ_SCAT: 12 shuffles + 4
 * 512-bit stores -> 16 x 128-bit stores; vextractf64x2-to-memory is also
 * store-port-only).  The L=16 z-phase is outright port-5-bound (96 shuffles
 * vs ~92 FMA per quad, the r6 finding), so it has the most to gain; store
 * bandwidth cost is +4 cyc/quad at 2 x 128-bit stores/cyc, well under the
 * 48 p5 uops deleted. */
#ifndef GP16_ZST
# define GP16_ZST 1
#endif
/* GP2_XFE (gen_r13): EXECUTE-PATH CONVERSION FUSION -- the r8-designed lever
 * ("if anyone ever scores 2^k singles"), built now that benchFFT's community
 * curve times exactly this: repeated B=1 fft3d_execute() calls, where the
 * two custody conversion sweeps are NOT amortized by a chain (L=128 single
 * call pays ~35% conversions and sat at 1.03x vs fftw3_measure).  The z-pass
 * is the first consumer of the input and the x-pass pass-2 store is the last
 * producer of the output, so:
 *   - load side: z-codelets read the NATURAL volume directly, deinterleaving
 *     with the same 2 DEIN shuffles/slot the conversion paid anyway (lanes =
 *     8 adjacent z = one natural row segment; no transpose exists in the
 *     conversion, so fusion is pure deletion of a store+load round trip);
 *   - store side: the vertical x-FFT's pass-2 stores ILV-interleave straight
 *     to the natural output (same 2 shuffles/slot cust_to_nat paid), never
 *     writing the custody volume at all.
 * Per single call this deletes TWO full custody-volume writes + reads
 * (~2.3 MB at 32, ~18 MB at 64, ~138 MB at 128 -- the DRAM-regime case is
 * where the community curve nearly loses).  CHAIN PATHS ARE UNTOUCHED: the
 * fused-execute step functions are separate instantiations; the scored
 * chain arithmetic and codegen do not change (=0 restores the r12 execute
 * for the race/cross-arch control). */
#ifndef GP2_XFE
# define GP2_XFE 1
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

/* GP2_CPS (gen_r7): CONSTANT-PER-SITE TWIDDLE ROUTING (Garrido, literature 11
 * Tier 1 -- the rounds-7/8 backlog item assigned to a mid-size cell; 32 is
 * this entry's).  The site->constant binding has been compile-time since r5's
 * FTW (every unrolled site indexes the tables with a constant j), but the
 * table VALUES were runtime-filled statics: gcc cannot prove the x-pass's
 * heap stores (S comes from posix_memalign, which it does not track as an
 * allocator) never alias a filled static double array, so every k2-group's
 * broadcast loads are re-issued per column and nothing can be hoisted or
 * rematerialized freely.  =1 compiles the L=32 hot-path twiddles in as
 * LITERAL static const arrays (generated on the node by the exact
 * fill_fast_tables expressions, %a-exact, verified against the runtime fill
 * at create time), making every twiddle a compile-time .rodata constant the
 * compiler may CSE, hoist, or re-materialize with no aliasing hazard --
 * Garrido's "twiddles become compiled-in broadcast constants", which is as
 * far as the idea can go on x86 (no 64-bit vector immediates exist; a
 * broadcast load from .rodata IS the constant form).
 * RACED OUT in gen_r7, both arms, despite the asm audit reading -107
 * insns/-75 loads in the fused step body with identical FMA count:
 * =1 (all tables) +0.7-1.6% in 5/5 clean rotated pairs; =2 (pass-2 scalars
 * only, z vector tables left runtime) +1.1-2.7% in 6/6.  Mechanism: the
 * runtime tables' per-iteration broadcast re-loads were L1-hot issues on
 * half-idle load ports (free, per r5's non-port-bound verdict), and they
 * kept every constant's live range ONE k2-group long -- compiler-enforced
 * rematerialization.  Folding the constants lets gcc extend those live
 * ranges across the column body, and the register pressure costs more than
 * the loads ever did.  Constant routing pays where twiddles are per-
 * butterfly TRAFFIC; this engine made them site-constant L1 broadcasts in
 * r5.  Default 0 (bit-identical, codegen-identical to the r6 ship); both
 * arms kept compilable as cross-arch race axes. */
#ifndef GP2_CPS
# define GP2_CPS 0
#endif
#if GP2_CPS
static const double cps_fwt32[32] = {
    -0x0p+0,                  -0x1.975f5e0553158p-3,
    -0x1.a827999fcef32p-2,    -0x1.561b82ab7f99p-1,
    -0x1p+0,                  -0x1.561b82ab7f99p-1,
    -0x1.a827999fcef32p-2,    -0x1.975f5e0553158p-3,
    0x1.d9cceba3f91f2p-66,    0x1.975f5e0553158p-3,
    0x1.a827999fcef32p-2,     0x1.561b82ab7f99p-1,
    0x1p+0,                   0x1.561b82ab7f99p-1,
    0x1.a827999fcef32p-2,     0x1.975f5e0553158p-3,
    -0x1.d9cceba3f91f2p-65,   -0x1.975f5e0553158p-3,
    -0x1.a827999fcef32p-2,    -0x1.561b82ab7f99p-1,
    -0x1p+0,                  -0x1.561b82ab7f99p-1,
    -0x1.a827999fcef32p-2,    -0x1.975f5e0553158p-3,
    0x1.b1acd85d7d6bbp-63,    0x1.975f5e0553158p-3,
    0x1.a827999fcef32p-2,     0x1.561b82ab7f99p-1,
    0x1p+0,                   0x1.561b82ab7f99p-1,
    0x1.a827999fcef32p-2,     0x1.975f5e0553158p-3,
};
static const double cps_fwf32[32] = {
    0x1p+0,                   0x1.f6297cff75cbp-1,
    0x1.d906bcf328d46p-1,     0x1.a9b66290ea1a3p-1,
    0x1.6a09e667f3bcdp-1,     -0x1.a9b66290ea1a3p-1,
    -0x1.d906bcf328d46p-1,    -0x1.f6297cff75cbp-1,
    -0x1p+0,                  -0x1.f6297cff75cbp-1,
    -0x1.d906bcf328d46p-1,    -0x1.a9b66290ea1a3p-1,
    -0x1.6a09e667f3bcdp-1,    -0x1.a9b66290ea1a3p-1,
    -0x1.d906bcf328d46p-1,    -0x1.f6297cff75cbp-1,
    -0x1p+0,                  -0x1.f6297cff75cbp-1,
    -0x1.d906bcf328d46p-1,    -0x1.a9b66290ea1a3p-1,
    -0x1.6a09e667f3bcdp-1,    0x1.a9b66290ea1a3p-1,
    0x1.d906bcf328d46p-1,     0x1.f6297cff75cbp-1,
    0x1p+0,                   0x1.f6297cff75cbp-1,
    0x1.d906bcf328d46p-1,     0x1.a9b66290ea1a3p-1,
    0x1.6a09e667f3bcdp-1,     0x1.a9b66290ea1a3p-1,
    0x1.d906bcf328d46p-1,     0x1.f6297cff75cbp-1,
};
static const double cps_wz32r[4][8] __attribute__((aligned(64))) = {
    { 0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0 },
    { 0x1p+0, 0x1.f6297cff75cbp-1, 0x1.d906bcf328d46p-1, 0x1.a9b66290ea1a3p-1,
      0x1.6a09e667f3bcdp-1, 0x1.1c73b39ae68c8p-1, 0x1.87de2a6aea963p-2, 0x1.8f8b83c69a60bp-3 },
    { 0x1p+0, 0x1.d906bcf328d46p-1, 0x1.6a09e667f3bcdp-1, 0x1.87de2a6aea963p-2,
      -0x1.d9cceba3f91f2p-66, -0x1.87de2a6aea963p-2, -0x1.6a09e667f3bcdp-1, -0x1.d906bcf328d46p-1 },
    { 0x1p+0, 0x1.a9b66290ea1a3p-1, 0x1.87de2a6aea963p-2, -0x1.8f8b83c69a60bp-3,
      -0x1.6a09e667f3bcdp-1, -0x1.f6297cff75cbp-1, -0x1.d906bcf328d46p-1, -0x1.1c73b39ae68c8p-1 },
};
static const double cps_wz32i[4][8] __attribute__((aligned(64))) = {
    { -0x0p+0, -0x0p+0, -0x0p+0, -0x0p+0, -0x0p+0, -0x0p+0, -0x0p+0, -0x0p+0 },
    { -0x0p+0, -0x1.8f8b83c69a60bp-3, -0x1.87de2a6aea963p-2, -0x1.1c73b39ae68c8p-1,
      -0x1.6a09e667f3bcdp-1, -0x1.a9b66290ea1a3p-1, -0x1.d906bcf328d46p-1, -0x1.f6297cff75cbp-1 },
    { -0x0p+0, -0x1.87de2a6aea963p-2, -0x1.6a09e667f3bcdp-1, -0x1.d906bcf328d46p-1,
      -0x1p+0, -0x1.d906bcf328d46p-1, -0x1.6a09e667f3bcdp-1, -0x1.87de2a6aea963p-2 },
    { -0x0p+0, -0x1.1c73b39ae68c8p-1, -0x1.d906bcf328d46p-1, -0x1.f6297cff75cbp-1,
      -0x1.6a09e667f3bcdp-1, -0x1.8f8b83c69a60bp-3, 0x1.87de2a6aea963p-2, 0x1.a9b66290ea1a3p-1 },
};
# define FWT32 cps_fwt32
# define FWF32 cps_fwf32
/* GP2_CPS=2: scalars only -- the zpair vector tables stay runtime-filled, so
 * gcc cannot hoist those 6 zmm loads into live-across-the-z/y-loop registers
 * (the scoping arm for WHERE the CPS=1 loss comes from). */
# if GP2_CPS == 2
#  define WZ32R wz32r
#  define WZ32I wz32i
# else
#  define WZ32R cps_wz32r
#  define WZ32I cps_wz32i
# endif
#else
# define FWT32 fwt32
# define FWF32 fwf32
# define WZ32R wz32r
# define WZ32I wz32i
#endif

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
#if GP2_CPS
    /* the compiled-in tables were generated by these exact expressions on
     * the node; a mismatch means a different libm rounding -- the constants
     * are still <=1-ulp twiddles and every gate holds with 20x margin, but
     * bit-identity to the GP2_CPS=0 build would be lost, so flag it. */
    if (memcmp(cps_fwt32, fwt32, sizeof fwt32) || memcmp(cps_fwf32, fwf32, sizeof fwf32) ||
        memcmp(cps_wz32r, wz32r, sizeof wz32r) || memcmp(cps_wz32i, wz32i, sizeof wz32i))
        fprintf(stderr, "gen_pow2: GP2_CPS constants differ from runtime libm fill\n");
#endif
}

/* m = the ratio-form factor of W32^j * h (x = fwf32[j] * m); j a
 * compile-time constant in every unrolled caller, so the form branch and
 * the j==8 special in twdft4 fold away. */
#define FWM(MR, MI, HR, HI, J) do {                                          \
    v8d tj_ = VSPL8(FWT32[J]);                                               \
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
    const v8d f1 = VSPL8(FWF32[j1]), f3 = VSPL8(FWF32[j3]);
    v8d t0r, t0i, t1r, t1i;
    if (j2 == 8) {           /* W32^8 = -i: x2 = (h2i, -h2r), pure adds */
        t0r = tr[0] + ti[2]; t0i = ti[0] - tr[2];
        t1r = tr[0] - ti[2]; t1i = ti[0] + tr[2];
    } else {
        v8d m2r, m2i;
        FWM(m2r, m2i, tr[2], ti[2], j2);
        const v8d f2 = VSPL8(FWF32[j2]);
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
        v8d tr = *(const v8d *)WZ32R[k2], ti = *(const v8d *)WZ32I[k2];
        CTWV(Mr[k2], Mi[k2], tr, ti);
        CTWV(Mr[4 + k2], Mi[4 + k2], tr, ti);
    }
    v8d Tr[8], Ti[8], Or[8], Oi[8];
    TR8(Mr, Tr);
    TR8(Mi, Ti);
    DFT8S(Tr, Ti, Or, Oi);
#if GP2_ZST
#pragma GCC unroll 4
    for (int m = 0; m < 4; ++m) {
        _mm256_storeu_pd(rA + 16 * m,      _mm512_castpd512_pd256((__m512d)Or[2 * m]));
        _mm256_storeu_pd(rA + 16 * m + 4,  _mm512_castpd512_pd256((__m512d)Or[2 * m + 1]));
        _mm256_storeu_pd(rA + 16 * m + 8,  _mm512_castpd512_pd256((__m512d)Oi[2 * m]));
        _mm256_storeu_pd(rA + 16 * m + 12, _mm512_castpd512_pd256((__m512d)Oi[2 * m + 1]));
        _mm256_storeu_pd(rB + 16 * m,      _mm512_extractf64x4_pd((__m512d)Or[2 * m], 1));
        _mm256_storeu_pd(rB + 16 * m + 4,  _mm512_extractf64x4_pd((__m512d)Or[2 * m + 1], 1));
        _mm256_storeu_pd(rB + 16 * m + 8,  _mm512_extractf64x4_pd((__m512d)Oi[2 * m], 1));
        _mm256_storeu_pd(rB + 16 * m + 12, _mm512_extractf64x4_pd((__m512d)Oi[2 * m + 1], 1));
    }
#else
#pragma GCC unroll 4
    for (int m = 0; m < 4; ++m) {
        *(v8d *)(rA + 16 * m)     = VSH8(Or[2 * m], Or[2 * m + 1], 0,1,2,3,8,9,10,11);
        *(v8d *)(rA + 16 * m + 8) = VSH8(Oi[2 * m], Oi[2 * m + 1], 0,1,2,3,8,9,10,11);
        *(v8d *)(rB + 16 * m)     = VSH8(Or[2 * m], Or[2 * m + 1], 4,5,6,7,12,13,14,15);
        *(v8d *)(rB + 16 * m + 8) = VSH8(Oi[2 * m], Oi[2 * m + 1], 4,5,6,7,12,13,14,15);
    }
#endif
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
             const long cst, const double *restrict nxt)
{
    v8d Hr[32], Hi[32];
#pragma GCC unroll 4
    for (int b = 0; b < 4; ++b) {
        v8d xr[8], xi[8], yr[8], yi[8];
#pragma GCC unroll 8
        for (int a = 0; a < 8; ++a) {
            const double *pp = base + (long)(4 * a + b) * st;
#if GP2_PFX1
            /* next column's slot at the same (a,b) index: consumed by the
             * NEXT column's pass 1, ~700 cyc from now; its L2->L1 transfer
             * overlaps this whole column instead of serializing at the top
             * of the next one (nxt == base at cc = 127: harmless L1 hits) */
            __builtin_prefetch(nxt + (long)(4 * a + b) * st, 0, 3);
            __builtin_prefetch(nxt + (long)(4 * a + b) * st + 8, 0, 3);
#endif
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
        for (int cc = 0; cc < 128; ++cc) {
            const int nc = cc < 127 ? cc + 1 : cc;
            vfft32m(S + (size_t)(cc >> 2) * KS + 16 * (cc & 3),
                    CCOL(C, cc), XS, CST,
                    S + (size_t)(nc >> 2) * KS + 16 * (nc & 3));
        }
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
                vfft32m(S + (size_t)y * KS + 16 * g, CCOL(C, 4 * y + g), XS, CST,
                        S + (size_t)y * KS + 16 * g);
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
                v8d A = LDU(r + 16 * g), B = LDU(r + 16 * g + 8);
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
                v8d A = LDU(r + 16 * g), B = LDU(r + 16 * g + 8);
                *(v8d *)(q + (size_t)g * CGS) = DEIN_RE(A, B);
                *(v8d *)(q + (size_t)g * CGS + 8) = DEIN_IM(A, B);
            }
        }
}

/* custody -> natural interleaved (execute path, no map; unused under
 * GP2_XFE=1 -- the raced-control unfused execute) */
static __attribute__((unused)) void cust_to_nat(const double *restrict S, double *restrict nat)
{
    for (int x = 0; x < 32; ++x)
        for (int y = 0; y < 32; ++y) {
            const double *q = S + (size_t)x * XS + (size_t)y * KS;
            double *r = nat + ((size_t)x * 32 + y) * 64;
#pragma GCC unroll 4
            for (int g = 0; g < 4; ++g) {
                v8d re = *(const v8d *)(q + 16 * g), im = *(const v8d *)(q + 16 * g + 8);
                STU(r + 16 * g, ILV_LO(re, im));
                STU(r + 16 * g + 8, ILV_HI(re, im));
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
                STU(r + 16 * g, ILV_LO(rr, ii));
                STU(r + 16 * g + 8, ILV_HI(rr, ii));
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

/* gen_r8: L=128 (G=16) -- the last unbuilt custody engine (the class's
 * 63-ms generic-path hole).  Volume 34.6 MB, S+C ~68 MB: the DRAM regime
 * (a80n0 LLC is 24 MB) -- per-step traffic ~170 MB bounds everything, so
 * this engine's job is the traffic-minimal two-sweep structure, not
 * scheduling.  Strides keep the odd-line rule. */
#define K128 264                           /* 33 lines, odd                  */
#define X128 (128 * K128 + 8)              /* 33800 doubles = 4225 lines, odd */
#define VOLD128 ((size_t)128 * X128)
#define CGS128 (128 * 16 + 8)              /* 2056 doubles = 257 lines, odd  */
#define CYS128 ((size_t)16 * CGS128)
#define CVOLD128 ((size_t)128 * CYS128)

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

/* L=128 tables: lane twiddles W128^{l*k2} (16 vector pairs) and the scalar
 * W128^j table for the vertical pass; the DFT16 codelet's W16^j combines
 * reuse tw16c/tw16s (refilled here so a 128-only plan has them). */
static double wz128r[16][8] __attribute__((aligned(64)));
static double wz128i[16][8] __attribute__((aligned(64)));
static double tw128c[128], tw128s[128];

static void fill_tables_128(void)
{
    const long double PI2 = 6.283185307179586476925286766559005768L;
    for (int j = 0; j < 128; ++j) {
        long double a = -PI2 * (long double)j / 128.0L;
        tw128c[j] = (double)cosl(a);
        tw128s[j] = (double)sinl(a);
    }
    for (int k2 = 0; k2 < 16; ++k2)
        for (int l = 0; l < 8; ++l) {
            long double a = -PI2 * (long double)((l * k2) % 128) / 128.0L;
            wz128r[k2][l] = (double)cosl(a);
            wz128i[k2][l] = (double)sinl(a);
        }
    for (int j = 0; j < 16; ++j) {
        long double a = -PI2 * (long double)j / 16.0L;
        tw16c[j] = (double)cosl(a);
        tw16s[j] = (double)sinl(a);
    }
}

/* forward split-complex DFT16 on arrays of 16 vector pairs, natural order,
 * DIT 16 = 2x8: DFT8 over evens/odds, W16^j twiddle on the odd branch
 * (j = 4 is W16^4 = -i: exact swap, no multiply), combine.  Inputs must not
 * alias outputs (unlike DFT8S) -- callers use distinct arrays. */
static inline __attribute__((always_inline))
void dft16s(const v8d *restrict xr, const v8d *restrict xi,
            v8d *restrict yr, v8d *restrict yi)
{
    v8d er[8], ei[8], or_[8], oi[8];
#pragma GCC unroll 8
    for (int t = 0; t < 8; ++t) {
        er[t] = xr[2 * t];      ei[t] = xi[2 * t];
        or_[t] = xr[2 * t + 1]; oi[t] = xi[2 * t + 1];
    }
    DFT8S(er, ei, er, ei);
    DFT8S(or_, oi, or_, oi);
#pragma GCC unroll 7
    for (int j = 1; j < 8; ++j) {
        if (j == 4) {           /* (or,oi) * -i = (oi, -or), pure moves */
            v8d t4 = or_[4];
            or_[4] = oi[4];
            oi[4] = -t4;
        } else {
            CTWS(or_[j], oi[j], tw16c[j], tw16s[j]);
        }
    }
#pragma GCC unroll 8
    for (int j = 0; j < 8; ++j) {
        yr[j] = er[j] + or_[j];     yi[j] = ei[j] + oi[j];
        yr[j + 8] = er[j] - or_[j]; yi[j + 8] = ei[j] - oi[j];
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
#if GP16_ZST
    /* granule (2r, 2r+1) of each O vector is one aligned 128-bit lane:
     * write the row as 16 xmm stores (store-port only; vextractf64x2-to-
     * memory has no p5 uop) instead of 12 shuffles + 4 zmm stores.
     * Bit-identical bytes at the same addresses. */
#define ZQ_STG(P, V, R) do {                                                  \
        if ((R) == 0) _mm_storeu_pd((P), _mm512_castpd512_pd128((__m512d)(V)));\
        else _mm_storeu_pd((P), _mm512_extractf64x2_pd((__m512d)(V), (R)));   \
    } while (0)
#define ZQ_SCAT(r, p) do {                                                    \
        ZQ_STG((p),      Or[0], (r)); ZQ_STG((p) + 2,  Or[1], (r));           \
        ZQ_STG((p) + 4,  Or[2], (r)); ZQ_STG((p) + 6,  Or[3], (r));           \
        ZQ_STG((p) + 8,  Oi[0], (r)); ZQ_STG((p) + 10, Oi[1], (r));           \
        ZQ_STG((p) + 12, Oi[2], (r)); ZQ_STG((p) + 14, Oi[3], (r));           \
        ZQ_STG((p) + 16, Or[4], (r)); ZQ_STG((p) + 18, Or[5], (r));           \
        ZQ_STG((p) + 20, Or[6], (r)); ZQ_STG((p) + 22, Or[7], (r));           \
        ZQ_STG((p) + 24, Oi[4], (r)); ZQ_STG((p) + 26, Oi[5], (r));           \
        ZQ_STG((p) + 28, Oi[6], (r)); ZQ_STG((p) + 30, Oi[7], (r));           \
    } while (0)
    ZQ_SCAT(0, r0); ZQ_SCAT(1, r1); ZQ_SCAT(2, r2); ZQ_SCAT(3, r3);
#undef ZQ_SCAT
#undef ZQ_STG
#else
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
#endif
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

/* one L=128 custody z-row, in place: DFT16 over the 16 slots (k2 = k mod
 * 16), lane twiddle W128^{l*k2}, then per k2-half one TR8 pair + DFT8 over
 * the former lanes (k1 = k div 16); X[k2 + 16*k1] has slot k/8 = 2*k1 +
 * (k2 >= 8) and lane k2 mod 8, so each DFT8 output stores DIRECTLY to every
 * second slot -- the "two TR8s per line-half" shape promised in r1. */
static inline __attribute__((always_inline))
void zrow128(double *restrict r)
{
    v8d Mr[16], Mi[16], Br[16], Bi[16];
#pragma GCC unroll 16
    for (int g = 0; g < 16; ++g) {
        Mr[g] = *(const v8d *)(r + 16 * g);
        Mi[g] = *(const v8d *)(r + 16 * g + 8);
    }
    dft16s(Mr, Mi, Br, Bi);
#pragma GCC unroll 15
    for (int k2 = 1; k2 < 16; ++k2)
        CTWV(Br[k2], Bi[k2], *(const v8d *)wz128r[k2], *(const v8d *)wz128i[k2]);
    v8d Tr[8], Ti[8], Or[8], Oi[8];
    TR8(Br, Tr);
    TR8(Bi, Ti);
    DFT8S(Tr, Ti, Or, Oi);
#pragma GCC unroll 8
    for (int k1 = 0; k1 < 8; ++k1) {
        *(v8d *)(r + 16 * (2 * k1)) = Or[k1];
        *(v8d *)(r + 16 * (2 * k1) + 8) = Oi[k1];
    }
    TR8(Br + 8, Tr);
    TR8(Bi + 8, Ti);
    DFT8S(Tr, Ti, Or, Oi);
#pragma GCC unroll 8
    for (int k1 = 0; k1 < 8; ++k1) {
        *(v8d *)(r + 16 * (2 * k1 + 1)) = Or[k1];
        *(v8d *)(r + 16 * (2 * k1 + 1) + 8) = Oi[k1];
    }
}

/* vertical split-complex 128-point FFT, elements at base + n*st doubles:
 * X[k1 + 8*m] = DFT16_b( W128^{b*k1} * DFT8_a( x[16a+b] ) ), H[128] line
 * buffer (16 KB).  Pass loops ROLLED (r3 DSB rule: one b-/k1-group per
 * iteration).  domap fuses the graded map into the stores (c x-fastest). */
static inline __attribute__((always_inline))
void vfft128i(double *restrict base, const double *restrict cb, const long st,
              const int domap)
{
    v8d Hr[128], Hi[128];
#pragma GCC unroll 1
    for (int b = 0; b < 16; ++b) {
        v8d xr[8], xi[8], yr[8], yi[8];
#pragma GCC unroll 8
        for (int a = 0; a < 8; ++a) {
            const double *pp = base + (long)(16 * a + b) * st;
            xr[a] = *(const v8d *)pp;
            xi[a] = *(const v8d *)(pp + 8);
        }
        DFT8S(xr, xi, yr, yi);
#pragma GCC unroll 8
        for (int k1 = 0; k1 < 8; ++k1) { Hr[16 * k1 + b] = yr[k1]; Hi[16 * k1 + b] = yi[k1]; }
    }
#pragma GCC unroll 1
    for (int k1 = 0; k1 < 8; ++k1) {
        v8d tr[16], ti[16], zr[16], zi[16];
#pragma GCC unroll 16
        for (int b = 0; b < 16; ++b) { tr[b] = Hr[16 * k1 + b]; ti[b] = Hi[16 * k1 + b]; }
        if (k1) {
#pragma GCC unroll 15
            for (int b = 1; b < 16; ++b)
                CTWS(tr[b], ti[b], tw128c[b * k1], tw128s[b * k1]);
        }
        dft16s(tr, ti, zr, zi);
#pragma GCC unroll 16
        for (int m = 0; m < 16; ++m) {
            const long n = k1 + 8 * m;
            double *q = base + n * st;
            if (domap) {
                const double *cq = cb + n * 16;
                v8d wr = zr[m] + *(const v8d *)cq;
                v8d wi = zi[m] + *(const v8d *)(cq + 8);
                v8d rr, ii;
                MAP8V(wr, wi, rr, ii);
                *(v8d *)q = rr;
                *(v8d *)(q + 8) = ii;
            } else {
                *(v8d *)q = zr[m];
                *(v8d *)(q + 8) = zi[m];
            }
        }
    }
}

/* one L=128 chain step, z/y skewed exactly like the L=32/64 steps.  No
 * software prefetch anywhere (seven losses/washes across this lineage; in
 * the DRAM regime the streamer owns the sequential halves and nothing owns
 * the X128-strided x-pass -- measured, not assumed, see the r8 record). */
static inline __attribute__((always_inline))
void fft_step128(double *restrict S, const double *restrict C, const int mode)
{
    for (int x = 0; x <= 128; ++x) {
        double *pz = S + (size_t)x * X128;
        double *py = S + (size_t)(x ? x - 1 : 0) * X128;
        if (mode == 1 && x < 128) {
#pragma GCC unroll 1
            for (int y = 0; y < 128; ++y) {
                double *r = pz + (size_t)y * K128;
#pragma GCC unroll 8
                for (int g = 0; g < 16; ++g) {
                    const double *cr = C + (size_t)y * CYS128 + (size_t)g * CGS128 + (size_t)x * 16;
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
        for (int q = 0; q < 16; ++q) {
            if (x < 128)
#pragma GCC unroll 1
                for (int y = 8 * q; y < 8 * q + 8; ++y)
                    zrow128(pz + (size_t)y * K128);
            if (x > 0)
                vfft128i(py + 16 * q, py, K128, 0);
        }
        PROF_T(t1_);
        PROF_ACC(prof_z, t0_, t1_);
    }
    PROF_T(t3_);
#pragma GCC unroll 1
    for (int cc = 0; cc < 2048; ++cc) {
        const int y = cc >> 4, g = cc & 15;
        vfft128i(S + (size_t)y * K128 + 16 * g,
                 C + (size_t)y * CYS128 + (size_t)g * CGS128, X128, mode == 2);
    }
    PROF_T(t4_);
    PROF_ACC(prof_x, t3_, t4_);
#if GP2_PROF
    ++prof_steps;
#endif
}

/* ===================== gen_r11: ONE-SWEEP FUSED CHAIN STEP ==================
 * The all-hands large-size round.  At L=64 (L3 regime, 32.5 GB/s wall, r4)
 * and L=128 (DRAM regime, ~173 MB/step, r8) the two-sweep step is what
 * binds: per step the state crosses the L3/DRAM boundary TWICE (z/y sweep
 * r+w, x sweep r+w) plus one c read -- ~22 MB at 64, ~173 MB at 128.  The r5
 * paper-kill of cross-step fusion assumed the x-pass stays ONE pass; it does
 * not have to.  Split the x-FFT into its own two radix-8(x16) stages and the
 * step becomes ONE volume sweep:
 *
 *   x-stage-2(step s) + map(s) + z(s+1) + y(s+1) + x-stage-1(s+1)
 *
 * all inside a TILE of G planes that stays cache-resident (8 planes = 557 KB
 * ~ L2 at 64; 16 planes = 4.3 MB ~ L3 at 128).  Per step the volume now
 * crosses the boundary ONCE r+w, plus the c read: ~13 MB at 64 (-40%),
 * ~104 MB at 128 (-40%).
 *
 * The stage split is exactly vfft64i/vfft128i's own two passes (same DFT8S/
 * dft16s and CTWS twiddles on the same values, in the same order -- the H
 * buffer is replaced by the tile planes), so the chain output is
 * BIT-IDENTICAL to the two-sweep engine's (cmp-verified).
 *
 * Placement algebra (in place, single volume, NO ping-pong -- and because the
 * stage-2 stores hit the tile lines its own loads just brought in, no RFO
 * either): label the stage-1 outputs A_b[k] (b = plane mod G within the
 * 8-plane DIT split, k = the DFT8-over-a output index) with label
 * l = 8k + b (64) / 16k + b (128).  A sweep's tile t reads labels
 * {G*t + b}, stores its stage-2 outputs (logical X-plane n = t + 8*k1) at
 * tile position k1, and after z/y the tail stage-1 re-stores in-tile.  The
 * physical home of a label therefore permutes by a fixed digit map each
 * sweep: at 64 it is the involution s(8a+b) = 8b+a, so sweeps simply
 * alternate consecutive-plane and stride-8-plane tiles (par flag); at 128
 * (8x16 split) the map is not an involution and an explicit perm[128] of
 * label -> physical plane is composed per sweep.  The epilogue reads logical
 * plane n from its final physical home.  All of it costs zero data motion --
 * plane identity is a base pointer. */
/* Measured verdict (gen_r11, a80n0, rotated same-core pairs, both fused
 * paths BIT-IDENTICAL to the two-sweep chain output):
 *   L=128 (DRAM regime): 12.4-13.0 vs 14.0-14.4 ms/step-vol, 3/3 pairs --
 *     -9..-14% WIN; LLC-load-misses (demand DRAM reads) 42.0M -> 16.4M
 *     (-61%), l2_lines_in +24% (the tile revisits) -- the trade as designed.
 *     SHIPS =1.
 *   L=64 (L3 regime): 715 vs 702 us -- a ~2% LOSS despite LLC demand loads
 *     dropping 36%: the LLC counters show the two-sweep engine's 22 MB/step
 *     was never at the L3 BW wall (demand L3 reads ~13 MB/step = 19 GB/s,
 *     prefetch covers the rest under compute), so the r4/r5 "32.5 GB/s
 *     L3-bound" theory is DEAD -- the number was a coincidence of compute
 *     time.  The fused sweep's +10% instructions (vertical-pass addressing +
 *     plane-pointer reloads) buy traffic nobody was paying for.  Default 0,
 *     kept compilable as the cross-arch/wisdom race control.
 * Transfer note for the L=100 owners (the round's all-hands cell): L=100's
 * 32 MB working set is the L=128 regime, not the L=64 one -- one-sweep
 * fusion with TILE-ORDER c pays there; at L=40/50 (15 MB, L3-resident,
 * the L=64 regime) expect the loss side unless counters show demand DRAM
 * traffic. */
/* gen_r12 FLIPS GP64_FUSE to 1: with the z-into-stage-2 fusion (GP64_ZF,
 * below) the fused sweep now beats the two-sweep step at L=64 by 11-12% at
 * both B=2 and B=1 (4/4 rotated rounds + 2/2 B=1 pairs, a80n0; counters:
 * LLC-loads -86%, l2_lines_in -29% against +9% instructions).  The r11
 * session measured the ZF=0 fused arm LOSING ~2%; that verdict did not
 * reproduce this session even at ZF=0 -- both arms stay compilable and the
 * knob remains a per-host wisdom-race axis. */
#ifndef GP64_FUSE
# define GP64_FUSE 1
#endif
#ifndef GP128_FUSE
# define GP128_FUSE 1
#endif

/* gen_r12: two fusions inside the one-sweep step, both pure reorderings of
 * independent row/column ops (bit-identical); RACED on a80n0, opposite fates:
 *   GP128_ZF=1 SHIPS (-8%, 6/6 rotated rounds): the next step's z-rows run
 *     INSIDE the stage-2 loop -- stage-2's slot (y,g) stores write row y of
 *     all 16 tile planes, so once row y's 16 slots are done the row is
 *     complete and L1-hot; zrow128 eats it immediately.  Measured mechanism
 *     (pmu.sh): instructions -9.3%, l1d.replacement -11%, l2_lines_in/LLC
 *     UNCHANGED -- the win is the zy pass's z-side loop overhead and L1
 *     reloads deleted, NOT an L3-trip cut (the y pass still re-reads the
 *     tile; trip count stays three).
 *   GP128_YF=0 default (a LOSS on top of ZF, +2%; alone -4% vs r11 but
 *     inferior to ZF alone): y fused with x-stage-1 per 128-KB column slab
 *     (8 vertical y-FFTs then the vertical DFT8 across the group, slab
 *     L2-hot).  It DOES cut L3->L2 traffic (l2_lines_in -18%) but the slab
 *     walk is 128-B touches at 2112-B stride -- no streamer covers it, and
 *     LLC-load-misses (demand) rise +33%: prefetch-covered traffic traded
 *     for latency-exposed demand misses.  Same lesson as the r11 L=64 fusion
 *     loss, one level down the hierarchy: covered traffic is nearly free;
 *     only DEMAND misses are worth restructuring against.
 * All four knob combinations compile; (0,0) is the exact r11 codegen. */
#ifndef GP128_ZF
# define GP128_ZF 1
#endif
#ifndef GP128_YF
# define GP128_YF 0
#endif

/* Fused-path c layout (the r4 GP2_CT lesson taken one step further): the
 * x-fastest custody c would be read by x-stage-2's map as 8 (16) touches at
 * 1-KB stride with an 8.25-KB per-PC stride -- no prefetcher covers that
 * (measured: the first fused build lost 4%/10% at 64/128 with exactly that
 * scatter).  Store c in TILE-CONSUMPTION order instead: slot (t, y, g, k1)
 * at t*CFT + (y*G + g)*(8*16) + k1*16, so each tile's c is ONE sequential
 * ~0.5 MB (64) / 4 MB (128) stream, and the stage-2 slot loop's c pointer
 * is linear in the loop index.  Fits inside the CVOLD64/CVOLD128 blocks. */
#define CFT64  ((size_t)64 * 8 * 128)       /* 65536 doubles per tile  */
#define CFT128 ((size_t)128 * 16 * 256)     /* 524288 doubles per tile */

/* x-stage-1: vertical in-place DFT8 across 8 custody planes at every slot
 * (arithmetic = vfft64i/vfft128i pass 1).  nslots = 512 (L=64, K64 rows) or
 * 2048 (L=128, K128 rows); gshift/gmask pick the slot decomposition. */
static inline __attribute__((always_inline))
void xs1_vert(double *const pl[8], const int nslots, const int gshift,
              const int gmask, const long ks)
{
#pragma GCC unroll 1
    for (int s = 0; s < nslots; ++s) {
        const long off = (long)(s >> gshift) * ks + 16 * (s & gmask);
        v8d xr[8], xi[8], yr[8], yi[8];
#pragma GCC unroll 8
        for (int a = 0; a < 8; ++a) {
            xr[a] = *(const v8d *)(pl[a] + off);
            xi[a] = *(const v8d *)(pl[a] + off + 8);
        }
        DFT8S(xr, xi, yr, yi);
#pragma GCC unroll 8
        for (int k = 0; k < 8; ++k) {
            *(v8d *)(pl[k] + off) = yr[k];
            *(v8d *)(pl[k] + off + 8) = yi[k];
        }
    }
}

/* x-stage-2 at L=64: vertical twiddle + DFT8 across the tile's 8 planes,
 * optional fused map (c x-fastest: the 8 c-slots of one (y,g) are 8 touches
 * at 1-KB stride inside one CGS64 group).  In place: all 16 loads precede
 * the stores. */
static inline __attribute__((always_inline))
void xs2_64(double *const pl[8], const double *restrict Ct, const int t,
            const int domap)
{
#pragma GCC unroll 1
    for (int s = 0; s < 512; ++s) {
        const long off = (long)(s >> 3) * K64 + 16 * (s & 7);
        const double *cs = Ct + (size_t)s * 128;
        v8d tr[8], ti[8], zr[8], zi[8];
#pragma GCC unroll 8
        for (int b = 0; b < 8; ++b) {
            tr[b] = *(const v8d *)(pl[b] + off);
            ti[b] = *(const v8d *)(pl[b] + off + 8);
        }
        if (t) {
#pragma GCC unroll 7
            for (int b = 1; b < 8; ++b)
                CTWS(tr[b], ti[b], tw64c[b * t], tw64s[b * t]);
        }
        DFT8S(tr, ti, zr, zi);
#pragma GCC unroll 8
        for (int k1 = 0; k1 < 8; ++k1) {
            double *q = pl[k1] + off;
            if (domap) {
                const double *cq = cs + 16 * k1;
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

/* x-stage-2 at L=128: vertical twiddle + DFT16 across the tile's 16 planes
 * (arithmetic = vfft128i pass 2: CTWS by W128^{b*t}, dft16s). */
static inline __attribute__((always_inline))
void xs2_128(double *const pl[16], const double *restrict Ct, const int t,
             const int domap)
{
#pragma GCC unroll 1
    for (int s = 0; s < 2048; ++s) {
        const long off = (long)(s >> 4) * K128 + 16 * (s & 15);
        const double *cs = Ct + (size_t)s * 256;
        v8d tr[16], ti[16], zr[16], zi[16];
#pragma GCC unroll 16
        for (int b = 0; b < 16; ++b) {
            tr[b] = *(const v8d *)(pl[b] + off);
            ti[b] = *(const v8d *)(pl[b] + off + 8);
        }
        if (t) {
#pragma GCC unroll 15
            for (int b = 1; b < 16; ++b)
                CTWS(tr[b], ti[b], tw128c[b * t], tw128s[b * t]);
        }
        dft16s(tr, ti, zr, zi);
#pragma GCC unroll 16
        for (int mm = 0; mm < 16; ++mm) {
            double *q = pl[mm] + off;
            if (domap) {
                const double *cq = cs + 16 * mm;
                v8d wr = zr[mm] + *(const v8d *)cq;
                v8d wi = zi[mm] + *(const v8d *)(cq + 8);
                v8d rr, ii;
                MAP8V(wr, wi, rr, ii);
                *(v8d *)q = rr;
                *(v8d *)(q + 8) = ii;
            } else {
                *(v8d *)q = zr[mm];
                *(v8d *)(q + 8) = zi[mm];
            }
        }
    }
}

/* gen_r12 GP128_ZF: xs2_128 with the next step's z-pass fused per row.
 * Slot order identical to xs2_128 (s = 16*y + g, y-major, same sequential c
 * walk); after row y's 16 slots are stored (mapped), row y of all 16 tile
 * planes is complete and L1-hot -- zrow128 runs on it immediately. */
static inline __attribute__((always_inline))
void xs2z_128(double *const pl[16], const double *restrict Ct, const int t,
              const int full)
{
#pragma GCC unroll 1
    for (int y = 0; y < 128; ++y) {
#pragma GCC unroll 1
        for (int g = 0; g < 16; ++g) {
            const long off = (long)y * K128 + 16 * g;
            const double *cs = Ct + ((size_t)(16 * y + g)) * 256;
            v8d tr[16], ti[16], zr[16], zi[16];
#pragma GCC unroll 16
            for (int b = 0; b < 16; ++b) {
                tr[b] = *(const v8d *)(pl[b] + off);
                ti[b] = *(const v8d *)(pl[b] + off + 8);
            }
            if (t) {
#pragma GCC unroll 15
                for (int b = 1; b < 16; ++b)
                    CTWS(tr[b], ti[b], tw128c[b * t], tw128s[b * t]);
            }
            dft16s(tr, ti, zr, zi);
#pragma GCC unroll 16
            for (int mm = 0; mm < 16; ++mm) {
                double *q = pl[mm] + off;
                if (full) {
                    const double *cq = cs + 16 * mm;
                    v8d wr = zr[mm] + *(const v8d *)cq;
                    v8d wi = zi[mm] + *(const v8d *)(cq + 8);
                    v8d rr, ii;
                    MAP8V(wr, wi, rr, ii);
                    *(v8d *)q = rr;
                    *(v8d *)(q + 8) = ii;
                } else {
                    *(v8d *)q = zr[mm];
                    *(v8d *)(q + 8) = zi[mm];
                }
            }
        }
        if (full) {
#pragma GCC unroll 1
            for (int mm = 0; mm < 16; ++mm)
                zrow128(pl[mm] + (size_t)y * K128);
        }
    }
}

/* gen_r12 GP64_ZF: the same z-into-stage-2 fusion at G=8 (only reachable
 * under GP64_FUSE=1; raced against both the r11 fused arm and the two-sweep
 * default -- see the strategy record). */
#ifndef GP64_ZF
# define GP64_ZF 1
#endif
static inline __attribute__((always_inline)) __attribute__((unused))
void xs2z_64(double *const pl[8], const double *restrict Ct, const int t,
             const int full)
{
#pragma GCC unroll 1
    for (int y = 0; y < 64; ++y) {
#pragma GCC unroll 1
        for (int g = 0; g < 8; ++g) {
            const long off = (long)y * K64 + 16 * g;
            const double *cs = Ct + ((size_t)(8 * y + g)) * 128;
            v8d tr[8], ti[8], zr[8], zi[8];
#pragma GCC unroll 8
            for (int b = 0; b < 8; ++b) {
                tr[b] = *(const v8d *)(pl[b] + off);
                ti[b] = *(const v8d *)(pl[b] + off + 8);
            }
            if (t) {
#pragma GCC unroll 7
                for (int b = 1; b < 8; ++b)
                    CTWS(tr[b], ti[b], tw64c[b * t], tw64s[b * t]);
            }
            DFT8S(tr, ti, zr, zi);
#pragma GCC unroll 8
            for (int k1 = 0; k1 < 8; ++k1) {
                double *q = pl[k1] + off;
                if (full) {
                    const double *cq = cs + 16 * k1;
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
        if (full) {
#pragma GCC unroll 1
            for (int mm = 0; mm < 8; ++mm)
                zrow64(pl[mm] + (size_t)y * K64);
        }
    }
}

static inline __attribute__((always_inline)) __attribute__((unused))
void y_tile64(double *const pl[], const int np)
{
    for (int i = 0; i < np; ++i)
#pragma GCC unroll 1
        for (int q = 0; q < 8; ++q)
            vfft64i(pl[i] + 16 * q, pl[i], K64, 0, 0);
}

/* gen_r12 GP128_YF: x-stage-1 restricted to column q of an 8-plane group
 * (the 128 slots at off = y*K128 + 16*q). */
static inline __attribute__((always_inline))
void xs1_col128(double *const gp[8], const int q)
{
#pragma GCC unroll 1
    for (int y = 0; y < 128; ++y) {
        const long off = (long)y * K128 + 16 * q;
        v8d xr[8], xi[8], yr[8], yi[8];
#pragma GCC unroll 8
        for (int a = 0; a < 8; ++a) {
            xr[a] = *(const v8d *)(gp[a] + off);
            xi[a] = *(const v8d *)(gp[a] + off + 8);
        }
        DFT8S(xr, xi, yr, yi);
#pragma GCC unroll 8
        for (int k = 0; k < 8; ++k) {
            *(v8d *)(gp[k] + off) = yr[k];
            *(v8d *)(gp[k] + off + 8) = yi[k];
        }
    }
}

/* gen_r12 GP128_YF: y + x-stage-1 as one pass over the group, per 128-KB
 * column slab (8 vertical y-FFTs, then the vertical DFT8 across the group
 * on the slots those y-FFTs just wrote, all L2-hot). */
static inline __attribute__((always_inline))
void yxs1_128(double *const gp[8])
{
#pragma GCC unroll 1
    for (int q = 0; q < 16; ++q) {
#pragma GCC unroll 1
        for (int a = 0; a < 8; ++a)
            vfft128i(gp[a] + 16 * q, gp[a], K128, 0);
        xs1_col128(gp, q);
    }
}

/* single-fusion race arms (GP128_ZF xor GP128_YF): plain per-plane z / y */
static inline __attribute__((always_inline)) __attribute__((unused))
void z_tile128(double *const pl[], const int np)
{
    for (int i = 0; i < np; ++i)
#pragma GCC unroll 1
        for (int y = 0; y < 128; ++y)
            zrow128(pl[i] + (size_t)y * K128);
}

static inline __attribute__((always_inline)) __attribute__((unused))
void y_tile128(double *const pl[], const int np)
{
    for (int i = 0; i < np; ++i)
#pragma GCC unroll 1
        for (int q = 0; q < 16; ++q)
            vfft128i(pl[i] + 16 * q, pl[i], K128, 0);
}

/* z + y over np tile planes, skewed exactly like fft_step64's q-loop
 * (plane i's z-rows at codelet granularity against plane i-1's y-lines;
 * bit-identical: each plane's y strictly after its own z). */
static inline __attribute__((always_inline))
void zy_tile64(double *const pl[], const int np)
{
    for (int i = 0; i <= np; ++i) {
#pragma GCC unroll 1
        for (int q = 0; q < 8; ++q) {
            if (i < np)
#pragma GCC unroll 1
                for (int y = 8 * q; y < 8 * q + 8; ++y)
                    zrow64(pl[i] + (size_t)y * K64);
            if (i > 0)
                vfft64i(pl[i - 1] + 16 * q, pl[i - 1], K64, 0, 0);
        }
    }
}

static inline __attribute__((always_inline))
void zy_tile128(double *const pl[], const int np)
{
    for (int i = 0; i <= np; ++i) {
#pragma GCC unroll 1
        for (int q = 0; q < 16; ++q) {
            if (i < np)
#pragma GCC unroll 1
                for (int y = 8 * q; y < 8 * q + 8; ++y)
                    zrow128(pl[i] + (size_t)y * K128);
            if (i > 0)
                vfft128i(pl[i - 1] + 16 * q, pl[i - 1], K128, 0);
        }
    }
}

/* prologue: z + y + x-stage-1 of step 1, per stride-G plane group; leaves
 * the A-form at label = physical (identity placement). */
static __attribute__((unused)) void fft_prologue64(double *restrict S)
{
    for (int t = 0; t < 8; ++t) {
        double *pl[8];
        for (int a = 0; a < 8; ++a)
            pl[a] = S + (size_t)(8 * a + t) * X64;
        zy_tile64(pl, 8);
        xs1_vert(pl, 512, 3, 7, K64);
    }
}

static void fft_prologue128(double *restrict S)
{
    for (int t = 0; t < 16; ++t) {
        double *pl[8];
        for (int a = 0; a < 8; ++a)
            pl[a] = S + (size_t)(16 * a + t) * X128;
        zy_tile128(pl, 8);
        xs1_vert(pl, 2048, 4, 15, K128);
    }
}

/* one fused sweep at L=64.  par: 0 = tiles are consecutive plane groups,
 * 1 = stride-8 groups (the placement involution s(8a+b)=8b+a alternates).
 * full=1: x-stage-2 + map + z + y + x-stage-1 (steady chain sweep);
 * full=0: x-stage-2 only, raw z (the chain-final sweep). */
static __attribute__((unused)) void fft_sweep64(double *restrict S, const double *restrict C,
                        const int par, const int full)
{
    for (int t = 0; t < 8; ++t) {
        double *pl[8];
        for (int b = 0; b < 8; ++b)
            pl[b] = S + (size_t)(par ? 8 * b + t : 8 * t + b) * X64;
#if GP64_ZF
        xs2z_64(pl, C + (size_t)t * CFT64, t, full);
        if (!full) continue;
        y_tile64(pl, 8);
#else
        xs2_64(pl, C + (size_t)t * CFT64, t, full);
        if (!full) continue;
        zy_tile64(pl, 8);
#endif
        xs1_vert(pl, 512, 3, 7, K64);
    }
}

/* one fused sweep at L=128.  pp[label] = physical plane; the 8x16 split's
 * placement map is not an involution, so pp is composed per sweep:
 * new_pp[16k+b] = pp[b<8 ? 16b+2k : 16(b-8)+2k+1].  Stage-1 splits into two
 * vertical DFT8s: even tile positions are next step's group t, odd are
 * group t+8. */
static void fft_sweep128(double *restrict S, const double *restrict C,
                         int *restrict pp, const int full)
{
    for (int t = 0; t < 8; ++t) {
        double *pl[16];
        for (int b = 0; b < 16; ++b)
            pl[b] = S + (size_t)pp[16 * t + b] * X128;
#if GP128_ZF
        xs2z_128(pl, C + (size_t)t * CFT128, t, full);
#else
        xs2_128(pl, C + (size_t)t * CFT128, t, full);
#endif
        if (!full) continue;
#if GP128_ZF && !GP128_YF
        y_tile128(pl, 16);
#elif !GP128_ZF && GP128_YF
        z_tile128(pl, 16);
#elif !GP128_ZF && !GP128_YF
        zy_tile128(pl, 16);          /* the exact gen_r11 arm */
#endif
        double *ple[8], *plo[8];
        for (int a = 0; a < 8; ++a) { ple[a] = pl[2 * a]; plo[a] = pl[2 * a + 1]; }
#if GP128_YF
        yxs1_128(ple);
        yxs1_128(plo);
#else
        xs1_vert(ple, 2048, 4, 15, K128);
        xs1_vert(plo, 2048, 4, 15, K128);
#endif
    }
    if (full) {
        int np[128];
        for (int k = 0; k < 8; ++k)
            for (int b = 0; b < 16; ++b)
                np[16 * k + b] = pp[b < 8 ? 16 * b + 2 * k : 16 * (b - 8) + 2 * k + 1];
        memcpy(pp, np, sizeof np);
    }
}

/* natural c -> fused tile-order custody c */
static __attribute__((unused)) void nat_to_cust_cf64(const double *restrict nat, double *restrict C)
{
    for (int x = 0; x < 64; ++x) {
        double *qb = C + (size_t)(x & 7) * CFT64 + (size_t)(x >> 3) * 16;
        for (int y = 0; y < 64; ++y) {
            const double *r = nat + ((size_t)x * 64 + y) * 128;
            double *q = qb + (size_t)y * (8 * 128);
            for (int g = 0; g < 8; ++g) {
                v8d A = LDU(r + 16 * g), B = LDU(r + 16 * g + 8);
                *(v8d *)(q + (size_t)g * 128) = DEIN_RE(A, B);
                *(v8d *)(q + (size_t)g * 128 + 8) = DEIN_IM(A, B);
            }
        }
    }
}

static void nat_to_cust_cf128(const double *restrict nat, double *restrict C)
{
    for (int x = 0; x < 128; ++x) {
        double *qb = C + (size_t)(x & 7) * CFT128 + (size_t)(x >> 3) * 16;
        for (int y = 0; y < 128; ++y) {
            const double *r = nat + ((size_t)x * 128 + y) * 256;
            double *q = qb + (size_t)y * (16 * 256);
            for (int g = 0; g < 16; ++g) {
                v8d A = LDU(r + 16 * g), B = LDU(r + 16 * g + 8);
                *(v8d *)(q + (size_t)g * 256) = DEIN_RE(A, B);
                *(v8d *)(q + (size_t)g * 256 + 8) = DEIN_IM(A, B);
            }
        }
    }
}

/* fused-path epilogues: map step m's raw z (custody, permuted planes) into
 * natural final_out.  At 64 the final placement is par ? identity : s. */
static __attribute__((unused)) void cust_map_to_nat_p64(const double *restrict S, const double *restrict C,
                                double *restrict nat, const int par)
{
    for (int x = 0; x < 64; ++x) {
        const int q = par ? x : (((x & 7) << 3) | (x >> 3));
        const double *pq = S + (size_t)q * X64;
        for (int y = 0; y < 64; ++y) {
            const double *row = pq + (size_t)y * K64;
            double *r = nat + ((size_t)x * 64 + y) * 128;
            for (int g = 0; g < 8; ++g) {
                const double *cq = C + (size_t)(x & 7) * CFT64
                                 + ((size_t)y * 8 + g) * 128 + (size_t)(x >> 3) * 16;
                v8d wr = *(const v8d *)(row + 16 * g) + *(const v8d *)cq;
                v8d wi = *(const v8d *)(row + 16 * g + 8) + *(const v8d *)(cq + 8);
                v8d rr, ii;
                MAP8V(wr, wi, rr, ii);
                STU(r + 16 * g, ILV_LO(rr, ii));
                STU(r + 16 * g + 8, ILV_HI(rr, ii));
            }
        }
    }
}

static void cust_map_to_nat_p128(const double *restrict S, const double *restrict C,
                                 double *restrict nat, const int *restrict pp)
{
    for (int x = 0; x < 128; ++x) {
        const double *pq = S + (size_t)pp[16 * (x & 7) + (x >> 3)] * X128;
        for (int y = 0; y < 128; ++y) {
            const double *row = pq + (size_t)y * K128;
            double *r = nat + ((size_t)x * 128 + y) * 256;
            for (int g = 0; g < 16; ++g) {
                const double *cq = C + (size_t)(x & 7) * CFT128
                                 + ((size_t)y * 16 + g) * 256 + (size_t)(x >> 3) * 16;
                v8d wr = *(const v8d *)(row + 16 * g) + *(const v8d *)cq;
                v8d wi = *(const v8d *)(row + 16 * g + 8) + *(const v8d *)(cq + 8);
                v8d rr, ii;
                MAP8V(wr, wi, rr, ii);
                STU(r + 16 * g, ILV_LO(rr, ii));
                STU(r + 16 * g + 8, ILV_HI(rr, ii));
            }
        }
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
static SCHED_ATTR void fft_step128_plain(double *restrict S, const double *restrict C)
{ fft_step128(S, C, 0); }
static __attribute__((unused)) SCHED_ATTR void fft_step128_premap(double *restrict S, const double *restrict C)
{ fft_step128(S, C, 1); }
static SCHED_ATTR void fft_step128_xmap(double *restrict S, const double *restrict C)
{ fft_step128(S, C, 2); }

/* ================= gen_r13: execute-path conversion fusion (GP2_XFE) =======
 * Natural-input z-codelets and natural-output vertical FFTs, one per engine.
 * Natural layout (driver convention): row (x,y) = nat + ((x*L + y)*2L)
 * doubles, slot g at +16g; so natural plane stride = 2*L*L doubles and the
 * x-pass output for index n of column (y,g) lands at nat + n*2LL + y*2L +
 * 16g.  The load side costs 2 DEIN shuffles per slot and the store side
 * 2 ILV shuffles per slot -- exactly what the deleted conversion sweeps
 * paid, so the fusion is a pure deletion of one custody-volume store+load
 * per side.  All codelet arithmetic (DFT order, twiddles, FTW fold) is the
 * byte-for-byte shape of the unfused variants: the transform values are
 * IDENTICAL to nat_to_cust + step_plain + cust_to_nat by construction. */

/* L=32: zpair loading two natural rows (the zpair body with DEIN loads) */
static inline __attribute__((always_inline))
void zpair_n(const double *restrict nA, const double *restrict nB,
             double *restrict rA, double *restrict rB)
{
    v8d Mr[8], Mi[8];
#pragma GCC unroll 4
    for (int g = 0; g < 4; ++g) {
        v8d A0 = LDU(nA + 16 * g), A1 = LDU(nA + 16 * g + 8);
        v8d B0 = LDU(nB + 16 * g), B1 = LDU(nB + 16 * g + 8);
        Mr[g] = DEIN_RE(A0, A1);     Mi[g] = DEIN_IM(A0, A1);
        Mr[4 + g] = DEIN_RE(B0, B1); Mi[4 + g] = DEIN_IM(B0, B1);
    }
    DFT4S(Mr, Mi, Mr, Mi);
    DFT4S(Mr + 4, Mi + 4, Mr + 4, Mi + 4);
#pragma GCC unroll 3
    for (int k2 = 1; k2 < 4; ++k2) {
        v8d tr = *(const v8d *)WZ32R[k2], ti = *(const v8d *)WZ32I[k2];
        CTWV(Mr[k2], Mi[k2], tr, ti);
        CTWV(Mr[4 + k2], Mi[4 + k2], tr, ti);
    }
    v8d Tr[8], Ti[8], Or[8], Oi[8];
    TR8(Mr, Tr);
    TR8(Mi, Ti);
    DFT8S(Tr, Ti, Or, Oi);
#if GP2_ZST
#pragma GCC unroll 4
    for (int m = 0; m < 4; ++m) {
        _mm256_storeu_pd(rA + 16 * m,      _mm512_castpd512_pd256((__m512d)Or[2 * m]));
        _mm256_storeu_pd(rA + 16 * m + 4,  _mm512_castpd512_pd256((__m512d)Or[2 * m + 1]));
        _mm256_storeu_pd(rA + 16 * m + 8,  _mm512_castpd512_pd256((__m512d)Oi[2 * m]));
        _mm256_storeu_pd(rA + 16 * m + 12, _mm512_castpd512_pd256((__m512d)Oi[2 * m + 1]));
        _mm256_storeu_pd(rB + 16 * m,      _mm512_extractf64x4_pd((__m512d)Or[2 * m], 1));
        _mm256_storeu_pd(rB + 16 * m + 4,  _mm512_extractf64x4_pd((__m512d)Or[2 * m + 1], 1));
        _mm256_storeu_pd(rB + 16 * m + 8,  _mm512_extractf64x4_pd((__m512d)Oi[2 * m], 1));
        _mm256_storeu_pd(rB + 16 * m + 12, _mm512_extractf64x4_pd((__m512d)Oi[2 * m + 1], 1));
    }
#else
#pragma GCC unroll 4
    for (int m = 0; m < 4; ++m) {
        *(v8d *)(rA + 16 * m)     = VSH8(Or[2 * m], Or[2 * m + 1], 0,1,2,3,8,9,10,11);
        *(v8d *)(rA + 16 * m + 8) = VSH8(Oi[2 * m], Oi[2 * m + 1], 0,1,2,3,8,9,10,11);
        *(v8d *)(rB + 16 * m)     = VSH8(Or[2 * m], Or[2 * m + 1], 4,5,6,7,12,13,14,15);
        *(v8d *)(rB + 16 * m + 8) = VSH8(Oi[2 * m], Oi[2 * m + 1], 4,5,6,7,12,13,14,15);
    }
#endif
}

/* L=32: vfft32 with pass-2 stores interleaved straight to natural out */
static inline __attribute__((always_inline))
void vfft32e(const double *restrict base, double *nout, const long st)
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
            double *q = nout + (long)(k2 + 8 * k1) * 2048;
            STU(q, ILV_LO(zr[k1], zi[k1]));
            STU(q + 8, ILV_HI(zr[k1], zi[k1]));
        }
    }
}

/* L=64: zrow64 loading one natural row */
static inline __attribute__((always_inline))
void zrow64_n(const double *restrict nr, double *restrict r)
{
    v8d Mr[8], Mi[8];
#pragma GCC unroll 8
    for (int g = 0; g < 8; ++g) {
        v8d A = LDU(nr + 16 * g), B = LDU(nr + 16 * g + 8);
        Mr[g] = DEIN_RE(A, B);
        Mi[g] = DEIN_IM(A, B);
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

/* L=64: vfft64i with pass-2 stores to natural out (no map, no prefetch) */
static inline __attribute__((always_inline))
void vfft64e(const double *restrict base, double *nout, const long st)
{
    v8d Hr[64], Hi[64];
#pragma GCC unroll 1
    for (int b = 0; b < 8; ++b) {
        v8d xr[8], xi[8], yr[8], yi[8];
#pragma GCC unroll 8
        for (int a = 0; a < 8; ++a) {
            const double *pp = base + (long)(8 * a + b) * st;
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
            double *q = nout + (long)(k2 + 8 * k1) * 8192;
            STU(q, ILV_LO(zr[k1], zi[k1]));
            STU(q + 8, ILV_HI(zr[k1], zi[k1]));
        }
    }
}

/* L=16: zquad16 loading four natural rows */
static inline __attribute__((always_inline))
void zquad16_n(const double *restrict n0, const double *restrict n1,
               const double *restrict n2, const double *restrict n3,
               double *restrict r0, double *restrict r1,
               double *restrict r2, double *restrict r3)
{
    v8d Ar[8], Ai[8];
    const v8d twr = *(const v8d *)wz16r, twi = *(const v8d *)wz16i;
#define ZQ_LOADN(r, p) do {                                                   \
        v8d A0_ = LDU(p),        B0_ = LDU((p) + 8);                          \
        v8d A1_ = LDU((p) + 16), B1_ = LDU((p) + 24);                         \
        v8d s0r_ = DEIN_RE(A0_, B0_), s0i_ = DEIN_IM(A0_, B0_);               \
        v8d s1r_ = DEIN_RE(A1_, B1_), s1i_ = DEIN_IM(A1_, B1_);               \
        Ar[2*(r)]   = s0r_ + s1r_; Ai[2*(r)]   = s0i_ + s1i_;                 \
        Ar[2*(r)+1] = s0r_ - s1r_; Ai[2*(r)+1] = s0i_ - s1i_;                 \
        CTWV(Ar[2*(r)+1], Ai[2*(r)+1], twr, twi);                             \
    } while (0)
    ZQ_LOADN(0, n0); ZQ_LOADN(1, n1); ZQ_LOADN(2, n2); ZQ_LOADN(3, n3);
#undef ZQ_LOADN
    v8d Tr[8], Ti[8], Or[8], Oi[8];
    TR8(Ar, Tr);
    TR8(Ai, Ti);
    DFT8S(Tr, Ti, Or, Oi);
#if GP16_ZST
#define ZQ_STG(P, V, R) do {                                                  \
        if ((R) == 0) _mm_storeu_pd((P), _mm512_castpd512_pd128((__m512d)(V)));\
        else _mm_storeu_pd((P), _mm512_extractf64x2_pd((__m512d)(V), (R)));   \
    } while (0)
#define ZQ_SCAT(r, p) do {                                                    \
        ZQ_STG((p),      Or[0], (r)); ZQ_STG((p) + 2,  Or[1], (r));           \
        ZQ_STG((p) + 4,  Or[2], (r)); ZQ_STG((p) + 6,  Or[3], (r));           \
        ZQ_STG((p) + 8,  Oi[0], (r)); ZQ_STG((p) + 10, Oi[1], (r));           \
        ZQ_STG((p) + 12, Oi[2], (r)); ZQ_STG((p) + 14, Oi[3], (r));           \
        ZQ_STG((p) + 16, Or[4], (r)); ZQ_STG((p) + 18, Or[5], (r));           \
        ZQ_STG((p) + 20, Or[6], (r)); ZQ_STG((p) + 22, Or[7], (r));           \
        ZQ_STG((p) + 24, Oi[4], (r)); ZQ_STG((p) + 26, Oi[5], (r));           \
        ZQ_STG((p) + 28, Oi[6], (r)); ZQ_STG((p) + 30, Oi[7], (r));           \
    } while (0)
    ZQ_SCAT(0, r0); ZQ_SCAT(1, r1); ZQ_SCAT(2, r2); ZQ_SCAT(3, r3);
#undef ZQ_SCAT
#undef ZQ_STG
#else
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
#endif
}

/* L=16: vfft16i with pass-2 stores to natural out (no map) */
static inline __attribute__((always_inline))
void vfft16e(const double *restrict base, double *nout, const long st)
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
            double *q = nout + (long)(k2 + 8 * k1) * 512;
            STU(q, ILV_LO(zr[k1], zi[k1]));
            STU(q + 8, ILV_HI(zr[k1], zi[k1]));
        }
    }
}

/* L=128: zrow128 loading one natural row */
static inline __attribute__((always_inline))
void zrow128_n(const double *restrict nr, double *restrict r)
{
    v8d Mr[16], Mi[16], Br[16], Bi[16];
#pragma GCC unroll 16
    for (int g = 0; g < 16; ++g) {
        v8d A = LDU(nr + 16 * g), B = LDU(nr + 16 * g + 8);
        Mr[g] = DEIN_RE(A, B);
        Mi[g] = DEIN_IM(A, B);
    }
    dft16s(Mr, Mi, Br, Bi);
#pragma GCC unroll 15
    for (int k2 = 1; k2 < 16; ++k2)
        CTWV(Br[k2], Bi[k2], *(const v8d *)wz128r[k2], *(const v8d *)wz128i[k2]);
    v8d Tr[8], Ti[8], Or[8], Oi[8];
    TR8(Br, Tr);
    TR8(Bi, Ti);
    DFT8S(Tr, Ti, Or, Oi);
#pragma GCC unroll 8
    for (int k1 = 0; k1 < 8; ++k1) {
        *(v8d *)(r + 16 * (2 * k1)) = Or[k1];
        *(v8d *)(r + 16 * (2 * k1) + 8) = Oi[k1];
    }
    TR8(Br + 8, Tr);
    TR8(Bi + 8, Ti);
    DFT8S(Tr, Ti, Or, Oi);
#pragma GCC unroll 8
    for (int k1 = 0; k1 < 8; ++k1) {
        *(v8d *)(r + 16 * (2 * k1 + 1)) = Or[k1];
        *(v8d *)(r + 16 * (2 * k1 + 1) + 8) = Oi[k1];
    }
}

/* L=128: vfft128i with pass-2 stores to natural out (no map) */
static inline __attribute__((always_inline))
void vfft128e(const double *restrict base, double *nout, const long st)
{
    v8d Hr[128], Hi[128];
#pragma GCC unroll 1
    for (int b = 0; b < 16; ++b) {
        v8d xr[8], xi[8], yr[8], yi[8];
#pragma GCC unroll 8
        for (int a = 0; a < 8; ++a) {
            const double *pp = base + (long)(16 * a + b) * st;
            xr[a] = *(const v8d *)pp;
            xi[a] = *(const v8d *)(pp + 8);
        }
        DFT8S(xr, xi, yr, yi);
#pragma GCC unroll 8
        for (int k1 = 0; k1 < 8; ++k1) { Hr[16 * k1 + b] = yr[k1]; Hi[16 * k1 + b] = yi[k1]; }
    }
#pragma GCC unroll 1
    for (int k1 = 0; k1 < 8; ++k1) {
        v8d tr[16], ti[16], zr[16], zi[16];
#pragma GCC unroll 16
        for (int b = 0; b < 16; ++b) { tr[b] = Hr[16 * k1 + b]; ti[b] = Hi[16 * k1 + b]; }
        if (k1) {
#pragma GCC unroll 15
            for (int b = 1; b < 16; ++b)
                CTWS(tr[b], ti[b], tw128c[b * k1], tw128s[b * k1]);
        }
        dft16s(tr, ti, zr, zi);
#pragma GCC unroll 16
        for (int m = 0; m < 16; ++m) {
            double *q = nout + (long)(k1 + 8 * m) * 32768;
            STU(q, ILV_LO(zr[m], zi[m]));
            STU(q + 8, ILV_HI(zr[m], zi[m]));
        }
    }
}

/* The fused-execute steps: the plain two-sweep step shapes with the z-phase
 * reading NATURAL input and the x-pass storing NATURAL output.  S is scratch
 * only.  Phase ordering makes them in-place-safe (every natural read happens
 * in the z-phase, every natural write in the x-phase).  nin/nout carry no
 * restrict against each other for exactly that reason. */
static SCHED_ATTR __attribute__((unused))
void fft_exec32(double *restrict S, const double *nin, double *nout)
{
    for (int x = 0; x <= 32; ++x) {
        const double *nz = nin + (size_t)x * 2048;
        double *pz = S + (size_t)x * XS;
        double *py = S + (size_t)(x ? x - 1 : 0) * XS;
#pragma GCC unroll 1
        for (int q = 0; q < 4; ++q) {
            if (x < 32)
ZPAIR_UNROLL
                for (int y = 8 * q; y < 8 * q + 8; y += 2)
                    zpair_n(nz + (size_t)y * 64, nz + (size_t)(y + 1) * 64,
                            pz + (size_t)y * KS, pz + (size_t)(y + 1) * KS);
            if (x > 0)
                vfft32(py + 16 * q, KS);
        }
    }
#pragma GCC unroll 1
    for (int cc = 0; cc < 128; ++cc)
        vfft32e(S + (size_t)(cc >> 2) * KS + 16 * (cc & 3),
                nout + (size_t)(cc >> 2) * 64 + 16 * (cc & 3), XS);
}

static SCHED_ATTR __attribute__((unused))
void fft_exec64(double *restrict S, const double *nin, double *nout)
{
    for (int x = 0; x <= 64; ++x) {
        const double *nz = nin + (size_t)x * 8192;
        double *pz = S + (size_t)x * X64;
        double *py = S + (size_t)(x ? x - 1 : 0) * X64;
#pragma GCC unroll 1
        for (int q = 0; q < 8; ++q) {
            if (x < 64)
#pragma GCC unroll 1
                for (int y = 8 * q; y < 8 * q + 8; ++y)
                    zrow64_n(nz + (size_t)y * 128, pz + (size_t)y * K64);
            if (x > 0)
                vfft64i(py + 16 * q, py, K64, 0, 0);
        }
    }
#pragma GCC unroll 1
    for (int cc = 0; cc < 512; ++cc) {
        const int y = cc >> 3, g = cc & 7;
        vfft64e(S + (size_t)y * K64 + 16 * g, nout + (size_t)y * 128 + 16 * g, X64);
    }
}

static SCHED_ATTR __attribute__((unused))
void fft_exec16(double *restrict S, const double *nin, double *nout)
{
    for (int x = 0; x <= 16; ++x) {
        const double *nz = nin + (size_t)x * 512;
        double *pz = S + (size_t)x * X16;
        double *py = S + (size_t)(x ? x - 1 : 0) * X16;
#pragma GCC unroll 1
        for (int q = 0; q < 2; ++q) {
            if (x < 16) {
#pragma GCC unroll 1
                for (int h = 0; h < 2; ++h) {
                    const int r = 8 * q + 4 * h;
                    zquad16_n(nz + (size_t)r * 32,       nz + (size_t)(r + 1) * 32,
                              nz + (size_t)(r + 2) * 32, nz + (size_t)(r + 3) * 32,
                              pz + (size_t)r * K16,       pz + (size_t)(r + 1) * K16,
                              pz + (size_t)(r + 2) * K16, pz + (size_t)(r + 3) * K16);
                }
            }
            if (x > 0)
                vfft16i(py + 16 * q, py, K16, 0);
        }
    }
#pragma GCC unroll 1
    for (int cc = 0; cc < 32; ++cc) {
        const int y = cc >> 1, g = cc & 1;
        vfft16e(S + (size_t)y * K16 + 16 * g, nout + (size_t)y * 32 + 16 * g, X16);
    }
}

/* GP128_XF: which side of the execute fusion runs at L=128.  The full form
 * (=2) is a WASH on the node despite deleting ~138 MB of DRAM traffic: the
 * custody block is huge-paged (GP128_HP), so the unfused x-pass stores at
 * X128 stride cost ~8 stores per 2-MB page, while the fused stores hit the
 * driver's 4-KB-paged natural buffer at 256-KB stride -- one page walk per
 * store line, which eats the whole traffic win (raced gen_r13: 17.2-17.4 vs
 * 17.1-17.6 ms, mixed signs).  =1 (SHIPS) fuses the LOAD side only: z reads
 * natural (sequential, streamer-covered), the x-pass keeps its huge-paged S
 * stores, and the sequential cust_to_nat sweep stays -- deletes one custody
 * volume write + read (~69 MB).  =0 unfused control. */
#ifndef GP128_XF
# define GP128_XF 1
#endif
static void cust_to_nat_g(const double *restrict S, double *restrict nat,
                          const int L, const int G, const size_t ks, const size_t xs);
static SCHED_ATTR __attribute__((unused))
void fft_exec128(double *restrict S, const double *nin, double *nout)
{
    for (int x = 0; x <= 128; ++x) {
        const double *nz = nin + (size_t)x * 32768;
        double *pz = S + (size_t)x * X128;
        double *py = S + (size_t)(x ? x - 1 : 0) * X128;
#pragma GCC unroll 1
        for (int q = 0; q < 16; ++q) {
            if (x < 128)
#pragma GCC unroll 1
                for (int y = 8 * q; y < 8 * q + 8; ++y)
                    zrow128_n(nz + (size_t)y * 256, pz + (size_t)y * K128);
            if (x > 0)
                vfft128i(py + 16 * q, py, K128, 0);
        }
    }
#if GP128_XF == 2
#pragma GCC unroll 1
    for (int cc = 0; cc < 2048; ++cc) {
        const int y = cc >> 4, g = cc & 15;
        vfft128e(S + (size_t)y * K128 + 16 * g, nout + (size_t)y * 256 + 16 * g, X128);
    }
#else
#pragma GCC unroll 1
    for (int cc = 0; cc < 2048; ++cc) {
        const int y = cc >> 4, g = cc & 15;
        vfft128i(S + (size_t)y * K128 + 16 * g, S, X128, 0);
    }
    cust_to_nat_g(S, nout, 128, 16, K128, X128);
#endif
}

/* ================= gen_r14: TILED ONE-SWEEP EXECUTE (GP128_XT / GP64_XT) ====
 * The r13 record's named next lever, built: the r11/r12 fused-sweep structure
 * applied to the EXECUTE path (its m=1 case).  Two sweeps:
 *   sweep 1 (16 stride-16 groups at 128 / 8 stride-8 groups at 64):
 *     z (NATURAL loads, zrow*_n) + y skewed per plane, then x-stage-1
 *     (xs1_vert) on the group -- the fft_prologue* shape with natural input;
 *     leaves A_b[k] at plane G*k+b (identity placement, no perm needed at
 *     m=1).
 *   sweep 2 (8 tiles of G consecutive planes): x-stage-2 (twiddle t + vertical
 *     DFT) reading the tile, EMITTING straight to natural out (ILV) -- the
 *     tile is read-only, so the in-place x-pass S write AND the separate
 *     cust_to_nat S read are both deleted (~69 MB of the L=128 call's ~205 MB
 *     DRAM set).
 * Why this dodges the GP128_XF=2 TLB inversion (r13): the unsplit x-pass
 * touches all 128 natural planes per COLUMN (one page walk per store line,
 * 128-page working set); the tiled stage-2 touches only the 16 planes
 * n = t + 8*mm per TILE, each written as sequential 2-KB row chunks -- 16
 * forward streams, 16 live pages, streamer- and DTLB-clean.  Arithmetic is
 * vfft64i/vfft128i's own two passes on the same values in the same order
 * (the r11 stage split), so the output is BIT-IDENTICAL to the GP2_XFE
 * two-sweep execute (cmp-verified).  =0 arms restore the r13 codegen. */
#ifndef GP128_XT
# define GP128_XT 1
#endif
#ifndef GP64_XT
# define GP64_XT 1
#endif

/* z (natural loads) + y over np tile planes, skewed exactly like
 * zy_tile128 / fft_exec128 (plane i's z-rows against plane i-1's y-lines;
 * bit-identical: each plane's y strictly after its own z).  b = the group's
 * plane residue: plane i of the group is natural plane G*i + b. */
static inline __attribute__((always_inline)) __attribute__((unused))
void zy_tile128_n(double *const pl[], const int np, const double *nin, const int b)
{
    for (int i = 0; i <= np; ++i) {
        const double *nz = (i < np) ? nin + (size_t)(16 * i + b) * 32768 : 0;
#pragma GCC unroll 1
        for (int q = 0; q < 16; ++q) {
            if (i < np)
#pragma GCC unroll 1
                for (int y = 8 * q; y < 8 * q + 8; ++y)
                    zrow128_n(nz + (size_t)y * 256, pl[i] + (size_t)y * K128);
            if (i > 0)
                vfft128i(pl[i - 1] + 16 * q, pl[i - 1], K128, 0);
        }
    }
}

static inline __attribute__((always_inline)) __attribute__((unused))
void zy_tile64_n(double *const pl[], const int np, const double *nin, const int b)
{
    for (int i = 0; i <= np; ++i) {
        const double *nz = (i < np) ? nin + (size_t)(8 * i + b) * 8192 : 0;
#pragma GCC unroll 1
        for (int q = 0; q < 8; ++q) {
            if (i < np)
#pragma GCC unroll 1
                for (int y = 8 * q; y < 8 * q + 8; ++y)
                    zrow64_n(nz + (size_t)y * 128, pl[i] + (size_t)y * K64);
            if (i > 0)
                vfft64i(pl[i - 1] + 16 * q, pl[i - 1], K64, 0, 0);
        }
    }
}

/* staged shifted-NT row emit for MISALIGNED caller buffers (the benchFFT
 * case: bench_malloc = malloc+16).  A 2-KB source row (64-B-aligned stack
 * staging) goes to dst at 16/32/48 mod 64: scalar head to the boundary, then
 * 31 zmm built by ONE vpermt2pd each (idx = sh..sh+7 over vector pairs)
 * streamed to the aligned interior, scalar tail.  Single forward stream per
 * row, whole lines only -- none of the WC-buffer thrash a 16-plane
 * interleave of partial-line xmm NT stores produces (raced: that form read
 * 16.7 ms vs 12.1 regular at L=128, a disaster worth recording). */
static inline __attribute__((always_inline)) __attribute__((unused))
void nt_row256(double *restrict dst, const double *restrict src,
               const int sh, const __m512i idx)
{
    for (int i = 0; i < sh; ++i) dst[i] = src[i];
    double *da = dst + sh;
    const v8d *sv = (const v8d *)src;
#pragma GCC unroll 1
    for (int k = 0; k < 31; ++k) {
        __m512d w = _mm512_permutex2var_pd((__m512d)sv[k], idx, (__m512d)sv[k + 1]);
        _mm512_stream_pd(da + 8 * k, w);
    }
    for (int i = sh + 248; i < 256; ++i) dst[i] = src[i];
}

/* x-stage-2 with the stores interleaved straight to natural out (the m=1
 * emit: xs2_128's slot order and arithmetic, cust_to_nat's ILV, no map, no
 * S write).  Tile t holds A_b[t] at pl[b]; output n = t + 8*mm.
 * nt=1: 64-B-aligned caller -- direct zmm non-temporal emit (the natural
 * write is a 33.5-MB pure stream in the DRAM regime; regular stores pay an
 * RFO read per line, vmovntpd deletes those reads -- gen_layout r4's
 * NT-on-DRAM-resident precedent).  nt=2: 16-B-aligned caller -- staged
 * shifted-NT rows (above).  nt=0: regular unaligned stores.  Same bytes at
 * the same addresses in every arm; caller fences after an NT sweep. */
static inline __attribute__((always_inline)) __attribute__((unused))
void xs2e_128(double *const pl[16], double *nout, const int t, const int nt)
{
    double *nq[16];
#pragma GCC unroll 16
    for (int mm = 0; mm < 16; ++mm)
        nq[mm] = nout + (size_t)(t + 8 * mm) * 32768;
    const int sh = (int)((64 - ((uintptr_t)nout & 63)) >> 3) & 7;
    const __m512i idx = _mm512_add_epi64(_mm512_set1_epi64(sh),
                                         _mm512_setr_epi64(0, 1, 2, 3, 4, 5, 6, 7));
#pragma GCC unroll 1
    for (int y = 0; y < 128; ++y) {
        v8d stg[16][32];
#pragma GCC unroll 1
        for (int g = 0; g < 16; ++g) {
            const long off = (long)y * K128 + 16 * g;
            v8d tr[16], ti[16], zr[16], zi[16];
#pragma GCC unroll 16
            for (int b = 0; b < 16; ++b) {
                tr[b] = *(const v8d *)(pl[b] + off);
                ti[b] = *(const v8d *)(pl[b] + off + 8);
            }
            if (t) {
#pragma GCC unroll 15
                for (int b = 1; b < 16; ++b)
                    CTWS(tr[b], ti[b], tw128c[b * t], tw128s[b * t]);
            }
            dft16s(tr, ti, zr, zi);
#pragma GCC unroll 16
            for (int mm = 0; mm < 16; ++mm) {
                if (nt == 2) {
                    stg[mm][2 * g] = ILV_LO(zr[mm], zi[mm]);
                    stg[mm][2 * g + 1] = ILV_HI(zr[mm], zi[mm]);
                } else {
                    double *q = nq[mm] + (long)y * 256 + 16 * g;
                    if (nt == 1) {
                        _mm512_stream_pd(q, (__m512d)ILV_LO(zr[mm], zi[mm]));
                        _mm512_stream_pd(q + 8, (__m512d)ILV_HI(zr[mm], zi[mm]));
                    } else {
                        STU(q, ILV_LO(zr[mm], zi[mm]));
                        STU(q + 8, ILV_HI(zr[mm], zi[mm]));
                    }
                }
            }
        }
        if (nt == 2) {
#pragma GCC unroll 1
            for (int mm = 0; mm < 16; ++mm)
                nt_row256(nq[mm] + (long)y * 256, (const double *)stg[mm], sh, idx);
        }
    }
}

static inline __attribute__((always_inline)) __attribute__((unused))
void xs2e_64(double *const pl[8], double *nout, const int t)
{
    double *nq[8];
#pragma GCC unroll 8
    for (int k1 = 0; k1 < 8; ++k1)
        nq[k1] = nout + (size_t)(t + 8 * k1) * 8192;
#pragma GCC unroll 1
    for (int s = 0; s < 512; ++s) {
        const long off = (long)(s >> 3) * K64 + 16 * (s & 7);
        const long noff = (long)(s >> 3) * 128 + 16 * (s & 7);
        v8d tr[8], ti[8], zr[8], zi[8];
#pragma GCC unroll 8
        for (int b = 0; b < 8; ++b) {
            tr[b] = *(const v8d *)(pl[b] + off);
            ti[b] = *(const v8d *)(pl[b] + off + 8);
        }
        if (t) {
#pragma GCC unroll 7
            for (int b = 1; b < 8; ++b)
                CTWS(tr[b], ti[b], tw64c[b * t], tw64s[b * t]);
        }
        DFT8S(tr, ti, zr, zi);
#pragma GCC unroll 8
        for (int k1 = 0; k1 < 8; ++k1) {
            double *q = nq[k1] + noff;
            STU(q, ILV_LO(zr[k1], zi[k1]));
            STU(q + 8, ILV_HI(zr[k1], zi[k1]));
        }
    }
}

/* The tiled-exec bodies live in their own .text section: they are called
 * once per execute() and must not perturb the hot chain code's placement
 * (the gen_twiddle r5 code-layout hazard -- a +0.5%-mean tax on the graded
 * L=32 chain was measured with them in default .text and vanished with the
 * section split; codegen inside the functions is unchanged). */
/* GP128_NTE=1: non-temporal emit at 128 when the caller's buffer is 64-B
 * aligned (runtime gate; raced -9%, 3/3).  GP128_NTS=1 (raced control,
 * default 0): use the staged shifted-NT rows for 16/32/48-mod-64 callers
 * too -- measured a WASH-to-loss vs plain unaligned stores on a80n0
 * (12.06 vs 11.4-12.1 ms; the L1 staging round trip + permutes eat the RFO
 * saving), and the naive 16-plane interleaved xmm-NT form is a disaster
 * (16.7 ms: partial-line WC thrash).  Misaligned callers therefore default
 * to regular unaligned stores; line splits measured free in the DRAM
 * regime (off-8 == off-16 == 12.07 vs aligned-regular 12.05). */
#ifndef GP128_NTE
# define GP128_NTE 1
#endif
#ifndef GP128_NTS
# define GP128_NTS 0
#endif
static SCHED_ATTR __attribute__((unused, section(".text.gp2exect")))
void fft_exec128t(double *restrict S, const double *nin, double *nout)
{
    for (int b = 0; b < 16; ++b) {
        double *pl[8];
        for (int a = 0; a < 8; ++a)
            pl[a] = S + (size_t)(16 * a + b) * X128;
        zy_tile128_n(pl, 8, nin, b);
        xs1_vert(pl, 2048, 4, 15, K128);
    }
    const uintptr_t al = (uintptr_t)nout & 63;
    const int nt = !GP128_NTE ? 0
                 : (al == 0 ? 1 : ((GP128_NTS && (al & 15) == 0) ? 2 : 0));
    if (nt == 1) {
        for (int t = 0; t < 8; ++t) {
            double *pl[16];
            for (int b = 0; b < 16; ++b)
                pl[b] = S + (size_t)(16 * t + b) * X128;
            xs2e_128(pl, nout, t, 1);
        }
        _mm_sfence();
    } else if (nt == 2) {
        for (int t = 0; t < 8; ++t) {
            double *pl[16];
            for (int b = 0; b < 16; ++b)
                pl[b] = S + (size_t)(16 * t + b) * X128;
            xs2e_128(pl, nout, t, 2);
        }
        _mm_sfence();
    } else {
        for (int t = 0; t < 8; ++t) {
            double *pl[16];
            for (int b = 0; b < 16; ++b)
                pl[b] = S + (size_t)(16 * t + b) * X128;
            xs2e_128(pl, nout, t, 0);
        }
    }
}

static SCHED_ATTR __attribute__((unused, section(".text.gp2exect")))
void fft_exec64t(double *restrict S, const double *nin, double *nout)
{
    for (int b = 0; b < 8; ++b) {
        double *pl[8];
        for (int a = 0; a < 8; ++a)
            pl[a] = S + (size_t)(8 * a + b) * X64;
        zy_tile64_n(pl, 8, nin, b);
        xs1_vert(pl, 512, 3, 7, K64);
    }
    for (int t = 0; t < 8; ++t) {
        double *pl[8];
        for (int b = 0; b < 8; ++b)
            pl[b] = S + (size_t)(8 * t + b) * X64;
        xs2e_64(pl, nout, t);
    }
}

/* natural <-> custody conversions, generic over G (chain-end cold paths) */
static void nat_to_cust_g(const double *restrict nat, double *restrict S,
                          const int L, const int G, const size_t ks, const size_t xs)
{
    for (int x = 0; x < L; ++x)
        for (int y = 0; y < L; ++y) {
            const double *r = nat + ((size_t)x * L + y) * (size_t)L * 2;
            double *q = S + (size_t)x * xs + (size_t)y * ks;
            for (int g = 0; g < G; ++g) {
                v8d A = LDU(r + 16 * g), B = LDU(r + 16 * g + 8);
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
                v8d A = LDU(r + 16 * g), B = LDU(r + 16 * g + 8);
                *(v8d *)(q + (size_t)g * cgs) = DEIN_RE(A, B);
                *(v8d *)(q + (size_t)g * cgs + 8) = DEIN_IM(A, B);
            }
        }
}

static __attribute__((unused)) void cust_to_nat_g(const double *restrict S, double *restrict nat,
                          const int L, const int G, const size_t ks, const size_t xs)
{
    for (int x = 0; x < L; ++x)
        for (int y = 0; y < L; ++y) {
            const double *q = S + (size_t)x * xs + (size_t)y * ks;
            double *r = nat + ((size_t)x * L + y) * (size_t)L * 2;
            for (int g = 0; g < G; ++g) {
                v8d re = *(const v8d *)(q + 16 * g), im = *(const v8d *)(q + 16 * g + 8);
                STU(r + 16 * g, ILV_LO(re, im));
                STU(r + 16 * g + 8, ILV_HI(re, im));
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
                STU(r + 16 * g, ILV_LO(rr, ii));
                STU(r + 16 * g + 8, ILV_HI(rr, ii));
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
    if (L == 32 || L == 64 || L == 16 || L == 128) {
        size_t sd, cd;
        if (L == 32) { fill_fast_tables(); sd = VOLD;   cd = GP2_CT ? CVOLD : VOLD; }
        else if (L == 64) { fill_tables_64(); sd = VOLD64; cd = CVOLD64; }
        else if (L == 128) { fill_tables_128(); sd = VOLD128; cd = CVOLD128; }
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
        /* GP2_HP32 (gen_r7): one 2-MB page for the L=32 S+C block (1.11 MB
         * total, fits a single frame).  Mechanism distinct from the raced-out
         * GP64_HP: the L=32 x-pass's pass 1 touches 32 SCATTERED 4-KB pages
         * per column (XS = 18.5 KB stride, 145 state pages cycled by 128
         * columns against the 64-entry DTLB), a TLB profile neither
         * batchlane's THP-null (sequential streams) nor the L=64 loss
         * (L3-bandwidth-bound) ever tested.  One 2-MB frame also maps the
         * whole set physically contiguous, so the odd-line VIRTUAL scatter
         * controls L2 set placement exactly.  RACED in gen_r7: a WASH
         * (+-0.3%, mixed signs, 6 rotated rounds vs ctl) -- the OOO core
         * hides the STLB-hit latency the 4-KB mapping pays, so the TLB
         * theory of the x-pass residual is dead alongside the r5 L2-edge
         * theory.  Default 0 by simplest-wins; knob kept (CLX's smaller
         * STLB may flip it). */
#ifndef GP2_HP32
# define GP2_HP32 0
#endif
        /* GP128_HP=1 (gen_r12, adopted from gen_layout/gen_powp gen_r11's
         * THP finding: madvise-mode host => posix_memalign gets ZERO huge
         * pages): huge-back the 68-MB L=128 S+C block, which streams through
         * ~17K 4-KB pages every step in the DRAM regime.  RACED on a80n0:
         * -0.8..-0.9% in every clean-window (sd<=0.06%) pair, 6/7 overall;
         * setup +13 ms (touch), trivially in budget.  The r4 GP64_HP loss
         * (page-color flattening at the L3 edge) does NOT transfer to the
         * DRAM regime -- TLB reach wins there.  Bit-identical output. */
#ifndef GP128_HP
# define GP128_HP 1
#endif
        const int hp = (GP64_HP && L == 64) || (GP2_HP32 && L == 32)
                    || (GP128_HP && L == 128);
        size_t bytes = (sd + cd + 1024) * sizeof(double);
        size_t align = hp ? (size_t)2 << 20 : 64;
        if (posix_memalign(&p->blk, align, bytes) != 0) { free(p); return NULL; }
#if defined(__linux__) && defined(MADV_HUGEPAGE)
        if (hp) {
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
#if GP2_XFE
            fft_exec32(p->S, (const double *)in + (size_t)b * vd,
                       (double *)out + (size_t)b * vd);
#else
            nat_to_cust((const double *)in + (size_t)b * vd, p->S);
            fft_step_plain(p->S, p->S);
            cust_to_nat(p->S, (double *)out + (size_t)b * vd);
#endif
        }
        return;
    }
    if (p->fast == 64) {
        const size_t vd = (size_t)64 * 64 * 64 * 2;
        for (int b = 0; b < p->batch; ++b) {
#if GP2_XFE && GP64_XT
            fft_exec64t(p->S, (const double *)in + (size_t)b * vd,
                        (double *)out + (size_t)b * vd);
#elif GP2_XFE
            fft_exec64(p->S, (const double *)in + (size_t)b * vd,
                       (double *)out + (size_t)b * vd);
#else
            nat_to_cust_g((const double *)in + (size_t)b * vd, p->S, 64, 8, K64, X64);
            fft_step64_plain(p->S, p->S);
            cust_to_nat_g(p->S, (double *)out + (size_t)b * vd, 64, 8, K64, X64);
#endif
        }
        return;
    }
    if (p->fast == 16) {
        const size_t vd = (size_t)16 * 16 * 16 * 2;
        for (int b = 0; b < p->batch; ++b) {
#if GP2_XFE
            fft_exec16(p->S, (const double *)in + (size_t)b * vd,
                       (double *)out + (size_t)b * vd);
#else
            nat_to_cust_g((const double *)in + (size_t)b * vd, p->S, 16, 2, K16, X16);
            fft_step16_plain(p->S, p->S);
            cust_to_nat_g(p->S, (double *)out + (size_t)b * vd, 16, 2, K16, X16);
#endif
        }
        return;
    }
    if (p->fast == 128) {
        const size_t vd = (size_t)128 * 128 * 128 * 2;
        for (int b = 0; b < p->batch; ++b) {
#if GP2_XFE && GP128_XT
            fft_exec128t(p->S, (const double *)in + (size_t)b * vd,
                         (double *)out + (size_t)b * vd);
#elif GP2_XFE
            fft_exec128(p->S, (const double *)in + (size_t)b * vd,
                        (double *)out + (size_t)b * vd);
#else
            nat_to_cust_g((const double *)in + (size_t)b * vd, p->S, 128, 16, K128, X128);
            fft_step128_plain(p->S, p->S);
            cust_to_nat_g(p->S, (double *)out + (size_t)b * vd, 128, 16, K128, X128);
#endif
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
    if (p->fast == 64 || p->fast == 16 || p->fast == 128) {
        const size_t vd = V * 2;
        const int L = p->fast, G = L / 8;
        const size_t ks = (L == 128) ? K128 : (L == 64) ? K64 : K16;
        const size_t xs = (L == 128) ? X128 : (L == 64) ? X64 : X16;
        const size_t cys = (L == 128) ? CYS128 : (L == 64) ? CYS64 : CYS16;
        const size_t cgs = (L == 128) ? CGS128 : (L == 64) ? CGS64 : CGS16;
        for (int b = 0; b < p->batch; ++b) {
            nat_to_cust_g((const double *)x0 + (size_t)b * vd, p->S, L, G, ks, xs);
#if GP64_FUSE
            if (L == 64) {
                /* gen_r11 one-sweep fused chain: prologue does z/y/x-stage-1
                 * of step 1; each fused sweep completes step s (x-stage-2 +
                 * map) and begins step s+1; the final sweep leaves step m's
                 * raw z for the permutation-aware epilogue. */
                nat_to_cust_cf64((const double *)c + (size_t)b * vd, p->C);
                fft_prologue64(p->S);
                int par = 0;
                for (int s = 1; s < m; ++s) { fft_sweep64(p->S, p->C, par, 1); par ^= 1; }
                fft_sweep64(p->S, p->C, par, 0);
                cust_map_to_nat_p64(p->S, p->C, (double *)final_out + (size_t)b * vd, par);
                continue;
            }
#endif
#if GP128_FUSE
            if (L == 128) {
                nat_to_cust_cf128((const double *)c + (size_t)b * vd, p->C);
                fft_prologue128(p->S);
                int pp[128];
                for (int i = 0; i < 128; ++i) pp[i] = i;
                for (int s = 1; s < m; ++s) fft_sweep128(p->S, p->C, pp, 1);
                fft_sweep128(p->S, p->C, pp, 0);
                cust_map_to_nat_p128(p->S, p->C, (double *)final_out + (size_t)b * vd, pp);
                continue;
            }
#endif
            nat_to_cust_c_g((const double *)c + (size_t)b * vd, p->C, L, G, cys, cgs);
#if GP2_PREMAP
            if (L == 64) {
                fft_step64_plain(p->S, p->C);
                for (int s = 1; s < m; ++s) fft_step64_premap(p->S, p->C);
            } else if (L == 128) {
                fft_step128_plain(p->S, p->C);
                for (int s = 1; s < m; ++s) fft_step128_premap(p->S, p->C);
            } else {
                fft_step16_plain(p->S, p->C);
                for (int s = 1; s < m; ++s) fft_step16_premap(p->S, p->C);
            }
#else
            if (L == 64) {
                for (int s = 1; s < m; ++s) fft_step64_xmap(p->S, p->C);
                fft_step64_plain(p->S, p->C);
            } else if (L == 128) {
                for (int s = 1; s < m; ++s) fft_step128_xmap(p->S, p->C);
                fft_step128_plain(p->S, p->C);
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
