/* ===========================================================================
 * L8_batchsimd -- forward, unnormalised, complex-double 3D DFT of a FIXED
 *                 8 x 8 x 8 cube, over a batch of B volumes, out-of-place.
 *
 * TECHNIQUE (round 3)
 *   Split-complex (SoA) straight-line radix-8 codelet applied as DFT_8 (x) I_W
 *   (the "vector terminal" of Franchetti & Puschel): every scalar operation of
 *   the codelet is exactly ONE W-wide vector instruction and no cross-lane
 *   operation appears inside any transform.  Two lane assignments compiled,
 *   one picked by measurement in fft3d_create():
 *
 *     LANEX2 (W == 8 only; the default) lanes = one spatial axis of ONE
 *            volume.  TWO passes through a 9 KiB L1-resident scratch:
 *              pass A, per slow plane (8x): transposing load (deinterleave
 *                AND axis->lanes in one 8x8 network, 48 shuffles), x DFT,
 *                one in-register transpose pair (48 shuffles), y DFT,
 *                16 stores to scratch.
 *              pass B, per y (8x): 16 loads, z DFT (shuffle-free),
 *                interleave (2 copy-free permutes/row), 16 stores to out.
 *            Round 3: this replaces the round-2 THREE-pass LANEX; the fusion
 *            deletes one full scratch round trip (128 loads + 128 stores per
 *            volume).  Structure idea from L8_fusedaxes (its fused phase B,
 *            and its round-1 measurement that the extra L1 round trip of a
 *            3-pass form costs ~2.7% at B=1); the codelet, the transposing
 *            load and the SW bookkeeping are mine.
 *     BATCH  (fallback, and the only W = 4 path) lanes = W consecutive
 *            VOLUMES; all three axis passes shuffle-free; volume-major <->
 *            lane-major repack at the two ends as 8x8 in-register transposes.
 *
 * OPERATION COUNT (per volume, LANEX2)
 *   radix-8 codelet, split complex, FMA-folded: 52 instructions
 *     = 44 add/sub + 8 FMA (Burrus T7.1 / FFTW n1_8's 4 mul + 52 add with the
 *       two (1-+i)/sqrt2 twiddles folded into the last butterfly: 56 -> 52)
 *   3 axes * 64 pencils * 52 = 9984 real FP ops = 1248 vector FP instructions
 *   data movement: 896 shuffle uops (768 pass A + 128 pass B interleaves),
 *   256 vector loads, 256 vector stores.  (Round 2 LANEX: 896 / 384 / 384.)
 *
 *   Port model, scored machine (Gold 5218, Cascade Lake, ONE 512-bit FMA
 *   unit): all 512-bit FP on port 0 (1/cycle), all 512-bit shuffles on port 5
 *   (1/cycle) -> p0 floor 1248 cycles/volume, the 896 shuffles fit under it.
 *   Pass A is 104 FP : 96 shuffles per plane (balanced); pass B is 52 FP :
 *   16 shuffles (p5 slack).
 *
 * MEMORY / BANDWIDTH
 *   B=1: src 8K + dst 8K + scratch 9K = 25 KiB, all L1-resident.
 *   Large batch: floor is 8 KiB read + 8 KiB written per volume.  Ordinary
 *   stores add an 8 KiB RFO per volume, so the final write-out uses
 *   _mm512_stream_pd when in+out exceed the L3 scale; every output store is a
 *   full 64-B-aligned line (Drepper's write-combining condition).  Software
 *   prefetch of the next volume (the L2 streamer stops at 4 KiB page
 *   boundaries and one volume spans two pages); the HINT (t0 vs t1) is a
 *   plan-time tuned choice, because L8_fusedaxes measured t1 >> t0 on
 *   Sapphire Rapids while t0 demonstrably helped on the Cascade Lake node --
 *   the create-time tuner decides on the machine that scores.
 *
 * TUNING / REPORTING
 *   fft3d_create() times every legal (mode, nt, pf) candidate round-robin-
 *   interleaved (min of 7 trials each, 3% hysteresis toward the default) and
 *   the chosen configuration is written into fft3d_description(), so the
 *   monitor can read the node's pick off the leaderboard / raw json
 *   (requested by the panel_r2 VERDICT, cross-cutting item 2).
 *
 * ASSUMPTIONS
 *   * L == 8 only.  in/out 64-byte aligned, distinct, in unmodified.
 *   * batch known at plan time (path choice, NT gate, prefetch clamping).
 *   * No library call of any kind inside fft3d_execute().
 * ===========================================================================*/

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fft3d_api.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

/* -------------------------------------------------------------------------
 * Vector abstraction.  Instantiations:
 *   L8_EMU8  : W = 8 emulated in plain C -- verifies the AVX-512 index logic
 *              and the LANEX2 path on a machine without AVX-512.
 *   AVX-512  : W = 8, the graded path.
 *   AVX2     : W = 4, locally testable (BATCH only).
 *   plain    : W = 1, portable reference (BATCH only).
 * ---------------------------------------------------------------------- */

#if defined(L8_EMU8)
#  define VW 8
#  define EMULATED 1
#elif defined(L8_SCALAR)
#  define VW 1
#elif defined(__AVX512F__)
#  define VW 8
#elif defined(__AVX2__) && defined(__FMA__)
#  define VW 4
#else
#  define VW 1
#endif

/* ---------------- emulated 8-wide (correctness harness only) ------------- */
#if defined(EMULATED)
typedef struct { double d[8]; } vd;
static inline vd vld(const double *p){ vd r; for(int k=0;k<8;k++) r.d[k]=p[k]; return r; }
static inline void vst(double *p, vd a){ for(int k=0;k<8;k++) p[k]=a.d[k]; }
static inline vd vset1(double x){ vd r; for(int k=0;k<8;k++) r.d[k]=x; return r; }
static inline vd vadd(vd a, vd b){ vd r; for(int k=0;k<8;k++) r.d[k]=a.d[k]+b.d[k]; return r; }
static inline vd vsub(vd a, vd b){ vd r; for(int k=0;k<8;k++) r.d[k]=a.d[k]-b.d[k]; return r; }
static inline vd vfma(vd a, vd b, vd c){ vd r; for(int k=0;k<8;k++) r.d[k]=a.d[k]*b.d[k]+c.d[k]; return r; }
static inline vd vfnma(vd a, vd b, vd c){ vd r; for(int k=0;k<8;k++) r.d[k]=c.d[k]-a.d[k]*b.d[k]; return r; }
/* faithful emulation of the intrinsics the shuffle networks use */
static inline vd vunplo(vd a, vd b){ vd r;
    r.d[0]=a.d[0]; r.d[1]=b.d[0]; r.d[2]=a.d[2]; r.d[3]=b.d[2];
    r.d[4]=a.d[4]; r.d[5]=b.d[4]; r.d[6]=a.d[6]; r.d[7]=b.d[6]; return r; }
static inline vd vunphi(vd a, vd b){ vd r;
    r.d[0]=a.d[1]; r.d[1]=b.d[1]; r.d[2]=a.d[3]; r.d[3]=b.d[3];
    r.d[4]=a.d[5]; r.d[5]=b.d[5]; r.d[6]=a.d[7]; r.d[7]=b.d[7]; return r; }
static inline vd vperm2(vd a, const int *idx, vd b){ vd r;
    for(int k=0;k<8;k++){ int j=idx[k]; r.d[k] = (j&8) ? b.d[j&7] : a.d[j&7]; } return r; }
/* _mm512_shuffle_f64x2(a,b,imm): lane0=a[imm&3], lane1=a[(imm>>2)&3],
 *                                lane2=b[(imm>>4)&3], lane3=b[(imm>>6)&3] */
static inline vd vshuf128(vd a, vd b, int imm){ vd r;
    const int s0=imm&3, s1=(imm>>2)&3, s2=(imm>>4)&3, s3=(imm>>6)&3;
    r.d[0]=a.d[2*s0]; r.d[1]=a.d[2*s0+1];
    r.d[2]=a.d[2*s1]; r.d[3]=a.d[2*s1+1];
    r.d[4]=b.d[2*s2]; r.d[5]=b.d[2*s2+1];
    r.d[6]=b.d[2*s3]; r.d[7]=b.d[2*s3+1]; return r; }
#define VLD(p)        vld(p)
#define VST(p,v)      vst((p),(v))
#define VSTNT(p,v)    vst((p),(v))
#define VSET1(x)      vset1(x)
#define VADD(a,b)     vadd((a),(b))
#define VSUB(a,b)     vsub((a),(b))
#define VFMA(a,b,c)   vfma((a),(b),(c))
#define VFNMA(a,b,c)  vfnma((a),(b),(c))
static const int EILVLO[8]  = {0,8,1,9,4,12,5,13};
static const int EILVHI2[8] = {10,2,11,3,14,6,15,7};
#define VUNPLO(a,b)   vunplo((a),(b))
#define VUNPHI(a,b)   vunphi((a),(b))
#define VSH44(a,b)    vshuf128((a),(b),0x44)
#define VSHEE(a,b)    vshuf128((a),(b),0xEE)
#define VSH88(a,b)    vshuf128((a),(b),0x88)
#define VSHDD(a,b)    vshuf128((a),(b),0xDD)
#define VILVLO(a,b)   vperm2((a),EILVLO,(b))
#define VILVHI2(b,a)  vperm2((b),EILVHI2,(a))
#define VFENCE()      do{}while(0)

/* ------------------------------- AVX-512 -------------------------------- */
#elif VW == 8
typedef __m512d vd;
#define VLD(p)        _mm512_load_pd(p)
#define VST(p,v)      _mm512_store_pd((p),(v))
#define VSTNT(p,v)    _mm512_stream_pd((p),(v))
#define VSET1(x)      _mm512_set1_pd(x)
#define VADD(a,b)     _mm512_add_pd((a),(b))
#define VSUB(a,b)     _mm512_sub_pd((a),(b))
#define VFMA(a,b,c)   _mm512_fmadd_pd((a),(b),(c))
#define VFNMA(a,b,c)  _mm512_fnmadd_pd((a),(b),(c))
#define VUNPLO(a,b)   _mm512_unpacklo_pd((a),(b))
#define VUNPHI(a,b)   _mm512_unpackhi_pd((a),(b))
#define VSH44(a,b)    _mm512_shuffle_f64x2((a),(b),0x44)
#define VSHEE(a,b)    _mm512_shuffle_f64x2((a),(b),0xEE)
#define VSH88(a,b)    _mm512_shuffle_f64x2((a),(b),0x88)
#define VSHDD(a,b)    _mm512_shuffle_f64x2((a),(b),0xDD)
/* Interleave, composed with the SW lane permutation the transpose network
 * leaves behind: lane m of the sources holds element SW(m).  Lane order in
 * set_epi64 is (lane7 ... lane0).  The HIGH form takes its operands SWAPPED
 * (imaginary first) with a correspondingly rewritten index vector, so the two
 * permutes destroy DIFFERENT sources and gcc emits no vmovapd copy --
 * borrowed from L8_radix8 round 2 ("copy-free interleave-source swap"). */
#define VILVLO(a,b)   _mm512_permutex2var_pd((a), \
                        _mm512_set_epi64(13,5,12,4,9,1,8,0), (b))
#define VILVHI2(b,a)  _mm512_permutex2var_pd((b), \
                        _mm512_set_epi64(7,15,6,14,3,11,2,10), (a))
#define VFENCE()      _mm_sfence()

/* --------------------------------- AVX2 --------------------------------- */
#elif VW == 4
typedef __m256d vd;
#define VLD(p)        _mm256_load_pd(p)
#define VST(p,v)      _mm256_store_pd((p),(v))
#define VSTNT(p,v)    _mm256_stream_pd((p),(v))
#define VSET1(x)      _mm256_set1_pd(x)
#define VADD(a,b)     _mm256_add_pd((a),(b))
#define VSUB(a,b)     _mm256_sub_pd((a),(b))
#define VFMA(a,b,c)   _mm256_fmadd_pd((a),(b),(c))
#define VFNMA(a,b,c)  _mm256_fnmadd_pd((a),(b),(c))
#define VUNPLO(a,b)   _mm256_unpacklo_pd((a),(b))
#define VUNPHI(a,b)   _mm256_unpackhi_pd((a),(b))
#define VPERM128L(a,b) _mm256_permute2f128_pd((a),(b),0x20)
#define VPERM128H(a,b) _mm256_permute2f128_pd((a),(b),0x31)
#define VFENCE()      _mm_sfence()

/* ------------------------------- scalar --------------------------------- */
#else
typedef double vd;
#define VLD(p)        (*(p))
#define VST(p,v)      (*(p) = (v))
#define VSTNT(p,v)    (*(p) = (v))
#define VSET1(x)      (x)
#define VADD(a,b)     ((a)+(b))
#define VSUB(a,b)     ((a)-(b))
#define VFMA(a,b,c)   ((a)*(b)+(c))
#define VFNMA(a,b,c)  ((c)-(a)*(b))
#define VFENCE()      do{}while(0)
#endif

#define AI __attribute__((always_inline)) inline

/* Software prefetch of the next volume.  The L2 streamer stops at 4 KiB page
 * boundaries and one 8^3 volume spans two pages, so in the streaming regime
 * the hardware prefetcher restarts twice per volume.  The hint is a tuned
 * plan-time choice: L8_fusedaxes measured t1 far ahead of t0 on Sapphire
 * Rapids (t0 fights the NT fill buffers there) while t0 helped on the
 * Cascade Lake node in round 2 -- so both are compiled and the create-time
 * tuner picks on the machine that scores. */
#if defined(__x86_64__) && !defined(EMULATED)
#  define PF0(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)
#  define PF1(p) _mm_prefetch((const char *)(p), _MM_HINT_T1)
#else
#  define PF0(p) ((void)(p))
#  define PF1(p) ((void)(p))
#endif

enum { PF_NONE = 0, PF_T0 = 1, PF_T1 = 2 };

/* -------------------------------------------------------------------------
 * VW x VW in-register double transpose, up to a fixed lane permutation:
 *
 *     out[k][l] = in[SW(l)][k],   SW = swap of lane bits 1 and 2 at W = 8
 *                                 (0,1,4,5,2,3,6,7), identity at W = 4.
 *
 * W = 8 network (24 shuffle uops, ALL two-source non-destructive forms with
 * immediate control -- no index vectors, no vpermt2pd, no register copies).
 * Borrowed from L8_fusedaxes round 1: a straight r1<->l1 middle level has no
 * non-destructive AVX-512 encoding, but the 3-cycle r1 -> l2 -> l1 -> r1
 * (imm 0x88/0xDD) is encodable, and composing
 *   stage A  r2 <-> l2          (vshuff64x2 0x44 / 0xEE)
 *   stage B  r1 -> l2 -> l1     (vshuff64x2 0x88 / 0xDD)
 *   stage C  r0 <-> l0          (vunpcklo / vunpckhi)
 * yields a transpose whose only residue is the lane permutation SW, absorbed
 * by every call site as a compile-time relabel.
 * ---------------------------------------------------------------------- */
#if VW == 8
#  define SW(l) ((((l) & 1)) | (((l) & 2) << 1) | (((l) & 4) >> 1))
#else
#  define SW(l) (l)
#endif

static AI void vtrans(vd *restrict o, const vd *restrict i)
{
#if VW == 8
    vd u0 = VSH44(i[0], i[4]), u4 = VSHEE(i[0], i[4]);
    vd u1 = VSH44(i[1], i[5]), u5 = VSHEE(i[1], i[5]);
    vd u2 = VSH44(i[2], i[6]), u6 = VSHEE(i[2], i[6]);
    vd u3 = VSH44(i[3], i[7]), u7 = VSHEE(i[3], i[7]);
    vd w0 = VSH88(u0, u2), w2 = VSHDD(u0, u2);
    vd w1 = VSH88(u1, u3), w3 = VSHDD(u1, u3);
    vd w4 = VSH88(u4, u6), w6 = VSHDD(u4, u6);
    vd w5 = VSH88(u5, u7), w7 = VSHDD(u5, u7);
    o[0] = VUNPLO(w0, w1); o[1] = VUNPHI(w0, w1);
    o[2] = VUNPLO(w2, w3); o[3] = VUNPHI(w2, w3);
    o[4] = VUNPLO(w4, w5); o[5] = VUNPHI(w4, w5);
    o[6] = VUNPLO(w6, w7); o[7] = VUNPHI(w6, w7);
#elif VW == 4
    vd t0 = VUNPLO(i[0], i[1]), t1 = VUNPHI(i[0], i[1]);
    vd t2 = VUNPLO(i[2], i[3]), t3 = VUNPHI(i[2], i[3]);
    o[0] = VPERM128L(t0, t2); o[1] = VPERM128L(t1, t3);
    o[2] = VPERM128H(t0, t2); o[3] = VPERM128H(t1, t3);
#else
    o[0] = i[0];
#endif
}

/* -------------------------------------------------------------------------
 * The radix-8 codelet, split complex, in place on r[0..7] / m[0..7].
 *
 *   X_k = sum_j x_j W^{jk},  W = exp(-2 pi i / 8)
 * decimated even/odd:  X_k = E_k + W^k O_k,  X_{k+4} = E_k - W^k O_k
 * with E = DFT4(x0,x2,x4,x6), O = DFT4(x1,x3,x5,x7).
 * W^0 = 1, W^1 = c(1-i), W^2 = -i, W^3 = -c(1+i), c = 1/sqrt(2).
 * +-i is a rename plus a sign folded into the neighbouring add/sub, hence
 * free in split layout; the two c-twiddles fold into the last butterfly as
 * fmadd/fnmadd.  52 instructions: 44 add/sub + 8 FMA.
 * ---------------------------------------------------------------------- */
static AI void r8(vd *restrict r, vd *restrict m)
{
    const vd C = VSET1(0.70710678118654752440084436210485);

    /* DFT4 on the even-indexed inputs */
    vd a0r = VADD(r[0], r[4]), a0i = VADD(m[0], m[4]);
    vd a2r = VSUB(r[0], r[4]), a2i = VSUB(m[0], m[4]);
    vd a1r = VADD(r[2], r[6]), a1i = VADD(m[2], m[6]);
    vd a3r = VSUB(r[2], r[6]), a3i = VSUB(m[2], m[6]);
    vd E0r = VADD(a0r, a1r), E0i = VADD(a0i, a1i);
    vd E2r = VSUB(a0r, a1r), E2i = VSUB(a0i, a1i);
    vd E1r = VADD(a2r, a3i), E1i = VSUB(a2i, a3r);
    vd E3r = VSUB(a2r, a3i), E3i = VADD(a2i, a3r);

    /* DFT4 on the odd-indexed inputs */
    vd b0r = VADD(r[1], r[5]), b0i = VADD(m[1], m[5]);
    vd b2r = VSUB(r[1], r[5]), b2i = VSUB(m[1], m[5]);
    vd b1r = VADD(r[3], r[7]), b1i = VADD(m[3], m[7]);
    vd b3r = VSUB(r[3], r[7]), b3i = VSUB(m[3], m[7]);
    vd O0r = VADD(b0r, b1r), O0i = VADD(b0i, b1i);
    vd O2r = VSUB(b0r, b1r), O2i = VSUB(b0i, b1i);
    vd O1r = VADD(b2r, b3i), O1i = VSUB(b2i, b3r);
    vd O3r = VSUB(b2r, b3i), O3i = VADD(b2i, b3r);

    /* twiddle-and-combine */
    vd s1 = VADD(O1r, O1i), d1 = VSUB(O1i, O1r);
    vd s3 = VADD(O3r, O3i), d3 = VSUB(O3i, O3r);

    r[0] = VADD(E0r, O0r); m[0] = VADD(E0i, O0i);
    r[4] = VSUB(E0r, O0r); m[4] = VSUB(E0i, O0i);
    r[2] = VADD(E2r, O2i); m[2] = VSUB(E2i, O2r);
    r[6] = VSUB(E2r, O2i); m[6] = VADD(E2i, O2r);
    r[1] = VFMA (C, s1, E1r); m[1] = VFMA (C, d1, E1i);
    r[5] = VFNMA(C, s1, E1r); m[5] = VFNMA(C, d1, E1i);
    r[3] = VFMA (C, d3, E3r); m[3] = VFNMA(C, s3, E3i);
    r[7] = VFNMA(C, d3, E3r); m[7] = VFMA (C, s3, E3i);
}

/* ---------------------------- geometry ---------------------------------- */
#define VOLD   1024            /* doubles per volume (8^3 complex)          */
#define ZSTR    128            /* doubles between consecutive slow planes   */
#define YSTR     16            /* doubles between consecutive middle rows   */
#define NGRP   (16 / VW)       /* vectors per interleaved 8-complex row     */

/* BATCH scratch: index ((x*9 + z)*9 + y), vectors of VW doubles.
 * strides in 64-B lines at VW=8:  y 1, z 9, x 81 -- all odd.  */
#define BPY 9
#define BPZ 9
#define BSLOT (8 * BPZ * BPY)          /* 648 vectors per component */

/* LANEX2 scratch: index (y*9 + s), vectors of 8 doubles whose lanes are the
 * contiguous axis.  s stride 1 line, y stride 9 lines -- both odd, so all
 * L1 sets are used.  9 KiB per component pair total. */
#define LPZ 9
#define LSLOT (8 * LPZ)                /* 72 vectors per component */

enum { MODE_BATCH = 0, MODE_LANEX = 1 };

struct fft3d_plan {
    int    batch;
    int    mode;
    int    nt;
    int    pf;          /* PF_NONE / PF_T0 / PF_T1                         */
    double *scr;        /* 64-B aligned working set                        */
    double *stage_in;   /* VW zero-padded volumes for the B % VW tail      */
    double *stage_out;
    void   *raw;
};

/* =========================================================================
 * BATCH path: lanes = VW consecutive volumes.  (Unchanged since round 2;
 * kept as the W = 4 path and as a measured fallback.)
 * ======================================================================= */
static AI void batch_block(double *restrict scr,
                           const double *restrict src,
                           double *restrict dst,
                           const int nt)
{
    double *restrict SR = scr;
    double *restrict SI = scr + (size_t)BSLOT * VW;

/* one z-pass codelet at (x,y): 52 port-0 instructions, zero shuffles */
#define ZPASS(X, Y) do {                                                      \
    double *restrict br_ = SR + (size_t)(X) * BPZ * BPY * VW;                 \
    double *restrict bi_ = SI + (size_t)(X) * BPZ * BPY * VW;                 \
    vd r_[8], m_[8];                                                          \
    for (int z_ = 0; z_ < 8; ++z_) {                                          \
        const size_t o_ = (size_t)(z_ * BPY + (Y)) * VW;                      \
        r_[z_] = VLD(br_ + o_); m_[z_] = VLD(bi_ + o_);                       \
    }                                                                         \
    r8(r_, m_);                                                               \
    for (int z_ = 0; z_ < 8; ++z_) {                                          \
        const size_t o_ = (size_t)(z_ * BPY + (Y)) * VW;                      \
        VST(br_ + o_, r_[z_]); VST(bi_ + o_, m_[z_]);                         \
    }                                                                         \
} while (0)

/* lane-major -> volume-major for one (y,z): 48 port-5 shuffles, zero FP */
#define TSTORE(Y, Z) do {                                                     \
    double *row_ = dst + (Z) * ZSTR + (Y) * YSTR;                             \
    vd xr_[8], xi_[8];                                                        \
    for (int x_ = 0; x_ < 8; ++x_) {                                          \
        const size_t o_ = (((size_t)x_ * BPZ + (Z)) * BPY + (Y)) * VW;        \
        xr_[x_] = VLD(SR + o_); xi_[x_] = VLD(SI + o_);                       \
    }                                                                         \
    for (int g_ = 0; g_ < NGRP; ++g_) {                                       \
        vd tin_[VW], tout_[VW];                                               \
        for (int q_ = 0; q_ < VW; ++q_) {                                     \
            /* feed doubles pre-permuted by SW: the residue cancels */        \
            const int j_ = g_ * VW + SW(q_);                                  \
            tin_[q_] = (j_ & 1) ? xi_[j_ >> 1] : xr_[j_ >> 1];                \
        }                                                                     \
        vtrans(tout_, tin_);                                                  \
        for (int b_ = 0; b_ < VW; ++b_) {                                     \
            double *p_ = row_ + (size_t)SW(b_) * VOLD + g_ * VW;              \
            if (nt) VSTNT(p_, tout_[b_]);                                     \
            else    VST  (p_, tout_[b_]);                                     \
        }                                                                     \
    }                                                                         \
} while (0)

/* one y-pass codelet at (x,z): 52 port-0 instructions, zero shuffles */
#define YPASS(X, Z) do {                                                      \
    double *restrict cr_ = SR + (size_t)(X) * BPZ * BPY * VW;                 \
    double *restrict ci_ = SI + (size_t)(X) * BPZ * BPY * VW;                 \
    vd r_[8], m_[8];                                                          \
    for (int y_ = 0; y_ < 8; ++y_) {                                          \
        const size_t o_ = (size_t)((Z) * BPY + y_) * VW;                      \
        r_[y_] = VLD(cr_ + o_); m_[y_] = VLD(ci_ + o_);                       \
    }                                                                         \
    r8(r_, m_);                                                               \
    for (int y_ = 0; y_ < 8; ++y_) {                                          \
        const size_t o_ = (size_t)((Z) * BPY + y_) * VW;                      \
        VST(cr_ + o_, r_[y_]); VST(ci_ + o_, m_[y_]);                         \
    }                                                                         \
} while (0)

    /* ---- phase 1: transposing load (volume-major -> lane-major) + x pass,
     * with the y pass fused per z-plane (the 8 KiB z-plane of scratch is
     * y-transformed while L1-hot).                                          */
    for (int z = 0; z < 8; ++z) {
        if (nt) {                       /* one block of lookahead */
            for (int b = 0; b < VW; ++b) {
                const double *nx = src + (size_t)VW * VOLD + (size_t)b * VOLD
                                       + z * ZSTR;
                PF0(nx); PF0(nx + 8); PF0(nx + 16); PF0(nx + 24);
                PF0(nx + 32); PF0(nx + 40); PF0(nx + 48); PF0(nx + 56);
                PF0(nx + 64); PF0(nx + 72); PF0(nx + 80); PF0(nx + 88);
                PF0(nx + 96); PF0(nx + 104); PF0(nx + 112); PF0(nx + 120);
            }
        }
        for (int y = 0; y < 8; ++y) {
            const double *row = src + z * ZSTR + y * YSTR;
            vd xr[8], xi[8];
            for (int g = 0; g < NGRP; ++g) {
                vd tin[VW], tout[VW];
                for (int b = 0; b < VW; ++b)
                    tin[b] = VLD(row + (size_t)b * VOLD + g * VW);
                vtrans(tout, tin);
                for (int q = 0; q < VW; ++q) {
                    const int j = g * VW + q;      /* double index in the row */
                    if (j & 1) xi[j >> 1] = tout[q];
                    else       xr[j >> 1] = tout[q];
                }
            }
            r8(xr, xi);
            for (int x = 0; x < 8; ++x) {
                const size_t o = (((size_t)x * BPZ + z) * BPY + y) * VW;
                VST(SR + o, xr[x]);
                VST(SI + o, xi[x]);
            }
        }
        /* ---- fused phase 2a: y pass over this z-plane while it is hot   */
        for (int x = 0; x < 8; ++x) YPASS(x, z);
    }

    /* ---- phase 2b: z pass, software-pipelined against the transposing
     * store of the previous y: the z codelet is 52 port-0 instructions with
     * no shuffles, the transposing store 48 port-5 shuffles with no
     * arithmetic; emitted adjacent and independent they hide in each other. */
    for (int x = 0; x < 8; ++x) ZPASS(x, 0);
    for (int y = 1; y < 8; ++y)
        for (int x = 0; x < 8; ++x) { ZPASS(x, y); TSTORE(y - 1, x); }
    for (int z = 0; z < 8; ++z) TSTORE(7, z);

}

static void batch_run(double *scr, const double *src, double *dst, long nblk, int nt)
{
    if (nt) {
        for (long k = 0; k < nblk; ++k)
            batch_block(scr, src + (size_t)k * VW * VOLD,
                             dst + (size_t)k * VW * VOLD, 1);
    } else {
        for (long k = 0; k < nblk; ++k)
            batch_block(scr, src + (size_t)k * VW * VOLD,
                             dst + (size_t)k * VW * VOLD, 0);
    }
}

/* =========================================================================
 * LANEX2 path (VW == 8 only): lanes = one spatial axis, TWO passes.
 *
 *   pass A, per slow plane s:
 *     transposing load (deinterleave AND contiguous-axis->registers in one
 *     8x8 network) -> xr/xi[x], lanes = y (up to SW);  x DFT;
 *     one in-register transpose pair -> yr/yi[y], lanes = x_spec (up to SW);
 *     y DFT;  16 stores to the 9 KiB scratch at (y*9 + s).
 *   pass B, per y:
 *     16 loads (s = 0..7 contiguous lines);  slow-axis DFT (shuffle-free);
 *     copy-free interleave (VILVLO/VILVHI2, SW-composed indices);  16 stores
 *     straight into the driver's interleaved layout.
 *
 * Per volume: 1248 FP, 896 shuffles (768 A + 128 B), 256 loads, 256 stores.
 * The round-2 three-pass form spent 384 loads + 384 stores; the fusion
 * deletes that scratch round trip (structure per L8_fusedaxes).
 * ======================================================================= */
#if VW == 8
static AI void lanex2_volume(double *restrict scr,
                             const double *restrict src,
                             double *restrict dst,
                             const int nt, const int pf)
{
    double *restrict SR = scr;
    double *restrict SI = scr + (size_t)LSLOT * 8;

    /* ---- pass A ---- */
    for (int s = 0; s < 8; ++s) {
        if (pf == PF_T0) {              /* one volume of lookahead */
            const double *nx = src + VOLD + s * ZSTR;
            PF0(nx); PF0(nx + 8); PF0(nx + 16); PF0(nx + 24);
            PF0(nx + 32); PF0(nx + 40); PF0(nx + 48); PF0(nx + 56);
            PF0(nx + 64); PF0(nx + 72); PF0(nx + 80); PF0(nx + 88);
            PF0(nx + 96); PF0(nx + 104); PF0(nx + 112); PF0(nx + 120);
        } else if (pf == PF_T1) {
            const double *nx = src + VOLD + s * ZSTR;
            PF1(nx); PF1(nx + 8); PF1(nx + 16); PF1(nx + 24);
            PF1(nx + 32); PF1(nx + 40); PF1(nx + 48); PF1(nx + 56);
            PF1(nx + 64); PF1(nx + 72); PF1(nx + 80); PF1(nx + 88);
            PF1(nx + 96); PF1(nx + 104); PF1(nx + 112); PF1(nx + 120);
        }
        vd xr[8], xi[8];
        for (int g = 0; g < 2; ++g) {
            vd tin[8], tout[8];
            for (int y = 0; y < 8; ++y)
                tin[y] = VLD(src + s * ZSTR + y * YSTR + g * 8);
            vtrans(tout, tin);      /* tout[j] = double j of the row, lanes=y.SW */
            for (int q = 0; q < 8; ++q) {
                const int j = g * 8 + q;
                if (j & 1) xi[j >> 1] = tout[q];
                else       xr[j >> 1] = tout[q];
            }
        }
        r8(xr, xi);                 /* x DFT: registers = x_spec, lanes = y.SW */

        /* transpose x_spec <-> y; the SW residue is absorbed by the rename
         * yr[SW(k)] = u[k] (free -- register renaming) */
        vd yr[8], yi[8], u[8];
        vtrans(u, xr);
        for (int k = 0; k < 8; ++k) yr[SW(k)] = u[k];
        vtrans(u, xi);
        for (int k = 0; k < 8; ++k) yi[SW(k)] = u[k];

        r8(yr, yi);                 /* y DFT: registers = y_spec, lanes = x_spec.SW */

        for (int m = 0; m < 8; ++m) {
            const size_t o = (size_t)(m * LPZ + s) * 8;
            VST(SR + o, yr[m]); VST(SI + o, yi[m]);
        }
    }

    /* ---- pass B ---- */
    for (int m = 0; m < 8; ++m) {
        vd r[8], q[8];
        for (int s = 0; s < 8; ++s) {
            const size_t o = (size_t)(m * LPZ + s) * 8;
            r[s] = VLD(SR + o); q[s] = VLD(SI + o);
        }
        r8(r, q);                   /* slow-axis DFT, shuffle-free */
        for (int s = 0; s < 8; ++s) {
            double *p = dst + s * ZSTR + m * YSTR;
            vd lo = VILVLO(r[s], q[s]);
            vd hi = VILVHI2(q[s], r[s]);
            if (nt) { VSTNT(p, lo); VSTNT(p + 8, hi); }
            else    { VST  (p, lo); VST  (p + 8, hi); }
        }
    }
}

/* One specialised runner per (nt, pf) combination so every branch inside
 * lanex2_volume folds away.  The last volume always runs pf = PF_NONE so no
 * prefetch ever reaches past the end of the input mapping. */
#define LANEX_RUN(NAME, NT, PF)                                               \
static void NAME(double *scr, const double *src, double *dst, long nvol)      \
{                                                                             \
    for (long v = 0; v + 1 < nvol; ++v)                                       \
        lanex2_volume(scr, src + (size_t)v * VOLD,                            \
                           dst + (size_t)v * VOLD, NT, PF);                   \
    lanex2_volume(scr, src + (size_t)(nvol - 1) * VOLD,                       \
                       dst + (size_t)(nvol - 1) * VOLD, NT, PF_NONE);         \
}
LANEX_RUN(lanex_run_p0,    0, PF_NONE)
LANEX_RUN(lanex_run_pt0,   0, PF_T0)
LANEX_RUN(lanex_run_n_p0,  1, PF_NONE)
LANEX_RUN(lanex_run_n_pt0, 1, PF_T0)
LANEX_RUN(lanex_run_n_pt1, 1, PF_T1)
#undef LANEX_RUN
#endif /* VW == 8 */

/* ========================== the API ==================================== */

const char *fft3d_name(void) { return "L8_batchsimd"; }

/* Filled in by fft3d_create() with the tuner's actual pick, so the node's
 * per-case choice is readable off the results (VERDICT panel_r2 request). */
static char g_desc[192] =
    "split-complex radix-8, 2-pass fused lanex (x+y | z), copy-free "
    "interleave; untuned";

const char *fft3d_description(void) { return g_desc; }

int fft3d_supports(int L) { return L == 8; }

/* --- driver-independent execution used by both the plan and the tuner --- */
static void run_all(const struct fft3d_plan *p,
                    const double *src, double *dst)
{
    const long nvol = p->batch;
#if VW == 8
    if (p->mode == MODE_LANEX) {
        if (p->nt) {
            switch (p->pf) {
            case PF_T0: lanex_run_n_pt0(p->scr, src, dst, nvol); break;
            case PF_T1: lanex_run_n_pt1(p->scr, src, dst, nvol); break;
            default:    lanex_run_n_p0 (p->scr, src, dst, nvol); break;
            }
            VFENCE();
        } else {
            if (p->pf == PF_T0) lanex_run_pt0(p->scr, src, dst, nvol);
            else                lanex_run_p0 (p->scr, src, dst, nvol);
        }
        return;
    }
#endif
    {
        const long nblk = nvol / VW;
        const long tail = nvol - nblk * VW;
        batch_run(p->scr, src, dst, nblk, p->nt);
        if (tail) {
            const double *ti = src + (size_t)nblk * VW * VOLD;
            double *to = dst + (size_t)nblk * VW * VOLD;
            memcpy(p->stage_in, ti, (size_t)tail * VOLD * sizeof(double));
            batch_run(p->scr, p->stage_in, p->stage_out, 1, 0);
            memcpy(to, p->stage_out, (size_t)tail * VOLD * sizeof(double));
        }
        if (p->nt) VFENCE();
    }
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in,
                   double _Complex *out)
{
    run_all(plan, (const double *)in, (double *)out);
}

/* ------------------------------ setup ---------------------------------- */

static double wall(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Self-tune: time every legal (mode, nt, pf) combination on a surrogate
 * batch big enough to reproduce the real cache regime, and keep the best.
 * Trials are round-robin-interleaved across candidates (min of 7 each) so a
 * frequency ramp or licence transition biases everyone equally (idea from
 * L8_radix8 round 1); 3% hysteresis toward the default so noise cannot flip
 * the plan for nothing.  All inside fft3d_create(), outside the score. */
static const char *mode_str(int m) { return m == MODE_LANEX ? "LANEX" : "BATCH"; }
static const char *pf_str(int pf)
{
    return pf == PF_T0 ? "t0" : pf == PF_T1 ? "t1" : "none";
}

static void autotune(struct fft3d_plan *p)
{
    /* Surrogate: cap raised to 4096 volumes (64 MiB working set) so the
     * B=16384 regime is tuned on a clearly-streaming arena on the node
     * (22 MiB L3), not a marginal 1.5x-L3 one. */
    long nsur = p->batch < 4096 ? p->batch : 4096;
    if (nsur < 1) nsur = 1;

    size_t bytes = (size_t)nsur * VOLD * sizeof(double);
    double *ti = NULL, *to = NULL;
    if (posix_memalign((void **)&ti, 64, bytes) != 0 || !ti) return;
    if (posix_memalign((void **)&to, 64, bytes) != 0 || !to) { free(ti); return; }
    for (size_t k = 0; k < (size_t)nsur * VOLD; ++k)
        ti[k] = 1.0 + 0.5 * (double)(k & 63);
    memset(to, 0, bytes);

    struct fft3d_plan t = *p;
    t.batch = (int)nsur;

    /* Candidate list.  NT/prefetch candidates only once the working set is
     * at L3 scale; prefetch-without-NT only above one volume (a next-volume
     * prefetch at B=1 has nothing to fetch). */
    const size_t ws = (size_t)p->batch * VOLD * sizeof(double) * 2;
    const int big = ws > ((size_t)6 << 20);
    int cmode[8], cnt[8], cpf[8], ncand = 0;
#if VW == 8
    cmode[ncand] = MODE_LANEX; cnt[ncand] = 0; cpf[ncand] = PF_NONE; ++ncand;
    if (p->batch > 1) {
        cmode[ncand] = MODE_LANEX; cnt[ncand] = 0; cpf[ncand] = PF_T0; ++ncand;
    }
    if (big) {
        cmode[ncand] = MODE_LANEX; cnt[ncand] = 1; cpf[ncand] = PF_NONE; ++ncand;
        cmode[ncand] = MODE_LANEX; cnt[ncand] = 1; cpf[ncand] = PF_T0;   ++ncand;
        cmode[ncand] = MODE_LANEX; cnt[ncand] = 1; cpf[ncand] = PF_T1;   ++ncand;
    }
#endif
    /* BATCH is legal at any B (remainder via the zero-padded staging block);
     * its prefetch is compiled in with nt, so cpf is informational only. */
    cmode[ncand] = MODE_BATCH; cnt[ncand] = 0; cpf[ncand] = PF_NONE; ++ncand;
    if (big) {
        cmode[ncand] = MODE_BATCH; cnt[ncand] = 1; cpf[ncand] = PF_T0; ++ncand;
    }

    /* At least ~2 ms of work per timing block so the clock is not the
     * measurement. */
    long reps = 1;
    {
        double t0 = wall();
        run_all(&t, ti, to);
        double dt = wall() - t0;
        if (dt > 0) { reps = (long)(0.002 / dt); }
        if (reps < 1) reps = 1;
        if (reps > 20000) reps = 20000;
    }

    double bt[8];
    for (int c = 0; c < ncand; ++c) {
        bt[c] = 1e30;
        t.mode = cmode[c]; t.nt = cnt[c]; t.pf = cpf[c];
        run_all(&t, ti, to);                /* warm each path once */
    }
    for (int trial = 0; trial < 7; ++trial) {
        for (int c = 0; c < ncand; ++c) {
            t.mode = cmode[c]; t.nt = cnt[c]; t.pf = cpf[c];
            double t0 = wall();
            for (long q = 0; q < reps; ++q) run_all(&t, ti, to);
            double dt = (wall() - t0) / (double)reps;
            if (dt < bt[c]) bt[c] = dt;
        }
    }

    int best_mode = p->mode, best_nt = p->nt, best_pf = p->pf;
    double best = 1e30;
    for (int c = 0; c < ncand; ++c) {
        const int incumbent = (cmode[c] == p->mode && cnt[c] == p->nt &&
                               cpf[c] == p->pf);
        const double score = bt[c] * (incumbent ? 1.0 : 1.03);
        if (score < best) {
            best = score;
            best_mode = cmode[c]; best_nt = cnt[c]; best_pf = cpf[c];
        }
    }

    p->mode = best_mode;
    p->nt = best_nt;
    p->pf = best_pf;

    if (getenv("L8_TUNE_DEBUG")) {
        fprintf(stderr, "[L8_batchsimd tune] batch=%d nsur=%ld reps=%ld ->"
                        " mode=%s nt=%d pf=%s |", p->batch, nsur, reps,
                mode_str(best_mode), best_nt, pf_str(best_pf));
        for (int c = 0; c < ncand; ++c)
            fprintf(stderr, " %s/nt%d/%s=%.3fus",
                    mode_str(cmode[c]), cnt[c], pf_str(cpf[c]),
                    1e6 * bt[c] / (double)nsur);
        fprintf(stderr, "\n");
    }
    free(ti);
    free(to);
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != 8 || batch < 1) return NULL;

    struct fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->batch = batch;

    /* one arena: BATCH scratch | LANEX2 scratch | tail staging in/out */
    const size_t nb_batch = (size_t)2 * BSLOT * VW;
    const size_t nb_lanex = (size_t)2 * LSLOT * 8;
    const size_t nb_stage = (size_t)2 * VW * VOLD;
    const size_t total = nb_batch + nb_lanex + nb_stage + 64;

    double *arena = NULL;
    if (posix_memalign((void **)&arena, 64, total * sizeof(double)) != 0 || !arena) {
        free(p);
        return NULL;
    }
    memset(arena, 0, total * sizeof(double));
    p->raw = arena;
    p->scr = arena;                                /* both paths share it */
    p->stage_in = arena + nb_batch + nb_lanex;
    p->stage_out = p->stage_in + (size_t)VW * VOLD;

    /* Defaults, which the autotune only overrides by a clear margin:
     * LANEX2 everywhere at VW = 8; NT past the L3 scale; next-volume
     * prefetch t0 whenever there is a next volume (node-verified in r2). */
#if VW == 8
    p->mode = MODE_LANEX;
#else
    p->mode = MODE_BATCH;
#endif
    p->nt = ((size_t)batch * VOLD * sizeof(double) * 2 > (size_t)12 << 20);
    p->pf = (batch > 1) ? PF_T0 : PF_NONE;

#if defined(L8_FORCE_MODE)
    p->mode = L8_FORCE_MODE;
#  if defined(L8_FORCE_NT)
    p->nt = L8_FORCE_NT;
#  endif
#  if defined(L8_FORCE_PF)
    p->pf = L8_FORCE_PF;
#  endif
#else
    autotune(p);
#endif

    snprintf(g_desc, sizeof g_desc,
             "2-pass fused lanex (x+y|z) radix-8 split; pick[B=%d]: "
             "mode=%s nt=%d pf=%s",
             batch, mode_str(p->mode), p->nt, pf_str(p->pf));
    return p;
}

void fft3d_destroy(fft3d_plan *plan)
{
    if (!plan) return;
    free(plan->raw);
    free(plan);
}
