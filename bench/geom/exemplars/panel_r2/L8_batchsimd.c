/* ===========================================================================
 * L8_batchsimd -- forward, unnormalised, complex-double 3D DFT of a FIXED
 *                 8 x 8 x 8 cube, over a batch of B volumes, out-of-place.
 *
 * TECHNIQUE
 *   Split-complex (SoA) straight-line radix-8 codelet applied as DFT_8 (x) I_W,
 *   i.e. the "vector terminal" of Franchetti & Puschel: every scalar operation
 *   of the codelet becomes exactly ONE W-wide vector instruction and there is
 *   not a single cross-lane operation anywhere inside the transform.  Two lane
 *   assignments are compiled and one is picked by measurement in
 *   fft3d_create():
 *
 *     LANEX  (W == 8 only; round-2 DEFAULT for every batch size) lanes = the
 *            8 x-values of ONE volume.  Needs no batch at all.  The y and z
 *            passes are free, the de/interleave at the ends costs 1 shuffle
 *            per 64 B, and ONE in-register 8x8 transpose per z-plane makes
 *            the x pass free.  9 KiB scratch, L1-resident.
 *     BATCH  (round-1 default; fallback, and the only W = 4 path) lanes = W
 *            consecutive VOLUMES.  All three axis passes are shuffle-free;
 *            the only data reorganisation is the volume-major(interleaved)
 *            <-> lane-major(split) repack at the two ends, done as an 8x8
 *            (4x4 on AVX2) in-register double transpose.
 *
 *   Round 2: every 8x8 transpose uses the NON-DESTRUCTIVE network borrowed
 *   from L8_fusedaxes (vshuff64x2 imm 0x44/0xEE, 0x88/0xDD, then vunpck),
 *   which needs no vpermt2pd, no index vectors and no register copies; its
 *   residual lane permutation SW = swap(lane bits 1,2) is absorbed into
 *   compile-time relabels at the load/store ends.  On wallaby this took
 *   LANEX from 0.63 to 0.31 us/volume and made it the winner in every
 *   regime, which is why it is now the default.
 *
 * OPERATION COUNT  (per volume; identical for both paths)
 *   radix-8 codelet, split complex, FMA-folded:   52 instructions
 *       = 44 add/sub + 8 FMA   (the textbook 4 real mul + 52 real add of
 *         Burrus T7.1 / T9.1 / FFTW n1_8, with the two (1-+i)/sqrt2 twiddles
 *         folded into the final butterfly as fmadd/fnmadd, which removes 4
 *         instructions:  52 instead of 56)
 *   3 axes * 64 pencils * 52          =  9984 real FP ops per volume
 *   at W = 8                          =  1248 vector FP instructions
 *   data movement per volume:
 *       BATCH   768 shuffle uops, 512 vector loads, 512 vector stores
 *       LANEX   896 shuffle uops, 384 vector loads, 384 vector stores
 *
 *   Port model for the scored machine (Xeon Gold 5218, Cascade Lake, ONE
 *   512-bit FMA unit): 512-bit FP issues on port 0 only (1/cycle) while all
 *   512-bit shuffles issue on port 5 (1/cycle), so 9984 real ops / 8 doubles
 *   per cycle = 1248 cycles/volume is the floor and the 768 shuffles fit
 *   underneath it -- PROVIDED they sit in the same loop bodies as the
 *   arithmetic.  That is why phase 2b is software-pipelined: a separate
 *   shuffle-only output pass leaves port 0 idle for 48 cycles per (y,z) and
 *   costs ~25%.  Measured with llvm-mca -mcpu=cascadelake on this exact
 *   assembly: BATCH 1338 cycles/volume (two-FMA model) / ~1423 (one-FMA,
 *   hand-recomputed from the port pressures), LANEX 1474 / ~1428.  Verified
 *   zero stack spills in both kernels at -march=cascadelake.  Which path
 *   wins is decided by measurement in fft3d_create(), not by this comment.
 *
 * MEMORY / BANDWIDTH
 *   Compute-bound regime (batch working set <= L3): as above.
 *   Bandwidth-bound regime (large batch): the floor is 8 KiB read + 8 KiB
 *   written per volume.  Ordinary stores add an 8 KiB read-for-ownership per
 *   volume (50% more traffic), so the final write-out uses _mm512_stream_pd
 *   when in+out exceeds ~12 MiB.  Every output store is a full, 64-B-aligned
 *   cache line, which is the condition under which write-combining succeeds
 *   (Drepper 6.1); no NT store is ever used on a partial line or on the
 *   L1-resident intermediate.
 *
 * LAYOUT / PADDING
 *   BATCH scratch  SRe/SIm[((x*9 + z)*9 + y)*W]  -> y stride 1 line,
 *          z stride 9 lines, x stride 81 lines: all ODD numbers of cache
 *          lines, so all 64 L1 sets and all L2 sets are used (Bailey E = 1,
 *          section 04 7.3).  Unpadded 8/8 would put the x stride on 8 lines
 *          => 8 of 64 sets.  Size 2*648*W*8 B = 81 KiB at W=8 (L2), but every
 *          pass touches only a 10 KiB x-plane or a 9 KiB row set at a time.
 *   LANEX scratch  SRe/SIm[(z*9 + y)*8]  -> 9 KiB total, L1-resident.
 *
 * ASSUMPTIONS
 *   * L == 8 only (fft3d_supports rejects everything else).
 *   * in and out are 64-byte aligned, distinct, and in is not modified
 *     (both guaranteed by driver.c).
 *   * batch is known when the plan is built (used for the path choice and the
 *     NT-store decision).
 *   * No library call of any kind is made inside fft3d_execute().
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
 * Vector abstraction.  Three instantiations:
 *   L8_EMU8  : W = 8 emulated in plain C -- lets the AVX-512 index logic and
 *              the LANEX path be verified on a machine without AVX-512.
 *   AVX-512  : W = 8, the graded path.
 *   AVX2     : W = 4, the locally testable path.
 *   plain    : W = 1, portable reference.
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
/* faithful emulation of the three intrinsics the transpose network uses */
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
static const int EILVLO[8] = {0,8,1,9,4,12,5,13};
static const int EILVHI[8] = {2,10,3,11,6,14,7,15};
#define VUNPLO(a,b)   vunplo((a),(b))
#define VUNPHI(a,b)   vunphi((a),(b))
#define VSH44(a,b)    vshuf128((a),(b),0x44)
#define VSHEE(a,b)    vshuf128((a),(b),0xEE)
#define VSH88(a,b)    vshuf128((a),(b),0x88)
#define VSHDD(a,b)    vshuf128((a),(b),0xDD)
#define VILVLO(a,b)   vperm2((a),EILVLO,(b))
#define VILVHI(a,b)   vperm2((a),EILVHI,(b))
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
/* interleave, composed with the SW lane permutation the transpose network
 * leaves behind: lane m of a/b holds element SW(m), so the index for
 * element x is SW(x).  Lane order is (lane7 ... lane0) in set_epi64. */
#define VILVLO(a,b)   _mm512_permutex2var_pd((a), \
                        _mm512_set_epi64(13,5,12,4,9,1,8,0), (b))
#define VILVHI(a,b)   _mm512_permutex2var_pd((a), \
                        _mm512_set_epi64(15,7,14,6,11,3,10,2), (b))
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
 * boundaries and one 8^3 volume spans two pages, so in the DRAM-streaming
 * regime the hardware prefetcher restarts twice per volume; issuing the next
 * volume's lines by hand costs 128 uops per volume against a 1248-cycle
 * kernel.  Only done when the batch is large enough for NT stores. */
#if defined(__x86_64__) && !defined(EMULATED)
#  define PF(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)
#else
#  define PF(p) ((void)(p))
#endif

/* -------------------------------------------------------------------------
 * VW x VW in-register double transpose, up to a fixed lane permutation:
 *
 *     out[k][l] = in[SW(l)][k],   SW = swap of lane bits 1 and 2 at W = 8
 *                                 (0,1,4,5,2,3,6,7), identity at W = 4.
 *
 * W = 8 network (24 shuffle uops, ALL two-source non-destructive forms with
 * immediate control -- no index vectors, no vpermt2pd, hence no compiler
 * register copies).  Borrowed from L8_fusedaxes round 1: a straight
 * r1<->l1 middle level has no non-destructive AVX-512 encoding (vshuff64x2
 * can only route the source-register bit to lane bit 2), but the 3-cycle
 * r1 -> l2 -> l1 -> r1 (imm 0x88/0xDD) is encodable, and composing
 *   stage A  r2 <-> l2          (vshuff64x2 0x44 / 0xEE)
 *   stage B  r1 -> l2 -> l1     (vshuff64x2 0x88 / 0xDD)
 *   stage C  r0 <-> l0          (vunpcklo / vunpckhi)
 * yields a transpose whose only residue is the lane permutation SW, which
 * every call site absorbs as a compile-time relabel.
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
 * Multiplication by +-i is a rename plus a sign folded into the neighbouring
 * add/sub, hence free in split layout; the two c-twiddles are folded into the
 * last butterfly as fmadd/fnmadd.
 * 52 instructions: 44 add/sub + 8 FMA.
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
#define ZSTR    128            /* doubles between consecutive z planes      */
#define YSTR     16            /* doubles between consecutive y rows        */
#define NGRP   (16 / VW)       /* vectors per interleaved 8-complex x-row   */

/* BATCH scratch: index ((x*9 + z)*9 + y), vectors of VW doubles.
 * strides in 64-B lines at VW=8:  y 1, z 9, x 81 -- all odd.  */
#define BPY 9
#define BPZ 9
#define BSLOT (8 * BPZ * BPY)          /* 648 vectors per component */

/* LANEX scratch: index (x*9 + z), vectors of 8 doubles whose lanes are y.
 * z stride 1 line, x stride 9 lines -- both odd, so all L1 sets are used. */
#define LPZ 9
#define LSLOT (8 * LPZ)                /* 72 vectors per component */

enum { MODE_BATCH = 0, MODE_LANEX = 1 };

struct fft3d_plan {
    int    batch;
    int    mode;
    int    nt;
    double *scr;        /* 64-B aligned working set                        */
    double *stage_in;   /* VW zero-padded volumes for the B % VW tail      */
    double *stage_out;
    void   *raw;
};

/* =========================================================================
 * BATCH path: lanes = VW consecutive volumes.
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
     * with the y pass fused per z-plane: the 8 KiB of scratch a z-plane
     * occupies is still L1-hot when its y pencils are transformed, which
     * saves one full 81 KiB scratch round trip through L2 per block.       */
    for (int z = 0; z < 8; ++z) {
        if (nt) {                       /* one block of lookahead */
            for (int b = 0; b < VW; ++b) {
                const double *nx = src + (size_t)VW * VOLD + (size_t)b * VOLD
                                       + z * ZSTR;
                PF(nx); PF(nx + 8); PF(nx + 16); PF(nx + 24);
                PF(nx + 32); PF(nx + 40); PF(nx + 48); PF(nx + 56);
                PF(nx + 64); PF(nx + 72); PF(nx + 80); PF(nx + 88);
                PF(nx + 96); PF(nx + 104); PF(nx + 112); PF(nx + 120);
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

    /* ---- phase 2b: z pass, SOFTWARE-PIPELINED against the transposing
     * store of the previous y.  The z codelet is 52 port-0 instructions with
     * no shuffles; the transposing store is 48 port-5 shuffles with no
     * arithmetic.  Emitted adjacent and mutually independent, the two hide
     * inside each other (they fit together in the 224-entry ROB), which is
     * what takes the kernel from ~1630 to ~1300 cycles/volume.  A separate
     * shuffle-only output pass wastes port 0 for 48 cycles per (y,z).       */
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
 * LANEX path (VW == 8 only): lanes = the 8 x values of a single volume.
 *   pass 1  deinterleave (1 shuffle / 64 B) fused with the y transform
 *   pass 2  z transform (free)
 *   pass 3  one 8x8 transpose x<->y, x transform, transposing interleaved
 *           store straight into `out`
 * ======================================================================= */
#if VW == 8
static AI void lanex_volume(double *restrict scr,
                            const double *restrict src,
                            double *restrict dst,
                            const int nt, const int pf)
{
    double *restrict SR = scr;
    double *restrict SI = scr + (size_t)LSLOT * 8;

    /* ---- pass 1: transposing load (deinterleave AND x<->y in one 8x8
     * network) fused with the x transform.  48 port-5 shuffles against 52
     * port-0 instructions per z plane, i.e. balanced.                      */
    for (int z = 0; z < 8; ++z) {
        if (pf) {                       /* one volume of lookahead */
            const double *nx = src + VOLD + z * ZSTR;
            PF(nx); PF(nx + 8); PF(nx + 16); PF(nx + 24);
            PF(nx + 32); PF(nx + 40); PF(nx + 48); PF(nx + 56);
            PF(nx + 64); PF(nx + 72); PF(nx + 80); PF(nx + 88);
            PF(nx + 96); PF(nx + 104); PF(nx + 112); PF(nx + 120);
        }
        vd xr[8], xi[8];
        for (int g = 0; g < 2; ++g) {
            vd tin[8], tout[8];
            for (int y = 0; y < 8; ++y)
                tin[y] = VLD(src + z * ZSTR + y * YSTR + g * 8);
            vtrans(tout, tin);          /* tout[j] = double j of the row, lanes = y */
            for (int q = 0; q < 8; ++q) {
                const int j = g * 8 + q;
                if (j & 1) xi[j >> 1] = tout[q];
                else       xr[j >> 1] = tout[q];
            }
        }
        r8(xr, xi);                     /* x transform, lanes = y */
        for (int x = 0; x < 8; ++x) {
            const size_t o = (size_t)(x * LPZ + z) * 8;
            VST(SR + o, xr[x]); VST(SI + o, xi[x]);
        }
    }

    /* ---- pass 2: z transform.  Lanes are y, so this is shuffle-free and
     * the 8 z values of one x are 8 consecutive vectors.                   */
    for (int x = 0; x < 8; ++x) {
        double *restrict br = SR + (size_t)x * LPZ * 8;
        double *restrict bi = SI + (size_t)x * LPZ * 8;
        vd r[8], m[8];
        for (int z = 0; z < 8; ++z) { r[z] = VLD(br + z * 8); m[z] = VLD(bi + z * 8); }
        r8(r, m);
        for (int z = 0; z < 8; ++z) { VST(br + z * 8, r[z]); VST(bi + z * 8, m[z]); }
    }

    /* ---- pass 3: one 8x8 transpose per component turns lanes y into lanes
     * x, the y transform is then free, and the interleaving store costs two
     * shuffles per row.  64 port-5 against 52 port-0 per z plane.          */
    for (int z = 0; z < 8; ++z) {
        vd t[8], u[8], pr[8], pi[8];
        for (int x = 0; x < 8; ++x) t[x] = VLD(SR + (size_t)(x * LPZ + z) * 8);
        vtrans(u, t);                   /* u[k] = Re at y=SW(k), lanes = x SW */
        for (int k = 0; k < 8; ++k) pr[SW(k)] = u[k];
        for (int x = 0; x < 8; ++x) t[x] = VLD(SI + (size_t)(x * LPZ + z) * 8);
        vtrans(u, t);
        for (int k = 0; k < 8; ++k) pi[SW(k)] = u[k];
        r8(pr, pi);                     /* y transform, lanes = x */
        for (int y = 0; y < 8; ++y) {
            double *p = dst + z * ZSTR + y * YSTR;
            vd lo = VILVLO(pr[y], pi[y]);
            vd hi = VILVHI(pr[y], pi[y]);
            if (nt) { VSTNT(p, lo); VSTNT(p + 8, hi); }
            else    { VST  (p, lo); VST  (p + 8, hi); }
        }
    }
}

static void lanex_run(double *scr, const double *src, double *dst, long nvol, int nt)
{
    if (nt) {
        for (long v = 0; v < nvol; ++v)
            lanex_volume(scr, src + (size_t)v * VOLD, dst + (size_t)v * VOLD, 1, 1);
    } else if (nvol > 1) {
        for (long v = 0; v < nvol; ++v)
            lanex_volume(scr, src + (size_t)v * VOLD, dst + (size_t)v * VOLD, 0, 1);
    } else {
        lanex_volume(scr, src, dst, 0, 0);
    }
}
#endif /* VW == 8 */

/* ========================== the API ==================================== */

const char *fft3d_name(void) { return "L8_batchsimd"; }

const char *fft3d_description(void)
{
    return "split-complex radix-8 (DFT8 (x) I_8), non-destructive vshuff64x2 "
           "8x8 repack, lane-per-x default, NT stores at large batch";
}

int fft3d_supports(int L) { return L == 8; }

/* --- driver-independent execution used by both the plan and the tuner --- */
static void run_all(const struct fft3d_plan *p,
                    const double *src, double *dst)
{
    const long nvol = p->batch;
#if VW == 8
    if (p->mode == MODE_LANEX) {
        lanex_run(p->scr, src, dst, nvol, p->nt);
        if (p->nt) VFENCE();
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

/* Self-tune: time every legal (mode, nt) combination on a surrogate batch
 * that is big enough to reproduce the real cache regime, and keep the best.
 * All of this is inside fft3d_create() and therefore outside the score. */
static void autotune(struct fft3d_plan *p)
{
    long nsur = p->batch < 2048 ? p->batch : 2048;
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

    /* Candidate list: (mode, nt).  LANEX needs VW == 8 lanes. */
    int cmode[4], cnt[4], ncand = 0;
    for (int mode = 0; mode < 2; ++mode) {
#if VW != 8
        if (mode == MODE_LANEX) continue;   /* lane count must equal L */
#endif
        /* BATCH is legal even below one full vector: run_all routes the
         * remainder through the zero-padded staging block.  Letting the tuner
         * see it means B=1 is a measured choice, not an assumption. */
        for (int nt = 0; nt < 2; ++nt) { cmode[ncand] = mode; cnt[ncand] = nt; ++ncand; }
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

    /* Trials are INTERLEAVED across candidates (round-robin, min per
     * candidate), so a frequency ramp, an AVX-512 licence transition, or a
     * noise burst on a shared node biases everyone equally instead of
     * whichever candidate ran first.  (Borrowed from L8_radix8 round 1.)
     * Round 2 lesson: the sequential 5-trials-per-candidate tuner picked
     * BATCH at B=64 on wallaby when a forced measurement showed LANEX 1.5x
     * faster -- one noisy block was enough to invert the choice. */
    double bt[4];
    for (int c = 0; c < ncand; ++c) {
        bt[c] = 1e30;
        t.mode = cmode[c]; t.nt = cnt[c];
        run_all(&t, ti, to);                /* warm each path once */
    }
    for (int trial = 0; trial < 7; ++trial) {
        for (int c = 0; c < ncand; ++c) {
            t.mode = cmode[c]; t.nt = cnt[c];
            double t0 = wall();
            for (long q = 0; q < reps; ++q) run_all(&t, ti, to);
            double dt = (wall() - t0) / (double)reps;
            if (dt < bt[c]) bt[c] = dt;
        }
    }

    /* 3% hysteresis in favour of the default choice, so timing noise cannot
     * flip the plan for nothing. */
    int best_mode = p->mode, best_nt = p->nt;
    double best = 1e30;
    for (int c = 0; c < ncand; ++c) {
        const int incumbent = (cmode[c] == p->mode && cnt[c] == p->nt);
        const double score = bt[c] * (incumbent ? 1.0 : 1.03);
        if (score < best) { best = score; best_mode = cmode[c]; best_nt = cnt[c]; }
    }

    p->mode = best_mode;
    p->nt = best_nt;

    if (getenv("L8_TUNE_DEBUG")) {
        fprintf(stderr, "[L8_batchsimd tune] batch=%d nsur=%ld reps=%ld ->"
                        " mode=%s nt=%d |", p->batch, nsur, reps,
                best_mode == MODE_LANEX ? "LANEX" : "BATCH", best_nt);
        for (int c = 0; c < ncand; ++c)
            fprintf(stderr, " %s/nt%d=%.3fus",
                    cmode[c] == MODE_LANEX ? "LANEX" : "BATCH", cnt[c],
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

    /* one arena: BATCH scratch | LANEX scratch | tail staging in/out */
    const size_t nb_batch = (size_t)2 * BSLOT * VW;
    const size_t nb_lanex = (size_t)4 * 72 * 8;       /* two 9 KiB halves */
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

    /* Defaults, which the autotune only overrides by a clear margin.
     * Round 2: LANEX everywhere at VW = 8.  With the non-destructive
     * transpose network, forced-mode measurements on wallaby put LANEX at
     * 0.31 us/vol against BATCH's 0.47-0.49 in every compute-bound regime
     * (B = 8, 64) and 2.5x ahead at B = 2048; it also serves any B. */
#if VW == 8
    p->mode = MODE_LANEX;
#else
    p->mode = MODE_BATCH;
#endif
    p->nt = ((size_t)batch * VOLD * sizeof(double) * 2 > (size_t)12 << 20);

#if defined(L8_FORCE_MODE)
    p->mode = L8_FORCE_MODE;
#  if defined(L8_FORCE_NT)
    p->nt = L8_FORCE_NT;
#  endif
#else
    autotune(p);
#endif
    return p;
}

void fft3d_destroy(fft3d_plan *plan)
{
    if (!plan) return;
    free(plan->raw);
    free(plan);
}
