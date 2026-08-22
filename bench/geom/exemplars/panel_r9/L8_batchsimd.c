/* ===========================================================================
 * L8_batchsimd -- forward, unnormalised, complex-double 3D DFT of a FIXED
 *                 8 x 8 x 8 cube, over a batch of B volumes, out-of-place.
 *
 * TECHNIQUE (round 9: retire the arena at the cells it mis-ranks; de-alias
 *   and de-lottery the scratch)
 *   The panel_r8 node data closed the structure question at B=1 and B=64 at
 *   DRIVER level while the create()-arena kept getting it wrong (B=1 pick
 *   strings: LANEX3 in 4 of 6 creates, and every LANEX3-picked driver run
 *   cost +5-6%: 0.5935 vs FUSED's 0.5643/0.5647; B=64: the one LANEX3 pick
 *   was the 0.6151 run vs FUSED+s0's 0.5886/0.5910).  So this round:
 *     B=1     NO tournament.  FUSED/plain hardwired -- the configuration the
 *             driver itself has measured fastest in 4 of 4 FUSED-picked runs
 *             across r7/r8 (0.5577/0.5647/0.5643/0.5647).  The arena cannot
 *             rank near-ties at this scale (r7/r8, twice documented) and its
 *             mistakes are the entire +5% tail of my B=1 spread.
 *     B=64    candidate set restricted to FUSED x {s0, none, s0w}: LANEX3's
 *             arena wins there are driver-level losses in both r7 and r8.
 *     stream  untouched, per the r8 VERDICT ("converged, stop tuning it"):
 *             FUSED+s0w default was picked 6/6 in r8.
 *   Scratch changes (free, structural):
 *     * FUSED/FUSED3 SI moved from scr+512 to scr+520 doubles: SR and SI were
 *       EXACTLY 4096 B apart, so every pass-B load of SR line j falsely
 *       aliases (bits 11:6) any in-flight pass-A store to SI line j at the
 *       pass boundary -- the ld_blocks_partial.address_alias mechanism
 *       L8_fusedaxes' r7 line-model names (their model, applied to my layout).
 *       +64 B breaks the relation for every same-index pair.  Note their own
 *       classic fused keeps the 4096 relation and still runs 0.552, so this
 *       is NOT the inter-entry gap; it is simply free to remove.
 *     * Arena page-aligned (4096): the scratch frame offset is now a
 *       compile-time fact instead of a glibc draw, per L17_matrixsimd's r8
 *       finding that heap offsets are a fixed property of a build -- this
 *       makes round-over-round comparisons of my own numbers meaningful.
 *   NOT done, with reasons in the strategy record: no fusedAA port (the node
 *   declined fusedaxes' AA variants at B=1 twice while plain fused won the
 *   cell); no runner pruning (four documented instances panel-wide of
 *   refactor-around-an-untouched-hot-path regressions); no streaming work.
 *
 * TECHNIQUE (round 8: consolidation on the node's own r7 verdicts)
 *   Round 8 changes NO kernel code.  The panel_r7 node runs settled the
 *   structure question cell by cell and this round writes those verdicts into
 *   the defaults and candidate sets:
 *     B=1     default LANEX2 -> FUSED (node picked FUSED 2/3 and won the cell
 *             with it, 0.5577/0.5647 vs the LANEX3-pick run's 0.5935);
 *     B=64    default stays FUSED+s0 (the r7 run that kept it was the fastest,
 *             0.588 vs LANEX3+s0's 0.605/0.613) but mid-regime hysteresis is
 *             widened 3% -> 6%: the create()-arena ranking inverted the
 *             driver-level ranking twice at exactly the 3% band;
 *             FUSED+s0w / LANEX3+s0w join the mid candidate set -- at B=64 the
 *             working set is exactly the node's 1 MiB L2, so output lines pay
 *             L3-latency RFOs that prefetchw can hide, and wallaby's 2 MiB L2
 *             is structurally unable to measure this;
 *     B=1 set drops LANEX2S, all sets drop FUSED3 (zero picks in 12/12 r7
 *             node runs; the code stays compiled for forced runs).
 *   Two nulls measured and recorded: the r7-VERDICT build-flag gap
 *   (-funroll-loops present in tryout, absent on the node) is worth nothing
 *   here (B=1 fast-state 0.305 vs 0.307; B=16384 straddles zero), and the
 *   scalar-instruction audit (L45_pfa r7) finds 51-74 scalar instructions per
 *   runner, all pointer bookkeeping, no table reloads.
 *
 * TECHNIQUE (round 7)
 *   Split-complex (SoA) straight-line radix-8 codelet applied as DFT_8 (x) I_8
 *   (the "vector terminal" of Franchetti & Puschel): every scalar operation of
 *   the codelet is exactly ONE 8-wide vector instruction and no cross-lane
 *   operation appears inside any transform.  FOUR structures are compiled and
 *   fft3d_create() picks one by measurement -- the panel_r3 VERDICT's process
 *   lesson ("add candidates; do not replace structures") applied literally.
 *
 *   ROUND 7 change: the FUSED structure, ported from L8_fusedaxes (whose
 *   fused+pfs+pfw won the node's B=64/2048/16384 cells in panel_r5 at
 *   0.594/0.910/1.254, picked 3/3 streaming; L8_radix8 ported the same shape
 *   in r5 as "1f" and fell 0.680 -> 0.619 at B=64).  What distinguishes it
 *   from my LANEX2S (same 256+256 mem ops, same sequential-ish store order)
 *   is SHUFFLE PLACEMENT: the DRAM-facing load pass carries only 16 light
 *   deinterleave shuffles per iteration (lane = the contiguous z axis, so no
 *   transposing load is needed at all -- the first DFT runs across registers),
 *   while both heavy transpose networks run in pass B against L1 scratch.
 *   On wallaby r6 their fused+pfs+pfw measured 0.637 vs my best LANEX+s0w
 *   0.725 -- the ~12% is the fusion/placement win on top of pfw, and this
 *   round puts that shape inside my own tournament.  My 52-op FMA codelet is
 *   substituted for their 56-op 4-mul dft8s (identical natural-order I/O),
 *   saving 48 FP instructions/volume vs their code.  Round 6's PF_SW
 *   (prefetchw of the next volume's output lines, RFO hiding) is carried
 *   unchanged and the fused shape gets the same s0/s0w hooks (8+8 lines per
 *   iteration of both passes).
 *
 *   ROUND 6 change (never node-timed -- panel_r6 was abandoned before its
 *   timing pass): WRITE-INTENT prefetch (prefetchw) of the next volume's
 *   OUTPUT lines, borrowed from L8_fusedaxes round 5; with plain stores every
 *   output line pays an RFO; __builtin_prefetch(p,1,3) issues it one volume
 *   early at the SAME spread cadence as the read prefetch (LANEX3: 6/5/5
 *   lines of each stream per iteration of passes 1/2/3; LANEX2/2S/FUSED: 8+8
 *   per iteration of both passes).  Mid-batch (B=64) default was moved off
 *   LANEX2S after VERDICT r5 3a; burst-t0 candidates deleted everywhere.
 *
 *     LANEX2  (round-3 shape, node-verified 0.570 us at B=1) two passes:
 *             pass A per slow plane: transposing load, fast-axis DFT, one
 *             in-register transpose pair, mid-axis DFT, store scratch;
 *             pass B per mid-spectral row: slow-axis DFT (shuffle-free),
 *             interleave, store.  Output writes are STRIDED (1 KiB jumps).
 *     LANEX2S (new this round) two passes with the axis roles swapped:
 *             pass A per mid row: transposing load (strided 1 KiB reads),
 *             fast-axis DFT, transpose pair, SLOW-axis DFT, store scratch;
 *             pass B per slow-spectral plane: mid-axis DFT, interleave,
 *             store.  Output writes are FULLY SEQUENTIAL (the mechanism the
 *             r3 VERDICT identified as worth 18% at batch, from L8_radix8's
 *             measurement), at LANEX2's memory-op count (256+256, not the
 *             3-pass 384+384).  The strided traffic moves to the READ side,
 *             which pays no RFO and is covered by the next-volume prefetch.
 *     LANEX3  (round-2 shape, node-verified 1.205 us at B=2048 and 1.557 at
 *             B=16384) three passes: fast-axis DFT to scratch, slow-axis DFT
 *             in place in scratch (shuffle-free), then per slow-spectral
 *             plane a transpose pair, mid-axis DFT, interleave, seq. store.
 *     FUSED   (new this round, ported from L8_fusedaxes -- the shape that won
 *             every batched node cell in panel_r5) two passes with lane = z:
 *             pass A per slow plane x: 8 contiguous z-pencil deinterleaves
 *             (16 unpck shuffles TOTAL, no transposing load), mid-axis DFT
 *             across registers, store scratch [ky][x]; pass B per ky: slow-
 *             axis DFT across registers, transpose pair, fast-axis DFT,
 *             48-op fused untranspose+interleave, 16 table-driven stores
 *             into the ky-row of every kx plane.  Heavy shuffle networks all
 *             sit against L1 scratch; the DRAM-facing pass is shuffle-light.
 *
 * OPERATION COUNT (per volume)
 *   radix-8 codelet, split complex, FMA-folded: 52 instructions
 *     = 44 add/sub + 8 FMA (Burrus T7.1 / FFTW n1_8's 4 mul + 52 add with the
 *       two (1-+i)/sqrt2 twiddles folded into the last butterfly: 56 -> 52)
 *   3 axes * 64 pencils * 52 = 9984 real FP ops = 1248 vector FP instructions
 *   shuffles: 896 in every structure -- LANEX: 768 in the two transposing
 *   passes + 128 interleaves; FUSED: 128 deinterleave + 384 transpose + 384
 *   fused untranspose/interleave.  Loads/stores: LANEX2/LANEX2S/FUSED
 *   256 + 256, LANEX3 384 + 384 (the extra 256 are L1-resident scratch
 *   traffic).  DRAM-facing traffic is identical for all four: 8 KiB read +
 *   8 KiB written per volume (16 KiB with NT); only the ORDER differs, and
 *   in FUSED which pass carries the shuffle pressure.
 *
 *   Port model, scored machine (Gold 5218, Cascade Lake, ONE 512-bit FMA
 *   unit): all 512-bit FP on port 0 (1/cycle), all 512-bit shuffles on port 5
 *   (1/cycle) -> p0 floor 1248 cycles/volume, the 896 shuffles fit under it.
 *
 * MEMORY / BANDWIDTH
 *   B=1: src 8K + dst 8K + scratch 9K = 25 KiB, all L1-resident.
 *   Large batch: floor is 8 KiB read + 8 KiB written per volume, PLUS the
 *   8 KiB RFO read of the output unless something removes it.  Four rounds
 *   of node evidence say NT stores (which avoid the RFO) LOSE to plain
 *   stores; the r5 evidence (fusedaxes' 0.910, L36_pfa's -16.6%) says the
 *   winning move is to HIDE the RFO instead: prefetchw the output lines one
 *   volume ahead.  Software prefetch of the next volume is needed anyway
 *   (the L2 streamer stops at 4 KiB page boundaries and one volume spans
 *   two); placement (spread t0 / spread t0+pfw / none) is plan-time tuned.
 *   prefetchw is pure uop tax on cache-resident output (L36_pfa +11-13%,
 *   fusedaxes +3%), so PF_SW is offered only in the streaming set.
 *
 * TUNING / REPORTING
 *   fft3d_create() times every legal (mode, nt, pf) candidate round-robin-
 *   interleaved (min of 7 trials each, one UNTIMED state-setting pass per
 *   candidate per trial so plain/NT candidates are timed against their own
 *   cache state -- borrowed from L8_fusedaxes r3 -- and 3% hysteresis toward
 *   the default).  The chosen configuration is written into
 *   fft3d_description() so the monitor can read the node's pick per case.
 *   BATCH (lanes = 4 consecutive volumes) remains only as the W=4/scalar
 *   path: the r3 node tuner never selected it in any cell (3/3 runs, all
 *   four batches), so at W=8 it is deleted from the candidate set.
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
#include <unistd.h>

#include "fft3d_api.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

/* -------------------------------------------------------------------------
 * Vector abstraction.  Instantiations:
 *   L8_EMU8  : W = 8 emulated in plain C -- verifies the AVX-512 index logic
 *              and all three LANEX paths on a machine without AVX-512.
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
 * the hardware prefetcher restarts twice per volume.  The PLACEMENT (burst
 * vs spread vs none) is the tuned plan-time choice this round; the hint is
 * fixed at t0 (t1 was never picked by the node tuner in rounds 3-4). */
#if defined(__x86_64__) && !defined(EMULATED)
#  define PF0(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)
/* write-intent: emits prefetchw where PRFCHW exists (the node, wallaby);
 * borrowed from L8_fusedaxes r5, who took it from L6_unrolled/L6_pfa. */
#  define PFW(p) __builtin_prefetch((const void *)(p), 1, 3)
#else
#  define PF0(p) ((void)(p))
#  define PFW(p) ((void)(p))
#endif

/* PF_T0: 16-line plane burst at the top of each pass-A/pass-1 iteration
 * (the r4 shape).  PF_S0: the same 128 lines of volume v+1 spread a few per
 * iteration across ALL passes -- borrowed from L8_radix8 round 4, where
 * spread + plain stores won both streaming node cells.  PF_SW: PF_S0 plus
 * the 128 OUTPUT lines of volume v+1 issued prefetchw at the same cadence
 * (RFO hiding, borrowed from L8_fusedaxes round 5's node winner). */
enum { PF_NONE = 0, PF_T0 = 1, PF_S0 = 2, PF_SW = 3 };

/* n consecutive 64-B lines of `base`, starting at line index `first`.
 * n is always a small compile-time constant, so -O3 fully unrolls this. */
#define PF_LINES(base, first, n)                                              \
    do { const double *pl_ = (base) + (size_t)(first) * 8;                    \
         for (int q_ = 0; q_ < (n); ++q_) PF0(pl_ + (size_t)q_ * 8);          \
    } while (0)

#define PFW_LINES(base, first, n)                                             \
    do { double *pw_ = (base) + (size_t)(first) * 8;                          \
         for (int q_ = 0; q_ < (n); ++q_) PFW(pw_ + (size_t)q_ * 8);          \
    } while (0)

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

/* ---------------- geometry (element (x,y,z) at ((x*8+y)*8+z)) ------------ */
#define VOLD   1024            /* doubles per volume (8^3 complex)          */
#define ZSTR    128            /* doubles between consecutive slow planes   */
#define YSTR     16            /* doubles between consecutive middle rows   */
#define NGRP   (16 / VW)       /* vectors per interleaved 8-complex row     */

/* BATCH scratch: index ((x*9 + z)*9 + y), vectors of VW doubles.
 * strides in 64-B lines at VW=4:  y 1, z 9, x 81 -- all odd.  */
#define BPY 9
#define BPZ 9
#define BSLOT (8 * BPZ * BPY)          /* 648 vectors per component */

/* LANEX scratch: index (row*9 + col), vectors of 8 doubles whose lanes are
 * the contiguous axis.  col stride 1 line, row stride 9 lines -- both odd,
 * so all L1 sets are used.  9 KiB per component pair total. */
#define LPZ 9
#define LSLOT (8 * LPZ)                /* 72 vectors per component */

enum { MODE_BATCH = 0, MODE_LANEX2 = 1, MODE_LANEX2S = 2, MODE_LANEX3 = 3,
       MODE_FUSED = 4, MODE_FUSED3 = 5 };

/* FUSED scratch SI offset in doubles.  520 (the round-9 default) puts SI
 * 4160 B after SR so no same-index SR/SI pair shares address bits 11:6;
 * -DL8_SI_OFF=512 restores the round-7/8 exactly-4096-B relation for A/B. */
#ifndef L8_SI_OFF
#define L8_SI_OFF 520
#endif

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
 * BATCH path: lanes = VW consecutive volumes.  Compiled only when VW != 8:
 * the r3 node tuner never selected it at W = 8 in any cell (3/3 runs, all
 * four batches), so there it is deleted from the candidate set entirely.
 * ======================================================================= */
#if VW != 8
static AI void batch_block(double *restrict scr,
                           const double *restrict src,
                           double *restrict dst,
                           const int nt)
{
    double *restrict SR = scr;
    double *restrict SI = scr + (size_t)BSLOT * VW;

/* one z-pass codelet at (x,y): 52 FP instructions, zero shuffles */
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

/* lane-major -> volume-major for one (y,z): shuffles only, zero FP */
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

/* one y-pass codelet at (x,z): 52 FP instructions, zero shuffles */
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
     * with the y pass fused per z-plane (the scratch z-plane is
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
     * store of the previous y: the z codelet has no shuffles, the transposing
     * store no arithmetic; emitted adjacent and independent they hide in
     * each other. */
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
#endif /* VW != 8 */

/* =========================================================================
 * The three W = 8 structures.  Common pieces, with the driver's layout
 * written as element (a,b,c) at a*ZSTR + b*YSTR + 2c (a slow, c contiguous):
 *
 *   transposing load of rows (fixed one of a/b, other = t):
 *     16 loads, 48 shuffles -> xr/xi[c], lanes = t.SW
 *   in-register transpose pair (48 shuffles), SW residue absorbed by the
 *     free rename reg[SW(k)] = u[k]
 *   copy-free interleave (VILVLO/VILVHI2, SW-composed indices), then 16
 *     stores into the driver's interleaved layout.
 *
 * LANEX2  pass A per a:  load(a,b=t) -> C DFT -> transpose -> B DFT
 *                        -> scratch[(m*9+a)]           (m = B-spectral)
 *         pass B per m:  A DFT (shuffle-free) -> interleave ->
 *                        dst[s*ZSTR + m*YSTR]           (STRIDED stores)
 * LANEX2S pass A per b:  load(a=t,b) -> C DFT -> transpose -> A DFT
 *                        -> scratch[(s*9+b)]           (s = A-spectral)
 *         pass B per s:  B DFT (shuffle-free) -> interleave ->
 *                        dst[s*ZSTR + m*YSTR]           (SEQUENTIAL stores)
 * LANEX3  pass 1 per a:  load(a,b=t) -> C DFT -> scratch[(c*9+a)]
 *         pass 2 per c:  A DFT in place in scratch (shuffle-free)
 *         pass 3 per s:  transpose -> B DFT -> interleave ->
 *                        dst[s*ZSTR + m*YSTR]           (SEQUENTIAL stores)
 * ======================================================================= */
#if VW == 8

/* Prefetch one slow plane (1 KiB = 16 lines) of the NEXT volume. */
#define PF_PLANE(HINT, base)                                                  \
    do { const double *nx_ = (base);                                          \
         HINT(nx_); HINT(nx_ + 8); HINT(nx_ + 16); HINT(nx_ + 24);            \
         HINT(nx_ + 32); HINT(nx_ + 40); HINT(nx_ + 48); HINT(nx_ + 56);      \
         HINT(nx_ + 64); HINT(nx_ + 72); HINT(nx_ + 80); HINT(nx_ + 88);      \
         HINT(nx_ + 96); HINT(nx_ + 104); HINT(nx_ + 112); HINT(nx_ + 120);   \
    } while (0)

#define PF_DISPATCH(pf, k, src)                                               \
    do { if ((pf) == PF_T0) PF_PLANE(PF0, (src) + VOLD + (k) * ZSTR);         \
    } while (0)

/* Transposing load of the 8 rows selected by (fixed, t = 0..7):
 * deinterleave AND row-index->lanes in one 8x8 network.  After it,
 * xr/xi[c] hold double component c of all 8 rows, lanes = t.SW. */
#define TLOAD(xr, xi, rowptr_of_t)                                            \
    do { for (int g_ = 0; g_ < 2; ++g_) {                                     \
             vd tin_[8], tout_[8];                                            \
             for (int t_ = 0; t_ < 8; ++t_)                                   \
                 tin_[t_] = VLD((rowptr_of_t) + g_ * 8);                      \
             vtrans(tout_, tin_);                                             \
             for (int q_ = 0; q_ < 8; ++q_) {                                 \
                 const int j_ = g_ * 8 + q_;                                  \
                 if (j_ & 1) (xi)[j_ >> 1] = tout_[q_];                       \
                 else        (xr)[j_ >> 1] = tout_[q_];                       \
             }                                                                \
         } } while (0)

/* ---- LANEX2: (C,B) fused | A; strided output stores (r3 shape) --------- */
static AI void lanex2_volume(double *restrict scr,
                             const double *restrict src,
                             double *restrict dst,
                             const int nt, const int pf)
{
    double *restrict SR = scr;
    double *restrict SI = scr + (size_t)LSLOT * 8;

    /* ---- pass A, per slow plane a ---- */
    for (int a = 0; a < 8; ++a) {
        if (pf == PF_S0 || pf == PF_SW) PF_LINES(src + VOLD, a * 8, 8);
        else                            PF_DISPATCH(pf, a, src);
        if (pf == PF_SW) PFW_LINES(dst + VOLD, a * 8, 8);
        vd xr[8], xi[8];
        TLOAD(xr, xi, src + a * ZSTR + t_ * YSTR);
        r8(xr, xi);                 /* C DFT: registers = c_spec, lanes = b.SW */

        vd yr[8], yi[8], u[8];
        vtrans(u, xr);
        for (int k = 0; k < 8; ++k) yr[SW(k)] = u[k];
        vtrans(u, xi);
        for (int k = 0; k < 8; ++k) yi[SW(k)] = u[k];
        r8(yr, yi);                 /* B DFT: registers = m, lanes = c_spec.SW */

        for (int m = 0; m < 8; ++m) {
            const size_t o = (size_t)(m * LPZ + a) * 8;
            VST(SR + o, yr[m]); VST(SI + o, yi[m]);
        }
    }

    /* ---- pass B, per B-spectral row m ---- */
    for (int m = 0; m < 8; ++m) {
        if (pf == PF_S0 || pf == PF_SW) PF_LINES(src + VOLD, 64 + m * 8, 8);
        if (pf == PF_SW) PFW_LINES(dst + VOLD, 64 + m * 8, 8);
        vd r[8], q[8];
        for (int a = 0; a < 8; ++a) {
            const size_t o = (size_t)(m * LPZ + a) * 8;
            r[a] = VLD(SR + o); q[a] = VLD(SI + o);
        }
        r8(r, q);                   /* A DFT, shuffle-free: registers = s */
        for (int s = 0; s < 8; ++s) {
            double *p = dst + s * ZSTR + m * YSTR;
            vd lo = VILVLO(r[s], q[s]);
            vd hi = VILVHI2(q[s], r[s]);
            if (nt) { VSTNT(p, lo); VSTNT(p + 8, hi); }
            else    { VST  (p, lo); VST  (p + 8, hi); }
        }
    }
}

/* ---- LANEX2S: (C,A) fused | B; SEQUENTIAL output stores (new, r4) ------ */
static AI void lanex2s_volume(double *restrict scr,
                              const double *restrict src,
                              double *restrict dst,
                              const int nt, const int pf)
{
    double *restrict SR = scr;
    double *restrict SI = scr + (size_t)LSLOT * 8;

    /* ---- pass A, per mid row b: reads strided (1 KiB apart) ---- */
    for (int b = 0; b < 8; ++b) {
        if (pf == PF_S0 || pf == PF_SW) PF_LINES(src + VOLD, b * 8, 8);
        else                            PF_DISPATCH(pf, b, src);
        if (pf == PF_SW) PFW_LINES(dst + VOLD, b * 8, 8);
        vd xr[8], xi[8];
        TLOAD(xr, xi, src + t_ * ZSTR + b * YSTR);
        r8(xr, xi);                 /* C DFT: registers = c_spec, lanes = a.SW */

        vd ar[8], ai[8], u[8];
        vtrans(u, xr);
        for (int k = 0; k < 8; ++k) ar[SW(k)] = u[k];
        vtrans(u, xi);
        for (int k = 0; k < 8; ++k) ai[SW(k)] = u[k];
        r8(ar, ai);                 /* A DFT: registers = s, lanes = c_spec.SW */

        for (int s = 0; s < 8; ++s) {
            const size_t o = (size_t)(s * LPZ + b) * 8;
            VST(SR + o, ar[s]); VST(SI + o, ai[s]);
        }
    }

    /* ---- pass B, per A-spectral plane s: sequential 1 KiB plane writes ---- */
    for (int s = 0; s < 8; ++s) {
        if (pf == PF_S0 || pf == PF_SW) PF_LINES(src + VOLD, 64 + s * 8, 8);
        if (pf == PF_SW) PFW_LINES(dst + VOLD, 64 + s * 8, 8);
        vd r[8], q[8];
        for (int b = 0; b < 8; ++b) {
            const size_t o = (size_t)(s * LPZ + b) * 8;
            r[b] = VLD(SR + o); q[b] = VLD(SI + o);
        }
        r8(r, q);                   /* B DFT, shuffle-free: registers = m */
        for (int m = 0; m < 8; ++m) {
            double *p = dst + s * ZSTR + m * YSTR;
            vd lo = VILVLO(r[m], q[m]);
            vd hi = VILVHI2(q[m], r[m]);
            if (nt) { VSTNT(p, lo); VSTNT(p + 8, hi); }
            else    { VST  (p, lo); VST  (p + 8, hi); }
        }
    }
}

/* ---- LANEX3: C | A | B, three passes; SEQUENTIAL output stores (r2 shape,
 * node-verified 1.205 us at B=2048 / 1.557 at B=16384 in panel_r2) -------- */
static AI void lanex3_volume(double *restrict scr,
                             const double *restrict src,
                             double *restrict dst,
                             const int nt, const int pf)
{
    double *restrict SR = scr;
    double *restrict SI = scr + (size_t)LSLOT * 8;

    /* ---- pass 1, per slow plane a: sequential reads, C DFT ---- */
    for (int a = 0; a < 8; ++a) {
        if (pf == PF_S0 || pf == PF_SW) PF_LINES(src + VOLD, a * 6, 6);
        else                            PF_DISPATCH(pf, a, src);
        if (pf == PF_SW) PFW_LINES(dst + VOLD, a * 6, 6);     /* lines 0..47  */
        vd xr[8], xi[8];
        TLOAD(xr, xi, src + a * ZSTR + t_ * YSTR);
        r8(xr, xi);                 /* C DFT: registers = c_spec, lanes = b.SW */
        for (int c = 0; c < 8; ++c) {
            const size_t o = (size_t)(c * LPZ + a) * 8;
            VST(SR + o, xr[c]); VST(SI + o, xi[c]);
        }
    }

    /* ---- pass 2, per c_spec: A DFT in place in the scratch column ---- */
    for (int c = 0; c < 8; ++c) {
        if (pf == PF_S0 || pf == PF_SW) PF_LINES(src + VOLD, 48 + c * 5, 5);
        if (pf == PF_SW) PFW_LINES(dst + VOLD, 48 + c * 5, 5); /* lines 48..87 */
        double *restrict br = SR + (size_t)c * LPZ * 8;
        double *restrict bi = SI + (size_t)c * LPZ * 8;
        vd r[8], q[8];
        for (int a = 0; a < 8; ++a) { r[a] = VLD(br + a * 8); q[a] = VLD(bi + a * 8); }
        r8(r, q);                   /* A DFT, shuffle-free: registers = s */
        for (int s = 0; s < 8; ++s) { VST(br + s * 8, r[s]); VST(bi + s * 8, q[s]); }
    }

    /* ---- pass 3, per A-spectral plane s: transpose, B DFT, sequential
     * 1 KiB plane writes front to back ---- */
    for (int s = 0; s < 8; ++s) {
        if (pf == PF_S0 || pf == PF_SW) PF_LINES(src + VOLD, 88 + s * 5, 5);
        if (pf == PF_SW) PFW_LINES(dst + VOLD, 88 + s * 5, 5); /* lines 88..127 */
        vd t[8], u[8], pr[8], pi[8];
        for (int c = 0; c < 8; ++c) t[c] = VLD(SR + (size_t)(c * LPZ + s) * 8);
        vtrans(u, t);               /* u[k] = b-element SW(k), lanes = c_spec.SW */
        for (int k = 0; k < 8; ++k) pr[SW(k)] = u[k];
        for (int c = 0; c < 8; ++c) t[c] = VLD(SI + (size_t)(c * LPZ + s) * 8);
        vtrans(u, t);
        for (int k = 0; k < 8; ++k) pi[SW(k)] = u[k];
        r8(pr, pi);                 /* B DFT: registers = m, lanes = c_spec.SW */
        for (int m = 0; m < 8; ++m) {
            double *p = dst + s * ZSTR + m * YSTR;
            vd lo = VILVLO(pr[m], pi[m]);
            vd hi = VILVHI2(pi[m], pr[m]);
            if (nt) { VSTNT(p, lo); VSTNT(p + 8, hi); }
            else    { VST  (p, lo); VST  (p + 8, hi); }
        }
    }
}

/* ---- FUSED: the L8_fusedaxes shape, ported (panel_r5's node winner in every
 * batched cell).  Lane = z (the contiguous axis) throughout, so the load pass
 * needs NO transposing network -- one z-pencil is 8 contiguous complex and a
 * single unpcklo/hi pair splits it re/im with lane l holding z = PI[l],
 * PI = (0,4,1,5,2,6,3,7).  All DFTs run ACROSS registers (shuffle-free); the
 * two heavy networks (transpose pair + fused untranspose/interleave) run in
 * pass B against L1 scratch.  The codelet is my 52-op r8 (natural-order in
 * and out, identical I/O contract to fusedaxes' 56-op dft8s).
 *
 * The three non-destructive butterfly primitives, as (bit acted on) -> (bit
 * permutation); r = the register bit distinguishing the pair:
 *   BF_T1  vunpcklpd/vunpckhpd            r <-> lane0
 *   BF_T2  vshuff64x2 0x44/0xEE           r <-> lane2
 *   BF_T3  vshuff64x2 0x88/0xDD           r -> lane2 -> lane1 -> r         */
#define BF_T1(a, b) do { vd bf_ = VUNPLO(a, b); (b) = VUNPHI(a, b); (a) = bf_; } while (0)
#define BF_T2(a, b) do { vd bf_ = VSH44(a, b);  (b) = VSHEE(a, b);  (a) = bf_; } while (0)
#define BF_T3(a, b) do { vd bf_ = VSH88(a, b);  (b) = VSHDD(a, b);  (a) = bf_; } while (0)

/* 8x8 transpose, 24 non-destructive permutes (fusedaxes' trans8): in
 * register x, lane l holding z = PI[l]; out register j holding z = PI[j],
 * lane l holding x = SW(l). */
static AI void trans8f(vd *restrict m)
{
    BF_T2(m[0], m[4]); BF_T2(m[1], m[5]); BF_T2(m[2], m[6]); BF_T2(m[3], m[7]);
    BF_T3(m[0], m[2]); BF_T3(m[1], m[3]); BF_T3(m[4], m[6]); BF_T3(m[5], m[7]);
    BF_T1(m[0], m[1]); BF_T1(m[2], m[3]); BF_T1(m[4], m[5]); BF_T1(m[6], m[7]);
}

/* Inverse transpose AND complex interleave in one 48-op network over all 16
 * registers (fusedaxes' untrans_interleave): T3 on kz bit0, T3 on kz bit1,
 * T1 on the re/im bit lands (lane2,lane1,lane0) = (kz_1, kz_0, re/im) --
 * interleaved complex.  Afterwards each register is a ready-to-store
 * half-pencil; its (kx, half) destination is the FOUT table below. */
static AI void untrans_ilv(vd *restrict r, vd *restrict q)
{
    BF_T3(r[0], r[1]); BF_T3(r[2], r[3]); BF_T3(r[4], r[5]); BF_T3(r[6], r[7]);
    BF_T3(q[0], q[1]); BF_T3(q[2], q[3]); BF_T3(q[4], q[5]); BF_T3(q[6], q[7]);
    BF_T3(r[0], r[2]); BF_T3(r[1], r[3]); BF_T3(r[4], r[6]); BF_T3(r[5], r[7]);
    BF_T3(q[0], q[2]); BF_T3(q[1], q[3]); BF_T3(q[4], q[6]); BF_T3(q[5], q[7]);
    BF_T1(r[0], q[0]); BF_T1(r[1], q[1]); BF_T1(r[2], q[2]); BF_T1(r[3], q[3]);
    BF_T1(r[4], q[4]); BF_T1(r[5], q[5]); BF_T1(r[6], q[6]); BF_T1(r[7], q[7]);
}

/* register index j of the transposed group holds z = PI[j]; feed r8 in z order */
static const unsigned char fpiinv[8] = { 0, 2, 4, 6, 1, 3, 5, 7 };
/* destination (in doubles, relative to dst + ky*YSTR) of untrans_ilv's 16
 * outputs, r[0..7] then q[0..7]: offset = kx*ZSTR + half*8 */
#define FOFF(kx, half) ((short)((kx) * ZSTR + (half) * 8))
static const short FOUT[16] = {
    FOFF(0,0), FOFF(4,0), FOFF(2,0), FOFF(6,0),
    FOFF(0,1), FOFF(4,1), FOFF(2,1), FOFF(6,1),
    FOFF(1,0), FOFF(5,0), FOFF(3,0), FOFF(7,0),
    FOFF(1,1), FOFF(5,1), FOFF(3,1), FOFF(7,1)
};

static AI void fused_volume(double *restrict scr,
                            const double *restrict src,
                            double *restrict dst,
                            const int nt, const int pf)
{
    double *restrict SR = scr;          /* [ky][x], 8 vectors per row */
    double *restrict SI = scr + L8_SI_OFF; /* 520 not 512: at +4096 B every
                                         * pass-B SR-line load falsely aliases
                                         * (bits 11:6) an in-flight pass-A
                                         * SI-line store at the pass boundary */

    /* ---- pass A, per slow plane x: deinterleave (16 shuffles total) +
     * mid-axis DFT across registers, store scratch [ky][x] ---- */
    for (int x = 0; x < 8; ++x) {
        if (pf == PF_S0 || pf == PF_SW) PF_LINES(src + VOLD, x * 8, 8);
        if (pf == PF_SW) PFW_LINES(dst + VOLD, x * 8, 8);
        const double *ap = src + (size_t)x * ZSTR;
        vd r[8], q[8];
        for (int y = 0; y < 8; ++y) {
            vd A = VLD(ap + y * YSTR), B = VLD(ap + y * YSTR + 8);
            r[y] = VUNPLO(A, B);        /* lane l holds z = PI[l] */
            q[y] = VUNPHI(A, B);
        }
        r8(r, q);                       /* y axis: registers = ky, lanes = z.PI */
        for (int ky = 0; ky < 8; ++ky) {
            VST(SR + ((size_t)ky * 8 + x) * 8, r[ky]);
            VST(SI + ((size_t)ky * 8 + x) * 8, q[ky]);
        }
    }

    /* ---- pass B, per ky: slow-axis DFT, transpose pair, fast-axis DFT,
     * fused untranspose/interleave, 16 table-driven stores ---- */
    for (int ky = 0; ky < 8; ++ky) {
        if (pf == PF_S0 || pf == PF_SW) PF_LINES(src + VOLD, 64 + ky * 8, 8);
        if (pf == PF_SW) PFW_LINES(dst + VOLD, 64 + ky * 8, 8);
        const double *pr = SR + (size_t)ky * 64;
        const double *pi = SI + (size_t)ky * 64;
        vd r[8], q[8], zr[8], zq[8];
        for (int x = 0; x < 8; ++x) { r[x] = VLD(pr + x * 8); q[x] = VLD(pi + x * 8); }
        r8(r, q);                       /* x axis: registers = kx, lanes = z.PI */
        trans8f(r); trans8f(q);         /* -> registers = z.PI, lanes = kx.SW  */
        for (int j = 0; j < 8; ++j) { zr[j] = r[fpiinv[j]]; zq[j] = q[fpiinv[j]]; }
        r8(zr, zq);                     /* z axis: registers = kz, lanes = kx.SW */
        untrans_ilv(zr, zq);
        double *op = dst + (size_t)ky * YSTR;
        if (nt) {
            for (int j = 0; j < 8; ++j) {
                VSTNT(op + FOUT[j], zr[j]); VSTNT(op + FOUT[j + 8], zq[j]);
            }
        } else {
            for (int j = 0; j < 8; ++j) {
                VST(op + FOUT[j], zr[j]); VST(op + FOUT[j + 8], zq[j]);
            }
        }
    }
}

/* ---- FUSED3: fusedaxes' "seq3" -- the fused shape with the fast(kx)-axis
 * DFT moved into a middle zero-shuffle pass through a second L1 scratch, so
 * pass B2's 16 half-pencil stores land in one kx plane and can be issued in
 * ASCENDING address order: the volume's write stream is fully sequential.
 * Same FP and shuffle counts as FUSED; +128 loads +128 stores, L1-resident.
 * scratch2 slot (kx*8 + ky), 16 doubles (re vector, im vector). ---- */
static AI void fused3_volume(double *restrict scr,
                             const double *restrict src,
                             double *restrict dst,
                             const int nt, const int pf)
{
    double *restrict SR = scr;
    double *restrict SI = scr + L8_SI_OFF;      /* de-aliased, see fused_volume */
    double *restrict S2 = scr + 2 * L8_SI_OFF;  /* at 520: S2-SR = 8320 B (2
                                         * lines mod 4096), S2-SI = 4160 B */

    /* ---- pass A, per slow plane x (identical to FUSED pass A) ---- */
    for (int x = 0; x < 8; ++x) {
        if (pf == PF_S0 || pf == PF_SW) PF_LINES(src + VOLD, x * 6, 6);
        if (pf == PF_SW) PFW_LINES(dst + VOLD, x * 6, 6);      /* lines 0..47 */
        const double *ap = src + (size_t)x * ZSTR;
        vd r[8], q[8];
        for (int y = 0; y < 8; ++y) {
            vd A = VLD(ap + y * YSTR), B = VLD(ap + y * YSTR + 8);
            r[y] = VUNPLO(A, B);
            q[y] = VUNPHI(A, B);
        }
        r8(r, q);
        for (int ky = 0; ky < 8; ++ky) {
            VST(SR + ((size_t)ky * 8 + x) * 8, r[ky]);
            VST(SI + ((size_t)ky * 8 + x) * 8, q[ky]);
        }
    }

    /* ---- pass B1, per ky: fast-axis DFT across registers, zero shuffles;
     * reads one contiguous scratch1 row, writes the scratch2 column ---- */
    for (int ky = 0; ky < 8; ++ky) {
        if (pf == PF_S0 || pf == PF_SW) PF_LINES(src + VOLD, 48 + ky * 5, 5);
        if (pf == PF_SW) PFW_LINES(dst + VOLD, 48 + ky * 5, 5); /* 48..87 */
        const double *pr = SR + (size_t)ky * 64;
        const double *pi = SI + (size_t)ky * 64;
        vd r[8], q[8];
        for (int x = 0; x < 8; ++x) { r[x] = VLD(pr + x * 8); q[x] = VLD(pi + x * 8); }
        r8(r, q);                       /* x axis: registers = kx */
        for (int kx = 0; kx < 8; ++kx) {
            VST(S2 + ((size_t)kx * 8 + ky) * 16,     r[kx]);
            VST(S2 + ((size_t)kx * 8 + ky) * 16 + 8, q[kx]);
        }
    }

    /* ---- pass B2, per kx: transpose pair, z DFT, fused untranspose/
     * interleave, 16 stores in ascending address order within the plane ---- */
    for (int kx = 0; kx < 8; ++kx) {
        if (pf == PF_S0 || pf == PF_SW) PF_LINES(src + VOLD, 88 + kx * 5, 5);
        if (pf == PF_SW) PFW_LINES(dst + VOLD, 88 + kx * 5, 5); /* 88..127 */
        const double *p2 = S2 + (size_t)kx * 128;
        vd r[8], q[8], zr[8], zq[8];
        for (int ky = 0; ky < 8; ++ky) {
            r[ky] = VLD(p2 + ky * 16);
            q[ky] = VLD(p2 + ky * 16 + 8);
        }
        trans8f(r); trans8f(q);         /* -> registers = z.PI, lanes = ky.SW */
        for (int j = 0; j < 8; ++j) { zr[j] = r[fpiinv[j]]; zq[j] = q[fpiinv[j]]; }
        r8(zr, zq);                     /* z axis: registers = kz */
        untrans_ilv(zr, zq);
        double *op = dst + (size_t)kx * ZSTR;
        if (nt) {
            VSTNT(op +   0, zr[0]); VSTNT(op +   8, zr[4]);
            VSTNT(op +  16, zq[0]); VSTNT(op +  24, zq[4]);
            VSTNT(op +  32, zr[2]); VSTNT(op +  40, zr[6]);
            VSTNT(op +  48, zq[2]); VSTNT(op +  56, zq[6]);
            VSTNT(op +  64, zr[1]); VSTNT(op +  72, zr[5]);
            VSTNT(op +  80, zq[1]); VSTNT(op +  88, zq[5]);
            VSTNT(op +  96, zr[3]); VSTNT(op + 104, zr[7]);
            VSTNT(op + 112, zq[3]); VSTNT(op + 120, zq[7]);
        } else {
            VST(op +   0, zr[0]); VST(op +   8, zr[4]);
            VST(op +  16, zq[0]); VST(op +  24, zq[4]);
            VST(op +  32, zr[2]); VST(op +  40, zr[6]);
            VST(op +  48, zq[2]); VST(op +  56, zq[6]);
            VST(op +  64, zr[1]); VST(op +  72, zr[5]);
            VST(op +  80, zq[1]); VST(op +  88, zq[5]);
            VST(op +  96, zr[3]); VST(op + 104, zr[7]);
            VST(op + 112, zq[3]); VST(op + 120, zq[7]);
        }
    }
}

/* One specialised runner per (structure, nt, pf) combination so every branch
 * inside the volume functions folds away.  The last volume always runs
 * pf = PF_NONE so no prefetch ever reaches past the end of the input mapping. */
#define LANEX_RUN(NAME, VOLFN, NT, PF)                                        \
static void NAME(double *scr, const double *src, double *dst, long nvol)      \
{                                                                             \
    for (long v = 0; v + 1 < nvol; ++v)                                       \
        VOLFN(scr, src + (size_t)v * VOLD,                                    \
                   dst + (size_t)v * VOLD, NT, PF);                           \
    VOLFN(scr, src + (size_t)(nvol - 1) * VOLD,                               \
               dst + (size_t)(nvol - 1) * VOLD, NT, PF_NONE);                 \
}
/* Per structure: plain x {none, burst t0, spread t0, spread t0 + pfw},
 * NT x {none, spread t0}.  Burst+NT is deliberately NOT instantiated -- the
 * fill-buffer clog that L8_radix8's r4 record documents (a 16-line burst and
 * the NT drain fight for the same ~12 fill buffers).  NT+pfw is nonsense by
 * construction (NT avoids the RFO that pfw hides). */
LANEX_RUN(l2_run_p0,     lanex2_volume,  0, PF_NONE)
LANEX_RUN(l2_run_pt0,    lanex2_volume,  0, PF_T0)
LANEX_RUN(l2_run_ps0,    lanex2_volume,  0, PF_S0)
LANEX_RUN(l2_run_psw,    lanex2_volume,  0, PF_SW)
LANEX_RUN(l2_run_n_p0,   lanex2_volume,  1, PF_NONE)
LANEX_RUN(l2_run_n_ps0,  lanex2_volume,  1, PF_S0)
LANEX_RUN(l2s_run_p0,    lanex2s_volume, 0, PF_NONE)
LANEX_RUN(l2s_run_pt0,   lanex2s_volume, 0, PF_T0)
LANEX_RUN(l2s_run_ps0,   lanex2s_volume, 0, PF_S0)
LANEX_RUN(l2s_run_psw,   lanex2s_volume, 0, PF_SW)
LANEX_RUN(l2s_run_n_p0,  lanex2s_volume, 1, PF_NONE)
LANEX_RUN(l2s_run_n_ps0, lanex2s_volume, 1, PF_S0)
LANEX_RUN(l3_run_p0,     lanex3_volume,  0, PF_NONE)
LANEX_RUN(l3_run_pt0,    lanex3_volume,  0, PF_T0)
LANEX_RUN(l3_run_ps0,    lanex3_volume,  0, PF_S0)
LANEX_RUN(l3_run_psw,    lanex3_volume,  0, PF_SW)
LANEX_RUN(l3_run_n_p0,   lanex3_volume,  1, PF_NONE)
LANEX_RUN(l3_run_n_ps0,  lanex3_volume,  1, PF_S0)
LANEX_RUN(f_run_p0,      fused_volume,   0, PF_NONE)
LANEX_RUN(f_run_ps0,     fused_volume,   0, PF_S0)
LANEX_RUN(f_run_psw,     fused_volume,   0, PF_SW)
LANEX_RUN(f_run_n_p0,    fused_volume,   1, PF_NONE)
LANEX_RUN(f_run_n_ps0,   fused_volume,   1, PF_S0)
LANEX_RUN(f3_run_p0,     fused3_volume,  0, PF_NONE)
LANEX_RUN(f3_run_ps0,    fused3_volume,  0, PF_S0)
LANEX_RUN(f3_run_psw,    fused3_volume,  0, PF_SW)
LANEX_RUN(f3_run_n_p0,   fused3_volume,  1, PF_NONE)
LANEX_RUN(f3_run_n_ps0,  fused3_volume,  1, PF_S0)
#undef LANEX_RUN
#endif /* VW == 8 */

/* ========================== the API ==================================== */

const char *fft3d_name(void) { return "L8_batchsimd"; }

/* Filled in by fft3d_create() with the tuner's actual pick, so the node's
 * per-case choice is readable off the results (VERDICT panel_r2 request). */
static char g_desc[192] =
    "split-complex radix-8; FUSED default, LANEX2/LANEX3 candidates; untuned";

const char *fft3d_description(void) { return g_desc; }

int fft3d_supports(int L) { return L == 8; }

/* --- driver-independent execution used by both the plan and the tuner --- */
static void run_all(const struct fft3d_plan *p,
                    const double *src, double *dst)
{
    const long nvol = p->batch;
#if VW == 8
#define LANEX_DISPATCH(PRE)                                                   \
    do { if (p->nt) {                                                         \
             if (p->pf == PF_S0) PRE##_run_n_ps0(p->scr, src, dst, nvol);     \
             else                PRE##_run_n_p0 (p->scr, src, dst, nvol);     \
             VFENCE();                                                        \
         } else {                                                             \
             if (p->pf == PF_SW)      PRE##_run_psw(p->scr, src, dst, nvol);  \
             else if (p->pf == PF_S0) PRE##_run_ps0(p->scr, src, dst, nvol);  \
             else if (p->pf == PF_T0) PRE##_run_pt0(p->scr, src, dst, nvol);  \
             else                     PRE##_run_p0 (p->scr, src, dst, nvol);  \
         } } while (0)

    switch (p->mode) {
    case MODE_LANEX2S: LANEX_DISPATCH(l2s); break;
    case MODE_LANEX3:  LANEX_DISPATCH(l3);  break;
    case MODE_FUSED:                        /* no burst-t0 runner: t0 -> s0 */
        if (p->nt) {
            if (p->pf == PF_S0) f_run_n_ps0(p->scr, src, dst, nvol);
            else                f_run_n_p0 (p->scr, src, dst, nvol);
            VFENCE();
        } else {
            if (p->pf == PF_SW)      f_run_psw(p->scr, src, dst, nvol);
            else if (p->pf == PF_S0 ||
                     p->pf == PF_T0) f_run_ps0(p->scr, src, dst, nvol);
            else                     f_run_p0 (p->scr, src, dst, nvol);
        }
        break;
    case MODE_FUSED3:
        if (p->nt) {
            if (p->pf == PF_S0) f3_run_n_ps0(p->scr, src, dst, nvol);
            else                f3_run_n_p0 (p->scr, src, dst, nvol);
            VFENCE();
        } else {
            if (p->pf == PF_SW)      f3_run_psw(p->scr, src, dst, nvol);
            else if (p->pf == PF_S0 ||
                     p->pf == PF_T0) f3_run_ps0(p->scr, src, dst, nvol);
            else                     f3_run_p0 (p->scr, src, dst, nvol);
        }
        break;
    default:           LANEX_DISPATCH(l2);  break;
    }
#undef LANEX_DISPATCH
#else
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
#endif
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

static const char *mode_str(int m)
{
    switch (m) {
    case MODE_LANEX2:  return "LANEX2";
    case MODE_LANEX2S: return "LANEX2S";
    case MODE_LANEX3:  return "LANEX3";
    case MODE_FUSED:   return "FUSED";
    case MODE_FUSED3:  return "FUSED3";
    default:           return "BATCH";
    }
}
static const char *pf_str(int pf)
{
    return pf == PF_T0 ? "t0" : pf == PF_S0 ? "s0"
         : pf == PF_SW ? "s0w" : "none";
}

/* Machine L3 size, for the NT/prefetch candidate gate and the tuner arena.
 * Borrowed from L8_radix8 round 4 (who got the idea from L8_fusedaxes). */
static long l3_bytes(void)
{
#ifdef _SC_LEVEL3_CACHE_SIZE
    long v = sysconf(_SC_LEVEL3_CACHE_SIZE);
    if (v > 0) return v;
#endif
    return 22L << 20;               /* the scoring node's 22 MiB */
}

/* Self-tune: time every legal (mode, nt, pf) combination on a surrogate
 * batch big enough to reproduce the real cache regime, and keep the best.
 * Trials are round-robin-interleaved across candidates (min of 7 each, 3%
 * hysteresis toward the default) so a frequency ramp or licence transition
 * biases everyone equally (idea from L8_radix8 round 1), and each candidate
 * gets one UNTIMED pass before its timed block so plain and NT candidates
 * are measured against the cache state they themselves leave behind
 * (borrowed from L8_fusedaxes round 3).  All inside fft3d_create(). */
static void autotune(struct fft3d_plan *p)
{
    /* Surrogate: cap machine-relatively at 4xL3 of volumes, clamped to
     * [4096, 8192] (borrowed from L8_radix8 r4 / L8_fusedaxes r3), so the
     * B=16384 store/prefetch policy is tuned on a faithful streaming arena
     * on the dev machine's 60 MiB L3 as well as the node's 22 MiB. */
    long cap = (4L * l3_bytes()) / (long)(2 * VOLD * sizeof(double));
    if (cap < 4096) cap = 4096;
    if (cap > 8192) cap = 8192;
    long nsur = p->batch < cap ? p->batch : cap;
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

    /* Candidate list, regime-gated (the gate idea from L8_radix8 r4).
     * NT is offered only when in+out exceed 0.9xL3 -- on the node no scored
     * cell sits in the excluded band (B=64 = 0.045xL3, B=2048 = 1.45xL3).
     * Burst+NT is never offered (fill-buffer clog).  Prefetch only above one
     * volume (a next-volume prefetch at B=1 has nothing to fetch). */
    const size_t ws = (size_t)p->batch * VOLD * sizeof(double) * 2;
    const int big = ws * 10 > (size_t)l3_bytes() * 9;
    int cmode[16], cnt[16], cpf[16], ncand = 0;
#define CAND(M, N, P) do { cmode[ncand] = (M); cnt[ncand] = (N);              \
                           cpf[ncand] = (P); ++ncand; } while (0)
#if VW == 8
    if (p->batch == 1) {
        /* unreachable since round 9: B=1 skips the tuner entirely (the arena
         * picked LANEX3 in 4 of 6 r8 creates; every LANEX3-picked driver run
         * cost +5-6%).  Kept only so a forced call stays well-defined. */
        CAND(MODE_FUSED,   0, PF_NONE);
    } else if (!big) {                       /* cache-resident batch (B=64) */
        /* Round 9: structure fixed at FUSED -- LANEX3's arena wins here were
         * driver-level losses in BOTH r7 (0.605/0.613 vs 0.588) and r8
         * (0.6151 vs 0.5886/0.5910).  Only the prefetch policy is raced. */
        CAND(MODE_FUSED,   0, PF_S0);        /* default: node r7/r8 best     */
        CAND(MODE_FUSED,   0, PF_SW);        /* rejected r8 (s0 picked) but   */
                                             /*   the L2-exact ws argument    */
                                             /*   stands; wallaby cannot test */
        CAND(MODE_FUSED,   0, PF_NONE);      /* fusedaxes' r4 winning policy */
    } else {                                 /* streaming (B=2048, 16384) */
        CAND(MODE_FUSED,   0, PF_SW);        /* default: node r7 pick 6/6,    */
        CAND(MODE_FUSED,   0, PF_S0);        /*   0.945(~0.984)/1.232         */
        CAND(MODE_LANEX3,  0, PF_SW);        /* pfw without fusion arm        */
        CAND(MODE_LANEX3,  0, PF_S0);        /* my node-verified 1.096/1.388 */
        CAND(MODE_FUSED,   1, PF_S0);        /* NT insurance, 5 rounds a loser */
        CAND(MODE_LANEX3,  0, PF_NONE);
        /* FUSED3+s0w dropped: zero picks in 6/6 r7 streaming runs. */
    }
#else
    CAND(MODE_BATCH, 0, PF_NONE);
    if (big) CAND(MODE_BATCH, 1, PF_NONE);
#endif
#undef CAND

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

    double bt[16];
    for (int c = 0; c < ncand; ++c) {
        bt[c] = 1e30;
        t.mode = cmode[c]; t.nt = cnt[c]; t.pf = cpf[c];
        run_all(&t, ti, to);                /* warm each path once */
    }
    for (int trial = 0; trial < 7; ++trial) {
        for (int c = 0; c < ncand; ++c) {
            t.mode = cmode[c]; t.nt = cnt[c]; t.pf = cpf[c];
            run_all(&t, ti, to);            /* untimed state-setting pass */
            double t0 = wall();
            for (long q = 0; q < reps; ++q) run_all(&t, ti, to);
            double dt = (wall() - t0) / (double)reps;
            if (dt < bt[c]) bt[c] = dt;
        }
    }

    /* Hysteresis toward the default.  Mid regime (cache-resident batch) gets
     * a wider band: in panel_r7 the create()-arena ranking inverted the
     * driver-level ranking twice at B=64 with gaps right at the old 3% band
     * (arena said LANEX3+s0 > FUSED+s0 by >3%; the driver measured FUSED+s0
     * 0.588 vs LANEX3+s0 0.605/0.613), so the driver-verified default must
     * only yield to a clear in-arena winner there. */
    const double hyst = (p->batch > 1 && !big) ? 1.06 : 1.03;
    int best_mode = p->mode, best_nt = p->nt, best_pf = p->pf;
    double best = 1e30;
    for (int c = 0; c < ncand; ++c) {
        const int incumbent = (cmode[c] == p->mode && cnt[c] == p->nt &&
                               cpf[c] == p->pf);
        const double score = bt[c] * (incumbent ? 1.0 : hyst);
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

#if VW == 8
    /* max over the structures: LANEX 2*LSLOT*8 = 1152 doubles; FUSED
     * 520 + 512 = 1032; FUSED3 1040 + 1024 = 2064 (SI at scr+520 and S2 at
     * scr+1040 since round 9 -- the de-aliased offsets). */
    const size_t total = 2112 + 64;
#else
    /* BATCH scratch | tail staging in/out */
    const size_t nb_batch = (size_t)2 * BSLOT * VW;
    const size_t nb_stage = (size_t)2 * VW * VOLD;
    const size_t total = nb_batch + nb_stage + 64;
#endif

    /* Page-aligned (round 9): the scratch frame offset mod 4096 is now a
     * fixed, known 0 instead of a glibc placement draw, so the alias
     * relations against the driver's buffers are at least deterministic
     * (L17_matrixsimd r8: heap offsets are a fixed property of a build). */
    double *arena = NULL;
    if (posix_memalign((void **)&arena, 4096, total * sizeof(double)) != 0 || !arena) {
        free(p);
        return NULL;
    }
    memset(arena, 0, total * sizeof(double));
    p->raw = arena;
    p->scr = arena;
#if VW != 8
    p->stage_in = arena + (size_t)2 * BSLOT * VW;
    p->stage_out = p->stage_in + (size_t)VW * VOLD;
#endif

    /* Defaults, which the autotune only overrides by a clear margin.
     * All regimes: FUSED -- every default is now the exact configuration the
     * node itself picked and won with in panel_r7 (B=1 FUSED/plain 0.558,
     * the first movement in that cell in six rounds; B=64 FUSED+s0 0.588;
     * streaming FUSED+s0w 0.945/1.232, picked 6/6).  My own node-verified
     * LANEX2/LANEX3 configurations remain candidates so the tournament can
     * fall back if a regime disagrees. */
    const size_t ws = (size_t)batch * VOLD * sizeof(double) * 2;
    const int big = ws * 10 > (size_t)l3_bytes() * 9;
#if VW == 8
    p->mode = MODE_FUSED;
    p->nt = 0;
    p->pf = (batch == 1) ? PF_NONE : (big ? PF_SW : PF_S0);
#else
    p->mode = MODE_BATCH;
    p->nt = big;
    p->pf = (batch > 1) ? PF_S0 : PF_NONE;
#endif

#if defined(L8_FORCE_MODE)
    p->mode = L8_FORCE_MODE;
#  if defined(L8_FORCE_NT)
    p->nt = L8_FORCE_NT;
#  endif
#  if defined(L8_FORCE_PF)
    p->pf = L8_FORCE_PF;
#  endif
#else
    /* Round 9: B=1 runs NO tournament.  FUSED/plain is the driver-verified
     * fastest configuration in every FUSED-picked node run of r7 and r8
     * (0.5577/0.5647/0.5643/0.5647); the create()-arena picked LANEX3 in 4
     * of 6 r8 creates and every such driver run cost +5-6% (0.5935).  The
     * arena has twice been shown unable to rank near-ties (r7 B=64, r8 B=1),
     * so at this cell it is retired rather than hysteresis-patched. */
    if (batch > 1 || VW != 8) autotune(p);
#endif

    snprintf(g_desc, sizeof g_desc,
             "radix-8 split; pick[B=%d]: mode=%s nt=%d pf=%s%s",
             batch, mode_str(p->mode), p->nt, pf_str(p->pf),
             (batch == 1 && VW == 8) ? " (fixed, no tuner)" : "");
    return p;
}

void fft3d_destroy(fft3d_plan *plan)
{
    if (!plan) return;
    free(plan->raw);
    free(plan);
}
