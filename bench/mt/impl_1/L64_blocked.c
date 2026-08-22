/* Carried over from the SINGLE-THREAD competition, where this file finished as
 * written below. Your job in the multicore phase is to parallelise it across
 * 32 cores without losing its single-core efficiency -- read
 * ../PANEL_BRIEF.md, and read ../../geom/strategies/L64_blocked.md for the full
 * history of how this kernel got here.
 */
/* L64_blocked.c -- forward complex 3D DFT of a 64^3 cube, batched, out-of-place.
 *
 * ROUND panel_r11 (r10 node: st=3 landed -13.2%/-12.5%, B=1 is a TIE at
 * 952.9 vs 949.9; B=8 still -3.7% at 1311.5 vs 1262.9, pick cached pf9 st3
 * 2/3 with run 3 falling back to st0 pf2 -- the verdict's named bit-class
 * provenance exposure.  Three schedule/tuner changes, no arithmetic change:
 *   1. ONE-BIT-CLASS DEFAULT POOL: st=0 leaves the default tournament and
 *      becomes the untimed numerical reference only (resurrect via
 *      FFT64B_ST=0).  Every default candidate is now st=3 pw4, differing
 *      only in prefetch/store-opcode -- all outputs bit-identical, so a
 *      run-to-run pick flip can never again produce an unvalidated number
 *      (the r10 VERDICT 3(a) mitigation, executed).
 *   2. pro: PROLOGUE PREFETCH, adopted from L64_radix8 r9 (node-picked pro1
 *      at B=2/B=8 in 2/3 runs): burst-T1 input plane 0 at each volume's
 *      pass-1 start (the next-plane cursor covers planes 1..63 only, and a
 *      cross-volume tail prefetch is mistimed -- pass 2+3's two 4.5-MB SC
 *      sweeps run between issue and use), and burst-T1 SC slab ky=0 between
 *      pass 1 and the ky loop (slabpf covers ky+1 while z-lining ky, so
 *      slab 0 was always L3-cold).  Decided by a create-time A/B on the
 *      picked candidate (the rival's protocol), env FFT64B_PRO=0|1.
 *   3. Hysteresis 3% -> 1% (and NT-vs-cached bar 2%): the old 3% band was
 *      sized for cross-structure flips that no longer exist in the pool;
 *      on the node it discarded exactly the 1-2% slabpf/p1pf wins the rival
 *      banks (their B=1 pick carries both; my node B=1 pick was bare pf0).
 *      pf=5 (pfw only) joins the st3 pool -- the rival's batched winner is
 *      pfw+slabpf and pfw-alone was never offered.
 *
 * ROUND panel_r10 (r9 node: st=2 DECLINED 3/3 -- cached pf0 st0 in every cell,
 * B=1 1098.4 vs L64_radix8's 952.9; the verdict's instruction is verbatim
 * "L64_blocked should be judged on the split-complex rewrite it has deferred
 * for two rounds": st=1 and st=2 are both dead, and the rival's node numbers
 * are standing evidence that the interleaved layout's ~0.67M port-5 shuffles
 * per volume are NOT free even on one-FMA-pipe CLX):
 *   NEW st=3 "split-sc": split-complex currency, lanes = 8 adjacent z, the
 *   whole kernel shape adopted from L64_radix8 r6-r9 (attribution in the
 *   strategy record; the codelet lineage is L8_batchsimd r1 via L8_radix8):
 *     pass 1  per x-plane: y-FFT (64 = 8x8 DIT, elementwise across vectors),
 *             deinterleave fused into the stage-1 loads (2 vpermt2pd per
 *             point), split (re,im) vector pairs stored to the odd-line-
 *             padded scratch SC (row 17 lines, plane 1089 lines)
 *     pass 2+3 fused, per ky: x-FFT over the 64 planes IN PLACE in SC
 *             (loads at plane stride through the 8-KB line buffer, next-
 *             column prefetcht0 on every load), then the 64 z-lines of the
 *             now-L2-hot ky-slab: DFT8 across the 8 slot pairs, 14-vector-
 *             table lane twiddles, 8x8 transpose pair (2x24 shuffles),
 *             second DFT8, re/im interleave, contiguous 1-KB row store to
 *             out (cached / NT / +prefetchw, tuner-decided)
 *   Butterflies in split-complex carry ZERO shuffles; the only shuffles left
 *   are 65536 deinterleave + 4096*(48 transpose + 16 interleave) = 328K per
 *   volume, half the interleaved bill, at an unchanged FMA-port count.
 *   st=3 prefetch axes (tuner): p1pf (pass-1 next-plane T1, node-picked 3/3
 *   for the rival at B=1), slabpf (next ky-slab T1 during z-lines, node-
 *   picked in ALL rival cells r7-r9), pfw (out-row write-intent, rival's
 *   batched pick).  st=2 leaves the default tournament (env-only, like st=1).
 * ROUND panel_r6: first implementation for this geometry.
 * ROUND panel_r7: two changes, both adopted from L64_radix8's r6 record:
 *   1. mid lives in a 2 MB-hugepage mmap (MADV_HUGEPAGE + touch in create):
 *      the strided column walks otherwise touch ~1090 4-KB pages per sweep.
 *   2. A 2-SWEEP structure variant (st=1) joins the tuner: pass A does z+y
 *      only (planes in natural order, so the cold in-read is one sequential
 *      run), and a new pass B2 does the FULL two-stage 64-point x-FFT per
 *      (ky, z-column) directly out of mid -- 64 read streams at the padded
 *      (odd-line) plane stride, per-load prefetcht0 FFT64B_PFXC columns
 *      ahead (L64_radix8 measured that hint +12% at B=8), stores straight
 *      to out (full-line NT at PW=4, or cached).  This removes the x1 RMW
 *      pass entirely; the 3-sweep st=0 path is kept as candidates and the
 *      create-time tournament decides per {B, machine}.
 * ROUND panel_r9 (r8 node: still behind L64_radix8 in all cells, B=1 1092.6
 * vs 966.8; node picks cached/pf0/pf0/pf2, pfb took ZERO picks -- scratch-
 * read latency is not the gap; the rival's edge is STRUCTURAL: two sweeps,
 * with the last axis fused against L2-hot data):
 *   NEW st=2 "x-first" 2-sweep, adopted from L64_radix8's fused pass 2+3
 *   but with the axis ORDER inverted so the strided stage is never last:
 *     pass 1: x stage 1 DIRECTLY off the cold input -- DFT-8 across in
 *             planes {s, s+8, ..}, twiddle W64^{s*d}, into mid plane 8d+s
 *             (the separate z/y pass that used to carry the cold read is
 *             gone; this pass IS the in-read, 8 sequential streams)
 *     pass 2: per octet d: x stage 2 over 8 CONSECUTIVE mid planes into an
 *             octet buffer OB (8 padded planes, ~560 KB, reused across
 *             octets, so it recirculates in L2/L3), then y THEN z per
 *             completed output plane straight out of OB to out -- y first so
 *             the 64-row scatter lands on the L2-hot OB reads and the
 *             z-transpose-store emits PW sequential row streams to cold out.
 *   This deletes st=0's x1 RMW round trip (~8.9 MB of L2/L3 traffic per
 *   volume) -- the whole reason the rival's 2-sweep beats my 3-sweep on the
 *   node -- while keeping every loop at <=8 streams (my dead st=1 put the
 *   STRIDED pass last with 64 out-streams; the fix is the pass order, not
 *   the sweep count).  st=0 stays in the tournament; the node decides.
 * ROUND panel_r8 (first node numbers exist: r7 board has this file 5-13%
 * behind L64_radix8 in all three cells, node picks st0/cached/pf0):
 *   1. st=1 is DEAD on both machines (node kept st0 in every cell; r7's own
 *      words) -- its candidates leave the default tournament and are now
 *      generated only when FFT64B_ST/FFT64B_FORCE_ST asks for them.
 *   2. pf=2 (prefetchw on out) was gated to batch>=3, which silenced it at
 *      exactly the two cells where the deficit is largest (B=1, B=2) while
 *      L64_radix8's node tuner chose pfw at B=2 AND B=8.  Un-gated.
 *   3. New pf=5: prefetchw ONLY, without pf=1's paced T1 in-read (the node
 *      rejected pf=1 in every cell, so pf=2 may have lost on its read half).
 *   4. New pf=6/7: paced T0 prefetch over pass B's 8 mid read streams
 *      (+FFT64B_PFBL rows, issued at exactly consumption rate), the analog
 *      of L64_radix8's slabpf, which the node selected in ALL THREE cells --
 *      those reads always miss L2 at B>=1 on the node (mid is 4.46 MB) and
 *      the L2 streamer must retrain at every 4-KB boundary.  pf=6 is pfb
 *      alone, pf=7 = pfb + prefetchw; pfb is also admitted with NT stores.
 *
 * TECHNIQUE (see ../strategies/L64_blocked.md for the full derivation)
 *   Row-column 3D DFT on INTERLEAVED complex vectors whose lanes are PW
 *   consecutive z (the contiguous axis), so the y- and x-passes are
 *   shuffle-free and only the z-transform pays the one unavoidable
 *   transpose pair (proof in L36_pencilfused r1, adopted).
 *
 *   Every 64-point line is TWO RADIX-8 STAGES (64 = 8*8, DIT):
 *       X[8c+d] = sum_s W8^{sc} * ( W64^{sd} * sum_a W8^{ad} x[8a+s] )
 *   The 8-point module is the 4-mul/52-add radix-8 codelet (26 FMA-port ops
 *   + 5 swaps per PW lanes); the only irrational constant is 1/sqrt(2).
 *
 *   The volume is 4.19 MB -- it does NOT fit the scoring node's 1 MB L2, and
 *   at L=64 every natural stride is a power of two (the z-row is exactly one
 *   L1 way, the x-stride exactly one L2 way), so this file's charter is the
 *   cache-blocking + padding question.  Structure:
 *
 *   pass A, per x-plane (64x64 complex = 64 KB), planes visited in stride-8
 *   groups {r, r+8, ..., r+56}:
 *       z transform: lanes = PW y-rows via PWxPW complex-granule register
 *                    transposes on load AND store (both against the cheap
 *                    side), into a PADDED plane scratch P[y][kz]
 *       y transform: lanes = PW kz (contiguous in P), store to the PADDED
 *                    scratch volume mid[p][ky][kz]
 *       x stage 1:   after the group's 8 planes land in mid (~545 KB, still
 *                    L2/L3-warm), DFT-8 across the group IN PLACE on mid
 *                    with twiddles W64^{r*d}; 8 sequential read + 8
 *                    sequential write streams, never 64
 *   pass B:
 *       x stage 2:   DFT-8 over 8 CONSECUTIVE mid planes (one sequential
 *                    ~545 KB read run per octet), writing out[8c+d] through
 *                    8 sequential plane-streams (cached / +prefetchw / NT,
 *                    autotuned)
 *
 *   Splitting the x-transform's two radix-8 stages across the two passes is
 *   what kills the classic pathology: a monolithic x-pass needs 64 concurrent
 *   read streams of stride 64 KB (all one L2 set, unprefetchable); here no
 *   loop in the file ever runs more than 8 streams, all sequential.
 *
 *   PADDING (the stub's charter): mid's z-row stride is 68 complex = 17
 *   cache lines (odd) and its plane stride 4356 complex = 1089 lines (odd),
 *   so gcd(stride, sets) = 1 at both L1 and L2 and Bailey's single-set
 *   worst case cannot form on any scratch access.  in/out keep the driver's
 *   power-of-two layout but are only ever touched as <=8 sequential streams.
 *
 * ATTRIBUTION (what this file borrows, per the panel rules)
 *   - Interleaved-complex spectator-axis lanes, CMUL/DFT-with-swap idioms,
 *     PWxPW TRNC transpose, the #include-__FILE__ two-width template, and the
 *     "z-first transpose-on-load against the cold buffer" pass shape:
 *     L36_mixedradix r1 via L36_pfa / L36_pencilfused (r2-r5 records).
 *   - Paced T1 read-prefetch cursor, write-intent prefetchw on cold out
 *     streams, NTA read at consumption rate to protect L2 residency, and
 *     next-volume pre-coverage: L36_pfa r3-r6 (`pf` levels; ultimately
 *     L6_unrolled r3's prefetchw), constants rescaled to 64 KB planes.
 *   - Self-warming interleaved-rounds tuner with correctness interlock,
 *     physics gates and simplest-wins hysteresis: L36_pencilfused r5 +
 *     L36_pfa r4.
 *   - NT stores as a gated candidate, never a default (node rejected NT at
 *     L=36 three rounds running; wallaby loves it -- the tournament decides):
 *     L36_pencilfused r1/r4 evidence.
 *
 * OPERATION COUNT (per 64-point line over PW lanes, FMA-port vector ops)
 *   16 x DFT8 (26 ops + 5 swaps) + 49 twiddle CMUL (2 ops + 1 swap)
 *     = 514 FMA-port ops + 129 swaps
 *   Per volume: 3 * 1024 line-groups at PW=4 -> ~1.6M FMA-port ops; on the
 *   node's single 512-bit FMA pipe at 2.9 GHz that is ~550 us of port work.
 *   The binding constraint is L3: ~18 MB of scratch round trips per volume
 *   at 18.2 GB/s single-core -- this geometry is memory-shaped everywhere,
 *   which is why the schedule (prefetch pacing, stream counts) is the design.
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <sys/mman.h>
#ifdef __x86_64__
# include <immintrin.h>
#endif

#include "fft3d_api.h"

#ifndef L64B_ONCE               /* ============ COMMON, first pass ============ */
#define L64B_ONCE

#define L    64
#define LSQ  4096                    /* 64*64                                  */
#define VDBL ((size_t)2 * L * LSQ)   /* doubles per volume = 524288 (4.19 MB)  */

/* padded scratch strides, in complex units: both an ODD number of 64-B lines.
 * -DFFT64B_NOPAD builds the power-of-two-stride control (the Bailey worst
 * case this file's charter is to measure). */
#ifdef FFT64B_NOPAD
# define RS  64
# define PS  4096
#else
# define RS  68                      /* z-row stride: 68*16 = 1088 B = 17 lines */
# define PS  4356                    /* plane stride: 64*RS+4 = 69696 B = 1089  */
#endif
#define MIDDBL ((size_t)2 * L * PS)  /* doubles in the scratch volume          */
#define OBDBL  ((size_t)16 * PS)     /* st=2 octet buffer: 8 padded planes     */

#define CAT_(a,b) a##b
#define CAT(a,b)  CAT_(a,b)

/* pf=1/2: paced T1 read cursor -- one plane's worth of prefetches per plane
 * processed, spread over both pass-A subloops, aimed at the NEXT plane in
 * VISIT order (the group walks planes at stride 8, so a linear +32KB lead
 * would waste half its coverage on a plane we do not read next). */
#ifndef FFT64B_PFH
# define FFT64B_PFH 2                /* 3=T0 2=T1 1=T2 0=NTA */
#endif
/* pf=3/4: NTA read lead in doubles (4 KB), consumption-rate paced in the A1
 * subloop only; fills L1, bypasses L2 on SKX-class cores, so the in-read
 * stops evicting the group's mid planes before x-stage-1 needs them. */
#ifndef FFT64B_PFDN
# define FFT64B_PFDN 512
#endif
/* cache lines of the NEXT volume's input pre-covered per pass-B ky step */
#ifndef FFT64B_PFN
# define FFT64B_PFN 2
#endif
/* pf=2/3/5/7: write-intent prefetch lead on pass B's 8 out streams, in lines */
#ifndef FFT64B_PFWL
# define FFT64B_PFWL 4
#endif
/* st=2 tail: prefetchw lead on the z-store row streams, in lines.  Separate
 * knob so sweeping it cannot disturb st0's node-picked pf2 (B=8).  Wallaby
 * B=1 sweep, pf5-st2 in-arena: lead 4 -> 685-696, 8 -> 675-687,
 * 16 -> 669-681, 32 -> 675-705; 16 wins. */
#ifndef FFT64B_PFWL2
# define FFT64B_PFWL2 16
#endif
/* pf=6/7: pass-B mid-read T0 prefetch lead, in PADDED ROWS (17 lines each).
 * One line prefetched per line consumed (8 per zb step, one per plane
 * stream); 2 rows = ~2.2 KB lead per stream = well past the node's L3
 * latency at pass B's consumption rate. */
#ifndef FFT64B_PFBL
# define FFT64B_PFBL 2
#endif
/* st=1 pass B2: read-prefetch lead over mid's strided columns, in COLUMNS
 * (one column = one vector = one line at PW=4).  One prefetch per load,
 * adopted from L64_radix8 r6's next-column prefetcht0 (+12% at B=8 there);
 * a 2-column lead is ~2 FFT64V bodies ~ 400+ cycles, well past L3 latency. */
#ifndef FFT64B_PFXC
# define FFT64B_PFXC 2
#endif

/* W64^m = twre8[m] + i*(-twia8[m][even]); rows are pre-splatted vector forms:
 * twre8[m][j] = Re(W64^m) in every lane; twia8[m][j] = (-Im, +Im) alternating,
 * so cmul is 1 swap + 1 mul + 1 fma with two 64-B table loads.  Filled once
 * in fft3d_create() (libm in setup is allowed; execute only loads). */
static double twre8[64][8] __attribute__((aligned(64)));
static double twia8[64][8] __attribute__((aligned(64)));

/* st=3 (split-complex) tables: plain scalar cos/sin for the broadcast stage
 * twiddles, and the 14 z-line lane-twiddle vectors twz*[k2][l] = W64^{l*k2}. */
static double twc64[64], tws64[64];
static double twzr8[8][8] __attribute__((aligned(64)));
static double twzi8[8][8] __attribute__((aligned(64)));

static void fill_twiddles(void)
{
    for (int m = 0; m < 64; ++m) {
        double a  = -2.0 * M_PI * (double)m / 64.0;   /* forward: W = e^{-2pi i m/64} */
        double cr = cos(a), ci = sin(a);
        twc64[m] = cr; tws64[m] = ci;
        for (int j = 0; j < 8; ++j) {
            twre8[m][j] = cr;
            twia8[m][j] = (j & 1) ? ci : -ci;
        }
    }
    for (int k2 = 0; k2 < 8; ++k2)
        for (int l = 0; l < 8; ++l) {
            double b = -2.0 * M_PI * (double)(l * k2) / 64.0;
            twzr8[k2][l] = cos(b);
            twzi8[k2][l] = sin(b);
        }
}

enum { M_CACHED = 0, M_NT = 1 };

/* instantiate the kernel template at 256-bit, and at 512-bit where possible */
#define PW 2
#include __FILE__
#undef PW
#ifdef __AVX512F__
# define PW 4
# include __FILE__
# undef PW
# define HAVE_PW4 1
#endif

/* ==== st=3: split-complex fused 2-sweep (kernel shape from L64_radix8) ==== */
#ifdef HAVE_PW4

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

/* interleaved <-> split conversions: 1 vpermt2pd each */
#define DEIN_RE(A,B) VSH8(A,B, 0,2,4,6,8,10,12,14)
#define DEIN_IM(A,B) VSH8(A,B, 1,3,5,7,9,11,13,15)
#define ILV_LO(R,I)  VSH8(R,I, 0,8,1,9,2,10,3,11)
#define ILV_HI(R,I)  VSH8(R,I, 4,12,5,13,6,14,7,15)

/* split scratch strides, in DOUBLES: slot (x, ky, zb) = x*SCXS + ky*SCKS +
 * zb*16, re vector at +0 and im at +8 (each one full 64-B line).  Row 136
 * doubles = 17 lines, plane 64*136+8 = 8712 doubles = 1089 lines -- both odd,
 * same Bailey-proofing as the interleaved mid (and the rival's SC).  Total
 * 64*8712 = 557568 doubles = exactly MIDDBL, so st=3 reuses the hugepage
 * mapping (the OB region after it absorbs any prefetch overrun). */
#define SCKS 136
#define SCXS (64 * SCKS + 8)
_Static_assert((size_t)64 * SCXS <= MIDDBL + OBDBL,
               "st=3 split scratch must fit the shared hugepage mapping");

/* Forward split-complex radix-8, natural order, arrays of 8 (re,im) vector
 * pairs; 44 add/sub + 8 FMA, zero shuffles, only irrational constant 1/sqrt2.
 * All reads complete before the first write, so YR/YI may alias XR/XI.
 * Lineage: L8_batchsimd r1's FMA form via L8_radix8 / L64_radix8. */
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

/* (RR,II) *= (c + i*s), broadcast scalar twiddle: 2 mul + 2 FMA */
#define CTWS(RR, II, c, s) do {                                               \
    v8d cr_ = VSPL8(c), ci_ = VSPL8(s), t0_ = (RR);                           \
    (RR) = V8FNMA((II), ci_, t0_ * cr_);                                      \
    (II) = V8FMA ((II), cr_, t0_ * ci_);                                      \
} while (0)
/* vector-table form for the z-line lane twiddles */
#define CTWV(RR, II, TR, TI) do {                                             \
    v8d tr_ = (TR), ti_ = (TI), t0_ = (RR);                                   \
    (RR) = V8FNMA((II), ti_, t0_ * tr_);                                      \
    (II) = V8FMA ((II), tr_, t0_ * ti_);                                      \
} while (0)

/* 8x8 transpose of one v8d[8] block, 24 two-source shuffles (3 rounds of 8) */
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

/* 64-point split-complex line, two radix-8 stages through an 8-KB (re,im)
 * line buffer:  X[k2+8k1] = DFT8_{s}( W64^{s*k2} * DFT8_{t}( x[s+8t] ) ).
 * LDP(n, rr, ii) yields point n; STS(k, rr, ii) consumes output k, both in
 * natural order.  All LDP reads happen in stage 1, so in-place use is legal.
 * SHOOK(s) runs once per stage-1 group (prefetch hook). */
#define FFT64S(LDP, STS, SHOOK) do {                                          \
    v8d Hr_[64], Hi_[64];                                                     \
    _Pragma("GCC unroll 8")                                                   \
    for (int s_ = 0; s_ < 8; ++s_) {                                          \
        v8d xr_[8], xi_[8], yr_[8], yi_[8];                                   \
        SHOOK(s_);                                                            \
        _Pragma("GCC unroll 8")                                               \
        for (int t_ = 0; t_ < 8; ++t_) LDP(s_ + 8 * t_, xr_[t_], xi_[t_]);    \
        DFT8S(xr_, xi_, yr_, yi_);                                            \
        if (s_) {                                                             \
            _Pragma("GCC unroll 8")                                           \
            for (int d_ = 1; d_ < 8; ++d_)                                    \
                CTWS(yr_[d_], yi_[d_], twc64[s_ * d_], tws64[s_ * d_]);       \
        }                                                                     \
        _Pragma("GCC unroll 8")                                               \
        for (int d_ = 0; d_ < 8; ++d_) {                                      \
            Hr_[8 * d_ + s_] = yr_[d_]; Hi_[8 * d_ + s_] = yi_[d_];           \
        }                                                                     \
    }                                                                         \
    _Pragma("GCC unroll 8")                                                   \
    for (int d_ = 0; d_ < 8; ++d_) {                                          \
        v8d zr_[8], zi_[8];                                                   \
        DFT8S(Hr_ + 8 * d_, Hi_ + 8 * d_, zr_, zi_);                          \
        _Pragma("GCC unroll 8")                                               \
        for (int c_ = 0; c_ < 8; ++c_) STS(8 * c_ + d_, zr_[c_], zi_[c_]);    \
    }                                                                         \
} while (0)

#define SC_NOHOOK(s) do { } while (0)

/* pass 1, one x-plane: y-FFT, deinterleave fused into the stage-1 loads
 * (input rows read as 128-B chunks at 1-KB stride, L2-resident per plane),
 * split store to SC.  pn: next plane to T1-prefetch (p1pf), 16 lines per
 * stage-1 group = exactly one plane per plane processed, or NULL. */
static void sc_pass1(const double *restrict ip, double *restrict sp,
                     const double *restrict pn)
{
    for (int zb = 0; zb < 8; ++zb) {
#define SC_HOOK1(s) do {                                                      \
        if (pn) {                                                             \
            const double *h_ = pn + ((size_t)(zb * 8 + (s)) * 128);           \
            _Pragma("GCC unroll 16")                                          \
            for (int q_ = 0; q_ < 16; ++q_)                                   \
                __builtin_prefetch(h_ + 8 * q_, 0, 2);                        \
        }                                                                     \
    } while (0)
#define SC_LD1(n, rr, ii) do {                                                \
        const double *r1_ = ip + ((size_t)(n) * L + (size_t)zb * 8) * 2;      \
        v8d A1_ = *(const v8d *)r1_, B1_ = *(const v8d *)(r1_ + 8);           \
        (rr) = DEIN_RE(A1_, B1_); (ii) = DEIN_IM(A1_, B1_);                   \
    } while (0)
#define SC_ST1(k, rr, ii) do {                                                \
        double *q1_ = sp + (size_t)(k) * SCKS + (size_t)zb * 16;              \
        *(v8d *)q1_ = (rr); *(v8d *)(q1_ + 8) = (ii);                         \
    } while (0)
        FFT64S(SC_LD1, SC_ST1, SC_HOOK1);
#undef SC_LD1
#undef SC_ST1
#undef SC_HOOK1
    }
}

/* fused pass 2+3, one ky: x-FFT over the 64 planes IN PLACE in SC (all 64
 * loads at SCXS stride complete into the line buffer before the writeback,
 * so in-place is legal; next-column prefetcht0 on every load -- L64_radix8
 * measured that hint +12% at B=8), then the 64 z-lines of the ky-slab, which
 * the x-writeback just made L2-hot: DFT8 across slots (g -> k2), lane
 * twiddles W64^{l*k2}, 8x8 transpose pair, DFT8 (l -> k1), re/im interleave,
 * contiguous 1-KB row store to out.  slab: T1-prefetch ky+1's slab rows (17
 * lines per kx step); pfw: prefetchw the out row FFT64B_PFWL3 kx ahead. */
#ifndef FFT64B_PFWL3
# define FFT64B_PFWL3 4
#endif
static void sc_pass23(double *restrict SC, double *restrict out, int ky,
                      int nt, int pfw, int slab)
{
    double *kb = SC + (size_t)ky * SCKS;
    for (int zb = 0; zb < 8; ++zb) {
        double *cb = kb + (size_t)zb * 16;
#define SC_LDX(n, rr, ii) do {                                                \
        const double *cx_ = cb + (size_t)(n) * SCXS;                          \
        __builtin_prefetch(cx_ + 16, 0, 3);                                   \
        __builtin_prefetch(cx_ + 24, 0, 3);                                   \
        (rr) = *(const v8d *)cx_; (ii) = *(const v8d *)(cx_ + 8);             \
    } while (0)
#define SC_STX(k, rr, ii) do {                                                \
        double *cx_ = cb + (size_t)(k) * SCXS;                                \
        *(v8d *)cx_ = (rr); *(v8d *)(cx_ + 8) = (ii);                         \
    } while (0)
        FFT64S(SC_LDX, SC_STX, SC_NOHOOK);
#undef SC_LDX
#undef SC_STX
    }
    for (int kx = 0; kx < L; ++kx) {
        if (slab && ky < L - 1) {
            const double *sl_ = SC + (size_t)kx * SCXS + (size_t)(ky + 1) * SCKS;
            _Pragma("GCC unroll 17")
            for (int q_ = 0; q_ < 17; ++q_)
                __builtin_prefetch(sl_ + 8 * q_, 0, 2);
        }
        const double *zl = SC + (size_t)kx * SCXS + (size_t)ky * SCKS;
        v8d Vr[8], Vi[8], Ur[8], Ui[8], Tr[8], Ti[8], Wr[8], Wi[8];
        _Pragma("GCC unroll 8")
        for (int g = 0; g < 8; ++g) {
            Vr[g] = *(const v8d *)(zl + (size_t)g * 16);
            Vi[g] = *(const v8d *)(zl + (size_t)g * 16 + 8);
        }
        DFT8S(Vr, Vi, Ur, Ui);
        _Pragma("GCC unroll 8")
        for (int k2 = 1; k2 < 8; ++k2)
            CTWV(Ur[k2], Ui[k2],
                 *(const v8d *)twzr8[k2], *(const v8d *)twzi8[k2]);
        TR8(Ur, Tr);
        TR8(Ui, Ti);
        DFT8S(Tr, Ti, Wr, Wi);
        double *orow = out + ((size_t)kx * L + (size_t)ky) * L * 2;
        if (pfw && kx + FFT64B_PFWL3 < L) {
            double *pw_ = orow + (size_t)FFT64B_PFWL3 * (2 * LSQ);
            _Pragma("GCC unroll 16")
            for (int q_ = 0; q_ < 16; ++q_)
                __builtin_prefetch(pw_ + 8 * q_, 1, 3);
        }
        _Pragma("GCC unroll 8")
        for (int k1 = 0; k1 < 8; ++k1) {
            v8d lo = ILV_LO(Wr[k1], Wi[k1]), hi = ILV_HI(Wr[k1], Wi[k1]);
            double *d_ = orow + (size_t)k1 * 16;
            if (nt) {
                _mm512_stream_pd(d_,     (__m512d)lo);
                _mm512_stream_pd(d_ + 8, (__m512d)hi);
            } else {
                *(v8d *)d_ = lo; *(v8d *)(d_ + 8) = hi;
            }
        }
    }
}

/* st=3 pf decode: 0 none; 1 p1pf; 5 pfw; 6 slabpf; 7 slabpf+pfw;
 * 8 slabpf+p1pf; 9 slabpf+pfw+p1pf (the rival's node-picked B=1 combo is
 * slabpf+p1pf under plain stores, and slabpf+pfw batched).
 * pro (r11, from L64_radix8 r9): burst-T1 input plane 0 before each volume's
 * pass 1 (p1pf covers planes 1..63 only; its cross-volume tail fires two
 * full SC sweeps before use, i.e. mistimed) and SC slab ky=0 between pass 1
 * and the ky loop (slabpf covers ky+1 while z-lining ky; slab 0 was written
 * a whole SC sweep ago, L3-cold on the node).  Prefetch-only: bit-identical. */
static void run_vols_sc(const double *restrict in, double *restrict out,
                        double *restrict SC, int nvol, int mode, int pf,
                        int pro)
{
    const int p1  = (pf == 1 || pf == 8 || pf == 9);
    const int pw_ = (pf == 5 || pf == 7 || pf == 9) && mode == M_CACHED;
    const int sl  = (pf >= 6);
    const int nt  = (mode == M_NT);
    for (int b = 0; b < nvol; ++b) {
        const double *iv = in  + (size_t)b * VDBL;
        double       *ov = out + (size_t)b * VDBL;
        if (pro)                       /* input plane 0: 1024 T1 lines */
            for (int i = 0; i < 1024; ++i)
                __builtin_prefetch(iv + 8 * (size_t)i, 0, 2);
        for (int p = 0; p < L; ++p) {
            const double *pn = NULL;
            if (p1) pn = (p < L - 1) ? iv + (size_t)(p + 1) * (2 * LSQ)
                       : (b + 1 < nvol ? iv + VDBL : NULL);
            sc_pass1(iv + (size_t)p * (2 * LSQ), SC + (size_t)p * SCXS, pn);
        }
        if (pro)                       /* SC slab ky=0: 64 x 16 T1 lines */
            for (int x = 0; x < 64; ++x) {
                const double *q_ = SC + (size_t)x * SCXS;
                _Pragma("GCC unroll 16")
                for (int i = 0; i < 16; ++i)
                    __builtin_prefetch(q_ + 8 * i, 0, 2);
            }
        for (int ky = 0; ky < L; ++ky)
            sc_pass23(SC, ov, ky, nt, pw_, sl);
    }
}

#endif /* HAVE_PW4 (st=3) */

/* ---- plan, tuner, API ---------------------------------------------------- */

static const char *const mode_name[] = {"cached", "nt"};
static const char *const st_name[]   = {"3-sweep", "2-sweep", "x-first", "split-sc"};

struct fft3d_plan {
    int     batch;
    int     pw;                  /* 2 or 4                                  */
    int     mode;                /* M_CACHED / M_NT (final out stores)      */
    int     pf;                  /* st<=2: 0 none; 1 paced T1 read (+PFNX); */
                                 /* 2 = 1 + prefetchw out; 3 NTA read +     */
                                 /* prefetchw; 4 NTA read alone;            */
                                 /* 5 prefetchw only; 6 pass-B mid-read T0  */
                                 /* (pfb) only; 7 = pfb + prefetchw.        */
                                 /* st=3: 1 p1pf; 5 pfw; 6 slabpf;          */
                                 /* 7 slabpf+pfw; 8 slabpf+p1pf; 9 all      */
    int     st;                  /* 0 = 3-sweep (A+x1, B octets);           */
                                 /* 1 = 2-sweep (A z+y only, B2 full-x);    */
                                 /* 2 = x-first 2-sweep (x1 off in, octet   */
                                 /*     x2 -> OB, fused z+y OB -> out);     */
                                 /* 3 = split-complex fused 2-sweep         */
    int     pro;                 /* st=3 only: prologue prefetch (in plane 0
                                  * + SC slab 0), create-time A/B decided   */
    double *S;                   /* padded scratch volume, reused per batch */
    void   *rawS;
    size_t  map_bytes;           /* nonzero: rawS is an mmap of this size   */
};

const char *fft3d_name(void) { return "L64_blocked"; }

static char g_desc[224];

const char *fft3d_description(void)
{
    return g_desc[0] ? g_desc
                     : "64=8x8 two-stage radix-8; interleaved 3-sweep (st=0) "
                       "or split-complex fused 2-sweep (st=3, lanes=8z); "
                       "hugepage odd-line-padded scratch; "
                       "{pw,mode,pf,st} autotuned";
}
int fft3d_supports(int Lq) { return Lq == L; }

static void run_vols(int pw, int mode, int pf, int st, int pro, double *S,
                     const double *in, double *out, int nvol)
{
#ifdef HAVE_PW4
    if (st == 3)      run_vols_sc(in, out, S, nvol, mode, pf, pro);
    else if (pw == 4) run_vols_pw4(in, out, S, nvol, mode, pf, st);
    else
#endif
    { (void)pw; (void)pro; run_vols_pw2(in, out, S, nvol, mode, pf, st); }
#if defined(__SSE2__)
    if (mode == M_NT) _mm_sfence();
#endif
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* hysteresis rank: lower = simpler (adopted from L36_pfa r4).  Read-side
 * prefetch levels before store-side machinery, incumbent 3-sweep before
 * 2-sweep, cached before NT. */
static int cand_rank(int mode, int pf, int st)
{
    /* simplicity order: 0 < 1 < 4 < 6 < 5 < 2 < 7 < 3 < 8 < 9 (read-side
     * pacing before store-side machinery, single mechanisms before
     * combinations); structures: incumbent st0 first, then the split-complex
     * st3, then st2 (x-first), st1 last */
    static const int pfc[10] = {0, 1, 5, 7, 2, 4, 3, 6, 8, 9};
    static const int stc[4]  = {0, 3, 2, 1};
    return pfc[pf] * 8 + stc[st] * 2 + mode;
}

fft3d_plan *fft3d_create(int Lq, int batch)
{
    if (Lq != L || batch < 1) return NULL;
    fill_twiddles();
    fft3d_plan *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->batch = batch;
    /* mid on 2 MB hugepages (adopted from L64_radix8 r6): the strided sweeps
     * otherwise walk ~1090 4-KB pages; madvise BEFORE the faulting memset so
     * THP-madvise kernels back it synchronously.  Over-map by one hugepage so
     * the working base can be 2 MB-aligned; fall back to posix_memalign. */
    {
        const size_t hp = (size_t)2 << 20;
        size_t bytes = (MIDDBL + OBDBL) * sizeof(double);   /* mid + st=2 OB */
        size_t mb = ((bytes + hp - 1) & ~(hp - 1)) + hp;
        void *m = mmap(NULL, mb, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m != MAP_FAILED) {
            uintptr_t a = ((uintptr_t)m + hp - 1) & ~(uintptr_t)(hp - 1);
            p->rawS = m; p->map_bytes = mb; p->S = (double *)a;
#if defined(MADV_HUGEPAGE) && !defined(FFT64B_NOHP)
            madvise(m, mb, MADV_HUGEPAGE);   /* -DFFT64B_NOHP = control */
#endif
        } else if (posix_memalign(&p->rawS, 64, bytes) == 0) {
            p->map_bytes = 0; p->S = (double *)p->rawS;
        } else { free(p); return NULL; }
    }
    memset(p->S, 0, (MIDDBL + OBDBL) * sizeof(double));
#ifdef HAVE_PW4
    p->pw = 4; p->mode = M_CACHED; p->pf = 0; p->st = 3;   /* safe default */
#else
    p->pw = 2; p->mode = M_CACHED; p->pf = 1; p->st = 0;
#endif
    p->pro = 0;

    /* streaming gate: NT / prefetchw only make sense once the batch's in+out
     * footprint decisively leaves the node's 22 MB L3 (8.4 MB per volume) */
    const int streaming = ((size_t)batch * 2 * VDBL * sizeof(double) > (size_t)24 << 20);

    struct cand { int pw, mode, pf, st; } cands[40];
    int nc = 0;
    /* r11: ONE bit class in the default pool.  On AVX-512 builds every
     * default candidate is st=3 pw4; the rows differ only in prefetch level
     * and store opcode, so all outputs are bit-identical and a run-to-run
     * pick flip can never produce a leaderboard number whose bit class the
     * checked run did not validate (r10 VERDICT 3(a)).  st=0 becomes the
     * untimed numerical reference; FFT64B_ST=0 resurrects its rows.
     * st=3 pf: 1 p1pf, 5 pfw, 6 slabpf, 7 slabpf+pfw, 8 slabpf+p1pf, 9 all.
     * pf=5 is new to the pool (the rival's batched winner is pfw+slabpf;
     * pfw-alone had never been offered).  NT is NOT streaming-gated (out is
     * written exactly once by the z tail -- structural). */
#ifdef HAVE_PW4
    { static const int pfs_s3[] = {0, 1, 5, 6, 8, 7, 9};
      for (int i = 0; i < 7; ++i)
          cands[nc++] = (struct cand){4, M_CACHED, pfs_s3[i], 3};
      cands[nc++] = (struct cand){4, M_NT, 0, 3};
      cands[nc++] = (struct cand){4, M_NT, 6, 3};
    }
#else
    /* no AVX-512: st=0 pw2 is the only implemented path; its r8-r9 shortlist */
    static const int pfs_c[] = {0, 1, 2, 5};
    for (int i = 0; i < 4; ++i)
        cands[nc++] = (struct cand){2, M_CACHED, pfs_c[i], 0};
    if (streaming)
        for (int i = 0; i < 2; ++i)             /* pf = 0, 1 */
            cands[nc++] = (struct cand){2, M_NT, i, 0};
#endif
    /* st=0 (interleaved 3-sweep, node incumbent until r10), st=1 and st=2
     * (both dead on the node, r7/r9): candidates only when the monitor
     * explicitly asks via FFT64B_ST=0|1|2 / -DFFT64B_FORCE_ST. */
    {
        const char *est = getenv("FFT64B_ST");
        int want_st0 = (est && est[0] && atoi(est) == 0);
        int want_st1 = (est && atoi(est) == 1);
        int want_st2 = (est && atoi(est) == 2);
#ifdef FFT64B_FORCE_ST
        want_st0 |= (FFT64B_FORCE_ST == 0);
        want_st1 |= (FFT64B_FORCE_ST == 1);
        want_st2 |= (FFT64B_FORCE_ST == 2);
#endif
#ifdef HAVE_PW4
        if (want_st0) {
            static const int pfs_c[] = {0, 1, 2, 5};
            for (int i = 0; i < 4; ++i) {
                cands[nc++] = (struct cand){2, M_CACHED, pfs_c[i], 0};
                cands[nc++] = (struct cand){4, M_CACHED, pfs_c[i], 0};
            }
            if (streaming)
                for (int i = 0; i < 2; ++i) {
                    cands[nc++] = (struct cand){2, M_NT, i, 0};
                    cands[nc++] = (struct cand){4, M_NT, i, 0};
                }
        }
#else
        (void)want_st0;                          /* st0 already the pool */
#endif
        if (want_st1) {
#ifdef HAVE_PW4
            for (int pf = 0; pf <= 1; ++pf) {
                cands[nc++] = (struct cand){4, M_CACHED, pf, 1};
                cands[nc++] = (struct cand){4, M_NT, pf, 1};
            }
#else
            cands[nc++] = (struct cand){2, M_CACHED, 0, 1};
            cands[nc++] = (struct cand){2, M_CACHED, 1, 1};
#endif
        }
        if (want_st2) {
#ifdef HAVE_PW4
            { static const int pfs_s2[] = {0, 5, 6, 7};
              for (int i = 0; i < 4; ++i)
                  cands[nc++] = (struct cand){4, M_CACHED, pfs_s2[i], 2};
              cands[nc++] = (struct cand){4, M_NT, 0, 2};
              cands[nc++] = (struct cand){4, M_NT, 6, 2};
            }
#endif
            cands[nc++] = (struct cand){2, M_CACHED, 0, 2};
            cands[nc++] = (struct cand){2, M_CACHED, 6, 2};
        }
    }

    /* run-time forcing for the monitor's control jobs */
    { const char *e;
      if ((e = getenv("FFT64B_PW"))) {
          int v = atoi(e), w = 0;
          for (int c = 0; c < nc; ++c) if (cands[c].pw == v) cands[w++] = cands[c];
          if (w) nc = w;
      }
      if ((e = getenv("FFT64B_MODE"))) {
          int num = (e[0] >= '0' && e[0] <= '9'), w = 0;
          for (int c = 0; c < nc; ++c)
              if (num ? cands[c].mode == atoi(e)
                      : !strcmp(mode_name[cands[c].mode], e)) cands[w++] = cands[c];
          if (w) nc = w;
      }
      if ((e = getenv("FFT64B_PF"))) {
          int v = atoi(e), w = 0;
          for (int c = 0; c < nc; ++c) if (cands[c].pf == v) cands[w++] = cands[c];
          if (w) nc = w;
      }
      if ((e = getenv("FFT64B_ST"))) {
          int v = atoi(e), w = 0;
          for (int c = 0; c < nc; ++c) if (cands[c].st == v) cands[w++] = cands[c];
          if (w) nc = w;
      } }
#ifdef FFT64B_FORCE_PW
    { int w = 0;
      for (int c = 0; c < nc; ++c) if (cands[c].pw == FFT64B_FORCE_PW) cands[w++] = cands[c];
      if (w) nc = w; }
#endif
#ifdef FFT64B_FORCE_MODE
    { int w = 0;
      for (int c = 0; c < nc; ++c) if (cands[c].mode == FFT64B_FORCE_MODE) cands[w++] = cands[c];
      if (w) nc = w; }
#endif
#ifdef FFT64B_FORCE_PF
    { int w = 0;
      for (int c = 0; c < nc; ++c) if (cands[c].pf == FFT64B_FORCE_PF) cands[w++] = cands[c];
      if (w) nc = w; }
#endif
#ifdef FFT64B_FORCE_ST
    { int w = 0;
      for (int c = 0; c < nc; ++c) if (cands[c].st == FFT64B_FORCE_ST) cands[w++] = cands[c];
      if (w) nc = w; }
#endif

    /* tuning arena: 8 volumes = 67 MB in+out streams past every L3 involved */
    const int nv = batch < 8 ? batch : 8;
    void *ri = NULL, *ro = NULL, *rr = NULL;
    if (posix_memalign(&ri, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&ro, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&rr, 64, (size_t)nv * VDBL * sizeof(double))) {
        free(ri); free(ro);
        p->pw = cands[0].pw; p->mode = cands[0].mode;
        p->pf = cands[0].pf; p->st = cands[0].st;
        { const char *ep = getenv("FFT64B_PRO");
          if (ep && p->st == 3) p->pro = (atoi(ep) != 0); }
        snprintf(g_desc, sizeof g_desc,
                 "L64 8x8 blocked; tuner SKIPPED (arena alloc failed): "
                 "pw=%d mode=%s pf=%d st=%d pro=%d",
                 p->pw, mode_name[p->mode], p->pf, p->st, p->pro);
        return p;
    }
    double *tin = ri, *tout = ro, *ref = rr;
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        tin[i] = (double)(long long)(s >> 11) * 0x1p-53;
    }

    /* numerical reference: the interleaved st=0 path, which no longer
     * competes -- so the interlock still cross-checks two INDEPENDENT
     * kernels even though the timed pool is one bit class. */
#ifdef HAVE_PW4
    run_vols(4, M_CACHED, 0, 0, 0, p->S, tin, ref, nv);
#else
    run_vols(2, M_CACHED, 0, 0, 0, p->S, tin, ref, nv);
#endif

    int    ok[40];
    double tc[40];
    for (int c = 0; c < nc; ++c) {
        run_vols(cands[c].pw, cands[c].mode, cands[c].pf, cands[c].st, 0, p->S, tin, tout, nv);
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
            double d = tout[i] - ref[i];
            num += d * d; den += ref[i] * ref[i];
        }
        ok[c] = (num <= den * 1e-26);       /* rel L2 < 1e-13 vs reference */
        tc[c] = 1e300;
    }
    const int R = (nv >= 4) ? 1 : (nv >= 2 ? 3 : 4);
    for (int round = 0; round < 4; ++round)
        for (int c = 0; c < nc; ++c) {
            if (!ok[c]) continue;
            /* self-warming: one untimed exec so each candidate is timed from
             * its OWN steady-state cache (L36_pencilfused r5's 86% phantom-
             * penalty fix: an NT candidate flushes tout and poisons whoever
             * runs next in the rotation). */
            run_vols(cands[c].pw, cands[c].mode, cands[c].pf, cands[c].st, 0, p->S, tin, tout, nv);
            double t0 = now_s();
            for (int r = 0; r < R; ++r)
                run_vols(cands[c].pw, cands[c].mode, cands[c].pf, cands[c].st, 0, p->S, tin, tout, nv);
            double t = (now_s() - t0) / R;
            if (t < tc[c]) tc[c] = t;
        }
    int best = -1;
    for (int c = 0; c < nc; ++c)
        if (ok[c] && (best < 0 || tc[c] < tc[best])) best = c;
    if (best >= 0) {
        int pick = best;
        /* NT must beat the best cached candidate by 2% (a store-mode flip
         * moves traffic, so a noise-level win must not take it -- the
         * L64_radix8 r6 bar; the node has never picked NT at L=64). */
        if (cands[pick].mode == M_NT) {
            int bc = -1;
            for (int c = 0; c < nc; ++c)
                if (ok[c] && cands[c].mode == M_CACHED &&
                    (bc < 0 || tc[c] < tc[bc])) bc = c;
            if (bc >= 0 && tc[bc] <= tc[pick] * 1.02) pick = bc;
        }
        /* simplest-wins hysteresis, 3% -> 1% in r11: the pool is now one bit
         * class differing only in prefetch level, so a "complex" pick risks
         * nothing -- while the 3% band was discarding exactly the 1-2%
         * slabpf/p1pf wins the rival banks on the node (its B=1 pick carries
         * both; my r10 node B=1 pick was bare pf0). */
        for (int c = 0; c < nc; ++c)
            if (ok[c] && cands[c].mode == cands[pick].mode &&
                cands[c].st == cands[pick].st && tc[c] <= tc[pick] * 1.01 &&
                cand_rank(cands[c].mode, cands[c].pf, cands[c].st) <
                cand_rank(cands[pick].mode, cands[pick].pf, cands[pick].st))
                pick = c;
        p->pw = cands[pick].pw; p->mode = cands[pick].mode;
        p->pf = cands[pick].pf; p->st = cands[pick].st;
    }

    /* pro A/B on the picked candidate (L64_radix8 r9's propf protocol):
     * 3 interleaved rounds, self-warmed, min per side, strict win to turn
     * on.  Prefetch-only, so it cannot change the output bit class. */
    double t_pro[2] = {1e300, 1e300};
#ifdef HAVE_PW4
    if (p->st == 3) {
        const char *ep = getenv("FFT64B_PRO");
        if (ep) p->pro = (atoi(ep) != 0);
        else {
            for (int round = 0; round < 3; ++round)
                for (int side = 0; side < 2; ++side) {
                    run_vols(p->pw, p->mode, p->pf, p->st, side, p->S, tin, tout, nv);
                    double t0 = now_s();
                    for (int r = 0; r < R; ++r)
                        run_vols(p->pw, p->mode, p->pf, p->st, side, p->S, tin, tout, nv);
                    double t = (now_s() - t0) / R;
                    if (t < t_pro[side]) t_pro[side] = t;
                }
            p->pro = (t_pro[1] < t_pro[0]);
        }
    }
#endif

    snprintf(g_desc, sizeof g_desc,
             "L64 8x8 two-stage, hugepage odd-line-padded scratch; "
             "tuner pick: pw=%d mode=%s pf=%d st=%d(%s) pro=%d (B=%d, nv=%d)",
             p->pw, mode_name[p->mode], p->pf, p->st,
             st_name[p->st], p->pro, batch, nv);

    if (getenv("FFT64B_VERBOSE")) {
        for (int c = 0; c < nc; ++c)
            fprintf(stderr, "L64_blocked tuner: pw=%d mode=%-7s pf=%d st=%d  %s  %.1f us/vol\n",
                    cands[c].pw, mode_name[cands[c].mode], cands[c].pf, cands[c].st,
                    ok[c] ? "ok " : "BAD", ok[c] ? tc[c] * 1e6 / nv : 0.0);
        if (t_pro[0] < 1e300)
            fprintf(stderr, "L64_blocked tuner: pro A/B off %.1f / on %.1f us/vol\n",
                    t_pro[0] * 1e6 / nv, t_pro[1] * 1e6 / nv);
        fprintf(stderr, "L64_blocked tuner: chose pw=%d mode=%s pf=%d st=%d pro=%d (nv=%d)\n",
                p->pw, mode_name[p->mode], p->pf, p->st, p->pro, nv);
    }
    free(ri); free(ro); free(rr);
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    if (p->map_bytes) munmap(p->rawS, p->map_bytes);
    else              free(p->rawS);
    free(p);
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    run_vols(plan->pw, plan->mode, plan->pf, plan->st, plan->pro, plan->S,
             (const double *)in, (double *)out, plan->batch);
}

#else /* ================= KERNEL TEMPLATE, PW = 2 or 4 ==================== */

#define vec   CAT(vec_pw,  PW)
#define veci  CAT(veci_pw, PW)
#define FN(n) CAT(n, CAT(_pw, PW))

typedef double    vec  __attribute__((vector_size(PW * 16)));
typedef long long veci __attribute__((vector_size(PW * 16)));

#ifdef __clang__
# define VSH(a,b,...) __builtin_shufflevector(a, b, __VA_ARGS__)
#else
# define VSH(a,b,...) __builtin_shuffle(a, b, (veci){__VA_ARGS__})
#endif

#define NV (L / PW)                  /* z-vectors per 64-complex row: 32 or 16 */
#define RSV (RS / PW)                /* padded P-buffer row stride in vecs     */

#if PW == 4
# define VSPLAT(a)  ((vec){(a),(a),(a),(a),(a),(a),(a),(a)})
# define VPAIR(a,b) ((vec){(a),(b),(a),(b),(a),(b),(a),(b)})
# define SWAP(v)    VSH((v),(v), 1,0,3,2,5,4,7,6)
# define VFMA(a,b,c)  ((vec)_mm512_fmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
# define VFNMA(a,b,c) ((vec)_mm512_fnmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
# define STREAM_ST(p,v) _mm512_stream_pd((p), (__m512d)(v))
#else
# define VSPLAT(a)  ((vec){(a),(a),(a),(a)})
# define VPAIR(a,b) ((vec){(a),(b),(a),(b)})
# define SWAP(v)    VSH((v),(v), 1,0,3,2)
# ifdef __FMA__
#  define VFMA(a,b,c)  ((vec)_mm256_fmadd_pd((__m256d)(a),(__m256d)(b),(__m256d)(c)))
#  define VFNMA(a,b,c) ((vec)_mm256_fnmadd_pd((__m256d)(a),(__m256d)(b),(__m256d)(c)))
# else
#  define VFMA(a,b,c)  ((a)*(b) + (c))
#  define VFNMA(a,b,c) ((c) - (a)*(b))
# endif
# ifdef __AVX__
#  define STREAM_ST(p,v) _mm256_stream_pd((p), (__m256d)(v))
# else
#  define STREAM_ST(p,v) (*(vec *)(p) = (v))
# endif
#endif

/* PW x PW transpose of 128-bit complex granules (from L36_pfa, verbatim) */
#if PW == 4
# define TRNC(r, c) do {                                                     \
    vec u0_ = VSH((r)[0], (r)[1], 0,1,8,9,4,5,12,13);                        \
    vec u1_ = VSH((r)[0], (r)[1], 2,3,10,11,6,7,14,15);                      \
    vec u2_ = VSH((r)[2], (r)[3], 0,1,8,9,4,5,12,13);                        \
    vec u3_ = VSH((r)[2], (r)[3], 2,3,10,11,6,7,14,15);                      \
    (c)[0] = VSH(u0_, u2_, 0,1,2,3,8,9,10,11);                               \
    (c)[2] = VSH(u0_, u2_, 4,5,6,7,12,13,14,15);                             \
    (c)[1] = VSH(u1_, u3_, 0,1,2,3,8,9,10,11);                               \
    (c)[3] = VSH(u1_, u3_, 4,5,6,7,12,13,14,15);                             \
} while (0)
#else
# define TRNC(r, c) do {                                                     \
    vec u0_ = VSH((r)[0], (r)[1], 0,1,4,5);                                  \
    vec u1_ = VSH((r)[0], (r)[1], 2,3,6,7);                                  \
    (c)[0] = u0_; (c)[1] = u1_;                                              \
} while (0)
#endif

#define PMV VPAIR(1.0, -1.0)
#define CSV VSPLAT(0.70710678118654752440084436210485)

/* pre-splatted twiddle table rows (first 2*PW doubles of the 8-double rows) */
#define TRE(m) (*(const vec *)&twre8[m][0])
#define TIA(m) (*(const vec *)&twia8[m][0])
/* v * W64^m: 1 swap + 1 mul + 1 fma, two table loads */
#define CMULT(v, m) VFMA((v), TRE(m), SWAP(v) * TIA(m))

/* Forward 8-point DFT, natural in/out order, inputs may be memory refs (all
 * reads happen in the first two statement rows).  26 FMA-port ops + 5 swaps. */
#define DFT8M(x0,x1,x2,x3,x4,x5,x6,x7, y0,y1,y2,y3,y4,y5,y6,y7) do {         \
    vec a0_=(x0)+(x4), a1_=(x0)-(x4), a2_=(x2)+(x6), a3_=(x2)-(x6);          \
    vec b0_=(x1)+(x5), b1_=(x1)-(x5), b2_=(x3)+(x7), b3_=(x3)-(x7);          \
    vec sE_ = SWAP(a3_), sO_ = SWAP(b3_);                                    \
    vec E0_=a0_+a2_, E2_=a0_-a2_;                                            \
    vec E1_=VFMA(sE_, PMV, a1_), E3_=VFNMA(sE_, PMV, a1_);                   \
    vec O0_=b0_+b2_, O2_=b0_-b2_;                                            \
    vec O1_=VFMA(sO_, PMV, b1_), O3_=VFNMA(sO_, PMV, b1_);                   \
    vec s2_ = SWAP(O2_);                                                     \
    vec q0_ = VFMA (SWAP(O1_), PMV, O1_);        /* (1-i)*O1 */              \
    vec q1_ = VFNMA(SWAP(O3_), PMV, O3_);        /* (1+i)*O3 */              \
    (y0) = E0_ + O0_;               (y4) = E0_ - O0_;                        \
    (y2) = VFMA (s2_, PMV, E2_);    (y6) = VFNMA(s2_, PMV, E2_);             \
    (y1) = VFMA (q0_, CSV, E1_);    (y5) = VFNMA(q0_, CSV, E1_);             \
    (y3) = VFNMA(q1_, CSV, E3_);    (y7) = VFMA (q1_, CSV, E3_);             \
} while (0)

/* The 64-point line as two radix-8 stages with W64^{s*d} between them.
 * LD(n) yields input element n; ST(k, v) consumes output element k, both in
 * NATURAL order.  All LD reads complete inside stage 1, so LD/ST may alias. */
#define FFT64V(LD, ST) do {                                                  \
    vec H_[64];                              /* H_[8d+s] = W64^{sd} G_s[d] */\
    _Pragma("GCC unroll 8")                                                  \
    for (int s_ = 0; s_ < 8; ++s_) {                                         \
        vec y0_,y1_,y2_,y3_,y4_,y5_,y6_,y7_;                                 \
        DFT8M(LD(s_),LD(8+s_),LD(16+s_),LD(24+s_),                           \
              LD(32+s_),LD(40+s_),LD(48+s_),LD(56+s_),                       \
              y0_,y1_,y2_,y3_,y4_,y5_,y6_,y7_);                              \
        if (s_) {                                                            \
            y1_ = CMULT(y1_, 1*s_); y2_ = CMULT(y2_, 2*s_);                  \
            y3_ = CMULT(y3_, 3*s_); y4_ = CMULT(y4_, 4*s_);                  \
            y5_ = CMULT(y5_, 5*s_); y6_ = CMULT(y6_, 6*s_);                  \
            y7_ = CMULT(y7_, 7*s_);                                          \
        }                                                                    \
        H_[     s_] = y0_; H_[ 8 + s_] = y1_; H_[16 + s_] = y2_;             \
        H_[24 + s_] = y3_; H_[32 + s_] = y4_; H_[40 + s_] = y5_;             \
        H_[48 + s_] = y6_; H_[56 + s_] = y7_;                                \
    }                                                                        \
    _Pragma("GCC unroll 8")                                                  \
    for (int d_ = 0; d_ < 8; ++d_) {                                         \
        vec z0_,z1_,z2_,z3_,z4_,z5_,z6_,z7_;                                 \
        DFT8M(H_[8*d_  ],H_[8*d_+1],H_[8*d_+2],H_[8*d_+3],                   \
              H_[8*d_+4],H_[8*d_+5],H_[8*d_+6],H_[8*d_+7],                   \
              z0_,z1_,z2_,z3_,z4_,z5_,z6_,z7_);                              \
        ST(     d_, z0_); ST( 8 + d_, z1_); ST(16 + d_, z2_);                \
        ST(24 + d_, z3_); ST(32 + d_, z4_); ST(40 + d_, z5_);                \
        ST(48 + d_, z6_); ST(56 + d_, z7_);                                  \
    }                                                                        \
} while (0)

/* pass-A pacing: one plane (2*LSQ doubles) of prefetches per plane processed,
 * spread over the 2*NV subloop iterations, aimed at pfnext (the next plane in
 * VISIT order, which is 8 planes away in memory). */
#define PFSTEP (LSQ * PW / L)        /* = LSQ*PW/64 doubles per iteration */
#define PFA1(p) do {                                                         \
    _Pragma("GCC unroll 32")                                                 \
    for (int q_ = 0; q_ < PFSTEP / 8; ++q_)                                  \
        __builtin_prefetch((p) + 8 * q_, 0, FFT64B_PFH);                     \
} while (0)
/* NTA at consumption rate: the A1 subloop reads 2*PFSTEP doubles per
 * iteration; constant lead FFT64B_PFDN, nothing issued in A2. */
#define PFA1_NTA(p) do {                                                     \
    _Pragma("GCC unroll 64")                                                 \
    for (int q_ = 0; q_ < PFSTEP / 4; ++q_)                                  \
        __builtin_prefetch((p) + 8 * q_, 0, 0);                              \
} while (0)

/* pass A for ONE x-plane p: z transform (transpose pair, both sides against
 * the L1 data) then y transform, in[p] -> mid[p].  Sequential cold reads;
 * pfr: 0 none, 1 paced T1 at pfnext, 2 NTA at consumption rate. */
static void FN(passA_plane)(const double *restrict in, double *restrict mid,
                            int p, const double *pfnext, int pfr)
{
    vec P_[L * RSV];                 /* padded plane scratch P[y][kz], ~70 KB */
    const double *px  = in  + (size_t)p * (2 * LSQ);
    double       *mx  = mid + (size_t)p * (2 * PS);
    const double *pfc = pfnext;
    const double *pfn = px + FFT64B_PFDN;

    for (int yb = 0; yb < L; yb += PW) {
        if (pfr == 1 && pfc) { PFA1(pfc); pfc += PFSTEP; }
        else if (pfr == 2)   { PFA1_NTA(pfn); pfn += 2 * PFSTEP; }
        vec Zv[64];
        _Pragma("GCC unroll 32")
        for (int zb = 0; zb < NV; ++zb) {
            vec r_[PW];
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j)
                r_[j] = *(const vec *)(px + ((size_t)(yb + j) * L + (size_t)zb * PW) * 2);
            TRNC(r_, &Zv[zb * PW]);
        }
#define LDZ(n)    Zv[n]
#define STZ(k, v) (Zv[k] = (v))
        FFT64V(LDZ, STZ);
#undef LDZ
#undef STZ
        _Pragma("GCC unroll 32")
        for (int kb = 0; kb < NV; ++kb) {
            vec r_[PW];
            TRNC(&Zv[kb * PW], r_);
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j)
                P_[(size_t)(yb + j) * RSV + kb] = r_[j];
        }
    }

    for (int zb = 0; zb < NV; ++zb) {
        if (pfr == 1 && pfc) { PFA1(pfc); pfc += PFSTEP; }
#define LDY(n)    P_[(size_t)(n) * RSV + zb]
#define STY(k, v) (*(vec *)(mx + ((size_t)(k) * RS + (size_t)zb * PW) * 2) = (v))
        FFT64V(LDY, STY);
#undef LDY
#undef STY
    }
}

/* x stage 1, group r: DFT-8 across planes {r, r+8, ..., r+56} of mid with
 * twiddles W64^{r*d}, IN PLACE (all 8 loads precede the first store).
 * 8 sequential read streams + 8 sequential write streams at padded stride. */
#define X1BODY(TWIDDLE) do {                                                 \
    for (int ky = 0; ky < L; ++ky)                                           \
        for (int zb = 0; zb < NV; ++zb) {                                    \
            double *b_ = mid + (size_t)r * (2 * PS)                          \
                             + ((size_t)ky * RS + (size_t)zb * PW) * 2;      \
            if (pfb) {  /* r8: write-intent, the group RMWs in place; on    \
                         * the node the 545-KB group half-misses L2 */       \
                _Pragma("GCC unroll 8")                                      \
                for (int c_ = 0; c_ < 8; ++c_)                               \
                    __builtin_prefetch(b_ + (size_t)c_ * (8 * 2 * PS)        \
                                          + FFT64B_PFBL * (2 * RS), 1, 3);   \
            }                                                                \
            vec v0_ = *(const vec *)(b_             );                       \
            vec v1_ = *(const vec *)(b_ +  8 * 2 * PS);                      \
            vec v2_ = *(const vec *)(b_ + 16 * 2 * PS);                      \
            vec v3_ = *(const vec *)(b_ + 24 * 2 * PS);                      \
            vec v4_ = *(const vec *)(b_ + 32 * 2 * PS);                      \
            vec v5_ = *(const vec *)(b_ + 40 * 2 * PS);                      \
            vec v6_ = *(const vec *)(b_ + 48 * 2 * PS);                      \
            vec v7_ = *(const vec *)(b_ + 56 * 2 * PS);                      \
            vec g0_,g1_,g2_,g3_,g4_,g5_,g6_,g7_;                             \
            DFT8M(v0_,v1_,v2_,v3_,v4_,v5_,v6_,v7_,                           \
                  g0_,g1_,g2_,g3_,g4_,g5_,g6_,g7_);                          \
            TWIDDLE;                                                         \
            *(vec *)(b_             ) = g0_;                                 \
            *(vec *)(b_ +  8 * 2 * PS) = g1_;                                \
            *(vec *)(b_ + 16 * 2 * PS) = g2_;                                \
            *(vec *)(b_ + 24 * 2 * PS) = g3_;                                \
            *(vec *)(b_ + 32 * 2 * PS) = g4_;                                \
            *(vec *)(b_ + 40 * 2 * PS) = g5_;                                \
            *(vec *)(b_ + 48 * 2 * PS) = g6_;                                \
            *(vec *)(b_ + 56 * 2 * PS) = g7_;                                \
        }                                                                    \
} while (0)

static void FN(x1_group)(double *restrict mid, int r, int pfb)
{
    if (r == 0) {
        X1BODY((void)0);
    } else {
        /* twiddle vectors are loop-invariant per group: hoisted here */
        const vec t1r = TRE(1*r), t1i = TIA(1*r), t2r = TRE(2*r), t2i = TIA(2*r);
        const vec t3r = TRE(3*r), t3i = TIA(3*r), t4r = TRE(4*r), t4i = TIA(4*r);
        const vec t5r = TRE(5*r), t5i = TIA(5*r), t6r = TRE(6*r), t6i = TIA(6*r);
        const vec t7r = TRE(7*r), t7i = TIA(7*r);
        X1BODY(do {
            g1_ = VFMA(g1_, t1r, SWAP(g1_) * t1i);
            g2_ = VFMA(g2_, t2r, SWAP(g2_) * t2i);
            g3_ = VFMA(g3_, t3r, SWAP(g3_) * t3i);
            g4_ = VFMA(g4_, t4r, SWAP(g4_) * t4i);
            g5_ = VFMA(g5_, t5r, SWAP(g5_) * t5i);
            g6_ = VFMA(g6_, t6r, SWAP(g6_) * t6i);
            g7_ = VFMA(g7_, t7r, SWAP(g7_) * t7i);
        } while (0));
    }
}
#undef X1BODY

/* x stage 2, octet d: DFT-8 over the 8 CONSECUTIVE mid planes {8d..8d+7}
 * (one ~545 KB sequential read run), outputs to out planes {d, 8+d, ..}
 * through 8 sequential plane-streams.  nt: stream stores; pfw: write-intent
 * prefetch FFT64B_PFWL lines ahead on the 8 cold out streams; pfb: T0
 * prefetch FFT64B_PFBL rows ahead on the 8 mid read streams, one line per
 * line consumed (r8, modeled on L64_radix8's slabpf: these reads always
 * miss L2 on the node and the L2 streamer retrains at every 4-KB boundary);
 * nx: next volume's in, pre-covered FFT64B_PFN lines per ky step. */
static void FN(passB_group)(const double *restrict mid, double *restrict out,
                            int d, int nt, int pfw, int pfb, const double *nx)
{
    const double *gb = mid + (size_t)(8 * d) * (2 * PS);
    double       *ob = out + (size_t)d * (2 * LSQ);
    for (int ky = 0; ky < L; ++ky) {
        if (nx) {
            const double *pn_ = nx + ((size_t)d * L + (size_t)ky) * (8 * FFT64B_PFN);
            _Pragma("GCC unroll 4")
            for (int q_ = 0; q_ < FFT64B_PFN; ++q_)
                __builtin_prefetch(pn_ + 8 * q_, 0, FFT64B_PFH);
        }
#if PW == 2
        if (nt) {   /* pair z-blocks so every NT store completes a 64-B line */
            for (int zb = 0; zb < NV; zb += 2) {
                const double *sa = gb + ((size_t)ky * RS + (size_t)zb * PW) * 2;
                const double *sb = sa + 2 * PW;
                if (pfb) {  /* one line per plane per paired step = 1:1 rate */
                    _Pragma("GCC unroll 8")
                    for (int c_ = 0; c_ < 8; ++c_)
                        __builtin_prefetch(sa + (size_t)c_ * (2 * PS)
                                              + FFT64B_PFBL * (2 * RS), 0, 3);
                }
                vec Za[8], Zb[8];
                DFT8M(*(const vec *)(sa           ), *(const vec *)(sa + 1*(2*PS)),
                      *(const vec *)(sa + 2*(2*PS)), *(const vec *)(sa + 3*(2*PS)),
                      *(const vec *)(sa + 4*(2*PS)), *(const vec *)(sa + 5*(2*PS)),
                      *(const vec *)(sa + 6*(2*PS)), *(const vec *)(sa + 7*(2*PS)),
                      Za[0],Za[1],Za[2],Za[3],Za[4],Za[5],Za[6],Za[7]);
                DFT8M(*(const vec *)(sb           ), *(const vec *)(sb + 1*(2*PS)),
                      *(const vec *)(sb + 2*(2*PS)), *(const vec *)(sb + 3*(2*PS)),
                      *(const vec *)(sb + 4*(2*PS)), *(const vec *)(sb + 5*(2*PS)),
                      *(const vec *)(sb + 6*(2*PS)), *(const vec *)(sb + 7*(2*PS)),
                      Zb[0],Zb[1],Zb[2],Zb[3],Zb[4],Zb[5],Zb[6],Zb[7]);
                double *db = ob + ((size_t)ky * L + (size_t)zb * PW) * 2;
                _Pragma("GCC unroll 8")
                for (int c_ = 0; c_ < 8; ++c_) {
                    STREAM_ST(db + (size_t)c_ * (8 * 2 * LSQ),          Za[c_]);
                    STREAM_ST(db + (size_t)c_ * (8 * 2 * LSQ) + 2 * PW, Zb[c_]);
                }
            }
            continue;
        }
#endif
        for (int zb = 0; zb < NV; ++zb) {
            const double *s_ = gb + ((size_t)ky * RS + (size_t)zb * PW) * 2;
            double       *d_ = ob + ((size_t)ky * L + (size_t)zb * PW) * 2;
            if (pfb) {   /* 8 T0 prefetches per step: one line per plane,
                          * FFT64B_PFBL padded rows ahead of the read cursor */
                _Pragma("GCC unroll 8")
                for (int c_ = 0; c_ < 8; ++c_)
                    __builtin_prefetch(s_ + (size_t)c_ * (2 * PS)
                                          + FFT64B_PFBL * (2 * RS), 0, 3);
            }
            if (pfw) {
                _Pragma("GCC unroll 8")
                for (int c_ = 0; c_ < 8; ++c_)
                    __builtin_prefetch(d_ + (size_t)c_ * (8 * 2 * LSQ) + 8 * FFT64B_PFWL, 1, 3);
            }
            vec z0_,z1_,z2_,z3_,z4_,z5_,z6_,z7_;
            DFT8M(*(const vec *)(s_           ), *(const vec *)(s_ + 1*(2*PS)),
                  *(const vec *)(s_ + 2*(2*PS)), *(const vec *)(s_ + 3*(2*PS)),
                  *(const vec *)(s_ + 4*(2*PS)), *(const vec *)(s_ + 5*(2*PS)),
                  *(const vec *)(s_ + 6*(2*PS)), *(const vec *)(s_ + 7*(2*PS)),
                  z0_,z1_,z2_,z3_,z4_,z5_,z6_,z7_);
            if (nt) {
                STREAM_ST(d_                          , z0_);
                STREAM_ST(d_ + (size_t)1 * (8*2*LSQ), z1_);
                STREAM_ST(d_ + (size_t)2 * (8*2*LSQ), z2_);
                STREAM_ST(d_ + (size_t)3 * (8*2*LSQ), z3_);
                STREAM_ST(d_ + (size_t)4 * (8*2*LSQ), z4_);
                STREAM_ST(d_ + (size_t)5 * (8*2*LSQ), z5_);
                STREAM_ST(d_ + (size_t)6 * (8*2*LSQ), z6_);
                STREAM_ST(d_ + (size_t)7 * (8*2*LSQ), z7_);
            } else {
                *(vec *)(d_                          ) = z0_;
                *(vec *)(d_ + (size_t)1 * (8*2*LSQ)) = z1_;
                *(vec *)(d_ + (size_t)2 * (8*2*LSQ)) = z2_;
                *(vec *)(d_ + (size_t)3 * (8*2*LSQ)) = z3_;
                *(vec *)(d_ + (size_t)4 * (8*2*LSQ)) = z4_;
                *(vec *)(d_ + (size_t)5 * (8*2*LSQ)) = z5_;
                *(vec *)(d_ + (size_t)6 * (8*2*LSQ)) = z6_;
                *(vec *)(d_ + (size_t)7 * (8*2*LSQ)) = z7_;
            }
        }
    }
}

/* st=1 pass B2: the FULL 64-point x-FFT (both radix-8 stages in registers,
 * the same FFT64V used for the z- and y-lines) per (ky, z-column), straight
 * from mid to out.  Reads are 64 streams at the padded plane stride --
 * odd-line padding spreads them over sets, hugepages keep them on 3 TLB
 * entries, and one prefetcht0 per load FFT64B_PFXC columns ahead covers the
 * L3 latency (the whole group of 64 loads is also independent, so the OoO
 * window supplies MLP on top).  Stores are 64 plane-streams to out: at PW=4
 * every store is exactly one full line, so NT stores are fill-buffer-clean
 * at any stride.  (At PW=2 an NT store is half a line -- correct but slow;
 * pw2/nt/st1 candidates are never generated, only env-forcible.)
 * Adopted from L64_radix8 r6's fused pass 2+3 (their strided in-place x-FFT
 * + next-column prefetch, +12% at B=8 there); this removes st=0's x1 RMW
 * sweep entirely. */
static void FN(passB2)(const double *restrict mid, double *restrict out,
                       int nt, const double *nx)
{
    for (int ky = 0; ky < L; ++ky)
        for (int zb = 0; zb < NV; ++zb) {
            const double *xsrc_ = mid + ((size_t)ky * RS + (size_t)zb * PW) * 2;
            double       *xdst_ = out + ((size_t)ky * L  + (size_t)zb * PW) * 2;
            if (nx) {   /* token pre-coverage of the next volume's input */
                const double *pn_ = nx + ((size_t)ky * NV + (size_t)zb)
                                         * (8 * FFT64B_PFN);
                _Pragma("GCC unroll 4")
                for (int q_ = 0; q_ < FFT64B_PFN; ++q_)
                    __builtin_prefetch(pn_ + 8 * q_, 0, FFT64B_PFH);
            }
            _Pragma("GCC unroll 64")
            for (int n_ = 0; n_ < L; ++n_)
                __builtin_prefetch(xsrc_ + (size_t)n_ * (2 * PS)
                                         + FFT64B_PFXC * (2 * PW), 0, 3);
#define LDX(n)    (*(const vec *)(xsrc_ + (size_t)(n) * (2 * PS)))
#define STX(k, v) do {                                                        \
                double *da_ = xdst_ + (size_t)(k) * (2 * LSQ);                \
                if (nt) STREAM_ST(da_, (v)); else *(vec *)da_ = (v);          \
            } while (0)
            FFT64V(LDX, STX);
#undef LDX
#undef STX
        }
}

/* ---- st=2: x-first 2-sweep (new in r9; fused tail adopted from
 * L64_radix8's pass 2+3, axis order inverted so the strided stage is never
 * the one writing out) --------------------------------------------------- */

/* st=2 pass 1, group s: x stage 1 DIRECTLY off the driver's input.  DFT-8
 * across in planes {s, s+8, ..., s+56} (8 sequential cold read streams --
 * this pass IS the volume's in-read), twiddle W64^{s*d}, store to mid plane
 * 8d+s (8 sequential write streams at padded stride).  Pure elementwise:
 * zero shuffles beyond the cmul swaps.  pfb: one T0 line per stream per
 * step, FFT64B_PFBL natural rows ahead of the read cursors. */
static void FN(x1_from_in)(const double *restrict in, double *restrict mid,
                           int s, int pfb)
{
    const double *bi = in  + (size_t)s * (2 * LSQ);
    double       *bo = mid + (size_t)s * (2 * PS);
#define X1IN_BODY(TWIDDLE) do {                                              \
    for (int y = 0; y < L; ++y)                                              \
        for (int zb = 0; zb < NV; ++zb) {                                    \
            const double *p_ = bi + ((size_t)y * L + (size_t)zb * PW) * 2;   \
            double       *q_ = bo + ((size_t)y * RS + (size_t)zb * PW) * 2;  \
            if (pfb) {                                                       \
                _Pragma("GCC unroll 8")                                      \
                for (int c_ = 0; c_ < 8; ++c_)                               \
                    __builtin_prefetch(p_ + (size_t)c_ * (8 * 2 * LSQ)       \
                                          + FFT64B_PFBL * (2 * L), 0, 3);    \
            }                                                                \
            vec v0_ = *(const vec *)(p_               );                     \
            vec v1_ = *(const vec *)(p_ + 1 * (8*2*LSQ));                    \
            vec v2_ = *(const vec *)(p_ + 2 * (8*2*LSQ));                    \
            vec v3_ = *(const vec *)(p_ + 3 * (8*2*LSQ));                    \
            vec v4_ = *(const vec *)(p_ + 4 * (8*2*LSQ));                    \
            vec v5_ = *(const vec *)(p_ + 5 * (8*2*LSQ));                    \
            vec v6_ = *(const vec *)(p_ + 6 * (8*2*LSQ));                    \
            vec v7_ = *(const vec *)(p_ + 7 * (8*2*LSQ));                    \
            vec g0_,g1_,g2_,g3_,g4_,g5_,g6_,g7_;                             \
            DFT8M(v0_,v1_,v2_,v3_,v4_,v5_,v6_,v7_,                           \
                  g0_,g1_,g2_,g3_,g4_,g5_,g6_,g7_);                          \
            TWIDDLE;                                                         \
            *(vec *)(q_               ) = g0_;                               \
            *(vec *)(q_ + 1 * (8*2*PS)) = g1_;                               \
            *(vec *)(q_ + 2 * (8*2*PS)) = g2_;                               \
            *(vec *)(q_ + 3 * (8*2*PS)) = g3_;                               \
            *(vec *)(q_ + 4 * (8*2*PS)) = g4_;                               \
            *(vec *)(q_ + 5 * (8*2*PS)) = g5_;                               \
            *(vec *)(q_ + 6 * (8*2*PS)) = g6_;                               \
            *(vec *)(q_ + 7 * (8*2*PS)) = g7_;                               \
        }                                                                    \
} while (0)
    if (s == 0) {
        X1IN_BODY((void)0);
    } else {
        const vec t1r = TRE(1*s), t1i = TIA(1*s), t2r = TRE(2*s), t2i = TIA(2*s);
        const vec t3r = TRE(3*s), t3i = TIA(3*s), t4r = TRE(4*s), t4i = TIA(4*s);
        const vec t5r = TRE(5*s), t5i = TIA(5*s), t6r = TRE(6*s), t6i = TIA(6*s);
        const vec t7r = TRE(7*s), t7i = TIA(7*s);
        X1IN_BODY(do {
            g1_ = VFMA(g1_, t1r, SWAP(g1_) * t1i);
            g2_ = VFMA(g2_, t2r, SWAP(g2_) * t2i);
            g3_ = VFMA(g3_, t3r, SWAP(g3_) * t3i);
            g4_ = VFMA(g4_, t4r, SWAP(g4_) * t4i);
            g5_ = VFMA(g5_, t5r, SWAP(g5_) * t5i);
            g6_ = VFMA(g6_, t6r, SWAP(g6_) * t6i);
            g7_ = VFMA(g7_, t7r, SWAP(g7_) * t7i);
        } while (0));
    }
#undef X1IN_BODY
}

/* st=2 tail: y then z transform of ONE completed output plane, out of the
 * L2-warm octet buffer plane ob (padded strides RS) straight to the out
 * plane o (natural strides).  Ordered y-FIRST, z-LAST so the expensive side
 * gets the good access pattern: the y-FFT's 64-row scatter reads land on the
 * L2-hot OB (reads carry no RFO and the 8 loads per DFT8 are independent),
 * while the z-FFT's transpose-on-store emits PW SEQUENTIAL row streams to
 * the cold out plane (the first z-last draft wrote out as a 64-row scatter
 * and lost 13% in-arena at B=1 on wallaby: 748 vs 663).  nt: stream the out
 * stores (full-line at PW=4); pfw: one prefetchw per line stored,
 * FFT64B_PFWL lines ahead in each row stream. */
static void FN(zy_plane_out)(const double *restrict ob, double *restrict o,
                             int nt, int pfw, int pfb)
{
    vec P_[L * RSV];                     /* P_[ky][zb] after the y pass */
    for (int zb = 0; zb < NV; ++zb) {
        if (pfb) {  /* next-column T0 at consumption rate (PFXC idiom from
                     * L64_radix8 r6 via my passB2); column zb+1 exists in
                     * the row padding at zb=NV-1, so no guard is needed */
            _Pragma("GCC unroll 64")
            for (int n_ = 0; n_ < L; ++n_)
                __builtin_prefetch(ob + ((size_t)n_ * RS + (size_t)(zb + 1) * PW) * 2, 0, 3);
        }
#define LDYF(n)    (*(const vec *)(ob + ((size_t)(n) * RS + (size_t)zb * PW) * 2))
#define STYF(k, v) (P_[(size_t)(k) * RSV + zb] = (v))
        FFT64V(LDYF, STYF);
#undef LDYF
#undef STYF
    }
    for (int yb = 0; yb < L; yb += PW) {
        vec Zv[64];
        _Pragma("GCC unroll 32")
        for (int kb = 0; kb < NV; ++kb) {
            vec r_[PW];
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j)
                r_[j] = P_[(size_t)(yb + j) * RSV + kb];
            TRNC(r_, &Zv[kb * PW]);
        }
#define LDZ(n)    Zv[n]
#define STZ(k, v) (Zv[k] = (v))
        FFT64V(LDZ, STZ);
#undef LDZ
#undef STZ
        _Pragma("GCC unroll 32")
        for (int kb = 0; kb < NV; ++kb) {
            vec r_[PW];
            TRNC(&Zv[kb * PW], r_);
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j) {
                double *da_ = o + ((size_t)(yb + j) * L + (size_t)kb * PW) * 2;
                if (pfw) __builtin_prefetch(da_ + 8 * FFT64B_PFWL2, 1, 3);
                if (nt) STREAM_ST(da_, r_[j]); else *(vec *)da_ = r_[j];
            }
        }
    }
}

/* st=2 pass 2, octet d: x stage 2 (plain DFT-8) over the 8 CONSECUTIVE mid
 * planes {8d..8d+7} -- one sequential ~545 KB read run -- into the octet
 * buffer OB's 8 padded planes (c = output digit), then the z+y transforms
 * of each completed plane 8c+d straight out of OB.  OB is reused across
 * octets and volumes, so its ~560 KB recirculates in L2/L3 instead of
 * costing a third full-volume sweep. */
static void FN(pass2_octet)(const double *restrict mid, double *restrict OB,
                            double *restrict out, int d, int nt, int pfw,
                            int pfb)
{
    const double *gb = mid + (size_t)(8 * d) * (2 * PS);
    for (int ky = 0; ky < L; ++ky)
        for (int zb = 0; zb < NV; ++zb) {
            const double *s_ = gb + ((size_t)ky * RS + (size_t)zb * PW) * 2;
            double       *t_ = OB + ((size_t)ky * RS + (size_t)zb * PW) * 2;
            if (pfb) {
                _Pragma("GCC unroll 8")
                for (int c_ = 0; c_ < 8; ++c_)
                    __builtin_prefetch(s_ + (size_t)c_ * (2 * PS)
                                          + FFT64B_PFBL * (2 * RS), 0, 3);
            }
            vec z0_,z1_,z2_,z3_,z4_,z5_,z6_,z7_;
            DFT8M(*(const vec *)(s_           ), *(const vec *)(s_ + 1*(2*PS)),
                  *(const vec *)(s_ + 2*(2*PS)), *(const vec *)(s_ + 3*(2*PS)),
                  *(const vec *)(s_ + 4*(2*PS)), *(const vec *)(s_ + 5*(2*PS)),
                  *(const vec *)(s_ + 6*(2*PS)), *(const vec *)(s_ + 7*(2*PS)),
                  z0_,z1_,z2_,z3_,z4_,z5_,z6_,z7_);
            *(vec *)(t_           ) = z0_;
            *(vec *)(t_ + 1*(2*PS)) = z1_;
            *(vec *)(t_ + 2*(2*PS)) = z2_;
            *(vec *)(t_ + 3*(2*PS)) = z3_;
            *(vec *)(t_ + 4*(2*PS)) = z4_;
            *(vec *)(t_ + 5*(2*PS)) = z5_;
            *(vec *)(t_ + 6*(2*PS)) = z6_;
            *(vec *)(t_ + 7*(2*PS)) = z7_;
        }
    for (int c = 0; c < 8; ++c)
        FN(zy_plane_out)(OB + (size_t)c * (2 * PS),
                         out + (size_t)(8 * c + d) * (2 * LSQ), nt, pfw, pfb);
}

static void FN(run_vols)(const double *restrict in, double *restrict out,
                         double *restrict S, int nvol, int mode, int pf, int st)
{
    const int pfr = (pf == 3 || pf == 4) ? 2 : (pf == 1 || pf == 2) ? 1 : 0;
    const int pfw = (pf == 2 || pf == 3 || pf == 5 || pf == 7) && mode == M_CACHED;
    const int pfb = (pf == 6 || pf == 7);
    const int nt  = (mode == M_NT);
    for (int b = 0; b < nvol; ++b) {
        const double *iv = in  + (size_t)b * VDBL;
        double       *ov = out + (size_t)b * VDBL;
        const double *nx = (pf == 1 || pf == 2) && b + 1 < nvol ? iv + VDBL : NULL;
        if (st == 2) {
#ifndef FFT64B_SKIPA
            for (int s = 0; s < 8; ++s)
                FN(x1_from_in)(iv, S, s, pfb);
#endif
#ifndef FFT64B_SKIPB
            for (int d = 0; d < 8; ++d)
                FN(pass2_octet)(S, S + MIDDBL, ov, d, nt, pfw, pfb);
#endif
            continue;
        }
        if (st == 1) {
            /* 2-sweep: no x1, so pass A visits planes in NATURAL order and
             * the cold in-read is one sequential 4.19 MB run */
#ifndef FFT64B_SKIPA
            for (int p = 0; p < L; ++p) {
                const double *pfnext =
                    (p < L - 1) ? iv + (size_t)(p + 1) * (2 * LSQ) : nx;
                FN(passA_plane)(iv, S, p, pfnext, pfr);
            }
#endif
#ifndef FFT64B_SKIPB
            FN(passB2)(S, ov, nt, nx);
#endif
            (void)pfw;
            continue;
        }
        /* FFT64B_SKIP*: timing-only diagnostics; the answer is WRONG with any
         * of them set (same convention as L36_pfa's FFT36_SKIP1/2) */
        for (int r = 0; r < 8; ++r) {
#ifndef FFT64B_SKIPA
            for (int a = 0; a < 8; ++a) {
                int p = r + 8 * a;
                /* next plane in VISIT order: a+1 in this group, else the
                 * next group's first plane, else the next volume */
                const double *pfnext =
                    (a < 7) ? iv + (size_t)(p + 8) * (2 * LSQ)
                  : (r < 7) ? iv + (size_t)(r + 1) * (2 * LSQ)
                  : nx;
                FN(passA_plane)(iv, S, p, pfnext, pfr);
            }
#endif
#ifndef FFT64B_SKIPX1
            FN(x1_group)(S, r, pfb);
#endif
        }
#ifndef FFT64B_SKIPB
        for (int d = 0; d < 8; ++d)
            FN(passB_group)(S, ov, d, nt, pfw, pfb, nx);
#else
        (void)ov; (void)nt; (void)pfw; (void)pfb;
#endif
    }
}

#undef PFA1
#undef PFA1_NTA
#undef PFSTEP
#undef FFT64V
#undef DFT8M
#undef CMULT
#undef TIA
#undef TRE
#undef CSV
#undef PMV
#undef TRNC
#undef STREAM_ST
#undef VFMA
#undef VFNMA
#undef SWAP
#undef VPAIR
#undef VSPLAT
#undef RSV
#undef NV
#undef VSH
#undef FN
#undef veci
#undef vec

#endif /* template */
