/* L45_mixedradix -- forward complex-double 3D DFT of a fixed 45^3 cube.
 *
 * TECHNIQUE
 *   Row-column (three 1-D passes) with a Good-Thomas / prime-factor 9x5 line
 *   transform, batch-vectorised across the *lines* of each pass, and blocked
 *   per x-plane.  This is L36_mixedradix's architecture (the L=36 leader since
 *   panel_r1) transplanted to L=45; the differences forced by 45 being odd are
 *   documented in strategies/L45_mixedradix.md.
 *
 *   45 = 9*5 with gcd(9,5)=1, so the Good-Thomas (prime-factor) map applies
 *   and the entire inter-stage twiddle stage vanishes:
 *
 *       n = (5*n1 + 9*n2) mod 45        (Ruritanian input map, n1<9, n2<5)
 *       k = (10*k1 + 36*k2) mod 45      (CRT output map: 10 = 5*[5^-1]_9,
 *                                        36 = 9*[9^-1]_5)
 *       n*k = 5*n1*k1 + 9*n2*k2 (mod 45)  =>  W45^{nk} = W9^{n1 k1} W5^{n2 k2}
 *
 *   so a 45-point DFT is exactly 9 independent 5-point DFTs followed by 5
 *   independent 9-point DFTs with *no* twiddles in between -- only a
 *   compile-time index permutation, free because every pass is already a
 *   strided gather/scatter.  The 9-point module is genfft's n1_9 FMA DAG
 *   (fftw-3.3.10, 40 FMA-port vector ops + 12 shuffles; transcription
 *   borrowed from L45_pfa r9 -- it replaced this entry's hand CT 3x3, which
 *   cost 44 + 10).  The 5-point module is FFTW n1_5's FMA DAG (sqrt(5)/4
 *   cosine split + scaled-sine trick; borrowed from L45_pfa r6), 2 FMA-port
 *   ops cheaper than the classic two-rotation form.
 *   A plain Cooley-Tukey 9x5 line transform (twiddles kept, the literal
 *   reading of this entry's name) was counted and rejected in r6: it needs 32
 *   nontrivial inter-stage CMULs = +64 FMA-port ops per line.
 *
 *   Interleaved complex is kept end to end.  A vector holds PW complex
 *   numbers = PW *lines* of the current pass, so every constant is
 *   lane-invariant and there is no cross-lane operation inside the transform.
 *   Only the z pass -- whose transform axis is the contiguous one -- needs a
 *   transpose, done in registers as PW x PW blocks of 128-bit complex lanes.
 *
 *   45 is not a multiple of PW (4 or 2), so each pass has a tail:
 *     * z pass (in -> plane) and y pass (plane -> out): NGF full groups plus
 *       ONE true PW=1 (128-bit xmm) tail line each (r11, borrowed from
 *       L45_pfa r10, node-verified there): the same PFA DAG at one complex
 *       per vector -- no transposes, 16 B accesses that never split a cache
 *       line, and xmm FMAs dual-issue on the node's ports 0 and 1.  This
 *       replaced the r6-r10 overlap-recompute tails (90 full-width calls per
 *       volume covering one line each); -DFFT45_R10TAIL restores those.
 *     * x pass (in place in `out`) cannot overlap, but it CAN tile across y
 *       boundaries: all 2025 (y,z) lines share the same x-stride and are
 *       contiguous in flat index, so 2025 = 506*PW + 1 at PW=4 leaves ONE
 *       masked-load/masked-store tail call per volume (was one per y).
 *     * the z-pass gather/scatter handles its odd 45th *column* with masked
 *       loads/stores through the same PWxPW transpose.
 *   The plane scratch rows are padded to PPITCH = 52 complex (832 B = 13
 *   cache lines, 64B-aligned rows, coprime with the 64 L1 sets -- see the
 *   PPITCH comment); accesses to `in`/`out` rotate alignment mod 64 (odd-L
 *   fact of life).
 *
 *   tr KERNEL (ice_r2, BORROWED from L36_pfa ice_r2; the scored pick on the
 *   Ice Lake node stacks it with pf1+pfw): on ICX-SP the second 512-bit FMA
 *   pipe shares PORT 5 with all 512-bit shuffles, and vbroadcastf64x2 from
 *   memory folds into one load-port uop.  tr=1 phase 1 therefore builds
 *   every lane-transposed vector as PW masked 128-bit broadcast loads and
 *   stores codelet output UNtransposed to a slot-major plane scratch
 *   (slot s = ygroup*45 + kz, lanes = y); the y subloop re-gathers lanes =
 *   kz the same way.  Both TRANSP passes, the Xv/Yv spill arrays, and the
 *   odd-column special forms vanish from phase 1: p5-only shuffles drop
 *   206,856 -> 116,766 per volume (the codelet swaps are all that remain).
 *   Node A/B (graded B=4 chain, same-process arena): tr-pf1-pfw 253.5 vs
 *   the r1 pick v1-pf0 280.3 us/vol = -9.6%.
 *
 *   Pass structure -- two sweeps over the volume, not three:
 *     phase 1, per x-plane (45x45 complex = 31.6 KiB + padded plane scratch):
 *              45 z-lines (transpose in, transform, transpose out into the
 *              plane scratch), then 45 y-lines straight into `out`.
 *     phase 2: x-lines, in place in `out` (stride 32400 B), tiled PW
 *              consecutive z at a time.
 *   The 1.39 MiB volume does NOT fit the target's 1 MiB L2 (unlike L=36), so
 *   phase 2 runs out of L3 at B=1; the volume+volume working set of 2.78 MiB
 *   sits comfortably in the 22 MiB L3.
 *
 *   MEMORY MECHANISMS (tournament-gated candidates, never defaults; since r8
 *   they are ORTHOGONAL candidates, not a cumulative ladder -- the node's r7
 *   verdict showed the phase-2 poke is pure overhead there, and it was
 *   poisoning every bundled pfin/pfw candidate):
 *   * pf:   phase 2 prefetches its 45 x-streams by hand (1 or, new in r11,
 *           3 lines ahead -- L45_pfa's B=16 node winner is a bare distance-3
 *           poke) -- more streams than the L2 streamer tracks.
 *           (L36_mixedradix r1.)
 *           pkw variant pokes with PREFETCHW instead of T0: phase 2 is an
 *           in-place read-modify-write, so fetch-for-ownership does both
 *           jobs (shape of L64_radix8's node-picked pfw).
 *   * pfin: paced T1 prefetch of the phase-1 `in` read stream, one x-plane
 *           (31.6 KB) ahead, spread over the z-subloop's 12 codelet calls, so
 *           the read of plane x+1 overlaps the compute of plane x.  Positional
 *           (derived from the plane pointer), so it cannot drift across a
 *           batch.  Plus a 1-line-per-call re-cover of in[b+1]'s first plane
 *           from phase 2 (L36_pfa's PFIN/PFNX, via L36_mixedradix r4, both
 *           attributed).
 *   * pfw:  paced WRITE-INTENT prefetch (__builtin_prefetch(p,1,3) ->
 *           prefetchw) over phase 1's cold-`out` store stream, one plane
 *           ahead, spread over the y-subloop; acquires lines exclusive so the
 *           RFO overlaps compute (L36_pfa r5's PFWMID; L6_unrolled r3).
 *   * cpy:  the y pass stores into an L1-hot plane image and one ERMS
 *           rep-movsb per x-plane copies it to `out`.  ERMS full-line writes
 *           skip the RFO read of the destination, deleting 1.46 MB/volume of
 *           cold-`out` fill traffic (L3 at B=1, DRAM when streaming) instead
 *           of merely overlapping it, and unlike NT stores the lines stay in
 *           the cache hierarchy for phase 2 to re-read.  NEW in r8, mine.
 *   No NT-store path: the node rejected NT in every L=36 tournament for four
 *   consecutive rounds (L36_mixedradix r6 retired it; see that record).
 *
 * OPERATION COUNT (per 45-point line, as vector instructions over PW lanes)
 *       DFT5 (n1_5)  : 16 FMA-port ops +  2 shuffles   x9 = 144 + 18
 *       DFT9 (n1_9)  : 40 FMA-port ops + 12 shuffles   x5 = 200 + 60
 *       total :                        344 FMA-port ops + 78 shuffles / PW lines
 *   (r7-r9's CT 3x3 DFT9 gave 364 + 68.)  Per volume at PW=4 (r11 tails):
 *   45*(11+11) + 506 + 1 = 1497 full-width calls x 344 = 514,968 zmm
 *   FMA-port ops, plus 90 xmm tail lines x 344 that dual-issue on ports
 *   0+1; port-0-equivalent floor ~183 us at 2.9 GHz (r10: 1587 calls,
 *   545,928 ops, 188 us).  Phase 2 is 506 full + 1 masked call.
 *
 * ASSUMPTIONS
 *   * L == 45 only; fft3d_supports() refuses everything else.
 *   * `in` and `out` are 64-byte aligned and distinct (driver guarantees).
 *   * `out` doubles as the working buffer between phase 1 and phase 2; `in`
 *     is never written.
 *   * Three kernels: V0 = AVX2+FMA (2 complex lanes, 16 ymm), V1 = AVX-512F
 *     (4 complex lanes, 32 zmm), V2 = AVX-512VL (2 complex lanes, 32 ymm).
 *     fft3d_create() verifies each against V0 numerically, then times all
 *     surviving (kernel x mechanism) combinations and installs the fastest,
 *     with an earlier-candidate-wins hysteresis (L36_mixedradix r4/r5
 *     tuner): 3% cached (wallaby's toggle needs the guard), 1.5% streaming
 *     since r11 (the node's streaming tournaments are quiet and 3% blocked
 *     every 2%-class mechanism move).
 */

#ifndef VAR
/* ======================= common (width-independent) part ================== */

#define _GNU_SOURCE             /* syscall() for the perf probe */
#define _POSIX_C_SOURCE 200809L

#include <complex.h>
#include <errno.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "fft3d_api.h"

#define LSIDE 45
#define NPLANE (LSIDE * LSIDE)          /* complex per x-plane = 2025    */
#define NVOL   (LSIDE * LSIDE * LSIDE)  /* complex per volume  = 91125   */
#ifndef PPITCH
#define PPITCH 52                       /* plane-scratch row pitch, complex:
                                           52*16 = 832 B = 13 cache lines,
                                           64-byte aligned rows.  13 is coprime
                                           with the 64 L1 sets, so the y-pass
                                           column walk (45 rows, stride PPITCH)
                                           spreads over all sets; the "obvious"
                                           pad 48 = 12 lines has gcd(12,64)=4
                                           and hits only 16 sets.  Borrowed
                                           from L45_pfa r6 (their A/B: pitch 52
                                           15-20% faster at B=1); overridable
                                           with -DPPITCH=48 for node A/B. */
#endif
#define PFLINES 508                     /* 64B lines covering one x-plane     */

/* ---- pre-splatted constants, 8-double 64-byte rows usable as memory
   operands by both the 256-bit and 512-bit kernels. ---------------------- */
#define SPLAT8(v) { (v), (v), (v), (v), (v), (v), (v), (v) }
#define ALT8(v)   { (v), -(v), (v), -(v), (v), -(v), (v), -(v) }

static const double KC_HALF[8] __attribute__((aligned(64))) = SPLAT8(0.5);
/* sqrt(3)/2 alternating: swap(m) * this = -i*s*m in interleaved layout */
static const double KC_KS[8] __attribute__((aligned(64)))
    = ALT8(8.66025403784438646764e-01);
/* genfft n1_9 FMA-DAG constants (fftw-3.3.10 dft/scalar/codelets/n1_9.c;
 * vector transcription borrowed from L45_pfa r9).  A* rows are alternating
 * [v,-v,...] (their VPAIR), S852 is a plain splat. */
static const double KC_A176[8]  __attribute__((aligned(64))) = ALT8( 1.76326980708464973471e-01); /* tan(pi/18) */
static const double KC_A176N[8] __attribute__((aligned(64))) = ALT8(-1.76326980708464973471e-01);
static const double KC_A839[8]  __attribute__((aligned(64))) = ALT8( 8.39099631177280011763e-01);
static const double KC_A777[8]  __attribute__((aligned(64))) = ALT8( 7.77861913430206160028e-01);
static const double KC_A984[8]  __attribute__((aligned(64))) = ALT8( 9.84807753012208059367e-01);
static const double KC_A492[8]  __attribute__((aligned(64))) = ALT8( 4.92403876506104029683e-01);
static const double KC_A363[8]  __attribute__((aligned(64))) = ALT8( 3.63970234266202361351e-01); /* tan(pi/9) */
static const double KC_A954N[8] __attribute__((aligned(64))) = ALT8(-9.54188894138671133499e-01);
static const double KC_S852[8]  __attribute__((aligned(64))) = SPLAT8(8.52868531952443209628e-01);
/* 5-point, FFTW n1_5 FMA constants (borrowed from L45_pfa r6):
 * c1,c2 = -1/4 +- sqrt(5)/4 (cosine split), s2 = phi^-1 * s1 (scaled sine) */
static const double KC_Q4[8] __attribute__((aligned(64))) = SPLAT8(2.5e-01);
static const double KC_S5[8] __attribute__((aligned(64))) = SPLAT8(5.59016994374947424102e-01);
static const double KC_PHI[8] __attribute__((aligned(64))) = SPLAT8(6.18033988749894848205e-01);
static const double KC_S1[8] __attribute__((aligned(64))) = ALT8(9.51056516295153572116e-01);
/* one-complex tail mask for the AVX2 maskload/maskstore path */
static const long long KC_TMASK[4] __attribute__((aligned(32))) = { -1, -1, 0, 0 };

/* ---- map constants (ice_r4 fused chain): state <- (z+c)/(1+|z+c|).
 * KC_TINYH = 0.5e-300 is added to BOTH squared halves before the re/im swap-
 * add, so s = re^2+im^2+1e-300: replaces a max() AND the rsqrt14(0)=inf NaN
 * trap (L23_matrixsimd ice_r4's clamp, L17_winograd's additive-bias form);
 * it perturbs any nonzero |w| by < 1e-284 relative. ---- */
static const double KC_TINYH[8] __attribute__((aligned(64), unused)) = SPLAT8(0.5e-300);
static const double KC_1P5[8]   __attribute__((aligned(64), unused)) = SPLAT8(1.5);
static const double KC_ONE[8]   __attribute__((aligned(64), unused)) = SPLAT8(1.0);
static const double KC_TWO[8]   __attribute__((aligned(64), unused)) = SPLAT8(2.0);

/* chain-step compile-time knobs (bit classes and prefetch shapes are fixed
 * per binary so the two-run repeatability cmp can never flip them):
 *   FFT45_CMS  map style: 0 rsqrt14+2NR then ONE exact vdivpd (default; the
 *              winner at L13_rader/L23_matrixsimd ice_r4), 1 rsqrt14+2NR then
 *              rcp14+2NR (divider-free; L17_winograd/L23_rader's winner),
 *              2 hw vsqrtpd then rcp14+2NR, 3 hw vsqrtpd + vdivpd (control).
 *   FFT45_CPF  poke the 45 dst x-streams in fused phase 2 (default 1)
 *   FFT45_CPFC poke the 45 c x-streams too (default 1)
 *   FFT45_CPFD poke distance in cache lines (default 1)
 *   FFT45_CPFW paced prefetchw over the dst store stream in phase 1 (dflt 1)
 */
#ifndef FFT45_CMS
#define FFT45_CMS 0
#endif
#ifndef FFT45_CPF
#define FFT45_CPF 1
#endif
#ifndef FFT45_CPFC
#define FFT45_CPFC 1
#endif
#ifndef FFT45_CPFD
#define FFT45_CPFD 1
#endif
#ifndef FFT45_CPFW
#define FFT45_CPFW 1
#endif
/* FFT45_CXF: x-first step order (pass A = x-lines IN PLACE on the state
 * volume, pass B = per-plane z subloop -> scratch -> MAPPED y subloop).
 * Moves the map's ~12 vector ops/point from phase 2 (measured slack ~31 us
 * against its port floor) into the y subloop (~65 us of gather-latency
 * slack), makes the c reads plane-sequential, and turns the whole chain
 * in-place in ONE private volume (2.9 MiB/volume working set vs 4.4).
 * FFT45_CPFN paces a T1 prefetch of the NEXT c plane over the z subloop;
 * FFT45_CPKW pokes pass A's 45 RMW streams with prefetchw instead of T0. */
/* Node-tuned defaults (ice_r4, graded B=4 m=177 cell, quiet-window minima):
 *   CXF=1  x-first mapped-y 287.1 vs classic fused-p2 347.0 us/xform
 *   CPFN=0 c-plane T1 pacing costs ~+12-20 us (pfin's pure-uop-tax history)
 *   CPKW=0 T0 poke beats prefetchw on pass A's RMW streams
 *   CPFD=1, CPF=1: distance 3 and no-poke both ~+11 us
 *   CMS=0 everywhere: one exact vdivpd; the rcp14-ladder form lost +25 us
 *          in the y subloop and +40 us in phase 2 (issue-bound, not
 *          divider-bound -- L23_matrixsimd's verdict, not L23_rader's). */
#ifndef FFT45_CXF
#define FFT45_CXF 1
#endif
#ifndef FFT45_CPFN
#define FFT45_CPFN 0
#endif
#ifndef FFT45_CPKW
#define FFT45_CPKW 0
#endif
/* FFT45_CPFY: T0-poke the y subloop's 45 c rows one call (64 B) ahead --
 * the c reads are 45 8-line streams per plane, too short for the L2
 * streamer to lock on (the pf1 shape, aimed at c instead of the data). */
#ifndef FFT45_CPFY
#define FFT45_CPFY 0
#endif

/* stage 1: nine 5-point DFTs; stage 2: five 9-point DFTs */
#define REP9(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7) M(8)
#define REP5(M) M(0) M(1) M(2) M(3) M(4)

/* ---- the PFA 9x5 line codelet, as lazily-expanded macro text.  Moved from
 * the per-variant section to here in r11 (following L45_pfa r10's
 * lazy-expansion trick) so the SAME DAG can also be instantiated at PW=1 for
 * the 128-bit tail lines below.  ST1G/ST2G reference VD / VADD / VSWAP /
 * VLOAD / C_* ..., which resolve to whichever width's macro layer is in
 * scope at the expansion site. ---- */
#define C_HALF VLOAD(KC_HALF)
#define C_KS   VLOAD(KC_KS)
#define C_Q4   VLOAD(KC_Q4)
#define C_S5   VLOAD(KC_S5)
#define C_PHI  VLOAD(KC_PHI)
#define C_S1   VLOAD(KC_S1)

/* stage 1: nine 5-point DFTs over n2, PFA input map n = (5*n1 + 9*n2) mod 45.
 * FFTW n1_5's FMA DAG (borrowed from L45_pfa r6, from the corpus):
 *   t1 = x1+x4, t4 = x1-x4, t2 = x2+x3, t3 = x2-x3
 *   te = t1+t2, ta = t1-t2;  y0 = x0 + te
 *   cosine split c1,c2 = -1/4 +- sqrt(5)/4:
 *     tm = x0 - te/4;  tp = tm + (sqrt(5)/4)*ta;  tq = tm - (sqrt(5)/4)*ta
 *   scaled sine s2 = phi^-1 * s1 (phi^-1 = 0.618...):
 *     w1 = swap(t4) + phi^-1*swap(t3);  w2 = phi^-1*swap(t4) - swap(t3)
 *   y1/y4 = tp +/- ALT(s1)*w1 ;  y2/y3 = tq +/- ALT(s1)*w2
 * with -i*s*t = swap(t) * [s,-s,...].  16 FMA-port ops + 2 shuffles
 * (2 ops fewer than the two-rotation form used in rounds up to r6).
 * Output y[k2] goes to TT[k2*9 + N1] so stage 2 reads contiguous runs. */
#define ST1G(TT, LS, N1) {                                     \
    VD x0 = LS((5 * (N1) + 9 * 0) % 45);                       \
    VD x1 = LS((5 * (N1) + 9 * 1) % 45);                       \
    VD x2 = LS((5 * (N1) + 9 * 2) % 45);                       \
    VD x3 = LS((5 * (N1) + 9 * 3) % 45);                       \
    VD x4 = LS((5 * (N1) + 9 * 4) % 45);                       \
    VD t1 = VADD(x1, x4), t4 = VSUB(x1, x4);                   \
    VD t2 = VADD(x2, x3), t3 = VSUB(x2, x3);                   \
    VD te = VADD(t1, t2), ta = VSUB(t1, t2);                   \
    VD tm = VFNMADD(C_Q4, te, x0);                             \
    VD tp = VFMADD(ta, C_S5, tm);                              \
    VD tq = VFNMADD(ta, C_S5, tm);                             \
    VD u4 = VSWAP(t4), u3 = VSWAP(t3);                         \
    VD w1 = VFMADD(u3, C_PHI, u4);                             \
    VD w2 = VFMSUB(u4, C_PHI, u3);                             \
    TT[0 * 9 + (N1)] = VADD(x0, te);                           \
    TT[1 * 9 + (N1)] = VFMADD(w1, C_S1, tp);                   \
    TT[2 * 9 + (N1)] = VFMADD(w2, C_S1, tq);                   \
    TT[3 * 9 + (N1)] = VFNMADD(w2, C_S1, tq);                  \
    TT[4 * 9 + (N1)] = VFNMADD(w1, C_S1, tp);                  \
}
#define ST1(N1) ST1G(T, LSRC, N1)

/* stage 2: five 9-point DFTs over n1, PFA output map k = (10*k1+36*k2) mod 45.
 * The 9-point module is genfft's n1_9 FMA DAG (fftw-3.3.10
 * dft/scalar/codelets/n1_9.c, 24 add + 56 fma = 80 scalar FMA-port ops),
 * transcribed to interleaved-complex vectors: each scalar re/im line pair is
 * one vector op, each re/im crossing one VSWAP with the signs folded into an
 * ALT8 constant row.  40 FMA-port ops + 12 shuffles (the CT 3x3 it replaces,
 * rounds r6-r9: 44 + 10).  Transcription BORROWED from L45_pfa r9 (their
 * DFT9F macro, node-verified at rel_l2 3.99e-16), re-spelt in this file's
 * macro vocabulary.  Shape: per radix-3 column {n,n+3,n+6}: s = column sum,
 * S = full sum, a = x_n - s/2, i = SWAP(x_{n+3}-x_{n+6}); p/q = the two
 * rotated DFT3 outputs a -+ i*866.  k={0,3,6} is a DFT3 on the sums;
 * k={1,4,7} and k={2,5,8} build the (1 +- c*i) spiral factors w from p/q
 * (one SWAP+FMA each), cross them (uu/zz), and fan out through the 984 /
 * 492+-852 cosine split. */
#define ST2G(TT, SD, K2) {                                     \
    VD u0 = TT[(K2) * 9 + 0], u1 = TT[(K2) * 9 + 1];           \
    VD u2 = TT[(K2) * 9 + 2], u3 = TT[(K2) * 9 + 3];           \
    VD u4 = TT[(K2) * 9 + 4], u5 = TT[(K2) * 9 + 5];           \
    VD u6 = TT[(K2) * 9 + 6], u7 = TT[(K2) * 9 + 7];           \
    VD u8 = TT[(K2) * 9 + 8];                                  \
    VD s0 = VADD(u3, u6), e0 = VSUB(u3, u6);                   \
    VD S0 = VADD(u0, s0), a0 = VFNMADD(C_HALF, s0, u0);        \
    VD i0 = VSWAP(e0);                                         \
    VD s1 = VADD(u4, u7), e1 = VSUB(u4, u7);                   \
    VD S1 = VADD(u1, s1), a1 = VFNMADD(C_HALF, s1, u1);        \
    VD i1 = VSWAP(e1);                                         \
    VD p1 = VFMADD(i1, C_KS, a1), q1 = VFNMADD(i1, C_KS, a1);  \
    VD s2 = VADD(u5, u8), e2 = VSUB(u5, u8);                   \
    VD S2 = VADD(u2, s2), a2 = VFNMADD(C_HALF, s2, u2);        \
    VD i2 = VSWAP(e2);                                         \
    VD p2 = VFMADD(i2, C_KS, a2), q2 = VFNMADD(i2, C_KS, a2);  \
    VD sg = VADD(S1, S2), d3 = VSUB(S2, S1), id = VSWAP(d3);   \
    VD b0 = VFNMADD(C_HALF, sg, S0);                           \
    SD((10 * 0 + 36 * (K2)) % 45, VADD(S0, sg));               \
    SD((10 * 3 + 36 * (K2)) % 45, VFNMADD(id, C_KS, b0));      \
    SD((10 * 6 + 36 * (K2)) % 45, VFMADD(id, C_KS, b0));       \
    {   /* k = 1, 4, 7 */                                      \
    VD v1 = VFMADD(i0, C_KS, a0);                              \
    VD w2 = VFMADD(VSWAP(p2), VLOAD(KC_A176N), p2);            \
    VD w1 = VFMADD(VSWAP(p1), VLOAD(KC_A839), p1);             \
    VD uu = VFMADD(w1, VLOAD(KC_A777), VSWAP(w2));             \
    VD zz = VFMADD(VSWAP(w1), VLOAD(KC_A777), w2);             \
    SD((10 * 1 + 36 * (K2)) % 45, VFMADD(uu, VLOAD(KC_A984), v1)); \
    VD r1 = VFNMADD(uu, VLOAD(KC_A492), v1);                   \
    SD((10 * 4 + 36 * (K2)) % 45, VFMADD(zz, VLOAD(KC_S852), r1)); \
    SD((10 * 7 + 36 * (K2)) % 45, VFNMADD(zz, VLOAD(KC_S852), r1)); \
    }                                                          \
    {   /* k = 2, 5, 8 */                                      \
    VD v2 = VFNMADD(i0, C_KS, a0);                             \
    VD wA = VFMADD(q1, VLOAD(KC_A176), VSWAP(q1));             \
    VD wB = VFNMADD(VSWAP(q2), VLOAD(KC_A363), q2);            \
    VD uB = VFMADD(wB, VLOAD(KC_A954N), wA);                   \
    VD zB = VFMADD(VSWAP(wB), VLOAD(KC_A954N), VSWAP(wA));     \
    SD((10 * 2 + 36 * (K2)) % 45, VFMADD(uB, VLOAD(KC_A984), v2)); \
    VD rB = VFNMADD(uB, VLOAD(KC_A492), v2);                   \
    SD((10 * 5 + 36 * (K2)) % 45, VFNMADD(zB, VLOAD(KC_S852), rB)); \
    SD((10 * 8 + 36 * (K2)) % 45, VFMADD(zB, VLOAD(KC_S852), rB)); \
    }                                                          \
}
#define ST2(K2) ST2G(T, SDST, K2)

/* ---- true PW=1 tail lines (r11, BORROWED from L45_pfa r10, node-verified
 * there at -1.6..-1.9% in all three cells).  45 = 11*4 + 1, so PW never
 * divides 45, and the r6-r10 answer -- recompute a full-width overlap group
 * -- burned one full 344-op group call per subloop per plane (90 full-width
 * calls/volume) to cover ONE line each.  These 128-bit lines cost the same
 * 344 ops on xmm vectors, which on the scoring node's CLX core dual-issue on
 * ports 0 AND 1 (only 512-bit is single-ported there), carry NO transposes,
 * and make 16 B accesses that can never split a cache line.  Output is
 * bit-identical to the overlap form: the DAG applies the same scalar
 * operations per lane, and the ALT8/SPLAT8 constant rows read the same
 * [v,-v] pair.  Compiled once here under target("fma"); called by every
 * width variant.  -DFFT45_R10TAIL restores the r10 overlap tails. */
#define VD               __m128d
#define VLOAD(p)         _mm_loadu_pd(p)
#define VADD(a, b)       _mm_add_pd((a), (b))
#define VSUB(a, b)       _mm_sub_pd((a), (b))
#define VFMADD(a, b, c)  _mm_fmadd_pd((a), (b), (c))
#define VFMSUB(a, b, c)  _mm_fmsub_pd((a), (b), (c))
#define VFNMADD(a, b, c) _mm_fnmadd_pd((a), (b), (c))
#define VSWAP(a)         _mm_shuffle_pd((a), (a), 1)

/* z-axis tail: the single y = 44 line of one x-plane; both the `in` row and
 * the plane row are contiguous, so there is nothing to transpose at PW=1. */
__attribute__((target("fma"), noinline, unused))
static void dft45_tz1(const double *src, double *dst)
{
    VD T[45];
#define LSRC(i)     VLOAD(src + (long)(i) * 2)
#define SDST(i, v)  _mm_storeu_pd(dst + (long)(i) * 2, (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* y-axis tail: the single z = 44 column (plane column walk -> out column
 * walk); 16 B accesses at 832 B / 720 B stride, never splitting a line. */
__attribute__((target("fma"), noinline, unused))
static void dft45_ty1(const double *src, double *dst)
{
    VD T[45];
#define LSRC(i)     VLOAD(src + (long)(i) * (PPITCH * 2))
#define SDST(i, v)  _mm_storeu_pd(dst + (long)(i) * (LSIDE * 2), (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* ---- tr (transpose-free phase 1, NEW ice_r2) tail twins.  The tr plane
 * layout is slot-major: slot s = (ygroup)*45 + kz holds one PW-lane vector
 * (PW*2 doubles), lane j = y-line ygroup*PW + j.  The z tail (y = 44) is
 * lane 0 of group NGF's slot row; the y tail (kz = 44) gathers lane (i%PW)
 * of slot ((i/PW)*45 + 44).  One function per slot width. ---- */
__attribute__((target("fma"), noinline, unused))
static void dft45_tz1t8(const double *src, double *pl)
{
    VD T[45];
#define LSRC(i)     VLOAD(src + (long)(i) * 2)
#define SDST(i, v)  _mm_storeu_pd(pl + (long)(11 * 45 + (i)) * 8, (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

__attribute__((target("fma"), noinline, unused))
static void dft45_tz1t4(const double *src, double *pl)
{
    VD T[45];
#define LSRC(i)     VLOAD(src + (long)(i) * 2)
#define SDST(i, v)  _mm_storeu_pd(pl + (long)(22 * 45 + (i)) * 4, (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

__attribute__((target("fma"), noinline, unused))
static void dft45_ty1t8(const double *pl, double *dst)
{
    VD T[45];
#define LSRC(i)     VLOAD(pl + (long)(((i) / 4) * 45 + 44) * 8 + ((i) % 4) * 2)
#define SDST(i, v)  _mm_storeu_pd(dst + (long)(i) * (LSIDE * 2), (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

__attribute__((target("fma"), noinline, unused))
static void dft45_ty1t4(const double *pl, double *dst)
{
    VD T[45];
#define LSRC(i)     VLOAD(pl + (long)(((i) / 2) * 45 + 44) * 4 + ((i) % 2) * 2)
#define SDST(i, v)  _mm_storeu_pd(dst + (long)(i) * (LSIDE * 2), (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* ---- ice_r4 CXF: mapped y-tail twins.  One 128-bit line per plane (2025
 * of 91125 points), so the map here is the EXACT IEEE form (sqrt + divide,
 * bitwise the driver reference's arithmetic); no bias needed since
 * sqrt(0) = 0 is well-defined and w = 0 maps to 0. ---- */
__attribute__((target("fma"), always_inline, unused))
static inline __m128d map1_sd(__m128d v, __m128d c)
{
    __m128d w = _mm_add_pd(v, c);
    __m128d t = _mm_mul_pd(w, w);
    __m128d s = _mm_add_pd(t, _mm_shuffle_pd(t, t, 1));
    __m128d d = _mm_add_pd(_mm_set1_pd(1.0), _mm_sqrt_pd(s));
    return _mm_div_pd(w, d);
}

__attribute__((target("fma"), noinline, unused))
static void dft45_ty1t8m(const double *pl, double *dst, const double *cp)
{
    VD T[45];
#define LSRC(i)     VLOAD(pl + (long)(((i) / 4) * 45 + 44) * 8 + ((i) % 4) * 2)
#define SDST(i, v)  _mm_storeu_pd(dst + (long)(i) * (LSIDE * 2),          \
                        map1_sd((v), _mm_loadu_pd(cp + (long)(i) * (LSIDE * 2))))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

__attribute__((target("fma"), noinline, unused))
static void dft45_ty1t4m(const double *pl, double *dst, const double *cp)
{
    VD T[45];
#define LSRC(i)     VLOAD(pl + (long)(((i) / 2) * 45 + 44) * 4 + ((i) % 2) * 2)
#define SDST(i, v)  _mm_storeu_pd(dst + (long)(i) * (LSIDE * 2),          \
                        map1_sd((v), _mm_loadu_pd(cp + (long)(i) * (LSIDE * 2))))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

#undef VD
#undef VLOAD
#undef VADD
#undef VSUB
#undef VFMADD
#undef VFMSUB
#undef VFNMADD
#undef VSWAP

typedef void (*exec_fn)(const double *, double *, long, double *);
/* one fused chain step: FFT of the state with state<-(z+c)/(1+|z+c|) fused
 * into the final pass's stores; cv = this volume's c field.  Classic order
 * (cstep): arg0 = src (read only), arg1 = dst.  x-first order (cstepx):
 * arg0 = the state volume (transformed IN PLACE by pass A), arg1 = where
 * pass B's mapped y subloop lands (== arg0 except the last step). */
typedef void (*cstep_fn)(double *, double *, const double *, double *);

struct fft3d_plan {
    long batch;
    exec_fn fn;
    cstep_fn cstep;    /* fused map step for fft3d_chain (ice_r4) */
    int cxf;           /* nonzero: cstep is the x-first in-place form */
    double *plane;     /* 45 rows x PPITCH-complex pitch scratch for phase 1 */
    double *st1, *st2; /* chain state ping-pong volumes, mod-4096 skewed */
    void *raw;
    void *raw2;
};

/* one ERMS rep-movsb per x-plane: with cpy, the y pass stores into an L1-hot
 * plane image and this copies it to `out`.  ERMS full-cache-line writes do
 * not read-for-ownership the destination, so the 1.46 MB/volume RFO read of
 * cold `out` (L3-resident at B=1, DRAM in the streaming regime) disappears
 * from the traffic, while the lines still land in the cache hierarchy
 * (unlike NT stores, which the node rejected four rounds running at L=36
 * and which sweep 2 would then have to re-read from DRAM). */
static inline void plane_copy(void *dst, const void *src, size_t n)
{
#if defined(__x86_64__)
    void *d = dst;
    const void *s = src;
    size_t c = n;
    __asm__ volatile("rep movsb" : "+D"(d), "+S"(s), "+c"(c) : : "memory");
#else
    memcpy(dst, src, n);
#endif
}

/* exec_<variant>_<mech>; mechanism codes (see MJ[] in fft3d_create):
 *   0 pf0            no memory mechanism
 *   1 pf1            phase-2 45-stream T0 poke, 1 line ahead
 *   2 pfin           paced phase-1 input T1 prefetch (+ next-volume re-cover)
 *   3 pfw            paced phase-1 prefetchw on cold out, 1 plane ahead
 *   4 pfin-pfw       2 + 3
 *   5 pkw            phase-2 poke with PREFETCHW (in-place RMW pass: one
 *                    fetch-for-ownership instead of T0 read + store upgrade)
 *   6 pkw-pfin-pfw   5 + 2 + 3
 *   7 pf1-pfin-pfw   1 + 2 + 3   (streaming incumbent since r7)
 *   8 cpy            y pass -> plane image, rep movsb -> out (no-RFO stores)
 *   9 cpy-pfin       8 + 2
 *  10 pf3            phase-2 poke 3 lines ahead (NEW r11: L45_pfa's B=16
 *                    node winner is a bare distance-3 poke; distance 1 gives
 *                    less than one call's slack against DRAM latency, and a
 *                    bare poke was never fielded streaming here)
 *  11 pf3-pfin-pfw   10 + 2 + 3
 *  12 pf1-pfin-pfw-oc  7 with the r8 odd-column forms (oddc axis)
 *  13 pf3-oc           10 with the r8 odd-column forms
 *  14 tr               transpose-free phase 1 (NEW ice_r2, BORROWED from
 *                      L36_pfa ice_r2's tr=1): lane-transposed vectors built
 *                      as PW masked 128-bit broadcast loads (vbroadcastf64x2
 *                      from memory = one load-port uop on ICX, zero port 5),
 *                      codelet output stored UNtransposed to a slot-major
 *                      plane; both TRANSP passes and the Xv/Yv spill arrays
 *                      vanish from the z subloop.  On ICX the 2nd FMA pipe
 *                      shares port 5 with ALL 512-bit shuffles, so each
 *                      deleted shuffle is a recovered FMA slot.
 *  15 tr-pf1           14 + phase-2 poke 1 line ahead
 *  16 tr-pfw           14 + paced phase-1 prefetchw on cold out
 *  17 tr-pf1-pfw       14 + 1 + 3 (both r2-A/B winners stacked)
 *  18 tr-cpy           14 + 8 (ERMS no-RFO plane copy, retired r9, refielded:
 *                      pfw's +6.9% under tr says the out-RFO is now a stall)
 *  19 tr-cpy-pf1       18 + phase-2 poke
 *  20 tr-pf1-pfin-pfw  17 + paced T1 input prefetch (all overlap mechanisms)
 *  21 tr-pkw-pfw       16 + phase-2 PREFETCHW poke (in-place RMW pass:
 *                      fetch-for-ownership; never node-tested on ICX)
 */
#define DECL_EXEC(V)                                                     \
static void exec_##V##_0(const double *, double *, long, double *);      \
static void exec_##V##_1(const double *, double *, long, double *);      \
static void exec_##V##_2(const double *, double *, long, double *);      \
static void exec_##V##_3(const double *, double *, long, double *);      \
static void exec_##V##_4(const double *, double *, long, double *);      \
static void exec_##V##_5(const double *, double *, long, double *);      \
static void exec_##V##_6(const double *, double *, long, double *);      \
static void exec_##V##_7(const double *, double *, long, double *);      \
static void exec_##V##_8(const double *, double *, long, double *);      \
static void exec_##V##_9(const double *, double *, long, double *);      \
static void exec_##V##_10(const double *, double *, long, double *);     \
static void exec_##V##_11(const double *, double *, long, double *);     \
static void exec_##V##_12(const double *, double *, long, double *);     \
static void exec_##V##_13(const double *, double *, long, double *);     \
static void exec_##V##_14(const double *, double *, long, double *);     \
static void exec_##V##_15(const double *, double *, long, double *);     \
static void exec_##V##_16(const double *, double *, long, double *);     \
static void exec_##V##_17(const double *, double *, long, double *);     \
static void exec_##V##_18(const double *, double *, long, double *);     \
static void exec_##V##_19(const double *, double *, long, double *);     \
static void exec_##V##_20(const double *, double *, long, double *);     \
static void exec_##V##_21(const double *, double *, long, double *);     \
static void exec_##V##_p1(const double *, double *, long, double *);     \
static void exec_##V##_p2(const double *, double *, long, double *);     \
static void exec_##V##_p1z(const double *, double *, long, double *);    \
static void exec_##V##_p1y(const double *, double *, long, double *);    \
static void exec_##V##_p1t(const double *, double *, long, double *);    \
static void exec_##V##_p1zt(const double *, double *, long, double *);   \
static void exec_##V##_p1yt(const double *, double *, long, double *);   \
static void exec_##V##_cstep(double *, double *, const double *,         \
                             double *);                                   \
static void exec_##V##_cstepx(double *, double *, const double *,        \
                              double *);
DECL_EXEC(0)
DECL_EXEC(1)
DECL_EXEC(2)
#undef DECL_EXEC

/* ---- optional front-end counter probe (create()-time only, never on the
 * execute path).  perf_event_paranoid is 4 on this site's interactive
 * machines, so this very likely degrades to "fe=na" -- which, printed from
 * the scoring node, is itself the answer to why four rounds of §6 perf-stat
 * asks were never run.  If it DOES work there, one leaderboard line settles
 * the DSB-vs-MITE fork that is L=45's (and L=36's) only surviving B=1
 * hypothesis. ---- */
#if defined(__linux__) && defined(__x86_64__) && defined(__has_include)
#if __has_include(<linux/perf_event.h>)
#define HAVE_PERF_PROBE 1
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
static long perf_open1(unsigned type, unsigned long long conf, long group)
{
    struct perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.size = sizeof a;
    a.type = type;
    a.config = conf;
    a.disabled = (group < 0);
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    a.read_format = PERF_FORMAT_GROUP;
    return syscall(SYS_perf_event_open, &a, 0, -1, (int)group, 0UL);
}
#endif
#endif

/* instantiate the kernel once per (ISA, vector width) variant */
#define VAR 0
#include __FILE__
#undef VAR
#define VAR 1
#include __FILE__
#undef VAR
#define VAR 2
#include __FILE__
#undef VAR

/* ------------------------------ the API ---------------------------------- */

const char *fft3d_name(void) { return "L45_mixedradix"; }

static char g_desc[400];

const char *fft3d_description(void)
{
    return g_desc[0] ? g_desc
                     : "row-column PFA 9x5 line codelet, batch-vectorised over "
                       "lines, 2 sweeps, AVX2/AVX-512 + prefetch autotuned";
}

int fft3d_supports(int L) { return L == LSIDE; }

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != LSIDE || batch < 1) return NULL;

    fft3d_plan *p = (fft3d_plan *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->batch = batch;
    void *pl = NULL;
    /* [0 .. LSIDE*PPITCH) padded plane scratch for the z->y handoff;
     * [LSIDE*PPITCH .. +NPLANE) plane image for the cpy mechanism (laid out
     * exactly as one x-plane of `out`; offset 37440 B keeps it 64B-aligned) */
    /* 4096-aligned since ice_r4: pins the plane scratch's page phase so the
     * chain's phase geometry is deterministic (see fft3d_chain). */
    if (posix_memalign(&pl, 4096,
                       (size_t)(LSIDE * PPITCH + NPLANE) * 2 * sizeof(double)) != 0 || !pl) {
        free(p);
        return NULL;
    }
    p->plane = (double *)pl;
    p->raw = pl;
    p->fn = exec_0_0;

    /* chain state ping-pong: two private volumes.  Deliberately skewed off
     * the 4096-aligned block base (+1088 B and +1088+1458000+448 B -> page
     * phases 1088 and 1360) so neither shares a mod-4096 phase with the
     * driver's page-aligned x0/c/final_out buffers -- L23_rader ice_r4
     * measured a 16% per-step 4K-aliasing pathology living entirely in the
     * scratch block's allocation phase. */
    {
        void *r2 = NULL;
        size_t vol = (size_t)NVOL * 2;
        size_t tot = vol * 2 + 136 + 56 + 8;
        if (posix_memalign(&r2, 4096, tot * sizeof(double)) != 0 || !r2) {
            free(p->raw);
            free(p);
            return NULL;
        }
        memset(r2, 0, tot * sizeof(double));
        p->raw2 = r2;
        p->st1 = (double *)r2 + 136;
        p->st2 = p->st1 + vol + 56;
    }
    p->cxf = FFT45_CXF;
#if defined(__x86_64__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq"))
        p->cstep = FFT45_CXF ? exec_1_cstepx : exec_1_cstep;
    else if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512vl"))
        p->cstep = FFT45_CXF ? exec_2_cstepx : exec_2_cstep;
    else
        p->cstep = FFT45_CXF ? exec_0_cstepx : exec_0_cstep;
#else
    p->cstep = FFT45_CXF ? exec_0_cstepx : exec_0_cstep;
#endif

    /* regime: does the batch stream through this machine's LLC?  Decides
     * which candidates are IN PLAY (pfw only where `out` is genuinely cold;
     * two entries measured prefetchw at +11..17% on cache-resident volumes)
     * and how the tuning arena is sized. */
    long l3 = -1;
    {
#ifdef _SC_LEVEL3_CACHE_SIZE
        l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
        if (l3 <= 0) l3 = 22l << 20;   /* the scoring node's 22 MiB */
    }
    double foot = (double)batch * (double)NVOL * 16.0 * 2.0;
    int streaming = foot > 1.25 * (double)l3;

    /* self-tuning; all setup, excluded from the score.  Every candidate must
     * match exec_0_0's output to 1e-13 relative before it is eligible.
     * In the streaming regime the arena must actually STREAM on the machine
     * doing the tuning (L36_pfa r2 lesson, reproduced by L36_mixedradix r3):
     * in+out = 2.5x this machine's L3, clamped to [12, 64] volumes. */
    exec_fn cand[72];
    const char *cnm[72];
    int candv[72];
    int candtr[72];
    int ncand = 0;

    long nt;
    if (streaming) {
        long arena = (long)(2.5 * (double)l3 / ((double)NVOL * 32.0)) + 1;
        if (arena < 12) arena = 12;
        if (arena > 64) arena = 64;
        nt = batch < arena ? batch : arena;
    } else {
        nt = batch < 4 ? batch : 4;
    }
    size_t nd = (size_t)NVOL * 2 * (size_t)nt;
    double *ti = NULL, *o0 = NULL, *ox = NULL;
    if (posix_memalign((void **)&ti, 4096, nd * sizeof(double)) == 0 &&
        posix_memalign((void **)&o0, 4096, nd * sizeof(double)) == 0 &&
        posix_memalign((void **)&ox, 4096, nd * sizeof(double)) == 0) {

        uint64_t s = 0x9E3779B97F4A7C15ull;
        for (size_t i = 0; i < nd; ++i) {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            ti[i] = (double)(int64_t)(s >> 11) * (1.0 / 9007199254740992.0);
        }
        memset(o0, 0, nd * sizeof(double));
        exec_0_0(ti, o0, nt, p->plane);

        int have_512 = 0, have_vl = 0;
#if defined(__x86_64__)
        __builtin_cpu_init();
        have_512 = __builtin_cpu_supports("avx512f")
                   && __builtin_cpu_supports("avx512dq");
        have_vl  = have_512 && __builtin_cpu_supports("avx512vl");
#endif
        /* Mechanism table; indices are the exec_<v>_<mech> suffix.  r8 change:
         * the mechanisms are now ORTHOGONAL candidates instead of a cumulative
         * ladder.  Node r7 evidence (9/9 runs each): v1-pf0 won B=1/B=2 and
         * the verdict calls the phase-2 T0 poke "pure overhead on the scoring
         * machine" -- but pfin and pfw were only ever offered bundled WITH
         * that poke, so they were never cleanly tested where they matter most
         * (out does not fit the node's 1 MB L2 even at B=1, unlike L=36, so
         * the L36 "no pfw when cache-resident" rule does not map here). */
        static const struct {
            unsigned char pf, pfk, pfin, pfw, cpy, oc, tr;
            const char *nm;
        } MJ[22] = {
            {0,0,0,0,0,0,0,"pf0"},          {1,0,0,0,0,0,0,"pf1"},
            {0,0,1,0,0,0,0,"pfin"},         {0,0,0,1,0,0,0,"pfw"},
            {0,0,1,1,0,0,0,"pfin-pfw"},     {1,1,0,0,0,0,0,"pkw"},
            {1,1,1,1,0,0,0,"pkw-pfin-pfw"}, {1,0,1,1,0,0,0,"pf1-pfin-pfw"},
            {0,0,0,0,1,0,0,"cpy"},          {0,0,1,0,1,0,0,"cpy-pfin"},
            {3,0,0,0,0,0,0,"pf3"},          {3,0,1,1,0,0,0,"pf3-pfin-pfw"},
            {1,0,1,1,0,1,0,"pf1-pfin-pfw-oc"}, {3,0,0,0,0,1,0,"pf3-oc"},
            {0,0,0,0,0,0,1,"tr"},           {1,0,0,0,0,0,1,"tr-pf1"},
            {0,0,0,1,0,0,1,"tr-pfw"},       {1,0,0,1,0,0,1,"tr-pf1-pfw"},
            {0,0,0,0,1,0,1,"tr-cpy"},       {1,0,0,0,1,0,1,"tr-cpy-pf1"},
            {1,0,1,1,0,0,1,"tr-pf1-pfin-pfw"}, {1,1,0,1,0,0,1,"tr-pkw-pfw"},
        };
        exec_fn XT[3][22] = {
            {exec_0_0, exec_0_1, exec_0_2, exec_0_3, exec_0_4,
             exec_0_5, exec_0_6, exec_0_7, exec_0_8, exec_0_9,
             exec_0_10, exec_0_11, exec_0_12, exec_0_13,
             exec_0_14, exec_0_15, exec_0_16, exec_0_17, exec_0_18,
             exec_0_19, exec_0_20, exec_0_21},
            {exec_1_0, exec_1_1, exec_1_2, exec_1_3, exec_1_4,
             exec_1_5, exec_1_6, exec_1_7, exec_1_8, exec_1_9,
             exec_1_10, exec_1_11, exec_1_12, exec_1_13,
             exec_1_14, exec_1_15, exec_1_16, exec_1_17, exec_1_18,
             exec_1_19, exec_1_20, exec_1_21},
            {exec_2_0, exec_2_1, exec_2_2, exec_2_3, exec_2_4,
             exec_2_5, exec_2_6, exec_2_7, exec_2_8, exec_2_9,
             exec_2_10, exec_2_11, exec_2_12, exec_2_13,
             exec_2_14, exec_2_15, exec_2_16, exec_2_17, exec_2_18,
             exec_2_19, exec_2_20, exec_2_21},
        };
        /* diagnostic overrides for paired A/B runs, read once at plan time so
         * execution stays repeatable: FFT45_PFIN=0|1, FFT45_PFW=0|1,
         * FFT45_CPY=0|1, FFT45_PF=0|1|3|w (phase-2 poke: none / T0 1 ahead /
         * T0 3 ahead / prefetchw), FFT45_ODDC=0|1 (odd-column form: r9
         * LDCOL / r8 masked-transpose), FFT45_V=0|1|2 restricts the pool to
         * one kernel width. */
        int pfinmode = -1, pfwmode = -1, vmode = -1, cpymode = -1, pfmode = -1;
        int oddcmode = -1, trmode = -1;
        {
            const char *po = getenv("FFT45_PFIN");
            if (po && (*po == '0' || *po == '1')) pfinmode = *po - '0';
            const char *wo = getenv("FFT45_PFW");
            if (wo && (*wo == '0' || *wo == '1')) pfwmode = *wo - '0';
            const char *co = getenv("FFT45_CPY");
            if (co && (*co == '0' || *co == '1')) cpymode = *co - '0';
            const char *fo = getenv("FFT45_PF");
            if (fo && (*fo == '0' || *fo == '1' || *fo == '3'))
                pfmode = *fo - '0';
            else if (fo && (*fo == 'w' || *fo == 'W')) pfmode = 9;
            const char *oo = getenv("FFT45_ODDC");
            if (oo && (*oo == '0' || *oo == '1')) oddcmode = *oo - '0';
            const char *to = getenv("FFT45_TR");
            if (to && (*to == '0' || *to == '1')) trmode = *to - '0';
            const char *vo = getenv("FFT45_V");
            if (vo && *vo >= '0' && *vo <= '2') vmode = *vo - '0';
        }

        /* Pools as (variant, mech) pairs, INCUMBENT FIRST: the 3% hysteresis
         * pick favours earlier candidates.  r9 PRUNE, on r8's node evidence
         * (picks 9/9-stable in every cell, second consecutive round): the
         * cached survivors are v1-pf0 (+ pfw and pfin-pfw as the only
         * challengers never separated by >3% anywhere), the streaming
         * survivor is pf1-pfin-pfw; cpy and pkw lost every cell they were
         * offered in and are retired from the pools (code and env overrides
         * kept for monitor A/Bs).  v0 is now the genuine VEX/AVX2 kernel
         * (see the VAR-0 pragma note) and is only fielded when no 512-bit
         * kernel is available. */
        /* cached pool, ice_r2: tr twins lead (kernel changes compete from the
         * incumbent slot -- L36_pfa ice_r2's equal-rank principle mapped onto
         * this tuner's earliest-wins hysteresis; node A/B below confirmed tr
         * faster outside the noise band).  pf1/pf3 phase-2 pokes fielded
         * cached for the FIRST time: unlike L=36, this volume (1.39 MiB)
         * overflows the node's 1.25 MiB L2, so phase 2's 45 x-streams read
         * from L3 and the L=36 "no prefetch when cache-resident" rule was
         * never actually tested here. */
        /* r2 node tables (same-process arena, B=4 graded cell), two windows:
         *   w1: tr 306.2  tr-pf1 297.2  tr-pfw 285.1 | pf0 317.7  pf1 313.9
         *       pf3 320.3  pfw 310.4  pfin-pfw 313.5  v2 346.0
         *   w2: tr-pf1-pfw 265.8  tr-pfw 272.8  tr-pf1 281.7  tr 287.2
         *       tr-cpy 317.2  tr-cpy-pf1 311.3 | pfw 292.2  pf1 300.4
         *       pf0 301.4  v2 318.8
         * Order below = that ranking: tr-pf1-pfw heads (w2's winner by 2.6%,
         * which the 3% gate would otherwise never let displace tr-pfw);
         * untested overlap stacks (pfin / pkw twins) behind it as
         * challengers; cpy re-retired (lost by +16% under tr, matching its
         * r9 retirement -- ERMS pays more than the RFO it deletes on an
         * L3-resident chain). */
        static const unsigned char POOL_CACHED[][2] = {
            {1,17}, {1,21}, {1,16}, {1,15}, {1,14},
            {1,3}, {1,1}, {1,0}, {1,20},
            {2,0},
            {0,0}, {0,3}, {0,4},
        };
        /* streaming pool, r11.  HEAD = the r8-odd-column (-oc) twin of the
         * four-round incumbent mechanism: r7's 406.6 us B=16 -- this entry's
         * best-ever node number in that cell -- ran the masked odd-column
         * forms, the r9 LDCOL rework was never node-validated there (the
         * cell regressed +2.0% the round it shipped), and the r11 wallaby
         * B=64 in-arena table has the -oc twin fastest (236.7 vs 239.9).
         * Ties go to the earlier candidate, so the burden of proof now sits
         * on LDCOL in the streaming regime; {2,7} second keeps the reverse
         * test.  Then the poke-distance/bare-poke singles (L45_pfa's B=16
         * node winner is a BARE distance-3 poke, and this pool never before
         * contained a bare poke or any distance but 1; wallaby hates them --
         * 361 vs 240 -- but wallaby's DDR5 core is not the node). */
        static const unsigned char POOL_STREAM[][2] = {
            {2,12}, {2,7},
            {1,20}, {1,17},      /* tr overlap stacks (ice_r2, untested
                                    streaming -- challengers only) */
            {2,10}, {2,1},
            {2,11}, {2,4},
            {2,13},
            {1,7}, {1,10},
            {0,7}, {0,4},
        };
        /* any env override switches the candidate source from the pruned
         * pool to the FULL (variant x mechanism) matrix, so a forced A/B
         * (e.g. FFT45_CPY=1) can still reach a retired mechanism instead of
         * falling through to the v0 fallback pair. */
        int fullmat = (pfinmode >= 0 || pfwmode >= 0 || cpymode >= 0 ||
                       pfmode >= 0 || vmode >= 0 || oddcmode >= 0 ||
                       trmode >= 0);
        unsigned char fm[66][2];
        int nfm = 0;
        static const unsigned char VORD[3] = {1, 2, 0};
        for (int vi = 0; vi < 3; ++vi)
            for (int m = 0; m < 22; ++m) {
                fm[nfm][0] = VORD[vi];
                fm[nfm][1] = (unsigned char)m;
                ++nfm;
            }
        const unsigned char (*pool)[2] =
            fullmat ? fm : (streaming ? POOL_STREAM : POOL_CACHED);
        int npool = fullmat ? nfm
                            : (streaming ? (int)(sizeof POOL_STREAM / 2)
                                         : (int)(sizeof POOL_CACHED / 2));

        exec_fn probe[72];
        int pv[72];
        int ptr_[72];
        const char *pnm[72];
        static char pbuf[72][24];
        int nprobe = 0;
        for (int i = 0; i < npool; ++i) {
            int v = pool[i][0], m = pool[i][1];
            if (v == 1 && !have_512) continue;
            if (v == 2 && !have_vl) continue;
            if (!fullmat && v == 0 && have_vl) continue;
            if (vmode >= 0 && v != vmode) continue;
            if (pfinmode >= 0 && MJ[m].pfin != pfinmode) continue;
            if (pfwmode >= 0 && MJ[m].pfw != pfwmode) continue;
            if (cpymode >= 0 && MJ[m].cpy != cpymode) continue;
            if (oddcmode >= 0 && MJ[m].oc != oddcmode) continue;
            if (trmode >= 0 && MJ[m].tr != trmode) continue;
            if (pfmode == 0 && MJ[m].pf) continue;
            if (pfmode == 1 && !(MJ[m].pf == 1 && !MJ[m].pfk)) continue;
            if (pfmode == 3 && !(MJ[m].pf == 3 && !MJ[m].pfk)) continue;
            if (pfmode == 9 && !MJ[m].pfk) continue;
            snprintf(pbuf[nprobe], sizeof pbuf[0], "v%d-%s", v, MJ[m].nm);
            probe[nprobe] = XT[v][m];
            pv[nprobe] = v;
            ptr_[nprobe] = MJ[m].tr;
            pnm[nprobe] = pbuf[nprobe];
            ++nprobe;
        }
        for (int k = 0; k < nprobe; ++k) {
            memset(ox, 0, nd * sizeof(double));
            probe[k](ti, ox, nt, p->plane);
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < nd; ++i) {
                double d = ox[i] - o0[i];
                num += d * d;
                den += o0[i] * o0[i];
            }
            if (den > 0.0 && sqrt(num / den) < 1e-13) {
                cand[ncand] = probe[k];
                cnm[ncand] = pnm[k];
                candv[ncand] = pv[k];
                candtr[ncand] = ptr_[k];
                ++ncand;
            }
        }
        if (ncand == 0) {
            cand[ncand] = exec_0_0; cnm[ncand] = "v0-pf0-fallback";
            candv[ncand] = 0; candtr[ncand] = 0; ++ncand;
            cand[ncand] = exec_0_1; cnm[ncand] = "v0-pf1-fallback";
            candv[ncand] = 0; candtr[ncand] = 0; ++ncand;
        }

        /* time every survivor, several interleaved rounds, keep per-candidate
         * minimum; small arenas get more reps (under-sampling flipped picks
         * at L=36 until r4 fixed it this way). */
        double best[72];
        for (int k = 0; k < ncand; ++k) best[k] = 1e300;
        /* more rounds at tiny arenas: wallaby showed a fast/slow machine
         * state toggling on a seconds scale, and a toggle edge crossing the
         * tournament mis-ranks kernels; min-over-more-interleaved-rounds
         * gives every candidate a shot at the fast state. */
        int reps = nt >= 16 ? 1 : (nt >= 4 ? 4 : 16);
        int rounds = nt >= 16 ? 4 : (nt >= 4 ? 6 : 10);
        for (int round = 0; round < rounds; ++round) {
            for (int k = 0; k < ncand; ++k) {
                cand[k](ti, ox, nt, p->plane);        /* warm */
                double t0 = now_s();
                for (int r = 0; r < reps; ++r)
                    cand[k](ti, ox, nt, p->plane);
                double dt = now_s() - t0;
                if (dt < best[k]) best[k] = dt;
            }
        }
        int dbg = getenv("FFT45_VERBOSE") != NULL;
#ifdef FFT45_DEBUG
        dbg = 1;   /* tryout.sh cannot pass env; -DFFT45_DEBUG forces the
                      table + probe dump to stderr for node A/Bs */
#endif
        if (dbg)
            for (int k = 0; k < ncand; ++k)
                fprintf(stderr, "cand %-20s best %.3f us/vol\n",
                        cnm[k], best[k] * 1e6 / ((double)reps * (double)nt));
        /* hysteresis pick: a later (more speculative) candidate must beat the
         * best-so-far by a margin.  Cached: 3% (coin-flip zone measured at
         * 2.4% at L=36; wallaby's toggle needs the guard).  Streaming, r11:
         * 1.5% -- the node's in-arena streaming tournaments have been
         * 9/9-pick-stable with <0.5% spread for three rounds, and a 3% gate
         * there permanently blocks exactly the 2%-class mechanism moves
         * (poke distance, odd-column form) this round fields; the scored
         * picks happen on that quiet node, not on wallaby. */
        double hyst = streaming ? 0.985 : 0.97;
        int bk = 0;
        for (int k = 1; k < ncand; ++k)
            if (best[k] < hyst * best[bk]) bk = k;
        p->fn = cand[bk];

        /* ---- chain-step gate (ice_r4): the fused map step must match
         * exec_0_0 + the driver's exact scalar map to 1e-13 on one volume
         * (c := the input itself; any field exercises every lane).  A fast
         * wrong answer scores nothing; on mismatch fall back to the VAR-0
         * exact-sqrt/exact-div chain step. */
        {
            memset(o0, 0, (size_t)NVOL * 2 * sizeof(double));
            exec_0_0(ti, o0, 1, p->plane);
            for (long i = 0; i < (long)NVOL; ++i) {
                double re = o0[2 * i] + ti[2 * i];
                double im = o0[2 * i + 1] + ti[2 * i + 1];
                double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
                o0[2 * i] = re * sc;
                o0[2 * i + 1] = im * sc;
            }
            memset(ox, 0, (size_t)NVOL * 2 * sizeof(double));
            if (p->cxf) {
                memcpy(p->st1, ti, (size_t)NVOL * 2 * sizeof(double));
                p->cstep(p->st1, ox, ti, p->plane);
            } else {
                p->cstep((double *)ti, ox, ti, p->plane);
            }
            double num = 0.0, den = 0.0;
            for (long i = 0; i < (long)NVOL * 2; ++i) {
                double d = ox[i] - o0[i];
                num += d * d;
                den += o0[i] * o0[i];
            }
            if (!(den > 0.0) || !(sqrt(num / den) < 1e-13)) {
                p->cstep = exec_0_cstep;
                p->cxf = 0;
            }
        }

        /* ---- in-plan phase-split probe at nv=1 (borrowed from L36_pfa r8:
         * put the measurement the monitor never has time for inside create()
         * and route it out through the description).  fu = the picked exec;
         * p1 = phase 1 only; p2w = phase 2 only, in place on a warm out.
         * If fu - p1 - p2w is ~0 there is no phase-boundary penalty (the
         * L36 result); the p1/p2w split says where the 120-µs-over-floor
         * B=1 residue actually lives on the scoring machine. */
        exec_fn PRB1[3] = {exec_0_p1, exec_1_p1, exec_2_p1};
        exec_fn PRB2[3] = {exec_0_p2, exec_1_p2, exec_2_p2};
        exec_fn PRBZ[3] = {exec_0_p1z, exec_1_p1z, exec_2_p1z};
        exec_fn PRBY[3] = {exec_0_p1y, exec_1_p1y, exec_2_p1y};
        exec_fn PRBT[3] = {exec_0_p1t, exec_1_p1t, exec_2_p1t};
        exec_fn PRBZT[3] = {exec_0_p1zt, exec_1_p1zt, exec_2_p1zt};
        exec_fn PRBYT[3] = {exec_0_p1yt, exec_1_p1yt, exec_2_p1yt};
        int wv = candv[bk];
        double tfu = 1e300, tp1 = 1e300, tp2 = 1e300;
        double tpz = 1e300, tpy = 1e300, tpt = 1e300;
        double tpzt = 1e300, tpyt = 1e300;
        p->fn(ti, ox, 1, p->plane);            /* warm; fills ox volume 0 */
        for (int round = 0; round < 5; ++round) {
            double t0, dt;
            t0 = now_s();
            for (int r = 0; r < 6; ++r) p->fn(ti, ox, 1, p->plane);
            dt = (now_s() - t0) / 6.0; if (dt < tfu) tfu = dt;
            t0 = now_s();
            for (int r = 0; r < 6; ++r) PRB1[wv](ti, ox, 1, p->plane);
            dt = (now_s() - t0) / 6.0; if (dt < tp1) tp1 = dt;
            /* p2w retransforms ox in place; values blow up to inf/nan after
             * a few reps, which is timing-neutral on this ISA (only
             * denormals stall, and nan arithmetic produces none).  The
             * instruction stream is what is being timed. */
            t0 = now_s();
            for (int r = 0; r < 6; ++r) PRB2[wv](ti, ox, 1, p->plane);
            dt = (now_s() - t0) / 6.0; if (dt < tp2) tp2 = dt;
            /* subpass splits (NEW ice_r2, shape from L36_pfa's p1z/p1y):
             * p1z = phase-1 z subloop only, p1y = y subloop only, p1t = the
             * tr=1 phase 1 -- all tr=0/tr=1 comparable at nv=1 regardless
             * of which kernel won the tournament. */
            t0 = now_s();
            for (int r = 0; r < 6; ++r) PRBZ[wv](ti, ox, 1, p->plane);
            dt = (now_s() - t0) / 6.0; if (dt < tpz) tpz = dt;
            t0 = now_s();
            for (int r = 0; r < 6; ++r) PRBY[wv](ti, ox, 1, p->plane);
            dt = (now_s() - t0) / 6.0; if (dt < tpy) tpy = dt;
            t0 = now_s();
            for (int r = 0; r < 6; ++r) PRBT[wv](ti, ox, 1, p->plane);
            dt = (now_s() - t0) / 6.0; if (dt < tpt) tpt = dt;
            t0 = now_s();
            for (int r = 0; r < 6; ++r) PRBZT[wv](ti, ox, 1, p->plane);
            dt = (now_s() - t0) / 6.0; if (dt < tpzt) tpzt = dt;
            t0 = now_s();
            for (int r = 0; r < 6; ++r) PRBYT[wv](ti, ox, 1, p->plane);
            dt = (now_s() - t0) / 6.0; if (dt < tpyt) tpyt = dt;
        }

        /* ---- front-end counters over the picked exec at nv=1, if the
         * kernel lets an unprivileged process count at all (site machines
         * have perf_event_paranoid=4, so expect fe=na -- reporting THAT
         * from the node is still information). */
        char febuf[64] = "fe=na";
        char pobuf[72] = "";
#ifdef HAVE_PERF_PROBE
        {
            long fdc = perf_open1(PERF_TYPE_HARDWARE,
                                  PERF_COUNT_HW_CPU_CYCLES, -1);
            if (fdc < 0)
                snprintf(febuf, sizeof febuf, "fe=na(open:%d)", errno);
            if (fdc >= 0) {
                /* Skylake-family encodings, identical on CLX and SPR:
                 * idq.dsb_uops 0x08:0x79, idq.mite_uops 0x04:0x79,
                 * uops_issued.any 0x01:0x0e */
                long fdd = perf_open1(PERF_TYPE_RAW, 0x0879, fdc);
                long fdm = perf_open1(PERF_TYPE_RAW, 0x0479, fdc);
                long fdi = perf_open1(PERF_TYPE_RAW, 0x010e, fdc);
                if (fdd >= 0 && fdm >= 0 && fdi >= 0) {
                    ioctl((int)fdc, PERF_EVENT_IOC_RESET,
                          PERF_IOC_FLAG_GROUP);
                    ioctl((int)fdc, PERF_EVENT_IOC_ENABLE,
                          PERF_IOC_FLAG_GROUP);
                    for (int r = 0; r < 4; ++r)
                        p->fn(ti, ox, 1, p->plane);
                    ioctl((int)fdc, PERF_EVENT_IOC_DISABLE,
                          PERF_IOC_FLAG_GROUP);
                    unsigned long long pb[8] = {0};
                    ssize_t got = read((int)fdc, pb, sizeof pb);
                    if (got >= (ssize_t)(5 * sizeof pb[0]) && pb[0] == 4 &&
                        pb[1] > 0 && pb[2] + pb[3] > 0) {
                        double cyc = (double)pb[1], dsb = (double)pb[2];
                        double mite = (double)pb[3], iss = (double)pb[4];
                        snprintf(febuf, sizeof febuf,
                                 "fe dsb=%.2f mite=%.2f upc=%.2f",
                                 dsb / (dsb + mite), mite / (dsb + mite),
                                 iss / cyc);
                    }
                }
                if (fdd >= 0) close((int)fdd);
                if (fdm >= 0) close((int)fdm);
                if (fdi >= 0) close((int)fdi);
                close((int)fdc);
            }
            /* port group (NEW ice_r2; the node's PMU is exposed per the
             * brief): Ice Lake UOPS_DISPATCHED.PORT_x, event 0xA1 --
             * umask 0x01 p0 (FMA pipe 1), 0x20 p5 (FMA pipe 2 + ALL 512-bit
             * shuffles), 0x04 p2+p3 (loads), 0x10 p4+p9 (stores).  This
             * settles whether the B=1 residue is port 5, load/store, or
             * neither (front end / L2-L3 latency). */
            long fc2 = perf_open1(PERF_TYPE_HARDWARE,
                                  PERF_COUNT_HW_CPU_CYCLES, -1);
            if (fc2 >= 0) {
                long f0 = perf_open1(PERF_TYPE_RAW, 0x01a1, fc2);
                long f5 = perf_open1(PERF_TYPE_RAW, 0x20a1, fc2);
                long fl = perf_open1(PERF_TYPE_RAW, 0x04a1, fc2);
                long fs = perf_open1(PERF_TYPE_RAW, 0x10a1, fc2);
                if (f0 >= 0 && f5 >= 0 && fl >= 0 && fs >= 0) {
                    ioctl((int)fc2, PERF_EVENT_IOC_RESET,
                          PERF_IOC_FLAG_GROUP);
                    ioctl((int)fc2, PERF_EVENT_IOC_ENABLE,
                          PERF_IOC_FLAG_GROUP);
                    for (int r = 0; r < 4; ++r)
                        p->fn(ti, ox, 1, p->plane);
                    ioctl((int)fc2, PERF_EVENT_IOC_DISABLE,
                          PERF_IOC_FLAG_GROUP);
                    unsigned long long pb[8] = {0};
                    ssize_t got = read((int)fc2, pb, sizeof pb);
                    if (got >= (ssize_t)(6 * sizeof pb[0]) && pb[0] == 5 &&
                        pb[1] > 0) {
                        double cyc = (double)pb[1];
                        snprintf(pobuf, sizeof pobuf,
                                 "; po cyc/vol=%.0fk p0=%.2f p5=%.2f "
                                 "ld=%.2f st=%.2f",
                                 cyc / 4e3, (double)pb[2] / cyc,
                                 (double)pb[3] / cyc, (double)pb[4] / cyc,
                                 (double)pb[5] / cyc);
                    }
                }
                if (f0 >= 0) close((int)f0);
                if (f5 >= 0) close((int)f5);
                if (fl >= 0) close((int)fl);
                if (fs >= 0) close((int)fs);
                close((int)fc2);
            }
        }
#endif
        snprintf(g_desc, sizeof g_desc,
                 "PFA 9x5 2-sweep; pick=%s (B=%d, arena=%ld, stream=%d, "
                 "%d cand); chain=%s ms%d cpf%d/%d d%d w%d n%d k%d %s; "
                 "nv1 us fu=%.1f p1=%.1f p1z=%.1f p1y=%.1f "
                 "p1t=%.1f p1zt=%.1f p1yt=%.1f p2w=%.1f; %s%s",
                 cnm[bk], batch, nt, streaming, ncand,
                 p->cxf ? "xfirst-ym" : "fused-p2",
                 (int)FFT45_CMS, (int)FFT45_CPF, (int)FFT45_CPFC,
                 (int)FFT45_CPFD, (int)FFT45_CPFW,
                 (int)FFT45_CPFN, (int)FFT45_CPKW,
                 p->cstep == exec_0_cstep || p->cstep == exec_0_cstepx ? "v0"
                     : (p->cstep == exec_2_cstep || p->cstep == exec_2_cstepx
                            ? "v2" : "v1"),
                 tfu * 1e6, tp1 * 1e6, tpz * 1e6, tpy * 1e6,
                 tpt * 1e6, tpzt * 1e6, tpyt * 1e6, tp2 * 1e6,
                 febuf, pobuf);
        if (dbg)
            fprintf(stderr, "desc: %s\n", g_desc);
    }
    free(ti); free(o0); free(ox);
    return p;
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    plan->fn((const double *)in, (double *)out, plan->batch, plan->plane);
}

/* ice_r4: own the whole graded chain.  state_0 = x0; step s is
 * z = FFT(state_{s-1}); state_s = (z+c)/(1+|z+c|); final_out = state_m.
 * The map is fused into phase 2's in-place stores (every point of z is
 * produced exactly once there, in a register), so raw z is never
 * materialized and the step costs zero extra passes.  Volumes chain
 * independently, so volume b runs all m steps before b+1 (corpus 10 s3
 * consensus): per-volume working set = src + dst + c ~ 4.4 MiB, L3-resident
 * for the whole chain, where batch-per-step order would slosh the full
 * 11+ MiB batch through L3 every step.  Steps 1..m-1 ping-pong the two
 * private skewed buffers; step m writes final_out directly, so the driver's
 * page-aligned buffer is only ever a store target (one RFO per volume,
 * covered by the phase-1 prefetchw pacing). */
void fft3d_chain(fft3d_plan *plan, const double _Complex *x0,
                 const double _Complex *c, double _Complex *final_out, int m)
{
    if (!plan || m < 1) return;
    const double *xd = (const double *)x0;
    const double *cd = (const double *)c;
    double *fd = (double *)final_out;
    for (long b = 0; b < plan->batch; ++b) {
        const double *xb = xd + b * (long)NVOL * 2;
        const double *cb = cd + b * (long)NVOL * 2;
        double *fb = fd + b * (long)NVOL * 2;
        if (plan->cxf) {
            /* x-first in-place form: the state lives in ONE private volume
             * for the whole chain (one memcpy of x0 per volume = 1/177th of
             * a step); the last step's mapped y subloop writes final_out
             * directly.  The state base's mod-4096 phase is DERIVED from
             * c's at runtime (+2048, maximally far): the driver's buffers
             * are only 64B-aligned, so their page phase is heap luck, and
             * when c's phase lands near the state's, pass B's c loads
             * false-depend on the plane stores at equal page offsets --
             * measured 294 vs 333 us/xform across two runs of the SAME
             * binary at B=1 (L23_rader ice_r4 hit the same pathology).
             * plan->raw2 is 4096-aligned with > 1 page of slack. */
            double *st = (double *)((uintptr_t)plan->raw2
                                    + ((((uintptr_t)cb & 4095) + 2048) & 4095));
            memcpy(st, xb, (size_t)NVOL * 2 * sizeof(double));
            for (int s = 1; s <= m; ++s)
                plan->cstep(st, (s == m) ? fb : st, cb, plan->plane);
        } else {
            for (int s = 1; s <= m; ++s) {
                const double *sv = (s == 1) ? xb
                                            : ((s & 1) ? plan->st2 : plan->st1);
                double *dv = (s == m) ? fb : ((s & 1) ? plan->st1 : plan->st2);
                plan->cstep((double *)sv, dv, cb, plan->plane);
            }
        }
    }
}

void fft3d_destroy(fft3d_plan *plan)
{
    if (!plan) return;
    free(plan->raw);
    free(plan->raw2);
    free(plan);
}

#else /* ================= per-variant instantiation ======================== */

#define XCAT2(a, b) a##b
#define XCAT(a, b) XCAT2(a, b)
#define FN(n)     XCAT(XCAT(n##_, VAR), _0)
#define FNC(n, c) XCAT(XCAT(XCAT(n##_, VAR), _), c)

#if VAR == 1
/* ---- 512-bit: 4 complex lanes per zmm, 32 registers.  avx512dq is needed
 * only for the insertf64x2/extractf64x2 odd-column forms (r9); the scoring
 * node and wallaby both have it, and fft3d_create() gates on it. ---- */
#pragma GCC push_options
#pragma GCC target("avx512f,avx512dq")
#define PW 4
#define VD __m512d
#define VLOAD(p)          _mm512_loadu_pd(p)
#define VSTORE(p, v)      _mm512_storeu_pd((p), (v))
#define VLOADT(p)         _mm512_maskz_loadu_pd((__mmask8)0x03, (p))
#define VSTORET(p, v)     _mm512_mask_storeu_pd((p), (__mmask8)0x03, (v))
#define VADD(a, b)        _mm512_add_pd((a), (b))
#define VSUB(a, b)        _mm512_sub_pd((a), (b))
#define VMUL(a, b)        _mm512_mul_pd((a), (b))
#define VFMADD(a, b, c)   _mm512_fmadd_pd((a), (b), (c))
#define VFMSUB(a, b, c)   _mm512_fmsub_pd((a), (b), (c))
#define VFNMADD(a, b, c)  _mm512_fnmadd_pd((a), (b), (c))
#define VFMADDSUB(a,b,c)  _mm512_fmaddsub_pd((a), (b), (c))
#define VSWAP(a)          _mm512_permute_pd((a), 0x55)
#define PWLIST(M)         M(0) M(1) M(2) M(3)
/* odd-column (z=44 / k=44) forms, r9: lane j of the vector <-> the 128-bit
 * complex at p + j*STR.  Replaces a masked-load/PW-wide-TRANSP round trip
 * (12 instr in, 16 instr out) with 7 in / 4 out; bit-identical bytes move. */
#define LDCOL(p, STR)                                                     \
    _mm512_insertf64x2(_mm512_insertf64x2(_mm512_insertf64x2(             \
        _mm512_castpd128_pd512(_mm_loadu_pd(p)),                          \
        _mm_loadu_pd((p) + (STR)), 1),                                    \
        _mm_loadu_pd((p) + 2 * (STR)), 2),                                \
        _mm_loadu_pd((p) + 3 * (STR)), 3)
#define STCOL(p, STR, v) do {                                             \
    _mm_storeu_pd((p),               _mm512_castpd512_pd128(v));          \
    _mm_storeu_pd((p) +     (STR),   _mm512_extractf64x2_pd((v), 1));     \
    _mm_storeu_pd((p) + 2 * (STR),   _mm512_extractf64x2_pd((v), 2));     \
    _mm_storeu_pd((p) + 3 * (STR),   _mm512_extractf64x2_pd((v), 3));     \
} while (0)
/* 4x4 transpose of 128-bit complex lanes (involution) */
#define TRANSP(A, B) do {                                                 \
    VD z0 = _mm512_shuffle_f64x2((A)[0], (A)[1], 0x88);                   \
    VD z1 = _mm512_shuffle_f64x2((A)[0], (A)[1], 0xDD);                   \
    VD z2 = _mm512_shuffle_f64x2((A)[2], (A)[3], 0x88);                   \
    VD z3 = _mm512_shuffle_f64x2((A)[2], (A)[3], 0xDD);                   \
    (B)[0] = _mm512_shuffle_f64x2(z0, z2, 0x88);                          \
    (B)[2] = _mm512_shuffle_f64x2(z0, z2, 0xDD);                          \
    (B)[1] = _mm512_shuffle_f64x2(z1, z3, 0x88);                          \
    (B)[3] = _mm512_shuffle_f64x2(z1, z3, 0xDD);                          \
} while (0)
/* map primitives (ice_r4 fused chain) */
#define VDIVP(a, b)  _mm512_div_pd((a), (b))
#define VSQRTP(a)    _mm512_sqrt_pd(a)
#define VRSQ14(a)    _mm512_rsqrt14_pd(a)
#define VRCP14(a)    _mm512_rcp14_pd(a)
#define HAVE_RSQ 1
#else
/* ---- 256-bit: 2 complex lanes per ymm.  VAR 0 = "AVX2" (16 registers on a
   genuinely AVX2-only host), VAR 2 = EVEX/AVX-512VL (32 registers, no 512-bit
   path).  KNOWN AND ACCEPTED (r9 audit): `#pragma GCC target` ADDS to -march
   rather than restricting it, so on an AVX-512 build host VAR 0 compiles
   EVEX with ymm16-31, comes out byte-identical to VAR 2, and gcc's ICF folds
   every exec_2_* into `jmp exec_0_*` -- "v0" and "v2" have been ONE kernel in
   every scored binary (the r6 "V0 253 vs V2 190" wallaby A/B was pure
   fast/slow-window noise on identical code).  A real VEX fence cannot be
   compiled here: gcc 11's intrinsic headers bind to the command-line ISA
   when -march already has AVX-512, so a narrower per-function target cannot
   inline them (verified: both "no-avx512f" and "arch=haswell" fail to
   build).  Consequence drawn instead: v0 is fielded only on hosts without
   AVX-512, so the pools stop timing the same code twice under two names. */
#pragma GCC push_options
#if VAR == 0
#pragma GCC target("avx2,fma")
#else
#pragma GCC target("avx512vl,avx512f,fma")
#endif
#define PW 2
#define VD __m256d
#define VLOAD(p)          _mm256_loadu_pd(p)
#define VSTORE(p, v)      _mm256_storeu_pd((p), (v))
#define VLOADT(p)         _mm256_maskload_pd((p), _mm256_load_si256((const __m256i *)KC_TMASK))
#define VSTORET(p, v)     _mm256_maskstore_pd((p), _mm256_load_si256((const __m256i *)KC_TMASK), (v))
#define VADD(a, b)        _mm256_add_pd((a), (b))
#define VSUB(a, b)        _mm256_sub_pd((a), (b))
#define VMUL(a, b)        _mm256_mul_pd((a), (b))
#define VFMADD(a, b, c)   _mm256_fmadd_pd((a), (b), (c))
#define VFMSUB(a, b, c)   _mm256_fmsub_pd((a), (b), (c))
#define VFNMADD(a, b, c)  _mm256_fnmadd_pd((a), (b), (c))
#define VFMADDSUB(a,b,c)  _mm256_fmaddsub_pd((a), (b), (c))
#define VSWAP(a)          _mm256_permute_pd((a), 0x5)
#define PWLIST(M)         M(0) M(1)
/* odd-column forms, 2-lane widths: plain AVX insert/extract */
#define LDCOL(p, STR)                                                     \
    _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(p)),         \
                         _mm_loadu_pd((p) + (STR)), 1)
#define STCOL(p, STR, v) do {                                             \
    _mm_storeu_pd((p),         _mm256_castpd256_pd128(v));                \
    _mm_storeu_pd((p) + (STR), _mm256_extractf128_pd((v), 1));            \
} while (0)
/* 2x2 transpose of 128-bit complex lanes; involution */
#define TRANSP(A, B) do {                                        \
    VD z0 = _mm256_permute2f128_pd((A)[0], (A)[1], 0x20);        \
    VD z1 = _mm256_permute2f128_pd((A)[0], (A)[1], 0x31);        \
    (B)[0] = z0; (B)[1] = z1;                                    \
} while (0)
/* map primitives (ice_r4 fused chain); rsqrt14/rcp14 at 256 bits need
 * avx512vl, so only VAR 2 gets the ladder -- VAR 0 (plain AVX2) always
 * runs the exact vsqrtpd + vdivpd form regardless of FFT45_CMS. */
#define VDIVP(a, b)  _mm256_div_pd((a), (b))
#define VSQRTP(a)    _mm256_sqrt_pd(a)
#if VAR == 2
#define VRSQ14(a)    _mm256_rsqrt14_pd(a)
#define VRCP14(a)    _mm256_rcp14_pd(a)
#define HAVE_RSQ 1
#else
#define HAVE_RSQ 0
#endif
#endif

#define NB   ((LSIDE + PW - 1) / PW)   /* blocks per pass: 12 (PW=4), 23 (PW=2) */
#define NGF  (LSIDE / PW)              /* full blocks: 11 (PW=4), 22 (PW=2)     */
#define PFCH ((PFLINES + NB - 1) / NB) /* prefetch lines per paced call         */
/* r11: phase 1's z and y subloops run NGF full groups + one PW=1 tail line;
 * -DFFT45_R10TAIL restores the r6-r10 overlap-recompute tails (NB groups,
 * last one clamped) for a control build. */
#ifdef FFT45_R10TAIL
#define NGRP NB
#else
#define NGRP NGF
#endif

/* (the codelet macro text ST1G/ST2G/C_* moved to the common section in r11
 * so the PW=1 tail lines can expand the same DAG at 128 bits; see there) */

/* 45-point PFA line transform, y axis: reads the padded plane scratch
 * (row pitch 48 complex, 64B-aligned rows), writes `out` at its natural
 * 45-complex stride.  Strides are compile-time literals. */
static inline __attribute__((always_inline))
void FN(dft45_y)(const double *src, double *dst)
{
    VD T[45];
#define LSRC(i)     VLOAD(src + (long)(i) * (PPITCH * 2))
#define SDST(i, v)  VSTORE(dst + (long)(i) * (LSIDE * 2), (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* 45-point PFA line transform, x axis, in place: stride 2025 complex. */
static inline __attribute__((always_inline))
void FN(dft45_x)(double *base)
{
    VD T[45];
#define LSRC(i)     VLOAD(base + (long)(i) * (NPLANE * 2))
#define SDST(i, v)  VSTORE(base + (long)(i) * (NPLANE * 2), (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* masked tail of the x pass: the single line at flat index 2024 = (y,z) =
 * (44,44) (2025 is odd and the in-place pass cannot overlap-recompute).
 * Only lane 0 is live; the dead lanes compute on zeros and are masked off
 * at the store.  One call per volume. */
static inline __attribute__((always_inline))
void FN(dft45_xt)(double *base)
{
    VD T[45];
#define LSRC(i)     VLOADT(base + (long)(i) * (NPLANE * 2))
#define SDST(i, v)  VSTORET(base + (long)(i) * (NPLANE * 2), (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* ---- ice_r4: the graded-step map, one vector at a time.  v = the raw FFT
 * output vector (PW complex, interleaved), cp = the matching c vector.
 * Returns (v+c)/(1+|v+c|) per complex lane.  Full double precision by
 * construction: rsqrt14 seeds at 2^-14 rel err, two Newton steps take that
 * to 1.5*(1.5*(2^-14)^2)^2 = 4.7e-17 < ulp, |w| = s*y and d = fma(s,y,1)
 * are ~1 ulp, and the vdivpd (FFT45_CMS=0) is exact -- ~2-3 ulp per
 * application, so the chain gate (1e-13/step budget) is met by ~500x, where
 * the rival pipelines' float-seed map drifts to 1.28e-8 at m=4856.  Shape
 * borrowed from the rivals via corpus 10 s2 (one divider op per point,
 * Newton on the FMA pipes) with the seed upgraded to vrsqrt14pd's double
 * lookup -- same op count, full precision (L13_rader ice_r4's argument). */
static inline __attribute__((always_inline))
VD FN(mapv)(VD v, VD cv)
{
    VD w = VADD(v, cv);
    VD t = VFMADD(w, w, VLOAD(KC_TINYH));
    VD s = VADD(t, VSWAP(t));          /* |w|^2 + 1e-300 in both halves */
#if HAVE_RSQ && (FFT45_CMS == 0 || FFT45_CMS == 1)
    VD y = VRSQ14(s);
    VD h = VMUL(s, C_HALF);
    y = VMUL(y, VFNMADD(h, VMUL(y, y), VLOAD(KC_1P5)));
    y = VMUL(y, VFNMADD(h, VMUL(y, y), VLOAD(KC_1P5)));
    VD d = VFMADD(s, y, VLOAD(KC_ONE));   /* 1 + s/sqrt(s) = 1 + |w| */
#else
    VD d = VADD(VLOAD(KC_ONE), VSQRTP(s));
#endif
#if !HAVE_RSQ || FFT45_CMS == 0 || FFT45_CMS == 3
    return VDIVP(w, d);
#else
    VD r = VRCP14(d);                  /* divider-free variant */
    r = VMUL(r, VFNMADD(d, r, VLOAD(KC_TWO)));
    r = VMUL(r, VFNMADD(d, r, VLOAD(KC_TWO)));
    return VMUL(w, r);
#endif
}

/* x-axis line transform with the map fused into the in-place stores: the
 * only extra memory traffic the whole map costs is the c read (which
 * strides exactly like the data, one 64 B line per row per tile). */
static inline __attribute__((always_inline))
void FN(dft45_xm)(double *base, const double *cb)
{
    VD T[45];
#define LSRC(i)     VLOAD(base + (long)(i) * (NPLANE * 2))
#define SDST(i, v)  VSTORE(base + (long)(i) * (NPLANE * 2),               \
                           FN(mapv)((v), VLOAD(cb + (long)(i) * (NPLANE * 2))))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* masked tail twin (flat index 2024).  c is masked-loaded too (a full-width
 * load would read past the end of the driver's c buffer on the last
 * volume).  Dead lanes run the map on zeros: w = 0, s = 1e-300 (normal, no
 * denormal stall), no NaN on any style, and the masked store discards
 * them. */
static inline __attribute__((always_inline))
void FN(dft45_xtm)(double *base, const double *cb)
{
    VD T[45];
#define LSRC(i)     VLOADT(base + (long)(i) * (NPLANE * 2))
#define SDST(i, v)  VSTORET(base + (long)(i) * (NPLANE * 2),              \
                            FN(mapv)((v), VLOADT(cb + (long)(i) * (NPLANE * 2))))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* 45-point PFA line transform, vector array in/out (z axis) */
static inline __attribute__((always_inline))
void FN(dft45_v)(const VD *X, VD *Y)
{
    VD T[45];
#define LSRC(i)     X[(i)]
#define SDST(i, v)  Y[(i)] = (v)
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* ---- tr = transpose-free phase 1 (NEW ice_r2, BORROWED from L36_pfa
 * ice_r2's tr=1 rework, adapted to L=45's PFA 9x5 + odd tail).  BCG(P,STR)
 * builds a lane-transposed vector -- lane j = the 128-bit complex at
 * P + j*STR -- with ZERO port-5 uops at PW=4: vbroadcastf64x2 from memory
 * is one load-port uop on ICX, and the masked forms merge into lanes.
 * At PW=2 there is no masked-broadcast win to have (VAR 0 is plain AVX2
 * anyway); LDCOL keeps those variants correct so the tuner can price them.
 * The 4-deep merge dependency chain per vector is covered by the 45
 * independent vectors in flight (L36_pfa's codegen note). ---- */
#if PW == 4
#define BCG(P, STR)                                                       \
    _mm512_mask_broadcast_f64x2(_mm512_mask_broadcast_f64x2(              \
        _mm512_mask_broadcast_f64x2(                                      \
            _mm512_broadcast_f64x2(_mm_loadu_pd(P)),                      \
            (__mmask8)0x0C, _mm_loadu_pd((P) + (STR))),                   \
        (__mmask8)0x30, _mm_loadu_pd((P) + 2 * (STR))),                   \
        (__mmask8)0xC0, _mm_loadu_pd((P) + 3 * (STR)))
#define TZ1T dft45_tz1t8
#define TY1T dft45_ty1t8
#define TY1TM dft45_ty1t8m
#else
#define BCG(P, STR) LDCOL(P, STR)
#define TZ1T dft45_tz1t4
#define TY1T dft45_ty1t4
#define TY1TM dft45_ty1t4m
#endif

/* tr z-subloop codelet: inputs gathered lane-transposed straight from the
 * PW y-rows (lanes = y), outputs stored UNtransposed to the slot-major tr
 * plane (slot k of this y-group; PW*2 doubles per slot, 64B-aligned full
 * stores at PW=4).  Replaces TRANSP-in + dft45_v + TRANSP-out AND the
 * Xv/Yv spill arrays. */
static inline __attribute__((always_inline))
void FN(dft45_zt)(const double *rows, double *pl)
{
    VD T[45];
#define LSRC(i)     BCG(rows + (long)(i) * 2, LSIDE * 2)
#define SDST(k, v)  VSTORE(pl + (long)(k) * (PW * 2), (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* tr y-subloop codelet: element (y=i, kz=z0+j) lives at tr-plane slot
 * ((i/PW)*45 + z0+j), lane i%PW; pl arrives pre-offset by z0*(PW*2), so
 * lane j's granule is PW*2 doubles after lane j-1's.  Output layout is
 * identical to dft45_y (lanes = PW consecutive kz, contiguous in out). */
static inline __attribute__((always_inline))
void FN(dft45_yt)(const double *pl, double *dst)
{
    VD T[45];
#define LSRC(i)     BCG(pl + (long)((i) / PW) * 45 * (PW * 2) + ((i) % PW) * 2, PW * 2)
#define SDST(k, v)  VSTORE(dst + (long)(k) * (LSIDE * 2), (v))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

/* ice_r4 CXF: dft45_yt with the graded map fused into the stores.  In the
 * x-first step order the y subloop is the FINAL pass, so its outputs are
 * the completed 3D FFT values; c is loaded at the matching plane offsets
 * (plane-sequential across the volume). */
static inline __attribute__((always_inline))
void FN(dft45_ytm)(const double *pl, double *dst, const double *cp)
{
    VD T[45];
#define LSRC(i)     BCG(pl + (long)((i) / PW) * 45 * (PW * 2) + ((i) % PW) * 2, PW * 2)
#define SDST(k, v)  VSTORE(dst + (long)(k) * (LSIDE * 2),                 \
                        FN(mapv)((v), VLOAD(cp + (long)(k) * (LSIDE * 2))))
    REP9(ST1)
    REP5(ST2)
#undef LSRC
#undef SDST
}

static inline __attribute__((always_inline))
void FN(body)(const double *in, double *out, long batch, double *plane,
              const int pf, const int pfk, const int pfin, const int pfw,
              const int cpy, const int oddc, const int tr, const int ponly)
{
    const double *inend = in + batch * (long)NVOL * 2;
    double *outend = out + batch * (long)NVOL * 2;
    double *plane2 = plane + (long)LSIDE * PPITCH * 2;
    for (long b = 0; b < batch; ++b) {
        const double *vin  = in  + b * (long)NVOL * 2;
        double       *vout = out + b * (long)NVOL * 2;

        /* -------- phase 1: for each x-plane, z-lines then y-lines --------
         * ponly gates the phases for the create()-time phase-split probe
         * (ponly = 1 -> phase 1 only, 2 -> phase 2 only, 0 -> normal);
         * it is a compile-time literal, so scored execs carry no branch. */
        if (ponly != 2)
        for (long x = 0; x < LSIDE; ++x) {
            const double *pin  = vin  + x * (long)NPLANE * 2;
            double       *pout = vout + x * (long)NPLANE * 2;
            /* the input/output planes one ahead of the ones in flight; `in`
             * and `out` are contiguous across planes AND volumes, so this
             * crosses volume boundaries by itself (the PFNX cold window is
             * only what phase 2 later evicts). */
            const double *npf = pin  + (long)NPLANE * 2;
            double       *npw = pout + (long)NPLANE * 2;

            /* z pass: lanes = PW consecutive y-rows; NGF full groups, then
             * ONE true PW=1 (xmm) tail line for y=44 (r11, borrowed from
             * L45_pfa r10 -- the r6-r10 overlap-recompute tail burned a full
             * 344-op group per subloop per plane to cover one line).
             * tr=1 (ice_r2): the same groups through the broadcast-gather
             * codelet, no transposes, slot-major plane; ponly 3/4 gate the
             * two subloops for the p1z/p1y probes (compile-time literals,
             * scored execs carry no branch). */
            if (ponly != 4) {
            if (tr) {
                for (long yb = 0; yb < NGF; ++yb) {
                    if (pfin) {
                        long s = yb * PFCH, e = s + PFCH;
                        if (e > PFLINES) e = PFLINES;
#pragma GCC unroll 8
                        for (long i = s; i < e; ++i) {
                            const double *q = npf + i * 8;
                            if (q < inend)
                                _mm_prefetch((const char *)q, _MM_HINT_T1);
                        }
                    }
                    FN(dft45_zt)(pin + yb * (long)(PW * LSIDE * 2),
                                 plane + yb * (long)(45 * PW * 2));
                }
                if (pfin) {
                    long s = (long)NGF * PFCH;
#pragma GCC unroll 8
                    for (long i = s; i < PFLINES; ++i) {
                        const double *q = npf + i * 8;
                        if (q < inend)
                            _mm_prefetch((const char *)q, _MM_HINT_T1);
                    }
                }
                TZ1T(pin + 44 * (LSIDE * 2), plane);
            } else {
            for (long yb = 0; yb < NGRP; ++yb) {
                long y0 = yb * PW;
#ifdef FFT45_R10TAIL
                if (y0 > LSIDE - PW) y0 = LSIDE - PW;
#endif
                if (pfin) {
                    /* paced positional cursor: cover one plane of in, one
                     * plane ahead, spread over this subloop's NB calls */
                    long s = yb * PFCH, e = s + PFCH;
                    if (e > PFLINES) e = PFLINES;
#pragma GCC unroll 8
                    for (long i = s; i < e; ++i) {
                        const double *q = npf + i * 8;
                        if (q < inend)
                            _mm_prefetch((const char *)q, _MM_HINT_T1);
                    }
                }
                VD Xv[LSIDE], Yv[LSIDE];
                const double *rows = pin + y0 * (LSIDE * 2);
                /* explicit unrolls: the monitor's build has no -funroll-loops,
                 * and leaving these rolled cost 30% end to end (measured) */
#pragma GCC unroll 22
                for (long g = 0; g < NGF; ++g) {
                    VD A[PW], B[PW];
#define LDR(j) A[j] = VLOAD(rows + (long)(j) * (LSIDE * 2) + g * (PW * 2));
                    PWLIST(LDR)
#undef LDR
                    TRANSP(A, B);
#define PUT(j) Xv[g * PW + (j)] = B[j];
                    PWLIST(PUT)
#undef PUT
                }
                /* odd 45th column, z = 44.  oddc=0: 128-bit column gather
                 * (r9 LDCOL form, fewer instructions).  oddc=1: the r8
                 * masked-load + PW-wide transpose form.  r11 makes the choice
                 * a TOURNAMENT AXIS instead of the never-run -DFFT45_ODDR8
                 * compile A/B: the r9 verdict flagged the LDCOL rework as the
                 * only change on the B=16 path that coincided with a +2.0%
                 * node regression at an unchanged pick (and r10 regressed
                 * again), so streaming-pool candidates now carry -oc twins
                 * and the node prices the two forms itself. */
                if (oddc) {
                    VD A[PW], B[PW];
#define LDT(j) A[j] = VLOADT(rows + (long)(j) * (LSIDE * 2) + 44 * 2);
                    PWLIST(LDT)
#undef LDT
                    TRANSP(A, B);
                    Xv[44] = B[0];
                } else {
                    Xv[44] = LDCOL(rows + 44 * 2, LSIDE * 2);
                }
                FN(dft45_v)(Xv, Yv);
                double *prow = plane + y0 * (PPITCH * 2);
#pragma GCC unroll 22
                for (long g = 0; g < NGF; ++g) {
                    VD A[PW], B[PW];
#define GET(j) A[j] = Yv[g * PW + (j)];
                    PWLIST(GET)
#undef GET
                    TRANSP(A, B);
#define PST(j) VSTORE(prow + (long)(j) * (PPITCH * 2) + g * (PW * 2), B[j]);
                    PWLIST(PST)
#undef PST
                }
                /* odd column scatter: lane j of Yv[44] -> row j's z=44 slot
                 * (oddc=0: r9 128-bit extract-stores; oddc=1: r8 copies +
                 * transpose + masked stores -- same tournament axis). */
                if (oddc) {
                    VD A[PW], B[PW];
#define GT(j) A[j] = Yv[44];
                    PWLIST(GT)
#undef GT
                    TRANSP(A, B);
#define PTT(j) VSTORET(prow + (long)(j) * (PPITCH * 2) + 44 * 2, B[j]);
                    PWLIST(PTT)
#undef PTT
                } else {
                    STCOL(prow + 44 * 2, PPITCH * 2, Yv[44]);
                }
            }
#ifndef FFT45_R10TAIL
            if (pfin) {
                /* the pacing chunks the removed 12th group would have issued */
                long s = (long)NGF * PFCH;
#pragma GCC unroll 8
                for (long i = s; i < PFLINES; ++i) {
                    const double *q = npf + i * 8;
                    if (q < inend)
                        _mm_prefetch((const char *)q, _MM_HINT_T1);
                }
            }
            dft45_tz1(pin + 44 * (LSIDE * 2), plane + 44 * (PPITCH * 2));
#endif
            }   /* !tr */
            }   /* ponly != 4 */

            if (ponly == 3) continue;

            /* y pass: lanes = PW consecutive z; last block overlaps
             * (plane -> out is out-of-place, recompute is idempotent).
             * With cpy the stores go to the L1-hot plane image instead of
             * cold `out`, and one ERMS rep-movsb per plane moves the image
             * out without the RFO read (see plane_copy). */
            double *ydst = cpy ? plane2 : pout;
            if (tr) {
                for (long zb = 0; zb < NGF; ++zb) {
                    if (pfw) {
                        long s = zb * PFCH, e = s + PFCH;
                        if (e > PFLINES) e = PFLINES;
#pragma GCC unroll 8
                        for (long i = s; i < e; ++i) {
                            double *q = npw + i * 8;
                            if (q < outend)
                                __builtin_prefetch(q, 1, 3);
                        }
                    }
                    FN(dft45_yt)(plane + zb * (long)(PW * PW * 2),
                                 ydst + zb * (PW * 2));
                }
                if (pfw) {
                    long s = (long)NGF * PFCH;
#pragma GCC unroll 8
                    for (long i = s; i < PFLINES; ++i) {
                        double *q = npw + i * 8;
                        if (q < outend)
                            __builtin_prefetch(q, 1, 3);
                    }
                }
                TY1T(plane, ydst + 44 * 2);
            } else {
            for (long zb = 0; zb < NGRP; ++zb) {
                long z0 = zb * PW;
#ifdef FFT45_R10TAIL
                if (z0 > LSIDE - PW) z0 = LSIDE - PW;
#endif
                if (pfw) {
                    long s = zb * PFCH, e = s + PFCH;
                    if (e > PFLINES) e = PFLINES;
#pragma GCC unroll 8
                    for (long i = s; i < e; ++i) {
                        double *q = npw + i * 8;
                        if (q < outend)
                            __builtin_prefetch(q, 1, 3);
                    }
                }
                FN(dft45_y)(plane + z0 * 2, ydst + z0 * 2);
            }
#ifndef FFT45_R10TAIL
            if (pfw) {
                long s = (long)NGF * PFCH;
#pragma GCC unroll 8
                for (long i = s; i < PFLINES; ++i) {
                    double *q = npw + i * 8;
                    if (q < outend)
                        __builtin_prefetch(q, 1, 3);
                }
            }
            dft45_ty1(plane + 44 * 2, ydst + 44 * 2);
#endif
            }   /* !tr */
            if (cpy)
                plane_copy(pout, plane2, (size_t)NPLANE * 2 * sizeof(double));
        }

        if (ponly == 1 || ponly == 3 || ponly == 4) continue;

        /* pfin cold-window re-cover of in[b+1]'s first plane (phase 2's
         * traffic would evict what phase 1's cursor already fetched) */
        const double *ncw = (pfin && b + 1 < batch)
                                ? in + (b + 1) * (long)NVOL * 2
                                : (const double *)0;
        long cwl = 0;

        /* -------- phase 2: x-lines, in place in `out`, tiled over the FLAT
         * (y,z) index.  All 2025 lines of a y-z plane share the same x-stride
         * and are contiguous in flat index, so nothing forces tiles to respect
         * y boundaries: 2025 = 506*4 + 1 (PW=4) or 1012*2 + 1 (PW=2), i.e.
         * ONE masked tail call per volume instead of one per y (was 45).
         * Kills 33 full-width codelet calls per volume (~2% of the ops). */
        for (long t = 0; t < NPLANE / PW; ++t) {
            double *base = vout + t * (PW * 2);
            if (pf) {
                /* 45 x-streams, each advancing 64 B per tile: more than
                 * the L2 streamer tracks, poke them `pf` lines ahead.
                 * pfk=1 pokes with PREFETCHW instead of T0: this pass reads
                 * then rewrites every line in place, so one
                 * fetch-for-ownership replaces a read fetch plus a store
                 * upgrade (the shape of L64_radix8's node-picked pfw). */
#pragma GCC unroll 45
                for (int i = 0; i < LSIDE; ++i) {
                    const char *q =
                        (const char *)(base + (long)i * (NPLANE * 2) + pf * 8);
                    if (pfk)
                        __builtin_prefetch(q, 1, 3);
                    else
                        _mm_prefetch(q, _MM_HINT_T0);
                }
            }
            if (ncw && cwl < PFLINES) {
                _mm_prefetch((const char *)(ncw + cwl * 8), _MM_HINT_T1);
                ++cwl;
            }
            FN(dft45_x)(base);
        }
        FN(dft45_xt)(vout + (long)(NPLANE - 1) * 2);
    }
}

/* one wrapper per mechanism code; each instantiates the always_inline body
 * with compile-time-constant flags (pf, pfk, pfin, pfw, cpy, oddc, ponly).
 * -DFFT45_ODDR8 (a standing monitor flag from r10) forces the r8 odd-column
 * forms into EVERY exec, overriding the per-candidate oddc axis. */
#ifdef FFT45_ODDR8
#define ODDCX(x) 1
#else
#define ODDCX(x) (x)
#endif
#define MKEXEC(code, pf, pfk, pfi, pfw, cpy, oc, tr, po)                      \
static void FNC(exec, code)(const double *in, double *out, long batch,        \
                            double *plane)                                    \
{                                                                             \
    FN(body)(in, out, batch, plane, pf, pfk, pfi, pfw, cpy, ODDCX(oc), tr,    \
             po);                                                             \
}
MKEXEC(0,  0, 0, 0, 0, 0, 0, 0, 0)   /* pf0               */
MKEXEC(1,  1, 0, 0, 0, 0, 0, 0, 0)   /* pf1               */
MKEXEC(2,  0, 0, 1, 0, 0, 0, 0, 0)   /* pfin              */
MKEXEC(3,  0, 0, 0, 1, 0, 0, 0, 0)   /* pfw               */
MKEXEC(4,  0, 0, 1, 1, 0, 0, 0, 0)   /* pfin-pfw          */
MKEXEC(5,  1, 1, 0, 0, 0, 0, 0, 0)   /* pkw               */
MKEXEC(6,  1, 1, 1, 1, 0, 0, 0, 0)   /* pkw-pfin-pfw      */
MKEXEC(7,  1, 0, 1, 1, 0, 0, 0, 0)   /* pf1-pfin-pfw      */
MKEXEC(8,  0, 0, 0, 0, 1, 0, 0, 0)   /* cpy               */
MKEXEC(9,  0, 0, 1, 0, 1, 0, 0, 0)   /* cpy-pfin          */
MKEXEC(10, 3, 0, 0, 0, 0, 0, 0, 0)   /* pf3               */
MKEXEC(11, 3, 0, 1, 1, 0, 0, 0, 0)   /* pf3-pfin-pfw      */
MKEXEC(12, 1, 0, 1, 1, 0, 1, 0, 0)   /* pf1-pfin-pfw-oc   */
MKEXEC(13, 3, 0, 0, 0, 0, 1, 0, 0)   /* pf3-oc            */
MKEXEC(14, 0, 0, 0, 0, 0, 0, 1, 0)   /* tr                */
MKEXEC(15, 1, 0, 0, 0, 0, 0, 1, 0)   /* tr-pf1            */
MKEXEC(16, 0, 0, 0, 1, 0, 0, 1, 0)   /* tr-pfw            */
MKEXEC(17, 1, 0, 0, 1, 0, 0, 1, 0)   /* tr-pf1-pfw        */
MKEXEC(18, 0, 0, 0, 0, 1, 0, 1, 0)   /* tr-cpy            */
MKEXEC(19, 1, 0, 0, 0, 1, 0, 1, 0)   /* tr-cpy-pf1        */
MKEXEC(20, 1, 0, 1, 1, 0, 0, 1, 0)   /* tr-pf1-pfin-pfw   */
MKEXEC(21, 1, 1, 0, 1, 0, 0, 1, 0)   /* tr-pkw-pfw        */
/* create()-only phase-split probes (never in a candidate pool): pf0
 * mechanisms, one phase each.  Borrowed shape: L36_pfa r8's in-plan probe;
 * p1z/p1y/p1t (ice_r2) follow L36_pfa's subpass-split probes. */
MKEXEC(p1,  0, 0, 0, 0, 0, 0, 0, 1)  /* phase 1 only          */
MKEXEC(p2,  0, 0, 0, 0, 0, 0, 0, 2)  /* phase 2 only          */
MKEXEC(p1z, 0, 0, 0, 0, 0, 0, 0, 3)  /* phase 1 z subloop     */
MKEXEC(p1y, 0, 0, 0, 0, 0, 0, 0, 4)  /* phase 1 y subloop     */
MKEXEC(p1t, 0, 0, 0, 0, 0, 0, 1, 1)  /* tr=1 phase 1 only     */
MKEXEC(p1zt, 0, 0, 0, 0, 0, 0, 1, 3) /* tr=1 z subloop        */
MKEXEC(p1yt, 0, 0, 0, 0, 0, 0, 1, 4) /* tr=1 y subloop        */
#undef MKEXEC
#undef ODDCX

/* ---- ice_r4: one fused chain step, FFT(src) -> dst with the map applied at
 * phase 2's in-place stores.  Skeleton = the graded incumbent mechanism
 * (v1-tr-pf1-pfw) hardwired: tr phase 1 (no pfin -- src is L3-hot chain
 * state, a linear HW-prefetchable stream; pfin lost every cell since
 * panel_r7), prefetchw pacing over the dst store stream (the r2 evidence:
 * worth -2.8% exactly because this volume does NOT fit L2 and the store
 * targets sit in L3), and the phase-2 45-stream T0 poke extended to the 45
 * c streams (FFT45_CPFC).  All knobs compile-time so the chain output is
 * bit-identical across processes and runs. */
static void FNC(exec, cstep)(double *src_, double *dst,
                             const double *cv, double *plane)
{
    const double *src = src_;      /* read-only in this form */
    double *outend = dst + (long)NVOL * 2;

    /* phase 1, per x-plane: tr z subloop -> slot-major plane scratch,
     * then tr y subloop -> dst */
    for (long x = 0; x < LSIDE; ++x) {
        const double *pin = src + x * (long)NPLANE * 2;
        double *pout = dst + x * (long)NPLANE * 2;
        double *npw = pout + (long)NPLANE * 2;

        for (long yb = 0; yb < NGF; ++yb)
            FN(dft45_zt)(pin + yb * (long)(PW * LSIDE * 2),
                         plane + yb * (long)(45 * PW * 2));
        TZ1T(pin + 44 * (LSIDE * 2), plane);

        for (long zb = 0; zb < NGF; ++zb) {
#if FFT45_CPFW
            {
                long s = zb * PFCH, e = s + PFCH;
                if (e > PFLINES) e = PFLINES;
#pragma GCC unroll 8
                for (long i = s; i < e; ++i) {
                    double *q = npw + i * 8;
                    if (q < outend)
                        __builtin_prefetch(q, 1, 3);
                }
            }
#endif
            FN(dft45_yt)(plane + zb * (long)(PW * PW * 2),
                         pout + zb * (PW * 2));
        }
#if FFT45_CPFW
        {
            long s = (long)NGF * PFCH;
#pragma GCC unroll 8
            for (long i = s; i < PFLINES; ++i) {
                double *q = npw + i * 8;
                if (q < outend)
                    __builtin_prefetch(q, 1, 3);
            }
        }
#endif
        TY1T(plane, pout + 44 * 2);
    }

    /* phase 2: x-lines in place in dst, map fused into every store */
    for (long t = 0; t < NPLANE / PW; ++t) {
        double *base = dst + t * (PW * 2);
        const double *cbt = cv + t * (PW * 2);
#if FFT45_CPF
#pragma GCC unroll 45
        for (int i = 0; i < LSIDE; ++i)
            _mm_prefetch((const char *)(base + (long)i * (NPLANE * 2)
                                        + FFT45_CPFD * 8), _MM_HINT_T0);
#endif
#if FFT45_CPFC
#pragma GCC unroll 45
        for (int i = 0; i < LSIDE; ++i)
            _mm_prefetch((const char *)(cbt + (long)i * (NPLANE * 2)
                                        + FFT45_CPFD * 8), _MM_HINT_T0);
#endif
        FN(dft45_xm)(base, cbt);
    }
    FN(dft45_xtm)(dst + (long)(NPLANE - 1) * 2,
                  cv + (long)(NPLANE - 1) * 2);
}

/* ---- ice_r4 CXF: the x-first, in-place step.  Pass A = the unfused x-line
 * pass IN PLACE on the state volume (the FFT is separable, so the axis
 * order is free); pass B = per x-plane, tr z subloop into the slot-major
 * scratch, then the MAPPED y subloop back onto the same plane (or onto
 * final_out's plane on the last step).  The map's vector ops move into the
 * pass whose gather-latency slack can absorb them, the c reads become
 * plane-sequential, and the chain state needs only ONE private volume. */
static void FNC(exec, cstepx)(double *st, double *dstB,
                              const double *cv, double *plane)
{
    /* pass A: x-lines, in place (identical shape to the unfused phase 2;
     * the 45 RMW streams are poked with prefetchw so one fetch acquires
     * the line exclusive -- the pkw mechanism, r2's tied node winner) */
    for (long t = 0; t < NPLANE / PW; ++t) {
        double *base = st + t * (PW * 2);
#if FFT45_CPF
#pragma GCC unroll 45
        for (int i = 0; i < LSIDE; ++i) {
            char *q = (char *)(base + (long)i * (NPLANE * 2)
                               + FFT45_CPFD * 8);
#if FFT45_CPKW
            __builtin_prefetch(q, 1, 3);
#else
            _mm_prefetch((const char *)q, _MM_HINT_T0);
#endif
        }
#endif
        FN(dft45_x)(base);
    }
    FN(dft45_xt)(st + (long)(NPLANE - 1) * 2);

    /* pass B, per x-plane */
    for (long x = 0; x < LSIDE; ++x) {
        const double *pin = st + x * (long)NPLANE * 2;
        double *pout = dstB + x * (long)NPLANE * 2;
        const double *cp = cv + x * (long)NPLANE * 2;
#if FFT45_CPFN
        /* pace a T1 prefetch of the NEXT c plane over the z subloop: c is
         * re-read every step and a full step's traffic has evicted it to
         * L3 by the time its plane comes around again (same economics as
         * the pfw win in the classic form). */
        const double *ncp = (x + 1 < LSIDE) ? cp + (long)NPLANE * 2 : cp;
#endif
        for (long yb = 0; yb < NGF; ++yb) {
#if FFT45_CPFN
            {
                long s = yb * PFCH, e = s + PFCH;
                if (e > PFLINES) e = PFLINES;
#pragma GCC unroll 8
                for (long i = s; i < e; ++i)
                    _mm_prefetch((const char *)(ncp + i * 8), _MM_HINT_T1);
            }
#endif
            FN(dft45_zt)(pin + yb * (long)(PW * LSIDE * 2),
                         plane + yb * (long)(45 * PW * 2));
        }
#if FFT45_CPFN
        {
            long s = (long)NGF * PFCH;
#pragma GCC unroll 8
            for (long i = s; i < PFLINES; ++i)
                _mm_prefetch((const char *)(ncp + i * 8), _MM_HINT_T1);
        }
#endif
        TZ1T(pin + 44 * (LSIDE * 2), plane);

        for (long zb = 0; zb < NGF; ++zb) {
#if FFT45_CPFY
#pragma GCC unroll 45
            for (int k = 0; k < LSIDE; ++k)
                _mm_prefetch((const char *)(cp + (long)k * (LSIDE * 2)
                                            + (zb + 1) * (PW * 2)),
                             _MM_HINT_T0);
#endif
            FN(dft45_ytm)(plane + zb * (long)(PW * PW * 2),
                          pout + zb * (PW * 2), cp + zb * (PW * 2));
        }
        TY1TM(plane, pout + 44 * 2, cp + 44 * 2);
    }
}

#pragma GCC pop_options

#undef XCAT2
#undef XCAT
#undef FN
#undef FNC
#undef PW
#undef VD
#undef VLOAD
#undef VSTORE
#undef VLOADT
#undef VSTORET
#undef VADD
#undef VSUB
#undef VMUL
#undef VFMADD
#undef VFMSUB
#undef VFNMADD
#undef VFMADDSUB
#undef VSWAP
#undef PWLIST
#undef LDCOL
#undef STCOL
#undef TRANSP
#undef BCG
#undef TZ1T
#undef TY1T
#undef TY1TM
#undef NB
#undef NGF
#undef PFCH
#undef VDIVP
#undef VSQRTP
#ifdef VRSQ14
#undef VRSQ14
#undef VRCP14
#endif
#undef HAVE_RSQ

#endif /* VAR */
