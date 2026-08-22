/* Carried over from the SINGLE-THREAD competition, where this file finished as
 * written below. Your job in the multicore phase is to parallelise it across
 * 32 cores without losing its single-core efficiency -- read
 * ../PANEL_BRIEF.md, and read ../../geom/strategies/L45_pfa.md for the full
 * history of how this kernel got here.
 */
/* L45_pfa.c -- forward complex 3D DFT of a 45^3 cube, batched, out-of-place.
 *
 * ROUND mt_r2 (second multicore round).  Node verdict on mt_r1: B=256 WON
 * (26.9 us/vol, mtn-pfi, 2.15x fftw3_patient); B=1 lost to L45_mixedradix
 * 60.6 vs 58.3 and B=16 lost 21.2 vs 16.9 -- both to their persistent
 * pinned pthread SPIN POOL (GOMP fork+barrier+join is a measured 6-8 us
 * per execute, L23_matrixsimd mt_r1) and, at B=16, their grp16x2 pairing
 * (2 threads per volume, ONE cheap pair barrier per volume, no global
 * straggler coupling).  The serial kernel is again UNTOUCHED; the changes
 * are all in the threading layer:
 *   1. The OpenMP execute regions are GONE, replaced by a persistent
 *      pinned spin pool built in fft3d_create(): 31 workers pinned to the
 *      CPUs one throwaway OMP region reports (so the pool lands exactly
 *      where OMP_PROC_BIND=close would), per-thread scratch allocated and
 *      first-touched by its owner after pinning, flag-array barriers
 *      (padded per-thread arrival flags + one release word; ~0.3-0.5 us
 *      vs GOMP's 6-8).  BORROWED wholesale from L23_matrixsimd mt_r1 via
 *      L45_mixedradix mt_r1, including the unbound-run pinning guard.
 *   2. NEW grp2 exec class (the rival's B=16 winner, taken with
 *      attribution): 2 adjacent threads own each volume block; member m
 *      does x-planes [45m/2,45(m+1)/2), one PAIR barrier (not global),
 *      then phase-2 tiles [507m/2, 507(m+1)/2).  Raced plain (g2-pf0),
 *      with pfin+prefetchw (g2-pfw = their v1 mech), and in a NT-staged
 *      form they do not have (g2n): phase 1 into the pair's private M,
 *      pair barrier, phase 2 via the staged-tile NT path below -- deletes
 *      the out-RFO at node B=16, which is streaming there (89 MB in+out
 *      vs 76 MB aggregate cache).
 *   3. NEW mts exec class (staged-tile NT, the B=256 refinement): phase 2
 *      reads M and writes each block of 16 flat tiles through a 45x128-
 *      double tile buffer flushed as 45 short NT segments -- against mtn
 *      (phase 2 in place in M + one linear ntcopy) this deletes one full
 *      write AND one full re-read of M per volume (~2.9 MB of L2/L3
 *      traffic; M at 1.46 MB does not fit the node's 1 MiB L2, so that
 *      traffic is real there).  Segment head/tail peels are regular
 *      stores: ~6% of out pays RFO at TBK=16, the price of out's 16 B
 *      per-plane phase rotation.
 *   4. DELETED per node evidence: the r11 zal aligned-z-load twins (zero
 *      node picks at any cell in mt_r1 -- the split-load class lost on
 *      CLX too), mtf-nt (lost every wallaby cell in r1 and took no node
 *      pick), and mtn-t16 (the VERDICT falsified the socket-0 premise:
 *      T=32 won every batched cell on the node; L=6 B=65536 sustains
 *      175 GB/s = both sockets' controllers, likely AutoNUMA migration).
 *   5. mtf (fused two-sweep) kept on the pool for B=1/small B (rr and blk
 *      plane schedules, the node's r1 picks), plus a blk+pf twin and a
 *      T=23 probe (L45_mixedradix's wallaby B=1 pick was T=23).
 *   Everything is still raced at plan time on the real batch with the
 *   r1 arena (nv = min(batch,128), serial fill, gate on first AND last
 *   volume, interleaved min, 3% simplest-first hysteresis).
 *
 * ROUND mt_r1 (first multicore round).  The serial kernel below is UNCHANGED
 * (same 344-op Good-Thomas 9x5 codelet, same two-sweep plane-fused schedule,
 * same 514,968 zmm FMA-port ops per volume); what is new is a threading
 * layer, tuned at plan time:
 *   1. mtv -- volume-parallel (the B>=32 shape): thread t owns the
 *      contiguous volume block [nvol*t/T, nvol*(t+1)/T) and runs the settled
 *      serial per-volume pipeline on it with its OWN plane scratch; no
 *      synchronisation at all except the join.  pf0/pf3/zal twins kept (the
 *      pf ladder re-raced under 32-thread contention: r11's single-core
 *      verdicts do not transfer to the bandwidth-shared regime).
 *   2. mtf -- fused two-sweep (the B=1 shape): omp-for(static,1) over ALL
 *      nvol*45 x-plane pipelines (z+y transform of one plane into mid=out,
 *      each thread on its own plane scratch), ONE implicit barrier (every
 *      phase-2 line reads all 45 mid planes of its volume), omp-for(static)
 *      over nvol*507 flat phase-2 tiles (506 full + the masked tail as its
 *      own unit).  static,1 on the plane sweep because 45 = 32+13: a blocked
 *      static would idle 9 threads behind 2-plane threads at B=1.
 *   3. mtn -- mtv with NT-STAGED output (the streaming shape): both phases
 *      run in the thread's PRIVATE mid volume M (1.46 MB, one SPR L2), then
 *      one linear vmovntpd burst (head/tail peel; out bases rotate 16 B per
 *      volume) flushes M -> out, deleting the out-RFO third of the DRAM
 *      traffic.  Direct NT on the transform's stores is impossible (16B-
 *      aligned 64 B stores, 720 B rows); staging is how L13_rader/L17_rader
 *      shipped it.  The NT-wins-at-threaded-streaming inversion is their
 *      measured result (-13..-25%), re-raced here per machine.
 *   4. Per-thread plane scratch Pt[t] (37 KB each) is allocated and
 *      FIRST-TOUCHED by thread t inside fft3d_create()'s parallel region
 *      (which also spins up the OpenMP pool): NUMA-local under
 *      OMP_PROC_BIND=close, page-aligned so threads never share a scratch
 *      line.  The caller's in/out are first-touched by the driver's main
 *      thread (one socket owns them on the node) -- not ours to fix.
 *   5. The SCRATCHP mode (padded mid-volume S) and the serial pf1/pf2 rungs
 *      are DELETED: sp took zero node picks in five phase-1 rounds (its
 *      pre-registered exit), and the serial rows kept below (ip-pf0, ip-pf3)
 *      are controls only -- with ~300 us of work per volume against a
 *      measured 3-8 us GOMP region cost (L13/L17/L23 mt_r1 records), the
 *      "B=1 does not parallelise" outcome of the small-L entries does not
 *      apply at L=45, and the create()-time race proves it per machine.
 *   OpenMP regions (not a custom pool) because the smallest scored cell has
 *   ~300 us of work; the L17/L23 spin pools attack a 2-8 us sync floor that
 *   is <3% here.  Threading is attributed to L13_direct mt_r1 (range execs,
 *   owner-first-touch, contiguous blocks) and L23_rader mt_r1 (fused-vs-
 *   batch decomposition split, "intra-volume splitting pays iff work >>
 *   region cost").
 *
 * ROUND panel_r11 (sixth round).  Both changes attack phase 1's data
 * movement, per the r10 VERDICT (phase 1 = 76% of B=1, port 0 non-binding,
 * front end closed; §4.1 priced the spill difference between the two L=45
 * entries' takes of the same DFT9 DAG):
 *   1. Per-site codelet stage order.  The memory-store sites (y-subloop,
 *      phase 2, PW=1 tail lines) now run the codelet DFT5-FIRST into
 *      T_[9*k2+n1], then DFT9s that read 9 contiguous hot slots and hand
 *      every output straight to the store macro (L45_mixedradix's
 *      ST1G/ST2G shape; VERDICT §4.1: their take of the identical DAG
 *      added +3 stack moves where mine added +23, worth ~3 points of B=1).
 *      The z-site, whose ST writes a vec array that is transposed
 *      afterwards, keeps the old DFT9-first order (PFA45R): a per-function
 *      audit under node flags shows the store-direct order is WORSE there
 *      (phase1_plane spills 83 -> 91) and better at the memory sites
 *      (phase2 spills 53/55 -> 46/51).  Same DAG, same 344 ops, same maps.
 *   2. NEW zal exec variants (pf0a/pf3a): the z-pass in-loads become
 *      aligned rolling loads recombined with valignq -- see the ZLA comment
 *      in the template for the phase arithmetic (row phase is
 *      16*(b+x+j) mod 64 within every y-group because yb == 0 mod PW, so
 *      ONE per-plane value zc selects among 4 compile-time shift patterns).
 *      Deletes ~15k of the ~16k split z-loads per volume.  Node-only by
 *      construction (wallaby's SPR core hides splits; its arena prices
 *      pf0a at +3.2% over pf0, as pre-registered); shipped tuner-gated so
 *      the node's tournament prices the split-load class directly -- the
 *      r10 record's named next step.  pf codes 4/5 are REUSED from the r10
 *      pfp candidates, which took zero node picks (3/3 pf0 at B=1/B=2,
 *      3/3 pf3 at B=16) and are deleted per their pre-registered branch
 *      ("closed for good").
 *
 * ROUND panel_r10.  r10 changes:
 *   1. The phase-1 overlap-recompute tails are replaced by true PW=1
 *      (128-bit) tail lines.  45 = 11*4 + 1, so the z and y subpasses each
 *      ran 12 full-width groups where 11.25 are needed: 90 full zmm codelet
 *      calls per volume (30,960 port-0 ops, plus their 44-load transpose
 *      blocks / 45-store rows) were recompute.  Now 11 full groups + ONE
 *      PW=1 line per subpass per plane: the 45-point codelet on a single
 *      complex per xmm vector (VEX-coded, dual-issues on ports 0 AND 1 on
 *      CLX, no transposes, 16 B loads/stores that never split a line).
 *      zmm calls/volume 1587 -> 1497 (-5.7% port-0); the 90 xmm lines add
 *      ~15.5k port-0-equivalent cycles back.  This REVERSES r7's change 3/4
 *      (overlap tails, chosen then for instruction-count leanness): the
 *      front-end premise died in r9 (L36's triple null), and the r7 form
 *      recomputed 8.3% of phase 1's full-width group work.
 *      -DFFT45_OVERLAP_TAIL restores the r7-r9 overlap form for a control
 *      (and compiles the r11 zal candidates out -- their phase arithmetic
 *      needs yb == 0 mod PW, which the clamped overlap group breaks).
 *   2. pl-column prefetch (pfp) candidates -- took zero node picks in r10,
 *      DELETED in r11 per their pre-registered branch.
 *
 * ROUND panel_r9 (fourth round).  r9 changes:
 *   1. DFT9 module replaced: the hand CT 3x3 form (44 FMA-port vector ops +
 *      10 swaps) is replaced by a pairwise transcription of genfft's FMA
 *      n1_9 DAG (fftw-3.3.10/dft/scalar/codelets/n1_9.c: 24 add + 56 fma =
 *      80 scalar FMA-port ops) onto interleaved-complex vectors: 40 FMA-port
 *      ops + 12 swaps.  Every scalar re/im line pair maps to ONE vector op;
 *      the points where the scalar DAG crosses re and im (multiplies by i and
 *      the (1 + c*i) spiral factors) each cost one SWAP, with all signs
 *      folded into VPAIR constants.  Per line: 364 -> 344 port-0 ops (-5.5%,
 *      the node's bottleneck port), 68 -> 78 port-5 shuffles (still <= half
 *      of port 0).  L36_pfa r1 burned three attempts deriving this by hand;
 *      the transcription of the actual generated DAG is mechanical and the
 *      create()-time reference gate proves it.
 *   2. The file-level '#pragma GCC optimize("unroll-loops")' is REMOVED
 *      (r8 VERDICT 3c: the scored build has carried -funroll-loops in its
 *      Makefile all along, so the pragma's premise was false, and L17_rader
 *      measured the pragma FORM as a ~2% tax because optimize() rebuilds the
 *      whole per-function option set).  -DFFT45_UNROLL_PRAGMA restores it
 *      for A/B.
 *
 * ROUND panel_r8 (r6, r7 before).  r8 changes:
 *   1. Compile-time specialization of the prefetch ladder and the mid-buffer
 *      layout (borrowed from L45_mixedradix's exec-variant structure): phase 1
 *      and phase 2 are always_inline bodies taking const flags and const
 *      strides, instantiated into per-(pw, mode, pf) exec functions.  The
 *      node's r7 pick at B=1/B=2 was pw4-inplace-pf0, and in r7 that path
 *      still carried runtime pf branches, two cursor pointers per plane, and
 *      ~180 never-executed prefetch instructions inside the hot loops (my
 *      3535 hot instructions vs the rival's 3134 -- and they beat me 328 vs
 *      343 us at equal op count).  Now pf0 compiles to zero prefetch code.
 *   2. NEW padded-scratch mode (scratchp): phase 1 writes a plan-owned S with
 *      row pitch 52 complex (832 B = 13 lines, odd, 64B-aligned rows) and
 *      plane pitch 45*52 = 2340 complex (37440 B = 585 lines, odd).  The
 *      y-pass stores and the x-pass loads become 64-byte ALIGNED, deleting
 *      two of the three split-access classes (~75% of the 64 B accesses at
 *      the natural 720 B row stride split a cache line; only the x-pass
 *      stores to `out` keep paying the odd-L toll).  The odd-line pitches
 *      also break the same-stride read/write mod-4096 correlation that
 *      L23_rader measured at -25..30% (their self-inflicted-aliasing rule):
 *      r6's unpadded scratch read S and wrote out at the SAME 32400 B plane
 *      stride and lost ~66% -- padding recovers most of that (wallaby r8:
 *      sp is within 11-13% of inplace at B=1, was ~66% behind) but does not
 *      win there; it is a candidate for the node's different cache/split
 *      physics, not a wallaby winner.  scratchp's x pass is out-of-place
 *      (S -> out), so its odd
 *      tails overlap-recompute per y-row (12 tiles/row at PW=4, +2.2% volume
 *      ops vs flat tiling) instead of needing masked calls.
 *
 * TECHNIQUE (unchanged from r7 otherwise)
 *   Row-column 3D DFT; every 45-point line is a Good-Thomas / prime-factor
 *   9 x 5 codelet, on INTERLEAVED complex vectors whose lanes are a spectator
 *   axis.  gcd(9,5) = 1, so with
 *
 *       input  (Ruritanian): n = (5*n1 + 9*n2) mod 45     n1 in [0,9), n2 in [0,5)
 *       output (CRT):        k = (10*k1 + 36*k2) mod 45   (10 = 5*[5^-1]_9,
 *                                                          36 = 9*[9^-1]_5)
 *
 *   W45^{nk} = W9^{n1 k1} * W5^{n2 k2} exactly: 5 DFT9s then 9 DFT5s with NO
 *   twiddles in between, both maps folded into compile-time addressing.
 *   DFT9 = genfft n1_9 FMA DAG transcribed to interleaved vectors (40
 *   FMA-port ops + 12 swaps, r9), DFT5 = FFTW n1_5's FMA form (16 ops).
 *   Per 45-point line: 344 FMA-port vector ops + 78 shuffles.
 *
 *   Two sweeps (the structure that won L=36 and L=45 on the node):
 *   phase 1, per x-plane:
 *       z transform: lanes = PW y-rows, PWxPW complex-granule register
 *                    transposes on load and store, into plane scratch
 *                    pl[y][kz] (row pitch PPITCH = 52 complex, 13 lines,
 *                    coprime with the 64 L1 sets)
 *       y transform: lanes = PW kz (contiguous in pl), store to mid[x][ky][kz]
 *                    where mid = out (INPLACE) or the padded S (SCRATCHP)
 *   phase 2:
 *       x transform: lanes = PW kz, 45 streams at the plane stride.
 *       INPLACE:  mid = out, in place; tiles the FLAT (y,z) index
 *                 (2025 = 506*4 + 1: ONE masked tail call per volume).
 *       SCRATCHP: S -> out, out of place; tiles per y-row with an
 *                 overlapping last tile (rows are padded so flat tiling
 *                 does not map affinely to out).
 *
 *   Tail policy (r10): phase 1's z and y subpasses run NFULL (11) full-width
 *   groups plus ONE PW=1 xmm line each (dft45_line1) for the odd 45th
 *   row/column -- no overlap recompute; the z-line's odd 45th column within
 *   a group is a 16 B column gather/scatter (GCOL/SCOL); the in-place x
 *   pass keeps the single masked flat-tail call; scratchp's out-of-place x
 *   pass keeps its per-row overlapping tile (sp takes no node picks; not
 *   worth a third tail scheme).
 *
 *   A transpose-count note for the record: a lanes=x phase-1 variant (x is a
 *   spectator of BOTH the z and y transforms) still needs one transpose-class
 *   gather on the z load and one more entering phase 2 -- every arrangement
 *   of this pass structure pays exactly 2 granule transposes per element,
 *   which is what both L=45 entries already pay.  Not a lever; documented so
 *   nobody chases it.
 *
 *   PREFETCH LADDER (tournament-gated, all compile-time now):
 *     ip-pf0: nothing.
 *     ip-pf1: phase 2 pokes its 45 x-read-streams one line ahead (PF45).
 *     ip-pf2: + paced T1 read prefetch of phase 1's in-stream (PFIN) and
 *             per-tile pre-coverage of the next volume's input (PFNX).
 *     ip-pf3: + write-intent prefetchw of phase 1's cold mid-plane stores.
 *     ip-pf0a / ip-pf3a (codes 4/5): pf0 / pf3 plus the r11 zal aligned
 *             z-load path (not a prefetch; the codes slot into the same
 *             forcing knob).
 *     sp-pf0: nothing (S is cache-hot; out is only touched in phase 2).
 *     sp-pfs: PFIN on in + prefetchw poke of phase 2's 45 cold out-streams
 *             + PFNX (the streaming set for scratchp).
 *   fft3d_create() gates every candidate against a scalar O(n^2)-per-line
 *   reference at 1e-13 (volume 0 AND the last arena volume, so S-reuse bugs
 *   across a batch cannot hide), times them interleaved, and installs the
 *   fastest with a 3% simplest-first hysteresis.  FFT45_PW / FFT45_MODE /
 *   FFT45_PF force the choice at plan time for the monitor's control runs.
 *
 * ATTRIBUTION
 *   - Const-propagated exec-variant instantiation (pf compiled out of the
 *     pf0 path): L45_mixedradix (their body()/exec_v_c structure, all
 *     rounds).  Their leaner pf0 loop is why they won r7 on the node.
 *   - Padded-scratch odd-line pitches: L23_rader r6/r7 (the 1058->1064
 *     stride pad, -25..30% at B=1, and the "pad so every stride is an odd
 *     number of cache lines" corpus rule, LITERATURE 04/08).
 *   - Two-sweep plane-fused structure, spectator lanes, 6-op DFT3 / 2-op
 *     CMUL, TRNC transpose, PFIN/PFNX, prefetchw, tuner + hysteresis + env
 *     forcing: L36_pfa (r2-r5), transitively L36_mixedradix r1, L6_unrolled
 *     r3.  Flat phase-2 tiling: L45_mixedradix r7.  Masked z-column tail:
 *     L45_mixedradix r6 (as 128-bit inserts/extracts).  16-op DFT5: FFTW
 *     n1_5 via the corpus.
 *
 * OPERATION COUNT (PW=4, r10 tails)
 *   INPLACE: 45*(11+11) + 506 + 1 = 1497 zmm codelet calls x 344 = 514,968
 *   zmm FMA-port ops/volume (r9: 545,928; the PW=1 tails cut 5.7%) plus
 *   90 xmm lines x 344 ops that dual-issue on ports 0+1 (~15.5k
 *   port-0-equivalent cycles).  Port-0-equivalent floor ~ 530k cycles
 *   ~ 183 us at 2.9 GHz (r9 floor: 188 us; r9 node B=1: 315.9 us = 1.68x,
 *   port 0 measured NON-binding, so the tails' real value is the deleted
 *   movement of 90 full-width groups, not the op count).
 *   SCRATCHP: 45*22 + 540 = 1530 calls = 526,320 + the same 90 xmm lines.
 *
 * ACCURACY: ~4e-16 relative L2 vs numpy (unchanged module family).
 */
#ifndef _GNU_SOURCE
# define _GNU_SOURCE 1          /* sched_getcpu, CPU_SET, pthread affinity */
#endif
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>

#include "fft3d_api.h"

#ifndef L45_PFA_ONCE            /* ============ COMMON, first pass ============ */
#define L45_PFA_ONCE

/* r7 shipped '#pragma GCC optimize("unroll-loops")' here on the belief the
 * scored build lacked -funroll-loops.  r8's VERDICT 3c proved the premise
 * false (Makefile:15 has carried the flag for >= 3 rounds), and L17_rader
 * measured the pragma FORM as a ~2% tax in its own file (optimize() rebuilds
 * the entire per-function option set, not just the named flag).  Default is
 * now OFF; -DFFT45_UNROLL_PRAGMA restores it for a control build. */
#ifdef FFT45_UNROLL_PRAGMA
# pragma GCC optimize("unroll-loops")
#endif

#define L      45
#define NPLANE 2025              /* 45*45 complex per x-plane                  */
#define PLND   4050              /* doubles per x-plane = x-stream stride      */
#define VDBL   ((size_t)2 * L * NPLANE)  /* doubles per volume = 182250        */
/* plane-scratch row pitch in complex.  52 complex = 832 B = 13 cache lines:
 * coprime with 64 sets, so the y-pass column walk spreads over all L1 sets
 * (pitch 48 = 12 lines measured 15-20% slower in r6, both entries). */
#ifndef PPITCH
# define PPITCH 52
#endif
#define CAT_(a,b) a##b
#define CAT(a,b)  CAT_(a,b)

/* phase-1 input prefetch distance, doubles (32 KB) */
#ifndef FFT45_PFD
# define FFT45_PFD 4096
#endif
/* prefetch hint: 3=T0, 2=T1 (the measured winner at L=36/L=6), 0=NTA */
#ifndef FFT45_PFH
# define FFT45_PFH 2
#endif
/* cache lines of the NEXT volume's input prefetched per phase-2 tile */
#ifndef FFT45_PFN
# define FFT45_PFN 2
#endif
/* ip-pf3: write-intent cursor lead over the mid-plane stores, doubles */
#ifndef FFT45_PFWD
# define FFT45_PFWD 4050
#endif

/* sqrt(3)/2 (the DFT3 rotation, and the radix-3 spine of the 9-point DAG) */
#define KS3  0.86602540378443864676372317075294
/* genfft n1_9 (FMA form) DAG constants, fftw-3.3.10 n1_9.c */
#define K176 0.17632698070846497347109038686862  /* tan(pi/18)                */
#define K839 0.83909963117728001176312729812318
#define K777 0.77786191343020616002817797731863
#define K984 0.98480775301220805936674302458952
#define K492 0.49240387650610402968337151229476
#define K852 0.85286853195244320962825096394007
#define K363 0.36397023426620236135104788277683  /* tan(pi/9)                 */
#define K954 0.95418889413867113349926836418725
/* 5-point module constants (FFTW n1_5 form) */
#define K59  0.55901699437494742410229341718282   /* sqrt(5)/4                 */
#define KIG  0.61803398874989484820458683436564   /* sin(4pi/5)/sin(2pi/5)     */
#define KS5  0.95105651629515357211665325776975   /* sin(2pi/5)                */

/* ---- multicore layer (mt_r2): persistent pinned spin pool ----------------
 * BORROWED from L23_matrixsimd mt_r1 via L45_mixedradix mt_r1: one GOMP
 * parallel+barrier+join costs a measured 6-8 us at T=8..32, several times
 * the entire sync need of a job; the pool replaces it with one release-
 * store dispatch, padded per-thread arrival flags and one release word per
 * barrier group.  Workers never sleep (the driver's timing loop is
 * back-to-back and the 32 cores are ours); OpenMP is entered exactly once,
 * in create(), to read the harness's thread->CPU map. */
#ifdef _OPENMP
# include <omp.h>
#else  /* shims so a no-OpenMP build still compiles */
static inline int omp_get_thread_num(void)  { return 0; }
static inline int omp_get_max_threads(void) { return 1; }
#endif

/* the harness's thread budget; never take more (PANEL_BRIEF rule 2) */
#define MAXT 32

static inline void cpu_relax(void)
{
#if defined(__x86_64__) || defined(__i386__)
    _mm_pause();
#endif
}

/* one cache line per flag: a barrier arrival is one uncontended store */
typedef struct { _Atomic unsigned long v; char pad[56]; } mt_flag;

struct l45mt;
typedef void (*wfn45)(struct l45mt *, int);
typedef struct { struct l45mt *pl; int tid; int cpu; } mt_warg;

/* Pool + job + per-thread context in one struct.  Pt[t] is thread t's
 * plane scratch (L*PPITCH complex), Mt[t] its private mid VOLUME (VDBL
 * doubles, the NT variants' working set), Tb[t] its staged-NT tile buffer
 * (45 x TBK*PW complex); all three live in one page-aligned block
 * allocated and FIRST-TOUCHED by thread t itself AFTER pinning, so they
 * are NUMA-local and no two threads ever share a scratch line.  Job
 * fields are plain stores published by the gen release-store; epochs on
 * the arrival flags derive from (generation, in-job volume index) and
 * only ever increase, so join / global-barrier / pair-barrier can share
 * the flag array. */
struct l45mt {
    _Atomic long gen;
    char pad0[56];
    mt_flag arr[MAXT];          /* per-thread arrival flags               */
    mt_flag rel[MAXT / 2];      /* per-group release; fused uses rel[0]   */
    _Atomic int ready, fail, shutdown;
    char pad1[52];
    /* the job (plain stores BEFORE the gen release-store) */
    wfn45         work;
    const double *in;
    double       *out;
    long          nvol;
    int           tw;           /* participants of the current job        */
    unsigned long emult;        /* epoch stride per generation            */
    long          gen_local;    /* main's copy of gen                     */
    int           nthr;         /* pool size incl. main                   */
    double       *Pt[MAXT];
    double       *Mt[MAXT];
    double       *Tb[MAXT];
    void         *raw[MAXT];
    pthread_t     th[MAXT];
    mt_warg       wa[MAXT];
};

/* JOIN (end of every job): workers post their flag and go straight back to
 * the generation spin; main cannot dispatch again (or return to the
 * driver) until it has seen every flag.  Main scans ALL pool threads --
 * non-participants of a narrow-team job post the same epoch. */
static inline void mt_join(struct l45mt *pl, int tid)
{
    const unsigned long e =
        (unsigned long)atomic_load_explicit(&pl->gen, memory_order_relaxed)
            * pl->emult + (pl->emult - 1);
    if (tid == 0) {
        for (int t = 1; t < pl->nthr; ++t)
            while (atomic_load_explicit(&pl->arr[t].v,
                                        memory_order_acquire) < e)
                cpu_relax();
    } else
        atomic_store_explicit(&pl->arr[tid].v, e, memory_order_release);
}

/* global barrier over the job's T participants (fused mode's ONE mid
 * barrier): arrivals + leader scan + one release word */
static inline void mt_gbar(struct l45mt *pl, int tid, int T, unsigned long e)
{
    if (tid == 0) {
        for (int t = 1; t < T; ++t)
            while (atomic_load_explicit(&pl->arr[t].v,
                                        memory_order_acquire) < e)
                cpu_relax();
        atomic_store_explicit(&pl->rel[0].v, e, memory_order_release);
    } else {
        atomic_store_explicit(&pl->arr[tid].v, e, memory_order_release);
        while (atomic_load_explicit(&pl->rel[0].v,
                                    memory_order_acquire) < e)
            cpu_relax();
    }
}

/* pair barrier (grp2 mode): even member scans its partner's flag and
 * publishes the pair's release word; release/acquire chains through the
 * leader so both members' phase-1 stores are visible to both readers */
static inline void mt_pbar(struct l45mt *pl, int tid, int g, unsigned long e)
{
    if ((tid & 1) == 0) {
        while (atomic_load_explicit(&pl->arr[tid + 1].v,
                                    memory_order_acquire) < e)
            cpu_relax();
        atomic_store_explicit(&pl->rel[g].v, e, memory_order_release);
    } else {
        atomic_store_explicit(&pl->arr[tid].v, e, memory_order_release);
        while (atomic_load_explicit(&pl->rel[g].v,
                                    memory_order_acquire) < e)
            cpu_relax();
    }
}

/* per-thread scratch block: Mt (page-aligned) + Pt + Tb, one allocation,
 * memset (= first touch) by the OWNER thread */
static int mt_ctx_alloc(struct l45mt *pl, int t)
{
    const size_t pd = (size_t)L * PPITCH * 2;   /* plane scratch, doubles  */
    const size_t td = 45 * 128;                 /* tile buffer, doubles    */
    const size_t nd = VDBL + pd + td + 64;
    void *b = NULL;
    if (posix_memalign(&b, 4096, nd * sizeof(double)) != 0 || !b) return 1;
    memset(b, 0, nd * sizeof(double));
    pl->raw[t] = b;
    pl->Mt[t]  = (double *)b;
    pl->Pt[t]  = (double *)b + VDBL;
    pl->Tb[t]  = (double *)b + VDBL + pd;
    return 0;
}

static void *mt_worker(void *arg)
{
    mt_warg *wa = (mt_warg *)arg;
    struct l45mt *pl = wa->pl;
    const int tid = wa->tid;
    const int cpu = wa->cpu;
    if (cpu >= 0) {
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(cpu, &s);
        pthread_setaffinity_np(pthread_self(), sizeof s, &s);
    }
    if (mt_ctx_alloc(pl, tid)) atomic_store(&pl->fail, 1);
    atomic_fetch_add(&pl->ready, 1);
    long last = 0;
    for (;;) {
        while (atomic_load_explicit(&pl->gen, memory_order_acquire) == last)
            cpu_relax();
        ++last;
        if (atomic_load_explicit(&pl->shutdown, memory_order_relaxed)) break;
        pl->work(pl, tid);
    }
    return NULL;
}

/* serial exec signature (runs on the main thread, uses slot-0 scratch) */
typedef void (*exec45_fn)(const double *in, double *out, long nvol,
                          struct l45mt *pl);

/* The 45-point Good-Thomas 9x5 codelet over PW interleaved-complex lanes.
 * LD(n) must yield input element n as a vec rvalue; ST(k, v) consumes output
 * element k.  Both index maps fold to compile-time constants once the loops
 * unroll.  ALL LDs (stage 1) complete before any ST (stage 2), so LD/ST may
 * alias freely.
 *
 * r11 STAGE ORDER SWAP (the r10 VERDICT 4.1 spill mechanism, adopted from
 * L45_mixedradix's ST1G/ST2G shape): stage 1 is now 9 x DFT5 (FFTW n1_5 FMA
 * form, 16 FMA-port ops + 2 swaps, SHORT live ranges) writing T_[9*k2 + n1];
 * stage 2 is 5 x DFT9 (genfft's FMA n1_9 DAG, 40 ops + 12 swaps, LONG live
 * ranges) reading 9 CONTIGUOUS hot slots and handing every output to ST the
 * moment it exists -- no DFT9 output ever parks in a named register, and the
 * T_ reads can fold into FMA memory operands.  r9-r10 ran the stages the
 * other way (5 x DFT9 into a stride-5-scattered array, then 9 x DFT5): same
 * 344 ops, but the DAG's q1/q2/a0/i0/S* held across three output blocks ON
 * TOP of pending array writes cost +23 stack moves where the rival's shape
 * pays +3, and the r10 VERDICT prices that difference at ~3 points of B=1.
 *
 * DFT9 DAG key (transcribed r9 from fftw-3.3.10 n1_9.c, 24 add + 56 fma =
 * 80 scalar FMA-port ops; each scalar re/im line pair is one vector op, each
 * re/im crossing one SWAP with signs folded into VPAIR constants).  Per
 * radix-3 column {n, n+3, n+6}: sJ_ = column sum, SJ_ = full sum,
 * aJ_ = xJ - sJ/2, iJ_ = SWAP(x(J+3) - x(J+6)); p/q = aJ -+ 866*i*e (the two
 * rotated DFT3 outputs).  k={0,3,6} is a DFT3 on the sums; k={1,4,7} and
 * k={2,5,8} build the (1 +- c*i) spiral factors w from p/q (one SWAP+FMA
 * each), cross them (u, z), and fan out through the 984 / 492+-852 split.
 * Total: 9*16 + 5*40 = 344 FMA-port ops + 78 shuffles per call, unchanged
 * since r9.  Output map k = (10*k1 + 36*k2) mod 45, input n = (5*n1 + 9*n2)
 * mod 45, both folded at compile time. */
#define PFA45(LD, ST) do {                                                   \
    vec T_[45];                            /* T_[9*k2 + n1] */               \
    _Pragma("GCC unroll 9")                                                  \
    for (int n1_ = 0; n1_ < 9; ++n1_) {                                      \
        vec x0_ = LD((5 * n1_     ) % 45);                                   \
        vec x1_ = LD((5 * n1_ +  9) % 45);                                   \
        vec x2_ = LD((5 * n1_ + 18) % 45);                                   \
        vec x3_ = LD((5 * n1_ + 27) % 45);                                   \
        vec x4_ = LD((5 * n1_ + 36) % 45);                                   \
        vec t1_ = x1_ + x4_, t4_ = x1_ - x4_;                                \
        vec t2_ = x2_ + x3_, t7_ = x2_ - x3_;                                \
        vec te_ = t1_ + t2_, ta_ = t1_ - t2_;                                \
        T_[n1_] = x0_ + te_;                                                 \
        vec tm_ = VFNMA(te_, VSPLAT(0.25), x0_);                             \
        vec tp_ = VFMA (ta_, VSPLAT(K59), tm_);                              \
        vec tq_ = VFNMA(ta_, VSPLAT(K59), tm_);                              \
        vec tv_ = VFMA (t7_, VSPLAT(KIG), t4_);                              \
        vec tw_ = VFNMA(t4_, VSPLAT(KIG), t7_);                              \
        vec sv_ = SWAP(tv_), sw_ = SWAP(tw_);                                \
        T_[ 9 + n1_] = VFMA (sv_, VPAIR(KS5, -KS5), tp_);                    \
        T_[18 + n1_] = VFNMA(sw_, VPAIR(KS5, -KS5), tq_);                    \
        T_[27 + n1_] = VFMA (sw_, VPAIR(KS5, -KS5), tq_);                    \
        T_[36 + n1_] = VFNMA(sv_, VPAIR(KS5, -KS5), tp_);                    \
    }                                                                        \
    _Pragma("GCC unroll 5")                                                  \
    for (int k2_ = 0; k2_ < 5; ++k2_) {                                      \
        const vec *f_ = T_ + 9 * k2_;                                        \
        vec s0_ = f_[3] + f_[6], e0_ = f_[3] - f_[6];                        \
        vec S0_ = f_[0] + s0_,  a0_ = VFNMA(s0_, VSPLAT(0.5), f_[0]);        \
        vec i0_ = SWAP(e0_);                                                 \
        vec s1_ = f_[4] + f_[7], e1_ = f_[4] - f_[7];                        \
        vec S1_ = f_[1] + s1_,  a1_ = VFNMA(s1_, VSPLAT(0.5), f_[1]);        \
        vec i1_ = SWAP(e1_);                                                 \
        vec p1_ = VFMA (i1_, VPAIR(KS3, -KS3), a1_);                         \
        vec q1_ = VFNMA(i1_, VPAIR(KS3, -KS3), a1_);                         \
        vec s2_ = f_[5] + f_[8], e2_ = f_[5] - f_[8];                        \
        vec S2_ = f_[2] + s2_,  a2_ = VFNMA(s2_, VSPLAT(0.5), f_[2]);        \
        vec i2_ = SWAP(e2_);                                                 \
        vec p2_ = VFMA (i2_, VPAIR(KS3, -KS3), a2_);                         \
        vec q2_ = VFNMA(i2_, VPAIR(KS3, -KS3), a2_);                         \
        /* k1 = 0, 3, 6: DFT3 on the column sums */                          \
        vec sg_ = S1_ + S2_, d3_ = S2_ - S1_, id_ = SWAP(d3_);               \
        vec b0_ = VFNMA(sg_, VSPLAT(0.5), S0_);                              \
        ST((36 * k2_      ) % 45, S0_ + sg_);                                \
        ST((36 * k2_ + 30) % 45, VFNMA(id_, VPAIR(KS3, -KS3), b0_));         \
        ST((36 * k2_ + 60) % 45, VFMA (id_, VPAIR(KS3, -KS3), b0_));         \
        /* k1 = 1, 4, 7 */                                                   \
        {                                                                    \
        vec v1_ = VFMA (i0_, VPAIR(KS3, -KS3), a0_);                         \
        vec w2_ = VFMA (SWAP(p2_), VPAIR(-K176, K176), p2_);                 \
        vec w1_ = VFMA (SWAP(p1_), VPAIR(K839, -K839), p1_);                 \
        vec u1_ = VFMA (w1_, VPAIR(K777, -K777), SWAP(w2_));                 \
        vec z1_ = VFMA (SWAP(w1_), VPAIR(K777, -K777), w2_);                 \
        ST((36 * k2_ + 10) % 45, VFMA (u1_, VPAIR(K984, -K984), v1_));       \
        vec r1_ = VFNMA(u1_, VPAIR(K492, -K492), v1_);                       \
        ST((36 * k2_ + 40) % 45, VFMA (z1_, VSPLAT(K852), r1_));             \
        ST((36 * k2_ + 70) % 45, VFNMA(z1_, VSPLAT(K852), r1_));             \
        }                                                                    \
        /* k1 = 2, 5, 8 */                                                   \
        {                                                                    \
        vec v2_ = VFNMA(i0_, VPAIR(KS3, -KS3), a0_);                         \
        vec wA_ = VFMA (q1_, VPAIR(K176, -K176), SWAP(q1_));                 \
        vec wB_ = VFNMA(SWAP(q2_), VPAIR(K363, -K363), q2_);                 \
        vec uB_ = VFMA (wB_, VPAIR(-K954, K954), wA_);                       \
        vec zB_ = VFMA (SWAP(wB_), VPAIR(-K954, K954), SWAP(wA_));           \
        ST((36 * k2_ + 20) % 45, VFMA (uB_, VPAIR(K984, -K984), v2_));       \
        vec rB_ = VFNMA(uB_, VPAIR(K492, -K492), v2_);                       \
        ST((36 * k2_ + 50) % 45, VFNMA(zB_, VSPLAT(K852), rB_));             \
        ST((36 * k2_ + 80) % 45, VFMA (zB_, VSPLAT(K852), rB_));             \
        }                                                                    \
    }                                                                        \
} while (0)

/* Register-array variant of the same codelet: the r9-r10 stage order
 * (5 x DFT9 into A_[5*k1 + n2], then 9 x DFT5).  Kept ONLY for the z-site,
 * where ST writes a vec array (Wv) that is transposed afterwards, not
 * memory: there the store-direct order above measures WORSE (per-function
 * audit under node flags, phase1_plane_pw4 spills 83 -> 91) because the
 * DFT9's nine scattered array writes pile onto its long live ranges, while
 * at the memory-ST sites the same order deletes spills (phase2_pw4
 * 53/55 -> 46/51).  Same DAG, same 344 ops + 78 swaps, same maps. */
#define DFT9F(x0,x1,x2,x3,x4,x5,x6,x7,x8, o0,o1,o2,o3,o4,o5,o6,o7,o8) do {   \
    vec s0_ = (x3) + (x6), e0_ = (x3) - (x6);                                \
    vec S0_ = (x0) + s0_,  a0_ = VFNMA(s0_, VSPLAT(0.5), (x0));              \
    vec i0_ = SWAP(e0_);                                                     \
    vec s1_ = (x4) + (x7), e1_ = (x4) - (x7);                                \
    vec S1_ = (x1) + s1_,  a1_ = VFNMA(s1_, VSPLAT(0.5), (x1));              \
    vec i1_ = SWAP(e1_);                                                     \
    vec p1_ = VFMA (i1_, VPAIR(KS3, -KS3), a1_);                             \
    vec q1_ = VFNMA(i1_, VPAIR(KS3, -KS3), a1_);                             \
    vec s2_ = (x5) + (x8), e2_ = (x5) - (x8);                                \
    vec S2_ = (x2) + s2_,  a2_ = VFNMA(s2_, VSPLAT(0.5), (x2));              \
    vec i2_ = SWAP(e2_);                                                     \
    vec p2_ = VFMA (i2_, VPAIR(KS3, -KS3), a2_);                             \
    vec q2_ = VFNMA(i2_, VPAIR(KS3, -KS3), a2_);                             \
    vec sg_ = S1_ + S2_, d3_ = S2_ - S1_, id_ = SWAP(d3_);                   \
    vec b0_ = VFNMA(sg_, VSPLAT(0.5), S0_);                                  \
    (o0) = S0_ + sg_;                                                        \
    (o3) = VFNMA(id_, VPAIR(KS3, -KS3), b0_);                                \
    (o6) = VFMA (id_, VPAIR(KS3, -KS3), b0_);                                \
    {                                                                        \
    vec v1_ = VFMA (i0_, VPAIR(KS3, -KS3), a0_);                             \
    vec w2_ = VFMA (SWAP(p2_), VPAIR(-K176, K176), p2_);                     \
    vec w1_ = VFMA (SWAP(p1_), VPAIR(K839, -K839), p1_);                     \
    vec u1_ = VFMA (w1_, VPAIR(K777, -K777), SWAP(w2_));                     \
    vec z1_ = VFMA (SWAP(w1_), VPAIR(K777, -K777), w2_);                     \
    (o1) = VFMA (u1_, VPAIR(K984, -K984), v1_);                              \
    vec r1_ = VFNMA(u1_, VPAIR(K492, -K492), v1_);                           \
    (o4) = VFMA (z1_, VSPLAT(K852), r1_);                                    \
    (o7) = VFNMA(z1_, VSPLAT(K852), r1_);                                    \
    }                                                                        \
    {                                                                        \
    vec v2_ = VFNMA(i0_, VPAIR(KS3, -KS3), a0_);                             \
    vec wA_ = VFMA (q1_, VPAIR(K176, -K176), SWAP(q1_));                     \
    vec wB_ = VFNMA(SWAP(q2_), VPAIR(K363, -K363), q2_);                     \
    vec uB_ = VFMA (wB_, VPAIR(-K954, K954), wA_);                           \
    vec zB_ = VFMA (SWAP(wB_), VPAIR(-K954, K954), SWAP(wA_));               \
    (o2) = VFMA (uB_, VPAIR(K984, -K984), v2_);                              \
    vec rB_ = VFNMA(uB_, VPAIR(K492, -K492), v2_);                           \
    (o5) = VFNMA(zB_, VSPLAT(K852), rB_);                                    \
    (o8) = VFMA (zB_, VSPLAT(K852), rB_);                                    \
    }                                                                        \
} while (0)

#define PFA45R(LD, ST) do {                                                  \
    vec A_[45];                            /* A_[5*k1 + n2] */               \
    _Pragma("GCC unroll 5")                                                  \
    for (int n2_ = 0; n2_ < 5; ++n2_) {                                      \
        vec g_[9];                                                           \
        _Pragma("GCC unroll 9")                                              \
        for (int n1_ = 0; n1_ < 9; ++n1_)                                    \
            g_[n1_] = LD((5 * n1_ + 9 * n2_) % 45);                          \
        DFT9F(g_[0], g_[1], g_[2], g_[3], g_[4], g_[5], g_[6], g_[7], g_[8],\
              A_[      n2_], A_[ 5 + n2_], A_[10 + n2_],                     \
              A_[15 + n2_], A_[20 + n2_], A_[25 + n2_],                      \
              A_[30 + n2_], A_[35 + n2_], A_[40 + n2_]);                     \
    }                                                                        \
    _Pragma("GCC unroll 9")                                                  \
    for (int k1_ = 0; k1_ < 9; ++k1_) {                                      \
        const vec *f_ = A_ + 5 * k1_;                                        \
        vec t1_ = f_[1] + f_[4], t4_ = f_[1] - f_[4];                        \
        vec t2_ = f_[2] + f_[3], t7_ = f_[2] - f_[3];                        \
        vec te_ = t1_ + t2_,     ta_ = t1_ - t2_;                            \
        ST((10 * k1_      ) % 45, f_[0] + te_);                              \
        vec tm_ = VFNMA(te_, VSPLAT(0.25), f_[0]);                           \
        vec tp_ = VFMA (ta_, VSPLAT(K59), tm_);                              \
        vec tq_ = VFNMA(ta_, VSPLAT(K59), tm_);                              \
        vec tv_ = VFMA (t7_, VSPLAT(KIG), t4_);                              \
        vec tw_ = VFNMA(t4_, VSPLAT(KIG), t7_);                              \
        vec sv_ = SWAP(tv_), sw_ = SWAP(tw_);                                \
        ST((10 * k1_ +  36) % 45, VFMA (sv_, VPAIR(KS5, -KS5), tp_));        \
        ST((10 * k1_ +  72) % 45, VFNMA(sw_, VPAIR(KS5, -KS5), tq_));        \
        ST((10 * k1_ + 108) % 45, VFMA (sw_, VPAIR(KS5, -KS5), tq_));        \
        ST((10 * k1_ + 144) % 45, VFNMA(sv_, VPAIR(KS5, -KS5), tp_));        \
    }                                                                        \
} while (0)

/* PW=1 instantiation of the same codelet: ONE complex per 128-bit vector,
 * for the odd 45th row/column of phase 1's subpasses (45 = 11*4 + 1).
 * VEX-coded xmm FMAs dual-issue on ports 0 AND 1 on the node (CLX runs two
 * 256-bit-or-narrower FMA pipes; only 512-bit is single-ported), and every
 * access is a 16 B complex that never splits a cache line.  Strides are in
 * DOUBLES.  Called ~90x per volume; gcc may inline it into the exec
 * variants or not -- either is fine (front end measured non-binding, r9). */
typedef double    v1d __attribute__((vector_size(16), aligned(16)));
typedef long long v1i __attribute__((vector_size(16)));

static void dft45_line1(const double *restrict s, const long sstr,
                        double *restrict d, const long dstr)
{
#define vec        v1d
#define VSPLAT(a)  ((vec){(a),(a)})
#define VPAIR(a,b) ((vec){(a),(b)})
#ifdef __clang__
# define SWAP(v)   __builtin_shufflevector((v),(v), 1,0)
#else
# define SWAP(v)   __builtin_shuffle((v),(v),(v1i){1,0})
#endif
#ifdef __FMA__
# define VFMA(a,b,c)  ((vec)_mm_fmadd_pd((__m128d)(a),(__m128d)(b),(__m128d)(c)))
# define VFNMA(a,b,c) ((vec)_mm_fnmadd_pd((__m128d)(a),(__m128d)(b),(__m128d)(c)))
#else
# define VFMA(a,b,c)  ((a)*(b) + (c))
# define VFNMA(a,b,c) ((c) - (a)*(b))
#endif
#define LDL1(n)    (*(const vec *)(s + (size_t)(n) * sstr))
#define STL1(k, v) (*(vec *)(d + (size_t)(k) * dstr) = (v))
    PFA45(LDL1, STL1);
#undef LDL1
#undef STL1
#undef VFMA
#undef VFNMA
#undef SWAP
#undef VPAIR
#undef VSPLAT
#undef vec
}

/* instantiate the kernel template: PW=2, and PW=4 on AVX-512 */
#define PW 2
#include __FILE__
#undef PW
#ifdef __AVX512F__
# define PW 4
# include __FILE__
# undef PW
# define HAVE_PW4 1
#endif

/* ---- plan, tuner, API ---------------------------------------------------- */

enum { MT_V = 0, MT_N = 1, MT_G = 2, MT_F = 3, MT_S = 4 };
/* vol-parallel, vol-parallel+NT (mtn=copy / mts=staged tiles), pair-grained
 * grp2 (in-place / NT-staged), fused two-sweep, serial control */

/* candidate table, rank = simplest-first tie-break order (3% hysteresis).
 * minb/maxb gate a row to the batch range where it can matter (a grp2 or
 * mtv row at B=1 is serial-with-idle-threads; the T=23 fused probe only
 * asks a B=1-shaped question). */
struct cand45 {
    wfn45       wfn;             /* pool job, or NULL for a serial row     */
    exec45_fn   sfn;
    int         pw, mt, pf, tw, rank, minb, maxb;
    const char *nm;
};
static const struct cand45 g_cands[] = {
#ifdef HAVE_PW4
    { w_v0_pw4,  0, 4, MT_V, 0, MAXT, 0,  2, 0, "pw4-mtv-pf0" },
    { w_v3_pw4,  0, 4, MT_V, 3, MAXT, 1,  2, 0, "pw4-mtv-pf3" },
    { w_n0_pw4,  0, 4, MT_N, 6, MAXT, 2,  2, 0, "pw4-mtn-pf0" },
    { w_ni_pw4,  0, 4, MT_N, 7, MAXT, 3,  2, 0, "pw4-mtn-pfi" },
    { w_s0_pw4,  0, 4, MT_N, 8, MAXT, 4,  2, 0, "pw4-mts-pf0" },
    { w_si_pw4,  0, 4, MT_N, 9, MAXT, 5,  2, 0, "pw4-mts-pfi" },
    { w_g0_pw4,  0, 4, MT_G, 0, MAXT, 6,  2, 0, "pw4-g2-pf0"  },
    { w_gi_pw4,  0, 4, MT_G, 3, MAXT, 7,  2, 0, "pw4-g2-pfw"  },
    { w_gn0_pw4, 0, 4, MT_G, 8, MAXT, 8,  2, 0, "pw4-g2n-pf0" },
    { w_gni_pw4, 0, 4, MT_G, 9, MAXT, 9,  2, 0, "pw4-g2n-pfi" },
    { w_f0_pw4,  0, 4, MT_F, 0, MAXT, 10, 1, 0, "pw4-mtf-rr"  },
    { w_fb_pw4,  0, 4, MT_F, 1, MAXT, 11, 1, 0, "pw4-mtf-blk" },
    { w_fb3_pw4, 0, 4, MT_F, 2, MAXT, 12, 1, 0, "pw4-mtf-bpf" },
    { w_fb_pw4,  0, 4, MT_F, 5, 23,   13, 1, 2, "pw4-mtf-t23" },
    { 0, x_ip0_pw4, 4, MT_S, 0, 1,    14, 1, 0, "pw4-ip-pf0"  },
    { 0, x_ip3_pw4, 4, MT_S, 3, 1,    15, 1, 0, "pw4-ip-pf3"  },
#endif
    { w_v0_pw2,  0, 2, MT_V, 0, MAXT, 16, 2, 0, "pw2-mtv-pf0" },
    { w_fb_pw2,  0, 2, MT_F, 1, MAXT, 17, 1, 0, "pw2-mtf-blk" },
    { 0, x_ip0_pw2, 2, MT_S, 0, 1,    18, 1, 0, "pw2-ip-pf0"  },
};
#define NCAND ((int)(sizeof g_cands / sizeof g_cands[0]))

struct fft3d_plan {
    long          batch;
    int           pick;
    struct l45mt *pl;
};

/* one pool dispatch (or a serial call): publish the job, release-store the
 * generation, participate as tid 0, return when every worker has joined */
static void run_cand(struct l45mt *pl, const struct cand45 *c,
                     const double *in, double *out, long nvol)
{
    if (c->sfn) { c->sfn(in, out, nvol, pl); return; }
    pl->in   = in;
    pl->out  = out;
    pl->nvol = nvol;
    pl->tw   = c->tw < pl->nthr ? c->tw : pl->nthr;
    pl->work = c->wfn;
    ++pl->gen_local;
    atomic_store_explicit(&pl->gen, pl->gen_local, memory_order_release);
    c->wfn(pl, 0);
}

const char *fft3d_name(void) { return "L45_pfa"; }

static char g_desc[512];

const char *fft3d_description(void)
{
    return g_desc[0] ? g_desc
                     : "Good-Thomas PFA 9x5, interleaved-complex lanes, two "
                       "sweeps, PW=1 xmm tail lines; pinned spin pool; "
                       "{vol, vol-NT, pair, pair-NT, fused, serial} x pf "
                       "autotuned in create()";
}
int fft3d_supports(int Lq) { return Lq == L; }

/* scalar O(L^2)-per-line reference: independent ground truth for the gate */
static void ref3d(const double _Complex *in, double _Complex *out)
{
    double _Complex Wt[L], buf[L];
    for (int k = 0; k < L; ++k)
        Wt[k] = cexp(-2.0 * M_PI * I * (double)k / (double)L);
    for (int x = 0; x < L; ++x)                       /* z axis: in -> out */
        for (int y = 0; y < L; ++y) {
            const double _Complex *r = in  + ((size_t)x * L + y) * L;
            double _Complex       *w = out + ((size_t)x * L + y) * L;
            for (int k = 0; k < L; ++k) {
                double _Complex s = 0;
                for (int j = 0; j < L; ++j) s += r[j] * Wt[(j * k) % L];
                w[k] = s;
            }
        }
    for (int x = 0; x < L; ++x)                       /* y axis, in place  */
        for (int z = 0; z < L; ++z) {
            double _Complex *base = out + (size_t)x * NPLANE + z;
            for (int j = 0; j < L; ++j) buf[j] = base[(size_t)j * L];
            for (int k = 0; k < L; ++k) {
                double _Complex s = 0;
                for (int j = 0; j < L; ++j) s += buf[j] * Wt[(j * k) % L];
                base[(size_t)k * L] = s;
            }
        }
    for (int y = 0; y < L; ++y)                       /* x axis, in place  */
        for (int z = 0; z < L; ++z) {
            double _Complex *base = out + (size_t)y * L + z;
            for (int j = 0; j < L; ++j) buf[j] = base[(size_t)j * NPLANE];
            for (int k = 0; k < L; ++k) {
                double _Complex s = 0;
                for (int j = 0; j < L; ++j) s += buf[j] * Wt[(j * k) % L];
                base[(size_t)k * NPLANE] = s;
            }
        }
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int rel_ok(const double *got, const double *ref, size_t n)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = got[i] - ref[i];
        num += d * d; den += ref[i] * ref[i];
    }
    return num <= den * 1e-26;                        /* rel L2 < 1e-13 */
}

fft3d_plan *fft3d_create(int Lq, int batch)
{
    if (Lq != L || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->batch = batch;

    int maxt = omp_get_max_threads();
    if (maxt > MAXT) maxt = MAXT;    /* raw shells can report 64/128: cap  */
    if (maxt < 1)    maxt = 1;

    /* Read the harness's OMP thread->CPU mapping from ONE throwaway OpenMP
     * region, so the pool pins exactly where OMP_PROC_BIND=close /
     * OMP_PLACES=cores would put each thread (this also leaves the initial
     * thread bound to place 0).  Unbound-run guard: if any two threads
     * report the same CPU, or PROC_BIND is unset, do not pin at all.
     * (Borrowed from L23_matrixsimd mt_r1 via L45_mixedradix.)  GOMP's own
     * workers sleep after their spin timeout and are never used again. */
    int cpus[MAXT];
    for (int t = 0; t < MAXT; ++t) cpus[t] = -1;
#ifdef _OPENMP
#pragma omp parallel num_threads(maxt)
    {
        int t = omp_get_thread_num();
        if (t < MAXT) cpus[t] = sched_getcpu();
    }
    {
        int pin = getenv("OMP_PROC_BIND") != NULL;
        for (int a = 0; pin && a < maxt; ++a)
            for (int b = a + 1; b < maxt; ++b)
                if (cpus[a] == cpus[b]) pin = 0;
        if (!pin)
            for (int t = 0; t < MAXT; ++t) cpus[t] = -1;
    }
#endif

    /* ---- the persistent pool (thread creation is setup, PANEL_BRIEF) --- */
    {
        void *pb = NULL;
        if (posix_memalign(&pb, 64, sizeof(struct l45mt)) != 0 || !pb) {
            free(p); return NULL;
        }
        memset(pb, 0, sizeof(struct l45mt));
        p->pl = (struct l45mt *)pb;
    }
    struct l45mt *pl = p->pl;
    pl->nthr  = maxt;
    pl->emult = (unsigned long)batch + 8;
    if (mt_ctx_alloc(pl, 0)) { fft3d_destroy(p); return NULL; }
    for (int t = 1; t < maxt; ++t) {
        pl->wa[t].pl  = pl;
        pl->wa[t].tid = t;
        pl->wa[t].cpu = cpus[t];
        if (pthread_create(&pl->th[t], NULL, mt_worker, &pl->wa[t]) != 0) {
            pl->nthr = t;               /* join only what was spawned */
            break;
        }
    }
    while (atomic_load(&pl->ready) < pl->nthr - 1) cpu_relax();
    if (atomic_load(&pl->fail)) { fft3d_destroy(p); return NULL; }

    /* safe default: the serial control row of the best compiled width */
    p->pick = 0;
    for (int c = 0; c < NCAND; ++c)
        if (g_cands[c].sfn) { p->pick = c; break; }

    int    live[NCAND];
    double tc[NCAND];
    for (int c = 0; c < NCAND; ++c) { live[c] = 1; tc[c] = 1e300; }

    /* batch-range and pool-availability gates */
    for (int c = 0; c < NCAND; ++c) {
        if (batch < g_cands[c].minb) live[c] = 0;
        if (g_cands[c].maxb && batch > g_cands[c].maxb) live[c] = 0;
        if (!g_cands[c].sfn && pl->nthr < 2) live[c] = 0;
        if (g_cands[c].mt == MT_G && pl->nthr < 4) live[c] = 0;
    }

    /* run-time forcing for the monitor's control jobs */
    { const char *e;
      if ((e = getenv("FFT45_PW"))) {
          int v = atoi(e);
          for (int c = 0; c < NCAND; ++c) if (g_cands[c].pw != v) live[c] = 0;
      }
      if ((e = getenv("FFT45_MT"))) { /* v / n(t+mts) / g / f / s */
          int v = (e[0] >= '0' && e[0] <= '9') ? atoi(e)
                : (e[0] == 'n' ? MT_N : (e[0] == 'g' ? MT_G
                :  (e[0] == 'f' ? MT_F : (e[0] == 's' ? MT_S : MT_V))));
          for (int c = 0; c < NCAND; ++c) if (g_cands[c].mt != v) live[c] = 0;
      }
      if ((e = getenv("FFT45_PF"))) {
          int v = atoi(e);
          for (int c = 0; c < NCAND; ++c) if (g_cands[c].pf != v) live[c] = 0;
      }
      { int any = 0;
        for (int c = 0; c < NCAND; ++c) any |= live[c];
        if (!any) for (int c = 0; c < NCAND; ++c) live[c] = 1; } }

    /* tuning arena: must actually stream at large batch (L36_pfa r2 lesson).
     * mt_r1 raises the cap from 32 to 128 volumes (2 x 178 MB in+out): at
     * 32 threads, 32 volumes is ONE volume per thread and the whole walk
     * sat inside wallaby's 60 MB L3 -- it mispriced the streaming NT/pf
     * class for B=256 (nv=32 said 9.6 us/vol, the driver said 25.7). */
    const int nv = batch < 128 ? batch : 128;
    void *ri = NULL, *ro = NULL, *r0 = NULL, *r1 = NULL;
    if (posix_memalign(&ri, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&ro, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&r0, 64, VDBL * sizeof(double)) ||
        (nv > 1 && posix_memalign(&r1, 64, VDBL * sizeof(double)))) {
        free(ri); free(ro); free(r0);
        snprintf(g_desc, sizeof g_desc,
                 "GT-PFA 9x5 pool; tuner SKIPPED (arena alloc failed): %s",
                 g_cands[p->pick].nm);
        return p;
    }
    double *tin = ri, *tout = ro, *ref0 = r0, *refN = r1;
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        tin[i] = (double)(long long)(s >> 11) * 0x1p-53;
    }
    ref3d((const double _Complex *)tin, (double _Complex *)ref0);
    if (nv > 1)          /* gate the LAST volume too: catches M-reuse bugs */
        ref3d((const double _Complex *)(tin + (size_t)(nv - 1) * VDBL),
              (double _Complex *)refN);

    for (int c = 0; c < NCAND; ++c) {
        if (!live[c]) continue;
        memset(tout, 0, (size_t)nv * VDBL * sizeof(double));
        run_cand(pl, &g_cands[c], tin, tout, nv);
        if (!rel_ok(tout, ref0, VDBL)) live[c] = 0;
        if (nv > 1 && live[c] &&
            !rel_ok(tout + (size_t)(nv - 1) * VDBL, refN, VDBL)) live[c] = 0;
    }
    /* small arenas get more interleaved rounds: wallaby toggles a fast/slow
     * machine state on a seconds scale, and a toggle edge crossing the
     * tournament mis-ranks kernels (observed here in r8: a slow-window race
     * installed sp-pfs at B=1, 205 vs 175 us).  Min over more rounds gives
     * every candidate a shot at the fast state.  (L36/L45_mixedradix's
     * tuner-hardening lesson.) */
    const int R = (nv >= 8) ? 1 : (nv >= 2 ? 3 : 8);
    const int NR = (nv >= 16) ? 4 : (nv >= 4 ? 6 : 10);
    for (int round = 0; round < NR; ++round)
        for (int c = 0; c < NCAND; ++c) {
            if (!live[c]) continue;
            /* self-warm so each candidate is timed from its own steady state */
            run_cand(pl, &g_cands[c], tin, tout, nv);
            double t0 = now_s();
            for (int r = 0; r < R; ++r)
                run_cand(pl, &g_cands[c], tin, tout, nv);
            double t = (now_s() - t0) / R;
            if (t < tc[c]) tc[c] = t;
        }
    int best = -1;
    for (int c = 0; c < NCAND; ++c)
        if (live[c] && (best < 0 || tc[c] < tc[best])) best = c;
    int pick = best < 0 ? p->pick : best;
    if (best >= 0)
        for (int c = 0; c < NCAND; ++c)
            if (live[c] && tc[c] <= tc[best] * 1.03 &&
                g_cands[c].rank < g_cands[pick].rank) pick = c;
    p->pick = pick;
    {   /* publish the pick AND the in-arena table (the monitor reads it) */
        char *q = g_desc, *e = g_desc + sizeof g_desc;
        q += snprintf(q, (size_t)(e - q),
                      "GT-PFA 9x5 pool T=%d; pick=%s (B=%d nv=%d)",
                      pl->nthr, g_cands[pick].nm, batch, nv);
        for (int c = 0; c < NCAND && q < e; ++c)
            if (live[c] && tc[c] < 1e299)
                q += snprintf(q, (size_t)(e - q), " %s=%.1f",
                              g_cands[c].nm + 4, tc[c] * 1e6 / nv);
    }

#ifdef FFT45_LOUD
    if (1) {
#else
    if (getenv("FFT45_VERBOSE")) {
#endif
        for (int c = 0; c < NCAND; ++c)
            fprintf(stderr, "L45_pfa tuner: %-12s  %s  %.1f us/vol\n",
                    g_cands[c].nm, live[c] ? "ok " : "OUT",
                    live[c] && tc[c] < 1e299 ? tc[c] * 1e6 / nv : 0.0);
        fprintf(stderr, "L45_pfa tuner: chose %s (nv=%d)\n",
                g_cands[pick].nm, nv);
    }
    free(ri); free(ro); free(r0); free(r1);
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    struct l45mt *pl = p->pl;
    if (pl) {
        if (pl->nthr > 1) {
            atomic_store(&pl->shutdown, 1);
            ++pl->gen_local;
            atomic_store_explicit(&pl->gen, pl->gen_local,
                                  memory_order_release);
            for (int t = 1; t < pl->nthr; ++t)
                pthread_join(pl->th[t], NULL);
        }
        for (int t = 0; t < MAXT; ++t) free(pl->raw[t]);
        free(pl);
    }
    free(p);
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    run_cand(plan->pl, &g_cands[plan->pick],
             (const double *)in, (double *)out, plan->batch);
}

#else /* ================ KERNEL TEMPLATE, PW = 2 or 4 ====================== */

#define vec   CAT(vec_pw,  PW)
#define uvec  CAT(uvec_pw, PW)
#define veci  CAT(veci_pw, PW)
#define FN(n) CAT(n, CAT(_pw, PW))

typedef double    vec  __attribute__((vector_size(PW * 16)));
typedef double    uvec __attribute__((vector_size(PW * 16), aligned(8)));
typedef long long veci __attribute__((vector_size(PW * 16)));

#ifdef __clang__
# define VSH(a,b,...) __builtin_shufflevector(a, b, __VA_ARGS__)
#else
# define VSH(a,b,...) __builtin_shuffle(a, b, (veci){__VA_ARGS__})
#endif

/* in/out accesses are potentially 16-byte aligned only (720 B rows); the
 * plane scratch and padded S are always 64B-aligned but share the macro
 * (an aligned address through an unaligned op costs nothing on these cores) */
#define LDU(p)    ((vec)*(const uvec *)(p))
#define STU(p, v) (*(uvec *)(p) = (uvec)(v))

#define NGRP  ((L + PW - 1) / PW)     /* lane groups incl. overlap tail: 23/12 */
#define NFULL (L / PW)                /* full groups: 22/11                    */
/* r10: phase 1's subpass group count.  Default = NFULL full groups + one
 * PW=1 tail line (dft45_line1); -DFFT45_OVERLAP_TAIL restores the r7-r9
 * clamped overlap group (NGRP groups, last one recomputing PW-1 lanes). */
#ifdef FFT45_OVERLAP_TAIL
# define NG1 NGRP
#else
# define NG1 NFULL
#endif

#if PW == 4
# define VSPLAT(a)  ((vec){(a),(a),(a),(a),(a),(a),(a),(a)})
# define VPAIR(a,b) ((vec){(a),(b),(a),(b),(a),(b),(a),(b)})
# define SWAP(v)    VSH((v),(v), 1,0,3,2,5,4,7,6)
# define VFMA(a,b,c)  ((vec)_mm512_fmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
# define VFNMA(a,b,c) ((vec)_mm512_fnmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
/* one-complex masked load/store (lane 0), dead lanes ZERO (no denormal risk) */
# define LDT(p)    ((vec)_mm512_maskz_loadu_pd((__mmask8)0x03, (p)))
# define STT(p, v) _mm512_mask_storeu_pd((p), (__mmask8)0x03, (__m512d)(v))
/* gather/scatter one complex COLUMN (PW rows at `st` doubles apart) into/from
 * the lanes of one vector: the z-pass odd-column tail.  4 x 16 B accesses:
 * complex elements are 16B-aligned, never split a line. */
# define GCOL(dst, p, st) do {                                               \
    __m256d lo_ = _mm256_insertf128_pd(                                      \
        _mm256_castpd128_pd256(_mm_loadu_pd((p))),                           \
        _mm_loadu_pd((p) + (st)), 1);                                        \
    __m256d hi_ = _mm256_insertf128_pd(                                      \
        _mm256_castpd128_pd256(_mm_loadu_pd((p) + 2 * (st))),                \
        _mm_loadu_pd((p) + 3 * (st)), 1);                                    \
    (dst) = (vec)_mm512_insertf64x4(_mm512_castpd256_pd512(lo_), hi_, 1);    \
} while (0)
# define SCOL(src, p, st) do {                                               \
    __m512d v_ = (__m512d)(src);                                             \
    __m256d h_ = _mm512_extractf64x4_pd(v_, 1);                              \
    _mm_storeu_pd((p),            _mm512_castpd512_pd128(v_));               \
    _mm_storeu_pd((p) +     (st), _mm256_extractf128_pd(                     \
                                      _mm512_castpd512_pd256(v_), 1));       \
    _mm_storeu_pd((p) + 2 * (st), _mm256_castpd256_pd128(h_));               \
    _mm_storeu_pd((p) + 3 * (st), _mm256_extractf128_pd(h_, 1));             \
} while (0)
#elif PW == 2
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
#  define LDT(p)    ((vec)_mm256_maskload_pd((p),                            \
                         (__m256i){-1LL, -1LL, 0LL, 0LL}))
#  define STT(p, v) _mm256_maskstore_pd((p),                                 \
                         (__m256i){-1LL, -1LL, 0LL, 0LL}, (__m256d)(v))
#  define GCOL(dst, p, st)                                                   \
    ((dst) = (vec)_mm256_insertf128_pd(                                      \
        _mm256_castpd128_pd256(_mm_loadu_pd((p))),                           \
        _mm_loadu_pd((p) + (st)), 1))
#  define SCOL(src, p, st) do {                                              \
    _mm_storeu_pd((p),        _mm256_castpd256_pd128((__m256d)(src)));       \
    _mm_storeu_pd((p) + (st), _mm256_extractf128_pd((__m256d)(src), 1));     \
} while (0)
# else  /* portable fallbacks so a pre-AVX build still compiles */
#  define LDT(p)    ((vec){(p)[0], (p)[1], 0.0, 0.0})
#  define STT(p, v) do { (p)[0] = (v)[0]; (p)[1] = (v)[1]; } while (0)
#  define GCOL(dst, p, st)                                                   \
    ((dst) = (vec){(p)[0], (p)[1], (p)[(st)], (p)[(st) + 1]})
#  define SCOL(src, p, st) do {                                              \
    (p)[0] = (src)[0]; (p)[1] = (src)[1];                                    \
    (p)[(st)] = (src)[2]; (p)[(st) + 1] = (src)[3];                          \
} while (0)
# endif
#endif

/* (mt_r2: the r11 "zal" aligned-z-load recombination is DELETED.  It was
 * shipped tuner-gated so the node's tournament could price the split-load
 * class directly; the mt_r1 node took ZERO zal picks at any cell -- CLX
 * hides the 16B-phase splits well enough under 32-thread contention too.
 * The mechanism and its phase arithmetic live in the r11 exemplar and in
 * ../geom/impl_11/L45_pfa.c if a future round wants it back.) */

/* PW x PW transpose of 128-bit complex granules (involution) */
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
#elif PW == 2
# define TRNC(r, c) do {                                                     \
    vec u0_ = VSH((r)[0], (r)[1], 0,1,4,5);                                  \
    vec u1_ = VSH((r)[0], (r)[1], 2,3,6,7);                                  \
    (c)[0] = u0_; (c)[1] = u1_;                                              \
} while (0)
#endif

/* paced-prefetch step: cover one x-plane (PLND doubles = 507 lines) in
 * 2*NG1 steps -> 24 lines/step at PW=4, 13 at PW=2 (r10: NG1 groups/pass) */
#define PFL    ((PLND + 16 * NG1 - 1) / (16 * NG1))
#define PFSTEP (8 * PFL)
#define PFIN(p) do {                                                          \
    _Pragma("GCC unroll 24")                                                  \
    for (int q_ = 0; q_ < PFL; ++q_)                                          \
        __builtin_prefetch((p) + 8 * q_, 0, FFT45_PFH);                       \
} while (0)
/* ip-pf3: write-intent (prefetchw) cursor over the mid-plane stores */
#define PFWMID(p) do {                                                        \
    _Pragma("GCC unroll 24")                                                  \
    for (int q_ = 0; q_ < PFL; ++q_)                                          \
        __builtin_prefetch((p) + 8 * q_, 1, 3);                               \
} while (0)
/* phase 1, ONE x-plane: z transform (transposed lanes) into plane scratch,
 * then y transform into mid[x][ky][kz].  pfr/pfw are compile-time constants
 * at every instantiation site (always_inline + const), so the pf0 execs
 * carry NO prefetch code and no cursor arithmetic.  mrow/mpln are the mid
 * row/plane strides in doubles: (90, 4050) inplace, (104, 4680) scratchp --
 * also compile-time, so every mid access folds to base + literal. */
static inline __attribute__((always_inline))
void FN(phase1_plane)(const double *restrict in, double *restrict mid,
                      double *restrict pld, int x, const int pfr,
                      const int pfw, const long mrow, const long mpln)
{
    const double *pfc = pfr ? in  + FFT45_PFD  + (size_t)x * PLND : 0;
    double       *pwc = pfw ? mid + FFT45_PFWD + (size_t)x * mpln : 0;
    const double *px  = in  + (size_t)x * PLND;
    double       *mx  = mid + (size_t)x * mpln;

    for (int yg = 0; yg < NG1; ++yg) {
        const int yb = (yg == NGRP - 1) ? (L - PW) : yg * PW;
        /* ONE runtime base per block; every access below is base + a
         * compile-time constant (the r7 single-base fix: folding yb into
         * each address made gcc spill a 48-entry offset table, -14%). */
        const double *rows = px  + (size_t)yb * (2 * L);
        double       *prow = pld + (size_t)yb * (2 * PPITCH);
        if (pfr) { PFIN(pfc);   pfc += PFSTEP; }
        if (pfw) { PFWMID(pwc); pwc += PFSTEP; }
        vec Zv[L], Wv[L];
        _Pragma("GCC unroll 22")
        for (int zg = 0; zg < NFULL; ++zg) {
            const int zb = zg * PW;
            vec r_[PW];
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j)
                r_[j] = LDU(rows + (size_t)j * (2 * L) + 2 * zb);
            TRNC(r_, &Zv[zb]);
        }
        /* odd 45th column via 16 B column gather (no overlapped split loads) */
        GCOL(Zv[L - 1], rows + 2 * (L - 1), 2 * L);
#define LD1(n)    Zv[n]
#define ST1(k, v) (Wv[k] = (v))
        PFA45R(LD1, ST1);        /* array-ST site: DFT9-first order wins */
#undef LD1
#undef ST1
        _Pragma("GCC unroll 22")
        for (int zg = 0; zg < NFULL; ++zg) {
            const int zb = zg * PW;
            vec r_[PW];
            TRNC(&Wv[zb], r_);
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j)
                STU(prow + (size_t)j * (2 * PPITCH) + 2 * zb, r_[j]);
        }
        /* odd column scatter: lane j of Wv[44] -> row yb+j, 16 B stores */
        SCOL(Wv[L - 1], prow + 2 * (L - 1), 2 * PPITCH);
    }
#ifndef FFT45_OVERLAP_TAIL
    /* odd 45th row: one PW=1 line, in row 44 -> pl row 44 (contiguous in,
     * no transposes, 16 B accesses that never split a line) */
    dft45_line1(px + (size_t)(L - 1) * (2 * L), 2,
                pld + (size_t)(L - 1) * (2 * PPITCH), 2);
#endif

    for (int zg = 0; zg < NG1; ++zg) {
        const int zb = (zg == NGRP - 1) ? (L - PW) : zg * PW;
        const double *pcol = pld + 2 * zb;      /* single runtime base, again */
        double       *mcol = mx  + 2 * zb;
        /* opaque-base barrier: without it gcc re-associates every load
         * address as pld + (zb*16) + n*832, hoists the 45 loop-invariant
         * (pld + n*832) leas, SPILLS them, and reloads one per vector load
         * -- the r6 offset-table pathology relocated into the y-subloop
         * (measured here: 48 leas + 37 GPR spills per exec, -163 instr and
         * all spill-reload serialization gone with the barrier). */
        __asm__("" : "+r"(pcol), "+r"(mcol));
        if (pfr) { PFIN(pfc);   pfc += PFSTEP; }
        if (pfw) { PFWMID(pwc); pwc += PFSTEP; }
#define LD2(n)    LDU(pcol + (size_t)(n) * (2 * PPITCH))
#define ST2(k, v) STU(mcol + (size_t)(k) * mrow, (v))
        PFA45(LD2, ST2);
#undef LD2
#undef ST2
    }
#ifndef FFT45_OVERLAP_TAIL
    /* odd 45th column: one PW=1 line, pl column 44 -> mid column 44 */
    dft45_line1(pld + 2 * (L - 1), 2 * PPITCH, mx + 2 * (L - 1), mrow);
#endif
}

static inline __attribute__((always_inline))
void FN(phase1)(const double *restrict in, double *restrict mid,
                double *restrict pld, const int pfr, const int pfw,
                const long mrow, const long mpln)
{
    for (int x = 0; x < L; ++x)
        FN(phase1_plane)(in, mid, pld, x, pfr, pfw, mrow, mpln);
}

/* phase-2 read prefetch: 45 sequential streams at PLND-double stride is more
 * than the L2 streamer tracks (a runtime pf level in r7; compile-time now) */
#define PF45(s_) do {                                                        \
    _Pragma("GCC unroll 45")                                                 \
    for (int n_ = 0; n_ < 45; ++n_)                                          \
        __builtin_prefetch((s_) + (size_t)n_ * PLND + 8, 0, 3);              \
} while (0)
/* sp-pfs: write-intent prefetch of the 45 cold out-streams one tile ahead */
#define PFW45(d_) do {                                                       \
    _Pragma("GCC unroll 45")                                                 \
    for (int n_ = 0; n_ < 45; ++n_)                                          \
        __builtin_prefetch((d_) + (size_t)n_ * PLND + 8, 1, 3);              \
} while (0)
/* per-tile pre-coverage of the NEXT volume's input (first ~63 KB) */
#define PFNX() do { if (pfn && pn_) {                                         \
    _Pragma("GCC unroll 4")                                                   \
    for (int q_ = 0; q_ < FFT45_PFN; ++q_)                                    \
        __builtin_prefetch(pn_ + 8 * q_, 0, FFT45_PFH);                       \
    pn_ += 8 * FFT45_PFN; } } while (0)

/* phase 2, INPLACE: x transform in place in `out` (the codelet reads all 45
 * inputs before its first store).  Tiled over the FLAT (y,z) index:
 * 2025 = 506*4 + 1 (PW=4) leaves ONE masked tail call per volume.
 * (Flat tiling borrowed from L45_mixedradix r7.) */
static inline __attribute__((always_inline))
void FN(phase2)(const double *mid, double *out, const double *pnext,
                const int pfx, const int pfn)
{
    const double *pn_ = pnext;
    for (int t = 0; t < NPLANE / PW; ++t) {
        const size_t o = (size_t)t * (PW * 2);
        const double *s_ = mid + o;
        double       *d_ = out + o;
        if (pfx) PF45(s_);
        PFNX();
#define LD3(n)    LDU(s_ + (size_t)(n) * PLND)
#define ST3(k, v) STU(d_ + (size_t)(k) * PLND, (v))
        PFA45(LD3, ST3);
#undef LD3
#undef ST3
    }
    {   /* tail: flat line 2024 = (y,z) = (44,44); lane 0 live, dead lanes 0 */
        const double *s_ = mid + (size_t)(NPLANE - 1) * 2;
        double       *d_ = out + (size_t)(NPLANE - 1) * 2;
#define LD4(n)    LDT(s_ + (size_t)(n) * PLND)
#define ST4(k, v) STT(d_ + (size_t)(k) * PLND, (v))
        PFA45(LD4, ST4);
#undef LD4
#undef ST4
    }
}

/* phase 2, ONE flat unit: tile t < NFT is a full-width tile, t == NFT the
 * masked PW=1 tail (flat line 2024).  The mtf/grp2 sweeps' unit;
 * recomputing s_/d_ from t each call is two leas against a 344-op codelet.
 * pfx (compile-time) pokes the 45 read streams one line ahead (PF45) --
 * at a fused/pair sweep the mid lines are dirty in OTHER cores' caches,
 * the cross-L2 handoff L17_matrixsimd's xpf won 0.4-0.7 us on. */
#define NFT (NPLANE / PW)
static inline __attribute__((always_inline))
void FN(phase2_unit)(double *vol, const int t, const int pfx)
{
    if (t < NFT) {
        const size_t o = (size_t)t * (PW * 2);
        const double *s_ = vol + o;
        double       *d_ = vol + o;
        if (pfx) PF45(s_);
#define LD6(n)    LDU(s_ + (size_t)(n) * PLND)
#define ST6(k, v) STU(d_ + (size_t)(k) * PLND, (v))
        PFA45(LD6, ST6);
#undef LD6
#undef ST6
    } else {
        const double *s_ = vol + (size_t)(NPLANE - 1) * 2;
        double       *d_ = vol + (size_t)(NPLANE - 1) * 2;
#define LD7(n)    LDT(s_ + (size_t)(n) * PLND)
#define ST7(k, v) STT(d_ + (size_t)(k) * PLND, (v))
        PFA45(LD7, ST7);
#undef LD7
#undef ST7
    }
}

/* One-volume non-temporal burst copy, M -> out.  The mtn variants' whole
 * point: with 32 cores sharing DRAM the batched cells are bandwidth-bound
 * and the out-RFO is a third of the traffic (in 1.46 + RFO 1.46 + out 1.46
 * MB/vol); L13_rader/L17_rader/L23_matrixsimd all measured NT flipping from
 * the settled phase-1 loser to a -13..-25% winner in this regime.  Direct
 * NT on the transform's own stores is impossible here (every 64 B store is
 * only 16B-aligned, rows are 720 B), so the volume is computed in the
 * thread's private M (1.46 MB, one SPR L2) and flushed linearly: head peel
 * to 64 B (a volume's out base rotates 16 B per volume: VDBL*8 mod 64 =
 * 16), aligned vmovntpd body, scalar tail.  Same values, same places =>
 * bit-identical.  sfence is the caller's (once per thread per execute). */
static void FN(ntcopy)(double *restrict dst, const double *restrict src,
                       const size_t n)
{
    size_t head = ((64 - ((uintptr_t)dst & 63)) & 63) / 8;
    if (head > n) head = n;
    for (size_t q = 0; q < head; ++q) dst[q] = src[q];
#if PW == 4
    const size_t body = (n - head) & ~(size_t)7;
    for (size_t q = head; q < head + body; q += 8)
        _mm512_stream_pd(dst + q, _mm512_loadu_pd(src + q));
#elif defined(__AVX__)
    const size_t body = (n - head) & ~(size_t)3;
    for (size_t q = head; q < head + body; q += 4)
        _mm256_stream_pd(dst + q, _mm256_loadu_pd(src + q));
#else
    const size_t body = n - head;
    memcpy(dst + head, src + head, body * sizeof(double));
#endif
    for (size_t q = head + body; q < n; ++q) dst[q] = src[q];
}

/* phase 2 with STAGED NON-TEMPORAL output (mt_r2, the "mts"/"g2n" path):
 * the x transform reads mid (the thread's or pair's private M) and writes
 * each block of TBK flat tiles into a small tile buffer tb (45 rows of
 * TBK*PW complex; row stride TBK*PW*2 doubles = 1 KiB, 64B-aligned), which
 * is then flushed to out as 45 short NT segments.  Against mtn (phase 2 in
 * place in M + one linear ntcopy) this deletes one full write AND one full
 * re-read of M per volume (~2.9 MB of L2/L3 traffic; M at 1.46 MB does not
 * fit the node's 1 MiB L2, so that traffic is real there).  The segment
 * head/tail peels are regular stores: at TBK=16 the peeled boundary lines
 * cost ~6% of the out traffic in RFOs -- the price of out's 16 B per-plane
 * phase rotation (PLND*8 mod 64 = 16), same values, same places, so the
 * result is bit-identical to every other exec.  Handles any tile range
 * [u0, u1); u1 == NFT+1 includes the masked PW=1 tail, written to out
 * directly (16 B masked stores, 45 lines per volume). */
#define TBK 16
static inline __attribute__((always_inline))
void FN(phase2_nts)(const double *restrict mid, double *restrict out,
                    double *restrict tb, const long u0, const long u1,
                    const double *pnext, const int pfn)
{
    const double *pn_ = pnext;
    const long uf = u1 < NFT ? u1 : NFT;
    for (long ub = u0; ub < uf; ) {
        long nb = uf - ub;
        if (nb > TBK) nb = TBK;
        for (long j = 0; j < nb; ++j) {
            const double *s_ = mid + (size_t)(ub + j) * (PW * 2);
            double       *d_ = tb  + (size_t)j * (PW * 2);
#define LD8(n)    LDU(s_ + (size_t)(n) * PLND)
#define ST8(k, v) STU(d_ + (size_t)(k) * (TBK * PW * 2), (v))
            PFA45(LD8, ST8);
#undef LD8
#undef ST8
            PFNX();
        }
        for (int k = 0; k < L; ++k)
            FN(ntcopy)(out + (size_t)k * PLND + (size_t)ub * (PW * 2),
                       tb  + (size_t)k * (TBK * PW * 2),
                       (size_t)nb * (PW * 2));
        ub += nb;
    }
    if (u1 > NFT) {   /* masked PW=1 tail: flat line 2024 = (y,z) = (44,44) */
        const double *s_ = mid + (size_t)(NPLANE - 1) * 2;
        double       *d_ = out + (size_t)(NPLANE - 1) * 2;
#define LD9(n)    LDT(s_ + (size_t)(n) * PLND)
#define ST9(k, v) STT(d_ + (size_t)(k) * PLND, (v))
        PFA45(LD9, ST9);
#undef LD9
#undef ST9
    }
}

/* ---- exec bodies: every pf/nt argument below is a literal, so const-
 * propagation through the always_inline bodies specializes each work
 * function completely (the L45_mixedradix structure). --------------------- */

/* serial per-volume pipeline, shared by the serial controls and mtv */
static inline __attribute__((always_inline))
void FN(vols_ip)(const double *in, double *out, long b0, long b1,
                 double *P, const int pfr, const int pfw)
{
    for (long b = b0; b < b1; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        const double *nx = (pfr && b + 1 < b1) ? i + VDBL : 0;
        FN(phase1)(i, o, P, pfr, pfw, 2 * L, PLND);
        FN(phase2)(o, o, nx, pfr, pfr);
    }
}

/* NT-staged per-volume pipeline (mtn): both phases run in the thread's
 * private M (mid writes and phase-2 reads never touch DRAM when M holds in
 * cache), then one NT burst to out.  pfr paces PFIN over `in` in phase 1
 * and PFNX pre-covers the NEXT volume's input during phase 2 (worth 30% to
 * L13_rader's threaded streaming); PF45/prefetchw stay off -- M is
 * cache-resident, and poking resident lines is a pure uop tax (L23's +0.9
 * us lesson, L36_pfa's +13%). */
static inline __attribute__((always_inline))
void FN(vols_nt)(const double *in, double *out, long b0, long b1,
                 double *P, double *M, const int pfr)
{
    for (long b = b0; b < b1; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        const double *nx = (pfr && b + 1 < b1) ? i + VDBL : 0;
        FN(phase1)(i, M, P, pfr, 0, 2 * L, PLND);
        FN(phase2)(M, M, nx, 0, pfr);
        FN(ntcopy)(out + (size_t)b * VDBL, M, VDBL);
    }
    if (b1 > b0) _mm_sfence();
}

/* staged-tile NT pipeline (mts): phase 1 into M, phase 2 M -> tb -> NT out */
static inline __attribute__((always_inline))
void FN(vols_nts)(const double *in, double *out, long b0, long b1,
                  double *P, double *M, double *tb, const int pfr)
{
    for (long b = b0; b < b1; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        const double *nx = (pfr && b + 1 < b1) ? i + VDBL : 0;
        FN(phase1)(i, M, P, pfr, 0, 2 * L, PLND);
        FN(phase2_nts)(M, out + (size_t)b * VDBL, tb, 0, NFT + 1, nx, pfr);
    }
    if (b1 > b0) _mm_sfence();
}

/* serial control rows (main thread, slot-0 scratch) */
static void FN(x_ip0)(const double *in, double *out, long nvol,
                      struct l45mt *pl)
{
    FN(vols_ip)(in, out, 0, nvol, pl->Pt[0], 0, 0);
}

static void FN(x_ip3)(const double *in, double *out, long nvol,
                      struct l45mt *pl)
{
    FN(vols_ip)(in, out, 0, nvol, pl->Pt[0], 1, 1);
}

/* mtv/mtn/mts: volume-parallel.  Thread t owns the CONTIGUOUS block
 * [nvol*t/T, nvol*(t+1)/T) and runs the serial pipeline on it with its own
 * NUMA-local scratch; no synchronisation except the join, and each thread
 * touches the same volumes every call (prefetcher- and NUMA-stable).
 * (Decomposition from L13_direct mt_r1.)  At nvol < T the spare threads
 * idle into the join -- grp2/mtf cover that regime, the tuner picks. */
static inline __attribute__((always_inline))
void FN(vol_body)(struct l45mt *pl, const int tid, const int nt,
                  const int nts, const int pfr, const int pfw)
{
    const int T = pl->tw;
    if (tid < T) {
        const long lo = pl->nvol * tid / T, hi = pl->nvol * (tid + 1) / T;
        if (nts)
            FN(vols_nts)(pl->in, pl->out, lo, hi,
                         pl->Pt[tid], pl->Mt[tid], pl->Tb[tid], pfr);
        else if (nt)
            FN(vols_nt)(pl->in, pl->out, lo, hi,
                        pl->Pt[tid], pl->Mt[tid], pfr);
        else
            FN(vols_ip)(pl->in, pl->out, lo, hi, pl->Pt[tid], pfr, pfw);
    }
    mt_join(pl, tid);
}

static void FN(w_v0)(struct l45mt *pl, int tid) { FN(vol_body)(pl, tid, 0, 0, 0, 0); }
static void FN(w_v3)(struct l45mt *pl, int tid) { FN(vol_body)(pl, tid, 0, 0, 1, 1); }
static void FN(w_n0)(struct l45mt *pl, int tid) { FN(vol_body)(pl, tid, 1, 0, 0, 0); }
static void FN(w_ni)(struct l45mt *pl, int tid) { FN(vol_body)(pl, tid, 1, 0, 1, 0); }
static void FN(w_s0)(struct l45mt *pl, int tid) { FN(vol_body)(pl, tid, 0, 1, 0, 0); }
static void FN(w_si)(struct l45mt *pl, int tid) { FN(vol_body)(pl, tid, 0, 1, 1, 0); }

/* grp2: pair-grained volumes (BORROWED from L45_mixedradix mt_r1's grp
 * mode, their node B=16 winner).  The pair (2g, 2g+1) -- adjacent cores
 * under close/cores, same L3 on the node -- owns the contiguous volume
 * block [nvol*g/ng, nvol*(g+1)/ng); member m does x-planes
 * [45m/2, 45(m+1)/2), ONE pair barrier, then phase-2 tiles
 * [507m/2, 507(m+1)/2).  One barrier per volume only: a member's phase 1
 * of volume b+1 touches only mid[b+1] and private scratch, so it never
 * waits on its sibling still in volume b's phase 2.  Against mtf this
 * removes the global straggler coupling (a pair moves on the moment ITS
 * volume's phase 1 is done) at B in [2, 32).
 *   nt=0: mid = out in place; pfr adds pfin+prefetchw in phase 1 (the
 *         rival's winning v1 mechanism).
 *   nt=1: mid = the pair's shared M, phase 2 via the staged-tile NT path,
 *         each member flushing exactly the tile range it computed -- full
 *         team AND no out-RFO, the combination neither mtf nor mtn has.
 *         This is my own r1 "next round" item 4, now affordable because a
 *         pair spin barrier is ~0.1 us against GOMP's 6-8.  M is DOUBLE-
 *         BUFFERED on volume parity across the pair's two Mt slots: with a
 *         single slot, a member's phase 1 of volume b+1 overwrites M while
 *         its sibling is still READING M in volume b's phase 2 -- a real
 *         race the create()-time gate caught at nv=128 (g2n-pf0 gated OUT;
 *         at B=16 every pair owns one volume and it cannot trigger).  With
 *         two slots the one pair barrier still orders slot reuse: a member
 *         reaches phase 1 of b+2 (slot of b) only through the barrier of
 *         b+1, which its partner only passes after finishing phase 2 of b. */
static inline __attribute__((always_inline))
void FN(grp2_body)(struct l45mt *pl, const int tid, const int nt,
                   const int pfr)
{
    const int T = pl->tw, ng = T / 2;
    if (ng >= 1 && tid < 2 * ng) {
        const long nb = pl->nvol;
        const int g = tid >> 1, m = tid & 1;
        const long lo = nb * (long)g / ng, hi = nb * (long)(g + 1) / ng;
        const int x0 = m ? L / 2 : 0, x1 = m ? L : L / 2;
        const long t0 = (long)(NFT + 1) * m / 2;
        const long t1 = (long)(NFT + 1) * (m + 1) / 2;
        const unsigned long eb =
            (unsigned long)atomic_load_explicit(&pl->gen,
                                                memory_order_relaxed)
                * pl->emult;
        double *P  = pl->Pt[tid];
        double *tb = pl->Tb[tid];
        for (long b = lo; b < hi; ++b) {
            const double *i = pl->in  + (size_t)b * VDBL;
            double       *o = pl->out + (size_t)b * VDBL;
            double     *mid = nt ? pl->Mt[(tid & ~1) | ((int)b & 1)] : o;
            const double *nx = (pfr && b + 1 < hi) ? i + VDBL : 0;
            for (int x = x0; x < x1; ++x)
                FN(phase1_plane)(i, mid, P, x, pfr, pfr && !nt,
                                 2 * L, PLND);
            mt_pbar(pl, tid, g, eb + (unsigned long)(b - lo) + 1);
            if (nt)
                FN(phase2_nts)(mid, o, tb, t0, t1, nx, pfr);
            else
                for (long u = t0; u < t1; ++u)
                    FN(phase2_unit)(o, (int)u, 0);
        }
        if (nt) _mm_sfence();
    }
    mt_join(pl, tid);
}

static void FN(w_g0)(struct l45mt *pl, int tid)  { FN(grp2_body)(pl, tid, 0, 0); }
static void FN(w_gi)(struct l45mt *pl, int tid)  { FN(grp2_body)(pl, tid, 0, 1); }
static void FN(w_gn0)(struct l45mt *pl, int tid) { FN(grp2_body)(pl, tid, 1, 0); }
static void FN(w_gni)(struct l45mt *pl, int tid) { FN(grp2_body)(pl, tid, 1, 1); }

/* mtf: fused two-sweep, the B=1 shape.  Sweep 1 runs ALL volumes' x-plane
 * pipelines (z+y transform of one plane into mid=out, each thread on its
 * own plane scratch); ONE flag-array barrier (every phase-2 line reads all
 * 45 mid planes of its volume); sweep 2 runs all flat phase-2 tiles in
 * place, and the pool join is the only remaining sync (L23_rader: a
 * redundant second barrier cost 11% at B=1).  The plane-sweep schedule is
 * a compile-time knob: rr=1 round-robins single planes (45 = 32 + 13, so a
 * blocked schedule idles 9 threads behind the 2-plane threads at B=1);
 * rr=0 hands out CONTIGUOUS plane blocks (the node's r1 pick at both B=1
 * and B=16).  pfr adds pfin+prefetchw to phase 1 and the PF45 cross-L2
 * poke to phase 2. */
static inline __attribute__((always_inline))
void FN(fused_body)(struct l45mt *pl, const int tid, const int rr,
                    const int pfr)
{
    const int T = pl->tw;
    if (tid < T) {
        double *P = pl->Pt[tid];
        const long n1 = pl->nvol * L;
        const unsigned long eb =
            (unsigned long)atomic_load_explicit(&pl->gen,
                                                memory_order_relaxed)
                * pl->emult;
        if (rr) {
            for (long u = tid; u < n1; u += T)
                FN(phase1_plane)(pl->in  + (size_t)(u / L) * VDBL,
                                 pl->out + (size_t)(u / L) * VDBL,
                                 P, (int)(u % L), pfr, pfr, 2 * L, PLND);
        } else {
            const long lo = n1 * tid / T, hi = n1 * (tid + 1) / T;
            for (long u = lo; u < hi; ++u)
                FN(phase1_plane)(pl->in  + (size_t)(u / L) * VDBL,
                                 pl->out + (size_t)(u / L) * VDBL,
                                 P, (int)(u % L), pfr, pfr, 2 * L, PLND);
        }
        mt_gbar(pl, tid, T, eb + 1);
        const long n2 = pl->nvol * (NFT + 1);
        const long lo2 = n2 * tid / T, hi2 = n2 * (tid + 1) / T;
        for (long u = lo2; u < hi2; ++u)
            FN(phase2_unit)(pl->out + (size_t)(u / (NFT + 1)) * VDBL,
                            (int)(u % (NFT + 1)), pfr);
    }
    mt_join(pl, tid);
}

static void FN(w_f0)(struct l45mt *pl, int tid)  { FN(fused_body)(pl, tid, 1, 0); }
static void FN(w_fb)(struct l45mt *pl, int tid)  { FN(fused_body)(pl, tid, 0, 0); }
static void FN(w_fb3)(struct l45mt *pl, int tid) { FN(fused_body)(pl, tid, 0, 1); }

#undef TBK

#undef PF45
#undef PFW45
#undef PFNX
#undef PFIN
#undef PFWMID
#undef PFSTEP
#undef PFL

#undef NFT
#undef TRNC
#undef LDT
#undef STT
#undef GCOL
#undef SCOL
#undef VFMA
#undef VFNMA
#undef SWAP
#undef VPAIR
#undef VSPLAT
#undef NG1
#undef NFULL
#undef NGRP
#undef LDU
#undef STU
#undef VSH
#undef FN
#undef veci
#undef uvec
#undef vec

#endif /* template */
