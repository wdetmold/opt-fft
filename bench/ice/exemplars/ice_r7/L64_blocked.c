/* L64_blocked.c -- forward complex 3D DFT of a 64^3 cube, batched, out-of-place.
 *
 * ROUND ice_r7 (the compiler round: both queued structural moves DECLINED by
 * the node; the win came out of their post-mortem):
 *   1. zs_ztail FORCE-INLINED: gcc 11.4 had declined the plain `inline` at
 *      all three z-row call sites since r5 -- every z-row spilled its 16
 *      Vr/Vi zmm to the stack, reloaded them in the callee, and paid a
 *      stack-protector canary.  Node vs the r6 exemplar, same window:
 *      615/622/623 vs 646/642/645 us/step-vol (-3.5%), bit-identical.
 *   2. ZAPF distance 3 -> 2 (the shorter post-inline body wants the shorter
 *      lead): 709.6/711.1 vs 718.8/718.8 vs 727.3/727.5 (d2/d3/d4).
 *   3. ZMS (rival v6_3f30d81f's x-pass map fusion + cross-step z-pipelining,
 *      both variants) DECLINED: 686-712 vs 624 -- the split phases lose
 *      sweep A's free y-line/map-ladder overlap; slab-major c (zs_build_c_
 *      slab) fixed its scattered-c pattern (~6 GB/s) and still lost.
 *      ZVP vol-pair row interleaving DECLINED: 706/709 vs 611/612 (+16%).
 *      ZYW y-phase next-plane warmup: wash.  All kept as env/-D controls,
 *      all cmp-identical to the shipped output.
 *   Node, final build: B=2 607.1 (MKL same-window 1717.5, ratio 0.353),
 *   B=1 606.2, B=8 608.7; chain drift 1.69e-12 vs tol 1.34e-11; repeatable
 *   bit-identical.  Best rival L=64 re-benchmarked on this node: 653 us.
 *
 * ROUND ice_r6 (schedule round, guided by the first-ever cycle attribution):
 *   1. FFT64B_PROF=1 rdtsc attribution (the item promised since r4): sweep A
 *      is 73% of the step (~1.47M cyc/step-vol) at only ~26 GB/s effective;
 *      sweep B is 26% (~0.53M) at ~49 GB/s.  The r5 record's "sweep B is the
 *      latency-exposed one" guess was exactly backwards.
 *   2. ZAPF: sweep A map-row prefetch, T0, 3 rows ahead, state AND c -- the
 *      rivals' 1000f989 iter_z64 schedule verbatim (they prefetch x+3 T0).
 *      Node: 652/660 vs 679 us/step-vol same-window pairs (~-3.5%).  T1
 *      hints or distance 2 LOSE (754, 699); plane-wrap coverage (ZAW) LOSES
 *      (653/655 vs 637/643): late-plane T0 bursts displace the L1-hot plane
 *      the y-line phase is about to re-read.
 *   3. ZTC: sweep B's next-body prefetch now crosses the g=7 -> (ky+1,0)
 *      seam (was aiming at the row pad line).  Wash on the node (679.2 vs
 *      679.7), kept: principled, free.  ZT0H=2 (T1) and ZT2 (2-ahead) lose.
 *   4. ZS2 g-slab rebalance (y-lines moved from sweep A to a per-g sweep B2,
 *      512-KB slab L2-resident for the x-pass, bit-identical): DECLINED,
 *      ~+3% (668.9/699.3 vs 649.6/656.6).  Kept env/-D-forcible.
 *   Node, final build: B=2 641.1 (MKL same-window 1927.8), B=1 713.5,
 *   B=8 644.5; chain drift 1.69e-12 at B=2 (tol 1.34e-11), bit-identical.
 *
 * ROUND ice_r5 (the z-split custody chain -- the corpus 10 6 structural move,
 * "the single biggest structural win of the best run", adapted):
 *   1. fft3d_chain now keeps the state, for the WHOLE m-step chain, in a
 *      split-complex padded custody layout (slot (x,y,g) at x*SCXS + y*SCKS
 *      + g*16 doubles, lanes = 8 ADJACENT indices), and iterates EACH VOLUME
 *      through all m steps while cache-resident: working set 8.9 MB (state
 *      4.46 + custody c 4.46) vs r4's ~30 MB, per-step traffic ~22 MB all-L3
 *      vs ~30 MB partly-DRAM.  One step = two in-place sweeps (A: lazy map +
 *      z-line + y-lines per 68-KB L2-hot x-plane; B: x-lines per ky-slab).
 *      Natural interleaved data is touched only at the chain ends.
 *   2. Unlike the rivals' octant-lane z-split (1000f989 run64_zsplit:
 *      natural/bit-reversed alternation, cross-lane masked DFT8 at ~2x the
 *      vector ops), lane = index mod 8 / slot = index div 8 makes the
 *      transpose-based z-line (DFT8S, CTWV lane twiddle, ONE TR8 pair,
 *      DFT8S -- the sc_pass23 tail verbatim) map custody form to custody
 *      form exactly: stable across steps, no bit reversal, no alternation.
 *   3. The r2-r4 chain-arena tournament is retired (it tuned the pipeline
 *      the custody chain replaces); create() is one correctness gate + two
 *      chain interlocks (custody first, r4 lazy-map pipeline as verified
 *      fallback; FFT64B_ZS=0 / FFT64B_CHAIN=0 for control runs).
 *
 * ROUND ice_r4 (task change: the graded step is now state <- (z+c)/(1+|z+c|),
 * z = FFT(state), and the driver detects an optional fft3d_chain weak symbol;
 * without it we are timed through the unfused execute+map fallback):
 *   1. fft3d_chain EXPORTED: the whole m-step chain fused.  LAZY MAP (corpus
 *      10 2, the rivals' 4/7-convergent shape): ping-pong buffers hold RAW
 *      FFT output between steps; the map is applied inside the NEXT step's
 *      pass 1, where raw z and c both stream sequentially plane-by-plane and
 *      the y-FFT consumes the mapped state in split form -- the map costs no
 *      extra memory pass, only c's 4.19 MB/vol of reads.  One streaming
 *      epilogue maps the final step (1/m of the chain).
 *   2. MAP8V: rsqrt14 seed + 2 Newton steps on the FMA pipes for sqrt(m2),
 *      then ONE exact hardware vdivpd per 8 points (divider runs parallel to
 *      FMA and is the idle unit here); 1e-300 bias kills the m2=0 NaN and
 *      denormal-assist traps.  Map rel err ~1e-16/application -- effectively
 *      exact, no precision-tier gamble against the 1e-13/step chain gate.
 *   3. Tuner arena re-targeted AGAIN at the graded semantics (the ice_r2
 *      lesson): candidates and the sb/pro A/Bs are timed on fused chain
 *      steps with a 0.1-scaled c field.  New create-time interlock: 3-step
 *      fused chain vs an exact chain built from the INDEPENDENT st=0 kernel
 *      + scalar sqrt/divide map, rel L2 < 1e-13 gate; chain_ok=0 (or env
 *      FFT64B_CHAIN=0) falls back to execute + exact scalar map.
 *
 * ROUND ice_r2 (Ice Lake panel.  ice_r1's agent crashed at launch, so this
 * file reached the ICX node as the untouched CLX build; its cold-streaming
 * tuner picked mode=nt at B=2 and the board read 1205.7 us vs MKL 1016.4 --
 * the panel's one loss.  Three changes, no arithmetic change):
 *   1. CHAIN-TUNED ARENA: the graded workload is --chain 134 --unitary; the
 *      driver re-reads every output immediately (cached unitary scaling
 *      pass) and feeds it back as the next step's input, ping-pong, all
 *      L3-resident.  The old arena timed cold streaming volumes, where NT
 *      wins; under the chain an NT store is a 4-MiB DRAM round trip per
 *      step.  The tuner now times steady-state chain steps (chain_step /
 *      chain_round: execute + the driver's scaling loop, two self-warming
 *      steps, then timed ping-pong).  Node, same window: nt-pf0 1514.9 vs
 *      cached-pf0 1263.0 us/step-vol.  Result: B=2 903.1 vs MKL 1006.8
 *      (0.90x), B=1 827.7 vs 978.3, B=8 1188.6 vs 1547.4 -- the loss is
 *      flipped to a win at every batch.
 *   2. sb (new A/B axis): pass-2's x-FFT may write a 68-KB slab SB (rows
 *      SBKS=136 doubles = 17 lines, odd; lives in the OB region) instead
 *      of in-place SC, making SC read-only in pass 2 -- kills ~4.5 MB/vol
 *      of dead dirty-writeback.  Node declined it at B=2 (off 975.9 / on
 *      1023.7): L3 writeback is not the wall under the chain.  Kept as an
 *      A/B; bit-identical either way (cmp-verified, single + chain).
 *   3. p1pf/slabpf hints -D-sweepable (FFT64B_H1 / FFT64B_HS, default T1);
 *      T0 twins were within window noise on the node, defaults kept.
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
/* st=3 prefetch hints (ice_r2, node-sweepable): H1 = pass-1 next-plane and
 * pro hint, HS = ky-slab hint (0=NTA 1=T2 2=T1 3=T0; CLX shipped T1 for
 * both, but under the graded chain every buffer is L3-hot, where T0's
 * L2-fill is the right target). */
#ifndef FFT64B_H1
# define FFT64B_H1 2
#endif
#ifndef FFT64B_HS
# define FFT64B_HS 2
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

/* ice_r4 graded map, split form:  (rr,ii) = w / (1 + |w|),  w = (wr,wi).
 * The rival-consensus shape (corpus 10 2, verified in the 1.00-scorer's
 * mapc): rsqrt14 seed + 2 Newton steps on the FMA pipes for sqrt(m2), then
 * ONE exact hardware divide per 8 points -- the divider is a separate unit
 * that hides under the FFT's FMA work, and the exact vdivpd protects the
 * 1e-13/step chain gate (rel err of the whole map ~1e-16: seed 2^-14 ->
 * 5.6e-9 -> 4.7e-17 after two quadratic steps).  The 1e-300 bias makes
 * m2=0 safe (rsqrt14(0)=inf would NaN via 0*inf) and is invisible: it only
 * perturbs m2 < 1e-287, where den = 1+|w| = 1.0 exactly either way.
 * 13 FMA-port ops + 1 rsqrt14 + 1 vdivpd per 8 complex points. */
/* FFT64B_MAPDIV selects the reciprocal/rsqrt ladder (create-time interlock
 * gates whichever is built):
 *   2 rsqrt14 + 2 quadratic Newtons, one exact vdivpd -- map rel err
 *     ~1e-16/application, 13 FMA-port ops + 1 divide per 8 points.
 *   1 rsqrt14 + 1 CUBIC Newton (err ~3e-13/app), one exact vdivpd -- 11
 *     FMA-port ops + 1 divide.
 *   0 (DEFAULT, node-raced r4: 1108 vs 1159/1162 us/step-vol -- pass 1 was
 *     vdivpd-throughput-bound, 32768 divides x ~16 cyc all in one pass)
 *     all-FMA: cubic rsqrt + rcp14 with 1 cubic Newton (err ~4e-13/app,
 *     measured m=134 chain end drift 1.68e-12 vs tol 1.34e-11), no divider
 *     use at all -- 15 FMA-port ops.  Legal at L=64 only because the map is
 *     a contraction; at L=6/8/13 chain lengths use MAPDIV=2. */
#ifndef FFT64B_MAPDIV
# define FFT64B_MAPDIV 0
#endif
#if FFT64B_MAPDIV == 2
# define MAPRSQ(q_, m2_) do {                                                 \
    v8d h_ = VSPL8(0.5) * m2_;                                                \
    q_ = q_ * V8FNMA(h_ * q_, q_, VSPL8(1.5));                                \
    q_ = q_ * V8FNMA(h_ * q_, q_, VSPL8(1.5));                                \
} while (0)
#else  /* one cubic step: y*(15/8 - 5/4 t + 3/8 t^2), t = m2*y^2 */
# define MAPRSQ(q_, m2_) do {                                                 \
    v8d t_ = m2_ * (q_ * q_);                                                 \
    q_ = q_ * V8FMA(t_, V8FMA(t_, VSPL8(0.375), VSPL8(-1.25)), VSPL8(1.875)); \
} while (0)
#endif
#if FFT64B_MAPDIV >= 1
# define MAPRCP(s_, den_) \
    v8d s_ = (v8d)_mm512_div_pd((__m512d)VSPL8(1.0), (__m512d)den_)
#else  /* rcp14 + 1 cubic step: r' = r + r*(e + e^2), e = 1 - den*r */
# define MAPRCP(s_, den_) \
    v8d s_ = (v8d)_mm512_rcp14_pd((__m512d)den_);                             \
    do {                                                                      \
        v8d e_ = V8FNMA(den_, s_, VSPL8(1.0));                                \
        s_ = V8FMA(s_, V8FMA(e_, e_, e_), s_);                                \
    } while (0)
#endif
#define MAP8V(wr, wi, rr, ii) do {                                            \
    v8d m2_ = V8FMA((wr), (wr), V8FMA((wi), (wi), VSPL8(1e-300)));            \
    v8d q_  = (v8d)_mm512_rsqrt14_pd((__m512d)m2_);                           \
    MAPRSQ(q_, m2_);                                                          \
    v8d den_ = V8FMA(m2_, q_, VSPL8(1.0));                                    \
    MAPRCP(s_, den_);                                                         \
    (rr) = (wr) * s_; (ii) = (wi) * s_;                                       \
} while (0)

/* standalone streaming map pass over interleaved data (the chain epilogue:
 * the ping-pong buffers stay RAW z between steps, so only the final state
 * needs an explicit map -- 1/m of the chain).  dst may alias z.
 * zsplit: z chunks are SPLIT (re vector at +0, im at +8) -- the chain-
 * internal layout below -- instead of interleaved. */
static void map_rows_v(const double *restrict z, const double *restrict c,
                       double *restrict dst, size_t ndbl, int zsplit)
{
    for (size_t i = 0; i < ndbl; i += 16) {
        v8d A_ = *(const v8d *)(z + i), B_ = *(const v8d *)(z + i + 8);
        v8d C_ = *(const v8d *)(c + i), D_ = *(const v8d *)(c + i + 8);
        v8d wr_, wi_;
        if (zsplit) { wr_ = A_ + DEIN_RE(C_, D_); wi_ = B_ + DEIN_IM(C_, D_); }
        else {
            wr_ = DEIN_RE(A_, B_) + DEIN_RE(C_, D_);
            wi_ = DEIN_IM(A_, B_) + DEIN_IM(C_, D_);
        }
        v8d rr_, ii_;
        MAP8V(wr_, wi_, rr_, ii_);
        *(v8d *)(dst + i)     = ILV_LO(rr_, ii_);
        *(v8d *)(dst + i + 8) = ILV_HI(rr_, ii_);
    }
}

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
                __builtin_prefetch(h_ + 8 * q_, 0, FFT64B_H1);                        \
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

/* pass 1 with the graded map FUSED into the stage-1 loads (the rivals' lazy
 * map, corpus 10 2: the buffer between chain steps stays raw FFT output;
 * the map is applied here, in the one pass where both the raw z and the c
 * field stream sequentially, plane by plane).  ip = raw z of the previous
 * step IN SPLIT LAYOUT (chain-internal steps store (re,im) vector pairs at
 * the interleaved addresses, skipping the ILV/DEIN shuffle round trip --
 * port 5 hosts both a shuffle unit and an FMA pipe here, so every shuffle
 * removed is FMA headroom), cp = the same plane of c (driver layout,
 * interleaved; its deinterleave rides pass 1's idle shuffle port).  The
 * y-FFT consumes the mapped state directly in split form -- the map costs
 * no extra memory pass, only c's 4.19 MB/vol of reads.  pn/pnc: next z/c
 * plane for p1pf. */
static void sc_pass1_map(const double *restrict ip, const double *restrict cp,
                         double *restrict sp,
                         const double *restrict pn, const double *restrict pnc)
{
    for (int zb = 0; zb < 8; ++zb) {
#define SC_HOOK1M(s) do {                                                     \
        if (pn) {                                                             \
            const double *h_  = pn  + ((size_t)(zb * 8 + (s)) * 128);         \
            const double *hc_ = pnc + ((size_t)(zb * 8 + (s)) * 128);         \
            _Pragma("GCC unroll 16")                                          \
            for (int q_ = 0; q_ < 16; ++q_) {                                 \
                __builtin_prefetch(h_  + 8 * q_, 0, FFT64B_H1);               \
                __builtin_prefetch(hc_ + 8 * q_, 0, FFT64B_H1);               \
            }                                                                 \
        }                                                                     \
    } while (0)
#define SC_LD1M(n, rr, ii) do {                                               \
        const double *r1_ = ip + ((size_t)(n) * L + (size_t)zb * 8) * 2;      \
        const double *c1_ = cp + ((size_t)(n) * L + (size_t)zb * 8) * 2;      \
        v8d A1_ = *(const v8d *)r1_, B1_ = *(const v8d *)(r1_ + 8);           \
        v8d C1_ = *(const v8d *)c1_, D1_ = *(const v8d *)(c1_ + 8);           \
        v8d wr_ = A1_ + DEIN_RE(C1_, D1_);        /* z is already split */   \
        v8d wi_ = B1_ + DEIN_IM(C1_, D1_);                                    \
        MAP8V(wr_, wi_, (rr), (ii));                                          \
    } while (0)
#define SC_ST1M(k, rr, ii) do {                                               \
        double *q1_ = sp + (size_t)(k) * SCKS + (size_t)zb * 16;              \
        *(v8d *)q1_ = (rr); *(v8d *)(q1_ + 8) = (ii);                         \
    } while (0)
        FFT64S(SC_LD1M, SC_ST1M, SC_HOOK1M);
#undef SC_LD1M
#undef SC_ST1M
#undef SC_HOOK1M
    }
}

/* fused pass 2+3, one ky: x-FFT over the 64 planes in SC (all 64 loads at
 * SCXS stride complete into the line buffer before the writeback, so
 * in-place is legal; next-column prefetcht0 on every load -- L64_radix8
 * measured that hint +12% at B=8), then the 64 z-lines of the ky-slab, which
 * the x-writeback just made L2-hot: DFT8 across slots (g -> k2), lane
 * twiddles W64^{l*k2}, 8x8 transpose pair, DFT8 (l -> k1), re/im interleave,
 * contiguous 1-KB row store to out.  slab: T1-prefetch ky+1's slab rows (17
 * lines per kx step); pfw: prefetchw the out row FFT64B_PFWL3 kx ahead.
 * SB (ice_r2): if non-NULL, the x-FFT writes its 68-KB ky-slab there (rows
 * of SBKS doubles = 17 lines, one per kx) instead of back in place -- SC
 * becomes read-only in this pass, so its 4.5 MB never turns dirty and the
 * dead post-z writeback traffic to L3 disappears; arithmetic and out bytes
 * are identical either way. */
#ifndef FFT64B_PFWL3
# define FFT64B_PFWL3 4
#endif
#define SBKS 136
_Static_assert((size_t)64 * SCXS + 64 * SBKS <= MIDDBL + OBDBL,
               "st=3 SB slab must fit after the split scratch");
static void sc_pass23(double *restrict SC, double *restrict SB,
                      double *restrict out, int ky,
                      int nt, int pfw, int slab, int splo)
{
    double *kb = SC + (size_t)ky * SCKS;
    for (int zb = 0; zb < 8; ++zb) {
        double *cb = kb + (size_t)zb * 16;
        double *sb = SB ? SB + (size_t)zb * 16 : cb;
        const size_t sbs = SB ? (size_t)SBKS : (size_t)SCXS;
#define SC_LDX(n, rr, ii) do {                                                \
        const double *cx_ = cb + (size_t)(n) * SCXS;                          \
        __builtin_prefetch(cx_ + 16, 0, 3);                                   \
        __builtin_prefetch(cx_ + 24, 0, 3);                                   \
        (rr) = *(const v8d *)cx_; (ii) = *(const v8d *)(cx_ + 8);             \
    } while (0)
#define SC_STX(k, rr, ii) do {                                                \
        double *cx_ = sb + (size_t)(k) * sbs;                                 \
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
                __builtin_prefetch(sl_ + 8 * q_, 0, FFT64B_HS);
        }
        const double *zl = SB ? SB + (size_t)kx * SBKS
                              : SC + (size_t)kx * SCXS + (size_t)ky * SCKS;
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
            /* splo (chain-internal): store the split (re,im) pair straight
             * to the interleaved addresses -- the next step's mapped pass 1
             * (or the epilogue) knows the layout; kills both ILVs here and
             * both DEINs there */
            v8d lo = splo ? Wr[k1] : ILV_LO(Wr[k1], Wi[k1]);
            v8d hi = splo ? Wi[k1] : ILV_HI(Wr[k1], Wi[k1]);
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
                        int pro, int sb, const double *restrict cv, int splo)
{
    const int p1  = (pf == 1 || pf == 8 || pf == 9);
    const int pw_ = (pf == 5 || pf == 7 || pf == 9) && mode == M_CACHED;
    const int sl  = (pf >= 6);
    const int nt  = (mode == M_NT);
    double *SB = sb ? SC + (size_t)64 * SCXS : NULL;  /* 68 KB in the OB region */
    for (int b = 0; b < nvol; ++b) {
        const double *iv = in  + (size_t)b * VDBL;
        double       *ov = out + (size_t)b * VDBL;
        const double *cb = cv ? cv + (size_t)b * VDBL : NULL;
        if (pro) {                     /* input plane 0: 1024 T1 lines (+c) */
            for (int i = 0; i < 1024; ++i)
                __builtin_prefetch(iv + 8 * (size_t)i, 0, FFT64B_H1);
            if (cb)
                for (int i = 0; i < 1024; ++i)
                    __builtin_prefetch(cb + 8 * (size_t)i, 0, FFT64B_H1);
        }
        for (int p = 0; p < L; ++p) {
            const double *pn = NULL, *pnc = NULL;
            if (p1) {
                if (p < L - 1) {
                    pn = iv + (size_t)(p + 1) * (2 * LSQ);
                    if (cb) pnc = cb + (size_t)(p + 1) * (2 * LSQ);
                } else if (b + 1 < nvol) {
                    pn = iv + VDBL;
                    if (cb) pnc = cb + VDBL;
                }
            }
            if (cb) sc_pass1_map(iv + (size_t)p * (2 * LSQ),
                                 cb + (size_t)p * (2 * LSQ),
                                 SC + (size_t)p * SCXS, pn, pnc);
            else    sc_pass1(iv + (size_t)p * (2 * LSQ),
                             SC + (size_t)p * SCXS, pn);
        }
        if (pro)                       /* SC slab ky=0: 64 x 16 T1 lines */
            for (int x = 0; x < 64; ++x) {
                const double *q_ = SC + (size_t)x * SCXS;
                _Pragma("GCC unroll 16")
                for (int i = 0; i < 16; ++i)
                    __builtin_prefetch(q_ + 8 * i, 0, FFT64B_HS);
            }
        for (int ky = 0; ky < L; ++ky)
            sc_pass23(SC, SB, ov, ky, nt, pw_, sl, splo);
    }
}

/* ================= ice_r5: z-split CUSTODY chain ========================= */
/* The corpus 10 6 structural move ("the single biggest structural win of the
 * best run"), finally applicable because fft3d_chain owns the inter-step
 * buffers: the state lives, for the whole m-step chain, in a split-complex
 * padded custody layout -- slot (x, y, g) at x*SCXS + y*SCKS + g*16 doubles
 * (re vector +0, im +8), lanes = 8 ADJACENT values (index = 8g + lane).
 *
 * KEY SIMPLIFICATION over the rivals' 1000f989 run64_zsplit (whose lanes are
 * z-OCTANTS, forcing a natural/bit-reversed form alternation and a cross-lane
 * masked DFT8 costing ~256 arith + 80 shuffle-class ops per 64-pt z-line):
 * with lane = index mod 8 and slot = index div 8, my transpose-based z-line
 *     DFT8S over slots (g -> k2), CTWV lane twiddle W64^{l*k2},
 *     one TR8 pair, DFT8S over slots (l -> k1)
 * maps custody form -> custody form EXACTLY (slot k1 = k div 8, lane k2 =
 * k mod 8): stable across steps, no bit reversal, no A/B alternation, at 132
 * arith + 48 shuffles per z-line -- the sc_pass23 z-tail reused verbatim.
 *
 * One chain step = TWO in-place sweeps of ONE 4.46-MB buffer:
 *   sweep A, per 68-KB (L2-hot) x-plane: lazy map (raw z + custody c, MAP8V,
 *            feeding the z-FFT in registers) on each of the 64 y-rows, then
 *            the 8 y-lines (FFT64S at SCKS stride) in place;
 *   sweep B, per ky-slab: the 8 x-lines (FFT64S at SCXS stride, T0 next-slot
 *            + next-slab prefetch) in place.
 * Each volume iterates through ALL m steps while cache-resident (corpus 10 3
 * verbatim): working set = state 4.46 + custody c 4.46 = 8.9 MB << L3, vs the
 * r4 lazy-map pipeline's ~30 MB (ping+pong+c+SC) that forced NT stores and
 * DRAM round trips.  Per-step traffic ~22 MB all-L3 (A: R+W state, R c;
 * B: R+W state) vs ~30 MB partly-DRAM.  The natural interleaved layout is
 * touched only at the chain ends: step 1 loads x0 rows with a DEIN pair per
 * slot, the epilogue maps and ILV-stores final_out; custody <-> natural is a
 * pure interleave (slot g = 8 consecutive indices), zero transposes. */

#define ZS_VOL ((size_t)64 * SCXS)   /* doubles per custody volume (4.46 MB) */

/* sweep-B next-ky-slab prefetch (the slabpf lineage; hint FFT64B_HS).
 * Node-raced OFF: 667.3/667.6 vs 709.2/710.3 us/step-vol (sd 0.05%, two
 * alternating same-window pairs) -- the 128 extra prefetches per FFT64S are
 * pure issue overhead once the whole volume is L3-resident. */
#ifndef FFT64B_ZSLAB
# define FFT64B_ZSLAB 0
#endif
/* sweep-B T0 next-slot-pair prefetch inside ZS_LDX (the sc_pass23 lineage) */
#ifndef FFT64B_ZT0
# define FFT64B_ZT0 1
#endif
/* ice_r6 sweep-B prefetch refinements (all bit-identical; node-raced):
 *   ZT0H: hint for the next-body prefetch (3=T0 keeps the line in L1 but each
 *         in-flight T0 miss occupies an L1 fill buffer -- the likely MLP cap
 *         against ~50-cycle L3 loads; 2=T1 fills L2 through the deeper L2
 *         queue and the demand load pays only the ~14-cycle L2 hop).
 *   ZTC:  cross-body coverage -- body g's prefetch used to aim at cb+16,
 *         which for g=7 is the row's PAD line, so the first body of every
 *         ky-slab (64 line-pairs, 1/8 of the sweep) was never covered; the
 *         next-body offset is +16 within the slab but SCKS-112 across the
 *         g=7 -> (ky+1,0) seam.
 *   ZT2:  additionally prefetch TWO bodies ahead (more latency slack at the
 *         cost of 2 extra issue slots per load). */
#ifndef FFT64B_ZT0H
# define FFT64B_ZT0H 3
#endif
#ifndef FFT64B_ZTC
# define FFT64B_ZTC 1
#endif
#ifndef FFT64B_ZT2
# define FFT64B_ZT2 0
#endif
/* ice_r6 sweep-A map-row prefetch: the custody sweep A had NO software
 * prefetch (the r5 rewrite dropped the old pipeline's p1pf); rows of state
 * and c stream at SCKS stride from L3, and the L2 streamer retrains at every
 * 4-KB boundary (~3.7 rows).  ZAPF prefetches the state+c rows ZAD ahead
 * (16 lines each) at hint ZAH while row y's map+z-line computes.  This is
 * the rivals' 1000f989 iter_z64 schedule verbatim (they prefetch x+3, T0,
 * state and c); node race: 652.4/660.0 (on) vs 678.6 (off) us/step-vol at
 * B=2, while T1 hints or distance 2 LOSE (754.2, 699.4) -- on this node a
 * T0 hint into L1 beats T1 in every prefetch site raced so far. */
#ifndef FFT64B_ZAPF
# define FFT64B_ZAPF 1
#endif
#ifndef FFT64B_ZAH
# define FFT64B_ZAH 3
#endif
/* ZAD=2 post-inline (ice_r7): with zs_ztail force-inlined the row body
 * dropped from ~310 to ~230 cyc, and the shorter body wants the shorter
 * lead -- node 709.6/711.1 (d2) vs 718.8/718.8 (d3) vs 727.3/727.5 (d4),
 * same-window, sd 0.05%.  (r6's "distance 2 LOSES" datum was d2-at-T1.) */
#ifndef FFT64B_ZAD
# define FFT64B_ZAD 2
#endif
/* ZAW: continue the ZAPF cursor into plane x+1's first ZAD rows.  Node-raced
 * OFF: 652.8/655.3 (on) vs 637.4/642.7 (off) -- the wrap prefetches drag
 * 192 T0 lines into L1 during rows 61..63, right before the y-line phase
 * re-reads the just-written plane from L1/L2. */
#ifndef FFT64B_ZAW
# define FFT64B_ZAW 0
#endif

/* ice_r7 ZMS: output-side map fusion + cross-step z fusion, adopted from the
 * rival v6_3f30d81f L=64 ("the elementwise map is fused into the x-pass
 * stores, and the next iteration's z-pass is pipelined behind each completed
 * row group").  In custody layout the fusion is legal at ky-slab grain:
 * after sweep B's 8 x-line bodies finish ky-slab k, ALL 64 z-rows of that
 * slab are complete and L1/L2-hot -- so the map and step s+1's z-lines run
 * on the hot slab, and the old sweep A shrinks to just the y-lines.  One
 * chain step becomes:
 *   sweep BZ, per ky-slab: x-lines of step s; map(z+c) lands the NEW state
 *     in place (ZMS=1: fused into the x-line stores; ZMS=2: x-lines store
 *     raw z, then the existing zs_row_map maps AND z-lines each of the
 *     slab's 64 rows -- map fused into the z-row loads, r4 lineage, but now
 *     the state reads are slab-hot); under ZMS=1 the z-lines are a separate
 *     hot-slab zs_row_raw pass.
 *   sweep Y, per x-plane: step s+1's y-lines -- now the only cold strided
 *     pass, and it gets sweep B's proven next-body T0 prefetch (ZYT0).
 * Per-step L3 traffic is unchanged (~22.3 MB) and the values are
 * BIT-IDENTICAL to the r5/r6 lazy-map scheme (same MAP8V on the same
 * numbers, same z/y/x reassociation) -- what moves is WHERE the map+z
 * compute runs: out of the r6-measured 73%/26-GB/s cold sweep and onto
 * L1/L2-hot slabs.  0 = the r5/r6 lazy-map scheme (control). */
#ifndef FFT64B_ZMS
# define FFT64B_ZMS 0
#endif
/* ZMS=1: T0-prefetch the body's own c slot pair at x-line load time (the
 * map load in the same body's store phase lands a few hundred cycles later) */
#ifndef FFT64B_ZMC
# define FFT64B_ZMC 1
#endif
/* ZMS=2: T0-prefetch the c row ZAD rows ahead during the hot-slab map+z
 * rows (state rows are hot; c is the only cold stream in that phase) */
#ifndef FFT64B_ZMCPF
# define FFT64B_ZMCPF 1
#endif
/* sweep-Y next-body prefetch (the sweep-B ZT0 pattern at SCKS stride; the
 * y-lines lost their free L1 residency when they left sweep A) */
#ifndef FFT64B_ZYT0
# define FFT64B_ZYT0 1
#endif
/* ice_r7 ZYW: cover plane x+1's first ZAD rows (state AND c) DURING plane
 * x's y-line phase -- the coverage hole ZAW tried to fix from the wrong
 * side (r6: ZAW's T0 burst during rows 61..63 displaced the plane the
 * y-lines were about to re-read; issuing the same lines during the y-lines
 * is safe because after them the plane is dead).  Body g < ZAD covers next
 * plane's row g + c row g (32 T0 lines per body). */
#ifndef FFT64B_ZYW
# define FFT64B_ZYW 0
#endif
#ifndef FFT64B_ZYH
# define FFT64B_ZYH 3
#endif
/* ice_r7 ZVP: volume-PAIR row interleaving in sweep A (the r6 next-item-3
 * lever; B=2 is the graded case).  The map+z row body is bound by its own
 * dependency chain, not bandwidth; alternating two volumes' rows keeps two
 * independent bodies in the ROB, so one volume's cold-row misses and ladder
 * latency hide under the other's compute.  Chain working set becomes
 * 2 states + 2 custody c = 17.8 MB (< 24 MB L3).  Bit-identical per volume
 * (row order across volumes is immaterial to each volume's arithmetic).
 * Node-raced OFF (ice_r7): 706.1/709.1 (on) vs 611.2/612.3 (off), two clean
 * same-window pairs at sd 0.06% -- a 16% LOSS.  Post-inline the row body is
 * no longer latency-starved enough to pay for quadrupling the row phase's
 * hot footprint (2 planes + 2 c planes = 272 KB vs 48-KB L1) and doubling
 * the chain working set.  Kept env/-D-forcible as a control. */
#ifndef FFT64B_ZVP
# define FFT64B_ZVP 0
#endif

/* common z-line tail: custody slots (Vr,Vi lanes = 8 adjacent z) -> custody
 * slots of the 64-point spectrum, stored to row (slot k1 at +k1*16).
 * ALWAYS_INLINE (ice_r7): gcc 11 declined the plain `inline` at all three
 * call sites, so every z-row call spilled all 16 Vr/Vi vectors to the stack,
 * reloaded them in the callee, and paid a stack-protector canary -- measured
 * on the node as the map+z row phase running at the same ~310 cyc/row hot or
 * cold (the r6 "sweep A is inefficient" mystery in one line of asm). */
static inline __attribute__((always_inline))
void zs_ztail(double *restrict row, v8d *Vr, v8d *Vi)
{
    v8d Ur[8], Ui[8], Tr[8], Ti[8], Wr[8], Wi[8];
    DFT8S(Vr, Vi, Ur, Ui);
    _Pragma("GCC unroll 8")
    for (int k2 = 1; k2 < 8; ++k2)
        CTWV(Ur[k2], Ui[k2],
             *(const v8d *)twzr8[k2], *(const v8d *)twzi8[k2]);
    TR8(Ur, Tr);
    TR8(Ui, Ti);
    DFT8S(Tr, Ti, Wr, Wi);
    _Pragma("GCC unroll 8")
    for (int g = 0; g < 8; ++g) {
        *(v8d *)(row + (size_t)g * 16)     = Wr[g];
        *(v8d *)(row + (size_t)g * 16 + 8) = Wi[g];
    }
}

/* steps 2..m row: row holds RAW z custody, crow the custody c; lazy map then
 * z-FFT, in place -- the map's outputs feed the z-line in registers. */
static void zs_row_map(double *restrict row, const double *restrict crow)
{
    v8d Vr[8], Vi[8];
    _Pragma("GCC unroll 8")
    for (int g = 0; g < 8; ++g) {
        const double *r_ = row  + (size_t)g * 16;
        const double *c_ = crow + (size_t)g * 16;
        v8d wr_ = *(const v8d *)r_       + *(const v8d *)c_;
        v8d wi_ = *(const v8d *)(r_ + 8) + *(const v8d *)(c_ + 8);
        MAP8V(wr_, wi_, Vr[g], Vi[g]);
    }
    zs_ztail(row, Vr, Vi);
}

/* step-1 row: natural interleaved input (64 complex at ip), no map */
static void zs_row_nat(const double *restrict ip, double *restrict row)
{
    v8d Vr[8], Vi[8];
    _Pragma("GCC unroll 8")
    for (int g = 0; g < 8; ++g) {
        v8d A_ = *(const v8d *)(ip + (size_t)g * 16);
        v8d B_ = *(const v8d *)(ip + (size_t)g * 16 + 8);
        Vr[g] = DEIN_RE(A_, B_);
        Vi[g] = DEIN_IM(A_, B_);
    }
    zs_ztail(row, Vr, Vi);
}

/* ZMS=1 row: the state row already holds MAPPED values (map was fused into
 * the x-line stores); plain z-line in place, no map, no c */
static void zs_row_raw(double *restrict row)
{
    v8d Vr[8], Vi[8];
    _Pragma("GCC unroll 8")
    for (int g = 0; g < 8; ++g) {
        Vr[g] = *(const v8d *)(row + (size_t)g * 16);
        Vi[g] = *(const v8d *)(row + (size_t)g * 16 + 8);
    }
    zs_ztail(row, Vr, Vi);
}

/* ZMS=2, final step: map the raw-z row in place WITHOUT a following z-line
 * (there is no step m+1; the buffer must end holding state m) */
static void zs_row_maponly(double *restrict row, const double *restrict crow)
{
    _Pragma("GCC unroll 8")
    for (int g = 0; g < 8; ++g) {
        double *r_ = row + (size_t)g * 16;
        const double *c_ = crow + (size_t)g * 16;
        v8d wr_ = *(const v8d *)r_       + *(const v8d *)c_;
        v8d wi_ = *(const v8d *)(r_ + 8) + *(const v8d *)(c_ + 8);
        v8d rr_, ii_;
        MAP8V(wr_, wi_, rr_, ii_);
        *(v8d *)r_ = rr_; *(v8d *)(r_ + 8) = ii_;
    }
}

/* the 8 y-lines of one x-plane, in place (L1/L2-hot after the row phase) */
static inline void zs_plane_y(double *restrict pb)
{
    for (int g = 0; g < 8; ++g) {
        double *cb = pb + (size_t)g * 16;
#define ZS_LDY2(n, rr, ii) do {                                               \
        const double *q_ = cb + (size_t)(n) * SCKS;                           \
        (rr) = *(const v8d *)q_; (ii) = *(const v8d *)(q_ + 8);               \
    } while (0)
#define ZS_STY2(k, rr, ii) do {                                               \
        double *q_ = cb + (size_t)(k) * SCKS;                                 \
        *(v8d *)q_ = (rr); *(v8d *)(q_ + 8) = (ii);                           \
    } while (0)
        FFT64S(ZS_LDY2, ZS_STY2, SC_NOHOOK);
#undef ZS_LDY2
#undef ZS_STY2
    }
}

/* ice_r7 ZVP sweep A over a volume PAIR (steps 2..m only): per x-plane, the
 * 64 (map + z-line) rows of BOTH volumes alternated, then each plane's
 * y-lines.  See the ZVP knob comment for why. */
static void zs_sweepA_pair(double *restrict Z0, double *restrict Z1,
                           const double *restrict c0,
                           const double *restrict c1)
{
    for (int x = 0; x < 64; ++x) {
        double *p0 = Z0 + (size_t)x * SCXS, *p1 = Z1 + (size_t)x * SCXS;
        const double *q0 = c0 + (size_t)x * SCXS;
        const double *q1 = c1 + (size_t)x * SCXS;
        for (int y = 0; y < 64; ++y) {
#if FFT64B_ZAPF
            if (y + FFT64B_ZAD < 64) {
                size_t od_ = (size_t)(y + FFT64B_ZAD) * SCKS;
                _Pragma("GCC unroll 16")
                for (int t_ = 0; t_ < 16; ++t_) {
                    __builtin_prefetch(p0 + od_ + 8 * t_, 0, FFT64B_ZAH);
                    __builtin_prefetch(q0 + od_ + 8 * t_, 0, FFT64B_ZAH);
                    __builtin_prefetch(p1 + od_ + 8 * t_, 0, FFT64B_ZAH);
                    __builtin_prefetch(q1 + od_ + 8 * t_, 0, FFT64B_ZAH);
                }
            }
#endif
            zs_row_map(p0 + (size_t)y * SCKS, q0 + (size_t)y * SCKS);
            zs_row_map(p1 + (size_t)y * SCKS, q1 + (size_t)y * SCKS);
        }
        zs_plane_y(p0);
        zs_plane_y(p1);
    }
}

/* sweep A over one volume: per x-plane, 64 rows (map or natural load + z-line)
 * then, if doy, the 8 y-lines in place.  x0 non-NULL = step 1 (natural input,
 * no map); otherwise rows are raw z mapped against cZ.  doy=0 is the ice_r6
 * ZS2 split (y-lines move to sweep B2's g-slabs). */
static void zs_sweepA(double *restrict ZB, const double *restrict cZ,
                      const double *restrict x0, int doy)
{
    for (int x = 0; x < 64; ++x) {
        double *pb = ZB + (size_t)x * SCXS;
        if (x0) {
            const double *ip = x0 + (size_t)x * (2 * LSQ);
            for (int y = 0; y < 64; ++y)
                zs_row_nat(ip + (size_t)y * 128, pb + (size_t)y * SCKS);
        } else {
            const double *cp = cZ + (size_t)x * SCXS;
            for (int y = 0; y < 64; ++y) {
#if FFT64B_ZAPF
                /* ZAW: wrap the +ZAD row target into plane x+1 (its first
                 * ZAD rows were otherwise never covered) */
                size_t od_ = (y + FFT64B_ZAD < 64)
                               ? (size_t)(y + FFT64B_ZAD) * SCKS
                               : (FFT64B_ZAW && x < 63)
                                   ? (size_t)SCXS
                                     + (size_t)(y + FFT64B_ZAD - 64) * SCKS
                                   : (size_t)0;
                if (od_) {
                    const double *nr_ = pb + od_;
                    const double *nc_ = cp + od_;
                    _Pragma("GCC unroll 16")
                    for (int q_ = 0; q_ < 16; ++q_) {
                        __builtin_prefetch(nr_ + 8 * q_, 0, FFT64B_ZAH);
                        __builtin_prefetch(nc_ + 8 * q_, 0, FFT64B_ZAH);
                    }
                }
#endif
                zs_row_map(pb + (size_t)y * SCKS, cp + (size_t)y * SCKS);
            }
        }
        if (!doy) continue;
        for (int g = 0; g < 8; ++g) {
            double *cb = pb + (size_t)g * 16;
#if FFT64B_ZYW
            if (cZ && x < 63 && g < FFT64B_ZAD) {
                const double *wr_ = pb + SCXS + (size_t)g * SCKS;
                const double *wc_ = cZ + (size_t)(x + 1) * SCXS
                                       + (size_t)g * SCKS;
                _Pragma("GCC unroll 16")
                for (int q_ = 0; q_ < 16; ++q_) {
                    __builtin_prefetch(wr_ + 8 * q_, 0, FFT64B_ZYH);
                    __builtin_prefetch(wc_ + 8 * q_, 0, FFT64B_ZYH);
                }
            }
#endif
#define ZS_LDY(n, rr, ii) do {                                                \
            const double *q_ = cb + (size_t)(n) * SCKS;                       \
            (rr) = *(const v8d *)q_; (ii) = *(const v8d *)(q_ + 8);           \
        } while (0)
#define ZS_STY(k, rr, ii) do {                                                \
            double *q_ = cb + (size_t)(k) * SCKS;                             \
            *(v8d *)q_ = (rr); *(v8d *)(q_ + 8) = (ii);                       \
        } while (0)
            FFT64S(ZS_LDY, ZS_STY, SC_NOHOOK);
#undef ZS_LDY
#undef ZS_STY
        }
    }
}

/* sweep B: the x-lines, in place, per ky-slab per g.  T0 prefetch covers the
 * next slot pair; the SHOOK hook T-prefetches the NEXT ky-slab's rows for
 * exactly the 8 planes the current stage-1 group reads (slabpf lineage). */
static void zs_sweepB(double *restrict ZB)
{
    for (int ky = 0; ky < 64; ++ky) {
        double *kb = ZB + (size_t)ky * SCKS;
        for (int g = 0; g < 8; ++g) {
            double *cb = kb + (size_t)g * 16;
            /* next BODY's line pair: +16 within the slab; across the
             * g=7 -> (ky+1, g=0) seam the old +16 hit the row's pad line,
             * leaving 1/8 of the sweep uncovered (ZTC=0 restores that). */
            const size_t nd_ = (FFT64B_ZTC && g == 7) ? (size_t)(SCKS - 112)
                                                      : (size_t)16;
#define ZS_HOOKX(s) do {                                                      \
            if (FFT64B_ZSLAB && ky < 63) {                                    \
                _Pragma("GCC unroll 8")                                       \
                for (int t_ = 0; t_ < 8; ++t_) {                              \
                    const double *q_ =                                        \
                        cb + (size_t)((s) + 8 * t_) * SCXS + SCKS;            \
                    __builtin_prefetch(q_,     0, FFT64B_HS);                 \
                    __builtin_prefetch(q_ + 8, 0, FFT64B_HS);                 \
                }                                                             \
            }                                                                 \
        } while (0)
#define ZS_LDX(n, rr, ii) do {                                                \
            const double *q_ = cb + (size_t)(n) * SCXS;                       \
            if (FFT64B_ZT0) {                                                 \
                __builtin_prefetch(q_ + nd_,     0, FFT64B_ZT0H);             \
                __builtin_prefetch(q_ + nd_ + 8, 0, FFT64B_ZT0H);             \
            }                                                                 \
            if (FFT64B_ZT2) {                                                 \
                __builtin_prefetch(q_ + 32, 0, FFT64B_ZT0H);                  \
                __builtin_prefetch(q_ + 40, 0, FFT64B_ZT0H);                  \
            }                                                                 \
            (rr) = *(const v8d *)q_; (ii) = *(const v8d *)(q_ + 8);           \
        } while (0)
#define ZS_STX(k, rr, ii) do {                                                \
            double *q_ = cb + (size_t)(k) * SCXS;                             \
            *(v8d *)q_ = (rr); *(v8d *)(q_ + 8) = (ii);                       \
        } while (0)
            FFT64S(ZS_LDX, ZS_STX, ZS_HOOKX);
#undef ZS_LDX
#undef ZS_STX
#undef ZS_HOOKX
        }
    }
}

/* ice_r6 ZS2 sweep B2: the y-lines AND x-lines, per g-slab (fixed slot-group
 * g = 8 adjacent z), moved here from sweep A / sweep B.  Rationale from the
 * FFT64B_PROF attribution (this round): sweep A was 73% of the step at only
 * ~26 GB/s effective while sweep B ran ~49 GB/s with compute to spare --
 * the y-lines' 526K vector ops were serialized behind sweep A's map+z rows
 * instead of hiding under sweep B's L3 traffic.  A g-slab (the 128 B touched
 * per 1088-B row over all (x,y): 512 KB) fits L2, so the y-pass pulls it in
 * once and the x-pass runs on L2 hits.  Same total L3 traffic, bit-identical
 * results (same FFT64S bodies, same z->y->x order, no cross-body deps). */
static void zs_sweepB2(double *restrict ZB)
{
    for (int g = 0; g < 8; ++g) {
        double *gb = ZB + (size_t)g * 16;
        for (int x = 0; x < 64; ++x) {          /* y-lines: body (x, g) */
            double *cb = gb + (size_t)x * SCXS;
            const int pfn_ = (x < 63);
#define ZS2_LDY(n, rr, ii) do {                                               \
            const double *q_ = cb + (size_t)(n) * SCKS;                       \
            if (FFT64B_ZT0 && pfn_) {           /* next body = next plane */  \
                __builtin_prefetch(q_ + SCXS,     0, 3);                      \
                __builtin_prefetch(q_ + SCXS + 8, 0, 3);                      \
            }                                                                 \
            (rr) = *(const v8d *)q_; (ii) = *(const v8d *)(q_ + 8);           \
        } while (0)
#define ZS2_STY(k, rr, ii) do {                                               \
            double *q_ = cb + (size_t)(k) * SCKS;                             \
            *(v8d *)q_ = (rr); *(v8d *)(q_ + 8) = (ii);                       \
        } while (0)
            FFT64S(ZS2_LDY, ZS2_STY, SC_NOHOOK);
#undef ZS2_LDY
#undef ZS2_STY
        }
        for (int ky = 0; ky < 64; ++ky) {       /* x-lines: body (ky, g),
                                                 * slab now L2-resident */
            double *cb = gb + (size_t)ky * SCKS;
#define ZS2_LDX(n, rr, ii) do {                                               \
            const double *q_ = cb + (size_t)(n) * SCXS;                       \
            (rr) = *(const v8d *)q_; (ii) = *(const v8d *)(q_ + 8);           \
        } while (0)
#define ZS2_STX(k, rr, ii) do {                                               \
            double *q_ = cb + (size_t)(k) * SCXS;                             \
            *(v8d *)q_ = (rr); *(v8d *)(q_ + 8) = (ii);                       \
        } while (0)
            FFT64S(ZS2_LDX, ZS2_STX, SC_NOHOOK);
#undef ZS2_LDX
#undef ZS2_STX
        }
    }
}

/* ice_r7 fused sweep BZ (see the ZMS knob comment): per ky-slab, the 8
 * x-line bodies of step s (v1: map(z+c) fused into the stores; else plain,
 * the r5/r6-proven body verbatim), then -- on the now-L1/L2-hot 68-KB slab --
 * the map (v1: already done) and step s+1's z-lines for the slab's 64 rows.
 * doz=0 is the final step: map lands state m, no z-lines follow. */
/* ice_r6: env-gated per-sweep cycle attribution (FFT64B_PROF=1, set at
 * create).  The r5 record's "sweep B is the latency-exposed one" was an
 * estimate; this prints measured cycles so schedule work aims at the real
 * wall.  The uninstrumented loop below is the shipped path, untouched. */
static int g_zsprof;
/* ZS2 sweep split (doy=0 sweep A + g-slab sweep B2); -D default, env
 * FFT64B_ZS2 overrides at create.  Both splits are bit-identical, so a
 * flip can never produce an unvalidated number.  Node-raced OFF (ice_r6):
 * build-level pairs 668.9/699.3 (zs2) vs 649.6/656.6 (zs0), same-window
 * env-alternated medians 2/3 pairs to zs0 -- the rebalance loses ~3%. */
#ifndef FFT64B_ZS2
# define FFT64B_ZS2 0
#endif
static int g_zs2 = FFT64B_ZS2;
static int g_zms = FFT64B_ZMS;   /* 0 lazy-map (r6) / 1 map-in-stores /
                                  * 2 map-in-z-row-loads on the hot slab */
static inline unsigned long long zs_tsc(void)
{
#ifdef __x86_64__
    unsigned int lo_, hi_;
    __asm__ __volatile__("rdtsc" : "=a"(lo_), "=d"(hi_));
    return ((unsigned long long)hi_ << 32) | lo_;
#else
    return 0;
#endif
}

static unsigned long long g_bzx, g_bzz;   /* PROF: BZ x-line vs z+map cycles */
static void zs_sweepBZ(double *restrict ZB, const double *restrict cZ,
                       int v1, int doz)
{
    for (int ky = 0; ky < 64; ++ky) {
        double *kb = ZB + (size_t)ky * SCKS;
        const double *ck = cZ + (size_t)ky * SCXS;   /* SLAB-MAJOR c */
        unsigned long long tx_ = 0;
        if (__builtin_expect(g_zsprof, 0)) tx_ = zs_tsc();
        for (int g = 0; g < 8; ++g) {
            double *cb = kb + (size_t)g * 16;
            const double *cc = ck + (size_t)g * 16;
            const size_t nd_ = (FFT64B_ZTC && g == 7) ? (size_t)(SCKS - 112)
                                                      : (size_t)16;
            if (v1) {
#define ZM_LDX(n, rr, ii) do {                                                \
                const double *q_ = cb + (size_t)(n) * SCXS;                   \
                if (FFT64B_ZT0) {                                             \
                    __builtin_prefetch(q_ + nd_,     0, FFT64B_ZT0H);         \
                    __builtin_prefetch(q_ + nd_ + 8, 0, FFT64B_ZT0H);         \
                }                                                             \
                if (FFT64B_ZMC) {                                             \
                    const double *qc_ = cc + (size_t)(n) * SCKS;              \
                    __builtin_prefetch(qc_,     0, 3);                        \
                    __builtin_prefetch(qc_ + 8, 0, 3);                        \
                }                                                             \
                (rr) = *(const v8d *)q_; (ii) = *(const v8d *)(q_ + 8);       \
            } while (0)
#define ZM_STX(k, rr, ii) do {                                                \
                double *q_ = cb + (size_t)(k) * SCXS;                         \
                const double *qc_ = cc + (size_t)(k) * SCKS;                  \
                v8d wr_ = (rr) + *(const v8d *)qc_;                           \
                v8d wi_ = (ii) + *(const v8d *)(qc_ + 8);                     \
                v8d mr_, mi_;                                                 \
                MAP8V(wr_, wi_, mr_, mi_);                                    \
                *(v8d *)q_ = mr_; *(v8d *)(q_ + 8) = mi_;                     \
            } while (0)
                FFT64S(ZM_LDX, ZM_STX, SC_NOHOOK);
#undef ZM_LDX
#undef ZM_STX
            } else {
#define ZM_LDX2(n, rr, ii) do {                                               \
                const double *q_ = cb + (size_t)(n) * SCXS;                   \
                if (FFT64B_ZT0) {                                             \
                    __builtin_prefetch(q_ + nd_,     0, FFT64B_ZT0H);         \
                    __builtin_prefetch(q_ + nd_ + 8, 0, FFT64B_ZT0H);         \
                }                                                             \
                (rr) = *(const v8d *)q_; (ii) = *(const v8d *)(q_ + 8);       \
            } while (0)
#define ZM_STX2(k, rr, ii) do {                                               \
                double *q_ = cb + (size_t)(k) * SCXS;                         \
                *(v8d *)q_ = (rr); *(v8d *)(q_ + 8) = (ii);                   \
            } while (0)
                FFT64S(ZM_LDX2, ZM_STX2, SC_NOHOOK);
#undef ZM_LDX2
#undef ZM_STX2
            }
        }
        if (__builtin_expect(g_zsprof, 0)) {
            unsigned long long t_ = zs_tsc(); g_bzx += t_ - tx_; tx_ = t_;
        }
        if (v1) {
            if (doz)
                for (int x = 0; x < 64; ++x)
                    zs_row_raw(kb + (size_t)x * SCXS);
        } else {
            for (int x = 0; x < 64; ++x) {
#if FFT64B_ZMCPF
                if (x + FFT64B_ZAD < 64) {
                    const double *nc_ = ck + (size_t)(x + FFT64B_ZAD) * SCKS;
                    _Pragma("GCC unroll 16")
                    for (int q_ = 0; q_ < 16; ++q_)
                        __builtin_prefetch(nc_ + 8 * q_, 0, FFT64B_ZAH);
                }
#endif
                if (doz) zs_row_map(kb + (size_t)x * SCXS,
                                    ck + (size_t)x * SCKS);
                else     zs_row_maponly(kb + (size_t)x * SCXS,
                                        ck + (size_t)x * SCKS);
            }
        }
        if (__builtin_expect(g_zsprof, 0)) g_bzz += zs_tsc() - tx_;
    }
}

/* ice_r7 sweep Y: the y-lines alone, per x-plane -- what is left of sweep A
 * once map+z move into sweep BZ.  Now a cold strided volume pass, so it gets
 * the next-body T0 prefetch that made sweep B run at ~49 GB/s: body (x,g)
 * covers body (x,g+1)'s line pair (+16 within the plane; across the g=7 ->
 * (x+1,0) seam the offset is SCXS-112). */
static void zs_sweepY(double *restrict ZB)
{
    for (int x = 0; x < 64; ++x) {
        double *pb = ZB + (size_t)x * SCXS;
        for (int g = 0; g < 8; ++g) {
            double *cb = pb + (size_t)g * 16;
            const size_t nd_ = (g == 7) ? (size_t)(SCXS - 112) : (size_t)16;
            const int pf_ = FFT64B_ZYT0 && (g < 7 || x < 63);
#define ZM_LDY(n, rr, ii) do {                                                \
            const double *q_ = cb + (size_t)(n) * SCKS;                       \
            if (pf_) {                                                        \
                __builtin_prefetch(q_ + nd_,     0, FFT64B_ZT0H);             \
                __builtin_prefetch(q_ + nd_ + 8, 0, FFT64B_ZT0H);             \
            }                                                                 \
            (rr) = *(const v8d *)q_; (ii) = *(const v8d *)(q_ + 8);           \
        } while (0)
#define ZM_STY(k, rr, ii) do {                                                \
            double *q_ = cb + (size_t)(k) * SCKS;                             \
            *(v8d *)q_ = (rr); *(v8d *)(q_ + 8) = (ii);                       \
        } while (0)
            FFT64S(ZM_LDY, ZM_STY, SC_NOHOOK);
#undef ZM_LDY
#undef ZM_STY
        }
    }
}

/* ZMS epilogue: the buffer already holds MAPPED state m; pure interleave to
 * the driver's natural layout (no map, no c) */
static void zs_epilogue_ilv(const double *restrict ZB, double *restrict fo)
{
    for (int x = 0; x < 64; ++x)
        for (int y = 0; y < 64; ++y) {
            const double *row = ZB + (size_t)x * SCXS + (size_t)y * SCKS;
            double *op = fo + (size_t)x * (2 * LSQ) + (size_t)y * 128;
            _Pragma("GCC unroll 8")
            for (int g = 0; g < 8; ++g) {
                v8d rr_ = *(const v8d *)(row + (size_t)g * 16);
                v8d ii_ = *(const v8d *)(row + (size_t)g * 16 + 8);
                *(v8d *)(op + (size_t)g * 16)     = ILV_LO(rr_, ii_);
                *(v8d *)(op + (size_t)g * 16 + 8) = ILV_HI(rr_, ii_);
            }
        }
}

/* repermute one volume of c (driver natural interleaved) into custody form:
 * the map is pointwise in the FFT-output index, and custody indexes k the
 * same way it indexes z, so this is a straight DEIN into the padded strides.
 * Once per volume per chain (1/m of the c traffic is the rebuild). */
static void zs_build_c(const double *restrict c, double *restrict cZ)
{
    for (int x = 0; x < 64; ++x)
        for (int y = 0; y < 64; ++y) {
            const double *ip = c + (size_t)x * (2 * LSQ) + (size_t)y * 128;
            double *row = cZ + (size_t)x * SCXS + (size_t)y * SCKS;
            _Pragma("GCC unroll 8")
            for (int g = 0; g < 8; ++g) {
                v8d A_ = *(const v8d *)(ip + (size_t)g * 16);
                v8d B_ = *(const v8d *)(ip + (size_t)g * 16 + 8);
                *(v8d *)(row + (size_t)g * 16)     = DEIN_RE(A_, B_);
                *(v8d *)(row + (size_t)g * 16 + 8) = DEIN_IM(A_, B_);
            }
        }
}

/* ice_r7 ZMS form: SLAB-MAJOR custody c -- row (x, ky) lives at
 * cZ + ky*SCXS + x*SCKS (the two stride roles swapped), so sweep BZ's map
 * phase reads c as ONE sequential 68-KB run per ky-slab.  The first ZMS
 * attempt consumed plane-major c per slab = 64 scattered 1-KB rows at 70-KB
 * stride; the node priced that pattern at ~6 GB/s effective (bz-zmap 1.32M
 * cyc/step-vol) and it ate the entire fusion gain. */
static void zs_build_c_slab(const double *restrict c, double *restrict cZ)
{
    for (int x = 0; x < 64; ++x)
        for (int y = 0; y < 64; ++y) {
            const double *ip = c + (size_t)x * (2 * LSQ) + (size_t)y * 128;
            double *row = cZ + (size_t)y * SCXS + (size_t)x * SCKS;
            _Pragma("GCC unroll 8")
            for (int g = 0; g < 8; ++g) {
                v8d A_ = *(const v8d *)(ip + (size_t)g * 16);
                v8d B_ = *(const v8d *)(ip + (size_t)g * 16 + 8);
                *(v8d *)(row + (size_t)g * 16)     = DEIN_RE(A_, B_);
                *(v8d *)(row + (size_t)g * 16 + 8) = DEIN_IM(A_, B_);
            }
        }
}

/* epilogue: ZB holds raw z of step m; map against custody c and ILV-store the
 * final state to the driver's natural interleaved final_out. */
static void zs_epilogue(const double *restrict ZB, const double *restrict cZ,
                        double *restrict fo)
{
    for (int x = 0; x < 64; ++x)
        for (int y = 0; y < 64; ++y) {
            const double *row  = ZB + (size_t)x * SCXS + (size_t)y * SCKS;
            const double *crow = cZ + (size_t)x * SCXS + (size_t)y * SCKS;
            double *op = fo + (size_t)x * (2 * LSQ) + (size_t)y * 128;
            _Pragma("GCC unroll 8")
            for (int g = 0; g < 8; ++g) {
                const double *r_ = row  + (size_t)g * 16;
                const double *c_ = crow + (size_t)g * 16;
                v8d wr_ = *(const v8d *)r_       + *(const v8d *)c_;
                v8d wi_ = *(const v8d *)(r_ + 8) + *(const v8d *)(c_ + 8);
                v8d rr_, ii_;
                MAP8V(wr_, wi_, rr_, ii_);
                *(v8d *)(op + (size_t)g * 16)     = ILV_LO(rr_, ii_);
                *(v8d *)(op + (size_t)g * 16 + 8) = ILV_HI(rr_, ii_);
            }
        }
}


/* the whole graded chain, custody-resident, ONE VOLUME AT A TIME through all
 * m steps (per-volume cache residency, corpus 10 3) -- or, under ZVP with a
 * pair arena, one volume PAIR at a time with sweep-A rows interleaved. */
static int g_zvp = FFT64B_ZVP;
static void chain_zs(double *restrict ZB, double *restrict cZ,
                     double *restrict ZB1, double *restrict cZ1,
                     const double *restrict x0, const double *restrict cf,
                     double *restrict fo, int m, int nv)
{
    const int z2 = g_zs2, doy = !z2, zms = g_zms, v1 = (zms == 1);
    const int zvp = g_zvp && !zms && !z2 && ZB1 != NULL;
    if (zvp && !__builtin_expect(g_zsprof, 0)) {
        int b = 0;
        for (; b + 1 < nv; b += 2) {          /* lockstep volume pairs */
            const double *xv0 = x0 + (size_t)b * VDBL, *xv1 = xv0 + VDBL;
            const double *cv0 = cf + (size_t)b * VDBL, *cv1 = cv0 + VDBL;
            double *ov0 = fo + (size_t)b * VDBL, *ov1 = ov0 + VDBL;
            zs_build_c(cv0, cZ);  zs_build_c(cv1, cZ1);
            zs_sweepA(ZB,  NULL, xv0, 1); zs_sweepB(ZB);   /* step 1 */
            zs_sweepA(ZB1, NULL, xv1, 1); zs_sweepB(ZB1);
            for (int s = 2; s <= m; ++s) {
                zs_sweepA_pair(ZB, ZB1, cZ, cZ1);
                zs_sweepB(ZB);
                zs_sweepB(ZB1);
            }
            zs_epilogue(ZB,  cZ,  ov0);
            zs_epilogue(ZB1, cZ1, ov1);
        }
        for (; b < nv; ++b) {                 /* odd remainder: single path */
            const double *xv = x0 + (size_t)b * VDBL;
            const double *cv = cf + (size_t)b * VDBL;
            double       *ov = fo + (size_t)b * VDBL;
            zs_build_c(cv, cZ);
            zs_sweepA(ZB, NULL, xv, 1);
            zs_sweepB(ZB);
            for (int s = 2; s <= m; ++s) {
                zs_sweepA(ZB, cZ, NULL, 1);
                zs_sweepB(ZB);
            }
            zs_epilogue(ZB, cZ, ov);
        }
        return;
    }
    if (__builtin_expect(g_zsprof, 0)) {
        unsigned long long tc = 0, ta = 0, tb = 0, te = 0, t0, t1;
        for (int b = 0; b < nv; ++b) {
            const double *xv = x0 + (size_t)b * VDBL;
            const double *cv = cf + (size_t)b * VDBL;
            double       *ov = fo + (size_t)b * VDBL;
            t0 = zs_tsc();
            if (zms) zs_build_c_slab(cv, cZ); else zs_build_c(cv, cZ);
            t1 = zs_tsc(); tc += t1 - t0;
            if (zms) {                    /* ta = fused BZ, tb = sweep Y */
                zs_sweepA(ZB, NULL, xv, 1);
                t0 = zs_tsc(); te += t0 - t1;   /* step-1 z+y under epi slot */
                for (int s = 1; s < m; ++s) {
                    t0 = zs_tsc(); zs_sweepBZ(ZB, cZ, v1, 1);
                    t1 = zs_tsc(); ta += t1 - t0;
                    zs_sweepY(ZB);
                    t0 = zs_tsc(); tb += t0 - t1;
                }
                t0 = zs_tsc(); zs_sweepBZ(ZB, cZ, v1, 0);
                t1 = zs_tsc(); ta += t1 - t0;
                zs_epilogue_ilv(ZB, ov);
                t0 = zs_tsc(); te += t0 - t1;
                continue;
            }
            zs_sweepA(ZB, NULL, xv, doy);
            t0 = zs_tsc(); ta += t0 - t1;
            if (z2) zs_sweepB2(ZB); else zs_sweepB(ZB);
            t1 = zs_tsc(); tb += t1 - t0;
            for (int s = 2; s <= m; ++s) {
                t0 = zs_tsc(); zs_sweepA(ZB, cZ, NULL, doy);
                t1 = zs_tsc(); ta += t1 - t0;
                if (z2) zs_sweepB2(ZB); else zs_sweepB(ZB);
                t0 = zs_tsc(); tb += t0 - t1;
            }
            t0 = zs_tsc(); zs_epilogue(ZB, cZ, ov);
            t1 = zs_tsc(); te += t1 - t0;
        }
        double sv = (double)m * (double)nv;
        fprintf(stderr, "L64B prof zms=%d m=%d nv=%d cyc/step-vol: %s=%.0f "
                "%s=%.0f (buildc=%.0f other=%.0f /chain-vol) total/step=%.0f "
                "[bz-x=%.0f bz-zmap=%.0f]\n",
                zms, m, nv, zms ? "sweepBZ" : "sweepA", (double)ta / sv,
                zms ? "sweepY" : "sweepB", (double)tb / sv,
                (double)tc / nv, (double)te / nv,
                ((double)(ta + tb + tc + te)) / sv,
                (double)g_bzx / sv, (double)g_bzz / sv);
        g_bzx = g_bzz = 0;
        return;
    }
    for (int b = 0; b < nv; ++b) {
        const double *xv = x0 + (size_t)b * VDBL;
        const double *cv = cf + (size_t)b * VDBL;
        double       *ov = fo + (size_t)b * VDBL;
        if (zms) {
            zs_build_c_slab(cv, cZ);      /* slab-major: sequential per slab */
            zs_sweepA(ZB, NULL, xv, 1);   /* step 1: z+y of x0, no map */
            for (int s = 1; s < m; ++s) {
                zs_sweepBZ(ZB, cZ, v1, 1);   /* x_s + map -> state_{s+1},
                                              * + z_{s+1} on the hot slabs */
                zs_sweepY(ZB);               /* y_{s+1} */
            }
            zs_sweepBZ(ZB, cZ, v1, 0);    /* x_m + map = final state */
            zs_epilogue_ilv(ZB, ov);      /* pure interleave */
            continue;
        }
        zs_build_c(cv, cZ);
        zs_sweepA(ZB, NULL, xv, doy);     /* step 1: raw FFT of x0 */
        if (z2) zs_sweepB2(ZB); else zs_sweepB(ZB);
        for (int s = 2; s <= m; ++s) {    /* steps 2..m: map + FFT in place */
            zs_sweepA(ZB, cZ, NULL, doy);
            if (z2) zs_sweepB2(ZB); else zs_sweepB(ZB);
        }
        zs_epilogue(ZB, cZ, ov);          /* state m -> natural interleaved */
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
    int     sb;                  /* st=3 only: pass-2 x-FFT writes the 68-KB
                                  * SB slab instead of in-place SC (kills
                                  * the dead SC writeback), A/B decided     */
    int     chain_ok;            /* fused fft3d_chain verified vs the exact
                                  * scalar chain on the st=0 kernel at
                                  * create time; 0 = fall back            */
    double *S;                   /* padded scratch volume, reused per batch */
    void   *rawS;
    size_t  map_bytes;           /* nonzero: rawS is an mmap of this size   */
    double *ping;                /* fft3d_chain's second ping-pong buffer   */
    void   *rawP;
    size_t  ping_bytes;          /* nonzero: rawP is an mmap of this size   */
    double *zsb;                 /* ice_r5 custody chain: state volume      */
    double *zsc;                 /* custody form of c (skewed off zsb)      */
    double *zsb1;                /* ice_r7 ZVP pair arena: second state     */
    double *zsc1;                /* second custody c (NULL at batch < 2)    */
    void   *rawZ;
    size_t  zs_bytes;            /* nonzero: rawZ is an mmap of this size   */
    int     zs_ok;               /* custody chain verified vs exact chain   */
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

static void run_vols(int pw, int mode, int pf, int st, int pro, int sb,
                     double *S, const double *in, double *out, int nvol)
{
#ifdef HAVE_PW4
    if (st == 3)      run_vols_sc(in, out, S, nvol, mode, pf, pro, sb, NULL, 0);
    else if (pw == 4) run_vols_pw4(in, out, S, nvol, mode, pf, st);
    else
#endif
    { (void)pw; (void)pro; (void)sb; run_vols_pw2(in, out, S, nvol, mode, pf, st); }
#if defined(__SSE2__)
    if (mode == M_NT) _mm_sfence();
#endif
}

/* exact scalar map (one sqrt + one divide per point, the driver/numpy
 * semantics verbatim): the create-time interlock's independent reference,
 * and the emergency path when the fused chain fails verification. */
static void map_rows_exact(const double *restrict z, const double *restrict c,
                           double *restrict dst, size_t ndbl)
{
    for (size_t i = 0; i < ndbl; i += 2) {
        double re = z[i] + c[i], im = z[i + 1] + c[i + 1];
        double s  = 1.0 / (1.0 + sqrt(re * re + im * im));
        dst[i] = re * s; dst[i + 1] = im * s;
    }
}

#ifdef HAVE_PW4
/* the whole m-step graded chain, fused: buffers stay RAW z between steps
 * (map applied inside the next step's pass 1), one streaming map epilogue.
 * Step s writes fo when (m-s) is even, so step m always lands in fo and
 * the epilogue maps it in place.  x0 is never written. */
static void chain_fused_sc(int mode, int pf, int pro, int sb, double *S,
                           const double *x0, const double *cf,
                           double *fo, double *other, int m, int nv)
{
    for (int s = 1; s <= m; ++s) {
        double *dst = ((m - s) & 1) ? other : fo;
        const double *src = (s == 1) ? x0
                          : (((m - s + 1) & 1) ? other : fo);
        run_vols_sc(src, dst, S, nv, mode, pf, pro, sb,
                    (s == 1) ? NULL : cf, 1);
#if defined(__SSE2__)
        if (mode == M_NT) _mm_sfence();
#endif
    }
    map_rows_v(fo, cf, fo, (size_t)nv * VDBL, 1);   /* fo is split raw z */
}
#endif

/* OPTIONAL ABI (ice_r4): own the whole graded chain.  Detected by the
 * driver as a weak symbol; without it we are timed through the driver's
 * unfused execute+map fallback (the 2.24 s configuration the brief calls
 * the battleground). */
void fft3d_chain(fft3d_plan *p, const double _Complex *x0,
                 const double _Complex *c, double _Complex *final_out, int m)
{
    const double *xc = (const double *)x0;
    const double *cf = (const double *)c;
    double *fo = (double *)final_out;
    const int nv = p->batch;
    if (m < 1) return;
    if (!p->ping) {          /* create-time alloc failed; one-shot retry */
        void *r = NULL;
        if (posix_memalign(&r, 64, (size_t)nv * VDBL * sizeof(double)) == 0) {
            p->rawP = r; p->ping_bytes = 0; p->ping = (double *)r;
        } else return;       /* unreachable in practice; fail loudly (stale
                              * final_out cannot pass the chain gate) */
    }
#ifdef HAVE_PW4
    if (p->st == 3 && p->zs_ok) {         /* ice_r5 custody chain (default) */
        chain_zs(p->zsb, p->zsc, p->zsb1, p->zsc1, xc, cf, fo, m, nv);
        return;
    }
    if (p->st == 3 && p->chain_ok) {      /* r4 lazy-map pipeline (control) */
        chain_fused_sc(p->mode, p->pf, p->pro, p->sb, p->S,
                       xc, cf, fo, p->ping, m, nv);
        return;
    }
#endif
    /* exact fallback (env-forced st!=3, or fused verification failed):
     * execute + exact scalar map, state materialized every step */
    for (int s = 1; s <= m; ++s) {
        double *dst = ((m - s) & 1) ? p->ping : fo;
        const double *src = (s == 1) ? xc
                          : (((m - s + 1) & 1) ? p->ping : fo);
        run_vols(p->pw, p->mode, p->pf, p->st, p->pro, p->sb, p->S, src, dst, nv);
        map_rows_exact(dst, cf, dst, (size_t)nv * VDBL);
    }
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
    /* fft3d_chain's second ping-pong buffer (raw z between steps), hugepage-
     * backed like the scratch: pass 2+3 stores rows into it at 512-KB plane
     * stride, which otherwise walks 1024 4-KB pages per volume. */
    {
        const size_t hp = (size_t)2 << 20;
        size_t bytes = (size_t)batch * VDBL * sizeof(double);
        size_t mb = ((bytes + hp - 1) & ~(hp - 1)) + hp;
        void *m = mmap(NULL, mb, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m != MAP_FAILED) {
            uintptr_t a = ((uintptr_t)m + hp - 1) & ~(uintptr_t)(hp - 1);
            p->rawP = m; p->ping_bytes = mb; p->ping = (double *)a;
#if defined(MADV_HUGEPAGE) && !defined(FFT64B_NOHP)
            madvise(m, mb, MADV_HUGEPAGE);
#endif
            memset(p->ping, 0, bytes);
        } else if (posix_memalign(&p->rawP, 64, bytes) == 0) {
            p->ping_bytes = 0; p->ping = (double *)p->rawP;
            memset(p->ping, 0, bytes);
        } else { p->rawP = NULL; p->ping = NULL; p->ping_bytes = 0; }
    }
    /* ice_r5 custody arena: state volume + custody c, hugepage-backed.  cZ is
     * skewed +136 doubles (17 lines, odd) off the state volume because ZS_VOL
     * is an exact multiple of 4 KB -- unskewed, every simultaneous (z row,
     * c row) load pair in sweep A would share L1/L2 page offsets. */
#ifdef HAVE_PW4
    {
        const size_t hp = (size_t)2 << 20;
        /* ZVP (ice_r7): at batch >= 2 size the arena for a volume PAIR
         * (2 states + 2 custody c, 17.8 MB).  Each buffer keeps a distinct
         * 4-KB page offset (ZS_VOL is 4-KB-exact; the 136-double skews make
         * the offsets 0 / 1088 / 2176 / 3264 B). */
        const int pair = (batch >= 2);
        size_t ndbl = pair ? (4 * ZS_VOL + 3 * 136 + 64)
                           : (2 * ZS_VOL + 256);
        size_t bytes = ndbl * sizeof(double);
        size_t mb = ((bytes + hp - 1) & ~(hp - 1)) + hp;
        void *m = mmap(NULL, mb, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (m != MAP_FAILED) {
            uintptr_t a = ((uintptr_t)m + hp - 1) & ~(uintptr_t)(hp - 1);
            p->rawZ = m; p->zs_bytes = mb; p->zsb = (double *)a;
#if defined(MADV_HUGEPAGE) && !defined(FFT64B_NOHP)
            madvise(m, mb, MADV_HUGEPAGE);
#endif
        } else if (posix_memalign(&p->rawZ, 64, bytes) == 0) {
            p->zs_bytes = 0; p->zsb = (double *)p->rawZ;
        } else { p->rawZ = NULL; p->zsb = NULL; }
        if (p->zsb) {
            memset(p->zsb, 0, bytes);
            p->zsc = p->zsb + ZS_VOL + 136;
            if (pair) {
                p->zsb1 = p->zsb + 2 * ZS_VOL + 2 * 136;
                p->zsc1 = p->zsb + 3 * ZS_VOL + 3 * 136;
            }
        }
    }
#endif

    /* ice_r5: the scored number is fft3d_chain, which the custody path owns;
     * fft3d_execute serves the single-transform correctness gate and the
     * emergency fallback only.  The r2-r4 chain-arena tournament tuned a
     * pipeline the custody chain replaces, so it is retired: fixed schedule
     * defaults (the node's most-picked r4 combo), env/-D-forcible, one
     * correctness gate, two chain interlocks.  Setup drops from ~1-4 s to
     * well under a second. */
#ifdef HAVE_PW4
    p->pw = 4; p->mode = M_CACHED; p->pf = 8; p->st = 3;
    p->pro = 1; p->sb = 0;
#else
    p->pw = 2; p->mode = M_CACHED; p->pf = 1; p->st = 0;
    p->pro = 0; p->sb = 0;
#endif
    { const char *e;
#ifdef HAVE_PW4
      if ((e = getenv("FFT64B_PROF"))) g_zsprof = (atoi(e) != 0);
      if ((e = getenv("FFT64B_ZS2")))  g_zs2 = (atoi(e) != 0);
      if ((e = getenv("FFT64B_ZMS"))) {
          g_zms = atoi(e);
          if (g_zms < 0 || g_zms > 2) g_zms = FFT64B_ZMS;
      }
      if ((e = getenv("FFT64B_ZVP"))) g_zvp = (atoi(e) != 0);
#endif
      if ((e = getenv("FFT64B_PW")))   p->pw = atoi(e);
      if ((e = getenv("FFT64B_MODE")))
          p->mode = (e[0] >= '0' && e[0] <= '9') ? (atoi(e) != 0)
                                                 : (strcmp(e, "nt") == 0);
      if ((e = getenv("FFT64B_PF")))   p->pf = atoi(e);
      if ((e = getenv("FFT64B_ST")))   p->st = atoi(e);
      if ((e = getenv("FFT64B_PRO")))  p->pro = (atoi(e) != 0);
      if ((e = getenv("FFT64B_SB")))   p->sb = (atoi(e) != 0);
    }
#ifdef FFT64B_FORCE_PW
    p->pw = FFT64B_FORCE_PW;
#endif
#ifdef FFT64B_FORCE_MODE
    p->mode = FFT64B_FORCE_MODE;
#endif
#ifdef FFT64B_FORCE_PF
    p->pf = FFT64B_FORCE_PF;
#endif
#ifdef FFT64B_FORCE_ST
    p->st = FFT64B_FORCE_ST;
#endif
#ifndef HAVE_PW4
    if (p->pw != 2) p->pw = 2;
    if (p->st == 3) p->st = 0;
#endif

    /* correctness arena (also feeds the chain interlocks); 2 volumes is
     * enough to exercise the custody chain's per-volume loop */
    const int nv = batch < 2 ? batch : 2;
    void *ri = NULL, *ro = NULL, *rr = NULL, *rc = NULL;
    if (posix_memalign(&ri, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&ro, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&rr, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&rc, 64, (size_t)nv * VDBL * sizeof(double))) {
        free(ri); free(ro); free(rr); free(rc);
        snprintf(g_desc, sizeof g_desc,
                 "L64 zsplit-custody; gates SKIPPED (arena alloc failed): "
                 "exec pw=%d mode=%s pf=%d st=%d",
                 p->pw, mode_name[p->mode], p->pf, p->st);
        return p;      /* zs_ok/chain_ok stay 0: exact fallback, never wrong */
    }
    double *tin = ri, *tout = ro, *ref = rr, *cf = rc;
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        tin[i] = (double)(long long)(s >> 11) * 0x1p-53;
    }
    for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        cf[i] = 0.1 * (double)(long long)(s >> 11) * 0x1p-53;
    }

    /* single-transform gate: the shipped execute config vs the INDEPENDENT
     * st=0 interleaved kernel.  A failure reverts execute to the reference
     * config itself -- slower, never wrong. */
    int exec_ok = 1;
    {
        const int rpw =
#ifdef HAVE_PW4
            4;
#else
            2;
#endif
        run_vols(rpw, M_CACHED, 0, 0, 0, 0, p->S, tin, ref, nv);
        if (p->st != 0 || p->mode != M_CACHED || p->pf != 0) {
            run_vols(p->pw, p->mode, p->pf, p->st, p->pro, p->sb,
                     p->S, tin, tout, nv);
            double num = 0.0, den = 0.0;
            for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
                double d = tout[i] - ref[i];
                num += d * d; den += ref[i] * ref[i];
            }
            exec_ok = (num <= den * 1e-26);     /* rel L2 < 1e-13 */
            if (!exec_ok) {
                p->pw = rpw; p->mode = M_CACHED; p->pf = 0; p->st = 0;
                p->pro = 0; p->sb = 0;
            }
        }
    }

    /* chain interlocks: 3-step fused chains vs a 3-step EXACT chain built
     * from the independent st=0 kernel + scalar sqrt/divide map.  Gate
     * rel L2 < 1e-13 (exact-class map) / < 2e-12 (the ~4e-13/application
     * cubic ladders; m=134 end drift stays ~8x under the 1.34e-11 budget,
     * measured in the strategy record).  Custody first; the r4 lazy-map
     * pipeline is verified only if custody is unavailable or vetoed
     * (FFT64B_ZS=0).  FFT64B_CHAIN=0 vetoes both (exact-fallback control). */
    p->zs_ok = 0; p->chain_ok = 0;
#ifdef HAVE_PW4
    {
        const char *ec = getenv("FFT64B_CHAIN");
        const char *ez = getenv("FFT64B_ZS");
        int allow_fused = (!ec || atoi(ec) != 0);
        int want_zs = allow_fused && (!ez || atoi(ez) != 0);
        if (allow_fused && p->st == 3) {
            memcpy(ref, tin, (size_t)nv * VDBL * sizeof(double));
            for (int s3 = 0; s3 < 3; ++s3) {
                run_vols(4, M_CACHED, 0, 0, 0, 0, p->S, ref, tout, nv);
                map_rows_exact(tout, cf, ref, (size_t)nv * VDBL);
            }
            const double gate = (FFT64B_MAPDIV == 2) ? 1e-26 : 4e-24;
            if (want_zs && p->zsb) {
                chain_zs(p->zsb, p->zsc, p->zsb1, p->zsc1, tin, cf, tout, 3, nv);
                double num = 0.0, den = 0.0;
                for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
                    double d = tout[i] - ref[i];
                    num += d * d; den += ref[i] * ref[i];
                }
                p->zs_ok = (num <= den * gate);
            }
            if (!p->zs_ok && p->ping) {
                chain_fused_sc(p->mode, p->pf, p->pro, p->sb, p->S,
                               tin, cf, tout, p->ping, 3, nv);
                double num = 0.0, den = 0.0;
                for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
                    double d = tout[i] - ref[i];
                    num += d * d; den += ref[i] * ref[i];
                }
                p->chain_ok = (num <= den * gate);
            }
        }
    }
#endif

#ifdef HAVE_PW4
    snprintf(g_desc, sizeof g_desc,
             "L64 zsplit-custody chain (ice_r7: ztail inlined, zvp=%d "
             "vol-pair rows, zapf%d@T%d, zms=%d zs2=%d) zs=%d ck=%d xk=%d; "
             "exec pw=%d mode=%s pf=%d st=%d(%s) pro=%d sb=%d (B=%d)",
             g_zvp && p->zsb1 != NULL, FFT64B_ZAPF ? FFT64B_ZAD : 0,
             FFT64B_ZAH, g_zms, g_zs2, p->zs_ok, p->chain_ok, exec_ok,
             p->pw, mode_name[p->mode],
             p->pf, p->st, st_name[p->st], p->pro, p->sb, batch);
#else
    snprintf(g_desc, sizeof g_desc,
             "L64 zsplit-custody chain zs=%d ck=%d xk=%d; exec pw=%d mode=%s "
             "pf=%d st=%d(%s) pro=%d sb=%d (B=%d)",
             p->zs_ok, p->chain_ok, exec_ok, p->pw, mode_name[p->mode],
             p->pf, p->st, st_name[p->st], p->pro, p->sb, batch);
#endif

    if (getenv("FFT64B_VERBOSE"))
        fprintf(stderr, "L64_blocked: zs_ok=%d chain_ok=%d exec_ok=%d "
                "exec{pw=%d mode=%s pf=%d st=%d pro=%d sb=%d} nv=%d\n",
                p->zs_ok, p->chain_ok, exec_ok, p->pw, mode_name[p->mode],
                p->pf, p->st, p->pro, p->sb, nv);
    free(ri); free(ro); free(rr); free(rc);
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    if (p->map_bytes) munmap(p->rawS, p->map_bytes);
    else              free(p->rawS);
    if (p->rawP) {
        if (p->ping_bytes) munmap(p->rawP, p->ping_bytes);
        else               free(p->rawP);
    }
    if (p->rawZ) {
        if (p->zs_bytes) munmap(p->rawZ, p->zs_bytes);
        else             free(p->rawZ);
    }
    free(p);
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    run_vols(plan->pw, plan->mode, plan->pf, plan->st, plan->pro, plan->sb,
             plan->S, (const double *)in, (double *)out, plan->batch);
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
