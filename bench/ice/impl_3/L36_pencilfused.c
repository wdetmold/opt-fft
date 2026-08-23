/* =====================================================================================
 * L36_pencilfused.c -- forward complex-double 3D DFT of a fixed 36^3 cube, batched.
 *
 * TECHNIQUE (round panel_r3 revision)
 *   Two-pass pencil/tile-fused row-column transform with Good-Thomas (PFA) 4x9
 *   line kernels, INTERLEAVED-complex SIMD with a *spectator* axis in the vector
 *   lanes (a lane = one 128-bit re/im pair = one line of the pass).
 *
 *   ROUND panel_r3 CHANGE: the two passes are SWAPPED relative to round r2.  The
 *   plane-fused pass now runs FIRST, reading `in` (the only DRAM-cold buffer)
 *   plane-sequentially, and the strided x-pass runs SECOND against the cache-
 *   resident intermediate.  Round r2 had it the other way around, so its 36
 *   concurrent 20736-B-stride read streams hit DRAM-cold `in` -- the structural
 *   reason it trailed L36_pfa/L36_mixedradix by 5% (B=1) to 19% (batched) on the
 *   node.  This ordering is adopted from L36_pfa round r2 (who adopted it from
 *   L36_mixedradix round 1); both put the strided access on the warm buffer.
 *
 *     pass A (per x-plane, 36x36 complex = 20.25 KB, L1-resident):
 *         A1  y-transform, lanes = z (contiguous, shuffle-free), output
 *             transposed in PWxPW complex-lane blocks into a plane buffer P[z][ky]
 *         A2  z-transform, lanes = ky (P rows contiguous in ky), transposed back,
 *             stored to mid[x][ky][kz] (plane-sequential write)
 *     pass B (streaming): x-transform, lanes = PW consecutive flat (ky,kz).
 *         36 read streams from mid + 36 write streams to out, both stride
 *         20736 B; the reads carry an unconditional one-line-ahead software
 *         prefetch (L36_pfa measured removing it costs 14% at B=1 -- 36 streams
 *         exceed the L2 streamer).  Stores are non-temporal in the NT modes.
 *
 *   Two crossings of the grid, no materialised transpose anywhere; the only
 *   shuffles are the per-module swaps and the PWxPW register transposes on
 *   L1-resident data.  Both unavoidable volume transposes (see the round-1
 *   record for the proof there must be exactly one pair) act inside pass A.
 *
 *   MODES (self-tuned in fft3d_create(), which is not scored):
 *     0 INPLACE     mid = out.  Smallest resident set (in + out = 1.46 MB), the
 *                   small-batch winner: pass B rewrites out in place out of L2.
 *     1 SCRATCH     mid = a private one-volume scratch reused for every volume
 *                   (cache-resident across the call), cached final stores.
 *     2 SCRATCH+NT  as 1 but pass B's stores are non-temporal: DRAM traffic per
 *                   volume drops to the compulsory read-in + NT-write-out
 *                   (~1.5 MB) vs INPLACE's ~2.2 MB with the RFO.  At PW=2 two
 *                   flat groups are paired so every NT write completes a full
 *                   64-byte line (half-line NT stores thrash the ~12 WC buffers;
 *                   pairing trick from L36_pfa/L36_mixedradix round r2).
 *     3 SCRATCH+NT+XV  as 2, plus a CROSS-VOLUME software pipeline: while pass B
 *                   of volume v streams NT stores (store-buffer-bound, load
 *                   ports idle), prefetch volume v+1's `in` into L3
 *                   (prefetcht2, 36 lines per line-group; 324 groups x 36 lines
 *                   = exactly one volume).  Pass A of v+1 then reads L3, not
 *                   DRAM.  This is new this round and targets the monitor's
 *                   panel_r2 finding that L=36 batched runs at half the node's
 *                   demonstrated single-core streaming rate because reads and
 *                   compute do not overlap (VERDICT panel_r2 section 6: ceiling
 *                   ~123 us/volume at B=256 vs the measured 238.8).
 *     4 PIPE (new in panel_r4)  TRUE cross-volume software pipeline with REAL
 *                   work, not prefetches.  Two ping-pong mid buffers; pass A of
 *                   volume v+1 is interleaved with pass B of volume v at plane
 *                   granularity (one pass-A plane of v+1, then 9 pass-B line
 *                   groups of v, 36 times per volume).  The node rejected the
 *                   prefetch-based XV in r3 -- the leading explanation is that
 *                   prefetches issued while pass B's NT drains hold the fill
 *                   buffers simply get DROPPED, exactly when they are needed.
 *                   Demand loads cannot be dropped: interleaving the two passes
 *                   forces volume v+1's DRAM reads to execute between volume
 *                   v's NT store bursts, so the memory system always has both
 *                   reads and writes in flight.  Cost: the live mid set is two
 *                   buffers (cur draining + next filling; combined live bytes
 *                   stay ~1 volume, but LRU demotes the older buffer to L3, so
 *                   pass B's mid reads become L3 hits on the node's 1 MB L2).
 *                   This is the ping-pong pipeline L36_pfa's r3 record designed
 *                   and deferred ("if B=256 lands >= 180 us on the node, build
 *                   it" -- it landed 227.5); built here as an ADDITIONAL tuner
 *                   candidate per the r3 verdict's add-don't-replace lesson.
 *     5 SEQNT (new in panel_r4)  Sequential-store discipline: pass B runs IN
 *                   PLACE on the cache-resident mid (cached stores, PFA36
 *                   loads a line group entirely before its first store, so
 *                   in-place is well-defined), then one perfectly SEQUENTIAL
 *                   NT copy mid -> out.  Rationale: modes 2-4 drain NT stores
 *                   through 36 concurrent streams of stride 20736 B -- a DRAM
 *                   row-buffer-thrash pattern -- while the node's demonstrated
 *                   12.3 GB/s single-core stream (L6_unrolled r3) is a
 *                   SEQUENTIAL store stream.  The r3 verdict (section 5) found
 *                   store ORDER worth 18.5% at L=8 (L36-transposed precedent:
 *                   L8_radix8 r3 added a pass to sequentialize stores); this is
 *                   that move at L=36, cost = one extra cache-resident volume
 *                   round trip (~746 KB through L2/L3, no extra DRAM traffic).
 *     6 PIPESEQ (new in panel_r4)  Mode 5 with the copy pipelined: the
 *                   sequential NT copy of volume v-1 is interleaved into pass
 *                   B of volume v (36 vectors = 9*PW lines per group), as pass B
 *                   in-place on mid is pure compute with an idle memory system
 *                   -- the natural slot to hide the entire NT drain under.
 *                   Pass A's cold reads hide under its own compute via the
 *                   plane-ahead prefetch.  Ideal node schedule: ~65 us (pass A,
 *                   reads under compute) + max(60 compute, 62 NT drain) ~= 130
 *                   us/volume against the monitor's ~124 ceiling.  Ping-pong
 *                   mids as in PIPE.
 *     7 ISTREAM (new in panel_r5)  INPLACE with the STREAMING pass A: pass A
 *                   (z-first, transpose-on-load, sequential cold reads,
 *                   plane-ahead prefetch) writes straight into out, then pass
 *                   B runs in place on out with CACHED stores, plus ~62 KB
 *                   pre-coverage of the NEXT volume's input (3 lines per line
 *                   group) so the next pass A never starts cold.  This is
 *                   L36_pfa's r4 node-winning configuration (pw=4 inplace
 *                   pf=1: 174.2 us at B=32, 218.9 at B=256 -- the node
 *                   rejected every NT/pipeline mode) translated onto my
 *                   kernels.  My old INPLACE (mode 0) is hardwired to the
 *                   y-first pass A whose 36 stride-576B load streams are a 2x
 *                   loss on DRAM-cold input, so at streaming batches my tuner
 *                   had no viable in-place candidate at all -- that is the
 *                   structural reason r4 lost B=32 by 26%.  No new kernels:
 *                   mode 7 is passA_plane + passB_cached on the same buffer.
 *     8 ISTREAM+PFW (new in panel_r6)  Mode 7 plus a paced WRITE-INTENT
 *                   prefetch cursor over pass A's out store stream, one plane
 *                   (FFT36PF_PFWD = 2592 doubles) ahead, 18 lines per loop
 *                   iteration through both pass-A subloops.  In istream mode
 *                   pass A stores to a DRAM-cold out volume: each of the
 *                   11664 lines costs a demand RFO nothing overlaps.
 *                   __builtin_prefetch(p,1,3) emits `prefetchw`, acquiring
 *                   the line exclusive under compute.  This is L36_pfa's r5
 *                   node-winning `pf=2` (picks `inplace pf=2` 3/3 at B=32 AND
 *                   B=256; their in-arena inplace-pf2 90.5 vs inplace-pf1
 *                   156.6 us/vol at B=256 -- the RFO was the dominant exposed
 *                   cost of the very mode the node runs me in).  Gated to
 *                   streaming batches: prefetchw on cache-resident lines is
 *                   pure uop tax (pfa measured +13%/+11% at B=1/B=4;
 *                   L6_unrolled +17%).
 *   ALSO panel_r6: pass A's cold-read prefetch is now a PACED CURSOR running
 *                   FFT36PF_PFD = 4096 doubles (32 KB) ahead, advanced 18
 *                   lines per iteration through BOTH subloops -- byte-faithful
 *                   to L36_pfa's PFIN.  The r5 scheme issued the whole next
 *                   plane during the first subloop only, so the DRAM read
 *                   stream idled during the y-transform half of every plane;
 *                   the monitor's r5 verdict asked for exactly this diff (my
 *                   istream port landed 10.9% behind pfa's identical structure
 *                   on the node while measuring parity on wallaby).
 *     9 ISTREAM+NTA (new in panel_r7)  Mode 7's structure with the read
 *                   cursor NTA-hinted and NO T1/PFNX prefetch anywhere: pass
 *                   A's `in` reads go through prefetchnta (fill L1, BYPASS
 *                   L2 on SKX-class cores), so `out` -- in-place mid --
 *                   stays L2-resident across the execute, and at B=1 across
 *                   executes: out's lines stay L2-M, deleting both the RFO
 *                   and the writeback; steady-state L3 traffic collapses to
 *                   the compulsory 746 KB in-read.  This is L36_pfa r6's
 *                   pf=4 bet (their L2-eviction diagnosis of the node's B=1
 *                   cell: in+out = 1.5 MB vs the node's 1 MB L2, which
 *                   wallaby's 2 MB L2 cannot exhibit), with their pacing
 *                   discipline copied exactly: CONSTANT lead FFT36PF_PFDN =
 *                   512 doubles (4 KB), issued at consumption rate in the
 *                   FIRST subloop only (2*PFSTEP/iter; the second subloop
 *                   reads no rsrc bytes and issues nothing) -- a swinging
 *                   lead is fine for L2 but fatal for quick-evict L1 NTA
 *                   lines, and their r3 32-KB-lead NTA catastrophe (135 vs
 *                   104) was exactly that.  Candidates at B<=2 only: the
 *                   mechanism requires out to FIT in L2, and it measurably
 *                   backfires once the volume set streams (see the tuner).
 *    11 INPLACE-CS (new in panel_r9)  Mode 0's exact arithmetic (bit-identical
 *                   output) through COMPACT-CODE twins, for the front-end
 *                   story the r8 verdict names as the L=36 B=1 lever: under
 *                   the node's own flags mode 0's pass-A x-plane loop body
 *                   measures 1221 instructions (~6.9 KB: two distinct
 *                   610-instruction subloop bodies alternating) and pass B's
 *                   group loop is unrolled x2 to ~7 KB -- both marginal-to-
 *                   over the CLX DSB (~1.5k uops), while wallaby's SPR DSB
 *                   (4k uops) holds them trivially, which would explain the
 *                   panel-worst wallaby->node ratio.  The compact twins:
 *                   (a) ONE shared noinline halfplane() does BOTH pass-A
 *                   subloops -- they are the same code, both load PW lanes at
 *                   stride 72 doubles and store PWxPW-transposed rows of
 *                   stride 72, because PST == 36 -- halving the pass-A hot
 *                   footprint to one ~3.4 KB body; (b) passB_small() fences
 *                   the group loop against -funroll-loops (unroll 1), so the
 *                   ~2.9 KB single-group body stays DSB-resident.  Zero
 *                   arithmetic change, zero extra memory traffic; cost is 72
 *                   call/rets + ~648 loop-control uops per volume.  Raced at
 *                   B<=8; the in-arena inplace-vs-cs pair is reported in the
 *                   description string so the node's own tournament doubles
 *                   as the DSB/MITE discriminator the monitor asked for.
 *    10 INPLACE+NTA (new in panel_r7, this file's own variant)  Mode 0's
 *                   register-friendly y-first pass A (no staging arrays, no
 *                   load-side shuffles -- beats z-first by ~4 us at B=1 on
 *                   wallaby and the node picked it over istream at B=1 in
 *                   r5) plus an NTA read cursor matched to ITS consumption
 *                   order, which is column-major over the plane's 36x9 line
 *                   grid: iteration zg consumes line-column zg (36 lines,
 *                   2.25 KB), so the cursor prefetches line-column zg+2 --
 *                   a constant 128-B-per-row (2-column, 4.5 KB) lead --
 *                   wrapping columns 9,10 into the NEXT plane so every line
 *                   is issued exactly once.  A sequential cursor cannot
 *                   pace this pass (that mismatch is why r6 left mode 0
 *                   prefetch-free); the column cursor can.  If NTA pays on
 *                   the node, this keeps the y-first compute advantage on
 *                   top of it.  Candidates at B<=2 only, as mode 9.
 *   ROUND ice_r2 (first ice-panel revision; ice_r1's agent crashed at launch).
 *                 Measured on the node itself under the GRADED CHAIN (B=8,
 *                 m=64, --unitary), forced-mode A/B: mode 0 = 129.4/140.1
 *                 (min/median), mode 11 = 129.9/129.9, mode 7 = 134.6,
 *                 mode 8 = 134.9, mode 12 = 138.0, mode 1 = 154.3 us/xform.
 *                 The y-first INPLACE class wins the chain cell by 4-6%; all
 *                 four changes below follow from that and from the ice_r1
 *                 verdict (plan races swing unmodified binaries ~12%; arena
 *                 probes ran +24-47% optimistic because no arena modelled the
 *                 chain):
 *                 (1) CHAIN-SHAPED TUNER ARENA: the timed unit is now chain
 *                     steps -- exec(src,dst), then the driver's own unitary
 *                     scale (x 1/216, sqrt(36^3) = 216 exactly) over dst,
 *                     then ping-pong -- through THREE buffers (din/dout/
 *                     dpong) exactly like driver.c RUN_UNIT.  Candidates are
 *                     now ranked by the workload the monitor scores, and the
 *                     probe values in the description are directly
 *                     comparable to leaderboard per-transform numbers.
 *                     (Chain-tuning idea from L17_matrixsimd ice_r1, who
 *                     measured the same fix worth -19% on their median.)
 *                 (2) CLOCK-SETTLE SPIN (~120 ms) before any candidate is
 *                     timed: schedutil leaves a fresh core at 2.90 GHz base
 *                     while a ramped one runs 3.3-3.5; probing unramped
 *                     mis-ranks plans (L17_matrixsimd ice_r1's diagnosis).
 *                 (3) GATES RE-KEYED TO THE DETECTED L3 (ICX rule): pfw and
 *                     NT/XV/pipe candidates admitted only once the batch
 *                     working set clearly exceeds L3 -- the chain keeps
 *                     in/out L3-resident, where prefetchw is a pure uop tax
 *                     (L13_rader ice_r1: +7.4%) and NT stores are
 *                     catastrophic (L17_matrixsimd ice_r1: 28.9 vs 14.3
 *                     us/step).  SCRATCH (mode 1) is dropped at B<=8: the
 *                     node's own chain A/B says -19%.
 *                 (4) ANTI-PHASE TRAVERSAL (new mechanism, this file): under
 *                     the chain the driver scales dst volume 0 -> 7, plane
 *                     0 -> 35, immediately before our next execute reads it
 *                     back; so the y-first modes (0/11) now process volumes
 *                     7 -> 0 and planes 35 -> 0.  Each execute starts on the
 *                     lines the scale pass touched LAST (still L2-resident)
 *                     and finishes on volume 0 plane 0 -- which is exactly
 *                     where the driver's next forward scale pass begins.
 *                     Bit-identical output (per-plane work is independent);
 *                     -DFFT36PF_FWD restores forward order for A/B.  The
 *                     cursor modes (7/8/9/10/12) keep forward order -- their
 *                     prefetch pacing assumes it.
 *                 (5) MODES 13 BCST0 / 14 BCST+PF: the transpose-free
 *                     broadcast pass A (adopted from L36_pfa's ice_r2 tr=1 --
 *                     masked vbroadcastf64x2 loads move the lane transposes
 *                     off port 5, where ICX's second FMA pipe lives, onto
 *                     the idle load ports; see passA_bcst below), run
 *                     istream-style straight into out with pass B cached in
 *                     place.  13 is the prefetch-free twin of mode 12; 14
 *                     carries mode 7's paced T1 cursor + PFNX.  PW=4 only.
 *
 *   ROUND ice_r3: modes 15 BCST+XV / 16 BCST+XV+VREV / 17 BCST+VREV.
 *                 Scored ice_r2: 110.477 us/xform, 1st (pfa 112.727,
 *                 mixedradix 116.814); pick pw4/bcst0, probe bc4=110.0 --
 *                 the chain probe now predicts the score to 0.4%.  This
 *                 round attacks the ~60 us/vol gap between the 48.4 us
 *                 two-pipe floor and the measured execute: per volume the
 *                 istream shape reads 746 KB of dirty-L3 `in` and RFOs
 *                 746 KB of `out` during pass A (memory-exposed), while
 *                 pass B runs compute-bound on the L2-resident volume with
 *                 the memory system IDLE (every out line is loaded once,
 *                 stored once, then dead until the driver's scale pass).
 *                 Mode 15 fills that idle window: pass B prefetches
 *                 (t1, into L2) the NEXT volume's input, 9*PW lines per
 *                 line group x 324 groups = exactly one volume, so pass A
 *                 of v+1 reads L2 instead of dirty L3.  This is mode 3's
 *                 XV idea relocated to the chain regime -- the r3/r4
 *                 failure mode (prefetches dropped while NT drains hold
 *                 the fill buffers) cannot occur here because there are no
 *                 NT stores and pass B's demand traffic is all L2 hits.
 *                 Mode 16 adds VOLUME-ONLY reversal (planes stay forward,
 *                 so each volume's cold read remains one ascending
 *                 sequential stream -- the descending-PLANE order is what
 *                 killed the r2 reversal at 142.5 vs 123.0): the driver's
 *                 scale pass ends at volume 7 (L2-warm when my execute
 *                 starts there) and next re-reads dst from volume 0 --
 *                 exactly where reversed execute finishes.  XV then stages
 *                 v-1 during pass B of v.  Mode 17 = reversal alone.
 *                 Modes 18 BCST+NTA / 19 BCST+NTA+VR (added mid-round after
 *                 the node's SKIPB phase split): at the graded cell pass B
 *                 measures ~27 us/vol against its 16.1 floor and pass A ~82
 *                 against 32 -- the excess is L2 THRASH: pass A's read-once
 *                 `in` stream (746 KB) evicts the freshly written out volume
 *                 (746 KB) from the 1.25 MB L2, so pass B re-reads dirty L3
 *                 lines and doubles the writeback traffic on a cell that the
 *                 phase split shows is L3-bandwidth-bound end to end.  Fix:
 *                 the bcst pass A's subloop-A reads carry a CONSTANT-LEAD
 *                 (4 KB = FFT36PF_PFDN) prefetchnta cursor paced at exactly
 *                 the consumption rate (36 lines per yb-iteration; subloop B
 *                 reads only pp and issues nothing) -- pfa r6's pacing
 *                 discipline, their B=1 L2-eviction diagnosis applied to the
 *                 B=8 chain, where in+out per VOLUME (not per batch) is what
 *                 must fit in L2.  NTA lines fill L1 and skip L2, so `out`
 *                 stays L2-resident through both passes.  pfa's ice_r2 chain
 *                 sweep priced pf=1/2/7 (all tax) but never the NTA flavor.
 *    12 ISTREAM0 (new in panel_r11)  Mode 7's structure with NO prefetch of
 *                   any kind: z-first pass A (transpose-on-load, sequential
 *                   plane reads) straight into out, pass B cached in place
 *                   on out, no read cursor, no PFNX.  This is L36_pfa's
 *                   node-winning B=1 configuration `pw4 inplace pf=0`
 *                   (picked 3/3 in both r9 and r10; their r10 in-plan probe
 *                   fu = 115.0-116.4 us against my mode-0 probe ip4 =
 *                   118.8-120.3 in the same node runs) translated onto my
 *                   kernels.  Mechanism: at node B=1 in+out = 1.5 MB
 *                   overflows the 1 MB L2, so mode 0's y-first pass A reads
 *                   L3-resident `in` through 36 stride-576B streams per
 *                   plane -- the access-order loss documented at r3 on cold
 *                   input (101 vs 58 us/vol), reappearing at L3 scale.
 *                   Wallaby's 2 MB L2 holds both buffers, which is why
 *                   y-first wins there (51.6 vs 55.9, r3) and why five
 *                   rounds of wallaby tuning never surfaced this.  Mode 7
 *                   is NOT this shape: it hardwires the paced T1 read
 *                   cursor = pfa's pf=1, which their node tuner rejects at
 *                   B=1 (pf=0 beat their pf=1 twin by 4.4% in-arena).  So
 *                   my small-batch candidate list never contained the
 *                   node's actual winning shape -- the same structural hole
 *                   as r5's "no viable in-place option", one level down.
 *                   Raced at B<=8; installs over the y-first incumbent only
 *                   past a 3% cross-bit-class margin (see the tuner).
 *
 * OPERATION COUNT (per 36-point line over PW lanes; unchanged from round r2)
 *   Good-Thomas 4x9: n = (9 n1 + 4 n2) mod 36, k = (9 k1 + 28 k2) mod 36, so
 *   W36^{nk} = W4^{n1 k1} * W9^{n2 k2} and the inter-factor twiddle stage vanishes.
 *
 *     DFT4  interleaved: 8 FMA-port ops + 1 swap            x9   72 + 9
 *     DFT9  genfft n1_9 FMA DAG: 40 FMA-port ops + 12 swaps x4  160 + 48
 *     PFA-36 line:       232 FMA-port ops + 57 port-5 shuffles, 36 ld + 36 st
 *     (round panel_r10: DFT9 was hand CT 3x3 at 44 + 10 -> line was 248 + 49;
 *      transcription rule from L45_pfa r9, DAG from fftw-3.3.10 n1_9.c)
 *
 *   Per volume: 3 x 1296 lines -> 3888/PW kernel calls; 226k FMA-port vector ops
 *   at PW=4, floor ~78 us at 2.89 GHz on the node's single 512-bit FMA pipe (or
 *   two 256-bit ones -- identical by construction, so both widths are built and
 *   the plan-time tuner decides).  Register transposes add 8 shuffles per PW
 *   vectors of PW complex, on port 5, which the FMA work leaves idle.
 *
 * ASSUMPTIONS
 *   * L == 36 only; in/out distinct and 64-byte aligned (driver guarantees); `in` const.
 *   * gcc vector extensions + explicit FMA forms (via contraction) for the modules.
 *   * Two instantiations from one source text via #include __FILE__: PW=2 with no
 *     target attribute (inherits -march=native: AVX2 on the dev box, EVEX-encoded
 *     256-bit with 32 registers on the node) and PW=4 under target("avx512f,...").
 *     fft3d_create() times all (width x mode) configurations, interleaved-rounds
 *     protocol, and every candidate must reproduce the PW=2 INPLACE answer to
 *     1e-11 relative before it is eligible, so a path that cannot be executed at
 *     development time can cost speed but never correctness.
 *   * FFT36PF_SKIPA / FFT36PF_SKIPB compile out one pass -- WRONG ANSWERS, purely
 *     a phase-timing diagnostic for tryout runs with explicit -D flags.
 * ===================================================================================== */

#ifndef L36_PENCILFUSED_TEMPLATE
#define _POSIX_C_SOURCE 200809L
/* ------------------------------------------------------------------ common part ---- */
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <immintrin.h>

/* Round panel_r9: the r8 `#pragma GCC optimize("unroll-loops")` is REMOVED.
 * The r8 verdict (section 3c) checked the harness: Makefile line 15 has carried
 * -funroll-loops on the scored build for at least three rounds, so the premise
 * was false, and L17_rader measured the pragma form itself as a ~2% tax
 * (optimize() rebuilds the whole per-function option set, not just the named
 * flag).  The scored and tryout builds are flag-identical; nothing to pin. */

#include "fft3d_api.h"

#define LL      36
#define NPLANE  1296                /* 36*36 complex in one (y,z) plane               */
#define NVOL2   93312               /* doubles per volume, interleaved complex        */
#define PST     36                  /* P-buffer row stride, complex (576 B = 9 lines, */
                                    /* odd in lines -> touches all 64 L1 sets)        */
#define MIDSKIP (NVOL2 + 64)        /* doubles between the two ping-pong mid buffers  */
                                    /* (+64 keeps 64-B alignment and offsets L2 sets) */

/* pass-A prefetch pacing (values = L36_pfa's node-selected r5 configuration).
 * PFD: how far (doubles) the paced READ cursor runs ahead of the plane being
 * consumed -- 32 KB clears the pacing deficit of the first subloop while
 * staying far inside the node's 1 MB L2.  PFWD: how far the WRITE-INTENT
 * cursor (mode 8) runs ahead of pass A's out stores -- one plane = 20.25 KB
 * gives every line a 0.5-1.5-plane lead, enough to cover a DRAM RFO, short
 * enough that L2 never evicts a line between prefetchw and store. */
#ifndef FFT36PF_PFD
# define FFT36PF_PFD  4096
#endif
#ifndef FFT36PF_PFWD
# define FFT36PF_PFWD 2592
#endif
/* PFDN: constant lead (doubles) of the NTA read cursor in modes 9/10.
 * 512 doubles = 4 KB = L36_pfa r6's node-shipped FFT36_PFDN default; their
 * sweep found 128 (1 KB) too late (demand loads outrun it, 135.7 vs 109.4)
 * and 256~512 within noise. Small because NTA lines are L1-quick-evict:
 * a line dropped before use was never put in L2 and is re-read from L3. */
#ifndef FFT36PF_PFDN
# define FFT36PF_PFDN 512
#endif

/* sqrt(3)/2 for the radix-3/4 modules, plus the genfft n1_9 DAG constants
 * (round panel_r10: DFT9 = FFTW 3.3.10 n1_9 FMA codelet transcribed to
 * interleaved vectors, 44 -> 40 FMA-port ops; transcription rule and constant
 * set from L45_pfa r9, values verbatim from dft/scalar/codelets/n1_9.c). */
#define K3     0.86602540378443864676
#define K176   0.17632698070846497347109038686862  /* tan(pi/18)             */
#define K839   0.83909963117728001176312729812318
#define K777   0.77786191343020616002817797731863
#define K984   0.98480775301220805936674302458952
#define K492   0.49240387650610402968337151229476
#define K852   0.85286853195244320962825096394007
#define K363   0.36397023426620236135104788277683  /* tan(pi/9)              */
#define K954   0.95418889413867113349926836418725

struct fft3d_plan {
    int    batch;
    int    pw;                      /* 2 or 4 complex lanes, chosen by measurement    */
    int    mode;                    /* 0 INPLACE, 1 SCRATCH, 2 +NT, 3 +NT+XV, 4 PIPE,
                                       5 SEQNT, 6 PIPESEQ, 7 ISTREAM, 8 ISTREAM+PFW,
                                       9 ISTREAM+NTA, 10 INPLACE+NTA, 11 INPLACE-CS,
                                       12 ISTREAM0 (istream, no prefetch at all),
                                       13 BCST0 / 14 BCST+PF (ice_r2: transpose-
                                       free broadcast pass A, pf=0 / pf=1 twins),
                                       15 BCST+XV / 16 BCST+XV+VREV / 17
                                       BCST+VREV (ice_r3: pass-B cross-volume
                                       L2 staging, volume-only reversal),
                                       18 BCST+NTA / 19 BCST+NTA+VREV (ice_r3:
                                       L2-bypassing paced in-read) */
    double *mid;                    /* two-volume ping-pong scratch (PIPE uses both)  */
    double *pp;                     /* 36 x PST complex plane buffer                  */
    void   *arena;
};

/* ---------------------------------------------------------------- instantiations ---- */
#define L36_PENCILFUSED_TEMPLATE 1

#define PW     2
#define TS(x)  x##_v2
#define TATTR
#include __FILE__
#undef PW
#undef TS
#undef TATTR

#define PW     4
#define TS(x)  x##_v4
#define TATTR  __attribute__((target("avx512f,avx512dq,avx512vl")))
#include __FILE__
#undef PW
#undef TS
#undef TATTR

/* --------------------------------------------------------------------------- API ---- */

static const char *mode_name(int m)
{
    return m == 0 ? "inplace" : m == 1 ? "scratch" : m == 2 ? "scratch+nt"
                              : m == 3 ? "scratch+nt+xvpf" : m == 4 ? "pipe"
                              : m == 5 ? "scratch+seqnt" : m == 6 ? "pipeseq"
                              : m == 7 ? "istream" : m == 8 ? "istream+pfw"
                              : m == 9 ? "istream+nta" : m == 10 ? "inplace+nta"
                              : m == 11 ? "inplace-cs" : m == 12 ? "istream0"
                              : m == 13 ? "bcst0" : m == 14 ? "bcst+pf"
                              : m == 15 ? "bcst+xv" : m == 16 ? "bcst+xv+vr"
                              : m == 17 ? "bcst+vr" : m == 18 ? "bcst+nta"
                              : "bcst+nta+vr";
}

/* the tuner's pick is written here so the monitor can read it off the raw
 * per-case json (VERDICT panel_r2, cross-cutting item 2) */
static char g_desc[288] =
    "L=36: plane-fused y+z pass then strided x pass, PFA 4x9 interleaved-complex "
    "line kernel, in-place/NT/cross-volume-prefetch modes self-tuned (pre-create)";

const char *fft3d_name(void)        { return "L36_pencilfused"; }
const char *fft3d_description(void) { return g_desc; }
int fft3d_supports(int L)           { return L == LL; }

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    const double *ind = (const double *)in;
    double *outd = (double *)out;
    if (p->pw == 4)
        exec_v4(ind, outd, p->mid, p->pp, p->batch, p->mode);
    else
        exec_v2(ind, outd, p->mid, p->pp, p->batch, p->mode);
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != LL || batch < 1) return NULL;

    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->batch = batch;

    const size_t nMid = (size_t)2 * MIDSKIP;      /* two volumes: PIPE ping-pongs */
    const size_t nP   = (size_t)36 * PST * 2 + 64;
    void *arena = NULL;
    if (posix_memalign(&arena, 64, (nMid + nP) * sizeof(double)) != 0 || !arena) {
        free(p);
        return NULL;
    }
    memset(arena, 0, (nMid + nP) * sizeof(double));
    p->arena = arena;
    p->mid = (double *)arena;
    p->pp  = p->mid + nMid;

    /* defaults if the tuning allocation fails: 256-bit; streaming batches get
     * ISTREAM+PFW (the node's r5-winning shape via L36_pfa), small batches
     * INPLACE */
    p->pw   = 2;
    p->mode = ((double)batch * 1492992.0 > 40.0 * 1024.0 * 1024.0) ? 8 : 0;

    /* ---- self-tune: {PW=2, PW=4} x {INPLACE, SCRATCH, NT, NT+XV}, timed and
     * CROSS-CHECKED.  Every configuration must reproduce the PW=2 INPLACE answer
     * on this machine to 1e-11 relative before it may be selected.
     * The arena is capped at 64 volumes (~191 MB in+out) so the NT-vs-cached
     * ranking actually streams past every L3 this code will meet (L36_pfa r2:
     * a 16-volume arena fit wallaby's 60 MB L3 and mis-picked cached stores;
     * this file's first r3 attempt repeated that at 48 volumes -- tuner said
     * cached 80.8 vs NT+XV 95.4, the full 382 MB run said 146.4 vs 119.2). */
    const long tb = batch < 64 ? batch : 64;
    const long rv = tb < 2 ? tb : 2;    /* volumes checked for admission */
    double *din = NULL, *dout = NULL, *dpong = NULL, *dref = NULL;
    if (posix_memalign((void **)&din,  64, (size_t)tb * NVOL2 * sizeof(double)) == 0 &&
        posix_memalign((void **)&dout, 64, (size_t)tb * NVOL2 * sizeof(double)) == 0 &&
        posix_memalign((void **)&dpong,64, (size_t)tb * NVOL2 * sizeof(double)) == 0 &&
        posix_memalign((void **)&dref, 64, (size_t)rv * NVOL2 * sizeof(double)) == 0) {

        /* clock-settle spin (ice_r2, from L17_matrixsimd ice_r1): schedutil
         * leaves a fresh core at the 2.90 GHz base clock; ~120 ms of
         * dependent FP work ramps it before any candidate is timed, so the
         * tournament measures plans at the frequency the scored run sees. */
        {
            double t0 = now_s(), x = 1.0000000001;
            volatile double sink;
            while (now_s() - t0 < 0.12) {
                for (int i = 0; i < 50000; ++i) x = x * 1.0000000001 + 1e-12;
                sink = x;
            }
            (void)sink;
        }

        unsigned st = 12345u;
        for (size_t i = 0; i < (size_t)tb * NVOL2; ++i) {
            st = st * 1103515245u + 12345u;
            din[i] = (double)(int)(st >> 16) * 3.0517578125e-5;
        }
        memset(dpong, 0, (size_t)tb * NVOL2 * sizeof(double));
        exec_v2(din, dout, p->mid, p->pp, rv, 0);
        memcpy(dref, dout, (size_t)rv * NVOL2 * sizeof(double));
        double refmax = 0.0;
        for (size_t i = 0; i < (size_t)rv * NVOL2; ++i) {
            double a = dref[i] < 0 ? -dref[i] : dref[i];
            if (a > refmax) refmax = a;
        }
        if (refmax <= 0.0) refmax = 1.0;

        /* Admission first (correctness gate), then time the survivors over
         * INTERLEAVED rounds keeping each candidate's minimum -- contiguous
         * per-candidate blocks let one load spike mis-rank the whole plan
         * (round-r2 lesson, protocol from L36_mixedradix round 1).  NT/XV
         * candidates are admitted only once the batch has clearly left L3;
         * below that a forced DRAM write per call can only lose (round-1
         * measurement: NT neutral at B<=4, 1.53x win at B=32). */
        const int have512  = __builtin_cpu_supports("avx512f");
        /* ice_r2: both gates re-keyed from fixed byte counts to the DETECTED
         * L3.  The graded chain keeps the whole batch L3-resident, and there
         * (a) NT stores force a DRAM round trip the next chain step reads
         * straight back (L17_matrixsimd ice_r1: 28.9 vs 14.3 us/step), and
         * (b) prefetchw on an L3-resident out is a pure uop tax (L13_rader
         * ice_r1: +7.4%; this file's own node chain A/B: mode 8 = 134.9 vs
         * mode 0 = 129.4 us/xform at B=8).  Both mechanisms pay only once
         * the working set streams past L3. */
        long l3sz = sysconf(_SC_LEVEL3_CACHE_SIZE);
        const double l3b = (l3sz > 0) ? (double)l3sz : 30.0 * 1024.0 * 1024.0;
        const int allow_nt  = ((double)batch * 1492992.0 > 1.25 * l3b);
        const int allow_pfw = ((double)batch * 1492992.0 > 1.25 * l3b);
        enum { NM = 20 };                            /* modes 0..19 */
        double bestc[2 * NM];
        int ok[2 * NM];
        for (int cfg = 0; cfg < 2 * NM; ++cfg) {     /* cfg = NM*(pw4?1:0) + mode */
            const int w = (cfg >= NM) ? 4 : 2;
            const int m = cfg % NM;
            bestc[cfg] = 1e300;
            ok[cfg] = 0;
            if (w == 4 && !have512) continue;
            /* forced diagnostics bypass the physics gates below, so the
             * monitor can take e.g. a forced istream0 number at any batch */
#ifdef FFT36PF_FORCE_PW
            if (w != FFT36PF_FORCE_PW) continue;
#endif
#ifdef FFT36PF_FORCE_MODE
            if (m != FFT36PF_FORCE_MODE) continue;
#else
            /* ISTREAM (7) ungated; ISTREAM+PFW (8) admitted from B>=3 (see
             * allow_pfw above).  NTA modes (9/10) exist to keep OUT
             * L2-resident, which is physically possible only while the
             * revisited set (~1.5 MB x B) is L2-scale -- B<=2.  Beyond that
             * the prize is gone and the prefetch is pure cost, measured:
             * wallaby B=4 mode 10 136.8 vs mode 0 ~76 us/vol (+80%), mode 9
             * 94.5 (+24%); same mechanism as L36_pfa r6's B=32 NTA loss
             * (109.4 vs 95.7) and their r3 catastrophe (135 vs 104). */
            if ((m >= 2 && m <= 6) && !allow_nt) continue;
            if (m == 8 && !allow_pfw) continue;
            /* SCRATCH at chain scale: the node's own graded-chain A/B reads
             * 154.3 vs 129.4 us/xform (ice_r2) -- the private mid buys
             * nothing while everything is L3-resident and costs a third
             * volume pass through L2.  Raced only where NT modes live. */
            if (m == 1 && !allow_nt) continue;
            if ((m == 9 || m == 10) && batch > 2) continue;
            /* the compact-code twin targets the small-batch (mode-0) cells;
             * at streaming batch its pass A has mode 0's documented 2x
             * cold-read loss (101 vs 58 us/vol, r3), so racing it there
             * only spends tuner time */
            if (m == 11 && batch > 8) continue;
            /* istream0 targets the B<=8 cells where out is the L2-scale
             * revisited set; at streaming batch mode 8 (istream+pfw) is the
             * node's two-round 3/3 pick and pf-less pass A only re-exposes
             * the cold-read stalls the cursor exists to hide */
            if (m == 12 && batch > 8) continue;
            /* the broadcast pass A is built at PW=4 only (its whole point is
             * the 512-bit port-5 economy); raced at every batch -- its
             * streaming behaviour is unknown and the tournament prices it */
            if (m >= 13 && w == 2) continue;
            /* ice_r3: the XV/vrev bcst modes target the L3-resident CHAIN
             * cells only.  At B=1 there is no next volume to stage and no
             * scale-pass phase to anti-align with; past L3 the t1 staging
             * prefetch competes with the compulsory DRAM streams for the
             * same bandwidth it is trying to hide (mode 3's r3 lesson). */
            if ((m >= 15 && m <= 17) && (batch < 2 || allow_nt)) continue;
            /* ice_r3: the NTA modes protect one VOLUME's out from the
             * read-once in stream -- valid at any L3-resident batch (the
             * per-volume 1.5 MB working set is what overflows L2, at every
             * B); past L3 the NTA read defeats the L2 streamer's role on a
             * DRAM stream.  +vr needs a chain phase to anti-align with. */
            if ((m == 18 || m == 19) && allow_nt) continue;
            if (m == 19 && batch < 2) continue;
#endif
            if (w == 4) exec_v4(din, dout, p->mid, p->pp, rv, m);
            else        exec_v2(din, dout, p->mid, p->pp, rv, m);
            double err = 0.0;
            for (size_t i = 0; i < (size_t)rv * NVOL2; ++i) {
                double d = dout[i] - dref[i];
                if (d < 0) d = -d;
                if (d > err) err = d;
            }
            ok[cfg] = (err <= 1e-11 * refmax);
        }
        /* ---- CHAIN-SHAPED timing (ice_r2) ----------------------------------
         * The graded workload is a chain: each execute's src is the previous
         * execute's dst, freshly rescaled BY THE DRIVER (x 1/sqrt(36^3) =
         * 1/216, a full extra pass over the batch inside the timed unit),
         * ping-ponging between two destination buffers.  ice_r1's verdict
         * (section 4a) shows every arena that ignored this ran 24-47%
         * optimistic and could mis-rank candidates; L36_mixedradix, whose
         * probe was chain-accurate to 2.2%, took the cell.  So the timed
         * unit here IS a chain step: exec + driver-style unitary scale +
         * ping-pong, through three buffers exactly like driver.c RUN_UNIT.
         * The chain state (chsrc/chdst) persists across candidates and
         * rounds -- the transform is norm-preserving under the scale, so the
         * data stays healthy (no decay, no denormals) indefinitely; one
         * untimed warm step per visit re-establishes THIS candidate's
         * steady-state cache before its timed steps (the r10 self-warm
         * lesson, unchanged).  Many short interleaved rounds, keep the min. */
#define CHSTEP(w_, m_) do {                                                    \
            if ((w_) == 4) exec_v4(chsrc, chdst, p->mid, p->pp, tb, (m_));     \
            else           exec_v2(chsrc, chdst, p->mid, p->pp, tb, (m_));     \
            for (size_t j_ = 0; j_ < (size_t)tb * NVOL2; ++j_)                 \
                chdst[j_] *= 4.6296296296296296e-3;   /* 1/216 exactly as    */ \
            chsrc = chdst;                            /* the driver applies  */ \
            chdst = (chdst == dout) ? dpong : dout;                            \
        } while (0)
        const double *chsrc = din;
        double *chdst = dout;
        const int csteps = (tb <= 2) ? 8 : (tb <= 8 ? 4 : 2);
        const int rounds = (tb <= 2) ? 12 : (tb <= 8 ? 8 : 4);
        for (int r = 0; r < rounds; ++r) {
            for (int cfg = 0; cfg < 2 * NM; ++cfg) {
                if (!ok[cfg]) continue;
                const int w = (cfg >= NM) ? 4 : 2;
                const int m = cfg % NM;
                CHSTEP(w, m);                           /* untimed self-warm */
                double t0 = now_s();
                for (int q = 0; q < csteps; ++q)
                    CHSTEP(w, m);
                double t = (now_s() - t0) / csteps;
                if (t < bestc[cfg]) bestc[cfg] = t;
            }
        }
#undef CHSTEP
        /* pick: min over candidates, then a simplest-wins hysteresis band
         * (adopted from L36_pfa r4, node-validated against pick coin-flips):
         * among candidates within the band of the best, install the
         * structurally simplest MODE (mrank: inplace < inplace-cs < istream
         * < istream+pfw < inplace+nta < istream+nta < scratch < nt < seqnt
         * < xv < pipe < pipeseq); at equal mode keep the faster width.
         * Band is REGIME-AWARE (adopted from L36_pfa r8): 1% when the arena
         * IS the scored regime (tb == batch <= 8 -- same buffers, same
         * steady loop the driver times, self-warmed, in-arena sd ~0.5%),
         * 3% otherwise (streaming arenas misrepresent the end-to-end run).
         * At 3% a genuine 1-3% B=1 win could never install -- exactly the
         * size of the gap to the cell leader.
         * inplace-cs carries the SAME rank as inplace: same structure, same
         * arithmetic, bit-identical output -- the only difference is code
         * bytes, so the tie-break between them is purely the measured time
         * (a pick flip between them has no timed-vs-checked consequence). */
        static const int mrank[NM] = {0, 6, 7, 9, 10, 8, 11, 2, 3, 5, 4, 0, 1,
                                      1, 2, 2, 3, 2, 2, 3};
                                               /* bcst0/bcst+pf rank with their
                                                  istream0/istream twins;
                                                  ice_r3: xv/vr/nta = 2,
                                                  two-mechanism combos = 3 */
        const double band = (tb == (long)batch && batch <= 8) ? 1.01 : 1.03;
        /* Bit-class discipline (panel_r10 rule, from L36_mixedradix r10: "a
         * tuner pool must be one bit class; cross-class comparisons ride the
         * description string, never the pick").  Modes {0,10,11} share the
         * y-first pass-A arithmetic (rel_l2 fingerprint 3.748e-16 family);
         * every other mode runs the z-first passA_plane order (3.586e-16
         * family) -- DIFFERENT bit classes.  Everything admitted is still
         * TIMED (the ip/cs/is0 probe pair rides the description), but the
         * PICK is restricted so two processes can never install different
         * classes:
         *   B<=2  -- the class is chosen DETERMINISTICALLY, by the physics
         *     the two classes differ on: if this machine's L2 cannot hold
         *     in+out (batch x 1.46 MB), mode 0's y-first pass A reads its
         *     plane through 36 stride-576B streams off L3 (the r3 access-
         *     order loss at L3 scale) and the z-first class installs; if L2
         *     holds both buffers, the y-first class installs (its register-
         *     friendly kernel wins, r3: 51.6 vs 55.9 on wallaby).  On the
         *     node (1 MB L2 < 1.46 MB) this selects istream0 at B=1 --
         *     L36_pfa's B=1-winning `inplace pf=0` shape, picked 3/3 there
         *     in r9 and r10 (their in-plan fu 115.0-116.4 vs my ip4 probe
         *     118.8-120.3, same r10 node runs); on wallaby (2 MB) and any
         *     L2-large machine it keeps the incumbent.  No timing coin flip
         *     exists because timing does not choose the class at all.
         *   2<B<=8 -- class A is the incumbent; a class-B candidate must
         *     beat the best class-A one by >3% to take the install (at the
         *     node's measured B=4 gap of <1% this never fires, so the pick
         *     stays deterministic in practice).
         *   B>8 -- unrestricted, as before: the class gap at streaming
         *     batch is the documented 2x cold-read loss, outside any band. */
#define CLSA_(m) ((m) == 0 || (m) == 10 || (m) == 11)
        int inst[2 * NM];
        for (int cfg = 0; cfg < 2 * NM; ++cfg) inst[cfg] = ok[cfg];
#ifndef FFT36PF_FORCE_MODE
        if (batch <= 2) {
            long l2sz = sysconf(_SC_LEVEL2_CACHE_SIZE);
            const int overflow = (l2sz > 0)
                && ((double)l2sz < (double)batch * 1492992.0);
            for (int cfg = 0; cfg < 2 * NM; ++cfg)
                if (inst[cfg] && (overflow ? CLSA_(cfg % NM)
                                           : !CLSA_(cfg % NM)))
                    inst[cfg] = 0;
            /* if the whole selected class failed admission, fall back to
             * anything correct rather than the untimed default */
            int any = 0;
            for (int cfg = 0; cfg < 2 * NM; ++cfg) any |= inst[cfg];
            if (!any)
                for (int cfg = 0; cfg < 2 * NM; ++cfg) inst[cfg] = ok[cfg];
        }
#endif
        double bestt = 1e300;
        for (int cfg = 0; cfg < 2 * NM; ++cfg)
            if (inst[cfg] && bestc[cfg] < bestt) bestt = bestc[cfg];
        int pick = -1;
        for (int cfg = 0; cfg < 2 * NM; ++cfg) {
            if (!inst[cfg] || bestc[cfg] > band * bestt) continue;
            if (pick < 0 || mrank[cfg % NM] < mrank[pick % NM]
                || (mrank[cfg % NM] == mrank[pick % NM]
                    && bestc[cfg] < bestc[pick]))
                pick = cfg;
        }
        if (pick >= 0 && batch > 2 && batch <= 8 && !CLSA_(pick % NM)) {
            double bA = 1e300;
            for (int cfg = 0; cfg < 2 * NM; ++cfg)
                if (inst[cfg] && CLSA_(cfg % NM) && bestc[cfg] < bA)
                    bA = bestc[cfg];
            if (bA < 1e300 && bestc[pick] > bA / 1.03) {
                int pa = -1;
                for (int cfg = 0; cfg < 2 * NM; ++cfg) {
                    if (!inst[cfg] || !CLSA_(cfg % NM)
                        || bestc[cfg] > band * bA) continue;
                    if (pa < 0 || mrank[cfg % NM] < mrank[pa % NM]
                        || (mrank[cfg % NM] == mrank[pa % NM]
                            && bestc[cfg] < bestc[pa]))
                        pa = cfg;
                }
                if (pa >= 0) pick = pa;
            }
        }
#undef CLSA_
        if (pick >= 0) {
            p->pw   = (pick >= NM) ? 4 : 2;
            p->mode = pick % NM;
        }
        /* the inplace-vs-compact in-arena pair rides the description string
         * onto the leaderboard (L36_pfa r8's probe-in-description pattern):
         * cs < ip on the node = the front-end/code-size story confirmed at
         * the shared-body level, even if the pick does not change. */
        {
            const int wb = have512 ? NM : 0;
            char probe[160] = "";
            int pn = 0;
            if (ok[wb + 0] && ok[wb + 11])
                pn = snprintf(probe, sizeof probe, "; chain probe us ip%d=%.1f cs%d=%.1f",
                              have512 ? 4 : 2, bestc[wb + 0] * 1e6 / tb,
                              have512 ? 4 : 2, bestc[wb + 11] * 1e6 / tb);
            /* the cross-class pair rides the description too: on the node,
             * is0 vs ip IS the y-first-vs-z-first B=1 discrimination even
             * if the 3% incumbency guard keeps the pick on ip */
            if (ok[wb + 12] && pn >= 0 && (size_t)pn < sizeof probe)
                pn += snprintf(probe + pn, sizeof probe - (size_t)pn, " is0%d=%.1f",
                               have512 ? 4 : 2, bestc[wb + 12] * 1e6 / tb);
            /* the transpose-free-vs-shuffle pair IS the ICX port-5 story at
             * this geometry; it rides the description either way the pick goes */
            if (have512 && ok[NM + 13] && pn >= 0 && (size_t)pn < sizeof probe)
                pn += snprintf(probe + pn, sizeof probe - (size_t)pn, " bc4=%.1f",
                               bestc[NM + 13] * 1e6 / tb);
            /* ice_r3: the XV/vrev family vs its bcst0 incumbent -- the
             * pass-B-idle-window story, priced in the chain arena */
            if (have512 && ok[NM + 15] && pn >= 0 && (size_t)pn < sizeof probe)
                pn += snprintf(probe + pn, sizeof probe - (size_t)pn, " xv4=%.1f",
                               bestc[NM + 15] * 1e6 / tb);
            if (have512 && ok[NM + 16] && pn >= 0 && (size_t)pn < sizeof probe)
                pn += snprintf(probe + pn, sizeof probe - (size_t)pn, " xr4=%.1f",
                               bestc[NM + 16] * 1e6 / tb);
            if (have512 && ok[NM + 17] && pn >= 0 && (size_t)pn < sizeof probe)
                pn += snprintf(probe + pn, sizeof probe - (size_t)pn, " vr4=%.1f",
                               bestc[NM + 17] * 1e6 / tb);
            if (have512 && ok[NM + 18] && pn >= 0 && (size_t)pn < sizeof probe)
                pn += snprintf(probe + pn, sizeof probe - (size_t)pn, " nt4=%.1f",
                               bestc[NM + 18] * 1e6 / tb);
            if (have512 && ok[NM + 19] && pn >= 0 && (size_t)pn < sizeof probe)
                snprintf(probe + pn, sizeof probe - (size_t)pn, " nv4=%.1f",
                         bestc[NM + 19] * 1e6 / tb);
            snprintf(g_desc, sizeof g_desc,
                     "L=36 plane-fused y+z then strided x, PFA4x9 interleaved "
                     "lanes, rev order; chain-shaped tuner picked pw=%d mode=%s "
                     "(B=%d)%s",
                     p->pw, mode_name(p->mode), batch, probe);
        }
        /* ice_r3: -DFFT36PF_VERBOSE_CC prints the table too -- env does not
         * cross tryout.sh's ssh (L17_matrixsimd's lesson, via mixedradix) */
#ifdef FFT36PF_VERBOSE_CC
        if (1) {
#else
        if (getenv("FFT36PF_VERBOSE")) {
#endif
            for (int cfg = 0; cfg < 2 * NM; ++cfg)
                if (ok[cfg])
                    fprintf(stderr, "L36_pencilfused tuner: pw=%d %-15s %8.1f us/vol\n",
                            (cfg >= NM) ? 4 : 2, mode_name(cfg % NM),
                            bestc[cfg] * 1e6 / tb);
            fprintf(stderr, "L36_pencilfused tuner: chose pw=%d %s (tb=%ld)\n",
                    p->pw, mode_name(p->mode), tb);
        }
    }
    free(din);
    free(dout);
    free(dpong);
    free(dref);
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->arena);
    free(p);
}

#else
/* ============================== templated kernel body ============================== */
/* A vector holds PW complex numbers = 2*PW doubles, interleaved re,im.               */

typedef double    TS(vd) __attribute__((vector_size(2 * PW * 8)));
typedef long long TS(vi) __attribute__((vector_size(2 * PW * 8)));
typedef double    TS(vu) __attribute__((vector_size(2 * PW * 8), aligned(8), may_alias));

#define vd  TS(vd)
#define vi  TS(vi)
#define vu  TS(vu)
#define LD(p)     (*(const vu *)(const void *)(p))
#define ST(p, v)  (*(vu *)(void *)(p) = (v))
#if PW == 2
#  ifdef __AVX__
#    define STNT(p, v) _mm256_stream_pd((double *)(p), (__m256d)(v))
#  else
#    define STNT(p, v) ST(p, v)
#  endif
#  define VC(a, b) ((vd){ (a), (b), (a), (b) })
#  define SWPM ((vi){ 1, 0, 3, 2 })
#else
#  define STNT(p, v) _mm512_stream_pd((double *)(p), (__m512d)(v))
#  define VC(a, b) ((vd){ (a), (b), (a), (b), (a), (b), (a), (b) })
#  define SWPM ((vi){ 1, 0, 3, 2, 5, 4, 7, 6 })
#endif
#define VS(a)    VC(a, a)
#define CSWAP(v) __builtin_shuffle((v), SWPM)   /* swap re/im in every complex lane */

/* --- PW x PW transpose of complex (128-bit) lanes, butterfly form ------------------ */
#define CTRSTEP(a, i, s) do {                                                          \
    vi ml_, mh_;                                                                      \
    for (int j_ = 0; j_ < 2 * PW; ++j_) {                                             \
        int g_ = j_ / 2, d_ = j_ & 1;                                                 \
        int b_ = g_ / (s), o_ = g_ % (s);                                             \
        long long sl_ = (b_ & 1) ? (PW + (b_ - 1) * (s) + o_) : (b_ * (s) + o_);      \
        ((long long *)&ml_)[j_] = 2 * sl_ + d_;                                       \
        ((long long *)&mh_)[j_] = 2 * (sl_ + (s)) + d_;                               \
    }                                                                                 \
    vd lo_ = __builtin_shuffle((a)[i], (a)[(i) + (s)], ml_);                           \
    vd hi_ = __builtin_shuffle((a)[i], (a)[(i) + (s)], mh_);                           \
    (a)[i] = lo_; (a)[(i) + (s)] = hi_;                                                \
} while (0)

#if PW == 2
#  define CTRANSPOSE(a) do { CTRSTEP(a,0,1); } while (0)
#else
#  define CTRANSPOSE(a) do { CTRSTEP(a,0,1); CTRSTEP(a,2,1);                           \
                             CTRSTEP(a,0,2); CTRSTEP(a,1,2); } while (0)
#endif

/* --- the modules (interleaved complex; constants are lane-splatted pairs) ---------- */

/* Good-Thomas index maps for 36 = 4 * 9:  n = (9 n1 + 4 n2) mod 36,
 * k = (9 k1 + 28 k2) mod 36, whence n k = 9 n1 k1 + 4 n2 k2 (mod 36). */
#define IX(n1,n2)  (((9 * (n1) + 4 * (n2)) % 36))
#define OX(k1,k2)  (((9 * (k1) + 28 * (k2)) % 36))
#define UU(k1,j)   ((k1) * 9 + (j))

/* stage A, one of nine DFT-4 over n1.  y1 = t1 - i*t3 and y3 = t1 + i*t3 via
 * (1,-1)*swap(t3).  8 FMA-port ops + 1 swap.  LOAD(j,X) supplied by the caller. */
#define SA_(n2, LOAD) do {                                                             \
    vd q0, q1, q2, q3;                                                                 \
    LOAD(IX(0,n2), q0)                                                                 \
    LOAD(IX(1,n2), q1)                                                                 \
    LOAD(IX(2,n2), q2)                                                                 \
    LOAD(IX(3,n2), q3)                                                                 \
    vd t0_ = q0 + q2, t1_ = q0 - q2;                                                   \
    vd t2_ = q1 + q3, t3_ = q1 - q3;                                                   \
    u[UU(0,n2)] = t0_ + t2_;                                                           \
    u[UU(2,n2)] = t0_ - t2_;                                                           \
    vd sw_ = CSWAP(t3_);                                                               \
    u[UU(1,n2)] = t1_ + VC(1.0, -1.0) * sw_;                                           \
    u[UU(3,n2)] = t1_ - VC(1.0, -1.0) * sw_;                                           \
} while (0)

/* stage B, one of four DFT-9 over n2 (round panel_r10): genfft's FMA n1_9 DAG
 * (fftw-3.3.10, 24 add + 56 fma = 80 scalar FMA-port ops) transcribed pairwise to
 * interleaved vectors -- each scalar re/im line pair is one vector op; each re/im
 * crossing (mult by i, the (1 +- c*i) spiral factors, and their cross terms) is one
 * CSWAP with the signs folded into a VC pair constant.  40 FMA-port ops + 12 swaps
 * (the hand CT 3x3 it replaces: 44 + 10).  Transcription rule and vector form
 * adopted from L45_pfa r9 (their DFT9F, node-proven at 4.0e-16); plain +/- and *
 * are used so gcc contracts to FMA exactly as in the rest of this file.
 *   Stage A (radix-3 columns {j, j+3, j+6}): sJ_ = column sum, SJ_ = full sum,
 *   aJ_ = xJ - sJ/2, eJ_ = x(J+3) - x(J+6), iJ_ = CSWAP(eJ_);
 *   p/q = aJ +- K3*i*eJ (the two rotated DFT3 outputs).  Blocks: k2 = {0,3,6} is a
 *   DFT3 on the sums; k2 = {1,4,7} and {2,5,8} build w = (1 +- c*i)*p (one
 *   CSWAP+FMA each), cross them (u/z), and fan out through the K984/K492/K852 tail. */
#define SB_(k1, STORE) do {                                                            \
    vd s0_ = u[UU(k1,3)] + u[UU(k1,6)], e0_ = u[UU(k1,3)] - u[UU(k1,6)];               \
    vd S0_ = u[UU(k1,0)] + s0_,  a0_ = u[UU(k1,0)] - VS(0.5) * s0_;                    \
    vd i0_ = CSWAP(e0_);                                                               \
    vd s1_ = u[UU(k1,4)] + u[UU(k1,7)], e1_ = u[UU(k1,4)] - u[UU(k1,7)];               \
    vd S1_ = u[UU(k1,1)] + s1_,  a1_ = u[UU(k1,1)] - VS(0.5) * s1_;                    \
    vd i1_ = CSWAP(e1_);                                                               \
    vd p1_ = a1_ + VC(K3, -K3) * i1_;                                                  \
    vd q1_ = a1_ - VC(K3, -K3) * i1_;                                                  \
    vd s2_ = u[UU(k1,5)] + u[UU(k1,8)], e2_ = u[UU(k1,5)] - u[UU(k1,8)];               \
    vd S2_ = u[UU(k1,2)] + s2_,  a2_ = u[UU(k1,2)] - VS(0.5) * s2_;                    \
    vd i2_ = CSWAP(e2_);                                                               \
    vd p2_ = a2_ + VC(K3, -K3) * i2_;                                                  \
    vd q2_ = a2_ - VC(K3, -K3) * i2_;                                                  \
    /* k2 = 0, 3, 6: DFT3 on the column sums */                                        \
    { vd sg_ = S1_ + S2_, d3_ = S2_ - S1_, id_ = CSWAP(d3_);                           \
      vd b0_ = S0_ - VS(0.5) * sg_;                                                    \
      vd o0 = S0_ + sg_;                                                               \
      vd o3 = b0_ - VC(K3, -K3) * id_;                                                 \
      vd o6 = b0_ + VC(K3, -K3) * id_;                                                 \
      STORE(OX(k1,0), o0) STORE(OX(k1,3), o3) STORE(OX(k1,6), o6) }                    \
    /* k2 = 1, 4, 7 */                                                                 \
    { vd v1_ = a0_ + VC(K3, -K3) * i0_;                                                \
      vd w2_ = p2_ + VC(-K176, K176) * CSWAP(p2_);                                     \
      vd w1_ = p1_ + VC(K839, -K839) * CSWAP(p1_);                                     \
      vd u1_ = CSWAP(w2_) + VC(K777, -K777) * w1_;                                     \
      vd z1_ = w2_ + VC(K777, -K777) * CSWAP(w1_);                                     \
      vd o1 = v1_ + VC(K984, -K984) * u1_;                                             \
      vd r1_ = v1_ - VC(K492, -K492) * u1_;                                            \
      vd o4 = r1_ + VS(K852) * z1_;                                                    \
      vd o7 = r1_ - VS(K852) * z1_;                                                    \
      STORE(OX(k1,1), o1) STORE(OX(k1,4), o4) STORE(OX(k1,7), o7) }                    \
    /* k2 = 2, 5, 8 */                                                                 \
    { vd v2_ = a0_ - VC(K3, -K3) * i0_;                                                \
      vd wA_ = CSWAP(q1_) + VC(K176, -K176) * q1_;                                     \
      vd wB_ = q2_ - VC(K363, -K363) * CSWAP(q2_);                                     \
      vd uB_ = wA_ + VC(-K954, K954) * wB_;                                            \
      vd zB_ = CSWAP(wA_) + VC(-K954, K954) * CSWAP(wB_);                              \
      vd o2 = v2_ + VC(K984, -K984) * uB_;                                             \
      vd rB_ = v2_ - VC(K492, -K492) * uB_;                                            \
      vd o5 = rB_ - VS(K852) * zB_;                                                    \
      vd o8 = rB_ + VS(K852) * zB_;                                                    \
      STORE(OX(k1,2), o2) STORE(OX(k1,5), o5) STORE(OX(k1,8), o8) }                    \
} while (0)

/* the whole 36-point line: 232 FMA-port ops + 57 shuffles over PW lanes
 * (9 DFT4 x (8+1) + 4 n1_9 DFT9 x (40+12); was 248 + 49 with the CT 3x3 DFT9).
 * ALL 36 loads happen in the SA_ stage, before the first SB_ store, so LOAD and
 * STORE may alias -- pass B relies on this for its in-place mode. */
#define PFA36(LOAD, STORE) do {                                                        \
    vd u[36];                                                                          \
    SA_(0,LOAD); SA_(1,LOAD); SA_(2,LOAD); SA_(3,LOAD); SA_(4,LOAD);                   \
    SA_(5,LOAD); SA_(6,LOAD); SA_(7,LOAD); SA_(8,LOAD);                                \
    SB_(0,STORE); SB_(1,STORE); SB_(2,STORE); SB_(3,STORE);                            \
} while (0)

/* TWO independent line-groups, stage-interleaved so their DFT3/DFT4 latency
 * chains dovetail in the scheduler (mixedradix r1 item 2, on every L=36 Next
 * list since round 1, never measured by anyone -- this is the measurement
 * vehicle).  SA_/SB_ index through the name `u`, so aliasing it to each
 * group's buffer in turn interleaves whole stages.  DIAGNOSTIC ONLY
 * (-DFFT36PF_PAIRB): doubles the live set, so at PW=4 gcc will spill hard. */
#define PFA36X2(LOAD0, STORE0, LOAD1, STORE1) do {                                     \
    vd u0_[36], u1_[36]; vd *u;                                                        \
    u = u0_; SA_(0,LOAD0);  u = u1_; SA_(0,LOAD1);                                     \
    u = u0_; SA_(1,LOAD0);  u = u1_; SA_(1,LOAD1);                                     \
    u = u0_; SA_(2,LOAD0);  u = u1_; SA_(2,LOAD1);                                     \
    u = u0_; SA_(3,LOAD0);  u = u1_; SA_(3,LOAD1);                                     \
    u = u0_; SA_(4,LOAD0);  u = u1_; SA_(4,LOAD1);                                     \
    u = u0_; SA_(5,LOAD0);  u = u1_; SA_(5,LOAD1);                                     \
    u = u0_; SA_(6,LOAD0);  u = u1_; SA_(6,LOAD1);                                     \
    u = u0_; SA_(7,LOAD0);  u = u1_; SA_(7,LOAD1);                                     \
    u = u0_; SA_(8,LOAD0);  u = u1_; SA_(8,LOAD1);                                     \
    u = u0_; SB_(0,STORE0); u = u1_; SB_(0,STORE1);                                    \
    u = u0_; SB_(1,STORE0); u = u1_; SB_(1,STORE1);                                    \
    u = u0_; SB_(2,STORE0); u = u1_; SB_(2,STORE1);                                    \
    u = u0_; SB_(3,STORE0); u = u1_; SB_(3,STORE1);                                    \
} while (0)

/* --- pass A over ONE x-plane, streaming (z-first, transpose-on-load) variant ----
 * A1: z-transform, lanes = PW y-rows via transpose-on-load, output transposed
 * back into P[y][kz].  The transpose-on-load order keeps the cold `in` reads
 * SEQUENTIAL in PW streams (adopted from L36_pfa r2 phase 1): the y-first order
 * read the plane through 36 stride-576B streams and measured 101 vs 58 us/vol
 * at B=256 cold.  A2: y-transform, lanes = kz (P rows contiguous, shuffle-free),
 * stored straight to the mid plane -- 36 scattered 64-B store streams, which
 * the store buffer absorbs (scattered LOADS were the r2 mistake).
 *
 * Prefetch (panel_r6, byte-faithful port of L36_pfa's PFIN/PFWMID pacing):
 * pfc, when non-null, is a paced READ cursor already offset FFT36PF_PFD
 * doubles ahead of this plane; pwc, when non-null (mode 8 only), is a paced
 * WRITE-INTENT cursor offset FFT36PF_PFWD ahead of this plane's rdst stores.
 * Each of the 2*(36/PW) loop iterations advances its cursor by PFSTEP =
 * 36*PW doubles, so exactly one plane's worth of prefetches issues per plane
 * processed, spread evenly over BOTH subloops -- the second subloop touches
 * no rsrc bytes, so pacing through it keeps the DRAM streams busy during the
 * y-transform too (the r5 scheme issued only during the first subloop).
 * Cursors may run past the volume end; prefetches never fault. ---------------- */
#define PFSTEP (36 * PW)
#define PFRD(p) do {                                                                  \
    for (int q_ = 0; q_ < PFSTEP / 8; ++q_)                                           \
        __builtin_prefetch((p) + 8 * q_, 0, 2);                                       \
} while (0)
#define PFWR(p) do {                                                                  \
    for (int q_ = 0; q_ < PFSTEP / 8; ++q_)                                           \
        __builtin_prefetch((p) + 8 * q_, 1, 3);                                       \
} while (0)
/* NTA read: 2*PFSTEP doubles per issue = one first-subloop iteration's
 * consumption (the first subloop reads ALL of rsrc; the second reads none),
 * locality 0 = prefetchnta -- fills L1, bypasses L2 on SKX-class cores. */
#define PFNTA(p) do {                                                                 \
    for (int q_ = 0; q_ < PFSTEP / 4; ++q_)                                           \
        __builtin_prefetch((p) + 8 * q_, 0, 0);                                       \
} while (0)

/* ntac (mode 9 only): CONSTANT-lead NTA cursor over rsrc, already offset
 * FFT36PF_PFDN doubles ahead; advances at exactly the first subloop's
 * consumption rate and issues NOTHING in the second subloop (pfa r6's NTA
 * pacing discipline -- a swinging lead drops quick-evict L1 lines).  When
 * ntac is set the caller passes pfc = pwc = 0: the whole point is keeping
 * the read stream out of L2. */
TATTR static void TS(passA_plane)(const double *restrict rsrc, double *restrict rdst,
                                  double *restrict pp,
                                  const double *restrict pfc, double *restrict pwc,
                                  const double *restrict ntac)
{
#ifdef FFT36PF_NOPAPF
    pfc = 0;
#endif
    for (int yb = 0; yb < 36; yb += PW) {
        if (pfc) { PFRD(pfc); pfc += PFSTEP; }
        if (pwc) { PFWR(pwc); pwc += PFSTEP; }
        if (ntac) { PFNTA(ntac); ntac += 2 * PFSTEP; }
        vd Zv[36], Wv[36];
        for (int zb = 0; zb < 36 / PW; ++zb) {
            vd t[PW];
            for (int j = 0; j < PW; ++j)
                t[j] = LD(rsrc + 2 * ((size_t)(yb + j) * 36 + (size_t)zb * PW));
            CTRANSPOSE(t);
            for (int l = 0; l < PW; ++l)
                Zv[zb * PW + l] = t[l];
        }
#define ZLOAD(j, X)  { (X) = Zv[j]; }
#define ZSTORE(k, X) { Wv[k] = (X); }
        PFA36(ZLOAD, ZSTORE);
#undef ZLOAD
#undef ZSTORE
        for (int kb = 0; kb < 36 / PW; ++kb) {
            vd t[PW];
            for (int i = 0; i < PW; ++i) t[i] = Wv[kb * PW + i];
            CTRANSPOSE(t);
            for (int l = 0; l < PW; ++l)
                ST(pp + 2 * ((size_t)(yb + l) * PST + (size_t)kb * PW), t[l]);
        }
    }
    for (int kb = 0; kb < 36 / PW; ++kb) {
        if (pfc) { PFRD(pfc); pfc += PFSTEP; }
        if (pwc) { PFWR(pwc); pwc += PFSTEP; }
        const double *s = pp + 2 * (size_t)kb * PW;
#define YLOAD(j, X)  { (X) = LD(s + 2 * ((size_t)(j) * PST)); }
#define YSTORE(k, X) { ST(rdst + 2 * ((size_t)(k) * 36 + (size_t)kb * PW), X); }
        PFA36(YLOAD, YSTORE);
#undef YLOAD
#undef YSTORE
    }
}

/* --- pass A, transpose-FREE broadcast variant (ice_r2; adopted from L36_pfa's
 * ice_r2 tr=1 kernel, translated onto this file's pass shapes) -----------------
 * On ICX the second 512-bit FMA pipe lives on PORT 5, the same port that runs
 * every 512-bit shuffle, so the PW x PW register transposes (1296 port-5 ops
 * per plane) now displace FMA work they used to hide under (corpus S10; the
 * ice_r1 verdict's L=17 port-5 mechanism).  This variant assembles every
 * lane-transposed input vector as 4 MASKED 128-BIT BROADCAST LOADS instead --
 * vbroadcastf64x2 (mem), zmm{k} is one load-port uop, zero port-5 -- and both
 * subloops then STORE plain full vectors:
 *   subloop A: z-transform, lanes = y.  LOAD(j) = element (y=yb+i, z=j) of the
 *              in-plane, 4 broadcasts at row stride 72 doubles; STORE(k) =
 *              plain vector to pp[kz=k][y=yb..yb+3] (row stride PST = 36 cplx).
 *   subloop B: y-transform, lanes = kz.  LOAD(j) = element (y=j, kz=zg+i) of
 *              pp, 4 broadcasts at the same 72-double row stride; STORE(k) =
 *              plain vector to the out plane row ky=k, cols kz=zg..zg+3.
 * No staging arrays (loads sit directly inside SA_), no transposes anywhere:
 * per plane this deletes all 1296 transpose shuffles and both 36-vd staging
 * arrays (the spill surgery r11 deferred), for 3x more load uops on ports the
 * pass leaves idle (2x64B loads/cyc on this bare-metal part, corpus S10).
 * The 4-broadcast merge chain is 4 loads deep per vector; 36 independent
 * vectors per PFA36 call keep the OOO window full (L13_rader's CLX loss on
 * 8-deep merge chains does not apply at depth 4 -- and L36_pfa verified this
 * exact builder compiles to 4 vbroadcastf64x2-from-memory, kmovs hoisted).
 * z-then-y order = the 3.586e-16 bit-class (same as modes 7/8/12).
 * pfc, when non-null, is the paced T1 read cursor exactly as in passA_plane
 * (18 iterations x PFSTEP doubles = one plane per plane processed).
 * ntac (ice_r3, modes 18/19), when non-null, is the CONSTANT-lead prefetchnta
 * cursor over rsrc, already offset FFT36PF_PFDN doubles ahead: subloop A's
 * iteration yb consumes rows yb..yb+3 = 2304 sequential bytes, so each
 * iteration issues exactly one iteration's worth (36 lines) at a constant
 * 4 KB lead -- pfa r6's NTA pacing discipline (a swinging lead drops
 * quick-evict L1 lines).  Subloop B reads only pp and issues nothing.
 * PW=4 only: at PW=2 the transposes are 2 shuffles per 2 vectors and PW=2
 * loses every tournament anyway (exec_v2 falls back to passA_plane). ---------- */
#if PW == 4
TATTR static void TS(passA_bcst)(const double *restrict rsrc, double *restrict rdst,
                                 double *restrict pp, const double *restrict pfc,
                                 const double *restrict ntac)
{
    for (int yb = 0; yb < 36; yb += 4) {
        if (pfc) { PFRD(pfc); pfc += PFSTEP; }
        if (ntac) { PFNTA(ntac); ntac += 2 * PFSTEP; }
        const double *s = rsrc + 2 * (size_t)yb * 36;
#define BCLD(j, X) {                                                            \
        __m512d x_ = _mm512_broadcast_f64x2(_mm_load_pd(s + 2 * (size_t)(j)));  \
        x_ = _mm512_mask_broadcast_f64x2(x_, 0x0C,                              \
                 _mm_load_pd(s + 2 * (size_t)(j) + 72));                        \
        x_ = _mm512_mask_broadcast_f64x2(x_, 0x30,                              \
                 _mm_load_pd(s + 2 * (size_t)(j) + 144));                       \
        x_ = _mm512_mask_broadcast_f64x2(x_, 0xC0,                              \
                 _mm_load_pd(s + 2 * (size_t)(j) + 216));                       \
        (X) = (vd)x_; }
#define BCST(k, X) { ST(pp + 2 * ((size_t)(k) * PST + yb), X); }
        PFA36(BCLD, BCST);
#undef BCLD
#undef BCST
    }
    for (int zg = 0; zg < 36; zg += 4) {
        if (pfc) { PFRD(pfc); pfc += PFSTEP; }
        const double *s2 = pp + 2 * (size_t)zg * PST;
#define BCLD2(j, X) {                                                           \
        __m512d x_ = _mm512_broadcast_f64x2(_mm_load_pd(s2 + 2 * (size_t)(j))); \
        x_ = _mm512_mask_broadcast_f64x2(x_, 0x0C,                              \
                 _mm_load_pd(s2 + 2 * (size_t)(j) + 72));                       \
        x_ = _mm512_mask_broadcast_f64x2(x_, 0x30,                              \
                 _mm_load_pd(s2 + 2 * (size_t)(j) + 144));                      \
        x_ = _mm512_mask_broadcast_f64x2(x_, 0xC0,                              \
                 _mm_load_pd(s2 + 2 * (size_t)(j) + 216));                      \
        (X) = (vd)x_; }
#define BCST2(k, X) { ST(rdst + 2 * ((size_t)(k) * 36 + zg), X); }
        PFA36(BCLD2, BCST2);
#undef BCLD2
#undef BCST2
    }
}
#endif /* PW == 4 */

/* --- pass B, cached-store path, flat groups [g0,g1).  NO restrict: INPLACE mode
 * calls this with mid == outv (PFA36 loads everything before its first store,
 * so aliasing is well-defined).  wpf adds a write-intent prefetch on the dst
 * streams 4 lines ahead (SCRATCH mode only, where dst is cold and demand-RFO
 * would serialize; adopted from L6_unrolled's prefetchw, node-validated).
 * nxt (ISTREAM mode) pre-covers the start of the NEXT volume's input: 3 lines
 * per line group, T1, so pass A of v+1 never starts cold -- L36_pfa's PFNX
 * trick (their r4, node-selected pf=1); 324 groups x 3 lines ~ 62 KB at PW=4. */
TATTR static void TS(passB_cached)(const double *mid, double *outv,
                                   int g0, int g1, int wpf,
                                   const double *restrict nxt)
{
#ifdef FFT36PF_PAIRB
    /* diagnostic: two stage-interleaved groups per iteration (group counts
     * NPLANE/PW = 324 or 648 are even, and every caller passes even g0/g1) */
    for (int g = g0; g < g1; g += 2) {
        const double *src = mid  + 2 * (size_t)g * PW;
        double       *dst = outv + 2 * (size_t)g * PW;
        for (int j_ = 0; j_ < 36; ++j_) {
            __builtin_prefetch(src + (size_t)j_ * (2 * NPLANE) + 8, 0, 3);
            __builtin_prefetch(src + (size_t)j_ * (2 * NPLANE) + 2 * PW + 8, 0, 3);
        }
        if (wpf)
            for (int j_ = 0; j_ < 36; ++j_) {
                __builtin_prefetch(dst + (size_t)j_ * (2 * NPLANE) + 32, 1, 3);
                __builtin_prefetch(dst + (size_t)j_ * (2 * NPLANE) + 2 * PW + 32, 1, 3);
            }
        if (nxt)
            for (int j_ = 0; j_ < 6; ++j_)
                __builtin_prefetch(nxt + ((size_t)g * 3 + j_) * 8, 0, 2);
#define B0LOAD(j, X)  { (X) = LD(src + (size_t)(j) * (2 * NPLANE)); }
#define B0STORE(k, X) { ST(dst + (size_t)(k) * (2 * NPLANE), X); }
#define B1LOAD(j, X)  { (X) = LD(src + (size_t)(j) * (2 * NPLANE) + 2 * PW); }
#define B1STORE(k, X) { ST(dst + (size_t)(k) * (2 * NPLANE) + 2 * PW, X); }
        PFA36X2(B0LOAD, B0STORE, B1LOAD, B1STORE);
#undef B0LOAD
#undef B0STORE
#undef B1LOAD
#undef B1STORE
    }
    return;
#endif
    for (int g = g0; g < g1; ++g) {
        const double *src = mid  + 2 * (size_t)g * PW;
        double       *dst = outv + 2 * (size_t)g * PW;
        for (int j_ = 0; j_ < 36; ++j_)
            __builtin_prefetch(src + (size_t)j_ * (2 * NPLANE) + 8, 0, 3);
        if (wpf)
            for (int j_ = 0; j_ < 36; ++j_)
                __builtin_prefetch(dst + (size_t)j_ * (2 * NPLANE) + 32, 1, 3);
        if (nxt)
            for (int j_ = 0; j_ < 3; ++j_)
                __builtin_prefetch(nxt + ((size_t)g * 3 + j_) * 8, 0, 2);
#define BLOAD(j, X)  { (X) = LD(src + (size_t)(j) * (2 * NPLANE)); }
#define BSTORE(k, X) { ST(dst + (size_t)(k) * (2 * NPLANE), X); }
        PFA36(BLOAD, BSTORE);
#undef BLOAD
#undef BSTORE
    }
}

/* --- pass B cached in place + CROSS-VOLUME L2 STAGING (ice_r3, modes 15/16).
 * This pass is compute-bound on the L2-resident volume -- every out line is
 * loaded once, stored once, then dead until the driver's scale pass -- so the
 * memory system idles for its whole ~20 us.  Fill that window: prefetcht1 the
 * NEXT volume's input, 9*PW lines per line group x (NPLANE/PW) groups =
 * exactly one volume, so pass A of the next volume reads L2 instead of dirty
 * L3.  The lines it evicts are this volume's already-processed dead rows.
 * Mode 3's XV failure mode (prefetches dropped while NT drains hold the fill
 * buffers) cannot occur: no NT stores, and the demand traffic is L2 hits.
 * NO restrict on mid/outv: called with mid == outv (in place). ---------------- */
TATTR static void TS(passB_xv)(const double *mid, double *outv,
                               const double *restrict xvn)
{
    for (int g = 0; g < NPLANE / PW; ++g) {
        const double *src = mid  + 2 * (size_t)g * PW;
        double       *dst = outv + 2 * (size_t)g * PW;
        for (int j_ = 0; j_ < 36; ++j_)
            __builtin_prefetch(src + (size_t)j_ * (2 * NPLANE) + 8, 0, 3);
        for (int j_ = 0; j_ < 9 * PW; ++j_)
            __builtin_prefetch(xvn + ((size_t)g * (9 * PW) + j_) * 8, 0, 2);
#define BLOAD(j, X)  { (X) = LD(src + (size_t)(j) * (2 * NPLANE)); }
#define BSTORE(k, X) { ST(dst + (size_t)(k) * (2 * NPLANE), X); }
        PFA36(BLOAD, BSTORE);
#undef BLOAD
#undef BSTORE
    }
}

/* --- pass B, NT-store path, over UNITS [u0,u1) where a unit is one flat group
 * at PW=4 and one PAIR of flat groups at PW=2 (32-B NT stores are half a cache
 * line; pairing completes every 64-B line back-to-back -- L36_pfa/mixedradix
 * r2 trick).  Either way there are exactly 324 units per volume, 9 per output
 * x-slot, which is what the PIPE interleave relies on.  nxt (mode 3 only)
 * prefetches the next volume's input, 36 lines per unit = one volume total.
 * pfd is the src prefetch distance in doubles: 8 (one line) when mid is
 * L2-resident (modes 2/3), 16 when mid lives in L3 (PIPE). ------------------- */
TATTR static void TS(passB_nt)(const double *restrict mid, double *restrict outv,
                               int u0, int u1, const double *restrict nxt, int pfd)
{
#if PW == 4
    for (int g = u0; g < u1; ++g) {
        const double *src = mid  + 2 * (size_t)g * PW;
        double       *dst = outv + 2 * (size_t)g * PW;
        for (int j_ = 0; j_ < 36; ++j_)
            __builtin_prefetch(src + (size_t)j_ * (2 * NPLANE) + pfd, 0, 3);
        /* XV: stage the NEXT volume's input into L3 while this pass is
         * store-drain-bound.  324 units x 36 lines = the whole volume. */
        if (nxt)
            for (int j_ = 0; j_ < 36; ++j_)
                __builtin_prefetch(nxt + ((size_t)g * 36 + j_) * 8, 0, 1);
#define BLOAD(j, X)    { (X) = LD(src + (size_t)(j) * (2 * NPLANE)); }
#define BSTORENT(k, X) { STNT(dst + (size_t)(k) * (2 * NPLANE), X); }
        PFA36(BLOAD, BSTORENT);
#undef BLOAD
#undef BSTORENT
    }
#else
    for (int gp = u0; gp < u1; ++gp) {
        const double *src = mid  + 4 * (size_t)gp * PW;
        double       *dst = outv + 4 * (size_t)gp * PW;
        for (int j_ = 0; j_ < 36; ++j_)
            __builtin_prefetch(src + (size_t)j_ * (2 * NPLANE) + pfd, 0, 3);
        if (nxt)
            for (int j_ = 0; j_ < 36; ++j_)
                __builtin_prefetch(nxt + ((size_t)gp * 36 + j_) * 8, 0, 1);
        vd Wa[36], Wb[36];
#define BLOADA(j, X) { (X) = LD(src + (size_t)(j) * (2 * NPLANE)); }
#define BSTA(k, X)   { Wa[k] = (X); }
        PFA36(BLOADA, BSTA);
#undef BLOADA
#undef BSTA
#define BLOADB(j, X) { (X) = LD(src + (size_t)(j) * (2 * NPLANE) + 2 * PW); }
#define BSTB(k, X)   { Wb[k] = (X); }
        PFA36(BLOADB, BSTB);
#undef BLOADB
#undef BSTB
        for (int k_ = 0; k_ < 36; ++k_) {
            STNT(dst + (size_t)k_ * (2 * NPLANE),          Wa[k_]);
            STNT(dst + (size_t)k_ * (2 * NPLANE) + 2 * PW, Wb[k_]);
        }
    }
#endif
}

/* --- pass B in place on mid, with an optional interleaved SEQUENTIAL NT copy of
 * ANOTHER (already fully transformed) volume csrc -> cdst: 36 vectors' worth of
 * copy per line group = 9*PW lines, so the copy finishes exactly with the pass.
 * The pass itself is pure cache-resident compute (loads and stores both on mid),
 * so the interleaved NT drain rides on an otherwise idle memory system. -------- */
TATTR static void TS(passB_copy)(double *mid, const double *restrict csrc,
                                 double *restrict cdst)
{
    for (int g = 0; g < NPLANE / PW; ++g) {
        const double *src = mid + 2 * (size_t)g * PW;
        double       *dst = mid + 2 * (size_t)g * PW;
        for (int j_ = 0; j_ < 36; ++j_)
            __builtin_prefetch(src + (size_t)j_ * (2 * NPLANE) + 8, 0, 3);
        if (csrc) {
            const double *cs = csrc + (size_t)g * (9 * PW) * 8;
            double       *cd = cdst + (size_t)g * (9 * PW) * 8;
            for (int c = 0; c < 36; ++c)
                STNT(cd + (size_t)c * (2 * PW), LD(cs + (size_t)c * (2 * PW)));
        }
#define BLOAD(j, X)  { (X) = LD(src + (size_t)(j) * (2 * NPLANE)); }
#define BSTORE(k, X) { ST(dst + (size_t)(k) * (2 * NPLANE), X); }
        PFA36(BLOAD, BSTORE);
#undef BLOAD
#undef BSTORE
    }
}

/* sequential NT copy of one whole volume (the pipeline tail / mode-5 phase 3) */
TATTR static void TS(seqcopy_nt)(const double *restrict csrc, double *restrict cdst)
{
    for (size_t c = 0; c < NVOL2 / (2 * PW); ++c)
        STNT(cdst + c * (2 * PW), LD(csrc + c * (2 * PW)));
}

/* --- mode 11 (INPLACE-CS) compact-code twins, new in panel_r9 ---------------------
 * halfplane: ONE shared body for BOTH subloops of mode 0's y-first pass A.  They
 * are the same code -- each loads PW lanes at stride 72 doubles (36 complex) from
 * src + 2*g*PW, runs PFA36, and stores PWxPW-transposed blocks to dst rows of
 * stride 72 doubles, because the plane buffer's row stride PST == 36 == LL.
 * y-subloop == halfplane(in-plane, pp); z-subloop == halfplane(pp, out-plane).
 * Under the node's flags mode 0's x-plane loop body is two DISTINCT ~610-
 * instruction copies of this (~6.9 KB alternating) -- marginal-to-over the CLX
 * DSB's ~1.5k uops; sharing one noinline body halves the hot footprint with ZERO
 * arithmetic change (output bit-identical to mode 0).  The `unroll 1` fence keeps
 * -funroll-loops from re-inflating it (the PW-wide transpose loops below still
 * unroll fully -- they must, so t[] stays in registers).  noinline is the point:
 * both call sites must execute the SAME cached code. ------------------------------ */
TATTR __attribute__((noinline)) static void TS(halfplane)(const double *restrict src,
                                                          double *restrict dst)
{
#pragma GCC unroll 1
    for (int g = 0; g < 36 / PW; ++g) {
        const double *s = src + 2 * (size_t)g * PW;
        vd yv[36];
#define HLOAD(j, X)  { (X) = LD(s + (size_t)(j) * 72); }
#define HSTORE(k, X) { yv[k] = (X); }
        PFA36(HLOAD, HSTORE);
#undef HLOAD
#undef HSTORE
        for (int c = 0; c < 36 / PW; ++c) {
            vd t[PW];
            for (int i = 0; i < PW; ++i) t[i] = yv[c * PW + i];
            CTRANSPOSE(t);
            for (int l = 0; l < PW; ++l)
                ST(dst + 2 * ((size_t)(g * PW + l) * 36 + c * PW), t[l]);
        }
    }
}

/* passB_small: compact twin of the mode-0 cached pass B (wpf = 0, nxt = 0 folded
 * out; runs in place, mid == outv, so NO restrict -- PFA36 loads a whole line
 * group before its first store).  The group loop is fenced against unrolling:
 * -funroll-loops doubles the ~470-instruction group body past the DSB.  The
 * one-line-ahead prefetch burst stays unrolled (load-bearing: L36_pfa measured
 * removing it at 14% at B=1).  Same arithmetic order as passB_cached. ------------- */
TATTR __attribute__((noinline)) static void TS(passB_small)(const double *mid,
                                                            double *outv)
{
#pragma GCC unroll 1
    for (int g = 0; g < NPLANE / PW; ++g) {
        const double *src = mid  + 2 * (size_t)g * PW;
        double       *dst = outv + 2 * (size_t)g * PW;
        for (int j_ = 0; j_ < 36; ++j_)
            __builtin_prefetch(src + (size_t)j_ * (2 * NPLANE) + 8, 0, 3);
#define BLOAD(j, X)  { (X) = LD(src + (size_t)(j) * (2 * NPLANE)); }
#define BSTORE(k, X) { ST(dst + (size_t)(k) * (2 * NPLANE), X); }
        PFA36(BLOAD, BSTORE);
#undef BLOAD
#undef BSTORE
    }
}

/* --- the driver ------------------------------------------------------------------ */
TATTR static void TS(exec)(const double *restrict in, double *restrict out,
                           double *restrict mids, double *restrict pp,
                           long nvol, int mode)
{
    if (mode == 11) {
        /* INPLACE-CS: mode 0's exact arithmetic through the shared-body
         * compact functions (bit-identical output; see halfplane above).
         * ice_r2: volumes and planes run in REVERSE (anti-phase with the
         * driver's forward chain-scale pass; see the header).  Per-plane
         * work is independent, so the output is bit-identical either way. */
        for (long vv = 0; vv < nvol; ++vv) {
#ifdef FFT36PF_FWD
            const long v = vv;
#else
            const long v = nvol - 1 - vv;
#endif
            const double *inv  = in  + (size_t)v * NVOL2;
            double       *outv = out + (size_t)v * NVOL2;
#ifndef FFT36PF_SKIPA
            for (int xi = 0; xi < 36; ++xi) {
#ifdef FFT36PF_FWD
                const int x = xi;
#else
                const int x = 35 - xi;
#endif
                TS(halfplane)(inv + (size_t)x * 2 * NPLANE, pp);
                TS(halfplane)(pp, outv + (size_t)x * 2 * NPLANE);
            }
#endif
#ifndef FFT36PF_SKIPB
            TS(passB_small)(outv, outv);
#endif
        }
        return;
    }

    if (mode == 5) {
        /* SEQNT, phase-serial: A (in -> mid), B in place on mid, sequential NT
         * copy mid -> out.  One mid; the extra volume round trip stays in cache. */
        for (long v = 0; v < nvol; ++v) {
            const double *inv  = in  + (size_t)v * NVOL2;
            double       *outv = out + (size_t)v * NVOL2;
#ifndef FFT36PF_SKIPA
            for (int x = 0; x < 36; ++x)
                TS(passA_plane)(inv + (size_t)x * 2 * NPLANE,
                                mids + (size_t)x * 2 * NPLANE, pp,
                                inv + FFT36PF_PFD + (size_t)x * 2 * NPLANE,
                                (double *)0, (const double *)0);
#endif
#ifndef FFT36PF_SKIPB
            TS(passB_copy)(mids, (const double *)0, (double *)0);
#endif
            TS(seqcopy_nt)(mids, outv);
        }
        _mm_sfence();
        return;
    }

    if (mode == 6) {
        /* PIPESEQ: per volume, pass A (cold reads under its own compute via the
         * plane-ahead prefetch), then pass B in place with the PREVIOUS volume's
         * sequential NT copy interleaved into it.  Last volume's copy runs bare. */
        double *bufa = mids, *bufb = mids + MIDSKIP;
        for (long v = 0; v < nvol; ++v) {
            const double *inv = in + (size_t)v * NVOL2;
            double *nb = (v & 1) ? bufb : bufa;      /* mid for volume v        */
            double *pb = (v & 1) ? bufa : bufb;      /* holds transformed v-1   */
#ifndef FFT36PF_SKIPA
            for (int x = 0; x < 36; ++x)
                TS(passA_plane)(inv + (size_t)x * 2 * NPLANE,
                                nb + (size_t)x * 2 * NPLANE, pp,
                                inv + FFT36PF_PFD + (size_t)x * 2 * NPLANE,
                                (double *)0, (const double *)0);
#endif
#ifndef FFT36PF_SKIPB
            TS(passB_copy)(nb, v > 0 ? pb : (const double *)0,
                           v > 0 ? out + (size_t)(v - 1) * NVOL2 : (double *)0);
#endif
        }
        TS(seqcopy_nt)((nvol & 1) ? bufa : bufb, out + (size_t)(nvol - 1) * NVOL2);
        _mm_sfence();
        return;
    }

    if (mode == 4) {
        /* PIPE: pass A of volume v+1 interleaved with pass B (NT) of volume v
         * at plane granularity -- one pass-A plane (cold sequential DRAM reads
         * + plane-ahead T1 prefetch), then 9 pass-B units (NT store drains).
         * The demand reads CANNOT be dropped the way mode 3's prefetches can,
         * so the memory system always holds both reads and writes in flight.
         * Two ping-pong mid buffers; combined live set stays ~1 volume, but
         * the buffer being read is LRU-old, so its lines come from L3 (hence
         * pfd=16, two lines ahead, on the mid streams). */
        double *bufa = mids, *bufb = mids + MIDSKIP;
#ifndef FFT36PF_SKIPA
        for (int x = 0; x < 36; ++x)
            TS(passA_plane)(in + (size_t)x * 2 * NPLANE,
                            bufa + (size_t)x * 2 * NPLANE, pp,
                            in + FFT36PF_PFD + (size_t)x * 2 * NPLANE,
                            (double *)0, (const double *)0);
#endif
        for (long v = 0; v < nvol; ++v) {
            double       *cur  = (v & 1) ? bufb : bufa;
            double       *nmid = (v & 1) ? bufa : bufb;
            double       *outv = out + (size_t)v * NVOL2;
            const double *nin  = (v + 1 < nvol)
                                 ? in + (size_t)(v + 1) * NVOL2 : (const double *)0;
            for (int x = 0; x < 36; ++x) {
#ifndef FFT36PF_SKIPA
                if (nin)
                    TS(passA_plane)(nin + (size_t)x * 2 * NPLANE,
                                    nmid + (size_t)x * 2 * NPLANE, pp,
                                    nin + FFT36PF_PFD + (size_t)x * 2 * NPLANE,
                                    (double *)0, (const double *)0);
#endif
#ifndef FFT36PF_SKIPB
                TS(passB_nt)(cur, outv, x * 9, x * 9 + 9, (const double *)0, 16);
#endif
            }
        }
        _mm_sfence();
        return;
    }

    const int nt = (mode == 2 || mode == 3);
    for (long vv = 0; vv < nvol; ++vv) {
        /* ice_r2 anti-phase order: mode 0 walks volumes 7 -> 0 against the
         * driver's forward scale pass (header item 4).  Cursor/PFNX modes
         * keep forward order -- their pacing assumes it.  ice_r3: modes
         * 16/17 reverse the VOLUME order only (planes stay forward, so each
         * volume's cold read remains one ascending sequential stream). */
#ifdef FFT36PF_FWD
        const long v = vv;
        const long vnx = v + 1;
#else
        const int  vrev = (mode == 0 || mode == 16 || mode == 17 || mode == 19);
        const long v    = vrev ? nvol - 1 - vv : vv;
        const long vnx  = vrev ? v - 1 : v + 1;    /* next PROCESSED volume */
#endif
        const double *inv  = in  + (size_t)v * NVOL2;
        double       *outv = out + (size_t)v * NVOL2;
        double       *mid  = (mode == 0 || mode >= 7) ? outv : mids;
        const double *nxt  = ((mode == 3 || mode == 7 || mode == 8 || mode == 14)
                              && v + 1 < nvol)
                             ? in + (size_t)(v + 1) * NVOL2 : (const double *)0;
        /* XV staging target (modes 15/16): the input volume pass A reads next */
        const double *xvn  = ((mode == 15 || mode == 16)
                              && vnx >= 0 && vnx < nvol)
                             ? in + (size_t)vnx * NVOL2 : (const double *)0;

#ifndef FFT36PF_SKIPA
        /* ------- pass A: two transforms fused over one L1-resident x-plane ------- */
        if (mode == 0 || mode == 10) {
        /* INPLACE (small-batch) variant: y-transform first, lanes = z, so the
         * kernel's loads come straight off the plane with no staging array and
         * no load-side shuffles -- the register-pressure-friendly order.  The
         * plane is cache-warm at small batch, so the scattered (stride-576B)
         * load pattern costs nothing here; at streaming batch it costs ~2x
         * (101 vs 58 us/vol measured at B=256 cold), hence the else-branch.
         * Mode 10 adds the column-order NTA read cursor: iteration zg consumes
         * line-column zg*PW/8 of the plane's 36x9 line grid, so prefetchnta
         * line-column +2 (constant 4.5 KB lead), wrapping into the next plane
         * so every line of the volume is issued exactly once. */
        for (int xi = 0; xi < 36; ++xi) {
            /* ice_r2: mode 0 walks planes 35 -> 0 (anti-phase, header item
             * 4); mode 10 keeps forward order for its column NTA cursor. */
#ifdef FFT36PF_FWD
            const int x = xi;
#else
            const int x = (mode == 0) ? 35 - xi : xi;
#endif
            const double *rsrc = inv  + (size_t)x * 2 * NPLANE;
            double       *rdst = outv + (size_t)x * 2 * NPLANE;

            /* y-transform (lanes = z), transposed on the way into P[z][ky] */
            for (int zg = 0; zg < 36 / PW; ++zg) {
                const double *s = rsrc + 2 * (size_t)zg * PW;
                if (mode == 10 && (PW == 4 || (zg & 1) == 0)) {
                    const int c_ = zg * PW / 4 + 2;      /* target line-column */
                    const double *cb_ = (c_ < 9)
                        ? rsrc + 8 * (size_t)c_
                        : rsrc + 2 * NPLANE + 8 * (size_t)(c_ - 9);
                    for (int j_ = 0; j_ < 36; ++j_)
                        __builtin_prefetch(cb_ + (size_t)j_ * 72, 0, 0);
                }
                vd yv[36];
#define YLOAD(j, X)  { (X) = LD(s + (size_t)(j) * 72); }
#define YSTORE(k, X) { yv[k] = (X); }
                PFA36(YLOAD, YSTORE);
#undef YLOAD
#undef YSTORE
                for (int kg = 0; kg < 36 / PW; ++kg) {
                    vd t[PW];
                    for (int i = 0; i < PW; ++i) t[i] = yv[kg * PW + i];
                    CTRANSPOSE(t);
                    for (int l = 0; l < PW; ++l)
                        ST(pp + 2 * ((size_t)(zg * PW + l) * PST + kg * PW), t[l]);
                }
            }

            /* z-transform (lanes = ky), transposed back, stored to the plane */
            for (int kg = 0; kg < 36 / PW; ++kg) {
                const double *s = pp + 2 * (size_t)kg * PW;
                vd zv[36];
#define ZLOAD(j, X)  { (X) = LD(s + 2 * ((size_t)(j) * PST)); }
#define ZSTORE(k, X) { zv[k] = (X); }
                PFA36(ZLOAD, ZSTORE);
#undef ZLOAD
#undef ZSTORE
                for (int cg = 0; cg < 36 / PW; ++cg) {
                    vd t[PW];
                    for (int i = 0; i < PW; ++i) t[i] = zv[cg * PW + i];
                    CTRANSPOSE(t);
                    for (int l = 0; l < PW; ++l)
                        ST(rdst + 2 * ((size_t)(kg * PW + l) * 36 + cg * PW), t[l]);
                }
            }
        }
        } else if (mode >= 13) {
        /* ISTREAM-BCST (ice_r2): the transpose-free broadcast pass A straight
         * into out, pass B cached in place.  13 = prefetch-free twin (pf=0),
         * 14 = paced T1 read cursor + PFNX (pf=1), mirroring 12 vs 7.
         * PLANES always run FORWARD: pass A's advantage is its sequential
         * cold read, and the anti-phase (descending-PLANE) order was measured
         * to defeat it -- 142.5 vs 123.0 us/xform same-session forced A/B at
         * B=8 (MKL drift between the runs only 7%).  Modes 15-17 (ice_r3)
         * differ only in volume order / pass-B XV staging, handled above. */
        for (int xi = 0; xi < 36; ++xi) {
            const int x = xi;
#if PW == 4
            TS(passA_bcst)(inv + (size_t)x * 2 * NPLANE,
                           mid + (size_t)x * 2 * NPLANE, pp,
                           mode == 14 ? inv + FFT36PF_PFD + (size_t)x * 2 * NPLANE
                                      : (const double *)0,
                           (mode == 18 || mode == 19)
                                      ? inv + FFT36PF_PFDN + (size_t)x * 2 * NPLANE
                                      : (const double *)0);
#else
            /* PW=2 fallback (FORCE diagnostics only): the classic z-first
             * pass A, same bit class */
            TS(passA_plane)(inv + (size_t)x * 2 * NPLANE,
                            mid + (size_t)x * 2 * NPLANE, pp,
                            mode == 14 ? inv + FFT36PF_PFD + (size_t)x * 2 * NPLANE
                                       : (const double *)0,
                            (double *)0, (const double *)0);
#endif
        }
        } else {
        /* SCRATCH/streaming variant: z-transform first via transpose-on-load.
         * Mode 8 adds the paced write-intent cursor on the (cold, mid==out)
         * store stream; modes 1-7 leave it off (their mid is either warm
         * scratch or, in mode 7, deliberately kept as the pf=1 control).
         * Mode 9 replaces the T1 read cursor with the constant-lead NTA one
         * (and carries no pwc/nxt at all: its bet is that out stays
         * L2-resident, so there is no RFO to hide and nothing to stage). */
        for (int x = 0; x < 36; ++x)
            TS(passA_plane)(inv + (size_t)x * 2 * NPLANE,
                            mid + (size_t)x * 2 * NPLANE, pp,
                            (mode == 9 || mode == 12) ? (const double *)0
                                      : inv + FFT36PF_PFD + (size_t)x * 2 * NPLANE,
                            mode == 8 ? outv + FFT36PF_PFWD + (size_t)x * 2 * NPLANE
                                      : (double *)0,
                            mode == 9 ? inv + FFT36PF_PFDN + (size_t)x * 2 * NPLANE
                                      : (const double *)0);
        }
#endif /* FFT36PF_SKIPA */

#ifndef FFT36PF_SKIPB
        /* -------- pass B: x-transform, lanes = PW consecutive flat (ky,kz).
         * 36 read + 36 write streams of stride 20736 B; the read streams exceed
         * the L2 streamer, so prefetch each one line ahead unconditionally. ---- */
        if (!nt) {
            if (xvn)
                TS(passB_xv)(mid, outv, xvn);
            else
                TS(passB_cached)(mid, outv, 0, NPLANE / PW, mode == 1,
                                 mode >= 7 ? nxt : (const double *)0);
        } else
            TS(passB_nt)(mid, outv, 0, 324, nxt, 8);
#endif /* FFT36PF_SKIPB */
    }
    if (nt) _mm_sfence();
}

#undef vd
#undef vi
#undef vu
#undef LD
#undef ST
#undef STNT
#undef VC
#undef VS
#undef SWPM
#undef CSWAP
#undef CTRSTEP
#undef CTRANSPOSE
#undef IX
#undef OX
#undef UU
#undef SA_
#undef SB_
#undef PFA36
#undef PFA36X2
#undef PFSTEP
#undef PFRD
#undef PFWR
#undef PFNTA
#endif
