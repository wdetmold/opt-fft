/* L8_radix8 -- forward complex-double 3D DFT of the 8x8x8 cube, batched.
 *
 * MULTICORE (mt_r1)
 *   The batch is the parallel axis and nothing else is: one 8^3 volume is
 *   8 KiB / ~0.6 us of work, and the GOMP fork alone is 3-5 us
 *   (L13_direct mt_r1 measured it, with monotone losses from intra-volume
 *   splits at a 4x larger size), so B=1 ships SERIAL, bit-identical to
 *   phase 1, with no parallel region at all.  For B>1, thread t of a
 *   min(32, batch)-thread team owns the contiguous volume block
 *   [nb*t/T, nb*(t+1)/T) and runs the UNTOUCHED serial per-chunk run
 *   function on it -- zero synchronisation inside the region, one
 *   fork+join per execute, ranges computed from the team actually
 *   delivered.  Per-thread scratch lives in page-aligned 12-KiB slots
 *   (3 whole pages: no false sharing, NUMA-local), allocated and
 *   FIRST-TOUCHED BY THE OWNING THREAD inside fft3d_create(), which also
 *   spins up the OpenMP pool so the first timed execute creates no thread
 *   (decomposition and first-touch pattern from L13_direct/L64_blocked
 *   mt_r1).  Chunk boundaries are whole 8-KiB volumes, so threads never
 *   share an output cache line.  The caller's buffers are first-touched
 *   single-threaded by the driver (one socket owns in/out); contiguous
 *   chunks are the best we can do about that.
 *
 *   NT stores RETURN to the candidate pool for B>1.  The phase-1
 *   retirement ("hide the RFO with prefetchw, do not avoid it", panel_r5
 *   VERDICT 4.5) was a single-core result: one core is fill-buffer-bound,
 *   not DRAM-bound.  32 cores ARE aggregate-bandwidth-bound, and NT cuts
 *   mandatory traffic from 24 KB/volume (in + RFO + out) to 16
 *   (L64_blocked mt_r1: NT beat cached stores by ~25 % at every engine at
 *   the memory wall).  Each thread issues its own sfence before the join.
 *
 * MULTICORE (mt_r2) -- three changes, each from mt_r1 node evidence:
 *   1. The B>1 chunk kernel is now the FUSED 1f shape, not 3p.  The node's
 *      B=2048 cell was won by the two fused-shape rivals at 0.026-0.028
 *      us/vol while my threaded 3p read 0.046 in-arena / 0.035 scored:
 *      at 32 threads the 3p shape's extra 128+128 L1 scratch loads/stores
 *      per volume and its three store passes are pure loss.  The 1f
 *      kernels were already in this file (ported from L8_fusedaxes in
 *      phase 1); mt_r2 threads them, adds an NT twin (every 1f output
 *      store is one full aligned 64-B line, so streaming stores drop in),
 *      and demotes the whole 3p family to probes.  The installable B>1
 *      pool is ONE BIT CLASS again by construction: every candidate is
 *      the 1f shape (y,x,z axis order), and threading, team size, NT
 *      stores and prefetch flavor change no arithmetic.
 *   2. Dispatch is a persistent pinned spin-wait pthread pool built in
 *      fft3d_create() (protocol from L36_pfa mt_r1 via L8_fusedaxes:
 *      epoch release word, per-worker 128-B ack lines, main = member 0,
 *      ALL workers ack every epoch -- L17_rader's double-run rule -- and
 *      a seq_cst fence between the epoch store and the parked read).
 *      L8_fusedaxes measured the pool at -10.4 % per call vs one OMP
 *      region per execute at B=2048 (31.7 vs 35.3 us, 3/3); the region
 *      stays as the fallback and behind L8R_POOL=0.  After the tuner
 *      picks, the pool is shrunk to the picked team (idle spinners drag
 *      the all-core clock, L36_pfa).
 *   3. The streaming default flips to NT + HALF TEAM (mth-1f-nt-pfs).
 *      The node's B=32768 cell is socket-0-DRAM-bound (~100 GB/s: all
 *      caller pages are first-touched by the driver's one thread), where
 *      cached stores cost 24 KB/volume (~0.24 us/vol, the bad r1 scored
 *      run) and NT+16-close-threads costs 16 KB/volume all-local
 *      (~0.15-0.18: my r1 arena 0.153, L8_fusedaxes' scored pick
 *      fused-nt+pfs/16).  mt_r1's default-first hysteresis put the
 *      cached full-team pick ahead in 1 of 3 runs and that run's ~0.30
 *      is the whole 71 % leaderboard spread.  B=2048 keeps the cached
 *      full-team default (NT lost 3.5x there on the node, all 3 runs).
 *
 * MULTICORE (mt_r3) -- the execute-time governor, answering the mt_r2
 *   VERDICT's named action for L=8 ("use the whole machine"):
 *   1. DEEP regime (B=32768): the node measured all three L=8 entries tied
 *      at 94 GB/s on 16-thread single-socket teams, losing to fftw's
 *      32-thread 103 GB/s, with fr=0 (all caller pages socket-0) -- and
 *      fftw's per-process bimodality (median 9.6 ms, min 5.2) says its fast
 *      mode ARRIVES DURING the run: numa_balancing migrates pages that a
 *      wide team's far-socket threads keep faulting on.  A create-time race
 *      can never see this (fusedaxes' r2 governor probed wide ~70 ms and
 *      read 0.21 vs half's 0.17).  mt_r3 therefore DWELLS the first 40
 *      executes at full-team NT (static contiguous cuts: socket-1 threads
 *      re-touch the upper half every ~7 ms = maximal migration pressure),
 *      probes the half team, locks the measured-faster config, and revisits
 *      the loser with PAIRED probes every ~24 calls (AutoNUMA migrates a
 *      private page on its second same-node fault), relocking on a >3 %
 *      win.  The driver takes 30 samples (~195 executes, ~1.2 s), so a lock
 *      at call 44 leaves ~25 clean samples and a late migration still flips
 *      ~half the samples.  Slow probe calls are free under the harness's
 *      min-of-samples statistic (fftw's scored win at this cell IS that
 *      statistic).  Both configs are the same 1f-NT bit class, so flips are
 *      invisible to correctness/repeatability checks.  fr (% caller pages
 *      off node 0, get_mempolicy = pure read, allowed) and numa_balancing
 *      publish as gov{fr0,nb,w,h,lock,fl,fr1} in the description.
 *   2. MID regime (B=2048): a short on-buffer race over the cached
 *      full-team 1f trio {pfs, 520-pfs, none} -- one bit class -- because
 *      the phase-B 4K-alias count is set by (scr - out) mod 4096, an
 *      allocation lottery the create surrogate re-rolls (LITERATURE s4.5,
 *      L8_fusedaxes' fusedAA analysis).  2 settle + 9 race calls, all
 *      inside the driver's calibration; lock needs >1.5 % over the create
 *      pick.  mt-1f520-pfs also joins the create-time installables.
 *   3. A governed plan keeps the FULL spin pool (the governor switches team
 *      width per call via job.T); the mt_r2 node arena priced 31 idle
 *      spinners at ~0 at these DRAM-bound cells (mth in-arena 0.173 ==
 *      0.174 scored with a shrunk pool).  L8R_GOV=0 disables the governor,
 *      L8R_DWELL=<n> resizes the dwell, L8R_FORCE disables it implicitly.
 *   B=1 stays closed (serial 2p, no pool, no governor).
 *
 * Everything below the multicore layer is the phase-1 file, unchanged;
 * ../../geom/strategies/L8_radix8.md has the full history.
 *
 * TECHNIQUE
 *   One fully unrolled radix-8 codelet (52 instructions = 44 add/sub + 8 FMA, no
 *   twiddle table: every 8th root of unity is +-1, +-i or +-(1+-i)/sqrt(2), so the only
 *   multiplicative constant in the whole transform is C = 1/sqrt(2)), applied along all
 *   three axes.  SPLIT-complex vectors (separate re/im registers) make every
 *   multiplication by +-i a register rename plus a sign folded into the following add.
 *   Lanes hold 8 independent lines, so one vector instruction = one scalar operation of
 *   the codelet, with no cross-lane traffic inside the butterflies.
 *
 *   THREE kernel shapes are compiled from the same primitives and fft3d_create()
 *   picks by self-timing at the real batch size:
 *
 *   2-PASS (optimized for B=1; round-1/2 shape):
 *     pass 1, per x-plane:  transpose in (z -> registers) -> radix-8 along z
 *                           -> transpose (lanes = k2) -> radix-8 along y -> scratch
 *     pass 2, per k1:       radix-8 along x -> interleave -> store out rows
 *                           at out + k0*128 + k1*16: a 1-KB-STRIDED store pattern.
 *
 *   FUSED "1f" (new in round panel_r5; ported from L8_fusedaxes, which won the
 *   node's B=64 cell with it three rounds running, 0.623 us in panel_r4):
 *     phase A, per x-plane: deinterleave each z-pencil with one vunpcklo/hi pair
 *                           (lane l = z = PI[l], PI = 0,4,1,5,2,6,3,7; only 16
 *                           shuffles per plane) -> radix-8 along y (elementwise)
 *                           -> split-planar scratch sr[k1][x] / si[k1][x]
 *     phase B, per k1:      radix-8 along x (elementwise, contiguous loads)
 *                           -> trans8 pair (z into registers via the T2,T3,T1
 *                           non-destructive network; register j = z = PI[j],
 *                           fed to the codelet through PI^-1 = 0,2,4,6,1,3,5,7)
 *                           -> radix-8 along z -> untrans+interleave fused into
 *                           one 48-shuffle network over all 16 registers ->
 *                           16 half-pencil stores via the out_off table.
 *     Same totals as 2-pass (1248 FP, 896 shuffles, 256+256 loads/stores); the
 *     shuffles sit in phase B next to L1-resident loads instead of in pass 1
 *     next to the input demand-loads, which is the whole (measured) difference.
 *
 *   3-PASS (optimized for batch; new in round panel_r3):
 *     pass 1, per x-plane:  transpose in -> radix-8 along z -> scratch[x][k2]
 *     pass 2, per k2:       radix-8 along x, IN PLACE in the scratch column
 *                           (reads scr[x][k2], writes scr[k0][k2]; 0 shuffles)
 *     pass 3, per k0:       transpose (y -> registers) -> radix-8 along y
 *                           -> interleave -> store the contiguous 1 KB output plane.
 *     The output volume is written FRONT TO BACK, fully sequentially -- which is what
 *     the DRAM/NT-store regimes want, and what the 2-pass shape structurally cannot do
 *     (its last DFT axis is x, so k0 strides the stores by 1 KB).  Cost: +128 loads
 *     +128 stores per volume, all L1-resident; shuffle and FP counts are IDENTICAL
 *     (the second transpose of pass 1 moves to pass 3).  Sequential-store idea borrowed
 *     from L8_batchsimd's LANEX pass structure, which owns every batched cell.
 *
 * OPERATION COUNT PER TRANSFORM (one 8^3 volume, either shape)
 *   8-point codelet: 4 real mul + 52 real add = 56 flops (the published optimum:
 *   Burrus T7.1/T9.1, FFTW n1_8, Yavne 4N log2 N - 6N + 8), issued as 52 instructions
 *   (44 add/sub + 8 FMA; w8^{1,3} folded as add,sub + 4 FMA, from L8_batchsimd).
 *   24 codelets x 52 = 1248 vector FP instructions (8-wide), 896 shuffles
 *   (768 transpose + 128 interleave, copy-free on AVX-512), and 512 (2-pass) or
 *   768 (3-pass) loads/stores.  No twiddle table is read at run time.
 *
 * STREAMING REGIME
 *   Next-volume software prefetch (t0, distance 1 volume) -- the L2 streamer stops
 *   at 4 KiB page boundaries and one 8 KiB volume spans two pages.  Placements:
 *   BURST (all 128 lines during pass 1) and SPREAD (6/5/5 lines per iteration of
 *   passes 1/2/3 or 8/8 per fused-phase iteration, ~1 prefetch per 11 cycles;
 *   borrowed from L8_fusedaxes r2).  NEW in panel_r7: write-intent prefetch (pfw,
 *   prefetchw) of the next volume's OUTPUT lines at the same spread cadence --
 *   with plain stores every output line pays a read-for-ownership, and issuing it
 *   one volume early hides that latency.  Borrowed from L8_fusedaxes' panel_r5
 *   node win (fused+pfs+pfw took B=2048 at 0.910 and B=16384 at 1.254, 3/3
 *   picks); the mechanism traces to L6_unrolled/L36_pfa.  NT stores are RETIRED
 *   from the candidate sets: they lost on the node in every streaming cell four
 *   rounds running (panel_r5 VERDICT 4.5: hide the RFO, do not avoid it).  The
 *   NT kernels stay compiled for the record.
 *
 * SCRATCH DE-ALIAS (new in panel_r10, borrowed from L8_batchsimd's panel_r9
 *   B=1 node win)
 *   The classic fused-shape scratch puts si EXACTLY 4096 B after sr
 *   (scr+512 doubles), so phase-B loads from sr share every bits-11:6 line
 *   residue with phase-A's in-flight si stores -- the in-scratch REACHABLE
 *   case of LITERATURE s4.5's 4K-alias mechanism.  The 1f520 kernels place
 *   si at scr+520 doubles (+64 B, still 64-B aligned) and are used at B=1
 *   and in the mid regime; the streaming kernels keep the r7 layout
 *   byte-identical, because batchsimd applied the fix to every mode and
 *   paid +5.2 % at B=2048 (r9 VERDICT s2).  -DL8R_SIOFF=512 reverts.
 *
 * TUNER (regime-gated since panel_r4; SINGLE-BIT-CLASS pools since panel_r11)
 *   panel_r11 rule (from the r10 VERDICT s3(a), which caught THIS FILE's
 *   B=2048 minimum being produced by a pick the round's correctness check
 *   never saw): every regime's pool of INSTALLABLE candidates is one bit
 *   class, so a cross-process pick flip can never put unchecked bits behind
 *   a leaderboard number.  Cross-class candidates are timed and published
 *   through fft3d_description() ('*' marks them in the arena string) but can
 *   never be picked -- L36_mixedradix's "installable" rule, adopted verbatim.
 *   B=1 -> HARDWIRED avx512-2p, no tournament.  This executes the r10
 *   pre-registered fork: 1f520 shipped fixed in r10 and read 0.5760 with its
 *   arena showing 1f520 == 1f, i.e. the de-alias bought nothing here and the
 *   branch taken is "r11 hardwires 2p and B=1 is closed".  The r10 node
 *   arena independently ranked 2p first 3/3 (0.560/0.569/0.569).
 *   Probes timed at B=1: {1f520, 1f, 3p}.
 *   Mid regime (B>1, in+out <= 0.9*L3) -> installable {1f-pfs (default),
 *   1f520-pfs} (one class: y,x,z order, 2.27e-16); probes {3p-pfs, 2p}.
 *   Default reverted 1f520-pfs -> 1f-pfs: the r10 node arena read 1f-pfs
 *   faster in 2 of 3 runs (0.605/0.606 vs 0.614/0.613).  pfw is not offered
 *   here (prefetchw on cache-resident lines is a documented loser:
 *   L36_pfa +13 %/+11 % at B=1/B=4).
 *   in+out > 0.9*L3 -> installable {3p-pfs-pfw (default), 3p-pfs, 3p-pf}
 *   (one class: z,x,y order, 1.92e-16); probe {1f-pfs-pfw}.  The default
 *   family flip 1f -> 3p is the node's own r10 data: 3p-pfs-pfw won the
 *   B=2048 arena 3/3 (0.900-0.907 vs 0.913-0.926) and the two driver runs
 *   that shipped it (0.984/0.993 vs 1.012), and tied the B=16384 arena.
 *   First entry is the regime default and displacing it needs a >2 % win.
 *   Each timed trial is preceded by one untimed pass so every candidate is
 *   measured against its own cache state (L8_fusedaxes r3 protocol).  The
 *   batch loop lives inside per-candidate run functions (direct call, kernel
 *   inlined) instead of a per-volume indirect call.  The pick is reported
 *   through fft3d_description().
 *
 * ASSUMPTIONS
 *   L == 8 only.  `in` and `out` are distinct and 64-byte aligned (driver guarantees
 *   both); every vector access is 64-byte aligned as a result (alignment-checked
 *   fallback exists).  Three backends -- portable C, AVX2, AVX-512 -- are compiled
 *   from ONE macro-defined kernel text, so the algorithm verified locally is the
 *   algorithm the node runs.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1   /* sched_getcpu, cpu_set_t, pthread_setaffinity_np */
#endif
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "fft3d_api.h"

#define LL   8
#define NDBL 1024   /* doubles per volume: 8^3 complex */

/* Scratch x-plane stride in doubles.  128 = dense.  144 pads each plane by one
 * 128-byte row, so the 8 strided pass-2 accesses spread over 8 distinct L1 sets
 * instead of 4 (Bailey/#04 s7.3; the padding L8_batchsimd carries as LPZ=9).
 * Overridable for A/B: -DL8R_SCRX=128. */
#ifndef L8R_SCRX
#define L8R_SCRX 144
#endif
#define SCRX L8R_SCRX

/* si offset (doubles) of the de-aliased 1f520 scratch: 520 breaks the exact
 * 4096-B sr/si relation (L8_batchsimd's r9 fix).  -DL8R_SIOFF=512 reverts. */
#ifndef L8R_SIOFF
#define L8R_SIOFF 520
#endif

/* Backends below the best available ISA stay compiled (they are how the algorithm
 * text gets verified on machines without AVX-512) but are not tuner candidates,
 * so silence the unused-function warning on them. */
#if defined(__GNUC__)
#define KFN_MAYBE_UNUSED __attribute__((unused))
#define KERNFN static inline __attribute__((always_inline, unused)) void
#else
#define KFN_MAYBE_UNUSED
#define KERNFN static void
#endif

/* Per-candidate run function: the batch loop lives HERE, in the same TU as the
 * kernel, with the kernel forced inline and the prefetch flag a compile-time
 * constant -- no per-volume indirect call, no per-volume branch (panel_r3 "Next"
 * item 3: the last structural difference to L8_batchsimd's lanex_run). */
#define DEFRUN(rname, kfn, dopf)                                               \
    static KFN_MAYBE_UNUSED void rname(const double *restrict in,              \
                                       double *restrict out,                   \
                                       double *restrict scr, int nb)           \
    {                                                                          \
        for (int b = 0; b < nb; ++b)                                           \
            kfn(in + (size_t)b * NDBL, out + (size_t)b * NDBL, scr,            \
                ((dopf) && b + 1 < nb) ? in + (size_t)(b + 1) * NDBL : NULL);  \
    }

static const double C_RSQ2 = 0.7071067811865475244008443621048490;

/* Software prefetch of the next volume's input (t0, one volume ahead; hint and
 * distance node-validated by L8_batchsimd r2, B=2048 = 1.205 us). */
#define PF8(p) __builtin_prefetch((p), 0, 3)

#define PFVOL(f)                                                               \
    do {                                                                       \
        PF8(f);      PF8(f + 8);   PF8(f + 16);  PF8(f + 24);                  \
        PF8(f + 32); PF8(f + 40);  PF8(f + 48);  PF8(f + 56);                  \
        PF8(f + 64); PF8(f + 72);  PF8(f + 80);  PF8(f + 88);                  \
        PF8(f + 96); PF8(f + 104); PF8(f + 112); PF8(f + 120);                 \
    } while (0)

/* Spread placement: the same 128 lines of volume b+1, issued 6/5/5 per iteration
 * of passes 1/2/3 (48+40+40 = 128, each 64-byte line exactly once) instead of 16
 * per pass-1 iteration.  ~1 prefetch per 11 cycles, so it never competes with the
 * pass-1 demand-load burst or the NT store drain for the ~12 fill buffers
 * (L8_fusedaxes r2 failure 4 documents the burst+NT clog with a 1.6x number). */
#define PF6(f) do { PF8(f); PF8((f) + 8);  PF8((f) + 16);                      \
                    PF8((f) + 24); PF8((f) + 32); PF8((f) + 40); } while (0)
#define PF5(f) do { PF8(f); PF8((f) + 8);  PF8((f) + 16);                      \
                    PF8((f) + 24); PF8((f) + 32); } while (0)
/* 8 lines (64 doubles): the per-iteration quantum of the 2-pass/fused spread
 * placements (2 passes x 8 iterations x 8 lines = the 128 lines of a volume). */
#define PF8L(f) do { PF8(f); PF8((f) + 8);  PF8((f) + 16); PF8((f) + 24);       \
                     PF8((f) + 32); PF8((f) + 40); PF8((f) + 48);              \
                     PF8((f) + 56); } while (0)

/* Write-intent prefetch of the next volume's OUTPUT lines (emits prefetchw on
 * CLX/SPR).  With plain stores every output line pays a read-for-ownership;
 * issuing it one volume early hides that latency instead of avoiding it with
 * NT stores -- the rule the panel_r5 node data settled (L8_fusedaxes'
 * fused+pfs+pfw picked 3/3 at B=2048/16384, 0.910/1.254 us; same mechanism as
 * L36_pfa's pf=2 and L6_unrolled's fused_pfw).  Same spread cadence as the
 * input t0 prefetches, mirrored onto out + NDBL (valid whenever pf != NULL,
 * since in and out advance through the batch together). */
#define PFW8(p) __builtin_prefetch((p), 1, 3)
#define PF6W(f) do { PFW8(f); PFW8((f) + 8);  PFW8((f) + 16);                   \
                     PFW8((f) + 24); PFW8((f) + 32); PFW8((f) + 40); } while (0)
#define PF5W(f) do { PFW8(f); PFW8((f) + 8);  PFW8((f) + 16);                   \
                     PFW8((f) + 24); PFW8((f) + 32); } while (0)
#define PF8LW(f) do { PFW8(f); PFW8((f) + 8);  PFW8((f) + 16); PFW8((f) + 24);  \
                      PFW8((f) + 32); PFW8((f) + 40); PFW8((f) + 48);           \
                      PFW8((f) + 56); } while (0)

/* Per-instantiation prefetch-placement hooks for KERNEL2_BODY / KERNEL3_BODY. */
#define KPF_NONE(pf, i)    do { (void)(pf); } while (0)
#define KPF1_BURST(pf, x)  do { if (pf) PFVOL((pf) + (size_t)(x) * 128); } while (0)
#define KPF1_SPREAD(pf, x) do { if (pf) PF6((pf) + (size_t)(x) * 48); } while (0)
#define KPF2_SPREAD(pf, k) do { if (pf) PF5((pf) + 384 + (size_t)(k) * 40); } while (0)
#define KPF3_SPREAD(pf, k) do { if (pf) PF5((pf) + 704 + (size_t)(k) * 40); } while (0)
/* Spread placement for the 2-pass shape: 8 lines per iteration of each pass. */
#define KPF1_SPREAD2(pf, x) do { if (pf) PF8L((pf) + (size_t)(x) * 64); } while (0)
#define KPF2_SPREAD2(pf, k) do { if (pf) PF8L((pf) + 512 + (size_t)(k) * 64); } while (0)
/* Spread + write-intent: the input hooks above plus the mirror-image prefetchw
 * on the next volume's output.  These reference the kernel bodies' `out`
 * parameter by name (in scope wherever they are expanded). */
#define KPF1_SPREADW(pf, x) do { if (pf) { PF6((pf) + (size_t)(x) * 48);        \
    PF6W(out + NDBL + (size_t)(x) * 48); } } while (0)
#define KPF2_SPREADW(pf, k) do { if (pf) { PF5((pf) + 384 + (size_t)(k) * 40);  \
    PF5W(out + NDBL + 384 + (size_t)(k) * 40); } } while (0)
#define KPF3_SPREADW(pf, k) do { if (pf) { PF5((pf) + 704 + (size_t)(k) * 40);  \
    PF5W(out + NDBL + 704 + (size_t)(k) * 40); } } while (0)

/* ------------------------------------------------------------------ *
 *  The radix-8 split-complex codelet, written once.
 *
 *  XR/XI in, YR/YI out (arrays of 8 vectors).  All inputs are consumed in
 *  stage 1, so YR/YI may be the same arrays as XR/XI.
 *
 *  Decimation in time, natural input order (the bit reversal is absorbed by
 *  choosing which register feeds which butterfly):
 *    b = size-2 stage, c = size-4 stage (w4^1 = -i, free in split form),
 *    Y = size-8 stage (w8^1 = (1-i)C, w8^2 = -i, w8^3 = (-1-i)C).
 *  Counts: 16 + 16 + 20 = 52 instructions (44 add/sub + 8 FMA).  The w8^{1,3}
 *  twiddle+butterfly is 1 add + 1 sub + 4 FMA/FNMA each; formulation borrowed
 *  from L8_batchsimd round 1.
 * ------------------------------------------------------------------ */
#define RADIX8(XR, XI, YR, YI)                                                 \
    do {                                                                       \
        KV b0r = KADD((XR)[0], (XR)[4]), b0i = KADD((XI)[0], (XI)[4]);          \
        KV b1r = KSUB((XR)[0], (XR)[4]), b1i = KSUB((XI)[0], (XI)[4]);          \
        KV b2r = KADD((XR)[2], (XR)[6]), b2i = KADD((XI)[2], (XI)[6]);          \
        KV b3r = KSUB((XR)[2], (XR)[6]), b3i = KSUB((XI)[2], (XI)[6]);          \
        KV b4r = KADD((XR)[1], (XR)[5]), b4i = KADD((XI)[1], (XI)[5]);          \
        KV b5r = KSUB((XR)[1], (XR)[5]), b5i = KSUB((XI)[1], (XI)[5]);          \
        KV b6r = KADD((XR)[3], (XR)[7]), b6i = KADD((XI)[3], (XI)[7]);          \
        KV b7r = KSUB((XR)[3], (XR)[7]), b7i = KSUB((XI)[3], (XI)[7]);          \
        KV c0r = KADD(b0r, b2r), c0i = KADD(b0i, b2i);                          \
        KV c2r = KSUB(b0r, b2r), c2i = KSUB(b0i, b2i);                          \
        KV c1r = KADD(b1r, b3i), c1i = KSUB(b1i, b3r);                          \
        KV c3r = KSUB(b1r, b3i), c3i = KADD(b1i, b3r);                          \
        KV c4r = KADD(b4r, b6r), c4i = KADD(b4i, b6i);                          \
        KV c6r = KSUB(b4r, b6r), c6i = KSUB(b4i, b6i);                          \
        KV c5r = KADD(b5r, b7i), c5i = KSUB(b5i, b7r);                          \
        KV c7r = KSUB(b5r, b7i), c7i = KADD(b5i, b7r);                          \
        KV s5 = KADD(c5r, c5i);                                                \
        KV t5 = KSUB(c5i, c5r);                                                \
        KV s7 = KSUB(c7i, c7r);                                                \
        KV t7 = KADD(c7r, c7i);                                                \
        (YR)[0] = KADD(c0r, c4r); (YI)[0] = KADD(c0i, c4i);                     \
        (YR)[4] = KSUB(c0r, c4r); (YI)[4] = KSUB(c0i, c4i);                     \
        (YR)[2] = KADD(c2r, c6i); (YI)[2] = KSUB(c2i, c6r);                     \
        (YR)[6] = KSUB(c2r, c6i); (YI)[6] = KADD(c2i, c6r);                     \
        (YR)[1] = KFMA(kc, s5, c1r);  (YI)[1] = KFMA(kc, t5, c1i);              \
        (YR)[5] = KFNMA(kc, s5, c1r); (YI)[5] = KFNMA(kc, t5, c1i);             \
        (YR)[3] = KFMA(kc, s7, c3r);  (YI)[3] = KFNMA(kc, t7, c3i);             \
        (YR)[7] = KFNMA(kc, s7, c3r); (YI)[7] = KFMA(kc, t7, c3i);              \
    } while (0)

/* RADIX8J -- association-order probe variant (panel_r10).  Identical DAG and
 * depth, but the 16 final add/subs feeding outputs 0,4,2,6 are issued as
 * FMA/FNMA with a broadcast 1.0, so EVERY output join is FMA-class -- the L=8
 * analog of the L=6 "store-feeding FMAs beat store-feeding adds by 3-6% on
 * CLX" result (panel_r9 VERDICT s2/s5, L6_pfa/L6_unrolled's association-order
 * experiment; the VERDICT s6 explicitly asks for this propagation).
 * round(1.0*x + y) == round(x + y), so the output is BIT-IDENTICAL to
 * RADIX8's.  Requires `kone` (broadcast 1.0) in scope.  Probe only: timed and
 * published at B=1, never picked. */
#define RADIX8J(XR, XI, YR, YI)                                                \
    do {                                                                       \
        KV b0r = KADD((XR)[0], (XR)[4]), b0i = KADD((XI)[0], (XI)[4]);          \
        KV b1r = KSUB((XR)[0], (XR)[4]), b1i = KSUB((XI)[0], (XI)[4]);          \
        KV b2r = KADD((XR)[2], (XR)[6]), b2i = KADD((XI)[2], (XI)[6]);          \
        KV b3r = KSUB((XR)[2], (XR)[6]), b3i = KSUB((XI)[2], (XI)[6]);          \
        KV b4r = KADD((XR)[1], (XR)[5]), b4i = KADD((XI)[1], (XI)[5]);          \
        KV b5r = KSUB((XR)[1], (XR)[5]), b5i = KSUB((XI)[1], (XI)[5]);          \
        KV b6r = KADD((XR)[3], (XR)[7]), b6i = KADD((XI)[3], (XI)[7]);          \
        KV b7r = KSUB((XR)[3], (XR)[7]), b7i = KSUB((XI)[3], (XI)[7]);          \
        KV c0r = KADD(b0r, b2r), c0i = KADD(b0i, b2i);                          \
        KV c2r = KSUB(b0r, b2r), c2i = KSUB(b0i, b2i);                          \
        KV c1r = KADD(b1r, b3i), c1i = KSUB(b1i, b3r);                          \
        KV c3r = KSUB(b1r, b3i), c3i = KADD(b1i, b3r);                          \
        KV c4r = KADD(b4r, b6r), c4i = KADD(b4i, b6i);                          \
        KV c6r = KSUB(b4r, b6r), c6i = KSUB(b4i, b6i);                          \
        KV c5r = KADD(b5r, b7i), c5i = KSUB(b5i, b7r);                          \
        KV c7r = KSUB(b5r, b7i), c7i = KADD(b5i, b7r);                          \
        KV s5 = KADD(c5r, c5i);                                                \
        KV t5 = KSUB(c5i, c5r);                                                \
        KV s7 = KSUB(c7i, c7r);                                                \
        KV t7 = KADD(c7r, c7i);                                                \
        (YR)[0] = KFMA(kone, c4r, c0r);  (YI)[0] = KFMA(kone, c4i, c0i);        \
        (YR)[4] = KFNMA(kone, c4r, c0r); (YI)[4] = KFNMA(kone, c4i, c0i);       \
        (YR)[2] = KFMA(kone, c6i, c2r);  (YI)[2] = KFNMA(kone, c6r, c2i);       \
        (YR)[6] = KFNMA(kone, c6i, c2r); (YI)[6] = KFMA(kone, c6r, c2i);        \
        (YR)[1] = KFMA(kc, s5, c1r);  (YI)[1] = KFMA(kc, t5, c1i);              \
        (YR)[5] = KFNMA(kc, s5, c1r); (YI)[5] = KFNMA(kc, t5, c1i);             \
        (YR)[3] = KFMA(kc, s7, c3r);  (YI)[3] = KFNMA(kc, t7, c3i);             \
        (YR)[7] = KFNMA(kc, s7, c3r); (YI)[7] = KFMA(kc, t7, c3i);              \
    } while (0)

/* ------------------------------------------------------------------ *
 *  Kernel shape A: 2-pass (B=1 shape).  z+y per x-plane, then x per k1,
 *  strided output stores.
 * ------------------------------------------------------------------ */
#define KERNEL2_BODY                                                           \
{                                                                              \
    const KV kc = KBCAST(C_RSQ2);                                              \
    for (int x = 0; x < 8; ++x) {                                              \
        const double *p = in + (size_t)x * 128;                                \
        double *q = scr + (size_t)x * SCRX;                                     \
        KPF1(pf, x);                                                           \
        KV ar[8], ai[8], br[8], bi[8];                                          \
        for (int y = 0; y < 8; ++y) {                                          \
            ar[y] = KLOAD(p + (size_t)y * 16);                                 \
            ai[y] = KLOAD(p + (size_t)y * 16 + 8);                             \
        }                                                                      \
        KTR8A(ar, br);                                                         \
        KTR8A(ai, bi);                                                         \
        for (int z = 0; z < 4; ++z) {                                          \
            ar[z]     = br[2 * z];                                             \
            ai[z]     = br[2 * z + 1];                                         \
            ar[z + 4] = bi[2 * z];                                             \
            ai[z + 4] = bi[2 * z + 1];                                         \
        }                                                                      \
        RADIX8(ar, ai, ar, ai);                                                \
        KTR8B(ar, br);                                                         \
        KTR8B(ai, bi);                                                         \
        RADIX8(br, bi, br, bi);                                                \
        for (int k1 = 0; k1 < 8; ++k1) {                                       \
            KSTORE(q + (size_t)k1 * 16,     br[k1]);                           \
            KSTORE(q + (size_t)k1 * 16 + 8, bi[k1]);                           \
        }                                                                      \
    }                                                                          \
    for (int k1 = 0; k1 < 8; ++k1) {                                           \
        KPF2(pf, k1);                                                          \
        KV ur[8], ui[8];                                                       \
        for (int x = 0; x < 8; ++x) {                                          \
            ur[x] = KLOAD(scr + (size_t)x * SCRX + (size_t)k1 * 16);            \
            ui[x] = KLOAD(scr + (size_t)x * SCRX + (size_t)k1 * 16 + 8);        \
        }                                                                      \
        RADIX8(ur, ui, ur, ui);                                                \
        for (int k0 = 0; k0 < 8; ++k0)                                         \
            KILV(ur[k0], ui[k0], out + (size_t)k0 * 128 + (size_t)k1 * 16);    \
    }                                                                          \
}

/* ------------------------------------------------------------------ *
 *  Kernel shape B: 3-pass (batch shape), sequential output stores.
 *  z per x-plane -> x in place per k2 column -> y per k0-plane.
 *  Lane residue on AVX-512: pass 1's KTR8A leaves lane l = y SW(l); the
 *  z and x codelets are elementwise so nobody observes it; pass 3's KTR8B
 *  (output k renamed to slot SW(k)) restores natural registers and leaves
 *  lanes carrying k2 = SW(l), which KILV's index vectors already invert.
 * ------------------------------------------------------------------ */
#define KERNEL3_BODY                                                           \
{                                                                              \
    const KV kc = KBCAST(C_RSQ2);                                              \
    for (int x = 0; x < 8; ++x) {                                              \
        const double *p = in + (size_t)x * 128;                                \
        double *q = scr + (size_t)x * SCRX;                                     \
        KPF1(pf, x);                                                           \
        KV ar[8], ai[8], br[8], bi[8];                                          \
        for (int y = 0; y < 8; ++y) {                                          \
            ar[y] = KLOAD(p + (size_t)y * 16);                                 \
            ai[y] = KLOAD(p + (size_t)y * 16 + 8);                             \
        }                                                                      \
        KTR8A(ar, br);                                                         \
        KTR8A(ai, bi);                                                         \
        for (int z = 0; z < 4; ++z) {                                          \
            ar[z]     = br[2 * z];                                             \
            ai[z]     = br[2 * z + 1];                                         \
            ar[z + 4] = bi[2 * z];                                             \
            ai[z + 4] = bi[2 * z + 1];                                         \
        }                                                                      \
        RADIX8(ar, ai, ar, ai);                                                \
        for (int k2 = 0; k2 < 8; ++k2) {                                       \
            KSTORE(q + (size_t)k2 * 16,     ar[k2]);                           \
            KSTORE(q + (size_t)k2 * 16 + 8, ai[k2]);                           \
        }                                                                      \
    }                                                                          \
    for (int k2 = 0; k2 < 8; ++k2) {                                           \
        KPF2(pf, k2);                                                          \
        KV ur[8], ui[8];                                                       \
        for (int x = 0; x < 8; ++x) {                                          \
            ur[x] = KLOAD(scr + (size_t)x * SCRX + (size_t)k2 * 16);            \
            ui[x] = KLOAD(scr + (size_t)x * SCRX + (size_t)k2 * 16 + 8);        \
        }                                                                      \
        RADIX8(ur, ui, ur, ui);                                                \
        for (int k0 = 0; k0 < 8; ++k0) {                                       \
            KSTORE(scr + (size_t)k0 * SCRX + (size_t)k2 * 16,     ur[k0]);      \
            KSTORE(scr + (size_t)k0 * SCRX + (size_t)k2 * 16 + 8, ui[k0]);      \
        }                                                                      \
    }                                                                          \
    for (int k0 = 0; k0 < 8; ++k0) {                                           \
        KPF3(pf, k0);                                                          \
        const double *q = scr + (size_t)k0 * SCRX;                              \
        double *o = out + (size_t)k0 * 128;                                    \
        KV vr[8], vi[8], wr[8], wi[8];                                          \
        for (int k2 = 0; k2 < 8; ++k2) {                                       \
            vr[k2] = KLOAD(q + (size_t)k2 * 16);                               \
            vi[k2] = KLOAD(q + (size_t)k2 * 16 + 8);                           \
        }                                                                      \
        KTR8B(vr, wr);                                                         \
        KTR8B(vi, wi);                                                         \
        RADIX8(wr, wi, wr, wi);                                                \
        for (int k1 = 0; k1 < 8; ++k1)                                         \
            KILV(wr[k1], wi[k1], o + (size_t)k1 * 16);                         \
    }                                                                          \
}

/* ================================================================== *
 *  Backend 1: portable C.  Always compiled; the reference the whole
 *  algorithm is verified against, and the fallback on non-x86.
 * ================================================================== */
typedef struct { double d[8]; } v8g;

static inline v8g g_add(v8g a, v8g b)
{ v8g r; for (int i = 0; i < 8; ++i) r.d[i] = a.d[i] + b.d[i]; return r; }
static inline v8g g_sub(v8g a, v8g b)
{ v8g r; for (int i = 0; i < 8; ++i) r.d[i] = a.d[i] - b.d[i]; return r; }
static inline v8g g_fma(v8g a, v8g b, v8g c)
{ v8g r; for (int i = 0; i < 8; ++i) r.d[i] = a.d[i] * b.d[i] + c.d[i]; return r; }
static inline v8g g_fnma(v8g a, v8g b, v8g c)
{ v8g r; for (int i = 0; i < 8; ++i) r.d[i] = c.d[i] - a.d[i] * b.d[i]; return r; }
static inline v8g g_bcast(double c)
{ v8g r; for (int i = 0; i < 8; ++i) r.d[i] = c; return r; }
static inline v8g g_load(const double *p)
{ v8g r; for (int i = 0; i < 8; ++i) r.d[i] = p[i]; return r; }
static inline void g_store(double *p, v8g a)
{ for (int i = 0; i < 8; ++i) p[i] = a.d[i]; }
static inline void g_tr8(const v8g *r, v8g *t)
{ for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) t[i].d[j] = r[j].d[i]; }
static inline void g_ilv(v8g re, v8g im, double *dst)
{ for (int k = 0; k < 8; ++k) { dst[2 * k] = re.d[k]; dst[2 * k + 1] = im.d[k]; } }

#define KV      v8g
#define KADD    g_add
#define KSUB    g_sub
#define KFMA    g_fma
#define KFNMA   g_fnma
#define KBCAST  g_bcast
#define KLOAD   g_load
#define KSTORE  g_store
#define KTR8A   g_tr8
#define KTR8B   g_tr8
#define KILV    g_ilv
#define KPF1    KPF1_BURST
#define KPF2    KPF_NONE
#define KPF3    KPF_NONE
KERNFN kernel2_gen(const double *restrict in, double *restrict out,
                   double *restrict scr, const double *restrict pf) KERNEL2_BODY
KERNFN kernel3_gen(const double *restrict in, double *restrict out,
                   double *restrict scr, const double *restrict pf) KERNEL3_BODY
DEFRUN(run_2p_gen, kernel2_gen, 0)
DEFRUN(run_3p_gen, kernel3_gen, 0)
#undef KV
#undef KADD
#undef KSUB
#undef KFMA
#undef KFNMA
#undef KBCAST
#undef KLOAD
#undef KSTORE
#undef KTR8A
#undef KTR8B
#undef KILV
#undef KPF1
#undef KPF2
#undef KPF3

/* ================================================================== *
 *  Backend 2: AVX2 + FMA.  One 8-wide vector = two ymm registers, so the
 *  identical kernel body compiles to 256-bit code (which is also the
 *  256-bit-port-scheme hedge on the grading node).
 * ================================================================== */
#if defined(__AVX2__) && defined(__FMA__)
typedef struct { __m256d lo, hi; } v8y;

static inline v8y y_add(v8y a, v8y b)
{ v8y r; r.lo = _mm256_add_pd(a.lo, b.lo); r.hi = _mm256_add_pd(a.hi, b.hi); return r; }
static inline v8y y_sub(v8y a, v8y b)
{ v8y r; r.lo = _mm256_sub_pd(a.lo, b.lo); r.hi = _mm256_sub_pd(a.hi, b.hi); return r; }
static inline v8y y_fma(v8y a, v8y b, v8y c)
{ v8y r; r.lo = _mm256_fmadd_pd(a.lo, b.lo, c.lo); r.hi = _mm256_fmadd_pd(a.hi, b.hi, c.hi); return r; }
static inline v8y y_fnma(v8y a, v8y b, v8y c)
{ v8y r; r.lo = _mm256_fnmadd_pd(a.lo, b.lo, c.lo); r.hi = _mm256_fnmadd_pd(a.hi, b.hi, c.hi); return r; }
static inline v8y y_bcast(double c)
{ v8y r; r.lo = _mm256_set1_pd(c); r.hi = r.lo; return r; }
static inline v8y y_load(const double *p)
{ v8y r; r.lo = _mm256_loadu_pd(p); r.hi = _mm256_loadu_pd(p + 4); return r; }
static inline void y_store(double *p, v8y a)
{ _mm256_storeu_pd(p, a.lo); _mm256_storeu_pd(p + 4, a.hi); }

/* 4x4 double transpose: 4 unpack + 4 vperm2f128. */
#define T4(a0, a1, a2, a3, o0, o1, o2, o3)                                     \
    do {                                                                       \
        __m256d u0 = _mm256_unpacklo_pd((a0), (a1));                           \
        __m256d u1 = _mm256_unpackhi_pd((a0), (a1));                           \
        __m256d u2 = _mm256_unpacklo_pd((a2), (a3));                           \
        __m256d u3 = _mm256_unpackhi_pd((a2), (a3));                           \
        (o0) = _mm256_permute2f128_pd(u0, u2, 0x20);                           \
        (o1) = _mm256_permute2f128_pd(u1, u3, 0x20);                           \
        (o2) = _mm256_permute2f128_pd(u0, u2, 0x31);                           \
        (o3) = _mm256_permute2f128_pd(u1, u3, 0x31);                           \
    } while (0)

static inline void y_tr8(const v8y *r, v8y *t)
{
    T4(r[0].lo, r[1].lo, r[2].lo, r[3].lo, t[0].lo, t[1].lo, t[2].lo, t[3].lo);
    T4(r[4].lo, r[5].lo, r[6].lo, r[7].lo, t[0].hi, t[1].hi, t[2].hi, t[3].hi);
    T4(r[0].hi, r[1].hi, r[2].hi, r[3].hi, t[4].lo, t[5].lo, t[6].lo, t[7].lo);
    T4(r[4].hi, r[5].hi, r[6].hi, r[7].hi, t[4].hi, t[5].hi, t[6].hi, t[7].hi);
}

static inline void y_ilv(v8y re, v8y im, double *dst)
{
    __m256d u = _mm256_unpacklo_pd(re.lo, im.lo);
    __m256d v = _mm256_unpackhi_pd(re.lo, im.lo);
    _mm256_storeu_pd(dst,     _mm256_permute2f128_pd(u, v, 0x20));
    _mm256_storeu_pd(dst + 4, _mm256_permute2f128_pd(u, v, 0x31));
    u = _mm256_unpacklo_pd(re.hi, im.hi);
    v = _mm256_unpackhi_pd(re.hi, im.hi);
    _mm256_storeu_pd(dst + 8,  _mm256_permute2f128_pd(u, v, 0x20));
    _mm256_storeu_pd(dst + 12, _mm256_permute2f128_pd(u, v, 0x31));
}

static inline void y_ilv_nt(v8y re, v8y im, double *dst)
{
    __m256d u = _mm256_unpacklo_pd(re.lo, im.lo);
    __m256d v = _mm256_unpackhi_pd(re.lo, im.lo);
    _mm256_stream_pd(dst,     _mm256_permute2f128_pd(u, v, 0x20));
    _mm256_stream_pd(dst + 4, _mm256_permute2f128_pd(u, v, 0x31));
    u = _mm256_unpacklo_pd(re.hi, im.hi);
    v = _mm256_unpackhi_pd(re.hi, im.hi);
    _mm256_stream_pd(dst + 8,  _mm256_permute2f128_pd(u, v, 0x20));
    _mm256_stream_pd(dst + 12, _mm256_permute2f128_pd(u, v, 0x31));
}

#define KV      v8y
#define KADD    y_add
#define KSUB    y_sub
#define KFMA    y_fma
#define KFNMA   y_fnma
#define KBCAST  y_bcast
#define KLOAD   y_load
#define KSTORE  y_store
#define KTR8A   y_tr8
#define KTR8B   y_tr8
#define KILV    y_ilv
#define KPF1    KPF1_BURST
#define KPF2    KPF_NONE
#define KPF3    KPF_NONE
KERNFN kernel2_avx2(const double *restrict in, double *restrict out,
                    double *restrict scr, const double *restrict pf) KERNEL2_BODY
KERNFN kernel3_avx2(const double *restrict in, double *restrict out,
                    double *restrict scr, const double *restrict pf) KERNEL3_BODY
#undef KPF1
#undef KPF2
#undef KPF3
#define KPF1    KPF1_SPREAD
#define KPF2    KPF2_SPREAD
#define KPF3    KPF3_SPREAD
KERNFN kernel3s_avx2(const double *restrict in, double *restrict out,
                     double *restrict scr, const double *restrict pf) KERNEL3_BODY
#undef KILV
#define KILV    y_ilv_nt
/* 3-pass kernel, non-temporal output stores, spread prefetch (a prefetch burst and
 * an NT drain fight for the same fill buffers -- L8_fusedaxes r2).  Every output
 * line is written in full by one KILV call and the plane order is sequential, so
 * no read-for-ownership and no write-combining games.  Only worth it when the
 * batch's footprint exceeds the last-level cache; fft3d_create() decides. */
KERNFN kernel3s_nt_avx2(const double *restrict in, double *restrict out,
                        double *restrict scr, const double *restrict pf) KERNEL3_BODY
DEFRUN(run_2p_avx2,        kernel2_avx2,     0)
DEFRUN(run_3p_avx2,        kernel3_avx2,     0)
DEFRUN(run_3p_pf_avx2,     kernel3_avx2,     1)
DEFRUN(run_3p_pfs_avx2,    kernel3s_avx2,    1)
DEFRUN(run_3p_nt_avx2,     kernel3s_nt_avx2, 0)
DEFRUN(run_3p_nt_pfs_avx2, kernel3s_nt_avx2, 1)
#undef KV
#undef KADD
#undef KSUB
#undef KFMA
#undef KFNMA
#undef KBCAST
#undef KLOAD
#undef KSTORE
#undef KTR8A
#undef KTR8B
#undef KILV
#undef KPF1
#undef KPF2
#undef KPF3
#endif /* AVX2 */

/* ================================================================== *
 *  Backend 3: AVX-512.  One 8-wide vector = one zmm.  8x8 transpose in
 *  24 shuffles, ALL two-source non-destructive forms with immediate
 *  control (vshuff64x2 / vunpck) -- no vpermt2pd, hence no compiler
 *  register copies and no index-vector registers.  Network borrowed from
 *  L8_fusedaxes round 1 (via L8_batchsimd): a straight r<->l1 middle
 *  level has no non-destructive encoding, but the 3-cycle
 *  r1 -> l2 -> l1 -> r1 (imm 0x88/0xDD) is encodable, and composing
 *      stage A  r2 <-> l2   (vshuff64x2 0x44 / 0xEE)
 *      stage B  r1 3-cycle  (vshuff64x2 0x88 / 0xDD)
 *      stage C  r0 <-> l0   (vunpcklo / vunpckhi)
 *  yields  o[k][l] = in[SW(l)][k],  SW = (0,1,4,5,2,3,6,7) (involution),
 *  verified by emulating the documented intrinsic semantics.  The lane
 *  residue SW is absorbed at compile time:
 *    - z_tr8a (transpose-in, de-interleave): register order is natural;
 *      lanes become y = SW(l), which the elementwise codelets never see.
 *    - z_tr8b (transpose-out): network output k is renamed into slot
 *      SW(k), so registers are natural again; lanes carry k2 = SW(l).
 *    - z_ilv: the interleave index vectors compose the inverse SW, so
 *      the output store is in natural memory order.
 * ================================================================== */
#if defined(__AVX512F__)
static inline __m512d z_add(__m512d a, __m512d b) { return _mm512_add_pd(a, b); }
static inline __m512d z_sub(__m512d a, __m512d b) { return _mm512_sub_pd(a, b); }
static inline __m512d z_fma(__m512d a, __m512d b, __m512d c) { return _mm512_fmadd_pd(a, b, c); }
static inline __m512d z_fnma(__m512d a, __m512d b, __m512d c) { return _mm512_fnmadd_pd(a, b, c); }
static inline __m512d z_bcast(double c) { return _mm512_set1_pd(c); }
static inline __m512d z_load(const double *p) { return _mm512_loadu_pd(p); }
static inline void z_store(double *p, __m512d a) { _mm512_storeu_pd(p, a); }

#define Z_TR8_NET(r, t, M0, M1, M2, M3, M4, M5, M6, M7)                        \
    do {                                                                       \
        __m512d u0 = _mm512_shuffle_f64x2((r)[0], (r)[4], 0x44);               \
        __m512d u4 = _mm512_shuffle_f64x2((r)[0], (r)[4], 0xEE);               \
        __m512d u1 = _mm512_shuffle_f64x2((r)[1], (r)[5], 0x44);               \
        __m512d u5 = _mm512_shuffle_f64x2((r)[1], (r)[5], 0xEE);               \
        __m512d u2 = _mm512_shuffle_f64x2((r)[2], (r)[6], 0x44);               \
        __m512d u6 = _mm512_shuffle_f64x2((r)[2], (r)[6], 0xEE);               \
        __m512d u3 = _mm512_shuffle_f64x2((r)[3], (r)[7], 0x44);               \
        __m512d u7 = _mm512_shuffle_f64x2((r)[3], (r)[7], 0xEE);               \
        __m512d w0 = _mm512_shuffle_f64x2(u0, u2, 0x88);                       \
        __m512d w2 = _mm512_shuffle_f64x2(u0, u2, 0xDD);                       \
        __m512d w1 = _mm512_shuffle_f64x2(u1, u3, 0x88);                       \
        __m512d w3 = _mm512_shuffle_f64x2(u1, u3, 0xDD);                       \
        __m512d w4 = _mm512_shuffle_f64x2(u4, u6, 0x88);                       \
        __m512d w6 = _mm512_shuffle_f64x2(u4, u6, 0xDD);                       \
        __m512d w5 = _mm512_shuffle_f64x2(u5, u7, 0x88);                       \
        __m512d w7 = _mm512_shuffle_f64x2(u5, u7, 0xDD);                       \
        (t)[M0] = _mm512_unpacklo_pd(w0, w1);                                  \
        (t)[M1] = _mm512_unpackhi_pd(w0, w1);                                  \
        (t)[M2] = _mm512_unpacklo_pd(w2, w3);                                  \
        (t)[M3] = _mm512_unpackhi_pd(w2, w3);                                  \
        (t)[M4] = _mm512_unpacklo_pd(w4, w5);                                  \
        (t)[M5] = _mm512_unpackhi_pd(w4, w5);                                  \
        (t)[M6] = _mm512_unpacklo_pd(w6, w7);                                  \
        (t)[M7] = _mm512_unpackhi_pd(w6, w7);                                  \
    } while (0)

/* natural register order out; lanes become SW(l) */
static inline void z_tr8a(const __m512d *r, __m512d *t)
{ Z_TR8_NET(r, t, 0, 1, 2, 3, 4, 5, 6, 7); }
/* output k renamed to slot SW(k): registers natural, lanes stay SW */
static inline void z_tr8b(const __m512d *r, __m512d *t)
{ Z_TR8_NET(r, t, 0, 1, 4, 5, 2, 3, 6, 7); }

/* Lane l of re/im holds k2 = SW(l), so the index vectors interleave lanes
 * SW(0..3) into the low line and SW(4..7) into the high line.  The two
 * permutes destroy different sources (re for the low, im for the high), so
 * neither source needs a compiler copy. */
static inline void z_ilv(__m512d re, __m512d im, double *dst)
{
    const __m512i il = _mm512_setr_epi64(0, 8, 1, 9, 4, 12, 5, 13);
    const __m512i ih = _mm512_setr_epi64(10, 2, 11, 3, 14, 6, 15, 7);
    _mm512_storeu_pd(dst,     _mm512_permutex2var_pd(re, il, im));
    _mm512_storeu_pd(dst + 8, _mm512_permutex2var_pd(im, ih, re));
}

static inline void z_ilv_nt(__m512d re, __m512d im, double *dst)
{
    const __m512i il = _mm512_setr_epi64(0, 8, 1, 9, 4, 12, 5, 13);
    const __m512i ih = _mm512_setr_epi64(10, 2, 11, 3, 14, 6, 15, 7);
    _mm512_stream_pd(dst,     _mm512_permutex2var_pd(re, il, im));
    _mm512_stream_pd(dst + 8, _mm512_permutex2var_pd(im, ih, re));
}

/* ---------------- fused-shape ("1f") primitives ----------------
 * Ported from L8_fusedaxes (its rounds 1-4 B=64-winning kernel).  Three
 * non-destructive 2-in/2-out lane butterflies, (a,b) <- (LO(a,b), HI(a,b)),
 * named by which register bit they trade with which lane bit:
 *   T1  vunpcklo/hi        r <-> l0
 *   T2  vshuff64x2 44/EE   r <-> l2
 *   T3  vshuff64x2 88/DD   r -> l2 -> l1 -> r   (the encodable 3-cycle) */
#define ZBF_T1(a, b)                                                           \
    do { __m512d t_ = _mm512_unpacklo_pd((a), (b));                            \
         (b) = _mm512_unpackhi_pd((a), (b)); (a) = t_; } while (0)
#define ZBF_T2(a, b)                                                           \
    do { __m512d t_ = _mm512_shuffle_f64x2((a), (b), 0x44);                    \
         (b) = _mm512_shuffle_f64x2((a), (b), 0xEE); (a) = t_; } while (0)
#define ZBF_T3(a, b)                                                           \
    do { __m512d t_ = _mm512_shuffle_f64x2((a), (b), 0x88);                    \
         (b) = _mm512_shuffle_f64x2((a), (b), 0xDD); (a) = t_; } while (0)

/* In-place 8x8 transpose, 24 ops: T2 on register bit 2, T3 on bit 1, T1 on
 * bit 0.  With input lane l = z = FPI[l] (FPI = 0,4,1,5,2,6,3,7, the residue
 * of the unpck deinterleave), output register j holds z = FPI[j] and the lane
 * residue moves to the untransposed axis; feed the codelet through fpiinv. */
#define ZTR8F(m)                                                               \
    do {                                                                       \
        ZBF_T2((m)[0], (m)[4]); ZBF_T2((m)[1], (m)[5]);                        \
        ZBF_T2((m)[2], (m)[6]); ZBF_T2((m)[3], (m)[7]);                        \
        ZBF_T3((m)[0], (m)[2]); ZBF_T3((m)[1], (m)[3]);                        \
        ZBF_T3((m)[4], (m)[6]); ZBF_T3((m)[5], (m)[7]);                        \
        ZBF_T1((m)[0], (m)[1]); ZBF_T1((m)[2], (m)[3]);                        \
        ZBF_T1((m)[4], (m)[5]); ZBF_T1((m)[6], (m)[7]);                        \
    } while (0)

/* Inverse transpose AND complex re-interleave fused into one 48-op network
 * over all 16 registers: T3 on k2 bit 0, T3 on k2 bit 1, T1 on the re/im bit
 * lands (l2,l1,l0) = (k2_1, k2_0, re/im) -- interleaved complex, each output
 * register a ready-to-store half-pencil (destination = f_off below). */
#define ZUNTRI(r, q)                                                           \
    do {                                                                       \
        ZBF_T3((r)[0], (r)[1]); ZBF_T3((r)[2], (r)[3]);                        \
        ZBF_T3((r)[4], (r)[5]); ZBF_T3((r)[6], (r)[7]);                        \
        ZBF_T3((q)[0], (q)[1]); ZBF_T3((q)[2], (q)[3]);                        \
        ZBF_T3((q)[4], (q)[5]); ZBF_T3((q)[6], (q)[7]);                        \
        ZBF_T3((r)[0], (r)[2]); ZBF_T3((r)[1], (r)[3]);                        \
        ZBF_T3((r)[4], (r)[6]); ZBF_T3((r)[5], (r)[7]);                        \
        ZBF_T3((q)[0], (q)[2]); ZBF_T3((q)[1], (q)[3]);                        \
        ZBF_T3((q)[4], (q)[6]); ZBF_T3((q)[5], (q)[7]);                        \
        ZBF_T1((r)[0], (q)[0]); ZBF_T1((r)[1], (q)[1]);                        \
        ZBF_T1((r)[2], (q)[2]); ZBF_T1((r)[3], (q)[3]);                        \
        ZBF_T1((r)[4], (q)[4]); ZBF_T1((r)[5], (q)[5]);                        \
        ZBF_T1((r)[6], (q)[6]); ZBF_T1((r)[7], (q)[7]);                        \
    } while (0)

/* FPI o fpiinv = identity: register j of the ZTR8F output, read through
 * fpiinv, is z = j natural. */
static const unsigned char fpiinv[8] = { 0, 2, 4, 6, 1, 3, 5, 7 };

/* Store destination (double offset k0*128 + half*8 into the k1-row) of
 * ZUNTRI's 16 outputs, r[0..7] then q[0..7]: the k0 order is the codelet's
 * bit-reversal-free output landing, (0,4,2,6) halves then (1,5,3,7) halves. */
static const int f_off[16] = { 0, 512, 256, 768, 8, 520, 264, 776,
                               128, 640, 384, 896, 136, 648, 392, 904 };

#define KV      __m512d
#define KADD    z_add
#define KSUB    z_sub
#define KFMA    z_fma
#define KFNMA   z_fnma
#define KBCAST  z_bcast
#define KLOAD   z_load
#define KSTORE  z_store
#define KTR8A   z_tr8a
#define KTR8B   z_tr8b
#define KILV    z_ilv
#define KPF1    KPF1_BURST
#define KPF2    KPF_NONE
#define KPF3    KPF_NONE
KERNFN kernel2_avx512(const double *restrict in, double *restrict out,
                      double *restrict scr, const double *restrict pf) KERNEL2_BODY
KERNFN kernel3_avx512(const double *restrict in, double *restrict out,
                      double *restrict scr, const double *restrict pf) KERNEL3_BODY
#undef KPF1
#undef KPF2
#undef KPF3
#define KPF1    KPF1_SPREAD
#define KPF2    KPF2_SPREAD
#define KPF3    KPF3_SPREAD
KERNFN kernel3s_avx512(const double *restrict in, double *restrict out,
                       double *restrict scr, const double *restrict pf) KERNEL3_BODY
#undef KPF1
#undef KPF2
#undef KPF3
#define KPF1    KPF1_SPREADW
#define KPF2    KPF2_SPREADW
#define KPF3    KPF3_SPREADW
KERNFN kernel3sw_avx512(const double *restrict in, double *restrict out,
                        double *restrict scr, const double *restrict pf) KERNEL3_BODY
#undef KPF1
#undef KPF2
#undef KPF3
#define KPF1    KPF1_SPREAD2
#define KPF2    KPF2_SPREAD2
#define KPF3    KPF_NONE
KERNFN kernel2s_avx512(const double *restrict in, double *restrict out,
                       double *restrict scr, const double *restrict pf) KERNEL2_BODY
#undef KPF1
#undef KPF2
#undef KPF3
#define KPF1    KPF1_SPREAD
#define KPF2    KPF2_SPREAD
#define KPF3    KPF3_SPREAD
#undef KILV
#define KILV    z_ilv_nt
KERNFN kernel3s_nt_avx512(const double *restrict in, double *restrict out,
                          double *restrict scr, const double *restrict pf) KERNEL3_BODY

/* ---- fused shape "1f", ported from L8_fusedaxes (phase structure, T-network
 * order, fpiinv feed and f_off store table all theirs; the codelet is my
 * 52-instruction RADIX8, which is natural-in/natural-out like their dft8s, so
 * it drops in unchanged).  Scratch use: sr = scr[0..511], si = scr[512..1023]
 * (split-planar, [k1][x] slots of 8 doubles; phase B reads each k1-row as 16
 * contiguous vectors).  Spread prefetch: 8 lines per iteration of each phase;
 * the _SW hooks add the mirror-image prefetchw on the next volume's output. */
#define K1FA_S(pf, x)  do { if (pf) PF8L((pf) + (size_t)(x) * 64); } while (0)
#define K1FB_S(pf, k)  do { if (pf) PF8L((pf) + 512 + (size_t)(k) * 64); } while (0)
#define K1FA_SW(pf, x) do { if (pf) { PF8L((pf) + (size_t)(x) * 64);            \
    PF8LW(out + NDBL + (size_t)(x) * 64); } } while (0)
#define K1FB_SW(pf, k) do { if (pf) { PF8L((pf) + 512 + (size_t)(k) * 64);      \
    PF8LW(out + NDBL + 512 + (size_t)(k) * 64); } } while (0)

#define KERNEL1F_BODY(PFA, PFB, SIOFF, R8, ZST)                                \
{                                                                              \
    const KV kc = KBCAST(C_RSQ2);                                              \
    const KV kone = KBCAST(1.0); (void)kone;   /* used only by RADIX8J */      \
    double *sr = scr, *si = scr + (SIOFF);                                     \
    for (int x = 0; x < 8; ++x) {                                              \
        const double *p = in + (size_t)x * 128;                                \
        PFA(pf, x);                                                            \
        KV r[8], q[8];                                                         \
        for (int y = 0; y < 8; ++y) {                                          \
            KV a = z_load(p + (size_t)y * 16);                                 \
            KV b = z_load(p + (size_t)y * 16 + 8);                             \
            r[y] = _mm512_unpacklo_pd(a, b);   /* lane l = z = FPI[l] */       \
            q[y] = _mm512_unpackhi_pd(a, b);                                   \
        }                                                                      \
        R8(r, q, r, q);                        /* y axis -> k1, elementwise */ \
        for (int k1 = 0; k1 < 8; ++k1) {                                       \
            z_store(sr + ((size_t)k1 * 8 + x) * 8, r[k1]);                     \
            z_store(si + ((size_t)k1 * 8 + x) * 8, q[k1]);                     \
        }                                                                      \
    }                                                                          \
    for (int k1 = 0; k1 < 8; ++k1) {                                           \
        PFB(pf, k1);                                                           \
        const double *rrow = sr + (size_t)k1 * 64;                             \
        const double *irow = si + (size_t)k1 * 64;                             \
        KV r[8], q[8], zr[8], zq[8];                                           \
        for (int x = 0; x < 8; ++x) {                                          \
            r[x] = z_load(rrow + (size_t)x * 8);                               \
            q[x] = z_load(irow + (size_t)x * 8);                               \
        }                                                                      \
        R8(r, q, r, q);                        /* x axis -> k0, elementwise */ \
        ZTR8F(r);                              /* z into regs: reg j = FPI[j] */\
        ZTR8F(q);                                                              \
        for (int j = 0; j < 8; ++j) { zr[j] = r[fpiinv[j]]; zq[j] = q[fpiinv[j]]; } \
        R8(zr, zq, zr, zq);                    /* z axis -> k2 */              \
        ZUNTRI(zr, zq);                        /* untranspose + interleave */  \
        double *o = out + (size_t)k1 * 16;                                     \
        for (int j = 0; j < 8; ++j) {                                          \
            ZST(o + f_off[j],     zr[j]);                                      \
            ZST(o + f_off[j + 8], zq[j]);                                      \
        }                                                                      \
    }                                                                          \
}

/* NT twin of z_store for the 1f output landing: every f_off destination is
 * one full 64-byte-aligned line written by exactly one store, so streaming
 * stores drop in with no read-for-ownership and no write-combining games
 * (new in mt_r2; the streaming cell is aggregate-DRAM-bound, see header). */
static inline void z_nstore(double *p, __m512d a) { _mm512_stream_pd(p, a); }

KERNFN kernel1f_avx512(const double *restrict in, double *restrict out,
                       double *restrict scr, const double *restrict pf)
    KERNEL1F_BODY(K1FA_S, K1FB_S, 512, RADIX8, z_store)
KERNFN kernel1fw_avx512(const double *restrict in, double *restrict out,
                        double *restrict scr, const double *restrict pf)
    KERNEL1F_BODY(K1FA_SW, K1FB_SW, 512, RADIX8, z_store)
/* NT-output 1f, input-only spread prefetch (a prefetch burst and the NT
 * drain fight for fill buffers -- L8_fusedaxes r2; pfw on NT output is
 * meaningless).  mt_r2: the threaded streaming candidate. */
KERNFN kernel1f_nt_avx512(const double *restrict in, double *restrict out,
                          double *restrict scr, const double *restrict pf)
    KERNEL1F_BODY(K1FA_S, K1FB_S, 512, RADIX8, z_nstore)

/* De-aliased 1f twins (panel_r10): si at scr + 520 doubles (+4160 B) instead
 * of +512 (+4096 B exactly).  With si at +512, every phase-A store pair
 * ST(sr+o)/ST(si+o) and every phase-B load pair LD(rrow+o)/LD(irow+o) lands on
 * the same bits-11:6 line residue, so phase-B loads from sr can be 4K-alias
 * blocked by phase-A's in-flight si stores.  This is the in-scratch reachable
 * case of LITERATURE s4.5, and it is L8_batchsimd's panel_r9 B=1 node win
 * (SI 512 -> 520: 0.5647 -> 0.5588 median, tail eliminated); L8_fusedaxes
 * adopted the same fix as its fusedSI variants.  +520 doubles stays 64-byte
 * aligned.  GATED to B=1 and the mid regime: batchsimd applied it to every
 * mode in r9 and paid +5.2 % at B=2048 (VERDICT s2), so the streaming kernels
 * keep the r7 layout, byte-identical.  -DL8R_SIOFF=512 reverts for A/B. */
KERNFN kernel1f520_avx512(const double *restrict in, double *restrict out,
                          double *restrict scr, const double *restrict pf)
    KERNEL1F_BODY(K1FA_S, K1FB_S, L8R_SIOFF, RADIX8, z_store)
/* Association-order probe: the same de-aliased kernel with the FMA-join
 * codelet RADIX8J.  Bit-identical output; timed at B=1, never picked. */
KERNFN kernel1f520j_avx512(const double *restrict in, double *restrict out,
                           double *restrict scr, const double *restrict pf)
    KERNEL1F_BODY(K1FA_S, K1FB_S, L8R_SIOFF, RADIX8J, z_store)

DEFRUN(run_2p_avx512,        kernel2_avx512,     0)
DEFRUN(run_2p_pfs_avx512,    kernel2s_avx512,    1)
DEFRUN(run_1f_avx512,        kernel1f_avx512,    0)
DEFRUN(run_1f_pfs_avx512,    kernel1f_avx512,    1)
DEFRUN(run_1f_pfsw_avx512,   kernel1fw_avx512,   1)
DEFRUN(run_1f_nt_pfs_avx512, kernel1f_nt_avx512, 1)
DEFRUN(run_1f520_avx512,     kernel1f520_avx512,  0)
DEFRUN(run_1f520_pfs_avx512, kernel1f520_avx512,  1)
DEFRUN(run_1f520j_avx512,    kernel1f520j_avx512, 0)
DEFRUN(run_3p_avx512,        kernel3_avx512,     0)
DEFRUN(run_3p_pf_avx512,     kernel3_avx512,     1)
DEFRUN(run_3p_pfs_avx512,    kernel3s_avx512,    1)
DEFRUN(run_3p_pfsw_avx512,   kernel3sw_avx512,   1)
DEFRUN(run_3p_nt_avx512,     kernel3s_nt_avx512, 0)
DEFRUN(run_3p_nt_pfs_avx512, kernel3s_nt_avx512, 1)
#undef KV
#undef KADD
#undef KSUB
#undef KFMA
#undef KFNMA
#undef KBCAST
#undef KLOAD
#undef KSTORE
#undef KTR8A
#undef KTR8B
#undef KILV
#undef KPF1
#undef KPF2
#undef KPF3
#endif /* AVX512F */

/* ================================================================== *
 *  Multicore layer (new in mt_r1): batch-parallel over contiguous
 *  volume chunks, each chunk run by the untouched serial run function
 *  with that thread's own page-aligned scratch slot.
 * ================================================================== */
typedef void (*rfn_t)(const double *restrict, double *restrict, double *restrict,
                      int);

#ifdef _OPENMP
#include <omp.h>
#endif

/* Per-thread scratch slot stride in doubles: 12 KiB = 3 whole 4-KiB pages,
 * so slots never share a page (no false sharing; each slot NUMA-local to
 * the core that first touches it).  Must cover the largest per-volume
 * scratch any kernel shape uses. */
#define SLOTD 1536
typedef char l8r_slot_big_enough[(SLOTD >= 8 * SCRX &&
                                  SLOTD >= L8R_SIOFF + 512) ? 1 : -1];

/* Threads the harness granted us (OMP_NUM_THREADS=32 on the node), read
 * once in fft3d_create().  Never raised -- the brief forbids it. */
static int g_team = 1;

/* ------------------------------------------------------------------ *
 *  Persistent pinned spin-wait pool (new in mt_r2).  One OMP parallel
 *  region + implicit barrier per execute costs 2.7-8.3 us (L17_rader /
 *  L23_matrixsimd mt_r1); L8_fusedaxes measured replacing it with this
 *  pool at -10.4 % per call at B=2048 (31.7 vs 35.3 us, 3/3 both sides).
 *  Protocol ADOPTED FROM L36_pfa mt_r1 via L8_fusedaxes mt_r1: one epoch
 *  release word, per-worker 128-B-padded ack lines, main = participant 0,
 *  publish-then-scan; including L17_rader's hard-won invariant that main
 *  waits for ALL workers to ack every epoch -- team members and idle
 *  alike -- because that is what makes rewriting the job descriptor for
 *  the next dispatch safe (their team-only-ack attempt double-ran jobs),
 *  and L8_fusedaxes' own seq_cst fence between the epoch store and the
 *  parked read (TSO lets the load pass the store; a just-parked worker
 *  could be missed and main would spin forever).  Workers are created in
 *  fft3d_create, pinned to the exact CPUs the harness's close/cores OMP
 *  mapping chose (read back via sched_getcpu in the first-touch region),
 *  spin ~25 ms after their last job, then park on a condvar.  Execute
 *  never creates a thread.  L8R_POOL=0 forces the OMP-region fallback.
 * ------------------------------------------------------------------ */
#ifdef _OPENMP

static inline void l8r_relax(void)
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    sched_yield();
#endif
}

/* thread t of T runs the contiguous volume slice [nb*t/T, nb*(t+1)/T) on
 * its own NUMA-local scratch slot -- same decomposition as mt_r1 */
static inline void l8r_slice(rfn_t chunk, const double *restrict in,
                             double *restrict out, double *restrict scr,
                             int nb, int T, int t)
{
    int lo = (int)((long long)nb * t / T);
    int hi = (int)((long long)nb * (t + 1) / T);
    if (hi > lo)
        chunk(in + (size_t)lo * NDBL, out + (size_t)lo * NDBL,
              scr + (size_t)t * SLOTD, hi - lo);
}

struct l8r_job {
    rfn_t  chunk;
    const double *in;
    double *out;
    double *scr;
    int    nb, T, ntf;
};

struct l8r_ack { _Atomic unsigned long e;
                 char pad[128 - sizeof(_Atomic unsigned long)]; };

struct l8r_pool {
    int nw;                        /* worker threads (team - 1) */
    pthread_t th[32];
    struct l8r_job job;            /* main writes BEFORE the epoch bump */
    _Atomic unsigned long epoch;   /* the release word */
    _Atomic int stop;
    _Atomic int parked;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    struct l8r_ack ack[32] __attribute__((aligned(128)));
};

static struct l8r_pool *g_pool = NULL;
static int g_cpumap[32];           /* harness's close/cores placement */

struct l8r_warg { struct l8r_pool *pl; int t; };

static void *l8r_worker(void *arg)
{
    struct l8r_warg *wa = (struct l8r_warg *)arg;
    struct l8r_pool *pl = wa->pl;
    const int t = wa->t;
    free(wa);
    if (g_cpumap[t] >= 0) {   /* pin to the CPU OMP thread t ran on */
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(g_cpumap[t], &cs);
        pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    }
    unsigned long e = 0;
    for (;;) {
        unsigned long ne;
        long spins = 0;
        double t_last = 0.0;
        while ((ne = atomic_load_explicit(&pl->epoch, memory_order_acquire)) == e) {
            if (atomic_load_explicit(&pl->stop, memory_order_acquire)) return NULL;
            l8r_relax();
            if (((++spins) & 0xfff) == 0) {          /* time check every 4096 */
                struct timespec ts;
                clock_gettime(CLOCK_MONOTONIC, &ts);
                const double now = (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
                if (t_last == 0.0) t_last = now;
                else if (now - t_last > 25e-3) {     /* park after ~25 ms idle */
                    pthread_mutex_lock(&pl->mu);
                    atomic_fetch_add(&pl->parked, 1);
                    while (atomic_load_explicit(&pl->epoch, memory_order_acquire) == e &&
                           !atomic_load_explicit(&pl->stop, memory_order_acquire))
                        pthread_cond_wait(&pl->cv, &pl->mu);
                    atomic_fetch_sub(&pl->parked, 1);
                    pthread_mutex_unlock(&pl->mu);
                    t_last = 0.0;
                    spins = 0;
                }
            }
        }
        e = ne;
        if (atomic_load_explicit(&pl->stop, memory_order_acquire)) return NULL;
        const struct l8r_job j = pl->job;   /* safe: main holds the next write
                                               until every worker acked e */
        if (t < j.T)
            l8r_slice(j.chunk, j.in, j.out, j.scr, j.nb, j.T, t);
#if defined(__SSE2__)
        /* NT stores are weakly ordered and a fence only orders the ISSUING
         * core's stores: fence the chunk BEFORE publishing the ack. */
        if (j.ntf) _mm_sfence();
#endif
        atomic_store_explicit(&pl->ack[t].e, e, memory_order_release);
    }
}

static void l8r_pool_run(struct l8r_pool *pl, rfn_t chunk, int ntf,
                         const double *restrict in, double *restrict out,
                         double *restrict scr, int nb, int T)
{
    if (T > pl->nw + 1) T = pl->nw + 1;
    pl->job.chunk = chunk; pl->job.in = in;  pl->job.out = out;
    pl->job.scr   = scr;   pl->job.nb = nb;  pl->job.T   = T;
    pl->job.ntf   = ntf;
    const unsigned long e =
        atomic_load_explicit(&pl->epoch, memory_order_relaxed) + 1;
    atomic_store_explicit(&pl->epoch, e, memory_order_release);
    atomic_thread_fence(memory_order_seq_cst);
    if (atomic_load_explicit(&pl->parked, memory_order_relaxed) > 0) {
        pthread_mutex_lock(&pl->mu);
        pthread_cond_broadcast(&pl->cv);
        pthread_mutex_unlock(&pl->mu);
    }
    l8r_slice(chunk, in, out, scr, nb, T, 0);
#if defined(__SSE2__)
    if (ntf) _mm_sfence();
#endif
    for (int t = 1; t <= pl->nw; ++t)     /* ALL workers ack, not just the team */
        while (atomic_load_explicit(&pl->ack[t].e, memory_order_acquire) != e)
            l8r_relax();
}

static void l8r_pool_destroy(struct l8r_pool *pl)
{
    if (!pl) return;
    atomic_store_explicit(&pl->stop, 1, memory_order_release);
    pthread_mutex_lock(&pl->mu);
    pthread_cond_broadcast(&pl->cv);
    pthread_mutex_unlock(&pl->mu);
    for (int t = 1; t <= pl->nw; ++t) pthread_join(pl->th[t], NULL);
    pthread_mutex_destroy(&pl->mu);
    pthread_cond_destroy(&pl->cv);
    free(pl);
}

/* nworkers = team - 1 (main is participant 0).  Uses the g_cpumap read in
 * fft3d_create's first-touch region.  NULL on any failure: mt_run falls
 * back to the OMP region, which is correct, just slower. */
static struct l8r_pool *l8r_pool_create(int nworkers)
{
    if (nworkers < 1 || nworkers > 31) return NULL;
    void *mem = NULL;
    if (posix_memalign(&mem, 128, sizeof(struct l8r_pool)) != 0 || !mem)
        return NULL;
    struct l8r_pool *pl = (struct l8r_pool *)mem;
    memset(pl, 0, sizeof *pl);
    pthread_mutex_init(&pl->mu, NULL);
    pthread_cond_init(&pl->cv, NULL);
    for (int t = 1; t <= nworkers; ++t) {
        struct l8r_warg *wa = (struct l8r_warg *)malloc(sizeof *wa);
        if (!wa) break;
        wa->pl = pl; wa->t = t;
        if (pthread_create(&pl->th[t], NULL, l8r_worker, wa) != 0) {
            free(wa);
            break;
        }
        pl->nw = t;
    }
    if (pl->nw != nworkers) {   /* incomplete team: tear down, fall back */
        l8r_pool_destroy(pl);
        return NULL;
    }
    return pl;
}
#endif /* _OPENMP */

static void mt_run(rfn_t chunk, int team, int ntf,
                   const double *restrict in, double *restrict out,
                   double *restrict scr, int nb)
{
#ifdef _OPENMP
    int T = team < nb ? team : nb;
    if (T > 1) {
        if (g_pool) {   /* the fast path: no thread creation, no OMP barrier */
            l8r_pool_run(g_pool, chunk, ntf, in, out, scr, nb, T);
            return;
        }
#pragma omp parallel num_threads(T)
        {
            /* ranges from the team ACTUALLY delivered, so a squeezed team
             * still computes the whole batch (L13_direct mt_r1) */
            int U  = omp_get_num_threads();
            int t  = omp_get_thread_num();
            int lo = (int)((long long)nb * t / U);
            int hi = (int)((long long)nb * (t + 1) / U);
            if (hi > lo)
                chunk(in + (size_t)lo * NDBL, out + (size_t)lo * NDBL,
                      scr + (size_t)t * SLOTD, hi - lo);
#if defined(__SSE2__)
            /* NT stores are weakly ordered and a fence only orders the
             * ISSUING core's stores: each thread fences its own chunk
             * BEFORE the join barrier. */
            if (ntf) _mm_sfence();
#endif
        }
        return;
    }
#endif
    (void)team; (void)ntf;
    chunk(in, out, scr, nb);
}

#define DEFMT(rname, chunkfn, teamexpr, ntf)                                   \
    static KFN_MAYBE_UNUSED void rname(const double *restrict in,              \
                                       double *restrict out,                   \
                                       double *restrict scr, int nb)           \
    { mt_run(chunkfn, (teamexpr), (ntf), in, out, scr, nb); }

#if defined(__AVX512F__)
/* mt_r2: the INSTALLABLE B>1 family is the threaded fused 1f shape -- the
 * node's B=2048 cell went to the two fused-shape rivals 3/3 while threaded
 * 3p read 35-75 % slower (see header).  One bit class: threading, team
 * size, NT and prefetch flavor change no arithmetic. */
DEFMT(run_mt_1f_pfs_avx512,      run_1f_pfs_avx512,    g_team,           0)
DEFMT(run_mt_1f_avx512,          run_1f_avx512,        g_team,           0)
DEFMT(run_mt_1f_pfsw_avx512,     run_1f_pfsw_avx512,   g_team,           0)
DEFMT(run_mt_1f520_pfs_avx512,   run_1f520_pfs_avx512, g_team,           0)
DEFMT(run_mt_1f_nt_pfs_avx512,   run_1f_nt_pfs_avx512, g_team,           1)
DEFMT(run_mth_1f_pfs_avx512,     run_1f_pfs_avx512,    (g_team + 1) / 2, 0)
DEFMT(run_mth_1f_nt_pfs_avx512,  run_1f_nt_pfs_avx512, (g_team + 1) / 2, 1)
/* the mt_r1 3p family stays compiled: probes (timed, published, never
 * picked -- different axis order = different bit class) */
DEFMT(run_mt_3p_avx512,          run_3p_avx512,        g_team,           0)
DEFMT(run_mt_3p_pfs_avx512,      run_3p_pfs_avx512,    g_team,           0)
DEFMT(run_mt_3p_pfsw_avx512,     run_3p_pfsw_avx512,   g_team,           0)
DEFMT(run_mt_3p_nt_avx512,       run_3p_nt_avx512,     g_team,           1)
DEFMT(run_mt_3p_nt_pfs_avx512,   run_3p_nt_pfs_avx512, g_team,           1)
/* Half-team twins: the two-socket question.  The driver first-touches all
 * caller pages on ONE socket, so on the node the far half of a close-bound
 * 32-thread team pays UPI for every line; a half team may be all-local.
 * Wallaby (one 32-core socket) cannot price this -- the node tuner does. */
DEFMT(run_mth_3p_nt_pfs_avx512,  run_3p_nt_pfs_avx512, (g_team + 1) / 2, 1)
DEFMT(run_mth_3p_pfs_avx512,     run_3p_pfs_avx512,    (g_team + 1) / 2, 0)
#elif defined(__AVX2__) && defined(__FMA__)
DEFMT(run_mt_3p_avx2,            run_3p_avx2,          g_team,           0)
DEFMT(run_mt_3p_pfs_avx2,        run_3p_pfs_avx2,      g_team,           0)
DEFMT(run_mt_3p_nt_pfs_avx2,     run_3p_nt_pfs_avx2,   g_team,           1)
#else
DEFMT(run_mt_3p_gen,             run_3p_gen,           g_team,           0)
#endif

/* ------------------------------------------------------------------ *
 *  Execute-time governor (new in mt_r3; design and legality precedent
 *  from L8_fusedaxes mt_r2, endorsed by the mt_r2 VERDICT).  The
 *  create-time arena races on a surrogate whose page placement and 4K
 *  line offsets are NOT the driver's; the governor re-decides ON THE
 *  CALLER'S REAL BUFFERS.  Every probe call is a full correct execute,
 *  and the harness statistic (min over samples) ignores slow early
 *  calls, so probing is free.  Two modes:
 *
 *  DEEP (ws > 3xL3, the B=32768 cell): the mt_r2 VERDICT's named action
 *  for L=8 -- "use the whole machine".  All three L=8 entries picked
 *  16-thread single-socket teams and tied at 94 GB/s while fftw's
 *  32-thread plan reached 103, and fftw's per-process bimodality
 *  (median 9.6 ms vs min 5.2) says the fast mode ARRIVES DURING the
 *  run: with numa_balancing=1, a wide team's socket-1 threads fault on
 *  scanner-marked pages and AutoNUMA migrates them, unlocking the far
 *  socket's controllers (the L=6 B=65536 cell sustains 200 GB/s the
 *  same way).  A create-time race can never see this -- fusedaxes' r2
 *  governor probed wide for only ~12 calls (~70 ms) and read 0.21.  So:
 *  DWELL the first 40 calls at FULL-team NT with static contiguous cuts
 *  (socket-1 threads continuously re-touch the upper half = maximal
 *  migration pressure), then probe the half team, lock the measured
 *  faster (recent-window minima), and REVISIT the loser with paired
 *  probes every 24 calls -- pairs because AutoNUMA migrates a private
 *  page on its second same-node fault -- relocking on a >3% win so a
 *  migration that completes late still flips the plan while ~half the
 *  driver's 30 samples remain.  Both configs are the same 1f-NT bit
 *  class (mt_r2 cmp-verified), so a lock flip can never put unchecked
 *  bits behind a scored number.  Placement is measured, not assumed:
 *  fr (% of sampled caller pages off node 0, get_mempolicy -- a pure
 *  read, explicitly allowed) is published at call 0 and at every lock.
 *
 *  MID (the B=2048 cell): a short on-buffer race over the cached
 *  full-team 1f trio {pfs, 520-pfs, none} -- one bit class -- because
 *  the phase-B out-store 4K-alias count is set by (scr - out) mod 4096,
 *  an allocation lottery the surrogate re-rolls (LITERATURE s4.5;
 *  L8_fusedaxes' fusedAA analysis).  2 settle calls + 3 trials each,
 *  lock needs >1.5% over the create pick, all inside the driver's
 *  calibration phase.  fusedaxes measured the same idea worth ~2% even
 *  with no NUMA asymmetry (their gov-off control, mt_r2).
 * ------------------------------------------------------------------ */
#if defined(_OPENMP) && defined(__AVX512F__) && defined(__linux__)
#define L8R_HAVE_GOV 1
#include <sys/syscall.h>
#ifndef MPOL_F_NODE
#define MPOL_F_NODE (1 << 0)
#endif
#ifndef MPOL_F_ADDR
#define MPOL_F_ADDR (1 << 1)
#endif

#define GOV_DEEP 1
#define GOV_MID  2
#define GOV_NCFG 3
#define GOV_RING 8

struct l8r_gov {
    int mode;                  /* GOV_DEEP or GOV_MID */
    int state;                 /* 0 dwell/settle, 1 probe/race, 2 locked */
    int n;                     /* execute calls seen */
    int ncfg, lock, flips;
    int dwell_end, probe_i, next_probe;
    int fr0, fr1, nb;          /* placement scans, -1 = unreadable */
    struct { rfn_t run; int ntf; const char *nm; } c[GOV_NCFG];
    double best[GOV_NCFG];             /* min ever, seconds/call */
    double ring[GOV_NCFG][GOV_RING];   /* last GOV_RING calls per config */
    int    rn[GOV_NCFG];
    double pair[2];            /* the current revisit probe pair */
};

/* % of sampled caller pages NOT on node 0 (32 pages per buffer).  Raw
 * syscall: no libnuma link dependency; a pure read of page homes. */
static int l8r_fr_scan(const double *ip, const double *op, int batch)
{
    size_t bytes = (size_t)batch * NDBL * sizeof(double);
    int rem = 0, tot = 0;
    for (int b = 0; b < 2; ++b) {
        const char *base = b ? (const char *)op : (const char *)ip;
        for (int i = 0; i < 32; ++i) {
            uintptr_t a = ((uintptr_t)base + (bytes / 32) * (size_t)i)
                          & ~(uintptr_t)4095;
            int node = -1;
            if (syscall(SYS_get_mempolicy, &node, NULL, 0UL, (void *)a,
                        (unsigned long)(MPOL_F_NODE | MPOL_F_ADDR)) == 0) {
                ++tot;
                if (node != 0) ++rem;
            }
        }
    }
    return tot ? (100 * rem + tot / 2) / tot : -1;
}

static int l8r_read_nb(void)
{
    FILE *f = fopen("/proc/sys/kernel/numa_balancing", "r");
    if (!f) return -1;
    int v = -1;
    if (fscanf(f, "%d", &v) != 1) v = -1;
    fclose(f);
    return v;
}

static double l8r_ring_min(const struct l8r_gov *g, int cfg)
{
    int nvis = g->rn[cfg] < GOV_RING ? g->rn[cfg] : GOV_RING;
    double m = 1e30;
    for (int i = 0; i < nvis; ++i)
        if (g->ring[cfg][i] < m) m = g->ring[cfg][i];
    return m;
}
#endif /* L8R_HAVE_GOV */

struct fft3d_plan {
    int    batch;
    int    nt;        /* chosen run uses non-temporal output stores */
    rfn_t  run;       /* whole-batch loop, kernel inlined, pf compile-time */
    rfn_t  run_safe;  /* same shape, ordinary stores (alignment fallback) */
    double *scr;      /* one volume of split-complex intermediate, 64B aligned */
    void   *scr_raw;
    const char *chosen;
#ifdef L8R_HAVE_GOV
    struct l8r_gov *gov;   /* NULL = no governor for this plan */
#endif
};

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

const char *fft3d_name(void) { return "L8_radix8"; }

/* Filled in by fft3d_create() with the tuner's pick, as the panel_r2 verdict
 * asked; the default text covers a call before any plan exists. */
static char g_desc[512] =
    "radix-8 split-complex 52-instr codelet, copy-free AVX-512 transposes; "
    "B=1 serial 2p, B>1 fused-1f chunks over a pinned spin pool (per-thread "
    "NUMA slots), NT/pfs/pfw x team tuned at create";

/* The governor appends its measured-on-the-real-buffers table here; the
 * leaderboard JSON captures whatever the string says at dump time. */
static char g_gov_desc[256] = "";
static char g_desc_full[792];

const char *fft3d_description(void)
{
    if (!g_gov_desc[0]) return g_desc;
    snprintf(g_desc_full, sizeof(g_desc_full), "%s %s", g_desc, g_gov_desc);
    return g_desc_full;
}

int fft3d_supports(int L) { return L == LL; }

#ifdef L8R_HAVE_GOV
static void l8r_gov_publish(const struct l8r_gov *g, int batch)
{
    if (g->mode == GOV_DEEP) {
        snprintf(g_gov_desc, sizeof(g_gov_desc),
                 "gov{fr0=%d,nb=%d,w=%.3f,h=%.3f,lock=%s,fl=%d,fr1=%d}",
                 g->fr0, g->nb,
                 1e6 * g->best[0] / (double)batch,
                 1e6 * g->best[1] / (double)batch,
                 g->c[g->lock].nm, g->flips, g->fr1);
    } else {
        snprintf(g_gov_desc, sizeof(g_gov_desc),
                 "govm{%s=%.4f,%s=%.4f,%s=%.4f,lock=%s}",
                 g->c[0].nm, 1e6 * g->best[0] / (double)batch,
                 g->c[1].nm, 1e6 * g->best[1] / (double)batch,
                 g->c[2].nm, 1e6 * g->best[2] / (double)batch,
                 g->c[g->lock].nm);
    }
}

/* One governed execute: choose the config for this call, run it timed,
 * advance the state machine.  Every call is a full correct execute of
 * the whole batch, and every config in a plan's table is the same bit
 * class, so lock flips are invisible to any correctness or
 * repeatability check. */
static void l8r_gov_execute(fft3d_plan *p, const double *ip, double *op)
{
    struct l8r_gov *g = p->gov;
    int cfg, probing = 0;

    if (g->n == 0) {
        g->fr0 = l8r_fr_scan(ip, op, p->batch);
        g->fr1 = g->fr0;
        g->nb  = l8r_read_nb();
    }

    if (g->mode == GOV_DEEP) {
        if (g->state == 0)      cfg = 0;             /* dwell: wide NT */
        else if (g->state == 1) cfg = 1;             /* half-team probe */
        else if (g->n == g->next_probe || g->n == g->next_probe + 1) {
            cfg = 1 - g->lock;                       /* paired revisit */
            probing = 1;
        } else cfg = g->lock;
    } else {                                          /* GOV_MID */
        if (g->state == 0)      cfg = 0;             /* settle */
        else if (g->state == 1) cfg = (g->n - 2) % g->ncfg;
        else                    cfg = g->lock;
    }

    if (g->mode == GOV_MID && g->state == 2) {       /* locked mid: no timing */
        g->c[cfg].run(ip, op, p->scr, p->batch);
#if defined(__SSE2__)
        if (g->c[cfg].ntf) _mm_sfence();
#endif
        g->n++;
        return;
    }

    double t0 = now_s();
    g->c[cfg].run(ip, op, p->scr, p->batch);
#if defined(__SSE2__)
    if (g->c[cfg].ntf) _mm_sfence();
#endif
    double dt = now_s() - t0;

    g->ring[cfg][g->rn[cfg] % GOV_RING] = dt;
    g->rn[cfg]++;
    if (dt < g->best[cfg]) g->best[cfg] = dt;

    if (g->mode == GOV_DEEP) {
        if (g->state == 0) {
            if (g->n + 1 >= g->dwell_end) { g->state = 1; g->probe_i = 0; }
        } else if (g->state == 1) {
            if (++g->probe_i >= 4) {
                /* recent-window minima: post-dwell wide reflects whatever
                 * migration the dwell bought, not the cold first calls */
                double w = l8r_ring_min(g, 0), h = l8r_ring_min(g, 1);
                g->lock  = (w < 0.98 * h) ? 0 : 1;
                g->state = 2;
                g->next_probe = g->n + 24;
                g->fr1 = l8r_fr_scan(ip, op, p->batch);
                l8r_gov_publish(g, p->batch);
            }
        } else if (probing) {
            g->pair[g->n == g->next_probe ? 0 : 1] = dt;
            if (g->n == g->next_probe + 1) {
                double o = g->pair[0] < g->pair[1] ? g->pair[0] : g->pair[1];
                double l = l8r_ring_min(g, g->lock);
                if (o < 0.97 * l) {
                    g->lock = 1 - g->lock;
                    g->flips++;
                    g->fr1 = l8r_fr_scan(ip, op, p->batch);
                }
                g->next_probe = g->n + 23;
                l8r_gov_publish(g, p->batch);
            }
        }
    } else {                                          /* GOV_MID */
        if (g->state == 0) {
            if (g->n + 1 >= 2) g->state = 1;
        } else if (g->n - 2 + 1 >= 3 * g->ncfg) {
            int am = 0;
            for (int v = 1; v < g->ncfg; ++v)
                if (g->best[v] < g->best[am]) am = v;
            g->lock  = (am != 0 && g->best[am] < 0.985 * g->best[0]) ? am : 0;
            g->state = 2;
            l8r_gov_publish(g, p->batch);
        }
    }
    g->n++;
}
#endif /* L8R_HAVE_GOV */

#define MAXCAND 12

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != LL || batch <= 0) return NULL;

    fft3d_plan *p = (fft3d_plan *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->batch = batch;

    /* Threads the harness granted (never raised).  Read once, kept for the
     * plan's lifetime in g_team, which the mt run wrappers use. */
    int team = 1;
#ifdef _OPENMP
    team = omp_get_max_threads();
    if (team > 32) team = 32;
    if (team < 1)  team = 1;
#endif
    g_team = team;

    /* One page-aligned SLOTD-double scratch slot per thread; slot 0 doubles
     * as the serial paths' scratch (SLOTD covers 8*SCRX and the 1f520
     * layout, compile-checked above).  Each slot is FIRST-TOUCHED BY ITS
     * OWNING THREAD so it is NUMA-local on the two-socket node; the same
     * parallel region spins up the OpenMP pool, so the first timed execute
     * creates no thread. */
    size_t scrd = (size_t)team * SLOTD;
    if (posix_memalign(&p->scr_raw, 4096, scrd * sizeof(double)) != 0 || !p->scr_raw) {
        free(p);
        return NULL;
    }
    p->scr = (double *)p->scr_raw;
#ifdef _OPENMP
    for (int t = 0; t < 32; ++t) g_cpumap[t] = -1;
#pragma omp parallel num_threads(team)
    {
        int t = omp_get_thread_num();
        if (t < team) {
            if (t < 32) g_cpumap[t] = sched_getcpu();
            memset(p->scr + (size_t)t * SLOTD, 0, SLOTD * sizeof(double));
        }
    }
    /* mt_r2: persistent pinned spin pool for every B>1 plan (workers pin to
     * the close/cores CPUs just read back; scratch slot t was first-touched
     * by OMP thread t on the same CPU, so NUMA locality carries over).
     * B=1 builds NO pool -- it stays literally serial, zero threads.
     * L8R_POOL=0 forces the OMP-region dispatcher for A/Bs. */
    {
        const char *pe = getenv("L8R_POOL");
        if (batch > 1 && team > 1 && !(pe && pe[0] == '0'))
            g_pool = l8r_pool_create(team - 1);
    }
#else
    memset(p->scr, 0, scrd * sizeof(double));
#endif

    /* ---- regime gate.  Prefetch/NT candidates are offered only when the batch's
     * in+out footprint exceeds ~0.9x the last-level cache: at cache-resident sizes
     * both are documented losers (L8_fusedaxes r2: prefetching L3-resident input =
     * 1.6x; node r3 at B=64: pf pick = +6.7 %), and offering them there is what
     * made the r3 tuner flip picks between runs. ---- */
    long l3 = -1;
#ifdef _SC_LEVEL3_CACHE_SIZE
    l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
    if (l3 <= 0) l3 = 22l << 20;   /* the scoring node's 22 MiB */
    size_t ws = (size_t)batch * 2u * NDBL * sizeof(double);   /* in + out bytes */
    int big = (ws * 10 > (size_t)l3 * 9) || getenv("L8R_ALLCAND") != NULL;

    /* ---- candidates on the best available backend (AVX-512 vs AVX2 settled on
     * the node in panel_r1).  FIRST entry is the node-validated default for the
     * regime (r3 picks: 2p at B=1 3/3, 3p at B=64 when it won, 3p-pf at
     * B=2048/16384 3/3); displacing it requires a >2 % measured win.  The burst-pf
     * NT combination is deliberately absent: a pass-1 prefetch burst and the NT
     * store drain fight for fill buffers (L8_fusedaxes r2, 1.6x). ---- */
    rfn_t       cand[MAXCAND];
    rfn_t       csafe[MAXCAND];
    const char *cname[MAXCAND];
    int         cnt[MAXCAND];   /* non-temporal flag */
    int         cinst[MAXCAND]; /* installable: candidate may be picked.  0 =
                                 * probe: timed and published ('*' in the arena
                                 * string), never picked -- keeps every regime's
                                 * pickable pool ONE BIT CLASS so a cross-process
                                 * pick flip can never put unchecked bits behind
                                 * the leaderboard minimum (r10 VERDICT s3(a);
                                 * rule from L36_mixedradix).  L8R_FORCE still
                                 * overrides for dev A/Bs. */
    int         cteamc[MAXCAND];/* team class: 0 serial, 1 full, 2 half --
                                 * drives the post-pick pool shrink */
    int nc = 0;
    int notune = 0;   /* 1 = candidates are timed and published, never picked */

#define ADDC(fn, safe, name, ntf, ins, tc)                                     \
    do { cand[nc] = (fn); csafe[nc] = (safe); cname[nc] = (name);              \
         cnt[nc] = (ntf); cinst[nc] = (ins); cteamc[nc] = (tc); ++nc; } while (0)

#if defined(__AVX512F__)
    if (batch == 1) {
        /* B=1, panel_r11: HARDWIRED to 2p.  This executes my own r10
         * pre-registered fork: the de-aliased fused shape (1f520) shipped
         * fixed in r10 and read 0.5760 with its arena showing 1f520 == 1f
         * (0.569/0.567, 0.574/0.572 x2) -- the "fix bought nothing, the tax
         * is code layout, r11 hardwires 2p and B=1 is closed here" branch.
         * The r10 node arena also ranked 2p FIRST in all three runs
         * (0.560/0.569/0.569), agreeing for the first time with the r8
         * driver measurement (2p 0.5700 vs 1f 0.5813/0.5829).  No
         * tournament (L8_batchsimd's r9 rule stands): candidates below are
         * timed and published, L8R_FORCE works for A/Bs, nothing displaces
         * the pick.  The 1f520j association-order probe is RETIRED from the
         * timed set -- the node answered it in r10 (+0.5..0.7 %, VERDICT
         * closed the propagation ask); the kernel stays compiled. */
        ADDC(run_2p_avx512,     run_2p_avx512,     "avx512-2p",     0, 1, 0);
        ADDC(run_1f520_avx512,  run_1f520_avx512,  "avx512-1f520",  0, 0, 0);
        ADDC(run_1f_avx512,     run_1f_avx512,     "avx512-1f",     0, 0, 0);
        ADDC(run_3p_avx512,     run_3p_avx512,     "avx512-3p",     0, 0, 0);
        notune = 1;
    } else if (ws > 3u * (size_t)l3) {
        /* DEEP-STREAMING regime (mt_r2; B=32768 on the node, 512 MiB vs
         * 22 MiB L3).  All caller pages live on socket 0 (driver fread /
         * memset on its main thread), so the cell is socket-0-DRAM-bound
         * at ~100 GB/s: fftw's winning 0.161 us/vol is exactly 16 KB over
         * that wall, cached stores' 24 KB/volume price out at ~0.24 (the
         * bad r1 scored run), and the r1 arena had mth-3p-nt-pfs at 0.153
         * with L8_fusedaxes' scored pick the same shape (fused-nt+pfs/16,
         * their arena 0.176 < nt/24 0.186 < nt/32 0.208).  So the default
         * is NT + HALF TEAM (16 close threads = socket 0, no UPI, no
         * RFO), on the fused 1f chunk kernel.  ONE BIT CLASS: every
         * installable is the 1f shape (y,x,z axis order); the 3p family
         * is demoted to probes.  NT candidates fall back to their cached
         * twin when the buffers are ever unaligned. */
        ADDC(run_mth_1f_nt_pfs_avx512, run_mth_1f_pfs_avx512, "mth-1f-nt-pfs", 1, 1, 2);
        ADDC(run_mt_1f_nt_pfs_avx512,  run_mt_1f_pfs_avx512,  "mt-1f-nt-pfs",  1, 1, 1);
        ADDC(run_mth_1f_pfs_avx512,    run_mth_1f_pfs_avx512, "mth-1f-pfs",    0, 1, 2);
        ADDC(run_mt_1f_avx512,         run_mt_1f_avx512,      "mt-1f",         0, 1, 1);
        ADDC(run_mt_1f_pfs_avx512,     run_mt_1f_pfs_avx512,  "mt-1f-pfs",     0, 1, 1);
        /* probes: the mt_r1 3p class (r1 picks at this cell), never picked */
        ADDC(run_mth_3p_nt_pfs_avx512, run_mth_3p_pfs_avx512, "mth-3p-nt-pfs", 1, 0, 2);
        ADDC(run_mt_3p_pfs_avx512,     run_mt_3p_pfs_avx512,  "mt-3p-pfs",     0, 0, 1);
    } else {
        /* MID/STREAMING batch regime (mt_r2; B=2048 on the node, 32 MiB vs
         * 22 MiB L3).  The fused 1f chunk kernel replaces 3p: the node's
         * B=2048 cell went to the two fused-shape rivals at 0.026-0.028
         * us/vol (fused+pfs/32 and T32/s0, 3/3 each) while my threaded 3p
         * read 0.046 in-arena -- the 3p shape's extra 128+128 L1 scratch
         * traffic and third store pass are pure loss at 32 threads.
         * Cached full team is the default; NT lost 3.5x here on the node
         * (0.177 vs 0.046, 3/3) but stays installable (same bits) for
         * footprints between the gates.  mt-1f (no prefetch) is
         * installable on L8_batchsimd's B=64 arena (prefetch +72 % at 32
         * threads).  The serial candidate covers small batches where the
         * dispatch can exceed the work (L13_direct mt_r1). */
        ADDC(run_mt_1f_pfs_avx512,     run_mt_1f_pfs_avx512,  "mt-1f-pfs",     0, 1, 1);
        ADDC(run_mt_1f_avx512,         run_mt_1f_avx512,      "mt-1f",         0, 1, 1);
        ADDC(run_mt_1f_pfsw_avx512,    run_mt_1f_pfsw_avx512, "mt-1f-pfw",     0, 1, 1);
        /* de-aliased scratch twin (si at +520): same bit class; the real
         * decision is the governor's on-buffer race (mt_r3) */
        ADDC(run_mt_1f520_pfs_avx512,  run_mt_1f520_pfs_avx512, "mt-1f520-pfs", 0, 1, 1);
        ADDC(run_mth_1f_pfs_avx512,    run_mth_1f_pfs_avx512, "mth-1f-pfs",    0, 1, 2);
        ADDC(run_mt_1f_nt_pfs_avx512,  run_mt_1f_pfs_avx512,  "mt-1f-nt-pfs",  1, 1, 1);
        if (batch <= 4 * team)
            ADDC(run_1f_pfs_avx512, run_1f_pfs_avx512, "st-1f-pfs", 0, 1, 0);
        /* probe: the mt_r1 pick at this cell, cross-family reference */
        ADDC(run_mt_3p_pfs_avx512,     run_mt_3p_pfs_avx512,  "mt-3p-pfs",     0, 0, 1);
        (void)big;
    }
#elif defined(__AVX2__) && defined(__FMA__)
    if (batch == 1) {
        ADDC(run_2p_avx2, run_2p_avx2, "avx2-2p", 0, 1, 0);
        ADDC(run_3p_avx2, run_3p_avx2, "avx2-3p", 0, 0, 0);
    } else {
        /* all 3p bit class, threaded over volume chunks */
        ADDC(run_mt_3p_nt_pfs_avx2, run_mt_3p_pfs_avx2, "mt-3p-nt-pfs", 1, 1, 1);
        ADDC(run_mt_3p_pfs_avx2,    run_mt_3p_pfs_avx2, "mt-3p-pfs",    0, 1, 1);
        ADDC(run_mt_3p_avx2,        run_mt_3p_avx2,     "mt-3p",        0, 1, 1);
        if (batch <= 4 * team)
            ADDC(run_3p_pfs_avx2, run_3p_pfs_avx2, "st-3p-pfs", 0, 1, 0);
        (void)big;
    }
#else
    if (batch == 1) {
        ADDC(run_2p_gen, run_2p_gen, "gen-2p", 0, 1, 0);
        ADDC(run_3p_gen, run_3p_gen, "gen-3p", 0, 0, 0);
    } else {
        ADDC(run_mt_3p_gen, run_mt_3p_gen, "mt-3p", 0, 1, 1);
        ADDC(run_3p_gen,    run_3p_gen,    "st-3p", 0, 1, 0);
        (void)big;
    }
#endif
#undef ADDC

    int pick  = 0;   /* the regime default */
    int tuned = 0;
    /* In-arena candidate times, published through fft3d_description() so the
     * leaderboard itself shows whether this plan's arena ranks the candidates
     * the way the driver's own measurement does (L8_batchsimd r8 documented
     * the arena-vs-driver inversion; L36_pfa's create-side-measurement
     * pattern, endorsed panel-wide by the r8 VERDICT). */
    char arena[224] = "";

    if (nc > 1) {
        /* Self-timing at the real batch size, capped machine-relatively at
         * 8x L3 of volumes (raised from 4x in mt_r2: L8_fusedaxes' mt_r1
         * surrogate at ~2x L3 crowned a pick the driver then ran 27 %
         * slower; their fix was >= 8x L3, adopted.  11264 volumes on the
         * node, 16384 cap elsewhere). */
        long cap = (long)(8.0 * (double)l3 / (2.0 * NDBL * sizeof(double)));
        if (cap < 8192)  cap = 8192;
        if (cap > 16384) cap = 16384;
        int nb = batch < (int)cap ? batch : (int)cap;
        size_t vol = NDBL * sizeof(double);
        void *raw_in = NULL, *raw_out = NULL;
        if (posix_memalign(&raw_in, 64, (size_t)nb * vol) == 0 && raw_in &&
            posix_memalign(&raw_out, 64, (size_t)nb * vol) == 0 && raw_out) {
            double *ti = (double *)raw_in, *to = (double *)raw_out;
            for (size_t i = 0; i < (size_t)nb * NDBL; ++i)
                ti[i] = 0.5 + 1e-3 * (double)(i % 37);
            memset(to, 0, (size_t)nb * vol);

            /* aim for ~3 ms of work per round per candidate (raised from
             * 1.5 in mt_r2: L8_fusedaxes' reps=1 trials coin-flipped a
             * 0.5 % tie across runs; >= 2 reps / ~2+ ms made pick strings
             * identical across runs).  The us/vol estimate drops to the
             * threaded rate when the pool is threaded. */
            double est = (batch > 1 && team > 1) ? 0.08 : 0.7;
            long reps = (long)(3000.0 / (est * (double)nb));
            if (reps < 2) reps = 2;
            if (reps > 4000) reps = 4000;

            /* warm-up: settle the frequency licence before anything is believed */
            long nw = (long)(1000.0 / (est * (double)nb));
            if (nw < 2) nw = 2;
            if (nw > 3000) nw = 3000;

            double best[MAXCAND];
            for (int v = 0; v < nc; ++v) {
                best[v] = 1e30;
                for (long w = 0; w < nw; ++w)
                    cand[v](ti, to, p->scr, nb);
            }
            /* Trials round-robin interleaved so frequency drift cannot favour
             * whoever runs first; ONE UNTIMED PASS before each timed block so a
             * candidate is measured against its own cache state, not its
             * predecessor's (plain and NT leave different L3 contents behind --
             * protocol borrowed from L8_fusedaxes r3). */
            for (int round = 0; round < 7; ++round) {
                for (int v = 0; v < nc; ++v) {
                    cand[v](ti, to, p->scr, nb);
                    double t0 = now_s();
                    for (long i = 0; i < reps; ++i)
                        cand[v](ti, to, p->scr, nb);
#if defined(__SSE2__)
                    _mm_sfence();
#endif
                    double dt = now_s() - t0;
                    if (dt < best[v]) best[v] = dt;
                }
            }
            /* argmin over INSTALLABLE candidates only; probes are timed and
             * published but can never be picked (single-bit-class pools) */
            int am = 0;
            for (int v = 1; v < nc; ++v)
                if (cinst[v] && best[v] < best[am]) am = v;
            if (!notune && am != 0 && best[am] < best[0] * 0.98) {
                pick  = am;
                tuned = 1;
            }

            /* compact arena table for the description: candidate short names
             * (strip only an ISA prefix -- "mt"/"mth"/"st" are load-bearing;
             * '*' marks a non-installable probe) with us-per-volume */
            {
                size_t off = 0;
                for (int v = 0; v < nc && v < 9; ++v) {
                    const char *nm = cname[v];
                    if (strncmp(nm, "avx512-", 7) == 0)     nm += 7;
                    else if (strncmp(nm, "avx2-", 5) == 0)  nm += 5;
                    else if (strncmp(nm, "gen-", 4) == 0)   nm += 4;
                    int w = snprintf(arena + off, sizeof(arena) - off,
                                     "%s%s%s=%.3f", off ? " " : "", nm,
                                     cinst[v] ? "" : "*",
                                     1e6 * best[v] / ((double)reps * (double)nb));
                    if (w < 0 || (size_t)w >= sizeof(arena) - off) break;
                    off += (size_t)w;
                }
            }

            /* create-time override for A/B runs: L8R_FORCE=<candidate name> */
            const char *force = getenv("L8R_FORCE");
            if (force)
                for (int v = 0; v < nc; ++v)
                    if (strcmp(force, cname[v]) == 0) pick = v;

            if (getenv("L8R_TUNE_DEBUG")) {
                fprintf(stderr, "[L8_radix8 tune] batch=%d nb=%d reps=%ld big=%d pick=%s |",
                        batch, nb, reps, big, cname[pick]);
                for (int v = 0; v < nc; ++v)
                    fprintf(stderr, " %s=%.4fus", cname[v],
                            1e6 * best[v] / ((double)reps * (double)nb));
                fprintf(stderr, "\n");
            }
        }
        free(raw_in);
        free(raw_out);
        memset(p->scr, 0, (size_t)8 * SCRX * sizeof(double));
    }

    p->run      = cand[pick];
    p->run_safe = csafe[pick];
    p->chosen   = cname[pick];
    p->nt       = cnt[pick];

#ifdef L8R_HAVE_GOV
    /* Arm the execute-time governor (see the block comment above struct
     * l8r_gov).  DEEP needs the full pool alive for team switching; MID
     * only re-races the cached full-team 1f trio, so it arms only when
     * the create pick is one of those three (an NT or half-team create
     * pick keeps the create decision untouched). */
    g_gov_desc[0] = '\0';
    {
        const char *ge = getenv("L8R_GOV");
        int deep = ws > 3u * (size_t)l3;
        if (!(ge && ge[0] == '0') && !getenv("L8R_FORCE") &&
            batch > 1 && team > 1 && nc > 1) {
            struct l8r_gov *g = NULL;
            if (deep && batch >= 4096 && team >= 8) {
                g = (struct l8r_gov *)calloc(1, sizeof *g);
                if (g) {
                    g->mode = GOV_DEEP;
                    g->ncfg = 2;
                    g->c[0].run = run_mt_1f_nt_pfs_avx512;
                    g->c[0].ntf = 1;
                    g->c[0].nm  = "mt-1f-nt-pfs";
                    g->c[1].run = run_mth_1f_nt_pfs_avx512;
                    g->c[1].ntf = 1;
                    g->c[1].nm  = "mth-1f-nt-pfs";
                    g->dwell_end = 40;
                    const char *de = getenv("L8R_DWELL");
                    if (de) {
                        int d = atoi(de);
                        if (d >= 4 && d <= 200) g->dwell_end = d;
                    }
                }
            } else if (!deep && batch >= 64 &&
                       (p->run == run_mt_1f_pfs_avx512 ||
                        p->run == run_mt_1f520_pfs_avx512 ||
                        p->run == run_mt_1f_avx512)) {
                g = (struct l8r_gov *)calloc(1, sizeof *g);
                if (g) {
                    g->mode = GOV_MID;
                    g->ncfg = 3;
                    g->c[0].run = run_mt_1f_pfs_avx512;    g->c[0].nm = "mt-1f-pfs";
                    g->c[1].run = run_mt_1f520_pfs_avx512; g->c[1].nm = "mt-1f520-pfs";
                    g->c[2].run = run_mt_1f_avx512;        g->c[2].nm = "mt-1f";
                    /* create pick first: it is the hysteresis anchor */
                    for (int v = 1; v < 3; ++v)
                        if (p->run == g->c[v].run) {
                            rfn_t tr = g->c[0].run;
                            const char *tm = g->c[0].nm;
                            g->c[0] = g->c[v];
                            g->c[v].run = tr;
                            g->c[v].nm  = tm;
                        }
                }
            }
            if (g) {
                for (int v = 0; v < GOV_NCFG; ++v) g->best[v] = 1e30;
                g->fr0 = g->fr1 = g->nb = -1;
                p->gov = g;
            }
        }
    }
#endif

#ifdef _OPENMP
    /* Shrink the pool to the picked team (L36_pfa via L8_fusedaxes mt_r1:
     * unpicked spinners drag the all-core clock).  A serial pick tears the
     * pool down entirely; a half-team pick rebuilds with (team+1)/2 - 1
     * workers, still pinned to the FIRST close CPUs = one socket on the
     * node.  A failed rebuild just falls back to the OMP region.
     * mt_r3: a governed plan keeps the FULL pool -- the governor switches
     * team width per call (job.T), and the mt_r2 node arena showed the
     * idle-spinner cost at these DRAM-bound cells is ~0 (mth in-arena
     * 0.173 with 31 spinners == 0.174 scored with a shrunk pool). */
    if (g_pool
#ifdef L8R_HAVE_GOV
        && !p->gov
#endif
        ) {
        int tneed = cteamc[pick] == 2 ? (team + 1) / 2
                  : cteamc[pick] == 1 ? team : 1;
        if (tneed <= 1) {
            l8r_pool_destroy(g_pool);
            g_pool = NULL;
        } else if (tneed - 1 < g_pool->nw) {
            l8r_pool_destroy(g_pool);
            g_pool = l8r_pool_create(tneed - 1);
        }
    }
#endif

    snprintf(g_desc, sizeof(g_desc),
             "radix-8 52-instr codelet; B=1 serial, B>1 pool-parallel 1f "
             "(3p probes); pick[B=%d,T=%d]=%s (%s%s) arena{%s}",
             batch, team, p->chosen,
             notune ? "fixed" : (tuned ? "tuned" : "default"),
#ifdef _OPENMP
             (batch > 1 && team > 1) ? (g_pool ? ",pool" : ",omp") :
#endif
             "", arena);
    return p;
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    const double *ip = (const double *)in;
    double *op = (double *)out;
    /* The ABI guarantees 64-byte-aligned buffers, which the non-temporal stores need;
     * fall back to the ordinary-store twin if that ever stops being true. */
    const int aligned = (((uintptr_t)op | (uintptr_t)ip) & 63u) == 0u;
#ifdef L8R_HAVE_GOV
    if (plan->gov && aligned) {
        l8r_gov_execute(plan, ip, op);   /* fences its own NT configs */
        return;
    }
#endif
    (aligned ? plan->run : plan->run_safe)(ip, op, plan->scr, plan->batch);
#if defined(__SSE2__)
    /* Non-temporal stores are weakly ordered; make the batch visible before returning.
     * One fence per execute, not per volume. */
    if (plan->nt && aligned) _mm_sfence();
#endif
}

void fft3d_destroy(fft3d_plan *plan)
{
    if (!plan) return;
#ifdef _OPENMP
    l8r_pool_destroy(g_pool);
    g_pool = NULL;
#endif
#ifdef L8R_HAVE_GOV
    free(plan->gov);
#endif
    free(plan->scr_raw);
    free(plan);
}
