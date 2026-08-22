/* Carried over from the SINGLE-THREAD competition, where this file finished as
 * written below. Your job in the multicore phase is to parallelise it across
 * 32 cores without losing its single-core efficiency -- read
 * ../PANEL_BRIEF.md, and read ../../geom/strategies/L17_rader.md for the full
 * history of how this kernel got here.
 */
/* L = 17, complex-double forward 3D DFT, batched, single-threaded.
 *
 * NEW THIS ROUND (panel_r10) -- one mechanism, aimed squarely at the r9
 * VERDICT's named direction for L=17 ("fund traffic deletion at B=256/2048";
 * every write-SPREADING mechanism -- my sp grid, winograd's q+pfw -- was
 * declined 6/6 by the node's own tuners in r9, so this round deletes traffic
 * instead of rescheduling it):
 *
 * 1. STAGED DENSE OUT FLUSH ("st" / "st dy" candidates, L17R_FORCE 13/14).
 *    The x pass's interleaving store emits 17 concurrent k-row streams of
 *    16-byte-aligned 128-byte pieces strided 4624 B apart.  At batch these
 *    are DRAM-destined partial-line RFOs: a piece at 16-B misalignment
 *    touches 3 cache lines, the boundary lines are only completed by the
 *    NEXT block of the same stream (~300 cycles later), and 17 concurrent
 *    streams exceed the ~12 fill buffers -- so write-combining lines are
 *    evicted half-filled and the same `out` line is fetched and written
 *    back more than once.  That waste is the one traffic term the fused
 *    rivals do not pay (matrixsimd's chunk store and winograd's g8 both
 *    emit finished planes densely), and my r9 probe put the output-side
 *    exposure at xp = 6.77 us/vol against a ~4 us compute share.  The st
 *    variants point the x-pass stores at a 78.6 KB L2-resident staging
 *    volume (vo) and then flush it to `out` as ONE sequential dense stream
 *    of full lines (vo_flush; pfw composes as a paced prefetchw ahead of
 *    the flush).  Costs ~157 KB of extra L2 round trip and ~2.5k movement
 *    uops (~0.3 us); deletes the partial-line refetch waste.  Same kernel
 *    operands, same values to the same final places: BIT-IDENTICAL, class
 *    A, stage-1-ranked and given the joint (variant, pf, pfw) grid's
 *    partner slot that sp held in r9 (sp itself: declined 6/6, demoted).
 *    At B=1 `out` is L2-resident and staging is pure overhead -- the tuner
 *    will (correctly) keep the incumbent there; this is a batch mechanism.
 *
 * 2. PIPELINED FLUSH ("stp" / "stp dy", L17R_FORCE 15/16).  The forced
 *    wallaby A/B of the immediate flush read +23% at B=2048 streaming: an
 *    un-paced flush is a serial burst with no compute behind it, fully
 *    exposed even when it deletes waste.  stp keeps the staged x pass and
 *    paces the flush of volume b-1 across volume b's plane phase, one 4.5 KB
 *    sequential chunk after each plane's y pass (vo ping-ponged).  This is
 *    NOT a rerun of the falsified sp: sp's paced pieces were kernel blocks
 *    with 17-stream scattered partial-line stores; stp's are single dense
 *    sequential streams, the pattern prefetchers and fill buffers handle at
 *    full rate.  Bit-identical (only the out-write order moves across
 *    independent volumes); the stage-1 rank orders all four staged shapes
 *    and the joint grid races the best of them against the incumbent over
 *    (pf, pfw).
 *
 * PREVIOUS ROUND (panel_r9) -- all three aimed at the batched cells, where
 * this entry trails the fused rivals ~15-20% while B=1 sits at the panel-wide
 * structural limit (r8 VERDICT: four mechanism classes falsified at 1.31x
 * floor):
 *
 * 1. "dy" CANDIDATES (ymm deint inside the otherwise-zmm w8 pipeline).
 *    r8's zmm transposes were -3.0% at B=1/B=8 on the node but REGRESSED the
 *    batched cells +1.7%/+1.4%.  The only stage whose behaviour differs
 *    between the regimes is the deinterleave: its source loads from `in` are
 *    the one unaligned zmm stream r8 added, and at batch they hit COLD
 *    lines, where a 64-byte load at the 16-byte alignment classes of an
 *    odd-length complex plane splits a cache line 3/4 of the time (vs ~1/2
 *    for 32-byte ymm loads).  "xl 512t dy" / "xl 512t sp dy" keep every zmm
 *    transpose and swap only the deint tile back to ymm; bit-identical, the
 *    tuner ranks them per cell (expected: dy wins at batch, plain zmm at
 *    B=1/B=8 -- both regimes keep their r8 win).
 *
 * 2. JOINT (variant, pf, pfw) GRID at batch -- ADOPTED FROM L23_rader
 *    panel_r8 (their joint grid found plain-xf + pf=2 + pw=1, a combination
 *    stage-1-then-grid ranking can never select, and it took their B=128
 *    cell).  The sp write-spreading variant now gets its (pf, pfw) shot
 *    alongside the stage-1 winner: sp's r7 node rejection was measured at
 *    (0,0) only, and its mechanism (spread the out burst across compute)
 *    composes with pfw (prefetchw what it is about to store).  Class-A only,
 *    so a measured variant switch cannot change output bits.
 *
 * 3. STREAMING DECOMPOSITION PROBE in fft3d_create(), reported in the
 *    description string as "probe ph/xp/fu" (us/volume on the streaming
 *    arena, (pf,pfw)=(0,0)) -- the pattern the r8 VERDICT says should become
 *    the panel default (L36_pfa's in-plan node probe).  ph = plane phase
 *    alone (cold `in` reads + A fill: input-side exposure), xp = x pass
 *    alone (hot A reads + `out` burst: output-side exposure), fu = full
 *    plain exec.  On the node this attributes my ~4 us/volume batched gap
 *    to the input or output side in one leaderboard line; the r10 move
 *    (respread the burst vs restage the input) reads directly off it.
 *
 * PREVIOUS ROUND (panel_r8):
 *
 * 1. 512-BIT PLANE TRANSPOSES in the w8 pipeline (tz8x8 / dz8x8 blocks) --
 *    the panel_r7 VERDICT's L=17 synthesis executed: the node rejected three
 *    scheduling attacks on the non-FP residue (ov r5, dz r7, matrixsimd's
 *    deferred-Z at the small cells) and rewarded exactly one mechanism, uop
 *    DELETION (L17_winograd's g8, -8..-10% in all four cells).  My residue
 *    is dominated by the serialized per-plane transpose/deinterleave loops,
 *    which were still round-1's 4x4 ymm tiles even inside the 512-bit
 *    pipeline.  Replaced with 8x8 zmm blocks: the classic 24-shuffle
 *    3-stage network (8 loads + 24 shuffles + 8 stores per 64 elements
 *    against 64 uops at 4x4), and an interleaved-complex 8x8 tile at
 *    16 + 48 + 16 = 80 uops per 64 complex against 128.  ~6.5k uops deleted
 *    per volume, roughly half of them load/store slots on a 2-load-port
 *    CLX.  Pure data movement -- identical values to identical places -- so
 *    every class-A candidate stays bit-identical by construction.  Measured
 *    (wallaby, same-window forced 512t A/B): B=1 8.80-8.83 vs 9.06 old
 *    (-2.8%), B=256 no worse.  The w4 path keeps the ymm tiles.
 *
 * PREVIOUS ROUND (panel_r7):
 *
 * 1. DEFERRED-JUNCTION PLANE SCHEDULE ("dz" variants, mixed-width only) --
 *    ADOPTED FROM L17_matrixsimd panel_r6 ("deferred-Z": defer the consuming
 *    group one slot behind its producer, double-buffering the plane buffer;
 *    measured -3.0% at B=1 and -5.7% at B=8 on wallaby for their structure,
 *    the only new L=17 mechanism of that round with a positive same-structure
 *    number).  My plane phase has THREE store->load junctions per plane, all
 *    back-to-back in exec_body:
 *        deint(x) stores T   -> z(x) loads T
 *        z(x)     stores T   -> transpose(x) loads T   (in place)
 *        transpose(x) stores U -> y(x) loads U
 *    Each junction is a group tail whose stores nothing hides, followed by a
 *    dependent load with no independent work in between.  exec_dz_body runs
 *    the SAME groups software-pipelined one plane deep, with T double-
 *    buffered by plane parity (T0/T1; U stays single -- its producer and
 *    consumer sit in the same iteration, one kernel group apart):
 *        iter x:  transpose(x-1)  T[(x-1)&1] -> U
 *                 z(x)            in place on T[x&1]
 *                 y(x-1)          U -> A[x-1]
 *                 deint(x+1)      in[x+1] -> T[(x+1)&1]
 *    Now every junction has at least one full independent group between
 *    producer and consumer (z->transpose has two).  Unlike panel_r5's ov --
 *    which chased the same non-FP cycles by SPLITTING the transpose loops
 *    into halves and slotting them into kernel drains, and lost to its own
 *    plumbing -- dz moves whole groups and adds ZERO instructions.  Same
 *    kernel calls, same operand values, same within-pass order: bit-identical
 *    to every class-A candidate (cmp-verified), so the tuner ranks it freely.
 *    Cost: one extra T pair (+6.5 KB scratch; plane-phase footprint 19.6 KB
 *    of a 32 KB node L1d).  Three candidates: dz, dz+pin, and dz+sp (the dz
 *    plane schedule with panel_r6's cross-volume x-block pipelining, for the
 *    batched cells).
 *
 * PREVIOUS ROUND (panel_r6):
 *
 * 1. SOFTWARE-PIPELINED X-LAST ("sp", mixed-width only) -- volume b's x-pass
 *    output burst (37 kernel blocks writing 78.6 KB of `out`, at batch a
 *    DRAM-destined burst that runs with the FMA stream idle on stores) is
 *    interleaved into volume b+1's plane phase: 2-3 x-pass blocks of the
 *    PREVIOUS volume run after each plane's y pass, using ping-pong A
 *    buffers so A(b) stays live while the plane phase fills A(b+1).  The
 *    per-volume arithmetic and kernel-call operands are unchanged, only the
 *    global order moves across volume boundaries, so the output is
 *    BIT-IDENTICAL to the other class-A candidates and the tuner ranks it
 *    freely.  This is r4's "Next" item 3, the monitor-named remaining lever
 *    for L=17 batched (~1.39x of un-overlapped memory time at B=2048).  The
 *    x-pass block runs behind a noinline+noclone helper so the pipeline adds
 *    ZERO extra inlined kernel copies (I-footprint discipline, r2's 38 KB
 *    kill line).
 *
 * 2. PACED WRITE-INTENT PREFETCH ("pfw") on the x pass's `out` stores --
 *    ADOPTED FROM the panel_r5 node result at L=8 (L8_fusedaxes, fused+pfs+
 *    pfw, B=2048 -31%) and L=36 (L36_pfa, inplace pf=2, B=256 -16.6%): on
 *    Cascade Lake, HIDE the RFO with prefetchw rather than avoid it with NT
 *    stores (NT lost on the node 4 rounds running).  Before x-pass block k
 *    runs, the 17 destination row regions of block k+2 are prefetched with
 *    write intent.  Plan-time flag, A/B'd in the streaming stage jointly
 *    with pf (4 configs, 3% margin, blocked): prefetchw on cache-resident
 *    lines is pure uop tax (L36_pfa: +13% at B=1), so it must stay gated.
 *
 * 3. PER-CANDIDATE LICENCE WARMUP in the tuner: each candidate now runs
 *    itself for >= 1.5 ms (not just 2 execs) before being timed.  The r5
 *    node probes measured clk256 = 3.89 vs clk512 = 2.89 GHz, and Intel's
 *    licence-up dwell is ~670 us -- so a ymm candidate ranked right after a
 *    zmm one (or after the zmm settle spin) was being measured inside the
 *    AVX-512 licence.  At the B=1 stage (nv <= 16, one exec ~ 200-350 us)
 *    the old 2-exec warmup could not cover the dwell; the time-based warmup
 *    does, so a 256-bit candidate finally gets an honest clock.  (Panel_r5
 *    VERDICT section 5: the licence-transition synthesis, and pencilfused's
 *    per-candidate self-warming fix, adapted.)
 *
 * PREVIOUS ROUND (panel_r5):
 *
 * 1. OVERLAPPED-SHUFFLE VARIANTS ("ov", mixed-width only).  The r4 VERDICT
 *    measured the node's sustained AVX2 clock at 3.89 GHz (not 2.30), which
 *    re-prices B=1: 17.74 us is ~69k cycles against a ~36k-cycle FP floor, so
 *    roughly half the runtime is NOT arithmetic.  The prime suspects are the
 *    per-plane transpose loops (deint_transpose17 + 2x transpose17, ~1.1k
 *    port-5-bound uops per plane) which run SERIALIZED between kernel calls:
 *    while they run, the FMA port is idle, and while a zmm kernel block
 *    drains its ~296 port-0 uops (1/cycle on the Gold 5218's single 512-bit
 *    FMA unit), alloc (4/cycle) runs far ahead -- younger independent
 *    shuffle/load uops issue on ports 5/2/3 essentially for free.  The ov
 *    exec bodies reorder the SAME operations so every shuffle burst sits in
 *    a zmm block's shadow:
 *      - z pass runs its ymm tail FIRST, zmm blocks last, so the T->U
 *        transpose that follows lands in a zmm drain;
 *      - T->U is split in half by output column range: cols 0..7 before the
 *        y pass, cols 8..16 in the shadow of the y pass's first zmm block;
 *      - the NEXT plane's input deinterleave (dead T buffer by then) is
 *        split in half and slotted after the y pass's second zmm block and
 *        after its ymm tail.
 *    Pure reordering of independent ops on disjoint regions: bit-identical
 *    to the other class-A variants (verified), so the plan-time tuner ranks
 *    it like any candidate.  Wallaby (TWO 512-bit FMA units -> drains are
 *    half as long, less shadow) will understate the node benefit, exactly
 *    like r4's mixed-width tail bet.
 *
 * 2. DUAL-WIDTH CLOCK PROBE, reported in fft3d_description() as
 *    clk256/clk512 -- the r4 VERDICT's explicit L=17 ask ("measure the
 *    AVX-512 licence clock, then re-derive").  Adopted from L6_unrolled's
 *    panel_r4 probe (serially dependent FMA chain, latency 4 on
 *    SKX/CLX/ICL/SPR, freq = iters*4/time), extended with a 512-bit chain.
 *    Runs after the tuning tournament (core warm), ~20 ms, unscored.
 *
 * 3. THE X-FIRST/X-LAST CLASS CHOICE IS NOW MEASURED AT PLAN TIME when
 *    batch >= 64: both classes are ranked on the streaming arena and
 *    X-first must win by >3% to be chosen (X-last is the incumbent; on
 *    wallaby it wins outright, r4 record).  This finally runs the node A/B
 *    that r4 could only request.  The choice is deterministic per plan, so
 *    rule 4 (same plan -> same answer) holds; across processes a near-tie
 *    could flip the class and with it the output bits (the classes round
 *    differently) -- the 3% margin plus the X-last default keeps that to
 *    genuinely-winning cases.  -DL17R_XF_CUT=<B> still force-selects
 *    class B for batch >= B, bypassing the measurement.
 *
 * 4. L3-SCALED TUNER ARENA (adopted from L17_matrixsimd panel_r4, itself
 *    from L36_mixedradix): the streaming re-rank uses
 *    nv = min(batch, clamp(2.5*L3/157KB, 384, 1024)) volumes instead of a
 *    fixed 384, so a 60 MB-L3 machine (wallaby) actually streams while the
 *    node's behaviour (22 MB -> 384) is unchanged bit-for-bit.
 *
 * TECHNIQUE (round panel_r4)
 * --------------------------
 * Rader structure for the prime 17, in the symmetrised cyclic/negacyclic form
 * ADOPTED FROM the L17_winograd entry (strategies/L17_winograd.md round 1):
 * 296 FP instructions (192 FMA + 104 add/sub, 488 flops) per 17-point
 * transform.  The plane-fused layout, split re/im, lane blocking and store
 * paths are this file's round-1 design.
 *
 *   g = 3 generates the units mod 17;  quotienting by {+-1} indexes j,k by the
 *   order-8 quotient group:  folded[m] = |3^m mod 17| in 1..8, sigma[m] the
 *   sign lost in folding.
 *     u_j = x_j + x_(17-j)  drives a length-8 CYCLIC correlation with the real
 *                           kernel c[r] = cos(2 pi 3^r / 17)
 *     v_j = x_j - x_(17-j)  drives a length-8 NEGACYCLIC correlation with the
 *                           kernel s[r] = sin(2 pi 3^r / 17)
 *   The cyclic half splits once more with sign-only reductions,
 *   x^8-1 = (x^4-1)(x^4+1):  P_m = U_m + U_(m+4), Q_m = U_m - U_(m+4) give a
 *   4x4 cyclic apply (x_0-seeded, so DC is free) plus a 4x4 negacyclic apply.
 *   x^8+1 is irreducible over R, so the negacyclic-8 for v stays dense --
 *   L17_winograd's record kills every split of it with counts; do not retry.
 *
 * NEW THIS ROUND (panel_r4), both aimed at the scoring node:
 *
 * 1. MIXED-WIDTH TAIL BLOCKS ("t" variants).  The Gold 5218 has ONE 512-bit
 *    FMA unit but TWO 256-bit FMA ports, so a ymm kernel block retires its
 *    296 FP ops in ~148 cycles where a zmm block needs ~296.  A 17-lane pass
 *    at pure VW=8 costs 3 zmm blocks = 888 cycles; as 2 zmm + 1 ymm tail it
 *    costs 296+296+148 = 740 -- the same floor as pure VW=4 (5 x 148) but
 *    with VW=8's lower instruction/load/store count.  148 cycles x 35 tail
 *    blocks/volume ~ 5.2k cycles ~ 2.2 us at 2.3 GHz, which is almost exactly
 *    the panel_r3 B=1 gap to L17_matrixsimd (18.49 vs 16.39; their zmm holds
 *    4 complex so their lane tax was always 5x4, not 3x8).  On a 2-FMA-unit
 *    machine (wallaby) the mix is FP-neutral, so only the node can rank it:
 *    it is a plan-time tuner candidate.  The tail lanes recompute a few
 *    overlapping transforms bit-identically (lane-independent arithmetic).
 *
 * 2. X-FIRST PASS ORDER, built and then DISABLED BY DEFAULT after measurement
 *    ("xf" variants; see the L17R_XF_CUT comment below for the wallaby
 *    numbers).  The idea is L17_matrixsimd's panel_r3 reorder (-10.8% at
 *    B=256 on the node for THEIR structure): transform the x axis straight
 *    out of the interleaved input (deinterleave folded into the kernel
 *    loads), then finish one kx-plane at a time so the output writes spread
 *    across the volume's compute.  For THIS pass structure it loses on
 *    wallaby in every batched regime, so the default cut keeps X-last at all
 *    batch sizes; -DL17R_XF_CUT=64 re-enables it for a node A/B.  X-first is
 *    NOT bit-identical to X-last (the axis order changes the rounding order),
 *    so per L17_matrixsimd's repeatability discipline the CLASS is a pure
 *    function of batch and the tuner selects freely only within the class,
 *    where all candidates are bit-identical (verified by cmp).
 *
 * Also from panel_r3's node verdict: the stage-2 prefetch A/B now needs a 3%
 * margin to switch pf on (the node picked pf=1 at B=256 on a near-tie and
 * lost 7.4% in the driver's steady state -- monitor's diagnosis, panel_r3
 * VERDICT section 2).
 *
 * LAYOUT / MEMORY  (round 1; 4 L1 crossings per volume against 12 for
 * separate movement passes, measured 35 us vs 64 us)
 * ---------------
 * X-last (the default at every batch size):
 *   for each x plane (17x17 complex = 4.6 KiB, comfortably L1-resident):
 *       in[x][y][z]  --deinterleave + 17x17 transpose-->  T[z][y]
 *       z pass on T   (axis stride TR, lanes = y)      ->  T[kz][y]
 *       17x17 transpose                                ->  U[y][kz]
 *       y pass on U   (axis stride TR, lanes = kz), storing straight into
 *                     A[x][ky][kz]                     ->  A
 *   x pass on A (axis stride PS, 289 contiguous lanes), interleaving store
 *                                                      ->  out[kx][ky][kz]
 * X-first "xfs" (class B; raced against class A at plan time when batch >= 64,
 * and through the joint (pf,pfw) grid since panel_r11):
 *   x pass on in (axis stride NPL complex, 289 contiguous lanes,
 *                 deinterleaving load -- the COLD reads ride under the
 *                 kernel's FMA drains), split store   ->  A[kx][y][z]
 *   for each kx plane:
 *       [pfw: prefetchw out plane kx, one plane ahead of its flush]
 *       [kx>0: dense 4.6 KB sequential flush of staged plane kx-1 -> out]
 *       A[kx][y][z]  --2 x 17x17 transpose-->  T[z][y]
 *       z pass on T                         ->  T[kz][y]
 *       17x17 transpose                     ->  U[y][kz]
 *       y pass on U, interleaving store (stride 17)  ->  staging plane
 *                                           (L1-hot, double-buffered by kx)
 *   final flush of plane 16
 *   (panel_r11: the y pass used to write out[kx] directly through 17-row
 *   strided 128-B partial-line pieces -- the store shape the r10 VERDICT
 *   called the wrong baseline.  Staging + deferred dense flush is the
 *   fused rivals' finished-plane store retrofitted; values and final
 *   addresses are unchanged, so bit class B is preserved.)
 *
 * ASSUMPTIONS
 * -----------
 *  * L == 17 only; fft3d_supports() rejects everything else.
 *  * `in` and `out` are distinct, as the driver guarantees.  Only unaligned
 *    vector moves are emitted, so alignment affects speed, never correctness.
 *  * fft3d_execute() never writes through `in`.  Pad lanes of the scratch
 *    buffers are zeroed once in fft3d_create(); every pass maps zeros to
 *    zeros, and each width owns a disjoint scratch region, so they stay zero
 *    for the life of the plan and repeated executes are bitwise identical.
 *  * In the kernel every input is loaded before the first output is stored,
 *    so the z pass may run it in place.
 *  * -ffp-contract=fast (gcc's default under -std=gnu11) for FMA formation.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1          /* sched_getcpu, pthread_setaffinity_np */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <stdint.h>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#else
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_thread_num(void)  { return 0; }
#endif

#include "../fft3d_api.h"

/* BUILD-FLAG NOTE (panel_r8): L45_pfa (panel_r7) found the scored build
 * lacks tryout.sh's -funroll-loops and lost 10% to it; their fix is a
 * file-level "#pragma GCC optimize(unroll-loops)".  Tested here and
 * REJECTED on measurement (wallaby, same-window 3-way, forced 512t pfw=0,
 * B=1): unroll ON via command line 9.29-9.37 us, unroll OFF 9.36-9.49,
 * pragma 9.55-9.62.  The flag itself is worth <1% for this file (the hot
 * tile loops fully unroll under -O3's complete peeling and the 2-trip
 * kernel loops are deliberately kept rolled with asm-opaque bounds), and
 * the pragma form actively costs ~2% -- the optimize attribute perturbs
 * codegen beyond the named flag.  Do not add it back. */

#ifndef L17R_TEMPLATE_PASS
/* ===========================================================================
 * COMMON SECTION: constants, transposes, plan, tuner.
 * =========================================================================== */

#define LN     17
#define NPL    289            /* 17*17  */
#define NVOL   4913           /* 17^3   */

/* batch size at which the X-first pass order (bit class B) is FORCED.  Since
 * panel_r5 this is a dev/monitor override only: at batch >= 64 the plan now
 * MEASURES both classes on the streaming arena and X-first must beat the
 * X-last incumbent by >3% to be selected (see fft3d_create).  The classes
 * round differently, so a selection is a bit-class choice: deterministic per
 * plan (rule 4 holds); across processes a flip changes output bits, which
 * the margin + X-last default confine to genuinely-winning cases.
 *
 * Wallaby background: the r4 strided-y-store X-first lost there by 11-60%;
 * the panel_r11 staged rebuild ("xfs": y pass -> L1-hot staging plane,
 * deferred dense 4.6 KB flush) closes most of that but still loses ~7%
 * L3-resident (nv=256: 11.56 vs 10.76) and ~18% streaming (B=1024 forced:
 * 17.0 vs 14.4 us/t, best config xfs+pf+pfw).  Wallaby CANNOT rank this
 * shape honestly: its prefetchers price the x pass's 17-row-stream read of
 * `in` at 1.28-1.31x a sequential read, while the node measures the same
 * shape at 0.81-0.83x (L17_matrixsimd's sbw s17/rd, panel_r10 -- the
 * largest cross-machine inversion the panel has measured), and the node is
 * also where dense finished-plane stores won -10.8% (their r3 X-first).
 * Only the node's own plan-time race (automatic, 3% margin, and since r11
 * also raced through the (pf,pfw) grid) can settle it; the stage-1 numbers
 * go out in the description string as `xrace xl/xfs=...`. */
#ifndef L17R_XF_CUT
#define L17R_XF_CUT (1 << 30)
#endif

/* per-width strides, duplicated here for the allocator (the template derives
 * the same numbers from VW) */
#define TR4  20
#define PS4  292
#define TR8  24
#define PS8  296
#define ABUF4 (LN * PS4)
#define TBUF4 (LN * TR4)
#define ABUF8 (LN * PS8)
#define TBUF8 (LN * TR8)

/* ---------------------------------------------------------------- constants
 * c[r] = cos(2 pi 3^r/17), s[r] = sin(2 pi 3^r/17), r = 0..7 indexing the
 * order-8 quotient group.  cp/cm are the cyclic-4 / negacyclic-4 kernels of
 * the sign-only split of the cyclic-8 half:
 *   cp[t] = (c[t] + c[t+4])/2      cm[t] = (c[t] - c[t+4])/2
 * Values verbatim from impl/L17_winograd.c (computed there in long double);
 * SN[t] = s[t] for t = 0..7, negacyclic wrap SN[t+8] = -SN[t]. */
#define CP0 ( 5.12370294433828866e-01)
#define CP1 ( 8.60376828522276954e-02)
#define CP2 (-1.21982091231621334e-01)
#define CP3 (-7.26425886054435255e-01)
#define CM0 ( 4.20101934970526891e-01)
#define CM1 ( 3.59700672924310572e-01)
#define CM2 (-8.60991008452280493e-01)
#define CM3 (-1.23791249675178877e-01)
#define SN0 ( 3.61241666187152921e-01)
#define SN1 ( 8.95163291355062340e-01)
#define SN2 (-1.83749517816570340e-01)
#define SN3 (-5.26432162877355836e-01)
#define SN4 (-9.95734176295034468e-01)
#define SN5 ( 9.61825643172819045e-01)
#define SN6 (-6.73695643646557207e-01)
#define SN7 (-7.98017227280239494e-01)

typedef double    v4d __attribute__((vector_size(32), aligned(8)));
typedef long long v4l __attribute__((vector_size(32)));

/* clang spells the two-operand shuffle differently; gcc is the graded
 * compiler but keeping both costs nothing. */
#if defined(__clang__)
#  define SH4(a,b,m0,m1,m2,m3) __builtin_shufflevector((a),(b),m0,m1,m2,m3)
#else
#  define SH4(a,b,m0,m1,m2,m3) __builtin_shuffle((a),(b),(v4l){m0,m1,m2,m3})
#endif

/* ------------------------------------------------------------- transposes */

/* d[j*ds + i] = s[i*ss + j], i,j in [0,17): 4x4 ymm blocks + scalar edges.
 * Split by SOURCE ROW range (= destination column range): part 0 writes
 * exactly d[.][0..7], part 1 writes d[.][8..16].  The ov exec bodies use the
 * halves to slot the shuffle work into kernel-block shadows; part 0 alone is
 * what a lane block over columns 0..7 needs. */
static inline __attribute__((always_inline))
void transpose17_part(const double *s, long ss, double *d, long ds,
                      const int part)
{
    const int i0lo = part ? 8 : 0, i0hi = part ? 16 : 8;
    for (int i0 = i0lo; i0 < i0hi; i0 += 4)
        for (int j0 = 0; j0 < 16; j0 += 4) {
            const double *p = s + (long)i0*ss + j0;
            v4d r0 = *(const v4d *)(const void *)(p);
            v4d r1 = *(const v4d *)(const void *)(p + ss);
            v4d r2 = *(const v4d *)(const void *)(p + 2*ss);
            v4d r3 = *(const v4d *)(const void *)(p + 3*ss);
            v4d h0 = SH4(r0,r1, 0,4,2,6), h1 = SH4(r0,r1, 1,5,3,7);
            v4d h2 = SH4(r2,r3, 0,4,2,6), h3 = SH4(r2,r3, 1,5,3,7);
            double *q = d + (long)j0*ds + i0;
            *(v4d *)(void *)(q)        = SH4(h0,h2, 0,1,4,5);
            *(v4d *)(void *)(q + ds)   = SH4(h1,h3, 0,1,4,5);
            *(v4d *)(void *)(q + 2*ds) = SH4(h0,h2, 2,3,6,7);
            *(v4d *)(void *)(q + 3*ds) = SH4(h1,h3, 2,3,6,7);
        }
    if (!part) {
        for (int i = 0; i < 8; ++i)  d[16*ds + i] = s[(long)i*ss + 16];
    } else {
        for (int i = 8; i < 16; ++i) d[16*ds + i] = s[(long)i*ss + 16];
        for (int j = 0; j < LN; ++j) d[(long)j*ds + 16] = s[16*ss + j];
    }
}

static inline __attribute__((always_inline))
void transpose17(const double *s, long ss, double *d, long ds)
{
    transpose17_part(s, ss, d, ds, 0);
    transpose17_part(s, ss, d, ds, 1);
}

/* dr[j*ds+i] = Re src[i*17+j], di likewise; src is interleaved complex.
 * Per 4x4 tile: transpose the COMPLEX values first, at 128-bit granularity
 * (one shuffle moves two doubles that stay together), and split re/im after.
 * 8 + 8 = 16 shuffles per 16 complex elements instead of the 8 deinterleaves
 * + 16 real transposes (24) of the obvious order (round 1: 5.31 -> 3.34 us).
 * Split by SOURCE ROW range like transpose17_part; the two parts write
 * disjoint destination regions, so any interleaving with other work on
 * distinct buffers is bit-identical to the back-to-back order. */
static inline __attribute__((always_inline))
void deint_transpose17_part(const double *s, double *dr, double *di, long ds,
                            const int part)
{
    const int i0lo = part ? 8 : 0, i0hi = part ? 16 : 8;
    for (int i0 = i0lo; i0 < i0hi; i0 += 4)
        for (int j0 = 0; j0 < 16; j0 += 4) {
            const double *p = s + 2*((long)i0*LN + j0);
            v4d a0 = *(const v4d *)(const void *)(p);              /* row i0+0, cols j0+0,1 */
            v4d b0 = *(const v4d *)(const void *)(p + 4);          /*           cols j0+2,3 */
            v4d a1 = *(const v4d *)(const void *)(p + 2*LN);
            v4d b1 = *(const v4d *)(const void *)(p + 2*LN + 4);
            v4d a2 = *(const v4d *)(const void *)(p + 4*LN);
            v4d b2 = *(const v4d *)(const void *)(p + 4*LN + 4);
            v4d a3 = *(const v4d *)(const void *)(p + 6*LN);
            v4d b3 = *(const v4d *)(const void *)(p + 6*LN + 4);
            /* out row jj holds the complex values of source rows i0..i0+3 */
            v4d l0 = SH4(a0,a1, 0,1,4,5), h0 = SH4(a2,a3, 0,1,4,5);   /* jj=0 */
            v4d l1 = SH4(a0,a1, 2,3,6,7), h1 = SH4(a2,a3, 2,3,6,7);   /* jj=1 */
            v4d l2 = SH4(b0,b1, 0,1,4,5), h2 = SH4(b2,b3, 0,1,4,5);   /* jj=2 */
            v4d l3 = SH4(b0,b1, 2,3,6,7), h3 = SH4(b2,b3, 2,3,6,7);   /* jj=3 */
            double *qr = dr + (long)j0*ds + i0, *qi = di + (long)j0*ds + i0;
            *(v4d *)(void *)(qr)        = SH4(l0,h0, 0,2,4,6);
            *(v4d *)(void *)(qi)        = SH4(l0,h0, 1,3,5,7);
            *(v4d *)(void *)(qr + ds)   = SH4(l1,h1, 0,2,4,6);
            *(v4d *)(void *)(qi + ds)   = SH4(l1,h1, 1,3,5,7);
            *(v4d *)(void *)(qr + 2*ds) = SH4(l2,h2, 0,2,4,6);
            *(v4d *)(void *)(qi + 2*ds) = SH4(l2,h2, 1,3,5,7);
            *(v4d *)(void *)(qr + 3*ds) = SH4(l3,h3, 0,2,4,6);
            *(v4d *)(void *)(qi + 3*ds) = SH4(l3,h3, 1,3,5,7);
        }
    if (!part) {
        for (int i = 0; i < 8; ++i) {
            dr[16*ds + i] = s[2*((long)i*LN + 16)];
            di[16*ds + i] = s[2*((long)i*LN + 16) + 1];
        }
    } else {
        for (int i = 8; i < 16; ++i) {
            dr[16*ds + i] = s[2*((long)i*LN + 16)];
            di[16*ds + i] = s[2*((long)i*LN + 16) + 1];
        }
        for (int j = 0; j < LN; ++j) {
            dr[(long)j*ds + 16] = s[2*(16*LN + j)];
            di[(long)j*ds + 16] = s[2*(16*LN + j) + 1];
        }
    }
}

static inline __attribute__((always_inline))
void deint_transpose17(const double *s, double *dr, double *di, long ds)
{
    deint_transpose17_part(s, dr, di, ds, 0);
    deint_transpose17_part(s, dr, di, ds, 1);
}

/* ---------------- 512-bit transposes (panel_r8, w8 pipeline only) ----------
 * The panel_r7 VERDICT's one-directional L=17 evidence: the node rejected
 * three scheduling attacks on the non-FP residue (ov r5, dz r7, matrixsimd's
 * deferred-Z at the small cells) and rewarded exactly one mechanism -- uop
 * DELETION (L17_winograd's g8, -8..-10% in all four cells).  My residue is
 * dominated by these serialized transpose loops, which were still 4x4 ymm
 * tiles from round 1 even inside the 512-bit pipeline.  An 8x8 zmm block is
 * 8 loads + 24 shuffles + 8 stores = 40 uops per 64 elements against the
 * 4x4 tiling's 64 (and half the load/store slots on a 2-load-port CLX);
 * the interleaved-complex tile drops from 128 to 80 uops per 64 elements.
 * Pure data movement: same values to the same places, so every class-A
 * candidate stays bit-identical by construction.  On a non-AVX512 host gcc
 * emulates the 64-byte vectors (correct, slow) and the w8 candidates
 * self-eliminate in the tuner, as before. */

typedef double    v8d __attribute__((vector_size(64), aligned(8)));
typedef long long v8l __attribute__((vector_size(64)));

#if defined(__clang__)
#  define SH8(a,b,m0,m1,m2,m3,m4,m5,m6,m7) \
        __builtin_shufflevector((a),(b),m0,m1,m2,m3,m4,m5,m6,m7)
#else
#  define SH8(a,b,m0,m1,m2,m3,m4,m5,m6,m7) \
        __builtin_shuffle((a),(b),(v8l){m0,m1,m2,m3,m4,m5,m6,m7})
#endif
#define VL8(p)   (*(const v8d *)(const void *)(p))
#define VS8(p,x) (*(v8d *)(void *)(p) = (x))

/* d[j*ds + i] = s[i*ss + j], i,j in [0,8): the classic 3-stage network
 * (vunpck / 128-bit-granule permute / 256-bit-granule permute), 24 two-source
 * shuffles, all single-uop on SKX/CLX/SPR. */
static inline __attribute__((always_inline))
void tz8x8(const double *s, long ss, double *d, long ds)
{
    v8d r0 = VL8(s),        r1 = VL8(s + ss),   r2 = VL8(s + 2*ss),
        r3 = VL8(s + 3*ss), r4 = VL8(s + 4*ss), r5 = VL8(s + 5*ss),
        r6 = VL8(s + 6*ss), r7 = VL8(s + 7*ss);
    v8d t0 = SH8(r0,r1, 0,8,2,10,4,12,6,14);
    v8d t1 = SH8(r0,r1, 1,9,3,11,5,13,7,15);
    v8d t2 = SH8(r2,r3, 0,8,2,10,4,12,6,14);
    v8d t3 = SH8(r2,r3, 1,9,3,11,5,13,7,15);
    v8d t4 = SH8(r4,r5, 0,8,2,10,4,12,6,14);
    v8d t5 = SH8(r4,r5, 1,9,3,11,5,13,7,15);
    v8d t6 = SH8(r6,r7, 0,8,2,10,4,12,6,14);
    v8d t7 = SH8(r6,r7, 1,9,3,11,5,13,7,15);
    v8d u0 = SH8(t0,t2, 0,1,8,9,4,5,12,13);
    v8d u2 = SH8(t0,t2, 2,3,10,11,6,7,14,15);
    v8d u1 = SH8(t1,t3, 0,1,8,9,4,5,12,13);
    v8d u3 = SH8(t1,t3, 2,3,10,11,6,7,14,15);
    v8d u4 = SH8(t4,t6, 0,1,8,9,4,5,12,13);
    v8d u6 = SH8(t4,t6, 2,3,10,11,6,7,14,15);
    v8d u5 = SH8(t5,t7, 0,1,8,9,4,5,12,13);
    v8d u7 = SH8(t5,t7, 2,3,10,11,6,7,14,15);
    VS8(d,        SH8(u0,u4, 0,1,2,3,8,9,10,11));
    VS8(d + ds,   SH8(u1,u5, 0,1,2,3,8,9,10,11));
    VS8(d + 2*ds, SH8(u2,u6, 0,1,2,3,8,9,10,11));
    VS8(d + 3*ds, SH8(u3,u7, 0,1,2,3,8,9,10,11));
    VS8(d + 4*ds, SH8(u0,u4, 4,5,6,7,12,13,14,15));
    VS8(d + 5*ds, SH8(u1,u5, 4,5,6,7,12,13,14,15));
    VS8(d + 6*ds, SH8(u2,u6, 4,5,6,7,12,13,14,15));
    VS8(d + 7*ds, SH8(u3,u7, 4,5,6,7,12,13,14,15));
}

/* Same contract and part split as transpose17_part (part 0 writes d[.][0..7],
 * part 1 writes d[.][8..16]); only the block size changed. */
static inline __attribute__((always_inline))
void transpose17z_part(const double *s, long ss, double *d, long ds,
                       const int part)
{
    const long i0 = part ? 8 : 0;
    tz8x8(s + i0*ss,     ss, d + i0,        ds);
    tz8x8(s + i0*ss + 8, ss, d + 8*ds + i0, ds);
    if (!part) {
        for (int i = 0; i < 8; ++i)  d[16*ds + i] = s[(long)i*ss + 16];
    } else {
        for (int i = 8; i < 16; ++i) d[16*ds + i] = s[(long)i*ss + 16];
        for (int j = 0; j < LN; ++j) d[(long)j*ds + 16] = s[16*ss + j];
    }
}

static inline __attribute__((always_inline))
void transpose17z(const double *s, long ss, double *d, long ds)
{
    transpose17z_part(s, ss, d, ds, 0);
    transpose17z_part(s, ss, d, ds, 1);
}

/* 8x8-complex deinterleaving transpose tile: s points at interleaved complex
 * (row 0, col 0) of the tile, row stride LN complex; writes the transposed
 * re/im 8x8 blocks at dr/di (row stride ds).  Same scheme as the ymm tile:
 * transpose the COMPLEX values first at 128-bit granularity (two quadrant
 * 4x4 granule transposes per register half), split re/im after.
 * 16 loads + 48 shuffles + 16 stores per 64 complex, against 128 uops for
 * the 4x4 tiling. */
static inline __attribute__((always_inline))
void dz8x8(const double *s, double *dr, double *di, long ds)
{
    v8d a0 = VL8(s),           b0 = VL8(s + 8);
    v8d a1 = VL8(s + 2*LN),    b1 = VL8(s + 2*LN + 8);
    v8d a2 = VL8(s + 4*LN),    b2 = VL8(s + 4*LN + 8);
    v8d a3 = VL8(s + 6*LN),    b3 = VL8(s + 6*LN + 8);
    v8d a4 = VL8(s + 8*LN),    b4 = VL8(s + 8*LN + 8);
    v8d a5 = VL8(s + 10*LN),   b5 = VL8(s + 10*LN + 8);
    v8d a6 = VL8(s + 12*LN),   b6 = VL8(s + 12*LN + 8);
    v8d a7 = VL8(s + 14*LN),   b7 = VL8(s + 14*LN + 8);
    /* 4x4 128-bit-granule transpose of each quadrant: l_j = complex of source
     * rows 0..3 at tile column j, h_j = rows 4..7 at column j. */
#define QT4(m0,m1,m2,m3, n0,n1,n2,n3) do {                                     \
        v8d p0 = SH8(m0,m1, 0,1,8,9,4,5,12,13);                                \
        v8d p1 = SH8(m0,m1, 2,3,10,11,6,7,14,15);                              \
        v8d p2 = SH8(m2,m3, 0,1,8,9,4,5,12,13);                                \
        v8d p3 = SH8(m2,m3, 2,3,10,11,6,7,14,15);                              \
        n0 = SH8(p0,p2, 0,1,2,3,8,9,10,11);                                    \
        n2 = SH8(p0,p2, 4,5,6,7,12,13,14,15);                                  \
        n1 = SH8(p1,p3, 0,1,2,3,8,9,10,11);                                    \
        n3 = SH8(p1,p3, 4,5,6,7,12,13,14,15);                                  \
    } while (0)
    v8d l0,l1,l2,l3,l4,l5,l6,l7, h0,h1,h2,h3,h4,h5,h6,h7;
    QT4(a0,a1,a2,a3, l0,l1,l2,l3);
    QT4(a4,a5,a6,a7, h0,h1,h2,h3);
    QT4(b0,b1,b2,b3, l4,l5,l6,l7);
    QT4(b4,b5,b6,b7, h4,h5,h6,h7);
#undef QT4
#define SPL(j, lj, hj) do {                                                    \
        VS8(dr + (j)*ds, SH8(lj,hj, 0,2,4,6,8,10,12,14));                      \
        VS8(di + (j)*ds, SH8(lj,hj, 1,3,5,7,9,11,13,15));                      \
    } while (0)
    SPL(0, l0, h0);  SPL(1, l1, h1);  SPL(2, l2, h2);  SPL(3, l3, h3);
    SPL(4, l4, h4);  SPL(5, l5, h5);  SPL(6, l6, h6);  SPL(7, l7, h7);
#undef SPL
}

/* Same contract and part split as deint_transpose17_part. */
static inline __attribute__((always_inline))
void deint_transpose17z_part(const double *s, double *dr, double *di, long ds,
                             const int part)
{
    const long i0 = part ? 8 : 0;
    dz8x8(s + 2*(i0*LN),     dr + i0,        di + i0,        ds);
    dz8x8(s + 2*(i0*LN + 8), dr + 8*ds + i0, di + 8*ds + i0, ds);
    if (!part) {
        for (int i = 0; i < 8; ++i) {
            dr[16*ds + i] = s[2*((long)i*LN + 16)];
            di[16*ds + i] = s[2*((long)i*LN + 16) + 1];
        }
    } else {
        for (int i = 8; i < 16; ++i) {
            dr[16*ds + i] = s[2*((long)i*LN + 16)];
            di[16*ds + i] = s[2*((long)i*LN + 16) + 1];
        }
        for (int j = 0; j < LN; ++j) {
            dr[(long)j*ds + 16] = s[2*(16*LN + j)];
            di[(long)j*ds + 16] = s[2*(16*LN + j) + 1];
        }
    }
}

static inline __attribute__((always_inline))
void deint_transpose17z(const double *s, double *dr, double *di, long ds)
{
    deint_transpose17z_part(s, dr, di, ds, 0);
    deint_transpose17z_part(s, dr, di, ds, 1);
}

/* ------------------------------------------------------------------- plan */

#define L17R_MAXT 32   /* hard cap: never take more threads than the harness's 32 */

struct fft3d_plan {
    int batch;
    int pf;                        /* cross-volume input prefetch enabled */
    int pfw;                       /* paced prefetchw on the x pass's out */
    /* ---- multicore (round mt_r1) ---- */
    int mode;                      /* 0 = single-thread (p->exec on the whole
                                    * batch), 1 = volume-parallel (each thread
                                    * runs its shadow plan's exec on its own
                                    * contiguous chunk of volumes), 2 = plane-
                                    * parallel within volumes (small batches:
                                    * (b,x) plane tasks, barrier, (b,blk) x
                                    * tasks) */
    int nt;                        /* team size for modes 1 and 2 */
    int dsp;                       /* mode-1 dispatch: 0 = spin pool, 1 = OMP
                                    * parallel region (mt_r3: the only entry
                                    * shape measured at ~193 GB/s on the node
                                    * at a streaming cell is L17_winograd's
                                    * OMP static split; raced at plan time) */
    int xpf;                       /* mode 2: prefetch the NEXT x block's 34
                                    * cross-core A lines under this block's
                                    * compute (mt_r3, from L17_matrixsimd) */
    int ntmax;                     /* omp_get_max_threads() clamped to 32 */
    int boff;                      /* shadow plans: this chunk's first volume */
    double tsec;                   /* shadow plans: min chunk wall time
                                    * (telemetry: equal-cut spread) */
    struct fft3d_plan *sh[L17R_MAXT]; /* per-thread shadow plans; scratch
                                    * first-touched by the owning thread in
                                    * fft3d_create (NUMA) */
    double *vpar, *vpai;           /* mode 2: per-VOLUME A slots (nvp*ABUF8
                                    * doubles each) shared across the team */
    void *vpmem;
    int nvp;
    struct l17r_pool *pool;        /* spin-wait worker pool (see mt layer) */
    double *mem;
    /* disjoint scratch per width, so each width's pad lanes stay zero */
    double *ar_w4, *ai_w4, *tr_w4, *ti_w4, *ur_w4, *ui_w4;
    double *ar_w8, *ai_w8, *tr_w8, *ti_w8, *ur_w8, *ui_w8;
    double *ar2_w8, *ai2_w8;       /* ping-pong A pair for the sp pipeline */
    double *tr2_w8, *ti2_w8;       /* second T pair for the dz plane schedule */
    double *vo_w8, *vo2_w8;        /* staging volumes for the st/stp flush
                                    * (ping-pong pair; stp flushes b-1 while
                                    * b computes) */
    void (*exec)(struct fft3d_plan *, const double _Complex *,
                 double _Complex *);
    double _Complex *tin, *tout;   /* transient tuner buffers */
    size_t tn;
};

/* ---- instantiate the pipeline at both vector widths by self-#include ----
 * (mechanism adopted from L17_matrixsimd; a quoted #include is searched in
 * the includer's own directory first, so the first form works from any
 * working directory). */
#if defined(__has_include)
#  if __has_include("L17_rader.c")
#    define L17R_SELF "L17_rader.c"
#  elif __has_include("impl/L17_rader.c")
#    define L17R_SELF "impl/L17_rader.c"
#  endif
#else
#  define L17R_SELF "L17_rader.c"
#endif
#ifndef L17R_SELF
#  error "L17_rader.c must be able to #include itself"
#endif

#define L17R_TEMPLATE_PASS 1
#define VW 4
#define SFX(x) x##_w4
#include L17R_SELF
#undef SFX
#undef VW
#undef L17R_TEMPLATE_PASS

#define L17R_TEMPLATE_PASS 2
#define VW 8
#define SFX(x) x##_w8
#include L17R_SELF
#undef SFX
#undef VW
#undef L17R_TEMPLATE_PASS

/* ------------------------------------------------------------- interface */

const char *fft3d_name(void) { return "L17_rader"; }

static char g_desc[512] =
    "Rader-17 in cyclic/negacyclic form (kernel from L17_winograd), split "
    "re/im, plane-fused, plan-time width tuning";

const char *fft3d_description(void) { return g_desc; }

int fft3d_supports(int L) { return L == LN; }

/* -------------------------------------------------------------- the tuner */

typedef void (*l17r_fn)(fft3d_plan *, const double _Complex *,
                        double _Complex *);

/* Candidate classes.  Class A (X-last) serves batch < L17R_XF_CUT, class B
 * (X-first) serves batch >= L17R_XF_CUT; the class is fixed by the batch size
 * (bit-equivalence discipline from L17_matrixsimd panel_r3), the tuner ranks
 * only within it.  The pinned kernels enter only where the EVEX file exists;
 * elsewhere the plain widths compete (the emulated 512-bit build on an AVX2
 * host measures slow and eliminates itself).  "t" = mixed ymm tail. */
#if defined(__AVX512VL__)
#  define L17R_NCA 21
#  define L17R_NCB 4
#else
#  define L17R_NCA 3
#  define L17R_NCB 3
#endif
static const l17r_fn l17r_cand_a[L17R_NCA] = {
    exec_np_w4, exec_np_w8, exec_npm_w8,
#if defined(__AVX512VL__)
    exec_pin_w8, exec_pinm_w8, exec_ovm_w8, exec_ovmpin_w8, exec_spm_w8,
    exec_dzm_w8, exec_dzmpin_w8, exec_dzspm_w8,
    exec_npmdy_w8, exec_spmdy_w8,          /* panel_r9: ymm-deint twins */
    exec_stm_w8, exec_stmdy_w8,            /* panel_r10: staged dense out */
    exec_stpm_w8, exec_stpmdy_w8,          /* panel_r10: + pipelined flush */
    exec_stntm_w8, exec_stntmdy_w8,        /* mt_r1: NT flush twins */
    exec_stpntm_w8, exec_stpntmdy_w8,
#endif
};
static const char *const l17r_tag_a[L17R_NCA] = {
    "xl 256", "xl 512", "xl 512t",
#if defined(__AVX512VL__)
    "xl 512 pin", "xl 512t pin", "xl 512t ov", "xl 512t ov pin", "xl 512t sp",
    "xl 512t dz", "xl 512t dz pin", "xl 512t dzsp",
    "xl 512t dy", "xl 512t sp dy",
    "xl 512t st", "xl 512t st dy",
    "xl 512t stp", "xl 512t stp dy",
    "xl 512t stnt", "xl 512t stnt dy",
    "xl 512t stpnt", "xl 512t stpnt dy",
#endif
};
static const l17r_fn l17r_cand_b[L17R_NCB] = {
    exec_xf_w4, exec_xf_w8, exec_xfm_w8,
#if defined(__AVX512VL__)
    exec_xfpinm_w8,
#endif
};
static const char *const l17r_tag_b[L17R_NCB] = {
    "xfs 256", "xfs 512", "xfs 512t",
#if defined(__AVX512VL__)
    "xfs 512t pin",
#endif
};

static void l17r_tune_free(fft3d_plan *p)
{
    free(p->tin);
    free(p->tout);
    p->tin = NULL;
    p->tout = NULL;
    p->tn = 0;
}

/* Deterministic pseudo-random scratch input: the tuner needs realistic
 * magnitudes, not the real data. */
static int l17r_tune_alloc(fft3d_plan *p, int nv)
{
    size_t n = (size_t)nv * NVOL;
    if (p->tn >= n) return 1;
    l17r_tune_free(p);
    if (posix_memalign((void **)&p->tin, 64, n * sizeof *p->tin) != 0) {
        p->tin = NULL;
        return 0;
    }
    if (posix_memalign((void **)&p->tout, 64, n * sizeof *p->tout) != 0) {
        free(p->tin);
        p->tin = NULL;
        p->tout = NULL;
        return 0;
    }
    p->tn = n;
    unsigned sr = 987654321u;
    for (size_t i = 0; i < n; ++i) {
        sr = sr * 1103515245u + 12345u;
        double a = (double)(sr >> 8) / 8388608.0 - 1.0;
        sr = sr * 1103515245u + 12345u;
        double b = (double)(sr >> 8) / 8388608.0 - 1.0;
        p->tin[i] = a + b * (double _Complex)I;
    }
    /* mt_r2: tout IS memset serially, by the main thread.  mt_r1 left it
     * untouched on the theory that the driver's `out` is first-touched by
     * the first warmup execute; that theory was WRONG -- driver.c memsets
     * `out` on the main thread BEFORE fft3d_create (driver.c:109), so every
     * caller page (in via fread, out via memset) is socket-0 on the
     * two-socket node.  The mt_r1 arena therefore raced candidates under a
     * distributed-out placement the scored run never sees, and its node
     * pick at B=4096 measured 0.793 us/t in the arena against 2.904 us/t
     * scored.  Matching the driver's placement is L23_matrixsimd mt_r1's
     * lesson ("the tuner arena is deliberately filled serially so
     * create()-time racing sees the same placement the scored run does"),
     * also adopted by L8_fusedaxes.  tin was already filled serially. */
    memset(p->tout, 0, n * sizeof *p->tout);
    return 1;
}

static int l17r_verbose(void)
{
    const char *e = getenv("L17R_VERBOSE");
    return e && *e && *e != '0';
}

static double l17r_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Streaming-arena size in volumes: min(batch, clamp(2.5*L3/157KB, 384, 1024)).
 * Adopted from L17_matrixsimd panel_r4 (itself from L36_mixedradix): a fixed
 * 384-volume arena is exactly wallaby's 60 MB L3 and tunes the L3-resident
 * regime there; the node (22 MB) still gets 384, bit-for-bit as before. */
static int l17r_arena_nv(void)
{
#ifdef _SC_LEVEL3_CACHE_SIZE
    long l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#else
    long l3 = 0;
#endif
    if (l3 <= 0) return 384;
    double nv = 2.5 * (double)l3 / (157.0 * 1024.0);
    if (nv < 384.0) nv = 384.0;
    if (nv > 1024.0) nv = 1024.0;
    return (int)nv;
}

/* How many NUMA nodes the TEAM's cores span (not the machine's: wallaby has
 * two sockets but the harness's 32 close-bound threads sit on one, so its
 * team spans 1 node; the benchmark node's 2x16 spans 2).
 * /sys/devices/system/node/nodeN/cpuC exists iff cpu C belongs to node N. */
static int __attribute__((unused)) l17r_team_nodes(const int *cpus, int nt)
{
    unsigned mask = 0;
    for (int nd = 0; nd < 8; ++nd)
        for (int t = 0; t < nt; ++t) {
            char path[96];
            if (cpus[t] < 0) continue;
            snprintf(path, sizeof path,
                     "/sys/devices/system/node/node%d/cpu%d", nd, cpus[t]);
            if (access(path, F_OK) == 0) { mask |= 1u << nd; break; }
        }
    int n = __builtin_popcount(mask);
    return n > 0 ? n : 1;
}

/* Sustained core clock at each vector width, measured at plan time
 * (unscored) and reported in fft3d_description() as clk256/clk512 -- the
 * panel_r4 VERDICT's explicit L=17 ask: every L=17 kernel is zmm and nobody
 * has measured the AVX-512 licence clock on the node.  Method ADOPTED FROM
 * L6_unrolled (panel_r4): a serially dependent FMA chain, latency 4 cycles
 * at BOTH widths on SKX/CLX/ICL/SPR, so freq = iters*4/time; best of 5
 * trials after the tournament has warmed the core.  (Haswell dev host: FMA
 * latency 5 and the 512-bit type is emulated -- both numbers are dev-only
 * noise there; wallaby and the node are what matters.) */
typedef double l17r_v8d __attribute__((vector_size(64), aligned(8)));

static double l17r_probe_ghz(int wide)
{
    double best = 0.0;
    if (wide) {
        l17r_v8d x = {1,1,1,1,1,1,1,1};
        const l17r_v8d a = ((l17r_v8d){0} + (1.0 + 1e-15));
        const l17r_v8d b = ((l17r_v8d){0} + 1e-300);
        double warm_until = l17r_now() + 4e-3;
        do {
            for (int i = 0; i < 8192; ++i) x = x * a + b;
        } while (l17r_now() < warm_until);
        for (int trial = 0; trial < 5; ++trial) {
            double t0 = l17r_now();
            for (int i = 0; i < 262144; ++i) x = x * a + b;
            double dt = l17r_now() - t0;
            double ghz = 262144.0 * 4.0 / dt * 1e-9;
            if (ghz > best) best = ghz;
        }
        if (!(x[0] > 0.0)) best = 0.0;     /* keep the chain observable */
    } else {
        v4d x = {1,1,1,1};
        const v4d a = ((v4d){0} + (1.0 + 1e-15));
        const v4d b = ((v4d){0} + 1e-300);
        double warm_until = l17r_now() + 4e-3;
        do {
            for (int i = 0; i < 8192; ++i) x = x * a + b;
        } while (l17r_now() < warm_until);
        for (int trial = 0; trial < 5; ++trial) {
            double t0 = l17r_now();
            for (int i = 0; i < 262144; ++i) x = x * a + b;
            double dt = l17r_now() - t0;
            double ghz = 262144.0 * 4.0 / dt * 1e-9;
            if (ghz > best) best = ghz;
        }
        if (!(x[0] > 0.0)) best = 0.0;
    }
    if (best > 9.9) best = 0.0;
    return best;
}

/* Spin a real zmm exec for ~150 ms before ranking, so the turbo/licence
 * clock has settled by the first candidate.  Without this the first two or
 * three candidates in a rank are measured on a still-ramping clock (seen on
 * wallaby at nv=256: early candidates 21-30 us/t, late ones 11.9 for
 * bit-identical work) and the pick is an artifact of table order. */
static void l17r_settle(fft3d_plan *p, int nv)
{
    int sb = p->batch, sp = p->pf, sw = p->pfw;
    p->batch = nv;
    p->pf = 0;
    p->pfw = 0;
    double t0 = l17r_now();
    do
        l17r_cand_a[L17R_NCA > 3 ? 1 : 0](p, p->tin, p->tout);
    while (l17r_now() - t0 < 0.15);
    p->batch = sb;
    p->pf = sp;
    p->pfw = sw;
}

/* Time each candidate in a block of >= 64 consecutive volume transforms,
 * never interleaved (L17_matrixsimd round-1 item 12: interleaving ISA widths
 * mis-ranked candidates by 35% on the node).  Returns the fastest index. */
static int l17r_rank(fft3d_plan *p, int nv, double *best_us,
                     const l17r_fn *cand, int ncand, int reps)
{
    int inner = (64 + nv - 1) / nv;
    if (inner < 1) inner = 1;
    int sb = p->batch, sp = p->pf, sw = p->pfw;
    p->batch = nv;
    p->pf = 0;
    p->pfw = 0;
    int bestv = 0;
    /* Two full fixed-order sweeps, per-candidate min across both (panel_r7):
     * even with the settle spin and per-candidate warmups, a machine whose
     * clock ramps over the whole tournament (wallaby contended: one verbose
     * table spanned 27 -> 10 us/t monotonically for near-identical work)
     * biases early table slots.  A second sweep runs on the settled clock,
     * so the min per candidate halves the order bias at ~2x plan time.
     * Within a sweep each candidate is still timed in its own contiguous
     * block (never interleaved -- L17_matrixsimd round-1 item 12). */
    for (int v = 0; v < ncand; ++v) best_us[v] = 1e30;
    for (int pass = 0; pass < 2; ++pass) {
        for (int v = 0; v < ncand; ++v) {
            double best = 1e30;
            /* Warmup: page in, and let the turbo/licence level settle to THIS
             * candidate's own.  Time-based (>= 1.5 ms > Intel's ~670 us
             * licence-up dwell): at the B=1 stage one exec is ~200-350 us, so
             * the old 2-exec warmup left a ymm candidate measured inside the
             * AVX-512 licence of the previous zmm candidate (clk256 = 3.89 vs
             * clk512 = 2.89 on the node, r5 probes). */
            double w0 = l17r_now();
            cand[v](p, p->tin, p->tout);
            cand[v](p, p->tin, p->tout);
            while (l17r_now() - w0 < 1.5e-3)
                cand[v](p, p->tin, p->tout);
            for (int r = 0; r < reps; ++r) {
                double t0 = l17r_now();
                for (int q = 0; q < inner; ++q) cand[v](p, p->tin, p->tout);
                double dt = l17r_now() - t0;
                if (dt < best) best = dt;
            }
            double us = best * 1e6 / ((double)nv * inner);
            if (us < best_us[v]) best_us[v] = us;
        }
    }
    for (int v = 0; v < ncand; ++v)
        if (best_us[v] < best_us[bestv]) bestv = v;
    p->batch = sb;
    p->pf = sp;
    p->pfw = sw;
    return bestv;
}

/* Time one exec-shaped function on the tuner arena at (pf, pfw) = (0, 0):
 * licence/turbo warmup >= 1.5 ms, then min over `reps` whole-arena passes.
 * Used by the panel_r9 streaming decomposition probe (and nothing scored). */
static double __attribute__((unused))
l17r_time_fn(fft3d_plan *p, int nv, l17r_fn f, int reps)
{
    int sb = p->batch, spf = p->pf, sw = p->pfw;
    p->batch = nv;
    p->pf = 0;
    p->pfw = 0;
    double w0 = l17r_now();
    f(p, p->tin, p->tout);
    f(p, p->tin, p->tout);
    while (l17r_now() - w0 < 1.5e-3)
        f(p, p->tin, p->tout);
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        double t0 = l17r_now();
        f(p, p->tin, p->tout);
        double dt = l17r_now() - t0;
        if (dt < best) best = dt;
    }
    p->batch = sb;
    p->pf = spf;
    p->pfw = sw;
    return best * 1e6 / nv;
}

/* ======================================================================
 * MULTICORE LAYER (round mt_r1)
 * ======================================================================
 * Two parallel modes on top of the phase-1 kernels, both bit-identical to
 * the single-thread class-A output (identical per-volume kernel calls and
 * operand values; only WHO computes each disjoint piece changes):
 *
 *   mode 1 (volume-parallel): the batch is cut into nt contiguous chunks;
 *     thread t runs the tuned single-thread exec on its chunk, with its own
 *     NUMA-local scratch (a full shadow plan first-touched by thread t in
 *     fft3d_create).  Contiguous chunks keep each thread's slice of `out`
 *     first-touched by its own warmup writes (the driver never touches out
 *     before warmup), so at batch the out pages are socket-local; `in` is
 *     fread by the driver's main thread and is socket-0 for everyone --
 *     nothing an implementation can do about that.
 *
 *   mode 2 (plane-parallel, small batches): the (b,x) plane pipeline
 *     (deint -> z -> transpose -> y -> A[b][x]) is a task; one barrier; the
 *     (b,blk) x-pass blocks (A[b] -> out[b]) are tasks.  A is per-volume
 *     (vpar/vpai slots) and plane regions are cache-line-aligned
 *     (PS8*8 = 2368 B = 37 lines), so plane tasks never share a store line.
 *     The x-pass blocks are handed out in contiguous runs, so only the
 *     nt-1 chunk-boundary lines of each of the 17 out rows are ever
 *     written by two threads.
 *
 * Everything is decided at plan time by measuring the REAL dispatch path
 * on the tuner arena, so rule 4 (same plan -> same answer) holds and the
 * B=1 "does it parallelise at all" question is answered by measurement,
 * not hope. */

static int l17r_alloc_scratch(fft3d_plan *p)
{
    size_t nd = (size_t)2*ABUF4 + (size_t)4*TBUF4
              + (size_t)4*ABUF8 + (size_t)6*TBUF8 + (size_t)4*NVOL;
    void *raw = NULL;
    if (posix_memalign(&raw, 64, nd * sizeof(double)) != 0 || !raw)
        return 0;
    p->mem = raw;
    memset(p->mem, 0, nd * sizeof(double));      /* pad lanes := 0, forever */
    double *q = p->mem;
    p->ar_w4 = q; q += ABUF4;
    p->ai_w4 = q; q += ABUF4;
    p->tr_w4 = q; q += TBUF4;
    p->ti_w4 = q; q += TBUF4;
    p->ur_w4 = q; q += TBUF4;
    p->ui_w4 = q; q += TBUF4;
    p->ar_w8 = q; q += ABUF8;
    p->ai_w8 = q; q += ABUF8;
    p->tr_w8 = q; q += TBUF8;
    p->ti_w8 = q; q += TBUF8;
    p->ur_w8 = q; q += TBUF8;
    p->ui_w8 = q; q += TBUF8;
    p->ar2_w8 = q; q += ABUF8;
    p->ai2_w8 = q; q += ABUF8;
    p->tr2_w8 = q; q += TBUF8;
    p->ti2_w8 = q; q += TBUF8;
    p->vo_w8 = q; q += (size_t)2*NVOL;  /* fully rewritten before every read;
                                         * no pad invariant on either */
    p->vo2_w8 = q;
    return 1;
}

/* contiguous partition of nv items over nt threads: item range of thread t
 * is [cut(t), cut(t+1)) */
static inline int l17r_cut(int nv, int nt, int t)
{
    return (int)(((long)nv * t) / nt);
}

/* One (b,x) plane task of mode 2: exec_body's mixed-width X-last plane
 * pipeline verbatim (deint -> z -> T->U -> y), reading the caller's plane
 * in[b][x] and writing A slot b's plane x.  T/U live in the calling
 * thread's shadow scratch; A is the shared per-volume slot.  noinline keeps
 * mode 2 from adding inlined kernel copies to the I-footprint (r2's 38 KB
 * kill line). */
static __attribute__((noinline, noclone)) void l17r_plane_w8(
        const double *src, int x, fft3d_plan *q, double *arr, double *aii)
{
    double *const tr = q->tr_w8, *const ti = q->ti_w8;
    double *const ur = q->ur_w8, *const ui = q->ui_w8;

    deint_transpose17z(src + 2*(long)x*NPL, tr, ti, TR8);

    {   /* z pass, in place on T (mixed: 2 zmm blocks + ymm tail) */
        long mlim = 8;
        __asm__("" : "+r"(mlim));
        for (long o = 0; o <= mlim; o += 8)
            wino17_w8(tr + o, ti + o, TR8, 0, 0, tr + o, ti + o,
                      0, 0, TR8, 0, 0, 0);
        wino17_w4(tr + 16, ti + 16, TR8, 0, 0, tr + 16, ti + 16,
                  0, 0, TR8, 0, 0, 0);
    }

    transpose17z(tr, TR8, ur, TR8);
    transpose17z(ti, TR8, ui, TR8);

    {   /* y pass into the volume's shared A slot */
        double *dr = arr + (long)x*PS8, *di = aii + (long)x*PS8;
        long mlim = 8;
        __asm__("" : "+r"(mlim));
        for (long o = 0; o <= mlim; o += 8)
            wino17_w8(ur + o, ui + o, TR8, 0, 0, dr + o, di + o,
                      0, 0, LN, 0, 0, 0);
        wino17_w4(ur + 13, ui + 13, TR8, 0, 0, dr + 13, di + 13,
                  0, 0, LN, 0, 0, 0);
    }
}

/* ---- spin-wait worker pool -------------------------------------------
 * Measured on wallaby (SPR, this round): one libgomp parallel region +
 * barrier costs 2.7 us at nt=2 rising to 8.3 us at nt=32 -- the entire
 * B=1 budget.  The pool replaces OpenMP at execute time: nt-1 workers are
 * created ONCE in fft3d_create (thread creation is setup), pinned to the
 * same cores the harness's OMP_PROC_BIND=close binding chose (recorded
 * from inside the create-time OpenMP region), and dispatched through one
 * release-store + spin-wait handshake.  Workers busy-poll with pause for
 * ~20 ms after their last job, then park on a condvar (so an idle plan
 * burns nothing); the first execute after an idle spell pays one futex
 * wake, which the driver's warmup absorbs.  Total threads never exceed
 * the given 32: the pool is main + 31 workers, and the create-time OpenMP
 * pool sleeps whenever the worker pool runs (execute never enters an
 * OpenMP region). */

typedef void (*l17r_workfn)(fft3d_plan *, int, int,
                            const double _Complex *, double _Complex *);

/* One flag per worker on its own cache line.  mt_r2: the mt_r1 handshake
 * (one shared `go` line read by all 31 workers + one shared `done` line
 * RFO'd by all 31, team member or not) put 15+ remote-socket line bounces
 * on EVERY dispatch on the two-socket node -- the standing suspect for the
 * node's B=1 9.13 us / 38.3% run spread against wallaby's stable 5.7.
 * Replaced by per-worker release/done flags, ADOPTED FROM L17_matrixsimd
 * mt_r1 ("only the ACTIVE team's flags are touched -- idle workers never
 * wake and cost nothing on the release path").  This also closes the r1
 * team-only-ack staleness race BY CONSTRUCTION rather than by all-ack:
 * a worker only reads the job descriptor when ITS OWN flag was bumped
 * (bump is a release store made after the descriptor writes), and the
 * caller collects that worker's done flag before mutating the descriptor
 * again -- a never-released worker never reads the descriptor at all. */
typedef struct l17r_flag { _Atomic int v; char pad[64 - sizeof(_Atomic int)]; }
    l17r_flag;

typedef struct l17r_pool {
    l17r_flag rel[L17R_MAXT];      /* per-worker job release generation  */
    l17r_flag fin[L17R_MAXT];      /* per-worker job done generation     */
    l17r_flag arr[L17R_MAXT];      /* per-RANK barrier arrival (flat)    */
    l17r_flag pkd[L17R_MAXT];      /* per-worker "I am parked" flag      */
    _Atomic int brel __attribute__((aligned(64)));   /* barrier release  */
    _Atomic int hold __attribute__((aligned(64)));   /* 1: park everyone */
    _Atomic int quit;
    int gen;                                          /* main thread only */
    struct {                                          /* current job */
        fft3d_plan *p;
        const double _Complex *in;
        double _Complex *out;
        l17r_workfn fn;
        int nt;
        int gen;      /* this job's generation: the barrier's epoch */
    } job __attribute__((aligned(64)));
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int nw;
    pthread_t th[L17R_MAXT];
    struct l17r_warg { struct l17r_pool *pl; int slot; int cpu; }
        warg[L17R_MAXT];
} l17r_pool;

#define L17R_SPIN_LIMIT (1L << 19)   /* pauses before self-park: ~20 ms */

static void *l17r_worker(void *argp)
{
    struct l17r_warg *a = argp;
    l17r_pool *const pl = a->pl;
    const int slot = a->slot;
    if (a->cpu >= 0) {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(a->cpu, &cs);
        pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    }
    int seen = 0;
    for (;;) {
        long spins = 0;
        while (atomic_load_explicit(&pl->rel[slot].v,
                                    memory_order_acquire) == seen
               && !atomic_load_explicit(&pl->quit, memory_order_relaxed)) {
            __builtin_ia32_pause();
            if (atomic_load_explicit(&pl->hold, memory_order_relaxed)
                || ++spins > L17R_SPIN_LIMIT) {
                pthread_mutex_lock(&pl->mu);
                atomic_store_explicit(&pl->pkd[slot].v, 1,
                                      memory_order_seq_cst);
                while (atomic_load_explicit(&pl->rel[slot].v,
                                            memory_order_seq_cst) == seen
                       && !atomic_load_explicit(&pl->quit,
                                                memory_order_relaxed))
                    pthread_cond_wait(&pl->cv, &pl->mu);
                atomic_store_explicit(&pl->pkd[slot].v, 0,
                                      memory_order_relaxed);
                pthread_mutex_unlock(&pl->mu);
                spins = 0;
            }
        }
        if (atomic_load_explicit(&pl->quit, memory_order_relaxed))
            return NULL;
        seen = atomic_load_explicit(&pl->rel[slot].v, memory_order_acquire);
        /* This worker's flag was bumped AFTER the descriptor was written
         * (release store), so the reads below are never stale; the caller
         * collects fin[slot] before the descriptor changes again.  Workers
         * outside the team are never released and never touch anything. */
        if (slot + 1 < pl->job.nt && pl->job.fn)
            pl->job.fn(pl->job.p, slot + 1, pl->job.nt,
                       pl->job.in, pl->job.out);
        atomic_store_explicit(&pl->fin[slot].v, seen, memory_order_release);
    }
}

static void l17r_pool_destroy(l17r_pool *pl)
{
    if (!pl) return;
    atomic_store(&pl->quit, 1);
    pthread_mutex_lock(&pl->mu);
    pthread_cond_broadcast(&pl->cv);
    pthread_mutex_unlock(&pl->mu);
    for (int k = 0; k < pl->nw; ++k)
        pthread_join(pl->th[k], NULL);
    pthread_mutex_destroy(&pl->mu);
    pthread_cond_destroy(&pl->cv);
    free(pl);
}

/* cpus[t] = the core OMP thread t is bound to (cpus[0] is the caller's) */
static l17r_pool *l17r_pool_create(int nw, const int *cpus)
{
    l17r_pool *pl = calloc(1, sizeof *pl);
    if (!pl) return NULL;
    pthread_mutex_init(&pl->mu, NULL);
    pthread_cond_init(&pl->cv, NULL);
    atomic_store(&pl->hold, 1);          /* born parked */
    for (int k = 0; k < nw; ++k) {
        pl->warg[k].pl = pl;
        pl->warg[k].slot = k;
        pl->warg[k].cpu = cpus ? cpus[k + 1] : -1;
        if (pthread_create(&pl->th[k], NULL, l17r_worker, &pl->warg[k])) {
            l17r_pool_destroy(pl);
            return NULL;
        }
        pl->nw = k + 1;
    }
    return pl;
}

static void l17r_pool_hold(l17r_pool *pl, int hold)
{
    if (!pl) return;
    atomic_store(&pl->hold, hold);
}

/* dispatch: publish the job, release-bump ONLY the team's per-worker
 * flags, run share 0, then collect only those workers' done flags.  On
 * the two-socket node a small team's dispatch now touches no remote-socket
 * line at all (mt_r1's shared go/done bounced through all 31 workers). */
static void l17r_pool_run(fft3d_plan *p, l17r_workfn fn, int nt,
                          const double _Complex *in, double _Complex *out)
{
    l17r_pool *const pl = p->pool;
    if (!pl || nt <= 1) {
        fn(p, 0, 1, in, out);
        return;
    }
    if (nt > pl->nw + 1) nt = pl->nw + 1;
    int g = ++pl->gen;
    pl->job.p = p;
    pl->job.in = in;
    pl->job.out = out;
    pl->job.fn = fn;
    pl->job.nt = nt;
    pl->job.gen = g;
    for (int t = 1; t < nt; ++t)
        atomic_store_explicit(&pl->rel[t - 1].v, g, memory_order_seq_cst);
    /* Wake only if a TEAM worker is parked.  A global "anyone parked?"
     * check broadcast-woke all ~30 permanently-parked non-team workers on
     * EVERY small-team dispatch (they re-slept at once): measured on
     * wallaby as B=1 vp nt=2 at 80 us against 7 us steady state.  The
     * per-worker parked flag is written under the mutex before the worker
     * re-checks its release flag (both seq_cst), so a dispatcher that
     * reads pkd=0 is guaranteed the worker's re-check sees this job's
     * release store -- the same Dekker pattern as mt_r1's nparked, made
     * per-worker. */
    int wake = 0;
    for (int t = 1; t < nt && !wake; ++t)
        wake = atomic_load_explicit(&pl->pkd[t - 1].v, memory_order_seq_cst);
    if (wake) {
        pthread_mutex_lock(&pl->mu);
        pthread_cond_broadcast(&pl->cv);
        pthread_mutex_unlock(&pl->mu);
    }
    fn(p, 0, nt, in, out);
    for (int t = 1; t < nt; ++t)
        while (atomic_load_explicit(&pl->fin[t - 1].v,
                                    memory_order_acquire) != g)
            __builtin_ia32_pause();
}

/* Flat arrival-flag/release barrier for the nt team members of a job,
 * ADOPTED FROM L17_winograd mt_r1: each arriver writes its OWN padded
 * line, rank 0 scans them (independent lines, the misses overlap) and
 * publishes one release word.  Their measurement: ~0.3-0.4 us against
 * ~1.2 us for a central atomic counter at nt=16 (the mt_r1 shape here) --
 * the counter serializes nt RFOs on one line.  The epoch is the dispatch
 * generation, so a thread that sat out earlier dispatches can never be
 * out of phase; one barrier per job by construction (work_vp). */
static inline void l17r_team_barrier(l17r_pool *pl, int rank, int nt)
{
    if (!pl || nt <= 1) return;
    const int g = pl->job.gen;
    if (rank == 0) {
        for (int t = 1; t < nt; ++t)
            while (atomic_load_explicit(&pl->arr[t].v,
                                        memory_order_acquire) != g)
                __builtin_ia32_pause();
        atomic_store_explicit(&pl->brel, g, memory_order_release);
    } else {
        atomic_store_explicit(&pl->arr[rank].v, g, memory_order_release);
        while (atomic_load_explicit(&pl->brel, memory_order_acquire) != g)
            __builtin_ia32_pause();
    }
}

/* mode 2 work share: (b,x) plane tasks, team barrier, (b,blk) x tasks. */
static void l17r_work_vp(fft3d_plan *p, int t, int nt,
                         const double _Complex *in, double _Complex *out)
{
    const int nb = p->batch;
    const int np = nb * LN;
    const int nx = nb * 37;
    fft3d_plan *const q = p->sh[t];
    const int p0 = l17r_cut(np, nt, t), p1 = l17r_cut(np, nt, t + 1);
    for (int i = p0; i < p1; ++i) {
        const int b = i / LN, x = i % LN;
        l17r_plane_w8((const double *)in + (size_t)2*NVOL*b, x, q,
                      p->vpar + (size_t)b*ABUF8,
                      p->vpai + (size_t)b*ABUF8);
    }
    l17r_team_barrier(p->pool, t, nt);
    const int x0 = l17r_cut(nx, nt, t), x1 = l17r_cut(nx, nt, t + 1);
    for (int i = x0; i < x1; ++i) {
        const int b = i / 37, blk = i % 37;
        if (p->xpf && i + 1 < x1) {
            /* mt_r3, ADOPTED FROM L17_matrixsimd mt_r1 ("xpf"): every x
             * block reads 2x17 lines of A that are dirty in the plane
             * writers' caches; pull the NEXT block's lines under THIS
             * block's ~300-cycle compute so the cross-core (on the node:
             * cross-socket) transfers overlap work instead of stalling
             * the kernel's loads.  Read prefetch only -- no value or
             * ordering change, bit class untouched. */
            const int b2 = (i + 1) / 37, k2 = (i + 1) % 37;
            const double *qr = p->vpar + (size_t)b2*ABUF8;
            const double *qi = p->vpai + (size_t)b2*ABUF8;
            const long m2 = k2 < 36 ? 8L*k2 : (long)(NPL - 4);
            for (int j = 0; j < LN; ++j) {
                __builtin_prefetch(qr + m2 + (long)j*PS8, 0, 3);
                __builtin_prefetch(qi + m2 + (long)j*PS8, 0, 3);
            }
        }
        xblk_run_w8(p->vpar + (size_t)b*ABUF8,
                    p->vpai + (size_t)b*ABUF8,
                    (double *)out + (size_t)2*NVOL*b, blk, 0);
    }
}

/* mode 1 work share: thread t runs its shadow plan's exec on its own
 * contiguous chunk of volumes.  The shadows' batch/boff/pf/pfw/exec are
 * fixed at plan time, so every execute does identical work on identical
 * addresses -- repeatable by construction. */
static void l17r_work_batch(fft3d_plan *p, int t, int nt,
                            const double _Complex *in, double _Complex *out)
{
    (void)nt;
    fft3d_plan *const q = p->sh[t];
    if (q && q->batch > 0) {
        double t0 = l17r_now();     /* ~50 ns against a >= 1 us chunk; pure
                                     * telemetry (the equal-cut spread in the
                                     * description) since mt_r2 removed the
                                     * weighted partition */
        q->exec(q, in + (size_t)NVOL*q->boff,
                out + (size_t)NVOL*q->boff);
        double dt = l17r_now() - t0;
        if (dt < q->tsec) q->tsec = dt;   /* min across reps: noise guard */
    }
}

/* configure mode 1 for nv volumes over nt threads, variant f, knobs pf/pfw */
static void l17r_set_mode1(fft3d_plan *p, int nv, int nt, l17r_fn f,
                           int pf, int pfw)
{
    p->mode = 1;
    p->nt = nt;
    for (int t = 0; t < nt; ++t) {
        fft3d_plan *q = p->sh[t];
        q->boff  = l17r_cut(nv, nt, t);
        q->batch = l17r_cut(nv, nt, t + 1) - q->boff;
        q->pf    = pf;
        q->pfw   = pfw;
        q->exec  = f;
        q->tsec  = 1e30;           /* reset the weighted-partition timer */
    }
}

/* per-volume A slots for mode 2: nvp * (ABUF8 re + ABUF8 im) doubles.
 * ABUF8*8 = 40256 B is an exact multiple of 64, so neither the re/im halves
 * nor adjacent volumes' slots ever share a cache line. */
static int l17r_vp_alloc(fft3d_plan *p, int nvp)
{
    if (p->vpmem && p->nvp >= nvp) return 1;
    free(p->vpmem);
    p->vpmem = NULL;
    size_t nd = (size_t)2 * nvp * ABUF8;
    void *raw = NULL;
    if (posix_memalign(&raw, 64, nd * sizeof(double)) != 0 || !raw)
        return 0;
    memset(raw, 0, nd * sizeof(double));
    p->vpmem = raw;
    p->vpar = (double *)raw;
    p->vpai = (double *)raw + (size_t)nvp * ABUF8;
    p->nvp = nvp;
    return 1;
}

/* Time the REAL dispatch path (fft3d_execute, whatever mode/team/variant is
 * currently configured) on nv tuner volumes: licence/turbo warmup >= 1.5 ms
 * (always >= 2 full execs, which also absorbs tout's first-touch faults the
 * way the driver's warmup does), then min over reps, >= 64 volumes per
 * timed sample. */
static double l17r_time_cfg(fft3d_plan *p, int nv, int reps)
{
    int sb = p->batch;
    p->batch = nv;
    int inner = (64 + nv - 1) / nv;
    double w0 = l17r_now();
    fft3d_execute(p, p->tin, p->tout);
    fft3d_execute(p, p->tin, p->tout);
    while (l17r_now() - w0 < 1.5e-3)
        fft3d_execute(p, p->tin, p->tout);
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        double t0 = l17r_now();
        for (int i = 0; i < inner; ++i)
            fft3d_execute(p, p->tin, p->tout);
        double dt = l17r_now() - t0;
        if (dt < best) best = dt;
    }
    p->batch = sb;
    return best * 1e6 / ((double)nv * inner);
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != LN || batch <= 0) return NULL;
    double tc0 = l17r_now();       /* mt_r3: create() wall clock (see the
                                    * streaming-cell dwell below) */
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->batch = batch;
    p->pf = 0;
    p->pfw = 0;
    p->mode = 0;
    p->nt = 1;
    if (!l17r_alloc_scratch(p)) {
        free(p);
        return NULL;
    }

    int bestv = 0;
    const char *const *tags = l17r_tag_a;
    p->exec = l17r_cand_a[0];
    double pr_ph = 0.0, pr_xp = 0.0, pr_fu = 0.0;   /* panel_r9 probe */
    double pr_xa = 0.0, pr_xb = 0.0;   /* panel_r11: stage-1 class race,
                                        * published so a decline is data */
    double b1_st = 0.0, b1_vp = 0.0;   /* mt_r1: B=1 mode race, us/t */
    int    b1_nt = 0;
    double mt_us = 0.0;                /* mt_r1: batched winner, us/vol */
    double mt_spr = 0.0;               /* mt_r2: equal-cut chunk spread */
    int    mt_str = 0;                 /* mt_r3: streaming-cell classification */
    int    mt_tn = 1;                  /* mt_r3: NUMA nodes the team spans */
    int    mt_trust = 1;               /* mt_r3: settled-arena regime check */
    double mt_bw = 0.0;                /* mt_r3: plain-variant arena GB/s */

    /* ---- mt_r1: thread pool + per-thread NUMA-local shadow scratch ----
     * The parallel region below is the pool's birth (libgomp keeps the
     * threads for the life of the process, so fft3d_execute never creates
     * one).  Each thread allocates and memsets ITS OWN shadow plan's
     * scratch, so with OMP_PROC_BIND=close every thread's A/T/U/vo pages
     * are on its own socket -- on the 2x16 benchmark node that is the whole
     * NUMA story for our scratch. */
    int ntm = omp_get_max_threads();
    if (ntm > L17R_MAXT) ntm = L17R_MAXT;   /* never take more than given */
    if (ntm < 1) ntm = 1;
    p->ntmax = ntm;
    int shok = 0;
    int cpus[L17R_MAXT];
    for (int t = 0; t < L17R_MAXT; ++t) cpus[t] = -1;
    if (ntm > 1) {
        shok = 1;
#pragma omp parallel num_threads(ntm)
        {
            int t = omp_get_thread_num();
            cpus[t] = sched_getcpu();  /* the harness's close binding */
            fft3d_plan *q = calloc(1, sizeof *q);
            if (q && !l17r_alloc_scratch(q)) {
                free(q);
                q = NULL;
            }
            p->sh[t] = q;              /* distinct slots: no race */
        }
        for (int t = 0; t < ntm; ++t)
            if (!p->sh[t]) shok = 0;   /* partial alloc: fall back, destroy
                                        * frees whatever succeeded */
        if (shok) {
            p->pool = l17r_pool_create(ntm - 1, cpus);
            if (!p->pool) shok = 0;
        }
    }

    if (batch == 1 || (!shok && batch < 64 && batch < L17R_XF_CUT)) {
        /* Small batches: class A (X-last) only, ranked at (a cap of) the
         * plan's own batch size -- for B=1 this times exactly what the driver
         * times: one L2-resident volume. */
        double us1[L17R_NCA];
        int nv = batch < 16 ? batch : 16;
        if (l17r_tune_alloc(p, nv)) {
            l17r_settle(p, nv);
            bestv = l17r_rank(p, nv, us1, l17r_cand_a, L17R_NCA, 3);
            p->exec = l17r_cand_a[bestv];
            if (l17r_verbose())
                for (int v = 0; v < L17R_NCA; ++v)
                    fprintf(stderr, "[L17_rader tune] nv=%d  %-14s %8.3f us/transform%s\n",
                            nv, l17r_tag_a[v], us1[v], v == bestv ? "  <== kept" : "");
#if defined(L17R_FORCE)
            bestv = (L17R_FORCE) % L17R_NCA;
            p->exec = l17r_cand_a[bestv];
#endif
            /* pfw A/B on the winner at the plan's own batch (panel_r6):
             * prefetchw measured -6.8% at B=1 on wallaby for this structure
             * (the x pass's strided 128 B stores expose the out RFO even
             * L2-resident), but L36_pfa measured +13% at B=1 for theirs --
             * so it is measured here, blocked, 3% margin to switch on. */
            double bw[2] = { 1e30, 1e30 };
            int inner = (64 + nv - 1) / nv;
            int sb1 = p->batch;
            p->batch = nv;
            for (int pass = 0; pass < 2; ++pass)   /* two sweeps: order bias */
            for (int f = 0; f < 2; ++f) {
                p->pfw = f;
                p->exec(p, p->tin, p->tout);
                p->exec(p, p->tin, p->tout);
                for (int r = 0; r < 3; ++r) {
                    double t0 = l17r_now();
                    for (int q = 0; q < inner; ++q)
                        p->exec(p, p->tin, p->tout);
                    double dt = l17r_now() - t0;
                    if (dt < bw[f]) bw[f] = dt;
                }
            }
            p->batch = sb1;
            p->pfw = bw[1] < 0.97 * bw[0];
            if (l17r_verbose())
                fprintf(stderr, "[L17_rader tune] nv=%d  pfw off %.3f  on %.3f"
                                "  -> pfw=%d\n", nv, bw[0] * 1e6 / (nv * inner),
                        bw[1] * 1e6 / (nv * inner), p->pfw);
        }
#if defined(L17R_FORCE)
        bestv = (L17R_FORCE) % L17R_NCA;
        p->exec = l17r_cand_a[bestv];
#endif
    } else if (!shok) {
        /* Batched: rank BOTH classes on a working set past L3 (the candidate
         * that wins L2-resident can lose in the streaming regime --
         * L17_winograd's round-2 lesson), arena scaled to the machine's L3
         * (L17_matrixsimd panel_r4).  The class choice is measured: X-first
         * must beat the X-last incumbent by >3% to be selected (on wallaby it
         * loses outright, r4 record; the node's 22 MB L3 makes B=256 truly
         * stream, so only the node can answer -- this runs r4's requested A/B
         * at every plan).  Then A/B the cross-volume input prefetch on the
         * winner, blocked, never alternating, 3% margin to switch pf on
         * (panel_r3: a near-tie pf=1 pick at B=256 cost 7.4% steady-state). */
        int nv2 = batch < l17r_arena_nv() ? batch : l17r_arena_nv();
        if (l17r_tune_alloc(p, nv2)) {
            double usa[L17R_NCA], usb[L17R_NCB];
            int ba = 0, bb, use_b;
            l17r_settle(p, nv2);
            if (batch >= L17R_XF_CUT) {
                use_b = 1;                       /* forced (dev/monitor A/B) */
                bb = l17r_rank(p, nv2, usb, l17r_cand_b, L17R_NCB, 4);
            } else {
                ba = l17r_rank(p, nv2, usa, l17r_cand_a, L17R_NCA, 4);
                bb = l17r_rank(p, nv2, usb, l17r_cand_b, L17R_NCB, 4);
                use_b = usb[bb] < 0.97 * usa[ba];
                pr_xa = usa[ba];        /* stage-1 (0,0) race, us/vol -- */
                pr_xb = usb[bb];        /* goes out in the description  */
                if (l17r_verbose()) {
                    for (int v = 0; v < L17R_NCA; ++v)
                        fprintf(stderr, "[L17_rader tune] nv=%d %-14s %8.3f us/transform%s\n",
                                nv2, l17r_tag_a[v], usa[v],
                                (!use_b && v == ba) ? "  <== kept" : "");
                    for (int v = 0; v < L17R_NCB; ++v)
                        fprintf(stderr, "[L17_rader tune] nv=%d %-14s %8.3f us/transform%s\n",
                                nv2, l17r_tag_b[v], usb[v],
                                (use_b && v == bb) ? "  <== kept" : "");
                }
            }
            if (use_b) {
                tags = l17r_tag_b;
                bestv = bb;
                p->exec = l17r_cand_b[bestv];
            } else {
                bestv = ba;
                p->exec = l17r_cand_a[bestv];
            }
#if defined(L17R_FORCE)
            bestv = (L17R_FORCE) % (use_b ? L17R_NCB : L17R_NCA);
            p->exec = use_b ? l17r_cand_b[bestv] : l17r_cand_a[bestv];
#endif
            /* Joint (variant, pf, pfw) grid (panel_r9) -- ADOPTED FROM
             * L23_rader's panel_r8 node result: racing knobs only on the
             * stage-1 winner is a documented, repeated mistake (their
             * plain-xf + pf=2 + pw=1 combo took B=128 where a
             * stage-1-then-grid tuner could never find it).  The sp
             * write-spreading variant's mechanism composes with pfw
             * (prefetchw the rows it is about to scatter across the plane
             * phase), so it gets its (pf, pfw) shot here alongside the
             * incumbent's.  Class A only: all class-A candidates are
             * bit-identical, so a measured variant switch cannot change
             * output bits.  Disabled under L17R_FORCE so forced A/Bs stay
             * clean.  The 3% margin vs the incumbent at (0,0) stands
             * (panel_r3: a near-tie pf=1 pick cost 7.4% steady state;
             * L36_pfa r5: prefetchw on resident lines is +13% uop tax). */
            int part = -1;
#if defined(__AVX512VL__) && !defined(L17R_FORCE)
            /* panel_r10: the grid's partner slot goes to the staged dense
             * out flush ("st"), dy-matched to the winner.  sp held this slot
             * in r9 and was declined 6/6 by the node's grid, alongside q+pfw
             * at L17_winograd -- write SPREADING is falsified on this
             * machine; st instead DELETES the partial-line RFO waste of the
             * 17-stream scattered store (the r9 VERDICT's named direction:
             * "fund traffic deletion at B=256/2048").  If st itself won
             * stage 1, race its unstaged twin so the grid still spans both. */
            if (!use_b) {
                if (bestv >= 13) {
                    /* a staged variant won stage 1 outright: race its
                     * unstaged twin so the grid still spans both shapes */
                    part = (bestv == 14 || bestv == 16) ? 11 : 2;
                } else {
                    /* race the best staged candidate from stage 1's own
                     * streaming rank (st / st dy / stp / stp dy) */
                    part = 13;
                    for (int v = 14; v <= 16; ++v)
                        if (usa[v] < usa[part]) part = v;
                }
            }
#endif
            /* panel_r11: class B (the staged X-first, "xfs") rides the grid
             * as a third variant when it did not already win stage 1 -- the
             * same L23_rader lesson one level up: xfs + pfw is the DESIGNED
             * combination (the one-plane-ahead prefetchw feeds the deferred
             * plane flush), so a (0,0)-only class race can never find it.
             * The cross-class pick keeps the 3% margin AGAINST THE BEST
             * CLASS-A CONFIG, not against (0,0): bits change only when xfs
             * wins the honest joint race.  Gated off under L17R_FORCE. */
            int partb = -1;
#if !defined(L17R_FORCE)
            if (!use_b && batch < L17R_XF_CUT) partb = bb;
#endif
            double bcfg[12];
            for (int g = 0; g < 12; ++g) bcfg[g] = 1e30;
            const int inc = bestv;              /* incumbent, for the log */
            l17r_fn gfn[3];
            int gcls[3], gvar[3], nvar = 0;
            gfn[0] = p->exec; gcls[0] = use_b; gvar[0] = bestv; nvar = 1;
            if (part >= 0) {
                gfn[nvar] = l17r_cand_a[part];
                gcls[nvar] = 0; gvar[nvar] = part; ++nvar;
            }
            if (partb >= 0) {
                gfn[nvar] = l17r_cand_b[partb];
                gcls[nvar] = 1; gvar[nvar] = partb; ++nvar;
            }
            int sb = p->batch;
            p->batch = nv2;
            for (int pass = 0; pass < 2; ++pass)   /* two sweeps: order bias */
            for (int g = 0; g < 4*nvar; ++g) {
                l17r_fn fn = gfn[g >> 2];
                p->pf = g & 1;
                p->pfw = (g >> 1) & 1;
                fn(p, p->tin, p->tout);
                for (int r = 0; r < 2; ++r) {
                    double t0 = l17r_now();
                    fn(p, p->tin, p->tout);
                    double dt = l17r_now() - t0;
                    if (dt < bcfg[g]) bcfg[g] = dt;
                }
            }
            p->batch = sb;
            /* same-class configs: 3% margin vs the incumbent at (0,0) */
            int bg = 0;
            for (int g = 1; g < 4*nvar; ++g)
                if (gcls[g >> 2] == gcls[0]
                    && bcfg[g] < bcfg[bg] && bcfg[g] < 0.97 * bcfg[0]) bg = g;
            /* cross-class challenger: 3% margin vs the same-class winner */
            int bgx = -1;
            for (int g = 0; g < 4*nvar; ++g)
                if (gcls[g >> 2] != gcls[0]
                    && (bgx < 0 || bcfg[g] < bcfg[bgx])) bgx = g;
            if (bgx >= 0 && bcfg[bgx] < 0.97 * bcfg[bg]) bg = bgx;
            p->pf = bg & 1;
            p->pfw = (bg >> 1) & 1;
            if (bg >= 4) {
                bestv = gvar[bg >> 2];
                use_b = gcls[bg >> 2];
                tags = use_b ? l17r_tag_b : l17r_tag_a;
                p->exec = use_b ? l17r_cand_b[bestv] : l17r_cand_a[bestv];
            }
            if (l17r_verbose()) {
                for (int v = 0; v < nvar; ++v)
                    fprintf(stderr, "[L17_rader tune] nv=%d  %s cfg 00 %.3f"
                                    "  10 %.3f  01 %.3f  11 %.3f\n", nv2,
                            (gcls[v] ? l17r_tag_b : l17r_tag_a)[v == 0 ? inc : gvar[v]],
                            bcfg[4*v + 0] * 1e6 / nv2, bcfg[4*v + 1] * 1e6 / nv2,
                            bcfg[4*v + 2] * 1e6 / nv2, bcfg[4*v + 3] * 1e6 / nv2);
                fprintf(stderr, "[L17_rader tune] nv=%d  grid -> %s pf=%d pfw=%d\n",
                        nv2, tags[bestv], p->pf, p->pfw);
            }

            /* Streaming decomposition probe (panel_r9; pattern ADOPTED FROM
             * L36_pfa panel_r8: when the monitor cannot run your counter,
             * put the discriminating measurement inside create() and route
             * it out through the description string).  Times, on the same
             * streaming arena at (pf,pfw)=(0,0): the plane phase alone
             * (reads `in` cold, fills A -- the input-side exposure), the x
             * pass alone (reads hot A, writes `out` -- the output-side
             * exposure), and the full plain exec of the same family.
             * fu - ph - xp ~ 0 means the phases simply add; xp >> its
             * ~4.1 us compute floor quantifies the un-hidden out-burst RFO
             * on the scoring machine, which is the standing suspect for the
             * ~4 us batched gap to the fused rivals.  ~0.1 s, unscored. */
#if defined(__AVX512VL__)
            if (!use_b) {
                const int dyf = (bestv == 11 || bestv == 12);
                pr_ph = l17r_time_fn(p, nv2, dyf ? exec_phdy_w8 : exec_ph_w8, 3);
                pr_xp = l17r_time_fn(p, nv2, exec_xp_w8, 3);
                pr_fu = l17r_time_fn(p, nv2, l17r_cand_a[dyf ? 11 : 2], 3);
                if (l17r_verbose())
                    fprintf(stderr, "[L17_rader tune] nv=%d  probe ph=%.3f "
                                    "xp=%.3f fu=%.3f us/vol\n",
                            nv2, pr_ph, pr_xp, pr_fu);
            }
#endif
        }
    }

    /* ================= mt_r1: parallel mode selection ================= */
#if defined(__AVX512VL__)
    if (shok && batch == 1) {
        /* One volume over many cores may not pay at all (PANEL_BRIEF): the
         * question is answered by racing the tuned single-thread incumbent
         * against the plane-parallel mode at several team sizes, timing the
         * real dispatch path.  All candidates are bit-identical (same
         * kernel ops on disjoint regions), so the pick is free. */
        if (l17r_vp_alloc(p, 1) && l17r_tune_alloc(p, 1)) {
            /* mt_r3: team grid refined around the node's nt=16 pick
             * (12/14/20 added -- L17_winograd's r2 refinement move), and
             * the x-phase cross-core prefetch (xpf, from L17_matrixsimd)
             * raced JOINTLY with the team size: its value depends on how
             * many writers' caches the A lines are spread over, so a
             * knob-after-team race can never see the interaction (the
             * panel's standing "never race interacting knobs
             * sequentially" lesson). */
            static const int teams[] = {2, 4, 8, 12, 14, 16, 17, 20, 24, 32};
            int tlist[12], ntc = 0;
            for (unsigned i = 0; i < sizeof teams / sizeof *teams; ++i)
                if (teams[i] <= ntm) tlist[ntc++] = teams[i];
            double us_vp[12][2];
            for (int i = 0; i < ntc; ++i)
                us_vp[i][0] = us_vp[i][1] = 1e30;
            double us_st = 1e30;
            for (int pass = 0; pass < 2; ++pass) {   /* order-bias guard */
                /* the incumbent is timed with the workers PARKED (idle
                 * cores, full single-core turbo -- its real conditions if
                 * it wins and the pool is torn down) and the vp candidates
                 * with the workers spinning (their real conditions) */
                l17r_pool_hold(p->pool, 1);
                p->mode = 0;
                double u = l17r_time_cfg(p, 1, 3);
                if (u < us_st) us_st = u;
                l17r_pool_hold(p->pool, 0);
                for (int i = 0; i < ntc; ++i)
                    for (int xp = 0; xp < 2; ++xp) {
                        p->mode = 2;
                        p->nt = tlist[i];
                        p->xpf = xp;
                        u = l17r_time_cfg(p, 1, 3);
                        if (u < us_vp[i][xp]) us_vp[i][xp] = u;
                    }
            }
            int bi = 0, bx = 0;
            for (int i = 0; i < ntc; ++i)
                for (int xp = 0; xp < 2; ++xp)
                    if (us_vp[i][xp] < us_vp[bi][bx]) { bi = i; bx = xp; }
            b1_st = us_st;
            b1_vp = us_vp[bi][bx];
            b1_nt = tlist[bi];
            if (us_vp[bi][bx] < 0.97 * us_st) {  /* 3% margin: incumbent
                                                  * wins ties */
                p->mode = 2;
                p->nt = tlist[bi];
                p->xpf = bx;
            } else {
                p->mode = 0;
                p->nt = 1;
                p->xpf = 0;
            }
            if (l17r_verbose()) {
                fprintf(stderr, "[L17_rader mt] B=1 st %.3f us\n", us_st);
                for (int i = 0; i < ntc; ++i)
                    fprintf(stderr, "[L17_rader mt] B=1 vp nt=%-2d "
                            "xpf0 %.3f  xpf1 %.3f us%s\n",
                            tlist[i], us_vp[i][0], us_vp[i][1],
                            (p->mode == 2 && i == bi) ? "  <== kept" : "");
            }
        }
    } else if (shok) {
        /* Batched: volumes are embarrassingly parallel.  Rank the class-A
         * variants under the REAL 32-thread contention (the single-thread
         * winner need not survive it: with all cores issuing misses the
         * bandwidth-shaped candidates change price), then the (variant,
         * pf, pfw) grid on winner + partner (L23_rader's r8 lesson), then a
         * team-size race, and for batch < 32 a plane-parallel challenge
         * (mode 1 can only use min(batch, nt) threads there). */
        int nv2 = batch < 1024 ? batch : 1024;
        int nt1 = ntm < batch ? ntm : batch;

        /* mt_r3: streaming-cell classification from the WORKING SET, not
         * from an arena race (the mt_r2 VERDICT's prescription: at a cell
         * whose working set exceeds aggregate L3, wide/plain is the
         * incumbent and a challenger must earn displacement).  Aggregate
         * L3 = per-socket L3 x the nodes the TEAM spans (wallaby's 32
         * close-bound threads span 1 of its 2 sockets; the benchmark
         * node's span 2). */
        mt_tn = l17r_team_nodes(cpus, ntm);
        {
            long l3s = 0;
#ifdef _SC_LEVEL3_CACHE_SIZE
            l3s = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
            if (l3s <= 0) l3s = 32L << 20;
            double ws = 2.0 * (double)batch * NVOL * 16.0;   /* in + out */
            mt_str = ws > 2.0 * (double)l3s * mt_tn;
        }

        if (l17r_tune_alloc(p, nv2)) {
            static const int cva[10] = {2, 11, 13, 14, 15, 16, 17, 18, 19, 20};
            /* xl 512t / dy / st / st dy / stp / stp dy / stnt / stnt dy /
             * stpnt / stpnt dy */
            const int ncv = 10;
            double us[10];
            for (int v = 0; v < ncv; ++v) us[v] = 1e30;
            l17r_pool_hold(p->pool, 0);

            /* Clock settle with the real parallel shape.  mt_r3: at a
             * streaming cell this is now a 2 s MIGRATION settle, not a
             * 150 ms clock settle.  The arena starts driver-faithful
             * (serial memset/fill = every page on the main thread's
             * socket, mt_r2), but the SCORED steady state is not that:
             * with kernel NUMA balancing active (anb=1 on the node,
             * L17_matrixsimd's probe), the driver's own in/out pages
             * migrate toward the statically-cut threads that keep
             * faulting them over the multi-second warmup+timed loop.
             * mt_r2 raced candidates in the pre-migration regime and
             * shipped an NT pick that scored 2.20 us/t while
             * L17_winograd's plain static split scored 1.22 on the same
             * node.  Two seconds of full-team static-cut streaming over
             * the arena walks it to the same owner-local steady state,
             * so the race below prices the regime the driver scores. */
            l17r_set_mode1(p, nv2, nt1, l17r_cand_a[2], 0, 0);
            {
                int sb2 = p->batch;
                p->batch = nv2;
                double t0 = l17r_now();
                do
                    fft3d_execute(p, p->tin, p->tout);
                while (l17r_now() - t0 < (mt_str ? 2.0 : 0.15));
                p->batch = sb2;
            }

            for (int pass = 0; pass < 2; ++pass)
                for (int v = 0; v < ncv; ++v) {
                    l17r_set_mode1(p, nv2, nt1, l17r_cand_a[cva[v]], 0, 0);
                    double u = l17r_time_cfg(p, nv2, 3);
                    if (u < us[v]) us[v] = u;
                }
            int w = 0;
            for (int v = 1; v < ncv; ++v)
                if (us[v] < us[w]) w = v;

            /* mt_r3 streaming discipline.  Plain traffic per volume is
             * rd + wr + RFO = 3 x 78608 B, so the best plain candidate's
             * arena time converts to achieved bandwidth.  On a multi-node
             * team, a plain full-team static split below ~130 GB/s means
             * the arena is still in the unspread (all-pages-one-socket)
             * regime -- the regime whose candidate ranking did NOT
             * transfer in mt_r1 (0.79 arena / 2.90 scored) or mt_r2 (NT
             * picked, 2.20 scored vs plain 1.22) -- so its NT-vs-plain
             * verdict is not evidence.  Then: pin plain (the only store
             * shape measured fast on the node's streaming cell, 193 GB/s,
             * L17_winograd mt_r2).  If the arena DID reach the spread
             * regime (or the team is single-node, like wallaby, where
             * mt_r2's honest race stands), trust the race -- but a
             * staged/NT winner must beat the best plain by 10%, not 3%:
             * the near-tie NT pick is the shape of both prior losses. */
            int wpl = us[0] <= us[1] ? 0 : 1;
            if (mt_str) {
                mt_bw = 235.824 / us[wpl];        /* GB/s from us/vol */
                mt_trust = (mt_tn <= 1) || (mt_bw >= 130.0);
                if (!mt_trust)
                    w = wpl;
                else if (w >= 2 && !(us[w] < 0.90 * us[wpl]))
                    w = wpl;
            }

            /* partner: best of the OTHER store-shape group (unstaged 0-1,
             * staged 2-9), so the grid spans both shapes.  mt_r3: in
             * streaming safe mode (unspread multi-node arena) the staged
             * group is not priceable, so the partner is the other plain. */
            int part;
            if (mt_str && !mt_trust) {
                part = w == 0 ? 1 : 0;
            } else if (w <= 1) {
                part = 2;
                for (int v = 3; v < ncv; ++v)
                    if (us[v] < us[part]) part = v;
            } else {
                part = us[0] <= us[1] ? 0 : 1;
            }
            if (l17r_verbose())
                for (int v = 0; v < ncv; ++v)
                    fprintf(stderr, "[L17_rader mt] nv=%d nt=%d %-14s %8.3f us/t%s\n",
                            nv2, nt1, l17r_tag_a[cva[v]], us[v],
                            v == w ? "  <== stage-1" : "");

            /* joint (variant, pf, pfw) grid; 3% margin to leave the
             * winner's (0,0) (panel_r3: near-tie knob flips cost steady
             * state) */
            double g[8];
            for (int gi = 0; gi < 8; ++gi) g[gi] = 1e30;
            for (int pass = 0; pass < 2; ++pass)
                for (int gi = 0; gi < 8; ++gi) {
                    int v = (gi >> 2) ? part : w;
                    l17r_set_mode1(p, nv2, nt1, l17r_cand_a[cva[v]],
                                   gi & 1, (gi >> 1) & 1);
                    double u = l17r_time_cfg(p, nv2, 2);
                    if (u < g[gi]) g[gi] = u;
                }
            /* winner-family cells keep the 3% margin vs the winner's
             * (0,0); a STAGED/NT partner at a streaming cell must beat
             * the winner family by 10% (mt_r3 -- the same asymmetric
             * margin as stage 1, for the same two scored losses). */
            int bg = 0;
            for (int gi = 1; gi < 4; ++gi)
                if (g[gi] < g[bg] && g[gi] < 0.97 * g[0]) bg = gi;
            {
                int bp = 4;
                for (int gi = 5; gi < 8; ++gi)
                    if (g[gi] < g[bp]) bp = gi;
                double pm = (mt_str && part >= 2) ? 0.90 : 0.97;
                if (g[bp] < pm * g[bg]) bg = bp;
            }
            int fv  = (bg >> 2) ? part : w;
            int fpf = bg & 1, fpfw = (bg >> 1) & 1;
            mt_us = g[bg];

            /* team-size race on the final config: at a bandwidth-bound
             * batch, fewer teams can win (less contention, same fill-buffer
             * pool per core) -- measured, 3% margin to shrink.  mt_r3: NOT
             * raced at streaming cells any more -- hard-pinned wide.  The
             * mt_r2 VERDICT measured the same binary spanning 85 vs 200
             * GB/s on team width alone (L6_pfa B=65536) and named the
             * mechanism: the arena races in the pre-spread regime where a
             * narrow team looks good, the scored loop reaches the spread
             * regime where the wide team wins, and the tuner has locked
             * itself out.  Its prescription ("stop racing team width at
             * streaming cells and hard-pin T=32") is applied verbatim. */
            int ntf = nt1;
            if (!mt_str && nt1 >= 8) {
                int nto[3] = { nt1, (3 * nt1) / 4, nt1 / 2 };
                double tn[3] = { 1e30, 1e30, 1e30 };
                for (int pass = 0; pass < 2; ++pass)
                    for (int i = 0; i < 3; ++i) {
                        l17r_set_mode1(p, nv2, nto[i], l17r_cand_a[cva[fv]],
                                       fpf, fpfw);
                        double u = l17r_time_cfg(p, nv2, 2);
                        if (u < tn[i]) tn[i] = u;
                    }
                int ti = 0;
                for (int i = 1; i < 3; ++i)
                    if (tn[i] < tn[ti] && tn[i] < 0.97 * tn[0]) ti = i;
                ntf = nto[ti];
                if (tn[ti] < mt_us) mt_us = tn[ti];
                if (l17r_verbose())
                    fprintf(stderr, "[L17_rader mt] nt race %d/%d/%d -> "
                            "%.3f/%.3f/%.3f us/t, nt=%d\n",
                            nto[0], nto[1], nto[2], tn[0], tn[1], tn[2], ntf);
            }

            /* mt_r2: the mt_r1 "measured weighted partition" is REMOVED.
             * Its premise was wrong twice over: (a) it assumed `out` pages
             * are distributed by warmup first-touch when the driver in fact
             * memsets them all onto socket 0 before create; (b) its trigger
             * fired on single-socket wallaby (measured "spread 1.44x" where
             * no structural imbalance exists), i.e. the per-thread min-tsec
             * rates are transient noise -- and a noise-derived cut FREEZES a
             * skew (clamped chunks up to ~1.6x the mean) that every scored
             * execute then pays.  Node evidence: B=4096 scored 2.904 us/t
             * against the arena's 0.793 and L17_winograd's equal-cut 1.222.
             * The per-thread chunk timing stays (2 vDSO clock reads per
             * thread per execute, ~0.001% of a batched call) purely as
             * telemetry: the equal-cut spread rides the description string
             * so the node can tell us whether a structural imbalance even
             * exists before any future round re-attempts weighting. */
            double spread = 0.0;
            if (ntf > 1) {
                l17r_set_mode1(p, nv2, ntf, l17r_cand_a[cva[fv]], fpf, fpfw);
                double ueq = l17r_time_cfg(p, nv2, 3);
                double tmin = 1e30, tmax = 0.0;
                for (int t = 0; t < ntf; ++t) {
                    double ts = p->sh[t]->tsec;
                    if (p->sh[t]->batch > 0 && ts > 0.0 && ts < 1e29) {
                        if (ts < tmin) tmin = ts;
                        if (ts > tmax) tmax = ts;
                    }
                }
                if (tmax > 0.0 && tmin < 1e29) spread = tmax / tmin;
                if (ueq < mt_us) mt_us = ueq;
                if (l17r_verbose())
                    fprintf(stderr, "[L17_rader mt] equal cut %.3f us/t, "
                            "chunk spread %.2fx (telemetry only)\n",
                            ueq, spread);
            }
            mt_spr = spread;

#ifdef _OPENMP
            /* mt_r3: dispatch-shape race at streaming cells, pool vs OMP
             * parallel region, identical work share either way (same
             * l17r_work_batch, same shadow plans, bit-identical).  Both
             * L=17 pool entries scored 2.2-2.9 us/t at B=4096 on the node
             * while the one OMP-region entry (L17_winograd) scored 1.220,
             * and the panel-wide split is the same (L6_unrolled disp=omp
             * 200 GB/s vs every pool entry at 66-82); whether the pool
             * itself is implicated cannot be settled from here, so it is
             * raced -- and on a multi-node team the OMP shape wins TIES,
             * because it is the shape with node evidence at this cell.
             * While the OMP team is timed the pool workers are parked
             * (they would otherwise spin on the same cores). */
            if (mt_str && ntf > 1) {
                l17r_set_mode1(p, nv2, ntf, l17r_cand_a[cva[fv]], fpf, fpfw);
                double du[2] = { 1e30, 1e30 };
                for (int pass = 0; pass < 2; ++pass)
                    for (int d = 0; d < 2; ++d) {
                        p->dsp = d;
                        if (d) {
                            l17r_pool_hold(p->pool, 1);
                            double th = l17r_now();
                            while (l17r_now() - th < 2e-3)
                                sched_yield();   /* let workers park */
                        } else {
                            l17r_pool_hold(p->pool, 0);
                        }
                        double u = l17r_time_cfg(p, nv2, 2);
                        if (u < du[d]) du[d] = u;
                    }
                if (mt_tn > 1) p->dsp = !(du[0] < 0.97 * du[1]);
                else           p->dsp = (du[1] < 0.97 * du[0]);
                l17r_pool_hold(p->pool, p->dsp);
                if (l17r_verbose())
                    fprintf(stderr, "[L17_rader mt] dispatch pool %.3f "
                            "omp %.3f us/t -> %s\n", du[0], du[1],
                            p->dsp ? "omp" : "pool");
                if (du[p->dsp] < mt_us) mt_us = du[p->dsp];
            }
#endif

            /* mt_r3: streaming-cell create() dwell.  The driver memsets
             * `out` and freads `in` on its main thread BEFORE create, so
             * on the node every caller page starts on socket 0; the pages
             * the scored loop actually runs on are wherever the kernel's
             * NUMA balancer has moved them by the time the timed samples
             * run, and the balancer needs WALL TIME while the process
             * runs to mark 1.2 GB of PTEs (default scan delay 1 s, ~256
             * MB per scan pass) before the warmup executes can fault
             * pages over to the threads that use them.  The mt_r2 node
             * data splits exactly on this: every streaming-cell setup
             * >= 2.7 s (L17_winograd 4.2, L6_unrolled 2.7, L6_pfa 12.8)
             * scored 193-200 GB/s, every setup <= 0.9 s (mine 0.68,
             * L17_matrixsimd 0.87, L23_matrixsimd 0.47) scored 66-82.
             * So create() now keeps the final configuration streaming on
             * the arena until 5.5 s of wall time have passed: honest work
             * (it is the exact scored code path, and further settles the
             * pick's pages and clocks), fully inside the "arbitrarily
             * expensive" setup budget, and it buys the scored loop the
             * same marked-PTE head start the round's fast entries had. */
            if (mt_str) {
                int sbd = p->batch;
                p->batch = nv2;
                while (l17r_now() - tc0 < 5.5)
                    fft3d_execute(p, p->tin, p->tout);
                p->batch = sbd;
            }

            /* final mode-1 config on the plan's real batch: equal cut */
            l17r_set_mode1(p, batch, ntf, l17r_cand_a[cva[fv]], fpf, fpfw);
            p->pf = fpf;
            p->pfw = fpfw;
            bestv = cva[fv];
            tags = l17r_tag_a;

            /* small batches: mode 1 idles nt-ntf cores when batch < ntm;
             * challenge with the plane-parallel mode on the full team */
            if (batch < 32 && l17r_vp_alloc(p, batch)) {
                int vteams[2] = { ntm >= 16 ? 16 : ntm, ntm };
                int nvt = (vteams[0] == vteams[1]) ? 1 : 2;
                double uv[2] = { 1e30, 1e30 };
                for (int pass = 0; pass < 2; ++pass)
                    for (int i = 0; i < nvt; ++i) {
                        p->mode = 2;
                        p->nt = vteams[i];
                        double u = l17r_time_cfg(p, batch, 3);
                        if (u < uv[i]) uv[i] = u;
                    }
                int vi = (nvt == 2 && uv[1] < uv[0]) ? 1 : 0;
                if (l17r_verbose())
                    fprintf(stderr, "[L17_rader mt] B=%d vp nt=%d %.3f "
                            "vs mode1 %.3f us/t\n",
                            batch, vteams[vi], uv[vi], mt_us);
                if (uv[vi] < 0.97 * mt_us) {
                    p->mode = 2;
                    p->nt = vteams[vi];
                    mt_us = uv[vi];
                } else {
                    p->mode = 1;   /* restore: shadows still hold the final
                                    * mode-1 config for the real batch */
                    p->nt = ntf;
                }
            }
        } else {
            /* arena OOM: parallelise with the safe default, no ranking */
            l17r_set_mode1(p, batch, nt1, l17r_cand_a[2], 0, 0);
            bestv = 2;
        }
    }
#else  /* !__AVX512VL__: dev hosts only -- parallelise the plain w4 path */
    if (shok && batch > 1) {
        int nt1 = ntm < batch ? ntm : batch;
        l17r_set_mode1(p, batch, nt1, l17r_cand_a[0], 0, 0);
        bestv = 0;
    }
#endif

    l17r_tune_free(p);

#if defined(L17R_FORCE_PF)
    p->pf = (L17R_FORCE_PF);
#endif
#if defined(L17R_FORCE_PFW)
    p->pfw = (L17R_FORCE_PFW);
#endif

    /* measured sustained clock at both widths (unscored; VERDICT r4's ask);
     * workers parked so the probe sees the quiet-machine clock */
    l17r_pool_hold(p->pool, 1);
    double g256 = l17r_probe_ghz(0);
    double g512 = l17r_probe_ghz(1);

    /* pool disposition: modes 1/2 keep it (unheld); mode 0 tears it down so
     * a single-thread plan leaves the other 31 cores truly idle.  mt_r3: an
     * OMP-dispatched mode-1 plan tears it down too -- its execute never
     * touches the pool, and 31 spinning/parked extra threads have no
     * business existing alongside the OMP team. */
    if (p->pool) {
        if (p->mode == 0 || (p->mode == 1 && p->dsp)) {
            l17r_pool_destroy(p->pool);
            p->pool = NULL;
        } else {
            l17r_pool_hold(p->pool, 0);
        }
    }

    {
        int n = snprintf(g_desc, sizeof g_desc,
                 "Rader-17 cyclic/negacyclic (kernel from L17_winograd), "
                 "plane-fused; mt mode=%d nt=%d, %s, pf=%d, pfw=%d, "
                 "clk256=%.2f clk512=%.2f",
                 p->mode, p->mode ? p->nt : 1,
                 tags[bestv], p->pf, p->pfw, g256, g512);
        if (b1_st > 0.0 && n > 0 && (size_t)n < sizeof g_desc)
            n += snprintf(g_desc + n, sizeof g_desc - (size_t)n,
                          ", b1 st/vp(nt=%d,xpf=%d)=%.2f/%.2f us",
                          b1_nt, p->xpf, b1_st, b1_vp);
        if (mt_us > 0.0 && n > 0 && (size_t)n < sizeof g_desc)
            n += snprintf(g_desc + n, sizeof g_desc - (size_t)n,
                          ", par=%.3f us/t cut=eq spr=%.2fx dsp=%s"
                          " str=%d tn=%d tr=%d bw=%.0fGB/s",
                          mt_us, mt_spr, p->dsp ? "omp" : "pool",
                          mt_str, mt_tn, mt_trust, mt_bw);
        if (pr_xb > 0.0 && n > 0 && (size_t)n < sizeof g_desc)
            n += snprintf(g_desc + n, sizeof g_desc - (size_t)n,
                          ", xrace xl/xfs=%.2f/%.2f", pr_xa, pr_xb);
        if (pr_fu > 0.0 && n > 0 && (size_t)n < sizeof g_desc)
            snprintf(g_desc + n, sizeof g_desc - (size_t)n,
                     ", probe ph/xp/fu=%.2f/%.2f/%.2f us/vol",
                     pr_ph, pr_xp, pr_fu);
    }
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    l17r_pool_destroy(p->pool);
    l17r_tune_free(p);   /* normally already freed at the end of create() */
    for (int t = 0; t < L17R_MAXT; ++t)
        if (p->sh[t]) {
            free(p->sh[t]->mem);
            free(p->sh[t]);
        }
    free(p->vpmem);
    free(p->mem);
    free(p);
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    if (p->mode == 1) {
        if (p->dsp) {
            /* mt_r3: OMP-region dispatch, raced against the pool at plan
             * time (streaming cells only; dsp is never set without
             * OpenMP).  At a >= 1 ms call the region cost (~10 us at
             * nt=32) is noise, and it is the dispatch shape of the ONLY
             * schedule measured at ~193 GB/s on the node's streaming cell
             * (L17_winograd mt_r2, 1.220 us/t, mt[n=32 dyn=0]).  Work
             * share and shadow plans identical to the pool path: OMP
             * thread t is close-bound to the same core whose shadow
             * scratch it first-touched in create -- bit-identical. */
#pragma omp parallel num_threads(p->nt)
            l17r_work_batch(p, omp_get_thread_num(), p->nt, in, out);
        } else {
            l17r_pool_run(p, l17r_work_batch, p->nt, in, out);
        }
    } else if (p->mode == 2)
        l17r_pool_run(p, l17r_work_vp, p->nt, in, out);
    else
        p->exec(p, in, out);
}

#else  /* L17R_TEMPLATE_PASS ============================================== */
/* ===========================================================================
 * TEMPLATE BODY -- instantiated once per vector width.
 * Inputs: VW (doubles per vector), SFX(x) (name mangler).
 * =========================================================================== */

#define TR  (((LN  + VW - 1) / VW) * VW)   /* plane row stride:  20 / 24  */
#define PS  (((NPL + VW - 1) / VW) * VW)   /* volume plane pitch: 292 / 296 */

typedef double    SFX(vdt) __attribute__((vector_size(VW * 8), aligned(8)));
typedef long long SFX(vlt) __attribute__((vector_size(VW * 8)));
#define vd SFX(vdt)
#define vl SFX(vlt)

#if defined(__clang__)
#  if VW == 8
#    define ILO(r,i) __builtin_shufflevector((r),(i),0,8,1,9,2,10,3,11)
#    define IHI(r,i) __builtin_shufflevector((r),(i),4,12,5,13,6,14,7,15)
#    define DLE(a,b) __builtin_shufflevector((a),(b),0,2,4,6,8,10,12,14)
#    define DLO(a,b) __builtin_shufflevector((a),(b),1,3,5,7,9,11,13,15)
#  else
#    define ILO(r,i) __builtin_shufflevector((r),(i),0,4,1,5)
#    define IHI(r,i) __builtin_shufflevector((r),(i),2,6,3,7)
#    define DLE(a,b) __builtin_shufflevector((a),(b),0,2,4,6)
#    define DLO(a,b) __builtin_shufflevector((a),(b),1,3,5,7)
#  endif
#else
#  if VW == 8
#    define ILO(r,i) __builtin_shuffle((r),(i),(vl){0,8,1,9,2,10,3,11})
#    define IHI(r,i) __builtin_shuffle((r),(i),(vl){4,12,5,13,6,14,7,15})
#    define DLE(a,b) __builtin_shuffle((a),(b),(vl){0,2,4,6,8,10,12,14})
#    define DLO(a,b) __builtin_shuffle((a),(b),(vl){1,3,5,7,9,11,13,15})
#  else
#    define ILO(r,i) __builtin_shuffle((r),(i),(vl){0,4,1,5})
#    define IHI(r,i) __builtin_shuffle((r),(i),(vl){2,6,3,7})
#    define DLE(a,b) __builtin_shuffle((a),(b),(vl){0,2,4,6})
#    define DLO(a,b) __builtin_shuffle((a),(b),(vl){1,3,5,7})
#  endif
#endif

#define VL(p)    (*(const vd *)(const void *)(p))
#define VS(p,x)  (*(vd *)(void *)(p) = (x))

/* Width-selected plane movement (panel_r8): the w8 pipeline uses the 8x8 zmm
 * transpose/deinterleave blocks, the w4 pipeline keeps the 4x4 ymm tiles.
 * Pure data movement either way -- identical values to identical places, so
 * the bit classes are untouched. */
#if VW == 8
#  define TP17(s,ss,d,ds)      transpose17z((s),(ss),(d),(ds))
#  define TP17P(s,ss,d,ds,pt)  transpose17z_part((s),(ss),(d),(ds),(pt))
#  define DT17(s,dr,di,ds)     deint_transpose17z((s),(dr),(di),(ds))
#  define DT17P(s,dr,di,ds,pt) deint_transpose17z_part((s),(dr),(di),(ds),(pt))
#else
#  define TP17(s,ss,d,ds)      transpose17((s),(ss),(d),(ds))
#  define TP17P(s,ss,d,ds,pt)  transpose17_part((s),(ss),(d),(ds),(pt))
#  define DT17(s,dr,di,ds)     deint_transpose17((s),(dr),(di),(ds))
#  define DT17P(s,dr,di,ds,pt) deint_transpose17_part((s),(dr),(di),(ds),(pt))
#endif

/* ymm deinterleave inside the w8 pipeline ("dy" candidates, panel_r9).  The
 * deint is the ONE movement stage whose source loads are unaligned AND cold
 * at batch (`in` streams from DRAM there); its 64-byte loads land on the
 * 16-byte alignment classes of an odd-length complex plane and split a cache
 * line 3 times in 4, against ~1 in 2 for the 4x4 ymm tile's 32-byte loads.
 * panel_r8 measured the zmm transposes -3.0% at B=1/B=8 but +1.7%/+1.4% at
 * B=256/2048 on the node -- a batch-only regression, which is exactly the
 * cold-split-load signature (T/U/A accesses are 64-B aligned and L1-hot at
 * every batch; the deint source is the only unaligned zmm stream r8 added).
 * "dy" keeps every zmm transpose and swaps ONLY the deint back to the ymm
 * tile.  Same values to the same places: bit-identical, tuner-ranked. */
#if VW == 8
#  define DT17Y(s,dr,di,ds)   deint_transpose17((s),(dr),(di),(ds))
#else
#  define DT17Y(s,dr,di,ds)   DT17(s,dr,di,ds)
#endif

/* KPIN(c): force a broadcast constant into a register with an empty asm
 * barrier.  Without it gcc folds every constant into an FMA memory operand
 * and materialises a separate negated .LC copy for each -= (it canonicalises
 * a -= b*c into a += b*(-c)): ~148 constant-load uops per kernel
 * instantiation.  MEASURED on wallaby round panel_r2 (3 load ports): pinning
 * LOSES 5% there; a 2-load-port Cascade Lake may disagree, so both kernels
 * are built and the plan-time tuner decides. */
#if defined(__AVX512VL__)
#  define KPIN(c) ({ vd _t = ((vd){0} + (c)); __asm__("" : "+v"(_t)); _t; })
#else
#  define KPIN(c) ((vd){0} + (c))
#endif

/* -------------------------------------------------------------- the kernel */

/* VW independent 17-point DFTs.
 *   load  mode lmode 0 : split inputs at xr/xi + k*xs, k = 0..16
 *   load  mode lmode 1 : interleaved inputs at isrc + 2*(k*NPL + im0)
 *                        (deinterleaving shuffle pair per input -- the
 *                        X-first x pass reads the caller's `in` directly)
 *   store mode 0 : split store at orr/oii + k*os
 *   store mode 1 : interleaved vector store at dst + 2*(k*os + m0)
 * `pin` (compile-time 0/1) selects the pinned-S-constant variant.
 * Every input is loaded before any output is stored, so lmode 0 / mode 0 may
 * be in place. */
static inline __attribute__((always_inline)) void SFX(wino17)(
        const double *xr, const double *xi, long xs,
        const double *isrc, long im0,
        double *orr, double *oii,
        double *dst, long m0,
        long os, const int lmode, const int mode, const int pin)
{
#define LDP(k, vr, vi) do {                                                    \
        if (lmode == 0) {                                                      \
            (vr) = VL(xr + (long)(k)*xs);  (vi) = VL(xi + (long)(k)*xs);       \
        } else {                                                               \
            const double *_q = isrc + 2*((long)(k)*NPL + im0);                 \
            vd _a = VL(_q), _b = VL(_q + VW);                                  \
            (vr) = DLE(_a,_b);  (vi) = DLO(_a,_b);                             \
        }                                                                      \
    } while (0)

#define ST(k, vr, vi) do {                                                     \
        if (mode == 0) {                                                       \
            VS(orr + (long)(k)*os, (vr));  VS(oii + (long)(k)*os, (vi));       \
        } else {                                                               \
            double *_p = dst + 2*((long)(k)*os + m0);                          \
            *(vd *)(void *)(_p)      = ILO((vr),(vi));                         \
            *(vd *)(void *)(_p + VW) = IHI((vr),(vi));                         \
        }                                                                      \
    } while (0)

    vd vvr[8], vvi[8];       /* V_m = sigma[m] * (x_folded[m] - x_(17-folded[m])) */
    vd ccr[8], cci[8];       /* C_n: cyclic-8 correlation of U with c, x0-seeded  */
    vd a0r,a1r,a2r,a3r, a0i,a1i,a2i,a3i;
    vd b0r,b1r,b2r,b3r, b0i,b1i,b2i,b3i;

    /* Fold-stage constants are never pinned: that stage has 16 live
     * accumulators, and pinning its 8 constants too pushes past 32 registers
     * (round panel_r2, measured: 361 stack moves, slower than either
     * alternative). */
    const vd kCP0 = ((vd){0} + CP0), kCP1 = ((vd){0} + CP1),
             kCP2 = ((vd){0} + CP2), kCP3 = ((vd){0} + CP3);
    const vd kCM0 = ((vd){0} + CM0), kCM1 = ((vd){0} + CM1),
             kCM2 = ((vd){0} + CM2), kCM3 = ((vd){0} + CM3);

    vd x0r, x0i;
    LDP(0, x0r, x0i);
    vd dcr = x0r, dci = x0i;

    /* Fold stage: block t serves quotient slots m = t and m = t+4.
     *   vv[t] = x[j0]-x[k0], vv[t+4] = x[j1]-x[k1]   (sigma folded into order)
     *   P_t = U_t + U_(t+4) -> cyclic-4 accumulators a (x0-seeded)
     *   Q_t = U_t - U_(t+4) -> negacyclic-4 accumulators b                    */
#define FOLDCOM(t, j0,k0, j1,k1)                                               \
        vd e0r, e0i, f0r, f0i, e1r, e1i, f1r, f1i;                             \
        LDP(j0, e0r, e0i);  LDP(k0, f0r, f0i);                                 \
        LDP(j1, e1r, e1i);  LDP(k1, f1r, f1i);                                 \
        vvr[t]     = e0r - f0r;  vvi[t]     = e0i - f0i;                       \
        vvr[(t)+4] = e1r - f1r;  vvi[(t)+4] = e1i - f1i;                       \
        vd u0r = e0r + f0r, u0i = e0i + f0i;                                   \
        vd u1r = e1r + f1r, u1i = e1i + f1i;                                   \
        vd pr = u0r + u1r, pi = u0i + u1i;                                     \
        vd qr = u0r - u1r, qi = u0i - u1i;                                     \
        dcr += pr;  dci += pi;

#define FOLD0(t, j0,k0, j1,k1, w0,w1,w2,w3, z0,z1,z2,z3) do {                  \
        FOLDCOM(t, j0,k0, j1,k1)                                               \
        a0r = x0r + pr*(w0);  a1r = x0r + pr*(w1);                             \
        a2r = x0r + pr*(w2);  a3r = x0r + pr*(w3);                             \
        a0i = x0i + pi*(w0);  a1i = x0i + pi*(w1);                             \
        a2i = x0i + pi*(w2);  a3i = x0i + pi*(w3);                             \
        b0r = qr*(z0);  b1r = qr*(z1);  b2r = qr*(z2);  b3r = qr*(z3);         \
        b0i = qi*(z0);  b1i = qi*(z1);  b2i = qi*(z2);  b3i = qi*(z3);         \
    } while (0)

/* sN are += / -= tokens: negacyclic wrap terms use -= with the POSITIVE
 * constant so gcc emits vfnmadd against the shared .LC slot. */
#define FOLDN(t, j0,k0, j1,k1, w0,w1,w2,w3, s0,z0, s1,z1, s2,z2, s3,z3) do {   \
        FOLDCOM(t, j0,k0, j1,k1)                                               \
        a0r += pr*(w0);  a1r += pr*(w1);  a2r += pr*(w2);  a3r += pr*(w3);     \
        a0i += pi*(w0);  a1i += pi*(w1);  a2i += pi*(w2);  a3i += pi*(w3);     \
        b0r s0 qr*(z0);  b1r s1 qr*(z1);  b2r s2 qr*(z2);  b3r s3 qr*(z3);     \
        b0i s0 qi*(z0);  b1i s1 qi*(z1);  b2i s2 qi*(z2);  b3i s3 qi*(z3);     \
    } while (0)

    /* folded = [1,3,8,7,4,5,2,6], sigma = [+,+,-,-,-,+,-,-]; the subtraction
     * order below realises sigma, and the row rotations realise the
     * (nega)circulant structure. */
    FOLD0(0,  1,16, 13, 4,  kCP0,kCP1,kCP2,kCP3,  kCM0, kCM1, kCM2, kCM3);
    FOLDN(1,  3,14,  5,12,  kCP1,kCP2,kCP3,kCP0,  +=,kCM1, +=,kCM2, +=,kCM3, -=,kCM0);
    FOLDN(2,  9, 8, 15, 2,  kCP2,kCP3,kCP0,kCP1,  +=,kCM2, +=,kCM3, -=,kCM0, -=,kCM1);
    FOLDN(3, 10, 7, 11, 6,  kCP3,kCP0,kCP1,kCP2,  +=,kCM3, -=,kCM0, -=,kCM1, -=,kCM2);
#undef FOLD0
#undef FOLDN
#undef FOLDCOM

    ST(0, dcr, dci);                       /* X[0] = x0 + sum P_t */

    ccr[0] = a0r + b0r;  cci[0] = a0i + b0i;
    ccr[4] = a0r - b0r;  cci[4] = a0i - b0i;
    ccr[1] = a1r + b1r;  cci[1] = a1i + b1i;
    ccr[5] = a1r - b1r;  cci[5] = a1i - b1i;
    ccr[2] = a2r + b2r;  cci[2] = a2i + b2i;
    ccr[6] = a2r - b2r;  cci[6] = a2i - b2i;
    ccr[3] = a3r + b3r;  cci[3] = a3i + b3i;
    ccr[7] = a3r - b3r;  cci[7] = a3i - b3i;

    /* Dense negacyclic-8 on V, four outputs at a time:
     *   Stilde[n] = sum_m SN[(m+n) mod 8] * (-1)^floor((m+n)/8) * V_m
     * The wrapped (negated) terms are written as -= with the POSITIVE
     * constant, so gcc emits vfnmadd against the same .LC slot instead of
     * materialising 8 negated duplicates (measured: without this gcc emits
     * zero vfnmadd and doubles the constant footprint).                       */
#define SACC1(m, w0,w1,w2,w3) do {                                             \
        vd tr_ = vvr[m], ti_ = vvi[m];                                         \
        a0r = tr_*(w0);  a1r = tr_*(w1);  a2r = tr_*(w2);  a3r = tr_*(w3);     \
        a0i = ti_*(w0);  a1i = ti_*(w1);  a2i = ti_*(w2);  a3i = ti_*(w3);     \
    } while (0)
#define SACC(m, s0,w0, s1,w1, s2,w2, s3,w3) do {                               \
        vd tr_ = vvr[m], ti_ = vvi[m];                                         \
        a0r s0 tr_*(w0); a1r s1 tr_*(w1); a2r s2 tr_*(w2); a3r s3 tr_*(w3);    \
        a0i s0 ti_*(w0); a1i s1 ti_*(w1); a2i s2 ti_*(w2); a3i s3 ti_*(w3);    \
    } while (0)

    const vd kSN0 = pin ? KPIN(SN0) : ((vd){0} + SN0);
    const vd kSN1 = pin ? KPIN(SN1) : ((vd){0} + SN1);
    const vd kSN2 = pin ? KPIN(SN2) : ((vd){0} + SN2);
    const vd kSN3 = pin ? KPIN(SN3) : ((vd){0} + SN3);
    const vd kSN4 = pin ? KPIN(SN4) : ((vd){0} + SN4);
    const vd kSN5 = pin ? KPIN(SN5) : ((vd){0} + SN5);
    const vd kSN6 = pin ? KPIN(SN6) : ((vd){0} + SN6);
    const vd kSN7 = pin ? KPIN(SN7) : ((vd){0} + SN7);

    /* first half: n = 0..3 */
    SACC1(0,     kSN0,    kSN1,    kSN2,    kSN3);
    SACC(1,  +=, kSN1, +=,kSN2, +=,kSN3, +=,kSN4);
    SACC(2,  +=, kSN2, +=,kSN3, +=,kSN4, +=,kSN5);
    SACC(3,  +=, kSN3, +=,kSN4, +=,kSN5, +=,kSN6);
    SACC(4,  +=, kSN4, +=,kSN5, +=,kSN6, +=,kSN7);
    SACC(5,  +=, kSN5, +=,kSN6, +=,kSN7, -=,kSN0);
    SACC(6,  +=, kSN6, +=,kSN7, -=,kSN0, -=,kSN1);
    SACC(7,  +=, kSN7, -=,kSN0, -=,kSN1, -=,kSN2);

    /* X[folded[n]] = C_n - i*sigma[n]*Stilde_n ; X[17-folded[n]] the conjugate
     * combine.  With split re/im the *(-i) is a rename + sign in the add. */
    { vd tr_ = ccr[0], ti_ = cci[0];
      ST( 1, tr_ + a0i, ti_ - a0r);  ST(16, tr_ - a0i, ti_ + a0r); }
    { vd tr_ = ccr[1], ti_ = cci[1];
      ST( 3, tr_ + a1i, ti_ - a1r);  ST(14, tr_ - a1i, ti_ + a1r); }
    { vd tr_ = ccr[2], ti_ = cci[2];
      ST( 8, tr_ - a2i, ti_ + a2r);  ST( 9, tr_ + a2i, ti_ - a2r); }
    { vd tr_ = ccr[3], ti_ = cci[3];
      ST( 7, tr_ - a3i, ti_ + a3r);  ST(10, tr_ + a3i, ti_ - a3r); }

    /* second half: n = 4..7 */
    SACC1(0,     kSN4,    kSN5,    kSN6,    kSN7);
    SACC(1,  +=, kSN5, +=,kSN6, +=,kSN7, -=,kSN0);
    SACC(2,  +=, kSN6, +=,kSN7, -=,kSN0, -=,kSN1);
    SACC(3,  +=, kSN7, -=,kSN0, -=,kSN1, -=,kSN2);
    SACC(4,  -=, kSN0, -=,kSN1, -=,kSN2, -=,kSN3);
    SACC(5,  -=, kSN1, -=,kSN2, -=,kSN3, -=,kSN4);
    SACC(6,  -=, kSN2, -=,kSN3, -=,kSN4, -=,kSN5);
    SACC(7,  -=, kSN3, -=,kSN4, -=,kSN5, -=,kSN6);
#undef SACC
#undef SACC1

    { vd tr_ = ccr[4], ti_ = cci[4];
      ST( 4, tr_ - a0i, ti_ + a0r);  ST(13, tr_ + a0i, ti_ - a0r); }
    { vd tr_ = ccr[5], ti_ = cci[5];
      ST( 5, tr_ + a1i, ti_ - a1r);  ST(12, tr_ - a1i, ti_ + a1r); }
    { vd tr_ = ccr[6], ti_ = cci[6];
      ST( 2, tr_ - a2i, ti_ + a2r);  ST(15, tr_ + a2i, ti_ - a2r); }
    { vd tr_ = ccr[7], ti_ = cci[7];
      ST( 6, tr_ - a3i, ti_ + a3r);  ST(11, tr_ + a3i, ti_ - a3r); }
#undef ST
#undef LDP
}

/* ------------------------------------------------------------- the passes */

/* lane-block starts for a 17-wide lane space whose stores must not overrun
 * lane 16 -- the last block overlaps the previous one and recomputes a few
 * transforms, which costs nothing because the block count is unchanged. */
#define NLB ((LN + VW - 1) / VW)
static const int SFX(LBOFF)[NLB] = {
#if VW == 8
    0, 8, 9
#else
    0, 4, 8, 12, 13
#endif
};
#define NXB ((NPL + VW - 1) / VW)

/* Dense per-plane flush for the staged X-first y store (panel_r11).  The old
 * X-first y pass wrote DRAM-destined `out` through 17-row strided 128-B
 * partial-line pieces from inside the compute loop -- the exact store shape
 * the r10 VERDICT identified as the wrong baseline (both fused rivals write
 * finished planes densely; L17_matrixsimd r3 won -10.8% at B=256 with it).
 * Now the y pass stores into an L1-hot 4.6 KB staging plane and this routine
 * emits the finished plane as ONE sequential stream of full lines.  It runs
 * at the TOP of the NEXT plane's compute (staging double-buffered by kx
 * parity), so the RFO misses resolve under ~1.5 us of independent kernel
 * work -- the "stage densely, pace the flush under compute" construction the
 * monitor named, applied to the structure that has dense finished planes.
 * 2*NPL = 578 = VW*n + 2 at both widths. */
static inline __attribute__((always_inline)) void SFX(pl_flush)(
        const double *sp, double *dp)
{
    long q = 0;
    for (; q + VW <= 2L*NPL; q += VW)
        VS(dp + q, VL(sp + q));
    dp[q]     = sp[q];
    dp[q + 1] = sp[q + 1];
}

#if VW == 8
/* Write-intent prefetch of one x-pass block's 17 destination row regions in
 * `out` (panel_r6, mechanism from L8_fusedaxes/L36_pfa's node-winning pfw:
 * hide the RFO, don't avoid it).  A zmm block writes 128 B per row starting
 * at a 16-byte-aligned address, touching at most 3 cache lines; the ymm
 * tail writes 64 B (2 lines). */
static inline __attribute__((always_inline)) void SFX(pfw_rows)(
        double *dst, long m0t, const int tail)
{
    for (int k = 0; k < LN; ++k) {
        char *q = (char *)(dst + 2*((long)k*NPL + m0t));
        __builtin_prefetch(q, 1, 2);
        if (tail) {
            __builtin_prefetch(q + 63, 1, 2);
        } else {
            __builtin_prefetch(q + 64, 1, 2);
            __builtin_prefetch(q + 127, 1, 2);
        }
    }
}

/* One x-pass block (mixed shape: blk 0..35 are zmm at m0 = 8*blk, blk 36 is
 * the ymm tail at NPL-4), with optional paced prefetchw of block blk+2's
 * destination.  noinline+noclone: the sp pipeline calls this from three
 * sites, and a real call keeps ONE instantiation of the two kernels in L1i
 * (r2: 38 KB of exec body kills; the ~5-cycle call is noise against a
 * ~300-cycle block).  Arithmetic identical to exec_body's mixed x pass
 * (pin=0), so any exec built from this is bit-identical to "xl 512t". */
static __attribute__((noinline, noclone)) void SFX(xblk_run)(
        const double *ar, const double *ai, double *dst, int blk,
        const int pfw)
{
    if (pfw) {
        if (blk == 0) { SFX(pfw_rows)(dst, 0, 0); SFX(pfw_rows)(dst, 8, 0); }
        if (blk <= 33)      SFX(pfw_rows)(dst, 8L*(blk + 2), 0);
        else if (blk == 34) SFX(pfw_rows)(dst, NPL - 4, 1);
        /* blk 35's zmm region [280,288) covers the tail's [285,289) rows,
         * so the tail is fully prefetched by blk 33 and 34. */
    }
    if (blk < 36) {
        long m0 = 8L*blk;
        SFX(wino17)(ar + m0, ai + m0, PS, 0, 0, 0, 0, dst, m0,
                    NPL, 0, 1, 0);
    } else {
        wino17_w4(ar + (NPL - 4), ai + (NPL - 4), PS, 0, 0, 0, 0,
                  dst, NPL - 4, NPL, 0, 1, 0);
    }
}

/* Dense volume flush for the staged x pass ("st", panel_r10).  The scattered
 * x-pass store is 17 concurrent k-row streams of 16-B-aligned 128-B pieces:
 * at batch those are partial-line DRAM RFOs whose write-combining lines are
 * evicted half-filled (17 streams against ~12 fill buffers), i.e. the same
 * `out` line is fetched and written back more than once.  The st variants
 * point those stores at the L2-resident vo staging volume instead and emit
 * `out` as ONE sequential stream of full lines here -- prefetchable RFO,
 * perfect row locality, no partial-line waste; this is what the fused
 * rivals' dense chunk/plane stores do structurally (L17_matrixsimd r3,
 * L17_winograd g8).  The rolled loop is 1228 iterations, ~30 B of code.
 * pfw composes: one write-intent prefetch per stored line, 512 B ahead
 * (prefetch never faults, so running past the last volume's end is safe).
 * nts (mt_r1): flush with NON-TEMPORAL stores.  Single-threaded the node
 * rejected NT four rounds running (prefetchw hid the RFO more cheaply), but
 * with 32 cores sharing DRAM the batched cases are bandwidth-bound and the
 * out RFO is a third of the traffic; NT deletes it.  A volume's out base is
 * only 16-B aligned (2*NVOL doubles = 78608 B is not a line multiple), so a
 * short normal-store head aligns the stream and a normal tail finishes it;
 * vmovntpd demands the alignment.  Same values to the same places:
 * bit-identical.  sfence before returning so the worker's completion store
 * publishes finished data. */
static inline __attribute__((always_inline)) void SFX(vo_flush)(
        const double *vo, double *dst, const int pfw, const int nts)
{
    long q = 0;
#if defined(__AVX512F__) && VW == 8
    if (nts) {
        long h = (long)(((64 - ((uintptr_t)dst & 63)) & 63) / 8);
        for (; q < h; ++q) dst[q] = vo[q];
        for (; q + 8 <= 2L*NVOL; q += 8)
            _mm512_stream_pd(dst + q, _mm512_loadu_pd(vo + q));
        for (; q < 2L*NVOL; ++q) dst[q] = vo[q];
        _mm_sfence();
        return;
    }
#else
    (void)nts;
#endif
    if (pfw) {
        for (; q + 8 <= 2L*NVOL; q += 8) {
            __builtin_prefetch((const char *)(dst + q) + 512, 1, 2);
            VS(dst + q, VL(vo + q));
        }
    } else {
        for (; q + 8 <= 2L*NVOL; q += 8)
            VS(dst + q, VL(vo + q));
    }
    dst[q]     = vo[q];              /* 2*NVOL = 9826 = 8*1228 + 2 */
    dst[q + 1] = vo[q + 1];
}

/* One of 17 dense flush chunks (the "stp" pipelined flush): 576 doubles
 * (4.5 KB, 72 zmm) per plane slot, the 17th takes the 610-double remainder.
 * Same stream as vo_flush, cut so that volume b-1's flush rides under
 * volume b's plane phase one chunk per plane -- the wallaby A/B of the
 * immediate flush showed the serial burst fully exposed (no compute behind
 * it), which is exactly what pacing it under the FMA stream removes. */
static inline __attribute__((always_inline)) void SFX(vo_flush_chunk)(
        const double *vo, double *dst, const int x, const int pfw,
        const int nts)
{
    long q = 576L*x;
    const long qe = (x == 16) ? 2L*NVOL : q + 576;
#if defined(__AVX512F__) && VW == 8
    if (nts) {
        long h = (long)(((64 - ((uintptr_t)(dst + q) & 63)) & 63) / 8);
        if (q + h > qe) h = qe - q;
        for (long e = q + h; q < e; ++q) dst[q] = vo[q];
        for (; q + 8 <= qe; q += 8)
            _mm512_stream_pd(dst + q, _mm512_loadu_pd(vo + q));
        for (; q < qe; ++q) dst[q] = vo[q];
        _mm_sfence();
        return;
    }
#else
    (void)nts;
#endif
    if (pfw) {
        for (; q + 8 <= qe; q += 8) {
            __builtin_prefetch((const char *)(dst + q) + 512, 1, 2);
            VS(dst + q, VL(vo + q));
        }
    } else {
        for (; q + 8 <= qe; q += 8)
            VS(dst + q, VL(vo + q));
    }
    if (x == 16) { dst[q] = vo[q]; dst[q + 1] = vo[q + 1]; }
}
#endif

/* Compile-time flags:
 *   pin    -- S constants pinned in registers (EVEX only, tuner candidate)
 *   xfirst -- X-first pass order (bit class B, batch >= L17R_XF_CUT only)
 *   mixed  -- VW==8 only: 17-lane passes run 2 zmm blocks + 1 ymm tail
 *             (wino17_w4), and the x pass's clamped last block runs at ymm.
 *             On a 1-FMA-unit part the ymm tail halves the tail block's
 *             cycles; recomputed overlap lanes are bit-identical.
 *   dey    -- w8 pipeline with the ymm deint tile (see DT17Y above); pure
 *             data movement, bit-identical.
 *   ph     -- probe phase (plan-time only, NEVER a candidate): 0 = full,
 *             1 = plane phase only (reads `in`, fills A, never touches out),
 *             2 = x pass only (reads whatever A holds, writes out).  Used by
 *             fft3d_create()'s streaming decomposition probe (panel_r9,
 *             pattern from L36_pfa panel_r8: put the discriminating
 *             measurement in create() and route it out via description()).
 *   stg    -- VW==8 mixed only: the x pass stores into the vo staging volume
 *             and the volume is then flushed to `out` as one dense sequential
 *             stream (see vo_flush).  Same kernel operands, same values to
 *             the same final places: bit-identical, class A, tuner-ranked. */
static inline __attribute__((always_inline)) void SFX(exec_body)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out,
        const int pin, const int xfirst, const int mixed,
        const int dey, const int ph, const int stg)
{
    double *const ar = p->SFX(ar), *const ai = p->SFX(ai);
    double *const tr = p->SFX(tr), *const ti = p->SFX(ti);
    double *const ur = p->SFX(ur), *const ui = p->SFX(ui);
    (void)mixed;
    (void)stg;      /* w4 instantiation: the staged x pass is VW==8-only */

    for (int b = 0; b < p->batch; ++b) {
        const double *src = (const double *)in + (size_t)2*NVOL*b;
        double *dst       = (double *)out      + (size_t)2*NVOL*b;
        const double *nxt =
            (p->pf && b + 1 < p->batch) ? src + (size_t)2*NVOL : NULL;

        if (xfirst) {
            /* ---- x pass first: interleaved `in` -> split A[kx][y][z] ---- */
#if VW == 8
            if (mixed) {
#pragma GCC unroll 1
                for (long m0 = 0; m0 + VW <= NPL; m0 += VW)
                    SFX(wino17)(0, 0, 0, src, m0, ar + m0, ai + m0,
                                0, 0, PS, 1, 0, pin);
                wino17_w4(0, 0, 0, src, NPL - 4, ar + (NPL - 4), ai + (NPL - 4),
                          0, 0, PS, 1, 0, 0);
            } else
#endif
            for (int blk = 0; blk < NXB; ++blk) {
                long m0 = (long)blk*VW;
                if (m0 > NPL - VW) m0 = NPL - VW;
                SFX(wino17)(0, 0, 0, src, m0, ar + m0, ai + m0,
                            0, 0, PS, 1, 0, pin);
            }

            /* ---- finish one kx plane at a time (L17_matrixsimd panel_r3's
             * X-first reorder).  panel_r11: the y pass stores into an L1-hot
             * staging plane (vp0/vp1, kx parity); the PREVIOUS plane's
             * staged output is flushed to `out` as one dense sequential
             * 4.6 KB stream at the top of this plane's compute, and pfw
             * prefetches plane kx's out region one plane ahead of its own
             * flush.  Values and final addresses unchanged: bit class B is
             * preserved, all class-B candidates stay cmp-identical. ---- */
            double *const vp0 = p->vo_w8, *const vp1 = p->vo_w8 + 1024;
            const int pfw = p->pfw;
            for (int kx = 0; kx < LN; ++kx) {
                if (nxt) {
                    const double *pp = nxt + 2*(long)kx*NPL;
                    for (int q = 0; q < 2*NPL; q += 8)
                        __builtin_prefetch(pp + q, 0, 2);
                }
                if (pfw) {
                    char *q0 = (char *)(dst + 2*(long)kx*NPL);
                    for (long qq = 0; qq < 16L*NPL; qq += 64)
                        __builtin_prefetch(q0 + qq, 1, 2);
                    __builtin_prefetch(q0 + 16L*NPL - 1, 1, 2);
                }
                if (kx > 0)
                    SFX(pl_flush)((kx & 1) ? vp0 : vp1,
                                  dst + 2*(long)(kx - 1)*NPL);

                /* A[kx][y][z] -> T[z][y] */
                TP17(ar + (long)kx*PS, LN, tr, TR);
                TP17(ai + (long)kx*PS, LN, ti, TR);

                /* z pass, in place on T: axis stride TR, lanes = y */
#if VW == 8
                if (mixed) {
                    long mlim = 8;              /* opaque: keep the 2-trip   */
                    __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
                    for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                        SFX(wino17)(tr + o, ti + o, TR, 0, 0, tr + o, ti + o,
                                    0, 0, TR, 0, 0, pin);
                    wino17_w4(tr + 16, ti + 16, TR, 0, 0, tr + 16, ti + 16,
                              0, 0, TR, 0, 0, 0);
                } else
#endif
                for (int lb = 0; lb < TR/VW; ++lb) {
                    long o = (long)lb*VW;
                    SFX(wino17)(tr + o, ti + o, TR, 0, 0, tr + o, ti + o,
                                0, 0, TR, 0, 0, pin);
                }

                TP17(tr, TR, ur, TR);       /* T[kz][y] -> U[y][kz] */
                TP17(ti, TR, ui, TR);

                /* y pass: axis stride TR, lanes = kz, interleaving store
                 * into the staging plane (row stride 17 complex, L1-hot) */
                double *dp = (kx & 1) ? vp1 : vp0;
#if VW == 8
                if (mixed) {
                    long mlim = 8;              /* opaque: keep the 2-trip   */
                    __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
                    for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                        SFX(wino17)(ur + o, ui + o, TR, 0, 0, 0, 0, dp, o,
                                    LN, 0, 1, pin);
                    wino17_w4(ur + 13, ui + 13, TR, 0, 0, 0, 0, dp, 13,
                              LN, 0, 1, 0);
                } else
#endif
                for (int lb = 0; lb < NLB; ++lb) {
                    long o = SFX(LBOFF)[lb];
                    SFX(wino17)(ur + o, ui + o, TR, 0, 0, 0, 0, dp, o,
                                LN, 0, 1, pin);
                }
            }
            SFX(pl_flush)(vp0, dst + 2L*(LN - 1)*NPL);   /* plane 16 (even) */
        } else {
            /* ---------------- X-last (round-1 order) ---------------- */
            if (ph != 2)
            for (int x = 0; x < LN; ++x) {
                /* cross-volume prefetch: pull the NEXT volume's plane x while
                 * this plane's two compute passes run (L17_winograd round 2:
                 * -4.4% at B=2048; a no-op at B=1). 73 lines per plane. */
                if (nxt) {
                    const double *pp = nxt + 2*(long)x*NPL;
                    for (int q = 0; q < 2*NPL; q += 8)
                        __builtin_prefetch(pp + q, 0, 2);
                }

                /* in[x][y][z] -> T[z][y] */
                if (dey) DT17Y(src + 2*(long)x*NPL, tr, ti, TR);
                else     DT17(src + 2*(long)x*NPL, tr, ti, TR);

                /* z pass, in place on T: axis stride TR, lanes = y */
#if VW == 8
                if (mixed) {
                    long mlim = 8;              /* opaque: keep the 2-trip   */
                    __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
                    for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                        SFX(wino17)(tr + o, ti + o, TR, 0, 0, tr + o, ti + o,
                                    0, 0, TR, 0, 0, pin);
                    wino17_w4(tr + 16, ti + 16, TR, 0, 0, tr + 16, ti + 16,
                              0, 0, TR, 0, 0, 0);
                } else
#endif
                for (int lb = 0; lb < TR/VW; ++lb) {
                    long o = (long)lb*VW;
                    SFX(wino17)(tr + o, ti + o, TR, 0, 0, tr + o, ti + o,
                                0, 0, TR, 0, 0, pin);
                }

                TP17(tr, TR, ur, TR);       /* T[kz][y] -> U[y][kz] */
                TP17(ti, TR, ui, TR);

                /* y pass: axis stride TR, lanes = kz, straight into A[x][ky][kz] */
                double *dr = ar + (long)x*PS, *di = ai + (long)x*PS;
#if VW == 8
                if (mixed) {
                    long mlim = 8;              /* opaque: keep the 2-trip   */
                    __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
                    for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                        SFX(wino17)(ur + o, ui + o, TR, 0, 0, dr + o, di + o,
                                    0, 0, LN, 0, 0, pin);
                    wino17_w4(ur + 13, ui + 13, TR, 0, 0, dr + 13, di + 13,
                              0, 0, LN, 0, 0, 0);
                } else
#endif
                for (int lb = 0; lb < NLB; ++lb) {
                    long o = SFX(LBOFF)[lb];
                    SFX(wino17)(ur + o, ui + o, TR, 0, 0, dr + o, di + o,
                                0, 0, LN, 0, 0, pin);
                }
            }

            /* x pass: axis stride PS, 289 contiguous lanes, interleaving store.
             * The last block is clamped to start at NPL-VW: it overlaps the
             * previous block and recomputes a few lanes bit-identically (each
             * lane's arithmetic is independent of m0). */
            if (ph == 1) continue;
#if VW == 8
            if (mixed) {
                /* stg: kernel stores go to the L2-hot staging volume (no pfw
                 * there -- nothing DRAM-destined), then one dense flush. */
                double *const xd = stg ? p->vo_w8 : dst;
                const int pfwl = stg ? 0 : p->pfw;
                if (pfwl) { SFX(pfw_rows)(dst, 0, 0); SFX(pfw_rows)(dst, 8, 0); }
#pragma GCC unroll 1
                for (long m0 = 0; m0 + VW <= NPL; m0 += VW) {
                    if (pfwl) {
                        if (m0 <= 264)      SFX(pfw_rows)(dst, m0 + 16, 0);
                        else if (m0 == 272) SFX(pfw_rows)(dst, NPL - 4, 1);
                    }
                    SFX(wino17)(ar + m0, ai + m0, PS, 0, 0, 0, 0, xd, m0,
                                NPL, 0, 1, pin);
                }
                wino17_w4(ar + (NPL - 4), ai + (NPL - 4), PS, 0, 0, 0, 0,
                          xd, NPL - 4, NPL, 0, 1, 0);
                if (stg) SFX(vo_flush)(p->vo_w8, dst, p->pfw, stg == 2);
            } else
#endif
            for (int blk = 0; blk < NXB; ++blk) {
                long m0 = (long)blk*VW;
                if (m0 > NPL - VW) m0 = NPL - VW;
                SFX(wino17)(ar + m0, ai + m0, PS, 0, 0, 0, 0, dst, m0,
                            NPL, 0, 1, pin);
            }
        }
    }
}

#if VW == 8
/* Overlapped-shuffle X-last body ("ov"), mixed-width shape only -- panel_r5.
 * The SAME operations as exec_body(pin, 0, 1), reordered so that every
 * serialized shuffle burst (the transposes) sits in the shadow of a zmm
 * kernel block: on the node's single 512-bit FMA unit a zmm block drains
 * ~296 port-0 uops at 1/cycle while alloc (4/cycle) runs ahead and issues
 * the younger, independent shuffle/load uops on ports 5/2/3.  Specifically:
 *   - the z pass runs its ymm tail FIRST and the two zmm blocks LAST, so
 *     the T->U transpose that follows lands in a zmm drain;
 *   - T->U is emitted in halves: columns 0..7 (all the y pass's first block
 *     needs) before the y loop, columns 8..16 after the first y kernel;
 *   - the NEXT plane's deinterleave (T is dead once T->U is complete) is
 *     emitted in halves after the y pass's second zmm block and after its
 *     ymm tail; plane 0's deinterleave runs at the top of the volume, in
 *     the shadow of the previous volume's x pass.
 * All moved pieces write regions disjoint from anything concurrently live,
 * so the output is BIT-IDENTICAL to the other class-A variants (verified by
 * cmp); the tuner may rank it freely.  Wallaby (two 512-bit FMA units ->
 * drains half as long) understates the node benefit, like r4's "t" bet. */
static inline __attribute__((always_inline)) void SFX(exec_ov_body)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out,
        const int pin)
{
    double *const ar = p->SFX(ar), *const ai = p->SFX(ai);
    double *const tr = p->SFX(tr), *const ti = p->SFX(ti);
    double *const ur = p->SFX(ur), *const ui = p->SFX(ui);

    for (int b = 0; b < p->batch; ++b) {
        const double *src = (const double *)in + (size_t)2*NVOL*b;
        double *dst       = (double *)out      + (size_t)2*NVOL*b;
        const double *nxt =
            (p->pf && b + 1 < p->batch) ? src + (size_t)2*NVOL : NULL;

        /* plane 0: shadowed by the previous volume's x-pass drain (b > 0) */
        DT17P(src, tr, ti, TR, 0);
        DT17P(src, tr, ti, TR, 1);

        for (int x = 0; x < LN; ++x) {
            if (nxt) {
                const double *pp = nxt + 2*(long)x*NPL;
                for (int q = 0; q < 2*NPL; q += 8)
                    __builtin_prefetch(pp + q, 0, 2);
            }

            /* z pass, in place on T (column-disjoint blocks, any order):
             * ymm tail first, zmm blocks last */
            wino17_w4(tr + 16, ti + 16, TR, 0, 0, tr + 16, ti + 16,
                      0, 0, TR, 0, 0, 0);
            {
                long mlim = 8;              /* opaque: keep the 2-trip   */
                __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
                for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                    SFX(wino17)(tr + o, ti + o, TR, 0, 0, tr + o, ti + o,
                                0, 0, TR, 0, 0, pin);
            }

            /* T->U, columns 0..7 -- in the z pass's final zmm drain */
            TP17P(tr, TR, ur, TR, 0);
            TP17P(ti, TR, ui, TR, 0);

            /* y pass into A[x][ky][kz]; fillers ride each block's drain */
            double *dr = ar + (long)x*PS, *di = ai + (long)x*PS;
            const double *nsrc = src + 2*(long)(x + 1)*NPL;
            {
                long mlim = 8;
                __asm__("" : "+r"(mlim));
                for (long o = 0; o <= mlim; o += 8) {
                    SFX(wino17)(ur + o, ui + o, TR, 0, 0, dr + o, di + o,
                                0, 0, LN, 0, 0, pin);
                    if (o == 0) {
                        /* T->U columns 8..16 (T still live, U cols >= 8) */
                        TP17P(tr, TR, ur, TR, 1);
                        TP17P(ti, TR, ui, TR, 1);
                    } else if (x < 16) {
                        /* T is dead now: next plane's deinterleave, half 1 */
                        DT17P(nsrc, tr, ti, TR, 0);
                    }
                }
            }
            wino17_w4(ur + 13, ui + 13, TR, 0, 0, dr + 13, di + 13,
                      0, 0, LN, 0, 0, 0);
            if (x < 16)
                DT17P(nsrc, tr, ti, TR, 1);
        }

        /* x pass: unchanged mixed shape (zmm blocks + clamped ymm tail) */
#pragma GCC unroll 1
        for (long m0 = 0; m0 + VW <= NPL; m0 += VW)
            SFX(wino17)(ar + m0, ai + m0, PS, 0, 0, 0, 0, dst, m0,
                        NPL, 0, 1, pin);
        wino17_w4(ar + (NPL - 4), ai + (NPL - 4), PS, 0, 0, 0, 0,
                  dst, NPL - 4, NPL, 0, 1, 0);
    }
}

/* Software-pipelined X-last body ("sp"), mixed-width shape, panel_r6 -- r4's
 * "Next" item 3 and the monitor's named remaining lever for L=17 batched.
 * At batch, volume b's x pass is a dedicated burst writing 78.6 KB of
 * DRAM-destined `out` while the FMA stream sits mostly idle, and the plane
 * phase is compute-bound while the store path sits mostly idle.  This body
 * interleaves them ACROSS volumes: after each plane's y pass of volume b,
 * 2-3 x-pass blocks of volume b-1 run (37 blocks spread over 17 planes),
 * reading the OTHER buffer of a ping-pong A pair.  Per-volume kernel calls
 * and operands are exactly exec_body(pin=0, xfirst=0, mixed=1)'s, only the
 * global order moves across volume boundaries and volumes are independent,
 * so the output is BIT-IDENTICAL to every class-A candidate.  The x blocks
 * run behind the noinline xblk_run, so this whole body adds no inlined
 * kernel copy.  pfw composes: each block prefetchw's block k+2's rows. */
static inline __attribute__((always_inline)) void SFX(exec_sp_body)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out,
        const int pin, const int dey)
{
    double *const tr = p->SFX(tr), *const ti = p->SFX(ti);
    double *const ur = p->SFX(ur), *const ui = p->SFX(ui);
    const int pfw = p->pfw;
    const int nb = p->batch;

    for (int b = 0; b < nb; ++b) {
        const double *src = (const double *)in + (size_t)2*NVOL*b;
        double *const ar = (b & 1) ? p->ar2_w8 : p->SFX(ar);
        double *const ai = (b & 1) ? p->ai2_w8 : p->SFX(ai);
        const double *const par = (b & 1) ? p->SFX(ar) : p->ar2_w8;
        const double *const pai = (b & 1) ? p->SFX(ai) : p->ai2_w8;
        double *const pdst = (b > 0)
            ? (double *)out + (size_t)2*NVOL*(b - 1) : (double *)out;
        const double *nxt =
            (p->pf && b + 1 < nb) ? src + (size_t)2*NVOL : NULL;
        int xq = 0;

        for (int x = 0; x < LN; ++x) {
            if (nxt) {
                const double *pp = nxt + 2*(long)x*NPL;
                for (int q = 0; q < 2*NPL; q += 8)
                    __builtin_prefetch(pp + q, 0, 2);
            }

            if (dey) DT17Y(src + 2*(long)x*NPL, tr, ti, TR);
            else     DT17(src + 2*(long)x*NPL, tr, ti, TR);

            /* z pass, in place on T (identical to exec_body's mixed order) */
            {
                long mlim = 8;              /* opaque: keep the 2-trip   */
                __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
                for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                    SFX(wino17)(tr + o, ti + o, TR, 0, 0, tr + o, ti + o,
                                0, 0, TR, 0, 0, pin);
            }
            wino17_w4(tr + 16, ti + 16, TR, 0, 0, tr + 16, ti + 16,
                      0, 0, TR, 0, 0, 0);

            TP17(tr, TR, ur, TR);       /* T[kz][y] -> U[y][kz] */
            TP17(ti, TR, ui, TR);

            /* y pass into this volume's A half of the ping-pong pair */
            double *dr = ar + (long)x*PS, *di = ai + (long)x*PS;
            {
                long mlim = 8;
                __asm__("" : "+r"(mlim));
                for (long o = 0; o <= mlim; o += 8)
                    SFX(wino17)(ur + o, ui + o, TR, 0, 0, dr + o, di + o,
                                0, 0, LN, 0, 0, pin);
            }
            wino17_w4(ur + 13, ui + 13, TR, 0, 0, dr + 13, di + 13,
                      0, 0, LN, 0, 0, 0);

            /* 2-3 x-pass blocks of the PREVIOUS volume (37 over 17 planes) */
            if (b > 0) {
                int jn = ((x + 1)*37)/LN - (x*37)/LN;
                __asm__("" : "+r"(jn));
                for (int j = 0; j < jn; ++j, ++xq)
                    SFX(xblk_run)(par, pai, pdst, xq, pfw);
            }
        }
    }

    /* drain: the last volume's x pass */
    {
        const double *const lar = ((nb - 1) & 1) ? p->ar2_w8 : p->SFX(ar);
        const double *const lai = ((nb - 1) & 1) ? p->ai2_w8 : p->SFX(ai);
        double *ldst = (double *)out + (size_t)2*NVOL*(nb - 1);
        int jn = 37;
        __asm__("" : "+r"(jn));
        for (int j = 0; j < jn; ++j)
            SFX(xblk_run)(lar, lai, ldst, j, pfw);
    }
}

/* Staged-pipelined X-last body ("stp"), mixed-width shape, panel_r10.  The
 * staged x pass (stores into vo, see exec_body's stg flag) with the dense
 * flush of the PREVIOUS volume's vo paced across this volume's plane phase:
 * one 4.5 KB sequential chunk after each plane's y pass, vo ping-ponged so
 * vo(b-1) stays live while volume b's x pass fills vo(b).  This composes the
 * two lessons of the round's forced wallaby A/Bs: the scattered 17-stream
 * store is the traffic to delete (st), and an un-paced flush is a fully
 * exposed serial burst (st loses ~23% forced on wallaby streaming).  Unlike
 * sp -- whose paced work was kernel blocks with scattered partial-line
 * stores -- each paced piece here is ONE dense sequential stream, which is
 * the store pattern the hardware prefetcher and fill buffers handle at full
 * rate.  Per-volume kernel calls and operand values are exactly the staged
 * exec's; only the out-write order moves across independent volumes, so the
 * output is BIT-IDENTICAL to every class-A candidate. */
static inline __attribute__((always_inline)) void SFX(exec_stp_body)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out,
        const int pin, const int dey, const int nts)
{
    double *const ar = p->SFX(ar), *const ai = p->SFX(ai);
    double *const tr = p->SFX(tr), *const ti = p->SFX(ti);
    double *const ur = p->SFX(ur), *const ui = p->SFX(ui);
    const int pfw = p->pfw;
    const int nb = p->batch;

    for (int b = 0; b < nb; ++b) {
        const double *src = (const double *)in + (size_t)2*NVOL*b;
        double *const vo  = (b & 1) ? p->vo2_w8 : p->vo_w8;
        const double *const pvo = (b & 1) ? p->vo_w8 : p->vo2_w8;
        double *const pdst = (b > 0)
            ? (double *)out + (size_t)2*NVOL*(b - 1) : (double *)out;
        const double *nxt =
            (p->pf && b + 1 < nb) ? src + (size_t)2*NVOL : NULL;

        for (int x = 0; x < LN; ++x) {
            if (nxt) {
                const double *pp = nxt + 2*(long)x*NPL;
                for (int q = 0; q < 2*NPL; q += 8)
                    __builtin_prefetch(pp + q, 0, 2);
            }

            if (dey) DT17Y(src + 2*(long)x*NPL, tr, ti, TR);
            else     DT17(src + 2*(long)x*NPL, tr, ti, TR);

            /* z pass, in place on T (identical to exec_body's mixed order) */
            {
                long mlim = 8;              /* opaque: keep the 2-trip   */
                __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
                for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                    SFX(wino17)(tr + o, ti + o, TR, 0, 0, tr + o, ti + o,
                                0, 0, TR, 0, 0, pin);
            }
            wino17_w4(tr + 16, ti + 16, TR, 0, 0, tr + 16, ti + 16,
                      0, 0, TR, 0, 0, 0);

            TP17(tr, TR, ur, TR);       /* T[kz][y] -> U[y][kz] */
            TP17(ti, TR, ui, TR);

            /* y pass into A[x] */
            double *dr = ar + (long)x*PS, *di = ai + (long)x*PS;
            {
                long mlim = 8;
                __asm__("" : "+r"(mlim));
                for (long o = 0; o <= mlim; o += 8)
                    SFX(wino17)(ur + o, ui + o, TR, 0, 0, dr + o, di + o,
                                0, 0, LN, 0, 0, pin);
            }
            wino17_w4(ur + 13, ui + 13, TR, 0, 0, dr + 13, di + 13,
                      0, 0, LN, 0, 0, 0);

            /* one dense flush chunk of the PREVIOUS volume's staged output */
            if (b > 0)
                SFX(vo_flush_chunk)(pvo, pdst, x, pfw, nts);
        }

        /* staged x pass: kernel stores into vo (L2-hot, no pfw needed) */
#pragma GCC unroll 1
        for (long m0 = 0; m0 + VW <= NPL; m0 += VW)
            SFX(wino17)(ar + m0, ai + m0, PS, 0, 0, 0, 0, vo, m0,
                        NPL, 0, 1, pin);
        wino17_w4(ar + (NPL - 4), ai + (NPL - 4), PS, 0, 0, 0, 0,
                  vo, NPL - 4, NPL, 0, 1, 0);
    }

    /* drain: the last volume's flush */
    SFX(vo_flush)(((nb - 1) & 1) ? p->vo2_w8 : p->vo_w8,
                  (double *)out + (size_t)2*NVOL*(nb - 1), pfw, nts);
}

/* Deferred-junction plane schedule ("dz"), mixed-width shape, panel_r7 --
 * ADOPTED FROM L17_matrixsimd panel_r6's deferred-Z (group-level deferral
 * with a double-buffered plane buffer; wallaby -3.0%/-5.7% at B=1/B=8 for
 * their structure).  exec_body's plane phase runs deint(x) -> z(x) ->
 * transpose(x) -> y(x) back to back: three store->load junctions per plane
 * where a group's tail stores are immediately re-read with no independent
 * work behind them.  This body runs the SAME groups pipelined one plane
 * deep, T double-buffered by plane parity (U single: its producer and
 * consumer are one kernel group apart in the same iteration):
 *
 *     deint(0) -> T0
 *     for x = 0..17:
 *         x>=1 : transpose(x-1)  T[(x-1)&1] -> U      [z(x) before y(x-1)]
 *         x<17 : z(x)            in place on T[x&1]
 *         x>=1 : y(x-1)          U -> A[x-1]  (+ sp: prev volume's x blocks)
 *         x<16 : deint(x+1)      in[x+1] -> T[(x+1)&1]
 *     then the x pass (or, under sp, the pipelined drain)
 *
 * Junction separations: deint(x+1)->z(x+1) has transpose(x) between;
 * z(x)->transpose(x) has y(x-1) and deint(x+1) between; transpose(x-1)->
 * y(x-1) has the whole z(x) kernel group (~740 zmm cycles) between.  The
 * only exposed junctions left are plane 0's deint->z and the final
 * transpose(16)->y(16).  Pure reordering of whole groups on disjoint
 * buffers -- zero extra instructions (the ov lesson: splitting loops costs
 * more than the shadow pays), same kernel operand values, so the output is
 * BIT-IDENTICAL to every class-A candidate.  `sp` (compile-time) composes
 * the panel_r6 cross-volume x-block pipeline into the same schedule. */
static inline __attribute__((always_inline)) void SFX(exec_dz_body)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out,
        const int pin, const int sp)
{
    double *const ur = p->SFX(ur), *const ui = p->SFX(ui);
    double *const t0r = p->SFX(tr),  *const t0i = p->SFX(ti);
    double *const t1r = p->tr2_w8,   *const t1i = p->ti2_w8;
    const int pfw = p->pfw;
    const int nb = p->batch;

    for (int b = 0; b < nb; ++b) {
        const double *src = (const double *)in + (size_t)2*NVOL*b;
        double *dst       = (double *)out      + (size_t)2*NVOL*b;
        double *const ar = (sp && (b & 1)) ? p->ar2_w8 : p->SFX(ar);
        double *const ai = (sp && (b & 1)) ? p->ai2_w8 : p->SFX(ai);
        const double *const par = (b & 1) ? p->SFX(ar) : p->ar2_w8;
        const double *const pai = (b & 1) ? p->SFX(ai) : p->ai2_w8;
        double *const pdst = (b > 0)
            ? (double *)out + (size_t)2*NVOL*(b - 1) : (double *)out;
        const double *nxt =
            (p->pf && b + 1 < nb) ? src + (size_t)2*NVOL : NULL;
        int xq = 0;

        DT17(src, t0r, t0i, TR);            /* plane 0 -> T0 */

        for (int x = 0; x <= LN; ++x) {
            if (nxt && x < LN) {
                const double *pp = nxt + 2*(long)x*NPL;
                for (int q = 0; q < 2*NPL; q += 8)
                    __builtin_prefetch(pp + q, 0, 2);
            }

            /* T[(x-1)&1] -> U; its consumer y(x-1) sits behind z(x) */
            if (x >= 1) {
                const double *sr = ((x - 1) & 1) ? t1r : t0r;
                const double *si = ((x - 1) & 1) ? t1i : t0i;
                TP17(sr, TR, ur, TR);
                TP17(si, TR, ui, TR);
            }

            /* z pass, in place on T[x&1] (exec_body's mixed order) */
            if (x < LN) {
                double *zr = (x & 1) ? t1r : t0r;
                double *zi = (x & 1) ? t1i : t0i;
                long mlim = 8;              /* opaque: keep the 2-trip   */
                __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
                for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                    SFX(wino17)(zr + o, zi + o, TR, 0, 0, zr + o, zi + o,
                                0, 0, TR, 0, 0, pin);
                wino17_w4(zr + 16, zi + 16, TR, 0, 0, zr + 16, zi + 16,
                          0, 0, TR, 0, 0, 0);
            }

            /* y pass of plane x-1: U -> A[x-1] */
            if (x >= 1) {
                double *dr = ar + (long)(x - 1)*PS, *di = ai + (long)(x - 1)*PS;
                long mlim = 8;
                __asm__("" : "+r"(mlim));
                for (long o = 0; o <= mlim; o += 8)
                    SFX(wino17)(ur + o, ui + o, TR, 0, 0, dr + o, di + o,
                                0, 0, LN, 0, 0, pin);
                wino17_w4(ur + 13, ui + 13, TR, 0, 0, dr + 13, di + 13,
                          0, 0, LN, 0, 0, 0);

                /* sp: 2-3 x-pass blocks of the PREVIOUS volume per y pass */
                if (sp && b > 0) {
                    const int k = x - 1;
                    int jn = ((k + 1)*37)/LN - (k*37)/LN;
                    __asm__("" : "+r"(jn));
                    for (int j = 0; j < jn; ++j, ++xq)
                        SFX(xblk_run)(par, pai, pdst, xq, pfw);
                }
            }

            /* NEXT plane's deinterleave -> T[(x+1)&1] (T[x-1] is dead:
             * transpose(x-1) consumed it above, in program order) */
            if (x + 1 < LN) {
                double *tdr = ((x + 1) & 1) ? t1r : t0r;
                double *tdi = ((x + 1) & 1) ? t1i : t0i;
                DT17(src + 2*(long)(x + 1)*NPL, tdr, tdi, TR);
            }
        }

        if (!sp) {
            /* x pass, mixed shape with pfw -- identical to exec_body's */
            if (pfw) { SFX(pfw_rows)(dst, 0, 0); SFX(pfw_rows)(dst, 8, 0); }
#pragma GCC unroll 1
            for (long m0 = 0; m0 + VW <= NPL; m0 += VW) {
                if (pfw) {
                    if (m0 <= 264)      SFX(pfw_rows)(dst, m0 + 16, 0);
                    else if (m0 == 272) SFX(pfw_rows)(dst, NPL - 4, 1);
                }
                SFX(wino17)(ar + m0, ai + m0, PS, 0, 0, 0, 0, dst, m0,
                            NPL, 0, 1, pin);
            }
            wino17_w4(ar + (NPL - 4), ai + (NPL - 4), PS, 0, 0, 0, 0,
                      dst, NPL - 4, NPL, 0, 1, 0);
        }
    }

    if (sp) {
        /* drain: the last volume's x pass */
        const double *const lar = ((nb - 1) & 1) ? p->ar2_w8 : p->SFX(ar);
        const double *const lai = ((nb - 1) & 1) ? p->ai2_w8 : p->SFX(ai);
        double *ldst = (double *)out + (size_t)2*NVOL*(nb - 1);
        int jn = 37;
        __asm__("" : "+r"(jn));
        for (int j = 0; j < jn; ++j)
            SFX(xblk_run)(lar, lai, ldst, j, pfw);
    }
}
#endif

/* exec variants.  VW==4 contributes the plain X-last and X-first entries;
 * VW==8 additionally contributes the pinned and mixed-tail ("t") ones. */
static void __attribute__((unused)) SFX(exec_np)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 0, 0, 0, 0);
}

static void __attribute__((unused)) SFX(exec_xf)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 1, 0, 0, 0, 0);
}

#if VW == 8
static void __attribute__((unused)) SFX(exec_pin)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 1, 0, 0, 0, 0, 0);
}

static void __attribute__((unused)) SFX(exec_npm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1, 0, 0, 0);
}

static void __attribute__((unused)) SFX(exec_pinm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 1, 0, 1, 0, 0, 0);
}

static void __attribute__((unused)) SFX(exec_xfm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 1, 1, 0, 0, 0);
}

static void __attribute__((unused)) SFX(exec_xfpinm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 1, 1, 1, 0, 0, 0);
}

/* "dy": mixed X-last with the ymm deint tile (panel_r9) */
static void __attribute__((unused)) SFX(exec_npmdy)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1, 1, 0, 0);
}

/* "st": mixed X-last with the staged dense out flush (panel_r10); the dy
 * twin composes with the node's r9 batched pick (`xl 512t dy`). */
static void __attribute__((unused)) SFX(exec_stm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1, 0, 0, 1);
}

static void __attribute__((unused)) SFX(exec_stmdy)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1, 1, 0, 1);
}

/* "stp": staged x pass + the previous volume's flush paced across the plane
 * phase (panel_r10) */
static void __attribute__((unused)) SFX(exec_stpm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_stp_body)(p, in, out, 0, 0, 0);
}

static void __attribute__((unused)) SFX(exec_stpmdy)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_stp_body)(p, in, out, 0, 1, 0);
}

/* mt_r1: non-temporal flush twins of st / stp (RFO deletion for the
 * 32-core bandwidth-bound batched cells; see vo_flush's nts note) */
static void __attribute__((unused)) SFX(exec_stntm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1, 0, 0, 2);
}

static void __attribute__((unused)) SFX(exec_stntmdy)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1, 1, 0, 2);
}

static void __attribute__((unused)) SFX(exec_stpntm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_stp_body)(p, in, out, 0, 0, 1);
}

static void __attribute__((unused)) SFX(exec_stpntmdy)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_stp_body)(p, in, out, 0, 1, 1);
}

/* plan-time probes (panel_r9): NEVER candidates -- exec_xp reads whatever A
 * holds, so its output is not the transform.  They exist so create() can
 * decompose the streaming exposure into plane-phase (input-side) and x-pass
 * (output-side) shares ON THE SCORING MACHINE and report it in the
 * description string (L36_pfa panel_r8 pattern). */
static void __attribute__((unused)) SFX(exec_ph)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1, 0, 1, 0);
}

static void __attribute__((unused)) SFX(exec_phdy)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1, 1, 1, 0);
}

static void __attribute__((unused)) SFX(exec_xp)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1, 0, 2, 0);
}

static void __attribute__((unused)) SFX(exec_ovm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_ov_body)(p, in, out, 0);
}

static void __attribute__((unused)) SFX(exec_ovmpin)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_ov_body)(p, in, out, 1);
}

static void __attribute__((unused)) SFX(exec_spm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_sp_body)(p, in, out, 0, 0);
}

/* "sp dy": the cross-volume x-block pipeline with the ymm deint (panel_r9) */
static void __attribute__((unused)) SFX(exec_spmdy)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_sp_body)(p, in, out, 0, 1);
}

static void __attribute__((unused)) SFX(exec_dzm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_dz_body)(p, in, out, 0, 0);
}

static void __attribute__((unused)) SFX(exec_dzmpin)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_dz_body)(p, in, out, 1, 0);
}

static void __attribute__((unused)) SFX(exec_dzspm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_dz_body)(p, in, out, 0, 1);
}
#endif

#undef NXB
#undef NLB
#undef KPIN
#undef DT17Y
#undef DT17P
#undef DT17
#undef TP17P
#undef TP17
#undef VS
#undef VL
#undef DLO
#undef DLE
#undef IHI
#undef ILO
#undef vl
#undef vd
#undef PS
#undef TR

#endif /* L17R_TEMPLATE_PASS */
