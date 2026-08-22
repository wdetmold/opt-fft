/* Carried over from the SINGLE-THREAD competition, where this file finished as
 * written below (phase-1 history: ../../geom/strategies/L6_unrolled.md).
 *
 * MULTICORE (mt_r1)
 * -----------------
 * The per-volume kernels below are untouched -- same arithmetic, same
 * operation count, output bit-identical to the serial execute at every team
 * size (each volume's DAG is unchanged; only which thread runs it changes).
 * What was added:
 *
 *  1. Batch-parallel decomposition: thread t owns the CONTIGUOUS volume block
 *     [B*t/T, B*(t+1)/T) and runs the chosen kernel over it with its OWN
 *     scratch (t1/t2 in a per-thread arena, allocated and FIRST-TOUCHED by
 *     the owning thread inside a full-width parallel region in
 *     fft3d_create(), which also spins up the OpenMP pool -- execute() never
 *     creates a thread).  No synchronisation inside the region at all; the
 *     only barrier is the implicit join.  Volumes are 3456 B = 54 whole
 *     cache lines, so chunk boundaries can never false-share the caller's
 *     buffers; the per-thread arenas are separate allocations.
 *  2. B=1 ships SERIAL (nthr=1: execute() has no parallel region at all, so
 *     the phase-1 single-core path and its node-tuned pick machinery are
 *     preserved exactly).  One 6^3 volume is 3.4 KB and ~0.21 us of work;
 *     L13_direct's mt_r1 record measured a GOMP fork + one barrier at +71%
 *     on a volume with 10x more work, so intra-volume splitting was not
 *     rebuilt here (see strategies/L6_unrolled.md).
 *  3. NT-store fused kernels join the race (fused_nt, fused_nt_pf,
 *     fused_zp_nt_pf, fused_nt_pfnta).  Phase 1 rejected NT stores at every
 *     batch size SINGLE-core (0-for-4 rounds: one core is concurrency-bound,
 *     not bandwidth-bound).  32 threads at B=65536 are DRAM-bound, where the
 *     write-allocate RFO is a third of all traffic -- exactly the regime NT
 *     exists for.  The threaded race decides per batch size (at B=4096 the
 *     two sockets' combined 44 MiB L3 holds the 27 MiB working set and NT
 *     should lose; at B=65536, 432 MiB, it should win).
 *  4. The plan-time tournament runs THREADED when batch > 1 (the winner at
 *     32 threads is not the winner at 1: that is the whole NT question),
 *     followed by a team-size race over {1,2,4,8,16,24,32} with the chosen
 *     kernel; T=16 = one socket under OMP_PROC_BIND=close is the NUMA
 *     question (the driver first-touches in/out single-threaded, so one
 *     socket owns them and the far 16 threads pay UPI).  The team curve is
 *     published as tm=... in the description on every leaderboard line.
 *     L6_FORCE still forces the kernel; L6_FORCE_T=<n> forces the team size
 *     (both for node A/Bs, reported with a trailing !).
 *  5. PRUNED: the r11 abL DRAM codelet A/B and its probe-only VD63/fused3
 *     kernel (its question was answered on the node: abL f524.0,f3529.6 --
 *     VD6 wins the DRAM regime; the r9 f3d numbers killed it in cache) and
 *     the B=1-only ab1/kclk probes now run only on the serial (batch==1)
 *     path.  Dead code is not free in this file (r5: +3.5% B=1 from unused
 *     zmm bodies), and 250 ms + 113 MiB of setup probe whose question is
 *     closed is dead weight.
 */
/* L6_unrolled -- forward complex-double 3D DFT of a 6x6x6 cube, batched.
 *
 * TECHNIQUE
 * ---------
 * A single fully-unrolled straight-line 6-point codelet, applied along all three
 * axes of the cube with no loops over the codelet body, no runtime twiddle table,
 * and no data-dependent branches.  The codelet is Good-Thomas / PFA 2x3, so it
 * carries *no* twiddle factors at all:
 *
 *     6 = 2*3, gcd(2,3)=1.  With  j = (3*j1 + 2*j2) mod 6  and
 *     k = (3*k1 + 4*k2) mod 6  the kernel factors exactly as
 *     w6^(jk) = (-1)^(j1 k1) * w3^(j2 k2),   w3 = exp(-2 pi i/3),
 *     i.e. DFT6 = DFT2 (x) DFT3 with a pure index permutation and nothing else.
 *
 * The only irrational constants in the whole transform are 1/2 and sqrt(3)/2,
 * both written literally.  Multiplication by -i*sqrt(3)/2 in the 3-point module
 * is done as one re/im swap (vpermilpd) followed by an FMA against the fixed
 * vector (+s,-s,+s,-s), so no complex multiply routine ever appears.
 *
 * OPERATION COUNT (per 6^3 volume)
 * --------------------------------
 *   DFT6 (PFA 2x3):  3*DFT2 + 2*DFT3
 *       3*DFT2 = 12 real adds
 *       1*DFT3 = 6 real adds + 1 FMA-pair(m) + 4 FMA  ->  6 arithmetic
 *                instructions + 1 shuffle per SIMD vector, 18 real flops
 *       DFT6   = 18 arithmetic instructions + 2 shuffles per vector,
 *                48 real flops per 6-point line  (= the provably optimal
 *                Good-Thomas count; FFTW's n1_6 reaches the same 48/36)
 *   volume = 3 axes * 36 lines = 108 line-DFTs
 *          = 5184 real flops and 3888 real arithmetic instructions
 *            (108 * 36) in scalar-equivalent terms
 *   SIMD: each __m256d holds 2 complex, so 54 vector codelet instances per
 *   volume = 972 vector arithmetic uops + 108 in-codelet shuffles.  On the
 *   Skylake/Cascade-Lake port scheme (FP add and FMA both on ports 0 and 1)
 *   that is a hard floor of 486 cycles per 6^3 volume for the whole 3D
 *   transform; the rest of the kernel (216 loads, 216 stores, 324 port-5
 *   shuffles in the y+z-fused variant) fits underneath it.
 *
 * LAYOUT / SIMD
 * -------------
 * Interleaved complex throughout (the driver's own layout, so no repack pass).
 * A __m256d = 2 adjacent complex = 2 adjacent z (or 2 adjacent (y,z)) values.
 *   x-pass: lanes are 2 adjacent (y,z) positions, axis stride 36 complex.
 *   y-pass: lanes are 2 adjacent z, axis stride 6 complex.
 *   z-pass: the axis is contiguous, so 2 whole z-pencils are brought into
 *           "z-major" form with 6 vperm2f128 and pushed back with 6 more; the
 *           12 outputs of the pair land as 6 consecutive 32-byte stores that
 *           cover exactly 3 whole 64-byte cache lines (=> clean NT stores).
 * Every load and store in the transform is 32-byte aligned by construction
 * (volume = 3456 B = 54*64 B, plane = 576 B, pencil = 96 B).
 *
 * ASSUMPTIONS
 * -----------
 *  * L == 6 only.
 *  * in/out are 64-byte aligned, as fft3d_api.h guarantees, and distinct.
 *  * double _Complex is a pair of adjacent doubles (C99 guarantees this).
 *  * 10 kernel variants are RACED AGAINST EACH OTHER AND VALIDATED against a
 *    scalar reference inside fft3d_create(); a variant that disagrees with the
 *    reference by more than 1e-11 relative is disqualified and can never be
 *    selected.  This is what makes an untestable-locally code path safe.
 *    The grid was pruned hard in panel_r6 on four rounds of node pick data:
 *    kept {3pass,fused} x {plain, pfT0, pfT0+W} (the only shapes the node has
 *    ever selected), the two split-z "_s" shapes (wallaby's B=1 winners, kept
 *    for dev A/Bs), one NT representative (3pass_nt_pf, wallaby DRAM winner;
 *    NT is 0-for-4 rounds on the node), and one mixed-width AVX-512 kernel
 *    (z2s, kept ONLY as the L6_FORCE perf-counter A/B target the r5 VERDICT
 *    asks for -- clk512=2.89 vs clk256>=2.89 killed zmm at L=6, 0 picks in
 *    12 invocations).
 *    L6_FORCE=<name> (env or -DL6_FORCE_DEFAULT) bypasses the race for node
 *    A/B tests; a forced pick is reported as variant=<name>! in the
 *    description.  The description also carries a five-point clock ladder,
 *    clk256=<sparse,mid,sat> clk512=<sparse,sat> sustained GHz (panel_r6:
 *    the saturating probes adopt L17_winograd's design; 256-bit is issued
 *    before 512-bit so licence dwell cannot leak backwards).  sat256 is the
 *    number the r5 VERDICT asks for: it decides whether the node's dense ymm
 *    kernels run at 3.89 or 2.89 GHz, i.e. whether B=1 has ~366 or ~147
 *    unexplained cycles.
 *  * panel_r6 fix: create() used to END with a 512-bit probe, leaving the
 *    core in the AVX-512 licence; at B=1 the driver's whole sample set is
 *    ~0.5 ms and can complete inside the licence-recovery window, which is
 *    the leading suspect for r5's B=1 regression (0.219 -> 0.227 typical,
 *    identical pick).  create() now ends with a ~20 ms 256-bit spin so the
 *    driver never times inside the 512 licence.
 *  * panel_r7: the brief's corrected turbo table (Intel 338848-028US: AVX2
 *    and AVX-512 licences are BOTH 2.9 GHz at 1-8 active cores on the Gold
 *    5218) reopens the zmm question r5 closed.  Two new mixed-width shapes
 *    join the race, ordered after every ymm incumbent so they must win by
 *    the 2.5% margin: zxf (zmm x-pass + the node-proven ymm fused y+z) and
 *    zff (zmm x-pass + fully fused zmm/ymm y+z per plane: 9 aligned zmm
 *    plane loads + 3 valignq -- no split loads, unlike r5's z2s -- and no
 *    t2 round trip; ~1296 uops/volume vs fused's 1728).  The tournament
 *    also gains a per-candidate licence warm-up (LITERATURE 08 s4.3): each
 *    candidate runs itself untimed ~0.7 ms before every timed trial, so a
 *    ymm candidate is never timed inside a zmm predecessor's licence tail.
 *  * panel_r9: (a) the codelet macros are parameterized (VD6 vs VD63) and
 *    three raced fused3 twins ship: VD63 is the SAME PFA 2x3 DFT factored
 *    DFT3-first/DFT2-last (L6_pfa's ordering; the r8 VERDICT's standing
 *    hypothesis for the B=64 gap), token-identical kernels otherwise --
 *    the controlled test of the last named mechanism at this geometry.
 *    Not bit-identical to VD6 (different association, ~1e-16); the plan
 *    gate and numpy bound it.  (b) an in-plan B=1 discriminator (adopted
 *    from L36_pfa's r8 in-plan probe pattern, per the r8 VERDICT's L=6
 *    instruction): create() times fused / fused3 / zff at nvol=1,
 *    licence-fair, and reports min ns/vol as ab1=... in the description,
 *    plus the fused3-vs-fused race delta as f3d=...  perf_event_open is
 *    closed cluster-wide (perf_event_paranoid=4), so timing is the only
 *    in-plan discriminator available.
 *    And the description now carries kclk=<GHz> -- L6_pfa's r6 probe design
 *    (adopted, attributed): dwell ~2 ms in the CHOSEN kernel, then time a
 *    ~150 us sparse ymm chain that reads the licence the kernel itself
 *    established; median of 9.  That is the number that converts this
 *    entry's node times into cycles.
 *  * panel_r8: the r7 node data closed the width question (zxf/zff/z2s and
 *    L6_pfa's fused_zx: 0 picks in 8 cells x 3 processes, with the race
 *    licence-fair and kclk measured at 2.89 both widths), so the zmm shapes
 *    no longer RACE: zff and z2s stay compiled, validated and L6_FORCE-able
 *    only, because the r7 VERDICT's outstanding perf-counter A/B names them.
 *    zxf/zxf_pf/zff_pf are deleted.  Two changes adopted from L6_pfa
 *    (attributed): (a) `restrict` on every kernel signature -- the one
 *    systematic codegen difference between our same-shape kernels; without
 *    it gcc must order every t1-plane load after the preceding out-stores
 *    and every next-group input load after the scratch stores, i.e. the
 *    exact pass-boundary joints the r7 VERDICT names as the remaining B=1
 *    suspect; their fused_pf_xa (identical shape, restrict-qualified) beat
 *    my fused_pf by ~3.7% at B=4096 in 2 of 3 r7 processes; (b) create()
 *    now ENDS by dwelling ~3 ms in the CHOSEN kernel, so the driver is
 *    handed a core in the scored kernel's own licence/clock steady state,
 *    never a probe's or a generic spin's.
 *  * panel_r10: the r9 node data answered every question the r9 machinery
 *    carried, so the file is cut to what is still open.  (a) DELETED: the
 *    fused3/VD63 twins (f3d read +3.3..+6.2% on the node -- the radix-2-
 *    first VD6 is confirmed the winning association, both L=6 entries now
 *    run it), the whole AVX-512 section incl. zff/z2s (ab1 zf/f = 1.14-1.18
 *    on the node closed the uop theory with a published number, and the r9
 *    VERDICT formally withdrew the perf-counter ask that kept them
 *    compiled), and the 5-probe clock ladder (the clock consensus is
 *    settled panel-wide: 3.89 non-AVX / 2.89 licence; kclk stays as the
 *    one-number regression check).  My own r5 lesson motivates the prune:
 *    ~500 lines of never-executed zmm code once moved B=1 by +3.5% at an
 *    identical pick -- dead weight is not free in this file.  (b) ADOPTED
 *    FROM L6_pfa: the zp-outer/y-inner x-pass group order (their PASS_X,
 *    in their file since round 1) as raced twins fused_zp{,_pf,_pfw}.
 *    Their r9 B=1 winner fused_d2 = my VD6 graph + THEIR zp-outer x order;
 *    the ascending-x + VD6 + no-pf combination (= my fused) raced only in
 *    MY file, and mine reads 0.2174-0.2305 where theirs reads 0.2068x3 --
 *    the x group order is the one structural difference left between the
 *    two kernels, so it races here and ab1 now publishes the f-vs-fx A/B
 *    (fused vs fused_zp at nvol=1) on every leaderboard line.  The twins
 *    are output-bit-identical to their parents (same per-line arithmetic,
 *    different order over independent lines), so they carry a reduced 1.0%
 *    takeover margin where genuinely-new shapes keep 2.5%.
 *  * panel_r11: (a) the zp twins take INCUMBENCY over their ascending
 *    parents (my r10 branch (iii), prescribed by the r10 VERDICT; the exact
 *    move L6_pfa made for their d2 twins in r10): the node picked fused_zp
 *    2/3 at B=1 with ab1 fx-f = -0.6/-3.1/-1.2%, and a node-measured
 *    sub-margin winner in a trailing slot is invisible to the tuner by
 *    construction.  Pure table reorder; the parents stay as 1.0%-margin
 *    challengers (bit-identical class, so a flip either way is harmless).
 *    (b) the r10 VERDICT's single L=6 item: an in-plan A/B at nvol >> L3
 *    that settles whether codelet association order matters in the
 *    DRAM-bound regime (L6_pfa's B=32768 dropped 3.3% on the d2 codelet
 *    flip, against both entries' models).  The VD63 (DFT3-first) codelet
 *    returns as ONE probe-only kernel, fused3_pfw -- NOT in cand[], never
 *    pickable (L36_mixedradix's r10 one-bit-class tuner rule: cross-class
 *    comparisons ride the description string, never the pick), validated
 *    against the scalar reference, then timed against fused_pfw over
 *    nvol=16384 volumes (113 MiB in+out: unambiguous DRAM on both
 *    machines, same size the race cap uses).  Result published as
 *    abL=f<ns>,f3<ns> per volume on every leaderboard line.
 *
 * MULTICORE round mt_r2 -- three adoptions from L6_pfa's mt_r1 (attributed),
 * which beat this entry in every node cell (0.210/0.0094/0.0395 us vs my
 * 0.221/0.0116/0.0721):
 *  1. THE MULTICORE RACE-ARENA CAP RISES 16384 -> 65536 VOLUMES (452 MiB).
 *     L6_pfa's central mt_r1 lesson: once threaded, the aggregate cache
 *     (node: 2x22 MiB L3 + 32x1 MiB L2 = ~76 MiB) -- not the largest single
 *     L3 -- sets the race-arena floor.  My 16384-volume (113 MiB) arena was
 *     near-cache-resident at 32 threads, so at B=65536 the kernel race
 *     mis-ranked the NT relatives and the team race read T=16 ~= T=32 (75
 *     vs 78 ns); the smallest-T rule then shipped T=16 = one socket's ~96
 *     GB/s, 72.1 ns/vol scored.  L6_pfa raced the same decision at 65536
 *     volumes, picked fused NT+pf at T=32, and scored 39.5 ns (~175 GB/s,
 *     both sockets).  Their winner is structurally MY fused_nt_pf
 *     (ascending x + T0 pf + the VD6 graph they adopted from me + NT fused
 *     stores) -- the 1.83x gap was race regime, not kernel.  The serial arm
 *     keeps the 16384 cap.
 *  2. SPIN-WAIT PTHREAD DISPATCH POOL, raced against OMP dispatch on the
 *     chosen (kernel, T): master release-stores an epoch, workers spin with
 *     pause and run their chunk, master scans padded done flags (~1-2 us
 *     round trip vs GOMP's 5-18 us fork/join; fork=13.5-18.3 us in their
 *     node descriptions).  Workers are pinned to the affinity masks the OMP
 *     threads report inside the pool-warm region, so thread t sits on the
 *     same core -- and socket as its first-touched scratch -- under either
 *     dispatch.  Sequenced race (OMP first with no pool alive, then the
 *     pool after ~3 ms so GOMP's spinners have slept); the pool is torn
 *     down unless it wins by >2%.  Their node evidence at B=4096: omp
 *     0.0141 -> pool 0.0132 us/vol.  The chunk split is IDENTICAL under
 *     both dispatches (bit-identical output either way).
 *  3. SP2 SOFTWARE-PIPELINED FUSED STAGE (their panel_r5 shape, node-picked
 *     for their B=4096 cell in mt_r1 as fused_sp2_pf_xa T=24): plane
 *     registers double-buffered P/Q, the next plane's 18 loads + 3
 *     y-codelets interleaved by thirds into the current plane's z-chunks,
 *     plane-pair loop kept rolled (DSB-resident; CLX ROB is 224 uops, a
 *     ~195-uop plane body otherwise serialises the z-tail).  Raced as
 *     fused_sp2 / fused_sp2_pf with my VD6 codelet at the full 2.5%
 *     new-shape margin.  Caveat carried from their record: their sp2 won
 *     with the radix-3-first DFT6V, and their d2 twin (= my VD6 graph)
 *     "composed badly with the interleave" on wallaby (-3.3%) -- so these
 *     twins may lose; the race decides, not me.
 *  Also new: a threaded end-to-end gate (their point 7): the final
 *  (kernel, T, dispatch) config must reproduce the scalar reference on an
 *  odd 61-volume batch (uneven chunks, idle threads exercised) at rel L2
 *  <= 1e-13, or the plan falls back to the gate-proven serial path.
 *  The description gains disp=omp|pool and od=<omp>,<pool>ns.
 *
 * MULTICORE round mt_r3 -- race what you ship (L6_pfa's mt_r2 payload,
 * adopted and attributed) plus the mt_r2 VERDICT's two L=6 asks:
 *  1. THE TOURNAMENT AND TEAM RACE RUN THROUGH THE SPIN POOL (rd=pool):
 *     the pool is created BEFORE the race; every T>1 cell dispatches via
 *     l6_run_pool (idle workers spinning, exactly as a scored pool run),
 *     T=1 cells stay direct calls.  mt_r2 raced through the OMP fork
 *     (13-18 us on the node) while the scored B=4096 run shipped the
 *     ~1-2 us pool -- ranking configurations under a dispatch the winner
 *     never uses.  The dispatch race is re-sequenced accordingly: pool
 *     timed first (warm from the tournament), DESTROYED (cores genuinely
 *     idle; GOMP's team futex-slept through the tournament), OMP timed,
 *     pool recreated only if it won by >2%.
 *  2. Pool join fixes (both L6_pfa mt_r2, attributed): done-flag elements
 *     were 56 B (pad[48] + aligned(64) on the ARRAY does not align
 *     elements), so adjacent workers' flags shared cache lines -- now 64 B
 *     each; and the master prefetches all done-flag lines before the join
 *     scan so the cross-core misses overlap in the fill buffers.
 *  3. WIDE-TEAM INCUMBENCY AT STREAMING CELLS (the VERDICT s6 mechanical
 *     fix for the panel-wide T=32->T=16 pick lottery): when the real
 *     working set exceeds 128 MiB (beyond any aggregate cache here), the
 *     widest T is the team-race incumbent and a narrower team must beat
 *     it by >2%; cache-resident cells keep smallest-T-within-2%.
 *  4. fr= PLACEMENT INSTRUMENT (VERDICT s6 ask; L8_fusedaxes's mt_r2
 *     governor scan, attributed): a read-only get_mempolicy sample of the
 *     caller's in/out page homes on the 1st and 49th threaded execute,
 *     published as fr=<pct0>[/<pct48>],nb=<autonuma> -- the 49th-call
 *     rescan is the "fr under a wide team during the timed loop" reading
 *     the VERDICT says nobody has taken yet.
 *  5. PRUNED (my r10 rule -- zero node picks closes a question): the sp2
 *     twins (0 picks in mt_r2, lost every wallaby race), fused_nt,
 *     fused_zp_nt_pf, fused_nt_pfnta (0 picks in 2 rounds; pfnta worst of
 *     all 14 on wallaby).  NT keeps 3pass_nt_pf (the node's B=65536
 *     winner) and fused_nt_pf (the controlled 3pass-vs-fused NT twin).
 */

#define _GNU_SOURCE 1   /* pthread_getaffinity_np / cpu_set_t for the pool */

#include "fft3d_api.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>   /* SYS_get_mempolicy / SYS_getcpu: fr= instrument */
#endif

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define L6_HAVE_AVX2 1
#endif

#ifdef _OPENMP
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#endif

#define L6_L    6
#define L6_VOL  216            /* complex per volume */
#define L6_VD   432            /* doubles  per volume */
#define L6_MAXT 32             /* the harness gives 32 cores; never take more */
/* Takeover margins are per-candidate since panel_r10 (see the cand[] table):
 * 2.5% for genuinely different shapes (raised from 1.5% in panel_r4 after an
 * L3-marginal mis-pick at B=4096), 1.0% for the fused_zp twins, whose output
 * is bit-identical to their parents so a mis-pick is bounded and harmless. */

static const double L6_S3 = 0.86602540378443864676372317075294; /* sqrt(3)/2 */

/* ------------------------------------------------------------------ *
 * Scalar reference: the same PFA 2x3 codelet, three strided passes.
 * Used as the plan-time correctness witness and as the portable path.
 * ------------------------------------------------------------------ */

#define SD6(A0,A1,A2,A3,A4,A5, R0,R1,R2,R3,R4,R5)                       \
    do {                                                                \
        double p0r=(A0##r)+(A3##r), p0i=(A0##i)+(A3##i);                 \
        double q0r=(A0##r)-(A3##r), q0i=(A0##i)-(A3##i);                 \
        double p1r=(A2##r)+(A5##r), p1i=(A2##i)+(A5##i);                 \
        double q1r=(A2##r)-(A5##r), q1i=(A2##i)-(A5##i);                 \
        double p2r=(A4##r)+(A1##r), p2i=(A4##i)+(A1##i);                 \
        double q2r=(A4##r)-(A1##r), q2i=(A4##i)-(A1##i);                 \
        double ar=p1r+p2r, ai=p1i+p2i, br=p1r-p2r, bi=p1i-p2i;          \
        double cr=q1r+q2r, ci=q1i+q2i, dr=q1r-q2r, di=q1i-q2i;          \
        double mr=p0r-0.5*ar, mi=p0i-0.5*ai;                            \
        double nr=q0r-0.5*cr, ni=q0i-0.5*ci;                            \
        double jbr= L6_S3*bi, jbi=-L6_S3*br;                            \
        double jdr= L6_S3*di, jdi=-L6_S3*dr;                            \
        (R0##r)=p0r+ar; (R0##i)=p0i+ai;                                 \
        (R3##r)=q0r+cr; (R3##i)=q0i+ci;                                 \
        (R4##r)=mr+jbr; (R4##i)=mi+jbi;                                 \
        (R2##r)=mr-jbr; (R2##i)=mi-jbi;                                 \
        (R1##r)=nr+jdr; (R1##i)=ni+jdi;                                 \
        (R5##r)=nr-jdr; (R5##i)=ni-jdi;                                 \
    } while (0)

static void l6_line_scalar(const double *s, long ss, double *d, long ds)
{
    /* ss, ds are strides in COMPLEX units */
    double x0r=s[0],        x0i=s[1];
    double x1r=s[2*ss],     x1i=s[2*ss+1];
    double x2r=s[4*ss],     x2i=s[4*ss+1];
    double x3r=s[6*ss],     x3i=s[6*ss+1];
    double x4r=s[8*ss],     x4i=s[8*ss+1];
    double x5r=s[10*ss],    x5i=s[10*ss+1];
    double y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i;
    SD6(x0,x1,x2,x3,x4,x5, y0,y1,y2,y3,y4,y5);
    d[0]=y0r;       d[1]=y0i;
    d[2*ds]=y1r;    d[2*ds+1]=y1i;
    d[4*ds]=y2r;    d[4*ds+1]=y2i;
    d[6*ds]=y3r;    d[6*ds+1]=y3i;
    d[8*ds]=y4r;    d[8*ds+1]=y4i;
    d[10*ds]=y5r;   d[10*ds+1]=y5i;
}

static void l6_run_scalar(double *restrict t1, double *restrict t2,
                          const double *restrict in, double *restrict out,
                          long nvol)
{
    for (long b = 0; b < nvol; ++b) {
        const double *ip = in  + b * (long)L6_VD;
        double       *op = out + b * (long)L6_VD;
        for (int p = 0; p < 36; ++p)                  /* x: stride 36 complex */
            l6_line_scalar(ip + 2*p, 36, t1 + 2*p, 36);
        for (int x = 0; x < 6; ++x)                   /* y: stride 6 complex  */
            for (int z = 0; z < 6; ++z)
                l6_line_scalar(t1 + 2*(36*x + z), 6, t2 + 2*(36*x + z), 6);
        for (int x = 0; x < 6; ++x)                   /* z: stride 1 complex  */
            for (int y = 0; y < 6; ++y)
                l6_line_scalar(t2 + 2*(36*x + 6*y), 1, op + 2*(36*x + 6*y), 1);
    }
}

/* ------------------------------------------------------------------ *
 * AVX2 / FMA kernels.  __m256d = 2 complex.
 * ------------------------------------------------------------------ */
#ifdef L6_HAVE_AVX2

/* One straight-line PFA 2x3 six-point codelet on 2 complex lanes.
 * 18 arithmetic instructions + 2 shuffles.  Safe to use in place. */
#define VD6(i0,i1,i2,i3,i4,i5, o0,o1,o2,o3,o4,o5)                       \
    do {                                                                \
        __m256d _p0 = _mm256_add_pd(i0,i3), _q0 = _mm256_sub_pd(i0,i3); \
        __m256d _p1 = _mm256_add_pd(i2,i5), _q1 = _mm256_sub_pd(i2,i5); \
        __m256d _p2 = _mm256_add_pd(i4,i1), _q2 = _mm256_sub_pd(i4,i1); \
        __m256d _a  = _mm256_add_pd(_p1,_p2), _b = _mm256_sub_pd(_p1,_p2); \
        __m256d _c  = _mm256_add_pd(_q1,_q2), _d = _mm256_sub_pd(_q1,_q2); \
        __m256d _m  = _mm256_fnmadd_pd(vhalf,_a,_p0);                   \
        __m256d _n  = _mm256_fnmadd_pd(vhalf,_c,_q0);                   \
        __m256d _bs = _mm256_permute_pd(_b,0x5);                        \
        __m256d _ds = _mm256_permute_pd(_d,0x5);                        \
        __m256d _s0 = _mm256_add_pd(_p0,_a);                            \
        __m256d _s3 = _mm256_add_pd(_q0,_c);                            \
        __m256d _s4 = _mm256_fmadd_pd (vk,_bs,_m);                      \
        __m256d _s2 = _mm256_fnmadd_pd(vk,_bs,_m);                      \
        __m256d _s1 = _mm256_fmadd_pd (vk,_ds,_n);                      \
        __m256d _s5 = _mm256_fnmadd_pd(vk,_ds,_n);                      \
        o0=_s0; o1=_s1; o2=_s2; o3=_s3; o4=_s4; o5=_s5;                 \
    } while (0)

#define VSET __m256d vhalf = _mm256_set1_pd(0.5);                        \
             __m256d vk    = _mm256_setr_pd(L6_S3,-L6_S3,L6_S3,-L6_S3)

/* panel_r10 prune: the VD63 (DFT3-first) codelet twin was DELETED (r9 node
 * data: f3d = +3.3..+6.2%, VD6 wins in cache).  panel_r11 brought it back
 * probe-only for the DRAM codelet A/B, which the node then also answered
 * (abL = f524.0,f3529.6 ns: VD6 wins the DRAM regime too), so mt_r1 deletes
 * VD63, the fused3_pfw probe kernel and the l6_abL machinery for good.
 * Derivations and numbers live in strategies/L6_unrolled.md (geom r9/r11). */

/* Software prefetch hooks for the x-pass loop (adopted from L6_pfa's v8,
 * which won the large-batch cases in panel_r1 with exactly this: touch the
 * NEXT volume's input, 3 cache lines per x-pass group = all 54 lines of a
 * 3456-B volume, giving one volume (~800 cycles) of lead time).  Prefetch
 * never faults, so running past the end of the batch is safe.
 *
 * New in panel_r3:
 *  - T1 hint on the input (adopted from L6_pfa's panel_r3 pf=3 column, which
 *    beat T0 by 7% in wallaby's DRAM regime): fills L2 without displacing the
 *    L1-resident scratch.
 *  - write-intent prefetch (prefetchw) of the NEXT volume's OUTPUT lines.
 *    The node rejected NT stores at every batch size (panel_r2, both L=6
 *    entries), so the scored kernels pay a write-allocate RFO per output
 *    line; prefetchw issues that RFO one volume early instead of letting it
 *    stall the store buffer.  __builtin_prefetch(p,1,3) compiles everywhere
 *    and emits prefetchw when the target has PRFCHW (Cascade Lake and
 *    Sapphire Rapids both do). */
#define L6_PF_NONE(SRC,OUT,g)  do { } while (0)
#define L6_PF_AT(SRC,g,DIST,HINT)                                       \
    do {                                                                \
        const char *_pf = (const char *)((SRC) + (DIST)*(long)L6_VD)    \
                          + 192*(g);                                    \
        _mm_prefetch(_pf,      HINT);                                   \
        _mm_prefetch(_pf + 64, HINT);                                   \
        _mm_prefetch(_pf + 128,HINT);                                   \
    } while (0)
#define L6_PF_W_AT(OUT,g,DIST)                                          \
    do {                                                                \
        const char *_pw = (const char *)((OUT) + (DIST)*(long)L6_VD)    \
                          + 192*(g);                                    \
        __builtin_prefetch(_pw,       1, 3);                            \
        __builtin_prefetch(_pw + 64,  1, 3);                            \
        __builtin_prefetch(_pw + 128, 1, 3);                            \
    } while (0)
#define L6_PF_T0_1(SRC,OUT,g)  L6_PF_AT(SRC,g,1,_MM_HINT_T0)
#define L6_PF_T0W_1(SRC,OUT,g)                                          \
    do { L6_PF_AT(SRC,g,1,_MM_HINT_T0); L6_PF_W_AT(OUT,g,1); } while (0)
/* pruned in panel_r6 (never picked on the node in 4 rounds of stable pick
 * reporting): distance-2 hooks, T1 hooks, W-only.  mt_r3 prune: the NTA
 * input-prefetch hook (worst of all 14 candidates on wallaby, +135%; 0
 * node picks in 2 rounds).  See strategies/ r6, mt_r1, mt_r3. */

/* x-pass: in -> t1.  lanes = 2 adjacent (y,z); axis stride 72 doubles.
 * PF(SRC,OUT,g) is a prefetch hook run once per group; OUT is the volume's
 * final output pointer, only used by the prefetchw hooks.  CD is the
 * codelet macro.  Groups walk offsets 4g, g = 0..17: loads and stores in
 * strictly ascending 32B steps. */
#define L6_PASS_X(SRC,DST,OUT,PF,CD)                                    \
    do {                                                                \
        for (int g = 0; g < 18; ++g) {                                  \
            const double *s = (SRC) + 4*g;                              \
            double *d = (DST) + 4*g;                                    \
            PF(SRC,OUT,g);                                              \
            __m256d v0=_mm256_load_pd(s+  0), v1=_mm256_load_pd(s+ 72); \
            __m256d v2=_mm256_load_pd(s+144), v3=_mm256_load_pd(s+216); \
            __m256d v4=_mm256_load_pd(s+288), v5=_mm256_load_pd(s+360); \
            CD(v0,v1,v2,v3,v4,v5, v0,v1,v2,v3,v4,v5);                   \
            _mm256_store_pd(d+  0,v0); _mm256_store_pd(d+ 72,v1);       \
            _mm256_store_pd(d+144,v2); _mm256_store_pd(d+216,v3);       \
            _mm256_store_pd(d+288,v4); _mm256_store_pd(d+360,v5);       \
        }                                                               \
    } while (0)

/* x-pass, zp-outer/y-inner group order -- ADOPTED FROM L6_pfa's PASS_X
 * (in their file since round 1; their r9 node B=1 winner fused_d2 = the
 * VD6 graph behind exactly this group order).  Same 18 groups, same
 * per-group body, walked as g = (y,zp) with zp outer: offsets 12y + 4zp
 * doubles, i.e. 96-byte steps within a zp block.  Output bit-identical to
 * L6_PASS_X (the groups are independent; only their order changes).  The
 * prefetch index 6*zp+y still walks the next volume's 54 lines in strictly
 * ascending address order, so the streamer sees the same stream. */
#define L6_PASS_X_ZP(SRC,DST,OUT,PF,CD)                                 \
    do {                                                                \
        for (int zp = 0; zp < 3; ++zp)                                  \
            for (int y = 0; y < 6; ++y) {                               \
                const double *s = (SRC) + 12*y + 4*zp;                  \
                double *d = (DST) + 12*y + 4*zp;                        \
                PF(SRC,OUT,6*zp + y);                                   \
                __m256d v0=_mm256_load_pd(s+  0), v1=_mm256_load_pd(s+ 72); \
                __m256d v2=_mm256_load_pd(s+144), v3=_mm256_load_pd(s+216); \
                __m256d v4=_mm256_load_pd(s+288), v5=_mm256_load_pd(s+360); \
                CD(v0,v1,v2,v3,v4,v5, v0,v1,v2,v3,v4,v5);               \
                _mm256_store_pd(d+  0,v0); _mm256_store_pd(d+ 72,v1);   \
                _mm256_store_pd(d+144,v2); _mm256_store_pd(d+216,v3);   \
                _mm256_store_pd(d+288,v4); _mm256_store_pd(d+360,v5);   \
            }                                                           \
    } while (0)

/* y-pass: t1 -> t2.  lanes = 2 adjacent z; axis stride 12 doubles. */
#define L6_PASS_Y(SRC,DST)                                              \
    do {                                                                \
        for (int x = 0; x < 6; ++x) {                                   \
            for (int c = 0; c < 3; ++c) {                               \
                const double *s = (SRC) + 72*x + 4*c;                   \
                double *d = (DST) + 72*x + 4*c;                         \
                __m256d v0=_mm256_load_pd(s+ 0), v1=_mm256_load_pd(s+12);\
                __m256d v2=_mm256_load_pd(s+24), v3=_mm256_load_pd(s+36);\
                __m256d v4=_mm256_load_pd(s+48), v5=_mm256_load_pd(s+60);\
                VD6(v0,v1,v2,v3,v4,v5, v0,v1,v2,v3,v4,v5);              \
                _mm256_store_pd(d+ 0,v0); _mm256_store_pd(d+12,v1);     \
                _mm256_store_pd(d+24,v2); _mm256_store_pd(d+36,v3);     \
                _mm256_store_pd(d+48,v4); _mm256_store_pd(d+60,v5);     \
            }                                                           \
        }                                                               \
    } while (0)

/* z-pass: SRC -> DST, two z-pencils at a time, in-register transposed.
 * ST is the store intrinsic (_mm256_store_pd or _mm256_stream_pd). */
#define L6_PASS_Z(SRC,DST,ST)                                           \
    do {                                                                \
        for (int x = 0; x < 6; ++x) {                                   \
            for (int yp = 0; yp < 3; ++yp) {                            \
                const double *s = (SRC) + 72*x + 24*yp;                 \
                double *d = (DST) + 72*x + 24*yp;                       \
                __m256d A0=_mm256_load_pd(s+ 0), A1=_mm256_load_pd(s+ 4);\
                __m256d A2=_mm256_load_pd(s+ 8), B0=_mm256_load_pd(s+12);\
                __m256d B1=_mm256_load_pd(s+16), B2=_mm256_load_pd(s+20);\
                __m256d v0=_mm256_permute2f128_pd(A0,B0,0x20);          \
                __m256d v1=_mm256_permute2f128_pd(A0,B0,0x31);          \
                __m256d v2=_mm256_permute2f128_pd(A1,B1,0x20);          \
                __m256d v3=_mm256_permute2f128_pd(A1,B1,0x31);          \
                __m256d v4=_mm256_permute2f128_pd(A2,B2,0x20);          \
                __m256d v5=_mm256_permute2f128_pd(A2,B2,0x31);          \
                VD6(v0,v1,v2,v3,v4,v5, v0,v1,v2,v3,v4,v5);              \
                ST(d+ 0,_mm256_permute2f128_pd(v0,v1,0x20));            \
                ST(d+ 4,_mm256_permute2f128_pd(v2,v3,0x20));            \
                ST(d+ 8,_mm256_permute2f128_pd(v4,v5,0x20));            \
                ST(d+12,_mm256_permute2f128_pd(v0,v1,0x31));            \
                ST(d+16,_mm256_permute2f128_pd(v2,v3,0x31));            \
                ST(d+20,_mm256_permute2f128_pd(v4,v5,0x31));            \
            }                                                           \
        }                                                               \
    } while (0)

/* All kernels are entered through a function pointer, so their placement in
 * the binary is at the linker's mercy; panel_r6 pins every kernel entry to a
 * 64-byte boundary after r5's B=1 regressed 0.219->0.227 typical with an
 * IDENTICAL pick string when ~500 lines of zmm code were added to the file
 * (same disease the r5 VERDICT names at L36_mixedradix B=1: code layout). */
#define L6_KALIGN __attribute__((aligned(64)))

/* --- variant A: three separate passes ---------------------------------- */
#define L6_DEF_3PASS(NAME,ST,PF)                                        \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,    \
                 const double *restrict in, double *restrict out,       \
                 long nvol)                                             \
{                                                                       \
    VSET;                                                               \
    for (long b = 0; b < nvol; ++b) {                                   \
        const double *ip = in  + b * (long)L6_VD;                       \
        double       *op = out + b * (long)L6_VD;                       \
        L6_PASS_X(ip, t1, op, PF, VD6);                                 \
        L6_PASS_Y(t1, t2);                                              \
        L6_PASS_Z(t2, op, ST);                                          \
    }                                                                   \
}

L6_DEF_3PASS(l6_run_3pass,        _mm256_store_pd,  L6_PF_NONE)
L6_DEF_3PASS(l6_run_3pass_pf,     _mm256_store_pd,  L6_PF_T0_1)
L6_DEF_3PASS(l6_run_3pass_pfw,    _mm256_store_pd,  L6_PF_T0W_1)
L6_DEF_3PASS(l6_run_3pass_nt_pf,  _mm256_stream_pd, L6_PF_T0_1)

/* --- variant B: x-pass, then y and z fused per x-plane in registers ----
 * The 6x6 (y,z) plane is 18 __m256d; with the codelet temporaries that is
 * ~26 live vectors, which fits the 32 ymm of an AVX-512VL machine but not
 * the 16 of plain AVX2.  Raced at plan time against variant A.
 * The plane loop is a macro of its own (panel_r7) so the zmm-x `zxf`
 * kernels can reuse it token-for-token behind a different x-pass.       */
#define L6_FUSED_YZ(SRC,DST,ST,CD)                                      \
    do {                                                                \
        for (int x = 0; x < 6; ++x) {                                   \
            const double *s = (SRC) + 72*x;                             \
            double *d = (DST) + 72*x;                                   \
            __m256d P00=_mm256_load_pd(s+ 0),P01=_mm256_load_pd(s+ 4),  \
                    P02=_mm256_load_pd(s+ 8);                           \
            __m256d P10=_mm256_load_pd(s+12),P11=_mm256_load_pd(s+16),  \
                    P12=_mm256_load_pd(s+20);                           \
            __m256d P20=_mm256_load_pd(s+24),P21=_mm256_load_pd(s+28),  \
                    P22=_mm256_load_pd(s+32);                           \
            __m256d P30=_mm256_load_pd(s+36),P31=_mm256_load_pd(s+40),  \
                    P32=_mm256_load_pd(s+44);                           \
            __m256d P40=_mm256_load_pd(s+48),P41=_mm256_load_pd(s+52),  \
                    P42=_mm256_load_pd(s+56);                           \
            __m256d P50=_mm256_load_pd(s+60),P51=_mm256_load_pd(s+64),  \
                    P52=_mm256_load_pd(s+68);                           \
            CD(P00,P10,P20,P30,P40,P50, P00,P10,P20,P30,P40,P50);       \
            CD(P01,P11,P21,P31,P41,P51, P01,P11,P21,P31,P41,P51);       \
            CD(P02,P12,P22,P32,P42,P52, P02,P12,P22,P32,P42,P52);       \
            L6_ZPAIR(P00,P01,P02,P10,P11,P12, d+ 0, ST, CD);            \
            L6_ZPAIR(P20,P21,P22,P30,P31,P32, d+24, ST, CD);            \
            L6_ZPAIR(P40,P41,P42,P50,P51,P52, d+48, ST, CD);            \
        }                                                               \
    } while (0)

/* PX = L6_PASS_X (ascending) or L6_PASS_X_ZP (zp-outer, panel_r10). */
#define L6_DEF_FUSED_PX(NAME,PX,ST,PF,CD)                               \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,    \
                 const double *restrict in, double *restrict out,       \
                 long nvol)                                             \
{                                                                       \
    VSET;                                                               \
    (void)t2;                                                           \
    for (long b = 0; b < nvol; ++b) {                                   \
        const double *ip = in  + b * (long)L6_VD;                       \
        double       *op = out + b * (long)L6_VD;                       \
        PX(ip, t1, op, PF, CD);                                         \
        L6_FUSED_YZ(t1, op, ST, CD);                                    \
    }                                                                   \
}
#define L6_DEF_FUSED(NAME,ST,PF,CD) \
        L6_DEF_FUSED_PX(NAME,L6_PASS_X,ST,PF,CD)

#define L6_ZPAIR(A0,A1,A2,B0,B1,B2, D, ST, CD)                          \
    do {                                                                \
        __m256d w0=_mm256_permute2f128_pd(A0,B0,0x20);                  \
        __m256d w1=_mm256_permute2f128_pd(A0,B0,0x31);                  \
        __m256d w2=_mm256_permute2f128_pd(A1,B1,0x20);                  \
        __m256d w3=_mm256_permute2f128_pd(A1,B1,0x31);                  \
        __m256d w4=_mm256_permute2f128_pd(A2,B2,0x20);                  \
        __m256d w5=_mm256_permute2f128_pd(A2,B2,0x31);                  \
        CD(w0,w1,w2,w3,w4,w5, w0,w1,w2,w3,w4,w5);                       \
        ST((D)+ 0,_mm256_permute2f128_pd(w0,w1,0x20));                  \
        ST((D)+ 4,_mm256_permute2f128_pd(w2,w3,0x20));                  \
        ST((D)+ 8,_mm256_permute2f128_pd(w4,w5,0x20));                  \
        ST((D)+12,_mm256_permute2f128_pd(w0,w1,0x31));                  \
        ST((D)+16,_mm256_permute2f128_pd(w2,w3,0x31));                  \
        ST((D)+20,_mm256_permute2f128_pd(w4,w5,0x31));                  \
    } while (0)

L6_DEF_FUSED(l6_run_fused,        _mm256_store_pd,  L6_PF_NONE,  VD6)
L6_DEF_FUSED(l6_run_fused_pf,     _mm256_store_pd,  L6_PF_T0_1,  VD6)
L6_DEF_FUSED(l6_run_fused_pfw,    _mm256_store_pd,  L6_PF_T0W_1, VD6)

/* panel_r10: the zp-outer x-pass twins (group order adopted from L6_pfa;
 * see L6_PASS_X_ZP).  Token-identical to the three kernels above except
 * for the x-pass group order; output bit-identical to them. */
L6_DEF_FUSED_PX(l6_run_fused_zp,     L6_PASS_X_ZP, _mm256_store_pd, L6_PF_NONE,  VD6)
L6_DEF_FUSED_PX(l6_run_fused_zp_pf,  L6_PASS_X_ZP, _mm256_store_pd, L6_PF_T0_1,  VD6)
L6_DEF_FUSED_PX(l6_run_fused_zp_pfw, L6_PASS_X_ZP, _mm256_store_pd, L6_PF_T0W_1, VD6)

/* panel_r7 prune: the ymm split-store shapes 3pass_s / fused_s (and the
 * L6_ZPAIR_S macro) are deleted -- 0 picks in 12 node invocations across
 * r4/r5, SPR-only mechanism.  panel_r9 prune: the fused3/VD63 twins are
 * deleted (question answered on the node, f3d = +3.3..+6.2%). */

/* mt_r3 prune (my r10 rule: zero node picks closes a question, and dead
 * code is not free in this file -- r5: +3.5% B=1 from unused bodies):
 * the sp2 software-pipelined twins (adopted from L6_pfa in mt_r2) are
 * DELETED: 0 node picks in mt_r2 (node picked fused_zp_pf at B=4096 and
 * 3pass_nt_pf at B=65536), and they lost their wallaby race 9.0-9.1 vs
 * 8.9 ns (B=4096) and 46.0 vs 43.1 (B=65536) -- consistent with L6_pfa's
 * own caveat that the VD6 codelet composes badly with the interleave.
 * Also deleted: fused_nt, fused_zp_nt_pf, fused_nt_pfnta (and the
 * L6_PF_NTA_1 hook): 0 node picks in 2 rounds; pfnta was the worst of
 * all 14 candidates on wallaby (+135%).  The NT question stays covered
 * by the two shapes with live questions: 3pass_nt_pf (the node's B=65536
 * winner) and fused_nt_pf (the controlled 3pass-vs-fused NT twin). */
#if 0   /* --- variant C (mt_r2): software-pipelined fused y/z stage -- ADOPTED FROM
 * L6_pfa (their panel_r5 FUSED_YZ_SP2, node-picked for their B=4096 cell in
 * mt_r1).  Same arithmetic as L6_FUSED_YZ, but plane registers are double-
 * buffered (P for even x, Q for odd x) and the NEXT plane's 18 loads + 3
 * y-codelets are interleaved, by thirds, into the CURRENT plane's z-chunks:
 * on CLX (ROB 224 uops) the ~195-uop plane body otherwise means plane x+1
 * cannot enter the window until plane x has nearly retired, serialising the
 * z-tail.  The plane-pair loop is kept rolled (body above gcc's complete-
 * peel limit) so the stage stays DSB-resident.  P[3*y + zp] indexing; a
 * z-chunk yp consumes rows y = 2yp, 2yp+1 = P[6yp..6yp+5], which is exactly
 * my L6_ZPAIR's operand order, so the z-chunk reuses that macro verbatim. */
#define L6_SP2_LD6(P, base, t)                                          \
    do {                                                                \
        const double *_pl = (base) + 24 * (t);                          \
        (P)[6*(t)]   = _mm256_load_pd(_pl);                             \
        (P)[6*(t)+1] = _mm256_load_pd(_pl + 4);                         \
        (P)[6*(t)+2] = _mm256_load_pd(_pl + 8);                         \
        (P)[6*(t)+3] = _mm256_load_pd(_pl + 12);                        \
        (P)[6*(t)+4] = _mm256_load_pd(_pl + 16);                        \
        (P)[6*(t)+5] = _mm256_load_pd(_pl + 20);                        \
    } while (0)

#define L6_SP2_YDFT(P, CD)                                              \
    do {                                                                \
        for (int _zp = 0; _zp < 3; ++_zp) {                             \
            __m256d _o0, _o1, _o2, _o3, _o4, _o5;                       \
            CD((P)[_zp], (P)[3+_zp], (P)[6+_zp], (P)[9+_zp],            \
               (P)[12+_zp], (P)[15+_zp], _o0, _o1, _o2, _o3, _o4, _o5); \
            (P)[_zp] = _o0; (P)[3+_zp] = _o1; (P)[6+_zp] = _o2;         \
            (P)[9+_zp] = _o3; (P)[12+_zp] = _o4; (P)[15+_zp] = _o5;     \
        }                                                               \
    } while (0)

#define L6_SP2_ZCHUNK(P, q, yp, ST, CD)                                 \
    L6_ZPAIR((P)[6*(yp)], (P)[6*(yp)+1], (P)[6*(yp)+2],                 \
             (P)[6*(yp)+3], (P)[6*(yp)+4], (P)[6*(yp)+5],               \
             (q) + 24*(yp), ST, CD)

/* consume plane xx from CUR while loading + y-transforming plane xx+1 */
#define L6_SP2_PLANE(CUR, NXT, xx, SRC, DST, ST, CD)                    \
    do {                                                                \
        const double *_pn = (SRC) + 72 * ((xx) + 1);                    \
        double *_qc = (DST) + 72 * (xx);                                \
        L6_SP2_ZCHUNK(CUR, _qc, 0, ST, CD); L6_SP2_LD6(NXT, _pn, 0);    \
        L6_SP2_ZCHUNK(CUR, _qc, 1, ST, CD); L6_SP2_LD6(NXT, _pn, 1);    \
        L6_SP2_ZCHUNK(CUR, _qc, 2, ST, CD); L6_SP2_LD6(NXT, _pn, 2);    \
        L6_SP2_YDFT(NXT, CD);                                           \
    } while (0)

#define L6_FUSED_YZ_SP2(SRC, DST, ST, CD)                               \
    do {                                                                \
        __m256d P[18], Q[18];                                           \
        L6_SP2_LD6(P, (SRC), 0); L6_SP2_LD6(P, (SRC), 1);               \
        L6_SP2_LD6(P, (SRC), 2);                                        \
        L6_SP2_YDFT(P, CD);                                             \
        for (int _x2 = 0; _x2 < 4; _x2 += 2) {                          \
            L6_SP2_PLANE(P, Q, _x2,     SRC, DST, ST, CD);              \
            L6_SP2_PLANE(Q, P, _x2 + 1, SRC, DST, ST, CD);              \
        }                                                               \
        L6_SP2_PLANE(P, Q, 4, SRC, DST, ST, CD);                        \
        L6_SP2_ZCHUNK(Q, (DST) + 360, 0, ST, CD);                       \
        L6_SP2_ZCHUNK(Q, (DST) + 360, 1, ST, CD);                       \
        L6_SP2_ZCHUNK(Q, (DST) + 360, 2, ST, CD);                       \
    } while (0)

#define L6_DEF_FUSED_SP2(NAME,PF)                                       \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,    \
                 const double *restrict in, double *restrict out,       \
                 long nvol)                                             \
{                                                                       \
    VSET;                                                               \
    (void)t2;                                                           \
    for (long b = 0; b < nvol; ++b) {                                   \
        const double *ip = in  + b * (long)L6_VD;                       \
        double       *op = out + b * (long)L6_VD;                       \
        L6_PASS_X(ip, t1, op, PF, VD6);                                 \
        L6_FUSED_YZ_SP2(t1, op, _mm256_store_pd, VD6);                  \
    }                                                                   \
}

L6_DEF_FUSED_SP2(l6_run_fused_sp2,    L6_PF_NONE)
L6_DEF_FUSED_SP2(l6_run_fused_sp2_pf, L6_PF_T0_1)
#endif  /* mt_r3 prune: sp2 twins out of the binary entirely */

/* mt_r1: NT-store fused kernel (mt_r3: only the fused_nt_pf twin remains
 * -- see the prune note above).  Same VD6 graph, same passes, stream_pd
 * in the z-pass stores (24 consecutive doubles per ZPAIR = 3 whole cache
 * lines, so the WC buffers always close cleanly).  fence=1 in cand[]: the
 * caller sfences on every thread that ran them.  pfw is deliberately NOT
 * combined with NT (a write-intent prefetch pulls the line into cache,
 * which defeats the streaming store's no-RFO purpose). */
L6_DEF_FUSED(l6_run_fused_nt_pf,     _mm256_stream_pd, L6_PF_T0_1,  VD6)

#endif /* L6_HAVE_AVX2 */

/* ------------------------------------------------------------------ *
 * Plan
 * ------------------------------------------------------------------ */

typedef void (*l6_kernel)(double *restrict t1, double *restrict t2,
                          const double *restrict in, double *restrict out,
                          long nvol);

struct fft3d_plan {
    int       L, batch;
    double   *arena;          /* owns the scratch; t1/t2 are placed inside it */
    double   *t1, *t2;
    l6_kernel run;
    int       fence;          /* nonzero if the chosen kernel uses NT stores */
    int       placed;         /* scratch already positioned for these buffers */
    int       forced;         /* kernel forced via L6_FORCE, not raced */
    int       tforced;        /* team size forced via L6_FORCE_T, not raced */
    int       nthr;           /* execute() team size; 1 = serial, no region */
    int       npool;          /* threads with a valid per-thread arena */
    const char *chosen;
    /* mt_r3 fr= placement instrument (ADOPTED FROM L8_fusedaxes's mt_r2
     * governor scan; the mt_r2 VERDICT s6 asks the L=6 tuner to carry it):
     * a read-only get_mempolicy sample of the CALLER's in/out page homes,
     * taken on the 1st and 49th threaded execute, published in the
     * description as fr=<pct-remote-at-call-1>/<at-call-49>,nb=<autonuma>.
     * Reading page homes is explicitly allowed; migrating them is not. */
    int       main_node;      /* NUMA node of the create()-time main thread */
    int       nb_on;          /* /proc/sys/kernel/numa_balancing, -1 unknown */
    int       fr0;            /* pct remote at call 1, -1 unknown */
    long      ncalls;         /* threaded executes seen */
    size_t    desc_off;       /* where the fr= field sits in l6_desc */
    /* mt_r1: per-thread scratch, allocated and first-touched by the owning
     * thread in fft3d_create() (NUMA-local by construction).  Each arena has
     * the same 4 KiB placement slack as the serial one. */
    double   *tarena[L6_MAXT];
    double   *tt1[L6_MAXT], *tt2[L6_MAXT];
#ifdef _OPENMP
    /* mt_r2 spin-wait dispatch pool (ADOPTED FROM L6_pfa mt_r1): 0 = one OMP
     * parallel region per execute; 1 = the pool below, kept only if it beat
     * OMP dispatch by >2% in the plan-time sequenced race. */
    int       use_pool;
    struct l6_pool *pool;
    cpu_set_t tset[L6_MAXT];  /* per-thread binding, captured from the OMP
                               * threads in the pool-warm region, so pool
                               * workers land on the same cores/sockets */
#endif
};

/* 4K-aliasing defence.  A store to S followed by a load from L with
 * (S-L) == 0 mod 4096 is falsely flagged as dependent and the load replays;
 * measured cost when a whole pass is in that state: +22% at B=1 (Haswell,
 * see strategies/L6_unrolled.md).  The scratch is therefore carved out of a
 * 4 KiB-oversized arena and positioned, once, so that every store->load
 * delta the kernels can produce sits as far as possible from 0 mod 4096.
 * This changes no arithmetic and no output, only addresses. */
static long l6_cyc(long d)
{
    d &= 4095;
    return d < 4096 - d ? d : 4096 - d;
}

static long l6_place_r(const void *in, const void *out)
{
    long D = (long)(((uintptr_t)out - (uintptr_t)in) & 4095u);
    long bestr = 0, bestscore = -1;
    for (long r = 0; r < 4096; r += 64) {
        long s1 = l6_cyc(r);                                  /* t1 - in      */
        long s2 = l6_cyc(D - r);                              /* out - t1     */
        long s3 = l6_cyc(D - r - L6_VD * (long)sizeof(double));/* out - t2    */
        long sc = s1 < s2 ? s1 : s2;
        if (s3 < sc) sc = s3;
        if (sc > bestscore) { bestscore = sc; bestr = r; }
    }
    return bestr;
}

static void l6_place(fft3d_plan *p, const void *in, const void *out)
{
    long bestr = l6_place_r(in, out);
    long off = (long)((((uintptr_t)in + (uintptr_t)bestr
                        - (uintptr_t)p->arena) & 4095u) / sizeof(double));
    p->t1 = p->arena + off;
    p->t2 = p->t1 + L6_VD;
    p->placed = 1;
}

/* Per-thread scratch placement for a T-way contiguous split of nvol volumes:
 * thread t's chunk base is its own "in"/"out" for the 4K-aliasing search.
 * Pointer arithmetic only (no touching, so NUMA residence is untouched);
 * serial and cheap (T*64 trivial iterations), run once per (in,out,T). */
static void l6_place_mt(fft3d_plan *p, const double *in, double *out,
                        long nvol, int T)
{
    for (int t = 0; t < T && t < p->npool; ++t) {
        long b0 = nvol * t / T;
        const double *ci = in + b0 * (long)L6_VD;
        double       *co = out + b0 * (long)L6_VD;
        long r = l6_place_r(ci, co);
        long off = (long)((((uintptr_t)ci + (uintptr_t)r
                            - (uintptr_t)p->tarena[t]) & 4095u)
                          / sizeof(double));
        p->tt1[t] = p->tarena[t] + off;
        p->tt2[t] = p->tt1[t] + L6_VD;
    }
}

/* mt_r3: sample the NUMA homes of ~128 pages of each caller buffer -- a
 * pure READ (get_mempolicy MPOL_F_NODE|MPOL_F_ADDR mutates nothing; the
 * mt_r1 ruling banned migration and the mt_r2 VERDICT asked for exactly
 * this diagnostic at L=6 B=65536).  Returns percent remote to main_node,
 * or -1 if unavailable.  ADOPTED FROM L8_fusedaxes mt_r2 (gov_scan_remote),
 * attributed. */
static int l6_scan_remote(const double *ip, const double *op, long batch,
                          int main_node)
{
#if defined(__linux__) && defined(SYS_get_mempolicy)
#ifndef MPOL_F_NODE
#define MPOL_F_NODE (1 << 0)
#endif
#ifndef MPOL_F_ADDR
#define MPOL_F_ADDR (1 << 1)
#endif
    if (main_node < 0) return -1;
    long step = batch / 128;
    if (step < 1) step = 1;
    int tot = 0, rem = 0;
    for (long b = 0; b < batch; b += step) {
        const void *pa[2] = { ip + (size_t)b * L6_VD,
                              op + (size_t)b * L6_VD };
        for (int k = 0; k < 2; ++k) {
            const uintptr_t pg = (uintptr_t)pa[k] & ~(uintptr_t)4095;
            int nd = -1;
            if (syscall(SYS_get_mempolicy, &nd, NULL, 0UL, (void *)pg,
                        (unsigned long)(MPOL_F_NODE | MPOL_F_ADDR)) == 0 &&
                nd >= 0) {
                ++tot;
                rem += (nd != main_node);
            }
        }
    }
    return tot ? (int)(100L * rem / tot) : -1;
#else
    (void)ip; (void)op; (void)batch; (void)main_node;
    return -1;
#endif
}

/* mt_r1 threaded call: T-way contiguous batch split, one kernel call per
 * thread on its own scratch, per-thread sfence for NT kernels (the join
 * barrier's release must not overtake WC-buffered stores), no other
 * synchronisation.  Ranges are computed from the ACTUAL team OpenMP
 * delivers, so a squeezed team still computes the whole batch (a squeezed
 * team only mis-places the 4K offsets, which is a perf matter, not a
 * correctness one). */
static void l6_mt_call(const fft3d_plan *p, l6_kernel run, int fence, int T,
                       const double *in, double *out, long nvol)
{
#ifdef _OPENMP
    if (T > 1) {
#pragma omp parallel num_threads(T)
        {
            int aT = omp_get_num_threads();
            int t  = omp_get_thread_num();
            long b0 = nvol * (long)t / aT;
            long b1 = nvol * (long)(t + 1) / aT;
            if (b1 > b0 && t < p->npool) {
                run(p->tt1[t], p->tt2[t],
                    in + b0 * (long)L6_VD, out + b0 * (long)L6_VD, b1 - b0);
#ifdef L6_HAVE_AVX2
                if (fence) _mm_sfence();
#endif
            }
        }
        return;
    }
#endif
    run(p->tt1[0] ? p->tt1[0] : p->t1, p->tt2[0] ? p->tt2[0] : p->t2,
        in, out, nvol);
#ifdef L6_HAVE_AVX2
    if (fence) _mm_sfence();
#endif
}

/* ------------------------------------------------------------------ *
 * mt_r2 spin-wait dispatch pool -- ADOPTED FROM L6_pfa's mt_r1 design
 * (epoch broadcast + padded per-worker done flags), with MY chunk split
 * (b0 = nvol*t/T, identical to l6_mt_call, so the output is bit-identical
 * under either dispatch).  GOMP's fork/join measured 13.5-18.3 us on the
 * node (their fork= probes) against ~1-2 us for this pool.  Workers spin
 * with pause and never sleep, so the pool is created AFTER the OMP arm of
 * the dispatch race is timed, and destroyed before create() returns unless
 * it wins by >2% -- a losing pool never leaves a spinning core in the
 * scored run.  Master included, the pool never exceeds the harness's
 * thread count.
 * ------------------------------------------------------------------ */
#if defined(L6_HAVE_AVX2) && defined(_OPENMP)

typedef struct {
    l6_kernel k;
    int fence, T;
    const double *in;
    double *out;
    long nvol;
} l6_job;

struct l6_pool;
typedef struct { struct l6_pool *pl; int idx; } l6_warg;

typedef struct l6_pool {
    _Atomic long epoch;
    _Atomic int  shutdown;
    l6_job job;
    fft3d_plan *plan;
    int nworkers;                    /* workers 1..nworkers; master is 0 */
    pthread_t th[L6_MAXT];
    l6_warg warg[L6_MAXT];
    /* mt_r3 fix (ADOPTED FROM L6_pfa mt_r2, who found the same bug in the
     * design I copied from them): pad[48] made each element 56 B, so
     * aligned(64) on the ARRAY still let adjacent workers' done flags share
     * cache lines and the join's release-stores ping-ponged.  64 B each. */
    struct { _Atomic long done; char pad[56]; } w[L6_MAXT]
        __attribute__((aligned(64)));
} l6_pool;

static void *l6_pool_worker(void *argp)
{
    l6_pool *pl = ((l6_warg *)argp)->pl;
    int t = ((l6_warg *)argp)->idx;
    long seen = 0;
    for (;;) {
        long e;
        while ((e = atomic_load_explicit(&pl->epoch, memory_order_acquire))
               == seen)
            _mm_pause();
        if (atomic_load_explicit(&pl->shutdown, memory_order_acquire)) break;
        l6_job jb = pl->job;         /* safe: written before the epoch bump */
        fft3d_plan *p = pl->plan;
        if (t < jb.T && t < p->npool) {
            long b0 = jb.nvol * (long)t / jb.T;
            long b1 = jb.nvol * (long)(t + 1) / jb.T;
            if (b1 > b0) {
                jb.k(p->tt1[t], p->tt2[t],
                     jb.in + b0 * (long)L6_VD, jb.out + b0 * (long)L6_VD,
                     b1 - b0);
                if (jb.fence) _mm_sfence();
            }
        }
        atomic_store_explicit(&pl->w[t].done, e, memory_order_release);
        seen = e;
    }
    return NULL;
}

static void l6_run_pool(l6_pool *pl, l6_kernel run, int fence, int T,
                        const double *in, double *out, long nvol)
{
    fft3d_plan *p = pl->plan;
    pl->job.k = run; pl->job.fence = fence; pl->job.T = T;
    pl->job.in = in; pl->job.out = out; pl->job.nvol = nvol;
    long e = atomic_load_explicit(&pl->epoch, memory_order_relaxed) + 1;
    atomic_store_explicit(&pl->epoch, e, memory_order_release);
    long b1 = nvol / (long)T;        /* master runs chunk 0 itself */
    if (b1 > 0) run(p->tt1[0], p->tt2[0], in, out, b1);
    if (fence) _mm_sfence();
    /* mt_r3 (ADOPTED FROM L6_pfa mt_r2): prefetch all done-flag lines
     * before scanning, so the cross-core misses (each flag Modified in its
     * worker's cache) overlap in the fill buffers instead of serializing. */
    for (int t = 1; t <= pl->nworkers; ++t)
        _mm_prefetch((const char *)&pl->w[t].done, _MM_HINT_T0);
    /* EVERY worker acks every epoch (idle ones immediately), so the next
     * job write can never race a straggler's read of this one */
    for (int t = 1; t <= pl->nworkers; ++t)
        while (atomic_load_explicit(&pl->w[t].done, memory_order_acquire) != e)
            _mm_pause();
}

static void l6_pool_destroy(l6_pool *pl)
{
    if (!pl) return;
    atomic_store_explicit(&pl->shutdown, 1, memory_order_release);
    atomic_fetch_add_explicit(&pl->epoch, 1, memory_order_release);
    for (int t = 1; t <= pl->nworkers; ++t) pthread_join(pl->th[t], NULL);
    free(pl);
}

static l6_pool *l6_pool_create(fft3d_plan *p, int tmax)
{
    l6_pool *pl = NULL;
    if (posix_memalign((void **)&pl, 64, sizeof(*pl)) != 0 || !pl)
        return NULL;
    memset(pl, 0, sizeof(*pl));
    pl->plan = p;
    pl->nworkers = tmax - 1;
    for (int t = 1; t <= pl->nworkers; ++t) {
        pl->warg[t].pl = pl; pl->warg[t].idx = t;
        if (pthread_create(&pl->th[t], NULL, l6_pool_worker, &pl->warg[t])) {
            pl->nworkers = t - 1;    /* join only what exists */
            l6_pool_destroy(pl);
            return NULL;
        }
        pthread_setaffinity_np(pl->th[t], sizeof(cpu_set_t), &p->tset[t]);
    }
    return pl;
}
#endif /* L6_HAVE_AVX2 && _OPENMP */

const char *fft3d_name(void) { return "L6_unrolled"; }

static char l6_desc[512] =
    "L=6: unrolled straight-line PFA 2x3 six-point codelet (48 flops/36 instr, "
    "no twiddles) on all three axes, AVX2/FMA 2-complex lanes, in-register "
    "z-pencil transposes; variant=auto";

const char *fft3d_description(void) { return l6_desc; }

int fft3d_supports(int L) { return L == L6_L; }

static void *l6_alloc(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
}

#ifdef L6_HAVE_AVX2
static double l6_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Busy 256-bit spin for SECS wall seconds.  Used (a) before the race, so
 * round 0 is not ranked on a ramping clock (L17_rader's r5 settle-spin
 * finding: 76% mis-ranking from table order alone), and (b) near the END
 * of create() as belt-and-braces before the chosen-kernel dwell.  (The
 * original motivation -- letting the AVX-512 licence expire after the r5
 * zmm probes -- is moot since panel_r10: no 512-bit instruction exists in
 * this file any more.) */
static void l6_spin256(double secs)
{
    __m256d x = _mm256_set1_pd(1.0);
    const __m256d a = _mm256_set1_pd(1.0 + 1e-15);
    const __m256d b = _mm256_set1_pd(1e-300);
    double until = l6_now() + secs;
    do {
        for (int i = 0; i < 8192; ++i) x = _mm256_fmadd_pd(x, a, b);
    } while (l6_now() < until);
    double lane[4];
    _mm256_storeu_pd(lane, x);
    if (!(lane[0] > 0.0)) fprintf(stderr, "l6_spin256: impossible\n");
}

/* ------------------------------------------------------------------ *
 * kclk (panel_r7) -- ADOPTED FROM L6_pfa's panel_r6 record: the clock
 * the CHOSEN kernel actually runs at, measured directly instead of
 * inferred from a synthetic chain's density.  Dwell ~2 ms in the real
 * kernel, then immediately time a ~150 us sparse ymm FMA chain: CLX
 * licence state persists ~670 us after the last heavy instruction, so
 * the sparse chain (which by itself never raises the licence) reads
 * the licence the kernel established.  Median of 9 dwell/read pairs.
 * freq = iters*4/dt (latency-4 FMA; the Haswell latency-5 over-read
 * caveat from the r4 probe applies to wombat only).  ~20 ms, unscored.
 * ------------------------------------------------------------------ */
static double l6_kclk(l6_kernel run, int fence, double *t1, double *t2)
{
    const long nd = 4;
    double *din  = (double *)l6_alloc((size_t)nd * L6_VD * sizeof(double));
    double *dout = (double *)l6_alloc((size_t)nd * L6_VD * sizeof(double));
    if (!din || !dout) { free(din); free(dout); return 0.0; }
    uint64_t st = 0xA0761D6478BD642Full;
    for (long i = 0; i < nd * L6_VD; ++i) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        din[i] = (double)(int64_t)(st >> 11) * (1.0 / 9007199254740992.0);
    }
    const __m256d a = _mm256_set1_pd(1.0 + 1e-15);
    const __m256d b = _mm256_set1_pd(1e-300);
    double vals[9];
    for (int r = 0; r < 9; ++r) {
        double until = l6_now() + 2e-3;
        do {
            run(t1, t2, din, dout, nd);
        } while (l6_now() < until);
        if (fence) _mm_sfence();
        __m256d x = _mm256_set1_pd(1.0);       /* named chain, never array */
        double t0 = l6_now();
        for (int i = 0; i < 131072; ++i) x = _mm256_fmadd_pd(x, a, b);
        double dt = l6_now() - t0;
        double lane[4];
        _mm256_storeu_pd(lane, x);
        vals[r] = (lane[0] > 0.0) ? 131072.0 * 4.0 / dt * 1e-9 : 0.0;
    }
    free(din); free(dout);
    for (int i = 1; i < 9; ++i) {              /* insertion sort, median */
        double v = vals[i]; int j = i;
        while (j > 0 && vals[j-1] > v) { vals[j] = vals[j-1]; --j; }
        vals[j] = v;
    }
    return vals[4] > 9.9 ? 0.0 : vals[4];
}

/* End-of-create dwell in the CHOSEN kernel (panel_r8, ADOPTED FROM
 * L6_pfa's panel_r7 refinement of my r6 licence-tail fix): ~3 ms of the
 * scored kernel itself, so the driver is handed a core in that kernel's
 * own licence/clock steady state -- never a probe's licence and never a
 * sparse spin's light-licence clock.  Falls back silently (the r6
 * spin256 tail already ran) if the scratch cannot be allocated. */
static void l6_dwell_chosen(l6_kernel run, int fence, double *t1, double *t2)
{
    const long nd = 4;
    double *din  = (double *)l6_alloc((size_t)nd * L6_VD * sizeof(double));
    double *dout = (double *)l6_alloc((size_t)nd * L6_VD * sizeof(double));
    if (!din || !dout) { free(din); free(dout); return; }
    uint64_t st = 0xE7037ED1A0B428DBull;
    for (long i = 0; i < nd * L6_VD; ++i) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        din[i] = (double)(int64_t)(st >> 11) * (1.0 / 9007199254740992.0);
    }
    double until = l6_now() + 3e-3;
    do {
        run(t1, t2, din, dout, nd);
    } while (l6_now() < until);
    if (fence) _mm_sfence();
    free(din); free(dout);
}

/* ------------------------------------------------------------------ *
 * In-plan B=1 discriminator (panel_r9) -- ADOPTED FROM L36_pfa's r8
 * in-plan node probe pattern, which the r8 VERDICT calls the round's
 * most reusable idea and explicitly asks L=6 to apply: when the monitor
 * cannot run your counter, build the discriminator into create() and
 * route the result through fft3d_description().  perf_event_open is
 * closed on this cluster (perf_event_paranoid=4 on every machine we can
 * see), so this is a TIMED discriminator: min ns/volume at nvol=1,
 * driver-like conditions (same in/out every call, placed scratch), 9
 * trials of 256 reps, each trial preceded by ~0.7 ms of the kernel
 * itself so every candidate is measured in its own licence/clock steady
 * state (the r7 licence-fair race rule).  panel_r10 retarget: the r9
 * questions (fused vs fused3, fused vs zff) are answered and those
 * kernels deleted; ab1 now runs for fused / fused_zp -- the ascending
 * vs zp-outer x-pass group order, the one structural difference left
 * between my B=1 kernel and L6_pfa's node-winning fused_d2 -- so the
 * node publishes the A/B regardless of which one the tournament picks.
 * ~17 ms total, unscored.
 * ------------------------------------------------------------------ */
static double l6_ab1(l6_kernel run, int fence, double *t1, double *t2,
                     const double *din, double *dout)
{
    double best = 1e300;
    for (int trial = 0; trial < 9; ++trial) {
        double wu = l6_now() + 7e-4;
        do { run(t1, t2, din, dout, 1); } while (l6_now() < wu);
        if (fence) _mm_sfence();
        double t0 = l6_now();
        for (int r = 0; r < 256; ++r) run(t1, t2, din, dout, 1);
        if (fence) _mm_sfence();
        double dt = (l6_now() - t0) * (1.0 / 256.0);
        if (dt < best) best = dt;
    }
    return best * 1e9;   /* ns per volume */
}

/* (mt_r1: the panel_r11 l6_abL DRAM codelet A/B was deleted -- its question
 * was answered on the node: abL=f524.0,f3529.6, VD6 wins the DRAM regime.) */
#endif

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != L6_L || batch <= 0) return NULL;

    fft3d_plan *p = (fft3d_plan *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    /* 4096 bytes of slack for the placement search + the two scratch volumes */
    p->arena = (double *)l6_alloc(4096 + 2 * L6_VD * sizeof(double) + 64);
    if (!p->arena) { fft3d_destroy(p); return NULL; }
    p->t1 = p->arena;
    p->t2 = p->arena + L6_VD;
    p->run = l6_run_scalar;
    p->fence = 0;
    p->nthr = 1;
    p->npool = 0;
    p->chosen = "scalar";
    /* mt_r3 fr= instrument state (see the struct comment) */
    p->main_node = -1;
    p->nb_on = -1;
    p->fr0 = -1;
#if defined(__linux__)
#if defined(SYS_getcpu)
    {
        unsigned cpuu = 0, nodeu = 0;
        if (syscall(SYS_getcpu, &cpuu, &nodeu, NULL) == 0)
            p->main_node = (int)nodeu;
    }
#endif
    {   /* is AutoNUMA even on?  (the VERDICT's cheap check) */
        FILE *nf = fopen("/proc/sys/kernel/numa_balancing", "r");
        if (nf) {
            int nb = -1;
            if (fscanf(nf, "%d", &nb) == 1) p->nb_on = nb;
            fclose(nf);
        }
    }
#endif

    /* ---- mt_r1: spin up the OpenMP pool at full width and give every pool
     * thread its own scratch arena, allocated and FIRST-TOUCHED by the
     * owning thread (NUMA-local; PANEL_BRIEF: we control our scratch's
     * first touch, never the caller's buffers).  This is also what creates
     * the pool threads, so execute() never pays thread creation.  Arena
     * size = the serial arena's: 4 KiB of placement slack + t1 + t2. ---- */
    {
        int pool = 1;
#ifdef _OPENMP
        pool = omp_get_max_threads();       /* the harness's 32; never more */
        if (pool > L6_MAXT) pool = L6_MAXT;
        if (pool < 1) pool = 1;
        size_t tbytes = 4096 + 2 * L6_VD * sizeof(double) + 64;
#pragma omp parallel num_threads(pool)
        {
            int t = omp_get_thread_num();
            if (t < L6_MAXT) {
                /* mt_r2: record this OMP thread's binding so the spin pool's
                 * worker t can be pinned to the same core (and socket as
                 * the scratch first-touched right below) */
                pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t),
                                       &p->tset[t]);
                double *a = (double *)l6_alloc(tbytes);
                if (a) {
                    memset(a, 0, tbytes);           /* first touch by owner */
                    p->tarena[t] = a;
                    p->tt1[t] = a;
                    p->tt2[t] = a + L6_VD;
                }
            }
        }
        p->npool = pool;
        for (int t = 0; t < pool; ++t)
            if (!p->tarena[t]) { p->npool = t; break; }
#endif
        (void)pool;
    }
    /* the widest team execute() may use: pool threads with scratch, and
     * never more threads than volumes */
    int tmax = p->npool < 1 ? 1 : p->npool;
    if (tmax > batch) tmax = batch;

    /* panel_r10 in-plan discriminator results (see l6_ab1) and the
     * fused_zp-vs-fused race delta at the plan's own batch size. */
    double ab_f = 0.0, ab_fx = 0.0, xod = 0.0;
    int have_xod = 0;
    /* mt_r1 team-size race curve, ns/volume, for the description (0 = not
     * raced); index into l6_tset below. */
    static const int l6_tset[] = { 1, 2, 4, 8, 16, 24, 32 };
    enum { L6_NTSET = (int)(sizeof(l6_tset) / sizeof(l6_tset[0])) };
    double tcurve[L6_NTSET];
    for (int i = 0; i < L6_NTSET; ++i) tcurve[i] = 0.0;
    /* mt_r2 dispatch race results, ns/volume (0 = not raced) */
    double disp_omp = 0.0, disp_pool = 0.0;
    /* mt_r3: which dispatch the TOURNAMENT itself ran through (pool unless
     * pool creation failed); published as rd= so a node pick can never be
     * misread against the wrong dispatch regime */
    int race_disp_pool = 0;

#ifdef L6_HAVE_AVX2
    {
        /* Ordered "safest first": a later candidate must beat the incumbent by
         * more than its own takeover margin, so measurement noise on a loaded
         * machine cannot promote a streaming-store kernel at a batch size
         * where the working set is cache resident.
         *
         * panel_r11 grid: the same 10 raced ymm kernels as r10, with each
         * zp twin and its ascending parent SWAPPED (my r10 branch (iii),
         * prescribed by the r10 VERDICT; the exact move L6_pfa made for
         * their d2 twins in r10, which delivered 4/4 predicted picks).
         * Rationale: the node picked fused_zp 2/3 at B=1 and read ab1
         * fx-f = -0.6/-3.1/-1.2%, but a node-measured sub-margin winner in
         * a trailing slot is invisible to the tuner by construction.  The
         * zp twins now hold each family's incumbent slot at the full 2.5%
         * margin; the ascending parents follow as 1.0%-margin challengers.
         * Each pair is output-BIT-IDENTICAL (same per-line arithmetic,
         * different order over independent groups), so a flip either way
         * is bounded and harmless -- which is what makes the reduced
         * margin safe (per-candidate margins since r10; 2.5% hysteresis
         * for genuinely different shapes since r4).
         * Layout defence: every kernel entry 64B-pinned (since r6). */
        static const struct {
            l6_kernel k; int fence; double mg; const char *nm;
        } cand[] = {
            { l6_run_3pass,         0, 0.025, "3pass"         },
            { l6_run_fused_zp,      0, 0.025, "fused_zp"      },
            { l6_run_fused,         0, 0.010, "fused"         },
            { l6_run_3pass_pf,      0, 0.025, "3pass_pf"      },
            { l6_run_fused_zp_pf,   0, 0.025, "fused_zp_pf"   },
            { l6_run_fused_pf,      0, 0.010, "fused_pf"      },
            { l6_run_3pass_pfw,     0, 0.025, "3pass_pfw"     },
            { l6_run_fused_zp_pfw,  0, 0.025, "fused_zp_pfw"  },
            { l6_run_fused_pfw,     0, 0.010, "fused_pfw"     },
            /* mt_r3 prune: sp2 twins, fused_nt, fused_zp_nt_pf and
             * fused_nt_pfnta deleted (0 node picks each in 2 rounds; see
             * the prune note at the kernel definitions).  The two NT
             * shapes that remain carry the live questions: 3pass_nt_pf
             * is the node's B=65536 winner (incumbent slot ordering keeps
             * it ahead of the fused twin), fused_nt_pf is the controlled
             * 3pass-vs-fused NT comparison.  Single-core NT still loses
             * (phase 1: 0-for-4 rounds); the race is threaded at batch>1,
             * which is the regime they exist for. */
            { l6_run_3pass_nt_pf,   1, 0.025, "3pass_nt_pf"   },
            { l6_run_fused_nt_pf,   1, 0.025, "fused_nt_pf"   },
        };
        const int ncand = (int)(sizeof(cand)/sizeof(cand[0]));

        /* ---- correctness gate: every candidate must reproduce the scalar
         * reference bit-closely on random data before it may be timed. ---- */
        long nval = batch < 4 ? batch : 4;
        double *vin  = (double *)l6_alloc((size_t)nval * L6_VD * sizeof(double));
        double *vref = (double *)l6_alloc((size_t)nval * L6_VD * sizeof(double));
        double *vgot = (double *)l6_alloc((size_t)nval * L6_VD * sizeof(double));
        int ok[48]; for (int i = 0; i < ncand; ++i) ok[i] = 0;

        if (vin && vref && vgot) {
            uint64_t st = 0x9E3779B97F4A7C15ull;
            for (long i = 0; i < nval * L6_VD; ++i) {
                st ^= st << 13; st ^= st >> 7; st ^= st << 17;
                vin[i] = (double)(int64_t)(st >> 11) * (1.0 / 9007199254740992.0);
            }
            l6_run_scalar(p->t1, p->t2, vin, vref, nval);
            double nrm = 0.0;
            for (long i = 0; i < nval * L6_VD; ++i) nrm += vref[i] * vref[i];
            nrm = sqrt(nrm) + 1e-300;
            for (int c = 0; c < ncand; ++c) {
                memset(vgot, 0, (size_t)nval * L6_VD * sizeof(double));
                cand[c].k(p->t1, p->t2, vin, vgot, nval);
                if (cand[c].fence) _mm_sfence();
                double e = 0.0;
                for (long i = 0; i < nval * L6_VD; ++i) {
                    double d = vgot[i] - vref[i];
                    e += d * d;
                }
                ok[c] = (sqrt(e) / nrm) < 1e-11;
            }
        }
        free(vin); free(vref); free(vgot);

        /* ---- forced variant (panel_r5): the L6_FORCE env var (or a
         * -DL6_FORCE_DEFAULT='"name"' compile flag) names a candidate that
         * is selected unconditionally, skipping the race -- it must still
         * pass the correctness gate above.  This is the node A/B switch
         * the panel_r4 VERDICT asked for ("force the _s shapes on the
         * node"); the pick is reported as variant=<name>! so a forced
         * leaderboard line cannot be mistaken for a tournament pick. */
        int forced = -1;
        {
            const char *fv = getenv("L6_FORCE");
#ifdef L6_FORCE_DEFAULT
            if (!fv || !fv[0]) fv = L6_FORCE_DEFAULT;
#endif
            if (fv && fv[0])
                for (int c = 0; c < ncand; ++c)
                    if (ok[c] && strcmp(cand[c].nm, fv) == 0) { forced = c; break; }
        }

        /* ---- race the survivors at (a truncation of) the real batch size ----
         * The cap must keep the raced working set out of L3 when the real one
         * is: 16384 volumes = 113 MiB is unambiguous DRAM even against the
         * node's two-socket combined 44 MiB L3.  batch=4096 races at its
         * REAL size (27 MiB), which on 32 threads can be aggregate-L3
         * resident -- a genuinely different regime from B=65536, and the
         * reason the cap must never round 4096 up.
         *
         * mt_r1: when batch > 1 and a thread pool exists, the race runs
         * THREADED at full width (use_mt), through the exact l6_mt_call
         * path execute() will use -- fork/join and all -- because the
         * 32-thread winner is not the 1-thread winner (NT stores lose
         * single-core, win DRAM-bound threaded). */
        int use_mt = (batch > 1 && tmax > 1);
        int tforce = 0;
        {
            const char *tv = getenv("L6_FORCE_T");
            if (tv && tv[0]) {
                int v = atoi(tv);
                if (v >= 1) tforce = v > p->npool ? p->npool : v;
            }
        }
        if (use_mt) p->nthr = tmax;      /* default; the team race refines */
        /* mt_r2 (ADOPTED FROM L6_pfa mt_r1, their round's central lesson):
         * the MULTICORE race cap is 65536 volumes = 452 MiB.  32 threads
         * see the AGGREGATE cache (node ~76 MiB, wallaby ~124 MiB), so the
         * old 16384-volume / 113 MiB arena raced a near-cache-resident
         * problem while the real B=65536 batch is DRAM-bound -- that
         * mis-ranked the NT kernels AND flattened the team-size curve
         * (T=16 ~= T=32 raced, when the real batch runs 1.8x faster at
         * T=32 with NT; node mt_r1: my 72.1 vs their 39.5 ns/vol).  The
         * serial arm keeps 16384 (one core sees one L3; phase-1 evidence
         * unchanged). */
        long nt = batch;
        long ntcap = use_mt ? 65536 : 16384;
        if (nt > ntcap) nt = ntcap;
        double *ain = NULL, *aout = NULL;
        int best = forced;
        if (best < 0) {
            ain  = (double *)l6_alloc((size_t)nt * L6_VD * sizeof(double));
            aout = (double *)l6_alloc((size_t)nt * L6_VD * sizeof(double));
        }
        /* mt_r3 (ADOPTED FROM L6_pfa mt_r2, their round's payload): the
         * tournament and the team race now RUN THROUGH THE SPIN POOL, so
         * every timed cell is dispatched the way a pool pick would ship.
         * mt_r2 timed the race through the OMP fork/join (13-18 us on the
         * node) while the scored B=4096 run used the ~1-2 us pool -- the
         * race was ranking configurations under a dispatch the winner
         * never uses, the exact defect that cost L6_pfa their mt_r1
         * B=4096 cell (sp2 T=24 mis-pick, 14.4 vs 9.4 ns/vol).  OMP
         * dispatch is only the race fallback if pool creation fails. */
#ifdef _OPENMP
        l6_pool *rpool = NULL;
#define L6_RACE_MT(k,f,T)                                                 \
        do {                                                              \
            if (rpool && (T) > 1)                                         \
                l6_run_pool(rpool, k, f, (T), ain, aout, nt);             \
            else                                                          \
                l6_mt_call(p, k, f, (T), ain, aout, nt);                  \
        } while (0)
#else
#define L6_RACE_MT(k,f,T) l6_mt_call(p, k, f, (T), ain, aout, nt)
#endif
#define L6_RACE_CALL(c)                                                   \
        do {                                                              \
            if (use_mt)                                                   \
                L6_RACE_MT(cand[c].k, cand[c].fence, tmax);               \
            else {                                                        \
                cand[c].k(p->t1, p->t2, ain, aout, nt);                   \
                if (cand[c].fence) _mm_sfence();                          \
            }                                                             \
        } while (0)
        if (best < 0 && ain && aout) {
#ifdef _OPENMP
            if (use_mt) {
                rpool = l6_pool_create(p, tmax);
                race_disp_pool = (rpool != NULL);
            }
#endif
            uint64_t st = 0xD1B54A32D192ED03ull;
            /* serial init, like the driver's fread: one thread first-touches
             * the whole buffer, so the race sees the driver's NUMA layout */
            for (long i = 0; i < nt * L6_VD; ++i) {
                st ^= st << 13; st ^= st >> 7; st ^= st << 17;
                ain[i] = (double)(int64_t)(st >> 11) * (1.0 / 9007199254740992.0);
            }
            memset(aout, 0, (size_t)nt * L6_VD * sizeof(double));
            if (use_mt) l6_place_mt(p, ain, aout, nt, tmax);
            else        l6_place(p, ain, aout);
            /* settle spin (L17_rader's r5 finding: on a ramping clock a
             * fixed-order table mis-ranked bit-identical work by 76%);
             * 100 ms of 256-bit work brings the core to its steady licence
             * clock before round 0 is timed.  Unscored. */
            l6_spin256(0.1);
            /* how many repeats make one trial ~1.5 ms: MEASURED (a
             * 32-thread call is ~10x faster per volume than the serial
             * model the phase-1 constant assumed) */
            long reps = 1;
            {
                int c0 = -1;
                for (int c = 0; c < ncand; ++c) if (ok[c]) { c0 = c; break; }
                if (c0 >= 0) {
                    L6_RACE_CALL(c0);                        /* page/pool warm */
                    double t0 = l6_now();
                    L6_RACE_CALL(c0);
                    double dt = l6_now() - t0;
                    if (dt > 1e-9) reps = (long)(1.5e-3 / dt);
                    if (reps < 1) reps = 1;
                    if (reps > 20000) reps = 20000;
                }
            }
            /* Round-robin tournament (adopted from L6_pfa's record, which
             * documents a sequential per-candidate race mis-picking by 21%
             * when background load drifts between candidates): every round
             * times each surviving candidate once, and each candidate keeps
             * its own minimum, so drift hits all candidates alike. */
            double tmin[48];
            for (int c = 0; c < ncand; ++c) {
                tmin[c] = 1e300;
                if (!ok[c]) continue;
                L6_RACE_CALL(c);                             /* warm */
            }
            for (int round = 0; round < 7; ++round) {
                for (int c = 0; c < ncand; ++c) {
                    if (!ok[c]) continue;
                    /* per-candidate licence warm-up (panel_r7, per
                     * LITERATURE 08 s4.3): run the candidate itself,
                     * untimed, for ~0.7 ms, so every candidate is timed
                     * in its own licence/clock steady state, never a
                     * predecessor's tail (CLX licence dwell is ~670 us
                     * -- comparable to a whole 2 ms trial). */
                    double wu = l6_now() + 7e-4;
                    do {
                        L6_RACE_CALL(c);
                    } while (l6_now() < wu);
                    double t0 = l6_now();
                    for (long r = 0; r < reps; ++r)
                        L6_RACE_CALL(c);
                    double dt = (l6_now() - t0) / (double)reps;
                    if (dt < tmin[c]) tmin[c] = dt;
                }
            }
            /* Safest-first with a per-candidate takeover margin (panel_r10:
             * the bit-identical zp twins carry 1.0%, everything else keeps
             * the 2.5% hysteresis).  The reference time tracks the true
             * minimum even when the incumbent survives, so a chain of
             * sub-margin steps cannot drift the pick. */
            double bestt = 1e300;
            for (int c = 0; c < ncand; ++c) {
                if (!ok[c]) continue;
                if (best < 0 || tmin[c] < bestt * (1.0 - cand[c].mg)) {
                    bestt = tmin[c]; best = c;
                } else if (tmin[c] < bestt) bestt = tmin[c];
            }
            if (getenv("L6_VERBOSE"))
                for (int c = 0; c < ncand; ++c)
                    fprintf(stderr, "L6_unrolled race: %-14s %s %10.4f us/vol%s\n",
                            cand[c].nm,
                            !ok[c] ? "BAD" : "ok ",
                            ok[c] ? tmin[c] / (double)nt * 1e6 : 0.0,
                            c == best ? "   <-- chosen" : "");
            /* fused_zp-vs-fused delta at the plan's own raced batch size,
             * best of family vs best of family, for the description
             * (positive = the zp-outer x order is slower). */
            {
                double bf = 1e300, bfx = 1e300;
                for (int c = 0; c < ncand; ++c) {
                    if (!ok[c]) continue;
                    if (strncmp(cand[c].nm, "fused_zp", 8) == 0) {
                        if (tmin[c] < bfx) bfx = tmin[c];
                    } else if (strncmp(cand[c].nm, "fused", 5) == 0) {
                        if (tmin[c] < bf) bf = tmin[c];
                    }
                }
                if (bf < 1e299 && bfx < 1e299) {
                    xod = (bfx / bf - 1.0) * 100.0;
                    have_xod = 1;
                }
            }

            /* ---- mt_r1 team-size race: the chosen kernel at T in
             * {1,2,4,8,16,24,32} over the same buffers, round-robin,
             * per-T minimum.  T=1 here is a DIRECT call (no parallel
             * region), exactly like the shipped serial path.  T=16 under
             * OMP_PROC_BIND=close is one socket on the node = every
             * thread local to the driver-touched in/out; T=32 adds the
             * far socket's fill buffers but pays UPI -- that is the NUMA
             * question, and the whole curve is published as tm=... so
             * the node answers it on every leaderboard line.  The
             * smallest T within 2% of the best wins (fewer threads, same
             * time = less contention and a cheaper join). ---- */
            if (use_mt && best >= 0 && !tforce) {
                l6_kernel kk = cand[best].k;
                int kf = cand[best].fence;
                double tval[L6_NTSET];
                for (int i = 0; i < L6_NTSET; ++i) tval[i] = 1e300;
                for (int round = 0; round < 5; ++round) {
                    for (int i = 0; i < L6_NTSET; ++i) {
                        int T = l6_tset[i];
                        if (T > tmax) continue;
                        l6_place_mt(p, ain, aout, nt, T);
                        double wu = l6_now() + 7e-4;
                        do {
                            L6_RACE_MT(kk, kf, T);
                        } while (l6_now() < wu);
                        double t0 = l6_now(), dt;
                        long r = 0;
                        do {
                            L6_RACE_MT(kk, kf, T);
                            ++r;
                        } while ((dt = l6_now() - t0) < 1.2e-3);
                        dt /= (double)r;
                        if (dt < tval[i]) tval[i] = dt;
                    }
                }
                /* mt_r3 pick rule (the mt_r2 VERDICT's mechanical fix for
                 * the panel-wide T=32->T=16 pick lottery, prescribed in its
                 * s6): at a STREAMING cell -- real working set beyond any
                 * aggregate cache here (node ~76 MiB, wallaby ~124 MiB) --
                 * the WIDEST team is the incumbent and a narrower one must
                 * beat it by >2%, because the create-time arena races in
                 * the pre-spread placement regime where a wide team looks
                 * spuriously bad (L6_pfa's B=65536: T=32 200 GB/s vs T=16
                 * 85 GB/s on the same binary, a 2.06x scored lottery).
                 * Cache-resident cells keep the old smallest-T-within-2%
                 * rule (fewer threads, same time = less contention). */
                double wsB = (double)batch * (double)L6_VD
                             * (double)sizeof(double) * 2.0;
                int streaming = wsB > 128.0 * 1024.0 * 1024.0;
                int bestT = tmax;
                if (streaming) {
                    int ii = -1;
                    for (int i = 0; i < L6_NTSET; ++i)
                        if (l6_tset[i] <= tmax) ii = i;   /* widest raced */
                    if (ii >= 0) {
                        bestT = l6_tset[ii];
                        double inc = tval[ii];
                        for (int i = ii - 1; i >= 0; --i)
                            if (tval[i] < inc * 0.98) {
                                bestT = l6_tset[i];
                                inc = tval[i];
                            }
                    }
                } else {
                    double tb = 1e300;
                    for (int i = 0; i < L6_NTSET; ++i)
                        if (l6_tset[i] <= tmax && tval[i] < tb) tb = tval[i];
                    for (int i = 0; i < L6_NTSET; ++i)
                        if (l6_tset[i] <= tmax && tval[i] <= tb * 1.02) {
                            bestT = l6_tset[i];
                            break;
                        }
                }
                for (int i = 0; i < L6_NTSET; ++i)
                    if (l6_tset[i] <= tmax)
                        tcurve[i] = tval[i] / (double)nt * 1e9;
                p->nthr = bestT;
                if (getenv("L6_VERBOSE"))
                    for (int i = 0; i < L6_NTSET; ++i)
                        if (l6_tset[i] <= tmax)
                            fprintf(stderr, "L6_unrolled team: T=%-2d "
                                    "%10.4f ns/vol%s\n", l6_tset[i],
                                    tcurve[i],
                                    l6_tset[i] == bestT ? "   <-- chosen" : "");
            }

            /* ---- dispatch race: OMP fork/join vs the spin-wait pthread
             * pool, on the CHOSEN (kernel, T) over the same arena.
             * SEQUENCED, not interleaved: the pool's workers never sleep,
             * so timing OMP with a live pool (or vice versa) poisons the
             * cell.  mt_r3 RE-SEQUENCED (ADOPTED FROM L6_pfa mt_r2): the
             * race pool already exists and is warm from the tournament,
             * so it is timed FIRST; it is then DESTROYED (workers joined,
             * cores genuinely idle -- GOMP's team futex-slept through the
             * multi-second pool-raced tournament) before OMP is timed,
             * and the pool is recreated only if it won by >2%.  A losing
             * pool never leaves a spinning core in the scored run. ---- */
            if (use_mt && best >= 0) {
                int Td = tforce ? tforce : p->nthr;
                if (Td > 1) {
                    l6_place_mt(p, ain, aout, nt, Td);
                    double tv;
#ifdef _OPENMP
                    if (rpool) {
                        tv = 1e300;
                        for (int trial = 0; trial < 5; ++trial) {
                            double wu = l6_now() + 7e-4;
                            do {
                                l6_run_pool(rpool, cand[best].k,
                                            cand[best].fence, Td,
                                            ain, aout, nt);
                            } while (l6_now() < wu);
                            double t0 = l6_now(), dt;
                            long r = 0;
                            do {
                                l6_run_pool(rpool, cand[best].k,
                                            cand[best].fence, Td,
                                            ain, aout, nt);
                                ++r;
                            } while ((dt = l6_now() - t0) < 1.2e-3);
                            dt /= (double)r;
                            if (dt < tv) tv = dt;
                        }
                        disp_pool = tv / (double)nt * 1e9;
                        l6_pool_destroy(rpool);  /* idle cores for OMP */
                        rpool = NULL;
                    }
#endif
                    tv = 1e300;
                    for (int trial = 0; trial < 5; ++trial) {
                        double wu = l6_now() + 7e-4;
                        do {
                            l6_mt_call(p, cand[best].k, cand[best].fence,
                                       Td, ain, aout, nt);
                        } while (l6_now() < wu);
                        double t0 = l6_now(), dt;
                        long r = 0;
                        do {
                            l6_mt_call(p, cand[best].k, cand[best].fence,
                                       Td, ain, aout, nt);
                            ++r;
                        } while ((dt = l6_now() - t0) < 1.2e-3);
                        dt /= (double)r;
                        if (dt < tv) tv = dt;
                    }
                    disp_omp = tv / (double)nt * 1e9;
#ifdef _OPENMP
                    if (disp_pool > 0.0) {
                        if (disp_pool < disp_omp * 0.98) {
                            l6_pool *pl = l6_pool_create(p, tmax);
                            if (pl) { p->pool = pl; p->use_pool = 1; }
                        }
                    } else {
                        /* the race pool never existed (creation failed):
                         * fall back to the mt_r2 order -- OMP was just
                         * timed; warm a fresh pool ~3 ms (GOMP's spinners
                         * futex-sleep) and time it second. */
                        l6_pool *pl = l6_pool_create(p, tmax);
                        if (pl) {
                            tv = 1e300;
                            double wu = l6_now() + 3e-3;
                            do {
                                l6_run_pool(pl, cand[best].k,
                                            cand[best].fence, Td,
                                            ain, aout, nt);
                            } while (l6_now() < wu);
                            for (int trial = 0; trial < 5; ++trial) {
                                wu = l6_now() + 7e-4;
                                do {
                                    l6_run_pool(pl, cand[best].k,
                                                cand[best].fence, Td,
                                                ain, aout, nt);
                                } while (l6_now() < wu);
                                double t0 = l6_now(), dt;
                                long r = 0;
                                do {
                                    l6_run_pool(pl, cand[best].k,
                                                cand[best].fence, Td,
                                                ain, aout, nt);
                                    ++r;
                                } while ((dt = l6_now() - t0) < 1.2e-3);
                                dt /= (double)r;
                                if (dt < tv) tv = dt;
                            }
                            disp_pool = tv / (double)nt * 1e9;
                            if (disp_pool < disp_omp * 0.98) {
                                p->pool = pl;
                                p->use_pool = 1;
                            } else
                                l6_pool_destroy(pl);
                        }
                    }
#endif
                }
            }
#ifdef _OPENMP
            /* any leftover race pool (T=1 pick, or the dispatch race did
             * not run) must not leave spinning cores behind */
            if (rpool) { l6_pool_destroy(rpool); rpool = NULL; }
#endif

            /* hand the driver the chosen configuration's steady state
             * (pool hot, every participating core in the chosen kernel's
             * licence/clock state) -- the mt analogue of l6_dwell_chosen.
             * mt_r2: routed through the CHOSEN dispatch. */
            if (use_mt && best >= 0) {
                int Td = tforce ? tforce : p->nthr;
                l6_place_mt(p, ain, aout, nt, Td);
                double until = l6_now() + 3e-3;
                do {
#ifdef _OPENMP
                    if (p->use_pool) {
                        l6_run_pool(p->pool, cand[best].k, cand[best].fence,
                                    Td, ain, aout, nt);
                        continue;
                    }
#endif
                    l6_mt_call(p, cand[best].k, cand[best].fence, Td,
                               ain, aout, nt);
                } while (l6_now() < until);
            }
        }
        if (best < 0)
            for (int c = 0; c < ncand; ++c)
                if (ok[c]) { best = c; break; }
#undef L6_RACE_CALL
#undef L6_RACE_MT
        free(ain); free(aout);
        p->t1 = p->arena; p->t2 = p->arena + L6_VD; p->placed = 0;

        if (best >= 0) {
            p->run    = cand[best].k;
            p->fence  = cand[best].fence;
            p->chosen = cand[best].nm;
            p->forced = (best == forced);
        }
        /* L6_FORCE_T overrides the team size unconditionally (node A/B
         * switch; reported with a trailing ! on nthr).  It may exceed
         * batch on purpose -- forcing T=2 at B=1 measures the pure
         * fork/join cost on an otherwise-serial transform. */
        if (tforce >= 1) {
            p->nthr = tforce;
            p->tforced = 1;
        }
        if (p->nthr < 1) p->nthr = 1;

        /* ---- mt_r2 threaded end-to-end gate (ADOPTED FROM L6_pfa mt_r1):
         * the FINAL (kernel, T, dispatch) configuration must reproduce the
         * scalar reference on an odd 61-volume batch -- uneven chunks and
         * idle threads exercised, and the pool's handoff protocol proven on
         * real data -- at rel L2 <= 1e-13, or the plan falls back to the
         * gate-proven serial path.  Insurance for the new dispatch code;
         * every kernel already passed the per-kernel gate above. ---- */
        if (p->nthr > 1) {
            long gn = 61;
            double *gin  = (double *)l6_alloc((size_t)gn * L6_VD * sizeof(double));
            double *gref = (double *)l6_alloc((size_t)gn * L6_VD * sizeof(double));
            double *ggot = (double *)l6_alloc((size_t)gn * L6_VD * sizeof(double));
            int pass = 0;
            if (gin && gref && ggot) {
                uint64_t st = 0x2545F4914F6CDD1Dull;
                for (long i = 0; i < gn * L6_VD; ++i) {
                    st ^= st << 13; st ^= st >> 7; st ^= st << 17;
                    gin[i] = (double)(int64_t)(st >> 11)
                             * (1.0 / 9007199254740992.0);
                }
                l6_run_scalar(p->t1, p->t2, gin, gref, gn);
                memset(ggot, 0, (size_t)gn * L6_VD * sizeof(double));
                l6_place_mt(p, gin, ggot, gn, p->nthr);
#ifdef _OPENMP
                if (p->use_pool)
                    l6_run_pool(p->pool, p->run, p->fence, p->nthr,
                                gin, ggot, gn);
                else
#endif
                    l6_mt_call(p, p->run, p->fence, p->nthr, gin, ggot, gn);
                double e = 0.0, nrm = 0.0;
                for (long i = 0; i < gn * L6_VD; ++i) {
                    double d = ggot[i] - gref[i];
                    e += d * d; nrm += gref[i] * gref[i];
                }
                pass = (sqrt(e) / (sqrt(nrm) + 1e-300)) < 1e-13;
            }
            free(gin); free(gref); free(ggot);
            if (!pass) {
                p->nthr = 1;
#ifdef _OPENMP
                if (p->use_pool) {
                    l6_pool_destroy((l6_pool *)p->pool);
                    p->pool = NULL;
                    p->use_pool = 0;
                }
#endif
            }
            p->placed = 0;   /* first execute re-places for the real buffers */
        }

        /* ---- in-plan B=1 discriminator (panel_r10, see l6_ab1's comment):
         * fused (ascending x-pass) vs fused_zp (zp-outer).  mt_r1: runs
         * only when batch==1 (its question is the serial B=1 pick; at
         * batch>1 the 17 ms buys nothing the team race does not). ---- */
        if (batch == 1) {
            double *bin  = (double *)l6_alloc(L6_VD * sizeof(double));
            double *bout = (double *)l6_alloc(L6_VD * sizeof(double));
            if (bin && bout) {
                uint64_t st = 0xBF58476D1CE4E5B9ull;
                for (long i = 0; i < L6_VD; ++i) {
                    st ^= st << 13; st ^= st >> 7; st ^= st << 17;
                    bin[i] = (double)(int64_t)(st >> 11)
                             * (1.0 / 9007199254740992.0);
                }
                memset(bout, 0, L6_VD * sizeof(double));
                l6_place(p, bin, bout);
                for (int c = 0; c < ncand; ++c) {
                    if (!ok[c]) continue;
                    if (strcmp(cand[c].nm, "fused") == 0)
                        ab_f  = l6_ab1(cand[c].k, cand[c].fence,
                                       p->t1, p->t2, bin, bout);
                    else if (strcmp(cand[c].nm, "fused_zp") == 0)
                        ab_fx = l6_ab1(cand[c].k, cand[c].fence,
                                       p->t1, p->t2, bin, bout);
                }
                p->t1 = p->arena; p->t2 = p->arena + L6_VD; p->placed = 0;
            }
            free(bin); free(bout);
        }
        /* (mt_r1: the panel_r11 abL DRAM codelet A/B is gone -- question
         * answered on the node, see the header.) */
    }
#endif
    if (p->nthr > 1) {
        /* mt description: the pick, the team size, and the team-size race
         * curve tm=T:ns,... (ns per volume at the raced batch size; the
         * T=16-vs-32 entry is the node's one-socket-vs-UPI NUMA answer,
         * published on every leaderboard line).  A trailing ! marks an
         * L6_FORCE / L6_FORCE_T pick (not a raced one). */
        const char *dnm = "omp";
#ifdef _OPENMP
        if (p->use_pool) dnm = "pool";
#endif
        int n = snprintf(l6_desc, sizeof(l6_desc),
                 "L=6: unrolled PFA 2x3 codelet ymm, batch-parallel "
                 "contiguous chunks, per-thread NUMA-local scratch, "
                 "pool-raced; variant=%s%s nthr=%d%s disp=%s rd=%s",
                 p->chosen, p->forced ? "!" : "",
                 p->nthr, p->tforced ? "!" : "", dnm,
                 race_disp_pool ? "pool" : "omp");
        if (n > 0 && (size_t)n < sizeof(l6_desc) && disp_omp > 0.0)
            n += snprintf(l6_desc + n, sizeof(l6_desc) - (size_t)n,
                          " od=%.1f,%.1fns", disp_omp, disp_pool);
        if (n > 0 && (size_t)n < sizeof(l6_desc) && tcurve[0] > 0.0) {
            n += snprintf(l6_desc + n, sizeof(l6_desc) - (size_t)n, " tm=");
            for (int i = 0; i < L6_NTSET; ++i) {
                if (tcurve[i] <= 0.0) continue;
                if (n <= 0 || (size_t)n >= sizeof(l6_desc)) break;
                n += snprintf(l6_desc + n, sizeof(l6_desc) - (size_t)n,
                              "%s%d:%.0f", l6_tset[i] == 1 ? "" : ",",
                              l6_tset[i], tcurve[i]);
            }
            if (n > 0 && (size_t)n < sizeof(l6_desc))
                snprintf(l6_desc + n, sizeof(l6_desc) - (size_t)n, "ns");
        }
    } else {
        /* serial path (batch==1, or no OpenMP): the phase-1 report, minus
         * the deleted abL.  kclk is the one clock number still worth a
         * line; the dwell hands the driver the chosen kernel's own
         * licence/clock steady state (r8). */
        double gk = 0.0;
#ifdef L6_HAVE_AVX2
        gk = l6_kclk(p->run, p->fence, p->t1, p->t2);
        l6_spin256(0.02);
        l6_dwell_chosen(p->run, p->fence, p->t1, p->t2);
#endif
        if (gk > 0.0) {
            /* ab1 = in-plan B=1 discriminator, min ns/volume at nvol=1,
             * licence-fair: f = fused (ascending x-pass), fx = fused_zp
             * (zp-outer, L6_pfa's group order); 0.0 = not gated.
             * xod = fused_zp vs fused family-best race delta at the
             * plan's batch size, positive = zp-outer slower. */
            int n = snprintf(l6_desc, sizeof(l6_desc),
                     "L=6: unrolled PFA 2x3 codelet (48 flop/36 instr, no "
                     "twiddles), radix-2-first VD6, ymm; variant=%s%s "
                     "nthr=1 kclk=%.2fGHz ab1=f%.1f,fx%.1fns",
                     p->chosen, p->forced ? "!" : "",
                     gk, ab_f, ab_fx);
            if (have_xod && n > 0 && (size_t)n < sizeof(l6_desc))
                snprintf(l6_desc + n, sizeof(l6_desc) - (size_t)n,
                         " xod=%+.1f%%", xod);
        } else
            snprintf(l6_desc, sizeof(l6_desc),
                     "L=6: unrolled PFA 2x3 codelet, portable scalar path; "
                     "variant=%s", p->chosen);
    }
    return p;
}

/* mt_r3 fr= tick: scan the caller's page homes on the 1st and 49th
 * threaded execute and route the result through the description (the
 * instrument-through-description discipline; the 49th-call rescan catches
 * an AutoNUMA migration DURING the timed loop, which is the regime split
 * the mt_r2 VERDICT s5 wants distinguished).  ~256 read-only syscalls on
 * exactly two calls; min-of-samples scoring makes that invisible. */
static void l6_fr_tick(fft3d_plan *plan, const double *ip, double *op)
{
    ++plan->ncalls;
    if (plan->ncalls != 1 && plan->ncalls != 49) return;
    int fr = l6_scan_remote(ip, op, (long)plan->batch, plan->main_node);
    if (plan->ncalls == 1) {
        plan->fr0 = fr;
        plan->desc_off = strlen(l6_desc);
        snprintf(l6_desc + plan->desc_off, sizeof(l6_desc) - plan->desc_off,
                 " fr=%d,nb=%d", fr, plan->nb_on);
    } else if (plan->desc_off > 0 && plan->desc_off < sizeof(l6_desc) - 1) {
        snprintf(l6_desc + plan->desc_off, sizeof(l6_desc) - plan->desc_off,
                 " fr=%d/%d,nb=%d", plan->fr0, fr, plan->nb_on);
    }
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    if (plan->nthr > 1) {
        /* threaded path: contiguous chunk per thread, per-thread scratch,
         * per-thread sfence for NT kernels, no other synchronisation.
         * First call re-places every thread's 4K offsets for the caller's
         * actual buffers (pointer arithmetic only, once). */
        if (!plan->placed) {
            l6_place_mt(plan, (const double *)in, (double *)out,
                        (long)plan->batch, plan->nthr);
            plan->placed = 1;
        }
#if defined(L6_HAVE_AVX2) && defined(_OPENMP)
        if (plan->use_pool) {
            /* mt_r2 spin-pool dispatch (see l6_run_pool): same chunk split
             * as l6_mt_call, so the output is bit-identical either way */
            l6_run_pool(plan->pool, plan->run, plan->fence, plan->nthr,
                        (const double *)in, (double *)out,
                        (long)plan->batch);
            l6_fr_tick(plan, (const double *)in, (double *)out);
            return;
        }
#endif
        l6_mt_call(plan, plan->run, plan->fence, plan->nthr,
                   (const double *)in, (double *)out, (long)plan->batch);
        l6_fr_tick(plan, (const double *)in, (double *)out);
        return;
    }
    /* serial path: no parallel region at all (B=1 ships here) */
    if (!plan->placed) l6_place(plan, in, out);
    plan->run(plan->t1, plan->t2,
              (const double *)in, (double *)out, (long)plan->batch);
#ifdef L6_HAVE_AVX2
    if (plan->fence) _mm_sfence();
#endif
}

void fft3d_destroy(fft3d_plan *plan)
{
    if (!plan) return;
#if defined(L6_HAVE_AVX2) && defined(_OPENMP)
    if (plan->pool) l6_pool_destroy((l6_pool *)plan->pool);
#endif
    for (int t = 0; t < L6_MAXT; ++t) free(plan->tarena[t]);
    free(plan->arena);
    free(plan);
}
