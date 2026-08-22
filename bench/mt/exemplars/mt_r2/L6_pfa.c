/* MULTICORE (rounds mt_r1 + mt_r2)
 * --------------------------------
 * mt_r2 (this round): RACE WHAT YOU SHIP.  The 2D (kernel x T) tournament
 * now runs through the spin pool -- created before the race -- instead of
 * the OMP fork/join, because the node's 13-18 us fork tax distorted the
 * T-columns and flipped the B=4096 pick in one of three mt_r1 processes
 * (fused_sp2_pf_xa T=24, 14.4 ns/vol scored, over fused_pf_d2 T=32, 9.4).
 * The dispatch race is re-sequenced (pool timed first, then destroyed --
 * workers joined -- before OMP is timed; recreated only if it won by >2%).
 * Pool fixes: done flags padded to a true 64 B each (mt_r1's 56 B elements
 * false-shared the join line pairs), and the master prefetches all 31 flag
 * lines before its join scan so the cross-core misses overlap.
 *
 * The batch is the parallel axis: thread t gets a CONTIGUOUS chunk of volumes
 * (static split, no communication), each thread running the unmodified
 * phase-1 kernel over its chunk with its OWN scratch, first-touched on its
 * own core in fft3d_create() so it is NUMA-local.  Volumes are 3456 B =
 * 54 cache lines, so chunk boundaries never share a line (no false sharing).
 *
 * What is raced at plan time (setup is unscored): a 2D tournament over
 * (kernel x thread count) at the real batch size, T in {1,2,4,8,16,24,32}.
 * The kernel rows are the phase-1 shapes whose differences could survive 32
 * threads sharing the bus, PLUS two NT-store twins: NT was REJECTED on the
 * node single-core (a lone core is limited by fill-buffer concurrency, not
 * bandwidth), but with 32 cores the bus IS the limiter and streaming stores
 * cut the out-traffic RFO -- 1/3 of all traffic -- which is precisely the
 * regime change this race exists to detect.  Column choice is smallest-T-
 * first with a 2% takeover margin, kernel rows keep the phase-1 safest-first
 * margins, so a noisy 32-thread cell cannot steal the pick.
 *
 * B=1 stays SINGLE-THREADED by design: the whole transform is ~600 cycles
 * (~0.21 us); an empty OMP fork/join at 32 threads costs microseconds (the
 * fork=... probe in the batched descriptions measures it on the node), so
 * any split loses before it starts.  The B=1 process never spawns a thread.
 *
 * fft3d_execute() creates no threads: the pool is warmed in fft3d_create()
 * and every execute re-enters it via one parallel region (GOMP spin-wait
 * keeps workers hot between the driver's back-to-back calls).
 */
/* Carried over from the SINGLE-THREAD competition, where this file finished as
 * written below (phase-1 node: B=1 0.208 us, B=4096 0.394 us/vol, B=32768
 * 0.565 us/vol, all 1.26-1.79x ahead of the best library) -- read
 * ../PANEL_BRIEF.md, and read ../../geom/strategies/L6_pfa.md for the full
 * history of how this kernel got here.
 */
/* L6_pfa.c -- 3D complex forward DFT of a 6x6x6 cube, Good-Thomas / prime-factor.
 *
 * TECHNIQUE
 * ---------
 * Row-column 3D transform (three axis passes), with every 6-point line transform
 * done by the Good-Thomas / prime-factor algorithm for 6 = 2*3.  Because 2 and 3
 * are coprime the index map
 *
 *      n = (3*n1 + 4*n2) mod 6,   k = (3*k1 + 4*k2) mod 6,
 *      n1,k1 in Z_2,  n2,k2 in Z_3
 *
 * makes  W6^(n k) = (-1)^(n1 k1) * W3^(-n2 k2)  exactly, so
 *
 *      X[3k1+4k2] = sum_{n1} (-1)^(n1 k1) * A[n1][k2],
 *      A[n1][k2]  = sum_{n2} x[(3 n1 + 4 n2) mod 6] * W3^(-n2 k2).
 *
 * There is NO twiddle-factor stage between the radix-3 and the radix-2 step --
 * that is the whole point of PFA versus Cooley-Tukey.  Both index maps are the
 * SAME involution p = [0,4,2,3,1,5] (it only swaps 1 <-> 4), so in a fully
 * unrolled codelet the permutation costs zero instructions: it is just which
 * register goes into which slot.  No index table exists anywhere in this file.
 *
 * OPERATION COUNT (per 6^3 volume = 216 points)
 * ---------------------------------------------
 *   DFT6 by PFA = 2*DFT3 + 3*DFT2 = 44 real flops in 36 scalar-shaped FP
 *   instructions; 3 axes * 36 lines = 4752 flops / volume; vectorised at
 *   2 complex per ymm that is 972 ymm FP instructions + 108 in-codelet
 *   vpermilpd per volume.  This matches FFTW's n1_6 codelet accounting and is
 *   the Good-Thomas optimum -- the arithmetic has been closed since round 1;
 *   every change since is scheduling, addresses and prefetch.
 *
 * TWO CODELET FACTORIZATIONS -- SETTLED ON THE NODE in panel_r9
 * --------------------------------------------------------------
 *   DFT6V  = radix-3-first (mine since round 1): two conjugate DFT3s over the
 *            CRT groups (x0,x4,x2), (x3,x1,x5), then 6 add/sub joins at the
 *            END feeding the stores.
 *   DFT6V2 = radix-2-first -- ADOPTED FROM L6_unrolled's VD6: the three DFT2s
 *            (x0+-x3, x4+-x1, x2+-x5) run FIRST, then two conjugate DFT3s
 *            whose FMA results feed the stores DIRECTLY.  Identical count
 *            (18 FP = 12 add/sub + 6 fma, + 2 vpermilpd), same depth-4
 *            critical path, different dependency graph: the join is at the
 *            top instead of the bottom, so the last ops before each store are
 *            FMAs instead of adds.
 *   r9 NODE VERDICT: on Cascade Lake store-feeding FMAs beat store-feeding
 *   adds by 3-6% at identical arithmetic -- fused_d2 took B=1 3/3 (0.2205 ->
 *   0.2068 median, -6.2%), fused_pf_xa_d2 took B=64 3/3 (-3.0%), and
 *   L6_unrolled's same-graph kernel holds B=32768 by ~1.4%.  panel_r10
 *   therefore FLIPS THE INCUMBENCY: each _d2 twin now precedes its
 *   radix-3-first parent in safest-first order (sub-margin deltas keep d2
 *   instead of losing it), except the sp2 pair where d2 composed badly with
 *   the interleave (wallaby r9: -3.3%; node: never picked).
 *
 * X-PASS GROUP ORDER (zp-outer vs ascending) -- REOPENED BY THE r10 NODE
 * ----------------------------------------------------------------------
 *   r10 said the preference is REGIME-DEPENDENT: L6_unrolled's fused_zp_pf
 *   (zp-outer + d2 + pfT0, structurally identical to my fused_pf_d2) took
 *   B=4096 outright at 0.3826 vs my ascending pick's 0.3926 (disjoint,
 *   -2.6%), their ab1 read zp 1-3% faster at nvol=1 in all three processes,
 *   while my ascending fused_pfw_xa_d2 took B=32768 (0.5540 vs their
 *   0.5627).  panel_r11 therefore (a) gives fused_pf_d2 incumbency over
 *   fused_pf_xa_d2, and (b) races bit-identical x-order twins at a reduced
 *   1.0% takeover margin (ADOPTED FROM L6_unrolled r10) so the node can
 *   resolve the order per batch size instead of the table deciding forever.
 *
 * VECTORISATION -- interleaved complex, 2 complex per ymm, no deinterleave
 * ------------------------------------------------------------------------
 * One ymm holds 2 complex = 2 independent lines of the current axis.  The x
 * and y passes are transpose-free (lanes = 2 adjacent z); only the z pass
 * changes lane axis, with 6 vperm2f128 in + 6 out per pencil pair, provably
 * minimal (see strategies/L6_pfa.md round 1).  Every access is 32B aligned.
 * The bulk of the file is 256-bit; from AVX-512 it takes the 32 evex ymm
 * registers (the fused variant needs ~26 live vectors).
 *
 * 512-BIT: CLOSED in panel_r7 (zero picks in eight cells at a measured equal
 * licence clock, kclk = 2.89 GHz).  PER-VOLUME SCRATCH ROTATION (_rot):
 * CLOSED in panel_r8 (zero picks in eight cells; the per-volume alias
 * phase-lock is a null on CLX -- demand-side 4K replays at these strides are
 * below the 1.5% margin).  Both kernel families are deleted; the r7/r8
 * records carry the falsifications.
 *
 * CLOCK: settled panel_r7.  kclk = 2.89 GHz (the ymm kernels hold the AVX2-
 * heavy licence).  B=1 is now 0.2068 us = ~598 cycles = 1.23x the 486-cycle
 * FP-port floor, with three node processes agreeing to 0.03%.
 *
 * BOUNDARY PROBES (panel_r9; question SETTLED on the node -- kept as a
 * regression check, ~40 ms of unscored setup)
 * --------------------------------------------------------------------------
 * Four unraceable probe kernels are timed at nb=1 (round-robin, min), all in
 * the plain fused shape:
 *   bf  = the real fused kernel: x: in->t1, then fused y/z: t1->out.
 *   bx  = x-pass alone: in->t1.
 *   byz = fused y/z alone: t1->out (t1 pre-filled, L1-resident).
 *   bsp = DEPENDENCE-BROKEN twin: x-pass in->t1, fused y/z reads t2 (pre-
 *         filled with the same x-pass output).  Identical instruction stream
 *         to bf -- only the cross-pass store->load forwarding is severed.
 * Reported in the description as bf/bx/byz/bsp in ns.  r9 NODE READING:
 * bf - bsp = +0.1..+0.6 ns (the t1 joint is FREE, against a pre-registered
 * 45-55 ns criterion) and bx + byz ~= bf (no fusion-scheduling deficit).
 * That was the seventh and last named B=1 mechanism; per the r9 VERDICT,
 * B=1 is closed at 1.23x floor and gets no further kernels.
 *
 * KERNEL SHAPES, raced at plan time (setup is not scored)
 * -------------------------------------------------------
 * Grid: the same 13 kernels as panel_r9 (no additions, no deletions --
 * .text is byte-identical; only the race table's ORDER changed, see the
 * incumbency flip above).
 *   3pass    x: in->t1, y: t1->t2, z: t2->out.  Safety representative.
 *   fused    x: in->t1, then y and z fused over one 6x6 (y,z) plane held in
 *            18 ymm (+ temps ~26 live): needs the target's 32 evex ymm.
 *            The node's B=1 pick seven rounds running (zp-outer x order).
 *   *_xa     x-pass in strictly ascending address order (ADOPTED FROM
 *            L6_unrolled, r6): node-picked at B>=64 in r7/r8; took B=4096.
 *   fused_sp2 software-pipelined fused stage, plane registers double-
 *            buffered, plane-pair loop kept rolled (DSB-resident).
 *            Node-picked at B=64 in r5 (pf) and r7 (pf_xa, 2 of 3 runs).
 *   *_d2     NEW panel_r9: radix-2-first codelet twin (see above).
 * prefetch hooks, one per x-pass group (3 cache lines each, 54 lines = the
 *            whole next volume, ~1 volume of lead):
 *   pf   = prefetcht0 of the next volume's INPUT
 *   w    = prefetchw of the next volume's OUTPUT -- write-intent prefetch,
 *          ADOPTED FROM L6_unrolled (panel_r3): with NT rejected on the node,
 *          every output line pays a write-allocate RFO; prefetchw issues that
 *          RFO one volume early, off the critical path.
 *
 * 64-BYTE KERNEL PINNING (panel_r8, ADOPTED FROM L6_unrolled r6): every raced
 * kernel entry is __attribute__((aligned(64))).  Kept -- it is cheap and
 * removes one source of process-luck variance -- but note it FAILED its own
 * r8 test as an explanation of the B=64 gap (0.228 median, unchanged).
 *
 * CLOCK PROBES (panel_r6; node-settled r7, kept as a regression check)
 * ---------------------------------------------------------------------
 * Four sustained-clock measurements at the end of fft3d_create() (unscored),
 * reported in fft3d_description(): clkS256 (serial ymm chain), clkD256
 * (12 independent ymm chains), clkS512 (serial zmm chain), and kclk (dwell
 * in the CHOSEN kernel, then read a short sparse chain: the licence the real
 * kernel establishes).  Node r7/r8: 3.89 / 2.89 / 2.89 / 2.89 GHz.
 *
 * PLAN-TIME TOURNAMENT
 * --------------------
 * Every candidate is validated against the scalar reference (rel L2 <= 1e-13
 * on a small batch; a miscompiled variant can never be selected), then raced
 * round-robin (drift-immune, per-candidate minimum) on a dummy buffer of the
 * real batch size truncated at 16384 volumes = 113 MiB, so a DRAM-bound real
 * batch is raced in a DRAM-bound arena on both wallaby (60 MiB L3) and the
 * node (22 MiB).  Safest-first ordering with a 1.5% takeover margin and the
 * reference minimum is tracked even when the incumbent survives, so a chain
 * of sub-margin steps cannot drift the pick.  The chosen kernel's name is
 * spliced into fft3d_description() so the leaderboard records which kernel
 * actually ran on the node.
 *
 * LICENCE HYGIENE (panel_r7; lessons from L6_unrolled r6 + L17_rader r5):
 * (1) ~100 ms of dense ymm FMA before the race so round 0 is not ranked on a
 * ramping clock; (2) each candidate runs UNTIMED for ~0.5 ms immediately
 * before each timed slice, so a candidate is always measured in its own
 * steady licence/clock state; (3) create() ends by dwelling ~3 ms in the
 * CHOSEN kernel, so the driver is handed a core in the scored kernel's own
 * steady state, never a probe's.
 *
 * 4K-ALIASING DEFENCE (from L6_unrolled round 1: +22% for an unlucky malloc)
 * --------------------------------------------------------------------------
 * A store->load pair whose addresses agree mod 4096 replays the load.  The
 * two scratch volumes are carved out of a 4 KiB-oversized arena and placed at
 * first execute (when in/out are known) to maximise the minimum cyclic
 * distance mod 4096 of every cross-buffer store->load base delta the kernels
 * produce (t1-in, out-t1, out-t2; t2-t1 is fixed at 3456 = 640 clear).
 * Addresses only; arithmetic, output and repeatability are untouched.
 *
 * Development hooks (no-ops in the graded build): L6_VERBOSE as a compile
 * define or environment variable prints the race table; -DL6_FORCE_VARIANT=n
 * pins candidate n (-1 = scalar reference path); L6_FORCE=<name> (env var,
 * adopted from L6_unrolled's r5 switch) pins a candidate BY NAME, skips the
 * race, and reports "variant=<name>!" -- the bang marks a forced pick.  This
 * is for the monitor's perf-stat A/Bs; forced picks still pass the
 * correctness gate.
 *
 * ASSUMPTIONS: L == 6 only; in/out distinct and 64B aligned (the driver
 * guarantees both); single-threaded; the plan's scratch is not re-entrant.
 */

#define _GNU_SOURCE 1     /* pthread_getaffinity_np for the spin pool */

#include "fft3d_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#endif

#define VOLC 216          /* complex per volume */
#define VOLD 432          /* doubles  per volume */
#define L6_MAXT 32        /* never take more threads than the harness grants */
/* Takeover margins are per-candidate since panel_r11 (see the cand[] table):
 * 1.5% default, 1.0% for bit-identical x-order twins. */

#if defined(__AVX2__) && defined(__FMA__)
#define L6_HAVE_AVX2 1
#include <immintrin.h>
#endif

/* ------------------------------------------------------------------ scalar --- */

#define C3 0.86602540378443864676372317075294

/* one 6-point PFA line, stride-strided, scalar reference path */
static inline void dft6_scalar(const double *x, long xs, double *y, long ys)
{
    /* a group = (0,4,2), b group = (3,1,5); conjugate-twiddle DFT3 each */
    double a0r = x[0], a0i = x[1];
    double a1r = x[4 * xs], a1i = x[4 * xs + 1];
    double a2r = x[2 * xs], a2i = x[2 * xs + 1];
    double b0r = x[3 * xs], b0i = x[3 * xs + 1];
    double b1r = x[1 * xs], b1i = x[1 * xs + 1];
    double b2r = x[5 * xs], b2i = x[5 * xs + 1];

    double t1r = a1r + a2r, t1i = a1i + a2i;
    double t2r = a1r - a2r, t2i = a1i - a2i;
    double A0r = a0r + t1r,  A0i = a0i + t1i;
    double mr  = a0r - 0.5 * t1r, mi = a0i - 0.5 * t1i;
    double A1r = mr - C3 * t2i, A1i = mi + C3 * t2r;
    double A2r = mr + C3 * t2i, A2i = mi - C3 * t2r;

    double u1r = b1r + b2r, u1i = b1i + b2i;
    double u2r = b1r - b2r, u2i = b1i - b2i;
    double B0r = b0r + u1r,  B0i = b0i + u1i;
    double nr  = b0r - 0.5 * u1r, ni = b0i - 0.5 * u1i;
    double B1r = nr - C3 * u2i, B1i = ni + C3 * u2r;
    double B2r = nr + C3 * u2i, B2i = ni - C3 * u2r;

    y[0]          = A0r + B0r;  y[1]          = A0i + B0i;
    y[3 * ys]     = A0r - B0r;  y[3 * ys + 1] = A0i - B0i;
    y[4 * ys]     = A1r + B1r;  y[4 * ys + 1] = A1i + B1i;
    y[1 * ys]     = A1r - B1r;  y[1 * ys + 1] = A1i - B1i;
    y[2 * ys]     = A2r + B2r;  y[2 * ys + 1] = A2i + B2i;
    y[5 * ys]     = A2r - B2r;  y[5 * ys + 1] = A2i - B2i;
}

static void kern_scalar(double *t1, double *t2,
                        const double *in, double *out, long nb)
{
    (void)t2;
    for (long b = 0; b < nb; ++b) {
        const double *X = in + b * VOLD;
        double *Y = out + b * VOLD;
        /* x pass: stride 36 complex = 72 doubles */
        for (int i = 0; i < 36; ++i) dft6_scalar(X + 2 * i, 72, t1 + 2 * i, 72);
        /* y pass: stride 6 complex = 12 doubles, in place */
        for (int x = 0; x < 6; ++x)
            for (int z = 0; z < 6; ++z) {
                double *p = t1 + 2 * (36 * x + z);
                dft6_scalar(p, 12, p, 12);
            }
        /* z pass: stride 1 complex = 2 doubles */
        for (int i = 0; i < 36; ++i) dft6_scalar(t1 + 12 * i, 2, Y + 12 * i, 2);
    }
}

/* -------------------------------------------------------------------- AVX2 --- */
#ifdef L6_HAVE_AVX2

#define LD(p)      _mm256_load_pd((const double *)(p))
#define STN(p, v)  _mm256_store_pd((double *)(p), (v))

/* Pin every raced kernel entry to a 64-byte boundary (adopted from
 * L6_unrolled r6): kernels run behind a function pointer, so without this
 * their placement -- and B=1/B=64 speed -- moves when unrelated code does. */
#define L6_KALIGN __attribute__((aligned(64)))

#define VSET const __m256d _half = _mm256_set1_pd(0.5);                  \
             const __m256d _cn   = _mm256_set_pd(C3, -C3, C3, -C3)

/* 6-point PFA on two interleaved complex lanes per register, radix-3 first.
 * 18 FP instructions (6 of them fma/fnmadd) + 2 vpermilpd.  In-place safe. */
#define DFT6V(v0, v1, v2, v3, v4, v5, o0, o1, o2, o3, o4, o5)            \
    do {                                                                 \
        __m256d _t1 = _mm256_add_pd((v4), (v2));                         \
        __m256d _t2 = _mm256_sub_pd((v4), (v2));                         \
        __m256d _u1 = _mm256_add_pd((v1), (v5));                         \
        __m256d _u2 = _mm256_sub_pd((v1), (v5));                         \
        __m256d _A0 = _mm256_add_pd((v0), _t1);                          \
        __m256d _B0 = _mm256_add_pd((v3), _u1);                          \
        __m256d _m  = _mm256_fnmadd_pd(_half, _t1, (v0));                \
        __m256d _n  = _mm256_fnmadd_pd(_half, _u1, (v3));                \
        __m256d _s  = _mm256_permute_pd(_t2, 0x5);                       \
        __m256d _r  = _mm256_permute_pd(_u2, 0x5);                       \
        __m256d _A1 = _mm256_fmadd_pd(_cn, _s, _m);                      \
        __m256d _A2 = _mm256_fnmadd_pd(_cn, _s, _m);                     \
        __m256d _B1 = _mm256_fmadd_pd(_cn, _r, _n);                      \
        __m256d _B2 = _mm256_fnmadd_pd(_cn, _r, _n);                     \
        (o0) = _mm256_add_pd(_A0, _B0);                                  \
        (o3) = _mm256_sub_pd(_A0, _B0);                                  \
        (o4) = _mm256_add_pd(_A1, _B1);                                  \
        (o1) = _mm256_sub_pd(_A1, _B1);                                  \
        (o2) = _mm256_add_pd(_A2, _B2);                                  \
        (o5) = _mm256_sub_pd(_A2, _B2);                                  \
    } while (0)

/* Radix-2-first twin (factorization ADOPTED FROM L6_unrolled's VD6): the
 * three DFT2s run first, by linearity DFT3(a)+-DFT3(b) = DFT3(a+-b), so the
 * two conjugate DFT3s' FMA results feed the stores directly.  Same count
 * (12 add/sub + 6 fma + 2 vpermilpd), same depth-4 path, different graph:
 * the 6-way join is at the TOP, and the last op before every store is an
 * FMA instead of an add.  In-place safe (all v reads happen first). */
#define DFT6V2(v0, v1, v2, v3, v4, v5, o0, o1, o2, o3, o4, o5)           \
    do {                                                                 \
        __m256d _s0 = _mm256_add_pd((v0), (v3));                         \
        __m256d _d0 = _mm256_sub_pd((v0), (v3));                         \
        __m256d _s1 = _mm256_add_pd((v4), (v1));                         \
        __m256d _d1 = _mm256_sub_pd((v4), (v1));                         \
        __m256d _s2 = _mm256_add_pd((v2), (v5));                         \
        __m256d _d2 = _mm256_sub_pd((v2), (v5));                         \
        __m256d _t1 = _mm256_add_pd(_s1, _s2);                           \
        __m256d _t2 = _mm256_sub_pd(_s1, _s2);                           \
        __m256d _u1 = _mm256_add_pd(_d1, _d2);                           \
        __m256d _u2 = _mm256_sub_pd(_d1, _d2);                           \
        __m256d _m  = _mm256_fnmadd_pd(_half, _t1, _s0);                 \
        __m256d _n  = _mm256_fnmadd_pd(_half, _u1, _d0);                 \
        __m256d _p  = _mm256_permute_pd(_t2, 0x5);                       \
        __m256d _q  = _mm256_permute_pd(_u2, 0x5);                       \
        (o0) = _mm256_add_pd(_s0, _t1);                                  \
        (o3) = _mm256_add_pd(_d0, _u1);                                  \
        (o4) = _mm256_fmadd_pd (_cn, _p, _m);                            \
        (o2) = _mm256_fnmadd_pd(_cn, _p, _m);                            \
        (o1) = _mm256_fmadd_pd (_cn, _q, _n);                            \
        (o5) = _mm256_fnmadd_pd(_cn, _q, _n);                            \
    } while (0)

#define LO2(a, b) _mm256_permute2f128_pd((a), (b), 0x20)
#define HI2(a, b) _mm256_permute2f128_pd((a), (b), 0x31)

/* Prefetch hooks, one per x-pass group.  Prefetch never faults, so running
 * one volume past the end of the batch is safe.  The W hook (write-intent
 * prefetch of the next volume's output) is adopted from L6_unrolled's
 * panel_r3 round: __builtin_prefetch(p,1,3) emits prefetchw on any PRFCHW
 * machine (Cascade Lake and Sapphire Rapids both). */
#define PF_NONE(SRC, OUT, g)  do { } while (0)
#define PF_IN(SRC, g, HINT)                                              \
    do {                                                                 \
        const char *_pf = (const char *)((SRC) + VOLD) + 192 * (g);      \
        _mm_prefetch(_pf,       HINT);                                   \
        _mm_prefetch(_pf + 64,  HINT);                                   \
        _mm_prefetch(_pf + 128, HINT);                                   \
    } while (0)
#define PF_OW(OUT, g)                                                    \
    do {                                                                 \
        const char *_pw = (const char *)((OUT) + VOLD) + 192 * (g);      \
        __builtin_prefetch(_pw,       1, 3);                             \
        __builtin_prefetch(_pw + 64,  1, 3);                             \
        __builtin_prefetch(_pw + 128, 1, 3);                             \
    } while (0)
#define PF_T0(SRC, OUT, g)   PF_IN(SRC, g, _MM_HINT_T0)
#define PF_T0W(SRC, OUT, g)  do { PF_IN(SRC, g, _MM_HINT_T0); PF_OW(OUT, g); } while (0)

/* Every pass macro takes the codelet macro name as its last parameter (CD =
 * DFT6V or DFT6V2); the preprocessed output for CD = DFT6V is identical to
 * the pre-r9 file, so the incumbent kernels' codegen is untouched. */

/* x pass: axis stride 72 doubles, lanes = 2 adjacent z.  zp-outer/y-inner
 * order (kept from round 1, measured +0.6%: pass 2's first groups need all
 * six y of a z-pair, which this order finishes 10 groups sooner than the
 * flat ascending order).  The prefetch index 6*zp+y still walks the next
 * volume's 54 lines in strictly ascending address order. */
#define PASS_X(SRC, DST, OUT, PF, CD)                                    \
    do {                                                                 \
        for (int zp = 0; zp < 3; ++zp)                                   \
            for (int y = 0; y < 6; ++y) {                                \
                const double *p = (SRC) + 12 * y + 4 * zp;               \
                double *q = (DST) + 12 * y + 4 * zp;                     \
                PF(SRC, OUT, 6 * zp + y);                                \
                __m256d o0, o1, o2, o3, o4, o5;                          \
                CD(LD(p), LD(p + 72), LD(p + 144), LD(p + 216),          \
                   LD(p + 288), LD(p + 360), o0, o1, o2, o3, o4, o5);    \
                STN(q, o0); STN(q + 72, o1); STN(q + 144, o2);           \
                STN(q + 216, o3); STN(q + 288, o4); STN(q + 360, o5);    \
            }                                                            \
    } while (0)

/* x pass, ascending-address twin (adopted from L6_unrolled, panel_r6): the
 * 18 groups walk offsets 4g, g = 0..17, i.e. loads and stores in strictly
 * ascending 32B steps -- friendlier to the node's L2 streamer at streaming
 * batch sizes; node-picked at B>=64 in r7/r8. */
#define PASS_X_A(SRC, DST, OUT, PF, CD)                                  \
    do {                                                                 \
        for (int g = 0; g < 18; ++g) {                                   \
            const double *p = (SRC) + 4 * g;                             \
            double *q = (DST) + 4 * g;                                   \
            PF(SRC, OUT, g);                                             \
            __m256d o0, o1, o2, o3, o4, o5;                              \
            CD(LD(p), LD(p + 72), LD(p + 144), LD(p + 216),              \
               LD(p + 288), LD(p + 360), o0, o1, o2, o3, o4, o5);        \
            STN(q, o0); STN(q + 72, o1); STN(q + 144, o2);               \
            STN(q + 216, o3); STN(q + 288, o4); STN(q + 360, o5);        \
        }                                                                \
    } while (0)

/* y pass: axis stride 12 doubles, lanes = 2 adjacent z.  SRC == DST is the
 * in-place shape (all 6 loads of a group precede its stores). */
#define PASS_Y(SRC, DST, CD)                                             \
    do {                                                                 \
        for (int x = 0; x < 6; ++x)                                      \
            for (int zp = 0; zp < 3; ++zp) {                             \
                const double *p = (SRC) + 72 * x + 4 * zp;               \
                double *q = (DST) + 72 * x + 4 * zp;                     \
                __m256d o0, o1, o2, o3, o4, o5;                          \
                CD(LD(p), LD(p + 12), LD(p + 24), LD(p + 36),            \
                   LD(p + 48), LD(p + 60), o0, o1, o2, o3, o4, o5);      \
                STN(q, o0); STN(q + 12, o1); STN(q + 24, o2);            \
                STN(q + 36, o3); STN(q + 48, o4); STN(q + 60, o5);       \
            }                                                            \
    } while (0)

/* z pass: axis contiguous; lanes = 2 adjacent y via in-register transpose.
 * The six 32B stores per pair cover exactly three whole 64B lines in
 * ascending order. */
#define PASS_Z(SRC, DST, ST, CD)                                         \
    do {                                                                 \
        for (int x = 0; x < 6; ++x)                                      \
            for (int yp = 0; yp < 3; ++yp) {                             \
                const double *p = (SRC) + 72 * x + 24 * yp;              \
                double *q = (DST) + 72 * x + 24 * yp;                    \
                __m256d r0 = LD(p),      r1 = LD(p + 4),  r2 = LD(p + 8);\
                __m256d r3 = LD(p + 12), r4 = LD(p + 16), r5 = LD(p + 20);\
                __m256d w0, w1, w2, w3, w4, w5;                          \
                CD(LO2(r0, r3), HI2(r0, r3), LO2(r1, r4), HI2(r1, r4),   \
                   LO2(r2, r5), HI2(r2, r5), w0, w1, w2, w3, w4, w5);    \
                ST(q,      LO2(w0, w1)); ST(q + 4,  LO2(w2, w3));        \
                ST(q + 8,  LO2(w4, w5)); ST(q + 12, HI2(w0, w1));        \
                ST(q + 16, HI2(w2, w3)); ST(q + 20, HI2(w4, w5));        \
            }                                                            \
    } while (0)

/* y and z fused over one 6x6 (y,z) plane held in 18 ymm (+~8 temps). */
#define FUSED_YZ(SRC, DST, ST, CD)                                       \
    do {                                                                 \
        for (int x = 0; x < 6; ++x) {                                    \
            const double *p = (SRC) + 72 * x;                            \
            double *q = (DST) + 72 * x;                                  \
            __m256d P[18];   /* P[3*y + zp] */                           \
            for (int y = 0; y < 6; ++y) {                                \
                P[3 * y]     = LD(p + 12 * y);                           \
                P[3 * y + 1] = LD(p + 12 * y + 4);                       \
                P[3 * y + 2] = LD(p + 12 * y + 8);                       \
            }                                                            \
            for (int zp = 0; zp < 3; ++zp) {                             \
                __m256d o0, o1, o2, o3, o4, o5;                          \
                CD(P[zp], P[3 + zp], P[6 + zp], P[9 + zp],               \
                   P[12 + zp], P[15 + zp], o0, o1, o2, o3, o4, o5);      \
                P[zp] = o0; P[3 + zp] = o1; P[6 + zp] = o2;              \
                P[9 + zp] = o3; P[12 + zp] = o4; P[15 + zp] = o5;        \
            }                                                            \
            for (int yp = 0; yp < 3; ++yp) {                             \
                __m256d r0 = P[6 * yp],     r1 = P[6 * yp + 1];          \
                __m256d r2 = P[6 * yp + 2], r3 = P[6 * yp + 3];          \
                __m256d r4 = P[6 * yp + 4], r5 = P[6 * yp + 5];          \
                __m256d w0, w1, w2, w3, w4, w5;                          \
                CD(LO2(r0, r3), HI2(r0, r3), LO2(r1, r4), HI2(r1, r4),   \
                   LO2(r2, r5), HI2(r2, r5), w0, w1, w2, w3, w4, w5);    \
                double *o = q + 24 * yp;                                 \
                ST(o,      LO2(w0, w1)); ST(o + 4,  LO2(w2, w3));        \
                ST(o + 8,  LO2(w4, w5)); ST(o + 12, HI2(w0, w1));        \
                ST(o + 16, HI2(w2, w3)); ST(o + 20, HI2(w4, w5));        \
            }                                                            \
        }                                                                \
    } while (0)

/* ---- software-pipelined fused stage (panel_r5) ----
 * Same arithmetic as FUSED_YZ, but the plane registers are double-buffered
 * (P for even x, Q for odd x) and the NEXT plane's 18 loads + 3 y-DFT6Vs are
 * interleaved, by thirds, into the CURRENT plane's z-chunks.  Purpose: on the
 * node (CLX, ROB 224 uops) the ~195-uop plane body means plane x+1's work
 * cannot enter the window until plane x has nearly retired, serialising the
 * z-tail; program-order interleaving moves plane x+1 ~130 uops earlier.
 * Register budget: CUR shrinks 18->12->6 as NXT grows 6->12->18, peak ~30
 * live ymm incl. temps -- needs the 32 evex ymm the target has.
 * P[3*y + zp] indexing; a z-chunk t consumes rows y = 2t, 2t+1. */
#define FYZ_LD6(P, base, t)                                              \
    do {                                                                 \
        const double *_pl = (base) + 24 * (t);                           \
        (P)[6*(t)]   = LD(_pl);      (P)[6*(t)+1] = LD(_pl + 4);         \
        (P)[6*(t)+2] = LD(_pl + 8);  (P)[6*(t)+3] = LD(_pl + 12);        \
        (P)[6*(t)+4] = LD(_pl + 16); (P)[6*(t)+5] = LD(_pl + 20);        \
    } while (0)

#define FYZ_YDFT(P, CD)                                                  \
    do {                                                                 \
        for (int _zp = 0; _zp < 3; ++_zp) {                              \
            __m256d o0, o1, o2, o3, o4, o5;                              \
            CD((P)[_zp], (P)[3+_zp], (P)[6+_zp], (P)[9+_zp],             \
               (P)[12+_zp], (P)[15+_zp], o0, o1, o2, o3, o4, o5);        \
            (P)[_zp] = o0; (P)[3+_zp] = o1; (P)[6+_zp] = o2;             \
            (P)[9+_zp] = o3; (P)[12+_zp] = o4; (P)[15+_zp] = o5;         \
        }                                                                \
    } while (0)

#define FYZ_ZCHUNK(P, q, yp, ST, CD)                                     \
    do {                                                                 \
        __m256d r0 = (P)[6*(yp)],   r1 = (P)[6*(yp)+1], r2 = (P)[6*(yp)+2];\
        __m256d r3 = (P)[6*(yp)+3], r4 = (P)[6*(yp)+4], r5 = (P)[6*(yp)+5];\
        __m256d w0, w1, w2, w3, w4, w5;                                  \
        CD(LO2(r0, r3), HI2(r0, r3), LO2(r1, r4), HI2(r1, r4),           \
           LO2(r2, r5), HI2(r2, r5), w0, w1, w2, w3, w4, w5);            \
        double *_o = (q) + 24 * (yp);                                    \
        ST(_o,      LO2(w0, w1)); ST(_o + 4,  LO2(w2, w3));              \
        ST(_o + 8,  LO2(w4, w5)); ST(_o + 12, HI2(w0, w1));              \
        ST(_o + 16, HI2(w2, w3)); ST(_o + 20, HI2(w4, w5));              \
    } while (0)

/* consume plane xx from CUR while loading + y-transforming plane xx+1 in NXT */
#define FYZ_PLANE(CUR, NXT, xx, SRC, DST, ST, CD)                        \
    do {                                                                 \
        const double *_pn = (SRC) + 72 * ((xx) + 1);                     \
        double *_qc = (DST) + 72 * (xx);                                 \
        FYZ_ZCHUNK(CUR, _qc, 0, ST, CD); FYZ_LD6(NXT, _pn, 0);           \
        FYZ_ZCHUNK(CUR, _qc, 1, ST, CD); FYZ_LD6(NXT, _pn, 1);           \
        FYZ_ZCHUNK(CUR, _qc, 2, ST, CD); FYZ_LD6(NXT, _pn, 2);           \
        FYZ_YDFT(NXT, CD);                                               \
    } while (0)

/* Rolled software-pipelined fused stage: planes 0-3 via a 2-iteration loop
 * over P/Q pairs (body ~450 insns, above gcc's complete-peel limit, so it
 * stays a loop), then plane 4 + the tail of plane 5.  ~5.5-6.4 KB total,
 * DSB-resident on the node. */
#define FUSED_YZ_SP2(SRC, DST, ST, CD)                                   \
    do {                                                                 \
        __m256d P[18], Q[18];                                            \
        FYZ_LD6(P, (SRC), 0); FYZ_LD6(P, (SRC), 1); FYZ_LD6(P, (SRC), 2);\
        FYZ_YDFT(P, CD);                                                 \
        for (int _x2 = 0; _x2 < 4; _x2 += 2) {                           \
            FYZ_PLANE(P, Q, _x2,     SRC, DST, ST, CD);                  \
            FYZ_PLANE(Q, P, _x2 + 1, SRC, DST, ST, CD);                  \
        }                                                                \
        FYZ_PLANE(P, Q, 4, SRC, DST, ST, CD);                            \
        FYZ_ZCHUNK(Q, (DST) + 360, 0, ST, CD);                           \
        FYZ_ZCHUNK(Q, (DST) + 360, 1, ST, CD);                           \
        FYZ_ZCHUNK(Q, (DST) + 360, 2, ST, CD);                           \
    } while (0)

/* Two-scratch 3pass (adopted from L6_unrolled), kept as the family's safety
 * representative. */
#define GEN_3P(NAME, MID, STZ, PF)                                       \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,     \
                 const double *restrict in, double *restrict out, long nb)\
{                                                                        \
    VSET;                                                                \
    (void)t2;                                                            \
    for (long b = 0; b < nb; ++b) {                                      \
        const double *ip = in + b * VOLD;                                \
        double *op = out + b * VOLD;                                     \
        PASS_X(ip, t1, op, PF, DFT6V);                                   \
        PASS_Y(t1, MID, DFT6V);                                          \
        PASS_Z(MID, op, STZ, DFT6V);                                     \
    }                                                                    \
}

/* PX = PASS_X (zp-outer) or PASS_X_A (ascending); CD = codelet */
#define GEN_FU(NAME, PX, STZ, PF, CD)                                    \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,     \
                 const double *restrict in, double *restrict out, long nb)\
{                                                                        \
    VSET;                                                                \
    (void)t2;                                                            \
    for (long b = 0; b < nb; ++b) {                                      \
        const double *ip = in + b * VOLD;                                \
        double *op = out + b * VOLD;                                     \
        PX(ip, t1, op, PF, CD);                                          \
        FUSED_YZ(t1, op, STZ, CD);                                       \
    }                                                                    \
}

/* x pass + rolled software-pipelined (double-buffered plane) fused y/z */
#define GEN_FU_SP2(NAME, PX, PF, CD)                                     \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,     \
                 const double *restrict in, double *restrict out, long nb)\
{                                                                        \
    VSET;                                                                \
    (void)t2;                                                            \
    for (long b = 0; b < nb; ++b) {                                      \
        const double *ip = in + b * VOLD;                                \
        double *op = out + b * VOLD;                                     \
        PX(ip, t1, op, PF, CD);                                          \
        FUSED_YZ_SP2(t1, op, STN, CD);                                   \
    }                                                                    \
}

GEN_FU(k_fu,                 PASS_X,    STN, PF_NONE, DFT6V)
GEN_FU(k_fu_d2,              PASS_X,    STN, PF_NONE, DFT6V2)
GEN_FU(k_fu_xa,              PASS_X_A,  STN, PF_NONE, DFT6V)
GEN_3P(k_3p,                 t2,        STN, PF_NONE)
GEN_FU(k_fu_pf_xa,           PASS_X_A,  STN, PF_T0,   DFT6V)
GEN_FU(k_fu_pf_xa_d2,        PASS_X_A,  STN, PF_T0,   DFT6V2)
GEN_FU(k_fu_pf,              PASS_X,    STN, PF_T0,   DFT6V)
GEN_FU(k_fu_pf_d2,           PASS_X,    STN, PF_T0,   DFT6V2)
GEN_FU_SP2(k_fu_sp2_pf_xa,   PASS_X_A,       PF_T0,   DFT6V)
GEN_FU_SP2(k_fu_sp2_pf_xa_d2, PASS_X_A,      PF_T0,   DFT6V2)
GEN_FU(k_fu_pfw_xa,          PASS_X_A,  STN, PF_T0W,  DFT6V)
GEN_FU(k_fu_pfw_xa_d2,       PASS_X_A,  STN, PF_T0W,  DFT6V2)
GEN_FU(k_fu_pfw,             PASS_X,    STN, PF_T0W,  DFT6V)

/* NT-store twins (round mt_r1, MULTICORE ONLY -- never in the B=1 race).
 * The fused stage's six 32B stores per z-chunk cover exactly three whole
 * 64B lines in ascending order, so pairs of stream_pd stores combine to
 * full-line WC writes: out is written with NO read-for-ownership.  The
 * x-pass still stores t1 normally (L1-resident scratch).  Needs sfence
 * after the batch (fence=1 in the candidate tables). */
#define STNT(p, v) _mm256_stream_pd((double *)(p), (v))
GEN_FU(k_fu_nt_xa_d2,        PASS_X_A,  STNT, PF_NONE, DFT6V2)
GEN_FU(k_fu_pf_nt_xa_d2,     PASS_X_A,  STNT, PF_T0,   DFT6V2)

/* The AVX-512 x-pass family (fused_zx*) was DELETED in panel_r8 (falsified
 * r7: zero picks in eight cells at measured-equal licence clock); the _rot
 * per-volume scratch-rotation twins were DELETED in panel_r9 (falsified r8:
 * zero picks in eight cells).  See the r7/r8 strategy records. */

/* ---- boundary probe kernels (create-time only; never raced, never chosen).
 * All in the plain fused shape (the node's B=1 pick).  k_probe_split is the
 * dependence-broken twin of k_fu: identical instruction stream, but the
 * fused stage reads t2 (pre-filled with the same x-pass output) instead of
 * the t1 the x-pass just stored, so the cross-pass store->load joint is
 * severed while every uop, address stride and cache footprint survives. */
L6_KALIGN static void k_probe_x(double *restrict t1, double *restrict t2,
                 const double *restrict in, double *restrict out, long nb)
{
    VSET;
    (void)t2;
    for (long b = 0; b < nb; ++b)
        PASS_X(in + b * VOLD, t1, out + b * VOLD, PF_NONE, DFT6V);
    (void)out;
}

L6_KALIGN static void k_probe_yz(double *restrict t1, double *restrict t2,
                 const double *restrict in, double *restrict out, long nb)
{
    VSET;
    (void)t2; (void)in;
    for (long b = 0; b < nb; ++b)
        FUSED_YZ(t1, out + b * VOLD, STN, DFT6V);
}

L6_KALIGN static void k_probe_split(double *restrict t1, double *restrict t2,
                 const double *restrict in, double *restrict out, long nb)
{
    VSET;
    for (long b = 0; b < nb; ++b) {
        const double *ip = in + b * VOLD;
        double *op = out + b * VOLD;
        PASS_X(ip, t1, op, PF_NONE, DFT6V);
        FUSED_YZ(t2, op, STN, DFT6V);
    }
}

#endif /* L6_HAVE_AVX2 */

/* ------------------------------------------------------------------ plan --- */

typedef void (*l6_kernel)(double *t1, double *t2,
                          const double *in, double *out, long nb);

struct fft3d_plan {
    int  L;
    long batch;
    double *arena;        /* owns the scratch; t1/t2 are placed inside it */
    double *t1, *t2;      /* 432 doubles each, 64B aligned, 4K-alias-placed */
    l6_kernel run;
    int  fence;           /* nonzero if the chosen kernel uses NT stores */
    int  placed;
    const char *chosen;
    /* multicore (round mt_r1): nthreads==1 is the phase-1 path, untouched.
     * Per-thread scratch is one arena per thread, allocated AND first-touched
     * by its own thread inside fft3d_create()'s pool-warm region, so each
     * lives in that thread's socket's memory.  Arenas are separate mallocs
     * (6976 B, 64B aligned): no two threads' scratch shares a cache line. */
    int  nthreads;
    double *tarena[L6_MAXT];
    double *tt1[L6_MAXT], *tt2[L6_MAXT];
    /* dispatch: 0 = one OMP parallel region per execute; 1 = the spin-wait
     * pthread pool below (raced at plan time, pool must win by >2%) */
    int  use_pool;
    void *pool;
};

/* 4K-aliasing defence (adopted from L6_unrolled round 1: +22% at B=1 for an
 * unlucky malloc).  Maximise the minimum cyclic distance mod 4096 of the
 * cross-buffer store->load base deltas: pass 1 stores t1 / loads in; the last
 * pass stores out / loads t1 (in-place, fused) or t2 (two-scratch).
 * t2 - t1 = 3456 is fixed and 640 clear of 0 mod 4096 on its own. */
static long cyc4k(long d)
{
    d &= 4095;
    return d < 4096 - d ? d : 4096 - d;
}

static void place_scratch(fft3d_plan *p, const void *in, const void *out)
{
    long D = (long)(((uintptr_t)out - (uintptr_t)in) & 4095u);
    long bestr = 0, best = -1;
    for (long r = 0; r < 4096; r += 64) {
        long s1 = cyc4k(r);                                    /* t1 - in  */
        long s2 = cyc4k(D - r);                                /* out - t1 */
        long s3 = cyc4k(D - r - (long)(VOLD * sizeof(double)));/* out - t2 */
        long sc = s1 < s2 ? s1 : s2;
        if (s3 < sc) sc = s3;
        if (sc > best) { best = sc; bestr = r; }
    }
    long off = (long)((((uintptr_t)in + (uintptr_t)bestr
                        - (uintptr_t)p->arena) & 4095u) / sizeof(double));
    p->t1 = p->arena + off;
    p->t2 = p->t1 + VOLD;
    p->placed = 1;
}

#if defined(L6_HAVE_AVX2) && defined(_OPENMP)
/* Run (kernel, T) over the first nb volumes with the static contiguous
 * split; the SAME function serves the plan-time 2D race and fft3d_execute,
 * so what is raced is bit-for-bit what runs.  T==1 is a direct call (no
 * parallel region), matching the single-thread path's overhead.  No 4K
 * scratch placement here: per-volume store->load deltas cycle through 32
 * residues mod 4096 inside any chunk of >=32 volumes, so placement is a
 * B=1-scale effect and the threaded path never runs at B=1. */
static void l6_run_cfg(fft3d_plan *p, l6_kernel k, int fence, int T,
                       const double *in, double *out, long nb)
{
    if (T <= 1) {
        k(p->tt1[0], p->tt2[0], in, out, nb);
        if (fence) _mm_sfence();
        return;
    }
    #pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num();
        long q = nb / T, r = nb % T;
        long b0 = q * t + (t < r ? t : r);
        long n  = q + (t < r ? 1 : 0);
        if (n > 0)
            k(p->tt1[t], p->tt2[t], in + b0 * VOLD, out + b0 * VOLD, n);
        if (fence) _mm_sfence();
    }
}

/* ---- spin-wait dispatch pool (round mt_r1) ----
 * GOMP's fork/join at 32 threads measures 5-6 us on wallaby -- 15% of a
 * whole B=4096 execute.  This pool replaces it with one release-store of an
 * epoch counter (workers spin on it with pause) and 31 padded done-flags the
 * master scans, ~1-2 us round trip.  Workers are pinned to the EXACT
 * affinity masks the OMP threads report (captured inside the pool-warm
 * region), so thread t runs on the same core -- and the same socket as its
 * first-touched scratch -- under either dispatch.  Workers never sleep, so
 * the pool is only kept if it BEATS the OMP dispatch by >2% in the plan-time
 * race; otherwise it is torn down before create() returns and the scored
 * run never sees a spinning core it did not pay for.  The pool never has
 * more than the harness's 32 threads in total (master included). */
typedef struct {
    l6_kernel k;
    int fence, T;
    const double *in;
    double *out;
    long nb;
} l6_job;

struct l6_pool;
typedef struct { struct l6_pool *pl; int idx; } l6_warg;

typedef struct l6_pool {
    _Atomic long epoch;
    _Atomic int shutdown;
    l6_job job;
    fft3d_plan *plan;
    int nworkers;                    /* workers 1..nworkers; master is 0 */
    pthread_t th[L6_MAXT];
    l6_warg warg[L6_MAXT];
    /* pad to a FULL 64 B per element: mt_r1 shipped pad[48] (56 B/element),
     * so adjacent workers' done flags shared cache lines and the join's 31
     * release-stores ping-ponged.  aligned(64) on the array does not align
     * the elements -- sizeof must be 64. */
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
        if (t < jb.T) {
            long q = jb.nb / jb.T, r = jb.nb % jb.T;
            long b0 = q * t + (t < r ? t : r);
            long n  = q + (t < r ? 1 : 0);
            fft3d_plan *p = pl->plan;
            if (n > 0)
                jb.k(p->tt1[t], p->tt2[t],
                     jb.in + b0 * VOLD, jb.out + b0 * VOLD, n);
            if (jb.fence) _mm_sfence();
        }
        atomic_store_explicit(&pl->w[t].done, e, memory_order_release);
        seen = e;
    }
    return NULL;
}

static void l6_run_pool(l6_pool *pl, l6_kernel k, int fence, int T,
                        const double *in, double *out, long nb)
{
    fft3d_plan *p = pl->plan;
    pl->job.k = k; pl->job.fence = fence; pl->job.T = T;
    pl->job.in = in; pl->job.out = out; pl->job.nb = nb;
    long e = atomic_load_explicit(&pl->epoch, memory_order_relaxed) + 1;
    atomic_store_explicit(&pl->epoch, e, memory_order_release);
    long q = nb / T, r = nb % T;     /* master takes chunk 0 itself */
    long n = q + (r > 0 ? 1 : 0);
    if (n > 0) k(p->tt1[0], p->tt2[0], in, out, n);
    if (fence) _mm_sfence();
    /* EVERY worker acks every epoch (idle ones immediately), so the next
     * job write can never race a straggler's read of this one.  Prefetch
     * all 31 flag lines first (round mt_r2): each flag is Modified in its
     * worker's cache, and a bare sequential scan pays ~31 serialized
     * cross-core misses AFTER the last worker finishes; the prefetch wave
     * overlaps them in the fill buffers (L23_matrixsimd's r1 record uses
     * the same overlap argument for its flag-array barrier). */
    for (int t = 1; t <= pl->nworkers; ++t)
        _mm_prefetch((const char *)&pl->w[t].done, _MM_HINT_T0);
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

static l6_pool *l6_pool_create(fft3d_plan *p, int tmax, const cpu_set_t *sets)
{
    l6_pool *pl = NULL;
    if (posix_memalign((void **)&pl, 64, sizeof(*pl)) != 0 || !pl) return NULL;
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
        pthread_setaffinity_np(pl->th[t], sizeof(cpu_set_t), &sets[t]);
    }
    return pl;
}

/* Dispatch used by the plan-time 2D tournament (round mt_r2): T==1 is a
 * direct call (what a T=1 pick would ship as); T>1 goes through the spin
 * pool when one is alive -- the dispatch the scored run actually uses --
 * and falls back to the OMP region only if pool creation failed.  mt_r1
 * raced every cell through the OMP fork/join (fork=13-18 us on the node)
 * while the scored run then used the ~1-2 us pool: the fork tax flattened
 * the T-columns and one of three node processes picked fused_sp2_pf_xa
 * T=24 (scored 14.4 ns/vol) over fused_pf_d2 T=32 (9.4 ns/vol) -- the
 * round's entire 52.9% B=4096 leaderboard spread was that one mis-pick.
 * Race what you ship. */
static void l6_race_run(fft3d_plan *p, l6_pool *pl, l6_kernel k, int fence,
                        int T, const double *in, double *out, long nb)
{
    if (T > 1 && pl) l6_run_pool(pl, k, fence, T, in, out, nb);
    else             l6_run_cfg(p, k, fence, T, in, out, nb);
}
#endif

const char *fft3d_name(void) { return "L6_pfa"; }

/* The chosen kernel, the clock probes, and the boundary probes are formatted
 * into the description so the leaderboard records which kernel ran on the
 * node, what clock it ran at, and what the x->yz joint costs (the r8
 * VERDICT's in-plan measurement route, after L36_pfa). */
static char l6_desc[384] =
    "Good-Thomas PFA 2x3 per axis, no twiddles, 2 cplx/ymm, plan-raced; "
    "variant=auto";

const char *fft3d_description(void) { return l6_desc; }

int fft3d_supports(int L) { return L == 6; }

static int l6_verbose(void)
{
#ifdef L6_VERBOSE
    return 1;
#else
    return getenv("L6_VERBOSE") != NULL;
#endif
}

#ifdef L6_HAVE_AVX2
static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

#ifdef _OPENMP
/* Cost of one empty fork/join at T threads (us), min of 5x128.  This is the
 * floor any threaded execute pays; at B=1 it alone is ~10-20x the whole
 * transform, which is the measured reason B=1 stays single-threaded. */
static double l6_fork_us(int T)
{
    double best = 1e30;
    for (int i = 0; i < 8; ++i) {
        #pragma omp parallel num_threads(T)
        { __asm__ __volatile__("" ::: "memory"); }
    }
    for (int trial = 0; trial < 5; ++trial) {
        double t0 = now_s();
        for (int i = 0; i < 128; ++i) {
            #pragma omp parallel num_threads(T)
            { __asm__ __volatile__("" ::: "memory"); }
        }
        double e = (now_s() - t0) / 128.0 * 1e6;
        if (e < best) best = e;
    }
    return best;
}
#endif

/* ---------------------------------------------------- clock probes ---
 * All setup-time only (unscored).  Frequencies from FMA chains of known
 * cycle cost: latency 4 / throughput 2-per-cycle for 256-bit FMA on both
 * Cascade Lake and Sapphire Rapids; zmm FMA latency is also 4 on both. */

static volatile double l6_probe_sink;

static double l6_med(double *v, int n)
{
    for (int i = 1; i < n; ++i) {
        double x = v[i];
        int j = i;
        while (j > 0 && v[j - 1] > x) { v[j] = v[j - 1]; --j; }
        v[j] = x;
    }
    return v[n / 2];
}

/* keep the core busy at the non-AVX licence for ~ms milliseconds, so a
 * heavier licence from the previous probe decays before the next one */
static void l6_scalar_spin(double ms)
{
    unsigned long x = 88172645463325252ul;
    unsigned long acc = 0;
    double t0 = now_s();
    while (now_s() - t0 < ms * 1e-3) {
        for (int i = 0; i < 4096; ++i) {
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            acc += x;
        }
    }
    l6_probe_sink = (double)(acc & 0xffffu);
}

/* sparse: 1 serially dependent ymm FMA in flight = 4 cycles/iter.
 * a stays bounded (a*c + tiny with c < 1), so no overflow ever. */
static double l6_clk_s256(long iters)
{
    __m256d a = _mm256_set1_pd(1.0);
    const __m256d c = _mm256_set1_pd(0.999999999);
    const __m256d d = _mm256_set1_pd(1e-12);
    double t0 = now_s();
    for (long i = 0; i < iters; ++i)
        a = _mm256_fmadd_pd(a, c, d);
    double t = now_s() - t0;
    double out[4];
    _mm256_storeu_pd(out, a);
    l6_probe_sink = out[0];
    return 4.0 * (double)iters / t * 1e-9;
}

/* dense: 12 independent ymm FMA chains, throughput-bound at 2/cycle on
 * both machines = 6 cycles/iter.  This is the licence-saturating design
 * (L17_winograd's); enough FMA density to engage the AVX2 heavy licence. */
static double l6_clk_d256(long iters)
{
    __m256d a0 = _mm256_set1_pd(1.00), a1 = _mm256_set1_pd(1.01);
    __m256d a2 = _mm256_set1_pd(1.02), a3 = _mm256_set1_pd(1.03);
    __m256d a4 = _mm256_set1_pd(1.04), a5 = _mm256_set1_pd(1.05);
    __m256d a6 = _mm256_set1_pd(1.06), a7 = _mm256_set1_pd(1.07);
    __m256d a8 = _mm256_set1_pd(1.08), a9 = _mm256_set1_pd(1.09);
    __m256d aa = _mm256_set1_pd(1.10), ab = _mm256_set1_pd(1.11);
    const __m256d c = _mm256_set1_pd(0.999999999);
    const __m256d d = _mm256_set1_pd(1e-12);
    double t0 = now_s();
    for (long i = 0; i < iters; ++i) {
        a0 = _mm256_fmadd_pd(a0, c, d); a1 = _mm256_fmadd_pd(a1, c, d);
        a2 = _mm256_fmadd_pd(a2, c, d); a3 = _mm256_fmadd_pd(a3, c, d);
        a4 = _mm256_fmadd_pd(a4, c, d); a5 = _mm256_fmadd_pd(a5, c, d);
        a6 = _mm256_fmadd_pd(a6, c, d); a7 = _mm256_fmadd_pd(a7, c, d);
        a8 = _mm256_fmadd_pd(a8, c, d); a9 = _mm256_fmadd_pd(a9, c, d);
        aa = _mm256_fmadd_pd(aa, c, d); ab = _mm256_fmadd_pd(ab, c, d);
    }
    double t = now_s() - t0;
    __m256d s = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(a0, a1),
                                            _mm256_add_pd(a2, a3)),
                              _mm256_add_pd(_mm256_add_pd(a4, a5),
                                            _mm256_add_pd(a6, a7)));
    s = _mm256_add_pd(s, _mm256_add_pd(_mm256_add_pd(a8, a9),
                                       _mm256_add_pd(aa, ab)));
    double out[4];
    _mm256_storeu_pd(out, s);
    l6_probe_sink = out[0];
    return 6.0 * (double)iters / t * 1e-9;
}

#ifdef __AVX512F__
/* sparse zmm: 1 serially dependent 512-bit FMA in flight = 4 cycles/iter. */
static double l6_clk_s512(long iters)
{
    __m512d a = _mm512_set1_pd(1.0);
    const __m512d c = _mm512_set1_pd(0.999999999);
    const __m512d d = _mm512_set1_pd(1e-12);
    double t0 = now_s();
    for (long i = 0; i < iters; ++i)
        a = _mm512_fmadd_pd(a, c, d);
    double t = now_s() - t0;
    l6_probe_sink = _mm512_reduce_add_pd(a);
    return 4.0 * (double)iters / t * 1e-9;
}
#endif

/* kclk: the clock the CHOSEN kernel actually runs at.  Dwell ~2 ms in the
 * kernel (establishes its licence), then immediately time a short (~140 us)
 * sparse ymm chain; CLX licence transitions persist >600 us, so the short
 * chain reads the kernel's licence, not its own.  Median of 9 alternations. */
static double l6_kclk(l6_kernel run, int fence, double *t1, double *t2,
                      const double *a, double *b)
{
    double r[9];
    for (int trial = 0; trial < 9; ++trial) {
        double t0 = now_s();
        do {
            for (int i = 0; i < 256; ++i) run(t1, t2, a, b, 1);
            if (fence) _mm_sfence();
        } while (now_s() - t0 < 2e-3);
        r[trial] = l6_clk_s256(100000L);
    }
    return l6_med(r, 9);
}

/* ---- boundary probes: time the four probe kernels at nb=1, round-robin
 * with per-kernel minimum (same drift discipline as the race), ns/call.
 * Called with the race arena as in/out and the placed scratch, i.e. under
 * exactly the address conditions the scored B=1 run sees. */
static void l6_boundary_probe(double *t1, double *t2,
                              const double *a, double *b, double r_ns[4])
{
    l6_kernel pk[4] = { k_fu, k_probe_split, k_probe_x, k_probe_yz };
    /* pre-fill t1 AND t2 with real x-pass output so k_probe_yz and
     * k_probe_split read realistic (non-denormal) data */
    k_probe_x(t1, NULL, a, b, 1);
    k_probe_x(t2, NULL, a, b, 1);
    long reps = 256;
    for (;;) {                       /* calibrate on the full kernel */
        double t0 = now_s();
        for (long r = 0; r < reps; ++r) k_fu(t1, t2, a, b, 1);
        if (now_s() - t0 > 3e-4 || reps > (1L << 22)) break;
        reps *= 2;
    }
    for (int i = 0; i < 4; ++i) r_ns[i] = 1e30;
    for (int round = 0; round < 9; ++round)
        for (int i = 0; i < 4; ++i) {
            double t0 = now_s();
            for (long r = 0; r < reps; ++r) pk[i](t1, t2, a, b, 1);
            double e = (now_s() - t0) / (double)reps * 1e9;
            if (e < r_ns[i]) r_ns[i] = e;
        }
    /* the probes scribbled on t1/t2; nothing downstream reads them */
}
#endif /* L6_HAVE_AVX2 */

static void *l6_alloc(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != 6 || batch <= 0) return NULL;
    fft3d_plan *p = (fft3d_plan *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    /* 4096 B of slack for the 4K-aliasing placement, then two scratch
     * volumes = 864 doubles (the arena keeps r8's 944-double window; the
     * extra 640 B is harmless slack now that the _rot kernels are gone). */
    p->arena = (double *)l6_alloc(4096 + 944 * sizeof(double) + 64);
    if (!p->arena) { free(p); return NULL; }
    memset(p->arena, 0, 4096 + 944 * sizeof(double) + 64);
    p->t1 = p->arena;
    p->t2 = p->arena + VOLD;
    p->run = kern_scalar;
    p->fence = 0;
    p->placed = 0;
    p->chosen = "scalar";
    p->nthreads = 1;

#ifdef L6_HAVE_AVX2
    {
        /* ---- multicore setup: warm the pool, give every thread its own
         * NUMA-local scratch (allocated AND first-touched by that thread,
         * under the same close/cores binding execute will see).  batch==1
         * never spawns a thread: the fork alone costs ~10x the transform. */
        int tmax = 1, use_mt = 0;
        double fork_us = 0.0;
#ifdef _OPENMP
        static cpu_set_t omp_sets[L6_MAXT];  /* per-thread binding, captured
                                              * for the spin pool's workers */
        if (batch > 1) {
            tmax = omp_get_max_threads();
            if (tmax > L6_MAXT) tmax = L6_MAXT;
            if (tmax < 1) tmax = 1;
            if (tmax > 1) {
                int fail = 0;
                #pragma omp parallel num_threads(tmax)
                {
                    int t = omp_get_thread_num();
                    pthread_getaffinity_np(pthread_self(),
                                           sizeof(cpu_set_t), &omp_sets[t]);
                    double *ar = (double *)l6_alloc(
                        (size_t)(2 * VOLD) * sizeof(double) + 64);
                    if (ar) {
                        memset(ar, 0, (size_t)(2 * VOLD) * sizeof(double) + 64);
                        p->tarena[t] = ar;
                        p->tt1[t] = ar;
                        p->tt2[t] = ar + VOLD;
                    } else {
                        #pragma omp atomic write
                        fail = 1;
                    }
                }
                for (int t = 0; t < tmax; ++t) if (!p->tarena[t]) fail = 1;
                use_mt = !fail;
                fork_us = l6_fork_us(tmax);
            }
        }
#endif
        (void)tmax; (void)fork_us;
        /* Safest-first order, re-derived from the r10 node evidence
         * (panel_r11).  Two changes, both selection plumbing -- no kernel is
         * added, deleted, or edited, so the raced .text is unchanged:
         *
         * (1) INCUMBENCY FLIP AT THE PF SLOT: fused_pf_d2 (zp-outer x-pass)
         *     now precedes fused_pf_xa_d2 (ascending).  r10 node evidence:
         *     L6_unrolled's fused_zp_pf -- structurally identical to my
         *     fused_pf_d2 (zp order + radix-2-first codelet + pfT0) -- took
         *     B=4096 OUTRIGHT at 0.3826 vs my ascending pick's 0.3926
         *     (disjoint distributions, -2.6%), and their ab1 instrument read
         *     the zp order 1-3% faster at nvol=1 in all three node processes.
         *     My own race kept picking the ascending twin 3/3 because a
         *     sub-1.5% zp edge cannot clear the takeover margin from the
         *     trailing slot -- exactly the ordering bug my r10 record warns
         *     about.  B=1's winner (fused_d2) is already zp-outer.
         * (2) PER-CANDIDATE TAKEOVER MARGINS -- ADOPTED FROM L6_unrolled r10:
         *     an x-order twin that is provably BIT-IDENTICAL to its
         *     neighbour (same codelet, same prefetch set, only the iteration
         *     order of the 18 independent x-pass groups differs) races at a
         *     reduced 1.0% margin, because a twin<->parent mis-pick is
         *     bounded and harmless; genuinely different shapes keep 1.5%.
         *     This lets the node resolve zp-vs-ascending per batch size
         *     (r10 says the preference is regime-dependent: zp won B=1 and
         *     B=4096, my ascending+W kernel won B=32768).
         *
         * Kept from r10: every d2 twin precedes its radix-3-first parent
         * (node-proven r9: store-feeding FMAs beat store-feeding adds 3-6%);
         * fused_sp2_pf_xa keeps the lead over its d2 twin (r9: composed
         * badly, wallaby -3.3%, node never picked it); fused_pfw_xa_d2 keeps
         * the B=32768 incumbency it won in r10 (0.5540, took the cell). */
        static const struct { l6_kernel k; int fence; double mar; const char *nm; } cand[] = {
            { k_fu_d2,              0, 0.015, "fused_d2"             },
            { k_fu,                 0, 0.015, "fused"                },
            { k_fu_xa,              0, 0.010, "fused_xa"             },
            { k_3p,                 0, 0.015, "3pass"                },
            { k_fu_pf_d2,           0, 0.015, "fused_pf_d2"          },
            { k_fu_pf_xa_d2,        0, 0.010, "fused_pf_xa_d2"       },
            { k_fu_pf_xa,           0, 0.015, "fused_pf_xa"          },
            { k_fu_pf,              0, 0.010, "fused_pf"             },
            { k_fu_sp2_pf_xa,       0, 0.015, "fused_sp2_pf_xa"      },
            { k_fu_sp2_pf_xa_d2,    0, 0.015, "fused_sp2_pf_xa_d2"   },
            { k_fu_pfw_xa_d2,       0, 0.015, "fused_pfw_xa_d2"      },
            { k_fu_pfw_xa,          0, 0.015, "fused_pfw_xa"         },
            { k_fu_pfw,             0, 0.010, "fused_pfw"            },
        };
        enum { NCAND = (int)(sizeof(cand) / sizeof(cand[0])) };
        int ok[NCAND];
        for (int c = 0; c < NCAND; ++c) ok[c] = 0;

        /* MULTICORE candidate rows (round mt_r1): the phase-1 shapes whose
         * differences could plausibly survive 32 threads sharing the bus
         * (prefetch set, x order, sp2), plus the NT-store twins -- see the
         * header for why NT re-enters in the multicore regime.  Safest-first
         * order and per-row margins follow the phase-1 discipline. */
        static const struct { l6_kernel k; int fence; double mar; const char *nm; } mtc[] = {
            { k_fu_pf_d2,        0, 0.015, "fused_pf_d2"        },
            { k_fu_pf_xa_d2,     0, 0.010, "fused_pf_xa_d2"     },
            { k_fu_pfw_xa_d2,    0, 0.015, "fused_pfw_xa_d2"    },
            { k_fu_d2,           0, 0.015, "fused_d2"           },
            { k_fu_sp2_pf_xa,    0, 0.015, "fused_sp2_pf_xa"    },
            { k_fu_nt_xa_d2,     1, 0.015, "fused_nt_xa_d2"     },
            { k_fu_pf_nt_xa_d2,  1, 0.010, "fused_pf_nt_xa_d2"  },
        };
        enum { NMT = (int)(sizeof(mtc) / sizeof(mtc[0])) };
        int okm[NMT];
        for (int c = 0; c < NMT; ++c) okm[c] = 0;

        /* ---- correctness gate: reproduce the scalar reference bit-closely
         * on random data or never be timed at all. ---- */
        long nval = batch < 4 ? batch : 4;
        double *vin  = (double *)l6_alloc((size_t)nval * VOLD * sizeof(double));
        double *vref = (double *)l6_alloc((size_t)nval * VOLD * sizeof(double));
        double *vgot = (double *)l6_alloc((size_t)nval * VOLD * sizeof(double));
        if (vin && vref && vgot) {
            unsigned s = 12345u;
            for (long i = 0; i < nval * VOLD; ++i) {
                s = s * 1664525u + 1013904223u;
                vin[i] = (double)(int)(s >> 8) * 1e-9;
            }
            kern_scalar(p->t1, p->t2, vin, vref, nval);
            double nrm = 0.0;
            for (long i = 0; i < nval * VOLD; ++i) nrm += vref[i] * vref[i];
            for (int c = 0; c < NCAND; ++c) {
                memset(vgot, 0, (size_t)nval * VOLD * sizeof(double));
                cand[c].k(p->t1, p->t2, vin, vgot, nval);
                if (cand[c].fence) _mm_sfence();
                double e = 0.0;
                for (long i = 0; i < nval * VOLD; ++i) {
                    double d = vgot[i] - vref[i];
                    e += d * d;
                }
                ok[c] = nrm > 0.0 && e <= 1e-26 * nrm;   /* rel L2 <= 1e-13 */
            }
            for (int c = 0; c < NMT; ++c) {
                memset(vgot, 0, (size_t)nval * VOLD * sizeof(double));
                mtc[c].k(p->t1, p->t2, vin, vgot, nval);
                if (mtc[c].fence) _mm_sfence();
                double e = 0.0;
                for (long i = 0; i < nval * VOLD; ++i) {
                    double d = vgot[i] - vref[i];
                    e += d * d;
                }
                okm[c] = nrm > 0.0 && e <= 1e-26 * nrm;
            }
        }
        free(vin); free(vref); free(vgot);

        /* ---- race arena, common to both arms: (a truncation of) the real
         * batch size.  Single-thread cap 16384 volumes = 113 MiB: DRAM on
         * wallaby (60 MiB L3) and the node (22 MiB), so a DRAM-bound real
         * batch is raced in a DRAM-bound arena (cap from L6_unrolled r2).
         * The MULTICORE cap is 65536: 32 threads see L3 + 32 L2s of
         * aggregate cache (~124 MiB on wallaby, ~54 MiB on the node), and a
         * 113 MiB arena raced 2.7x FASTER than the real 452 MiB batch ran
         * (measured mt_r1 on wallaby: 0.0159 vs 0.0434 us/vol) -- a capped
         * race would crown a cache-regime kernel for a DRAM-regime problem.
         * Filled SERIALLY by this thread, exactly as the driver first-touches
         * the caller's buffers -- the NUMA page placement the race sees is
         * the placement the scored run sees. ---- */
        long nb = batch;
        long nbcap = use_mt ? 65536 : 16384;
        if (nb > nbcap) nb = nbcap;
        size_t nd = (size_t)nb * VOLD;
        double *a = (double *)l6_alloc((nd + VOLD) * sizeof(double));
        double *b = (double *)l6_alloc(nd * sizeof(double));
        if (a && b) {
            unsigned s = 12345u;
            for (size_t i = 0; i < nd + VOLD; ++i) {
                s = s * 1664525u + 1013904223u;
                a[i] = (double)(int)(s >> 8) * 1e-9;
            }
            memset(b, 0, nd * sizeof(double));
            place_scratch(p, a, b);      /* race with realistically-placed scratch */
        }

        if (!use_mt) {
        /* ============== single-thread arm: phase 1, verbatim ============ */
        /* Forced pick by NAME (env L6_FORCE, adopted from L6_unrolled's r5
         * switch): skips the race, still requires the correctness gate, and
         * is marked with '!' in the description so it can never be mistaken
         * for a tournament result.  For the monitor's perf-stat A/Bs. */
        int forced = -1;
        {
            const char *fn = getenv("L6_FORCE");
            if (fn)
                for (int c = 0; c < NCAND; ++c)
                    if (ok[c] && strcmp(fn, cand[c].nm) == 0) { forced = c; break; }
        }
        int best = -1;
        if (a && b && forced < 0) {
            /* settle spin: ~100 ms of dense ymm FMA so the calibration and
             * round 0 are not ranked on a ramping clock (adopted from
             * L17_rader r5 via L6_unrolled r6) */
            {
                double t0 = now_s();
                while (now_s() - t0 < 0.1) (void)l6_clk_d256(400000L);
            }
            long reps = 1;
            int cal = 0;
            while (!ok[cal] && cal < NCAND - 1) ++cal;
            if (ok[cal]) {
                for (;;) {               /* calibrate to ~1 ms per timing */
                    double t = now_s();
                    for (long r = 0; r < reps; ++r)
                        cand[cal].k(p->t1, p->t2, a, b, nb);
                    if (cand[cal].fence) _mm_sfence();
                    double e = now_s() - t;
                    if (e > 1e-3 || reps > (1L << 22)) break;
                    reps *= 2;
                }
            }
            double bt[NCAND];
            for (int c = 0; c < NCAND; ++c) {
                bt[c] = 1e30;
                if (ok[c]) {
                    cand[c].k(p->t1, p->t2, a, b, nb);          /* warm */
                    if (cand[c].fence) _mm_sfence();
                }
            }
            /* Round-robin so machine drift hits every candidate equally;
             * per-candidate minimum is the noise-robust statistic. */
            for (int round = 0; round < 7; ++round) {
                for (int c = 0; c < NCAND; ++c) {
                    if (!ok[c]) continue;
                    /* licence dwell: run the candidate untimed for ~0.5 ms
                     * so it is timed in its own steady licence/clock state
                     * and cannot poison the next candidate's slice (CLX
                     * licence transitions persist >600 us) */
                    double td = now_s();
                    do {
                        cand[c].k(p->t1, p->t2, a, b, nb);
                        if (cand[c].fence) _mm_sfence();
                    } while (now_s() - td < 5e-4);
                    double t = now_s();
                    for (long r = 0; r < reps; ++r)
                        cand[c].k(p->t1, p->t2, a, b, nb);
                    if (cand[c].fence) _mm_sfence();
                    double e = now_s() - t;
                    if (e < bt[c]) bt[c] = e;
                }
            }
            /* Safest-first with a PER-CANDIDATE takeover margin (1.5%
             * default, 1.0% for bit-identical x-order twins -- adopted from
             * L6_unrolled r10); the reference time tracks the true minimum
             * even when the incumbent survives, so a chain of sub-margin
             * steps cannot drift the pick. */
            double bestt = 1e30;
            for (int c = 0; c < NCAND; ++c) {
                if (!ok[c]) continue;
                if (best < 0 || bt[c] < bestt * (1.0 - cand[c].mar)) {
                    best = c; bestt = bt[c];
                } else if (bt[c] < bestt) bestt = bt[c];
            }
            if (l6_verbose())
                for (int c = 0; c < NCAND; ++c)
                    fprintf(stderr, "L6_pfa race: %-20s %s %10.4f us/vol%s\n",
                            cand[c].nm, ok[c] ? "ok " : "BAD",
                            ok[c] ? bt[c] / (double)(reps * nb) * 1e6 : 0.0,
                            c == best ? "   <-- chosen" : "");
        } else if (forced < 0) {
            for (int c = 0; c < NCAND; ++c) if (ok[c]) { best = c; break; }
        }
        if (forced >= 0) best = forced;

#ifdef L6_FORCE_VARIANT
        best = L6_FORCE_VARIANT;         /* development override only */
        if (best >= 0 && best < NCAND && !ok[best]) best = -1;
#endif
        if (best >= 0 && best < NCAND) {
            p->run    = cand[best].k;
            p->fence  = cand[best].fence;
            p->chosen = cand[best].nm;
        }

        /* ---- boundary probes: the x->yz joint measurement (see header).
         * Runs under the same placed-scratch address conditions as the race.
         * The probes are built on k_fu, so guard on THAT entry having passed
         * the gate (its slot moved in the r10 reorder -- look it up). ---- */
        double bp[4] = { 0.0, 0.0, 0.0, 0.0 };
        int okfu = 0;
        for (int c = 0; c < NCAND; ++c)
            if (cand[c].k == k_fu) { okfu = ok[c]; break; }
        if (a && b && okfu)
            l6_boundary_probe(p->t1, p->t2, a, b, bp);

        /* ---- clock probes, back to back in this one process.  Sparse-first
         * with scalar-spin gaps so a heavier licence cannot leak backward
         * into a lighter probe. ---- */
        double clkS256 = 0.0, clkD256 = 0.0, clkS512 = 0.0, kclk = 0.0;
        {
            double r[5];
            l6_scalar_spin(8.0);
            for (int i = 0; i < 5; ++i) r[i] = l6_clk_s256(4000000L);
            clkS256 = l6_med(r, 5);
            for (int i = 0; i < 5; ++i) r[i] = l6_clk_d256(2500000L);
            clkD256 = l6_med(r, 5);
#ifdef __AVX512F__
            l6_scalar_spin(4.0);
            for (int i = 0; i < 5; ++i) r[i] = l6_clk_s512(4000000L);
            clkS512 = l6_med(r, 5);
#endif
            if (a && b && best >= 0) {
                l6_scalar_spin(4.0);
                kclk = l6_kclk(p->run, p->fence, p->t1, p->t2, a, b);
            }
        }
        /* licence tail (L6_unrolled r6's lesson): hand the driver a core in
         * the CHOSEN kernel's own steady state, never a probe's */
        if (a && b && best >= 0) {
            double t0 = now_s();
            do {
                p->run(p->t1, p->t2, a, b, 1);
                if (p->fence) _mm_sfence();
            } while (now_s() - t0 < 3e-3);
        }
        snprintf(l6_desc, sizeof l6_desc,
                 "Good-Thomas PFA 2x3 per axis, no twiddles, 2 cplx/ymm, "
                 "plan-raced; variant=%s%s clkS256=%.2f clkD256=%.2f "
                 "clkS512=%.2f kclk=%.2fGHz "
                 "bf=%.1f bsp=%.1f bx=%.1f byz=%.1fns",
                 p->chosen, forced >= 0 ? "!" : "",
                 clkS256, clkD256, clkS512, kclk,
                 bp[0], bp[1], bp[2], bp[3]);
        if (l6_verbose())
            fprintf(stderr, "L6_pfa probes: clkS256=%.3f clkD256=%.3f "
                    "clkS512=%.3f kclk=%.3f GHz; boundary bf=%.1f bsp=%.1f "
                    "bx=%.1f byz=%.1f ns (bf-bsp=%.1f joint, bf-bx-byz=%.1f)\n",
                    clkS256, clkD256, clkS512, kclk,
                    bp[0], bp[1], bp[2], bp[3],
                    bp[0] - bp[1], bp[0] - bp[2] - bp[3]);
        }
#ifdef _OPENMP
        else {
        /* ================= MULTICORE arm (round mt_r1) =================
         * 2D tournament over (kernel row x thread column) on the race arena.
         * Round mt_r2: cells are timed through the SPIN POOL (l6_race_run),
         * i.e. the exact dispatch a pool-shipped execute uses -- mt_r1 raced
         * through the OMP fork/join and its 13-18 us node fork tax mis-picked
         * B=4096 in one of three processes (+52.9% spread).  Column pick is
         * smallest-T-first with a 2% takeover margin (a wide, noisier column
         * must prove itself); rows keep phase-1 margins. */
        static const int Tset[] = { 1, 2, 4, 8, 16, 24, 32 };
        enum { NTS = (int)(sizeof(Tset) / sizeof(Tset[0])) };
        int Tuse[NTS], nTu = 0;
        for (int i = 0; i < NTS; ++i)
            if (Tset[i] <= tmax && (long)Tset[i] <= nb) Tuse[nTu++] = Tset[i];

        double bt[NTS][NMT];
        for (int ci = 0; ci < NTS; ++ci)
            for (int c = 0; c < NMT; ++c) bt[ci][c] = 1e30;
        int ck = -1, cT = 1;
        double t1c = 0.0, tbc = 0.0;

        /* Build the spin pool BEFORE the tournament (round mt_r2) so every
         * T>1 cell is timed through the dispatch the scored run uses; see
         * l6_race_run for the mt_r1 mis-pick this fixes.  Idle workers spin
         * during T<32 cells exactly as they do in a scored pool run. */
        l6_pool *mtpool = NULL;
        if (a && b && nTu > 0 && tmax > 1)
            mtpool = l6_pool_create(p, tmax, omp_sets);
        int rdisp_pool = (mtpool != NULL);

        if (a && b && nTu > 0) {
            /* settle spin: ~100 ms of dense ymm FMA so calibration and round
             * 0 are not ranked on a ramping clock (phase-1 discipline) */
            double ts0 = now_s();
            while (now_s() - ts0 < 0.1) (void)l6_clk_d256(400000L);

            int cal = 0;
            while (cal < NMT - 1 && !okm[cal]) ++cal;
            long reps[NTS];
            for (int ci = 0; ci < nTu; ++ci) {
                reps[ci] = 1;
                if (!okm[cal]) continue;
                for (;;) {           /* per-column reps: slices ~1.2 ms */
                    double tc = now_s();
                    for (long r = 0; r < reps[ci]; ++r)
                        l6_race_run(p, mtpool, mtc[cal].k, mtc[cal].fence,
                                    Tuse[ci], a, b, nb);
                    double e = now_s() - tc;
                    if (e > 1.2e-3 || reps[ci] > (1L << 20)) break;
                    reps[ci] *= 2;
                }
            }
            /* round-robin over ALL cells so drift hits every cell equally;
             * per-cell minimum; ~0.3 ms untimed dwell before each slice so a
             * cell is measured in its own steady licence/cache state */
            for (int round = 0; round < 7; ++round)
                for (int ci = 0; ci < nTu; ++ci)
                    for (int c = 0; c < NMT; ++c) {
                        if (!okm[c]) continue;
                        double td = now_s();
                        do
                            l6_race_run(p, mtpool, mtc[c].k, mtc[c].fence,
                                        Tuse[ci], a, b, nb);
                        while (now_s() - td < 3e-4);
                        double tt = now_s();
                        for (long r = 0; r < reps[ci]; ++r)
                            l6_race_run(p, mtpool, mtc[c].k, mtc[c].fence,
                                        Tuse[ci], a, b, nb);
                        double e = (now_s() - tt) / (double)reps[ci];
                        if (e < bt[ci][c]) bt[ci][c] = e;
                    }
            /* per-column winner: safest-first with per-row margins and the
             * running true minimum (no sub-margin drift, phase-1 rule) */
            int colk[NTS];
            for (int ci = 0; ci < nTu; ++ci) {
                int kk = -1; double ref = 1e30;
                for (int c = 0; c < NMT; ++c) {
                    if (!okm[c]) continue;
                    if (kk < 0 || bt[ci][c] < ref * (1.0 - mtc[c].mar)) {
                        kk = c; ref = bt[ci][c];
                    } else if (bt[ci][c] < ref) ref = bt[ci][c];
                }
                colk[ci] = kk;
            }
            int bci = -1; double cref = 1e30;
            for (int ci = 0; ci < nTu; ++ci) {
                if (colk[ci] < 0) continue;
                double tc = bt[ci][colk[ci]];
                if (bci < 0 || tc < cref * 0.98) { bci = ci; cref = tc; }
                else if (tc < cref) cref = tc;
            }
            if (bci >= 0) {
                ck = colk[bci]; cT = Tuse[bci];
                tbc = bt[bci][colk[bci]] / (double)nb * 1e6;
            }
            if (colk[0] >= 0) t1c = bt[0][colk[0]] / (double)nb * 1e6;
            if (l6_verbose()) {
                fprintf(stderr, "L6_pfa mt race (nb=%ld, us/vol):\n%22s",
                        nb, "");
                for (int ci = 0; ci < nTu; ++ci)
                    fprintf(stderr, "     T=%-2d", Tuse[ci]);
                for (int c = 0; c < NMT; ++c) {
                    fprintf(stderr, "\n  %-20s", mtc[c].nm);
                    for (int ci = 0; ci < nTu; ++ci)
                        fprintf(stderr, " %8.4f",
                                okm[c] ? bt[ci][c] / (double)nb * 1e6 : 0.0);
                }
                fprintf(stderr, "\n");
                if (ck >= 0)
                    fprintf(stderr, "  -> chosen %s T=%d (T=1 col best "
                            "%.4f us/vol)\n", mtc[ck].nm, cT, t1c);
            }
        }

        /* forced overrides for the monitor's A/Bs: L6_FORCE=<row name> and/or
         * L6_FORCE_T=<threads>; the gate still applies, '!' marks the pick */
        int forcedmt = 0;
        {
            const char *fn = getenv("L6_FORCE");
            if (fn)
                for (int c = 0; c < NMT; ++c)
                    if (okm[c] && strcmp(fn, mtc[c].nm) == 0) {
                        ck = c; forcedmt = 1; break;
                    }
            const char *ft = getenv("L6_FORCE_T");
            if (ft) {
                int T = atoi(ft);
                if (T >= 1 && T <= tmax && (long)T <= batch) {
                    cT = T; forcedmt = 1;
                }
            }
        }

        if (ck >= 0) {
            p->run = mtc[ck].k; p->fence = mtc[ck].fence;
            p->chosen = mtc[ck].nm; p->nthreads = cT;
        } else {
            for (int c = 0; c < NCAND; ++c)
                if (ok[c]) { p->run = cand[c].k; p->fence = cand[c].fence;
                             p->chosen = cand[c].nm; break; }
            p->nthreads = 1;
        }

        /* ---- dispatch race: spin pool vs OMP fork/join, on the CHOSEN
         * (kernel, T).  Sequenced, not interleaved (mt_r1 rule), now in the
         * OTHER order because the pool already exists from the tournament:
         * the pool is timed first (GOMP's workers have long futex-slept
         * during the multi-second tournament), then DESTROYED -- workers
         * joined, cores idle -- before OMP is timed, and only recreated if
         * it won by >2%.  Neither dispatch is ever timed with the other's
         * threads spinning. ---- */
        double t_omp = 0.0, t_pool = 0.0;
        if (p->nthreads > 1 && a && b) {
            long reps2 = 1;
            for (;;) {
                double tc = now_s();
                for (long r = 0; r < reps2; ++r)
                    l6_race_run(p, mtpool, p->run, p->fence, p->nthreads,
                                a, b, nb);
                if (now_s() - tc > 1.5e-3 || reps2 > (1L << 20)) break;
                reps2 *= 2;
            }
            if (mtpool) {
                t_pool = 1e30;
                for (int round = 0; round < 5; ++round) {
                    double td = now_s();
                    do
                        l6_run_pool(mtpool, p->run, p->fence, p->nthreads,
                                    a, b, nb);
                    while (now_s() - td < 3e-4);
                    double tt = now_s();
                    for (long r = 0; r < reps2; ++r)
                        l6_run_pool(mtpool, p->run, p->fence, p->nthreads,
                                    a, b, nb);
                    double e = (now_s() - tt) / (double)reps2;
                    if (e < t_pool) t_pool = e;
                }
                l6_pool_destroy(mtpool);   /* joined: OMP timed on idle cores */
                mtpool = NULL;
            }
            t_omp = 1e30;
            for (int round = 0; round < 5; ++round) {
                double td = now_s();
                do
                    l6_run_cfg(p, p->run, p->fence, p->nthreads, a, b, nb);
                while (now_s() - td < 3e-4);
                double tt = now_s();
                for (long r = 0; r < reps2; ++r)
                    l6_run_cfg(p, p->run, p->fence, p->nthreads, a, b, nb);
                double e = (now_s() - tt) / (double)reps2;
                if (e < t_omp) t_omp = e;
            }
            if (t_pool > 0.0 && t_pool < 1e29 && t_pool < t_omp * 0.98) {
                l6_pool *pl = l6_pool_create(p, tmax, omp_sets);
                if (pl) { p->pool = pl; p->use_pool = 1; }
            }
            if (l6_verbose())
                fprintf(stderr, "L6_pfa dispatch: omp=%.4f pool=%.4f us/vol"
                        " -> %s\n", t_omp / (double)nb * 1e6,
                        t_pool > 0 ? t_pool / (double)nb * 1e6 : 0.0,
                        p->use_pool ? "pool" : "omp");
        }
        /* a T=1 pick (or an arena-less plan) still owns a live pool here */
        if (mtpool) { l6_pool_destroy(mtpool); mtpool = NULL; }

        /* threaded-split correctness check of the FINAL configuration on an
         * ODD volume count (uneven chunks), against the scalar reference: a
         * split bug can never ship.  Falls back to T=1 (gate-proven). */
        if (p->nthreads > 1) {
            long nchk = batch < 61 ? batch : 61;
            double *cin = (double *)l6_alloc((size_t)nchk * VOLD * sizeof(double));
            double *crf = (double *)l6_alloc((size_t)nchk * VOLD * sizeof(double));
            double *cgt = (double *)l6_alloc((size_t)nchk * VOLD * sizeof(double));
            int good = 0;
            if (cin && crf && cgt) {
                unsigned s2 = 987654321u;
                for (long i = 0; i < nchk * VOLD; ++i) {
                    s2 = s2 * 1664525u + 1013904223u;
                    cin[i] = (double)(int)(s2 >> 8) * 1e-9;
                }
                kern_scalar(p->t1, p->t2, cin, crf, nchk);
                memset(cgt, 0, (size_t)nchk * VOLD * sizeof(double));
                if (p->use_pool)
                    l6_run_pool((l6_pool *)p->pool, p->run, p->fence,
                                p->nthreads, cin, cgt, nchk);
                else
                    l6_run_cfg(p, p->run, p->fence, p->nthreads,
                               cin, cgt, nchk);
                double nrm2 = 0.0, e2 = 0.0;
                for (long i = 0; i < nchk * VOLD; ++i) {
                    nrm2 += crf[i] * crf[i];
                    double d = cgt[i] - crf[i];
                    e2 += d * d;
                }
                good = nrm2 > 0.0 && e2 <= 1e-26 * nrm2;
            }
            free(cin); free(crf); free(cgt);
            if (!good) {                 /* the kernel itself is gate-proven */
                p->nthreads = 1;
                if (p->use_pool) {
                    l6_pool_destroy((l6_pool *)p->pool);
                    p->pool = NULL; p->use_pool = 0;
                }
            }
        }

        /* steady-state tail: hand the driver the pool, licence and caches in
         * the CHOSEN configuration's own state (phase-1 lesson, threaded) */
        if (a && b) {
            double tt0 = now_s();
            do {
                if (p->use_pool)
                    l6_run_pool((l6_pool *)p->pool, p->run, p->fence,
                                p->nthreads, a, b, nb);
                else
                    l6_run_cfg(p, p->run, p->fence, p->nthreads, a, b, nb);
            } while (now_s() - tt0 < 3e-3);
        }

        snprintf(l6_desc, sizeof l6_desc,
                 "Good-Thomas PFA 2x3 per axis, no twiddles, 2 cplx/ymm; mt "
                 "batch-split, per-thread NUMA scratch, 2D-raced@%s (%d "
                 "kernels x T<=%d); variant=%s%s T=%d disp=%s fork=%.2fus "
                 "raceT1=%.4f raceBest=%.4f omp=%.4f pool=%.4fus/vol",
                 rdisp_pool ? "pool" : "omp",
                 (int)NMT, tmax, p->chosen, forcedmt ? "!" : "", p->nthreads,
                 p->use_pool ? "pool" : "omp", fork_us, t1c, tbc,
                 t_omp > 0.0 ? t_omp / (double)nb * 1e6 : 0.0,
                 t_pool > 0.0 ? t_pool / (double)nb * 1e6 : 0.0);
        if (l6_verbose())
            fprintf(stderr, "L6_pfa mt: chosen %s T=%d disp=%s fork=%.2fus "
                    "raceT1=%.4f raceBest=%.4f us/vol\n",
                    p->chosen, p->nthreads, p->use_pool ? "pool" : "omp",
                    fork_us, t1c, tbc);
        }
#endif /* _OPENMP */

        free(a); free(b);
        p->t1 = p->arena; p->t2 = p->arena + VOLD; p->placed = 0;
    }
#else
    {
        char *v = strstr(l6_desc, "variant=");
        if (v) snprintf(v, (size_t)(l6_desc + sizeof(l6_desc) - v),
                        "variant=%s", p->chosen);
    }
#endif
    if (l6_verbose())
        fprintf(stderr, "L6_pfa: chosen %s for batch=%ld\n", p->chosen, p->batch);
    return p;
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
#if defined(L6_HAVE_AVX2) && defined(_OPENMP)
    if (plan->nthreads > 1) {
        /* no thread is created here: either one parallel region re-entering
         * the pool warmed in create(), or the plan's own spin pool */
        if (plan->use_pool)
            l6_run_pool((l6_pool *)plan->pool, plan->run, plan->fence,
                        plan->nthreads, (const double *)in, (double *)out,
                        plan->batch);
        else
            l6_run_cfg(plan, plan->run, plan->fence, plan->nthreads,
                       (const double *)in, (double *)out, plan->batch);
        return;
    }
#endif
    if (!plan->placed) place_scratch(plan, in, out);
    plan->run(plan->t1, plan->t2, (const double *)in, (double *)out, plan->batch);
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
