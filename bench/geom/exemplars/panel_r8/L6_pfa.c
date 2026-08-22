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
 * VECTORISATION -- interleaved complex, 2 complex per ymm, no deinterleave
 * ------------------------------------------------------------------------
 * One ymm holds 2 complex = 2 independent lines of the current axis.  The x
 * and y passes are transpose-free (lanes = 2 adjacent z); only the z pass
 * changes lane axis, with 6 vperm2f128 in + 6 out per pencil pair, provably
 * minimal (see strategies/L6_pfa.md round 1).  Every access is 32B aligned.
 * The bulk of the file is 256-bit; from AVX-512 it takes the 32 evex ymm
 * registers (the fused variant needs ~26 live vectors).
 *
 * 512-BIT: CLOSED in panel_r7.  Three zmm families across the two L=6 entries
 * (my fused_zx at -16% uops, L6_unrolled's zxf/zff at -17%/-25%), raced with
 * the licence bias removed (per-candidate dwell) at a MEASURED equal licence
 * clock (kclk = 2.89 GHz on the node): zero picks in eight cells, all three
 * processes.  Width buys nothing where the kernel is not front-end-bound,
 * and at L=6 it is not.  The zmm kernels are deleted this round; the r7
 * record and VERDICT §5 carry the falsification.
 *
 * CLOCK: settled panel_r7.  kclk = 2.89 GHz (the ymm kernels hold the AVX2-
 * heavy licence).  B=1 at 0.219 us = 632 cycles = 1.30x the 486-cycle
 * FP-port floor: ~147 cycles of overhead that three node-falsified theories
 * (uop count, OoO window, port 5) do not explain.  Per the r7 VERDICT, B=1
 * mechanism work is stopped pending the monitor's perf-stat A/B
 * (L6_FORCE=fused, the switch is shipped); the standing suspect is the t1
 * store->load joint at the pass boundary.
 *
 * KERNEL SHAPES, raced at plan time (setup is not scored)
 * -------------------------------------------------------
 * Grid pruned to 11 in panel_r8 (from 20): every kernel the node has picked
 * in r4-r7 survives, everything 0-for-N dies.  Deleted: the zmm family
 * (falsified, above), all NT kernels (0-for-6 rounds on the node), 3pass_ip,
 * 3pass_pf/pfw, fused_sp2 plain/pfw, fused_pft1w (never picked).
 *   3pass    x: in->t1, y: t1->t2, z: t2->out.  Safety representative.
 *   fused    x: in->t1, then y and z fused over one 6x6 (y,z) plane held in
 *            18 ymm (+ temps ~26 live): needs the target's 32 evex ymm.
 *            The node's B=1 pick six rounds running (zp-outer x order).
 *   *_xa     x-pass in strictly ascending address order (ADOPTED FROM
 *            L6_unrolled, r6): node-picked at B=64/4096/32768 in r7 -- took
 *            B=4096 outright (0.381, -2.6%).  zp-outer keeps B=1.
 *   fused_sp2 software-pipelined fused stage, plane registers double-
 *            buffered, plane-pair loop kept rolled (DSB-resident).
 *            Node-picked at B=64 in r5 (pf) and r7 (pf_xa, 2 of 3 runs).
 *   *_rot    NEW panel_r8: per-volume scratch rotation, see below.
 * prefetch hooks, one per x-pass group (3 cache lines each, 54 lines = the
 *            whole next volume, ~1 volume of lead):
 *   pf   = prefetcht0 of the next volume's INPUT
 *   w    = prefetchw of the next volume's OUTPUT -- write-intent prefetch,
 *          ADOPTED FROM L6_unrolled (panel_r3): with NT rejected on the node,
 *          every output line pays a write-allocate RFO; prefetchw issues that
 *          RFO one volume early, off the critical path.
 *
 * PER-VOLUME SCRATCH ROTATION (NEW panel_r8)
 * -------------------------------------------
 * The 4K-alias placement below is only guaranteed for volume 0: the volume
 * stride 3456 = -640 mod 4096 walks all 32 residues of (t1 - in_b) mod 4096
 * as b advances, including the replay-prone ones near 0 (my r2 record noted
 * exactly this as the reason placement is "B=1 only in effect").  The _rot
 * kernels advance t1 by the volume stride mod 4096 each volume,
 *      t1_b = t1 + ((432*b) mod 512) doubles,
 * so BOTH cross-buffer deltas (t1-in_b and out_b-t1) are constant in b and
 * equal to the optimum place_scratch found for volume 0.  Cost: one AND+ADD
 * per volume, and the scratch window grows 3.4 KB -> 7.5 KB (still well
 * inside L1).  At B=1 (b=0) a _rot kernel is bit- and address-identical to
 * its twin, so only the batched cells race them.  Mechanism inspired by
 * L23_rader's panel_r7 result (owning the offending stride mod 4096 was
 * worth 25-30% there); the per-volume phase-lock is mine.
 *
 * 64-BYTE KERNEL PINNING (NEW panel_r8, ADOPTED FROM L6_unrolled r6): every
 * raced kernel entry is __attribute__((aligned(64))).  Their B=1 regressed
 * 0.219 -> 0.227 typical with an IDENTICAL pick string when unrelated code
 * was added to the file (code layout, the disease the r5 VERDICT names at
 * L36_mixedradix); the kernels are reached through a function pointer, so
 * their placement is otherwise at the linker's mercy.  My same-shape B>=64
 * deficit to their file (0.226 vs 0.214-0.222 at B=64, three rounds) has
 * exactly this fingerprint, and this plus the grid shrink (20 -> 11 kernels,
 * less .text between the hot entries) is the round's bet on closing it.
 *
 * CLOCK PROBES (NEW panel_r6 -- the r5 VERDICT's #1 ask for L=6)
 * ---------------------------------------------------------------
 * Four sustained-clock measurements run back to back IN ONE PROCESS at the
 * end of fft3d_create() (unscored) and are reported in fft3d_description():
 *   clkS256 = serially dependent ymm FMA chain, 1 FMA in flight (latency 4
 *             on CLX and SPR): the "sparse" design L6_unrolled shipped in
 *             r4/r5, which read 3.89 GHz on the node.
 *   clkD256 = 12 independent ymm FMA chains, 2/cycle throughput-bound: the
 *             saturating design L17_winograd shipped, which read 2.89.
 *   clkS512 = serially dependent zmm chain (settled: 2.89 on the node).
 *   kclk    = MY ADDITION, the number the panel actually needs: dwell ~2 ms
 *             in the CHOSEN ymm kernel, then immediately time a short
 *             (~140 us) sparse ymm chain, alternated 9x, median.  Licence
 *             transitions persist >600 us on CLX, so the short chain reads
 *             the licence the real kernel established: this is the clock the
 *             scored kernels run at, measured directly.  If kclk = 2.89,
 *             B=1's 0.219 us is 633 cycles = 1.30x the 486-cycle FP floor
 *             (23% unexplained); if 3.89, it is 852 = 1.75x (43%).
 * Probe order is sparse-first with scalar-spin gaps so a heavier licence
 * cannot leak backward into a lighter probe (L17_winograd's r5 rationale).
 *
 * PLAN-TIME TOURNAMENT
 * --------------------
 * Every candidate is validated against the scalar reference (rel L2 <= 1e-13
 * on a small batch; a miscompiled variant can never be selected), then raced
 * round-robin (drift-immune, per-candidate minimum) on a dummy buffer of the
 * real batch size truncated at 16384 volumes = 113 MiB, so a DRAM-bound real
 * batch is raced in a DRAM-bound arena on both wallaby (60 MiB L3) and the
 * node (22 MiB).  Safest-first ordering with a 1.5% takeover margin (raised
 * from 1% this round; the panel_r3 VERDICT measured unstable tuners costing
 * 3.9-6.7% elsewhere) and the reference minimum is tracked even when the
 * incumbent survives, so a chain of sub-margin steps cannot drift the pick.
 * The chosen kernel's name is spliced into fft3d_description() so the
 * leaderboard records which kernel actually ran on the node.
 *
 * LICENCE HYGIENE (NEW panel_r7; lessons from L6_unrolled r6 + L17_rader r5):
 * (1) ~100 ms of dense ymm FMA before the race so round 0 is not ranked on a
 * ramping clock; (2) each candidate runs UNTIMED for ~0.5 ms immediately
 * before each timed slice, so a candidate is always measured in its own
 * steady licence/clock state and a zmm candidate cannot poison the next ymm
 * slice (CLX licence transitions persist >600 us); (3) create() ends by
 * dwelling ~3 ms in the CHOSEN kernel, so the driver is handed a core in the
 * scored kernel's own steady state, never a probe's.
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

#include "fft3d_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VOLC 216          /* complex per volume */
#define VOLD 432          /* doubles  per volume */
#define L6_MARGIN 0.015   /* a later race candidate must win by >1.5% */

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

/* 6-point PFA on two interleaved complex lanes per register.
 * 18 FP instructions (8 of them fma/fnmadd) + 2 vpermilpd.  In-place safe. */
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

/* x pass: axis stride 72 doubles, lanes = 2 adjacent z.  zp-outer/y-inner
 * order (kept from round 1, measured +0.6%: pass 2's first groups need all
 * six y of a z-pair, which this order finishes 10 groups sooner than the
 * flat ascending order).  The prefetch index 6*zp+y still walks the next
 * volume's 54 lines in strictly ascending address order. */
#define PASS_X(SRC, DST, OUT, PF)                                        \
    do {                                                                 \
        for (int zp = 0; zp < 3; ++zp)                                   \
            for (int y = 0; y < 6; ++y) {                                \
                const double *p = (SRC) + 12 * y + 4 * zp;               \
                double *q = (DST) + 12 * y + 4 * zp;                     \
                PF(SRC, OUT, 6 * zp + y);                                \
                __m256d o0, o1, o2, o3, o4, o5;                          \
                DFT6V(LD(p), LD(p + 72), LD(p + 144), LD(p + 216),       \
                      LD(p + 288), LD(p + 360), o0, o1, o2, o3, o4, o5); \
                STN(q, o0); STN(q + 72, o1); STN(q + 144, o2);           \
                STN(q + 216, o3); STN(q + 288, o4); STN(q + 360, o5);    \
            }                                                            \
    } while (0)

/* x pass, ascending-address twin (adopted from L6_unrolled, panel_r6): the
 * 18 groups walk offsets 4g, g = 0..17, i.e. loads and stores in strictly
 * ascending 32B steps -- friendlier to the node's L2 streamer at streaming
 * batch sizes, where their same-shape kernels beat the zp-outer order by
 * 1.3-7%.  Identical arithmetic, identical prefetch index walk. */
#define PASS_X_A(SRC, DST, OUT, PF)                                      \
    do {                                                                 \
        for (int g = 0; g < 18; ++g) {                                   \
            const double *p = (SRC) + 4 * g;                             \
            double *q = (DST) + 4 * g;                                   \
            PF(SRC, OUT, g);                                             \
            __m256d o0, o1, o2, o3, o4, o5;                              \
            DFT6V(LD(p), LD(p + 72), LD(p + 144), LD(p + 216),           \
                  LD(p + 288), LD(p + 360), o0, o1, o2, o3, o4, o5);     \
            STN(q, o0); STN(q + 72, o1); STN(q + 144, o2);               \
            STN(q + 216, o3); STN(q + 288, o4); STN(q + 360, o5);        \
        }                                                                \
    } while (0)

/* y pass: axis stride 12 doubles, lanes = 2 adjacent z.  SRC == DST is the
 * in-place shape (all 6 loads of a group precede its stores). */
#define PASS_Y(SRC, DST)                                                 \
    do {                                                                 \
        for (int x = 0; x < 6; ++x)                                      \
            for (int zp = 0; zp < 3; ++zp) {                             \
                const double *p = (SRC) + 72 * x + 4 * zp;               \
                double *q = (DST) + 72 * x + 4 * zp;                     \
                __m256d o0, o1, o2, o3, o4, o5;                          \
                DFT6V(LD(p), LD(p + 12), LD(p + 24), LD(p + 36),         \
                      LD(p + 48), LD(p + 60), o0, o1, o2, o3, o4, o5);   \
                STN(q, o0); STN(q + 12, o1); STN(q + 24, o2);            \
                STN(q + 36, o3); STN(q + 48, o4); STN(q + 60, o5);       \
            }                                                            \
    } while (0)

/* z pass: axis contiguous; lanes = 2 adjacent y via in-register transpose.
 * The six 32B stores per pair cover exactly three whole 64B lines in
 * ascending order (clean write-combining when ST is the NT store). */
#define PASS_Z(SRC, DST, ST)                                             \
    do {                                                                 \
        for (int x = 0; x < 6; ++x)                                      \
            for (int yp = 0; yp < 3; ++yp) {                             \
                const double *p = (SRC) + 72 * x + 24 * yp;              \
                double *q = (DST) + 72 * x + 24 * yp;                    \
                __m256d r0 = LD(p),      r1 = LD(p + 4),  r2 = LD(p + 8);\
                __m256d r3 = LD(p + 12), r4 = LD(p + 16), r5 = LD(p + 20);\
                __m256d w0, w1, w2, w3, w4, w5;                          \
                DFT6V(LO2(r0, r3), HI2(r0, r3), LO2(r1, r4), HI2(r1, r4),\
                      LO2(r2, r5), HI2(r2, r5), w0, w1, w2, w3, w4, w5); \
                ST(q,      LO2(w0, w1)); ST(q + 4,  LO2(w2, w3));        \
                ST(q + 8,  LO2(w4, w5)); ST(q + 12, HI2(w0, w1));        \
                ST(q + 16, HI2(w2, w3)); ST(q + 20, HI2(w4, w5));        \
            }                                                            \
    } while (0)

/* y and z fused over one 6x6 (y,z) plane held in 18 ymm (+~8 temps). */
#define FUSED_YZ(SRC, DST, ST)                                           \
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
                DFT6V(P[zp], P[3 + zp], P[6 + zp], P[9 + zp],            \
                      P[12 + zp], P[15 + zp], o0, o1, o2, o3, o4, o5);   \
                P[zp] = o0; P[3 + zp] = o1; P[6 + zp] = o2;              \
                P[9 + zp] = o3; P[12 + zp] = o4; P[15 + zp] = o5;        \
            }                                                            \
            for (int yp = 0; yp < 3; ++yp) {                             \
                __m256d r0 = P[6 * yp],     r1 = P[6 * yp + 1];          \
                __m256d r2 = P[6 * yp + 2], r3 = P[6 * yp + 3];          \
                __m256d r4 = P[6 * yp + 4], r5 = P[6 * yp + 5];          \
                __m256d w0, w1, w2, w3, w4, w5;                          \
                DFT6V(LO2(r0, r3), HI2(r0, r3), LO2(r1, r4), HI2(r1, r4),\
                      LO2(r2, r5), HI2(r2, r5), w0, w1, w2, w3, w4, w5); \
                double *o = q + 24 * yp;                                 \
                ST(o,      LO2(w0, w1)); ST(o + 4,  LO2(w2, w3));        \
                ST(o + 8,  LO2(w4, w5)); ST(o + 12, HI2(w0, w1));        \
                ST(o + 16, HI2(w2, w3)); ST(o + 20, HI2(w4, w5));        \
            }                                                            \
        }                                                                \
    } while (0)

/* ---- software-pipelined fused stage (NEW in panel_r5) ----
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

#define FYZ_YDFT(P)                                                      \
    do {                                                                 \
        for (int _zp = 0; _zp < 3; ++_zp) {                              \
            __m256d o0, o1, o2, o3, o4, o5;                              \
            DFT6V((P)[_zp], (P)[3+_zp], (P)[6+_zp], (P)[9+_zp],          \
                  (P)[12+_zp], (P)[15+_zp], o0, o1, o2, o3, o4, o5);     \
            (P)[_zp] = o0; (P)[3+_zp] = o1; (P)[6+_zp] = o2;             \
            (P)[9+_zp] = o3; (P)[12+_zp] = o4; (P)[15+_zp] = o5;         \
        }                                                                \
    } while (0)

#define FYZ_ZCHUNK(P, q, yp, ST)                                         \
    do {                                                                 \
        __m256d r0 = (P)[6*(yp)],   r1 = (P)[6*(yp)+1], r2 = (P)[6*(yp)+2];\
        __m256d r3 = (P)[6*(yp)+3], r4 = (P)[6*(yp)+4], r5 = (P)[6*(yp)+5];\
        __m256d w0, w1, w2, w3, w4, w5;                                  \
        DFT6V(LO2(r0, r3), HI2(r0, r3), LO2(r1, r4), HI2(r1, r4),        \
              LO2(r2, r5), HI2(r2, r5), w0, w1, w2, w3, w4, w5);         \
        double *_o = (q) + 24 * (yp);                                    \
        ST(_o,      LO2(w0, w1)); ST(_o + 4,  LO2(w2, w3));              \
        ST(_o + 8,  LO2(w4, w5)); ST(_o + 12, HI2(w0, w1));              \
        ST(_o + 16, HI2(w2, w3)); ST(_o + 20, HI2(w4, w5));              \
    } while (0)

/* consume plane xx from CUR while loading + y-transforming plane xx+1 in NXT */
#define FYZ_PLANE(CUR, NXT, xx, SRC, DST, ST)                            \
    do {                                                                 \
        const double *_pn = (SRC) + 72 * ((xx) + 1);                     \
        double *_qc = (DST) + 72 * (xx);                                 \
        FYZ_ZCHUNK(CUR, _qc, 0, ST); FYZ_LD6(NXT, _pn, 0);               \
        FYZ_ZCHUNK(CUR, _qc, 1, ST); FYZ_LD6(NXT, _pn, 1);               \
        FYZ_ZCHUNK(CUR, _qc, 2, ST); FYZ_LD6(NXT, _pn, 2);               \
        FYZ_YDFT(NXT);                                                   \
    } while (0)

/* Rolled software-pipelined fused stage: planes 0-3 via a 2-iteration loop
 * over P/Q pairs (body ~450 insns, above gcc's complete-peel limit, so it
 * stays a loop), then plane 4 + the tail of plane 5.  ~5.5-6.4 KB total,
 * DSB-resident on the node.  The node picked this over the fully-unrolled
 * FUSED_YZ_SP twin at B=64 in panel_r5 (>=1.5% margin, 3/3 processes), so
 * the DSB-footprint experiment resolved FOR the rolled form and the
 * unrolled twin is deleted this round. */
#define FUSED_YZ_SP2(SRC, DST, ST)                                       \
    do {                                                                 \
        __m256d P[18], Q[18];                                            \
        FYZ_LD6(P, (SRC), 0); FYZ_LD6(P, (SRC), 1); FYZ_LD6(P, (SRC), 2);\
        FYZ_YDFT(P);                                                     \
        for (int _x2 = 0; _x2 < 4; _x2 += 2) {                           \
            FYZ_PLANE(P, Q, _x2,     SRC, DST, ST);                      \
            FYZ_PLANE(Q, P, _x2 + 1, SRC, DST, ST);                      \
        }                                                                \
        FYZ_PLANE(P, Q, 4, SRC, DST, ST);                                \
        FYZ_ZCHUNK(Q, (DST) + 360, 0, ST);                               \
        FYZ_ZCHUNK(Q, (DST) + 360, 1, ST);                               \
        FYZ_ZCHUNK(Q, (DST) + 360, 2, ST);                               \
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
        PASS_X(ip, t1, op, PF);                                          \
        PASS_Y(t1, MID);                                                 \
        PASS_Z(MID, op, STZ);                                            \
    }                                                                    \
}

/* PX = PASS_X (zp-outer) or PASS_X_A (ascending) */
#define GEN_FU(NAME, PX, STZ, PF)                                        \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,     \
                 const double *restrict in, double *restrict out, long nb)\
{                                                                        \
    VSET;                                                                \
    (void)t2;                                                            \
    for (long b = 0; b < nb; ++b) {                                      \
        const double *ip = in + b * VOLD;                                \
        double *op = out + b * VOLD;                                     \
        PX(ip, t1, op, PF);                                              \
        FUSED_YZ(t1, op, STZ);                                           \
    }                                                                    \
}

/* x pass + rolled software-pipelined (double-buffered plane) fused y/z */
#define GEN_FU_SP2(NAME, PX, PF)                                         \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,     \
                 const double *restrict in, double *restrict out, long nb)\
{                                                                        \
    VSET;                                                                \
    (void)t2;                                                            \
    for (long b = 0; b < nb; ++b) {                                      \
        const double *ip = in + b * VOLD;                                \
        double *op = out + b * VOLD;                                     \
        PX(ip, t1, op, PF);                                              \
        FUSED_YZ_SP2(t1, op, STN);                                       \
    }                                                                    \
}

/* Per-volume scratch rotation (see header): t1 advances by the volume
 * stride mod 4096 B, so the volume-0-optimal alias deltas hold for every
 * volume.  432*b mod 512 doubles is always a multiple of 16 doubles =
 * 128 B, so every access stays 64B-aligned; the window is 944 doubles. */
#define ROTW 512                     /* doubles = 4096 B, the alias period */

#define GEN_FU_ROT(NAME, PX, STZ, PF)                                    \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,     \
                 const double *restrict in, double *restrict out, long nb)\
{                                                                        \
    VSET;                                                                \
    (void)t2;                                                            \
    for (long b = 0; b < nb; ++b) {                                      \
        const double *ip = in + b * VOLD;                                \
        double *op = out + b * VOLD;                                     \
        double *tv = t1 + ((b * VOLD) & (ROTW - 1));                     \
        PX(ip, tv, op, PF);                                              \
        FUSED_YZ(tv, op, STZ);                                           \
    }                                                                    \
}

#define GEN_FU_SP2_ROT(NAME, PX, PF)                                     \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,     \
                 const double *restrict in, double *restrict out, long nb)\
{                                                                        \
    VSET;                                                                \
    (void)t2;                                                            \
    for (long b = 0; b < nb; ++b) {                                      \
        const double *ip = in + b * VOLD;                                \
        double *op = out + b * VOLD;                                     \
        double *tv = t1 + ((b * VOLD) & (ROTW - 1));                     \
        PX(ip, tv, op, PF);                                              \
        FUSED_YZ_SP2(tv, op, STN);                                       \
    }                                                                    \
}

GEN_FU(k_fu,                 PASS_X,    STN, PF_NONE)
GEN_FU(k_fu_xa,              PASS_X_A,  STN, PF_NONE)
GEN_3P(k_3p,                 t2,        STN, PF_NONE)
GEN_FU(k_fu_pf_xa,           PASS_X_A,  STN, PF_T0)
GEN_FU_ROT(k_fu_pf_xa_rot,   PASS_X_A,  STN, PF_T0)
GEN_FU(k_fu_pf,              PASS_X,    STN, PF_T0)
GEN_FU_SP2(k_fu_sp2_pf_xa,   PASS_X_A,       PF_T0)
GEN_FU_SP2_ROT(k_fu_sp2_pf_xa_rot, PASS_X_A, PF_T0)
GEN_FU(k_fu_pfw_xa,          PASS_X_A,  STN, PF_T0W)
GEN_FU_ROT(k_fu_pfw_xa_rot,  PASS_X_A,  STN, PF_T0W)
GEN_FU(k_fu_pfw,             PASS_X,    STN, PF_T0W)

/* The AVX-512 x-pass kernel family (fused_zx*) was DELETED in panel_r8:
 * panel_r7 measured it (and L6_unrolled's zxf/zff) at zero picks in eight
 * cells with the licence bias removed and kclk = 2.89 -- uops are not the
 * limiter at L=6.  See the r7 strategy record for the falsification. */

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

const char *fft3d_name(void) { return "L6_pfa"; }

/* The chosen kernel and the four clock probes are formatted into the
 * description so the leaderboard records which kernel actually ran on the
 * node and what clock it ran at (variant reporting from L6_unrolled; the
 * probe set is this round's experiment -- see the header comment). */
static char l6_desc[256] =
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

/* ---------------------------------------------------- clock probes ---
 * All setup-time only (unscored).  Frequencies from FMA chains of known
 * cycle cost: latency 4 / throughput 2-per-cycle for 256-bit FMA on both
 * Cascade Lake and Sapphire Rapids; zmm FMA latency is also 4 on both.
 * (On a Haswell dev host FMA latency is 5 and the numbers over-read by
 * 25% -- caveat from L6_unrolled's r4 probe, applies to wombat only.) */

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
/* sparse zmm: 1 serially dependent 512-bit FMA in flight = 4 cycles/iter.
 * Even sparse 512-bit engages the AVX-512 licence (L6_unrolled's r5 probe
 * read 2.89 with exactly this design). */
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
    /* 4096 B of slack for the 4K-aliasing placement, then the scratch
     * window: max(two static volumes = 864, rot window 512+432 = 944)
     * doubles.  944 covers both. */
    p->arena = (double *)l6_alloc(4096 + 944 * sizeof(double) + 64);
    if (!p->arena) { free(p); return NULL; }
    memset(p->arena, 0, 4096 + 944 * sizeof(double) + 64);
    p->t1 = p->arena;
    p->t2 = p->arena + VOLD;
    p->run = kern_scalar;
    p->fence = 0;
    p->placed = 0;
    p->chosen = "scalar";

#ifdef L6_HAVE_AVX2
    {
        /* Safest-first order, pruned to the node's demonstrated picks
         * (r4-r7) plus the new _rot twins.  'fused' (zp-outer) leads: it is
         * the node's B=1 pick six rounds running, and a later candidate must
         * beat the running best by >1.5% to take over.  Each _rot twin sits
         * AFTER its parent: rotation is a new, node-unmeasured mechanism, so
         * it must earn the takeover margin rather than inherit the incumbent
         * slot (the xa twins got the forward slot in r6 only because the
         * node itself had already measured ascending winning at B>=64). */
        static const struct { l6_kernel k; int fence; const char *nm; } cand[] = {
            { k_fu,                0, "fused"               },
            { k_fu_xa,             0, "fused_xa"            },
            { k_3p,                0, "3pass"               },
            { k_fu_pf_xa,          0, "fused_pf_xa"         },
            { k_fu_pf_xa_rot,      0, "fused_pf_xa_rot"     },
            { k_fu_pf,             0, "fused_pf"            },
            { k_fu_sp2_pf_xa,      0, "fused_sp2_pf_xa"     },
            { k_fu_sp2_pf_xa_rot,  0, "fused_sp2_pf_xa_rot" },
            { k_fu_pfw_xa,         0, "fused_pfw_xa"        },
            { k_fu_pfw_xa_rot,     0, "fused_pfw_xa_rot"    },
            { k_fu_pfw,            0, "fused_pfw"           },
        };
        enum { NCAND = (int)(sizeof(cand) / sizeof(cand[0])) };
        int ok[NCAND];
        for (int c = 0; c < NCAND; ++c) ok[c] = 0;

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
        }
        free(vin); free(vref); free(vgot);

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

        /* ---- race the survivors at (a truncation of) the real batch size.
         * Cap 16384 volumes = 113 MiB: unambiguously DRAM on wallaby (60 MiB
         * L3) and the node (22 MiB), so a DRAM-bound real batch is raced in a
         * DRAM-bound arena (cap rationale from L6_unrolled round 2). ---- */
        long nb = batch;
        if (nb > 16384) nb = 16384;
        size_t nd = (size_t)nb * VOLD;
        double *a = (double *)l6_alloc((nd + VOLD) * sizeof(double));
        double *b = (double *)l6_alloc(nd * sizeof(double));
        int best = -1;
        if (a && b) {
            unsigned s = 12345u;
            for (size_t i = 0; i < nd + VOLD; ++i) {
                s = s * 1664525u + 1013904223u;
                a[i] = (double)(int)(s >> 8) * 1e-9;
            }
            memset(b, 0, nd * sizeof(double));
            place_scratch(p, a, b);      /* race with realistically-placed scratch */
        }
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
            /* Safest-first with a 1.5% takeover margin; the reference time
             * tracks the true minimum even when the incumbent survives, so a
             * chain of sub-margin steps cannot drift the pick. */
            double bestt = 1e30;
            for (int c = 0; c < NCAND; ++c) {
                if (!ok[c]) continue;
                if (best < 0 || bt[c] < bestt * (1.0 - L6_MARGIN)) {
                    best = c; bestt = bt[c];
                } else if (bt[c] < bestt) bestt = bt[c];
            }
            if (l6_verbose())
                for (int c = 0; c < NCAND; ++c)
                    fprintf(stderr, "L6_pfa race: %-16s %s %10.4f us/vol%s\n",
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

        /* ---- clock probes, back to back in this one process (the r5
         * VERDICT's #1 L=6 ask).  Sparse-first with scalar-spin gaps so a
         * heavier licence cannot leak backward into a lighter probe. ---- */
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
        free(a); free(b);
        p->t1 = p->arena; p->t2 = p->arena + VOLD; p->placed = 0;

        snprintf(l6_desc, sizeof l6_desc,
                 "Good-Thomas PFA 2x3 per axis, no twiddles, 2 cplx/ymm, "
                 "plan-raced; variant=%s%s clkS256=%.2f clkD256=%.2f "
                 "clkS512=%.2f kclk=%.2fGHz",
                 p->chosen, forced >= 0 ? "!" : "",
                 clkS256, clkD256, clkS512, kclk);
        if (l6_verbose())
            fprintf(stderr, "L6_pfa probes: clkS256=%.3f clkD256=%.3f "
                    "clkS512=%.3f kclk=%.3f GHz\n",
                    clkS256, clkD256, clkS512, kclk);
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
    if (!plan->placed) place_scratch(plan, in, out);
    plan->run(plan->t1, plan->t2, (const double *)in, (double *)out, plan->batch);
#ifdef L6_HAVE_AVX2
    if (plan->fence) _mm_sfence();
#endif
}

void fft3d_destroy(fft3d_plan *plan)
{
    if (!plan) return;
    free(plan->arena);
    free(plan);
}
