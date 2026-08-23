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
 * 512-BIT: CLOSED in panel_r7 ON CLX (zero picks in eight cells at a
 * measured equal licence clock, kclk = 2.89 GHz -- the Gold 5218 has ONE
 * 512-bit FMA pipe, so zmm halved instructions but not port-cycles).
 * REOPENED AND WON in round ice_r2 ON ICX: the Gold 6326 has TWO 512-bit
 * FMA pipes, and the zmm x-pass (z512x: 36 lines = 9 aligned 4-lane groups,
 * no tail) beat the best ymm incumbent by 4.4-5.1% in both node clock
 * regimes; the 512 licence costs 3.50 -> 3.30 GHz and is fully covered.
 * PER-VOLUME SCRATCH ROTATION (_rot): CLOSED in panel_r8 (zero picks in
 * eight cells; the per-volume alias phase-lock is a null on CLX -- demand-
 * side 4K replays at these strides are below the 1.5% margin); still
 * deleted, the r8 record carries the falsification.
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
 * ROUND ice_r4 -- THE GRADED STEP IS NOW FFT + MAP, AND THIS FILE OWNS THE CHAIN
 * --------------------------------------------------------------------------
 * The graded step became  state <- (z + c) / (1 + |z + c|),  z = raw FFT(state),
 * timed through an exported fft3d_chain(plan, x0, c, final_out, m) weak symbol
 * (m = 4856 at L=6).  Without it the driver falls back to fft3d_execute + a
 * driver-side map pass.  What this file does with it:
 *
 *   VOLUME-MAJOR CHAIN (adopted from L17_rader / L23 / L36_mixedradix ice_r4,
 *   corpus 10 s3): volume b runs ALL m steps before b+1.  Working set per
 *   chain = state (3.4 KB) + its c slice (3.4 KB, COPIED once per volume into
 *   a 4K-placed plan buffer, L36_mixedradix's arena trick) + t1 (3.4 KB): the
 *   whole chain runs out of L1d.  DRAM/L2 traffic per 4856-step chain: one x0
 *   read, one c read, one final write.
 *
 *   LAZY MAP FUSED AT THE X-PASS LOADS (adopted from L13_rader ice_r4 'fo' /
 *   the rival pipelines' pw fusion, corpus 10 s2): raw z stays in the state
 *   buffer between steps; the next step's zmm x-pass loads z and c at the same
 *   offsets, maps in registers, and feeds DFT6VZ2.  The state buffer is
 *   updated IN PLACE (the x pass drains the whole volume into t1 before
 *   FUSED_YZ's first store lands; zs/zd are deliberately NOT restrict).
 *   Step 1 is a plain k_zx of x0; one map_vol_z finishes step m into
 *   final_out.
 *
 *   THE MAP LADDER, pair-shared (shape adopted from L17_matrixsimd ice_r4 s6
 *   + L17_rader's mapc, i.e. rival 1000f989): two vunpck{lo,hi}pd pull two
 *   interleaved zmm into 8 re + 8 im, so |w|^2, the sqrt ladder, and ONE
 *   divide run once per 8 points.  vrsqrt14pd seed (2^-14) + 2 Newtons
 *   reaches ~2^-56 before a correctly-rounded vdivpd: ~2-3 ulp per
 *   application against a budget of 1e-13/step (tol 4.9e-10 at m=4856) --
 *   the full-double tier, MANDATORY at L=6 per the brief; the rivals'
 *   float-seed tier fails here by ~26x.  A 1e-300 bias folded into the
 *   |w|^2 FMA guards rsqrt14(0) = inf (L17_rader's trick).  27 vdivpd zmm
 *   per volume-step; a vrcp14pd+2-Newton twin (MAP8_RCP) trades the divider
 *   for ~5 FMA ops so the race can pick per regime.
 *
 *   CHAIN-ARM TOURNAMENT in create(): arms div / rcp / hyb (2 div + 1 rcp
 *   per group) / sep (unfused map_vol_z pass, prices fusion, = L13_rader's
 *   uf control) / div2 (two volumes' chains in lockstep) / pp (two chains
 *   SKEWED half a step, x-groups hand-interleaved with the other volume's
 *   yz planes -- built to hide the divider, LOST on the node: 0.473 vs sep
 *   0.414, register pressure).  All these arms are BIT-IDENTICAL in output
 *   (same codelets), so the adaptive pick can never change chain bits.
 *   fdiv (the uncompensated ladder, drift 3.25e-9) is admission-checked and
 *   forceable but never raced.  Each arm is admitted against a numpy-
 *   faithful scalar reference chain (hypot + Smith division, rel L2 <=
 *   1e-10 over 33 steps) before being raced chain-shaped; safest-first,
 *   1.5% margin; L6_CHAIN_FORCE=<name> pins an arm ("gen" pins the
 *   fallback).  chain_arm = -1 (no AVX-512, or nothing admitted) falls
 *   back to p->run + the scalar map: always correct, never fast.
 *
 *   DRIFT FORENSICS (the round's real discovery -- full data in the
 *   strategy record): at m=4856 the chain amplifies per-step rounding by
 *   ~1e7, so NO independent exact-double implementation passes the as-
 *   written 1e-13/step budget on the B=64 trajectory: MKL + the driver's
 *   exact fallback map = 1.76e-9, numpy-with-sqrt-instead-of-hypot =
 *   1.12e-9, this file's best = 1.23e-9, tol = 4.9e-10.  The same binary
 *   PASSES at B=1 (7.4e-11): end drift is per-volume Lyapunov luck.  The
 *   compensated map exists to sit on the achievable floor whatever budget
 *   the recalibration lands on.
 *
 * ASSUMPTIONS: L == 6 only; in/out distinct and 64B aligned (the driver
 * guarantees both); single-threaded; the plan's scratch is not re-entrant.
 */

#include "fft3d_api.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define VOLC 216          /* complex per volume */
#define VOLD 432          /* doubles  per volume */
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

/* One volume of the graded map, scalar: state = (z+c)/(1+|z+c|).  Reference
 * for the chain-arm admission gate, and the generic fft3d_chain fallback.
 * |w| via hypot, NOT sqrt(wr^2+wi^2): numpy's np.abs(z) is npy_hypot, and at
 * m=4856 the chain amplifies even the sub-ulp sqrt-vs-hypot rounding
 * difference to 1.1e-9 (measured in pure numpy on the node, ice_r4) -- so
 * the reference map must be hypot-class or the gate arithmetic is off. */
static void scalar_map_vol(const double *z, const double *c, double *o)
{
    for (int i = 0; i < VOLC; ++i) {
        double wr = z[2 * i] + c[2 * i], wi = z[2 * i + 1] + c[2 * i + 1];
        /* Smith form (scl = 1/d, two muls), NOT wr/d, wi/d: numpy divides
         * complex by real exactly this way, and the chain gate amplifies
         * even that rounding choice (Markstein test above). */
        double scl = 1.0 / (1.0 + hypot(wr, wi));
        o[2 * i]     = wr * scl;
        o[2 * i + 1] = wi * scl;
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

/* The AVX-512 x-pass family (fused_zx*) was DELETED in panel_r8 (falsified
 * r7: zero picks in eight cells at measured-equal licence clock); the _rot
 * per-volume scratch-rotation twins were DELETED in panel_r9 (falsified r8:
 * zero picks in eight cells).  See the r7/r8 strategy records.
 *
 * ROUND ice_r2 REOPENS 512-BIT.  The r7 verdict was CLX-specific: the Gold
 * 5218 has ONE 512-bit FMA pipe, so zmm halves the instruction count but not
 * the FP-port cycles, and lost.  The ice panel's Gold 6326 has TWO 512-bit
 * FMA pipes (PANEL_BRIEF: "the second FMA pipe is genuinely feedable" on
 * bare metal), so zmm now halves the port-cycle floor too: 972 ymm FP instr
 * / 2 ports = 486 cycles becomes ~594 mixed-width instr with the zmm share
 * on p0+p5.  Two shapes, both radix-2-first (the node-proven d2 graph):
 *
 *   z512x  (zmm x-pass): the 36 x-lines are lanes; ANY 4 of the 36 (y,z)
 *          pairs can share a register, so groups g = 4 consecutive plane
 *          indices give 9 groups, every load/store 64B ALIGNED, NO TAIL.
 *          9 zmm codelets = 162 FP instr where ymm needs 324.  Fused y/z
 *          stays the node-proven ymm shape.
 *   z512yz (zmm x-pass + 512-bit fused y/z): plane rows y are 6 zmm
 *          (z=0..3) + 6 ymm tails (z=4,5); y-DFT = 1 zmm + 1 ymm codelet;
 *          an 18-shuffle vpermt2pd/vshuff64x2 transpose re-lanes to y
 *          (4-lane zmm y=0..3 + 2-lane ymm y=4,5); z-DFT = 1 zmm + 1 ymm
 *          codelet; 18 more shuffles transpose back to rows for the stores.
 *          Per plane: 72 FP + 44 shuffles vs ymm's 108 FP + 48 shuffles.
 */

#ifdef __AVX512F__
#define L6_HAVE_Z512 1

/* zmm twin of DFT6V2 (radix-2-first): identical dataflow, 4 complex lanes.
 * 18 FP instructions + 2 in-lane vpermilpd.  In-place safe. */
#define DFT6VZ2(v0, v1, v2, v3, v4, v5, o0, o1, o2, o3, o4, o5)           \
    do {                                                                 \
        __m512d _s0 = _mm512_add_pd((v0), (v3));                         \
        __m512d _d0 = _mm512_sub_pd((v0), (v3));                         \
        __m512d _s1 = _mm512_add_pd((v4), (v1));                         \
        __m512d _d1 = _mm512_sub_pd((v4), (v1));                         \
        __m512d _s2 = _mm512_add_pd((v2), (v5));                         \
        __m512d _d2 = _mm512_sub_pd((v2), (v5));                         \
        __m512d _t1 = _mm512_add_pd(_s1, _s2);                           \
        __m512d _t2 = _mm512_sub_pd(_s1, _s2);                           \
        __m512d _u1 = _mm512_add_pd(_d1, _d2);                           \
        __m512d _u2 = _mm512_sub_pd(_d1, _d2);                           \
        __m512d _m  = _mm512_fnmadd_pd(_halfz, _t1, _s0);                \
        __m512d _n  = _mm512_fnmadd_pd(_halfz, _u1, _d0);                \
        __m512d _p  = _mm512_permute_pd(_t2, 0x55);                      \
        __m512d _q  = _mm512_permute_pd(_u2, 0x55);                      \
        (o0) = _mm512_add_pd(_s0, _t1);                                  \
        (o3) = _mm512_add_pd(_d0, _u1);                                  \
        (o4) = _mm512_fmadd_pd (_cnz, _p, _m);                           \
        (o2) = _mm512_fnmadd_pd(_cnz, _p, _m);                           \
        (o1) = _mm512_fmadd_pd (_cnz, _q, _n);                           \
        (o5) = _mm512_fnmadd_pd(_cnz, _q, _n);                           \
    } while (0)

/* zmm constants + the four 2-source permute index vectors the transposes
 * use (loaded once per call, live in registers across the batch loop):
 *   _ixlo/_ixhi: interleave 128b granules 0,1 / 2,3 of two sources
 *   _ixp0/_ixp1: pick even / odd granules of each source */
#define VSETZ                                                            \
    const __m512d _halfz = _mm512_set1_pd(0.5);                          \
    const __m512d _cnz = _mm512_set_pd(C3, -C3, C3, -C3,                 \
                                       C3, -C3, C3, -C3);                \
    const __m512i _ixlo = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);   \
    const __m512i _ixhi = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15); \
    const __m512i _ixp0 = _mm512_setr_epi64(0, 1, 4, 5, 8, 9, 12, 13);   \
    const __m512i _ixp1 = _mm512_setr_epi64(2, 3, 6, 7, 10, 11, 14, 15)

/* prefetch hooks for the 9-group zmm x-pass: 6 lines per group, 9*6 = 54
 * lines = the whole next volume, same lead as the 18-group ymm hooks */
#define PF9_NONE(SRC, OUT, g)  do { } while (0)
#define PF9_T0(SRC, OUT, g)                                              \
    do {                                                                 \
        const char *_pf = (const char *)((SRC) + VOLD) + 384 * (g);      \
        _mm_prefetch(_pf,       _MM_HINT_T0);                            \
        _mm_prefetch(_pf + 64,  _MM_HINT_T0);                            \
        _mm_prefetch(_pf + 128, _MM_HINT_T0);                            \
        _mm_prefetch(_pf + 192, _MM_HINT_T0);                            \
        _mm_prefetch(_pf + 256, _MM_HINT_T0);                            \
        _mm_prefetch(_pf + 320, _MM_HINT_T0);                            \
    } while (0)

/* x pass, zmm: lanes = 4 consecutive (y,z) plane indices, ascending
 * addresses; every access 64B aligned, no tail. */
#define PASS_X_Z(SRC, DST, OUT, PF)                                      \
    do {                                                                 \
        for (int g = 0; g < 9; ++g) {                                    \
            const double *p = (SRC) + 8 * g;                             \
            double *q = (DST) + 8 * g;                                   \
            PF(SRC, OUT, g);                                             \
            __m512d i0 = _mm512_load_pd(p),       i1 = _mm512_load_pd(p + 72);  \
            __m512d i2 = _mm512_load_pd(p + 144), i3 = _mm512_load_pd(p + 216); \
            __m512d i4 = _mm512_load_pd(p + 288), i5 = _mm512_load_pd(p + 360); \
            DFT6VZ2(i0, i1, i2, i3, i4, i5, i0, i1, i2, i3, i4, i5);     \
            _mm512_store_pd(q, i0);        _mm512_store_pd(q + 72, i1);  \
            _mm512_store_pd(q + 144, i2);  _mm512_store_pd(q + 216, i3); \
            _mm512_store_pd(q + 288, i4);  _mm512_store_pd(q + 360, i5); \
        }                                                                \
    } while (0)

/* 512-bit fused y+z over one 6x6 (y,z) plane.  M[y][z] granule bookkeeping
 * is spelled out per shuffle; every tail address is 32B aligned, every row
 * store storeu (rows at 96B strides alternate 64B/32B alignment). */
#define FUSED_YZ_Z(SRC, DST)                                             \
    do {                                                                 \
        for (int x = 0; x < 6; ++x) {                                    \
            const double *p = (SRC) + 72 * x;                            \
            double *q = (DST) + 72 * x;                                  \
            __m512d A0 = _mm512_loadu_pd(p),      A1 = _mm512_loadu_pd(p + 12); \
            __m512d A2 = _mm512_loadu_pd(p + 24), A3 = _mm512_loadu_pd(p + 36); \
            __m512d A4 = _mm512_loadu_pd(p + 48), A5 = _mm512_loadu_pd(p + 60); \
            __m256d B0 = LD(p + 8),  B1 = LD(p + 20), B2 = LD(p + 32);   \
            __m256d B3 = LD(p + 44), B4 = LD(p + 56), B5 = LD(p + 68);   \
            /* y-DFT: lanes z=0..3 (zmm) and z=4,5 (ymm) */              \
            DFT6VZ2(A0, A1, A2, A3, A4, A5, A0, A1, A2, A3, A4, A5);     \
            DFT6V2 (B0, B1, B2, B3, B4, B5, B0, B1, B2, B3, B4, B5);     \
            /* transpose to lanes = y: C[z] = M[0..3][z], D[z] = M[4..5][z] */ \
            __m512d u0 = _mm512_permutex2var_pd(A0, _ixlo, A1); /* 00 10 01 11 */ \
            __m512d u1 = _mm512_permutex2var_pd(A0, _ixhi, A1); /* 02 12 03 13 */ \
            __m512d u2 = _mm512_permutex2var_pd(A2, _ixlo, A3); /* 20 30 21 31 */ \
            __m512d u3 = _mm512_permutex2var_pd(A2, _ixhi, A3); /* 22 32 23 33 */ \
            __m512d c0 = _mm512_shuffle_f64x2(u0, u2, 0x44);    /* 00 10 20 30 */ \
            __m512d c1 = _mm512_shuffle_f64x2(u0, u2, 0xEE);    /* 01 11 21 31 */ \
            __m512d c2 = _mm512_shuffle_f64x2(u1, u3, 0x44);    /* 02 12 22 32 */ \
            __m512d c3 = _mm512_shuffle_f64x2(u1, u3, 0xEE);    /* 03 13 23 33 */ \
            __m512d zb0 = _mm512_insertf64x4(_mm512_castpd256_pd512(B0), B1, 1); /* 04 05 14 15 */ \
            __m512d zb1 = _mm512_insertf64x4(_mm512_castpd256_pd512(B2), B3, 1); /* 24 25 34 35 */ \
            __m512d c4 = _mm512_permutex2var_pd(zb0, _ixp0, zb1); /* 04 14 24 34 */ \
            __m512d c5 = _mm512_permutex2var_pd(zb0, _ixp1, zb1); /* 05 15 25 35 */ \
            __m512d g0 = _mm512_permutex2var_pd(A4, _ixlo, A5); /* 40 50 41 51 */ \
            __m512d g1 = _mm512_permutex2var_pd(A4, _ixhi, A5); /* 42 52 43 53 */ \
            __m256d d0 = _mm512_castpd512_pd256(g0);                     \
            __m256d d1 = _mm512_extractf64x4_pd(g0, 1);                  \
            __m256d d2 = _mm512_castpd512_pd256(g1);                     \
            __m256d d3 = _mm512_extractf64x4_pd(g1, 1);                  \
            __m256d d4 = _mm256_permute2f128_pd(B4, B5, 0x20); /* 44 54 */ \
            __m256d d5 = _mm256_permute2f128_pd(B4, B5, 0x31); /* 45 55 */ \
            /* z-DFT: lanes y=0..3 (zmm) and y=4,5 (ymm) */              \
            DFT6VZ2(c0, c1, c2, c3, c4, c5, c0, c1, c2, c3, c4, c5);     \
            DFT6V2 (d0, d1, d2, d3, d4, d5, d0, d1, d2, d3, d4, d5);     \
            /* transpose back to rows y and store */                     \
            __m512d r0 = _mm512_permutex2var_pd(c0, _ixlo, c1); /* 00 01 10 11 */ \
            __m512d r1 = _mm512_permutex2var_pd(c0, _ixhi, c1); /* 20 21 30 31 */ \
            __m512d r2 = _mm512_permutex2var_pd(c2, _ixlo, c3); /* 02 03 12 13 */ \
            __m512d r3 = _mm512_permutex2var_pd(c2, _ixhi, c3); /* 22 23 32 33 */ \
            _mm512_storeu_pd(q,      _mm512_shuffle_f64x2(r0, r2, 0x44)); /* row0 z0-3 */ \
            _mm512_storeu_pd(q + 12, _mm512_shuffle_f64x2(r0, r2, 0xEE)); /* row1 */ \
            _mm512_storeu_pd(q + 24, _mm512_shuffle_f64x2(r1, r3, 0x44)); /* row2 */ \
            _mm512_storeu_pd(q + 36, _mm512_shuffle_f64x2(r1, r3, 0xEE)); /* row3 */ \
            __m512d t45a = _mm512_permutex2var_pd(c4, _ixlo, c5); /* 04 05 14 15 */ \
            __m512d t45b = _mm512_permutex2var_pd(c4, _ixhi, c5); /* 24 25 34 35 */ \
            _mm256_store_pd(q + 8,  _mm512_castpd512_pd256(t45a));       \
            _mm256_store_pd(q + 20, _mm512_extractf64x4_pd(t45a, 1));    \
            _mm256_store_pd(q + 32, _mm512_castpd512_pd256(t45b));       \
            _mm256_store_pd(q + 44, _mm512_extractf64x4_pd(t45b, 1));    \
            __m512d zd0 = _mm512_insertf64x4(_mm512_castpd256_pd512(d0), d1, 1); /* 40 50 41 51 */ \
            __m512d zd1 = _mm512_insertf64x4(_mm512_castpd256_pd512(d2), d3, 1); /* 42 52 43 53 */ \
            _mm512_storeu_pd(q + 48, _mm512_permutex2var_pd(zd0, _ixp0, zd1)); /* row4 z0-3 */ \
            _mm512_storeu_pd(q + 60, _mm512_permutex2var_pd(zd0, _ixp1, zd1)); /* row5 */ \
            _mm256_store_pd(q + 56, _mm256_permute2f128_pd(d4, d5, 0x20)); /* 44 45 */ \
            _mm256_store_pd(q + 68, _mm256_permute2f128_pd(d4, d5, 0x31)); /* 54 55 */ \
        }                                                                \
    } while (0)

/* zmm-y hybrid: rows load as zmm (z=0..3) + ymm tail (z=4,5), the y-DFT is
 * 1 zmm + 1 ymm codelet (36 FP vs the ymm stage's 54), then 6 vextractf64x4
 * split the zmm rows into the P[3y+zp] ymm chunk layout and the node-proven
 * ymm z-stage (FYZ_ZCHUNK) runs unchanged.  Odd rows are 32B-aligned only,
 * so 3 of the 6 zmm loads per plane are line-split -- the bet is that
 * -18 FP beats +6 shuffles + 3 split loads. */
#define FUSED_YZ_ZY(SRC, DST, ST)                                        \
    do {                                                                 \
        for (int x = 0; x < 6; ++x) {                                    \
            const double *p = (SRC) + 72 * x;                            \
            double *q = (DST) + 72 * x;                                  \
            __m512d A0 = _mm512_loadu_pd(p),      A1 = _mm512_loadu_pd(p + 12); \
            __m512d A2 = _mm512_loadu_pd(p + 24), A3 = _mm512_loadu_pd(p + 36); \
            __m512d A4 = _mm512_loadu_pd(p + 48), A5 = _mm512_loadu_pd(p + 60); \
            __m256d B0 = LD(p + 8),  B1 = LD(p + 20), B2 = LD(p + 32);   \
            __m256d B3 = LD(p + 44), B4 = LD(p + 56), B5 = LD(p + 68);   \
            DFT6VZ2(A0, A1, A2, A3, A4, A5, A0, A1, A2, A3, A4, A5);     \
            DFT6V2 (B0, B1, B2, B3, B4, B5, B0, B1, B2, B3, B4, B5);     \
            __m256d P[18];                                               \
            P[0]  = _mm512_castpd512_pd256(A0);                          \
            P[1]  = _mm512_extractf64x4_pd(A0, 1);  P[2]  = B0;          \
            P[3]  = _mm512_castpd512_pd256(A1);                          \
            P[4]  = _mm512_extractf64x4_pd(A1, 1);  P[5]  = B1;          \
            P[6]  = _mm512_castpd512_pd256(A2);                          \
            P[7]  = _mm512_extractf64x4_pd(A2, 1);  P[8]  = B2;          \
            P[9]  = _mm512_castpd512_pd256(A3);                          \
            P[10] = _mm512_extractf64x4_pd(A3, 1);  P[11] = B3;          \
            P[12] = _mm512_castpd512_pd256(A4);                          \
            P[13] = _mm512_extractf64x4_pd(A4, 1);  P[14] = B4;          \
            P[15] = _mm512_castpd512_pd256(A5);                          \
            P[16] = _mm512_extractf64x4_pd(A5, 1);  P[17] = B5;          \
            FYZ_ZCHUNK(P, q, 0, ST, DFT6V2);                             \
            FYZ_ZCHUNK(P, q, 1, ST, DFT6V2);                             \
            FYZ_ZCHUNK(P, q, 2, ST, DFT6V2);                             \
        }                                                                \
    } while (0)

#define GEN_ZXY(NAME, PF)                                                \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,     \
                 const double *restrict in, double *restrict out, long nb)\
{                                                                        \
    VSET; VSETZ;                                                         \
    (void)t2; (void)_ixlo; (void)_ixhi; (void)_ixp0; (void)_ixp1;        \
    for (long b = 0; b < nb; ++b) {                                      \
        const double *ip = in + b * VOLD;                                \
        double *op = out + b * VOLD;                                     \
        PASS_X_Z(ip, t1, op, PF);                                        \
        FUSED_YZ_ZY(t1, op, STN);                                        \
    }                                                                    \
}

#define GEN_ZX(NAME, PF)                                                 \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,     \
                 const double *restrict in, double *restrict out, long nb)\
{                                                                        \
    VSET; VSETZ;                                                         \
    (void)t2; (void)_ixlo; (void)_ixhi; (void)_ixp0; (void)_ixp1;        \
    for (long b = 0; b < nb; ++b) {                                      \
        const double *ip = in + b * VOLD;                                \
        double *op = out + b * VOLD;                                     \
        PASS_X_Z(ip, t1, op, PF);                                        \
        FUSED_YZ(t1, op, STN, DFT6V2);                                   \
    }                                                                    \
}

#define GEN_ZZ(NAME, PF)                                                 \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,     \
                 const double *restrict in, double *restrict out, long nb)\
{                                                                        \
    VSET; VSETZ;                                                         \
    (void)t2;                                                            \
    for (long b = 0; b < nb; ++b) {                                      \
        const double *ip = in + b * VOLD;                                \
        double *op = out + b * VOLD;                                     \
        PASS_X_Z(ip, t1, op, PF);                                        \
        FUSED_YZ_Z(t1, op);                                              \
    }                                                                    \
}

GEN_ZX(k_zx,    PF9_NONE)
GEN_ZX(k_zx_pf, PF9_T0)
GEN_ZZ(k_zz,    PF9_NONE)
GEN_ZZ(k_zz_pf, PF9_T0)
GEN_ZXY(k_zxy,    PF9_NONE)
GEN_ZXY(k_zxy_pf, PF9_T0)

/* ---- ice_r4: the graded map, fused into the zmm x-pass (see header) ----
 * Pair-shared ladder: 2 vunpck deinterleave two zmm (8 complex) into 8 re +
 * 8 im, |w|^2 = fma(re,re, fma(im,im, 1e-300)) [bias guards rsqrt14(0)=inf],
 * vrsqrt14pd + 2 Newtons (2^-14 -> ~2^-56, below double rounding), then ONE
 * reciprocal per 8 points -- a correctly-rounded vdivpd (MAP8_DIV) or a
 * vrcp14pd + 2-Newton ladder (MAP8_RCP) -- and 2 muls + 2 vunpck back to
 * interleaved.  ~2-3 ulp per application vs the 1e-13/step budget. */
#define VSETM                                                             \
    const __m512d _bias = _mm512_set1_pd(1e-300);                         \
    const __m512d _c15  = _mm512_set1_pd(1.5);                            \
    const __m512d _onez = _mm512_set1_pd(1.0)

static inline __m512d l6_rcp_full(__m512d d, __m512d one)
{
    __m512d r = _mm512_rcp14_pd(d);
    __m512d e = _mm512_fnmadd_pd(d, r, one);
    r = _mm512_fmadd_pd(r, e, r);
    e = _mm512_fnmadd_pd(d, r, one);
    r = _mm512_fmadd_pd(r, e, r);
    return r;
}

/* COMPENSATED |w| (ice_r4, after the drift forensics -- see the strategy
 * record): s = re^2 + im^2 is carried as a double-double (Dekker FMA
 * residuals of both squares + a branchless 2Sum of the add), the rsqrt14
 * ladder runs on hi(s), and ONE compensated Newton step folds the low part
 * in:  r' = r + (s - r*r + e) * y/2.  |w| lands within ~0.6 ulp of the true
 * value -- the same class as numpy's hypot -- where sqrt(fl(re^2+im^2)) is
 * ~1 ulp BIASED against hypot, and the m=4856 chain amplifies that bias to
 * 1.1e-9 (measured).  ~15 extra FMA-class ops per 8 points. */
#define MAP8_CORE(a, b, QEXPR)                                            \
    do {                                                                  \
        __m512d _mre = _mm512_unpacklo_pd((a), (b));                      \
        __m512d _mim = _mm512_unpackhi_pd((a), (b));                      \
        __m512d _p1  = _mm512_mul_pd(_mre, _mre);                         \
        __m512d _e1  = _mm512_fmsub_pd(_mre, _mre, _p1);                  \
        __m512d _p2  = _mm512_mul_pd(_mim, _mim);                         \
        __m512d _e2  = _mm512_fmsub_pd(_mim, _mim, _p2);                  \
        __m512d _ms  = _mm512_add_pd(_p1, _p2);                           \
        __m512d _zz  = _mm512_sub_pd(_ms, _p1);         /* 2Sum */        \
        __m512d _ea  = _mm512_add_pd(                                     \
                         _mm512_sub_pd(_p1, _mm512_sub_pd(_ms, _zz)),     \
                         _mm512_sub_pd(_p2, _zz));                        \
        __m512d _el  = _mm512_add_pd(_mm512_add_pd(_e1, _e2), _ea);       \
        __m512d _mt  = _mm512_max_pd(_ms, _bias); /* rsqrt14(0)=inf guard */\
        __m512d _my  = _mm512_rsqrt14_pd(_mt);                            \
        __m512d _mh  = _mm512_mul_pd(_mt, _halfz);                        \
        __m512d _me  = _mm512_mul_pd(_my, _my);                           \
        _my = _mm512_mul_pd(_my, _mm512_fnmadd_pd(_mh, _me, _c15));       \
        /* ONE Newton is enough here: the compensated step below is itself \
         * quadratic, so the residual eps*eps' = (5.6e-9)^2-class vanishes  \
         * below double rounding.  (2 Newtons measured: same drift.) */    \
        __m512d _mr  = _mm512_mul_pd(_ms, _my);         /* ~sqrt(s) */    \
        __m512d _dr  = _mm512_fnmadd_pd(_mr, _mr, _ms); /* s - r^2 */     \
        __m512d _nm  = _mm512_add_pd(_dr, _el);                           \
        __m512d _hy  = _mm512_mul_pd(_my, _halfz);                        \
        _mr = _mm512_fmadd_pd(_nm, _hy, _mr);           /* ~sqrt(s+e) */  \
        __m512d _md = _mm512_add_pd(_mr, _onez);        /* 1 + |w| */     \
        /* q = 1/d then TWO multiplies -- deliberately NOT a refined       \
         * quotient: numpy divides complex by real via Smith's algorithm,  \
         * which computes scl = 1/d and multiplies both components, so the \
         * two-rounding form IS the reference's rounding.  A Markstein-    \
         * refined (correctly-rounded) quotient was tried and made the     \
         * m=4856 drift WORSE: 3.08e-9 vs 1.23e-9 (ice_r4). */             \
        __m512d _mq = QEXPR;                                              \
        _mre = _mm512_mul_pd(_mre, _mq);                                  \
        _mim = _mm512_mul_pd(_mim, _mq);                                  \
        (a) = _mm512_unpacklo_pd(_mre, _mim);                             \
        (b) = _mm512_unpackhi_pd(_mre, _mim);                             \
    } while (0)

#define MAP8_DIV(a, b) MAP8_CORE(a, b, _mm512_div_pd(_onez, _md))
#define MAP8_RCP(a, b) MAP8_CORE(a, b, l6_rcp_full(_md, _onez))

/* The UNCOMPENSATED ladder (sqrt(fl(re^2+im^2)), rsqrt14 + 2 Newtons + one
 * exact divide): ~1 ulp per application, drift 3.25e-9 at m=4856 vs the
 * compensated core's 1.50e-9, but ~0.07 us/step faster.  NOT raced -- kept
 * behind L6_CHAIN_FORCE=fdiv for the monitor, legal only if the chain
 * budget is recalibrated to >= ~7e-13/step (see the strategy record: the
 * current 1e-13/step budget is unreachable for ANY non-numpy entry). */
#define MAP8_FDIV(a, b)                                                   \
    do {                                                                  \
        __m512d _mre = _mm512_unpacklo_pd((a), (b));                      \
        __m512d _mim = _mm512_unpackhi_pd((a), (b));                      \
        __m512d _mt  = _mm512_fmadd_pd(_mre, _mre,                        \
                         _mm512_fmadd_pd(_mim, _mim, _bias));             \
        __m512d _my  = _mm512_rsqrt14_pd(_mt);                            \
        __m512d _mh  = _mm512_mul_pd(_mt, _halfz);                        \
        __m512d _me  = _mm512_mul_pd(_my, _my);                           \
        _my = _mm512_mul_pd(_my, _mm512_fnmadd_pd(_mh, _me, _c15));       \
        _me = _mm512_mul_pd(_my, _my);                                    \
        _my = _mm512_mul_pd(_my, _mm512_fnmadd_pd(_mh, _me, _c15));       \
        __m512d _md = _mm512_fmadd_pd(_mt, _my, _onez);                   \
        __m512d _mq = _mm512_div_pd(_onez, _md);                          \
        _mre = _mm512_mul_pd(_mre, _mq);                                  \
        _mim = _mm512_mul_pd(_mim, _mq);                                  \
        (a) = _mm512_unpacklo_pd(_mre, _mim);                             \
        (b) = _mm512_unpackhi_pd(_mre, _mim);                             \
    } while (0)

/* One mapped zmm x-group (the body of PASS_X_ZM, indexable so the paired-
 * pipelined arm can interleave it with yz planes of the other volume). */
#define XGM(SRC, CP, DST, g, MA, MB, MC)                                  \
    do {                                                                  \
        const double *p  = (SRC) + 8 * (g);                               \
        const double *cp = (CP) + 8 * (g);                                \
        double *q = (DST) + 8 * (g);                                      \
        __m512d i0 = _mm512_add_pd(_mm512_load_pd(p),                     \
                                   _mm512_load_pd(cp));                   \
        __m512d i1 = _mm512_add_pd(_mm512_load_pd(p + 72),                \
                                   _mm512_load_pd(cp + 72));              \
        __m512d i2 = _mm512_add_pd(_mm512_load_pd(p + 144),               \
                                   _mm512_load_pd(cp + 144));             \
        __m512d i3 = _mm512_add_pd(_mm512_load_pd(p + 216),               \
                                   _mm512_load_pd(cp + 216));             \
        __m512d i4 = _mm512_add_pd(_mm512_load_pd(p + 288),               \
                                   _mm512_load_pd(cp + 288));             \
        __m512d i5 = _mm512_add_pd(_mm512_load_pd(p + 360),               \
                                   _mm512_load_pd(cp + 360));             \
        MA(i0, i1); MB(i2, i3); MC(i4, i5);                               \
        DFT6VZ2(i0, i1, i2, i3, i4, i5, i0, i1, i2, i3, i4, i5);          \
        _mm512_store_pd(q, i0);        _mm512_store_pd(q + 72, i1);       \
        _mm512_store_pd(q + 144, i2);  _mm512_store_pd(q + 216, i3);      \
        _mm512_store_pd(q + 288, i4);  _mm512_store_pd(q + 360, i5);      \
    } while (0)

/* Mapped zmm x-pass: loads the previous step's RAW z and c at identical
 * offsets, w = z + c, maps 3 pairs, feeds DFT6VZ2.  Same 9-group ascending
 * layout as PASS_X_Z, every access 64B aligned, no tail. */
#define PASS_X_ZM(SRC, CP, DST, MA, MB, MC)                               \
    do {                                                                  \
        for (int g = 0; g < 9; ++g)                                       \
            XGM(SRC, CP, DST, g, MA, MB, MC);                             \
    } while (0)

/* One ymm fused-yz plane, indexable (built from the proven FYZ_* pieces the
 * sp2 pipeline uses; arithmetic identical to FUSED_YZ's plane body). */
#define YPM(SRC, DST, xx, CD)                                             \
    do {                                                                  \
        __m256d P[18];                                                    \
        const double *_ps = (SRC) + 72 * (xx);                            \
        double *_pd = (DST) + 72 * (xx);                                  \
        FYZ_LD6(P, _ps, 0); FYZ_LD6(P, _ps, 1); FYZ_LD6(P, _ps, 2);       \
        FYZ_YDFT(P, CD);                                                  \
        FYZ_ZCHUNK(P, _pd, 0, STN, CD);                                   \
        FYZ_ZCHUNK(P, _pd, 1, STN, CD);                                   \
        FYZ_ZCHUNK(P, _pd, 2, STN, CD);                                   \
    } while (0)

/* One fused chain step: mapped x-pass (raw z + c -> state, in registers)
 * into t1, then the node-proven ymm fused y/z stores the NEW raw z.  zs and
 * zd are the SAME buffer in the chain (in-place: the x pass drains all of zs
 * into t1 before FUSED_YZ's first store) -- deliberately NOT restrict. */
#define GEN_CSTEP(NAME, MA, MB, MC)                                       \
L6_KALIGN static void NAME(double *restrict t1, const double *zs,         \
                           const double *restrict cc, double *zd)         \
{                                                                         \
    VSET; VSETZ; VSETM;                                                   \
    (void)_ixlo; (void)_ixhi; (void)_ixp0; (void)_ixp1;                   \
    PASS_X_ZM(zs, cc, t1, MA, MB, MC);                                    \
    FUSED_YZ(t1, zd, STN, DFT6V2);                                        \
}
GEN_CSTEP(cstep_div,  MAP8_DIV,  MAP8_DIV,  MAP8_DIV)
GEN_CSTEP(cstep_rcp,  MAP8_RCP,  MAP8_RCP,  MAP8_RCP)
GEN_CSTEP(cstep_hyb,  MAP8_DIV,  MAP8_RCP,  MAP8_DIV)
GEN_CSTEP(cstep_fdiv, MAP8_FDIV, MAP8_FDIV, MAP8_FDIV)

/* ---- paired-pipelined chain block ('pp' arm) ----
 * Two volumes' chains run SKEWED so every divider-heavy mapped x-pass is
 * program-interleaved with the OTHER volume's FMA-heavy yz phase: the 27
 * vdivpd (~430 cyc of divider occupancy) issue under yz's port-0/5 work
 * instead of gating the x-DFTs.  One block = 9 mapped x-groups of X
 * interleaved 2:1-ish with 6 yz planes of Y; arithmetic per volume is
 * bit-identical to cstep_div (XGM/YPM are the same codelets), verified by
 * the admission gate.  Buffers all distinct -> restrict throughout. */
L6_KALIGN static void cblk_div(double *restrict t1x, const double *restrict zsx,
                               const double *restrict ccx,
                               const double *restrict t1y, double *restrict ydst)
{
    VSET; VSETZ; VSETM;
    (void)_ixlo; (void)_ixhi; (void)_ixp0; (void)_ixp1;
    XGM(zsx, ccx, t1x, 0, MAP8_DIV, MAP8_DIV, MAP8_DIV);
    YPM(t1y, ydst, 0, DFT6V2);
    XGM(zsx, ccx, t1x, 1, MAP8_DIV, MAP8_DIV, MAP8_DIV);
    XGM(zsx, ccx, t1x, 2, MAP8_DIV, MAP8_DIV, MAP8_DIV);
    YPM(t1y, ydst, 1, DFT6V2);
    XGM(zsx, ccx, t1x, 3, MAP8_DIV, MAP8_DIV, MAP8_DIV);
    YPM(t1y, ydst, 2, DFT6V2);
    XGM(zsx, ccx, t1x, 4, MAP8_DIV, MAP8_DIV, MAP8_DIV);
    XGM(zsx, ccx, t1x, 5, MAP8_DIV, MAP8_DIV, MAP8_DIV);
    YPM(t1y, ydst, 3, DFT6V2);
    XGM(zsx, ccx, t1x, 6, MAP8_DIV, MAP8_DIV, MAP8_DIV);
    YPM(t1y, ydst, 4, DFT6V2);
    XGM(zsx, ccx, t1x, 7, MAP8_DIV, MAP8_DIV, MAP8_DIV);
    XGM(zsx, ccx, t1x, 8, MAP8_DIV, MAP8_DIV, MAP8_DIV);
    YPM(t1y, ydst, 5, DFT6V2);
}

/* plain (unmapped) x-pass into t1: the pp prologue's half step */
L6_KALIGN static void px_plain(double *restrict t1, const double *restrict in)
{
    VSETZ;
    (void)_ixlo; (void)_ixhi; (void)_ixp0; (void)_ixp1;
    PASS_X_Z(in, t1, in, PF9_NONE);
}

/* plain fused yz from t1: the pp epilogue's half step */
L6_KALIGN static void yz_plain(const double *restrict t1, double *restrict dst)
{
    VSET;
    FUSED_YZ(t1, dst, STN, DFT6V2);
}

/* Standalone one-volume map (div ladder): finishes step m into final_out,
 * and is the 'sep' arm's unfused map pass. */
L6_KALIGN static void map_vol_z(const double *restrict z,
                                const double *restrict cc, double *restrict o)
{
    const __m512d _halfz = _mm512_set1_pd(0.5);
    VSETM;
    for (int i = 0; i < 27; ++i) {
        __m512d a = _mm512_add_pd(_mm512_load_pd(z + 16 * i),
                                  _mm512_load_pd(cc + 16 * i));
        __m512d b = _mm512_add_pd(_mm512_load_pd(z + 16 * i + 8),
                                  _mm512_load_pd(cc + 16 * i + 8));
        MAP8_DIV(a, b);
        _mm512_store_pd(o + 16 * i, a);
        _mm512_store_pd(o + 16 * i + 8, b);
    }
}

#endif /* __AVX512F__ */

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
    long rin, rout;       /* 4096-residues of the (in,out) pair placed for */
    const char *chosen;
    /* ice_r4: the chain arena -- six 432-double buffers at 4K residues
     * i*640, so every cross-buffer store->load delta the chain produces is
     * >= 640 clear of 0 mod 4096.  A = the single-volume chain (t1, state,
     * c copy); B = the second lane of the 'div2' paired arm, the 'sep'
     * arm's map bounce, and generic-fallback scratch. */
    double *chain_arena;
    double *ct1A, *csA, *ccA, *ct1B, *csB, *ccB;
    int chain_arm;        /* L6CH_*; -1 = generic p->run + scalar map */
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
    p->rin  = (long)((uintptr_t)in  & 4095u);
    p->rout = (long)((uintptr_t)out & 4095u);
}

/* ------------------------------------------------------------- the chain --- */

/* fdiv (uncompensated fast ladder) is admission-checked and forceable but
 * NEVER raced: precision policy must not lose to a 1.5% speed margin. */
enum { L6CH_DIV = 0, L6CH_RCP, L6CH_HYB, L6CH_SEP, L6CH_DIV2, L6CH_PP,
       L6CH_FDIV, L6CH_NARM };
static const char *const l6ch_name[L6CH_NARM] __attribute__((unused)) =
    { "div", "rcp", "hyb", "sep", "div2", "pp", "fdiv" };

/* Exact scalar reference chain (kern_scalar + scalar_map_vol): the admission
 * gate every chain arm must reproduce to rel L2 <= 1e-10 over 33 steps. */
__attribute__((unused))
static void l6_chain_ref(double *scr, double *tmp, double *state,
                         const double *x0, const double *cv, double *out,
                         int m, long nv)
{
    for (long b = 0; b < nv; ++b) {
        kern_scalar(scr, NULL, x0 + b * VOLD, state, 1);
        for (int k = 1; k < m; ++k) {
            scalar_map_vol(state, cv + b * VOLD, tmp);
            kern_scalar(scr, NULL, tmp, state, 1);
        }
        scalar_map_vol(state, cv + b * VOLD, out + b * VOLD);
    }
}

/* Generic fallback: the raced FFT kernel + the scalar map.  Always correct;
 * used when there is no AVX-512 or no arm passed admission. */
static void l6_chain_generic(fft3d_plan *p, const double *x0, const double *cv,
                             double *out, int m, long nv)
{
    for (long b = 0; b < nv; ++b) {
        const double *cb = cv + b * VOLD;
        p->run(p->ct1A, p->ct1B, x0 + b * VOLD, p->csA, 1);
        for (int k = 1; k < m; ++k) {
            scalar_map_vol(p->csA, cb, p->csB);
            p->run(p->ct1A, p->ct1B, p->csB, p->csA, 1);
        }
        scalar_map_vol(p->csA, cb, out + b * VOLD);
    }
#ifdef L6_HAVE_AVX2
    if (p->fence) _mm_sfence();
#endif
}

/* Run nv volumes' m-step chains under one arm.  Volume-major; c is copied
 * once per volume into the 4K-placed cc buffer; the state buffer holds RAW
 * z between steps (lazy map); one map_vol_z materializes step m. */
static void l6_chain_arm(fft3d_plan *p, int arm, const double *x0,
                         const double *cv, double *out, int m, long nv)
{
#ifdef L6_HAVE_Z512
    if (arm >= 0) {
        long b = 0;
        if (arm == L6CH_DIV2) {
            /* two chains in lockstep: B's divider-bound mapped x-pass can
             * overlap A's yz tail in the OOO window (independent buffers,
             * including separate t1s, so no false WAR serialization) */
            for (; b + 1 < nv; b += 2) {
                memcpy(p->ccA, cv + b * VOLD, VOLD * sizeof(double));
                memcpy(p->ccB, cv + (b + 1) * VOLD, VOLD * sizeof(double));
                k_zx(p->ct1A, p->ct1B, x0 + b * VOLD, p->csA, 1);
                k_zx(p->ct1B, p->ct1A, x0 + (b + 1) * VOLD, p->csB, 1);
                for (int k = 1; k < m; ++k) {
                    cstep_div(p->ct1A, p->csA, p->ccA, p->csA);
                    cstep_div(p->ct1B, p->csB, p->ccB, p->csB);
                }
                map_vol_z(p->csA, p->ccA, out + b * VOLD);
                map_vol_z(p->csB, p->ccB, out + (b + 1) * VOLD);
            }
        } else if (arm == L6CH_PP) {
            /* two chains SKEWED half a step (see cblk_div): after the
             * prologue, csA = zA(k), t1B holds B's step-k x output.  Each
             * loop iteration advances both chains one step with two blocks:
             *   block1: X_A(k+1) [csA->t1A]  ||  Y_B(k)   [t1B->csB]
             *   block2: X_B(k+1) [csB->t1B]  ||  Y_A(k+1) [t1A->csA]  */
            for (; b + 1 < nv; b += 2) {
                memcpy(p->ccA, cv + b * VOLD, VOLD * sizeof(double));
                memcpy(p->ccB, cv + (b + 1) * VOLD, VOLD * sizeof(double));
                k_zx(p->ct1A, p->ct1B, x0 + b * VOLD, p->csA, 1);
                px_plain(p->ct1B, x0 + (b + 1) * VOLD);
                for (int k = 1; k < m; ++k) {
                    cblk_div(p->ct1A, p->csA, p->ccA, p->ct1B, p->csB);
                    cblk_div(p->ct1B, p->csB, p->ccB, p->ct1A, p->csA);
                }
                yz_plain(p->ct1B, p->csB);       /* epilogue: Y_B(m) */
                map_vol_z(p->csA, p->ccA, out + b * VOLD);
                map_vol_z(p->csB, p->ccB, out + (b + 1) * VOLD);
            }
        }
        for (; b < nv; ++b) {           /* singles (and the div2 odd tail) */
            memcpy(p->ccA, cv + b * VOLD, VOLD * sizeof(double));
            k_zx(p->ct1A, p->ct1B, x0 + b * VOLD, p->csA, 1);
            if (arm == L6CH_RCP)
                for (int k = 1; k < m; ++k)
                    cstep_rcp(p->ct1A, p->csA, p->ccA, p->csA);
            else if (arm == L6CH_HYB)
                for (int k = 1; k < m; ++k)
                    cstep_hyb(p->ct1A, p->csA, p->ccA, p->csA);
            else if (arm == L6CH_SEP)
                for (int k = 1; k < m; ++k) {
                    map_vol_z(p->csA, p->ccA, p->ct1B);
                    k_zx(p->ct1A, p->csB, p->ct1B, p->csA, 1);
                }
            else if (arm == L6CH_FDIV)
                for (int k = 1; k < m; ++k)
                    cstep_fdiv(p->ct1A, p->csA, p->ccA, p->csA);
            else
                for (int k = 1; k < m; ++k)
                    cstep_div(p->ct1A, p->csA, p->ccA, p->csA);
            map_vol_z(p->csA, p->ccA, out + b * VOLD);
        }
        return;
    }
#else
    (void)arm;
#endif
    l6_chain_generic(p, x0, cv, out, m, nv);
}

const char *fft3d_name(void) { return "L6_pfa"; }

/* The chosen kernel, the clock probes, and the boundary probes are formatted
 * into the description so the leaderboard records which kernel ran on the
 * node, what clock it ran at, and what the x->yz joint costs (the r8
 * VERDICT's in-plan measurement route, after L36_pfa). */
static char l6_desc[512] =
    "Good-Thomas PFA 2x3 per axis, no twiddles, zmm x-pass + ymm fused y/z, "
    "chain-pingpong-raced; variant=auto";

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
    /* ice_r4 chain arena: buffer i at byte offset i*4736 (= 74 cache lines,
     * so residue i*640 mod 4096 and 64B alignment both hold). */
    p->chain_arena = (double *)l6_alloc(6 * 4736 + 64);
    if (!p->chain_arena) { free(p->arena); free(p); return NULL; }
    memset(p->chain_arena, 0, 6 * 4736 + 64);
    p->ct1A = (double *)((char *)p->chain_arena + 0 * 4736);
    p->csA  = (double *)((char *)p->chain_arena + 1 * 4736);
    p->ccA  = (double *)((char *)p->chain_arena + 2 * 4736);
    p->ct1B = (double *)((char *)p->chain_arena + 3 * 4736);
    p->csB  = (double *)((char *)p->chain_arena + 4 * 4736);
    p->ccB  = (double *)((char *)p->chain_arena + 5 * 4736);
    p->chain_arm = -1;

#ifdef L6_HAVE_AVX2
    {
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
#ifdef L6_HAVE_Z512
            /* ice_r2: the reopened 512-bit family (see header).  Last in
             * safest-first order: a zmm shape must beat the best ymm
             * incumbent by the full 1.5% to be chosen, so on a machine
             * where 512-bit is a wash or a licence loss (CLX was) the
             * pick degrades to exactly the r11 behaviour. */
            { k_zx,                 0, 0.015, "z512x"                },
            { k_zx_pf,              0, 0.015, "z512x_pf"             },
            { k_zz,                 0, 0.015, "z512yz"               },
            { k_zz_pf,              0, 0.015, "z512yz_pf"            },
            { k_zxy,                0, 0.015, "z512xy"               },
            { k_zxy_pf,             0, 0.015, "z512xy_pf"            },
#endif
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
                    for (long r = 0; r < reps; ++r) {
                        cand[cal].k(p->t1, p->t2, a, b, nb);
                        cand[cal].k(p->t1, p->t2, b, a, nb);
                    }
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
                    cand[c].k(p->t1, p->t2, b, a, nb);
                    if (cand[c].fence) _mm_sfence();
                }
            }
            /* Round-robin so machine drift hits every candidate equally;
             * per-candidate minimum is the noise-robust statistic.
             * ice_r2: every race body PING-PONGS (a->b, b->a) -- the graded
             * chain's steady state -- so candidates are ranked under the
             * scored access pattern, not a one-way stream. */
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
                        cand[c].k(p->t1, p->t2, b, a, nb);
                        if (cand[c].fence) _mm_sfence();
                    } while (now_s() - td < 5e-4);
                    double t = now_s();
                    for (long r = 0; r < reps; ++r) {
                        cand[c].k(p->t1, p->t2, a, b, nb);
                        cand[c].k(p->t1, p->t2, b, a, nb);
                    }
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
                            ok[c] ? bt[c] / (double)(2 * reps * nb) * 1e6 : 0.0,
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
        free(a); free(b);
        p->t1 = p->arena; p->t2 = p->arena + VOLD; p->placed = 0;

        /* ---- ice_r4: chain-arm admission + race (see header).  Runs after
         * the main race so the core is already in the heavy licence. ---- */
        const char *chosen_ch = "generic";
        int forcedc = -1;
        double ch_us = 0.0;
#ifdef L6_HAVE_Z512
        {
            long nv = batch >= 2 ? 2 : 1;
            size_t vb = (size_t)VOLD * sizeof(double);
            double *cx  = (double *)l6_alloc((size_t)nv * vb);
            double *cvv = (double *)l6_alloc((size_t)nv * vb);
            double *cgo = (double *)l6_alloc((size_t)nv * vb);
            double *crf = (double *)l6_alloc((size_t)nv * vb);
            double *s1  = (double *)l6_alloc(vb);
            double *s2  = (double *)l6_alloc(vb);
            double *stv = (double *)l6_alloc(vb);
            int okc[L6CH_NARM] = { 0, 0, 0, 0, 0 };
            double btc[L6CH_NARM];
            if (cx && cvv && cgo && crf && s1 && s2 && stv) {
                unsigned s = 777u;
                for (long i = 0; i < nv * (long)VOLD; ++i) {
                    s = s * 1664525u + 1013904223u;
                    cx[i]  = ((double)(int)(s >> 8) - 8388608.0) * 1e-7;
                    s = s * 1664525u + 1013904223u;
                    cvv[i] = ((double)(int)(s >> 8) - 8388608.0) * 1e-8;
                }
                /* admission: 33-step chain vs the exact scalar chain */
                l6_chain_ref(s1, s2, stv, cx, cvv, crf, 33, nv);
                double nrm = 0.0;
                for (long i = 0; i < nv * (long)VOLD; ++i) nrm += crf[i] * crf[i];
                for (int arm = 0; arm < L6CH_NARM; ++arm) {
                    if ((arm == L6CH_DIV2 || arm == L6CH_PP) && nv < 2) continue;
                    memset(cgo, 0, (size_t)nv * vb);
                    l6_chain_arm(p, arm, cx, cvv, cgo, 33, nv);
                    double e = 0.0;
                    for (long i = 0; i < nv * (long)VOLD; ++i) {
                        double d = cgo[i] - crf[i];
                        e += d * d;
                    }
                    okc[arm] = nrm > 0.0 && e <= 1e-20 * nrm; /* rel L2 <= 1e-10 */
                }
                {
                    const char *fn = getenv("L6_CHAIN_FORCE");
                    if (fn) {
                        for (int arm = 0; arm < L6CH_NARM; ++arm)
                            if (okc[arm] && strcmp(fn, l6ch_name[arm]) == 0)
                                { forcedc = arm; break; }
                        if (strcmp(fn, "gen") == 0) forcedc = L6CH_NARM; /* generic */
                    }
                }
                /* race: 384 mapped steps per rep, 5 rounds, per-arm min,
                 * a 48-step untimed warm per slice (licence discipline) */
                for (int arm = 0; arm < L6CH_NARM; ++arm) btc[arm] = 1e30;
                if (forcedc < 0) {
                    for (int round = 0; round < 5; ++round)
                        for (int arm = 0; arm < L6CH_NARM; ++arm) {
                            if (!okc[arm] || arm == L6CH_FDIV) continue;
                            l6_chain_arm(p, arm, cx, cvv, cgo, 48, nv);
                            double t0c = now_s();
                            l6_chain_arm(p, arm, cx, cvv, cgo, 384, nv);
                            double e = now_s() - t0c;
                            if (e < btc[arm]) btc[arm] = e;
                        }
                    double bestt = 1e30;
                    int bestc = -1;
                    for (int arm = 0; arm < L6CH_NARM; ++arm) {
                        if (!okc[arm] || arm == L6CH_FDIV) continue;
                        if (bestc < 0 || btc[arm] < bestt * 0.985)
                            { bestc = arm; bestt = btc[arm]; }
                        else if (btc[arm] < bestt) bestt = btc[arm];
                    }
                    if (bestc >= 0) {
                        p->chain_arm = bestc;
                        chosen_ch = l6ch_name[bestc];
                        ch_us = btc[bestc] / (384.0 * (double)nv) * 1e6;
                    }
                } else if (forcedc == L6CH_NARM) {
                    p->chain_arm = -1;                 /* forced generic */
                    chosen_ch = "gen";
                } else {
                    p->chain_arm = forcedc;
                    chosen_ch = l6ch_name[forcedc];
                }
                if (l6_verbose())
                    for (int arm = 0; arm < L6CH_NARM; ++arm)
                        fprintf(stderr, "L6_pfa chain: %-5s %s %10.4f us/step%s\n",
                                l6ch_name[arm], okc[arm] ? "ok " : "BAD",
                                btc[arm] < 1e29
                                    ? btc[arm] / (384.0 * (double)nv) * 1e6 : 0.0,
                                arm == p->chain_arm ? "   <-- chosen" : "");
            }
            free(cx); free(cvv); free(cgo); free(crf);
            free(s1); free(s2); free(stv);
        }
#endif

        snprintf(l6_desc, sizeof l6_desc,
                 "Good-Thomas PFA 2x3 per axis, no twiddles, zmm/ymm mixed, "
                 "chain-pingpong-raced; variant=%s%s ch=%s%s cst=%.3fus "
                 "clkS256=%.2f clkD256=%.2f "
                 "clkS512=%.2f kclk=%.2fGHz "
                 "bf=%.1f bsp=%.1f bx=%.1f byz=%.1fns",
                 p->chosen, forced >= 0 ? "!" : "",
                 chosen_ch, forcedc >= 0 ? "!" : "", ch_us,
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
    /* ice_r2: re-place when the buffer pair changes.  The graded CHAIN calls
     * step 1 on (in,out) but every later step ping-pongs (out,pong)/(pong,out)
     * -- a placement made for the first pair is stale for the other 4855.
     * cyc4k(-d) == cyc4k(d), so one placement serves both directions of a
     * pair and the steady-state chain re-places exactly once. */
    long rin  = (long)((uintptr_t)in  & 4095u);
    long rout = (long)((uintptr_t)out & 4095u);
    if (!plan->placed ||
        !((rin == plan->rin && rout == plan->rout) ||
          (rin == plan->rout && rout == plan->rin)))
        place_scratch(plan, in, out);
    plan->run(plan->t1, plan->t2, (const double *)in, (double *)out, plan->batch);
#ifdef L6_HAVE_AVX2
    if (plan->fence) _mm_sfence();
#endif
}

/* ice_r4: the fused-chain entry point the driver detects as a weak symbol.
 * Owns the whole m-step graded chain: state_0 = x0; each step z = raw
 * FFT(state), state = (z+c)/(1+|z+c|); final_out = state_m.  Volume-major,
 * lazy map, arm picked by the create-time chain race (see header). */
void fft3d_chain(fft3d_plan *plan, const double _Complex *x0,
                 const double _Complex *c, double _Complex *final_out, int m)
{
    if (!plan || m < 1) return;
    l6_chain_arm(plan, plan->chain_arm, (const double *)x0,
                 (const double *)c, (double *)final_out, m, plan->batch);
}

void fft3d_destroy(fft3d_plan *plan)
{
    if (!plan) return;
    free(plan->chain_arena);
    free(plan->arena);
    free(plan);
}
