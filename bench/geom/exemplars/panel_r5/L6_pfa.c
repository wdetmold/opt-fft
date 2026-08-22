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
 * 256-bit deliberately, not 512-bit: the scored Gold 5218 has ONE AVX-512 FMA
 * unit, so zmm buys zero FP throughput and pays licence downclocking; what we
 * do take from AVX-512 is the 32 evex ymm registers (the fused variant needs
 * ~26 live vectors).  Full argument in the round-1 strategy record.
 *
 * KERNEL SHAPES, raced at plan time (setup is not scored)
 * -------------------------------------------------------
 *   3pass    x: in->t1, y: t1->t2, z: t2->out.  Two scratch volumes.
 *            ADOPTED FROM L6_unrolled this round: on the node their
 *            two-scratch 3pass ran 0.572 us at B=32768 (panel_r2) where my
 *            in-place-y 3pass ran 0.637 (panel_r3), same nominal shape.
 *   3pass_ip x: in->t1, y: in place in t1, z: t1->out.  My original shape,
 *            kept as a raced candidate rather than deleted (panel_r3 VERDICT:
 *            "add candidates; do not replace structures").
 *   fused    x: in->t1, then y and z fused over one 6x6 (y,z) plane held in
 *            18 ymm (+ temps ~26 live): needs the target's 32 evex ymm.
 *            Won B=1/B=64/B=4096 on the node in panel_r3 (variant 2/6/6).
 *   *_u2     NEW panel_r5: the strided passes unrolled x2, two independent
 *            DFT6V chains per iteration (loads-loads-comp-comp-stores-stores
 *            program order).  Aimed at the node's small OoO window: the r4
 *            clock probe (L6_unrolled) showed the node at 3.89 GHz, so B=1's
 *            0.219 us is 852 cycles vs the 486-cycle FP floor -- B=1 is NOT
 *            FP-bound there, and CLX's ROB/RS (224/97 vs SPR 512/205) is the
 *            prime suspect.  u2 doubles the independent work per window slot.
 *   fused_sp NEW panel_r5: fused stage software-pipelined -- plane registers
 *            double-buffered (P even x, Q odd x), the next plane's 18 loads
 *            and 3 y-DFT6Vs interleaved by thirds into the current plane's
 *            z-chunks, so plane x+1 enters the ROB ~130 uops earlier than
 *            program order otherwise allows.  Peak ~30 live ymm: needs (and
 *            gets, on both wallaby and the node) the 32 evex ymm.
 * store policy: normal or NT (streaming).  NT has been rejected by the node
 *            tuner three rounds running at every batch size but wins wallaby's
 *            DRAM regime; kept because the tournament picks per machine.
 * prefetch hooks, one per x-pass group (3 cache lines each, 54 lines = the
 *            whole next volume, ~1 volume of lead):
 *   pf   = prefetcht0 of the next volume's INPUT
 *   pft1 = prefetcht1 of the next volume's INPUT (fills L2, not L1)
 *   w    = prefetchw of the next volume's OUTPUT -- write-intent prefetch,
 *          ADOPTED FROM L6_unrolled (panel_r3): with NT rejected on the node,
 *          every output line pays a write-allocate RFO; prefetchw issues that
 *          RFO one volume early, off the critical path.  Their fused_pfw won
 *          the node's B=32768 cell (0.563 us vs my 0.637).
 * Pruned in panel_r5 (never picked by the node in 3 rounds of stable pick
 * reporting): w-only, pfT1-without-W, 3pass_ip prefetch twins, distance-2,
 * and 4 of 6 NT kernels (NT is 0-for-4 rounds on the node; 3pass_nt and
 * 3pass_nt_pf remain as representatives).
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
 * pins candidate n (-1 = scalar reference path).
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
#define STS(p, v)  _mm256_stream_pd((double *)(p), (v))

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
#define PF_T1(SRC, OUT, g)   PF_IN(SRC, g, _MM_HINT_T1)
#define PF_W(SRC, OUT, g)    PF_OW(OUT, g)
#define PF_T0W(SRC, OUT, g)  do { PF_IN(SRC, g, _MM_HINT_T0); PF_OW(OUT, g); } while (0)
#define PF_T1W(SRC, OUT, g)  do { PF_IN(SRC, g, _MM_HINT_T1); PF_OW(OUT, g); } while (0)

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

/* ---- unroll-2 passes: two independent DFT6V chains per iteration ----
 * NEW in panel_r5, aimed at the node's small OoO window (CLX ROB 224 /
 * RS 97 vs SPR 512/205; the r4 clock probe showed the node at 3.89 GHz,
 * i.e. B=1 runs 852 cycles against a 486-cycle FP floor, so B=1 is NOT
 * FP-bound there -- the suspect is window-limited overlap).  Loads of
 * both groups precede both codelets, which precede all stores, so the
 * scheduler sees ~40 independent FP ops per iteration. */
#define PASS_X_U2(SRC, DST, OUT, PF)                                     \
    do {                                                                 \
        for (int zp = 0; zp < 3; ++zp)                                   \
            for (int y = 0; y < 6; y += 2) {                             \
                const double *p0 = (SRC) + 12 * y + 4 * zp;              \
                double *q0 = (DST) + 12 * y + 4 * zp;                    \
                PF(SRC, OUT, 6 * zp + y);                                \
                PF(SRC, OUT, 6 * zp + y + 1);                            \
                __m256d a0 = LD(p0),       a1 = LD(p0 + 72);             \
                __m256d a2 = LD(p0 + 144), a3 = LD(p0 + 216);            \
                __m256d a4 = LD(p0 + 288), a5 = LD(p0 + 360);            \
                __m256d b0 = LD(p0 + 12),  b1 = LD(p0 + 84);             \
                __m256d b2 = LD(p0 + 156), b3 = LD(p0 + 228);            \
                __m256d b4 = LD(p0 + 300), b5 = LD(p0 + 372);            \
                __m256d oa0, oa1, oa2, oa3, oa4, oa5;                    \
                __m256d ob0, ob1, ob2, ob3, ob4, ob5;                    \
                DFT6V(a0, a1, a2, a3, a4, a5, oa0, oa1, oa2, oa3, oa4, oa5);\
                DFT6V(b0, b1, b2, b3, b4, b5, ob0, ob1, ob2, ob3, ob4, ob5);\
                STN(q0, oa0); STN(q0 + 72, oa1); STN(q0 + 144, oa2);     \
                STN(q0 + 216, oa3); STN(q0 + 288, oa4); STN(q0 + 360, oa5);\
                STN(q0 + 12, ob0); STN(q0 + 84, ob1); STN(q0 + 156, ob2);\
                STN(q0 + 228, ob3); STN(q0 + 300, ob4); STN(q0 + 372, ob5);\
            }                                                            \
    } while (0)

#define PASS_Y_U2(SRC, DST)                                              \
    do {                                                                 \
        for (int xp = 0; xp < 3; ++xp)                                   \
            for (int zp = 0; zp < 3; ++zp) {                             \
                const double *p0 = (SRC) + 144 * xp + 4 * zp;            \
                double *q0 = (DST) + 144 * xp + 4 * zp;                  \
                __m256d a0 = LD(p0),      a1 = LD(p0 + 12);              \
                __m256d a2 = LD(p0 + 24), a3 = LD(p0 + 36);              \
                __m256d a4 = LD(p0 + 48), a5 = LD(p0 + 60);              \
                __m256d b0 = LD(p0 + 72), b1 = LD(p0 + 84);              \
                __m256d b2 = LD(p0 + 96), b3 = LD(p0 + 108);             \
                __m256d b4 = LD(p0 + 120), b5 = LD(p0 + 132);            \
                __m256d oa0, oa1, oa2, oa3, oa4, oa5;                    \
                __m256d ob0, ob1, ob2, ob3, ob4, ob5;                    \
                DFT6V(a0, a1, a2, a3, a4, a5, oa0, oa1, oa2, oa3, oa4, oa5);\
                DFT6V(b0, b1, b2, b3, b4, b5, ob0, ob1, ob2, ob3, ob4, ob5);\
                STN(q0, oa0); STN(q0 + 12, oa1); STN(q0 + 24, oa2);      \
                STN(q0 + 36, oa3); STN(q0 + 48, oa4); STN(q0 + 60, oa5); \
                STN(q0 + 72, ob0); STN(q0 + 84, ob1); STN(q0 + 96, ob2); \
                STN(q0 + 108, ob3); STN(q0 + 120, ob4); STN(q0 + 132, ob5);\
            }                                                            \
    } while (0)

#define PASS_Z_U2(SRC, DST, ST)                                          \
    do {                                                                 \
        for (int xp = 0; xp < 3; ++xp)                                   \
            for (int yp = 0; yp < 3; ++yp) {                             \
                const double *p0 = (SRC) + 144 * xp + 24 * yp;           \
                double *q0 = (DST) + 144 * xp + 24 * yp;                 \
                __m256d r0 = LD(p0),      r1 = LD(p0 + 4);               \
                __m256d r2 = LD(p0 + 8),  r3 = LD(p0 + 12);              \
                __m256d r4 = LD(p0 + 16), r5 = LD(p0 + 20);              \
                __m256d s0 = LD(p0 + 72), s1 = LD(p0 + 76);              \
                __m256d s2 = LD(p0 + 80), s3 = LD(p0 + 84);              \
                __m256d s4 = LD(p0 + 88), s5 = LD(p0 + 92);              \
                __m256d wa0, wa1, wa2, wa3, wa4, wa5;                    \
                __m256d wb0, wb1, wb2, wb3, wb4, wb5;                    \
                DFT6V(LO2(r0, r3), HI2(r0, r3), LO2(r1, r4), HI2(r1, r4),\
                      LO2(r2, r5), HI2(r2, r5), wa0, wa1, wa2, wa3, wa4, wa5);\
                DFT6V(LO2(s0, s3), HI2(s0, s3), LO2(s1, s4), HI2(s1, s4),\
                      LO2(s2, s5), HI2(s2, s5), wb0, wb1, wb2, wb3, wb4, wb5);\
                ST(q0,      LO2(wa0, wa1)); ST(q0 + 4,  LO2(wa2, wa3));  \
                ST(q0 + 8,  LO2(wa4, wa5)); ST(q0 + 12, HI2(wa0, wa1));  \
                ST(q0 + 16, HI2(wa2, wa3)); ST(q0 + 20, HI2(wa4, wa5));  \
                ST(q0 + 72, LO2(wb0, wb1)); ST(q0 + 76, LO2(wb2, wb3));  \
                ST(q0 + 80, LO2(wb4, wb5)); ST(q0 + 84, HI2(wb0, wb1));  \
                ST(q0 + 88, HI2(wb2, wb3)); ST(q0 + 92, HI2(wb4, wb5));  \
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

#define FUSED_YZ_SP(SRC, DST, ST)                                        \
    do {                                                                 \
        __m256d P[18], Q[18];                                            \
        FYZ_LD6(P, (SRC), 0); FYZ_LD6(P, (SRC), 1); FYZ_LD6(P, (SRC), 2);\
        FYZ_YDFT(P);                                                     \
        FYZ_PLANE(P, Q, 0, SRC, DST, ST);                                \
        FYZ_PLANE(Q, P, 1, SRC, DST, ST);                                \
        FYZ_PLANE(P, Q, 2, SRC, DST, ST);                                \
        FYZ_PLANE(Q, P, 3, SRC, DST, ST);                                \
        FYZ_PLANE(P, Q, 4, SRC, DST, ST);                                \
        FYZ_ZCHUNK(Q, (DST) + 360, 0, ST);                               \
        FYZ_ZCHUNK(Q, (DST) + 360, 1, ST);                               \
        FYZ_ZCHUNK(Q, (DST) + 360, 2, ST);                               \
    } while (0)

/* Rolled twin of FUSED_YZ_SP: planes 0-3 via a 2-iteration loop over P/Q
 * pairs (body ~450 insns, above gcc's complete-peel limit, so it stays a
 * loop), then plane 4 + the tail of plane 5.  Purpose: FUSED_YZ_SP fully
 * unrolls to ~7 KB of straight-line code per volume, which overflows the
 * node's ~1.5K-uop DSB and would put it on 16 B/cycle legacy decode; this
 * form is ~half the code and DSB-resident.  The sp-vs-sp2 race on the node
 * is a direct measurement of whether code footprint matters there. */
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

/* MID = t2 gives the two-scratch 3pass (adopted from L6_unrolled); MID = t1
 * gives the original in-place-y shape.  Both stay raced candidates. */
#define GEN_3P(NAME, MID, STZ, PF)                                       \
static void NAME(double *restrict t1, double *restrict t2,               \
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

#define GEN_FU(NAME, STZ, PF)                                            \
static void NAME(double *restrict t1, double *restrict t2,               \
                 const double *restrict in, double *restrict out, long nb)\
{                                                                        \
    VSET;                                                                \
    (void)t2;                                                            \
    for (long b = 0; b < nb; ++b) {                                      \
        const double *ip = in + b * VOLD;                                \
        double *op = out + b * VOLD;                                     \
        PASS_X(ip, t1, op, PF);                                          \
        FUSED_YZ(t1, op, STZ);                                           \
    }                                                                    \
}

/* unroll-2 3pass: all three passes with two interleaved DFT6V chains */
#define GEN_3P_U2(NAME, MID, PF)                                         \
static void NAME(double *restrict t1, double *restrict t2,               \
                 const double *restrict in, double *restrict out, long nb)\
{                                                                        \
    VSET;                                                                \
    (void)t2;                                                            \
    for (long b = 0; b < nb; ++b) {                                      \
        const double *ip = in + b * VOLD;                                \
        double *op = out + b * VOLD;                                     \
        PASS_X_U2(ip, t1, op, PF);                                       \
        PASS_Y_U2(t1, MID);                                              \
        PASS_Z_U2(MID, op, STN);                                         \
    }                                                                    \
}

/* unroll-2 x pass + register-resident fused y/z */
#define GEN_FU_U2(NAME, PF)                                              \
static void NAME(double *restrict t1, double *restrict t2,               \
                 const double *restrict in, double *restrict out, long nb)\
{                                                                        \
    VSET;                                                                \
    (void)t2;                                                            \
    for (long b = 0; b < nb; ++b) {                                      \
        const double *ip = in + b * VOLD;                                \
        double *op = out + b * VOLD;                                     \
        PASS_X_U2(ip, t1, op, PF);                                       \
        FUSED_YZ(t1, op, STN);                                           \
    }                                                                    \
}

/* x pass + software-pipelined (double-buffered plane) fused y/z */
#define GEN_FU_SP(NAME, PF)                                              \
static void NAME(double *restrict t1, double *restrict t2,               \
                 const double *restrict in, double *restrict out, long nb)\
{                                                                        \
    VSET;                                                                \
    (void)t2;                                                            \
    for (long b = 0; b < nb; ++b) {                                      \
        const double *ip = in + b * VOLD;                                \
        double *op = out + b * VOLD;                                     \
        PASS_X(ip, t1, op, PF);                                          \
        FUSED_YZ_SP(t1, op, STN);                                        \
    }                                                                    \
}

/* same, rolled plane-pair loop (DSB-resident code footprint) */
#define GEN_FU_SP2(NAME, PF)                                             \
static void NAME(double *restrict t1, double *restrict t2,               \
                 const double *restrict in, double *restrict out, long nb)\
{                                                                        \
    VSET;                                                                \
    (void)t2;                                                            \
    for (long b = 0; b < nb; ++b) {                                      \
        const double *ip = in + b * VOLD;                                \
        double *op = out + b * VOLD;                                     \
        PASS_X(ip, t1, op, PF);                                          \
        FUSED_YZ_SP2(t1, op, STN);                                       \
    }                                                                    \
}

GEN_3P(k_3p,          t2, STN, PF_NONE)
GEN_3P(k_3pip,        t1, STN, PF_NONE)
GEN_FU(k_fu,              STN, PF_NONE)
GEN_3P_U2(k_3p_u2,    t2,      PF_NONE)
GEN_FU_U2(k_fu_u2,             PF_NONE)
GEN_FU_SP(k_fu_sp,             PF_NONE)
GEN_FU_SP2(k_fu_sp2,           PF_NONE)
GEN_3P(k_3p_pf,       t2, STN, PF_T0)
GEN_FU(k_fu_pf,            STN, PF_T0)
GEN_3P_U2(k_3p_u2_pf, t2,      PF_T0)
GEN_FU_U2(k_fu_u2_pf,          PF_T0)
GEN_FU_SP(k_fu_sp_pf,          PF_T0)
GEN_FU_SP2(k_fu_sp2_pf,        PF_T0)
GEN_3P(k_3p_pfw,      t2, STN, PF_T0W)
GEN_FU(k_fu_pfw,           STN, PF_T0W)
GEN_FU(k_fu_pft1w,         STN, PF_T1W)
GEN_3P_U2(k_3p_u2_pfw, t2,     PF_T0W)
GEN_FU_U2(k_fu_u2_pfw,         PF_T0W)
GEN_FU_SP(k_fu_sp_pfw,         PF_T0W)
GEN_FU_SP2(k_fu_sp2_pfw,       PF_T0W)
GEN_3P(k_3p_nt,       t2, STS, PF_NONE)
GEN_3P(k_3p_nt_pf,    t2, STS, PF_T0)

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

/* The chosen kernel is spliced into the description so the leaderboard
 * records which one actually ran on the node (idea from L6_unrolled). */
static char l6_desc[224] =
    "Good-Thomas PFA 2x3 per axis, unrolled, 2 complex/ymm interleaved, "
    "no twiddles, plan-time raced kernels; variant=auto";

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
#endif

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
    /* 4096 B of slack for the 4K-aliasing placement + two scratch volumes */
    p->arena = (double *)l6_alloc(4096 + 2 * VOLD * sizeof(double) + 64);
    if (!p->arena) { free(p); return NULL; }
    memset(p->arena, 0, 4096 + 2 * VOLD * sizeof(double) + 64);
    p->t1 = p->arena;
    p->t2 = p->arena + VOLD;
    p->run = kern_scalar;
    p->fence = 0;
    p->placed = 0;
    p->chosen = "scalar";

#ifdef L6_HAVE_AVX2
    {
        /* Safest-first order: plain kernels, then input-prefetch, then the
         * prefetchw rows, then NT last.  A later candidate must beat the
         * running best by >1.5% to take over. */
        static const struct { l6_kernel k; int fence; const char *nm; } cand[] = {
            { k_3p,         0, "3pass"         },
            { k_3pip,       0, "3pass_ip"      },
            { k_fu,         0, "fused"         },
            { k_3p_u2,      0, "3pass_u2"      },
            { k_fu_u2,      0, "fused_u2"      },
            { k_fu_sp,      0, "fused_sp"      },
            { k_fu_sp2,     0, "fused_sp2"     },
            { k_3p_pf,      0, "3pass_pf"      },
            { k_fu_pf,      0, "fused_pf"      },
            { k_3p_u2_pf,   0, "3pass_u2_pf"   },
            { k_fu_u2_pf,   0, "fused_u2_pf"   },
            { k_fu_sp_pf,   0, "fused_sp_pf"   },
            { k_fu_sp2_pf,  0, "fused_sp2_pf"  },
            { k_3p_pfw,     0, "3pass_pfw"     },
            { k_fu_pfw,     0, "fused_pfw"     },
            { k_fu_pft1w,   0, "fused_pft1w"   },
            { k_3p_u2_pfw,  0, "3pass_u2_pfw"  },
            { k_fu_u2_pfw,  0, "fused_u2_pfw"  },
            { k_fu_sp_pfw,  0, "fused_sp_pfw"  },
            { k_fu_sp2_pfw, 0, "fused_sp2_pfw" },
            { k_3p_nt,      1, "3pass_nt"      },
            { k_3p_nt_pf,   1, "3pass_nt_pf"   },
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
                    fprintf(stderr, "L6_pfa race: %-14s %s %10.4f us/vol%s\n",
                            cand[c].nm, ok[c] ? "ok " : "BAD",
                            ok[c] ? bt[c] / (double)(reps * nb) * 1e6 : 0.0,
                            c == best ? "   <-- chosen" : "");
        } else {
            for (int c = 0; c < NCAND; ++c) if (ok[c]) { best = c; break; }
        }
        free(a); free(b);
        p->t1 = p->arena; p->t2 = p->arena + VOLD; p->placed = 0;

#ifdef L6_FORCE_VARIANT
        best = L6_FORCE_VARIANT;         /* development override only */
        if (best >= 0 && best < NCAND && !ok[best]) best = -1;
#endif
        if (best >= 0 && best < NCAND) {
            p->run    = cand[best].k;
            p->fence  = cand[best].fence;
            p->chosen = cand[best].nm;
        }
    }
#endif
    if (l6_verbose())
        fprintf(stderr, "L6_pfa: chosen %s for batch=%ld\n", p->chosen, p->batch);
    {
        char *v = strstr(l6_desc, "variant=");
        if (v) snprintf(v, (size_t)(l6_desc + sizeof(l6_desc) - v),
                        "variant=%s", p->chosen);
    }
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
