/* L = 17, complex-double forward 3D DFT, batched, single-threaded.
 *
 * NEW THIS ROUND (panel_r5):
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
 * X-first (only under -DL17R_XF_CUT, for a node A/B):
 *   x pass on in (axis stride NPL complex, 289 contiguous lanes,
 *                 deinterleaving load), split store    ->  A[kx][y][z]
 *   for each kx plane:
 *       A[kx][y][z]  --2 x 17x17 transpose-->  T[z][y]
 *       z pass on T                         ->  T[kz][y]
 *       17x17 transpose                     ->  U[y][kz]
 *       y pass on U, interleaving store (stride 17)  ->  out[kx][ky][kz]
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../fft3d_api.h"

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
 * Wallaby background (r4): X-first LOSES there in both batched regimes for
 * THIS pass structure -- B=256 12.40 vs 11.17 us/t, B=2048 26.9 vs 16.9
 * (strided y-store) and 23.3 vs 20.1 (sequential per-plane staging) --
 * unlike L17_matrixsimd, whose X-first won on the node.  The difference: my
 * y pass writes DRAM-destined output from inside the compute loop through
 * 17-row strided partial-line stores.  The node's 22 MB L3 makes B=256
 * truly stream, so only its own measurement (now automatic) can settle it. */
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

/* ------------------------------------------------------------------- plan */

struct fft3d_plan {
    int batch;
    int pf;                        /* cross-volume input prefetch enabled */
    double *mem;
    /* disjoint scratch per width, so each width's pad lanes stay zero */
    double *ar_w4, *ai_w4, *tr_w4, *ti_w4, *ur_w4, *ui_w4;
    double *ar_w8, *ai_w8, *tr_w8, *ti_w8, *ur_w8, *ui_w8;
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

static char g_desc[192] =
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
#  define L17R_NCA 7
#  define L17R_NCB 4
#else
#  define L17R_NCA 3
#  define L17R_NCB 3
#endif
static const l17r_fn l17r_cand_a[L17R_NCA] = {
    exec_np_w4, exec_np_w8, exec_npm_w8,
#if defined(__AVX512VL__)
    exec_pin_w8, exec_pinm_w8, exec_ovm_w8, exec_ovmpin_w8,
#endif
};
static const char *const l17r_tag_a[L17R_NCA] = {
    "xl 256", "xl 512", "xl 512t",
#if defined(__AVX512VL__)
    "xl 512 pin", "xl 512t pin", "xl 512t ov", "xl 512t ov pin",
#endif
};
static const l17r_fn l17r_cand_b[L17R_NCB] = {
    exec_xf_w4, exec_xf_w8, exec_xfm_w8,
#if defined(__AVX512VL__)
    exec_xfpinm_w8,
#endif
};
static const char *const l17r_tag_b[L17R_NCB] = {
    "xf 256", "xf 512", "xf 512t",
#if defined(__AVX512VL__)
    "xf 512t pin",
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
    int sb = p->batch, sp = p->pf;
    p->batch = nv;
    p->pf = 0;
    double t0 = l17r_now();
    do
        l17r_cand_a[L17R_NCA > 3 ? 1 : 0](p, p->tin, p->tout);
    while (l17r_now() - t0 < 0.15);
    p->batch = sb;
    p->pf = sp;
}

/* Time each candidate in a block of >= 64 consecutive volume transforms,
 * never interleaved (L17_matrixsimd round-1 item 12: interleaving ISA widths
 * mis-ranked candidates by 35% on the node).  Returns the fastest index. */
static int l17r_rank(fft3d_plan *p, int nv, double *best_us,
                     const l17r_fn *cand, int ncand, int reps)
{
    int inner = (64 + nv - 1) / nv;
    if (inner < 1) inner = 1;
    int sb = p->batch, sp = p->pf;
    p->batch = nv;
    p->pf = 0;
    int bestv = 0;
    for (int v = 0; v < ncand; ++v) {
        double best = 1e30;
        cand[v](p, p->tin, p->tout);   /* warmup: page in, settle clock */
        cand[v](p, p->tin, p->tout);
        for (int r = 0; r < reps; ++r) {
            double t0 = l17r_now();
            for (int q = 0; q < inner; ++q) cand[v](p, p->tin, p->tout);
            double dt = l17r_now() - t0;
            if (dt < best) best = dt;
        }
        best_us[v] = best * 1e6 / ((double)nv * inner);
        if (best_us[v] < best_us[bestv]) bestv = v;
    }
    p->batch = sb;
    p->pf = sp;
    return bestv;
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != LN || batch <= 0) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->batch = batch;
    p->pf = 0;
    size_t nd = (size_t)2*ABUF4 + (size_t)4*TBUF4
              + (size_t)2*ABUF8 + (size_t)4*TBUF8;
    void *raw = NULL;
    if (posix_memalign(&raw, 64, nd * sizeof(double)) != 0 || !raw) {
        free(p);
        return NULL;
    }
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
    p->ui_w8 = q;

    int bestv = 0;
    const char *const *tags = l17r_tag_a;
    p->exec = l17r_cand_a[0];

    if (batch < 64 && batch < L17R_XF_CUT) {
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
        }
#if defined(L17R_FORCE)
        bestv = (L17R_FORCE) % L17R_NCA;
        p->exec = l17r_cand_a[bestv];
#endif
    } else {
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
            double bpf[2];
            int sb = p->batch;
            p->batch = nv2;
            for (int f = 0; f < 2; ++f) {
                p->pf = f;
                bpf[f] = 1e30;
                p->exec(p, p->tin, p->tout);
                for (int r = 0; r < 4; ++r) {
                    double t0 = l17r_now();
                    p->exec(p, p->tin, p->tout);
                    double dt = l17r_now() - t0;
                    if (dt < bpf[f]) bpf[f] = dt;
                }
            }
            p->batch = sb;
            p->pf = bpf[1] < 0.97 * bpf[0];
            if (l17r_verbose())
                fprintf(stderr, "[L17_rader tune] nv=%d  pf off %.3f  pf on %.3f"
                                "  -> pf=%d\n", nv2, bpf[0] * 1e6 / nv2,
                        bpf[1] * 1e6 / nv2, p->pf);
        }
    }
    l17r_tune_free(p);

#if defined(L17R_FORCE_PF)
    p->pf = (L17R_FORCE_PF);
#endif

    /* measured sustained clock at both widths (unscored; VERDICT r4's ask) */
    double g256 = l17r_probe_ghz(0);
    double g512 = l17r_probe_ghz(1);

    snprintf(g_desc, sizeof g_desc,
             "Rader-17 cyclic/negacyclic (kernel from L17_winograd), "
             "plane-fused; tuned: %s, pf=%d, clk256=%.2f clk512=%.2f",
             tags[bestv], p->pf, g256, g512);
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    l17r_tune_free(p);   /* normally already freed at the end of create() */
    free(p->mem);
    free(p);
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
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

/* Compile-time flags:
 *   pin    -- S constants pinned in registers (EVEX only, tuner candidate)
 *   xfirst -- X-first pass order (bit class B, batch >= L17R_XF_CUT only)
 *   mixed  -- VW==8 only: 17-lane passes run 2 zmm blocks + 1 ymm tail
 *             (wino17_w4), and the x pass's clamped last block runs at ymm.
 *             On a 1-FMA-unit part the ymm tail halves the tail block's
 *             cycles; recomputed overlap lanes are bit-identical. */
static inline __attribute__((always_inline)) void SFX(exec_body)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out,
        const int pin, const int xfirst, const int mixed)
{
    double *const ar = p->SFX(ar), *const ai = p->SFX(ai);
    double *const tr = p->SFX(tr), *const ti = p->SFX(ti);
    double *const ur = p->SFX(ur), *const ui = p->SFX(ui);
    (void)mixed;

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

            /* ---- finish one kx plane at a time, storing straight to out:
             * the output writes spread across the volume's compute
             * (L17_matrixsimd panel_r3's X-first reorder). ---- */
            for (int kx = 0; kx < LN; ++kx) {
                if (nxt) {
                    const double *pp = nxt + 2*(long)kx*NPL;
                    for (int q = 0; q < 2*NPL; q += 8)
                        __builtin_prefetch(pp + q, 0, 2);
                }

                /* A[kx][y][z] -> T[z][y] */
                transpose17(ar + (long)kx*PS, LN, tr, TR);
                transpose17(ai + (long)kx*PS, LN, ti, TR);

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

                transpose17(tr, TR, ur, TR);       /* T[kz][y] -> U[y][kz] */
                transpose17(ti, TR, ui, TR);

                /* y pass: axis stride TR, lanes = kz, interleaving store
                 * into out[kx][ky][kz] (output stride 17 complex) */
                double *dp = dst + 2*(long)kx*NPL;
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
        } else {
            /* ---------------- X-last (round-1 order) ---------------- */
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
                deint_transpose17(src + 2*(long)x*NPL, tr, ti, TR);

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

                transpose17(tr, TR, ur, TR);       /* T[kz][y] -> U[y][kz] */
                transpose17(ti, TR, ui, TR);

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
#if VW == 8
            if (mixed) {
#pragma GCC unroll 1
                for (long m0 = 0; m0 + VW <= NPL; m0 += VW)
                    SFX(wino17)(ar + m0, ai + m0, PS, 0, 0, 0, 0, dst, m0,
                                NPL, 0, 1, pin);
                wino17_w4(ar + (NPL - 4), ai + (NPL - 4), PS, 0, 0, 0, 0,
                          dst, NPL - 4, NPL, 0, 1, 0);
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
        deint_transpose17_part(src, tr, ti, TR, 0);
        deint_transpose17_part(src, tr, ti, TR, 1);

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
            transpose17_part(tr, TR, ur, TR, 0);
            transpose17_part(ti, TR, ui, TR, 0);

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
                        transpose17_part(tr, TR, ur, TR, 1);
                        transpose17_part(ti, TR, ui, TR, 1);
                    } else if (x < 16) {
                        /* T is dead now: next plane's deinterleave, half 1 */
                        deint_transpose17_part(nsrc, tr, ti, TR, 0);
                    }
                }
            }
            wino17_w4(ur + 13, ui + 13, TR, 0, 0, dr + 13, di + 13,
                      0, 0, LN, 0, 0, 0);
            if (x < 16)
                deint_transpose17_part(nsrc, tr, ti, TR, 1);
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
#endif

/* exec variants.  VW==4 contributes the plain X-last and X-first entries;
 * VW==8 additionally contributes the pinned and mixed-tail ("t") ones. */
static void __attribute__((unused)) SFX(exec_np)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 0);
}

static void __attribute__((unused)) SFX(exec_xf)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 1, 0);
}

#if VW == 8
static void __attribute__((unused)) SFX(exec_pin)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 1, 0, 0);
}

static void __attribute__((unused)) SFX(exec_npm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1);
}

static void __attribute__((unused)) SFX(exec_pinm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 1, 0, 1);
}

static void __attribute__((unused)) SFX(exec_xfm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 1, 1);
}

static void __attribute__((unused)) SFX(exec_xfpinm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 1, 1, 1);
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
#endif

#undef NXB
#undef NLB
#undef KPIN
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
