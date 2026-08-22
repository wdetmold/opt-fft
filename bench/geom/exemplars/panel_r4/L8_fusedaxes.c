/* L8_fusedaxes -- forward complex-double 3D DFT of an 8x8x8 cube with all three
 * axes fused: one trip in from memory, one trip out, nothing but an 8 KiB L1
 * scratch in between.
 *
 * TECHNIQUE
 *   Split-complex, and the SIMD width is filled by a *spatial* axis, never by the
 *   batch: one 64-byte vector holds the 8 values of one axis, so every 8-point DFT
 *   is 8 lanes wide with lane-invariant twiddles and contains zero shuffles.  The
 *   volume is carried in two lane assignments and the change of assignment is done
 *   with in-register transposes, so B=1 is as wide as B=2048.
 *
 *     phase A (per x-plane)  lane = z, register index = y.
 *         A z-pencil is 8 contiguous complex; vunpcklpd/vunpckhpd of its two
 *         vectors split it into Re/Im (lane l holds z = PI[l], PI = 0,4,1,5,2,6,3,7).
 *         The y-axis DFT is then elementwise across registers: 0 shuffles.
 *         Result parked in the 8 KiB L1 scratch, reindexed [y][x].
 *     phase B (per y)        lane = z, register index = x.
 *         x-axis DFT elementwise (0 shuffles).  A 24-op in-register 8x8 transpose
 *         per Re/Im group turns (reg=x, lane=z) into (reg=z, lane=x), so the z-axis
 *         DFT is elementwise too.  The way back out is a single 48-op network over
 *         all 16 registers that performs the inverse transpose *and* the complex
 *         re-interleave at once (3 bit-swaps: ri->lane0, k2_0->lane1, k2_1->lane2),
 *         landing the 16 result vectors exactly in the driver's interleaved layout.
 *
 *   Every shuffle in the kernel is a 3-operand non-destructive form -- vshuff64x2
 *   (two patterns: a pure register/lane2 swap, and a register->lane2->lane1 cycle)
 *   or vunpcklpd/vunpckhpd -- so the kernel emits *no* register-copy instructions
 *   competing for port 5, which is the scarce port here.
 *
 * OPERATION COUNT (per 8^3 volume)  -- unchanged since round 1, both at their floor
 *   1296 vector FP instructions (3 axes * 8 groups * 54; 8 useful lanes each)
 *    896 vector shuffles (128 deinterleave + 384 transpose + 384 fused
 *                         inverse-transpose/interleave)
 *    256 vector loads, 256 vector stores, 0 spills, 0 register copies.
 *   56 real flops per 8-point DFT is the published optimum (Burrus T7.1 = FFTW
 *   n1_8); 896 shuffles is provably minimal for this layout.  Round panel_r2's
 *   node data (three entries within 1.2% for two rounds; radix8's 54->52 codelet
 *   cut measured +0.7% WORSE) says B=1 on the node is NOT moved by instruction
 *   cuts, so the arithmetic is frozen.
 *
 * STORE POLICY AND PREFETCH ARE MEASURED, NOT RULED (new in round panel_r3)
 *   Round r2 shipped rules (NT iff in+out > 1.18*L3, prefetch iff NT).  Two pieces
 *   of node evidence against rules: (a) the prefetch-iff-NT coupling prefetched an
 *   L3-RESIDENT input at node B=2048 (in = 16 MiB < 22 MiB L3), which is the exact
 *   regime my own r2 wallaby measurement showed costing 1.6x; the code comment
 *   even said "prefetch only when the input alone exceeds L3" while the code did
 *   pf = nt.  (b) The panel_r2 VERDICT shows NT stores LOSING on the node at L=6
 *   at every batch size, including 216 MiB working sets, against wallaby showing
 *   NT winning 1.6x -- the store policy inverts between machines and cannot be
 *   chosen by arithmetic.  So fft3d_create() now times all six
 *   (plain|nt) x (no-pf|pf-t1|pf-t0) variants on a surrogate batch ON THE MACHINE
 *   THAT SCORES, round-robin interleaved (drift-robust trial protocol adopted
 *   from L8_batchsimd / L8_radix8), min-of-5, with a 2% hysteresis toward the
 *   rule-based default.  The pick is reported through fft3d_description() so it
 *   lands in the leaderboard JSON (the panel_r2 VERDICT's cross-cutting ask).
 *   The tuner only runs when in+out > 0.25*L3; below that the plain path is
 *   already known optimal and B=1 / B=64 stay byte-identical to rounds 1-2.
 *
 * ASSUMPTIONS
 *   * L == 8 only.
 *   * in/out 64-byte aligned and distinct (the driver guarantees both); every
 *     z-pencil therefore starts 128-byte aligned and all vector accesses are
 *     naturally aligned.
 *   * GCC/Clang vector extensions.  ONE arithmetic path for every ISA: a 64-byte
 *     `v8d` becomes zmm on the scored Cascade Lake node and a 2 x ymm pair on the
 *     AVX2 development node, with identical arithmetic in identical order, so what
 *     is verified locally is what runs there.  The only __AVX512F__-guarded code
 *     is the non-temporal store.
 *
 * SEQ3 SHAPE (new in round panel_r4, adopted from L8_radix8's node B=2048 win)
 *   Six additional tuner candidates with a third pass that makes the output
 *   store stream fully SEQUENTIAL: phase A unchanged -> pass B1 (x-axis DFT,
 *   scratch1 row -> scratch2 [k0][k1], zero shuffles) -> pass B2 per k0
 *   (transpose + z-axis DFT + fused untranspose/interleave, then the 16
 *   half-pencils of the 1 KiB k0-plane stored in ascending address order).
 *   Identical FP and shuffle counts to the fused shape; +128 loads +128 stores,
 *   all L1-resident.  L8_radix8 measured this store order worth -18.5% at node
 *   B=2048 while my fused scatter (8 interleaved write streams) kept B=16384 --
 *   so BOTH shapes are candidates and the node's tuner picks per batch (the
 *   panel_r3 VERDICT's "add candidates, do not replace structures").
 *
 * COMPILE-TIME SWITCHES (default 0 / auto is what ships)
 *   L8_MODE   0 fused (ships) | 1 three passes via L1 scratch | 2 three passes
 *             over the whole batch (controls, unchanged since round 1)
 *   L8_NT     -1 auto, 0/1 force store policy (forcing skips the tuner)
 *   L8_PFSEL  -1 auto, 0/1 force prefetch     (forcing skips the tuner)
 *   L8_SHAPE  -1 auto, 0 force fused, 1 force seq3 (forcing skips the tuner)
 *   L8_PF_LOC 3 = prefetcht0, 2 = t1 (only honoured under a forced L8_PFSEL=1;
 *             t2/nta were measured losers in r2 and are no longer plumbed)
 *   L8_PF_DIST prefetch distance in volumes (1; r2 measured 2 as worse)
 *   L8_PF     0 compiles every prefetch out
 *   L8_TUNE   0 disables the plan-time tuner (rule-based auto only)
 */
#include <complex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../fft3d_api.h"

#ifndef L8_MODE
#define L8_MODE 0
#endif
#ifndef L8_PF_LOC
#define L8_PF_LOC 2
#endif
#ifndef L8_PF_DIST
#define L8_PF_DIST 1
#endif
#ifndef L8_NT
#define L8_NT (-1)
#endif
#ifndef L8_PFSEL
#define L8_PFSEL (-1)
#endif
#ifndef L8_SHAPE
#define L8_SHAPE (-1)
#endif
#ifndef L8_TUNE
#define L8_TUNE 1
#endif

typedef double v8d __attribute__((vector_size(64)));
typedef long long v8i __attribute__((vector_size(64)));
/* may_alias + minimal alignment: load/store straight out of the driver's
 * double _Complex buffers with no strict-aliasing question. */
typedef double v8du __attribute__((vector_size(64), aligned(8), may_alias));

#define SQ 0.70710678118654752440084436210485 /* 1/sqrt(2) = cos(pi/4) */

#define LD(p)     (*(const v8du *)(const void *)(p))
#define ST(p, v)  (*(v8du *)(void *)(p) = (v))
#if defined(__clang__)
#define SH(a, b, ...) __builtin_shufflevector((a), (b), __VA_ARGS__)
#else
#define SH(a, b, ...) __builtin_shuffle((a), (b), (v8i){__VA_ARGS__})
#endif

/* The three non-destructive 2-in/2-out lane primitives, as (bit acted on) ->
 * (bit permutation).  r = the register bit distinguishing the pair.
 *   T1  vunpcklpd / vunpckhpd    : r <-> lane0
 *   T2  vshuff64x2 (0,1|0,1)/(2,3|2,3) : r <-> lane2
 *   T3  vshuff64x2 (0,2|0,2)/(1,3|1,3) : r -> lane2 -> lane1 -> r          */
#define T1_LO 0,8,2,10,4,12,6,14
#define T1_HI 1,9,3,11,5,13,7,15
#define T2_LO 0,1,2,3,8,9,10,11
#define T2_HI 4,5,6,7,12,13,14,15
#define T3_LO 0,1,4,5,8,9,12,13
#define T3_HI 2,3,6,7,10,11,14,15

/* in-place non-destructive butterfly: (a,b) <- (SH(a,b,LO), SH(a,b,HI)) */
#define BF(a, b, LO, HI)                                                       \
    do {                                                                      \
        const v8d bf_ = SH((a), (b), LO);                                     \
        (b) = SH((a), (b), HI);                                               \
        (a) = bf_;                                                            \
    } while (0)

/* ---- 8-point complex DFT, split layout, 8 independent transforms in the lanes.
 * r[j], q[j] = real/imag of element j of the transformed axis.  Radix-8 DIF;
 * 52 adds + 4 multiplies, no shuffles, no cross-lane anything.               */
static inline void dft8s(v8d *restrict r, v8d *restrict q)
{
    const v8d t0r = r[0] + r[4], t1r = r[1] + r[5], t2r = r[2] + r[6], t3r = r[3] + r[7];
    const v8d t0i = q[0] + q[4], t1i = q[1] + q[5], t2i = q[2] + q[6], t3i = q[3] + q[7];
    const v8d s0r = r[0] - r[4], s1r = r[1] - r[5], s2r = r[2] - r[6], s3r = r[3] - r[7];
    const v8d s0i = q[0] - q[4], s1i = q[1] - q[5], s2i = q[2] - q[6], s3i = q[3] - q[7];

    /* odd half: b_j = s_j * W8^j, W8 = exp(-2 pi i/8).  W8^0 = 1 and W8^2 = -i are
     * free (the -i is a rename plus a sign folded into the adds below). */
    const v8d b1r = (s1r + s1i) * SQ, b1i = (s1i - s1r) * SQ;
    const v8d b3r = (s3i - s3r) * SQ, b3i = (s3i + s3r) * -SQ;

    { /* even outputs X[2k] = DFT4(t)[k] */
        const v8d u0r = t0r + t2r, u0i = t0i + t2i;
        const v8d u1r = t1r + t3r, u1i = t1i + t3i;
        const v8d v0r = t0r - t2r, v0i = t0i - t2i;
        const v8d dr  = t1r - t3r, di  = t1i - t3i;   /* v1 = -i*d = (di, -dr) */
        r[0] = u0r + u1r; q[0] = u0i + u1i;
        r[4] = u0r - u1r; q[4] = u0i - u1i;
        r[2] = v0r + di;  q[2] = v0i - dr;
        r[6] = v0r - di;  q[6] = v0i + dr;
    }
    { /* odd outputs X[2k+1] = DFT4(b)[k]; b0 = s0, b2 = -i*s2 folded in */
        const v8d u0r = s0r + s2i, u0i = s0i - s2r;
        const v8d u1r = b1r + b3r, u1i = b1i + b3i;
        const v8d v0r = s0r - s2i, v0i = s0i + s2r;
        const v8d dr  = b1r - b3r, di  = b1i - b3i;
        r[1] = u0r + u1r; q[1] = u0i + u1i;
        r[5] = u0r - u1r; q[5] = u0i - u1i;
        r[3] = v0r + di;  q[3] = v0i - dr;
        r[7] = v0r - di;  q[7] = v0i + dr;
    }
}

/* 8x8 transpose, 24 non-destructive lane permutes: T2 on register bit 2, T3 on
 * bit 1, T1 on bit 0.  Verified index map (see strategies record):
 *   in  register x, lane l holding z = PI[l]
 *   out register j holding z = PI[j], lane l holding x = 0,1,4,5,2,3,6,7  */
static inline void trans8(v8d *restrict m)
{
    BF(m[0], m[4], T2_LO, T2_HI);  BF(m[1], m[5], T2_LO, T2_HI);
    BF(m[2], m[6], T2_LO, T2_HI);  BF(m[3], m[7], T2_LO, T2_HI);
    BF(m[0], m[2], T3_LO, T3_HI);  BF(m[1], m[3], T3_LO, T3_HI);
    BF(m[4], m[6], T3_LO, T3_HI);  BF(m[5], m[7], T3_LO, T3_HI);
    BF(m[0], m[1], T1_LO, T1_HI);  BF(m[2], m[3], T1_LO, T1_HI);
    BF(m[4], m[5], T1_LO, T1_HI);  BF(m[6], m[7], T1_LO, T1_HI);
}

/* Inverse transpose AND complex interleave in one 48-op network over all 16
 * registers (r = real part per k2, q = imag part per k2, lane = k0).  Bit-level:
 * T3 on k2 bit0, T3 on k2 bit1, T1 on the real/imag bit, which lands
 * (lane2,lane1,lane0) = (k2_1, k2_0, re/im) -- exactly interleaved complex.
 * Afterwards register (r|q)[j] is a ready-to-store half-pencil; the (k0, half)
 * it belongs to is the OUT_K0/OUT_H table below. */
static inline void untrans_interleave(v8d *restrict r, v8d *restrict q)
{
    BF(r[0], r[1], T3_LO, T3_HI);  BF(r[2], r[3], T3_LO, T3_HI);
    BF(r[4], r[5], T3_LO, T3_HI);  BF(r[6], r[7], T3_LO, T3_HI);
    BF(q[0], q[1], T3_LO, T3_HI);  BF(q[2], q[3], T3_LO, T3_HI);
    BF(q[4], q[5], T3_LO, T3_HI);  BF(q[6], q[7], T3_LO, T3_HI);

    BF(r[0], r[2], T3_LO, T3_HI);  BF(r[1], r[3], T3_LO, T3_HI);
    BF(r[4], r[6], T3_LO, T3_HI);  BF(r[5], r[7], T3_LO, T3_HI);
    BF(q[0], q[2], T3_LO, T3_HI);  BF(q[1], q[3], T3_LO, T3_HI);
    BF(q[4], q[6], T3_LO, T3_HI);  BF(q[5], q[7], T3_LO, T3_HI);

    BF(r[0], q[0], T1_LO, T1_HI);  BF(r[1], q[1], T1_LO, T1_HI);
    BF(r[2], q[2], T1_LO, T1_HI);  BF(r[3], q[3], T1_LO, T1_HI);
    BF(r[4], q[4], T1_LO, T1_HI);  BF(r[5], q[5], T1_LO, T1_HI);
    BF(r[6], q[6], T1_LO, T1_HI);  BF(r[7], q[7], T1_LO, T1_HI);
}

/* destination of untrans_interleave's 16 outputs: r[0..7] then q[0..7] */
#define OUT_OFF(k0, half) ((size_t)(k0) * 128 + (size_t)(half) * 8)
static const short out_off[16] = {
    OUT_OFF(0,0), OUT_OFF(4,0), OUT_OFF(2,0), OUT_OFF(6,0),
    OUT_OFF(0,1), OUT_OFF(4,1), OUT_OFF(2,1), OUT_OFF(6,1),
    OUT_OFF(1,0), OUT_OFF(5,0), OUT_OFF(3,0), OUT_OFF(7,0),
    OUT_OFF(1,1), OUT_OFF(5,1), OUT_OFF(3,1), OUT_OFF(7,1)
};

/* deinterleave one z-pencil (8 contiguous complex) -> Re/Im, lane l holds z=PI[l] */
#define DEINT(p, re, im)                                                       \
    do {                                                                      \
        const v8d A_ = LD(p), B_ = LD((p) + 8);                               \
        (re) = SH(A_, B_, T1_LO);                                             \
        (im) = SH(A_, B_, T1_HI);                                             \
    } while (0)
/* and its exact inverse */
#define INTERL(p, re, im)                                                      \
    do {                                                                      \
        ST((p),     SH((re), (im), T1_LO));                                   \
        ST((p) + 8, SH((re), (im), T1_HI));                                   \
    } while (0)

#if defined(__AVX512F__)
#include <immintrin.h>
#define NTST(p, v) _mm512_stream_pd((double *)(void *)(p), (__m512d)(v))
#else
#define NTST(p, v) ST(p, v)
#endif

/* one 1 KiB chunk (16 cache lines) of the next volume, read-prefetched.  LOC is
 * a literal per instantiation: 3 = prefetcht0, 2 = prefetcht1.
 * L8_PF=0 compiles every prefetch out (the A/B switch for measuring). */
#ifndef L8_PF
#define L8_PF 1
#endif
#if L8_PF
#define PF(p, LOC) __builtin_prefetch((const void *)(p), 0, LOC)
#else
#define PF(p, LOC) ((void)0)
#endif
#define PF_CHUNK(p, LOC)                                                       \
    do {                                                                      \
        for (int t_ = 0; t_ < 16; ++t_) PF((p) + (size_t)t_ * 8, LOC);        \
    } while (0)

/* register index j of the transposed group holds z = PI[j]; feed dft8s in z order */
static const unsigned char piinv[8] = { 0, 2, 4, 6, 1, 3, 5, 7 };

struct fft3d_plan {
    int L, batch;
    int variant;     /* 6*shape + 3*nt + pf; shape: 0 fused, 1 seq3;
                        pf: 0 = off, 1 = t1, 2 = t0 (mode 0 only) */
    double *scr;     /* 16 KiB: sr[64][8], si[64][8] (indexed [y*8+x]), then
                        the seq3 scratch2 s2[64][16] (slot (k0*8+k1), re;im) */
    void *raw;
};

/* description carries the plan-time pick (the driver reads it after create) */
static char g_desc[192] =
    "8^3 fused x/y/z in L1, split-complex, spatial axis in the SIMD lanes";

const char *fft3d_name(void) { return "L8_fusedaxes"; }
const char *fft3d_description(void)
{
#if L8_MODE == 0
    return g_desc;
#elif L8_MODE == 1
    return "8^3 three passes through an L1 scratch (fusion control)";
#else
    return "8^3 three separate passes over the whole batch (row-column control)";
#endif
}
int fft3d_supports(int L) { return L == 8; }

/* ---- phase A: deinterleave + y axis, for one x-plane, into the scratch ---- */
#define PHASE_A(in, sr, si)                                                    \
    for (int x = 0; x < 8; ++x) {                                             \
        const double *ap_ = (in) + (size_t)x * 128;                           \
        v8d r[8], q[8];                                                        \
        DEINT(ap_ +   0, r[0], q[0]);  DEINT(ap_ +  16, r[1], q[1]);          \
        DEINT(ap_ +  32, r[2], q[2]);  DEINT(ap_ +  48, r[3], q[3]);          \
        DEINT(ap_ +  64, r[4], q[4]);  DEINT(ap_ +  80, r[5], q[5]);          \
        DEINT(ap_ +  96, r[6], q[6]);  DEINT(ap_ + 112, r[7], q[7]);          \
        dft8s(r, q);                                                           \
        for (int y = 0; y < 8; ++y) {                                          \
            ST((sr) + ((size_t)y * 8 + x) * 8, r[y]);                          \
            ST((si) + ((size_t)y * 8 + x) * 8, q[y]);                          \
        }                                                                      \
    }

/* ---- x axis, z axis and the store, for one y-slab held entirely in registers ---- */
#define PHASE_B_BODY(pr, pi, out, y, STOREOP)                                  \
    do {                                                                      \
        v8d r[8], q[8], zr[8], zq[8];                                          \
        for (int x = 0; x < 8; ++x) { r[x] = LD((pr) + x * 8); q[x] = LD((pi) + x * 8); } \
        dft8s(r, q);                     /* x axis, elementwise */             \
        trans8(r); trans8(q);            /* (reg=x,lane=z) -> (reg=z,lane=x) */\
        for (int j = 0; j < 8; ++j) { zr[j] = r[piinv[j]]; zq[j] = q[piinv[j]]; } \
        dft8s(zr, zq);                   /* z axis, elementwise */             \
        untrans_interleave(zr, zq);       /* back + interleave in one network */\
        double *op_ = (out) + (size_t)(y) * 16;                                \
        for (int j = 0; j < 8; ++j) {                                          \
            STOREOP(op_ + out_off[j],     zr[j]);                              \
            STOREOP(op_ + out_off[j + 8], zq[j]);                              \
        }                                                                      \
    } while (0)

/* ---- seq3 shape (round panel_r4): x axis through a 2nd scratch, then a
 * per-k0 final pass so the volume is written front to back ---- */

/* pass B1, per k1: x-axis DFT along the registers, zero shuffles.  Reads one
 * contiguous 1 KiB scratch1 row, writes the scratch2 column [k0][k1]
 * (slot = (k0*8+k1)*16 doubles, re then im, so pass B2 reads contiguously). */
#define PASS_B1(sr, si, s2)                                                    \
    for (int k1 = 0; k1 < 8; ++k1) {                                          \
        const double *pr_ = (sr) + (size_t)k1 * 64;                            \
        const double *pi_ = (si) + (size_t)k1 * 64;                            \
        v8d r[8], q[8];                                                        \
        for (int x = 0; x < 8; ++x) { r[x] = LD(pr_ + x * 8); q[x] = LD(pi_ + x * 8); } \
        dft8s(r, q);                     /* x axis -> register index = k0 */   \
        for (int k0 = 0; k0 < 8; ++k0) {                                       \
            ST((s2) + ((size_t)k0 * 8 + k1) * 16,     r[k0]);                  \
            ST((s2) + ((size_t)k0 * 8 + k1) * 16 + 8, q[k0]);                  \
        }                                                                      \
    }

/* pass B2, per k0: registers indexed k1, lane = z (PI order) -- the same state
 * current phase B is in after its x-DFT, so the trans8/dft8s/untrans chain is
 * reused verbatim with k1 playing k0's role.  The 16 output half-pencils all
 * land in the one 1 KiB k0-plane; they are stored in ASCENDING address order
 * (the fused shape's out_off table, sorted), so the volume's write stream is
 * fully sequential across the k0 loop. */
#define PASS_B2_BODY(s2, out, k0, STOREOP)                                     \
    do {                                                                      \
        const double *p2_ = (s2) + (size_t)(k0) * 128;                         \
        v8d r[8], q[8], zr[8], zq[8];                                          \
        for (int k1 = 0; k1 < 8; ++k1) {                                       \
            r[k1] = LD(p2_ + k1 * 16);                                         \
            q[k1] = LD(p2_ + k1 * 16 + 8);                                     \
        }                                                                      \
        trans8(r); trans8(q);            /* (reg=k1,lane=z) -> (reg=z,lane=k1) */\
        for (int j = 0; j < 8; ++j) { zr[j] = r[piinv[j]]; zq[j] = q[piinv[j]]; } \
        dft8s(zr, zq);                   /* z axis, elementwise */             \
        untrans_interleave(zr, zq);                                            \
        double *op_ = (out) + (size_t)(k0) * 128;                              \
        STOREOP(op_ +   0, zr[0]); STOREOP(op_ +   8, zr[4]);                  \
        STOREOP(op_ +  16, zq[0]); STOREOP(op_ +  24, zq[4]);                  \
        STOREOP(op_ +  32, zr[2]); STOREOP(op_ +  40, zr[6]);                  \
        STOREOP(op_ +  48, zq[2]); STOREOP(op_ +  56, zq[6]);                  \
        STOREOP(op_ +  64, zr[1]); STOREOP(op_ +  72, zr[5]);                  \
        STOREOP(op_ +  80, zq[1]); STOREOP(op_ +  88, zq[5]);                  \
        STOREOP(op_ +  96, zr[3]); STOREOP(op_ + 104, zr[7]);                  \
        STOREOP(op_ + 112, zq[3]); STOREOP(op_ + 120, zq[7]);                  \
    } while (0)

fft3d_plan *fft3d_create(int L, int batch);   /* defined per mode below */

void fft3d_destroy(fft3d_plan *plan)
{
    if (!plan) return;
    free(plan->raw);
    free(plan);
}

static fft3d_plan *plan_alloc(int L, int batch)
{
    if (L != 8 || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    void *raw = NULL;
    if (posix_memalign(&raw, 64, 4 * 512 * sizeof(double)) != 0 || !raw) { free(p); return NULL; }
    memset(raw, 0, 4 * 512 * sizeof(double));
    p->raw = raw;
    p->scr = (double *)raw;
    return p;
}

/* ============================ mode 0: fused ============================ */
#if L8_MODE == 0

/* DOPF is a compile-time 0/1 per instantiation, so the prefetch costs the
 * non-prefetching variants nothing; LOC is the prefetch level literal. */
#define VOLUME_FUSED_BODY(in, nxt, out, scr, STOREOP, DOPF, LOC)               \
    do {                                                                      \
        double *const sr_ = (scr), *const si_ = (scr) + 512;                   \
        PHASE_A(in, sr_, si_)                                                  \
        for (int y = 0; y < 8; ++y) {                                          \
            if (DOPF) PF_CHUNK((nxt) + (size_t)y * 128, LOC);                  \
            PHASE_B_BODY(sr_ + (size_t)y * 64, si_ + (size_t)y * 64, out, y, STOREOP); \
        }                                                                      \
    } while (0)

/* seq3: prefetch of the next volume sits in pass B2, whose loop bodies are the
 * shuffle/FP-dense ones with the load ports idle (same placement argument as
 * the fused shape's phase B). */
#define VOLUME_SEQ3_BODY(in, nxt, out, scr, STOREOP, DOPF, LOC)                \
    do {                                                                      \
        double *const sr_ = (scr), *const si_ = (scr) + 512;                   \
        double *const s2_ = (scr) + 1024;                                      \
        PHASE_A(in, sr_, si_)                                                  \
        PASS_B1(sr_, si_, s2_)                                                 \
        for (int k0 = 0; k0 < 8; ++k0) {                                       \
            if (DOPF) PF_CHUNK((nxt) + (size_t)k0 * 128, LOC);                 \
            PASS_B2_BODY(s2_, out, k0, STOREOP);                               \
        }                                                                      \
    } while (0)

#define DEF_VOL(NAME, BODY, STOREOP, DOPF, LOC)                                \
static void NAME(const double *restrict in, const double *restrict nxt,       \
                 double *restrict out, double *restrict scr)                   \
{                                                                             \
    (void)nxt;                                                                \
    BODY(in, nxt, out, scr, STOREOP, DOPF, LOC);                              \
}

DEF_VOL(vol_p,      VOLUME_FUSED_BODY, ST,   0, 2)   /*  0: plain               */
DEF_VOL(vol_p_t1,   VOLUME_FUSED_BODY, ST,   1, 2)   /*  1: plain + pf t1       */
DEF_VOL(vol_p_t0,   VOLUME_FUSED_BODY, ST,   1, 3)   /*  2: plain + pf t0       */
DEF_VOL(vol_nt,     VOLUME_FUSED_BODY, NTST, 0, 2)   /*  3: NT                  */
DEF_VOL(vol_nt_t1,  VOLUME_FUSED_BODY, NTST, 1, 2)   /*  4: NT + pf t1          */
DEF_VOL(vol_nt_t0,  VOLUME_FUSED_BODY, NTST, 1, 3)   /*  5: NT + pf t0          */
DEF_VOL(vol_s,      VOLUME_SEQ3_BODY,  ST,   0, 2)   /*  6: seq3 plain          */
DEF_VOL(vol_s_t1,   VOLUME_SEQ3_BODY,  ST,   1, 2)   /*  7: seq3 plain + pf t1  */
DEF_VOL(vol_s_t0,   VOLUME_SEQ3_BODY,  ST,   1, 3)   /*  8: seq3 plain + pf t0  */
DEF_VOL(vol_s_nt,   VOLUME_SEQ3_BODY,  NTST, 0, 2)   /*  9: seq3 NT             */
DEF_VOL(vol_s_nt_t1,VOLUME_SEQ3_BODY,  NTST, 1, 2)   /* 10: seq3 NT + pf t1     */
DEF_VOL(vol_s_nt_t0,VOLUME_SEQ3_BODY,  NTST, 1, 3)   /* 11: seq3 NT + pf t0     */

/* One batch pass with variant v.  Prefetch target: L8_PF_DIST volumes ahead,
 * clamped to the last volume so every prefetched address stays inside the
 * caller's mapping (an out-of-range prefetch cannot fault, but a not-present
 * page costs a wasted TLB walk). */
static void run_variant(int v, const double *restrict ip, double *restrict op,
                        double *restrict scr, long B)
{
    const double *const last = ip + (size_t)(B - 1) * 1024;
#define NXT (b + L8_PF_DIST < B ? ip + 1024 * L8_PF_DIST : last)
    switch (v) {
    case 0:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_p(ip, ip, op, scr);        break;
    case 1:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_p_t1(ip, NXT, op, scr);    break;
    case 2:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_p_t0(ip, NXT, op, scr);    break;
    case 3:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_nt(ip, ip, op, scr);       break;
    case 4:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_nt_t1(ip, NXT, op, scr);   break;
    case 5:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_nt_t0(ip, NXT, op, scr);   break;
    case 6:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_s(ip, ip, op, scr);        break;
    case 7:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_s_t1(ip, NXT, op, scr);    break;
    case 8:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_s_t0(ip, NXT, op, scr);    break;
    case 9:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_s_nt(ip, ip, op, scr);     break;
    case 10: for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_s_nt_t1(ip, NXT, op, scr); break;
    default: for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_s_nt_t0(ip, NXT, op, scr); break;
    }
#undef NXT
}

#if L8_TUNE
static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

#define NVAR 12
static const char *vname[NVAR] = {
    "plain", "plain+pf_t1", "plain+pf_t0", "nt", "nt+pf_t1", "nt+pf_t0",
    "seq3", "seq3+pf_t1", "seq3+pf_t0", "seq3-nt", "seq3-nt+pf_t1", "seq3-nt+pf_t0"
};

/* Time all twelve variants on a surrogate batch on THIS machine and keep the
 * fastest.  Protocol (drift lessons from L8_batchsimd / L6_pfa): round-robin
 * the candidates so slow machine states hit all of them, one untimed pass per
 * trial to set the candidate's own cache state (plain and NT leave different
 * L3 contents behind), min of 5 timed trials each, and a 2% hysteresis in
 * favour of the rule-based default. */
static void tune(fft3d_plan *p, double l3)
{
    long bsur = p->batch;
    long bcap = (long)(4.0 * l3 / 16384.0);   /* WS cap ~4x L3: deep enough to be
                                                 in the same residency regime as
                                                 any larger batch */
    if (bcap > 8192) bcap = 8192;
    if (bcap < 1024) bcap = 1024;
    if (bsur > bcap) bsur = bcap;
    const size_t nd = (size_t)bsur * 1024;
    void *ri = NULL, *ro = NULL;
    if (posix_memalign(&ri, 64, nd * sizeof(double)) != 0 || !ri) return;
    if (posix_memalign(&ro, 64, nd * sizeof(double)) != 0 || !ro) { free(ri); return; }
    double *ti = (double *)ri, *to = (double *)ro;
    uint64_t s = 0x243F6A8885A308D3ull;       /* deterministic, denormal-free fill */
    for (size_t i = 0; i < nd; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        ti[i] = (double)(int64_t)(s >> 12) * 0x1p-52;
    }
    memset(to, 0, nd * sizeof(double));
    double best[NVAR];
    for (int c = 0; c < NVAR; ++c) { best[c] = 1e300; run_variant(c, ti, to, p->scr, bsur); }
    for (int r = 0; r < 5; ++r)
        for (int c = 0; c < NVAR; ++c) {
            run_variant(c, ti, to, p->scr, bsur);       /* set own cache state */
            const double t0 = now_s();
            run_variant(c, ti, to, p->scr, bsur);
            const double t = now_s() - t0;
            if (t < best[c]) best[c] = t;
        }
    int mn = 0;
    for (int c = 1; c < NVAR; ++c)
        if (best[c] < best[mn]) mn = c;
    const int dflt = p->variant;
    if (best[mn] < 0.98 * best[dflt]) p->variant = mn;
    if (getenv("L8_TUNE_DEBUG")) {
        fprintf(stderr, "L8_fusedaxes tune B=%d bsur=%ld:", p->batch, bsur);
        for (int c = 0; c < NVAR; ++c)
            fprintf(stderr, " %s=%.3fus", vname[c], 1e6 * best[c] / (double)bsur);
        fprintf(stderr, " -> %s (default %s)\n", vname[p->variant], vname[dflt]);
    }
    free(ri);
    free(ro);
}
#endif /* L8_TUNE */

fft3d_plan *fft3d_create(int L, int batch)
{
    fft3d_plan *p = plan_alloc(L, batch);
    if (!p) return NULL;
    /* L3 from sysconf (per-socket; single-threaded on an exclusive node so the
     * whole L3 is ours); fall back to the scored node's 22 MiB if unknown. */
    double l3 = (double)sysconf(_SC_LEVEL3_CACHE_SIZE);
    if (l3 <= 0.0) l3 = 22.0 * 1024.0 * 1024.0;
    const double ws = (double)batch * 16384.0;         /* in + out */
    /* Rule-based default (= the tuner's hysteresis anchor):
     *   NT stores when in+out cannot be cache resident anyway;
     *   prefetch (t1) only when the INPUT ALONE exceeds L3, i.e. the reads must
     *   come from DRAM.  Round r2 shipped pf = nt instead, which prefetched an
     *   L3-resident input at node B=2048 -- the regime my own r2 measurement
     *   showed costing 1.6x.  Fixed this round. */
    int nt = ws > 1.18 * l3;
    int pf = ((double)batch * 8192.0 > 1.05 * l3) ? 1 : 0;
    int shape = 0;    /* rule default stays fused: the known-good anchor */
    int forced = 0;
#if L8_NT >= 0
    nt = L8_NT; forced = 1;
#endif
#if L8_PFSEL >= 0
    pf = L8_PFSEL ? ((L8_PF_LOC == 3) ? 2 : 1) : 0; forced = 1;
#endif
#if L8_SHAPE >= 0
    shape = L8_SHAPE; forced = 1;
#endif
    p->variant = 6 * shape + 3 * nt + pf;
    const char *how = forced ? "forced" : "rule";
#if L8_TUNE
    /* 0.25*L3, not ~1: wallaby r2 data shows NT already winning 1.9x at
     * ws = 0.53*L3, so the store-policy crossover sits well below capacity
     * and has to be inside the tuned region. */
    if (!forced && ws > 0.25 * l3) {
        tune(p, l3);
        how = "tuned";
    }
    snprintf(g_desc, sizeof g_desc,
             "8^3 fused x/y/z in L1, lanes=z; B=%d pick=%s (%s)",
             batch, vname[p->variant], how);
#else
    snprintf(g_desc, sizeof g_desc,
             "8^3 fused x/y/z in L1, lanes=z; B=%d pick=6*%d+3*%d+%d (%s)",
             batch, shape, nt, pf, how);
#endif
    return p;
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    int v = plan->variant;
    double *op = (double *)out;
    /* the non-temporal store faults on a misaligned address, so re-check rather
     * than trust the contract; the plain twin of each NT variant is v - 3
     * (variants are 6*shape + 3*nt + pf, so NT means v mod 6 >= 3) */
    if ((v % 6) >= 3 && ((uintptr_t)op & 63u) != 0u) v -= 3;
    run_variant(v, (const double *)in, op, plan->scr, plan->batch);
}

/* ================= mode 1: three passes, same scratch ================= */
#elif L8_MODE == 1

fft3d_plan *fft3d_create(int L, int batch) { return plan_alloc(L, batch); }

static void volume_3pass(const double *restrict in, double *restrict out,
                         double *restrict scr)
{
    double *const sr = scr, *const si = scr + 512;
    PHASE_A(in, sr, si)
    for (int y = 0; y < 8; ++y) {                 /* pass 2: x axis, back to L1 */
        double *pr = sr + (size_t)y * 64, *pi = si + (size_t)y * 64;
        v8d r[8], q[8];
        for (int x = 0; x < 8; ++x) { r[x] = LD(pr + x * 8); q[x] = LD(pi + x * 8); }
        dft8s(r, q);
        for (int x = 0; x < 8; ++x) { ST(pr + x * 8, r[x]); ST(pi + x * 8, q[x]); }
    }
    for (int y = 0; y < 8; ++y) {                 /* pass 3: z axis + store */
        double *pr = sr + (size_t)y * 64, *pi = si + (size_t)y * 64;
        v8d r[8], q[8], zr[8], zq[8];
        for (int x = 0; x < 8; ++x) { r[x] = LD(pr + x * 8); q[x] = LD(pi + x * 8); }
        trans8(r); trans8(q);
        for (int j = 0; j < 8; ++j) { zr[j] = r[piinv[j]]; zq[j] = q[piinv[j]]; }
        dft8s(zr, zq);
        untrans_interleave(zr, zq);
        double *op = out + (size_t)y * 16;
        for (int j = 0; j < 8; ++j) {
            ST(op + out_off[j],     zr[j]);
            ST(op + out_off[j + 8], zq[j]);
        }
    }
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    const double *ip = (const double *)in;
    double *op = (double *)out;
    for (long b = 0, B = plan->batch; b < B; ++b, ip += 1024, op += 1024)
        volume_3pass(ip, op, plan->scr);
}

/* ============ mode 2: three passes over the whole batch ============ */
#else

fft3d_plan *fft3d_create(int L, int batch) { return plan_alloc(L, batch); }

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    const long B = plan->batch;

    {   /* pass 1: y axis, in -> out */
        const double *ip = (const double *)in;
        double *op = (double *)out;
        for (long b = 0; b < B; ++b, ip += 1024, op += 1024)
            for (int x = 0; x < 8; ++x) {
                v8d r[8], q[8];
                for (int y = 0; y < 8; ++y) DEINT(ip + x * 128 + y * 16, r[y], q[y]);
                dft8s(r, q);
                for (int y = 0; y < 8; ++y) INTERL(op + x * 128 + y * 16, r[y], q[y]);
            }
    }
    {   /* pass 2: x axis, in place on out */
        double *op = (double *)out;
        for (long b = 0; b < B; ++b, op += 1024)
            for (int y = 0; y < 8; ++y) {
                v8d r[8], q[8];
                for (int x = 0; x < 8; ++x) DEINT(op + x * 128 + y * 16, r[x], q[x]);
                dft8s(r, q);
                for (int x = 0; x < 8; ++x) INTERL(op + x * 128 + y * 16, r[x], q[x]);
            }
    }
    {   /* pass 3: z axis, in place on out */
        double *op = (double *)out;
        for (long b = 0; b < B; ++b, op += 1024)
            for (int y = 0; y < 8; ++y) {
                v8d r[8], q[8], zr[8], zq[8];
                for (int x = 0; x < 8; ++x) DEINT(op + x * 128 + y * 16, r[x], q[x]);
                trans8(r); trans8(q);
                for (int j = 0; j < 8; ++j) { zr[j] = r[piinv[j]]; zq[j] = q[piinv[j]]; }
                dft8s(zr, zq);
                untrans_interleave(zr, zq);
                double *tp = op + (size_t)y * 16;
                for (int j = 0; j < 8; ++j) {
                    ST(tp + out_off[j],     zr[j]);
                    ST(tp + out_off[j + 8], zq[j]);
                }
            }
    }
}
#endif
