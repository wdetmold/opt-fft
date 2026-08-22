/* L8_fusedaxes -- forward complex-double 3D DFT of an 8x8x8 cube with all three
 * axes fused: one trip in from memory, one trip out, nothing but an L1
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
 *   n1_8); 896 shuffles is provably minimal for this layout.  Four rounds of node
 *   data say B=1 is NOT moved by instruction cuts; the arithmetic is frozen.
 *
 * SEQ3 SHAPE (round panel_r4, adopted from L8_radix8): a third pass through a 2nd
 *   L1 scratch so the output volume is written fully SEQUENTIALLY: phase A
 *   unchanged -> pass B1 (x-axis DFT, scratch1 row -> scratch2 [k0][k1], zero
 *   shuffles) -> pass B2 per k0 (transpose + z-axis DFT + fused untranspose/
 *   interleave, 16 half-pencils of the 1 KiB k0-plane stored ascending).
 *   Identical FP/shuffle counts; +128 loads +128 stores, all L1-resident.
 *
 * PREFETCH PLACEMENT (new in round panel_r5, adopted from L8_radix8's r4 node
 *   wins at B=2048 = 1.136 us and B=16384 = 1.418 us, both with PLAIN stores):
 *   the next volume's 128 input lines are prefetched t0, SPREAD ~5-8 lines at the
 *   top of every loop iteration (~1 line per 10 cycles of L1-resident compute)
 *   instead of my old 16-line chunks.  radix8 measured spread-vs-burst worth
 *   -8.6% at node B=2048 on top of the sequential store order it had already
 *   taken from its r3 win; NT lost to plain+spread on the node in every batched
 *   L=8 cell in r4.  Also new: write-intent prefetch (prefetchw) of the NEXT
 *   volume's output lines at the same cadence, adopted from L6_unrolled /
 *   L6_pfa (their fused_pfw variants won the L=6 DRAM cells): with plain stores
 *   every output line pays an RFO; prefetchw issues it one volume early.
 *
 * STORE POLICY / SHAPE / PREFETCH ARE MEASURED AT PLAN TIME, NOT RULED (r3):
 *   fft3d_create() times a regime-gated candidate set on a surrogate batch ON THE
 *   MACHINE THAT SCORES, round-robin interleaved, one untimed own-cache-state pass
 *   per trial, min-of-5, hysteresis toward the regime anchor.  New in r5: the
 *   tuner also runs in the ws ~ L2 regime (the panel_r4 VERDICT's "B=64 L2 cliff"
 *   ask) with a small plain-only candidate set and 3% hysteresis, so a scored
 *   cell that sits exactly on the node's 1 MiB L2 gets measured choices instead
 *   of an assumed one.  B=1 stays untuned on the rule path (fused plain).
 *
 * CLOCK PROBE (new in round panel_r5, extends L6_unrolled's r4 probe): create()
 *   measures the sustained core clock under BOTH a 256-bit and a 512-bit serial
 *   FMA chain (latency 4 on CLX/SPR) and reports clk256/clk512 in
 *   fft3d_description().  L6's 256-bit probe measured the node at 3.89 GHz, not
 *   2.30; whether the AVX-512 licence clock matches decides if B=1's 0.57 us is
 *   at its 1296-cycle port floor (clk512 ~ 2.3) or 40% above it (clk512 ~ 3.5+).
 *   Nobody has measured clk512 on the node; this puts it in the leaderboard JSON.
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
 *     is the non-temporal store and the 512-bit half of the clock probe.
 *
 * COMPILE-TIME SWITCHES (default is what ships)
 *   L8_MODE     0 fused+tuner (ships) | 1 three passes via L1 scratch | 2 three
 *               passes over the whole batch (controls, unchanged since round 1)
 *   L8_VARIANT  -1 auto (rule+tuner), 0..9 force one variant (skips the tuner):
 *               0 fused-plain            1 fused-plain+spread-t0
 *               2 fused-plain+spread-t0+pfw   3 fused-nt+spread-t0
 *               4 fused-nt+chunk-t1 (r4 B=16384 shipping config)
 *               5 seq3-plain             6 seq3-plain+spread-t0
 *               7 seq3-plain+spread-t0+pfw    8 seq3-nt+spread-t0
 *               9 seq3-nt+chunk-t0 (r4 wallaby B=5632 pick)
 *               (supersedes r2-r4's L8_NT/L8_PFSEL/L8_SHAPE triplet)
 *   L8_PF_DIST  prefetch distance in volumes (1; r2 measured 2 as worse)
 *   L8_PF       0 compiles every prefetch out (spread variants alias plain)
 *   L8_TUNE     0 disables the plan-time tuner (rule-based auto only)
 *   L8_PROBE    0 disables the clock probe
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
#ifndef L8_PF_DIST
#define L8_PF_DIST 1
#endif
#ifndef L8_VARIANT
#define L8_VARIANT (-1)
#endif
#ifndef L8_TUNE
#define L8_TUNE 1
#endif
#ifndef L8_PROBE
#define L8_PROBE 1
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
#if defined(__AVX2__) || defined(__FMA__)
#include <immintrin.h>
#endif
#define NTST(p, v) ST(p, v)
#endif

/* ---- prefetch primitives.  L8_PF=0 compiles every prefetch out (the A/B
 * switch for measuring); with constant n these fully unroll. */
#ifndef L8_PF
#define L8_PF 1
#endif
#if L8_PF
static inline void pf_lines_t0(const double *p, int n)
{   for (int i = 0; i < n; ++i) __builtin_prefetch(p + (size_t)i * 8, 0, 3); }
static inline void pf_lines_t1(const double *p, int n)
{   for (int i = 0; i < n; ++i) __builtin_prefetch(p + (size_t)i * 8, 0, 2); }
/* write-intent: emits prefetchw where PRFCHW exists (CLX, SPR); issues the
 * RFO of a to-be-fully-written output line one volume early. */
static inline void pf_lines_w(double *p, int n)
{   for (int i = 0; i < n; ++i) __builtin_prefetch(p + (size_t)i * 8, 1, 3); }
#else
static inline void pf_lines_t0(const double *p, int n) { (void)p; (void)n; }
static inline void pf_lines_t1(const double *p, int n) { (void)p; (void)n; }
static inline void pf_lines_w(double *p, int n)        { (void)p; (void)n; }
#endif

/* register index j of the transposed group holds z = PI[j]; feed dft8s in z order */
static const unsigned char piinv[8] = { 0, 2, 4, 6, 1, 3, 5, 7 };

struct fft3d_plan {
    int L, batch;
    int variant;     /* index into the variant table, see L8_VARIANT above */
    double *scr;     /* 16 KiB: sr[64][8], si[64][8] (indexed [y*8+x]), then
                        the seq3 scratch2 s2[64][16] (slot (k0*8+k1), re;im) */
    void *raw;
};

/* description carries the plan-time pick and the clock probe (the driver reads
 * it after create, so both land in the leaderboard JSON) */
static char g_desc[256] =
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

/* ---- phase A: deinterleave + y axis, for ONE x-plane, into the scratch ---- */
#define PHASE_A_ONE(in, sr, si, x)                                             \
    do {                                                                      \
        const double *ap_ = (in) + (size_t)(x) * 128;                          \
        v8d r[8], q[8];                                                        \
        DEINT(ap_ +   0, r[0], q[0]);  DEINT(ap_ +  16, r[1], q[1]);          \
        DEINT(ap_ +  32, r[2], q[2]);  DEINT(ap_ +  48, r[3], q[3]);          \
        DEINT(ap_ +  64, r[4], q[4]);  DEINT(ap_ +  80, r[5], q[5]);          \
        DEINT(ap_ +  96, r[6], q[6]);  DEINT(ap_ + 112, r[7], q[7]);          \
        dft8s(r, q);                                                           \
        for (int y_ = 0; y_ < 8; ++y_) {                                       \
            ST((sr) + ((size_t)y_ * 8 + (x)) * 8, r[y_]);                      \
            ST((si) + ((size_t)y_ * 8 + (x)) * 8, q[y_]);                      \
        }                                                                      \
    } while (0)

#define PHASE_A(in, sr, si)                                                    \
    for (int x = 0; x < 8; ++x) PHASE_A_ONE(in, sr, si, x);

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
#define PASS_B1_ONE(sr, si, s2, k1)                                            \
    do {                                                                      \
        const double *pr_ = (sr) + (size_t)(k1) * 64;                          \
        const double *pi_ = (si) + (size_t)(k1) * 64;                          \
        v8d r[8], q[8];                                                        \
        for (int x = 0; x < 8; ++x) { r[x] = LD(pr_ + x * 8); q[x] = LD(pi_ + x * 8); } \
        dft8s(r, q);                     /* x axis -> register index = k0 */   \
        for (int k0_ = 0; k0_ < 8; ++k0_) {                                    \
            ST((s2) + ((size_t)k0_ * 8 + (k1)) * 16,     r[k0_]);              \
            ST((s2) + ((size_t)k0_ * 8 + (k1)) * 16 + 8, q[k0_]);              \
        }                                                                      \
    } while (0)

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

/* ---- per-iteration prefetch hooks.  nx = next volume's input, no = next
 * volume's output, i = iteration index 0..7.  Hook macro names are passed as
 * macro arguments into the VOLUME bodies, radix8-style, so each variant gets
 * its placement compiled in with zero runtime flags.
 *
 * Spread cadence (adopted from L8_radix8 r4): the 128 input lines of volume
 * b+1 are issued 8/8 per iteration over the fused shape's 16 iterations, and
 * 6/5/5 over the seq3 shape's 24 iterations -- ~1 prefetch per 10 cycles of
 * L1-resident compute, never a burst.  Chunk hooks (16 lines at the top of a
 * phase-B/B2 iteration) are the r2-r4 placement, kept as continuity
 * candidates under NT where they were the shipping node config. */
#define H_NONE(nx, no, i)      ((void)0)
/* fused: 8 lines/iter; phase A covers doubles [0,512), phase B [512,1024) */
#define HFA_S(nx, no, i)   pf_lines_t0((nx) + (size_t)(i) * 64, 8)
#define HFB_S(nx, no, i)   pf_lines_t0((nx) + 512 + (size_t)(i) * 64, 8)
#define HFA_SW(nx, no, i)  do { pf_lines_t0((nx) + (size_t)(i) * 64, 8);       \
                                pf_lines_w((no) + (size_t)(i) * 64, 8); } while (0)
#define HFB_SW(nx, no, i)  do { pf_lines_t0((nx) + 512 + (size_t)(i) * 64, 8); \
                                pf_lines_w((no) + 512 + (size_t)(i) * 64, 8); } while (0)
/* fused chunk (the r2-r4 placement): 16 lines t1 at the top of each phase-B y */
#define HFB_C1(nx, no, i)  pf_lines_t1((nx) + (size_t)(i) * 128, 16)
/* seq3: 6/5/5 lines per iteration of A/B1/B2; 8*48=384, 384+8*40=704, +320=1024 */
#define HSA_S(nx, no, i)   pf_lines_t0((nx) + (size_t)(i) * 48, 6)
#define HS1_S(nx, no, i)   pf_lines_t0((nx) + 384 + (size_t)(i) * 40, 5)
#define HS2_S(nx, no, i)   pf_lines_t0((nx) + 704 + (size_t)(i) * 40, 5)
#define HSA_SW(nx, no, i)  do { pf_lines_t0((nx) + (size_t)(i) * 48, 6);       \
                                pf_lines_w((no) + (size_t)(i) * 48, 6); } while (0)
#define HS1_SW(nx, no, i)  do { pf_lines_t0((nx) + 384 + (size_t)(i) * 40, 5); \
                                pf_lines_w((no) + 384 + (size_t)(i) * 40, 5); } while (0)
#define HS2_SW(nx, no, i)  do { pf_lines_t0((nx) + 704 + (size_t)(i) * 40, 5); \
                                pf_lines_w((no) + 704 + (size_t)(i) * 40, 5); } while (0)
/* seq3 chunk (the r4 placement): 16 lines t0 at the top of each B2 k0 */
#define HS2_C0(nx, no, i)  pf_lines_t0((nx) + (size_t)(i) * 128, 16)

#define VOLUME_FUSED_BODY(in, nxt, nxo, out, scr, STOREOP, HA, HB)             \
    do {                                                                      \
        double *const sr_ = (scr), *const si_ = (scr) + 512;                   \
        for (int x = 0; x < 8; ++x) {                                          \
            HA(nxt, nxo, x);                                                   \
            PHASE_A_ONE(in, sr_, si_, x);                                      \
        }                                                                      \
        for (int y = 0; y < 8; ++y) {                                          \
            HB(nxt, nxo, y);                                                   \
            PHASE_B_BODY(sr_ + (size_t)y * 64, si_ + (size_t)y * 64, out, y, STOREOP); \
        }                                                                      \
    } while (0)

#define VOLUME_SEQ3_BODY(in, nxt, nxo, out, scr, STOREOP, HA, H1, H2)          \
    do {                                                                      \
        double *const sr_ = (scr), *const si_ = (scr) + 512;                   \
        double *const s2_ = (scr) + 1024;                                      \
        for (int x = 0; x < 8; ++x) {                                          \
            HA(nxt, nxo, x);                                                   \
            PHASE_A_ONE(in, sr_, si_, x);                                      \
        }                                                                      \
        for (int k1 = 0; k1 < 8; ++k1) {                                       \
            H1(nxt, nxo, k1);                                                  \
            PASS_B1_ONE(sr_, si_, s2_, k1);                                    \
        }                                                                      \
        for (int k0 = 0; k0 < 8; ++k0) {                                       \
            H2(nxt, nxo, k0);                                                  \
            PASS_B2_BODY(s2_, out, k0, STOREOP);                               \
        }                                                                      \
    } while (0)

#define DEF_VOL_F(NAME, STOREOP, HA, HB)                                       \
static void NAME(const double *restrict in, const double *restrict nxt,       \
                 double *restrict nxo, double *restrict out,                   \
                 double *restrict scr)                                         \
{                                                                             \
    (void)nxt; (void)nxo;                                                      \
    VOLUME_FUSED_BODY(in, nxt, nxo, out, scr, STOREOP, HA, HB);                \
}
#define DEF_VOL_S(NAME, STOREOP, HA, H1, H2)                                   \
static void NAME(const double *restrict in, const double *restrict nxt,       \
                 double *restrict nxo, double *restrict out,                   \
                 double *restrict scr)                                         \
{                                                                             \
    (void)nxt; (void)nxo;                                                      \
    VOLUME_SEQ3_BODY(in, nxt, nxo, out, scr, STOREOP, HA, H1, H2);             \
}

DEF_VOL_F(vol_p,      ST,   H_NONE, H_NONE)          /* 0 fused plain          */
DEF_VOL_F(vol_p_s,    ST,   HFA_S,  HFB_S)           /* 1 fused plain sprd t0  */
DEF_VOL_F(vol_p_sw,   ST,   HFA_SW, HFB_SW)          /* 2 fused plain sprd+pfw */
DEF_VOL_F(vol_nt_s,   NTST, HFA_S,  HFB_S)           /* 3 fused nt sprd t0     */
DEF_VOL_F(vol_nt_c1,  NTST, H_NONE, HFB_C1)          /* 4 fused nt chunk t1    */
DEF_VOL_S(vol_s,      ST,   H_NONE, H_NONE, H_NONE)  /* 5 seq3 plain           */
DEF_VOL_S(vol_s_s,    ST,   HSA_S,  HS1_S,  HS2_S)   /* 6 seq3 plain sprd t0   */
DEF_VOL_S(vol_s_sw,   ST,   HSA_SW, HS1_SW, HS2_SW)  /* 7 seq3 plain sprd+pfw  */
DEF_VOL_S(vol_s_nt_s, NTST, HSA_S,  HS1_S,  HS2_S)   /* 8 seq3 nt sprd t0      */
DEF_VOL_S(vol_s_nt_c0,NTST, H_NONE, H_NONE, HS2_C0)  /* 9 seq3 nt chunk t0     */

#define NVAR 10
static const char *vname[NVAR] = {
    "fused",      "fused+pfs",  "fused+pfs+pfw", "fused-nt+pfs", "fused-nt+pf_t1",
    "seq3",       "seq3+pfs",   "seq3+pfs+pfw",  "seq3-nt+pfs",  "seq3-nt+pf_t0"
};
/* the plain twin of each NT variant, for the alignment fallback in execute */
static const unsigned char plain_twin[NVAR] = { 0,1,2,1,0, 5,6,7,6,5 };
/* variants that touch nxo (write-intent prefetch) */

/* One batch pass with variant v.  Prefetch target: L8_PF_DIST volumes ahead,
 * clamped to the last volume so every prefetched address stays inside the
 * caller's mapping (an out-of-range prefetch cannot fault, but a not-present
 * page costs a wasted TLB walk). */
static void run_variant(int v, const double *restrict ip, double *restrict op,
                        double *restrict scr, long B)
{
    const double *const lasti = ip + (size_t)(B - 1) * 1024;
    double *const lasto = op + (size_t)(B - 1) * 1024;
#define NXT  (b + L8_PF_DIST < B ? ip + 1024 * L8_PF_DIST : lasti)
#define NXTO (b + L8_PF_DIST < B ? op + 1024 * L8_PF_DIST : lasto)
    switch (v) {
    case 0:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_p(ip, NULL, NULL, op, scr);        break;
    case 1:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_p_s(ip, NXT, NXTO, op, scr);       break;
    case 2:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_p_sw(ip, NXT, NXTO, op, scr);      break;
    case 3:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_nt_s(ip, NXT, NXTO, op, scr);      break;
    case 4:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_nt_c1(ip, NXT, NXTO, op, scr);     break;
    case 5:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_s(ip, NULL, NULL, op, scr);        break;
    case 6:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_s_s(ip, NXT, NXTO, op, scr);       break;
    case 7:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_s_sw(ip, NXT, NXTO, op, scr);      break;
    case 8:  for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_s_nt_s(ip, NXT, NXTO, op, scr);    break;
    default: for (long b = 0; b < B; ++b, ip += 1024, op += 1024) vol_s_nt_c0(ip, NXT, NXTO, op, scr);   break;
    }
#undef NXT
#undef NXTO
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

#if L8_TUNE
/* Time a candidate set on a surrogate batch on THIS machine and keep the
 * fastest.  Protocol (drift lessons from L8_batchsimd / L6_pfa, r2-r3):
 * round-robin the candidates so slow machine states hit all of them, one
 * untimed pass per trial to set the candidate's own cache state (plain and NT
 * leave different L3 contents behind), min of 5 timed trials each, and a
 * hysteresis toward the regime anchor.  New in r5: an inner repeat count so a
 * timed trial covers >= ~2 ms even at a small surrogate batch (B=64 alone is
 * ~35 us, far too short to trust), which is what makes tuning at the L2 cliff
 * meaningful at all. */
static void tune(fft3d_plan *p, const unsigned char *cand, int nc,
                 long bsur, double hyst)
{
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
    long reps = 4096 / bsur;                  /* >= ~2 ms per timed trial */
    if (reps < 1) reps = 1;
    double best[NVAR];
    for (int c = 0; c < nc; ++c) {
        best[cand[c]] = 1e300;
        run_variant(cand[c], ti, to, p->scr, bsur);
    }
    for (int r = 0; r < 5; ++r)
        for (int c = 0; c < nc; ++c) {
            const int v = cand[c];
            run_variant(v, ti, to, p->scr, bsur);       /* set own cache state */
            const double t0 = now_s();
            for (long k = 0; k < reps; ++k)
                run_variant(v, ti, to, p->scr, bsur);
            const double t = (now_s() - t0) / (double)reps;
            if (t < best[v]) best[v] = t;
        }
    int mn = cand[0];
    for (int c = 1; c < nc; ++c)
        if (best[cand[c]] < best[mn]) mn = cand[c];
    const int dflt = p->variant;              /* the regime anchor */
    if (best[mn] < (1.0 - hyst) * best[dflt]) p->variant = mn;
    if (getenv("L8_TUNE_DEBUG")) {
        fprintf(stderr, "L8_fusedaxes tune B=%d bsur=%ld reps=%ld:", p->batch, bsur, reps);
        for (int c = 0; c < nc; ++c)
            fprintf(stderr, " %s=%.3fus", vname[cand[c]],
                    1e6 * best[cand[c]] / (double)bsur);
        fprintf(stderr, " -> %s (anchor %s)\n", vname[p->variant], vname[dflt]);
    }
    free(ri);
    free(ro);
}
#endif /* L8_TUNE */

#if L8_PROBE && (defined(__AVX2__) && defined(__FMA__))
/* Sustained core clock under a serially dependent FMA chain, 256-bit and (on
 * AVX-512 builds) 512-bit.  Method from L6_unrolled r4 (its 256-bit probe
 * measured the node at 3.89 GHz and reopened L=6); the 512-bit half is new --
 * no AVX-512 licence clock has ever been measured on the node, and it decides
 * whether B=1's 0.57 us is at its 1296-cycle port floor.  FMA latency is 4 on
 * CLX/SPR (5 on the Haswell dev machine, where the numbers over-read by 25%
 * -- dev caveat only).  ~10 ms per width, in create(), unscored. */
/* Both probes: spin the FMA chain in 5 ms chunks, track the max chunk
 * frequency, and stop once the max has not improved by >0.5% for 100 ms
 * (or at the hard cap).  A powersave-governed core sits STABLY at base for
 * ~100+ ms before the governor jumps it, and keeps stepping for several
 * hundred ms after (fixed 150 ms warms read wallaby's 4.1 GHz core at 3.4),
 * so neither a fixed warm nor a consecutive-stability test is reliable --
 * max-stagnation is.  Worst case ~500 ms in create(), unscored. */
static double probe_ghz_256(void)
{
    __m256d x = _mm256_set1_pd(1.0);
    const __m256d a = _mm256_set1_pd(1.0 + 1e-15);
    const __m256d b = _mm256_set1_pd(1e-300);
    double best = 0.0, best_at = now_s();
    const double cap = now_s() + 500e-3;
    for (;;) {
        double t0 = now_s(); long it = 0;
        do { for (int i = 0; i < 8192; ++i) x = _mm256_fmadd_pd(x, a, b);
             it += 8192; } while (now_s() - t0 < 5e-3);
        double t1 = now_s(), g = (double)it * 4.0 / (t1 - t0) * 1e-9;
        if (g > 1.005 * best) { best = g; best_at = t1; }
        if (t1 - best_at > 100e-3 || t1 > cap) break;
    }
    double lane[4];
    _mm256_storeu_pd(lane, x);
    if (!(lane[0] > 0.0) || best > 9.9) best = 0.0;
    return best;
}
#if defined(__AVX512F__)
static double probe_ghz_512(void)
{
    /* runs right after the 256-bit probe, so the core is already at turbo;
     * the licence-based transition (CLX drops the clock a few hundred us
     * after the first 512-bit uops) settles inside the first chunks and the
     * max-stagnation rule then reports the SUSTAINED 512-bit clock */
    __m512d x = _mm512_set1_pd(1.0);
    const __m512d a = _mm512_set1_pd(1.0 + 1e-15);
    const __m512d b = _mm512_set1_pd(1e-300);
    double best = 0.0, best_at = now_s();
    const double cap = now_s() + 300e-3;
    for (;;) {
        double t0 = now_s(); long it = 0;
        do { for (int i = 0; i < 8192; ++i) x = _mm512_fmadd_pd(x, a, b);
             it += 8192; } while (now_s() - t0 < 5e-3);
        double t1 = now_s(), g = (double)it * 4.0 / (t1 - t0) * 1e-9;
        if (g > 1.005 * best) { best = g; best_at = t1; }
        if (t1 - best_at > 100e-3 || t1 > cap) break;
    }
    double lane[8];
    _mm512_storeu_pd(lane, x);
    if (!(lane[0] > 0.0) || best > 9.9) best = 0.0;
    return best;
}
#else
static double probe_ghz_512(void) { return 0.0; }
#endif
#define HAVE_PROBE 1
#endif /* L8_PROBE */

fft3d_plan *fft3d_create(int L, int batch)
{
    fft3d_plan *p = plan_alloc(L, batch);
    if (!p) return NULL;
    /* cache sizes from sysconf (single-threaded on an exclusive node so the
     * whole L3 is ours); fall back to the scored node's 22 MiB / 1 MiB. */
    double l3 = (double)sysconf(_SC_LEVEL3_CACHE_SIZE);
    if (l3 <= 0.0) l3 = 22.0 * 1024.0 * 1024.0;
    double l2 = (double)sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (l2 <= 0.0) l2 = 1024.0 * 1024.0;
    const double ws = (double)batch * 16384.0;         /* in + out */
    /* Regime anchors (= the tuner's hysteresis defaults, and the rule picks
     * when the tuner is off).  Node panel_r4 evidence:
     *   streaming (ws > 1.18*L3): plain stores + spread t0 prefetch +
     *     sequential store order won BOTH node streaming cells via L8_radix8
     *     (1.136 / 1.418); NT lost everywhere on the node -> anchor seq3+pfs.
     *   mid band (0.25..1.18*L3): no scored node cell lives here; wallaby
     *     r3/r4 picked fused-nt reliably -> anchor fused-nt+pfs.
     *   L2 band (0.5*L2..0.25*L3): the B=64 cliff; anchor stays fused-plain
     *     (the four-round incumbent) and only a >3% measured win displaces it.
     *   below: fused plain, untuned, byte-identical arithmetic to rounds 1-4. */
    int anchor = 0;
    if (ws > 1.18 * l3)      anchor = 6;
    else if (ws > 0.25 * l3) anchor = 3;
    p->variant = anchor;
    const char *how = "rule";
#if L8_VARIANT >= 0
    p->variant = L8_VARIANT < NVAR ? L8_VARIANT : 0;
    how = "forced";
#elif L8_TUNE
    static const unsigned char cand_small[] = { 0, 1, 5, 6 };
    static const unsigned char cand_big[]   = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    if (ws > 0.25 * l3) {
        long bcap = (long)(4.0 * l3 / 16384.0);   /* WS cap ~4x L3: deep enough
                                                     to be in the same residency
                                                     regime as any larger batch */
        if (bcap > 8192) bcap = 8192;
        if (bcap < 1024) bcap = 1024;
        long bsur = batch > bcap ? bcap : batch;
        tune(p, cand_big, (int)sizeof cand_big, bsur, 0.02);
        how = "tuned";
    } else if (ws >= 0.5 * l2) {
        /* the L2 cliff: surrogate = the exact batch, so the tuner sees the
         * driver's steady state (same in/out reused every call).  Plain-store
         * shapes +- spread prefetch only: NT and pfw have nothing to offer
         * when the output is L2/L3-resident, and a small set keeps the pick
         * stable (radix8 r3's pf coin-flip cost 6.7% at this exact cell). */
        tune(p, cand_small, (int)sizeof cand_small, batch, 0.03);
        how = "tuned";
    }
#endif
    double c256 = 0.0, c512 = 0.0;
#ifdef HAVE_PROBE
    c256 = probe_ghz_256();
    c512 = probe_ghz_512();
#endif
    snprintf(g_desc, sizeof g_desc,
             "8^3 lanes=z fused/seq3; B=%d pick=%s (%s) clk256=%.2f clk512=%.2f",
             batch, vname[p->variant], how, c256, c512);
    return p;
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    int v = plan->variant;
    double *op = (double *)out;
    /* the non-temporal store faults on a misaligned address, so re-check rather
     * than trust the contract; each NT variant has a plain twin */
    if (((uintptr_t)op & 63u) != 0u) v = plain_twin[v];
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
