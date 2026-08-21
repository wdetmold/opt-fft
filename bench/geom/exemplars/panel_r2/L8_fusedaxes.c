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
 *   competing for port 5, which is the scarce port here.  An earlier version that
 *   used vpermt2pd (destructive) for the middle transpose stage and for the
 *   interleave cost 147 port-5 slots per phase-B iteration against 112 shuffles;
 *   this one costs 96.
 *
 * OPERATION COUNT (per 8^3 volume)
 *   8-point complex DFT, radix-8 DIF, every trivial twiddle written out:
 *     16 adds (stage 1) + 8 (the two nontrivial twiddles W8^1, W8^3: 2 mul + 2 add
 *     each) + 16 + 16 (two DFT4s; the W4 = -i multiplies are free in split layout)
 *     = 52 real adds + 4 real multiplies = 56 real flops -- the published optimum
 *     (Burrus T7.1 = T9.1 = FFTW n1_8).
 *   3 axes * 64 lines = 192 line DFTs = 10 752 real flops per volume, issued as
 *     1344 vector FP instructions  (3 * 8 * 56, eight lanes each, no waste; 1296
 *                                   as emitted -- gcc contracts two of the four
 *                                   twiddle multiplies per DFT into FMAs)
 *      896 vector shuffles          (128 deinterleave + 384 transpose + 384 fused
 *                                    inverse-transpose/interleave)
 *      256 vector loads, 256 vector stores, 0 spills, 0 register copies.
 *   Nominal 5*N*log2(N) yardstick, for comparison only: 23 040 flops.
 *   Lower bound on the scored node: FP and shuffles issue on different ports, so
 *   ~1296 cycles/volume (0.56 us at 2.3 GHz) with one 512-bit FMA unit, ~1120 with
 *   two.  Both counts are at their floor: 896 shuffles is provably minimal for this
 *   layout (a 2-input lane shuffle resolves one bit of source-register index, so
 *   moving k register bits into the lanes costs k stages of 16 ops) and 56 real
 *   flops is the published optimum for an 8-point complex DFT.
 *
 * ASSUMPTIONS
 *   * L == 8 only.
 *   * in/out 64-byte aligned and distinct (the driver guarantees both); every
 *     z-pencil therefore starts 128-byte aligned and all vector accesses are
 *     naturally aligned.
 *   * GCC/Clang vector extensions.  ONE arithmetic path for every ISA: a 64-byte
 *     `v8d` becomes zmm on the scored Cascade Lake node and a 2 x ymm pair on the
 *     AVX2 development node, with identical arithmetic in identical order, so what
 *     is verified locally is what runs there.  (GCC lowers 8-lane 64-byte shuffles
 *     poorly without AVX-512, so the *local* timings are heavily pessimistic --
 *     see strategies/L8_fusedaxes.md.)  The only __AVX512F__-guarded code is the
 *     optional non-temporal final store.
 *
 * L8_MODE (compile-time; default 0 is what ships)
 *   0 : fused, y | (x,z), one L1 scratch round trip
 *   1 : three passes over one volume through the same L1 scratch -- identical
 *       arithmetic and identical shuffle count, so it isolates the cost of one
 *       extra 8 KiB L1 round trip
 *   2 : three separate passes over the whole batch in the driver's interleaved
 *       layout -- the naive row-column shape: 3x the memory traffic and 3x the
 *       de/interleave work
 */
#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../fft3d_api.h"

#ifndef L8_MODE
#define L8_MODE 0
#endif

/* Software prefetch of the NEXT volume's input, issued from phase B, whose 8
 * loop bodies are pure L1-resident compute with the load ports idle.  The whole
 * 8 KiB (128 lines) is covered: 16 prefetches per y-iteration.  Tunables so the
 * variant sweep is a -D flag, not an edit:
 *   L8_PF_LOC   __builtin_prefetch locality: 3 = prefetcht0, 2 = t1, 1 = t2, 0 = nta
 *   L8_PF_DIST  distance in volumes (1 = next volume)                          */
#ifndef L8_PF_LOC
#define L8_PF_LOC 2
#endif
#ifndef L8_PF_DIST
#define L8_PF_DIST 1
#endif
/* L8_NT: -1 = auto (in+out past L3), 0 = force regular stores, 1 = force NT.
 * L8_PFSEL: -1 = auto (prefetch only when the INPUT alone exceeds L3, i.e. the
 * reads must come from DRAM), 0/1 force.  Both are measurement switches; the
 * shipped default is auto for both. */
#ifndef L8_NT
#define L8_NT (-1)
#endif
#ifndef L8_PFSEL
#define L8_PFSEL (-1)
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

/* one 1 KiB chunk (16 cache lines) of the next volume, read-prefetched.
 * L8_PF=0 compiles every prefetch out (the A/B switch for measuring). */
#ifndef L8_PF
#define L8_PF 1
#endif
#if L8_PF
#define PF(p) __builtin_prefetch((const void *)(p), 0, L8_PF_LOC)
#else
#define PF(p) ((void)0)
#endif
#define PF_CHUNK(p)                                                            \
    do {                                                                      \
        for (int t_ = 0; t_ < 16; ++t_) PF((p) + (size_t)t_ * 8);             \
    } while (0)

/* register index j of the transposed group holds z = PI[j]; feed dft8s in z order */
static const unsigned char piinv[8] = { 0, 2, 4, 6, 1, 3, 5, 7 };

struct fft3d_plan {
    int L, batch, nt, pf;
    double *scr;     /* 8 KiB: sr[64][8] then si[64][8], indexed [y*8+x] */
    void *raw;
};

const char *fft3d_name(void) { return "L8_fusedaxes"; }
const char *fft3d_description(void)
{
#if L8_MODE == 0
    return "8^3 fused x/y/z in L1, split-complex, spatial axis in the SIMD lanes";
#elif L8_MODE == 1
    return "8^3 three passes through an L1 scratch (fusion control)";
#else
    return "8^3 three separate passes over the whole batch (row-column control)";
#endif
}
int fft3d_supports(int L) { return L == 8; }

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != 8 || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    void *raw = NULL;
    if (posix_memalign(&raw, 64, 2 * 512 * sizeof(double)) != 0 || !raw) { free(p); return NULL; }
    memset(raw, 0, 2 * 512 * sizeof(double));
    p->raw = raw;
    p->scr = (double *)raw;
    /* Cache-size-relative regime switches, decided at plan time.  L3 comes from
     * sysconf (per-socket; the driver is single-threaded on an exclusive node so
     * the whole L3 is ours); fall back to the scored node's 22 MiB if unknown. */
    double l3 = (double)sysconf(_SC_LEVEL3_CACHE_SIZE);
    if (l3 <= 0.0) l3 = 22.0 * 1024.0 * 1024.0;
    /* Non-temporal final store only when in+out cannot be cache resident anyway. */
    p->nt = ((double)batch * 16384.0 > 1.18 * l3);
    /* Read-prefetch the next volume exactly when NT stores are on (same rule as
     * L8_batchsimd, which beat this entry at both node batched cases with it).
     * Measured on wallaby (SPR, 60 MiB L3): at in = 0.27 L3 prefetch costs 1.6x
     * (it clogs the fill buffers the NT stores drain through), at in = 0.76 L3
     * it is a tie, at in = 2.1 L3 it wins 1.25x.  The NT gate above already
     * guarantees in > 0.59 L3, so the pathological case cannot arise. */
    p->pf = p->nt;
#if L8_NT >= 0
    p->nt = L8_NT;
#endif
#if L8_PFSEL >= 0
    p->pf = L8_PFSEL;
#endif
    return p;
}

void fft3d_destroy(fft3d_plan *plan)
{
    if (!plan) return;
    free(plan->raw);
    free(plan);
}

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

/* ============================ mode 0: fused ============================ */
#if L8_MODE == 0

/* DOPF is a compile-time 0/1 per instantiation, so the prefetch costs the
 * non-prefetching paths nothing. */
#define VOLUME_FUSED_BODY(in, nxt, out, scr, STOREOP, DOPF)                    \
    do {                                                                      \
        double *const sr_ = (scr), *const si_ = (scr) + 512;                   \
        PHASE_A(in, sr_, si_)                                                  \
        for (int y = 0; y < 8; ++y) {                                          \
            if (DOPF) PF_CHUNK((nxt) + (size_t)y * 128);                       \
            PHASE_B_BODY(sr_ + (size_t)y * 64, si_ + (size_t)y * 64, out, y, STOREOP); \
        }                                                                      \
    } while (0)

static void volume_fused(const double *restrict in, double *restrict out,
                         double *restrict scr)
{
    VOLUME_FUSED_BODY(in, in, out, scr, ST, 0);
}

static void volume_fused_pf(const double *restrict in, const double *restrict nxt,
                            double *restrict out, double *restrict scr)
{
    VOLUME_FUSED_BODY(in, nxt, out, scr, ST, 1);
}

static void volume_fused_nt(const double *restrict in, double *restrict out,
                            double *restrict scr)
{
    VOLUME_FUSED_BODY(in, in, out, scr, NTST, 0);
}

static void volume_fused_nt_pf(const double *restrict in, const double *restrict nxt,
                               double *restrict out, double *restrict scr)
{
    VOLUME_FUSED_BODY(in, nxt, out, scr, NTST, 1);
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    const double *ip = (const double *)in;
    double *op = (double *)out;
    const long B = plan->batch;
    double *const scr = plan->scr;
    /* prefetch target: L8_PF_DIST volumes ahead, clamped to the last volume so
     * every prefetched address stays inside the caller's mapping (an out-of-range
     * prefetch cannot fault, but a not-present page costs a wasted TLB walk). */
    const double *const last = ip + (size_t)(B - 1) * 1024;
#define NXT (b + L8_PF_DIST < B ? ip + 1024 * L8_PF_DIST : last)
    /* the non-temporal store faults on a misaligned address, so re-check rather
     * than trust the contract: the driver's buffers are 64-byte aligned and every
     * half-pencil offset is a multiple of 64 bytes, but this costs one branch. */
    if (plan->nt && (((uintptr_t)op & 63u) == 0u)) {
        if (plan->pf)
            for (long b = 0; b < B; ++b, ip += 1024, op += 1024)
                volume_fused_nt_pf(ip, NXT, op, scr);
        else
            for (long b = 0; b < B; ++b, ip += 1024, op += 1024)
                volume_fused_nt(ip, op, scr);
    } else {
        if (plan->pf)
            for (long b = 0; b < B; ++b, ip += 1024, op += 1024)
                volume_fused_pf(ip, NXT, op, scr);
        else
            for (long b = 0; b < B; ++b, ip += 1024, op += 1024)
                volume_fused(ip, op, scr);
    }
#undef NXT
}

/* ================= mode 1: three passes, same scratch ================= */
#elif L8_MODE == 1

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
