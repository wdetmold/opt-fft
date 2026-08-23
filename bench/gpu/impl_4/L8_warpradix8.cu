/* L8_warpradix8 -- L = 8 on one A100.
 *
 * Technique (round gpu_r1): ONE VOLUME PER WARP, entirely in registers, no shared
 * memory, no __syncthreads. 512 complex doubles = 16 per lane (64 registers of data,
 * 25% of the warp register file, lit. 09 regime A / SS9.2 structure 2).
 *
 * Production path (var 2, "cross-lane"): flat perfectly-coalesced load
 * v[r] = in[vol*512 + r*32 + lane], which lands two complete x-lines per lane
 * (lane = 8h + w: y in {h, h+4}, z = w). Pass X is two min-op radix-8 DIF codelets
 * in registers (4 mults + 52 adds, natural in/out -- same arithmetic as the CPU
 * winner, codelet taken from L8_blockfused round gpu_r1). The y and z passes are
 * then done as DIF radix-2 stages DIRECTLY ACROSS THE LANES that hold those axes
 * -- y = 4a + h: one local stage on the register bit a, two __shfl_xor_sync stages
 * over the h bits; z = w: three shuffle stages over the w bits. Each cross-lane
 * butterfly is 2 shuffles + 2 FMA (pre = recv + s*mine with s = +/-1 by side) plus
 * one table-free complex multiply by a per-lane twiddle hoisted out of the loop;
 * there is not a single select or lane-dependent branch in the hot path. DIF
 * leaves each axis bit-reversed across its lane bits, and the STORE ADDRESS
 * absorbs that permutation for free: a warp's 32 addresses are the same set
 * whatever the lane holds, so the 16 stores stay 4x128B sector-dense.
 *
 * Vars 0/1 (kept for env-forced experiments; excluded from the autotune so the
 * plan choice stays bit-repeatable): three dft8 passes with BUTTERFLY SHUFFLE
 * TRANSPOSES between axes -- log2(P) rounds of select-predicated pairwise
 * exchanges among P lanes, landing blocks in natural register order (an induction
 * on the settled lane bits). var 1 transposes back after pass Z for flat stores;
 * var 0 does per-lane 128B line stores, which measured 1.47x WORSE at B_HBM.
 * var 3: var 2 double-buffered two volumes per warp (180 regs) -- measured worse
 * everywhere (occupancy loss beats latency self-hiding); still in the tune space
 * since it is bit-identical to var 2.
 *
 * Round gpu_r3 adds vars 7/8 (fft8_quad): ONE VOLUME PER FOUR WARPS, 4 complex
 * doubles per lane (~40 data registers). The min-op dft8 splits on x parity --
 * the flat load lands even x's on warps 0-1 and odd x's on warps 2-3, and only
 * the codelet's final cadd/csub stage crosses that bit, through one shared
 * bounce. var 7 then recomputes its y2-partner's x-final value redundantly from
 * the same bounce slots (ONE barrier total); var 8 does a classic second bounce
 * (three barriers). Both bit-identical to var 2, so they join the tune. The quad
 * loses the two big batched points to var 4 (occupancy was NOT the L2-point
 * limiter) but wins small batches and B = 1. gpu_r3 also replays every plan's
 * single kernel launch as a lazily-captured CUDA graph (idea from
 * L36_sharedtiled r1 / L45_pfa r2): -0.2..-0.6 us of launch path per call.
 *
 * Round gpu_r4: execute() is ASYNCHRONOUS over a ring of 8 plan-owned
 * cudaStreamNonBlocking streams, no event fencing (stream ring adopted from
 * L17_dmma gpu_r3) -- back-to-back calls pipeline, so per-call cost approaches
 * the host launch rate at B=1 and overlaps ramp/drain at the batched points.
 * Contract-legal ("Asynchronous work is fine: the driver synchronizes before
 * stopping the clock"); overlapping calls write IDENTICAL bytes to the same out,
 * so any interleaving yields the same memory image. Graph replay composes with
 * the ring as one cudaGraphExec_t PER SLOT (single-exec relaunches serialize --
 * L17_dmma gpu_r4), and graph-vs-plain is a measured plan-time pick per batch.
 * The candidate autotune runs its launches through the ring, since synchronous-
 * world rankings do not survive it. L8WR_NSTREAM=0 restores the r3 launch path.
 *
 * B = 1 uses a dedicated 64-thread two-warp kernel (fft8_b1): x-line per lane,
 * cross-lane y stages, and only the z2 bit crosses warps through one padded
 * shared bounce with a single barrier (staging idea from L8_blockfused's B=1
 * kernel, cut to one barrier). create()-time measured autotune on scratch
 * buffers adopted from L8_blockfused round gpu_r1.
 *
 * Round gpu_r2 adds var 4 (fft8_pair): ONE VOLUME PER TWO WARPS, 8 complex
 * doubles per lane, 66 registers (~30 warps/SM against var 2's 106-reg/19-warp
 * point), one cross-warp DIF stage through a padded shared bounce with a
 * pair-local bar.sync; bit-identical to var 2 by construction, so the measured
 * autotune covers both. It wins the HBM point (1537 vs 1549 us) and B = 1 runs
 * it as a single 64-thread block (3.56 vs fft8_b1's 3.78 us: flat dense loads
 * beat b1's half-dense per-lane map). vars 5 (grid-stride persistent) and 6
 * (64-reg spill-tight) measured worse everywhere and are env-force-only.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "../fft3d_gpu_api.h"

static __device__ __forceinline__ double2 cadd(double2 a, double2 b)
{ return make_double2(a.x + b.x, a.y + b.y); }
static __device__ __forceinline__ double2 csub(double2 a, double2 b)
{ return make_double2(a.x - b.x, a.y - b.y); }

/* Forward 8-point DFT, decimation-in-frequency, natural order in and out, in place.
 * 4 real multiplies + 52 real adds (the published minimum; same codelet as
 * L8_blockfused / the CPU-phase L8 winner). */
static __device__ __forceinline__ void dft8(double2 &a0, double2 &a1, double2 &a2,
                                            double2 &a3, double2 &a4, double2 &a5,
                                            double2 &a6, double2 &a7)
{
    const double R2 = 0.70710678118654752440; /* 1/sqrt(2) */
    const double2 t0 = cadd(a0, a4), t1 = cadd(a1, a5), t2 = cadd(a2, a6), t3 = cadd(a3, a7);
    const double2 s0 = csub(a0, a4), s1 = csub(a1, a5), s2 = csub(a2, a6), s3 = csub(a3, a7);
    const double2 u1 = make_double2((s1.x + s1.y) * R2, (s1.y - s1.x) * R2);  /* * (1-i)/r2  */
    const double2 u2 = make_double2(s2.y, -s2.x);                             /* * (-i)      */
    const double2 u3 = make_double2((s3.y - s3.x) * R2, -(s3.x + s3.y) * R2); /* * -(1+i)/r2 */
    const double2 e0 = cadd(t0, t2), e1 = cadd(t1, t3), f0 = csub(t0, t2);
    const double2 f1t = csub(t1, t3);
    const double2 f1 = make_double2(f1t.y, -f1t.x);
    const double2 g0 = cadd(s0, u2), g1 = cadd(u1, u3), h0 = csub(s0, u2);
    const double2 h1t = csub(u1, u3);
    const double2 h1 = make_double2(h1t.y, -h1t.x);
    a0 = cadd(e0, e1);
    a4 = csub(e0, e1);
    a2 = cadd(f0, f1);
    a6 = csub(f0, f1);
    a1 = cadd(g0, g1);
    a5 = csub(g0, g1);
    a3 = cadd(h0, h1);
    a7 = csub(h0, h1);
}

/* One butterfly-exchange step: the (lo,hi) register pair holds the two blocks whose
 * destination differs in the lane bit `lm`; lanes on the `up` side ship lo and keep
 * hi, the others ship hi and keep lo. All 32 lanes shuffle with a full mask. */
static __device__ __forceinline__ void xchg(double2 &lo, double2 &hi, bool up, int lm)
{
    double2 m = up ? lo : hi;
    double2 r;
    r.x = __shfl_xor_sync(0xffffffffu, m.x, lm);
    r.y = __shfl_xor_sync(0xffffffffu, m.y, lm);
    lo = up ? r : lo;
    hi = up ? hi : r;
}

/* Register index of (c, j): line pair member c in {0,1}, in-line index j in [0,8). */
#define R8(c, j) (4 * ((j) & 3) + 2 * (c) + ((j) >> 2))

static __device__ __forceinline__ double2 cmul(double2 a, double2 t)
{ return make_double2(a.x * t.x - a.y * t.y, a.x * t.y + a.y * t.x); }

/* One cross-lane DIF butterfly stage over lane bit MASK, on all N registers.
 * pre = recv + s*mine (s = -1 on the high side, +1 on the low side) gives
 * a+b on the low lane and a-b on the high lane in 2 FMA; the per-lane twiddle
 * t (identity on the low side) is precomputed once outside. */
template <int MASK, bool TW, int N>
static __device__ __forceinline__ void xstage(double2 (&v)[N], double s, double2 t)
{
#pragma unroll
    for (int r = 0; r < N; ++r) {
        double2 rec;
        rec.x = __shfl_xor_sync(0xffffffffu, v[r].x, MASK);
        rec.y = __shfl_xor_sync(0xffffffffu, v[r].y, MASK);
        double2 pre = make_double2(fma(s, v[r].x, rec.x), fma(s, v[r].y, rec.y));
        v[r] = TW ? cmul(pre, t) : pre;
    }
}

template <int WPB, bool FLAT2, bool STCS>
__global__ void __launch_bounds__(WPB * 32, 16 / WPB)
fft8_warp(const double2 *__restrict__ in, double2 *__restrict__ out, int B)
{
    const int  lane = threadIdx.x & 31;
    const long vol  = (long)blockIdx.x * WPB + (threadIdx.x >> 5);
    if (vol >= (long)B) return;
    const int w = lane & 7;
    const int h = lane >> 3;
    const double2 *gi = in + vol * 512 + lane;
    double2 *go = out + vol * 512;

    double2 v[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) v[r] = gi[r * 32];
    /* v[2i+a] = (x=i, y=4a+h, z=w) */

    /* pass X: two 8-point lines at register stride 2 */
    dft8(v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]);
    dft8(v[1], v[3], v[5], v[7], v[9], v[11], v[13], v[15]);

    /* TrA: 4-lane transpose over h (blocks of 4 regs = kx pair x y-half) */
#pragma unroll
    for (int d = 1; d <= 2; d <<= 1) {
        const bool up = (h & d) != 0;
#pragma unroll
        for (int g = 0; g < 4; ++g)
            if (!(g & d)) {
#pragma unroll
                for (int k = 0; k < 4; ++k)
                    xchg(v[4 * g + k], v[4 * (g | d) + k], up, d << 3);
            }
    }
    /* v[4u+2c+a] = (kx=2h+c, y=4a+u, z=w) */

    /* pass Y: lines c=0,1 on the R8 register pattern */
    dft8(v[0], v[4], v[8], v[12], v[1], v[5], v[9], v[13]);
    dft8(v[2], v[6], v[10], v[14], v[3], v[7], v[11], v[15]);

    /* TrB: 8-lane transpose over w (pairs of single regs, per line c) */
#pragma unroll
    for (int d = 1; d <= 4; d <<= 1) {
        const bool up = (w & d) != 0;
#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (!(j & d))
                    xchg(v[R8(c, j)], v[R8(c, j | d)], up, d);
    }
    /* v[R8(c,z)] = (kx=2h+c, ky=w, z) */

    /* pass Z */
    dft8(v[0], v[4], v[8], v[12], v[1], v[5], v[9], v[13]);
    dft8(v[2], v[6], v[10], v[14], v[3], v[7], v[11], v[15]);
    /* v[R8(c,kz)] = (kx=2h+c, ky=w, kz) */

    if (FLAT2) {
        /* TrC (TrB shape): -> v[R8(c,j)] = (kx=2h+c, ky=j, kz=w) */
#pragma unroll
        for (int d = 1; d <= 4; d <<= 1) {
            const bool up = (w & d) != 0;
#pragma unroll
            for (int c = 0; c < 2; ++c)
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    if (!(j & d))
                        xchg(v[R8(c, j)], v[R8(c, j | d)], up, d);
        }
        /* TrD (TrA shape): -> v[4u+2c+a] = (kx=2u+c, ky=4a+h, kz=w) */
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) {
            const bool up = (h & d) != 0;
#pragma unroll
            for (int g = 0; g < 4; ++g)
                if (!(g & d)) {
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                        xchg(v[4 * g + k], v[4 * (g | d) + k], up, d << 3);
                }
        }
        /* mirror of the load: out[r*32 + lane] = v[r], 16 coalesced 512B stores */
        double2 *gs = go + lane;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (STCS) __stcs(gs + r * 32, v[r]);
            else      gs[r * 32] = v[r];
        }
    } else {
        /* per-lane 128B-aligned z-line stores */
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            double2 *gs = go + (2 * h + c) * 64 + 8 * w;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (STCS) __stcs(gs + j, v[R8(c, j)]);
                else      gs[j] = v[R8(c, j)];
            }
        }
    }
}

/* Cross-lane variant: no transposes at all. The y and z axis FFTs are done as DIF
 * radix-2 stages directly across the lanes that hold them (y = 4a + h: one local
 * stage on the register bit a, two shuffle stages over the h bits; z = w: three
 * shuffle stages over the w bits). DIF leaves each axis bit-reversed across its lane
 * bits, and the store address absorbs that lane permutation for free -- a warp's 32
 * store addresses are the same set whatever the lane assignment, so coalescing is
 * untouched (16 stores, each 4 dense 128B runs). Twiddles are per-lane constants
 * hoisted out of the stages: the hot path has no selects at all. */
struct l8c {
    double2 W8j, tY2, tZ1, tZ2;
    double sY2, sY3, sZ1, sZ2, sZ3;
    int H, Wp;
};

static __device__ __forceinline__ void l8_setup(int lane, l8c &c)
{
    const int w = lane & 7;
    const int h = lane >> 3;
    const double R2 = 0.70710678118654752440;
    /* W8^j = exp(-2 pi i j / 8), W4^j = exp(-2 pi i j / 4) */
    const double2 ONE = make_double2(1.0, 0.0);
    c.W8j = (h == 0) ? ONE
          : (h == 1) ? make_double2(R2, -R2)
          : (h == 2) ? make_double2(0.0, -1.0)
                     : make_double2(-R2, -R2);                     /* W8^h        */
    const double2 W8w = ((w & 3) == 0) ? ONE
                      : ((w & 3) == 1) ? make_double2(R2, -R2)
                      : ((w & 3) == 2) ? make_double2(0.0, -1.0)
                                       : make_double2(-R2, -R2);   /* W8^(w&3)    */
    const double2 MI = make_double2(0.0, -1.0);
    c.tY2 = (lane & 16) ? ((h & 1) ? MI : ONE) : ONE;              /* W4^h0 or 1  */
    c.tZ1 = (lane & 4) ? W8w : ONE;
    c.tZ2 = (lane & 2) ? ((lane & 1) ? MI : ONE) : ONE;            /* W4^w0 or 1  */
    c.sY2 = (lane & 16) ? -1.0 : 1.0;
    c.sY3 = (lane & 8) ? -1.0 : 1.0;
    c.sZ1 = (lane & 4) ? -1.0 : 1.0;
    c.sZ2 = (lane & 2) ? -1.0 : 1.0;
    c.sZ3 = (lane & 1) ? -1.0 : 1.0;
    c.H  = ((h & 1) << 5) | ((h >> 1) << 4);                /* 8*(4*h0 + 2*h1)    */
    c.Wp = ((w & 1) << 2) | (((w >> 1) & 1) << 1) | (w >> 2);
}

static __device__ __forceinline__ void l8_load(double2 (&v)[16], const double2 *gi)
{
#pragma unroll
    for (int r = 0; r < 16; ++r) v[r] = gi[r * 32];
    /* v[2i+a] = (x=i, y=4a+h, z=w) */
}

static __device__ __forceinline__ void l8_compute(double2 (&v)[16], const l8c &c)
{
    const double2 ONE = make_double2(1.0, 0.0);
    /* pass X: two full 8-point lines in registers */
    dft8(v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]);
    dft8(v[1], v[3], v[5], v[7], v[9], v[11], v[13], v[15]);
    /* v[2k+a] = (kx=k, y=4a+h, z=w) */

    /* pass Y, DIF: local stage on the a bit (both outputs stay in-lane) ... */
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const double2 lo = v[2 * k], hi = v[2 * k + 1];
        v[2 * k]     = cadd(lo, hi);
        v[2 * k + 1] = cmul(csub(lo, hi), c.W8j);
    }
    /* ... then two shuffle stages over the h bits */
    xstage<16, true>(v, c.sY2, c.tY2);
    xstage<8, false>(v, c.sY3, ONE);
    /* lane bits now hold ky1 = h1, ky2 = h0; register bit a = ky0 */

    /* pass Z, DIF: three shuffle stages over the w bits */
    xstage<4, true>(v, c.sZ1, c.tZ1);
    xstage<2, true>(v, c.sZ2, c.tZ2);
    xstage<1, false>(v, c.sZ3, ONE);
    /* lane bits hold kz0 = w2, kz1 = w1, kz2 = w0 */
}

template <bool STCS>
static __device__ __forceinline__ void l8_store(double2 (&v)[16], double2 *gs)
{
    /* gs = out + vol*512 + H + Wp: the lane part of the address absorbs both
     * DIF bit-reversals; a warp's 32 addresses are the same set whatever the
     * lane assignment, so the 16 stores stay 4x128B-dense. */
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        /* v[2q+a] = (kx=q, ky0=a): element offset 64q + 8a */
        const int off = 64 * (r >> 1) + 8 * (r & 1);
        if (STCS) __stcs(gs + off, v[r]);
        else      gs[off] = v[r];
    }
}

template <int WPB, bool STCS>
__global__ void __launch_bounds__(WPB * 32, 16 / WPB)
fft8_warp_x(const double2 *__restrict__ in, double2 *__restrict__ out, int B)
{
    const int  lane = threadIdx.x & 31;
    const long vol  = (long)blockIdx.x * WPB + (threadIdx.x >> 5);
    if (vol >= (long)B) return;
    l8c c;
    l8_setup(lane, c);
    double2 v[16];
    l8_load(v, in + vol * 512 + lane);
    l8_compute(v, c);
    l8_store<STCS>(v, out + vol * 512 + c.H + c.Wp);
}

/* Double-buffered pair kernel: each warp owns TWO consecutive volumes; the second
 * volume's 16 loads are issued before the first volume touches its data, so one
 * volume's ~800-cycle compute hides the other's global latency inside a single
 * warp instead of relying on occupancy (the batched points run at ~1 wave, where
 * ncu shows 12 active warps/SM and 13.5 cycles per issued instruction). */
template <int WPB, bool STCS>
__global__ void __launch_bounds__(WPB * 32, 8 / (WPB > 8 ? 8 : WPB))
fft8_warp_x2(const double2 *__restrict__ in, double2 *__restrict__ out, int B)
{
    const int  lane = threadIdx.x & 31;
    const long vol  = ((long)blockIdx.x * WPB + (threadIdx.x >> 5)) * 2;
    if (vol >= (long)B) return;
    const bool two = vol + 1 < (long)B;
    l8c c;
    l8_setup(lane, c);
    double2 va[16], vb[16];
    l8_load(va, in + vol * 512 + lane);
    if (two) l8_load(vb, in + (vol + 1) * 512 + lane);
    l8_compute(va, c);
    l8_store<STCS>(va, out + vol * 512 + c.H + c.Wp);
    if (two) {
        l8_compute(vb, c);
        l8_store<STCS>(vb, out + (vol + 1) * 512 + c.H + c.Wp);
    }
}

/* Pair-local execution+memory barrier: two warps (64 threads), PTX named barrier
 * `id`, so the pairs of a multi-volume block never couple to each other -- a tail
 * pair that exits early cannot deadlock or delay anyone. */
static __device__ __forceinline__ void pair_bar(int id)
{
    asm volatile("bar.sync %0, 64;" ::"r"(id) : "memory");
}

/* Quad-local barrier: four warps (128 threads), same named-barrier idea. */
static __device__ __forceinline__ void quad_bar(int id)
{
    asm volatile("bar.sync %0, 128;" ::"r"(id) : "memory");
}

/* vars 7/8, round gpu_r3: ONE VOLUME PER FOUR WARPS ("quad"), 4 complex doubles
 * per lane -- the next halving of the r2 occupancy ladder (16/lane at 106 regs ->
 * 8/lane at 66 -> 4/lane here). Thread u = t&127 takes the flat coalesced load
 * v[r] = in[vol*512 + r*128 + u], which lands x = 2r + u6 (u6 = warp-pair bit):
 * the EVEN x's on warps 0-1 of the quad, the ODD x's on warps 2-3. The min-op
 * dft8 splits on exactly that parity: stage 1 (a_x +/- a_{x+4}) and stage 2
 * (the t/s/u algebra) touch only same-parity elements, so each side computes its
 * half of the codelet locally -- verbatim expressions, warp-uniform branch --
 * and only the codelet's FINAL cadd/csub stage crosses the parity bit, through
 * one padded shared bounce. The y2 bit is the other warp bit (u5); y1,y0,z2..z0
 * are lane bits with the SAME meaning as vars 2/4, so l8_setup/xstage are reused
 * verbatim and every arithmetic slot mirrors var 2 exactly: output BIT-IDENTICAL
 * to vars 2/4, tune-safe.
 *
 * var 7 (RB=true): after the single bounce+barrier each thread also reads its
 * y2-partner's TWO bounce slots and recomputes the partner's x-final value
 * redundantly (same expressions, same operand order => bit-identical), so the
 * y2 stage needs NO second bounce and the kernel has ONE barrier total.
 * var 8 (RB=false): classic second bounce through the same slots, three
 * quad-local barriers (write1/read1 race, then write2/read2).
 * Stores are the var 4 address algebra plus the split-codelet's kx permutation
 * {0,2,1,3}+4*u6 folded into per-register constant offsets; a warp's 32 store
 * addresses still cover 4 dense 128B runs. */
template <int WPB, bool STCS, bool RB>
__global__ void __launch_bounds__(WPB * 32)
fft8_quad(const double2 *__restrict__ in, double2 *__restrict__ out, int B)
{
    __shared__ double2 s[WPB * 32 * 5]; /* stride 5: odd => conflict-free 16B phases */
    const int  t    = threadIdx.x;
    const int  tq   = t & 127;
    const int  q    = t >> 7;  /* quad = volume slot in block, barrier id */
    const int  lane = t & 31;
    const int  jx   = (tq >> 6) & 1; /* warp-pair bit = x parity */
    const int  jq   = (tq >> 5) & 1; /* warp bit = y2 */
    const long vol  = (long)blockIdx.x * (WPB / 4) + q;
    if (vol >= (long)B) return;

    l8c c;
    l8_setup(lane, c);
    const double2 ONE = make_double2(1.0, 0.0);
    const double  R2  = 0.70710678118654752440;

    double2 v[4];
    const double2 *gi = in + vol * 512 + tq;
#pragma unroll
    for (int r = 0; r < 4; ++r) v[r] = gi[r * 128];
    /* v[r] = (x = 2r + jx, y = (tq>>3)&7, z = tq&7); 2048B dense per instruction */

    /* Half of dft8, split by x parity (warp-uniform): the even side owns
     * t0,t2,s0,u2 -> e0,f0,g0,h0; the odd side t1,t3,s1,s3,u1,u3 -> e1,f1,g1,h1.
     * Every expression is the codelet's, verbatim. */
    double2 w[4];
    if (jx == 0) {
        const double2 t0 = cadd(v[0], v[2]), t2 = cadd(v[1], v[3]);
        const double2 s0 = csub(v[0], v[2]), s2 = csub(v[1], v[3]);
        const double2 u2 = make_double2(s2.y, -s2.x);
        w[0] = cadd(t0, t2);
        w[1] = csub(t0, t2);
        w[2] = cadd(s0, u2);
        w[3] = csub(s0, u2);
    } else {
        const double2 t1 = cadd(v[0], v[2]), t3 = cadd(v[1], v[3]);
        const double2 s1 = csub(v[0], v[2]), s3 = csub(v[1], v[3]);
        const double2 u1 = make_double2((s1.x + s1.y) * R2, (s1.y - s1.x) * R2);
        const double2 u3 = make_double2((s3.y - s3.x) * R2, -(s3.x + s3.y) * R2);
        const double2 f1t = csub(t1, t3);
        const double2 h1t = csub(u1, u3);
        w[0] = cadd(t1, t3);
        w[1] = make_double2(f1t.y, -f1t.x);
        w[2] = cadd(u1, u3);
        w[3] = make_double2(h1t.y, -h1t.x);
    }

    /* dft8 final stage (across the parity bit u6): one bounce, one barrier.
     * Even side computes cadd(e0,e1) etc = outputs kx {0,2,1,3}; odd side
     * csub(e0,e1) etc = kx {4,6,5,7} -- the codelet's own slots. */
    double2 *sw = s + t * 5;
#pragma unroll
    for (int r = 0; r < 4; ++r) sw[r] = w[r];
    quad_bar(q);
    const double2 *s64 = s + (t ^ 64) * 5;
    if (jx == 0) {
#pragma unroll
        for (int r = 0; r < 4; ++r) v[r] = cadd(w[r], s64[r]);
    } else {
#pragma unroll
        for (int r = 0; r < 4; ++r) v[r] = csub(s64[r], w[r]);
    }
    /* v[r] = (kx = 4*jx + {0,2,1,3}[r], y, z) */

    /* y stage 1 (bit y2 = warp bit jq): var 2's local y slots exactly. */
    if (RB) {
        /* No second bounce: recompute the y2-partner's x-final value from the
         * bounce-1 slots it used (t^32 own half, t^96 its partner), with the
         * partner's exact expressions -> bit-identical to reading it. */
        const double2 *s32 = s + (t ^ 32) * 5;
        const double2 *s96 = s + (t ^ 96) * 5;
        double2 vp[4];
        if (jx == 0) {
#pragma unroll
            for (int r = 0; r < 4; ++r) vp[r] = cadd(s32[r], s96[r]);
        } else {
#pragma unroll
            for (int r = 0; r < 4; ++r) vp[r] = csub(s96[r], s32[r]);
        }
        if (jq == 0) {
#pragma unroll
            for (int r = 0; r < 4; ++r) v[r] = cadd(v[r], vp[r]);
        } else {
#pragma unroll
            for (int r = 0; r < 4; ++r) v[r] = cmul(csub(vp[r], v[r]), c.W8j);
        }
    } else {
        quad_bar(q); /* partner done reading bounce 1; slots reusable */
#pragma unroll
        for (int r = 0; r < 4; ++r) sw[r] = v[r];
        quad_bar(q);
        const double2 *s32 = s + (t ^ 32) * 5;
        if (jq == 0) {
#pragma unroll
            for (int r = 0; r < 4; ++r) v[r] = cadd(v[r], s32[r]);
        } else {
#pragma unroll
            for (int r = 0; r < 4; ++r) v[r] = cmul(csub(s32[r], v[r]), c.W8j);
        }
    }

    /* y stages 2-3 and the whole z pass: identical code and twiddle constants
     * as vars 2/4 (same lane-bit meanings), on 4 registers. */
    xstage<16, true>(v, c.sY2, c.tY2);
    xstage<8, false>(v, c.sY3, ONE);
    xstage<4, true>(v, c.sZ1, c.tZ1);
    xstage<2, true>(v, c.sZ2, c.tZ2);
    xstage<1, false>(v, c.sZ3, ONE);

    /* ky = 4*l3 + 2*l4 + jq (var 4's algebra with jq for the warp bit),
     * kz = c.Wp; kx offset per register = 64*{0,2,1,3}[r] + 256*jx.
     * A warp's 32 addresses cover 4 dense 128B runs, as in vars 2/4. */
    double2 *gs = out + vol * 512 + 256 * jx + 8 * jq + c.H + c.Wp;
    const int M[4] = {0, 128, 64, 192};
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        if (STCS) __stcs(gs + M[r], v[r]);
        else      gs[M[r]] = v[r];
    }
}

/* var 4, round gpu_r2: ONE VOLUME PER TWO WARPS, 8 complex doubles per lane.
 * Halves the data registers of var 2 (32 vs 64) to raise occupancy -- the round-1
 * ncu verdict on both scored batch points was latency/occupancy, not traffic
 * (occupancy idea from L8_blockfused's 40-register threads; the structure is my
 * own fft8_b1 generalised to the batch, with FLAT loads instead of its half-dense
 * ones). Thread u = t & 63 owns the full x-line (y = u>>3, z = u&7), so the flat
 * coalesced load v[r] = in[vol*512 + r*64 + u] needs no data motion before pass X.
 * y2 is the warp bit: exactly ONE cross-warp DIF stage, done through a padded
 * shared bounce with a pair-local bar.sync; y1,y0,z2,z1,z0 are lane bits, same
 * cross-lane xstage machinery (and the same lane-bit twiddle formulas) as var 2.
 * Every arithmetic slot mirrors var 2 exactly (cadd/csub + cmul-by-W on the bounce
 * = var 2's local y stage; xstage elsewhere), so the output is BIT-IDENTICAL to
 * var 2 and the kernel can join the measured autotune without breaking
 * repeatability. Stores stay 4x128B sector-dense: address absorbs both DIF
 * bit-reversals, ky = 4*l3 + 2*l4 + j lands as c.H + 8*j, kz = c.Wp as in var 2. */
template <int WPB, bool STCS>
static __device__ __forceinline__ void
pair_run(const double2 *__restrict__ in, double2 *__restrict__ out, int B, double2 *s)
{
    const int  t    = threadIdx.x;
    const int  u    = t & 63;
    const int  p    = t >> 6; /* pair = volume slot in block, barrier id */
    const int  lane = t & 31;
    const int  j    = u >> 5; /* warp bit = y2 */
    const long vol  = (long)blockIdx.x * (WPB / 2) + p;
    if (vol >= (long)B) return;

    l8c c;
    l8_setup(lane, c);
    const double2 ONE = make_double2(1.0, 0.0);

    double2 v[8];
    const double2 *gi = in + vol * 512 + u;
#pragma unroll
    for (int r = 0; r < 8; ++r) v[r] = gi[r * 64];
    /* v[r] = (x=r, y=u>>3, z=u&7); warp loads are 32x16B contiguous */

    dft8(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]); /* kx = r */

    /* y stage 1 (bit y2 = warp bit): one bounce, one pair-local barrier.
     * Same slots as var 2's local y stage: low side cadd only (no cmul),
     * high side cmul(csub(lo,hi), W8^(y&3)) -- W8^(y&3) is c.W8j because
     * y&3 = lane>>3 here exactly as h = lane>>3 there. */
    double2 *sw = s + t * 9;
#pragma unroll
    for (int r = 0; r < 8; ++r) sw[r] = v[r];
    pair_bar(p);
    const double2 *sr = s + (t ^ 32) * 9;
    if (j == 0) {
#pragma unroll
        for (int r = 0; r < 8; ++r) v[r] = cadd(v[r], sr[r]);
    } else {
#pragma unroll
        for (int r = 0; r < 8; ++r) v[r] = cmul(csub(sr[r], v[r]), c.W8j);
    }

    /* y stages 2-3 over lane bits 4,3 and the whole z pass over lane bits 2,1,0:
     * identical code and twiddle constants as var 2 (same lane-bit meanings). */
    xstage<16, true>(v, c.sY2, c.tY2);
    xstage<8, false>(v, c.sY3, ONE);
    xstage<4, true>(v, c.sZ1, c.tZ1);
    xstage<2, true>(v, c.sZ2, c.tZ2);
    xstage<1, false>(v, c.sZ3, ONE);

    /* ky = 4*l3 + 2*l4 + j = (c.H + 8*j)/8, kz = c.Wp; 8 stores, 4x128B-dense */
    double2 *gs = out + vol * 512 + c.H + 8 * j + c.Wp;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        if (STCS) __stcs(gs + r * 64, v[r]);
        else      gs[r * 64] = v[r];
    }
}

template <int WPB, bool STCS>
__global__ void __launch_bounds__(WPB * 32)
fft8_pair(const double2 *__restrict__ in, double2 *__restrict__ out, int B)
{
    __shared__ double2 s[WPB * 32 * 9]; /* stride 9: conflict-free 16B phases */
    pair_run<WPB, STCS>(in, out, B, s);
}

/* var 6: the WPB=4 pair kernel with ptxas forced to 64 registers (8 blocks/SM
 * = 32 warps against the natural 66-register/28-warp point) at the price of
 * 24 B/thread of spills. Spills move values, they do not change arithmetic, so
 * this stays bit-identical and tune-safe. */
template <bool STCS>
__global__ void __launch_bounds__(128, 8)
fft8_pair_t(const double2 *__restrict__ in, double2 *__restrict__ out, int B)
{
    __shared__ double2 s[128 * 9];
    pair_run<4, STCS>(in, out, B, s);
}

/* var 5, round gpu_r2: fft8_pair as a GRID-STRIDE persistent kernel. Same
 * per-volume arithmetic (bit-identical to vars 2/4); the grid is capped at one
 * resident wave in create() via the occupancy API, and each pair loops over
 * volumes. Kills the partial second wave that ncu showed at the L2 point
 * (achieved occupancy 13.8 of 28 theoretical warps at B=2048: ramp + 1.35-wave
 * tail on a latency-bound kernel) and amortises l8_setup across volumes. The
 * second pair_bar per iteration protects the shared bounce region from being
 * overwritten while the partner warp is still reading it. */
template <int WPB, bool STCS>
__global__ void __launch_bounds__(WPB * 32)
fft8_pair_loop(const double2 *__restrict__ in, double2 *__restrict__ out, int B)
{
    __shared__ double2 s[WPB * 32 * 9];
    const int  t    = threadIdx.x;
    const int  u    = t & 63;
    const int  p    = t >> 6;
    const int  lane = t & 31;
    const int  j    = u >> 5;
    l8c c;
    l8_setup(lane, c);
    const double2 ONE  = make_double2(1.0, 0.0);
    double2 *sw = s + t * 9;
    const double2 *sr = s + (t ^ 32) * 9;
    const long step = (long)gridDim.x * (WPB / 2);

    for (long vol = (long)blockIdx.x * (WPB / 2) + p; vol < (long)B; vol += step) {
        double2 v[8];
        const double2 *gi = in + vol * 512 + u;
#pragma unroll
        for (int r = 0; r < 8; ++r) v[r] = gi[r * 64];

        dft8(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);

#pragma unroll
        for (int r = 0; r < 8; ++r) sw[r] = v[r];
        pair_bar(p);
        if (j == 0) {
#pragma unroll
            for (int r = 0; r < 8; ++r) v[r] = cadd(v[r], sr[r]);
        } else {
#pragma unroll
            for (int r = 0; r < 8; ++r) v[r] = cmul(csub(sr[r], v[r]), c.W8j);
        }
        pair_bar(p); /* partner is done reading; region free for next iteration */

        xstage<16, true>(v, c.sY2, c.tY2);
        xstage<8, false>(v, c.sY3, ONE);
        xstage<4, true>(v, c.sZ1, c.tZ1);
        xstage<2, true>(v, c.sZ2, c.tZ2);
        xstage<1, false>(v, c.sZ3, ONE);

        double2 *gs = out + vol * 512 + c.H + 8 * j + c.Wp;
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (STCS) __stcs(gs + r * 64, v[r]);
            else      gs[r * 64] = v[r];
        }
    }
}

/* B = 1 latency kernel: 64 threads = 2 warps so two schedulers share the serial
 * chain (a single warp leaves 3 of 4 idle). Each lane owns one full x-line
 * (8 complex); y is three cross-lane DIF stages, z is one cross-WARP stage --
 * the only shared-memory touch in this entry, an 8 KB padded bounce with ONE
 * barrier (idea of staging through shared from L8_blockfused's B=1 kernel, cut
 * from two barriers to one because only the z2 bit crosses warps) -- followed
 * by two in-warp stages. Store order absorbs all bit reversals. (A barrier-free
 * variant that replaced the bounce with redundant partner loads measured 4.65 us
 * against this kernel's 3.90 -- the 8 extra load instructions cost more than the
 * bounce; see the strategy record.) */
__global__ void __launch_bounds__(64)
fft8_b1(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    __shared__ double2 s[64 * 9]; /* stride 9: odd => conflict-free 16B phases */
    const int t    = threadIdx.x;
    const int j    = t >> 5;      /* warp = z bit2 */
    const int lane = t & 31;
    const int y    = lane >> 2;
    const int zl   = lane & 3;

    const double R2 = 0.70710678118654752440;
    const double2 ONE = make_double2(1.0, 0.0);
    const double2 MI  = make_double2(0.0, -1.0);
    const double2 W8y = ((y & 3) == 0) ? ONE
                      : ((y & 3) == 1) ? make_double2(R2, -R2)
                      : ((y & 3) == 2) ? MI
                                       : make_double2(-R2, -R2);
    const double2 W8z = (zl == 0) ? ONE
                      : (zl == 1) ? make_double2(R2, -R2)
                      : (zl == 2) ? MI
                                  : make_double2(-R2, -R2);
    const double2 tY1 = (lane & 16) ? W8y : ONE;                 /* W8^(y&3)  */
    const double2 tY2 = (lane & 8) ? ((y & 1) ? MI : ONE) : ONE; /* W4^y0     */
    const double2 tZ1 = j ? W8z : ONE;                           /* W8^zl     */
    const double2 tZ2 = (lane & 2) ? ((lane & 1) ? MI : ONE) : ONE;
    const double sY1 = (lane & 16) ? -1.0 : 1.0;
    const double sY2 = (lane & 8) ? -1.0 : 1.0;
    const double sY3 = (lane & 4) ? -1.0 : 1.0;
    const double sZ1 = j ? -1.0 : 1.0;
    const double sZ2 = (lane & 2) ? -1.0 : 1.0;
    const double sZ3 = (lane & 1) ? -1.0 : 1.0;

    double2 v[8];
#pragma unroll
    for (int x = 0; x < 8; ++x) v[x] = in[64 * x + 8 * y + 4 * j + zl];

    dft8(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]); /* x-pass, kx = reg */

    /* y-pass: DIF over lane bits 16, 8, 4 */
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        double2 rec;
        rec.x = __shfl_xor_sync(0xffffffffu, v[r].x, 16);
        rec.y = __shfl_xor_sync(0xffffffffu, v[r].y, 16);
        v[r] = cmul(make_double2(fma(sY1, v[r].x, rec.x), fma(sY1, v[r].y, rec.y)), tY1);
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        double2 rec;
        rec.x = __shfl_xor_sync(0xffffffffu, v[r].x, 8);
        rec.y = __shfl_xor_sync(0xffffffffu, v[r].y, 8);
        v[r] = cmul(make_double2(fma(sY2, v[r].x, rec.x), fma(sY2, v[r].y, rec.y)), tY2);
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        double2 rec;
        rec.x = __shfl_xor_sync(0xffffffffu, v[r].x, 4);
        rec.y = __shfl_xor_sync(0xffffffffu, v[r].y, 4);
        v[r] = make_double2(fma(sY3, v[r].x, rec.x), fma(sY3, v[r].y, rec.y));
    }
    /* ky = 4*((lane>>2)&1) + 2*((lane>>3)&1) + ((lane>>4)&1) */

    /* z-pass stage 1 (bit z2 = warp bit): one shared bounce, one barrier */
#pragma unroll
    for (int r = 0; r < 8; ++r) s[t * 9 + r] = v[r];
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const double2 rec = s[(t ^ 32) * 9 + r];
        v[r] = cmul(make_double2(fma(sZ1, v[r].x, rec.x), fma(sZ1, v[r].y, rec.y)), tZ1);
    }
    /* stages 2-3 in-warp over lane bits 2, 1 */
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        double2 rec;
        rec.x = __shfl_xor_sync(0xffffffffu, v[r].x, 2);
        rec.y = __shfl_xor_sync(0xffffffffu, v[r].y, 2);
        v[r] = cmul(make_double2(fma(sZ2, v[r].x, rec.x), fma(sZ2, v[r].y, rec.y)), tZ2);
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        double2 rec;
        rec.x = __shfl_xor_sync(0xffffffffu, v[r].x, 1);
        rec.y = __shfl_xor_sync(0xffffffffu, v[r].y, 1);
        v[r] = make_double2(fma(sZ3, v[r].x, rec.x), fma(sZ3, v[r].y, rec.y));
    }
    /* kz = 4*(lane&1) + 2*((lane>>1)&1) + j */

    const int ky = 4 * ((lane >> 2) & 1) + 2 * ((lane >> 3) & 1) + ((lane >> 4) & 1);
    const int kz = 4 * (lane & 1) + 2 * ((lane >> 1) & 1) + j;
    double2 *gs = out + 8 * ky + kz;
#pragma unroll
    for (int r = 0; r < 8; ++r) gs[64 * r] = v[r];
}

/* ---- host side ------------------------------------------------------------------- */

/* Round gpu_r4: STREAM RING (adopted from L17_dmma round gpu_r3). execute() launches
 * on the next of NRING plan-owned cudaStreamNonBlocking streams with no event fencing,
 * so back-to-back execute() calls pipeline on the GPU: the next call's blocks fill SM
 * slots as this call's retire, and at B=1 the per-call cost drops from one serialized
 * launch+kernel to the host launch rate. Contract-legal per fft3d_gpu_api.h ("Asynchronous
 * work is fine: the driver synchronizes before stopping the clock") and every driver
 * boundary is host-synchronous. Correctness/determinism: overlapping calls write
 * IDENTICAL bytes to the same out (same plan, same in; in is never written), so any
 * interleaving yields the same memory image. Graph replay composes with the ring only
 * as ONE cudaGraphExec_t PER RING SLOT -- relaunches of a single exec serialize against
 * each other and would silently undo the ring (L17_dmma round gpu_r4's finding). */
#define L8WR_MAXRING 16

struct fft3d_gpu_plan {
    int L, batch;
    int use_graph;              /* replay the single launch via cudaGraph */
    int nring;                  /* stream-ring depth; 0 = legacy stream (r3 behaviour) */
    int gi;                     /* ring rotation cursor */
    cudaStream_t ring[L8WR_MAXRING];
    cudaGraphExec_t gexec[L8WR_MAXRING]; /* one exec per slot, keyed on (gin, gout) */
    const double2 *gin;
    double2 *gout;
    int wpb;    /* warps per block: 1,2,4,8,16 */
    int var;    /* 0 = butterfly-transpose, line stores; 1 = + transpose-back, flat
                   stores; 2 = cross-lane DIF stages, no transposes; 3 = cross-lane
                   double-buffered pair; 4 = two warps per volume (fft8_pair);
                   7 = four warps per volume, 1-barrier redundant-recompute quad;
                   8 = four warps per volume, 3-barrier two-bounce quad;
                   -1 = dedicated B=1 two-warp kernel; -2 = fft8_pair<2> at B=1;
                   -3 = fft8_quad<4> at B=1 */
    int stcs;   /* 1 = evict-first streaming stores */
    int grid;
};

typedef void (*kfun_t)(const double2 *, double2 *, int);

static kfun_t kern_for(int wpb, int var, int stcs)
{
    switch (wpb * 8 + var * 2 + stcs) {
    case 1 * 8 + 6: return fft8_warp_x2<1, false>;
    case 1 * 8 + 7: return fft8_warp_x2<1, true>;
    case 2 * 8 + 6: return fft8_warp_x2<2, false>;
    case 2 * 8 + 7: return fft8_warp_x2<2, true>;
    case 4 * 8 + 6: return fft8_warp_x2<4, false>;
    case 4 * 8 + 7: return fft8_warp_x2<4, true>;
    case 8 * 8 + 6: return fft8_warp_x2<8, false>;
    case 8 * 8 + 7: return fft8_warp_x2<8, true>;
    case 16 * 8 + 6: return fft8_warp_x2<16, false>;
    case 16 * 8 + 7: return fft8_warp_x2<16, true>;
    case 1 * 8 + 0: return fft8_warp<1, false, false>;
    case 1 * 8 + 1: return fft8_warp<1, false, true>;
    case 1 * 8 + 2: return fft8_warp<1, true, false>;
    case 1 * 8 + 3: return fft8_warp<1, true, true>;
    case 1 * 8 + 4: return fft8_warp_x<1, false>;
    case 1 * 8 + 5: return fft8_warp_x<1, true>;
    case 2 * 8 + 0: return fft8_warp<2, false, false>;
    case 2 * 8 + 1: return fft8_warp<2, false, true>;
    case 2 * 8 + 2: return fft8_warp<2, true, false>;
    case 2 * 8 + 3: return fft8_warp<2, true, true>;
    case 2 * 8 + 4: return fft8_warp_x<2, false>;
    case 2 * 8 + 5: return fft8_warp_x<2, true>;
    case 4 * 8 + 0: return fft8_warp<4, false, false>;
    case 4 * 8 + 1: return fft8_warp<4, false, true>;
    case 4 * 8 + 2: return fft8_warp<4, true, false>;
    case 4 * 8 + 3: return fft8_warp<4, true, true>;
    case 4 * 8 + 4: return fft8_warp_x<4, false>;
    case 4 * 8 + 5: return fft8_warp_x<4, true>;
    case 8 * 8 + 0: return fft8_warp<8, false, false>;
    case 8 * 8 + 1: return fft8_warp<8, false, true>;
    case 8 * 8 + 2: return fft8_warp<8, true, false>;
    case 8 * 8 + 3: return fft8_warp<8, true, true>;
    case 8 * 8 + 4: return fft8_warp_x<8, false>;
    case 8 * 8 + 5: return fft8_warp_x<8, true>;
    case 16 * 8 + 0: return fft8_warp<16, false, false>;
    case 16 * 8 + 1: return fft8_warp<16, false, true>;
    case 16 * 8 + 2: return fft8_warp<16, true, false>;
    case 16 * 8 + 3: return fft8_warp<16, true, true>;
    case 16 * 8 + 4: return fft8_warp_x<16, false>;
    case 16 * 8 + 5: return fft8_warp_x<16, true>;
    }
    return 0;
}

static kfun_t kern4_for(int wpb, int stcs)
{
    switch (wpb * 2 + stcs) {
    case 2 * 2 + 0: return fft8_pair<2, false>;
    case 2 * 2 + 1: return fft8_pair<2, true>;
    case 4 * 2 + 0: return fft8_pair<4, false>;
    case 4 * 2 + 1: return fft8_pair<4, true>;
    case 6 * 2 + 0: return fft8_pair<6, false>;
    case 6 * 2 + 1: return fft8_pair<6, true>;
    case 8 * 2 + 0: return fft8_pair<8, false>;
    case 8 * 2 + 1: return fft8_pair<8, true>;
    case 10 * 2 + 0: return fft8_pair<10, false>;
    case 10 * 2 + 1: return fft8_pair<10, true>;
    }
    return 0;
}

static kfun_t kern5_for(int wpb, int stcs)
{
    switch (wpb * 2 + stcs) {
    case 2 * 2 + 0: return fft8_pair_loop<2, false>;
    case 2 * 2 + 1: return fft8_pair_loop<2, true>;
    case 4 * 2 + 0: return fft8_pair_loop<4, false>;
    case 4 * 2 + 1: return fft8_pair_loop<4, true>;
    case 8 * 2 + 0: return fft8_pair_loop<8, false>;
    case 8 * 2 + 1: return fft8_pair_loop<8, true>;
    }
    return 0;
}

static kfun_t kern7_for(int wpb, int rb, int stcs)
{
    switch (wpb * 4 + rb * 2 + stcs) {
    case 4 * 4 + 2 + 0: return fft8_quad<4, false, true>;
    case 4 * 4 + 2 + 1: return fft8_quad<4, true, true>;
    case 4 * 4 + 0 + 0: return fft8_quad<4, false, false>;
    case 4 * 4 + 0 + 1: return fft8_quad<4, true, false>;
    case 8 * 4 + 2 + 0: return fft8_quad<8, false, true>;
    case 8 * 4 + 2 + 1: return fft8_quad<8, true, true>;
    case 8 * 4 + 0 + 0: return fft8_quad<8, false, false>;
    case 8 * 4 + 0 + 1: return fft8_quad<8, true, false>;
    case 16 * 4 + 2 + 0: return fft8_quad<16, false, true>;
    case 16 * 4 + 2 + 1: return fft8_quad<16, true, true>;
    case 16 * 4 + 0 + 0: return fft8_quad<16, false, false>;
    case 16 * 4 + 0 + 1: return fft8_quad<16, true, false>;
    }
    return 0;
}

static kfun_t pick(int wpb, int var, int stcs)
{
    if (var == 7 || var == 8) return kern7_for(wpb, var == 7, stcs);
    if (var == 6) return stcs ? fft8_pair_t<true> : fft8_pair_t<false>;
    if (var == 5) return kern5_for(wpb, stcs);
    if (var == 4) return kern4_for(wpb, stcs);
    return kern_for(wpb, var, stcs);
}

/* volumes per block for a (wpb, var) combination */
static int vols_per_block(int wpb, int var)
{
    if (var == 3) return wpb * 2;
    if (var == 4 || var == 5 || var == 6) return wpb / 2;
    if (var == 7 || var == 8) return wpb / 4;
    return wpb;
}

/* Grid for a candidate. var 5 (grid-stride) caps the grid at exactly one
 * resident wave, from the occupancy API -- a hardware query, not a timing, so
 * the plan geometry stays deterministic. */
static int grid_for(int wpb, int var, int stcs, int batch)
{
    const int vpb  = vols_per_block(wpb, var);
    long       need = ((long)batch + vpb - 1) / vpb;
    if (var == 5) {
        int sms = 0, occ = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern5_for(wpb, stcs), wpb * 32, 0);
        long wave = (long)occ * (sms > 0 ? sms : 108);
        if (wave >= 1 && need > wave) need = wave;
    }
    return (int)need;
}

extern "C" const char *fft3d_gpu_name(void) { return "L8_warpradix8"; }
extern "C" const char *fft3d_gpu_description(void)
{ return "volume per warp/pair/quad in registers, cross-lane DIF radix-8, measured autotune; async execute over an 8-stream ring with per-slot graph execs"; }
extern "C" int fft3d_gpu_supports(int L) { return L == 8; }

/* Measured candidate selection on scratch buffers -- pattern adopted from
 * L8_blockfused round gpu_r1. Round gpu_r4: candidates are launched round-robin
 * over the plan's stream ring, matching the production launch shape -- the r3
 * synchronous-world rankings do not survive the ring (L17_dmma gpu_r3's lesson),
 * so the tune must run through it. A device synchronize closes each sample,
 * exactly as the driver's timing boundary does. */
static float time_candidate(kfun_t k, int grid, int block,
                            const double2 *a, double2 *b, int B, int reps,
                            cudaStream_t *ring, int nring)
{
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    k<<<grid, block, 0, nring ? ring[0] : 0>>>(a, b, B); /* warm */
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r)
        k<<<grid, block, 0, nring ? ring[r % nring] : 0>>>(a, b, B);
    cudaDeviceSynchronize();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 1e30f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms;
}

/* The plan's single kernel launch, on an explicit stream: shared by the plain
 * path, the graph capture, and the plan-time graph-vs-plain measurement. */
static void launch_plan(const fft3d_gpu_plan *p, const double2 *in, double2 *out,
                        cudaStream_t s)
{
    if (p->var == -1)      fft8_b1<<<1, 64, 0, s>>>(in, out);
    else if (p->var == -3) fft8_quad<4, false, true><<<1, 128, 0, s>>>(in, out, 1);
    else if (p->var == -2) fft8_pair<2, false><<<1, 64, 0, s>>>(in, out, 1);
    else pick(p->wpb, p->var, p->stcs)<<<p->grid, p->wpb * 32, 0, s>>>(in, out, p->batch);
}

/* Capture the launch once, instantiate one exec per ring slot (a single exec's
 * relaunches serialize against each other -- L17_dmma gpu_r4), key on (in, out). */
static void build_graphs(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    const int slots = p->nring ? p->nring : 1;
    for (int i = 0; i < slots; ++i)
        if (p->gexec[i]) { cudaGraphExecDestroy(p->gexec[i]); p->gexec[i] = 0; }
    cudaStream_t cs;
    cudaGraph_t g;
    cudaStreamCreate(&cs);
    cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal);
    launch_plan(p, in, out, cs);
    cudaStreamEndCapture(cs, &g);
    for (int i = 0; i < slots; ++i)
        cudaGraphInstantiate(&p->gexec[i], g, 0, 0, 0);
    cudaGraphDestroy(g);
    cudaStreamDestroy(cs);
    p->gin = in;
    p->gout = out;
}

/* Time the production launch loop (graph or plain) through the ring; used for the
 * plan-time graph-vs-plain pick. Both paths are the same kernel with the same
 * arguments, so the pick cannot change a bit of the output. */
static float time_launch_loop(fft3d_gpu_plan *p, const double2 *in, double2 *out,
                              int graph, int reps)
{
    if (graph && (!p->gexec[0] || in != p->gin || out != p->gout))
        build_graphs(p, in, out);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    /* warm */
    if (graph) cudaGraphLaunch(p->gexec[0], p->nring ? p->ring[0] : 0);
    else       launch_plan(p, in, out, p->nring ? p->ring[0] : 0);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) {
        const int i = p->nring ? r % p->nring : 0;
        if (graph) cudaGraphLaunch(p->gexec[i], p->nring ? p->ring[i] : 0);
        else       launch_plan(p, in, out, p->nring ? p->ring[i] : 0);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 1e30f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms;
}

extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)
{
    if (L != 8 || batch < 1) return NULL;

    fft3d_gpu_plan *p = (fft3d_gpu_plan *)malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->wpb = 4;
    p->var = 2;
    p->stcs = 0;
    p->use_graph = 1; /* measured against plain launches below (both bit-identical);
                         L8WR_GRAPH forces. */
    p->nring = 8;     /* stream ring (L17_dmma gpu_r3): depth 8 saturates the host
                         launch rate, 16 measured no better there. L8WR_NSTREAM
                         overrides; 0 restores the synchronous r3 launch path. */
    p->gi = 0;
    p->gin = 0;
    p->gout = 0;
    for (int i = 0; i < L8WR_MAXRING; ++i) p->gexec[i] = 0;
    const char *fn = getenv("L8WR_NSTREAM");
    if (fn) p->nring = atoi(fn);
    if (p->nring < 0) p->nring = 0;
    if (p->nring > L8WR_MAXRING) p->nring = L8WR_MAXRING;
    for (int i = 0; i < p->nring; ++i)
        cudaStreamCreateWithFlags(&p->ring[i], cudaStreamNonBlocking);

    /* fft8_pair blocks want the full shared carveout so residency is register-
     * limited, not carveout-limited (hint pattern from L13_dmma/L8_blockfused). */
    cudaFuncSetAttribute(fft8_pair<2, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_pair<2, true>,  cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_pair<4, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_pair<4, true>,  cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_pair<6, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_pair<6, true>,  cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_pair<8, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_pair<8, true>,  cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_pair<10, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_pair<10, true>,  cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_pair_loop<2, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_pair_loop<2, true>,  cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_pair_loop<4, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_pair_loop<4, true>,  cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_pair_loop<8, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_pair_loop<8, true>,  cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_pair_t<false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_pair_t<true>,  cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_quad<4, false, true>,   cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_quad<4, true, true>,    cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_quad<4, false, false>,  cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_quad<4, true, false>,   cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_quad<8, false, true>,   cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_quad<8, true, true>,    cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_quad<8, false, false>,  cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_quad<8, true, false>,   cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_quad<16, false, true>,  cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_quad<16, true, true>,   cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_quad<16, false, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_quad<16, true, false>,  cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    const char *fw = getenv("L8WR_WPB");
    const char *ff = getenv("L8WR_VAR");
    const char *fs = getenv("L8WR_STCS");
    const char *fg = getenv("L8WR_GRAPH");
    if (fg) p->use_graph = atoi(fg);
    if (fw || ff || fs) {
        if (fw) p->wpb = atoi(fw);
        if (ff) p->var = atoi(ff);
        if (fs) p->stcs = atoi(fs);
        if (batch == 1 && !fw) p->wpb = 1;
    } else if (batch == 1) {
        /* Two candidate single-block kernels, both bit-identical to var 2:
         * fft8_pair<2> (64 threads, 1 barrier, r2 pick) and fft8_quad<4>
         * (128 threads = all 4 schedulers of the SM, half the per-thread
         * chain, 1 barrier + redundant partner recompute). Measured pick on
         * scratch, same discipline as the batched tune. */
        p->wpb = 1;
        p->var = -2;
        p->stcs = 0;
        double2 *a = 0, *b = 0;
        if (cudaMalloc(&a, 512 * sizeof(double2)) == cudaSuccess &&
            cudaMalloc(&b, 512 * sizeof(double2)) == cudaSuccess) {
            cudaMemset(a, 0, 512 * sizeof(double2));
            float t2 = 1e30f, t4 = 1e30f;
            for (int cyc = 0; cyc < 3; ++cyc) {
                float ms;
                ms = time_candidate(fft8_pair<2, false>, 1, 64, a, b, 1, 400,
                                    p->ring, p->nring);
                if (ms < t2) t2 = ms;
                ms = time_candidate(fft8_quad<4, false, true>, 1, 128, a, b, 1, 400,
                                    p->ring, p->nring);
                if (ms < t4) t4 = ms;
            }
            if (t4 < t2) p->var = -3;
            /* graph-vs-plain through the ring, measured (bit-identical either way) */
            if (!getenv("L8WR_GRAPH")) {
                float tg = 1e30f, tp = 1e30f;
                for (int cyc = 0; cyc < 3; ++cyc) {
                    float ms;
                    ms = time_launch_loop(p, a, b, 1, 400);
                    if (ms < tg) tg = ms;
                    ms = time_launch_loop(p, a, b, 0, 400);
                    if (ms < tp) tp = ms;
                }
                p->use_graph = (tg <= tp);
                for (int i = 0; i < L8WR_MAXRING; ++i)
                    if (p->gexec[i]) { cudaGraphExecDestroy(p->gexec[i]); p->gexec[i] = 0; }
                p->gin = 0;
                p->gout = 0;
            }
        }
        if (a) cudaFree(a);
        if (b) cudaFree(b);
        cudaGetLastError();
    } else {
        double2 *a = 0, *b = 0;
        size_t bytes = (size_t)batch * 512 * sizeof(double2);
        if (cudaMalloc(&a, bytes) == cudaSuccess && cudaMalloc(&b, bytes) == cudaSuccess) {
            cudaMemset(a, 0, bytes);
            /* Tuning space is vars 2, 3 and 4 ONLY: they share every arithmetic
             * slot exactly (var 4's bounce stage mirrors var 2's local y stage,
             * everything else is the same xstage code), so every candidate is
             * bit-identical and the plan choice cannot break run-to-run
             * repeatability. Vars 0/1 (dft8+transposes) round their z pass
             * differently and are kept for env-forced experiments only. */
            static const int CANDS[][3] = { /* {wpb, var, stcs} */
                {1, 2, 0},  {1, 2, 1},  {1, 3, 0},  {1, 3, 1},
                {2, 2, 0},  {2, 2, 1},  {2, 3, 0},  {2, 3, 1},
                {4, 2, 0},  {4, 2, 1},  {4, 3, 0},  {4, 3, 1},
                {8, 2, 0},  {8, 2, 1},  {8, 3, 0},  {8, 3, 1},
                {16, 2, 0}, {16, 2, 1}, {16, 3, 0}, {16, 3, 1},
                {2, 4, 0},  {2, 4, 1},  {4, 4, 0},  {4, 4, 1},
                {8, 4, 0},  {8, 4, 1},
                {4, 7, 0},  {4, 7, 1},  {8, 7, 0},  {8, 7, 1},
                {16, 7, 0}, {16, 7, 1},
                {4, 8, 0},  {4, 8, 1},  {8, 8, 0},  {8, 8, 1},
                /* var 5 (grid-stride) measured worse everywhere it was tried
                 * (15.5 vs 14.4 at B=2048, 1642 vs 1537 at B=131072: the two
                 * pair barriers per iteration serialize inter-volume MLP);
                 * env-forceable but excluded from the tune. */
            };
            const int NC = (int)(sizeof(CANDS) / sizeof(CANDS[0]));
            int reps = (int)(60000000L / ((long)batch * 512) + 3);
            if (reps > 400) reps = 400;
            if (reps < 6) reps = 6; /* >=6 even at the HBM batch: the top var-4
                                       candidates sit ~0.1% apart and 3-rep
                                       samples flip the pick under noise */
            float best[NC];
            for (int i = 0; i < NC; ++i) best[i] = 1e30f;
            for (int cyc = 0; cyc < 3; ++cyc)
                for (int i = 0; i < NC; ++i) {
                    const int wpb = CANDS[i][0], vr = CANDS[i][1], st = CANDS[i][2];
                    const int grid = grid_for(wpb, vr, st, batch);
                    float ms = time_candidate(pick(wpb, vr, st), grid, wpb * 32,
                                              a, b, batch, reps, p->ring, p->nring);
                    if (ms < best[i]) best[i] = ms;
                }
            int ibest = 0;
            for (int i = 1; i < NC; ++i)
                if (best[i] < best[ibest]) ibest = i;
            p->wpb = CANDS[ibest][0];
            p->var = CANDS[ibest][1];
            p->stcs = CANDS[ibest][2];

            /* Graph-vs-plain launch, measured on the chosen candidate through the
             * ring (same kernel, same arguments -> bit-identical either way).
             * L17_dmma gpu_r4 found graphs ~0.5% WORSE at kernel-bound batches
             * under a ring; my r3 found them 0.2-0.6 us BETTER at the latency
             * points. Measure, don't guess. L8WR_GRAPH forces. */
            if (!getenv("L8WR_GRAPH")) {
                p->grid = grid_for(p->wpb, p->var, p->stcs, batch);
                float tg = 1e30f, tp = 1e30f;
                for (int cyc = 0; cyc < 3; ++cyc) {
                    float ms;
                    ms = time_launch_loop(p, a, b, 1, reps);
                    if (ms < tg) tg = ms;
                    ms = time_launch_loop(p, a, b, 0, reps);
                    if (ms < tp) tp = ms;
                }
                p->use_graph = (tg <= tp);
                /* drop the scratch-keyed execs; real pointers rebuild lazily */
                for (int i = 0; i < L8WR_MAXRING; ++i)
                    if (p->gexec[i]) { cudaGraphExecDestroy(p->gexec[i]); p->gexec[i] = 0; }
                p->gin = 0;
                p->gout = 0;
            }
        }
        if (a) cudaFree(a);
        if (b) cudaFree(b);
        cudaGetLastError(); /* clear any autotune-time error state */
    }

    p->grid = grid_for(p->wpb, p->var, p->stcs, batch);
    if (getenv("L8WR_DEBUG"))
        fprintf(stderr, "L8WR plan: var=%d wpb=%d stcs=%d graph=%d grid=%d nring=%d\n",
                p->var, p->wpb, p->stcs, p->use_graph, p->grid, p->nring);
    return p;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    /* One kernel launch on the next ring stream, graph-replayed if the plan-time
     * measurement said so. No fencing between calls: overlapping calls write
     * identical bytes, and the driver's boundaries are host-synchronous (stream
     * ring from L17_dmma gpu_r3; per-slot graph execs from their gpu_r4). */
    cudaStream_t s = p->nring ? p->ring[p->gi] : (cudaStream_t)0;
    if (p->use_graph) {
        if (!p->gexec[0] || in != p->gin || out != p->gout)
            build_graphs(p, in, out);
        cudaGraphLaunch(p->gexec[p->gi], s);
    } else {
        launch_plan(p, in, out, s);
    }
    if (p->nring) p->gi = (p->gi + 1) % p->nring;
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p)
{
    if (!p) return;
    for (int i = 0; i < L8WR_MAXRING; ++i)
        if (p->gexec[i]) cudaGraphExecDestroy(p->gexec[i]);
    for (int i = 0; i < p->nring; ++i) cudaStreamDestroy(p->ring[i]);
    free(p);
}
