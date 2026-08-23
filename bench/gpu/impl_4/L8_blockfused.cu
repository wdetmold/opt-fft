/* L8_blockfused -- L = 8 on one A100.
 *
 * Technique (round gpu_r1): fused single kernel, V volumes of 8^3 per block staged in
 * shared memory, one thread per 8-point line (64 threads per volume), three axis passes
 * separated by barriers, one global read + one global write per volume.
 *
 * Round gpu_r2 additions:
 *   - the z->y barrier in the fused kernel is demoted to __syncwarp(): with t = tid&63,
 *     warp w holds t in [0,32) or [32,64) of ONE volume, the z-pass writes row t and the
 *     y-pass thread t reads rows 8*(t>>3)+0..7 -- both sides stay inside the same warp's
 *     half-volume, so a block-wide barrier there was never needed. Only the staging->z
 *     and y->x handoffs cross warps.
 *   - a warp-per-volume kernel (fft8_wpv: one warp owns a volume, two lines per thread,
 *     ZERO block barriers) and a persistent grid-stride variant (GS: one-wave grid) are
 *     in the measured autotune space; both LOSE at every scored point (see the strategy
 *     record) -- the autotuner keeps them out, and the env knobs allow re-A/B.
 *   - a single-warp B=1 kernel exists behind L8_FORCE_SINGLEWARP only (3.82 vs 3.20 us:
 *     serializing two lines per thread beats removing two barriers -- it lost).
 *
 * The shared z-row stride is padded 8 -> 9 complex doubles: gcd(8,8) = 8 makes the
 * unpadded z-pass a worst-case 8-way bank conflict (literature 09 6.2 -- "the single
 * most likely silent performance bug at this geometry"); an odd stride is conflict-free
 * for 16-byte accesses, and with the (x*8+y)*9+z layout all three passes and both
 * global-facing copies are conflict-free per 8-lane phase.
 *
 * The line transform is the minimum-operation-count radix-8 DIF codelet: the only
 * irrational twiddle is 1/sqrt(2), giving 4 real mults + 52 real adds per line
 * (the published minimum, same arithmetic as the CPU phase's L8_radix8 winner) --
 * ~10.5 flop/point/axis, invisible under the 32 B/point HBM floor.
 *
 * The x-axis pass reads shared and writes directly to global (fixed k: consecutive
 * threads hit consecutive complex doubles -- coalesced), saving one shared round trip.
 * Structure adopted from L13_dmma round gpu_r1 (staging pattern, direct final store,
 * full-carveout hint) with attribution in the strategy record.
 *
 * Round gpu_r4 additions:
 *   - every execute now replays its single kernel launch through a lazily-captured
 *     CUDA graph keyed on the (in,out) pointers -- adopted from L8_warpradix8 r3
 *     (who took it from L36_sharedtiled r1 / L45_pfa r2 and measured 0.2-0.6 us of
 *     launch path at the latency points). Bit-identical: same kernel, same args.
 *     L8_GRAPH=0 disables.
 *   - B=1 parity-split kernels (128 and 256 threads): thread groups share one line
 *     per pass and each computes an output-parity subset of the dft8 (verbatim
 *     subset expressions -> bit-identical to the full codelet), ping-ponging between
 *     two shared buffers so in-place halves never race. Measured pick in create()
 *     among {64t single, 128t half-split, 256t quarter-split}.
 *
 * Every autotune candidate runs the identical per-line arithmetic, so all picks are
 * bit-identical in output (repeatability lesson recorded by L8_warpradix8 r1).
 */
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../fft3d_gpu_api.h"

#define SROW 9    /* padded z-row stride in complex doubles (8 would be 8-way conflict) */
#define SVOL 576  /* 8 * 8 * SROW complex doubles staged per volume (9216 B)            */

static __device__ __forceinline__ double2 cadd(double2 a, double2 b)
{ return make_double2(a.x + b.x, a.y + b.y); }
static __device__ __forceinline__ double2 csub(double2 a, double2 b)
{ return make_double2(a.x - b.x, a.y - b.y); }

/* Forward 8-point DFT, decimation-in-frequency, natural order in and out.
 * 4 real multiplies (two twiddles by (1-i)/sqrt2 and -(1+i)/sqrt2) + 52 real adds.
 * STCS: use evict-first streaming stores (final global write, never re-read). */
template <int SS, int DS, bool STCS = false>
static __device__ __forceinline__ void dft8(const double2 *__restrict__ src,
                                            double2 *__restrict__ dst)
{
    const double R2 = 0.70710678118654752440; /* 1/sqrt(2) */
    const double2 a0 = src[0],      a1 = src[SS],     a2 = src[2 * SS], a3 = src[3 * SS],
                  a4 = src[4 * SS], a5 = src[5 * SS], a6 = src[6 * SS], a7 = src[7 * SS];
    /* stage 1: even half t_j = a_j + a_{j+4}; odd half s_j = a_j - a_{j+4}, twiddled */
    const double2 t0 = cadd(a0, a4), t1 = cadd(a1, a5), t2 = cadd(a2, a6), t3 = cadd(a3, a7);
    const double2 s0 = csub(a0, a4), s1 = csub(a1, a5), s2 = csub(a2, a6), s3 = csub(a3, a7);
    const double2 u1 = make_double2((s1.x + s1.y) * R2, (s1.y - s1.x) * R2);  /* * (1-i)/r2  */
    const double2 u2 = make_double2(s2.y, -s2.x);                             /* * (-i)      */
    const double2 u3 = make_double2((s3.y - s3.x) * R2, -(s3.x + s3.y) * R2); /* * -(1+i)/r2 */
    /* stage 2: two 4-point DIFs */
    const double2 e0 = cadd(t0, t2), e1 = cadd(t1, t3), f0 = csub(t0, t2);
    const double2 f1t = csub(t1, t3);
    const double2 f1 = make_double2(f1t.y, -f1t.x);
    const double2 g0 = cadd(s0, u2), g1 = cadd(u1, u3), h0 = csub(s0, u2);
    const double2 h1t = csub(u1, u3);
    const double2 h1 = make_double2(h1t.y, -h1t.x);
    /* stage 3, outputs in natural order */
    if (STCS) {
        __stcs(dst + 0,      cadd(e0, e1));
        __stcs(dst + 4 * DS, csub(e0, e1));
        __stcs(dst + 2 * DS, cadd(f0, f1));
        __stcs(dst + 6 * DS, csub(f0, f1));
        __stcs(dst + DS,     cadd(g0, g1));
        __stcs(dst + 5 * DS, csub(g0, g1));
        __stcs(dst + 3 * DS, cadd(h0, h1));
        __stcs(dst + 7 * DS, csub(h0, h1));
    } else {
        dst[0]      = cadd(e0, e1);
        dst[4 * DS] = csub(e0, e1);
        dst[2 * DS] = cadd(f0, f1);
        dst[6 * DS] = csub(f0, f1);
        dst[DS]     = cadd(g0, g1);
        dst[5 * DS] = csub(g0, g1);
        dst[3 * DS] = cadd(h0, h1);
        dst[7 * DS] = csub(h0, h1);
    }
}

/* V volumes per block, 64 threads per volume, dynamic shared V*SVOL complex doubles.
 * Shared element (v,x,y,z) at s[v*576 + (x*8+y)*9 + z]; every pass's 8-lane phase has
 * an odd (9) or unit lane stride, so all shared traffic is conflict-free.
 * Barriers: staging->z and y->x cross warps (__syncthreads); z->y is warp-local
 * (warp w's t-range [0,32) or [32,64): z writes rows t, y reads rows 8*(t>>3)+0..7,
 * both inside the warp's half-volume) so it is only a __syncwarp. */
template <int V, int MINB, bool STCS, bool LOADZ, bool CPA = false, bool GS = false>
__global__ void __launch_bounds__(V * 64, MINB)
fft8_fused(const double2 *__restrict__ in, double2 *__restrict__ out, int B)
{
    extern __shared__ double2 s[];
    const int  tid = threadIdx.x;
    const int  v   = tid >> 6;
    const int  t   = tid & 63;
    double2 *sv = s + v * SVOL;
    /* GS: persistent grid-stride loop over volume groups (grid sized to one wave in
     * create()) -- attacks wave quantisation at the L2 point. step pushes past B after
     * one iteration when GS is off. */
    const long step = GS ? (long)gridDim.x * V : (long)B + V;
    for (long vb = (long)blockIdx.x * V; vb < (long)B; vb += step) {
    const double2 *gin = in + vb * 512;
    const bool act = (vb + v) < (long)B;

    if (LOADZ) {
        /* z-pass fused into the load: each thread's z-line is a contiguous 128 B run
         * of global memory (sector-pair reads merge in L1). Saves one shared-memory
         * round trip of the volume (6 -> 4 shared accesses per point) and one barrier;
         * the cost is doubled L1 wavefronts on the read. Autotuned against the staged
         * path per batch -- at L=13 this variant lost 8% at B_HBM (their record). */
        if (act) dft8<1, 1>(gin + v * 512 + t * 8, sv + t * SROW);
    } else {
        if (vb + V <= (long)B) { /* full block: V*512 elements, unrolled, no bounds check */
            /* Flat block-interleaved staging, NOT warp-local: a warp-local layout
             * (warp w stages its own half-volume, load->z demoted to __syncwarp)
             * measured 1556-1561 vs 1537 us at B_HBM in gpu_r2 -- same 512 B/warp
             * sector efficiency, but the changed read ORDER costs ~1.3% at the DRAM
             * wall, and B_L2 was neutral. The block barrier stays. */
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int e = tid + i * (V * 64);
                const int r = e & 511;
                if (CPA)
                    __pipeline_memcpy_async(&s[(e >> 9) * SVOL + (r >> 3) * SROW + (r & 7)],
                                            &gin[e], sizeof(double2));
                else
                    s[(e >> 9) * SVOL + (r >> 3) * SROW + (r & 7)] = gin[e];
            }
            if (CPA) { __pipeline_commit(); __pipeline_wait_prior(0); }
        } else {                 /* tail block */
            const int n = (int)((long)B - vb) * 512;
            for (int e = tid; e < n; e += V * 64) {
                const int r = e & 511;
                s[(e >> 9) * SVOL + (r >> 3) * SROW + (r & 7)] = gin[e];
            }
        }
        __syncthreads();
        if (act) dft8<1, 1>(sv + t * SROW, sv + t * SROW);          /* z: stride 1  */
    }
    __syncwarp();                                                   /* z->y is warp-local */
    if (act) {                                                      /* y: stride 9  */
        double2 *p = sv + (t >> 3) * (8 * SROW) + (t & 7);
        dft8<SROW, SROW>(p, p);
    }
    __syncthreads();
    if (act)                                                        /* x: stride 72, write global */
        dft8<8 * SROW, 64, STCS>(sv + (t >> 3) * SROW + (t & 7),
                                 out + (vb + v) * 512 + t);
    if (GS) __syncthreads();    /* protect sv against the next iteration's staging */
    }
}

/* Warp-per-volume: W warps per block, each 32-thread warp owns one whole volume in its
 * own 9216 B shared slice and runs two 8-point lines per thread per pass. All exchange
 * is inside one warp, so there are NO block barriers -- __syncwarp only. Staging is 16
 * rounds of 512 B contiguous per warp (fully coalesced); the x pass writes straight to
 * global (two 512 B-contiguous store instructions per k), optionally evict-first.
 * Tail warps (vol >= B) exit as a whole warp; no guarded loads needed. */
template <int W, int MINB, bool STCS>
__global__ void __launch_bounds__(W * 32, MINB)
fft8_wpv(const double2 *__restrict__ in, double2 *__restrict__ out, int B)
{
    extern __shared__ double2 s[];
    const int  w    = threadIdx.x >> 5;
    const int  lane = threadIdx.x & 31;
    const long vol  = (long)blockIdx.x * W + w;
    if (vol >= (long)B) return;                    /* warp-uniform, no barriers below */
    const double2 *gv = in + vol * 512;
    double2 *sv = s + w * SVOL;

#pragma unroll
    for (int i = 0; i < 16; ++i) {                 /* stage: 512 B contiguous per round */
        const int e = i * 32 + lane;
        sv[(e >> 3) * SROW + (e & 7)] = gv[e];
    }
    __syncwarp();
    dft8<1, 1>(sv + lane * SROW, sv + lane * SROW);              /* z: rows lane, lane+32 */
    dft8<1, 1>(sv + (lane + 32) * SROW, sv + (lane + 32) * SROW);
    __syncwarp();
    {                                                            /* y: stride 9 */
        double2 *p = sv + (lane >> 3) * (8 * SROW) + (lane & 7);
        dft8<SROW, SROW>(p, p);
        double2 *q = sv + (((lane + 32) >> 3) & 7) * (8 * SROW) + (lane & 7);
        dft8<SROW, SROW>(q, q);
    }
    __syncwarp();
    dft8<8 * SROW, 64, STCS>(sv + (lane >> 3) * SROW + (lane & 7),   /* x: write global */
                             out + vol * 512 + lane);
    dft8<8 * SROW, 64, STCS>(sv + (((lane + 32) >> 3) & 7) * SROW + (lane & 7),
                             out + vol * 512 + lane + 32);
}

/* B = 1 latency kernel: one block, 64 threads, no staging pass. The z-pass reads each
 * thread's contiguous 128 B z-line straight from global (coalescing is irrelevant for
 * one 8 KB volume; the win is one less barrier and no staging round trip).
 * (The unused int parameter only unifies the signature for the create()-time timer.) */
__global__ void __launch_bounds__(64)
fft8_single(const double2 *__restrict__ in, double2 *__restrict__ out, int)
{
    __shared__ double2 s[SVOL];
    const int t = threadIdx.x;
    dft8<1, 1>(in + t * 8, s + t * SROW);                           /* z from global   */
    __syncthreads();
    {
        double2 *p = s + (t >> 3) * (8 * SROW) + (t & 7);
        dft8<SROW, SROW>(p, p);                                     /* y               */
    }
    __syncthreads();
    dft8<8 * SROW, 64>(s + (t >> 3) * SROW + (t & 7), out + t);     /* x, write global */
}

/* B = 1, single-warp variant: 32 threads, two lines per thread per pass, no block
 * barriers at all (__syncwarp only). Measured against fft8_single in create(). */
__global__ void __launch_bounds__(32)
fft8_single_warp(const double2 *__restrict__ in, double2 *__restrict__ out, int)
{
    __shared__ double2 s[SVOL];
    const int t = threadIdx.x;
    dft8<1, 1>(in + t * 8, s + t * SROW);                           /* z from global   */
    dft8<1, 1>(in + (t + 32) * 8, s + (t + 32) * SROW);
    __syncwarp();
    {
        double2 *p = s + (t >> 3) * (8 * SROW) + (t & 7);
        dft8<SROW, SROW>(p, p);                                     /* y               */
        double2 *q = s + (((t + 32) >> 3) & 7) * (8 * SROW) + (t & 7);
        dft8<SROW, SROW>(q, q);
    }
    __syncwarp();
    dft8<8 * SROW, 64>(s + (t >> 3) * SROW + (t & 7), out + t);     /* x, write global */
    dft8<8 * SROW, 64>(s + (((t + 32) >> 3) & 7) * SROW + (t & 7), out + t + 32);
}

/* Half of dft8, split by OUTPUT parity: EVEN computes k = 0,4,2,6 (sums only, no
 * irrational twiddle), !EVEN computes k = 1,5,3,7 (all 4 real mults). Both halves
 * read all 8 inputs; the expressions are verbatim subsets of dft8 in the same
 * operand order, so a pair of halves is bit-identical to one full dft8. */
template <bool EVEN, int SS, int DS>
static __device__ __forceinline__ void dft8_half(const double2 *__restrict__ src,
                                                 double2 *__restrict__ dst)
{
    const double R2 = 0.70710678118654752440; /* 1/sqrt(2) */
    const double2 a0 = src[0],      a1 = src[SS],     a2 = src[2 * SS], a3 = src[3 * SS],
                  a4 = src[4 * SS], a5 = src[5 * SS], a6 = src[6 * SS], a7 = src[7 * SS];
    if (EVEN) {
        const double2 t0 = cadd(a0, a4), t1 = cadd(a1, a5), t2 = cadd(a2, a6), t3 = cadd(a3, a7);
        const double2 e0 = cadd(t0, t2), e1 = cadd(t1, t3), f0 = csub(t0, t2);
        const double2 f1t = csub(t1, t3);
        const double2 f1 = make_double2(f1t.y, -f1t.x);
        dst[0]      = cadd(e0, e1);
        dst[4 * DS] = csub(e0, e1);
        dst[2 * DS] = cadd(f0, f1);
        dst[6 * DS] = csub(f0, f1);
    } else {
        const double2 s0 = csub(a0, a4), s1 = csub(a1, a5), s2 = csub(a2, a6), s3 = csub(a3, a7);
        const double2 u1 = make_double2((s1.x + s1.y) * R2, (s1.y - s1.x) * R2);  /* * (1-i)/r2  */
        const double2 u2 = make_double2(s2.y, -s2.x);                             /* * (-i)      */
        const double2 u3 = make_double2((s3.y - s3.x) * R2, -(s3.x + s3.y) * R2); /* * -(1+i)/r2 */
        const double2 g0 = cadd(s0, u2), g1 = cadd(u1, u3), h0 = csub(s0, u2);
        const double2 h1t = csub(u1, u3);
        const double2 h1 = make_double2(h1t.y, -h1t.x);
        dst[DS]     = cadd(g0, g1);
        dst[5 * DS] = csub(g0, g1);
        dst[3 * DS] = cadd(h0, h1);
        dst[7 * DS] = csub(h0, h1);
    }
}

/* Quarter of dft8, by output pair: Q=0 -> k {0,4}, Q=1 -> {2,6}, Q=2 -> {1,5},
 * Q=3 -> {3,7}. Same verbatim-subset property as dft8_half. */
template <int Q, int SS, int DS>
static __device__ __forceinline__ void dft8_quarter(const double2 *__restrict__ src,
                                                    double2 *__restrict__ dst)
{
    const double R2 = 0.70710678118654752440;
    const double2 a0 = src[0],      a1 = src[SS],     a2 = src[2 * SS], a3 = src[3 * SS],
                  a4 = src[4 * SS], a5 = src[5 * SS], a6 = src[6 * SS], a7 = src[7 * SS];
    if (Q == 0) {
        const double2 t0 = cadd(a0, a4), t1 = cadd(a1, a5), t2 = cadd(a2, a6), t3 = cadd(a3, a7);
        const double2 e0 = cadd(t0, t2), e1 = cadd(t1, t3);
        dst[0]      = cadd(e0, e1);
        dst[4 * DS] = csub(e0, e1);
    } else if (Q == 1) {
        const double2 t0 = cadd(a0, a4), t1 = cadd(a1, a5), t2 = cadd(a2, a6), t3 = cadd(a3, a7);
        const double2 f0 = csub(t0, t2);
        const double2 f1t = csub(t1, t3);
        const double2 f1 = make_double2(f1t.y, -f1t.x);
        dst[2 * DS] = cadd(f0, f1);
        dst[6 * DS] = csub(f0, f1);
    } else if (Q == 2) {
        const double2 s0 = csub(a0, a4), s1 = csub(a1, a5), s2 = csub(a2, a6), s3 = csub(a3, a7);
        const double2 u1 = make_double2((s1.x + s1.y) * R2, (s1.y - s1.x) * R2);
        const double2 u2 = make_double2(s2.y, -s2.x);
        const double2 u3 = make_double2((s3.y - s3.x) * R2, -(s3.x + s3.y) * R2);
        const double2 g0 = cadd(s0, u2), g1 = cadd(u1, u3);
        dst[DS]     = cadd(g0, g1);
        dst[5 * DS] = csub(g0, g1);
    } else {
        const double2 s0 = csub(a0, a4), s1 = csub(a1, a5), s2 = csub(a2, a6), s3 = csub(a3, a7);
        const double2 u1 = make_double2((s1.x + s1.y) * R2, (s1.y - s1.x) * R2);
        const double2 u2 = make_double2(s2.y, -s2.x);
        const double2 u3 = make_double2((s3.y - s3.x) * R2, -(s3.x + s3.y) * R2);
        const double2 h0 = csub(s0, u2);
        const double2 h1t = csub(u1, u3);
        const double2 h1 = make_double2(h1t.y, -h1t.x);
        dst[3 * DS] = cadd(h0, h1);
        dst[7 * DS] = csub(h0, h1);
    }
}

/* B = 1 half-split kernel: 128 threads, pair (t, t+64) shares one line per pass,
 * each computing one output-parity half of the dft8. Both halves of a pair read the
 * same 8 inputs (the duplicated global z-line loads hit L1 on an 8 KB volume);
 * passes ping-pong sa -> sb so an in-place half can never race its cross-warp
 * partner. Same 2 barriers as fft8_single at roughly half the per-thread dependent
 * chain -- the structure L8_warpradix8 r3 measured to win at B=1 (their 128-thread
 * quad), rebuilt in thread-per-line shared-memory form. */
__global__ void __launch_bounds__(128)
fft8_single_split(const double2 *__restrict__ in, double2 *__restrict__ out, int)
{
    __shared__ double2 sa[SVOL], sb[SVOL];
    const int  t  = threadIdx.x & 63;
    const bool ev = threadIdx.x < 64;
    if (ev) dft8_half<true, 1, 1>(in + t * 8, sa + t * SROW);       /* z from global */
    else    dft8_half<false, 1, 1>(in + t * 8, sa + t * SROW);
    __syncthreads();
    {
        const double2 *ys = sa + (t >> 3) * (8 * SROW) + (t & 7);   /* y: stride 9   */
        double2       *yd = sb + (t >> 3) * (8 * SROW) + (t & 7);
        if (ev) dft8_half<true, SROW, SROW>(ys, yd);
        else    dft8_half<false, SROW, SROW>(ys, yd);
    }
    __syncthreads();
    if (ev) dft8_half<true, 8 * SROW, 64>(sb + (t >> 3) * SROW + (t & 7),   /* x -> global */
                                          out + t);
    else    dft8_half<false, 8 * SROW, 64>(sb + (t >> 3) * SROW + (t & 7),
                                           out + t);
}

/* B = 1 quarter-split kernel: 256 threads (8 warps, one block), quad (t + 64q)
 * shares one line per pass, each computing one output pair. Branch is warp-uniform
 * (q constant across a warp). */
__global__ void __launch_bounds__(256)
fft8_single_split4(const double2 *__restrict__ in, double2 *__restrict__ out, int)
{
    __shared__ double2 sa[SVOL], sb[SVOL];
    const int t = threadIdx.x & 63;
    const int q = threadIdx.x >> 6;
    switch (q) {                                                    /* z from global */
    case 0:  dft8_quarter<0, 1, 1>(in + t * 8, sa + t * SROW); break;
    case 1:  dft8_quarter<1, 1, 1>(in + t * 8, sa + t * SROW); break;
    case 2:  dft8_quarter<2, 1, 1>(in + t * 8, sa + t * SROW); break;
    default: dft8_quarter<3, 1, 1>(in + t * 8, sa + t * SROW); break;
    }
    __syncthreads();
    {
        const double2 *ys = sa + (t >> 3) * (8 * SROW) + (t & 7);   /* y: stride 9   */
        double2       *yd = sb + (t >> 3) * (8 * SROW) + (t & 7);
        switch (q) {
        case 0:  dft8_quarter<0, SROW, SROW>(ys, yd); break;
        case 1:  dft8_quarter<1, SROW, SROW>(ys, yd); break;
        case 2:  dft8_quarter<2, SROW, SROW>(ys, yd); break;
        default: dft8_quarter<3, SROW, SROW>(ys, yd); break;
        }
    }
    __syncthreads();
    {
        const double2 *xs = sb + (t >> 3) * SROW + (t & 7);         /* x -> global   */
        switch (q) {
        case 0:  dft8_quarter<0, 8 * SROW, 64>(xs, out + t); break;
        case 1:  dft8_quarter<1, 8 * SROW, 64>(xs, out + t); break;
        case 2:  dft8_quarter<2, 8 * SROW, 64>(xs, out + t); break;
        default: dft8_quarter<3, 8 * SROW, 64>(xs, out + t); break;
        }
    }
}

/* ---- host side ------------------------------------------------------------------- */

struct fft3d_gpu_plan {
    int L, batch;
    int kind;   /* 0 = fused, 1 = single (64t), 2 = single-warp (32t), 3 = warp-per-volume,
                   4 = persistent grid-stride, 5 = split128 (B=1), 6 = split256 (B=1) */
    int vsel;   /* V for fused, W for wpv */
    int stcs;
    int mode;
    int grid;
    int block;
    int smem;
    int graph;                  /* replay the single launch via a captured CUDA graph */
    cudaGraphExec_t gexec;      /* lazily captured, keyed on (gin, gout) */
    const double2 *gin;
    double2 *gout;
};

typedef void (*kfun_t)(const double2 *, double2 *, int);

template <bool STCS, bool LOADZ, bool CPA>
static kfun_t kern_v(int V)
{
    switch (V) {
    case 1: return fft8_fused<1, 16, STCS, LOADZ, CPA>;
    case 2: return fft8_fused<2, 8, STCS, LOADZ, CPA>;
    case 4: return fft8_fused<4, 4, STCS, LOADZ, CPA>;
    case 8: return fft8_fused<8, 2, STCS, LOADZ, CPA>;
    }
    return 0;
}

/* mode: 0 = staged, 1 = staged with cp.async, 2 = z-pass fused into the load
 * (mode 2 lost at both scored batch points -- kept for env-forced experiments only) */
static kfun_t kern_for(int V, int stcs, int mode)
{
    if (stcs) {
        if (mode == 2) return kern_v<true, true, false>(V);
        return mode ? kern_v<true, false, true>(V) : kern_v<true, false, false>(V);
    }
    if (mode == 2) return kern_v<false, true, false>(V);
    return mode ? kern_v<false, false, true>(V) : kern_v<false, false, false>(V);
}

template <bool STCS>
static kfun_t gs_v(int V)
{
    switch (V) {
    case 1: return fft8_fused<1, 16, STCS, false, false, true>;
    case 2: return fft8_fused<2, 8, STCS, false, false, true>;
    case 4: return fft8_fused<4, 4, STCS, false, false, true>;
    case 8: return fft8_fused<8, 2, STCS, false, false, true>;
    }
    return 0;
}

static kfun_t gs_for(int V, int stcs)
{ return stcs ? gs_v<true>(V) : gs_v<false>(V); }

/* one-wave grid for the persistent kernel: co-resident blocks x SMs, capped at need */
static int gs_grid(kfun_t k, int V, int batch)
{
    int nb = 0, sm = 108;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, k, V * 64,
                                                  V * SVOL * (int)sizeof(double2));
    cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0);
    long need = ((long)batch + V - 1) / V;
    long grid = (long)nb * sm;
    if (grid > need || grid < 1) grid = need;
    return (int)grid;
}

template <bool STCS>
static kfun_t wpv_v(int W)
{
    switch (W) {
    case 1: return fft8_wpv<1, 16, STCS>;
    case 2: return fft8_wpv<2, 8, STCS>;
    case 4: return fft8_wpv<4, 4, STCS>;
    case 8: return fft8_wpv<8, 2, STCS>;
    }
    return 0;
}

static kfun_t wpv_for(int W, int stcs)
{ return stcs ? wpv_v<true>(W) : wpv_v<false>(W); }

extern "C" const char *fft3d_gpu_name(void) { return "L8_blockfused"; }
extern "C" const char *fft3d_gpu_description(void)
{ return "fused block of V volumes in shared (row pad 8->9), min-op radix-8 per thread-line, autotuned; graph-replayed single launch; parity-split B=1 kernel"; }
extern "C" int fft3d_gpu_supports(int L) { return L == 8; }

static float time_candidate(kfun_t k, int grid, int block, int smem,
                            const double2 *a, double2 *b, int B, int reps)
{
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    k<<<grid, block, smem>>>(a, b, B); /* warm */
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) k<<<grid, block, smem>>>(a, b, B);
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

    static const int VS[4] = {1, 2, 4, 8};
    for (int i = 0; i < 4; ++i) {
        for (int st = 0; st < 2; ++st) {
            for (int m = 0; m < 3; ++m) {
                kfun_t k = kern_for(VS[i], st, m);
                cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     VS[i] * SVOL * (int)sizeof(double2));
                cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
            }
            kfun_t k = wpv_for(VS[i], st);
            cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 VS[i] * SVOL * (int)sizeof(double2));
            cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
            k = gs_for(VS[i], st);
            cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 VS[i] * SVOL * (int)sizeof(double2));
            cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        }
    }
    cudaFuncSetAttribute(fft8_single, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_single_warp, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_single_split, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(fft8_single_split4, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    fft3d_gpu_plan *p = (fft3d_gpu_plan *)malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->kind = 0;
    p->vsel = 4;
    p->stcs = 0;
    p->mode = 0;
    p->graph = 1;   /* single-launch graph replay (L8_warpradix8 r3): L8_GRAPH=0 off */
    p->gexec = 0;
    p->gin = 0;
    p->gout = 0;
    { const char *gr = getenv("L8_GRAPH"); if (gr) p->graph = atoi(gr); }

    const char *fs = getenv("L8_FORCE_SINGLE");
    const char *fw = getenv("L8_FORCE_SINGLEWARP");
    const char *f1 = getenv("L8_FORCE_SPLIT1");
    const char *fv = getenv("L8_FORCE_V");
    const char *fp = getenv("L8_FORCE_WPV");
    const char *fg = getenv("L8_FORCE_GS");
    const char *ft = getenv("L8_FORCE_STCS");
    const char *fz = getenv("L8_FORCE_LOADZ");
    const char *fc = getenv("L8_FORCE_CPA");
    if (ft) p->stcs = atoi(ft);
    if (fz && atoi(fz)) p->mode = 2;
    if (fc && atoi(fc)) p->mode = 1;
    if (fs && atoi(fs)) p->kind = 1;
    else if (fw && atoi(fw)) p->kind = 2;
    else if (f1 && atoi(f1) > 0) p->kind = (atoi(f1) >= 2) ? 6 : 5;
    else if (fp && atoi(fp) > 0) { p->kind = 3; p->vsel = atoi(fp); }
    else if (fg && atoi(fg) > 0) { p->kind = 4; p->vsel = atoi(fg); }
    else if (fv && atoi(fv) > 0) { p->kind = 0; p->vsel = atoi(fv); }
    else if (batch == 1) {
        /* Measured pick among the three B=1 latency kernels (all bit-identical in
         * output: the split codelets are verbatim subsets of dft8). The 32-thread
         * single-warp variant stays out (3.82 vs 3.20 us, gpu_r2). */
        double2 *a = 0, *b = 0;
        p->kind = 1;
        if (cudaMalloc(&a, 512 * sizeof(double2)) == cudaSuccess &&
            cudaMalloc(&b, 512 * sizeof(double2)) == cudaSuccess) {
            cudaMemset(a, 0, 512 * sizeof(double2));
            const int reps = 2000; /* launch-bound ~3 us kernels: ~6 ms per timing */
            float best[3] = {1e30f, 1e30f, 1e30f};
            for (int cyc = 0; cyc < 3; ++cyc) {
                float ms;
                ms = time_candidate(fft8_single, 1, 64, 0, a, b, 1, reps);
                if (ms < best[0]) best[0] = ms;
                ms = time_candidate(fft8_single_split, 1, 128, 0, a, b, 1, reps);
                if (ms < best[1]) best[1] = ms;
                ms = time_candidate(fft8_single_split4, 1, 256, 0, a, b, 1, reps);
                if (ms < best[2]) best[2] = ms;
            }
            int ib = 0;
            if (best[1] < best[ib]) ib = 1;
            if (best[2] < best[ib]) ib = 2;
            p->kind = (ib == 0) ? 1 : (ib == 1) ? 5 : 6;
        }
        if (a) cudaFree(a);
        if (b) cudaFree(b);
        cudaGetLastError();
    } else {
        /* measured selection on scratch buffers (excluded from execute) over
         * fused (V, stcs, cp.async) -- 16 combos -- plus warp-per-volume (W, stcs) --
         * 8 combos. The loadz mode is not tuned over: it measured 5% worse at B_HBM
         * and 33% worse at B_L2 than the best staged variant (round gpu_r1). */
        double2 *a = 0, *b = 0;
        size_t bytes = (size_t)batch * 512 * sizeof(double2);
        if (cudaMalloc(&a, bytes) == cudaSuccess && cudaMalloc(&b, bytes) == cudaSuccess) {
            cudaMemset(a, 0, bytes);
            /* enough reps that one timing is a few ms even at large B; 3 round-robin
             * cycles so clock ramp does not bias the first candidate; keep the min */
            int reps = (int)(60000000L / ((long)batch * 512) + 3);
            if (reps > 400) reps = 400;
            float best[32];
            for (int i = 0; i < 32; ++i) best[i] = 1e30f;
            for (int cyc = 0; cyc < 3; ++cyc)
                for (int i = 0; i < 32; ++i) {
                    float ms;
                    const int n = VS[i & 3];
                    const int st = (i >> 2) & 1;
                    if (i < 16) {
                        const int m = i >> 3;
                        const int grid = (batch + n - 1) / n;
                        ms = time_candidate(kern_for(n, st, m), grid, n * 64,
                                            n * SVOL * (int)sizeof(double2), a, b,
                                            batch, reps);
                    } else if (i < 24) {   /* warp-per-volume */
                        const int grid = (batch + n - 1) / n;
                        ms = time_candidate(wpv_for(n, st), grid, n * 32,
                                            n * SVOL * (int)sizeof(double2), a, b,
                                            batch, reps);
                    } else {               /* persistent grid-stride */
                        kfun_t k = gs_for(n, st);
                        ms = time_candidate(k, gs_grid(k, n, batch), n * 64,
                                            n * SVOL * (int)sizeof(double2), a, b,
                                            batch, reps);
                    }
                    if (ms < best[i]) best[i] = ms;
                }
            int ibest = 0;
            for (int i = 1; i < 32; ++i)
                if (best[i] < best[ibest]) ibest = i;
            p->vsel = VS[ibest & 3];
            p->stcs = (ibest >> 2) & 1;
            if (ibest < 16)      { p->kind = 0; p->mode = ibest >> 3; }
            else if (ibest < 24) { p->kind = 3; p->mode = 0; }
            else                 { p->kind = 4; p->mode = 0; }
        }
        if (a) cudaFree(a);
        if (b) cudaFree(b);
        cudaGetLastError(); /* clear any autotune-time error state */
    }

    if (p->kind == 1)      { p->grid = 1; p->block = 64; p->smem = 0; }
    else if (p->kind == 2) { p->grid = 1; p->block = 32; p->smem = 0; }
    else if (p->kind == 5) { p->grid = 1; p->block = 128; p->smem = 0; }
    else if (p->kind == 6) { p->grid = 1; p->block = 256; p->smem = 0; }
    else if (p->kind == 3) {
        p->grid = (batch + p->vsel - 1) / p->vsel;
        p->block = p->vsel * 32;
        p->smem = p->vsel * SVOL * (int)sizeof(double2);
    } else if (p->kind == 4) {
        p->grid = gs_grid(gs_for(p->vsel, p->stcs), p->vsel, batch);
        p->block = p->vsel * 64;
        p->smem = p->vsel * SVOL * (int)sizeof(double2);
    } else {
        p->grid = (batch + p->vsel - 1) / p->vsel;
        p->block = p->vsel * 64;
        p->smem = p->vsel * SVOL * (int)sizeof(double2);
    }
    if (getenv("L8_DEBUG"))
        fprintf(stderr, "L8_blockfused plan: kind=%d V/W=%d stcs=%d mode=%d grid=%d block=%d\n",
                p->kind, p->vsel, p->stcs, p->mode, p->grid, p->block);
    return p;
}

static void launch_raw(fft3d_gpu_plan *p, const double2 *in, double2 *out, cudaStream_t st)
{
    if (p->kind == 1)      fft8_single<<<1, 64, 0, st>>>(in, out, 1);
    else if (p->kind == 2) fft8_single_warp<<<1, 32, 0, st>>>(in, out, 1);
    else if (p->kind == 5) fft8_single_split<<<1, 128, 0, st>>>(in, out, 1);
    else if (p->kind == 6) fft8_single_split4<<<1, 256, 0, st>>>(in, out, 1);
    else if (p->kind == 3)
        wpv_for(p->vsel, p->stcs)<<<p->grid, p->block, p->smem, st>>>(in, out, p->batch);
    else if (p->kind == 4)
        gs_for(p->vsel, p->stcs)<<<p->grid, p->block, p->smem, st>>>(in, out, p->batch);
    else
        kern_for(p->vsel, p->stcs, p->mode)<<<p->grid, p->block, p->smem, st>>>(in, out,
                                                                                p->batch);
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (p->graph) {
        /* Every plan is one kernel launch; replay it as a single-node CUDA graph,
         * lazily captured and keyed on the pointers (adopted from L8_warpradix8 r3,
         * originally L36_sharedtiled r1 / L45_pfa r2). Same kernel, same arguments
         * -> bit-identical output. */
        if (!p->gexec || in != p->gin || out != p->gout) {
            if (p->gexec) { cudaGraphExecDestroy(p->gexec); p->gexec = 0; }
            cudaStream_t cs;
            cudaGraph_t g;
            cudaStreamCreate(&cs);
            cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal);
            launch_raw(p, in, out, cs);
            cudaStreamEndCapture(cs, &g);
            cudaGraphInstantiate(&p->gexec, g, 0, 0, 0);
            cudaGraphDestroy(g);
            cudaStreamDestroy(cs);
            p->gin = in;
            p->gout = out;
        }
        cudaGraphLaunch(p->gexec, 0);
        return;
    }
    launch_raw(p, in, out, 0);
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p)
{
    if (p && p->gexec) cudaGraphExecDestroy(p->gexec);
    free(p);
}
