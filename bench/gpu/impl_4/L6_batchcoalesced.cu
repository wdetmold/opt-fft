/* L6_batchcoalesced -- L = 6 on one A100, BATCH-MAJOR, PERFECTLY COALESCED.
 *
 * Strategy (round gpu_r1): the batch is the vector lane. Each block stages V = 8
 * consecutive volumes into shared memory in a BATCH-MAJOR layout s[point][volume]
 * (with an XOR-free additive swizzle on the volume slot), so that during every axis
 * pass the 8 lanes working on the same 6-point line of 8 different volumes touch
 * 8 consecutive complex doubles = one 128-byte shared transaction. No axis is
 * "strided" in a bank sense, because the batch index is always the fast dimension;
 * the 3D transpose problem disappears instead of being fought.
 *
 *   phase 0  load:   coalesced global read (consecutive threads, consecutive
 *                    elements of the block's contiguous V*216-element chunk),
 *                    swizzled batch-major shared write (4-way conflict, paid once)
 *   phase 1  z-axis: stride 1,  lanes batch-fast, conflict-free read+write
 *   phase 2  y-axis: stride 6,  lanes batch-fast, conflict-free read+write
 *   phase 3  x-axis: stride 36, lanes line-fast so the RESULT goes straight to
 *                    global memory coalesced (for fixed output plane j, consecutive
 *                    threads write consecutive complex doubles) -- saves the whole
 *                    store phase and one __syncthreads (borrowed from L13_dmma's
 *                    "x-pass writes directly to global" and lit. 09 Section 9.1)
 *
 * One global read + one global write per volume; 6-point lines via a split-radix-free
 * DIT 6 = 2x3 codelet (two DFT-3s + two twiddle rotations), arithmetic invisible
 * under the bandwidth floor (lit. 09 Section 2.2: L=6 is 5.15x bandwidth-bound).
 *
 * Round gpu_r2: plan-time choice of evict-first streaming stores (__stcs) for the
 * x-pass when the input buffer fits in L2, so the write stream cannot evict in's
 * residency across the benchmark repeat loop (borrowed from L6_warpvolume gpu_r1,
 * confirmed by L8_warpradix8 gpu_r1: the sign flips between the L2 and HBM points,
 * so it is selected in create(), never hardcoded). B=4854: 24.0 -> 14.8 us.
 *
 * Round gpu_r3: two additions.
 *   - B = 1 dedicated kernel fft6_single: one 36-thread block, z-pass fused into the
 *     global load (each thread reads its own contiguous 96 B z-line -- fine at B=1
 *     where latency, not sector economy, rules), padded shared, TWO barriers, x-pass
 *     stores direct. Shape borrowed from L6_warpvolume gpu_r2's fft6_single and
 *     L8_blockfused gpu_r1's staging-free B=1 kernel (which measured the fused-z
 *     load the winner at B=1 even though it loses batched). 3.69 -> ~2.9 us.
 *   - z->y named-barrier demotion (bar.sync id,96 over 3 warps; the dependence
 *     closes within aligned 48-tid groups) was built and MEASURED A LOSS both at
 *     B_L2 (14.97 vs 14.76 us interleaved same-lease) and B_HBM (1543.7 vs 1541.0):
 *     the barrier-unit named path costs more than waking 6 fewer warps saves.
 *     Kept behind -DNAMED_BAR=1, default off. Do not rediscover.
 *
 * Round gpu_r4: every execute replays its single kernel launch through a lazily
 * captured CUDA graph keyed on the (in,out) pointers (taken whole from L8_warpradix8
 * gpu_r3, originally L36_sharedtiled r1 / L45_pfa r2). Rotated same-lease A/B:
 * B=1 2.62 vs 2.96 us, B_L2 14.36-14.47 vs 14.78-14.89, B_HBM a wash with graph
 * marginally ahead -- so on everywhere (L6BC_GRAPH=0 disables). A 216-thread
 * one-point-per-thread B=1 kernel (fft6_single216, idea from L8's quad-at-B=1)
 * measured 3.24-3.39 vs 2.63-2.87 us: kept behind L6BC_B1=1, default off.
 *
 * Block = 288 threads (8 volumes x 36 lines), static shared 216*8*16 = 27,648 B.
 */
#include <cuda_runtime.h>
#include <stdlib.h>

#include "../fft3d_gpu_api.h"

#define NP     216   /* 6^3 points per volume  */
#define NLINES 36    /* 6^2 lines per axis     */
#define VPB    8     /* volumes per block (power of 2: swizzle uses & (VPB-1)) */

#ifndef NAMED_BAR    /* -DNAMED_BAR=1 re-enables the z->y named-barrier demotion;
                        measured a loss at every scored point (see header) */
#define NAMED_BAR 0
#endif

/* forward DFT-3: X = F3 * (a,b,c), F3 built from w3 = exp(-2*pi*i/3) */
static __device__ __forceinline__ void dft3(double2 a, double2 b, double2 c,
                                            double2 &X0, double2 &X1, double2 &X2)
{
    const double S3 = 0.86602540378443864676;  /* sqrt(3)/2 */
    double tr = b.x + c.x, ti = b.y + c.y;
    double ur = a.x - 0.5 * tr, ui = a.y - 0.5 * ti;
    double dr = S3 * (b.x - c.x), di = S3 * (b.y - c.y);
    X0.x = a.x + tr; X0.y = a.y + ti;
    X1.x = ur + di;  X1.y = ui - dr;
    X2.x = ur - di;  X2.y = ui + dr;
}

/* forward DFT-6 in place: DIT even/odd split into two DFT-3s.
 * X[k]   = E[k mod 3] + w6^k * O[k mod 3]
 * X[k+3] = E[k mod 3] - w6^k * O[k mod 3],  w6 = exp(-2*pi*i/6) = (1/2, -sqrt3/2) */
static __device__ __forceinline__ void dft6(double2 r[6])
{
    const double S3 = 0.86602540378443864676;
    double2 e0, e1, e2, o0, o1, o2;
    dft3(r[0], r[2], r[4], e0, e1, e2);
    dft3(r[1], r[3], r[5], o0, o1, o2);
    double2 w1, w2;                              /* o1*w6, o2*w6^2 */
    w1.x =  0.5 * o1.x + S3 * o1.y;  w1.y =  0.5 * o1.y - S3 * o1.x;
    w2.x = -0.5 * o2.x + S3 * o2.y;  w2.y = -0.5 * o2.y - S3 * o2.x;
    r[0].x = e0.x + o0.x; r[0].y = e0.y + o0.y;
    r[3].x = e0.x - o0.x; r[3].y = e0.y - o0.y;
    r[1].x = e1.x + w1.x; r[1].y = e1.y + w1.y;
    r[4].x = e1.x - w1.x; r[4].y = e1.y - w1.y;
    r[2].x = e2.x + w2.x; r[2].y = e2.y + w2.y;
    r[5].x = e2.x - w2.x; r[5].y = e2.y - w2.y;
}

/* shared slot for (point i, volume v): batch-major with an additive swizzle so the
 * load phase's strided writes spread over banks while every axis pass's lane group
 * (fixed i, consecutive v) still lands inside one aligned 128-byte block. */
static __device__ __forceinline__ int slot(int i, int v)
{
    return i * VPB + ((v + i) & (VPB - 1));
}

template <bool STREAM_ST>
__global__ void
#if defined(MINB) && MINB > 0   /* A/B: force MINB blocks/SM (costs spills) */
__launch_bounds__(VPB * NLINES, MINB)
#else
__launch_bounds__(VPB * NLINES)
#endif
fft6_batch(const double2 *__restrict__ in, double2 *__restrict__ out, int nvol)
{
    __shared__ double2 s[NP * VPB];
    const int tid = (int)threadIdx.x;
    const int T   = VPB * NLINES;
    const long long v0 = (long long)blockIdx.x * VPB;
    int vrem = nvol - (int)v0;
    if (vrem > VPB) vrem = VPB;
    const double2 *gin  = in  + v0 * NP;
    double2       *gout = out + v0 * NP;

    /* ---- load: consecutive threads read consecutive global elements ---- */
    if (vrem == VPB) {
        /* full block: NP*VPB = 1728 = 6*T exactly; unroll so 6 independent
           16-byte loads are in flight per thread */
#pragma unroll
        for (int c = 0; c < NP * VPB / (VPB * NLINES); ++c) {
            int e = c * T + tid;
            int v = e / NP, i = e - v * NP;
            s[slot(i, v)] = gin[e];
        }
    } else {
        const int E = vrem * NP;
        for (int e = tid; e < E; e += T) {
            int v = e / NP, i = e - v * NP;
            s[slot(i, v)] = gin[e];
        }
    }
    __syncthreads();

    /* ---- z axis (stride 1), batch-fast lanes ---- */
    {
        int v = tid & (VPB - 1), l = tid / VPB;
        if (v < vrem) {
            int base = l * 6;
            double2 r[6];
#pragma unroll
            for (int j = 0; j < 6; ++j) r[j] = s[slot(base + j, v)];
            dft6(r);
#pragma unroll
            for (int j = 0; j < 6; ++j) s[slot(base + j, v)] = r[j];
        }
    }
    /* z->y: the dependence closes within aligned 48-tid groups (a y-pass thread
       reads only its own volume's points, written by z-pass threads 48x..48x+47),
       so a 96-thread named barrier over 3 warps is CORRECT here -- but it measured
       slower than __syncthreads at both batched points (header). Off by default. */
#if NAMED_BAR
    asm volatile("bar.sync %0, 96;" :: "r"(tid / 96 + 1) : "memory");
#else
    __syncthreads();
#endif

    /* ---- y axis (stride 6), batch-fast lanes ---- */
    {
        int v = tid & (VPB - 1), l = tid / VPB;
        if (v < vrem) {
            int x = l / 6, z = l - x * 6;
            int base = x * 36 + z;
            double2 r[6];
#pragma unroll
            for (int j = 0; j < 6; ++j) r[j] = s[slot(base + 6 * j, v)];
            dft6(r);
#pragma unroll
            for (int j = 0; j < 6; ++j) s[slot(base + 6 * j, v)] = r[j];
        }
    }
    __syncthreads();

    /* ---- x axis (stride 36), line-fast lanes, results straight to global ---- */
    {
        int l = tid % NLINES, v = tid / NLINES;
        if (v < vrem) {
            double2 r[6];
#pragma unroll
            for (int j = 0; j < 6; ++j) r[j] = s[slot(l + 36 * j, v)];
            dft6(r);
            /* evict-first streaming stores when in can stay L2-resident: out is
               never re-read, and keeping the write stream out of L2 protects in's
               residency across the repeat loop (borrowed from L6_warpvolume round
               gpu_r1; 24.0 -> 14.9 us at B=4854). At HBM-scale batches there is no
               residency to protect and __stcs costs ~0.7%, so plain stores there. */
#pragma unroll
            for (int j = 0; j < 6; ++j) {
                if (STREAM_ST) __stcs(&gout[v * NP + l + 36 * j], r[j]);
                else           gout[v * NP + l + 36 * j] = r[j];
            }
        }
    }
}

/* B = 1 latency kernel: one 36-thread block (2 warps), z-pass fused into the global
 * load -- thread l = (x,y) reads its contiguous 96 B z-line straight from global,
 * transforms in registers, writes shared once. Two barriers total instead of three,
 * and one shared round trip fewer than the staged shape. Padded slot i + i/6 keeps
 * the y/x passes bank-clean enough; at one block none of it is on the critical path.
 * Borrowed: fft6_single shape from L6_warpvolume gpu_r2, fused-z-at-B=1 from
 * L8_blockfused gpu_r1's staging-free single kernel. */
#define PSLOT(i) ((i) + (i) / 6)
__global__ void __launch_bounds__(NLINES)
fft6_single(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    __shared__ double2 s[NP + NP / 6];
    const int l = (int)threadIdx.x;              /* 36 threads, no guards */
    double2 r[6];
    /* z-pass fused with load: line (x,y) = l, points l*6 .. l*6+5 contiguous */
#pragma unroll
    for (int z = 0; z < 6; ++z) r[z] = in[l * 6 + z];
    dft6(r);
#pragma unroll
    for (int z = 0; z < 6; ++z) s[PSLOT(l * 6 + z)] = r[z];
    __syncthreads();
    {   /* y-pass: line (x,z), stride 6 */
        const int i0 = (l / 6) * 36 + (l % 6);
#pragma unroll
        for (int y = 0; y < 6; ++y) r[y] = s[PSLOT(i0 + 6 * y)];
        dft6(r);
#pragma unroll
        for (int y = 0; y < 6; ++y) s[PSLOT(i0 + 6 * y)] = r[y];
    }
    __syncthreads();
    {   /* x-pass: line (y,z) = l, stride 36, straight to global */
#pragma unroll
        for (int x = 0; x < 6; ++x) r[x] = s[PSLOT(l + 36 * x)];
        dft6(r);
#pragma unroll
        for (int x = 0; x < 6; ++x) out[l + 36 * x] = r[x];
    }
}

/* B = 1 experiment (gpu_r4, idea from L8_warpradix8 r3's quad-at-B=1: use all four
 * SM schedulers and shorten the per-thread chain): 216 threads = 7 warps, ONE point
 * per thread per pass, each pass a direct 6-point DFT for a single output k from a
 * ping-pong shared buffer (no RAW hazard, still one barrier per handoff). Twiddles
 * from a 6-entry table with an incremental (j*k) mod 6 index -- no divergence. */
#define S3C 0.86602540378443864676
__constant__ double2 W6C[6] = { { 1.0,  0.0 }, { 0.5, -S3C }, { -0.5, -S3C },
                                { -1.0, 0.0 }, { -0.5, S3C }, {  0.5,  S3C } };

__global__ void __launch_bounds__(NP)
fft6_single216(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    __shared__ double2 s0[NP], s1[NP];
    const int t = (int)threadIdx.x;              /* 216 threads, no guards */
    s0[t] = in[t];                               /* coalesced 16 B per thread */
    __syncthreads();
    const int k = t % 6;                         /* output index within the line */
#pragma unroll
    for (int pass = 0; pass < 3; ++pass) {
        /* pass 0: z (stride 1, line base 6*(t/6));   pass 1: y (stride 6, base
           36*(t/36) + t%6... ); pass 2: x (stride 36). Recompute base per pass. */
        int stride, base;
        if (pass == 0)      { stride = 1;  base = (t / 6) * 6; }
        else if (pass == 1) { stride = 6;  int r = t / 6; base = (r / 6) * 36 + (t % 6); }
        else                { stride = 36; base = t % 36; }
        /* line index along the axis for MY output: k for z (t%6); for y it is
           (t/6)%6; for x it is t/36. Output slot is base + kk*stride = t always
           when kk is my own coordinate -- so read the line, dot with W6^(j*kk). */
        int kk = (pass == 0) ? k : (pass == 1 ? (t / 6) % 6 : t / 36);
        const double2 *src = (pass == 1) ? s1 : s0;
        double2 acc = { 0.0, 0.0 };
        int m = 0;
#pragma unroll
        for (int j = 0; j < 6; ++j) {
            double2 v = src[base + j * stride], w = W6C[m];
            acc.x += v.x * w.x - v.y * w.y;
            acc.y += v.x * w.y + v.y * w.x;
            m += kk; if (m >= 6) m -= 6;
        }
        if (pass == 2) { out[t] = acc; }
        else {
            double2 *dst = (pass == 0) ? s1 : s0;
            dst[t] = acc;
            __syncthreads();
        }
    }
}

struct fft3d_gpu_plan {
    int L; int batch; int grid; int stream_st;
    int b1_wide;                /* B=1: 216-thread one-point-per-thread kernel */
    int use_graph;              /* replay the single launch via a CUDA graph */
    cudaGraphExec_t gexec;      /* lazily captured, keyed on (gin, gout) */
    const double2 *gin;
    double2 *gout;
};

extern "C" const char *fft3d_gpu_name(void) { return "L6_batchcoalesced"; }
extern "C" const char *fft3d_gpu_description(void)
{
    return "L6: 8 volumes/block batch-major swizzled shared, DIT 2x3 codelet, "
           "last axis direct to global, __stcs when in fits L2; fused-z 36-thread "
           "single-volume kernel at B=1; graph-replayed launch";
}

extern "C" int fft3d_gpu_supports(int L) { return L == 6; }

extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)
{
    if (L != 6 || batch < 1) return NULL;
    fft3d_gpu_plan *p = (fft3d_gpu_plan *)malloc(sizeof(*p));
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->grid = (batch + VPB - 1) / VPB;
    /* streaming stores pay off exactly when the input buffer fits in the 40 MB L2
       so its residency across calls is worth protecting from the write stream */
    p->stream_st = ((long long)batch * NP * 16 <= 36u * 1024 * 1024);
    /* Single-kernel CUDA-graph replay of the launch, borrowed from L8_warpradix8
       gpu_r3 (idea originally L36_sharedtiled r1 / L45_pfa r2): saves launch path
       at the latency-shaped points and collapses the B=1 launch-noise spread.
       Rotated same-lease A/B (gpu_r4): B=1 2.62 vs 2.96 us, B_L2 14.36-14.47 vs
       14.78-14.89, B_HBM a wash with graph marginally ahead in every pair -- so
       on everywhere, unlike L8's HBM sign. L6BC_GRAPH=0/1 forces. */
    p->use_graph = 1;
    p->b1_wide = 0;             /* dev A/B knob only; see strategy record */
    {
        const char *fg = getenv("L6BC_GRAPH");
        if (fg) p->use_graph = atoi(fg);
        const char *fb = getenv("L6BC_B1");
        if (fb) p->b1_wide = atoi(fb);
    }
    p->gexec = 0;
    p->gin = 0;
    p->gout = 0;
    /* prefer the largest shared carveout so occupancy is capped by threads/regs,
       not by a small default carveout (5 blocks/SM need 138 KB) */
    cudaFuncSetAttribute(fft6_batch<true>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(fft6_batch<false>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    return p;
}

/* The one launch this plan performs, into stream cs (0 = the plain hot path). */
static void launch_plan(fft3d_gpu_plan *p, const double2 *in, double2 *out, cudaStream_t cs)
{
    if (p->batch == 1) {
        if (p->b1_wide) fft6_single216<<<1, NP, 0, cs>>>(in, out);
        else            fft6_single<<<1, NLINES, 0, cs>>>(in, out);
        return;
    }
    if (p->stream_st) fft6_batch<true ><<<p->grid, VPB * NLINES, 0, cs>>>(in, out, p->batch);
    else              fft6_batch<false><<<p->grid, VPB * NLINES, 0, cs>>>(in, out, p->batch);
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (p->use_graph) {
        /* Replay the single kernel launch as a CUDA graph, lazily captured and
         * keyed on the pointers (from L8_warpradix8 gpu_r3). Same kernel, same
         * arguments -> bit-identical output; capture cost lands in warmup. */
        if (!p->gexec || in != p->gin || out != p->gout) {
            if (p->gexec) { cudaGraphExecDestroy(p->gexec); p->gexec = 0; }
            cudaStream_t cs;
            cudaGraph_t g;
            cudaStreamCreate(&cs);
            cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal);
            launch_plan(p, in, out, cs);
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
    launch_plan(p, in, out, 0);
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p)
{
    if (p && p->gexec) cudaGraphExecDestroy(p->gexec);
    free(p);
}
