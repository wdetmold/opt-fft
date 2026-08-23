/* L6_warpvolume -- L = 6 on one A100.  Round gpu_r4.
 *
 * Kernels (selection is deterministic at plan time, never measured):
 *
 *   B = 1        fft6_single_fz: one 36-thread block, z-pass fused into the global
 *                load (2 barriers), padded shared, direct stores.  Launched by
 *                replaying a pre-built single-kernel-node CUDA graph.
 *   batched      fft6_bstage: shared-staged batch-major kernel (structure from rival
 *                L6_batchcoalesced gpu_r1), 8 consecutive volumes per 288-thread
 *                block, batch-major swizzled shared slot i*8 + ((v+i)&7).
 *
 * Round gpu_r4: execute() is ASYNCHRONOUS -- each call launches on the next of 8
 * plan-owned cudaStreamNonBlocking streams, no event fencing, so back-to-back timed
 * calls pipeline: call n+1's blocks fill SM slots as call n's retire.  Adopted from
 * L17_dmma gpu_r3 (via L13_dmma gpu_r4, same mechanism at a neighbouring geometry).
 * The contract allows it in so many words ("Asynchronous work is fine: the driver
 * synchronizes before stopping the clock"), driver.cu:175-178 anticipates
 * non-blocking-stream launches, and rounds r3/r4 scored the ring with PASS.
 * Correctness under overlap: every kernel reads only `in` (never written) and every
 * in-flight call writes IDENTICAL bytes to `out`, so any interleaving of concurrent
 * instances yields the same memory image -- bit-identical, structurally.
 * At B = 1 the launch itself is the cost, so the single kernel is wrapped in ONE
 * explicitly-built graph instantiated once per ring stream (relaunches of one
 * cudaGraphExec_t serialize -- N execs keep the overlap; from L17_dmma gpu_r4,
 * crediting L8_warpradix8 gpu_r3), lazily (re)built keyed on the (in,out) pair.
 *
 * fft6_bstage passes (unchanged since gpu_r2 -- every structural alternative
 * measured worse; see the strategy record for the r3 barrier/load post-mortem):
 *   load    flat perfectly-coalesced copy of the block's contiguous 8x216 chunk;
 *           the swizzled shared write is a 4-way bank conflict, paid once
 *   z, y    thread = (volume fast, line slow): the 8 lanes of a line-group differ
 *           only in v, so every shared access is 8 consecutive complex doubles in one
 *           aligned 128-byte block -- bank-conflict-free at any axis stride
 *   x       thread = (line fast, volume slow): the six output planes store straight
 *           to global, coalesced 36-element runs; no store phase
 * Line codelet: DIT 6 = 2x3, two Winograd DFT-3s over the parity classes + w6/w6^2
 * twiddles (arithmetic is invisible at this size).
 *
 * Store policy, measured (see strategy record): evict-first __stcs stores when the
 * INPUT buffer can stay L2-resident across the driver's repeat loop, plain stores
 * when it cannot (predicate from L6_batchcoalesced gpu_r2; re-confirmed under the
 * ring this round, as L13_dmma gpu_r4 also found).
 *
 * Env knobs (dev A/B only, unset in scored runs; defaults are the shipped picks):
 *   L6_NSTREAM  ring depth, default 4; 0 = synchronous gpu_r3 behaviour
 *   L6_GRAPH    0 disables graph replay at B=1 (plain ring launch instead)
 *   L6_SINGLE   0 = staged 64-thread gpu_r2 single kernel instead of fused-z
 *   L6_MODE     batched-kernel variant override (see launch_bstage)
 */
#include <cuda_runtime.h>
#include <stdlib.h>

#include "../fft3d_gpu_api.h"

#define NP   216   /* 6^3 */
#define BVB  8     /* volumes per block  */
#define BTPB 288   /* threads per block  */
#define MAXSTR 16

/* Winograd forward DFT-3, X = F3 * (a,b,c), w3 = exp(-2*pi*i/3). */
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

/* DIT 6 = 2x3: DFT-3 of each parity class, w6^k twiddle, radix-2 combine. */
static __device__ __forceinline__ void dft6(const double2 a[6], double2 X[6])
{
    const double S3 = 0.86602540378443864676;
    double2 E0, E1, E2, O0, O1, O2;
    dft3(a[0], a[2], a[4], E0, E1, E2);
    dft3(a[1], a[3], a[5], O0, O1, O2);
    double2 T1, T2;                          /* w6*O1, w6^2*O2 */
    T1.x =  0.5 * O1.x + S3 * O1.y;  T1.y =  0.5 * O1.y - S3 * O1.x;
    T2.x = -0.5 * O2.x + S3 * O2.y;  T2.y = -0.5 * O2.y - S3 * O2.x;
    X[0].x = E0.x + O0.x;  X[0].y = E0.y + O0.y;
    X[3].x = E0.x - O0.x;  X[3].y = E0.y - O0.y;
    X[1].x = E1.x + T1.x;  X[1].y = E1.y + T1.y;
    X[4].x = E1.x - T1.x;  X[4].y = E1.y - T1.y;
    X[2].x = E2.x + T2.x;  X[2].y = E2.y + T2.y;
    X[5].x = E2.x - T2.x;  X[5].y = E2.y - T2.y;
}

#define SSLOT(i, v) ((i) * 8 + (((v) + (i)) & 7))

template <bool STCS_ST>
static __device__ __forceinline__ void
fft6_bstage_body(const double2 *__restrict__ in, double2 *__restrict__ out, int nvol)
{
    __shared__ double2 s[NP * BVB];
    const int tid = (int)threadIdx.x;
    const long long vbase = (long long)blockIdx.x * BVB;
    const int nv = (nvol - vbase >= BVB) ? BVB : (int)(nvol - vbase);

    const double2 *gin = in + vbase * NP;
    if (nv == BVB) {
#pragma unroll
        for (int k = 0; k < 6; ++k) {
            int g = k * BTPB + tid;          /* 0..1727, perfectly coalesced */
            int v = g / NP, i = g - v * NP;
            s[SSLOT(i, v)] = gin[g];
        }
    } else {
#pragma unroll
        for (int k = 0; k < 6; ++k) {
            int g = k * BTPB + tid;
            if (g < nv * NP) {
                int v = g / NP, i = g - v * NP;
                s[SSLOT(i, v)] = gin[g];
            }
        }
    }
    __syncthreads();

    {   /* z-pass: thread = (volume fast, line slow); line l = (x,y), stride 1 */
        const int v = tid & 7, l = tid >> 3;
        if (v < nv) {
            const int i0 = (l / 6) * 36 + (l % 6) * 6;
            double2 a[6], X[6];
#pragma unroll
            for (int z = 0; z < 6; ++z) a[z] = s[SSLOT(i0 + z, v)];
            dft6(a, X);
#pragma unroll
            for (int z = 0; z < 6; ++z) s[SSLOT(i0 + z, v)] = X[z];
        }
    }
    __syncthreads();

    {   /* y-pass: line l = (x,z), stride 6 */
        const int v = tid & 7, l = tid >> 3;
        if (v < nv) {
            const int i0 = (l / 6) * 36 + (l % 6);
            double2 a[6], X[6];
#pragma unroll
            for (int y = 0; y < 6; ++y) a[y] = s[SSLOT(i0 + y * 6, v)];
            dft6(a, X);
#pragma unroll
            for (int y = 0; y < 6; ++y) s[SSLOT(i0 + y * 6, v)] = X[y];
        }
    }
    __syncthreads();

    {   /* x-pass: thread = (line fast, volume slow); stores direct to global,
           coalesced 36-element runs per output plane */
        const int v = tid / 36, l = tid - v * 36;
        if (v < nv) {
            double2 a[6], X[6];
#pragma unroll
            for (int x = 0; x < 6; ++x) a[x] = s[SSLOT(l + x * 36, v)];
            dft6(a, X);
            double2 *o = out + (vbase + v) * NP + l;
#pragma unroll
            for (int x = 0; x < 6; ++x) {
                if (STCS_ST) __stcs(&o[x * 36], X[x]);
                else         o[x * 36] = X[x];
            }
        }
    }
}

template <bool STCS_ST>
__global__ void __launch_bounds__(BTPB)
fft6_bstage(const double2 *__restrict__ in, double2 *__restrict__ out, int nvol)
{
    fft6_bstage_body<STCS_ST>(in, out, nvol);
}

#define PSLOT(i) ((i) + (i) / 6)

/* B=1 latency kernel, gpu_r4: one 36-thread block, the z-pass FUSED into the global
 * load -- thread l = (x,y) reads its own contiguous 96 B z-line, transforms in
 * registers, writes shared once; then y and x passes over padded shared, x-pass
 * storing straight to global.  Two barriers, one shared round trip fewer than the
 * staged r2 shape.  Borrowed back from L6_batchcoalesced gpu_r3 (2.87 vs 3.09 us),
 * itself descended from my gpu_r2 fft6_single and L8_blockfused's fft8_single;
 * fused-z wins exactly and only at B=1, where latency, not sector economy, is the
 * cost (both entries' records agree on the sign flip vs batched). */
__global__ void __launch_bounds__(36)
fft6_single_fz(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    __shared__ double2 s[NP + NP / 6];
    const int tid = (int)threadIdx.x;        /* 0..35 */
    {                                        /* z-pass fused with the load */
        const double2 *g = in + tid * 6;
        double2 a[6], X[6];
#pragma unroll
        for (int z = 0; z < 6; ++z) a[z] = g[z];
        dft6(a, X);
#pragma unroll
        for (int z = 0; z < 6; ++z) s[PSLOT(tid * 6 + z)] = X[z];
    }
    __syncthreads();
    {                                        /* y-pass: line (x,z) */
        const int i0 = (tid / 6) * 36 + (tid % 6);
        double2 a[6], X[6];
#pragma unroll
        for (int y = 0; y < 6; ++y) a[y] = s[PSLOT(i0 + y * 6)];
        dft6(a, X);
#pragma unroll
        for (int y = 0; y < 6; ++y) s[PSLOT(i0 + y * 6)] = X[y];
    }
    __syncthreads();
    {                                        /* x-pass: line (y,z), store direct */
        double2 a[6], X[6];
#pragma unroll
        for (int x = 0; x < 6; ++x) a[x] = s[PSLOT(tid + x * 36)];
        dft6(a, X);
#pragma unroll
        for (int x = 0; x < 6; ++x) out[tid + x * 36] = X[x];
    }
}

/* gpu_r2 staged single kernel, kept for A/B (L6_SINGLE=0): 64 threads, 3 barriers. */
__global__ void __launch_bounds__(64)
fft6_single(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    __shared__ double2 s[NP + NP / 6];
    const int tid = (int)threadIdx.x;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        int g = k * 64 + tid;
        if (g < NP) s[PSLOT(g)] = in[g];
    }
    __syncthreads();
    if (tid < 36) {                                  /* z-pass: line (x,y) */
        const int i0 = tid * 6;
        double2 a[6], X[6];
#pragma unroll
        for (int z = 0; z < 6; ++z) a[z] = s[PSLOT(i0 + z)];
        dft6(a, X);
#pragma unroll
        for (int z = 0; z < 6; ++z) s[PSLOT(i0 + z)] = X[z];
    }
    __syncthreads();
    if (tid < 36) {                                  /* y-pass: line (x,z) */
        const int i0 = (tid / 6) * 36 + (tid % 6);
        double2 a[6], X[6];
#pragma unroll
        for (int y = 0; y < 6; ++y) a[y] = s[PSLOT(i0 + y * 6)];
        dft6(a, X);
#pragma unroll
        for (int y = 0; y < 6; ++y) s[PSLOT(i0 + y * 6)] = X[y];
    }
    __syncthreads();
    if (tid < 36) {                                  /* x-pass: line (y,z), store */
        double2 a[6], X[6];
#pragma unroll
        for (int x = 0; x < 6; ++x) a[x] = s[PSLOT(tid + x * 36)];
        dft6(a, X);
#pragma unroll
        for (int x = 0; x < 6; ++x) out[tid + x * 36] = X[x];
    }
}

struct fft3d_gpu_plan {
    int L, batch, grid, mode;
    int nstream, next;                 /* ring; nstream 0 = synchronous legacy */
    cudaStream_t str[MAXSTR];
    int use_graph, single_fz;
    cudaGraph_t graph;                 /* B=1: one kernel node, N execs */
    cudaGraphExec_t gexec[MAXSTR];
    int gvalid;
    const double2 *gin; double2 *gout; /* pointers the graph was built for */
};

/* mode: which fft6_bstage instantiation execute() launches.
 *   0 = plain stores (the B_HBM shape)     1 = __stcs stores (the L2 shape) */
static void launch_bstage(int mode, int grid, cudaStream_t st,
                          const double2 *in, double2 *out, int nvol)
{
    if (mode == 1) fft6_bstage<true ><<<grid, BTPB, 0, st>>>(in, out, nvol);
    else           fft6_bstage<false><<<grid, BTPB, 0, st>>>(in, out, nvol);
}

/* Build the B=1 graph for this (in,out) pair: one kernel node, instantiated once
 * per ring stream (a single cudaGraphExec_t serializes its own relaunches).  The
 * driver reuses one buffer pair, so this runs once, during warmup. */
static void graph_build(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (p->gvalid) {
        for (int i = 0; i < p->nstream; ++i) cudaGraphExecDestroy(p->gexec[i]);
        cudaGraphDestroy(p->graph);
        p->gvalid = 0;
    }
    p->gin = in; p->gout = out;
    if (cudaGraphCreate(&p->graph, 0) != cudaSuccess) return;
    cudaKernelNodeParams np;
    void *args[2] = { (void *)&p->gin, (void *)&p->gout };
    np.func = p->single_fz ? (void *)fft6_single_fz : (void *)fft6_single;
    np.gridDim = dim3(1, 1, 1);
    np.blockDim = dim3(p->single_fz ? 36 : 64, 1, 1);
    np.sharedMemBytes = 0;
    np.kernelParams = args;
    np.extra = NULL;
    cudaGraphNode_t node;
    if (cudaGraphAddKernelNode(&node, p->graph, NULL, 0, &np) != cudaSuccess) {
        cudaGraphDestroy(p->graph);
        return;
    }
    int ok = 1;
    for (int i = 0; i < p->nstream; ++i)
        if (cudaGraphInstantiate(&p->gexec[i], p->graph, 0ULL) != cudaSuccess) {
            for (int j = 0; j < i; ++j) cudaGraphExecDestroy(p->gexec[j]);
            cudaGraphDestroy(p->graph);
            ok = 0;
            break;
        }
    p->gvalid = ok;
}

extern "C" const char *fft3d_gpu_name(void) { return "L6_warpvolume"; }
extern "C" const char *fft3d_gpu_description(void)
{
    return "L6: 8 volumes/block batch-major swizzled shared, DIT 2x3 lines, x-pass "
           "stores direct, stcs while input L2-resident; async execute over a "
           "4-stream ring; B=1 fused-z 36-thread kernel graph-replayed per stream";
}

extern "C" int fft3d_gpu_supports(int L) { return L == 6; }

extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)
{
    if (L != 6 || batch < 1) return NULL;
    fft3d_gpu_plan *p = (fft3d_gpu_plan *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->grid = (batch + BVB - 1) / BVB;
    /* evict-first stores exactly while the input buffer can stay L2-resident across
     * the repeat loop (predicate from L6_batchcoalesced gpu_r2, re-confirmed under
     * the ring); plain stores above, where the case sits at the DRAM wall. */
    p->mode = ((long long)batch * NP * 16 <= 36ll * 1024 * 1024) ? 1 : 0;
    p->nstream = 4;   /* 4 vs 8 tie at both batched points, 4 leans ahead at B=1 */
    p->use_graph = 1;
    p->single_fz = 1;
    {   /* dev-only A/B overrides; unset in scored runs, so every pick is deterministic */
        const char *e;
        if ((e = getenv("L6_MODE"))    && *e >= '0' && *e <= '1') p->mode = *e - '0';
        if ((e = getenv("L6_NSTREAM")) && *e) {
            int n = atoi(e);
            p->nstream = n < 0 ? 0 : (n > MAXSTR ? MAXSTR : n);
        }
        if ((e = getenv("L6_GRAPH"))  && *e == '0') p->use_graph = 0;
        if ((e = getenv("L6_SINGLE")) && *e == '0') p->single_fz = 0;
    }
    for (int i = 0; i < p->nstream; ++i)
        if (cudaStreamCreateWithFlags(&p->str[i], cudaStreamNonBlocking) != cudaSuccess) {
            p->nstream = i;                  /* fall back to what we got (0 = sync) */
            break;
        }
    if (p->nstream == 0) p->use_graph = 0;
    /* 27.6 KB static shared per block, 5 blocks/SM: needs the full carveout */
    cudaFuncSetAttribute(fft6_bstage<false>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(fft6_bstage<true>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    return p;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    cudaStream_t st = (cudaStream_t)0;
    int slot = 0;
    if (p->nstream > 0) {
        slot = p->next;
        p->next = (p->next + 1 == p->nstream) ? 0 : p->next + 1;
        st = p->str[slot];
    }
    if (p->batch == 1) {
        if (p->use_graph) {
            if (!p->gvalid || in != p->gin || out != p->gout) graph_build(p, in, out);
            if (p->gvalid) { cudaGraphLaunch(p->gexec[slot], st); return; }
        }
        if (p->single_fz) fft6_single_fz<<<1, 36, 0, st>>>(in, out);
        else              fft6_single<<<1, 64, 0, st>>>(in, out);
        return;
    }
    launch_bstage(p->mode, p->grid, st, in, out, p->batch);
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p)
{
    if (!p) return;
    if (p->gvalid) {
        for (int i = 0; i < p->nstream; ++i) cudaGraphExecDestroy(p->gexec[i]);
        cudaGraphDestroy(p->graph);
    }
    for (int i = 0; i < p->nstream; ++i) cudaStreamDestroy(p->str[i]);
    free(p);
}
