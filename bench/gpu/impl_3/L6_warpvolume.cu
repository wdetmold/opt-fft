/* L6_warpvolume -- L = 6 on one A100.  Round gpu_r2.
 *
 * The r1 register-resident warp-volume kernel (4 volumes/warp, 8 lanes/volume, DIF-z/
 * DIT 2x3 parity split, shuffle butterflies) was measured this round against two
 * alternatives and lost everywhere, so this file now ships the winners:
 *
 *   B = 1        fft6_single: one 64-thread block, 36 line-threads per pass, padded
 *                shared, direct stores. 3.13 us vs the r1 kernel's 5.29.
 *   batched      fft6_bstage: shared-staged batch-major kernel, structure borrowed
 *                outright from rival L6_batchcoalesced (round gpu_r1 record) -- their
 *                occupancy argument is simply correct: 40 registers/thread -> 5
 *                blocks/SM = 45 warps/SM hides L2/DRAM latency that my 148-register
 *                13-warp kernel could not, and at L=6 (5.15x bandwidth-bound) latency
 *                hiding is the whole game once the access pattern is clean.
 *
 * fft6_bstage: 8 consecutive volumes per 288-thread block staged through a batch-major
 * swizzled shared array, slot i*8 + ((v+i)&7):
 *   load    perfectly coalesced flat copy of the block's contiguous 8x216 chunk
 *           (6 independent 16-byte loads/thread); the swizzled shared write is a
 *           4-way bank conflict, paid once
 *   z, y    thread = (volume fast, line slow): the 8 lanes of a line-group differ
 *           only in v, so every shared access is 8 consecutive complex doubles in one
 *           aligned 128-byte block -- bank-conflict-free at any axis stride
 *   x       thread = (line fast, volume slow): the six output planes store straight
 *           to global, coalesced 36-element runs; no store phase, no fourth barrier
 * Line codelet: DIT 6 = 2x3, two Winograd DFT-3s over the parity classes + w6/w6^2
 * twiddles (arithmetic is invisible at this size; ncu shows the kernel at 89% of
 * sustained DRAM peak at B_HBM with exact-minimum bytes moved).
 *
 * Store policy, measured (see strategy record): evict-first __stcs stores when the
 * INPUT buffer can stay L2-resident across the driver's repeat loop (keeps the write
 * stream from evicting it: 18.6 -> 15.1 us at B=4854), plain stores when it cannot
 * (at B_HBM __stcs measured 1543.6 vs plain 1540.8 -- same sign rival found).
 * Streaming load hints, cp.async staging, and cudaAccessPolicyWindow L2 persistence
 * were all measured and all lost; numbers in the strategy record.
 */
#include <cuda_runtime.h>
#include <stdlib.h>

#include "../fft3d_gpu_api.h"

#define NP   216   /* 6^3 */
#define BVB  8     /* volumes per block  */
#define BTPB 288   /* threads per block  */

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
__global__ void __launch_bounds__(BTPB)
fft6_bstage(const double2 *__restrict__ in, double2 *__restrict__ out, int nvol)
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

/* B=1 latency kernel: one volume, one 64-thread block (2 warps -- barriers are cheap),
 * 36 line-threads per pass, padded shared (slot = i + i/6, worst 4-way on z), direct
 * global stores from the x-pass. Same idea as L8_blockfused's fft8_single. */
__global__ void __launch_bounds__(64)
fft6_single(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    __shared__ double2 s[NP + NP / 6];
    const int tid = (int)threadIdx.x;
#define PSLOT(i) ((i) + (i) / 6)
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

struct fft3d_gpu_plan { int L; int batch; int grid; int stcs; };

extern "C" const char *fft3d_gpu_name(void) { return "L6_warpvolume"; }
extern "C" const char *fft3d_gpu_description(void)
{
    return "L6: 8 volumes/block batch-major swizzled shared, DIT 2x3 lines, x-pass "
           "stores direct; stcs stores while input is L2-resident; 64-thread B=1 kernel";
}

extern "C" int fft3d_gpu_supports(int L) { return L == 6; }

extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)
{
    if (L != 6 || batch < 1) return NULL;
    fft3d_gpu_plan *p = (fft3d_gpu_plan *)malloc(sizeof(*p));
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->grid = (batch + BVB - 1) / BVB;
    /* evict-first stores exactly while the input buffer can stay L2-resident across
     * the repeat loop (predicate borrowed from L6_batchcoalesced round gpu_r2) */
    p->stcs = ((long long)batch * NP * 16 <= 36ll * 1024 * 1024);
    /* 27.6 KB static shared per block, 5 blocks/SM: needs the full carveout */
    cudaFuncSetAttribute(fft6_bstage<true>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(fft6_bstage<false>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    return p;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (p->batch == 1) { fft6_single<<<1, 64>>>(in, out); return; }
    if (p->stcs) fft6_bstage<true><<<p->grid, BTPB>>>(in, out, p->batch);
    else         fft6_bstage<false><<<p->grid, BTPB>>>(in, out, p->batch);
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p) { free(p); }
