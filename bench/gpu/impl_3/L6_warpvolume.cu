/* L6_warpvolume -- L = 6 on one A100.  Round gpu_r3.
 *
 * Kernels (selection is deterministic at plan time, never measured):
 *
 *   B = 1        fft6_single: one 64-thread block, 36 line-threads per pass, padded
 *                shared, direct stores.
 *   batched      fft6_bstage: shared-staged batch-major kernel (structure from rival
 *                L6_batchcoalesced gpu_r1), 8 consecutive volumes per 288-thread
 *                block, batch-major swizzled shared slot i*8 + ((v+i)&7).
 *
 * Round gpu_r3 verdict: the SHIPPED configuration is unchanged from gpu_r2 -- every
 * structural alternative built this round measured neutral or worse once ordering /
 * clock-ramp bias was controlled (rotated-order A/B within one lease). The barrier
 * chain is NOT the B_L2 residual: narrowing load->z and z->y to 96-thread named
 * barriers (`bar.sync id, 96` over one x-slab group -- the dependence provably closes
 * inside 3 warps once the load is made group-local) measured 15.14-15.42 us vs the
 * baseline's 14.92-14.96 at B=4854, in every rotation, on 10+ independent processes.
 * All variants stay compiled behind the L6_MODE env knob (dev A/B only; unset in
 * scored runs), numbers in the strategy record: group-local load, narrow barriers,
 * fused-z (18.5 us -- 96-byte-run global reads lose to a flat staged load even from
 * L2), incremental vs per-k division indexing.
 *
 * fft6_bstage passes:
 *   load    LOADMODE 0 (shipped): perfectly coalesced flat copy of the block's
 *           contiguous 8x216 chunk. LOADMODE 1: group-local copy. LOADMODE 2:
 *           fused-z, no staging. The swizzled shared write is a 4-way bank
 *           conflict, paid once.
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
 * when it cannot (predicate from L6_batchcoalesced gpu_r2).
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

/* Named barrier over one 96-thread (3-warp) x-slab group; ids 1..3 (0 is the
 * block-wide barrier __syncthreads uses). bar.sync orders shared-memory accesses
 * among the participating threads exactly like __syncthreads does block-wide. */
static __device__ __forceinline__ void bar_group(int grp)
{
    asm volatile("bar.sync %0, 96;" :: "r"(grp + 1) : "memory");
}

/* LOADMODE 0: flat coalesced load, full load->z barrier (the B_HBM shape).
 * LOADMODE 1: group-local load, load->z is a 96-thread named barrier.
 * LOADMODE 2: fused z: each z-pass thread reads its own 96-byte contiguous line
 *             straight from global -- no staging phase, no load->z barrier at all.
 * NARROWZY:   z->y barrier is bar.sync,96 instead of __syncthreads.
 * All variants run identical dft6 arithmetic on identical data, so the output is
 * bit-identical across modes. */
template <bool STCS_ST, int LOADMODE, bool NARROWZY>
static __device__ __forceinline__ void
fft6_bstage_body(const double2 *__restrict__ in, double2 *__restrict__ out, int nvol)
{
    __shared__ double2 s[NP * BVB];
    const int tid = (int)threadIdx.x;
    const int grp = tid / 96;                /* x-slab group: planes 2g, 2g+1 */
    const long long vbase = (long long)blockIdx.x * BVB;
    const int nv = (nvol - vbase >= BVB) ? BVB : (int)(nvol - vbase);

    const double2 *gin = in + vbase * NP;
    if (LOADMODE == 2) {
        /* no staging: the z-pass below reads global directly */
    } else if (LOADMODE == 1) {
        /* group g loads i in [72g, 72g+72) of every volume: 6 elements/thread,
         * contiguous 1152-byte runs per (volume, slab-pair).  Per-k independent
         * division keeps the 6 address computations parallel -- an incremental
         * (v,off) carry measured 15.1-15.2 vs 14.9 us at B_L2 (serial chain). */
        const int j = tid - grp * 96;
        const int ibase = grp * 72;
#pragma unroll
        for (int k = 0; k < 6; ++k) {
            int e = k * 96 + j;              /* 0..575 within the group */
            int v = e / 72, off = e - v * 72;
            if (v < nv) s[SSLOT(ibase + off, v)] = gin[v * NP + ibase + off];
        }
        bar_group(grp);
    } else {
        if (nv == BVB) {
#pragma unroll
            for (int k = 0; k < 6; ++k) {
                int g = k * BTPB + tid;      /* 0..1727, perfectly coalesced */
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
    }

    {   /* z-pass: thread = (volume fast, line slow); line l = (x,y), stride 1.
         * l/6 = x, so thread tid works planes x = 2*grp, 2*grp+1: group-local. */
        const int v = tid & 7, l = tid >> 3;
        if (v < nv) {
            const int i0 = (l / 6) * 36 + (l % 6) * 6;
            double2 a[6], X[6];
#pragma unroll
            for (int z = 0; z < 6; ++z)
                a[z] = (LOADMODE == 2) ? gin[v * NP + i0 + z] : s[SSLOT(i0 + z, v)];
            dft6(a, X);
#pragma unroll
            for (int z = 0; z < 6; ++z) s[SSLOT(i0 + z, v)] = X[z];
        }
    }
    /* z->y closes within the x-slab group (y-pass line l=(x,z) has the same
     * l/6 = x grouping), so 3 warps synchronize instead of 9 */
    if (NARROWZY) bar_group(grp);
    else          __syncthreads();

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
    __syncthreads();                         /* x-pass lines cross all x slabs */

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

/* Two launch shells around the same body: the full-barrier kernels compile to 40
 * registers naturally and ship at the DRAM wall -- leave them uncapped; the named-
 * barrier variants bloat to 50 (the asm memory clobber pins values), so cap them at
 * 45 regs to keep 5 blocks/SM.  Verified: no spills in either shell. */
template <bool STCS_ST, int LOADMODE, bool NARROWZY>
__global__ void __launch_bounds__(BTPB)
fft6_bstage(const double2 *__restrict__ in, double2 *__restrict__ out, int nvol)
{
    fft6_bstage_body<STCS_ST, LOADMODE, NARROWZY>(in, out, nvol);
}

template <bool STCS_ST, int LOADMODE, bool NARROWZY>
__global__ void __launch_bounds__(BTPB, 5)
fft6_bstage_nb(const double2 *__restrict__ in, double2 *__restrict__ out, int nvol)
{
    fft6_bstage_body<STCS_ST, LOADMODE, NARROWZY>(in, out, nvol);
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

struct fft3d_gpu_plan { int L; int batch; int grid; int mode; };

/* mode: which fft6_bstage instantiation execute() launches.
 *   0 = <false,0,false>  plain stores, flat load, full barriers   (r2 HBM shape)
 *   1 = <true, 0,false>  stcs stores,  flat load, full barriers   (r2 L2 shape)
 *   2 = <true, 1,true >  stcs stores,  group load, narrow barriers
 *   3 = <true, 0,true >  stcs stores,  flat load,  narrow z->y
 *   4 = <false,0,true >  plain stores, flat load,  narrow z->y
 *   5 = <true, 1,false>  stcs stores,  group load, full z->y
 *   6 = <true, 2,true >  stcs stores,  fused-z,    narrow z->y
 *   7 = <false,2,true >  plain stores, fused-z,    narrow z->y
 *   8 = <false,1,true >  plain stores, group load, narrow barriers
 *   9 = mode 2 in the uncapped shell (50 regs, 4 blocks/SM; dev A/B only)
 * All modes are bit-identical in output; the L6_MODE env var (dev A/B only)
 * overrides the deterministic default pick. */
static void launch_bstage(int mode, int grid, const double2 *in, double2 *out, int nvol)
{
    switch (mode) {
    case 0: fft6_bstage<false, 0, false><<<grid, BTPB>>>(in, out, nvol); break;
    case 1: fft6_bstage<true,  0, false><<<grid, BTPB>>>(in, out, nvol); break;
    default:
    case 2: fft6_bstage_nb<true,  1, true ><<<grid, BTPB>>>(in, out, nvol); break;
    case 3: fft6_bstage_nb<true,  0, true ><<<grid, BTPB>>>(in, out, nvol); break;
    case 4: fft6_bstage_nb<false, 0, true ><<<grid, BTPB>>>(in, out, nvol); break;
    case 5: fft6_bstage_nb<true,  1, false><<<grid, BTPB>>>(in, out, nvol); break;
    case 6: fft6_bstage_nb<true,  2, true ><<<grid, BTPB>>>(in, out, nvol); break;
    case 7: fft6_bstage_nb<false, 2, true ><<<grid, BTPB>>>(in, out, nvol); break;
    case 8: fft6_bstage_nb<false, 1, true ><<<grid, BTPB>>>(in, out, nvol); break;
    case 9: fft6_bstage<true,  1, true ><<<grid, BTPB>>>(in, out, nvol); break;
    }
}

extern "C" const char *fft3d_gpu_name(void) { return "L6_warpvolume"; }
extern "C" const char *fft3d_gpu_description(void)
{
    return "L6: 8 volumes/block batch-major swizzled shared, DIT 2x3 lines, x-pass "
           "stores direct; stcs stores while input is L2-resident; 64-thread B=1 "
           "kernel; r3: barrier/load variants measured, all lost, r2 config kept";
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
     * the repeat loop (predicate from L6_batchcoalesced gpu_r2); plain stores above,
     * where the case sits at the DRAM wall.  Flat load + full barriers both regimes:
     * every alternative measured worse (see file header / strategy record). */
    p->mode = ((long long)batch * NP * 16 <= 36ll * 1024 * 1024) ? 1 : 0;
    {   /* dev-only A/B override; unset in scored runs, so the pick is deterministic */
        const char *m = getenv("L6_MODE");
        if (m && *m >= '0' && *m <= '8') p->mode = *m - '0';
    }
    /* 27.6 KB static shared per block, 5 blocks/SM: needs the full carveout */
    cudaFuncSetAttribute(fft6_bstage<false, 0, false>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(fft6_bstage<true,  0, false>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(fft6_bstage<true,  1, true >,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(fft6_bstage_nb<true,  1, true >,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(fft6_bstage_nb<true,  0, true >,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(fft6_bstage_nb<false, 0, true >,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(fft6_bstage_nb<true,  1, false>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(fft6_bstage_nb<true,  2, true >,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(fft6_bstage_nb<false, 2, true >,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(fft6_bstage_nb<false, 1, true >,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    return p;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (p->batch == 1) { fft6_single<<<1, 64>>>(in, out); return; }
    launch_bstage(p->mode, p->grid, in, out, p->batch);
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p) { free(p); }
