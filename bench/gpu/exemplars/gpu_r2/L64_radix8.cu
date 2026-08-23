/* L64_radix8 -- 64^3 complex-double 3D FFT on one A100, two global passes,
 * L2-blocked across the batch.
 *
 * Structure (literature 09 section 9.8, structures 1+2+3 combined):
 *   kernel_yz: one block per (b,x) plane. 512 threads = 64 lines x 8 lanes.
 *              z-axis FFT straight from global into registers (coalesced 128B),
 *              one padded shared-memory transpose (stride 65 double2, conflict-free
 *              for 16-byte accesses), y-axis FFT from shared rows, write to `out`
 *              in standard layout (64B segments, no amplification).
 *   kernel_x:  x-axis, IN PLACE on `out` (each lane rewrites exactly its own 8
 *              elements), no shared memory, no workspace buffer at all.
 *
 * The 64-point line FFT is radix-8 x radix-8, register-resident: each of 8 lanes
 * holds 8 complex doubles (a 64-point line cannot live in one thread: 256 regs).
 *   j = 8*j1 + j2 (lane t = j2),  k = k1 + 8*k2:
 *   X[k1+8k2] = DFT8_{j2}( W64^{j2*k1} * DFT8_{j1}( x[8j1+j2] ) )
 * The j2<->k1 exchange is an 8x8 xor-butterfly shuffle transpose (3 stages),
 * so the whole line FFT touches neither shared nor local memory.
 *
 * Nominal global traffic is 2 reads + 2 writes = 64 B/point (vs 3-pass 96), and
 * the batch is processed in 4-volume chunks on two streams so that kernel_x's
 * read+rewrite of the intermediate hits L2 (verified with ncu: kernel_x moves
 * ~17.5 MB of DRAM per 33.6 MB chunk instead of 33.6). Streaming accesses carry
 * evict-first hints (__ldcs/__ldlu/__stcs) so they cannot push the intermediate
 * out of L2; measured effective HBM traffic is ~42% below the unchunked version.
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "../fft3d_gpu_api.h"

#define SH_STRIDE 65   /* 64 padded to odd: conflict-free 16-byte shared access */

struct fft3d_gpu_plan {
    int L, batch;
    int chunk;         /* volumes per chunk for L2 blocking (0 = monolithic) */
    double2 *w64;      /* device: W64[j] = exp(-2 pi i j / 64), 64 entries */
    cudaStream_t str[4];
    cudaGraphExec_t gexec; /* B=1: captured two-kernel graph, keyed on (in,out) */
    const double2 *gin;
    double2 *gout;
};

__device__ __forceinline__ double2 cmul(double2 a, double2 b)
{
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
/* multiply by -i */
__device__ __forceinline__ double2 mul_mi(double2 a) { return make_double2(a.y, -a.x); }
/* multiply by +i */
__device__ __forceinline__ double2 mul_pi(double2 a) { return make_double2(-a.y, a.x); }
__device__ __forceinline__ double2 cadd(double2 a, double2 b) { return make_double2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ double2 csub(double2 a, double2 b) { return make_double2(a.x - b.x, a.y - b.y); }

/* 8-point DFT, natural order in and out, forward (exp(-2 pi i jk/8)). */
__device__ __forceinline__ void dft8(double2 r[8])
{
    const double C = 0.70710678118654752440; /* sqrt(2)/2 */
    /* radix-2 DIT, inputs taken bit-reversed */
    double2 y0 = cadd(r[0], r[4]), y1 = csub(r[0], r[4]);
    double2 y2 = cadd(r[2], r[6]), y3 = csub(r[2], r[6]);
    double2 y4 = cadd(r[1], r[5]), y5 = csub(r[1], r[5]);
    double2 y6 = cadd(r[3], r[7]), y7 = csub(r[3], r[7]);
    /* two 4-point DFTs: A = DFT4(x0,x2,x4,x6), B = DFT4(x1,x3,x5,x7) */
    double2 my3 = mul_mi(y3), my7 = mul_mi(y7);
    double2 a0 = cadd(y0, y2), a2 = csub(y0, y2);
    double2 a1 = cadd(y1, my3), a3 = csub(y1, my3);
    double2 b0 = cadd(y4, y6), b2 = csub(y4, y6);
    double2 b1 = cadd(y5, my7), b3 = csub(y5, my7);
    /* combine with w8^k: w0=1, w1=C(1-i), w2=-i, w3=-C(1+i) */
    double2 t1 = make_double2(C * (b1.x + b1.y), C * (b1.y - b1.x));
    double2 t2 = mul_mi(b2);
    double2 t3 = make_double2(C * (b3.y - b3.x), -C * (b3.x + b3.y));
    r[0] = cadd(a0, b0); r[4] = csub(a0, b0);
    r[1] = cadd(a1, t1); r[5] = csub(a1, t1);
    r[2] = cadd(a2, t2); r[6] = csub(a2, t2);
    r[3] = cadd(a3, t3); r[7] = csub(a3, t3);
}

/* 8x8 complex transpose across the 8 lanes of an aligned lane-group.
 * Before: lane t holds M[t][j] in r[j].  After: lane t holds M[j][t] in r[j]. */
__device__ __forceinline__ void transpose8(double2 r[8], int t)
{
#pragma unroll
    for (int m = 1; m < 8; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j & m) {
                const int j0 = j ^ m;
                double2 v = (t & m) ? r[j0] : r[j];
                v.x = __shfl_xor_sync(0xffffffffu, v.x, m);
                v.y = __shfl_xor_sync(0xffffffffu, v.y, m);
                if (t & m) r[j0] = v; else r[j] = v;
            }
        }
    }
}

/* 64-point forward FFT of one line spread over 8 lanes; tw[k1-1] = W64^{t*k1}.
 * In: lane t holds x[8*j + t] in r[j].  Out: lane t holds X[t + 8*j] in r[j]. */
__device__ __forceinline__ void fft64_reg(double2 r[8], int t, const double2 tw[7])
{
    dft8(r); /* over j1: r[k1] = sum_j1 x[8j1+t] W8^{j1 k1} */
#pragma unroll
    for (int k1 = 1; k1 < 8; ++k1)
        r[k1] = cmul(r[k1], tw[k1 - 1]);
    transpose8(r, t); /* r[j2] = A[j2][k1=t] */
    dft8(r); /* over j2: r[k2] = X[t + 8*k2] */
}

/* per-lane twiddles, loaded once per kernel (L1-resident table) */
__device__ __forceinline__ void load_tw(double2 tw[7], int t, const double2 *__restrict__ w64)
{
#pragma unroll
    for (int k1 = 1; k1 < 8; ++k1) tw[k1 - 1] = __ldg(&w64[t * k1]);
}

/* Pass 1: one block per (b,x) plane; z-axis then y-axis. tmp gets standard layout.
 * STREAM=1 marks input loads evict-first so they cannot push the L2-resident
 * intermediate out (chunked path); STREAM=0 keeps them cacheable, which is what a
 * batch small enough to be wholly L2-resident across repeats wants. */
template <int STREAM>
__global__ void __launch_bounds__(512, 2) kernel_yz(const double2 *__restrict__ in,
                                                 double2 *__restrict__ tmp,
                                                 const double2 *__restrict__ w64)
{
    extern __shared__ double2 sh[]; /* [k_z][y], row stride SH_STRIDE */
    const size_t base = (size_t)blockIdx.x * 4096; /* (b*64 + x) * 64*64 */
    const int t = threadIdx.x & 7;
    const int g = threadIdx.x >> 3; /* 0..63 */
    double2 r[8], tw[7];
    load_tw(tw, t, w64);

    /* z lines: g = y. Warp reads 4 rows x 8 consecutive = 128B segments. */
    const double2 *p = in + base + (size_t)g * 64 + t;
#pragma unroll
    for (int j = 0; j < 8; ++j) r[j] = STREAM ? __ldcs(p + 8 * j) : __ldg(p + 8 * j);
    fft64_reg(r, t, tw);
    /* lane t holds Xz[t+8j] for line y=g: store transposed so y-lines become rows */
#pragma unroll
    for (int j = 0; j < 8; ++j) sh[(size_t)(8 * j + t) * SH_STRIDE + g] = r[j];
    __syncthreads();

    /* y lines: g = k_z, row of sh is contiguous */
#pragma unroll
    for (int j = 0; j < 8; ++j) r[j] = sh[(size_t)g * SH_STRIDE + 8 * j + t];
    fft64_reg(r, t, tw);
    /* lane t holds Y[t+8j] for k_z=g. Round-trip through shared so the global
     * store is fully coalesced 128B segments (the direct r1 store was 64B
     * segments = 2x the L1 wavefronts and 2.4/4 sectors per L2 store line). */
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) sh[(size_t)(8 * j + t) * SH_STRIDE + g] = r[j];
    __syncthreads();
    double2 *q = tmp + base;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int m = (int)threadIdx.x + 512 * j;
        q[m] = sh[(size_t)(m >> 6) * SH_STRIDE + (m & 63)];
    }
}

/* Pass 2: x-axis, in place on `out` (no workspace buffer), staged through a
 * padded shared (x,z) plane so BOTH global sides are fully coalesced 128B
 * segments. The r1 version read/wrote straight from registers in 64B segments;
 * ncu showed L1TEX at 79% SOL (the kernel's roof) with 39% of warp cycles in
 * long-scoreboard stalls, while DRAM sat at 31%. Between the line read and the
 * line write-back no barrier is needed: lane t reads and rewrites exactly the
 * slot set {(8j+t), col g}, disjoint across lanes.
 * HALF=0: full (x,z) plane per block at fixed (b,y), 512 threads, 66.5 KB.
 * HALF=1: half plane (z split), 256 threads, 33.25 KB, 2x blocks so even B=1
 * (128 blocks) covers all 108 SMs. */
template <int HALF>
__global__ void __launch_bounds__(HALF ? 256 : 512, 2) kernel_x(double2 *data,
                                                const double2 *__restrict__ w64)
{
    extern __shared__ double2 sh[]; /* [x][z], row stride ZT+1 */
    const int ZT = HALF ? 32 : 64;
    const int STRIDE = ZT + 1;
    const int NT = HALF ? 256 : 512;
    const int b = blockIdx.x >> (6 + HALF);
    const int y = (blockIdx.x >> HALF) & 63;
    const int zoff = HALF ? ((blockIdx.x & 1) << 5) : 0;
    const size_t base = (size_t)b * 262144 + (size_t)y * 64 + zoff;
    const int tid = threadIdx.x;
    double2 r[8], tw[7];
    load_tw(tw, tid & 7, w64);

    /* coalesced load: ZT consecutive z per x-row, 128B segments */
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int m = tid + NT * j;               /* 0 .. 64*ZT-1 */
        const int x = m / ZT, z = m % ZT;         /* shifts: ZT power of two */
        sh[(size_t)x * STRIDE + z] = __ldlu(&data[base + (size_t)x * 4096 + z]);
    }
    __syncthreads();

    const int t = tid & 7;
    const int g = tid >> 3; /* local z: 0..ZT-1 */
#pragma unroll
    for (int j = 0; j < 8; ++j) r[j] = sh[(size_t)(8 * j + t) * STRIDE + g];
    fft64_reg(r, t, tw);
#pragma unroll
    for (int j = 0; j < 8; ++j) sh[(size_t)(8 * j + t) * STRIDE + g] = r[j];
    __syncthreads();

    /* final result: stream it out coalesced, nothing rereads it */
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int m = tid + NT * j;
        const int x = m / ZT, z = m % ZT;
        __stcs(&data[base + (size_t)x * 4096 + z], sh[(size_t)x * STRIDE + z]);
    }
}

extern "C" const char *fft3d_gpu_name(void) { return "L64_radix8"; }
extern "C" const char *fft3d_gpu_description(void)
{ return "two-pass plane-per-block, register radix-8^2 lines, shuffle transpose"; }
extern "C" int fft3d_gpu_supports(int L) { return L == 64; }

extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)
{
    if (L != 64) return NULL;
    fft3d_gpu_plan *p = (fft3d_gpu_plan *)malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;

    double2 h[64];
    for (int j = 0; j < 64; ++j) {
        /* angle = -pi * j/32: j/32 is exact in binary, so sin/cos are ~0.5 ulp */
        h[j].x = cos(-M_PI * (double)j / 32.0);
        h[j].y = sin(-M_PI * (double)j / 32.0);
    }
    if (cudaMalloc((void **)&p->w64, sizeof h) != cudaSuccess) { free(p); return NULL; }
    cudaMemcpy(p->w64, h, sizeof h, cudaMemcpyHostToDevice);

    cudaFuncSetAttribute(kernel_yz<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         64 * SH_STRIDE * (int)sizeof(double2));
    cudaFuncSetAttribute(kernel_yz<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         64 * SH_STRIDE * (int)sizeof(double2));
    cudaFuncSetAttribute(kernel_x<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         64 * 65 * (int)sizeof(double2));
    cudaFuncSetAttribute(kernel_x<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         64 * 33 * (int)sizeof(double2));

    p->chunk = 3; /* B=256 scan (gpu_r2, staged kernels): C=2 2900, C=3 2622-2642,
                     C=4 2749-2768, C=6 3144 us. Two chunks in flight x 3 volumes
                     x (in+out) well inside L2. */
    for (int i = 0; i < 4; ++i)
        cudaStreamCreateWithFlags(&p->str[i], cudaStreamNonBlocking);
    p->gexec = NULL;
    p->gin = NULL;
    p->gout = NULL;
    {
        const char *e = getenv("L64_CHUNK");
        if (e) p->chunk = atoi(e);
    }
    return p;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    const size_t shbytes = 64 * SH_STRIDE * sizeof(double2);
    const size_t shhalf = 64 * 33 * sizeof(double2);
    const int C = p->chunk;
    if (p->batch == 1) {
        /* Two launches replayed as one captured graph (borrowed from
         * L36_sharedtiled gpu_r1: graph beat plain launches and the fused
         * cooperative kernel at B=1). Keyed on (in,out); recaptured if the
         * driver ever pokes us with different pointers. */
        if (!p->gexec || in != p->gin || out != p->gout) {
            if (p->gexec) { cudaGraphExecDestroy(p->gexec); p->gexec = NULL; }
            cudaGraph_t g;
            cudaStreamBeginCapture(p->str[0], cudaStreamCaptureModeThreadLocal);
            kernel_yz<0><<<64, 512, shbytes, p->str[0]>>>(in, out, p->w64);
            kernel_x<1><<<128, 256, shhalf, p->str[0]>>>(out, p->w64);
            cudaStreamEndCapture(p->str[0], &g);
            cudaGraphInstantiate(&p->gexec, g, 0);
            cudaGraphDestroy(g);
            p->gin = in;
            p->gout = out;
        }
        cudaGraphLaunch(p->gexec, p->str[0]);
        return;
    }
    if (p->batch <= 8) {
        /* One chunk per stream (borrowed from L36_sharedtiled gpu_r2's batch
         * split: pass 2 of one slice fills the tail waves of pass 1 of
         * another; >=2 chunks per stream is worse than unchunked). */
        const int ns = p->batch < 4 ? p->batch : 4;
        const int cs = (p->batch + ns - 1) / ns;
        for (int c0 = 0, c = 0; c0 < p->batch; c0 += cs, ++c) {
            const int nv = p->batch - c0 < cs ? p->batch - c0 : cs;
            const size_t off = (size_t)c0 * 262144;
            cudaStream_t s = p->str[c & 3];
            kernel_yz<0><<<nv * 64, 512, shbytes, s>>>(in + off, out + off, p->w64);
            kernel_x<1><<<nv * 128, 256, shhalf, s>>>(out + off, p->w64);
        }
        return;
    }
    /* L2 blocking: per chunk, pass 2 rereads and rewrites pass 1's output while
     * it is still L2-resident, so only the input read and the final writeback
     * touch HBM. Two streams overlap chunk c+1's pass 1 with chunk c's pass 2.
     * Chunk->stream mapping is fixed, so back-to-back executes stay ordered. */
    for (int c0 = 0, c = 0; c0 < p->batch; c0 += C, ++c) {
        const int nb = (p->batch - c0 < C ? p->batch - c0 : C) * 64;
        const size_t off = (size_t)c0 * 262144;
        cudaStream_t s = p->str[c & 1];
        kernel_yz<1><<<nb, 512, shbytes, s>>>(in + off, out + off, p->w64);
        kernel_x<0><<<nb, 512, 64 * 65 * sizeof(double2), s>>>(out + off, p->w64);
    }
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p)
{
    if (!p) return;
    if (p->gexec) cudaGraphExecDestroy(p->gexec);
    for (int i = 0; i < 4; ++i) cudaStreamDestroy(p->str[i]);
    cudaFree(p->w64);
    free(p);
}
