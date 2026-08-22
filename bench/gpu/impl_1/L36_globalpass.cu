/* L36_globalpass -- two bandwidth-optimal global passes for L = 36 on one A100.
 *
 * The entry began life as "three coalesced global passes"; the corpus (09-gpu §9.6) shows
 * two passes is achievable with the same perfect coalescing, so that is what this is:
 *
 *   Kernel 1 (one block per (b,x) plane):  the 36x36 (y,z) plane at fixed (b,x) is a
 *     contiguous 1296-complex block (20.25 KiB).  Load it into shared memory with the row
 *     stride padded 36 -> 37 (gcd(36,8)=4: unpadded column reads are a 4-way bank conflict
 *     that eats exactly the shared-memory headroom, 09-gpu §6.2), transform z then y in
 *     shared, write back.  Reads and writes are straight streams of the volume.
 *   Kernel 2 (one block per (b,y) slab):   the x axis.  The slab {x,z} at fixed (b,y) is
 *     36 rows of 36 contiguous complex (576 B = 18 full 32-byte transactions, zero waste),
 *     strided 20736 B apart.  Load to shared (stride 37), transform x, write back in place.
 *
 * The 36-point line is two radix-6 stages (36 = 6*6 Cooley-Tukey, DFT-6 = 2 x DFT-3);
 * arithmetic is 2.6x under the bandwidth floor at this size so the codelet just has to not
 * be a bottleneck.  Stage twiddles W36^(j1*k1) live in constant memory.
 *
 * L2 chunking across the batch (09-gpu §2.4 / §9.6 structure 2): for large B the kernel
 * pair is launched per chunk of volumes, so kernel 2 reads what kernel 1 just wrote while
 * it is still in the 40 MiB L2 (7.2 TB/s, 4.6x HBM).  The read-once input uses __ldcs and
 * the write-once final output __stcs so the streaming traffic does not evict the hot
 * intermediate.  Ideal effect: HBM traffic drops from 64 to ~32 bytes/point.
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

#include "../fft3d_gpu_api.h"

#define LDIM   36
#define PLANE  1296              /* 36*36 */
#define VOL    46656L            /* 36^3 */
#define SSTRIDE 37               /* padded shared row stride */
#define NTHREADS 216             /* 36 lines * 6 groups; also 1296/6 for the copy loops */

/* -sqrt(3)/2: the imaginary part of exp(-2*pi*i/3) and exp(-2*pi*i/6). */
#define S3 (-0.8660254037844386467637231707529362)

/* W36^(j1*k1), j1,k1 in [0,6), indexed [j1*6 + k1]. */
static __constant__ double2 c_tw36[36];

struct fft3d_gpu_plan {
    int L; long batch; int chunk;
    long nchunks;
    cudaStream_t sA, sB;      /* pass-1 and pass-2 pipelines */
    cudaEvent_t *evA, *evB;   /* per-chunk: K1 done / K2 done */
};

static __device__ __forceinline__ double2 cadd(double2 a, double2 b)
{ return make_double2(a.x + b.x, a.y + b.y); }
static __device__ __forceinline__ double2 csub(double2 a, double2 b)
{ return make_double2(a.x - b.x, a.y - b.y); }
static __device__ __forceinline__ double2 cmul(double2 a, double2 w)
{ return make_double2(fma(a.x, w.x, -(a.y * w.y)), fma(a.x, w.y, a.y * w.x)); }

static __device__ __forceinline__ void dft3(double2 x0, double2 x1, double2 x2,
                                            double2 &y0, double2 &y1, double2 &y2)
{
    double2 t = cadd(x1, x2), u = csub(x1, x2);
    y0 = cadd(x0, t);
    double2 v = make_double2(x0.x - 0.5 * t.x, x0.y - 0.5 * t.y);
    /* X1 = v + i*S3*u,  X2 = v - i*S3*u */
    y1 = make_double2(v.x - S3 * u.y, v.y + S3 * u.x);
    y2 = make_double2(v.x + S3 * u.y, v.y - S3 * u.x);
}

/* In-place 6-point DFT: a[k] = sum_j a[j] W6^(jk).  Even/odd split into two DFT-3s. */
static __device__ __forceinline__ void dft6(double2 a[6])
{
    double2 e0, e1, e2, o0, o1, o2;
    dft3(a[0], a[2], a[4], e0, e1, e2);
    dft3(a[1], a[3], a[5], o0, o1, o2);
    const double2 W1 = make_double2(0.5, S3);   /* W6^1 */
    const double2 W2 = make_double2(-0.5, S3);  /* W6^2 */
    double2 t1 = cmul(o1, W1), t2 = cmul(o2, W2);
    a[0] = cadd(e0, o0); a[3] = csub(e0, o0);
    a[1] = cadd(e1, t1); a[4] = csub(e1, t1);
    a[2] = cadd(e2, t2); a[5] = csub(e2, t2);
}

/* 36-point DFT of all 36 lines of one shared plane, executed by 216 threads.
 * Thread (g, l): l = line index, g = which of the 6 six-point DFTs per stage.
 * Line l occupies elements  lb(l) + es*idx, idx = 0..35.
 * Two radix-6 stages: X[k1+6*k2] = sum_j1 W6^(j1*k2) * [ W36^(j1*k1) *
 *                                   sum_j2 x[6*j2+j1] W6^(j2*k1) ].
 * Stage 1 stores A'[j1][k1] at digit-transposed slot 6*k1+j1; stage 2 reads a
 * contiguous group and scatters to natural order k1+6*k2.  Both stages read all
 * six values into registers, sync, then write -- slots overlap across threads. */
static __device__ __forceinline__ void fft36_lines(double2 *s, int lb, int es, int g)
{
    double2 a[6];
    /* stage 1: this thread's j1 = g */
#pragma unroll
    for (int j2 = 0; j2 < 6; ++j2) a[j2] = s[lb + es * (6 * j2 + g)];
    dft6(a);
#pragma unroll
    for (int k1 = 1; k1 < 6; ++k1) a[k1] = cmul(a[k1], c_tw36[g * 6 + k1]);
    __syncthreads();
#pragma unroll
    for (int k1 = 0; k1 < 6; ++k1) s[lb + es * (6 * k1 + g)] = a[k1];
    __syncthreads();
    /* stage 2: this thread's k1 = g */
#pragma unroll
    for (int j1 = 0; j1 < 6; ++j1) a[j1] = s[lb + es * (6 * g + j1)];
    dft6(a);
    __syncthreads();
#pragma unroll
    for (int k2 = 0; k2 < 6; ++k2) s[lb + es * (g + 6 * k2)] = a[k2];
    __syncthreads();
}

/* Pass 1: one block per (b,x) plane; z then y in shared. */
__global__ void __launch_bounds__(NTHREADS, 4)
k_zy(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    __shared__ double2 s[LDIM * SSTRIDE];
    const size_t base = (size_t)blockIdx.x * PLANE;
    const int tid = threadIdx.x;
#pragma unroll
    for (int k = 0; k < 6; ++k) {
        int i = tid + k * NTHREADS;
        int y = i / LDIM, z = i - LDIM * y;
        s[y * SSTRIDE + z] = __ldcs(in + base + i);   /* read-once: don't pollute L2 */
    }
    __syncthreads();
    const int g = tid / LDIM, l = tid - LDIM * g;
    fft36_lines(s, l * SSTRIDE, 1, g);   /* z: contiguous lines */
    fft36_lines(s, l, SSTRIDE, g);       /* y: stride-37 lines (odd -> conflict-free) */
#pragma unroll
    for (int k = 0; k < 6; ++k) {
        int i = tid + k * NTHREADS;
        int y = i / LDIM, z = i - LDIM * y;
        out[base + i] = s[y * SSTRIDE + z];  /* default policy: stays L2-hot for pass 2 */
    }
}

/* Pass 2: one block per (b,y) slab; x in shared, in place on out. */
__global__ void __launch_bounds__(NTHREADS, 4)
k_x(double2 *__restrict__ io)
{
    __shared__ double2 s[LDIM * SSTRIDE];
    const int by = blockIdx.x;                 /* = (chunk-local b)*36 + y */
    const int b = by / LDIM, y = by - LDIM * b;
    const size_t base = (size_t)b * VOL + (size_t)y * LDIM;
    const int tid = threadIdx.x;
#pragma unroll
    for (int k = 0; k < 6; ++k) {
        int i = tid + k * NTHREADS;
        int x = i / LDIM, z = i - LDIM * x;
        s[x * SSTRIDE + z] = io[base + (size_t)x * PLANE + z];
    }
    __syncthreads();
    const int g = tid / LDIM, l = tid - LDIM * g;
    fft36_lines(s, l, SSTRIDE, g);       /* x: stride-37 lines, line index = z */
#pragma unroll
    for (int k = 0; k < 6; ++k) {
        int i = tid + k * NTHREADS;
        int x = i / LDIM, z = i - LDIM * x;
        __stcs(io + base + (size_t)x * PLANE + z, s[x * SSTRIDE + z]); /* write-once */
    }
}

extern "C" const char *fft3d_gpu_name(void) { return "L36_globalpass"; }
extern "C" const char *fft3d_gpu_description(void)
{ return "two-pass plane-per-block radix-6^2, padded shared stride 37, L2-chunked batch"; }

extern "C" int fft3d_gpu_supports(int L) { return L == 36; }

extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)
{
    if (L != LDIM) return NULL;

    double2 h[36];
    for (int j = 0; j < 6; ++j)
        for (int k = 0; k < 6; ++k) {
            double phase = -2.0 * M_PI * (double)(j * k) / 36.0;
            h[j * 6 + k].x = cos(phase);
            h[j * 6 + k].y = sin(phase);
        }
    if (cudaMemcpyToSymbol(c_tw36, h, sizeof h) != cudaSuccess) return NULL;

    /* 21312 B of static shared per block; ask for the 100 KB carveout so 4 blocks/SM fit. */
    cudaFuncSetAttribute(k_zy, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    cudaFuncSetAttribute(k_x, cudaFuncAttributePreferredSharedMemoryCarveout, 50);

    fft3d_gpu_plan *p = (fft3d_gpu_plan *)malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    /* Chunk so a pipelined pair of chunks stays L2-resident: two chunks of 16 volumes
     * are 23.3 MiB hot against 40 MiB L2, leaving room for the in/out streams.
     * Batches at or below B_L2 (=22) are L2-resident anyway -- no chunking. */
    p->chunk = (batch > 32) ? 16 : 0;
    p->nchunks = 0;
    p->evA = p->evB = NULL;
    if (p->chunk > 0 && batch > p->chunk) {
        p->nchunks = ((long)batch + p->chunk - 1) / p->chunk;
        cudaStreamCreate(&p->sA);
        cudaStreamCreate(&p->sB);
        p->evA = (cudaEvent_t *)malloc(2 * p->nchunks * sizeof(cudaEvent_t));
        p->evB = p->evA + p->nchunks;
        for (long i = 0; i < 2 * p->nchunks; ++i)
            cudaEventCreateWithFlags(&p->evA[i], cudaEventDisableTiming);
    }
    return p;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    const long B = p->batch;
    const int C = p->chunk;
    if (C <= 0 || B <= C) {
        k_zy<<<(unsigned)(B * LDIM), NTHREADS>>>(in, out);
        k_x<<<(unsigned)(B * LDIM), NTHREADS>>>(out);
        return;
    }
    /* Two-stream pipeline: K1 of chunk c+1 overlaps K2 of chunk c, so the SMs never
     * drain between the dependent halves of a chunk.  evB also orders K2 of the
     * previous execute() against K1 of the next on the same range (in-place hazard);
     * waiting on a never-recorded event is a documented no-op, so the first call is fine. */
    long ci = 0;
    for (long c0 = 0; c0 < B; c0 += C, ++ci) {
        long c = (B - c0 < C) ? (B - c0) : C;
        cudaStreamWaitEvent(p->sA, p->evB[ci], 0);
        k_zy<<<(unsigned)(c * LDIM), NTHREADS, 0, p->sA>>>(in + c0 * VOL, out + c0 * VOL);
        cudaEventRecord(p->evA[ci], p->sA);
        cudaStreamWaitEvent(p->sB, p->evA[ci], 0);
        k_x<<<(unsigned)(c * LDIM), NTHREADS, 0, p->sB>>>(out + c0 * VOL);
        cudaEventRecord(p->evB[ci], p->sB);
    }
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p)
{
    if (!p) return;
    if (p->evA) {
        for (long i = 0; i < 2 * p->nchunks; ++i) cudaEventDestroy(p->evA[i]);
        free(p->evA);
        cudaStreamDestroy(p->sA);
        cudaStreamDestroy(p->sB);
    }
    free(p);
}
