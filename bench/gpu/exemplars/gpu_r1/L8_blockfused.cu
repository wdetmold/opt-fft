/* L8_blockfused -- L = 8 on one A100.
 *
 * Technique (round gpu_r1): fused single kernel, V volumes of 8^3 per block staged in
 * shared memory, one thread per 8-point line (64 threads per volume), three axis passes
 * separated by barriers, one global read + one global write per volume.
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
 * V is autotuned in create() on scratch buffers (measured, per batch); B = 1 uses a
 * dedicated staging-free kernel whose z-pass reads each thread's contiguous 128-byte
 * z-line straight from global (latency case: two barriers instead of three, no staging).
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
 * an odd (9) or unit lane stride, so all shared traffic is conflict-free. */
template <int V, int MINB, bool STCS, bool LOADZ, bool CPA = false>
__global__ void __launch_bounds__(V * 64, MINB)
fft8_fused(const double2 *__restrict__ in, double2 *__restrict__ out, int B)
{
    extern __shared__ double2 s[];
    const int  tid = threadIdx.x;
    const int  v   = tid >> 6;
    const int  t   = tid & 63;
    const long vb  = (long)blockIdx.x * V;
    const double2 *gin = in + vb * 512;
    double2 *sv = s + v * SVOL;
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
    __syncthreads();
    if (act) {                                                      /* y: stride 9  */
        double2 *p = sv + (t >> 3) * (8 * SROW) + (t & 7);
        dft8<SROW, SROW>(p, p);
    }
    __syncthreads();
    if (act)                                                        /* x: stride 72, write global */
        dft8<8 * SROW, 64, STCS>(sv + (t >> 3) * SROW + (t & 7),
                                 out + (vb + v) * 512 + t);
}

/* B = 1 latency kernel: one block, 64 threads, no staging pass. The z-pass reads each
 * thread's contiguous 128 B z-line straight from global (coalescing is irrelevant for
 * one 8 KB volume; the win is one less barrier and no staging round trip). */
__global__ void __launch_bounds__(64)
fft8_single(const double2 *__restrict__ in, double2 *__restrict__ out)
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

/* ---- host side ------------------------------------------------------------------- */

struct fft3d_gpu_plan {
    int L, batch;
    int kind;   /* 0 = fused with V = vsel, 1 = single-volume kernel (B == 1) */
    int vsel;
    int stcs;
    int mode;
    int grid;
    int smem;
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

extern "C" const char *fft3d_gpu_name(void) { return "L8_blockfused"; }
extern "C" const char *fft3d_gpu_description(void)
{ return "fused block of V volumes in shared (row pad 8->9), min-op radix-8 per thread-line, V autotuned"; }
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
    for (int i = 0; i < 4; ++i)
        for (int st = 0; st < 2; ++st)
            for (int m = 0; m < 3; ++m) {
                kfun_t k = kern_for(VS[i], st, m);
                cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     VS[i] * SVOL * (int)sizeof(double2));
                cudaFuncSetAttribute(k, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
            }
    cudaFuncSetAttribute(fft8_single, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    fft3d_gpu_plan *p = (fft3d_gpu_plan *)malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->kind = 0;
    p->vsel = 4;
    p->stcs = 0;
    p->mode = 0;

    const char *fs = getenv("L8_FORCE_SINGLE");
    const char *fv = getenv("L8_FORCE_V");
    const char *ft = getenv("L8_FORCE_STCS");
    const char *fz = getenv("L8_FORCE_LOADZ");
    const char *fc = getenv("L8_FORCE_CPA");
    if (ft) p->stcs = atoi(ft);
    if (fz && atoi(fz)) p->mode = 2;
    if (fc && atoi(fc)) p->mode = 1;
    if (batch == 1 && !fv) p->kind = 1;         /* default at B = 1 */
    if (fs && atoi(fs)) { p->kind = 1; }
    else if (fv && atoi(fv) > 0) { p->kind = 0; p->vsel = atoi(fv); }
    else if (batch > 1) {
        /* measured (V, stcs, cp.async) selection on scratch buffers (excluded from
         * execute); the loadz mode is not tuned over -- it measured 5% worse at B_HBM
         * and 33% worse at B_L2 than the best staged variant */
        double2 *a = 0, *b = 0;
        size_t bytes = (size_t)batch * 512 * sizeof(double2);
        if (cudaMalloc(&a, bytes) == cudaSuccess && cudaMalloc(&b, bytes) == cudaSuccess) {
            cudaMemset(a, 0, bytes);
            /* enough reps that one timing is a few ms even at large B; 3 round-robin
             * cycles so clock ramp does not bias the first candidate; keep the min */
            int reps = (int)(60000000L / ((long)batch * 512) + 3);
            if (reps > 400) reps = 400;
            float best[16];
            for (int i = 0; i < 16; ++i) best[i] = 1e30f;
            for (int cyc = 0; cyc < 3; ++cyc)
                for (int i = 0; i < 16; ++i) {
                    const int V = VS[i & 3], st = (i >> 2) & 1, m = i >> 3;
                    const int grid = (batch + V - 1) / V;
                    float ms = time_candidate(kern_for(V, st, m), grid, V * 64,
                                              V * SVOL * (int)sizeof(double2), a, b,
                                              batch, reps);
                    if (ms < best[i]) best[i] = ms;
                }
            int ibest = 0;
            for (int i = 1; i < 16; ++i)
                if (best[i] < best[ibest]) ibest = i;
            p->vsel = VS[ibest & 3];
            p->stcs = (ibest >> 2) & 1;
            p->mode = ibest >> 3;
        }
        if (a) cudaFree(a);
        if (b) cudaFree(b);
        cudaGetLastError(); /* clear any autotune-time error state */
    }

    p->grid = (batch + p->vsel - 1) / p->vsel;
    p->smem = p->vsel * SVOL * (int)sizeof(double2);
    return p;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (p->kind == 1) {
        fft8_single<<<1, 64>>>(in, out);
        return;
    }
    kern_for(p->vsel, p->stcs, p->mode)<<<p->grid, p->vsel * 64, p->smem>>>(in, out, p->batch);
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p) { free(p); }
