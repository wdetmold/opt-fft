/* L17_raderfused -- L = 17 on one A100: whole 17^3 volume resident in one block's
 * shared memory, all three axes fused, ONE global read + ONE global write per volume.
 *
 * A 17^3 volume of complex doubles is 78,608 B: 48% of the 163 KB a block may address
 * in shared memory (opt-in required above 48 KB), so one block stages the volume,
 * transforms z, y, x in place in shared, and streams it back out (the x pass writes
 * its results straight to global, coalesced). The volume is contiguous in global
 * memory, so both global phases are perfectly coalesced double2 accesses; the 3D
 * transpose problem lives entirely in shared memory, where L = 17 (odd stride) is
 * naturally bank-conflict-free. 320 threads, one 17-point line per thread per pass,
 * __launch_bounds__(320, 2) so two blocks (157 KB shared) share an SM.
 *
 * Per-line 17-point module (line17w): the CPU panel's winning algebra at L = 17 --
 * conjugate-pair fold to all-real coefficients (../../geom/strategies/L17_matrixsimd.md)
 * plus the primitive-root cyclic/negacyclic split of the cosine half
 * (../../geom/strategies/L17_winograd.md): 496 real flops per line = 87 flop/point,
 * arithmetic intensity 2.7 flop/B against the A100's 6.24 flop/B FP64 balance, so the
 * arithmetic hides under the 32 B/point HBM floor. Rader proper (the entry's name) was
 * analysed and rejected: its FFT16 convolution needs ~160+ live registers per line,
 * which breaks the 2-blocks/SM occupancy for a flop saving that is already free.
 *
 * Below batch 12 a split path runs instead: fft17_planes does z+y of one x-plane per
 * block (B*17 blocks, thread-per-output), then fft17_xlines does x thread-per-line in
 * place on out -- twice the global traffic, but everything is L2-resident there and
 * the fused kernel cannot fill 108 SMs with so few blocks.
 */
#include <cuda_pipeline_primitives.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../fft3d_gpu_api.h"

#define L17 17
#define NPT (17 * 17 * 17) /* 4913 points per volume */
#define NTHREADS 320
#define SMEM_BYTES (NPT * sizeof(double2)) /* 78,608 B */

/* cos/sin(2*pi*k*j/17), k = 1..8 (row), j = 1..8 (col). */
__constant__ double c_cos[8][8];
__constant__ double c_sin[8][8];

/* Tables for the cyclic/negacyclic split (adopted from the CPU phase: derivation in
 * ../../geom/strategies/L17_winograd.md via L17_matrixsimd round 2). With c[r] =
 * cos(2pi 3^r/17), s[r] = sin(2pi 3^r/17):
 *   c_ca[t] = (c[t] + c[t+4])/2                       cyclic-4 kernel, index (m+n)%4
 *   c_cd[t] = (c[t] - c[t+4])/2, c_cd[t+4] = -c_cd[t] negacyclic-4 kernel, index m+n<=6
 *   c_st[t] = s[t] (t<8), -s[t-8] (t>=8)              negacyclic-8 sine, index m+n<=14 */
__constant__ double c_ca[4];
__constant__ double c_cd[7];
__constant__ double c_st[15];

/* One 17-point forward DFT: read the line from src[sbase + t*sstride], write it to
 * dst[dbase + t*dstride], t = 0..16. src/dst may be shared or global; all reads happen
 * before any write, so src == dst in place is fine (the line is private to this thread). */
template <typename S, typename D>
static __device__ __forceinline__ void line17(const S *src, int sbase, int sstride, D *dst,
                                              int dbase, int dstride)
{
    const double2 x0 = src[sbase];
    double ur[8], ui[8], vr[8], vi[8];
#pragma unroll
    for (int j = 1; j <= 8; ++j) {
        const double2 a = src[sbase + j * sstride];
        const double2 b = src[sbase + (17 - j) * sstride];
        ur[j - 1] = a.x + b.x;
        ui[j - 1] = a.y + b.y;
        vr[j - 1] = a.x - b.x;
        vi[j - 1] = a.y - b.y;
    }
    double dr = x0.x, di = x0.y;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        dr += ur[j];
        di += ui[j];
    }
    dst[dbase] = make_double2(dr, di);
#pragma unroll
    for (int k = 1; k <= 8; ++k) {
        double pr = x0.x, pi = x0.y, rr = 0.0, ri = 0.0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const double c = c_cos[k - 1][j];
            const double sn = c_sin[k - 1][j];
            pr = fma(c, ur[j], pr);
            pi = fma(c, ui[j], pi);
            rr = fma(sn, vr[j], rr);
            ri = fma(sn, vi[j], ri);
        }
        /* X_k = P - i*R, X_{17-k} = P + i*R */
        dst[dbase + k * dstride] = make_double2(pr + ri, pi - rr);
        dst[dbase + (17 - k) * dstride] = make_double2(pr - ri, pi + rr);
    }
}

/* Same transform, 496 real flops per line instead of 608: reindex j and k by powers of
 * the primitive root 3 (3^m = sig[m]*f[m] mod 17), which turns the folded 8x8 cosine
 * matrix into a cyclic-8 correlation that splits into cyclic-4 + negacyclic-4 halves,
 * while the sine matrix becomes a sign-decorated negacyclic-8 handled by a 15-entry
 * table. All index arithmetic is compile-time constant after unrolling. */
/* Fold state: everything the emit stage needs; once this is built the source line
 * (shared or global) is no longer referenced, which is what lets the fused kernel
 * overwrite shared memory with the next volume while the x-pass arithmetic runs. */
struct Fold17 {
    double ur[8], ui[8], vr[8], vi[8];
    double2 x0;
};

template <typename S>
static __device__ __forceinline__ void fold17w(const S *src, int sbase, int sstride,
                                               Fold17 &f)
{
    const int F[8] = {1, 3, 8, 7, 4, 5, 2, 6}; /* |3^m mod 17| */
    const int SIG[8] = {1, 1, -1, -1, -1, 1, -1, -1};

    f.x0 = src[sbase];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        const double2 a = src[sbase + F[n] * sstride];
        const double2 b = src[sbase + (17 - F[n]) * sstride];
        f.ur[n] = a.x + b.x;
        f.ui[n] = a.y + b.y;
        if (SIG[n] > 0) {
            f.vr[n] = a.x - b.x;
            f.vi[n] = a.y - b.y;
        } else {
            f.vr[n] = b.x - a.x;
            f.vi[n] = b.y - a.y;
        }
    }
}

template <typename D>
static __device__ __forceinline__ void emit17w(Fold17 &f, D *dst, int dbase, int dstride)
{
    const int F[8] = {1, 3, 8, 7, 4, 5, 2, 6};
    const int SIG[8] = {1, 1, -1, -1, -1, 1, -1, -1};
    const double2 x0 = f.x0;
    double *ur = f.ur, *ui = f.ui, *vr = f.vr, *vi = f.vi;

    double dr = x0.x, di = x0.y;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        dr += ur[n];
        di += ui[n];
    }
    dst[dbase] = make_double2(dr, di);
    /* A/B reduction of u~ in place: A = u~n + u~n+4 in [0..3], B = difference in [4..7] */
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        const double tr = ur[n], ti = ui[n];
        ur[n] = tr + ur[n + 4];
        ui[n] = ti + ui[n + 4];
        ur[n + 4] = tr - ur[n + 4];
        ui[n + 4] = ti - ui[n + 4];
    }
    /* cosine half: P~[m] = x0 + CY[m] + NG[m], P~[m+4] = x0 + CY[m] - NG[m] */
    double cyr[4], cyi[4], ngr[4], ngi[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        double ar = x0.x, ai = x0.y, br = 0.0, bi = 0.0;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            ar = fma(c_ca[(m + n) & 3], ur[n], ar);
            ai = fma(c_ca[(m + n) & 3], ui[n], ai);
            br = fma(c_cd[m + n], ur[n + 4], br);
            bi = fma(c_cd[m + n], ui[n + 4], bi);
        }
        cyr[m] = ar;
        cyi[m] = ai;
        ngr[m] = br;
        ngi[m] = bi;
    }
    /* sine half + output pairs: X_{f[m]} = P~ - i*sig[m]*R~, X_{17-f[m]} the conjugate */
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        double rr = 0.0, ri = 0.0;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            rr = fma(c_st[m + n], vr[n], rr);
            ri = fma(c_st[m + n], vi[n], ri);
        }
        const double pr = (m < 4) ? cyr[m] + ngr[m] : cyr[m - 4] - ngr[m - 4];
        const double pi = (m < 4) ? cyi[m] + ngi[m] : cyi[m - 4] - ngi[m - 4];
        if (SIG[m] > 0) {
            dst[dbase + F[m] * dstride] = make_double2(pr + ri, pi - rr);
            dst[dbase + (17 - F[m]) * dstride] = make_double2(pr - ri, pi + rr);
        } else {
            dst[dbase + F[m] * dstride] = make_double2(pr - ri, pi + rr);
            dst[dbase + (17 - F[m]) * dstride] = make_double2(pr + ri, pi - rr);
        }
    }
}

/* whole line transform = fold + emit */
template <typename S, typename D>
static __device__ __forceinline__ void line17w(const S *src, int sbase, int sstride, D *dst,
                                               int dbase, int dstride)
{
    Fold17 f;
    fold17w(src, sbase, sstride, f);
    emit17w(f, dst, dbase, dstride);
}

/* Module selection for the per-line kernel: default is the split module; -DL17RF_DENSE
 * reverts to the plain folded dense matvec for A/B runs. */
template <typename S, typename D>
static __device__ __forceinline__ void line17sel(const S *src, int sbase, int sstride,
                                                 D *dst, int dbase, int dstride)
{
#ifdef L17RF_DENSE
    line17(src, sbase, sstride, dst, dbase, dstride);
#else
    line17w(src, sbase, sstride, dst, dbase, dstride);
#endif
}

/* The volume is staged into shared with a coalesced unrolled loop; the global store
 * IS fused into the x pass: output index tid + k*289 means for each k the warp writes
 * 32 consecutive double2 -- perfectly coalesced. Two structures that were measured and
 * REJECTED (numbers in the strategy record): fusing the load into the z pass (strided
 * reads defeat the 32 B sectors once L1 shrinks to 28 KB under the max carveout), and a
 * persistent cp.async pipeline prefetching volume v+gridDim.x behind the x pass (the
 * fold state held across the barrier spills at 2 blocks/SM, and at 1 block/SM the lost
 * co-residency costs more than the hidden load buys). */
extern "C" __global__ void __launch_bounds__(NTHREADS, 2)
    fft17_fused(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    extern __shared__ double2 s[];
    const size_t vbase = (size_t)blockIdx.x * NPT;
    const int tid = threadIdx.x;

    /* Staging load, fully unrolled: 15 guaranteed rounds (15*320 = 4800 < 4913) issue
     * as independent loads, so one block keeps ~75 KB in flight instead of chaining
     * load->store->load at one round of latency each. */
    {
        double2 t[15];
#pragma unroll
        for (int r = 0; r < 15; ++r) t[r] = in[vbase + tid + r * NTHREADS];
        const int itail = tid + 15 * NTHREADS;
        double2 tt;
        if (itail < NPT) tt = in[vbase + itail];
#pragma unroll
        for (int r = 0; r < 15; ++r) s[tid + r * NTHREADS] = t[r];
        if (itail < NPT) s[itail] = tt;
    }
    __syncthreads();

    /* z axis: line (x,y), base (x*17+y)*17, elements contiguous, in place in shared */
    if (tid < 289) line17sel(s, tid * 17, 1, s, tid * 17, 1);
    __syncthreads();

    /* y axis: line (x,z), base x*289 + z, stride 17, in place in shared */
    if (tid < 289) {
        const int b = (tid / 17) * 289 + (tid % 17);
        line17sel(s, b, 17, s, b, 17);
    }
    __syncthreads();

    /* x axis: line (y,z), base y*17 + z = tid, stride 289; results straight to global */
    if (tid < 289) line17sel(s, tid, 289, out + vbase, tid, 289);
}

/* ---- small-batch path: split passes across many small blocks ----------------------
 * The fused kernel gives batch blocks, so at B=1 it uses one SM of 108 and B=1 is pure
 * kernel latency. Below a batch cut the plan instead runs two kernels: P does the z and
 * y axes of one 17x17 x-plane per block (B*17 blocks, thread-per-OUTPUT, ~30 registers,
 * so ~6 blocks/SM and near-full occupancy), then X does the x axis thread-per-line, in
 * place on `out`. Twice the global traffic of the fused kernel (4 volume passes instead
 * of 2), but in this regime everything is L2-resident and latency is what is scored. */

/* Row k of the full 17-row coefficient tables: gcos[k][j] = cos(2pi k j/17), gsin
 * likewise. Row 0 is {1,0}: the same P -/+ iR formula then yields X_0 with no branch.
 * Rows k and 17-k share cosines and negate sines, so the table handles the conjugate
 * output pair automatically. Plain __device__ arrays (not __constant__): k diverges
 * across the warp and the constant cache serializes divergent addresses. */
__device__ double g_cos17[17][8];
__device__ double g_sin17[17][8];

extern "C" __global__ void __launch_bounds__(320, 4)
    fft17_planes(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    __shared__ double2 s[289];
    const int tid = threadIdx.x;
    const size_t pbase = (size_t)blockIdx.x * 289; /* plane (v,x) = contiguous 289 */

    if (tid < 289) s[tid] = in[pbase + tid];
    __syncthreads();

    /* k = tid/17 so the coefficient row is warp-uniform (a warp spans at most 2 k
     * values): the in-loop __ldg is then 1-2 addresses per warp, not 17. The price is
     * a mild 2-way shared bank conflict on the z-pass line reads; measured cheaper
     * than either divergent coefficient loads or spilled per-thread arrays. */
    const int k = tid / 17;  /* which output of the line */
    const int ln = tid % 17; /* which line of the plane */

    /* z pass: line y = ln, elements s[ln*17 + j] */
    double2 r0;
    if (tid < 289) {
        const int base = ln * 17;
        double pr = s[base].x, pi = s[base].y, rr = 0.0, ri = 0.0;
#pragma unroll
        for (int j = 1; j <= 8; ++j) {
            const double2 a = s[base + j];
            const double2 b = s[base + 17 - j];
            pr = fma(__ldg(&g_cos17[k][j - 1]), a.x + b.x, pr);
            pi = fma(__ldg(&g_cos17[k][j - 1]), a.y + b.y, pi);
            rr = fma(__ldg(&g_sin17[k][j - 1]), a.x - b.x, rr);
            ri = fma(__ldg(&g_sin17[k][j - 1]), a.y - b.y, ri);
        }
        r0 = make_double2(pr + ri, pi - rr); /* X_k = P - i*R */
    }
    __syncthreads();
    if (tid < 289) s[ln * 17 + k] = r0;
    __syncthreads();

    /* y pass: line z = ln, elements s[j*17 + ln] */
    if (tid < 289) {
        double pr = s[ln].x, pi = s[ln].y, rr = 0.0, ri = 0.0;
#pragma unroll
        for (int j = 1; j <= 8; ++j) {
            const double2 a = s[j * 17 + ln];
            const double2 b = s[(17 - j) * 17 + ln];
            pr = fma(__ldg(&g_cos17[k][j - 1]), a.x + b.x, pr);
            pi = fma(__ldg(&g_cos17[k][j - 1]), a.y + b.y, pi);
            rr = fma(__ldg(&g_sin17[k][j - 1]), a.x - b.x, rr);
            ri = fma(__ldg(&g_sin17[k][j - 1]), a.y - b.y, ri);
        }
        r0 = make_double2(pr + ri, pi - rr);
    }
    __syncthreads();
    /* stage through shared so the global store is contiguous, not stride-17 */
    if (tid < 289) s[k * 17 + ln] = r0;
    __syncthreads();
    if (tid < 289) out[pbase + tid] = s[tid];
}

#define XLINE_T 128
extern "C" __global__ void __launch_bounds__(XLINE_T, 4)
    fft17_xlines(double2 *__restrict__ out, int nlines)
{
    const int q = blockIdx.x * XLINE_T + threadIdx.x;
    if (q >= nlines) return;
    const int v = q / 289, p = q % 289;
    double2 *vol = out + (size_t)v * NPT;
    line17sel(vol, p, 289, vol, p, 289); /* in place; lines are disjoint per thread */
}

struct fft3d_gpu_plan {
    int L;
    int batch;
    int split; /* 1: planes+xlines path, 0: fused single kernel */
};

extern "C" const char *fft3d_gpu_name(void) { return "L17_raderfused"; }
extern "C" const char *fft3d_gpu_description(void)
{
    return "17^3 volume per block in shared, 3 axes fused, 1 global read + 1 write; "
           "conj-folded cyclic/negacyclic 17-pt lines; plane+line split path below B=12";
}

extern "C" int fft3d_gpu_supports(int L) { return L == L17; }

/* Below this batch the split (planes + xlines) path wins: the fused kernel cannot fill
 * the machine with fewer than ~216 blocks, while everything is still L2-resident so the
 * split path's doubled traffic is cheap. Set from measurement; override with -DL17RF_CUT,
 * and -DL17RF_FORCE=1 (always fused) / =2 (always split) for A/B runs. */
#ifndef L17RF_CUT
#define L17RF_CUT 12
#endif

extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)
{
    if (L != L17 || batch < 1) return NULL;

    double hc[8][8], hs[8][8], gc[17][8], gs[17][8];
    for (int k = 0; k <= 16; ++k)
        for (int j = 1; j <= 8; ++j) {
            /* long double keeps the table good to ~1e-19; the 17-term sums then land
             * near 3e-16 relative, same as the CPU phase. */
            long double th = 2.0L * 3.14159265358979323846264338327950288L *
                             (long double)(k * j) / 17.0L;
            gc[k][j - 1] = (double)cosl(th);
            gs[k][j - 1] = (double)sinl(th);
            if (k >= 1 && k <= 8) {
                hc[k - 1][j - 1] = gc[k][j - 1];
                hs[k - 1][j - 1] = gs[k][j - 1];
            }
        }
    if (cudaMemcpyToSymbol(c_cos, hc, sizeof hc) != cudaSuccess) return NULL;
    if (cudaMemcpyToSymbol(c_sin, hs, sizeof hs) != cudaSuccess) return NULL;
    if (cudaMemcpyToSymbol(g_cos17, gc, sizeof gc) != cudaSuccess) return NULL;
    if (cudaMemcpyToSymbol(g_sin17, gs, sizeof gs) != cudaSuccess) return NULL;

    /* tables for the cyclic/negacyclic split module */
    {
        const int pow3[8] = {1, 3, 9, 10, 13, 5, 15, 11}; /* 3^r mod 17 */
        long double cc[8], ss[8];
        double ca[4], cd[7], st[15];
        for (int r = 0; r < 8; ++r) {
            long double th = 2.0L * 3.14159265358979323846264338327950288L *
                             (long double)pow3[r] / 17.0L;
            cc[r] = cosl(th);
            ss[r] = sinl(th);
        }
        for (int t = 0; t < 4; ++t) {
            ca[t] = (double)((cc[t] + cc[t + 4]) * 0.5L);
            cd[t] = (double)((cc[t] - cc[t + 4]) * 0.5L);
        }
        for (int t = 4; t < 7; ++t) cd[t] = -cd[t - 4];
        for (int t = 0; t < 8; ++t) st[t] = (double)ss[t];
        for (int t = 8; t < 15; ++t) st[t] = (double)-ss[t - 8];
        if (cudaMemcpyToSymbol(c_ca, ca, sizeof ca) != cudaSuccess) return NULL;
        if (cudaMemcpyToSymbol(c_cd, cd, sizeof cd) != cudaSuccess) return NULL;
        if (cudaMemcpyToSymbol(c_st, st, sizeof st) != cudaSuccess) return NULL;
    }

    if (cudaFuncSetAttribute(fft17_fused, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)SMEM_BYTES) != cudaSuccess)
        return NULL;
    /* ask for the full shared carveout so two 78.6 KB blocks co-reside per SM */
    cudaFuncSetAttribute(fft17_fused, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);

    fft3d_gpu_plan *p = (fft3d_gpu_plan *)malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->split = (batch < L17RF_CUT);
#if defined(L17RF_FORCE) && L17RF_FORCE == 1
    p->split = 0;
#elif defined(L17RF_FORCE) && L17RF_FORCE == 2
    p->split = 1;
#endif
    return p;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (p->split) {
        const int nlines = 289 * p->batch;
        fft17_planes<<<p->batch * L17, 320>>>(in, out);
        fft17_xlines<<<(nlines + XLINE_T - 1) / XLINE_T, XLINE_T>>>(out, nlines);
    } else {
        fft17_fused<<<p->batch, NTHREADS, SMEM_BYTES>>>(in, out);
    }
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p) { free(p); }
