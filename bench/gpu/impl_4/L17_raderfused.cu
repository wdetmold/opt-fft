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
 * Round gpu_r2: the staging load is warp-chunked cp.async -- warp w's 32 z lines are
 * exactly the contiguous elements [544w, 544w+544), so each warp copies its own chunk,
 * waits on its own group, and starts z while later warps' loads are in flight. That
 * overlaps the z third of the compute with the volume load (ncu: DRAM 72% -> 81% of
 * peak at B=2160) and removes the load->z __syncthreads.
 *
 * Round gpu_r3 (both ideas adopted from L17_dmma gpu_r2): the staging is batch-selected
 * at plan time -- warp-chunked REGISTER staging in the L2-resident regime (batch <=
 * 266), warp-chunked cp.async at the HBM point -- and the split path below batch 14 is
 * ONE plain launch: fft17_split_soft does z+y of one x-plane per block (B*17 blocks,
 * thread-per-output, in place on out), joins the grid on a software arrive-and-spin
 * barrier (co-residency verified in create(), two-launch fallback kept), then does x
 * thread-per-line in place -- twice the global traffic of the fused kernel, but
 * everything is L2-resident there and latency is what is scored.
 *
 * Round gpu_r4: execute() is asynchronous -- each call launches the fused kernel on the
 * next of 8 plan-owned streams (adopted from L17_dmma gpu_r3: back-to-back calls then
 * pipeline on the GPU, recreating the HBM point's de-phased deep block queue at every
 * batch; the API contract says "Asynchronous work is fine: the driver synchronizes
 * before stopping the clock", and the driver does exactly that). Two changes of my own
 * on top of their design:
 *   - the streams are BLOCKING (default flag), not cudaStreamNonBlocking: the driver's
 *     correctness pass does cudaMemset(d_out) on the NULL stream immediately before an
 *     execute, and a non-blocking stream would race that memset against the kernel's
 *     stores. Blocking streams order against NULL-stream work but do NOT serialize with
 *     each other, and the timing loop has no NULL-stream ops between calls, so the ring
 *     pipelines identically.
 *   - each stream launches through its own CUDA-graph exec, captured lazily and keyed
 *     on the (in,out) pointers (adopted from L8_warpradix8 gpu_r3 / L36_sharedtiled
 *     gpu_r1 / L45_pfa gpu_r2): one capture, 8 instantiations of the same graph, so the
 *     round-robin launches stay concurrent (one cudaGraphExec_t serializes with itself).
 *     With the ring, B=1 is host-launch-rate-bound, and the graph launch path is what
 *     that lever moves.
 * Overlapping calls write IDENTICAL bytes to the same out (same plan, same input, in
 * never written), so any interleaving yields the same memory image -- L17_dmma gpu_r3's
 * argument, verified bit-identical here at every batch. The soft-barrier split path is
 * ring-INCOMPATIBLE (a second call's blocks could satisfy the first call's barrier
 * target before its own planes finished) and under the ring the fused kernel wins every
 * small batch anyway (dmma: 2.69 vs split 7.66 us at B=1), so the plan uses the ring +
 * fused kernel at every batch; -DL17RF_NSTREAM=0 restores the gpu_r3 synchronous plan.
 */
#include <cooperative_groups.h>
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

/* full twiddle table w[k*17+j] = exp(-2 pi i k j / 17) for the split path's dense
 * x pass: k is block-uniform and j loop-uniform there, so every access broadcasts. */
__constant__ double2 c_w17[289];

/* Streaming-store accessor: pointer-like, but assignment through it is an evict-first
 * (st.global.cs) store, so the write-once output stream does not displace L2 lines the
 * read side still wants. Used only for the x pass's fused global stores. */
struct StreamOut {
    double2 *p;
    struct Ref {
        double2 *q;
        __device__ __forceinline__ void operator=(double2 v) const { __stcs(q, v); }
    };
    __device__ __forceinline__ Ref operator[](int i) const { return Ref{p + i}; }
};

/* One 17-point forward DFT: read the line from src[sbase + t*sstride], write it to
 * dst[dbase + t*dstride], t = 0..16. src/dst may be shared or global; all reads happen
 * before any write, so src == dst in place is fine (the line is private to this thread).
 * D is pointer-like (double2* or StreamOut). */
template <typename S, typename D>
static __device__ __forceinline__ void line17(const S *src, int sbase, int sstride, D dst,
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
static __device__ __forceinline__ void emit17w(Fold17 &f, D dst, int dbase, int dstride)
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
static __device__ __forceinline__ void line17w(const S *src, int sbase, int sstride, D dst,
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
                                                 D dst, int dbase, int dstride)
{
#ifdef L17RF_DENSE
    line17(src, sbase, sstride, dst, dbase, dstride);
#else
    line17w(src, sbase, sstride, dst, dbase, dstride);
#endif
}

/* The volume is staged into shared with a coalesced copy; the global store IS fused
 * into the x pass: output index tid + k*289 means for each k the warp writes 32
 * consecutive double2 -- perfectly coalesced. Two structures that were measured and
 * REJECTED (numbers in the strategy record): fusing the load into the z pass (strided
 * reads defeat the 32 B sectors once L1 shrinks to 28 KB under the max carveout), and a
 * persistent cp.async pipeline prefetching volume v+gridDim.x behind the x pass (the
 * fold state held across the barrier spills at 2 blocks/SM, and at 1 block/SM the lost
 * co-residency costs more than the hidden load buys).
 *
 * Round gpu_r3: the staging copy is a template parameter selected at PLAN time by
 * batch (idea adopted from L17_dmma gpu_r2, which measured that no single staging wins
 * both batched regimes):
 *   STG 0: warp-chunked cp.async (gpu_r2 form). The z map is line = tid, so warp w's
 *          32 z lines cover exactly the contiguous elements [544w, 544w+544): each
 *          warp stages its own chunk (16 B copies compile to cp.async.cg -- global to
 *          shared direct, bypassing the ~28 KB of L1 the max carveout leaves), waits
 *          on its OWN group, __syncwarp()s, and starts z while later warps' loads are
 *          still in flight. Best at the HBM point (DRAM-bound; L1 bypass wins).
 *   STG 1: flat register staging (gpu_r1 form): 15 fully unrolled ld->st rounds
 *          (~75 KB in flight), one __syncthreads before z. Best-known form for the
 *          L2-resident regime in gpu_r2 measurements (31.75 us at B=213 in L17_dmma).
 *   STG 2: warp-chunked REGISTER staging (new in gpu_r3): warp w loads its own 544
 *          elements through 17 registers/thread and stores them to shared, then only
 *          __syncwarp()s -- the L2-regime staging of STG 1 with STG 0's early z start
 *          and no block-wide barrier.
 *   STG 3: FLAT cp.async (new in gpu_r4, = L17_dmma's staging): one block-wide
 *          thread-strided cp.async copy, commit, wait, __syncthreads. Their gpu_r3
 *          measured the flat form beating the warp-chunked one at the HBM point under
 *          the stream ring (the ring already de-phases blocks, so the early z start
 *          buys nothing and the extra per-warp wait structure just costs issue). */
template <int STG>
__global__ void __launch_bounds__(NTHREADS, 2)
    fft17_fused(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    extern __shared__ double2 s[];
    const size_t vbase = (size_t)blockIdx.x * NPT;
    const int tid = threadIdx.x;

    if (STG == 1) {
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
    } else if (STG == 3) {
#pragma unroll
        for (int r = 0; r < 15; ++r)
            __pipeline_memcpy_async(&s[tid + r * NTHREADS], &in[vbase + tid + r * NTHREADS],
                                    sizeof(double2));
        {
            const int itail = tid + 15 * NTHREADS;
            if (itail < NPT)
                __pipeline_memcpy_async(&s[itail], &in[vbase + itail], sizeof(double2));
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
    } else {
        const int warp = tid >> 5, lane = tid & 31;
        if (STG == 0) {
            if (warp < 9) {
                const int cbase = warp * 544;
#pragma unroll
                for (int k = 0; k < 17; ++k)
                    __pipeline_memcpy_async(&s[cbase + 32 * k + lane],
                                            &in[vbase + cbase + 32 * k + lane],
                                            sizeof(double2));
            } else if (lane < 17) { /* tail: elements 4896..4912 = z line of tid 288 */
                __pipeline_memcpy_async(&s[4896 + lane], &in[vbase + 4896 + lane],
                                        sizeof(double2));
            }
            __pipeline_commit();
            __pipeline_wait_prior(0);
        } else { /* STG == 2 */
            if (warp < 9) {
                const int cbase = warp * 544;
                double2 t[17];
#pragma unroll
                for (int k = 0; k < 17; ++k) t[k] = in[vbase + cbase + 32 * k + lane];
#pragma unroll
                for (int k = 0; k < 17; ++k) s[cbase + 32 * k + lane] = t[k];
            } else if (lane < 17) {
                s[4896 + lane] = in[vbase + 4896 + lane];
            }
        }
        __syncwarp();
    }
    /* z axis: line (x,y), base tid*17, contiguous, in place in shared; under STG 0/2
     * every element of line tid lies inside this warp's own staged chunk. */
    if (tid < 289) line17sel(s, tid * 17, 1, s, tid * 17, 1);
    __syncthreads();

    /* y axis: line (x,z), base x*289 + z, stride 17, in place in shared */
    if (tid < 289) {
        const int b = (tid / 17) * 289 + (tid % 17);
        line17sel(s, b, 17, s, b, 17);
    }
    __syncthreads();

    /* x axis: line (y,z), base y*17 + z = tid, stride 289; results straight to global */
    if (tid < 289) {
#ifdef L17RF_STCS /* measured 1.5% WORSE at the HBM point; kept for re-testing only */
        line17sel(s, tid, 289, StreamOut{out + vbase}, tid, 289);
#else
        line17sel(s, tid, 289, out + vbase, tid, 289);
#endif
    }
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

static __device__ __forceinline__ void fft17_planes_body(const double2 *__restrict__ in,
                                                         double2 *__restrict__ out)
{
    /* two 4.6 KB buffers: z reads s, writes its transposed output to s2, so the
     * z->store->y chain needs ONE barrier instead of two (gpu_r3; the single-buffer
     * form needed a sync on each side of the in-place transposed store) */
    __shared__ double2 s[289], s2[289];
    const int tid = threadIdx.x;
    const size_t pbase = (size_t)blockIdx.x * 289; /* plane (v,x) = contiguous 289 */

    if (tid < 289) s[tid] = __ldg(&in[pbase + tid]);
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
        s2[ln * 17 + k] = r0;
    }
    __syncthreads();

    /* y pass: line z = ln, elements s2[j*17 + ln] */
    if (tid < 289) {
        double pr = s2[ln].x, pi = s2[ln].y, rr = 0.0, ri = 0.0;
#pragma unroll
        for (int j = 1; j <= 8; ++j) {
            const double2 a = s2[j * 17 + ln];
            const double2 b = s2[(17 - j) * 17 + ln];
            pr = fma(__ldg(&g_cos17[k][j - 1]), a.x + b.x, pr);
            pi = fma(__ldg(&g_cos17[k][j - 1]), a.y + b.y, pi);
            rr = fma(__ldg(&g_sin17[k][j - 1]), a.x - b.x, rr);
            ri = fma(__ldg(&g_sin17[k][j - 1]), a.y - b.y, ri);
        }
        r0 = make_double2(pr + ri, pi - rr);
    }
    /* The y output (ky=k, kz=ln) sits at plane offset k*17+ln == tid, so the direct
     * global store is already coalesced: no staging, and no barrier since shared is
     * not written again. (r1 staged this through shared -- an indexing thinko: the
     * store was never stride-17.) */
    if (tid < 289) out[pbase + tid] = r0;
}

extern "C" __global__ void __launch_bounds__(320, 4)
    fft17_planes(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    fft17_planes_body(in, out);
}

#ifndef XLINE_T /* 32 measured best at B=1 (9.1 vs 10.1 us at 128) */
#define XLINE_T 32
#endif
extern "C" __global__ void __launch_bounds__(XLINE_T, 4)
    fft17_xlines(double2 *__restrict__ out, int nlines)
{
    const int q = blockIdx.x * XLINE_T + threadIdx.x;
    if (q >= nlines) return;
    const int v = q / 289, p = q % 289;
    double2 *vol = out + (size_t)v * NPT;
    line17sel(vol, p, 289, vol, p, 289); /* in place; lines are disjoint per thread */
}

/* Both split halves in ONE PLAIN launch joined by a hand-rolled software grid barrier
 * (adopted from L17_dmma round gpu_r2, which measured the launch mechanics at B=1:
 * cooperative launch 11.9 us, two plain launches 10.5, one plain launch + soft barrier
 * 7.7-8.9 -- cudaLaunchCooperativeKernel itself costs ~1.4 us over <<<>>>, the second
 * launch ~1.6 us). Grid = B*17 blocks of 320: planes body (z+y of one x-plane, in place
 * on out), soft barrier, then the x pass thread-per-line in place on out -- block b's
 * first 17 threads take lines [17b, 17b+17), so unlike L17_dmma's version there is no
 * scratch buffer and no second read amplification. The barrier is an arrive-and-spin on
 * a plan-owned monotonic u64 counter: each call adds gridDim blocks and passes the
 * running total as `target`, so back-to-back calls on one stream cannot confuse epochs.
 * Release/acquire = __syncthreads + __threadfence around the atomics (the cg grid.sync
 * pattern, minus the cooperative-launch tax). LEGAL ONLY when the whole grid is
 * co-resident: create() verifies with an occupancy query and otherwise falls back to
 * the two-launch path. */
__global__ void __launch_bounds__(320, 4)
    fft17_split_soft(const double2 *__restrict__ in, double2 *__restrict__ tmp,
                     double2 *__restrict__ out, unsigned long long *bar,
                     unsigned long long target)
{
    fft17_planes_body(in, tmp);
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(bar, 1ULL);
#if L17RF_SPIN_NS > 0
        while (atomicAdd(bar, 0ULL) < target) __nanosleep(L17RF_SPIN_NS);
#else
        while (atomicAdd(bar, 0ULL) < target) {}
#endif
        __threadfence();
    }
    __syncthreads();
    /* x pass, thread-per-OUTPUT with the folded form: block = (volume, kx), thread
     * t = (y,z). A first cut ran x thread-per-line in place (17 active threads/block,
     * no tmp): 11.27 us at B=1 -- one seventeenth of the block computing a serial
     * line17w loses to 289 threads each doing 1/17th of the work. kx is block-uniform,
     * so the folded coefficient rows broadcast (__ldg, one address per warp); tmp
     * reads and out writes are 32-consecutive per warp. */
    if (threadIdx.x < 289) {
        const int t = threadIdx.x;
        const int kx = blockIdx.x % L17;
        const size_t vb = (size_t)(blockIdx.x / L17) * NPT;
        const double2 *g = tmp + vb + t;
        const double2 x0 = __ldg(&g[0]);
        double pr = x0.x, pi = x0.y, rr = 0.0, ri = 0.0;
#pragma unroll
        for (int j = 1; j <= 8; ++j) {
            const double2 a = __ldg(&g[289 * j]);
            const double2 b = __ldg(&g[289 * (17 - j)]);
            const double c = __ldg(&g_cos17[kx][j - 1]);
            const double sn = __ldg(&g_sin17[kx][j - 1]);
            pr = fma(c, a.x + b.x, pr);
            pi = fma(c, a.y + b.y, pi);
            rr = fma(sn, a.x - b.x, rr);
            ri = fma(sn, a.y - b.y, ri);
        }
        out[vb + (size_t)kx * 289 + t] = make_double2(pr + ri, pi - rr);
    }
}

/* Dense thread-per-output x pass (adopted from L17_dmma round gpu_r1's fft17_x_dense):
 * block = (volume, kx) -- 17*B blocks of 289 threads against fft17_xlines' 289*B/128 --
 * so at B=1 it puts 17 SMs on the axis instead of 3. Reads scratch (planes output),
 * writes out; both 32-consecutive per warp. Twiddles broadcast from constant memory. */
extern "C" __global__ void __launch_bounds__(289, 4)
    fft17_xdense(const double2 *__restrict__ tmp, double2 *__restrict__ out)
{
    const int t = threadIdx.x;
    const int kx = blockIdx.x % L17;
    const size_t vb = (size_t)(blockIdx.x / L17) * NPT;
    const double2 *g = tmp + vb + t;
    double ar = 0.0, ai = 0.0;
#pragma unroll
    for (int j = 0; j < L17; ++j) {
        const double2 v = __ldg(&g[289 * j]);
        const double2 c = c_w17[kx * L17 + j];
        ar = fma(v.x, c.x, fma(-v.y, c.y, ar));
        ai = fma(v.x, c.y, fma(v.y, c.x, ai));
    }
    out[vb + (size_t)kx * 289 + t] = make_double2(ar, ai);
}

/* Both split halves in ONE cooperative launch (adopted from L17_dmma round gpu_r1,
 * which measured the two-launch gap at ~1.7 us at B=1): grid = B*17 blocks of 320,
 * planes body, grid-wide sync, then the x lines spread over the same threads
 * (B*17*320 threads >= 289*B lines always). launch_bounds (320,2) keeps the register
 * count at <=102 so two blocks/SM co-reside, which the occupancy check in create()
 * needs to co-schedule up to B=11 (187 blocks on 108 SMs). */
extern "C" __global__ void __launch_bounds__(320, 2)
    fft17_split_coop(const double2 *__restrict__ in, double2 *__restrict__ out, int nlines)
{
    fft17_planes_body(in, out);
    cooperative_groups::this_grid().sync();
    const int q = blockIdx.x * 320 + threadIdx.x;
    if (q < nlines) {
        const int v = q / 289, p = q % 289;
        double2 *vol = out + (size_t)v * NPT;
        line17sel(vol, p, 289, vol, p, 289);
    }
}

#define L17RF_MAX_STREAMS 32

struct fft3d_gpu_plan {
    int L;
    int batch;
    int split;    /* 1: planes+x path, 0: fused single kernel */
    int soft;     /* split path: 1 if the single-launch soft-barrier kernel is usable */
    int coop;     /* split path: 1 if the single cooperative launch is usable */
    int stg;      /* fused path: staging arg (0 warp cp.async, 1 flat reg, 2 warp reg,
                   * 3 flat cp.async) */
    double2 *tmp; /* split path: scratch between the planes and dense-x kernels */
    unsigned long long *bar;  /* device counter for the soft grid barrier */
    unsigned long long ncall; /* host-side epoch: barrier target / ring index */
    /* gpu_r4 stream ring + graph replay */
    int nstream;  /* ring depth; 0 = synchronous gpu_r3 behaviour */
    int graph_ok; /* 1 if graph replay is usable (capture+instantiate succeeded) */
    cudaStream_t stream[L17RF_MAX_STREAMS];
    cudaGraphExec_t gexec[L17RF_MAX_STREAMS]; /* one exec per stream: a single
                   * cudaGraphExec_t serializes with itself, so concurrent ring slots
                   * each need their own instantiation of the (identical) graph */
    const double2 *gkey_in; /* pointers the graphs were captured against */
    double2 *gkey_out;
};

extern "C" const char *fft3d_gpu_name(void) { return "L17_raderfused"; }
extern "C" const char *fft3d_gpu_description(void)
{
    return "17^3 volume per block in shared, 3 axes fused, 1 global read + 1 write; "
           "conj-folded cyclic/negacyclic 17-pt lines; batch-picked staging; async "
           "execute over an 8-stream ring of per-stream CUDA-graph replays so "
           "back-to-back calls pipeline (driver syncs per sample)";
}

extern "C" int fft3d_gpu_supports(int L) { return L == L17; }

/* Below this batch the split (planes + xlines) path wins: the fused kernel cannot fill
 * the machine with fewer than ~216 blocks, while everything is still L2-resident so the
 * split path's doubled traffic is cheap. Set from measurement; override with -DL17RF_CUT,
 * and -DL17RF_FORCE=1 (always fused) / =2 (always split) for A/B runs. */
#ifndef L17RF_CUT /* retuned in gpu_r3: soft split 12.7 at B=12 vs fused 13.2; at B=13
                   * (221 blocks > 2/SM * 108) barrier-coupled imbalance flips it:
                   * soft 15.3 vs fused 13.1 -> fused from B=13 */
#define L17RF_CUT 13
#endif

/* Fused-path staging by batch: at in+out <= the 40 MB L2 (batch <= 266) register
 * staging wins; above it cp.async (L1-bypassing) wins. Boundary and the plan-time
 * selection idea from L17_dmma gpu_r2. L17RF_STG_L2 picks which register form the
 * L2 regime uses (1 flat, 2 warp-chunked); -DL17RF_STG_FORCE=n pins every batch. */
#ifndef L17RF_REG_MAX_B
#define L17RF_REG_MAX_B 266
#endif
#ifndef L17RF_STG_L2
#define L17RF_STG_L2 2
#endif
#ifndef L17RF_SPIN_NS /* soft-barrier poll interval; 0 (default) = hot spin. Measured
                       * at B=1: hot 7.94 us, 20 ns 8.21, 100 ns 10.4; no contention
                       * penalty seen up to the 204 spinning blocks of B=12. */
#define L17RF_SPIN_NS 0
#endif

/* gpu_r4 ring/graph tunables. L17RF_NSTREAM: ring depth (0 = synchronous gpu_r3 plan).
 * 16 by default: at B=1 the GPU drain floor is one-volume-latency/depth = 13.5/16 =
 * 0.84 us, safely under the ~1.3-1.8 us host enqueue floor that then decides the cell
 * (dmma's ring-8 drain floor of 1.7 us is exactly their scored 1.743); measured B=1
 * ring8 2.50, ring16 1.80 (quiet window), ring32 2.52 (noisy window, no drain gain
 * left). L17RF_GRAPH: 0 disables graph replay (plain <<<>>> on the ring streams;
 * graph replay measured -0.17 us/call at B=1). Staging under the ring is selected in
 * create() by batch from this round's A/B table (see the strategy record):
 * B=1 STG3, B=2-4 STG2, B=5-108 STG3, B=109-266 STG0, B>266 STG3. */
#ifndef L17RF_NSTREAM
#define L17RF_NSTREAM 16
#endif
#ifndef L17RF_GRAPH
#define L17RF_GRAPH 1
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

    /* full twiddle table for the split path's dense x pass */
    {
        double2 w[289];
        for (int k = 0; k < 17; ++k)
            for (int j = 0; j < 17; ++j) {
                long double th = -2.0L * 3.14159265358979323846264338327950288L *
                                 (long double)((k * j) % 17) / 17.0L;
                w[k * 17 + j].x = (double)cosl(th);
                w[k * 17 + j].y = (double)sinl(th);
            }
        if (cudaMemcpyToSymbol(c_w17, w, sizeof w) != cudaSuccess) return NULL;
    }

    /* all three staging instantiations: 78.6 KB dynamic shared opt-in + full carveout
     * so two blocks co-reside per SM */
    if (cudaFuncSetAttribute(fft17_fused<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)SMEM_BYTES) != cudaSuccess)
        return NULL;
    if (cudaFuncSetAttribute(fft17_fused<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)SMEM_BYTES) != cudaSuccess)
        return NULL;
    if (cudaFuncSetAttribute(fft17_fused<2>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)SMEM_BYTES) != cudaSuccess)
        return NULL;
    if (cudaFuncSetAttribute(fft17_fused<3>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)SMEM_BYTES) != cudaSuccess)
        return NULL;
    cudaFuncSetAttribute(fft17_fused<0>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(fft17_fused<1>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(fft17_fused<2>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(fft17_fused<3>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);

    fft3d_gpu_plan *p = (fft3d_gpu_plan *)malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->nstream = L17RF_NSTREAM;
    if (p->nstream > L17RF_MAX_STREAMS) p->nstream = L17RF_MAX_STREAMS;
    /* Under the ring the fused kernel wins every batch (dmma gpu_r3: fused+ring 2.69 vs
     * soft split 7.66 us at B=1), and the soft barrier is ring-UNSAFE (a later call's
     * blocks could satisfy an earlier call's arrive count), so split implies no ring. */
    p->split = (batch < L17RF_CUT) && (p->nstream == 0);
#if defined(L17RF_FORCE) && L17RF_FORCE == 1
    p->split = 0;
#elif defined(L17RF_FORCE) && L17RF_FORCE == 2
    p->split = 1;
    p->nstream = 0;
#endif
#ifdef L17RF_STG_FORCE
    p->stg = L17RF_STG_FORCE;
#else
    if (p->nstream > 0) {
        /* ring-mode staging from this round's A/B (numbers in the strategy record):
         * flat cp.async everywhere except the B=2-4 register pocket (dmma gpu_r3's
         * boundary, confirmed at B=4: 2.50 vs 2.57) and the L2-wave window 109-266
         * where the warp-chunked cp.async early-z start wins (B=213: 19.51/19.61 vs
         * 19.60/19.72 in two windows; tie at B=64, loses at B=8 and at HBM). */
        if (batch >= 2 && batch <= 4)
            p->stg = 2;
        else if (batch >= 109 && batch <= 266)
            p->stg = 0;
        else
            p->stg = 3;
    } else {
        p->stg = (batch <= L17RF_REG_MAX_B) ? L17RF_STG_L2 : 0;
    }
#endif
    p->soft = 0;
    p->coop = 0;
    p->tmp = NULL;
    p->bar = NULL;
    p->ncall = 0;
    p->graph_ok = 0;
    p->gkey_in = NULL;
    p->gkey_out = NULL;
    for (int i = 0; i < L17RF_MAX_STREAMS; ++i) {
        p->stream[i] = NULL;
        p->gexec[i] = NULL;
    }
    if (p->nstream > 0) {
        /* BLOCKING streams on purpose (see the header comment): they order against the
         * driver's NULL-stream memset in the correctness pass, but not against each
         * other, so the ring still pipelines back-to-back calls. */
        for (int i = 0; i < p->nstream; ++i)
            if (cudaStreamCreate(&p->stream[i]) != cudaSuccess) {
                for (int j = 0; j < i; ++j) cudaStreamDestroy(p->stream[j]);
                free(p);
                return NULL;
            }
#if L17RF_GRAPH
        p->graph_ok = 1; /* graphs captured lazily on first execute (keyed on in/out) */
#endif
    }
#ifndef L17RF_NOSOFT
    /* single-launch soft-barrier split: legal only if all 17*batch blocks are provably
     * co-resident (the barrier spins), checked against the kernel's real occupancy */
    if (p->split) {
        int dev = 0, nblk = 0, nsm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblk, fft17_split_soft, 320,
                                                          0) == cudaSuccess &&
            (long)nblk * nsm >= (long)L17 * batch &&
            cudaMalloc((void **)&p->bar, sizeof(unsigned long long)) == cudaSuccess) {
            if (cudaMemset(p->bar, 0, sizeof(unsigned long long)) == cudaSuccess &&
                cudaMalloc((void **)&p->tmp,
                           (size_t)batch * NPT * sizeof(double2)) == cudaSuccess) {
                p->soft = 1;
            } else {
                cudaFree(p->bar);
                p->bar = NULL;
            }
        }
    }
#endif
#if defined(L17RF_COOP)
    /* Cooperative single-launch split, measured 13.95 us vs 10.08 two-launch at B=1:
     * cudaLaunchCooperativeKernel + grid.sync cost more than the second launch. Kept
     * behind this flag for the record only. */
    if (p->split) {
        int dev = 0, coop_attr = 0, nblk = 0, nsm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&coop_attr, cudaDevAttrCooperativeLaunch, dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (coop_attr &&
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblk, fft17_split_coop, 320,
                                                          0) == cudaSuccess &&
            (long)nblk * nsm >= (long)L17 * batch)
            p->coop = 1;
    }
#endif
#if defined(L17RF_XDENSE)
    /* planes->scratch->dense-x: measured 11.96 us vs 10.08 in-place at B=1; kept
     * behind this flag for the record only. */
    if (p->split && !p->coop) {
        /* scratch for the planes -> dense-x pipeline (split batches only: <= 850 KB) */
        if (cudaMalloc((void **)&p->tmp, (size_t)batch * NPT * sizeof(double2)) !=
            cudaSuccess) {
            free(p);
            return NULL;
        }
    }
#endif
    return p;
}

static void launch_fused(const fft3d_gpu_plan *p, const double2 *in, double2 *out,
                         cudaStream_t st)
{
    switch (p->stg) {
    case 0: fft17_fused<0><<<p->batch, NTHREADS, SMEM_BYTES, st>>>(in, out); break;
    case 1: fft17_fused<1><<<p->batch, NTHREADS, SMEM_BYTES, st>>>(in, out); break;
    case 2: fft17_fused<2><<<p->batch, NTHREADS, SMEM_BYTES, st>>>(in, out); break;
    default: fft17_fused<3><<<p->batch, NTHREADS, SMEM_BYTES, st>>>(in, out); break;
    }
}

/* Capture ONE single-node graph of the fused launch against (in,out) and instantiate
 * it nstream times -- one exec per ring stream, because a single cudaGraphExec_t
 * serializes with itself and would undo the ring. Returns 1 on success. On any failure
 * the plan permanently falls back to plain stream launches (same semantics, ~launch
 * overhead slower at B=1). Called lazily from execute (first call / pointer change --
 * in this driver that is the first warmup call, which is discarded). */
static int rebuild_graphs(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    for (int i = 0; i < p->nstream; ++i)
        if (p->gexec[i]) {
            cudaGraphExecDestroy(p->gexec[i]);
            p->gexec[i] = NULL;
        }
    cudaGraph_t g = NULL;
    if (cudaStreamBeginCapture(p->stream[0], cudaStreamCaptureModeThreadLocal) !=
        cudaSuccess)
        return 0;
    launch_fused(p, in, out, p->stream[0]);
    if (cudaStreamEndCapture(p->stream[0], &g) != cudaSuccess || g == NULL) return 0;
    int ok = 1;
    for (int i = 0; i < p->nstream; ++i)
        if (cudaGraphInstantiate(&p->gexec[i], g, NULL, NULL, 0) != cudaSuccess) {
            ok = 0;
            break;
        }
    cudaGraphDestroy(g);
    if (!ok) {
        for (int i = 0; i < p->nstream; ++i)
            if (p->gexec[i]) {
                cudaGraphExecDestroy(p->gexec[i]);
                p->gexec[i] = NULL;
            }
        return 0;
    }
    p->gkey_in = in;
    p->gkey_out = out;
    return 1;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (p->nstream > 0) {
        const int slot = (int)(p->ncall % (unsigned long long)p->nstream);
        p->ncall++;
        if (p->graph_ok && (in != p->gkey_in || out != p->gkey_out) &&
            !rebuild_graphs(p, in, out))
            p->graph_ok = 0;
        if (p->graph_ok)
            cudaGraphLaunch(p->gexec[slot], p->stream[slot]);
        else
            launch_fused(p, in, out, p->stream[slot]);
        return;
    }
    if (p->split) {
        int nlines = 289 * p->batch;
        if (p->soft) {
            const int nblk = p->batch * L17;
            p->ncall += (unsigned long long)nblk;
            fft17_split_soft<<<nblk, 320>>>(in, p->tmp, out, p->bar, p->ncall);
        } else if (p->coop) {
            void *args[] = {(void *)&in, (void *)&out, (void *)&nlines};
            cudaLaunchCooperativeKernel((void *)fft17_split_coop, dim3(p->batch * L17),
                                        dim3(320), args, 0, 0);
        } else if (p->tmp) {
            fft17_planes<<<p->batch * L17, 320>>>(in, p->tmp);
            fft17_xdense<<<p->batch * L17, 289>>>(p->tmp, out);
        } else {
            fft17_planes<<<p->batch * L17, 320>>>(in, out);
            fft17_xlines<<<(nlines + XLINE_T - 1) / XLINE_T, XLINE_T>>>(out, nlines);
        }
    } else {
        launch_fused(p, in, out, NULL);
    }
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p)
{
    if (!p) return;
    for (int i = 0; i < L17RF_MAX_STREAMS; ++i) {
        if (p->gexec[i]) cudaGraphExecDestroy(p->gexec[i]);
        if (p->stream[i]) cudaStreamDestroy(p->stream[i]);
    }
    if (p->tmp) cudaFree(p->tmp);
    if (p->bar) cudaFree(p->bar);
    free(p);
}
