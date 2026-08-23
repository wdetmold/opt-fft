/* L36_sharedtiled -- two-pass plane-per-block shared-memory 3D FFT for L = 36.
 *
 * Structure (literature 09 section 9.6, structures 1 and 2):
 *   Kernel 1: one (y,z) plane per block. The plane is a contiguous 1296-element
 *             (20.25 KiB) run of the volume, so the load and store are perfectly
 *             coalesced. It is staged in shared memory with the row stride padded
 *             36 -> 37 complex doubles (gcd(36,8)=4 would otherwise be a 4-way bank
 *             conflict), then the z-lines and the y-lines are transformed in place.
 *   Kernel 2: one (x,z) slab per block, read coalesced along z, transforms the
 *             x-lines the same way, in place on `out`.
 *
 * Line engine: 36 = 6*6 Cooley-Tukey, negative exponent. 6 threads per line, 36
 * lines per plane = 216 threads per block. Stage 1: thread b does the DFT-6 over
 * a of u[6a+b]; stage 2: thread q twiddles by W36^{bq} and does the DFT-6 over b,
 * writing X[6c+q]. Stage 2's read/write slots are thread-private, so the whole
 * line pass costs two __syncthreads. All shared accesses are conflict-free per
 * quarter-warp because the padded stride 37 is odd in complex-double units.
 *
 * Cache-hint policy POL (gpu_r2, the regime split L45_pfa's record teaches):
 *   0 PLAIN   no hints. For batches whose in+out working set fits the 40 MiB L2
 *             (B <~ 26): across repeated executes both buffers stay L2-resident,
 *             and a streaming hint on them is actively harmful -- it marks the
 *             resident data evict-first and forces an HBM refetch every call.
 *   1 STREAM  __ldcs input read, __stcs final store. For unchunked big batches.
 *   2 CHUNKED STREAM plus __ldcs on kernel 2's read of the chunk intermediate.
 *
 * L2 chunking (the batched-case optimization): out is used as the intermediate,
 * and for large B the batch is processed in chunks of C volumes round-robined
 * over a few nonblocking streams, so kernel 2 reads kernel 1's freshly written
 * intermediate out of the 40 MiB L2 (7.2 TB/s) instead of HBM, and the k1 write
 * is overwritten in L2 before it ever flushes. Ideal effect: 2 HBM passes -> 1.
 * The chunk size and stream count are autotuned in create() on scratch buffers.
 *
 * Traffic: unchunked 4 x 746 KiB per volume; chunked ~2 x 746 KiB HBM;
 * L2-resident batches ~0 HBM in steady state.
 */
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "../fft3d_gpu_api.h"

#define LDIM 36
#define PSTRIDE 37                 /* padded shared row stride, odd */
#define PLANE (LDIM * LDIM)        /* 1296 */
#define VOL (LDIM * LDIM * LDIM)   /* 46656 */
#define NTHREADS 216               /* 6 threads/line * 36 lines; 1296 = 6*216 exactly */
#define MAXSTREAMS 8

/* cache-hint policies */
#define POL_PLAIN 0
#define POL_STREAM 1
#define POL_CHUNKED 2

/* W36^{b*q} for b,q in 0..5, indexed [b*6+q], negative exponent. */
__constant__ double2 c_tw36[36];

__device__ __forceinline__ double2 cadd(double2 a, double2 b)
{ return make_double2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ double2 csub(double2 a, double2 b)
{ return make_double2(a.x - b.x, a.y - b.y); }
__device__ __forceinline__ double2 cmul(double2 a, double2 b)
{ return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }

/* 6-point DFT, negative exponent, in place: DIT split into two DFT-3s. */
__device__ __forceinline__ void dft6(double2 *v)
{
    const double S = 0.86602540378443864676; /* sin(60) */
    double2 e0 = v[0], e1 = v[2], e2 = v[4];
    double2 o0 = v[1], o1 = v[3], o2 = v[5];
    double2 t, m, s, E0, E1, E2, O0, O1, O2;

    t = cadd(e1, e2);
    E0 = cadd(e0, t);
    m = make_double2(e0.x - 0.5 * t.x, e0.y - 0.5 * t.y);
    s = make_double2(S * (e1.x - e2.x), S * (e1.y - e2.y));
    E1 = make_double2(m.x + s.y, m.y - s.x);   /* m - i*s */
    E2 = make_double2(m.x - s.y, m.y + s.x);   /* m + i*s */

    t = cadd(o1, o2);
    O0 = cadd(o0, t);
    m = make_double2(o0.x - 0.5 * t.x, o0.y - 0.5 * t.y);
    s = make_double2(S * (o1.x - o2.x), S * (o1.y - o2.y));
    O1 = make_double2(m.x + s.y, m.y - s.x);
    O2 = make_double2(m.x - s.y, m.y + s.x);

    /* w6^1 = (1/2, -S), w6^2 = (-1/2, -S) */
    double2 w1o = cmul(make_double2(0.5, -S), O1);
    double2 w2o = cmul(make_double2(-0.5, -S), O2);
    v[0] = cadd(E0, O0);  v[3] = csub(E0, O0);
    v[1] = cadd(E1, w1o); v[4] = csub(E1, w1o);
    v[2] = cadd(E2, w2o); v[5] = csub(E2, w2o);
}

/* One 36-point line pass over the whole plane in shared memory.
 * Thread (l = tid%36, j = tid/36): line l, sixth j. `base` is the slot of the
 * line's element 0, `estr` the slot stride between consecutive line elements.
 * Every thread participates in every __syncthreads (no divergence). */
__device__ __forceinline__ void line36_pass(double2 *sh, int j, int base, int estr)
{
    double2 v[6];
    /* stage 1: DFT-6 over a of u[6a+j] */
#pragma unroll
    for (int a = 0; a < 6; ++a) v[a] = sh[base + (6 * a + j) * estr];
    dft6(v);
    __syncthreads();               /* all reads done before anyone writes */
#pragma unroll
    for (int q = 0; q < 6; ++q) sh[base + (6 * j + q) * estr] = v[q];
    __syncthreads();
    /* stage 2: thread j = q; slots {6b+j} are thread-private from here on */
#pragma unroll
    for (int b = 0; b < 6; ++b)
        v[b] = cmul(sh[base + (6 * b + j) * estr], c_tw36[b * 6 + j]);
    dft6(v);
#pragma unroll
    for (int c = 0; c < 6; ++c) sh[base + (6 * c + j) * estr] = v[c];
    __syncthreads();
}

/* Plane pass (z then y) for plane index `pl` = v*36+x: contiguous 1296 elements. */
template <int POL>
__device__ __forceinline__ void zy_plane(double2 *sh, const double2 *__restrict__ in,
                                         double2 *__restrict__ out, int pl)
{
    const size_t base = (size_t)pl * PLANE;
    const int tid = threadIdx.x;

#pragma unroll
    for (int i = tid; i < PLANE; i += NTHREADS) {
        const double2 *sp = &in[base + i];
        /* STREAM/CHUNKED: never reused, don't let it evict the hot lines.
           PLAIN: the batch is L2-resident across executes -- keep it cached. */
        sh[i + i / LDIM] = (POL >= POL_STREAM) ? __ldcs(sp) : *sp;
    }
    __syncthreads();

    const int l = tid % LDIM, j = tid / LDIM;
    line36_pass(sh, j, l * PSTRIDE, 1);      /* z-lines: l = y  */
    line36_pass(sh, j, l, PSTRIDE);          /* y-lines: l = z  */

#pragma unroll
    for (int i = tid; i < PLANE; i += NTHREADS) out[base + i] = sh[i + i / LDIM];
    /* note: an L2::evict_last cache_hint store on the CHUNKED intermediate was
       measured 24% SLOWER at B=1438 (2118 -> 2623 us, chunk=6/4) -- without a
       persisting-L2 carve the hint distorts replacement; do not reintroduce. */
    (void)0;
}

/* x pass for slab index `sl` = v*36+y. src == dst is safe: the slab is fully
 * staged in shared before the store. CHUNKED: the intermediate is dead after
 * this read (the store overwrites the same lines), so read it evict-first. */
template <int POL>
__device__ __forceinline__ void x_slab(double2 *sh, const double2 *__restrict__ src,
                                       double2 *__restrict__ dst, int sl)
{
    const int v = sl / LDIM, y = sl % LDIM;
    const size_t base = (size_t)v * VOL + (size_t)y * LDIM;
    const int tid = threadIdx.x;

#pragma unroll
    for (int i = tid; i < PLANE; i += NTHREADS) {
        int x = i / LDIM, z = i % LDIM;
        const double2 *sp = &src[base + (size_t)x * PLANE + z];
        sh[x * PSTRIDE + z] = (POL == POL_CHUNKED) ? __ldlu(sp) : *sp;
    }
    __syncthreads();

    const int l = tid % LDIM, j = tid / LDIM;
    line36_pass(sh, j, l, PSTRIDE);          /* x-lines: l = z */

#pragma unroll
    for (int i = tid; i < PLANE; i += NTHREADS) {
        int x = i / LDIM, z = i % LDIM;
        double2 *dp = &dst[base + (size_t)x * PLANE + z];
        /* STREAM/CHUNKED: final result, never re-read -- evict-first so it does
           not push the next chunk's intermediate out of L2. PLAIN: the buffer is
           re-dirtied every execute; keep the lines resident. */
        if (POL >= POL_STREAM) __stcs(dp, sh[x * PSTRIDE + z]);
        else *dp = sh[x * PSTRIDE + z];
    }
}

template <int POL>
__global__ void __launch_bounds__(NTHREADS)
k36_zy(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    __shared__ double2 sh[LDIM * PSTRIDE];
    zy_plane<POL>(sh, in, out, blockIdx.x);
}

template <int POL>
__global__ void __launch_bounds__(NTHREADS)
k36_x(const double2 *__restrict__ src, double2 *__restrict__ dst)
{
    __shared__ double2 sh[LDIM * PSTRIDE];
    x_slab<POL>(sh, src, dst, blockIdx.x);
}

/* Fused single-launch variant for small batches (grid must be co-resident):
 * phase 1 on the plane, grid-wide sync, phase 2 on the slab. At tiny B the
 * per-execute cost is CPU launch submission, so one launch beats two. */
__global__ void __launch_bounds__(NTHREADS)
k36_fused(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    __shared__ double2 sh[LDIM * PSTRIDE];
    zy_plane<POL_PLAIN>(sh, in, out, blockIdx.x);
    cooperative_groups::this_grid().sync();
    x_slab<POL_PLAIN>(sh, out, out, blockIdx.x);
}

struct fft3d_gpu_plan {
    int L, B;
    int chunk;      /* volumes per chunk; 0 = whole batch in one launch pair */
    int nstreams;
    int pol;        /* cache-hint policy for the UNCHUNKED path (0 or 1) */
    int fused;      /* small-B: one cooperative launch instead of two kernels */
    int use_graph;  /* unchunked path: replay a captured 2-kernel graph */
    cudaStream_t st[MAXSTREAMS];
    cudaEvent_t ev[MAXSTREAMS];   /* fork/join events for multi-stream capture */
    /* cached executable graph, keyed by the buffers it was captured with;
       recaptured if the driver ever passes new pointers */
    cudaGraphExec_t gexec;
    const double2 *g_in;
    double2 *g_out;
    /* per-stream intermediate: the same small address range is re-dirtied every
       chunk, so it stays L2-resident instead of flushing to HBM the way fresh
       out-buffer addresses would */
    double2 *scratch;   /* nstreams * chunk * VOL elements, or NULL */
};

extern "C" const char *fft3d_gpu_name(void) { return "L36_sharedtiled"; }

static char g_desc[160] =
    "two-pass plane-per-block, 6x6 CT lines in padded shared, L2-chunked batch";
extern "C" const char *fft3d_gpu_description(void) { return g_desc; }

extern "C" int fft3d_gpu_supports(int L) { return L == LDIM; }

/* The raw launches: fused / unchunked pair on st[0] / multi-stream chunk loop. */
static void enqueue_work(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    const int nv = p->B;
    if (p->fused) {
        void *args[] = { (void *)&in, (void *)&out };
        cudaLaunchCooperativeKernel((void *)k36_fused, dim3(nv * LDIM),
                                    dim3(NTHREADS), args, 0, 0);
        return;
    }
    if (p->chunk <= 0 || p->chunk >= nv) {
        if (p->pol == POL_PLAIN) {
            k36_zy<POL_PLAIN><<<nv * LDIM, NTHREADS, 0, p->st[0]>>>(in, out);
            k36_x<POL_PLAIN><<<nv * LDIM, NTHREADS, 0, p->st[0]>>>(out, out);
        } else {
            k36_zy<POL_STREAM><<<nv * LDIM, NTHREADS, 0, p->st[0]>>>(in, out);
            k36_x<POL_STREAM><<<nv * LDIM, NTHREADS, 0, p->st[0]>>>(out, out);
        }
        return;
    }
    int c = 0;
    for (int v0 = 0; v0 < nv; v0 += p->chunk, ++c) {
        int cc = nv - v0 < p->chunk ? nv - v0 : p->chunk;
        int si = c % p->nstreams;
        cudaStream_t s = p->st[si];
        const double2 *pin = in + (size_t)v0 * VOL;
        double2 *pout = out + (size_t)v0 * VOL;
        if (p->scratch) {
            double2 *mid = p->scratch + (size_t)si * p->chunk * VOL;
            k36_zy<POL_CHUNKED><<<cc * LDIM, NTHREADS, 0, s>>>(pin, mid);
            k36_x<POL_CHUNKED><<<cc * LDIM, NTHREADS, 0, s>>>(mid, pout);
        } else if (p->pol == POL_PLAIN) {
            /* small-B "split" mode: the batch is L2-resident, the chunking is
               only there to overlap pass 1 and pass 2 across streams */
            k36_zy<POL_PLAIN><<<cc * LDIM, NTHREADS, 0, s>>>(pin, pout);
            k36_x<POL_PLAIN><<<cc * LDIM, NTHREADS, 0, s>>>(pout, pout);
        } else {
            k36_zy<POL_CHUNKED><<<cc * LDIM, NTHREADS, 0, s>>>(pin, pout);
            k36_x<POL_CHUNKED><<<cc * LDIM, NTHREADS, 0, s>>>(pout, pout);
        }
    }
}

/* The one enqueue path, shared by execute() and the create()-time autotuner.
 * use_graph: capture enqueue_work once (multi-stream, fork/join through st[0])
 * and replay it -- one CPU graph launch instead of up to 2*B/chunk launches. */
static void enqueue(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (!p->use_graph || p->fused) { enqueue_work(p, in, out); return; }
    if (p->gexec == NULL || in != p->g_in || out != p->g_out) {
        if (p->gexec) { cudaGraphExecDestroy(p->gexec); p->gexec = NULL; }
        int forked = p->chunk > 0 && p->chunk < p->B && p->nstreams > 1;
        cudaGraph_t g;
        cudaStreamBeginCapture(p->st[0], cudaStreamCaptureModeThreadLocal);
        if (forked) {
            cudaEventRecord(p->ev[0], p->st[0]);
            for (int i = 1; i < p->nstreams; ++i)
                cudaStreamWaitEvent(p->st[i], p->ev[0], 0);
        }
        enqueue_work(p, in, out);
        if (forked)
            for (int i = 1; i < p->nstreams; ++i) {
                cudaEventRecord(p->ev[i], p->st[i]);
                cudaStreamWaitEvent(p->st[0], p->ev[i], 0);
            }
        if (cudaStreamEndCapture(p->st[0], &g) == cudaSuccess) {
            if (cudaGraphInstantiate(&p->gexec, g, 0) == cudaSuccess) {
                p->g_in = in; p->g_out = out;
            } else p->gexec = NULL;
            cudaGraphDestroy(g);
        }
        if (p->gexec == NULL) {           /* capture failed: run directly */
            enqueue_work(p, in, out);
            return;
        }
    }
    cudaGraphLaunch(p->gexec, p->st[0]);
}

extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)
{
    if (L != LDIM || batch < 1) return NULL;

    /* twiddles W36^{bq} */
    double2 tw[36];
    for (int b = 0; b < 6; ++b)
        for (int q = 0; q < 6; ++q) {
            double ang = -2.0 * M_PI * (double)(b * q) / 36.0;
            tw[b * 6 + q] = make_double2(cos(ang), sin(ang));
        }
    if (cudaMemcpyToSymbol(c_tw36, tw, sizeof tw) != cudaSuccess) return NULL;

    /* all data flows through shared explicitly; give the carveout to shared */
    cudaFuncSetAttribute(k36_zy<POL_PLAIN>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(k36_zy<POL_STREAM>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(k36_zy<POL_CHUNKED>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(k36_x<POL_PLAIN>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(k36_x<POL_STREAM>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(k36_x<POL_CHUNKED>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaGetLastError(); /* the carveout hint is allowed to fail */

    fft3d_gpu_plan *p = (fft3d_gpu_plan *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L; p->B = batch; p->chunk = 0; p->nstreams = 1;
    p->pol = POL_STREAM;
    for (int i = 0; i < MAXSTREAMS; ++i)
        if (cudaStreamCreateWithFlags(&p->st[i], cudaStreamNonBlocking) != cudaSuccess) {
            for (int k = 0; k < i; ++k) cudaStreamDestroy(p->st[k]);
            free(p); return NULL;
        }
    for (int i = 0; i < MAXSTREAMS; ++i)
        cudaEventCreateWithFlags(&p->ev[i], cudaEventDisableTiming);

    /* debug override: L36_CHUNK="chunk,streams[,scratch]" skips the autotune */
    const char *ov = getenv("L36_CHUNK");
    if (ov) {
        int c = 0, n = 1, s = 1;
        if (sscanf(ov, "%d,%d,%d", &c, &n, &s) >= 1) {
            p->chunk = c; p->nstreams = n < 1 ? 1 : (n > MAXSTREAMS ? MAXSTREAMS : n);
            if (c > 0 && s &&
                cudaMalloc(&p->scratch,
                           (size_t)p->nstreams * c * VOL * sizeof(double2)) != cudaSuccess)
                p->scratch = NULL;
            fprintf(stderr, "L36_sharedtiled: override chunk=%d streams=%d scratch=%d\n",
                    p->chunk, p->nstreams, p->scratch ? 1 : 0);
            return p;
        }
    }

    /* ---- autotune the chunking on scratch buffers (excluded from the score) ----
     * Chunked only pays when the batch working set overflows L2; a chunk pair is
     * ~1.5 MiB * C live in L2 per stream. Candidates keep ns*C*1.5MiB under ~40. */
    if (batch > 32) {
        size_t bytes = (size_t)batch * VOL * sizeof(double2);
        double2 *da = NULL, *db = NULL, *sc = NULL;
        /* max chunk*nstreams over the candidate list below is 48 volumes */
        size_t scbytes = (size_t)48 * VOL * sizeof(double2);
        if (cudaMalloc(&da, bytes) == cudaSuccess &&
            cudaMalloc(&db, bytes) == cudaSuccess &&
            cudaMalloc(&sc, scbytes) == cudaSuccess) {
            cudaMemset(da, 0, bytes);
            cudaMemset(db, 0, bytes);
            cudaMemset(sc, 0, scbytes);
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            static const int cand[][2] = { {0,1}, {4,4}, {5,4}, {6,3}, {6,4},
                                           {7,4}, {8,4}, {10,4},
                                           {12,2}, {12,3}, {16,2} };
            int ncand = (int)(sizeof cand / sizeof cand[0]);
            float best = 1e30f; int bc = 0, bn = 1, bs = 0;
            int reps = batch >= 512 ? 3 : 20;
            for (int i = 0; i < ncand; ++i) {
                if (cand[i][0] >= batch && cand[i][0] != 0) continue;
                for (int use_sc = 0; use_sc <= (cand[i][0] ? 1 : 0); ++use_sc) {
                    p->chunk = cand[i][0]; p->nstreams = cand[i][1];
                    p->scratch = use_sc ? sc : NULL;
                    enqueue(p, da, db);                    /* warm */
                    cudaDeviceSynchronize();
                    float t = 1e30f;
                    for (int r = 0; r < 3; ++r) {          /* best of 3 samples */
                        cudaEventRecord(e0);
                        for (int k = 0; k < reps; ++k) enqueue(p, da, db);
                        cudaDeviceSynchronize();
                        cudaEventRecord(e1);
                        cudaEventSynchronize(e1);
                        float ms; cudaEventElapsedTime(&ms, e0, e1);
                        if (ms < t) t = ms;
                    }
                    if (t < best) { best = t; bc = cand[i][0]; bn = cand[i][1]; bs = use_sc; }
                }
            }
            p->chunk = bc; p->nstreams = bn;
            p->scratch = NULL;
            if (bs && bc) {
                size_t need = (size_t)bn * bc * VOL * sizeof(double2);
                if (cudaMalloc(&p->scratch, need) != cudaSuccess) p->scratch = NULL;
            }
            /* with the shape fixed, also measure plain launches vs graph replay */
            float tg[2] = { 1e30f, 1e30f };
            for (int f = 0; f <= 1; ++f) {
                p->use_graph = f;
                enqueue(p, da, db);
                cudaDeviceSynchronize();
                if (cudaGetLastError() != cudaSuccess) continue;
                for (int r = 0; r < 3; ++r) {
                    cudaEventRecord(e0);
                    for (int k = 0; k < reps; ++k) enqueue(p, da, db);
                    cudaDeviceSynchronize();
                    cudaEventRecord(e1);
                    cudaEventSynchronize(e1);
                    float ms; cudaEventElapsedTime(&ms, e0, e1);
                    if (ms < tg[f]) tg[f] = ms;
                }
            }
            p->use_graph = tg[1] < tg[0] ? 1 : 0;
            if (p->gexec) { cudaGraphExecDestroy(p->gexec); p->gexec = NULL; }
            p->g_in = NULL; p->g_out = NULL;
            snprintf(g_desc, sizeof g_desc,
                     "two-pass plane-per-block, 6x6 CT lines, padded shared; "
                     "L2 chunk=%d streams=%d scratch=%d graph=%d",
                     bc, bn, p->scratch ? 1 : 0, p->use_graph);
            fprintf(stderr, "L36_sharedtiled: autotuned chunk=%d streams=%d scratch=%d "
                            "graph=%d (plain %.0f us, graph %.0f us)\n",
                    bc, bn, p->scratch ? 1 : 0, p->use_graph,
                    tg[0] * 1e3 / reps, tg[1] * 1e3 / reps);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
        if (da) cudaFree(da);
        if (db) cudaFree(db);
        if (sc) cudaFree(sc);
        cudaGetLastError();
    } else {
        /* ---- small batch: everything is L2-resident, so the leverage is in
         * concurrency and launch overhead, both answered by measurement.
         * (a) shape: unchunked pair, or the batch SPLIT one-chunk-per-stream so
         *     pass 1 of one slice overlaps pass 2 of another (measured -27% at
         *     B=22: 40.9 -> 29.7 us). Slices with >1 chunk per stream are WORSE
         *     than unchunked -- only the one-chunk-per-stream shapes are tried.
         *     Each shape is tried with plain and with streaming cache hints.
         * (b) launch mode on the best shape: plain launches, the fused
         *     cooperative kernel (grid co-resident only), or graph replay. */
        int dev = 0, coop = 0, sms = 0, occ = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k36_fused, NTHREADS, 0);
        int fused_ok = coop && sms > 0 && batch * LDIM <= occ * sms;
        size_t bytes = (size_t)batch * VOL * sizeof(double2);
        double2 *da = NULL, *db = NULL;
        if (cudaMalloc(&da, bytes) == cudaSuccess &&
            cudaMalloc(&db, bytes) == cudaSuccess) {
            cudaMemset(da, 0, bytes);
            cudaMemset(db, 0, bytes);
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            /* (a) shape sweep: {chunk, nstreams, pol} */
            int cfg[16][3], ncfg = 0;
            cfg[ncfg][0] = 0; cfg[ncfg][1] = 1; cfg[ncfg][2] = POL_PLAIN;  ++ncfg;
            cfg[ncfg][0] = 0; cfg[ncfg][1] = 1; cfg[ncfg][2] = POL_STREAM; ++ncfg;
            static const int nss[3] = { 2, 4, 8 };
            for (int i = 0; i < 3; ++i) {
                int ns = nss[i];
                if (batch < 2 * ns) continue;    /* split needs >=2 vols/stream */
                int c = (batch + ns - 1) / ns;   /* one chunk per stream */
                cfg[ncfg][0] = c; cfg[ncfg][1] = ns; cfg[ncfg][2] = POL_PLAIN;  ++ncfg;
                cfg[ncfg][0] = c; cfg[ncfg][1] = ns; cfg[ncfg][2] = POL_STREAM; ++ncfg;
            }
            float tbest = 1e30f; int bi = 0;
            for (int i = 0; i < ncfg; ++i) {
                p->chunk = cfg[i][0]; p->nstreams = cfg[i][1]; p->pol = cfg[i][2];
                enqueue(p, da, db);
                cudaDeviceSynchronize();
                if (cudaGetLastError() != cudaSuccess) continue;
                float t = 1e30f;
                for (int r = 0; r < 3; ++r) {
                    cudaEventRecord(e0);
                    for (int k = 0; k < 400; ++k) enqueue(p, da, db);
                    cudaDeviceSynchronize();
                    cudaEventRecord(e1);
                    cudaEventSynchronize(e1);
                    float ms; cudaEventElapsedTime(&ms, e0, e1);
                    if (ms < t) t = ms;
                }
                if (t < tbest) { tbest = t; bi = i; }
            }
            p->chunk = cfg[bi][0]; p->nstreams = cfg[bi][1]; p->pol = cfg[bi][2];
            /* (b) launch mode, at the chosen shape */
            float tv[3] = { 1e30f, 1e30f, 1e30f };
            tv[0] = tbest;
            for (int f = 1; f <= 2; ++f) {
                if (f == 1 && !fused_ok) continue;
                p->fused = (f == 1);
                p->use_graph = (f == 2);
                enqueue(p, da, db);
                cudaDeviceSynchronize();
                if (cudaGetLastError() != cudaSuccess) continue;
                for (int r = 0; r < 3; ++r) {
                    cudaEventRecord(e0);
                    for (int k = 0; k < 400; ++k) enqueue(p, da, db);
                    cudaDeviceSynchronize();
                    cudaEventRecord(e1);
                    cudaEventSynchronize(e1);
                    float ms; cudaEventElapsedTime(&ms, e0, e1);
                    if (ms < tv[f]) tv[f] = ms;
                }
            }
            int bestf = 0;
            if (tv[1] < tv[bestf]) bestf = 1;
            if (tv[2] < tv[bestf]) bestf = 2;
            p->fused = (bestf == 1);
            p->use_graph = (bestf == 2);
            /* drop the tuning-time graph: it is keyed to the scratch buffers */
            if (p->gexec) { cudaGraphExecDestroy(p->gexec); p->gexec = NULL; }
            p->g_in = NULL; p->g_out = NULL;
            fprintf(stderr, "L36_sharedtiled: smallB chunk=%d streams=%d pol=%s "
                            "pick=%s (best plain %.1f, fused %.1f, graph %.1f us)\n",
                    p->chunk, p->nstreams,
                    p->pol == POL_PLAIN ? "plain" : "stream",
                    bestf == 0 ? "plain" : bestf == 1 ? "fused" : "graph",
                    tv[0] * 1e3 / 400, tv[1] * 1e3 / 400, tv[2] * 1e3 / 400);
            snprintf(g_desc, sizeof g_desc,
                     "two-pass plane-per-block, 6x6 CT lines, padded shared; "
                     "smallB split chunk=%d ns=%d pol=%s launch=%s",
                     p->chunk, p->nstreams,
                     p->pol == POL_PLAIN ? "plain" : "stream",
                     bestf == 0 ? "plain" : bestf == 1 ? "fused" : "graph");
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
        if (da) cudaFree(da);
        if (db) cudaFree(db);
        cudaGetLastError();
    }
    return p;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    enqueue(p, in, out);
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p)
{
    if (!p) return;
    if (p->gexec) cudaGraphExecDestroy(p->gexec);
    for (int i = 0; i < MAXSTREAMS; ++i) cudaEventDestroy(p->ev[i]);
    for (int i = 0; i < MAXSTREAMS; ++i) cudaStreamDestroy(p->st[i]);
    if (p->scratch) cudaFree(p->scratch);
    free(p);
}
