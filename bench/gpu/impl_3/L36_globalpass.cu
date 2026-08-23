/* L36_globalpass -- two bandwidth-optimal global passes for L = 36 on one A100.
 *
 * Pass structure (corpus 09-gpu section 9.6):
 *   Pass 1 (one block per (b,x) plane):  the 36x36 (y,z) plane at fixed (b,x) is a
 *     contiguous 1296-complex block (20.25 KiB).  Load it into shared memory with the row
 *     stride padded 36 -> 37 (gcd(36,8)=4: unpadded column reads are a 4-way bank
 *     conflict), transform z then y in shared, write back.  Pure streams.
 *   Pass 2 (one block per (b,y) slab):   the x axis, in place on out.  The slab {x,z} at
 *     fixed (b,y) is 36 rows of 36 contiguous complex (576 B), strided 20736 B apart.
 *
 * The 36-point line is two radix-6 stages (36 = 6*6 Cooley-Tukey, DFT-6 = 2 x DFT-3);
 * 216 threads = 36 lines x 6 six-point DFTs per stage.  Stage-2 slots are thread-private
 * (read set == write set), so a full 36-point pass costs three __syncthreads, not four --
 * that slot assignment is borrowed from L36_sharedtiled (gpu_r1).
 *
 * Batched execution: three interchangeable schedules, picked by measurement in create():
 *   mode 0  one k_zy + one k_x over the whole batch (optionally replayed as a CUDA graph
 *           -- the graph idea is borrowed from L36_sharedtiled's small-B autotune).
 *   mode 1  L2 chunking: the batch walked in chunks of C volumes round-robined over ns
 *           streams, so k_x reads k_zy's intermediate out of the 40 MiB L2 instead of
 *           HBM.  (Round-robin form borrowed from L36_sharedtiled / L64_radix8: a fixed
 *           chunk->stream map also orders back-to-back executes on the same ranges.)
 *           At SMALL batches the same machinery with chunk = ceil(B/ns) becomes the
 *           batch-split one-slice-per-stream overlap that won L36_sharedtiled their
 *           gpu_r2 B_L2 point (their -29% at B=22): pass 2 of one slice fills the tail
 *           waves of pass 1 of another instead of the whole batch draining at the
 *           k1->k2 barrier.  Borrowed from L36_sharedtiled gpu_r2 (also L45_pfa gpu_r2).
 *   mode 2  gpu_r2: a single persistent kernel.  One resident wave of blocks
 *           pulls work tickets (one ticket = one plane/slab) from a global atomic; K1
 *           tickets run LEAD volumes ahead of K2 tickets, and each K2 ticket spin-waits
 *           on a per-volume done-counter (release: __threadfence + atomicAdd; acquire:
 *           poll + __threadfence).  This is the volume-grain producer/consumer that the
 *           r1 record called for: the live intermediate is ~LEAD volumes (~6 MB) instead
 *           of a whole chunk pair, there is no per-chunk fill/drain, and every batch size
 *           costs exactly ONE kernel launch.  Epoch bases make the counters valid across
 *           executes without any reset (each execute advances `next` by exactly
 *           ntickets+grid and each done[v] by exactly 36; unsigned wrap keeps the
 *           arithmetic exact forever).
 *           NEW gpu_r3: next-ticket PREFETCH.  Thread 0 grabs ticket t+1 before the
 *           block processes ticket t, so the ~0.5 us global-atomic round trip and the
 *           broadcast barrier overlap the current ticket's work instead of stalling
 *           the whole block between tickets.  Deadlock-free: a prefetched ticket is
 *           always higher-numbered than the block's current one, and every K1 a
 *           spinning K2 waits on is lower-numbered than that K2, so the smallest
 *           unprocessed K1 is always held as a *current* ticket somewhere (or its
 *           holder's spin target is already satisfied) and progress follows by
 *           induction.  Grab count per execute is unchanged (processed+1 per block =
 *           ntickets+grid total), so the epoch bases are untouched.
 *
 * Streaming cache hints: the read-once input uses __ldcs and the write-once final output
 * __stcs so the streaming traffic does not evict the hot intermediate (worth 1.50x at the
 * HBM point, r1).  Toggleable because at L2-resident batches the input IS reused across
 * back-to-back executes.
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../fft3d_gpu_api.h"

#define LDIM   36
#define PLANE  1296              /* 36*36 */
#define VOL    46656L            /* 36^3 */
#define SSTRIDE 37               /* padded shared row stride */
#define NTHREADS 216             /* 36 lines * 6 groups; also 1296/6 for the copy loops */
#define MAXST 8

/* -sqrt(3)/2: the imaginary part of exp(-2*pi*i/3) and exp(-2*pi*i/6). */
#define S3 (-0.8660254037844386467637231707529362)

/* W36^(j1*k1), j1,k1 in [0,6), indexed [j1*6 + k1]. */
static __constant__ double2 c_tw36[36];

struct fft3d_gpu_plan {
    int L; long batch;
    int mode;            /* 0 pair, 1 chunked round-robin, 2 persistent fused */
    int hints;           /* streaming cache hints on the in-read / final store */
    int chunk, nstreams; /* mode 1 */
    int lead;            /* mode 2: K1 runway in volumes */
    int pf;              /* mode 2: next-ticket prefetch */
    int grid;            /* mode 2: resident-wave grid size */
    unsigned tbase, dbase; /* mode 2: epoch bases for next / done counters */
    unsigned *ctr;       /* device: [0] = next ticket, [1..batch] = done[v] */
    cudaStream_t st[MAXST];
    int use_graph;       /* mode 0 only */
    cudaGraphExec_t gexec;
    const double2 *g_in; double2 *g_out;
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
 * Line l occupies elements  lb + es*idx, idx = 0..35.
 * X[k1+6*k2] = sum_j1 W6^(j1*k2) * [ W36^(j1*k1) * sum_j2 x[6*j2+j1] W6^(j2*k1) ].
 * Stage 1 (j1 = g) stores twiddled A'[j1][k1] at slot 6*j1+k1; stage 2 (k1 = g) then
 * reads {6*j1+g} and writes X[g+6*k2] to slot g+6*k2 -- the SAME slot set, so no barrier
 * between stage-2 read and write.  Three syncs per pass, not four (slot-private stage-2
 * borrowed from L36_sharedtiled).  Output lands in natural order. */
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
    for (int k1 = 0; k1 < 6; ++k1) s[lb + es * (6 * g + k1)] = a[k1];
    __syncthreads();
    /* stage 2: this thread's k1 = g; slots {6*j1+g} are private to this thread */
#pragma unroll
    for (int j1 = 0; j1 < 6; ++j1) a[j1] = s[lb + es * (6 * j1 + g)];
    dft6(a);
#pragma unroll
    for (int k2 = 0; k2 < 6; ++k2) s[lb + es * (g + 6 * k2)] = a[k2];
    __syncthreads();
}

/* z+y passes of plane `pl` (global plane index b*36+x: contiguous 1296 elements). */
template <bool HINTS>
static __device__ __forceinline__ void dev_zy(double2 *s, const double2 *__restrict__ in,
                                              double2 *__restrict__ out, size_t pl)
{
    const size_t base = pl * PLANE;
    const int tid = threadIdx.x;
#pragma unroll
    for (int k = 0; k < 6; ++k) {
        int i = tid + k * NTHREADS;
        int y = i / LDIM, z = i - LDIM * y;
        s[y * SSTRIDE + z] = HINTS ? __ldcs(in + base + i) : in[base + i];
    }
    __syncthreads();
    const int g = tid / LDIM, l = tid - LDIM * g;
    fft36_lines(s, l * SSTRIDE, 1, g);   /* z: contiguous lines */
    fft36_lines(s, l, SSTRIDE, g);       /* y: stride-37 lines */
#pragma unroll
    for (int k = 0; k < 6; ++k) {
        int i = tid + k * NTHREADS;
        int y = i / LDIM, z = i - LDIM * y;
        out[base + i] = s[y * SSTRIDE + z];  /* default policy: stays L2-hot for pass 2 */
    }
}

/* x pass of slab at byte base `base` = b*VOL + y*36, in place. */
template <bool HINTS>
static __device__ __forceinline__ void dev_x(double2 *s, double2 *__restrict__ io,
                                             size_t base)
{
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
        double2 v = s[x * SSTRIDE + z];
        if (HINTS) __stcs(io + base + (size_t)x * PLANE + z, v); /* write-once */
        else io[base + (size_t)x * PLANE + z] = v;
    }
}

template <bool HINTS>
__global__ void __launch_bounds__(NTHREADS, 4)
k_zy(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    __shared__ double2 s[LDIM * SSTRIDE];
    dev_zy<HINTS>(s, in, out, blockIdx.x);
}

template <bool HINTS>
__global__ void __launch_bounds__(NTHREADS, 4)
k_x(double2 *__restrict__ io)
{
    __shared__ double2 s[LDIM * SSTRIDE];
    const int b = blockIdx.x / LDIM, y = blockIdx.x - LDIM * b;
    dev_x<HINTS>(s, io, (size_t)b * VOL + (size_t)y * LDIM);
}

/* mode 2: persistent producer/consumer.  Ticket t = group*36+slot; group decode:
 *   groups [0,P)                         K1 of volume g            (P = min(lead,B))
 *   groups [P, P+2*(B-P)) alternating    even d: K1 of volume P+d/2, odd d: K2 of d/2
 *   groups [P+2*(B-P), 2B)               K2 of the last P volumes
 * so K1 stays `lead` volumes ahead of K2.  Every block does exactly one failed grab, so
 * `next` advances by ntick+gridDim per execute -- the host mirrors that in tbase. */
template <bool HINTS, bool PF>
__global__ void __launch_bounds__(NTHREADS, 4)
k_fused(const double2 *__restrict__ in, double2 *__restrict__ out,
        unsigned nvol, unsigned lead, unsigned ntick, unsigned tbase, unsigned dbase,
        unsigned *__restrict__ ctr)
{
    __shared__ double2 s[LDIM * SSTRIDE];
    __shared__ unsigned s_t[2];
    const int tid = threadIdx.x;
    unsigned *done = ctr + 1;
    const unsigned P = lead < nvol ? lead : nvol;
    if (tid == 0) s_t[0] = atomicAdd(ctr, 1u) - tbase;
    __syncthreads();
    unsigned t = s_t[0];
    int par = 1;
    while (t < ntick) {
        /* PF: grab the NEXT ticket now, so the atomic's latency overlaps this
         * ticket's work instead of stalling all 216 threads between tickets. */
        if (PF && tid == 0) s_t[par] = atomicAdd(ctr, 1u) - tbase;
        const unsigned grp = t / 36u, slot = t - 36u * grp;
        unsigned v; int isK1;
        if (grp < P) { isK1 = 1; v = grp; }
        else {
            const unsigned d = grp - P, npair = nvol - P;
            if (d < 2u * npair) {
                isK1 = !(d & 1u);
                v = isK1 ? P + (d >> 1) : (d >> 1);
            } else { isK1 = 0; v = npair + (d - 2u * npair); }
        }
        if (isK1) {
            dev_zy<HINTS>(s, in, out, (size_t)v * LDIM + slot);
            __threadfence();           /* release this thread's plane stores */
            __syncthreads();
            if (tid == 0) atomicAdd(done + v, 1u);
        } else {
            if (tid == 0) {
                unsigned spins = 0;
                while (*(volatile unsigned *)(done + v) - dbase < 36u)
                    if (++spins > 512u) __nanosleep(64);
                __threadfence();       /* acquire the producers' stores */
            }
            __syncthreads();
            dev_x<HINTS>(s, out, (size_t)v * VOL + (size_t)slot * LDIM);
        }
        if (PF) {
            __syncthreads();           /* s_t[par] visible; shared s reusable */
            t = s_t[par]; par ^= 1;
        } else {
            if (tid == 0) s_t[0] = atomicAdd(ctr, 1u) - tbase;
            __syncthreads();           /* also orders shared reuse across tickets */
            t = s_t[0];
        }
    }
}

extern "C" const char *fft3d_gpu_name(void) { return "L36_globalpass"; }

static char g_desc[160] =
    "two-pass plane-per-block radix-6^2, padded shared stride 37";
extern "C" const char *fft3d_gpu_description(void) { return g_desc; }

extern "C" int fft3d_gpu_supports(int L) { return L == LDIM; }

static void launch_zy(fft3d_gpu_plan *p, unsigned g, cudaStream_t s,
                      const double2 *in, double2 *out)
{
    if (p->hints) k_zy<true><<<g, NTHREADS, 0, s>>>(in, out);
    else          k_zy<false><<<g, NTHREADS, 0, s>>>(in, out);
}

static void launch_x(fft3d_gpu_plan *p, unsigned g, cudaStream_t s, double2 *io)
{
    if (p->hints) k_x<true><<<g, NTHREADS, 0, s>>>(io);
    else          k_x<false><<<g, NTHREADS, 0, s>>>(io);
}

static void launch_fused(fft3d_gpu_plan *p, const double2 *in, double2 *out,
                         unsigned nvol, unsigned ntick)
{
    const unsigned ld = (unsigned)p->lead;
    cudaStream_t s = p->st[0];
    if (p->hints) {
        if (p->pf) k_fused<true, true><<<p->grid, NTHREADS, 0, s>>>(
            in, out, nvol, ld, ntick, p->tbase, p->dbase, p->ctr);
        else       k_fused<true, false><<<p->grid, NTHREADS, 0, s>>>(
            in, out, nvol, ld, ntick, p->tbase, p->dbase, p->ctr);
    } else {
        if (p->pf) k_fused<false, true><<<p->grid, NTHREADS, 0, s>>>(
            in, out, nvol, ld, ntick, p->tbase, p->dbase, p->ctr);
        else       k_fused<false, false><<<p->grid, NTHREADS, 0, s>>>(
            in, out, nvol, ld, ntick, p->tbase, p->dbase, p->ctr);
    }
}

/* mode 2 grid = one resident wave of the fused kernel (deadlock-free by construction). */
static void set_grid(fft3d_gpu_plan *p)
{
    int occ = 0, sms = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    const void *f;
    if (p->hints) f = p->pf ? (const void *)k_fused<true, true>
                            : (const void *)k_fused<true, false>;
    else          f = p->pf ? (const void *)k_fused<false, true>
                            : (const void *)k_fused<false, false>;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, f, NTHREADS, 0);
    long wave = (long)occ * sms, need = 72L * p->batch;
    p->grid = (int)(need < wave ? need : wave);
    if (occ <= 0 || sms <= 0) p->grid = 0;
}

/* The raw launches for the current plan config (no graph). */
static void enqueue_plain(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    const long B = p->batch;
    if (p->mode == 2) {
        const unsigned ntick = 72u * (unsigned)B;
        launch_fused(p, in, out, (unsigned)B, ntick);
        p->tbase += ntick + (unsigned)p->grid;
        p->dbase += 36u;
        return;
    }
    if (p->mode == 1 && p->chunk > 0 && p->chunk < B) {
        int c = 0;
        for (long v0 = 0; v0 < B; v0 += p->chunk, ++c) {
            long cc = (B - v0 < p->chunk) ? B - v0 : p->chunk;
            cudaStream_t s = p->st[c % p->nstreams];
            launch_zy(p, (unsigned)(cc * LDIM), s, in + v0 * VOL, out + v0 * VOL);
            launch_x(p, (unsigned)(cc * LDIM), s, out + v0 * VOL);
        }
        return;
    }
    launch_zy(p, (unsigned)(B * LDIM), p->st[0], in, out);
    launch_x(p, (unsigned)(B * LDIM), p->st[0], out);
}

/* Graph replay wrapper (mode 0 only): capture once per (in,out) pair, replay after. */
static void enqueue(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (!p->use_graph || p->mode != 0) { enqueue_plain(p, in, out); return; }
    if (p->gexec == NULL || in != p->g_in || out != p->g_out) {
        if (p->gexec) { cudaGraphExecDestroy(p->gexec); p->gexec = NULL; }
        cudaGraph_t g;
        cudaStreamBeginCapture(p->st[0], cudaStreamCaptureModeThreadLocal);
        enqueue_plain(p, in, out);
        if (cudaStreamEndCapture(p->st[0], &g) == cudaSuccess) {
            if (cudaGraphInstantiate(&p->gexec, g, 0) == cudaSuccess) {
                p->g_in = in; p->g_out = out;
            } else p->gexec = NULL;
            cudaGraphDestroy(g);
        }
        if (p->gexec == NULL) { enqueue_plain(p, in, out); return; }
    }
    cudaGraphLaunch(p->gexec, p->st[0]);
}

/* best-of-3 samples of `reps` executes, in ms; 1e30 if the config faulted. */
static float time_cfg(fft3d_gpu_plan *p, const double2 *da, double2 *db, int reps)
{
    enqueue(p, da, db);                    /* warm (and graph-capture, if enabled) */
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) return 1e30f;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int r = 0; r < 3; ++r) {
        cudaEventRecord(e0);
        for (int k = 0; k < reps; ++k) enqueue(p, da, db);
        cudaDeviceSynchronize();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return best;
}

struct Cand { int mode, hints, chunk, ns, lead, graph, pf; };

extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)
{
    if (L != LDIM || batch < 1) return NULL;

    double2 h[36];
    for (int j = 0; j < 6; ++j)
        for (int k = 0; k < 6; ++k) {
            double phase = -2.0 * M_PI * (double)(j * k) / 36.0;
            h[j * 6 + k].x = cos(phase);
            h[j * 6 + k].y = sin(phase);
        }
    if (cudaMemcpyToSymbol(c_tw36, h, sizeof h) != cudaSuccess) return NULL;

    /* 21312 B of static shared per block; ask for the 100 KB carveout so 4 blocks/SM fit. */
    cudaFuncSetAttribute(k_zy<true>,  cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    cudaFuncSetAttribute(k_zy<false>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    cudaFuncSetAttribute(k_x<true>,   cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    cudaFuncSetAttribute(k_x<false>,  cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    cudaFuncSetAttribute(k_fused<true, true>,   cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    cudaFuncSetAttribute(k_fused<true, false>,  cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    cudaFuncSetAttribute(k_fused<false, true>,  cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    cudaFuncSetAttribute(k_fused<false, false>, cudaFuncAttributePreferredSharedMemoryCarveout, 50);
    cudaGetLastError();  /* the carveout hint may fail harmlessly */

    fft3d_gpu_plan *p = (fft3d_gpu_plan *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L; p->batch = batch;
    p->mode = 0; p->hints = 1; p->nstreams = 1; p->pf = 1;
    for (int i = 0; i < MAXST; ++i)
        if (cudaStreamCreateWithFlags(&p->st[i], cudaStreamNonBlocking) != cudaSuccess) {
            for (int k = 0; k < i; ++k) cudaStreamDestroy(p->st[k]);
            free(p); return NULL;
        }

    /* fused-mode machinery: work + done counters */
    int fused_ok = 0;
    if (cudaMalloc(&p->ctr, (size_t)(1 + batch) * sizeof(unsigned)) == cudaSuccess) {
        cudaMemset(p->ctr, 0, (size_t)(1 + batch) * sizeof(unsigned));
        set_grid(p);
        if (p->grid > 0) fused_ok = 1;
        else { cudaFree(p->ctr); p->ctr = NULL; }
    } else p->ctr = NULL;

    /* debug override: L36GP_CFG="mode,hints,chunk,ns,lead,graph,pf" skips the autotune */
    const char *ov = getenv("L36GP_CFG");
    if (ov) {
        int m = 0, hh = 1, c = 0, n = 1, l = 8, g = 0, f = 1;
        if (sscanf(ov, "%d,%d,%d,%d,%d,%d,%d", &m, &hh, &c, &n, &l, &g, &f) >= 1) {
            p->mode = (m == 2 && !fused_ok) ? 0 : m;
            p->hints = hh; p->chunk = c;
            p->nstreams = n < 1 ? 1 : (n > MAXST ? MAXST : n);
            p->lead = l < 1 ? 1 : l; p->use_graph = g; p->pf = f;
            if (p->mode == 2) set_grid(p);
            fprintf(stderr, "L36_globalpass: override mode=%d hints=%d chunk=%d ns=%d "
                            "lead=%d graph=%d pf=%d\n",
                    p->mode, p->hints, p->chunk, p->nstreams, p->lead, p->use_graph, p->pf);
            return p;
        }
    }

    /* ---- autotune on scratch buffers (create() time is not scored) ---- */
    size_t bytes = (size_t)batch * VOL * sizeof(double2);
    double2 *da = NULL, *db = NULL;
    if (cudaMalloc(&da, bytes) == cudaSuccess && cudaMalloc(&db, bytes) == cudaSuccess) {
        cudaMemset(da, 0, bytes);
        cudaMemset(db, 0, bytes);
        Cand cand[24]; int nc = 0;
        if (batch > 32) {
            cand[nc++] = {0, 1, 0, 1, 0, 0, 1};           /* unchunked pair */
            cand[nc++] = {1, 1, 8, 4, 0, 0, 1};           /* L2-chunked round-robin */
            cand[nc++] = {1, 1, 12, 2, 0, 0, 1};
            if (fused_ok) {                               /* persistent; 12 vol = 1 wave */
                int leads[] = {9, 12, 16};
                for (int i = 0; i < 3; ++i) cand[nc++] = {2, 1, 0, 1, leads[i], 0, 1};
                cand[nc++] = {2, 1, 0, 1, 12, 0, 0};      /* prefetch-off guard */
            }
        } else {
            cand[nc++] = {0, 1, 0, 1, 0, 0, 1};
            cand[nc++] = {0, 0, 0, 1, 0, 0, 1};
            cand[nc++] = {0, 1, 0, 1, 0, 1, 1};           /* graph replay of the pair */
            cand[nc++] = {0, 0, 0, 1, 0, 1, 1};
            /* batch-split, one slice per stream (L36_sharedtiled gpu_r2's B_L2 win;
             * slice >= 2 volumes -- 1-volume slices lose, L45_pfa gpu_r2 ns=11) */
            for (int ns = 2; ns <= MAXST; ns *= 2)
                if (batch >= 2 * ns) {
                    int c = (batch + ns - 1) / ns;
                    cand[nc++] = {1, 1, c, ns, 0, 0, 1};
                    cand[nc++] = {1, 0, c, ns, 0, 0, 1};
                }
            if (fused_ok) {
                int l1 = batch < 4 ? batch : 4;
                cand[nc++] = {2, 1, 0, 1, l1, 0, 1};
                cand[nc++] = {2, 0, 0, 1, l1, 0, 1};
                if (batch > 4)  cand[nc++] = {2, 1, 0, 1, 12 < batch ? 12 : batch, 0, 1};
                if (batch > 12) cand[nc++] = {2, 1, 0, 1, batch, 0, 1};
            }
        }
        /* each timed sample must exceed the ~20 ms boost-clock cliff (PANEL_BRIEF), or
         * the ranking is decided by SM-clock noise: reps ~ 20 ms / (1.6 us * batch) */
        int reps = (int)(20000.0 / (1.6 * batch)) + 1;
        if (reps < 3) reps = 3;
        if (reps > 2000) reps = 2000;
        float best = 1e30f; Cand bc = cand[0];
        for (int i = 0; i < nc; ++i) {
            p->mode = cand[i].mode; p->hints = cand[i].hints;
            p->chunk = cand[i].chunk; p->nstreams = cand[i].ns;
            p->lead = cand[i].lead; p->use_graph = cand[i].graph;
            p->pf = cand[i].pf;
            if (p->mode == 2) { set_grid(p); if (p->grid <= 0) continue; }
            float t = time_cfg(p, da, db, reps);
            if (t < best) { best = t; bc = cand[i]; }
        }
        p->mode = bc.mode; p->hints = bc.hints; p->chunk = bc.chunk;
        p->nstreams = bc.ns; p->lead = bc.lead; p->use_graph = bc.graph;
        p->pf = bc.pf;
        if (p->mode == 2) set_grid(p);
        fprintf(stderr, "L36_globalpass: autotuned mode=%d hints=%d chunk=%d ns=%d "
                        "lead=%d graph=%d pf=%d (%.1f us/exec, %d cands)\n",
                p->mode, p->hints, p->chunk, p->nstreams, p->lead, p->use_graph, p->pf,
                best * 1e3f / reps, nc);
        snprintf(g_desc, sizeof g_desc,
                 "two-pass radix-6^2 shared s37; %s hints=%d chunk=%d ns=%d lead=%d graph=%d pf=%d",
                 p->mode == 2 ? "persistent ticket fused" :
                 p->mode == 1 ? "split/chunked rr streams" : "plain pair",
                 p->hints, p->chunk, p->nstreams, p->lead, p->use_graph, p->pf);
        /* the tuning-time graph is keyed to the scratch buffers: drop it */
        if (p->gexec) { cudaGraphExecDestroy(p->gexec); p->gexec = NULL; }
        p->g_in = NULL; p->g_out = NULL;
        cudaFree(da); cudaFree(db);
    } else {
        if (da) cudaFree(da);
        /* alloc failed: safe defaults from the r1 record */
        if (batch > 32) {
            if (fused_ok) { p->mode = 2; p->lead = 8; }
            else { p->mode = 1; p->chunk = 12; p->nstreams = 2; }
        } else p->mode = 0;
        p->hints = 1;
    }
    cudaGetLastError();
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
    for (int i = 0; i < MAXST; ++i) cudaStreamDestroy(p->st[i]);
    if (p->ctr) cudaFree(p->ctr);
    free(p);
}
