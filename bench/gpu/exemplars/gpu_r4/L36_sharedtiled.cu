/* L36_sharedtiled -- two-pass plane-per-block shared-memory 3D FFT for L = 36.
 *
 * Structure (literature 09 section 9.6, structures 1 and 2):
 *   Kernel 1: one (y,z) plane per block. The plane is a contiguous 1296-element
 *             (20.25 KiB) run of the volume, so the load is perfectly coalesced.
 *             It is staged in shared memory with the row stride padded 36 -> 37
 *             complex doubles (gcd(36,8)=4 would otherwise be a 4-way bank
 *             conflict); the z-lines transform in place, and the y-lines' second
 *             stage streams straight to global in coalesced 576 B rows
 *             (gpu_r4, from L45_pfa r3's direct-out).
 *   Kernel 2: one (x,z) slab per block, in place on `out` (gpu_r4 direct form,
 *             same source): stage 1 reads global straight into registers
 *             (coalesced along z), only the inter-stage exchange goes through
 *             shared, stage 2 stores straight back. One internal barrier.
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
 *   2 CHUNKED STREAM plus __ldlu on kernel 2's read of the chunk intermediate.
 *
 * Batched execution, picked by measurement in create():
 *   - unchunked kernel pair, or L2 chunking: chunks of C volumes round-robined
 *     over ns streams so kernel 2 reads kernel 1's intermediate out of L2.
 *   - small-B batch-split: one slice per stream, each running its own k1;k2 pair
 *     (pass 2 of one slice fills the tail waves of pass 1 of another).
 *   - NEW gpu_r3: a persistent producer/consumer TICKET kernel, ported from
 *     L36_globalpass (gpu_r2). One resident wave of blocks pulls tickets (one
 *     ticket = one (b,x) plane or one (b,y) slab) off a global atomic; K1 tickets
 *     run LEAD volumes ahead, K2 tickets spin on a per-volume done-counter. The
 *     live intermediate is ~LEAD volumes in L2, there is no chunk fill/drain, and
 *     every batch costs ONE launch. Epoch bases keep the counters valid across
 *     executes with no reset. NEVER graph-captured (tbase/dbase advance per call).
 *
 * Traffic: unchunked 4 x 746 KiB per volume; chunked/ticket ~2 x 746 KiB HBM
 * (ticket measured at the compulsory DRAM floor by L36_globalpass's ncu run);
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

#ifndef MINB                       /* min blocks/SM for the ticket kernel's reg cap */
#define MINB 4
#endif

/* cache-hint policies */
#define POL_PLAIN 0
#define POL_STREAM 1
#define POL_CHUNKED 2

/* gpu_r4: direct global-output forms (borrowed from L45_pfa r3, who measured
 * K1-direct-out -4.8% and K2-direct a further win inside their persistent
 * kernel). Compile-time A/B: -DL36_NODIR1 / -DL36_NODIR2 restore the staged
 * forms. DIR1 = K1's y-pass stage-2 streams straight to global (drops one
 * shared write+read round trip and one barrier per plane). DIR2 = K2 reads
 * global straight into registers and stage-2 stores straight to global
 * (drops the staging copy loops and two barriers per slab). */
#ifndef L36_NODIR1
#define L36_DIR1 1
#else
#define L36_DIR1 0
#endif
#ifndef L36_NODIR2
#define L36_DIR2 1
#else
#define L36_DIR2 0
#endif

/* W36^{b*q} for b,q in 0..5, indexed [b*6+q], negative exponent. */
__constant__ double2 c_tw36[36];

__device__ __forceinline__ double2 cadd(double2 a, double2 b)
{ return make_double2(a.x + b.x, a.y + b.y); }
__device__ __forceinline__ double2 csub(double2 a, double2 b)
{ return make_double2(a.x - b.x, a.y - b.y); }
/* fma form (borrowed from L36_globalpass): 4 instr instead of 4 mul + 2 add */
__device__ __forceinline__ double2 cmul(double2 a, double2 b)
{ return make_double2(fma(a.x, b.x, -(a.y * b.y)), fma(a.x, b.y, a.y * b.x)); }

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
 * line's element 0, ESTR the slot stride between consecutive line elements
 * (compile-time 1 or PSTRIDE, so the address arithmetic folds to immediates).
 * Every thread participates in every __syncthreads (no divergence). */
template <int ESTR>
__device__ __forceinline__ void line36_pass(double2 *sh, int j, int base)
{
    double2 v[6];
    /* stage 1: DFT-6 over a of u[6a+j] */
#pragma unroll
    for (int a = 0; a < 6; ++a) v[a] = sh[base + (6 * a + j) * ESTR];
    dft6(v);
    __syncthreads();               /* all reads done before anyone writes */
#pragma unroll
    for (int q = 0; q < 6; ++q) sh[base + (6 * j + q) * ESTR] = v[q];
    __syncthreads();
    /* stage 2: thread j = q; slots {6b+j} are thread-private from here on.
       b = 0 has twiddle W36^0 = 1: skip the identity cmul. */
    v[0] = sh[base + j * ESTR];
#pragma unroll
    for (int b = 1; b < 6; ++b)
        v[b] = cmul(sh[base + (6 * b + j) * ESTR], c_tw36[b * 6 + j]);
    dft6(v);
#pragma unroll
    for (int c = 0; c < 6; ++c) sh[base + (6 * c + j) * ESTR] = v[c];
    __syncthreads();
}

/* 16-byte async global->shared copy: no register round-trip, L1 bypassed (.cg),
 * and the LSU drains it asynchronously -- the A100 latency-hiding mechanism for
 * a copy loop in a 43%-occupancy kernel (used by the L17 entries' records). */
__device__ __forceinline__ void cp16(double2 *dst, const double2 *src)
{
    unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(src));
}
__device__ __forceinline__ void cp16_wait(void)
{
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
}

/* Plane pass (z then y) for plane index `pl` = v*36+x: contiguous 1296 elements. */
template <int POL, bool CPA = false>
__device__ __forceinline__ void zy_plane(double2 *sh, const double2 *__restrict__ in,
                                         double2 *__restrict__ out, int pl)
{
    const size_t base = (size_t)pl * PLANE;
    const int tid = threadIdx.x;

    /* exact 6 trips (1296 = 6*216), no guard (loop form from L36_globalpass) */
#pragma unroll
    for (int k = 0; k < 6; ++k) {
        const int i = tid + k * NTHREADS;
        const double2 *sp = &in[base + i];
        /* STREAM/CHUNKED: never reused, don't let it evict the hot lines.
           PLAIN: the batch is L2-resident across executes -- keep it cached. */
        if (CPA) cp16(&sh[i + i / LDIM], sp);
        else sh[i + i / LDIM] = (POL >= POL_STREAM) ? __ldcs(sp) : *sp;
    }
    if (CPA) cp16_wait();
    __syncthreads();

    const int l = tid % LDIM, j = tid / LDIM;
    line36_pass<1>(sh, j, l * PSTRIDE);      /* z-lines: l = y  */

#if L36_DIR1
    /* y-lines, stage-2 direct to global: thread (l=z, j) puts X[6c+j] of y-line
       l at out[(6c+j)*36 + l]; warp lanes share (c,j) with consecutive l, so
       each store is a coalesced 576 B row. Plain stores: the intermediate must
       land in L2 for pass 2. No trailing barrier needed -- the ticket loop /
       kernel end / grid.sync covers shared reuse. */
    {
        double2 v[6];
#pragma unroll
        for (int a = 0; a < 6; ++a) v[a] = sh[l + (6 * a + j) * PSTRIDE];
        dft6(v);
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 6; ++q) sh[l + (6 * j + q) * PSTRIDE] = v[q];
        __syncthreads();
        v[0] = sh[l + j * PSTRIDE];
#pragma unroll
        for (int b = 1; b < 6; ++b)
            v[b] = cmul(sh[l + (6 * b + j) * PSTRIDE], c_tw36[b * 6 + j]);
        dft6(v);
#pragma unroll
        for (int c = 0; c < 6; ++c) out[base + (6 * c + j) * LDIM + l] = v[c];
    }
#else
    line36_pass<PSTRIDE>(sh, j, l);          /* y-lines: l = z  */

#pragma unroll
    for (int k = 0; k < 6; ++k) {
        const int i = tid + k * NTHREADS;
        out[base + i] = sh[i + i / LDIM];    /* plain: must land in L2 for pass 2 */
    }
#endif
    /* note: an L2::evict_last cache_hint store on the CHUNKED intermediate was
       measured 24% SLOWER at B=1438 (2118 -> 2623 us, chunk=6/4) -- without a
       persisting-L2 carve the hint distorts replacement; do not reintroduce. */
}

/* x pass for slab index `sl` = v*36+y. src == dst is safe: the slab is fully
 * staged in shared before the store. CHUNKED: the intermediate is dead after
 * this read (the store overwrites the same lines), so read it evict-first. */
template <int POL, bool CPA = false>
__device__ __forceinline__ void x_slab(double2 *sh, const double2 *__restrict__ src,
                                       double2 *__restrict__ dst, int sl)
{
    const int v = sl / LDIM, y = sl % LDIM;
    const size_t base = (size_t)v * VOL + (size_t)y * LDIM;
    const int tid = threadIdx.x;

#if L36_DIR2
    /* Direct form: stage 1 reads global straight into registers (fixed (a,j),
       consecutive l=z -> coalesced 576 B rows), only the inter-stage exchange
       goes through shared, stage 2 stores straight to global. In-place safe:
       every global read happens before the barrier, every write after. CPA is
       unused here (nothing to stage). */
    const int l = tid % LDIM, j = tid / LDIM;    /* l = z */
    double2 v6[6];
#pragma unroll
    for (int a = 0; a < 6; ++a) {
        const double2 *sp = &src[base + (size_t)(6 * a + j) * PLANE + l];
        v6[a] = (POL == POL_CHUNKED) ? __ldlu(sp) : *sp;
    }
    dft6(v6);
#pragma unroll
    for (int q = 0; q < 6; ++q) sh[l + (6 * j + q) * PSTRIDE] = v6[q];
    __syncthreads();
    v6[0] = sh[l + j * PSTRIDE];
#pragma unroll
    for (int b = 1; b < 6; ++b)
        v6[b] = cmul(sh[l + (6 * b + j) * PSTRIDE], c_tw36[b * 6 + j]);
    dft6(v6);
#pragma unroll
    for (int c = 0; c < 6; ++c) {
        double2 *dp = &dst[base + (size_t)(6 * c + j) * PLANE + l];
        if (POL >= POL_STREAM) __stcs(dp, v6[c]);
        else *dp = v6[c];
    }
#else
#pragma unroll
    for (int k = 0; k < 6; ++k) {
        const int i = tid + k * NTHREADS;
        const int x = i / LDIM, z = i - LDIM * x;
        const double2 *sp = &src[base + (size_t)x * PLANE + z];
        if (CPA) cp16(&sh[x * PSTRIDE + z], sp);
        else sh[x * PSTRIDE + z] = (POL == POL_CHUNKED) ? __ldlu(sp) : *sp;
    }
    if (CPA) cp16_wait();
    __syncthreads();

    const int l = tid % LDIM, j = tid / LDIM;
    line36_pass<PSTRIDE>(sh, j, l);          /* x-lines: l = z */

#pragma unroll
    for (int k = 0; k < 6; ++k) {
        const int i = tid + k * NTHREADS;
        const int x = i / LDIM, z = i - LDIM * x;
        double2 *dp = &dst[base + (size_t)x * PLANE + z];
        /* STREAM/CHUNKED: final result, never re-read -- evict-first so it does
           not push the next chunk's intermediate out of L2. PLAIN: the buffer is
           re-dirtied every execute; keep the lines resident. */
        if (POL >= POL_STREAM) __stcs(dp, sh[x * PSTRIDE + z]);
        else *dp = sh[x * PSTRIDE + z];
    }
#endif
}

/* noinline wrappers for the ticket kernel's bodies (L45_pfa r3's register-frame
 * isolation trick): under a tight launch-bounds cap, inlining both bodies makes
 * ptxas allocate one union frame that spills; separate ABI frames may not.
 * Enabled with -DL36_TNI, only used by k36_ticket. */
#ifdef L36_TNI
template <int POL, bool CPA>
__device__ __noinline__ void zy_plane_ni(double2 *sh, const double2 *__restrict__ in,
                                         double2 *__restrict__ out, int pl)
{ zy_plane<POL, CPA>(sh, in, out, pl); }
template <int POL>
__device__ __noinline__ void x_slab_ni(double2 *sh, const double2 *__restrict__ src,
                                       double2 *__restrict__ dst, int sl)
{ x_slab<POL, false>(sh, src, dst, sl); }
#endif

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

/* Persistent producer/consumer ticket kernel -- ported from L36_globalpass
 * (gpu_r2), which measured it at the compulsory DRAM floor at B_HBM.
 * Ticket t = grp*36+slot; group decode with P = min(lead, B):
 *   groups [0,P)                       K1 of volume grp
 *   groups [P, P+2*(B-P)) alternating  even d: K1 of volume P+d/2, odd d: K2 of d/2
 *   groups [P+2*(B-P), 2B)             K2 of the last P volumes
 * Every K1(v) ticket precedes every K2(v) ticket and the grid is <= one resident
 * wave, so a grabbed K2 can only wait on K1s that are running or done: no
 * deadlock. Each block ends on one failed grab, so `next` advances by exactly
 * ntick+gridDim per execute and done[v] by 36 -- host epoch bases (tbase/dbase)
 * mirror that in unsigned mod-2^32 arithmetic, no counter resets ever. */
template <int POL, bool CPA>
__global__ void __launch_bounds__(NTHREADS, MINB)
k36_ticket(const double2 *__restrict__ in, double2 *__restrict__ out,
           unsigned nvol, unsigned lead, unsigned ntick,
           unsigned tbase, unsigned dbase, unsigned *__restrict__ ctr)
{
    __shared__ double2 sh[LDIM * PSTRIDE];
    __shared__ unsigned s_t;
    const int tid = threadIdx.x;
    unsigned *done = ctr + 1;
    const unsigned P = lead < nvol ? lead : nvol;
    for (;;) {
        if (tid == 0) s_t = atomicAdd(ctr, 1u) - tbase;
        __syncthreads();               /* also orders shared reuse across tickets */
        const unsigned t = s_t;
        if (t >= ntick) return;
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
#ifdef L36_TNI
            zy_plane_ni<POL, CPA>(sh, in, out, (int)(v * 36u + slot));
#else
            zy_plane<POL, CPA>(sh, in, out, (int)(v * 36u + slot));
#endif
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
#ifdef L36_TNI
            x_slab_ni<POL>(sh, out, out, (int)(v * 36u + slot));
#else
            x_slab<POL, CPA>(sh, out, out, (int)(v * 36u + slot));
#endif
        }
    }
}

struct fft3d_gpu_plan {
    int L, B;
    int chunk;      /* volumes per chunk; 0 = whole batch in one launch pair */
    int nstreams;
    int pol;        /* cache-hint policy (also selects the ticket template) */
    int fused;      /* small-B: one cooperative launch instead of two kernels */
    int use_graph;  /* pair/split path only: replay a captured graph */
    int ticket;     /* persistent producer/consumer single-launch mode */
    int cpa;        /* ticket: cp.async staging of the shared loads */
    int lead;       /* ticket: K1 runway in volumes */
    int tgrid;      /* ticket: resident-wave grid size */
    unsigned tbase, dbase;  /* ticket: epoch bases for next / done counters */
    unsigned *ctr;  /* device: [0] = next ticket, [1..B] = done[v] */
    cudaStream_t st[MAXSTREAMS];
    cudaEvent_t ev[MAXSTREAMS];   /* fork/join events for multi-stream capture */
    /* cached executable graph, keyed by the buffers it was captured with;
       recaptured if the driver ever passes new pointers */
    cudaGraphExec_t gexec;
    const double2 *g_in;
    double2 *g_out;
    /* per-stream intermediate for the L36_CHUNK override's scratch variant */
    double2 *scratch;   /* nstreams * chunk * VOL elements, or NULL */
};

extern "C" const char *fft3d_gpu_name(void) { return "L36_sharedtiled"; }

static char g_desc[160] =
    "two-pass plane-per-block, 6x6 CT lines in padded shared, ticket/chunk/split";
extern "C" const char *fft3d_gpu_description(void) { return g_desc; }

extern "C" int fft3d_gpu_supports(int L) { return L == LDIM; }

/* The raw launches: ticket / fused / unchunked pair / multi-stream chunk loop. */
static void enqueue_work(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    const int nv = p->B;
    if (p->ticket) {
        const unsigned ntick = 72u * (unsigned)nv;
        const unsigned nvu = (unsigned)nv, ld = (unsigned)p->lead;
        if (p->cpa) {
            if (p->pol == POL_PLAIN)
                k36_ticket<POL_PLAIN, true><<<p->tgrid, NTHREADS, 0, p->st[0]>>>(
                    in, out, nvu, ld, ntick, p->tbase, p->dbase, p->ctr);
            else
                k36_ticket<POL_CHUNKED, true><<<p->tgrid, NTHREADS, 0, p->st[0]>>>(
                    in, out, nvu, ld, ntick, p->tbase, p->dbase, p->ctr);
        } else {
            if (p->pol == POL_PLAIN)
                k36_ticket<POL_PLAIN, false><<<p->tgrid, NTHREADS, 0, p->st[0]>>>(
                    in, out, nvu, ld, ntick, p->tbase, p->dbase, p->ctr);
            else
                k36_ticket<POL_CHUNKED, false><<<p->tgrid, NTHREADS, 0, p->st[0]>>>(
                    in, out, nvu, ld, ntick, p->tbase, p->dbase, p->ctr);
        }
        p->tbase += ntick + (unsigned)p->tgrid;
        p->dbase += 36u;
        return;
    }
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
 * and replay it -- one CPU graph launch instead of up to 2*B/chunk launches.
 * The ticket kernel is never captured: its tbase/dbase args advance per call. */
static void enqueue(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (!p->use_graph || p->fused || p->ticket) { enqueue_work(p, in, out); return; }
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

struct Cand { int ticket, lead, chunk, ns, pol, cpa; };

static void apply_cand(fft3d_gpu_plan *p, const Cand &c)
{
    p->ticket = c.ticket; p->lead = c.lead; p->cpa = c.cpa;
    p->chunk = c.chunk; p->nstreams = c.ns; p->pol = c.pol;
    p->fused = 0; p->use_graph = 0;
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

    /* all data flows through shared explicitly; give the carveout to shared.
       L36_PAIRCO: pair-kernel carveout, A/B-able now that the direct bodies are
       register-light enough (40/60 regs) for >4 blocks/SM at higher carveouts */
#ifndef L36_PAIRCO
#define L36_PAIRCO 50
#endif
    cudaFuncSetAttribute(k36_zy<POL_PLAIN>, cudaFuncAttributePreferredSharedMemoryCarveout, L36_PAIRCO);
    cudaFuncSetAttribute(k36_zy<POL_STREAM>, cudaFuncAttributePreferredSharedMemoryCarveout, L36_PAIRCO);
    cudaFuncSetAttribute(k36_zy<POL_CHUNKED>, cudaFuncAttributePreferredSharedMemoryCarveout, L36_PAIRCO);
    cudaFuncSetAttribute(k36_x<POL_PLAIN>, cudaFuncAttributePreferredSharedMemoryCarveout, L36_PAIRCO);
    cudaFuncSetAttribute(k36_x<POL_STREAM>, cudaFuncAttributePreferredSharedMemoryCarveout, L36_PAIRCO);
    cudaFuncSetAttribute(k36_x<POL_CHUNKED>, cudaFuncAttributePreferredSharedMemoryCarveout, L36_PAIRCO);
    /* the ticket kernel's carveout must fit MINB blocks of 21.3 KB shared:
       MINB=4 -> 100 KB config (50%), MINB=5 -> 132 KB config (81%) */
    const int tco = MINB >= 5 ? 81 : 50;
    cudaFuncSetAttribute((k36_ticket<POL_PLAIN, false>), cudaFuncAttributePreferredSharedMemoryCarveout, tco);
    cudaFuncSetAttribute((k36_ticket<POL_CHUNKED, false>), cudaFuncAttributePreferredSharedMemoryCarveout, tco);
    cudaFuncSetAttribute((k36_ticket<POL_PLAIN, true>), cudaFuncAttributePreferredSharedMemoryCarveout, tco);
    cudaFuncSetAttribute((k36_ticket<POL_CHUNKED, true>), cudaFuncAttributePreferredSharedMemoryCarveout, tco);
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

    /* ticket machinery: work + done counters, one-resident-wave grid */
    int ticket_ok = 0;
    if (cudaMalloc(&p->ctr, (size_t)(1 + batch) * sizeof(unsigned)) == cudaSuccess) {
        cudaMemset(p->ctr, 0, (size_t)(1 + batch) * sizeof(unsigned));
        int occ = 0, sms = 0, dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k36_ticket<POL_CHUNKED, false>,
                                                      NTHREADS, 0);
        long wave = (long)occ * sms, need = 72L * batch;
        p->tgrid = (int)(need < wave ? need : wave);
        if (occ > 0 && sms > 0 && p->tgrid > 0) ticket_ok = 1;
        else { cudaFree(p->ctr); p->ctr = NULL; p->tgrid = 0; }
    } else p->ctr = NULL;

    /* debug overrides skip the autotune:
       L36_TICKET="lead[,pol]"          force the persistent ticket kernel
       L36_CHUNK="chunk,streams[,scratch]"  force the chunked pair path */
    const char *tov = getenv("L36_TICKET");
    if (tov && ticket_ok) {
        int l = 12, pp = POL_CHUNKED, ca = 0;
        sscanf(tov, "%d,%d,%d", &l, &pp, &ca);
        p->ticket = 1; p->lead = l < 1 ? 1 : l;
        p->pol = (pp == POL_PLAIN) ? POL_PLAIN : POL_CHUNKED;
        p->cpa = ca ? 1 : 0;
        fprintf(stderr, "L36_sharedtiled: override ticket lead=%d pol=%d cpa=%d grid=%d\n",
                p->lead, p->pol, p->cpa, p->tgrid);
        return p;
    }
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

    /* ---- autotune on scratch buffers (create() is excluded from the score) ----
     * Every timed sample must exceed the ~20 ms boost-clock cliff (PANEL_BRIEF),
     * or the ranking is decided by SM-clock noise -- lesson taken from
     * L36_globalpass's gpu_r2 record after this entry's r2 tuner sampled 6 ms. */
    size_t bytes = (size_t)batch * VOL * sizeof(double2);
    double2 *da = NULL, *db = NULL;
    if (cudaMalloc(&da, bytes) == cudaSuccess && cudaMalloc(&db, bytes) == cudaSuccess) {
        cudaMemset(da, 0, bytes);
        cudaMemset(db, 0, bytes);
        double est_us = 1.7 * batch + 9.0;              /* rough per-execute cost */
        int reps = (int)(22000.0 / est_us) + 1;
        if (reps < 3) reps = 3;
        if (reps > 2500) reps = 2500;

        Cand cand[24]; int nc = 0;
        if (batch > 32) {
            cand[nc++] = {0, 0, 0, 1, POL_STREAM, 0};    /* unchunked pair */
            cand[nc++] = {0, 0, 7, 4, POL_CHUNKED, 0};   /* r1/r2 chunk plateau */
            cand[nc++] = {0, 0, 12, 2, POL_CHUNKED, 0};
            if (ticket_ok) {
                static const int leads[3] = { 9, 12, 14 };
                for (int i = 0; i < 3; ++i) {
                    cand[nc++] = {1, leads[i], 0, 1, POL_CHUNKED, 0};
                    cand[nc++] = {1, leads[i], 0, 1, POL_CHUNKED, 1};
                }
                cand[nc++] = {1, 12, 0, 1, POL_PLAIN, 0};
            }
        } else {
            /* small batch: everything is L2-resident; the leverage is concurrency
               and launch overhead. Shapes: unchunked pair, or batch SPLIT
               one-chunk-per-stream (gpu_r2: -29% at B=22; >=2 chunks/stream is
               WORSE than unchunked, so only these shapes are tried), each with
               plain and streaming hints; plus the ticket kernel. */
            cand[nc++] = {0, 0, 0, 1, POL_PLAIN, 0};
            cand[nc++] = {0, 0, 0, 1, POL_STREAM, 0};
            static const int nss[3] = { 2, 4, 8 };
            for (int i = 0; i < 3; ++i) {
                int ns = nss[i];
                if (batch < 2 * ns) continue;    /* split needs >=2 vols/stream */
                int c = (batch + ns - 1) / ns;   /* one chunk per stream */
                cand[nc++] = {0, 0, c, ns, POL_PLAIN, 0};
                cand[nc++] = {0, 0, c, ns, POL_STREAM, 0};
            }
            if (ticket_ok) {
                int l1 = batch < 4 ? batch : 4;
                cand[nc++] = {1, l1, 0, 1, POL_CHUNKED, 0};
                cand[nc++] = {1, l1, 0, 1, POL_PLAIN, 0};
                cand[nc++] = {1, l1, 0, 1, POL_PLAIN, 1};
                if (batch > 4) {
                    int l2 = batch < 12 ? batch : 12;
                    cand[nc++] = {1, l2, 0, 1, POL_CHUNKED, 0};
                    cand[nc++] = {1, l2, 0, 1, POL_PLAIN, 0};
                }
                if (batch > 12) {
                    cand[nc++] = {1, batch, 0, 1, POL_CHUNKED, 0};
                    cand[nc++] = {1, batch, 0, 1, POL_PLAIN, 0};
                }
            }
        }

        float tbest = 1e30f, tbest_nt = 1e30f;
        Cand cbest = cand[0], cbest_nt = cand[0];
        for (int i = 0; i < nc; ++i) {
            apply_cand(p, cand[i]);
            float t = time_cfg(p, da, db, reps);
            if (t < tbest) { tbest = t; cbest = cand[i]; }
            if (!cand[i].ticket && t < tbest_nt) { tbest_nt = t; cbest_nt = cand[i]; }
        }

        /* launch-mode probe on the best NON-ticket shape: plain launches vs
           fused cooperative kernel (small co-resident grids only) vs graph
           replay. The ticket kernel is a single launch already. */
        int fused_ok = 0;
        if (batch <= 32) {
            int dev = 0, coop = 0, sms = 0, occ = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, dev);
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k36_fused, NTHREADS, 0);
            fused_ok = coop && sms > 0 && batch * LDIM <= occ * sms;
        }
        float tv[3] = { tbest_nt, 1e30f, 1e30f };
        for (int f = 1; f <= 2; ++f) {
            if (f == 1 && !fused_ok) continue;
            apply_cand(p, cbest_nt);
            p->fused = (f == 1);
            p->use_graph = (f == 2);
            tv[f] = time_cfg(p, da, db, reps);
        }
        int bestf = 0;
        if (tv[1] < tv[bestf]) bestf = 1;
        if (tv[2] < tv[bestf]) bestf = 2;

        if (cbest.ticket && tbest < tv[bestf]) {
            apply_cand(p, cbest);
        } else {
            apply_cand(p, cbest_nt);
            p->fused = (bestf == 1);
            p->use_graph = (bestf == 2);
            tbest = tv[bestf];
        }
        /* drop the tuning-time graph: it is keyed to the scratch buffers */
        if (p->gexec) { cudaGraphExecDestroy(p->gexec); p->gexec = NULL; }
        p->g_in = NULL; p->g_out = NULL;
        snprintf(g_desc, sizeof g_desc,
                 "6x6 CT lines padded shared; %s lead=%d chunk=%d ns=%d pol=%d "
                 "launch=%s",
                 p->ticket ? "persistent ticket" : (p->chunk ? "split/chunk" : "pair"),
                 p->lead, p->chunk, p->nstreams, p->pol,
                 p->fused ? "coop" : p->use_graph ? "graph" : "plain");
        fprintf(stderr, "L36_sharedtiled: autotuned ticket=%d lead=%d cpa=%d chunk=%d "
                        "ns=%d pol=%d fused=%d graph=%d (%.1f us/exec, %d cands, reps=%d)\n",
                p->ticket, p->lead, p->cpa, p->chunk, p->nstreams, p->pol, p->fused,
                p->use_graph, tbest * 1e3f / reps, nc, reps);
        cudaFree(da); cudaFree(db);
    } else {
        if (da) cudaFree(da);
        /* alloc failed: safe defaults from the r2 record */
        if (batch > 32 && ticket_ok) { p->ticket = 1; p->lead = 12; p->pol = POL_CHUNKED; }
        else if (batch > 32) { p->chunk = 7; p->nstreams = 4; p->pol = POL_CHUNKED; }
        else { p->chunk = 0; p->nstreams = 1; p->pol = POL_STREAM; }
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
    for (int i = 0; i < MAXSTREAMS; ++i) cudaEventDestroy(p->ev[i]);
    for (int i = 0; i < MAXSTREAMS; ++i) cudaStreamDestroy(p->st[i]);
    if (p->scratch) cudaFree(p->scratch);
    if (p->ctr) cudaFree(p->ctr);
    free(p);
}
