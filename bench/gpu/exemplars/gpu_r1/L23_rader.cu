/* L23_rader -- L = 23 on one A100.
 *
 * ARITHMETIC (settled by the CPU phase, ../../geom/strategies/L23_rader.md r6):
 * at p = 23, Rader-proper buys nothing: p-1 = 22 = 2*11 and the conv-11 pair has no
 * realization under 121 fused FMAs, so the honest optimal "Rader" IS the conjugate-folded
 * dense form.  Per 23-point line: u_j = x_j + x_{23-j}, d_j = x_j - x_{23-j};
 * X_k = P_k - i*Q_k, X_{23-k} = P_k + i*Q_k with P_k = x_0 + sum_j cos(2pi jk/23) u_j and
 * Q_k = sum_j sin(2pi jk/23) d_j.  All coefficients real; the -i is a free component swap.
 * 594 real FP instructions per line (484 FMA + 110 add/sub), 943k per volume: at the A100's
 * 4.87 T FP64-instr/s that is 194 ns/volume -- almost exactly the 195 ns single-read+
 * single-write HBM floor, so folded-dense on CUDA cores is balanced and DMMA is not needed.
 *
 * STRUCTURE (190.1 KiB/volume does not fit one block's 163 KiB shared -- PANEL_BRIEF):
 * two-pass. Kernel A: per x-plane, z then y transforms in shared memory (a 23x23 plane is
 * 8.46 KiB; stride 23 is odd so every shared access pattern is bank-conflict-free).
 * Kernel B: x axis, one line per thread, reads/writes warp-coalesced (consecutive threads =
 * consecutive (ky,kz), so at each x the warp touches contiguous 128B segments).
 * The 2-pass traffic penalty is reduced at large batch by CHUNKING: execute() loops over
 * batch chunks re-using NSTR small rotating tmp buffers, aiming to keep the intermediate
 * resident in the 40 MiB L2.  Chunk c runs on stream (c % NSTR) with tmp (c % NSTR):
 * neighbouring chunks' A and B overlap so small chunks still fill the machine, and every
 * data hazard is ordered by stream identity alone (A(c)->B(c) same stream;
 * B(c)->A(c+NSTR) same tmp = same residue = same stream) -- no events needed.
 * 'in' is read with __ldcs and 'out' written with __stcs so L2 stays for the tmp.
 *
 * At tiny batch the machine is thread-starved, not throughput-bound, and both coarse
 * kernels expose a 594-deep FP chain on a handful of blocks; a FINE mode (dft23_pair:
 * one thread per conjugate output pair, 12 threads/line, ~30 live registers, 1.86x the
 * FP ops) wins there by pure parallelism and is auto-selected at B=1.
 *
 * create() autotunes (A shape, unrolled load, kernel-B block, chunk, fine) with CUDA
 * events; the pick is reported in fft3d_gpu_description() for the leaderboard record.
 */
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../fft3d_gpu_api.h"

#define LL   23
#define LL2  529      /* 23^2 */
#define LL3  12167    /* 23^3 */

/* cos/sin(2*pi*(k*j mod 23)/23), k,j = 1..11, row-major [k-1][j-1]. */
__constant__ double c_cos[121];
__constant__ double c_sin[121];

/* Folded dense 23-point DFT of a strided line.  All loads complete (into the fold
 * registers) before the first store, so pin == pout with equal strides is safe.
 * 484 FMA + 110 add/sub.  Live set ~ 44 fold doubles + P/Q + x0: ~110 registers. */
template <bool STREAM_OUT>
static __device__ __forceinline__ void st_out(double2 *p, double2 v)
{
    if (STREAM_OUT) __stcs(p, v); else *p = v;
}

template <bool STREAM_OUT = false>
static __device__ __forceinline__ void dft23(const double2 *pin, int sin_,
                                             double2 *pout, int sout)
{
    double ur[11], ui[11], dr[11], di[11];
    const double2 a0 = pin[0];
#pragma unroll
    for (int j = 0; j < 11; ++j) {
        const double2 a = pin[(j + 1) * sin_];
        const double2 b = pin[(22 - j) * sin_];
        ur[j] = a.x + b.x;  ui[j] = a.y + b.y;
        dr[j] = a.x - b.x;  di[j] = a.y - b.y;
    }
    double s0r = a0.x, s0i = a0.y;
#pragma unroll
    for (int j = 0; j < 11; ++j) { s0r += ur[j]; s0i += ui[j]; }
    st_out<STREAM_OUT>(pout, make_double2(s0r, s0i));
#pragma unroll
    for (int k = 0; k < 11; ++k) {
        double pr = a0.x, pi = a0.y, qr = 0.0, qi = 0.0;
#pragma unroll
        for (int j = 0; j < 11; ++j) {
            const double c = c_cos[k * 11 + j];
            const double s = c_sin[k * 11 + j];
            pr = fma(c, ur[j], pr);  pi = fma(c, ui[j], pi);
            qr = fma(s, dr[j], qr);  qi = fma(s, di[j], qi);
        }
        st_out<STREAM_OUT>(pout + (k + 1)  * sout,
                           make_double2(pr + qi, pi - qr));       /* X_k      = P - iQ */
        st_out<STREAM_OUT>(pout + (22 - k) * sout,
                           make_double2(pr - qi, pi + qr));       /* X_{23-k} = P + iQ */
    }
}

/* Kernel A: z then y over P consecutive x-planes staged through shared memory.
 * grid = (ceil(23/P), nvol); block = 32*ceil(23P/32) threads; shared = P*529*16 B. */
template <int P, int T, bool ULOAD>
__global__ void __launch_bounds__(T)
kernA(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    extern __shared__ double2 sh[];               /* P planes of 529 double2 */
    const int b  = blockIdx.y;
    const int x0 = blockIdx.x * P;
    const int np = (LL - x0 < P) ? (LL - x0) : P;
    const size_t base = (size_t)b * LL3 + (size_t)x0 * LL2;
    const int nel = np * LL2;

    /* Coalesced cooperative load; 'in' is read exactly once, so evict-first (__ldcs)
     * to keep L2 for the intermediate.  The full-group case is unrolled with the loads
     * batched ahead of the stores: the naive load->store loop is a chain of dependent
     * DRAM latencies (measured 14 us for one plane-group at B=1, ~3x the FP time). */
    if (ULOAD && np == P) {
        /* All loads issued before any store: kills the serial-latency chain of the
           naive loop (14 us for one plane-group at B=1).  Only selected at small
           batch -- at B_HBM the same unroll floods the LSU and loses 11%. */
        constexpr int NIT = (P * LL2 + T - 1) / T;
        double2 v[NIT];
#pragma unroll
        for (int it = 0; it < NIT; ++it) {
            const int i = threadIdx.x + it * T;
            if (i < P * LL2) v[it] = __ldcs(in + base + i);
        }
#pragma unroll
        for (int it = 0; it < NIT; ++it) {
            const int i = threadIdx.x + it * T;
            if (i < P * LL2) sh[i] = v[it];
        }
    } else {
        for (int i = threadIdx.x; i < nel; i += T)
            sh[i] = __ldcs(in + base + i);
    }
    __syncthreads();

    const int l = threadIdx.x;
    if (l < np * LL) {                             /* z lines: contiguous rows   */
        double2 *row = sh + (l / LL) * LL2 + (l % LL) * LL;
        dft23(row, 1, row, 1);
    }
    __syncthreads();
    if (l < np * LL) {                             /* y lines: stride-23 columns; results
        go straight to global (runs of 23 double2 per warp stay coalesced), which drops
        the separate store phase, one barrier, and a shared round-trip */
        const int pp = l / LL, z = l % LL;
        double2 *col = sh + pp * LL2 + z;
        dft23(col, LL, out + base + (size_t)pp * LL2 + z, LL);
    }
}

/* Fine-grained line transform: one thread per conjugate OUTPUT PAIR (k, 23-k),
 * 12 threads per line.  u_j/d_j are recomputed per thread and consumed immediately,
 * so only ~30 registers are live: 12x the parallelism of dft23 at 1.86x the FP ops.
 * That trade wins exactly when the machine is starved for threads (B=1), where the
 * one-line-per-thread kernels expose a 594-deep FP chain on a handful of blocks.
 * Accumulation order matches dft23 (x0 first, j ascending), so outputs are
 * bit-identical to the coarse path. */
template <bool STREAM_OUT>
static __device__ __forceinline__ void dft23_pair(const double2 *pin, int sin_,
                                                  double2 *pout, int sout, int p)
{
    const double2 a0 = pin[0];
    if (p == 0) {                                  /* k = 0: plain sum */
        double sr = a0.x, si = a0.y;
#pragma unroll
        for (int j = 0; j < 11; ++j) {
            const double2 a = pin[(j + 1) * sin_];
            const double2 b = pin[(22 - j) * sin_];
            sr += a.x + b.x;  si += a.y + b.y;
        }
        st_out<STREAM_OUT>(pout, make_double2(sr, si));
        return;
    }
    const int k = p - 1;
    double pr = a0.x, pi = a0.y, qr = 0.0, qi = 0.0;
#pragma unroll
    for (int j = 0; j < 11; ++j) {
        const double2 a = pin[(j + 1) * sin_];
        const double2 b = pin[(22 - j) * sin_];
        const double c = c_cos[k * 11 + j];
        const double s = c_sin[k * 11 + j];
        pr = fma(c, a.x + b.x, pr);  pi = fma(c, a.y + b.y, pi);
        qr = fma(s, a.x - b.x, qr);  qi = fma(s, a.y - b.y, qi);
    }
    st_out<STREAM_OUT>(pout + (k + 1)  * sout, make_double2(pr + qi, pi - qr));
    st_out<STREAM_OUT>(pout + (22 - k) * sout, make_double2(pr - qi, pi + qr));
}

/* Fine kernel A: one x-plane per block, 12 pair-threads per line (276 of 288 active
 * per phase), z from shared plane 0 into plane 1, y from plane 1 straight to global. */
__global__ void __launch_bounds__(288)
kernAf(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    __shared__ double2 s0[LL2], s1[LL2];
    const size_t base = (size_t)blockIdx.y * LL3 + (size_t)blockIdx.x * LL2;
    for (int i = threadIdx.x; i < LL2; i += 288)
        s0[i] = __ldcs(in + base + i);
    __syncthreads();
    const int p = threadIdx.x / LL, r = threadIdx.x % LL;
    if (p < 12)                                    /* z: rows of s0 -> rows of s1 */
        dft23_pair<false>(s0 + r * LL, 1, s1 + r * LL, 1, p);
    __syncthreads();
    if (p < 12)                                    /* y: columns of s1 -> global  */
        dft23_pair<false>(s1 + r, LL, out + base + r, LL, p);
}

/* Fine kernel B: 12 pair-threads per x-line, straight global-to-global. */
__global__ void __launch_bounds__(256, 4)
kernBf(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 12 * LL2) return;
    const int p = t / LL2, line = t % LL2;
    const size_t base = (size_t)blockIdx.y * LL3 + line;
    dft23_pair<true>(in + base, LL2, out + base, LL2, p);
}

/* Kernel B: x axis.  One 23-point line per thread; at each x the warp's addresses are
 * contiguous, so every load/store instruction is fully coalesced. */
__global__ void __launch_bounds__(256, 2)
kernB(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    const int line = blockIdx.x * blockDim.x + threadIdx.x;   /* ky*23 + kz */
    if (line >= LL2) return;
    const size_t base = (size_t)blockIdx.y * LL3 + line;
    /* final outputs are never re-read: stream them past L2 (__stcs) so the cache
       stays reserved for the ping-pong intermediate */
    dft23<true>(in + base, LL2, out + base, LL2);
}

/* ------------------------------------------------------------------------- */

#define NSTR 4             /* concurrent chunks in flight           */

struct fft3d_gpu_plan {
    int          batch;
    int          aidx;     /* index into acfgs[] for kernel A       */
    int          uload;    /* kernel A unrolled load (small batch)  */
    int          fine;     /* pair-per-thread kernels (tiny batch)  */
    int          TB;       /* kernel B block size                   */
    int          chunk;    /* volumes per A/B pair (L2 blocking)    */
    int          nstr;     /* streams actually cycled (<= NSTR)     */
    double2     *tmp[NSTR];/* rotating intermediates, reused        */
    cudaStream_t st[NSTR]; /* chunk c runs on st[c%nstr], tmp[c%nstr] */
};

extern "C" const char *fft3d_gpu_name(void) { return "L23_rader"; }

static char g_desc[192] =
    "L23_rader: folded-dense 23pt (594 op/line), 2-pass z+y|x, L2-chunked";
extern "C" const char *fft3d_gpu_description(void) { return g_desc; }

extern "C" int fft3d_gpu_supports(int L) { return L == 23; }

/* Pin each stream's tmp buffer in L2 (A100 residency control): kernB's reads of the
 * intermediate then hit L2 instead of DRAM, which is the whole point of chunking.
 * The persisting pool is capped (~30 MB), so scale hitRatio down for large chunks. */
static void set_l2_window(fft3d_gpu_plan *p)
{
    int devid = 0;  cudaGetDevice(&devid);
    int maxwin = 0, maxpers = 0;
    cudaDeviceGetAttribute(&maxwin, cudaDevAttrMaxAccessPolicyWindowSize, devid);
    cudaDeviceGetAttribute(&maxpers, cudaDevAttrMaxPersistingL2CacheSize, devid);
    if (maxwin <= 0 || maxpers <= 0) return;
    cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, (size_t)maxpers);
    size_t wb = (size_t)p->chunk * LL3 * sizeof(double2);
    if (wb > (size_t)maxwin) wb = (size_t)maxwin;
    float hr = (float)((double)maxpers / (double)(NSTR * wb));
    if (hr > 1.0f) hr = 1.0f;
    for (int w = 0; w < NSTR; ++w) {
        cudaStreamAttrValue v;
        memset(&v, 0, sizeof(v));
        v.accessPolicyWindow.base_ptr  = (void *)p->tmp[w];
        v.accessPolicyWindow.num_bytes = wb;
        v.accessPolicyWindow.hitRatio  = hr;
        v.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
        v.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
        cudaStreamSetAttribute(p->st[w], cudaStreamAttributeAccessPolicyWindow, &v);
    }
}

/* kernel-A shapes: P planes staged, T threads.  T > 23P buys nothing in the line
 * phases but loads the planes with more warps -- decisive at tiny batch. */
struct ACfg { int P, T; };
static const ACfg acfgs[] = { {1, 32}, {1, 64}, {2, 64}, {2, 128},
                              {4, 96}, {5, 128}, {8, 192} };
#define NACFG 7

static void launchA(int aidx, int uload, dim3 grid, cudaStream_t s,
                    const double2 *in, double2 *out)
{
    const size_t shb = (size_t)acfgs[aidx].P * LL2 * sizeof(double2);
    if (uload) switch (aidx) {
    case 0: kernA<1,  32, true><<<grid,  32, shb, s>>>(in, out); break;
    case 1: kernA<1,  64, true><<<grid,  64, shb, s>>>(in, out); break;
    case 2: kernA<2,  64, true><<<grid,  64, shb, s>>>(in, out); break;
    case 3: kernA<2, 128, true><<<grid, 128, shb, s>>>(in, out); break;
    case 4: kernA<4,  96, true><<<grid,  96, shb, s>>>(in, out); break;
    case 5: kernA<5, 128, true><<<grid, 128, shb, s>>>(in, out); break;
    case 6: kernA<8, 192, true><<<grid, 192, shb, s>>>(in, out); break;
    } else switch (aidx) {
    case 0: kernA<1,  32, false><<<grid,  32, shb, s>>>(in, out); break;
    case 1: kernA<1,  64, false><<<grid,  64, shb, s>>>(in, out); break;
    case 2: kernA<2,  64, false><<<grid,  64, shb, s>>>(in, out); break;
    case 3: kernA<2, 128, false><<<grid, 128, shb, s>>>(in, out); break;
    case 4: kernA<4,  96, false><<<grid,  96, shb, s>>>(in, out); break;
    case 5: kernA<5, 128, false><<<grid, 128, shb, s>>>(in, out); break;
    case 6: kernA<8, 192, false><<<grid, 192, shb, s>>>(in, out); break;
    }
}

static void run_case(const fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    const int PA  = acfgs[p->aidx].P;
    const int nPG = (LL + PA - 1) / PA;
    const int nBB = (LL2 + p->TB - 1) / p->TB;
    const int nBF = (12 * LL2 + 255) / 256;
    int ci = 0;
    for (int c = 0; c < p->batch; c += p->chunk, ++ci) {
        const int nb = (p->batch - c < p->chunk) ? (p->batch - c) : p->chunk;
        const int w  = ci % p->nstr;
        const double2 *cin = in + (size_t)c * LL3;
        double2 *cout = out + (size_t)c * LL3;
        if (p->fine) {
            kernAf<<<dim3(LL, nb), 288, 0, p->st[w]>>>(cin, p->tmp[w]);
            kernBf<<<dim3(nBF, nb), 256, 0, p->st[w]>>>(p->tmp[w], cout);
        } else {
            launchA(p->aidx, p->uload, dim3(nPG, nb), p->st[w], cin, p->tmp[w]);
            kernB<<<dim3(nBB, nb), p->TB, 0, p->st[w]>>>(p->tmp[w], cout);
        }
    }
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    run_case(p, in, out);
}

extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)
{
    if (L != LL || batch < 1) return NULL;

    /* coefficient tables, extended precision on the host */
    {
        double cc[121], ss[121];
        const long double twopi = 6.283185307179586476925286766559L;
        for (int k = 1; k <= 11; ++k)
            for (int j = 1; j <= 11; ++j) {
                const int m = (k * j) % 23;
                const long double th = twopi * (long double)m / 23.0L;
                cc[(k - 1) * 11 + (j - 1)] = (double)cosl(th);
                ss[(k - 1) * 11 + (j - 1)] = (double)sinl(th);
            }
        if (cudaMemcpyToSymbol(c_cos, cc, sizeof(cc)) != cudaSuccess) return NULL;
        if (cudaMemcpyToSymbol(c_sin, ss, sizeof(ss)) != cudaSuccess) return NULL;
    }

    /* P=8 needs 67.7 KB dynamic shared: opt in above the 48 KB default */
    cudaFuncSetAttribute(kernA<8, 192, false>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         8 * LL2 * (int)sizeof(double2));
    cudaFuncSetAttribute(kernA<8, 192, true>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         8 * LL2 * (int)sizeof(double2));

    fft3d_gpu_plan *p = (fft3d_gpu_plan *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->batch = batch;

    const int tmpcap = (batch < 128) ? batch : 128;
    for (int w = 0; w < NSTR; ++w) {
        if (cudaMalloc((void **)&p->tmp[w],
                       (size_t)tmpcap * LL3 * sizeof(double2)) != cudaSuccess ||
            cudaStreamCreateWithFlags(&p->st[w], cudaStreamNonBlocking) != cudaSuccess) {
            fft3d_gpu_destroy(p); return NULL;
        }
    }
    p->nstr = NSTR;

    /* -------- plan-time autotune over (P, TB, chunk) with CUDA events -------- */
    double2 *din = NULL, *dout = NULL;
    size_t vb = (size_t)batch * LL3 * sizeof(double2);
    if (cudaMalloc((void **)&din, vb) != cudaSuccess ||
        cudaMalloc((void **)&dout, vb) != cudaSuccess) {
        if (din) cudaFree(din);
        /* fall back to a safe static pick */
        p->aidx = (batch == 1) ? 1 : 4;  p->TB = (batch == 1) ? 64 : 256;
        p->chunk = (batch < tmpcap) ? batch : tmpcap;
        return p;
    }
    cudaMemset(din, 0, vb);

    /* L23R_FORCE="aidx,TB,chunk" pins the config (aidx indexes acfgs[]; for ncu
       runs and forced A/Bs); L23R_VERBOSE=1 prints the tuner table to stderr. */
    if (const char *f = getenv("L23R_FORCE")) {
        int fa, ft, fc, fu = 0, ff = 0;
        if (sscanf(f, "%d,%d,%d,%d,%d", &fa, &ft, &fc, &fu, &ff) >= 3) {
            p->uload = fu ? 1 : 0;
            p->fine  = ff ? 1 : 0;
            cudaFree(din);  cudaFree(dout);
            p->aidx = (fa < 0) ? 0 : ((fa >= NACFG) ? NACFG - 1 : fa);
            p->TB = ft;
            p->chunk = (fc < 1) ? 1 : ((fc > tmpcap) ? tmpcap : fc);
            if (!getenv("L23R_NOL2WIN")) set_l2_window(p);
            snprintf(g_desc, sizeof(g_desc),
                     "L23_rader: FORCED A(P=%d,T=%d) TB=%d chunk=%d",
                     acfgs[p->aidx].P, acfgs[p->aidx].T, p->TB, p->chunk);
            return p;
        }
    }
    const int verbose = getenv("L23R_VERBOSE") != NULL;

    const int TBs[] = { 32, 64, 256 };
    const int Cs[]  = { 0 /* = full batch (capped) */, 24, 32, 48, 64 };

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);  cudaEventCreate(&ev1);

    float bestms = 1e30f;
    int bestA = 4, bestTB = 256, bestC = (batch < tmpcap) ? batch : tmpcap;
    int bestU = 0;
    fft3d_gpu_plan trial = *p;

    const int nul = (batch <= 32) ? 2 : 1;   /* try the unrolled load only when the
                                                machine is latency- not throughput-bound */
    trial.fine = 0;
    for (int iu = 0; iu < nul; ++iu)
    for (int ia = 0; ia < NACFG; ++ia)
        for (int it = 0; it < 3; ++it)
            for (int ic = 0; ic < 5; ++ic) {
                int C = Cs[ic] ? Cs[ic] : ((batch < tmpcap) ? batch : tmpcap);
                if (C > batch) C = batch;
                if (C > tmpcap) C = tmpcap;
                if (ic > 0 && C >= batch) continue;      /* duplicate of full-batch cell */
                trial.aidx = ia;  trial.TB = TBs[it];  trial.chunk = C;
                trial.uload = iu;
                set_l2_window(&trial);
                cudaCtxResetPersistingL2Cache();

                run_case(&trial, din, dout);             /* warm-up (also JIT/carveout) */
                cudaDeviceSynchronize();
                float best = 1e30f;
                for (int r = 0; r < 3; ++r) {
                    cudaEventRecord(ev0);
                    run_case(&trial, din, dout);
                    /* work is on p->st[*], which does not sync with the default
                       stream: drain the whole device before the stop event */
                    cudaDeviceSynchronize();
                    cudaEventRecord(ev1);
                    cudaEventSynchronize(ev1);
                    float ms = 1e30f;
                    cudaEventElapsedTime(&ms, ev0, ev1);
                    if (ms < best) best = ms;
                }
                if (verbose)
                    fprintf(stderr,
                            "L23R tuner: A(P=%d,T=%-3d) TB=%-3d chunk=%-3d  %8.1f us\n",
                            acfgs[ia].P, acfgs[ia].T, TBs[it], C, 1e3 * best);
                if (best < bestms) {
                    bestms = best;  bestA = ia;  bestTB = TBs[it];  bestC = C;
                    bestU = iu;
                }
            }

    /* pair-per-thread mode: 12x the threads at 1.86x the FP -- a tiny-batch play */
    int bestF = 0;
    if (batch <= 32) {
        trial.fine = 1;  trial.aidx = 0;  trial.TB = 256;
        trial.chunk = (batch < tmpcap) ? batch : tmpcap;
        run_case(&trial, din, dout);
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int r = 0; r < 5; ++r) {
            cudaEventRecord(ev0);
            run_case(&trial, din, dout);
            cudaDeviceSynchronize();
            cudaEventRecord(ev1);
            cudaEventSynchronize(ev1);
            float ms = 1e30f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            if (ms < best) best = ms;
        }
        if (verbose)
            fprintf(stderr, "L23R tuner: FINE               chunk=%-3d  %8.1f us\n",
                    trial.chunk, 1e3 * best);
        if (best < bestms) { bestms = best; bestF = 1; bestC = trial.chunk; }
    }

    cudaEventDestroy(ev0);  cudaEventDestroy(ev1);
    cudaFree(din);  cudaFree(dout);
    if (cudaGetLastError() != cudaSuccess) { fft3d_gpu_destroy(p); return NULL; }

    p->aidx = bestA;  p->TB = bestTB;  p->chunk = bestC;  p->uload = bestU;
    p->fine = bestF;
    set_l2_window(p);
    cudaCtxResetPersistingL2Cache();
    snprintf(g_desc, sizeof(g_desc),
             "L23_rader: folded-dense 23pt, 2-pass z+y|x; tuner %s A(P=%d,T=%d) TB=%d "
             "chunk=%d ul=%d (%.2f us/xform in-plan, nv=%d)",
             bestF ? "FINE" : "coarse", acfgs[bestA].P, acfgs[bestA].T, bestTB, bestC,
             bestU, 1e3 * bestms / batch, batch);
    return p;
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p)
{
    if (!p) return;
    for (int w = 0; w < NSTR; ++w) {
        if (p->tmp[w]) cudaFree(p->tmp[w]);
        if (p->st[w])  cudaStreamDestroy(p->st[w]);
    }
    free(p);
}
