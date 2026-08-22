/* L17_dmma -- L = 17 on one A100: single-kernel fused 3D DFT, one volume per block,
 * whole volume resident in shared memory, conjugate-folded dense 17-point DFT per line.
 *
 * Structure (corpus 09 regime C): a 17^3 volume is 78,608 B = 47% of the 163 KB a block
 * may address in shared memory, so all three axes are done with ONE global read and ONE
 * global write.  Grid = batch, one block per volume, 320 threads (289 active lines/pass).
 *
 * Per-line arithmetic is the CPU panel's L=17 winner (L17_matrixsimd round 1), folded on
 * both sides of the matrix:
 *     u_j = x_j + x_{17-j},  v_j = x_j - x_{17-j}          (j = 1..8)
 *     P_k = x_0 + sum_j cos(2 pi kj/17) u_j                 (k = 0..8)
 *     Q_k =       sum_j sin(2 pi kj/17) (-i v_j)            (k = 1..8)
 *     X_k = P_k + Q_k,  X_{17-k} = P_k - Q_k,  X_0 = P_0
 * 272 real FMA + 64 add/sub per line = 608 flop; every coefficient is REAL, so complex
 * data never needs a split/interleave and coefficients broadcast from __constant__.
 * At 867 lines/volume this is 107 flop/point: 0.54x of the 32 B/point HBM floor on the
 * VANILLA FP64 pipe, i.e. the arithmetic hides under bandwidth without tensor cores.
 * (The DMMA question this entry is named for is addressed in the strategy record with
 * this kernel's measured bandwidth as the evidence.)
 *
 * Shared memory is AoS double2: a 16-byte shared access is serviced in phases of 8
 * consecutive lanes, and within a phase the 8 accesses hit distinct bank quads iff the
 * lane-to-lane stride in complex elements is ODD (corpus 09 6.2).  All three passes'
 * line-start strides here are 17, 289+z-wrap, and 1 -- all odd steps mod 8 -- so AoS is
 * conflict-free on every load and store, at HALF the LSU instruction count of SoA.
 */
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../fft3d_gpu_api.h"

namespace cg = cooperative_groups;

#define NPT 17
#define NSQ 289          /* 17*17 */
#define NVOL 4913        /* 17^3 */
#define NTHREADS 320     /* 10 warps; threads 289..319 idle per pass */
#ifndef L17_MINB
#define L17_MINB 2       /* blocks per SM the compiler must allow (register budget) */
#endif
#ifndef L17_STAGED
#define L17_STAGED 1     /* 1: stage the input via a flat coalesced copy into shared */
#endif
#ifndef L17_SPLIT_MAX_BATCH
#define L17_SPLIT_MAX_BATCH 4   /* batch <= this: two-kernel plane-split path (see below) */
#endif
#ifndef L17_CPASYNC
#define L17_CPASYNC 1    /* 1: stage the input with cp.async instead of ld/st */
#endif
#ifndef L17_NESTED
#define L17_NESTED 0     /* 1: cyclic-4 + negacyclic-4 split of the cosine half */
#endif
#if L17_CPASYNC
#include <cuda_pipeline.h>
#endif

/* cos/sin(2 pi k j / 17), k = 0..8, j = 0..8 (j=0 column unused). Filled at plan time
 * from long-double host trig; all 320 lanes of a j-iteration read the same entry, so
 * these broadcast from the constant cache. */
static __constant__ double d_cos17[9][9];
static __constant__ double d_sin17[9][9];

/* One folded 17-point DFT.  Load(j, re, im) supplies element j of the line;
 * Store(k, re, im) receives output k.  Fully unrolled: u/v live in registers. */
template <typename Load, typename Store>
__device__ __forceinline__ void dft17_line(Load load, Store store)
{
    double x0r, x0i;
    load(0, x0r, x0i);
    double ur[9], ui[9], vr[9], vi[9];
#pragma unroll
    for (int j = 1; j <= 8; ++j) {
        double ar, ai, br, bi;
        load(j, ar, ai);
        load(17 - j, br, bi);
        ur[j] = ar + br;
        ui[j] = ai + bi;
        vr[j] = ar - br;
        vi[j] = ai - bi;
    }
    double s0r = x0r, s0i = x0i;
#pragma unroll
    for (int j = 1; j <= 8; ++j) {
        s0r += ur[j];
        s0i += ui[j];
    }
    store(0, s0r, s0i);
#pragma unroll
    for (int k = 1; k <= 8; ++k) {
        double pr = x0r, pi = x0i, qr = 0.0, qi = 0.0;
#pragma unroll
        for (int j = 1; j <= 8; ++j) {
            const double c = d_cos17[k][j];
            const double s = d_sin17[k][j];
            pr += c * ur[j];
            pi += c * ui[j];
            qr += s * vi[j];   /* -i*v = (v_im, -v_re) */
            qi -= s * vr[j];
        }
        store(k, pr + qr, pi + qi);
        store(17 - k, pr - qr, pi - qi);
    }
}

/* Nested variant (ported from the CPU L17_matrixsimd rounds panel_r2/r7, with
 * attribution -- the cyclic-4 (+) negacyclic-4 split of the cosine half via the
 * primitive root 3):  3^m = sig[m] f[m] mod 17, f = {1,3,8,7,4,5,2,6},
 * sig = {+,+,-,-,-,+,-,-}, IA[m] = 3^m mod 17, IB = 17-IA.  Exactly:
 *   cos(2pi f[m]f[n]/17) = c[(m+n)%8],   c[r] = cos(2pi 3^r/17)   (circulant)
 *   sin(2pi f[m]f[n]/17) = sig[m]sig[n](-1)^floor((m+n)/8) s[(m+n)%8]
 * Cosine side splits by x^8-1 = (x^4-1)(x^4+1):
 *   U_m = x_IA[m] + x_IB[m];  P_m = U_m + U_{m+4}, Q_m = U_m - U_{m+4}  (m=0..3)
 *   A_n = x0 + sum_m cp[(m+n)%4] P_m,   cp[r] = (c[r]+c[r+4])/2
 *   B_n = sum_m eps(m+n) cm[(m+n)%4] Q_m, cm[r] = (c[r]-c[r+4])/2, eps = -1 iff m+n>=4
 *   C_n = A_n + B_n, C_{n+4} = A_n - B_n,  X_0 = x0 + sum_m P_m
 * Sine side stays a dense negacyclic-8 on W_m = -i(x_IA[m] - x_IB[m]):
 *   S_n = sum_m nst[m][n] W_m,  nst[m][n] = (-1)^floor((m+n)/8) s[(m+n)%8]
 * Outputs: slot f[n] = C_n + sig[n] S_n, slot 17-f[n] = C_n - sig[n] S_n.
 * 296 scalar FP64 ops per line against the dense form's 336 (-12%), FMAs 192 vs 256. */
static __constant__ double d_cp17[4][4];   /* cp[(m+n)%4] */
static __constant__ double d_cms17[4][4];  /* eps(m+n) * cm[(m+n)%4] */
static __constant__ double d_nst17[8][8];  /* (-1)^floor((m+n)/8) * s[(m+n)%8] */

template <typename Load, typename Store>
__device__ __forceinline__ void dft17_line_nested(Load load, Store store)
{
    const int IA[8] = {1, 3, 9, 10, 13, 5, 15, 11};
    const int IB[8] = {16, 14, 8, 7, 4, 12, 2, 6};
    double x0r, x0i;
    load(0, x0r, x0i);
    double Ar[4], Ai[4], Br[4], Bi[4], Sr[8], Si[8];
    double X0r = x0r, X0i = x0i;
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        Ar[n] = x0r;
        Ai[n] = x0i;
        Br[n] = 0.0;
        Bi[n] = 0.0;
    }
#pragma unroll
    for (int n = 0; n < 8; ++n) Sr[n] = Si[n] = 0.0;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        double ar, ai, br, bi, a2r, a2i, b2r, b2i;
        load(IA[m], ar, ai);
        load(IB[m], br, bi);
        load(IA[m + 4], a2r, a2i);
        load(IB[m + 4], b2r, b2i);
        const double ur = ar + br, ui = ai + bi;       /* U_m */
        const double u2r = a2r + b2r, u2i = a2i + b2i; /* U_{m+4} */
        const double pr = ur + u2r, pi = ui + u2i;     /* P_m */
        const double qr = ur - u2r, qi = ui - u2i;     /* Q_m */
        X0r += pr;
        X0i += pi;
        /* W_m = -i(x_IA - x_IB) = (im diff, -re diff); same for m+4 */
        const double wr = ai - bi, wi = br - ar;
        const double w2r = a2i - b2i, w2i = b2r - a2r;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            const double cp = d_cp17[m][n], cm = d_cms17[m][n];
            Ar[n] += cp * pr;
            Ai[n] += cp * pi;
            Br[n] += cm * qr;
            Bi[n] += cm * qi;
        }
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const double s1 = d_nst17[m][n], s2 = d_nst17[m + 4][n];
            Sr[n] += s1 * wr + s2 * w2r;
            Si[n] += s1 * wi + s2 * w2i;
        }
    }
    store(0, X0r, X0i);
    /* slot f[n] = C_n + sig[n] S_n, slot 17-f[n] = C_n - sig[n] S_n; the CPU record's
     * cmp-checked slot table, C_{n+4} = A_n - B_n expanded in place */
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        /* C_n = A_n + B_n pairs with S_n; sig-resolved slots from the CPU table:
         * n: 0->(1,16,+) 1->(3,14,+) 2->(8,9,-) 3->(7,10,-)   for C_0..C_3
         * n+4: 4->(4,13,-) 5->(5,12,+) 6->(2,15,-) 7->(6,11,-) for C_4..C_7 */
        const int slotp[8] = {1, 3, 8, 7, 4, 5, 2, 6};      /* f[n] */
        const int sig[8] = {+1, +1, -1, -1, -1, +1, -1, -1};
        const double Cr = Ar[n] + Br[n], Ci = Ai[n] + Bi[n];
        const double Dr = Ar[n] - Br[n], Di = Ai[n] - Bi[n]; /* C_{n+4} */
        {
            const int f = slotp[n];
            const double sr = sig[n] * Sr[n], si = sig[n] * Si[n];
            store(f, Cr + sr, Ci + si);
            store(17 - f, Cr - sr, Ci - si);
        }
        {
            const int f = slotp[n + 4];
            const double sr = sig[n + 4] * Sr[n + 4], si = sig[n + 4] * Si[n + 4];
            store(f, Dr + sr, Di + si);
            store(17 - f, Dr - sr, Di - si);
        }
    }
}

#if L17_NESTED
#define DFT17_LINE dft17_line_nested
#else
#define DFT17_LINE dft17_line
#endif

/* One block transforms one whole volume: z-pass straight from global into shared,
 * y-pass in shared, x-pass from shared straight to global. */
__global__ void __launch_bounds__(NTHREADS, L17_MINB)
fft17_volume(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    extern __shared__ double2 sv[];

    const int t = threadIdx.x;
    const size_t vb = (size_t)blockIdx.x * NVOL;

#if L17_STAGED
    /* flat coalesced copy global -> shared, then pass Z runs out of shared */
#if L17_CPASYNC
#pragma unroll
    for (int i = t; i < NVOL; i += NTHREADS)
        __pipeline_memcpy_async(&sv[i], &in[vb + i], sizeof(double2));
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
#else
#pragma unroll
    for (int i = t; i < NVOL; i += NTHREADS) sv[i] = __ldg(&in[vb + i]);
    __syncthreads();
#endif
    if (t < NSQ) {
        /* pass Z in place: line (x,y) = t, contiguous in shared (odd stride 17) */
        DFT17_LINE(
            [&](int j, double &r, double &i) {
                double2 v = sv[t * NPT + j];
                r = v.x;
                i = v.y;
            },
            [&](int k, double r, double i) {
                double2 v;
                v.x = r;
                v.y = i;
                sv[t * NPT + k] = v;
            });
    }
    __syncthreads();
#else
    if (t < NSQ) {
        /* pass Z: line (x,y) = t, elements contiguous in global */
        const double2 *g = in + vb + (size_t)t * NPT;
        DFT17_LINE(
            [&](int j, double &r, double &i) {
                double2 v = __ldg(&g[j]);
                r = v.x;
                i = v.y;
            },
            [&](int k, double r, double i) {
                double2 v;
                v.x = r;
                v.y = i;
                sv[t * NPT + k] = v;
            });
    }
    __syncthreads();
#endif
    if (t < NSQ) {
        /* pass Y: line (x,z), x = t/17, z = t%17, stride 17 in shared */
        const int base = (t / NPT) * NSQ + (t % NPT);
        DFT17_LINE(
            [&](int j, double &r, double &i) {
                double2 v = sv[base + NPT * j];
                r = v.x;
                i = v.y;
            },
            [&](int k, double r, double i) {
                double2 v;
                v.x = r;
                v.y = i;
                sv[base + NPT * k] = v;
            });
    }
    __syncthreads();
    if (t < NSQ) {
        /* pass X: line (y,z) = t, stride 289 in shared; stores are 32 consecutive
         * double2 per warp per k -- perfectly coalesced */
        double2 *g = out + vb + t;
        DFT17_LINE(
            [&](int j, double &r, double &i) {
                double2 v = sv[t + NSQ * j];
                r = v.x;
                i = v.y;
            },
            [&](int k, double r, double i) {
                double2 v;
                v.x = r;
                v.y = i;
                g[(size_t)NSQ * k] = v;
            });
    }
}

/* ---- small-batch path ----------------------------------------------------------------
 * At B=1 the fused kernel is one block on one SM: latency-bound at ~15 us.  Instead split:
 *   kernel A: one block per (volume, x-plane) -- 17B blocks -- dense per-output z- then
 *             y-DFT of the 17x17 plane out of shared memory, into a global scratch;
 *   kernel B: folded x-pass, one thread per x-line, scratch -> out (L2-resident at B=1).
 * Twice the global traffic and denser arithmetic, but 17x the SM coverage; at B=1 both
 * kernels are latency-bound and the trade is a large net win.  Used only for tiny batch. */

/* w[k*17+j] = exp(-2 pi i k j / 17), 289 entries.  The constant-memory copy serves the
 * x-pass kernel, where every lane reads the SAME entry (broadcast, which is the one thing
 * the constant cache is good at); the zy kernel reads per-lane DIFFERENT entries, which
 * would serialize in constant memory, so it takes the same table via global memory. */
static __constant__ double2 d_w17[NSQ];

__device__ __forceinline__ void
fft17_zy_plane_body(const double2 *__restrict__ in, double2 *__restrict__ tmp,
                    const double2 *__restrict__ gw)
{
    __shared__ double2 pin[NSQ], pout[NSQ], w[NSQ];
    const int t = threadIdx.x;                    /* 289 threads */
    const int x = blockIdx.x % NPT;
    const size_t vb = (size_t)(blockIdx.x / NPT) * NVOL;
    const double2 *gp = in + vb + (size_t)x * NSQ;

    pin[t] = __ldg(&gp[t]);
    w[t] = __ldg(&gw[t]);
    __syncthreads();

    /* z-DFT: thread (y = t/17, kz = t%17); pin reads broadcast per y, w reads stride 17 */
    {
        const int y = t / NPT, kz = t % NPT;
        double ar = 0.0, ai = 0.0;
#pragma unroll
        for (int j = 0; j < NPT; ++j) {
            const double2 v = pin[y * NPT + j];
            const double2 c = w[kz * NPT + j];
            ar += v.x * c.x - v.y * c.y;
            ai += v.x * c.y + v.y * c.x;
        }
        double2 r;
        r.x = ar;
        r.y = ai;
        pout[y * NPT + kz] = r;
    }
    __syncthreads();

    /* y-DFT: thread (ky = t/17, kz = t%17); pout reads consecutive, w broadcast per ky */
    {
        const int ky = t / NPT, kz = t % NPT;
        double ar = 0.0, ai = 0.0;
#pragma unroll
        for (int y = 0; y < NPT; ++y) {
            const double2 v = pout[y * NPT + kz];
            const double2 c = w[ky * NPT + y];
            ar += v.x * c.x - v.y * c.y;
            ai += v.x * c.y + v.y * c.x;
        }
        double2 r;
        r.x = ar;
        r.y = ai;
        tmp[vb + (size_t)x * NSQ + t] = r;   /* t = ky*17+kz: coalesced */
    }
}

/* x-pass, dense per-output: block = (volume, kx), thread = (y,z).  All lanes read the
 * same twiddle (constant broadcast); tmp reads and out writes are 32-consecutive per
 * warp.  17B blocks of 289 threads: at B=1 this is 17x the coverage of a per-line map. */
__device__ __forceinline__ void
fft17_x_dense_body(const double2 *__restrict__ tmp, double2 *__restrict__ out)
{
    const int t = threadIdx.x;                    /* (y,z) */
    const int kx = blockIdx.x % NPT;
    const size_t vb = (size_t)(blockIdx.x / NPT) * NVOL;
    const double2 *g = tmp + vb + t;
    double ar = 0.0, ai = 0.0;
#pragma unroll
    for (int j = 0; j < NPT; ++j) {
        const double2 v = __ldg(&g[(size_t)NSQ * j]);
        const double2 c = d_w17[kx * NPT + j];
        ar += v.x * c.x - v.y * c.y;
        ai += v.x * c.y + v.y * c.x;
    }
    double2 r;
    r.x = ar;
    r.y = ai;
    out[vb + (size_t)kx * NSQ + t] = r;
}

/* Both halves in ONE cooperative launch: removes the second kernel launch and the
 * GPU-side gap between them, which at B=1 is a measurable fraction of the total. */
__global__ void __launch_bounds__(NSQ)
fft17_small_coop(const double2 *__restrict__ in, double2 *__restrict__ tmp,
                 double2 *__restrict__ out, const double2 *__restrict__ gw)
{
    fft17_zy_plane_body(in, tmp, gw);
    cg::this_grid().sync();
    fft17_x_dense_body(tmp, out);
}

/* Plain two-launch fallback if the cooperative launch is unavailable. */
__global__ void __launch_bounds__(NSQ)
fft17_zy_planes(const double2 *__restrict__ in, double2 *__restrict__ tmp,
                const double2 *__restrict__ gw)
{
    fft17_zy_plane_body(in, tmp, gw);
}

__global__ void __launch_bounds__(NSQ)
fft17_x_dense(const double2 *__restrict__ tmp, double2 *__restrict__ out)
{
    fft17_x_dense_body(tmp, out);
}

struct fft3d_gpu_plan {
    int L;
    int batch;
    double2 *tmp;     /* scratch for the small-batch split path, else NULL */
    double2 *wglob;   /* global-memory copy of the twiddle table, split path only */
    int coop;         /* split path: nonzero if the single cooperative launch is usable */
};

extern "C" const char *fft3d_gpu_name(void) { return "L17_dmma"; }
extern "C" const char *fft3d_gpu_description(void)
{
    return "one volume/block in shared, fused 3-axis conj-folded dense 17-pt DFT, "
           "cp.async staged, 1 global read + 1 write; coop plane-split at batch<=4";
}

extern "C" int fft3d_gpu_supports(int L) { return L == 17; }

extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)
{
    if (L != 17 || batch < 1) return NULL;

    /* Trig tables in long double, rounded once: table entries good to 0.5 ulp. */
    double hc[9][9], hs[9][9];
    for (int k = 0; k <= 8; ++k)
        for (int j = 0; j <= 8; ++j) {
            long double th = 2.0L * 3.14159265358979323846264338327950288L *
                             (long double)((k * j) % 17) / 17.0L;
            hc[k][j] = (double)cosl(th);
            hs[k][j] = (double)sinl(th);
        }
    if (cudaMemcpyToSymbol(d_cos17, hc, sizeof hc) != cudaSuccess) return NULL;
    if (cudaMemcpyToSymbol(d_sin17, hs, sizeof hs) != cudaSuccess) return NULL;

    /* nested-split tables: c[r], s[r] at 3^r mod 17 */
    {
        const long double PI = 3.14159265358979323846264338327950288L;
        long double c8[8], s8[8];
        int p = 1;
        for (int r = 0; r < 8; ++r) {
            c8[r] = cosl(2.0L * PI * (long double)p / 17.0L);
            s8[r] = sinl(2.0L * PI * (long double)p / 17.0L);
            p = (p * 3) % 17;
        }
        double hcp[4][4], hcm[4][4], hnst[8][8];
        for (int m = 0; m < 4; ++m)
            for (int n = 0; n < 4; ++n) {
                const int r = (m + n) % 4;
                hcp[m][n] = (double)((c8[r] + c8[r + 4]) / 2.0L);
                hcm[m][n] = (double)((c8[r] - c8[r + 4]) / 2.0L) *
                            ((m + n >= 4) ? -1.0 : 1.0);
            }
        for (int m = 0; m < 8; ++m)
            for (int n = 0; n < 8; ++n)
                hnst[m][n] = (double)s8[(m + n) % 8] * ((m + n >= 8) ? -1.0 : 1.0);
        if (cudaMemcpyToSymbol(d_cp17, hcp, sizeof hcp) != cudaSuccess) return NULL;
        if (cudaMemcpyToSymbol(d_cms17, hcm, sizeof hcm) != cudaSuccess) return NULL;
        if (cudaMemcpyToSymbol(d_nst17, hnst, sizeof hnst) != cudaSuccess) return NULL;
    }

    double2 hw[NSQ];
    for (int k = 0; k < NPT; ++k)
        for (int j = 0; j < NPT; ++j) {
            long double th = -2.0L * 3.14159265358979323846264338327950288L *
                             (long double)((k * j) % 17) / 17.0L;
            hw[k * NPT + j].x = (double)cosl(th);
            hw[k * NPT + j].y = (double)sinl(th);
        }
    if (cudaMemcpyToSymbol(d_w17, hw, sizeof hw) != cudaSuccess) return NULL;

    /* 78,608 B of dynamic shared needs the explicit sm_80 opt-in. */
    if (cudaFuncSetAttribute(fft17_volume, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             2 * NVOL * (int)sizeof(double)) != cudaSuccess)
        return NULL;
    /* Ask for the full 164 KB carveout so two blocks can share an SM. */
    cudaFuncSetAttribute(fft17_volume, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);

    fft3d_gpu_plan *p = (fft3d_gpu_plan *)malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->tmp = NULL;
    p->wglob = NULL;
    p->coop = 0;
    if (batch <= L17_SPLIT_MAX_BATCH) {
        if (cudaMalloc((void **)&p->tmp, (size_t)batch * NVOL * sizeof(double2)) !=
                cudaSuccess ||
            cudaMalloc((void **)&p->wglob, sizeof hw) != cudaSuccess ||
            cudaMemcpy(p->wglob, hw, sizeof hw, cudaMemcpyHostToDevice) != cudaSuccess) {
            if (p->tmp) cudaFree(p->tmp);
            if (p->wglob) cudaFree(p->wglob);
            free(p);
            return NULL;
        }
        /* single cooperative launch, if the device can co-schedule the whole grid */
        int dev = 0, coop_attr = 0, nblk = 0, nsm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&coop_attr, cudaDevAttrCooperativeLaunch, dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (coop_attr &&
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblk, fft17_small_coop, NSQ,
                                                          0) == cudaSuccess &&
            (long)nblk * nsm >= (long)NPT * batch)
            p->coop = 1;
    }
    return p;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (p->tmp) {
        if (p->coop) {
            void *args[] = {(void *)&in, (void *)&p->tmp, (void *)&out,
                            (void *)&p->wglob};
            cudaLaunchCooperativeKernel((void *)fft17_small_coop,
                                        dim3(NPT * p->batch), dim3(NSQ), args, 0, 0);
        } else {
            fft17_zy_planes<<<NPT * p->batch, NSQ>>>(in, p->tmp, p->wglob);
            fft17_x_dense<<<NPT * p->batch, NSQ>>>(p->tmp, out);
        }
    } else {
        fft17_volume<<<p->batch, NTHREADS, 2 * NVOL * sizeof(double)>>>(in, out);
    }
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p)
{
    if (!p) return;
    if (p->tmp) cudaFree(p->tmp);
    if (p->wglob) cudaFree(p->wglob);
    free(p);
}
