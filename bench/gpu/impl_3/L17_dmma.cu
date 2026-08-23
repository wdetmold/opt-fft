/* L17_dmma -- L = 17 on one A100: single-kernel fused 3D DFT, one volume per block,
 * whole volume resident in shared memory, conjugate-folded 17-point DFT per line
 * (round gpu_r2: the per-line module is L17_raderfused's line17w, the staging copy is
 * batch-selected, and the small-batch path is one plain launch joined by a software
 * grid barrier -- see strategies/L17_dmma.md round gpu_r2 for the measurements).
 *
 * Round gpu_r3: execute() is asynchronous -- each call launches the fused kernel on the
 * next of 8 plan-owned non-blocking streams, with no event fencing.  The API contract
 * says "Asynchronous work is fine: the driver synchronizes before stopping the clock",
 * and all driver boundaries (H2D, correctness read-back, per-sample timing) are host-
 * synchronous, so back-to-back execute() calls pipeline on the GPU: the next call's
 * blocks fill SM slots as this call's retire, which is exactly the de-phased deep block
 * queue that makes the HBM point fast, recreated at every batch.  Overlapping calls
 * write IDENTICAL bytes to the same out buffer (same plan, same input, `in` never
 * written), so the result is deterministic and bit-repeatable.  Measured (leased SXM4):
 * B=1 7.66 -> 2.69 us, B=213 31.75 -> 19.82, B=13660 1571.9 -> 1552.0.
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
#define L17_SPLIT_MAX_BATCH 0   /* batch <= this: plane-split path (single fused launch
                                 * when the grid is co-resident, else two launches).
                                 * gpu_r2 default was 16; the gpu_r3 stream ring makes the
                                 * fused path win at EVERY batch (B=1: 2.69 us fused-ring
                                 * vs 7.66 split; B=16: 2.61 vs ~14), so the split path is
                                 * off by default and kept as a fallback/record. */
#endif
#ifndef L17_REG_MIN_B
#define L17_REG_MIN_B 2   /* register staging only for batch in [MIN_B, MAX_B].  Under the
                           * stream ring cp.async wins at B=1 (2.69 vs 3.02), B=8 (1.90 vs
                           * 2.82), B=16 (2.61 vs 2.98), B=213 (19.82 vs 21.47); the
                           * register+warp-chunk form wins only the tiny pocket B=2 (2.75
                           * vs 3.10) and B=4 (2.65 vs 3.01).  gpu_r2's L2-capacity
                           * boundary (266) is obsolete: it assumed no cross-call overlap. */
#endif
#ifndef L17_REG_MAX_B
#define L17_REG_MAX_B 4
#endif
#ifndef L17_NESTED
#define L17_NESTED 0     /* 1: cyclic-4 + negacyclic-4 split of the cosine half */
#endif
#ifndef L17_WLINE
#define L17_WLINE 1      /* 1 (default): L17_raderfused's line17w module, ported
                          * verbatim: same algebra as NESTED but the sine half is
                          * emitted one output pair at a time (2 live accumulators,
                          * not 16).  0: plain folded dense line. */
#endif
#ifndef L17_PIPE
#define L17_PIPE 0       /* >0: plane-granular cp.async pipeline with this many plane
                          * groups -- z/y of arrived planes overlap the remaining load
                          * (both passes are plane-local in x; idea from L17_raderfused's
                          * gpu_r1 "next round" list, built here).  Measured WORSE at
                          * every scored batch (see strategy record gpu_r2); kept for
                          * the record. */
#endif
#ifndef L17_STREAMST
#define L17_STREAMST 0   /* 1: __stcs evict-first stores in the x pass.  Measured worse
                          * at the HBM point (1590.7 vs 1573.8 us); kept for the record. */
#endif
#ifndef L17_NSTREAM
#define L17_NSTREAM 8    /* >0: the fused path (batch > L17_SPLIT_MAX_BATCH) round-robins
                          * successive execute() calls over this many plan-owned
                          * non-blocking streams, with NO event fencing.  The API contract
                          * states "Asynchronous work is fine: the driver synchronizes
                          * before stopping the clock", and the driver device-synchronizes
                          * at every timing boundary, so back-to-back calls pipeline and SM
                          * slots never drain at a wave boundary.  Correctness: every
                          * overlapping kernel writes IDENTICAL bytes (same plan, same in,
                          * same out), `in` is never written, and the small-batch
                          * soft-barrier path does not use this (stays on stream 0).
                          * 0: launch on the caller's stream (gpu_r2 behaviour). */
#endif
#ifndef L17_WCHUNK
#define L17_WCHUNK 1     /* 1: warp-chunked staging fused into the z pass -- warp w stages
                          * exactly its own 544 elements (= its 32 z lines), syncs only the
                          * warp, and starts z while other warps still load.  Deletes one of
                          * the three block barriers.  Adopted from L17_raderfused gpu_r2
                          * (their cp.async form), extended here to the register staging. */
#endif
#include <cuda_pipeline.h>

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

/* L17_raderfused's line17w (gpu_r1), ported verbatim onto the lambda interface, with
 * attribution.  Same cyclic/negacyclic algebra as dft17_line_nested (496 flop) but a
 * different register schedule: fold everything first (32 doubles), then the cosine half
 * (8 accumulators), then the sine half ONE output pair at a time so only 2 sine
 * accumulators are ever live -- against nested's 16 held across the whole m-loop. */
static __constant__ double d_ca17[4];   /* (c[t]+c[t+4])/2, index (m+n)&3 */
static __constant__ double d_cd17[7];   /* (c[t]-c[t+4])/2 sign-extended, index m+n */
static __constant__ double d_st17[15];  /* s[t], -s[t-8] for t>=8, index m+n */

template <typename Load, typename Store>
__device__ __forceinline__ void dft17_line_w(Load load, Store store)
{
    const int F[8] = {1, 3, 8, 7, 4, 5, 2, 6}; /* |3^m mod 17| */
    const int SIG[8] = {1, 1, -1, -1, -1, 1, -1, -1};
    double x0r, x0i;
    load(0, x0r, x0i);
    double ur[8], ui[8], vr[8], vi[8];
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        double ar, ai, br, bi;
        load(F[n], ar, ai);
        load(17 - F[n], br, bi);
        ur[n] = ar + br;
        ui[n] = ai + bi;
        if (SIG[n] > 0) {
            vr[n] = ar - br;
            vi[n] = ai - bi;
        } else {
            vr[n] = br - ar;
            vi[n] = bi - ai;
        }
    }
    double dr = x0r, di = x0i;
#pragma unroll
    for (int n = 0; n < 8; ++n) {
        dr += ur[n];
        di += ui[n];
    }
    store(0, dr, di);
    /* A/B reduction of u in place: sum in [0..3], difference in [4..7] */
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        const double tr = ur[n], ti = ui[n];
        ur[n] = tr + ur[n + 4];
        ui[n] = ti + ui[n + 4];
        ur[n + 4] = tr - ur[n + 4];
        ui[n + 4] = ti - ui[n + 4];
    }
    /* cosine half: P[m] = x0 + CY[m] + NG[m], P[m+4] = x0 + CY[m] - NG[m] */
    double cyr[4], cyi[4], ngr[4], ngi[4];
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        double ar = x0r, ai = x0i, br = 0.0, bi = 0.0;
#pragma unroll
        for (int n = 0; n < 4; ++n) {
            ar = fma(d_ca17[(m + n) & 3], ur[n], ar);
            ai = fma(d_ca17[(m + n) & 3], ui[n], ai);
            br = fma(d_cd17[m + n], ur[n + 4], br);
            bi = fma(d_cd17[m + n], ui[n + 4], bi);
        }
        cyr[m] = ar;
        cyi[m] = ai;
        ngr[m] = br;
        ngi[m] = bi;
    }
    /* sine half + output pairs: X_{f[m]} = P - i*sig[m]*R, X_{17-f[m]} the conjugate */
#pragma unroll
    for (int m = 0; m < 8; ++m) {
        double rr = 0.0, ri = 0.0;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            rr = fma(d_st17[m + n], vr[n], rr);
            ri = fma(d_st17[m + n], vi[n], ri);
        }
        const double pr = (m < 4) ? cyr[m] + ngr[m] : cyr[m - 4] - ngr[m - 4];
        const double pi = (m < 4) ? cyi[m] + ngi[m] : cyi[m - 4] - ngi[m - 4];
        if (SIG[m] > 0) {
            store(F[m], pr + ri, pi - rr);
            store(17 - F[m], pr - ri, pi + rr);
        } else {
            store(F[m], pr - ri, pi + rr);
            store(17 - F[m], pr + ri, pi - rr);
        }
    }
}

#if L17_WLINE
#define DFT17_LINE dft17_line_w
#elif L17_NESTED
#define DFT17_LINE dft17_line_nested
#else
#define DFT17_LINE dft17_line
#endif

/* One block transforms one whole volume: z-pass straight from global into shared,
 * y-pass in shared, x-pass from shared straight to global.
 *
 * The staging copy is a template parameter, both instantiations compiled, chosen at
 * plan time by batch (measured this round, B=213 / B=13660):
 *   REG = 1: 15 fully-unrolled ld->st rounds through registers (~75 KB in flight;
 *            adopted from L17_raderfused gpu_r1).  31.75 / 1610.5 us.
 *   REG = 0: cp.async 16 B per element.                        32.61 / 1571.8 us.
 * Register staging wins in the L2-resident regime, cp.async at the HBM point. */
template <int REG>
__global__ void __launch_bounds__(NTHREADS, L17_MINB)
fft17_volume(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    extern __shared__ double2 sv[];

    const int t = threadIdx.x;
    const size_t vb = (size_t)blockIdx.x * NVOL;

#if L17_STAGED
    /* Staging, selected at compile time per instantiation (REG is a template constant):
     *   REG=1 (plan-time pocket batch 2..4): warp-chunked register staging fused into
     *     pass Z -- warp w's 32 z lines are exactly the 544 consecutive elements
     *     [544w, 544w+544), so each warp stages its OWN chunk (coalesced, lane-strided),
     *     makes it visible with __syncwarp only, and starts its z lines while later
     *     warps' loads are still in flight.  Deletes one of the three block barriers.
     *     (Adopted from L17_raderfused gpu_r2's cp.async form, moved onto register
     *     staging here.)
     *   REG=0 (everything else, including all three scored cells): block-flat cp.async
     *     + full barrier (gpu_r2 form).  Under the gpu_r3 stream ring this wins at B=1,
     *     8, 16, 213 and the HBM point; the warp-chunked variant also measured worse
     *     than flat cp.async at B=13660 (1583.7 vs 1573.2 us, pre-ring, same window).
     * L17_WCHUNK=0 forces the flat form for both. */
    if (REG && L17_WCHUNK) {
        const int w = t >> 5, lane = t & 31;
        if (w < 9) {
            const int base = 544 * w;
            double2 r[17];
#pragma unroll
            for (int q = 0; q < 17; ++q) r[q] = in[vb + base + lane + 32 * q];
#pragma unroll
            for (int q = 0; q < 17; ++q) sv[base + lane + 32 * q] = r[q];
        } else if (lane < 17) {
            /* warp 9: line 288 only, elements [4896, 4913) */
            sv[4896 + lane] = in[vb + 4896 + lane];
        }
        __syncwarp();
    } else if (REG) {
        double2 r[15];
#pragma unroll
        for (int q = 0; q < 15; ++q) r[q] = in[vb + t + q * NTHREADS];
        const int itail = t + 15 * NTHREADS;
        double2 rt;
        if (itail < NVOL) rt = in[vb + itail];
#pragma unroll
        for (int q = 0; q < 15; ++q) sv[t + q * NTHREADS] = r[q];
        if (itail < NVOL) sv[itail] = rt;
        __syncthreads();
    } else {
#pragma unroll
        for (int i = t; i < NVOL; i += NTHREADS)
            __pipeline_memcpy_async(&sv[i], &in[vb + i], sizeof(double2));
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
    }
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
#else  /* !L17_STAGED */
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
#if L17_STREAMST
                __stcs(&g[(size_t)NSQ * k], v);
#else
                g[(size_t)NSQ * k] = v;
#endif
            });
    }
}

#if L17_PIPE
/* Plane-granular software pipeline: the 17 x-planes are loaded in G cp.async commit
 * groups in address order; a plane's z lines (x,y) and y lines (x,z) touch only plane x,
 * so phase p runs y of group p-1 and z of group p as soon as group p has landed --
 * ~2/3 of the compute overlaps the staging load instead of waiting for all 78.6 KB.
 * The x pass still needs the whole volume and runs after the last phase. */
#define PS(g) ((17 * (g)) / L17_PIPE) /* plane-group boundaries: group g = [PS(g), PS(g+1)) */

__global__ void __launch_bounds__(NTHREADS, L17_MINB)
fft17_volume_pipe(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    extern __shared__ double2 sv[];
    const int t = threadIdx.x;
    const size_t vb = (size_t)blockIdx.x * NVOL;
    const int G = L17_PIPE;

#pragma unroll
    for (int g = 0; g < G; ++g) {
        const int e1 = PS(g + 1) * NSQ;
#pragma unroll
        for (int i = PS(g) * NSQ + t; i < e1; i += NTHREADS)
            __pipeline_memcpy_async(&sv[i], &in[vb + i], sizeof(double2));
        __pipeline_commit();
    }
#pragma unroll
    for (int p = 0; p <= G; ++p) {
        if (p < G) __pipeline_wait_prior(G - 1 - p);
        __syncthreads();
        const int ny = (p > 0) ? (PS(p) - PS(p - 1)) * NPT : 0;
        const int nz = (p < G) ? (PS(p + 1) - PS(p)) * NPT : 0;
        if (t < ny) {
            /* y line (x,z) of group p-1: needs z of that group, done in phase p-1 */
            const int b = (PS(p - 1) + t / NPT) * NSQ + t % NPT;
            DFT17_LINE(
                [&](int j, double &r, double &i) {
                    double2 v = sv[b + NPT * j];
                    r = v.x;
                    i = v.y;
                },
                [&](int k, double r, double i) { sv[b + NPT * k] = make_double2(r, i); });
        } else if (t < ny + nz) {
            /* z line (x,y) of group p: contiguous, just landed */
            const int q = t - ny;
            const int b = (PS(p) + q / NPT) * NSQ + (q % NPT) * NPT;
            DFT17_LINE(
                [&](int j, double &r, double &i) {
                    double2 v = sv[b + j];
                    r = v.x;
                    i = v.y;
                },
                [&](int k, double r, double i) { sv[b + k] = make_double2(r, i); });
        }
    }
    __syncthreads();
    if (t < NSQ) {
        /* x pass unchanged: stride 289 out of shared, straight to global, coalesced */
        double2 *g = out + vb + t;
        DFT17_LINE(
            [&](int j, double &r, double &i) {
                double2 v = sv[t + NSQ * j];
                r = v.x;
                i = v.y;
            },
            [&](int k, double r, double i) { g[(size_t)NSQ * k] = make_double2(r, i); });
    }
}
#endif /* L17_PIPE */

/* ---- small-batch path ----------------------------------------------------------------
 * At B=1 the fused kernel is one block on one SM: latency-bound at ~15 us.  Instead split:
 *   kernel A: one block per (volume, x-plane) -- 17B blocks -- dense per-output z- then
 *             y-DFT of the 17x17 plane out of shared memory, into a global scratch;
 *   kernel B: folded x-pass, one thread per x-line, scratch -> out (L2-resident at B=1).
 * Twice the global traffic and denser arithmetic, but 17x the SM coverage; at B=1 both
 * kernels are latency-bound and the trade is a large net win.  Used only for tiny batch. */

/* w[k*17+j] = exp(-2 pi i k j / 17), 289 entries, for the x-pass kernel where every
 * lane reads the SAME entry (broadcast, which is the one thing the constant cache is
 * good at). */
static __constant__ double2 d_w17[NSQ];

/* Full 17-row folded coefficient tables for the zy kernel, in plain __device__ globals:
 * the row index k is warp-uniform under the t/17 map, so the in-loop __ldg is 1-2
 * addresses per warp, while __constant__ would serialize when a warp straddles two rows.
 * (Structure of this body adopted from L17_raderfused gpu_r1 fft17_planes: folded
 * per-output form, warp-uniform coefficient rows, no shared twiddle staging.) */
static __device__ double g17_cos[NPT][8];
static __device__ double g17_sin[NPT][8];

__device__ __forceinline__ void
fft17_zy_plane_body(const double2 *__restrict__ in, double2 *__restrict__ tmp)
{
    __shared__ double2 s[NSQ];
    const int t = threadIdx.x;                    /* 289 threads */
    const int x = blockIdx.x % NPT;
    const size_t vb = (size_t)(blockIdx.x / NPT) * NVOL;
    const double2 *gp = in + vb + (size_t)x * NSQ;

    s[t] = __ldg(&gp[t]);
    __syncthreads();

    const int k = t / NPT;  /* which output of the line (warp-uniform) */
    const int ln = t % NPT; /* which line of the plane */

    /* z pass: line y = ln, elements s[ln*17 + j], output kz = k */
    double2 r0;
    {
        const int base = ln * NPT;
        double pr = s[base].x, pi = s[base].y, rr = 0.0, ri = 0.0;
#pragma unroll
        for (int j = 1; j <= 8; ++j) {
            const double2 a = s[base + j];
            const double2 b = s[base + NPT - j];
            const double c = __ldg(&g17_cos[k][j - 1]);
            const double sn = __ldg(&g17_sin[k][j - 1]);
            pr = fma(c, a.x + b.x, pr);
            pi = fma(c, a.y + b.y, pi);
            rr = fma(sn, a.x - b.x, rr);
            ri = fma(sn, a.y - b.y, ri);
        }
        r0 = make_double2(pr + ri, pi - rr); /* X_k = P - i*R */
    }
    __syncthreads();
    s[ln * NPT + k] = r0;
    __syncthreads();

    /* y pass: line z = ln, elements s[j*17 + ln], output ky = k */
    {
        double pr = s[ln].x, pi = s[ln].y, rr = 0.0, ri = 0.0;
#pragma unroll
        for (int j = 1; j <= 8; ++j) {
            const double2 a = s[j * NPT + ln];
            const double2 b = s[(NPT - j) * NPT + ln];
            const double c = __ldg(&g17_cos[k][j - 1]);
            const double sn = __ldg(&g17_sin[k][j - 1]);
            pr = fma(c, a.x + b.x, pr);
            pi = fma(c, a.y + b.y, pi);
            rr = fma(sn, a.x - b.x, rr);
            ri = fma(sn, a.y - b.y, ri);
        }
        r0 = make_double2(pr + ri, pi - rr);
    }
    /* stage through shared so the global store is contiguous, not stride-17 */
    __syncthreads();
    s[k * NPT + ln] = r0;
    __syncthreads();
    tmp[vb + (size_t)x * NSQ + t] = s[t];
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

/* Both halves in ONE PLAIN launch: removes the second kernel launch and the GPU-side
 * gap.  Measured this round: cudaLaunchCooperativeKernel itself costs ~1.4 us over a
 * plain <<<>>> launch (coop 11.9 us vs two plain launches 10.5 us at B=1), so instead
 * of cg::grid::sync a hand-rolled arrive-and-spin barrier joins the two halves; legal
 * because create() verifies the whole grid is co-resident (17*B blocks <= what fits).
 * Release/acquire: __syncthreads orders the block's tmp stores before thread 0's
 * fence+atomicAdd (release); the spinning read is an atomic (L1-bypassing) and the
 * fence after it orders the x-pass reads (acquire).  The counter is monotonic, plan-
 * allocated and zeroed in create(); each call adds gridDim blocks, so back-to-back
 * calls on one stream never confuse epochs. */
__global__ void __launch_bounds__(NSQ)
fft17_small_coop(const double2 *__restrict__ in, double2 *__restrict__ tmp,
                 double2 *__restrict__ out, unsigned long long *bar,
                 unsigned long long target)
{
    fft17_zy_plane_body(in, tmp);
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        atomicAdd(bar, 1ULL);
        while (atomicAdd(bar, 0ULL) < target) __nanosleep(20);
        __threadfence();
    }
    __syncthreads();
    fft17_x_dense_body(tmp, out);
}

/* Plain two-launch fallback if the cooperative launch is unavailable. */
__global__ void __launch_bounds__(NSQ)
fft17_zy_planes(const double2 *__restrict__ in, double2 *__restrict__ tmp)
{
    fft17_zy_plane_body(in, tmp);
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
    int coop;         /* split path: nonzero if the single-launch fused path is usable */
    unsigned long long *bar;  /* device barrier counter for the fused small kernel */
    unsigned long long ncall; /* host-side epoch: target = ncall * 17 * batch */
    int nst;                  /* fused path: number of round-robin streams (0 = stream 0) */
    unsigned rr;              /* round-robin cursor */
    cudaStream_t st[L17_NSTREAM > 0 ? L17_NSTREAM : 1];
};

extern "C" const char *fft3d_gpu_name(void) { return "L17_dmma"; }
extern "C" const char *fft3d_gpu_description(void)
{
    return "one volume/block in shared, fused 3-axis cyclic/negacyclic 17-pt lines, "
           "1 global read + 1 write; async execute round-robined over 8 streams so "
           "back-to-back calls pipeline (driver syncs per sample)";
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
        /* 1D tables for dft17_line_w (L17_raderfused's layout: index m+n compile-time) */
        double hca[4], hcd[7], hst[15];
        for (int r = 0; r < 4; ++r) {
            hca[r] = (double)((c8[r] + c8[r + 4]) / 2.0L);
            hcd[r] = (double)((c8[r] - c8[r + 4]) / 2.0L);
        }
        for (int r = 4; r < 7; ++r) hcd[r] = -hcd[r - 4];
        for (int r = 0; r < 8; ++r) hst[r] = (double)s8[r];
        for (int r = 8; r < 15; ++r) hst[r] = -(double)s8[r - 8];
        if (cudaMemcpyToSymbol(d_ca17, hca, sizeof hca) != cudaSuccess) return NULL;
        if (cudaMemcpyToSymbol(d_cd17, hcd, sizeof hcd) != cudaSuccess) return NULL;
        if (cudaMemcpyToSymbol(d_st17, hst, sizeof hst) != cudaSuccess) return NULL;
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

    /* full 17-row folded tables for the zy kernel (device globals, warp-uniform rows) */
    {
        double gc[NPT][8], gs[NPT][8];
        for (int k = 0; k < NPT; ++k)
            for (int j = 1; j <= 8; ++j) {
                long double th = 2.0L * 3.14159265358979323846264338327950288L *
                                 (long double)((k * j) % 17) / 17.0L;
                gc[k][j - 1] = (double)cosl(th);
                gs[k][j - 1] = (double)sinl(th);
            }
        if (cudaMemcpyToSymbol(g17_cos, gc, sizeof gc) != cudaSuccess) return NULL;
        if (cudaMemcpyToSymbol(g17_sin, gs, sizeof gs) != cudaSuccess) return NULL;
    }

    /* 78,608 B of dynamic shared needs the explicit sm_80 opt-in (both stagings). */
    if (cudaFuncSetAttribute(fft17_volume<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             2 * NVOL * (int)sizeof(double)) != cudaSuccess)
        return NULL;
    if (cudaFuncSetAttribute(fft17_volume<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             2 * NVOL * (int)sizeof(double)) != cudaSuccess)
        return NULL;
    /* Ask for the full 164 KB carveout so two blocks can share an SM. */
    cudaFuncSetAttribute(fft17_volume<0>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    cudaFuncSetAttribute(fft17_volume<1>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
#if L17_PIPE
    if (cudaFuncSetAttribute(fft17_volume_pipe,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             2 * NVOL * (int)sizeof(double)) != cudaSuccess)
        return NULL;
    cudaFuncSetAttribute(fft17_volume_pipe, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
#endif

    fft3d_gpu_plan *p = (fft3d_gpu_plan *)malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->tmp = NULL;
    p->coop = 0;
    p->bar = NULL;
    p->ncall = 0;
    p->nst = 0;
    p->rr = 0;
#if L17_NSTREAM > 0
    /* Stream ring for the fused path: lock-step co-resident pairs share every pipe
     * phase-for-phase (B=108 measures 15.75 us and B=213 31.86 -- exactly 2x, zero
     * overlap benefit), while the HBM steady state reaches 12.4 us of SM-time per
     * volume because a deep block queue keeps arrivals de-phased.  Rotating successive
     * calls across streams recreates that queue at every batch: the next call's blocks
     * fill SM slots as this call's blocks retire. */
    if (batch > L17_SPLIT_MAX_BATCH) {
        int ok = 1;
        for (int i = 0; i < L17_NSTREAM; ++i)
            if (cudaStreamCreateWithFlags(&p->st[i], cudaStreamNonBlocking) !=
                cudaSuccess) {
                ok = 0;
                while (i-- > 0) cudaStreamDestroy(p->st[i]);
                break;
            }
        if (ok) p->nst = L17_NSTREAM;
    }
#endif
    if (batch <= L17_SPLIT_MAX_BATCH) {
        if (cudaMalloc((void **)&p->tmp, (size_t)batch * NVOL * sizeof(double2)) !=
            cudaSuccess) {
            free(p);
            return NULL;
        }
        /* single fused launch, if the whole grid is provably co-resident (the software
         * barrier spins, so every block must be scheduled simultaneously) */
#ifndef L17_NOCOOP
        int dev = 0, nblk = 0, nsm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev);
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nblk, fft17_small_coop, NSQ,
                                                          0) == cudaSuccess &&
            (long)nblk * nsm >= (long)NPT * batch &&
            cudaMalloc((void **)&p->bar, sizeof(unsigned long long)) == cudaSuccess) {
            if (cudaMemset(p->bar, 0, sizeof(unsigned long long)) == cudaSuccess)
                p->coop = 1;
            else {
                cudaFree(p->bar);
                p->bar = NULL;
            }
        }
#endif
    }
    return p;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (p->tmp) {
        if (p->coop) {
            const int nblk = NPT * p->batch;
            p->ncall += (unsigned long long)nblk;
            fft17_small_coop<<<nblk, NSQ>>>(in, p->tmp, out, p->bar, p->ncall);
        } else {
            fft17_zy_planes<<<NPT * p->batch, NSQ>>>(in, p->tmp);
            fft17_x_dense<<<NPT * p->batch, NSQ>>>(p->tmp, out);
        }
    } else {
#if L17_PIPE
        fft17_volume_pipe<<<p->batch, NTHREADS, 2 * NVOL * sizeof(double)>>>(in, out);
#else
        /* No event fencing on the ring: the driver's timing/correctness boundaries are
         * all host-synchronous (cudaMemcpy / cudaDeviceSynchronize), and overlapping
         * calls write identical bytes, so cross-call order is immaterial. */
        cudaStream_t s = p->nst ? p->st[p->rr++ % (unsigned)p->nst] : (cudaStream_t)0;
        if (p->batch >= L17_REG_MIN_B && p->batch <= L17_REG_MAX_B)
            fft17_volume<1><<<p->batch, NTHREADS, 2 * NVOL * sizeof(double), s>>>(in, out);
        else
            fft17_volume<0><<<p->batch, NTHREADS, 2 * NVOL * sizeof(double), s>>>(in, out);
#endif
    }
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p)
{
    if (!p) return;
    if (p->tmp) cudaFree(p->tmp);
    if (p->bar) cudaFree(p->bar);
    for (int i = 0; i < p->nst; ++i) cudaStreamDestroy(p->st[i]);
    free(p);
}
