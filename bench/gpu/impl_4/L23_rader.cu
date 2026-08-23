/* L23_rader -- L = 23 on one A100.
 *
 * ARITHMETIC (settled by the CPU phase, ../../geom/strategies/L23_rader.md r6):
 * at p = 23, Rader-proper buys nothing: p-1 = 22 = 2*11 and the conv-11 pair has no
 * realization under 121 fused FMAs, so the honest optimal "Rader" IS the conjugate-folded
 * dense form.  Per 23-point line: u_j = x_j + x_{23-j}, d_j = x_j - x_{23-j};
 * X_k = P_k - i*Q_k, X_{23-k} = P_k + i*Q_k with P_k = x_0 + sum_j cos(2pi jk/23) u_j and
 * Q_k = sum_j sin(2pi jk/23) d_j.  All coefficients real; the -i is a free component swap.
 * 594 real FP instructions per line (484 FMA + 110 add/sub), 943k per volume.
 *
 * CANONICAL ACCUMULATION ORDER (new in r2): every line engine accumulates each P/Q sum
 * as TWO independent chains, j = 1..6 (seeded with x0) and j = 7..11 (seeded with 0),
 * added at the end.  This is what lets the split-k kernel (half a line per thread,
 * partials combined with __shfl_xor) be bit-identical to the one-thread-per-line and
 * pair-per-thread engines -- the tuner may pick any of them, and tryout.sh re-runs the
 * whole binary (including the tuner) to check bitwise repeatability, so ALL engines must
 * agree to the last bit.  Same trick as r1's "fine matches coarse order", one level up.
 *
 * STRUCTURE (190.1 KiB/volume does not fit one block's 163 KiB shared -- PANEL_BRIEF):
 * two-pass. Kernel A: per x-plane, z then y transforms in shared memory (a 23x23 plane is
 * 8.46 KiB; stride 23 is odd so every shared access pattern is bank-conflict-free).
 * Kernel B: x axis, warp-coalesced global-to-global.  Three interchangeable B engines:
 *   coarse  -- one line per thread (r1 form; 128 regs, 512 threads/SM ceiling)
 *   split-k -- TWO threads per line, each folding half the j's (~80 regs), full P/Q
 *              rebuilt with 4 __shfl_xor(16) per k-pair: the r1 ncu verdict was
 *              "latency-bound at 2.5-3 warps/scheduler", and this is the register escape
 *   fine    -- one thread per conjugate output pair (12/line, ~30 regs, 1.86x FP) --
 *              r1's B=1 winner, now raced at every batch instead of only batch<=32
 * Kernel A likewise has coarse (P planes staged, one line/thread) and fine (kernAf) forms.
 *
 * NOSCRATCH (new in r2, borrowed from L36_sharedtiled/L64_radix8 gpu_r1): kernel A can
 * write straight to `out` and kernel B transform it IN PLACE (every engine loads its
 * line completely -- through registers or a shuffle-synchronized partner -- before its
 * first store).  out-as-intermediate uses ONE L2 line per element where a scratch tmp
 * needs the tmp line PLUS a cold `out` line; L36 measured the scratch variant 10% slower.
 * The fine B engine spreads one line over 12 threads in different blocks, so it cannot
 * run in place and keeps the tmp path.
 *
 * L2 chunking + stream rotation as in r1: chunk c on stream (c % NSTR) with tmp (c % NSTR)
 * (or the matching out region when noscratch); all hazards ordered by stream identity.
 *
 * B=1: a captured CUDA graph replay (borrowed from L36_sharedtiled r1, where it beat both
 * plain launches and a cooperative fused kernel) is raced against plain launches whenever
 * the whole batch is a single chunk; keyed on the (in,out) pointers, recaptured if they
 * change, so the driver's buffer poke stays safe.
 *
 * create() autotunes (A engine/shape, B engine/TB, chunk, noscratch, uload, graph) with
 * CUDA events; the pick is reported in fft3d_gpu_description() for the leaderboard record.
 *
 * PERSISTENT TICKET KERNEL (new in r4, ported from L36_globalpass gpu_r2 via L45_pfa
 * gpu_r3): one launch per execute, grid = one resident wave (occupancy-probed).  Blocks
 * loop pulling tickets off a global atomic; a ticket is either one K1 plane-group
 * (P x-planes: load, z, y, write to out -- the kernA body) or one K2 x-line tile
 * (T lines of out transformed in place -- the kernB body).  Dispatch order: LEAD
 * volumes of K1 runway, then per volume v the tickets K1(v+LEAD) and K2(v) interleaved
 * nA:nB by Bresenham, then a K2-only tail.  K2 tickets poll a per-volume done counter
 * (release: __threadfence + atomicAdd by the producer; acquire: poll + __nanosleep +
 * fence).  Deadlock-free by their argument: the grid is at most one resident wave and
 * every K1 ticket of v is dispatched before v's first K2 ticket, so a grabbed K1 ticket
 * is always held by a running block and K1 tickets never wait.  Unlike L36 I do NOT
 * keep host-side epoch bases: execute() resets the counters with one cudaMemsetAsync
 * on the same stream (~22 KB at the HBM batch, ordered by stream identity) -- this
 * makes the launch args constant, so the kernel stays CUDA-graph-capturable and the
 * tuner's copy-by-value plan trials cannot desynchronize an epoch.  Bodies are
 * __noinline__ per L45_pfa r3's lesson (the inlined union frame spilled under the
 * launch-bounds cap and cost them 1.8x until isolated).
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
/* same values as device globals: the medium-grain kernels stage them into shared,
 * because their pair phases index the table by a per-THREAD k, and multi-address
 * constant-cache reads serialize while multi-address shared reads do not */
__device__ double g_cosd[121];
__device__ double g_sind[121];

template <bool STREAM_OUT>
static __device__ __forceinline__ void st_out(double2 *p, double2 v)
{
    if (STREAM_OUT) __stcs(p, v); else *p = v;
}

/* Load modes for global pointers that are read for the LAST time (the B pass reads
 * the intermediate exactly once, then the line is overwritten or dead) -- L64's r1
 * lesson that the cache hints are the mechanism, not a polish.  0 = plain (the A pass,
 * whose dft23 operates on SHARED pointers -- a hint there would be illegal), 1 = __ldcs
 * (evict-first), 2 = __ldlu (last-use: the exact semantic for the in-place B read;
 * borrowed from L36_sharedtiled r2 / L45_pfa r2's CHUNKED hint set). */
template <int LDM>
static __device__ __forceinline__ double2 ld_in(const double2 *p)
{
    return LDM == 2 ? __ldlu(p) : (LDM == 1 ? __ldcs(p) : *p);
}

/* Folded dense 23-point DFT of a strided line, one thread.  All loads complete (into
 * the fold registers) before the first store, so pin == pout with equal strides is safe.
 * 484 FMA + 110 add/sub.  Canonical two-chain accumulation (see header). */
template <bool STREAM_OUT = false, int LDM = 0>
static __device__ __forceinline__ void dft23(const double2 *pin, int sin_,
                                             double2 *pout, int sout)
{
    double ur[11], ui[11], dr[11], di[11];
    const double2 a0 = ld_in<LDM>(pin);
#pragma unroll
    for (int j = 0; j < 11; ++j) {
        const double2 a = ld_in<LDM>(pin + (j + 1) * sin_);
        const double2 b = ld_in<LDM>(pin + (22 - j) * sin_);
        ur[j] = a.x + b.x;  ui[j] = a.y + b.y;
        dr[j] = a.x - b.x;  di[j] = a.y - b.y;
    }
    {
        double s0r = a0.x, s0i = a0.y, s1r = 0.0, s1i = 0.0;
#pragma unroll
        for (int j = 0; j < 6; ++j)  { s0r += ur[j]; s0i += ui[j]; }
#pragma unroll
        for (int j = 6; j < 11; ++j) { s1r += ur[j]; s1i += ui[j]; }
        st_out<STREAM_OUT>(pout, make_double2(s0r + s1r, s0i + s1i));
    }
#pragma unroll
    for (int k = 0; k < 11; ++k) {
        double pr0 = a0.x, pi0 = a0.y, qr0 = 0.0, qi0 = 0.0;
        double pr1 = 0.0,  pi1 = 0.0,  qr1 = 0.0, qi1 = 0.0;
#pragma unroll
        for (int j = 0; j < 6; ++j) {
            const double c = c_cos[k * 11 + j];
            const double s = c_sin[k * 11 + j];
            pr0 = fma(c, ur[j], pr0);  pi0 = fma(c, ui[j], pi0);
            qr0 = fma(s, dr[j], qr0);  qi0 = fma(s, di[j], qi0);
        }
#pragma unroll
        for (int j = 6; j < 11; ++j) {
            const double c = c_cos[k * 11 + j];
            const double s = c_sin[k * 11 + j];
            pr1 = fma(c, ur[j], pr1);  pi1 = fma(c, ui[j], pi1);
            qr1 = fma(s, dr[j], qr1);  qi1 = fma(s, di[j], qi1);
        }
        const double pr = pr0 + pr1, pi = pi0 + pi1;
        const double qr = qr0 + qr1, qi = qi0 + qi1;
        st_out<STREAM_OUT>(pout + (k + 1)  * sout,
                           make_double2(pr + qi, pi - qr));       /* X_k      = P - iQ */
        st_out<STREAM_OUT>(pout + (22 - k) * sout,
                           make_double2(pr - qi, pi + qr));       /* X_{23-k} = P + iQ */
    }
}

/* Kernel A: z then y over P consecutive x-planes staged through shared memory.
 * grid = (ceil(23/P), nvol); block = T threads; shared = P*529*16 B. */
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
     * to keep L2 for the intermediate. */
    if (ULOAD && np == P) {
        /* All loads issued before any store: kills the serial-latency chain of the
           naive loop (r1: 14 us for one plane-group at B=1).  Only raced at small
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
        go straight to global (runs of 23 double2 per warp stay coalesced) */
        const int pp = l / LL, z = l % LL;
        double2 *col = sh + pp * LL2 + z;
        dft23(col, LL, out + base + (size_t)pp * LL2 + z, LL);
    }
}

/* Fine-grained line transform: one thread per conjugate OUTPUT PAIR (k, 23-k),
 * 12 threads per line, ~30 live registers, 1.86x the FP of dft23 but 12x the threads.
 * Canonical two-chain accumulation, bit-identical to dft23. */
template <bool STREAM_OUT>
static __device__ __forceinline__ void dft23_pair(const double2 *pin, int sin_,
                                                  double2 *pout, int sout, int p)
{
    const double2 a0 = pin[0];
    if (p == 0) {                                  /* k = 0: plain sum */
        double sr0 = a0.x, si0 = a0.y, sr1 = 0.0, si1 = 0.0;
#pragma unroll
        for (int j = 0; j < 6; ++j) {
            const double2 a = pin[(j + 1) * sin_];
            const double2 b = pin[(22 - j) * sin_];
            sr0 += a.x + b.x;  si0 += a.y + b.y;
        }
#pragma unroll
        for (int j = 6; j < 11; ++j) {
            const double2 a = pin[(j + 1) * sin_];
            const double2 b = pin[(22 - j) * sin_];
            sr1 += a.x + b.x;  si1 += a.y + b.y;
        }
        st_out<STREAM_OUT>(pout, make_double2(sr0 + sr1, si0 + si1));
        return;
    }
    const int k = p - 1;
    double pr0 = a0.x, pi0 = a0.y, qr0 = 0.0, qi0 = 0.0;
    double pr1 = 0.0,  pi1 = 0.0,  qr1 = 0.0, qi1 = 0.0;
#pragma unroll
    for (int j = 0; j < 6; ++j) {
        const double2 a = pin[(j + 1) * sin_];
        const double2 b = pin[(22 - j) * sin_];
        const double c = c_cos[k * 11 + j];
        const double s = c_sin[k * 11 + j];
        pr0 = fma(c, a.x + b.x, pr0);  pi0 = fma(c, a.y + b.y, pi0);
        qr0 = fma(s, a.x - b.x, qr0);  qi0 = fma(s, a.y - b.y, qi0);
    }
#pragma unroll
    for (int j = 6; j < 11; ++j) {
        const double2 a = pin[(j + 1) * sin_];
        const double2 b = pin[(22 - j) * sin_];
        const double c = c_cos[k * 11 + j];
        const double s = c_sin[k * 11 + j];
        pr1 = fma(c, a.x + b.x, pr1);  pi1 = fma(c, a.y + b.y, pi1);
        qr1 = fma(s, a.x - b.x, qr1);  qi1 = fma(s, a.y - b.y, qi1);
    }
    const double pr = pr0 + pr1, pi = pi0 + pi1;
    const double qr = qr0 + qr1, qi = qi0 + qi1;
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

/* Fine kernel B: 12 pair-threads per x-line, straight global-to-global.  One line's
 * 12 threads land in DIFFERENT blocks, so this engine can never run in place. */
__global__ void __launch_bounds__(256, 4)
kernBf(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= 12 * LL2) return;
    const int p = t / LL2, line = t % LL2;
    const size_t base = (size_t)blockIdx.y * LL3 + line;
    dft23_pair<true>(in + base, LL2, out + base, LL2, p);
}

/* One conjugate output pair from a line whose u/d folds are ALREADY in shared memory
 * (slot 0: x0, slot j: u_j, slot 23-j: d_j).  This is the medium-grain engine: unlike
 * dft23_pair it does not re-fold (1.08x the coarse FP instead of 1.86x), and unlike
 * dft23 it holds no fold set (~1/3 the registers).  Chains are canonical, the folds
 * are the same rounded adds, so outputs stay bit-identical to every other engine.
 * Borrowed idea: L45_pfa r1's "unit-parallel" lesson (thread-per-line >204 regs ->
 * unit-per-thread 96 regs was their round's central step). */
template <bool STREAM_OUT>
static __device__ __forceinline__ void pair_folded(const double2 *f, int fs,
                                                   double2 *o, int os, int p,
                                                   const double *cs, const double *sn)
{
    const double2 x0 = f[0];
    if (p == 0) {
        double sr0 = x0.x, si0 = x0.y, sr1 = 0.0, si1 = 0.0;
#pragma unroll
        for (int j = 1; j <= 6; ++j)  { const double2 u = f[j * fs]; sr0 += u.x; si0 += u.y; }
#pragma unroll
        for (int j = 7; j <= 11; ++j) { const double2 u = f[j * fs]; sr1 += u.x; si1 += u.y; }
        st_out<STREAM_OUT>(o, make_double2(sr0 + sr1, si0 + si1));
        return;
    }
    const int k = p - 1;
    double pr0 = x0.x, pi0 = x0.y, qr0 = 0.0, qi0 = 0.0;
    double pr1 = 0.0,  pi1 = 0.0,  qr1 = 0.0, qi1 = 0.0;
#pragma unroll
    for (int j = 1; j <= 6; ++j) {
        const double2 u = f[j * fs];
        const double2 d = f[(LL - j) * fs];
        const double c = cs[k * 11 + j - 1];
        const double s = sn[k * 11 + j - 1];
        pr0 = fma(c, u.x, pr0);  pi0 = fma(c, u.y, pi0);
        qr0 = fma(s, d.x, qr0);  qi0 = fma(s, d.y, qi0);
    }
#pragma unroll
    for (int j = 7; j <= 11; ++j) {
        const double2 u = f[j * fs];
        const double2 d = f[(LL - j) * fs];
        const double c = cs[k * 11 + j - 1];
        const double s = sn[k * 11 + j - 1];
        pr1 = fma(c, u.x, pr1);  pi1 = fma(c, u.y, pi1);
        qr1 = fma(s, d.x, qr1);  qi1 = fma(s, d.y, qi1);
    }
    const double pr = pr0 + pr1, pi = pi0 + pi1;
    const double qr = qr0 + qr1, qi = qi0 + qi1;
    st_out<STREAM_OUT>(o + (k + 1)  * os, make_double2(pr + qi, pi - qr));
    st_out<STREAM_OUT>(o + (22 - k) * os, make_double2(pr - qi, pi + qr));
}

/* Medium-grain kernel A: one x-plane per block, each axis pass = a FOLD phase (one
 * (line,j) u/d fold per thread, in place -- each unit owns its two slots) then a PAIR
 * phase (one output pair per thread).  ~36 regs against the coarse engine's 128, so
 * the 512-threads/SM register ceiling that r1 diagnosed as the wall lifts to ~1440. */
#define TAM 288
__global__ void __launch_bounds__(TAM)
kernAm(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    __shared__ double2 s0[LL2], s1[LL2];
    __shared__ double shc[121], shs[121];
    const size_t base = (size_t)blockIdx.y * LL3 + (size_t)blockIdx.x * LL2;
    for (int i = threadIdx.x; i < 121; i += TAM) {
        shc[i] = g_cosd[i];  shs[i] = g_sind[i];
    }
    for (int i = threadIdx.x; i < LL2; i += TAM)
        s0[i] = __ldcs(in + base + i);
    __syncthreads();
    /* z pass: rows of s0 */
    if (threadIdx.x < 23 * 11) {                   /* fold (line, j), in place */
        const int r = threadIdx.x / 11, j = threadIdx.x % 11 + 1;
        double2 *row = s0 + r * LL;
        const double2 a = row[j], b = row[LL - j];
        row[j]      = make_double2(a.x + b.x, a.y + b.y);   /* u_j */
        row[LL - j] = make_double2(a.x - b.x, a.y - b.y);   /* d_j */
    }
    __syncthreads();
    if (threadIdx.x < 23 * 12) {                   /* pairs: rows of s0 -> rows of s1 */
        const int r = threadIdx.x / 12, p = threadIdx.x % 12;
        pair_folded<false>(s0 + r * LL, 1, s1 + r * LL, 1, p, shc, shs);
    }
    __syncthreads();
    /* y pass: columns of s1 */
    if (threadIdx.x < 23 * 11) {
        const int c = threadIdx.x / 11, j = threadIdx.x % 11 + 1;
        double2 *col = s1 + c;
        const double2 a = col[j * LL], b = col[(LL - j) * LL];
        col[j * LL]        = make_double2(a.x + b.x, a.y + b.y);
        col[(LL - j) * LL] = make_double2(a.x - b.x, a.y - b.y);
    }
    __syncthreads();
    if (threadIdx.x < 23 * 12) {                   /* pairs: columns -> global.  q = p*23+c
        makes the store address base+q: contiguous across the whole phase */
        const int p = threadIdx.x / 23, c = threadIdx.x % 23;
        pair_folded<false>(s1 + c, LL, out + base + c, LL, p, shc, shs);
    }
}

/* Medium-grain kernel B: a tile of 32 x-lines per block staged through shared
 * (coalesced 512 B runs per x), fold in place, pairs write straight to global
 * (fixed p, consecutive lines -> contiguous).  In place (noscratch) is safe per
 * block: global reads all precede the load-phase barrier, and tiles are disjoint. */
#define TBM 384
__global__ void __launch_bounds__(TBM)
kernBm(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    __shared__ double2 s0[23 * 32];
    __shared__ double shc[121], shs[121];
    const int l0 = blockIdx.x * 32;
    const int nl = (LL2 - l0 < 32) ? (LL2 - l0) : 32;
    const size_t base = (size_t)blockIdx.y * LL3 + l0;
    for (int i = threadIdx.x; i < 121; i += TBM) {
        shc[i] = g_cosd[i];  shs[i] = g_sind[i];
    }
#pragma unroll
    for (int it = 0; it < 2; ++it) {               /* 736 = 2*TBM - 32 slots */
        const int i = threadIdx.x + it * TBM;
        if (i < 23 * 32) {
            const int x = i >> 5, l = i & 31;
            if (l < nl) s0[i] = ld_in<2>(in + base + (size_t)x * LL2 + l);
        }
    }
    __syncthreads();
    if (threadIdx.x < 32 * 11) {                   /* fold (line, j), in place */
        const int l = threadIdx.x & 31, j = (threadIdx.x >> 5) + 1;
        if (l < nl) {
            const double2 a = s0[j * 32 + l], b = s0[(LL - j) * 32 + l];
            s0[j * 32 + l]        = make_double2(a.x + b.x, a.y + b.y);
            s0[(LL - j) * 32 + l] = make_double2(a.x - b.x, a.y - b.y);
        }
    }
    __syncthreads();
    {                                              /* pairs: q = p*32+l, so each warp has
        ONE k (uniform shared coeff reads) and stores contiguous 512 B runs */
        const int l = threadIdx.x & 31, p = threadIdx.x >> 5;
        if (l < nl)
            pair_folded<true>(s0 + l, 32, out + base + l, LL2, p, shc, shs);
    }
}

/* Kernel B, coarse: one 23-point line per thread; at each x the warp's addresses are
 * contiguous, so every load/store instruction is fully coalesced.  Loads complete into
 * the fold registers before any store, so in == out (noscratch) is safe. */
__global__ void __launch_bounds__(256, 2)
kernB(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    const int line = blockIdx.x * blockDim.x + threadIdx.x;   /* ky*23 + kz */
    if (line >= LL2) return;
    const size_t base = (size_t)blockIdx.y * LL3 + line;
    dft23<true, 2>(in + base, LL2, out + base, LL2);
}

/* Kernel B, split-k (new in r2, the r1 agenda's #1 item): TWO threads per line.
 * Lane l of a warp handles line (warp*16 + (l&15)); half = l>>4 folds j = 1..6
 * (seeded with x0) or j = 7..11 (seeded with 0) -- exactly the two canonical chains --
 * and the full P/Q are rebuilt with 4 __shfl_xor(16) per k-pair.  Half 0 stores
 * X_1..X_11 and X_0; half 1 stores X_12..X_22.  ~half the fold registers of the
 * coarse engine, so the 512-threads/SM register ceiling (r1's diagnosed wall) lifts.
 *
 * In place (noscratch) this is race-free by dataflow: every lane's loads feed its k=1
 * partials, the first shuffle converges the warp, and all stores come after it -- so
 * both lanes of a pair have finished reading their line before either writes to it.
 * Tail lanes clamp to line 528 and compute without storing (shuffle masks stay full).
 */
template <int TBB, int MINB, bool STREAM_OUT>
__global__ void __launch_bounds__(TBB, MINB)
kernB2(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    const int lane = (int)(threadIdx.x & 31u);
    const int half = lane >> 4;                    /* 0: j=1..6+x0   1: j=7..11 */
    int line = blockIdx.x * (TBB / 2) + (int)(threadIdx.x >> 5) * 16 + (lane & 15);
    const bool live = (line < LL2);
    if (!live) line = LL2 - 1;
    const size_t base = (size_t)blockIdx.y * LL3 + (size_t)line;
    const double2 *pin = in + base;
    double2 *pout = out + base;

    const int nj = half ? 5 : 6;
    const int jb = half * 6;
    const double2 a0 = ld_in<2>(pin);
    double ur[6], ui[6], dr[6], di[6];
#pragma unroll
    for (int jj = 0; jj < 6; ++jj) {
        if (jj < nj) {
            const int j = jb + jj + 1;
            const double2 a = ld_in<2>(pin + (size_t)j * LL2);
            const double2 b = ld_in<2>(pin + (size_t)(23 - j) * LL2);
            ur[jj] = a.x + b.x;  ui[jj] = a.y + b.y;
            dr[jj] = a.x - b.x;  di[jj] = a.y - b.y;
        }
    }
    {                                              /* k = 0 */
        double sr = half ? 0.0 : a0.x, si = half ? 0.0 : a0.y;
#pragma unroll
        for (int jj = 0; jj < 6; ++jj)
            if (jj < nj) { sr += ur[jj]; si += ui[jj]; }
        const double sro = __shfl_xor_sync(0xffffffffu, sr, 16);
        const double sio = __shfl_xor_sync(0xffffffffu, si, 16);
        if (live && half == 0)
            st_out<STREAM_OUT>(pout, make_double2(sr + sro, si + sio));
    }
#pragma unroll
    for (int k = 0; k < 11; ++k) {
        double pr = half ? 0.0 : a0.x, pi = half ? 0.0 : a0.y;
        double qr = 0.0, qi = 0.0;
#pragma unroll
        for (int jj = 0; jj < 6; ++jj) {
            if (jj < nj) {
                const double c = c_cos[k * 11 + jb + jj];
                const double s = c_sin[k * 11 + jb + jj];
                pr = fma(c, ur[jj], pr);  pi = fma(c, ui[jj], pi);
                qr = fma(s, dr[jj], qr);  qi = fma(s, di[jj], qi);
            }
        }
        pr += __shfl_xor_sync(0xffffffffu, pr, 16);
        pi += __shfl_xor_sync(0xffffffffu, pi, 16);
        qr += __shfl_xor_sync(0xffffffffu, qr, 16);
        qi += __shfl_xor_sync(0xffffffffu, qi, 16);
        if (live) {
            if (half == 0)
                st_out<STREAM_OUT>(pout + (size_t)(k + 1) * LL2,
                                   make_double2(pr + qi, pi - qr));
            else
                st_out<STREAM_OUT>(pout + (size_t)(22 - k) * LL2,
                                   make_double2(pr - qi, pi + qr));
        }
    }
}

/* ---- persistent producer/consumer ticket kernel (new in r4) --------------------
 * Ported from L36_globalpass gpu_r2 (design) + L45_pfa gpu_r3 (noinline lesson).
 * K1 ticket = the kernA body (P planes: coalesced load, z, y, plain store to out so
 * it lands in L2); K2 ticket = the kernB body (T x-lines of out in place, __ldlu
 * reads, __stcs stores).  Counters are memset to zero per execute (see header). */

template <int P, int T>
static __device__ __noinline__ void pers_k1(const double2 *__restrict__ in,
                                            double2 *__restrict__ out,
                                            int v, int x0, double2 *sh)
{
    const int np = (LL - x0 < P) ? (LL - x0) : P;
    const size_t base = (size_t)v * LL3 + (size_t)x0 * LL2;
    const int nel = np * LL2;
    for (int i = threadIdx.x; i < nel; i += T)
        sh[i] = __ldcs(in + base + i);
    __syncthreads();
    const int l = threadIdx.x;
    if (l < np * LL) {                             /* z lines: contiguous rows */
        double2 *row = sh + (l / LL) * LL2 + (l % LL) * LL;
        dft23(row, 1, row, 1);
    }
    __syncthreads();
    if (l < np * LL) {                             /* y lines -> global (plain: L2) */
        const int pp = l / LL, z = l % LL;
        double2 *col = sh + pp * LL2 + z;
        dft23(col, LL, out + base + (size_t)pp * LL2 + z, LL);
    }
}

template <int T>
static __device__ __noinline__ void pers_k2(double2 *__restrict__ io, int v, int tile)
{
    __threadfence();                               /* acquire (tid0 saw the done flag,
                                                      the grab barrier ordered us) */
    const int line = tile * T + threadIdx.x;
    if (line < LL2) {
        double2 *p = io + (size_t)v * LL3 + line;
        dft23<true, 2>(p, LL2, p, LL2);
    }
}

template <int nA, int nB>
static __device__ __forceinline__ void decode_ticket(unsigned t, int nvol, int lead,
                                                     int *isA, int *v, int *idx)
{
    constexpr int G = nA + nB;
    const unsigned runway = (unsigned)lead * (unsigned)nA;
    if (t < runway) {                              /* K1 runway: volumes 0..lead-1 */
        *isA = 1;  *v = (int)(t / nA);  *idx = (int)(t % nA);
        return;
    }
    unsigned u = t - runway;
    const unsigned mid = (unsigned)(nvol - lead) * (unsigned)G;
    if (u < mid) {                                 /* K1(v+lead) : K2(v), Bresenham */
        const int i = (int)(u % G);
        const int a0 = (i * nA) / G, a1 = ((i + 1) * nA) / G;
        if (a1 > a0) { *isA = 1; *idx = a0; *v = (int)(u / G) + lead; }
        else         { *isA = 0; *idx = i - a1; *v = (int)(u / G); }
    } else {                                       /* K2-only tail: last lead volumes */
        u -= mid;
        *isA = 0;  *v = (nvol - lead) + (int)(u / nB);  *idx = (int)(u % nB);
    }
}

/* grid must be <= one resident wave (create() sizes it from the occupancy API);
 * every K1 ticket of v is dispatched before v's first K2 ticket -> deadlock-free.
 *
 * ONE barrier serves grab + poll + broadcast: thread 0 grabs the ticket, does the
 * K2 dependency poll itself (the other threads would only have waited at a barrier
 * anyway), and publishes into a parity-indexed slot; the double buffer is what makes
 * a separate loop-top barrier unnecessary (a laggard can still be reading slot i&1
 * only until it arrives at barrier i+1, and slot i&1 is next written after barrier
 * i+1 -- barrier-ordered either way).  Cuts barriers per 6K1+5K2 ticket group from
 * 45 to 29; ncu had CTA-barrier waits at 35.7% of all stall cycles. */
template <int P, int T, int MINB>
__global__ void __launch_bounds__(T, MINB)
kernP(const double2 *__restrict__ in, double2 *__restrict__ out,
      int nvol, int lead, unsigned *ctr)
{
    constexpr int nA = (LL + P - 1) / P;           /* K1 tickets per volume */
    constexpr int nB = (LL2 + T - 1) / T;          /* K2 tickets per volume */
    extern __shared__ double2 sh[];                /* P planes of 529 double2 */
    __shared__ unsigned tsh[2];
    unsigned *next = ctr;
    volatile unsigned *done = ctr + 1;             /* one counter per volume */
    const unsigned ntick = (unsigned)nvol * (unsigned)(nA + nB);
    for (unsigned itn = 0;; ++itn) {
        if (threadIdx.x == 0) {
            const unsigned tt = atomicAdd(next, 1u);
            if (tt < ntick) {
                int isA, v, idx;
                decode_ticket<nA, nB>(tt, nvol, lead, &isA, &v, &idx);
                if (!isA) {                        /* poll + backoff before the sync */
                    int ns = 8;
                    while (done[v] < (unsigned)nA) {
                        __nanosleep(ns);
                        if (ns < 256) ns <<= 1;
                    }
                }
            }
            tsh[itn & 1u] = tt;
        }
        __syncthreads();                           /* the only barrier per K2 ticket */
        const unsigned t = tsh[itn & 1u];
        if (t >= ntick) return;                    /* one failed grab per block */
        int isA, v, idx;
        decode_ticket<nA, nB>(t, nvol, lead, &isA, &v, &idx);
        if (isA) {
            pers_k1<P, T>(in, out, v, idx * P, sh);
            __threadfence();                       /* release before the done tick */
            __syncthreads();
            if (threadIdx.x == 0) atomicAdd(ctr + 1 + v, 1u);
        } else {
            pers_k2<T>(out, v, idx);
        }
    }
}

/* probed shapes: {P planes/K1 ticket, T threads, min blocks/SM}.  Shared/regs per SM:
 * 0: 4x33.9K + 4x128x128r = 135.5K/64K-full  (512 thr/SM, fine tickets)
 * 1: 3 blocks, 170-reg headroom              (384 thr/SM, spill-free safety)
 * 2: 2x67.7K, T=256                          (512 thr/SM, coarse tickets)
 * 3: 8x16.9K, T=64                           (512 thr/SM, finest tickets)
 * 4: 2x67.7K, T=192                          (384 thr/SM, 184/192 K1 lanes live)
 * 5: 3x50.8K, T=144                          (432 thr/SM, 138/144 K1 lanes live) */
struct PCfg { int P, T, minb, shb, nA, nB; };
static const PCfg pcfgs[] = {
    { 4, 128, 4, 4 * LL2 * (int)sizeof(double2), 6, 5 },
    { 4, 128, 3, 4 * LL2 * (int)sizeof(double2), 6, 5 },
    { 8, 256, 2, 8 * LL2 * (int)sizeof(double2), 3, 3 },
    { 2,  64, 8, 2 * LL2 * (int)sizeof(double2), 12, 9 },
    { 8, 192, 2, 8 * LL2 * (int)sizeof(double2), 3, 3 },
    { 6, 144, 3, 6 * LL2 * (int)sizeof(double2), 4, 4 },
};
#define NPCFG 6

static void launchP(int pc, int grid, cudaStream_t s, const double2 *in,
                    double2 *out, int nvol, int lead, unsigned *ctr)
{
    const int shb = pcfgs[pc].shb;
    switch (pc) {
    case 0: kernP<4, 128, 4><<<grid, 128, shb, s>>>(in, out, nvol, lead, ctr); break;
    case 1: kernP<4, 128, 3><<<grid, 128, shb, s>>>(in, out, nvol, lead, ctr); break;
    case 2: kernP<8, 256, 2><<<grid, 256, shb, s>>>(in, out, nvol, lead, ctr); break;
    case 3: kernP<2,  64, 8><<<grid,  64, shb, s>>>(in, out, nvol, lead, ctr); break;
    case 4: kernP<8, 192, 2><<<grid, 192, shb, s>>>(in, out, nvol, lead, ctr); break;
    case 5: kernP<6, 144, 3><<<grid, 144, shb, s>>>(in, out, nvol, lead, ctr); break;
    }
}

/* one resident wave for config pc, or 0 if it cannot launch */
static int pers_grid(int pc)
{
    static int sms = 0;
    if (!sms) {
        int dev = 0;  cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) sms = 108;
    }
    const int shb = pcfgs[pc].shb;
    int occ = 0;
    cudaError_t e = cudaErrorInvalidValue;
    switch (pc) {
    case 0:
        cudaFuncSetAttribute(kernP<4, 128, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, shb);
        e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernP<4, 128, 4>, 128, shb);
        break;
    case 1:
        cudaFuncSetAttribute(kernP<4, 128, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, shb);
        e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernP<4, 128, 3>, 128, shb);
        break;
    case 2:
        cudaFuncSetAttribute(kernP<8, 256, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shb);
        e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernP<8, 256, 2>, 256, shb);
        break;
    case 3:
        cudaFuncSetAttribute(kernP<2, 64, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize, shb);
        e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernP<2, 64, 8>, 64, shb);
        break;
    case 4:
        cudaFuncSetAttribute(kernP<8, 192, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, shb);
        e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernP<8, 192, 2>, 192, shb);
        break;
    case 5:
        cudaFuncSetAttribute(kernP<6, 144, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, shb);
        e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernP<6, 144, 3>, 144, shb);
        break;
    }
    if (e != cudaSuccess || occ < 1) { cudaGetLastError(); return 0; }
    return occ * sms;
}

/* ------------------------------------------------------------------------- */

#define NSTR 8             /* stream pool; chunks in flight = plan nstr <= NSTR */

enum { BK_COARSE = 0, BK_SPLIT = 1, BK_SPLITSQ = 2, BK_FINE = 3, BK_MED = 4 };

struct fft3d_gpu_plan {
    int          batch;
    int          aidx;     /* index into acfgs[] for coarse kernel A     */
    int          afine;    /* kernel A engine: 0 coarse, 1 fine (kernAf), 2 medium (kernAm) */
    int          uload;    /* coarse kernel A unrolled load              */
    int          bkind;    /* BK_*                                       */
    int          TB;       /* kernel B block size                        */
    int          chunk;    /* volumes per A/B pair (L2 blocking)         */
    int          noscratch;/* out-as-intermediate, B in place            */
    int          nstr;     /* streams actually cycled (<= NSTR)          */
    double2     *tmp[NSTR];/* rotating intermediates (scratch mode)      */
    cudaStream_t st[NSTR]; /* chunk c runs on st[c%nstr]                 */
    int          use_graph;/* replay a captured graph (single chunk)     */
    int          gvalid;
    cudaGraphExec_t gx;
    const double2 *gin;
    double2       *gout;
    int          pers;     /* persistent ticket kernel path              */
    int          pcfg;     /* index into pcfgs[]                         */
    int          lead;     /* K1 runway volumes (clamped to batch)       */
    int          pgrid;    /* one resident wave for pcfg                 */
    unsigned    *d_ctr;    /* [0] ticket counter, [1..batch] done counts */
};

extern "C" const char *fft3d_gpu_name(void) { return "L23_rader"; }

static char g_desc[224] =
    "L23_rader: folded-dense 23pt (594 op/line), 2-pass z+y|x, L2-chunked";
extern "C" const char *fft3d_gpu_description(void) { return g_desc; }

extern "C" int fft3d_gpu_supports(int L) { return L == 23; }

/* Pin each stream's tmp buffer in L2 (scratch mode; r1 measured ~zero effect but it is
 * free).  Off (num_bytes = 0) for noscratch cells so a stale window never lingers. */
static void set_l2_window(fft3d_gpu_plan *p, int on)
{
    static int maxwin = -2, maxpers = -2;
    if (maxwin == -2) {
        int devid = 0;  cudaGetDevice(&devid);
        cudaDeviceGetAttribute(&maxwin, cudaDevAttrMaxAccessPolicyWindowSize, devid);
        cudaDeviceGetAttribute(&maxpers, cudaDevAttrMaxPersistingL2CacheSize, devid);
        if (maxwin > 0 && maxpers > 0)
            cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, (size_t)maxpers);
    }
    if (maxwin <= 0 || maxpers <= 0) return;
    if (getenv("L23R_NOL2WIN")) on = 0;
    size_t wb = 0;
    float hr = 0.0f;
    if (on) {
        wb = (size_t)p->chunk * LL3 * sizeof(double2);
        if (wb > (size_t)maxwin) wb = (size_t)maxwin;
        hr = (float)((double)maxpers / (double)(NSTR * wb));
        if (hr > 1.0f) hr = 1.0f;
    }
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

/* coarse kernel-A shapes: P planes staged, T threads. */
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
    if (p->pers) {
        /* counter reset by stream-ordered memset instead of epoch bases: keeps the
           launch args constant (graph-capturable) and immune to plan-copy trials */
        const unsigned ntick = (unsigned)p->batch
                             * (unsigned)(pcfgs[p->pcfg].nA + pcfgs[p->pcfg].nB);
        int grid = p->pgrid;
        if ((unsigned)grid > ntick) grid = (int)ntick;
        cudaMemsetAsync(p->d_ctr, 0, (size_t)(p->batch + 1) * sizeof(unsigned),
                        p->st[0]);
        launchP(p->pcfg, grid, p->st[0], in, out, p->batch, p->lead, p->d_ctr);
        return;
    }
    const int PA  = acfgs[p->aidx].P;
    const int nPG = (LL + PA - 1) / PA;
    const int nBB = (LL2 + p->TB - 1) / p->TB;
    const int nB2 = (LL2 + p->TB / 2 - 1) / (p->TB / 2);
    const int nBF = (12 * LL2 + 255) / 256;
    int ci = 0;
    for (int c = 0; c < p->batch; c += p->chunk, ++ci) {
        const int nb = (p->batch - c < p->chunk) ? (p->batch - c) : p->chunk;
        const int w  = ci % p->nstr;
        const double2 *cin = in + (size_t)c * LL3;
        double2 *cout = out + (size_t)c * LL3;
        double2 *mid  = p->noscratch ? cout : p->tmp[w];
        cudaStream_t s = p->st[w];

        if (p->afine == 2)      kernAm<<<dim3(LL, nb), TAM, 0, s>>>(cin, mid);
        else if (p->afine == 1) kernAf<<<dim3(LL, nb), 288, 0, s>>>(cin, mid);
        else                    launchA(p->aidx, p->uload, dim3(nPG, nb), s, cin, mid);

        switch (p->bkind) {
        case BK_COARSE:
            kernB<<<dim3(nBB, nb), p->TB, 0, s>>>(mid, cout);
            break;
        case BK_SPLIT:
            if (p->TB == 128)
                kernB2<128, 1, true><<<dim3(nB2, nb), 128, 0, s>>>(mid, cout);
            else
                kernB2<256, 1, true><<<dim3(nB2, nb), 256, 0, s>>>(mid, cout);
            break;
        case BK_SPLITSQ:
            kernB2<256, 4, true><<<dim3(nB2, nb), 256, 0, s>>>(mid, cout);
            break;
        case BK_FINE:
            kernBf<<<dim3(nBF, nb), 256, 0, s>>>(mid, cout);
            break;
        case BK_MED:
            kernBm<<<dim3((LL2 + 31) / 32, nb), TBM, 0, s>>>(mid, cout);
            break;
        }
    }
}

/* Capture one whole execute (must be a single chunk => single stream) as a graph. */
static int capture_graph(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (!p->pers && (p->batch + p->chunk - 1) / p->chunk != 1) return 0;
    if (cudaStreamBeginCapture(p->st[0], cudaStreamCaptureModeThreadLocal)
        != cudaSuccess) { cudaGetLastError(); return 0; }
    run_case(p, in, out);
    cudaGraph_t g = NULL;
    if (cudaStreamEndCapture(p->st[0], &g) != cudaSuccess || !g) {
        cudaGetLastError(); return 0;
    }
    cudaGraphExec_t gx = NULL;
    const cudaError_t e = cudaGraphInstantiateWithFlags(&gx, g, 0);
    cudaGraphDestroy(g);
    if (e != cudaSuccess || !gx) { cudaGetLastError(); return 0; }
    if (p->gvalid) cudaGraphExecDestroy(p->gx);
    p->gx = gx;  p->gvalid = 1;  p->gin = in;  p->gout = out;
    return 1;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (p->use_graph) {
        if (!p->gvalid || in != p->gin || out != p->gout) {
            if (!capture_graph(p, in, out)) {      /* capture failed: plain path */
                run_case(p, in, out);
                return;
            }
        }
        cudaGraphLaunch(p->gx, p->st[0]);
        return;
    }
    run_case(p, in, out);
}

/* B engine candidates.  BK_SPLITSQ is not raced: the forced 64-reg squeeze spills and
 * lost 15% at B_HBM (2663 vs 2312 us), exactly as r1's 96-reg experiment predicted;
 * the kernel stays for L23R_FORCE experiments. */
struct BCfg { int bkind, TB; };
static const BCfg bcfgs[] = { {BK_COARSE, 32}, {BK_COARSE, 64}, {BK_COARSE, 256},
                              {BK_SPLIT, 128}, {BK_SPLIT, 256},
                              {BK_FINE, 256}, {BK_MED, TBM} };
#define NBCFG 7

static float time_case(fft3d_gpu_plan *t, const double2 *din, double2 *dout,
                       cudaEvent_t e0, cudaEvent_t e1, int reps, int nin)
{
    /* nin executes per sample: the finals run with nin sized so a sample exceeds the
       ~20 ms boost-clock cliff (PANEL_BRIEF; tuner-sample form of the bug recorded by
       L36_globalpass r2 -- their 6 ms samples once ranked a 10%-slower config first) */
    run_case(t, din, dout);                       /* warm-up */
    cudaDeviceSynchronize();
    float best = 1e30f;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(e0);
        for (int i = 0; i < nin; ++i)
            run_case(t, din, dout);
        /* work is on t->st[*], which does not sync with the default stream:
           drain the whole device before the stop event */
        cudaDeviceSynchronize();
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms = 1e30f;
        cudaEventElapsedTime(&ms, e0, e1);
        ms /= (float)nin;
        if (ms < best) best = ms;
    }
    return best;
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
        if (cudaMemcpyToSymbol(g_cosd, cc, sizeof(cc)) != cudaSuccess) return NULL;
        if (cudaMemcpyToSymbol(g_sind, ss, sizeof(ss)) != cudaSuccess) return NULL;
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

    /* persistent-path ticket + done counters (always allocated: 4(B+1) bytes) */
    if (cudaMalloc((void **)&p->d_ctr, (size_t)(batch + 1) * sizeof(unsigned))
        != cudaSuccess) { fft3d_gpu_destroy(p); return NULL; }
    cudaMemset(p->d_ctr, 0, (size_t)(batch + 1) * sizeof(unsigned));
    int pgrids[NPCFG];
    for (int pc = 0; pc < NPCFG; ++pc) pgrids[pc] = pers_grid(pc);

    /* ---- plan-time autotune over (A engine, B engine, chunk, noscratch) ---- */
    double2 *din = NULL, *dout = NULL;
    size_t vb = (size_t)batch * LL3 * sizeof(double2);
    if (cudaMalloc((void **)&din, vb) != cudaSuccess ||
        cudaMalloc((void **)&dout, vb) != cudaSuccess) {
        if (din) cudaFree(din);
        /* fall back to a safe static pick */
        p->aidx = (batch == 1) ? 1 : 4;  p->TB = (batch == 1) ? 64 : 256;
        p->bkind = BK_COARSE;
        p->chunk = (batch < tmpcap) ? batch : tmpcap;
        return p;
    }
    cudaMemset(din, 0, vb);

    /* L23R_FORCE="aidx,TB,chunk[,uload[,bkind[,ns[,afine[,graph[,pers[,pcfg[,lead]]]]]]]]"
       pins a cell (bkind: 0 coarse, 1 split, 2 splitsq, 3 fine);
       L23R_VERBOSE=1 prints the table. */
    if (const char *f = getenv("L23R_FORCE")) {
        int fa, ft, fc, fu = 0, fb = 0, fns = 0, faf = 0, fg = 0;
        int fp = 0, fpc = 0, fl = 16;
        if (sscanf(f, "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
                   &fa, &ft, &fc, &fu, &fb, &fns, &faf, &fg, &fp, &fpc, &fl) >= 3) {
            cudaFree(din);  cudaFree(dout);
            p->aidx = (fa < 0) ? 0 : ((fa >= NACFG) ? NACFG - 1 : fa);
            p->TB = ft;
            p->chunk = (fc < 1) ? 1 : ((fc > batch) ? batch : fc);
            if (!fns && p->chunk > tmpcap) p->chunk = tmpcap;
            p->uload = fu ? 1 : 0;
            p->bkind = fb;
            p->noscratch = (fns && fb != BK_FINE) ? 1 : 0;
            p->afine = (faf < 0) ? 0 : ((faf > 2) ? 2 : faf);
            p->use_graph = fg ? 1 : 0;
            p->pcfg = (fpc < 0) ? 0 : ((fpc >= NPCFG) ? NPCFG - 1 : fpc);
            p->pgrid = pgrids[p->pcfg];
            p->pers = (fp && p->pgrid > 0) ? 1 : 0;
            p->lead = (fl < 1) ? 1 : ((fl > batch) ? batch : fl);
            set_l2_window(p, !p->noscratch && !p->pers);
            snprintf(g_desc, sizeof(g_desc),
                     "L23_rader: FORCED A(P=%d,T=%d) af=%d TB=%d bk=%d chunk=%d ns=%d "
                     "g=%d pers=%d pc=%d lead=%d",
                     acfgs[p->aidx].P, acfgs[p->aidx].T, p->afine, p->TB, p->bkind,
                     p->chunk, p->noscratch, p->use_graph, p->pers, p->pcfg, p->lead);
            return p;
        }
    }
    const int verbose = getenv("L23R_VERBOSE") != NULL;
    const int reps = (batch <= 32) ? 5 : 3;

    const int Cs[] = { 0 /* = full */, 24, 32, 48 };

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);  cudaEventCreate(&ev1);

    float bestms = 1e30f;
    fft3d_gpu_plan best = *p;
    best.aidx = 4;  best.TB = 256;  best.bkind = BK_COARSE;
    best.chunk = (batch < tmpcap) ? batch : tmpcap;
    best.nstr = 4;
    fft3d_gpu_plan trial = *p;
    trial.nstr = 4;                      /* stage 1 at 4 streams; stage 2 refines nstr */

    for (int iaf = 0; iaf <= 2; ++iaf)
    for (int ia = 0; ia < (iaf ? 1 : NACFG); ++ia)
    for (int iu = 0; iu < (!iaf ? 2 : 1); ++iu)
    for (int ib = 0; ib < NBCFG; ++ib)
    for (int ins = 0; ins <= ((bcfgs[ib].bkind == BK_FINE) ? 0 : 1); ++ins)
        for (int ic = 0; ic < 4; ++ic) {
            int C = Cs[ic];
            const int cap = ins ? batch : tmpcap;
            if (C == 0) C = cap;
            if (C > batch) C = batch;
            if (C > cap) C = cap;
            if (ic > 0 && C >= ((batch < cap) ? batch : cap)) continue; /* dup of full */
            trial.afine = iaf;  trial.aidx = ia;  trial.uload = iu;
            trial.bkind = bcfgs[ib].bkind;  trial.TB = bcfgs[ib].TB;
            trial.chunk = C;  trial.noscratch = ins;
            set_l2_window(&trial, !ins);
            cudaCtxResetPersistingL2Cache();

            const float ms = time_case(&trial, din, dout, ev0, ev1, reps, 1);
            if (verbose)
                fprintf(stderr,
                        "L23R tuner: A%s(P=%d,T=%-3d) u%d B%d TB=%-3d ns%d chunk=%-4d %9.1f us\n",
                        iaf == 2 ? "m" : (iaf ? "f" : " "), acfgs[ia].P, acfgs[ia].T, iu,
                        bcfgs[ib].bkind, bcfgs[ib].TB, ins, C, 1e3 * ms);
            if (ms < bestms) { bestms = ms; best = trial; }
        }

    /* stage 2: refine (chunk, nstr) on the winning shape.  The in-flight footprint is
       nstr * chunk * 380 KiB against 40 MiB of L2, and the chunk curve turned out to be
       sharp once the kernels got fast (B_HBM: 2312 us at chunk 32 vs 3016 at 64), so
       the fine grid is worth its cells. */
    if (batch > 32) {
        const int C2s[] = { 12, 16, 20, 24, 28, 32, 40, 48, 64 };
        const int c1 = best.chunk, n1 = best.nstr;
        for (int in2 = 3; in2 <= NSTR; ++in2)
            for (int ic = 0; ic < 9; ++ic) {
                trial = best;
                trial.nstr = in2;
                trial.chunk = C2s[ic];
                if (trial.chunk > batch) continue;
                if (!trial.noscratch && trial.chunk > tmpcap) continue;
                if (in2 == n1 && trial.chunk == c1) continue;   /* already timed */
                set_l2_window(&trial, !trial.noscratch);
                cudaCtxResetPersistingL2Cache();
                const float ms = time_case(&trial, din, dout, ev0, ev1, reps, 1);
                if (verbose)
                    fprintf(stderr,
                            "L23R tuner: refine nstr=%d chunk=%-4d           %9.1f us\n",
                            in2, trial.chunk, 1e3 * ms);
                if (ms < bestms) { bestms = ms; best = trial; }
            }
    }

    /* ---- stage 3: persistent ticket kernel vs the classic winner ----
       Finals run at >= 20 ms per sample (see time_case); the classic winner is
       re-timed the same way first so the playoff is apples-to-apples. */
    if (!getenv("L23R_NOPERS")) {
        int nin = (int)(22.0f / (bestms > 1e-4f ? bestms : 1e-4f)) + 1;
        if (nin > 4096) nin = 4096;
        trial = best;
        set_l2_window(&trial, !trial.noscratch);
        cudaCtxResetPersistingL2Cache();
        bestms = time_case(&trial, din, dout, ev0, ev1, reps, nin);
        if (verbose)
            fprintf(stderr, "L23R tuner: classic winner re-timed (nin=%d) %9.1f us\n",
                    nin, 1e3 * bestms);
        /* lead sweep: B=5515 measured a plateau 56-88 (~2015 us) with cliffs on both
           sides (48 -> 2023, 96 -> 2088, 128 -> 2586, 176+ -> 3280+: the runway's
           intermediate falls out of L2).  ~72 = one 432-block wave / 6 K1 tickets,
           the same one-wave rule L36_globalpass found for their lead. */
        const int leads[] = { 4, 8, 12, 16, 24, 32, 48, 64, 72, 80, 96 };
        for (int pc = 0; pc < NPCFG; ++pc) {
            if (pgrids[pc] <= 0) continue;
            int prev = -1;
            for (int il = 0; il < 11; ++il) {
                int ld = (leads[il] < batch) ? leads[il] : batch;
                if (ld == prev) continue;
                prev = ld;
                trial = best;
                trial.pers = 1;  trial.pcfg = pc;  trial.lead = ld;
                trial.pgrid = pgrids[pc];
                set_l2_window(&trial, 0);
                cudaCtxResetPersistingL2Cache();
                const float ms = time_case(&trial, din, dout, ev0, ev1, reps, nin);
                if (verbose)
                    fprintf(stderr,
                            "L23R tuner: pers pc=%d(P=%d,T=%-3d) lead=%-3d    %9.1f us\n",
                            pc, pcfgs[pc].P, pcfgs[pc].T, ld, 1e3 * ms);
                if (ms < bestms) { bestms = ms; best = trial; }
            }
        }
    }

    /* graph replay of the winning config (single-chunk cases only; L36_sharedtiled
       r1 measured it beating both plain launches and a cooperative fused kernel) */
    int bestG = 0;
    if (batch <= 32 && (best.pers || (batch + best.chunk - 1) / best.chunk == 1)) {
        trial = best;
        set_l2_window(&trial, !trial.noscratch && !trial.pers);
        if (capture_graph(&trial, din, dout)) {
            int ning = (int)(22.0f / (bestms > 1e-4f ? bestms : 1e-4f)) + 1;
            if (ning > 4096) ning = 4096;
            cudaGraphLaunch(trial.gx, trial.st[0]);   /* warm-up */
            cudaDeviceSynchronize();
            float gbest = 1e30f;
            for (int r = 0; r < reps; ++r) {
                cudaEventRecord(ev0);
                for (int i = 0; i < ning; ++i)
                    cudaGraphLaunch(trial.gx, trial.st[0]);
                cudaDeviceSynchronize();
                cudaEventRecord(ev1);
                cudaEventSynchronize(ev1);
                float ms = 1e30f;
                cudaEventElapsedTime(&ms, ev0, ev1);
                ms /= (float)ning;
                if (ms < gbest) gbest = ms;
            }
            if (verbose)
                fprintf(stderr, "L23R tuner: GRAPH of best             %9.1f us\n",
                        1e3 * gbest);
            cudaGraphExecDestroy(trial.gx);
            if (gbest < bestms) { bestms = gbest; bestG = 1; }
        }
    }

    cudaEventDestroy(ev0);  cudaEventDestroy(ev1);
    cudaFree(din);  cudaFree(dout);
    if (cudaGetLastError() != cudaSuccess) { fft3d_gpu_destroy(p); return NULL; }

    p->aidx = best.aidx;  p->afine = best.afine;  p->uload = best.uload;
    p->bkind = best.bkind;  p->TB = best.TB;  p->chunk = best.chunk;
    p->noscratch = best.noscratch;  p->nstr = best.nstr;  p->use_graph = bestG;
    p->pers = best.pers;  p->pcfg = best.pcfg;  p->lead = best.lead;
    p->pgrid = best.pgrid;
    set_l2_window(p, !p->noscratch && !p->pers);
    cudaCtxResetPersistingL2Cache();
    if (p->pers)
        snprintf(g_desc, sizeof(g_desc),
                 "L23_rader: folded-dense 23pt, persistent ticket z+y|x "
                 "(P=%d,T=%d,wave=%d) lead=%d g%d (%.2f us/xform in-plan, nv=%d)",
                 pcfgs[p->pcfg].P, pcfgs[p->pcfg].T, p->pgrid, p->lead,
                 p->use_graph, 1e3 * bestms / batch, batch);
    else
        snprintf(g_desc, sizeof(g_desc),
                 "L23_rader: folded-dense 23pt, 2-pass z+y|x; tuner A%s(P=%d,T=%d) u%d "
                 "B%d TB=%d ns%d chunk=%d/%d g%d (%.2f us/xform in-plan, nv=%d)",
                 p->afine == 2 ? "m" : (p->afine ? "f" : ""),
                 acfgs[p->aidx].P, acfgs[p->aidx].T, p->uload,
                 p->bkind, p->TB, p->noscratch, p->chunk, p->nstr, p->use_graph,
                 1e3 * bestms / batch, batch);
    return p;
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p)
{
    if (!p) return;
    if (p->gvalid) cudaGraphExecDestroy(p->gx);
    if (p->d_ctr) cudaFree(p->d_ctr);
    for (int w = 0; w < NSTR; ++w) {
        if (p->tmp[w]) cudaFree(p->tmp[w]);
        if (p->st[w])  cudaStreamDestroy(p->st[w]);
    }
    free(p);
}
