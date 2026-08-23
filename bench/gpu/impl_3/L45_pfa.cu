/* L45_pfa — L = 45 on one A100.
 *
 * Two-pass Good–Thomas (prime-factor) 9x5, plane-per-block:
 *
 *   kernel 1: one (y,z) plane per block (grid B*45). The plane is 2025 complex =
 *             31.6 KiB, contiguous in global -> staged into shared coalesced.
 *             z-pass then y-pass, each 45 lines, UNIT-PARALLEL (one thread = one
 *             DFT5 or DFT9 of one line; thread-per-line compiled to >204 regs
 *             and 10% occupancy), PFA 9x5 done IN PLACE in shared using the
 *             classic PFA scrambled-slot property (see below). One global read
 *             + one global write for two axes.
 *   kernel 2: x-pass. Tile = 45(x) x 32(flat yz) per block (grid B*64, shared
 *             pitch 33), same unit-parallel in-place PFA, written back in
 *             place. out serves as the intermediate: no scratch allocation,
 *             and at the L2-resident batch (B=11) the inter-kernel round trip
 *             never leaves L2.
 *
 * PFA in place, no temp: 45 = 9*5 coprime. Ruritanian input map n=(5a+9b)%45,
 * CRT output map k=(10k1+36k2)%45 kill all inter-stage twiddles. Each DFT5 over
 * b (fixed a=g) reads slots {(5g+9j)%45} and writes its outputs back to those
 * same slots; each DFT9 over a (fixed k2=c) then reads {(5g+9c)%45} and writes
 * back to the same set. Result: X[k] lands at slot sig(k) = (5*(k%9)+9*(k%5))%45,
 * a fixed permutation folded into the (coalesced) global store's shared-side
 * gather. Stride 45 is odd, so every shared access pattern here is bank-conflict
 * free with no padding (the odd-L gift; lit 09 §6.2).
 *
 * Chunked execution (lit 09 §2.4, structure 3 for L=45): at the HBM batch the
 * intermediate between the two kernels is the whole 1 GiB, but processed in
 * chunks of CH volumes the K1->K2 round trip stays in the 40 MiB L2 and the
 * HBM traffic drops from 64 to ~32 B/point. FFT45_CHUNK overrides for A/B.
 *
 * Codelets: folded real-coefficient DFT5/DFT9 (u/v fold, P +- iS), the same
 * family every CPU-phase winner used. Arithmetic is ~96 flop/point for the full
 * 3D transform against a 2-pass bandwidth budget of ~400 — memory is the game.
 */
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../fft3d_gpu_api.h"

#define NL     45
#define NPLANE 2025      /* 45*45 */
#define NVOL   91125     /* 45^3  */

#ifndef T1
#define T1 128           /* threads per block, kernel 1 */
#endif
#ifndef T2
#define T2 128           /* threads per block, kernel 2 */
#endif
#ifndef K1MB
#define K1MB 5           /* min blocks/SM hint, kernel 1 */
#endif
#ifndef K2MB
#define K2MB 7           /* min blocks/SM hint, kernel 2 */
#endif

/* Streaming cache hints on the chunked-HBM path (borrowed from L36_globalpass
 * and L64_radix8, where the evict-first pairing is what makes L2 chunking pay):
 * __ldcs on the input (read once), __ldlu on K2's intermediate read (the line is
 * dead after the load — it gets rewritten), __stcs on the final store (never
 * re-read). -DNOHINTS reverts to plain accesses for A/B. */
#ifdef NOHINTS
#define LDCS(p)    __ldg(p)
#define LDLU(p)    (*(p))
#define STCS(p, v) (*(p) = (v))
#else
#define LDCS(p)    __ldcs(p)
#define LDLU(p)    __ldlu(p)
#define STCS(p, v) __stcs(p, v)
#endif

/* ---- constant tables (filled from host in create) ---- */
struct Coef {
    double c9[4][4], s9[4][4];   /* cos/sin(2*pi*k*j/9), k,j = 1..4 -> [k-1][j-1] */
    double c5[2][2], s5[2][2];   /* cos/sin(2*pi*k*j/5), k,j = 1..2 */
};
static __constant__ Coef CF;

/* PFA output scramble: X[k] sits at slot sig(k). Computed inline — a constant-
 * memory table here is read at divergent addresses inside the store loops and
 * the constant cache serializes those. */
static __device__ __forceinline__ int sig45(int k)
{
    int s = 5 * (k % 9) + 9 * (k % 5);   /* <= 76 */
    return (s >= 45) ? s - 45 : s;
}

/* ---- complex helpers ---- */
static __device__ __forceinline__ double2 cadd(double2 a, double2 b)
{ return make_double2(a.x + b.x, a.y + b.y); }
static __device__ __forceinline__ double2 csub(double2 a, double2 b)
{ return make_double2(a.x - b.x, a.y - b.y); }
/* c*a + b, componentwise real scale */
static __device__ __forceinline__ double2 cfma(double c, double2 a, double2 b)
{ return make_double2(fma(c, a.x, b.x), fma(c, a.y, b.y)); }

/* Forward DFT5 in registers: X_k = P_k - i*S_k, X_{5-k} = P_k + i*S_k */
static __device__ __forceinline__ void dft5(double2 a[5])
{
    double2 u1 = cadd(a[1], a[4]), v1 = csub(a[1], a[4]);
    double2 u2 = cadd(a[2], a[3]), v2 = csub(a[2], a[3]);
    double2 x0 = a[0];
    a[0] = cadd(x0, cadd(u1, u2));
#pragma unroll
    for (int k = 1; k <= 2; ++k) {
        double2 P = cfma(CF.c5[k-1][1], u2, cfma(CF.c5[k-1][0], u1, x0));
        double2 S = cfma(CF.s5[k-1][1], v2,
                         make_double2(CF.s5[k-1][0] * v1.x, CF.s5[k-1][0] * v1.y));
        a[k]     = make_double2(P.x + S.y, P.y - S.x);
        a[5 - k] = make_double2(P.x - S.y, P.y + S.x);
    }
}

/* Forward DFT9, same folded form */
static __device__ __forceinline__ void dft9(double2 a[9])
{
    double2 u[4], v[4];
#pragma unroll
    for (int j = 1; j <= 4; ++j) {
        u[j-1] = cadd(a[j], a[9 - j]);
        v[j-1] = csub(a[j], a[9 - j]);
    }
    double2 x0 = a[0];
    a[0] = cadd(x0, cadd(cadd(u[0], u[1]), cadd(u[2], u[3])));
#pragma unroll
    for (int k = 1; k <= 4; ++k) {
        double2 P = x0;
        double2 S = make_double2(0.0, 0.0);
#pragma unroll
        for (int j = 1; j <= 4; ++j) {
            P = cfma(CF.c9[k-1][j-1], u[j-1], P);
            S = cfma(CF.s9[k-1][j-1], v[j-1], S);
        }
        a[k]     = make_double2(P.x + S.y, P.y - S.x);
        a[9 - k] = make_double2(P.x - S.y, P.y + S.x);
    }
}

/* Unit-parallel PFA stages: one thread = one DFT5 (or DFT9) of one line, so the
 * per-thread register footprint stays small (the thread-per-line form compiled
 * to >204 regs and capped the SM at 4 blocks / 10% occupancy — measured, ncu).
 * A plane's z- or y-pass = 45 lines x 9 DFT5 units, then 45 x 5 DFT9 units.
 * The (5g+9j)%45 slots: 5g+9j <= 76, so the mod is one conditional subtract. */
static __device__ __forceinline__ void dft5_unit(double2 *p, int S, int g)
{
    double2 a[5];
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        int idx = 5 * g + 9 * j; if (idx >= 45) idx -= 45;
        a[j] = p[idx * S];
    }
    dft5(a);
#pragma unroll
    for (int j = 0; j < 5; ++j) {
        int idx = 5 * g + 9 * j; if (idx >= 45) idx -= 45;
        p[idx * S] = a[j];
    }
}

static __device__ __forceinline__ void dft9_unit(double2 *p, int S, int c)
{
    double2 a[9];
#pragma unroll
    for (int g = 0; g < 9; ++g) {
        int idx = 5 * g + 9 * c; if (idx >= 45) idx -= 45;
        a[g] = p[idx * S];
    }
    dft9(a);
#pragma unroll
    for (int k = 0; k < 9; ++k) {
        int idx = 5 * k + 9 * c; if (idx >= 45) idx -= 45;
        p[idx * S] = a[k];
    }
}

/* One full 45-point pass over 45 parallel lines living in shared.
 * LS = stride between slots of a line, RS = stride between lines.
 * (z-pass: LS=1, RS=45.  y-pass / x-pass: LS=45, RS=1.)
 * (Two variants measured slower at B=736 and dropped: warp-owns-lines with
 * __syncwarp stage boundaries, -14%; paired two-line DFT5 units for ILP, -1%.) */
template<int T, int LS, int RS>
static __device__ __forceinline__ void pass45(double2 *sh)
{
#pragma unroll
    for (int k = 0; k < (45 * 9 + T - 1) / T; ++k) {  /* stage 1: 405 DFT5s */
        int i = threadIdx.x + k * T;
        if ((45 * 9) % T == 0 || i < 45 * 9) {
            int line = i % 45, g = i / 45;
            dft5_unit(sh + line * RS, LS, g);
        }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < (45 * 5 + T - 1) / T; ++k) {  /* stage 2: 225 DFT9s */
        int i = threadIdx.x + k * T;
        if ((45 * 5) % T == 0 || i < 45 * 5) {
            int line = i % 45, c = i / 45;
            dft9_unit(sh + line * RS, LS, c);
        }
    }
}

/* shared tile geometry of kernel 2 (needed by the extracted bodies below) */
#define K2W   32                 /* tile width (flat yz columns) */
#define K2P   33                 /* shared pitch, odd */
#define K2NT  64                 /* tiles per volume: ceil(2025/32) */
#define K2ELE (NL * K2W)         /* full-tile element count: 1440 */

/* The STAGED (chunked-HBM regime) bodies of both kernels, extracted so the
 * persistent producer/consumer kernel (gpu_r3) can run either unit of work
 * from one launch. Hints as on the chunked path: __ldcs input read, PLAIN
 * intermediate write (must land in L2 for the consumer), __ldlu intermediate
 * read (dead after the load), __stcs final store. */
static __device__ __forceinline__ void
k1_plane_staged(const double2 *__restrict__ src, double2 *__restrict__ dst,
                double2 *sh)
{
    /* stage in: compile-time trip count so all loads issue back to back.
       (cp.async here measured 10% slower at B=736 — nothing overlaps it.) */
    {
        double2 r[(NPLANE + T1 - 1) / T1];
#pragma unroll
        for (int k = 0; k < NPLANE / T1; ++k)
            r[k] = LDCS(&src[threadIdx.x + k * T1]);
        int e = threadIdx.x + (NPLANE / T1) * T1;
        if (e < NPLANE) r[NPLANE / T1] = LDCS(&src[e]);
#pragma unroll
        for (int k = 0; k < NPLANE / T1; ++k) sh[threadIdx.x + k * T1] = r[k];
        if (e < NPLANE) sh[e] = r[NPLANE / T1];
    }
    __syncthreads();
    pass45<T1, 1, NL>(sh);                 /* z: lines are rows      */
    __syncthreads();                       /* z->y is a transpose    */
    pass45<T1, NL, 1>(sh);                 /* y: lines are columns   */
    __syncthreads();
#pragma unroll
    for (int k = 0; k < (NPLANE + T1 - 1) / T1; ++k) {  /* unscramble both */
        int e = threadIdx.x + k * T1;
        if (e < NPLANE) {
            int ky = e / NL, kz = e - ky * NL;
            dst[e] = sh[sig45(ky) * NL + sig45(kz)];
        }
    }
}

static __device__ __forceinline__ void
k2_tile_staged(double2 *__restrict__ vol, int tf, double2 *sh)
{
    const int f0 = tf * K2W;
    const int w  = (NPLANE - f0 < K2W) ? NPLANE - f0 : K2W;   /* 32, or 9 at the tail */
    double2 *base = vol + f0;

    if (w == K2W) {
#pragma unroll
        for (int k = 0; k < (K2ELE + T2 - 1) / T2; ++k) {
            int e = threadIdx.x + k * T2;
            if (K2ELE % T2 == 0 || e < K2ELE) {
                int x = e / K2W, f = e - x * K2W;
                sh[x * K2P + f] = LDLU(&base[(long)x * NPLANE + f]);
            }
        }
    } else {
        for (int e = threadIdx.x; e < NL * w; e += T2) {
            int x = e / w, f = e - x * w;
            sh[x * K2P + f] = LDLU(&base[(long)x * NPLANE + f]);
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < w * 9; i += T2) {
        int f = i % w, g = i / w;
        dft5_unit(sh + f, K2P, g);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < w * 5; i += T2) {
        int f = i % w, c = i / w;
        dft9_unit(sh + f, K2P, c);
    }
    __syncthreads();
    if (w == K2W) {
#pragma unroll
        for (int k = 0; k < (K2ELE + T2 - 1) / T2; ++k) {
            int e = threadIdx.x + k * T2;
            if (K2ELE % T2 == 0 || e < K2ELE) {
                int k0 = e / K2W, f = e - k0 * K2W;
                STCS(&base[(long)k0 * NPLANE + f], sh[sig45(k0) * K2P + f]);
            }
        }
    } else {
        for (int e = threadIdx.x; e < NL * w; e += T2) {
            int k0 = e / w, f = e - k0 * w;
            STCS(&base[(long)k0 * NPLANE + f], sh[sig45(k0) * K2P + f]);
        }
    }
}

/* kernel 1: z- and y-pass over one contiguous (y,z) plane per block.
 * DIRECT=1: the y-pass's DFT9 stage streams straight to global — units indexed
 * by true output kz (column sig45(kz)), so for fixed (k1,c) consecutive threads
 * write consecutive dst addresses. Picked for the L2-resident batches. */
template<int DIRECT>
static __global__ void __launch_bounds__(T1, K1MB)
k1_zy(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    __shared__ double2 sh[NPLANE];
    const double2 *src = in  + (long)blockIdx.x * NPLANE;
    double2       *dst = out + (long)blockIdx.x * NPLANE;

    if (!DIRECT) { k1_plane_staged(src, dst, sh); return; }

    /* stage in: DIRECT (L2-resident batches): __ldg — the same input is
       re-read every execute and fits L2, don't evict it. */
    {
        double2 r[(NPLANE + T1 - 1) / T1];
#pragma unroll
        for (int k = 0; k < NPLANE / T1; ++k)
            r[k] = __ldg(&src[threadIdx.x + k * T1]);
        int e = threadIdx.x + (NPLANE / T1) * T1;
        if (e < NPLANE) r[NPLANE / T1] = __ldg(&src[e]);
#pragma unroll
        for (int k = 0; k < NPLANE / T1; ++k) sh[threadIdx.x + k * T1] = r[k];
        if (e < NPLANE) sh[e] = r[NPLANE / T1];
    }
    __syncthreads();
    pass45<T1, 1, NL>(sh);                 /* z: lines are rows      */
    __syncthreads();                       /* z->y is a transpose    */

    {
        /* y stage 1 as usual */
#pragma unroll
        for (int k = 0; k < (45 * 9 + T1 - 1) / T1; ++k) {
            int i = threadIdx.x + k * T1;
            if ((45 * 9) % T1 == 0 || i < 45 * 9) {
                int col = i % 45, g = i / 45;
                dft5_unit(sh + col, NL, g);
            }
        }
        __syncthreads();
        /* y stage 2, streamed out: unit (kz, c) works on column sig45(kz) and
           holds the 9 outputs X[ky][kz], ky = (10*k1+36*c)%45 */
#pragma unroll
        for (int k = 0; k < (45 * 5 + T1 - 1) / T1; ++k) {
            int i = threadIdx.x + k * T1;
            if ((45 * 5) % T1 == 0 || i < 45 * 5) {
                int kz = i % 45, c = i / 45;
                double2 *p = sh + sig45(kz);
                double2 a[9];
#pragma unroll
                for (int g = 0; g < 9; ++g) {
                    int idx = 5 * g + 9 * c; if (idx >= 45) idx -= 45;
                    a[g] = p[idx * NL];
                }
                dft9(a);
                int r36 = c ? 45 - 9 * c : 0;      /* (36*c) % 45 */
#pragma unroll
                for (int k1 = 0; k1 < 9; ++k1) {
                    int ky = (10 * k1) % 45 + r36; if (ky >= 45) ky -= 45;
                    dst[ky * NL + kz] = a[k1];
                }
            }
        }
    }
}

/* kernel 2: x-pass. Tile = 45(x) rows x 32(flat yz) columns per block, shared
 * pitch 33 (odd, conflict-free), 23.2 KiB -> up to 7 blocks/SM instead of the
 * 5 a 45-wide tile allows. 2025 = 63*32 + 9, so 64 tiles/volume, last one 9
 * wide. In place on out.
 *
 * DIRECT=1: the final DFT9 stage streams its outputs straight to global
 * (coalesced: fixed (k1,c), consecutive f). Wins 6% at the L2-resident batch,
 * loses 2% at the chunked HBM batch, so create() picks per regime. */
template<int DIRECT>
static __global__ void __launch_bounds__(T2, K2MB)
k2_x(double2 *__restrict__ io)
{
    __shared__ double2 sh[NL * K2P];
    const int b  = blockIdx.x / K2NT;
    const int tf = blockIdx.x - b * K2NT;

    if (!DIRECT) { k2_tile_staged(io + (long)b * NVOL, tf, sh); return; }

    const int f0 = tf * K2W;
    const int w  = (NPLANE - f0 < K2W) ? NPLANE - f0 : K2W;   /* 32, or 9 at the tail */
    double2 *base = io + (long)b * NVOL + f0;

    {
        /* stage 1 reads GLOBAL directly — for fixed slot idx, consecutive f is
           coalesced — and writes only the inter-stage intermediate to shared:
           2 shared accesses/point instead of 5, no staging loop, less barriers */
        for (int i = threadIdx.x; i < w * 9; i += T2) {
            int f = i % w, g = i / w;
            double2 a[5];
#pragma unroll
            for (int j = 0; j < 5; ++j) {
                int idx = 5 * g + 9 * j; if (idx >= 45) idx -= 45;
                a[j] = __ldg(&base[(long)idx * NPLANE + f]);
            }
            dft5(a);
#pragma unroll
            for (int j = 0; j < 5; ++j) {
                int idx = 5 * g + 9 * j; if (idx >= 45) idx -= 45;
                sh[idx * K2P + f] = a[j];
            }
        }
        __syncthreads();
    } else {
        /* chunked/HBM regime: stage the tile (measured faster there than the
           direct-load form, which in turn wins in the L2-resident regime) */
        if (w == K2W) {
#pragma unroll
            for (int k = 0; k < (K2ELE + T2 - 1) / T2; ++k) {
                int e = threadIdx.x + k * T2;
                if (K2ELE % T2 == 0 || e < K2ELE) {
                    int x = e / K2W, f = e - x * K2W;
                    sh[x * K2P + f] = LDLU(&base[(long)x * NPLANE + f]);
                }
            }
        } else {
            for (int e = threadIdx.x; e < NL * w; e += T2) {
                int x = e / w, f = e - x * w;
                sh[x * K2P + f] = LDLU(&base[(long)x * NPLANE + f]);
            }
        }
        __syncthreads();
        for (int i = threadIdx.x; i < w * 9; i += T2) {
            int f = i % w, g = i / w;
            dft5_unit(sh + f, K2P, g);
        }
        __syncthreads();
    }
    if (DIRECT) {
        /* final stage streams straight to global: unit (f,c) holds the 9
           outputs X[(10*k1+36*c)%45]; fixed (k1,c) -> consecutive f -> coalesced */
        for (int i = threadIdx.x; i < w * 5; i += T2) {
            int f = i % w, c = i / w;
            double2 a[9];
#pragma unroll
            for (int g = 0; g < 9; ++g) {
                int idx = 5 * g + 9 * c; if (idx >= 45) idx -= 45;
                a[g] = sh[idx * K2P + f];
            }
            dft9(a);
            int r36 = c ? 45 - 9 * c : 0;      /* (36*c) % 45 */
#pragma unroll
            for (int k1 = 0; k1 < 9; ++k1) {
                int k0 = (10 * k1) % 45 + r36; if (k0 >= 45) k0 -= 45;
                base[(long)k0 * NPLANE + f] = a[k1];
            }
        }
    } else {
        for (int i = threadIdx.x; i < w * 5; i += T2) {
            int f = i % w, c = i / w;
            dft9_unit(sh + f, K2P, c);
        }
        __syncthreads();
        if (w == K2W) {
#pragma unroll
            for (int k = 0; k < (K2ELE + T2 - 1) / T2; ++k) {
                int e = threadIdx.x + k * T2;
                if (K2ELE % T2 == 0 || e < K2ELE) {
                    int k0 = e / K2W, f = e - k0 * K2W;
                    STCS(&base[(long)k0 * NPLANE + f], sh[sig45(k0) * K2P + f]);
                }
            }
        } else {
            for (int e = threadIdx.x; e < NL * w; e += T2) {
                int k0 = e / w, f = e - k0 * w;
                STCS(&base[(long)k0 * NPLANE + f], sh[sig45(k0) * K2P + f]);
            }
        }
    }
}

/* ---- host side ---- */
#define MAXNS 8
struct fft3d_gpu_plan {
    int L, B;
    int chunk;   /* volumes per K1+K2 chunk (L2 blocking); B if unchunked */
    int nchunk;
    int d1, d2;  /* per-kernel DIRECT toggles (K1 / K2) */
    int ns;      /* small-batch path: batch split into ns slices, one stream each;
                    0 = use the chunked dual-stream path */
    int rr;      /* >0: chunked path becomes chunk->stream round-robin over rr
                    streams, each chunk's k1;k2 on one stream (L64_radix8's shape) */
    int graph;   /* B=1: replay a captured k1;k2 graph (lazy, keyed on pointers) */
    cudaStream_t s[MAXNS];   /* [0]=K1 pipeline, [1]=K2 pipeline in the chunked
                                path; per-slice streams in the small-batch path */
    cudaEvent_t *evA;        /* K1_n done: K2_n waits on it */
    cudaEvent_t *evB;        /* K2_n done: the NEXT execute's K1_n waits on it */
    cudaGraphExec_t gexec;
    const double2 *gin; double2 *gout;   /* pointers the graph was captured on */
};

static void launch_k1(int d, int nblk, cudaStream_t s, const double2 *in, double2 *out)
{
    if (d) k1_zy<1><<<nblk, T1, 0, s>>>(in, out);
    else   k1_zy<0><<<nblk, T1, 0, s>>>(in, out);
}
static void launch_k2(int d, int nblk, cudaStream_t s, double2 *io)
{
    if (d) k2_x<1><<<nblk, T2, 0, s>>>(io);
    else   k2_x<0><<<nblk, T2, 0, s>>>(io);
}

extern "C" const char *fft3d_gpu_name(void) { return "L45_pfa"; }
extern "C" const char *fft3d_gpu_description(void)
{ return "two-pass PFA 9x5: unit-parallel zy-plane + x-tile kernels; evict-first-hinted L2 chunks (9) round-robined on 2 streams; slice-per-stream + direct stores at L2-resident B; B=1 graph"; }
extern "C" int fft3d_gpu_supports(int L) { return L == 45; }

extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)
{
    if (L != 45 || batch < 1) return NULL;

    Coef h;
    for (int k = 1; k <= 4; ++k)
        for (int j = 1; j <= 4; ++j) {
            h.c9[k-1][j-1] = cos(2.0 * M_PI * k * j / 9.0);
            h.s9[k-1][j-1] = sin(2.0 * M_PI * k * j / 9.0);
        }
    for (int k = 1; k <= 2; ++k)
        for (int j = 1; j <= 2; ++j) {
            h.c5[k-1][j-1] = cos(2.0 * M_PI * k * j / 5.0);
            h.s5[k-1][j-1] = sin(2.0 * M_PI * k * j / 5.0);
        }
    if (cudaMemcpyToSymbol(CF, &h, sizeof h) != cudaSuccess) return NULL;

    /* full shared carveout -> 5 blocks/SM at 31.6 KiB each */
    cudaFuncSetAttribute(k1_zy<0>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(k1_zy<1>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(k2_x<0>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    cudaFuncSetAttribute(k2_x<1>, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    fft3d_gpu_plan *p = (fft3d_gpu_plan *)malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L; p->B = batch;
    p->gexec = NULL; p->gin = NULL; p->gout = NULL;

    /* chunk so the K1->K2 intermediate stays L2-resident; whole batch if it
       already fits (in+out at B<=14 is < 40 MiB), else 9 volumes. gpu_r1 had 12
       and the sweep was flat; with the evict-first streaming hints the chunk
       lever finally works (L64_radix8's "hints are the mechanism" lesson) and
       the swept optimum moved to 9 (2604 vs 3023 µs at 12, B=736). */
    int ch = (batch <= 14) ? batch : 9;
    const char *e = getenv("FFT45_CHUNK");
    if (e && atoi(e) > 0) ch = atoi(e);
    if (ch > batch) ch = batch;
    p->chunk = ch;
    p->nchunk = (batch + ch - 1) / ch;

    /* per-kernel direct-store regime: both direct when L2-resident, both staged
       chunked. FFT45_DIRECT sets both; FFT45_D1/FFT45_D2 override individually. */
    p->d1 = p->d2 = (batch <= 14);
    const char *ed = getenv("FFT45_DIRECT");
    if (ed) p->d1 = p->d2 = atoi(ed);
    if ((ed = getenv("FFT45_D1"))) p->d1 = atoi(ed);
    if ((ed = getenv("FFT45_D2"))) p->d2 = atoi(ed);

    /* small-batch path (borrowed from L36_sharedtiled gpu_r2): split the batch
       into one contiguous slice per stream, each stream running its own k1;k2
       pair, so k2 of one slice fills the tail waves of k1 of another instead of
       the whole batch draining at the k1->k2 barrier. ns=0 forces the chunked
       path. B=1 has nothing to split: one stream, optionally graph-replayed. */
    p->ns = 0;
    if (batch <= 14) p->ns = (batch < 4) ? batch : 4;   /* swept at B=11: 4 > 2,3 > 6 >> 1,11 */
    if ((ed = getenv("FFT45_NS"))) p->ns = atoi(ed);
    if (p->ns > batch) p->ns = batch;
    if (p->ns > MAXNS) p->ns = MAXNS;
    /* large batches: chunk->stream round-robin over 2 streams (L64_radix8's
       shape) replaced the r1 producer/consumer split — 2449 vs 2608 µs at
       B=736 (rr∈{3,4} and the event pipeline all lose; sweep in the record). */
    p->rr = (batch > 14) ? 2 : 0;
    if ((ed = getenv("FFT45_RR"))) p->rr = atoi(ed);
    if (p->rr > MAXNS) p->rr = MAXNS;
    if (p->rr < 0) p->rr = 0;
    p->graph = (batch == 1 && p->ns == 1);
    if ((ed = getenv("FFT45_GRAPH"))) p->graph = atoi(ed);

    /* K1s run on s[0], K2s on s[1]: K1 of chunk n+1 overlaps K2 of chunk n, so
       the load-compute-store phases of the two kernels interleave on the SMs
       instead of the whole one-wave grid bursting and idling in lockstep. */
    int nstream = (p->ns > 2) ? p->ns : 2;
    if (p->rr > nstream) nstream = p->rr;
    for (int i = 0; i < nstream; ++i)
        cudaStreamCreateWithFlags(&p->s[i], cudaStreamNonBlocking);
    for (int i = nstream; i < MAXNS; ++i) p->s[i] = NULL;
    p->evA = (cudaEvent_t *)malloc(2 * p->nchunk * sizeof(cudaEvent_t));
    p->evB = p->evA + p->nchunk;
    for (int i = 0; i < 2 * p->nchunk; ++i)
        cudaEventCreateWithFlags(&p->evA[i], cudaEventDisableTiming);
    return p;
}

/* small-batch path: slice i (contiguous volumes) runs k1;k2 back to back on its
 * own stream — ordering within a slice and across executes is implicit (same
 * stream, same slice boundaries every call), and slices touch disjoint regions. */
static void execute_sliced(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    int base = p->B / p->ns, rem = p->B % p->ns;
    for (int i = 0, b0 = 0; i < p->ns; ++i) {
        int c = base + (i < rem);
        launch_k1(p->d1, c * NL, p->s[i], in + (long)b0 * NVOL, out + (long)b0 * NVOL);
        launch_k2(p->d2, c * K2NT, p->s[i], out + (long)b0 * NVOL);
        b0 += c;
    }
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (p->ns == 1 && p->graph) {
        /* B=1: one graph launch instead of two kernel launches. Captured lazily
           and keyed on the buffer pointers, so a pointer change just recaptures
           (repeatability is unaffected: the graph replays the same kernels). */
        if (in != p->gin || out != p->gout) {
            if (p->gexec) { cudaGraphExecDestroy(p->gexec); p->gexec = NULL; }
            cudaGraph_t g;
            cudaStreamBeginCapture(p->s[0], cudaStreamCaptureModeThreadLocal);
            execute_sliced(p, in, out);
            cudaStreamEndCapture(p->s[0], &g);
            cudaGraphInstantiateWithFlags(&p->gexec, g, 0);
            cudaGraphDestroy(g);
            p->gin = in; p->gout = out;
        }
        cudaGraphLaunch(p->gexec, p->s[0]);
        return;
    }
    if (p->ns > 0) { execute_sliced(p, in, out); return; }

    if (p->rr > 0) {
        /* chunk->stream round-robin (L64_radix8's shape): chunk n runs k1;k2 on
           stream n%rr — same map every execute, so ordering per region is
           implicit and no events are needed. */
        for (int b0 = 0, n = 0; b0 < p->B; b0 += p->chunk, ++n) {
            int c = p->B - b0; if (c > p->chunk) c = p->chunk;
            cudaStream_t s = p->s[n % p->rr];
            launch_k1(p->d1, c * NL, s, in + (long)b0 * NVOL, out + (long)b0 * NVOL);
            launch_k2(p->d2, c * K2NT, s, out + (long)b0 * NVOL);
        }
        return;
    }

    for (int b0 = 0, n = 0; b0 < p->B; b0 += p->chunk, ++n) {
        int c = p->B - b0; if (c > p->chunk) c = p->chunk;
        /* region n may still be in flight in K2 of the PREVIOUS execute */
        cudaStreamWaitEvent(p->s[0], p->evB[n], 0);
        launch_k1(p->d1, c * NL, p->s[0], in + (long)b0 * NVOL, out + (long)b0 * NVOL);
        cudaEventRecord(p->evA[n], p->s[0]);
        cudaStreamWaitEvent(p->s[1], p->evA[n], 0);
        launch_k2(p->d2, c * K2NT, p->s[1], out + (long)b0 * NVOL);
        cudaEventRecord(p->evB[n], p->s[1]);
    }
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p)
{
    if (!p) return;
    if (p->gexec) cudaGraphExecDestroy(p->gexec);
    for (int i = 0; i < 2 * p->nchunk; ++i) cudaEventDestroy(p->evA[i]);
    free(p->evA);
    for (int i = 0; i < MAXNS; ++i)
        if (p->s[i]) cudaStreamDestroy(p->s[i]);
    free(p);
}
