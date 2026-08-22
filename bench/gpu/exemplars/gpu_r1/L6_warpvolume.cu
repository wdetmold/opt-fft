/* L6_warpvolume -- L = 6 on one A100. WHOLE VOLUME IN WARP REGISTERS, ZERO SHARED MEMORY.
 *
 * Strategy (round gpu_r1): literature 09 Section 9.1 structure #1 -- the register-resident
 * fused kernel. 8 lanes per volume, 4 volumes per warp, no idle lanes. Lane q = (ox,oy,oz)
 * owns the 27 points with x = ox (mod 2), y = oy (mod 2), z = oz (mod 2) -- a 3x3x3
 * parity sub-lattice, 27 complex doubles = 108 registers of data.
 *
 * Each axis is one radix-2x3 stage split across the lane pair that differs in that
 * axis's parity bit: a local Winograd DFT3, a w6^m twiddle on the odd lane (2 cmuls,
 * invisible at 5.15x bandwidth-bound -- lit. 09 Section 2.2), and a cross-lane DFT2
 * butterfly: 2 __shfl_xor_sync of 8 B per complex. No shared memory, no __syncthreads,
 * no bank conflicts. One global read, three all-register passes, one global write.
 *
 * The z axis runs DIF (butterfly -> twiddle -> DFT3; input halves, output interleaved)
 * and y, x run DIT (DFT3 -> twiddle -> butterfly; input interleaved, output halves).
 * z is the only axis where 16-byte elements share 32-byte sectors, and DIF-z makes the
 * two z-parity lanes WRITE the two halves of the same 32-byte sector in the same store
 * instruction (perfect full-sector stores) while the loads become 48-byte per-lane runs
 * the L1 heals across adjacent instructions. The alternatives measured worse at
 * B=310608: Good-Thomas per axis (CRT map scatters both sides; 1117 GB/s), all-DIT
 * (perfect loads, half-sector stores; 1256 GB/s) vs this (1373 GB/s) -- nothing heals
 * store wavefronts before L2, so the store side is the one to make perfect. Every
 * load/store offset is a compile-time immediate. Stores are evict-first (__stcs; out is
 * never re-read), which stops the write stream thrashing in's L2 residency at the
 * L2-resident batch point (28.0 -> 18.6 us there).
 *
 * Why registers and not shared (the difference from L6_batchcoalesced): a fused 3-axis
 * shared-memory kernel moves ~128 B of shared traffic per 32 B of HBM traffic; at the
 * L2-resident batch point that needs ~29 TB/s of shared bandwidth against the chip's
 * 19.5 TB/s. The warp-resident version's only inter-lane motion is 162 shuffle
 * instructions per warp per 4 volumes. See lit. 09 Sections 2.4, 6.2, 6.4.
 *
 * Occupancy: 148 registers/thread makes the register file, not threads, the residency
 * limiter; one-warp blocks dodge the block-granularity penalty (256-thread blocks =
 * 1 block/SM = 12.5% occupancy, 1922 us at B=310608; 32/64-thread blocks = 12-13 warps,
 * ~1550 us). Forcing 128 regs via launch_bounds spilled 40 B/thread and LOST (1970 us):
 * keep the natural allocation.
 */
#include <cuda_runtime.h>
#include <stdlib.h>

#include "../fft3d_gpu_api.h"

#define NP  216   /* 6^3 */
#define VPW 4     /* volumes per warp  */
#define TPB 32    /* threads per block: warps are independent; small blocks dodge the
                     register-file block-granularity penalty (see header) */
#define VPB (VPW * (TPB / 32))

/* Winograd forward DFT-3, X = F3 * (a,b,c), w3 = exp(-2*pi*i/3).
 * Standard module; same as L6_batchcoalesced and ../../geom/strategies/L6_unrolled.md. */
static __device__ __forceinline__ void dft3(double2 a, double2 b, double2 c,
                                            double2 &X0, double2 &X1, double2 &X2)
{
    const double S3 = 0.86602540378443864676;  /* sqrt(3)/2 */
    double tr = b.x + c.x, ti = b.y + c.y;
    double ur = a.x - 0.5 * tr, ui = a.y - 0.5 * ti;
    double dr = S3 * (b.x - c.x), di = S3 * (b.y - c.y);
    X0.x = a.x + tr; X0.y = a.y + ti;
    X1.x = ur + di;  X1.y = ui - dr;
    X2.x = ur - di;  X2.y = ui + dr;
}

static __device__ __forceinline__ double2 shflx(double2 v, int laneMask)
{
    double2 r;
    r.x = __shfl_xor_sync(0xffffffffu, v.x, laneMask, 32);
    r.y = __shfl_xor_sync(0xffffffffu, v.y, laneMask, 32);
    return r;
}

/* One DIT 2x3 axis pass over the 9 lines of this lane's 3x3x3 block.
 * HIBIT = this lane's parity bit along the axis (0: holds evens, produces X[0..2];
 * 1: holds odds, produces X[3..5]). IDX(l, j) must be a compile-time constant. */
#define AXIS_PASS(XORM, HIBIT, IDX)                                              \
    {                                                                            \
        const double S3 = 0.86602540378443864676;                                \
        const int    hi = (HIBIT);                                               \
        const double sgn  = hi ? -1.0 : 1.0;                                     \
        const double t1r  = hi ?  0.5 : 1.0,  t1i = hi ? -S3 : 0.0;              \
        const double t2r  = hi ? -0.5 : 1.0,  t2i = hi ? -S3 : 0.0;              \
        _Pragma("unroll")                                                        \
        for (int l = 0; l < 9; ++l) {                                            \
            double2 e0, e1, e2;                                                  \
            dft3(r[IDX(l, 0)], r[IDX(l, 1)], r[IDX(l, 2)], e0, e1, e2);          \
            double2 a1, a2;                       /* w6^m twiddle (odd lane) */  \
            a1.x = t1r * e1.x - t1i * e1.y; a1.y = t1r * e1.y + t1i * e1.x;      \
            a2.x = t2r * e2.x - t2i * e2.y; a2.y = t2r * e2.y + t2i * e2.x;      \
            double2 b0 = shflx(e0, XORM), b1 = shflx(a1, XORM),                  \
                    b2 = shflx(a2, XORM);                                        \
            r[IDX(l, 0)].x = b0.x + sgn * e0.x;                                  \
            r[IDX(l, 0)].y = b0.y + sgn * e0.y;                                  \
            r[IDX(l, 1)].x = b1.x + sgn * a1.x;                                  \
            r[IDX(l, 1)].y = b1.y + sgn * a1.y;                                  \
            r[IDX(l, 2)].x = b2.x + sgn * a2.x;                                  \
            r[IDX(l, 2)].y = b2.y + sgn * a2.y;                                  \
        }                                                                        \
    }

/* DIF 2x3 axis pass (butterfly -> twiddle -> DFT3): input halves, output interleaved.
 * Used on the z axis so the STORE side gets the full-sector pairing instead of the load
 * side (A/B experiment against DIT-z; see strategy record). */
#define AXIS_PASS_DIF(XORM, HIBIT, IDX)                                          \
    {                                                                            \
        const double S3 = 0.86602540378443864676;                                \
        const int    hi = (HIBIT);                                               \
        const double sgn  = hi ? -1.0 : 1.0;                                     \
        const double t1r  = hi ?  0.5 : 1.0,  t1i = hi ? -S3 : 0.0;              \
        const double t2r  = hi ? -0.5 : 1.0,  t2i = hi ? -S3 : 0.0;              \
        _Pragma("unroll")                                                        \
        for (int l = 0; l < 9; ++l) {                                            \
            double2 a0 = r[IDX(l, 0)], a1 = r[IDX(l, 1)], a2 = r[IDX(l, 2)];     \
            double2 b0 = shflx(a0, XORM), b1 = shflx(a1, XORM),                  \
                    b2 = shflx(a2, XORM);                                        \
            double2 t0, u1, u2, t1, t2;                                          \
            t0.x = b0.x + sgn * a0.x; t0.y = b0.y + sgn * a0.y;                  \
            u1.x = b1.x + sgn * a1.x; u1.y = b1.y + sgn * a1.y;                  \
            u2.x = b2.x + sgn * a2.x; u2.y = b2.y + sgn * a2.y;                  \
            t1.x = t1r * u1.x - t1i * u1.y; t1.y = t1r * u1.y + t1i * u1.x;      \
            t2.x = t2r * u2.x - t2i * u2.y; t2.y = t2r * u2.y + t2i * u2.x;      \
            dft3(t0, t1, t2, r[IDX(l, 0)], r[IDX(l, 1)], r[IDX(l, 2)]);          \
        }                                                                        \
    }

#define IDXZ(l, j) ((l) * 3 + (j))                          /* z fast, stride 1 */
#define IDXY(l, j) (((l) / 3) * 9 + (j) * 3 + ((l) % 3))    /* y stride 3       */
#define IDXX(l, j) ((j) * 9 + (l))                          /* x stride 9       */

/* STREAM=true is instantiated for batches whose working set cannot live in the 40 MiB
 * L2: reads and writes get evict-first hints (__ldcs/__stcs) so streamed-once lines do
 * not fight for L2. For L2-resident batches the default policy keeps them cached. */
template <bool STREAM>
__global__ void __launch_bounds__(TPB)
fft6_warpvolume(const double2 *__restrict__ in, double2 *__restrict__ out, int nvol)
{
    const int lane = (int)threadIdx.x & 31;
    const int gwarp = (int)blockIdx.x * (TPB / 32) + ((int)threadIdx.x >> 5);

    if ((long long)gwarp * VPW >= nvol) return;   /* warp-uniform early out */

    const int q  = lane & 7;                       /* octant within the volume  */
    const int ox = (q >> 2) & 1, oy = (q >> 1) & 1, oz = q & 1;
    const long long vol = (long long)gwarp * VPW + (lane >> 3);
    const bool active = vol < (long long)nvol;

    double2 r[27];
    /* DIT input map: this lane holds (x,y,z) = (2sx+ox, 2sy+oy, 2sz+oz).
     * The two z-parity lanes of a pair read the two halves of one 32 B sector in the
     * same instruction; every offset below is a compile-time immediate. */
    const double2 *p = in + vol * NP + (ox * 36 + oy * 6 + oz * 3);
    if (active) {
#pragma unroll
        for (int sx = 0; sx < 3; ++sx)
#pragma unroll
            for (int sy = 0; sy < 3; ++sy)
#pragma unroll
                for (int sz = 0; sz < 3; ++sz)
                    r[(sx * 3 + sy) * 3 + sz] =
                        STREAM ? __ldcs(&p[sx * 72 + sy * 12 + sz])
                               : p[sx * 72 + sy * 12 + sz];
    } else {
#pragma unroll
        for (int i = 0; i < 27; ++i) { r[i].x = 0.0; r[i].y = 0.0; }
    }

    AXIS_PASS_DIF(1, oz, IDXZ)   /* z axis DIF: full-sector stores on the fast axis */
    AXIS_PASS(2, oy, IDXY)   /* y axis */
    AXIS_PASS(4, ox, IDXX)   /* x axis */

    /* DIT output map: this lane owns (kx,ky,kz) = (3ox+sx, 3oy+sy, 3oz+sz). */
    if (active) {
        double2 *o = out + vol * NP + (ox * 108 + oy * 18 + oz);
#pragma unroll
        for (int sx = 0; sx < 3; ++sx)
#pragma unroll
            for (int sy = 0; sy < 3; ++sy)
#pragma unroll
                for (int sz = 0; sz < 3; ++sz)
                    /* out is never re-read: evict-first stores keep the write stream
                       from thrashing in's L2 residency at the L2-resident batch */
                    __stcs(&o[sx * 36 + sy * 6 + sz * 2], r[(sx * 3 + sy) * 3 + sz]);
    }
}

struct fft3d_gpu_plan { int L; int batch; int grid; int stream; };

extern "C" const char *fft3d_gpu_name(void) { return "L6_warpvolume"; }
extern "C" const char *fft3d_gpu_description(void)
{
    return "L6: 4 volumes/warp all-register, DIF-z/DIT 2x3 parity split, "
           "shuffle butterflies, zero shared memory, one kernel";
}

extern "C" int fft3d_gpu_supports(int L) { return L == 6; }

extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)
{
    if (L != 6 || batch < 1) return NULL;
    fft3d_gpu_plan *p = (fft3d_gpu_plan *)malloc(sizeof(*p));
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->grid = (batch + VPB - 1) / VPB;
    /* evict-first streaming hints only when in+out cannot sit in the 40 MiB L2 */
    p->stream = ((long long)batch * NP * 32 > 40ll * 1024 * 1024);
    /* no shared memory at all -- give the whole carveout to L1 */
    cudaFuncSetAttribute(fft6_warpvolume<false>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxL1);
    cudaFuncSetAttribute(fft6_warpvolume<true>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxL1);
    return p;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (p->stream) fft6_warpvolume<true><<<p->grid, TPB>>>(in, out, p->batch);
    else           fft6_warpvolume<false><<<p->grid, TPB>>>(in, out, p->batch);
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p) { free(p); }
