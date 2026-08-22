/* L6_warpvolume -- L = 6 on one A100. WHOLE VOLUME IN WARP REGISTERS, ZERO SHARED MEMORY.
 *
 * Strategy (round gpu_r1): literature 09 Section 9.1 structure #1 -- the register-resident
 * fused kernel. 8 lanes per volume, 4 volumes per warp, no idle lanes. Lane q = (ox,oy,oz)
 * owns the 27 points with x = ox (mod 2), y = oy (mod 2), z = oz (mod 2) -- a 3x3x3
 * parity sub-lattice, 27 complex doubles = 108 registers of data.
 *
 * Each axis is a DIT 2x3 stage split across the lane pair that differs in that axis's
 * parity bit:
 *      X[m]   = E[m] + w6^m O[m]         E = DFT3(even samples)   (this lane if o=0)
 *      X[m+3] = E[m] - w6^m O[m]         O = DFT3(odd samples)    (this lane if o=1)
 * i.e. local Winograd DFT3, a twiddle by w6^m on the odd lane (2 cmuls, invisible at
 * 5.15x bandwidth-bound -- lit. 09 Section 2.2), then a cross-lane butterfly:
 * 2 __shfl_xor_sync of 8 B per complex. No shared memory, no __syncthreads, no bank
 * conflicts. One global read, three all-register passes, one global write.
 *
 * Why DIT and not the CPU record's twiddle-free Good-Thomas (tried first, this round):
 * PFA's CRT input map hands each lane the z-order {0,2,4}/{3,5,1}, so a warp load
 * instruction touches 32 DISTINCT 32-byte sectors and uses 16 bytes of each -- ncu
 * showed the L1 healing that at 67% hit rate but at twice the wavefronts, 12.5%
 * occupancy, and 1117 GB/s. DIT's parity map makes the two z-parity lanes read the two
 * halves of the SAME 32-byte sector in the same instruction (full sectors, zero waste),
 * and every load/store offset is a compile-time immediate. Stores are the mirror image
 * (48-byte half-runs); write sectors merge in L2 before DRAM writeback, so reads are
 * the side to make perfect.
 *
 * Why registers and not shared (the difference from L6_batchcoalesced): a fused 3-axis
 * shared-memory kernel moves ~128 B of shared traffic per 32 B of HBM traffic; at the
 * L2-resident batch point that needs ~29 TB/s of shared bandwidth against the chip's
 * 19.5 TB/s. The warp-resident version's only inter-lane motion is 162 shuffle
 * instructions per warp per 4 volumes. See lit. 09 Sections 2.4, 6.2, 6.4.
 *
 * Occupancy: 136 registers/thread makes registers the block limiter; small blocks pack
 * more warps per SM (256-thread blocks = 1 block = 12.5%; 64-thread blocks = 7 blocks =
 * 21.9% -- measured 1922 -> 1737 us at B=310608). Forcing 128 regs via launch_bounds
 * spilled 40 B/thread and LOST (1970 us): keep the natural allocation.
 */
#include <cuda_runtime.h>
#include <stdlib.h>

#include "../fft3d_gpu_api.h"

#define NP  216   /* 6^3 */
#define VPW 4     /* volumes per warp  */
#define TPB 64    /* threads per block: warps are independent; small blocks dodge the
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

#define IDXZ(l, j) ((l) * 3 + (j))                          /* z fast, stride 1 */
#define IDXY(l, j) (((l) / 3) * 9 + (j) * 3 + ((l) % 3))    /* y stride 3       */
#define IDXX(l, j) ((j) * 9 + (l))                          /* x stride 9       */

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
    const double2 *p = in + vol * NP + (ox * 36 + oy * 6 + oz);
    if (active) {
#pragma unroll
        for (int sx = 0; sx < 3; ++sx)
#pragma unroll
            for (int sy = 0; sy < 3; ++sy)
#pragma unroll
                for (int sz = 0; sz < 3; ++sz)
                    r[(sx * 3 + sy) * 3 + sz] = p[sx * 72 + sy * 12 + sz * 2];
    } else {
#pragma unroll
        for (int i = 0; i < 27; ++i) { r[i].x = 0.0; r[i].y = 0.0; }
    }

    AXIS_PASS(1, oz, IDXZ)   /* z axis: partner differs in oz */
    AXIS_PASS(2, oy, IDXY)   /* y axis */
    AXIS_PASS(4, ox, IDXX)   /* x axis */

    /* DIT output map: this lane owns (kx,ky,kz) = (3ox+sx, 3oy+sy, 3oz+sz). */
    if (active) {
        double2 *o = out + vol * NP + (ox * 108 + oy * 18 + oz * 3);
#pragma unroll
        for (int sx = 0; sx < 3; ++sx)
#pragma unroll
            for (int sy = 0; sy < 3; ++sy)
#pragma unroll
                for (int sz = 0; sz < 3; ++sz)
                    o[sx * 36 + sy * 6 + sz] = r[(sx * 3 + sy) * 3 + sz];
    }
}

struct fft3d_gpu_plan { int L; int batch; int grid; };

extern "C" const char *fft3d_gpu_name(void) { return "L6_warpvolume"; }
extern "C" const char *fft3d_gpu_description(void)
{
    return "L6: 4 volumes/warp all-register, DIT 2x3 parity split, "
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
    /* no shared memory at all -- give the whole carveout to L1 */
    cudaFuncSetAttribute(fft6_warpvolume, cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxL1);
    return p;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    fft6_warpvolume<<<p->grid, TPB>>>(in, out, p->batch);
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p) { free(p); }
