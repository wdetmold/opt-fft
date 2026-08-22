/* L8_warpradix8 -- L = 8 on one A100.
 *
 * Technique (round gpu_r1): ONE VOLUME PER WARP, entirely in registers, no shared
 * memory, no __syncthreads. 512 complex doubles = 16 per lane (64 registers of data,
 * 25% of the warp register file, lit. 09 regime A / SS9.2 structure 2).
 *
 * Production path (var 2, "cross-lane"): flat perfectly-coalesced load
 * v[r] = in[vol*512 + r*32 + lane], which lands two complete x-lines per lane
 * (lane = 8h + w: y in {h, h+4}, z = w). Pass X is two min-op radix-8 DIF codelets
 * in registers (4 mults + 52 adds, natural in/out -- same arithmetic as the CPU
 * winner, codelet taken from L8_blockfused round gpu_r1). The y and z passes are
 * then done as DIF radix-2 stages DIRECTLY ACROSS THE LANES that hold those axes
 * -- y = 4a + h: one local stage on the register bit a, two __shfl_xor_sync stages
 * over the h bits; z = w: three shuffle stages over the w bits. Each cross-lane
 * butterfly is 2 shuffles + 2 FMA (pre = recv + s*mine with s = +/-1 by side) plus
 * one table-free complex multiply by a per-lane twiddle hoisted out of the loop;
 * there is not a single select or lane-dependent branch in the hot path. DIF
 * leaves each axis bit-reversed across its lane bits, and the STORE ADDRESS
 * absorbs that permutation for free: a warp's 32 addresses are the same set
 * whatever the lane holds, so the 16 stores stay 4x128B sector-dense.
 *
 * Vars 0/1 (kept for env-forced experiments; excluded from the autotune so the
 * plan choice stays bit-repeatable): three dft8 passes with BUTTERFLY SHUFFLE
 * TRANSPOSES between axes -- log2(P) rounds of select-predicated pairwise
 * exchanges among P lanes, landing blocks in natural register order (an induction
 * on the settled lane bits). var 1 transposes back after pass Z for flat stores;
 * var 0 does per-lane 128B line stores, which measured 1.47x WORSE at B_HBM.
 * var 3: var 2 double-buffered two volumes per warp (180 regs) -- measured worse
 * everywhere (occupancy loss beats latency self-hiding); still in the tune space
 * since it is bit-identical to var 2.
 *
 * B = 1 uses a dedicated 64-thread two-warp kernel (fft8_b1): x-line per lane,
 * cross-lane y stages, and only the z2 bit crosses warps through one padded
 * shared bounce with a single barrier (staging idea from L8_blockfused's B=1
 * kernel, cut to one barrier). create()-time measured autotune on scratch
 * buffers adopted from L8_blockfused round gpu_r1.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "../fft3d_gpu_api.h"

static __device__ __forceinline__ double2 cadd(double2 a, double2 b)
{ return make_double2(a.x + b.x, a.y + b.y); }
static __device__ __forceinline__ double2 csub(double2 a, double2 b)
{ return make_double2(a.x - b.x, a.y - b.y); }

/* Forward 8-point DFT, decimation-in-frequency, natural order in and out, in place.
 * 4 real multiplies + 52 real adds (the published minimum; same codelet as
 * L8_blockfused / the CPU-phase L8 winner). */
static __device__ __forceinline__ void dft8(double2 &a0, double2 &a1, double2 &a2,
                                            double2 &a3, double2 &a4, double2 &a5,
                                            double2 &a6, double2 &a7)
{
    const double R2 = 0.70710678118654752440; /* 1/sqrt(2) */
    const double2 t0 = cadd(a0, a4), t1 = cadd(a1, a5), t2 = cadd(a2, a6), t3 = cadd(a3, a7);
    const double2 s0 = csub(a0, a4), s1 = csub(a1, a5), s2 = csub(a2, a6), s3 = csub(a3, a7);
    const double2 u1 = make_double2((s1.x + s1.y) * R2, (s1.y - s1.x) * R2);  /* * (1-i)/r2  */
    const double2 u2 = make_double2(s2.y, -s2.x);                             /* * (-i)      */
    const double2 u3 = make_double2((s3.y - s3.x) * R2, -(s3.x + s3.y) * R2); /* * -(1+i)/r2 */
    const double2 e0 = cadd(t0, t2), e1 = cadd(t1, t3), f0 = csub(t0, t2);
    const double2 f1t = csub(t1, t3);
    const double2 f1 = make_double2(f1t.y, -f1t.x);
    const double2 g0 = cadd(s0, u2), g1 = cadd(u1, u3), h0 = csub(s0, u2);
    const double2 h1t = csub(u1, u3);
    const double2 h1 = make_double2(h1t.y, -h1t.x);
    a0 = cadd(e0, e1);
    a4 = csub(e0, e1);
    a2 = cadd(f0, f1);
    a6 = csub(f0, f1);
    a1 = cadd(g0, g1);
    a5 = csub(g0, g1);
    a3 = cadd(h0, h1);
    a7 = csub(h0, h1);
}

/* One butterfly-exchange step: the (lo,hi) register pair holds the two blocks whose
 * destination differs in the lane bit `lm`; lanes on the `up` side ship lo and keep
 * hi, the others ship hi and keep lo. All 32 lanes shuffle with a full mask. */
static __device__ __forceinline__ void xchg(double2 &lo, double2 &hi, bool up, int lm)
{
    double2 m = up ? lo : hi;
    double2 r;
    r.x = __shfl_xor_sync(0xffffffffu, m.x, lm);
    r.y = __shfl_xor_sync(0xffffffffu, m.y, lm);
    lo = up ? r : lo;
    hi = up ? hi : r;
}

/* Register index of (c, j): line pair member c in {0,1}, in-line index j in [0,8). */
#define R8(c, j) (4 * ((j) & 3) + 2 * (c) + ((j) >> 2))

static __device__ __forceinline__ double2 cmul(double2 a, double2 t)
{ return make_double2(a.x * t.x - a.y * t.y, a.x * t.y + a.y * t.x); }

/* One cross-lane DIF butterfly stage over lane bit MASK, on all 16 registers.
 * pre = recv + s*mine (s = -1 on the high side, +1 on the low side) gives
 * a+b on the low lane and a-b on the high lane in 2 FMA; the per-lane twiddle
 * t (identity on the low side) is precomputed once outside. */
template <int MASK, bool TW>
static __device__ __forceinline__ void xstage(double2 (&v)[16], double s, double2 t)
{
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        double2 rec;
        rec.x = __shfl_xor_sync(0xffffffffu, v[r].x, MASK);
        rec.y = __shfl_xor_sync(0xffffffffu, v[r].y, MASK);
        double2 pre = make_double2(fma(s, v[r].x, rec.x), fma(s, v[r].y, rec.y));
        v[r] = TW ? cmul(pre, t) : pre;
    }
}

template <int WPB, bool FLAT2, bool STCS>
__global__ void __launch_bounds__(WPB * 32, 16 / WPB)
fft8_warp(const double2 *__restrict__ in, double2 *__restrict__ out, int B)
{
    const int  lane = threadIdx.x & 31;
    const long vol  = (long)blockIdx.x * WPB + (threadIdx.x >> 5);
    if (vol >= (long)B) return;
    const int w = lane & 7;
    const int h = lane >> 3;
    const double2 *gi = in + vol * 512 + lane;
    double2 *go = out + vol * 512;

    double2 v[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) v[r] = gi[r * 32];
    /* v[2i+a] = (x=i, y=4a+h, z=w) */

    /* pass X: two 8-point lines at register stride 2 */
    dft8(v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]);
    dft8(v[1], v[3], v[5], v[7], v[9], v[11], v[13], v[15]);

    /* TrA: 4-lane transpose over h (blocks of 4 regs = kx pair x y-half) */
#pragma unroll
    for (int d = 1; d <= 2; d <<= 1) {
        const bool up = (h & d) != 0;
#pragma unroll
        for (int g = 0; g < 4; ++g)
            if (!(g & d)) {
#pragma unroll
                for (int k = 0; k < 4; ++k)
                    xchg(v[4 * g + k], v[4 * (g | d) + k], up, d << 3);
            }
    }
    /* v[4u+2c+a] = (kx=2h+c, y=4a+u, z=w) */

    /* pass Y: lines c=0,1 on the R8 register pattern */
    dft8(v[0], v[4], v[8], v[12], v[1], v[5], v[9], v[13]);
    dft8(v[2], v[6], v[10], v[14], v[3], v[7], v[11], v[15]);

    /* TrB: 8-lane transpose over w (pairs of single regs, per line c) */
#pragma unroll
    for (int d = 1; d <= 4; d <<= 1) {
        const bool up = (w & d) != 0;
#pragma unroll
        for (int c = 0; c < 2; ++c)
#pragma unroll
            for (int j = 0; j < 8; ++j)
                if (!(j & d))
                    xchg(v[R8(c, j)], v[R8(c, j | d)], up, d);
    }
    /* v[R8(c,z)] = (kx=2h+c, ky=w, z) */

    /* pass Z */
    dft8(v[0], v[4], v[8], v[12], v[1], v[5], v[9], v[13]);
    dft8(v[2], v[6], v[10], v[14], v[3], v[7], v[11], v[15]);
    /* v[R8(c,kz)] = (kx=2h+c, ky=w, kz) */

    if (FLAT2) {
        /* TrC (TrB shape): -> v[R8(c,j)] = (kx=2h+c, ky=j, kz=w) */
#pragma unroll
        for (int d = 1; d <= 4; d <<= 1) {
            const bool up = (w & d) != 0;
#pragma unroll
            for (int c = 0; c < 2; ++c)
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    if (!(j & d))
                        xchg(v[R8(c, j)], v[R8(c, j | d)], up, d);
        }
        /* TrD (TrA shape): -> v[4u+2c+a] = (kx=2u+c, ky=4a+h, kz=w) */
#pragma unroll
        for (int d = 1; d <= 2; d <<= 1) {
            const bool up = (h & d) != 0;
#pragma unroll
            for (int g = 0; g < 4; ++g)
                if (!(g & d)) {
#pragma unroll
                    for (int k = 0; k < 4; ++k)
                        xchg(v[4 * g + k], v[4 * (g | d) + k], up, d << 3);
                }
        }
        /* mirror of the load: out[r*32 + lane] = v[r], 16 coalesced 512B stores */
        double2 *gs = go + lane;
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (STCS) __stcs(gs + r * 32, v[r]);
            else      gs[r * 32] = v[r];
        }
    } else {
        /* per-lane 128B-aligned z-line stores */
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            double2 *gs = go + (2 * h + c) * 64 + 8 * w;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (STCS) __stcs(gs + j, v[R8(c, j)]);
                else      gs[j] = v[R8(c, j)];
            }
        }
    }
}

/* Cross-lane variant: no transposes at all. The y and z axis FFTs are done as DIF
 * radix-2 stages directly across the lanes that hold them (y = 4a + h: one local
 * stage on the register bit a, two shuffle stages over the h bits; z = w: three
 * shuffle stages over the w bits). DIF leaves each axis bit-reversed across its lane
 * bits, and the store address absorbs that lane permutation for free -- a warp's 32
 * store addresses are the same set whatever the lane assignment, so coalescing is
 * untouched (16 stores, each 4 dense 128B runs). Twiddles are per-lane constants
 * hoisted out of the stages: the hot path has no selects at all. */
struct l8c {
    double2 W8j, tY2, tZ1, tZ2;
    double sY2, sY3, sZ1, sZ2, sZ3;
    int H, Wp;
};

static __device__ __forceinline__ void l8_setup(int lane, l8c &c)
{
    const int w = lane & 7;
    const int h = lane >> 3;
    const double R2 = 0.70710678118654752440;
    /* W8^j = exp(-2 pi i j / 8), W4^j = exp(-2 pi i j / 4) */
    const double2 ONE = make_double2(1.0, 0.0);
    c.W8j = (h == 0) ? ONE
          : (h == 1) ? make_double2(R2, -R2)
          : (h == 2) ? make_double2(0.0, -1.0)
                     : make_double2(-R2, -R2);                     /* W8^h        */
    const double2 W8w = ((w & 3) == 0) ? ONE
                      : ((w & 3) == 1) ? make_double2(R2, -R2)
                      : ((w & 3) == 2) ? make_double2(0.0, -1.0)
                                       : make_double2(-R2, -R2);   /* W8^(w&3)    */
    const double2 MI = make_double2(0.0, -1.0);
    c.tY2 = (lane & 16) ? ((h & 1) ? MI : ONE) : ONE;              /* W4^h0 or 1  */
    c.tZ1 = (lane & 4) ? W8w : ONE;
    c.tZ2 = (lane & 2) ? ((lane & 1) ? MI : ONE) : ONE;            /* W4^w0 or 1  */
    c.sY2 = (lane & 16) ? -1.0 : 1.0;
    c.sY3 = (lane & 8) ? -1.0 : 1.0;
    c.sZ1 = (lane & 4) ? -1.0 : 1.0;
    c.sZ2 = (lane & 2) ? -1.0 : 1.0;
    c.sZ3 = (lane & 1) ? -1.0 : 1.0;
    c.H  = ((h & 1) << 5) | ((h >> 1) << 4);                /* 8*(4*h0 + 2*h1)    */
    c.Wp = ((w & 1) << 2) | (((w >> 1) & 1) << 1) | (w >> 2);
}

static __device__ __forceinline__ void l8_load(double2 (&v)[16], const double2 *gi)
{
#pragma unroll
    for (int r = 0; r < 16; ++r) v[r] = gi[r * 32];
    /* v[2i+a] = (x=i, y=4a+h, z=w) */
}

static __device__ __forceinline__ void l8_compute(double2 (&v)[16], const l8c &c)
{
    const double2 ONE = make_double2(1.0, 0.0);
    /* pass X: two full 8-point lines in registers */
    dft8(v[0], v[2], v[4], v[6], v[8], v[10], v[12], v[14]);
    dft8(v[1], v[3], v[5], v[7], v[9], v[11], v[13], v[15]);
    /* v[2k+a] = (kx=k, y=4a+h, z=w) */

    /* pass Y, DIF: local stage on the a bit (both outputs stay in-lane) ... */
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const double2 lo = v[2 * k], hi = v[2 * k + 1];
        v[2 * k]     = cadd(lo, hi);
        v[2 * k + 1] = cmul(csub(lo, hi), c.W8j);
    }
    /* ... then two shuffle stages over the h bits */
    xstage<16, true>(v, c.sY2, c.tY2);
    xstage<8, false>(v, c.sY3, ONE);
    /* lane bits now hold ky1 = h1, ky2 = h0; register bit a = ky0 */

    /* pass Z, DIF: three shuffle stages over the w bits */
    xstage<4, true>(v, c.sZ1, c.tZ1);
    xstage<2, true>(v, c.sZ2, c.tZ2);
    xstage<1, false>(v, c.sZ3, ONE);
    /* lane bits hold kz0 = w2, kz1 = w1, kz2 = w0 */
}

template <bool STCS>
static __device__ __forceinline__ void l8_store(double2 (&v)[16], double2 *gs)
{
    /* gs = out + vol*512 + H + Wp: the lane part of the address absorbs both
     * DIF bit-reversals; a warp's 32 addresses are the same set whatever the
     * lane assignment, so the 16 stores stay 4x128B-dense. */
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        /* v[2q+a] = (kx=q, ky0=a): element offset 64q + 8a */
        const int off = 64 * (r >> 1) + 8 * (r & 1);
        if (STCS) __stcs(gs + off, v[r]);
        else      gs[off] = v[r];
    }
}

template <int WPB, bool STCS>
__global__ void __launch_bounds__(WPB * 32, 16 / WPB)
fft8_warp_x(const double2 *__restrict__ in, double2 *__restrict__ out, int B)
{
    const int  lane = threadIdx.x & 31;
    const long vol  = (long)blockIdx.x * WPB + (threadIdx.x >> 5);
    if (vol >= (long)B) return;
    l8c c;
    l8_setup(lane, c);
    double2 v[16];
    l8_load(v, in + vol * 512 + lane);
    l8_compute(v, c);
    l8_store<STCS>(v, out + vol * 512 + c.H + c.Wp);
}

/* Double-buffered pair kernel: each warp owns TWO consecutive volumes; the second
 * volume's 16 loads are issued before the first volume touches its data, so one
 * volume's ~800-cycle compute hides the other's global latency inside a single
 * warp instead of relying on occupancy (the batched points run at ~1 wave, where
 * ncu shows 12 active warps/SM and 13.5 cycles per issued instruction). */
template <int WPB, bool STCS>
__global__ void __launch_bounds__(WPB * 32, 8 / (WPB > 8 ? 8 : WPB))
fft8_warp_x2(const double2 *__restrict__ in, double2 *__restrict__ out, int B)
{
    const int  lane = threadIdx.x & 31;
    const long vol  = ((long)blockIdx.x * WPB + (threadIdx.x >> 5)) * 2;
    if (vol >= (long)B) return;
    const bool two = vol + 1 < (long)B;
    l8c c;
    l8_setup(lane, c);
    double2 va[16], vb[16];
    l8_load(va, in + vol * 512 + lane);
    if (two) l8_load(vb, in + (vol + 1) * 512 + lane);
    l8_compute(va, c);
    l8_store<STCS>(va, out + vol * 512 + c.H + c.Wp);
    if (two) {
        l8_compute(vb, c);
        l8_store<STCS>(vb, out + (vol + 1) * 512 + c.H + c.Wp);
    }
}

/* B = 1 latency kernel: 64 threads = 2 warps so two schedulers share the serial
 * chain (a single warp leaves 3 of 4 idle). Each lane owns one full x-line
 * (8 complex); y is three cross-lane DIF stages, z is one cross-WARP stage --
 * the only shared-memory touch in this entry, an 8 KB padded bounce with ONE
 * barrier (idea of staging through shared from L8_blockfused's B=1 kernel, cut
 * from two barriers to one because only the z2 bit crosses warps) -- followed
 * by two in-warp stages. Store order absorbs all bit reversals. (A barrier-free
 * variant that replaced the bounce with redundant partner loads measured 4.65 us
 * against this kernel's 3.90 -- the 8 extra load instructions cost more than the
 * bounce; see the strategy record.) */
__global__ void __launch_bounds__(64)
fft8_b1(const double2 *__restrict__ in, double2 *__restrict__ out)
{
    __shared__ double2 s[64 * 9]; /* stride 9: odd => conflict-free 16B phases */
    const int t    = threadIdx.x;
    const int j    = t >> 5;      /* warp = z bit2 */
    const int lane = t & 31;
    const int y    = lane >> 2;
    const int zl   = lane & 3;

    const double R2 = 0.70710678118654752440;
    const double2 ONE = make_double2(1.0, 0.0);
    const double2 MI  = make_double2(0.0, -1.0);
    const double2 W8y = ((y & 3) == 0) ? ONE
                      : ((y & 3) == 1) ? make_double2(R2, -R2)
                      : ((y & 3) == 2) ? MI
                                       : make_double2(-R2, -R2);
    const double2 W8z = (zl == 0) ? ONE
                      : (zl == 1) ? make_double2(R2, -R2)
                      : (zl == 2) ? MI
                                  : make_double2(-R2, -R2);
    const double2 tY1 = (lane & 16) ? W8y : ONE;                 /* W8^(y&3)  */
    const double2 tY2 = (lane & 8) ? ((y & 1) ? MI : ONE) : ONE; /* W4^y0     */
    const double2 tZ1 = j ? W8z : ONE;                           /* W8^zl     */
    const double2 tZ2 = (lane & 2) ? ((lane & 1) ? MI : ONE) : ONE;
    const double sY1 = (lane & 16) ? -1.0 : 1.0;
    const double sY2 = (lane & 8) ? -1.0 : 1.0;
    const double sY3 = (lane & 4) ? -1.0 : 1.0;
    const double sZ1 = j ? -1.0 : 1.0;
    const double sZ2 = (lane & 2) ? -1.0 : 1.0;
    const double sZ3 = (lane & 1) ? -1.0 : 1.0;

    double2 v[8];
#pragma unroll
    for (int x = 0; x < 8; ++x) v[x] = in[64 * x + 8 * y + 4 * j + zl];

    dft8(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]); /* x-pass, kx = reg */

    /* y-pass: DIF over lane bits 16, 8, 4 */
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        double2 rec;
        rec.x = __shfl_xor_sync(0xffffffffu, v[r].x, 16);
        rec.y = __shfl_xor_sync(0xffffffffu, v[r].y, 16);
        v[r] = cmul(make_double2(fma(sY1, v[r].x, rec.x), fma(sY1, v[r].y, rec.y)), tY1);
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        double2 rec;
        rec.x = __shfl_xor_sync(0xffffffffu, v[r].x, 8);
        rec.y = __shfl_xor_sync(0xffffffffu, v[r].y, 8);
        v[r] = cmul(make_double2(fma(sY2, v[r].x, rec.x), fma(sY2, v[r].y, rec.y)), tY2);
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        double2 rec;
        rec.x = __shfl_xor_sync(0xffffffffu, v[r].x, 4);
        rec.y = __shfl_xor_sync(0xffffffffu, v[r].y, 4);
        v[r] = make_double2(fma(sY3, v[r].x, rec.x), fma(sY3, v[r].y, rec.y));
    }
    /* ky = 4*((lane>>2)&1) + 2*((lane>>3)&1) + ((lane>>4)&1) */

    /* z-pass stage 1 (bit z2 = warp bit): one shared bounce, one barrier */
#pragma unroll
    for (int r = 0; r < 8; ++r) s[t * 9 + r] = v[r];
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        const double2 rec = s[(t ^ 32) * 9 + r];
        v[r] = cmul(make_double2(fma(sZ1, v[r].x, rec.x), fma(sZ1, v[r].y, rec.y)), tZ1);
    }
    /* stages 2-3 in-warp over lane bits 2, 1 */
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        double2 rec;
        rec.x = __shfl_xor_sync(0xffffffffu, v[r].x, 2);
        rec.y = __shfl_xor_sync(0xffffffffu, v[r].y, 2);
        v[r] = cmul(make_double2(fma(sZ2, v[r].x, rec.x), fma(sZ2, v[r].y, rec.y)), tZ2);
    }
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        double2 rec;
        rec.x = __shfl_xor_sync(0xffffffffu, v[r].x, 1);
        rec.y = __shfl_xor_sync(0xffffffffu, v[r].y, 1);
        v[r] = make_double2(fma(sZ3, v[r].x, rec.x), fma(sZ3, v[r].y, rec.y));
    }
    /* kz = 4*(lane&1) + 2*((lane>>1)&1) + j */

    const int ky = 4 * ((lane >> 2) & 1) + 2 * ((lane >> 3) & 1) + ((lane >> 4) & 1);
    const int kz = 4 * (lane & 1) + 2 * ((lane >> 1) & 1) + j;
    double2 *gs = out + 8 * ky + kz;
#pragma unroll
    for (int r = 0; r < 8; ++r) gs[64 * r] = v[r];
}

/* ---- host side ------------------------------------------------------------------- */

struct fft3d_gpu_plan {
    int L, batch;
    int wpb;    /* warps (= volumes) per block: 1,2,4,8,16 */
    int var;    /* 0 = butterfly-transpose, line stores; 1 = + transpose-back, flat
                   stores; 2 = cross-lane DIF stages, no transposes; 3 = cross-lane
                   double-buffered pair; -1 = dedicated B=1 two-warp kernel */
    int stcs;   /* 1 = evict-first streaming stores */
    int grid;
};

typedef void (*kfun_t)(const double2 *, double2 *, int);

static kfun_t kern_for(int wpb, int var, int stcs)
{
    switch (wpb * 8 + var * 2 + stcs) {
    case 1 * 8 + 6: return fft8_warp_x2<1, false>;
    case 1 * 8 + 7: return fft8_warp_x2<1, true>;
    case 2 * 8 + 6: return fft8_warp_x2<2, false>;
    case 2 * 8 + 7: return fft8_warp_x2<2, true>;
    case 4 * 8 + 6: return fft8_warp_x2<4, false>;
    case 4 * 8 + 7: return fft8_warp_x2<4, true>;
    case 8 * 8 + 6: return fft8_warp_x2<8, false>;
    case 8 * 8 + 7: return fft8_warp_x2<8, true>;
    case 16 * 8 + 6: return fft8_warp_x2<16, false>;
    case 16 * 8 + 7: return fft8_warp_x2<16, true>;
    case 1 * 8 + 0: return fft8_warp<1, false, false>;
    case 1 * 8 + 1: return fft8_warp<1, false, true>;
    case 1 * 8 + 2: return fft8_warp<1, true, false>;
    case 1 * 8 + 3: return fft8_warp<1, true, true>;
    case 1 * 8 + 4: return fft8_warp_x<1, false>;
    case 1 * 8 + 5: return fft8_warp_x<1, true>;
    case 2 * 8 + 0: return fft8_warp<2, false, false>;
    case 2 * 8 + 1: return fft8_warp<2, false, true>;
    case 2 * 8 + 2: return fft8_warp<2, true, false>;
    case 2 * 8 + 3: return fft8_warp<2, true, true>;
    case 2 * 8 + 4: return fft8_warp_x<2, false>;
    case 2 * 8 + 5: return fft8_warp_x<2, true>;
    case 4 * 8 + 0: return fft8_warp<4, false, false>;
    case 4 * 8 + 1: return fft8_warp<4, false, true>;
    case 4 * 8 + 2: return fft8_warp<4, true, false>;
    case 4 * 8 + 3: return fft8_warp<4, true, true>;
    case 4 * 8 + 4: return fft8_warp_x<4, false>;
    case 4 * 8 + 5: return fft8_warp_x<4, true>;
    case 8 * 8 + 0: return fft8_warp<8, false, false>;
    case 8 * 8 + 1: return fft8_warp<8, false, true>;
    case 8 * 8 + 2: return fft8_warp<8, true, false>;
    case 8 * 8 + 3: return fft8_warp<8, true, true>;
    case 8 * 8 + 4: return fft8_warp_x<8, false>;
    case 8 * 8 + 5: return fft8_warp_x<8, true>;
    case 16 * 8 + 0: return fft8_warp<16, false, false>;
    case 16 * 8 + 1: return fft8_warp<16, false, true>;
    case 16 * 8 + 2: return fft8_warp<16, true, false>;
    case 16 * 8 + 3: return fft8_warp<16, true, true>;
    case 16 * 8 + 4: return fft8_warp_x<16, false>;
    case 16 * 8 + 5: return fft8_warp_x<16, true>;
    }
    return 0;
}

extern "C" const char *fft3d_gpu_name(void) { return "L8_warpradix8"; }
extern "C" const char *fft3d_gpu_description(void)
{ return "one volume per warp in registers, radix-8 x3, butterfly shuffle transposes, no shared/barriers"; }
extern "C" int fft3d_gpu_supports(int L) { return L == 8; }

/* Measured candidate selection on scratch buffers -- pattern adopted from
 * L8_blockfused round gpu_r1. */
static float time_candidate(kfun_t k, int grid, int block,
                            const double2 *a, double2 *b, int B, int reps)
{
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    k<<<grid, block>>>(a, b, B); /* warm */
    cudaEventRecord(e0);
    for (int r = 0; r < reps; ++r) k<<<grid, block>>>(a, b, B);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 1e30f;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms;
}

extern "C" fft3d_gpu_plan *fft3d_gpu_create(int L, int batch)
{
    if (L != 8 || batch < 1) return NULL;

    fft3d_gpu_plan *p = (fft3d_gpu_plan *)malloc(sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    p->wpb = 4;
    p->var = 2;
    p->stcs = 0;

    static const int WS[5] = {1, 2, 4, 8, 16};

    const char *fw = getenv("L8WR_WPB");
    const char *ff = getenv("L8WR_VAR");
    const char *fs = getenv("L8WR_STCS");
    if (fw || ff || fs) {
        if (fw) p->wpb = atoi(fw);
        if (ff) p->var = atoi(ff);
        if (fs) p->stcs = atoi(fs);
        if (batch == 1 && !fw) p->wpb = 1;
    } else if (batch == 1) {
        p->wpb = 1;
        p->var = -1; /* dedicated two-warp latency kernel */
        p->stcs = 0;
    } else {
        double2 *a = 0, *b = 0;
        size_t bytes = (size_t)batch * 512 * sizeof(double2);
        if (cudaMalloc(&a, bytes) == cudaSuccess && cudaMalloc(&b, bytes) == cudaSuccess) {
            cudaMemset(a, 0, bytes);
            /* Tuning space is vars 2 and 3 ONLY: they share l8_compute exactly, so
             * every candidate is bit-identical and the plan choice cannot break
             * run-to-run repeatability. Vars 0/1 (dft8+transposes) round their z
             * pass differently and are kept for env-forced experiments only. */
            int reps = (int)(60000000L / ((long)batch * 512) + 3);
            if (reps > 400) reps = 400;
            float best[20];
            for (int i = 0; i < 20; ++i) best[i] = 1e30f;
            for (int cyc = 0; cyc < 3; ++cyc)
                for (int i = 0; i < 20; ++i) {
                    const int wpb = WS[i / 4], vr = 2 + ((i % 4) >> 1), st = i & 1;
                    const int vpb = wpb * (vr == 3 ? 2 : 1);
                    const int grid = (batch + vpb - 1) / vpb;
                    float ms = time_candidate(kern_for(wpb, vr, st), grid, wpb * 32,
                                              a, b, batch, reps);
                    if (ms < best[i]) best[i] = ms;
                }
            int ibest = 0;
            for (int i = 1; i < 20; ++i)
                if (best[i] < best[ibest]) ibest = i;
            p->wpb = WS[ibest / 4];
            p->var = 2 + ((ibest % 4) >> 1);
            p->stcs = ibest & 1;
        }
        if (a) cudaFree(a);
        if (b) cudaFree(b);
        cudaGetLastError(); /* clear any autotune-time error state */
    }

    const int vpb = p->wpb * (p->var == 3 ? 2 : 1);
    p->grid = (batch + vpb - 1) / vpb;
    return p;
}

extern "C" void fft3d_gpu_execute(fft3d_gpu_plan *p, const double2 *in, double2 *out)
{
    if (p->var < 0) {
        fft8_b1<<<1, 64>>>(in, out);
        return;
    }
    kern_for(p->wpb, p->var, p->stcs)<<<p->grid, p->wpb * 32>>>(in, out, p->batch);
}

extern "C" void fft3d_gpu_destroy(fft3d_gpu_plan *p) { free(p); }
