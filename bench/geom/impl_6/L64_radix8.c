/* L64_radix8 -- forward complex-double 3D DFT of a 64^3 cube, batched.
 *
 * Technique: 64 = 8*8, so every axis is two radix-8 passes with w64 twiddles in
 * between (Cooley-Tukey N1=N2=8).  The radix-8 codelet is the panel's proven
 * 52-instruction / 56-flop module (borrowed from L8_radix8/L8_batchsimd): the only
 * irrational constant is 1/sqrt(2), every +-i is free in split-complex layout.
 *
 * Pass structure (split-complex scratch volume SC, lanes always hold 8 adjacent z):
 *   pass 1 (y-FFT): per (x, z-octet): deinterleave input rows, 64-pt FFT across
 *           vectors (fully elementwise, zero shuffles in the butterflies), -> SC.
 *   pass 2 (x-FFT): per (ky, z-octet): same elementwise 64-pt FFT over x, in place
 *           in SC through an 8 KiB L1 line buffer.
 *   pass 3 (z-FFT): per (kx,ky) row: z lives in the lanes, so: radix-8 across the
 *           8 z-octet vectors (registers), lane twiddle w64^(l*k2) from a vector
 *           table, one 8x8 transpose pair (24+24 non-destructive shuffles, the
 *           L8_fusedaxes network with its SW lane residue absorbed into the final
 *           interleave index vectors), radix-8 over lanes-now-registers, interleave,
 *           and a fully SEQUENTIAL store of the whole output volume (NT option).
 *
 * SC strides are padded to an ODD number of cache lines (Bailey / corpus section 04):
 * at L=64 the natural x-stride is 64 KiB which maps 8-deep into a single L1 set.
 * KY stride = 136 doubles (17 lines), x stride = 64*136+8 = 8712 doubles (1089 lines).
 *
 * fft3d_create() self-tunes the pass-3 store type (plain vs streaming) by timing
 * both at (a clamp of) the real batch size, L8-style: the machine decides, not me.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>

#include "../fft3d_api.h"

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

#define VOLC   (64 * 64 * 64)        /* complex points per volume */
#define VOLD   (2 * VOLC)            /* doubles per volume */

/* Scratch strides, in doubles.  Slot (x, ky, zb) = x*SCXS + ky*SCKS + zb*16;
 * re vector at +0, im at +8.  Both strides are an odd number of 64-B lines. */
#ifndef SCKS
#define SCKS 136                     /* 17 cache lines between ky rows   */
#endif
#ifndef SCXPAD
#define SCXPAD 8                     /* +1 line so x stride is odd too   */
#endif
#define SCXS (64 * SCKS + SCXPAD)    /* 8712 doubles = 1089 cache lines  */
#define SCSZ ((size_t)64 * SCXS * sizeof(double))

struct fft3d_plan {
    int L, B;
    int nt;                          /* pass-3 stores: 0 plain, 1 non-temporal */
    double *sc;                      /* split-complex scratch volume           */
    double *lb;                      /* 8 KiB line buffer (two radix-8 stages) */
    int sc_is_mmap;
    double tw1[8][8][2];             /* w64^(j1*k2), scalar broadcasts         */
    double tw3r[8][8] __attribute__((aligned(64)));   /* pass-3 lane twiddles  */
    double tw3i[8][8] __attribute__((aligned(64)));   /* [k2][lane l]          */
} __attribute__((aligned(64)));

static char g_desc[160] =
    "radix-8^2 per axis, split-complex AVX-512, padded scratch, seq NT out";

const char *fft3d_name(void) { return "L64_radix8"; }
const char *fft3d_description(void) { return g_desc; }
int fft3d_supports(int L) { return L == 64; }

/* ------------------------------------------------------------------------ */
/* The radix-8 codelet: natural-order in, natural-order out, forward DFT_8.
 * 44 add/sub + 8 FMA = 52 ops, only constant is C = 1/sqrt(2).
 * Generic over VT/VADD/VSUB/VFMA/VFNMA, so it instantiates for __m512d and for
 * plain double below (macro bodies resolve at expansion time).               */
#define RADIX8(C, r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7) do {      \
    VT b0r=VADD(r0,r4), b1r=VSUB(r0,r4), b0i=VADD(i0,i4), b1i=VSUB(i0,i4);    \
    VT b2r=VADD(r2,r6), b3r=VSUB(r2,r6), b2i=VADD(i2,i6), b3i=VSUB(i2,i6);    \
    VT b4r=VADD(r1,r5), b5r=VSUB(r1,r5), b4i=VADD(i1,i5), b5i=VSUB(i1,i5);    \
    VT b6r=VADD(r3,r7), b7r=VSUB(r3,r7), b6i=VADD(i3,i7), b7i=VSUB(i3,i7);    \
    VT c0r=VADD(b0r,b2r), c2r=VSUB(b0r,b2r), c0i=VADD(b0i,b2i), c2i=VSUB(b0i,b2i); \
    VT c1r=VADD(b1r,b3i), c3r=VSUB(b1r,b3i), c1i=VSUB(b1i,b3r), c3i=VADD(b1i,b3r); \
    VT c4r=VADD(b4r,b6r), c6r=VSUB(b4r,b6r), c4i=VADD(b4i,b6i), c6i=VSUB(b4i,b6i); \
    VT c5r=VADD(b5r,b7i), c7r=VSUB(b5r,b7i), c5i=VSUB(b5i,b7r), c7i=VADD(b5i,b7r); \
    VT s5=VADD(c5r,c5i), t5=VSUB(c5i,c5r);                                    \
    VT u7=VSUB(c7i,c7r), v7=VADD(c7r,c7i);                                    \
    r0=VADD(c0r,c4r); r4=VSUB(c0r,c4r); i0=VADD(c0i,c4i); i4=VSUB(c0i,c4i);   \
    r2=VADD(c2r,c6i); r6=VSUB(c2r,c6i); i2=VSUB(c2i,c6r); i6=VADD(c2i,c6r);   \
    r1=VFMA(C,s5,c1r); r5=VFNMA(C,s5,c1r); i1=VFMA(C,t5,c1i); i5=VFNMA(C,t5,c1i); \
    r3=VFMA(C,u7,c3r); r7=VFNMA(C,u7,c3r); i3=VFNMA(C,v7,c3i); i7=VFMA(C,v7,c3i); \
} while (0)

/* twiddle: (r,i) *= (wr + i*wi); 2 mul + 2 fma */
#define CTW(r, i, wr, wi) do {                                                \
    VT _pr = VMUL(r, wr), _pi = VMUL(r, wi);                                  \
    r = VFNMA(i, wi, _pr); i = VFMA(i, wr, _pi);                              \
} while (0)

/* ======================================================================== */
/*                             AVX-512 kernels                              */
/* ======================================================================== */
#if defined(__AVX512F__)
#include <immintrin.h>

#define VT __m512d
#define VADD  _mm512_add_pd
#define VSUB  _mm512_sub_pd
#define VMUL  _mm512_mul_pd
#define VFMA  _mm512_fmadd_pd
#define VFNMA _mm512_fnmadd_pd
#define VLD   _mm512_load_pd
#define VST   _mm512_store_pd

/* 8x8 double transpose, 24 non-destructive shuffles (L8_fusedaxes network).
 * Output residue: o[r][lane m] = in[SW(m)][r], SW = swap of lane bits 1,2. */
#define TR8(v0,v1,v2,v3,v4,v5,v6,v7) do {                                     \
    VT _t0=_mm512_shuffle_f64x2(v0,v4,0x44), _t4=_mm512_shuffle_f64x2(v0,v4,0xEE); \
    VT _t1=_mm512_shuffle_f64x2(v1,v5,0x44), _t5=_mm512_shuffle_f64x2(v1,v5,0xEE); \
    VT _t2=_mm512_shuffle_f64x2(v2,v6,0x44), _t6=_mm512_shuffle_f64x2(v2,v6,0xEE); \
    VT _t3=_mm512_shuffle_f64x2(v3,v7,0x44), _t7=_mm512_shuffle_f64x2(v3,v7,0xEE); \
    VT _u0=_mm512_shuffle_f64x2(_t0,_t2,0x88), _u2=_mm512_shuffle_f64x2(_t0,_t2,0xDD); \
    VT _u1=_mm512_shuffle_f64x2(_t1,_t3,0x88), _u3=_mm512_shuffle_f64x2(_t1,_t3,0xDD); \
    VT _u4=_mm512_shuffle_f64x2(_t4,_t6,0x88), _u6=_mm512_shuffle_f64x2(_t4,_t6,0xDD); \
    VT _u5=_mm512_shuffle_f64x2(_t5,_t7,0x88), _u7=_mm512_shuffle_f64x2(_t5,_t7,0xDD); \
    v0=_mm512_unpacklo_pd(_u0,_u1); v1=_mm512_unpackhi_pd(_u0,_u1);           \
    v2=_mm512_unpacklo_pd(_u2,_u3); v3=_mm512_unpackhi_pd(_u2,_u3);           \
    v4=_mm512_unpacklo_pd(_u4,_u5); v5=_mm512_unpackhi_pd(_u4,_u5);           \
    v6=_mm512_unpacklo_pd(_u6,_u7); v7=_mm512_unpackhi_pd(_u6,_u7);           \
} while (0)

/* pass 1: y-FFT.  in: interleaved input plane x; out: SC (split, lanes = z).
 * Deinterleave is fused into the stage-1 loads (one permutex2var per vector,
 * measured free next to the FP work).  PF=1 variant additionally prefetches
 * the NEXT x-plane into L2, spread 16 lines per stage-1 iteration -- used
 * when the batch streams from DRAM (measured ~5% on wallaby; a plane-buffer
 * sequential-copy variant was measured SLOWER and was removed).             */
#define PASS1_DEF(NAME, PF)                                                   \
static void NAME(const struct fft3d_plan *p, const double *in, double *sc)    \
{                                                                             \
    const __m512i IEV = _mm512_setr_epi64(0,2,4,6,8,10,12,14);                \
    const __m512i IOD = _mm512_setr_epi64(1,3,5,7,9,11,13,15);                \
    const VT C = _mm512_set1_pd(M_SQRT1_2);                                   \
    double *lb = p->lb;                                                       \
    for (int x = 0; x < 64; x++) {                                            \
        const double *px = in + (size_t)x * 8192;                             \
        const double *pnext = px + 8192;                                      \
        double *scx = sc + (size_t)x * SCXS;                                  \
        for (int zb = 0; zb < 8; zb++) {                                      \
            const double *pz = px + zb * 16;                                  \
            /* stage 1: for each residue j1, DFT_8 over y = j1 + 8t */        \
            for (int j1 = 0; j1 < 8; j1++) {                                  \
                if (PF) {                                                     \
                    const char *pf = (const char *)(pnext + (zb*8 + j1)*128); \
                    _mm_prefetch(pf,       _MM_HINT_T1);                      \
                    _mm_prefetch(pf +  64, _MM_HINT_T1);                      \
                    _mm_prefetch(pf + 128, _MM_HINT_T1);                      \
                    _mm_prefetch(pf + 192, _MM_HINT_T1);                      \
                    _mm_prefetch(pf + 256, _MM_HINT_T1);                      \
                    _mm_prefetch(pf + 320, _MM_HINT_T1);                      \
                    _mm_prefetch(pf + 384, _MM_HINT_T1);                      \
                    _mm_prefetch(pf + 448, _MM_HINT_T1);                      \
                    _mm_prefetch(pf + 512, _MM_HINT_T1);                      \
                    _mm_prefetch(pf + 576, _MM_HINT_T1);                      \
                    _mm_prefetch(pf + 640, _MM_HINT_T1);                      \
                    _mm_prefetch(pf + 704, _MM_HINT_T1);                      \
                    _mm_prefetch(pf + 768, _MM_HINT_T1);                      \
                    _mm_prefetch(pf + 832, _MM_HINT_T1);                      \
                    _mm_prefetch(pf + 896, _MM_HINT_T1);                      \
                    _mm_prefetch(pf + 960, _MM_HINT_T1);                      \
                }                                                             \
                const double *py = pz + j1 * 128;                             \
                VT r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7;          \
                {                                                             \
                    VT a0=VLD(py+0*1024), b0=VLD(py+0*1024+8);                \
                    VT a1=VLD(py+1*1024), b1=VLD(py+1*1024+8);                \
                    VT a2=VLD(py+2*1024), b2=VLD(py+2*1024+8);                \
                    VT a3=VLD(py+3*1024), b3=VLD(py+3*1024+8);                \
                    VT a4=VLD(py+4*1024), b4=VLD(py+4*1024+8);                \
                    VT a5=VLD(py+5*1024), b5=VLD(py+5*1024+8);                \
                    VT a6=VLD(py+6*1024), b6=VLD(py+6*1024+8);                \
                    VT a7=VLD(py+7*1024), b7=VLD(py+7*1024+8);                \
                    r0=_mm512_permutex2var_pd(a0,IEV,b0); i0=_mm512_permutex2var_pd(a0,IOD,b0); \
                    r1=_mm512_permutex2var_pd(a1,IEV,b1); i1=_mm512_permutex2var_pd(a1,IOD,b1); \
                    r2=_mm512_permutex2var_pd(a2,IEV,b2); i2=_mm512_permutex2var_pd(a2,IOD,b2); \
                    r3=_mm512_permutex2var_pd(a3,IEV,b3); i3=_mm512_permutex2var_pd(a3,IOD,b3); \
                    r4=_mm512_permutex2var_pd(a4,IEV,b4); i4=_mm512_permutex2var_pd(a4,IOD,b4); \
                    r5=_mm512_permutex2var_pd(a5,IEV,b5); i5=_mm512_permutex2var_pd(a5,IOD,b5); \
                    r6=_mm512_permutex2var_pd(a6,IEV,b6); i6=_mm512_permutex2var_pd(a6,IOD,b6); \
                    r7=_mm512_permutex2var_pd(a7,IEV,b7); i7=_mm512_permutex2var_pd(a7,IOD,b7); \
                }                                                             \
                RADIX8(C, r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7);  \
                if (j1) {                                                     \
                    const double (*tw)[2] = p->tw1[j1];                       \
                    CTW(r1,i1,_mm512_set1_pd(tw[1][0]),_mm512_set1_pd(tw[1][1])); \
                    CTW(r2,i2,_mm512_set1_pd(tw[2][0]),_mm512_set1_pd(tw[2][1])); \
                    CTW(r3,i3,_mm512_set1_pd(tw[3][0]),_mm512_set1_pd(tw[3][1])); \
                    CTW(r4,i4,_mm512_set1_pd(tw[4][0]),_mm512_set1_pd(tw[4][1])); \
                    CTW(r5,i5,_mm512_set1_pd(tw[5][0]),_mm512_set1_pd(tw[5][1])); \
                    CTW(r6,i6,_mm512_set1_pd(tw[6][0]),_mm512_set1_pd(tw[6][1])); \
                    CTW(r7,i7,_mm512_set1_pd(tw[7][0]),_mm512_set1_pd(tw[7][1])); \
                }                                                             \
                VST(lb+(0*8+j1)*16,r0); VST(lb+(0*8+j1)*16+8,i0);             \
                VST(lb+(1*8+j1)*16,r1); VST(lb+(1*8+j1)*16+8,i1);             \
                VST(lb+(2*8+j1)*16,r2); VST(lb+(2*8+j1)*16+8,i2);             \
                VST(lb+(3*8+j1)*16,r3); VST(lb+(3*8+j1)*16+8,i3);             \
                VST(lb+(4*8+j1)*16,r4); VST(lb+(4*8+j1)*16+8,i4);             \
                VST(lb+(5*8+j1)*16,r5); VST(lb+(5*8+j1)*16+8,i5);             \
                VST(lb+(6*8+j1)*16,r6); VST(lb+(6*8+j1)*16+8,i6);             \
                VST(lb+(7*8+j1)*16,r7); VST(lb+(7*8+j1)*16+8,i7);             \
            }                                                                 \
            /* stage 2: for each k2, DFT_8 over j1; ky = k2 + 8*k1 */         \
            double *sz = scx + zb * 16;                                       \
            for (int k2 = 0; k2 < 8; k2++) {                                  \
                const double *lk = lb + k2 * 128;                             \
                VT r0=VLD(lk+0),   i0=VLD(lk+8);                              \
                VT r1=VLD(lk+16),  i1=VLD(lk+24);                             \
                VT r2=VLD(lk+32),  i2=VLD(lk+40);                             \
                VT r3=VLD(lk+48),  i3=VLD(lk+56);                             \
                VT r4=VLD(lk+64),  i4=VLD(lk+72);                             \
                VT r5=VLD(lk+80),  i5=VLD(lk+88);                             \
                VT r6=VLD(lk+96),  i6=VLD(lk+104);                            \
                VT r7=VLD(lk+112), i7=VLD(lk+120);                            \
                RADIX8(C, r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7);  \
                double *so = sz + k2 * SCKS;                                  \
                VST(so+0*(8*SCKS),r0); VST(so+0*(8*SCKS)+8,i0);               \
                VST(so+1*(8*SCKS),r1); VST(so+1*(8*SCKS)+8,i1);               \
                VST(so+2*(8*SCKS),r2); VST(so+2*(8*SCKS)+8,i2);               \
                VST(so+3*(8*SCKS),r3); VST(so+3*(8*SCKS)+8,i3);               \
                VST(so+4*(8*SCKS),r4); VST(so+4*(8*SCKS)+8,i4);               \
                VST(so+5*(8*SCKS),r5); VST(so+5*(8*SCKS)+8,i5);               \
                VST(so+6*(8*SCKS),r6); VST(so+6*(8*SCKS)+8,i6);               \
                VST(so+7*(8*SCKS),r7); VST(so+7*(8*SCKS)+8,i7);               \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

PASS1_DEF(pass1_plain, 0)
PASS1_DEF(pass1_pf, 1)

/* x-line FFT for one (ky, zb) group, in place in SC (all inputs land in the
 * line buffer before any write-back; in-place beat a small slab buffer by
 * 3-4% in-process -- the stores hit just-read lines).  lanes = z, zero
 * shuffles.  The +16 prefetch covers the next zb column (removing it costs
 * ~12% at B=8).                                                              */
static inline void xline_avx512(const struct fft3d_plan *p, double *sz, double *lb)
{
    const VT C = _mm512_set1_pd(M_SQRT1_2);
    for (int j1 = 0; j1 < 8; j1++) {
        const double *bx = sz + (size_t)j1 * SCXS;
#define P2LD(t) VT r##t = VLD(bx + (size_t)(t)*(8*SCXS)),                     \
                   i##t = VLD(bx + (size_t)(t)*(8*SCXS) + 8);                 \
        _mm_prefetch((const char *)(bx + (size_t)(t)*(8*SCXS) + 16), _MM_HINT_T0);
        P2LD(0) P2LD(1) P2LD(2) P2LD(3) P2LD(4) P2LD(5) P2LD(6) P2LD(7)
#undef P2LD
        RADIX8(C, r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7);
        if (j1) {
            const double (*tw)[2] = p->tw1[j1];
#define TW(k) CTW(r##k, i##k, _mm512_set1_pd(tw[k][0]), _mm512_set1_pd(tw[k][1]))
            TW(1); TW(2); TW(3); TW(4); TW(5); TW(6); TW(7);
#undef TW
        }
#define LBST(k) VST(lb + ((k)*8 + j1)*16, r##k); VST(lb + ((k)*8 + j1)*16 + 8, i##k);
        LBST(0) LBST(1) LBST(2) LBST(3) LBST(4) LBST(5) LBST(6) LBST(7)
#undef LBST
    }
    for (int k2 = 0; k2 < 8; k2++) {
        const double *lk = lb + k2 * 128;
#define LBLD(t) VT r##t = VLD(lk + (t)*16), i##t = VLD(lk + (t)*16 + 8);
        LBLD(0) LBLD(1) LBLD(2) LBLD(3) LBLD(4) LBLD(5) LBLD(6) LBLD(7)
#undef LBLD
        RADIX8(C, r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7);
        double *so = sz + (size_t)k2 * SCXS;
#define SST(k1) VST(so + (size_t)(k1)*(8*SCXS), r##k1);                       \
                VST(so + (size_t)(k1)*(8*SCXS) + 8, i##k1);
        SST(0) SST(1) SST(2) SST(3) SST(4) SST(5) SST(6) SST(7)
#undef SST
    }
}

/* FUSED pass 2+3: for each ky, first the 8 x-line groups (x-FFT in place in
 * SC), then immediately the 64 z-lines of that ky-slab -- the slab (64 KiB)
 * is still L2-hot, which beats a third full-volume L3 sweep by ~10% both
 * regimes (measured in-process on wallaby, bitwise-identical output).
 * z-line: z = 8g + l: radix-8 over g (registers), lane twiddle w64^(l k2),
 * transpose (lanes -> registers, SW residue), radix-8 over l, interleave with
 * SW-composed index vectors, 1 KiB contiguous store per (kx,ky) row.  Output
 * rows go out ky-major (stride 64 KiB between consecutive rows).            */
#define PASS23_DEF(NAME, STORE)                                               \
static void NAME(const struct fft3d_plan *p, double *sc, double *out)         \
{                                                                             \
    const __m512i ILO = _mm512_setr_epi64(0,8,1,9,4,12,5,13);                 \
    const __m512i IHI = _mm512_setr_epi64(2,10,3,11,6,14,7,15);               \
    const VT C = _mm512_set1_pd(M_SQRT1_2);                                   \
    for (int ky = 0; ky < 64; ky++) {                                         \
        double *sy = sc + (size_t)ky * SCKS;                                  \
        for (int zb = 0; zb < 8; zb++)                                        \
            xline_avx512(p, sy + zb * 16, p->lb);                             \
        for (int kx = 0; kx < 64; kx++) {                                     \
            const double *s = sy + (size_t)kx * SCXS;                         \
            double *o = out + (size_t)kx * 8192 + ky * 128;                   \
            VT r0 = VLD(s),       i0 = VLD(s + 8);                            \
            VT r1 = VLD(s + 16),  i1 = VLD(s + 24);                           \
            VT r2 = VLD(s + 32),  i2 = VLD(s + 40);                           \
            VT r3 = VLD(s + 48),  i3 = VLD(s + 56);                           \
            VT r4 = VLD(s + 64),  i4 = VLD(s + 72);                           \
            VT r5 = VLD(s + 80),  i5 = VLD(s + 88);                           \
            VT r6 = VLD(s + 96),  i6 = VLD(s + 104);                          \
            VT r7 = VLD(s + 112), i7 = VLD(s + 120);                          \
            RADIX8(C, r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7);      \
            CTW(r1,i1,VLD(p->tw3r[1]),VLD(p->tw3i[1]));                       \
            CTW(r2,i2,VLD(p->tw3r[2]),VLD(p->tw3i[2]));                       \
            CTW(r3,i3,VLD(p->tw3r[3]),VLD(p->tw3i[3]));                       \
            CTW(r4,i4,VLD(p->tw3r[4]),VLD(p->tw3i[4]));                       \
            CTW(r5,i5,VLD(p->tw3r[5]),VLD(p->tw3i[5]));                       \
            CTW(r6,i6,VLD(p->tw3r[6]),VLD(p->tw3i[6]));                       \
            CTW(r7,i7,VLD(p->tw3r[7]),VLD(p->tw3i[7]));                       \
            TR8(r0,r1,r2,r3,r4,r5,r6,r7);                                     \
            TR8(i0,i1,i2,i3,i4,i5,i6,i7);                                     \
            RADIX8(C, r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7);      \
            STORE(o,       _mm512_permutex2var_pd(r0, ILO, i0));              \
            STORE(o + 8,   _mm512_permutex2var_pd(r0, IHI, i0));              \
            STORE(o + 16,  _mm512_permutex2var_pd(r1, ILO, i1));              \
            STORE(o + 24,  _mm512_permutex2var_pd(r1, IHI, i1));              \
            STORE(o + 32,  _mm512_permutex2var_pd(r2, ILO, i2));              \
            STORE(o + 40,  _mm512_permutex2var_pd(r2, IHI, i2));              \
            STORE(o + 48,  _mm512_permutex2var_pd(r3, ILO, i3));              \
            STORE(o + 56,  _mm512_permutex2var_pd(r3, IHI, i3));              \
            STORE(o + 64,  _mm512_permutex2var_pd(r4, ILO, i4));              \
            STORE(o + 72,  _mm512_permutex2var_pd(r4, IHI, i4));              \
            STORE(o + 80,  _mm512_permutex2var_pd(r5, ILO, i5));              \
            STORE(o + 88,  _mm512_permutex2var_pd(r5, IHI, i5));              \
            STORE(o + 96,  _mm512_permutex2var_pd(r6, ILO, i6));              \
            STORE(o + 104, _mm512_permutex2var_pd(r6, IHI, i6));              \
            STORE(o + 112, _mm512_permutex2var_pd(r7, ILO, i7));              \
            STORE(o + 120, _mm512_permutex2var_pd(r7, IHI, i7));              \
        }                                                                     \
    }                                                                         \
}

#define ST_PLAIN(a, v) _mm512_store_pd((a), (v))
#define ST_NT(a, v)    _mm512_stream_pd((a), (v))
PASS23_DEF(pass23_plain, ST_PLAIN)
PASS23_DEF(pass23_nt, ST_NT)
#undef ST_PLAIN
#undef ST_NT

static void exec_avx512(const struct fft3d_plan *p, const double *in, double *out,
                        int nvol, int nt)
{
    for (int b = 0; b < nvol; b++) {
        const double *vi = in + (size_t)b * VOLD;
        double *vo = out + (size_t)b * VOLD;
        if (nvol > 1) pass1_pf(p, vi, p->sc);
        else          pass1_plain(p, vi, p->sc);
        if (nt) pass23_nt(p, p->sc, vo);
        else    pass23_plain(p, p->sc, vo);
    }
    if (nt) _mm_sfence();
}

#undef VT
#undef VADD
#undef VSUB
#undef VMUL
#undef VFMA
#undef VFNMA
#undef VLD
#undef VST
#endif /* __AVX512F__ */

/* ======================================================================== */
/*        Portable scalar fallback (same algorithm, same twiddles)          */
/* ======================================================================== */
#define VT double
#define VADD(a,b)    ((a) + (b))
#define VSUB(a,b)    ((a) - (b))
#define VMUL(a,b)    ((a) * (b))
#define VFMA(a,b,c)  ((a) * (b) + (c))
#define VFNMA(a,b,c) ((c) - (a) * (b))

static void line64_scalar(const double tw1[8][8][2], double *ar, double *ai)
{
    double br[64], bi[64];
    for (int j1 = 0; j1 < 8; j1++) {
        double r0=ar[j1],    r1=ar[j1+8],  r2=ar[j1+16], r3=ar[j1+24],
               r4=ar[j1+32], r5=ar[j1+40], r6=ar[j1+48], r7=ar[j1+56];
        double i0=ai[j1],    i1=ai[j1+8],  i2=ai[j1+16], i3=ai[j1+24],
               i4=ai[j1+32], i5=ai[j1+40], i6=ai[j1+48], i7=ai[j1+56];
        RADIX8(M_SQRT1_2, r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7);
        if (j1) {
            const double (*tw)[2] = tw1[j1];
#define TW(k) CTW(r##k, i##k, tw[k][0], tw[k][1])
            TW(1); TW(2); TW(3); TW(4); TW(5); TW(6); TW(7);
#undef TW
        }
        br[0*8+j1]=r0; br[1*8+j1]=r1; br[2*8+j1]=r2; br[3*8+j1]=r3;
        br[4*8+j1]=r4; br[5*8+j1]=r5; br[6*8+j1]=r6; br[7*8+j1]=r7;
        bi[0*8+j1]=i0; bi[1*8+j1]=i1; bi[2*8+j1]=i2; bi[3*8+j1]=i3;
        bi[4*8+j1]=i4; bi[5*8+j1]=i5; bi[6*8+j1]=i6; bi[7*8+j1]=i7;
    }
    for (int k2 = 0; k2 < 8; k2++) {
        double r0=br[k2*8],   r1=br[k2*8+1], r2=br[k2*8+2], r3=br[k2*8+3],
               r4=br[k2*8+4], r5=br[k2*8+5], r6=br[k2*8+6], r7=br[k2*8+7];
        double i0=bi[k2*8],   i1=bi[k2*8+1], i2=bi[k2*8+2], i3=bi[k2*8+3],
               i4=bi[k2*8+4], i5=bi[k2*8+5], i6=bi[k2*8+6], i7=bi[k2*8+7];
        RADIX8(M_SQRT1_2, r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7);
        ar[k2]=r0; ar[k2+8]=r1;  ar[k2+16]=r2; ar[k2+24]=r3;
        ar[k2+32]=r4; ar[k2+40]=r5; ar[k2+48]=r6; ar[k2+56]=r7;
        ai[k2]=i0; ai[k2+8]=i1;  ai[k2+16]=i2; ai[k2+24]=i3;
        ai[k2+32]=i4; ai[k2+40]=i5; ai[k2+48]=i6; ai[k2+56]=i7;
    }
}

#undef VT
#undef VADD
#undef VSUB
#undef VMUL
#undef VFMA
#undef VFNMA

__attribute__((unused))
static void exec_scalar(const struct fft3d_plan *p, const double *in, double *out,
                        int nvol)
{
    double ar[64], ai[64];
    for (int b = 0; b < nvol; b++) {
        const double *vi = in + (size_t)b * VOLD;
        double *vo = out + (size_t)b * VOLD;
        /* z axis: in -> out */
        for (int xy = 0; xy < 4096; xy++) {
            const double *q = vi + (size_t)xy * 128;
            double *o = vo + (size_t)xy * 128;
            for (int z = 0; z < 64; z++) { ar[z] = q[2*z]; ai[z] = q[2*z+1]; }
            line64_scalar(p->tw1, ar, ai);
            for (int z = 0; z < 64; z++) { o[2*z] = ar[z]; o[2*z+1] = ai[z]; }
        }
        /* y axis: in place on out, stride 64 complex */
        for (int x = 0; x < 64; x++)
            for (int z = 0; z < 64; z++) {
                double *q = vo + (size_t)x * 8192 + 2 * z;
                for (int y = 0; y < 64; y++) { ar[y] = q[y*128]; ai[y] = q[y*128+1]; }
                line64_scalar(p->tw1, ar, ai);
                for (int y = 0; y < 64; y++) { q[y*128] = ar[y]; q[y*128+1] = ai[y]; }
            }
        /* x axis: in place on out, stride 4096 complex */
        for (int y = 0; y < 64; y++)
            for (int z = 0; z < 64; z++) {
                double *q = vo + (size_t)y * 128 + 2 * z;
                for (int x = 0; x < 64; x++) { ar[x] = q[x*8192]; ai[x] = q[x*8192+1]; }
                line64_scalar(p->tw1, ar, ai);
                for (int x = 0; x < 64; x++) { q[x*8192] = ar[x]; q[x*8192+1] = ai[x]; }
            }
    }
}

/* ======================================================================== */
/*                          plan setup / teardown                           */
/* ======================================================================== */

__attribute__((unused))
static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != 64 || batch < 1) return NULL;
    struct fft3d_plan *p = aligned_alloc(64, sizeof *p);
    if (!p) return NULL;
    memset(p, 0, sizeof *p);
    p->L = 64;
    p->B = batch;
    p->nt = 0;

    for (int j = 0; j < 8; j++)
        for (int k = 0; k < 8; k++) {
            double a = -2.0 * M_PI * (double)(j * k) / 64.0;
            p->tw1[j][k][0] = cos(a);
            p->tw1[j][k][1] = sin(a);
            /* pass-3 lane twiddles: [k2][lane l] = w64^(l*k2) */
            p->tw3r[k][j] = cos(a);
            p->tw3i[k][j] = sin(a);
        }

    p->sc = mmap(NULL, SCSZ, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p->sc != MAP_FAILED) {
        p->sc_is_mmap = 1;
#ifdef MADV_HUGEPAGE
        madvise(p->sc, SCSZ, MADV_HUGEPAGE);
#endif
    } else {
        p->sc = aligned_alloc(64, SCSZ);
        p->sc_is_mmap = 0;
        if (!p->sc) { free(p); return NULL; }
    }
    memset(p->sc, 0, SCSZ);

    p->lb = aligned_alloc(64, 64 * 16 * sizeof(double));
    if (!p->lb) { fft3d_destroy(p); return NULL; }
    memset(p->lb, 0, 64 * 16 * sizeof(double));

#if defined(__AVX512F__)
    /* Self-tune pass-3 store type at (a clamp of) the real batch size:
     * plain stores win when the working set re-fits the cache across driver
     * repeats (B=1), streaming stores win once the batch leaves L3.        */
    {
        int bt = batch < 4 ? batch : 4;
        double *ti = aligned_alloc(64, (size_t)bt * VOLD * sizeof(double));
        double *to = aligned_alloc(64, (size_t)bt * VOLD * sizeof(double));
        if (ti && to) {
            for (size_t k = 0; k < (size_t)bt * VOLD; k++)
                ti[k] = (double)((k * 2654435761u) & 1023) * 9.765625e-4 - 0.5;
            double best[2] = { 1e30, 1e30 };
            for (int round = 0; round < 3; round++)
                for (int v = 0; v < 2; v++) {
                    exec_avx512(p, ti, to, bt, v);          /* warm, own state */
                    double t0 = now_s();
                    exec_avx512(p, ti, to, bt, v);
                    exec_avx512(p, ti, to, bt, v);
                    double dt = (now_s() - t0) / 2.0;
                    if (dt < best[v]) best[v] = dt;
                }
            p->nt = best[1] < 0.98 * best[0];
        }
        free(ti); free(to);
        snprintf(g_desc, sizeof g_desc,
                 "radix-8^2 per axis, split-complex AVX-512, padded scratch; "
                 "tuner pick[B=%d]=%s", batch, p->nt ? "nt" : "plain");
    }
#else
    snprintf(g_desc, sizeof g_desc,
             "radix-8^2 per axis, portable scalar path (no AVX-512 at build)");
#endif
    return p;
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    const double *vi = (const double *)in;
    double *vo = (double *)out;
#if defined(__AVX512F__)
    exec_avx512(p, vi, vo, p->B, p->nt);
#else
    exec_scalar(p, vi, vo, p->B);
#endif
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    if (p->sc) {
        if (p->sc_is_mmap) munmap(p->sc, SCSZ);
        else free(p->sc);
    }
    free(p->lb);
    free(p);
}
