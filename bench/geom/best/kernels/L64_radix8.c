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
 * Round panel_r8 adds a second STRUCTURE as a tuner candidate: the L2<->DRAM
 * tiling the r7 verdict named as the largest untried structural move
 * (LITERATURE 4.3 / corpus section 08 1.9).  "tiled" runs, per z-octet slab
 * (64x64x8 points = 528 KB padded, L2-resident on the node's 1 MB L2):
 *   pass A: y-FFT for all x (dense sequential 8-KB slab rows), then x-FFT
 *           for all ky IN PLACE in the slab -- the x-FFT's strided loads,
 *           which in the fused structure miss to L3 (SC is 4.5 MB), become
 *           L2 hits;
 *   pass B (after all 8 slabs): z-FFT per (kx,ky) -- 8 SEQUENTIAL read
 *           streams (one per slab) + 1 sequential write stream, the most
 *           prefetch-friendly memory shape this geometry allows.
 * The fused structure stays the default candidate; the create-time tuner
 * decides per machine.
 *
 * fft3d_create() self-tunes over a {structure} x {store mode} x {slab
 * prefetch lead} grid at (a clamp of) the real batch size, L8-style: the
 * machine decides, not me.
 *   store mode: plain / non-temporal / plain+prefetchw of the row PFW_LEAD
 *     ahead (the node kept prefetchw and rejected NT three rounds at L=36,
 *     and again at L=64 in r7);
 *   slabpf (fused structure only): while z-lining ky-slab k, T1-prefetch
 *     slab k+lead (lead 1 or 2) -- the exact lines the next x-line stage
 *     would otherwise miss to L3 (worth ~2% on wallaby, in-process A/B).
 * Round panel_r9 adds two prologue prefetches (the r8 verdict's named item:
 * "the first slab of every volume is always cold") and one tuner axis:
 *   propf: before the ky loop, T1-prefetch the first max(1,slabpf) ky-slabs
 *     of SC (1024 lines each) -- slabpf covers slab k+lead while z-lining k,
 *     so slab 0 was always demand-missed; symmetrically, plane 0 of the
 *     input in pass 1 (the next-plane prefetch covers planes 1..63 only).
 *     A/B'd on the picked candidate at create time, kept only if it wins.
 *   p1pf: pass-1 next-plane prefetch FORCED at B=1 (r6 gated it to batch on
 *     a wallaby number; the node's relatively slower L3 gets its own rows).
 * Round panel_r10: the r9 verdict names the remaining B=1 residual -- "the SC
 * store RFOs: pass 1 writes 4.5 MB scattered, the one component nothing has
 * ever hidden" -- and names the route: an in-plan create-time A/B applied to
 * a store-mode twin.  So pass 1's SC stores are now a three-way twin (scst):
 *   scst=0 plain (incumbent), scst=1 plain+prefetchw of the next plane's SC
 *   rows (the old env-only scpfw, promoted into the A/B so the node finally
 *   runs it), scst=2 NON-TEMPORAL (no RFO at all: the 4.5 MB of stale-line
 *   reads disappears, traded for pass 2 re-reading SC from DRAM -- which
 *   slabpf/propf then prefetch).  A/B'd on the picked candidate at create
 *   time, per machine; all three produce bit-identical output.  If scst!=0
 *   wins, the propf A/B is re-run under it (NT SC stores make the slab-0
 *   coldness strictly worse, so the earlier propf verdict may flip).
 * Round panel_r11 (scst read plain 3/3 on the node, so the SC-RFO theory is
 * dead; B=1 flat at 949.9 us = 1.73x the port floor, still the board's worst):
 *   xb -- compact x-FFT output buffer (68 KB, L2-resident).  The x-line pass
 *     currently updates SC IN PLACE, which dirties 4.5 MB/volume of SC lines
 *     and forces an L2->L3 writeback sweep that nothing has ever hidden --
 *     the last unaddressed traffic component at B=1.  xb=1 sends the x-FFT
 *     stage-2 output to a 68-KB buffer instead (slot (kx,zb) = kx*XBS+zb*16,
 *     XBS = 17 lines odd) and the z-lines read it back from L1/L2; SC is
 *     then READ-ONLY in pass 2+3 and never written back.  r6 measured the
 *     equivalent slab buffer 3-4% SLOWER on wallaby (fast L3 makes WBs free,
 *     extra store sweep costs) and rejected it -- but that was a WALLABY
 *     verdict, never run on the node, and p1pf (r9->r10) proved small wallaby
 *     losses can invert there.  Bit-identical output; in-plan A/B, 1% bar.
 *   fout -- the r9 verdict's law ("on CLX, store-feeding FMAs beat
 *     store-feeding adds by 3-6% on identical arithmetic") tested at the one
 *     site it can apply: the x-line stage-2 codelet, whose 16 outputs feed
 *     the L3-bound stores.  The codelet already produces its odd outputs by
 *     FMA (the d2-winning shape); the even outputs are adds and have no
 *     irrational factor to fold, so the twin converts them with
 *     fma(x, 1.0, y) -- IEEE-identical to add(x, y) (x*1 is exact, single
 *     rounding), so the twin is bit-identical and costs zero extra ops.
 *     In-plan A/B, strict win.  Expected: wallaby declines (SPR has fast
 *     dedicated FP-add ports that the FMA form gives up), node decides.
 *   tuner arena clamp raised 4 -> 8 volumes: B=8 is scored streaming 64 MB
 *     from DRAM, but its picks were being decided in a 32-MB arena that is
 *     half L3-resident on the node.  bt = min(B, 8) tunes B=8 in the regime
 *     it is scored in.
 * Env forcing for the monitor: FFT64R_STRUCT=0|1 (fused|tiled),
 * FFT64R_MODE=0|1|2, FFT64R_SLABPF=0|1|2 (lead), FFT64R_SCST=0|1|2 (pass-1
 * SC store mode; FFT64R_SCPFW=1 still accepted as an alias for scst=1),
 * FFT64R_P1PF=0|1, FFT64R_PROPF=0|1, FFT64R_XB=0|1, FFT64R_FOUT=0|1,
 * FFT64R_NOHP=1 (skip MADV_HUGEPAGE on SC), -DPFW_LEAD=n, FFT64R_TUNEDBG=1
 * to dump the tuner table to stderr.
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

/* Tiled-structure strides (doubles).  One z-octet slab holds slot (x, ky) at
 * x*TXS + ky*16 (re at +0, im at +8): ky rows are DENSE (16 doubles), the x
 * stride is 129 lines (odd, so gcd(stride, sets) = 1 at L1/L2 -- Bailey).
 * Slab = 64*TXS + 8 = 8257 lines (odd).  8 slabs = 4.23 MB. */
#define TXS   1032                   /* 129 cache lines between x rows       */
#define TSLAB ((size_t)(64 * TXS + 8))

/* Compact x-FFT output buffer (panel_r11 xb twin): slot (kx, zb) =
 * kx*XBS + zb*16 doubles (re at +0, im at +8), row stride 17 lines (odd). */
#define XBS  136
#define XBSZ ((size_t)64 * XBS * sizeof(double))

/* SC must hold whichever structure is larger (SCKS/SCXPAD are -D-sweepable);
 * the xb buffer is appended to the same mapping so it shares the hugepages. */
#define SCSZ_FUSED ((size_t)64 * SCXS * sizeof(double))
#define SCSZ_TILED (8 * TSLAB * sizeof(double))
#define SCSZ (SCSZ_FUSED > SCSZ_TILED ? SCSZ_FUSED : SCSZ_TILED)
#define SCTOT (SCSZ + XBSZ)

struct fft3d_plan {
    int L, B;
    int sstruct;                     /* 0 fused (2 sweeps), 1 tiled (slab + z sweep) */
    int mode;                        /* final stores: 0 plain, 1 NT, 2 plain+prefetchw */
    int slabpf;                      /* fused: T1-prefetch slab ky+lead during z-lines */
    int scst;                        /* pass-1 SC store mode: 0 plain, 1 +prefetchw, 2 NT */
    int p1pf;                        /* force pass-1 next-plane prefetch even at B=1 */
    int propf;                       /* prologue: prefetch slab 0 (+lead) / plane 0 */
    int xb_on;                       /* x-FFT output -> compact 68-KB buffer (SC read-only) */
    int fout;                        /* x-line stage-2 codelet: all-FMA store feeds */
    double *xb;                      /* the compact buffer (tail of the SC mapping) */
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

/* RADIX8F (panel_r11 fout twin): identical to RADIX8 except the final-stage
 * EVEN outputs are produced by FMA-class ops instead of add/sub --
 * fma(x, 1.0, y) is IEEE-identical to add(x, y) (x*1 exact, one rounding),
 * so the twin is bit-identical at the same op count.  The odd outputs were
 * already FMA-fed.  This is the r9 verdict's store-feeding-class law, tested
 * at the one L=64 site where codelet outputs feed L3-bound stores.          */
#define RADIX8F(C, ONE, r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7) do { \
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
    r0=VFMA(c0r,ONE,c4r); r4=VFNMA(c4r,ONE,c0r);                              \
    i0=VFMA(c0i,ONE,c4i); i4=VFNMA(c4i,ONE,c0i);                              \
    r2=VFMA(c2r,ONE,c6i); r6=VFNMA(c6i,ONE,c2r);                              \
    i2=VFNMA(c6r,ONE,c2i); i6=VFMA(c2i,ONE,c6r);                              \
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

#define ST_PLAIN(a, v) _mm512_store_pd((a), (v))
#define ST_NT(a, v)    _mm512_stream_pd((a), (v))

/* pass 1: y-FFT.  in: interleaved input plane x; out: SC (split, lanes = z).
 * Deinterleave is fused into the stage-1 loads (one permutex2var per vector,
 * measured free next to the FP work).  PF=1 variant additionally prefetches
 * the NEXT x-plane into L2, spread 16 lines per stage-1 iteration -- used
 * when the batch streams from DRAM (measured ~5% on wallaby; a plane-buffer
 * sequential-copy variant was measured SLOWER and was removed).
 * ST is the SC store instruction (panel_r10 store-mode twin): plain stores
 * pay an RFO per line (4.5 MB/volume of stale-line reads, the r9 verdict's
 * named residual); ST_NT skips the RFO entirely.  Every SC line is written
 * whole (each __m512d store covers exactly one 64-B line), so NT stores
 * write-combine cleanly even though the slots are scattered.                */
#define PASS1_DEF(NAME, PF, SCPFW, ST)                                        \
static void NAME(const struct fft3d_plan *p, const double *in, double *sc)    \
{                                                                             \
    const __m512i IEV = _mm512_setr_epi64(0,2,4,6,8,10,12,14);                \
    const __m512i IOD = _mm512_setr_epi64(1,3,5,7,9,11,13,15);                \
    const VT C = _mm512_set1_pd(M_SQRT1_2);                                   \
    double *lb = p->lb;                                                       \
    if (PF && p->propf)                                                       \
        /* plane-0 prologue: the next-plane prefetch below covers planes     \
         * 1..63 (and, at batch, the next volume's plane 0), so plane 0 of   \
         * the first volume is the one plane always demand-missed.  1024     \
         * T1 lines, ~0.15 us of issue -- worst case a wash at batch.        */\
        for (int ln = 0; ln < 1024; ln++)                                     \
            _mm_prefetch((const char *)in + (size_t)ln * 64, _MM_HINT_T1);    \
    for (int x = 0; x < 64; x++) {                                            \
        const double *px = in + (size_t)x * 8192;                             \
        const double *pnext = px + 8192;                                      \
        double *scx = sc + (size_t)x * SCXS;                                  \
        for (int zb = 0; zb < 8; zb++) {                                      \
            const double *pz = px + zb * 16;                                  \
            /* stage 1: for each residue j1, DFT_8 over y = j1 + 8t */        \
            for (int j1 = 0; j1 < 8; j1++) {                                  \
                if (SCPFW && x < 63) {                                        \
                    /* write-intent prefetch of the NEXT plane's SC slots:   \
                     * one 17-line ky row per stage-1 iteration hides the    \
                     * 4.5 MB/volume of SC-store RFOs behind a plane of lead */\
                    const char *_pw = (const char *)(scx + SCXS              \
                                                     + (zb*8 + j1) * SCKS);  \
                    __builtin_prefetch(_pw,        1, 2);                    \
                    __builtin_prefetch(_pw +   64, 1, 2);                    \
                    __builtin_prefetch(_pw +  128, 1, 2);                    \
                    __builtin_prefetch(_pw +  192, 1, 2);                    \
                    __builtin_prefetch(_pw +  256, 1, 2);                    \
                    __builtin_prefetch(_pw +  320, 1, 2);                    \
                    __builtin_prefetch(_pw +  384, 1, 2);                    \
                    __builtin_prefetch(_pw +  448, 1, 2);                    \
                    __builtin_prefetch(_pw +  512, 1, 2);                    \
                    __builtin_prefetch(_pw +  576, 1, 2);                    \
                    __builtin_prefetch(_pw +  640, 1, 2);                    \
                    __builtin_prefetch(_pw +  704, 1, 2);                    \
                    __builtin_prefetch(_pw +  768, 1, 2);                    \
                    __builtin_prefetch(_pw +  832, 1, 2);                    \
                    __builtin_prefetch(_pw +  896, 1, 2);                    \
                    __builtin_prefetch(_pw +  960, 1, 2);                    \
                    __builtin_prefetch(_pw + 1024, 1, 2);                    \
                }                                                             \
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
                ST(so+0*(8*SCKS),r0); ST(so+0*(8*SCKS)+8,i0);                 \
                ST(so+1*(8*SCKS),r1); ST(so+1*(8*SCKS)+8,i1);                 \
                ST(so+2*(8*SCKS),r2); ST(so+2*(8*SCKS)+8,i2);                 \
                ST(so+3*(8*SCKS),r3); ST(so+3*(8*SCKS)+8,i3);                 \
                ST(so+4*(8*SCKS),r4); ST(so+4*(8*SCKS)+8,i4);                 \
                ST(so+5*(8*SCKS),r5); ST(so+5*(8*SCKS)+8,i5);                 \
                ST(so+6*(8*SCKS),r6); ST(so+6*(8*SCKS)+8,i6);                 \
                ST(so+7*(8*SCKS),r7); ST(so+7*(8*SCKS)+8,i7);                 \
            }                                                                 \
        }                                                                     \
    }                                                                         \
}

PASS1_DEF(pass1_plain, 0, 0, ST_PLAIN)
PASS1_DEF(pass1_pf, 1, 0, ST_PLAIN)
PASS1_DEF(pass1_plain_w, 0, 1, ST_PLAIN)
PASS1_DEF(pass1_pf_w, 1, 1, ST_PLAIN)
PASS1_DEF(pass1_plain_nt, 0, 0, ST_NT)
PASS1_DEF(pass1_pf_nt, 1, 0, ST_NT)

/* x-line FFT for one (ky, zb) group.  lanes = z, zero shuffles.  The +16
 * prefetch covers the next zb column (removing it costs ~12% at B=8).
 * Four variants (panel_r11):
 *   destination: stage 2 IN PLACE in SC (incumbent; the stores hit
 *     just-read lines, no RFO, but dirty 4.5 MB/volume of SC that must be
 *     written back L2->L3), or XB (compact 68-KB buffer, L2-hot, SC stays
 *     clean -- the writeback sweep disappears; slot (kx,zb) = kx*XBS+zb*16
 *     with kx = k2 + 8*k1, same mapping the z-lines read back).
 *   stage-2 codelet: RADIX8 (incumbent) or RADIX8F (all 16 store feeds
 *     FMA-class, bit-identical -- the r9 store-feeding law twin).
 * sz is the (ky,zb) slot in SC; xd is the zb slot in XB (unused by SC
 * variants).                                                                 */
#define XLINE_DEF(NAME, R8S2, SOBASE, SOSTRIDE)                               \
static void NAME(const struct fft3d_plan *p, double *sz, double *xd,          \
                 double *lb)                                                  \
{                                                                             \
    const VT C = _mm512_set1_pd(M_SQRT1_2);                                   \
    const VT ONE __attribute__((unused)) = _mm512_set1_pd(1.0);               \
    (void)xd;                                                                 \
    for (int j1 = 0; j1 < 8; j1++) {                                          \
        const double *bx = sz + (size_t)j1 * SCXS;                            \
        P2LD(0) P2LD(1) P2LD(2) P2LD(3) P2LD(4) P2LD(5) P2LD(6) P2LD(7)       \
        RADIX8(C, r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7);          \
        if (j1) {                                                             \
            const double (*tw)[2] = p->tw1[j1];                               \
            XTW(1); XTW(2); XTW(3); XTW(4); XTW(5); XTW(6); XTW(7);           \
        }                                                                     \
        LBST(0) LBST(1) LBST(2) LBST(3) LBST(4) LBST(5) LBST(6) LBST(7)       \
    }                                                                         \
    for (int k2 = 0; k2 < 8; k2++) {                                          \
        const double *lk = lb + k2 * 128;                                     \
        LBLD(0) LBLD(1) LBLD(2) LBLD(3) LBLD(4) LBLD(5) LBLD(6) LBLD(7)       \
        R8S2;                                                                 \
        double *so = (SOBASE) + (size_t)k2 * (SOSTRIDE);                      \
        SST(0,SOSTRIDE) SST(1,SOSTRIDE) SST(2,SOSTRIDE) SST(3,SOSTRIDE)       \
        SST(4,SOSTRIDE) SST(5,SOSTRIDE) SST(6,SOSTRIDE) SST(7,SOSTRIDE)       \
    }                                                                         \
}

#define P2LD(t) VT r##t = VLD(bx + (size_t)(t)*(8*SCXS)),                     \
                   i##t = VLD(bx + (size_t)(t)*(8*SCXS) + 8);                 \
        _mm_prefetch((const char *)(bx + (size_t)(t)*(8*SCXS) + 16), _MM_HINT_T0);
#define XTW(k) CTW(r##k, i##k, _mm512_set1_pd(tw[k][0]), _mm512_set1_pd(tw[k][1]))
#define LBST(k) VST(lb + ((k)*8 + j1)*16, r##k); VST(lb + ((k)*8 + j1)*16 + 8, i##k);
#define LBLD(t) VT r##t = VLD(lk + (t)*16), i##t = VLD(lk + (t)*16 + 8);
#define SST(k1, S) VST(so + (size_t)(k1)*8*(S), r##k1);                       \
                   VST(so + (size_t)(k1)*8*(S) + 8, i##k1);
#define R8A RADIX8(C, r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7)
#define R8F RADIX8F(C, ONE, r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7)

XLINE_DEF(xline_sc,   R8A, sz, SCXS)
XLINE_DEF(xline_sc_f, R8F, sz, SCXS)
XLINE_DEF(xline_xb,   R8A, xd, XBS)
XLINE_DEF(xline_xb_f, R8F, xd, XBS)

#undef P2LD
#undef XTW
#undef LBST
#undef LBLD
#undef SST
#undef R8A
#undef R8F

/* write-intent prefetch of one 16-line (1 KiB) output row: issues the RFO
 * early instead of letting the store buffer stall on it (L6_unrolled r3 /
 * L36_pfa idiom; the node rejected NT and kept prefetchw 3 rounds at L=36).
 * __builtin_prefetch(p,1,3) emits prefetchw wherever PRFCHW exists.         */
#ifndef PFW_LEAD
#define PFW_LEAD 4                   /* rows (kx iterations) of lead */
#endif
#define PFW_ROW(addr) do {                                                     \
    const char *_pw = (const char *)(addr);                                    \
    __builtin_prefetch(_pw,       1, 3); __builtin_prefetch(_pw +  64, 1, 3);  \
    __builtin_prefetch(_pw + 128, 1, 3); __builtin_prefetch(_pw + 192, 1, 3);  \
    __builtin_prefetch(_pw + 256, 1, 3); __builtin_prefetch(_pw + 320, 1, 3);  \
    __builtin_prefetch(_pw + 384, 1, 3); __builtin_prefetch(_pw + 448, 1, 3);  \
    __builtin_prefetch(_pw + 512, 1, 3); __builtin_prefetch(_pw + 576, 1, 3);  \
    __builtin_prefetch(_pw + 640, 1, 3); __builtin_prefetch(_pw + 704, 1, 3);  \
    __builtin_prefetch(_pw + 768, 1, 3); __builtin_prefetch(_pw + 832, 1, 3);  \
    __builtin_prefetch(_pw + 896, 1, 3); __builtin_prefetch(_pw + 960, 1, 3);  \
} while (0)

/* 8 T1 line prefetches at base+off: half of the next ky slab's (x,*) slot */
#define PFS8(base, off) do {                                                   \
    const char *_ps = (const char *)(base) + (off);                            \
    _mm_prefetch(_ps,       _MM_HINT_T1); _mm_prefetch(_ps +  64, _MM_HINT_T1);\
    _mm_prefetch(_ps + 128, _MM_HINT_T1); _mm_prefetch(_ps + 192, _MM_HINT_T1);\
    _mm_prefetch(_ps + 256, _MM_HINT_T1); _mm_prefetch(_ps + 320, _MM_HINT_T1);\
    _mm_prefetch(_ps + 384, _MM_HINT_T1); _mm_prefetch(_ps + 448, _MM_HINT_T1);\
} while (0)

/* FUSED pass 2+3: for each ky, first the 8 x-line groups (x-FFT in place in
 * SC), then immediately the 64 z-lines of that ky-slab -- the slab (64 KiB)
 * is still L2-hot, which beats a third full-volume L3 sweep by ~10% both
 * regimes (measured in-process on wallaby, bitwise-identical output).
 * z-line: z = 8g + l: radix-8 over g (registers), lane twiddle w64^(l k2),
 * transpose (lanes -> registers, SW residue), radix-8 over l, interleave with
 * SW-composed index vectors, 1 KiB contiguous store per (kx,ky) row.  Output
 * rows go out ky-major (stride 64 KiB between consecutive rows).
 * slabpf: while z-lining slab ky, T1-prefetch slab ky+1 (the exact lines the
 * next round of x-line stage-1 loads will otherwise miss to L3), 16 lines per
 * kx iteration, split in two 8-line groups against the FP work.             */
#define PASS23_DEF(NAME, STORE, PFW, XB)                                      \
static void NAME(const struct fft3d_plan *p, double *sc, double *out,         \
                 int slabpf, int propf, int fout)                             \
{                                                                             \
    const __m512i ILO = _mm512_setr_epi64(0,8,1,9,4,12,5,13);                 \
    const __m512i IHI = _mm512_setr_epi64(2,10,3,11,6,14,7,15);               \
    const VT C = _mm512_set1_pd(M_SQRT1_2);                                   \
    /* prologue (r9): slabpf prefetches slab ky+lead while z-lining ky, so   \
     * the FIRST slab(s) of every volume are always demand-missed to L3 --   \
     * pass 1 wrote slab 0's low-x lines a whole SC sweep ago.  Pull the     \
     * first max(1,lead) slabs (1024 T1 lines each) up front.                */\
    if (propf) {                                                              \
        const int npro = slabpf > 0 ? slabpf : 1;                             \
        for (int k = 0; k < npro; k++) {                                      \
            const double *pp = sc + (size_t)k * SCKS;                         \
            for (int x = 0; x < 64; x++) {                                    \
                PFS8(pp + (size_t)x * SCXS, 0);                               \
                PFS8(pp + (size_t)x * SCXS, 512);                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    for (int ky = 0; ky < 64; ky++) {                                         \
        double *sy = sc + (size_t)ky * SCKS;                                  \
        for (int zb = 0; zb < 8; zb++) {                                      \
            if (XB) { if (fout) xline_xb_f(p, sy + zb*16, p->xb + zb*16, p->lb); \
                      else      xline_xb  (p, sy + zb*16, p->xb + zb*16, p->lb); } \
            else    { if (fout) xline_sc_f(p, sy + zb*16, 0, p->lb);          \
                      else      xline_sc  (p, sy + zb*16, 0, p->lb); }        \
        }                                                                     \
        const double *pfs = sy + (size_t)slabpf * SCKS;                       \
        const int dopf = slabpf && ky < 64 - slabpf;                          \
        for (int kx = 0; kx < 64; kx++) {                                     \
            const double *s = XB ? p->xb + (size_t)kx * XBS                   \
                                 : sy + (size_t)kx * SCXS;                    \
            double *o = out + (size_t)kx * 8192 + ky * 128;                   \
            if (PFW && kx + PFW_LEAD < 64)                                    \
                PFW_ROW(o + PFW_LEAD * 8192);                                 \
            VT r0 = VLD(s),       i0 = VLD(s + 8);                            \
            VT r1 = VLD(s + 16),  i1 = VLD(s + 24);                           \
            VT r2 = VLD(s + 32),  i2 = VLD(s + 40);                           \
            VT r3 = VLD(s + 48),  i3 = VLD(s + 56);                           \
            VT r4 = VLD(s + 64),  i4 = VLD(s + 72);                           \
            VT r5 = VLD(s + 80),  i5 = VLD(s + 88);                           \
            VT r6 = VLD(s + 96),  i6 = VLD(s + 104);                          \
            VT r7 = VLD(s + 112), i7 = VLD(s + 120);                          \
            if (dopf) PFS8(pfs + (size_t)kx * SCXS, 0);                       \
            RADIX8(C, r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7);      \
            CTW(r1,i1,VLD(p->tw3r[1]),VLD(p->tw3i[1]));                       \
            CTW(r2,i2,VLD(p->tw3r[2]),VLD(p->tw3i[2]));                       \
            CTW(r3,i3,VLD(p->tw3r[3]),VLD(p->tw3i[3]));                       \
            CTW(r4,i4,VLD(p->tw3r[4]),VLD(p->tw3i[4]));                       \
            CTW(r5,i5,VLD(p->tw3r[5]),VLD(p->tw3i[5]));                       \
            CTW(r6,i6,VLD(p->tw3r[6]),VLD(p->tw3i[6]));                       \
            CTW(r7,i7,VLD(p->tw3r[7]),VLD(p->tw3i[7]));                       \
            if (dopf) PFS8(pfs + (size_t)kx * SCXS, 512);                     \
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

PASS23_DEF(pass23_plain, ST_PLAIN, 0, 0)
PASS23_DEF(pass23_nt, ST_NT, 0, 0)
PASS23_DEF(pass23_pfw, ST_PLAIN, 1, 0)
PASS23_DEF(pass23_plain_xb, ST_PLAIN, 0, 1)
PASS23_DEF(pass23_nt_xb, ST_NT, 0, 1)
PASS23_DEF(pass23_pfw_xb, ST_PLAIN, 1, 1)

/* ======================= tiled structure (panel_r8) ===================== */
/* x-line FFT for one ky inside an L2-resident z-octet slab, in place.  Same
 * algorithm and twiddles as xline_avx512, strides TXS instead of SCXS; the
 * +16 T0 prefetch covers the next ky column (dense: +128 B).               */
static inline void xline_tiled(const struct fft3d_plan *p, double *base, double *lb)
{
    const VT C = _mm512_set1_pd(M_SQRT1_2);
    for (int j1 = 0; j1 < 8; j1++) {
        const double *bx = base + (size_t)j1 * TXS;
#define P2LD(t) VT r##t = VLD(bx + (size_t)(t)*(8*TXS)),                      \
                   i##t = VLD(bx + (size_t)(t)*(8*TXS) + 8);                  \
        _mm_prefetch((const char *)(bx + (size_t)(t)*(8*TXS) + 16), _MM_HINT_T0);
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
        double *so = base + (size_t)k2 * TXS;
#define SST(k1) VST(so + (size_t)(k1)*(8*TXS), r##k1);                        \
                VST(so + (size_t)(k1)*(8*TXS) + 8, i##k1);
        SST(0) SST(1) SST(2) SST(3) SST(4) SST(5) SST(6) SST(7)
#undef SST
    }
}

/* pass A, one z-octet slab: y-FFT for every x (input reads are this zb's
 * 128-B chunks at 1-KiB stride; slab stores are DENSE sequential 8-KiB
 * rows), then the x-FFT for every ky in place in the slab, which is
 * L2-resident (528 KB against the node's 1 MB).  PF=1 T1-prefetches the
 * next x's chunk set, 16 lines per stage-1 iteration (the exact loads
 * iteration j1 of x+1 will perform) -- for streaming batches.             */
#define PASSA_DEF(NAME, PF)                                                   \
static void NAME(const struct fft3d_plan *p, const double *in, double *slab,  \
                 int zb)                                                      \
{                                                                             \
    const __m512i IEV = _mm512_setr_epi64(0,2,4,6,8,10,12,14);                \
    const __m512i IOD = _mm512_setr_epi64(1,3,5,7,9,11,13,15);                \
    const VT C = _mm512_set1_pd(M_SQRT1_2);                                   \
    double *lb = p->lb;                                                       \
    for (int x = 0; x < 64; x++) {                                            \
        const double *px = in + (size_t)x * 8192 + zb * 16;                   \
        double *dst = slab + (size_t)x * TXS;                                 \
        for (int j1 = 0; j1 < 8; j1++) {                                      \
            if (PF && x < 63) {                                               \
                const char *pf = (const char *)(px + 8192 + j1 * 128);        \
                _mm_prefetch(pf,          _MM_HINT_T1);                       \
                _mm_prefetch(pf +    64,  _MM_HINT_T1);                       \
                _mm_prefetch(pf + 1*8192, _MM_HINT_T1);                       \
                _mm_prefetch(pf + 1*8192 + 64, _MM_HINT_T1);                  \
                _mm_prefetch(pf + 2*8192, _MM_HINT_T1);                       \
                _mm_prefetch(pf + 2*8192 + 64, _MM_HINT_T1);                  \
                _mm_prefetch(pf + 3*8192, _MM_HINT_T1);                       \
                _mm_prefetch(pf + 3*8192 + 64, _MM_HINT_T1);                  \
                _mm_prefetch(pf + 4*8192, _MM_HINT_T1);                       \
                _mm_prefetch(pf + 4*8192 + 64, _MM_HINT_T1);                  \
                _mm_prefetch(pf + 5*8192, _MM_HINT_T1);                       \
                _mm_prefetch(pf + 5*8192 + 64, _MM_HINT_T1);                  \
                _mm_prefetch(pf + 6*8192, _MM_HINT_T1);                       \
                _mm_prefetch(pf + 6*8192 + 64, _MM_HINT_T1);                  \
                _mm_prefetch(pf + 7*8192, _MM_HINT_T1);                       \
                _mm_prefetch(pf + 7*8192 + 64, _MM_HINT_T1);                  \
            }                                                                 \
            const double *py = px + j1 * 128;                                 \
            VT r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7;              \
            {                                                                 \
                VT a0=VLD(py+0*1024), b0=VLD(py+0*1024+8);                    \
                VT a1=VLD(py+1*1024), b1=VLD(py+1*1024+8);                    \
                VT a2=VLD(py+2*1024), b2=VLD(py+2*1024+8);                    \
                VT a3=VLD(py+3*1024), b3=VLD(py+3*1024+8);                    \
                VT a4=VLD(py+4*1024), b4=VLD(py+4*1024+8);                    \
                VT a5=VLD(py+5*1024), b5=VLD(py+5*1024+8);                    \
                VT a6=VLD(py+6*1024), b6=VLD(py+6*1024+8);                    \
                VT a7=VLD(py+7*1024), b7=VLD(py+7*1024+8);                    \
                r0=_mm512_permutex2var_pd(a0,IEV,b0); i0=_mm512_permutex2var_pd(a0,IOD,b0); \
                r1=_mm512_permutex2var_pd(a1,IEV,b1); i1=_mm512_permutex2var_pd(a1,IOD,b1); \
                r2=_mm512_permutex2var_pd(a2,IEV,b2); i2=_mm512_permutex2var_pd(a2,IOD,b2); \
                r3=_mm512_permutex2var_pd(a3,IEV,b3); i3=_mm512_permutex2var_pd(a3,IOD,b3); \
                r4=_mm512_permutex2var_pd(a4,IEV,b4); i4=_mm512_permutex2var_pd(a4,IOD,b4); \
                r5=_mm512_permutex2var_pd(a5,IEV,b5); i5=_mm512_permutex2var_pd(a5,IOD,b5); \
                r6=_mm512_permutex2var_pd(a6,IEV,b6); i6=_mm512_permutex2var_pd(a6,IOD,b6); \
                r7=_mm512_permutex2var_pd(a7,IEV,b7); i7=_mm512_permutex2var_pd(a7,IOD,b7); \
            }                                                                 \
            RADIX8(C, r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7);      \
            if (j1) {                                                         \
                const double (*tw)[2] = p->tw1[j1];                           \
                CTW(r1,i1,_mm512_set1_pd(tw[1][0]),_mm512_set1_pd(tw[1][1])); \
                CTW(r2,i2,_mm512_set1_pd(tw[2][0]),_mm512_set1_pd(tw[2][1])); \
                CTW(r3,i3,_mm512_set1_pd(tw[3][0]),_mm512_set1_pd(tw[3][1])); \
                CTW(r4,i4,_mm512_set1_pd(tw[4][0]),_mm512_set1_pd(tw[4][1])); \
                CTW(r5,i5,_mm512_set1_pd(tw[5][0]),_mm512_set1_pd(tw[5][1])); \
                CTW(r6,i6,_mm512_set1_pd(tw[6][0]),_mm512_set1_pd(tw[6][1])); \
                CTW(r7,i7,_mm512_set1_pd(tw[7][0]),_mm512_set1_pd(tw[7][1])); \
            }                                                                 \
            VST(lb+(0*8+j1)*16,r0); VST(lb+(0*8+j1)*16+8,i0);                 \
            VST(lb+(1*8+j1)*16,r1); VST(lb+(1*8+j1)*16+8,i1);                 \
            VST(lb+(2*8+j1)*16,r2); VST(lb+(2*8+j1)*16+8,i2);                 \
            VST(lb+(3*8+j1)*16,r3); VST(lb+(3*8+j1)*16+8,i3);                 \
            VST(lb+(4*8+j1)*16,r4); VST(lb+(4*8+j1)*16+8,i4);                 \
            VST(lb+(5*8+j1)*16,r5); VST(lb+(5*8+j1)*16+8,i5);                 \
            VST(lb+(6*8+j1)*16,r6); VST(lb+(6*8+j1)*16+8,i6);                 \
            VST(lb+(7*8+j1)*16,r7); VST(lb+(7*8+j1)*16+8,i7);                 \
        }                                                                     \
        for (int k2 = 0; k2 < 8; k2++) {                                      \
            const double *lk = lb + k2 * 128;                                 \
            VT r0=VLD(lk+0),   i0=VLD(lk+8);                                  \
            VT r1=VLD(lk+16),  i1=VLD(lk+24);                                 \
            VT r2=VLD(lk+32),  i2=VLD(lk+40);                                 \
            VT r3=VLD(lk+48),  i3=VLD(lk+56);                                 \
            VT r4=VLD(lk+64),  i4=VLD(lk+72);                                 \
            VT r5=VLD(lk+80),  i5=VLD(lk+88);                                 \
            VT r6=VLD(lk+96),  i6=VLD(lk+104);                                \
            VT r7=VLD(lk+112), i7=VLD(lk+120);                                \
            RADIX8(C, r0,r1,r2,r3,r4,r5,r6,r7, i0,i1,i2,i3,i4,i5,i6,i7);      \
            /* ky = k2 + 8*k1 -> dst + k2*16 + k1*128: dense 8-KiB row */     \
            double *so = dst + k2 * 16;                                       \
            VST(so+0*128,r0); VST(so+0*128+8,i0);                             \
            VST(so+1*128,r1); VST(so+1*128+8,i1);                             \
            VST(so+2*128,r2); VST(so+2*128+8,i2);                             \
            VST(so+3*128,r3); VST(so+3*128+8,i3);                             \
            VST(so+4*128,r4); VST(so+4*128+8,i4);                             \
            VST(so+5*128,r5); VST(so+5*128+8,i5);                             \
            VST(so+6*128,r6); VST(so+6*128+8,i6);                             \
            VST(so+7*128,r7); VST(so+7*128+8,i7);                             \
        }                                                                     \
    }                                                                         \
    for (int ky = 0; ky < 64; ky++)                                           \
        xline_tiled(p, slab + ky * 16, lb);                                   \
}

PASSA_DEF(passA_plain, 0)
PASSA_DEF(passA_pf, 1)

/* pass B: z-FFT.  kx outer / ky inner, so the 8 slab reads are 8 SEQUENTIAL
 * streams (consecutive ky = +128 B in each slab) and the output is ONE
 * sequential write stream (consecutive 1-KiB rows of the kx plane).  The
 * z-line body is identical to the fused pass's: radix-8 over the octets
 * (g -> k2), lane twiddle w64^(l*k2), transpose pair, radix-8 over lanes
 * (l -> k1), SW-composed interleave, 1-KiB row store.  A paced T0 read
 * prefetch runs PFB_LEAD rows ahead in every slab (16 lines/iteration,
 * exactly consumption rate) -- the slabs live in L3, not L2.               */
#ifndef PFB_LEAD
#define PFB_LEAD 8                   /* rows (ky iterations) of read lead */
#endif
#define PASSB_DEF(NAME, STORE, PFW)                                           \
static void NAME(const struct fft3d_plan *p, const double *sc, double *out)   \
{                                                                             \
    const __m512i ILO = _mm512_setr_epi64(0,8,1,9,4,12,5,13);                 \
    const __m512i IHI = _mm512_setr_epi64(2,10,3,11,6,14,7,15);               \
    const VT C = _mm512_set1_pd(M_SQRT1_2);                                   \
    for (int kx = 0; kx < 64; kx++) {                                         \
        const double *sx = sc + (size_t)kx * TXS;                             \
        double *ox = out + (size_t)kx * 8192;                                 \
        for (int ky = 0; ky < 64; ky++) {                                     \
            const double *s = sx + ky * 16;                                   \
            double *o = ox + ky * 128;                                        \
            if (ky + PFB_LEAD < 64) {                                         \
                const char *pb = (const char *)(s + PFB_LEAD * 16);           \
                _mm_prefetch(pb + 0*(TSLAB*8),      _MM_HINT_T0);             \
                _mm_prefetch(pb + 0*(TSLAB*8) + 64, _MM_HINT_T0);             \
                _mm_prefetch(pb + 1*(TSLAB*8),      _MM_HINT_T0);             \
                _mm_prefetch(pb + 1*(TSLAB*8) + 64, _MM_HINT_T0);             \
                _mm_prefetch(pb + 2*(TSLAB*8),      _MM_HINT_T0);             \
                _mm_prefetch(pb + 2*(TSLAB*8) + 64, _MM_HINT_T0);             \
                _mm_prefetch(pb + 3*(TSLAB*8),      _MM_HINT_T0);             \
                _mm_prefetch(pb + 3*(TSLAB*8) + 64, _MM_HINT_T0);             \
                _mm_prefetch(pb + 4*(TSLAB*8),      _MM_HINT_T0);             \
                _mm_prefetch(pb + 4*(TSLAB*8) + 64, _MM_HINT_T0);             \
                _mm_prefetch(pb + 5*(TSLAB*8),      _MM_HINT_T0);             \
                _mm_prefetch(pb + 5*(TSLAB*8) + 64, _MM_HINT_T0);             \
                _mm_prefetch(pb + 6*(TSLAB*8),      _MM_HINT_T0);             \
                _mm_prefetch(pb + 6*(TSLAB*8) + 64, _MM_HINT_T0);             \
                _mm_prefetch(pb + 7*(TSLAB*8),      _MM_HINT_T0);             \
                _mm_prefetch(pb + 7*(TSLAB*8) + 64, _MM_HINT_T0);             \
            }                                                                 \
            if (PFW && ky + PFW_LEAD < 64)                                    \
                PFW_ROW(o + PFW_LEAD * 128);                                  \
            VT r0 = VLD(s + 0*TSLAB), i0 = VLD(s + 0*TSLAB + 8);              \
            VT r1 = VLD(s + 1*TSLAB), i1 = VLD(s + 1*TSLAB + 8);              \
            VT r2 = VLD(s + 2*TSLAB), i2 = VLD(s + 2*TSLAB + 8);              \
            VT r3 = VLD(s + 3*TSLAB), i3 = VLD(s + 3*TSLAB + 8);              \
            VT r4 = VLD(s + 4*TSLAB), i4 = VLD(s + 4*TSLAB + 8);              \
            VT r5 = VLD(s + 5*TSLAB), i5 = VLD(s + 5*TSLAB + 8);              \
            VT r6 = VLD(s + 6*TSLAB), i6 = VLD(s + 6*TSLAB + 8);              \
            VT r7 = VLD(s + 7*TSLAB), i7 = VLD(s + 7*TSLAB + 8);              \
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

PASSB_DEF(passB_plain, ST_PLAIN, 0)
PASSB_DEF(passB_nt, ST_NT, 0)
PASSB_DEF(passB_pfw, ST_PLAIN, 1)
#undef ST_PLAIN
#undef ST_NT

static void exec_avx512(const struct fft3d_plan *p, const double *in, double *out,
                        int nvol, int sstruct, int mode, int slabpf, int scst,
                        int p1pf, int propf, int xb, int fout)
{
    if (sstruct == 1) {
        for (int b = 0; b < nvol; b++) {
            const double *vi = in + (size_t)b * VOLD;
            double *vo = out + (size_t)b * VOLD;
            for (int zb = 0; zb < 8; zb++) {
                double *slab = p->sc + (size_t)zb * TSLAB;
                if (nvol > 1) passA_pf(p, vi, slab, zb);
                else          passA_plain(p, vi, slab, zb);
            }
            if (mode == 1)      passB_nt(p, p->sc, vo);
            else if (mode == 2) passB_pfw(p, p->sc, vo);
            else                passB_plain(p, p->sc, vo);
        }
        if (mode == 1) _mm_sfence();
        return;
    }
    for (int b = 0; b < nvol; b++) {
        const double *vi = in + (size_t)b * VOLD;
        double *vo = out + (size_t)b * VOLD;
        if (nvol > 1 || p1pf) { if (scst == 2)      pass1_pf_nt(p, vi, p->sc);
                                else if (scst == 1) pass1_pf_w(p, vi, p->sc);
                                else                pass1_pf(p, vi, p->sc); }
        else                  { if (scst == 2)      pass1_plain_nt(p, vi, p->sc);
                                else if (scst == 1) pass1_plain_w(p, vi, p->sc);
                                else                pass1_plain(p, vi, p->sc); }
        /* NT SC stores: drain the WC buffers before pass 2 reads the same
         * lines back (same-core data dependence is architecturally safe,
         * but a straggling half-filled buffer would flush mid-pass-2)      */
        if (scst == 2) _mm_sfence();
        if (xb) {
            if (mode == 1)      pass23_nt_xb(p, p->sc, vo, slabpf, propf, fout);
            else if (mode == 2) pass23_pfw_xb(p, p->sc, vo, slabpf, propf, fout);
            else                pass23_plain_xb(p, p->sc, vo, slabpf, propf, fout);
        } else {
            if (mode == 1)      pass23_nt(p, p->sc, vo, slabpf, propf, fout);
            else if (mode == 2) pass23_pfw(p, p->sc, vo, slabpf, propf, fout);
            else                pass23_plain(p, p->sc, vo, slabpf, propf, fout);
        }
    }
    if (mode == 1) _mm_sfence();
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
    p->sstruct = 0;
    p->mode = 0;
    p->slabpf = 0;
    p->scst = 0;
    p->p1pf = 0;
    p->propf = 0;

    for (int j = 0; j < 8; j++)
        for (int k = 0; k < 8; k++) {
            double a = -2.0 * M_PI * (double)(j * k) / 64.0;
            p->tw1[j][k][0] = cos(a);
            p->tw1[j][k][1] = sin(a);
            /* pass-3 lane twiddles: [k2][lane l] = w64^(l*k2) */
            p->tw3r[k][j] = cos(a);
            p->tw3i[k][j] = sin(a);
        }

    p->sc = mmap(NULL, SCTOT, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p->sc != MAP_FAILED) {
        p->sc_is_mmap = 1;
#ifdef MADV_HUGEPAGE
        /* FFT64R_NOHP=1: monitor sweep of the hugepage question (L64_blocked
         * measured +3.3-3.7% FOR hugepages on wallaby in r7; never swept on
         * the node's smaller STLB -- r8 verdict standing ask).              */
        if (!getenv("FFT64R_NOHP"))
            madvise(p->sc, SCTOT, MADV_HUGEPAGE);
#endif
    } else {
        p->sc = aligned_alloc(64, SCTOT);
        p->sc_is_mmap = 0;
        if (!p->sc) { free(p); return NULL; }
    }
    memset(p->sc, 0, SCTOT);
    p->xb = (double *)((char *)p->sc + SCSZ);   /* 68-KB tail of the mapping */

    p->lb = aligned_alloc(64, 64 * 16 * sizeof(double));
    if (!p->lb) { fft3d_destroy(p); return NULL; }
    memset(p->lb, 0, 64 * 16 * sizeof(double));

#if defined(__AVX512F__)
    /* Self-tune (store mode x slab prefetch) at (a clamp of) the real batch
     * size: plain stores win when the working set re-fits the cache across
     * driver repeats (B=1), NT or prefetchw wins once the batch leaves L3
     * (which of the two is machine-dependent: the node rejected NT and kept
     * prefetchw three rounds running at L=36).  Non-baseline picks must beat
     * the baseline (plain, slabpf on) by 2%.                                */
    {
        static const int cand[][4] = /* {struct, mode, slabpf-lead, p1pf} */
            { {0,0,0,0}, {0,0,1,0}, {0,0,2,0},
              {0,1,0,0}, {0,1,1,0}, {0,1,2,0},
              {0,2,0,0}, {0,2,1,0}, {0,2,2,0},
              {1,0,0,0}, {1,1,0,0}, {1,2,0,0},
              /* B=1 only: force the pass-1 next-plane prefetch (r6 gated it
               * to nvol>1 on a wallaby number; the node's L3 is relatively
               * slower, so it gets its own tournament rows there)          */
              {0,0,1,1}, {0,1,1,1}, {0,2,1,1} };
        enum { NC = 15 };
        /* panel_r11: clamp raised 4 -> 8.  B=8 is scored streaming 64 MB from
         * DRAM; a 32-MB arena is half L3-resident on the node and tunes the
         * store mode / prefetch picks in the wrong regime.                   */
        int bt = batch < 8 ? batch : 8;
        int nc = bt == 1 ? 15 : 12;
        double *ti = aligned_alloc(64, (size_t)bt * VOLD * sizeof(double));
        double *to = aligned_alloc(64, (size_t)bt * VOLD * sizeof(double));
        if (ti && to) {
            for (size_t k = 0; k < (size_t)bt * VOLD; k++)
                ti[k] = (double)((k * 2654435761u) & 1023) * 9.765625e-4 - 0.5;
            double best[NC];
            for (int v = 0; v < nc; v++) best[v] = 1e30;
            p->propf = 1;            /* prologues on for the whole grid */
            for (int round = 0; round < 3; round++)
                for (int v = 0; v < nc; v++) {
                    int st = cand[v][0], m = cand[v][1], s = cand[v][2],
                        p1 = cand[v][3];
                    exec_avx512(p, ti, to, bt, st, m, s, 0, p1, 1, 0, 0); /* warm */
                    double t0 = now_s();
                    exec_avx512(p, ti, to, bt, st, m, s, 0, p1, 1, 0, 0);
                    exec_avx512(p, ti, to, bt, st, m, s, 0, p1, 1, 0, 0);
                    double dt = (now_s() - t0) / 2.0;
                    if (dt < best[v]) best[v] = dt;
                }
            int pick = 0;
            for (int v = 1; v < nc; v++)
                if (best[v] < best[pick] && best[v] < 0.98 * best[0]) pick = v;
            p->sstruct = cand[pick][0];
            p->mode    = cand[pick][1];
            p->slabpf  = cand[pick][2];
            p->p1pf    = cand[pick][3];
            /* propf A/B on the picked candidate (fused only; tiled has no
             * prologue).  New mechanism, so it must WIN to stay on.        */
            double tpro[2] = { 1e30, 1e30 };
            if (p->sstruct == 0)
                for (int round = 0; round < 3; round++)
                    for (int pv = 0; pv < 2; pv++) {
                        p->propf = pv;   /* pass 1 reads the plan field */
                        exec_avx512(p, ti, to, bt, 0, p->mode, p->slabpf, 0,
                                    p->p1pf, pv, 0, 0);
                        double t0 = now_s();
                        exec_avx512(p, ti, to, bt, 0, p->mode, p->slabpf, 0,
                                    p->p1pf, pv, 0, 0);
                        exec_avx512(p, ti, to, bt, 0, p->mode, p->slabpf, 0,
                                    p->p1pf, pv, 0, 0);
                        double dt = (now_s() - t0) / 2.0;
                        if (dt < tpro[pv]) tpro[pv] = dt;
                    }
            p->propf = (p->sstruct == 0 && tpro[1] < tpro[0]) ? 1 : 0;
            /* panel_r10: pass-1 SC store-mode twin, A/B'd on the picked
             * candidate (fused only; tiled's pass A has dense rows and its
             * own store path).  The r9 verdict's named residual is the SC
             * store RFOs; scst=1 hides them behind a plane of prefetchw
             * lead, scst=2 (NT) eliminates them and moves the re-read to
             * DRAM.  Non-plain must beat plain by 1% -- it changes traffic,
             * so a noise-level win must not flip it.                        */
            double tsc[3] = { 1e30, 1e30, 1e30 };
            if (p->sstruct == 0) {
                for (int round = 0; round < 3; round++)
                    for (int sv = 0; sv < 3; sv++) {
                        exec_avx512(p, ti, to, bt, 0, p->mode, p->slabpf, sv,
                                    p->p1pf, p->propf, 0, 0);
                        double t0 = now_s();
                        exec_avx512(p, ti, to, bt, 0, p->mode, p->slabpf, sv,
                                    p->p1pf, p->propf, 0, 0);
                        exec_avx512(p, ti, to, bt, 0, p->mode, p->slabpf, sv,
                                    p->p1pf, p->propf, 0, 0);
                        double dt = (now_s() - t0) / 2.0;
                        if (dt < tsc[sv]) tsc[sv] = dt;
                    }
                p->scst = 0;
                for (int sv = 1; sv < 3; sv++)
                    if (tsc[sv] < tsc[p->scst] && tsc[sv] < 0.99 * tsc[0])
                        p->scst = sv;
                /* NT SC stores make slab-0 coldness strictly worse, so the
                 * propf verdict taken under plain stores may flip: re-run
                 * its A/B under the winning store mode.                    */
                if (p->scst != 0) {
                    tpro[0] = tpro[1] = 1e30;
                    for (int round = 0; round < 3; round++)
                        for (int pv = 0; pv < 2; pv++) {
                            p->propf = pv;
                            exec_avx512(p, ti, to, bt, 0, p->mode, p->slabpf,
                                        p->scst, p->p1pf, pv, 0, 0);
                            double t0 = now_s();
                            exec_avx512(p, ti, to, bt, 0, p->mode, p->slabpf,
                                        p->scst, p->p1pf, pv, 0, 0);
                            exec_avx512(p, ti, to, bt, 0, p->mode, p->slabpf,
                                        p->scst, p->p1pf, pv, 0, 0);
                            double dt = (now_s() - t0) / 2.0;
                            if (dt < tpro[pv]) tpro[pv] = dt;
                        }
                    p->propf = tpro[1] < tpro[0] ? 1 : 0;
                }
            }
            /* panel_r11 xb A/B on the settled configuration (fused only):
             * xb=1 keeps SC read-only in pass 2+3 (no 4.5-MB writeback
             * sweep) at the cost of an extra L2-hot 68-KB store target.
             * It moves traffic, so like scst it must win by 1%.  This
             * re-runs r6's slab-buffer rejection ON THE NODE for the first
             * time; wallaby is expected to decline it again (fast L3).
             * Output is bit-identical either way.                          */
            double txb[2] = { 1e30, 1e30 };
            if (p->sstruct == 0) {
                for (int round = 0; round < 3; round++)
                    for (int xv = 0; xv < 2; xv++) {
                        exec_avx512(p, ti, to, bt, 0, p->mode, p->slabpf,
                                    p->scst, p->p1pf, p->propf, xv, 0);
                        double t0 = now_s();
                        exec_avx512(p, ti, to, bt, 0, p->mode, p->slabpf,
                                    p->scst, p->p1pf, p->propf, xv, 0);
                        exec_avx512(p, ti, to, bt, 0, p->mode, p->slabpf,
                                    p->scst, p->p1pf, p->propf, xv, 0);
                        double dt = (now_s() - t0) / 2.0;
                        if (dt < txb[xv]) txb[xv] = dt;
                    }
                p->xb_on = txb[1] < 0.99 * txb[0] ? 1 : 0;
            }
            /* panel_r11 fout A/B on the final configuration: the x-line
             * stage-2 codelet twin with all 16 store feeds FMA-class
             * (bit-identical, zero extra ops -- the r9 store-feeding law
             * at the one L=64 site it can apply to).  Zero-traffic change,
             * so strict win keeps it, like propf.                          */
            double tfo[2] = { 1e30, 1e30 };
            if (p->sstruct == 0) {
                for (int round = 0; round < 3; round++)
                    for (int fv = 0; fv < 2; fv++) {
                        exec_avx512(p, ti, to, bt, 0, p->mode, p->slabpf,
                                    p->scst, p->p1pf, p->propf, p->xb_on, fv);
                        double t0 = now_s();
                        exec_avx512(p, ti, to, bt, 0, p->mode, p->slabpf,
                                    p->scst, p->p1pf, p->propf, p->xb_on, fv);
                        exec_avx512(p, ti, to, bt, 0, p->mode, p->slabpf,
                                    p->scst, p->p1pf, p->propf, p->xb_on, fv);
                        double dt = (now_s() - t0) / 2.0;
                        if (dt < tfo[fv]) tfo[fv] = dt;
                    }
                p->fout = tfo[1] < tfo[0] ? 1 : 0;
            }
            if (getenv("FFT64R_TUNEDBG")) {
                for (int v = 0; v < nc; v++)
                    fprintf(stderr, "tuner %s mode=%d slabpf=%d p1pf=%d : %.1f us/vol%s\n",
                            cand[v][0] ? "tiled" : "fused", cand[v][1], cand[v][2],
                            cand[v][3], best[v] * 1e6 / bt,
                            v == pick ? "  <-- pick" : "");
                if (p->sstruct == 0) {
                    fprintf(stderr, "tuner propf A/B on pick: off %.1f / on %.1f"
                            " us/vol -> propf=%d\n", tpro[0] * 1e6 / bt,
                            tpro[1] * 1e6 / bt, p->propf);
                    fprintf(stderr, "tuner scst A/B on pick: plain %.1f / pfw %.1f"
                            " / nt %.1f us/vol -> scst=%d\n", tsc[0] * 1e6 / bt,
                            tsc[1] * 1e6 / bt, tsc[2] * 1e6 / bt, p->scst);
                    fprintf(stderr, "tuner xb A/B on pick: inplace %.1f / xbuf %.1f"
                            " us/vol -> xb=%d\n", txb[0] * 1e6 / bt,
                            txb[1] * 1e6 / bt, p->xb_on);
                    fprintf(stderr, "tuner fout A/B on pick: add %.1f / fma %.1f"
                            " us/vol -> fout=%d\n", tfo[0] * 1e6 / bt,
                            tfo[1] * 1e6 / bt, p->fout);
                }
            }
        }
        free(ti); free(to);
        {   /* env forcing for the monitor's one-flag experiments */
            const char *e;
            if ((e = getenv("FFT64R_STRUCT"))) p->sstruct = atoi(e);
            if ((e = getenv("FFT64R_MODE")))   p->mode   = atoi(e);
            if ((e = getenv("FFT64R_SLABPF"))) p->slabpf = atoi(e);
            if ((e = getenv("FFT64R_SCPFW")) && atoi(e)) p->scst = 1;
            if ((e = getenv("FFT64R_SCST")))   p->scst   = atoi(e);
            if ((e = getenv("FFT64R_P1PF")))   p->p1pf   = atoi(e);
            if ((e = getenv("FFT64R_PROPF")))  p->propf  = atoi(e);
            if ((e = getenv("FFT64R_XB")))     p->xb_on  = atoi(e);
            if ((e = getenv("FFT64R_FOUT")))   p->fout   = atoi(e);
        }
        snprintf(g_desc, sizeof g_desc,
                 "radix-8^2 per axis, split-complex AVX-512, padded scratch; "
                 "tuner pick[B=%d]=%s-%s%s%d+pro%d+p1%d+sc%d+xb%d+fo%d", batch,
                 p->sstruct ? "tiled" : "fused",
                 p->mode == 1 ? "nt" : p->mode == 2 ? "pfw" : "plain",
                 "+slabpf", p->slabpf, p->propf, p->p1pf, p->scst,
                 p->xb_on, p->fout);
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
    exec_avx512(p, vi, vo, p->B, p->sstruct, p->mode, p->slabpf, p->scst,
                p->p1pf, p->propf, p->xb_on, p->fout);
#else
    exec_scalar(p, vi, vo, p->B);
#endif
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    if (p->sc) {
        if (p->sc_is_mmap) munmap(p->sc, SCTOT);
        else free(p->sc);
    }
    free(p->lb);
    free(p);
}
