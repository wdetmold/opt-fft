/* L64_radix8 -- forward complex-double 3D DFT of a 64^3 cube, batched.
 * MULTICORE round mt_r1.  The phase-1 single-thread kernel (11 rounds, see
 * ../../geom/strategies/L64_radix8.md) is kept verbatim as the per-thread
 * body; this round adds the 32-core layer:
 *
 *   split mode (mt=1): within-volume split, volumes sequential.  Pass 1 is an
 *     `omp for` over the 64 x-planes (each thread's planes write disjoint
 *     x-rows of the SHARED scratch volume), barrier, pass 2+3 an `omp for`
 *     over the 64 ky-slabs (disjoint SC columns in-place, disjoint 16-line
 *     output rows).  Two barriers per volume; every partition boundary is
 *     >= one cache line, so no false sharing anywhere.  The shared SC is
 *     first-touched in create() by the same static map pass 1 uses.
 *   vol mode (mt=2 static, mt=3 dynamic): volume-parallel, zero sync; each
 *     thread runs the tuned serial body on its own NUMA-local scratch
 *     (first-touched by its pinned owner in create()).  dynamic,1 is the
 *     work-stealing twin for batch > team (borrowed from L36_mixedradix
 *     mt_r1: the driver first-touches both caller buffers on the main
 *     thread, so remote-socket threads run volumes slower through UPI and a
 *     static split parks the near half at the join).
 *
 * fft3d_create() races {split, vol-fused, vol-tiled, vol-dyn} x {store mode}
 * (x {p1pf, T=16} at B=1) with the REAL threaded paths at bt = min(B,32)
 * volumes, then A/Bs slabpf and scst on the winner -- same greedy protocol
 * as phase 1, re-timed under 32 threads because every phase-1 store-mode
 * verdict was taken in a single-core traffic regime that no longer exists
 * (32 threads' SC no longer fits L3, so plain-vs-NT flips are expected).
 * The OpenMP pool is spun up (and scratch first-touched) in create();
 * execute() only re-enters the warm pool.
 *
 * Env forcing for the monitor (new this phase): FFT64R_MT=0|1|2|3
 * (serial|split|vol|vol-dyn), FFT64R_T=n (team size, never above the
 * harness's), plus all phase-1 flags below with their old meanings.
 *
 * Round mt_r2: the gang layer is PIPELINED (structure adopted from
 * L64_blocked mt_r1, which beat this kernel 95.7-vs-146.9 us at B=128 on
 * the node with the SAME 8-gangs-of-4 decomposition -- the difference was
 * sync structure, not arithmetic):
 *   mt=5 "gangp": ONE spin barrier per volume instead of two.  Each gang
 *     double-buffers its shared SC between lane 0's and lane 1's scratch
 *     regions (both first-touched by their owners = same socket, no new
 *     memory): volume k uses buffer k&1, so a lane that finishes its
 *     pass-2+3 slice flows straight into the next volume's pass 1 while
 *     gang siblings are still z-lining -- the pass23(v_k) / pass1(v_{k+2})
 *     WAR hazard on buffer k&1 is fenced by the pass-1 barrier of v_{k+1},
 *     which every lane crosses only after its own pass23(v_k).  On the
 *     node the caller's in/out pages all sit on socket 0 (driver first
 *     touch), so socket-1 lanes see long-latency UPI reads with high
 *     variance; the overlap absorbs exactly that straggling.
 *   mt=6 "gangd": gangp + DYNAMIC volume assignment (the L64_blocked mt_r1
 *     "Next" item they worked out but did not ship, with their leader-
 *     overwrite race solved by folding the claim into the barrier): the
 *     LAST-arriving lane claims the gang's next volume from one global
 *     atomic counter and writes it into the barrier line BEFORE the phase
 *     release, so every waiter reads it race-free after the acquire.
 *     Static round-robin makes the whole call wait on the slowest
 *     (UPI-remote) gang; work-conserving claims rebalance volumes toward
 *     the socket that owns the caller's pages.
 *   mt=4 keeps the mt_r1 two-barrier gang verbatim (env-forcible control).
 * FFT64R_MT=0..6, FFT64R_GSZ as before.
 *
 * Round mt_r3 (driven by the mt_r2 node picks: B=1 split-T16-pfw lost to
 * L64_blocked's one-socket slab split 136.2-vs-127.0; B=8 legacy gang-g8-pfw
 * lost to mkl 90.6-vs-73.0 while L64_blocked's G=2/G=4 nth=32 NT groups made
 * 76.7; B=128 won at 69.5):
 *   1. scs16 -- a second SHARED split-mode SC, first-touched by the T16
 *      static map (threads 0..15 = the node's socket 0 under close binding).
 *      The old shared SC is touched by the T32 map, so when the node picks
 *      split-T16 at B=1 HALF its pages sit on socket 1 and every volume pays
 *      UPI for half the scratch writes and reads back.  With scs16 the whole
 *      B=1 path (in/out driver-touched on socket 0, SC, lb) is socket-local.
 *   2. legacy gang gsz=16 rows at batch (one socket per volume, ONE live
 *      4.46-MB SC per socket -- the best L3-residency shape; adopted from
 *      L64_blocked mt_r2, whose node B=8 picks were G=2/G=4 at nth=32).
 *   3. pass-1 input-prefetch A/B on the batch winner: p1pf=2 now means
 *      "force the next-plane prefetch OFF even when streaming" -- mt_r1
 *      measured forced p1 prefetch losing 8% in gang mode at B=8 on wallaby,
 *      and the node's g8 pick runs with it ON (nvol > ngang), never raced.
 *   4. legacy gang skips the trailing barrier of each gang's last volume
 *      (the parallel-region join orders it) -- one less barrier per call.
 * FFT64R_P1PF=0|1|2 (2 = force off); everything else keeps its meaning.
 */
/* Phase-1 header, kept because the per-thread body is unchanged:
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
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/mman.h>
#ifdef _OPENMP
#include <omp.h>
#endif

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

/* Per-thread working set: SC volume + xb buffer + 8-KiB line buffer, the
 * whole region rounded up to a 2-MiB multiple so every thread's slice is
 * hugepage-aligned and 2 MiB away from its neighbours (no false sharing at
 * any granularity, THP-friendly).                                          */
#define NT_MAX 32
#define LBSZ   ((size_t)64 * 16 * sizeof(double))
#define WREG   (((SCTOT + LBSZ) + ((size_t)2 << 20) - 1) & ~((((size_t)2) << 20) - 1))

struct mtws { double *sc, *xb, *lb; };

/* Per-gang sense-free spin barrier (central counter + phase).  Reusable
 * across execute calls without reset: phase only ever advances.  One cache
 * line per gang so gangs never share barrier lines.  nextvol (mt_r2) is the
 * barrier's payload: the releasing lane stores the gang's next volume there
 * before the phase release, so waiters read it race-free after the acquire. */
struct gbar { int arrive; int phase; int nextvol; char pad[52]; }
    __attribute__((aligned(64)));

struct fft3d_plan {
    int L, B;
    int mt;                          /* 0 ser, 1 split, 2 vol, 3 vdyn, 4 gang,
                                        5 gangp (pipelined), 6 gangd (+dyn)   */
    int gsz;                         /* gang size (mt=4): threads per volume   */
    int tsz;                         /* team size (<= harness's thread count)  */
    int nthr;                        /* pool size = per-thread scratch count   */
    int sstruct;                     /* 0 fused (2 sweeps), 1 tiled (slab + z sweep) */
    int mode;                        /* final stores: 0 plain, 1 NT, 2 plain+prefetchw */
    int slabpf;                      /* fused: T1-prefetch slab ky+lead during z-lines */
    int scst;                        /* pass-1 SC store mode: 0 plain, 1 +prefetchw, 2 NT */
    int p1pf;                        /* force pass-1 next-plane prefetch even at B=1 */
    int propf;                       /* prologue: prefetch slab 0 (+lead) / plane 0 */
    int xb_on;                       /* x-FFT output -> compact 68-KB buffer (SC read-only) */
    int fout;                        /* x-line stage-2 codelet: all-FMA store feeds */
    double *scs;                     /* SHARED split-mode scratch volume       */
    double *scs16;                   /* T16 twin, first-touched by the T16 map */
    struct mtws ws[NT_MAX];          /* per-thread scratch (NUMA-local)        */
    struct gbar gbars[NT_MAX / 2];   /* one barrier per gang (mt=4/5/6)        */
    int gctr __attribute__((aligned(64)));  /* mt=6 global next-volume counter */
    char gctrpad[60];                /* own line: claimed once per volume      */
    struct gbar fb[NT_MAX + 1];      /* mt=7 flat barrier: flag line per      */
                                     /* thread + one release line at [NT_MAX] */
    void *basemap; size_t basesz;    /* one mapping holds scs + all ws slices  */
    int base_is_mmap;
    double tser, tpick;              /* tuner: serial ref / picked, s per call */
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
static void NAME(const struct fft3d_plan *p, const double *in, double *sc,    \
                 int x0, int x1, double *lb, int propf)                       \
{                                                                             \
    const __m512i IEV = _mm512_setr_epi64(0,2,4,6,8,10,12,14);                \
    const __m512i IOD = _mm512_setr_epi64(1,3,5,7,9,11,13,15);                \
    const VT C = _mm512_set1_pd(M_SQRT1_2);                                   \
    if (PF && propf && x0 == 0)                                               \
        /* plane-0 prologue: the next-plane prefetch below covers planes     \
         * 1..63 (and, at batch, the next volume's plane 0), so plane 0 of   \
         * the first volume is the one plane always demand-missed.  1024     \
         * T1 lines, ~0.15 us of issue -- worst case a wash at batch.        */\
        for (int ln = 0; ln < 1024; ln++)                                     \
            _mm_prefetch((const char *)in + (size_t)ln * 64, _MM_HINT_T1);    \
    for (int x = x0; x < x1; x++) {                                           \
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
                 int slabpf, int propf, int fout, int ky0, int ky1,           \
                 double *lb, double *xbuf)                                    \
{                                                                             \
    const __m512i ILO = _mm512_setr_epi64(0,8,1,9,4,12,5,13);                 \
    const __m512i IHI = _mm512_setr_epi64(2,10,3,11,6,14,7,15);               \
    const VT C = _mm512_set1_pd(M_SQRT1_2);                                   \
    /* prologue (r9): slabpf prefetches slab ky+lead while z-lining ky, so   \
     * the FIRST slab(s) of every chunk are always demand-missed to L3 --    \
     * pass 1 wrote slab ky0's low-x lines a whole SC sweep ago.  Pull the   \
     * first max(1,lead) slabs (1024 T1 lines each) up front.                */\
    if (propf) {                                                              \
        const int npro = slabpf > 0 ? slabpf : 1;                             \
        for (int k = ky0; k < ky0 + npro && k < 64; k++) {                    \
            const double *pp = sc + (size_t)k * SCKS;                         \
            for (int x = 0; x < 64; x++) {                                    \
                PFS8(pp + (size_t)x * SCXS, 0);                               \
                PFS8(pp + (size_t)x * SCXS, 512);                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    for (int ky = ky0; ky < ky1; ky++) {                                      \
        double *sy = sc + (size_t)ky * SCKS;                                  \
        for (int zb = 0; zb < 8; zb++) {                                      \
            if (XB) { if (fout) xline_xb_f(p, sy + zb*16, xbuf + zb*16, lb);  \
                      else      xline_xb  (p, sy + zb*16, xbuf + zb*16, lb); }\
            else    { if (fout) xline_sc_f(p, sy + zb*16, 0, lb);             \
                      else      xline_sc  (p, sy + zb*16, 0, lb); }           \
        }                                                                     \
        const double *pfs = sy + (size_t)slabpf * SCKS;                       \
        const int dopf = slabpf && ky < 64 - slabpf;                          \
        for (int kx = 0; kx < 64; kx++) {                                     \
            const double *s = XB ? xbuf + (size_t)kx * XBS                    \
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
                 int zb, double *lb)                                          \
{                                                                             \
    const __m512i IEV = _mm512_setr_epi64(0,2,4,6,8,10,12,14);                \
    const __m512i IOD = _mm512_setr_epi64(1,3,5,7,9,11,13,15);                \
    const VT C = _mm512_set1_pd(M_SQRT1_2);                                   \
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

/* ---------------------------- execution layer --------------------------- */

typedef void (*p1_fn)(const struct fft3d_plan *, const double *, double *,
                      int, int, double *, int);
typedef void (*p23_fn)(const struct fft3d_plan *, double *, double *,
                       int, int, int, int, int, double *, double *);

static p1_fn p1_pick(int stream, int scst)
{
    static const p1_fn tab[2][3] = {
        { pass1_plain, pass1_plain_w, pass1_plain_nt },
        { pass1_pf,    pass1_pf_w,    pass1_pf_nt } };
    return tab[stream ? 1 : 0][scst];
}

static p23_fn p23_pick(int mode, int xb)
{
    static const p23_fn tab[2][3] = {
        { pass23_plain,    pass23_nt,    pass23_pfw },
        { pass23_plain_xb, pass23_nt_xb, pass23_pfw_xb } };
    return tab[xb ? 1 : 0][mode];
}

/* One volume, serial, on the given scratch -- the phase-1 body verbatim.   */
static void exec_one(const struct fft3d_plan *p, const double *vi, double *vo,
                     double *sc, double *lb, double *xbuf, int stream,
                     int sstruct, int mode, int slabpf, int scst, int p1pf,
                     int propf, int xb, int fout)
{
    if (p1pf == 2) stream = 0;       /* mt_r3: force input prefetch OFF */
    if (sstruct == 1) {
        for (int zb = 0; zb < 8; zb++) {
            double *slab = sc + (size_t)zb * TSLAB;
            if (stream) passA_pf(p, vi, slab, zb, lb);
            else        passA_plain(p, vi, slab, zb, lb);
        }
        if (mode == 1)      passB_nt(p, sc, vo);
        else if (mode == 2) passB_pfw(p, sc, vo);
        else                passB_plain(p, sc, vo);
        return;
    }
    p1_pick(stream || (p1pf == 1), scst)(p, vi, sc, 0, 64, lb, propf);
    /* NT SC stores: drain the WC buffers before pass 2 reads the same
     * lines back (same-core data dependence is architecturally safe,
     * but a straggling half-filled buffer would flush mid-pass-2)      */
    if (scst == 2) _mm_sfence();
    p23_pick(mode, xb)(p, sc, vo, slabpf, propf, fout, 0, 64, lb, xbuf);
}

static void exec_serial(const struct fft3d_plan *p, const double *in, double *out,
                        int nvol, int sstruct, int mode, int slabpf, int scst,
                        int p1pf, int propf, int xb, int fout)
{
    const struct mtws *w = &p->ws[0];
    for (int b = 0; b < nvol; b++)
        exec_one(p, in + (size_t)b * VOLD, out + (size_t)b * VOLD,
                 w->sc, w->lb, w->xb, nvol > 1, sstruct, mode, slabpf, scst,
                 p1pf, propf, xb, fout);
    if (mode == 1) _mm_sfence();
}

#ifdef _OPENMP
/* split mode: within-volume split, volumes sequential.  Pass 1 statically
 * over x-plane pairs (each pair writes its own SCXS rows of the SHARED sc),
 * sfence-if-NT + barrier, pass 2+3 statically over ky pairs (in-place SC
 * columns + disjoint 16-line output rows), barrier before the next volume
 * overwrites SC.  Chunks of 2 x/ky per iteration = 32 units, one per thread
 * at T=32; the create-time first touch replays the same static map.        */
static void exec_split(const struct fft3d_plan *p, const double *in, double *out,
                       int nvol, int tsz, int mode, int slabpf, int scst,
                       int p1pf, int propf, int xb, int fout)
{
    /* mt_r3: the T16 team uses the SC whose pages the T16 map first-touched
     * (all on the socket that owns the caller's buffers); the T32-mapped SC
     * has half its pages on the other socket.                               */
    double *sc = (tsz == 16 && p->scs16) ? p->scs16 : p->scs;
    const int stream = (p1pf == 2) ? 0 : ((nvol > 1) || p1pf);
    p1_fn p1 = p1_pick(stream, scst);
    p23_fn p23 = p23_pick(mode, xb);
#pragma omp parallel num_threads(tsz)
    {
        const int t = omp_get_thread_num();
        double *lb = p->ws[t].lb, *xbuf = p->ws[t].xb;
        for (int b = 0; b < nvol; b++) {
            const double *vi = in + (size_t)b * VOLD;
            double *vo = out + (size_t)b * VOLD;
#pragma omp for schedule(static) nowait
            for (int x = 0; x < 64; x += 2)
                p1(p, vi, sc, x, x + 2, lb, propf);
            if (scst == 2) _mm_sfence();
#pragma omp barrier
#pragma omp for schedule(static) nowait
            for (int ky = 0; ky < 64; ky += 2)
                p23(p, sc, vo, slabpf, propf, fout, ky, ky + 2, lb, xbuf);
            if (mode == 1) _mm_sfence();
            if (b < nvol - 1) {   /* last volume: the region join orders it */
#pragma omp barrier
            }
        }
    }
}

/* mt=7 "splitf" (mt_r2): the split decomposition with a FLAT arrival-flag
 * spin barrier instead of GOMP's -- borrowed from L17_winograd mt_r1 via
 * L36_pencilfused mt_r1 (their measured fix for barrier latency when the
 * per-volume work is tens of us): each arriver writes its OWN padded line
 * (misses overlap under thread 0's scan, no shared-RFO serialization --
 * exactly the failure L64_blocked measured in the 32-wide CENTRAL counter
 * barrier, 89.6 vs 69.5 us), thread 0 publishes one release word.  Epochs
 * persist in the plan across calls; each thread re-reads its own flag at
 * region entry, so no reset is ever needed.  Static ranges are computed by
 * hand (64 % tsz == 0 for the raced teams), and the trailing barrier of the
 * LAST volume is dropped -- the parallel-region join already orders it.    */
static inline void flat_barrier(struct gbar *fb, int t, int tsz, int ep)
{
    __atomic_store_n(&fb[t].phase, ep, __ATOMIC_RELEASE);
    if (t == 0) {
        for (int i = 1; i < tsz; i++)
            while (__atomic_load_n(&fb[i].phase, __ATOMIC_ACQUIRE) < ep)
                _mm_pause();
        __atomic_store_n(&fb[NT_MAX].phase, ep, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&fb[NT_MAX].phase, __ATOMIC_ACQUIRE) < ep)
            _mm_pause();
    }
}

static void exec_splitflat(struct fft3d_plan *p, const double *in, double *out,
                           int nvol, int tsz, int mode, int slabpf, int scst,
                           int p1pf, int propf, int xb, int fout)
{
    double *sc = (tsz == 16 && p->scs16) ? p->scs16 : p->scs;
    const int stream = (p1pf == 2) ? 0 : ((nvol > 1) || p1pf);
    p1_fn p1 = p1_pick(stream, scst);
    p23_fn p23 = p23_pick(mode, xb);
#pragma omp parallel num_threads(tsz)
    {
        const int t = omp_get_thread_num();
        double *lb = p->ws[t].lb, *xbuf = p->ws[t].xb;
        const int c0 = t * 64 / tsz, c1 = (t + 1) * 64 / tsz;
        /* Seed the epoch from the RELEASE line, not the thread's own flag:
         * the release value is the global maximum and identical for every
         * entering thread, so a T16 call followed by a T32 call (the tuner
         * does exactly this) cannot leave threads 16..31 on stale epochs --
         * seeding from own flags livelocked there (found the hard way).    */
        int ep = __atomic_load_n(&p->fb[NT_MAX].phase, __ATOMIC_RELAXED);
        for (int b = 0; b < nvol; b++) {
            const double *vi = in + (size_t)b * VOLD;
            double *vo = out + (size_t)b * VOLD;
            p1(p, vi, sc, c0, c1, lb, propf);
            if (scst == 2) _mm_sfence();
            flat_barrier(p->fb, t, tsz, ++ep);
            p23(p, sc, vo, slabpf, propf, fout, c0, c1, lb, xbuf);
            if (mode == 1) _mm_sfence();
            if (b < nvol - 1)          /* last volume: the join orders it */
                flat_barrier(p->fb, t, tsz, ++ep);
        }
    }
}

/* vol mode: volume-parallel on per-thread NUMA-local scratch, zero sync.
 * dyn=1 uses dynamic,1 (work stealing against the socket-0 page asymmetry
 * of the caller's buffers -- L36_mixedradix mt_r1's argument).             */
static void exec_volpar(const struct fft3d_plan *p, const double *in, double *out,
                        int nvol, int tsz, int dyn, int sstruct, int mode,
                        int slabpf, int scst, int p1pf, int propf, int xb,
                        int fout)
{
#pragma omp parallel num_threads(tsz)
    {
        const struct mtws *w = &p->ws[omp_get_thread_num()];
        if (dyn) {
#pragma omp for schedule(dynamic, 1) nowait
            for (int b = 0; b < nvol; b++)
                exec_one(p, in + (size_t)b * VOLD, out + (size_t)b * VOLD,
                         w->sc, w->lb, w->xb, p1pf != 2, sstruct, mode,
                         slabpf, scst, p1pf, propf, xb, fout);
        } else {
#pragma omp for schedule(static) nowait
            for (int b = 0; b < nvol; b++)
                exec_one(p, in + (size_t)b * VOLD, out + (size_t)b * VOLD,
                         w->sc, w->lb, w->xb, p1pf != 2, sstruct, mode,
                         slabpf, scst, p1pf, propf, xb, fout);
        }
        if (mode == 1) _mm_sfence();
    }
}
/* gang mode: the team splits into tsz/gsz gangs of gsz adjacent threads
 * (adjacent = same socket under PROC_BIND=close whenever gsz divides the
 * socket size).  Volumes round-robin over gangs; within a volume each lane
 * takes 64/gsz x-planes then 64/gsz ky-slabs, synchronised by a per-gang
 * spin barrier (~100 ns) instead of a 32-thread OMP barrier.  The gang's
 * shared SC is lane 0's per-thread region: one socket owns it entirely, and
 * only gsz cores ever exchange a volume's SC lines -- the cross-core
 * transpose traffic that makes the 32-thread split stop scaling at small
 * volumes drops by tsz/gsz.  (Direction borrowed from L36_mixedradix mt_r1
 * "Next" item 3: per-socket gangs for B in [2,32).)                         */
static inline void gang_barrier(struct gbar *b, int gsz)
{
    int ph = __atomic_load_n(&b->phase, __ATOMIC_RELAXED);
    if (__atomic_add_fetch(&b->arrive, 1, __ATOMIC_ACQ_REL) == gsz) {
        __atomic_store_n(&b->arrive, 0, __ATOMIC_RELAXED);
        __atomic_fetch_add(&b->phase, 1, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&b->phase, __ATOMIC_ACQUIRE) == ph)
            _mm_pause();
    }
}

static void exec_gang(struct fft3d_plan *p, const double *in, double *out,
                      int nvol, int tsz, int gsz, int mode, int slabpf,
                      int scst, int p1pf, int propf, int xb, int fout)
{
    const int ngang = tsz / gsz;
    const int xch = 64 / gsz;                 /* planes / kys per lane */
    p1_fn p1 = p1_pick(p1pf == 2 ? 0 : (nvol > ngang || p1pf), scst);
    p23_fn p23 = p23_pick(mode, xb);
#pragma omp parallel num_threads(ngang * gsz)
    {
        const int t = omp_get_thread_num();
        const int g = t / gsz, l = t % gsz;
        double *sc = p->ws[g * gsz].sc;       /* gang-shared, socket-local */
        double *lb = p->ws[t].lb, *xbuf = p->ws[t].xb;
        struct gbar *gb = &p->gbars[g];
        for (int b = g; b < nvol; b += ngang) {
            const double *vi = in + (size_t)b * VOLD;
            double *vo = out + (size_t)b * VOLD;
            p1(p, vi, sc, l * xch, (l + 1) * xch, lb, propf);
            if (scst == 2) _mm_sfence();
            gang_barrier(gb, gsz);
            p23(p, sc, vo, slabpf, propf, fout, l * xch, (l + 1) * xch,
                lb, xbuf);
            if (mode == 1) _mm_sfence();
            if (b + ngang < nvol)  /* last volume: the region join orders it */
                gang_barrier(gb, gsz);
        }
    }
}

/* mt_r2 pipelined gang (structure from L64_blocked mt_r1 eng=2): ONE barrier
 * per volume, gang SC double-buffered between lane 0's and lane 1's regions
 * (volume k -> buffer k&1).  Safety of the single barrier: when any lane
 * starts pass1(v_{k+2}) it has crossed the barrier of v_{k+1}, which every
 * lane reaches only after finishing pass23(v_k) -- the last reader of buffer
 * k&1 -- so the WAR hazard is ordered by that barrier's release/acquire.
 * The barrier also carries the gang's NEXT volume as payload: static = v +
 * ngang (all lanes compute the releaser's value identically), dynamic = one
 * fetch_add on p->gctr by the releasing lane only.  Gangs whose next claim
 * is >= nvol simply leave the loop; barriers are gang-local and control flow
 * is lane-uniform, so no deadlock is possible.                              */
static inline int gang_barrier_next(struct gbar *b, int gsz, int *ctr,
                                    int vstat, int dyn)
{
    int ph = __atomic_load_n(&b->phase, __ATOMIC_RELAXED);
    if (__atomic_add_fetch(&b->arrive, 1, __ATOMIC_ACQ_REL) == gsz) {
        __atomic_store_n(&b->arrive, 0, __ATOMIC_RELAXED);
        int nv = dyn ? __atomic_fetch_add(ctr, 1, __ATOMIC_RELAXED) : vstat;
        b->nextvol = nv;              /* published by the phase release below */
        __atomic_fetch_add(&b->phase, 1, __ATOMIC_RELEASE);
        return nv;
    }
    while (__atomic_load_n(&b->phase, __ATOMIC_ACQUIRE) == ph)
        _mm_pause();
    return b->nextvol;                /* ordered after the acquire above */
}

static void exec_gangpipe(struct fft3d_plan *p, const double *in, double *out,
                          int nvol, int tsz, int gsz, int dyn, int mode,
                          int slabpf, int scst, int p1pf, int propf, int xb,
                          int fout)
{
    const int ngang = tsz / gsz;
    const int xch = 64 / gsz;                 /* planes / kys per lane */
    p1_fn p1 = p1_pick(p1pf == 2 ? 0 : (nvol > ngang || p1pf), scst);
    p23_fn p23 = p23_pick(mode, xb);
    p->gctr = ngang;                          /* first dynamic claim */
#pragma omp parallel num_threads(ngang * gsz)
    {
        const int t = omp_get_thread_num();
        const int g = t / gsz, l = t % gsz;
        /* the two gang-shared SC buffers: lane 0's and lane 1's regions,
         * both first-touched by their (same-socket) owners in create()   */
        double *scb[2] = { p->ws[g * gsz].sc, p->ws[g * gsz + 1].sc };
        double *lb = p->ws[t].lb, *xbuf = p->ws[t].xb;
        struct gbar *gb = &p->gbars[g];
        int v = g, k = 0;
        while (v < nvol) {
            const double *vi = in + (size_t)v * VOLD;
            double *vo = out + (size_t)v * VOLD;
            double *sc = scb[k & 1];
            p1(p, vi, sc, l * xch, (l + 1) * xch, lb, propf);
            if (scst == 2) _mm_sfence();
            int nv = gang_barrier_next(gb, gsz, &p->gctr, v + ngang, dyn);
            p23(p, sc, vo, slabpf, propf, fout, l * xch, (l + 1) * xch,
                lb, xbuf);
            if (mode == 1) _mm_sfence();
            v = nv; k++;
        }
    }
}
#endif /* _OPENMP */

static void exec_mt(struct fft3d_plan *p, const double *in, double *out,
                    int nvol, int mt, int tsz, int gsz, int sstruct, int mode,
                    int slabpf, int scst, int p1pf, int propf, int xb, int fout)
{
#ifdef _OPENMP
    if (mt == 1 && tsz > 1) {
        exec_split(p, in, out, nvol, tsz, mode, slabpf, scst, p1pf, propf,
                   xb, fout);
        return;
    }
    if ((mt == 2 || mt == 3) && tsz > 1) {
        exec_volpar(p, in, out, nvol, tsz, mt == 3, sstruct, mode, slabpf,
                    scst, p1pf, propf, xb, fout);
        return;
    }
    if (mt == 4 && tsz > 1 && gsz >= 1 && gsz < tsz
        && tsz % gsz == 0 && 64 % gsz == 0) {
        exec_gang(p, in, out, nvol, tsz, gsz, mode, slabpf, scst, p1pf,
                  propf, xb, fout);
        return;
    }
    if ((mt == 5 || mt == 6) && tsz > 1 && gsz >= 2 && gsz < tsz
        && tsz % gsz == 0 && 64 % gsz == 0) {
        exec_gangpipe(p, in, out, nvol, tsz, gsz, mt == 6, mode, slabpf,
                      scst, p1pf, propf, xb, fout);
        return;
    }
    if (mt == 7 && tsz > 1 && 64 % tsz == 0) {
        exec_splitflat(p, in, out, nvol, tsz, mode, slabpf, scst, p1pf,
                       propf, xb, fout);
        return;
    }
#else
    (void)mt; (void)tsz; (void)gsz;
#endif
    exec_serial(p, in, out, nvol, sstruct, mode, slabpf, scst, p1pf, propf,
                xb, fout);
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

    for (int j = 0; j < 8; j++)
        for (int k = 0; k < 8; k++) {
            double a = -2.0 * M_PI * (double)(j * k) / 64.0;
            p->tw1[j][k][0] = cos(a);
            p->tw1[j][k][1] = sin(a);
            /* pass-3 lane twiddles: [k2][lane l] = w64^(l*k2) */
            p->tw3r[k][j] = cos(a);
            p->tw3i[k][j] = sin(a);
        }

#ifdef _OPENMP
    /* take exactly what the harness gives, never more */
    int nthr = omp_get_max_threads();
    if (nthr > NT_MAX) nthr = NT_MAX;
    if (nthr < 1) nthr = 1;
#else
    int nthr = 1;
#endif
    p->nthr = nthr;
    p->tsz  = nthr;
    p->mt   = 0;

    /* One mapping holds the shared split-mode SC plus nthr per-thread
     * regions, every slice a 2-MiB-aligned WREG.  Placement (= first touch)
     * happens below, per owning thread.                                     */
    {
        /* mt_r3: +1 slice for the T16-mapped shared SC (scs16) when the team
         * can be split 16-wide (i.e. spans two sockets on the node).        */
        int n16 = nthr > 16 ? 1 : 0;
        size_t need = WREG * (size_t)(nthr + 1 + n16) + ((size_t)2 << 20);
        p->basesz = need;
        p->basemap = mmap(NULL, need, PROT_READ | PROT_WRITE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p->basemap != MAP_FAILED) {
            p->base_is_mmap = 1;
#ifdef MADV_HUGEPAGE
            /* FFT64R_NOHP=1: monitor sweep of the hugepage question */
            if (!getenv("FFT64R_NOHP"))
                madvise(p->basemap, need, MADV_HUGEPAGE);
#endif
        } else {
            p->basemap = aligned_alloc(4096, need);
            p->base_is_mmap = 0;
            if (!p->basemap) { free(p); return NULL; }
        }
        char *base = (char *)(((uintptr_t)p->basemap + (((size_t)2 << 20) - 1))
                              & ~(uintptr_t)((((size_t)2) << 20) - 1));
        p->scs = (double *)base;
        for (int t = 0; t < nthr; t++) {
            char *r = base + WREG * (size_t)(t + 1);
            p->ws[t].sc = (double *)r;
            p->ws[t].xb = (double *)(r + SCSZ);
            p->ws[t].lb = (double *)(r + SCSZ + XBSZ);
        }
        p->scs16 = n16 ? (double *)(base + WREG * (size_t)(nthr + 1)) : NULL;
#ifdef _OPENMP
        /* Spin the pool up NOW (execute only re-enters the warm team) and
         * first-touch: each thread its own region (NUMA-local under
         * PROC_BIND=close), then the shared SC by the same static x-map
         * split-mode pass 1 uses, so its pages sit with their writers.     */
#pragma omp parallel num_threads(nthr)
        {
            memset(base + WREG * (size_t)(omp_get_thread_num() + 1), 0, WREG);
        }
#pragma omp parallel num_threads(nthr)
        {
#pragma omp for schedule(static)
            for (int x = 0; x < 64; x += 2)
                memset(p->scs + (size_t)x * SCXS, 0,
                       2 * SCXS * sizeof(double));
        }
        memset((char *)p->scs + SCSZ_FUSED, 0, WREG - SCSZ_FUSED);
        if (p->scs16) {
            /* first-touch by the T16 static map exec_split/splitflat use at
             * tsz=16 (thread t owns planes 4t..4t+3): every page lands on
             * the socket of threads 0..15 = the socket that also owns the
             * driver-touched caller buffers on the node                     */
#pragma omp parallel num_threads(16)
            {
                const int t = omp_get_thread_num();
                memset(p->scs16 + (size_t)(4 * t) * SCXS, 0,
                       4 * SCXS * sizeof(double));
            }
            memset((char *)p->scs16 + SCSZ_FUSED, 0, WREG - SCSZ_FUSED);
        }
#else
        memset(base, 0, WREG * (size_t)(nthr + 1 + n16));
#endif
    }

#if defined(__AVX512F__)
    /* Self-tune with the REAL threaded paths at bt = min(B,32) volumes: the
     * scheme (split / vol-static / vol-tiled / vol-dyn / serial) x the final
     * store mode, then slabpf and scst A/Bs on the winner.  Every phase-1
     * store-mode verdict was single-core; 32 threads change the traffic
     * regime (SC no longer fits L3), so everything is re-raced here.  The
     * arena is filled SERIALLY on purpose: that reproduces the driver's
     * main-thread first touch (all caller pages on one socket on the node).
     * Non-first candidates must beat cand[0] (split, plain stores) by 2%;
     * the slabpf/scst incumbents defend at 1%.                              */
    {
        struct cnd { int mt, tsz, gsz, sstruct, mode, p1pf; };
        struct cnd cv[40];
        int nc = 0;
        int bt = batch < 32 ? batch : 32;
        /* defaults if the tuner cannot run: phase-1 node picks + batch rule */
        p->sstruct = 0; p->mode = 2; p->slabpf = 1; p->scst = 0;
        p->p1pf = (batch == 1); p->propf = 0; p->xb_on = 0; p->fout = 0;
        p->mt = (nthr > 1) ? ((batch >= 2 * nthr) ? 2 : 1) : 0;
        p->gsz = 0;
        if (nthr > 1) {
            if (batch == 1) {
                for (int m = 0; m < 3; m++)
                    cv[nc++] = (struct cnd){1, nthr, 0, 0, m, 0};
                for (int m = 0; m < 3; m++)
                    cv[nc++] = (struct cnd){1, nthr, 0, 0, m, 1};
                /* one socket of the node (PLACES=cores, close => threads
                 * 0..15 are socket 0 there): the L36 mt_r1 prediction that
                 * the cross-socket barrier may beat 16 idle cores at B=1  */
                if (nthr > 16)
                    for (int m = 0; m < 3; m++)
                        cv[nc++] = (struct cnd){1, 16, 0, 0, m, 1};
                /* flat-barrier split (mt=7): B=1 is 2-barriers-per-30us of
                 * arithmetic, so barrier latency is on the critical path  */
                for (int m = 0; m < 3; m += 2) {
                    if (64 % nthr == 0)
                        cv[nc++] = (struct cnd){7, nthr, 0, 0, m, 1};
                    if (nthr > 16)
                        cv[nc++] = (struct cnd){7, 16, 0, 0, m, 1};
                }
                cv[nc++] = (struct cnd){0, 1, 0, 0, 2, 1};  /* serial ref */
            } else {
                for (int m = 0; m < 3; m++)
                    cv[nc++] = (struct cnd){1, nthr, 0, 0, m, 0}; /* split  */
                for (int m = 0; m < 3; m++)
                    cv[nc++] = (struct cnd){2, nthr, 0, 0, m, 0}; /* vol    */
                for (int m = 0; m < 3; m++)
                    cv[nc++] = (struct cnd){2, nthr, 0, 1, m, 0}; /* tiled  */
                /* gangs of 4/8/16: one volume per gang of adjacent threads,
                 * gang-local spin barriers -- the middle ground between
                 * split (32 threads sweep one volume: cross-core transpose
                 * traffic) and vol (only B threads busy at B < 32).
                 * mt_r2: raced PIPELINED (mt=5, one barrier/volume, double-
                 * buffered SC) and, when a gang gets more than one volume,
                 * DYNAMIC (mt=6) -- static round-robin waits on the slowest
                 * UPI-remote gang.  The mt_r1 two-barrier gang (mt=4) keeps
                 * two insurance rows in case double-buffering thrashes the
                 * node's smaller L3.                                       */
                for (int gz = 4; gz <= 16; gz *= 2)
                    if (nthr % gz == 0 && nthr / gz >= 2 && bt >= nthr / gz) {
                        for (int m = 0; m < 3; m++)
                            cv[nc++] = (struct cnd){5, nthr, gz, 0, m, 0};
                        if (bt > nthr / gz)
                            for (int m = 0; m < 3; m++)
                                cv[nc++] = (struct cnd){6, nthr, gz, 0, m, 0};
                        /* legacy two-barrier gang, nt + pfw.  mt_r3: g16
                         * joins (one socket per volume, ONE live 4.46-MB SC
                         * per socket -- L64_blocked mt_r2's winning node
                         * B=8 shape was G=2/G=4 at nth=32); the node picked
                         * legacy over gangp/gangd in every mt_r2 cell.     */
                        for (int m = 1; m < 3; m++)
                            cv[nc++] = (struct cnd){4, nthr, gz, 0, m, 0};
                    }
                if (batch > nthr)
                    for (int m = 0; m < 3; m++)
                        cv[nc++] = (struct cnd){3, nthr, 0, 0, m, 0}; /* dyn */
            }
        } else {
            cv[nc++] = (struct cnd){0, 1, 0, 0, 0, batch == 1};
            cv[nc++] = (struct cnd){0, 1, 0, 0, 1, batch == 1};
            cv[nc++] = (struct cnd){0, 1, 0, 0, 2, batch == 1};
        }
        double *ti = aligned_alloc(64, (size_t)bt * VOLD * sizeof(double));
        double *to = aligned_alloc(64, (size_t)bt * VOLD * sizeof(double));
        double best[40], tsl[3] = {1e30, 1e30, 1e30},
               tsc[3] = {1e30, 1e30, 1e30}, tpp[2] = {1e30, 1e30};
        int pick = 0;
        if (ti && to) {
            for (size_t k = 0; k < (size_t)bt * VOLD; k++)
                ti[k] = (double)((k * 2654435761u) & 1023) * 9.765625e-4 - 0.5;
            for (int v = 0; v < nc; v++) best[v] = 1e30;
            for (int round = 0; round < 3; round++)
                for (int v = 0; v < nc; v++) {
                    exec_mt(p, ti, to, bt, cv[v].mt, cv[v].tsz, cv[v].gsz,
                            cv[v].sstruct, cv[v].mode, 1, 0, cv[v].p1pf,
                            0, 0, 0);                                /* warm */
                    double t0 = now_s();
                    exec_mt(p, ti, to, bt, cv[v].mt, cv[v].tsz, cv[v].gsz,
                            cv[v].sstruct, cv[v].mode, 1, 0, cv[v].p1pf,
                            0, 0, 0);
                    exec_mt(p, ti, to, bt, cv[v].mt, cv[v].tsz, cv[v].gsz,
                            cv[v].sstruct, cv[v].mode, 1, 0, cv[v].p1pf,
                            0, 0, 0);
                    double dt = (now_s() - t0) / 2.0;
                    if (dt < best[v]) best[v] = dt;
                }
            for (int v = 1; v < nc; v++)
                if (best[v] < best[pick] && best[v] < 0.98 * best[0]) pick = v;
            p->mt      = cv[pick].mt;
            p->tsz     = cv[pick].tsz;
            p->gsz     = cv[pick].gsz;
            p->sstruct = cv[pick].sstruct;
            p->mode    = cv[pick].mode;
            p->p1pf    = cv[pick].p1pf;
            p->tpick   = best[pick] / bt;
            /* slabpf A/B on the winner (fused pass 2+3 only).  Incumbent 1
             * (the node kept it three rounds in phase 1) defends at 1%.    */
            if (p->sstruct == 0) {
                for (int round = 0; round < 3; round++)
                    for (int sv = 0; sv < 3; sv++) {
                        exec_mt(p, ti, to, bt, p->mt, p->tsz, p->gsz, 0,
                                p->mode, sv, 0, p->p1pf, 0, 0, 0);
                        double t0 = now_s();
                        exec_mt(p, ti, to, bt, p->mt, p->tsz, p->gsz, 0,
                                p->mode, sv, 0, p->p1pf, 0, 0, 0);
                        exec_mt(p, ti, to, bt, p->mt, p->tsz, p->gsz, 0,
                                p->mode, sv, 0, p->p1pf, 0, 0, 0);
                        double dt = (now_s() - t0) / 2.0;
                        if (dt < tsl[sv]) tsl[sv] = dt;
                    }
                p->slabpf = 1;
                for (int sv = 0; sv < 3; sv += 2)
                    if (tsl[sv] < tsl[p->slabpf] && tsl[sv] < 0.99 * tsl[1])
                        p->slabpf = sv;
                /* pass-1 SC store mode re-raced under threads (the phase-1
                 * "plain 3/3" verdict was single-core, SC L3-resident --
                 * with 16 threads/socket SC streams and NT may flip).  Non-
                 * plain must beat plain by 1%.                             */
                for (int round = 0; round < 3; round++)
                    for (int sv = 0; sv < 3; sv++) {
                        exec_mt(p, ti, to, bt, p->mt, p->tsz, p->gsz, 0,
                                p->mode, p->slabpf, sv, p->p1pf, 0, 0, 0);
                        double t0 = now_s();
                        exec_mt(p, ti, to, bt, p->mt, p->tsz, p->gsz, 0,
                                p->mode, p->slabpf, sv, p->p1pf, 0, 0, 0);
                        exec_mt(p, ti, to, bt, p->mt, p->tsz, p->gsz, 0,
                                p->mode, p->slabpf, sv, p->p1pf, 0, 0, 0);
                        double dt = (now_s() - t0) / 2.0;
                        if (dt < tsc[sv]) tsc[sv] = dt;
                    }
                p->scst = 0;
                for (int sv = 1; sv < 3; sv++)
                    if (tsc[sv] < tsc[p->scst] && tsc[sv] < 0.99 * tsc[0])
                        p->scst = sv;
                /* tsc rows ran with the final slabpf, so the picked row is
                 * the fully-settled configuration's time */
                if (tsc[p->scst] / bt < p->tpick) p->tpick = tsc[p->scst] / bt;
                /* mt_r3: pass-1 input-prefetch A/B on the batch winner.  In
                 * gang mode the next-plane prefetch is ON whenever a gang
                 * runs more volumes than there are gangs (nvol > ngang);
                 * mt_r1 measured FORCING it in a loaded gang costing 8% on
                 * wallaby at B=8 (371.8 vs 343.0), and the node has never
                 * voted on off-vs-on there.  p1pf=2 forces it off; off must
                 * beat the incumbent by 1%.                                 */
                if (bt > 1 && p->mt != 0) {
                    const int pv[2] = {p->p1pf, 2};
                    for (int round = 0; round < 3; round++)
                        for (int i = 0; i < 2; i++) {
                            exec_mt(p, ti, to, bt, p->mt, p->tsz, p->gsz, 0,
                                    p->mode, p->slabpf, p->scst, pv[i],
                                    0, 0, 0);
                            double t0 = now_s();
                            exec_mt(p, ti, to, bt, p->mt, p->tsz, p->gsz, 0,
                                    p->mode, p->slabpf, p->scst, pv[i],
                                    0, 0, 0);
                            exec_mt(p, ti, to, bt, p->mt, p->tsz, p->gsz, 0,
                                    p->mode, p->slabpf, p->scst, pv[i],
                                    0, 0, 0);
                            double dt = (now_s() - t0) / 2.0;
                            if (dt < tpp[i]) tpp[i] = dt;
                        }
                    if (tpp[1] < 0.99 * tpp[0]) {
                        p->p1pf = 2;
                        if (tpp[1] / bt < p->tpick) p->tpick = tpp[1] / bt;
                    }
                }
            }
            /* serial reference, for the parallel-efficiency report only
             * (clamped: it is a reference, not a candidate, except at B=1
             * where it raced in the grid)                                  */
            if (batch == 1 && nthr > 1) {
                p->tser = best[nc - 1];
            } else {
                int bs = bt < 4 ? bt : 4;
                exec_serial(p, ti, to, bs, 0, 2, 1, 0, bs == 1, 0, 0, 0);
                double t0 = now_s();
                exec_serial(p, ti, to, bs, 0, 2, 1, 0, bs == 1, 0, 0, 0);
                exec_serial(p, ti, to, bs, 0, 2, 1, 0, bs == 1, 0, 0, 0);
                p->tser = (now_s() - t0) / 2.0 / bs;
            }
            if (getenv("FFT64R_TUNEDBG")) {
                static const char *mtn[8] = {"ser", "split", "vol", "vdyn",
                                             "gang", "gangp", "gangd", "splitf"};
                for (int v = 0; v < nc; v++)
                    fprintf(stderr,
                            "tuner %-5s T%-2d g%-2d %s mode=%d p1pf=%d : %.1f us/vol%s\n",
                            mtn[cv[v].mt], cv[v].tsz, cv[v].gsz,
                            cv[v].sstruct ? "tiled" : "fused", cv[v].mode,
                            cv[v].p1pf, best[v] * 1e6 / bt,
                            v == pick ? "  <-- pick" : "");
                if (p->sstruct == 0) {
                    fprintf(stderr, "tuner slabpf A/B: 0 %.1f / 1 %.1f / 2 %.1f"
                            " us/vol -> slabpf=%d\n", tsl[0] * 1e6 / bt,
                            tsl[1] * 1e6 / bt, tsl[2] * 1e6 / bt, p->slabpf);
                    fprintf(stderr, "tuner scst A/B: plain %.1f / pfw %.1f"
                            " / nt %.1f us/vol -> scst=%d\n", tsc[0] * 1e6 / bt,
                            tsc[1] * 1e6 / bt, tsc[2] * 1e6 / bt, p->scst);
                    if (bt > 1 && p->mt != 0)
                        fprintf(stderr, "tuner p1pf A/B: on %.1f / off %.1f"
                                " us/vol -> p1pf=%d\n", tpp[0] * 1e6 / bt,
                                tpp[1] * 1e6 / bt, p->p1pf);
                }
                fprintf(stderr, "tuner serial ref %.1f us/vol, pick %.1f -> "
                        "eff %.2f at T%d\n", p->tser * 1e6, p->tpick * 1e6,
                        p->tpick > 0 ? p->tser / (p->tpick * p->tsz) : 0.0,
                        p->tsz);
            }
        }
        free(ti); free(to);
        {   /* env forcing for the monitor's one-flag experiments */
            const char *e;
            if ((e = getenv("FFT64R_MT")))     p->mt      = atoi(e);
            if ((e = getenv("FFT64R_GSZ")))    p->gsz     = atoi(e);
            if ((e = getenv("FFT64R_T"))) {
                int tv = atoi(e);
                if (tv >= 1 && tv <= nthr) p->tsz = tv;
            }
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
        {
            static const char *mtn[8] = {"ser", "split", "vol", "vdyn",
                                         "gang", "gangp", "gangd", "splitf"};
            snprintf(g_desc, sizeof g_desc,
                     "radix-8^2/axis AVX-512 MT; pick[B=%d]=%s-T%d-g%d-%s-%s"
                     "+slabpf%d+sc%d+p1%d; ser=%.0fus/vol pick=%.0f eff=%.2f",
                     batch, mtn[p->mt >= 0 && p->mt <= 7 ? p->mt : 0],
                     p->tsz, p->gsz, p->sstruct ? "tiled" : "fused",
                     p->mode == 1 ? "nt" : p->mode == 2 ? "pfw" : "plain",
                     p->slabpf, p->scst, p->p1pf, p->tser * 1e6,
                     p->tpick * 1e6,
                     p->tpick > 0 ? p->tser / (p->tpick * p->tsz) : 0.0);
        }
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
    exec_mt(p, vi, vo, p->B, p->mt, p->tsz, p->gsz, p->sstruct, p->mode,
            p->slabpf, p->scst, p->p1pf, p->propf, p->xb_on, p->fout);
#else
    exec_scalar(p, vi, vo, p->B);
#endif
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    if (p->basemap) {
        if (p->base_is_mmap) munmap(p->basemap, p->basesz);
        else free(p->basemap);
    }
    free(p);
}
