/* L8_radix8 -- forward complex-double 3D DFT of the 8x8x8 cube, batched.
 *
 * TECHNIQUE
 *   One fully unrolled radix-8 codelet (52 instructions = 44 add/sub + 8 FMA, no
 *   twiddle table: every 8th root of unity is +-1, +-i or +-(1+-i)/sqrt(2), so the only
 *   multiplicative constant in the whole transform is C = 1/sqrt(2)), applied along all
 *   three axes.  SPLIT-complex vectors (separate re/im registers) make every
 *   multiplication by +-i a register rename plus a sign folded into the following add.
 *   Lanes hold 8 independent lines, so one vector instruction = one scalar operation of
 *   the codelet, with no cross-lane traffic inside the butterflies.
 *
 *   THREE kernel shapes are compiled from the same primitives and fft3d_create()
 *   picks by self-timing at the real batch size:
 *
 *   2-PASS (optimized for B=1; round-1/2 shape):
 *     pass 1, per x-plane:  transpose in (z -> registers) -> radix-8 along z
 *                           -> transpose (lanes = k2) -> radix-8 along y -> scratch
 *     pass 2, per k1:       radix-8 along x -> interleave -> store out rows
 *                           at out + k0*128 + k1*16: a 1-KB-STRIDED store pattern.
 *
 *   FUSED "1f" (new in round panel_r5; ported from L8_fusedaxes, which won the
 *   node's B=64 cell with it three rounds running, 0.623 us in panel_r4):
 *     phase A, per x-plane: deinterleave each z-pencil with one vunpcklo/hi pair
 *                           (lane l = z = PI[l], PI = 0,4,1,5,2,6,3,7; only 16
 *                           shuffles per plane) -> radix-8 along y (elementwise)
 *                           -> split-planar scratch sr[k1][x] / si[k1][x]
 *     phase B, per k1:      radix-8 along x (elementwise, contiguous loads)
 *                           -> trans8 pair (z into registers via the T2,T3,T1
 *                           non-destructive network; register j = z = PI[j],
 *                           fed to the codelet through PI^-1 = 0,2,4,6,1,3,5,7)
 *                           -> radix-8 along z -> untrans+interleave fused into
 *                           one 48-shuffle network over all 16 registers ->
 *                           16 half-pencil stores via the out_off table.
 *     Same totals as 2-pass (1248 FP, 896 shuffles, 256+256 loads/stores); the
 *     shuffles sit in phase B next to L1-resident loads instead of in pass 1
 *     next to the input demand-loads, which is the whole (measured) difference.
 *
 *   3-PASS (optimized for batch; new in round panel_r3):
 *     pass 1, per x-plane:  transpose in -> radix-8 along z -> scratch[x][k2]
 *     pass 2, per k2:       radix-8 along x, IN PLACE in the scratch column
 *                           (reads scr[x][k2], writes scr[k0][k2]; 0 shuffles)
 *     pass 3, per k0:       transpose (y -> registers) -> radix-8 along y
 *                           -> interleave -> store the contiguous 1 KB output plane.
 *     The output volume is written FRONT TO BACK, fully sequentially -- which is what
 *     the DRAM/NT-store regimes want, and what the 2-pass shape structurally cannot do
 *     (its last DFT axis is x, so k0 strides the stores by 1 KB).  Cost: +128 loads
 *     +128 stores per volume, all L1-resident; shuffle and FP counts are IDENTICAL
 *     (the second transpose of pass 1 moves to pass 3).  Sequential-store idea borrowed
 *     from L8_batchsimd's LANEX pass structure, which owns every batched cell.
 *
 * OPERATION COUNT PER TRANSFORM (one 8^3 volume, either shape)
 *   8-point codelet: 4 real mul + 52 real add = 56 flops (the published optimum:
 *   Burrus T7.1/T9.1, FFTW n1_8, Yavne 4N log2 N - 6N + 8), issued as 52 instructions
 *   (44 add/sub + 8 FMA; w8^{1,3} folded as add,sub + 4 FMA, from L8_batchsimd).
 *   24 codelets x 52 = 1248 vector FP instructions (8-wide), 896 shuffles
 *   (768 transpose + 128 interleave, copy-free on AVX-512), and 512 (2-pass) or
 *   768 (3-pass) loads/stores.  No twiddle table is read at run time.
 *
 * STREAMING REGIME
 *   Next-volume software prefetch (t0, distance 1 volume) -- the L2 streamer stops
 *   at 4 KiB page boundaries and one 8 KiB volume spans two pages.  Placements:
 *   BURST (all 128 lines during pass 1) and SPREAD (6/5/5 lines per iteration of
 *   passes 1/2/3 or 8/8 per fused-phase iteration, ~1 prefetch per 11 cycles;
 *   borrowed from L8_fusedaxes r2).  NEW in panel_r7: write-intent prefetch (pfw,
 *   prefetchw) of the next volume's OUTPUT lines at the same spread cadence --
 *   with plain stores every output line pays a read-for-ownership, and issuing it
 *   one volume early hides that latency.  Borrowed from L8_fusedaxes' panel_r5
 *   node win (fused+pfs+pfw took B=2048 at 0.910 and B=16384 at 1.254, 3/3
 *   picks); the mechanism traces to L6_unrolled/L36_pfa.  NT stores are RETIRED
 *   from the candidate sets: they lost on the node in every streaming cell four
 *   rounds running (panel_r5 VERDICT 4.5: hide the RFO, do not avoid it).  The
 *   NT kernels stay compiled for the record.
 *
 * TUNER (regime-gated since panel_r4; the panel_r3 verdict showed a plan-time
 *   coin-flip shipping a 6.7 % slower pick at B=64 in 2 of 3 runs)
 *   Candidate sets are gated by working-set size so a regime's known losers are
 *   never offered in it:  B=1 -> {2p, 3p, 1f} (node picked 2p every round);
 *   mid regime (B>1, in+out <= 0.9*L3) -> {1f-pfs, 2p, 1f, 2p-pfs, 3p}: 1f-pfs
 *   is the node-validated default (r5 B=64 pick 3/3, 0.619); pfw is not offered
 *   here (prefetchw on cache-resident lines is a documented loser: L36_pfa
 *   +13 %/+11 % at B=1/B=4);
 *   in+out > 0.9*L3 -> {1f-pfs-pfw, 3p-pfs-pfw, 3p-pfs, 1f-pfs, 3p-pf}: default
 *   is the family that won both node streaming cells in r5, and the {1f,3p} x
 *   {pfw,no-pfw} table is the panel_r5 VERDICT's fusion-vs-pfw isolating
 *   experiment run inside one create() call.  First entry is the regime default
 *   and displacing it needs a >2 % win.  Each timed trial is preceded by one
 *   untimed pass so every candidate is measured against its own cache state
 *   (L8_fusedaxes r3 protocol).  The batch loop lives inside per-candidate run
 *   functions (direct call, kernel inlined) instead of a per-volume indirect
 *   call.  The pick is reported through fft3d_description().
 *
 * ASSUMPTIONS
 *   L == 8 only.  `in` and `out` are distinct and 64-byte aligned (driver guarantees
 *   both); every vector access is 64-byte aligned as a result (alignment-checked
 *   fallback exists).  Three backends -- portable C, AVX2, AVX-512 -- are compiled
 *   from ONE macro-defined kernel text, so the algorithm verified locally is the
 *   algorithm the node runs.
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "fft3d_api.h"

#define LL   8
#define NDBL 1024   /* doubles per volume: 8^3 complex */

/* Scratch x-plane stride in doubles.  128 = dense.  144 pads each plane by one
 * 128-byte row, so the 8 strided pass-2 accesses spread over 8 distinct L1 sets
 * instead of 4 (Bailey/#04 s7.3; the padding L8_batchsimd carries as LPZ=9).
 * Overridable for A/B: -DL8R_SCRX=128. */
#ifndef L8R_SCRX
#define L8R_SCRX 144
#endif
#define SCRX L8R_SCRX

/* Backends below the best available ISA stay compiled (they are how the algorithm
 * text gets verified on machines without AVX-512) but are not tuner candidates,
 * so silence the unused-function warning on them. */
#if defined(__GNUC__)
#define KFN_MAYBE_UNUSED __attribute__((unused))
#define KERNFN static inline __attribute__((always_inline, unused)) void
#else
#define KFN_MAYBE_UNUSED
#define KERNFN static void
#endif

/* Per-candidate run function: the batch loop lives HERE, in the same TU as the
 * kernel, with the kernel forced inline and the prefetch flag a compile-time
 * constant -- no per-volume indirect call, no per-volume branch (panel_r3 "Next"
 * item 3: the last structural difference to L8_batchsimd's lanex_run). */
#define DEFRUN(rname, kfn, dopf)                                               \
    static KFN_MAYBE_UNUSED void rname(const double *restrict in,              \
                                       double *restrict out,                   \
                                       double *restrict scr, int nb)           \
    {                                                                          \
        for (int b = 0; b < nb; ++b)                                           \
            kfn(in + (size_t)b * NDBL, out + (size_t)b * NDBL, scr,            \
                ((dopf) && b + 1 < nb) ? in + (size_t)(b + 1) * NDBL : NULL);  \
    }

static const double C_RSQ2 = 0.7071067811865475244008443621048490;

/* Software prefetch of the next volume's input (t0, one volume ahead; hint and
 * distance node-validated by L8_batchsimd r2, B=2048 = 1.205 us). */
#define PF8(p) __builtin_prefetch((p), 0, 3)

#define PFVOL(f)                                                               \
    do {                                                                       \
        PF8(f);      PF8(f + 8);   PF8(f + 16);  PF8(f + 24);                  \
        PF8(f + 32); PF8(f + 40);  PF8(f + 48);  PF8(f + 56);                  \
        PF8(f + 64); PF8(f + 72);  PF8(f + 80);  PF8(f + 88);                  \
        PF8(f + 96); PF8(f + 104); PF8(f + 112); PF8(f + 120);                 \
    } while (0)

/* Spread placement: the same 128 lines of volume b+1, issued 6/5/5 per iteration
 * of passes 1/2/3 (48+40+40 = 128, each 64-byte line exactly once) instead of 16
 * per pass-1 iteration.  ~1 prefetch per 11 cycles, so it never competes with the
 * pass-1 demand-load burst or the NT store drain for the ~12 fill buffers
 * (L8_fusedaxes r2 failure 4 documents the burst+NT clog with a 1.6x number). */
#define PF6(f) do { PF8(f); PF8((f) + 8);  PF8((f) + 16);                      \
                    PF8((f) + 24); PF8((f) + 32); PF8((f) + 40); } while (0)
#define PF5(f) do { PF8(f); PF8((f) + 8);  PF8((f) + 16);                      \
                    PF8((f) + 24); PF8((f) + 32); } while (0)
/* 8 lines (64 doubles): the per-iteration quantum of the 2-pass/fused spread
 * placements (2 passes x 8 iterations x 8 lines = the 128 lines of a volume). */
#define PF8L(f) do { PF8(f); PF8((f) + 8);  PF8((f) + 16); PF8((f) + 24);       \
                     PF8((f) + 32); PF8((f) + 40); PF8((f) + 48);              \
                     PF8((f) + 56); } while (0)

/* Write-intent prefetch of the next volume's OUTPUT lines (emits prefetchw on
 * CLX/SPR).  With plain stores every output line pays a read-for-ownership;
 * issuing it one volume early hides that latency instead of avoiding it with
 * NT stores -- the rule the panel_r5 node data settled (L8_fusedaxes'
 * fused+pfs+pfw picked 3/3 at B=2048/16384, 0.910/1.254 us; same mechanism as
 * L36_pfa's pf=2 and L6_unrolled's fused_pfw).  Same spread cadence as the
 * input t0 prefetches, mirrored onto out + NDBL (valid whenever pf != NULL,
 * since in and out advance through the batch together). */
#define PFW8(p) __builtin_prefetch((p), 1, 3)
#define PF6W(f) do { PFW8(f); PFW8((f) + 8);  PFW8((f) + 16);                   \
                     PFW8((f) + 24); PFW8((f) + 32); PFW8((f) + 40); } while (0)
#define PF5W(f) do { PFW8(f); PFW8((f) + 8);  PFW8((f) + 16);                   \
                     PFW8((f) + 24); PFW8((f) + 32); } while (0)
#define PF8LW(f) do { PFW8(f); PFW8((f) + 8);  PFW8((f) + 16); PFW8((f) + 24);  \
                      PFW8((f) + 32); PFW8((f) + 40); PFW8((f) + 48);           \
                      PFW8((f) + 56); } while (0)

/* Per-instantiation prefetch-placement hooks for KERNEL2_BODY / KERNEL3_BODY. */
#define KPF_NONE(pf, i)    do { (void)(pf); } while (0)
#define KPF1_BURST(pf, x)  do { if (pf) PFVOL((pf) + (size_t)(x) * 128); } while (0)
#define KPF1_SPREAD(pf, x) do { if (pf) PF6((pf) + (size_t)(x) * 48); } while (0)
#define KPF2_SPREAD(pf, k) do { if (pf) PF5((pf) + 384 + (size_t)(k) * 40); } while (0)
#define KPF3_SPREAD(pf, k) do { if (pf) PF5((pf) + 704 + (size_t)(k) * 40); } while (0)
/* Spread placement for the 2-pass shape: 8 lines per iteration of each pass. */
#define KPF1_SPREAD2(pf, x) do { if (pf) PF8L((pf) + (size_t)(x) * 64); } while (0)
#define KPF2_SPREAD2(pf, k) do { if (pf) PF8L((pf) + 512 + (size_t)(k) * 64); } while (0)
/* Spread + write-intent: the input hooks above plus the mirror-image prefetchw
 * on the next volume's output.  These reference the kernel bodies' `out`
 * parameter by name (in scope wherever they are expanded). */
#define KPF1_SPREADW(pf, x) do { if (pf) { PF6((pf) + (size_t)(x) * 48);        \
    PF6W(out + NDBL + (size_t)(x) * 48); } } while (0)
#define KPF2_SPREADW(pf, k) do { if (pf) { PF5((pf) + 384 + (size_t)(k) * 40);  \
    PF5W(out + NDBL + 384 + (size_t)(k) * 40); } } while (0)
#define KPF3_SPREADW(pf, k) do { if (pf) { PF5((pf) + 704 + (size_t)(k) * 40);  \
    PF5W(out + NDBL + 704 + (size_t)(k) * 40); } } while (0)

/* ------------------------------------------------------------------ *
 *  The radix-8 split-complex codelet, written once.
 *
 *  XR/XI in, YR/YI out (arrays of 8 vectors).  All inputs are consumed in
 *  stage 1, so YR/YI may be the same arrays as XR/XI.
 *
 *  Decimation in time, natural input order (the bit reversal is absorbed by
 *  choosing which register feeds which butterfly):
 *    b = size-2 stage, c = size-4 stage (w4^1 = -i, free in split form),
 *    Y = size-8 stage (w8^1 = (1-i)C, w8^2 = -i, w8^3 = (-1-i)C).
 *  Counts: 16 + 16 + 20 = 52 instructions (44 add/sub + 8 FMA).  The w8^{1,3}
 *  twiddle+butterfly is 1 add + 1 sub + 4 FMA/FNMA each; formulation borrowed
 *  from L8_batchsimd round 1.
 * ------------------------------------------------------------------ */
#define RADIX8(XR, XI, YR, YI)                                                 \
    do {                                                                       \
        KV b0r = KADD((XR)[0], (XR)[4]), b0i = KADD((XI)[0], (XI)[4]);          \
        KV b1r = KSUB((XR)[0], (XR)[4]), b1i = KSUB((XI)[0], (XI)[4]);          \
        KV b2r = KADD((XR)[2], (XR)[6]), b2i = KADD((XI)[2], (XI)[6]);          \
        KV b3r = KSUB((XR)[2], (XR)[6]), b3i = KSUB((XI)[2], (XI)[6]);          \
        KV b4r = KADD((XR)[1], (XR)[5]), b4i = KADD((XI)[1], (XI)[5]);          \
        KV b5r = KSUB((XR)[1], (XR)[5]), b5i = KSUB((XI)[1], (XI)[5]);          \
        KV b6r = KADD((XR)[3], (XR)[7]), b6i = KADD((XI)[3], (XI)[7]);          \
        KV b7r = KSUB((XR)[3], (XR)[7]), b7i = KSUB((XI)[3], (XI)[7]);          \
        KV c0r = KADD(b0r, b2r), c0i = KADD(b0i, b2i);                          \
        KV c2r = KSUB(b0r, b2r), c2i = KSUB(b0i, b2i);                          \
        KV c1r = KADD(b1r, b3i), c1i = KSUB(b1i, b3r);                          \
        KV c3r = KSUB(b1r, b3i), c3i = KADD(b1i, b3r);                          \
        KV c4r = KADD(b4r, b6r), c4i = KADD(b4i, b6i);                          \
        KV c6r = KSUB(b4r, b6r), c6i = KSUB(b4i, b6i);                          \
        KV c5r = KADD(b5r, b7i), c5i = KSUB(b5i, b7r);                          \
        KV c7r = KSUB(b5r, b7i), c7i = KADD(b5i, b7r);                          \
        KV s5 = KADD(c5r, c5i);                                                \
        KV t5 = KSUB(c5i, c5r);                                                \
        KV s7 = KSUB(c7i, c7r);                                                \
        KV t7 = KADD(c7r, c7i);                                                \
        (YR)[0] = KADD(c0r, c4r); (YI)[0] = KADD(c0i, c4i);                     \
        (YR)[4] = KSUB(c0r, c4r); (YI)[4] = KSUB(c0i, c4i);                     \
        (YR)[2] = KADD(c2r, c6i); (YI)[2] = KSUB(c2i, c6r);                     \
        (YR)[6] = KSUB(c2r, c6i); (YI)[6] = KADD(c2i, c6r);                     \
        (YR)[1] = KFMA(kc, s5, c1r);  (YI)[1] = KFMA(kc, t5, c1i);              \
        (YR)[5] = KFNMA(kc, s5, c1r); (YI)[5] = KFNMA(kc, t5, c1i);             \
        (YR)[3] = KFMA(kc, s7, c3r);  (YI)[3] = KFNMA(kc, t7, c3i);             \
        (YR)[7] = KFNMA(kc, s7, c3r); (YI)[7] = KFMA(kc, t7, c3i);              \
    } while (0)

/* ------------------------------------------------------------------ *
 *  Kernel shape A: 2-pass (B=1 shape).  z+y per x-plane, then x per k1,
 *  strided output stores.
 * ------------------------------------------------------------------ */
#define KERNEL2_BODY                                                           \
{                                                                              \
    const KV kc = KBCAST(C_RSQ2);                                              \
    for (int x = 0; x < 8; ++x) {                                              \
        const double *p = in + (size_t)x * 128;                                \
        double *q = scr + (size_t)x * SCRX;                                     \
        KPF1(pf, x);                                                           \
        KV ar[8], ai[8], br[8], bi[8];                                          \
        for (int y = 0; y < 8; ++y) {                                          \
            ar[y] = KLOAD(p + (size_t)y * 16);                                 \
            ai[y] = KLOAD(p + (size_t)y * 16 + 8);                             \
        }                                                                      \
        KTR8A(ar, br);                                                         \
        KTR8A(ai, bi);                                                         \
        for (int z = 0; z < 4; ++z) {                                          \
            ar[z]     = br[2 * z];                                             \
            ai[z]     = br[2 * z + 1];                                         \
            ar[z + 4] = bi[2 * z];                                             \
            ai[z + 4] = bi[2 * z + 1];                                         \
        }                                                                      \
        RADIX8(ar, ai, ar, ai);                                                \
        KTR8B(ar, br);                                                         \
        KTR8B(ai, bi);                                                         \
        RADIX8(br, bi, br, bi);                                                \
        for (int k1 = 0; k1 < 8; ++k1) {                                       \
            KSTORE(q + (size_t)k1 * 16,     br[k1]);                           \
            KSTORE(q + (size_t)k1 * 16 + 8, bi[k1]);                           \
        }                                                                      \
    }                                                                          \
    for (int k1 = 0; k1 < 8; ++k1) {                                           \
        KPF2(pf, k1);                                                          \
        KV ur[8], ui[8];                                                       \
        for (int x = 0; x < 8; ++x) {                                          \
            ur[x] = KLOAD(scr + (size_t)x * SCRX + (size_t)k1 * 16);            \
            ui[x] = KLOAD(scr + (size_t)x * SCRX + (size_t)k1 * 16 + 8);        \
        }                                                                      \
        RADIX8(ur, ui, ur, ui);                                                \
        for (int k0 = 0; k0 < 8; ++k0)                                         \
            KILV(ur[k0], ui[k0], out + (size_t)k0 * 128 + (size_t)k1 * 16);    \
    }                                                                          \
}

/* ------------------------------------------------------------------ *
 *  Kernel shape B: 3-pass (batch shape), sequential output stores.
 *  z per x-plane -> x in place per k2 column -> y per k0-plane.
 *  Lane residue on AVX-512: pass 1's KTR8A leaves lane l = y SW(l); the
 *  z and x codelets are elementwise so nobody observes it; pass 3's KTR8B
 *  (output k renamed to slot SW(k)) restores natural registers and leaves
 *  lanes carrying k2 = SW(l), which KILV's index vectors already invert.
 * ------------------------------------------------------------------ */
#define KERNEL3_BODY                                                           \
{                                                                              \
    const KV kc = KBCAST(C_RSQ2);                                              \
    for (int x = 0; x < 8; ++x) {                                              \
        const double *p = in + (size_t)x * 128;                                \
        double *q = scr + (size_t)x * SCRX;                                     \
        KPF1(pf, x);                                                           \
        KV ar[8], ai[8], br[8], bi[8];                                          \
        for (int y = 0; y < 8; ++y) {                                          \
            ar[y] = KLOAD(p + (size_t)y * 16);                                 \
            ai[y] = KLOAD(p + (size_t)y * 16 + 8);                             \
        }                                                                      \
        KTR8A(ar, br);                                                         \
        KTR8A(ai, bi);                                                         \
        for (int z = 0; z < 4; ++z) {                                          \
            ar[z]     = br[2 * z];                                             \
            ai[z]     = br[2 * z + 1];                                         \
            ar[z + 4] = bi[2 * z];                                             \
            ai[z + 4] = bi[2 * z + 1];                                         \
        }                                                                      \
        RADIX8(ar, ai, ar, ai);                                                \
        for (int k2 = 0; k2 < 8; ++k2) {                                       \
            KSTORE(q + (size_t)k2 * 16,     ar[k2]);                           \
            KSTORE(q + (size_t)k2 * 16 + 8, ai[k2]);                           \
        }                                                                      \
    }                                                                          \
    for (int k2 = 0; k2 < 8; ++k2) {                                           \
        KPF2(pf, k2);                                                          \
        KV ur[8], ui[8];                                                       \
        for (int x = 0; x < 8; ++x) {                                          \
            ur[x] = KLOAD(scr + (size_t)x * SCRX + (size_t)k2 * 16);            \
            ui[x] = KLOAD(scr + (size_t)x * SCRX + (size_t)k2 * 16 + 8);        \
        }                                                                      \
        RADIX8(ur, ui, ur, ui);                                                \
        for (int k0 = 0; k0 < 8; ++k0) {                                       \
            KSTORE(scr + (size_t)k0 * SCRX + (size_t)k2 * 16,     ur[k0]);      \
            KSTORE(scr + (size_t)k0 * SCRX + (size_t)k2 * 16 + 8, ui[k0]);      \
        }                                                                      \
    }                                                                          \
    for (int k0 = 0; k0 < 8; ++k0) {                                           \
        KPF3(pf, k0);                                                          \
        const double *q = scr + (size_t)k0 * SCRX;                              \
        double *o = out + (size_t)k0 * 128;                                    \
        KV vr[8], vi[8], wr[8], wi[8];                                          \
        for (int k2 = 0; k2 < 8; ++k2) {                                       \
            vr[k2] = KLOAD(q + (size_t)k2 * 16);                               \
            vi[k2] = KLOAD(q + (size_t)k2 * 16 + 8);                           \
        }                                                                      \
        KTR8B(vr, wr);                                                         \
        KTR8B(vi, wi);                                                         \
        RADIX8(wr, wi, wr, wi);                                                \
        for (int k1 = 0; k1 < 8; ++k1)                                         \
            KILV(wr[k1], wi[k1], o + (size_t)k1 * 16);                         \
    }                                                                          \
}

/* ================================================================== *
 *  Backend 1: portable C.  Always compiled; the reference the whole
 *  algorithm is verified against, and the fallback on non-x86.
 * ================================================================== */
typedef struct { double d[8]; } v8g;

static inline v8g g_add(v8g a, v8g b)
{ v8g r; for (int i = 0; i < 8; ++i) r.d[i] = a.d[i] + b.d[i]; return r; }
static inline v8g g_sub(v8g a, v8g b)
{ v8g r; for (int i = 0; i < 8; ++i) r.d[i] = a.d[i] - b.d[i]; return r; }
static inline v8g g_fma(v8g a, v8g b, v8g c)
{ v8g r; for (int i = 0; i < 8; ++i) r.d[i] = a.d[i] * b.d[i] + c.d[i]; return r; }
static inline v8g g_fnma(v8g a, v8g b, v8g c)
{ v8g r; for (int i = 0; i < 8; ++i) r.d[i] = c.d[i] - a.d[i] * b.d[i]; return r; }
static inline v8g g_bcast(double c)
{ v8g r; for (int i = 0; i < 8; ++i) r.d[i] = c; return r; }
static inline v8g g_load(const double *p)
{ v8g r; for (int i = 0; i < 8; ++i) r.d[i] = p[i]; return r; }
static inline void g_store(double *p, v8g a)
{ for (int i = 0; i < 8; ++i) p[i] = a.d[i]; }
static inline void g_tr8(const v8g *r, v8g *t)
{ for (int i = 0; i < 8; ++i) for (int j = 0; j < 8; ++j) t[i].d[j] = r[j].d[i]; }
static inline void g_ilv(v8g re, v8g im, double *dst)
{ for (int k = 0; k < 8; ++k) { dst[2 * k] = re.d[k]; dst[2 * k + 1] = im.d[k]; } }

#define KV      v8g
#define KADD    g_add
#define KSUB    g_sub
#define KFMA    g_fma
#define KFNMA   g_fnma
#define KBCAST  g_bcast
#define KLOAD   g_load
#define KSTORE  g_store
#define KTR8A   g_tr8
#define KTR8B   g_tr8
#define KILV    g_ilv
#define KPF1    KPF1_BURST
#define KPF2    KPF_NONE
#define KPF3    KPF_NONE
KERNFN kernel2_gen(const double *restrict in, double *restrict out,
                   double *restrict scr, const double *restrict pf) KERNEL2_BODY
KERNFN kernel3_gen(const double *restrict in, double *restrict out,
                   double *restrict scr, const double *restrict pf) KERNEL3_BODY
DEFRUN(run_2p_gen, kernel2_gen, 0)
DEFRUN(run_3p_gen, kernel3_gen, 0)
#undef KV
#undef KADD
#undef KSUB
#undef KFMA
#undef KFNMA
#undef KBCAST
#undef KLOAD
#undef KSTORE
#undef KTR8A
#undef KTR8B
#undef KILV
#undef KPF1
#undef KPF2
#undef KPF3

/* ================================================================== *
 *  Backend 2: AVX2 + FMA.  One 8-wide vector = two ymm registers, so the
 *  identical kernel body compiles to 256-bit code (which is also the
 *  256-bit-port-scheme hedge on the grading node).
 * ================================================================== */
#if defined(__AVX2__) && defined(__FMA__)
typedef struct { __m256d lo, hi; } v8y;

static inline v8y y_add(v8y a, v8y b)
{ v8y r; r.lo = _mm256_add_pd(a.lo, b.lo); r.hi = _mm256_add_pd(a.hi, b.hi); return r; }
static inline v8y y_sub(v8y a, v8y b)
{ v8y r; r.lo = _mm256_sub_pd(a.lo, b.lo); r.hi = _mm256_sub_pd(a.hi, b.hi); return r; }
static inline v8y y_fma(v8y a, v8y b, v8y c)
{ v8y r; r.lo = _mm256_fmadd_pd(a.lo, b.lo, c.lo); r.hi = _mm256_fmadd_pd(a.hi, b.hi, c.hi); return r; }
static inline v8y y_fnma(v8y a, v8y b, v8y c)
{ v8y r; r.lo = _mm256_fnmadd_pd(a.lo, b.lo, c.lo); r.hi = _mm256_fnmadd_pd(a.hi, b.hi, c.hi); return r; }
static inline v8y y_bcast(double c)
{ v8y r; r.lo = _mm256_set1_pd(c); r.hi = r.lo; return r; }
static inline v8y y_load(const double *p)
{ v8y r; r.lo = _mm256_loadu_pd(p); r.hi = _mm256_loadu_pd(p + 4); return r; }
static inline void y_store(double *p, v8y a)
{ _mm256_storeu_pd(p, a.lo); _mm256_storeu_pd(p + 4, a.hi); }

/* 4x4 double transpose: 4 unpack + 4 vperm2f128. */
#define T4(a0, a1, a2, a3, o0, o1, o2, o3)                                     \
    do {                                                                       \
        __m256d u0 = _mm256_unpacklo_pd((a0), (a1));                           \
        __m256d u1 = _mm256_unpackhi_pd((a0), (a1));                           \
        __m256d u2 = _mm256_unpacklo_pd((a2), (a3));                           \
        __m256d u3 = _mm256_unpackhi_pd((a2), (a3));                           \
        (o0) = _mm256_permute2f128_pd(u0, u2, 0x20);                           \
        (o1) = _mm256_permute2f128_pd(u1, u3, 0x20);                           \
        (o2) = _mm256_permute2f128_pd(u0, u2, 0x31);                           \
        (o3) = _mm256_permute2f128_pd(u1, u3, 0x31);                           \
    } while (0)

static inline void y_tr8(const v8y *r, v8y *t)
{
    T4(r[0].lo, r[1].lo, r[2].lo, r[3].lo, t[0].lo, t[1].lo, t[2].lo, t[3].lo);
    T4(r[4].lo, r[5].lo, r[6].lo, r[7].lo, t[0].hi, t[1].hi, t[2].hi, t[3].hi);
    T4(r[0].hi, r[1].hi, r[2].hi, r[3].hi, t[4].lo, t[5].lo, t[6].lo, t[7].lo);
    T4(r[4].hi, r[5].hi, r[6].hi, r[7].hi, t[4].hi, t[5].hi, t[6].hi, t[7].hi);
}

static inline void y_ilv(v8y re, v8y im, double *dst)
{
    __m256d u = _mm256_unpacklo_pd(re.lo, im.lo);
    __m256d v = _mm256_unpackhi_pd(re.lo, im.lo);
    _mm256_storeu_pd(dst,     _mm256_permute2f128_pd(u, v, 0x20));
    _mm256_storeu_pd(dst + 4, _mm256_permute2f128_pd(u, v, 0x31));
    u = _mm256_unpacklo_pd(re.hi, im.hi);
    v = _mm256_unpackhi_pd(re.hi, im.hi);
    _mm256_storeu_pd(dst + 8,  _mm256_permute2f128_pd(u, v, 0x20));
    _mm256_storeu_pd(dst + 12, _mm256_permute2f128_pd(u, v, 0x31));
}

static inline void y_ilv_nt(v8y re, v8y im, double *dst)
{
    __m256d u = _mm256_unpacklo_pd(re.lo, im.lo);
    __m256d v = _mm256_unpackhi_pd(re.lo, im.lo);
    _mm256_stream_pd(dst,     _mm256_permute2f128_pd(u, v, 0x20));
    _mm256_stream_pd(dst + 4, _mm256_permute2f128_pd(u, v, 0x31));
    u = _mm256_unpacklo_pd(re.hi, im.hi);
    v = _mm256_unpackhi_pd(re.hi, im.hi);
    _mm256_stream_pd(dst + 8,  _mm256_permute2f128_pd(u, v, 0x20));
    _mm256_stream_pd(dst + 12, _mm256_permute2f128_pd(u, v, 0x31));
}

#define KV      v8y
#define KADD    y_add
#define KSUB    y_sub
#define KFMA    y_fma
#define KFNMA   y_fnma
#define KBCAST  y_bcast
#define KLOAD   y_load
#define KSTORE  y_store
#define KTR8A   y_tr8
#define KTR8B   y_tr8
#define KILV    y_ilv
#define KPF1    KPF1_BURST
#define KPF2    KPF_NONE
#define KPF3    KPF_NONE
KERNFN kernel2_avx2(const double *restrict in, double *restrict out,
                    double *restrict scr, const double *restrict pf) KERNEL2_BODY
KERNFN kernel3_avx2(const double *restrict in, double *restrict out,
                    double *restrict scr, const double *restrict pf) KERNEL3_BODY
#undef KPF1
#undef KPF2
#undef KPF3
#define KPF1    KPF1_SPREAD
#define KPF2    KPF2_SPREAD
#define KPF3    KPF3_SPREAD
KERNFN kernel3s_avx2(const double *restrict in, double *restrict out,
                     double *restrict scr, const double *restrict pf) KERNEL3_BODY
#undef KILV
#define KILV    y_ilv_nt
/* 3-pass kernel, non-temporal output stores, spread prefetch (a prefetch burst and
 * an NT drain fight for the same fill buffers -- L8_fusedaxes r2).  Every output
 * line is written in full by one KILV call and the plane order is sequential, so
 * no read-for-ownership and no write-combining games.  Only worth it when the
 * batch's footprint exceeds the last-level cache; fft3d_create() decides. */
KERNFN kernel3s_nt_avx2(const double *restrict in, double *restrict out,
                        double *restrict scr, const double *restrict pf) KERNEL3_BODY
DEFRUN(run_2p_avx2,        kernel2_avx2,     0)
DEFRUN(run_3p_avx2,        kernel3_avx2,     0)
DEFRUN(run_3p_pf_avx2,     kernel3_avx2,     1)
DEFRUN(run_3p_pfs_avx2,    kernel3s_avx2,    1)
DEFRUN(run_3p_nt_avx2,     kernel3s_nt_avx2, 0)
DEFRUN(run_3p_nt_pfs_avx2, kernel3s_nt_avx2, 1)
#undef KV
#undef KADD
#undef KSUB
#undef KFMA
#undef KFNMA
#undef KBCAST
#undef KLOAD
#undef KSTORE
#undef KTR8A
#undef KTR8B
#undef KILV
#undef KPF1
#undef KPF2
#undef KPF3
#endif /* AVX2 */

/* ================================================================== *
 *  Backend 3: AVX-512.  One 8-wide vector = one zmm.  8x8 transpose in
 *  24 shuffles, ALL two-source non-destructive forms with immediate
 *  control (vshuff64x2 / vunpck) -- no vpermt2pd, hence no compiler
 *  register copies and no index-vector registers.  Network borrowed from
 *  L8_fusedaxes round 1 (via L8_batchsimd): a straight r<->l1 middle
 *  level has no non-destructive encoding, but the 3-cycle
 *  r1 -> l2 -> l1 -> r1 (imm 0x88/0xDD) is encodable, and composing
 *      stage A  r2 <-> l2   (vshuff64x2 0x44 / 0xEE)
 *      stage B  r1 3-cycle  (vshuff64x2 0x88 / 0xDD)
 *      stage C  r0 <-> l0   (vunpcklo / vunpckhi)
 *  yields  o[k][l] = in[SW(l)][k],  SW = (0,1,4,5,2,3,6,7) (involution),
 *  verified by emulating the documented intrinsic semantics.  The lane
 *  residue SW is absorbed at compile time:
 *    - z_tr8a (transpose-in, de-interleave): register order is natural;
 *      lanes become y = SW(l), which the elementwise codelets never see.
 *    - z_tr8b (transpose-out): network output k is renamed into slot
 *      SW(k), so registers are natural again; lanes carry k2 = SW(l).
 *    - z_ilv: the interleave index vectors compose the inverse SW, so
 *      the output store is in natural memory order.
 * ================================================================== */
#if defined(__AVX512F__)
static inline __m512d z_add(__m512d a, __m512d b) { return _mm512_add_pd(a, b); }
static inline __m512d z_sub(__m512d a, __m512d b) { return _mm512_sub_pd(a, b); }
static inline __m512d z_fma(__m512d a, __m512d b, __m512d c) { return _mm512_fmadd_pd(a, b, c); }
static inline __m512d z_fnma(__m512d a, __m512d b, __m512d c) { return _mm512_fnmadd_pd(a, b, c); }
static inline __m512d z_bcast(double c) { return _mm512_set1_pd(c); }
static inline __m512d z_load(const double *p) { return _mm512_loadu_pd(p); }
static inline void z_store(double *p, __m512d a) { _mm512_storeu_pd(p, a); }

#define Z_TR8_NET(r, t, M0, M1, M2, M3, M4, M5, M6, M7)                        \
    do {                                                                       \
        __m512d u0 = _mm512_shuffle_f64x2((r)[0], (r)[4], 0x44);               \
        __m512d u4 = _mm512_shuffle_f64x2((r)[0], (r)[4], 0xEE);               \
        __m512d u1 = _mm512_shuffle_f64x2((r)[1], (r)[5], 0x44);               \
        __m512d u5 = _mm512_shuffle_f64x2((r)[1], (r)[5], 0xEE);               \
        __m512d u2 = _mm512_shuffle_f64x2((r)[2], (r)[6], 0x44);               \
        __m512d u6 = _mm512_shuffle_f64x2((r)[2], (r)[6], 0xEE);               \
        __m512d u3 = _mm512_shuffle_f64x2((r)[3], (r)[7], 0x44);               \
        __m512d u7 = _mm512_shuffle_f64x2((r)[3], (r)[7], 0xEE);               \
        __m512d w0 = _mm512_shuffle_f64x2(u0, u2, 0x88);                       \
        __m512d w2 = _mm512_shuffle_f64x2(u0, u2, 0xDD);                       \
        __m512d w1 = _mm512_shuffle_f64x2(u1, u3, 0x88);                       \
        __m512d w3 = _mm512_shuffle_f64x2(u1, u3, 0xDD);                       \
        __m512d w4 = _mm512_shuffle_f64x2(u4, u6, 0x88);                       \
        __m512d w6 = _mm512_shuffle_f64x2(u4, u6, 0xDD);                       \
        __m512d w5 = _mm512_shuffle_f64x2(u5, u7, 0x88);                       \
        __m512d w7 = _mm512_shuffle_f64x2(u5, u7, 0xDD);                       \
        (t)[M0] = _mm512_unpacklo_pd(w0, w1);                                  \
        (t)[M1] = _mm512_unpackhi_pd(w0, w1);                                  \
        (t)[M2] = _mm512_unpacklo_pd(w2, w3);                                  \
        (t)[M3] = _mm512_unpackhi_pd(w2, w3);                                  \
        (t)[M4] = _mm512_unpacklo_pd(w4, w5);                                  \
        (t)[M5] = _mm512_unpackhi_pd(w4, w5);                                  \
        (t)[M6] = _mm512_unpacklo_pd(w6, w7);                                  \
        (t)[M7] = _mm512_unpackhi_pd(w6, w7);                                  \
    } while (0)

/* natural register order out; lanes become SW(l) */
static inline void z_tr8a(const __m512d *r, __m512d *t)
{ Z_TR8_NET(r, t, 0, 1, 2, 3, 4, 5, 6, 7); }
/* output k renamed to slot SW(k): registers natural, lanes stay SW */
static inline void z_tr8b(const __m512d *r, __m512d *t)
{ Z_TR8_NET(r, t, 0, 1, 4, 5, 2, 3, 6, 7); }

/* Lane l of re/im holds k2 = SW(l), so the index vectors interleave lanes
 * SW(0..3) into the low line and SW(4..7) into the high line.  The two
 * permutes destroy different sources (re for the low, im for the high), so
 * neither source needs a compiler copy. */
static inline void z_ilv(__m512d re, __m512d im, double *dst)
{
    const __m512i il = _mm512_setr_epi64(0, 8, 1, 9, 4, 12, 5, 13);
    const __m512i ih = _mm512_setr_epi64(10, 2, 11, 3, 14, 6, 15, 7);
    _mm512_storeu_pd(dst,     _mm512_permutex2var_pd(re, il, im));
    _mm512_storeu_pd(dst + 8, _mm512_permutex2var_pd(im, ih, re));
}

static inline void z_ilv_nt(__m512d re, __m512d im, double *dst)
{
    const __m512i il = _mm512_setr_epi64(0, 8, 1, 9, 4, 12, 5, 13);
    const __m512i ih = _mm512_setr_epi64(10, 2, 11, 3, 14, 6, 15, 7);
    _mm512_stream_pd(dst,     _mm512_permutex2var_pd(re, il, im));
    _mm512_stream_pd(dst + 8, _mm512_permutex2var_pd(im, ih, re));
}

/* ---------------- fused-shape ("1f") primitives ----------------
 * Ported from L8_fusedaxes (its rounds 1-4 B=64-winning kernel).  Three
 * non-destructive 2-in/2-out lane butterflies, (a,b) <- (LO(a,b), HI(a,b)),
 * named by which register bit they trade with which lane bit:
 *   T1  vunpcklo/hi        r <-> l0
 *   T2  vshuff64x2 44/EE   r <-> l2
 *   T3  vshuff64x2 88/DD   r -> l2 -> l1 -> r   (the encodable 3-cycle) */
#define ZBF_T1(a, b)                                                           \
    do { __m512d t_ = _mm512_unpacklo_pd((a), (b));                            \
         (b) = _mm512_unpackhi_pd((a), (b)); (a) = t_; } while (0)
#define ZBF_T2(a, b)                                                           \
    do { __m512d t_ = _mm512_shuffle_f64x2((a), (b), 0x44);                    \
         (b) = _mm512_shuffle_f64x2((a), (b), 0xEE); (a) = t_; } while (0)
#define ZBF_T3(a, b)                                                           \
    do { __m512d t_ = _mm512_shuffle_f64x2((a), (b), 0x88);                    \
         (b) = _mm512_shuffle_f64x2((a), (b), 0xDD); (a) = t_; } while (0)

/* In-place 8x8 transpose, 24 ops: T2 on register bit 2, T3 on bit 1, T1 on
 * bit 0.  With input lane l = z = FPI[l] (FPI = 0,4,1,5,2,6,3,7, the residue
 * of the unpck deinterleave), output register j holds z = FPI[j] and the lane
 * residue moves to the untransposed axis; feed the codelet through fpiinv. */
#define ZTR8F(m)                                                               \
    do {                                                                       \
        ZBF_T2((m)[0], (m)[4]); ZBF_T2((m)[1], (m)[5]);                        \
        ZBF_T2((m)[2], (m)[6]); ZBF_T2((m)[3], (m)[7]);                        \
        ZBF_T3((m)[0], (m)[2]); ZBF_T3((m)[1], (m)[3]);                        \
        ZBF_T3((m)[4], (m)[6]); ZBF_T3((m)[5], (m)[7]);                        \
        ZBF_T1((m)[0], (m)[1]); ZBF_T1((m)[2], (m)[3]);                        \
        ZBF_T1((m)[4], (m)[5]); ZBF_T1((m)[6], (m)[7]);                        \
    } while (0)

/* Inverse transpose AND complex re-interleave fused into one 48-op network
 * over all 16 registers: T3 on k2 bit 0, T3 on k2 bit 1, T1 on the re/im bit
 * lands (l2,l1,l0) = (k2_1, k2_0, re/im) -- interleaved complex, each output
 * register a ready-to-store half-pencil (destination = f_off below). */
#define ZUNTRI(r, q)                                                           \
    do {                                                                       \
        ZBF_T3((r)[0], (r)[1]); ZBF_T3((r)[2], (r)[3]);                        \
        ZBF_T3((r)[4], (r)[5]); ZBF_T3((r)[6], (r)[7]);                        \
        ZBF_T3((q)[0], (q)[1]); ZBF_T3((q)[2], (q)[3]);                        \
        ZBF_T3((q)[4], (q)[5]); ZBF_T3((q)[6], (q)[7]);                        \
        ZBF_T3((r)[0], (r)[2]); ZBF_T3((r)[1], (r)[3]);                        \
        ZBF_T3((r)[4], (r)[6]); ZBF_T3((r)[5], (r)[7]);                        \
        ZBF_T3((q)[0], (q)[2]); ZBF_T3((q)[1], (q)[3]);                        \
        ZBF_T3((q)[4], (q)[6]); ZBF_T3((q)[5], (q)[7]);                        \
        ZBF_T1((r)[0], (q)[0]); ZBF_T1((r)[1], (q)[1]);                        \
        ZBF_T1((r)[2], (q)[2]); ZBF_T1((r)[3], (q)[3]);                        \
        ZBF_T1((r)[4], (q)[4]); ZBF_T1((r)[5], (q)[5]);                        \
        ZBF_T1((r)[6], (q)[6]); ZBF_T1((r)[7], (q)[7]);                        \
    } while (0)

/* FPI o fpiinv = identity: register j of the ZTR8F output, read through
 * fpiinv, is z = j natural. */
static const unsigned char fpiinv[8] = { 0, 2, 4, 6, 1, 3, 5, 7 };

/* Store destination (double offset k0*128 + half*8 into the k1-row) of
 * ZUNTRI's 16 outputs, r[0..7] then q[0..7]: the k0 order is the codelet's
 * bit-reversal-free output landing, (0,4,2,6) halves then (1,5,3,7) halves. */
static const int f_off[16] = { 0, 512, 256, 768, 8, 520, 264, 776,
                               128, 640, 384, 896, 136, 648, 392, 904 };

#define KV      __m512d
#define KADD    z_add
#define KSUB    z_sub
#define KFMA    z_fma
#define KFNMA   z_fnma
#define KBCAST  z_bcast
#define KLOAD   z_load
#define KSTORE  z_store
#define KTR8A   z_tr8a
#define KTR8B   z_tr8b
#define KILV    z_ilv
#define KPF1    KPF1_BURST
#define KPF2    KPF_NONE
#define KPF3    KPF_NONE
KERNFN kernel2_avx512(const double *restrict in, double *restrict out,
                      double *restrict scr, const double *restrict pf) KERNEL2_BODY
KERNFN kernel3_avx512(const double *restrict in, double *restrict out,
                      double *restrict scr, const double *restrict pf) KERNEL3_BODY
#undef KPF1
#undef KPF2
#undef KPF3
#define KPF1    KPF1_SPREAD
#define KPF2    KPF2_SPREAD
#define KPF3    KPF3_SPREAD
KERNFN kernel3s_avx512(const double *restrict in, double *restrict out,
                       double *restrict scr, const double *restrict pf) KERNEL3_BODY
#undef KPF1
#undef KPF2
#undef KPF3
#define KPF1    KPF1_SPREADW
#define KPF2    KPF2_SPREADW
#define KPF3    KPF3_SPREADW
KERNFN kernel3sw_avx512(const double *restrict in, double *restrict out,
                        double *restrict scr, const double *restrict pf) KERNEL3_BODY
#undef KPF1
#undef KPF2
#undef KPF3
#define KPF1    KPF1_SPREAD2
#define KPF2    KPF2_SPREAD2
#define KPF3    KPF_NONE
KERNFN kernel2s_avx512(const double *restrict in, double *restrict out,
                       double *restrict scr, const double *restrict pf) KERNEL2_BODY
#undef KPF1
#undef KPF2
#undef KPF3
#define KPF1    KPF1_SPREAD
#define KPF2    KPF2_SPREAD
#define KPF3    KPF3_SPREAD
#undef KILV
#define KILV    z_ilv_nt
KERNFN kernel3s_nt_avx512(const double *restrict in, double *restrict out,
                          double *restrict scr, const double *restrict pf) KERNEL3_BODY

/* ---- fused shape "1f", ported from L8_fusedaxes (phase structure, T-network
 * order, fpiinv feed and f_off store table all theirs; the codelet is my
 * 52-instruction RADIX8, which is natural-in/natural-out like their dft8s, so
 * it drops in unchanged).  Scratch use: sr = scr[0..511], si = scr[512..1023]
 * (split-planar, [k1][x] slots of 8 doubles; phase B reads each k1-row as 16
 * contiguous vectors).  Spread prefetch: 8 lines per iteration of each phase;
 * the _SW hooks add the mirror-image prefetchw on the next volume's output. */
#define K1FA_S(pf, x)  do { if (pf) PF8L((pf) + (size_t)(x) * 64); } while (0)
#define K1FB_S(pf, k)  do { if (pf) PF8L((pf) + 512 + (size_t)(k) * 64); } while (0)
#define K1FA_SW(pf, x) do { if (pf) { PF8L((pf) + (size_t)(x) * 64);            \
    PF8LW(out + NDBL + (size_t)(x) * 64); } } while (0)
#define K1FB_SW(pf, k) do { if (pf) { PF8L((pf) + 512 + (size_t)(k) * 64);      \
    PF8LW(out + NDBL + 512 + (size_t)(k) * 64); } } while (0)

#define KERNEL1F_BODY(PFA, PFB)                                                \
{                                                                              \
    const KV kc = KBCAST(C_RSQ2);                                              \
    double *sr = scr, *si = scr + 512;                                         \
    for (int x = 0; x < 8; ++x) {                                              \
        const double *p = in + (size_t)x * 128;                                \
        PFA(pf, x);                                                            \
        KV r[8], q[8];                                                         \
        for (int y = 0; y < 8; ++y) {                                          \
            KV a = z_load(p + (size_t)y * 16);                                 \
            KV b = z_load(p + (size_t)y * 16 + 8);                             \
            r[y] = _mm512_unpacklo_pd(a, b);   /* lane l = z = FPI[l] */       \
            q[y] = _mm512_unpackhi_pd(a, b);                                   \
        }                                                                      \
        RADIX8(r, q, r, q);                    /* y axis -> k1, elementwise */ \
        for (int k1 = 0; k1 < 8; ++k1) {                                       \
            z_store(sr + ((size_t)k1 * 8 + x) * 8, r[k1]);                     \
            z_store(si + ((size_t)k1 * 8 + x) * 8, q[k1]);                     \
        }                                                                      \
    }                                                                          \
    for (int k1 = 0; k1 < 8; ++k1) {                                           \
        PFB(pf, k1);                                                           \
        const double *rrow = sr + (size_t)k1 * 64;                             \
        const double *irow = si + (size_t)k1 * 64;                             \
        KV r[8], q[8], zr[8], zq[8];                                           \
        for (int x = 0; x < 8; ++x) {                                          \
            r[x] = z_load(rrow + (size_t)x * 8);                               \
            q[x] = z_load(irow + (size_t)x * 8);                               \
        }                                                                      \
        RADIX8(r, q, r, q);                    /* x axis -> k0, elementwise */ \
        ZTR8F(r);                              /* z into regs: reg j = FPI[j] */\
        ZTR8F(q);                                                              \
        for (int j = 0; j < 8; ++j) { zr[j] = r[fpiinv[j]]; zq[j] = q[fpiinv[j]]; } \
        RADIX8(zr, zq, zr, zq);                /* z axis -> k2 */              \
        ZUNTRI(zr, zq);                        /* untranspose + interleave */  \
        double *o = out + (size_t)k1 * 16;                                     \
        for (int j = 0; j < 8; ++j) {                                          \
            z_store(o + f_off[j],     zr[j]);                                  \
            z_store(o + f_off[j + 8], zq[j]);                                  \
        }                                                                      \
    }                                                                          \
}

KERNFN kernel1f_avx512(const double *restrict in, double *restrict out,
                       double *restrict scr, const double *restrict pf)
    KERNEL1F_BODY(K1FA_S, K1FB_S)
KERNFN kernel1fw_avx512(const double *restrict in, double *restrict out,
                        double *restrict scr, const double *restrict pf)
    KERNEL1F_BODY(K1FA_SW, K1FB_SW)

DEFRUN(run_2p_avx512,        kernel2_avx512,     0)
DEFRUN(run_2p_pfs_avx512,    kernel2s_avx512,    1)
DEFRUN(run_1f_avx512,        kernel1f_avx512,    0)
DEFRUN(run_1f_pfs_avx512,    kernel1f_avx512,    1)
DEFRUN(run_1f_pfsw_avx512,   kernel1fw_avx512,   1)
DEFRUN(run_3p_avx512,        kernel3_avx512,     0)
DEFRUN(run_3p_pf_avx512,     kernel3_avx512,     1)
DEFRUN(run_3p_pfs_avx512,    kernel3s_avx512,    1)
DEFRUN(run_3p_pfsw_avx512,   kernel3sw_avx512,   1)
DEFRUN(run_3p_nt_avx512,     kernel3s_nt_avx512, 0)
DEFRUN(run_3p_nt_pfs_avx512, kernel3s_nt_avx512, 1)
#undef KV
#undef KADD
#undef KSUB
#undef KFMA
#undef KFNMA
#undef KBCAST
#undef KLOAD
#undef KSTORE
#undef KTR8A
#undef KTR8B
#undef KILV
#undef KPF1
#undef KPF2
#undef KPF3
#endif /* AVX512F */

/* ================================================================== *
 *  Plan, self-tuning, and the API.
 * ================================================================== */
typedef void (*rfn_t)(const double *restrict, double *restrict, double *restrict,
                      int);

struct fft3d_plan {
    int    batch;
    int    nt;        /* chosen run uses non-temporal output stores */
    rfn_t  run;       /* whole-batch loop, kernel inlined, pf compile-time */
    rfn_t  run_safe;  /* same shape, ordinary stores (alignment fallback) */
    double *scr;      /* one volume of split-complex intermediate, 64B aligned */
    void   *scr_raw;
    const char *chosen;
};

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

const char *fft3d_name(void) { return "L8_radix8"; }

/* Filled in by fft3d_create() with the tuner's pick, as the panel_r2 verdict
 * asked; the default text covers a call before any plan exists. */
static char g_desc[224] =
    "radix-8 split-complex, 52-instr codelet, copy-free AVX-512 transposes; "
    "2p/fused-1f/3p shapes x spread-t0 x prefetchw, tuned at create";

const char *fft3d_description(void) { return g_desc; }

int fft3d_supports(int L) { return L == LL; }

#define MAXCAND 12

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != LL || batch <= 0) return NULL;

    fft3d_plan *p = (fft3d_plan *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->batch = batch;
    if (posix_memalign(&p->scr_raw, 64, (size_t)8 * SCRX * sizeof(double)) != 0 || !p->scr_raw) {
        free(p);
        return NULL;
    }
    p->scr = (double *)p->scr_raw;
    memset(p->scr, 0, (size_t)8 * SCRX * sizeof(double));

    /* ---- regime gate.  Prefetch/NT candidates are offered only when the batch's
     * in+out footprint exceeds ~0.9x the last-level cache: at cache-resident sizes
     * both are documented losers (L8_fusedaxes r2: prefetching L3-resident input =
     * 1.6x; node r3 at B=64: pf pick = +6.7 %), and offering them there is what
     * made the r3 tuner flip picks between runs. ---- */
    long l3 = -1;
#ifdef _SC_LEVEL3_CACHE_SIZE
    l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
    if (l3 <= 0) l3 = 22l << 20;   /* the scoring node's 22 MiB */
    size_t ws = (size_t)batch * 2u * NDBL * sizeof(double);   /* in + out bytes */
    int big = (ws * 10 > (size_t)l3 * 9) || getenv("L8R_ALLCAND") != NULL;

    /* ---- candidates on the best available backend (AVX-512 vs AVX2 settled on
     * the node in panel_r1).  FIRST entry is the node-validated default for the
     * regime (r3 picks: 2p at B=1 3/3, 3p at B=64 when it won, 3p-pf at
     * B=2048/16384 3/3); displacing it requires a >2 % measured win.  The burst-pf
     * NT combination is deliberately absent: a pass-1 prefetch burst and the NT
     * store drain fight for fill buffers (L8_fusedaxes r2, 1.6x). ---- */
    rfn_t       cand[MAXCAND];
    rfn_t       csafe[MAXCAND];
    const char *cname[MAXCAND];
    int         cnt[MAXCAND];   /* non-temporal flag */
    int nc = 0;

#define ADDC(fn, safe, name, ntf)                                              \
    do { cand[nc] = (fn); csafe[nc] = (safe); cname[nc] = (name);              \
         cnt[nc] = (ntf); ++nc; } while (0)

#if defined(__AVX512F__)
    if (batch == 1) {
        ADDC(run_2p_avx512, run_2p_avx512, "avx512-2p", 0);
        ADDC(run_3p_avx512, run_3p_avx512, "avx512-3p", 0);
        ADDC(run_1f_avx512, run_1f_avx512, "avx512-1f", 0);
    } else if (!big) {
        /* Mid regime (the node's B=64 cell).  Default is now 1f-pfs: the node
         * tuned to it 3/3 in panel_r5 (0.619 us), so putting it first removes
         * the tuner-noise exposure.  pfw is NOT offered here: prefetchw on
         * cache-resident output lines is a documented loser (L36_pfa +13%/+11%
         * in-arena at B=1/B=4; L8_fusedaxes +3% at an L3-resident wallaby
         * cell). */
        ADDC(run_1f_pfs_avx512, run_1f_pfs_avx512, "avx512-1f-pfs", 0);
        ADDC(run_2p_avx512,     run_2p_avx512,     "avx512-2p",     0);
        ADDC(run_1f_avx512,     run_1f_avx512,     "avx512-1f",     0);
        ADDC(run_2p_pfs_avx512, run_2p_pfs_avx512, "avx512-2p-pfs", 0);
        ADDC(run_3p_avx512,     run_3p_avx512,     "avx512-3p",     0);
    } else {
        /* Streaming regime.  Default 1f-pfs-pfw: L8_fusedaxes' fused+pfs+pfw
         * won both node streaming cells in panel_r5 (0.910 / 1.254, picked
         * 3/3) against my 3p-pfs (1.116 / 1.402); this reproduces that family
         * with my 52-instr codelet, and having {1f,3p} x {pfw,no-pfw} in one
         * table IS the panel_r5 VERDICT's fusion-vs-pfw isolating experiment.
         * NT candidates retired: NT stores lost on the node in every
         * streaming cell four rounds running (VERDICT 4.5: hide the RFO with
         * prefetchw, do not avoid it with NT). */
        ADDC(run_1f_pfsw_avx512, run_1f_pfsw_avx512, "avx512-1f-pfs-pfw", 0);
        ADDC(run_3p_pfsw_avx512, run_3p_pfsw_avx512, "avx512-3p-pfs-pfw", 0);
        ADDC(run_3p_pfs_avx512,  run_3p_pfs_avx512,  "avx512-3p-pfs",     0);
        ADDC(run_1f_pfs_avx512,  run_1f_pfs_avx512,  "avx512-1f-pfs",     0);
        ADDC(run_3p_pf_avx512,   run_3p_pf_avx512,   "avx512-3p-pf",      0);
    }
#elif defined(__AVX2__) && defined(__FMA__)
    if (batch == 1) {
        ADDC(run_2p_avx2, run_2p_avx2, "avx2-2p", 0);
        ADDC(run_3p_avx2, run_3p_avx2, "avx2-3p", 0);
    } else if (!big) {
        ADDC(run_3p_avx2, run_3p_avx2, "avx2-3p", 0);
        ADDC(run_2p_avx2, run_2p_avx2, "avx2-2p", 0);
    } else {
        ADDC(run_3p_pf_avx2,     run_3p_pf_avx2,  "avx2-3p-pf",     0);
        ADDC(run_3p_pfs_avx2,    run_3p_pfs_avx2, "avx2-3p-pfs",    0);
        ADDC(run_3p_nt_pfs_avx2, run_3p_pfs_avx2, "avx2-3p-nt-pfs", 1);
        ADDC(run_3p_nt_avx2,     run_3p_avx2,     "avx2-3p-nt",     1);
        ADDC(run_3p_avx2,        run_3p_avx2,     "avx2-3p",        0);
    }
#else
    if (batch == 1) {
        ADDC(run_2p_gen, run_2p_gen, "gen-2p", 0);
        ADDC(run_3p_gen, run_3p_gen, "gen-3p", 0);
    } else {
        ADDC(run_3p_gen, run_3p_gen, "gen-3p", 0);
        ADDC(run_2p_gen, run_2p_gen, "gen-2p", 0);
    }
#endif
#undef ADDC

    int pick  = 0;   /* the regime default */
    int tuned = 0;

    if (nc > 1) {
        /* Self-timing at the real batch size, capped machine-relatively at
         * 4x L3 of volumes (borrowed from L8_fusedaxes: same residency regime as
         * any larger batch; 5632 on the node, 8192 cap elsewhere). */
        long cap = (long)(4.0 * (double)l3 / (2.0 * NDBL * sizeof(double)));
        if (cap < 4096) cap = 4096;
        if (cap > 8192) cap = 8192;
        int nb = batch < (int)cap ? batch : (int)cap;
        size_t vol = NDBL * sizeof(double);
        void *raw_in = NULL, *raw_out = NULL;
        if (posix_memalign(&raw_in, 64, (size_t)nb * vol) == 0 && raw_in &&
            posix_memalign(&raw_out, 64, (size_t)nb * vol) == 0 && raw_out) {
            double *ti = (double *)raw_in, *to = (double *)raw_out;
            for (size_t i = 0; i < (size_t)nb * NDBL; ++i)
                ti[i] = 0.5 + 1e-3 * (double)(i % 37);
            memset(to, 0, (size_t)nb * vol);

            /* aim for ~1.5 ms of work per round per candidate */
            long reps = (long)(1500.0 / (0.7 * (double)nb));
            if (reps < 1) reps = 1;
            if (reps > 4000) reps = 4000;

            /* warm-up: settle the frequency licence before anything is believed */
            long nw = (long)(1000.0 / (0.7 * (double)nb));
            if (nw < 2) nw = 2;
            if (nw > 3000) nw = 3000;

            double best[MAXCAND];
            for (int v = 0; v < nc; ++v) {
                best[v] = 1e30;
                for (long w = 0; w < nw; ++w)
                    cand[v](ti, to, p->scr, nb);
            }
            /* Trials round-robin interleaved so frequency drift cannot favour
             * whoever runs first; ONE UNTIMED PASS before each timed block so a
             * candidate is measured against its own cache state, not its
             * predecessor's (plain and NT leave different L3 contents behind --
             * protocol borrowed from L8_fusedaxes r3). */
            for (int round = 0; round < 7; ++round) {
                for (int v = 0; v < nc; ++v) {
                    cand[v](ti, to, p->scr, nb);
                    double t0 = now_s();
                    for (long i = 0; i < reps; ++i)
                        cand[v](ti, to, p->scr, nb);
#if defined(__SSE2__)
                    _mm_sfence();
#endif
                    double dt = now_s() - t0;
                    if (dt < best[v]) best[v] = dt;
                }
            }
            int am = 0;
            for (int v = 1; v < nc; ++v)
                if (best[v] < best[am]) am = v;
            if (am != 0 && best[am] < best[0] * 0.98) {
                pick  = am;
                tuned = 1;
            }

            /* create-time override for A/B runs: L8R_FORCE=<candidate name> */
            const char *force = getenv("L8R_FORCE");
            if (force)
                for (int v = 0; v < nc; ++v)
                    if (strcmp(force, cname[v]) == 0) pick = v;

            if (getenv("L8R_TUNE_DEBUG")) {
                fprintf(stderr, "[L8_radix8 tune] batch=%d nb=%d reps=%ld big=%d pick=%s |",
                        batch, nb, reps, big, cname[pick]);
                for (int v = 0; v < nc; ++v)
                    fprintf(stderr, " %s=%.4fus", cname[v],
                            1e6 * best[v] / ((double)reps * (double)nb));
                fprintf(stderr, "\n");
            }
        }
        free(raw_in);
        free(raw_out);
        memset(p->scr, 0, (size_t)8 * SCRX * sizeof(double));
    }

    p->run      = cand[pick];
    p->run_safe = csafe[pick];
    p->chosen   = cname[pick];
    p->nt       = cnt[pick];

    snprintf(g_desc, sizeof(g_desc),
             "radix-8 split-complex 52-instr codelet; 2p/fused-1f/3p-seq shapes; "
             "tuner pick[B=%d]=%s (%s)",
             batch, p->chosen, tuned ? "tuned" : "default");
    return p;
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    const double *ip = (const double *)in;
    double *op = (double *)out;
    /* The ABI guarantees 64-byte-aligned buffers, which the non-temporal stores need;
     * fall back to the ordinary-store twin if that ever stops being true. */
    const int aligned = (((uintptr_t)op | (uintptr_t)ip) & 63u) == 0u;
    (aligned ? plan->run : plan->run_safe)(ip, op, plan->scr, plan->batch);
#if defined(__SSE2__)
    /* Non-temporal stores are weakly ordered; make the batch visible before returning.
     * One fence per execute, not per volume. */
    if (plan->nt && aligned) _mm_sfence();
#endif
}

void fft3d_destroy(fft3d_plan *plan)
{
    if (!plan) return;
    free(plan->scr_raw);
    free(plan);
}
