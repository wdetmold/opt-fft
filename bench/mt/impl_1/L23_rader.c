/* Carried over from the SINGLE-THREAD competition, where this file finished as
 * written below. Your job in the multicore phase is to parallelise it across
 * 32 cores without losing its single-core efficiency -- read
 * ../PANEL_BRIEF.md, and read ../../geom/strategies/L23_rader.md for the full
 * history of how this kernel got here.
 */
/* =============================================================================
 * L23_rader -- 23^3 complex-double forward DFT; Rader realization for p = 23.
 *
 * THE ROUND'S QUESTION (this entry exists to answer it): was the panel's L=17
 * prime win about primes being easy, or about 17 being a lucky prime?
 * p-1 = 22 = 2*11, so Rader's length-22 cyclic convolution splits, via
 * g^11 = -1 (g = 5 primitive root mod 23), into a CYCLIC-11 correlation with
 * the real kernel cos(2*pi*5^q/23) acting on the folded even part
 * u_j = x_j + x_{23-j}, plus a CYCLIC-11 correlation with the real kernel
 * sin(2*pi*5^q/23) acting on w_j = -i(x_j - x_{23-j}).  That +/- fold is the
 * ONLY sign-only reduction available: the quotient group has odd prime order
 * 11, so unlike 17 (where conv-16 kept splitting: cyclic-8 -> cyclic-4 +
 * negacyclic-4, all free), the chain stops immediately.  Every sub-quadratic
 * realization of a length-11 convolution known to the corpus (Winograd CRT
 * mod (x-1)*Phi_11: 20 mults but >130 unfusable adds; Karatsuba linear conv:
 * ~51 mults + ~150 adds; conv via two 11-point DFTs: ~2x total ops) EXCEEDS
 * the 121 fused FMAs of the direct circulant matvec on FMA hardware, where
 * one FMA = one add = one FP-port cycle.  Therefore the optimal Rader-23 on
 * this machine IS the conjugate-folded direct form -- mathematically the
 * dense-symmetric kernel, i.e. the same arithmetic L23_matrixsimd arrived at
 * from the dense side.  17 was lucky.  Full counts in the strategy record.
 *
 * WHAT THIS FILE DOES
 *   Row-column 3D: one folded 23-point kernel per axis, SIMD lanes = WC
 *   adjacent independent lines, all coefficients REAL so interleaved complex
 *   is already the right layout (one vector FMA per real coefficient per
 *   line-group).  Per line: 253 FMA + 44 add/sub = 297 vector FP ops; per
 *   volume 3*529 lines -> 943 kflop (yardstick 5*N*log2(N) = 824 kflop).
 *
 * KERNELS (both fully unrolled, generated; tuner picks within cmp-verified
 * bit-identical sets only):
 *   krn_ts : two-sweep, the 11 distinct cosines then the 11 distinct sines
 *            pinned in registers (all coefficient loads eliminated); optional
 *            parking of the 12 P accumulators in L1 across the sine sweep.
 *            Adapted from L23_matrixsimd's chunk23p (itself from
 *            L17_matrixsimd r3's pinned-constant kernel) -- attribution in
 *            the strategy record.
 *   krn_il : interleaved single-load sweep -- each x_j / x_{23-j} pair is
 *            loaded ONCE per line (23 vs 46 data loads), coefficients come
 *            from a pre-splatted j-major table as FMA memory operands.  This
 *            is the experiment the rival entry did not run at L=23: on the
 *            strided Y-pass loads, halving data loads halves L1 traffic.
 *
 * PASSES (structure adopted from L23_matrixsimd, in turn from L17_matrixsimd):
 *   X-last  (batch < 64):  per plane x: Y (lanes over z, transposing store
 *            into a 23x24-padded plane buffer), Z (lanes over ky, transposing
 *            store into t1); then X over the whole volume (lanes over the 529
 *            contiguous (y,z) pairs, plain stores into out).
 *   X-first (batch >= 64): X from `in` into t1 first, then Y,Z per plane into
 *            `out` -- spreads the output writes across the volume's compute
 *            (L17_matrixsimd r3, measured -14..-17% at batch there).
 *   The pass ORDER changes the association of the triple sum, so the class is
 *   a pure function of the batch size (bit-repeatability across processes);
 *   the tuner selects freely only within a class, among variants cmp-verified
 *   bit-identical on wallaby (width / parking / kernel form).
 *
 * ROUND panel_r7 additions (scheduling only; the arithmetic is settled):
 *   - t1 plane stride padded 1058 -> 1064 doubles (8512 B = 133 whole cache
 *     lines), so every X-pass store to t1 and every plane-phase base is
 *     64-byte aligned (was 16-mod-64: 3/4 of the X stores split lines).
 *     This was L23_matrixsimd r6's own unexecuted "Next" item 3.
 *   - deferred-Z plane schedule (adopted from L17_matrixsimd r6): pb is
 *     double-buffered and Y(x+1) runs between Y(x) and Z(x), so the Z
 *     group's loads never sit directly on their own Y group's stores
 *     (the one true intra-volume store->load junction, 23 per volume).
 *   - write-intent prefetch (p->pw): prefetchw the 133 out-lines of the
 *     plane the Z group is about to store, half before the Y group, half
 *     after (adopted from L8_fusedaxes/L36_pfa r5 via VERDICT r5 §4.5 --
 *     "hide the RFO, don't avoid it" -- with L17_matrixsimd r6's pacing).
 *     Judged jointly with pf as a plan-time (pf,pw) grid; a node bet.
 *   - licence-honest tuner warmup >= 1.5 ms per candidate (L17_rader r6).
 *
 * ROUND panel_r8 additions (scheduling only; node r7: B=1 47.854 us = 1.14x
 * the 41.9 us port floor, the board's tightest ratio; B=128 65.2 us = 1.56x
 * is the weak cell):
 *   - row-padded t1 (rp variants): rows 23 -> 24 complex (384 B = 6 whole
 *     lines), plane stride 1104 doubles, so the Y pass's strided row loads
 *     (previously 368 B apart, ~3/4 cache-line-split) are 64-byte aligned
 *     for 5 of 6 chunk columns.  Costs the X pass per-row chunking, +5
 *     chunks/volume (+1.2% FP).  This is r7's own "Next" item 2, endorsed
 *     by VERDICT r7 §6 as the remaining kernel lever at L=23.
 *   - in-pass X prefetch (pf==2): pull the X chunk 4 slots ahead of the one
 *     being computed (23 lines, one per x-plane), for the streaming batch
 *     where `in` misses LLC and the 8464-B plane stride defeats page-local
 *     hardware prefetch at each 4K boundary.
 *   - joint (variant, pf, pw) selection: the r7 tuner raced variants at
 *     pf=pw=0 and gridded (pf,pw) on the winner only, so e.g. deferred+pw
 *     was never compared against plain+pw at B=128 -- exactly the
 *     interacting-knobs mistake L17_matrixsimd r4/r6 warned about.  Now the
 *     top-3 stage-1 candidates each get the full legal (pf,pw) grid.
 *
 * ROUND panel_r9 (protocol only; node r8: B=1 47.688 = 1.14x floor flat,
 * B=4 49.557, B=128 64.835 -- the joint grid's rp+pf2+pw1 won B=128 but was
 * picked in only 1 of 3 processes, and B=1 reported a timed variant the
 * checked run did not make, VERDICT r8 3d):
 *   - DETERMINISTIC JOINT-CELL HYSTERETIC TUNER, replacing the two-stage
 *     top-3-then-grid racer: one flat canonical list of legal
 *     (variant, pf, pw) cells per regime, walked in order, a challenger
 *     displacing the incumbent only if >2% faster; two fixed-order sweeps
 *     with per-cell minima.  Adopted from L23_matrixsimd r8 (canonical-
 *     order hysteresis, which took its timed!=checked exposure 3 -> 0)
 *     and extended from variants to whole cells so interacting knobs stay
 *     jointly judged (my r8 grid lesson).  List head = the node's own r8
 *     pick per regime: batch >= 64 gets rp-t1+pf2+pw1 as incumbent
 *     (VERDICT r8 6's named instruction), batch < 64 plain two-sweep
 *     pf0 pw0.  pf=1 leaves the canonical list (0 picks in 3 rounds of
 *     wallaby grids and r8's node grid); X-last variants are no longer
 *     timed at plan time.  Both stay reachable via L23R_PF / L23R_FORCE.
 *   - tuner telemetry in the description string (pick and incumbent
 *     tuned us/t, arena size) -- L36_pfa r8's in-plan probe pattern, so
 *     the leaderboard itself carries one line of the node's tuner table.
 *   - kernels unchanged; checked by construction that every kernel call
 *     site passes compile-time-constant strides, so L45_pfa r8's gcc
 *     lea-spill pathology (runtime-offset codelet macros) cannot occur
 *     here and its asm barrier is not needed.
 *
 * ROUND panel_r10 (one pure deletion; node r9: B=1 47.795 = 1.14x floor,
 * B=4 49.524, B=128 64.882, timed == checked 3/3 everywhere, VERDICT r9
 * declares L=23 closed and the ±2% recompile noise floor real):
 *   - the -i rotation's vpxor is DELETED from both kernels: the sine
 *     tables (sd8/sd4 and the il-table sine halves) now carry alternating
 *     {+s, -s} lane signs, so w = SWAPRI(x_j - x_{23-j}) needs no sign
 *     flip -- s*(-x) == (-s)*x bitwise, so the output stays in the same
 *     bit class (cmp-verified, never assumed).  11 zmm XORs per
 *     line-group, ~4.5k per volume (3.6% of vector ALU ops), off the
 *     port-0/5 budget.  Adopted from L13_direct r9 (its -882-XOR sine-
 *     sign fold, node -1.4/-1.8% at batch), the only r9 change class
 *     with a positive node result.  Everything else is frozen.
 *
 * ROUND panel_r11 (tuner head-slot correction only; node r10: B=1 47.469,
 * B=4 49.678, B=128 64.793 -- all flat, VERDICT r10 keeps L=23 CLOSED and
 * "stop funding it"; the one instruction to this file is housekeeping):
 *   - the batch >= 64 canonical list head moves from (rp-t1, pf=2, pw=1)
 *     to (plain two-sweep X-first, pf=2, pw=1): L23_matrixsimd r10's
 *     FASTEST-KNOWN-HEAD rule, named by VERDICT r10 §6 as the correction
 *     this file must inherit as the surviving L=23 arm.  My r10 B=128
 *     pick flipped three ways across the node's processes because the
 *     rp-vs-flat gap (node r9 arenas: flat 3.4% FASTER) straddles the 2%
 *     hysteresis margin from the head slot; matrixsimd's flat head held
 *     3/3 at an equal cell time (64.874 vs 64.793).  Kernels, exec
 *     variants, knobs and both cell sets are otherwise untouched.
 *   - carried into the record from the retired arm (VERDICT r10 §7): the
 *     tail-paced pipeline null (tp = pick +2.3..+5.7%, 3/3) that closes
 *     the L=23 streaming schedule space for good.
 *
 * ASSUMPTIONS: L == 23 only; in/out distinct and 8-byte aligned (driver gives
 * 64); gcc/clang vector extensions with -ffp-contract=fast; self-#include
 * instantiates the template at 512-bit (WC=4) and 256-bit (WC=2).
 * =============================================================================
 */
#ifdef L23R_TEMPLATE
/* ===========================================================================
 *  TEMPLATE BODY -- instantiated once per vector width.
 *  Inputs: WC (complex per vector), SUF(x) (name mangler).
 * ===========================================================================
 */
#define VDW (2 * WC) /* doubles per vector */

typedef double SUF(vd) __attribute__((vector_size(8 * VDW), aligned(8)));
typedef long long SUF(vi) __attribute__((vector_size(8 * VDW)));
#define VT SUF(vd)
#define IT SUF(vi)

#define VLD(p) (*(const VT *)(p))
#define VST(p, v) (*(VT *)(p) = (v))

#if defined(__clang__)
#  define SHUF1(a, ...) __builtin_shufflevector(a, a, __VA_ARGS__)
#  define SHUF2(a, b, ...) __builtin_shufflevector(a, b, __VA_ARGS__)
#else
#  define SHUF1(a, ...) __builtin_shuffle(a, (IT){__VA_ARGS__})
#  define SHUF2(a, b, ...) __builtin_shuffle(a, b, (IT){__VA_ARGS__})
#endif

/* -i*t on interleaved complex used to be SWAPRI + a vpxor on the im lanes
 * (11 XORs per line-group, ~4.5k zmm XORs per volume).  Round r10 deletes
 * the XOR entirely by folding the im-lane negation into the SINE CONSTANTS:
 * the sd/ct sine entries are alternating {+s, -s} across the re/im lanes,
 * so R += S'*SWAPRI(d) computes (s*d_im, -s*d_re) = s * (-i*d) directly.
 * Bit-identical: s*(-x) and (-s)*x are the same IEEE double, and the FMA
 * rounds the same product.  Adopted from L13_direct r9 ("fold the -i sign
 * into the sine tables", -882 XORs there, node -1.4/-1.8% at batch); this
 * is the only change class with a positive node record in r9. */
#if WC == 4
#  define SWAPRI(x) SHUF1(x, 1, 0, 3, 2, 5, 4, 7, 6)
#else
#  define SWAPRI(x) SHUF1(x, 1, 0, 3, 2)
#endif

/* full-width read of entry i from a pre-splatted table */
#define CGET(base, i) VLD((base) + (size_t)(i) * VDW)

#if WC == 4
#  define CDIST(p) ((p)->cd8)
#  define SDIST(p) ((p)->sd8)
#  define CTAB(p) ((p)->ct8)
#else
#  define CDIST(p) ((p)->cd4)
#  define SDIST(p) ((p)->sd4)
#  define CTAB(p) ((p)->ct4)
#endif

/* ---- WCxWC transpose of complex elements (128-bit blocks) --------------- */
#if WC == 4
#  define TTILE(x, y)                                                          \
      do {                                                                     \
          VT q0 = SHUF2((x)[0], (x)[1], 0, 1, 2, 3, 8, 9, 10, 11);             \
          VT q1 = SHUF2((x)[2], (x)[3], 0, 1, 2, 3, 8, 9, 10, 11);             \
          VT q2 = SHUF2((x)[0], (x)[1], 4, 5, 6, 7, 12, 13, 14, 15);           \
          VT q3 = SHUF2((x)[2], (x)[3], 4, 5, 6, 7, 12, 13, 14, 15);           \
          (y)[0] = SHUF2(q0, q1, 0, 1, 4, 5, 8, 9, 12, 13);                    \
          (y)[1] = SHUF2(q0, q1, 2, 3, 6, 7, 10, 11, 14, 15);                  \
          (y)[2] = SHUF2(q2, q3, 0, 1, 4, 5, 8, 9, 12, 13);                    \
          (y)[3] = SHUF2(q2, q3, 2, 3, 6, 7, 10, 11, 14, 15);                  \
      } while (0)
#  define TILE(m0, e0, e1, e2, e3)                                             \
      do {                                                                     \
          VT xx[4], yy[4];                                                     \
          xx[0] = (e0); xx[1] = (e1); xx[2] = (e2); xx[3] = (e3);              \
          TTILE(xx, yy);                                                       \
          VST(dst + 0 * da + (m0) * 2, yy[0]);                                 \
          VST(dst + 1 * da + (m0) * 2, yy[1]);                                 \
          VST(dst + 2 * da + (m0) * 2, yy[2]);                                 \
          VST(dst + 3 * da + (m0) * 2, yy[3]);                                 \
      } while (0)
#else
#  define TTILE(x, y)                                                          \
      do {                                                                     \
          (y)[0] = SHUF2((x)[0], (x)[1], 0, 1, 4, 5);                          \
          (y)[1] = SHUF2((x)[0], (x)[1], 2, 3, 6, 7);                          \
      } while (0)
#  define TILE(m0, e0, e1)                                                     \
      do {                                                                     \
          VT xx[2], yy[2];                                                     \
          xx[0] = (e0); xx[1] = (e1);                                          \
          TTILE(xx, yy);                                                       \
          VST(dst + 0 * da + (m0) * 2, yy[0]);                                 \
          VST(dst + 1 * da + (m0) * 2, yy[1]);                                 \
      } while (0)
#endif

/* Store tail: X_0 = P_0, X_k = P_k + R_k, X_{23-k} = P_k - R_k.
 * tr=0: plain vector stores, output m at dst + m*db (lanes contiguous).
 * tr=1: WCxWC tile-transposed stores, outputs contiguous, lane stride da. */
#if WC == 4
#define L23R_STORE_TAIL()                                                      \
    do {                                                                       \
        if (!tr) {                                                             \
            VST(dst + 0 * db, P0);                                             \
            VST(dst + 1 * db, P1 + R1);    VST(dst + 2 * db, P2 + R2);         \
            VST(dst + 3 * db, P3 + R3);    VST(dst + 4 * db, P4 + R4);         \
            VST(dst + 5 * db, P5 + R5);    VST(dst + 6 * db, P6 + R6);         \
            VST(dst + 7 * db, P7 + R7);    VST(dst + 8 * db, P8 + R8);         \
            VST(dst + 9 * db, P9 + R9);    VST(dst + 10 * db, P10 + R10);      \
            VST(dst + 11 * db, P11 + R11); VST(dst + 12 * db, P11 - R11);      \
            VST(dst + 13 * db, P10 - R10); VST(dst + 14 * db, P9 - R9);        \
            VST(dst + 15 * db, P8 - R8);   VST(dst + 16 * db, P7 - R7);        \
            VST(dst + 17 * db, P6 - R6);   VST(dst + 18 * db, P5 - R5);        \
            VST(dst + 19 * db, P4 - R4);   VST(dst + 20 * db, P3 - R3);        \
            VST(dst + 21 * db, P2 - R2);   VST(dst + 22 * db, P1 - R1);        \
        } else {                                                               \
            TILE(0, P0, P1 + R1, P2 + R2, P3 + R3);                            \
            TILE(4, P4 + R4, P5 + R5, P6 + R6, P7 + R7);                       \
            TILE(8, P8 + R8, P9 + R9, P10 + R10, P11 + R11);                   \
            TILE(12, P11 - R11, P10 - R10, P9 - R9, P8 - R8);                  \
            TILE(16, P7 - R7, P6 - R6, P5 - R5, P4 - R4);                      \
            TILE(19, P4 - R4, P3 - R3, P2 - R2, P1 - R1);                      \
        }                                                                      \
    } while (0)
#else
#define L23R_STORE_TAIL()                                                      \
    do {                                                                       \
        if (!tr) {                                                             \
            VST(dst + 0 * db, P0);                                             \
            VST(dst + 1 * db, P1 + R1);    VST(dst + 2 * db, P2 + R2);         \
            VST(dst + 3 * db, P3 + R3);    VST(dst + 4 * db, P4 + R4);         \
            VST(dst + 5 * db, P5 + R5);    VST(dst + 6 * db, P6 + R6);         \
            VST(dst + 7 * db, P7 + R7);    VST(dst + 8 * db, P8 + R8);         \
            VST(dst + 9 * db, P9 + R9);    VST(dst + 10 * db, P10 + R10);      \
            VST(dst + 11 * db, P11 + R11); VST(dst + 12 * db, P11 - R11);      \
            VST(dst + 13 * db, P10 - R10); VST(dst + 14 * db, P9 - R9);        \
            VST(dst + 15 * db, P8 - R8);   VST(dst + 16 * db, P7 - R7);        \
            VST(dst + 17 * db, P6 - R6);   VST(dst + 18 * db, P5 - R5);        \
            VST(dst + 19 * db, P4 - R4);   VST(dst + 20 * db, P3 - R3);        \
            VST(dst + 21 * db, P2 - R2);   VST(dst + 22 * db, P1 - R1);        \
        } else {                                                               \
            TILE(0, P0, P1 + R1);                                              \
            TILE(2, P2 + R2, P3 + R3);                                         \
            TILE(4, P4 + R4, P5 + R5);                                         \
            TILE(6, P6 + R6, P7 + R7);                                         \
            TILE(8, P8 + R8, P9 + R9);                                         \
            TILE(10, P10 + R10, P11 + R11);                                    \
            TILE(12, P11 - R11, P10 - R10);                                    \
            TILE(14, P9 - R9, P8 - R8);                                        \
            TILE(16, P7 - R7, P6 - R6);                                        \
            TILE(18, P5 - R5, P4 - R4);                                        \
            TILE(20, P3 - R3, P2 - R2);                                        \
            TILE(21, P2 - R2, P1 - R1);                                        \
        }                                                                      \
    } while (0)
#endif

/* -----------------------------------------------------------------------
 *  krn_ts: two-sweep pinned-constant kernel.  cd holds the 11 distinct
 *  cosine magnitudes splatted; sd holds the 11 sine magnitudes with
 *  alternating {+s, -s} lane signs (the -i rotation, folded -- r10).
 *  Every coefficient is a register constant (cosines carry their own
 *  sign; per-(k,j) sine signs are compile-time +/- FMAs).
 *  pc=1 parks the 12 P accumulators in L1 scratch across the sine sweep
 *  (same arithmetic order, bit-identical; only data movement differs).
 * --------------------------------------------------------------------- */
static inline __attribute__((always_inline)) void
SUF(krn_ts)(const double *restrict src, long rs, double *restrict dst, long da,
            long db, const double *restrict cd, const double *restrict sd,
            double *restrict sc, int tr, int pc)
{
    VT P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11;
    VT R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11;

    { /* ---- cosine sweep: 12 accumulators, 11 pinned constants ---- */
        VT C1 = CGET(cd, 0), C2 = CGET(cd, 1), C3 = CGET(cd, 2),
           C4 = CGET(cd, 3), C5 = CGET(cd, 4), C6 = CGET(cd, 5),
           C7 = CGET(cd, 6), C8 = CGET(cd, 7), C9 = CGET(cd, 8),
           C10 = CGET(cd, 9), C11 = CGET(cd, 10);
        VT v0 = VLD(src);
        P0 = v0; P1 = v0; P2 = v0; P3 = v0; P4 = v0; P5 = v0;
        P6 = v0; P7 = v0; P8 = v0; P9 = v0; P10 = v0; P11 = v0;
        { VT u = VLD(src + 1 * rs) + VLD(src + 22 * rs);
          P0 += u; P1 += C1 * u; P2 += C2 * u; P3 += C3 * u;
          P4 += C4 * u; P5 += C5 * u; P6 += C6 * u; P7 += C7 * u;
          P8 += C8 * u; P9 += C9 * u; P10 += C10 * u; P11 += C11 * u; }
        { VT u = VLD(src + 2 * rs) + VLD(src + 21 * rs);
          P0 += u; P1 += C2 * u; P2 += C4 * u; P3 += C6 * u;
          P4 += C8 * u; P5 += C10 * u; P6 += C11 * u; P7 += C9 * u;
          P8 += C7 * u; P9 += C5 * u; P10 += C3 * u; P11 += C1 * u; }
        { VT u = VLD(src + 3 * rs) + VLD(src + 20 * rs);
          P0 += u; P1 += C3 * u; P2 += C6 * u; P3 += C9 * u;
          P4 += C11 * u; P5 += C8 * u; P6 += C5 * u; P7 += C2 * u;
          P8 += C1 * u; P9 += C4 * u; P10 += C7 * u; P11 += C10 * u; }
        { VT u = VLD(src + 4 * rs) + VLD(src + 19 * rs);
          P0 += u; P1 += C4 * u; P2 += C8 * u; P3 += C11 * u;
          P4 += C7 * u; P5 += C3 * u; P6 += C1 * u; P7 += C5 * u;
          P8 += C9 * u; P9 += C10 * u; P10 += C6 * u; P11 += C2 * u; }
        { VT u = VLD(src + 5 * rs) + VLD(src + 18 * rs);
          P0 += u; P1 += C5 * u; P2 += C10 * u; P3 += C8 * u;
          P4 += C3 * u; P5 += C2 * u; P6 += C7 * u; P7 += C11 * u;
          P8 += C6 * u; P9 += C1 * u; P10 += C4 * u; P11 += C9 * u; }
        { VT u = VLD(src + 6 * rs) + VLD(src + 17 * rs);
          P0 += u; P1 += C6 * u; P2 += C11 * u; P3 += C5 * u;
          P4 += C1 * u; P5 += C7 * u; P6 += C10 * u; P7 += C4 * u;
          P8 += C2 * u; P9 += C8 * u; P10 += C9 * u; P11 += C3 * u; }
        { VT u = VLD(src + 7 * rs) + VLD(src + 16 * rs);
          P0 += u; P1 += C7 * u; P2 += C9 * u; P3 += C2 * u;
          P4 += C5 * u; P5 += C11 * u; P6 += C4 * u; P7 += C3 * u;
          P8 += C10 * u; P9 += C6 * u; P10 += C1 * u; P11 += C8 * u; }
        { VT u = VLD(src + 8 * rs) + VLD(src + 15 * rs);
          P0 += u; P1 += C8 * u; P2 += C7 * u; P3 += C1 * u;
          P4 += C9 * u; P5 += C6 * u; P6 += C2 * u; P7 += C10 * u;
          P8 += C5 * u; P9 += C3 * u; P10 += C11 * u; P11 += C4 * u; }
        { VT u = VLD(src + 9 * rs) + VLD(src + 14 * rs);
          P0 += u; P1 += C9 * u; P2 += C5 * u; P3 += C4 * u;
          P4 += C10 * u; P5 += C1 * u; P6 += C8 * u; P7 += C6 * u;
          P8 += C3 * u; P9 += C11 * u; P10 += C2 * u; P11 += C7 * u; }
        { VT u = VLD(src + 10 * rs) + VLD(src + 13 * rs);
          P0 += u; P1 += C10 * u; P2 += C3 * u; P3 += C7 * u;
          P4 += C6 * u; P5 += C4 * u; P6 += C9 * u; P7 += C1 * u;
          P8 += C11 * u; P9 += C2 * u; P10 += C8 * u; P11 += C5 * u; }
        { VT u = VLD(src + 11 * rs) + VLD(src + 12 * rs);
          P0 += u; P1 += C11 * u; P2 += C1 * u; P3 += C10 * u;
          P4 += C2 * u; P5 += C9 * u; P6 += C3 * u; P7 += C8 * u;
          P8 += C4 * u; P9 += C7 * u; P10 += C5 * u; P11 += C6 * u; }
    }
    if (pc) {
        VST(sc + 0 * VDW, P0);   VST(sc + 1 * VDW, P1);
        VST(sc + 2 * VDW, P2);   VST(sc + 3 * VDW, P3);
        VST(sc + 4 * VDW, P4);   VST(sc + 5 * VDW, P5);
        VST(sc + 6 * VDW, P6);   VST(sc + 7 * VDW, P7);
        VST(sc + 8 * VDW, P8);   VST(sc + 9 * VDW, P9);
        VST(sc + 10 * VDW, P10); VST(sc + 11 * VDW, P11);
    }
    { /* ---- sine sweep: 11 accumulators, 11 pinned constants ---- */
        VT S1 = CGET(sd, 0), S2 = CGET(sd, 1), S3 = CGET(sd, 2),
           S4 = CGET(sd, 3), S5 = CGET(sd, 4), S6 = CGET(sd, 5),
           S7 = CGET(sd, 6), S8 = CGET(sd, 7), S9 = CGET(sd, 8),
           S10 = CGET(sd, 9), S11 = CGET(sd, 10);
        { VT w = SWAPRI(VLD(src + 1 * rs) - VLD(src + 22 * rs));
          R1 = S1 * w; R2 = S2 * w; R3 = S3 * w; R4 = S4 * w;
          R5 = S5 * w; R6 = S6 * w; R7 = S7 * w; R8 = S8 * w;
          R9 = S9 * w; R10 = S10 * w; R11 = S11 * w; }
        { VT w = SWAPRI(VLD(src + 2 * rs) - VLD(src + 21 * rs));
          R1 += S2 * w; R2 += S4 * w; R3 += S6 * w; R4 += S8 * w;
          R5 += S10 * w; R6 -= S11 * w; R7 -= S9 * w; R8 -= S7 * w;
          R9 -= S5 * w; R10 -= S3 * w; R11 -= S1 * w; }
        { VT w = SWAPRI(VLD(src + 3 * rs) - VLD(src + 20 * rs));
          R1 += S3 * w; R2 += S6 * w; R3 += S9 * w; R4 -= S11 * w;
          R5 -= S8 * w; R6 -= S5 * w; R7 -= S2 * w; R8 += S1 * w;
          R9 += S4 * w; R10 += S7 * w; R11 += S10 * w; }
        { VT w = SWAPRI(VLD(src + 4 * rs) - VLD(src + 19 * rs));
          R1 += S4 * w; R2 += S8 * w; R3 -= S11 * w; R4 -= S7 * w;
          R5 -= S3 * w; R6 += S1 * w; R7 += S5 * w; R8 += S9 * w;
          R9 -= S10 * w; R10 -= S6 * w; R11 -= S2 * w; }
        { VT w = SWAPRI(VLD(src + 5 * rs) - VLD(src + 18 * rs));
          R1 += S5 * w; R2 += S10 * w; R3 -= S8 * w; R4 -= S3 * w;
          R5 += S2 * w; R6 += S7 * w; R7 -= S11 * w; R8 -= S6 * w;
          R9 -= S1 * w; R10 += S4 * w; R11 += S9 * w; }
        { VT w = SWAPRI(VLD(src + 6 * rs) - VLD(src + 17 * rs));
          R1 += S6 * w; R2 -= S11 * w; R3 -= S5 * w; R4 += S1 * w;
          R5 += S7 * w; R6 -= S10 * w; R7 -= S4 * w; R8 += S2 * w;
          R9 += S8 * w; R10 -= S9 * w; R11 -= S3 * w; }
        { VT w = SWAPRI(VLD(src + 7 * rs) - VLD(src + 16 * rs));
          R1 += S7 * w; R2 -= S9 * w; R3 -= S2 * w; R4 += S5 * w;
          R5 -= S11 * w; R6 -= S4 * w; R7 += S3 * w; R8 += S10 * w;
          R9 -= S6 * w; R10 += S1 * w; R11 += S8 * w; }
        { VT w = SWAPRI(VLD(src + 8 * rs) - VLD(src + 15 * rs));
          R1 += S8 * w; R2 -= S7 * w; R3 += S1 * w; R4 += S9 * w;
          R5 -= S6 * w; R6 += S2 * w; R7 += S10 * w; R8 -= S5 * w;
          R9 += S3 * w; R10 += S11 * w; R11 -= S4 * w; }
        { VT w = SWAPRI(VLD(src + 9 * rs) - VLD(src + 14 * rs));
          R1 += S9 * w; R2 -= S5 * w; R3 += S4 * w; R4 -= S10 * w;
          R5 -= S1 * w; R6 += S8 * w; R7 -= S6 * w; R8 += S3 * w;
          R9 -= S11 * w; R10 -= S2 * w; R11 += S7 * w; }
        { VT w = SWAPRI(VLD(src + 10 * rs) - VLD(src + 13 * rs));
          R1 += S10 * w; R2 -= S3 * w; R3 += S7 * w; R4 -= S6 * w;
          R5 += S4 * w; R6 -= S9 * w; R7 += S1 * w; R8 += S11 * w;
          R9 -= S2 * w; R10 += S8 * w; R11 -= S5 * w; }
        { VT w = SWAPRI(VLD(src + 11 * rs) - VLD(src + 12 * rs));
          R1 += S11 * w; R2 -= S1 * w; R3 += S10 * w; R4 -= S2 * w;
          R5 += S9 * w; R6 -= S3 * w; R7 += S8 * w; R8 -= S4 * w;
          R9 += S7 * w; R10 -= S5 * w; R11 += S6 * w; }
    }
    if (pc) {
        P0 = VLD(sc + 0 * VDW);   P1 = VLD(sc + 1 * VDW);
        P2 = VLD(sc + 2 * VDW);   P3 = VLD(sc + 3 * VDW);
        P4 = VLD(sc + 4 * VDW);   P5 = VLD(sc + 5 * VDW);
        P6 = VLD(sc + 6 * VDW);   P7 = VLD(sc + 7 * VDW);
        P8 = VLD(sc + 8 * VDW);   P9 = VLD(sc + 9 * VDW);
        P10 = VLD(sc + 10 * VDW); P11 = VLD(sc + 11 * VDW);
    }
    L23R_STORE_TAIL();
}

/* -----------------------------------------------------------------------
 *  krn_il: interleaved single-load kernel.  Each (x_j, x_{23-j}) pair is
 *  loaded once; both accumulator banks advance together.  Coefficients are
 *  FMA memory operands from the pre-splatted j-major table ct (row j-1:
 *  entries 0..10 = cos(2pi kj/23) k=1..11, 11..21 = sin(2pi kj/23) with the
 *  per-(k,j) sign already in the value and, since r10, the -i rotation's
 *  alternating im-lane sign too).  23 accumulators live -> gcc spills a few on
 *  EVEX (32 regs), heavily on 16; the tuner decides whether the halved data
 *  loads pay for the spills.
 * --------------------------------------------------------------------- */
static inline __attribute__((always_inline)) void
SUF(krn_il)(const double *restrict src, long rs, double *restrict dst, long da,
            long db, const double *restrict ct, int tr)
{
    VT P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11;
    VT R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11;
    {
        VT v0 = VLD(src);
        P0 = v0; P1 = v0; P2 = v0; P3 = v0; P4 = v0; P5 = v0;
        P6 = v0; P7 = v0; P8 = v0; P9 = v0; P10 = v0; P11 = v0;
        { VT a = VLD(src + 1 * rs), b = VLD(src + 22 * rs);
          VT u = a + b, w = SWAPRI(a - b);
          const double *cj = ct + 0 * 22 * VDW;
          P0 += u; P1 += CGET(cj, 0) * u; P2 += CGET(cj, 1) * u;
          P3 += CGET(cj, 2) * u; P4 += CGET(cj, 3) * u; P5 += CGET(cj, 4) * u;
          P6 += CGET(cj, 5) * u; P7 += CGET(cj, 6) * u; P8 += CGET(cj, 7) * u;
          P9 += CGET(cj, 8) * u; P10 += CGET(cj, 9) * u; P11 += CGET(cj, 10) * u;
          R1 = CGET(cj, 11) * w; R2 = CGET(cj, 12) * w; R3 = CGET(cj, 13) * w;
          R4 = CGET(cj, 14) * w; R5 = CGET(cj, 15) * w; R6 = CGET(cj, 16) * w;
          R7 = CGET(cj, 17) * w; R8 = CGET(cj, 18) * w; R9 = CGET(cj, 19) * w;
          R10 = CGET(cj, 20) * w; R11 = CGET(cj, 21) * w; }
        { VT a = VLD(src + 2 * rs), b = VLD(src + 21 * rs);
          VT u = a + b, w = SWAPRI(a - b);
          const double *cj = ct + 1 * 22 * VDW;
          P0 += u; P1 += CGET(cj, 0) * u; P2 += CGET(cj, 1) * u;
          P3 += CGET(cj, 2) * u; P4 += CGET(cj, 3) * u; P5 += CGET(cj, 4) * u;
          P6 += CGET(cj, 5) * u; P7 += CGET(cj, 6) * u; P8 += CGET(cj, 7) * u;
          P9 += CGET(cj, 8) * u; P10 += CGET(cj, 9) * u; P11 += CGET(cj, 10) * u;
          R1 += CGET(cj, 11) * w; R2 += CGET(cj, 12) * w; R3 += CGET(cj, 13) * w;
          R4 += CGET(cj, 14) * w; R5 += CGET(cj, 15) * w; R6 += CGET(cj, 16) * w;
          R7 += CGET(cj, 17) * w; R8 += CGET(cj, 18) * w; R9 += CGET(cj, 19) * w;
          R10 += CGET(cj, 20) * w; R11 += CGET(cj, 21) * w; }
        { VT a = VLD(src + 3 * rs), b = VLD(src + 20 * rs);
          VT u = a + b, w = SWAPRI(a - b);
          const double *cj = ct + 2 * 22 * VDW;
          P0 += u; P1 += CGET(cj, 0) * u; P2 += CGET(cj, 1) * u;
          P3 += CGET(cj, 2) * u; P4 += CGET(cj, 3) * u; P5 += CGET(cj, 4) * u;
          P6 += CGET(cj, 5) * u; P7 += CGET(cj, 6) * u; P8 += CGET(cj, 7) * u;
          P9 += CGET(cj, 8) * u; P10 += CGET(cj, 9) * u; P11 += CGET(cj, 10) * u;
          R1 += CGET(cj, 11) * w; R2 += CGET(cj, 12) * w; R3 += CGET(cj, 13) * w;
          R4 += CGET(cj, 14) * w; R5 += CGET(cj, 15) * w; R6 += CGET(cj, 16) * w;
          R7 += CGET(cj, 17) * w; R8 += CGET(cj, 18) * w; R9 += CGET(cj, 19) * w;
          R10 += CGET(cj, 20) * w; R11 += CGET(cj, 21) * w; }
        { VT a = VLD(src + 4 * rs), b = VLD(src + 19 * rs);
          VT u = a + b, w = SWAPRI(a - b);
          const double *cj = ct + 3 * 22 * VDW;
          P0 += u; P1 += CGET(cj, 0) * u; P2 += CGET(cj, 1) * u;
          P3 += CGET(cj, 2) * u; P4 += CGET(cj, 3) * u; P5 += CGET(cj, 4) * u;
          P6 += CGET(cj, 5) * u; P7 += CGET(cj, 6) * u; P8 += CGET(cj, 7) * u;
          P9 += CGET(cj, 8) * u; P10 += CGET(cj, 9) * u; P11 += CGET(cj, 10) * u;
          R1 += CGET(cj, 11) * w; R2 += CGET(cj, 12) * w; R3 += CGET(cj, 13) * w;
          R4 += CGET(cj, 14) * w; R5 += CGET(cj, 15) * w; R6 += CGET(cj, 16) * w;
          R7 += CGET(cj, 17) * w; R8 += CGET(cj, 18) * w; R9 += CGET(cj, 19) * w;
          R10 += CGET(cj, 20) * w; R11 += CGET(cj, 21) * w; }
        { VT a = VLD(src + 5 * rs), b = VLD(src + 18 * rs);
          VT u = a + b, w = SWAPRI(a - b);
          const double *cj = ct + 4 * 22 * VDW;
          P0 += u; P1 += CGET(cj, 0) * u; P2 += CGET(cj, 1) * u;
          P3 += CGET(cj, 2) * u; P4 += CGET(cj, 3) * u; P5 += CGET(cj, 4) * u;
          P6 += CGET(cj, 5) * u; P7 += CGET(cj, 6) * u; P8 += CGET(cj, 7) * u;
          P9 += CGET(cj, 8) * u; P10 += CGET(cj, 9) * u; P11 += CGET(cj, 10) * u;
          R1 += CGET(cj, 11) * w; R2 += CGET(cj, 12) * w; R3 += CGET(cj, 13) * w;
          R4 += CGET(cj, 14) * w; R5 += CGET(cj, 15) * w; R6 += CGET(cj, 16) * w;
          R7 += CGET(cj, 17) * w; R8 += CGET(cj, 18) * w; R9 += CGET(cj, 19) * w;
          R10 += CGET(cj, 20) * w; R11 += CGET(cj, 21) * w; }
        { VT a = VLD(src + 6 * rs), b = VLD(src + 17 * rs);
          VT u = a + b, w = SWAPRI(a - b);
          const double *cj = ct + 5 * 22 * VDW;
          P0 += u; P1 += CGET(cj, 0) * u; P2 += CGET(cj, 1) * u;
          P3 += CGET(cj, 2) * u; P4 += CGET(cj, 3) * u; P5 += CGET(cj, 4) * u;
          P6 += CGET(cj, 5) * u; P7 += CGET(cj, 6) * u; P8 += CGET(cj, 7) * u;
          P9 += CGET(cj, 8) * u; P10 += CGET(cj, 9) * u; P11 += CGET(cj, 10) * u;
          R1 += CGET(cj, 11) * w; R2 += CGET(cj, 12) * w; R3 += CGET(cj, 13) * w;
          R4 += CGET(cj, 14) * w; R5 += CGET(cj, 15) * w; R6 += CGET(cj, 16) * w;
          R7 += CGET(cj, 17) * w; R8 += CGET(cj, 18) * w; R9 += CGET(cj, 19) * w;
          R10 += CGET(cj, 20) * w; R11 += CGET(cj, 21) * w; }
        { VT a = VLD(src + 7 * rs), b = VLD(src + 16 * rs);
          VT u = a + b, w = SWAPRI(a - b);
          const double *cj = ct + 6 * 22 * VDW;
          P0 += u; P1 += CGET(cj, 0) * u; P2 += CGET(cj, 1) * u;
          P3 += CGET(cj, 2) * u; P4 += CGET(cj, 3) * u; P5 += CGET(cj, 4) * u;
          P6 += CGET(cj, 5) * u; P7 += CGET(cj, 6) * u; P8 += CGET(cj, 7) * u;
          P9 += CGET(cj, 8) * u; P10 += CGET(cj, 9) * u; P11 += CGET(cj, 10) * u;
          R1 += CGET(cj, 11) * w; R2 += CGET(cj, 12) * w; R3 += CGET(cj, 13) * w;
          R4 += CGET(cj, 14) * w; R5 += CGET(cj, 15) * w; R6 += CGET(cj, 16) * w;
          R7 += CGET(cj, 17) * w; R8 += CGET(cj, 18) * w; R9 += CGET(cj, 19) * w;
          R10 += CGET(cj, 20) * w; R11 += CGET(cj, 21) * w; }
        { VT a = VLD(src + 8 * rs), b = VLD(src + 15 * rs);
          VT u = a + b, w = SWAPRI(a - b);
          const double *cj = ct + 7 * 22 * VDW;
          P0 += u; P1 += CGET(cj, 0) * u; P2 += CGET(cj, 1) * u;
          P3 += CGET(cj, 2) * u; P4 += CGET(cj, 3) * u; P5 += CGET(cj, 4) * u;
          P6 += CGET(cj, 5) * u; P7 += CGET(cj, 6) * u; P8 += CGET(cj, 7) * u;
          P9 += CGET(cj, 8) * u; P10 += CGET(cj, 9) * u; P11 += CGET(cj, 10) * u;
          R1 += CGET(cj, 11) * w; R2 += CGET(cj, 12) * w; R3 += CGET(cj, 13) * w;
          R4 += CGET(cj, 14) * w; R5 += CGET(cj, 15) * w; R6 += CGET(cj, 16) * w;
          R7 += CGET(cj, 17) * w; R8 += CGET(cj, 18) * w; R9 += CGET(cj, 19) * w;
          R10 += CGET(cj, 20) * w; R11 += CGET(cj, 21) * w; }
        { VT a = VLD(src + 9 * rs), b = VLD(src + 14 * rs);
          VT u = a + b, w = SWAPRI(a - b);
          const double *cj = ct + 8 * 22 * VDW;
          P0 += u; P1 += CGET(cj, 0) * u; P2 += CGET(cj, 1) * u;
          P3 += CGET(cj, 2) * u; P4 += CGET(cj, 3) * u; P5 += CGET(cj, 4) * u;
          P6 += CGET(cj, 5) * u; P7 += CGET(cj, 6) * u; P8 += CGET(cj, 7) * u;
          P9 += CGET(cj, 8) * u; P10 += CGET(cj, 9) * u; P11 += CGET(cj, 10) * u;
          R1 += CGET(cj, 11) * w; R2 += CGET(cj, 12) * w; R3 += CGET(cj, 13) * w;
          R4 += CGET(cj, 14) * w; R5 += CGET(cj, 15) * w; R6 += CGET(cj, 16) * w;
          R7 += CGET(cj, 17) * w; R8 += CGET(cj, 18) * w; R9 += CGET(cj, 19) * w;
          R10 += CGET(cj, 20) * w; R11 += CGET(cj, 21) * w; }
        { VT a = VLD(src + 10 * rs), b = VLD(src + 13 * rs);
          VT u = a + b, w = SWAPRI(a - b);
          const double *cj = ct + 9 * 22 * VDW;
          P0 += u; P1 += CGET(cj, 0) * u; P2 += CGET(cj, 1) * u;
          P3 += CGET(cj, 2) * u; P4 += CGET(cj, 3) * u; P5 += CGET(cj, 4) * u;
          P6 += CGET(cj, 5) * u; P7 += CGET(cj, 6) * u; P8 += CGET(cj, 7) * u;
          P9 += CGET(cj, 8) * u; P10 += CGET(cj, 9) * u; P11 += CGET(cj, 10) * u;
          R1 += CGET(cj, 11) * w; R2 += CGET(cj, 12) * w; R3 += CGET(cj, 13) * w;
          R4 += CGET(cj, 14) * w; R5 += CGET(cj, 15) * w; R6 += CGET(cj, 16) * w;
          R7 += CGET(cj, 17) * w; R8 += CGET(cj, 18) * w; R9 += CGET(cj, 19) * w;
          R10 += CGET(cj, 20) * w; R11 += CGET(cj, 21) * w; }
        { VT a = VLD(src + 11 * rs), b = VLD(src + 12 * rs);
          VT u = a + b, w = SWAPRI(a - b);
          const double *cj = ct + 10 * 22 * VDW;
          P0 += u; P1 += CGET(cj, 0) * u; P2 += CGET(cj, 1) * u;
          P3 += CGET(cj, 2) * u; P4 += CGET(cj, 3) * u; P5 += CGET(cj, 4) * u;
          P6 += CGET(cj, 5) * u; P7 += CGET(cj, 6) * u; P8 += CGET(cj, 7) * u;
          P9 += CGET(cj, 8) * u; P10 += CGET(cj, 9) * u; P11 += CGET(cj, 10) * u;
          R1 += CGET(cj, 11) * w; R2 += CGET(cj, 12) * w; R3 += CGET(cj, 13) * w;
          R4 += CGET(cj, 14) * w; R5 += CGET(cj, 15) * w; R6 += CGET(cj, 16) * w;
          R7 += CGET(cj, 17) * w; R8 += CGET(cj, 18) * w; R9 += CGET(cj, 19) * w;
          R10 += CGET(cj, 20) * w; R11 += CGET(cj, 21) * w; }
    }
    L23R_STORE_TAIL();
}

/* chunk start offsets covering a 23-long index; the last chunk overlaps the
 * previous and rewrites bit-identical values (cheaper than masking --
 * L17_matrixsimd's measured lesson, via L23_matrixsimd) */
#if WC == 4
static const int SUF(off23)[6] = {0, 4, 8, 12, 16, 19};
#  define NOFF23 6
#else
static const int SUF(off23)[12] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21};
#  define NOFF23 12
#endif

#define PBROW (24 * 2) /* plane buffer row stride, doubles (23 padded to 24) */

/* paced write-intent prefetch: pull [l0,l1) of the 133 cache lines of one
 * out plane (8464 B, 16-mod-64 base) into M-state ahead of the Z-group
 * stores.  prefetchw hides the read-for-ownership instead of avoiding it
 * (VERDICT r5 §4.5: pfw beat NT on the node at L=8 and L=36; NT lost four
 * rounds running).  Changes no bits. */
#define L23R_PWB(base, l0, l1)                                                 \
    do {                                                                       \
        const char *pwp = (const char *)(base);                                \
        for (int q5 = (l0); q5 < (l1); ++q5)                                   \
            __builtin_prefetch(pwp + (long)q5 * 64, 1, 3);                     \
    } while (0)

/* kernel dispatch: KRN is a compile-time constant inside each exec.
 *   0 = pinned two-sweep   1 = pinned two-sweep, P parked   2 = interleaved */
#define L23R_CALL(KRN, srcp, rs, dstp, da, db, tr)                             \
    do {                                                                       \
        if ((KRN) == 0)                                                        \
            SUF(krn_ts)((srcp), (rs), (dstp), (da), (db), cd, sd, sc, (tr), 0);\
        else if ((KRN) == 1)                                                   \
            SUF(krn_ts)((srcp), (rs), (dstp), (da), (db), cd, sd, sc, (tr), 1);\
        else                                                                   \
            SUF(krn_il)((srcp), (rs), (dstp), (da), (db), ct, (tr));           \
    } while (0)

/* X-pass chunk slots: slot i covers lanes i*WC..; the last is the
 * overlapping tail (529 % WC != 0 for both widths) */
#define NX23 (529 / WC + 1)
#define L23R_XF0(i) ((i) < 529 / WC ? (long)(i) * WC : (long)(529 - WC))

/* Row-padded-t1 X-pass chunk slots: chunks never cross a y-row, so slot i
 * decomposes as (y = i / NOFF23, column off23[i % NOFF23]).  The input
 * offset walks the packed 23-row `in` plane, the output offset the padded
 * 24-complex-row t1 plane.  Same lanes, same per-line arithmetic as the
 * flat X pass (lanes are independent lines), so rp variants stay in the
 * X-first bit class -- cmp-VERIFIED, never assumed. */
#define NXRP23 (23 * NOFF23)
static inline __attribute__((always_inline)) long SUF(xrp_in)(int i)
{
    int y = i / NOFF23, t = i - y * NOFF23;
    return 46L * y + 2 * SUF(off23)[t];
}
static inline __attribute__((always_inline)) long SUF(xrp_t1)(int i)
{
    int y = i / NOFF23, t = i - y * NOFF23;
    return 48L * y + 2 * SUF(off23)[t];
}
#define L23R_NXQ(RP) ((RP) ? NXRP23 : NX23)
#define L23R_XIN(RP, i) ((RP) ? SUF(xrp_in)(i) : 2 * L23R_XF0(i))
#define L23R_XT1(RP, i) ((RP) ? SUF(xrp_t1)(i) : 2 * L23R_XF0(i))
#define L23R_T1PS(RP) ((RP) ? L23R_T1PR : L23R_T1P)
#define L23R_YRS(RP) ((RP) ? 48 : 46)

/* in-pass input prefetch (pf==2): pull the X chunk 4 slots ahead (23 lines,
 * one per 8464-B-strided x-plane) -- for the streaming batch, where `in`
 * misses LLC and the hardware streamer restarts at every 4K boundary of
 * each of the 23 concurrent planes.  Changes no bits. */
#define L23R_PFX(RP, i, NXQ)                                                   \
    do {                                                                       \
        if (p->pf == 2 && (i) + 4 < (NXQ)) {                                   \
            const char *px = (const char *)(vin + L23R_XIN(RP, (i) + 4));      \
            for (int q4 = 0; q4 < 23; ++q4)                                    \
                __builtin_prefetch(px + (long)q4 * 8464, 0, 3);                \
        }                                                                      \
    } while (0)

/* -----------------------------------------------------------------------
 *  Exec variants.  REORD: 0 = X-last, 1 = X-first (see header).
 *  NTC: Z pass writes a staging plane, then streamed (NT) to `out` --
 *  skips the read-for-ownership of every output line in the streaming
 *  regime (adopted from L23_matrixsimd's exec_rfn <- L17_matrixsimd).
 *  The cross-volume input prefetch (p->pf, A/B'd at create, changes no
 *  bits) is from L17_winograd r2 via L17_matrixsimd.
 * --------------------------------------------------------------------- */
#define L23R_EXEC(NAME, KRN, REORD) L23R_EXEC_N(NAME, KRN, REORD, 0, 0)
#define L23R_EXEC_N(NAME, KRN, REORD, NTC, RP)                                 \
    static __attribute__((unused)) void                                       \
    NAME(const fft3d_plan *restrict p, const double _Complex *restrict in,     \
         double _Complex *restrict out)                                        \
    {                                                                          \
        const double *restrict cd = CDIST(p);                                  \
        const double *restrict sd = SDIST(p);                                  \
        const double *restrict ct = CTAB(p);                                   \
        double *restrict pb = p->pb;                                           \
        double *restrict t1 = p->t1;                                           \
        double *restrict sc = p->sc;                                           \
        const int nb = p->batch;                                               \
        (void)cd; (void)sd; (void)ct;                                          \
        for (int b = 0; b < nb; ++b) {                                         \
            const double *vin = (const double *)(in + (size_t)b * 12167);      \
            double *vout = (double *)(out + (size_t)b * 12167);                \
            if (REORD) { /* X first: in -> t1[kx][y][z], padded plane rows */  \
                const int nxq = L23R_NXQ(RP);                                  \
                for (int i = 0; i < nxq; ++i) {                                \
                    L23R_PFX(RP, i, nxq);                                      \
                    L23R_CALL(KRN, vin + L23R_XIN(RP, i), 1058,                \
                              t1 + L23R_XT1(RP, i), 2, L23R_T1PS(RP), 0);      \
                }                                                              \
            }                                                                  \
            for (int x = 0; x < 23; ++x) {                                     \
                const double *pin2 = REORD ? (const double *)t1                \
                                             + (long)x * L23R_T1PS(RP)        \
                                           : vin + (long)x * 1058;             \
                double *pt = NTC ? p->ps                                       \
                                 : (REORD ? vout + (long)x * 1058              \
                                          : t1 + (long)x * L23R_T1P);         \
                if (REORD && p->pf == 1 && b + 1 < nb) {                       \
                    /* pull the NEXT volume's input toward L2 while the       \
                     * plane phase runs from cache-resident scratch: 133      \
                     * lines per plane x 23 planes covers the volume */       \
                    const char *nx = (const char *)(vin + 24334)               \
                                     + (long)x * 8464;                         \
                    for (int q3 = 0; q3 < 133; ++q3)                           \
                        __builtin_prefetch(nx + (long)q3 * 64, 0, 2);          \
                }                                                              \
                if (REORD && !NTC && p->pw) L23R_PWB(pt, 0, 67);               \
                /* Y: along y (row stride 46, 48 if rp), lanes over z */      \
                for (int t = 0; t < NOFF23; ++t) {                             \
                    long f0 = SUF(off23)[t];                                   \
                    L23R_CALL(KRN, pin2 + 2 * f0, L23R_YRS(RP),                \
                              pb + f0 * PBROW, PBROW, 2, 1);                   \
                }                                                              \
                if (REORD && !NTC && p->pw) L23R_PWB(pt, 67, 133);             \
                /* Z: along z (row stride PBROW), lanes over ky, transposed */\
                for (int t = 0; t < NOFF23; ++t) {                             \
                    long f0 = SUF(off23)[t];                                   \
                    L23R_CALL(KRN, pb + 2 * f0, PBROW, pt + f0 * 46,           \
                              46, 2, 1);                                       \
                }                                                              \
                if (NTC) l23r_ntcopy(vout + (long)x * 1058, p->ps, 1058);      \
            }                                                                  \
            if (!REORD) { /* X last: t1 -> out */                              \
                for (int i = 0; i < NX23; ++i) {                               \
                    long f0 = L23R_XF0(i);                                     \
                    L23R_CALL(KRN, t1 + 2 * f0, L23R_T1P, vout + 2 * f0,       \
                              2, 1058, 0);                                     \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

L23R_EXEC(SUF(exec_ts), 0, 0)
L23R_EXEC(SUF(exec_tsp), 1, 0)
L23R_EXEC(SUF(exec_il), 2, 0)
L23R_EXEC(SUF(exec_ts_f), 0, 1)
L23R_EXEC(SUF(exec_tsp_f), 1, 1)
L23R_EXEC(SUF(exec_il_f), 2, 1)
L23R_EXEC_N(SUF(exec_ts_fn), 0, 1, 1, 0)
L23R_EXEC_N(SUF(exec_ts_fr), 0, 1, 0, 1)
#undef L23R_EXEC
#undef L23R_EXEC_N

/* -----------------------------------------------------------------------
 *  Deferred-Z plane schedule (adopted from L17_matrixsimd r6's execm_x*d):
 *  X-first, with the plane buffer double-buffered (pb / pb2) and the Y
 *  group run ONE PLANE AHEAD of the Z group -- order Y0; Y1 Z0; Y2 Z1;
 *  ... Y22 Z21; Z22 -- so a Z group's loads never sit directly on their
 *  own Y group's stores to the same 8.6 KiB buffer (a store->load
 *  forwarding junction once per plane, 23 per volume, with no independent
 *  work behind the group tail).  Pure scheduling: every chunk computes the
 *  same values from the same operands, planes are independent, so the
 *  output stays in the X-first bit class (cmp-VERIFIED, never assumed).
 * --------------------------------------------------------------------- */
#define L23R_EXEC_D(NAME, KRN, RP)                                             \
    static __attribute__((unused)) void                                       \
    NAME(const fft3d_plan *restrict p, const double _Complex *restrict in,     \
         double _Complex *restrict out)                                        \
    {                                                                          \
        const double *restrict cd = CDIST(p);                                  \
        const double *restrict sd = SDIST(p);                                  \
        const double *restrict ct = CTAB(p);                                   \
        double *restrict t1 = p->t1;                                           \
        double *restrict sc = p->sc;                                           \
        const int nb = p->batch;                                               \
        (void)cd; (void)sd; (void)ct;                                          \
        for (int b = 0; b < nb; ++b) {                                         \
            const double *vin = (const double *)(in + (size_t)b * 12167);      \
            double *vout = (double *)(out + (size_t)b * 12167);                \
            {                                                                  \
                const int nxq = L23R_NXQ(RP);                                  \
                for (int i = 0; i < nxq; ++i) {                                \
                    L23R_PFX(RP, i, nxq);                                      \
                    L23R_CALL(KRN, vin + L23R_XIN(RP, i), 1058,                \
                              t1 + L23R_XT1(RP, i), 2, L23R_T1PS(RP), 0);      \
                }                                                              \
            }                                                                  \
            { /* prologue: Y(0) into pb */                                     \
                double *restrict pb = p->pb;                                   \
                for (int t = 0; t < NOFF23; ++t) {                             \
                    long f0 = SUF(off23)[t];                                   \
                    L23R_CALL(KRN, (const double *)t1 + 2 * f0,                \
                              L23R_YRS(RP), pb + f0 * PBROW, PBROW, 2, 1);     \
                }                                                              \
            }                                                                  \
            for (int x = 0; x < 23; ++x) {                                     \
                double *restrict pbc = (x & 1) ? p->pb2 : p->pb;               \
                double *restrict pbn = (x & 1) ? p->pb : p->pb2;               \
                double *pt = vout + (long)x * 1058;                            \
                if (p->pf == 1 && b + 1 < nb) {                                \
                    const char *nx = (const char *)(vin + 24334)               \
                                     + (long)x * 8464;                         \
                    for (int q3 = 0; q3 < 133; ++q3)                           \
                        __builtin_prefetch(nx + (long)q3 * 64, 0, 2);          \
                }                                                              \
                if (p->pw) L23R_PWB(pt, 0, 67);                                \
                if (x + 1 < 23) { /* Y(x+1) between Y(x) and Z(x) */           \
                    const double *pin2 = (const double *)t1                    \
                                         + (long)(x + 1) * L23R_T1PS(RP);      \
                    for (int t = 0; t < NOFF23; ++t) {                         \
                        long f0 = SUF(off23)[t];                               \
                        L23R_CALL(KRN, pin2 + 2 * f0, L23R_YRS(RP),            \
                                  pbn + f0 * PBROW, PBROW, 2, 1);              \
                    }                                                          \
                }                                                              \
                if (p->pw) L23R_PWB(pt, 67, 133);                              \
                for (int t = 0; t < NOFF23; ++t) {                             \
                    long f0 = SUF(off23)[t];                                   \
                    L23R_CALL(KRN, pbc + 2 * f0, PBROW, pt + f0 * 46,          \
                              46, 2, 1);                                       \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

L23R_EXEC_D(SUF(exec_ts_fd), 0, 0)
L23R_EXEC_D(SUF(exec_ts_frd), 0, 1)
#undef L23R_EXEC_D

/* -----------------------------------------------------------------------
 *  Cross-volume software pipelining (adopted from L17_matrixsimd r4's
 *  exec18 via L23_matrixsimd's exec_p*): X-first with t1 double-buffered;
 *  volume b+1's X chunks are interleaved, ~6 per plane at two insertion
 *  points, into volume b's plane phase, so every volume's input read (but
 *  the first) overlaps compute.  Pure scheduling: each chunk computes the
 *  same values from the same operands as plain X-first; bit-identity is
 *  cmp-VERIFIED, never assumed (contraction can differ at a new site).
 * --------------------------------------------------------------------- */
#define L23R_EXEC_P(NAME, KRN, NTC, RP)                                        \
    static __attribute__((unused)) void                                       \
    NAME(const fft3d_plan *restrict p, const double _Complex *restrict in,     \
         double _Complex *restrict out)                                        \
    {                                                                          \
        const double *restrict cd = CDIST(p);                                  \
        const double *restrict sd = SDIST(p);                                  \
        const double *restrict ct = CTAB(p);                                   \
        double *restrict pb = p->pb;                                           \
        double *restrict sc = p->sc;                                           \
        const int nb = p->batch;                                               \
        (void)cd; (void)sd; (void)ct;                                          \
        { /* prologue: volume 0's X pass, the only un-overlapped read */      \
            const double *vin0 = (const double *)in;                           \
            for (int i = 0; i < L23R_NXQ(RP); ++i) {                           \
                L23R_CALL(KRN, vin0 + L23R_XIN(RP, i), 1058,                   \
                          p->t1 + L23R_XT1(RP, i), 2, L23R_T1PS(RP), 0);       \
            }                                                                  \
        }                                                                      \
        for (int b = 0; b < nb; ++b) {                                         \
            double *vout = (double *)(out + (size_t)b * 12167);                \
            const double *restrict cur = (b & 1) ? p->t1b : p->t1;             \
            double *restrict nxt = (b & 1) ? p->t1 : p->t1b;                   \
            const double *vinN = (const double *)(in + (size_t)(b + 1) * 12167);\
            const int hn = (b + 1 < nb);                                       \
            for (int x = 0; x < 23; ++x) {                                     \
                int i0 = 0, ih = 0, i1 = 0;                                    \
                if (hn) {                                                      \
                    i0 = (x * L23R_NXQ(RP)) / 23;                              \
                    i1 = ((x + 1) * L23R_NXQ(RP)) / 23;                        \
                    ih = i0 + (i1 - i0) / 2;                                   \
                    for (int i = i0; i < ih; ++i) {                            \
                        L23R_CALL(KRN, vinN + L23R_XIN(RP, i), 1058,           \
                                  nxt + L23R_XT1(RP, i), 2, L23R_T1PS(RP), 0); \
                    }                                                          \
                }                                                              \
                const double *pin2 = cur + (long)x * L23R_T1PS(RP);            \
                double *pt = NTC ? p->ps : vout + (long)x * 1058;              \
                if (!NTC && p->pw) L23R_PWB(pt, 0, 67);                        \
                for (int t = 0; t < NOFF23; ++t) {                             \
                    long f0 = SUF(off23)[t];                                   \
                    L23R_CALL(KRN, pin2 + 2 * f0, L23R_YRS(RP),                \
                              pb + f0 * PBROW, PBROW, 2, 1);                   \
                }                                                              \
                if (!NTC && p->pw) L23R_PWB(pt, 67, 133);                      \
                if (hn) {                                                      \
                    for (int i = ih; i < i1; ++i) {                            \
                        L23R_CALL(KRN, vinN + L23R_XIN(RP, i), 1058,           \
                                  nxt + L23R_XT1(RP, i), 2, L23R_T1PS(RP), 0); \
                    }                                                          \
                }                                                              \
                for (int t = 0; t < NOFF23; ++t) {                             \
                    long f0 = SUF(off23)[t];                                   \
                    L23R_CALL(KRN, pb + 2 * f0, PBROW, pt + f0 * 46,           \
                              46, 2, 1);                                       \
                }                                                              \
                if (NTC) l23r_ntcopy(vout + (long)x * 1058, p->ps, 1058);      \
            }                                                                  \
        }                                                                      \
    }

L23R_EXEC_P(SUF(exec_ts_p), 0, 0, 0)
L23R_EXEC_P(SUF(exec_ts_pn), 0, 1, 0)
L23R_EXEC_P(SUF(exec_ts_frp), 0, 0, 1)
#undef L23R_EXEC_P

#undef VDW
#undef VT
#undef IT
#undef VLD
#undef VST
#undef SHUF1
#undef SHUF2
#undef SWAPRI
#undef CGET
#undef CDIST
#undef SDIST
#undef CTAB
#undef L23R_STORE_TAIL
#undef TTILE
#undef TILE
#undef NOFF23
#undef PBROW
#undef L23R_PWB
#undef L23R_CALL
#undef NX23
#undef L23R_XF0
#undef NXRP23
#undef L23R_NXQ
#undef L23R_XIN
#undef L23R_XT1
#undef L23R_T1PS
#undef L23R_YRS
#undef L23R_PFX

#else /* ================= main body ================= */

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../fft3d_api.h"

/* Streaming copy of one finished 23x23 plane (1058 doubles) into `out`:
 * NT stores skip the read-for-ownership of every output line in the
 * streaming regime.  Plane starts are 16 (mod 64)-byte aligned (8464-byte
 * plane stride), hence the step-up loop.  Same bits, so it stays in the
 * bit class.  Adopted from L23_matrixsimd's l23_ntcopy (<- L17_matrixsimd). */
#if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE2__)
#  include <immintrin.h>
static void l23r_ntcopy(double *restrict dst, const double *restrict src, size_t nd)
{
    size_t i = 0;
#if defined(__AVX512F__)
    while (i + 2 <= nd && (((uintptr_t)(dst + i)) & 63u)) {
        _mm_stream_pd(dst + i, _mm_loadu_pd(src + i));
        i += 2;
    }
    for (; i + 8 <= nd; i += 8) _mm512_stream_pd(dst + i, _mm512_loadu_pd(src + i));
#elif defined(__AVX__)
    while (i + 2 <= nd && (((uintptr_t)(dst + i)) & 31u)) {
        _mm_stream_pd(dst + i, _mm_loadu_pd(src + i));
        i += 2;
    }
    for (; i + 4 <= nd; i += 4) _mm256_stream_pd(dst + i, _mm256_loadu_pd(src + i));
#endif
    for (; i + 2 <= nd; i += 2) _mm_stream_pd(dst + i, _mm_loadu_pd(src + i));
    for (; i < nd; ++i) dst[i] = src[i];
    _mm_sfence();
}
#else
static void l23r_ntcopy(double *restrict dst, const double *restrict src, size_t nd)
{
    memcpy(dst, src, nd * sizeof *dst);
}
#endif

struct fft3d_plan {
    int L, batch;
    int pf; /* cross-volume input prefetch in the X-first plane phase (A/B'd) */
    int pw; /* write-intent prefetch on the Z-group out stores (A/B'd) */
    void (*exec)(const struct fft3d_plan *, const double _Complex *, double _Complex *);
    double *cd8, *sd8; /* 11 distinct cos / sin magnitudes, splatted 8x */
    double *cd4, *sd4; /* the same, splatted 4x */
    double *ct8, *ct4; /* interleaved-kernel tables: 11 rows x 22, splatted */
    double *sc;        /* 12 vectors: parked P accumulators (pc=1 kernel) */
    double *pb;        /* 23 x 24 complex plane buffer (pass Y -> pass Z) */
    double *pb2;       /* second plane buffer for the deferred-Z variants */
    double *t1;        /* one 23^3 volume, plane stride padded to 1064 dbl */
    double *t1b;       /* second volume buffer for the pipelined variants */
    double *ps;        /* staging plane for the NT-copy variants */
    void *block;
    double _Complex *ti, *to; /* transient buffers for the plan-time tuner */
    size_t tn;
};

/* t1 plane stride, doubles: 23x23 complex = 1058, padded to 1064 so every
 * plane base (and so every X-pass store) is 64-byte aligned (8512 = 133
 * whole lines; 8464 was 16-mod-64). */
#define L23R_T1P 1064

/* row-padded t1 layout (rp variants): rows 23 -> 24 complex (384 B = 6
 * whole lines, so every Y-pass row load lands 64-byte aligned for chunk
 * columns 0/4/8/12/16), plane stride 23 rows = 1104 doubles (8832 B, a
 * 64-byte multiple; decorrelated from `in`'s 8464-B stride mod 4096, which
 * preserves the r7 anti-aliasing property). */
#define L23R_T1PR 1104

/* ---- instantiate the kernel template: 512-bit and 256-bit ---- */
#if defined(__has_include)
#  if __has_include("L23_rader.c")
#    define L23R_SELF "L23_rader.c"
#  elif __has_include("impl/L23_rader.c")
#    define L23R_SELF "impl/L23_rader.c"
#  endif
#else
#  define L23R_SELF "L23_rader.c"
#endif

#ifdef L23R_SELF
#  define L23R_TEMPLATE 1
#  define WC 4
#  define SUF(x) x##_w4
#  include L23R_SELF
#  undef SUF
#  undef WC
#  undef L23R_TEMPLATE

#  define L23R_TEMPLATE 2
#  define WC 2
#  define SUF(x) x##_w2
#  include L23R_SELF
#  undef SUF
#  undef WC
#  undef L23R_TEMPLATE
#else
#  error "L23_rader.c must be able to #include itself"
#endif

static double l23r_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

const char *fft3d_name(void) { return "L23_rader"; }

static const char *g_desc =
    "Rader p=23 folded to cyclic-11 pair = dense conj-folded kernel, SIMD lines";

const char *fft3d_description(void) { return g_desc; }

int fft3d_supports(int L) { return L == 23; }

static int l23r_verbose(void)
{
    const char *e = getenv("L23R_VERBOSE");
    return e && *e && *e != '0';
}

static void l23r_tune_free(fft3d_plan *p)
{
    free(p->ti); free(p->to);
    p->ti = NULL; p->to = NULL; p->tn = 0;
}

/* Deterministic pseudo-random tuning data (realistic magnitudes suffice). */
static int l23r_tune_alloc(fft3d_plan *p, int nv)
{
    size_t n = (size_t)nv * 12167;
    if (p->tn >= n) return 1;
    l23r_tune_free(p);
    if (posix_memalign((void **)&p->ti, 64, n * sizeof *p->ti) != 0) { p->ti = NULL; return 0; }
    if (posix_memalign((void **)&p->to, 64, n * sizeof *p->to) != 0) {
        free(p->ti); p->ti = NULL; p->to = NULL; return 0;
    }
    p->tn = n;
    unsigned sr = 20260821u;
    for (size_t i = 0; i < n; ++i) {
        sr = sr * 1103515245u + 12345u;
        double a = (double)(sr >> 8) / 8388608.0 - 1.0;
        sr = sr * 1103515245u + 12345u;
        double b = (double)(sr >> 8) / 8388608.0 - 1.0;
        p->ti[i] = a + b * (double _Complex)I;
    }
    memset(p->to, 0, n * sizeof *p->to);
    return 1;
}

/* Finish tuning: apply forced-pair env overrides (L23R_PF / L23R_PW, for
 * same-window A/B on a loaded dev machine), bake the picks into the
 * description string so the leaderboard records them, free tuner arenas.
 * The picked cell's tuned time and the canonical incumbent's are baked in
 * too -- in-plan measurement routed out through fft3d_description(), the
 * L36_pfa r8 pattern the r8 verdict told the panel to adopt: it puts one
 * line of the node's own tuner table on the leaderboard without a monitor
 * run. */
static char g_dbuf[240];
static double g_tpick, g_tinc; /* tuned us/transform: picked cell, list head */
static int g_tnv;              /* tuner arena volumes (0 = tuner never ran) */
static void l23r_tune_finish(fft3d_plan *p, const char *tag)
{
    const char *e = getenv("L23R_PF");
    if (e && *e) {
        int v = atoi(e);
        p->pf = v < 0 ? 0 : (v > 2 ? 2 : v);
    }
    e = getenv("L23R_PW");
    if (e && *e) p->pw = atoi(e) ? 1 : 0;
    if (g_tnv > 0)
        snprintf(g_dbuf, sizeof g_dbuf,
                 "%s, pf=%d pw=%d, tuner pick=%.2f inc=%.2f us/t nv=%d",
                 tag, p->pf, p->pw, g_tpick, g_tinc, g_tnv);
    else
        snprintf(g_dbuf, sizeof g_dbuf, "%s, pf=%d pw=%d", tag, p->pf, p->pw);
    g_desc = g_dbuf;
    l23r_tune_free(p);
}

/* Streaming-regime tuner arena: in + out together exceed ~2.5x L3
 * (machine-relative sizing, from L36_mixedradix via L17_matrixsimd r3). */
static int l23r_tune_nv(int batch)
{
    long l3 = -1;
#ifdef _SC_LEVEL3_CACHE_SIZE
    l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
    int cap = 144;
    if (l3 > 0) {
        double nv = 2.5 * (double)l3 / (2.0 * 12167.0 * 16.0);
        cap = nv < 144.0 ? 144 : (nv > 448.0 ? 448 : (int)nv);
    }
    return batch < cap ? batch : cap;
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != 23 || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;

    const size_t nd = 11;       /* distinct constants */
    const size_t nt = 11 * 22;  /* il table entries */
    const size_t nsc = 12 * 8;
    const size_t npb = 23 * 24 * 2;
    const size_t nt1 = 23 * L23R_T1PR; /* sized for the larger (rp) layout */
    const size_t nps = 1058 + 6;
    const size_t ntot = 24 * nd + 12 * nt + nsc + 2 * npb + 2 * nt1 + nps + 640;
    void *blk = NULL;
    if (posix_memalign(&blk, 64, ntot * sizeof(double)) != 0 || !blk) {
        free(p);
        return NULL;
    }
    memset(blk, 0, ntot * sizeof(double));
    p->block = blk;

#define L23R_ALIGN64(q) ((double *)(((uintptr_t)(q) + 63u) & ~(uintptr_t)63u))
    double *q = (double *)blk;
    p->cd8 = q; q += 8 * nd;
    p->sd8 = q; q += 8 * nd;
    p->cd4 = q; q += 4 * nd;
    p->sd4 = q; q += 4 * nd;
    q = L23R_ALIGN64(q);
    p->ct8 = q; q += 8 * nt;
    q = L23R_ALIGN64(q);
    p->ct4 = q; q += 4 * nt;
    q = L23R_ALIGN64(q);
    p->sc = q; q += nsc;
    q = L23R_ALIGN64(q);
    p->pb = q; q += npb;
    q = L23R_ALIGN64(q);
    p->pb2 = q; q += npb;
    q = L23R_ALIGN64(q);
    p->t1 = q; q += nt1;
    q = L23R_ALIGN64(q);
    p->t1b = q; q += nt1;
    q = L23R_ALIGN64(q);
    p->ps = q;
#undef L23R_ALIGN64

    /* long double trig so the double table is correctly rounded.
     * SINE entries carry the -i rotation's im-lane sign: alternating
     * {+s, -s} across the re/im lanes (see the SWAPRI comment in the
     * template), which deleted the per-j vpxor from both kernels (r10). */
    const long double twopi = 6.283185307179586476925286766559005768394L;
    for (int m = 1; m <= 11; ++m) {
        double c = (double)cosl(twopi * (long double)m / 23.0L);
        double s = (double)sinl(twopi * (long double)m / 23.0L);
        for (int t = 0; t < 8; ++t) p->cd8[(m - 1) * 8 + t] = c;
        for (int t = 0; t < 8; ++t) p->sd8[(m - 1) * 8 + t] = (t & 1) ? -s : s;
        for (int t = 0; t < 4; ++t) p->cd4[(m - 1) * 4 + t] = c;
        for (int t = 0; t < 4; ++t) p->sd4[(m - 1) * 4 + t] = (t & 1) ? -s : s;
    }
    for (int j = 1; j <= 11; ++j) {
        for (int k = 1; k <= 11; ++k) {
            int m = (k * j) % 23;
            double c = (double)cosl(twopi * (long double)m / 23.0L);
            double s = (double)sinl(twopi * (long double)m / 23.0L);
            size_t rc = (size_t)(j - 1) * 22 + (k - 1);
            size_t rs2 = (size_t)(j - 1) * 22 + 11 + (k - 1);
            for (int t = 0; t < 8; ++t) p->ct8[rc * 8 + t] = c;
            for (int t = 0; t < 8; ++t) p->ct8[rs2 * 8 + t] = (t & 1) ? -s : s;
            for (int t = 0; t < 4; ++t) p->ct4[rc * 4 + t] = c;
            for (int t = 0; t < 4; ++t) p->ct4[rs2 * 4 + t] = (t & 1) ? -s : s;
        }
    }

    /* ---- variant selection.  The X order is a pure function of the batch
     * size (X-last < 64 <= X-first) because the two associate the triple sum
     * differently; the tuner times ALL variants for the record but may only
     * PICK within the batch class, and only among forms cmp-verified
     * bit-identical on wallaby (see strategy record for the verification). */
    {
        typedef void (*l23r_fn)(const fft3d_plan *, const double _Complex *,
                                double _Complex *);
        enum { NC = 26 };
        static const l23r_fn cand[NC] = {
            exec_ts_w4,   exec_tsp_w4,   exec_il_w4,
            exec_ts_w2,   exec_tsp_w2,   exec_il_w2,
            exec_ts_f_w4, exec_tsp_f_w4, exec_il_f_w4,
            exec_ts_f_w2, exec_tsp_f_w2, exec_il_f_w2,
            exec_ts_fn_w4, exec_ts_fn_w2,
            exec_ts_p_w4,  exec_ts_p_w2,
            exec_ts_pn_w4, exec_ts_pn_w2,
            exec_ts_fd_w4, exec_ts_fd_w2,
            exec_ts_fr_w4,  exec_ts_fr_w2,
            exec_ts_frd_w4, exec_ts_frd_w2,
            exec_ts_frp_w4, exec_ts_frp_w2,
        };
        static const char *const tags[NC] = {
            "rader23 folded pair, 512-bit, pinned two-sweep, X-last",
            "rader23 folded pair, 512-bit, pinned two-sweep P-parked, X-last",
            "rader23 folded pair, 512-bit, interleaved single-load, X-last",
            "rader23 folded pair, 256-bit, pinned two-sweep, X-last",
            "rader23 folded pair, 256-bit, pinned two-sweep P-parked, X-last",
            "rader23 folded pair, 256-bit, interleaved single-load, X-last",
            "rader23 folded pair, 512-bit, pinned two-sweep, X-first",
            "rader23 folded pair, 512-bit, pinned two-sweep P-parked, X-first",
            "rader23 folded pair, 512-bit, interleaved single-load, X-first",
            "rader23 folded pair, 256-bit, pinned two-sweep, X-first",
            "rader23 folded pair, 256-bit, pinned two-sweep P-parked, X-first",
            "rader23 folded pair, 256-bit, interleaved single-load, X-first",
            "rader23 folded pair, 512-bit, pinned, X-first, NT planes",
            "rader23 folded pair, 256-bit, pinned, X-first, NT planes",
            "rader23 folded pair, 512-bit, pinned, X-first, pipelined",
            "rader23 folded pair, 256-bit, pinned, X-first, pipelined",
            "rader23 folded pair, 512-bit, pinned, X-first, pipelined, NT planes",
            "rader23 folded pair, 256-bit, pinned, X-first, pipelined, NT planes",
            "rader23 folded pair, 512-bit, pinned, X-first, deferred-Z",
            "rader23 folded pair, 256-bit, pinned, X-first, deferred-Z",
            "rader23 folded pair, 512-bit, pinned, X-first, rp-t1",
            "rader23 folded pair, 256-bit, pinned, X-first, rp-t1",
            "rader23 folded pair, 512-bit, pinned, X-first, deferred-Z, rp-t1",
            "rader23 folded pair, 256-bit, pinned, X-first, deferred-Z, rp-t1",
            "rader23 folded pair, 512-bit, pinned, X-first, pipelined, rp-t1",
            "rader23 folded pair, 256-bit, pinned, X-first, pipelined, rp-t1",
        };
        /* Selectable set: X-first only, ALL batch sizes.  cmp-verified on
         * wallaby: the X-first variants are bit-identical to each other
         * (kernel form, width, parking, rp, deferral, pipelining and NT
         * change no bits) and prefetch knobs change no bits, so ANY cell
         * below keeps output independent of the tuner's pick.  X-last
         * variants (0-5, the other bit class) are never selectable and,
         * from this round, never timed at plan time either (adopted from
         * L23_matrixsimd r8) -- they remain reachable via L23R_FORCE.
         *
         * DETERMINISTIC JOINT-CELL HYSTERESIS (round r9, adopted from
         * L23_matrixsimd r8 and extended from variants to (variant,pf,pw)
         * cells): r8's min-of-noise top-3-then-grid picked differently in
         * different processes -- three different picks at B=1 on the node
         * -- so the scored minimum came from a variant the harness never
         * checked (VERDICT r8 3d).  Now ONE flat canonical list of legal
         * cells per regime is walked in order and a challenger displaces
         * the incumbent only if >2% faster: near-ties resolve to the same
         * cell in every process (timed == checked), a real mechanism still
         * wins.  Extending the hysteresis to whole cells keeps r8's
         * interacting-knobs lesson (deferred+pw must race plain+pw, not
         * inherit the pf=pw=0 ranking).
         *
         * List order is a policy statement (L45_pfa r8's pool-ordering
         * lesson), and the round-r11 rule for choosing it is
         * L23_matrixsimd r10's FASTEST-KNOWN-HEAD correction: the head
         * must be the fastest cell the SCORING NODE has itself measured,
         * not a hoped-for one -- a wrong head converts a real sub-margin
         * speed difference into cross-process pick flips (my r10 B=128
         * pick flipped three ways from the rp head while matrixsimd's
         * flat head held 3/3).  Heads:
         *   batch >= 64: PLAIN two-sweep X-first + pf=2 + pw=1.  The
         *     node's own arenas displaced rp with flat 3/3 in r9
         *     (pick=59.5-60.2 vs inc=61.6-62.4 us/t, flat 3.4% faster);
         *     VERDICT r9: "the r8 B=128 win was the knobs (pf=2, pw=1),
         *     not the folded-pair layout".  rp rows move below the whole
         *     flat family: rp must now be >2% faster to be picked, and
         *     the node measured it slower.
         *   batch < 64: plain pinned two-sweep X-first, pf=0 pw=0 -- the
         *     node's checked pick at B=1 and B=4 in r7-r10 (pick == inc
         *     3/3 in r9 and r10).
         * pf=1 (cross-volume plane-phase prefetch) lost every wallaby grid
         * cell three rounds running and was not picked in any node cell of
         * r8's joint grid; it leaves the canonical list (reachable via
         * L23R_PF).  256-bit cells are kept at pf=pw=0 only, as fallback
         * for non-AVX-512 hosts; they have never been within 15% on either
         * AVX-512 machine. */
        typedef struct { signed char v, pf, pw; } l23r_cell;
        static const l23r_cell cellsB[] = {
            /* batch >= 64; incumbent = fastest node-measured cell (r9/r10
             * arenas: flat displaced rp 3/3; r10 checked pick, run 3) */
            {6, 2, 1},  {6, 0, 1},  {6, 2, 0},  {6, 0, 0},  /* plain        */
            {20, 2, 1}, {20, 0, 1}, {20, 2, 0}, {20, 0, 0}, /* rp-t1        */
            {18, 2, 1}, {18, 0, 1}, {18, 2, 0}, {18, 0, 0}, /* deferred-Z   */
            {22, 2, 1}, {22, 0, 1}, {22, 2, 0}, {22, 0, 0}, /* rp+deferred  */
            {7, 0, 1},  {7, 0, 0},  {8, 0, 0},              /* park, il     */
            {14, 0, 1}, {14, 0, 0}, {24, 0, 1}, {24, 0, 0}, /* pipelined    */
            {12, 2, 0}, {12, 0, 0}, {16, 0, 0},             /* NT planes    */
            {9, 0, 0},  {10, 0, 0}, {11, 0, 0}, {13, 0, 0}, /* 256-bit      */
            {15, 0, 0}, {19, 0, 0}, {21, 0, 0}, {23, 0, 0}, {25, 0, 0},
        };
        static const l23r_cell cellsS[] = {
            /* batch < 64; incumbent = node r7/r8 B=1 and B=4 checked pick */
            {6, 0, 0},  {6, 0, 1},                          /* plain        */
            {20, 0, 0}, {20, 0, 1},                         /* rp-t1        */
            {7, 0, 0},  {7, 0, 1},                          /* P-parked     */
            {18, 0, 0}, {18, 0, 1}, {22, 0, 0}, {22, 0, 1}, /* deferred-Z   */
            {14, 0, 0}, {14, 0, 1}, {24, 0, 0}, {24, 0, 1}, /* pipelined    */
            {8, 0, 0},  {12, 0, 0}, {16, 0, 0},             /* il, NT       */
            {9, 0, 0},  {10, 0, 0}, {11, 0, 0}, {13, 0, 0}, /* 256-bit      */
            {15, 0, 0}, {19, 0, 0}, {21, 0, 0}, {23, 0, 0}, {25, 0, 0},
        };
        const l23r_cell *cells = batch < 64 ? cellsS : cellsB;
        const int ncell = batch < 64 ? (int)(sizeof cellsS / sizeof cellsS[0])
                                     : (int)(sizeof cellsB / sizeof cellsB[0]);
        int bestc = 0;
        p->exec = cand[cells[0].v];
        p->pf = cells[0].pf;
        p->pw = cells[0].pw;
        g_desc = tags[cells[0].v];

        const char *force = getenv("L23R_FORCE");
        if (force && *force) {
            int v = atoi(force);
            if (v >= 0 && v < NC) {
                p->exec = cand[v];
                p->pf = 0;
                p->pw = 0;
                g_desc = tags[v];
                if (l23r_verbose())
                    fprintf(stderr, "[L23_rader] FORCED variant %d: %s\n", v, tags[v]);
                l23r_tune_finish(p, tags[v]);
                return p;
            }
        }

        {
            double best[64];
            for (int c = 0; c < ncell; ++c) best[c] = 1e30;
            int nv = batch < 64 ? (batch < 8 ? batch : 8) : l23r_tune_nv(batch);
            if (l23r_tune_alloc(p, nv)) {
                int sb = p->batch;
                p->batch = nv;
                int inner = batch < 64 ? (32 + nv - 1) / nv : 1;
                int nrep = batch < 64 ? 5 : 3;
                /* TWO full fixed-order sweeps, per-cell min across both
                 * (L17_rader r6 via L23_matrixsimd r8): one sweep on a
                 * drifting clock can mis-rank cells far apart in the list.
                 * Cells are never interleaved (512/256-bit licence
                 * transition mis-ranks -- L17_matrixsimd r1) and each gets
                 * a licence-honest warmup >= 1.5 ms (L17_rader r6), past
                 * Intel's ~670 us licence-up dwell. */
                for (int sweep = 0; sweep < 2; ++sweep)
                    for (int c = 0; c < ncell; ++c) {
                        p->exec = cand[cells[c].v];
                        p->pf = cells[c].pf;
                        p->pw = cells[c].pw;
                        double tw = l23r_now();
                        do p->exec(p, p->ti, p->to);
                        while (l23r_now() - tw < 1.5e-3);
                        for (int r = 0; r < nrep; ++r) {
                            double t0 = l23r_now();
                            for (int q2 = 0; q2 < inner; ++q2)
                                p->exec(p, p->ti, p->to);
                            double dt = l23r_now() - t0;
                            if (dt < best[c]) best[c] = dt;
                        }
                    }
                p->batch = sb;
                /* hysteretic walk: >2% to displace the running incumbent */
                for (int c = 1; c < ncell; ++c)
                    if (best[c] < 0.98 * best[bestc]) bestc = c;
                g_tinc = best[0] * 1e6 / ((double)nv * inner);
                g_tpick = best[bestc] * 1e6 / ((double)nv * inner);
                g_tnv = nv;
                if (l23r_verbose())
                    for (int c = 0; c < ncell; ++c)
                        fprintf(stderr,
                                "[L23_rader tune nv=%d] %-62s pf=%d pw=%d %8.2f us/transform%s\n",
                                nv, tags[cells[c].v], cells[c].pf, cells[c].pw,
                                best[c] * 1e6 / ((double)nv * inner),
                                c == bestc ? "  <== kept" : "");
            }
            p->exec = cand[cells[bestc].v];
            p->pf = cells[bestc].pf;
            p->pw = cells[bestc].pw;
            g_desc = tags[cells[bestc].v];
        }
    }

    l23r_tune_finish(p, g_desc);
    return p;
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    p->exec(p, in, out);
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    l23r_tune_free(p);
    free(p->block);
    free(p);
}

#endif /* template / main body */
