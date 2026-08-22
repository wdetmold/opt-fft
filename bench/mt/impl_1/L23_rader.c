/* =============================================================================
 * L23_rader -- 23^3 complex-double forward DFT; Rader realization for p = 23.
 * MULTICORE phase (mt_r1).  Phase-1 history: ../../geom/strategies/L23_rader.md.
 *
 * THE ARITHMETIC (settled in phase 1, untouched here):
 * p-1 = 22 = 2*11, so Rader's length-22 cyclic convolution splits, via
 * g^11 = -1 (g = 5 primitive root mod 23), into a CYCLIC-11 correlation with
 * the real kernel cos(2*pi*5^q/23) on the folded even part u_j = x_j+x_{23-j}
 * plus one with sin(2*pi*5^q/23) on w_j = -i(x_j - x_{23-j}).  Every
 * sub-quadratic realization of the length-11 convolution known to the corpus
 * exceeds the 121 fused FMAs of the direct circulant matvec on FMA hardware,
 * so the optimal Rader-23 IS the conjugate-folded direct form.  Per line:
 * 253 FMA + 44 add/sub = 297 vector FP ops; per volume 3*529 lines ->
 * 943 kflop (yardstick 5*N*log2(N) = 824 kflop).  The -i rotation's sign
 * lives in the sine tables (alternating {+s,-s} lanes -- phase-1 r10, from
 * L13_direct r9), so there is no vpxor anywhere.
 *
 * WHAT PHASE 1 SETTLED, KEPT VERBATIM:
 *   krn_ts    two-sweep pinned-constant kernel (11 cosines then 11 sines in
 *             registers), SIMD lanes = WC adjacent independent lines.
 *   X-first   pass order everywhere: X (in -> t1, plane stride padded to
 *             1064 doubles = 133 whole lines), then per plane x: Y (into a
 *             23x24-padded plane buffer), Z (transposing store into out).
 *   pf=2      in-pass X prefetch 4 chunks ahead (the 8464-B plane stride
 *             defeats page-local hardware prefetch); pw=1 write-intent
 *             prefetch of the out plane, half before Y, half between Y and Z
 *             ("hide the RFO, don't avoid it" -- VERDICT r5 4.5).
 *   NT        staging-plane + streaming-store variant, kept raced: it lost
 *             single-threaded, but with 32 cores sharing DRAM the RFO third
 *             of the out traffic is a bigger fraction of a scarcer resource.
 *   Retired for good (documented streaming nulls, do not rebuild): rp-t1,
 *   deferred-Z, uniform + tail-paced pipelining, pf=1, krn_il, P-parking,
 *   X-last.  See phase-1 record, VERDICT r10 7.
 *
 * WHAT IS NEW IN mt_r1 (threading layer only; per-volume DAG identical, so
 * output is bit-identical to the serial exec at every team size):
 *   - range execs: (plan, in, out, b0, b1, slot) with per-thread page-sized
 *     scratch slots (t1 + pb + ps), allocated once and FIRST-TOUCHED BY THE
 *     OWNING THREAD in a full-width parallel region in fft3d_create() --
 *     NUMA-local scratch, and the region also spins up the OpenMP pool so
 *     the first timed execute pays no thread creation.  (Layer design
 *     adopted from L13_direct mt_r1, the only other record this round.)
 *   - mode=batch: team of min(batch,T) threads, thread t owns the contiguous
 *     volume block [nb*t/T, nb*(t+1)/T); no synchronisation but the join.
 *     Ranges are computed from the team OpenMP actually delivers.
 *   - mode=fused (batch <= nslots): X phase = omp-for over all batch*133
 *     chunks (chunk i writes whole distinct 64-B lines of volume v's t1 --
 *     no false sharing by construction; the overlapping tail chunk rewrites
 *     bit-identical values, benign also cross-thread), implicit barrier,
 *     plane phase = omp-for over all batch*23 (v,x) planes, each Y->Z with
 *     the THREAD's own pb.  At B=1 this is the intra-volume split: L=23 is
 *     ~21 us of work on wallaby against a ~2-4 us fork+2-barrier cost, the
 *     opposite balance from L13_direct's measured B=1 null (2.5 us of work).
 *   - mode=serial kept and raced (it is the phase-1 node-measured pick and
 *     the honest B=1 fallback if the fork cost eats the split).
 *   - deterministic joint-cell hysteretic tuner (phase-1 r9 mechanism,
 *     cells extended to (mode, width, team, pf, pw)): one canonical list
 *     per regime, two fixed-order sweeps, per-cell min, challenger must
 *     beat the incumbent by >2%.  Heads are the fastest KNOWN cells
 *     (fastest-known-head rule, phase-1 r11): serial for B=1, one-volume-
 *     per-thread for 1<B<32, full-team batch pf2 pw1 for B>=32.  team=16
 *     cells are the two-socket NUMA question (in/out are first-touched by
 *     the single-threaded driver, so 16 far threads pay UPI on the node;
 *     the plan-time race answers it on the machine that matters).
 *
 * ASSUMPTIONS: L == 23 only; in/out distinct, 64-byte aligned (driver);
 * gcc/clang vector extensions with -ffp-contract=fast; self-#include
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

/* -i*t on interleaved complex: SWAPRI only -- the im-lane negation lives in
 * the sine tables' alternating {+s,-s} lane signs (phase-1 r10, from
 * L13_direct r9).  s*(-x) == (-s)*x bitwise, so no bits change. */
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
#else
#  define CDIST(p) ((p)->cd4)
#  define SDIST(p) ((p)->sd4)
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
 *  alternating {+s, -s} lane signs (the -i rotation, folded).  Every
 *  coefficient is a register constant.  All call sites pass compile-time-
 *  constant strides (keeps gcc from the runtime-offset lea-spill
 *  pathology L45_pfa r8 documented).
 * --------------------------------------------------------------------- */
static inline __attribute__((always_inline)) void
SUF(krn_ts)(const double *restrict src, long rs, double *restrict dst, long da,
            long db, const double *restrict cd, const double *restrict sd,
            int tr)
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
    L23R_STORE_TAIL();
}

/* chunk start offsets covering a 23-long index; the last chunk overlaps the
 * previous and rewrites bit-identical values (cheaper than masking --
 * phase-1 measured lesson).  Benign cross-thread too: both writers store
 * the exact same bytes. */
#if WC == 4
static const int SUF(off23)[6] = {0, 4, 8, 12, 16, 19};
#  define NOFF23 6
#else
static const int SUF(off23)[12] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21};
#  define NOFF23 12
#endif

#define PBROW (24 * 2) /* plane buffer row stride, doubles (23 padded to 24) */

/* X-pass chunk slots: slot i covers lanes i*WC..; the last is the
 * overlapping tail (529 % WC != 0 for both widths) */
#define NX23 (529 / WC + 1)
#define L23R_XF0(i) ((i) < 529 / WC ? (long)(i) * WC : (long)(529 - WC))

/* one X chunk: 23-point DFT along x, lanes = WC adjacent (y,z) pairs,
 * in (plane stride 1058) -> t1 (plane stride 1064 = 133 whole lines, so
 * every store is 64-byte aligned) */
static inline __attribute__((always_inline)) void
SUF(xchunk)(const double *restrict vin, double *restrict t1, long f0,
            const double *restrict cd, const double *restrict sd)
{
    SUF(krn_ts)(vin + 2 * f0, 1058, t1 + 2 * f0, 2, L23R_T1P, cd, sd, 0);
}

/* one (v,x) plane: Y (t1 rows, stride 46 -> pb, transposed) then Z (pb ->
 * out plane, transposed).  pw!=0 write-intent-prefetches the 133 out lines,
 * half before Y, half between Y and Z (phase-1 r7 pacing). */
static inline __attribute__((always_inline)) void
SUF(plane)(const double *restrict t1p, double *restrict pt,
           double *restrict pb, const double *restrict cd,
           const double *restrict sd, int pw)
{
    if (pw) L23R_PWB(pt, 0, 67);
    for (int t = 0; t < NOFF23; ++t) {
        long f0 = SUF(off23)[t];
        SUF(krn_ts)(t1p + 2 * f0, 46, pb + f0 * PBROW, PBROW, 2, cd, sd, 1);
    }
    if (pw) L23R_PWB(pt, 67, 133);
    for (int t = 0; t < NOFF23; ++t) {
        long f0 = SUF(off23)[t];
        SUF(krn_ts)(pb + 2 * f0, PBROW, pt + f0 * 46, 46, 2, cd, sd, 1);
    }
}

/* -----------------------------------------------------------------------
 *  Range exec: volumes [b0,b1) on ONE thread's scratch slot.  Plain
 *  X-first two-sweep, pf/pw as in phase 1; ntc!=0 stages each finished
 *  plane and streams it to out (NT stores skip the read-for-ownership).
 * --------------------------------------------------------------------- */
static void SUF(rng)(const fft3d_plan *restrict p,
                     const double _Complex *restrict in,
                     double _Complex *restrict out, int b0, int b1,
                     double *restrict slot, int ntc)
{
    const double *restrict cd = CDIST(p);
    const double *restrict sd = SDIST(p);
    double *restrict t1 = slot + L23R_SL_T1;
    double *restrict pb = slot + L23R_SL_PB;
    double *restrict ps = slot + L23R_SL_PS;
    const int pf = p->pf, pw = p->pw;
    for (int b = b0; b < b1; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 12167);
        double *vout = (double *)(out + (size_t)b * 12167);
        for (int i = 0; i < NX23; ++i) {
            if (pf == 2 && i + 4 < NX23) {
                /* pull the chunk 4 slots ahead: 23 lines, one per 8464-B-
                 * strided x-plane, where hardware prefetch restarts at
                 * every 4K boundary */
                const char *px = (const char *)(vin + 2 * L23R_XF0(i + 4));
                for (int q4 = 0; q4 < 23; ++q4)
                    __builtin_prefetch(px + (long)q4 * 8464, 0, 3);
            }
            SUF(xchunk)(vin, t1, L23R_XF0(i), cd, sd);
        }
        for (int x = 0; x < 23; ++x) {
            double *pt = vout + (long)x * 1058;
            if (!ntc) {
                SUF(plane)(t1 + (long)x * L23R_T1P, pt, pb, cd, sd, pw);
            } else {
                SUF(plane)(t1 + (long)x * L23R_T1P, ps, pb, cd, sd, 0);
                l23r_ntcopy(pt, ps, 1058);
            }
        }
    }
}

/* ---- top-level exec variants (dispatch targets) ------------------------ */

static __attribute__((unused)) void
SUF(x_serial)(const fft3d_plan *restrict p, const double _Complex *restrict in,
              double _Complex *restrict out)
{
    SUF(rng)(p, in, out, 0, p->batch, L23R_SLOT(p, 0), 0);
}

/* batch-parallel: thread t owns the contiguous block [nb*t/T, nb*(t+1)/T)
 * on its own NUMA-local slot; ranges from the team actually delivered */
static __attribute__((unused)) void
SUF(x_batch)(const fft3d_plan *restrict p, const double _Complex *restrict in,
             double _Complex *restrict out)
{
    const int nb = p->batch;
#pragma omp parallel num_threads(p->team)
    {
        int nt = omp_get_num_threads(), tid = omp_get_thread_num();
        int b0 = (int)((long)nb * tid / nt);
        int b1 = (int)((long)nb * (tid + 1) / nt);
        if (b1 > b0) SUF(rng)(p, in, out, b0, b1, L23R_SLOT(p, tid), 0);
    }
}

static __attribute__((unused)) void
SUF(x_batch_nt)(const fft3d_plan *restrict p,
                const double _Complex *restrict in,
                double _Complex *restrict out)
{
    const int nb = p->batch;
#pragma omp parallel num_threads(p->team)
    {
        int nt = omp_get_num_threads(), tid = omp_get_thread_num();
        int b0 = (int)((long)nb * tid / nt);
        int b1 = (int)((long)nb * (tid + 1) / nt);
        if (b1 > b0) SUF(rng)(p, in, out, b0, b1, L23R_SLOT(p, tid), 1);
    }
}

/* fused small-batch / intra-volume exec (requires batch <= nslot):
 * phase 1 = all batch*NX23 X chunks (volume v's chunks write slot v's t1;
 * each chunk owns whole distinct 64-B lines), implicit barrier, phase 2 =
 * all batch*23 (v,x) planes, each on the executing THREAD's pb.  Any
 * schedule gives identical bits (disjoint-or-identical element ownership,
 * fixed per-element arithmetic); schedule(static) for minimal overhead. */
static __attribute__((unused)) void
SUF(x_fused)(const fft3d_plan *restrict p, const double _Complex *restrict in,
             double _Complex *restrict out)
{
    const int nb = p->batch;
    const double *restrict cd = CDIST(p);
    const double *restrict sd = SDIST(p);
#pragma omp parallel num_threads(p->team)
    {
        double *restrict pb = L23R_SLOT(p, omp_get_thread_num()) + L23R_SL_PB;
#pragma omp for schedule(static)
        for (int q = 0; q < nb * NX23; ++q) {
            int v = q / NX23, i = q - v * NX23;
            const double *vin = (const double *)(in + (size_t)v * 12167);
            SUF(xchunk)(vin, L23R_SLOT(p, v) + L23R_SL_T1, L23R_XF0(i), cd, sd);
        }
        /* nowait: the region join right below is the only sync needed */
#pragma omp for schedule(static) nowait
        for (int q = 0; q < nb * 23; ++q) {
            int v = q / 23, x = q - v * 23;
            double *vout = (double *)(out + (size_t)v * 12167);
            SUF(plane)(L23R_SLOT(p, v) + L23R_SL_T1 + (long)x * L23R_T1P,
                       vout + (long)x * 1058, pb, cd, sd, 0);
        }
    }
}

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
#undef L23R_STORE_TAIL
#undef TTILE
#undef TILE
#undef NOFF23
#undef PBROW
#undef NX23
#undef L23R_XF0

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

#ifdef _OPENMP
#  include <omp.h>
#else
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_num_threads(void) { return 1; }
static inline int omp_get_thread_num(void) { return 0; }
#endif

/* Streaming copy of one finished 23x23 plane (1058 doubles) into `out`:
 * NT stores skip the read-for-ownership of every output line.  Lost
 * single-threaded (phase 1), raced again here: with 32 cores sharing DRAM
 * the RFO third of the out traffic is a bigger slice of a scarcer
 * resource.  Plane starts are 16 (mod 64), hence the step-up loop. */
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

/* paced write-intent prefetch of [l0,l1) of the 133 out-plane lines */
#define L23R_PWB(base, l0, l1)                                                 \
    do {                                                                       \
        const char *pwp = (const char *)(base);                                \
        for (int q5 = (l0); q5 < (l1); ++q5)                                   \
            __builtin_prefetch(pwp + (long)q5 * 64, 1, 3);                     \
    } while (0)

struct fft3d_plan {
    int L, batch;
    int team;   /* threads the exec's parallel region requests (<= harness's) */
    int pf;     /* in-pass X input prefetch: 0 off, 2 = 4 chunks ahead */
    int pw;     /* write-intent prefetch on the Z-group out stores */
    void (*exec)(const struct fft3d_plan *, const double _Complex *, double _Complex *);
    double *cd8, *sd8; /* 11 distinct cos / sin magnitudes, splatted 8x */
    double *cd4, *sd4; /* the same, splatted 4x */
    double *slab;      /* nslot per-thread scratch slots, page-aligned */
    int nslot;
    void *block;
    double _Complex *ti, *to; /* transient buffers for the plan-time tuner */
    size_t tn;
};

/* t1 plane stride, doubles: 23x23 complex = 1058, padded to 1064 so every
 * plane base (and so every X-pass store) is 64-byte aligned (8512 = 133
 * whole lines). */
#define L23R_T1P 1064

/* per-thread scratch slot layout, in doubles.  Slots are PAGE-sized so a
 * slot's pages are NUMA-local to its first-touching thread and no two
 * threads' scratch shares a line. */
#define L23R_SL_T1 0                       /* 23 * 1064 = 24472 */
#define L23R_SL_PB 24472                   /* 23 * 48   =  1104 */
#define L23R_SL_PS 25576                   /* staging plane 1064 */
#define L23R_SLOTSZ 27136                  /* 53 pages = 217088 B */

#define L23R_SLOT(p, t) ((p)->slab + (size_t)(t) * L23R_SLOTSZ)

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
    "Rader p=23 folded pair, two-sweep X-first, MT batch/fused decomposition";

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

/* Deterministic pseudo-random tuning data (realistic magnitudes suffice).
 * Filled single-threaded ON PURPOSE: the driver first-touches in/out
 * single-threaded, so the tuner's arena must live on one socket too. */
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

/* Streaming-regime tuner arena: in + out together exceed ~2.5x L3
 * (machine-relative sizing, phase-1 lineage). */
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

/* tuner cell: decomposition mode x width x team x prefetch knobs */
typedef struct {
    signed char mode; /* 0 serial, 1 batch, 2 batch-NT, 3 fused */
    signed char w;    /* vector width in complex: 4 or 2 */
    signed char team;
    signed char pf, pw;
} l23r_cell;

static const char *const l23r_mode_name[4] = {"serial", "batch", "batchNT",
                                              "fused"};

typedef void (*l23r_fn)(const fft3d_plan *, const double _Complex *,
                        double _Complex *);

static l23r_fn l23r_resolve(int mode, int w)
{
    switch (mode) {
    case 0: return w == 4 ? x_serial_w4 : x_serial_w2;
    case 1: return w == 4 ? x_batch_w4 : x_batch_w2;
    case 2: return w == 4 ? x_batch_nt_w4 : x_batch_nt_w2;
    default: return w == 4 ? x_fused_w4 : x_fused_w2;
    }
}

static void l23r_apply(fft3d_plan *p, const l23r_cell *c)
{
    p->exec = l23r_resolve(c->mode, c->w);
    p->team = c->team;
    p->pf = c->pf;
    p->pw = c->pw;
}

/* Finish: env overrides (L23R_PF / L23R_PW / L23R_TEAM, for same-window
 * A/Bs), bake pick + tuner telemetry into the description (leaderboard
 * carries one line of the node's own tuner table -- phase-1 pattern). */
static char g_dbuf[240];
static double g_tpick, g_tinc; /* tuned us/transform: picked cell, list head */
static int g_tnv;              /* tuner arena volumes (0 = tuner never ran) */
static void l23r_tune_finish(fft3d_plan *p, const l23r_cell *c)
{
    const char *e = getenv("L23R_PF");
    if (e && *e) {
        int v = atoi(e);
        p->pf = v <= 0 ? 0 : 2;
    }
    e = getenv("L23R_PW");
    if (e && *e) p->pw = atoi(e) ? 1 : 0;
    e = getenv("L23R_TEAM");
    if (e && *e) {
        int v = atoi(e);
        if (v >= 1 && v <= p->nslot) p->team = v;
    }
    if (g_tnv > 0)
        snprintf(g_dbuf, sizeof g_dbuf,
                 "rader23 MT %s w%d team=%d pf=%d pw=%d, tuner pick=%.2f inc=%.2f us/t nv=%d",
                 l23r_mode_name[(int)c->mode], (int)c->w, p->team, p->pf,
                 p->pw, g_tpick, g_tinc, g_tnv);
    else
        snprintf(g_dbuf, sizeof g_dbuf,
                 "rader23 MT %s w%d team=%d pf=%d pw=%d",
                 l23r_mode_name[(int)c->mode], (int)c->w, p->team, p->pf, p->pw);
    g_desc = g_dbuf;
    l23r_tune_free(p);
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != 23 || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;

    /* never more threads than the harness gave us */
    int T = omp_get_max_threads();
    if (T > 32) T = 32;
    if (T < 1) T = 1;
    p->nslot = T;

    /* one page of tables + T page-aligned scratch slots */
    size_t bytes = 4096 + (size_t)T * L23R_SLOTSZ * sizeof(double);
    void *blk = NULL;
    if (posix_memalign(&blk, 4096, bytes) != 0 || !blk) {
        free(p);
        return NULL;
    }
    p->block = blk;
    memset(blk, 0, 4096); /* tables page: touched by the planning thread */
    p->cd8 = (double *)blk;
    p->sd8 = p->cd8 + 88;
    p->cd4 = p->cd8 + 176;
    p->sd4 = p->cd8 + 220; /* 264 doubles total, well inside the page */
    p->slab = (double *)((char *)blk + 4096);

    /* first-touch each slot on its OWNING thread (NUMA-local scratch) and
     * spin up the OpenMP pool -- both are setup, excluded from the time */
#pragma omp parallel num_threads(T)
    {
        int tid = omp_get_thread_num();
        if (tid < T)
            memset(L23R_SLOT(p, tid), 0, L23R_SLOTSZ * sizeof(double));
    }

    /* long double trig so the double table is correctly rounded.  SINE
     * entries carry the -i rotation's im-lane sign: alternating {+s,-s}. */
    const long double twopi = 6.283185307179586476925286766559005768394L;
    for (int m = 1; m <= 11; ++m) {
        double c = (double)cosl(twopi * (long double)m / 23.0L);
        double s = (double)sinl(twopi * (long double)m / 23.0L);
        for (int t = 0; t < 8; ++t) p->cd8[(m - 1) * 8 + t] = c;
        for (int t = 0; t < 8; ++t) p->sd8[(m - 1) * 8 + t] = (t & 1) ? -s : s;
        for (int t = 0; t < 4; ++t) p->cd4[(m - 1) * 4 + t] = c;
        for (int t = 0; t < 4; ++t) p->sd4[(m - 1) * 4 + t] = (t & 1) ? -s : s;
    }

    /* ---- canonical cell list per regime.  Order is a policy statement;
     * heads follow the fastest-known-head rule (phase-1 r11): the head is
     * the fastest cell ANY scoring measurement supports, and a challenger
     * must beat the running incumbent by >2% (two fixed-order sweeps,
     * per-cell min) so near-ties resolve identically in every process.
     * All cells are one bit class: element ownership is disjoint (or
     * overlapping with identical values) and per-element arithmetic is
     * fixed, so the tuner's pick can never change the output. ---- */
    l23r_cell cells[24];
    int ncell = 0;
#define ADDC(m_, w_, t_, pf_, pw_)                                             \
    do {                                                                       \
        if (ncell < 24) {                                                      \
            cells[ncell].mode = (m_); cells[ncell].w = (w_);                   \
            cells[ncell].team = (signed char)(t_);                             \
            cells[ncell].pf = (pf_); cells[ncell].pw = (pw_); ++ncell;         \
        }                                                                      \
    } while (0)

    const int TB = batch < T ? batch : T;
    if (batch >= 32) {
        /* streaming regime: volumes across the full team; the phase-1
         * node-picked knobs (pf=2, pw=1) head the list */
        ADDC(1, 4, T, 2, 1);
        ADDC(1, 4, T, 0, 1);
        ADDC(1, 4, T, 2, 0);
        ADDC(1, 4, T, 0, 0);
        ADDC(2, 4, T, 2, 0); /* NT: the 32-core RFO question */
        ADDC(2, 4, T, 0, 0);
        if (T > 16) { /* the two-socket question: 16 = one CLX socket */
            ADDC(1, 4, 16, 2, 1);
            ADDC(1, 4, 16, 0, 0);
            ADDC(2, 4, 16, 2, 0);
        }
        if (T > 24) ADDC(1, 4, 24, 2, 1);
        ADDC(1, 2, T, 0, 0);  /* non-AVX-512 fallback */
        ADDC(0, 4, 1, 2, 1);  /* serial baseline: efficiency telemetry */
    } else if (batch == 1) {
        /* B=1: serial is the node-measured phase-1 pick, so it is the
         * head; the fused intra-volume split must earn its keep */
        ADDC(0, 4, 1, 0, 0);
        if (T >= 2) {
            ADDC(3, 4, T, 0, 0);
            if (T > 23) ADDC(3, 4, 23, 0, 0); /* 23 plane tasks */
            if (T > 16) ADDC(3, 4, 16, 0, 0);
            if (T > 8) ADDC(3, 4, 8, 0, 0);
            if (T > 4) ADDC(3, 4, 4, 0, 0);
        }
        ADDC(0, 2, 1, 0, 0);
    } else {
        /* 2..31 volumes: one volume per thread heads (L13_direct's
         * measured mid-batch lesson); fused uses the idle threads */
        ADDC(1, 4, TB, 0, 0);
        ADDC(1, 4, TB, 2, 1);
        if (batch <= p->nslot && T >= 2) {
            ADDC(3, 4, T, 0, 0);
            if (T > 16) ADDC(3, 4, 16, 0, 0);
            if (T > 8) ADDC(3, 4, 8, 0, 0);
        }
        ADDC(0, 4, 1, 0, 0);
        ADDC(1, 2, TB, 0, 0);
    }
#undef ADDC

    int bestc = 0;
    l23r_apply(p, &cells[0]);

    const char *force = getenv("L23R_FORCE");
    if (force && *force) {
        int v = atoi(force);
        if (v >= 0 && v < ncell) {
            l23r_apply(p, &cells[v]);
            if (l23r_verbose())
                fprintf(stderr, "[L23_rader] FORCED cell %d: %s w%d team=%d pf=%d pw=%d\n",
                        v, l23r_mode_name[(int)cells[v].mode], cells[v].w,
                        cells[v].team, cells[v].pf, cells[v].pw);
            l23r_tune_finish(p, &cells[v]);
            return p;
        }
    }

    {
        double best[24];
        for (int c = 0; c < ncell; ++c) best[c] = 1e30;
        /* small batches tune at the TRUE batch (the decomposition is a
         * function of it); streaming batches on an L3-exceeding arena */
        int nv = batch < 32 ? batch : l23r_tune_nv(batch);
        if (l23r_tune_alloc(p, nv)) {
            int sb = p->batch;
            p->batch = nv;
            int inner = batch < 32 ? (32 + nv - 1) / nv : 1;
            int nrep = batch < 32 ? 5 : 3;
            /* TWO full fixed-order sweeps, per-cell min across both; each
             * cell gets a licence-honest warmup >= 1.5 ms (phase-1 rules:
             * one sweep on a drifting clock mis-ranks distant cells, and
             * 512/256-bit licence transitions need the dwell) */
            for (int sweep = 0; sweep < 2; ++sweep)
                for (int c = 0; c < ncell; ++c) {
                    l23r_apply(p, &cells[c]);
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
                            "[L23_rader tune nv=%d] %-8s w%d team=%2d pf=%d pw=%d %9.2f us/transform%s\n",
                            nv, l23r_mode_name[(int)cells[c].mode], cells[c].w,
                            cells[c].team, cells[c].pf, cells[c].pw,
                            best[c] * 1e6 / ((double)nv * inner),
                            c == bestc ? "  <== kept" : "");
        }
        l23r_apply(p, &cells[bestc]);
    }

    l23r_tune_finish(p, &cells[bestc]);
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
