/* =============================================================================
 * L23_matrixsimd -- 23^3 complex-double forward DFT as three dense 23x23
 *                   matrix passes, conjugate-pair folded, vectorised across
 *                   whole lines -- MULTICORE (round mt_r1, 32 threads).
 *
 * SINGLE-CORE BASIS (phase 1, see ../../geom/strategies/L23_matrixsimd.md):
 *   arithmetic and layout frozen from panel_r11's winner -- the PINNED-
 *   CONSTANT kernel (the whole 23x23 DFT matrix folds to 11 distinct
 *   cosines + 11 distinct sines held in registers), X-first pass order,
 *   flat 1064-double padded t1 planes.  Node numbers: B=1 47.7 us (1.13x
 *   the one-FMA-port floor, arithmetically closed), B=128 64.9 us/vol.
 *   Everything phase 1 rejected on the node at ONE core (za, deferred-Z,
 *   pipelining, pf=1) is deleted, not re-raced.
 *
 * WHAT IS NEW FOR 32 CORES:
 *   1. A PERSISTENT SPIN POOL, not per-call OpenMP regions.  Measured on
 *      wallaby: one GOMP fork + barrier + join costs 6.2-8.2 us at T=8..32
 *      -- more than the entire 32-way-parallel compute of one volume
 *      (~1 us).  fft3d_create() spawns the pool once (PANEL_BRIEF: thread
 *      creation is setup), pins each worker to the CPU the harness's
 *      OMP_PROC_BIND=close / OMP_PLACES=cores mapping would give it (the
 *      mapping is read back from one throwaway OpenMP region via
 *      sched_getcpu(), so whatever GOMP would do, the pool does), and
 *      execute() only flips an atomic generation counter.  Phase barriers
 *      are generation-counting spin barriers, ~1 us at 32 threads.
 *      OpenMP itself is never entered again after create().
 *   2. Two decompositions, chosen structurally by batch:
 *      * batch >= 32, VOLUME-PARALLEL: thread t owns the contiguous volume
 *        range [nb*t/T, nb*(t+1)/T) and runs the settled single-core
 *        schedule on its own NUMA-local scratch (allocated and first-
 *        touched by the owning worker at spawn, after pinning).  One
 *        barrier per execute (the join).
 *      * batch < 32, FUSED: the X pass is 132 independent chunk items per
 *        volume (last two overlapping slots fused into one item so the
 *        identical-value overlap lanes stay in one thread), then ONE
 *        barrier (every t1 plane needs every X chunk of its volume), then
 *        the plane phase is 23 independent planes per volume (Y into the
 *        thread's own plane buffer, Z into the caller's out plane).  Item
 *        loops run over ALL volumes of the batch at once: 2 barriers per
 *        execute total, not per volume.  Active thread count tw is a tuned
 *        knob (barrier cost vs width; a serial row is the floor, so B=1
 *        can never lose to phase 1's 47.7 us).
 *   3. NT stores (phase-1 loser at one core, four rounds) are RE-RACED at
 *      batch >= 32 only: at one core the cell was latency-bound; at 32
 *      threads the batch cells sit at the socket bandwidth wall and NT
 *      deletes the out-RFO third of the traffic.  The driver fread()s `in`
 *      and memset()s `out` on thread 0, so on the two-socket node ALL
 *      caller pages live on socket 0; per-thread scratch locality and the
 *      RFO savings are the NUMA levers this file actually owns.
 *   4. pf=2 / pw prefetch knobs kept for the volume path (single-core
 *      streaming winners; at 32 threads possibly pure tax -- the create()-
 *      time walk decides on the node).
 *
 * BIT CLASS: ONE class.  Every candidate (either decomposition, any thread
 *   count, either SIMD width, NT on/off, any prefetch knob) computes every
 *   output element in the SAME chunk with the SAME per-value accumulation
 *   order -- parallelisation only reassigns whole chunks to threads.  The
 *   only doubly-written lanes are the overlap lanes of tail chunks, which
 *   are written with bit-identical values and are kept inside one item.
 *   cmp-verified across forced modes on wallaby, never assumed.
 *
 * TUNER: phase-1 discipline -- canonical-order walk, two full sweeps,
 *   per-candidate min, hysteretic displacement (>3% fused / >4% streaming),
 *   deterministic serially-filled arena (matches the driver's socket-0
 *   first touch of the real buffers), per-candidate licence warmup,
 *   telemetry in the description string.
 *
 * OPERATION COUNT (unchanged): 594 real flop/line, 1587 lines, 943 kflop
 *   per volume; 297 vector FP ops/chunk, 409 zmm chunks/volume.
 * =============================================================================
 */
#ifdef L23_TEMPLATE_PASS
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

/* MULI(t) = -i*t for interleaved complex: swap re/im, flip the sign of the
 * imaginary lanes.  XOR instead of a multiply so it can issue on port 5
 * rather than the FMA port (phase-1 measured choice). */
#define SIGN64 ((long long)0x8000000000000000LL)
#if WC == 4
#  define SWAPRI(x) SHUF1(x, 1, 0, 3, 2, 5, 4, 7, 6)
#  define NEGMASK ((IT){0, SIGN64, 0, SIGN64, 0, SIGN64, 0, SIGN64})
#else
#  define SWAPRI(x) SHUF1(x, 1, 0, 3, 2)
#  define NEGMASK ((IT){0, SIGN64, 0, SIGN64})
#endif
#define MULI(t) ((VT)((IT)SWAPRI(t) ^ NEGMASK))

/* Coefficients from the PRE-SPLATTED table: each stored VDW times, read as a
 * plain full-width memory operand (phase-1 lesson: scalar table + embedded
 * broadcast makes gcc materialise every splat into a stack slot). */
#define CGET(base, i) VLD((base) + (size_t)(i) * VDW)
#if WC == 4
#  define CDTAB(p) ((p)->cdist8)
#  define SDTAB(p) ((p)->sdist8)
#else
#  define CDTAB(p) ((p)->cdist4)
#  define SDTAB(p) ((p)->sdist4)
#endif

/* ---- WCxWC transpose of complex (i.e. of 128-bit blocks) ---------------- */
#if WC == 4
#  define TTILE(x, y)                                                          \
      do {                                                                     \
          VT q0 = SHUF2((x)[0], (x)[1], 0, 1, 2, 3, 8, 9, 10, 11);              \
          VT q1 = SHUF2((x)[2], (x)[3], 0, 1, 2, 3, 8, 9, 10, 11);              \
          VT q2 = SHUF2((x)[0], (x)[1], 4, 5, 6, 7, 12, 13, 14, 15);            \
          VT q3 = SHUF2((x)[2], (x)[3], 4, 5, 6, 7, 12, 13, 14, 15);            \
          (y)[0] = SHUF2(q0, q1, 0, 1, 4, 5, 8, 9, 12, 13);                     \
          (y)[1] = SHUF2(q0, q1, 2, 3, 6, 7, 10, 11, 14, 15);                   \
          (y)[2] = SHUF2(q2, q3, 0, 1, 4, 5, 8, 9, 12, 13);                     \
          (y)[3] = SHUF2(q2, q3, 2, 3, 6, 7, 10, 11, 14, 15);                   \
      } while (0)
#else
#  define TTILE(x, y)                                                          \
      do {                                                                     \
          (y)[0] = SHUF2((x)[0], (x)[1], 0, 1, 4, 5);                           \
          (y)[1] = SHUF2((x)[0], (x)[1], 2, 3, 6, 7);                           \
      } while (0)
#endif

#if WC == 4
#  define TILE(m0, e0, e1, e2, e3)                                             \
      do {                                                                     \
          VT xx[4], yy[4];                                                     \
          xx[0] = (e0); xx[1] = (e1); xx[2] = (e2); xx[3] = (e3);               \
          TTILE(xx, yy);                                                       \
          VST(dst + 0 * da + (m0) * 2, yy[0]);                                 \
          VST(dst + 1 * da + (m0) * 2, yy[1]);                                 \
          VST(dst + 2 * da + (m0) * 2, yy[2]);                                 \
          VST(dst + 3 * da + (m0) * 2, yy[3]);                                 \
      } while (0)
#else
#  define TILE(m0, e0, e1)                                                     \
      do {                                                                     \
          VT xx[2], yy[2];                                                     \
          xx[0] = (e0); xx[1] = (e1);                                          \
          TTILE(xx, yy);                                                       \
          VST(dst + 0 * da + (m0) * 2, yy[0]);                                 \
          VST(dst + 1 * da + (m0) * 2, yy[1]);                                 \
      } while (0)
#endif

/* Shared store tail: combine P/R and store, plain or tile-transposed.
 * output index m: X_0 = P_0, X_k = P_k + R_k, X_{23-k} = P_k - R_k */
#if WC == 4
#define L23_STORE_TAIL()                                                       \
    do {                                                                       \
        if (!tr) {                                                             \
            VST(dst + 0 * db, P0);                                             \
            VST(dst + 1 * db, P1 + R1);   VST(dst + 2 * db, P2 + R2);          \
            VST(dst + 3 * db, P3 + R3);   VST(dst + 4 * db, P4 + R4);          \
            VST(dst + 5 * db, P5 + R5);   VST(dst + 6 * db, P6 + R6);          \
            VST(dst + 7 * db, P7 + R7);   VST(dst + 8 * db, P8 + R8);          \
            VST(dst + 9 * db, P9 + R9);   VST(dst + 10 * db, P10 + R10);       \
            VST(dst + 11 * db, P11 + R11); VST(dst + 12 * db, P11 - R11);      \
            VST(dst + 13 * db, P10 - R10); VST(dst + 14 * db, P9 - R9);        \
            VST(dst + 15 * db, P8 - R8);  VST(dst + 16 * db, P7 - R7);         \
            VST(dst + 17 * db, P6 - R6);  VST(dst + 18 * db, P5 - R5);         \
            VST(dst + 19 * db, P4 - R4);  VST(dst + 20 * db, P3 - R3);         \
            VST(dst + 21 * db, P2 - R2);  VST(dst + 22 * db, P1 - R1);         \
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
#define L23_STORE_TAIL()                                                       \
    do {                                                                       \
        if (!tr) {                                                             \
            VST(dst + 0 * db, P0);                                             \
            VST(dst + 1 * db, P1 + R1);   VST(dst + 2 * db, P2 + R2);          \
            VST(dst + 3 * db, P3 + R3);   VST(dst + 4 * db, P4 + R4);          \
            VST(dst + 5 * db, P5 + R5);   VST(dst + 6 * db, P6 + R6);          \
            VST(dst + 7 * db, P7 + R7);   VST(dst + 8 * db, P8 + R8);          \
            VST(dst + 9 * db, P9 + R9);   VST(dst + 10 * db, P10 + R10);       \
            VST(dst + 11 * db, P11 + R11); VST(dst + 12 * db, P11 - R11);      \
            VST(dst + 13 * db, P10 - R10); VST(dst + 14 * db, P9 - R9);        \
            VST(dst + 15 * db, P8 - R8);  VST(dst + 16 * db, P7 - R7);         \
            VST(dst + 17 * db, P6 - R6);  VST(dst + 18 * db, P5 - R5);         \
            VST(dst + 19 * db, P4 - R4);  VST(dst + 20 * db, P3 - R3);         \
            VST(dst + 21 * db, P2 - R2);  VST(dst + 22 * db, P1 - R1);         \
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
 *  Pinned-constant kernel (phase-1 winner): the 23x23 DFT matrix has only
 *  11 distinct cosine and 11 distinct sine values, because
 *  cos(2pi r/23) = cos(2pi (23-r)/23) and sin(2pi r/23) = -sin(...) (a
 *  compile-time vfnmadd), so the whole matrix runs from 22 register-
 *  resident constants.  Fully unrolled, j-major, same accumulation order
 *  at both widths.
 *    src, rs : element (j=0, lane 0); rs doubles between successive j
 *    dst, da, db : output element (m) of lane (f) at dst + f*da + m*db
 *    tr : 1 -> in-register WCxWC tile transpose into rows of dst
 * --------------------------------------------------------------------- */
static inline __attribute__((always_inline)) void
SUF(chunk23p)(const double *restrict src, long rs, double *restrict dst, long da,
              long db, const double *restrict cd, const double *restrict sd,
              int tr)
{
    VT P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11;
    VT R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11;

    { /* ---- P sweep (cosine side), 11 pinned constants, unrolled ---- */
        VT C1 = CGET(cd, 0), C2 = CGET(cd, 1), C3 = CGET(cd, 2),
           C4 = CGET(cd, 3), C5 = CGET(cd, 4), C6 = CGET(cd, 5),
           C7 = CGET(cd, 6), C8 = CGET(cd, 7), C9 = CGET(cd, 8),
           C10 = CGET(cd, 9), C11 = CGET(cd, 10);
        VT v0 = VLD(src);
        P0 = v0; P1 = v0; P2 = v0; P3 = v0; P4 = v0; P5 = v0;
        P6 = v0; P7 = v0; P8 = v0; P9 = v0; P10 = v0; P11 = v0;
        { VT a = VLD(src + 1 * rs), b = VLD(src + 22 * rs); VT u = a + b;
          P0 += u; P1 += C1 * u; P2 += C2 * u; P3 += C3 * u; P4 += C4 * u;
          P5 += C5 * u; P6 += C6 * u; P7 += C7 * u; P8 += C8 * u;
          P9 += C9 * u; P10 += C10 * u; P11 += C11 * u; }
        { VT a = VLD(src + 2 * rs), b = VLD(src + 21 * rs); VT u = a + b;
          P0 += u; P1 += C2 * u; P2 += C4 * u; P3 += C6 * u; P4 += C8 * u;
          P5 += C10 * u; P6 += C11 * u; P7 += C9 * u; P8 += C7 * u;
          P9 += C5 * u; P10 += C3 * u; P11 += C1 * u; }
        { VT a = VLD(src + 3 * rs), b = VLD(src + 20 * rs); VT u = a + b;
          P0 += u; P1 += C3 * u; P2 += C6 * u; P3 += C9 * u; P4 += C11 * u;
          P5 += C8 * u; P6 += C5 * u; P7 += C2 * u; P8 += C1 * u;
          P9 += C4 * u; P10 += C7 * u; P11 += C10 * u; }
        { VT a = VLD(src + 4 * rs), b = VLD(src + 19 * rs); VT u = a + b;
          P0 += u; P1 += C4 * u; P2 += C8 * u; P3 += C11 * u; P4 += C7 * u;
          P5 += C3 * u; P6 += C1 * u; P7 += C5 * u; P8 += C9 * u;
          P9 += C10 * u; P10 += C6 * u; P11 += C2 * u; }
        { VT a = VLD(src + 5 * rs), b = VLD(src + 18 * rs); VT u = a + b;
          P0 += u; P1 += C5 * u; P2 += C10 * u; P3 += C8 * u; P4 += C3 * u;
          P5 += C2 * u; P6 += C7 * u; P7 += C11 * u; P8 += C6 * u;
          P9 += C1 * u; P10 += C4 * u; P11 += C9 * u; }
        { VT a = VLD(src + 6 * rs), b = VLD(src + 17 * rs); VT u = a + b;
          P0 += u; P1 += C6 * u; P2 += C11 * u; P3 += C5 * u; P4 += C1 * u;
          P5 += C7 * u; P6 += C10 * u; P7 += C4 * u; P8 += C2 * u;
          P9 += C8 * u; P10 += C9 * u; P11 += C3 * u; }
        { VT a = VLD(src + 7 * rs), b = VLD(src + 16 * rs); VT u = a + b;
          P0 += u; P1 += C7 * u; P2 += C9 * u; P3 += C2 * u; P4 += C5 * u;
          P5 += C11 * u; P6 += C4 * u; P7 += C3 * u; P8 += C10 * u;
          P9 += C6 * u; P10 += C1 * u; P11 += C8 * u; }
        { VT a = VLD(src + 8 * rs), b = VLD(src + 15 * rs); VT u = a + b;
          P0 += u; P1 += C8 * u; P2 += C7 * u; P3 += C1 * u; P4 += C9 * u;
          P5 += C6 * u; P6 += C2 * u; P7 += C10 * u; P8 += C5 * u;
          P9 += C3 * u; P10 += C11 * u; P11 += C4 * u; }
        { VT a = VLD(src + 9 * rs), b = VLD(src + 14 * rs); VT u = a + b;
          P0 += u; P1 += C9 * u; P2 += C5 * u; P3 += C4 * u; P4 += C10 * u;
          P5 += C1 * u; P6 += C8 * u; P7 += C6 * u; P8 += C3 * u;
          P9 += C11 * u; P10 += C2 * u; P11 += C7 * u; }
        { VT a = VLD(src + 10 * rs), b = VLD(src + 13 * rs); VT u = a + b;
          P0 += u; P1 += C10 * u; P2 += C3 * u; P3 += C7 * u; P4 += C6 * u;
          P5 += C4 * u; P6 += C9 * u; P7 += C1 * u; P8 += C11 * u;
          P9 += C2 * u; P10 += C8 * u; P11 += C5 * u; }
        { VT a = VLD(src + 11 * rs), b = VLD(src + 12 * rs); VT u = a + b;
          P0 += u; P1 += C11 * u; P2 += C1 * u; P3 += C10 * u; P4 += C2 * u;
          P5 += C9 * u; P6 += C3 * u; P7 += C8 * u; P8 += C4 * u;
          P9 += C7 * u; P10 += C5 * u; P11 += C6 * u; }
    }

    { /* ---- R sweep (sine side): sign = vfnmadd when jk mod 23 > 11 ---- */
        VT S1 = CGET(sd, 0), S2 = CGET(sd, 1), S3 = CGET(sd, 2),
           S4 = CGET(sd, 3), S5 = CGET(sd, 4), S6 = CGET(sd, 5),
           S7 = CGET(sd, 6), S8 = CGET(sd, 7), S9 = CGET(sd, 8),
           S10 = CGET(sd, 9), S11 = CGET(sd, 10);
        { VT w = MULI(VLD(src + 1 * rs) - VLD(src + 22 * rs));
          R1 = S1 * w; R2 = S2 * w; R3 = S3 * w; R4 = S4 * w;
          R5 = S5 * w; R6 = S6 * w; R7 = S7 * w; R8 = S8 * w;
          R9 = S9 * w; R10 = S10 * w; R11 = S11 * w; }
        { VT w = MULI(VLD(src + 2 * rs) - VLD(src + 21 * rs));
          R1 += S2 * w; R2 += S4 * w; R3 += S6 * w; R4 += S8 * w;
          R5 += S10 * w; R6 -= S11 * w; R7 -= S9 * w; R8 -= S7 * w;
          R9 -= S5 * w; R10 -= S3 * w; R11 -= S1 * w; }
        { VT w = MULI(VLD(src + 3 * rs) - VLD(src + 20 * rs));
          R1 += S3 * w; R2 += S6 * w; R3 += S9 * w; R4 -= S11 * w;
          R5 -= S8 * w; R6 -= S5 * w; R7 -= S2 * w; R8 += S1 * w;
          R9 += S4 * w; R10 += S7 * w; R11 += S10 * w; }
        { VT w = MULI(VLD(src + 4 * rs) - VLD(src + 19 * rs));
          R1 += S4 * w; R2 += S8 * w; R3 -= S11 * w; R4 -= S7 * w;
          R5 -= S3 * w; R6 += S1 * w; R7 += S5 * w; R8 += S9 * w;
          R9 -= S10 * w; R10 -= S6 * w; R11 -= S2 * w; }
        { VT w = MULI(VLD(src + 5 * rs) - VLD(src + 18 * rs));
          R1 += S5 * w; R2 += S10 * w; R3 -= S8 * w; R4 -= S3 * w;
          R5 += S2 * w; R6 += S7 * w; R7 -= S11 * w; R8 -= S6 * w;
          R9 -= S1 * w; R10 += S4 * w; R11 += S9 * w; }
        { VT w = MULI(VLD(src + 6 * rs) - VLD(src + 17 * rs));
          R1 += S6 * w; R2 -= S11 * w; R3 -= S5 * w; R4 += S1 * w;
          R5 += S7 * w; R6 -= S10 * w; R7 -= S4 * w; R8 += S2 * w;
          R9 += S8 * w; R10 -= S9 * w; R11 -= S3 * w; }
        { VT w = MULI(VLD(src + 7 * rs) - VLD(src + 16 * rs));
          R1 += S7 * w; R2 -= S9 * w; R3 -= S2 * w; R4 += S5 * w;
          R5 -= S11 * w; R6 -= S4 * w; R7 += S3 * w; R8 += S10 * w;
          R9 -= S6 * w; R10 += S1 * w; R11 += S8 * w; }
        { VT w = MULI(VLD(src + 8 * rs) - VLD(src + 15 * rs));
          R1 += S8 * w; R2 -= S7 * w; R3 += S1 * w; R4 += S9 * w;
          R5 -= S6 * w; R6 += S2 * w; R7 += S10 * w; R8 -= S5 * w;
          R9 += S3 * w; R10 += S11 * w; R11 -= S4 * w; }
        { VT w = MULI(VLD(src + 9 * rs) - VLD(src + 14 * rs));
          R1 += S9 * w; R2 -= S5 * w; R3 += S4 * w; R4 -= S10 * w;
          R5 -= S1 * w; R6 += S8 * w; R7 -= S6 * w; R8 += S3 * w;
          R9 -= S11 * w; R10 -= S2 * w; R11 += S7 * w; }
        { VT w = MULI(VLD(src + 10 * rs) - VLD(src + 13 * rs));
          R1 += S10 * w; R2 -= S3 * w; R3 += S7 * w; R4 -= S6 * w;
          R5 += S4 * w; R6 -= S9 * w; R7 += S1 * w; R8 += S11 * w;
          R9 -= S2 * w; R10 += S8 * w; R11 -= S5 * w; }
        { VT w = MULI(VLD(src + 11 * rs) - VLD(src + 12 * rs));
          R1 += S11 * w; R2 -= S1 * w; R3 += S10 * w; R4 -= S2 * w;
          R5 += S9 * w; R6 -= S3 * w; R7 += S8 * w; R8 -= S4 * w;
          R9 += S7 * w; R10 -= S5 * w; R11 += S6 * w; }
    }
    L23_STORE_TAIL();
}

/* chunk start offsets covering a 23-long index; the last one overlaps the
 * previous by WC-1 lanes and recomputes/rewrites them with identical values */
#if WC == 4
static const int SUF(off23)[6] = {0, 4, 8, 12, 16, 19};
#  define NOFF23 6
#else
static const int SUF(off23)[12] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21};
#  define NOFF23 12
#endif

#define PBROW (24 * 2) /* plane buffer row stride, doubles (23 padded to 24) */

/* number of X-pass chunk slots (slot i covers lanes i*WC..; last slot is the
 * overlapping tail at 529-WC -- 529 % WC == 1 for both widths).  In the
 * fused parallel loops the last TWO slots form ONE work item, so the
 * identical-value overlap lanes are always written by a single thread. */
#define NX23 (529 / WC + 1)
#define NXI23 (NX23 - 1)
#define L23_XF0(i) ((i) < 529 / WC ? (long)(i) * WC : (long)(529 - WC))

/* One X-pass chunk slot: in[.][y][z], lanes over the flat 529-wide (y,z)
 * index -> t1[kx][y][z] (plain stores, padded planes).  pf2 = in-pass input
 * prefetch 4 slots ahead, one line per 8464-B-strided x-plane (single-core
 * streaming winner; runtime knob, changes no bits). */
static inline void SUF(xslot)(const double *restrict vin, double *restrict t1v,
                              int i, int pf2, const double *restrict cd,
                              const double *restrict sd)
{
    if (pf2 && i + 4 < NX23) {
        const char *px = (const char *)(vin + 2 * L23_XF0(i + 4));
        for (int q = 0; q < 23; ++q)
            __builtin_prefetch(px + (long)q * 8464, 0, 3);
    }
    long f0 = L23_XF0(i);
    SUF(chunk23p)(vin + 2 * f0, 1058, t1v + 2 * f0, 2, L23_T1P, cd, sd, 0);
}

/* One x-plane of the plane phase: Y pass (lanes over z, transposing store
 * into the thread's plane buffer), then Z pass (lanes over ky, transposing
 * store into the caller's out plane, optionally staged through ps and
 * streamed with NT stores).  pw = paced prefetchw of the out plane, split
 * in two half-bursts around the Y group (phase-1 schedule). */
static inline void SUF(plane)(const double *restrict t1x, double *restrict outp,
                              double *restrict pb, const double *restrict cd,
                              const double *restrict sd, int pw, int nt,
                              double *restrict ps, int pt)
{
    if (pt) {
        /* fused mode only: this t1 plane was just written by up to 32
         * OTHER cores' X chunks (transposing all-to-all), so its lines sit
         * dirty in remote L2s; a sequential prefetch burst starts those
         * transfers with full memory-level parallelism instead of letting
         * the Y pass's 368-B-strided loads discover them one miss window
         * at a time.  Changes no bits. */
        const char *tb = (const char *)t1x;
        for (int q = 0; q < 133; ++q)
            __builtin_prefetch(tb + (long)q * 64, 0, 3);
    }
    if (pw) {
        const char *ob = (const char *)outp;
        for (int q = 0; q < 66; ++q) __builtin_prefetch(ob + (long)q * 64, 1, 3);
    }
    for (int t = 0; t < NOFF23; ++t) {
        long f0 = SUF(off23)[t];
        SUF(chunk23p)(t1x + 2 * f0, 46, pb + f0 * PBROW, PBROW, 2, cd, sd, 1);
    }
    if (pw) {
        const char *ob = (const char *)outp;
        for (int q = 66; q < 133; ++q) __builtin_prefetch(ob + (long)q * 64, 1, 3);
    }
    double *restrict dstp = nt ? ps : outp;
    for (int t = 0; t < NOFF23; ++t) {
        long f0 = SUF(off23)[t];
        SUF(chunk23p)(pb + 2 * f0, PBROW, dstp + f0 * 46, 46, 2, cd, sd, 1);
    }
    if (nt) l23_ntcopy(outp, ps, 1058);
}

/* One whole volume: the settled single-core X-first schedule. */
static void SUF(vol)(const double *restrict vin, double *restrict vout,
                     double *restrict t1v, double *restrict pb,
                     double *restrict ps, const double *restrict cd,
                     const double *restrict sd, int pf2, int pw, int nt)
{
    for (int i = 0; i < NX23; ++i) SUF(xslot)(vin, t1v, i, pf2, cd, sd);
    for (int x = 0; x < 23; ++x)
        SUF(plane)(t1v + (size_t)x * L23_T1P,
                   vout + (size_t)x * 1058, pb, cd, sd, pw, nt, ps, 0);
}

/* -----------------------------------------------------------------------
 *  Pool jobs.  Convention: threads tid >= pl->tw return immediately and
 *  take part in NO barrier of the job, so every barrier has exactly tw
 *  participants (generation-counting barriers tolerate a varying set).
 * --------------------------------------------------------------------- */

/* VOLUME-PARALLEL job (batch >= 32): thread t owns volumes
 * [nb*t/T, nb*(t+1)/T) on its own NUMA-local scratch; one join barrier. */
static void SUF(work_vol)(const fft3d_plan *restrict p, int tid)
{
    l23_pool *restrict pl = p->pl;
    const int T = pl->tw;
    if (tid >= T) return;
    const double *restrict cd = CDTAB(p);
    const double *restrict sd = SDTAB(p);
    const int nb = p->batch;
    const int nt = p->nt, pf2 = (p->pf == 2), pw = p->nt ? 0 : p->pw;
    const double _Complex *in = pl->in;
    double _Complex *out = pl->out;
    const l23_ctx *c = &p->ctx[tid];
    long b0 = (long)nb * tid / T, b1 = (long)nb * (tid + 1) / T;
    for (long b = b0; b < b1; ++b)
        SUF(vol)((const double *)(in + (size_t)b * 12167),
                 (double *)(out + (size_t)b * 12167),
                 c->t1, c->pb, c->ps, cd, sd, pf2, pw, nt);
    l23_bar_join(pl, tid, T);
}

/* FUSED job (batch < 32): all volumes' X items, ONE barrier (every t1
 * plane needs every X chunk of its volume), all volumes' planes, join. */
static void SUF(work_fused)(const fft3d_plan *restrict p, int tid)
{
    l23_pool *restrict pl = p->pl;
    const int T = pl->tw;
    if (tid >= T) return;
    const double *restrict cd = CDTAB(p);
    const double *restrict sd = SDTAB(p);
    const int nb = p->batch;
    const long nxi = (long)nb * NXI23;
    const long npl = (long)nb * 23;
    const double _Complex *in = pl->in;
    double _Complex *out = pl->out;
    double *restrict t1f = p->t1f;
    {
        long g0 = nxi * tid / T, g1 = nxi * (tid + 1) / T;
        for (long g = g0; g < g1; ++g) {
            long b = g / NXI23;
            int i = (int)(g % NXI23);
            const double *vin = (const double *)(in + (size_t)b * 12167);
            double *t1v = t1f + (size_t)b * L23_T1V;
            SUF(xslot)(vin, t1v, i, 0, cd, sd);
            if (i == NXI23 - 1) SUF(xslot)(vin, t1v, NX23 - 1, 0, cd, sd);
        }
    }
    l23_bar_mid(pl, tid, T);
    {
        const int pt = p->pt;
        double *restrict pb = p->ctx[tid].pb;
        long g0 = npl * tid / T, g1 = npl * (tid + 1) / T;
        for (long g = g0; g < g1; ++g) {
            long b = g / 23;
            int x = (int)(g % 23);
            SUF(plane)(t1f + (size_t)b * L23_T1V + (size_t)x * L23_T1P,
                       (double *)(out + (size_t)b * 12167) + (size_t)x * 1058,
                       pb, cd, sd, 0, 0, NULL, pt);
        }
    }
    l23_bar_join(pl, tid, T);
}

#undef VDW
#undef VT
#undef IT
#undef VLD
#undef VST
#undef SHUF1
#undef SHUF2
#undef SWAPRI
#undef NEGMASK
#undef MULI
#undef SIGN64
#undef CGET
#undef CDTAB
#undef SDTAB
#undef L23_STORE_TAIL
#undef TTILE
#undef TILE
#undef NOFF23
#undef PBROW
#undef NX23
#undef NXI23
#undef L23_XF0

#else /* ================= main body ================= */

#define _GNU_SOURCE /* sched_getcpu, CPU_SET, pthread_setaffinity_np */

#include <complex.h>
#include <math.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../fft3d_api.h"

/* Streaming copy of one finished 23x23 plane (1058 doubles) into the
 * caller's buffer, skipping the read-for-ownership.  Phase-1 loser at ONE
 * core (latency-bound cell); re-raced at 32 threads where the batch cells
 * sit at the socket bandwidth wall.  Plane starts are 16 (mod 64)-byte
 * aligned, hence the step-up loop.  NT stores write the same bits. */
#if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE2__)
#  include <immintrin.h>
static void l23_ntcopy(double *restrict dst, const double *restrict src, size_t nd)
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
static void l23_ntcopy(double *restrict dst, const double *restrict src, size_t nd)
{
    memcpy(dst, src, nd * sizeof *dst);
}
#endif

/* t1 plane stride in doubles: 529 complex = 1058 doubles padded to 1064
 * (8512 B = 133 whole cache lines), so X-pass stores into t1 and plane-phase
 * loads of t1 never split a line.  L23_T1V = one padded volume. */
#define L23_T1P 1064
#define L23_T1V (23 * L23_T1P)

/* fused mode below this batch size; volume-parallel at or above (there
 * every thread owns >= 1 whole volume; below, the fused t1 arena of
 * batch * 191 KiB <= 5.9 MiB stays cache-scale). */
#define L23_FUSEB 32
#define L23_MAXT 32 /* never take more than the harness's 32 cores */

#if defined(__x86_64__) || defined(__i386__)
#  define l23_cpu_relax() __builtin_ia32_pause()
#else
#  define l23_cpu_relax() ((void)0)
#endif

typedef struct {
    double *pb; /* 24 x 24 complex plane buffer (pass Y -> pass Z) */
    double *t1; /* one padded 23^3 volume (volume-parallel mode)   */
    double *ps; /* one 23x23 plane: NT staging                     */
    void *blk;
} l23_ctx;

/* Per-thread arrival flag, one cache line each: a barrier arrival is one
 * uncontended store, not 32 serialized RMWs on a shared counter (the
 * central fetch-add barrier measured ~2.5 us per episode at 32 threads on
 * wallaby -- more than the barrier'd work).  Main (tid 0) is always the
 * collector: it scans the flags (the loads overlap in the fill buffers)
 * and broadcasts one release word.  Epochs are derived from the job
 * generation, so a varying participant set (the tw knob) stays correct:
 * flags and release only ever increase. */
typedef struct {
    _Atomic unsigned long v;
    char pad[56];
} l23_flag;

struct fft3d_plan;
typedef void (*l23_workfn)(const struct fft3d_plan *, int);

typedef struct {
    struct fft3d_plan *p;
    int tid;
    int cpu;
} l23_warg;

/* The persistent pool.  Workers spin on `gen`; main publishes a job (work
 * fn + buffers, plain stores) then release-stores gen+1 and runs the job
 * itself as tid 0.  Every job ends in a barrier over its tw participants,
 * so when main returns from the job fn the whole batch is done and the
 * workers are already back on the gen spin.  Workers never sleep: the
 * driver's timing loop is back-to-back executes, and the cores are ours. */
typedef struct l23_pool {
    _Atomic long gen;
    char pad0[56];
    _Atomic unsigned long rel;
    char pad1[56];
    l23_flag arr[L23_MAXT];
    _Atomic int ready;
    _Atomic int fail;
    _Atomic int shutdown;
    char pad2[52];
    l23_workfn workfn;
    const double _Complex *in;
    double _Complex *out;
    long gen_local;
    int nthr; /* pool size incl. main */
    int tw;   /* active participants of the current job (<= nthr) */
    pthread_t th[L23_MAXT];
    l23_warg wa[L23_MAXT];
} l23_pool;

struct fft3d_plan {
    int L, batch;
    int pf, pw, nt; /* volume-mode knobs */
    int pt;         /* fused-mode plane-phase t1 prefetch */
    void (*exec)(const struct fft3d_plan *, const double _Complex *,
                 double _Complex *);
    double *cdist8; /* the 11 DISTINCT cosines cos(2pi m/23), splatted 8x */
    double *sdist8; /* the 11 DISTINCT sines   sin(2pi m/23), splatted 8x */
    double *cdist4; /* the same, splatted 4x */
    double *sdist4;
    double *t1f; /* fused-mode t1 arena: batch padded volumes (batch < 32) */
    void *t1f_blk;
    l23_ctx ctx[L23_MAXT];
    l23_pool *pl;
    void *block;
    double _Complex *ti, *to; /* transient buffers for the plan-time tuner */
    size_t tnn;
};

/* MID barrier (between the fused X pass and plane phase): workers post
 * their arrival flag and spin on the release word; main collects and
 * releases.  Transitive release/acquire through main makes every worker's
 * X stores visible to every plane reader. */
static inline void l23_bar_mid(l23_pool *pl, int tid, int T)
{
    unsigned long e =
        2ul * (unsigned long)atomic_load_explicit(&pl->gen,
                                                  memory_order_relaxed);
    if (tid == 0) {
        for (int t = 1; t < T; ++t)
            while (atomic_load_explicit(&pl->arr[t].v, memory_order_acquire) < e)
                l23_cpu_relax();
        atomic_store_explicit(&pl->rel, e, memory_order_release);
    } else {
        atomic_store_explicit(&pl->arr[tid].v, e, memory_order_release);
        while (atomic_load_explicit(&pl->rel, memory_order_acquire) < e)
            l23_cpu_relax();
    }
}

/* JOIN (end of every job): workers post their flag and go straight back to
 * the generation spin -- they need not wait for each other, because main
 * cannot dispatch the next job (or return to the driver) until it has seen
 * every flag. */
static inline void l23_bar_join(l23_pool *pl, int tid, int T)
{
    unsigned long e =
        2ul * (unsigned long)atomic_load_explicit(&pl->gen,
                                                  memory_order_relaxed) + 1;
    if (tid == 0) {
        for (int t = 1; t < T; ++t)
            while (atomic_load_explicit(&pl->arr[t].v, memory_order_acquire) < e)
                l23_cpu_relax();
    } else {
        atomic_store_explicit(&pl->arr[tid].v, e, memory_order_release);
    }
}

/* ---- instantiate the kernel template: 512-bit and 256-bit ---- */
#if defined(__has_include)
#  if __has_include("L23_matrixsimd.c")
#    define L23_SELF "L23_matrixsimd.c"
#  elif __has_include("impl/L23_matrixsimd.c")
#    define L23_SELF "impl/L23_matrixsimd.c"
#  endif
#else
#  define L23_SELF "L23_matrixsimd.c"
#endif

#ifdef L23_SELF
#  define L23_TEMPLATE_PASS 1
#  define WC 4
#  define SUF(x) x##_w4
#  include L23_SELF
#  undef SUF
#  undef WC
#  undef L23_TEMPLATE_PASS

#  define L23_TEMPLATE_PASS 2
#  define WC 2
#  define SUF(x) x##_w2
#  include L23_SELF
#  undef SUF
#  undef WC
#  undef L23_TEMPLATE_PASS
#else
#  error "L23_matrixsimd.c must be able to #include itself"
#endif

/* Width-independent pool job: first-touch the fused t1 arena with the SAME
 * static plane partition the fused plane loop uses, so a plane's reader
 * owns its pages. */
static void l23_work_t1f(const fft3d_plan *p, int tid)
{
    l23_pool *pl = p->pl;
    const int T = pl->tw;
    if (tid >= T) return;
    const long npl = (long)p->batch * 23;
    long g0 = npl * tid / T, g1 = npl * (tid + 1) / T;
    for (long g = g0; g < g1; ++g)
        memset(p->t1f + (size_t)g * L23_T1P, 0, L23_T1P * sizeof(double));
    l23_bar_join(pl, tid, T);
}

/* Dispatch one job to the pool and run it as tid 0; returns when the whole
 * job is done (every job ends in its own barrier). */
static void l23_run_job(const fft3d_plan *p, l23_workfn fn,
                        const double _Complex *in, double _Complex *out)
{
    l23_pool *pl = p->pl;
    pl->in = in;
    pl->out = out;
    pl->workfn = fn;
    ++pl->gen_local;
    atomic_store_explicit(&pl->gen, pl->gen_local, memory_order_release);
    fn(p, 0);
}

static void l23_exec_fused_w4(const fft3d_plan *p, const double _Complex *in,
                              double _Complex *out)
{ l23_run_job(p, work_fused_w4, in, out); }
static void l23_exec_fused_w2(const fft3d_plan *p, const double _Complex *in,
                              double _Complex *out)
{ l23_run_job(p, work_fused_w2, in, out); }
static void l23_exec_vol_w4(const fft3d_plan *p, const double _Complex *in,
                            double _Complex *out)
{ l23_run_job(p, work_vol_w4, in, out); }
static void l23_exec_vol_w2(const fft3d_plan *p, const double _Complex *in,
                            double _Complex *out)
{ l23_run_job(p, work_vol_w2, in, out); }

/* Worker: pin to the harness-consistent CPU, allocate and FIRST-TOUCH this
 * thread's scratch (NUMA locality: the owner touches it, after pinning),
 * then spin on the pool generation. */
static void *l23_worker(void *arg)
{
    l23_warg *wa = arg;
    struct fft3d_plan *p = wa->p;
    l23_pool *pl = p->pl;
    const int tid = wa->tid;

    if (wa->cpu >= 0) {
        cpu_set_t s;
        CPU_ZERO(&s);
        CPU_SET(wa->cpu, &s);
        pthread_setaffinity_np(pthread_self(), sizeof s, &s);
    }
    {
        const size_t npb = 24 * 24 * 2;
        const size_t nps = 1058 + 6;
        const size_t nctxd = npb + L23_T1V + nps;
        void *cb = NULL;
        if (posix_memalign(&cb, 64, nctxd * sizeof(double)) != 0 || !cb) {
            atomic_store(&pl->fail, 1);
        } else {
            memset(cb, 0, nctxd * sizeof(double));
            double *q = (double *)cb;
            p->ctx[tid].blk = cb;
            p->ctx[tid].pb = q; q += npb;
            p->ctx[tid].t1 = q; q += L23_T1V;
            p->ctx[tid].ps = q;
        }
    }
    atomic_fetch_add(&pl->ready, 1);

    long last = 0;
    for (;;) {
        while (atomic_load_explicit(&pl->gen, memory_order_acquire) == last)
            l23_cpu_relax();
        ++last;
        if (atomic_load_explicit(&pl->shutdown, memory_order_relaxed)) break;
        pl->workfn(p, tid);
    }
    return NULL;
}

static double l23_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Sustained-clock probe (phase-1 pattern, serial): four independent 4-cycle
 * FMA chains issue 1 FMA/cycle, so clk = 4*N / elapsed at either width. */
static volatile double l23_clk_sink;
#define L23_CLK_PROBE(FN, VTYPE, ...)                                          \
    static double FN(void)                                                     \
    {                                                                          \
        const long N = 500000;                                                 \
        double best = 1e30;                                                    \
        VTYPE b = {__VA_ARGS__}, c = {__VA_ARGS__};                            \
        VTYPE a0 = {__VA_ARGS__}, a1 = a0, a2 = a0, a3 = a0;                   \
        b = b * 1e-15 + b;                                                     \
        c = c * 1e-300;                                                        \
        for (int rep = 0; rep < 3; ++rep) {                                    \
            double t0 = l23_now();                                             \
            for (long i = 0; i < N; ++i) {                                     \
                a0 = a0 * b + c; a1 = a1 * b + c;                              \
                a2 = a2 * b + c; a3 = a3 * b + c;                              \
            }                                                                  \
            double dt = l23_now() - t0;                                        \
            if (dt < best) best = dt;                                          \
        }                                                                      \
        l23_clk_sink = a0[0] + a1[0] + a2[0] + a3[0];                          \
        return 4.0 * (double)N / best;                                         \
    }
L23_CLK_PROBE(l23_clk512, vd_w4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
L23_CLK_PROBE(l23_clk256, vd_w2, 1.0, 1.0, 1.0, 1.0)
#undef L23_CLK_PROBE

static void l23_tune_free(fft3d_plan *p)
{
    free(p->ti);
    free(p->to);
    p->ti = NULL;
    p->to = NULL;
    p->tnn = 0;
}

/* Deterministic pseudo-random tuning data.  Filled SERIALLY on purpose: the
 * driver fread()s `in` and memset()s `out` on thread 0, so the real caller
 * buffers are socket-0 resident on the node -- the arena must reproduce
 * that placement or the tuner races candidates on a NUMA layout the scored
 * run never sees. */
static int l23_tune_alloc(fft3d_plan *p, int nv)
{
    size_t n = (size_t)nv * 12167;
    if (p->tnn >= n) return 1;
    l23_tune_free(p);
    if (posix_memalign((void **)&p->ti, 64, n * sizeof *p->ti) != 0) { p->ti = NULL; return 0; }
    if (posix_memalign((void **)&p->to, 64, n * sizeof *p->to) != 0) {
        free(p->ti); p->ti = NULL; p->to = NULL; return 0;
    }
    p->tnn = n;
    unsigned sr = 12345u;
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

/* Streaming-regime tuner arena: enough volumes that in + out together exceed
 * ~2.5x this machine's L3, rounded up to a multiple of 32 so every thread
 * gets the same volume count during the walk. */
static int l23_tune_nv(int batch)
{
    long l3 = -1;
#ifdef _SC_LEVEL3_CACHE_SIZE
    l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
    int cap = 160;
    if (l3 > 0) {
        double nv = 2.5 * (double)l3 / (2.0 * 12167.0 * 16.0);
        cap = nv < 160.0 ? 160 : (nv > 448.0 ? 448 : (int)nv);
    }
    cap = ((cap + 31) / 32) * 32;
    return batch < cap ? batch : cap;
}

const char *fft3d_name(void) { return "L23_matrixsimd"; }

static const char *g_desc =
    "dense 23x23 DFT matrix per axis, conjugate-pair folded, SIMD across lines";
static char g_tune[128];

const char *fft3d_description(void) { return g_desc; }

int fft3d_supports(int L) { return L == 23; }

static int l23_verbose(void)
{
    const char *e = getenv("L23_VERBOSE");
    return e && *e && *e != '0';
}

typedef void (*l23_fn)(const fft3d_plan *, const double _Complex *,
                       double _Complex *);

/* Time one candidate on the tuner arena: per-candidate licence warmup
 * >= 1.5 ms (phase-1 lesson: a ymm candidate timed inside another's
 * not-yet-decayed AVX-512 licence window can never show a win), then
 * `reps` samples of `inner` executes, min.  Returns seconds per execute. */
static double l23_time_cand(fft3d_plan *p, l23_fn fn, int tw, int nt, int pf,
                            int pw, int nv, int inner, int reps)
{
    int sb = p->batch, stw = p->pl->tw, snt = p->nt, spf = p->pf, spw = p->pw;
    p->batch = nv; p->pl->tw = tw; p->nt = nt; p->pf = pf; p->pw = pw;
    double best = 1e30;
    { double t0 = l23_now();
      do { fn(p, p->ti, p->to); } while (l23_now() - t0 < 1.5e-3); }
    for (int r = 0; r < reps; ++r) {
        double t0 = l23_now();
        for (int q = 0; q < inner; ++q) fn(p, p->ti, p->to);
        double dt = l23_now() - t0;
        if (dt < best) best = dt;
    }
    p->batch = sb; p->pl->tw = stw; p->nt = snt; p->pf = spf; p->pw = spw;
    return best / (double)inner;
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != 23 || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;

    int maxt = omp_get_max_threads();
    if (maxt > L23_MAXT) maxt = L23_MAXT; /* never take more than given */
    if (maxt < 1) maxt = 1;

    /* ---- splatted constant tables (small, shared, read-only) ---- */
    {
        const size_t nd = 11;
        const size_t ntot = 24 * nd + 64;
        void *blk = NULL;
        if (posix_memalign(&blk, 64, ntot * sizeof(double)) != 0 || !blk) {
            free(p);
            return NULL;
        }
        memset(blk, 0, ntot * sizeof(double));
        p->block = blk;
        double *q = (double *)blk;
        p->cdist8 = q; q += 8 * nd;
        p->sdist8 = q; q += 8 * nd;
        p->cdist4 = q; q += 4 * nd;
        p->sdist4 = q;
        /* long double trig: the table itself is then good to ~1e-19 */
        const long double twopi = 6.283185307179586476925286766559005768394L;
        for (int m = 1; m <= 11; ++m) {
            double c = (double)cosl(twopi * (long double)m / 23.0L);
            double sv = (double)sinl(twopi * (long double)m / 23.0L);
            for (int t = 0; t < 8; ++t) p->cdist8[(m - 1) * 8 + t] = c;
            for (int t = 0; t < 8; ++t) p->sdist8[(m - 1) * 8 + t] = sv;
            for (int t = 0; t < 4; ++t) p->cdist4[(m - 1) * 4 + t] = c;
            for (int t = 0; t < 4; ++t) p->sdist4[(m - 1) * 4 + t] = sv;
        }
    }

    /* ---- read the harness's OMP thread->CPU mapping from one throwaway
     * OpenMP region, so the pool pins exactly where OMP_PROC_BIND=close /
     * OMP_PLACES=cores would put each thread.  This also pins the initial
     * thread (pool tid 0) to place 0.  GOMP's own workers go to sleep
     * after their spin timeout and are never used again. ---- */
    int cpus[L23_MAXT];
    for (int t = 0; t < L23_MAXT; ++t) cpus[t] = -1;
#pragma omp parallel num_threads(maxt)
    {
        int t = omp_get_thread_num();
        if (t < L23_MAXT) cpus[t] = sched_getcpu();
    }
    {
        int pin = getenv("OMP_PROC_BIND") != NULL;
        for (int a = 0; pin && a < maxt; ++a)
            for (int b2 = a + 1; b2 < maxt; ++b2)
                if (cpus[a] == cpus[b2]) pin = 0; /* unbound run: don't pin */
        if (!pin)
            for (int t = 0; t < L23_MAXT; ++t) cpus[t] = -1;
    }

    /* ---- main thread's own scratch (tid 0; already pinned to place 0) ---- */
    {
        const size_t npb = 24 * 24 * 2;
        const size_t nps = 1058 + 6;
        const size_t nctxd = npb + L23_T1V + nps;
        void *cb = NULL;
        if (posix_memalign(&cb, 64, nctxd * sizeof(double)) != 0 || !cb) {
            fft3d_destroy(p);
            return NULL;
        }
        memset(cb, 0, nctxd * sizeof(double));
        double *q = (double *)cb;
        p->ctx[0].blk = cb;
        p->ctx[0].pb = q; q += npb;
        p->ctx[0].t1 = q; q += L23_T1V;
        p->ctx[0].ps = q;
    }

    /* ---- the persistent pool (PANEL_BRIEF: thread creation is setup) ---- */
    {
        void *pb = NULL;
        if (posix_memalign(&pb, 64, sizeof(l23_pool)) != 0 || !pb) {
            fft3d_destroy(p);
            return NULL;
        }
        memset(pb, 0, sizeof(l23_pool));
        p->pl = (l23_pool *)pb;
        p->pl->nthr = maxt;
        p->pl->tw = maxt;
        for (int t = 1; t < maxt; ++t) {
            p->pl->wa[t].p = p;
            p->pl->wa[t].tid = t;
            p->pl->wa[t].cpu = cpus[t];
            if (pthread_create(&p->pl->th[t], NULL, l23_worker,
                               &p->pl->wa[t]) != 0) {
                p->pl->nthr = t; /* join only what was spawned */
                atomic_store(&p->pl->fail, 1);
                break;
            }
        }
        maxt = p->pl->nthr;
        while (atomic_load(&p->pl->ready) < maxt - 1) l23_cpu_relax();
        if (atomic_load(&p->pl->fail)) {
            fft3d_destroy(p);
            return NULL;
        }
    }

    /* ---- fused-mode t1 arena (batch < 32), first-touched by the pool ---- */
    if (batch < L23_FUSEB) {
        if (posix_memalign(&p->t1f_blk, 64,
                           (size_t)batch * L23_T1V * sizeof(double)) != 0 ||
            !p->t1f_blk) {
            fft3d_destroy(p);
            return NULL;
        }
        p->t1f = (double *)p->t1f_blk;
        l23_run_job(p, l23_work_t1f, NULL, NULL);
    }

    /* =====================================================================
     * Plan-time tuner.  Phase-1 discipline: canonical-order walk, two full
     * sweeps with per-candidate min, hysteretic displacement, deterministic
     * arena.  All candidates are in ONE bit class (see header), so a
     * different pick on a different day changes no output bits.
     * ===================================================================== */
    p->exec = batch < L23_FUSEB ? l23_exec_fused_w4 : l23_exec_vol_w4;
    if (batch < L23_FUSEB) {
        /* FUSED regime.  tw walk (barrier cost and imbalance move against
         * each other: 23 planes/volume at B=1 leaves 32-23 threads idle in
         * the plane phase); a volume-parallel row for 1 < B; a serial row
         * as the phase-1 floor (B=1 may not parallelise -- if it does not,
         * say so with the pick, do not fake it). */
        struct { l23_fn fn; int tw; const char *tag; } cand[12];
        int nc = 0;
        static const int tlist[5] = {L23_MAXT, 24, 16, 12, 8};
        for (int s = 0; s < 5; ++s) {
            int T = tlist[s] > maxt ? maxt : tlist[s];
            int dup = 0;
            for (int u = 0; u < nc; ++u)
                if (cand[u].tw == T) dup = 1;
            if (!dup) {
                cand[nc].fn = l23_exec_fused_w4; cand[nc].tw = T;
                cand[nc].tag = "512-bit fused"; ++nc;
            }
        }
        if (batch > 1) {
            cand[nc].fn = l23_exec_vol_w4;
            cand[nc].tw = batch < maxt ? batch : maxt;
            cand[nc].tag = "512-bit vol-par"; ++nc;
        }
        cand[nc].fn = l23_exec_vol_w4; cand[nc].tw = 1;
        cand[nc].tag = "512-bit serial"; ++nc;
        cand[nc].fn = l23_exec_fused_w2; cand[nc].tw = maxt;
        cand[nc].tag = "256-bit fused"; ++nc;

        double bestt[12];
        for (int s = 0; s < nc; ++s) bestt[s] = 1e30;
        int nv = batch;
        int inner = (256 + nv - 1) / nv;
        if (inner > 256) inner = 256;
        if (l23_tune_alloc(p, nv)) {
            for (int sweep = 0; sweep < 2; ++sweep)
                for (int s = 0; s < nc; ++s) {
                    /* the serial row is ~50 us/exec; trim its inner count
                     * so the walk stays cheap without losing resolution */
                    int inn = cand[s].tw == 1 ? (inner + 3) / 4 : inner;
                    if (inn < 1) inn = 1;
                    double t = l23_time_cand(p, cand[s].fn, cand[s].tw,
                                             0, 0, 0, nv, inn, 3);
                    if (t < bestt[s]) bestt[s] = t;
                }
            int bs = 0;
            for (int s = 1; s < nc; ++s)
                if (bestt[s] < 0.97 * bestt[bs]) bs = s;
            p->exec = cand[bs].fn;
            p->pl->tw = cand[bs].tw;
            p->nt = 0; p->pf = 0; p->pw = 0;
            snprintf(g_tune, sizeof g_tune,
                     ", tune[pick=%.2f(%s T=%d) inc=%.2f us/t nv=%d]",
                     bestt[bs] * 1e6 / nv, cand[bs].tag, cand[bs].tw,
                     bestt[0] * 1e6 / nv, nv);
            if (l23_verbose())
                for (int s = 0; s < nc; ++s)
                    fprintf(stderr, "[L23_matrixsimd mt tune] %-16s T=%-2d %8.3f us/transform%s\n",
                            cand[s].tag, cand[s].tw, bestt[s] * 1e6 / nv,
                            s == bs ? "  <== kept" : "");
        }
    } else {
        /* VOLUME-PARALLEL regime.  Joint (width, nt, pf, pw) walk at the
         * full thread count on a >L3 arena.  Head = NT + no prefetch: at
         * 32 threads the batch cells sit at the socket bandwidth wall, NT
         * deletes the out-RFO third of the traffic, and prefetch across 32
         * competing streams is expected tax; the single-core streaming
         * winner (plain + pf=2 + pw=1) races right behind.  4% margin
         * (phase-1 streaming noise exceeded 2%). */
        static const struct { signed char w2, nt, pf, pw; } combo[] = {
            {0, 1, 0, 0}, /* head: NT, no prefetch                      */
            {0, 1, 2, 0}, /* NT + in-pass X prefetch                    */
            {0, 0, 2, 1}, /* the single-core streaming winner           */
            {0, 0, 0, 1},
            {0, 0, 0, 0},
            {0, 0, 2, 0},
            {1, 1, 0, 0}, /* 256-bit sanity                             */
            {1, 0, 0, 0},
        };
        enum { NCOMBO = (int)(sizeof combo / sizeof combo[0]) };
        double bc[NCOMBO];
        for (int s = 0; s < NCOMBO; ++s) bc[s] = 1e30;
        int nv = l23_tune_nv(batch);
        if (l23_tune_alloc(p, nv)) {
            for (int sweep = 0; sweep < 2; ++sweep)
                for (int s = 0; s < NCOMBO; ++s) {
                    l23_fn fn = combo[s].w2 ? l23_exec_vol_w2 : l23_exec_vol_w4;
                    double t = l23_time_cand(p, fn, maxt, combo[s].nt,
                                             combo[s].pf, combo[s].pw,
                                             nv, 1, 4);
                    if (t < bc[s]) bc[s] = t;
                }
            int bs = 0;
            for (int s = 1; s < NCOMBO; ++s)
                if (bc[s] < 0.96 * bc[bs]) bs = s;
            p->exec = combo[bs].w2 ? l23_exec_vol_w2 : l23_exec_vol_w4;
            p->pl->tw = maxt;
            p->nt = combo[bs].nt; p->pf = combo[bs].pf; p->pw = combo[bs].pw;
            snprintf(g_tune, sizeof g_tune,
                     ", tune[pick=%.2f(nt%d pf%d pw%d) inc=%.2f us/t nv=%d]",
                     bc[bs] * 1e6 / nv, p->nt, p->pf, p->pw,
                     bc[0] * 1e6 / nv, nv);
            if (l23_verbose())
                for (int s = 0; s < NCOMBO; ++s)
                    fprintf(stderr, "[L23_matrixsimd mt tune >L3 nv=%d T=%d] w2=%d nt=%d pf=%d pw=%d %8.3f us/transform%s\n",
                            nv, maxt, combo[s].w2, combo[s].nt, combo[s].pf,
                            combo[s].pw, bc[s] * 1e6 / nv,
                            s == bs ? "  <== kept" : "");
        } else {
            p->nt = 1;
        }
    }
    l23_tune_free(p);

    /* env overrides for same-window forced A/Bs without a recompile; the
     * node sets none of these.  What actually runs is baked into the
     * description string either way. */
    { const char *e = getenv("L23_T");    if (e && *e) { int t = atoi(e);
        if (t >= 1 && t <= maxt) p->pl->tw = t; } }
    { const char *e = getenv("L23_NT");   if (e && *e) p->nt = atoi(e); }
    { const char *e = getenv("L23_PT");   if (e && *e) p->pt = atoi(e); }
    { const char *e = getenv("L23_PF");   if (e && *e) p->pf = atoi(e); }
    { const char *e = getenv("L23_PW");   if (e && *e) p->pw = atoi(e); }
    { const char *e = getenv("L23_MODE"); if (e && *e) {
        /* 1=fused, 2=vol-parallel, 3=serial, 4=fused-w2, 5=vol-w2 */
        int m = atoi(e);
        if (m == 1 && batch < L23_FUSEB) p->exec = l23_exec_fused_w4;
        else if (m == 2) p->exec = l23_exec_vol_w4;
        else if (m == 3) { p->exec = l23_exec_vol_w4; p->pl->tw = 1; }
        else if (m == 4 && batch < L23_FUSEB) p->exec = l23_exec_fused_w2;
        else if (m == 5) p->exec = l23_exec_vol_w2;
    } }

    {
        const char *mode =
            (p->exec == l23_exec_fused_w4) ? "fused 512-bit" :
            (p->exec == l23_exec_fused_w2) ? "fused 256-bit" :
            (p->exec == l23_exec_vol_w2)   ? "vol-parallel 256-bit" :
            (p->pl->tw == 1)               ? "serial 512-bit" :
                                             "vol-parallel 512-bit";
        double c512 = l23_clk512(), c256 = l23_clk256();
        static char g_desc_buf[448];
        snprintf(g_desc_buf, sizeof g_desc_buf,
                 "dense 23x23/axis conj-folded, pinned consts, X-first, "
                 "spin-pool %s T=%d, nt=%d pf=%d pw=%d%s, clk512/256=%.2f/%.2f GHz",
                 mode, p->pl->tw, p->nt, p->pf, p->pw, g_tune,
                 c512 * 1e-9, c256 * 1e-9);
        g_desc = g_desc_buf;
    }
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    if (p->pl) {
        atomic_store_explicit(&p->pl->shutdown, 1, memory_order_relaxed);
        ++p->pl->gen_local;
        atomic_store_explicit(&p->pl->gen, p->pl->gen_local,
                              memory_order_release);
        for (int t = 1; t < p->pl->nthr; ++t) pthread_join(p->pl->th[t], NULL);
        free(p->pl);
    }
    l23_tune_free(p);
    for (int t = 0; t < L23_MAXT; ++t) free(p->ctx[t].blk);
    free(p->t1f_blk);
    free(p->block);
    free(p);
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    plan->exec(plan, in, out);
}

#endif /* L23_TEMPLATE_PASS */
