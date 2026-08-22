/* =============================================================================
 * L23_matrixsimd -- 23^3 complex-double forward DFT as three dense 23x23
 *                   matrix passes, conjugate-pair folded, vectorised across
 *                   whole lines (SIMD lanes = independent lines).
 *
 * FIRST IMPLEMENTATION FOR THIS GEOMETRY (round panel_r6).  The architecture
 * is adopted wholesale from L17_matrixsimd (the L=17 leader), with attribution:
 * that entry's five rounds of records establish the layout, the pre-splatted
 * coefficient tables, the fused transposing stores, the k-blocked kernel, the
 * X-first batch ordering, cross-volume pipelining, and the bit-class tuner
 * discipline.  This file re-derives all of it at L = 23 and measures where the
 * dense approach stands at a prime half again bigger than 17.
 *
 * TECHNIQUE
 *   Row-column: one dense length-23 DFT matrix applied along each axis, with
 *   the j <-> 23-j conjugate pair folded first (FFTW's dft-generic form), so
 *   one complex 23x23 matrix-vector product becomes two REAL matrix products
 *   on complex data:
 *
 *     u_j = x_j + x_{23-j},  v_j = x_j - x_{23-j}          (j = 1..11)
 *     P_k = x_0 + sum_{j=1..11} cos(2pi kj/23) u_j         (k = 0..11)
 *     R_k =       sum_{j=1..11} sin(2pi kj/23) (-i v_j)    (k = 1..11)
 *     X_k = P_k + R_k,  X_{23-k} = P_k - R_k,  X_0 = P_0
 *
 *   All surviving coefficients are REAL, so the interleaved complex layout the
 *   driver hands us is already the right SIMD layout: a real scalar times an
 *   interleaved complex vector is one vector FMA with a full-width memory
 *   operand.  No cross-lane arithmetic anywhere; the only permutes are the
 *   re/im swap of (-i v) and the WCxWC tile transposes that feed axes 2 and 3.
 *
 *   Why not a nested (Rader-permuted) kernel like L17's cyclic/negacyclic
 *   split: (Z/23)^x / {+-1} is cyclic of order 11, and x^11 - 1 has no
 *   sign-only factorisation (x^8-1 at L=17 did).  A Winograd cyclic-11
 *   convolution needs real pre/post-addition networks; that is a next-round
 *   experiment, not a free win.  See the strategy record.
 *
 * OPERATION COUNT  (per 23^3 volume = 3*529 = 1587 lines)
 *   Per line in real arithmetic: (12+11) accumulators x 11 rank-1 updates
 *   = 253 FMA (506 flop) + 22 butterfly + 22 combine add/sub (88 flop)
 *   = 594 flop, against 4232 flop for the naive complex 23x23 matvec (7.1x).
 *   Per volume: 1587 * 594 = 943 kflop.  The driver's 5 N log2 N yardstick is
 *   824 kflop, so reported GF/s ~= real Gflop/s at this size.
 *   As vector work: one chunk (WC lines) costs 253 FMA + 44 add/sub = 297 FP
 *   ops; a volume is 2*23*NOFF + (529/WC rounded up) chunks = 409 (WC=4).
 *   409 * 297 = 121.5k vector FP ops; on the node's one 512-bit FMA unit at
 *   2.9 GHz that is a ~42 us floor at B=1.
 *
 * LAYOUT / SIMD
 *   WC complex per vector (4 = zmm, 2 = ymm).  Lanes hold WC different lines
 *   from a contiguous run of a free index; every load/store is a contiguous
 *   vector access, every coefficient a lane-invariant memory operand from a
 *   PRE-SPLATTED table (L17 lesson: scalar table + embedded broadcast makes
 *   gcc materialise every splat into a stack slot -- measured there, not
 *   retried here).  The last chunk of a 23-long index overlaps the previous
 *   one (offsets 0,4,8,12,16,19): it recomputes 1 line and stores
 *   bit-identical values over it -- cheaper than masking.
 *
 *   Passes (X-last order; X-first swaps the X pass to the front):
 *     Y : in[x][y][z] --lanes over z --> pb[z][ky]        (transposing store)
 *     Z : pb[z][ky]   --lanes over ky--> t1[x][ky][kz]    (transposing store)
 *     X : t1[x][p]    --lanes over p (529 wide) --> out[kx][p]  (plain store)
 *   Y and Z run plane by plane (8.5 KiB plane stays in L1); pass X walks the
 *   190 KiB volume (19% of the node's L2).
 *
 * KERNELS AND VARIANTS (all measured, selected within bit classes)
 *   chunk23  : unblocked, 23 accumulators live -- fits the 32-register EVEX
 *              file, guaranteed spills on 16 registers.
 *   chunk23b : k-blocked (block A: k=0..5, block B: k=6..11), peak ~12 live;
 *              block B either recomputes the butterflies (+22 FP ops) or
 *              reloads them from an L1 scratch (bst=1), as in L17's chunk17b.
 *   X-first ordering (REORD) for batch >= 64: adopted from L17_matrixsimd r3,
 *   which measured -14..-17% at batch by spreading the output writes across
 *   the volume's compute instead of bursting them at the end.
 *   Cross-volume pipelining: adopted from L17_matrixsimd r4 (exec18): t1 is
 *   double-buffered and volume b+1's X chunks interleave into volume b's
 *   plane phase, so every volume's input read overlaps compute.
 *
 * TUNER / BIT-CLASS DISCIPLINE (adopted from L17_matrixsimd r3)
 *   The pass ORDER changes the association of the triple sum, so X-last and
 *   X-first are within-tolerance but not bit-identical.  A wall-clock tuner is
 *   not deterministic across processes, and the panel checks bit-identical
 *   output across runs; therefore since r7 there is ONE class at every batch
 *   size -- pinned X-first (adopted from L23_rader r6, which measured X-first
 *   winning at B=1 too) -- and the tuner selects freely only within it
 *   (kernel width, P-parking, pipelining, NT staging, deferred-Z, za -- all
 *   cmp-verified on wallaby to change no bits; see the strategy record).
 *   X-last and the table kernels exist for forced experiments only; since
 *   r8 they are not even timed at plan time.
 *   The prefetch knobs (pf, pw) change no bits by construction.
 *
 * ROUND r7 ADDITIONS (all scheduling; the arithmetic is settled -- see
 * L23_rader r6's operation-count argument that no cyclic-11 convolution
 * beats the folded direct form on FMA hardware):
 *   * X-first at every batch size (L23_rader r6).
 *   * t1 planes padded 1058 -> 1064 doubles (L23_T1P) so X-pass stores and
 *     plane-phase reloads of t1 never split cache lines.
 *   * Deferred-Z plane schedule (L17_matrixsimd r6): pb double-buffered,
 *     Y(x+1) runs between Y(x) and Z(x) to break the once-per-plane
 *     store->load junction on the plane buffer.
 *   * Paced prefetchw on the out plane ahead of the Z stores (panel-r5
 *     VERDICT via L17_matrixsimd r6), tuner-gated jointly with pf.
 *   * Per-candidate licence warmup in the tuner (L17_rader r6).
 *
 * ROUND r8 ADDITIONS:
 *   * DETERMINISTIC TUNER: canonical-order candidate list with a 2%
 *     hysteresis margin (a challenger must beat the incumbent by >2% to be
 *     picked) and two full fixed-order sweeps with per-candidate minima
 *     (L17_rader r6).  The r7 verdict (§3a) found all three L=23 cells
 *     reported a timed variant the harness never checked, because
 *     min-of-noise across 18 near-tied variants picked differently across
 *     processes.  Near-ties now resolve identically in every process.
 *     Never-selectable variants are no longer timed at plan time.
 *   * za variants (40..47): z-extent of the t1 scratch padded 23 -> 24
 *     complex per row, so every plane-phase Y load is 64-byte aligned
 *     (flat layout: ~3/4 split a line); costs 138 instead of 133 X-pass
 *     chunks (+1.2% volume FP).  Wallaby measures it a wash both regimes;
 *     kept selectable for the node (1 FMA unit, different split economics).
 *   * -funroll-loops build-flag gap (L45_pfa r7): CHECKED, does not apply
 *     here -- same-window A/B is a wash, hot kernels are hand-unrolled.
 *     Scalar-instruction audit (L45_pfa r7): 142 scalar instructions in the
 *     whole hot exec, no offset-table materialisation; nothing to fix.
 *
 * ASSUMPTIONS
 *   * L == 23 only; in/out distinct (driver guarantees), 8-byte aligned.
 *   * gcc/clang vector extensions, -ffp-contract=fast for FMA formation.  No
 *     intrinsics, so one source serves AVX-512, AVX2 and SSE2, and the
 *     512-bit path can be run (emulated) on an AVX2 machine for verification.
 *   * The template body is instantiated twice by a self-#include.
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
 * rather than the FMA port (L17_matrixsimd's measured choice). */
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
 * plain full-width memory operand. */
#define CGET(base, i) VLD((base) + (size_t)(i) * VDW)
#define CSTEP_C (12 * VDW)
#define CSTEP_S (11 * VDW)
#if WC == 4
#  define CTAB(p) ((p)->ctab8)
#  define STAB(p) ((p)->stab8)
#else
#  define CTAB(p) ((p)->ctab4)
#  define STAB(p) ((p)->stab4)
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
 *  One chunk: the length-23 DFT of WC lines at once, unblocked.
 *    src, rs : element (j=0, lane 0); rs doubles between successive j
 *    dst, da, db : output element (m) of lane (f) at dst + f*da + m*db
 *    tr : 1 -> in-register tile transpose into rows of dst
 *         0 -> lanes stay contiguous, plain vector stores
 * --------------------------------------------------------------------- */
static inline __attribute__((always_inline)) void
SUF(chunk23)(const double *restrict src, long rs, double *restrict dst, long da,
             long db, const double *restrict ct, const double *restrict st, int tr)
{
    VT P0, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11;
    VT R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11;

    /* ---- symmetric half: 12 accumulators, 11 rank-1 updates ---- */
    {
        VT v0 = VLD(src);
        P0 = v0; P1 = v0; P2 = v0; P3 = v0; P4 = v0; P5 = v0;
        P6 = v0; P7 = v0; P8 = v0; P9 = v0; P10 = v0; P11 = v0;
/* Do NOT unroll: 11 iterations x 12 independent chains already saturate the
 * FMA units; unrolling makes gcc spill (L17_matrixsimd's measured lesson). */
#pragma GCC unroll 1
        for (int j = 1; j <= 11; ++j) {
            VT u = VLD(src + (long)j * rs) + VLD(src + (long)(23 - j) * rs);
            const double *c = ct + (size_t)(j - 1) * CSTEP_C;
            P0 += CGET(c, 0) * u;   P1 += CGET(c, 1) * u;
            P2 += CGET(c, 2) * u;   P3 += CGET(c, 3) * u;
            P4 += CGET(c, 4) * u;   P5 += CGET(c, 5) * u;
            P6 += CGET(c, 6) * u;   P7 += CGET(c, 7) * u;
            P8 += CGET(c, 8) * u;   P9 += CGET(c, 9) * u;
            P10 += CGET(c, 10) * u; P11 += CGET(c, 11) * u;
        }
    }

    /* ---- antisymmetric half: 11 accumulators, j=1 peeled to a multiply ---- */
    {
        VT w = MULI(VLD(src + rs) - VLD(src + 22 * rs));
        R1 = CGET(st, 0) * w;  R2 = CGET(st, 1) * w;
        R3 = CGET(st, 2) * w;  R4 = CGET(st, 3) * w;
        R5 = CGET(st, 4) * w;  R6 = CGET(st, 5) * w;
        R7 = CGET(st, 6) * w;  R8 = CGET(st, 7) * w;
        R9 = CGET(st, 8) * w;  R10 = CGET(st, 9) * w;
        R11 = CGET(st, 10) * w;
#pragma GCC unroll 1
        for (int j = 2; j <= 11; ++j) {
            w = MULI(VLD(src + (long)j * rs) - VLD(src + (long)(23 - j) * rs));
            const double *s = st + (size_t)(j - 1) * CSTEP_S;
            R1 += CGET(s, 0) * w;  R2 += CGET(s, 1) * w;
            R3 += CGET(s, 2) * w;  R4 += CGET(s, 3) * w;
            R5 += CGET(s, 4) * w;  R6 += CGET(s, 5) * w;
            R7 += CGET(s, 6) * w;  R8 += CGET(s, 7) * w;
            R9 += CGET(s, 8) * w;  R10 += CGET(s, 9) * w;
            R11 += CGET(s, 10) * w;
        }
    }

    L23_STORE_TAIL();
}

/* -----------------------------------------------------------------------
 *  Same transform, blocked over k so peak register pressure is ~12 vectors
 *  instead of ~25.  Block A: k = 0..5 (outputs m = 0..5 and 18..22);
 *  block B: k = 6..11 (outputs m = 6..17).
 *    bst = 0: block B recomputes the butterflies (+22 FP ops per chunk)
 *    bst = 1: block A parks u_j, w_j in an L1 scratch, block B reloads
 * --------------------------------------------------------------------- */
static inline __attribute__((always_inline)) void
SUF(chunk23b)(const double *restrict src, long rs, double *restrict dst, long da,
              long db, const double *restrict ct, const double *restrict st,
              double *restrict sc, int tr, int bst)
{
#define BFLY_A(j)                                                              \
    const double *c = ct + (size_t)((j) - 1) * CSTEP_C;                        \
    const double *sn = st + (size_t)((j) - 1) * CSTEP_S;                       \
    VT u, w;                                                                   \
    do {                                                                       \
        VT a = VLD(src + (long)(j) * rs), b = VLD(src + (long)(23 - (j)) * rs);\
        u = a + b;                                                             \
        w = MULI(a - b);                                                       \
        if (bst) {                                                             \
            VST(sc + (size_t)(2 * ((j) - 1)) * VDW, u);                        \
            VST(sc + (size_t)(2 * ((j) - 1) + 1) * VDW, w);                    \
        }                                                                      \
    } while (0)

#define BFLY_B(j)                                                              \
    const double *c = ct + (size_t)((j) - 1) * CSTEP_C;                        \
    const double *sn = st + (size_t)((j) - 1) * CSTEP_S;                       \
    VT u, w;                                                                   \
    do {                                                                       \
        if (bst) {                                                             \
            u = VLD(sc + (size_t)(2 * ((j) - 1)) * VDW);                       \
            w = VLD(sc + (size_t)(2 * ((j) - 1) + 1) * VDW);                   \
        } else {                                                               \
            VT a = VLD(src + (long)(j) * rs),                                  \
               b = VLD(src + (long)(23 - (j)) * rs);                           \
            u = a + b;                                                         \
            w = MULI(a - b);                                                   \
        }                                                                      \
    } while (0)

    { /* ---- block A: k = 0..5  ->  outputs m = 0..5, 18..22 ---- */
        VT Q0, X1, X2, X3, X4, X5, Y1, Y2, Y3, Y4, Y5;
        {
            VT P0, P1, P2, P3, P4, P5, R1, R2, R3, R4, R5;
            VT v0 = VLD(src);
            P0 = v0; P1 = v0; P2 = v0; P3 = v0; P4 = v0; P5 = v0;
            {
                BFLY_A(1);
                P0 += CGET(c, 0) * u; P1 += CGET(c, 1) * u; P2 += CGET(c, 2) * u;
                P3 += CGET(c, 3) * u; P4 += CGET(c, 4) * u; P5 += CGET(c, 5) * u;
                R1 = CGET(sn, 0) * w; R2 = CGET(sn, 1) * w; R3 = CGET(sn, 2) * w;
                R4 = CGET(sn, 3) * w; R5 = CGET(sn, 4) * w;
            }
#pragma GCC unroll 1
            for (int j = 2; j <= 11; ++j) {
                BFLY_A(j);
                P0 += CGET(c, 0) * u; P1 += CGET(c, 1) * u; P2 += CGET(c, 2) * u;
                P3 += CGET(c, 3) * u; P4 += CGET(c, 4) * u; P5 += CGET(c, 5) * u;
                R1 += CGET(sn, 0) * w; R2 += CGET(sn, 1) * w; R3 += CGET(sn, 2) * w;
                R4 += CGET(sn, 3) * w; R5 += CGET(sn, 4) * w;
            }
            Q0 = P0;
            X1 = P1 + R1; X2 = P2 + R2; X3 = P3 + R3; X4 = P4 + R4; X5 = P5 + R5;
            Y1 = P1 - R1; Y2 = P2 - R2; Y3 = P3 - R3; Y4 = P4 - R4; Y5 = P5 - R5;
        }
        if (!tr) {
            VST(dst + 0 * db, Q0);
            VST(dst + 1 * db, X1);  VST(dst + 2 * db, X2);
            VST(dst + 3 * db, X3);  VST(dst + 4 * db, X4);
            VST(dst + 5 * db, X5);
            VST(dst + 18 * db, Y5); VST(dst + 19 * db, Y4);
            VST(dst + 20 * db, Y3); VST(dst + 21 * db, Y2);
            VST(dst + 22 * db, Y1);
        } else {
#if WC == 4
            TILE(0, Q0, X1, X2, X3);
            TILE(2, X2, X3, X4, X5);
            TILE(18, Y5, Y4, Y3, Y2);
            TILE(19, Y4, Y3, Y2, Y1);
#else
            TILE(0, Q0, X1);
            TILE(2, X2, X3);
            TILE(4, X4, X5);
            TILE(18, Y5, Y4);
            TILE(20, Y3, Y2);
            TILE(21, Y2, Y1);
#endif
        }
    }
    { /* ---- block B: k = 6..11  ->  outputs m = 6..17 ---- */
        VT X6, X7, X8, X9, X10, X11, Y6, Y7, Y8, Y9, Y10, Y11;
        {
            VT P6, P7, P8, P9, P10, P11, R6, R7, R8, R9, R10, R11;
            VT v0 = VLD(src);
            P6 = v0; P7 = v0; P8 = v0; P9 = v0; P10 = v0; P11 = v0;
            {
                BFLY_B(1);
                P6 += CGET(c, 6) * u;  P7 += CGET(c, 7) * u;
                P8 += CGET(c, 8) * u;  P9 += CGET(c, 9) * u;
                P10 += CGET(c, 10) * u; P11 += CGET(c, 11) * u;
                R6 = CGET(sn, 5) * w;  R7 = CGET(sn, 6) * w;
                R8 = CGET(sn, 7) * w;  R9 = CGET(sn, 8) * w;
                R10 = CGET(sn, 9) * w; R11 = CGET(sn, 10) * w;
            }
#pragma GCC unroll 1
            for (int j = 2; j <= 11; ++j) {
                BFLY_B(j);
                P6 += CGET(c, 6) * u;  P7 += CGET(c, 7) * u;
                P8 += CGET(c, 8) * u;  P9 += CGET(c, 9) * u;
                P10 += CGET(c, 10) * u; P11 += CGET(c, 11) * u;
                R6 += CGET(sn, 5) * w;  R7 += CGET(sn, 6) * w;
                R8 += CGET(sn, 7) * w;  R9 += CGET(sn, 8) * w;
                R10 += CGET(sn, 9) * w; R11 += CGET(sn, 10) * w;
            }
            X6 = P6 + R6; X7 = P7 + R7; X8 = P8 + R8;
            X9 = P9 + R9; X10 = P10 + R10; X11 = P11 + R11;
            Y6 = P6 - R6; Y7 = P7 - R7; Y8 = P8 - R8;
            Y9 = P9 - R9; Y10 = P10 - R10; Y11 = P11 - R11;
        }
        if (!tr) {
            VST(dst + 6 * db, X6);   VST(dst + 7 * db, X7);
            VST(dst + 8 * db, X8);   VST(dst + 9 * db, X9);
            VST(dst + 10 * db, X10); VST(dst + 11 * db, X11);
            VST(dst + 12 * db, Y11); VST(dst + 13 * db, Y10);
            VST(dst + 14 * db, Y9);  VST(dst + 15 * db, Y8);
            VST(dst + 16 * db, Y7);  VST(dst + 17 * db, Y6);
        } else {
#if WC == 4
            TILE(6, X6, X7, X8, X9);
            TILE(10, X10, X11, Y11, Y10);
            TILE(14, Y9, Y8, Y7, Y6);
#else
            TILE(6, X6, X7);
            TILE(8, X8, X9);
            TILE(10, X10, X11);
            TILE(12, Y11, Y10);
            TILE(14, Y9, Y8);
            TILE(16, Y7, Y6);
#endif
        }
    }
#undef BFLY_A
#undef BFLY_B
}

/* -----------------------------------------------------------------------
 *  Pinned-constant kernel: the 23x23 DFT matrix has only 11 distinct cosine
 *  and 11 distinct sine values, because cos(2pi r/23) = cos(2pi (23-r)/23)
 *  (no sign) and sin(2pi r/23) = -sin(2pi (23-r)/23) (a compile-time
 *  vfnmadd).  So the whole matrix runs from 22 register-resident constants
 *  and the ~253 coefficient-table loads per chunk of the rolled kernels
 *  disappear.  This generalises L17_matrixsimd r3's "pinned sine constants"
 *  (which could pin only its 8 nested-kernel sines) to the ENTIRE dense
 *  matrix -- the fold to 11 magnitudes is what L=23 buys over a generic
 *  table.  Both sweeps are fully unrolled (generated, j-major, same
 *  accumulation order as chunk23, so bit-identity is plausible and is
 *  cmp-verified before the tuner may select across the pair).
 *    pc = 0: P0..P11 stay live across the R sweep (peak ~24 registers)
 *    pc = 1: P parked in an L1 scratch across the R sweep (peak ~13)
 * --------------------------------------------------------------------- */
static inline __attribute__((always_inline)) void
SUF(chunk23p)(const double *restrict src, long rs, double *restrict dst, long da,
              long db, const double *restrict cd, const double *restrict sd,
              double *restrict sc, int tr, int pc)
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
    if (pc) {
        VST(sc + 0 * VDW, P0);  VST(sc + 1 * VDW, P1);
        VST(sc + 2 * VDW, P2);  VST(sc + 3 * VDW, P3);
        VST(sc + 4 * VDW, P4);  VST(sc + 5 * VDW, P5);
        VST(sc + 6 * VDW, P6);  VST(sc + 7 * VDW, P7);
        VST(sc + 8 * VDW, P8);  VST(sc + 9 * VDW, P9);
        VST(sc + 10 * VDW, P10); VST(sc + 11 * VDW, P11);
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
    if (pc) {
        P0 = VLD(sc + 0 * VDW);  P1 = VLD(sc + 1 * VDW);
        P2 = VLD(sc + 2 * VDW);  P3 = VLD(sc + 3 * VDW);
        P4 = VLD(sc + 4 * VDW);  P5 = VLD(sc + 5 * VDW);
        P6 = VLD(sc + 6 * VDW);  P7 = VLD(sc + 7 * VDW);
        P8 = VLD(sc + 8 * VDW);  P9 = VLD(sc + 9 * VDW);
        P10 = VLD(sc + 10 * VDW); P11 = VLD(sc + 11 * VDW);
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

/* za layout: offsets covering a z-index PADDED 23 -> 24, exactly WC-aligned
 * (no overlap, no split): the 6th zmm chunk runs lanes 20..23 where lane 23
 * is a pad column that is computed but never read by the Z pass. */
#if WC == 4
static const int SUF(zoff24)[6] = {0, 4, 8, 12, 16, 20};
#else
static const int SUF(zoff24)[12] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};
#endif

#define PBROW (24 * 2) /* plane buffer row stride, doubles (23 padded to 24) */

/* kernel dispatch: KRN is a compile-time constant inside each exec.
 *   0 = unblocked table kernel        1 = k-blocked, recomputed butterflies
 *   2 = k-blocked, L1-parked bflies   3 = pinned constants, P in registers
 *   4 = pinned constants, P parked in L1 */
#define L23_CALL(KRN, srcp, rs, dstp, da, db, tr)                              \
    do {                                                                       \
        if ((KRN) == 0)                                                        \
            SUF(chunk23)((srcp), (rs), (dstp), (da), (db), ct, st, (tr));      \
        else if ((KRN) == 1)                                                   \
            SUF(chunk23b)((srcp), (rs), (dstp), (da), (db), ct, st, sc,        \
                          (tr), 0);                                            \
        else if ((KRN) == 2)                                                   \
            SUF(chunk23b)((srcp), (rs), (dstp), (da), (db), ct, st, sc,        \
                          (tr), 1);                                            \
        else if ((KRN) == 3)                                                   \
            SUF(chunk23p)((srcp), (rs), (dstp), (da), (db), cd, sd, sc,        \
                          (tr), 0);                                            \
        else                                                                   \
            SUF(chunk23p)((srcp), (rs), (dstp), (da), (db), cd, sd, sc,        \
                          (tr), 1);                                            \
    } while (0)

#if WC == 4
#  define CDTAB(p) ((p)->cdist8)
#  define SDTAB(p) ((p)->sdist8)
#else
#  define CDTAB(p) ((p)->cdist4)
#  define SDTAB(p) ((p)->sdist4)
#endif

/* number of X-pass chunk slots (slot i covers lanes i*WC..; last slot is the
 * overlapping tail at 529-WC -- 529 % WC == 1 for both widths) */
#define NX23 (529 / WC + 1)
#define L23_XF0(i) ((i) < 529 / WC ? (long)(i) * WC : (long)(529 - WC))

/* Y pass of one plane: along y (row stride 46), lanes over z, transposed
 * into a (padded) plane buffer.  Z pass: along z (row stride PBROW), lanes
 * over ky, transposed into the destination plane. */
#define L23_YPASS(K, SRCP, DPB)                                                \
    for (int t = 0; t < NOFF23; ++t) {                                         \
        long f0 = SUF(off23)[t];                                               \
        L23_CALL(K, (SRCP) + 2 * f0, 46, (DPB) + f0 * PBROW, PBROW, 2, 1);     \
    }
#define L23_ZPASS(K, SPB, PT)                                                  \
    for (int t = 0; t < NOFF23; ++t) {                                         \
        long f0 = SUF(off23)[t];                                               \
        L23_CALL(K, (SPB) + 2 * f0, PBROW, (PT) + f0 * 46, 46, 2, 1);          \
    }

/* ---- za (z-padded scratch) passes: t1 rows padded 23 -> 24 complex ------
 * (row stride 48 doubles, plane stride L23_T1PZ), so EVERY plane-phase Y
 * load from t1 is 64-byte aligned (in the flat 1058-layout, row stride 46,
 * ~3/4 of them split a cache line).  Cost: the X pass can no longer run the
 * 529 (y,z) points flat -- it goes row by row, 6 chunks per row via
 * zoff24, 138 chunks per volume instead of 133 (+1.2% volume FP).  Rows
 * y < 22 use the aligned zoff24 offsets; lane 23 of the last chunk then
 * reads input element (y+1, 0) -- a real, in-bounds value -- and writes the
 * pad column, which only ever flows into plane-buffer row 23, never read by
 * the Z pass.  Row y = 22 must not read past the volume, so it uses the
 * overlapping off23 offsets and never writes its pad element: t1[x][22][23]
 * stays whatever the buffer holds (zeroed at create; finite always).
 * Per-element arithmetic is identical to the flat variants (vector ops are
 * lanewise; only the lane position of an element changes), so bit-identity
 * with the class is expected -- and is cmp-VERIFIED, never assumed. */
#define NXZ23 (23 * NOFF23)
#define L23_XZ(K, VINP, T1B, i)                                                \
    do {                                                                       \
        int yz_ = (i) / NOFF23, tz_ = (i) % NOFF23;                            \
        long f0z_ = (yz_ == 22) ? (long)SUF(off23)[tz_]                        \
                                : (long)SUF(zoff24)[tz_];                      \
        L23_CALL(K, (VINP) + 2 * (23 * yz_ + f0z_), 1058,                      \
                 (T1B) + 48 * yz_ + 2 * f0z_, 2, L23_T1PZ, 0);                 \
    } while (0)
#define L23_YPASSZ(K, SRCP, DPB)                                               \
    for (int t = 0; t < NOFF23; ++t) {                                         \
        long f0 = SUF(zoff24)[t];                                              \
        L23_CALL(K, (SRCP) + 2 * f0, 48, (DPB) + f0 * PBROW, PBROW, 2, 1);     \
    }
/* Paced write-intent prefetch of the out plane the Z group is about to
 * store (panel-r5 VERDICT §4.5 via L17_matrixsimd r6, orig. L8_fusedaxes /
 * L36_pfa: hide the RFO with prefetchw, do not avoid it with NT stores).
 * 8464 B from a 16-mod-64 start spans 133 lines; issued in two half-bursts
 * around the Y group.  Runtime-gated (p->pw); changes no bits. */
#define L23_PW(VOUTX, Q0, Q1)                                                  \
    do {                                                                       \
        if (pw) {                                                              \
            const char *ob = (const char *)(VOUTX);                            \
            for (int q4 = (Q0); q4 < (Q1); ++q4)                               \
                __builtin_prefetch(ob + (long)q4 * 64, 1, 3);                  \
        }                                                                      \
    } while (0)

/* -----------------------------------------------------------------------
 *  Exec variants.
 *    KRN     : which chunk kernel (see above)
 *    REORD   : 0 = X-last (Y,Z per plane from `in`, then X into `out`)
 *              1 = X-first (X from `in` into t1, then Y,Z per plane into
 *                  `out`).  Since r7 X-first is the selected order at ALL
 *                  batch sizes (adopted from L23_rader r6: ~5% at B=1 on
 *                  wallaby -- the scattered X-pass stores then hit the hot
 *                  t1 scratch instead of a cold `out`); X-last variants are
 *                  timed for the record but never picked.
 *    NTC     : Z pass stages each plane in p->ps, streamed out via NT stores
 *    DZ      : deferred-Z plane schedule (adopted from L17_matrixsimd r6):
 *              plane buffer double-buffered, Y(x+1) runs between Y(x) and
 *              Z(x), so a Z group's loads never sit directly behind their
 *              own Y group's stores to the same L1 plane buffer.  Pure
 *              scheduling -- same chunks, same operands, same per-value
 *              order; bit-identity is cmp-verified, never assumed.
 *  t1 planes are padded to L23_T1P doubles (64-byte multiple) so the
 *  X-pass stores and the plane-phase loads of t1 never split cache lines
 *  (in the 1058-stride layout 3 of 4 zmm accesses did).
 * --------------------------------------------------------------------- */
#define L23_EXEC_G(NAME, KRN, REORD, NTC, DZ, ZA)                              \
    static __attribute__((unused)) void                                       \
    NAME(const fft3d_plan *restrict p, const double _Complex *restrict in,     \
         double _Complex *restrict out)                                        \
    {                                                                          \
        const double *restrict ct = CTAB(p);                                   \
        const double *restrict st = STAB(p);                                   \
        const double *restrict cd = CDTAB(p);                                  \
        const double *restrict sd = SDTAB(p);                                  \
        double *restrict pb = p->pb;                                           \
        double *restrict pb2 = p->pb2;                                         \
        double *restrict t1 = p->t1;                                           \
        double *restrict sc = p->sc;                                           \
        const int nb = p->batch;                                               \
        const long t1p = (ZA) ? L23_T1PZ : L23_T1P;                            \
        const int pw = (REORD && !NTC) ? p->pw : 0;                            \
        (void)ct; (void)st; (void)cd; (void)sd; (void)pb2; (void)pw;           \
        for (int b = 0; b < nb; ++b) {                                         \
            const double *vin = (const double *)(in + (size_t)b * 12167);      \
            double *vout = (double *)(out + (size_t)b * 12167);                \
            if (REORD) { /* X first: in -> t1[kx][y][z], padded planes */      \
                if (ZA) {                                                      \
                    for (int i = 0; i < NXZ23; ++i)                            \
                        L23_XZ(KRN, vin, t1, i);                               \
                } else {                                                       \
                    for (int i = 0; i < NX23; ++i) {                           \
                        long f0 = L23_XF0(i);                                  \
                        L23_CALL(KRN, vin + 2 * f0, 1058, t1 + 2 * f0,         \
                                 2, L23_T1P, 0);                               \
                    }                                                          \
                }                                                              \
            }                                                                  \
            if (DZ && REORD) { /* Y(0) */                                      \
                if (ZA) { L23_YPASSZ(KRN, (const double *)t1, pb) }            \
                else    { L23_YPASS(KRN, (const double *)t1, pb) }             \
            }                                                                  \
            for (int x = 0; x < 23; ++x) {                                     \
                double *pt = (NTC ? p->ps                                      \
                                  : (REORD ? vout + (long)x * 1058             \
                                           : t1 + (long)x * L23_T1P));         \
                if (REORD && p->pf && b + 1 < nb) {                            \
                    /* cross-volume input prefetch (L17_winograd r2 via       \
                     * L17_matrixsimd): during the plane phase, whose sources \
                     * are cache-resident scratch, pull the NEXT volume's     \
                     * input toward L2.  133 lines per plane covers the       \
                     * 3042-line volume in 23 planes.  Changes no bits. */    \
                    const char *nx = (const char *)(vin + 24334)               \
                                     + (long)x * 8464;                         \
                    for (int q3 = 0; q3 < 133; ++q3)                           \
                        __builtin_prefetch(nx + (long)q3 * 64, 0, 2);          \
                }                                                              \
                L23_PW(vout + (long)x * 1058, 0, 66);                          \
                if (DZ) {                                                      \
                    double *pbc = (x & 1) ? pb2 : pb; /* holds Y(x)   */       \
                    double *pbn = (x & 1) ? pb : pb2; /* gets  Y(x+1) */       \
                    if (x + 1 < 23) {                                          \
                        if (ZA) { L23_YPASSZ(KRN, (const double *)t1           \
                                          + (long)(x + 1) * t1p, pbn) }        \
                        else    { L23_YPASS(KRN, (const double *)t1            \
                                          + (long)(x + 1) * t1p, pbn) }        \
                    }                                                          \
                    L23_PW(vout + (long)x * 1058, 66, 133);                    \
                    L23_ZPASS(KRN, pbc, pt)                                         \
                } else {                                                       \
                    const double *pin2 = (REORD ? (const double *)t1           \
                                                    + (long)x * t1p            \
                                                : vin + (long)x * 1058);       \
                    if (ZA && REORD) { L23_YPASSZ(KRN, pin2, pb) }             \
                    else             { L23_YPASS(KRN, pin2, pb) }              \
                    L23_PW(vout + (long)x * 1058, 66, 133);                    \
                    L23_ZPASS(KRN, pb, pt)                                          \
                }                                                              \
                if (NTC) l23_ntcopy(vout + (long)x * 1058, p->ps, 1058);       \
            }                                                                  \
            if (!REORD) { /* X last: t1 -> out */                              \
                for (int i = 0; i < NX23; ++i) {                               \
                    long f0 = L23_XF0(i);                                      \
                    L23_CALL(KRN, t1 + 2 * f0, L23_T1P, vout + 2 * f0,         \
                             2, 1058, 0);                                      \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }
#define L23_EXEC(NAME, KRN, REORD) L23_EXEC_G(NAME, KRN, REORD, 0, 0, 0)

L23_EXEC(SUF(exec_u), 0, 0)
L23_EXEC(SUF(exec_b), 1, 0)
L23_EXEC(SUF(exec_bp), 2, 0)
L23_EXEC(SUF(exec_r), 3, 0)
L23_EXEC(SUF(exec_rp), 4, 0)
L23_EXEC(SUF(exec_uf), 0, 1)
L23_EXEC(SUF(exec_bf), 1, 1)
L23_EXEC(SUF(exec_bpf), 2, 1)
L23_EXEC(SUF(exec_rf), 3, 1)
L23_EXEC(SUF(exec_rpf), 4, 1)
L23_EXEC_G(SUF(exec_rfn), 3, 1, 1, 0, 0)
L23_EXEC_G(SUF(exec_rfd), 3, 1, 0, 1, 0)
L23_EXEC_G(SUF(exec_rfdn), 3, 1, 1, 1, 0)
L23_EXEC_G(SUF(exec_zf), 3, 1, 0, 0, 1)  /* za: pinned, X-first          */
L23_EXEC_G(SUF(exec_zfd), 3, 1, 0, 1, 1) /* za + deferred-Z              */
L23_EXEC_G(SUF(exec_zfn), 3, 1, 1, 0, 1) /* za + NT-staged planes        */
#undef L23_EXEC
#undef L23_EXEC_G

/* -----------------------------------------------------------------------
 *  Cross-volume software pipelining (adopted from L17_matrixsimd r4,
 *  exec18): X-first with t1 double-buffered; volume b+1's X chunks are
 *  interleaved, ~6 per plane, into volume b's plane phase, so the input
 *  read of every volume but the first overlaps the previous volume's
 *  compute.  Pure scheduling: every chunk computes exactly what it computes
 *  in the plain X-first variant, on the same operands, in the same
 *  per-value order -- bit-identity with exec_*f is VERIFIED BY cmp, never
 *  assumed (L17 r3's lesson: gcc may contract differently at a new site).
 * --------------------------------------------------------------------- */
#define L23_EXEC_PG(NAME, KRN, NTC, DZ, ZA)                                    \
    static __attribute__((unused)) void                                       \
    NAME(const fft3d_plan *restrict p, const double _Complex *restrict in,     \
         double _Complex *restrict out)                                        \
    {                                                                          \
        const double *restrict ct = CTAB(p);                                   \
        const double *restrict st = STAB(p);                                   \
        const double *restrict cd = CDTAB(p);                                  \
        const double *restrict sd = SDTAB(p);                                  \
        double *restrict pb = p->pb;                                           \
        double *restrict pb2 = p->pb2;                                         \
        double *restrict sc = p->sc;                                           \
        const int nb = p->batch;                                               \
        const long t1p = (ZA) ? L23_T1PZ : L23_T1P;                            \
        const int nxc = (ZA) ? NXZ23 : NX23;                                   \
        const int pw = NTC ? 0 : p->pw;                                        \
        (void)ct; (void)st; (void)cd; (void)sd; (void)pb2; (void)pw;           \
        { /* prologue: volume 0's X pass, the only un-overlapped read */      \
            const double *vin0 = (const double *)in;                           \
            for (int i = 0; i < nxc; ++i) {                                    \
                if (ZA) { L23_XZ(KRN, vin0, p->t1, i); }                       \
                else {                                                         \
                    long f0 = L23_XF0(i);                                      \
                    L23_CALL(KRN, vin0 + 2 * f0, 1058, p->t1 + 2 * f0,         \
                             2, L23_T1P, 0);                                   \
                }                                                              \
            }                                                                  \
        }                                                                      \
        for (int b = 0; b < nb; ++b) {                                         \
            double *vout = (double *)(out + (size_t)b * 12167);                \
            const double *restrict cur = (b & 1) ? p->t1b : p->t1;             \
            double *restrict nxt = (b & 1) ? p->t1 : p->t1b;                   \
            const double *vinN = (const double *)(in + (size_t)(b + 1) * 12167);\
            const int hn = (b + 1 < nb);                                       \
            if (DZ) { /* prologue Y(0) of this volume */                       \
                if (ZA) { L23_YPASSZ(KRN, cur, pb) }                           \
                else    { L23_YPASS(KRN, cur, pb) }                            \
            }                                                                  \
            for (int x = 0; x < 23; ++x) {                                     \
                /* ~6 X chunks of volume b+1 per plane, split across two      \
                 * insertion points so the DRAM read stream never bursts */   \
                int i0 = 0, ih = 0, i1 = 0;                                    \
                if (hn) {                                                      \
                    i0 = (x * nxc) / 23; i1 = ((x + 1) * nxc) / 23;            \
                    ih = i0 + (i1 - i0) / 2;                                   \
                    for (int i = i0; i < ih; ++i) {                            \
                        if (ZA) { L23_XZ(KRN, vinN, nxt, i); }                 \
                        else {                                                 \
                            long f0 = L23_XF0(i);                              \
                            L23_CALL(KRN, vinN + 2 * f0, 1058,                 \
                                     nxt + 2 * f0, 2, L23_T1P, 0);             \
                        }                                                      \
                    }                                                          \
                }                                                              \
                double *pt = NTC ? p->ps : vout + (long)x * 1058;              \
                L23_PW(vout + (long)x * 1058, 0, 66);                          \
                if (DZ) {                                                      \
                    double *pbn = (x & 1) ? pb : pb2;                          \
                    if (x + 1 < 23) {                                          \
                        if (ZA) { L23_YPASSZ(KRN, cur + (long)(x + 1) * t1p,   \
                                             pbn) }                            \
                        else    { L23_YPASS(KRN, cur + (long)(x + 1) * t1p,    \
                                            pbn) }                             \
                    }                                                          \
                } else {                                                       \
                    if (ZA) { L23_YPASSZ(KRN, cur + (long)x * t1p, pb) }       \
                    else    { L23_YPASS(KRN, cur + (long)x * t1p, pb) }        \
                }                                                              \
                if (hn) {                                                      \
                    for (int i = ih; i < i1; ++i) {                            \
                        if (ZA) { L23_XZ(KRN, vinN, nxt, i); }                 \
                        else {                                                 \
                            long f0 = L23_XF0(i);                              \
                            L23_CALL(KRN, vinN + 2 * f0, 1058,                 \
                                     nxt + 2 * f0, 2, L23_T1P, 0);             \
                        }                                                      \
                    }                                                          \
                }                                                              \
                L23_PW(vout + (long)x * 1058, 66, 133);                        \
                if (DZ) {                                                      \
                    double *pbc = (x & 1) ? pb2 : pb;                          \
                    L23_ZPASS(KRN, pbc, pt)                                         \
                } else {                                                       \
                    L23_ZPASS(KRN, pb, pt)                                          \
                }                                                              \
                if (NTC) l23_ntcopy(vout + (long)x * 1058, p->ps, 1058);       \
            }                                                                  \
        }                                                                      \
    }
#define L23_EXEC_P(NAME, KRN, NTC) L23_EXEC_PG(NAME, KRN, NTC, 0, 0)

L23_EXEC_P(SUF(exec_pu), 0, 0)
L23_EXEC_P(SUF(exec_pb), 1, 0)
L23_EXEC_P(SUF(exec_pbp), 2, 0)
L23_EXEC_P(SUF(exec_pr), 3, 0)
L23_EXEC_P(SUF(exec_prp), 4, 0)
L23_EXEC_P(SUF(exec_prn), 3, 1)
L23_EXEC_PG(SUF(exec_prd), 3, 0, 1, 0)
L23_EXEC_PG(SUF(exec_pz), 3, 0, 0, 1) /* za, pipelined */
#undef L23_EXEC_P
#undef L23_EXEC_PG

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
#undef CSTEP_C
#undef CSTEP_S
#undef CTAB
#undef STAB
#undef CDTAB
#undef SDTAB
#undef L23_STORE_TAIL
#undef TTILE
#undef TILE
#undef NOFF23
#undef PBROW
#undef L23_CALL
#undef NX23
#undef L23_XF0
#undef L23_YPASS
#undef L23_ZPASS
#undef L23_PW
#undef NXZ23
#undef L23_XZ
#undef L23_YPASSZ

#else /* ================= main body ================= */

/* Build-flag note (L45_pfa r7's -funroll-loops gap CHECKED HERE, does not
 * apply): a same-window alternating A/B on wallaby with node flags reads
 * unroll {21.91, 21.49, 22.23} vs no-unroll {21.73, 21.82, 21.17} us at
 * B=1 -- a wash, bit-identical outputs.  This entry's hot kernels are
 * hand-unrolled straight-line code and its rolled loops carry an
 * intentional "#pragma GCC unroll 1", so the flag has nothing to act on.
 * No pragma is needed; do not re-add one. */

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../fft3d_api.h"

/* Streaming copy of one finished 23x23 plane (1058 doubles) into the
 * caller's buffer.  Writing `out` directly costs a read-for-ownership of
 * every line; at streaming batch NT stores skip it (adopted from
 * L17_matrixsimd's l17_ntcopy, plane-wise as in its exec20).  Plane starts
 * are 16 (mod 64)-byte aligned (8464-byte plane stride), hence the step-up
 * loop.  NT stores write the same bits, so this stays in the bit class. */
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
#  include <string.h>
static void l23_ntcopy(double *restrict dst, const double *restrict src, size_t nd)
{
    memcpy(dst, src, nd * sizeof *dst);
}
#endif

/* t1 plane stride in doubles: 529 complex = 1058 doubles padded to 1064
 * (8512 B = 133 whole cache lines), so X-pass stores into t1 and plane-phase
 * loads of t1 never split a line (at stride 1058, 3 of 4 zmm accesses did). */
#define L23_T1P 1064
/* za layout: t1 rows padded 23 -> 24 complex (48 doubles), plane = 23 rows
 * = 1104 doubles = 8832 B = 138 whole lines; mod 4096 = 640, decorrelated
 * from the driver buffers' 8464-byte stride (272 mod 4096). */
#define L23_T1PZ 1104

struct fft3d_plan {
    int L, batch;
    int pf; /* cross-volume input prefetch in the X-first plane phase (A/B'd) */
    int pw; /* paced prefetchw on the out plane ahead of the Z stores (A/B'd) */
    void (*exec)(const struct fft3d_plan *, const double _Complex *, double _Complex *);
    double *ctab8; /* 11 rows x 12 cosines, j-major, each splatted 8x (zmm) */
    double *stab8; /* 11 rows x 11 sines,   j-major, each splatted 8x (zmm) */
    double *ctab4; /* the same, splatted 4x (ymm) */
    double *stab4;
    double *cdist8; /* the 11 DISTINCT cosines cos(2pi m/23), splatted 8x */
    double *sdist8; /* the 11 DISTINCT sines   sin(2pi m/23), splatted 8x */
    double *cdist4; /* the same, splatted 4x */
    double *sdist4;
    double *sc;  /* 22 vectors: parked butterflies for the bst=1 kernel      */
    double *pb;  /* 23 x 24 complex plane buffer (pass Y -> pass Z)          */
    double *pb2; /* second plane buffer for the deferred-Z variants          */
    double *t1;  /* one 23^3 complex volume (padded planes): pass Z <-> X    */
    double *t1b; /* second volume buffer for the pipelined variants          */
    double *ps;  /* one 23x23 complex plane: NT staging (pass Z -> stream out) */
    void *block;
    double _Complex *ti, *to; /* transient buffers for the plan-time tuner   */
    size_t tn;
};

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

/* -----------------------------------------------------------------------
 * Sustained-clock probe (pattern from L17_matrixsimd r5, kept for the new
 * geometry's first node measurement): four independent 4-cycle FMA chains
 * issue 1 FMA/cycle, so clk = 4*N / elapsed at either width.
 * --------------------------------------------------------------------- */
static volatile double l23_clk_sink;
static double l23_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
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
    p->tn = 0;
}

/* Deterministic pseudo-random tuning data (realistic magnitudes suffice). */
static int l23_tune_alloc(fft3d_plan *p, int nv)
{
    size_t n = (size_t)nv * 12167;
    if (p->tn >= n) return 1;
    l23_tune_free(p);
    if (posix_memalign((void **)&p->ti, 64, n * sizeof *p->ti) != 0) { p->ti = NULL; return 0; }
    if (posix_memalign((void **)&p->to, 64, n * sizeof *p->to) != 0) {
        free(p->ti); p->ti = NULL; p->to = NULL; return 0;
    }
    p->tn = n;
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

const char *fft3d_name(void) { return "L23_matrixsimd"; }

static const char *g_desc =
    "dense 23x23 DFT matrix per axis, conjugate-pair folded, SIMD across lines";

const char *fft3d_description(void) { return g_desc; }

int fft3d_supports(int L) { return L == 23; }

static int l23_verbose(void)
{
    const char *e = getenv("L23_VERBOSE");
    return e && *e && *e != '0';
}

/* Streaming-regime tuner arena: enough volumes that in + out together exceed
 * ~2.5x this machine's L3 (machine-relative, adopted from L36_mixedradix via
 * L17_matrixsimd r3, with the same clamps rescaled for the 190 KiB volume). */
static int l23_tune_nv(int batch)
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

    const size_t nc = 11 * 12, ns = 11 * 11, nd = 11;
    const size_t nsc = 22 * 8;
    const size_t npb = 24 * 24 * 2; /* 24 rows: za's Y pass writes a 24th (pad) row */
    const size_t nt1 = 23 * L23_T1PZ; /* the larger (za) plane stride, see L23_T1PZ */
    const size_t nps = 1058 + 6;
    const size_t ntot = 12 * nc + 12 * ns + 24 * nd + nsc + 2 * npb + 2 * nt1 + nps + 1024;
    void *blk = NULL;
    if (posix_memalign(&blk, 64, ntot * sizeof(double)) != 0 || !blk) {
        free(p);
        return NULL;
    }
    memset(blk, 0, ntot * sizeof(double));
    p->block = blk;

#define L23_ALIGN64(q) ((double *)(((uintptr_t)(q) + 63u) & ~(uintptr_t)63u))
    double *q = (double *)blk;
    p->ctab8 = q; q += 8 * nc;
    p->stab8 = q; q += 8 * ns;
    q = L23_ALIGN64(q);
    p->ctab4 = q; q += 4 * nc;
    p->stab4 = q; q += 4 * ns;
    q = L23_ALIGN64(q);
    p->cdist8 = q; q += 8 * nd;
    p->sdist8 = q; q += 8 * nd;
    p->cdist4 = q; q += 4 * nd;
    p->sdist4 = q; q += 4 * nd;
    q = L23_ALIGN64(q);
    p->sc = q; q += nsc;
    q = L23_ALIGN64(q);
    p->pb = q; q += npb;
    q = L23_ALIGN64(q);
    p->pb2 = q; q += npb;
    q = L23_ALIGN64(q);
    p->t1 = q; q += nt1;
    q = L23_ALIGN64(q);
    p->t1b = q; q += nt1;
    q = L23_ALIGN64(q);
    p->ps = q;
#undef L23_ALIGN64

    /* long double trig: the table itself is then good to ~1e-19 */
    const long double twopi = 6.283185307179586476925286766559005768394L;
    for (int j = 1; j <= 11; ++j) {
        for (int k = 0; k <= 11; ++k) {
            int m = (k * j) % 23;
            double c = (double)cosl(twopi * (long double)m / 23.0L);
            for (int t = 0; t < 8; ++t) p->ctab8[((j - 1) * 12 + k) * 8 + t] = c;
            for (int t = 0; t < 4; ++t) p->ctab4[((j - 1) * 12 + k) * 4 + t] = c;
        }
        for (int k = 1; k <= 11; ++k) {
            int m = (k * j) % 23;
            double sv = (double)sinl(twopi * (long double)m / 23.0L);
            for (int t = 0; t < 8; ++t) p->stab8[((j - 1) * 11 + (k - 1)) * 8 + t] = sv;
            for (int t = 0; t < 4; ++t) p->stab4[((j - 1) * 11 + (k - 1)) * 4 + t] = sv;
        }
    }
    for (int m = 1; m <= 11; ++m) { /* the 22 distinct constants, splatted */
        double c = (double)cosl(twopi * (long double)m / 23.0L);
        double sv = (double)sinl(twopi * (long double)m / 23.0L);
        for (int t = 0; t < 8; ++t) p->cdist8[(m - 1) * 8 + t] = c;
        for (int t = 0; t < 8; ++t) p->sdist8[(m - 1) * 8 + t] = sv;
        for (int t = 0; t < 4; ++t) p->cdist4[(m - 1) * 4 + t] = c;
        for (int t = 0; t < 4; ++t) p->sdist4[(m - 1) * 4 + t] = sv;
    }

    /* ---- pick the variant by measuring, on this machine, within the batch
     * regime's bit class (see the header block for why the class is a pure
     * function of the batch size). ---- */
    {
        typedef void (*l23_fn)(const fft3d_plan *, const double _Complex *,
                               double _Complex *);
        enum { L23_NCAND = 48 };
        static const l23_fn cand[L23_NCAND] = {
            /* 0..4:  512-bit X-last  (unblocked, kblk, kblk+L1, pinned, pinned+park) */
            exec_u_w4, exec_b_w4, exec_bp_w4, exec_r_w4, exec_rp_w4,
            /* 5..9:  512-bit X-first */
            exec_uf_w4, exec_bf_w4, exec_bpf_w4, exec_rf_w4, exec_rpf_w4,
            /* 10..14: 512-bit X-first pipelined */
            exec_pu_w4, exec_pb_w4, exec_pbp_w4, exec_pr_w4, exec_prp_w4,
            /* 15..29: the same, 256-bit */
            exec_u_w2, exec_b_w2, exec_bp_w2, exec_r_w2, exec_rp_w2,
            exec_uf_w2, exec_bf_w2, exec_bpf_w2, exec_rf_w2, exec_rpf_w2,
            exec_pu_w2, exec_pb_w2, exec_pbp_w2, exec_pr_w2, exec_prp_w2,
            /* 30..33: NT-staged planes, pinned kernel, X-first {plain,pipe} x width */
            exec_rfn_w4, exec_prn_w4, exec_rfn_w2, exec_prn_w2,
            /* 34..39: deferred-Z (L17_matrixsimd r6), pinned, X-first:
             *         {plain, pipelined, NT} x width */
            exec_rfd_w4, exec_prd_w4, exec_rfdn_w4,
            exec_rfd_w2, exec_prd_w2, exec_rfdn_w2,
            /* 40..47: za (z-padded scratch: aligned Y loads, +5 X chunks),
             *         pinned, X-first: {plain, pipelined, dz, NT} x width */
            exec_zf_w4, exec_pz_w4, exec_zfd_w4, exec_zfn_w4,
            exec_zf_w2, exec_pz_w2, exec_zfd_w2, exec_zfn_w2,
        };
        static const char *const tags[L23_NCAND] = {
            "dense 23x23/axis conj-folded, 512-bit, unblocked",
            "dense 23x23/axis conj-folded, 512-bit, k-blocked",
            "dense 23x23/axis conj-folded, 512-bit, k-blocked+L1 bfly",
            "dense 23x23/axis conj-folded, 512-bit, pinned 22 consts",
            "dense 23x23/axis conj-folded, 512-bit, pinned, P parked",
            "dense 23x23/axis conj-folded, 512-bit, unblocked, X-first",
            "dense 23x23/axis conj-folded, 512-bit, k-blocked, X-first",
            "dense 23x23/axis conj-folded, 512-bit, k-blocked+L1 bfly, X-first",
            "dense 23x23/axis conj-folded, 512-bit, pinned, X-first",
            "dense 23x23/axis conj-folded, 512-bit, pinned+park, X-first",
            "dense 23x23/axis conj-folded, 512-bit, unblocked, X-first, pipelined",
            "dense 23x23/axis conj-folded, 512-bit, k-blocked, X-first, pipelined",
            "dense 23x23/axis conj-folded, 512-bit, k-blocked+L1, X-first, pipelined",
            "dense 23x23/axis conj-folded, 512-bit, pinned, X-first, pipelined",
            "dense 23x23/axis conj-folded, 512-bit, pinned+park, X-first, pipelined",
            "dense 23x23/axis conj-folded, 256-bit, unblocked",
            "dense 23x23/axis conj-folded, 256-bit, k-blocked",
            "dense 23x23/axis conj-folded, 256-bit, k-blocked+L1 bfly",
            "dense 23x23/axis conj-folded, 256-bit, pinned 22 consts",
            "dense 23x23/axis conj-folded, 256-bit, pinned, P parked",
            "dense 23x23/axis conj-folded, 256-bit, unblocked, X-first",
            "dense 23x23/axis conj-folded, 256-bit, k-blocked, X-first",
            "dense 23x23/axis conj-folded, 256-bit, k-blocked+L1 bfly, X-first",
            "dense 23x23/axis conj-folded, 256-bit, pinned, X-first",
            "dense 23x23/axis conj-folded, 256-bit, pinned+park, X-first",
            "dense 23x23/axis conj-folded, 256-bit, unblocked, X-first, pipelined",
            "dense 23x23/axis conj-folded, 256-bit, k-blocked, X-first, pipelined",
            "dense 23x23/axis conj-folded, 256-bit, k-blocked+L1, X-first, pipelined",
            "dense 23x23/axis conj-folded, 256-bit, pinned, X-first, pipelined",
            "dense 23x23/axis conj-folded, 256-bit, pinned+park, X-first, pipelined",
            "dense 23x23/axis conj-folded, 512-bit, pinned, X-first, NT planes",
            "dense 23x23/axis conj-folded, 512-bit, pinned, X-first, pipelined, NT planes",
            "dense 23x23/axis conj-folded, 256-bit, pinned, X-first, NT planes",
            "dense 23x23/axis conj-folded, 256-bit, pinned, X-first, pipelined, NT planes",
            "dense 23x23/axis conj-folded, 512-bit, pinned, X-first, deferred-Z",
            "dense 23x23/axis conj-folded, 512-bit, pinned, X-first, pipelined, deferred-Z",
            "dense 23x23/axis conj-folded, 512-bit, pinned, X-first, deferred-Z, NT planes",
            "dense 23x23/axis conj-folded, 256-bit, pinned, X-first, deferred-Z",
            "dense 23x23/axis conj-folded, 256-bit, pinned, X-first, pipelined, deferred-Z",
            "dense 23x23/axis conj-folded, 256-bit, pinned, X-first, deferred-Z, NT planes",
            "dense 23x23/axis conj-folded, 512-bit, pinned, X-first, za",
            "dense 23x23/axis conj-folded, 512-bit, pinned, X-first, pipelined, za",
            "dense 23x23/axis conj-folded, 512-bit, pinned, X-first, deferred-Z, za",
            "dense 23x23/axis conj-folded, 512-bit, pinned, X-first, za, NT planes",
            "dense 23x23/axis conj-folded, 256-bit, pinned, X-first, za",
            "dense 23x23/axis conj-folded, 256-bit, pinned, X-first, pipelined, za",
            "dense 23x23/axis conj-folded, 256-bit, pinned, X-first, deferred-Z, za",
            "dense 23x23/axis conj-folded, 256-bit, pinned, X-first, za, NT planes",
        };
        /* ONE bit class since r7: pinned X-first, all batch sizes (adopted
         * from L23_rader r6, which measured X-first winning B=1 too and
         * cmp-verified its twelve X-first variants mutually bit-identical).
         * The PINNED kernel is NOT bit-identical to the table kernels (gcc
         * contracts the two differently -- L17_matrixsimd r3's failure), and
         * X-last reassociates the triple sum, so table kernels and X-last
         * variants exist for forced experiments but are never timed by the
         * tuner and never picked.  Within the class: width, P-parking,
         * pipelining, NT staging, deferred-Z and za are cmp-verified
         * bit-identical on wallaby (see the strategy record).
         *
         * The list is in CANONICAL ORDER and the pick is HYSTERETIC (r8):
         * walking the list, a candidate displaces the incumbent only if it
         * is >2% faster.  r7's lesson (verdict §3a): min-of-noise across 18
         * near-tied variants picked differently in different processes, so
         * every scored L=23 cell reported a variant the harness never
         * checked.  With the margin, near-ties resolve to the same canonical
         * variant in every process; a real mechanism (>2%) still wins. */
        static const int selC[26] = {
            8, 13, 9, 14, 34, 35, 40, 41, 42, /* 512-bit cached family     */
            30, 31, 36, 43,                   /* 512-bit NT family         */
            23, 28, 24, 29, 37, 38, 44, 45,   /* 256-bit cached family     */
            32, 33, 39, 46, 47                /* 256-bit NT family         */
        };
        const int *sel = selC;
        const int nsel = 26;
        int bestv = sel[0];
        p->exec = cand[bestv];
        g_desc = tags[bestv];
        double best[L23_NCAND];
        for (int v = 0; v < L23_NCAND; ++v) best[v] = 1e30;

        /* stage 1 (batch < 64): blocked timing, small cache-resident set.
         * Candidates are never interleaved (a 512/256-bit licence transition
         * per sample cost 35% and mis-ranked candidates on the node --
         * L17_matrixsimd r1).  TWO full fixed-order sweeps with a
         * per-candidate min across both (adopted from L17_rader r6): one
         * sweep on a powersave governor can drift monotonically down the
         * table and mis-rank candidates far apart in it. */
        if (batch < 64) {
            int nv = batch < 8 ? batch : 8;
            int inner = (32 + nv - 1) / nv;
            if (l23_tune_alloc(p, nv)) {
                int sb = p->batch;
                p->batch = nv;
                for (int sweep = 0; sweep < 2; ++sweep)
                    for (int s = 0; s < nsel; ++s) {
                        const int v = sel[s];
                        /* per-candidate licence warmup >= 1.5 ms (adopted
                         * from L17_rader r6): a ymm candidate timed inside
                         * another's not-yet-decayed AVX-512 licence window
                         * (~670 us) can never show a licence-clock win. */
                        { double tw = l23_now();
                          do { cand[v](p, p->ti, p->to); }
                          while (l23_now() - tw < 1.5e-3); }
                        for (int r = 0; r < 3; ++r) {
                            double t0 = l23_now();
                            for (int q2 = 0; q2 < inner; ++q2)
                                cand[v](p, p->ti, p->to);
                            double dt = l23_now() - t0;
                            if (dt < best[v]) best[v] = dt;
                        }
                    }
                p->batch = sb;
                for (int s = 1; s < nsel; ++s)
                    if (best[sel[s]] < 0.98 * best[bestv]) bestv = sel[s];
                p->exec = cand[bestv];
                g_desc = tags[bestv];
                if (l23_verbose())
                    for (int s = 0; s < nsel; ++s)
                        fprintf(stderr, "[L23_matrixsimd tune] %-70s %8.2f us/transform%s\n",
                                tags[sel[s]], best[sel[s]] * 1e6 / (nv * inner),
                                sel[s] == bestv ? "  <== kept" : "");
            }
        }

        /* stage 1b (batch >= 64): tune on a working set that actually leaves
         * L3 (L17_winograd r2's lesson: the streaming regime can prefer a
         * different kernel outright; trusting the small-set pick cost 90%).
         * Same two-sweep + hysteresis discipline as stage 1. */
        if (batch >= 64) {
            int nv2 = l23_tune_nv(batch);
            if (l23_tune_alloc(p, nv2)) {
                int sb = p->batch;
                p->batch = nv2;
                for (int sweep = 0; sweep < 2; ++sweep)
                    for (int s = 0; s < nsel; ++s) {
                        const int v = sel[s];
                        /* licence warmup, as in stage 1 (one streaming exec
                         * is already >> 1.5 ms; the loop runs at least once) */
                        { double tw = l23_now();
                          do { cand[v](p, p->ti, p->to); }
                          while (l23_now() - tw < 1.5e-3); }
                        for (int r = 0; r < 3; ++r) {
                            double t0 = l23_now();
                            cand[v](p, p->ti, p->to);
                            double dt = l23_now() - t0;
                            if (dt < best[v]) best[v] = dt;
                        }
                    }
                p->batch = sb;
                for (int s = 1; s < nsel; ++s)
                    if (best[sel[s]] < 0.98 * best[bestv]) bestv = sel[s];
                p->exec = cand[bestv];
                g_desc = tags[bestv];
                if (l23_verbose())
                    for (int s = 0; s < nsel; ++s)
                        fprintf(stderr, "[L23_matrixsimd tune >L3 nv=%d] %-70s %8.2f us/transform%s\n",
                                nv2, tags[sel[s]], best[sel[s]] * 1e6 / nv2,
                                sel[s] == bestv ? "  <== kept" : "");
            }
        }

        /* stage 2: prefetch knobs on the final pick, judged JOINTLY as a
         * (pf, pw) grid (interacting knobs need a grid -- L17_matrixsimd r6
         * carrying its r4 lesson), with a 3% margin against (0,0): prefetch
         * on cache-resident lines is pure uop tax (L36_pfa measured +13%),
         * so a prefetch is never defaulted on.  Prefetches change no bits.
         *   pf: cross-volume input prefetch; hook only in non-pipelined
         *       X-first variants, and only meaningful at batch >= 64.
         *   pw: prefetchw on the out plane ahead of the Z stores; hook in
         *       all non-NT X-first variants (panel-r5 VERDICT: the node
         *       hides the RFO with prefetchw; NT stores lost 4 rounds). */
        {
            const int pipewin = (bestv >= 10 && bestv <= 14) ||
                                (bestv >= 25 && bestv <= 29) ||
                                bestv == 31 || bestv == 33 ||
                                bestv == 35 || bestv == 38 ||
                                bestv == 41 || bestv == 45;
            const int ntwin = bestv == 30 || bestv == 31 || bestv == 32 ||
                              bestv == 33 || bestv == 36 || bestv == 39 ||
                              bestv == 43 || bestv == 47;
            const int haspf = (batch >= 64) && !pipewin;
            const int haspw = !ntwin;
            p->pf = 0;
            p->pw = 0;
            int nv2 = (batch >= 64) ? l23_tune_nv(batch)
                                    : (batch < 8 ? batch : 8);
            if ((haspf || haspw) && l23_tune_alloc(p, nv2)) {
                int inner = (batch >= 64) ? 1 : (32 + nv2 - 1) / nv2;
                int sb = p->batch;
                p->batch = nv2;
                double tt[2][2] = {{1e30, 1e30}, {1e30, 1e30}};
                for (int sweep = 0; sweep < 2; ++sweep) /* two-sweep, as stage 1 */
                    for (int fv = 0; fv <= haspf; ++fv)
                        for (int wv = 0; wv <= haspw; ++wv) {
                            p->pf = fv;
                            p->pw = wv;
                            { double tw = l23_now();
                              do { p->exec(p, p->ti, p->to); }
                              while (l23_now() - tw < 1.5e-3); }
                            for (int r = 0; r < 3; ++r) {
                                double t0 = l23_now();
                                for (int q2 = 0; q2 < inner; ++q2)
                                    p->exec(p, p->ti, p->to);
                                double dt = l23_now() - t0;
                                if (dt < tt[fv][wv]) tt[fv][wv] = dt;
                            }
                        }
                p->batch = sb;
                int bf = 0, bw = 0;
                for (int fv = 0; fv <= haspf; ++fv)
                    for (int wv = 0; wv <= haspw; ++wv)
                        if (tt[fv][wv] < tt[bf][bw]) { bf = fv; bw = wv; }
                if (!(tt[bf][bw] < 0.97 * tt[0][0])) { bf = 0; bw = 0; }
                p->pf = bf;
                p->pw = bw;
                if (l23_verbose())
                    fprintf(stderr,
                            "[L23_matrixsimd tune] nv=%d (pf,pw): 00 %.2f  01 %.2f  10 %.2f  11 %.2f us/t -> pf=%d pw=%d\n",
                            nv2, tt[0][0] * 1e6 / (nv2 * inner),
                            tt[0][1] * 1e6 / (nv2 * inner),
                            tt[1][0] * 1e6 / (nv2 * inner),
                            tt[1][1] * 1e6 / (nv2 * inner), p->pf, p->pw);
            }
            static char g_desc_buf[192];
            snprintf(g_desc_buf, sizeof g_desc_buf, "%s, pf=%d, pw=%d",
                     g_desc, p->pf, p->pw);
            g_desc = g_desc_buf;
        }
        l23_tune_free(p);
#if defined(L23_FORCE)
        /* development override: -DL23_FORCE=0..47 pins one variant */
        p->exec = cand[L23_FORCE];
        g_desc = tags[L23_FORCE];
        p->pf = 0;
        p->pw = 0;
#endif
    }
    { /* measured sustained licence clocks, carried back via the description */
        double c512 = l23_clk512(), c256 = l23_clk256();
        static char g_desc_clk[224];
        snprintf(g_desc_clk, sizeof g_desc_clk, "%s, clk512/256=%.2f/%.2f GHz",
                 g_desc, c512 * 1e-9, c256 * 1e-9);
        g_desc = g_desc_clk;
    }
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    l23_tune_free(p);
    free(p->block);
    free(p);
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    plan->exec(plan, in, out);
}

#endif /* L23_TEMPLATE_PASS */
