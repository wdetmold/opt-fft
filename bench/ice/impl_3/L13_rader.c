/* =============================================================================
 * L13_rader -- 13^3 complex-double forward DFT, batched, single-threaded.
 *
 * ICE ROUND ice_r2: STRUCTURAL REBUILD.  The ice_r1 leaderboard read
 * L13_direct 4.661 us/xform against this entry's 6.131 on the graded chain --
 * a structural gap, not a knob (my split-re/im pipeline pays two full
 * de/re-interleave passes plus zmm transpose networks that their
 * interleaved-lanes pipeline simply does not have).  So this round the entry
 * ADOPTS L13_direct's pipeline WHOLESALE (attribution: L13_direct panel_r6..
 * r11 via ice exemplars -- lanes = whole lines in the driver's interleaved
 * layout, 128-bit tile transposes fused into stores, zsolid Y groups, xmm
 * tails, the T1P=344 anti-split/anti-alias pad, X-first order, the streaming
 * prefetch schedule, and the all-pinned-constant kernel discipline) and keeps
 * only this entry's own contribution: the RADER/CRT ARITHMETIC of the
 * 13-point kernel, which needs 93 vector FP ops per chunk against the dense
 * conjugate-folded matvec's 102 (-8.8%).
 *
 * KERNEL MATH (per 13-point line; this entry's lineage, re-expressed for
 * interleaved lanes).  g = 2 generates (Z/13)*; quotient by {+-1} gives the
 * order-6 orbit g^t mod 13 = [1,2,4,8,3,6].  Fold conjugate pairs first:
 *     u_t = x[g^t] + x[13-g^t],   v_t = x[g^t] - x[13-g^t]      (t = 0..5)
 * The cosine side is a length-6 cyclic correlation of u, split once more
 * (Z2 x Z3) into an x0-seeded cyclic-3 on P_t = u_t + u_{t+3} plus a
 * negacyclic-3 on Q_t = u_t - u_{t+3}, both with HALVED constants:
 *     CP_t = (cos(2pi g^t/13) + cos(2pi g^(t+3)/13)) / 2
 *     CM_t = (cos(2pi g^t/13) - cos(2pi g^(t+3)/13)) / 2
 *     a_n  = x0 + sum_t P_t CP_{(n+t) mod 3},  b_n = sum_t +-Q_t CM_{..}
 *     cc_n = a_n + b_n,  cc_{n+3} = a_n - b_n                    (n = 0..2)
 * The sine side is a dense length-6 NEGACYCLIC correlation of v; with
 * interleaved lanes the required *i is a re/im swap with the sign folded
 * into LANE-ALTERNATING sine splats (L13_direct panel_r9's MULI sign-fold,
 * generalized to Rader order):
 *     w_t  = SWAPRI(v_t),   S_t = splat(+s_t, -s_t, ...),  s_t = sin(2pi g^t/13)
 *     T_n  = i * SS_n = sum_t sigma(n+t) S_{(n+t) mod 6} (.) w_t
 * and the outputs are plain adds:  X[0] = x0 + sum u,
 *     X[g^n] = cc_n + T_n,   X[13-g^n] = cc_n - T_n.
 *
 * OPERATION COUNT per chunk (WC lines): 12 fold add/sub + 6 swaps + 6 P/Q
 * add/sub + 3 dc adds + 9 CP FMA + 9 CM FMA + 6 cc add/sub + 36 sine FMA +
 * 12 output add/sub = 93 vector FP + 6 swaps, vs L13_direct's 102 + 6.
 * Per volume (their census, unchanged): X 42 zmm + 1 xmm, per plane Y 4 zmm
 * (zsolid) + Z 3 zmm + 1 xmm = 133 zmm + 14 xmm chunks; at 93 FP that is
 * 12,369 + 651 zmm-op-equivalents.  On this node's TWO 512-bit FMA pipes
 * (p0+p5) the port floor is ~6.5k cycles/volume; the graded chain is
 * latency/L3-bound well above it (ice_r1), so the bet is on the pipeline's
 * memory behaviour, which is exactly what L13_direct already proved: their
 * in-plan number 3947 ns/vol vs this entry's 5455 on the same node.
 *
 * Constants: 6 broadcast (CP0-2, CM0-2) + 6 lane-alternating sine splats,
 * all 12 pinned in registers across each execute (L13_direct r6: "below ~15
 * distinct constants, pin everything"); 13 loads per chunk, ZERO constant
 * loads in the loop.  Liveness peaks ~30 of 32 EVEX registers.
 *
 * PASSES (L13_direct's, verbatim; X-first everywhere per their r10 retest):
 *     X : in[x][p]    --lanes over p (169 contiguous)--> t1[kx][p]   (plain)
 *     Y : t1[kx][y][z]--lanes over z --> pb[z][ky]     (transposing store)
 *     Z : pb[z][ky]   --lanes over ky--> out[kx][ky][kz]  (transposing store)
 * Y groups are ZSOLID (4 zmm chunks at 0,4,8,9: pb written 100% as 64 B
 * tiles, zero store-forward blocks at the pb junction); Z and X tails are
 * xmm (WC=1) chunks (16 B ops never split a line; their narrow-store output
 * is never wide-loaded hot).  Their FORCE=9 lesson holds: do NOT put xmm
 * tails in the Y position.  t1 planes at T1P=344 doubles (43 lines, 64 B
 * aligned rows, 4K-residue decorrelated from in/out's 338).
 *
 * SELECTION is deterministic (sysconf L3): ws = 32*B*2197 <= L3 -> plain
 * X-first; past L3 -> X-first + their r8 prefetch schedule (pfw the next out
 * plane ahead of the Z stores + paced read-prefetch of the next volume's
 * input; -DL13R_PW=0 / -DL13R_PFIN=0 kill either half).  On top of that,
 * create() runs this entry's established IN-PLAN TIMED RACE (pattern
 * L13_direct r10 ab <- L6_unrolled r9 <- L36_pfa r8; incumbency rule from
 * L6_pfa: adopt a challenger only if it beats the incumbent by >3% in BOTH
 * of two independent trial blocks) over three OUTPUT-BIT-IDENTICAL X-first
 * shapes: zs (default), y2 (ymm tails), p7 (pure zmm zsolid everywhere --
 * the natural two-FMA-pipe shape).  X-last is priced as a 4th arm but NEVER
 * adopted (different pass order = different rounding).  Readings ship in
 * fft3d_description() and on stderr.  -DL13R_AB=0 removes the race,
 * -DL13R_FORCE=n pins a variant.
 *
 * Everything is gcc/clang generic vector extensions, template-instantiated
 * at WC = 4/2/1 complex per vector by self-#include (L13_direct's scheme);
 * the WC=2 execs are the AVX2 fallback path.
 *
 * ICE ROUND ice_r3: adopts L13_direct ice_r2's two wins on top of this
 * entry's cheaper kernel (attribution: L13_direct ice_r2, itself crediting
 * L17_rader ice_r1's "sp" overlap and L17_matrixsimd ice_r1's chain arena):
 *   1. OV -- cross-volume X-pass overlap, the new ws<=L3 default: the chain
 *      regime's only L3-cold accesses are the X pass's reads of `in`; OV
 *      interleaves the NEXT volume's 43 X chunks into the current volume's
 *      L1-hot plane phase, floor(43(x+1)/13) schedule (3-4 per plane,
 *      between the Y and Z groups), storing into a second t1 buffer t1b
 *      that ping-pongs with t1.  Volume 0's X pass runs un-overlapped up
 *      front; at nb=1 the schedule is identical to zs.  Same chunks, same
 *      per-volume out-store order => bit-identical output.
 *   2. Chain-shaped ADOPTING race replacing ice_r2's fixed src->dst race
 *      (that regime was ~L2-resident and optimistic): tb=min(batch,32)
 *      volumes ping-pong two private buffers with an untimed driver-style
 *      unitary scale pass (x 2197^-1/2) between steps, preceded by a
 *      ~120 ms dense-FMA clock-settle spin (L17_winograd's schedutil
 *      lesson, via L13_direct).  Arms ov(incumbent)/zs/y2/t1, all
 *      output-bit-identical; challenger adopted only if >1.5% better in
 *      BOTH of two independent blocks (L6_pfa hysteresis, tighter margin
 *      now that the race regime is honest).  p7/xl retired from the race
 *      (priced twice, documented in the record); still reachable by FORCE.
 * Contract: ../fft3d_api.h.  Strategy record: ../strategies/L13_rader.md.
 * =============================================================================
 */

#ifdef L13R_TEMPLATE_PASS
/* ===========================================================================
 *  TEMPLATE BODY -- instantiated once per vector width.
 *  Inputs: WC (complex per vector), SUF(x) (name mangler).
 * ===========================================================================
 */
#define VDW (2 * WC) /* doubles per vector */

typedef double SUF(vd) __attribute__((vector_size(8 * VDW), aligned(8)));
typedef long long SUF(vi) __attribute__((vector_size(8 * VDW)));
typedef double SUF(v2) __attribute__((vector_size(16), aligned(8)));
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

/* re/im swap within each complex lane; the *i sign lives in the sine tables
 * (lane-alternating +s,-s -- L13_direct panel_r9's fold, adopted). */
#if WC == 4
#  define SWAPRI(x) SHUF1(x, 1, 0, 3, 2, 5, 4, 7, 6)
#elif WC == 2
#  define SWAPRI(x) SHUF1(x, 1, 0, 3, 2)
#else
#  define SWAPRI(x) SHUF1(x, 1, 0)
#endif

/* Pin the 12 kernel constants in registers across a whole execute (empty asm
 * = opaque to gcc).  Only on a 32-register EVEX file. */
#if (WC == 4 && defined(__AVX512F__)) || (WC == 2 && defined(__AVX512VL__))
#  define L13R_PIN12_ASM() __asm__("" : "+v"(C0), "+v"(C1), "+v"(C2), "+v"(C3),\
                                        "+v"(C4), "+v"(C5), "+v"(S0), "+v"(S1),\
                                        "+v"(S2), "+v"(S3), "+v"(S4), "+v"(S5))
#else
#  define L13R_PIN12_ASM() do { } while (0)
#endif

#define CGET(base, i) VLD((base) + (size_t)(i) * VDW)
#if WC == 4
#  define CPD(p) ((p)->cpd8)
#  define SNT(p) ((p)->snt8)
#else
#  define CPD(p) ((p)->cpd4)
#  define SNT(p) ((p)->snt4)
#endif

/* ---- WCxWC transpose of complex (128-bit blocks); L13_direct verbatim ---- */
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
#elif WC == 2
#  define TTILE(x, y)                                                          \
      do {                                                                     \
          (y)[0] = SHUF2((x)[0], (x)[1], 0, 1, 4, 5);                          \
          (y)[1] = SHUF2((x)[0], (x)[1], 2, 3, 6, 7);                          \
      } while (0)
#endif

/* Output expressions in row order m = 0..12 (Rader output permutation:
 * rows (g^n, 13-g^n) = cc_n +- T_n; see the header derivation). */
#define X0_  dc
#define X1_  (cc0 + T0)
#define X2_  (cc1 + T1)
#define X3_  (cc4 + T4)
#define X4_  (cc2 + T2)
#define X5_  (cc3 - T3)
#define X6_  (cc5 + T5)
#define X7_  (cc5 - T5)
#define X8_  (cc3 + T3)
#define X9_  (cc2 - T2)
#define X10_ (cc4 - T4)
#define X11_ (cc1 - T1)
#define X12_ (cc0 - T0)

/* Combine + store (L13_direct's store bodies with the Rader output map).
 * tr=1: 3 full WCxWC tiles + the m=12 column as 16 B extract-stores
 * (their panel_r7 LASTCOL_); tr=0: plain stores at dst + m*db. */
#if WC == 4
#  define L13R_STORE_BODY()                                                    \
    do {                                                                       \
        if (!tr) {                                                             \
            VST(dst + 0 * db, X0_);                                            \
            VST(dst + 1 * db, X1_);   VST(dst + 2 * db, X2_);                  \
            VST(dst + 3 * db, X3_);   VST(dst + 4 * db, X4_);                  \
            VST(dst + 5 * db, X5_);   VST(dst + 6 * db, X6_);                  \
            VST(dst + 7 * db, X7_);   VST(dst + 8 * db, X8_);                  \
            VST(dst + 9 * db, X9_);   VST(dst + 10 * db, X10_);                \
            VST(dst + 11 * db, X11_); VST(dst + 12 * db, X12_);                \
        } else {                                                               \
            TILE4_(0, X0_, X1_, X2_, X3_);                                     \
            TILE4_(4, X4_, X5_, X6_, X7_);                                     \
            TILE4_(8, X8_, X9_, X10_, X11_);                                   \
            LASTCOL_(X12_);                                                    \
        }                                                                      \
    } while (0)
#  define LASTCOL_(e)                                                          \
      do {                                                                     \
          VT c_ = (e);                                                         \
          *(SUF(v2) *)(dst + 0 * da + 24) = (SUF(v2)){c_[0], c_[1]};           \
          *(SUF(v2) *)(dst + 1 * da + 24) = (SUF(v2)){c_[2], c_[3]};           \
          *(SUF(v2) *)(dst + 2 * da + 24) = (SUF(v2)){c_[4], c_[5]};           \
          *(SUF(v2) *)(dst + 3 * da + 24) = (SUF(v2)){c_[6], c_[7]};           \
      } while (0)
#  define TILE4_(m0, e0, e1, e2, e3)                                           \
      do {                                                                     \
          VT xx[4], yy[4];                                                     \
          xx[0] = (e0); xx[1] = (e1); xx[2] = (e2); xx[3] = (e3);              \
          TTILE(xx, yy);                                                       \
          VST(dst + 0 * da + (m0) * 2, yy[0]);                                 \
          VST(dst + 1 * da + (m0) * 2, yy[1]);                                 \
          VST(dst + 2 * da + (m0) * 2, yy[2]);                                 \
          VST(dst + 3 * da + (m0) * 2, yy[3]);                                 \
      } while (0)
#elif WC == 2
#  define L13R_STORE_BODY()                                                    \
    do {                                                                       \
        if (!tr) {                                                             \
            VST(dst + 0 * db, X0_);                                            \
            VST(dst + 1 * db, X1_);   VST(dst + 2 * db, X2_);                  \
            VST(dst + 3 * db, X3_);   VST(dst + 4 * db, X4_);                  \
            VST(dst + 5 * db, X5_);   VST(dst + 6 * db, X6_);                  \
            VST(dst + 7 * db, X7_);   VST(dst + 8 * db, X8_);                  \
            VST(dst + 9 * db, X9_);   VST(dst + 10 * db, X10_);                \
            VST(dst + 11 * db, X11_); VST(dst + 12 * db, X12_);                \
        } else {                                                               \
            TILE2_(0, X0_, X1_);                                               \
            TILE2_(2, X2_, X3_);                                               \
            TILE2_(4, X4_, X5_);                                               \
            TILE2_(6, X6_, X7_);                                               \
            TILE2_(8, X8_, X9_);                                               \
            TILE2_(10, X10_, X11_);                                            \
            LASTCOL_(X12_);                                                    \
        }                                                                      \
    } while (0)
#  define LASTCOL_(e)                                                          \
      do {                                                                     \
          VT c_ = (e);                                                         \
          *(SUF(v2) *)(dst + 0 * da + 24) = (SUF(v2)){c_[0], c_[1]};           \
          *(SUF(v2) *)(dst + 1 * da + 24) = (SUF(v2)){c_[2], c_[3]};           \
      } while (0)
#  define TILE2_(m0, e0, e1)                                                   \
      do {                                                                     \
          VT xx[2], yy[2];                                                     \
          xx[0] = (e0); xx[1] = (e1);                                          \
          TTILE(xx, yy);                                                       \
          VST(dst + 0 * da + (m0) * 2, yy[0]);                                 \
          VST(dst + 1 * da + (m0) * 2, yy[1]);                                 \
      } while (0)
#else /* WC == 1 -- xmm tail: one lane, tr irrelevant (db is 2 at every tr=1
       * call site), 13 16-byte stores on 16-byte-aligned complex addresses
       * (never split, exactly containable by later 16 B loads). */
#  define L13R_STORE_BODY()                                                    \
    do {                                                                       \
        (void)da; (void)tr;                                                    \
        VST(dst + 0 * db, X0_);                                                \
        VST(dst + 1 * db, X1_);   VST(dst + 2 * db, X2_);                      \
        VST(dst + 3 * db, X3_);   VST(dst + 4 * db, X4_);                      \
        VST(dst + 5 * db, X5_);   VST(dst + 6 * db, X6_);                      \
        VST(dst + 7 * db, X7_);   VST(dst + 8 * db, X8_);                      \
        VST(dst + 9 * db, X9_);   VST(dst + 10 * db, X10_);                    \
        VST(dst + 11 * db, X11_); VST(dst + 12 * db, X12_);                    \
    } while (0)
#endif

/* -----------------------------------------------------------------------
 *  One chunk: the Rader-13 DFT of WC lines at once, all constants in
 *  registers.  src, rs: element (j=0, lane 0), rs doubles between rows;
 *  dst, da, db: output element m of lane f at dst + f*da + m*db (tr=1) or
 *  lanes contiguous, stores at dst + m*db (tr=0).
 *  C0..C2 = CP (seeded cyclic-3), C3..C5 = CM (negacyclic-3), broadcast;
 *  S0..S5 = lane-alternating (+sin(2pi g^t/13), -sin..) splats.
 * --------------------------------------------------------------------- */
static inline __attribute__((always_inline)) void
SUF(chunk13r)(const double *restrict src, long rs, double *restrict dst,
              long da, long db,
              VT C0, VT C1, VT C2, VT C3, VT C4, VT C5,
              VT S0, VT S1, VT S2, VT S3, VT S4, VT S5, int tr)
{
    VT dc, cc0, cc1, cc2, cc3, cc4, cc5, T0, T1, T2, T3, T4, T5;
    {
        VT w0, w1, w2, w3, w4, w5;
        VT a0, a1, a2, b0, b1, b2;
        {
            VT x0 = VLD(src);
            dc = x0;
            /* fold block t=0 / t=3: rows (1,12), (8,5) */
            {
                VT e0 = VLD(src + 1 * rs), f0 = VLD(src + 12 * rs);
                VT e1 = VLD(src + 8 * rs), f1 = VLD(src + 5 * rs);
                w0 = SWAPRI(e0 - f0);  w3 = SWAPRI(e1 - f1);
                VT u0 = e0 + f0, u1 = e1 + f1;
                VT pp = u0 + u1, qq = u0 - u1;
                dc += pp;
                a0 = x0 + pp * C0;  a1 = x0 + pp * C1;  a2 = x0 + pp * C2;
                b0 = qq * C3;  b1 = qq * C4;  b2 = qq * C5;
            }
            /* t=1 / t=4: rows (2,11), (3,10) */
            {
                VT e0 = VLD(src + 2 * rs), f0 = VLD(src + 11 * rs);
                VT e1 = VLD(src + 3 * rs), f1 = VLD(src + 10 * rs);
                w1 = SWAPRI(e0 - f0);  w4 = SWAPRI(e1 - f1);
                VT u0 = e0 + f0, u1 = e1 + f1;
                VT pp = u0 + u1, qq = u0 - u1;
                dc += pp;
                a0 += pp * C1;  a1 += pp * C2;  a2 += pp * C0;
                b0 += qq * C4;  b1 += qq * C5;  b2 -= qq * C3;
            }
            /* t=2 / t=5: rows (4,9), (6,7) */
            {
                VT e0 = VLD(src + 4 * rs), f0 = VLD(src + 9 * rs);
                VT e1 = VLD(src + 6 * rs), f1 = VLD(src + 7 * rs);
                w2 = SWAPRI(e0 - f0);  w5 = SWAPRI(e1 - f1);
                VT u0 = e0 + f0, u1 = e1 + f1;
                VT pp = u0 + u1, qq = u0 - u1;
                dc += pp;
                a0 += pp * C2;  a1 += pp * C0;  a2 += pp * C1;
                b0 += qq * C5;  b1 -= qq * C3;  b2 -= qq * C4;
            }
        }
        cc0 = a0 + b0;  cc3 = a0 - b0;
        cc1 = a1 + b1;  cc4 = a1 - b1;
        cc2 = a2 + b2;  cc5 = a2 - b2;

        /* dense negacyclic-6: T_n = sum_t sigma(n+t) S_{(n+t) mod 6} w_t,
         * sigma = -1 when n+t >= 6.  Negated wraps written as -= against
         * the SAME constant (one register, vfnmadd). */
        T0  = w0 * S0;  T1  = w0 * S1;  T2  = w0 * S2;
        T3  = w0 * S3;  T4  = w0 * S4;  T5  = w0 * S5;
        T0 += w1 * S1;  T1 += w1 * S2;  T2 += w1 * S3;
        T3 += w1 * S4;  T4 += w1 * S5;  T5 -= w1 * S0;
        T0 += w2 * S2;  T1 += w2 * S3;  T2 += w2 * S4;
        T3 += w2 * S5;  T4 -= w2 * S0;  T5 -= w2 * S1;
        T0 += w3 * S3;  T1 += w3 * S4;  T2 += w3 * S5;
        T3 -= w3 * S0;  T4 -= w3 * S1;  T5 -= w3 * S2;
        T0 += w4 * S4;  T1 += w4 * S5;  T2 -= w4 * S0;
        T3 -= w4 * S1;  T4 -= w4 * S2;  T5 -= w4 * S3;
        T0 += w5 * S5;  T1 -= w5 * S0;  T2 -= w5 * S1;
        T3 -= w5 * S2;  T4 -= w5 * S3;  T5 -= w5 * S4;
    }

    L13R_STORE_BODY();
}

/* The whole-width execs below are not instantiated at WC==1 (the w1 kernel
 * exists only as the mixed execs' tail). */
#if WC != 1

/* chunk start offsets covering a 13-long index; the last overlaps and
 * rewrites identical bits (zsolid at WC==4 -- L13_direct panel_r11) */
#if WC == 4
static const int SUF(off13r)[4] = {0, 4, 8, 9};
#  define NOFF13R 4
#else
static const int SUF(off13r)[7] = {0, 2, 4, 6, 8, 10, 11};
#  define NOFF13R 7
#endif

/* generic X-first exec: pure full-width chunks in every position.  At WC==4
 * this is the "p7" race arm (zsolid everywhere -- the natural shape for this
 * node's TWO 512-bit FMA pipes, where narrow tails have no port advantage);
 * at WC==2 it is the AVX2 fallback. */
static __attribute__((unused)) void
SUF(exec_rxf)(const fft3d_plan *restrict p, const double _Complex *restrict in,
              double _Complex *restrict out)
{
    const double *restrict cd = CPD(p);
    const double *restrict st = SNT(p);
    double *restrict pb = p->pb;
    double *restrict t1 = p->t1;
    const int nb = p->batch;
    VT C0 = CGET(cd, 0), C1 = CGET(cd, 1), C2 = CGET(cd, 2),
       C3 = CGET(cd, 3), C4 = CGET(cd, 4), C5 = CGET(cd, 5);
    VT S0 = CGET(st, 0), S1 = CGET(st, 1), S2 = CGET(st, 2),
       S3 = CGET(st, 3), S4 = CGET(st, 4), S5 = CGET(st, 5);
    L13R_PIN12_ASM();

    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);

        for (int i = 0; i < 169 / WC; ++i) {
            long f0 = (long)i * WC;
            SUF(chunk13r)(vin + 2 * f0, 338, t1 + 2 * f0, 2, L13R_T1P,
                          C0, C1, C2, C3, C4, C5, S0, S1, S2, S3, S4, S5, 0);
        }
#if (169 % WC) != 0
        SUF(chunk13r)(vin + 2 * (169 - WC), 338, t1 + 2 * (169 - WC), 2,
                      L13R_T1P, C0, C1, C2, C3, C4, C5,
                      S0, S1, S2, S3, S4, S5, 0);
#endif
        for (int x = 0; x < 13; ++x) {
            const double *pin = t1 + (long)x * L13R_T1P;
            double *pt = vout + (long)x * 338;
            for (int t = 0; t < NOFF13R; ++t) {
                long f0 = SUF(off13r)[t];
                SUF(chunk13r)(pin + 2 * f0, 26, pb + f0 * L13R_PBROW,
                              L13R_PBROW, 2, C0, C1, C2, C3, C4, C5,
                              S0, S1, S2, S3, S4, S5, 1);
            }
            for (int t = 0; t < NOFF13R; ++t) {
                long f0 = SUF(off13r)[t];
                SUF(chunk13r)(pb + 2 * f0, L13R_PBROW, pt + f0 * 26, 26,
                              2, C0, C1, C2, C3, C4, C5,
                              S0, S1, S2, S3, S4, S5, 1);
            }
        }
    }
}

/* generic X-last exec: per x-plane Y,Z into t1, then the X pass into out.
 * At WC==4 this is the "xl" instrument arm (priced, never adopted: a
 * different pass order is a different rounding order); at WC==2 the AVX2
 * cache-resident fallback. */
static __attribute__((unused)) void
SUF(exec_rxl)(const fft3d_plan *restrict p, const double _Complex *restrict in,
              double _Complex *restrict out)
{
    const double *restrict cd = CPD(p);
    const double *restrict st = SNT(p);
    double *restrict pb = p->pb;
    double *restrict t1 = p->t1;
    const int nb = p->batch;
    VT C0 = CGET(cd, 0), C1 = CGET(cd, 1), C2 = CGET(cd, 2),
       C3 = CGET(cd, 3), C4 = CGET(cd, 4), C5 = CGET(cd, 5);
    VT S0 = CGET(st, 0), S1 = CGET(st, 1), S2 = CGET(st, 2),
       S3 = CGET(st, 3), S4 = CGET(st, 4), S5 = CGET(st, 5);
    L13R_PIN12_ASM();

    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);

        for (int x = 0; x < 13; ++x) {
            const double *pin = vin + (long)x * 338;
            double *pt = t1 + (long)x * L13R_T1P;
            for (int t = 0; t < NOFF13R; ++t) {
                long f0 = SUF(off13r)[t];
                SUF(chunk13r)(pin + 2 * f0, 26, pb + f0 * L13R_PBROW,
                              L13R_PBROW, 2, C0, C1, C2, C3, C4, C5,
                              S0, S1, S2, S3, S4, S5, 1);
            }
            for (int t = 0; t < NOFF13R; ++t) {
                long f0 = SUF(off13r)[t];
                SUF(chunk13r)(pb + 2 * f0, L13R_PBROW, pt + f0 * 26, 26,
                              2, C0, C1, C2, C3, C4, C5,
                              S0, S1, S2, S3, S4, S5, 1);
            }
        }
        for (int i = 0; i < 169 / WC; ++i) {
            long f0 = (long)i * WC;
            SUF(chunk13r)(t1 + 2 * f0, L13R_T1P, vout + 2 * f0, 2, 338,
                          C0, C1, C2, C3, C4, C5, S0, S1, S2, S3, S4, S5, 0);
        }
#if (169 % WC) != 0
        SUF(chunk13r)(t1 + 2 * (169 - WC), L13R_T1P, vout + 2 * (169 - WC), 2,
                      338, C0, C1, C2, C3, C4, C5, S0, S1, S2, S3, S4, S5, 0);
#endif
    }
}

#undef NOFF13R
#endif /* WC != 1 */

#undef CPD
#undef SNT
#undef CGET
#undef L13R_PIN12_ASM
#undef L13R_STORE_BODY
#undef X0_
#undef X1_
#undef X2_
#undef X3_
#undef X4_
#undef X5_
#undef X6_
#undef X7_
#undef X8_
#undef X9_
#undef X10_
#undef X11_
#undef X12_
#if WC == 4
#  undef TILE4_
#  undef LASTCOL_
#elif WC == 2
#  undef TILE2_
#  undef LASTCOL_
#endif
#undef SWAPRI
#undef TTILE
#undef SHUF1
#undef SHUF2
#undef VLD
#undef VST
#undef VT
#undef IT
#undef VDW

#else /* !L13R_TEMPLATE_PASS ================================================== */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/* long-double trig so the splatted tables are good to ~1e-19 */
#include <math.h>

#include "../fft3d_api.h"

/* plane buffer row stride, in doubles: 13 complex padded to 16 (rows stay
 * 64-byte aligned; 13 rows = 3.3 KiB, L1-resident).  L13_direct's PBROW. */
enum { L13R_PBROW = 32 };

/* t1 x-plane stride, in doubles.  Two in-plane layouts coexist (raced):
 *   contiguous: 169 complex, rows of 26 doubles back to back (L13_direct's
 *     r8 form; their pad rule wants 43 lines = 344, see their record);
 *   row-padded (the "t1" race arm, ice_r2 -- L13_direct r11's sketched-but-
 *     unbuilt fix, built here): rows padded 26 -> 32 doubles and the X pass
 *     chunked per y-row zsolid (z0 = 0,4,8,9), so EVERY Y-pass load out of
 *     t1 is exactly one X-pass store (zero split loads, zero store-forward
 *     blocks at the t1->Y junction, where the contiguous form 3/4-splits
 *     ~480 zmm loads per volume).  Costs +10 zmm chunks/volume -- port
 *     slack this node's two FMA pipes have.
 * One stride serves both: 13 rows x 32 = 416 padded to 424 = 3392 B = 53
 * cache lines (odd-line rule), mod-4096 residue differs from in/out's 2704.
 * Pad tails are never read. */
enum { L13R_T1P = 424 };
enum { L13R_T1RS = 32 };   /* row stride inside a t1 plane, row-padded form */

struct fft3d_plan {
    int L, batch;
    void (*exec)(const struct fft3d_plan *, const double _Complex *, double _Complex *);
    double *cpd8; /* CP0-2,CM0-2 splatted 8x (broadcast constants)      */
    double *snt8; /* s_t = sin(2pi g^t/13), lane-alternating (s,-s), 8x */
    double *cpd4; /* the same, splatted 4x (ymm; xmm reads first 16 B)  */
    double *snt4;
    double *pb;   /* 13 x PBROW plane buffer (pass Y -> pass Z)         */
    double *t1;   /* one 13^3 volume of scratch, x-planes at T1P        */
    double *t1b;  /* second t1: OV ping-pong (L13_direct ice_r2)        */
    void *block;
};

/* ---- instantiate the kernel template at 512/256/128-bit ---- */
#if defined(__has_include)
#  if __has_include("L13_rader.c")
#    define L13R_SELF "L13_rader.c"
#  elif __has_include("impl/L13_rader.c")
#    define L13R_SELF "impl/L13_rader.c"
#  endif
#else
#  define L13R_SELF "L13_rader.c"
#endif

#ifdef L13R_SELF
#  define L13R_TEMPLATE_PASS 1
#  define WC 4
#  define SUF(x) x##_w4
#  include L13R_SELF
#  undef SUF
#  undef WC
#  undef L13R_TEMPLATE_PASS

#  define L13R_TEMPLATE_PASS 2
#  define WC 2
#  define SUF(x) x##_w2
#  include L13R_SELF
#  undef SUF
#  undef WC
#  undef L13R_TEMPLATE_PASS

#  define L13R_TEMPLATE_PASS 3
#  define WC 1
#  define SUF(x) x##_w1
#  include L13R_SELF
#  undef SUF
#  undef WC
#  undef L13R_TEMPLATE_PASS
#else
#  error "L13_rader.c must be able to #include itself"
#endif

/* =============================================================================
 * Mixed-width execs (AVX-512 only): 512-bit chunk bodies with L13_direct's
 * panel_r11 junction hygiene -- ZSOLID Y groups (pb written 100% as 64 B
 * tiles: zero store-forward blocks where the Z pass reloads it) and xmm
 * (WC=1) tails in the Z and X positions (16 B ops never split a line; their
 * outputs go to out/t1 and are not wide-loaded hot).  Their FORCE=9 lesson:
 * NEVER put the xmm tail in the Y position (Z's 64 B loads would straddle
 * 13 narrow stores; +17..44% on wallaby).  The r10 ymm-tail shape is kept
 * as the y2 race arm.
 * =============================================================================
 */
#if defined(__AVX512F__)
#  define L13R_PIN12_W4M() __asm__("" : "+v"(C0), "+v"(C1), "+v"(C2), "+v"(C3),\
                                        "+v"(C4), "+v"(C5), "+v"(K0), "+v"(K1),\
                                        "+v"(K2), "+v"(K3), "+v"(K4), "+v"(K5))
#else
#  define L13R_PIN12_W4M() do { } while (0)
#endif

/* ---- streaming prefetch (L13_direct panel_r8, adopted verbatim): pfw the
 * out stream one plane ahead of the Z-pass stores that RFO it; paced read
 * prefetch of the NEXT volume's input.  Fires only from the _pf exec, which
 * create() selects only when in+out exceed this host's L3 (prefetch at
 * L3-resident batch measured a loss on both machines -- ice_r1 pf!=+3.3%,
 * their r7).  -DL13R_PW=0 / -DL13R_PFIN=0 kill either half. */
#ifndef L13R_PW
#  define L13R_PW 1
#endif
#ifndef L13R_PFIN
#  define L13R_PFIN 1
#endif
#if L13R_PW
#  define L13R_PW1(P, OFF)                                                     \
      do { if (P) __builtin_prefetch((const char *)(P) + (OFF), 1, 3); } while (0)
#else
#  define L13R_PW1(P, OFF) do { (void)(P); } while (0)
#endif

static inline void l13r_pfw43(const double *p)
{
#if L13R_PW
    for (int i = 0; i < 43; ++i)
        __builtin_prefetch((const char *)p + 64 * i, 1, 3);
#else
    (void)p;
#endif
}
static inline void l13r_pfr43(const double *p)
{
#if L13R_PFIN
    for (int i = 0; i < 43; ++i)
        __builtin_prefetch((const char *)p + 64 * i, 0, 2);
#else
    (void)p;
#endif
}

#define L13R_MIXP_DECLS                                                        \
    const double *restrict cd8 = p->cpd8, *restrict st8 = p->snt8;             \
    const double *restrict cd4 = p->cpd4, *restrict st4 = p->snt4;             \
    double *restrict pb = p->pb;                                               \
    double *restrict t1 = p->t1;                                               \
    const int nb = p->batch;                                                   \
    vd_w4 C0 = *(const vd_w4 *)(cd8 + 0),  C1 = *(const vd_w4 *)(cd8 + 8),     \
          C2 = *(const vd_w4 *)(cd8 + 16), C3 = *(const vd_w4 *)(cd8 + 24),    \
          C4 = *(const vd_w4 *)(cd8 + 32), C5 = *(const vd_w4 *)(cd8 + 40);    \
    vd_w4 K0 = *(const vd_w4 *)(st8 + 0),  K1 = *(const vd_w4 *)(st8 + 8),     \
          K2 = *(const vd_w4 *)(st8 + 16), K3 = *(const vd_w4 *)(st8 + 24),    \
          K4 = *(const vd_w4 *)(st8 + 32), K5 = *(const vd_w4 *)(st8 + 40);    \
    vd_w2 D0 = *(const vd_w2 *)(cd4 + 0),  D1 = *(const vd_w2 *)(cd4 + 4),     \
          D2 = *(const vd_w2 *)(cd4 + 8),  D3 = *(const vd_w2 *)(cd4 + 12),    \
          D4 = *(const vd_w2 *)(cd4 + 16), D5 = *(const vd_w2 *)(cd4 + 20);    \
    vd_w2 Q0 = *(const vd_w2 *)(st4 + 0),  Q1 = *(const vd_w2 *)(st4 + 4),     \
          Q2 = *(const vd_w2 *)(st4 + 8),  Q3 = *(const vd_w2 *)(st4 + 12),    \
          Q4 = *(const vd_w2 *)(st4 + 16), Q5 = *(const vd_w2 *)(st4 + 20);    \
    /* xmm-tail constants: lane pair (c,c)/(s,-s) = first 16 B of each 4-splat \
     * row.  Unpinned; whichever width an exec's tails skip is a dead load. */ \
    vd_w1 E0 = *(const vd_w1 *)(cd4 + 0),  E1 = *(const vd_w1 *)(cd4 + 4),     \
          E2 = *(const vd_w1 *)(cd4 + 8),  E3 = *(const vd_w1 *)(cd4 + 12),    \
          E4 = *(const vd_w1 *)(cd4 + 16), E5 = *(const vd_w1 *)(cd4 + 20);    \
    vd_w1 F0 = *(const vd_w1 *)(st4 + 0),  F1 = *(const vd_w1 *)(st4 + 4),     \
          F2 = *(const vd_w1 *)(st4 + 8),  F3 = *(const vd_w1 *)(st4 + 12),    \
          F4 = *(const vd_w1 *)(st4 + 16), F5 = *(const vd_w1 *)(st4 + 20);    \
    /* whichever tail width an exec skips is a dead load gcc deletes */        \
    (void)D0; (void)D1; (void)D2; (void)D3; (void)D4; (void)D5;                \
    (void)Q0; (void)Q1; (void)Q2; (void)Q3; (void)Q4; (void)Q5;                \
    (void)E0; (void)E1; (void)E2; (void)E3; (void)E4; (void)E5;                \
    (void)F0; (void)F1; (void)F2; (void)F3; (void)F4; (void)F5;                \
    L13R_PIN12_W4M()

/* zsolid group: 4 zmm chunks at 0,4,8,9 (Y position -- pb written 100% as
 * 64 B tiles).  The asm-opaque bound keeps the loop rolled (gcc 11 ignores
 * `#pragma GCC unroll 1` around an always_inline callee -- L17_rader r4). */
#define L13R_GROUP_ZS(SRCB, RS, DSTB, DA)                                      \
    do {                                                                       \
        int nt_ = 4;                                                           \
        __asm__("" : "+r"(nt_));                                               \
        for (int t_ = 0; t_ < nt_; ++t_) {                                     \
            long f0_ = (t_ == 3) ? 9L : 4L * t_;                               \
            chunk13r_w4((SRCB) + 2 * f0_, (RS), (DSTB) + f0_ * (DA), (DA), 2,  \
                C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5, 1);            \
        }                                                                      \
    } while (0)

/* mixed group: 3 zmm chunks + one xmm tail at 12 (Z position only) */
#define L13R_GROUP_MX(SRCB, RS, DSTB, DA)                                      \
    do {                                                                       \
        int nt_ = 3;                                                           \
        __asm__("" : "+r"(nt_));                                               \
        for (int t_ = 0; t_ < nt_; ++t_) {                                     \
            long f0_ = 4L * t_;                                                \
            chunk13r_w4((SRCB) + 2 * f0_, (RS), (DSTB) + f0_ * (DA), (DA), 2,  \
                C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5, 1);            \
        }                                                                      \
        chunk13r_w1((SRCB) + 2 * 12, (RS), (DSTB) + 12 * (DA), (DA), 2,        \
            E0, E1, E2, E3, E4, E5, F0, F1, F2, F3, F4, F5, 1);                \
    } while (0)

/* the r10 ymm-tail group shape (y2 race arm): 3 zmm + ymm at 11 */
#define L13R_GROUP_Y2(SRCB, RS, DSTB, DA)                                      \
    do {                                                                       \
        int nt_ = 3;                                                           \
        __asm__("" : "+r"(nt_));                                               \
        for (int t_ = 0; t_ < nt_; ++t_) {                                     \
            long f0_ = 4L * t_;                                                \
            chunk13r_w4((SRCB) + 2 * f0_, (RS), (DSTB) + f0_ * (DA), (DA), 2,  \
                C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5, 1);            \
        }                                                                      \
        chunk13r_w2((SRCB) + 2 * 11, (RS), (DSTB) + 11 * (DA), (DA), 2,        \
            D0, D1, D2, D3, D4, D5, Q0, Q1, Q2, Q3, Q4, Q5, 1);                \
    } while (0)

/* X pass, mixed: 42 zmm chunks + one xmm tail at 168.  PFW0: optional
 * prefetchw target (first out plane, one line per chunk); 0 elides. */
#define L13R_XPASS_MX(SRCB, SRS, DSTB, DRS, PFW0)                              \
    do {                                                                       \
        int nx_ = 42;                                                          \
        __asm__("" : "+r"(nx_));                                               \
        for (int i_ = 0; i_ < nx_; ++i_) {                                     \
            long f0_ = 4L * i_;                                                \
            chunk13r_w4((SRCB) + 2 * f0_, (SRS), (DSTB) + 2 * f0_, 2, (DRS),   \
                C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5, 0);            \
            L13R_PW1((PFW0), 64 * i_);                                         \
        }                                                                      \
        chunk13r_w1((SRCB) + 2 * 168, (SRS), (DSTB) + 2 * 168, 2, (DRS),       \
            E0, E1, E2, E3, E4, E5, F0, F1, F2, F3, F4, F5, 0);                \
        L13R_PW1((PFW0), 64 * 42);                                             \
    } while (0)

/* X pass, row-padded t1 (the "t1" arm): per y-row r, 4 zsolid zmm chunks at
 * z0 = 0,4,8,9 storing into t1 rows of L13R_T1RS doubles.  The z0=9 overlap
 * chunk rewrites z=9..11 with identical bits and delivers z=12 inside a full
 * 64 B store, so the Y pass's f0=9 load (offset 18 doubles into the row)
 * matches it EXACTLY -- address-exact store-forward instead of the
 * contiguous layout's two-store straddle.  52 zmm chunks, no tail. */
#define L13R_XPASS_ZR(SRCB, SRS, DSTB, PFW0)                                   \
    do {                                                                       \
        int pw_ = 0;                                                           \
        int nr_ = 13;                                                          \
        __asm__("" : "+r"(nr_));                                               \
        for (int r_ = 0; r_ < nr_; ++r_) {                                     \
            int nz_ = 4;                                                       \
            __asm__("" : "+r"(nz_));                                           \
            for (int t_ = 0; t_ < nz_; ++t_) {                                 \
                long z0_ = (t_ == 3) ? 9L : 4L * t_;                           \
                chunk13r_w4((SRCB) + 2 * (13 * r_ + z0_), (SRS),               \
                    (DSTB) + (long)L13R_T1RS * r_ + 2 * z0_, 2, L13R_T1P,      \
                    C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5, 0);        \
                if (pw_ < 43) { L13R_PW1((PFW0), 64 * pw_); ++pw_; }           \
            }                                                                  \
        }                                                                      \
    } while (0)

#define L13R_XPASS_Y2(SRCB, SRS, DSTB, DRS)                                    \
    do {                                                                       \
        int nx_ = 42;                                                          \
        __asm__("" : "+r"(nx_));                                               \
        for (int i_ = 0; i_ < nx_; ++i_) {                                     \
            long f0_ = 4L * i_;                                                \
            chunk13r_w4((SRCB) + 2 * f0_, (SRS), (DSTB) + 2 * f0_, 2, (DRS),   \
                C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5, 0);            \
        }                                                                      \
        chunk13r_w2((SRCB) + 2 * 167, (SRS), (DSTB) + 2 * 167, 2, (DRS),       \
            D0, D1, D2, D3, D4, D5, Q0, Q1, Q2, Q3, Q4, Q5, 0);                \
    } while (0)

/* THE DEFAULT (ws <= L3): X-first, zsolid Y groups, xmm Z/X tails */
static __attribute__((unused)) void
l13r_exec_xfzs_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                  double _Complex *restrict out)
{
    L13R_MIXP_DECLS;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);
        L13R_XPASS_MX(vin, 338, t1, L13R_T1P, (double *)0);
        for (int x = 0; x < 13; ++x) {
            const double *pin = t1 + (long)x * L13R_T1P;
            double *pt = vout + (long)x * 338;
            L13R_GROUP_ZS(pin, 26, pb, L13R_PBROW);
            L13R_GROUP_MX(pb, L13R_PBROW, pt, 26);
        }
    }
}

/* OV chunk emitter: X chunks [KA,KB) of the NEXT volume (43 = 42 zmm + 1
 * xmm tail at 168) into the inactive t1 buffer.  L13_direct ice_r2's ov,
 * re-expressed for this kernel's macros. */
#define L13R_OVX(VNX, TDST, KA, KB)                                            \
    do {                                                                       \
        for (int k_ = (KA); k_ < (KB); ++k_) {                                 \
            if (k_ < 42) {                                                     \
                long f0_ = 4L * k_;                                            \
                chunk13r_w4((VNX) + 2 * f0_, 338, (TDST) + 2 * f0_, 2,         \
                    L13R_T1P, C0, C1, C2, C3, C4, C5,                          \
                    K0, K1, K2, K3, K4, K5, 0);                                \
            } else {                                                           \
                chunk13r_w1((VNX) + 2 * 168, 338, (TDST) + 2 * 168, 2,         \
                    L13R_T1P, E0, E1, E2, E3, E4, E5,                          \
                    F0, F1, F2, F3, F4, F5, 0);                                \
            }                                                                  \
        }                                                                      \
    } while (0)

/* OV -- cross-volume X-pass overlap (adopted: L13_direct ice_r2 <-
 * L17_rader ice_r1 "sp").  Volume 0's X pass up front; then each volume's
 * plane phase carries the NEXT volume's 43 X chunks, 3-4 per plane between
 * the Y and Z groups, into the inactive t1 buffer (ping-pong t1/t1b).
 * Same chunks, same per-volume out-store order => bit-identical to zs;
 * at nb=1 the schedule IS zs. */
static __attribute__((unused)) void
l13r_exec_xfov_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                  double _Complex *restrict out)
{
    L13R_MIXP_DECLS;
    double *ta = t1, *tb = p->t1b;
    L13R_XPASS_MX((const double *)in, 338, ta, L13R_T1P, (double *)0);
    for (int b = 0; b < nb; ++b) {
        double *vout = (double *)(out + (size_t)b * 2197);
        const double *vnx =
            (b + 1 < nb) ? (const double *)(in + (size_t)(b + 1) * 2197) : 0;
        int k = 0;
        for (int x = 0; x < 13; ++x) {
            const double *pin = ta + (long)x * L13R_T1P;
            double *pt = vout + (long)x * 338;
            L13R_GROUP_ZS(pin, 26, pb, L13R_PBROW);
            if (vnx) {
                int k1 = (43 * (x + 1)) / 13;
                L13R_OVX(vnx, tb, k, k1);
                k = k1;
            }
            L13R_GROUP_MX(pb, L13R_PBROW, pt, 26);
        }
        if (vnx) { double *sw = ta; ta = tb; tb = sw; }
    }
}

/* the r10 ymm-tail shape (y2 race arm; bit-identical to the default) */
static __attribute__((unused)) void
l13r_exec_xf_y2_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                   double _Complex *restrict out)
{
    L13R_MIXP_DECLS;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);
        L13R_XPASS_Y2(vin, 338, t1, L13R_T1P);
        for (int x = 0; x < 13; ++x) {
            const double *pin = t1 + (long)x * L13R_T1P;
            double *pt = vout + (long)x * 338;
            L13R_GROUP_Y2(pin, 26, pb, L13R_PBROW);
            L13R_GROUP_Y2(pb, L13R_PBROW, pt, 26);
        }
    }
}

/* row-padded t1 (the "t1" race arm): zsolid per-row X pass, Y groups read
 * t1 rows at L13R_T1RS; Z groups unchanged.  Bit-identical to the default. */
static __attribute__((unused)) void
l13r_exec_xft1_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                  double _Complex *restrict out)
{
    L13R_MIXP_DECLS;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);
        L13R_XPASS_ZR(vin, 338, t1, (double *)0);
        for (int x = 0; x < 13; ++x) {
            const double *pin = t1 + (long)x * L13R_T1P;
            double *pt = vout + (long)x * 338;
            L13R_GROUP_ZS(pin, L13R_T1RS, pb, L13R_PBROW);
            L13R_GROUP_MX(pb, L13R_PBROW, pt, 26);
        }
    }
}

/* row-padded t1 + the streaming prefetch schedule (>L3 tier of the t1 form) */
static __attribute__((unused)) void
l13r_exec_xft1_pf_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                     double _Complex *restrict out)
{
    L13R_MIXP_DECLS;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);
        const double *vnx =
            (b + 1 < nb) ? (const double *)(in + (size_t)(b + 1) * 2197) : 0;
        L13R_XPASS_ZR(vin, 338, t1, vout);
        for (int x = 0; x < 13; ++x) {
            const double *pin = t1 + (long)x * L13R_T1P;
            double *pt = vout + (long)x * 338;
            if (x < 12) l13r_pfw43(vout + (long)(x + 1) * 338);
            L13R_GROUP_ZS(pin, L13R_T1RS, pb, L13R_PBROW);
            if (vnx) l13r_pfr43(vnx + (long)x * 43 * 8);
            L13R_GROUP_MX(pb, L13R_PBROW, pt, 26);
        }
    }
}

/* streaming (ws > L3): the default shape + L13_direct r8's prefetch schedule */
static __attribute__((unused)) void
l13r_exec_xfzs_pf_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                     double _Complex *restrict out)
{
    L13R_MIXP_DECLS;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);
        const double *vnx =
            (b + 1 < nb) ? (const double *)(in + (size_t)(b + 1) * 2197) : 0;
        L13R_XPASS_MX(vin, 338, t1, L13R_T1P, vout);
        for (int x = 0; x < 13; ++x) {
            const double *pin = t1 + (long)x * L13R_T1P;
            double *pt = vout + (long)x * 338;
            if (x < 12) l13r_pfw43(vout + (long)(x + 1) * 338);
            L13R_GROUP_ZS(pin, 26, pb, L13R_PBROW);
            if (vnx) l13r_pfr43(vnx + (long)x * 43 * 8);
            L13R_GROUP_MX(pb, L13R_PBROW, pt, 26);
        }
    }
}

/* ---------------------------------------------------------------------------
 * plumbing
 * ---------------------------------------------------------------------------
 */
static char g_desc[384] =
    "Rader-13 (93 FP/chunk) in the lanes=lines pipeline";

const char *fft3d_name(void) { return "L13_rader"; }
const char *fft3d_description(void) { return g_desc; }
int fft3d_supports(int L) { return L == 13; }

#define L13R_ALIGN64(q) ((double *)(((uintptr_t)(q) + 63u) & ~(uintptr_t)63u))

static double __attribute__((unused)) l13r_now(void)   /* unused if no race */
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e9 * (double)ts.tv_sec + (double)ts.tv_nsec;
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != 13 || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = 13;
    p->batch = batch;

    size_t nd = 6 * 8 + 6 * 8 + 6 * 4 + 6 * 4
                + 13 * L13R_PBROW + 2 * 13 * L13R_T1P + 16;
    p->block = malloc(nd * sizeof(double) + 64);
    if (!p->block) { free(p); return NULL; }
    double *q = L13R_ALIGN64((double *)p->block);
    p->cpd8 = q; q += 6 * 8;
    p->snt8 = q; q += 6 * 8;
    p->cpd4 = q; q += 6 * 4;
    p->snt4 = q; q += 6 * 4;
    p->pb = q;   q += 13 * L13R_PBROW;
    p->t1 = q;   q += 13 * L13R_T1P;
    p->t1b = q;
    memset(p->pb, 0, (13 * L13R_PBROW + 2 * 13 * L13R_T1P) * sizeof(double));

    /* g^t mod 13 for t = 0..5; CP/CM are the halved cyclic-3/negacyclic-3
     * cosine kernels, snt rows are LANE-ALTERNATING (+s,-s): the *i of the
     * sine side lives here, not in a per-chunk XOR (L13_direct panel_r9). */
    {
        static const int gt[6] = {1, 2, 4, 8, 3, 6};
        const long double PI2 = 6.283185307179586476925286766559L;
        for (int t = 0; t < 3; ++t) {
            double ca = (double)cosl(PI2 * (long double)gt[t] / 13.0L);
            double cb = (double)cosl(PI2 * (long double)gt[t + 3] / 13.0L);
            double cp = 0.5 * (ca + cb), cm = 0.5 * (ca - cb);
            for (int i = 0; i < 8; ++i) {
                p->cpd8[t * 8 + i] = cp;
                p->cpd8[(t + 3) * 8 + i] = cm;
            }
            for (int i = 0; i < 4; ++i) {
                p->cpd4[t * 4 + i] = cp;
                p->cpd4[(t + 3) * 4 + i] = cm;
            }
        }
        for (int t = 0; t < 6; ++t) {
            double s = (double)sinl(PI2 * (long double)gt[t] / 13.0L);
            for (int i = 0; i < 8; ++i)
                p->snt8[t * 8 + i] = (i & 1) ? -s : s;
            for (int i = 0; i < 4; ++i)
                p->snt4[t * 4 + i] = (i & 1) ? -s : s;
        }
    }

    /* Deterministic selection (L13_direct's tiers): X-first everywhere;
     * OV when in+out fit this host's L3 (their ice_r2 default flip: ov
     * never lost a window), + prefetch schedule past L3. */
    long l2c = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (l2c <= 0) l2c = 1 << 20;
    long l3c = sysconf(_SC_LEVEL3_CACHE_SIZE);
    if (l3c <= 0) l3c = 22l << 20;
    unsigned long long ws = 32ull * (unsigned long long)batch * 2197ull;
    const char *pick;
#if defined(__AVX512F__)
    if (ws > (unsigned long long)l3c) { p->exec = l13r_exec_xfzs_pf_mx; pick = "zs+pf"; }
    else                              { p->exec = l13r_exec_xfov_mx;    pick = "ov"; }
#else
    if (ws > (unsigned long long)l2c) { p->exec = exec_rxf_w2; pick = "256b xf"; }
    else                              { p->exec = exec_rxl_w2; pick = "256b xl"; }
#endif

#if defined(L13R_FORCE)
    switch ((int)L13R_FORCE) {
    case 0: p->exec = l13r_exec_xfzs_mx;    pick = "FORCED zs"; break;
    case 1: p->exec = l13r_exec_xfzs_pf_mx; pick = "FORCED zs+pf"; break;
    case 2: p->exec = l13r_exec_xf_y2_mx;   pick = "FORCED y2"; break;
    case 3: p->exec = exec_rxf_w4;          pick = "FORCED p7 (pure zmm)"; break;
    case 4: p->exec = exec_rxl_w4;          pick = "FORCED xl (pure zmm X-last)"; break;
    case 5: p->exec = exec_rxf_w2;          pick = "FORCED 256b xf"; break;
    case 6: p->exec = exec_rxl_w2;          pick = "FORCED 256b xl"; break;
    case 7: p->exec = l13r_exec_xft1_mx;    pick = "FORCED t1 (row-padded t1)"; break;
    case 8: p->exec = l13r_exec_xft1_pf_mx; pick = "FORCED t1+pf"; break;
    case 9: p->exec = l13r_exec_xfov_mx;    pick = "FORCED ov"; break;
    default: break;
    }
#endif

    /* ---- CHAIN-SHAPED in-plan adopting race (adopted: L13_direct ice_r2
     * <- L17_matrixsimd ice_r1's chain arena; replaces ice_r2's fixed
     * src->dst race, whose L2-resident regime read ~250 ns/vol optimistic).
     * tb=min(batch,32) volumes ping-pong two private buffers with an
     * UNTIMED driver-style unitary scale pass (x 2197^-1/2) between steps,
     * so candidates are ranked in the graded cache regime.  Preceded by a
     * ~120 ms dense-FMA clock-settle spin (L17_winograd via L13_direct:
     * schedutil leaves create() on an unramped core).  Arms ov (incumbent)/
     * zs/y2/t1 are OUTPUT-BIT-IDENTICAL (same per-lane DAG and constants;
     * ov only reorders whole chunks across volumes), so a flip can change
     * time but never results.  Incumbency rule kept from L6_pfa: adopt a
     * challenger only if >1.5% better in BOTH of two independent blocks
     * (min over laps, variant order alternating).  p7/xl retired from the
     * race -- priced twice, see the record; FORCE=3/4 still reach them.
     * Skipped when the pf tier is selected.  -DL13R_AB=0 removes it. */
#ifndef L13R_AB
#  define L13R_AB 1
#endif
#ifndef L13R_ABTH
#  define L13R_ABTH 0.985
#endif
    char abuf[160] = "";
#if L13R_AB && defined(__AVX512F__) && !defined(L13R_FORCE)
    if (ws <= (unsigned long long)l3c) do {
        typedef void (*l13r_fn)(const fft3d_plan *, const double _Complex *,
                                double _Complex *);
        l13r_fn cf[4] = { l13r_exec_xfov_mx, l13r_exec_xfzs_mx,
                          l13r_exec_xf_y2_mx, l13r_exec_xft1_mx };
        const char *tag[4] = { "ov", "zs", "y2", "t1" };
        const int nv = 4;
        const int tb = batch < 32 ? batch : 32;
        size_t vn = (size_t)2197 * tb;
        double _Complex *ra = 0, *rb = 0;
        if (posix_memalign((void **)&ra, 64, vn * 16)) break;
        if (posix_memalign((void **)&rb, 64, vn * 16)) { free(ra); break; }
        {   /* deterministic fill, no rand() */
            unsigned long long s = 0x9e3779b97f4a7c15ull;
            double *rd = (double *)ra;
            for (size_t i = 0; i < 2 * vn; ++i) {
                s = s * 6364136223846793005ull + 1442695040888963407ull;
                rd[i] = (double)(long long)(s >> 12) * 0x1p-50;
            }
            memset(rb, 0, vn * 16);
        }
        {   /* ~120 ms dense-FMA clock-settle spin before any timing */
            vd_w4 sa = {1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08};
            const vd_w4 sb = {0.9990, 0.9991, 0.9992, 0.9993,
                              0.9994, 0.9995, 0.9996, 0.9997};
            const vd_w4 sc = {1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3};
            double t0 = l13r_now();
            while (l13r_now() - t0 < 1.2e8) {
                for (int i = 0; i < 8192; ++i) sa = sa * sb + sc;
                __asm__("" : "+v"(sa));
            }
        }
        fft3d_plan tp = *p;
        tp.batch = tb;
        const double usc = 1.0 / sqrt(2197.0);      /* unitary 13^3 scale */
        int nsteps = 128 / tb;
        if (nsteps < 4) nsteps = 4;
        double b1[4] = {1e30, 1e30, 1e30, 1e30},
               b2[4] = {1e30, 1e30, 1e30, 1e30};
        for (int t = -1; t < 12; ++t)          /* t = -1 warms the buffers */
            for (int j = 0; j < nv; ++j) {
                int k = (t & 1) ? nv - 1 - j : j;
                double _Complex *src = ra, *dst = rb;
                double acc = 0.0;
                for (int s = 0; s < nsteps; ++s) {
                    double t0 = l13r_now();
                    cf[k](&tp, src, dst);
                    acc += l13r_now() - t0;
                    double *dd = (double *)dst;      /* untimed scale step */
                    for (size_t i = 0; i < 2 * vn; ++i) dd[i] *= usc;
                    double _Complex *sw = src; src = dst; dst = sw;
                }
                if (t < 0) continue;
                double dt = acc / ((double)nsteps * tb);
                double *bb = (t < 6) ? b1 : b2;
                if (dt < bb[k]) bb[k] = dt;
            }
        int win = 0;
        for (int k = 1; k < nv; ++k)
            if (b1[k] < L13R_ABTH * b1[win] && b2[k] < L13R_ABTH * b2[win])
                win = k;
        p->exec = cf[win];
        pick = tag[win];
        {
            int off = snprintf(abuf, sizeof abuf, " chain-ab[B%d]=", tb);
            for (int k = 0; k < nv && off < (int)sizeof abuf; ++k) {
                double best = b1[k] < b2[k] ? b1[k] : b2[k];
                off += snprintf(abuf + off, sizeof abuf - off, "%s%s:%.0f",
                                k ? "," : "", tag[k], best);
            }
        }
        fprintf(stderr, "L13_rader%s pick=%s\n", abuf, pick);
        free(ra); free(rb);
    } while (0);
#endif

    snprintf(g_desc, sizeof g_desc,
             "Rader-13 CRT (93 FP/chunk) in L13_direct's lanes=lines pipeline "
             "+ov cross-volume X overlap, %d-bit, zsolidY+xmm-tail X-first; "
             "pick=%s%s",
#if defined(__AVX512F__)
             512,
#else
             256,
#endif
             pick, abuf);
    return p;
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    plan->exec(plan, in, out);
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->block);
    free(p);
}

#endif /* L13R_TEMPLATE_PASS */
