/* =============================================================================
 * L13_direct -- 13^3 complex-double forward DFT as three dense 13x13 matrix
 *               passes, conjugate-pair folded, vectorised across whole lines
 *               (SIMD lanes = independent lines).
 *
 * TECHNIQUE (adopted wholesale from L17_matrixsimd, rounds 1-5 -- see
 * strategies/L17_matrixsimd.md; this entry is that design re-derived for L=13)
 *   Row-column: one dense length-13 DFT matrix applied along each axis.  The
 *   j <-> 13-j conjugate pair is folded first (FFTW's dft-generic form):
 *
 *     u_j = x_j + x_{13-j},  v_j = x_j - x_{13-j}          (j = 1..6)
 *     P_k = x_0 + sum_{j=1..6} cos(2pi kj/13) u_j          (k = 0..6)
 *     R_k =       sum_{j=1..6} sin(2pi kj/13) (-i v_j)     (k = 1..6)
 *     X_k = P_k + R_k,  X_{13-k} = P_k - R_k,  X_0 = P_0
 *
 *   Every surviving coefficient is REAL, so the driver's interleaved complex
 *   layout is already the right SIMD layout: one vector FMA per coefficient,
 *   no cross-lane arithmetic.  The only permutes are the re/im swap in (-i v)
 *   (integer XOR for the sign flip -- port 5, off the FMA port) and the plane
 *   transposes fused into the stores of passes Y and Z.
 *
 * WHY L=13 IS EASIER THAN L=17 (the one structural difference)
 *   The folded transform needs 7 P + 6 R = 13 accumulators.  L17's 17
 *   accumulators + temporaries spilled (its round-1 record: 342-805 stack refs
 *   in the monolithic kernel, fixed by k-blocking); 13 accumulators + 6 pinned
 *   sine constants + temporaries peak at ~26 of 32 EVEX registers, the same
 *   count as L17's winning pinned-nested kernel, so the monolithic form is
 *   used directly and no k-blocking is required.
 *
 * OPERATION COUNT (per 13^3 volume = 3*169 = 507 lines)
 *   Per chunk (WC lines at once): 6 u-adds + 42 cos FMA + 6 v-subs + 36 sin
 *   FMA + 12 combine = 102 vector FP ops.  Per line: 78 FMAs (156 flop) + 12
 *   complex add/sub (48 flop) = 204... in real flops per line at WC lanes:
 *   102 ops * 2*WC doubles include the lane width; per line = 360 flop.
 *   Compare: naive dense complex 13x13 matvec = 1352 flop/line (169 cMACs);
 *   folded = 360 flop/line (3.8x less).  Per volume: 507 * 360 = 182.5 kflop
 *   (yardstick 5 N log2 N = 121.9 kflop, so "GF/s" x1.5 = real Gflop/s).
 *
 *   Chunk economics on the scoring node (ONE 512-bit FMA unit, TWO 256-bit
 *   ports -- L17_rader r4 / L17_matrixsimd r5 "mixed tail", adopted):
 *     pure zmm:  13-long index = 4 chunks (offsets 0,4,8,9; last recomputes 3)
 *                volume = 2*13*4 + 43 = 147 chunks * 102 ops = 15.0k ops
 *     mixed:     Z groups 3 zmm + 1 XMM tail per 13 (offsets 0,4,8 + 12,
 *                panel_r11 -- the tail covers the ONE remaining line; through
 *                r10 it was a ymm chunk at 11 recomputing a line, same
 *                51-cycle port cost but its 32 B accesses at byte 48 (mod 64)
 *                split cache lines and its pb loads straddled two Y-pass
 *                stores, blocking store-forwarding at the plane junction);
 *                X pass 42 zmm + 1 xmm at 168; Y groups ZSOLID -- 4 zmm at
 *                0,4,8,9 so pb is written 100% as 64 B tiles and the Z pass
 *                store-forwards cleanly (the xmm tail in the Y position was
 *                a measured catastrophe, see the macro comments).
 *                volume = 133 zmm + 14 xmm = 133*102 + 14*51 = 14.3k cycles
 *                at 1 zmm-op/cycle == xmm pairs on 2 ports.
 *   Node floor ~14.3k cycles = 4.94 us at 2.9 GHz (the r10 all-mixed census
 *   was 13.6k = 4.7 -- zsolid trades +663 port cycles for zero SF blocks and
 *   zero tail splits).  On wallaby (TWO 512-bit units) the 4th zmm chunk is
 *   port-free and zsolid won outright; the +663 is a node-only cost.
 *
 * LAYOUT / PASSES (L17_matrixsimd's, unchanged)
 *   WC complex per vector (4 = zmm, 2 = ymm); lanes hold WC different lines
 *   from a contiguous run of a free index; tail chunks OVERLAP the previous
 *   chunk and rewrite identical values (never reads out of range, no masks).
 *     Y : in[x][y][z]  --lanes over z --> pb[z][ky]        (transposing store)
 *     Z : pb[z][ky]    --lanes over ky--> t1[x][ky][kz]    (transposing store)
 *     X : t1[x][p]     --lanes over p (169 wide)--> out[kx][p]  (plain store)
 *   A 13x13 plane is 2.6 KiB (L1-resident); a volume is 34.3 KiB (fits L1 on
 *   wallaby, 1.07x L1 on the node, comfortably in L2 everywhere).
 *   X-FIRST variant when in+out exceed this host's L2 (L17_matrixsimd r3 via
 *   L13_rader's sysconf gate): X from `in` into t1, then per-kx-plane Y,Z
 *   straight into `out`, so at streaming batch the output writes spread
 *   across the volume's compute instead of bursting.
 *   PINNED SINES (L17_matrixsimd r3 <- L17_winograd r2, adopted): the 6
 *   distinct sine magnitudes s_m = sin(2pi m/13) are broadcast once per
 *   execute into 6 registers made asm-opaque, and the sine sweep is fully
 *   unrolled with the signs sin(2pi kj/13) = +-s_{fold(kj)} baked in at
 *   compile time; 36 coefficient loads per chunk gone.  Cosines stay a
 *   PRE-SPLATTED table read as full-width memory operands in a rolled j-loop
 *   (L17_matrixsimd r1 item 1: scalar table + embedded broadcast makes gcc 11
 *   materialise every splat into a stack slot -- do not retry).
 *
 * DETERMINISM
 *   No runtime tuner: the exec is a pure function of the batch size, the
 *   compile-time ISA, and the host's sysconf L3 size (ws = in+out bytes:
 *   ws <= L3 X-first plain, ws > L3 X-first with the streaming prefetch
 *   schedule; mixed tail iff AVX-512), so repeated runs of the same binary
 *   are bit-identical by construction.  The X-LAST cache-resident tier was
 *   deleted in panel_r10 (its r6 justification predated the r8 t1 pad; see
 *   the selection comment in create()).  The in-plan timed discriminator in
 *   create() is INSTRUMENT ONLY (never changes the pick, so no pick lottery);
 *   panel_r11 slots: y2 (r10 ymm tails), xf (default), zs (zsolid Y), xl.
 *   -DL13_FORCE=n pins one variant; -DL13_PW=0 / -DL13_PFIN=0 kill the
 *   write/read halves of the prefetch schedule; -DL13_AB=0 removes the
 *   discriminator.
 *
 * ASSUMPTIONS
 *   gcc/clang vector extensions, -ffp-contract=fast (gnu11 default) for FMA
 *   formation; no intrinsics, so the same source builds (and can be verified,
 *   emulated) without AVX-512.  Template instantiated 3x by self-#include
 *   (zmm, ymm, and -- panel_r11 -- the xmm tail width).
 * =============================================================================
 */
/* panel_r9: the `#pragma GCC optimize("unroll-loops")` shipped in r8 is GONE.
 * The r8 verdict (§3c) showed the scored build has carried -funroll-loops all
 * along (the r7 "build-flag gap" never existed), and L17_rader measured the
 * pragma form as a ~2% TAX (optimize() rebuilds the whole per-function option
 * set, not just the named flag).  The kernels' deliberately-rolled loops are
 * protected by asm-opaque bounds and stay rolled either way. */

#ifdef L13_TEMPLATE_PASS
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

/* MULI(t): the -i*t for interleaved complex used to be swap re/im + an
 * integer XOR flipping the imaginary lanes' signs (6 vpxor per chunk).
 * panel_r9: the sign pattern is PER-LANE and constant, so it is folded into
 * the sine tables instead -- stab8/stab4 hold lane-alternating (s,-s,s,-s,..)
 * splats and MULI is just the swap.  (-sigma*s)*v = sigma*s*(-v) exactly in
 * IEEE, so the output is bit-identical; ~880 vector XORs per volume gone at
 * zero register cost. */
#if WC == 4
#  define SWAPRI(x) SHUF1(x, 1, 0, 3, 2, 5, 4, 7, 6)
#elif WC == 2
#  define SWAPRI(x) SHUF1(x, 1, 0, 3, 2)
#else
#  define SWAPRI(x) SHUF1(x, 1, 0)
#endif
#define MULI(t) SWAPRI(t)

/* Pin the 6 sine constants in registers across a whole execute (empty asm =
 * opaque to gcc: no rematerialisation, no folding back to memory operands).
 * Only on a 32-register EVEX file; elsewhere it is a no-op and the constants
 * decay gracefully to stack/memory operands. */
#if (WC == 4 && defined(__AVX512F__)) || (WC == 2 && defined(__AVX512VL__))
#  define L13_PIN_ASM() __asm__("" : "+v"(K0), "+v"(K1), "+v"(K2), "+v"(K3),  \
                                     "+v"(K4), "+v"(K5))
#  define L13_PIN12_ASM() __asm__("" : "+v"(C0), "+v"(C1), "+v"(C2), "+v"(C3),\
                                       "+v"(C4), "+v"(C5), "+v"(S0), "+v"(S1),\
                                       "+v"(S2), "+v"(S3), "+v"(S4), "+v"(S5))
#else
#  define L13_PIN_ASM() do { } while (0)
#  define L13_PIN12_ASM() do { } while (0)
#endif

/* Cosine coefficients from a PRE-SPLATTED table (full-width memory operands;
 * see the header block for why not scalar + embedded broadcast). */
#define CGET(base, i) VLD((base) + (size_t)(i) * VDW)
#define CSTEP_C (7 * VDW)
#if WC == 4
#  define CTAB(p) ((p)->ctab8)
#  define STAB(p) ((p)->stab8)
#  define CTD(p) ((p)->ctd8)
#else
#  define CTAB(p) ((p)->ctab4)
#  define STAB(p) ((p)->stab4)
#  define CTD(p) ((p)->ctd4)
#endif

/* ---- WCxWC transpose of complex (i.e. of 128-bit blocks) ----------------
 * WC==1 needs no transpose at all: one lane, so tr=1 and tr=0 stores are the
 * same 13 xmm stores at stride db (db is 2 at every tr=1 call site). */
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

/* Combine + store, shared by both kernels.  Expects P0..P6, R1..R6, dst, da,
 * db, tr in scope.  Output index m: X_0 = P_0, X_k = P_k + R_k,
 * X_{13-k} = P_k - R_k.  For tr=1, 13 outputs = 3 full WCxWC tiles + one
 * overlapping tile that rewrites its shared columns with identical bits
 * (same registers, same shuffles). */
#define XP_(k) (P##k + R##k)
#define XM_(k) (P##k - R##k)
#if WC == 4
#  define L13_STORE_BODY()                                                     \
    do {                                                                       \
        if (!tr) {                                                             \
            VST(dst + 0 * db, P0);                                             \
            VST(dst + 1 * db, XP_(1));  VST(dst + 2 * db, XP_(2));             \
            VST(dst + 3 * db, XP_(3));  VST(dst + 4 * db, XP_(4));             \
            VST(dst + 5 * db, XP_(5));  VST(dst + 6 * db, XP_(6));             \
            VST(dst + 7 * db, XM_(6));  VST(dst + 8 * db, XM_(5));             \
            VST(dst + 9 * db, XM_(4));  VST(dst + 10 * db, XM_(3));            \
            VST(dst + 11 * db, XM_(2)); VST(dst + 12 * db, XM_(1));            \
        } else {                                                               \
            TILE4_(0, P0, XP_(1), XP_(2), XP_(3));                             \
            TILE4_(4, XP_(4), XP_(5), XP_(6), XM_(6));                         \
            TILE4_(8, XM_(5), XM_(4), XM_(3), XM_(2));                         \
            LASTCOL_(XM_(1));                                                  \
        }                                                                      \
    } while (0)
/* Column m=12 straight from the one register that holds it: 4 16-byte
 * extract-stores instead of a 4th overlapping tile (8 shuffles + 4 wide
 * stores rewriting 3 identical columns).  Panel_r7; addresses are 16B
 * aligned for every da used here (26 and L13_PBROW doubles). */
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
#  define L13_STORE_BODY()                                                     \
    do {                                                                       \
        if (!tr) {                                                             \
            VST(dst + 0 * db, P0);                                             \
            VST(dst + 1 * db, XP_(1));  VST(dst + 2 * db, XP_(2));             \
            VST(dst + 3 * db, XP_(3));  VST(dst + 4 * db, XP_(4));             \
            VST(dst + 5 * db, XP_(5));  VST(dst + 6 * db, XP_(6));             \
            VST(dst + 7 * db, XM_(6));  VST(dst + 8 * db, XM_(5));             \
            VST(dst + 9 * db, XM_(4));  VST(dst + 10 * db, XM_(3));            \
            VST(dst + 11 * db, XM_(2)); VST(dst + 12 * db, XM_(1));            \
        } else {                                                               \
            TILE2_(0, P0, XP_(1));                                             \
            TILE2_(2, XP_(2), XP_(3));                                         \
            TILE2_(4, XP_(4), XP_(5));                                         \
            TILE2_(6, XP_(6), XM_(6));                                         \
            TILE2_(8, XM_(5), XM_(4));                                         \
            TILE2_(10, XM_(3), XM_(2));                                        \
            LASTCOL_(XM_(1));                                                  \
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
#else /* WC == 1 -- panel_r11 xmm tail: one lane, tr irrelevant (db is 2 at
       * every tr=1 call site), 13 16-byte stores on 16-byte-aligned complex
       * addresses.  16 B accesses never split a cache line, and each store is
       * fully containable by a later exact-address 16 B load, so the Z pass's
       * tail loads store-forward cleanly out of the Y pass's stores. */
#  define L13_STORE_BODY()                                                     \
    do {                                                                       \
        (void)da; (void)tr;                                                    \
        VST(dst + 0 * db, P0);                                                 \
        VST(dst + 1 * db, XP_(1));  VST(dst + 2 * db, XP_(2));                 \
        VST(dst + 3 * db, XP_(3));  VST(dst + 4 * db, XP_(4));                 \
        VST(dst + 5 * db, XP_(5));  VST(dst + 6 * db, XP_(6));                 \
        VST(dst + 7 * db, XM_(6));  VST(dst + 8 * db, XM_(5));                 \
        VST(dst + 9 * db, XM_(4));  VST(dst + 10 * db, XM_(3));                \
        VST(dst + 11 * db, XM_(2)); VST(dst + 12 * db, XM_(1));                \
    } while (0)
#endif

/* -----------------------------------------------------------------------
 *  One chunk: the length-13 DFT of WC lines at once.
 *  (not instantiated at WC==1: the pre-splatted table rows are laid out per
 *   splat width, and the tail only ever runs the all-pinned kernel anyway)
 *    src, rs : element (j=0, lane 0); rs doubles between successive j
 *    dst, da, db : output element (m) of lane (f) is at dst + f*da + m*db
 *    tr : 1 -> lanes are separate rows of dst (in-register transpose fused
 *             into the store);  0 -> lanes contiguous in dst (plain stores)
 *    K0..K5 : s_1..s_6 = sin(2pi m/13), splatted (pinned by the caller)
 * --------------------------------------------------------------------- */
#if WC != 1
static inline __attribute__((always_inline)) void
SUF(chunk13)(const double *restrict src, long rs, double *restrict dst, long da,
             long db, const double *restrict ct,
             VT K0, VT K1, VT K2, VT K3, VT K4, VT K5, int tr)
{
    VT P0, P1, P2, P3, P4, P5, P6, R1, R2, R3, R4, R5, R6;

    /* ---- symmetric half: 7 accumulators, 6 rank-1 updates, rolled ---- */
    {
        VT v0 = VLD(src);
        P0 = v0; P1 = v0; P2 = v0; P3 = v0; P4 = v0; P5 = v0; P6 = v0;
/* Do NOT unroll: the rolled loop keeps the table row pointer advancing (no
 * loop-invariant vector loads for gcc to hoist and spill; L17_matrixsimd r1
 * items 1 and 7). */
#pragma GCC unroll 1
        for (int j = 1; j <= 6; ++j) {
            VT u = VLD(src + (long)j * rs) + VLD(src + (long)(13 - j) * rs);
            const double *c = ct + (size_t)(j - 1) * CSTEP_C;
            P0 += CGET(c, 0) * u;
            P1 += CGET(c, 1) * u;
            P2 += CGET(c, 2) * u;
            P3 += CGET(c, 3) * u;
            P4 += CGET(c, 4) * u;
            P5 += CGET(c, 5) * u;
            P6 += CGET(c, 6) * u;
        }
    }

    /* ---- antisymmetric half: fully unrolled, pinned constants, the signs
     * sin(2pi kj/13) = +-s_{fold(kj mod 13)} baked in at compile time.
     * Row j updates R_k with sign/index from the fold of (k*j) mod 13. ---- */
    {
        VT w = MULI(VLD(src + 1 * rs) - VLD(src + 12 * rs));            /* j=1 */
        R1 = K0 * w; R2 = K1 * w; R3 = K2 * w;
        R4 = K3 * w; R5 = K4 * w; R6 = K5 * w;
        w = MULI(VLD(src + 2 * rs) - VLD(src + 11 * rs));               /* j=2 */
        R1 += K1 * w; R2 += K3 * w; R3 += K5 * w;
        R4 -= K4 * w; R5 -= K2 * w; R6 -= K0 * w;
        w = MULI(VLD(src + 3 * rs) - VLD(src + 10 * rs));               /* j=3 */
        R1 += K2 * w; R2 += K5 * w; R3 -= K3 * w;
        R4 -= K0 * w; R5 += K1 * w; R6 += K4 * w;
        w = MULI(VLD(src + 4 * rs) - VLD(src + 9 * rs));                /* j=4 */
        R1 += K3 * w; R2 -= K4 * w; R3 -= K0 * w;
        R4 += K2 * w; R5 -= K5 * w; R6 -= K1 * w;
        w = MULI(VLD(src + 5 * rs) - VLD(src + 8 * rs));                /* j=5 */
        R1 += K4 * w; R2 -= K2 * w; R3 += K1 * w;
        R4 -= K5 * w; R5 -= K0 * w; R6 += K3 * w;
        w = MULI(VLD(src + 6 * rs) - VLD(src + 7 * rs));                /* j=6 */
        R1 += K5 * w; R2 -= K0 * w; R3 += K4 * w;
        R4 -= K1 * w; R5 += K3 * w; R6 -= K2 * w;
    }

    L13_STORE_BODY();
}
#endif /* WC != 1 */

/* -----------------------------------------------------------------------
 *  chunk13p: the same transform with the WHOLE folded matrix register-
 *  resident.  L=13 has only 6 distinct cosine magnitudes c_m = cos(2pi m/13)
 *  (cos is even, so cos(2pi kj/13) = c_{fold(kj mod 13)} with the sign inside
 *  the value) and 6 distinct sines; all 12 are pinned, and a SINGLE fused
 *  sweep loads each input line exactly once: 13 line loads per chunk and
 *  ZERO coefficient loads (the table kernel above reads 42 splatted vectors
 *  = 2.6 KiB of table per chunk, which the first wallaby measurement showed
 *  to be the bottleneck -- only 102 FP ops to hide it under).  Liveness:
 *  13 accumulators + 12 pinned + a,b,u,w = 29 of 32 EVEX registers.  This is
 *  the one thing L=13 can do that L=17 could not (8+8 constants + 17
 *  accumulators do not fit).  The k=0 cosine row is 1.0: P0 += u, a plain
 *  add, same port and throughput as the FMA it replaces.
 * --------------------------------------------------------------------- */
/* Accumulation rows j=1..5 of the all-pinned kernel: loads each line once,
 * updates all 13 accumulators.  fold/sign tables: index m of (k*j mod 13)
 * folded to 1..6; sine sign is -1 when (k*j mod 13) > 6.  Cosine has no sign
 * (cos is even).  Same expression DAG as the r9 chunk13p body, so the default
 * kernel's values stay bit-identical through this refactor. */
#define L13_ACC15()                                                            \
    {                                                                          \
        VT v0 = VLD(src);                                                      \
        P0 = v0; P1 = v0; P2 = v0; P3 = v0; P4 = v0; P5 = v0; P6 = v0;         \
    }                                                                          \
    {                                                                          \
        VT a = VLD(src + 1 * rs), b = VLD(src + 12 * rs);           /* j=1 */  \
        VT u = a + b, w = MULI(a - b);                                         \
        P0 += u;                                                               \
        P1 += C0 * u; P2 += C1 * u; P3 += C2 * u;                              \
        P4 += C3 * u; P5 += C4 * u; P6 += C5 * u;                              \
        R1 = S0 * w; R2 = S1 * w; R3 = S2 * w;                                 \
        R4 = S3 * w; R5 = S4 * w; R6 = S5 * w;                                 \
    }                                                                          \
    {                                                                          \
        VT a = VLD(src + 2 * rs), b = VLD(src + 11 * rs);           /* j=2 */  \
        VT u = a + b, w = MULI(a - b);                                         \
        P0 += u;                                                               \
        P1 += C1 * u; P2 += C3 * u; P3 += C5 * u;                              \
        P4 += C4 * u; P5 += C2 * u; P6 += C0 * u;                              \
        R1 += S1 * w; R2 += S3 * w; R3 += S5 * w;                              \
        R4 -= S4 * w; R5 -= S2 * w; R6 -= S0 * w;                              \
    }                                                                          \
    {                                                                          \
        VT a = VLD(src + 3 * rs), b = VLD(src + 10 * rs);           /* j=3 */  \
        VT u = a + b, w = MULI(a - b);                                         \
        P0 += u;                                                               \
        P1 += C2 * u; P2 += C5 * u; P3 += C3 * u;                              \
        P4 += C0 * u; P5 += C1 * u; P6 += C4 * u;                              \
        R1 += S2 * w; R2 += S5 * w; R3 -= S3 * w;                              \
        R4 -= S0 * w; R5 += S1 * w; R6 += S4 * w;                              \
    }                                                                          \
    {                                                                          \
        VT a = VLD(src + 4 * rs), b = VLD(src + 9 * rs);            /* j=4 */  \
        VT u = a + b, w = MULI(a - b);                                         \
        P0 += u;                                                               \
        P1 += C3 * u; P2 += C4 * u; P3 += C0 * u;                              \
        P4 += C2 * u; P5 += C5 * u; P6 += C1 * u;                              \
        R1 += S3 * w; R2 -= S4 * w; R3 -= S0 * w;                              \
        R4 += S2 * w; R5 -= S5 * w; R6 -= S1 * w;                              \
    }                                                                          \
    {                                                                          \
        VT a = VLD(src + 5 * rs), b = VLD(src + 8 * rs);            /* j=5 */  \
        VT u = a + b, w = MULI(a - b);                                         \
        P0 += u;                                                               \
        P1 += C4 * u; P2 += C2 * u; P3 += C1 * u;                              \
        P4 += C5 * u; P5 += C0 * u; P6 += C3 * u;                              \
        R1 += S4 * w; R2 -= S2 * w; R3 += S1 * w;                              \
        R4 -= S5 * w; R5 -= S0 * w; R6 += S3 * w;                              \
    }

/* row j=6 with its sine contributions accumulated into R (chunk13p) */
#define L13_ACC6R()                                                            \
    {                                                                          \
        VT a = VLD(src + 6 * rs), b = VLD(src + 7 * rs);            /* j=6 */  \
        VT u = a + b, w = MULI(a - b);                                         \
        P0 += u;                                                               \
        P1 += C5 * u; P2 += C0 * u; P3 += C4 * u;                              \
        P4 += C1 * u; P5 += C3 * u; P6 += C2 * u;                              \
        R1 += S5 * w; R2 -= S0 * w; R3 += S4 * w;                              \
        R4 -= S1 * w; R5 += S3 * w; R6 -= S2 * w;                              \
    }

static inline __attribute__((always_inline)) void
SUF(chunk13p)(const double *restrict src, long rs, double *restrict dst, long da,
              long db, VT C0, VT C1, VT C2, VT C3, VT C4, VT C5,
              VT S0, VT S1, VT S2, VT S3, VT S4, VT S5, int tr)
{
    VT P0, P1, P2, P3, P4, P5, P6, R1, R2, R3, R4, R5, R6;
    L13_ACC15();
    L13_ACC6R();
    L13_STORE_BODY();
}

/* panel_r10's association twins (chunk13q "T1", chunk13s "TS") are DELETED
 * in panel_r11: the r10 node instrument priced them at q +0.5..0.7%,
 * s +3.7..3.8% over xf, and the r10 VERDICT SS5 bounded the L=6 association
 * mechanism to "joins that feed stores" and closed the propagation ask.
 * Their derivation and numbers live in strategies/L13_direct.md r10. */

/* The whole-width execs below are not instantiated at WC==1 (the w1 kernel
 * exists only as the mixed execs' tail). */
#if WC != 1

/* chunk start offsets covering a 13-long index; the last one overlaps the
 * previous chunk and rewrites identical values */
#if WC == 4
static const int SUF(off13)[4] = {0, 4, 8, 9};
#  define NOFF13 4
#else
static const int SUF(off13)[7] = {0, 2, 4, 6, 8, 10, 11};
#  define NOFF13 7
#endif

/* ---- X-last: Y,Z per x-plane (L1-resident), then X with the long
 * sequential stores into the caller's out.  Used for batch < L13_XF_MIN. ---- */
static __attribute__((unused)) void
SUF(exec_xl)(const fft3d_plan *restrict p, const double _Complex *restrict in,
             double _Complex *restrict out)
{
    const double *restrict ct = CTAB(p);
    const double *restrict st = STAB(p);
    double *restrict pb = p->pb;
    double *restrict t1 = p->t1;
    const int nb = p->batch;
    VT K0 = CGET(st, 0), K1 = CGET(st, 1), K2 = CGET(st, 2),
       K3 = CGET(st, 3), K4 = CGET(st, 4), K5 = CGET(st, 5);
    L13_PIN_ASM();

    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);

        for (int x = 0; x < 13; ++x) {
            const double *pin = vin + (long)x * 338;
            double *pt = t1 + (long)x * L13_T1P;
            /* Y: along y (row stride 26), lanes over z, store transposed */
            for (int t = 0; t < NOFF13; ++t) {
                long f0 = SUF(off13)[t];
                SUF(chunk13)(pin + 2 * f0, 26, pb + f0 * L13_PBROW, L13_PBROW,
                             2, ct, K0, K1, K2, K3, K4, K5, 1);
            }
            /* Z: along z (row stride PBROW), lanes over ky, store transposed */
            for (int t = 0; t < NOFF13; ++t) {
                long f0 = SUF(off13)[t];
                SUF(chunk13)(pb + 2 * f0, L13_PBROW, pt + f0 * 26, 26,
                             2, ct, K0, K1, K2, K3, K4, K5, 1);
            }
        }
        /* X: along x (t1 row stride T1P, out 338), lanes over the 169
         * contiguous (ky,kz) */
        for (int i = 0; i < 169 / WC; ++i) {
            long f0 = (long)i * WC;
            SUF(chunk13)(t1 + 2 * f0, L13_T1P, vout + 2 * f0, 2, 338,
                         ct, K0, K1, K2, K3, K4, K5, 0);
        }
#if (169 % WC) != 0
        SUF(chunk13)(t1 + 2 * (169 - WC), L13_T1P, vout + 2 * (169 - WC), 2,
                     338, ct, K0, K1, K2, K3, K4, K5, 0);
#endif
    }
}

/* ---- X-first: X from in into t1, then per-kx-plane Y,Z straight into out,
 * spreading the output writes across the volume's compute (L17_matrixsimd r3;
 * wins only at streaming batch, loses ~5% cache-resident). ---- */
static __attribute__((unused)) void
SUF(exec_xf)(const fft3d_plan *restrict p, const double _Complex *restrict in,
             double _Complex *restrict out)
{
    const double *restrict ct = CTAB(p);
    const double *restrict st = STAB(p);
    double *restrict pb = p->pb;
    double *restrict t1 = p->t1;
    const int nb = p->batch;
    VT K0 = CGET(st, 0), K1 = CGET(st, 1), K2 = CGET(st, 2),
       K3 = CGET(st, 3), K4 = CGET(st, 4), K5 = CGET(st, 5);
    L13_PIN_ASM();

    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);

        for (int i = 0; i < 169 / WC; ++i) {
            long f0 = (long)i * WC;
            SUF(chunk13)(vin + 2 * f0, 338, t1 + 2 * f0, 2, L13_T1P,
                         ct, K0, K1, K2, K3, K4, K5, 0);
        }
#if (169 % WC) != 0
        SUF(chunk13)(vin + 2 * (169 - WC), 338, t1 + 2 * (169 - WC), 2,
                     L13_T1P, ct, K0, K1, K2, K3, K4, K5, 0);
#endif
        for (int x = 0; x < 13; ++x) {
            const double *pin = t1 + (long)x * L13_T1P;
            double *pt = vout + (long)x * 338;
            for (int t = 0; t < NOFF13; ++t) {
                long f0 = SUF(off13)[t];
                SUF(chunk13)(pin + 2 * f0, 26, pb + f0 * L13_PBROW, L13_PBROW,
                             2, ct, K0, K1, K2, K3, K4, K5, 1);
            }
            for (int t = 0; t < NOFF13; ++t) {
                long f0 = SUF(off13)[t];
                SUF(chunk13)(pb + 2 * f0, L13_PBROW, pt + f0 * 26, 26,
                             2, ct, K0, K1, K2, K3, K4, K5, 1);
            }
        }
    }
}

/* ---- all-pinned variants of the two pass orders (chunk13p) ---- */
static __attribute__((unused)) void
SUF(exec_xlp)(const fft3d_plan *restrict p, const double _Complex *restrict in,
              double _Complex *restrict out)
{
    const double *restrict cd = CTD(p);
    const double *restrict st = STAB(p);
    double *restrict pb = p->pb;
    double *restrict t1 = p->t1;
    const int nb = p->batch;
    VT C0 = CGET(cd, 0), C1 = CGET(cd, 1), C2 = CGET(cd, 2),
       C3 = CGET(cd, 3), C4 = CGET(cd, 4), C5 = CGET(cd, 5);
    VT S0 = CGET(st, 0), S1 = CGET(st, 1), S2 = CGET(st, 2),
       S3 = CGET(st, 3), S4 = CGET(st, 4), S5 = CGET(st, 5);
    L13_PIN12_ASM();

    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);

        for (int x = 0; x < 13; ++x) {
            const double *pin = vin + (long)x * 338;
            double *pt = t1 + (long)x * L13_T1P;
            for (int t = 0; t < NOFF13; ++t) {
                long f0 = SUF(off13)[t];
                SUF(chunk13p)(pin + 2 * f0, 26, pb + f0 * L13_PBROW, L13_PBROW,
                              2, C0, C1, C2, C3, C4, C5,
                              S0, S1, S2, S3, S4, S5, 1);
            }
            for (int t = 0; t < NOFF13; ++t) {
                long f0 = SUF(off13)[t];
                SUF(chunk13p)(pb + 2 * f0, L13_PBROW, pt + f0 * 26, 26,
                              2, C0, C1, C2, C3, C4, C5,
                              S0, S1, S2, S3, S4, S5, 1);
            }
        }
        for (int i = 0; i < 169 / WC; ++i) {
            long f0 = (long)i * WC;
            SUF(chunk13p)(t1 + 2 * f0, L13_T1P, vout + 2 * f0, 2, 338,
                          C0, C1, C2, C3, C4, C5, S0, S1, S2, S3, S4, S5, 0);
        }
#if (169 % WC) != 0
        SUF(chunk13p)(t1 + 2 * (169 - WC), L13_T1P, vout + 2 * (169 - WC), 2,
                      338, C0, C1, C2, C3, C4, C5, S0, S1, S2, S3, S4, S5, 0);
#endif
    }
}

static __attribute__((unused)) void
SUF(exec_xfp)(const fft3d_plan *restrict p, const double _Complex *restrict in,
              double _Complex *restrict out)
{
    const double *restrict cd = CTD(p);
    const double *restrict st = STAB(p);
    double *restrict pb = p->pb;
    double *restrict t1 = p->t1;
    const int nb = p->batch;
    VT C0 = CGET(cd, 0), C1 = CGET(cd, 1), C2 = CGET(cd, 2),
       C3 = CGET(cd, 3), C4 = CGET(cd, 4), C5 = CGET(cd, 5);
    VT S0 = CGET(st, 0), S1 = CGET(st, 1), S2 = CGET(st, 2),
       S3 = CGET(st, 3), S4 = CGET(st, 4), S5 = CGET(st, 5);
    L13_PIN12_ASM();

    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);

        for (int i = 0; i < 169 / WC; ++i) {
            long f0 = (long)i * WC;
            SUF(chunk13p)(vin + 2 * f0, 338, t1 + 2 * f0, 2, L13_T1P,
                          C0, C1, C2, C3, C4, C5, S0, S1, S2, S3, S4, S5, 0);
        }
#if (169 % WC) != 0
        SUF(chunk13p)(vin + 2 * (169 - WC), 338, t1 + 2 * (169 - WC), 2,
                      L13_T1P, C0, C1, C2, C3, C4, C5,
                      S0, S1, S2, S3, S4, S5, 0);
#endif
        for (int x = 0; x < 13; ++x) {
            const double *pin = t1 + (long)x * L13_T1P;
            double *pt = vout + (long)x * 338;
            for (int t = 0; t < NOFF13; ++t) {
                long f0 = SUF(off13)[t];
                SUF(chunk13p)(pin + 2 * f0, 26, pb + f0 * L13_PBROW, L13_PBROW,
                              2, C0, C1, C2, C3, C4, C5,
                              S0, S1, S2, S3, S4, S5, 1);
            }
            for (int t = 0; t < NOFF13; ++t) {
                long f0 = SUF(off13)[t];
                SUF(chunk13p)(pb + 2 * f0, L13_PBROW, pt + f0 * 26, 26,
                              2, C0, C1, C2, C3, C4, C5,
                              S0, S1, S2, S3, S4, S5, 1);
            }
        }
    }
}

#undef NOFF13
#endif /* WC != 1 */
#undef CTAB
#undef STAB
#undef CTD
#undef CSTEP_C
#undef CGET
#undef L13_PIN_ASM
#undef L13_PIN12_ASM
#undef L13_STORE_BODY
#undef L13_ACC15
#undef L13_ACC6R
#undef XP_
#undef XM_
#undef LASTCOL_
#if WC == 4
#  undef TILE4_
#elif WC == 2
#  undef TILE2_
#endif
#undef MULI
#undef SWAPRI
#undef TTILE
#undef SHUF1
#undef SHUF2
#undef VLD
#undef VST
#undef VT
#undef IT
#undef VDW

#else /* !L13_TEMPLATE_PASS =================================================== */

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
 * 64-byte aligned relative to the base; 13 rows = 3.3 KiB, L1-resident) */
enum { L13_PBROW = 32 };

/* t1 x-plane stride, in doubles: 169 complex (338) padded to 344 = 2752 B =
 * 43 cache lines (panel_r8; L23_rader panel_r6's padding rule applied to the
 * one big stride this plan owns).  At the driver's natural 338 (2704 B =
 * 42.25 lines) every X-pass vector access to t1 at plane j sits at byte
 * 16*j (mod 64), so 3/4 of the X pass's zmm loads (X-last) or stores
 * (X-first) SPLIT a cache line.  344 doubles makes them all 64-byte aligned,
 * and 2752 mod 4096 differs from in/out's 2704 residue comb, so the X pass's
 * t1 stream can no longer 4K-alias the driver-buffer stream it runs against.
 * The pad tail (doubles 338..343 of each plane) is never read. */
enum { L13_T1P = 344 };

struct fft3d_plan {
    int L, batch;
    void (*exec)(const struct fft3d_plan *, const double _Complex *, double _Complex *);
    double *ctab8; /* 6 rows x 7 cosines, j-major, each splatted 8x (zmm) */
    double *stab8; /* s_1..s_6, each splatted 8x (zmm)                    */
    double *ctd8;  /* the 6 DISTINCT cosines c_1..c_6, splatted 8x (zmm)  */
    double *ctab4; /* the same three, splatted 4x (ymm)                   */
    double *stab4;
    double *ctd4;
    double *pb;    /* 13 x PBROW plane buffer (pass Y -> pass Z)          */
    double *t1;    /* one 13^3 complex volume of scratch, x-planes at T1P */
    double *sb;    /* 13x13 hot staging plane (pass Z -> burst memcpy)    */
    void *block;
};

/* ---- instantiate the kernel at 512-bit and 256-bit ---- */
#if defined(__has_include)
#  if __has_include("L13_direct.c")
#    define L13_SELF "L13_direct.c"
#  elif __has_include("impl/L13_direct.c")
#    define L13_SELF "impl/L13_direct.c"
#  endif
#else
#  define L13_SELF "L13_direct.c"
#endif

#ifdef L13_SELF
#  define L13_TEMPLATE_PASS 1
#  define WC 4
#  define SUF(x) x##_w4
#  include L13_SELF
#  undef SUF
#  undef WC
#  undef L13_TEMPLATE_PASS

#  define L13_TEMPLATE_PASS 2
#  define WC 2
#  define SUF(x) x##_w2
#  include L13_SELF
#  undef SUF
#  undef WC
#  undef L13_TEMPLATE_PASS

/* panel_r11: xmm width, kernels only -- the mixed execs' tail chunks */
#  define L13_TEMPLATE_PASS 3
#  define WC 1
#  define SUF(x) x##_w1
#  include L13_SELF
#  undef SUF
#  undef WC
#  undef L13_TEMPLATE_PASS
#else
#  error "L13_direct.c must be able to #include itself"
#endif

/* =============================================================================
 * Mixed-width execs: 512-bit chunk body, 256-bit (ymm) tail chunks.
 * Adopted from L17_rader panel_r4 ("512t") via L17_matrixsimd panel_r5: the
 * scoring node's Gold 5218 has ONE fused 512-bit FMA unit but TWO 256-bit FMA
 * ports, so a ymm chunk retires its 102 FP ops in ~51 cycles where a zmm chunk
 * needs ~102.  A 13-long index costs 3 zmm + 1 ymm tail (offsets 0,4,8 + 11,
 * one line recomputed) instead of 4 zmm (0,4,8,9, three recomputed); the X
 * pass 42 zmm + 1 ymm at 167.  On a 2x512-bit-FMA machine (wallaby) the mix
 * is FP-neutral and slightly worse on instruction count: it is a node bet.
 * Only the zmm K set is asm-pinned; the ymm Q set spilling a little on 1
 * chunk in 4 is cheaper than holding 12 of 32 registers everywhere.
 * =============================================================================
 */
#if defined(__AVX512F__)
#  define L13_PIN_W4M() __asm__("" : "+v"(K0), "+v"(K1), "+v"(K2), "+v"(K3),   \
                                     "+v"(K4), "+v"(K5))
#else
#  define L13_PIN_W4M() do { } while (0)
#endif

#define L13_MIX_DECLS                                                          \
    const double *restrict ct8 = p->ctab8, *restrict st8 = p->stab8;           \
    const double *restrict ct4 = p->ctab4, *restrict st4 = p->stab4;           \
    double *restrict pb = p->pb;                                               \
    double *restrict t1 = p->t1;                                               \
    const int nb = p->batch;                                                   \
    vd_w4 K0 = *(const vd_w4 *)(st8 + 0),  K1 = *(const vd_w4 *)(st8 + 8),     \
          K2 = *(const vd_w4 *)(st8 + 16), K3 = *(const vd_w4 *)(st8 + 24),    \
          K4 = *(const vd_w4 *)(st8 + 32), K5 = *(const vd_w4 *)(st8 + 40);    \
    vd_w2 Q0 = *(const vd_w2 *)(st4 + 0),  Q1 = *(const vd_w2 *)(st4 + 4),     \
          Q2 = *(const vd_w2 *)(st4 + 8),  Q3 = *(const vd_w2 *)(st4 + 12),    \
          Q4 = *(const vd_w2 *)(st4 + 16), Q5 = *(const vd_w2 *)(st4 + 20);    \
    L13_PIN_W4M()

/* Mixed decls for the ALL-PINNED zmm kernel: 12 zmm constants pinned; the
 * ymm tail runs the all-pinned w2 kernel with its own UNPINNED 12 ymm
 * constants (panel_r9 -- previously the tail ran the table-cosine kernel at
 * 42 ct4 loads + 6 Q reloads per chunk; the unpinned D/Q set spills, but 12
 * spill-reloads per tail chunk beat 48 loads, a pure load deletion, and NOT
 * pinning them keeps the zmm body's registers intact).  Bit-exact swap: the
 * two kernels share the fold, the accumulation order, and the store body
 * (the table kernel's k=0 row is FMA(1.0,u,P0), exact). */
#if defined(__AVX512F__)
#  define L13_PIN12_W4M() __asm__("" : "+v"(C0), "+v"(C1), "+v"(C2), "+v"(C3),\
                                       "+v"(C4), "+v"(C5), "+v"(K0), "+v"(K1),\
                                       "+v"(K2), "+v"(K3), "+v"(K4), "+v"(K5))
#else
#  define L13_PIN12_W4M() do { } while (0)
#endif

/* ---- streaming prefetch (panel_r8) --------------------------------------
 * prefetchw on the out stream, one plane ahead of the Z-pass stores that
 * will RFO it (adopted from the r7 verdict rule "hide the RFO with
 * prefetchw, do not avoid it with NT stores" -- node-selected at every
 * streaming cell at L=6/8/23/36/64; the L=13-local evidence is L13_rader
 * panel_r7's pw A/B at wallaby B=2048: -6.0%).  Plus a t1-hint read
 * prefetch of the NEXT volume's input, paced one plane-slice per plane
 * (L17_winograd r2's cross-volume prefetch, -4.4% streaming; L64_radix8's
 * node-picked slabpf).  Both fire only from the _pf exec, which create()
 * selects only when in+out exceed this host's L3 (L13_rader r7 measured
 * pfw at L3-resident batch as a ~3% LOSS; same gate).  -DL13_PW=0 /
 * -DL13_PFIN=0 kill either half for node A/Bs. */
#ifndef L13_PW
#  define L13_PW 1
#endif
#ifndef L13_PFIN
#  define L13_PFIN 1
#endif
#if L13_PW
#  define L13_PW1(P, OFF)                                                      \
      do { if (P) __builtin_prefetch((const char *)(P) + (OFF), 1, 3); } while (0)
#else
#  define L13_PW1(P, OFF) do { (void)(P); } while (0)
#endif

/* a 13x13 complex plane at row stride 338 doubles spans 43 line boundaries
 * for every base offset the driver's layout produces (16x mod 64 <= 48) */
static inline void l13_pfw43(const double *p)
{
#if L13_PW
    for (int i = 0; i < 43; ++i)
        __builtin_prefetch((const char *)p + 64 * i, 1, 3);
#else
    (void)p;
#endif
}
static inline void l13_pfr43(const double *p)
{
#if L13_PFIN
    for (int i = 0; i < 43; ++i)
        __builtin_prefetch((const char *)p + 64 * i, 0, 2);
#else
    (void)p;
#endif
}

#define L13_MIXP_DECLS                                                         \
    const double *restrict cd8 = p->ctd8, *restrict st8 = p->stab8;            \
    const double *restrict cd4 = p->ctd4, *restrict st4 = p->stab4;            \
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
    /* panel_r11: xmm-tail constants -- lane pair (c,c) / (s,-s) is the first  \
     * 16 B of each 4-splat table row.  Unpinned like the w2 set; whichever    \
     * width an exec's tails do not use is a dead load gcc deletes. */         \
    vd_w1 E0 = *(const vd_w1 *)(cd4 + 0),  E1 = *(const vd_w1 *)(cd4 + 4),     \
          E2 = *(const vd_w1 *)(cd4 + 8),  E3 = *(const vd_w1 *)(cd4 + 12),    \
          E4 = *(const vd_w1 *)(cd4 + 16), E5 = *(const vd_w1 *)(cd4 + 20);    \
    vd_w1 F0 = *(const vd_w1 *)(st4 + 0),  F1 = *(const vd_w1 *)(st4 + 4),     \
          F2 = *(const vd_w1 *)(st4 + 8),  F3 = *(const vd_w1 *)(st4 + 12),    \
          F4 = *(const vd_w1 *)(st4 + 16), F5 = *(const vd_w1 *)(st4 + 20);    \
    L13_PIN12_W4M()

/* One 13-long free index, transposing store: 3 zmm chunks at 0,4,8 and one
 * ymm tail at 11 (lines 11,12; line 11 recomputed).  The asm-opaque bound
 * keeps the zmm loop rolled (gcc 11 ignores `#pragma GCC unroll 1` around an
 * always_inline callee -- L17_rader r4). */
#define L13_MIX_GROUP(SRCB, RS, DSTB, DA)                                      \
    do {                                                                       \
        int nt_ = 3;                                                           \
        __asm__("" : "+r"(nt_));                                               \
        for (int t_ = 0; t_ < nt_; ++t_) {                                     \
            long f0_ = 4L * t_;                                                \
            chunk13_w4((SRCB) + 2 * f0_, (RS), (DSTB) + f0_ * (DA), (DA), 2,   \
                       ct8, K0, K1, K2, K3, K4, K5, 1);                        \
        }                                                                      \
        chunk13_w2((SRCB) + 2 * 11, (RS), (DSTB) + 11 * (DA), (DA), 2,         \
                   ct4, Q0, Q1, Q2, Q3, Q4, Q5, 1);                            \
    } while (0)

/* The X pass, mixed: 42 zmm chunks cover lines 0..167 of the 169-long free
 * index, one ymm tail at 167 covers 167,168 (167 recomputed).  SRS/DRS are
 * the x-row strides of the two buffers (338 for in/out, L13_T1P for t1). */
#define L13_MIX_XPASS(SRCB, SRS, DSTB, DRS)                                    \
    do {                                                                       \
        int nx_ = 42;                                                          \
        __asm__("" : "+r"(nx_));                                               \
        for (int i_ = 0; i_ < nx_; ++i_) {                                     \
            long f0_ = 4L * i_;                                                \
            chunk13_w4((SRCB) + 2 * f0_, (SRS), (DSTB) + 2 * f0_, 2, (DRS),    \
                       ct8, K0, K1, K2, K3, K4, K5, 0);                        \
        }                                                                      \
        chunk13_w2((SRCB) + 2 * 167, (SRS), (DSTB) + 2 * 167, 2, (DRS),        \
                   ct4, Q0, Q1, Q2, Q3, Q4, Q5, 0);                            \
    } while (0)

static __attribute__((unused)) void
l13_exec_xl_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
               double _Complex *restrict out)
{
    L13_MIX_DECLS;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);
        for (int x = 0; x < 13; ++x) {
            const double *pin = vin + (long)x * 338;
            double *pt = t1 + (long)x * L13_T1P;
            L13_MIX_GROUP(pin, 26, pb, L13_PBROW);
            L13_MIX_GROUP(pb, L13_PBROW, pt, 26);
        }
        L13_MIX_XPASS(t1, L13_T1P, vout, 338);
    }
}

static __attribute__((unused)) void
l13_exec_xf_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
               double _Complex *restrict out)
{
    L13_MIX_DECLS;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);
        L13_MIX_XPASS(vin, 338, t1, L13_T1P);
        for (int x = 0; x < 13; ++x) {
            const double *pin = t1 + (long)x * L13_T1P;
            double *pt = vout + (long)x * 338;
            L13_MIX_GROUP(pin, 26, pb, L13_PBROW);
            L13_MIX_GROUP(pb, L13_PBROW, pt, 26);
        }
    }
}

/* ---- the same two pass orders with the ALL-PINNED zmm kernel ----
 * panel_r11: the tail chunk is now the WC==1 (xmm) kernel at offset 12/168
 * covering the ONE line the 3/42 zmm chunks leave, instead of a ymm chunk at
 * 11/167 recomputing a line to deliver it.  Port-identical (102 xmm ops pair
 * on two ports = 51 cycles, exactly the ymm tail's cost) and bit-identical
 * (same DAG per lane, same constant values), and a pure deletion of tail
 * split accesses: 16 B ops on 16 B-aligned complex addresses NEVER split a
 * cache line, where the ymm tails sat at byte 48 (mod 64) -- every Z-group
 * tail load from pb split (169/volume), the X tail's stores split (13), the
 * Y tails' loads split (~52).  It also deletes the ymm-tail store-forward
 * BLOCK: the Z group's 32 B tail load spanned the Y group's zmm tile store
 * AND its 16 B LASTCOL_ store (~169 blocked loads/volume at the plane-hot pb
 * junction -- the standing suspect for the B=1 residue, r8/r9 "next").
 *
 * BUT the xmm tail is only safe where nothing soon LOADS WIDE what it
 * stored.  In the Y position it writes pb row 12 as 13 xmm stores, and the
 * Z group's three zmm chunks each load that row 64 B at a time -- a load
 * spanning four narrow stores cannot forward, and on wallaby/SPR that form
 * (FORCE=9, "xt") measured a catastrophe: +44% at B=1, +17% at B=16, +29%
 * at B=512, driver-level pinned.  So the DEFAULT pairs the xmm tail (Z and
 * X positions, whose outputs go to `out`/t1 and are not wide-loaded hot)
 * with the ZSOLID Y group below.  The r10 ymm-tail shape is kept verbatim
 * as the _Y2 macros (FORCE=13 and the discriminator) so the node can price
 * both changes directly. */
#define L13_MIXP_GROUP(SRCB, RS, DSTB, DA)                                     \
    do {                                                                       \
        int nt_ = 3;                                                           \
        __asm__("" : "+r"(nt_));                                               \
        for (int t_ = 0; t_ < nt_; ++t_) {                                     \
            long f0_ = 4L * t_;                                                \
            chunk13p_w4((SRCB) + 2 * f0_, (RS), (DSTB) + f0_ * (DA), (DA), 2,  \
                C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5, 1);            \
        }                                                                      \
        chunk13p_w1((SRCB) + 2 * 12, (RS), (DSTB) + 12 * (DA), (DA), 2,        \
            E0, E1, E2, E3, E4, E5, F0, F1, F2, F3, F4, F5, 1);                \
    } while (0)

/* zsolid Y group (panel_r11, THE DEFAULT's Y position): 4 zmm chunks at
 * 0,4,8,9 -- the overlap chunk rewrites rows 9..11 with identical bits and
 * delivers row 12 as full 64 B tile stores, so EVERY Z-group load out of pb
 * (zmm or xmm) is exactly contained in one Y store: ZERO store-forward
 * blocks at the pb junction, by construction.  Costs +51 port cycles/plane
 * (+663/volume, only on the node's single 512-bit unit -- on wallaby's two
 * units a 4th zmm chunk and a tail cost the same 51 cycles).  Wallaby,
 * pinned, disjoint distributions: beats the r10 shape -1.6/-2.5/-1.8% at
 * B=1/16/512.  Node bet: SF stalls are latency, not port pressure, and B=1
 * sits 1.21x over its port floor with ~2.9k cycles of slack. */
#define L13_MIXP_GROUP_ZS(SRCB, RS, DSTB, DA)                                  \
    do {                                                                       \
        int nt_ = 4;                                                           \
        __asm__("" : "+r"(nt_));                                               \
        for (int t_ = 0; t_ < nt_; ++t_) {                                     \
            long f0_ = (t_ == 3) ? 9L : 4L * t_;                               \
            chunk13p_w4((SRCB) + 2 * f0_, (RS), (DSTB) + f0_ * (DA), (DA), 2,  \
                C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5, 1);            \
        }                                                                      \
    } while (0)

/* the r10 default's ymm tail, verbatim (rollback + discriminator reference) */
#define L13_MIXP_GROUP_Y2(SRCB, RS, DSTB, DA)                                  \
    do {                                                                       \
        int nt_ = 3;                                                           \
        __asm__("" : "+r"(nt_));                                               \
        for (int t_ = 0; t_ < nt_; ++t_) {                                     \
            long f0_ = 4L * t_;                                                \
            chunk13p_w4((SRCB) + 2 * f0_, (RS), (DSTB) + f0_ * (DA), (DA), 2,  \
                C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5, 1);            \
        }                                                                      \
        chunk13p_w2((SRCB) + 2 * 11, (RS), (DSTB) + 11 * (DA), (DA), 2,        \
            D0, D1, D2, D3, D4, D5, Q0, Q1, Q2, Q3, Q4, Q5, 1);                \
    } while (0)

/* PFW0: optional prefetchw target (the first out plane, one pfw line per
 * chunk = perfectly paced); pass 0 to elide. */
#define L13_MIXP_XPASS(SRCB, SRS, DSTB, DRS, PFW0)                             \
    do {                                                                       \
        int nx_ = 42;                                                          \
        __asm__("" : "+r"(nx_));                                               \
        for (int i_ = 0; i_ < nx_; ++i_) {                                     \
            long f0_ = 4L * i_;                                                \
            chunk13p_w4((SRCB) + 2 * f0_, (SRS), (DSTB) + 2 * f0_, 2, (DRS),   \
                C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5, 0);            \
            L13_PW1((PFW0), 64 * i_);                                          \
        }                                                                      \
        chunk13p_w1((SRCB) + 2 * 168, (SRS), (DSTB) + 2 * 168, 2, (DRS),       \
            E0, E1, E2, E3, E4, E5, F0, F1, F2, F3, F4, F5, 0);                \
        L13_PW1((PFW0), 64 * 42);                                              \
    } while (0)

#define L13_MIXP_XPASS_Y2(SRCB, SRS, DSTB, DRS, PFW0)                          \
    do {                                                                       \
        int nx_ = 42;                                                          \
        __asm__("" : "+r"(nx_));                                               \
        for (int i_ = 0; i_ < nx_; ++i_) {                                     \
            long f0_ = 4L * i_;                                                \
            chunk13p_w4((SRCB) + 2 * f0_, (SRS), (DSTB) + 2 * f0_, 2, (DRS),   \
                C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5, 0);            \
            L13_PW1((PFW0), 64 * i_);                                          \
        }                                                                      \
        chunk13p_w2((SRCB) + 2 * 167, (SRS), (DSTB) + 2 * 167, 2, (DRS),       \
            D0, D1, D2, D3, D4, D5, Q0, Q1, Q2, Q3, Q4, Q5, 0);                \
        L13_PW1((PFW0), 64 * 42);                                              \
    } while (0)

static __attribute__((unused)) void
l13_exec_xlzs_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                 double _Complex *restrict out)
{
    L13_MIXP_DECLS;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);
        for (int x = 0; x < 13; ++x) {
            const double *pin = vin + (long)x * 338;
            double *pt = t1 + (long)x * L13_T1P;
            L13_MIXP_GROUP_ZS(pin, 26, pb, L13_PBROW);
            L13_MIXP_GROUP(pb, L13_PBROW, pt, 26);
        }
        L13_MIXP_XPASS(t1, L13_T1P, vout, 338, (double *)0);
    }
}

static __attribute__((unused)) void
l13_exec_xfp_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                double _Complex *restrict out)
{
    L13_MIXP_DECLS;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);
        L13_MIXP_XPASS(vin, 338, t1, L13_T1P, (double *)0);
        for (int x = 0; x < 13; ++x) {
            const double *pin = t1 + (long)x * L13_T1P;
            double *pt = vout + (long)x * 338;
            L13_MIXP_GROUP(pin, 26, pb, L13_PBROW);
            L13_MIXP_GROUP(pb, L13_PBROW, pt, 26);
        }
    }
}

/* ---- xfp_y2: the r10 default verbatim (ymm tails), FORCE=13 and the
 * discriminator's y2 slot -- prices this round's change on the node.
 * xfzs: THE DEFAULT (ws <= L3): zsolid Y groups + xmm Z/X tails.
 * Both bit-identical to l13_exec_xfp_mx. ---- */
static __attribute__((unused)) void
l13_exec_xfp_y2_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                   double _Complex *restrict out)
{
    L13_MIXP_DECLS;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);
        L13_MIXP_XPASS_Y2(vin, 338, t1, L13_T1P, (double *)0);
        for (int x = 0; x < 13; ++x) {
            const double *pin = t1 + (long)x * L13_T1P;
            double *pt = vout + (long)x * 338;
            L13_MIXP_GROUP_Y2(pin, 26, pb, L13_PBROW);
            L13_MIXP_GROUP_Y2(pb, L13_PBROW, pt, 26);
        }
    }
}

static __attribute__((unused)) void
l13_exec_xfzs_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                 double _Complex *restrict out)
{
    L13_MIXP_DECLS;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);
        L13_MIXP_XPASS(vin, 338, t1, L13_T1P, (double *)0);
        for (int x = 0; x < 13; ++x) {
            const double *pin = t1 + (long)x * L13_T1P;
            double *pt = vout + (long)x * 338;
            L13_MIXP_GROUP_ZS(pin, 26, pb, L13_PBROW);
            L13_MIXP_GROUP(pb, L13_PBROW, pt, 26);
        }
    }
}

/* X-first with the streaming prefetch schedule (panel_r8; see the comment at
 * l13_pfw43).  Per plane x: prefetchw the NEXT out plane before the Y group
 * (lead time = one Y+Z group, ~700 node cycles); read-prefetch plane-slice x
 * of the next volume's input between the groups.  Plane 0's out lines are
 * prefetched from inside the X pass, one line per chunk.  43 lines per
 * plane-slice covers the volume's 550 lines with margin; prefetch never
 * faults, so running a few lines past the last volume's tail is harmless. */
static __attribute__((unused)) void
l13_exec_xfzs_pf_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                    double _Complex *restrict out)
{
    L13_MIXP_DECLS;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);
        const double *vnx =
            (b + 1 < nb) ? (const double *)(in + (size_t)(b + 1) * 2197) : 0;
        L13_MIXP_XPASS(vin, 338, t1, L13_T1P, vout);
        for (int x = 0; x < 13; ++x) {
            const double *pin = t1 + (long)x * L13_T1P;
            double *pt = vout + (long)x * 338;
            if (x < 12) l13_pfw43(vout + (long)(x + 1) * 338);
            L13_MIXP_GROUP_ZS(pin, 26, pb, L13_PBROW);
            if (vnx) l13_pfr43(vnx + (long)x * 43 * 8);
            L13_MIXP_GROUP(pb, L13_PBROW, pt, 26);
        }
    }
}

/* X-first with the Z pass STAGED: the transposing tile stores land in a hot
 * 2.7 KB plane, which is then burst-copied sequentially into out (adopted
 * from L13_rader panel_r6 / L8_radix8 via L17_rader r4: kernel-store
 * patterns into a cold/streaming `out` are the expensive thing; a sequential
 * memcpy of full lines is the cheap way to touch it). */
static __attribute__((unused)) void
l13_exec_xsp_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                double _Complex *restrict out)
{
    L13_MIXP_DECLS;
    double *restrict sb = p->sb;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);
        L13_MIXP_XPASS(vin, 338, t1, L13_T1P, (double *)0);
        for (int x = 0; x < 13; ++x) {
            const double *pin = t1 + (long)x * L13_T1P;
            /* ZS in BOTH positions: the memcpy's 64 B reads of sb would
             * otherwise straddle an xmm-tail row (the same SF-block the
             * zsolid group exists to delete) */
            L13_MIXP_GROUP_ZS(pin, 26, pb, L13_PBROW);
            L13_MIXP_GROUP_ZS(pb, L13_PBROW, sb, 26);
            memcpy(vout + (long)x * 338, sb, 338 * sizeof(double));
        }
    }
}

/* staged Z + the full streaming prefetch schedule (FORCE=12): pfw still pays
 * under the memcpy's RFOs even though the kernel stores land in hot sb. */
static __attribute__((unused)) void
l13_exec_xsp_pf_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                   double _Complex *restrict out)
{
    L13_MIXP_DECLS;
    double *restrict sb = p->sb;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 2197);
        double *vout = (double *)(out + (size_t)b * 2197);
        const double *vnx =
            (b + 1 < nb) ? (const double *)(in + (size_t)(b + 1) * 2197) : 0;
        L13_MIXP_XPASS(vin, 338, t1, L13_T1P, vout);
        for (int x = 0; x < 13; ++x) {
            const double *pin = t1 + (long)x * L13_T1P;
            if (x < 12) l13_pfw43(vout + (long)(x + 1) * 338);
            L13_MIXP_GROUP_ZS(pin, 26, pb, L13_PBROW);
            if (vnx) l13_pfr43(vnx + (long)x * 43 * 8);
            L13_MIXP_GROUP_ZS(pb, L13_PBROW, sb, 26);
            memcpy(vout + (long)x * 338, sb, 338 * sizeof(double));
        }
    }
}

/* ---------------------------------------------------------------------------
 * plumbing
 * ---------------------------------------------------------------------------
 */
static char g_desc[384] =
    "conj-folded dense 13x13 per axis, lanes=lines, pinned sines";

const char *fft3d_name(void) { return "L13_direct"; }
const char *fft3d_description(void) { return g_desc; }
int fft3d_supports(int L) { return L == 13; }

#define L13_ALIGN64(q) ((double *)(((uintptr_t)(q) + 63u) & ~(uintptr_t)63u))

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != 13 || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = 13;
    p->batch = batch;

    /* ctab8 6*7*8 + stab8 6*8 + ctd8 6*8 + ctab4 6*7*4 + stab4 6*4 + ctd4 6*4
     * + pb 13*PBROW + sb 344 + t1 13*T1P (padded planes) */
    size_t nd = 6 * 7 * 8 + 6 * 8 + 6 * 8 + 6 * 7 * 4 + 6 * 4 + 6 * 4
                + 13 * L13_PBROW + 344 + 13 * L13_T1P + 48;
    p->block = malloc(nd * sizeof(double) + 64);
    if (!p->block) { free(p); return NULL; }
    double *q = L13_ALIGN64((double *)p->block);
    p->ctab8 = q; q += 6 * 7 * 8;
    p->stab8 = q; q += 6 * 8;
    p->ctd8 = q;  q += 6 * 8;
    p->ctab4 = q; q += 6 * 7 * 4;
    p->stab4 = q; q += 6 * 4;
    p->ctd4 = q;  q += 6 * 4;
    p->pb = q;    q += 13 * L13_PBROW;
    p->sb = q;    q += 344;
    p->t1 = q;

    const long double PI2 = 6.283185307179586476925286766559L;
    for (int j = 1; j <= 6; ++j)
        for (int k = 0; k <= 6; ++k) {
            double c = (double)cosl(PI2 * (long double)((k * j) % 13) / 13.0L);
            for (int i = 0; i < 8; ++i) p->ctab8[((j - 1) * 7 + k) * 8 + i] = c;
            for (int i = 0; i < 4; ++i) p->ctab4[((j - 1) * 7 + k) * 4 + i] = c;
        }
    /* sine tables are LANE-ALTERNATING (s at re lanes, -s at im lanes): the
     * -i sign of MULI lives here now, not in a per-chunk XOR (panel_r9). */
    for (int m = 1; m <= 6; ++m) {
        double s = (double)sinl(PI2 * (long double)m / 13.0L);
        double c = (double)cosl(PI2 * (long double)m / 13.0L);
        for (int i = 0; i < 8; ++i) p->stab8[(m - 1) * 8 + i] = (i & 1) ? -s : s;
        for (int i = 0; i < 4; ++i) p->stab4[(m - 1) * 4 + i] = (i & 1) ? -s : s;
        for (int i = 0; i < 8; ++i) p->ctd8[(m - 1) * 8 + i] = c;
        for (int i = 0; i < 4; ++i) p->ctd4[(m - 1) * 4 + i] = c;
    }

    /* Deterministic selection: a pure function of batch size and compile-time
     * ISA, so repeated runs are bit-identical by construction (no wall-clock
     * tuner; see the header block). */
    /* ...from the batch working set (in+out bytes) against THIS machine's L2,
     * read once via sysconf (adopted from L13_rader panel_r6: a fixed batch
     * threshold tuned on wallaby's 2 MB L2 is wrong on the node's 1 MB;
     * sysconf is a fixed property of the host, so runs stay bit-identical).
     *   ws <= L2 : everything cache-hot -> X-last (plain kernel stores)
     *   ws >  L2 : out stores leave L2  -> X-first (stores in a plane window)
     * Node (1 MB): B=1 X-last, B=16/512 X-first.  Wallaby (2 MB): B=1/16
     * X-last, B=512 X-first.  The staged-Z variant (FORCE=10) measured +16%
     * on wallaby's L3-resident B=512 and is NOT shipped as a default; it is
     * kept for the monitor to A/B on the node's truly streaming B=512. */
    /* Three tiers now (panel_r8): past L3 the out stores truly stream and
     * the pf exec's prefetchw/read-prefetch schedule turns on; between L2
     * and L3 prefetch is a measured loss (L13_rader r7: pfw at L3-resident
     * batch ~ -3%, input pf -4%) and plain X-first stays. */
    long l2c = sysconf(_SC_LEVEL2_CACHE_SIZE);
    if (l2c <= 0) l2c = 1 << 20;
    long l3c = sysconf(_SC_LEVEL3_CACHE_SIZE);
    if (l3c <= 0) l3c = 22l << 20;
    unsigned long long ws = 32ull * (unsigned long long)batch * 2197ull;
    (void)l2c;
    const char *pick;
    /* panel_r10: the cache-resident X-LAST tier is GONE.  Its justification
     * was r6's "X-first +28% at B=1" -- measured BEFORE the r8 t1 pad, whose
     * whole point was to fix the X pass's split/4K-aliased t1 accesses, i.e.
     * exactly the accesses that penalised X-first.  Re-measured pinned on
     * wallaby (driver-level, interleaved, min over 8 instances): X-first
     * 2.571 vs X-last 3.021 us at B=1 (-15%), 41.6 vs 50.0 us/call at B=16
     * (-17%).  Node evidence pending; FORCE=8 is the X-last rollback and the
     * in-plan discriminator below prints the node's own xl-vs-xf on every
     * leaderboard line. */
    /* panel_r11: the default shape is ZSOLID-Y + xmm tails (see the macro
     * comments): Y groups pure zmm so pb is written 100% as 64 B tiles (zero
     * store-forward blocks at the pb junction), Z/X tails xmm (zero split
     * accesses, no recompute).  Wallaby, pinned, min over >=6 interleaved
     * instances: 2.535/40.47/1578 vs the r10 shape's 2.575/41.50/1607 at
     * B=1/16(call)/512(call).  The all-xmm-tail form (FORCE=9) LOST big
     * there (+17..29%: Z's 64 B loads straddle the Y tail's 13 xmm stores);
     * kept in the discriminator so the node prices both. */
#if defined(__AVX512F__)
    if (ws > (unsigned long long)l3c) { p->exec = l13_exec_xfzs_pf_mx; pick = "512b all-pinned zsolidY+xmm-tail X-first+pf"; }
    else                              { p->exec = l13_exec_xfzs_mx; pick = "512b all-pinned zsolidY+xmm-tail X-first"; }
#else
    if (ws > (unsigned long long)l2c) { p->exec = exec_xf_w2; pick = "256b X-first"; }
    else                              { p->exec = exec_xl_w2; pick = "256b X-last"; }
#endif

#if defined(L13_FORCE)
    switch ((int)L13_FORCE) {
    case 0: p->exec = exec_xl_w4;     pick = "FORCED 512b table X-last"; break;
    case 1: p->exec = exec_xf_w4;     pick = "FORCED 512b table X-first"; break;
    case 2: p->exec = exec_xl_w2;     pick = "FORCED 256b X-last"; break;
    case 3: p->exec = exec_xf_w2;     pick = "FORCED 256b X-first"; break;
    case 4: p->exec = l13_exec_xl_mx; pick = "FORCED 512b table+ymm-tail X-last"; break;
    case 5: p->exec = l13_exec_xf_mx; pick = "FORCED 512b table+ymm-tail X-first"; break;
    case 6: p->exec = exec_xlp_w4;    pick = "FORCED 512b all-pinned pure X-last"; break;
    case 7: p->exec = exec_xfp_w4;    pick = "FORCED 512b all-pinned pure X-first"; break;
    case 8: p->exec = l13_exec_xlzs_mx; pick = "FORCED 512b zsolidY+xmm-tail X-last"; break;
    case 9: p->exec = l13_exec_xfp_mx; pick = "FORCED 512b all-xmm-tail X-first (r11 failed form)"; break;
    case 10: p->exec = l13_exec_xsp_mx; pick = "FORCED 512b zsolid X-first staged"; break;
    case 11: p->exec = l13_exec_xfzs_pf_mx; pick = "FORCED 512b zsolidY+xmm-tail X-first+pf"; break;
    case 12: p->exec = l13_exec_xsp_pf_mx; pick = "FORCED 512b zsolid X-first staged+pf"; break;
    /* panel_r11: 13-16 (the association twins) are retired; 13/14 reused */
    case 13: p->exec = l13_exec_xfp_y2_mx; pick = "FORCED 512b all-pinned+ymm-tail(r10) X-first"; break;
    case 14: p->exec = l13_exec_xfzs_mx; pick = "FORCED 512b zsolidY+xmm-tail X-first (=default)"; break;
    default: break;
    }
#endif
    snprintf(g_desc, sizeof g_desc,
             "conj-folded dense 13x13 per axis, lanes=lines, pinned sines; %s",
             pick);

    /* ---- in-plan timed discriminator (INSTRUMENT ONLY: it never changes
     * the pick, so the exec stays a pure function of batch/ISA/sysconf and
     * there is no pick lottery across the monitor's processes).  Pattern
     * adopted from L6_unrolled r9's ab1 <- L36_pfa r8.  panel_r11 slots (the
     * r10 q/s twins were answered by the node -- q +0.5..0.7%, s +3.7..3.8%
     * -- and are deleted); races at tb = min(batch,16) volumes on private
     * buffers, all four bit-compatible with the shipped X-first defaults:
     *   y2 : the r10 default verbatim (ymm tails) -- prices this round
     *   zs : this round's default (zsolid Y + xmm tails)
     *   xt : all-xmm tails (LOST +17..29% on wallaby/SPR; does CLX invert?)
     *   xl : X-last, zsolid (pass-order continuity reading)
     * Timing: interleaved round-robin, min of 9 trials, licence-warmed;
     * min is the statistic (every entry's records agree).  ~10 ms, unscored.
     * -DL13_AB=0 removes it entirely. */
#ifndef L13_AB
#  define L13_AB 1
#endif
#if L13_AB && defined(__AVX512F__)
    {
        int tb = batch < 16 ? batch : 16;
        size_t vs = ((size_t)tb * 2197 * 16 + 63) & ~(size_t)63;
        double _Complex *ti = aligned_alloc(64, vs);
        double _Complex *to = aligned_alloc(64, vs);
        if (ti && to) {
            uint64_t s = 0x9E3779B97F4A7C15ull;
            double *d = (double *)ti;
            for (size_t i = 0; i < (size_t)tb * 2197 * 2; ++i) {
                s = s * 6364136223846793005ull + 1442695040888963407ull;
                d[i] = (double)(int64_t)(s >> 17) * 0x1p-40;
            }
            typedef void (*l13_fn)(const fft3d_plan *, const double _Complex *,
                                   double _Complex *);
            l13_fn cf[4];
            double best[4] = {1e300, 1e300, 1e300, 1e300};
            cf[0] = l13_exec_xfp_y2_mx;
            cf[1] = l13_exec_xfzs_mx;
            cf[2] = l13_exec_xfp_mx;
            cf[3] = l13_exec_xlzs_mx;
            fft3d_plan tp = *p;
            tp.batch = tb;
            int reps = tb >= 8 ? 2 : 24;
            for (int c = 0; c < 4; ++c)                    /* licence warm */
                for (int r = 0; r < 4 * reps; ++r) cf[c](&tp, ti, to);
            for (int t = 0; t < 9; ++t)
                for (int c = 0; c < 4; ++c) {
                    struct timespec t0, t1;
                    clock_gettime(CLOCK_MONOTONIC, &t0);
                    for (int r = 0; r < reps; ++r) cf[c](&tp, ti, to);
                    clock_gettime(CLOCK_MONOTONIC, &t1);
                    double ns = (double)(t1.tv_sec - t0.tv_sec) * 1e9 +
                                (double)(t1.tv_nsec - t0.tv_nsec);
                    ns /= (double)reps * (double)tb;
                    if (ns < best[c]) best[c] = ns;
                }
            size_t n = strlen(g_desc);
            snprintf(g_desc + n, sizeof g_desc - n,
                     "; ab[B%d]=y2%.0f,zs%.0f,xt%.0f,xl%.0f ns/vol",
                     tb, best[0], best[1], best[2], best[3]);
        }
        free(ti);
        free(to);
    }
#endif
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

#endif /* L13_TEMPLATE_PASS */
