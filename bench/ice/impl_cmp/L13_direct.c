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
 * ICE_R2 (Ice Lake panel, bare-metal Gold 6326, graded chain workload)
 *   Default exec is OV: xfzs with the NEXT volume's X pass (the only
 *   L3-cold loads in the chain regime) interleaved 3-4 chunks per plane
 *   into the current volume's L1-hot plane phase, t1 ping-ponged (t1/t1b).
 *   Adopted from L17_rader ice_r1's winning "sp" shape.  The graded L=13
 *   cell is latency/L3-bound (L13_rader ice_r1), so overlap of independent
 *   long-latency work is the lever; port/width/shuffle tricks wash out.
 *
 * ICE_R5: the fused chain is VOLUME-GROUP-MAJOR by default (vm2): pairs of
 *   volumes run ALL m steps while their state+c (~215 KB with t1/t1b) stay
 *   L2-resident, with the cross-step ov pipeline intact inside each pair.
 *   Adopted from L17_matrixsimd/L8_radix8/L6_unrolled ice_r4 (volmajor
 *   in-place chains, per the brief's cache-residency directive), extended
 *   with the pair grouping so the overlap survives.  See l13_chain_vm.
 *
 * ICE_R6: the chain map moved from the X-pass loads (lazy) to the Z-pass
 *   TILE STORES, register-level (chunk13pz/map2st; adopted from
 *   L23_matrixsimd ice_r5 <- L23_rader ice_r4).  The 1.7 KB rotating stage
 *   and its 1092 L1 accesses/volume-step are deleted; the X pass is 100%
 *   plain.  Default chain shape is MZ1 (volume-major SERIAL): with the map
 *   off the X path, the pair overlap became dead weight (m1 < m2 in 3/3
 *   windows).  All map DAGs are PINNED against per-instance -ffp-contract
 *   variance (see the mapld comment) -- that variance broke lazy-vs-mz
 *   bit-identity on the ICX build until pinned, and any future map edit
 *   must re-verify arm bit-identity by cmp on the node.
 *
 * ICE_R7: batch >= 8 chains run in a SoA-8 LAYOUT -- 8 volumes per zmm,
 *   split re/im, in-place per-pencil DFTs, ZERO shuffles in the transform
 *   (adopted from rival 1000f989_score1.00, the fastest gate-passing L=13
 *   on our node; see the s8 section comment for the full argument and for
 *   what was deliberately NOT copied).  Map default flipped v2 -> v0 (the
 *   hw sqrt was the s8 store-path bottleneck: 4.04 -> 3.48 us/xform).
 *   The interleaved pipeline above is untouched and still serves
 *   fft3d_execute, batch < 8, and the sub-8 batch remainder.
 *
 * DETERMINISM (ice_r2 revision)
 *   The default exec is still a pure function of batch/ISA/sysconf-L3
 *   (ws <= L3 X-first+ov, ws > L3 X-first+pf; mixed tail iff AVX-512), but
 *   create() now runs a chain-shaped in-plan RACE (zs/ov/os/pf) that may
 *   adopt a non-default candidate at a 1.5% margin.  Every raced candidate
 *   is BIT-IDENTICAL (cmp/md5-verified on the node), so run outputs stay
 *   bit-identical regardless of the pick — only the timing varies.  This
 *   panel's records (L13_rader/L17_rader ice_r1) show dev windows are
 *   contention-poisoned and the monitor's quiet window, where create()
 *   runs, is the only honest ranking site; the ice_r1 instrument-only
 *   stance was therefore upgraded to adopt-with-hysteresis.
 *   The X-LAST cache-resident tier was deleted in panel_r10 (its r6
 *   justification predated the r8 t1 pad; see the selection comment in
 *   create()).  -DL13_FORCE=n pins one variant; -DL13_PW=0 / -DL13_PFIN=0
 *   kill the write/read halves of the prefetch schedule; -DL13_AB=0
 *   removes the race and the clock-settle spin.
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
        VT v0 = L13_LD1(0);                                                    \
        P0 = v0; P1 = v0; P2 = v0; P3 = v0; P4 = v0; P5 = v0; P6 = v0;         \
    }                                                                          \
    {                                                                          \
        VT a = L13_LD1(1 * rs), b = L13_LD1(12 * rs);               /* j=1 */  \
        VT u = a + b, w = MULI(a - b);                                         \
        P0 += u;                                                               \
        P1 += C0 * u; P2 += C1 * u; P3 += C2 * u;                              \
        P4 += C3 * u; P5 += C4 * u; P6 += C5 * u;                              \
        R1 = S0 * w; R2 = S1 * w; R3 = S2 * w;                                 \
        R4 = S3 * w; R5 = S4 * w; R6 = S5 * w;                                 \
    }                                                                          \
    {                                                                          \
        VT a = L13_LD1(2 * rs), b = L13_LD1(11 * rs);               /* j=2 */  \
        VT u = a + b, w = MULI(a - b);                                         \
        P0 += u;                                                               \
        P1 += C1 * u; P2 += C3 * u; P3 += C5 * u;                              \
        P4 += C4 * u; P5 += C2 * u; P6 += C0 * u;                              \
        R1 += S1 * w; R2 += S3 * w; R3 += S5 * w;                              \
        R4 -= S4 * w; R5 -= S2 * w; R6 -= S0 * w;                              \
    }                                                                          \
    {                                                                          \
        VT a = L13_LD1(3 * rs), b = L13_LD1(10 * rs);               /* j=3 */  \
        VT u = a + b, w = MULI(a - b);                                         \
        P0 += u;                                                               \
        P1 += C2 * u; P2 += C5 * u; P3 += C3 * u;                              \
        P4 += C0 * u; P5 += C1 * u; P6 += C4 * u;                              \
        R1 += S2 * w; R2 += S5 * w; R3 -= S3 * w;                              \
        R4 -= S0 * w; R5 += S1 * w; R6 += S4 * w;                              \
    }                                                                          \
    {                                                                          \
        VT a = L13_LD1(4 * rs), b = L13_LD1(9 * rs);                /* j=4 */  \
        VT u = a + b, w = MULI(a - b);                                         \
        P0 += u;                                                               \
        P1 += C3 * u; P2 += C4 * u; P3 += C0 * u;                              \
        P4 += C2 * u; P5 += C5 * u; P6 += C1 * u;                              \
        R1 += S3 * w; R2 -= S4 * w; R3 -= S0 * w;                              \
        R4 += S2 * w; R5 -= S5 * w; R6 -= S1 * w;                              \
    }                                                                          \
    {                                                                          \
        VT a = L13_LD1(5 * rs), b = L13_LD1(8 * rs);                /* j=5 */  \
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
        VT a = L13_LD1(6 * rs), b = L13_LD1(7 * rs);                /* j=6 */  \
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
#define L13_LD1(o) VLD(src + (o))
    L13_ACC15();
    L13_ACC6R();
#undef L13_LD1
    L13_STORE_BODY();
}

/* -----------------------------------------------------------------------
 *  ICE_R4: the graded step's pointwise map, fused into loads (the rivals'
 *  "lazy map", corpus §10 §2 / 1760b1bf's pw_core, re-derived at full
 *  double precision).  Given raw z (the previous step's unnormalised FFT
 *  output) and the constant field c at the same offsets, returns the next
 *  state  (z+c) / (1 + |z+c|)  for WC interleaved complex lanes:
 *    u = z+c;  m2 = |u|^2 duplicated per lane (mul + in-lane swap + add);
 *    y = rsqrt14(m2) + TWO Newton steps  (2^-14 -> 5.6e-9 -> 4.7e-17
 *    systematic, ~2-3 ulp with rounding -- vrsqrt14pd's double-precision
 *    seed needs one fewer step than the rivals' float-rcp roundtrip);
 *    a = 1 + m2*y = 1 + |u|;  s = 1/a by ONE hardware divide (L13_MAPV=0,
 *    default: the divider port is otherwise idle in this kernel and vdivpd
 *    is correctly rounded) or by rcp14 + 2 Newtons (L13_MAPV=1: no divider
 *    uop, +5 FMA-port ops);  out = u*s.
 *  Total rel error ~3-4 ulp per application, vs the 1e-13/step chain
 *  budget: PANEL_BRIEF mandates the full-precision tier at L=13 (m=1278);
 *  the rivals' float-seed 2-Newton map (~1e-12/app) is the thing our
 *  chain gate exists to reject.  The vmaxpd clamp keeps m2=0 from turning
 *  rsqrt14's inf into NaN (u=0 stores 0 either way) and keeps denormal
 *  m2 out of the rsqrt: max(m2,1e-300) == m2 exactly for all normal m2.
 * --------------------------------------------------------------------- */
#if defined(__AVX512F__) && defined(__AVX512VL__)
/* map flavors, all inside the 1e-13/step budget with >=100x margin:
 *   0: rsqrt14 + 2 Newton (sqrt on FMA ports) + hardware divide
 *   1: rsqrt14 + 2 Newton + rcp14 + 2 Newton (no divider uop at all)
 *   2: hardware sqrt (divider) + rcp14 + 2 Newton  (the rivals' PW_STYLE 1
 *      at full precision: divider does the sqrt, FMA pipes the reciprocal)
 *   3: hardware sqrt + hardware divide (both on the divider unit; exact)
 * ICE_R7: default flipped v2 -> v0.  In the SoA-8 chain (one ladder per
 * output SLOT, 13 per pencil on the y-pass store path) the hw vsqrtpd is
 * the bottleneck: node A/B at the graded cell 3.476 (v0) / 3.595 (v1) /
 * 4.043 (v2) us/xform.  v0 also wins B=1 through the classic mz path
 * (5.293 vs 6.18 same day; r5 had called the flavors a wash there --
 * that verdict did not survive the r6 map@Z-store shape).  Flavors are
 * compile-time only (never raced; not bit-identical). */
#ifndef L13_MAPV
#  define L13_MAPV 0
#endif
/* ICE_R6: the map DAG is PINNED against -ffp-contract=fast variance.  gcc's
 * convert_mult_to_fma makes per-INSTANCE choices (it may duplicate a
 * two-use multiply into an fma at one inline site and not another), and on
 * this file's ICX build that made the mapld instance inside l13_map_pass
 * round m2 differently from the pair maps -- lazy and mz chain arms
 * diverged a few ulp at ~10% of points (contract=off build: bit-identical;
 * bisected on the node this round).  Two pins make every map instance the
 * same instruction DAG by construction: (1) p2 is asm-opaque, so m2 is
 * always round(re^2) + round(im^2) -- never fma(u,u,SWAPRI(p2)); (2) every
 * Newton line is written with explicit FMA builtins, so contraction shape
 * is not gcc's choice.  Values match the E+O order of the pair maps. */
#if WC == 4
#  define MFMA_(a, b, c)  (VT)_mm512_fmadd_pd((__m512d)(a), (__m512d)(b), (__m512d)(c))
#  define MFNMA_(a, b, c) (VT)_mm512_fnmadd_pd((__m512d)(a), (__m512d)(b), (__m512d)(c))
#elif WC == 2
#  define MFMA_(a, b, c)  (VT)_mm256_fmadd_pd((__m256d)(a), (__m256d)(b), (__m256d)(c))
#  define MFNMA_(a, b, c) (VT)_mm256_fnmadd_pd((__m256d)(a), (__m256d)(b), (__m256d)(c))
#else
#  define MFMA_(a, b, c)  (VT)_mm_fmadd_pd((__m128d)(a), (__m128d)(b), (__m128d)(c))
#  define MFNMA_(a, b, c) (VT)_mm_fnmadd_pd((__m128d)(a), (__m128d)(b), (__m128d)(c))
#endif
/* rcp14 seed + 2 pinned Newtons; expects VT one in scope */
#define L13_RCPLADDER_(s, a)                                                   \
    do {                                                                       \
        (s) = MFMA_((s), MFNMA_((a), (s), one), (s));                          \
        (s) = MFMA_((s), MFNMA_((a), (s), one), (s));                          \
    } while (0)

static inline __attribute__((always_inline)) VT
SUF(mapld)(VT z, VT cv)
{
    const VT one = 1.0 + (VT){0};
    VT u = z + cv;
    VT p2 = u * u;
    __asm__("" : "+v"(p2));             /* pin: no fma into the m2 add */
    VT m2 = p2 + SWAPRI(p2);
#if L13_MAPV >= 2
#if WC == 4
    VT mm = (VT)_mm512_sqrt_pd((__m512d)m2);
#elif WC == 2
    VT mm = (VT)_mm256_sqrt_pd((__m256d)m2);
#else
    VT mm = (VT)_mm_sqrt_pd((__m128d)m2);
#endif
    VT a = mm + 1.0;                    /* 1 + |u|, sqrt exact */
#else
#if WC == 4
    m2 = (VT)_mm512_max_pd((__m512d)m2, _mm512_set1_pd(1e-300));
    VT y = (VT)_mm512_rsqrt14_pd((__m512d)m2);
#elif WC == 2
    m2 = (VT)_mm256_max_pd((__m256d)m2, _mm256_set1_pd(1e-300));
    VT y = (VT)_mm256_rsqrt14_pd((__m256d)m2);
#else
    m2 = (VT)_mm_max_pd((__m128d)m2, _mm_set1_pd(1e-300));
    VT y = (VT)_mm_rsqrt14_pd((__m128d)m2);
#endif
    {                                   /* Newton 1: e -> 1.5 e^2 */
        VT q = m2 * y;
        VT r = MFNMA_(q, y, one);
        y = MFMA_(0.5 * y, r, y);
    }
    {                                   /* Newton 2 */
        VT q = m2 * y;
        VT r = MFNMA_(q, y, one);
        y = MFMA_(0.5 * y, r, y);
    }
    VT a = MFMA_(m2, y, one);           /* 1 + |u| */
#endif
#if L13_MAPV == 1 || L13_MAPV == 2
    VT s;
#if WC == 4
    s = (VT)_mm512_rcp14_pd((__m512d)a);
#elif WC == 2
    s = (VT)_mm256_rcp14_pd((__m256d)a);
#else
    s = (VT)_mm_rcp14_pd((__m128d)a);
#endif
    L13_RCPLADDER_(s, a);
#else
    VT s = 1.0 / a;
#endif
    /* pin the returned product: in chunk13pm the result feeds the fold's
     * adds, and gcc contracted (u*s) + b into an fma at that instance only
     * -- an UNROUNDED map output entering the FFT, the last lazy-vs-mz
     * divergence site (ice_r6 node bisect). */
    VT o = u * s;
    __asm__("" : "+v"(o));
    return o;
}

/* chunk13p with every line load mapped: reads the RAW previous-step z at
 * src and the c field at the same offsets from cs, transforms the mapped
 * state.  Same accumulation DAG as chunk13p (shared ACC macros). */
static inline __attribute__((always_inline)) void
SUF(chunk13pm)(const double *restrict src, const double *restrict cs, long rs,
               double *restrict dst, long da, long db,
               VT C0, VT C1, VT C2, VT C3, VT C4, VT C5,
               VT S0, VT S1, VT S2, VT S3, VT S4, VT S5, int tr)
{
    VT P0, P1, P2, P3, P4, P5, P6, R1, R2, R3, R4, R5, R6;
#define L13_LD1(o) SUF(mapld)(VLD(src + (o)), VLD(cs + (o)))
    L13_ACC15();
    L13_ACC6R();
#undef L13_LD1
    L13_STORE_BODY();
}

/* -----------------------------------------------------------------------
 *  ICE_R6: the map fused into the Z pass's TILE STORES, register-level
 *  (adopted from L23_matrixsimd ice_r5 <- L23_rader ice_r4, executing my
 *  own r5 "next round" #1).  map2st maps TWO transposed output rows (8
 *  points) in registers against the c field at the DESTINATION offsets --
 *  the Z pass stores in driver layout, so c streams at the same addresses
 *  and no transposed c copy is needed (unlike L17_matrixsimd's failed r5
 *  register fusion, whose chunk also overflowed the ROB at ~400 uops;
 *  chunk13pz is ~230 uops and fits).  The |u|^2 merge/expand uses
 *  unpcklpd/unpckhpd pairs (L23_rader's compression): NO index-vector
 *  constants, zero register cost beyond the temporaries.  Per point the
 *  DAG is identical to mapld/l13_mappair (re^2+im^2 in that order, same
 *  MAPV ladder), so mz and lazy chain shapes stay BIT-IDENTICAL and may
 *  legally race.
 * --------------------------------------------------------------------- */
#if WC == 4
static inline __attribute__((always_inline)) void
SUF(map2st)(VT y0, VT y1, const double *c0, const double *c1,
            double *d0, double *d1)
{
    const VT one = 1.0 + (VT){0};
    VT u0 = y0 + VLD(c0), u1 = y1 + VLD(c1);
    VT p0 = u0 * u0, p1 = u1 * u1;
    VT E = (VT)_mm512_unpacklo_pd((__m512d)p0, (__m512d)p1);
    VT O = (VT)_mm512_unpackhi_pd((__m512d)p0, (__m512d)p1);
    VT m2 = E + O;
#if L13_MAPV >= 2
    VT a = (VT)_mm512_sqrt_pd((__m512d)m2) + 1.0;
#else
    m2 = (VT)_mm512_max_pd((__m512d)m2, _mm512_set1_pd(1e-300));
    VT y = (VT)_mm512_rsqrt14_pd((__m512d)m2);
    { VT q = m2 * y; VT r = MFNMA_(q, y, one); y = MFMA_(0.5 * y, r, y); }
    { VT q = m2 * y; VT r = MFNMA_(q, y, one); y = MFMA_(0.5 * y, r, y); }
    VT a = MFMA_(m2, y, one);
#endif
#if L13_MAPV == 1 || L13_MAPV == 2
    VT s = (VT)_mm512_rcp14_pd((__m512d)a);
    L13_RCPLADDER_(s, a);
#else
    VT s = 1.0 / a;
#endif
    VST(d0, u0 * (VT)_mm512_unpacklo_pd((__m512d)s, (__m512d)s));
    VST(d1, u1 * (VT)_mm512_unpackhi_pd((__m512d)s, (__m512d)s));
}

/* chunk13p, tr=1 shape, with every tile store MAPPED (the "mz" Z chunk).
 * cq = c field at the same (da, m*2) offsets as dst.  Column m=12 comes
 * from one register (LASTCOL_ shape): gather its 4 c pairs, map with the
 * duplicated-lane mapld (39/volume -- pairing across chunks not worth the
 * plumbing), extract-store. */
static inline __attribute__((always_inline)) void
SUF(chunk13pz)(const double *restrict src, long rs, double *restrict dst,
               const double *restrict cq, long da,
               VT C0, VT C1, VT C2, VT C3, VT C4, VT C5,
               VT S0, VT S1, VT S2, VT S3, VT S4, VT S5)
{
    VT P0, P1, P2, P3, P4, P5, P6, R1, R2, R3, R4, R5, R6;
#define L13_LD1(o) VLD(src + (o))
    L13_ACC15();
    L13_ACC6R();
#undef L13_LD1
#define TILE4M_(m0, e0, e1, e2, e3)                                            \
    do {                                                                       \
        VT xx[4], yy[4];                                                       \
        xx[0] = (e0); xx[1] = (e1); xx[2] = (e2); xx[3] = (e3);                \
        TTILE(xx, yy);                                                         \
        SUF(map2st)(yy[0], yy[1], cq + 0 * da + (m0) * 2,                      \
                    cq + 1 * da + (m0) * 2, dst + 0 * da + (m0) * 2,           \
                    dst + 1 * da + (m0) * 2);                                  \
        SUF(map2st)(yy[2], yy[3], cq + 2 * da + (m0) * 2,                      \
                    cq + 3 * da + (m0) * 2, dst + 2 * da + (m0) * 2,           \
                    dst + 3 * da + (m0) * 2);                                  \
    } while (0)
    TILE4M_(0, P0, XP_(1), XP_(2), XP_(3));
    TILE4M_(4, XP_(4), XP_(5), XP_(6), XM_(6));
    TILE4M_(8, XM_(5), XM_(4), XM_(3), XM_(2));
#undef TILE4M_
    {
        VT c_ = XM_(1);
        VT cc = {cq[0 * da + 24], cq[0 * da + 25], cq[1 * da + 24],
                 cq[1 * da + 25], cq[2 * da + 24], cq[2 * da + 25],
                 cq[3 * da + 24], cq[3 * da + 25]};
        VT o_ = SUF(mapld)(c_, cc);
        *(SUF(v2) *)(dst + 0 * da + 24) = (SUF(v2)){o_[0], o_[1]};
        *(SUF(v2) *)(dst + 1 * da + 24) = (SUF(v2)){o_[2], o_[3]};
        *(SUF(v2) *)(dst + 2 * da + 24) = (SUF(v2)){o_[4], o_[5]};
        *(SUF(v2) *)(dst + 3 * da + 24) = (SUF(v2)){o_[6], o_[7]};
    }
}
#endif /* WC == 4 */
#undef MFMA_
#undef MFNMA_
#undef L13_RCPLADDER_
#endif /* AVX512F && AVX512VL (mapped kernels) */

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
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

/* long-double trig so the splatted tables are good to ~1e-19 */
#include <math.h>

/* ice_r4: vrsqrt14pd/vrcp14pd/vmaxpd seeds for the fused chain map.  Only
 * the AVX512-guarded mapped kernels use intrinsics; everything else stays
 * plain vector extensions and the file still builds without AVX-512 (the
 * chain then routes through the generic scalar-map fallback below). */
#if defined(__AVX512F__) && defined(__AVX512VL__)
#  include <immintrin.h>
#endif

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

/* ICE_R7: SoA-8 group-buffer geometry (see the s8 section below).  One SLOT
 * = one lattice site of 8 volumes, split re/im: 8 re doubles + 8 im doubles
 * = 128 B.  slot(x,y,z) = x*174 + y*13 + z (the 1000f989 padded strides:
 * 169 -> 174 keeps the x stride 22272 B, whose mod-4096 residue 1792 spreads
 * the 13 x-pencil elements across page-offset space instead of stacking
 * them).  Strides below are in DOUBLES. */
enum { L13_S8Z = 16, L13_S8PY = 13 * 16, L13_S8PX = 174 * 16 };
enum { L13_S8_GS = 13 * 174 * 16 };   /* doubles per SoA group buffer */

struct fft3d_plan {
    int L, batch;
    void (*exec)(const struct fft3d_plan *, const double _Complex *, double _Complex *);
    /* ice_r4: fused map-chain body (steps 2..m, in place on the state
     * buffer with the map applied on the X pass's loads); null when the
     * build has no AVX-512 and fft3d_chain uses the generic fallback. */
    void (*cexec)(const struct fft3d_plan *, double *, const double *, int);
    int cmz;       /* ice_r6: cexec is an mz body (map at the Z store; the
                    * state between steps is rows 0..11 mapped / row 12 raw
                    * and fft3d_chain does the entry/exit format passes) */
    double _Complex *ztmp; /* generic-fallback z buffer (non-AVX512 builds) */
    double *ctab8; /* 6 rows x 7 cosines, j-major, each splatted 8x (zmm) */
    double *stab8; /* s_1..s_6, each splatted 8x (zmm)                    */
    double *ctd8;  /* the 6 DISTINCT cosines c_1..c_6, splatted 8x (zmm)  */
    double *ssd8;  /* ice_r7: s_1..s_6 PLAIN splats (no lane-alternating
                    * sign -- the SoA-8 kernel has no interleaved lanes)   */
    double *srk8;  /* ice_r8: Rader-split pencil constants, splatted 8x:
                    * [0..2] CP_t, [3..5] CM_t (halved cyclic/negacyclic
                    * cosines), [6..11] sin(2pi g^t/13), g^t = 1,2,4,8,3,6
                    * (adopted from L13_rader ice_r7's sork)               */
    double *s8;    /* ice_r7: SoA-8 state group buffer (L13_S8_GS doubles) */
    double *s8c;   /* ice_r7: SoA-8 c-field group buffer                   */
    void *s8blk;   /* ice_r8: 2MB-aligned mmap backing s8/s8c (may be null:
                    * fall back to in-block placement).  One THP holds both
                    * buffers, so the 22272 B x-pencil strides stop walking
                    * a new 4K dTLB entry per load (adopted from the rival
                    * engines' alloc_huge; my r7 port skipped it). */
    size_t s8len;
    double *ctab4; /* the same three, splatted 4x (ymm)                   */
    double *stab4;
    double *ctd4;
    double *pb;    /* 13 x PBROW plane buffer (pass Y -> pass Z)          */
    double *t1;    /* one 13^3 complex volume of scratch, x-planes at T1P */
    double *t1b;   /* second t1 (ice_r2): ping-pong for the ov overlap    */
    double *sb;    /* 13x13 hot staging plane (pass Z -> burst memcpy)    */
    double *sg;    /* ice_r4: 2 rotating 13x16 map stages (chain X pairs) */
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

/* Shared bodies so the pre-RA-scheduled twins below (ice_r2) are the same
 * code compiled under different pass options, not a second copy to maintain. */
#define L13_XFZS_BODY()                                                        \
    for (int b = 0; b < nb; ++b) {                                             \
        const double *vin = (const double *)(in + (size_t)b * 2197);           \
        double *vout = (double *)(out + (size_t)b * 2197);                     \
        L13_MIXP_XPASS(vin, 338, t1, L13_T1P, (double *)0);                    \
        for (int x = 0; x < 13; ++x) {                                         \
            const double *pin = t1 + (long)x * L13_T1P;                        \
            double *pt = vout + (long)x * 338;                                 \
            L13_MIXP_GROUP_ZS(pin, 26, pb, L13_PBROW);                         \
            L13_MIXP_GROUP(pb, L13_PBROW, pt, 26);                             \
        }                                                                      \
    }

/* ov (ice_r2): xfzs with the NEXT volume's X pass interleaved into this
 * volume's plane phase — adopted from L17_rader ice_r1, whose graded-cell
 * winner was "sp" (cross-volume x-block overlap): on this latency-bound,
 * L3-resident chain the only cold loads are the X pass's reads of `in`, and
 * padding the plane phase's store/junction stalls with that independent
 * long-latency work is what won there, where every port/width trick washed
 * out.  t1 ping-pongs (t1/t1b) so volume b+1's X stores never collide with
 * volume b's plane reads; the X chunks are spread 3–4 per plane
 * (floor((x+1)*43/13) schedule = 43 chunks over 13 planes).  Volume 0's X
 * pass runs un-overlapped up front; at nb=1 this is bit- and
 * schedule-identical to xfzs.
 *   PW: 1 adds prefetchw of the out plane the NEXT plane iteration will
 * store to (plane 0's at volume start) — under ov the Z stores' RFOs into
 * `out` (last touched a whole chain step ago, so L3) are the remaining
 * exposed latency.  L13_rader ice_r1 priced pw at this cell as a ±2%
 * page-coloring coin flip, so it is a raced slot ("ow"), never a blind
 * default. */
#define L13_XFZS_OV_BODY(PW)                                                   \
    double *restrict t1b = p->t1b;                                             \
    L13_MIXP_XPASS((const double *)in, 338, t1, L13_T1P, (double *)0);         \
    for (int b = 0; b < nb; ++b) {                                             \
        double *vout = (double *)(out + (size_t)b * 2197);                     \
        const double *tc = (b & 1) ? t1b : t1;                                 \
        double *tn = (b & 1) ? t1 : t1b;                                       \
        const double *vnx = (b + 1 < nb)                                       \
            ? (const double *)(in + (size_t)(b + 1) * 2197) : 0;               \
        int xi = 0;                                                            \
        if (PW) l13_pfw43(vout);                                               \
        for (int x = 0; x < 13; ++x) {                                         \
            const double *pin = tc + (long)x * L13_T1P;                        \
            double *pt = vout + (long)x * 338;                                 \
            if (PW && x < 12) l13_pfw43(vout + (long)(x + 1) * 338);           \
            L13_MIXP_GROUP_ZS(pin, 26, pb, L13_PBROW);                         \
            if (vnx) {                                                         \
                int xe = ((x + 1) * 43) / 13;                                  \
                for (; xi < xe; ++xi) {                                        \
                    if (xi < 42) {                                             \
                        long f0_ = 4L * xi;                                    \
                        chunk13p_w4(vnx + 2 * f0_, 338, tn + 2 * f0_, 2,       \
                                    L13_T1P, C0, C1, C2, C3, C4, C5,           \
                                    K0, K1, K2, K3, K4, K5, 0);                \
                    } else {                                                   \
                        chunk13p_w1(vnx + 2 * 168, 338, tn + 2 * 168, 2,       \
                                    L13_T1P, E0, E1, E2, E3, E4, E5,           \
                                    F0, F1, F2, F3, F4, F5, 0);                \
                    }                                                          \
                }                                                              \
            }                                                                  \
            L13_MIXP_GROUP(pb, L13_PBROW, pt, 26);                             \
        }                                                                      \
    }

static __attribute__((unused)) void
l13_exec_xfzs_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                 double _Complex *restrict out)
{
    L13_MIXP_DECLS;
    L13_XFZS_BODY();
}

static __attribute__((unused)) void
l13_exec_xfzs_ov_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                    double _Complex *restrict out)
{
    L13_MIXP_DECLS;
    L13_XFZS_OV_BODY(0);
}

static __attribute__((unused)) void
l13_exec_xfzs_ow_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                    double _Complex *restrict out)
{
    L13_MIXP_DECLS;
    L13_XFZS_OV_BODY(1);
}

/* ---- oz (ice_r3): ov with the overlap X chunks interleaved INTO the Z
 * group at CHUNK granularity.  Why: under ov the ~600 ns/vol residue above
 * the port floor is the Z stores' RFOs of `out` (L3-resident, last touched
 * a full chain step ago).  ov's overlap X chunks all sit BETWEEN the Y and
 * Z groups, but a Z group is 4 chunks x ~120 instructions -- longer than
 * the ROB -- so the misses at the group's tail drain with no independent
 * work in the window.  oz dispatches one overlap X chunk after each zmm Z
 * chunk (and the remainder after the xmm tail), so every Z chunk's RFO
 * misses always have a full independent X chunk (13 L3 loads + 102 FP ops)
 * in flight beside them.  Same chunks, same values, same
 * floor((x+1)*43/13) budget per plane => output bit-identical to ov/zs.
 * At nb=1 (vnx==0) it degenerates to the zs schedule exactly like ov.
 *   PWL=1 ("op") additionally paces prefetchw of the NEXT plane's out
 * lines, 11 per interleave slot, one plane ahead -- ow's -13% failure
 * issued all 43 in one burst at plane start (fill-buffer saturation on
 * top of the Y group's loads); this is the paced form the L36 winners'
 * chain tuners kept picking (pw=4), priced here as its own race arm. */
#define L13_OVX_STEP(VNX, TN)                                                  \
    do {                                                                       \
        if (xi < 42) {                                                         \
            long fo_ = 4L * xi;                                                \
            chunk13p_w4((VNX) + 2 * fo_, 338, (TN) + 2 * fo_, 2, L13_T1P,      \
                        C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5, 0);    \
        } else {                                                               \
            chunk13p_w1((VNX) + 2 * 168, 338, (TN) + 2 * 168, 2, L13_T1P,      \
                        E0, E1, E2, E3, E4, E5, F0, F1, F2, F3, F4, F5, 0);    \
        }                                                                      \
        ++xi;                                                                  \
    } while (0)

#define L13_XFZS_OZ_BODY(PWL)                                                  \
    double *restrict t1b = p->t1b;                                             \
    L13_MIXP_XPASS((const double *)in, 338, t1, L13_T1P, (double *)0);         \
    for (int b = 0; b < nb; ++b) {                                             \
        double *vout = (double *)(out + (size_t)b * 2197);                     \
        const double *tc = (b & 1) ? t1b : t1;                                 \
        double *tn = (b & 1) ? t1 : t1b;                                       \
        const double *vnx = (b + 1 < nb)                                       \
            ? (const double *)(in + (size_t)(b + 1) * 2197) : 0;               \
        int xi = 0;                                                            \
        for (int x = 0; x < 13; ++x) {                                         \
            const double *pin = tc + (long)x * L13_T1P;                        \
            double *pt = vout + (long)x * 338;                                 \
            const char *pw_ = (PWL && x < 12)                                  \
                ? (const char *)(vout + (long)(x + 1) * 338) : 0;              \
            L13_MIXP_GROUP_ZS(pin, 26, pb, L13_PBROW);                         \
            int xe = ((x + 1) * 43) / 13;                                      \
            {                                                                  \
                int nt_ = 3;                                                   \
                __asm__("" : "+r"(nt_));                                       \
                for (int t_ = 0; t_ < nt_; ++t_) {                             \
                    long f0_ = 4L * t_;                                        \
                    chunk13p_w4(pb + 2 * f0_, L13_PBROW, pt + f0_ * 26, 26, 2, \
                        C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5, 1);    \
                    if (pw_)                                                   \
                        for (int q_ = 11 * t_; q_ < 11 * t_ + 11; ++q_)        \
                            __builtin_prefetch(pw_ + 64 * q_, 1, 3);           \
                    if (vnx && xi < xe) L13_OVX_STEP(vnx, tn);                 \
                }                                                              \
                chunk13p_w1(pb + 2 * 12, L13_PBROW, pt + 12 * 26, 26, 2,       \
                    E0, E1, E2, E3, E4, E5, F0, F1, F2, F3, F4, F5, 1);        \
                if (pw_)                                                       \
                    for (int q_ = 33; q_ < 43; ++q_)                           \
                        __builtin_prefetch(pw_ + 64 * q_, 1, 3);               \
                while (vnx && xi < xe) L13_OVX_STEP(vnx, tn);                  \
            }                                                                  \
        }                                                                      \
    }

static __attribute__((unused)) void
l13_exec_xfzs_oz_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                    double _Complex *restrict out)
{
    L13_MIXP_DECLS;
    L13_XFZS_OZ_BODY(0);
}

static __attribute__((unused)) void
l13_exec_xfzs_op_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                    double _Complex *restrict out)
{
    L13_MIXP_DECLS;
    L13_XFZS_OZ_BODY(1);
}

/* ---- pre-RA-scheduled twins (ice_r2, adopted from L17_matrixsimd ice_r1:
 * gcc does no pre-RA scheduling on x86, so each chunk's phase-serial source
 * reaches the issue queue unmixed; their matched node A/B on the graded
 * chain was −7.7% with exactly this pragma pair, outputs unchanged since
 * contraction is decided before sched1).  Same body macros, so any
 * difference is scheduling only; bit-identity is cmp-verified on the node.
 * Raced in-plan, never defaulted blind (my own r9 note: the optimize()
 * pragma rebuilds the whole option set and cost L17_rader ~2% in the
 * unroll-loops form — the race prices the net effect per process). */
#pragma GCC push_options
#pragma GCC optimize("schedule-insns", "sched-pressure")
static __attribute__((unused)) void
l13_exec_xfzs_s_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                   double _Complex *restrict out)
{
    L13_MIXP_DECLS;
    L13_XFZS_BODY();
}

static __attribute__((unused)) void
l13_exec_xfzs_ov_s_mx(const fft3d_plan *restrict p, const double _Complex *restrict in,
                      double _Complex *restrict out)
{
    L13_MIXP_DECLS;
    L13_XFZS_OV_BODY(0);
}
#pragma GCC pop_options

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

/* =============================================================================
 * ICE_R4: the fused map chain --  state <- (FFT(state)+c)/(1+|FFT(state)+c|).
 *
 * LAZY MAP (adopted from the rival pipelines' winning codes, corpus §10 §2,
 * source 1760b1bf's generator; concept confirmed by the brief): the state
 * buffer holds the RAW unnormalised FFT output between steps, and the map is
 * applied inside the NEXT step's X pass, fused into each line load
 * (chunk13pm).  That deletes the separate map pass entirely -- one full
 * read+write of the 1.07 MB state per step plus a pass with nothing to hide
 * its latency under -- which is the exact unfused configuration the brief
 * prices at ~57% of total time.  The final step's raw output gets ONE
 * standalone map pass (amortised 1/m ~ 1/1278: nothing).
 *
 * IN PLACE, ONE BUFFER: each step reads volume b fully (X pass -> t1) before
 * its plane phase overwrites volume b, so the chain ping-pong collapses to a
 * single state buffer -- the driver's final_out.  Working set drops to
 * state (1.07 MB) + c (1.07 MB), vs 3x 1.07 MB through the fallback.
 *
 * CROSS-STEP OV (own extension; unblocked by owning the chain): ice_r2's ov
 * overlap ran the NEXT volume's X pass inside the current plane phase but
 * had to drain at every fft3d_execute boundary -- the driver's scale pass
 * made precomputed spectra stale (ice_r2 record, "for next round" #1).
 * fft3d_chain owns all m steps, so at the last volume of step s the overlap
 * X chunks simply come from step s+1's volume 0 (its z was finished by this
 * step's volume-0 plane phase long ago).  One X-pass prologue for the WHOLE
 * chain instead of one per step, and no pipeline drain at step boundaries.
 * Hazard check: X of (s,b+1) reads z_{s-1}[b+1], overwritten only by plane
 * phase (s,b+1) later; X of (s+1,0) reads z_s[0], written by plane phase
 * (s,0) earlier in step s.  t1/t1b ping-pong by a running per-volume flip.
 *
 * nb==1 is inherently serial (the only next volume IS the one being
 * written), so it runs the per-step zs-shaped chain body -- still fused-map,
 * no overlap.  Both bodies share every chunk DAG with the default execs;
 * the map is bit-identical across shapes.
 * =============================================================================
 */
#if defined(__AVX512F__) && defined(__AVX512VL__)
#define L13_CHAIN_FAST 1

/* PAIRED lazy map (1760b1bf's pw_pair_gen shape, full-precision seeds): two
 * ADJACENT zmm of interleaved complex (8 points), ONE rsqrt/Newton/divide
 * chain on 8 DISTINCT magnitudes -- half the scalar map work of the
 * duplicated-lane form, and half its divider pressure.  Elementwise the
 * per-point DAG is identical to mapld_w4 (the merge/expand permutes only
 * reroute lanes; IEEE a+b == b+a makes the odd-lane |u|^2 order immaterial),
 * so paired and unpaired points can mix bit-identically. */
static inline __attribute__((always_inline)) void
l13_mappair(vd_w4 za, vd_w4 zb, vd_w4 ca, vd_w4 cb, vd_w4 *va, vd_w4 *vb)
{
    static const long long IE_[8] = {0, 2, 4, 6, 8, 10, 12, 14};
    static const long long IO_[8] = {1, 3, 5, 7, 9, 11, 13, 15};
    static const long long IA_[8] = {0, 0, 1, 1, 2, 2, 3, 3};
    static const long long IB_[8] = {4, 4, 5, 5, 6, 6, 7, 7};
    /* ice_r6: Newton lines via explicit FMA builtins -- the pinned map DAG
     * (see the mapld comment; per-instance contraction variance is what
     * broke lazy-vs-mz bit-identity on the ICX build). */
#define MP_FMA_(a, b, c)  (vd_w4)_mm512_fmadd_pd((__m512d)(a), (__m512d)(b), (__m512d)(c))
#define MP_FNMA_(a, b, c) (vd_w4)_mm512_fnmadd_pd((__m512d)(a), (__m512d)(b), (__m512d)(c))
    const vd_w4 one = 1.0 + (vd_w4){0};
    vd_w4 ua = za + ca, ub = zb + cb;
    vd_w4 pa = ua * ua, pb = ub * ub;
    vd_w4 E = (vd_w4)_mm512_permutex2var_pd((__m512d)pa,
                  _mm512_loadu_si512((const void *)IE_), (__m512d)pb);
    vd_w4 O = (vd_w4)_mm512_permutex2var_pd((__m512d)pa,
                  _mm512_loadu_si512((const void *)IO_), (__m512d)pb);
    vd_w4 m2 = E + O;
#if L13_MAPV >= 2
    vd_w4 a = (vd_w4)_mm512_sqrt_pd((__m512d)m2) + 1.0;
#else
    m2 = (vd_w4)_mm512_max_pd((__m512d)m2, _mm512_set1_pd(1e-300));
    vd_w4 y = (vd_w4)_mm512_rsqrt14_pd((__m512d)m2);
    { vd_w4 q = m2 * y; vd_w4 r = MP_FNMA_(q, y, one); y = MP_FMA_(0.5 * y, r, y); }
    { vd_w4 q = m2 * y; vd_w4 r = MP_FNMA_(q, y, one); y = MP_FMA_(0.5 * y, r, y); }
    vd_w4 a = MP_FMA_(m2, y, one);
#endif
#if L13_MAPV == 1 || L13_MAPV == 2
    vd_w4 s = (vd_w4)_mm512_rcp14_pd((__m512d)a);
    s = MP_FMA_(s, MP_FNMA_(a, s, one), s);
    s = MP_FMA_(s, MP_FNMA_(a, s, one), s);
#else
    vd_w4 s = 1.0 / a;
#endif
#undef MP_FMA_
#undef MP_FNMA_
    *va = ua * (vd_w4)_mm512_permutexvar_pd(
                  _mm512_loadu_si512((const void *)IA_), (__m512d)s);
    *vb = ub * (vd_w4)_mm512_permutexvar_pd(
                  _mm512_loadu_si512((const void *)IB_), (__m512d)s);
}

/* One X-pass PAIR unit: columns [8u, 8u+8) of one volume.  Phase 1 maps the
 * 13 row-pairs (adjacent zmm) into a 1.7 KB L1 stage; phase 2 runs the
 * proven spill-free chunk13p on the staged state (64 B stores -> exact
 * 64 B loads: clean store-forward).  The inline-map kernel (chunk13pm_w4)
 * measured 7.06 us/xform graded: 12 pinned + 13 accumulators + map temps =
 * ~1010 stack refs of spill, and every divide sat on the accumulate DAG.
 * This split deletes both: map registers and FFT registers never coexist,
 * and the divide's consumer is a store, not an FMA chain. */
#define L13_CHX_MAP(SRCB, CB, U, SG)                                           \
    do {                                                                       \
        long fq_ = 8L * (U);                                                   \
        const double *sp_ = (SRCB) + 2 * fq_;                                  \
        const double *cp_ = (CB) + 2 * fq_;                                    \
        double *sg_ = (SG);                                                    \
        for (int x_ = 0; x_ < 13; ++x_) {                                      \
            vd_w4 va_, vb_;                                                    \
            l13_mappair(*(const vd_w4 *)(sp_ + 338L * x_),                     \
                        *(const vd_w4 *)(sp_ + 338L * x_ + 8),                 \
                        *(const vd_w4 *)(cp_ + 338L * x_),                     \
                        *(const vd_w4 *)(cp_ + 338L * x_ + 8), &va_, &vb_);    \
            *(vd_w4 *)(sg_ + 16 * x_) = va_;                                   \
            *(vd_w4 *)(sg_ + 16 * x_ + 8) = vb_;                               \
        }                                                                      \
    } while (0)

#define L13_CHX_FFT(DSTB, U, SG)                                               \
    do {                                                                       \
        long fq_ = 8L * (U);                                                   \
        double *sg_ = (SG);                                                    \
        chunk13p_w4(sg_, 16, (DSTB) + 2 * fq_, 2, L13_T1P,                     \
                    C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5, 0);        \
        chunk13p_w4(sg_ + 8, 16, (DSTB) + 2 * (fq_ + 4), 2, L13_T1P,           \
                    C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5, 0);        \
    } while (0)

/* mapped X pass of one volume: raw z at SRCB, c at CB, X-transformed mapped
 * state into DSTB (t1 layout).  169 = 21*8 + 1: 21 pair units + the w1 tail
 * (inline-mapped: xmm pressure is trivial), every point mapped once.
 * DEPTH-1 SOFTWARE PIPELINE across the rotating stages: unit u+1's whole
 * map phase (~270 independent uops incl. the sqrts) runs between unit u's
 * stage stores and its FFT chunks -- a pair unit is bigger than the ROB, so
 * without this the chunk loads sit on raw store-forward latency and the
 * divider bursts have nothing to hide under (measured +0.15 us/xform). */
#define L13_MIXP_XPASS_MAP(SRCB, CB, DSTB, SG0)                                \
    do {                                                                       \
        L13_CHX_MAP(SRCB, CB, 0, (SG0));                                       \
        int nu_ = 20;                                                          \
        __asm__("" : "+r"(nu_));                                               \
        for (int u_ = 0; u_ < nu_; ++u_) {                                     \
            L13_CHX_MAP(SRCB, CB, u_ + 1, (SG0) + ((u_ + 1) & 1) * 208);       \
            L13_CHX_FFT(DSTB, u_, (SG0) + (u_ & 1) * 208);                     \
        }                                                                      \
        L13_CHX_FFT(DSTB, 20, (SG0));                                          \
        chunk13pm_w1((SRCB) + 2 * 168, (CB) + 2 * 168, 338,                    \
                     (DSTB) + 2 * 168, 2, L13_T1P,                             \
                     E0, E1, E2, E3, E4, E5, F0, F1, F2, F3, F4, F5, 0);       \
    } while (0)

/* one interleaved mapped X ACTION of the next volume (ov placement): the
 * same depth-1 pipeline unrolled into 23 dispatchable actions per volume:
 *   0: map(0);  1..20: map(k)+fft(k-1);  21: fft(20);  22: w1 tail */
#define L13_CHOVX_STEP(VNX, VCX, TN, SG0)                                      \
    do {                                                                       \
        if (xi == 0) {                                                         \
            L13_CHX_MAP(VNX, VCX, 0, (SG0));                                   \
        } else if (xi <= 20) {                                                 \
            L13_CHX_MAP(VNX, VCX, xi, (SG0) + (xi & 1) * 208);                 \
            L13_CHX_FFT(TN, xi - 1, (SG0) + ((xi - 1) & 1) * 208);             \
        } else if (xi == 21) {                                                 \
            L13_CHX_FFT(TN, 20, (SG0));                                        \
        } else {                                                               \
            chunk13pm_w1((VNX) + 2 * 168, (VCX) + 2 * 168, 338,                \
                         (TN) + 2 * 168, 2, L13_T1P,                           \
                         E0, E1, E2, E3, E4, E5, F0, F1, F2, F3, F4, F5, 0);   \
        }                                                                      \
        ++xi;                                                                  \
    } while (0)

/* steps 2..m, cross-step ov pipeline.  st holds z_1 on entry, z_m on exit. */
static void
l13_chain_ov_mx(const fft3d_plan *restrict p, double *st, const double *cf,
                int m)
{
    L13_MIXP_DECLS;
    double *restrict t1b = p->t1b;
    double *restrict sg = p->sg;
    if (m < 2) return;
    L13_MIXP_XPASS_MAP(st, cf, t1, sg);                 /* X of (step 2, vol 0) */
    int flip = 0;
    for (int s = 2; s <= m; ++s) {
        for (int b = 0; b < nb; ++b) {
            double *vout = st + (size_t)b * 4394;
            const double *tc = flip ? t1b : t1;
            double *tn = flip ? t1 : t1b;
            flip ^= 1;
            const double *vnx, *vcx;
            if (b + 1 < nb) { vnx = st + (size_t)(b + 1) * 4394;
                              vcx = cf + (size_t)(b + 1) * 4394; }
            else if (s < m) { vnx = st; vcx = cf; }     /* next STEP's vol 0 */
            else            { vnx = 0;  vcx = 0; }
#if defined(L13_CPF) && L13_CPF
            /* paced L2 read-prefetch of the c field ONE VOLUME BEYOND the
             * overlap (b+2), one 43-line slice per plane: the c stream is
             * the X window's new second L3 stream and its schedule is known
             * exactly.  Knob only -- the panel has priced input prefetch
             * negative three times when it raced ov's real work. */
            const double *vc2 = (b + 2 < nb) ? cf + (size_t)(b + 2) * 4394
                              : (s < m ? cf + (size_t)(b + 2 - nb) * 4394 : 0);
#endif
            int xi = 0;
            for (int x = 0; x < 13; ++x) {
                const double *pin = tc + (long)x * L13_T1P;
                double *pt = vout + (long)x * 338;
                L13_MIXP_GROUP_ZS(pin, 26, pb, L13_PBROW);
#if defined(L13_CPF) && L13_CPF
                if (vc2)
                    for (int q_ = 0; q_ < 42; ++q_)
                        __builtin_prefetch((const char *)(vc2 + x * 338) +
                                           64 * q_, 0, 2);
#endif
                if (vnx) {
                    int xe = ((x + 1) * 23) / 13;
                    while (xi < xe) L13_CHOVX_STEP(vnx, vcx, tn, sg);
                }
                L13_MIXP_GROUP(pb, L13_PBROW, pt, 26);
            }
        }
    }
}

/* steps 2..m, per-step zs shape (no overlap): the nb==1 body (in-place
 * same-volume overlap would be a read-write hazard) and the L13_CF=0 A/B. */
static void
l13_chain_zs_mx(const fft3d_plan *restrict p, double *st, const double *cf,
                int m)
{
    L13_MIXP_DECLS;
    double *restrict sg = p->sg;
    for (int s = 2; s <= m; ++s) {
        for (int b = 0; b < nb; ++b) {
            double *v = st + (size_t)b * 4394;
            const double *cv = cf + (size_t)b * 4394;
            L13_MIXP_XPASS_MAP(v, cv, t1, sg);
            for (int x = 0; x < 13; ++x) {
                const double *pin = t1 + (long)x * L13_T1P;
                double *pt = v + (long)x * 338;
                L13_MIXP_GROUP_ZS(pin, 26, pb, L13_PBROW);
                L13_MIXP_GROUP(pb, L13_PBROW, pt, 26);
            }
        }
    }
}

/* -----------------------------------------------------------------------
 *  ICE_R5: VOLUME-GROUP-MAJOR chain (vm).  The step-major r4 chain walks
 *  all nb volumes each step, so every step re-streams state (1.07 MB) + c
 *  (1.07 MB) through L3 -- the whole cell was diagnosed latency/L3-bound
 *  (L13_rader ice_r1, my ice_r2).  But the m chains are INDEPENDENT per
 *  volume, and the brief's own directive ("iterate a volume through steps
 *  while it is cache-resident", corpus §10 §3) says to exploit that:
 *  L17_matrixsimd ice_r4 executed it literally (volmajor-inplace: L3
 *  traffic for the WHOLE chain = one x0 read, one c read, one writeback)
 *  and so did L8_radix8 / L6_unrolled / the L36 pair.  Adopted here, with
 *  the one thing my lineage adds that theirs lack: groups of G=2 volumes,
 *  so the cross-volume/cross-step ov pipeline still has an independent X
 *  pass to hide plane-phase junction latency under.  Working set per
 *  group at G=2: state 68.6K + c 68.6K + t1/t1b 71.6K + pb/sg ~= 215 KB,
 *  well inside the node's 1.25 MB L2 -- every X-pass load and Z-store RFO
 *  that was an L3 round trip becomes an L2 hit from step 2 on.
 *    Pure per-volume serial (G=1) is priced too: my own r4 B=1 fused
 *  number (6.16 vs B=32's 5.86) says an UN-overlapped L2-resident chain
 *  loses to the overlapped L3-streaming one -- residency alone is not
 *  enough at this size; the pair keeps both.
 *    Bit-identity: grouping only reorders WHOLE volume-steps across
 *  volumes; each volume's DAG (and the zs/ov bodies executing it) is
 *  unchanged, so .chain outputs are bit-identical to r4's for every G --
 *  which is what makes G race-adoptable.  Odd tails fall back per volume
 *  to the zs body.  -DL13_CG=0/1/2/4 pins step-major-ov/vm1/vm2/vm4. */
static void
l13_chain_vm(const fft3d_plan *restrict p, double *st, const double *cf,
             int m, int G)
{
    fft3d_plan tp = *p;
    const int nb = p->batch;
    for (int g = 0; g < nb; g += G) {
        int tb = nb - g;
        if (tb > G) tb = G;
        tp.batch = tb;
        if (tb >= 2)
            l13_chain_ov_mx(&tp, st + (size_t)g * 4394,
                            cf + (size_t)g * 4394, m);
        else
            l13_chain_zs_mx(&tp, st + (size_t)g * 4394,
                            cf + (size_t)g * 4394, m);
    }
}

static void
l13_chain_vm1_mx(const fft3d_plan *restrict p, double *st, const double *cf,
                 int m)
{ l13_chain_vm(p, st, cf, m, 1); }

static void
l13_chain_vm2_mx(const fft3d_plan *restrict p, double *st, const double *cf,
                 int m)
{ l13_chain_vm(p, st, cf, m, 2); }

static void
l13_chain_vm4_mx(const fft3d_plan *restrict p, double *st, const double *cf,
                 int m)
{ l13_chain_vm(p, st, cf, m, 4); }

/* -----------------------------------------------------------------------
 *  ICE_R6: "mz" chain shapes -- the map moves from the X-pass LOADS (lazy,
 *  r4) to the Z-pass TILE STORES (register-level, chunk13pz).  What this
 *  deletes per volume-step vs the lazy pipeline: the whole 1.7 KB rotating
 *  stage (546 stage stores + 546 stage loads of L1 traffic), the pair-unit
 *  bookkeeping, and the map latency in front of the X chunks.  What it
 *  adds: the map ladder on the Z store path (fits the ROB at ~230
 *  uops/chunk -- see chunk13pz) and ~14% more divider ops (the 39 lastcol
 *  mapld ladders).  L23_matrixsimd ice_r5 measured this placement a clear
 *  win on their identical-structure cell; L17_matrixsimd's r5 register
 *  fusion loss is diagnosed in their record as ROB overflow, which does
 *  not apply at 13.
 *
 *  STATE FORMAT between steps: FULLY MAPPED.  Rows ky=0..11 map in the Z
 *  tile stores; the w1 Z tail row (ky=12, 13 complex, contiguous 208 B)
 *  stays raw for ONE PLANE and is strip-mapped (l13_map_span, 3 small
 *  ladders) after the NEXT plane's Y group -- deferring one plane keeps
 *  the strip's 64 B loads out of the store-forward window of the tail's
 *  16 B stores.  The first mz revision instead kept row 12 raw across the
 *  whole step and mapped it with 3 inline-map X chunks (chunk13pm_w4);
 *  that lost 0.7-1.5% to the lazy arms (fch-ab m1 6303 / m2 6321 vs v2
 *  6226) -- 13 serial map ladders on an X chunk's accumulate DAG are the
 *  r4 inline-map failure in miniature, and worse as ov interleave
 *  actions.  The strip form keeps the X pass 100% plain (the proven
 *  42-zmm + w1 shape) and the whole-state map pass moves to CHAIN ENTRY
 *  (state_1 = map(z_1), amortised 1/m; exit needs nothing).
 *  Per-point map DAG identical everywhere => bit-identical to the lazy
 *  arms, so mz shapes are legal fch-ab candidates.
 * --------------------------------------------------------------------- */
/* map a contiguous span of npts complex in place (pre/post format passes) */
static inline void
l13_map_span(double *st, const double *cf, long npts)
{
    long i = 0;
    for (; i + 8 <= npts; i += 8) {
        vd_w4 va, vb;
        l13_mappair(*(const vd_w4 *)(st + 2 * i),
                    *(const vd_w4 *)(st + 2 * i + 8),
                    *(const vd_w4 *)(cf + 2 * i),
                    *(const vd_w4 *)(cf + 2 * i + 8), &va, &vb);
        *(vd_w4 *)(st + 2 * i) = va;
        *(vd_w4 *)(st + 2 * i + 8) = vb;
    }
    for (; i + 4 <= npts; i += 4)
        *(vd_w4 *)(st + 2 * i) = mapld_w4(*(const vd_w4 *)(st + 2 * i),
                                          *(const vd_w4 *)(cf + 2 * i));
    {
        double *sp = st + 2 * i;
        const double *cp = cf + 2 * i;
        for (; i < npts; ++i, sp += 2, cp += 2)
            *(vd_w1 *)sp = mapld_w1(*(const vd_w1 *)sp, *(const vd_w1 *)cp);
    }
}

/* Z group with mapped tile stores; the w1 tail row (ky=12) stays RAW */
#define L13_MIXP_GROUP_MZ(SRCB, RS, DSTB, CQB, DA)                             \
    do {                                                                       \
        int nt_ = 3;                                                           \
        __asm__("" : "+r"(nt_));                                               \
        for (int t_ = 0; t_ < nt_; ++t_) {                                     \
            long f0_ = 4L * t_;                                                \
            chunk13pz_w4((SRCB) + 2 * f0_, (RS), (DSTB) + f0_ * (DA),          \
                         (CQB) + f0_ * (DA), (DA),                             \
                         C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5);      \
        }                                                                      \
        chunk13p_w1((SRCB) + 2 * 12, (RS), (DSTB) + 12 * (DA), (DA), 2,        \
                    E0, E1, E2, E3, E4, E5, F0, F1, F2, F3, F4, F5, 1);        \
    } while (0)

/* steps 2..m, mz, per-step serial (the nb==1 body).  Entry state must be
 * FULLY MAPPED (fft3d_chain runs the whole-state map pass first); exit is
 * fully mapped state_m.  In-place legality unchanged: the X pass fully
 * drains a volume into t1 before its plane phase rewrites it.  pr/pcr
 * carry the previous plane's raw row 12 to its deferred strip map. */
static void
l13_chain_mzs_mx(const fft3d_plan *restrict p, double *st, const double *cf,
                 int m)
{
    L13_MIXP_DECLS;
    for (int s = 2; s <= m; ++s) {
        for (int b = 0; b < nb; ++b) {
            double *v = st + (size_t)b * 4394;
            const double *cv = cf + (size_t)b * 4394;
            L13_MIXP_XPASS(v, 338, t1, L13_T1P, (double *)0);
            double *pr = 0;
            const double *pcr = 0;
            for (int x = 0; x < 13; ++x) {
                const double *pin = t1 + (long)x * L13_T1P;
                double *pt = v + (long)x * 338;
                L13_MIXP_GROUP_ZS(pin, 26, pb, L13_PBROW);
                if (pr) l13_map_span(pr, pcr, 13);
                L13_MIXP_GROUP_MZ(pb, L13_PBROW, pt, cv + (long)x * 338, 26);
                pr = pt + 312;
                pcr = cv + (long)x * 338 + 312;
            }
            l13_map_span(pr, pcr, 13);
        }
    }
}

/* steps 2..m, mz + cross-step ov pipeline (same hazard argument as
 * l13_chain_ov_mx: X of (s,b+1) reads state_{s-1}[b+1], finished by plane
 * phase (s-1,b+1) INCLUDING its final strip; X of (s+1,0) reads state_s[0],
 * finished by plane phase (s,0) earlier in step s.  The overlap X actions
 * are the plain unitary 43-chunk schedule -- nothing mapped on the X side). */
static void
l13_chain_mv_mx(const fft3d_plan *restrict p, double *st, const double *cf,
                int m)
{
    L13_MIXP_DECLS;
    double *restrict t1b = p->t1b;
    if (m < 2) return;
    L13_MIXP_XPASS(st, 338, t1, L13_T1P, (double *)0); /* X of (step 2, vol 0) */
    int flip = 0;
    for (int s = 2; s <= m; ++s) {
        for (int b = 0; b < nb; ++b) {
            double *vout = st + (size_t)b * 4394;
            const double *cv = cf + (size_t)b * 4394;
            const double *tc = flip ? t1b : t1;
            double *tn = flip ? t1 : t1b;
            flip ^= 1;
            const double *vnx;
            if (b + 1 < nb)  vnx = st + (size_t)(b + 1) * 4394;
            else if (s < m)  vnx = st;
            else             vnx = 0;
            int xi = 0;
            double *pr = 0;
            const double *pcr = 0;
            for (int x = 0; x < 13; ++x) {
                const double *pin = tc + (long)x * L13_T1P;
                double *pt = vout + (long)x * 338;
                L13_MIXP_GROUP_ZS(pin, 26, pb, L13_PBROW);
                if (pr) l13_map_span(pr, pcr, 13);
                if (vnx) {
                    int xe = ((x + 1) * 43) / 13;
                    while (xi < xe) L13_OVX_STEP(vnx, tn);
                }
                L13_MIXP_GROUP_MZ(pb, L13_PBROW, pt, cv + (long)x * 338, 26);
                pr = pt + 312;
                pcr = cv + (long)x * 338 + 312;
            }
            l13_map_span(pr, pcr, 13);
        }
    }
}

/* volume-group-major mz (the r5 vm wrapper on the mz bodies) */
static void
l13_chain_mzvm(const fft3d_plan *restrict p, double *st, const double *cf,
               int m, int G)
{
    fft3d_plan tp = *p;
    const int nb = p->batch;
    for (int g = 0; g < nb; g += G) {
        int tb = nb - g;
        if (tb > G) tb = G;
        tp.batch = tb;
        if (tb >= 2)
            l13_chain_mv_mx(&tp, st + (size_t)g * 4394,
                            cf + (size_t)g * 4394, m);
        else
            l13_chain_mzs_mx(&tp, st + (size_t)g * 4394,
                             cf + (size_t)g * 4394, m);
    }
}

static void
l13_chain_mz1_mx(const fft3d_plan *restrict p, double *st, const double *cf,
                 int m)
{ l13_chain_mzvm(p, st, cf, m, 1); }

static void
l13_chain_mz2_mx(const fft3d_plan *restrict p, double *st, const double *cf,
                 int m)
{ l13_chain_mzvm(p, st, cf, m, 2); }

/* the one standalone map pass: after step m the buffer holds raw z_m; map it
 * in place.  Runs once per CHAIN (1/m of a step's work -- noise). */
static void
l13_map_pass(double *st, const double *cf, size_t npts)
{
    size_t i = 0;
    for (; i + 4 <= npts; i += 4) {
        vd_w4 z = *(const vd_w4 *)(st + 2 * i);
        vd_w4 c = *(const vd_w4 *)(cf + 2 * i);
        *(vd_w4 *)(st + 2 * i) = mapld_w4(z, c);
    }
    for (; i < npts; ++i) {
        vd_w1 z = *(const vd_w1 *)(st + 2 * i);
        vd_w1 c = *(const vd_w1 *)(cf + 2 * i);
        *(vd_w1 *)(st + 2 * i) = mapld_w1(z, c);
    }
}
/* =============================================================================
 * ICE_R7: the SoA-8 chain path ("s8") -- 8 VOLUMES PER ZMM, split re/im.
 *
 * Adopted from the rival 1000f989_score1.00 (the 1,045-line v4 1.00-scorer;
 * `ext/reference/fft_v4_solutions/1000f989_score1.00/implementation.c`,
 * GEN_PRIME/GEN_DRIVER_A read before writing a line), which is the fastest
 * GATE-PASSING L=13 on our own node: 0.1587 s (chain drift 1.13e-13 under
 * OUR budget) vs my ice_r6 0.2162 s.  Same idea independently: the v6
 * generator's SoA "Hartley-split" primes (its README calls them "the
 * decisive win over MKL"), though that binary's output is WRONG on this
 * node (chain_rel 1.4, results/rivals_icelake) so only 1000f989's numbers
 * are honest targets.
 *
 * WHY THIS LAYOUT WINS AT THE GRADED CELL (B=32, m=1278)
 *   Lane l of every vector = volume g*8+l at one lattice site, re and im in
 *   SEPARATE vectors.  Consequences, vs my interleaved lanes=lines pipeline:
 *     - ZERO shuffles in the transform: no TTILE tile transposes, no MULI
 *       re/im swap (the -i is just "read the other component's array"), no
 *       map unpck merge/expand.  On ICX both FMA pipes share port 5 with the
 *       shuffle unit, so my ~3.2k shuffle uops/volume-step were stealing FMA
 *       slots; the ice_r2 roofline note priced the port floor at ~9.0k
 *       cycles/volume WITH shuffles vs ~6.6k without.
 *     - IN-PLACE per pencil: a length-13 DFT along any axis reads and writes
 *       the same 13 slots, so pb/t1/t1b and every transposing store DIE.
 *       Two passes of the buffer per step (y-plane visit does z- then
 *       x-pencils L1-hot; x-plane visit does y-pencils + map), all L2-hot.
 *     - The map is naturally "pair-compressed": one slot IS 8 independent
 *       points, one sqrt/rcp ladder per slot, no permutes.
 *   Cost: an entry/exit 8x8 transpose between driver layout and SoA, once
 *   per CHAIN (1/1278 of a step -- noise), and the format only engages at
 *   batch >= 8 (the graded cell is 32; B=1 keeps the ice_r6 mz path).
 *
 * WHAT IS *NOT* ADOPTED FROM 1000f989, AND WHY
 *   - Their per-coefficient EMBEDDED-BROADCAST FMAs (bc(CTj[j][k]) memory
 *     operands): the f40c5e25 forensics (and our own L17 lineage) measured
 *     embedded-broadcast FMA at ~1.3/cyc vs 2/cyc register-register on this
 *     silicon.  My kernel keeps the 12 DISTINCT folded constants (6 cos + 6
 *     sin) PINNED in registers across the whole pass -- the panel's proven
 *     chunk13p structure -- so the FMA stream runs at register speed.  The
 *     13-accumulator liveness that allows this forces a TWO-SWEEP component
 *     split (below), which their j-outer CR/CI+SR/SI form avoids by paying
 *     the broadcast tax and some spills.  Op count is identical (204 vector
 *     FP per pencil); mine trades 6 extra subs + a 384 B L1 spill buffer
 *     for ~72 broadcast loads and the 0.7-per-FMA-cycle tax.
 *   - Their rsqrt-only map seed: my mapst keeps the proven MAPV ladder
 *     (default v2: exact hw sqrt + rcp14 + 2 explicit-FMA Newtons).
 *
 * THE TWO-SWEEP PENCIL (component split; own derivation on their layout)
 *   With a_j = u_j+u_{13-j}, d_j = u_j-u_{13-j} per component:
 *     X_k.re = [x0.re + sum_j c_kj (re_j+re_{13-j})] + sum_j s_kj (im_j-im_{13-j})
 *     X_k.im = [x0.im + sum_j c_kj (im_j+im_{13-j})] - sum_j s_kj (re_j-re_{13-j})
 *   Sweep R produces all 13 real outputs (13 accumulators + 12 pinned
 *   constants = the exact liveness of chunk13p, which fits), and while the
 *   original re values are in hand it spills nbr_j = re_{13-j}-re_j (6
 *   vectors, 384 B, L1-pinned) for sweep I; with w_j = nbr_j the imag
 *   combine becomes IDENTICAL to the real one (X_k.im = acc + r, X_{13-k}.im
 *   = acc - r), one shared code shape.  Sweep I re-reads the untouched im
 *   component; in-place is safe because sweep R only overwrites re.
 *   Fold/sign tables are chunk13p's (verified against L13_ACC15/ACC6R).
 *
 * OPERATION COUNT (per pencil = 13 sites x 8 volumes)
 *   Sweep R: 6 a-adds + 6 spill-subs + 6 w-subs + 6 P0-adds + 36 cos FMA +
 *   36 sin FMA/mul + 12 combine = 108;  sweep I: 96 (w comes from the spill,
 *   no subs).  204+6 vector FP / pencil, 507 pencils per group-step
 *   = 106.5k FP -> 53.2k cyc/group at 2 FMA pipes = 2.30 us/volume-step,
 *   ZERO shuffle uops.  Map: 2197 ladders/group (~12 FP + 1 sqrt + 1 rcp14
 *   each) = +0.57 us/vol on the FMA pipes, sqrt on the otherwise-idle
 *   divider, hidden under the adjacent pencils' FMA streams by OoO (each
 *   pencil is ~260 uops, ~2 fit in the window).  Floor ~2.9 us/vol vs my
 *   ice_r6 5.29 measured and 1000f989's 3.89 measured.
 *
 * Working set per group: state 283 KB + c 283 KB (+320 B stagger so the two
 * buffers' 4K residues differ) = 566 KB, L2-resident on the node (1.25 MB).
 * Batch handling: floor(batch/8) SoA groups, remainder volumes through the
 * unchanged ice_r6 classic path.  s8 vs classic is a DETERMINISTIC dispatch
 * on batch -- never raced (the two paths are not bit-identical; adoption
 * across them would break the monitor's repeatability cmp).  -DL13_S8=0
 * disables the path entirely.
 * =============================================================================
 */
#ifndef L13_S8
#  define L13_S8 1
#endif
#if L13_S8

#define S8LD(q) (*(const vd_w4 *)(q))
#define S8ST(q, v) (*(vd_w4 *)(q) = (v))

/* per-j accumulate: cos row into A1..A6 (+u into A0), sin row into R1..R6.
 * Fold/sign source: L13_ACC15/L13_ACC6R (index m = fold((k*j) mod 13),
 * sign = -1 iff (k*j) mod 13 > 6).  u, w must be in scope. */
#define S8ACC_J1                                                               \
    A0 += u; A1 += C0*u; A2 += C1*u; A3 += C2*u; A4 += C3*u; A5 += C4*u; A6 += C5*u; \
    R1 = S0*w; R2 = S1*w; R3 = S2*w; R4 = S3*w; R5 = S4*w; R6 = S5*w;
#define S8ACC_J2                                                               \
    A0 += u; A1 += C1*u; A2 += C3*u; A3 += C5*u; A4 += C4*u; A5 += C2*u; A6 += C0*u; \
    R1 += S1*w; R2 += S3*w; R3 += S5*w; R4 -= S4*w; R5 -= S2*w; R6 -= S0*w;
#define S8ACC_J3                                                               \
    A0 += u; A1 += C2*u; A2 += C5*u; A3 += C3*u; A4 += C0*u; A5 += C1*u; A6 += C4*u; \
    R1 += S2*w; R2 += S5*w; R3 -= S3*w; R4 -= S0*w; R5 += S1*w; R6 += S4*w;
#define S8ACC_J4                                                               \
    A0 += u; A1 += C3*u; A2 += C4*u; A3 += C0*u; A4 += C2*u; A5 += C5*u; A6 += C1*u; \
    R1 += S3*w; R2 -= S4*w; R3 -= S0*w; R4 += S2*w; R5 -= S5*w; R6 -= S1*w;
#define S8ACC_J5                                                               \
    A0 += u; A1 += C4*u; A2 += C2*u; A3 += C1*u; A4 += C5*u; A5 += C0*u; A6 += C3*u; \
    R1 += S4*w; R2 -= S2*w; R3 += S1*w; R4 -= S5*w; R5 -= S0*w; R6 += S3*w;
#define S8ACC_J6                                                               \
    A0 += u; A1 += C5*u; A2 += C0*u; A3 += C4*u; A4 += C1*u; A5 += C3*u; A6 += C2*u; \
    R1 += S5*w; R2 -= S0*w; R3 += S4*w; R4 -= S1*w; R5 += S3*w; R6 -= S2*w;

/* map + store one output slot: xi (imag DFT output) in a register, the real
 * output re-read from p+o (stored seconds ago by sweep R: exact-address
 * 64 B load, clean store-forward), c at the SAME offsets.  Same MAPV ladder
 * family as mapld/map2st, Newtons via explicit FMA builtins (the r6 pinned-
 * DAG discipline; s8 is never raced against the interleaved arms, but the
 * pins keep every map in this file the same instruction shape). */
static inline __attribute__((always_inline)) void
l13_s8_mapst(double *restrict p, const double *restrict cq, long o, vd_w4 xi)
{
#define S8FMA_(a, b, c)  (vd_w4)_mm512_fmadd_pd((__m512d)(a), (__m512d)(b), (__m512d)(c))
#define S8FNMA_(a, b, c) (vd_w4)_mm512_fnmadd_pd((__m512d)(a), (__m512d)(b), (__m512d)(c))
    const vd_w4 one = 1.0 + (vd_w4){0};
    vd_w4 ur = S8LD(p + o) + S8LD(cq + o);
    vd_w4 ui = xi + S8LD(cq + o + 8);
    vd_w4 m2 = ur * ur + ui * ui;
#if L13_MAPV >= 2
    vd_w4 a = (vd_w4)_mm512_sqrt_pd((__m512d)m2) + 1.0;
#else
    m2 = (vd_w4)_mm512_max_pd((__m512d)m2, _mm512_set1_pd(1e-300));
    vd_w4 y = (vd_w4)_mm512_rsqrt14_pd((__m512d)m2);
    { vd_w4 q = m2 * y; vd_w4 r = S8FNMA_(q, y, one); y = S8FMA_(0.5 * y, r, y); }
    { vd_w4 q = m2 * y; vd_w4 r = S8FNMA_(q, y, one); y = S8FMA_(0.5 * y, r, y); }
    vd_w4 a = S8FMA_(m2, y, one);
#endif
#if L13_MAPV == 1 || L13_MAPV == 2
    vd_w4 s = (vd_w4)_mm512_rcp14_pd((__m512d)a);
    s = S8FMA_(s, S8FNMA_(a, s, one), s);
    s = S8FMA_(s, S8FNMA_(a, s, one), s);
#else
    vd_w4 s = 1.0 / a;
#endif
#undef S8FMA_
#undef S8FNMA_
    S8ST(p + o, ur * s);
    S8ST(p + o + 8, ui * s);
}

/* ICE_R8: the bare map ladder on values whose c is ALREADY added (the lazy
 * "lz" shape below stores raw z+c between steps).  Same MAPV family and
 * explicit-FMA pins as mapst; register-light, so it can run as a slab in
 * front of a pencil without touching the pencil's 25-register budget (the
 * r4 inline-map lesson: map temps and pinned FFT constants must not share
 * a register window). */
static inline __attribute__((always_inline)) void
l13_s8_mapv(vd_w4 ur, vd_w4 ui, vd_w4 *pr, vd_w4 *pi)
{
#define S8FMA2_(a, b, c)  (vd_w4)_mm512_fmadd_pd((__m512d)(a), (__m512d)(b), (__m512d)(c))
#define S8FNMA2_(a, b, c) (vd_w4)_mm512_fnmadd_pd((__m512d)(a), (__m512d)(b), (__m512d)(c))
    const vd_w4 one = 1.0 + (vd_w4){0};
    vd_w4 m2 = ur * ur + ui * ui;
#if L13_MAPV >= 2
    vd_w4 a = (vd_w4)_mm512_sqrt_pd((__m512d)m2) + 1.0;
#else
    m2 = (vd_w4)_mm512_max_pd((__m512d)m2, _mm512_set1_pd(1e-300));
    vd_w4 y = (vd_w4)_mm512_rsqrt14_pd((__m512d)m2);
    { vd_w4 q = m2 * y; vd_w4 r = S8FNMA2_(q, y, one); y = S8FMA2_(0.5 * y, r, y); }
    { vd_w4 q = m2 * y; vd_w4 r = S8FNMA2_(q, y, one); y = S8FMA2_(0.5 * y, r, y); }
    vd_w4 a = S8FMA2_(m2, y, one);
#endif
#if L13_MAPV == 1 || L13_MAPV == 2
    vd_w4 s = (vd_w4)_mm512_rcp14_pd((__m512d)a);
    s = S8FMA2_(s, S8FNMA2_(a, s, one), s);
    s = S8FMA2_(s, S8FNMA2_(a, s, one), s);
#else
    vd_w4 s = 1.0 / a;
#endif
#undef S8FMA2_
#undef S8FNMA2_
    *pr = ur * s;
    *pi = ui * s;
}

/* map a z-pencil's 13 contiguous slots (1664 B, L1/L2-hot) in place, ahead
 * of the plain pencil that re-reads them (exact-address 64 B store-forward).
 * Pencil-granular so the 13 independent ladders overlap the neighbouring
 * pencils' FMA streams; this is the warm rival d43251c2's MAPZB shape (its
 * default at L=13 integrates the map into a de-pinned phase-split codelet;
 * the slab form keeps MY pinned two-sweep pencil untouched). */
static inline __attribute__((always_inline)) void
l13_s8_mapblk(double *restrict p)
{
#pragma GCC unroll 1
    for (int e = 0; e < 13; ++e) {
        vd_w4 ur = S8LD(p + e * 16), ui = S8LD(p + e * 16 + 8);
        l13_s8_mapv(ur, ui, &ur, &ui);
        S8ST(p + e * 16, ur);
        S8ST(p + e * 16 + 8, ui);
    }
}

/* map + store one output slot with BOTH components in registers (no
 * store-forward reload of the real half -- the two-sweep form's mapst had
 * to re-read it).  Only reachable from the Rader-split pencil below. */
static inline __attribute__((always_inline)) void
l13_s8_maprst(double *restrict p, const double *restrict cq, long o,
              __m512d zr, __m512d zi)
{
    vd_w4 ur = (vd_w4)zr + S8LD(cq + o);
    vd_w4 ui = (vd_w4)zi + S8LD(cq + o + 8);
    l13_s8_mapv(ur, ui, &ur, &ui);
    S8ST(p + o, ur);
    S8ST(p + o + 8, ui);
}

/* ICE_R8: Rader/CRT split pencil, adopted VERBATIM from L13_rader ice_r7
 * (their l13r_dft13s_r / L13R_R_BODY; their node A/B in this exact layout:
 * rader-split 4.183 vs the v6 Hartley reg6 5.457).  186 vector FP per
 * pencil vs the two-sweep dense form's 210, ONE read of each component
 * instead of two, no 384 B spill strip.  Math: rows paired (g^t, 13-g^t),
 * g^t = 1,2,4,8,3,6; u_t folded again into 3-cyclic P (CP constants) and
 * 3-negacyclic Q (CM); sine side is a 6-cyclic sweep on v_t with the sign
 * fold sig = -1 when n+t >= 6.  X[g^n] = (a_n +- b_n) +- (-i sin) terms.
 * Constants live in two SCOPED blocks (6 cos, then 6 sin) so peak pressure
 * stays under the EVEX file; cc0..cc5 pairs spill to stack in the sine
 * block by design (theirs do too).  domap = 1 maps sweep-final stores via
 * maprst (register pair, c at the same offsets). */
static inline __attribute__((always_inline)) void
l13_s8_pencil_r(double *restrict p, const long SS, const double *restrict cq,
                const int domap, const double *restrict rk)
{
#define RLD(er, ei, k)                                                         \
    do { er = _mm512_load_pd(p + (long)(k) * SS);                              \
         ei = _mm512_load_pd(p + (long)(k) * SS + 8); } while (0)
#define RST(k, r, i)                                                           \
    do { if (!domap) {                                                         \
             _mm512_store_pd(p + (long)(k) * SS, r);                           \
             _mm512_store_pd(p + (long)(k) * SS + 8, i);                       \
         } else                                                                \
             l13_s8_maprst(p, cq, (long)(k) * SS, r, i);                       \
    } while (0)
    __m512d x0r, x0i;
    RLD(x0r, x0i, 0);
    const __m512d C0 = _mm512_load_pd(rk + 0),  C1 = _mm512_load_pd(rk + 8);
    const __m512d C2 = _mm512_load_pd(rk + 16), C3 = _mm512_load_pd(rk + 24);
    const __m512d C4 = _mm512_load_pd(rk + 32), C5 = _mm512_load_pd(rk + 40);
    __m512d vr0, vr1, vr2, vr3, vr4, vr5, vi0, vi1, vi2, vi3, vi4, vi5;
    __m512d a0r, a1r, a2r, b0r, b1r, b2r, a0i, a1i, a2i, b0i, b1i, b2i;
    __m512d dcr = x0r, dci = x0i;
    { /* t=0/3: rows (1,12), (8,5) */
        __m512d e0r, e0i, f0r, f0i, e1r, e1i, f1r, f1i;
        RLD(e0r, e0i, 1);  RLD(f0r, f0i, 12);
        RLD(e1r, e1i, 8);  RLD(f1r, f1i, 5);
        vr0 = _mm512_sub_pd(e0r, f0r); vi0 = _mm512_sub_pd(e0i, f0i);
        vr3 = _mm512_sub_pd(e1r, f1r); vi3 = _mm512_sub_pd(e1i, f1i);
        __m512d u0r = _mm512_add_pd(e0r, f0r), u1r = _mm512_add_pd(e1r, f1r);
        __m512d u0i = _mm512_add_pd(e0i, f0i), u1i = _mm512_add_pd(e1i, f1i);
        __m512d ppr = _mm512_add_pd(u0r, u1r), qqr = _mm512_sub_pd(u0r, u1r);
        __m512d ppi = _mm512_add_pd(u0i, u1i), qqi = _mm512_sub_pd(u0i, u1i);
        dcr = _mm512_add_pd(dcr, ppr); dci = _mm512_add_pd(dci, ppi);
        a0r = _mm512_fmadd_pd(ppr, C0, x0r); a1r = _mm512_fmadd_pd(ppr, C1, x0r);
        a2r = _mm512_fmadd_pd(ppr, C2, x0r);
        a0i = _mm512_fmadd_pd(ppi, C0, x0i); a1i = _mm512_fmadd_pd(ppi, C1, x0i);
        a2i = _mm512_fmadd_pd(ppi, C2, x0i);
        b0r = _mm512_mul_pd(qqr, C3); b1r = _mm512_mul_pd(qqr, C4);
        b2r = _mm512_mul_pd(qqr, C5);
        b0i = _mm512_mul_pd(qqi, C3); b1i = _mm512_mul_pd(qqi, C4);
        b2i = _mm512_mul_pd(qqi, C5);
    }
    { /* t=1/4: rows (2,11), (3,10) */
        __m512d e0r, e0i, f0r, f0i, e1r, e1i, f1r, f1i;
        RLD(e0r, e0i, 2);  RLD(f0r, f0i, 11);
        RLD(e1r, e1i, 3);  RLD(f1r, f1i, 10);
        vr1 = _mm512_sub_pd(e0r, f0r); vi1 = _mm512_sub_pd(e0i, f0i);
        vr4 = _mm512_sub_pd(e1r, f1r); vi4 = _mm512_sub_pd(e1i, f1i);
        __m512d u0r = _mm512_add_pd(e0r, f0r), u1r = _mm512_add_pd(e1r, f1r);
        __m512d u0i = _mm512_add_pd(e0i, f0i), u1i = _mm512_add_pd(e1i, f1i);
        __m512d ppr = _mm512_add_pd(u0r, u1r), qqr = _mm512_sub_pd(u0r, u1r);
        __m512d ppi = _mm512_add_pd(u0i, u1i), qqi = _mm512_sub_pd(u0i, u1i);
        dcr = _mm512_add_pd(dcr, ppr); dci = _mm512_add_pd(dci, ppi);
        a0r = _mm512_fmadd_pd(ppr, C1, a0r); a1r = _mm512_fmadd_pd(ppr, C2, a1r);
        a2r = _mm512_fmadd_pd(ppr, C0, a2r);
        a0i = _mm512_fmadd_pd(ppi, C1, a0i); a1i = _mm512_fmadd_pd(ppi, C2, a1i);
        a2i = _mm512_fmadd_pd(ppi, C0, a2i);
        b0r = _mm512_fmadd_pd(qqr, C4, b0r); b1r = _mm512_fmadd_pd(qqr, C5, b1r);
        b2r = _mm512_fnmadd_pd(qqr, C3, b2r);
        b0i = _mm512_fmadd_pd(qqi, C4, b0i); b1i = _mm512_fmadd_pd(qqi, C5, b1i);
        b2i = _mm512_fnmadd_pd(qqi, C3, b2i);
    }
    { /* t=2/5: rows (4,9), (6,7) */
        __m512d e0r, e0i, f0r, f0i, e1r, e1i, f1r, f1i;
        RLD(e0r, e0i, 4);  RLD(f0r, f0i, 9);
        RLD(e1r, e1i, 6);  RLD(f1r, f1i, 7);
        vr2 = _mm512_sub_pd(e0r, f0r); vi2 = _mm512_sub_pd(e0i, f0i);
        vr5 = _mm512_sub_pd(e1r, f1r); vi5 = _mm512_sub_pd(e1i, f1i);
        __m512d u0r = _mm512_add_pd(e0r, f0r), u1r = _mm512_add_pd(e1r, f1r);
        __m512d u0i = _mm512_add_pd(e0i, f0i), u1i = _mm512_add_pd(e1i, f1i);
        __m512d ppr = _mm512_add_pd(u0r, u1r), qqr = _mm512_sub_pd(u0r, u1r);
        __m512d ppi = _mm512_add_pd(u0i, u1i), qqi = _mm512_sub_pd(u0i, u1i);
        dcr = _mm512_add_pd(dcr, ppr); dci = _mm512_add_pd(dci, ppi);
        a0r = _mm512_fmadd_pd(ppr, C2, a0r); a1r = _mm512_fmadd_pd(ppr, C0, a1r);
        a2r = _mm512_fmadd_pd(ppr, C1, a2r);
        a0i = _mm512_fmadd_pd(ppi, C2, a0i); a1i = _mm512_fmadd_pd(ppi, C0, a1i);
        a2i = _mm512_fmadd_pd(ppi, C1, a2i);
        b0r = _mm512_fmadd_pd(qqr, C5, b0r); b1r = _mm512_fnmadd_pd(qqr, C3, b1r);
        b2r = _mm512_fnmadd_pd(qqr, C4, b2r);
        b0i = _mm512_fmadd_pd(qqi, C5, b0i); b1i = _mm512_fnmadd_pd(qqi, C3, b1i);
        b2i = _mm512_fnmadd_pd(qqi, C4, b2i);
    }
    __m512d cc0r = _mm512_add_pd(a0r, b0r), cc3r = _mm512_sub_pd(a0r, b0r);
    __m512d cc1r = _mm512_add_pd(a1r, b1r), cc4r = _mm512_sub_pd(a1r, b1r);
    __m512d cc2r = _mm512_add_pd(a2r, b2r), cc5r = _mm512_sub_pd(a2r, b2r);
    __m512d cc0i = _mm512_add_pd(a0i, b0i), cc3i = _mm512_sub_pd(a0i, b0i);
    __m512d cc1i = _mm512_add_pd(a1i, b1i), cc4i = _mm512_sub_pd(a1i, b1i);
    __m512d cc2i = _mm512_add_pd(a2i, b2i), cc5i = _mm512_sub_pd(a2i, b2i);
    RST(0, dcr, dci);
    {
        const __m512d S0 = _mm512_load_pd(rk + 48), S1 = _mm512_load_pd(rk + 56);
        const __m512d S2 = _mm512_load_pd(rk + 64), S3 = _mm512_load_pd(rk + 72);
        const __m512d S4 = _mm512_load_pd(rk + 80), S5 = _mm512_load_pd(rk + 88);
        __m512d TR0 = _mm512_mul_pd(vi0, S0), TI0 = _mm512_mul_pd(vr0, S0);
        __m512d TR1 = _mm512_mul_pd(vi0, S1), TI1 = _mm512_mul_pd(vr0, S1);
        __m512d TR2 = _mm512_mul_pd(vi0, S2), TI2 = _mm512_mul_pd(vr0, S2);
        __m512d TR3 = _mm512_mul_pd(vi0, S3), TI3 = _mm512_mul_pd(vr0, S3);
        __m512d TR4 = _mm512_mul_pd(vi0, S4), TI4 = _mm512_mul_pd(vr0, S4);
        __m512d TR5 = _mm512_mul_pd(vi0, S5), TI5 = _mm512_mul_pd(vr0, S5);
        TR0 = _mm512_fmadd_pd(vi1, S1, TR0);  TI0 = _mm512_fmadd_pd(vr1, S1, TI0);
        TR1 = _mm512_fmadd_pd(vi1, S2, TR1);  TI1 = _mm512_fmadd_pd(vr1, S2, TI1);
        TR2 = _mm512_fmadd_pd(vi1, S3, TR2);  TI2 = _mm512_fmadd_pd(vr1, S3, TI2);
        TR3 = _mm512_fmadd_pd(vi1, S4, TR3);  TI3 = _mm512_fmadd_pd(vr1, S4, TI3);
        TR4 = _mm512_fmadd_pd(vi1, S5, TR4);  TI4 = _mm512_fmadd_pd(vr1, S5, TI4);
        TR5 = _mm512_fnmadd_pd(vi1, S0, TR5); TI5 = _mm512_fnmadd_pd(vr1, S0, TI5);
        TR0 = _mm512_fmadd_pd(vi2, S2, TR0);  TI0 = _mm512_fmadd_pd(vr2, S2, TI0);
        TR1 = _mm512_fmadd_pd(vi2, S3, TR1);  TI1 = _mm512_fmadd_pd(vr2, S3, TI1);
        TR2 = _mm512_fmadd_pd(vi2, S4, TR2);  TI2 = _mm512_fmadd_pd(vr2, S4, TI2);
        TR3 = _mm512_fmadd_pd(vi2, S5, TR3);  TI3 = _mm512_fmadd_pd(vr2, S5, TI3);
        TR4 = _mm512_fnmadd_pd(vi2, S0, TR4); TI4 = _mm512_fnmadd_pd(vr2, S0, TI4);
        TR5 = _mm512_fnmadd_pd(vi2, S1, TR5); TI5 = _mm512_fnmadd_pd(vr2, S1, TI5);
        TR0 = _mm512_fmadd_pd(vi3, S3, TR0);  TI0 = _mm512_fmadd_pd(vr3, S3, TI0);
        TR1 = _mm512_fmadd_pd(vi3, S4, TR1);  TI1 = _mm512_fmadd_pd(vr3, S4, TI1);
        TR2 = _mm512_fmadd_pd(vi3, S5, TR2);  TI2 = _mm512_fmadd_pd(vr3, S5, TI2);
        TR3 = _mm512_fnmadd_pd(vi3, S0, TR3); TI3 = _mm512_fnmadd_pd(vr3, S0, TI3);
        TR4 = _mm512_fnmadd_pd(vi3, S1, TR4); TI4 = _mm512_fnmadd_pd(vr3, S1, TI4);
        TR5 = _mm512_fnmadd_pd(vi3, S2, TR5); TI5 = _mm512_fnmadd_pd(vr3, S2, TI5);
        TR0 = _mm512_fmadd_pd(vi4, S4, TR0);  TI0 = _mm512_fmadd_pd(vr4, S4, TI0);
        TR1 = _mm512_fmadd_pd(vi4, S5, TR1);  TI1 = _mm512_fmadd_pd(vr4, S5, TI1);
        TR2 = _mm512_fnmadd_pd(vi4, S0, TR2); TI2 = _mm512_fnmadd_pd(vr4, S0, TI2);
        TR3 = _mm512_fnmadd_pd(vi4, S1, TR3); TI3 = _mm512_fnmadd_pd(vr4, S1, TI3);
        TR4 = _mm512_fnmadd_pd(vi4, S2, TR4); TI4 = _mm512_fnmadd_pd(vr4, S2, TI4);
        TR5 = _mm512_fnmadd_pd(vi4, S3, TR5); TI5 = _mm512_fnmadd_pd(vr4, S3, TI5);
        TR0 = _mm512_fmadd_pd(vi5, S5, TR0);  TI0 = _mm512_fmadd_pd(vr5, S5, TI0);
        TR1 = _mm512_fnmadd_pd(vi5, S0, TR1); TI1 = _mm512_fnmadd_pd(vr5, S0, TI1);
        TR2 = _mm512_fnmadd_pd(vi5, S1, TR2); TI2 = _mm512_fnmadd_pd(vr5, S1, TI2);
        TR3 = _mm512_fnmadd_pd(vi5, S2, TR3); TI3 = _mm512_fnmadd_pd(vr5, S2, TI3);
        TR4 = _mm512_fnmadd_pd(vi5, S3, TR4); TI4 = _mm512_fnmadd_pd(vr5, S3, TI4);
        TR5 = _mm512_fnmadd_pd(vi5, S4, TR5); TI5 = _mm512_fnmadd_pd(vr5, S4, TI5);
        RST(1,  _mm512_add_pd(cc0r, TR0), _mm512_sub_pd(cc0i, TI0));
        RST(12, _mm512_sub_pd(cc0r, TR0), _mm512_add_pd(cc0i, TI0));
        RST(2,  _mm512_add_pd(cc1r, TR1), _mm512_sub_pd(cc1i, TI1));
        RST(11, _mm512_sub_pd(cc1r, TR1), _mm512_add_pd(cc1i, TI1));
        RST(3,  _mm512_add_pd(cc4r, TR4), _mm512_sub_pd(cc4i, TI4));
        RST(10, _mm512_sub_pd(cc4r, TR4), _mm512_add_pd(cc4i, TI4));
        RST(4,  _mm512_add_pd(cc2r, TR2), _mm512_sub_pd(cc2i, TI2));
        RST(9,  _mm512_sub_pd(cc2r, TR2), _mm512_add_pd(cc2i, TI2));
        RST(8,  _mm512_add_pd(cc3r, TR3), _mm512_sub_pd(cc3i, TI3));
        RST(5,  _mm512_sub_pd(cc3r, TR3), _mm512_add_pd(cc3i, TI3));
        RST(6,  _mm512_add_pd(cc5r, TR5), _mm512_sub_pd(cc5i, TI5));
        RST(7,  _mm512_sub_pd(cc5r, TR5), _mm512_add_pd(cc5i, TI5));
    }
#undef RLD
#undef RST
}

/* One in-place SoA pencil: 13 slots at p, p+SS, ..., p+12*SS (re at +0, im
 * at +8).  domap (compile-time at every call site) selects the store shape:
 *   0 = plain stores;
 *   1 = chain map fused into sweep I's stores (mapst) against the c slots
 *       at cq + same offsets (the r7 default);
 *   2 = ICE_R8 lazy: ADD c raw at both sweeps' stores (no ladder) -- the
 *       buffer holds raw z+c between steps and the ladder runs at the next
 *       step's z-pencil loads (l13_s8_mapblk) / at exit (l13_s8_out_lz).
 * sp = 384 B L1 spill strip for the 6 cross-sweep nbr vectors. */
static inline __attribute__((always_inline)) void
l13_s8_pencil(double *restrict p, const long SS, const double *restrict cq,
              const int domap, double *restrict sp,
              vd_w4 C0, vd_w4 C1, vd_w4 C2, vd_w4 C3, vd_w4 C4, vd_w4 C5,
              vd_w4 S0, vd_w4 S1, vd_w4 S2, vd_w4 S3, vd_w4 S4, vd_w4 S5)
{
    vd_w4 A0, A1, A2, A3, A4, A5, A6, R1, R2, R3, R4, R5, R6;

    /* ---- sweep R: real outputs; spill nbr_j = re_{13-j} - re_j ---- */
    A0 = A1 = A2 = A3 = A4 = A5 = A6 = S8LD(p);
    {
        vd_w4 aj, bj, u, w;
        aj = S8LD(p + 1 * SS); bj = S8LD(p + 12 * SS);
        u = aj + bj; S8ST(sp + 0, bj - aj);
        w = S8LD(p + 8 + 1 * SS) - S8LD(p + 8 + 12 * SS);
        S8ACC_J1
        aj = S8LD(p + 2 * SS); bj = S8LD(p + 11 * SS);
        u = aj + bj; S8ST(sp + 8, bj - aj);
        w = S8LD(p + 8 + 2 * SS) - S8LD(p + 8 + 11 * SS);
        S8ACC_J2
        aj = S8LD(p + 3 * SS); bj = S8LD(p + 10 * SS);
        u = aj + bj; S8ST(sp + 16, bj - aj);
        w = S8LD(p + 8 + 3 * SS) - S8LD(p + 8 + 10 * SS);
        S8ACC_J3
        aj = S8LD(p + 4 * SS); bj = S8LD(p + 9 * SS);
        u = aj + bj; S8ST(sp + 24, bj - aj);
        w = S8LD(p + 8 + 4 * SS) - S8LD(p + 8 + 9 * SS);
        S8ACC_J4
        aj = S8LD(p + 5 * SS); bj = S8LD(p + 8 * SS);
        u = aj + bj; S8ST(sp + 32, bj - aj);
        w = S8LD(p + 8 + 5 * SS) - S8LD(p + 8 + 8 * SS);
        S8ACC_J5
        aj = S8LD(p + 6 * SS); bj = S8LD(p + 7 * SS);
        u = aj + bj; S8ST(sp + 40, bj - aj);
        w = S8LD(p + 8 + 6 * SS) - S8LD(p + 8 + 7 * SS);
        S8ACC_J6
    }
    if (domap != 2) {
        S8ST(p, A0);
        S8ST(p + 1 * SS, A1 + R1);  S8ST(p + 12 * SS, A1 - R1);
        S8ST(p + 2 * SS, A2 + R2);  S8ST(p + 11 * SS, A2 - R2);
        S8ST(p + 3 * SS, A3 + R3);  S8ST(p + 10 * SS, A3 - R3);
        S8ST(p + 4 * SS, A4 + R4);  S8ST(p + 9 * SS, A4 - R4);
        S8ST(p + 5 * SS, A5 + R5);  S8ST(p + 8 * SS, A5 - R5);
        S8ST(p + 6 * SS, A6 + R6);  S8ST(p + 7 * SS, A6 - R6);
    } else {
        S8ST(p, A0 + S8LD(cq));
        S8ST(p + 1 * SS, A1 + R1 + S8LD(cq + 1 * SS));
        S8ST(p + 12 * SS, A1 - R1 + S8LD(cq + 12 * SS));
        S8ST(p + 2 * SS, A2 + R2 + S8LD(cq + 2 * SS));
        S8ST(p + 11 * SS, A2 - R2 + S8LD(cq + 11 * SS));
        S8ST(p + 3 * SS, A3 + R3 + S8LD(cq + 3 * SS));
        S8ST(p + 10 * SS, A3 - R3 + S8LD(cq + 10 * SS));
        S8ST(p + 4 * SS, A4 + R4 + S8LD(cq + 4 * SS));
        S8ST(p + 9 * SS, A4 - R4 + S8LD(cq + 9 * SS));
        S8ST(p + 5 * SS, A5 + R5 + S8LD(cq + 5 * SS));
        S8ST(p + 8 * SS, A5 - R5 + S8LD(cq + 8 * SS));
        S8ST(p + 6 * SS, A6 + R6 + S8LD(cq + 6 * SS));
        S8ST(p + 7 * SS, A6 - R6 + S8LD(cq + 7 * SS));
    }

    /* ---- sweep I: imag outputs from the untouched im component + the
     * spilled nbr strip; identical combine by the sign folded into nbr ---- */
    A0 = A1 = A2 = A3 = A4 = A5 = A6 = S8LD(p + 8);
    {
        vd_w4 u, w;
        u = S8LD(p + 8 + 1 * SS) + S8LD(p + 8 + 12 * SS); w = S8LD(sp + 0);
        S8ACC_J1
        u = S8LD(p + 8 + 2 * SS) + S8LD(p + 8 + 11 * SS); w = S8LD(sp + 8);
        S8ACC_J2
        u = S8LD(p + 8 + 3 * SS) + S8LD(p + 8 + 10 * SS); w = S8LD(sp + 16);
        S8ACC_J3
        u = S8LD(p + 8 + 4 * SS) + S8LD(p + 8 + 9 * SS); w = S8LD(sp + 24);
        S8ACC_J4
        u = S8LD(p + 8 + 5 * SS) + S8LD(p + 8 + 8 * SS); w = S8LD(sp + 32);
        S8ACC_J5
        u = S8LD(p + 8 + 6 * SS) + S8LD(p + 8 + 7 * SS); w = S8LD(sp + 40);
        S8ACC_J6
    }
    if (!domap) {
        S8ST(p + 8, A0);
        S8ST(p + 8 + 1 * SS, A1 + R1);  S8ST(p + 8 + 12 * SS, A1 - R1);
        S8ST(p + 8 + 2 * SS, A2 + R2);  S8ST(p + 8 + 11 * SS, A2 - R2);
        S8ST(p + 8 + 3 * SS, A3 + R3);  S8ST(p + 8 + 10 * SS, A3 - R3);
        S8ST(p + 8 + 4 * SS, A4 + R4);  S8ST(p + 8 + 9 * SS, A4 - R4);
        S8ST(p + 8 + 5 * SS, A5 + R5);  S8ST(p + 8 + 8 * SS, A5 - R5);
        S8ST(p + 8 + 6 * SS, A6 + R6);  S8ST(p + 8 + 7 * SS, A6 - R6);
    } else if (domap == 2) {
        S8ST(p + 8, A0 + S8LD(cq + 8));
        S8ST(p + 8 + 1 * SS, A1 + R1 + S8LD(cq + 8 + 1 * SS));
        S8ST(p + 8 + 12 * SS, A1 - R1 + S8LD(cq + 8 + 12 * SS));
        S8ST(p + 8 + 2 * SS, A2 + R2 + S8LD(cq + 8 + 2 * SS));
        S8ST(p + 8 + 11 * SS, A2 - R2 + S8LD(cq + 8 + 11 * SS));
        S8ST(p + 8 + 3 * SS, A3 + R3 + S8LD(cq + 8 + 3 * SS));
        S8ST(p + 8 + 10 * SS, A3 - R3 + S8LD(cq + 8 + 10 * SS));
        S8ST(p + 8 + 4 * SS, A4 + R4 + S8LD(cq + 8 + 4 * SS));
        S8ST(p + 8 + 9 * SS, A4 - R4 + S8LD(cq + 8 + 9 * SS));
        S8ST(p + 8 + 5 * SS, A5 + R5 + S8LD(cq + 8 + 5 * SS));
        S8ST(p + 8 + 8 * SS, A5 - R5 + S8LD(cq + 8 + 8 * SS));
        S8ST(p + 8 + 6 * SS, A6 + R6 + S8LD(cq + 8 + 6 * SS));
        S8ST(p + 8 + 7 * SS, A6 - R6 + S8LD(cq + 8 + 7 * SS));
    } else {
        l13_s8_mapst(p, cq, 0, A0);
        l13_s8_mapst(p, cq, 1 * SS, A1 + R1);
        l13_s8_mapst(p, cq, 12 * SS, A1 - R1);
        l13_s8_mapst(p, cq, 2 * SS, A2 + R2);
        l13_s8_mapst(p, cq, 11 * SS, A2 - R2);
        l13_s8_mapst(p, cq, 3 * SS, A3 + R3);
        l13_s8_mapst(p, cq, 10 * SS, A3 - R3);
        l13_s8_mapst(p, cq, 4 * SS, A4 + R4);
        l13_s8_mapst(p, cq, 9 * SS, A4 - R4);
        l13_s8_mapst(p, cq, 5 * SS, A5 + R5);
        l13_s8_mapst(p, cq, 8 * SS, A5 - R5);
        l13_s8_mapst(p, cq, 6 * SS, A6 + R6);
        l13_s8_mapst(p, cq, 7 * SS, A6 - R6);
    }
}

/* One full chain step over one SoA group, in place: z = FFT3(state) then
 * state = (z+c)/(1+|z+c|), the map fused into the last pass's stores
 * (1000f989's GEN_DRIVER_A shape).  Pass grouping for locality: each
 * y-plane visit runs its 13 z-pencils then its 13 x-pencils on the same
 * L1-hot 21.6 KB; the x-plane visit runs y-pencils + map on a contiguous
 * L1-hot 21.75 KB slab.  Axis order z, x, y. */
static __attribute__((unused)) void
l13_s8_iter(double *restrict b, const double *restrict cb,
            const double *restrict cd, const double *restrict sd)
{
    vd_w4 C0 = S8LD(cd + 0),  C1 = S8LD(cd + 8),  C2 = S8LD(cd + 16),
          C3 = S8LD(cd + 24), C4 = S8LD(cd + 32), C5 = S8LD(cd + 40);
    vd_w4 S0 = S8LD(sd + 0),  S1 = S8LD(sd + 8),  S2 = S8LD(sd + 16),
          S3 = S8LD(sd + 24), S4 = S8LD(sd + 32), S5 = S8LD(sd + 40);
    __asm__("" : "+v"(C0), "+v"(C1), "+v"(C2), "+v"(C3), "+v"(C4), "+v"(C5),
                 "+v"(S0), "+v"(S1), "+v"(S2), "+v"(S3), "+v"(S4), "+v"(S5));
    double sp[48] __attribute__((aligned(64)));

    for (int y = 0; y < 13; ++y) {
        double *pl = b + (long)y * L13_S8PY;
        for (int x = 0; x < 13; ++x)
            l13_s8_pencil(pl + (long)x * L13_S8PX, L13_S8Z, 0, 0, sp,
                          C0, C1, C2, C3, C4, C5, S0, S1, S2, S3, S4, S5);
        for (int z = 0; z < 13; ++z)
            l13_s8_pencil(pl + (long)z * L13_S8Z, L13_S8PX, 0, 0, sp,
                          C0, C1, C2, C3, C4, C5, S0, S1, S2, S3, S4, S5);
    }
    for (int x = 0; x < 13; ++x) {
        double *pl = b + (long)x * L13_S8PX;
        const double *cl = cb + (long)x * L13_S8PX;
        for (int z = 0; z < 13; ++z)
            l13_s8_pencil(pl + (long)z * L13_S8Z, L13_S8PY,
                          cl + (long)z * L13_S8Z, 1, sp,
                          C0, C1, C2, C3, C4, C5, S0, S1, S2, S3, S4, S5);
    }
}

/* ICE_R8 lazy step ("lz"; the warm rival d43251c2's L=13 shape re-derived
 * on my pencil): the y-pencil stores ADD c raw (2 vector adds per slot, no
 * ladder on the store path at all), the buffer holds raw z+c between steps,
 * and the ladder runs as a pencil-granular slab in front of the NEXT step's
 * z-pencils, where the 13 slots are one contiguous L1-hot 1664 B strip and
 * the 13 independent ladders overlap the neighbouring pencils' FMA streams.
 * mapz = 0 on step 1 (the entry state is already mapped by contract). */
static void
l13_s8_iter_lz(double *restrict b, const double *restrict cb,
               const double *restrict cd, const double *restrict sd,
               const int mapz)
{
    vd_w4 C0 = S8LD(cd + 0),  C1 = S8LD(cd + 8),  C2 = S8LD(cd + 16),
          C3 = S8LD(cd + 24), C4 = S8LD(cd + 32), C5 = S8LD(cd + 40);
    vd_w4 S0 = S8LD(sd + 0),  S1 = S8LD(sd + 8),  S2 = S8LD(sd + 16),
          S3 = S8LD(sd + 24), S4 = S8LD(sd + 32), S5 = S8LD(sd + 40);
    __asm__("" : "+v"(C0), "+v"(C1), "+v"(C2), "+v"(C3), "+v"(C4), "+v"(C5),
                 "+v"(S0), "+v"(S1), "+v"(S2), "+v"(S3), "+v"(S4), "+v"(S5));
    double sp[48] __attribute__((aligned(64)));

    for (int y = 0; y < 13; ++y) {
        double *pl = b + (long)y * L13_S8PY;
        if (mapz) {
#if L13_S8LZ >= 2
            /* lz2: map the whole y-plane's 13 strips first, then run the 13
             * plain z-pencils -- a full plane of distance between each
             * ladder and its consumer. */
            for (int x = 0; x < 13; ++x)
                l13_s8_mapblk(pl + (long)x * L13_S8PX);
            for (int x = 0; x < 13; ++x)
                l13_s8_pencil(pl + (long)x * L13_S8PX, L13_S8Z, 0, 0, sp,
                              C0, C1, C2, C3, C4, C5, S0, S1, S2, S3, S4, S5);
#else
            for (int x = 0; x < 13; ++x) {
                double *pp = pl + (long)x * L13_S8PX;
                l13_s8_mapblk(pp);
                l13_s8_pencil(pp, L13_S8Z, 0, 0, sp,
                              C0, C1, C2, C3, C4, C5, S0, S1, S2, S3, S4, S5);
            }
#endif
        } else {
            for (int x = 0; x < 13; ++x)
                l13_s8_pencil(pl + (long)x * L13_S8PX, L13_S8Z, 0, 0, sp,
                              C0, C1, C2, C3, C4, C5, S0, S1, S2, S3, S4, S5);
        }
        for (int z = 0; z < 13; ++z)
            l13_s8_pencil(pl + (long)z * L13_S8Z, L13_S8PX, 0, 0, sp,
                          C0, C1, C2, C3, C4, C5, S0, S1, S2, S3, S4, S5);
    }
    for (int x = 0; x < 13; ++x) {
        double *pl = b + (long)x * L13_S8PX;
        const double *cl = cb + (long)x * L13_S8PX;
        for (int z = 0; z < 13; ++z)
            l13_s8_pencil(pl + (long)z * L13_S8Z, L13_S8PY,
                          cl + (long)z * L13_S8Z, 2, sp,
                          C0, C1, C2, C3, C4, C5, S0, S1, S2, S3, S4, S5);
    }
}

/* ---- driver-layout <-> SoA 8x8 transposes (1000f989's tr_fwd/tr_bwd
 * network, verbatim: unpack pairs then two shuffle_f64x2 rounds; 4 complex
 * sites x 8 volumes per call).  Driver volume bases are only 16 B aligned
 * (odd volumes at +35152 B), so the driver side uses unaligned ops; the SoA
 * side is 64 B aligned by construction. ---- */
static inline __attribute__((always_inline)) void
l13_s8_trf(const double *restrict const *vb, long off, double *restrict dst)
{
    __m512d r0 = _mm512_loadu_pd(vb[0] + off), r1 = _mm512_loadu_pd(vb[1] + off),
            r2 = _mm512_loadu_pd(vb[2] + off), r3 = _mm512_loadu_pd(vb[3] + off),
            r4 = _mm512_loadu_pd(vb[4] + off), r5 = _mm512_loadu_pd(vb[5] + off),
            r6 = _mm512_loadu_pd(vb[6] + off), r7 = _mm512_loadu_pd(vb[7] + off);
    __m512d t0 = _mm512_unpacklo_pd(r0, r1), t1 = _mm512_unpackhi_pd(r0, r1);
    __m512d t2 = _mm512_unpacklo_pd(r2, r3), t3 = _mm512_unpackhi_pd(r2, r3);
    __m512d t4 = _mm512_unpacklo_pd(r4, r5), t5 = _mm512_unpackhi_pd(r4, r5);
    __m512d t6 = _mm512_unpacklo_pd(r6, r7), t7 = _mm512_unpackhi_pd(r6, r7);
    __m512d m0 = _mm512_shuffle_f64x2(t0, t2, 0x88), m1 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
    __m512d m2 = _mm512_shuffle_f64x2(t1, t3, 0x88), m3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
    __m512d m4 = _mm512_shuffle_f64x2(t4, t6, 0x88), m5 = _mm512_shuffle_f64x2(t4, t6, 0xDD);
    __m512d m6 = _mm512_shuffle_f64x2(t5, t7, 0x88), m7 = _mm512_shuffle_f64x2(t5, t7, 0xDD);
    _mm512_store_pd(dst + 0,  _mm512_shuffle_f64x2(m0, m4, 0x88));
    _mm512_store_pd(dst + 8,  _mm512_shuffle_f64x2(m2, m6, 0x88));
    _mm512_store_pd(dst + 16, _mm512_shuffle_f64x2(m1, m5, 0x88));
    _mm512_store_pd(dst + 24, _mm512_shuffle_f64x2(m3, m7, 0x88));
    _mm512_store_pd(dst + 32, _mm512_shuffle_f64x2(m0, m4, 0xDD));
    _mm512_store_pd(dst + 40, _mm512_shuffle_f64x2(m2, m6, 0xDD));
    _mm512_store_pd(dst + 48, _mm512_shuffle_f64x2(m1, m5, 0xDD));
    _mm512_store_pd(dst + 56, _mm512_shuffle_f64x2(m3, m7, 0xDD));
}

static inline __attribute__((always_inline)) void
l13_s8_trb(const double *restrict src, double *restrict const *vb, long off)
{
    __m512d r0 = _mm512_load_pd(src + 0),  r1 = _mm512_load_pd(src + 8),
            r2 = _mm512_load_pd(src + 16), r3 = _mm512_load_pd(src + 24),
            r4 = _mm512_load_pd(src + 32), r5 = _mm512_load_pd(src + 40),
            r6 = _mm512_load_pd(src + 48), r7 = _mm512_load_pd(src + 56);
    __m512d t0 = _mm512_unpacklo_pd(r0, r1), t1 = _mm512_unpackhi_pd(r0, r1);
    __m512d t2 = _mm512_unpacklo_pd(r2, r3), t3 = _mm512_unpackhi_pd(r2, r3);
    __m512d t4 = _mm512_unpacklo_pd(r4, r5), t5 = _mm512_unpackhi_pd(r4, r5);
    __m512d t6 = _mm512_unpacklo_pd(r6, r7), t7 = _mm512_unpackhi_pd(r6, r7);
    __m512d m0 = _mm512_shuffle_f64x2(t0, t2, 0x88), m1 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
    __m512d m2 = _mm512_shuffle_f64x2(t1, t3, 0x88), m3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
    __m512d m4 = _mm512_shuffle_f64x2(t4, t6, 0x88), m5 = _mm512_shuffle_f64x2(t4, t6, 0xDD);
    __m512d m6 = _mm512_shuffle_f64x2(t5, t7, 0x88), m7 = _mm512_shuffle_f64x2(t5, t7, 0xDD);
    _mm512_storeu_pd(vb[0] + off, _mm512_shuffle_f64x2(m0, m4, 0x88));
    _mm512_storeu_pd(vb[1] + off, _mm512_shuffle_f64x2(m2, m6, 0x88));
    _mm512_storeu_pd(vb[2] + off, _mm512_shuffle_f64x2(m1, m5, 0x88));
    _mm512_storeu_pd(vb[3] + off, _mm512_shuffle_f64x2(m3, m7, 0x88));
    _mm512_storeu_pd(vb[4] + off, _mm512_shuffle_f64x2(m0, m4, 0xDD));
    _mm512_storeu_pd(vb[5] + off, _mm512_shuffle_f64x2(m2, m6, 0xDD));
    _mm512_storeu_pd(vb[6] + off, _mm512_shuffle_f64x2(m1, m5, 0xDD));
    _mm512_storeu_pd(vb[7] + off, _mm512_shuffle_f64x2(m3, m7, 0xDD));
}

/* z-blocks 0,4,8 + an overlapping one at 9 covering z=9..12 (rewrites
 * z=9..11 with identical bits -- the panel's standard overlap-tail trick;
 * pure copies, so it is exact on entry and exit both). */
static void
l13_s8_in(const double *restrict v0, double *restrict dst)
{
    const double *vb[8];
    for (int l = 0; l < 8; ++l) vb[l] = v0 + (size_t)l * 4394;
    static const int zb[4] = {0, 4, 8, 9};
    for (int x = 0; x < 13; ++x)
        for (int y = 0; y < 13; ++y) {
            long soff = ((long)x * 169 + (long)y * 13) * 2;
            double *drow = dst + (long)x * L13_S8PX + (long)y * L13_S8PY;
            for (int t = 0; t < 4; ++t)
                l13_s8_trf(vb, soff + 2 * zb[t], drow + 16 * zb[t]);
        }
}

static __attribute__((unused)) void
l13_s8_out(const double *restrict src, double *restrict v0)
{
    double *vb[8];
    for (int l = 0; l < 8; ++l) vb[l] = v0 + (size_t)l * 4394;
    static const int zb[4] = {0, 4, 8, 9};
    for (int x = 0; x < 13; ++x)
        for (int y = 0; y < 13; ++y) {
            long soff = ((long)x * 169 + (long)y * 13) * 2;
            const double *srow = src + (long)x * L13_S8PX + (long)y * L13_S8PY;
            for (int t = 0; t < 4; ++t)
                l13_s8_trb(srow + 16 * zb[t], vb, soff + 2 * zb[t]);
        }
}

/* lz exit: the buffer holds raw z+c after the final step's y-pencils, so
 * the map ladder runs on the 4 slots of each tile during the exit
 * transpose (once per CHAIN -- noise).  The overlapping zb block at 9
 * recomputes slots 9..11 from identical inputs => identical bits. */
static inline __attribute__((always_inline)) void
l13_s8_trb_map(const double *restrict src, double *restrict const *vb, long off)
{
    vd_w4 q0 = S8LD(src + 0),  q1 = S8LD(src + 8),
          q2 = S8LD(src + 16), q3 = S8LD(src + 24),
          q4 = S8LD(src + 32), q5 = S8LD(src + 40),
          q6 = S8LD(src + 48), q7 = S8LD(src + 56);
    l13_s8_mapv(q0, q1, &q0, &q1);
    l13_s8_mapv(q2, q3, &q2, &q3);
    l13_s8_mapv(q4, q5, &q4, &q5);
    l13_s8_mapv(q6, q7, &q6, &q7);
    __m512d r0 = (__m512d)q0, r1 = (__m512d)q1, r2 = (__m512d)q2,
            r3 = (__m512d)q3, r4 = (__m512d)q4, r5 = (__m512d)q5,
            r6 = (__m512d)q6, r7 = (__m512d)q7;
    __m512d t0 = _mm512_unpacklo_pd(r0, r1), t1 = _mm512_unpackhi_pd(r0, r1);
    __m512d t2 = _mm512_unpacklo_pd(r2, r3), t3 = _mm512_unpackhi_pd(r2, r3);
    __m512d t4 = _mm512_unpacklo_pd(r4, r5), t5 = _mm512_unpackhi_pd(r4, r5);
    __m512d t6 = _mm512_unpacklo_pd(r6, r7), t7 = _mm512_unpackhi_pd(r6, r7);
    __m512d m0 = _mm512_shuffle_f64x2(t0, t2, 0x88), m1 = _mm512_shuffle_f64x2(t0, t2, 0xDD);
    __m512d m2 = _mm512_shuffle_f64x2(t1, t3, 0x88), m3 = _mm512_shuffle_f64x2(t1, t3, 0xDD);
    __m512d m4 = _mm512_shuffle_f64x2(t4, t6, 0x88), m5 = _mm512_shuffle_f64x2(t4, t6, 0xDD);
    __m512d m6 = _mm512_shuffle_f64x2(t5, t7, 0x88), m7 = _mm512_shuffle_f64x2(t5, t7, 0xDD);
    _mm512_storeu_pd(vb[0] + off, _mm512_shuffle_f64x2(m0, m4, 0x88));
    _mm512_storeu_pd(vb[1] + off, _mm512_shuffle_f64x2(m2, m6, 0x88));
    _mm512_storeu_pd(vb[2] + off, _mm512_shuffle_f64x2(m1, m5, 0x88));
    _mm512_storeu_pd(vb[3] + off, _mm512_shuffle_f64x2(m3, m7, 0x88));
    _mm512_storeu_pd(vb[4] + off, _mm512_shuffle_f64x2(m0, m4, 0xDD));
    _mm512_storeu_pd(vb[5] + off, _mm512_shuffle_f64x2(m2, m6, 0xDD));
    _mm512_storeu_pd(vb[6] + off, _mm512_shuffle_f64x2(m1, m5, 0xDD));
    _mm512_storeu_pd(vb[7] + off, _mm512_shuffle_f64x2(m3, m7, 0xDD));
}

static void
l13_s8_out_lz(const double *restrict src, double *restrict v0)
{
    double *vb[8];
    for (int l = 0; l < 8; ++l) vb[l] = v0 + (size_t)l * 4394;
    static const int zb[4] = {0, 4, 8, 9};
    for (int x = 0; x < 13; ++x)
        for (int y = 0; y < 13; ++y) {
            long soff = ((long)x * 169 + (long)y * 13) * 2;
            const double *srow = src + (long)x * L13_S8PX + (long)y * L13_S8PY;
            for (int t = 0; t < 4; ++t)
                l13_s8_trb_map(srow + 16 * zb[t], vb, soff + 2 * zb[t]);
        }
}

/* ICE_R8 "fz": cross-step sweep fusion (executing my r7 "next structural
 * lever"; 1000f989 does the same at L>=36).  The r7 step is two full buffer
 * passes: pass A (z+x pencils per y-plane) + pass B (y-pencils + map per
 * x-plane) -- so the state streams through L2 twice per step.  But step
 * t+1's z-pencils of x-plane k depend ONLY on step t's y-pencils of the
 * same plane, so they can run in the same visit while the 21.6 KB plane is
 * L1-hot.  Steady state becomes: pass X (x-pencils per y-plane) + pass YZ
 * (per x-plane: 13 y-pencils + map@store, then 13 z-pencils of the NEXT
 * step) -- still 2 passes and identical FP, but the z-pencils' 283 KB of
 * L2 reads per step become L1 hits.  Per-point DAG unchanged =>
 * bit-identical to the r7 iter shape (pencils only reordered across
 * independent pencils). */
/* -DL13_S8SCHED=1: pre-RA scheduling on the s8 passes (the warm rival
 * d43251c2 ships exactly this on its 13-point wrappers, and the r2 lesson
 * says the pragma transfers to PHASE-SPLIT kernels -- which the two-sweep
 * s8 pencil is). */
#if defined(L13_S8SCHED) && L13_S8SCHED
#  define L13_S8OPT __attribute__((optimize("schedule-insns", "sched-pressure")))
#else
#  define L13_S8OPT
#endif

/* -DL13_S8RK=1 swaps the fz passes onto the Rader-split pencil.  Measured
 * on the node (ice_r8, same window): 3.540 vs 3.439 (dense) -- the 24
 * fewer FP lose to the sine block's cc spills and 36-vector liveness; the
 * pinned two-sweep dense pencil schedules better.  Third confirmation
 * (after r3/r5's interleaved tie) that 13-point kernel arithmetic is not
 * the lever at this cell.  Kept for the record. */
#ifndef L13_S8RK
#  define L13_S8RK 0
#endif

static void L13_S8OPT
l13_s8_pass_x(double *restrict b, const fft3d_plan *restrict pp)
{
#if L13_S8RK
    const double *restrict rk = pp->srk8;
    for (int y = 0; y < 13; ++y) {
        double *pl = b + (long)y * L13_S8PY;
        for (int z = 0; z < 13; ++z)
            l13_s8_pencil_r(pl + (long)z * L13_S8Z, L13_S8PX, 0, 0, rk);
    }
#else
    const double *cd = pp->ctd8, *sd = pp->ssd8;
    vd_w4 C0 = S8LD(cd + 0),  C1 = S8LD(cd + 8),  C2 = S8LD(cd + 16),
          C3 = S8LD(cd + 24), C4 = S8LD(cd + 32), C5 = S8LD(cd + 40);
    vd_w4 S0 = S8LD(sd + 0),  S1 = S8LD(sd + 8),  S2 = S8LD(sd + 16),
          S3 = S8LD(sd + 24), S4 = S8LD(sd + 32), S5 = S8LD(sd + 40);
    __asm__("" : "+v"(C0), "+v"(C1), "+v"(C2), "+v"(C3), "+v"(C4), "+v"(C5),
                 "+v"(S0), "+v"(S1), "+v"(S2), "+v"(S3), "+v"(S4), "+v"(S5));
    double sp[48] __attribute__((aligned(64)));
    for (int y = 0; y < 13; ++y) {
        double *pl = b + (long)y * L13_S8PY;
        for (int z = 0; z < 13; ++z)
            l13_s8_pencil(pl + (long)z * L13_S8Z, L13_S8PX, 0, 0, sp,
                          C0, C1, C2, C3, C4, C5, S0, S1, S2, S3, S4, S5);
    }
#endif
}

/* per x-plane: y-pencils + map@store (step t), then z-pencils (step t+1,
 * skipped when last).  Also serves as the step-1 z prologue via yfirst=0. */
static void L13_S8OPT
l13_s8_pass_yz(double *restrict b, const double *restrict cb,
               const fft3d_plan *restrict pp, const int doy, const int doz)
{
#if L13_S8RK
    const double *restrict rk = pp->srk8;
    for (int x = 0; x < 13; ++x) {
        double *pl = b + (long)x * L13_S8PX;
        const double *cl = cb + (long)x * L13_S8PX;
        if (doy)
            for (int z = 0; z < 13; ++z)
                l13_s8_pencil_r(pl + (long)z * L13_S8Z, L13_S8PY,
                                cl + (long)z * L13_S8Z, 1, rk);
        if (doz)
            for (int y = 0; y < 13; ++y)
                l13_s8_pencil_r(pl + (long)y * L13_S8PY, L13_S8Z, 0, 0, rk);
    }
#else
    const double *cd = pp->ctd8, *sd = pp->ssd8;
    vd_w4 C0 = S8LD(cd + 0),  C1 = S8LD(cd + 8),  C2 = S8LD(cd + 16),
          C3 = S8LD(cd + 24), C4 = S8LD(cd + 32), C5 = S8LD(cd + 40);
    vd_w4 S0 = S8LD(sd + 0),  S1 = S8LD(sd + 8),  S2 = S8LD(sd + 16),
          S3 = S8LD(sd + 24), S4 = S8LD(sd + 32), S5 = S8LD(sd + 40);
    __asm__("" : "+v"(C0), "+v"(C1), "+v"(C2), "+v"(C3), "+v"(C4), "+v"(C5),
                 "+v"(S0), "+v"(S1), "+v"(S2), "+v"(S3), "+v"(S4), "+v"(S5));
    double sp[48] __attribute__((aligned(64)));
    for (int x = 0; x < 13; ++x) {
        double *pl = b + (long)x * L13_S8PX;
        const double *cl = cb + (long)x * L13_S8PX;
        if (doy)
            for (int z = 0; z < 13; ++z)
                l13_s8_pencil(pl + (long)z * L13_S8Z, L13_S8PY,
                              cl + (long)z * L13_S8Z, 1, sp,
                              C0, C1, C2, C3, C4, C5, S0, S1, S2, S3, S4, S5);
        if (doz)
            for (int y = 0; y < 13; ++y)
                l13_s8_pencil(pl + (long)y * L13_S8PY, L13_S8Z, 0, 0, sp,
                              C0, C1, C2, C3, C4, C5, S0, S1, S2, S3, S4, S5);
    }
#endif
}

/* all m steps of one 8-volume group, SoA-resident end to end.
 * -DL13_S8FZ=0 rolls back to the r7 unfused iter; -DL13_S8LZ=1/2 selects
 * the lazy-map experiment (measured a 3-12% LOSS on the node, ice_r8). */
#ifndef L13_S8LZ
#  define L13_S8LZ 0
#endif
#ifndef L13_S8FZ
#  define L13_S8FZ 1
#endif
static void
l13_s8_chain_group(const fft3d_plan *restrict p, const double *restrict x0,
                   const double *restrict cf, double *restrict out, int m)
{
    l13_s8_in(x0, p->s8);
    l13_s8_in(cf, p->s8c);
#if L13_S8LZ
    for (int it = 0; it < m; ++it)
        l13_s8_iter_lz(p->s8, p->s8c, p->ctd8, p->ssd8, it > 0);
    l13_s8_out_lz(p->s8, out);
#elif L13_S8FZ
    l13_s8_pass_yz(p->s8, p->s8c, p, 0, 1);               /* step-1 z */
    for (int it = 1; it <= m; ++it) {
        l13_s8_pass_x(p->s8, p);
        l13_s8_pass_yz(p->s8, p->s8c, p, 1, it < m);
    }
    l13_s8_out(p->s8, out);
#else
    for (int it = 0; it < m; ++it)
        l13_s8_iter(p->s8, p->s8c, p->ctd8, p->ssd8);
    l13_s8_out(p->s8, out);
#endif
}
#endif /* L13_S8 */

/* the ice_r6 chain body, unchanged: step 1 through exec, then the mz/lazy
 * cexec arms.  Now also serves the s8 path's sub-8 batch remainder. */
static void
l13_chain_classic(fft3d_plan *plan, const double _Complex *x0,
                  const double _Complex *c, double _Complex *final_out, int m)
{
    const size_t npts = (size_t)plan->batch * 2197;
    plan->exec(plan, x0, final_out);          /* z_1, raw, into the state buf */
    if (m > 1 && plan->cmz) {
        /* ice_r6 mz: the whole-state map pass moves to CHAIN ENTRY
         * (state_1 = map(z_1), amortised 1/m); the mz bodies keep the
         * state fully mapped from then on, so exit needs nothing. */
        l13_map_pass((double *)final_out, (const double *)c, npts);
        plan->cexec(plan, (double *)final_out, (const double *)c, m);
    } else {
        if (m > 1)
            plan->cexec(plan, (double *)final_out, (const double *)c, m);
        l13_map_pass((double *)final_out, (const double *)c, npts);
    }
}
#endif /* L13_CHAIN_FAST */

/* The optional chain entry point the ice_r4 driver detects (weak symbol).
 * Contract: state_0 = x0; m times { z = FFT(state); state = (z+c)/(1+|z+c|) };
 * final state into final_out.  x0 is const; final_out doubles as the single
 * in-place state buffer.  -DL13_NOCHAIN hides the symbol so the driver's
 * unfused fallback (execute + driver-side map) can be priced.
 * ICE_R7: batch >= 8 routes through the SoA-8 groups (deterministic
 * dispatch on batch, never raced); the remainder and batch < 8 keep the
 * ice_r6 classic path. */
#if !defined(L13_NOCHAIN)
void fft3d_chain(fft3d_plan *plan, const double _Complex *x0,
                 const double _Complex *c, double _Complex *final_out, int m)
{
    if (m < 1) return;
#if defined(L13_CHAIN_FAST)
#if L13_S8
    if (plan->s8 && plan->batch >= 8) {
        const int ng = plan->batch / 8, rem = plan->batch % 8;
        for (int g = 0; g < ng; ++g) {
            size_t off = (size_t)g * 8 * 2197;
            l13_s8_chain_group(plan, (const double *)(x0 + off),
                               (const double *)(c + off),
                               (double *)(final_out + off), m);
        }
        if (rem) {
            fft3d_plan tp = *plan;
            tp.batch = rem;
            size_t off = (size_t)ng * 8 * 2197;
            l13_chain_classic(&tp, x0 + off, c + off, final_out + off, m);
        }
        return;
    }
#endif
    l13_chain_classic(plan, x0, c, final_out, m);
#else
    /* generic fallback: execute + full-precision scalar map (the unfused
     * shape; only ever runs on non-AVX512 builds, which are unscored). */
    const size_t npts = (size_t)plan->batch * 2197;
    const double _Complex *cur = x0;
    for (int s = 0; s < m; ++s) {
        fft3d_execute(plan, cur, plan->ztmp);
        const double *zr = (const double *)plan->ztmp;
        const double *cr = (const double *)c;
        double *o = (double *)final_out;
        for (size_t i = 0; i < npts; ++i) {
            double re = zr[2 * i] + cr[2 * i];
            double im = zr[2 * i + 1] + cr[2 * i + 1];
            double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
            o[2 * i] = re * sc;
            o[2 * i + 1] = im * sc;
        }
        cur = final_out;
    }
#endif
}
#endif /* !L13_NOCHAIN */

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

    /* ctab8 6*7*8 + stab8 6*8 + ctd8 6*8 + ssd8 6*8 + ctab4 6*7*4 + stab4 6*4
     * + ctd4 6*4 + pb 13*PBROW + sb 344 + t1,t1b 2*13*T1P (padded planes)
     * + ice_r7: two SoA-8 group buffers (state + c) with a 320 B stagger so
     * their 4K residues differ (the s8 map streams both at the same slot
     * offsets) */
    size_t nd = 6 * 7 * 8 + 6 * 8 + 6 * 8 + 6 * 8 + 12 * 8
                + 6 * 7 * 4 + 6 * 4 + 6 * 4
                + 13 * L13_PBROW + 344 + 2 * 13 * L13_T1P + 2 * 208 + 48
                + 2 * L13_S8_GS + 40;
    p->block = malloc(nd * sizeof(double) + 64);
    if (!p->block) { free(p); return NULL; }
#if !(defined(__AVX512F__) && defined(__AVX512VL__))
    /* generic fft3d_chain fallback needs a z scratch volume-batch */
    p->ztmp = malloc((size_t)batch * 2197 * sizeof(double _Complex));
    if (!p->ztmp) { free(p->block); free(p); return NULL; }
#endif
    double *q = L13_ALIGN64((double *)p->block);
    p->ctab8 = q; q += 6 * 7 * 8;
    p->stab8 = q; q += 6 * 8;
    p->ctd8 = q;  q += 6 * 8;
    p->ssd8 = q;  q += 6 * 8;
    p->srk8 = q;  q += 12 * 8;
    p->ctab4 = q; q += 6 * 7 * 4;
    p->stab4 = q; q += 6 * 4;
    p->ctd4 = q;  q += 6 * 4;
    p->pb = q;    q += 13 * L13_PBROW;
    p->sb = q;    q += 344;
    p->t1 = q;    q += 13 * L13_T1P;
    p->t1b = q;   q += 13 * L13_T1P;
    p->sg = q;    q += 2 * 208 + 48;
    q = L13_ALIGN64(q);
    p->s8 = q;    q += L13_S8_GS + 40;
    p->s8c = q;
    /* ice_r8: prefer a single 2MB-aligned THP for the two SoA buffers.  The
     * mmap+madvise is allocation-only (identical bits either way); on
     * failure the in-block placement above stands. */
    {
        const size_t HP = 2u << 20;
        size_t need = (2 * L13_S8_GS + 40) * sizeof(double);
        size_t len = (need + HP - 1) & ~(HP - 1);
        void *raw = mmap(0, len + HP, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (raw != MAP_FAILED) {
            uintptr_t a = ((uintptr_t)raw + HP - 1) & ~(uintptr_t)(HP - 1);
            if (a > (uintptr_t)raw) munmap(raw, a - (uintptr_t)raw);
            size_t tail = ((uintptr_t)raw + len + HP) - (a + len);
            if (tail) munmap((void *)(a + len), tail);
#ifdef MADV_HUGEPAGE
            madvise((void *)a, len, MADV_HUGEPAGE);
#endif
            memset((void *)a, 0, len);   /* touch: fault the huge page(s) now,
                                          * not inside the timed chain */
            p->s8blk = (void *)a;
            p->s8len = len;
            p->s8 = (double *)a;
            p->s8c = (double *)a + L13_S8_GS + 40;
        }
    }

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
        for (int i = 0; i < 8; ++i) p->ssd8[(m - 1) * 8 + i] = s;  /* plain */
    }
    /* ice_r8 Rader-split pencil constants (L13_rader ice_r7's sork, splatted) */
    {
        static const int gt[6] = {1, 2, 4, 8, 3, 6};
        for (int t = 0; t < 3; ++t) {
            double ca = (double)cosl(PI2 * (long double)gt[t] / 13.0L);
            double cb = (double)cosl(PI2 * (long double)gt[t + 3] / 13.0L);
            for (int i = 0; i < 8; ++i) {
                p->srk8[t * 8 + i] = 0.5 * (ca + cb);
                p->srk8[(3 + t) * 8 + i] = 0.5 * (ca - cb);
            }
        }
        for (int t = 0; t < 6; ++t) {
            double s = (double)sinl(PI2 * (long double)gt[t] / 13.0L);
            for (int i = 0; i < 8; ++i) p->srk8[(6 + t) * 8 + i] = s;
        }
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
    /* ice_r2: the cache-resident default is OV — xfzs with the next
     * volume's X pass interleaved into the plane phase.  It beat zs in
     * every measurement this round (in-plan race −2.0%/−1.3%, graded
     * FORCE A/B −0.6% in a contended window), never lost, and is
     * schedule-identical to zs at nb=1, so the flip is pure upside.  The
     * race below keeps zs priced every process with hysteresis toward
     * this incumbent. */
#if defined(__AVX512F__)
    if (ws > (unsigned long long)l3c) { p->exec = l13_exec_xfzs_pf_mx; pick = "512b all-pinned zsolidY+xmm-tail X-first+pf"; }
    else                              { p->exec = l13_exec_xfzs_ov_mx; pick = "512b all-pinned zsolidY+xmm-tail X-first+ov"; }
#else
    if (ws > (unsigned long long)l2c) { p->exec = exec_xf_w2; pick = "256b X-first"; }
    else                              { p->exec = exec_xl_w2; pick = "256b X-last"; }
#endif

    /* ice_r4: fused-chain body; nb==1 is structurally serial (zs shape).
     * ice_r5: the nb>=2 default is VM2 -- volume-group-major, G=2 (see the
     * l13_chain_vm comment: L2-resident state+c for the whole chain, ov
     * overlap kept inside each pair).  Step-major ov (the r4 default) and
     * G=1/4 stay as race arms with hysteresis toward vm2.
     * -DL13_CF=0 pins the zs chain body; -DL13_CG=0/1/2/4 pins a shape. */
#if defined(L13_CHAIN_FAST)
#  if defined(L13_CF) && (L13_CF == 0)
    p->cexec = l13_chain_zs_mx;
#  elif defined(L13_CMZ) && L13_CMZ
    /* ice_r6 pin: mz at the CMZ group size (1 = serial, 2 = paired ov) */
    p->cexec = (batch < 2)     ? l13_chain_mzs_mx
             : (L13_CMZ == 1)  ? l13_chain_mz1_mx
             :                   l13_chain_mz2_mx;
    p->cmz = 1;
#  elif defined(L13_CG)
    p->cexec = (batch < 2)   ? l13_chain_zs_mx
             : (L13_CG == 0) ? l13_chain_ov_mx
             : (L13_CG == 1) ? l13_chain_vm1_mx
             : (L13_CG == 2) ? l13_chain_vm2_mx
             :                 l13_chain_vm4_mx;
#  else
    /* ice_r6: the nb>=2 default is MZ1 -- map at the Z store, volume-major
     * SERIAL (m1 beat m2 in 3/3 dev windows: 6086/6191, 6184/6307,
     * 6164/6240 ns/vol-step -- with the map off the X path and the state
     * L2-resident, the pair overlap is dead weight, closing my r5 "if v1
     * beats v2, simplify" note in the mz regime).  The lazy vm2 and the
     * pair-ov mz2 stay raced with hysteresis so the quiet window can undo
     * the flip if it disagrees.  nb==1 default is the serial mz body. */
    p->cexec = (batch >= 2) ? l13_chain_mz1_mx : l13_chain_mzs_mx;
    p->cmz = 1;
#  endif
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
    /* ice_r2 */
    case 15: p->exec = l13_exec_xfzs_s_mx; pick = "FORCED 512b xfzs pre-RA-sched"; break;
    case 16: p->exec = l13_exec_xfzs_ov_mx; pick = "FORCED 512b xfzs cross-volume X overlap"; break;
    case 17: p->exec = l13_exec_xfzs_ov_s_mx; pick = "FORCED 512b xfzs overlap+sched"; break;
    case 18: p->exec = l13_exec_xfzs_ow_mx; pick = "FORCED 512b xfzs overlap+pfw"; break;
    /* ice_r3 */
    case 19: p->exec = l13_exec_xfzs_oz_mx; pick = "FORCED 512b xfzs Z-interleaved overlap"; break;
    case 20: p->exec = l13_exec_xfzs_op_mx; pick = "FORCED 512b xfzs Z-interleaved overlap+paced pfw"; break;
    default: break;
    }
#endif
    snprintf(g_desc, sizeof g_desc,
             "conj-folded dense 13x13 per axis, lanes=lines, pinned sines; %s"
#if defined(L13_CHAIN_FAST)
#if L13_S8
             "; chain=S8[B>=8] soa8-2sweep-pinned map@Ystore(v%d) "
             "r8[fz-xstep-fused hp2M rk0 lz0] else pairmap%s"
#else
             "; fchain=pairmap(v%d)%s"     /* +mz* = map@Z-store, else lazy */
#endif
#endif
             , pick
#if defined(L13_CHAIN_FAST)
             , (int)L13_MAPV,
             (p->cexec == l13_chain_ov_mx)  ? "+xstep-ov" :
             (p->cexec == l13_chain_vm1_mx) ? "+vm1" :
             (p->cexec == l13_chain_vm2_mx) ? "+vm2" :
             (p->cexec == l13_chain_vm4_mx) ? "+vm4" :
             (p->cexec == l13_chain_mz1_mx) ? "+mz1" :
             (p->cexec == l13_chain_mz2_mx) ? "+mz2" :
             (p->cexec == l13_chain_mzs_mx) ? "+mzs" : "+zs"
#endif
             );

    /* ---- in-plan CHAIN-SHAPED ADOPTING race (ice_r2; replaces the ice_r1
     * instrument-only discriminator).  Why adoption is now allowed, and why
     * it is safe: (a) every candidate below is BIT-IDENTICAL to the default
     * (same chunk DAGs, same store order per volume — sched twins reorder
     * instructions only, ov reorders whole independent chunks across
     * volumes; cmp-verified on the node), so whatever the race picks, the
     * OUTPUT of every run is unchanged and the driver's repeatability cmp
     * stays clean by construction.  Only the timing pick varies per
     * process.  (b) This panel's records (L13_rader ice_r1 window drift,
     * L17_rader ice_r1 "tryout numbers are contention-poisoned") establish
     * that the monitor's quiet scoring window — where create() runs — is
     * the ONLY honest place to rank ICX variants; a hard-coded pick from a
     * contended dev window would be adopting noise.  Pattern from
     * L17_rader/L17_matrixsimd ice_r1 create-time tuners, with a 3%
     * hysteresis toward the incumbent so page-coloring coin flips
     * (L13_rader ice_r1: pw was a ±2% lottery) cannot flip it.
     *
     * The race is CHAIN-SHAPED (L17_matrixsimd ice_r1 stage 1g): tb=min(
     * batch,32) volumes ping-ponging two private buffers with a driver-style
     * unitary scale pass between steps, so candidates are ranked in the
     * graded regime (in AND out L3-resident, L2-evicted between visits, the
     * scale pass's traffic included) instead of ice_r1's L2-resident fixed
     * src->dst loop.  Values stay O(1) (forward DFT grows ~sqrt(2197)/step,
     * the scale divides it back), so no denormal drift.  A ~120 ms dense-FMA
     * settle spin first (L17_matrixsimd <- L17_winograd: schedutil leaves
     * short creates on an unramped core, and this create was 8 ms).
     * Slots (all X-first zsolid-Y lineage):
     *   zs : the r11 default (rollback reference)
     *   ov : cross-volume X-pass overlap (L17_rader's winning "sp" shape)
     *        — THE ice_r2 DEFAULT/incumbent (beat zs −2.0/−1.3% in-race,
     *        −0.6% graded FORCE A/B, never lost)
     *   os : ov + pre-RA scheduling (L17_matrixsimd's pragma — plain-zs
     *        sched measured +5.2% in this shape's first race, so only the
     *        ov twin is kept racing)
     *   pf : zs + full prefetch schedule (pfr next volume + pfw next
     *        plane; +10.6% in the first race — kept for the quiet window,
     *        L13_rader called pw a ±2% coin flip at this cell).
     * The ow slot (ov + pfw of the next out plane) was raced once and
     * retired same-round: +13% (4774 vs ov 4166 ns/vol) — the Z stores'
     * RFOs are already hidden by the ov overlap and the extra 559
     * prefetchw/volume only add traffic; kept as FORCE=18.
     * Adoption margin 1.5%: the pick binds only the process that measured
     * it, on the same buffers, min-of-9 — the ±2-5% cross-window coloring
     * lottery does not apply within one race.  Adoption only when
     * ws <= L3 (the chain regime the race reproduces); past L3 the pf
     * default stands and the race is print-only.  y2/p7/xl slots retired:
     * ice_r1 priced them +1%/+3%/+22% behind zs.
     * ~15 ms + settle.  -DL13_AB=0 removes race AND spin; -DL13_FORCE pins
     * the exec and demotes the race to print-only. */
#ifndef L13_AB
#  define L13_AB 1
#endif
#if L13_AB && defined(__AVX512F__)
    {
        /* clock-settle spin: 4 independent zmm FMA chains, both pipes fed */
        struct timespec s0, s1;
        clock_gettime(CLOCK_MONOTONIC, &s0);
        vd_w4 z0 = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8};
        vd_w4 z1 = z0 + 0.25, z2 = z0 + 0.5, z3 = z0 + 0.75;
        const vd_w4 zm = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
        const vd_w4 za = {0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75};
        for (;;) {
            for (int i = 0; i < 50000; ++i) {
                z0 = z0 * zm + za; z1 = z1 * zm + za;
                z2 = z2 * zm + za; z3 = z3 * zm + za;
            }
            __asm__ volatile("" : "+v"(z0), "+v"(z1), "+v"(z2), "+v"(z3));
            clock_gettime(CLOCK_MONOTONIC, &s1);
            if ((double)(s1.tv_sec - s0.tv_sec) +
                (double)(s1.tv_nsec - s0.tv_nsec) * 1e-9 > 0.12) break;
        }
    }
    {
        int tb = batch < 32 ? batch : 32;
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
            /* ice_r3: 6 arms — oz/op join, all still bit-identical.  pf
             * stays raced for telemetry (it is the ws>L3 default). */
            enum { L13_NC = 6 };
            l13_fn cf[L13_NC];
            static const char cn[L13_NC][3] =
                {"zs", "ov", "oz", "op", "os", "pf"};
            double best[L13_NC];
            for (int c = 0; c < L13_NC; ++c) best[c] = 1e300;
            cf[0] = l13_exec_xfzs_mx;
            cf[1] = l13_exec_xfzs_ov_mx;
            cf[2] = l13_exec_xfzs_oz_mx;
            cf[3] = l13_exec_xfzs_op_mx;
            cf[4] = l13_exec_xfzs_ov_s_mx;
            cf[5] = l13_exec_xfzs_pf_mx;
            fft3d_plan tp = *p;
            tp.batch = tb;
            const double usc = 1.0 / sqrt(2197.0);   /* driver's unitary */
            double _Complex *ci = ti, *co = to;
            for (int t = -2; t < 9; ++t)     /* 2 warm rounds + 9 timed */
                for (int c = 0; c < L13_NC; ++c) {
                    struct timespec t0, t1;
                    clock_gettime(CLOCK_MONOTONIC, &t0);
                    cf[c](&tp, ci, co);
                    clock_gettime(CLOCK_MONOTONIC, &t1);
                    double *q = (double *)co;   /* driver-style scale pass */
                    for (size_t i = 0; i < (size_t)tb * 2197 * 2; ++i)
                        q[i] *= usc;
                    double _Complex *sw = ci; ci = co; co = sw;
                    double ns = (double)(t1.tv_sec - t0.tv_sec) * 1e9 +
                                (double)(t1.tv_nsec - t0.tv_nsec);
                    ns /= (double)tb;
                    if (t >= 0 && ns < best[c]) best[c] = ns;
                }
            int inc = -1;
            for (int c = 0; c < L13_NC; ++c)
                if (p->exec == cf[c]) inc = c;
            int w = inc >= 0 ? inc : 0;
            for (int c = 0; c < L13_NC; ++c)
                if (best[c] < best[w]) w = c;
            int adopted = -1;
#if !defined(L13_FORCE)
            if (inc >= 0 && w != inc && ws <= (unsigned long long)l3c &&
                best[w] < 0.985 * best[inc]) {
                p->exec = cf[w];
                adopted = w;
            }
#endif
            size_t n = strlen(g_desc);
            snprintf(g_desc + n, sizeof g_desc - n,
                     "; chain-ab[B%d]=zs%.0f,ov%.0f,oz%.0f,op%.0f,os%.0f,"
                     "pf%.0f ns/vol pick=%s%s",
                     tb, best[0], best[1], best[2], best[3], best[4], best[5],
                     adopted >= 0 ? cn[adopted] : (inc >= 0 ? cn[inc] : "?"),
                     adopted >= 0 ? "" : "(inc)");
            fprintf(stderr, "L13_direct chain-ab[B%d]: zs=%.0f ov=%.0f "
                            "oz=%.0f op=%.0f os=%.0f pf=%.0f ns/vol "
                            "pick=%s%s\n",
                    tb, best[0], best[1], best[2], best[3], best[4], best[5],
                    adopted >= 0 ? cn[adopted] : (inc >= 0 ? cn[inc] : "?"),
                    adopted >= 0 ? "" : "(inc)");

            /* ---- ice_r4: MAP-CHAIN race for the fused-chain body (cexec),
             * the shape the scored regime actually runs.  Two candidates,
             * BIT-IDENTICAL by construction (cz = per-step zs, cv =
             * cross-step ov pipeline; same chunk and map DAGs per point,
             * whole-unit reordering only, cmp-verified above through the
             * chain repeatability check).  ti/to are reused: ti as state
             * (any values work; the map bounds them), to as a stand-in c
             * field.  6 steps per timed rep, min over 9, adopt at the same
             * 1.5% hysteresis toward the incumbent (ov at nb>=2).  nb==1
             * stays pinned to cz: the cross-step overlap would read the
             * volume the plane phase is writing. */
#if defined(L13_CHAIN_FAST) && !defined(L13_CF) && !defined(L13_CG) && \
    !defined(L13_CMZ)
            /* ice_r6: seven arms.  cz/cv = step-major zs/ov (r4); v1/v2/v4
             * = lazy volume-group-major at G=1/2/4 (r5); m1/m2 = MAP-AT-
             * Z-STORE volume-group-major at G=1/2 (this round).  All
             * bit-identical per point (see chunk13pz/map2st), so adoption
             * stays output-invariant.  The mz bodies are raced WITHOUT
             * their 1/m entry/exit format passes (those live in
             * fft3d_chain), so the race prices exactly the steady-state
             * step -- the honest comparison at m=1278.  The CG/CMZ pins
             * now silence the race entirely (hard pins for A/Bs; before
             * r6 a CG pin could still be overridden by adoption). */
            if (tb >= 2 && ws <= (unsigned long long)l3c) {
                typedef void (*l13_cfn)(const fft3d_plan *, double *,
                                        const double *, int);
                enum { L13_NCC = 7 };
                l13_cfn ccand[L13_NCC] = { l13_chain_zs_mx, l13_chain_ov_mx,
                                           l13_chain_vm1_mx, l13_chain_vm2_mx,
                                           l13_chain_vm4_mx, l13_chain_mz1_mx,
                                           l13_chain_mz2_mx };
                static const char ccn[L13_NCC][3] =
                    {"cz", "cv", "v1", "v2", "v4", "m1", "m2"};
                static const int ccmz[L13_NCC] = {0, 0, 0, 0, 0, 1, 1};
                double cbest[L13_NCC];
                for (int c = 0; c < L13_NCC; ++c) cbest[c] = 1e300;
                enum { L13_CMS = 6 };
                for (int t = -1; t < 9; ++t)
                    for (int c = 0; c < L13_NCC; ++c) {
                        struct timespec t0, t1;
                        clock_gettime(CLOCK_MONOTONIC, &t0);
                        ccand[c](&tp, (double *)ti, (const double *)to,
                                 L13_CMS);
                        clock_gettime(CLOCK_MONOTONIC, &t1);
                        double ns = (double)(t1.tv_sec - t0.tv_sec) * 1e9 +
                                    (double)(t1.tv_nsec - t0.tv_nsec);
                        ns /= (double)tb * (L13_CMS - 1);
                        if (t >= 0 && ns < cbest[c]) cbest[c] = ns;
                    }
                int cinc = 0;
                for (int c = 0; c < L13_NCC; ++c)
                    if (p->cexec == ccand[c]) cinc = c;
                int cb = cinc;
                for (int c = 0; c < L13_NCC; ++c)
                    if (cbest[c] < cbest[cb]) cb = c;
                int cw = cinc;
                if (cb != cinc && cbest[cb] < 0.985 * cbest[cinc]) cw = cb;
                p->cexec = ccand[cw];
                p->cmz = ccmz[cw];
                size_t n2 = strlen(g_desc);
                snprintf(g_desc + n2, sizeof g_desc - n2,
                         "; fch-ab=cz%.0f,cv%.0f,v1%.0f,v2%.0f,v4%.0f,"
                         "m1%.0f,m2%.0f pick=%s%s",
                         cbest[0], cbest[1], cbest[2], cbest[3], cbest[4],
                         cbest[5], cbest[6],
                         ccn[cw], cw == cinc ? "(inc)" : "");
                fprintf(stderr, "L13_direct fch-ab[B%d]: cz=%.0f cv=%.0f "
                                "v1=%.0f v2=%.0f v4=%.0f m1=%.0f m2=%.0f "
                                "ns/vol-step pick=%s%s\n",
                        tb, cbest[0], cbest[1], cbest[2], cbest[3], cbest[4],
                        cbest[5], cbest[6],
                        ccn[cw], cw == cinc ? "(inc)" : "");
            }
#endif
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
    if (p->s8blk) munmap(p->s8blk, p->s8len);
    free(p->ztmp);
    free(p->block);
    free(p);
}

#endif /* L13_TEMPLATE_PASS */
