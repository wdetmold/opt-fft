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
 *
 * ICE ROUND ice_r4: THE GRADED STEP IS NOW THE FULL RIVAL MAP,
 *     state <- (z + c) / (1 + |z + c|),  z = FFT(state)   (raw, no unitary),
 * and this entry exports the optional fft3d_chain to own the whole m-step
 * chain instead of paying the driver's unfused map pass.  Technique adopted
 * from the rival pipelines (corpus 10 s2 and 1760b1bf/1000f989's sources):
 *   1. LAZY MAP fused into the X pass: each element of the previous step's
 *      raw z is read exactly once there, so the map happens ON LOAD --
 *      w = z + c, then w / (1 + |w|) -- and `state` is never materialized.
 *      Steps ping-pong raw-z buffers (internal zb <-> final_out, parity-
 *      chosen so step m lands in final_out); one in-place map finishes.
 *   2. ONE DIVIDER OP PER POINT: |w| by exact vsqrtpd (the divider is a
 *      separate unit; its latency hides under the kernel's FMA work), the
 *      reciprocal of 1 + |w| by vrcp14pd seed + 2 Newton steps on the FMA
 *      pipes (2^-14 -> 3.7e-9 -> 1.4e-17: below double rounding, so the map
 *      is exact to ~2-3 ulp).  The rivals' float-seed 2-Newton variant
 *      (~1e-12/step) is ILLEGAL at this point's m=1278 budget of 1e-13/step;
 *      this ladder is their pw_full-equivalent at pw_full_fast's port cost.
 *      -DL13R_MAPRCP=0 swaps the ladder for one exact vdivpd; -DL13R_PW3=1
 *      adds a third (paranoia) Newton step.  The xmm tail (1/43 of points)
 *      uses exact sqrt + div.
 *   3. CHAIN-ARM RACE replacing the obsolete unitary-scale race (the graded
 *      unit is now the map chain): arms fo (fused X + ov overlap, incumbent)
 *      / fz (fused, no ov) / uf (in-place map pass + plain exec -- prices
 *      fusion itself on this window's silicon).  All arms are output-bit-
 *      identical (same pw and chunk arithmetic per element; only pass
 *      interleaving differs).  -DL13R_CFORCE=0/1/2 pins fo/fz/uf.
 * fft3d_execute and its exemplar pipeline are unchanged (the single-
 * transform gate uses them); its exec pick is now deterministic (ov, or
 * zs+pf past L3) -- the old race regime no longer exists in the grading.
 *
 * ICE ROUND ice_r5: THE CHAIN MACHINERY IS REBUILT ON L13_direct's ice_r4
 * DESIGN (attribution: L13_direct ice_r4, which scored 5.837 vs this entry's
 * 6.363 with the same map semantics; their record predicted this adoption).
 * Four pieces, all theirs, carrying my 93-FP/chunk Rader kernel:
 *   1. PAIRED LAZY MAP (their l13_mappair <- 1760b1bf pw_pair_gen): two
 *      ADJACENT zmm (8 points) merge |w|^2 into ONE vector via 2x
 *      vpermutex2var, so ONE vsqrtpd + ONE rcp14+2-Newton ladder serve 8
 *      DISTINCT magnitudes; 2x vpermutexvar expands the scale back.  Halves
 *      both the divider pressure and the ladder FMA of my r4 per-row map.
 *      Map flavor = their measured MAPV sweep winner v2 (hw sqrt + rcp14
 *      ladder: 6.14 vs v0 6.25 / v1 6.39 / v3 6.92 on their pipeline) --
 *      adopted with their sweep, NOT re-swept here.  ONE per-point DAG at
 *      every width and call site keeps every chain arm bit-identical.
 *   2. TWO-PHASE PAIR UNITS + DEPTH-1 SOFTWARE PIPELINE (their fix for the
 *      inline-map spill disaster, and the answer to my r4 fs failure: a
 *      bounce split only pays when unit u+1's whole map phase issues
 *      between unit u's stage stores and its FFT chunks): 21 pair units per
 *      volume map into 2 rotating 13x16-double L1 stages; the FFT phase is
 *      the UNMODIFIED spill-free chunk13r_w4 reading the stage at rs=16
 *      (64 B stores -> exact 64 B loads).  169 = 21*8 + 1: the w1 tail
 *      column stays inline-mapped (chunk13rm_w1).
 *   3. CROSS-STEP OV (their own extension): fft3d_chain owns all m steps,
 *      so the overlap X actions at the last volume of step s come from step
 *      s+1's volume 0 -- ONE X prologue for the whole chain, no per-step
 *      drain.  23 dispatchable actions per volume (map(0); map(k)+fft(k-1)
 *      x20; fft(20); w1 tail), paced floor((x+1)*23/13).
 *   4. IN PLACE, ONE BUFFER: each volume's X pass fully consumes it before
 *      the plane phase overwrites it, so the fused chain runs entirely in
 *      the driver's final_out (working set 2.15 MiB state+c, not my r4
 *      ping-pong's 3.2).  zb survives only for the uf arm.
 * The r4 monolithic mapped chunks' variants (fo/fz/fs/f2 bodies, MAPSTYLE
 * 0-3, Montgomery MROW4) are DELETED -- all priced in the r4 record,
 * superseded by the structure above.
 *
 * ICE ROUND ice_r6: the chain goes VOLUME-GROUP-MAJOR (attribution:
 * L13_direct ice_r5's vm2, itself crediting L17_matrixsimd ice_r4's
 * volmajor-inplace; their r5 scored 5.401 vs this entry's 5.808 on that one
 * structural change).  The per-volume chains are INDEPENDENT, so instead of
 * step-major (`for s { for b }` -- every one of the m steps re-streams
 * state + c, ~2.15 MiB, through L3), the chain partitions the batch into
 * groups of G volumes and runs ALL m steps over one group before moving on:
 * working set per G=2 group = state 68.6K + c 68.6K + t1/t1b 71.6K + pb/sg
 * ~ 215 KB, L2-RESIDENT (node L2 = 1.25 MB).  From step 2 on, every X-pass
 * load and Z-store RFO that used to be an L3 round trip is an L2 hit.  G=2
 * (not 1) keeps the cross-step ov pipeline alive INSIDE the group: the last
 * volume of step s overlaps step s+1's volume 0.  Implementation is a thin
 * wrapper over the UNCHANGED r5 chain bodies (per group: exec step 1, ov
 * body for g>=2 / zs for the odd tail, final in-place map) -- grouping only
 * reorders whole volume-steps across independent volumes, so outputs stay
 * bit-identical for every G, which is what makes G legal to race.  Race
 * arms now cv/cz (step-major, the r5 pair) + v1/v2/v4 (volume-major at
 * G=1/2/4, incumbent v2); uf is DROPPED from the race (lost 27-38% in
 * every r5 window; still reachable).  -DL13R_CFORCE=0..5 pins
 * cv/cz/uf/v1/v2/v4.
 *
 * ICE ROUND ice_r7: the chain goes SoA BATCH-LANE (the structural item every
 * fast rival shares -- attribution: 1000f989/v5_3907583b/v6_f40c5e25 sources
 * in fft_v4_solutions/ and fft_v5v6_solutions/, all ~3.9-4.0 us/xform on this
 * node vs this entry's r6 5.38).  fft3d_chain owns all m steps, so the
 * INTERNAL layout is free: per group of 8 volumes, x0 and c are transposed
 * ONCE (24-shuffle 8x8 networks, amortized 1/m) into split re/im SoA with
 * the batch in the lanes -- re[p][v], im[p][v], p = x*169 + y*13 + z.  Every
 * FFT axis pass is then PURE VERTICAL SIMD: no SWAPRI, no 128-bit tile
 * transposes, no t1/pb junction, no split loads -- the whole shuffle/junction
 * tax of the lanes=lines pipeline (measured ~2x above port floor since
 * ice_r1) is deleted rather than optimized.  Each step is TWO sweeps:
 *   sweep A: per x-plane (169 contiguous points, 21.6 KB x2, L1-RESIDENT):
 *            13 z-pencils (es=8) + 13 y-pencils (es=104);
 *   sweep B: 169 x-pencils (es=1352), plain.
 * The MAP runs as a per-x-plane streaming pass at the head of sweep A
 * (-DL13R_MF=2, the measured winner: the mapped plane is L1-hot for its
 * FFTs and the ladders pipeline perfectly across 169 independent vectors;
 * MF=0 whole-volume pass 3.80, MF=1 fused-into-sweep-B-stores 3.88 --
 * ROB-bound).  The map in SoA needs NO pair merge/expand permutes:
 * |w|^2 = fma(wr,wr, wi*wi) per point-vector, and with -DL13R_SQR=1
 * (default) |w| itself is an rsqrt14+2-Newton+Heron ladder, so the map has
 * ZERO divider ops -- the divider was the binding unit (hw sqrt 4.18, hw
 * div 5.10 vs 3.69); still full-double (5.6e-9 -> 4.7e-17, below rounding),
 * 1e-13-budget-legal at m=1278 (measured 1.145e-13, ~1100x margin).  Two
 * pencil kernels, -DL13R_SK selects: SK=0 the v6 generator's Hartley-split
 * reg6 (h=6: 206 vector FP / 8 pencils, 72 broadcast constants; node 5.46);
 * SK=1 (default) this entry's own RADER-SPLIT form (the r2 CRT kernel
 * re-derived for split re/im: 186 vector FP / 8 pencils, i-mults become
 * free register renames, only 12 constants; node 4.18 same machinery --
 * beats the rival pencil by 23%).  -DL13R_SOA=0 kills the path.  Volume
 * tails (batch%8) and every batch<8 (incl the scored B=1) run the UNCHANGED
 * r6 classic machinery; SoA groups are a deterministic dispatch, not a
 * raced arm (different arithmetic = different bits; racing would break
 * cross-process repeatability).
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

/* ---- ice_r5: the graded map  w / (1 + |w|),  w = z + c ----
 * ONE per-point DAG at every width and call site (paired, unpaired zmm, xmm
 * tail, standalone pass) so every chain arm stays output-bit-identical:
 * |w| by exact vsqrtpd (divider unit, correctly rounded -- sp=0 gives a=1,
 * no clamp needed), reciprocal of 1+|w| by vrcp14pd seed + 2 Newtons on the
 * FMA pipes (2^-14 -> 3.7e-9 -> 1.4e-17, below double rounding; ~2-3 ulp
 * per application vs the 1e-13/step budget, measured whole-chain margin
 * ~1000x in r4 with the equivalent-tier ladder).  This is L13_direct
 * ice_r4's MAPV sweep winner (v2) -- adopted with their sweep, not re-run;
 * it replaces my r4 MAPSTYLE=1 (rsqrt14 ladder + one vdivpd per ROW).
 * -DL13R_MAPRCP=0 -> r = 1/a by one exact vdivpd (kept as w*r, NOT w/a, so
 * the paired form stays bit-identical); -DL13R_PW3 -> third Newton. */
#if WC == 4 && defined(__AVX512F__)
static inline __attribute__((always_inline)) VT
SUF(pw13)(VT z, VT c)
{
    VT w = z + c;
    VT t = w * w;
    VT sp = t + SWAPRI(t);                    /* |w|^2 in both lanes */
    VT a = (VT)_mm512_sqrt_pd((__m512d)sp) + 1.0;
#if L13R_MAPRCP
    VT r = (VT)_mm512_rcp14_pd((__m512d)a);
    r = r + r * (1.0 - a * r);
    r = r + r * (1.0 - a * r);
#if L13R_PW3
    r = r + r * (1.0 - a * r);
#endif
#else
    VT r = 1.0 / a;
#endif
    return w * r;
}
#define L13R_HAVE_CHUNKM 1
#elif WC == 1
static inline __attribute__((always_inline)) VT
SUF(pw13)(VT z, VT c)
{
    VT w = z + c;
    VT t = w * w;
    VT sp = t + SWAPRI(t);
    VT a = (VT)_mm_sqrt_pd((__m128d)sp) + 1.0;
#if L13R_MAPRCP && defined(__AVX512VL__)
    VT r = (VT)_mm_rcp14_pd((__m128d)a);
    r = r + r * (1.0 - a * r);
    r = r + r * (1.0 - a * r);
#if L13R_PW3
    r = r + r * (1.0 - a * r);
#endif
#else
    VT r = 1.0 / a;   /* no AVX512VL: unscored hardware, fused chain off */
#endif
    return w * r;
}
#define L13R_HAVE_CHUNKM 1
#endif

#ifdef L13R_HAVE_CHUNKM

/* chunk13r with the map fused into every load (the lazy-map X pass reads
 * the previous step's RAW z; each element is read exactly once there).
 * csrc = the c field at the same offsets/stride as src.  ice_r5: only the
 * WC==1 instantiation is still called (the fused X pass's tail column);
 * the zmm rows go through the staged pair units instead. */
static inline __attribute__((always_inline)) void
SUF(chunk13rm)(const double *restrict src, const double *restrict csrc,
               long rs, double *restrict dst, long da, long db,
               VT C0, VT C1, VT C2, VT C3, VT C4, VT C5,
               VT S0, VT S1, VT S2, VT S3, VT S4, VT S5, int tr)
{
    VT dc, cc0, cc1, cc2, cc3, cc4, cc5, T0, T1, T2, T3, T4, T5;
    {
        VT w0, w1, w2, w3, w4, w5;
        VT a0, a1, a2, b0, b1, b2;
        {
            VT x0 = SUF(pw13)(VLD(src), VLD(csrc));
            dc = x0;
            /* fold block t=0 / t=3: rows (1,12), (8,5) */
            {
                VT e0 = SUF(pw13)(VLD(src + 1 * rs), VLD(csrc + 1 * rs));
                VT f0 = SUF(pw13)(VLD(src + 12 * rs), VLD(csrc + 12 * rs));
                VT e1 = SUF(pw13)(VLD(src + 8 * rs), VLD(csrc + 8 * rs));
                VT f1 = SUF(pw13)(VLD(src + 5 * rs), VLD(csrc + 5 * rs));
                w0 = SWAPRI(e0 - f0);  w3 = SWAPRI(e1 - f1);
                VT u0 = e0 + f0, u1 = e1 + f1;
                VT pp = u0 + u1, qq = u0 - u1;
                dc += pp;
                a0 = x0 + pp * C0;  a1 = x0 + pp * C1;  a2 = x0 + pp * C2;
                b0 = qq * C3;  b1 = qq * C4;  b2 = qq * C5;
            }
            /* t=1 / t=4: rows (2,11), (3,10) */
            {
                VT e0 = SUF(pw13)(VLD(src + 2 * rs), VLD(csrc + 2 * rs));
                VT f0 = SUF(pw13)(VLD(src + 11 * rs), VLD(csrc + 11 * rs));
                VT e1 = SUF(pw13)(VLD(src + 3 * rs), VLD(csrc + 3 * rs));
                VT f1 = SUF(pw13)(VLD(src + 10 * rs), VLD(csrc + 10 * rs));
                w1 = SWAPRI(e0 - f0);  w4 = SWAPRI(e1 - f1);
                VT u0 = e0 + f0, u1 = e1 + f1;
                VT pp = u0 + u1, qq = u0 - u1;
                dc += pp;
                a0 += pp * C1;  a1 += pp * C2;  a2 += pp * C0;
                b0 += qq * C4;  b1 += qq * C5;  b2 -= qq * C3;
            }
            /* t=2 / t=5: rows (4,9), (6,7) */
            {
                VT e0 = SUF(pw13)(VLD(src + 4 * rs), VLD(csrc + 4 * rs));
                VT f0 = SUF(pw13)(VLD(src + 9 * rs), VLD(csrc + 9 * rs));
                VT e1 = SUF(pw13)(VLD(src + 6 * rs), VLD(csrc + 6 * rs));
                VT f1 = SUF(pw13)(VLD(src + 7 * rs), VLD(csrc + 7 * rs));
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
#undef L13R_HAVE_CHUNKM
#endif /* map-fused chunk */

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
#include <immintrin.h>

/* long-double trig so the splatted tables are good to ~1e-19 */
#include <math.h>

/* map knobs (used inside the template -- must precede the self-includes):
 * MAPRCP=1 rcp14+2 Newton reciprocal (default), 0 = one exact vdivpd
 * (as r=1/a then w*r, preserving pair/unpair bit-identity);
 * PW3=1 adds a third Newton step (paranoia margin; default off -- two
 * steps from a 2^-14 seed already land below double rounding).
 * The r4 MAPSTYLE knob is gone: the DAG is fixed to L13_direct ice_r4's
 * swept winner (exact vsqrtpd + rcp14 ladder), see the header. */
#ifndef L13R_MAPRCP
#  define L13R_MAPRCP 1
#endif
#ifndef L13R_PW3
#  define L13R_PW3 0
#endif

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
    /* ice_r5 fused chain (machinery adopted from L13_direct ice_r4): cbody
     * runs steps 2..m of the graded map chain IN PLACE on the state buffer
     * (raw z between steps, map fused into each step's X pass via the
     * staged pair units).  cuf=1 selects the unfused shape instead
     * (in-place map pass + exec ping-pong via zb). */
    void (*cbody)(const struct fft3d_plan *, double *, const double *, int);
    int cuf;
    /* ice_r6: volume-group size for the vm chain (L13_direct ice_r5's vm2).
     * 0 = step-major via cbody; G>0 = all m steps per G-volume group
     * (ov body inside a group of >=2, zs for the odd tail). */
    int cg;
    double _Complex *zb; /* uf arm's ping-pong mate (fused bodies: unused) */
    double *cpd8; /* CP0-2,CM0-2 splatted 8x (broadcast constants)      */
    double *snt8; /* s_t = sin(2pi g^t/13), lane-alternating (s,-s), 8x */
    double *cpd4; /* the same, splatted 4x (ymm; xmm reads first 16 B)  */
    double *snt4;
    double *pb;   /* 13 x PBROW plane buffer (pass Y -> pass Z)         */
    double *t1;   /* one 13^3 volume of scratch, x-planes at T1P        */
    double *t1b;  /* second t1: OV ping-pong (L13_direct ice_r2)        */
    double *sg;   /* 2 rotating 13x16-double map stages (chain X pairs) */
    /* ice_r7 SoA batch-lane chain (groups of 8 volumes; see header) */
    double *sre, *sim; /* state, split re/im, SoA-8: [point][lane]      */
    double *cre, *cim; /* c field converted once per group              */
    double *soah;      /* Hartley reg6 table: 72 = (j-1)*12+(k-1)*2+... */
    double *sork;      /* Rader split: CP0-2, CM0-2, s0-5 (12 scalars)  */
    void *soab;        /* SoA buffer block; NULL = path disabled        */
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

/* =============================================================================
 * ice_r5 -- the fused map chain, L13_direct ice_r4's machinery (see header).
 * =============================================================================
 */

/* standalone map pass: dst[i] = w/(1+|w|), w = z[i]+c[i].  In-place safe
 * (dst == z), used for the final chain step and the uf arm.  Same per-point
 * DAG as the paired/fused loads, so every chain arm is bit-identical. */
#if defined(__AVX512F__)
static void l13r_map_pass(double *dst, const double *z,
                          const double *restrict cf, size_t ncplx)
{
    size_t n4 = ncplx / 4;
    for (size_t i = 0; i < n4; ++i) {
        vd_w4 zv = *(const vd_w4 *)(z + 8 * i);
        vd_w4 cv = *(const vd_w4 *)(cf + 8 * i);
        *(vd_w4 *)(dst + 8 * i) = pw13_w4(zv, cv);
    }
    for (size_t j = 4 * n4; j < ncplx; ++j) {
        vd_w1 zv = *(const vd_w1 *)(z + 2 * j);
        vd_w1 cv = *(const vd_w1 *)(cf + 2 * j);
        *(vd_w1 *)(dst + 2 * j) = pw13_w1(zv, cv);
    }
}
#else
static void l13r_map_pass(double *dst, const double *z,
                          const double *restrict cf, size_t ncplx)
{
    for (size_t j = 0; j < ncplx; ++j) {
        double re = z[2 * j] + cf[2 * j];
        double im = z[2 * j + 1] + cf[2 * j + 1];
        double sc = 1.0 / (1.0 + __builtin_sqrt(re * re + im * im));
        dst[2 * j] = re * sc;
        dst[2 * j + 1] = im * sc;
    }
}
#endif

#if defined(__AVX512F__) && defined(__AVX512VL__)
#define L13R_CHAIN_FAST 1

/* PAIRED lazy map (adopted verbatim: L13_direct ice_r4's l13_mappair <-
 * 1760b1bf pw_pair_gen): two ADJACENT zmm of interleaved complex (8
 * points), ONE sqrt + ONE rcp14/Newton chain on 8 DISTINCT magnitudes --
 * half the map work and half the divider pressure of the duplicated-lane
 * pw13 form.  Elementwise the per-point DAG is identical to pw13_w4 (the
 * merge/expand permutes only reroute lanes; IEEE a+b == b+a makes the
 * lane order immaterial), so paired and unpaired points mix
 * bit-identically. */
static inline __attribute__((always_inline)) void
l13r_mappair(vd_w4 za, vd_w4 zq, vd_w4 ca, vd_w4 cq, vd_w4 *va, vd_w4 *vq)
{
    static const long long IE_[8] = {0, 2, 4, 6, 8, 10, 12, 14};
    static const long long IO_[8] = {1, 3, 5, 7, 9, 11, 13, 15};
    static const long long IA_[8] = {0, 0, 1, 1, 2, 2, 3, 3};
    static const long long IB_[8] = {4, 4, 5, 5, 6, 6, 7, 7};
    vd_w4 ua = za + ca, ub = zq + cq;
    vd_w4 pa = ua * ua, pq = ub * ub;
    vd_w4 E = (vd_w4)_mm512_permutex2var_pd((__m512d)pa,
                  _mm512_loadu_si512((const void *)IE_), (__m512d)pq);
    vd_w4 O = (vd_w4)_mm512_permutex2var_pd((__m512d)pa,
                  _mm512_loadu_si512((const void *)IO_), (__m512d)pq);
    vd_w4 m2 = E + O;
    vd_w4 a = (vd_w4)_mm512_sqrt_pd((__m512d)m2) + 1.0;
#if L13R_MAPRCP
    vd_w4 s = (vd_w4)_mm512_rcp14_pd((__m512d)a);
    s = s + s * (1.0 - a * s);
    s = s + s * (1.0 - a * s);
#if L13R_PW3
    s = s + s * (1.0 - a * s);
#endif
#else
    vd_w4 s = 1.0 / a;
#endif
    *va = ua * (vd_w4)_mm512_permutexvar_pd(
                  _mm512_loadu_si512((const void *)IA_), (__m512d)s);
    *vq = ub * (vd_w4)_mm512_permutexvar_pd(
                  _mm512_loadu_si512((const void *)IB_), (__m512d)s);
}

/* One X-pass PAIR unit: columns [8u, 8u+8) of one volume (adopted:
 * L13_direct ice_r4's two-phase pair unit).  Phase 1 maps the 13 row-pairs
 * (adjacent zmm) into a 1.7 KB L1 stage; phase 2 runs the proven spill-free
 * chunk13r_w4 on the staged state at rs=16 (64 B stores -> exact 64 B
 * loads: clean store-forward).  Map registers and FFT registers never
 * coexist, and the sqrt/rcp results' consumers are stores, not the
 * accumulate DAG -- the two properties my r4 fs bounce split lacked the
 * second of (it also lacked the depth-1 pipeline below, which is where
 * the win actually lives). */
#define L13R_CHX_MAP(SRCB, CB, U, SG)                                          \
    do {                                                                       \
        long fq_ = 8L * (U);                                                   \
        const double *sp_ = (SRCB) + 2 * fq_;                                  \
        const double *cp_ = (CB) + 2 * fq_;                                    \
        double *sg_ = (SG);                                                    \
        for (int x_ = 0; x_ < 13; ++x_) {                                      \
            vd_w4 va_, vb_;                                                    \
            l13r_mappair(*(const vd_w4 *)(sp_ + 338L * x_),                    \
                         *(const vd_w4 *)(sp_ + 338L * x_ + 8),                \
                         *(const vd_w4 *)(cp_ + 338L * x_),                    \
                         *(const vd_w4 *)(cp_ + 338L * x_ + 8), &va_, &vb_);   \
            *(vd_w4 *)(sg_ + 16 * x_) = va_;                                   \
            *(vd_w4 *)(sg_ + 16 * x_ + 8) = vb_;                               \
        }                                                                      \
    } while (0)

#define L13R_CHX_FFT(DSTB, U, SG)                                              \
    do {                                                                       \
        long fq_ = 8L * (U);                                                   \
        double *sg_ = (SG);                                                    \
        chunk13r_w4(sg_, 16, (DSTB) + 2 * fq_, 2, L13R_T1P,                    \
                    C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5, 0);        \
        chunk13r_w4(sg_ + 8, 16, (DSTB) + 2 * (fq_ + 4), 2, L13R_T1P,          \
                    C0, C1, C2, C3, C4, C5, K0, K1, K2, K3, K4, K5, 0);        \
    } while (0)

/* mapped X pass of one volume: raw z at SRCB, c at CB, X-transformed mapped
 * state into DSTB (t1 layout).  169 = 21*8 + 1: 21 pair units + the w1 tail
 * (inline-mapped chunk13rm_w1: xmm pressure is trivial), every point mapped
 * exactly once.  DEPTH-1 SOFTWARE PIPELINE across the 2 rotating stages
 * (L13_direct priced the unpipelined form at +0.15 us/xform): unit u+1's
 * whole map phase (~270 independent uops incl. the sqrts) issues between
 * unit u's stage stores and its FFT chunks. */
#define L13R_XPASS_MAP(SRCB, CB, DSTB, SG0)                                    \
    do {                                                                       \
        L13R_CHX_MAP(SRCB, CB, 0, (SG0));                                      \
        int nu_ = 20;                                                          \
        __asm__("" : "+r"(nu_));                                               \
        for (int u_ = 0; u_ < nu_; ++u_) {                                     \
            L13R_CHX_MAP(SRCB, CB, u_ + 1, (SG0) + ((u_ + 1) & 1) * 208);      \
            L13R_CHX_FFT(DSTB, u_, (SG0) + (u_ & 1) * 208);                    \
        }                                                                      \
        L13R_CHX_FFT(DSTB, 20, (SG0));                                         \
        chunk13rm_w1((SRCB) + 2 * 168, (CB) + 2 * 168, 338,                    \
                     (DSTB) + 2 * 168, 2, L13R_T1P,                            \
                     E0, E1, E2, E3, E4, E5, F0, F1, F2, F3, F4, F5, 0);       \
    } while (0)

/* one interleaved mapped X ACTION of the next volume (ov placement): the
 * same depth-1 pipeline unrolled into 23 dispatchable actions per volume:
 *   0: map(0);  1..20: map(k)+fft(k-1);  21: fft(20);  22: w1 tail */
#define L13R_CHOVX_STEP(VNX, VCX, TN, SG0)                                     \
    do {                                                                       \
        if (xi == 0) {                                                         \
            L13R_CHX_MAP(VNX, VCX, 0, (SG0));                                  \
        } else if (xi <= 20) {                                                 \
            L13R_CHX_MAP(VNX, VCX, xi, (SG0) + (xi & 1) * 208);                \
            L13R_CHX_FFT(TN, xi - 1, (SG0) + ((xi - 1) & 1) * 208);            \
        } else if (xi == 21) {                                                 \
            L13R_CHX_FFT(TN, 20, (SG0));                                       \
        } else {                                                               \
            chunk13rm_w1((VNX) + 2 * 168, (VCX) + 2 * 168, 338,                \
                         (TN) + 2 * 168, 2, L13R_T1P,                          \
                         E0, E1, E2, E3, E4, E5, F0, F1, F2, F3, F4, F5, 0);   \
        }                                                                      \
        ++xi;                                                                  \
    } while (0)

/* cv -- steps 2..m, CROSS-STEP ov pipeline (adopted: L13_direct ice_r4).
 * st holds raw z_1 on entry, raw z_m on exit; the chain runs IN PLACE on st
 * (each volume's X pass fully consumes it before its plane phase overwrites
 * it).  The overlap X actions at the last volume of step s read step s+1's
 * volume 0, whose raw z_s[0] was finished by plane phase (s,0) earlier in
 * the same step -- ONE X prologue for the whole chain, no per-step drain.
 * Hazard check: X of (s,b+1) reads z_{s-1}[b+1], overwritten only by plane
 * phase (s,b+1) later; X of (s+1,0) reads z_s[0], already written.
 * Requires nb >= 2 (at nb==1 the only next volume IS the one being
 * written -- use cz). */
static void
l13r_chain_ov(const fft3d_plan *restrict p, double *st, const double *cf,
              int m)
{
    L13R_MIXP_DECLS;
    double *restrict t1b = p->t1b;
    double *restrict sg = p->sg;
    if (m < 2) return;
    L13R_XPASS_MAP(st, cf, t1, sg);                 /* X of (step 2, vol 0) */
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
            else if (s < m) { vnx = st; vcx = cf; }  /* next STEP's vol 0 */
            else            { vnx = 0;  vcx = 0; }
            int xi = 0;
            for (int x = 0; x < 13; ++x) {
                const double *pin = tc + (long)x * L13R_T1P;
                double *pt = vout + (long)x * 338;
                L13R_GROUP_ZS(pin, 26, pb, L13R_PBROW);
                if (vnx) {
                    int xe = ((x + 1) * 23) / 13;
                    while (xi < xe) L13R_CHOVX_STEP(vnx, vcx, tn, sg);
                }
                L13R_GROUP_MX(pb, L13R_PBROW, pt, 26);
            }
        }
    }
}

/* cz -- steps 2..m, per-step zs shape: monolithic paired X pass per volume,
 * still in place on st.  The nb==1 body and the cz race arm. */
static void
l13r_chain_zs(const fft3d_plan *restrict p, double *st, const double *cf,
              int m)
{
    L13R_MIXP_DECLS;
    double *restrict sg = p->sg;
    for (int s = 2; s <= m; ++s) {
        for (int b = 0; b < nb; ++b) {
            double *v = st + (size_t)b * 4394;
            const double *cv = cf + (size_t)b * 4394;
            L13R_XPASS_MAP(v, cv, t1, sg);
            for (int x = 0; x < 13; ++x) {
                const double *pin = t1 + (long)x * L13R_T1P;
                double *pt = v + (long)x * 338;
                L13R_GROUP_ZS(pin, 26, pb, L13R_PBROW);
                L13R_GROUP_MX(pb, L13R_PBROW, pt, 26);
            }
        }
    }
}

/* the ice_r6 volume-group-major classic chain over nvol volumes at the given
 * pointers -- now also the tail path under the SoA dispatch (ice_r7). */
static void l13r_chain_vm(const fft3d_plan *p0, int G, int nvol, int m,
                          const double _Complex *x0, const double *cf,
                          double _Complex *final_out)
{
    fft3d_plan tp = *p0;
    for (int b0 = 0; b0 < nvol; b0 += G) {
        int g = nvol - b0; if (g > G) g = G;
        tp.batch = g;
        const double _Complex *xg = x0 + (size_t)b0 * 2197;
        double _Complex *fg = final_out + (size_t)b0 * 2197;
        const double *cg_ = cf + (size_t)b0 * 4394;
        tp.exec(&tp, xg, fg);              /* z_1 of the group, raw */
        if (m > 1) {
            if (g >= 2) l13r_chain_ov(&tp, (double *)fg, cg_, m);
            else        l13r_chain_zs(&tp, (double *)fg, cg_, m);
        }
        l13r_map_pass((double *)fg, (const double *)fg, cg_,
                      (size_t)g * 2197);
    }
}

/* =============================================================================
 * ice_r7 -- SoA BATCH-LANE chain, groups of 8 volumes (see file header).
 * =============================================================================
 */
#ifndef L13R_SOA
#  define L13R_SOA 1
#endif
#ifndef L13R_SK
#  define L13R_SK 1     /* SoA pencil kernel: 0 = Hartley reg6, 1 = Rader split */
#endif
#ifndef L13R_MF
#  define L13R_MF 2     /* map placement (node A/B under SQR=1): 2 = per-x-plane
                         * streaming map at the head of sweep A, L1-hot (3.696);
                         * 0 = whole-volume pass (3.799); 1 = fused into sweep-B
                         * stores (3.878 -- ROB-bound, and the winner ONLY under
                         * the old divider-bound SQR=0 map: 4.183 vs 4.449) */
#endif
#ifndef L13R_SQR
#  define L13R_SQR 1    /* 1 = all-FMA rsqrt14 |w| (ZERO divider ops in the map:
                         * node A/B 3.879 vs 4.183 hw-sqrt vs 5.100 hw-div --
                         * the divider serializes sweep B), 0 = vsqrtpd */
#endif

#if L13R_SOA

/* The graded map on one SoA point-vector (8 volumes' worth of one point):
 * w = z + c; out = w / (1 + |w|).  Same precision tier as pw13 (hw sqrt +
 * rcp14 + 2 Newtons, full double, ~1000x margin vs the 1e-13/step budget at
 * m=1278); NO pair merge/expand permutes -- SoA lanes are already distinct
 * magnitudes.  sp=0 gives a=1: rcp14(1)=1 exactly, no clamp needed. */
static inline __attribute__((always_inline)) void
l13r_soamap(__m512d zr, __m512d zi, __m512d cr, __m512d ci,
            __m512d *outr, __m512d *outi)
{
    const __m512d one = _mm512_set1_pd(1.0);
    __m512d wr = _mm512_add_pd(zr, cr), wi = _mm512_add_pd(zi, ci);
    __m512d sp = _mm512_fmadd_pd(wr, wr, _mm512_mul_pd(wi, wi));
#if L13R_SQR
    /* all-FMA |w|: rsqrt14 + 2 Newtons (2^-14 -> 5.6e-9 -> 4.7e-17, below
     * double rounding) + Heron mul; ZERO divider ops in the whole map.
     * sp=0: rsqrt14(0)=inf, but inf*0=nan -- clamp with max(sp, tiny). */
    __m512d spc = _mm512_max_pd(sp, _mm512_set1_pd(1e-300));
    __m512d rq = _mm512_rsqrt14_pd(spc);
    const __m512d half = _mm512_set1_pd(0.5), thr = _mm512_set1_pd(1.5);
    __m512d hs = _mm512_mul_pd(spc, half);
    rq = _mm512_mul_pd(rq, _mm512_fnmadd_pd(hs, _mm512_mul_pd(rq, rq), thr));
    rq = _mm512_mul_pd(rq, _mm512_fnmadd_pd(hs, _mm512_mul_pd(rq, rq), thr));
    __m512d a = _mm512_add_pd(_mm512_mul_pd(spc, rq), one);
#else
    __m512d a = _mm512_add_pd(_mm512_sqrt_pd(sp), one);
#endif
#if L13R_MAPRCP
    __m512d s = _mm512_rcp14_pd(a);
    s = _mm512_fmadd_pd(s, _mm512_fnmadd_pd(a, s, one), s);
    s = _mm512_fmadd_pd(s, _mm512_fnmadd_pd(a, s, one), s);
#if L13R_PW3
    s = _mm512_fmadd_pd(s, _mm512_fnmadd_pd(a, s, one), s);
#endif
#else
    __m512d s = _mm512_div_pd(one, a);
#endif
    *outr = _mm512_mul_pd(wr, s);
    *outi = _mm512_mul_pd(wi, s);
}

/* plain / mapped SoA stores for one output row M of a pencil */
#define L13R_SST(M, RV, IV)                                                    \
    do {                                                                       \
        _mm512_store_pd(re + (M) * es, (RV));                                  \
        _mm512_store_pd(im + (M) * es, (IV));                                  \
    } while (0)
#define L13R_SSTM(M, RV, IV)                                                   \
    do {                                                                       \
        __m512d mr_, mi_;                                                      \
        l13r_soamap((RV), (IV), _mm512_load_pd(cre + (M) * es),                \
                    _mm512_load_pd(cim + (M) * es), &mr_, &mi_);               \
        _mm512_store_pd(re + (M) * es, mr_);                                   \
        _mm512_store_pd(im + (M) * es, mi_);                                   \
    } while (0)

/* ---- SK=0: Hartley-split reg6 pencil (v6_f40c5e25 gen_hartley, H13=reg6,
 * transcribed; forward DFT).  s_j = x_j + x_{13-j}, d_j = x_j - x_{13-j};
 * A_k = x0 + sum_j cos(2pi kj/13) s_j,  B_k = sum_j sin(2pi kj/13) d_j;
 * X_k = (Ar + Bi_acc, Ai - Br_acc), X_{13-k} the conjugate-mirrored combine.
 * 206 vector FP / 8 pencils, 70% FMA, 72 embedded-broadcast constants. ---- */
#define L13R_HK1(K)                                                            \
    do {                                                                       \
        __m512d c_ = _mm512_set1_pd(ht[(K - 1) * 2]);                          \
        __m512d s_ = _mm512_set1_pd(ht[(K - 1) * 2 + 1]);                      \
        Ar##K = _mm512_fmadd_pd(c_, sr, x0r);                                  \
        Ai##K = _mm512_fmadd_pd(c_, si, x0i);                                  \
        Br##K = _mm512_mul_pd(s_, di);                                         \
        Bi##K = _mm512_mul_pd(s_, dr);                                         \
    } while (0)
#define L13R_HKA(J, K)                                                         \
    do {                                                                       \
        __m512d c_ = _mm512_set1_pd(ht[((J) - 1) * 12 + (K - 1) * 2]);         \
        __m512d s_ = _mm512_set1_pd(ht[((J) - 1) * 12 + (K - 1) * 2 + 1]);     \
        Ar##K = _mm512_fmadd_pd(c_, sr, Ar##K);                                \
        Ai##K = _mm512_fmadd_pd(c_, si, Ai##K);                                \
        Br##K = _mm512_fmadd_pd(s_, di, Br##K);                                \
        Bi##K = _mm512_fmadd_pd(s_, dr, Bi##K);                                \
    } while (0)
#define L13R_HJ(J)                                                             \
    do {                                                                       \
        __m512d ar = _mm512_load_pd(re + (J) * es);                            \
        __m512d ai = _mm512_load_pd(im + (J) * es);                            \
        __m512d br = _mm512_load_pd(re + (13 - (J)) * es);                     \
        __m512d bi = _mm512_load_pd(im + (13 - (J)) * es);                     \
        __m512d sr = _mm512_add_pd(ar, br), si = _mm512_add_pd(ai, bi);        \
        __m512d dr = _mm512_sub_pd(ar, br), di = _mm512_sub_pd(ai, bi);        \
        o0r = _mm512_add_pd(o0r, sr); o0i = _mm512_add_pd(o0i, si);            \
        if ((J) == 1) {                                                        \
            L13R_HK1(1); L13R_HK1(2); L13R_HK1(3);                             \
            L13R_HK1(4); L13R_HK1(5); L13R_HK1(6);                             \
        } else {                                                               \
            L13R_HKA(J, 1); L13R_HKA(J, 2); L13R_HKA(J, 3);                    \
            L13R_HKA(J, 4); L13R_HKA(J, 5); L13R_HKA(J, 6);                    \
        }                                                                      \
    } while (0)
#define L13R_H_BODY(ST)                                                        \
    __m512d x0r = _mm512_load_pd(re), x0i = _mm512_load_pd(im);                \
    __m512d o0r = x0r, o0i = x0i;                                              \
    __m512d Ar1, Ai1, Br1, Bi1, Ar2, Ai2, Br2, Bi2, Ar3, Ai3, Br3, Bi3;       \
    __m512d Ar4, Ai4, Br4, Bi4, Ar5, Ai5, Br5, Bi5, Ar6, Ai6, Br6, Bi6;       \
    L13R_HJ(1); L13R_HJ(2); L13R_HJ(3); L13R_HJ(4); L13R_HJ(5); L13R_HJ(6);   \
    ST(0, o0r, o0i);                                                           \
    L13R_HOUT(1, ST); L13R_HOUT(2, ST); L13R_HOUT(3, ST);                      \
    L13R_HOUT(4, ST); L13R_HOUT(5, ST); L13R_HOUT(6, ST);
#define L13R_HOUT(K, ST)                                                       \
    do {                                                                       \
        __m512d yr = _mm512_add_pd(Ar##K, Br##K);                              \
        __m512d yi = _mm512_sub_pd(Ai##K, Bi##K);                              \
        __m512d zr = _mm512_sub_pd(Ar##K, Br##K);                              \
        __m512d zi = _mm512_add_pd(Ai##K, Bi##K);                              \
        ST(K, yr, yi);                                                         \
        ST(13 - K, zr, zi);                                                    \
    } while (0)

static inline __attribute__((always_inline)) void
l13r_dft13s_h(double *restrict re, double *restrict im, long es,
              const double *restrict ht)
{
    L13R_H_BODY(L13R_SST)
}
static inline __attribute__((always_inline)) void
l13r_dft13s_hm(double *restrict re, double *restrict im, long es,
               const double *restrict cre, const double *restrict cim,
               const double *restrict ht)
{
    L13R_H_BODY(L13R_SSTM)
}

/* ---- SK=1: this entry's Rader/CRT kernel re-derived for split re/im (the
 * chunk13r math verbatim; SWAPRI becomes a register rename, the sine sign
 * fold becomes plain constants).  186 vector FP / 8 pencils, 12 constants
 * (rk[0..2]=CP, rk[3..5]=CM, rk[6..11]=sin(2pi g^t/13), g^t = 1,2,4,8,3,6).
 * Sine side: TR_n = sum_t sig s_{(n+t)%6} vi_t, TI_n = same on vr_t,
 * sig = -1 when n+t >= 6; X[g^n] = (ccr_n + TR_n, cci_n - TI_n),
 * X[13-g^n] = (ccr_n - TR_n, cci_n + TI_n). ---- */
#define L13R_R_BODY(ST)                                                        \
    __m512d x0r = _mm512_load_pd(re), x0i = _mm512_load_pd(im);                \
    const __m512d C0 = _mm512_set1_pd(rk[0]), C1 = _mm512_set1_pd(rk[1]);      \
    const __m512d C2 = _mm512_set1_pd(rk[2]), C3 = _mm512_set1_pd(rk[3]);      \
    const __m512d C4 = _mm512_set1_pd(rk[4]), C5 = _mm512_set1_pd(rk[5]);      \
    __m512d vr0, vr1, vr2, vr3, vr4, vr5, vi0, vi1, vi2, vi3, vi4, vi5;        \
    __m512d a0r, a1r, a2r, b0r, b1r, b2r, a0i, a1i, a2i, b0i, b1i, b2i;        \
    __m512d dcr = x0r, dci = x0i;                                              \
    { /* t=0/3: rows (1,12), (8,5) */                                          \
        __m512d e0r = _mm512_load_pd(re + 1 * es), f0r = _mm512_load_pd(re + 12 * es); \
        __m512d e0i = _mm512_load_pd(im + 1 * es), f0i = _mm512_load_pd(im + 12 * es); \
        __m512d e1r = _mm512_load_pd(re + 8 * es), f1r = _mm512_load_pd(re + 5 * es);  \
        __m512d e1i = _mm512_load_pd(im + 8 * es), f1i = _mm512_load_pd(im + 5 * es);  \
        vr0 = _mm512_sub_pd(e0r, f0r); vi0 = _mm512_sub_pd(e0i, f0i);          \
        vr3 = _mm512_sub_pd(e1r, f1r); vi3 = _mm512_sub_pd(e1i, f1i);          \
        __m512d u0r = _mm512_add_pd(e0r, f0r), u1r = _mm512_add_pd(e1r, f1r);  \
        __m512d u0i = _mm512_add_pd(e0i, f0i), u1i = _mm512_add_pd(e1i, f1i);  \
        __m512d ppr = _mm512_add_pd(u0r, u1r), qqr = _mm512_sub_pd(u0r, u1r);  \
        __m512d ppi = _mm512_add_pd(u0i, u1i), qqi = _mm512_sub_pd(u0i, u1i);  \
        dcr = _mm512_add_pd(dcr, ppr); dci = _mm512_add_pd(dci, ppi);          \
        a0r = _mm512_fmadd_pd(ppr, C0, x0r); a1r = _mm512_fmadd_pd(ppr, C1, x0r); \
        a2r = _mm512_fmadd_pd(ppr, C2, x0r);                                   \
        a0i = _mm512_fmadd_pd(ppi, C0, x0i); a1i = _mm512_fmadd_pd(ppi, C1, x0i); \
        a2i = _mm512_fmadd_pd(ppi, C2, x0i);                                   \
        b0r = _mm512_mul_pd(qqr, C3); b1r = _mm512_mul_pd(qqr, C4);            \
        b2r = _mm512_mul_pd(qqr, C5);                                          \
        b0i = _mm512_mul_pd(qqi, C3); b1i = _mm512_mul_pd(qqi, C4);            \
        b2i = _mm512_mul_pd(qqi, C5);                                          \
    }                                                                          \
    { /* t=1/4: rows (2,11), (3,10) */                                         \
        __m512d e0r = _mm512_load_pd(re + 2 * es), f0r = _mm512_load_pd(re + 11 * es); \
        __m512d e0i = _mm512_load_pd(im + 2 * es), f0i = _mm512_load_pd(im + 11 * es); \
        __m512d e1r = _mm512_load_pd(re + 3 * es), f1r = _mm512_load_pd(re + 10 * es); \
        __m512d e1i = _mm512_load_pd(im + 3 * es), f1i = _mm512_load_pd(im + 10 * es); \
        vr1 = _mm512_sub_pd(e0r, f0r); vi1 = _mm512_sub_pd(e0i, f0i);          \
        vr4 = _mm512_sub_pd(e1r, f1r); vi4 = _mm512_sub_pd(e1i, f1i);          \
        __m512d u0r = _mm512_add_pd(e0r, f0r), u1r = _mm512_add_pd(e1r, f1r);  \
        __m512d u0i = _mm512_add_pd(e0i, f0i), u1i = _mm512_add_pd(e1i, f1i);  \
        __m512d ppr = _mm512_add_pd(u0r, u1r), qqr = _mm512_sub_pd(u0r, u1r);  \
        __m512d ppi = _mm512_add_pd(u0i, u1i), qqi = _mm512_sub_pd(u0i, u1i);  \
        dcr = _mm512_add_pd(dcr, ppr); dci = _mm512_add_pd(dci, ppi);          \
        a0r = _mm512_fmadd_pd(ppr, C1, a0r); a1r = _mm512_fmadd_pd(ppr, C2, a1r); \
        a2r = _mm512_fmadd_pd(ppr, C0, a2r);                                   \
        a0i = _mm512_fmadd_pd(ppi, C1, a0i); a1i = _mm512_fmadd_pd(ppi, C2, a1i); \
        a2i = _mm512_fmadd_pd(ppi, C0, a2i);                                   \
        b0r = _mm512_fmadd_pd(qqr, C4, b0r); b1r = _mm512_fmadd_pd(qqr, C5, b1r); \
        b2r = _mm512_fnmadd_pd(qqr, C3, b2r);                                  \
        b0i = _mm512_fmadd_pd(qqi, C4, b0i); b1i = _mm512_fmadd_pd(qqi, C5, b1i); \
        b2i = _mm512_fnmadd_pd(qqi, C3, b2i);                                  \
    }                                                                          \
    { /* t=2/5: rows (4,9), (6,7) */                                           \
        __m512d e0r = _mm512_load_pd(re + 4 * es), f0r = _mm512_load_pd(re + 9 * es); \
        __m512d e0i = _mm512_load_pd(im + 4 * es), f0i = _mm512_load_pd(im + 9 * es); \
        __m512d e1r = _mm512_load_pd(re + 6 * es), f1r = _mm512_load_pd(re + 7 * es); \
        __m512d e1i = _mm512_load_pd(im + 6 * es), f1i = _mm512_load_pd(im + 7 * es); \
        vr2 = _mm512_sub_pd(e0r, f0r); vi2 = _mm512_sub_pd(e0i, f0i);          \
        vr5 = _mm512_sub_pd(e1r, f1r); vi5 = _mm512_sub_pd(e1i, f1i);          \
        __m512d u0r = _mm512_add_pd(e0r, f0r), u1r = _mm512_add_pd(e1r, f1r);  \
        __m512d u0i = _mm512_add_pd(e0i, f0i), u1i = _mm512_add_pd(e1i, f1i);  \
        __m512d ppr = _mm512_add_pd(u0r, u1r), qqr = _mm512_sub_pd(u0r, u1r);  \
        __m512d ppi = _mm512_add_pd(u0i, u1i), qqi = _mm512_sub_pd(u0i, u1i);  \
        dcr = _mm512_add_pd(dcr, ppr); dci = _mm512_add_pd(dci, ppi);          \
        a0r = _mm512_fmadd_pd(ppr, C2, a0r); a1r = _mm512_fmadd_pd(ppr, C0, a1r); \
        a2r = _mm512_fmadd_pd(ppr, C1, a2r);                                   \
        a0i = _mm512_fmadd_pd(ppi, C2, a0i); a1i = _mm512_fmadd_pd(ppi, C0, a1i); \
        a2i = _mm512_fmadd_pd(ppi, C1, a2i);                                   \
        b0r = _mm512_fmadd_pd(qqr, C5, b0r); b1r = _mm512_fnmadd_pd(qqr, C3, b1r); \
        b2r = _mm512_fnmadd_pd(qqr, C4, b2r);                                  \
        b0i = _mm512_fmadd_pd(qqi, C5, b0i); b1i = _mm512_fnmadd_pd(qqi, C3, b1i); \
        b2i = _mm512_fnmadd_pd(qqi, C4, b2i);                                  \
    }                                                                          \
    __m512d cc0r = _mm512_add_pd(a0r, b0r), cc3r = _mm512_sub_pd(a0r, b0r);    \
    __m512d cc1r = _mm512_add_pd(a1r, b1r), cc4r = _mm512_sub_pd(a1r, b1r);    \
    __m512d cc2r = _mm512_add_pd(a2r, b2r), cc5r = _mm512_sub_pd(a2r, b2r);    \
    __m512d cc0i = _mm512_add_pd(a0i, b0i), cc3i = _mm512_sub_pd(a0i, b0i);    \
    __m512d cc1i = _mm512_add_pd(a1i, b1i), cc4i = _mm512_sub_pd(a1i, b1i);    \
    __m512d cc2i = _mm512_add_pd(a2i, b2i), cc5i = _mm512_sub_pd(a2i, b2i);    \
    ST(0, dcr, dci);                                                           \
    {                                                                          \
        const __m512d S0 = _mm512_set1_pd(rk[6]),  S1 = _mm512_set1_pd(rk[7]); \
        const __m512d S2 = _mm512_set1_pd(rk[8]),  S3 = _mm512_set1_pd(rk[9]); \
        const __m512d S4 = _mm512_set1_pd(rk[10]), S5 = _mm512_set1_pd(rk[11]);\
        __m512d TR0 = _mm512_mul_pd(vi0, S0), TI0 = _mm512_mul_pd(vr0, S0);    \
        __m512d TR1 = _mm512_mul_pd(vi0, S1), TI1 = _mm512_mul_pd(vr0, S1);    \
        __m512d TR2 = _mm512_mul_pd(vi0, S2), TI2 = _mm512_mul_pd(vr0, S2);    \
        __m512d TR3 = _mm512_mul_pd(vi0, S3), TI3 = _mm512_mul_pd(vr0, S3);    \
        __m512d TR4 = _mm512_mul_pd(vi0, S4), TI4 = _mm512_mul_pd(vr0, S4);    \
        __m512d TR5 = _mm512_mul_pd(vi0, S5), TI5 = _mm512_mul_pd(vr0, S5);    \
        TR0 = _mm512_fmadd_pd(vi1, S1, TR0);  TI0 = _mm512_fmadd_pd(vr1, S1, TI0); \
        TR1 = _mm512_fmadd_pd(vi1, S2, TR1);  TI1 = _mm512_fmadd_pd(vr1, S2, TI1); \
        TR2 = _mm512_fmadd_pd(vi1, S3, TR2);  TI2 = _mm512_fmadd_pd(vr1, S3, TI2); \
        TR3 = _mm512_fmadd_pd(vi1, S4, TR3);  TI3 = _mm512_fmadd_pd(vr1, S4, TI3); \
        TR4 = _mm512_fmadd_pd(vi1, S5, TR4);  TI4 = _mm512_fmadd_pd(vr1, S5, TI4); \
        TR5 = _mm512_fnmadd_pd(vi1, S0, TR5); TI5 = _mm512_fnmadd_pd(vr1, S0, TI5); \
        TR0 = _mm512_fmadd_pd(vi2, S2, TR0);  TI0 = _mm512_fmadd_pd(vr2, S2, TI0); \
        TR1 = _mm512_fmadd_pd(vi2, S3, TR1);  TI1 = _mm512_fmadd_pd(vr2, S3, TI1); \
        TR2 = _mm512_fmadd_pd(vi2, S4, TR2);  TI2 = _mm512_fmadd_pd(vr2, S4, TI2); \
        TR3 = _mm512_fmadd_pd(vi2, S5, TR3);  TI3 = _mm512_fmadd_pd(vr2, S5, TI3); \
        TR4 = _mm512_fnmadd_pd(vi2, S0, TR4); TI4 = _mm512_fnmadd_pd(vr2, S0, TI4); \
        TR5 = _mm512_fnmadd_pd(vi2, S1, TR5); TI5 = _mm512_fnmadd_pd(vr2, S1, TI5); \
        TR0 = _mm512_fmadd_pd(vi3, S3, TR0);  TI0 = _mm512_fmadd_pd(vr3, S3, TI0); \
        TR1 = _mm512_fmadd_pd(vi3, S4, TR1);  TI1 = _mm512_fmadd_pd(vr3, S4, TI1); \
        TR2 = _mm512_fmadd_pd(vi3, S5, TR2);  TI2 = _mm512_fmadd_pd(vr3, S5, TI2); \
        TR3 = _mm512_fnmadd_pd(vi3, S0, TR3); TI3 = _mm512_fnmadd_pd(vr3, S0, TI3); \
        TR4 = _mm512_fnmadd_pd(vi3, S1, TR4); TI4 = _mm512_fnmadd_pd(vr3, S1, TI4); \
        TR5 = _mm512_fnmadd_pd(vi3, S2, TR5); TI5 = _mm512_fnmadd_pd(vr3, S2, TI5); \
        TR0 = _mm512_fmadd_pd(vi4, S4, TR0);  TI0 = _mm512_fmadd_pd(vr4, S4, TI0); \
        TR1 = _mm512_fmadd_pd(vi4, S5, TR1);  TI1 = _mm512_fmadd_pd(vr4, S5, TI1); \
        TR2 = _mm512_fnmadd_pd(vi4, S0, TR2); TI2 = _mm512_fnmadd_pd(vr4, S0, TI2); \
        TR3 = _mm512_fnmadd_pd(vi4, S1, TR3); TI3 = _mm512_fnmadd_pd(vr4, S1, TI3); \
        TR4 = _mm512_fnmadd_pd(vi4, S2, TR4); TI4 = _mm512_fnmadd_pd(vr4, S2, TI4); \
        TR5 = _mm512_fnmadd_pd(vi4, S3, TR5); TI5 = _mm512_fnmadd_pd(vr4, S3, TI5); \
        TR0 = _mm512_fmadd_pd(vi5, S5, TR0);  TI0 = _mm512_fmadd_pd(vr5, S5, TI0); \
        TR1 = _mm512_fnmadd_pd(vi5, S0, TR1); TI1 = _mm512_fnmadd_pd(vr5, S0, TI1); \
        TR2 = _mm512_fnmadd_pd(vi5, S1, TR2); TI2 = _mm512_fnmadd_pd(vr5, S1, TI2); \
        TR3 = _mm512_fnmadd_pd(vi5, S2, TR3); TI3 = _mm512_fnmadd_pd(vr5, S2, TI3); \
        TR4 = _mm512_fnmadd_pd(vi5, S3, TR4); TI4 = _mm512_fnmadd_pd(vr5, S3, TI4); \
        TR5 = _mm512_fnmadd_pd(vi5, S4, TR5); TI5 = _mm512_fnmadd_pd(vr5, S4, TI5); \
        ST(1,  _mm512_add_pd(cc0r, TR0), _mm512_sub_pd(cc0i, TI0));            \
        ST(12, _mm512_sub_pd(cc0r, TR0), _mm512_add_pd(cc0i, TI0));            \
        ST(2,  _mm512_add_pd(cc1r, TR1), _mm512_sub_pd(cc1i, TI1));            \
        ST(11, _mm512_sub_pd(cc1r, TR1), _mm512_add_pd(cc1i, TI1));            \
        ST(3,  _mm512_add_pd(cc4r, TR4), _mm512_sub_pd(cc4i, TI4));            \
        ST(10, _mm512_sub_pd(cc4r, TR4), _mm512_add_pd(cc4i, TI4));            \
        ST(4,  _mm512_add_pd(cc2r, TR2), _mm512_sub_pd(cc2i, TI2));            \
        ST(9,  _mm512_sub_pd(cc2r, TR2), _mm512_add_pd(cc2i, TI2));            \
        ST(8,  _mm512_add_pd(cc3r, TR3), _mm512_sub_pd(cc3i, TI3));            \
        ST(5,  _mm512_sub_pd(cc3r, TR3), _mm512_add_pd(cc3i, TI3));            \
        ST(6,  _mm512_add_pd(cc5r, TR5), _mm512_sub_pd(cc5i, TI5));            \
        ST(7,  _mm512_sub_pd(cc5r, TR5), _mm512_add_pd(cc5i, TI5));            \
    }

static inline __attribute__((always_inline)) void
l13r_dft13s_r(double *restrict re, double *restrict im, long es,
              const double *restrict rk)
{
    L13R_R_BODY(L13R_SST)
}
static inline __attribute__((always_inline)) void
l13r_dft13s_rm(double *restrict re, double *restrict im, long es,
               const double *restrict cre, const double *restrict cim,
               const double *restrict rk)
{
    L13R_R_BODY(L13R_SSTM)
}

#if L13R_SK
#  define L13R_DFT13S(RE, IM, ES, KT)            l13r_dft13s_r(RE, IM, ES, KT)
#  define L13R_DFT13SM(RE, IM, ES, CR, CI, KT)   l13r_dft13s_rm(RE, IM, ES, CR, CI, KT)
#  define L13R_SOKT(P) ((P)->sork)
#else
#  define L13R_DFT13S(RE, IM, ES, KT)            l13r_dft13s_h(RE, IM, ES, KT)
#  define L13R_DFT13SM(RE, IM, ES, CR, CI, KT)   l13r_dft13s_hm(RE, IM, ES, CR, CI, KT)
#  define L13R_SOKT(P) ((P)->soah)
#endif

/* 8x8 double transpose (self-inverse index map): row v elem j <-> row j
 * elem v.  24 shuffles: 8 unpck + 16 shuff64x2. */
static inline __attribute__((always_inline)) void
l13r_tr8x8(__m512d r[8])
{
    __m512d t0 = _mm512_unpacklo_pd(r[0], r[1]);
    __m512d t1 = _mm512_unpackhi_pd(r[0], r[1]);
    __m512d t2 = _mm512_unpacklo_pd(r[2], r[3]);
    __m512d t3 = _mm512_unpackhi_pd(r[2], r[3]);
    __m512d t4 = _mm512_unpacklo_pd(r[4], r[5]);
    __m512d t5 = _mm512_unpackhi_pd(r[4], r[5]);
    __m512d t6 = _mm512_unpacklo_pd(r[6], r[7]);
    __m512d t7 = _mm512_unpackhi_pd(r[6], r[7]);
    __m512d u0 = _mm512_shuffle_f64x2(t0, t2, 0x88);
    __m512d u1 = _mm512_shuffle_f64x2(t1, t3, 0x88);
    __m512d u2 = _mm512_shuffle_f64x2(t0, t2, 0xdd);
    __m512d u3 = _mm512_shuffle_f64x2(t1, t3, 0xdd);
    __m512d u4 = _mm512_shuffle_f64x2(t4, t6, 0x88);
    __m512d u5 = _mm512_shuffle_f64x2(t5, t7, 0x88);
    __m512d u6 = _mm512_shuffle_f64x2(t4, t6, 0xdd);
    __m512d u7 = _mm512_shuffle_f64x2(t5, t7, 0xdd);
    r[0] = _mm512_shuffle_f64x2(u0, u4, 0x88);
    r[4] = _mm512_shuffle_f64x2(u0, u4, 0xdd);
    r[1] = _mm512_shuffle_f64x2(u1, u5, 0x88);
    r[5] = _mm512_shuffle_f64x2(u1, u5, 0xdd);
    r[2] = _mm512_shuffle_f64x2(u2, u6, 0x88);
    r[6] = _mm512_shuffle_f64x2(u2, u6, 0xdd);
    r[3] = _mm512_shuffle_f64x2(u3, u7, 0x88);
    r[7] = _mm512_shuffle_f64x2(u3, u7, 0xdd);
}

/* 8 interleaved volumes (vol stride 4394 doubles) -> SoA-8 split re/im.
 * 4 points per iteration; the transpose of 8 vol-rows of 4 complex lands
 * exactly on (re,im) x 4 points across the 8 lanes.  Runs ONCE per group
 * per chain (1/m amortized). */
static void l13r_conv_in(const double *restrict x, double *restrict qr,
                         double *restrict qi)
{
    for (long q = 0; q < 549; ++q) {
        __m512d r[8];
        for (int v = 0; v < 8; ++v)
            r[v] = _mm512_loadu_pd(x + (size_t)v * 4394 + (size_t)q * 8);
        l13r_tr8x8(r);
        _mm512_store_pd(qr + q * 32,      r[0]);
        _mm512_store_pd(qi + q * 32,      r[1]);
        _mm512_store_pd(qr + q * 32 + 8,  r[2]);
        _mm512_store_pd(qi + q * 32 + 8,  r[3]);
        _mm512_store_pd(qr + q * 32 + 16, r[4]);
        _mm512_store_pd(qi + q * 32 + 16, r[5]);
        _mm512_store_pd(qr + q * 32 + 24, r[6]);
        _mm512_store_pd(qi + q * 32 + 24, r[7]);
    }
    for (int v = 0; v < 8; ++v) {          /* point 2196 = 549*4 tail */
        qr[2196 * 8 + v] = x[(size_t)v * 4394 + 4392];
        qi[2196 * 8 + v] = x[(size_t)v * 4394 + 4393];
    }
}

static void l13r_conv_out(const double *restrict qr, const double *restrict qi,
                          double *restrict x)
{
    for (long q = 0; q < 549; ++q) {
        __m512d r[8];
        r[0] = _mm512_load_pd(qr + q * 32);
        r[1] = _mm512_load_pd(qi + q * 32);
        r[2] = _mm512_load_pd(qr + q * 32 + 8);
        r[3] = _mm512_load_pd(qi + q * 32 + 8);
        r[4] = _mm512_load_pd(qr + q * 32 + 16);
        r[5] = _mm512_load_pd(qi + q * 32 + 16);
        r[6] = _mm512_load_pd(qr + q * 32 + 24);
        r[7] = _mm512_load_pd(qi + q * 32 + 24);
        l13r_tr8x8(r);
        for (int v = 0; v < 8; ++v)
            _mm512_storeu_pd(x + (size_t)v * 4394 + (size_t)q * 8, r[v]);
    }
    for (int v = 0; v < 8; ++v) {
        x[(size_t)v * 4394 + 4392] = qr[2196 * 8 + v];
        x[(size_t)v * 4394 + 4393] = qi[2196 * 8 + v];
    }
}

#if L13R_MF != 1
/* MF=0/2: the map as a streaming pass (whole group volume at MF=0; one
 * x-plane at MF=2, issued right before that plane's sweep-A pencils) */
static void l13r_soa_map_pass(double *restrict sre, double *restrict sim,
                              const double *restrict cre,
                              const double *restrict cim, long nq)
{
    for (long q = 0; q < nq; ++q) {
        __m512d orr, oii;
        l13r_soamap(_mm512_load_pd(sre + q * 8), _mm512_load_pd(sim + q * 8),
                    _mm512_load_pd(cre + q * 8), _mm512_load_pd(cim + q * 8),
                    &orr, &oii);
        _mm512_store_pd(sre + q * 8, orr);
        _mm512_store_pd(sim + q * 8, oii);
    }
}
#endif

/* One chain step on the SoA group.  Sweep A per x-plane (contiguous 169
 * points, L1-resident through both pencil passes): z-pencils then
 * y-pencils.  Sweep B: x-pencils; domap fuses the graded map into the
 * output stores (the buffer then holds STATE, not raw z). */
static void l13r_soa_step(double *restrict sre, double *restrict sim,
                          const double *restrict cre,
                          const double *restrict cim,
                          const double *restrict kt, int domap)
{
    for (int x = 0; x < 13; ++x) {
        double *br = sre + (long)x * 1352;
        double *bi = sim + (long)x * 1352;
#if L13R_MF == 2
        if (domap >= 0)   /* map this x-plane L1-hot, right before its FFTs */
            l13r_soa_map_pass(br, bi, cre + (long)x * 1352,
                              cim + (long)x * 1352, 169);
#endif
        for (int y = 0; y < 13; ++y)
            L13R_DFT13S(br + y * 104, bi + y * 104, 8, kt);
        for (int z = 0; z < 13; ++z)
            L13R_DFT13S(br + z * 8, bi + z * 8, 104, kt);
    }
    if (L13R_MF == 1 && domap > 0) {
        for (int y = 0; y < 13; ++y)
            for (int z = 0; z < 13; ++z) {
                long o = (y * 13 + z) * 8;
#if L13R_SBPF
                if (z < 12) {   /* next pencil's state+c lines, L2 -> L1 */
                    for (int j = 0; j < 13; ++j) {
                        _mm_prefetch((const char *)(sre + o + 8 + j * 1352), _MM_HINT_T0);
                        _mm_prefetch((const char *)(sim + o + 8 + j * 1352), _MM_HINT_T0);
                        _mm_prefetch((const char *)(cre + o + 8 + j * 1352), _MM_HINT_T0);
                        _mm_prefetch((const char *)(cim + o + 8 + j * 1352), _MM_HINT_T0);
                    }
                }
#endif
                L13R_DFT13SM(sre + o, sim + o, 1352, cre + o, cim + o, kt);
            }
    } else {
        for (int y = 0; y < 13; ++y)
            for (int z = 0; z < 13; ++z) {
                long o = (y * 13 + z) * 8;
                L13R_DFT13S(sre + o, sim + o, 1352, kt);
            }
    }
}

/* the whole m-step chain for ONE group of 8 volumes, entirely inside the
 * L2-resident SoA buffers (state 281 KB + c 281 KB) */
static void l13r_chain_soa(const fft3d_plan *restrict p,
                           const double _Complex *restrict x0,
                           const double *restrict cf,
                           double _Complex *restrict fo, int m)
{
    double *sre = p->sre, *sim = p->sim;
    const double *kt = L13R_SOKT(p);
    l13r_conv_in((const double *)x0, sre, sim);
    l13r_conv_in(cf, p->cre, p->cim);
#if L13R_MF == 1
    /* buffer holds STATE between steps; map fused into sweep-B stores */
    for (int s = 1; s <= m; ++s)
        l13r_soa_step(sre, sim, p->cre, p->cim, kt, 1);
#elif L13R_MF == 2
    /* buffer holds raw z; map per x-plane at the head of sweep A */
    l13r_soa_step(sre, sim, p->cre, p->cim, kt, -1);      /* x0 = state_0 */
    for (int s = 2; s <= m; ++s)
        l13r_soa_step(sre, sim, p->cre, p->cim, kt, 1);
    l13r_soa_map_pass(sre, sim, p->cre, p->cim, 2197);
#else
    /* buffer holds raw z; map as its own streaming pass */
    l13r_soa_step(sre, sim, p->cre, p->cim, kt, 0);       /* z_1 raw */
    for (int s = 2; s <= m; ++s) {
        l13r_soa_map_pass(sre, sim, p->cre, p->cim, 2197);
        l13r_soa_step(sre, sim, p->cre, p->cim, kt, 0);
    }
    l13r_soa_map_pass(sre, sim, p->cre, p->cim, 2197);
#endif
    l13r_conv_out(sre, sim, (double *)fo);
}

#endif /* L13R_SOA */

#endif /* L13R_CHAIN_FAST */

/* ---------------------------------------------------------------------------
 * plumbing
 * ---------------------------------------------------------------------------
 */
static char g_desc[512] =
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
                + 13 * L13R_PBROW + 2 * 13 * L13R_T1P + 2 * 208 + 96 + 16;
    p->block = malloc(nd * sizeof(double) + 64);
    if (!p->block) { free(p); return NULL; }
    double *q = L13R_ALIGN64((double *)p->block);
    p->cpd8 = q; q += 6 * 8;
    p->snt8 = q; q += 6 * 8;
    p->cpd4 = q; q += 6 * 4;
    p->snt4 = q; q += 6 * 4;
    p->soah = q; q += 72;
    p->sork = q; q += 24;   /* 12 used, padded to keep 64 B alignment */
    p->pb = q;   q += 13 * L13R_PBROW;
    p->t1 = q;   q += 13 * L13R_T1P;
    p->t1b = q;  q += 13 * L13R_T1P;
    p->sg = q;   /* 2 rotating 13x16-double map stages, 64 B aligned */
    memset(p->pb, 0,
           (13 * L13R_PBROW + 2 * 13 * L13R_T1P + 2 * 208) * sizeof(double));

    /* uf arm's ping-pong mate for final_out (the fused bodies run in place
     * and never touch it).  Never read before written. */
    if (posix_memalign((void **)&p->zb, 64, (size_t)batch * 2197 * 16)) {
        free(p->block); free(p); return NULL;
    }

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
        /* ice_r7 SoA pencil tables.  Hartley reg6 (exact mod-13 reduction):
         * soah[(j-1)*12 + (k-1)*2 + {0,1}] = cos/sin(2pi (k*j mod 13)/13).
         * Rader split: sork[0..2] = CP, [3..5] = CM (same halved cyclic/
         * negacyclic constants as cpd8), [6..11] = sin(2pi g^t/13), PLAIN
         * (the lane-alternating sign fold is a rename in split form). */
        for (int j = 1; j <= 6; ++j)
            for (int k = 1; k <= 6; ++k) {
                int r = (k * j) % 13;
                long double a = PI2 * (long double)r / 13.0L;
                p->soah[(j - 1) * 12 + (k - 1) * 2] = (double)cosl(a);
                p->soah[(j - 1) * 12 + (k - 1) * 2 + 1] = (double)sinl(a);
            }
        for (int t = 0; t < 3; ++t) {
            double ca = (double)cosl(PI2 * (long double)gt[t] / 13.0L);
            double cb = (double)cosl(PI2 * (long double)gt[t + 3] / 13.0L);
            p->sork[t] = 0.5 * (ca + cb);
            p->sork[3 + t] = 0.5 * (ca - cb);
        }
        for (int t = 0; t < 6; ++t)
            p->sork[6 + t] = (double)sinl(PI2 * (long double)gt[t] / 13.0L);
    }

    /* ice_r7: SoA batch-lane buffers (state + converted c for one group of
     * 8 volumes; 4 x 17576 doubles + 5-line skews = 564 KB, L2-resident).
     * NULL when batch < 8 or the fast chain is compiled out. */
#if defined(L13R_CHAIN_FAST) && L13R_SOA && !defined(L13R_CFORCE)
    if (batch >= 8) {   /* a CFORCE pin means "measure that classic arm" */
        size_t one = 17576 + 40;
        if (posix_memalign((void **)&p->soab, 64, 4 * one * sizeof(double)))
            p->soab = NULL;
        if (p->soab) {
            double *sb = (double *)p->soab;
            p->sre = sb;
            p->sim = sb + one;
            p->cre = sb + 2 * one;
            p->cim = sb + 3 * one;
        }
    }
#endif

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

    /* ---- ice_r6 CHAIN-ARM race.  Arms (all OUTPUT-BIT-IDENTICAL: one map
     * DAG per point at every call site, whole-unit/whole-volume reordering
     * only, so the per-process pick can never change output bits):
     *   cv  step-major cross-step ov pipeline (the r4/r5 shape)
     *   cz  step-major per-step paired X pass (the nb==1 shape)
     *   v1/v2/v4  VOLUME-GROUP-MAJOR at G=1/2/4 (adopted: L13_direct
     *       ice_r5 vm2 <- L17_matrixsimd ice_r4): all m steps over one
     *       G-volume group (~215 KB at G=2, L2-resident) before the next.
     *       Incumbent v2 at nb>=2 (their race picked v2 every window; v1
     *       and v4 tied within 0.6% -- kept as cheap insurance).
     * uf (map pass + exec ping-pong) is DROPPED from the race: it lost
     * 27-38% in every r5 window; -DL13R_CFORCE=2 still reaches it.
     * tb=min(batch,32) volumes of genuine raw z (one exec of LCG noise)
     * evolve IN PLACE under real step semantics with a synthetic
     * 0.1-scaled c; ~120 ms dense-FMA clock-settle spin first
     * (L17_winograd via L13_direct).  6 steps per timed rep (cv/v-arms
     * re-run their X prologue each rep: small equal-shape bias, same as
     * L13_direct's fch-ab).  Challenger adopted only if >1.5% better in
     * BOTH of two independent blocks (L6_pfa hysteresis).
     * -DL13R_AB=0 removes the race; -DL13R_CFORCE=0..5 pins
     * cv/cz/uf/v1/v2/v4. */
#ifndef L13R_AB
#  define L13R_AB 1
#endif
#ifndef L13R_ABTH
#  define L13R_ABTH 0.985
#endif
    p->cuf = 1;
    p->cg = 0;
    const char *cpick = "uf";
#if defined(L13R_CHAIN_FAST)
    if (batch >= 2) { p->cbody = l13r_chain_ov; p->cg = 2; cpick = "v2"; }
    else            { p->cbody = l13r_chain_zs; p->cg = 0; cpick = "cz"; }
    p->cuf = 0;
#endif
    char abuf[160] = "";
#if defined(L13R_CFORCE) && defined(L13R_CHAIN_FAST)
    switch ((int)L13R_CFORCE) {
    case 0: p->cbody = l13r_chain_ov; p->cuf = 0; p->cg = 0; cpick = "FORCED cv"; break;
    case 1: p->cbody = l13r_chain_zs; p->cuf = 0; p->cg = 0; cpick = "FORCED cz"; break;
    case 2: p->cuf = 1;                                      cpick = "FORCED uf"; break;
    case 3: p->cbody = l13r_chain_zs; p->cuf = 0; p->cg = 1; cpick = "FORCED v1"; break;
    case 4: p->cbody = l13r_chain_zs; p->cuf = 0; p->cg = 2; cpick = "FORCED v2"; break;
    case 5: p->cbody = l13r_chain_zs; p->cuf = 0; p->cg = 4; cpick = "FORCED v4"; break;
    default: break;
    }
#elif L13R_AB && defined(L13R_CHAIN_FAST)
    do {
#if L13R_SOA
        /* every volume goes through the SoA path: the classic-arm race would
         * pick a shape that never runs -- skip it (create-time only). */
        if (p->soab && (batch & 7) == 0) { cpick = "soa8"; break; }
#endif
        const char *tag[5] = { "cv", "cz", "v1", "v2", "v4" };
        static const int gsz[5] = { 0, 0, 1, 2, 4 };
        const int nv = 5;
        const int tb = batch < 32 ? batch : 32;
        size_t vn = (size_t)2197 * tb;
        double _Complex *ra = 0, *rb = 0, *rc = 0;
        if (posix_memalign((void **)&ra, 64, vn * 16)) break;
        if (posix_memalign((void **)&rb, 64, vn * 16)) { free(ra); break; }
        if (posix_memalign((void **)&rc, 64, vn * 16)) { free(ra); free(rb); break; }
        {   /* deterministic fill, no rand(): x0-like noise + ~0.1-scaled c */
            unsigned long long s = 0x9e3779b97f4a7c15ull;
            double *rd = (double *)ra, *rcd = (double *)rc;
            for (size_t i = 0; i < 2 * vn; ++i) {
                s = s * 6364136223846793005ull + 1442695040888963407ull;
                rd[i] = (double)(long long)(s >> 12) * 0x1p-50;
                s = s * 6364136223846793005ull + 1442695040888963407ull;
                rcd[i] = (double)(long long)(s >> 12) * 0x1p-53;
            }
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
        tp.exec(&tp, ra, rb);        /* genuine raw z_1 into rb */
        enum { CMS = 6 };            /* steps per timed rep: CMS-1 fused */
        double b1[5] = {1e30, 1e30, 1e30, 1e30, 1e30};
        double b2[5] = {1e30, 1e30, 1e30, 1e30, 1e30};
        double _Complex *cur = rb;
        for (int t = -1; t < 12; ++t)          /* t = -1 warms the buffers */
            for (int j = 0; j < nv; ++j) {
                int k = (t & 1) ? nv - 1 - j : j;
                if (tb < 2 && k != 1) continue;  /* cv needs nb>=2; the
                                                  * v-arms all equal cz */
                double t0 = l13r_now();
                if (k == 0)
                    l13r_chain_ov(&tp, (double *)cur, (const double *)rc, CMS);
                else if (k == 1)
                    l13r_chain_zs(&tp, (double *)cur, (const double *)rc, CMS);
                else {
                    const int G = gsz[k];
                    for (int b0 = 0; b0 < tb; b0 += G) {
                        int g = tb - b0; if (g > G) g = G;
                        tp.batch = g;
                        double *sg_ = (double *)cur + (size_t)b0 * 4394;
                        const double *cg_ = (const double *)rc
                                            + (size_t)b0 * 4394;
                        if (g >= 2) l13r_chain_ov(&tp, sg_, cg_, CMS);
                        else        l13r_chain_zs(&tp, sg_, cg_, CMS);
                    }
                    tp.batch = tb;
                }
                double dt = (l13r_now() - t0) / ((double)tb * (CMS - 1));
                if (t < 0) continue;
                double *bb = (t < 6) ? b1 : b2;
                if (dt < bb[k]) bb[k] = dt;
            }
        int win = (tb >= 2) ? 3 : 1;           /* deterministic incumbent */
        for (int k = 0; k < nv; ++k) {
            if (k == win || (tb < 2 && k != 1)) continue;
            if (b1[k] < L13R_ABTH * b1[win] && b2[k] < L13R_ABTH * b2[win])
                win = k;
        }
        p->cuf = 0;
        p->cg = gsz[win];
        p->cbody = (win == 0) ? l13r_chain_ov : l13r_chain_zs;
        cpick = tag[win];
        {
            int off = snprintf(abuf, sizeof abuf, " fch-ab[B%d]=", tb);
            int np = 0;
            for (int k = 0; k < nv && off < (int)sizeof abuf; ++k) {
                if (tb < 2 && k != 1) continue;
                double best = b1[k] < b2[k] ? b1[k] : b2[k];
                off += snprintf(abuf + off, sizeof abuf - off, "%s%s:%.0f",
                                np++ ? "," : "", tag[k], best);
            }
        }
        fprintf(stderr, "L13_rader%s cpick=%s\n", abuf, cpick);
        free(ra); free(rb); free(rc);
    } while (0);
#endif

    snprintf(g_desc, sizeof g_desc,
             "Rader-13; ice_r7 SoA BATCH-LANE chain (8 vols/zmm split re/im, "
             "zero-shuffle vertical pencils, 2 sweeps/step: map+z+y per L1 "
             "x-plane, then x-pencils; kernel=%s mf=%d sqr=%d; convert 1/m; "
             "mined from 1000f989/v5_3907583b/v6_f40c5e25), tails+B<8 = r6 "
             "vm classic, %d-bit; exec=%s chain=%s%s",
#if defined(L13R_CHAIN_FAST) && L13R_SOA
             L13R_SK ? "rader-split186" : "hartley206",
             (int)L13R_MF, (int)L13R_SQR,
#else
             "off", 0, 0,
#endif
#if defined(__AVX512F__)
             512,
#else
             256,
#endif
             pick, cpick, abuf);
    return p;
}

/* ice_r5: own the whole graded chain.  state_0 = x0; m times
 * { z = FFT(state); state = (z+c)/(1+|z+c|) }; final_out = state_m.
 * Fused bodies run IN PLACE on final_out (raw z between steps, map fused
 * into each step's X pass -- L13_direct r4's single-buffer collapse); one
 * in-place map pass ends the chain.  The uf arm keeps the r4 ping-pong
 * (exec's restrict contract forbids in==out). */
void fft3d_chain(fft3d_plan *p, const double _Complex *x0,
                 const double _Complex *c, double _Complex *final_out, int m)
{
    if (!p || m < 1) return;
    const size_t vn = (size_t)p->batch * 2197;
    const double *cf = (const double *)c;
#if defined(L13R_CHAIN_FAST)
    if (!p->cuf) {
#if L13R_SOA
        if (p->soab) {
            /* ice_r7 SoA batch-lane groups of 8 (see header); the tail
             * volumes (batch % 8) run the r6 classic machinery.  This is a
             * DETERMINISTIC dispatch, not a raced arm: SoA arithmetic is
             * not bit-identical to the classic pipeline's, so the split by
             * volume index must be the same in every process -- and it is. */
            int nb8 = p->batch & ~7;
            for (int b0 = 0; b0 < nb8; b0 += 8)
                l13r_chain_soa(p, x0 + (size_t)b0 * 2197,
                               cf + (size_t)b0 * 4394,
                               final_out + (size_t)b0 * 2197, m);
            if (nb8 < p->batch)
                l13r_chain_vm(p, p->cg > 0 ? p->cg : 2, p->batch - nb8, m,
                              x0 + (size_t)nb8 * 2197,
                              cf + (size_t)nb8 * 4394,
                              final_out + (size_t)nb8 * 2197);
            return;
        }
#endif
        if (p->cg > 0) {
            /* ice_r6 VOLUME-GROUP-MAJOR (L13_direct ice_r5 vm2 <-
             * L17_matrixsimd ice_r4): all m steps over one G-volume group
             * (~215 KB at G=2: L2-resident) before the next group.  Whole-
             * volume reordering across independent volumes: bit-identical
             * to the step-major bodies for every G. */
            l13r_chain_vm(p, p->cg, p->batch, m, x0, cf, final_out);
            return;
        }
        p->exec(p, x0, final_out);         /* z_1, raw, into the state buf */
        if (m > 1)
            p->cbody(p, (double *)final_out, cf, m);
        l13r_map_pass((double *)final_out, (const double *)final_out, cf, vn);
        return;
    }
#endif
    {
        double _Complex *A = (m & 1) ? final_out : p->zb;
        double _Complex *B = (m & 1) ? p->zb : final_out;
        p->exec(p, x0, A);                 /* step 1: plain FFT of x0 */
        for (int s = 1; s < m; ++s) {      /* unfused: in-place map + exec */
            l13r_map_pass((double *)A, (const double *)A, cf, vn);
            p->exec(p, A, B);
            double _Complex *sw = A; A = B; B = sw;
        }
        l13r_map_pass((double *)final_out, (const double *)final_out, cf, vn);
    }
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    plan->exec(plan, in, out);
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->soab);
    free(p->zb);
    free(p->block);
    free(p);
}

#endif /* L13R_TEMPLATE_PASS */
