/* =============================================================================
 * L17_matrixsimd -- 17^3 complex-double forward DFT as three dense 17x17
 *                   matrix passes, conjugate-pair folded, vectorised across
 *                   whole lines (SIMD lanes = independent lines).
 *
 * TECHNIQUE
 *   Row-column: one dense length-17 DFT matrix applied along each axis.  The
 *   17x17 complex matrix is not applied naively; the j <-> 17-j conjugate pair
 *   is folded first (this is FFTW's dft-generic form), which turns one complex
 *   17x17 matrix-vector product into two REAL matrix products on complex data:
 *
 *     u_j = x_j + x_{17-j},  v_j = x_j - x_{17-j}          (j = 1..8)
 *     P_k = x_0 + sum_{j=1..8} cos(2pi kj/17) u_j          (k = 0..8)
 *     R_k =       sum_{j=1..8} sin(2pi kj/17) (-i v_j)     (k = 1..8)
 *     X_k = P_k + R_k,  X_{17-k} = P_k - R_k,  X_0 = P_0
 *
 *   The surviving coefficients are REAL, so the interleaved complex layout the
 *   driver hands us is already the right SIMD layout: a real scalar times an
 *   interleaved complex vector is one vector FMA with a broadcast memory
 *   operand.  There is not a single cross-lane operation in the arithmetic --
 *   the only permutes are the re/im swap that implements (-i v) and the
 *   in-register plane transposes that feed the 2nd and 3rd axis.
 *
 * OPERATION COUNT  (per 17^3 volume = 3*289 = 867 lines)
 *   per line, in real arithmetic: 272 FMAs (544 flop) + 16 complex adds and 16
 *   complex subs (butterfly + combine, 64 flop) = 608 flop.  Compare:
 *     naive dense complex 17x17 matvec  2312 flop/line  (289 complex MACs)
 *     this kernel (conj-pair folded)     608 flop/line  (3.8x less)
 *     FFTW's own dft-generic-17          592 flop/line  (k=0 row as adds)
 *     Rader-17                           468 flop/line  but 388 instructions
 *                                        vs 336, plus a gather (LITERATURE 3.4)
 *   per volume: 867*608 = 527 kflop.  The driver's yardstick, 5 N log2 N, is
 *   301 kflop, so a reported "GF/s" of 17 means ~30 real Gflop/s.
 *
 *   As VECTOR instructions, which is what actually costs: one chunk transforms
 *   WC lines at once and needs 136 FMAs + 16 add/sub (butterfly) + 16 add/sub
 *   (combine) = 168 FP ops, plus 8 integer XORs, 33 loads, 20 stores and
 *   40 shuffles (the tile transpose).  17 = 4*4+1 forces 5 chunks per 17-long
 *   free index instead of 4.25, so a volume costs 243 chunks at WC=4 (ideal
 *   217) => 40.8k vector FP ops per volume.  On a Cascade Lake with ONE 512-bit
 *   FMA unit that is ~41k cycles = 17.7 us at 2.3 GHz; the 256-bit kernel does
 *   451 chunks at 2 FMA/cycle = 37.9k cycles, i.e. the two widths are predicted
 *   to tie, which is why both are built and measured (see below).
 *
 * LAYOUT / SIMD
 *   WC complex per vector (4 = zmm, 2 = ymm).  Lanes always hold WC *different
 *   lines*, taken from a contiguous run of a free index, so every load and
 *   store is a contiguous vector access and every coefficient is a
 *   lane-invariant broadcast.  The last chunk of a 17-long index overlaps the
 *   previous one (offsets 0,4,8,12,13): it recomputes 3 lines and stores
 *   bit-identical values over them -- cheaper than masking, and it never reads
 *   out of range.
 *
 *   Passes (one kernel, two store modes):
 *     Y : in[x][y][z] --lanes over z --> pb[z][ky]       (transposing store)
 *     Z : pb[z][ky]   --lanes over ky--> t1[x][ky][kz]   (transposing store)
 *     X : t1[x][p]    --lanes over p (289 wide) --> out[kx][p]   (plain store)
 *   Y and Z run plane by plane, so the 4.6 KiB plane stays in L1; only pass X
 *   walks the whole 78.6 KiB volume (8% of L2).  Exactly two plane transposes
 *   are unavoidable (both in and out are [x][y][z]); they are pure port-5 work
 *   and hide under the FMA stream.  Pass X is last so that the stores into the
 *   caller's `out` are the sequential ones.
 *
 * ROUND panel_r2: NESTED KERNEL (borrowed from L17_winograd round 1)
 *   The cosine half of the folded matrix is a length-8 CYCLIC correlation once
 *   j,k are reindexed by powers of 3 mod 17, and it splits with sign-only
 *   reductions into a cyclic-4 + negacyclic-4 (see chunk17n below for the full
 *   derivation).  That takes a chunk from 168 FP ops to 148 (-12%) with the
 *   data flow unchanged -- every coefficient is still real, the permutation is
 *   compile-time load offsets.  Round 1 measured this kernel FMA-port-bound at
 *   1.045 FP ops/cycle on the node, so the saving should be nearly 1:1 in time.
 *   The sine half stays a dense negacyclic-8: L17_winograd's record counts all
 *   the splits of x^8+1 and each one loses; do not retry them.
 *
 * ROUND panel_r3: PINNED SINE CONSTANTS + X-FIRST PASS ORDER
 *   (1) The negacyclic-8 sine matrix has only 8 distinct constants; they are
 *   broadcast once per execute into 8 registers made opaque with an empty asm
 *   (adopted from L17_winograd's round-2 variant C) and the sine sweep is
 *   fully unrolled with the signs baked in as compile-time vfnmadd -- 64 of
 *   ~137 loads per chunk gone.  (2) X-first order: transform x from `in` into
 *   t1, then finish each kx plane with Y,Z and store it straight into its
 *   4.6 KiB slot of `out`, so at batch the output writes spread across the
 *   volume's compute instead of bursting in 73 chunks at the end (wallaby:
 *   -17% at B=256, -14% at B=2048; ~5% slower when cache-resident, so it is
 *   used only for batch >= 64).
 *
 * ROUND panel_r4: CROSS-VOLUME SOFTWARE PIPELINING (batch >= 64)
 *   The r3 verdict quantified 6.3 us/volume of un-overlapped memory time at
 *   B=2048 (measured 22.7 vs a 16.4 us compute floor).  The X-first order
 *   still opens every volume with a serial 78.6 KiB DRAM read burst (the 73
 *   X chunks).  The pipelined variants (exec18/19/20) double-buffer t1 and
 *   interleave volume b+1's X chunks, 4-5 per plane, into volume b's plane
 *   phase, so the input read of every volume but the first overlaps the
 *   previous volume's compute.  Pure scheduling; cmp-verified to stay in the
 *   X-first bit class.
 *
 * ROUND panel_r5: MIXED-WIDTH TAIL (512-bit body + ymm tail chunks)
 *   Adopted from L17_rader's panel_r4 "512t" candidates, which the node tuner
 *   picked in every cell: the Gold 5218 has ONE fused 512-bit FMA unit but
 *   TWO 256-bit FMA ports, so a ymm chunk retires its 148 FP ops in ~74
 *   cycles where a zmm chunk needs ~148.  A 17-long free index then costs
 *   4 zmm chunks + 1 ymm tail (offsets 0,4,8,12 + 15, recomputing one line)
 *   instead of 5 zmm (0,4,8,12,13, recomputing three): ~4.5 zmm-equivalents
 *   instead of 5.  Per volume 225.5 equivalents against 243 pure-zmm, -7.2%
 *   of FMA-port time on the node.  On wallaby (TWO 512-bit units) the mix is
 *   FP-neutral and slightly worse on instruction count, so it ships as tuner
 *   candidates: wallaby rejects them, the node should take them.  Also new:
 *   a sustained-clock probe (512-bit and 256-bit licence clocks measured at
 *   plan time, reported in the description string -- the monitor's r4 ask).
 *
 * ROUND panel_r6: PREFETCHW + DEFERRED-Z + DENSE-256 CLOCK PROBE
 *   (1) pw: write-intent prefetch (prefetchw) of each out plane, issued in two
 *   half-bursts ~0.25 us before the Z group stores it, for the X-first batch
 *   variants.  Adopted from L8_fusedaxes r5 (pfw) and L36_pfa r5 (pf=2), the
 *   only store-side mechanism the node has ever selected (NT stores lost four
 *   rounds running: hide the RFO, do not avoid it).  A/B'd jointly with pf.
 *   (2) Deferred-Z plane schedule (execm_xld/xfd): Y(x+1) runs between Y(x)
 *   and Z(x) on double-buffered plane buffers, so the Z group's store->load
 *   dependence on its own Y group always has a full independent group between
 *   them -- aimed at the ~44 cycles/chunk of non-FP time the r5 VERDICT
 *   quantified at B=1.  (3) A dense (2 FMA/cycle) 256-bit clock probe next to
 *   the sparse one, same process, to settle the r5 clk256 disagreement.
 *
 * KERNELS, CHOSEN BY MEASUREMENT IN fft3d_create() -- WITHIN A BIT-CLASS
 *   Not every variant rounds identically: X-first reorders the passes, and
 *   the pinned path contracts differently under gcc (measured with cmp; the
 *   fnmadd-vs-negated-constant argument that says they must match is wrong in
 *   practice).  A wall-clock tuner is not deterministic across processes, so
 *   letting it choose across rounding classes makes two runs of the same
 *   binary disagree bitwise, which the panel's repeatability check rightly
 *   flags.  Therefore: the CLASS is a pure function of the batch size
 *   (batch < 64: X-last pinned; batch >= 64: X-first pinned), and the tuner
 *   selects freely only within it -- vector width, C-parking, NT staging and
 *   prefetch, all verified by cmp to change no bits.  Every other variant is
 *   still measured for the record (L17_VERBOSE=1 prints the table).
 *
 *   Stage 1 (batch < 64): blocked timing of all candidates on <= 16 volumes,
 *   >= 64 transforms per block, never interleaved: interleaving 512-bit and
 *   256-bit kernels makes every sample pay an AVX-512 licence/frequency
 *   transition, which cost 35% and mis-ranked the candidates on the real node.
 *   Stage 1b (batch >= 64): the same, but on 384 volumes (~60 MB, past any
 *   L3), because the streaming regime can prefer a different kernel outright
 *   (adopted from L17_winograd round 2, which measured a 90% penalty for
 *   trusting the small-set pick at batch).
 *   Stage 2 (batch >= 64): NT-store A/B on the winner, blocked (a plain run
 *   leaves `out` dirty in L3 and an alternating A/B inverts the answer).  NT
 *   won 19% on a contended machine and LOST 20% on the isolated node in r1 --
 *   measured, not assumed.  Stage 2b: cross-volume input prefetch A/B
 *   (adopted from L17_winograd round 2); prefetch changes no bits.
 *
 *   fft3d_description() reports the winner (and pf=) so the leaderboard JSON
 *   carries the machine's answer back.  Setup is excluded from the score.
 *
 *   Measured on the benchmark node (Xeon Gold 5218, exclusive, gcc 11.4,
 *   -march=native), per transform, against MKL 2022 on the same data:
 *     B=1     16.99 us  (MKL  98.8)      B=8     19.37 us  (MKL  99.9)
 *     B=256   28.19 us  (MKL 101.2)      B=2048  29.41 us  (MKL 103.8)
 *   The winner was the 512-bit k-blocked kernel at every batch size.
 *   See strategies/L17_matrixsimd.md for the full tables.
 *
 * ASSUMPTIONS
 *   * L == 17 only; in and out distinct (driver guarantees).  Only 8-byte
 *     alignment of in/out is required.
 *   * gcc/clang vector extensions and -ffp-contract=fast (default for gnu11)
 *     for FMA formation.  No intrinsics anywhere, so one source serves AVX-512,
 *     AVX2 and plain SSE2, and the 512-bit kernel can be RUN (emulated, slowly,
 *     but bit-correct) on a machine without AVX-512 -- which is how the graded
 *     path was verified on an AVX2-only development machine.
 *   * The template body is instantiated twice by a self-#include of this file
 *     ("L17_matrixsimd.c", found in the includer's own directory).
 * =============================================================================
 */
#ifdef L17_TEMPLATE_PASS
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

/* MULI(t) = -i*t for interleaved complex: swap re/im, then flip the sign of
 * the odd (imaginary) lanes.  The flip is an integer XOR rather than a
 * multiply by {1,-1,...} on purpose: on the 512-bit port scheme a vmulpd can
 * only issue to the FMA port, which is the bottleneck, whereas vpxorq can also
 * go to port 5, which is idle. */
#define SIGN64 ((long long)0x8000000000000000LL)
#if WC == 4
#  define SWAPRI(x) SHUF1(x, 1, 0, 3, 2, 5, 4, 7, 6)
#  define NEGMASK ((IT){0, SIGN64, 0, SIGN64, 0, SIGN64, 0, SIGN64})
#else
#  define SWAPRI(x) SHUF1(x, 1, 0, 3, 2)
#  define NEGMASK ((IT){0, SIGN64, 0, SIGN64})
#endif
#define MULI(t) ((VT)((IT)SWAPRI(t) ^ NEGMASK))

/* Pin the 8 sine constants of the nested kernel in registers across a whole
 * execute: the empty asm makes them opaque, so gcc can neither rematerialise
 * the broadcasts nor fold them back into per-FMA memory operands.  Adopted
 * from L17_winograd's round-2 "variant C" (its register-residency fix).  Only
 * meaningful on a 32-register EVEX file; on a 16-register file the pin would
 * guarantee spills, so it is a no-op there and the constants simply decay back
 * to memory operands (the kernel is then equivalent to the unpinned one). */
#if (WC == 4 && defined(__AVX512F__)) || (WC == 2 && defined(__AVX512VL__))
#  define L17_PIN_ASM() __asm__("" : "+v"(K0), "+v"(K1), "+v"(K2), "+v"(K3),  \
                                     "+v"(K4), "+v"(K5), "+v"(K6), "+v"(K7))
#else
#  define L17_PIN_ASM() do { } while (0)
#endif

/* Coefficients come from a PRE-SPLATTED table: each coefficient is stored
 * VDW times, so an FMA reads it as a plain full-width memory operand.  Using
 * the scalar table plus a broadcast looks cheaper (AVX-512 has an embedded
 * broadcast) but gcc then materialises all 136 splats into stack slots inside
 * the chunk loop -- 272 extra instructions per chunk, measured. */
#define CGET(base, i) VLD((base) + (size_t)(i) * VDW)
#define CSTEP_C (9 * VDW)
#define CSTEP_S (8 * VDW)
#if WC == 4
#  define CTAB(p) ((p)->ctab8)
#  define STAB(p) ((p)->stab8)
#  define NTCTAB(p) ((p)->ntc8)
#  define NTSTAB(p) ((p)->nts8)
#else
#  define CTAB(p) ((p)->ctab4)
#  define STAB(p) ((p)->stab4)
#  define NTCTAB(p) ((p)->ntc4)
#  define NTSTAB(p) ((p)->nts4)
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

/* -----------------------------------------------------------------------
 *  One chunk: the length-17 DFT of WC lines at once.
 *    src, rs : element (j=0, lane 0); rs doubles between successive j
 *    dst, da, db : output element (m) of lane (f) is at dst + f*da + m*db
 *    tr : 1 -> lanes are separate rows of dst (in-register transpose needed)
 *         0 -> lanes are contiguous in dst (plain vector stores)
 * --------------------------------------------------------------------- */
static inline __attribute__((always_inline)) void
SUF(chunk17)(const double *restrict src, long rs, double *restrict dst, long da,
             long db, const double *restrict ct, const double *restrict st, int tr)
{
    VT P0, P1, P2, P3, P4, P5, P6, P7, P8, R1, R2, R3, R4, R5, R6, R7, R8;

    /* ---- symmetric half: 9 accumulators, 8 rank-1 updates ---- */
    {
        VT v0 = VLD(src);
        P0 = v0; P1 = v0; P2 = v0; P3 = v0; P4 = v0;
        P5 = v0; P6 = v0; P7 = v0; P8 = v0;
        for (int j = 1; j <= 8; ++j) {
            VT u = VLD(src + (long)j * rs) + VLD(src + (long)(17 - j) * rs);
            const double *c = ct + (size_t)(j - 1) * CSTEP_C;
            P0 += CGET(c, 0) * u;
            P1 += CGET(c, 1) * u;
            P2 += CGET(c, 2) * u;
            P3 += CGET(c, 3) * u;
            P4 += CGET(c, 4) * u;
            P5 += CGET(c, 5) * u;
            P6 += CGET(c, 6) * u;
            P7 += CGET(c, 7) * u;
            P8 += CGET(c, 8) * u;
        }
    }

    /* ---- antisymmetric half: 8 accumulators, j=1 peeled to a multiply ---- */
    {
        VT w = MULI(VLD(src + rs) - VLD(src + 16 * rs));
        R1 = CGET(st, 0) * w;
        R2 = CGET(st, 1) * w;
        R3 = CGET(st, 2) * w;
        R4 = CGET(st, 3) * w;
        R5 = CGET(st, 4) * w;
        R6 = CGET(st, 5) * w;
        R7 = CGET(st, 6) * w;
        R8 = CGET(st, 7) * w;
        for (int j = 2; j <= 8; ++j) {
            w = MULI(VLD(src + (long)j * rs) - VLD(src + (long)(17 - j) * rs));
            const double *s = st + (size_t)(j - 1) * CSTEP_S;
            R1 += CGET(s, 0) * w;
            R2 += CGET(s, 1) * w;
            R3 += CGET(s, 2) * w;
            R4 += CGET(s, 3) * w;
            R5 += CGET(s, 4) * w;
            R6 += CGET(s, 5) * w;
            R7 += CGET(s, 6) * w;
            R8 += CGET(s, 7) * w;
        }
    }
#define GP(k) P##k
#define GR(k) R##k

/* output index m: X_0 = P_0, X_k = P_k + R_k, X_{17-k} = P_k - R_k */
#define X0 GP(0)
#define XP(k) (GP(k) + GR(k))
#define XM(k) (GP(k) - GR(k))

    if (!tr) {
        VST(dst + 0 * db, X0);
        VST(dst + 1 * db, XP(1));  VST(dst + 2 * db, XP(2));
        VST(dst + 3 * db, XP(3));  VST(dst + 4 * db, XP(4));
        VST(dst + 5 * db, XP(5));  VST(dst + 6 * db, XP(6));
        VST(dst + 7 * db, XP(7));  VST(dst + 8 * db, XP(8));
        VST(dst + 9 * db, XM(8));  VST(dst + 10 * db, XM(7));
        VST(dst + 11 * db, XM(6)); VST(dst + 12 * db, XM(5));
        VST(dst + 13 * db, XM(4)); VST(dst + 14 * db, XM(3));
        VST(dst + 15 * db, XM(2)); VST(dst + 16 * db, XM(1));
    } else {
#if WC == 4
#  define TILE4(m0, e0, e1, e2, e3)                                            \
      do {                                                                     \
          VT xx[4], yy[4];                                                     \
          xx[0] = (e0); xx[1] = (e1); xx[2] = (e2); xx[3] = (e3);               \
          TTILE(xx, yy);                                                       \
          VST(dst + 0 * da + (m0) * 2, yy[0]);                                 \
          VST(dst + 1 * da + (m0) * 2, yy[1]);                                 \
          VST(dst + 2 * da + (m0) * 2, yy[2]);                                 \
          VST(dst + 3 * da + (m0) * 2, yy[3]);                                 \
      } while (0)
        TILE4(0, X0, XP(1), XP(2), XP(3));
        TILE4(4, XP(4), XP(5), XP(6), XP(7));
        TILE4(8, XP(8), XM(8), XM(7), XM(6));
        TILE4(12, XM(5), XM(4), XM(3), XM(2));
        TILE4(13, XM(4), XM(3), XM(2), XM(1));
#  undef TILE4
#else
#  define TILE2(m0, e0, e1)                                                    \
      do {                                                                     \
          VT xx[2], yy[2];                                                     \
          xx[0] = (e0); xx[1] = (e1);                                          \
          TTILE(xx, yy);                                                       \
          VST(dst + 0 * da + (m0) * 2, yy[0]);                                 \
          VST(dst + 1 * da + (m0) * 2, yy[1]);                                 \
      } while (0)
        TILE2(0, X0, XP(1));
        TILE2(2, XP(2), XP(3));
        TILE2(4, XP(4), XP(5));
        TILE2(6, XP(6), XP(7));
        TILE2(8, XP(8), XM(8));
        TILE2(10, XM(7), XM(6));
        TILE2(12, XM(5), XM(4));
        TILE2(14, XM(3), XM(2));
        TILE2(15, XM(2), XM(1));
#  undef TILE2
#endif
    }
#undef X0
#undef XP
#undef XM
#undef GP
#undef GR
}

/* -----------------------------------------------------------------------
 *  Same transform, blocked over k so that the tile transposes happen while
 *  only 9 accumulators are live (peak 17 vectors instead of 25).  Costs one
 *  extra butterfly pass over j (+24 FP ops/chunk, +11%) and buys the register
 *  allocator a lot of room; which trade wins is measured in fft3d_create().
 *    block A: k = 0..4  -> outputs m = 0..4 and 13..16
 *    block B: k = 5..8  -> outputs m = 5..12
 * --------------------------------------------------------------------- */
static inline __attribute__((always_inline)) void
SUF(chunk17b)(const double *restrict src, long rs, double *restrict dst, long da,
              long db, const double *restrict ct, const double *restrict st,
              double *restrict sc, int tr, int bst)
{
/* bst = 0: block B recomputes the butterfly (24 extra FP ops per chunk)
 * bst = 1: block A parks u_j and w_j in an L1 scratch and block B reads them
 *          back (16 stores + 16 loads instead, off the FMA port) */
#define BFLY_DECL(j)                                                           \
    const double *c = ct + (size_t)((j) - 1) * CSTEP_C;                        \
    const double *sn = st + (size_t)((j) - 1) * CSTEP_S;                       \
    VT u, w;                                                                   \
    do {                                                                       \
        VT a = VLD(src + (long)(j) * rs), b = VLD(src + (long)(17 - (j)) * rs); \
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
            VT a = VLD(src + (long)(j) * rs), b = VLD(src + (long)(17 - (j)) * rs); \
            u = a + b;                                                         \
            w = MULI(a - b);                                                   \
        }                                                                      \
    } while (0)

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

    { /* ---- block A: k = 0..4  ->  outputs m = 0..4, 13..16 ---- */
        VT Q0, X1, X2, X3, X4, Y1, Y2, Y3, Y4;
        {
            VT P0, P1, P2, P3, P4, R1, R2, R3, R4;
            VT v0 = VLD(src);
            P0 = v0; P1 = v0; P2 = v0; P3 = v0; P4 = v0;
            {
                BFLY_DECL(1);
                P0 += CGET(c, 0) * u; P1 += CGET(c, 1) * u; P2 += CGET(c, 2) * u;
                P3 += CGET(c, 3) * u; P4 += CGET(c, 4) * u;
                R1 = CGET(sn, 0) * w; R2 = CGET(sn, 1) * w;
                R3 = CGET(sn, 2) * w; R4 = CGET(sn, 3) * w;
            }
/* Do NOT unroll: 8 iterations x 9 independent accumulator chains already
 * saturate the FMA units, and unrolling makes gcc spill (measured: 815 stack
 * references per exec vs 60 with the pragma, in the AVX-512 build). */
#pragma GCC unroll 1
            for (int j = 2; j <= 8; ++j) {
                BFLY_DECL(j);
                P0 += CGET(c, 0) * u; P1 += CGET(c, 1) * u; P2 += CGET(c, 2) * u;
                P3 += CGET(c, 3) * u; P4 += CGET(c, 4) * u;
                R1 += CGET(sn, 0) * w; R2 += CGET(sn, 1) * w;
                R3 += CGET(sn, 2) * w; R4 += CGET(sn, 3) * w;
            }
            Q0 = P0;
            X1 = P1 + R1; X2 = P2 + R2; X3 = P3 + R3; X4 = P4 + R4;
            Y1 = P1 - R1; Y2 = P2 - R2; Y3 = P3 - R3; Y4 = P4 - R4;
        }
        if (!tr) {
            VST(dst + 0 * db, Q0);
            VST(dst + 1 * db, X1);  VST(dst + 2 * db, X2);
            VST(dst + 3 * db, X3);  VST(dst + 4 * db, X4);
            VST(dst + 13 * db, Y4); VST(dst + 14 * db, Y3);
            VST(dst + 15 * db, Y2); VST(dst + 16 * db, Y1);
        } else {
#if WC == 4
            TILE(0, Q0, X1, X2, X3);
            TILE(1, X1, X2, X3, X4);
            TILE(13, Y4, Y3, Y2, Y1);
#else
            TILE(0, Q0, X1);
            TILE(2, X2, X3);
            TILE(3, X3, X4);
            TILE(13, Y4, Y3);
            TILE(15, Y2, Y1);
#endif
        }
    }
    { /* ---- block B: k = 5..8  ->  outputs m = 5..12 ---- */
        VT X5, X6, X7, X8, Y5, Y6, Y7, Y8;
        {
            VT P5, P6, P7, P8, R5, R6, R7, R8;
            VT v0 = VLD(src);
            P5 = v0; P6 = v0; P7 = v0; P8 = v0;
            {
                BFLY_B(1);
                P5 += CGET(c, 5) * u; P6 += CGET(c, 6) * u;
                P7 += CGET(c, 7) * u; P8 += CGET(c, 8) * u;
                R5 = CGET(sn, 4) * w; R6 = CGET(sn, 5) * w;
                R7 = CGET(sn, 6) * w; R8 = CGET(sn, 7) * w;
            }
#pragma GCC unroll 1
            for (int j = 2; j <= 8; ++j) {
                BFLY_B(j);
                P5 += CGET(c, 5) * u; P6 += CGET(c, 6) * u;
                P7 += CGET(c, 7) * u; P8 += CGET(c, 8) * u;
                R5 += CGET(sn, 4) * w; R6 += CGET(sn, 5) * w;
                R7 += CGET(sn, 6) * w; R8 += CGET(sn, 7) * w;
            }
            X5 = P5 + R5; X6 = P6 + R6; X7 = P7 + R7; X8 = P8 + R8;
            Y5 = P5 - R5; Y6 = P6 - R6; Y7 = P7 - R7; Y8 = P8 - R8;
        }
        if (!tr) {
            VST(dst + 5 * db, X5);  VST(dst + 6 * db, X6);
            VST(dst + 7 * db, X7);  VST(dst + 8 * db, X8);
            VST(dst + 9 * db, Y8);  VST(dst + 10 * db, Y7);
            VST(dst + 11 * db, Y6); VST(dst + 12 * db, Y5);
        } else {
#if WC == 4
            TILE(5, X5, X6, X7, X8);
            TILE(9, Y8, Y7, Y6, Y5);
#else
            TILE(5, X5, X6);
            TILE(7, X7, X8);
            TILE(9, Y8, Y7);
            TILE(11, Y6, Y5);
#endif
        }
    }
#undef TILE
#undef BFLY_DECL
#undef BFLY_B
}

/* -----------------------------------------------------------------------
 *  Nested kernel (round panel_r2): the same conjugate-folded transform with
 *  the COSINE side halved via the group structure of (Z/17)^x mod {+-1}.
 *  Borrowed from L17_winograd's round-1 module (see strategies/), re-derived
 *  for this entry's interleaved lanes-are-lines layout.
 *
 *  Index k and j by m,n = 0..7 through g = 3:  3^m = sig[m]*f[m] mod 17,
 *  f = {1,3,8,7,4,5,2,6}, sig = {+,+,-,-,-,+,-,-}.  Then exactly:
 *    cos(2pi f[m]f[n]/17) = c[(m+n) mod 8]                     (circulant)
 *    sin(2pi f[m]f[n]/17) = sig[m]sig[n](-1)^floor((m+n)/8) s[(m+n) mod 8]
 *  with c[r] = cos(2pi 3^r/17), s[r] = sin(2pi 3^r/17).  The cosine side is a
 *  length-8 CYCLIC correlation and splits with sign-only reductions
 *  (x^8-1 = (x^4-1)(x^4+1)) into a cyclic-4 + a negacyclic-4:
 *    U_m  = x_{f[m]} + x_{17-f[m]}            (= x_{IA[m]} + x_{IB[m]})
 *    P_m  = U_m + U_{m+4},  Q_m = U_m - U_{m+4}          m = 0..3
 *    A_n  = x0 + sum_m cp[(m+n)%4] P_m                   n = 0..3
 *    B_n  =      sum_m eps(m+n) cm[(m+n)%4] Q_m          eps = +,+,+,+,-,-,-
 *    C_n  = A_n + B_n,  C_{n+4} = A_n - B_n,  X_0 = x0 + sum_m P_m
 *  The sine side stays a dense negacyclic-8 (x^8+1 is irreducible; every
 *  split was counted by L17_winograd and loses):
 *    W_m  = -i*(x_{IA[m]} - x_{IB[m]})        IA[m] = 3^m mod 17, IB = 17-IA
 *    S_n  = sum_m nst[m][n] W_m,   nst[m][n] = (-1)^floor((m+n)/8) s[(m+n)%8]
 *    X_{f[n]} = C_n + sig[n] S_n,  X_{17-f[n]} = C_n - sig[n] S_n
 *  sig[m] on W is folded into the operand order of the subtraction (IA/IB),
 *  sig[n] into which output of the pair takes the +; both are free.  All
 *  coefficients stay REAL, so the layout, the broadcasts and the (absence of)
 *  cross-lane traffic are identical to the dense kernel.  Cost per chunk:
 *  96 FMA + 52 add/sub = 148 FP ops against the dense kernel's 168 (-12%),
 *  and the round-1 node measurement says this kernel is FMA-port-bound at
 *  1.045 FP ops/cycle, so the saving should land almost 1:1 in time.
 *    pc = 0: C_0..7 + X0 stay in registers across the S loop (peak ~19 live)
 *    pc = 1: they are parked in an L1 scratch (peak ~11 live); which one the
 *            register allocator prefers is measured, not guessed.
 * --------------------------------------------------------------------- */
static inline __attribute__((always_inline)) void
SUF(chunk17n)(const double *restrict src, long rs, double *restrict dst, long da,
              long db, const double *restrict cn, const double *restrict sn,
              double *restrict sc,
              VT K0, VT K1, VT K2, VT K3, VT K4, VT K5, VT K6, VT K7,
              int tr, int pc, int pin)
{
    static const long IA[8] = {1, 3, 9, 10, 13, 5, 15, 11};
    static const long IB[8] = {16, 14, 8, 7, 4, 12, 2, 6};

    VT C0, C1, C2, C3, C4, C5, C6, C7, X0v;
    { /* ---- cosine side: cyclic-4 (A) + negacyclic-4 (B) on P,Q ---- */
        VT x0 = VLD(src);
        VT A0 = x0, A1 = x0, A2 = x0, A3 = x0, X0a = x0;
        VT B0, B1, B2, B3;
        { /* m = 0 peeled so the B accumulators start with a multiply */
            VT a = VLD(src + 1 * rs), b = VLD(src + 16 * rs);
            VT a2 = VLD(src + 13 * rs), b2 = VLD(src + 4 * rs);
            VT u = a + b, u2 = a2 + b2;
            VT pp = u + u2, q = u - u2;
            X0a += pp;
            A0 += CGET(cn, 0) * pp; A1 += CGET(cn, 1) * pp;
            A2 += CGET(cn, 2) * pp; A3 += CGET(cn, 3) * pp;
            B0 = CGET(cn, 4) * q;  B1 = CGET(cn, 5) * q;
            B2 = CGET(cn, 6) * q;  B3 = CGET(cn, 7) * q;
        }
#pragma GCC unroll 1
        for (int m = 1; m < 4; ++m) {
            VT a = VLD(src + IA[m] * rs), b = VLD(src + IB[m] * rs);
            VT a2 = VLD(src + IA[m + 4] * rs), b2 = VLD(src + IB[m + 4] * rs);
            VT u = a + b, u2 = a2 + b2;
            VT pp = u + u2, q = u - u2;
            const double *r = cn + (size_t)m * 8 * VDW;
            X0a += pp;
            A0 += CGET(r, 0) * pp; A1 += CGET(r, 1) * pp;
            A2 += CGET(r, 2) * pp; A3 += CGET(r, 3) * pp;
            B0 += CGET(r, 4) * q;  B1 += CGET(r, 5) * q;
            B2 += CGET(r, 6) * q;  B3 += CGET(r, 7) * q;
        }
        C0 = A0 + B0; C1 = A1 + B1; C2 = A2 + B2; C3 = A3 + B3;
        C4 = A0 - B0; C5 = A1 - B1; C6 = A2 - B2; C7 = A3 - B3;
        X0v = X0a;
        if (pc) {
            VST(sc + 0 * VDW, C0); VST(sc + 1 * VDW, C1);
            VST(sc + 2 * VDW, C2); VST(sc + 3 * VDW, C3);
            VST(sc + 4 * VDW, C4); VST(sc + 5 * VDW, C5);
            VST(sc + 6 * VDW, C6); VST(sc + 7 * VDW, C7);
            VST(sc + 8 * VDW, X0v);
        }
    }

    VT S0, S1, S2, S3, S4, S5, S6, S7;
    if (pin) {
        /* ---- sine side, pinned: K_r = s[r] broadcast once per execute; the
         * negacyclic sign (-1)^floor((m+n)/8) is a compile-time vfnmadd, which
         * rounds identically to the table's baked-in negative constant, so
         * this path is bit-identical to the unpinned one.  Fully unrolled: no
         * memory operands left to spill over, and 64 coefficient loads per
         * chunk are gone. ---- */
        VT w = MULI(VLD(src + 1 * rs) - VLD(src + 16 * rs));           /* m=0 */
        S0 = K0 * w; S1 = K1 * w; S2 = K2 * w; S3 = K3 * w;
        S4 = K4 * w; S5 = K5 * w; S6 = K6 * w; S7 = K7 * w;
        w = MULI(VLD(src + 3 * rs) - VLD(src + 14 * rs));              /* m=1 */
        S0 += K1 * w; S1 += K2 * w; S2 += K3 * w; S3 += K4 * w;
        S4 += K5 * w; S5 += K6 * w; S6 += K7 * w; S7 -= K0 * w;
        w = MULI(VLD(src + 9 * rs) - VLD(src + 8 * rs));               /* m=2 */
        S0 += K2 * w; S1 += K3 * w; S2 += K4 * w; S3 += K5 * w;
        S4 += K6 * w; S5 += K7 * w; S6 -= K0 * w; S7 -= K1 * w;
        w = MULI(VLD(src + 10 * rs) - VLD(src + 7 * rs));              /* m=3 */
        S0 += K3 * w; S1 += K4 * w; S2 += K5 * w; S3 += K6 * w;
        S4 += K7 * w; S5 -= K0 * w; S6 -= K1 * w; S7 -= K2 * w;
        w = MULI(VLD(src + 13 * rs) - VLD(src + 4 * rs));              /* m=4 */
        S0 += K4 * w; S1 += K5 * w; S2 += K6 * w; S3 += K7 * w;
        S4 -= K0 * w; S5 -= K1 * w; S6 -= K2 * w; S7 -= K3 * w;
        w = MULI(VLD(src + 5 * rs) - VLD(src + 12 * rs));              /* m=5 */
        S0 += K5 * w; S1 += K6 * w; S2 += K7 * w; S3 -= K0 * w;
        S4 -= K1 * w; S5 -= K2 * w; S6 -= K3 * w; S7 -= K4 * w;
        w = MULI(VLD(src + 15 * rs) - VLD(src + 2 * rs));              /* m=6 */
        S0 += K6 * w; S1 += K7 * w; S2 -= K0 * w; S3 -= K1 * w;
        S4 -= K2 * w; S5 -= K3 * w; S6 -= K4 * w; S7 -= K5 * w;
        w = MULI(VLD(src + 11 * rs) - VLD(src + 6 * rs));              /* m=7 */
        S0 += K7 * w; S1 -= K0 * w; S2 -= K1 * w; S3 -= K2 * w;
        S4 -= K3 * w; S5 -= K4 * w; S6 -= K5 * w; S7 -= K6 * w;
    } else { /* ---- sine side: dense negacyclic-8, coefficients as memory
              * operands, sig[m] folded into IA/IB order ---- */
        VT w = MULI(VLD(src + 1 * rs) - VLD(src + 16 * rs));
        S0 = CGET(sn, 0) * w; S1 = CGET(sn, 1) * w;
        S2 = CGET(sn, 2) * w; S3 = CGET(sn, 3) * w;
        S4 = CGET(sn, 4) * w; S5 = CGET(sn, 5) * w;
        S6 = CGET(sn, 6) * w; S7 = CGET(sn, 7) * w;
#pragma GCC unroll 1
        for (int m = 1; m < 8; ++m) {
            w = MULI(VLD(src + IA[m] * rs) - VLD(src + IB[m] * rs));
            const double *r = sn + (size_t)m * 8 * VDW;
            S0 += CGET(r, 0) * w; S1 += CGET(r, 1) * w;
            S2 += CGET(r, 2) * w; S3 += CGET(r, 3) * w;
            S4 += CGET(r, 4) * w; S5 += CGET(r, 5) * w;
            S6 += CGET(r, 6) * w; S7 += CGET(r, 7) * w;
        }
    }

    if (pc) {
        C0 = VLD(sc + 0 * VDW); C1 = VLD(sc + 1 * VDW);
        C2 = VLD(sc + 2 * VDW); C3 = VLD(sc + 3 * VDW);
        C4 = VLD(sc + 4 * VDW); C5 = VLD(sc + 5 * VDW);
        C6 = VLD(sc + 6 * VDW); C7 = VLD(sc + 7 * VDW);
        X0v = VLD(sc + 8 * VDW);
    }

/* output slot table: slot f[n] = C_n + sig[n] S_n, slot 17-f[n] = the pair;
 * f = {1,3,8,7,4,5,2,6}, sig = {+,+,-,-,-,+,-,-}, checked against numpy */
#define E0 X0v
#define E1 (C0 + S0)
#define E2 (C6 - S6)
#define E3 (C1 + S1)
#define E4 (C4 - S4)
#define E5 (C5 + S5)
#define E6 (C7 - S7)
#define E7 (C3 - S3)
#define E8 (C2 - S2)
#define E9 (C2 + S2)
#define E10 (C3 + S3)
#define E11 (C7 + S7)
#define E12 (C5 - S5)
#define E13 (C4 + S4)
#define E14 (C1 - S1)
#define E15 (C6 + S6)
#define E16 (C0 - S0)

    if (!tr) {
        VST(dst + 0 * db, E0);   VST(dst + 1 * db, E1);
        VST(dst + 2 * db, E2);   VST(dst + 3 * db, E3);
        VST(dst + 4 * db, E4);   VST(dst + 5 * db, E5);
        VST(dst + 6 * db, E6);   VST(dst + 7 * db, E7);
        VST(dst + 8 * db, E8);   VST(dst + 9 * db, E9);
        VST(dst + 10 * db, E10); VST(dst + 11 * db, E11);
        VST(dst + 12 * db, E12); VST(dst + 13 * db, E13);
        VST(dst + 14 * db, E14); VST(dst + 15 * db, E15);
        VST(dst + 16 * db, E16);
    } else {
#if WC == 4
#  define TILEN(m0, e0, e1, e2, e3)                                            \
      do {                                                                     \
          VT xx[4], yy[4];                                                     \
          xx[0] = (e0); xx[1] = (e1); xx[2] = (e2); xx[3] = (e3);               \
          TTILE(xx, yy);                                                       \
          VST(dst + 0 * da + (m0) * 2, yy[0]);                                 \
          VST(dst + 1 * da + (m0) * 2, yy[1]);                                 \
          VST(dst + 2 * da + (m0) * 2, yy[2]);                                 \
          VST(dst + 3 * da + (m0) * 2, yy[3]);                                 \
      } while (0)
        TILEN(0, E0, E1, E2, E3);
        TILEN(4, E4, E5, E6, E7);
        TILEN(8, E8, E9, E10, E11);
        TILEN(12, E12, E13, E14, E15);
        TILEN(13, E13, E14, E15, E16);
#  undef TILEN
#else
#  define TILEN(m0, e0, e1)                                                    \
      do {                                                                     \
          VT xx[2], yy[2];                                                     \
          xx[0] = (e0); xx[1] = (e1);                                          \
          TTILE(xx, yy);                                                       \
          VST(dst + 0 * da + (m0) * 2, yy[0]);                                 \
          VST(dst + 1 * da + (m0) * 2, yy[1]);                                 \
      } while (0)
        TILEN(0, E0, E1);
        TILEN(2, E2, E3);
        TILEN(4, E4, E5);
        TILEN(6, E6, E7);
        TILEN(8, E8, E9);
        TILEN(10, E10, E11);
        TILEN(12, E12, E13);
        TILEN(14, E14, E15);
        TILEN(15, E15, E16);
#  undef TILEN
#endif
    }
#undef E0
#undef E1
#undef E2
#undef E3
#undef E4
#undef E5
#undef E6
#undef E7
#undef E8
#undef E9
#undef E10
#undef E11
#undef E12
#undef E13
#undef E14
#undef E15
#undef E16
}

/* chunk start offsets covering a 17-long index; the last one overlaps the
 * previous by WC-1 lanes and recomputes/rewrites them with identical values */
#if WC == 4
static const int SUF(off17)[5] = {0, 4, 8, 12, 13};
#  define NOFF17 5
#else
static const int SUF(off17)[9] = {0, 2, 4, 6, 8, 10, 12, 14, 15};
#  define NOFF17 9
#endif

#define PBROW (20 * 2) /* plane buffer row stride, doubles (17 padded to 20) */

/* ---- 17x17 complex plane transpose: dst[b][a] = src[a][b] --------------- */
static void SUF(trans17)(const double *restrict src, long srs, double *restrict dst,
                         long drs)
{
    for (int ta = 0; ta < NOFF17; ++ta) {
        long a0 = SUF(off17)[ta];
        for (int tb = 0; tb < NOFF17; ++tb) {
            long b0 = SUF(off17)[tb];
            VT xx[WC], yy[WC];
            for (int i = 0; i < WC; ++i) xx[i] = VLD(src + (a0 + i) * srs + b0 * 2);
            TTILE(xx, yy);
            for (int i = 0; i < WC; ++i) VST(dst + (b0 + i) * drs + a0 * 2, yy[i]);
        }
    }
}

/* ---- variant 1: the plane transposes are fused into the chunk stores ---- */
static void SUF(exec)(const fft3d_plan *restrict p, const double _Complex *restrict in,
                      double _Complex *restrict out)
{
    const double *restrict ct = CTAB(p);
    const double *restrict st = STAB(p);
    double *restrict pb = p->pb;
    double *restrict t1 = p->t1;
    const int nb = p->batch;

    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 4913);
        double *vout = (double *)(out + (size_t)b * 4913);

        for (int x = 0; x < 17; ++x) {
            const double *pin = vin + (long)x * 289 * 2;
            double *pt = t1 + (long)x * 289 * 2;
            /* Y: along y (row stride 34), lanes over z, store transposed */
            for (int t = 0; t < NOFF17; ++t) {
                long f0 = SUF(off17)[t];
                SUF(chunk17)(pin + 2 * f0, 34, pb + f0 * PBROW, PBROW, 2, ct, st, 1);
            }
            /* Z: along z (row stride PBROW), lanes over ky, store transposed */
            for (int t = 0; t < NOFF17; ++t) {
                long f0 = SUF(off17)[t];
                SUF(chunk17)(pb + 2 * f0, PBROW, pt + f0 * 34, 34, 2, ct, st, 1);
            }
        }
        /* X: along x (row stride 578), lanes over the 289 contiguous (ky,kz) */
        for (int i = 0; i < 289 / WC; ++i) {
            long f0 = (long)i * WC;
            SUF(chunk17)(t1 + 2 * f0, 578, vout + 2 * f0, 2, 578, ct, st, 0);
        }
#if (289 % WC) != 0
        SUF(chunk17)(t1 + 2 * (289 - WC), 578, vout + 2 * (289 - WC), 2, 578, ct, st, 0);
#endif
    }
}

/* ---- variant 2: every pass stores untransposed (lanes stay contiguous in
 * the destination) and the two unavoidable plane transposes are done by a
 * separate tight loop.  More instructions, but the chunk kernel then needs no
 * extra registers for the tile transpose, which is what the register allocator
 * chokes on. ---- */
static void SUF(exec2)(const fft3d_plan *restrict p, const double _Complex *restrict in,
                       double _Complex *restrict out)
{
    const double *restrict ct = CTAB(p);
    const double *restrict st = STAB(p);
    double *restrict pa = p->pb;
    double *restrict pbb = p->pb2;
    double *restrict t1 = p->t1;
    const int nb = p->batch;

    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 4913);
        double *vout = (double *)(out + (size_t)b * 4913);

        for (int x = 0; x < 17; ++x) {
            const double *pin = vin + (long)x * 289 * 2;
            double *pt = t1 + (long)x * 289 * 2;
            /* Y: along y, lanes over z, plain store -> pa[ky][z] */
            for (int t = 0; t < NOFF17; ++t) {
                long f0 = SUF(off17)[t];
                SUF(chunk17)(pin + 2 * f0, 34, pa + 2 * f0, 2, PBROW, ct, st, 0);
            }
            SUF(trans17)(pa, PBROW, pbb, PBROW); /* pbb[z][ky] */
            /* Z: along z, lanes over ky, plain store -> pa[kz][ky] */
            for (int t = 0; t < NOFF17; ++t) {
                long f0 = SUF(off17)[t];
                SUF(chunk17)(pbb + 2 * f0, PBROW, pa + 2 * f0, 2, PBROW, ct, st, 0);
            }
            SUF(trans17)(pa, PBROW, pt, 34); /* t1[x][ky][kz] */
        }
        for (int i = 0; i < 289 / WC; ++i) {
            long f0 = (long)i * WC;
            SUF(chunk17)(t1 + 2 * f0, 578, vout + 2 * f0, 2, 578, ct, st, 0);
        }
#if (289 % WC) != 0
        SUF(chunk17)(t1 + 2 * (289 - WC), 578, vout + 2 * (289 - WC), 2, 578, ct, st, 0);
#endif
    }
}

/* ---- variants 3 and 4: k-blocked kernel, transposes fused into the stores;
 * variant 4 additionally parks the butterfly results in L1 ---- */
#define L17_EXEC_KB(NAME, BST, NTC)                                            \
    static __attribute__((unused)) void                                        \
    NAME(const fft3d_plan *restrict p,                                         \
                     const double _Complex *restrict in,                       \
                     double _Complex *restrict out)                            \
    {                                                                          \
        const double *restrict ct = CTAB(p);                                   \
        const double *restrict st = STAB(p);                                   \
        double *restrict pb = p->pb;                                           \
        double *restrict t1 = p->t1;                                           \
        double *restrict sc = p->sc;                                           \
        const int nb = p->batch;                                               \
        for (int b = 0; b < nb; ++b) {                                         \
            const double *vin = (const double *)(in + (size_t)b * 4913);       \
            double *vout = (double *)(out + (size_t)b * 4913);                  \
            double *dstv = NTC ? p->t2 : vout;                                 \
            for (int x = 0; x < 17; ++x) {                                     \
                const double *pin = vin + (long)x * 289 * 2;                   \
                double *pt = t1 + (long)x * 289 * 2;                           \
                for (int t = 0; t < NOFF17; ++t) {                             \
                    long f0 = SUF(off17)[t];                                   \
                    SUF(chunk17b)(pin + 2 * f0, 34, pb + f0 * PBROW, PBROW, 2, \
                                  ct, st, sc, 1, BST);                          \
                }                                                              \
                for (int t = 0; t < NOFF17; ++t) {                             \
                    long f0 = SUF(off17)[t];                                   \
                    SUF(chunk17b)(pb + 2 * f0, PBROW, pt + f0 * 34, 34, 2,     \
                                  ct, st, sc, 1, BST);                          \
                }                                                              \
            }                                                                  \
            for (int i = 0; i < 289 / WC; ++i) {                               \
                long f0 = (long)i * WC;                                        \
                SUF(chunk17b)(t1 + 2 * f0, 578, dstv + 2 * f0, 2, 578,         \
                              ct, st, sc, 0, BST);                              \
            }                                                                  \
            if (289 % WC)                                                      \
                SUF(chunk17b)(t1 + 2 * (289 - WC), 578, dstv + 2 * (289 - WC), \
                              2, 578, ct, st, sc, 0, BST);                      \
            if (NTC) l17_ntcopy(vout, dstv, 9826);                             \
        }                                                                      \
    }

L17_EXEC_KB(SUF(exec3), 0, 0)
L17_EXEC_KB(SUF(exec4), 1, 0)
L17_EXEC_KB(SUF(exec5), 1, 1)
#undef L17_EXEC_KB

/* ---- variants 6..15: the nested (cyclic/negacyclic split) kernel ----
 *  PC   : park the C accumulators in an L1 scratch across the sine loop
 *  NTC  : stage the finished volume and stream it to out with NT stores
 *  PIN  : sine constants pinned in registers (see L17_PIN_ASM above)
 *  REORD: X-first pass order.  The default order (Y,Z per plane, then X) reads
 *         `in` sequentially but writes the whole `out` volume in one burst of
 *         73 chunks at the end; at large batch that write burst (RFO + write-
 *         back) serialises against the next volume's input reads.  X-first
 *         transforms x from `in` into t1, then finishes each kx plane with Y,Z
 *         and stores it straight into its 4.6 KiB slot of `out`, so the output
 *         writes are spread evenly across the volume's compute.  The axes
 *         commute exactly, but the rounding order differs, so REORD variants
 *         are within-tolerance-identical, not bit-identical, to the others
 *         (each plan pins one variant, so repeatability is unaffected). */
#define L17_EXEC_N(NAME, PC, NTC, PIN, REORD)                                  \
    static __attribute__((unused)) void                                        \
    NAME(const fft3d_plan *restrict p,                                         \
                     const double _Complex *restrict in,                       \
                     double _Complex *restrict out)                            \
    {                                                                          \
        const double *restrict cn = NTCTAB(p);                                 \
        const double *restrict sn = NTSTAB(p);                                 \
        double *restrict pb = p->pb;                                           \
        double *restrict t1 = p->t1;                                           \
        double *restrict sc = p->sc;                                           \
        const int nb = p->batch;                                               \
        VT K0 = CGET(sn, 0), K1 = CGET(sn, 1), K2 = CGET(sn, 2),               \
           K3 = CGET(sn, 3), K4 = CGET(sn, 4), K5 = CGET(sn, 5),               \
           K6 = CGET(sn, 6), K7 = CGET(sn, 7);                                 \
        if (PIN) L17_PIN_ASM();                                                \
        for (int b = 0; b < nb; ++b) {                                         \
            const double *vin = (const double *)(in + (size_t)b * 4913);       \
            double *vout = (double *)(out + (size_t)b * 4913);                  \
            double *dstv = NTC ? p->t2 : vout;                                 \
            if (REORD) { /* X first: in -> t1[kx][y][z] */                     \
                for (int i = 0; i < 289 / WC; ++i) {                           \
                    long f0 = (long)i * WC;                                    \
                    SUF(chunk17n)(vin + 2 * f0, 578, t1 + 2 * f0, 2, 578,      \
                                  cn, sn, sc, K0, K1, K2, K3, K4, K5, K6, K7,  \
                                  0, PC, PIN);                                 \
                }                                                              \
                if (289 % WC)                                                  \
                    SUF(chunk17n)(vin + 2 * (289 - WC), 578,                   \
                                  t1 + 2 * (289 - WC), 2, 578,                 \
                                  cn, sn, sc, K0, K1, K2, K3, K4, K5, K6, K7,  \
                                  0, PC, PIN);                                 \
            }                                                                  \
            for (int x = 0; x < 17; ++x) {                                     \
                const double *pin2 = (REORD ? (const double *)t1 : vin)        \
                                     + (long)x * 289 * 2;                      \
                double *pt = (REORD ? dstv : t1) + (long)x * 289 * 2;          \
                if (REORD && p->pf && b + 1 < nb) {                            \
                    /* cross-volume prefetch (L17_winograd r2): during the     \
                     * plane phase, whose sources are L2-resident scratch,     \
                     * pull the NEXT volume's input toward L2 so the X pass    \
                     * that opens volume b+1 does not stall on DRAM.  73 lines \
                     * per plane covers the 1229-line volume in 17 planes.     \
                     * Prefetches change no results; this is a tunable flag. */\
                    const char *nx = (const char *)(vin + 9826) + (long)x * 4672;\
                    for (int q3 = 0; q3 < 73; ++q3)                            \
                        __builtin_prefetch(nx + (long)q3 * 64, 0, 2);          \
                }                                                              \
                if (REORD && p->pw) { /* prefetchw this plane's out lines,    \
                     * half before each group, ~0.25 us ahead of the Z stores */\
                    const char *po = (const char *)pt;                         \
                    for (int q4 = 0; q4 < 37; ++q4)                            \
                        __builtin_prefetch(po + (long)q4 * 64, 1, 3);          \
                }                                                              \
                for (int t = 0; t < NOFF17; ++t) {                             \
                    long f0 = SUF(off17)[t];                                   \
                    SUF(chunk17n)(pin2 + 2 * f0, 34, pb + f0 * PBROW, PBROW, 2,\
                                  cn, sn, sc, K0, K1, K2, K3, K4, K5, K6, K7,  \
                                  1, PC, PIN);                                 \
                }                                                              \
                if (REORD && p->pw) {                                          \
                    const char *po = (const char *)pt;                         \
                    for (int q4 = 37; q4 < 73; ++q4)                           \
                        __builtin_prefetch(po + (long)q4 * 64, 1, 3);          \
                }                                                              \
                for (int t = 0; t < NOFF17; ++t) {                             \
                    long f0 = SUF(off17)[t];                                   \
                    SUF(chunk17n)(pb + 2 * f0, PBROW, pt + f0 * 34, 34, 2,     \
                                  cn, sn, sc, K0, K1, K2, K3, K4, K5, K6, K7,  \
                                  1, PC, PIN);                                 \
                }                                                              \
            }                                                                  \
            if (!REORD) { /* X last: t1 -> out (or the NT staging buffer) */   \
                for (int i = 0; i < 289 / WC; ++i) {                           \
                    long f0 = (long)i * WC;                                    \
                    SUF(chunk17n)(t1 + 2 * f0, 578, dstv + 2 * f0, 2, 578,     \
                                  cn, sn, sc, K0, K1, K2, K3, K4, K5, K6, K7,  \
                                  0, PC, PIN);                                 \
                }                                                              \
                if (289 % WC)                                                  \
                    SUF(chunk17n)(t1 + 2 * (289 - WC), 578,                    \
                                  dstv + 2 * (289 - WC), 2, 578,               \
                                  cn, sn, sc, K0, K1, K2, K3, K4, K5, K6, K7,  \
                                  0, PC, PIN);                                 \
            }                                                                  \
            if (NTC) l17_ntcopy(vout, dstv, 9826);                             \
        }                                                                      \
    }

L17_EXEC_N(SUF(exec6), 0, 0, 0, 0)
L17_EXEC_N(SUF(exec7), 1, 0, 0, 0)
L17_EXEC_N(SUF(exec8), 0, 1, 0, 0)
L17_EXEC_N(SUF(exec9), 1, 1, 0, 0)
L17_EXEC_N(SUF(exec10), 0, 0, 0, 1)
L17_EXEC_N(SUF(exec11), 1, 0, 0, 1)
L17_EXEC_N(SUF(exec12), 1, 0, 1, 0)
L17_EXEC_N(SUF(exec13), 1, 0, 1, 1)
L17_EXEC_N(SUF(exec14), 0, 0, 1, 0)
L17_EXEC_N(SUF(exec15), 0, 0, 1, 1)
L17_EXEC_N(SUF(exec16), 1, 1, 1, 0)
L17_EXEC_N(SUF(exec17), 1, 1, 1, 1)
#undef L17_EXEC_N

/* ---- variants 18..20: cross-volume software pipelining (round panel_r4) ----
 * X-first at batch has one remaining serial DRAM burst: the 73 X chunks that
 * open volume b+1 read its whole 78.6 KiB input back-to-back, with nothing to
 * hide the misses under.  These variants interleave that burst into volume b's
 * plane phase instead: t1 is double-buffered (t1/t1b), and while plane x of
 * volume b runs its Y,Z chunks (whose sources are L1/L2-resident scratch),
 * NX/17 = 4-5 X chunks of volume b+1 are executed between the planes, so the
 * input reads of every volume except the first overlap the compute of the
 * previous one.  Pure scheduling: every chunk computes exactly what it
 * computes in the non-pipelined X-first variant, on the same operands, in the
 * same per-value order, so the output should be bit-identical to exec13/15
 * (bit-class D) -- VERIFIED BY cmp, not assumed (r3's lesson: gcc may
 * contract differently at a new call site).  The pf prefetch flag is
 * meaningless here (the interleaved X chunks ARE the prefetch, and they do
 * the work at the same time) and is ignored.  NTC stages each finished kx
 * plane in t2 and streams it to out per plane, spreading the NT writes the
 * same way the plain stores are spread. */
#define L17_EXEC_P(NAME, PC, NTC)                                              \
    static __attribute__((unused)) void                                        \
    NAME(const fft3d_plan *restrict p,                                         \
                     const double _Complex *restrict in,                       \
                     double _Complex *restrict out)                            \
    {                                                                          \
        const double *restrict cn = NTCTAB(p);                                 \
        const double *restrict sn = NTSTAB(p);                                 \
        double *restrict pb = p->pb;                                           \
        double *restrict sc = p->sc;                                           \
        const int nb = p->batch;                                               \
        enum { NX = 289 / WC + ((289 % WC) ? 1 : 0) };                         \
        VT K0 = CGET(sn, 0), K1 = CGET(sn, 1), K2 = CGET(sn, 2),               \
           K3 = CGET(sn, 3), K4 = CGET(sn, 4), K5 = CGET(sn, 5),               \
           K6 = CGET(sn, 6), K7 = CGET(sn, 7);                                 \
        L17_PIN_ASM();                                                         \
        { /* prologue: volume 0's X pass, the only un-overlapped input read */ \
            const double *vin0 = (const double *)in;                           \
            for (int i = 0; i < NX; ++i) {                                     \
                long f0 = i < 289 / WC ? (long)i * WC : 289 - WC;              \
                SUF(chunk17n)(vin0 + 2 * f0, 578, p->t1 + 2 * f0, 2, 578,      \
                              cn, sn, sc, K0, K1, K2, K3, K4, K5, K6, K7,      \
                              0, PC, 1);                                       \
            }                                                                  \
        }                                                                      \
        for (int b = 0; b < nb; ++b) {                                         \
            double *vout = (double *)(out + (size_t)b * 4913);                 \
            double *dstv = NTC ? p->t2 : vout;                                 \
            const double *restrict cur = (b & 1) ? p->t1b : p->t1;             \
            double *restrict nxt = (b & 1) ? p->t1 : p->t1b;                   \
            const double *vinN = (const double *)(in + (size_t)(b + 1) * 4913);\
            const int hn = (b + 1 < nb);                                       \
            for (int x = 0; x < 17; ++x) {                                     \
                /* 4-5 X chunks of volume b+1, spread evenly over the planes   \
                 * and split across two insertion points (before Y, between    \
                 * Y and Z) so the DRAM read stream never bursts more than     \
                 * 2-3 chunks between compute-bound plane chunks */            \
                int i0 = 0, ih = 0, i1 = 0;                                    \
                if (hn) {                                                      \
                    i0 = (x * NX) / 17; i1 = ((x + 1) * NX) / 17;              \
                    ih = i0 + (i1 - i0) / 2;                                   \
                    for (int i = i0; i < ih; ++i) {                            \
                        long f0 = i < 289 / WC ? (long)i * WC : 289 - WC;      \
                        SUF(chunk17n)(vinN + 2 * f0, 578, nxt + 2 * f0, 2, 578,\
                                      cn, sn, sc, K0, K1, K2, K3, K4, K5, K6,  \
                                      K7, 0, PC, 1);                           \
                    }                                                          \
                }                                                              \
                const double *pin2 = cur + (long)x * 578;                      \
                double *pt = dstv + (long)x * 578;                             \
                for (int t = 0; t < NOFF17; ++t) {                             \
                    long f0 = SUF(off17)[t];                                   \
                    SUF(chunk17n)(pin2 + 2 * f0, 34, pb + f0 * PBROW, PBROW, 2,\
                                  cn, sn, sc, K0, K1, K2, K3, K4, K5, K6, K7,  \
                                  1, PC, 1);                                   \
                }                                                              \
                if (hn) {                                                      \
                    for (int i = ih; i < i1; ++i) {                            \
                        long f0 = i < 289 / WC ? (long)i * WC : 289 - WC;      \
                        SUF(chunk17n)(vinN + 2 * f0, 578, nxt + 2 * f0, 2, 578,\
                                      cn, sn, sc, K0, K1, K2, K3, K4, K5, K6,  \
                                      K7, 0, PC, 1);                           \
                    }                                                          \
                }                                                              \
                for (int t = 0; t < NOFF17; ++t) {                             \
                    long f0 = SUF(off17)[t];                                   \
                    SUF(chunk17n)(pb + 2 * f0, PBROW, pt + f0 * 34, 34, 2,     \
                                  cn, sn, sc, K0, K1, K2, K3, K4, K5, K6, K7,  \
                                  1, PC, 1);                                   \
                }                                                              \
                if (NTC) l17_ntcopy(vout + (long)x * 578,                      \
                                    dstv + (long)x * 578, 578);                \
            }                                                                  \
        }                                                                      \
    }

L17_EXEC_P(SUF(exec18), 0, 0)
L17_EXEC_P(SUF(exec19), 1, 0)
L17_EXEC_P(SUF(exec20), 1, 1)
#undef L17_EXEC_P

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
#undef L17_PIN_ASM
#undef CGET
#undef CSTEP_C
#undef CSTEP_S
#undef CTAB
#undef STAB
#undef NTCTAB
#undef NTSTAB
#undef TTILE
#undef NOFF17
#undef PBROW

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

/* Streaming copy of one finished volume into the caller's buffer.  Writing
 * `out` directly costs a read-for-ownership of every line it touches, which at
 * large batch is a third of all DRAM traffic; non-temporal stores skip it.  It
 * cannot be done in the pass-X store itself: out's row stride is 289 complex =
 * 4624 B = 16 (mod 64), so only 5 of the 17 rows a chunk writes are 64-byte
 * aligned.  Hence the volume is finished in an aligned staging buffer and
 * streamed out here.  Selected by measurement, and only offered at batch >= 64. */
#if defined(__AVX512F__) || defined(__AVX__) || defined(__SSE2__)
#  include <immintrin.h>
static void l17_ntcopy(double *restrict dst, const double *restrict src, size_t nd)
{
    size_t i = 0;
    /* dst is 16-byte aligned (a volume is 4913 complex = 78608 B); step up to the
     * widest alignment with 16-byte non-temporal pairs, then stream. */
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
static void l17_ntcopy(double *restrict dst, const double *restrict src, size_t nd)
{
    memcpy(dst, src, nd * sizeof *dst);
}
#endif

struct fft3d_plan {
    int L, batch;
    int pf; /* cross-volume input prefetch in the X-first plane phase (A/B'd) */
    int pw; /* write-intent prefetch (prefetchw) of the out plane before its
             * Z group stores it -- hides the RFO instead of avoiding it (NT).
             * Adopted from L8_fusedaxes r5 (pfw) / L36_pfa r5 (pf=2), the
             * only store-side mechanism the node has ever selected.  A/B'd. */
    void (*exec)(const struct fft3d_plan *, const double _Complex *, double _Complex *);
    double *ctab8; /* 8*9 cosines, j-major, each splatted 8x (zmm) */
    double *stab8; /* 8*8 sines,   j-major, each splatted 8x (zmm) */
    double *ctab4; /* the same, splatted 4x (ymm)                  */
    double *stab4;
    double *ntc8; /* nested kernel: 4 rows x {cp[(m+n)%4], eps*cm[(m+n)%4]} */
    double *nts8; /* nested kernel: 8 rows x nst[m][n], splatted 8x        */
    double *ntc4; /* the same, splatted 4x                                 */
    double *nts4;
    double *sc;  /* 16 vectors: the butterfly results, for the L1-butterfly kernel */
    double *pb;  /* 17 x 20 complex plane buffer (pass Y -> pass Z)               */
    double *pb2; /* second plane buffer, split-transpose variant only             */
    double *t1;  /* one 17^3 complex volume: pass Z -> pass X                     */
    double *t1b; /* second volume buffer: t1/t1b ping-pong for the pipelined
                  * variants (volume b+1's X output while volume b drains)       */
    double *t2;  /* one 17^3 complex volume: pass X -> NT copy into out           */
    void *block;
    double _Complex *ti, *to; /* transient buffers used only by the plan-time tuner */
    size_t tn;
};

/* ---- instantiate the kernel: {512-bit, 256-bit} x {staged, in-register} ---- */
/* A quoted #include is searched in the includer's own directory first, so the
 * first form works from any working directory; the second covers a build that
 * passes the file by a path from the parent directory with -I there. */
#if defined(__has_include)
#  if __has_include("L17_matrixsimd.c")
#    define L17_SELF "L17_matrixsimd.c"
#  elif __has_include("impl/L17_matrixsimd.c")
#    define L17_SELF "impl/L17_matrixsimd.c"
#  endif
#else
#  define L17_SELF "L17_matrixsimd.c"
#endif

#ifdef L17_SELF
#  define L17_TEMPLATE_PASS 1
#  define WC 4
#  define SUF(x) x##_w4
#  include L17_SELF
#  undef SUF
#  undef WC
#  undef L17_TEMPLATE_PASS

#  define L17_TEMPLATE_PASS 2
#  define WC 2
#  define SUF(x) x##_w2
#  include L17_SELF
#  undef SUF
#  undef WC
#  undef L17_TEMPLATE_PASS
#  define L17_HAVE_ALL 1
#else
#  error "L17_matrixsimd.c must be able to #include itself"
#endif

/* =============================================================================
 * ROUND panel_r5: mixed-width execs -- 512-bit chunk body, 256-bit (ymm) tail.
 * Adopted from L17_rader round panel_r4 ("512t"), with attribution; see the
 * header block for the port arithmetic.  These call BOTH kernel instantiations,
 * so they live here, after the two template passes.  The ymm tail runs the same
 * pinned code path with its own (unpinned) K set; whether the recomputed
 * overlap lines and the tail lines round bit-identically to the pure-zmm
 * variants is VERIFIED BY cmp on full outputs, never assumed (r3's lesson).
 * =============================================================================
 */
enum { L17_PBROWM = 40 }; /* == the template's PBROW: 17 rows padded to 20 complex */

#if defined(__AVX512F__)
#  define L17_PIN_W4M() __asm__("" : "+v"(K0), "+v"(K1), "+v"(K2), "+v"(K3),   \
                                     "+v"(K4), "+v"(K5), "+v"(K6), "+v"(K7))
#else
#  define L17_PIN_W4M() do { } while (0)
#endif

/* K = the 8 sine constants splatted to zmm (row m=0 of nts8 is s[0..7]);
 * Q = the same constants splatted to ymm for the tail kernel.  Only K is
 * asm-pinned: pinning both sets would hold 16 of 32 registers across the
 * whole execute and make the zmm kernel spill. */
#define L17_MIX_DECLS                                                          \
    const double *restrict cn8 = p->ntc8, *restrict sn8 = p->nts8;             \
    const double *restrict cn4 = p->ntc4, *restrict sn4 = p->nts4;             \
    double *restrict pb = p->pb;                                               \
    double *restrict sc = p->sc;                                               \
    const int nb = p->batch;                                                   \
    vd_w4 K0 = *(const vd_w4 *)(sn8 + 0),  K1 = *(const vd_w4 *)(sn8 + 8),     \
          K2 = *(const vd_w4 *)(sn8 + 16), K3 = *(const vd_w4 *)(sn8 + 24),    \
          K4 = *(const vd_w4 *)(sn8 + 32), K5 = *(const vd_w4 *)(sn8 + 40),    \
          K6 = *(const vd_w4 *)(sn8 + 48), K7 = *(const vd_w4 *)(sn8 + 56);    \
    vd_w2 Q0 = *(const vd_w2 *)(sn4 + 0),  Q1 = *(const vd_w2 *)(sn4 + 4),     \
          Q2 = *(const vd_w2 *)(sn4 + 8),  Q3 = *(const vd_w2 *)(sn4 + 12),    \
          Q4 = *(const vd_w2 *)(sn4 + 16), Q5 = *(const vd_w2 *)(sn4 + 20),    \
          Q6 = *(const vd_w2 *)(sn4 + 24), Q7 = *(const vd_w2 *)(sn4 + 28);    \
    L17_PIN_W4M()

/* One 17-long free index, transposing store (tr=1): 4 zmm chunks at offsets
 * 0,4,8,12 and one ymm tail at 15 (lines 15,16; line 15 recomputed).  The
 * asm-opaque bound keeps the zmm loop rolled -- gcc 11 ignores `#pragma GCC
 * unroll 1` around an always_inline callee (L17_rader r4) and 4 inlined
 * copies would bloat L1i for nothing. */
#define L17_MIX_GROUP(SRCB, RS, DSTB, DA, PC)                                  \
    do {                                                                       \
        int nt_ = 4;                                                           \
        __asm__("" : "+r"(nt_));                                               \
        for (int t_ = 0; t_ < nt_; ++t_) {                                     \
            long f0_ = 4L * t_;                                                \
            chunk17n_w4((SRCB) + 2 * f0_, (RS), (DSTB) + f0_ * (DA), (DA), 2,  \
                        cn8, sn8, sc, K0, K1, K2, K3, K4, K5, K6, K7,          \
                        1, (PC), 1);                                           \
        }                                                                      \
        chunk17n_w2((SRCB) + 2 * 15, (RS), (DSTB) + 15 * (DA), (DA), 2,        \
                    cn4, sn4, sc, Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, 1, 0, 1);    \
    } while (0)

/* The X pass, mixed: 72 zmm chunks cover lines 0..287 of the 289-long free
 * index, one ymm tail at 287 covers 287,288 (287 recomputed). */
#define L17_MIX_XPASS(SRCB, DSTB, PC)                                          \
    do {                                                                       \
        for (int i_ = 0; i_ < 72; ++i_) {                                      \
            long f0_ = 4L * i_;                                                \
            chunk17n_w4((SRCB) + 2 * f0_, 578, (DSTB) + 2 * f0_, 2, 578,       \
                        cn8, sn8, sc, K0, K1, K2, K3, K4, K5, K6, K7,          \
                        0, (PC), 1);                                           \
        }                                                                      \
        chunk17n_w2((SRCB) + 2 * 287, 578, (DSTB) + 2 * 287, 2, 578,           \
                    cn4, sn4, sc, Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, 0, 0, 1);    \
    } while (0)

/* one X chunk-slot for the pipelined exec: slots 0..71 are zmm, 72 the tail */
#define L17_MIX_XSLOT(SRC, DST, I)                                             \
    do {                                                                       \
        if ((I) < 72) {                                                        \
            long f0_ = 4L * (I);                                               \
            chunk17n_w4((SRC) + 2 * f0_, 578, (DST) + 2 * f0_, 2, 578,         \
                        cn8, sn8, sc, K0, K1, K2, K3, K4, K5, K6, K7, 0, 0, 1);\
        } else {                                                               \
            chunk17n_w2((SRC) + 2 * 287, 578, (DST) + 2 * 287, 2, 578,         \
                        cn4, sn4, sc, Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, 0, 0, 1);\
        }                                                                      \
    } while (0)

/* mirror of exec14 (PC=0) / exec12 (PC=1): X-last, pinned sines, mixed tail */
#define L17_EXEC_MXL(NAME, PC)                                                 \
    static __attribute__((unused)) void                                        \
    NAME(const fft3d_plan *restrict p, const double _Complex *restrict in,     \
         double _Complex *restrict out)                                        \
    {                                                                          \
        L17_MIX_DECLS;                                                         \
        double *restrict t1 = p->t1;                                           \
        for (int b = 0; b < nb; ++b) {                                         \
            const double *vin = (const double *)(in + (size_t)b * 4913);       \
            double *vout = (double *)(out + (size_t)b * 4913);                 \
            for (int x = 0; x < 17; ++x) {                                     \
                const double *pin2 = vin + (long)x * 578;                      \
                double *pt = t1 + (long)x * 578;                               \
                L17_MIX_GROUP(pin2, 34, pb, L17_PBROWM, PC);                   \
                L17_MIX_GROUP(pb, L17_PBROWM, pt, 34, PC);                     \
            }                                                                  \
            L17_MIX_XPASS(t1, vout, PC);                                       \
        }                                                                      \
    }

/* mirror of exec15: X-first, pinned sines, mixed tail (batch regime) */
#define L17_EXEC_MXF(NAME, PC)                                                 \
    static __attribute__((unused)) void                                        \
    NAME(const fft3d_plan *restrict p, const double _Complex *restrict in,     \
         double _Complex *restrict out)                                        \
    {                                                                          \
        L17_MIX_DECLS;                                                         \
        double *restrict t1 = p->t1;                                           \
        for (int b = 0; b < nb; ++b) {                                         \
            const double *vin = (const double *)(in + (size_t)b * 4913);       \
            double *vout = (double *)(out + (size_t)b * 4913);                 \
            L17_MIX_XPASS(vin, t1, PC);                                        \
            for (int x = 0; x < 17; ++x) {                                     \
                const double *pin2 = t1 + (long)x * 578;                       \
                double *pt = vout + (long)x * 578;                             \
                if (p->pf && b + 1 < nb) {                                     \
                    const char *nx = (const char *)(vin + 9826) + (long)x * 4672; \
                    for (int q3 = 0; q3 < 73; ++q3)                            \
                        __builtin_prefetch(nx + (long)q3 * 64, 0, 2);          \
                }                                                              \
                if (p->pw) { /* prefetchw this plane's out lines, half before  \
                              * each group, ~0.25 us ahead of the Z stores */  \
                    const char *po = (const char *)pt;                         \
                    for (int q4 = 0; q4 < 37; ++q4)                            \
                        __builtin_prefetch(po + (long)q4 * 64, 1, 3);          \
                }                                                              \
                L17_MIX_GROUP(pin2, 34, pb, L17_PBROWM, PC);                   \
                if (p->pw) {                                                   \
                    const char *po = (const char *)pt;                         \
                    for (int q4 = 37; q4 < 73; ++q4)                           \
                        __builtin_prefetch(po + (long)q4 * 64, 1, 3);          \
                }                                                              \
                L17_MIX_GROUP(pb, L17_PBROWM, pt, 34, PC);                     \
            }                                                                  \
        }                                                                      \
    }

L17_EXEC_MXL(l17_execm_xl, 0)
L17_EXEC_MXL(l17_execm_xlp, 1)
L17_EXEC_MXF(l17_execm_xf, 0)

/* mirror of exec18: X-first, cross-volume pipelined, mixed tail */
static __attribute__((unused)) void
l17_execm_pipe(const fft3d_plan *restrict p, const double _Complex *restrict in,
               double _Complex *restrict out)
{
    L17_MIX_DECLS;
    enum { NXM = 73 };
    { /* prologue: volume 0's X pass, the only un-overlapped input read */
        const double *vin0 = (const double *)in;
        for (int i = 0; i < NXM; ++i) L17_MIX_XSLOT(vin0, p->t1, i);
    }
    for (int b = 0; b < nb; ++b) {
        double *vout = (double *)(out + (size_t)b * 4913);
        const double *restrict cur = (b & 1) ? p->t1b : p->t1;
        double *restrict nxt = (b & 1) ? p->t1 : p->t1b;
        const double *vinN = (const double *)(in + (size_t)(b + 1) * 4913);
        const int hn = (b + 1 < nb);
        for (int x = 0; x < 17; ++x) {
            int i0 = 0, ih = 0, i1 = 0;
            if (hn) {
                i0 = (x * NXM) / 17;
                i1 = ((x + 1) * NXM) / 17;
                ih = i0 + (i1 - i0) / 2;
                for (int i = i0; i < ih; ++i) L17_MIX_XSLOT(vinN, nxt, i);
            }
            const double *pin2 = cur + (long)x * 578;
            double *pt = vout + (long)x * 578;
            if (p->pw) {
                const char *po = (const char *)pt;
                for (int q4 = 0; q4 < 37; ++q4)
                    __builtin_prefetch(po + (long)q4 * 64, 1, 3);
            }
            L17_MIX_GROUP(pin2, 34, pb, L17_PBROWM, 0);
            if (hn)
                for (int i = ih; i < i1; ++i) L17_MIX_XSLOT(vinN, nxt, i);
            if (p->pw) {
                const char *po = (const char *)pt;
                for (int q4 = 37; q4 < 73; ++q4)
                    __builtin_prefetch(po + (long)q4 * 64, 1, 3);
            }
            L17_MIX_GROUP(pb, L17_PBROWM, pt, 34, 0);
        }
    }
}

/* -----------------------------------------------------------------------
 * ROUND panel_r6: deferred-Z plane schedule (mixed tail, both pass orders).
 * The r5 VERDICT quantified B=1 at 44.0k cycles against the 33.4k-cycle
 * mixed-shape port floor -- ~44 cycles/chunk of non-FP time -- and all three
 * L=17 records name the serialized per-plane traffic as the suspect.  This
 * entry's version of that serialization is the Y->Z junction: the Z group's
 * loads DEPEND on the Y group's stores to the same plane buffer (store->load
 * forwarding across a whole 5.4 KiB plane), once per plane, plus the group
 * tail latency drains at every group boundary with no independent work
 * behind them.  Fix, in the spirit of L17_winograd's d8 (defer the dependent
 * group one slot) but at group granularity with a double-buffered plane
 * buffer: run Y(x+1) -> pbB between Y(x) -> pbA and Z(x) <- pbA, so every
 * group junction has a full independent group (~5 chunks, >= 700 cycles on
 * the node) between a store group and the load group that depends on it.
 * Order: Y0; Y1 Z0; Y2 Z1; ... Y16 Z15; Z16.  Pure scheduling: every chunk
 * computes the same values from the same operands in the same order, so the
 * output must be bit-identical to the non-deferred variant of the same pass
 * order -- VERIFIED BY cmp on full outputs, never assumed (r3's lesson).
 * Costs one extra 5.4 KiB plane buffer (pb2, already allocated) of L1.
 * --------------------------------------------------------------------- */
static __attribute__((unused)) void
l17_execm_xld(const fft3d_plan *restrict p, const double _Complex *restrict in,
              double _Complex *restrict out)
{
    L17_MIX_DECLS;
    double *restrict pb2 = p->pb2;
    double *restrict t1 = p->t1;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 4913);
        double *vout = (double *)(out + (size_t)b * 4913);
        L17_MIX_GROUP(vin, 34, pb, L17_PBROWM, 0); /* Y(0) -> pb */
        for (int x = 0; x < 17; ++x) {
            double *pba = (x & 1) ? pb2 : pb; /* holds Y(x)'s plane */
            double *pbn = (x & 1) ? pb : pb2; /* Y(x+1)'s target    */
            if (x + 1 < 17)
                L17_MIX_GROUP(vin + (long)(x + 1) * 578, 34, pbn, L17_PBROWM, 0);
            L17_MIX_GROUP(pba, L17_PBROWM, t1 + (long)x * 578, 34, 0);
        }
        L17_MIX_XPASS(t1, vout, 0);
    }
}

/* X-first + deferred-Z: the batch-regime twin, with the pf/pw hooks of
 * l17_execm_xf (prefetches change no bits, so it stays in bit class D). */
static __attribute__((unused)) void
l17_execm_xfd(const fft3d_plan *restrict p, const double _Complex *restrict in,
              double _Complex *restrict out)
{
    L17_MIX_DECLS;
    double *restrict pb2 = p->pb2;
    double *restrict t1 = p->t1;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 4913);
        double *vout = (double *)(out + (size_t)b * 4913);
        L17_MIX_XPASS(vin, t1, 0);
        L17_MIX_GROUP(t1, 34, pb, L17_PBROWM, 0); /* Y(0) */
        for (int x = 0; x < 17; ++x) {
            double *pba = (x & 1) ? pb2 : pb;
            double *pbn = (x & 1) ? pb : pb2;
            double *pt = vout + (long)x * 578;
            if (p->pf && b + 1 < nb) {
                const char *nx = (const char *)(vin + 9826) + (long)x * 4672;
                for (int q3 = 0; q3 < 73; ++q3)
                    __builtin_prefetch(nx + (long)q3 * 64, 0, 2);
            }
            if (p->pw) {
                const char *po = (const char *)pt;
                for (int q4 = 0; q4 < 37; ++q4)
                    __builtin_prefetch(po + (long)q4 * 64, 1, 3);
            }
            if (x + 1 < 17)
                L17_MIX_GROUP(t1 + (long)(x + 1) * 578, 34, pbn, L17_PBROWM, 0);
            if (p->pw) {
                const char *po = (const char *)pt;
                for (int q4 = 37; q4 < 73; ++q4)
                    __builtin_prefetch(po + (long)q4 * 64, 1, 3);
            }
            L17_MIX_GROUP(pba, L17_PBROWM, pt, 34, 0);
        }
    }
}

/* -----------------------------------------------------------------------
 * Sustained-clock probe (the monitor's panel_r4 ask for L=17): four
 * independent 4-cycle-latency FMA chains issue exactly 1 FMA/cycle --
 * dense enough in "heavy" ops to hold the width's licence clock, and
 * latency-bound rather than unit-count-bound on 1- and 2-FMA-unit parts
 * alike, so cycles = 4*N and clk = 4*N / elapsed.  FMA latency is 4 on
 * SKX/CLX/ICX/SPR (5 on Haswell, where the report is 25% low, but that
 * machine is not scored).  ~1 ms per rep, best of 3, the first rep
 * absorbing the licence transition.
 * --------------------------------------------------------------------- */
static volatile double l17_clk_sink;
static double l17_now(void);
#define L17_CLK_PROBE(FN, VTYPE, ...)                                          \
    static double FN(void)                                                     \
    {                                                                          \
        const long N = 500000;                                                 \
        double best = 1e30;                                                    \
        VTYPE b = {__VA_ARGS__}, c = {__VA_ARGS__};                            \
        VTYPE a0 = {__VA_ARGS__}, a1 = a0, a2 = a0, a3 = a0;                   \
        b = b * 1e-15 + b;  /* 1 + 1e-15: keeps the chains near 1.0 */         \
        c = c * 1e-300;                                                        \
        for (int rep = 0; rep < 3; ++rep) {                                    \
            double t0 = l17_now();                                             \
            for (long i = 0; i < N; ++i) {                                     \
                a0 = a0 * b + c; a1 = a1 * b + c;                              \
                a2 = a2 * b + c; a3 = a3 * b + c;                              \
            }                                                                  \
            double dt = l17_now() - t0;                                        \
            if (dt < best) best = dt;                                          \
        }                                                                      \
        l17_clk_sink = a0[0] + a1[0] + a2[0] + a3[0];                          \
        return 4.0 * (double)N / best;                                         \
    }
L17_CLK_PROBE(l17_clk512, vd_w4, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
L17_CLK_PROBE(l17_clk256, vd_w2, 1.0, 1.0, 1.0, 1.0)
#undef L17_CLK_PROBE

/* DENSE 256-bit probe (round panel_r6): 8 independent latency-4 chains issue
 * 2 FMA/cycle -- both 256-bit FMA ports saturated.  The r5 VERDICT (section 5)
 * left clk256 unsettled: my 4-chain probe (1 FMA/cycle) read 3.89 GHz on the
 * node while L17_winograd's saturating probe read 2.89, and the plausible
 * discriminator is DENSITY (whether the chain is heavy enough to engage the
 * AVX2 licence) -- but nobody had run both designs in one process.  This is
 * that experiment: same process, same method, only the chain count differs.
 * cycles = 4*N/2 per rep since two chains retire per cycle.  Best of 5 reps,
 * the first absorbing any licence transition. */
static double l17_clk256d(void)
{
    const long N = 500000;
    double best = 1e30;
    vd_w2 b = {1.0, 1.0, 1.0, 1.0}, c = b;
    vd_w2 a0 = b, a1 = b, a2 = b, a3 = b, a4 = b, a5 = b, a6 = b, a7 = b;
    b = b * 1e-15 + b;
    c = c * 1e-300;
    for (int rep = 0; rep < 5; ++rep) {
        double t0 = l17_now();
        for (long i = 0; i < N; ++i) {
            a0 = a0 * b + c; a1 = a1 * b + c; a2 = a2 * b + c; a3 = a3 * b + c;
            a4 = a4 * b + c; a5 = a5 * b + c; a6 = a6 * b + c; a7 = a7 * b + c;
        }
        double dt = l17_now() - t0;
        if (dt < best) best = dt;
    }
    l17_clk_sink = a0[0] + a1[0] + a2[0] + a3[0] + a4[0] + a5[0] + a6[0] + a7[0];
    return 4.0 * (double)N / best;
}

static void l17_tune_free(fft3d_plan *p)
{
    free(p->ti);
    free(p->to);
    p->ti = NULL;
    p->to = NULL;
    p->tn = 0;
}

/* Scratch input/output for the plan-time tuning runs.  Deterministic pseudo
 * random data: the tuner only needs realistic magnitudes, not the real input. */
static int l17_tune_alloc(fft3d_plan *p, int nv)
{
    size_t n = (size_t)nv * 4913;
    if (p->tn >= n) return 1;
    l17_tune_free(p);
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

const char *fft3d_name(void) { return "L17_matrixsimd"; }

static const char *g_desc =
    "dense 17x17 DFT matrix per axis, conjugate-pair folded, SIMD across lines";

const char *fft3d_description(void) { return g_desc; }

int fft3d_supports(int L) { return L == 17; }

/* Set L17_VERBOSE=1 in the environment to have the plan-time tuner print what it
 * measured to stderr.  Silent otherwise, so a graded run says nothing. */
static int l17_verbose(void)
{
    const char *e = getenv("L17_VERBOSE");
    return e && *e && *e != '0';
}

static double l17_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* How many volumes the streaming-regime tuner should use: enough that
 * in + out together exceed ~2.5x this machine's L3, so the tuner sees the
 * DRAM-bound regime it is choosing for.  (Adopted from L36_mixedradix's
 * machine-relative tuner arena, with attribution: a fixed 384 volumes =
 * 60 MB streams on the node's 22 MB L3 but sits INSIDE wallaby's 60 MB L3,
 * where it picked plain stores at B=2048 while the driver's steady state
 * had pipelined+NT 13% faster.)  Clamped to [384, 1024]: the floor keeps
 * the node's behaviour identical to r3, the ceiling bounds plan time. */
static int l17_tune_nv(int batch)
{
    long l3 = -1;
#ifdef _SC_LEVEL3_CACHE_SIZE
    l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
    int cap = 384;
    if (l3 > 0) {
        double nv = 2.5 * (double)l3 / (2.0 * 4913.0 * 16.0);
        cap = nv < 384.0 ? 384 : (nv > 1024.0 ? 1024 : (int)nv);
    }
    return batch < cap ? batch : cap;
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != 17 || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;

    const size_t nc = 8 * 9, ns = 8 * 8;
    const size_t nnc = 4 * 8, nns = 8 * 8; /* nested kernel coefficient rows */
    const size_t nsc = 17 * 8;
    const size_t npb = 17 * 20 * 2;
    const size_t nt1 = 4913 * 2;
    const size_t ntot = 12 * nc + 12 * ns + 12 * nnc + 12 * nns + nsc +
                        2 * npb + 3 * nt1 + 1024;
    void *blk = NULL;
    if (posix_memalign(&blk, 64, ntot * sizeof(double)) != 0 || !blk) {
        free(p);
        return NULL;
    }
    memset(blk, 0, ntot * sizeof(double));
    p->block = blk;

#define L17_ALIGN64(q) ((double *)(((uintptr_t)(q) + 63u) & ~(uintptr_t)63u))
    double *q = (double *)blk;
    p->ctab8 = q; q += 8 * nc;
    p->stab8 = q; q += 8 * ns;
    q = L17_ALIGN64(q);
    p->ctab4 = q; q += 4 * nc;
    p->stab4 = q; q += 4 * ns;
    q = L17_ALIGN64(q);
    p->ntc8 = q; q += 8 * nnc;
    p->nts8 = q; q += 8 * nns;
    p->ntc4 = q; q += 4 * nnc;
    p->nts4 = q; q += 4 * nns;
    q = L17_ALIGN64(q);
    p->sc = q; q += nsc;
    q = L17_ALIGN64(q);
    p->pb = q; q += npb;
    q = L17_ALIGN64(q);
    p->pb2 = q; q += npb;
    q = L17_ALIGN64(q);
    p->t1 = q; q += nt1;
    q = L17_ALIGN64(q);
    p->t1b = q; q += nt1;
    q = L17_ALIGN64(q);
    p->t2 = q;
#undef L17_ALIGN64

    /* long double trig: the table itself is then good to ~1e-19 */
    const long double twopi = 6.283185307179586476925286766559005768394L;
    for (int j = 1; j <= 8; ++j) {
        for (int k = 0; k <= 8; ++k) {
            int m = (k * j) % 17;
            double c = (double)cosl(twopi * (long double)m / 17.0L);
            for (int t = 0; t < 8; ++t) p->ctab8[((j - 1) * 9 + k) * 8 + t] = c;
            for (int t = 0; t < 4; ++t) p->ctab4[((j - 1) * 9 + k) * 4 + t] = c;
        }
        for (int k = 1; k <= 8; ++k) {
            int m = (k * j) % 17;
            double sv = (double)sinl(twopi * (long double)m / 17.0L);
            for (int t = 0; t < 8; ++t) p->stab8[((j - 1) * 8 + (k - 1)) * 8 + t] = sv;
            for (int t = 0; t < 4; ++t) p->stab4[((j - 1) * 8 + (k - 1)) * 4 + t] = sv;
        }
    }

    /* nested-kernel tables: c[r] = cos(2pi 3^r/17), s[r] = sin(2pi 3^r/17);
     * cp/cm are the half-sums/half-differences of the cyclic-8 kernel, and
     * every sign (eps of the negacyclic-4, (-1)^floor((m+n)/8) of the
     * negacyclic-8) is baked into the table so the kernel sees plain reals. */
    {
        long double c8v[8], s8v[8], cp[4], cm[4];
        int p3 = 1;
        for (int r = 0; r < 8; ++r) {
            c8v[r] = cosl(twopi * (long double)p3 / 17.0L);
            s8v[r] = sinl(twopi * (long double)p3 / 17.0L);
            p3 = (p3 * 3) % 17;
        }
        for (int t = 0; t < 4; ++t) {
            cp[t] = (c8v[t] + c8v[t + 4]) * 0.5L;
            cm[t] = (c8v[t] - c8v[t + 4]) * 0.5L;
        }
        for (int m = 0; m < 4; ++m)
            for (int n = 0; n < 8; ++n) {
                double v;
                if (n < 4)
                    v = (double)cp[(m + n) & 3];
                else {
                    int nn = n - 4;
                    v = (double)cm[(m + nn) & 3] * ((m + nn) < 4 ? 1.0 : -1.0);
                }
                for (int t = 0; t < 8; ++t) p->ntc8[(m * 8 + n) * 8 + t] = v;
                for (int t = 0; t < 4; ++t) p->ntc4[(m * 8 + n) * 4 + t] = v;
            }
        for (int m = 0; m < 8; ++m)
            for (int n = 0; n < 8; ++n) {
                int t2 = m + n;
                double v = (double)s8v[t2 & 7] * (t2 < 8 ? 1.0 : -1.0);
                for (int t = 0; t < 8; ++t) p->nts8[(m * 8 + n) * 8 + t] = v;
                for (int t = 0; t < 4; ++t) p->nts4[(m * 8 + n) * 4 + t] = v;
            }
    }

    /* ---- pick the variant by measuring it, here, on this machine ----
     * Stage 1 chooses the compute structure: vector width (512/256-bit) x how
     * the chunk kernel spends registers.  Which one wins depends on the number
     * of 512-bit FMA units, on the AVX-512 licence clock and on what the
     * register allocator did -- all properties of the machine doing the run.
     * Stage 2 (batch >= 64 only) decides whether to stream the output with
     * non-temporal stores, which is a question about DRAM traffic and so has to
     * be timed on a working set that actually leaves L3. */
    {
        typedef void (*l17_fn)(const fft3d_plan *, const double _Complex *,
                               double _Complex *);
        enum { L17_NCAND = 38 };
        static const l17_fn cand[L17_NCAND] = {
            exec_w4, exec2_w4, exec3_w4, exec4_w4, exec6_w4, exec7_w4,
            exec10_w4, exec11_w4, exec12_w4, exec13_w4, exec14_w4, exec15_w4,
            exec_w2, exec2_w2, exec3_w2, exec4_w2, exec6_w2, exec7_w2,
            exec10_w2, exec11_w2, exec12_w2, exec13_w2, exec14_w2, exec15_w2,
            exec18_w4, exec19_w4, exec18_w2, exec19_w2,
            exec17_w4, exec17_w2, exec20_w4, exec20_w2,
            l17_execm_xl, l17_execm_xlp, l17_execm_xf, l17_execm_pipe,
            l17_execm_xld, l17_execm_xfd,
        };
        static const char *const tags[L17_NCAND] = {
            "dense 17x17 matrix/axis, conj-folded, 512-bit, fused transpose",
            "dense 17x17 matrix/axis, conj-folded, 512-bit, split transpose",
            "dense 17x17 matrix/axis, conj-folded, 512-bit, k-blocked",
            "dense 17x17 matrix/axis, conj-folded, 512-bit, k-blocked+L1 butterfly",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit, C parked in L1",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit, X-first",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit, C parked, X-first",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit, C parked, pinned sines",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit, C parked, pinned, X-first",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit, pinned sines",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit, pinned, X-first",
            "dense 17x17 matrix/axis, conj-folded, 256-bit, fused transpose",
            "dense 17x17 matrix/axis, conj-folded, 256-bit, split transpose",
            "dense 17x17 matrix/axis, conj-folded, 256-bit, k-blocked",
            "dense 17x17 matrix/axis, conj-folded, 256-bit, k-blocked+L1 butterfly",
            "nested cyclic/negacyclic 17-pt/axis, 256-bit",
            "nested cyclic/negacyclic 17-pt/axis, 256-bit, C parked in L1",
            "nested cyclic/negacyclic 17-pt/axis, 256-bit, X-first",
            "nested cyclic/negacyclic 17-pt/axis, 256-bit, C parked, X-first",
            "nested cyclic/negacyclic 17-pt/axis, 256-bit, C parked, pinned sines",
            "nested cyclic/negacyclic 17-pt/axis, 256-bit, C parked, pinned, X-first",
            "nested cyclic/negacyclic 17-pt/axis, 256-bit, pinned sines",
            "nested cyclic/negacyclic 17-pt/axis, 256-bit, pinned, X-first",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit, pinned, X-first, pipelined",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit, C parked, pinned, X-first, pipelined",
            "nested cyclic/negacyclic 17-pt/axis, 256-bit, pinned, X-first, pipelined",
            "nested cyclic/negacyclic 17-pt/axis, 256-bit, C parked, pinned, X-first, pipelined",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit, C parked, pinned, X-first, NT store",
            "nested cyclic/negacyclic 17-pt/axis, 256-bit, C parked, pinned, X-first, NT store",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit, C parked, pinned, X-first, pipelined, NT store",
            "nested cyclic/negacyclic 17-pt/axis, 256-bit, C parked, pinned, X-first, pipelined, NT store",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned sines",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, C parked, pinned sines",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, X-first",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, X-first, pipelined",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, deferred-Z",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, X-first, deferred-Z",
        };
        /* The tuner may only SELECT within one bit-equivalence class, chosen
         * deterministically by the batch size; everything else is measured for
         * the verbose table but never picked.  Reason: variants that round
         * differently (X-first reorders the passes; the pinned sine path
         * contracts differently under gcc) are all within tolerance but not
         * bit-identical, and a wall-clock tuner is not deterministic across
         * processes -- two runs of the same binary could otherwise return
         * different bits, which the panel's repeatability check treats as a
         * failure.  Within a class (verified by cmp: width, C-parking and the
         * NT copy do not change a single bit) the tuner is free.
         *   batch <  64: X-last pinned class  {exec12/14, both widths}
         *   batch >= 64: X-first pinned class {exec13/15, both widths}
         * The regime rule itself is measurement-based (wallaby, panel_r3):
         * X-first won batch 256/2048 by 14-17%, lost B=1/B=8 by ~5%. */
        const int selB[7] = {8, 10, 20, 22, 32, 33, 36};  /* X-last pinned  */
        const int selD[15] = {9, 11, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                              34, 35, 37};
        /* selD = the X-first pinned bit class: plain/pipelined/NT, both
         * widths, ALL cmp-verified bit-identical (r4).  NT variants are in
         * the class itself rather than a separate stage-2 A/B because the
         * choice does not factorize: on wallaby NT loses without pipelining
         * (13.75 vs 12.58, r3) and wins 14% WITH it at B=2048 -- a
         * winner-then-A/B tuner structurally cannot find that corner.
         * r5: the mixed-tail variants 32/33 (X-last) and 34/35 (X-first)
         * were cmp-verified bit-identical to their class on wallaby (full
         * outputs, B=8 and B=256) before being added here.
         * r6: the deferred-Z variants 36 (X-last -> class B) and 37
         * (X-first -> class D) are pure scheduling on double-buffered plane
         * buffers; cmp-verified bit-identical to 32 resp. 34 on full outputs
         * before being added. */
        const int *sel = (batch >= 64) ? selD : selB;
        const int nsel = (batch >= 64) ? 15 : 7;
        int bestv = sel[0];
        p->exec = cand[bestv];
        g_desc = tags[bestv];

        /* stage 1 (batch < 64).  Each candidate is measured in a BLOCK of
         * consecutive runs, never interleaved with the others: interleaving
         * 512-bit and 256-bit kernels makes every sample pay an AVX-512
         * licence/frequency transition, which on the Gold 5218 cost ~35% and
         * biased the ranking.  Blocks of >= 64 volume-transforms also give the
         * clock time to settle.  Everything is measured (the table is the
         * record); only the batch-regime's bit-class may be selected. */
        if (batch < 64) {
            int nv = batch < 16 ? batch : 16;
            int inner = (64 + nv - 1) / nv;
            if (l17_tune_alloc(p, nv)) {
                int sb = p->batch;
                p->batch = nv;
                double best[L17_NCAND];
                for (int v = 0; v < L17_NCAND; ++v) {
                    best[v] = 1e30;
                    cand[v](p, p->ti, p->to);
                    cand[v](p, p->ti, p->to);
                    for (int r = 0; r < 3; ++r) {
                        double t0 = l17_now();
                        for (int q2 = 0; q2 < inner; ++q2) cand[v](p, p->ti, p->to);
                        double dt = l17_now() - t0;
                        if (dt < best[v]) best[v] = dt;
                    }
                }
                p->batch = sb;
                for (int s = 1; s < nsel; ++s)
                    if (best[sel[s]] < best[bestv]) bestv = sel[s];
                p->exec = cand[bestv];
                g_desc = tags[bestv];
                if (l17_verbose())
                    for (int v = 0; v < L17_NCAND; ++v)
                        fprintf(stderr, "[L17_matrixsimd tune] %-72s %8.2f us/transform%s\n",
                                tags[v], best[v] * 1e6 / (nv * inner), v == bestv ? "  <== kept" : "");
            }
        }

        /* stage 1b (batch >= 64): tune on a working set that actually leaves
         * L3, in consecutive blocks.  Round 2's stage 1 tuned on 16 volumes
         * (2.4 MB -- L3-resident) even at B=2048 and then only A/B'd the NT
         * store on that winner; but the streaming regime can prefer a
         * different kernel outright.  Borrowed from L17_winograd round 2,
         * which measured a 90% penalty for trusting the small-set pick at
         * batch. */
        if (batch >= 64) {
            int nv2 = l17_tune_nv(batch);
            if (l17_tune_alloc(p, nv2)) {
                int sb = p->batch;
                p->batch = nv2;
                double best2[L17_NCAND];
                for (int v = 0; v < L17_NCAND; ++v) {
                    best2[v] = 1e30;
                    cand[v](p, p->ti, p->to);
                    for (int r = 0; r < 3; ++r) {
                        double t0 = l17_now();
                        cand[v](p, p->ti, p->to);
                        double dt = l17_now() - t0;
                        if (dt < best2[v]) best2[v] = dt;
                    }
                }
                p->batch = sb;
                for (int s = 1; s < nsel; ++s)
                    if (best2[sel[s]] < best2[bestv]) bestv = sel[s];
                p->exec = cand[bestv];
                g_desc = tags[bestv];
                if (l17_verbose())
                    for (int v = 0; v < L17_NCAND; ++v)
                        fprintf(stderr, "[L17_matrixsimd tune >L3 nv=%d] %-72s %8.2f us/transform%s\n",
                                nv2, tags[v], best2[v] * 1e6 / nv2,
                                v == bestv ? "  <== kept" : "");
            }
        }

        /* stage 2 (batch >= 64): prefetch flags, A/B'd JOINTLY on the final
         * pick, blocked -- pf (cross-volume input prefetch) x pw (write-
         * intent prefetch of the out plane, new in r6).  Prefetches change
         * no bits, so this cannot affect repeatability.  Joint rather than
         * sequential because r4 established that interacting knobs must be
         * in one tournament (winner-then-A/B cannot find a non-factorizing
         * corner).  pf is skipped when a pipelined variant won (its
         * interleaved X chunks already touch volume b+1's input); pw is
         * offered only to the variants that have the hook (the X-first
         * nested/mixed family and the mixed pipelined exec). */
        if (batch >= 64) {
            const int pipewin = (bestv >= 24 && bestv <= 27) ||
                                bestv == 30 || bestv == 31 || bestv == 35;
            const int haspf = !pipewin;
            const int haspw = bestv == 9 || bestv == 11 || bestv == 21 ||
                              bestv == 23 || bestv == 28 || bestv == 29 ||
                              bestv == 34 || bestv == 35 || bestv == 37;
            int nv2 = l17_tune_nv(batch);
            if ((haspf || haspw) && l17_tune_alloc(p, nv2)) {
                int sb = p->batch;
                p->batch = nv2;
                double bt = 1e30;
                int bpf = 0, bpw = 0;
                for (int fp = 0; fp <= haspf; ++fp)
                    for (int fw = 0; fw <= haspw; ++fw) {
                        p->pf = fp;
                        p->pw = fw;
                        double bb = 1e30;
                        p->exec(p, p->ti, p->to);
                        for (int r = 0; r < 3; ++r) {
                            double t0 = l17_now();
                            p->exec(p, p->ti, p->to);
                            double dt = l17_now() - t0;
                            if (dt < bb) bb = dt;
                        }
                        if (l17_verbose())
                            fprintf(stderr, "[L17_matrixsimd tune] nv=%d  "
                                            "pf=%d pw=%d  %.2f us/transform\n",
                                    nv2, fp, fw, bb * 1e6 / nv2);
                        if (bb < bt) { bt = bb; bpf = fp; bpw = fw; }
                    }
                p->batch = sb;
                p->pf = bpf;
                p->pw = bpw;
                static char g_desc_buf[200];
                snprintf(g_desc_buf, sizeof g_desc_buf, "%s, pf=%d, pw=%d",
                         g_desc, p->pf, p->pw);
                g_desc = g_desc_buf;
            } else {
                p->pf = 0;
                p->pw = 0;
            }
        }
        l17_tune_free(p);
#if defined(L17_FORCE)
        /* development override: -DL17_FORCE=0..43 pins one variant;
         * -DL17_FORCE_PF / -DL17_FORCE_PW pin the prefetch flags */
        {
            static const l17_fn all[44] = {
                exec_w4, exec2_w4, exec3_w4, exec4_w4,
                exec_w2, exec2_w2, exec3_w2, exec4_w2, exec5_w4, exec5_w2,
                exec6_w4, exec7_w4, exec6_w2, exec7_w2, exec8_w4, exec8_w2,
                exec10_w4, exec11_w4, exec12_w4, exec13_w4, exec14_w4,
                exec15_w4, exec10_w2, exec11_w2,
                exec12_w2, exec13_w2, exec14_w2, exec15_w2,
                exec16_w4, exec17_w4, exec16_w2, exec17_w2,
                exec18_w4, exec19_w4, exec20_w4,   /* 32,33,34: pipelined 512 */
                exec18_w2, exec19_w2, exec20_w2,   /* 35,36,37: pipelined 256 */
                l17_execm_xl, l17_execm_xlp,       /* 38,39: mixed tail X-last */
                l17_execm_xf, l17_execm_pipe,      /* 40,41: mixed tail X-first */
                l17_execm_xld, l17_execm_xfd,      /* 42,43: deferred-Z (r6) */
            };
            p->exec = all[L17_FORCE];
            g_desc = "forced variant (L17_FORCE)";
#  ifdef L17_FORCE_PF
            p->pf = L17_FORCE_PF;
#  endif
#  ifdef L17_FORCE_PW
            p->pw = L17_FORCE_PW;
#  endif
        }
#endif
    }
    { /* measured sustained licence clocks (the monitor's r4 ask for L=17);
       * carried back on the leaderboard via the description string.  r6 adds
       * d256 = the DENSE (2 FMA/cycle) 256-bit clock, same process, to close
       * the r5 VERDICT's clk256 question: sparse-256 3.89 vs dense-256 2.89
       * in one string would confirm density as the licence discriminator. */
        double c512 = l17_clk512(), c256 = l17_clk256(), c256d = l17_clk256d();
        static char g_desc_clk[320];
        snprintf(g_desc_clk, sizeof g_desc_clk,
                 "%s, clk512/256=%.2f/%.2f GHz, d256=%.2f",
                 g_desc, c512 * 1e-9, c256 * 1e-9, c256d * 1e-9);
        g_desc = g_desc_clk;
    }
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    l17_tune_free(p); /* normally already freed at the end of create() */
    free(p->block);
    free(p);
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    plan->exec(plan, in, out);
}

#endif /* L17_TEMPLATE_PASS */
