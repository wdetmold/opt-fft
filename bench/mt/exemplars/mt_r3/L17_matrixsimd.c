/* Carried over from the SINGLE-THREAD competition, where this file finished as
 * written below. Your job in the multicore phase is to parallelise it across
 * 32 cores without losing its single-core efficiency -- read
 * ../PANEL_BRIEF.md, and read ../../geom/strategies/L17_matrixsimd.md for the full
 * history of how this kernel got here.
 */
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
 * ROUND panel_r7: COSINE-RESIDENT KERNEL (execm_xlc/xldc/xfc/xfdc)
 *   The nested kernel's cosine phase still read its 32 coefficients per chunk
 *   as memory-operand FMAs, although only 8 distinct constants exist (cp[0..3],
 *   cm[0..3] -- row m=0 of the table).  The cr variants load those 8 into
 *   registers once per chunk (asm-opaque, so gcc neither refolds nor hoists
 *   them into spill slots) and fully unroll the 4 m-iterations with the
 *   rotation (m+n)%4 and the negacyclic signs resolved at compile time as
 *   vfnmadd: 24 of ~73 remaining loads per zmm chunk gone, FP count unchanged.
 *   The same shape as r3's pinned sines, applied per chunk because pinning
 *   both constant sets per execute (16 of 32 registers) is a documented dead
 *   end.  The ymm tail keeps the table path: on a 16-register file the 8
 *   residents would spill.  NOT selectable this round: cmp shows the unrolled
 *   cosine is a different rounding fixed point (correct, 3.258e-16, but not
 *   bit-identical to the shipping classes), so the variants are measured and
 *   forceable only, pending a node number.
 *
 * ROUND panel_r8: ADDRESS-SAFE t1 (execm_xla/xlda/xfa/xfda, FORCE 48-51)
 *   Two address pathologies on the X pass, whose load and store streams walk
 *   289-complex rows in lockstep: 3 of 4 accesses split a cache line (row
 *   stride 4624 B = 16 mod 64), and with the measured (deterministic) heap
 *   layout the load stream 4K-aliases the store stream (~8 false store->load
 *   dependencies per chunk, stalling the FMAs the loads feed).  The twins pad
 *   t1's plane stride to 5120 B (80 lines -- chosen by the collision model:
 *   73-line padding makes aliasing WORSE, see l17_as_build) and shift t1's
 *   base per volume by a table lookup on (t1 - vout|vin) mod 4096, computed
 *   in create() by pure integer arithmetic.  Addresses only, so the twins
 *   sit INSIDE the r3 bit classes (cmp-verified).  Also this round: the r7
 *   cosine-resident variants are DEAD -- disassembly loop-body counts show
 *   gcc 11 rematerializes the unrolled cosine into +26 instructions/chunk
 *   (254 vs 228), the opposite of the intended 24-load deletion.
 *
 * ROUND panel_r9: IN-PLAN B=1 DECOMPOSITION PROBE + IN-PASS SOURCE PREFETCH
 *   r8 left B=1 at 43.9k cycles against the 33.4k mixed-shape port floor for
 *   the fifth round, with four mechanism classes falsified on the node
 *   (rescheduling, address alignment, spill deletion, movement-uop deletion).
 *   (1) Probe (adopted from L36_pfa r8's in-create() discriminator): each
 *   phase timed in situ (L2-resident sources) and again with an IDENTICAL
 *   instruction stream on L1-hot sources; the four numbers ride the
 *   description string as b1dec[yz/kyz/x/kx].  They separate the two
 *   surviving hypotheses -- L2->L1 fill latency at chunk heads vs the kernel
 *   itself being >floor even from L1 -- on the scoring machine, without a
 *   monitor counter run.  (2) pt: prefetcht0 of the volume's own upcoming
 *   chunk sources (next Y-source plane during each plane's work; the X
 *   pass's next-but-one chunk lines, 17 rows x 64 B, ~300 cycles of lead),
 *   the fill-latency fix matching hypothesis (a).  All earlier prefetch at
 *   L=17 was cross-volume DRAM-side (pf) or store-side (pw); nothing ever
 *   attacked the same-volume L2->L1 load side.  A/B'd blocked at plan time
 *   (batch < 64: on the stage-1 winner; batch >= 64: jointly in the
 *   (pf,pw,pt) grid); prefetches change no bits, so the classes stand.
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
/* Same trick for the 8 cosine constants of the cr (cosine-resident) kernel,
 * but per CHUNK, not per execute: pinning both sets across the whole execute
 * would hold 16 of 32 registers and make the kernel spill (r5 item 2, a
 * documented dead end).  Loaded fresh each chunk (8 loads instead of 32
 * memory-operand FMAs), made opaque so gcc neither refolds them into memory
 * operands nor hoists them out of the chunk loop into spill slots (r1 item 1,
 * the 272-instructions-per-chunk disaster). */
#  define L17_CR_ASM() __asm__("" : "+v"(CP0), "+v"(CP1), "+v"(CP2), "+v"(CP3), \
                                    "+v"(CM0), "+v"(CM1), "+v"(CM2), "+v"(CM3))
#else
#  define L17_PIN_ASM() do { } while (0)
#  define L17_CR_ASM() do { } while (0)
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
              int tr, int pc, int pin, int cr)
{
    static const long IA[8] = {1, 3, 9, 10, 13, 5, 15, 11};
    static const long IB[8] = {16, 14, 8, 7, 4, 12, 2, 6};

    VT C0, C1, C2, C3, C4, C5, C6, C7, X0v;
    if (cr) {
        /* ---- cosine side, REGISTER-RESIDENT (round panel_r7): the whole
         * cyclic-4 + negacyclic-4 uses only 8 distinct constants, cp[0..3] and
         * cm[0..3] -- exactly row m=0 of the table, where every negacyclic
         * sign eps(m+n) = +.  Load them once per chunk (8 loads) and fully
         * unroll the 4 m-iterations with the rotation (m+n)%4 and the signs
         * resolved at compile time as vfnmadd -- 24 of 32 cosine coefficient
         * loads per chunk gone, FP count unchanged.  Same shape as the r3
         * pinned sines, one level down: per-chunk residency instead of
         * per-execute, because per-execute would hold 16 of 32 registers.
         * fnmadd(c,q,B) rounds identically to fmadd(-c,q,B), but r3 says
         * derive nothing: selectable only after cmp on full outputs. ---- */
        VT CP0 = CGET(cn, 0), CP1 = CGET(cn, 1), CP2 = CGET(cn, 2), CP3 = CGET(cn, 3);
        VT CM0 = CGET(cn, 4), CM1 = CGET(cn, 5), CM2 = CGET(cn, 6), CM3 = CGET(cn, 7);
        L17_CR_ASM();
        VT x0 = VLD(src);
        VT A0 = x0, A1 = x0, A2 = x0, A3 = x0, X0a = x0;
        VT B0, B1, B2, B3;
        { /* m = 0: IA/IB pairs (1,16) and (13,4) */
            VT a = VLD(src + 1 * rs), b = VLD(src + 16 * rs);
            VT a2 = VLD(src + 13 * rs), b2 = VLD(src + 4 * rs);
            VT u = a + b, u2 = a2 + b2;
            VT pp = u + u2, q = u - u2;
            X0a += pp;
            A0 += CP0 * pp; A1 += CP1 * pp; A2 += CP2 * pp; A3 += CP3 * pp;
            B0 = CM0 * q;  B1 = CM1 * q;  B2 = CM2 * q;  B3 = CM3 * q;
        }
        { /* m = 1: (3,14) and (5,12) */
            VT a = VLD(src + 3 * rs), b = VLD(src + 14 * rs);
            VT a2 = VLD(src + 5 * rs), b2 = VLD(src + 12 * rs);
            VT u = a + b, u2 = a2 + b2;
            VT pp = u + u2, q = u - u2;
            X0a += pp;
            A0 += CP1 * pp; A1 += CP2 * pp; A2 += CP3 * pp; A3 += CP0 * pp;
            B0 += CM1 * q; B1 += CM2 * q; B2 += CM3 * q; B3 -= CM0 * q;
        }
        { /* m = 2: (9,8) and (15,2) */
            VT a = VLD(src + 9 * rs), b = VLD(src + 8 * rs);
            VT a2 = VLD(src + 15 * rs), b2 = VLD(src + 2 * rs);
            VT u = a + b, u2 = a2 + b2;
            VT pp = u + u2, q = u - u2;
            X0a += pp;
            A0 += CP2 * pp; A1 += CP3 * pp; A2 += CP0 * pp; A3 += CP1 * pp;
            B0 += CM2 * q; B1 += CM3 * q; B2 -= CM0 * q; B3 -= CM1 * q;
        }
        { /* m = 3: (10,7) and (11,6) */
            VT a = VLD(src + 10 * rs), b = VLD(src + 7 * rs);
            VT a2 = VLD(src + 11 * rs), b2 = VLD(src + 6 * rs);
            VT u = a + b, u2 = a2 + b2;
            VT pp = u + u2, q = u - u2;
            X0a += pp;
            A0 += CP3 * pp; A1 += CP0 * pp; A2 += CP1 * pp; A3 += CP2 * pp;
            B0 += CM3 * q; B1 -= CM0 * q; B2 -= CM1 * q; B3 -= CM2 * q;
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
    } else { /* ---- cosine side: cyclic-4 (A) + negacyclic-4 (B) on P,Q ---- */
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
                                  0, PC, PIN, 0);                                 \
                }                                                              \
                if (289 % WC)                                                  \
                    SUF(chunk17n)(vin + 2 * (289 - WC), 578,                   \
                                  t1 + 2 * (289 - WC), 2, 578,                 \
                                  cn, sn, sc, K0, K1, K2, K3, K4, K5, K6, K7,  \
                                  0, PC, PIN, 0);                                 \
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
                                  1, PC, PIN, 0);                                 \
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
                                  1, PC, PIN, 0);                                 \
                }                                                              \
            }                                                                  \
            if (!REORD) { /* X last: t1 -> out (or the NT staging buffer) */   \
                for (int i = 0; i < 289 / WC; ++i) {                           \
                    long f0 = (long)i * WC;                                    \
                    SUF(chunk17n)(t1 + 2 * f0, 578, dstv + 2 * f0, 2, 578,     \
                                  cn, sn, sc, K0, K1, K2, K3, K4, K5, K6, K7,  \
                                  0, PC, PIN, 0);                                 \
                }                                                              \
                if (289 % WC)                                                  \
                    SUF(chunk17n)(t1 + 2 * (289 - WC), 578,                    \
                                  dstv + 2 * (289 - WC), 2, 578,               \
                                  cn, sn, sc, K0, K1, K2, K3, K4, K5, K6, K7,  \
                                  0, PC, PIN, 0);                                 \
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
                              0, PC, 1, 0);                                       \
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
                                      K7, 0, PC, 1, 0);                           \
                    }                                                          \
                }                                                              \
                const double *pin2 = cur + (long)x * 578;                      \
                double *pt = dstv + (long)x * 578;                             \
                for (int t = 0; t < NOFF17; ++t) {                             \
                    long f0 = SUF(off17)[t];                                   \
                    SUF(chunk17n)(pin2 + 2 * f0, 34, pb + f0 * PBROW, PBROW, 2,\
                                  cn, sn, sc, K0, K1, K2, K3, K4, K5, K6, K7,  \
                                  1, PC, 1, 0);                                   \
                }                                                              \
                if (hn) {                                                      \
                    for (int i = ih; i < i1; ++i) {                            \
                        long f0 = i < 289 / WC ? (long)i * WC : 289 - WC;      \
                        SUF(chunk17n)(vinN + 2 * f0, 578, nxt + 2 * f0, 2, 578,\
                                      cn, sn, sc, K0, K1, K2, K3, K4, K5, K6,  \
                                      K7, 0, PC, 1, 0);                           \
                    }                                                          \
                }                                                              \
                for (int t = 0; t < NOFF17; ++t) {                             \
                    long f0 = SUF(off17)[t];                                   \
                    SUF(chunk17n)(pb + 2 * f0, PBROW, pt + f0 * 34, 34, 2,     \
                                  cn, sn, sc, K0, K1, K2, K3, K4, K5, K6, K7,  \
                                  1, PC, 1, 0);                                   \
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
#undef L17_CR_ASM
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

#ifndef _GNU_SOURCE
#  define _GNU_SOURCE /* pthread_setaffinity_np, sched_getaffinity */
#endif

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifdef _OPENMP
#  include <omp.h>
#  include <pthread.h>
#  include <sched.h>
#  include <sys/mman.h> /* mt_r2: t1g is mmap'd so a re-touch gets fresh pages */
#endif

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
    int pt; /* round panel_r9: in-pass prefetcht0 of the current volume's own
             * chunk sources -- next plane of the Y-group source during each
             * plane's work, and the X pass's next-chunk source lines (17 rows
             * x 64 B, two chunks ahead).  Aimed at L2->L1 fill latency at the
             * head of each chunk while the ROB is full of the previous one:
             * the one stall class no L=17 mechanism has attacked (pf was
             * cross-volume DRAM-side, pw is store-side, deferred-Z was the
             * store->load junction).  A/B'd at plan time; changes no bits. */
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
    double *so0; /* staged-output ping-pong stages (round panel_r11): volume b's
                  * Z groups store here; the stage is flushed to out in paced
                  * half-plane bursts under volume b+1's plane-phase compute   */
    double *so1;
    void *block;
    double _Complex *ti, *to; /* transient buffers used only by the plan-time tuner */
    size_t tn;
    /* ---- multicore phase (round mt_r1) ------------------------------------
     * mode 0: single thread, master exec (the phase-1 path, unchanged)
     * mode 1: volume-parallel -- each thread runs the SAME selected exec on
     *         its own contiguous run of volumes (static) or on dynamically
     *         grabbed blocks of dynb volumes, with fully private scratch
     * mode 2: intra-volume -- the X-last class-B transform decomposed into
     *         17*B independent (plane: Y+Z) units, one barrier, then 73*B
     *         independent X-chunk units, all chunk-for-chunk identical to
     *         l17_execm_xla, so the bits match class B exactly
     * All tuning of (mode, nthr, dynb, kernel, flags) happens in create() by
     * timing the real parallel path; every candidate within a batch regime
     * is in one bit class, so the wall-clock pick cannot change the output. */
    int mode;
    int nthr;               /* team size for modes 1 and 2 */
    int dynb;               /* mode 1: >0 = dynamic blocks of dynb volumes */
    int nxr;                /* mode 2: X-phase team size (<= nthr; the plane
                             * phase scales to ~17 threads, the X phase
                             * saturates near 4 -- measured, see strategies) */
    int xpf;                /* mode 2 X-phase cross-core prefetch: 0 = none,
                             * 1 = chunk h prefetches chunk h+2's rows,
                             * 2 = mt_r2 bulk pull of the whole range at the
                             *     barrier exit (capped at 24 chunks) */
    int nkids;
    struct fft3d_plan **kids; /* per-thread plans: own tables + scratch,
                               * first-touched by their own thread (NUMA) */
    double *t1g;            /* mode 2: batch padded t1 volumes (17*640 dbl each) */
    void *t1g_raw;          /* mmap'd (mt_r2): re-touching needs fresh pages */
    size_t t1g_sz;
    void *poolv;            /* persistent spin pool (l17mt_pool), created once */
    int mtskip;             /* dev-only (L17MT_SKIP): bit0 skip planes, bit1
                             * skip xrange, bit2 skip barrier -- OUTPUT IS
                             * WRONG, timing decomposition only */
    unsigned char astab[2][256]; /* per-volume t1 base shifts (round panel_r8):
                                  * astab[mode][((t1 - ref) & 4095) >> 4] is the
                                  * 64-byte-step shift minimizing X-pass 4K
                                  * aliasing; mode 0 = X-last (ref = vout),
                                  * mode 1 = X-first (ref = vin). */
    /* ---- round mt_r3: streaming-regime engine (adopted from L17_winograd,
     * see the block comment above wg_fused23_h4).  Selected by a
     * DETERMINISTIC working-set gate in create(), never by a raced pick. */
    double *wgbuf;          /* per-child 4 x 17*SP doubles, own pages */
    void *wgbuf_raw;
    const int32_t *wgt;     /* store-base tables, shared read-only (master
                             * owns wgt_raw; children only point at it) */
    void *wgt_raw;
    int wgi4;               /* 1 = split-free pass 1 (i4), 0 = h4 */
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

/* =============================================================================
 * ROUND panel_r8: address-safe t1 (the "a" twins, xla/xlda/xfa/xfda).
 *
 * Two address pathologies sit on the X pass, whose load and store streams both
 * walk 289-complex rows in lockstep (same 16*p0 offsets, rows 4624 B apart):
 *
 *  (1) LINE SPLITS: 4624 B = 16 (mod 64), so consecutive t1 planes cycle the
 *      four 16-byte alignment classes and 3 of 4 zmm accesses to t1 split a
 *      cache line (L23_matrixsimd r7 fixed the same disease with its 1058->
 *      1064 t1 padding, credited by the r7 VERDICT as that round's best
 *      mechanism).  Fix: pad the t1 plane stride to a whole number of lines
 *      (640 doubles = 5120 B = 80 lines, see below for why not 73), so every
 *      access at a multiple-of-4 lane offset is 64-byte aligned.  (t1 only --
 *      `in`/`out` strides are the contract's.)
 *
 *  (2) 4K ALIASING: a load whose address matches an in-flight store in bits
 *      11:0 pays a false store->load dependency (ld_blocks_partial.
 *      address_alias), and here that stalls the FP stream itself, because the
 *      X chunk's loads feed FMAs directly.  The exposure is FIXED, not random:
 *      the heap layout is deterministic (measured on wallaby: (in - t1) mod
 *      4096 = 2624, (out - t1) mod 4096 = 3456 in every process), and with
 *      those residues chunk c's loads at row j alias chunk c-1's stores at
 *      row j+9 within 48 bytes -- ~8 aliasing pairs per X chunk, every chunk,
 *      every volume.  Fix: shift t1's base (mod 4096, in 64 B steps, inside
 *      slack carved in create()) PER VOLUME -- legal because t1 is fully
 *      rewritten every volume -- so that the X pass's load and store streams
 *      collide as little as the stride arithmetic allows.  The shift is a
 *      256-entry table lookup on ((t1 - vout|vin) mod 4096) >> 4; the table
 *      is pure integer arithmetic, built once in create().
 *
 * The stride was CHOSEN BY THE COLLISION MODEL, not just rounded up: over all
 * start residues, the best reachable weighted collision count (17x17 row
 * pairs, store-buffer lag 1..3 chunks, window +-64 B) is
 *      dense 4624 B: 8     4672 B (73 lines): 31     5120 B (80 lines): <= 4
 * i.e. the natural 73-line padding makes aliasing WORSE; 80 lines = 1.25
 * pages makes it (near-)zero from every start residue, for both pass orders.
 *
 * Both fixes are address-only: no value changes, so the twins are candidates
 * INSIDE the existing bit classes (cmp-verified on full outputs before being
 * added).
 * =============================================================================
 */
enum { L17_T1SP = 640 }; /* padded t1 plane stride, doubles (5120 B, 80 lines) */

/* per-volume de-aliased t1 base: BASE + 64*astab[MODE][class] bytes */
#define L17_AS_T1V(P, BASE, REFPTR, MODE)                                      \
    ((BASE) + 8L * (P)->astab[MODE]                                            \
                  [(((uintptr_t)(BASE) - (uintptr_t)(REFPTR)) & 4095u) >> 4])

/* Build the shift tables: for each 16-byte residue class of
 * (t1 - ref) mod 4096, the 64-byte-step shift d minimizing the weighted
 * collision count between the X pass's load and store streams.
 *   mode 0 (X-last):  loads t1 (rows 5120 B), stores out (rows 4624 B)
 *   mode 1 (X-first): loads in (rows 4624 B), stores t1 (rows 5120 B)
 * A hit is a load of chunk c+g (lane offset +64g B) landing within +-64 B,
 * mod 4096, of a store of chunk c (still in the store buffer for g <= 3);
 * nearer stores weigh more.  Pure integer arithmetic: same table on every
 * machine, and addresses change no bits. */
static void l17_as_build(unsigned char astab[2][256])
{
    for (int mode = 0; mode < 2; ++mode)
        for (int cls = 0; cls < 256; ++cls) {
            long bests = 0x7fffffffL;
            int bestj = 0;
            for (int j = 0; j < 64; ++j) {
                long t = (16L * cls + 64L * j) & 4095L; /* (t1+d-ref) mod 4096 */
                long score = 0;
                for (int x = 0; x < 17 && score < bests; ++x)
                    for (int k = 0; k < 17; ++k)
                        for (int g = 1; g <= 3; ++g) {
                            long diff = mode == 0
                                ? t + 5120L * x + 64L * g - 4624L * k
                                : 4624L * x + 64L * g - t - 5120L * k;
                            diff &= 4095L;
                            if (diff >= 2048) diff -= 4096;
                            if (diff > -64 && diff < 64) score += 4 - g;
                        }
                if (score < bests) { bests = score; bestj = j; }
            }
            astab[mode][cls] = (unsigned char)bestj;
        }
}

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
#define L17_MIX_GROUP(SRCB, RS, DSTB, DA, PC, CR)                              \
    do {                                                                       \
        int nt_ = 4;                                                           \
        __asm__("" : "+r"(nt_));                                               \
        for (int t_ = 0; t_ < nt_; ++t_) {                                     \
            long f0_ = 4L * t_;                                                \
            chunk17n_w4((SRCB) + 2 * f0_, (RS), (DSTB) + f0_ * (DA), (DA), 2,  \
                        cn8, sn8, sc, K0, K1, K2, K3, K4, K5, K6, K7,          \
                        1, (PC), 1, (CR));                                     \
        }                                                                      \
        chunk17n_w2((SRCB) + 2 * 15, (RS), (DSTB) + 15 * (DA), (DA), 2,        \
                    cn4, sn4, sc, Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, 1, 0, 1, 0); \
    } while (0)

/* The X pass, mixed: 72 zmm chunks cover lines 0..287 of the 289-long free
 * index, one ymm tail at 287 covers 287,288 (287 recomputed).  SRS/DRS are
 * the source/destination plane strides in doubles (578 dense, L17_T1SP for
 * the padded-t1 "a" twins). */
#define L17_MIX_XPASS(SRCB, SRS, DSTB, DRS, PC, CR)                            \
    do {                                                                       \
        for (int i_ = 0; i_ < 72; ++i_) {                                      \
            long f0_ = 4L * i_;                                                \
            chunk17n_w4((SRCB) + 2 * f0_, (SRS), (DSTB) + 2 * f0_, 2, (DRS),   \
                        cn8, sn8, sc, K0, K1, K2, K3, K4, K5, K6, K7,          \
                        0, (PC), 1, (CR));                                     \
        }                                                                      \
        chunk17n_w2((SRCB) + 2 * 287, (SRS), (DSTB) + 2 * 287, 2, (DRS),       \
                    cn4, sn4, sc, Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, 0, 0, 1, 0); \
    } while (0)

/* ROUND panel_r9: the same X pass with an in-pass source prefetch -- inside
 * chunk i, prefetcht0 the 17 source lines chunk i+2 will load (one 64 B line
 * per row; ~300 cycles of lead, enough for an L2 hit and most of a DRAM miss).
 * A SEPARATE macro rather than a branch in the chunk loop, so the pt=0 path's
 * loop body is untouched (the r8 VERDICT documents four regressions from
 * refactors around an untouched hot path).  The last two chunks prefetch a few
 * lines past the row end; those stay inside t1's padded planes in the X-last
 * order and at worst 48 B past the last volume of `in` in the X-first order --
 * prefetch is a hint and never faults, and the tail lines are simply wasted. */
#define L17_MIX_XPASS_PT(SRCB, SRS, DSTB, DRS, PC, CR)                          \
    do {                                                                       \
        for (int i_ = 0; i_ < 72; ++i_) {                                      \
            long f0_ = 4L * i_;                                                \
            const double *pfsrc_ = (SRCB) + 2 * (f0_ + 8);                     \
            for (int r_ = 0; r_ < 17; ++r_)                                    \
                __builtin_prefetch(pfsrc_ + (long)r_ * (SRS), 0, 3);           \
            chunk17n_w4((SRCB) + 2 * f0_, (SRS), (DSTB) + 2 * f0_, 2, (DRS),   \
                        cn8, sn8, sc, K0, K1, K2, K3, K4, K5, K6, K7,          \
                        0, (PC), 1, (CR));                                     \
        }                                                                      \
        chunk17n_w2((SRCB) + 2 * 287, (SRS), (DSTB) + 2 * 287, 2, (DRS),       \
                    cn4, sn4, sc, Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, 0, 0, 1, 0); \
    } while (0)

/* prefetcht0 one whole source plane (STRIDE doubles) for the pt variants */
#define L17_MIX_YPF(BASE, STRIDE)                                              \
    do {                                                                       \
        const char *ypf_ = (const char *)(BASE);                               \
        for (int qy_ = 0; qy_ < (int)((STRIDE) / 8); ++qy_)                    \
            __builtin_prefetch(ypf_ + (long)qy_ * 64, 0, 3);                   \
    } while (0)

/* one X chunk-slot for the pipelined exec: slots 0..71 are zmm, 72 the tail */
#define L17_MIX_XSLOT(SRC, DST, I)                                             \
    do {                                                                       \
        if ((I) < 72) {                                                        \
            long f0_ = 4L * (I);                                               \
            chunk17n_w4((SRC) + 2 * f0_, 578, (DST) + 2 * f0_, 2, 578,         \
                        cn8, sn8, sc, K0, K1, K2, K3, K4, K5, K6, K7, 0, 0, 1, \
                        0);                                                    \
        } else {                                                               \
            chunk17n_w2((SRC) + 2 * 287, 578, (DST) + 2 * 287, 2, 578,         \
                        cn4, sn4, sc, Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, 0, 0, 1, \
                        0);                                                    \
        }                                                                      \
    } while (0)

/* mirror of exec14 (PC=0) / exec12 (PC=1): X-last, pinned sines, mixed tail.
 * T1S = t1 plane stride (doubles); AS = 1 selects the execute-time
 * de-aliased t1 base (the X pass LOADS t1 while storing `out`). */
#define L17_EXEC_MXL(NAME, PC, CR, T1S, AS)                                    \
    static __attribute__((unused)) void                                        \
    NAME(const fft3d_plan *restrict p, const double _Complex *restrict in,     \
         double _Complex *restrict out)                                        \
    {                                                                          \
        L17_MIX_DECLS;                                                         \
        for (int b = 0; b < nb; ++b) {                                         \
            const double *vin = (const double *)(in + (size_t)b * 4913);       \
            double *vout = (double *)(out + (size_t)b * 4913);                 \
            double *restrict t1 = (AS) ? L17_AS_T1V(p, p->t1, vout, 0)         \
                                       : p->t1;                                \
            for (int x = 0; x < 17; ++x) {                                     \
                const double *pin2 = vin + (long)x * 578;                      \
                double *pt = t1 + (long)x * (T1S);                             \
                if (p->pt && x + 1 < 17) /* r9: next in-plane into L1 */       \
                    L17_MIX_YPF(vin + (long)(x + 1) * 578, 578);               \
                L17_MIX_GROUP(pin2, 34, pb, L17_PBROWM, PC, CR);               \
                L17_MIX_GROUP(pb, L17_PBROWM, pt, 34, PC, CR);                 \
            }                                                                  \
            if (p->pt)                                                         \
                L17_MIX_XPASS_PT(t1, (T1S), vout, 578, PC, CR);                \
            else                                                               \
                L17_MIX_XPASS(t1, (T1S), vout, 578, PC, CR);                   \
        }                                                                      \
    }

/* mirror of exec15: X-first, pinned sines, mixed tail (batch regime).
 * With AS = 1 the X pass STORES t1 while loading `in`, so the de-aliasing
 * reference is `in`. */
#define L17_EXEC_MXF(NAME, PC, CR, T1S, AS)                                    \
    static __attribute__((unused)) void                                        \
    NAME(const fft3d_plan *restrict p, const double _Complex *restrict in,     \
         double _Complex *restrict out)                                        \
    {                                                                          \
        L17_MIX_DECLS;                                                         \
        for (int b = 0; b < nb; ++b) {                                         \
            const double *vin = (const double *)(in + (size_t)b * 4913);       \
            double *vout = (double *)(out + (size_t)b * 4913);                 \
            double *restrict t1 = (AS) ? L17_AS_T1V(p, p->t1, vin, 1)          \
                                       : p->t1;                                \
            if (p->pt)                                                         \
                L17_MIX_XPASS_PT(vin, 578, t1, (T1S), PC, CR);                 \
            else                                                               \
                L17_MIX_XPASS(vin, 578, t1, (T1S), PC, CR);                    \
            for (int x = 0; x < 17; ++x) {                                     \
                const double *pin2 = t1 + (long)x * (T1S);                     \
                double *pt = vout + (long)x * 578;                             \
                if (p->pt && x + 1 < 17) /* r9: next t1 plane into L1 */       \
                    L17_MIX_YPF(t1 + (long)(x + 1) * (T1S), (T1S));            \
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
                L17_MIX_GROUP(pin2, 34, pb, L17_PBROWM, PC, CR);               \
                if (p->pw) {                                                   \
                    const char *po = (const char *)pt;                         \
                    for (int q4 = 37; q4 < 73; ++q4)                           \
                        __builtin_prefetch(po + (long)q4 * 64, 1, 3);          \
                }                                                              \
                L17_MIX_GROUP(pb, L17_PBROWM, pt, 34, PC, CR);                 \
            }                                                                  \
        }                                                                      \
    }

L17_EXEC_MXL(l17_execm_xl, 0, 0, 578, 0)
L17_EXEC_MXL(l17_execm_xlp, 1, 0, 578, 0)
L17_EXEC_MXF(l17_execm_xf, 0, 0, 578, 0)
/* round panel_r7: cosine-register-resident twins (cr=1 in the zmm body) */
L17_EXEC_MXL(l17_execm_xlc, 0, 1, 578, 0)
L17_EXEC_MXF(l17_execm_xfc, 0, 1, 578, 0)
/* round panel_r8: address-safe twins (padded t1 stride + de-aliased base) */
L17_EXEC_MXL(l17_execm_xla, 0, 0, L17_T1SP, 1)
L17_EXEC_MXF(l17_execm_xfa, 0, 0, L17_T1SP, 1)

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
            L17_MIX_GROUP(pin2, 34, pb, L17_PBROWM, 0, 0);
            if (hn)
                for (int i = ih; i < i1; ++i) L17_MIX_XSLOT(vinN, nxt, i);
            if (p->pw) {
                const char *po = (const char *)pt;
                for (int q4 = 37; q4 < 73; ++q4)
                    __builtin_prefetch(po + (long)q4 * 64, 1, 3);
            }
            L17_MIX_GROUP(pb, L17_PBROWM, pt, 34, 0, 0);
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
#define L17_EXEC_MXLD(NAME, CR, T1S, AS)                                       \
    static __attribute__((unused)) void                                        \
    NAME(const fft3d_plan *restrict p, const double _Complex *restrict in,     \
         double _Complex *restrict out)                                        \
    {                                                                          \
        L17_MIX_DECLS;                                                         \
        double *restrict pb2 = p->pb2;                                         \
        for (int b = 0; b < nb; ++b) {                                         \
            const double *vin = (const double *)(in + (size_t)b * 4913);       \
            double *vout = (double *)(out + (size_t)b * 4913);                 \
            double *restrict t1 = (AS) ? L17_AS_T1V(p, p->t1, vout, 0)         \
                                       : p->t1;                                \
            L17_MIX_GROUP(vin, 34, pb, L17_PBROWM, 0, CR); /* Y(0) -> pb */    \
            for (int x = 0; x < 17; ++x) {                                     \
                double *pba = (x & 1) ? pb2 : pb; /* holds Y(x)'s plane */     \
                double *pbn = (x & 1) ? pb : pb2; /* Y(x+1)'s target    */     \
                if (p->pt && x + 2 < 17) /* r9: plane ahead of Y(x+1) */       \
                    L17_MIX_YPF(vin + (long)(x + 2) * 578, 578);               \
                if (x + 1 < 17)                                                \
                    L17_MIX_GROUP(vin + (long)(x + 1) * 578, 34, pbn,          \
                                  L17_PBROWM, 0, CR);                          \
                L17_MIX_GROUP(pba, L17_PBROWM, t1 + (long)x * (T1S), 34, 0,    \
                              CR);                                             \
            }                                                                  \
            if (p->pt)                                                         \
                L17_MIX_XPASS_PT(t1, (T1S), vout, 578, 0, CR);                 \
            else                                                               \
                L17_MIX_XPASS(t1, (T1S), vout, 578, 0, CR);                    \
        }                                                                      \
    }

L17_EXEC_MXLD(l17_execm_xld, 0, 578, 0)
L17_EXEC_MXLD(l17_execm_xldc, 1, 578, 0) /* r7: cosine-resident twin */
L17_EXEC_MXLD(l17_execm_xlda, 0, L17_T1SP, 1) /* r8: address-safe twin */

/* X-first + deferred-Z: the batch-regime twin, with the pf/pw hooks of
 * l17_execm_xf (prefetches change no bits, so it stays in bit class D). */
#define L17_EXEC_MXFD(NAME, CR, T1S, AS)                                       \
    static __attribute__((unused)) void                                        \
    NAME(const fft3d_plan *restrict p, const double _Complex *restrict in,     \
         double _Complex *restrict out)                                        \
    {                                                                          \
        L17_MIX_DECLS;                                                         \
        double *restrict pb2 = p->pb2;                                         \
        for (int b = 0; b < nb; ++b) {                                         \
            const double *vin = (const double *)(in + (size_t)b * 4913);       \
            double *vout = (double *)(out + (size_t)b * 4913);                 \
            double *restrict t1 = (AS) ? L17_AS_T1V(p, p->t1, vin, 1)          \
                                       : p->t1;                                \
            if (p->pt)                                                         \
                L17_MIX_XPASS_PT(vin, 578, t1, (T1S), 0, CR);                  \
            else                                                               \
                L17_MIX_XPASS(vin, 578, t1, (T1S), 0, CR);                     \
            L17_MIX_GROUP(t1, 34, pb, L17_PBROWM, 0, CR); /* Y(0) */           \
            for (int x = 0; x < 17; ++x) {                                     \
                double *pba = (x & 1) ? pb2 : pb;                              \
                double *pbn = (x & 1) ? pb : pb2;                              \
                double *pt = vout + (long)x * 578;                             \
                if (p->pt && x + 2 < 17) /* r9: plane ahead of Y(x+1) */       \
                    L17_MIX_YPF(t1 + (long)(x + 2) * (T1S), (T1S));            \
                if (p->pf && b + 1 < nb) {                                     \
                    const char *nx = (const char *)(vin + 9826) + (long)x * 4672; \
                    for (int q3 = 0; q3 < 73; ++q3)                            \
                        __builtin_prefetch(nx + (long)q3 * 64, 0, 2);          \
                }                                                              \
                if (p->pw) {                                                   \
                    const char *po = (const char *)pt;                         \
                    for (int q4 = 0; q4 < 37; ++q4)                            \
                        __builtin_prefetch(po + (long)q4 * 64, 1, 3);          \
                }                                                              \
                if (x + 1 < 17)                                                \
                    L17_MIX_GROUP(t1 + (long)(x + 1) * (T1S), 34, pbn,         \
                                  L17_PBROWM, 0, CR);                          \
                if (p->pw) {                                                   \
                    const char *po = (const char *)pt;                         \
                    for (int q4 = 37; q4 < 73; ++q4)                           \
                        __builtin_prefetch(po + (long)q4 * 64, 1, 3);          \
                }                                                              \
                L17_MIX_GROUP(pba, L17_PBROWM, pt, 34, 0, CR);                 \
            }                                                                  \
        }                                                                      \
    }

L17_EXEC_MXFD(l17_execm_xfd, 0, 578, 0)
L17_EXEC_MXFD(l17_execm_xfdc, 1, 578, 0) /* r7: cosine-resident twin */
L17_EXEC_MXFD(l17_execm_xfda, 0, L17_T1SP, 1) /* r8: address-safe twin */

/* -----------------------------------------------------------------------
 * ROUND panel_r10: staged-input streaming twins (xfsa / xfdsa, FORCE 52/53).
 * The r9 VERDICT priced the streaming cells at ~21.7 us = ~10.9 GB/s against
 * ~236 KB/volume of compulsory DRAM traffic and asked for "a new idea or an
 * honest closed".  The one traffic SHAPE nobody has attacked: in the X-first
 * order the volume's whole DRAM read is the X pass, which walks `in` as 17
 * interleaved streams of one row (4624 B = 1.13 pages) each -- too short for
 * the L2 streamer to lock onto before the page ends, so the read runs at
 * demand-miss speed, and it is issued by the same instructions that feed the
 * FMAs.  Fix: during volume b's compute-rich plane phase, copy volume b+1's
 * input SEQUENTIALLY (one long stream, the prefetch-friendliest pattern that
 * exists) into an L2-resident stage; volume b+1's X pass then reads the
 * stage at L2 latency.  DRAM traffic is unchanged -- the read is moved and
 * reshaped, not duplicated -- at the cost of a 78.6 KiB L2-resident
 * store+reload per volume (~2.5k extra uops in a phase that is not
 * port-bound at batch).  Differs from the r4 pipelined execs (which move the
 * X COMPUTE into the plane phase but keep the 17-stream read shape) exactly
 * where the sbw probe (below) can price the difference: s17 vs rd.
 *   - The stage lives in t1b (pipelined execs' second buffer; never
 *     co-selected) with a per-volume base shift keeping (stage - vin) mod
 *     4096 near 2048, so the copy's load and store streams cannot 4K-alias
 *     (the copy is otherwise the textbook aliasing case: 1:1 load/store at
 *     equal offsets from sliding bases).
 *   - The X pass's t1 base shift reuses the r8 astab, mode 1, with the
 *     STAGE as the load-side reference: the stage is a dense volume image
 *     (rows 4624 B apart), exactly the geometry the mode-1 model scores.
 *   - Volume 0's X pass reads `in` directly (same bits, and it never reads
 *     a stale stage, so repeated executes on one plan stay bit-identical).
 * Values are copied bit-exactly and every chunk runs in the same order on
 * the same values as xfa/xfda, so the twins belong to bit class D --
 * VERIFIED BY cmp on full outputs before being added to selD (r3 protocol).
 * --------------------------------------------------------------------- */
typedef double vd8u __attribute__((vector_size(64), aligned(8)));

static inline double *l17_stage(const fft3d_plan *p, const double *vin)
{
    uintptr_t b = (uintptr_t)p->t1b;
    uintptr_t off = (((uintptr_t)vin + 2048u) - b) & 4095u;
    off &= ~(uintptr_t)63; /* 64 B steps keep the stage 64-byte aligned */
    return (double *)(b + off);
}

/* NV doubles, NV % 8 == 0; unaligned vector moves (both sides are only
 * guaranteed 16-byte aligned) */
#define L17_STG_CPY8(DST, SRC, NV)                                             \
    do {                                                                       \
        for (long q_ = 0; q_ < (NV); q_ += 8)                                  \
            *(vd8u *)((DST) + q_) = *(const vd8u *)((SRC) + q_);               \
    } while (0)

static __attribute__((unused)) void
l17_execm_xfsa(const fft3d_plan *restrict p, const double _Complex *restrict in,
               double _Complex *restrict out)
{
    L17_MIX_DECLS;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 4913);
        double *vout = (double *)(out + (size_t)b * 4913);
        const double *vinN = (const double *)(in + (size_t)(b + 1) * 4913);
        const int hn = (b + 1 < nb);
        const double *xsrc = b ? l17_stage(p, vin) : vin;
        double *restrict sgn = hn ? l17_stage(p, vinN) : p->t1b;
        double *restrict t1 = L17_AS_T1V(p, p->t1, xsrc, 1);
        L17_MIX_XPASS(xsrc, 578, t1, L17_T1SP, 0, 0);
        for (int x = 0; x < 17; ++x) {
            const double *pin2 = t1 + (long)x * L17_T1SP;
            double *pt = vout + (long)x * 578;
            const double *cs = vinN + (long)x * 578;
            double *cd = sgn + (long)x * 578;
            if (hn) /* first half of this plane-slot's stage copy */
                L17_STG_CPY8(cd, cs, 288);
            if (p->pw) {
                const char *po = (const char *)pt;
                for (int q4 = 0; q4 < 37; ++q4)
                    __builtin_prefetch(po + (long)q4 * 64, 1, 3);
            }
            L17_MIX_GROUP(pin2, 34, pb, L17_PBROWM, 0, 0);
            if (hn) { /* second half, plus the 2-double plane tail */
                L17_STG_CPY8(cd + 288, cs + 288, 288);
                cd[576] = cs[576];
                cd[577] = cs[577];
            }
            if (p->pw) {
                const char *po = (const char *)pt;
                for (int q4 = 37; q4 < 73; ++q4)
                    __builtin_prefetch(po + (long)q4 * 64, 1, 3);
            }
            L17_MIX_GROUP(pb, L17_PBROWM, pt, 34, 0, 0);
        }
    }
}

/* deferred-Z twin of the above (Y(x+1) between Y(x) and Z(x), r6 schedule) */
static __attribute__((unused)) void
l17_execm_xfdsa(const fft3d_plan *restrict p, const double _Complex *restrict in,
                double _Complex *restrict out)
{
    L17_MIX_DECLS;
    double *restrict pb2 = p->pb2;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 4913);
        double *vout = (double *)(out + (size_t)b * 4913);
        const double *vinN = (const double *)(in + (size_t)(b + 1) * 4913);
        const int hn = (b + 1 < nb);
        const double *xsrc = b ? l17_stage(p, vin) : vin;
        double *restrict sgn = hn ? l17_stage(p, vinN) : p->t1b;
        double *restrict t1 = L17_AS_T1V(p, p->t1, xsrc, 1);
        L17_MIX_XPASS(xsrc, 578, t1, L17_T1SP, 0, 0);
        L17_MIX_GROUP(t1, 34, pb, L17_PBROWM, 0, 0); /* Y(0) */
        for (int x = 0; x < 17; ++x) {
            double *pba = (x & 1) ? pb2 : pb;
            double *pbn = (x & 1) ? pb : pb2;
            double *pt = vout + (long)x * 578;
            const double *cs = vinN + (long)x * 578;
            double *cd = sgn + (long)x * 578;
            if (hn)
                L17_STG_CPY8(cd, cs, 288);
            if (p->pw) {
                const char *po = (const char *)pt;
                for (int q4 = 0; q4 < 37; ++q4)
                    __builtin_prefetch(po + (long)q4 * 64, 1, 3);
            }
            if (x + 1 < 17)
                L17_MIX_GROUP(t1 + (long)(x + 1) * L17_T1SP, 34, pbn,
                              L17_PBROWM, 0, 0);
            if (hn) {
                L17_STG_CPY8(cd + 288, cs + 288, 288);
                cd[576] = cs[576];
                cd[577] = cs[577];
            }
            if (p->pw) {
                const char *po = (const char *)pt;
                for (int q4 = 37; q4 < 73; ++q4)
                    __builtin_prefetch(po + (long)q4 * 64, 1, 3);
            }
            L17_MIX_GROUP(pba, L17_PBROWM, pt, 34, 0, 0);
        }
    }
}

/* -----------------------------------------------------------------------
 * ROUND panel_r11: staged-OUTPUT streaming twins (xfso / xfdso, FORCE 54/55).
 * The r10 sbw probe on the node re-opened the streaming cells: cp (the
 * machine's own read-burst-then-write-burst floor for the cell's exact
 * 236 KB of compulsory traffic) reads 13.05-13.31 us against a 21.6 us
 * cell, s17/rd = 0.82 (the 17-stream X read is FINE there -- faster than
 * sequential), and cp ~= rd + wr says the node overlaps read bursts and
 * write bursts not at all.  So the ~8 us residual is WRITE-SIDE: the Z
 * group's per-plane store burst (73 RFO-missing lines, more than the
 * store buffer holds) stalls the compute that issues it, and the r10
 * VERDICT names the write-side analogue of the staged-input twins as this
 * round's one candidate.  Build: volume b's Z groups store their finished
 * planes into an L2-resident stage (fast stores, no RFO stall under the
 * compute); volume b's stage is flushed to out during volume b+1's plane
 * phase in HALF-PLANE bursts (289 doubles = 36/37 lines, sized to the
 * store buffer), each burst immediately followed by a compute group under
 * which it drains.  DRAM traffic is unchanged -- the RFO+writeback is
 * moved and paced, not duplicated -- at the cost of a 78.6 KiB L2
 * store+reload per volume (~2.5k uops in a phase that is not port-bound
 * at batch).  The flush is NOT interleaved into the X pass: cp ~= rd + wr
 * says the node serializes DRAM reads against DRAM writes, so the X
 * pass's read burst keeps the memory pipe to itself.  The last volume's
 * stage is flushed in an un-overlapped epilogue (~8 us once per call,
 * <=0.13 us/t at batch >= 64).
 *   - Each stage gets a per-volume base shift keeping (stage - vout) mod
 *     4096 near 2048: the flush is a 1:1 load/store copy at equal offsets
 *     from sliding bases, the textbook 4K-aliasing case (same rule as the
 *     r10 staged-input copy).  Z-group stores to stage b vs flush loads
 *     from stage b-1 then sit 784 B apart mod 4096 (the out volume
 *     stride's residue) -- clear of the +-64 B alias window.
 *   - pw prefetches the flush DESTINATION lines one compute group ahead
 *     of each burst (the r6 hook, repointed at the stream that actually
 *     misses; the Z stores now hit L2 and need no prefetch).
 *   - pf/pt are not offered: pf would push volume b+1's input reads into
 *     the write-paced plane phase (the exact mix cp says serializes), and
 *     the X pass is unchanged so pt keeps its existing owners.
 * Values are identical bits stored at a staged address and copied
 * bit-exactly, every chunk in the same order on the same operands as
 * xfa/xfda, so the twins belong to bit class D -- VERIFIED BY cmp on full
 * outputs before being added to selD (r3 protocol).
 * --------------------------------------------------------------------- */
static inline double *l17_so_base(double *sob, const double *vout)
{
    uintptr_t b = (uintptr_t)sob;
    uintptr_t off = (((uintptr_t)vout + 2048u) - b) & 4095u;
    off &= ~(uintptr_t)63; /* 64 B steps keep the stage 64-byte aligned */
    return (double *)(b + off);
}

/* flush the last volume's stage (epilogue): 9826 doubles, one burst */
#define L17_SO_EPILOGUE(P, OUT, NB)                                            \
    do {                                                                       \
        double *fdL_ = (double *)((OUT) + (size_t)((NB) - 1) * 4913);          \
        const double *fsL_ =                                                   \
            l17_so_base(((NB) - 1) & 1 ? (P)->so1 : (P)->so0, fdL_);           \
        L17_STG_CPY8(fdL_, fsL_, 9824);                                        \
        fdL_[9824] = fsL_[9824];                                               \
        fdL_[9825] = fsL_[9825];                                               \
    } while (0)

static __attribute__((unused)) void
l17_execm_xfso(const fft3d_plan *restrict p, const double _Complex *restrict in,
               double _Complex *restrict out)
{
    L17_MIX_DECLS;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 4913);
        double *vout = (double *)(out + (size_t)b * 4913);
        /* at b == 0 there is nothing to flush; the pointers below are then
         * dummies at valid addresses, guarded by `if (b)` at every use */
        double *voutP = b ? (double *)(out + (size_t)(b - 1) * 4913) : vout;
        double *sgw = l17_so_base((b & 1) ? p->so1 : p->so0, vout);
        const double *sgr = b ? l17_so_base((b & 1) ? p->so0 : p->so1, voutP)
                              : sgw;
        double *restrict t1 = L17_AS_T1V(p, p->t1, vin, 1);
        L17_MIX_XPASS(vin, 578, t1, L17_T1SP, 0, 0);
        for (int x = 0; x < 17; ++x) {
            const double *pin2 = t1 + (long)x * L17_T1SP;
            double *pt = sgw + (long)x * 578;
            double *fd = voutP + (long)x * 578;
            const double *fs = sgr + (long)x * 578;
            if (b && p->pw) { /* burst A's lines, one Y group of lead */
                const char *po = (const char *)fd;
                for (int q4 = 0; q4 < 37; ++q4)
                    __builtin_prefetch(po + (long)q4 * 64, 1, 3);
            }
            L17_MIX_GROUP(pin2, 34, pb, L17_PBROWM, 0, 0);
            if (b) { /* flush burst A; drains under the Z group below */
                L17_STG_CPY8(fd, fs, 288);
                if (p->pw) {
                    const char *po = (const char *)fd;
                    for (int q4 = 37; q4 < 73; ++q4)
                        __builtin_prefetch(po + (long)q4 * 64, 1, 3);
                }
            }
            L17_MIX_GROUP(pb, L17_PBROWM, pt, 34, 0, 0);
            if (b) { /* burst B; drains under the next slot's Y group */
                L17_STG_CPY8(fd + 288, fs + 288, 288);
                fd[576] = fs[576];
                fd[577] = fs[577];
            }
        }
    }
    L17_SO_EPILOGUE(p, out, nb);
}

/* deferred-Z twin of the above (Y(x+1) between Y(x) and Z(x), r6 schedule) */
static __attribute__((unused)) void
l17_execm_xfdso(const fft3d_plan *restrict p, const double _Complex *restrict in,
                double _Complex *restrict out)
{
    L17_MIX_DECLS;
    double *restrict pb2 = p->pb2;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 4913);
        double *vout = (double *)(out + (size_t)b * 4913);
        double *voutP = b ? (double *)(out + (size_t)(b - 1) * 4913) : vout;
        double *sgw = l17_so_base((b & 1) ? p->so1 : p->so0, vout);
        const double *sgr = b ? l17_so_base((b & 1) ? p->so0 : p->so1, voutP)
                              : sgw;
        double *restrict t1 = L17_AS_T1V(p, p->t1, vin, 1);
        L17_MIX_XPASS(vin, 578, t1, L17_T1SP, 0, 0);
        L17_MIX_GROUP(t1, 34, pb, L17_PBROWM, 0, 0); /* Y(0) */
        for (int x = 0; x < 17; ++x) {
            double *pba = (x & 1) ? pb2 : pb;
            double *pbn = (x & 1) ? pb : pb2;
            double *pt = sgw + (long)x * 578;
            double *fd = voutP + (long)x * 578;
            const double *fs = sgr + (long)x * 578;
            if (b && p->pw) {
                const char *po = (const char *)fd;
                for (int q4 = 0; q4 < 37; ++q4)
                    __builtin_prefetch(po + (long)q4 * 64, 1, 3);
            }
            if (x + 1 < 17)
                L17_MIX_GROUP(t1 + (long)(x + 1) * L17_T1SP, 34, pbn,
                              L17_PBROWM, 0, 0);
            if (b) {
                L17_STG_CPY8(fd, fs, 288);
                if (p->pw) {
                    const char *po = (const char *)fd;
                    for (int q4 = 37; q4 < 73; ++q4)
                        __builtin_prefetch(po + (long)q4 * 64, 1, 3);
                }
            }
            L17_MIX_GROUP(pba, L17_PBROWM, pt, 34, 0, 0);
            if (b) {
                L17_STG_CPY8(fd + 288, fs + 288, 288);
                fd[576] = fs[576];
                fd[577] = fs[577];
            }
        }
    }
    L17_SO_EPILOGUE(p, out, nb);
}

/* -----------------------------------------------------------------------
 * ROUND panel_r9: in-plan B=1 residual decomposition (pattern adopted from
 * L36_pfa's r8 in-create() probe, with attribution -- "when the monitor
 * cannot run your counter, build the discriminator into create() and route
 * it out through fft3d_description()", which the r8 VERDICT told the panel
 * to make its default).  The question it answers on the SCORING machine:
 * B=1 has sat at 43.9k cycles against the 33.4k-cycle mixed-shape port
 * floor for five rounds, and four mechanism classes (rescheduling, address
 * alignment, spill deletion, movement-uop deletion) are falsified.  The two
 * surviving hypotheses are (a) L2->L1 fill latency at chunk heads while the
 * ROB is full of the previous chunk, and (b) the kernel itself (front-end /
 * dependency limited even from L1).  Discriminator: time each phase in
 * situ (sources L2-resident, as in a real volume) and again with an
 * IDENTICAL instruction stream on L1-hot sources (stride argument = 0 or
 * 34, so the same code walks a 4.6-9 KiB footprint).  If hot ~= in-situ,
 * memory is innocent and the kernel carries the residual (hypothesis b, the
 * honest "B=1 is done" answer); if hot is near the port floor while in-situ
 * is not, the residual is fill latency (hypothesis a) and the pt prefetch
 * built this round is the matching fix.  Values are junk in the hot runs
 * (rows overlap); only time is read, and nothing here touches the output
 * path -- t1 is fully rewritten by every execute.
 * --------------------------------------------------------------------- */
static __attribute__((noinline)) void
l17_probe_yz(const fft3d_plan *restrict p, const double *restrict src, long ss,
             double *restrict dst, long ds)
{
    L17_MIX_DECLS;
    (void)nb;
    for (int x = 0; x < 17; ++x) {
        const double *pin2 = src + (long)x * ss;
        L17_MIX_GROUP(pin2, 34, pb, L17_PBROWM, 0, 0);
        L17_MIX_GROUP(pb, L17_PBROWM, dst + (long)x * ds, 34, 0, 0);
    }
}

static __attribute__((noinline)) void
l17_probe_x(const fft3d_plan *restrict p, const double *restrict src, long ss,
            double *restrict dst)
{
    L17_MIX_DECLS;
    (void)nb;
    L17_MIX_XPASS(src, ss, dst, 578, 0, 0);
}

/* =============================================================================
 * ROUND mt_r1: MULTICORE LAYER.
 *
 * Volume-parallel (mode 1, batch >= 64): volumes are independent, so each
 * thread runs the SAME plan-selected exec variant on its own contiguous run
 * of volumes (static split) or on dynamically grabbed blocks (dynb volumes
 * per grab; the caller's buffers are first-touched serially by the driver,
 * so on the two-socket node the remote socket's threads run slower and a
 * dynamic schedule rebalances that -- which of the two wins is measured).
 * Every thread has a fully private child plan (own coefficient tables, own
 * pb/sc/t1/stage scratch, first-touched by its own thread so it is
 * NUMA-local), so there is no shared mutable state at all and no false
 * sharing inside scratch.  Which thread computes a volume cannot change its
 * bits: the kernel and tables are identical, and the r8 address-shift is
 * address-only.
 *
 * Intra-volume (mode 2, batch < 64): one 17^3 volume is 78.6 KiB -- the
 * batch axis alone cannot use 32 cores at B=1.  The X-last class-B
 * transform decomposes into 17*B independent plane units (Y group + Z group
 * on a private plane buffer, in -> t1) and, after ONE barrier, 73*B
 * independent X-chunk units (t1 -> out).  Both phases run exactly the
 * chunk sequence of l17_execm_xla (mixed zmm+ymm tail, pinned sines, padded
 * t1 stride), so the output is bit-identical to class B -- the same
 * argument as the r4 pipelined variants: same chunks, same operands, same
 * per-value order, only the interleaving across independent units differs.
 * t1 uses the r8 padded plane stride (5120 B = 80 lines), so plane
 * boundaries never share a cache line between threads; the only remaining
 * sharing is the 17 out-lines at each phase-2 thread boundary (out rows
 * are 16 mod 64), which is bounded and measured, not guessed.
 *
 * ROUND mt_r2 (all schedule/address/timing-only; no bit class changes):
 *   1. ARENA FIDELITY.  The r1 arena was filled serially ("to match the
 *      driver's first touch"), but the r1 VERDICT (section 4, last para)
 *      established that the driver's pages do NOT stay on socket 0 through
 *      the multi-second scored loop -- AutoNUMA migrates them toward the
 *      static owners (L=6 B=65536 sustains 175 GB/s, above one socket), and
 *      the serial-touch arena therefore mis-priced the streaming regime:
 *      this entry's node pick (NT+pipelined, 2.106 us/t) lost 1.72x to
 *      L17_winograd's plain static schedule (1.222), and L17_rader's arena
 *      read 1.319 for a config that scored 2.904.  The mt arena is now
 *      first-touched IN PARALLEL by the static owner map (thread t touches
 *      the volumes it will process), which is the migrated steady state,
 *      and stage A races under static (dyn=0), which is what the whole
 *      panel's node results say wins.
 *   2. Race statistic: median of 5 blocked reps instead of min of 3 (the
 *      VERDICT's section 3.2: nine entries scored on their luckiest
 *      process's create-time pick; medians flip far less).
 *   3. Flat arrival-flag/release barrier for mode 2, ADOPTED FROM
 *      L17_winograd mt_r1 (its record: central atomic counter ~1.2 us at
 *      T=16 -- sixteen serialized RFOs on one line -- vs 0.3-0.4 us flat).
 *   4. Mode-2 X-phase BULK prefetch (xpf=2): pull the thread's whole column
 *      range right at the barrier exit instead of 2 chunks ahead.
 *   5. t1g re-homed after the team-size pick: the race needs t1g to exist,
 *      so it is first touched by the full-team map, but the picked team is
 *      smaller (node: nt=16), and under close binding that map leaves
 *      planes 8..16's pages homed on socket 1 while every user of them
 *      runs on socket 0.  mmap + re-touch by the picked map fixes the
 *      exposure (free()+malloc could recycle already-homed heap pages).
 * =============================================================================
 */
enum { L17MT_T1VOL = 17 * L17_T1SP }; /* padded t1 volume, doubles (87 KiB) */

static double l17_now(void);
static int l17_verbose(void);
static void l17_tune_free(fft3d_plan *p);

/* one thread's share of the plane phase: global plane index g = 17*b + x */
static __attribute__((noinline)) void
l17mt_planes(const fft3d_plan *restrict p, const double *restrict inb,
             double *restrict t1g, int g0, int g1)
{
    L17_MIX_DECLS;
    (void)nb;
    for (int g = g0; g < g1; ++g) {
        int b = g / 17, x = g % 17;
        const double *pin2 = inb + (size_t)b * 9826 + (long)x * 578;
        double *pt = t1g + (size_t)b * L17MT_T1VOL + (long)x * L17_T1SP;
        L17_MIX_GROUP(pin2, 34, pb, L17_PBROWM, 0, 0);
        L17_MIX_GROUP(pb, L17_PBROWM, pt, 34, 0, 0);
    }
}

/* one thread's share of the X pass: global slot index h = 73*b + i.
 * After the barrier most source lines are dirty in OTHER cores' caches (the
 * plane phase wrote them), so the X chunks prefetch ahead -- the X chunk's
 * dependent FMA chains cannot hide a cross-core dirty-line latency by
 * themselves.  Prefetch only, changes no bits (r9 pt mechanism, repointed
 * at coherence misses instead of L2->L1 fills).  Two shapes, raced:
 *   pf=1 (mt_r1): chunk h prefetches chunk h+2's 17 rows (~2 chunks of
 *        latency hiding, bounded MLP).
 *   pf=2 (mt_r2): BULK -- the thread's whole column range is prefetched
 *        back-to-back right at the barrier exit, so the cross-core pulls
 *        overlap EACH OTHER (aggregate MLP) instead of only two chunks
 *        ahead.  This is the "overlap the pull, bulk, not per-chunk" item
 *        from the r1 record, and the r1 VERDICT's named L=17 fix.  Capped
 *        at 24 chunks (~26 KiB, half of L1) so larger small-batch ranges
 *        do not flush their own L1; past the cap pf=1's lookahead resumes. */
static __attribute__((noinline)) void
l17mt_xrange(const fft3d_plan *restrict p, const double *restrict t1g,
             double *restrict outb, int h0, int h1, int pf)
{
    L17_MIX_DECLS;
    (void)nb;
    int hb = h0; /* chunks below hb are already prefetched (bulk) */
    if (pf == 2) {
        hb = h1 < h0 + 24 ? h1 : h0 + 24;
        for (int h = h0; h < hb; ++h) {
            int b = h / 73, i = h % 73;
            const double *pfv = t1g + (size_t)b * L17MT_T1VOL +
                                2 * (i < 72 ? 4L * i : 287L);
            for (int r = 0; r < 17; ++r)
                __builtin_prefetch(pfv + (long)r * L17_T1SP, 0, 3);
        }
    }
    for (int h = h0; h < h1; ++h) {
        int b = h / 73, i = h % 73;
        const double *t1v = t1g + (size_t)b * L17MT_T1VOL;
        double *vout = outb + (size_t)b * 9826;
        if (pf && h + 2 < h1 && h + 2 >= hb) {
            int b2 = (h + 2) / 73, i2 = (h + 2) % 73;
            const double *pfv = t1g + (size_t)b2 * L17MT_T1VOL +
                                2 * (i2 < 72 ? 4L * i2 : 287L);
            for (int r = 0; r < 17; ++r)
                __builtin_prefetch(pfv + (long)r * L17_T1SP, 0, 3);
        }
        if (i < 72) {
            long f0 = 4L * i;
            chunk17n_w4(t1v + 2 * f0, L17_T1SP, vout + 2 * f0, 2, 578,
                        cn8, sn8, sc, K0, K1, K2, K3, K4, K5, K6, K7,
                        0, 0, 1, 0);
        } else {
            chunk17n_w2(t1v + 2 * 287, L17_T1SP, vout + 2 * 287, 2, 578,
                        cn4, sn4, sc, Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7,
                        0, 0, 1, 0);
        }
    }
}


/* =============================================================================
 * ROUND mt_r3: THE STREAMING-REGIME ENGINE, ADOPTED FROM L17_winograd.
 *
 * The mt_r2 leaderboard and VERDICT settled B=4096: this entry's dense
 * X-first schedule collapsed to 82 GB/s on the node (2.858 us/t, both
 * rounds, stable) while L17_winograd's rotating-pass fused engine ran the
 * identical cell at ~193 GB/s effective (1.220 us/t, both rounds), under
 * the same static split, same team, plain stores.  ~2x of the gap is
 * schedule, not machine (VERDICT mt_r2 section 4 item 4), and the VERDICT's
 * L=17 directive is to MERGE: dense conj-symmetric X-first below the
 * aggregate-cache threshold, the Winograd rotating-pass engine above it,
 * "with the threshold set from the working set rather than from an arena"
 * (this entry's create-time arena misranked the streaming cell twice --
 * r1 NT+pipelined 2.106, r2 plain X-first 2.858, both priced as winners).
 *
 * Everything from here to the end of wg_fused23_h4 is copied verbatim from
 * impl/L17_winograd.c (its 17-point module: Rader/quotient-derived cyclic +
 * negacyclic correlations, kernels A/B/E/H, the kx-blocked fused pass 2+3,
 * and the split-free i4 pass 1 that L17_rader's node runs picked at batch),
 * with two local changes, marked below: fused23_h4's plan indirection is
 * replaced by explicit scratch/table arguments, and its pf/pfw/CLWB hooks
 * are dropped (the node picked pf=0 pfw=0 cw=0 in every winning process;
 * they are hints only).  The engine produces a DIFFERENT (correct) rounding
 * fixed point from the dense classes; it is selected by a DETERMINISTIC
 * working-set gate, never by a wall-clock race, so process-to-process
 * repeatability is preserved by construction.
 * ============================================================================= */
#define SP 296          /* winograd's padded spectator stride, doubles */
typedef double    v8  __attribute__((vector_size(64)));
typedef double    v8u __attribute__((vector_size(64), aligned(8)));
typedef long long i8v __attribute__((vector_size(64)));
typedef double    v4  __attribute__((vector_size(32)));
typedef double    v4u __attribute__((vector_size(32), aligned(8)));
typedef long long i4v __attribute__((vector_size(32)));
typedef double    v2u __attribute__((vector_size(16), aligned(8)));

#define BCAST(TY,x) ((TY){0} + (x))   /* scalar -> lane-broadcast, folded at compile time */
#define SH8(a,b,...) ((v8)__builtin_shuffle((v8)(a),(v8)(b),(i8v){__VA_ARGS__}))
#define SH4(a,b,...) ((v4)__builtin_shuffle((v4)(a),(v4)(b),(i4v){__VA_ARGS__}))
#define DEF_K17_A(SUF, T) \
static inline __attribute__((always_inline)) void k17_##SUF(const T *xr, const T *xi, T *yr, T *yi) \
{ \
    T pq[8], vv[16], cc[16]; \
    T a0r, a1r, a2r, a3r, a0i, a1i, a2i, a3i, dr, di, u0r,u0i,u1r,u1i; \
    dr = xr[0]; di = xi[0]; \
    u0r = xr[1] + xr[16];  u0i = xi[1] + xi[16]; \
    u1r = xr[4] + xr[13];  u1i = xi[4] + xi[13]; \
    vv[0] = xr[1] - xr[16];  vv[1] = xi[1] - xi[16]; \
    vv[8] = xr[13] - xr[4];  vv[9] = xi[13] - xi[4]; \
    { const T pr = u0r + u1r, pi = u0i + u1i; \
      pq[0] = u0r - u1r;  pq[1] = u0i - u1i; \
      dr = dr + pr;  di = di + pi; \
      a0r = xr[0] + pr*5.12370294433828866e-01; \
      a1r = xr[0] + pr*8.60376828522276954e-02; \
      a2r = xr[0] + pr*-1.21982091231621334e-01; \
      a3r = xr[0] + pr*-7.26425886054435255e-01; \
      a0i = xi[0] + pi*5.12370294433828866e-01; \
      a1i = xi[0] + pi*8.60376828522276954e-02; \
      a2i = xi[0] + pi*-1.21982091231621334e-01; \
      a3i = xi[0] + pi*-7.26425886054435255e-01; \
    } \
    u0r = xr[3] + xr[14];  u0i = xi[3] + xi[14]; \
    u1r = xr[5] + xr[12];  u1i = xi[5] + xi[12]; \
    vv[2] = xr[3] - xr[14];  vv[3] = xi[3] - xi[14]; \
    vv[10] = xr[5] - xr[12];  vv[11] = xi[5] - xi[12]; \
    { const T pr = u0r + u1r, pi = u0i + u1i; \
      pq[2] = u0r - u1r;  pq[3] = u0i - u1i; \
      dr = dr + pr;  di = di + pi; \
      a0r = a0r + pr*8.60376828522276954e-02; \
      a1r = a1r + pr*-1.21982091231621334e-01; \
      a2r = a2r + pr*-7.26425886054435255e-01; \
      a3r = a3r + pr*5.12370294433828866e-01; \
      a0i = a0i + pi*8.60376828522276954e-02; \
      a1i = a1i + pi*-1.21982091231621334e-01; \
      a2i = a2i + pi*-7.26425886054435255e-01; \
      a3i = a3i + pi*5.12370294433828866e-01; \
    } \
    u0r = xr[8] + xr[9];  u0i = xi[8] + xi[9]; \
    u1r = xr[2] + xr[15];  u1i = xi[2] + xi[15]; \
    vv[4] = xr[9] - xr[8];  vv[5] = xi[9] - xi[8]; \
    vv[12] = xr[15] - xr[2];  vv[13] = xi[15] - xi[2]; \
    { const T pr = u0r + u1r, pi = u0i + u1i; \
      pq[4] = u0r - u1r;  pq[5] = u0i - u1i; \
      dr = dr + pr;  di = di + pi; \
      a0r = a0r + pr*-1.21982091231621334e-01; \
      a1r = a1r + pr*-7.26425886054435255e-01; \
      a2r = a2r + pr*5.12370294433828866e-01; \
      a3r = a3r + pr*8.60376828522276954e-02; \
      a0i = a0i + pi*-1.21982091231621334e-01; \
      a1i = a1i + pi*-7.26425886054435255e-01; \
      a2i = a2i + pi*5.12370294433828866e-01; \
      a3i = a3i + pi*8.60376828522276954e-02; \
    } \
    u0r = xr[7] + xr[10];  u0i = xi[7] + xi[10]; \
    u1r = xr[6] + xr[11];  u1i = xi[6] + xi[11]; \
    vv[6] = xr[10] - xr[7];  vv[7] = xi[10] - xi[7]; \
    vv[14] = xr[11] - xr[6];  vv[15] = xi[11] - xi[6]; \
    { const T pr = u0r + u1r, pi = u0i + u1i; \
      pq[6] = u0r - u1r;  pq[7] = u0i - u1i; \
      dr = dr + pr;  di = di + pi; \
      a0r = a0r + pr*-7.26425886054435255e-01; \
      a1r = a1r + pr*5.12370294433828866e-01; \
      a2r = a2r + pr*8.60376828522276954e-02; \
      a3r = a3r + pr*-1.21982091231621334e-01; \
      a0i = a0i + pi*-7.26425886054435255e-01; \
      a1i = a1i + pi*5.12370294433828866e-01; \
      a2i = a2i + pi*8.60376828522276954e-02; \
      a3i = a3i + pi*-1.21982091231621334e-01; \
    } \
    yr[0] = dr;  yi[0] = di; \
    cc[0] = a0r; cc[1] = a0i; cc[2] = a1r; cc[3] = a1i; \
    cc[4] = a2r; cc[5] = a2i; cc[6] = a3r; cc[7] = a3i; \
    { const T qr = pq[0], qi = pq[1]; \
      a0r = qr*4.20101934970526891e-01; \
      a1r = qr*3.59700672924310572e-01; \
      a2r = qr*-8.60991008452280493e-01; \
      a3r = qr*-1.23791249675178877e-01; \
      a0i = qi*4.20101934970526891e-01; \
      a1i = qi*3.59700672924310572e-01; \
      a2i = qi*-8.60991008452280493e-01; \
      a3i = qi*-1.23791249675178877e-01; \
    } \
    { const T qr = pq[2], qi = pq[3]; \
      a0r = a0r + qr*3.59700672924310572e-01; \
      a1r = a1r + qr*-8.60991008452280493e-01; \
      a2r = a2r + qr*-1.23791249675178877e-01; \
      a3r = a3r + qr*-4.20101934970526891e-01; \
      a0i = a0i + qi*3.59700672924310572e-01; \
      a1i = a1i + qi*-8.60991008452280493e-01; \
      a2i = a2i + qi*-1.23791249675178877e-01; \
      a3i = a3i + qi*-4.20101934970526891e-01; \
    } \
    { const T qr = pq[4], qi = pq[5]; \
      a0r = a0r + qr*-8.60991008452280493e-01; \
      a1r = a1r + qr*-1.23791249675178877e-01; \
      a2r = a2r + qr*-4.20101934970526891e-01; \
      a3r = a3r + qr*-3.59700672924310572e-01; \
      a0i = a0i + qi*-8.60991008452280493e-01; \
      a1i = a1i + qi*-1.23791249675178877e-01; \
      a2i = a2i + qi*-4.20101934970526891e-01; \
      a3i = a3i + qi*-3.59700672924310572e-01; \
    } \
    { const T qr = pq[6], qi = pq[7]; \
      a0r = a0r + qr*-1.23791249675178877e-01; \
      a1r = a1r + qr*-4.20101934970526891e-01; \
      a2r = a2r + qr*-3.59700672924310572e-01; \
      a3r = a3r + qr*8.60991008452280493e-01; \
      a0i = a0i + qi*-1.23791249675178877e-01; \
      a1i = a1i + qi*-4.20101934970526891e-01; \
      a2i = a2i + qi*-3.59700672924310572e-01; \
      a3i = a3i + qi*8.60991008452280493e-01; \
    } \
    { const T ta = cc[0], tb = cc[1]; \
      cc[0] = ta + a0r;  cc[1] = tb + a0i; \
      cc[8] = ta - a0r;  cc[9] = tb - a0i; } \
    { const T ta = cc[2], tb = cc[3]; \
      cc[2] = ta + a1r;  cc[3] = tb + a1i; \
      cc[10] = ta - a1r;  cc[11] = tb - a1i; } \
    { const T ta = cc[4], tb = cc[5]; \
      cc[4] = ta + a2r;  cc[5] = tb + a2i; \
      cc[12] = ta - a2r;  cc[13] = tb - a2i; } \
    { const T ta = cc[6], tb = cc[7]; \
      cc[6] = ta + a3r;  cc[7] = tb + a3i; \
      cc[14] = ta - a3r;  cc[15] = tb - a3i; } \
    { const T tvr = vv[0], tvi = vv[1]; \
      a0r = tvr*3.61241666187152921e-01; \
      a1r = tvr*8.95163291355062340e-01; \
      a2r = tvr*-1.83749517816570340e-01; \
      a3r = tvr*-5.26432162877355836e-01; \
      a0i = tvi*3.61241666187152921e-01; \
      a1i = tvi*8.95163291355062340e-01; \
      a2i = tvi*-1.83749517816570340e-01; \
      a3i = tvi*-5.26432162877355836e-01; \
    } \
    { const T tvr = vv[2], tvi = vv[3]; \
      a0r = a0r + tvr*8.95163291355062340e-01; \
      a1r = a1r + tvr*-1.83749517816570340e-01; \
      a2r = a2r + tvr*-5.26432162877355836e-01; \
      a3r = a3r + tvr*-9.95734176295034468e-01; \
      a0i = a0i + tvi*8.95163291355062340e-01; \
      a1i = a1i + tvi*-1.83749517816570340e-01; \
      a2i = a2i + tvi*-5.26432162877355836e-01; \
      a3i = a3i + tvi*-9.95734176295034468e-01; \
    } \
    { const T tvr = vv[4], tvi = vv[5]; \
      a0r = a0r + tvr*-1.83749517816570340e-01; \
      a1r = a1r + tvr*-5.26432162877355836e-01; \
      a2r = a2r + tvr*-9.95734176295034468e-01; \
      a3r = a3r + tvr*9.61825643172819045e-01; \
      a0i = a0i + tvi*-1.83749517816570340e-01; \
      a1i = a1i + tvi*-5.26432162877355836e-01; \
      a2i = a2i + tvi*-9.95734176295034468e-01; \
      a3i = a3i + tvi*9.61825643172819045e-01; \
    } \
    { const T tvr = vv[6], tvi = vv[7]; \
      a0r = a0r + tvr*-5.26432162877355836e-01; \
      a1r = a1r + tvr*-9.95734176295034468e-01; \
      a2r = a2r + tvr*9.61825643172819045e-01; \
      a3r = a3r + tvr*-6.73695643646557207e-01; \
      a0i = a0i + tvi*-5.26432162877355836e-01; \
      a1i = a1i + tvi*-9.95734176295034468e-01; \
      a2i = a2i + tvi*9.61825643172819045e-01; \
      a3i = a3i + tvi*-6.73695643646557207e-01; \
    } \
    { const T tvr = vv[8], tvi = vv[9]; \
      a0r = a0r + tvr*-9.95734176295034468e-01; \
      a1r = a1r + tvr*9.61825643172819045e-01; \
      a2r = a2r + tvr*-6.73695643646557207e-01; \
      a3r = a3r + tvr*-7.98017227280239494e-01; \
      a0i = a0i + tvi*-9.95734176295034468e-01; \
      a1i = a1i + tvi*9.61825643172819045e-01; \
      a2i = a2i + tvi*-6.73695643646557207e-01; \
      a3i = a3i + tvi*-7.98017227280239494e-01; \
    } \
    { const T tvr = vv[10], tvi = vv[11]; \
      a0r = a0r + tvr*9.61825643172819045e-01; \
      a1r = a1r + tvr*-6.73695643646557207e-01; \
      a2r = a2r + tvr*-7.98017227280239494e-01; \
      a3r = a3r + tvr*-3.61241666187152921e-01; \
      a0i = a0i + tvi*9.61825643172819045e-01; \
      a1i = a1i + tvi*-6.73695643646557207e-01; \
      a2i = a2i + tvi*-7.98017227280239494e-01; \
      a3i = a3i + tvi*-3.61241666187152921e-01; \
    } \
    { const T tvr = vv[12], tvi = vv[13]; \
      a0r = a0r + tvr*-6.73695643646557207e-01; \
      a1r = a1r + tvr*-7.98017227280239494e-01; \
      a2r = a2r + tvr*-3.61241666187152921e-01; \
      a3r = a3r + tvr*-8.95163291355062340e-01; \
      a0i = a0i + tvi*-6.73695643646557207e-01; \
      a1i = a1i + tvi*-7.98017227280239494e-01; \
      a2i = a2i + tvi*-3.61241666187152921e-01; \
      a3i = a3i + tvi*-8.95163291355062340e-01; \
    } \
    { const T tvr = vv[14], tvi = vv[15]; \
      a0r = a0r + tvr*-7.98017227280239494e-01; \
      a1r = a1r + tvr*-3.61241666187152921e-01; \
      a2r = a2r + tvr*-8.95163291355062340e-01; \
      a3r = a3r + tvr*1.83749517816570340e-01; \
      a0i = a0i + tvi*-7.98017227280239494e-01; \
      a1i = a1i + tvi*-3.61241666187152921e-01; \
      a2i = a2i + tvi*-8.95163291355062340e-01; \
      a3i = a3i + tvi*1.83749517816570340e-01; \
    } \
    { const T tcr = cc[0], tci = cc[1]; \
      yr[1] = tcr + a0i;  yi[1] = tci - a0r; \
      yr[16] = tcr - a0i;  yi[16] = tci + a0r; } \
    { const T tcr = cc[2], tci = cc[3]; \
      yr[3] = tcr + a1i;  yi[3] = tci - a1r; \
      yr[14] = tcr - a1i;  yi[14] = tci + a1r; } \
    { const T tcr = cc[4], tci = cc[5]; \
      yr[8] = tcr - a2i;  yi[8] = tci + a2r; \
      yr[9] = tcr + a2i;  yi[9] = tci - a2r; } \
    { const T tcr = cc[6], tci = cc[7]; \
      yr[7] = tcr - a3i;  yi[7] = tci + a3r; \
      yr[10] = tcr + a3i;  yi[10] = tci - a3r; } \
    { const T tvr = vv[0], tvi = vv[1]; \
      a0r = tvr*-9.95734176295034468e-01; \
      a1r = tvr*9.61825643172819045e-01; \
      a2r = tvr*-6.73695643646557207e-01; \
      a3r = tvr*-7.98017227280239494e-01; \
      a0i = tvi*-9.95734176295034468e-01; \
      a1i = tvi*9.61825643172819045e-01; \
      a2i = tvi*-6.73695643646557207e-01; \
      a3i = tvi*-7.98017227280239494e-01; \
    } \
    { const T tvr = vv[2], tvi = vv[3]; \
      a0r = a0r + tvr*9.61825643172819045e-01; \
      a1r = a1r + tvr*-6.73695643646557207e-01; \
      a2r = a2r + tvr*-7.98017227280239494e-01; \
      a3r = a3r + tvr*-3.61241666187152921e-01; \
      a0i = a0i + tvi*9.61825643172819045e-01; \
      a1i = a1i + tvi*-6.73695643646557207e-01; \
      a2i = a2i + tvi*-7.98017227280239494e-01; \
      a3i = a3i + tvi*-3.61241666187152921e-01; \
    } \
    { const T tvr = vv[4], tvi = vv[5]; \
      a0r = a0r + tvr*-6.73695643646557207e-01; \
      a1r = a1r + tvr*-7.98017227280239494e-01; \
      a2r = a2r + tvr*-3.61241666187152921e-01; \
      a3r = a3r + tvr*-8.95163291355062340e-01; \
      a0i = a0i + tvi*-6.73695643646557207e-01; \
      a1i = a1i + tvi*-7.98017227280239494e-01; \
      a2i = a2i + tvi*-3.61241666187152921e-01; \
      a3i = a3i + tvi*-8.95163291355062340e-01; \
    } \
    { const T tvr = vv[6], tvi = vv[7]; \
      a0r = a0r + tvr*-7.98017227280239494e-01; \
      a1r = a1r + tvr*-3.61241666187152921e-01; \
      a2r = a2r + tvr*-8.95163291355062340e-01; \
      a3r = a3r + tvr*1.83749517816570340e-01; \
      a0i = a0i + tvi*-7.98017227280239494e-01; \
      a1i = a1i + tvi*-3.61241666187152921e-01; \
      a2i = a2i + tvi*-8.95163291355062340e-01; \
      a3i = a3i + tvi*1.83749517816570340e-01; \
    } \
    { const T tvr = vv[8], tvi = vv[9]; \
      a0r = a0r + tvr*-3.61241666187152921e-01; \
      a1r = a1r + tvr*-8.95163291355062340e-01; \
      a2r = a2r + tvr*1.83749517816570340e-01; \
      a3r = a3r + tvr*5.26432162877355836e-01; \
      a0i = a0i + tvi*-3.61241666187152921e-01; \
      a1i = a1i + tvi*-8.95163291355062340e-01; \
      a2i = a2i + tvi*1.83749517816570340e-01; \
      a3i = a3i + tvi*5.26432162877355836e-01; \
    } \
    { const T tvr = vv[10], tvi = vv[11]; \
      a0r = a0r + tvr*-8.95163291355062340e-01; \
      a1r = a1r + tvr*1.83749517816570340e-01; \
      a2r = a2r + tvr*5.26432162877355836e-01; \
      a3r = a3r + tvr*9.95734176295034468e-01; \
      a0i = a0i + tvi*-8.95163291355062340e-01; \
      a1i = a1i + tvi*1.83749517816570340e-01; \
      a2i = a2i + tvi*5.26432162877355836e-01; \
      a3i = a3i + tvi*9.95734176295034468e-01; \
    } \
    { const T tvr = vv[12], tvi = vv[13]; \
      a0r = a0r + tvr*1.83749517816570340e-01; \
      a1r = a1r + tvr*5.26432162877355836e-01; \
      a2r = a2r + tvr*9.95734176295034468e-01; \
      a3r = a3r + tvr*-9.61825643172819045e-01; \
      a0i = a0i + tvi*1.83749517816570340e-01; \
      a1i = a1i + tvi*5.26432162877355836e-01; \
      a2i = a2i + tvi*9.95734176295034468e-01; \
      a3i = a3i + tvi*-9.61825643172819045e-01; \
    } \
    { const T tvr = vv[14], tvi = vv[15]; \
      a0r = a0r + tvr*5.26432162877355836e-01; \
      a1r = a1r + tvr*9.95734176295034468e-01; \
      a2r = a2r + tvr*-9.61825643172819045e-01; \
      a3r = a3r + tvr*6.73695643646557207e-01; \
      a0i = a0i + tvi*5.26432162877355836e-01; \
      a1i = a1i + tvi*9.95734176295034468e-01; \
      a2i = a2i + tvi*-9.61825643172819045e-01; \
      a3i = a3i + tvi*6.73695643646557207e-01; \
    } \
    { const T tcr = cc[8], tci = cc[9]; \
      yr[4] = tcr - a0i;  yi[4] = tci + a0r; \
      yr[13] = tcr + a0i;  yi[13] = tci - a0r; } \
    { const T tcr = cc[10], tci = cc[11]; \
      yr[5] = tcr + a1i;  yi[5] = tci - a1r; \
      yr[12] = tcr - a1i;  yi[12] = tci + a1r; } \
    { const T tcr = cc[12], tci = cc[13]; \
      yr[2] = tcr - a2i;  yi[2] = tci + a2r; \
      yr[15] = tcr + a2i;  yi[15] = tci - a2r; } \
    { const T tcr = cc[14], tci = cc[15]; \
      yr[6] = tcr - a3i;  yi[6] = tci + a3r; \
      yr[11] = tcr + a3i;  yi[11] = tci - a3r; } \
}
#define DEF_K17_B(SUF, T) \
static inline __attribute__((always_inline)) void k17_##SUF(const T *xr, const T *xi, T *yr, T *yi) \
{ \
    T vv[16], cc[16]; \
    T a0r, a1r, a2r, a3r, a0i, a1i, a2i, a3i, b0r, b1r, b2r, b3r, b0i, b1i, b2i, b3i, dr, di, u0r,u0i,u1r,u1i; \
    dr = xr[0]; di = xi[0]; \
    u0r = xr[1] + xr[16];  u0i = xi[1] + xi[16]; \
    u1r = xr[4] + xr[13];  u1i = xi[4] + xi[13]; \
    vv[0] = xr[1] - xr[16];  vv[1] = xi[1] - xi[16]; \
    vv[8] = xr[13] - xr[4];  vv[9] = xi[13] - xi[4]; \
    { const T pr = u0r + u1r, pi = u0i + u1i; \
      const T qr = u0r - u1r, qi = u0i - u1i; \
      dr = dr + pr;  di = di + pi; \
      a0r = xr[0] + pr*5.12370294433828866e-01; \
      a1r = xr[0] + pr*8.60376828522276954e-02; \
      a2r = xr[0] + pr*-1.21982091231621334e-01; \
      a3r = xr[0] + pr*-7.26425886054435255e-01; \
      a0i = xi[0] + pi*5.12370294433828866e-01; \
      a1i = xi[0] + pi*8.60376828522276954e-02; \
      a2i = xi[0] + pi*-1.21982091231621334e-01; \
      a3i = xi[0] + pi*-7.26425886054435255e-01; \
      b0r = qr*4.20101934970526891e-01; \
      b1r = qr*3.59700672924310572e-01; \
      b2r = qr*-8.60991008452280493e-01; \
      b3r = qr*-1.23791249675178877e-01; \
      b0i = qi*4.20101934970526891e-01; \
      b1i = qi*3.59700672924310572e-01; \
      b2i = qi*-8.60991008452280493e-01; \
      b3i = qi*-1.23791249675178877e-01; \
    } \
    u0r = xr[3] + xr[14];  u0i = xi[3] + xi[14]; \
    u1r = xr[5] + xr[12];  u1i = xi[5] + xi[12]; \
    vv[2] = xr[3] - xr[14];  vv[3] = xi[3] - xi[14]; \
    vv[10] = xr[5] - xr[12];  vv[11] = xi[5] - xi[12]; \
    { const T pr = u0r + u1r, pi = u0i + u1i; \
      const T qr = u0r - u1r, qi = u0i - u1i; \
      dr = dr + pr;  di = di + pi; \
      a0r = a0r + pr*8.60376828522276954e-02; \
      a1r = a1r + pr*-1.21982091231621334e-01; \
      a2r = a2r + pr*-7.26425886054435255e-01; \
      a3r = a3r + pr*5.12370294433828866e-01; \
      a0i = a0i + pi*8.60376828522276954e-02; \
      a1i = a1i + pi*-1.21982091231621334e-01; \
      a2i = a2i + pi*-7.26425886054435255e-01; \
      a3i = a3i + pi*5.12370294433828866e-01; \
      b0r = b0r + qr*3.59700672924310572e-01; \
      b1r = b1r + qr*-8.60991008452280493e-01; \
      b2r = b2r + qr*-1.23791249675178877e-01; \
      b3r = b3r + qr*-4.20101934970526891e-01; \
      b0i = b0i + qi*3.59700672924310572e-01; \
      b1i = b1i + qi*-8.60991008452280493e-01; \
      b2i = b2i + qi*-1.23791249675178877e-01; \
      b3i = b3i + qi*-4.20101934970526891e-01; \
    } \
    u0r = xr[8] + xr[9];  u0i = xi[8] + xi[9]; \
    u1r = xr[2] + xr[15];  u1i = xi[2] + xi[15]; \
    vv[4] = xr[9] - xr[8];  vv[5] = xi[9] - xi[8]; \
    vv[12] = xr[15] - xr[2];  vv[13] = xi[15] - xi[2]; \
    { const T pr = u0r + u1r, pi = u0i + u1i; \
      const T qr = u0r - u1r, qi = u0i - u1i; \
      dr = dr + pr;  di = di + pi; \
      a0r = a0r + pr*-1.21982091231621334e-01; \
      a1r = a1r + pr*-7.26425886054435255e-01; \
      a2r = a2r + pr*5.12370294433828866e-01; \
      a3r = a3r + pr*8.60376828522276954e-02; \
      a0i = a0i + pi*-1.21982091231621334e-01; \
      a1i = a1i + pi*-7.26425886054435255e-01; \
      a2i = a2i + pi*5.12370294433828866e-01; \
      a3i = a3i + pi*8.60376828522276954e-02; \
      b0r = b0r + qr*-8.60991008452280493e-01; \
      b1r = b1r + qr*-1.23791249675178877e-01; \
      b2r = b2r + qr*-4.20101934970526891e-01; \
      b3r = b3r + qr*-3.59700672924310572e-01; \
      b0i = b0i + qi*-8.60991008452280493e-01; \
      b1i = b1i + qi*-1.23791249675178877e-01; \
      b2i = b2i + qi*-4.20101934970526891e-01; \
      b3i = b3i + qi*-3.59700672924310572e-01; \
    } \
    u0r = xr[7] + xr[10];  u0i = xi[7] + xi[10]; \
    u1r = xr[6] + xr[11];  u1i = xi[6] + xi[11]; \
    vv[6] = xr[10] - xr[7];  vv[7] = xi[10] - xi[7]; \
    vv[14] = xr[11] - xr[6];  vv[15] = xi[11] - xi[6]; \
    { const T pr = u0r + u1r, pi = u0i + u1i; \
      const T qr = u0r - u1r, qi = u0i - u1i; \
      dr = dr + pr;  di = di + pi; \
      a0r = a0r + pr*-7.26425886054435255e-01; \
      a1r = a1r + pr*5.12370294433828866e-01; \
      a2r = a2r + pr*8.60376828522276954e-02; \
      a3r = a3r + pr*-1.21982091231621334e-01; \
      a0i = a0i + pi*-7.26425886054435255e-01; \
      a1i = a1i + pi*5.12370294433828866e-01; \
      a2i = a2i + pi*8.60376828522276954e-02; \
      a3i = a3i + pi*-1.21982091231621334e-01; \
      b0r = b0r + qr*-1.23791249675178877e-01; \
      b1r = b1r + qr*-4.20101934970526891e-01; \
      b2r = b2r + qr*-3.59700672924310572e-01; \
      b3r = b3r + qr*8.60991008452280493e-01; \
      b0i = b0i + qi*-1.23791249675178877e-01; \
      b1i = b1i + qi*-4.20101934970526891e-01; \
      b2i = b2i + qi*-3.59700672924310572e-01; \
      b3i = b3i + qi*8.60991008452280493e-01; \
    } \
    yr[0] = dr;  yi[0] = di; \
    cc[0] = a0r + b0r;  cc[1] = a0i + b0i; \
    cc[8] = a0r - b0r;  cc[9] = a0i - b0i; \
    cc[2] = a1r + b1r;  cc[3] = a1i + b1i; \
    cc[10] = a1r - b1r;  cc[11] = a1i - b1i; \
    cc[4] = a2r + b2r;  cc[5] = a2i + b2i; \
    cc[12] = a2r - b2r;  cc[13] = a2i - b2i; \
    cc[6] = a3r + b3r;  cc[7] = a3i + b3i; \
    cc[14] = a3r - b3r;  cc[15] = a3i - b3i; \
    { const T tvr = vv[0], tvi = vv[1]; \
      a0r = tvr*3.61241666187152921e-01; \
      a1r = tvr*8.95163291355062340e-01; \
      a2r = tvr*-1.83749517816570340e-01; \
      a3r = tvr*-5.26432162877355836e-01; \
      a0i = tvi*3.61241666187152921e-01; \
      a1i = tvi*8.95163291355062340e-01; \
      a2i = tvi*-1.83749517816570340e-01; \
      a3i = tvi*-5.26432162877355836e-01; \
    } \
    { const T tvr = vv[2], tvi = vv[3]; \
      a0r = a0r + tvr*8.95163291355062340e-01; \
      a1r = a1r + tvr*-1.83749517816570340e-01; \
      a2r = a2r + tvr*-5.26432162877355836e-01; \
      a3r = a3r + tvr*-9.95734176295034468e-01; \
      a0i = a0i + tvi*8.95163291355062340e-01; \
      a1i = a1i + tvi*-1.83749517816570340e-01; \
      a2i = a2i + tvi*-5.26432162877355836e-01; \
      a3i = a3i + tvi*-9.95734176295034468e-01; \
    } \
    { const T tvr = vv[4], tvi = vv[5]; \
      a0r = a0r + tvr*-1.83749517816570340e-01; \
      a1r = a1r + tvr*-5.26432162877355836e-01; \
      a2r = a2r + tvr*-9.95734176295034468e-01; \
      a3r = a3r + tvr*9.61825643172819045e-01; \
      a0i = a0i + tvi*-1.83749517816570340e-01; \
      a1i = a1i + tvi*-5.26432162877355836e-01; \
      a2i = a2i + tvi*-9.95734176295034468e-01; \
      a3i = a3i + tvi*9.61825643172819045e-01; \
    } \
    { const T tvr = vv[6], tvi = vv[7]; \
      a0r = a0r + tvr*-5.26432162877355836e-01; \
      a1r = a1r + tvr*-9.95734176295034468e-01; \
      a2r = a2r + tvr*9.61825643172819045e-01; \
      a3r = a3r + tvr*-6.73695643646557207e-01; \
      a0i = a0i + tvi*-5.26432162877355836e-01; \
      a1i = a1i + tvi*-9.95734176295034468e-01; \
      a2i = a2i + tvi*9.61825643172819045e-01; \
      a3i = a3i + tvi*-6.73695643646557207e-01; \
    } \
    { const T tvr = vv[8], tvi = vv[9]; \
      a0r = a0r + tvr*-9.95734176295034468e-01; \
      a1r = a1r + tvr*9.61825643172819045e-01; \
      a2r = a2r + tvr*-6.73695643646557207e-01; \
      a3r = a3r + tvr*-7.98017227280239494e-01; \
      a0i = a0i + tvi*-9.95734176295034468e-01; \
      a1i = a1i + tvi*9.61825643172819045e-01; \
      a2i = a2i + tvi*-6.73695643646557207e-01; \
      a3i = a3i + tvi*-7.98017227280239494e-01; \
    } \
    { const T tvr = vv[10], tvi = vv[11]; \
      a0r = a0r + tvr*9.61825643172819045e-01; \
      a1r = a1r + tvr*-6.73695643646557207e-01; \
      a2r = a2r + tvr*-7.98017227280239494e-01; \
      a3r = a3r + tvr*-3.61241666187152921e-01; \
      a0i = a0i + tvi*9.61825643172819045e-01; \
      a1i = a1i + tvi*-6.73695643646557207e-01; \
      a2i = a2i + tvi*-7.98017227280239494e-01; \
      a3i = a3i + tvi*-3.61241666187152921e-01; \
    } \
    { const T tvr = vv[12], tvi = vv[13]; \
      a0r = a0r + tvr*-6.73695643646557207e-01; \
      a1r = a1r + tvr*-7.98017227280239494e-01; \
      a2r = a2r + tvr*-3.61241666187152921e-01; \
      a3r = a3r + tvr*-8.95163291355062340e-01; \
      a0i = a0i + tvi*-6.73695643646557207e-01; \
      a1i = a1i + tvi*-7.98017227280239494e-01; \
      a2i = a2i + tvi*-3.61241666187152921e-01; \
      a3i = a3i + tvi*-8.95163291355062340e-01; \
    } \
    { const T tvr = vv[14], tvi = vv[15]; \
      a0r = a0r + tvr*-7.98017227280239494e-01; \
      a1r = a1r + tvr*-3.61241666187152921e-01; \
      a2r = a2r + tvr*-8.95163291355062340e-01; \
      a3r = a3r + tvr*1.83749517816570340e-01; \
      a0i = a0i + tvi*-7.98017227280239494e-01; \
      a1i = a1i + tvi*-3.61241666187152921e-01; \
      a2i = a2i + tvi*-8.95163291355062340e-01; \
      a3i = a3i + tvi*1.83749517816570340e-01; \
    } \
    { const T tcr = cc[0], tci = cc[1]; \
      yr[1] = tcr + a0i;  yi[1] = tci - a0r; \
      yr[16] = tcr - a0i;  yi[16] = tci + a0r; } \
    { const T tcr = cc[2], tci = cc[3]; \
      yr[3] = tcr + a1i;  yi[3] = tci - a1r; \
      yr[14] = tcr - a1i;  yi[14] = tci + a1r; } \
    { const T tcr = cc[4], tci = cc[5]; \
      yr[8] = tcr - a2i;  yi[8] = tci + a2r; \
      yr[9] = tcr + a2i;  yi[9] = tci - a2r; } \
    { const T tcr = cc[6], tci = cc[7]; \
      yr[7] = tcr - a3i;  yi[7] = tci + a3r; \
      yr[10] = tcr + a3i;  yi[10] = tci - a3r; } \
    { const T tvr = vv[0], tvi = vv[1]; \
      a0r = tvr*-9.95734176295034468e-01; \
      a1r = tvr*9.61825643172819045e-01; \
      a2r = tvr*-6.73695643646557207e-01; \
      a3r = tvr*-7.98017227280239494e-01; \
      a0i = tvi*-9.95734176295034468e-01; \
      a1i = tvi*9.61825643172819045e-01; \
      a2i = tvi*-6.73695643646557207e-01; \
      a3i = tvi*-7.98017227280239494e-01; \
    } \
    { const T tvr = vv[2], tvi = vv[3]; \
      a0r = a0r + tvr*9.61825643172819045e-01; \
      a1r = a1r + tvr*-6.73695643646557207e-01; \
      a2r = a2r + tvr*-7.98017227280239494e-01; \
      a3r = a3r + tvr*-3.61241666187152921e-01; \
      a0i = a0i + tvi*9.61825643172819045e-01; \
      a1i = a1i + tvi*-6.73695643646557207e-01; \
      a2i = a2i + tvi*-7.98017227280239494e-01; \
      a3i = a3i + tvi*-3.61241666187152921e-01; \
    } \
    { const T tvr = vv[4], tvi = vv[5]; \
      a0r = a0r + tvr*-6.73695643646557207e-01; \
      a1r = a1r + tvr*-7.98017227280239494e-01; \
      a2r = a2r + tvr*-3.61241666187152921e-01; \
      a3r = a3r + tvr*-8.95163291355062340e-01; \
      a0i = a0i + tvi*-6.73695643646557207e-01; \
      a1i = a1i + tvi*-7.98017227280239494e-01; \
      a2i = a2i + tvi*-3.61241666187152921e-01; \
      a3i = a3i + tvi*-8.95163291355062340e-01; \
    } \
    { const T tvr = vv[6], tvi = vv[7]; \
      a0r = a0r + tvr*-7.98017227280239494e-01; \
      a1r = a1r + tvr*-3.61241666187152921e-01; \
      a2r = a2r + tvr*-8.95163291355062340e-01; \
      a3r = a3r + tvr*1.83749517816570340e-01; \
      a0i = a0i + tvi*-7.98017227280239494e-01; \
      a1i = a1i + tvi*-3.61241666187152921e-01; \
      a2i = a2i + tvi*-8.95163291355062340e-01; \
      a3i = a3i + tvi*1.83749517816570340e-01; \
    } \
    { const T tvr = vv[8], tvi = vv[9]; \
      a0r = a0r + tvr*-3.61241666187152921e-01; \
      a1r = a1r + tvr*-8.95163291355062340e-01; \
      a2r = a2r + tvr*1.83749517816570340e-01; \
      a3r = a3r + tvr*5.26432162877355836e-01; \
      a0i = a0i + tvi*-3.61241666187152921e-01; \
      a1i = a1i + tvi*-8.95163291355062340e-01; \
      a2i = a2i + tvi*1.83749517816570340e-01; \
      a3i = a3i + tvi*5.26432162877355836e-01; \
    } \
    { const T tvr = vv[10], tvi = vv[11]; \
      a0r = a0r + tvr*-8.95163291355062340e-01; \
      a1r = a1r + tvr*1.83749517816570340e-01; \
      a2r = a2r + tvr*5.26432162877355836e-01; \
      a3r = a3r + tvr*9.95734176295034468e-01; \
      a0i = a0i + tvi*-8.95163291355062340e-01; \
      a1i = a1i + tvi*1.83749517816570340e-01; \
      a2i = a2i + tvi*5.26432162877355836e-01; \
      a3i = a3i + tvi*9.95734176295034468e-01; \
    } \
    { const T tvr = vv[12], tvi = vv[13]; \
      a0r = a0r + tvr*1.83749517816570340e-01; \
      a1r = a1r + tvr*5.26432162877355836e-01; \
      a2r = a2r + tvr*9.95734176295034468e-01; \
      a3r = a3r + tvr*-9.61825643172819045e-01; \
      a0i = a0i + tvi*1.83749517816570340e-01; \
      a1i = a1i + tvi*5.26432162877355836e-01; \
      a2i = a2i + tvi*9.95734176295034468e-01; \
      a3i = a3i + tvi*-9.61825643172819045e-01; \
    } \
    { const T tvr = vv[14], tvi = vv[15]; \
      a0r = a0r + tvr*5.26432162877355836e-01; \
      a1r = a1r + tvr*9.95734176295034468e-01; \
      a2r = a2r + tvr*-9.61825643172819045e-01; \
      a3r = a3r + tvr*6.73695643646557207e-01; \
      a0i = a0i + tvi*5.26432162877355836e-01; \
      a1i = a1i + tvi*9.95734176295034468e-01; \
      a2i = a2i + tvi*-9.61825643172819045e-01; \
      a3i = a3i + tvi*6.73695643646557207e-01; \
    } \
    { const T tcr = cc[8], tci = cc[9]; \
      yr[4] = tcr - a0i;  yi[4] = tci + a0r; \
      yr[13] = tcr + a0i;  yi[13] = tci - a0r; } \
    { const T tcr = cc[10], tci = cc[11]; \
      yr[5] = tcr + a1i;  yi[5] = tci - a1r; \
      yr[12] = tcr - a1i;  yi[12] = tci + a1r; } \
    { const T tcr = cc[12], tci = cc[13]; \
      yr[2] = tcr - a2i;  yi[2] = tci + a2r; \
      yr[15] = tcr + a2i;  yi[15] = tci - a2r; } \
    { const T tcr = cc[14], tci = cc[15]; \
      yr[6] = tcr - a3i;  yi[6] = tci + a3r; \
      yr[11] = tcr + a3i;  yi[11] = tci - a3r; } \
}
/* The 8 distinct negacyclic sine constants s~(t) = "S column" of the skew-
 * circulant, natural signs, exactly the literals kernels A/B use. */
#define DECL_SC(T) \
    T C0 = BCAST(T, 3.61241666187152921e-01); \
    T C1 = BCAST(T, 8.95163291355062340e-01); \
    T C2 = BCAST(T,-1.83749517816570340e-01); \
    T C3 = BCAST(T,-5.26432162877355836e-01); \
    T C4 = BCAST(T,-9.95734176295034468e-01); \
    T C5 = BCAST(T, 9.61825643172819045e-01); \
    T C6 = BCAST(T,-6.73695643646557207e-01); \
    T C7 = BCAST(T,-7.98017227280239494e-01)

/* Make the constants opaque so gcc cannot fold them back into .LC memory
 * operands or rematerialise the broadcast inside the loop.  Only meaningful
 * (and only encodable as "v") when the EVEX register file exists. */
#if defined(__AVX512F__) && defined(__AVX512VL__)
#define OPAQUE_SC() __asm__("" : "+v"(C0),"+v"(C1),"+v"(C2),"+v"(C3),"+v"(C4),"+v"(C5),"+v"(C6),"+v"(C7))
#else
#define OPAQUE_SC() ((void)0)
#endif

/* scalar kernel for the one spectator that does not fill a lane, and the
 * 4-wide kernel B used by pass 1's vector groups and the kx=16 tail combine */
DEF_K17_A(s, double)
DEF_K17_B(b4, v4)
#define EXL(T,S,P,a) (*(const T *)((P) + (size_t)(a)*(S) + off))
#define DEF_K17_E(SUF, T, STRIDE) \
static inline __attribute__((always_inline)) void k17_##SUF( \
        const double *sr, const double *si, size_t off, T *yr, T *yi, \
        const T C0, const T C1, const T C2, const T C3, \
        const T C4, const T C5, const T C6, const T C7) \
{ \
    T vv[16], cc[16]; \
    T a0r, a1r, a2r, a3r, a0i, a1i, a2i, a3i, b0r, b1r, b2r, b3r, b0i, b1i, b2i, b3i, dr, di, u0r,u0i,u1r,u1i; \
    const T x0r = EXL(T,STRIDE,sr,0), x0i = EXL(T,STRIDE,si,0); \
    dr = x0r; di = x0i; \
    { const T xa = EXL(T,STRIDE,sr,1), xb = EXL(T,STRIDE,sr,16); \
      const T za = EXL(T,STRIDE,si,1), zb = EXL(T,STRIDE,si,16); \
      u0r = xa + xb;  u0i = za + zb; \
      vv[0] = xa - xb;  vv[1] = za - zb; } \
    { const T xa = EXL(T,STRIDE,sr,4), xb = EXL(T,STRIDE,sr,13); \
      const T za = EXL(T,STRIDE,si,4), zb = EXL(T,STRIDE,si,13); \
      u1r = xa + xb;  u1i = za + zb; \
      vv[8] = xb - xa;  vv[9] = zb - za; } \
    { const T pr = u0r + u1r, pi = u0i + u1i; \
      const T qr = u0r - u1r, qi = u0i - u1i; \
      dr = dr + pr;  di = di + pi; \
      a0r = x0r + pr*5.12370294433828866e-01; \
      a1r = x0r + pr*8.60376828522276954e-02; \
      a2r = x0r + pr*-1.21982091231621334e-01; \
      a3r = x0r + pr*-7.26425886054435255e-01; \
      a0i = x0i + pi*5.12370294433828866e-01; \
      a1i = x0i + pi*8.60376828522276954e-02; \
      a2i = x0i + pi*-1.21982091231621334e-01; \
      a3i = x0i + pi*-7.26425886054435255e-01; \
      b0r = qr*4.20101934970526891e-01; \
      b1r = qr*3.59700672924310572e-01; \
      b2r = qr*-8.60991008452280493e-01; \
      b3r = qr*-1.23791249675178877e-01; \
      b0i = qi*4.20101934970526891e-01; \
      b1i = qi*3.59700672924310572e-01; \
      b2i = qi*-8.60991008452280493e-01; \
      b3i = qi*-1.23791249675178877e-01; \
    } \
    { const T xa = EXL(T,STRIDE,sr,3), xb = EXL(T,STRIDE,sr,14); \
      const T za = EXL(T,STRIDE,si,3), zb = EXL(T,STRIDE,si,14); \
      u0r = xa + xb;  u0i = za + zb; \
      vv[2] = xa - xb;  vv[3] = za - zb; } \
    { const T xa = EXL(T,STRIDE,sr,5), xb = EXL(T,STRIDE,sr,12); \
      const T za = EXL(T,STRIDE,si,5), zb = EXL(T,STRIDE,si,12); \
      u1r = xa + xb;  u1i = za + zb; \
      vv[10] = xa - xb;  vv[11] = za - zb; } \
    { const T pr = u0r + u1r, pi = u0i + u1i; \
      const T qr = u0r - u1r, qi = u0i - u1i; \
      dr = dr + pr;  di = di + pi; \
      a0r = a0r + pr*8.60376828522276954e-02; \
      a1r = a1r + pr*-1.21982091231621334e-01; \
      a2r = a2r + pr*-7.26425886054435255e-01; \
      a3r = a3r + pr*5.12370294433828866e-01; \
      a0i = a0i + pi*8.60376828522276954e-02; \
      a1i = a1i + pi*-1.21982091231621334e-01; \
      a2i = a2i + pi*-7.26425886054435255e-01; \
      a3i = a3i + pi*5.12370294433828866e-01; \
      b0r = b0r + qr*3.59700672924310572e-01; \
      b1r = b1r + qr*-8.60991008452280493e-01; \
      b2r = b2r + qr*-1.23791249675178877e-01; \
      b3r = b3r + qr*-4.20101934970526891e-01; \
      b0i = b0i + qi*3.59700672924310572e-01; \
      b1i = b1i + qi*-8.60991008452280493e-01; \
      b2i = b2i + qi*-1.23791249675178877e-01; \
      b3i = b3i + qi*-4.20101934970526891e-01; \
    } \
    { const T xa = EXL(T,STRIDE,sr,8), xb = EXL(T,STRIDE,sr,9); \
      const T za = EXL(T,STRIDE,si,8), zb = EXL(T,STRIDE,si,9); \
      u0r = xa + xb;  u0i = za + zb; \
      vv[4] = xb - xa;  vv[5] = zb - za; } \
    { const T xa = EXL(T,STRIDE,sr,2), xb = EXL(T,STRIDE,sr,15); \
      const T za = EXL(T,STRIDE,si,2), zb = EXL(T,STRIDE,si,15); \
      u1r = xa + xb;  u1i = za + zb; \
      vv[12] = xb - xa;  vv[13] = zb - za; } \
    { const T pr = u0r + u1r, pi = u0i + u1i; \
      const T qr = u0r - u1r, qi = u0i - u1i; \
      dr = dr + pr;  di = di + pi; \
      a0r = a0r + pr*-1.21982091231621334e-01; \
      a1r = a1r + pr*-7.26425886054435255e-01; \
      a2r = a2r + pr*5.12370294433828866e-01; \
      a3r = a3r + pr*8.60376828522276954e-02; \
      a0i = a0i + pi*-1.21982091231621334e-01; \
      a1i = a1i + pi*-7.26425886054435255e-01; \
      a2i = a2i + pi*5.12370294433828866e-01; \
      a3i = a3i + pi*8.60376828522276954e-02; \
      b0r = b0r + qr*-8.60991008452280493e-01; \
      b1r = b1r + qr*-1.23791249675178877e-01; \
      b2r = b2r + qr*-4.20101934970526891e-01; \
      b3r = b3r + qr*-3.59700672924310572e-01; \
      b0i = b0i + qi*-8.60991008452280493e-01; \
      b1i = b1i + qi*-1.23791249675178877e-01; \
      b2i = b2i + qi*-4.20101934970526891e-01; \
      b3i = b3i + qi*-3.59700672924310572e-01; \
    } \
    { const T xa = EXL(T,STRIDE,sr,7), xb = EXL(T,STRIDE,sr,10); \
      const T za = EXL(T,STRIDE,si,7), zb = EXL(T,STRIDE,si,10); \
      u0r = xa + xb;  u0i = za + zb; \
      vv[6] = xb - xa;  vv[7] = zb - za; } \
    { const T xa = EXL(T,STRIDE,sr,6), xb = EXL(T,STRIDE,sr,11); \
      const T za = EXL(T,STRIDE,si,6), zb = EXL(T,STRIDE,si,11); \
      u1r = xa + xb;  u1i = za + zb; \
      vv[14] = xb - xa;  vv[15] = zb - za; } \
    { const T pr = u0r + u1r, pi = u0i + u1i; \
      const T qr = u0r - u1r, qi = u0i - u1i; \
      dr = dr + pr;  di = di + pi; \
      a0r = a0r + pr*-7.26425886054435255e-01; \
      a1r = a1r + pr*5.12370294433828866e-01; \
      a2r = a2r + pr*8.60376828522276954e-02; \
      a3r = a3r + pr*-1.21982091231621334e-01; \
      a0i = a0i + pi*-7.26425886054435255e-01; \
      a1i = a1i + pi*5.12370294433828866e-01; \
      a2i = a2i + pi*8.60376828522276954e-02; \
      a3i = a3i + pi*-1.21982091231621334e-01; \
      b0r = b0r + qr*-1.23791249675178877e-01; \
      b1r = b1r + qr*-4.20101934970526891e-01; \
      b2r = b2r + qr*-3.59700672924310572e-01; \
      b3r = b3r + qr*8.60991008452280493e-01; \
      b0i = b0i + qi*-1.23791249675178877e-01; \
      b1i = b1i + qi*-4.20101934970526891e-01; \
      b2i = b2i + qi*-3.59700672924310572e-01; \
      b3i = b3i + qi*8.60991008452280493e-01; \
    } \
    yr[0] = dr;  yi[0] = di; \
    cc[0] = a0r + b0r;  cc[1] = a0i + b0i; \
    cc[8] = a0r - b0r;  cc[9] = a0i - b0i; \
    cc[2] = a1r + b1r;  cc[3] = a1i + b1i; \
    cc[10] = a1r - b1r;  cc[11] = a1i - b1i; \
    cc[4] = a2r + b2r;  cc[5] = a2i + b2i; \
    cc[12] = a2r - b2r;  cc[13] = a2i - b2i; \
    cc[6] = a3r + b3r;  cc[7] = a3i + b3i; \
    cc[14] = a3r - b3r;  cc[15] = a3i - b3i; \
    { \
    T s0r,s1r,s2r,s3r,s4r,s5r,s6r,s7r, s0i,s1i,s2i,s3i,s4i,s5i,s6i,s7i; \
    { const T vr = vv[0], vi = vv[1];   /* m=0 */ \
      s0r = vr*C0;  s0i = vi*C0; \
      s1r = vr*C1;  s1i = vi*C1; \
      s2r = vr*C2;  s2i = vi*C2; \
      s3r = vr*C3;  s3i = vi*C3; \
      s4r = vr*C4;  s4i = vi*C4; \
      s5r = vr*C5;  s5i = vi*C5; \
      s6r = vr*C6;  s6i = vi*C6; \
      s7r = vr*C7;  s7i = vi*C7; \
    } \
    { const T vr = vv[2], vi = vv[3];   /* m=1 */ \
      s0r = s0r + vr*C1;  s0i = s0i + vi*C1; \
      s1r = s1r + vr*C2;  s1i = s1i + vi*C2; \
      s2r = s2r + vr*C3;  s2i = s2i + vi*C3; \
      s3r = s3r + vr*C4;  s3i = s3i + vi*C4; \
      s4r = s4r + vr*C5;  s4i = s4i + vi*C5; \
      s5r = s5r + vr*C6;  s5i = s5i + vi*C6; \
      s6r = s6r + vr*C7;  s6i = s6i + vi*C7; \
      s7r = s7r - vr*C0;  s7i = s7i - vi*C0; \
    } \
    { const T vr = vv[4], vi = vv[5];   /* m=2 */ \
      s0r = s0r + vr*C2;  s0i = s0i + vi*C2; \
      s1r = s1r + vr*C3;  s1i = s1i + vi*C3; \
      s2r = s2r + vr*C4;  s2i = s2i + vi*C4; \
      s3r = s3r + vr*C5;  s3i = s3i + vi*C5; \
      s4r = s4r + vr*C6;  s4i = s4i + vi*C6; \
      s5r = s5r + vr*C7;  s5i = s5i + vi*C7; \
      s6r = s6r - vr*C0;  s6i = s6i - vi*C0; \
      s7r = s7r - vr*C1;  s7i = s7i - vi*C1; \
    } \
    { const T vr = vv[6], vi = vv[7];   /* m=3 */ \
      s0r = s0r + vr*C3;  s0i = s0i + vi*C3; \
      s1r = s1r + vr*C4;  s1i = s1i + vi*C4; \
      s2r = s2r + vr*C5;  s2i = s2i + vi*C5; \
      s3r = s3r + vr*C6;  s3i = s3i + vi*C6; \
      s4r = s4r + vr*C7;  s4i = s4i + vi*C7; \
      s5r = s5r - vr*C0;  s5i = s5i - vi*C0; \
      s6r = s6r - vr*C1;  s6i = s6i - vi*C1; \
      s7r = s7r - vr*C2;  s7i = s7i - vi*C2; \
    } \
    { const T vr = vv[8], vi = vv[9];   /* m=4 */ \
      s0r = s0r + vr*C4;  s0i = s0i + vi*C4; \
      s1r = s1r + vr*C5;  s1i = s1i + vi*C5; \
      s2r = s2r + vr*C6;  s2i = s2i + vi*C6; \
      s3r = s3r + vr*C7;  s3i = s3i + vi*C7; \
      s4r = s4r - vr*C0;  s4i = s4i - vi*C0; \
      s5r = s5r - vr*C1;  s5i = s5i - vi*C1; \
      s6r = s6r - vr*C2;  s6i = s6i - vi*C2; \
      s7r = s7r - vr*C3;  s7i = s7i - vi*C3; \
    } \
    { const T vr = vv[10], vi = vv[11]; /* m=5 */ \
      s0r = s0r + vr*C5;  s0i = s0i + vi*C5; \
      s1r = s1r + vr*C6;  s1i = s1i + vi*C6; \
      s2r = s2r + vr*C7;  s2i = s2i + vi*C7; \
      s3r = s3r - vr*C0;  s3i = s3i - vi*C0; \
      s4r = s4r - vr*C1;  s4i = s4i - vi*C1; \
      s5r = s5r - vr*C2;  s5i = s5i - vi*C2; \
      s6r = s6r - vr*C3;  s6i = s6i - vi*C3; \
      s7r = s7r - vr*C4;  s7i = s7i - vi*C4; \
    } \
    { const T vr = vv[12], vi = vv[13]; /* m=6 */ \
      s0r = s0r + vr*C6;  s0i = s0i + vi*C6; \
      s1r = s1r + vr*C7;  s1i = s1i + vi*C7; \
      s2r = s2r - vr*C0;  s2i = s2i - vi*C0; \
      s3r = s3r - vr*C1;  s3i = s3i - vi*C1; \
      s4r = s4r - vr*C2;  s4i = s4i - vi*C2; \
      s5r = s5r - vr*C3;  s5i = s5i - vi*C3; \
      s6r = s6r - vr*C4;  s6i = s6i - vi*C4; \
      s7r = s7r - vr*C5;  s7i = s7i - vi*C5; \
    } \
    { const T vr = vv[14], vi = vv[15]; /* m=7 */ \
      s0r = s0r + vr*C7;  s0i = s0i + vi*C7; \
      s1r = s1r - vr*C0;  s1i = s1i - vi*C0; \
      s2r = s2r - vr*C1;  s2i = s2i - vi*C1; \
      s3r = s3r - vr*C2;  s3i = s3i - vi*C2; \
      s4r = s4r - vr*C3;  s4i = s4i - vi*C3; \
      s5r = s5r - vr*C4;  s5i = s5i - vi*C4; \
      s6r = s6r - vr*C5;  s6i = s6i - vi*C5; \
      s7r = s7r - vr*C6;  s7i = s7i - vi*C6; \
    } \
    { const T tcr = cc[0], tci = cc[1]; \
      yr[1] = tcr + s0i;  yi[1] = tci - s0r; \
      yr[16] = tcr - s0i;  yi[16] = tci + s0r; } \
    { const T tcr = cc[2], tci = cc[3]; \
      yr[3] = tcr + s1i;  yi[3] = tci - s1r; \
      yr[14] = tcr - s1i;  yi[14] = tci + s1r; } \
    { const T tcr = cc[4], tci = cc[5]; \
      yr[8] = tcr - s2i;  yi[8] = tci + s2r; \
      yr[9] = tcr + s2i;  yi[9] = tci - s2r; } \
    { const T tcr = cc[6], tci = cc[7]; \
      yr[7] = tcr - s3i;  yi[7] = tci + s3r; \
      yr[10] = tcr + s3i;  yi[10] = tci - s3r; } \
    { const T tcr = cc[8], tci = cc[9]; \
      yr[4] = tcr - s4i;  yi[4] = tci + s4r; \
      yr[13] = tcr + s4i;  yi[13] = tci - s4r; } \
    { const T tcr = cc[10], tci = cc[11]; \
      yr[5] = tcr + s5i;  yi[5] = tci - s5r; \
      yr[12] = tcr - s5i;  yi[12] = tci + s5r; } \
    { const T tcr = cc[12], tci = cc[13]; \
      yr[2] = tcr - s6i;  yi[2] = tci + s6r; \
      yr[15] = tcr + s6i;  yi[15] = tci - s6r; } \
    { const T tcr = cc[14], tci = cc[15]; \
      yr[6] = tcr - s7i;  yi[6] = tci + s7r; \
      yr[11] = tcr + s7i;  yi[11] = tci - s7r; } \
    } \
}
#define ESL(T,S,P,a) (*(T *)((P) + (size_t)(a)*(S) + doff))

DEF_K17_E(e2_4, v4, SP)
/* Kernel H (panel_r8): the COMPONENT-SPLIT kernel.  The re and im streams of
 * this module are fully independent until the final +-i combine -- every
 * constant is real, and yr[k] needs only { the re a/b chains (from xr sums)
 * and the s*i sweep (from xi differences) }, never an im intermediate.  So
 * the kernel runs as two sequential phases: phase R computes and stores all
 * 17 re outputs, phase I all 17 im outputs.  Why: the r7 disassembly showed
 * the remaining non-FP block in ES is the vv[16]/cc[16] stack round trip
 * (~32 spill stores + ~40 reloads per group), which is STRUCTURAL in the
 * fused-component form -- 16 v values + 8 deferred cc values must live across
 * a sweep whose own liveness is 26.  Per phase the peak liveness is only
 * ~8 acc + 8 v + 8 cc + temps ~= 26-28, inside the 32-register EVEX file, so
 * the arrays disappear entirely.  Cost: the 33 input loads of each phase are
 * issued twice (66 vs 34 per group) -- and r7's kernel-E experiment measured
 * folded loads to be nearly free (a wash at +-1%), while its g8 result
 * measured store deletion at -2.9% wallaby / -8.9% node.  This trades ~72
 * store-heavy stack uops for ~32 cheap loads per group.
 * BIT-IDENTITY: each output value is produced by the identical operation
 *   chain in the identical order as kernels A/B/C/E/ES -- the phases reorder
 *   work only BETWEEN the two independent component chains, never within
 *   one, and stores go to disjoint addresses.  cmp-verified against g8/g4.
 * The OPA/OPB tokens are the +-i combine signs: phase R uses (+,-), phase I
 * (-,+) -- exactly the sign pattern of the ES combine, split by component. */
#define K17H_PHASE(T, STRIDE, PU, PV, DSTRIDE, DP, OPA, OPB, C0,C1,C2,C3,C4,C5,C6,C7) \
{ \
    T w0,w1,w2,w3,w4,w5,w6,w7, a0,a1,a2,a3, b0,b1,b2,b3, d, u0,u1; \
    const T x0 = EXL(T,STRIDE,PU,0); \
    d = x0; \
    { const T xa = EXL(T,STRIDE,PU,1), xb = EXL(T,STRIDE,PU,16); \
      const T za = EXL(T,STRIDE,PV,1), zb = EXL(T,STRIDE,PV,16); \
      u0 = xa + xb;  w0 = za - zb; } \
    { const T xa = EXL(T,STRIDE,PU,4), xb = EXL(T,STRIDE,PU,13); \
      const T za = EXL(T,STRIDE,PV,4), zb = EXL(T,STRIDE,PV,13); \
      u1 = xa + xb;  w4 = zb - za; } \
    { const T p = u0 + u1, q = u0 - u1; \
      d = d + p; \
      a0 = x0 + p*5.12370294433828866e-01; \
      a1 = x0 + p*8.60376828522276954e-02; \
      a2 = x0 + p*-1.21982091231621334e-01; \
      a3 = x0 + p*-7.26425886054435255e-01; \
      b0 = q*4.20101934970526891e-01; \
      b1 = q*3.59700672924310572e-01; \
      b2 = q*-8.60991008452280493e-01; \
      b3 = q*-1.23791249675178877e-01; \
    } \
    { const T xa = EXL(T,STRIDE,PU,3), xb = EXL(T,STRIDE,PU,14); \
      const T za = EXL(T,STRIDE,PV,3), zb = EXL(T,STRIDE,PV,14); \
      u0 = xa + xb;  w1 = za - zb; } \
    { const T xa = EXL(T,STRIDE,PU,5), xb = EXL(T,STRIDE,PU,12); \
      const T za = EXL(T,STRIDE,PV,5), zb = EXL(T,STRIDE,PV,12); \
      u1 = xa + xb;  w5 = za - zb; } \
    { const T p = u0 + u1, q = u0 - u1; \
      d = d + p; \
      a0 = a0 + p*8.60376828522276954e-02; \
      a1 = a1 + p*-1.21982091231621334e-01; \
      a2 = a2 + p*-7.26425886054435255e-01; \
      a3 = a3 + p*5.12370294433828866e-01; \
      b0 = b0 + q*3.59700672924310572e-01; \
      b1 = b1 + q*-8.60991008452280493e-01; \
      b2 = b2 + q*-1.23791249675178877e-01; \
      b3 = b3 + q*-4.20101934970526891e-01; \
    } \
    { const T xa = EXL(T,STRIDE,PU,8), xb = EXL(T,STRIDE,PU,9); \
      const T za = EXL(T,STRIDE,PV,8), zb = EXL(T,STRIDE,PV,9); \
      u0 = xa + xb;  w2 = zb - za; } \
    { const T xa = EXL(T,STRIDE,PU,2), xb = EXL(T,STRIDE,PU,15); \
      const T za = EXL(T,STRIDE,PV,2), zb = EXL(T,STRIDE,PV,15); \
      u1 = xa + xb;  w6 = zb - za; } \
    { const T p = u0 + u1, q = u0 - u1; \
      d = d + p; \
      a0 = a0 + p*-1.21982091231621334e-01; \
      a1 = a1 + p*-7.26425886054435255e-01; \
      a2 = a2 + p*5.12370294433828866e-01; \
      a3 = a3 + p*8.60376828522276954e-02; \
      b0 = b0 + q*-8.60991008452280493e-01; \
      b1 = b1 + q*-1.23791249675178877e-01; \
      b2 = b2 + q*-4.20101934970526891e-01; \
      b3 = b3 + q*-3.59700672924310572e-01; \
    } \
    { const T xa = EXL(T,STRIDE,PU,7), xb = EXL(T,STRIDE,PU,10); \
      const T za = EXL(T,STRIDE,PV,7), zb = EXL(T,STRIDE,PV,10); \
      u0 = xa + xb;  w3 = zb - za; } \
    { const T xa = EXL(T,STRIDE,PU,6), xb = EXL(T,STRIDE,PU,11); \
      const T za = EXL(T,STRIDE,PV,6), zb = EXL(T,STRIDE,PV,11); \
      u1 = xa + xb;  w7 = zb - za; } \
    { const T p = u0 + u1, q = u0 - u1; \
      d = d + p; \
      a0 = a0 + p*-7.26425886054435255e-01; \
      a1 = a1 + p*5.12370294433828866e-01; \
      a2 = a2 + p*8.60376828522276954e-02; \
      a3 = a3 + p*-1.21982091231621334e-01; \
      b0 = b0 + q*-1.23791249675178877e-01; \
      b1 = b1 + q*-4.20101934970526891e-01; \
      b2 = b2 + q*-3.59700672924310572e-01; \
      b3 = b3 + q*8.60991008452280493e-01; \
    } \
    ESL(T,DSTRIDE,DP,0) = d; \
    { const T cp0 = a0 + b0, cm0 = a0 - b0; \
      const T cp1 = a1 + b1, cm1 = a1 - b1; \
      const T cp2 = a2 + b2, cm2 = a2 - b2; \
      const T cp3 = a3 + b3, cm3 = a3 - b3; \
      T s0,s1,s2,s3,s4,s5,s6,s7; \
      s0 = w0*C0;  s1 = w0*C1;  s2 = w0*C2;  s3 = w0*C3; \
      s4 = w0*C4;  s5 = w0*C5;  s6 = w0*C6;  s7 = w0*C7; \
      s0 = s0 + w1*C1;  s1 = s1 + w1*C2;  s2 = s2 + w1*C3;  s3 = s3 + w1*C4; \
      s4 = s4 + w1*C5;  s5 = s5 + w1*C6;  s6 = s6 + w1*C7;  s7 = s7 - w1*C0; \
      s0 = s0 + w2*C2;  s1 = s1 + w2*C3;  s2 = s2 + w2*C4;  s3 = s3 + w2*C5; \
      s4 = s4 + w2*C6;  s5 = s5 + w2*C7;  s6 = s6 - w2*C0;  s7 = s7 - w2*C1; \
      s0 = s0 + w3*C3;  s1 = s1 + w3*C4;  s2 = s2 + w3*C5;  s3 = s3 + w3*C6; \
      s4 = s4 + w3*C7;  s5 = s5 - w3*C0;  s6 = s6 - w3*C1;  s7 = s7 - w3*C2; \
      s0 = s0 + w4*C4;  s1 = s1 + w4*C5;  s2 = s2 + w4*C6;  s3 = s3 + w4*C7; \
      s4 = s4 - w4*C0;  s5 = s5 - w4*C1;  s6 = s6 - w4*C2;  s7 = s7 - w4*C3; \
      s0 = s0 + w5*C5;  s1 = s1 + w5*C6;  s2 = s2 + w5*C7;  s3 = s3 - w5*C0; \
      s4 = s4 - w5*C1;  s5 = s5 - w5*C2;  s6 = s6 - w5*C3;  s7 = s7 - w5*C4; \
      s0 = s0 + w6*C6;  s1 = s1 + w6*C7;  s2 = s2 - w6*C0;  s3 = s3 - w6*C1; \
      s4 = s4 - w6*C2;  s5 = s5 - w6*C3;  s6 = s6 - w6*C4;  s7 = s7 - w6*C5; \
      s0 = s0 + w7*C7;  s1 = s1 - w7*C0;  s2 = s2 - w7*C1;  s3 = s3 - w7*C2; \
      s4 = s4 - w7*C3;  s5 = s5 - w7*C4;  s6 = s6 - w7*C5;  s7 = s7 - w7*C6; \
      ESL(T,DSTRIDE,DP,1)  = cp0 OPA s0;  ESL(T,DSTRIDE,DP,16) = cp0 OPB s0; \
      ESL(T,DSTRIDE,DP,3)  = cp1 OPA s1;  ESL(T,DSTRIDE,DP,14) = cp1 OPB s1; \
      ESL(T,DSTRIDE,DP,8)  = cp2 OPB s2;  ESL(T,DSTRIDE,DP,9)  = cp2 OPA s2; \
      ESL(T,DSTRIDE,DP,7)  = cp3 OPB s3;  ESL(T,DSTRIDE,DP,10) = cp3 OPA s3; \
      ESL(T,DSTRIDE,DP,4)  = cm0 OPB s4;  ESL(T,DSTRIDE,DP,13) = cm0 OPA s4; \
      ESL(T,DSTRIDE,DP,5)  = cm1 OPA s5;  ESL(T,DSTRIDE,DP,12) = cm1 OPB s5; \
      ESL(T,DSTRIDE,DP,2)  = cm2 OPB s6;  ESL(T,DSTRIDE,DP,15) = cm2 OPA s6; \
      ESL(T,DSTRIDE,DP,6)  = cm3 OPB s7;  ESL(T,DSTRIDE,DP,11) = cm3 OPA s7; \
    } \
}

#define DEF_K17_H(SUF, T, STRIDE, DSTRIDE) \
static inline __attribute__((always_inline)) void k17_##SUF( \
        const double *sr, const double *si, size_t off, \
        double *dpr, double *dpi, size_t doff, \
        const T C0, const T C1, const T C2, const T C3, \
        const T C4, const T C5, const T C6, const T C7) \
{ \
    K17H_PHASE(T, STRIDE, sr, si, DSTRIDE, dpr, +, -, C0,C1,C2,C3,C4,C5,C6,C7) \
    K17H_PHASE(T, STRIDE, si, sr, DSTRIDE, dpi, -, +, C0,C1,C2,C3,C4,C5,C6,C7) \
}

/* h2_*: fused pass-2 (A scratch in, ky-major mini out).  h3_*: fused pass-3
 * (mini in; "array out" is the strided store with row stride w and doff 0,
 * i.e. exactly a T[17] -- the tsto transpose then consumes it unchanged). */
DEF_K17_H(h2_8, v8, SP, 136)
DEF_K17_H(h2_4, v4, SP, 68)
DEF_K17_H(h3_8, v8, 8, 8)
DEF_K17_H(h3_4, v4, 4, 4)
static inline void tr8(v8 *a)
{
    const v8 t2_0 = SH8(a[0], a[4], 0,1,2,3,8,9,10,11);
    const v8 t2_4 = SH8(a[0], a[4], 4,5,6,7,12,13,14,15);
    const v8 t2_1 = SH8(a[1], a[5], 0,1,2,3,8,9,10,11);
    const v8 t2_5 = SH8(a[1], a[5], 4,5,6,7,12,13,14,15);
    const v8 t2_2 = SH8(a[2], a[6], 0,1,2,3,8,9,10,11);
    const v8 t2_6 = SH8(a[2], a[6], 4,5,6,7,12,13,14,15);
    const v8 t2_3 = SH8(a[3], a[7], 0,1,2,3,8,9,10,11);
    const v8 t2_7 = SH8(a[3], a[7], 4,5,6,7,12,13,14,15);
    const v8 t1_0 = SH8(t2_0, t2_2, 0,1,8,9,4,5,12,13);
    const v8 t1_2 = SH8(t2_0, t2_2, 2,3,10,11,6,7,14,15);
    const v8 t1_1 = SH8(t2_1, t2_3, 0,1,8,9,4,5,12,13);
    const v8 t1_3 = SH8(t2_1, t2_3, 2,3,10,11,6,7,14,15);
    const v8 t1_4 = SH8(t2_4, t2_6, 0,1,8,9,4,5,12,13);
    const v8 t1_6 = SH8(t2_4, t2_6, 2,3,10,11,6,7,14,15);
    const v8 t1_5 = SH8(t2_5, t2_7, 0,1,8,9,4,5,12,13);
    const v8 t1_7 = SH8(t2_5, t2_7, 2,3,10,11,6,7,14,15);
    a[0] = SH8(t1_0, t1_1, 0,8,2,10,4,12,6,14);
    a[1] = SH8(t1_0, t1_1, 1,9,3,11,5,13,7,15);
    a[2] = SH8(t1_2, t1_3, 0,8,2,10,4,12,6,14);
    a[3] = SH8(t1_2, t1_3, 1,9,3,11,5,13,7,15);
    a[4] = SH8(t1_4, t1_5, 0,8,2,10,4,12,6,14);
    a[5] = SH8(t1_4, t1_5, 1,9,3,11,5,13,7,15);
    a[6] = SH8(t1_6, t1_7, 0,8,2,10,4,12,6,14);
    a[7] = SH8(t1_6, t1_7, 1,9,3,11,5,13,7,15);
}

static inline void tr4(v4 *a)
{
    const v4 t1_0 = SH4(a[0], a[2], 0,1,4,5);
    const v4 t1_2 = SH4(a[0], a[2], 2,3,6,7);
    const v4 t1_1 = SH4(a[1], a[3], 0,1,4,5);
    const v4 t1_3 = SH4(a[1], a[3], 2,3,6,7);
    a[0] = SH4(t1_0, t1_1, 0,4,2,6);
    a[1] = SH4(t1_0, t1_1, 1,5,3,7);
    a[2] = SH4(t1_2, t1_3, 0,4,2,6);
    a[3] = SH4(t1_2, t1_3, 1,5,3,7);
}

static inline void tst8(const v8 *yr, const v8 *yi, double *dr, double *di,
                        const int32_t *b)
{
    v8 t[8];
    for (int k = 0; k < 2; ++k) {
        for (int i = 0; i < 8; ++i) t[i] = yr[k*8 + i];
        tr8(t);
        for (int j = 0; j < 8; ++j) *(v8u *)(dr + b[j] + k*8) = t[j];
        for (int i = 0; i < 8; ++i) t[i] = yi[k*8 + i];
        tr8(t);
        for (int j = 0; j < 8; ++j) *(v8u *)(di + b[j] + k*8) = t[j];
    }
    for (int j = 0; j < 8; ++j) { dr[b[j]+16] = yr[16][j]; di[b[j]+16] = yi[16][j]; }
}

static inline void tsto8(const v8 *yr, const v8 *yi, double *out, const int32_t *b)
{
    v8 tr_[8], ti_[8];
    for (int k = 0; k < 2; ++k) {
        for (int i = 0; i < 8; ++i) { tr_[i] = yr[k*8 + i]; ti_[i] = yi[k*8 + i]; }
        tr8(tr_); tr8(ti_);
        for (int j = 0; j < 8; ++j) {
            double *q = out + b[j] + k*16;
            *(v8u *)(q)     = SH8(tr_[j], ti_[j], 0,8,1,9,2,10,3,11);
            *(v8u *)(q + 8) = SH8(tr_[j], ti_[j], 4,12,5,13,6,14,7,15);
        }
    }
    for (int j = 0; j < 8; ++j) { out[b[j]+32] = yr[16][j]; out[b[j]+33] = yi[16][j]; }
}

static inline void tst4(const v4 *yr, const v4 *yi, double *dr, double *di,
                        const int32_t *b)
{
    v4 t[4];
    for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < 4; ++i) t[i] = yr[k*4 + i];
        tr4(t);
        for (int j = 0; j < 4; ++j) *(v4u *)(dr + b[j] + k*4) = t[j];
        for (int i = 0; i < 4; ++i) t[i] = yi[k*4 + i];
        tr4(t);
        for (int j = 0; j < 4; ++j) *(v4u *)(di + b[j] + k*4) = t[j];
    }
    for (int j = 0; j < 4; ++j) { dr[b[j]+16] = yr[16][j]; di[b[j]+16] = yi[16][j]; }
}

static inline void tsto4(const v4 *yr, const v4 *yi, double *out, const int32_t *b)
{
    v4 tr_[4], ti_[4];
    for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < 4; ++i) { tr_[i] = yr[k*4 + i]; ti_[i] = yi[k*4 + i]; }
        tr4(tr_); tr4(ti_);
        for (int j = 0; j < 4; ++j) {
            double *q = out + b[j] + k*8;
            *(v4u *)(q)     = SH4(tr_[j], ti_[j], 0,4,1,5);
            *(v4u *)(q + 4) = SH4(tr_[j], ti_[j], 2,6,3,7);
        }
    }
    for (int j = 0; j < 4; ++j) { out[b[j]+32] = yr[16][j]; out[b[j]+33] = yi[16][j]; }
}

#ifndef L17_DENSE_KERNEL
/* Blocked pass-1 stores for the fused variants: lane j's 17 kx values go to
 * the kx-BLOCKED layout -- kx < 16 in chunks of the vector width at
 * b[j] + (kx/w)*(17*w), kx = 16 to the separate tail region at t[j]. */
static inline void tst8f(const v8 *yr, const v8 *yi, double *dr, double *di,
                         const int32_t *b, const int32_t *t)
{
    v8 tmp[8];
    for (int k = 0; k < 2; ++k) {
        for (int i = 0; i < 8; ++i) tmp[i] = yr[k*8 + i];
        tr8(tmp);
        for (int j = 0; j < 8; ++j) *(v8u *)(dr + b[j] + k*136) = tmp[j];
        for (int i = 0; i < 8; ++i) tmp[i] = yi[k*8 + i];
        tr8(tmp);
        for (int j = 0; j < 8; ++j) *(v8u *)(di + b[j] + k*136) = tmp[j];
    }
    for (int j = 0; j < 8; ++j) { dr[t[j]] = yr[16][j]; di[t[j]] = yi[16][j]; }
}

static inline void tst4f(const v4 *yr, const v4 *yi, double *dr, double *di,
                         const int32_t *b, const int32_t *t)
{
    v4 tmp[4];
    for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < 4; ++i) tmp[i] = yr[k*4 + i];
        tr4(tmp);
        for (int j = 0; j < 4; ++j) *(v4u *)(dr + b[j] + k*68) = tmp[j];
        for (int i = 0; i < 4; ++i) tmp[i] = yi[k*4 + i];
        tr4(tmp);
        for (int j = 0; j < 4; ++j) *(v4u *)(di + b[j] + k*68) = tmp[j];
    }
    for (int j = 0; j < 4; ++j) { dr[t[j]] = yr[16][j]; di[t[j]] = yi[16][j]; }
}
#endif /* !L17_DENSE_KERNEL */
static inline __attribute__((always_inline)) void
p1g4(const double *in, double *dr, double *di,
     const int32_t *b, const int32_t *t, int g, int lookahead)
{
    if (lookahead && g + 2 < 72) {      /* see p1g8; w=4 rows are 1-2 lines */
        const double *q = in + (size_t)2*4*(g + 2);
        for (int a = 0; a < 17; ++a)
            __builtin_prefetch(q + (size_t)578*a, 0, 3);
    }
    const double *p = in + (size_t)2*4*g;
    v4 xr[17], xi[17], yr[17], yi[17];
    for (int a = 0; a < 17; ++a) {
        const v4u lo = *(const v4u *)(p + (size_t)578*a);
        const v4u hi = *(const v4u *)(p + (size_t)578*a + 4);
        xr[a] = SH4(lo, hi, 0,2,4,6);
        xi[a] = SH4(lo, hi, 1,3,5,7);
    }
    k17_b4(xr, xi, yr, yi);
    tst4f(yr, yi, dr, di, b + (size_t)4*g, t + (size_t)4*g);
}
static void pass1_f4(const double *in, double *dr, double *di,
                     const int32_t *b, const int32_t *t)
{
    for (int g = 0; g < 72; ++g)
        p1g4(in, dr, di, b, t, g, 0);
}
static void tail1_f4(const double *in, double *dr, double *di)
{
    double xr[17], xi[17], yr[17], yi[17];
    for (int a = 0; a < 17; ++a) { xr[a] = in[2*(289*a + 288)]; xi[a] = in[2*(289*a + 288)+1]; }
    k17_s(xr, xi, yr, yi);
    for (int a = 0; a < 16; ++a) {
        const int o = 16*SP + (a/4)*68 + 16*4 + a%4;
        dr[o] = yr[a];  di[o] = yi[a];
    }
    dr[16*SP + 288] = yr[16];  di[16*SP + 288] = yi[16];
}
/* ---------- split-free pass 1 (variant i4 = h4 with this pass 1)
 *
 * ADOPTED FROM L17_rader panel_r8/r9 ("dy", the one mechanism the node picked
 * at batch this round: -2.8% at B=2048).  Pass 1's row loads sit on the 16-B
 * alignment classes of the 4624-B row stride (class = 16*(a mod 4) bytes), so
 * at w=4 the ymm pair (lo @ c, hi @ c+32) crosses a cache line exactly when
 * the piece lands on class 48: rows a%4==1 split on hi, rows a%4==3 split on
 * lo -- 8 of 34 loads per group, 576 per volume, and at batch those lines are
 * COLD, where a split holds line-fill resources for two fills on the node's
 * 2-load-port Cascade Lake.  ld4ns re-issues exactly those 8 loads as two
 * xmm halves (class-48 xmm pieces end at the line boundary, so neither half
 * splits) + one vinsertf128: +8 loads +8 shuffles per group buys 0 splits.
 * The loaded VALUES are identical, kernel and stores are h4's, so i4 is
 * bit-identical to h4 by construction.  Wallaby (3 load ports, DDR5) cannot
 * price cold splits -- rader measured its dy LOSING ~1.5% there while the
 * node picked it -- so i4 ships strictly as a tuner candidate. */
static inline __attribute__((always_inline)) v4 ld4ns(const double *q)
{
    const v2u a_ = *(const v2u *)q;
    const v2u b_ = *(const v2u *)(q + 2);
    return (v4){a_[0], a_[1], b_[0], b_[1]};
}

static void pass1_i4(const double *in, double *dr, double *di,
                     const int32_t *b, const int32_t *t)
{
    for (int g = 0; g < 72; ++g) {
        const double *p = in + (size_t)2*4*g;
        v4 xr[17], xi[17], yr[17], yi[17];
#pragma GCC unroll 17
        for (int a = 0; a < 17; ++a) {
            const double *ra = p + (size_t)578*a;
            const v4 lo = ((a & 3) == 3) ? ld4ns(ra)     : (v4)*(const v4u *)ra;
            const v4 hi = ((a & 3) == 1) ? ld4ns(ra + 4) : (v4)*(const v4u *)(ra + 4);
            xr[a] = SH4(lo, hi, 0,2,4,6);
            xi[a] = SH4(lo, hi, 1,3,5,7);
        }
        k17_b4(xr, xi, yr, yi);
        tst4f(yr, yi, dr, di, b + (size_t)4*g, t + (size_t)4*g);
    }
}
#define TAIL_DST (16*SP + 16*17)
#define TAIL_OUT (2*(16*289 + 16*17))

/* fused pass 2+3 (h4 = component-split kernel H, 4-wide), verbatim from
 * L17_winograd's fused23_h4 except: the plan indirection is replaced by
 * explicit scratch/table arguments, and the pf/pfw/CLWB hooks are removed
 * (hints/flushes only -- the node's winning processes ran pf=0 pfw=0 cw=0). */
static void wg_fused23_h4(const double *ar, const double *ai, double *out,
                          double *br, double *bi,
                          const int32_t *bt, const int32_t *boN4,
                          const int32_t *botail)
{
    DECL_SC(v4); OPAQUE_SC();
    double *btr = br + 2432, *bti = bi + 2432;
    for (int blk = 0; blk < 4; ++blk) {
        const double *sr = ar + 68*blk, *si = ai + 68*blk;
        for (int z = 0; z < 17; ++z)
            k17_h2_4(sr, si, (size_t)4*z, br, bi, (size_t)4*z,
                     C0,C1,C2,C3,C4,C5,C6,C7);
        double *o = out + (size_t)2312*blk;      /* 2*4*289 doubles per block */
        for (int h = 0; h < 17; ++h) {
            v4 yr[17], yi[17];
            k17_h3_4(br, bi, (size_t)68*h, (double *)yr, (double *)yi, 0,
                     C0,C1,C2,C3,C4,C5,C6,C7);
            tsto4(yr, yi, o, boN4 + (size_t)4*h);
        }
    }
    /* tail block, kx = 16 */
    for (int zg = 0; zg < 4; ++zg) {
        v4 yr[17], yi[17];
        k17_e2_4(ar, ai, (size_t)(272 + 4*zg), yr, yi, C0,C1,C2,C3,C4,C5,C6,C7);
        tst4(yr, yi, btr, bti, bt + (size_t)4*zg);
    }
    {
        double xr[17], xi[17], yr[17], yi[17];
        for (int a = 0; a < 17; ++a) { xr[a] = ar[(size_t)a*SP + 288]; xi[a] = ai[(size_t)a*SP + 288]; }
        k17_s(xr, xi, yr, yi);
        for (int a = 0; a < 17; ++a) { btr[16*24 + a] = yr[a]; bti[16*24 + a] = yi[a]; }
    }
    for (int kg = 0; kg < 4; ++kg) {
        v4 xr[17], xi[17], yr[17], yi[17];
        for (int a = 0; a < 17; ++a) {
            xr[a] = *(const v4 *)(btr + (size_t)a*24 + 4*kg);
            xi[a] = *(const v4 *)(bti + (size_t)a*24 + 4*kg);
        }
        k17_b4(xr, xi, yr, yi);
        tsto4(yr, yi, out, botail + (size_t)4*kg);
    }
    {
        double xr[17], xi[17], yr[17], yi[17];
        for (int a = 0; a < 17; ++a) { xr[a] = btr[(size_t)a*24 + 16]; xi[a] = bti[(size_t)a*24 + 16]; }
        k17_s(xr, xi, yr, yi);
        for (int a = 0; a < 17; ++a) { out[TAIL_OUT + 2*a] = yr[a]; out[TAIL_OUT + 2*a+1] = yi[a]; }
    }
}

/* one shared int32 table arena for the wg engine (layout offsets, doubles):
 * b4f 288 (pass-1 blocked store bases) | tf 288 (pass-1 kx=16 tail bases) |
 * bt 17 (tail-block mini-buffer rows) | boN4 68 (fused pass-3 -> out, lane =
 * kx-in-block) | botail 17 (kx=16 -> out) */
enum { L17WG_TB4F = 0, L17WG_TTF = 288, L17WG_TBT = 576,
       L17WG_TBON4 = 593, L17WG_TBOTAIL = 661, L17WG_TN = 678 };

/* mode-1 exec body for the streaming regime: each pool thread runs this
 * serially over its contiguous static block of volumes, out of its OWN
 * 4 x 17*SP scratch (first-touched by that thread in l17wg_setup).  DRAM
 * traffic per volume: one contiguous-ish read of `in` (72 vector groups x 17
 * rows), one write of `out` (fused pass 3, one plane region at a time);
 * everything between lives in the 157 KB scratch (L2-resident on both
 * machines).  wgi4 = 1 re-issues pass 1's 8 line-splitting loads per group
 * as xmm pairs (i4) -- identical VALUES, so h4/i4 are bit-identical. */
static void l17_execm_wgh4(const fft3d_plan *restrict p,
                           const double _Complex *restrict in,
                           double _Complex *restrict out)
{
    const int32_t *tb = p->wgt;
    double *ar = p->wgbuf, *ai = p->wgbuf + (size_t)17*SP;
    double *br = p->wgbuf + (size_t)2*17*SP, *bi = p->wgbuf + (size_t)3*17*SP;
    const int nb = p->batch;
    for (int b = 0; b < nb; ++b) {
        const double *vin = (const double *)(in + (size_t)b * 4913);
        double *vout = (double *)(out + (size_t)b * 4913);
        if (p->wgi4)
            pass1_i4(vin, ar, ai, tb + L17WG_TB4F, tb + L17WG_TTF);
        else
            pass1_f4(vin, ar, ai, tb + L17WG_TB4F, tb + L17WG_TTF);
        tail1_f4(vin, ar, ai);
        wg_fused23_h4(ar, ai, vout, br, bi,
                      tb + L17WG_TBT, tb + L17WG_TBON4, tb + L17WG_TBOTAIL);
    }
}

/* Aggregate last-level cache reachable by the team: per-package L3 (sysconf)
 * times the number of DISTINCT physical packages the team's threads are bound
 * to (read from sysfs; a pure read, no policy change).  Returns 0 if the
 * machine will not say -- the caller then falls back to a fixed threshold. */
static size_t l17wg_aggl3(const cpu_set_t *masks, const int *have, int maxt)
{
    long l3 = -1;
#ifdef _SC_LEVEL3_CACHE_SIZE
    l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#endif
    if (l3 <= 0) return 0;
    int ids[8];
    int np = 0;
    for (int t = 0; t < maxt; ++t) {
        if (!masks || !have || !have[t]) continue;
        int cpu = -1;
        for (int c = 0; c < CPU_SETSIZE; ++c)
            if (CPU_ISSET(c, &masks[t])) { cpu = c; break; }
        if (cpu < 0) continue;
        char path[96];
        int id = -1;
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%d/topology/physical_package_id",
                 cpu);
        FILE *fp = fopen(path, "r");
        if (fp) {
            if (fscanf(fp, "%d", &id) != 1) id = -1;
            fclose(fp);
        }
        if (id < 0) continue;
        int seen = 0;
        for (int u = 0; u < np; ++u)
            if (ids[u] == id) seen = 1;
        if (!seen && np < 8) ids[np++] = id;
    }
    if (np < 1) np = 1;
    return (size_t)l3 * (size_t)np;
}

#ifdef _OPENMP
/* wg engine setup: one shared read-only table arena (master-owned), and one
 * private 4 x 17*SP scratch per child, allocated AND first-touched by its own
 * thread (page-aligned, so no two threads share a scratch line or page). */
static int l17wg_setup(fft3d_plan *p, int maxt)
{
    void *traw = NULL;
    if (posix_memalign(&traw, 64, L17WG_TN * sizeof(int32_t)) != 0 || !traw)
        return 0;
    int32_t *tb = traw;
    for (int s = 0; s < 288; ++s) {
        int q = s / 17, r = s % 17;
        tb[L17WG_TB4F + s] = (int32_t)(q * SP + r * 4);
        tb[L17WG_TTF + s] = (int32_t)(q * SP + 272 + r);
    }
    for (int z = 0; z < 17; ++z) {
        tb[L17WG_TBT + z] = (int32_t)(z * 24);
        tb[L17WG_TBOTAIL + z] = (int32_t)(2 * (16 * 289 + z * 17));
    }
    for (int ky = 0; ky < 17; ++ky)
        for (int j = 0; j < 4; ++j)
            tb[L17WG_TBON4 + ky * 4 + j] = (int32_t)(2 * (j * 289 + ky * 17));
    p->wgt_raw = traw;
    p->wgt = tb;
    int fail = 0;
#pragma omp parallel num_threads(maxt) reduction(||: fail)
    {
        int t = omp_get_thread_num();
        fft3d_plan *k = (t < p->nkids) ? p->kids[t] : NULL;
        void *raw = NULL;
        if (k && posix_memalign(&raw, 4096,
                                (size_t)4 * 17 * SP * sizeof(double)) == 0 &&
            raw) {
            memset(raw, 0, (size_t)4 * 17 * SP * sizeof(double));
            k->wgbuf_raw = raw;
            k->wgbuf = (double *)raw;
            k->wgt = tb;
        } else {
            fail = 1;
        }
    }
    return !fail;
}
#endif /* _OPENMP */

#ifdef _OPENMP
/* ---- persistent spin pool ----------------------------------------------
 * gcc's fork/join for an execute-time OpenMP region measured 3.4 us (4
 * threads) to 13.9 us (32) per execute on wallaby -- more than the whole
 * parallel B=1 transform.  So the execute-time team is pthreads created
 * ONCE in create() (the brief: "thread pools belong in fft3d_create()"),
 * pinned to the SAME cores OpenMP was given (each OMP thread's affinity
 * mask is captured inside the child-building parallel region and copied to
 * the pool worker of the same index, so `close/cores` is reproduced
 * exactly).  A job is released by one atomic generation store and
 * collected by per-thread padded done flags; workers busy-spin (pause)
 * between back-to-back executes -- the driver's timing loop -- and decay
 * to a futex sleep after ~4 ms idle so the plan-time single-thread probes
 * and anything else on the machine are not perturbed. */
enum { L17MT_MAXT = 64 };

typedef struct { volatile int v; char pad[60]; } l17mt_flag;

struct l17mt_pool;
typedef struct l17mt_warg { struct l17mt_pool *pl; int t; } l17mt_warg;

typedef struct l17mt_pool {
    fft3d_plan *plan;                   /* job spec: written before release */
    const double _Complex *in;
    double _Complex *out;
    int mode, nthr, dynb, nxr;
    int gen;                            /* master generation (main thread only) */
    volatile int quit;
    volatile int next;                  /* mode-1 dynamic block counter */
    int nwork;
    /* mode-2 barrier state (mt_r2: flat arrival/release, one line per
     * arriver plus one release line -- see l17mt_barrier) */
    l17mt_flag barr[L17MT_MAXT];
    l17mt_flag brel;
    /* per-worker release/ack flags, one cache line each: only the ACTIVE
     * team's flags are touched per execute, idle workers never wake, never
     * read the job fields, and cause no coherence traffic at all */
    l17mt_flag gof[L17MT_MAXT];
    l17mt_flag done[L17MT_MAXT];
    pthread_t th[L17MT_MAXT];
    l17mt_warg warg[L17MT_MAXT];
    cpu_set_t mask[L17MT_MAXT];
    int have_mask[L17MT_MAXT];
} l17mt_pool;

static inline void l17mt_pause(void)
{
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
}

/* Flat arrival-flag/release barrier (round mt_r2), ADOPTED FROM
 * L17_winograd's mt_r1 record: the r1 central atomic counter is nt
 * serialized RFOs on one line (winograd measured ~1.2 us at T=16 for that
 * design and 0.3-0.4 us for this one; this entry's own skip-probe read
 * 1.56 us at nt=16).  Each arriver writes its OWN padded line -- the
 * caller's scan misses then overlap -- and the caller publishes one release
 * word.  The sequence number is the pool generation g of the current job:
 * unique per execute and strictly increasing, so a thread that sat out
 * earlier (smaller-team) dispatches can never be out of phase, exactly
 * winograd's derive-the-seq-from-the-dispatch-epoch argument.  Mode 2 has
 * one barrier per execute, so g needs no per-barrier subdivision. */
static void l17mt_barrier(l17mt_pool *pl, int nt, int t, int g)
{
    if (t == 0) {
        for (int u = 1; u < nt; ++u)
            while (__atomic_load_n(&pl->barr[u].v, __ATOMIC_ACQUIRE) < g)
                l17mt_pause();
        __atomic_store_n(&pl->brel.v, g, __ATOMIC_RELEASE);
    } else {
        __atomic_store_n(&pl->barr[t].v, g, __ATOMIC_RELEASE);
        while (__atomic_load_n(&pl->brel.v, __ATOMIC_ACQUIRE) < g)
            l17mt_pause();
    }
}

/* thread t's share of the current job (t == 0 is the caller's thread);
 * g is the job's pool generation, used as the barrier sequence */
static void l17mt_work(l17mt_pool *pl, int t, int g)
{
    fft3d_plan *p = pl->plan;
    fft3d_plan *k = p->kids[t];
    const int nt = pl->nthr, nb = p->batch;
    const double _Complex *in = pl->in;
    double _Complex *out = pl->out;
    if (pl->mode == 2) {
        const int NPL = 17 * nb, NCH = 73 * nb;
        const int nx = pl->nxr < nt ? pl->nxr : nt;
        if (!(p->mtskip & 1))
            l17mt_planes(k, (const double *)in, p->t1g,
                         (int)(((long)NPL * t) / nt),
                         (int)(((long)NPL * (t + 1)) / nt));
        if (!(p->mtskip & 4))
            l17mt_barrier(pl, nt, t, g);
        if (!(p->mtskip & 2) && t < nx)
            l17mt_xrange(k, p->t1g, (double *)out,
                         (int)(((long)NCH * t) / nx),
                         (int)(((long)NCH * (t + 1)) / nx), p->xpf);
    } else if (pl->dynb > 0) {
        const int db = pl->dynb, nblk = (nb + db - 1) / db;
        int blk;
        while ((blk = __atomic_fetch_add(&pl->next, 1, __ATOMIC_RELAXED)) < nblk) {
            int v0 = blk * db;
            int nv = nb - v0 < db ? nb - v0 : db;
            k->batch = nv;
            k->exec(k, in + (size_t)v0 * 4913, out + (size_t)v0 * 4913);
        }
    } else {
        int v0 = (int)(((long)nb * t) / nt);
        int v1 = (int)(((long)nb * (t + 1)) / nt);
        if (v1 > v0) {
            k->batch = v1 - v0;
            k->exec(k, in + (size_t)v0 * 4913, out + (size_t)v0 * 4913);
        }
    }
}

static double l17_now(void); /* fwd (defined with the tuner utilities) */

static void *l17mt_worker(void *argp)
{
    l17mt_pool *pl = ((l17mt_warg *)argp)->pl;
    const int t = ((l17mt_warg *)argp)->t;
    if (pl->have_mask[t])
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &pl->mask[t]);
    volatile int *mygo = &pl->gof[t].v;
    int myg = 0;
    for (;;) {
        int g;
        long spins = 0;
        double idle0 = 0.0;
        while ((g = __atomic_load_n(mygo, __ATOMIC_ACQUIRE)) == myg) {
            l17mt_pause();
            if ((++spins & 8191) == 0) {
                /* decay from a hot spin to a 0.1-1 ms poll after ~4 ms idle:
                 * an idle worker then costs ~0.1% of a core and nothing on
                 * the release path (no futex handshake); the up-to-1 ms wake
                 * latency hits only the first execute after a long gap,
                 * which the driver's discarded warmups absorb */
                double tn = l17_now();
                if (idle0 == 0.0) {
                    idle0 = tn;
                } else if (tn - idle0 > 4e-3) {
                    struct timespec ts;
                    ts.tv_sec = 0;
                    ts.tv_nsec = (tn - idle0 > 0.1) ? 1000000 : 100000;
                    nanosleep(&ts, NULL);
                }
            }
        }
        myg = g;
        if (__atomic_load_n(&pl->quit, __ATOMIC_ACQUIRE)) break;
        l17mt_work(pl, t, g); /* released <=> t < nthr, so no membership test */
        __atomic_store_n(&pl->done[t].v, g, __ATOMIC_RELEASE);
    }
    return NULL;
}

static void l17mt_pool_run(fft3d_plan *p, const double _Complex *in,
                           double _Complex *out)
{
    l17mt_pool *pl = p->poolv;
    pl->plan = p;
    pl->in = in;
    pl->out = out;
    pl->mode = p->mode;
    pl->nthr = p->nthr > pl->nwork ? pl->nwork : p->nthr;
    pl->dynb = p->dynb;
    pl->nxr = p->nxr > 0 ? p->nxr : pl->nthr;
    pl->next = 0;
    const int g = ++pl->gen, nt = pl->nthr;
    for (int t = 1; t < nt; ++t)
        __atomic_store_n(&pl->gof[t].v, g, __ATOMIC_RELEASE);
    l17mt_work(pl, 0, g);
    for (int t = 1; t < nt; ++t)
        while (__atomic_load_n(&pl->done[t].v, __ATOMIC_ACQUIRE) != g)
            l17mt_pause();
}

static l17mt_pool *l17mt_pool_new(int nwork, const cpu_set_t *masks,
                                  const int *have)
{
    if (nwork > L17MT_MAXT) nwork = L17MT_MAXT;
    l17mt_pool *pl = calloc(1, sizeof *pl);
    if (!pl) return NULL;
    pl->nwork = nwork;
    for (int t = 0; t < nwork; ++t) {
        pl->mask[t] = masks[t];
        pl->have_mask[t] = have[t];
        pl->warg[t].pl = pl;
        pl->warg[t].t = t;
    }
    for (int t = 1; t < nwork; ++t) {
        if (pthread_create(&pl->th[t], NULL, l17mt_worker, &pl->warg[t]) != 0) {
            __atomic_store_n(&pl->quit, 1, __ATOMIC_SEQ_CST);
            for (int u = 1; u < t; ++u)
                __atomic_store_n(&pl->gof[u].v, pl->gen + 1, __ATOMIC_SEQ_CST);
            for (int u = 1; u < t; ++u) pthread_join(pl->th[u], NULL);
            free(pl);
            return NULL;
        }
    }
    return pl;
}

static void l17mt_pool_free(l17mt_pool *pl)
{
    if (!pl) return;
    __atomic_store_n(&pl->quit, 1, __ATOMIC_SEQ_CST);
    for (int t = 1; t < pl->nwork; ++t)
        __atomic_store_n(&pl->gof[t].v, pl->gen + 1, __ATOMIC_SEQ_CST);
    for (int t = 1; t < pl->nwork; ++t) pthread_join(pl->th[t], NULL);
    free(pl);
}
#endif /* _OPENMP */

static void l17mt_dispatch(fft3d_plan *p, const double _Complex *in,
                           double _Complex *out)
{
#ifdef _OPENMP
    if (p->mode != 0 && p->poolv) { l17mt_pool_run(p, in, out); return; }
#endif
    p->exec(p, in, out);
}

#ifdef _OPENMP
static void l17mt_set_kids(fft3d_plan *p,
                           void (*fn)(const fft3d_plan *,
                                      const double _Complex *,
                                      double _Complex *),
                           int pf, int pw, int pt)
{
    for (int t = 0; t < p->nkids; ++t)
        if (p->kids[t]) {
            p->kids[t]->exec = fn;
            p->kids[t]->pf = pf;
            p->kids[t]->pw = pw;
            p->kids[t]->pt = pt;
        }
}

/* seconds per transform for p AS CURRENTLY CONFIGURED (mode/nthr/dynb/kids),
 * on the tuner arena with nv volumes: 2 warmups (the first also calibrates an
 * inner count that clears timer resolution), MEDIAN of 5 blocked samples.
 * mt_r2: the statistic was min-of-3; the r1 VERDICT (section 3.2) showed nine
 * entries -- this one included, 1.12x at B=256 -- installing different picks
 * in different processes because two candidates within noise flip on their
 * luckiest sample.  A median flips only when the underlying distributions
 * actually cross. */
static double l17mt_time_cfg(fft3d_plan *p, int nv)
{
    int sb = p->batch;
    p->batch = nv;
    l17mt_dispatch(p, p->ti, p->to);
    double t0 = l17_now();
    l17mt_dispatch(p, p->ti, p->to);
    double dt = l17_now() - t0;
    long inner = dt > 1e-9 ? (long)(2e-3 / dt) + 1 : 1000;
    if (inner > 4000) inner = 4000;
    double s[5];
    for (int r = 0; r < 5; ++r) {
        t0 = l17_now();
        for (long i = 0; i < inner; ++i) l17mt_dispatch(p, p->ti, p->to);
        s[r] = l17_now() - t0;
    }
    for (int a = 1; a < 5; ++a) { /* insertion sort, take s[2] */
        double v = s[a];
        int b = a;
        while (b > 0 && s[b - 1] > v) { s[b] = s[b - 1]; --b; }
        s[b] = v;
    }
    p->batch = sb;
    return s[2] / (double)inner / (double)nv;
}

/* mt_r2: the tuner arena for the batch>=64 races, with its pages placed the
 * way the SCORED run's pages end up, not the way they start.  r1 filled the
 * arena serially "to match the driver's first touch"; the r1 VERDICT
 * established that the driver's serially-touched pages do not STAY on
 * socket 0 through the multi-second scored loop (AutoNUMA migrates them
 * toward the threads that keep faulting them -- L=6 B=65536 sustains
 * 175 GB/s, well above one socket), so the serial-touch arena prices a
 * transient, and this entry's B=4096 pick (NT+pipelined, 2.106 us/t on the
 * node) lost 1.72x to a plain static schedule.  Touching volume v's pages
 * with the thread that will process v under the static split is the
 * migrated steady state; it is also what the harness would produce if it
 * adopts the VERDICT's parallel-first-touch fix.  Values are then filled
 * serially into the already-homed pages (first touch is per PAGE, at first
 * write after mmap; later writes move nothing). */
static int l17_tune_alloc_mt(fft3d_plan *p, int nv, int nt)
{
    size_t n = (size_t)nv * 4913;
    if (p->tn >= n) return 1; /* only ever called first; already owner-touched */
    l17_tune_free(p);
    if (posix_memalign((void **)&p->ti, 64, n * sizeof *p->ti) != 0) { p->ti = NULL; return 0; }
    if (posix_memalign((void **)&p->to, 64, n * sizeof *p->to) != 0) {
        free(p->ti); p->ti = NULL; p->to = NULL; return 0;
    }
    p->tn = n;
#pragma omp parallel num_threads(nt)
    {
        int t = omp_get_thread_num();
        size_t d0 = ((size_t)nv * t / nt) * 4913;
        size_t d1 = ((size_t)nv * (t + 1) / nt) * 4913;
        if (d1 > d0) {
            memset(p->ti + d0, 0, (d1 - d0) * sizeof *p->ti);
            memset(p->to + d0, 0, (d1 - d0) * sizeof *p->to);
        }
    }
    unsigned sr = 12345u;
    for (size_t i = 0; i < n; ++i) {
        sr = sr * 1103515245u + 12345u;
        double a = (double)(sr >> 8) / 8388608.0 - 1.0;
        sr = sr * 1103515245u + 12345u;
        double b = (double)(sr >> 8) / 8388608.0 - 1.0;
        p->ti[i] = a + b * (double _Complex)I;
    }
    return 1;
}

/* mt_r2: (re)map and first-touch the mode-2 t1g by the map of an nt-thread
 * team.  mmap rather than malloc because a RE-touch must start from fresh
 * untouched pages (free()+malloc may recycle pages that already have a home
 * node); the memset below is then the first touch, page by page, by the
 * thread that will write that plane in phase 1.  Called once with the full
 * team before the mode race (t1g must exist to race mode 2 at all), and
 * again with the PICKED team: under close binding on a two-socket node a
 * 16-thread pick would otherwise read planes 8..16 out of pages homed on
 * socket 1 forever -- the full-team touch map and the picked-team use map
 * disagree exactly when nthr < maxt. */
static void l17mt_t1g_map(fft3d_plan *p, int batch, int nt)
{
    const size_t nd = (size_t)batch * L17MT_T1VOL * sizeof(double);
    if (p->t1g_raw) {
        munmap(p->t1g_raw, p->t1g_sz);
        p->t1g_raw = NULL;
    }
    void *m = mmap(NULL, nd, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    p->t1g_raw = m == MAP_FAILED ? NULL : m;
    p->t1g_sz = p->t1g_raw ? nd : 0;
    p->t1g = (double *)p->t1g_raw;
    if (!p->t1g) return;
    const int NPL = 17 * batch;
#pragma omp parallel num_threads(nt)
    {
        int t = omp_get_thread_num();
        int g0 = (int)(((long)NPL * t) / nt);
        int g1 = (int)(((long)NPL * (t + 1)) / nt);
        for (int g = g0; g < g1; ++g)
            memset(p->t1g + (size_t)(g / 17) * L17MT_T1VOL +
                       (size_t)(g % 17) * L17_T1SP,
                   0, L17_T1SP * sizeof(double));
    }
}
#endif /* _OPENMP */

/* -----------------------------------------------------------------------
 * ROUND panel_r10: in-plan STREAMING bandwidth decomposition (sbw), the
 * batch-regime sibling of r9's b1dec, same instrument (timed in create(),
 * routed out through fft3d_description()).  The question it answers on the
 * scoring machine: the streaming cells sit at ~21.7 us/volume moving
 * ~236 KB/volume of compulsory DRAM traffic (in read + out RFO + out
 * writeback) = 10.9 GB/s -- is that the machine's own single-core copy
 * speed (cells CLOSED: nothing but traffic deletion could move them, and
 * RFO deletion without NT stores does not exist on CLX), or is the code
 * losing bandwidth to its access pattern?  Four pure memory patterns, no
 * compute, us per 78.6 KiB volume-equivalent on the >L3 tuner arena:
 *   rd  = sequential zmm read of a volume        (best-case read)
 *   wr  = sequential zmm write (RFO + writeback) (best-case plain write)
 *   cp  = per-volume read burst then write burst (the X-first exec's own
 *         phase alternation with the compute removed: the streaming floor)
 *   s17 = the X pass's ACTUAL read pattern: 17 interleaved row streams,
 *         one 64 B line per row per step, rows 4624 B apart
 * Ledger, pre-registered: measured cell ~= cp  => bandwidth-closed, say so.
 * cp well below the cell with s17 >> rd        => the 17-stream read shape
 * is the recoverable share and the staged-input twins are the matching fix.
 * cp well below the cell with s17 ~= rd        => headroom is elsewhere
 * (write side / phase overlap), staging will not be picked.
 * --------------------------------------------------------------------- */
static volatile double l17_sbw_sink;

static __attribute__((noinline)) void l17_sbw_rd(const double *restrict s, long nvol)
{
    vd8u a0 = {0}, a1 = {0}, a2 = {0}, a3 = {0};
    for (long v = 0; v < nvol; ++v) {
        const double *q = s + v * 9826;
        for (long i = 0; i + 32 <= 9826; i += 32) {
            a0 += *(const vd8u *)(q + i);
            a1 += *(const vd8u *)(q + i + 8);
            a2 += *(const vd8u *)(q + i + 16);
            a3 += *(const vd8u *)(q + i + 24);
        }
    }
    a0 += a1;
    a2 += a3;
    a0 += a2;
    l17_sbw_sink = a0[0] + a0[1] + a0[2] + a0[3] + a0[4] + a0[5] + a0[6] + a0[7];
}

static __attribute__((noinline)) void l17_sbw_wr(double *restrict d, long nvol)
{
    const vd8u w = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    for (long v = 0; v < nvol; ++v) {
        double *q = d + v * 9826;
        for (long i = 0; i + 32 <= 9826; i += 32) {
            *(vd8u *)(q + i) = w;
            *(vd8u *)(q + i + 8) = w;
            *(vd8u *)(q + i + 16) = w;
            *(vd8u *)(q + i + 24) = w;
        }
    }
}

static __attribute__((noinline)) void
l17_sbw_cp(const double *restrict s, double *restrict d, long nvol)
{
    const vd8u w = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    vd8u a0 = {0}, a1 = {0}, a2 = {0}, a3 = {0};
    for (long v = 0; v < nvol; ++v) {
        const double *q = s + v * 9826;
        double *r = d + v * 9826;
        for (long i = 0; i + 32 <= 9826; i += 32) { /* read burst = X pass */
            a0 += *(const vd8u *)(q + i);
            a1 += *(const vd8u *)(q + i + 8);
            a2 += *(const vd8u *)(q + i + 16);
            a3 += *(const vd8u *)(q + i + 24);
        }
        for (long i = 0; i + 32 <= 9826; i += 32) { /* write burst = planes */
            *(vd8u *)(r + i) = w;
            *(vd8u *)(r + i + 8) = w;
            *(vd8u *)(r + i + 16) = w;
            *(vd8u *)(r + i + 24) = w;
        }
    }
    a0 += a1;
    a2 += a3;
    a0 += a2;
    l17_sbw_sink = a0[0] + a0[1] + a0[2] + a0[3] + a0[4] + a0[5] + a0[6] + a0[7];
}

static __attribute__((noinline)) void l17_sbw_s17(const double *restrict s, long nvol)
{
    vd8u a0 = {0}, a1 = {0}, a2 = {0}, a3 = {0};
    for (long v = 0; v < nvol; ++v) {
        const double *q = s + v * 9826;
        for (long i = 0; i < 72; ++i) { /* chunk column i, rows 0..16 */
            const double *c = q + 8 * i;
            for (int r = 0; r < 17; ++r) {
                vd8u t = *(const vd8u *)(c + (long)r * 578);
                switch (r & 3) {
                case 0: a0 += t; break;
                case 1: a1 += t; break;
                case 2: a2 += t; break;
                default: a3 += t; break;
                }
            }
        }
    }
    a0 += a1;
    a2 += a3;
    a0 += a2;
    l17_sbw_sink = a0[0] + a0[1] + a0[2] + a0[3] + a0[4] + a0[5] + a0[6] + a0[7];
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

/* Allocate the coefficient/scratch block and fill every table.  Factored out
 * of fft3d_create() for the multicore phase: each thread's child plan calls
 * this from inside the parallel region, so its scratch is first-touched (and
 * therefore homed) on that thread's own socket.  astab_src != NULL copies the
 * master's de-aliasing tables instead of rebuilding them (they are pure
 * integer arithmetic, identical on every thread). */
static int l17_init_block(fft3d_plan *p, const void *astab_src)
{
    const size_t nc = 8 * 9, ns = 8 * 8;
    const size_t nnc = 4 * 8, nns = 8 * 8; /* nested kernel coefficient rows */
    const size_t nsc = 17 * 8;
    const size_t npb = 17 * 20 * 2;
    const size_t nt1 = 4913 * 2;
    /* t1/t1b are carved oversized (round panel_r8): 17 planes at the padded
     * stride (640 doubles) plus 512 doubles (4 KiB) of per-volume base-shift
     * slack for the address-safe variants. */
    const size_t nt1e = 17 * 640 + 512 + 64;
    /* staged-output stages (round panel_r11): a dense volume image (9826
     * doubles) plus 4 KiB of per-volume base-shift slack plus alignment. */
    const size_t nso = 9826 + 512 + 64;
    const size_t ntot = 12 * nc + 12 * ns + 12 * nnc + 12 * nns + nsc +
                        2 * npb + 2 * nt1e + 2 * nso + nt1 + 1024;
    void *blk = NULL;
    if (posix_memalign(&blk, 64, ntot * sizeof(double)) != 0 || !blk)
        return 0;
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
    p->t1 = q; q += nt1e;
    q = L17_ALIGN64(q);
    p->t1b = q; q += nt1e;
    q = L17_ALIGN64(q);
    p->so0 = q; q += nso;
    q = L17_ALIGN64(q);
    p->so1 = q; q += nso;
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

    /* de-aliasing shift tables for the address-safe twins -- must exist
     * before the tuner runs them.  Pure integer arithmetic, ~30 ms. */
    if (astab_src)
        memcpy(p->astab, astab_src, sizeof p->astab);
    else
        l17_as_build(p->astab);
    return 1;
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != 17 || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    if (!l17_init_block(p, NULL)) {
        free(p);
        return NULL;
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
        enum { L17_NCAND = 50 };
        /* round mt_r1: when OpenMP gives us a team, the batch>=64 kernel
         * choice moves to the MULTITHREADED race below (the single-thread
         * ranking does not predict the 32-thread, bandwidth-shared one), so
         * the old single-thread stage 1b/2 run only as a no-OpenMP fallback. */
#ifdef _OPENMP
        const int l17mt_on = omp_get_max_threads() > 1;
#else
        const int l17mt_on = 0;
#endif
        static const l17_fn cand[L17_NCAND] = {
            exec_w4, exec2_w4, exec3_w4, exec4_w4, exec6_w4, exec7_w4,
            exec10_w4, exec11_w4, exec12_w4, exec13_w4, exec14_w4, exec15_w4,
            exec_w2, exec2_w2, exec3_w2, exec4_w2, exec6_w2, exec7_w2,
            exec10_w2, exec11_w2, exec12_w2, exec13_w2, exec14_w2, exec15_w2,
            exec18_w4, exec19_w4, exec18_w2, exec19_w2,
            exec17_w4, exec17_w2, exec20_w4, exec20_w2,
            l17_execm_xl, l17_execm_xlp, l17_execm_xf, l17_execm_pipe,
            l17_execm_xld, l17_execm_xfd,
            l17_execm_xlc, l17_execm_xldc, l17_execm_xfc, l17_execm_xfdc,
            l17_execm_xla, l17_execm_xlda, l17_execm_xfa, l17_execm_xfda,
            l17_execm_xfsa, l17_execm_xfdsa,
            l17_execm_xfso, l17_execm_xfdso,
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
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, cos-resident",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, cos-resident, deferred-Z",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, cos-resident, X-first",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, cos-resident, X-first, deferred-Z",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, addr-safe t1",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, deferred-Z, addr-safe t1",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, X-first, addr-safe t1",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, X-first, deferred-Z, addr-safe t1",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, X-first, staged input, addr-safe t1",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, X-first, deferred-Z, staged input, addr-safe t1",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, X-first, staged output, addr-safe t1",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, X-first, deferred-Z, staged output, addr-safe t1",
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
        const int selB[9] = {8, 10, 20, 22, 32, 33, 36, 42, 43}; /* X-last pinned */
        const int selD[21] = {9, 11, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                              34, 35, 37, 44, 45, 46, 47, 48, 49};
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
         * before being added.
         * r7: the cosine-resident variants 38/39 (X-last) and 40/41 (X-first)
         * are NOT in either class: cmp on full outputs (wallaby, B=8 and
         * B=256) shows they are a different rounding fixed point (PASS at
         * 3.258e-16, but bits differ -- gcc contracts the unrolled cosine
         * differently, the same phenomenon that made the r3 pinned-sine path
         * its own class; on paper fnmadd(c,q,B) == fmadd(-c,q,B) exactly, and
         * the divergence is compiler freedom, not the sign algebra).  They
         * are measured for the verbose table and forceable (-DL17_FORCE=44..
         * 47); if the node shows a win, the class rule itself moves next
         * round.  Never add them to selB/selD without re-deriving the class
         * structure. */
        /* r8: the address-safe twins 42/43 (X-last -> class B) and 44/45
         * (X-first -> class D) change ADDRESSES only (padded t1 plane stride
         * + execute-time base offset); cmp-verified bit-identical to 32
         * resp. 34 on full outputs before being added here. */
        /* r10: the staged-input twins 46/47 (X-first -> class D) copy the
         * next volume's input bit-exactly into an L2 stage and run the same
         * chunks in the same order on the same values; cmp-verified
         * bit-identical to 44 resp. 45 on full outputs before being added. */
        /* r11: the staged-OUTPUT twins 48/49 (X-first -> class D) store the
         * same bits at a staged address and flush them by bit-exact copy;
         * cmp-verified bit-identical to 44 resp. 45 on full outputs before
         * being added. */
        const int *sel = (batch >= 64) ? selD : selB;
        const int nsel = (batch >= 64) ? 21 : 9;
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

        /* stage 1c (batch < 64, round panel_r9): A/B the in-pass source
         * prefetch (pt) on the stage-1 winner, blocked, two warmups --
         * prefetches change no bits, so the choice is free within the class.
         * Only the mixed execs carry the hook. */
        if (batch < 64) {
            const int haspt = bestv == 32 || bestv == 33 || bestv == 36 ||
                              bestv == 42 || bestv == 43;
            int nv = batch < 16 ? batch : 16;
            if (haspt && l17_tune_alloc(p, nv)) {
                int inner = (64 + nv - 1) / nv;
                int sb = p->batch;
                p->batch = nv;
                double bt[2];
                for (int ft = 0; ft < 2; ++ft) {
                    p->pt = ft;
                    bt[ft] = 1e30;
                    p->exec(p, p->ti, p->to);
                    p->exec(p, p->ti, p->to);
                    for (int r = 0; r < 3; ++r) {
                        double t0 = l17_now();
                        for (int q2 = 0; q2 < inner; ++q2) p->exec(p, p->ti, p->to);
                        double dt = l17_now() - t0;
                        if (dt < bt[ft]) bt[ft] = dt;
                    }
                }
                p->batch = sb;
                p->pt = bt[1] < bt[0];
                if (l17_verbose())
                    fprintf(stderr, "[L17_matrixsimd tune] pt A/B: pt=0 %.2f  "
                                    "pt=1 %.2f us/transform  -> pt=%d\n",
                            bt[0] * 1e6 / (nv * inner), bt[1] * 1e6 / (nv * inner),
                            p->pt);
                static char g_desc_pt[200];
                snprintf(g_desc_pt, sizeof g_desc_pt, "%s, pt=%d", g_desc, p->pt);
                g_desc = g_desc_pt;
            } else {
                p->pt = 0;
            }
        }

        /* stage 1b (batch >= 64): tune on a working set that actually leaves
         * L3, in consecutive blocks.  Round 2's stage 1 tuned on 16 volumes
         * (2.4 MB -- L3-resident) even at B=2048 and then only A/B'd the NT
         * store on that winner; but the streaming regime can prefer a
         * different kernel outright.  Borrowed from L17_winograd round 2,
         * which measured a 90% penalty for trusting the small-set pick at
         * batch. */
        if (batch >= 64 && !l17mt_on) {
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
        if (batch >= 64 && !l17mt_on) {
            const int pipewin = (bestv >= 24 && bestv <= 27) ||
                                bestv == 30 || bestv == 31 || bestv == 35;
            /* r10: the staged twins' sequential copy IS the cross-volume
             * input touch, so pf is meaningless there (as for pipelined);
             * their X pass reads the L2-hot stage, so pt is not offered. */
            const int stagewin = bestv == 46 || bestv == 47;
            /* r11: the staged-output twins pace the flush write stream
             * through the plane phase; pf would push volume b+1's input
             * reads into that same phase -- exactly the read/write mix the
             * node's cp says serializes -- so pf is not offered there. */
            const int sowin = bestv == 48 || bestv == 49;
            const int haspf = !pipewin && !stagewin && !sowin;
            const int haspw = bestv == 9 || bestv == 11 || bestv == 21 ||
                              bestv == 23 || bestv == 28 || bestv == 29 ||
                              bestv == 34 || bestv == 35 || bestv == 37 ||
                              bestv == 40 || bestv == 41 ||
                              bestv == 44 || bestv == 45 ||
                              bestv == 46 || bestv == 47 ||
                              bestv == 48 || bestv == 49;
            /* r9: the in-pass source prefetch, offered to the mixed X-first
             * non-pipelined execs.  In the (pf, pw, pt) grid jointly, per
             * the r4 lesson that interacting knobs cannot be raced
             * sequentially. */
            const int haspt = bestv == 34 || bestv == 37 ||
                              bestv == 44 || bestv == 45;
            int nv2 = l17_tune_nv(batch);
            if ((haspf || haspw || haspt) && l17_tune_alloc(p, nv2)) {
                int sb = p->batch;
                p->batch = nv2;
                double bt = 1e30;
                int bpf = 0, bpw = 0, bpt = 0;
                for (int fp = 0; fp <= haspf; ++fp)
                    for (int fw = 0; fw <= haspw; ++fw)
                        for (int ft = 0; ft <= haspt; ++ft) {
                            p->pf = fp;
                            p->pw = fw;
                            p->pt = ft;
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
                                                "pf=%d pw=%d pt=%d  %.2f us/transform\n",
                                        nv2, fp, fw, ft, bb * 1e6 / nv2);
                            if (bb < bt) { bt = bb; bpf = fp; bpw = fw; bpt = ft; }
                        }
                p->batch = sb;
                p->pf = bpf;
                p->pw = bpw;
                p->pt = bpt;
                static char g_desc_buf[200];
                snprintf(g_desc_buf, sizeof g_desc_buf, "%s, pf=%d, pw=%d, pt=%d",
                         g_desc, p->pf, p->pw, p->pt);
                g_desc = g_desc_buf;
            } else {
                p->pf = 0;
                p->pw = 0;
                p->pt = 0;
            }
        }
        l17_tune_free(p);
#if defined(L17_FORCE)
        /* development override: -DL17_FORCE=0..55 pins one variant;
         * -DL17_FORCE_PF / -DL17_FORCE_PW / -DL17_FORCE_PT pin the
         * prefetch flags */
        {
            static const l17_fn all[56] = {
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
                l17_execm_xlc, l17_execm_xldc,     /* 44,45: cos-resident X-last (r7) */
                l17_execm_xfc, l17_execm_xfdc,     /* 46,47: cos-resident X-first (r7) */
                l17_execm_xla, l17_execm_xlda,     /* 48,49: addr-safe X-last (r8) */
                l17_execm_xfa, l17_execm_xfda,     /* 50,51: addr-safe X-first (r8) */
                l17_execm_xfsa, l17_execm_xfdsa,   /* 52,53: staged input (r10) */
                l17_execm_xfso, l17_execm_xfdso,   /* 54,55: staged output (r11) */
            };
            p->exec = all[L17_FORCE];
            g_desc = "forced variant (L17_FORCE)";
#  ifdef L17_FORCE_PF
            p->pf = L17_FORCE_PF;
#  endif
#  ifdef L17_FORCE_PW
            p->pw = L17_FORCE_PW;
#  endif
#  ifdef L17_FORCE_PT
            p->pt = L17_FORCE_PT;
#  endif
        }
#endif

#ifdef _OPENMP
        /* ============== round mt_r1: the multicore layer ==============
         * Children first (per-thread plans, NUMA-local scratch), then the
         * (mode, kernel, team, schedule, flags) choice by timing the REAL
         * parallel path on this machine.  Every candidate raced within one
         * batch regime is in that regime's bit class, so the wall-clock
         * pick cannot change the output bits (the phase-1 discipline,
         * carried over unchanged). */
        if (l17mt_on) {
            int maxt = omp_get_max_threads();
            if (maxt > L17MT_MAXT) maxt = L17MT_MAXT;
            {   /* dev-only timing decomposition -- read before the races so
                 * the verbose race table reflects it (output goes wrong) */
                const char *es = getenv("L17MT_SKIP");
                if (es && *es) p->mtskip = atoi(es);
            }
            p->nkids = maxt;
            p->kids = calloc((size_t)maxt, sizeof *p->kids);
            cpu_set_t *masks = calloc((size_t)maxt, sizeof *masks);
            int *havem = calloc((size_t)maxt, sizeof *havem);
            int okk = p->kids && masks && havem;
            if (okk) {
                int fail = 0;
#pragma omp parallel num_threads(maxt) reduction(||: fail)
                {
                    int t = omp_get_thread_num();
                    fft3d_plan *k = calloc(1, sizeof *k);
                    if (k) {
                        k->L = 17;
                        k->batch = 1;
                        if (!l17_init_block(k, p->astab)) {
                            free(k);
                            k = NULL;
                        }
                    }
                    if (!k) fail = 1;
                    p->kids[t] = k;
                    /* capture this OMP thread's binding so the pool worker
                     * of the same index runs on exactly the same core(s) */
                    if (sched_getaffinity(0, sizeof(cpu_set_t), &masks[t]) == 0)
                        havem[t] = 1;
                }
                okk = !fail;
            }
            if (okk) {
                p->poolv = l17mt_pool_new(maxt, masks, havem);
                okk = p->poolv != NULL;
            }
            /* mt_r3: aggregate LLC reachable by this team (per-package L3 x
             * distinct packages), read while the captured masks are alive --
             * the input to the working-set gate below */
            const size_t wg_aggl3 = okk ? l17wg_aggl3(masks, havem, maxt) : 0;
            free(masks);
            free(havem);
            if (okk && batch < 64) {
                /* intra-volume t1: batch padded volumes, first-touched by the
                 * full-team plane->thread map for the race; re-homed with the
                 * PICKED team's map after the mode race below (mt_r2) */
                l17mt_t1g_map(p, batch, maxt);
            }

            /* ============ round mt_r3: the working-set gate ============
             * Above the aggregate-cache threshold the streaming engine is
             * INSTALLED, not raced: this entry's create-time arena misranked
             * the streaming cell in both prior rounds (r1 NT+pipelined
             * arena-picked, scored 2.106 us/t; r2 plain X-first arena-picked,
             * scored 2.858 -- while L17_winograd's engine held 1.220 on the
             * same node in both rounds), and the mt_r2 VERDICT's L=17
             * directive is exactly this: "the threshold set from the working
             * set rather than from an arena".  The gate is deterministic --
             * same machine, same batch => same path in every process -- so
             * repeatability is by construction, not by racing discipline.
             * Threshold: in+out > 1.5x aggregate L3 (node: 66 MB, so B=256's
             * 40 MB working set stays on the dense X-first path that WON that
             * cell; wallaby: 90 MB, same split), floored at 96 MB in case
             * sysconf/sysfs will not answer.  dyn is gone here: dyn != 0 lost
             * in every scored process that picked it, panel-wide (VERDICT
             * mt_r2 section 6). */
            int wgon = 0;
            if (okk && batch >= 64) {
                const size_t wg_ws = (size_t)batch * (size_t)(4913 * 16 * 2);
                size_t wg_thr = wg_aggl3 + wg_aggl3 / 2;
                if (wg_thr < ((size_t)96 << 20)) wg_thr = (size_t)96 << 20;
                wgon = wg_ws > wg_thr;
                {   /* dev-only override for wallaby A/Bs; harness never sets it */
                    const char *ew = getenv("L17MT_WG");
                    if (ew && *ew) wgon = atoi(ew) != 0;
                }
                if (wgon && !l17wg_setup(p, maxt)) wgon = 0;
                if (wgon) {
                    p->mode = 1;
                    p->nthr = maxt;
                    p->dynb = 0;
                    p->pf = p->pw = p->pt = 0;
                    p->wgi4 = 1; /* split-free pass 1: the node's 2-load-port
                                  * CLX pays double fill resources for cold
                                  * line-splitting loads (L17_rader's "dy" was
                                  * node-picked at batch); wallaby measures i4
                                  * ~1.5% behind h4 and cannot price the cold
                                  * case -- see the strategy record */
                    {
                        const char *ei = getenv("L17MT_WGI4");
                        if (ei && *ei) p->wgi4 = atoi(ei) != 0;
                    }
                    for (int t = 0; t < p->nkids; ++t)
                        if (p->kids[t]) {
                            p->kids[t]->exec = l17_execm_wgh4;
                            p->kids[t]->pf = 0;
                            p->kids[t]->pw = 0;
                            p->kids[t]->pt = 0;
                            p->kids[t]->wgi4 = p->wgi4;
                        }
                    static char g_desc_wg[380];
                    snprintf(g_desc_wg, sizeof g_desc_wg,
                             "17-pt rotating-pass fused engine (adopted from "
                             "L17_winograd, %s), mt[wg nt=%d static "
                             "ws=%zuMB aggl3=%zuMB]",
                             p->wgi4 ? "i4" : "h4", p->nthr,
                             wg_ws >> 20, wg_aggl3 >> 20);
                    g_desc = g_desc_wg;
                }
            }

            if (!wgon && okk && batch >= 64 &&
                l17_tune_alloc_mt(p, batch < 1024 ? batch : 1024, maxt)) {
                const int nv = batch < 1024 ? batch : 1024;
                /* stage A: kernel race inside bit class D under the full
                 * team, on the owner-touched arena (see l17_tune_alloc_mt),
                 * under STATIC scheduling.  r1 raced under dyn=2 because the
                 * serial-touch arena starved the remote socket and dynamic
                 * hid that; with owner-local pages there is no imbalance to
                 * hide, static is what the r1 node picked at every batched
                 * cell panel-wide, and the ranking should be taken in the
                 * schedule that will actually run.  dyn stays in stage B. */
                p->mode = 1;
                p->nthr = maxt;
                p->dynb = 0;
                int mbest = sel[0];
                double bt = 1e30;
                for (int s = 0; s < nsel; ++s) {
                    int v = sel[s];
                    l17mt_set_kids(p, cand[v], 0, 0, 0);
                    double tt = l17mt_time_cfg(p, nv);
                    if (l17_verbose())
                        fprintf(stderr, "[L17_matrixsimd mtA nt=%d nv=%d] %-72s %8.3f us/t\n",
                                maxt, nv, tags[v], tt * 1e6);
                    if (tt < bt) { bt = tt; mbest = v; }
                }
                l17mt_set_kids(p, cand[mbest], 0, 0, 0);
                bestv = mbest;
                g_desc = tags[mbest];
                /* stage B: team size x schedule on the winner.  Fewer threads
                 * can win when the caller's serially-first-touched buffers
                 * leave the remote socket bandwidth-starved; dynamic blocks
                 * rebalance that at the cost of one atomic per grab. */
                {
                    const int ntc[4] = {maxt, (3 * maxt) / 4, maxt / 2, maxt / 4};
                    const int dyc[4] = {0, 1, 2, 4};
                    int bnt = maxt, bdy = 2;
                    bt = 1e30;
                    for (int a = 0; a < 4; ++a) {
                        int dup = ntc[a] < 1;
                        for (int a2 = 0; a2 < a; ++a2)
                            if (ntc[a2] == ntc[a]) dup = 1;
                        if (dup) continue;
                        for (int d = 0; d < 4; ++d) {
                            p->nthr = ntc[a];
                            p->dynb = dyc[d];
                            double tt = l17mt_time_cfg(p, nv);
                            if (l17_verbose())
                                fprintf(stderr, "[L17_matrixsimd mtB] nt=%d dyn=%d  %8.3f us/t\n",
                                        ntc[a], dyc[d], tt * 1e6);
                            if (tt < bt) { bt = tt; bnt = ntc[a]; bdy = dyc[d]; }
                        }
                    }
                    p->nthr = bnt;
                    p->dynb = bdy;
                }
                /* stage C: (pf,pw,pt) jointly on the final configuration --
                 * same grid as the phase-1 stage 2, timed multithreaded */
                {
                    const int pipew = (mbest >= 24 && mbest <= 27) ||
                                      mbest == 30 || mbest == 31 || mbest == 35;
                    const int hpf = !pipew && mbest != 46 && mbest != 47 &&
                                    mbest != 48 && mbest != 49;
                    const int hpw = mbest == 9 || mbest == 11 || mbest == 21 ||
                                    mbest == 23 || mbest == 28 || mbest == 29 ||
                                    mbest == 34 || mbest == 35 || mbest == 37 ||
                                    (mbest >= 44 && mbest <= 49);
                    const int hpt = mbest == 34 || mbest == 37 ||
                                    mbest == 44 || mbest == 45;
                    int bpf = 0, bpw = 0, bpt = 0;
                    if (hpf || hpw || hpt) {
                        bt = 1e30;
                        for (int fp = 0; fp <= hpf; ++fp)
                            for (int fw = 0; fw <= hpw; ++fw)
                                for (int ft = 0; ft <= hpt; ++ft) {
                                    l17mt_set_kids(p, cand[mbest], fp, fw, ft);
                                    double tt = l17mt_time_cfg(p, nv);
                                    if (l17_verbose())
                                        fprintf(stderr, "[L17_matrixsimd mtC] pf=%d pw=%d pt=%d  %8.3f us/t\n",
                                                fp, fw, ft, tt * 1e6);
                                    if (tt < bt) { bt = tt; bpf = fp; bpw = fw; bpt = ft; }
                                }
                    }
                    l17mt_set_kids(p, cand[mbest], bpf, bpw, bpt);
                    p->pf = bpf;
                    p->pw = bpw;
                    p->pt = bpt;
                }
                static char g_desc_mtv[380];
                snprintf(g_desc_mtv, sizeof g_desc_mtv,
                         "%s, mt[vol nt=%d dyn=%d pf=%d pw=%d pt=%d ar=ot]",
                         g_desc, p->nthr, p->dynb, p->pf, p->pw, p->pt);
                g_desc = g_desc_mtv;
            }

            if (okk && batch < 64 && l17_tune_alloc(p, batch)) {
                /* small batch: single-thread vs volume-parallel vs the
                 * intra-volume decomposition, all bit class B */
                l17mt_set_kids(p, p->exec, 0, 0, p->pt);
                int bmode = 0, bnt = 1;
                p->mode = 0;
                double bt = l17mt_time_cfg(p, batch);
                if (l17_verbose())
                    fprintf(stderr, "[L17_matrixsimd mt] single       %8.3f us/t\n", bt * 1e6);
                if (batch > 1) {
                    int nt = batch < maxt ? batch : maxt;
                    p->mode = 1;
                    p->nthr = nt;
                    p->dynb = 0;
                    double tt = l17mt_time_cfg(p, batch);
                    if (l17_verbose())
                        fprintf(stderr, "[L17_matrixsimd mt] vol nt=%-3d   %8.3f us/t\n", nt, tt * 1e6);
                    if (tt < bt) { bt = tt; bmode = 1; bnt = nt; }
                }
                if (p->t1g) {
                    static const int ntl[9] = {1, 2, 4, 8, 12, 16, 17, 24, 32};
                    p->mode = 2;
                    p->nxr = 0; /* = nthr */
                    p->xpf = 0;
                    for (int a = 0; a < 9 && ntl[a] <= maxt; ++a) {
                        p->nthr = ntl[a];
                        double tt = l17mt_time_cfg(p, batch);
                        if (l17_verbose())
                            fprintf(stderr, "[L17_matrixsimd mt] intra nt=%-3d %8.3f us/t\n",
                                    ntl[a], tt * 1e6);
                        if (tt < bt) { bt = tt; bmode = 2; bnt = ntl[a]; }
                    }
                }
                int bnx = 0, bxpf = 0;
                if (bmode == 2) {
                    /* mt_r2: re-home t1g with the picked team's map BEFORE
                     * the nxr/xpf race, so that race (and every execute
                     * after it) runs on pages homed where the team is */
                    l17mt_t1g_map(p, batch, bnt);
                    if (!p->t1g) bmode = 0;
                }
                if (bmode == 2) {
                    /* the two phases saturate at different team sizes (the
                     * plane phase scales to ~17 threads, the X phase stops
                     * near 4 -- see strategies), so race the X-phase team,
                     * and the cross-core prefetch shape (mt_r2: 0 = none,
                     * 1 = 2-chunks-ahead, 2 = bulk at barrier exit) on the
                     * winning nthr */
                    static const int nxl[5] = {2, 4, 8, 12, 0};
                    p->mode = 2;
                    p->nthr = bnt;
                    for (int a = 0; a < 5; ++a) {
                        int nx = nxl[a] ? nxl[a] : bnt;
                        if (nx > bnt || (nx == bnt && nxl[a])) continue;
                        for (int fx = 0; fx <= 2; ++fx) {
                            p->nxr = nx;
                            p->xpf = fx;
                            double tt = l17mt_time_cfg(p, batch);
                            if (l17_verbose())
                                fprintf(stderr, "[L17_matrixsimd mt] intra nt=%d nxr=%-3d xpf=%d %8.3f us/t\n",
                                        bnt, nx, fx, tt * 1e6);
                            if (tt < bt) { bt = tt; bnx = nx; bxpf = fx; }
                        }
                    }
                }
                p->mode = bmode;
                p->nthr = bnt;
                p->nxr = bnx;
                p->xpf = bxpf;
                p->dynb = 0;
                static char g_desc_mts[380];
                snprintf(g_desc_mts, sizeof g_desc_mts, "%s, mt[%s nt=%d nxr=%d xpf=%d]",
                         g_desc,
                         bmode == 0 ? "single" : bmode == 1 ? "vol" : "intra",
                         bnt, bnx ? bnx : bnt, bxpf);
                g_desc = g_desc_mts;
            }

            /* dev-only overrides (never set by the harness), for bit-class
             * verification: L17MT_MODE / L17MT_NT pin mode and team size. */
            {
                const char *em = getenv("L17MT_MODE");
                const char *en = getenv("L17MT_NT");
                const char *es = getenv("L17MT_SKIP");
                const char *ex = getenv("L17MT_NXR");
                if (em && *em) p->mode = atoi(em);
                if (en && *en) p->nthr = atoi(en);
                if (es && *es) p->mtskip = atoi(es);
                if (ex && *ex) p->nxr = atoi(ex);
                if (p->mode != 0 && p->nthr < 1) p->nthr = 1;
                if (p->mode == 2 && !p->t1g) p->mode = 0;
                if (p->mode != 0 && (!okk || !p->kids || !p->poolv)) p->mode = 0;
            }
        }
#endif /* _OPENMP */
    }
    if (batch >= 64) {
        /* ROUND panel_r10: the streaming bandwidth decomposition (see the
         * comment at l17_sbw_rd).  Four numbers, us per 78.6 KiB
         * volume-equivalent, on the >L3 tuner arena, blocked, min of 3 with
         * a discarded warmup rep.  The transform's compulsory DRAM work per
         * volume is one rd plus one wr, alternated per volume = cp; the
         * scored streaming cell against cp is the whole remaining question:
         * cell ~= cp means the cells are bandwidth-closed on this machine,
         * cell >> cp means scheduling headroom, and (s17 - rd) prices the
         * staged-input twins' mechanism specifically. */
        int nvp = l17_tune_nv(batch);
        if (l17_tune_alloc(p, nvp)) {
            const double *ps = (const double *)p->ti;
            double *pd = (double *)p->to;
            double us[4];
            for (int m = 0; m < 4; ++m) {
                double bestp = 1e30;
                for (int r = 0; r < 4; ++r) { /* rep 0 = warmup, discarded */
                    double t0 = l17_now();
                    switch (m) {
                    case 0: l17_sbw_rd(ps, nvp); break;
                    case 1: l17_sbw_wr(pd, nvp); break;
                    case 2: l17_sbw_cp(ps, pd, nvp); break;
                    default: l17_sbw_s17(ps, nvp); break;
                    }
                    double dt = l17_now() - t0;
                    if (r > 0 && dt < bestp) bestp = dt;
                }
                us[m] = bestp * 1e6 / nvp;
            }
            static char g_desc_sbw[420];
            snprintf(g_desc_sbw, sizeof g_desc_sbw,
                     "%s, sbw[rd/wr/cp/s17]=%.2f/%.2f/%.2f/%.2f",
                     g_desc, us[0], us[1], us[2], us[3]);
            g_desc = g_desc_sbw;
            if (l17_verbose())
                fprintf(stderr, "[L17_matrixsimd sbw nv=%d] rd=%.2f wr=%.2f "
                                "cp=%.2f s17=%.2f us/vol\n",
                        nvp, us[0], us[1], us[2], us[3]);
        }
    }
    { /* ROUND panel_r9: the in-plan B=1 residual decomposition (see the
       * comment at l17_probe_yz).  Four numbers, us per volume-equivalent:
       *   yz  = plane phase in situ (source = a 78.6 KiB volume, L2)
       *   kyz = the same instruction stream, L1-hot source/dest (ss=ds=0)
       *   x   = X pass in situ (source = t1, 87 KiB, L2)
       *   kx  = the same instruction stream, L1-hot source (ss=34)
       * Port floors at 2.89 GHz for calibration: yz 7.83 us, x 3.71 us.
       * (kyz ~ floor, x >> kx)  => fill latency; expect pt=1 to be picked.
       * (kyz and kx both high)  => the kernel itself carries the residual
       * from L1 -- B=1 is at its structural limit for this kernel family. */
        if (l17_tune_alloc(p, 1)) {
            const double *ps = (const double *)p->ti;
            double *pd = (double *)p->to;
            double us[4];
            l17_probe_yz(p, ps, 578, p->t1, 640); /* fill t1 for the X probes */
            for (int m = 0; m < 4; ++m) {
                const int inner = 128;
                double bestp = 1e30;
                for (int r = 0; r < 4; ++r) { /* rep 0 = warmup, discarded */
                    double t0 = l17_now();
                    for (int i = 0; i < inner; ++i) {
                        switch (m) {
                        case 0: l17_probe_yz(p, ps, 578, p->t1, 640); break;
                        case 1: l17_probe_yz(p, ps, 0, p->t1, 0); break;
                        case 2: l17_probe_x(p, p->t1, 640, pd); break;
                        default: l17_probe_x(p, p->t1, 34, pd); break;
                        }
                    }
                    double dt = l17_now() - t0;
                    if (r > 0 && dt < bestp) bestp = dt;
                }
                us[m] = bestp * 1e6 / inner;
            }
            static char g_desc_b1[460];
            snprintf(g_desc_b1, sizeof g_desc_b1,
                     "%s, b1dec[yz/kyz/x/kx]=%.2f/%.2f/%.2f/%.2f",
                     g_desc, us[0], us[1], us[2], us[3]);
            g_desc = g_desc_b1;
        }
        l17_tune_free(p);
    }
    { /* measured sustained licence clocks (the monitor's r4 ask for L=17);
       * carried back on the leaderboard via the description string.  r6 adds
       * d256 = the DENSE (2 FMA/cycle) 256-bit clock, same process, to close
       * the r5 VERDICT's clk256 question: sparse-256 3.89 vs dense-256 2.89
       * in one string would confirm density as the licence discriminator.
       * mt_r2 adds anb = /proc/sys/kernel/numa_balancing on the scoring
       * node (-1 if unreadable): the mt_r1 VERDICT's leading explanation
       * for the >=500 MiB band's page-placement behaviour is AutoNUMA
       * migration, named as a hypothesis needing exactly this check. */
        double c512 = l17_clk512(), c256 = l17_clk256(), c256d = l17_clk256d();
        int anb = -1;
        {
            FILE *f = fopen("/proc/sys/kernel/numa_balancing", "r");
            if (f) {
                if (fscanf(f, "%d", &anb) != 1) anb = -1;
                fclose(f);
            }
        }
        static char g_desc_clk[520];
        snprintf(g_desc_clk, sizeof g_desc_clk,
                 "%s, clk512/256=%.2f/%.2f GHz, d256=%.2f, anb=%d",
                 g_desc, c512 * 1e-9, c256 * 1e-9, c256d * 1e-9, anb);
        g_desc = g_desc_clk;
    }
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    l17_tune_free(p); /* normally already freed at the end of create() */
#ifdef _OPENMP
    l17mt_pool_free(p->poolv);
#endif
    if (p->kids) {
        for (int t = 0; t < p->nkids; ++t)
            if (p->kids[t]) {
                free(p->kids[t]->wgbuf_raw); /* mt_r3 streaming scratch */
                free(p->kids[t]->block);
                free(p->kids[t]);
            }
        free(p->kids);
    }
    free(p->wgt_raw); /* mt_r3 shared wg tables (master-owned) */
#ifdef _OPENMP
    if (p->t1g_raw) munmap(p->t1g_raw, p->t1g_sz);
#endif
    free(p->block);
    free(p);
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    /* dev-only diagnostic (L17_ASDBG=1): print the page-offset relations that
     * the de-aliasing analysis cares about, once per (in,out) pair. */
    static int asdbg = -1;
    if (asdbg < 0) {
        const char *e = getenv("L17_ASDBG");
        asdbg = (e && *e && *e != '0');
    }
    if (asdbg) {
        static const void *li, *lo;
        if (li != in || lo != out) {
            li = in; lo = out;
            fprintf(stderr, "[L17 asdbg] (in-t1)&4095=%ld (out-t1)&4095=%ld (out-in)&4095=%ld\n",
                    (long)(((uintptr_t)in - (uintptr_t)plan->t1) & 4095u),
                    (long)(((uintptr_t)out - (uintptr_t)plan->t1) & 4095u),
                    (long)(((uintptr_t)out - (uintptr_t)in) & 4095u));
        }
    }
    l17mt_dispatch(plan, in, out);
}

#endif /* L17_TEMPLATE_PASS */
