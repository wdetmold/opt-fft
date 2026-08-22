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
SUF(chunk17n_g)(const double *restrict src, long rs, double *restrict dst, long da,
                long db, const double *restrict cn, const double *restrict sn,
                double *restrict sc,
                VT K0, VT K1, VT K2, VT K3, VT K4, VT K5, VT K6, VT K7,
                int tr, int pc, int pin, int cr, int xst)
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
#if WC == 4 && defined(__AVX512F__)
    } else if (xst) {
        /* ---- ROUND ice_r1: EXTRACT-STORE transpose.  On Ice Lake both
         * 512-bit FMA pipes are ports 0+5 and every 512-bit shuffle is
         * port-5-only, so the 40 vshuff64x2 of the tile transpose steal
         * port slots from the second FMA pipe (on CLX port 5 was idle and
         * the transpose was free -- that assumption is machine-specific).
         * The memory-destination form of vextractf64x2 executes as a plain
         * store (store-data + store-AGU, no shuffle uop), and Ice Lake
         * commits 2 stores/cycle: 68 16-byte stores replace 40 shuffles +
         * 20 stores.  Values stored are bit-identical lanes of the same
         * vectors, so this stays inside the bit class (cmp-verified). ---- */
#  define XST1(m, e)                                                           \
      do {                                                                     \
          __m512d z_ = (__m512d)(e);                                           \
          _mm_storeu_pd(dst + 0 * da + (m) * 2, _mm512_castpd512_pd128(z_));   \
          _mm_storeu_pd(dst + 1 * da + (m) * 2, _mm512_extractf64x2_pd(z_, 1));\
          _mm_storeu_pd(dst + 2 * da + (m) * 2, _mm512_extractf64x2_pd(z_, 2));\
          _mm_storeu_pd(dst + 3 * da + (m) * 2, _mm512_extractf64x2_pd(z_, 3));\
      } while (0)
        XST1(0, E0);   XST1(1, E1);   XST1(2, E2);   XST1(3, E3);
        XST1(4, E4);   XST1(5, E5);   XST1(6, E6);   XST1(7, E7);
        XST1(8, E8);   XST1(9, E9);   XST1(10, E10); XST1(11, E11);
        XST1(12, E12); XST1(13, E13); XST1(14, E14); XST1(15, E15);
        XST1(16, E16);
#  undef XST1
#endif
    } else {
        (void)xst;
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

/* The two public forms: chunk17n is the historical kernel (tile-transpose
 * stores), chunk17nx routes the tr=1 transpose through extract-stores
 * (round ice_r1).  Both are thin always_inline wrappers over chunk17n_g,
 * so the arithmetic -- and therefore the bit class -- is shared. */
static inline __attribute__((always_inline)) void
SUF(chunk17n)(const double *restrict src, long rs, double *restrict dst, long da,
              long db, const double *restrict cn, const double *restrict sn,
              double *restrict sc,
              VT K0, VT K1, VT K2, VT K3, VT K4, VT K5, VT K6, VT K7,
              int tr, int pc, int pin, int cr)
{
    SUF(chunk17n_g)(src, rs, dst, da, db, cn, sn, sc,
                    K0, K1, K2, K3, K4, K5, K6, K7, tr, pc, pin, cr, 0);
}

static inline __attribute__((always_inline)) void
SUF(chunk17nx)(const double *restrict src, long rs, double *restrict dst, long da,
               long db, const double *restrict cn, const double *restrict sn,
               double *restrict sc,
               VT K0, VT K1, VT K2, VT K3, VT K4, VT K5, VT K6, VT K7,
               int tr, int pc, int pin, int cr)
{
    SUF(chunk17n_g)(src, rs, dst, da, db, cn, sn, sc,
                    K0, K1, K2, K3, K4, K5, K6, K7, tr, pc, pin, cr, 1);
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
    unsigned char astab[2][256]; /* per-volume t1 base shifts (round panel_r8):
                                  * astab[mode][((t1 - ref) & 4095) >> 4] is the
                                  * 64-byte-step shift minimizing X-pass 4K
                                  * aliasing; mode 0 = X-last (ref = vout),
                                  * mode 1 = X-first (ref = vin). */
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

/* ROUND ice_r1: pre-RA instruction scheduling for every function from here
 * down (the chunk kernels and all exec variants).  GCC does no pre-RA
 * scheduling on x86 by default and keeps text order, so the chunk's phase-
 * serial source (cosine block, then sine block, then combine/stores) reaches
 * the issue queue unmixed; -fschedule-insns interleaves the independent
 * chains and -fsched-pressure keeps it from spilling while it does.  The
 * corpus (docs §10, GCC cures item on prime passes) reports ~+20% from this
 * pair on ICX-class front ends; measured here as a matched A/B on the node's
 * graded chain: 15.732 -> 14.528 us/step (-7.7%).  It must live in the
 * source as a pragma because the Makefile's CFLAGS are fixed.  Outputs are
 * unchanged (FMA contraction happens in combine, before sched1); the graded
 * verify + repeatability checks pass on the flagged build.
 * -DL17_NO_SCHED disables it (matched-A/B hook for dev runs only). */
#ifndef L17_NO_SCHED
#pragma GCC optimize("schedule-insns", "sched-pressure")
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

/* ROUND ice_r1: the same group with the zmm chunks' transposing store routed
 * through memory-destination vextractf64x2 (see chunk17n_g's xst branch) --
 * on Ice Lake the tile transpose's 40 vshuff64x2 are port-5-only and steal
 * slots from the second 512-bit FMA pipe; extract-stores are store-port work.
 * The ymm tail keeps the tile path (256-bit shuffles are not the contended
 * resource, and w2 has no xst form).  Same values stored -> same bit class
 * as L17_MIX_GROUP (cmp-verified on full outputs before being selectable). */
#define L17_MIX_GROUP_X(SRCB, RS, DSTB, DA, PC, CR)                            \
    do {                                                                       \
        int nt_ = 4;                                                           \
        __asm__("" : "+r"(nt_));                                               \
        for (int t_ = 0; t_ < nt_; ++t_) {                                     \
            long f0_ = 4L * t_;                                                \
            chunk17nx_w4((SRCB) + 2 * f0_, (RS), (DSTB) + f0_ * (DA), (DA), 2, \
                         cn8, sn8, sc, K0, K1, K2, K3, K4, K5, K6, K7,         \
                         1, (PC), 1, (CR));                                    \
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
#define L17_EXEC_MXL(NAME, PC, CR, T1S, AS, GRP)                                    \
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
                GRP(pin2, 34, pb, L17_PBROWM, PC, CR);               \
                GRP(pb, L17_PBROWM, pt, 34, PC, CR);                 \
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
#define L17_EXEC_MXF(NAME, PC, CR, T1S, AS, GRP)                                    \
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
                GRP(pin2, 34, pb, L17_PBROWM, PC, CR);               \
                if (p->pw) {                                                   \
                    const char *po = (const char *)pt;                         \
                    for (int q4 = 37; q4 < 73; ++q4)                           \
                        __builtin_prefetch(po + (long)q4 * 64, 1, 3);          \
                }                                                              \
                GRP(pb, L17_PBROWM, pt, 34, PC, CR);                 \
            }                                                                  \
        }                                                                      \
    }

L17_EXEC_MXL(l17_execm_xl, 0, 0, 578, 0, L17_MIX_GROUP)
L17_EXEC_MXL(l17_execm_xlp, 1, 0, 578, 0, L17_MIX_GROUP)
L17_EXEC_MXF(l17_execm_xf, 0, 0, 578, 0, L17_MIX_GROUP)
/* round panel_r7: cosine-register-resident twins (cr=1 in the zmm body) */
L17_EXEC_MXL(l17_execm_xlc, 0, 1, 578, 0, L17_MIX_GROUP)
L17_EXEC_MXF(l17_execm_xfc, 0, 1, 578, 0, L17_MIX_GROUP)
/* round panel_r8: address-safe twins (padded t1 stride + de-aliased base) */
L17_EXEC_MXL(l17_execm_xla, 0, 0, L17_T1SP, 1, L17_MIX_GROUP)
L17_EXEC_MXF(l17_execm_xfa, 0, 0, L17_T1SP, 1, L17_MIX_GROUP)
/* round ice_r1: extract-store twins (Y/Z transposing stores via memory-
 * destination vextractf64x2 -- port-5 relief on 2x512-FMA parts) */
L17_EXEC_MXL(l17_execm_xlax, 0, 0, L17_T1SP, 1, L17_MIX_GROUP_X)
L17_EXEC_MXF(l17_execm_xfax, 0, 0, L17_T1SP, 1, L17_MIX_GROUP_X)

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
#define L17_EXEC_MXLD(NAME, CR, T1S, AS, GRP)                                       \
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
            GRP(vin, 34, pb, L17_PBROWM, 0, CR); /* Y(0) -> pb */    \
            for (int x = 0; x < 17; ++x) {                                     \
                double *pba = (x & 1) ? pb2 : pb; /* holds Y(x)'s plane */     \
                double *pbn = (x & 1) ? pb : pb2; /* Y(x+1)'s target    */     \
                if (p->pt && x + 2 < 17) /* r9: plane ahead of Y(x+1) */       \
                    L17_MIX_YPF(vin + (long)(x + 2) * 578, 578);               \
                if (x + 1 < 17)                                                \
                    GRP(vin + (long)(x + 1) * 578, 34, pbn,          \
                                  L17_PBROWM, 0, CR);                          \
                GRP(pba, L17_PBROWM, t1 + (long)x * (T1S), 34, 0,    \
                              CR);                                             \
            }                                                                  \
            if (p->pt)                                                         \
                L17_MIX_XPASS_PT(t1, (T1S), vout, 578, 0, CR);                 \
            else                                                               \
                L17_MIX_XPASS(t1, (T1S), vout, 578, 0, CR);                    \
        }                                                                      \
    }

L17_EXEC_MXLD(l17_execm_xld, 0, 578, 0, L17_MIX_GROUP)
L17_EXEC_MXLD(l17_execm_xldc, 1, 578, 0, L17_MIX_GROUP) /* r7: cosine-resident twin */
L17_EXEC_MXLD(l17_execm_xlda, 0, L17_T1SP, 1, L17_MIX_GROUP)
L17_EXEC_MXLD(l17_execm_xldax, 0, L17_T1SP, 1, L17_MIX_GROUP_X) /* ice_r1: extract-store twin */

/* X-first + deferred-Z: the batch-regime twin, with the pf/pw hooks of
 * l17_execm_xf (prefetches change no bits, so it stays in bit class D). */
#define L17_EXEC_MXFD(NAME, CR, T1S, AS, GRP)                                       \
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
            GRP(t1, 34, pb, L17_PBROWM, 0, CR); /* Y(0) */           \
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
                    GRP(t1 + (long)(x + 1) * (T1S), 34, pbn,         \
                                  L17_PBROWM, 0, CR);                          \
                if (p->pw) {                                                   \
                    const char *po = (const char *)pt;                         \
                    for (int q4 = 37; q4 < 73; ++q4)                           \
                        __builtin_prefetch(po + (long)q4 * 64, 1, 3);          \
                }                                                              \
                GRP(pba, L17_PBROWM, pt, 34, 0, CR);                 \
            }                                                                  \
        }                                                                      \
    }

L17_EXEC_MXFD(l17_execm_xfd, 0, 578, 0, L17_MIX_GROUP)
L17_EXEC_MXFD(l17_execm_xfdc, 1, 578, 0, L17_MIX_GROUP) /* r7: cosine-resident twin */
L17_EXEC_MXFD(l17_execm_xfda, 0, L17_T1SP, 1, L17_MIX_GROUP)
L17_EXEC_MXFD(l17_execm_xfdax, 0, L17_T1SP, 1, L17_MIX_GROUP_X) /* ice_r1: extract-store twin */

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
#ifdef L17_VERBOSE_BUILD
    return 1; /* tryout.sh cannot pass environment through ssh; -DL17_VERBOSE_BUILD can */
#else
    const char *e = getenv("L17_VERBOSE");
    return e && *e && *e != '0';
#endif
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

/* -----------------------------------------------------------------------
 * ROUND ice_r1: chain-shaped tuner unit for the graded regime.
 * The scored workload (cases.txt "17:32:98") is a CHAIN: execute over 32
 * volumes, then a driver-side unitary scale of the whole 2.4 MB output,
 * then the output becomes the next step's input, ping-ponging between two
 * destination buffers.  All three buffers are L3-resident (7.2 MiB total,
 * L2 is 1.25 MB), so each execute reads its input from L3 (it was written
 * by the scale pass one step ago) and RFOs an output that was last written
 * two steps ago.  The old stage-1 arena (16 volumes, fixed src->dst, no
 * scale pass) measures a DIFFERENT regime; candidates are now timed under
 * the driver's own loop shape.  Values stay O(1) under the unitary scale,
 * so no overflow and no denormal drift (corpus 10's trap) in the arena.
 * --------------------------------------------------------------------- */
static void l17_chain_scale(double *restrict v, size_t nd)
{
    const double s = 1.0 / 70.09279563550022; /* 1/sqrt(17^3), as the driver */
    for (size_t i = 0; i < nd; ++i) v[i] *= s;
}

typedef void (*l17_execfn)(const fft3d_plan *, const double _Complex *,
                           double _Complex *);

static double l17_chain_unit(fft3d_plan *p, l17_execfn fn,
                             double _Complex *tb2, int steps)
{
    const size_t nd = (size_t)p->batch * 4913 * 2;
    const double _Complex *src = p->ti;
    double _Complex *dst = p->to;
    double t0 = l17_now();
    for (int s = 0; s < steps; ++s) {
        fn(p, src, dst);
        l17_chain_scale((double *)dst, nd);
        src = dst;
        dst = (dst == p->to) ? tb2 : p->to;
    }
    return l17_now() - t0;
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
    l17_as_build(p->astab);

    /* Clock-settle spin (round ice_r1, adopted from L17_winograd's tuner
     * protocol): ~150 ms of dense 512-bit FMA before anything is timed.
     * This entry's create() is short (~0.23 s); on the Ice Lake node's
     * schedutil governor every clock probe read exactly the 2.90 GHz base
     * while L17_winograd's longer create read 3.50/3.30 -- i.e. the tuner
     * was ranking candidates and the b1dec/clk probes were reporting on a
     * partially unramped core.  The spin costs unscored plan time only. */
    {
        double t0 = l17_now();
        vd_w4 sa = {1, 1, 1, 1, 1, 1, 1, 1}, sb = sa, sc2 = sa, sd = sa;
        vd_w4 sm = {1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15};
        do {
            for (long i = 0; i < 100000; ++i) {
                sa = sa * sm + sa; sb = sb * sm + sb;
                sc2 = sc2 * sm + sc2; sd = sd * sm + sd;
            }
        } while (l17_now() - t0 < 0.15);
        l17_clk_sink = sa[0] + sb[0] + sc2[0] + sd[0];
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
        enum { L17_NCAND = 54 };
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
            l17_execm_xfax, l17_execm_xfdax,
            l17_execm_xlax, l17_execm_xldax,
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
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, X-first, addr-safe t1, extract-store",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, X-first, deferred-Z, addr-safe t1, extract-store",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, addr-safe t1, extract-store",
            "nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, deferred-Z, addr-safe t1, extract-store",
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
        const int selB[11] = {8, 10, 20, 22, 32, 33, 36, 42, 43, 52, 53}; /* X-last pinned */
        const int selD[23] = {9, 11, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                              34, 35, 37, 44, 45, 46, 47, 48, 49, 50, 51};
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
        /* ice_r1 adds 50/51 (X-first extract-store twins) to class D and
         * 52/53 (X-last extract-store twins) to class B; all four store
         * bit-identical values through different store instructions and were
         * cmp-verified against 44 resp. 42 on full outputs before being
         * listed here.
         *
         * CLASS RULE (ice_r1): the X-first class D now starts at batch 17,
         * not 64.  The old boundary was set on CLX DRAM-streaming data; on
         * the Ice Lake node the GRADED chain (cases.txt 17:32:98, all three
         * buffers L3-resident) was measured chain-shaped across the whole
         * table: X-first addr-safe 14.29-14.36 us/step vs the best X-last
         * 15.80 -- the X-last order serialises the 73-chunk output RFO burst
         * at the end of every volume, exactly what X-first spreads across
         * the plane phase.  batch < 17 keeps class B and its whole tuner
         * path untouched. */
        const int *sel = (batch >= 17) ? selD : selB;
        const int nsel = (batch >= 17) ? 23 : 11;
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
        if (batch < 17) {
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

        /* stage 1g (17 <= batch < 64, round ice_r1): the GRADED-CHAIN regime.
         * Candidates are timed under the driver's own chain loop -- nv = batch
         * volumes, output scaled unitarily and fed back as the next input,
         * ping-ponging between two destination arenas -- because the scored
         * number at batch 32 is exactly that loop, and its memory behaviour
         * (L3-resident ping-pong, RFO on a two-steps-old buffer, a scale
         * pass between executes) matches neither the old 16-volume fixed
         * src->dst arena nor the >L3 streaming arena.  6 steps per unit
         * (step 0 from the pristine arena, 5 in steady state), blocked,
         * 1 warmup unit + min of 3.  Selection stays inside the batch
         * regime's bit class; everything is measured for the table. */
        if (batch >= 17 && batch < 64) {
            double _Complex *tb2 = NULL;
            if (l17_tune_alloc(p, batch) &&
                posix_memalign((void **)&tb2, 64,
                               (size_t)batch * 4913 * sizeof *tb2) != 0)
                tb2 = NULL;
            if (tb2) {
                memset(tb2, 0, (size_t)batch * 4913 * sizeof *tb2);
                const int steps = 6;
                double best[L17_NCAND];
                for (int v = 0; v < L17_NCAND; ++v) {
                    best[v] = 1e30;
                    l17_chain_unit(p, cand[v], tb2, steps);
                    for (int r = 0; r < 3; ++r) {
                        double dt = l17_chain_unit(p, cand[v], tb2, steps);
                        if (dt < best[v]) best[v] = dt;
                    }
                }
                for (int s = 1; s < nsel; ++s)
                    if (best[sel[s]] < best[bestv]) bestv = sel[s];
                p->exec = cand[bestv];
                g_desc = tags[bestv];
                if (l17_verbose())
                    for (int v = 0; v < L17_NCAND; ++v)
                        fprintf(stderr, "[L17_matrixsimd tune chain nv=%d] %-72s %8.2f us/transform%s\n",
                                batch, tags[v],
                                best[v] * 1e6 / ((double)batch * steps),
                                v == bestv ? "  <== kept" : "");

                /* joint (pf,pw,pt) grid on the winner, chain-shaped, same
                 * hook conditions as stage 2 (interacting knobs are never
                 * A/B'd sequentially -- the r4 lesson).  Prefetches change
                 * no bits, so the grid is free within the class. */
                {
                    const int pipewin = (bestv >= 24 && bestv <= 27) ||
                                        bestv == 30 || bestv == 31 || bestv == 35;
                    const int stagewin = bestv == 46 || bestv == 47;
                    const int sowin = bestv == 48 || bestv == 49;
                    const int haspf = ((bestv >= 4 && bestv <= 11) ||
                                       (bestv >= 16 && bestv <= 23) ||
                                       bestv == 28 || bestv == 29 ||
                                       bestv == 34 || bestv == 37 ||
                                       bestv == 44 || bestv == 45) &&
                                      !pipewin && !stagewin && !sowin &&
                                      (tags[bestv] && strstr(tags[bestv], "X-first") != NULL);
                    const int haspw = bestv == 9 || bestv == 11 || bestv == 21 ||
                                      bestv == 23 || bestv == 28 || bestv == 29 ||
                                      bestv == 34 || bestv == 35 || bestv == 37 ||
                                      bestv == 40 || bestv == 41 ||
                                      bestv == 44 || bestv == 45 ||
                                      bestv == 46 || bestv == 47 ||
                                      bestv == 48 || bestv == 49 ||
                                      bestv == 50 || bestv == 51;
                    const int haspt = bestv == 32 || bestv == 33 || bestv == 36 ||
                                      bestv == 42 || bestv == 43 ||
                                      bestv == 34 || bestv == 37 ||
                                      bestv == 44 || bestv == 45 ||
                                      bestv == 50 || bestv == 51 ||
                                      bestv == 52 || bestv == 53;
                    if (haspf || haspw || haspt) {
                        double bt = 1e30;
                        int bpf = 0, bpw = 0, bpt = 0;
                        for (int fp = 0; fp <= haspf; ++fp)
                            for (int fw = 0; fw <= haspw; ++fw)
                                for (int ft = 0; ft <= haspt; ++ft) {
                                    p->pf = fp;
                                    p->pw = fw;
                                    p->pt = ft;
                                    double bb = 1e30;
                                    l17_chain_unit(p, p->exec, tb2, steps);
                                    for (int r = 0; r < 3; ++r) {
                                        double dt = l17_chain_unit(p, p->exec, tb2, steps);
                                        if (dt < bb) bb = dt;
                                    }
                                    if (l17_verbose())
                                        fprintf(stderr, "[L17_matrixsimd tune chain] pf=%d pw=%d pt=%d  %.2f us/transform\n",
                                                fp, fw, ft, bb * 1e6 / ((double)batch * steps));
                                    if (bb < bt) { bt = bb; bpf = fp; bpw = fw; bpt = ft; }
                                }
                        p->pf = bpf;
                        p->pw = bpw;
                        p->pt = bpt;
                        static char g_desc_cg[220];
                        snprintf(g_desc_cg, sizeof g_desc_cg, "%s, pf=%d, pw=%d, pt=%d",
                                 g_desc, p->pf, p->pw, p->pt);
                        g_desc = g_desc_cg;
                    } else {
                        p->pf = 0;
                        p->pw = 0;
                        p->pt = 0;
                    }
                }
                free(tb2);
            } else {
                free(tb2);
            }
        }

        /* stage 1c (batch < 17, round panel_r9; ice_r1 narrowed it from
         * batch < 64 -- the graded-chain regime now has its own joint
         * (pf,pw,pt) grid inside stage 1g): A/B the in-pass source
         * prefetch (pt) on the stage-1 winner, blocked, two warmups --
         * prefetches change no bits, so the choice is free within the class.
         * Only the mixed execs carry the hook. */
        if (batch < 17) {
            const int haspt = bestv == 32 || bestv == 33 || bestv == 36 ||
                              bestv == 42 || bestv == 43 ||
                              bestv == 52 || bestv == 53;
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
                              bestv == 48 || bestv == 49 ||
                              bestv == 50 || bestv == 51;
            /* r9: the in-pass source prefetch, offered to the mixed X-first
             * non-pipelined execs.  In the (pf, pw, pt) grid jointly, per
             * the r4 lesson that interacting knobs cannot be raced
             * sequentially. */
            const int haspt = bestv == 34 || bestv == 37 ||
                              bestv == 44 || bestv == 45 ||
                              bestv == 50 || bestv == 51;
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
        /* development override: -DL17_FORCE=0..59 pins one variant;
         * -DL17_FORCE_PF / -DL17_FORCE_PW / -DL17_FORCE_PT pin the
         * prefetch flags */
        {
            static const l17_fn all[60] = {
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
                l17_execm_xfax, l17_execm_xfdax,   /* 56,57: extract-store X-first (ice_r1) */
                l17_execm_xlax, l17_execm_xldax,   /* 58,59: extract-store X-last (ice_r1) */
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
            static char g_desc_sbw[340];
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
            static char g_desc_b1[280];
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
       * in one string would confirm density as the licence discriminator. */
        double c512 = l17_clk512(), c256 = l17_clk256(), c256d = l17_clk256d();
        static char g_desc_clk[400];
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
    plan->exec(plan, in, out);
}

#endif /* L17_TEMPLATE_PASS */
