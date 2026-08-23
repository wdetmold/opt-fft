/* L = 17, complex-double forward 3D DFT, batched, single-threaded.
 *
 * NEW THIS ROUND (ice_r7) -- the chain engine advances to a port of
 * L17_matrixsimd's ice_r6 "chain v6" ("ms6" here): rows padded 17 -> 20
 * complex in a private arena (every strided access one aligned line), slab
 * stride 101 lines (odd, L1 set spread), passes b/a1 IN PLACE along the
 * stride via a non-restrict kernel twin (no t1, no plane buffers, no
 * addr-safe machinery), pass a2 through 4x4-complex tile transposes with
 * the s6 map at group granularity, cross-slab row-16 fringe, a1(s0+1)
 * software-pipelined into a2(s0).  Structure is theirs (originally rival
 * 1760b1bf's); raced at create() behind an m=3 numerical cross-check
 * against the already-gated msr chain.  One deviation: the a2 stack tile
 * array is explicitly 64-B aligned (their vector typedef is aligned(8), so
 * their stack tiles have no line-alignment guarantee).
 *
 * PREVIOUS ROUND (ice_r6) -- three bit-identical fixes to the r5-ported msr
 * engine: (1) THE BIG ONE, a 64 B realignment of the msr arena -- the five
 * 2*NVOL buffers ahead of it left every msr table/buffer 16 B past a cache
 * line, so every coefficient FMA operand, t1/pb store and pb load was a
 * line-SPLIT access (~the whole 15.1-vs-12.7 gap to L17_matrixsimd's
 * identical engine); (2) their ice_r5 sign-fold adopted (odd-lane sign into
 * the K pins, MULI = swap only, -8 p0/p5 uops per chunk); (3) the overlapped
 * tail chunk now stores ONLY its new row (tr=2 lane-3 tiler: 15 shuffles +
 * 5 stores instead of 40 + 20 per group tail).
 *
 * PREVIOUS ROUND (ice_r5) -- the chain engine is REPLACED WHOLESALE by a
 * port of L17_matrixsimd's scored class-R X-first pipeline (chunk17zr merged
 * kernel, addr-safe t1, deferred-Z v2 schedule, s6 pair-shared map,
 * volume-major in-place chaining), raced at create() against the ice_r4 xk
 * incumbent behind a numerical self-check.  See the "msr" section below for
 * the full attribution and the one deliberate deviation (pure-zmm overlapped
 * tail instead of their ymm tail kernel).  fft3d_execute and everything else
 * is unchanged from ice_r4.
 *
 * PREVIOUS ROUND (ice_r1) -- first Ice Lake round; the graded workload is a
 * chain (B=32, m=98, in/out both L3-resident every step), the machine has
 * TWO 512-bit FMA pipes, p5-only 512-bit shuffles, no AVX-512 licence
 * downclock (clk256 = clk512 in-plan), and gcc 11.4:
 *
 * 1. The graded cell B=32 now takes the FULL batched tuner path (threshold
 *    64 -> 32): both classes ranked at nv=32 (the chain's real L3-resident
 *    working set), joint (variant, pf, pfw) grid, decomposition probe --
 *    the smoke round tuned this cell at nv=16, class A only, no grid.
 * 2. Width discipline in stage 1: the w4 candidates (table entries 0 and
 *    NCA-1) must beat the best w8 candidate by >3% to be picked.
 *    Bit-identical within the class, so a near-tie switch is free; the
 *    smoke round's noisy create picked "xl 256" against two 512-bit FMA
 *    pipes and scored 1.47x behind the leader.
 * 3. pf=2 (plane-ahead src prefetch, exec_body X-last): pf=1's
 *    volume-ahead shape is sized for DRAM streaming; in the chain every
 *    src plane is an L3 HIT and only ~60 cycles of latency need hiding.
 *    The grid races pf in {0,1,2} x pfw in {0,1}.
 * 4. "ty" candidates: ymm 4x4 transposes + ymm deint inside the zmm
 *    pipeline -- on ICX every 512-bit two-source shuffle is p5-only and
 *    competes with the second FMA pipe; ymm shuffles dual-issue p1/p5.
 * 5. "xl 256 pin" and the sp-family pin twins ("xl 512t sp pin",
 *    "xl 512t sp dy pin"): gcc 11.4 materialises kernel constants as
 *    full-width .LC memory operands (~200 load uops per block); pinning
 *    was w8-only before, and "xl 512t sp dy" won the first honest nv=32
 *    rank on the node while the sp family had no pin variants.
 *    First honest-window node table (contention-free ranks): plain
 *    variants 18.2-19.0 us/vol within 2%, sp dy best; st/stp +15-25%
 *    (no DRAM RFO to delete in a cache-resident chain); xfs +8-15%;
 *    probe ph/xp = 13.0/5.3 us/vol -- the plane phase (src reads + A
 *    fill) is 70% of the cell and the next structural target.
 *
 * PREVIOUS ROUND (panel_r10) -- one mechanism, aimed squarely at the r9
 * VERDICT's named direction for L=17 ("fund traffic deletion at B=256/2048";
 * every write-SPREADING mechanism -- my sp grid, winograd's q+pfw -- was
 * declined 6/6 by the node's own tuners in r9, so this round deletes traffic
 * instead of rescheduling it):
 *
 * 1. STAGED DENSE OUT FLUSH ("st" / "st dy" candidates, L17R_FORCE 13/14).
 *    The x pass's interleaving store emits 17 concurrent k-row streams of
 *    16-byte-aligned 128-byte pieces strided 4624 B apart.  At batch these
 *    are DRAM-destined partial-line RFOs: a piece at 16-B misalignment
 *    touches 3 cache lines, the boundary lines are only completed by the
 *    NEXT block of the same stream (~300 cycles later), and 17 concurrent
 *    streams exceed the ~12 fill buffers -- so write-combining lines are
 *    evicted half-filled and the same `out` line is fetched and written
 *    back more than once.  That waste is the one traffic term the fused
 *    rivals do not pay (matrixsimd's chunk store and winograd's g8 both
 *    emit finished planes densely), and my r9 probe put the output-side
 *    exposure at xp = 6.77 us/vol against a ~4 us compute share.  The st
 *    variants point the x-pass stores at a 78.6 KB L2-resident staging
 *    volume (vo) and then flush it to `out` as ONE sequential dense stream
 *    of full lines (vo_flush; pfw composes as a paced prefetchw ahead of
 *    the flush).  Costs ~157 KB of extra L2 round trip and ~2.5k movement
 *    uops (~0.3 us); deletes the partial-line refetch waste.  Same kernel
 *    operands, same values to the same final places: BIT-IDENTICAL, class
 *    A, stage-1-ranked and given the joint (variant, pf, pfw) grid's
 *    partner slot that sp held in r9 (sp itself: declined 6/6, demoted).
 *    At B=1 `out` is L2-resident and staging is pure overhead -- the tuner
 *    will (correctly) keep the incumbent there; this is a batch mechanism.
 *
 * 2. PIPELINED FLUSH ("stp" / "stp dy", L17R_FORCE 15/16).  The forced
 *    wallaby A/B of the immediate flush read +23% at B=2048 streaming: an
 *    un-paced flush is a serial burst with no compute behind it, fully
 *    exposed even when it deletes waste.  stp keeps the staged x pass and
 *    paces the flush of volume b-1 across volume b's plane phase, one 4.5 KB
 *    sequential chunk after each plane's y pass (vo ping-ponged).  This is
 *    NOT a rerun of the falsified sp: sp's paced pieces were kernel blocks
 *    with 17-stream scattered partial-line stores; stp's are single dense
 *    sequential streams, the pattern prefetchers and fill buffers handle at
 *    full rate.  Bit-identical (only the out-write order moves across
 *    independent volumes); the stage-1 rank orders all four staged shapes
 *    and the joint grid races the best of them against the incumbent over
 *    (pf, pfw).
 *
 * PREVIOUS ROUND (panel_r9) -- all three aimed at the batched cells, where
 * this entry trails the fused rivals ~15-20% while B=1 sits at the panel-wide
 * structural limit (r8 VERDICT: four mechanism classes falsified at 1.31x
 * floor):
 *
 * 1. "dy" CANDIDATES (ymm deint inside the otherwise-zmm w8 pipeline).
 *    r8's zmm transposes were -3.0% at B=1/B=8 on the node but REGRESSED the
 *    batched cells +1.7%/+1.4%.  The only stage whose behaviour differs
 *    between the regimes is the deinterleave: its source loads from `in` are
 *    the one unaligned zmm stream r8 added, and at batch they hit COLD
 *    lines, where a 64-byte load at the 16-byte alignment classes of an
 *    odd-length complex plane splits a cache line 3/4 of the time (vs ~1/2
 *    for 32-byte ymm loads).  "xl 512t dy" / "xl 512t sp dy" keep every zmm
 *    transpose and swap only the deint tile back to ymm; bit-identical, the
 *    tuner ranks them per cell (expected: dy wins at batch, plain zmm at
 *    B=1/B=8 -- both regimes keep their r8 win).
 *
 * 2. JOINT (variant, pf, pfw) GRID at batch -- ADOPTED FROM L23_rader
 *    panel_r8 (their joint grid found plain-xf + pf=2 + pw=1, a combination
 *    stage-1-then-grid ranking can never select, and it took their B=128
 *    cell).  The sp write-spreading variant now gets its (pf, pfw) shot
 *    alongside the stage-1 winner: sp's r7 node rejection was measured at
 *    (0,0) only, and its mechanism (spread the out burst across compute)
 *    composes with pfw (prefetchw what it is about to store).  Class-A only,
 *    so a measured variant switch cannot change output bits.
 *
 * 3. STREAMING DECOMPOSITION PROBE in fft3d_create(), reported in the
 *    description string as "probe ph/xp/fu" (us/volume on the streaming
 *    arena, (pf,pfw)=(0,0)) -- the pattern the r8 VERDICT says should become
 *    the panel default (L36_pfa's in-plan node probe).  ph = plane phase
 *    alone (cold `in` reads + A fill: input-side exposure), xp = x pass
 *    alone (hot A reads + `out` burst: output-side exposure), fu = full
 *    plain exec.  On the node this attributes my ~4 us/volume batched gap
 *    to the input or output side in one leaderboard line; the r10 move
 *    (respread the burst vs restage the input) reads directly off it.
 *
 * PREVIOUS ROUND (panel_r8):
 *
 * 1. 512-BIT PLANE TRANSPOSES in the w8 pipeline (tz8x8 / dz8x8 blocks) --
 *    the panel_r7 VERDICT's L=17 synthesis executed: the node rejected three
 *    scheduling attacks on the non-FP residue (ov r5, dz r7, matrixsimd's
 *    deferred-Z at the small cells) and rewarded exactly one mechanism, uop
 *    DELETION (L17_winograd's g8, -8..-10% in all four cells).  My residue
 *    is dominated by the serialized per-plane transpose/deinterleave loops,
 *    which were still round-1's 4x4 ymm tiles even inside the 512-bit
 *    pipeline.  Replaced with 8x8 zmm blocks: the classic 24-shuffle
 *    3-stage network (8 loads + 24 shuffles + 8 stores per 64 elements
 *    against 64 uops at 4x4), and an interleaved-complex 8x8 tile at
 *    16 + 48 + 16 = 80 uops per 64 complex against 128.  ~6.5k uops deleted
 *    per volume, roughly half of them load/store slots on a 2-load-port
 *    CLX.  Pure data movement -- identical values to identical places -- so
 *    every class-A candidate stays bit-identical by construction.  Measured
 *    (wallaby, same-window forced 512t A/B): B=1 8.80-8.83 vs 9.06 old
 *    (-2.8%), B=256 no worse.  The w4 path keeps the ymm tiles.
 *
 * PREVIOUS ROUND (panel_r7):
 *
 * 1. DEFERRED-JUNCTION PLANE SCHEDULE ("dz" variants, mixed-width only) --
 *    ADOPTED FROM L17_matrixsimd panel_r6 ("deferred-Z": defer the consuming
 *    group one slot behind its producer, double-buffering the plane buffer;
 *    measured -3.0% at B=1 and -5.7% at B=8 on wallaby for their structure,
 *    the only new L=17 mechanism of that round with a positive same-structure
 *    number).  My plane phase has THREE store->load junctions per plane, all
 *    back-to-back in exec_body:
 *        deint(x) stores T   -> z(x) loads T
 *        z(x)     stores T   -> transpose(x) loads T   (in place)
 *        transpose(x) stores U -> y(x) loads U
 *    Each junction is a group tail whose stores nothing hides, followed by a
 *    dependent load with no independent work in between.  exec_dz_body runs
 *    the SAME groups software-pipelined one plane deep, with T double-
 *    buffered by plane parity (T0/T1; U stays single -- its producer and
 *    consumer sit in the same iteration, one kernel group apart):
 *        iter x:  transpose(x-1)  T[(x-1)&1] -> U
 *                 z(x)            in place on T[x&1]
 *                 y(x-1)          U -> A[x-1]
 *                 deint(x+1)      in[x+1] -> T[(x+1)&1]
 *    Now every junction has at least one full independent group between
 *    producer and consumer (z->transpose has two).  Unlike panel_r5's ov --
 *    which chased the same non-FP cycles by SPLITTING the transpose loops
 *    into halves and slotting them into kernel drains, and lost to its own
 *    plumbing -- dz moves whole groups and adds ZERO instructions.  Same
 *    kernel calls, same operand values, same within-pass order: bit-identical
 *    to every class-A candidate (cmp-verified), so the tuner ranks it freely.
 *    Cost: one extra T pair (+6.5 KB scratch; plane-phase footprint 19.6 KB
 *    of a 32 KB node L1d).  Three candidates: dz, dz+pin, and dz+sp (the dz
 *    plane schedule with panel_r6's cross-volume x-block pipelining, for the
 *    batched cells).
 *
 * PREVIOUS ROUND (panel_r6):
 *
 * 1. SOFTWARE-PIPELINED X-LAST ("sp", mixed-width only) -- volume b's x-pass
 *    output burst (37 kernel blocks writing 78.6 KB of `out`, at batch a
 *    DRAM-destined burst that runs with the FMA stream idle on stores) is
 *    interleaved into volume b+1's plane phase: 2-3 x-pass blocks of the
 *    PREVIOUS volume run after each plane's y pass, using ping-pong A
 *    buffers so A(b) stays live while the plane phase fills A(b+1).  The
 *    per-volume arithmetic and kernel-call operands are unchanged, only the
 *    global order moves across volume boundaries, so the output is
 *    BIT-IDENTICAL to the other class-A candidates and the tuner ranks it
 *    freely.  This is r4's "Next" item 3, the monitor-named remaining lever
 *    for L=17 batched (~1.39x of un-overlapped memory time at B=2048).  The
 *    x-pass block runs behind a noinline+noclone helper so the pipeline adds
 *    ZERO extra inlined kernel copies (I-footprint discipline, r2's 38 KB
 *    kill line).
 *
 * 2. PACED WRITE-INTENT PREFETCH ("pfw") on the x pass's `out` stores --
 *    ADOPTED FROM the panel_r5 node result at L=8 (L8_fusedaxes, fused+pfs+
 *    pfw, B=2048 -31%) and L=36 (L36_pfa, inplace pf=2, B=256 -16.6%): on
 *    Cascade Lake, HIDE the RFO with prefetchw rather than avoid it with NT
 *    stores (NT lost on the node 4 rounds running).  Before x-pass block k
 *    runs, the 17 destination row regions of block k+2 are prefetched with
 *    write intent.  Plan-time flag, A/B'd in the streaming stage jointly
 *    with pf (4 configs, 3% margin, blocked): prefetchw on cache-resident
 *    lines is pure uop tax (L36_pfa: +13% at B=1), so it must stay gated.
 *
 * 3. PER-CANDIDATE LICENCE WARMUP in the tuner: each candidate now runs
 *    itself for >= 1.5 ms (not just 2 execs) before being timed.  The r5
 *    node probes measured clk256 = 3.89 vs clk512 = 2.89 GHz, and Intel's
 *    licence-up dwell is ~670 us -- so a ymm candidate ranked right after a
 *    zmm one (or after the zmm settle spin) was being measured inside the
 *    AVX-512 licence.  At the B=1 stage (nv <= 16, one exec ~ 200-350 us)
 *    the old 2-exec warmup could not cover the dwell; the time-based warmup
 *    does, so a 256-bit candidate finally gets an honest clock.  (Panel_r5
 *    VERDICT section 5: the licence-transition synthesis, and pencilfused's
 *    per-candidate self-warming fix, adapted.)
 *
 * PREVIOUS ROUND (panel_r5):
 *
 * 1. OVERLAPPED-SHUFFLE VARIANTS ("ov", mixed-width only).  The r4 VERDICT
 *    measured the node's sustained AVX2 clock at 3.89 GHz (not 2.30), which
 *    re-prices B=1: 17.74 us is ~69k cycles against a ~36k-cycle FP floor, so
 *    roughly half the runtime is NOT arithmetic.  The prime suspects are the
 *    per-plane transpose loops (deint_transpose17 + 2x transpose17, ~1.1k
 *    port-5-bound uops per plane) which run SERIALIZED between kernel calls:
 *    while they run, the FMA port is idle, and while a zmm kernel block
 *    drains its ~296 port-0 uops (1/cycle on the Gold 5218's single 512-bit
 *    FMA unit), alloc (4/cycle) runs far ahead -- younger independent
 *    shuffle/load uops issue on ports 5/2/3 essentially for free.  The ov
 *    exec bodies reorder the SAME operations so every shuffle burst sits in
 *    a zmm block's shadow:
 *      - z pass runs its ymm tail FIRST, zmm blocks last, so the T->U
 *        transpose that follows lands in a zmm drain;
 *      - T->U is split in half by output column range: cols 0..7 before the
 *        y pass, cols 8..16 in the shadow of the y pass's first zmm block;
 *      - the NEXT plane's input deinterleave (dead T buffer by then) is
 *        split in half and slotted after the y pass's second zmm block and
 *        after its ymm tail.
 *    Pure reordering of independent ops on disjoint regions: bit-identical
 *    to the other class-A variants (verified), so the plan-time tuner ranks
 *    it like any candidate.  Wallaby (TWO 512-bit FMA units -> drains are
 *    half as long, less shadow) will understate the node benefit, exactly
 *    like r4's mixed-width tail bet.
 *
 * 2. DUAL-WIDTH CLOCK PROBE, reported in fft3d_description() as
 *    clk256/clk512 -- the r4 VERDICT's explicit L=17 ask ("measure the
 *    AVX-512 licence clock, then re-derive").  Adopted from L6_unrolled's
 *    panel_r4 probe (serially dependent FMA chain, latency 4 on
 *    SKX/CLX/ICL/SPR, freq = iters*4/time), extended with a 512-bit chain.
 *    Runs after the tuning tournament (core warm), ~20 ms, unscored.
 *
 * 3. THE X-FIRST/X-LAST CLASS CHOICE IS NOW MEASURED AT PLAN TIME when
 *    batch >= 64: both classes are ranked on the streaming arena and
 *    X-first must win by >3% to be chosen (X-last is the incumbent; on
 *    wallaby it wins outright, r4 record).  This finally runs the node A/B
 *    that r4 could only request.  The choice is deterministic per plan, so
 *    rule 4 (same plan -> same answer) holds; across processes a near-tie
 *    could flip the class and with it the output bits (the classes round
 *    differently) -- the 3% margin plus the X-last default keeps that to
 *    genuinely-winning cases.  -DL17R_XF_CUT=<B> still force-selects
 *    class B for batch >= B, bypassing the measurement.
 *
 * 4. L3-SCALED TUNER ARENA (adopted from L17_matrixsimd panel_r4, itself
 *    from L36_mixedradix): the streaming re-rank uses
 *    nv = min(batch, clamp(2.5*L3/157KB, 384, 1024)) volumes instead of a
 *    fixed 384, so a 60 MB-L3 machine (wallaby) actually streams while the
 *    node's behaviour (22 MB -> 384) is unchanged bit-for-bit.
 *
 * TECHNIQUE (round panel_r4)
 * --------------------------
 * Rader structure for the prime 17, in the symmetrised cyclic/negacyclic form
 * ADOPTED FROM the L17_winograd entry (strategies/L17_winograd.md round 1):
 * 296 FP instructions (192 FMA + 104 add/sub, 488 flops) per 17-point
 * transform.  The plane-fused layout, split re/im, lane blocking and store
 * paths are this file's round-1 design.
 *
 *   g = 3 generates the units mod 17;  quotienting by {+-1} indexes j,k by the
 *   order-8 quotient group:  folded[m] = |3^m mod 17| in 1..8, sigma[m] the
 *   sign lost in folding.
 *     u_j = x_j + x_(17-j)  drives a length-8 CYCLIC correlation with the real
 *                           kernel c[r] = cos(2 pi 3^r / 17)
 *     v_j = x_j - x_(17-j)  drives a length-8 NEGACYCLIC correlation with the
 *                           kernel s[r] = sin(2 pi 3^r / 17)
 *   The cyclic half splits once more with sign-only reductions,
 *   x^8-1 = (x^4-1)(x^4+1):  P_m = U_m + U_(m+4), Q_m = U_m - U_(m+4) give a
 *   4x4 cyclic apply (x_0-seeded, so DC is free) plus a 4x4 negacyclic apply.
 *   x^8+1 is irreducible over R, so the negacyclic-8 for v stays dense --
 *   L17_winograd's record kills every split of it with counts; do not retry.
 *
 * NEW THIS ROUND (panel_r4), both aimed at the scoring node:
 *
 * 1. MIXED-WIDTH TAIL BLOCKS ("t" variants).  The Gold 5218 has ONE 512-bit
 *    FMA unit but TWO 256-bit FMA ports, so a ymm kernel block retires its
 *    296 FP ops in ~148 cycles where a zmm block needs ~296.  A 17-lane pass
 *    at pure VW=8 costs 3 zmm blocks = 888 cycles; as 2 zmm + 1 ymm tail it
 *    costs 296+296+148 = 740 -- the same floor as pure VW=4 (5 x 148) but
 *    with VW=8's lower instruction/load/store count.  148 cycles x 35 tail
 *    blocks/volume ~ 5.2k cycles ~ 2.2 us at 2.3 GHz, which is almost exactly
 *    the panel_r3 B=1 gap to L17_matrixsimd (18.49 vs 16.39; their zmm holds
 *    4 complex so their lane tax was always 5x4, not 3x8).  On a 2-FMA-unit
 *    machine (wallaby) the mix is FP-neutral, so only the node can rank it:
 *    it is a plan-time tuner candidate.  The tail lanes recompute a few
 *    overlapping transforms bit-identically (lane-independent arithmetic).
 *
 * 2. X-FIRST PASS ORDER, built and then DISABLED BY DEFAULT after measurement
 *    ("xf" variants; see the L17R_XF_CUT comment below for the wallaby
 *    numbers).  The idea is L17_matrixsimd's panel_r3 reorder (-10.8% at
 *    B=256 on the node for THEIR structure): transform the x axis straight
 *    out of the interleaved input (deinterleave folded into the kernel
 *    loads), then finish one kx-plane at a time so the output writes spread
 *    across the volume's compute.  For THIS pass structure it loses on
 *    wallaby in every batched regime, so the default cut keeps X-last at all
 *    batch sizes; -DL17R_XF_CUT=64 re-enables it for a node A/B.  X-first is
 *    NOT bit-identical to X-last (the axis order changes the rounding order),
 *    so per L17_matrixsimd's repeatability discipline the CLASS is a pure
 *    function of batch and the tuner selects freely only within the class,
 *    where all candidates are bit-identical (verified by cmp).
 *
 * Also from panel_r3's node verdict: the stage-2 prefetch A/B now needs a 3%
 * margin to switch pf on (the node picked pf=1 at B=256 on a near-tie and
 * lost 7.4% in the driver's steady state -- monitor's diagnosis, panel_r3
 * VERDICT section 2).
 *
 * LAYOUT / MEMORY  (round 1; 4 L1 crossings per volume against 12 for
 * separate movement passes, measured 35 us vs 64 us)
 * ---------------
 * X-last (the default at every batch size):
 *   for each x plane (17x17 complex = 4.6 KiB, comfortably L1-resident):
 *       in[x][y][z]  --deinterleave + 17x17 transpose-->  T[z][y]
 *       z pass on T   (axis stride TR, lanes = y)      ->  T[kz][y]
 *       17x17 transpose                                ->  U[y][kz]
 *       y pass on U   (axis stride TR, lanes = kz), storing straight into
 *                     A[x][ky][kz]                     ->  A
 *   x pass on A (axis stride PS, 289 contiguous lanes), interleaving store
 *                                                      ->  out[kx][ky][kz]
 * X-first "xfs" (class B; raced against class A at plan time when batch >= 64,
 * and through the joint (pf,pfw) grid since panel_r11):
 *   x pass on in (axis stride NPL complex, 289 contiguous lanes,
 *                 deinterleaving load -- the COLD reads ride under the
 *                 kernel's FMA drains), split store   ->  A[kx][y][z]
 *   for each kx plane:
 *       [pfw: prefetchw out plane kx, one plane ahead of its flush]
 *       [kx>0: dense 4.6 KB sequential flush of staged plane kx-1 -> out]
 *       A[kx][y][z]  --2 x 17x17 transpose-->  T[z][y]
 *       z pass on T                         ->  T[kz][y]
 *       17x17 transpose                     ->  U[y][kz]
 *       y pass on U, interleaving store (stride 17)  ->  staging plane
 *                                           (L1-hot, double-buffered by kx)
 *   final flush of plane 16
 *   (panel_r11: the y pass used to write out[kx] directly through 17-row
 *   strided 128-B partial-line pieces -- the store shape the r10 VERDICT
 *   called the wrong baseline.  Staging + deferred dense flush is the
 *   fused rivals' finished-plane store retrofitted; values and final
 *   addresses are unchanged, so bit class B is preserved.)
 *
 * ASSUMPTIONS
 * -----------
 *  * L == 17 only; fft3d_supports() rejects everything else.
 *  * `in` and `out` are distinct, as the driver guarantees.  Only unaligned
 *    vector moves are emitted, so alignment affects speed, never correctness.
 *  * fft3d_execute() never writes through `in`.  Pad lanes of the scratch
 *    buffers are zeroed once in fft3d_create(); every pass maps zeros to
 *    zeros, and each width owns a disjoint scratch region, so they stay zero
 *    for the life of the plan and repeated executes are bitwise identical.
 *  * In the kernel every input is loaded before the first output is stored,
 *    so the z pass may run it in place.
 *  * -ffp-contract=fast (gcc's default under -std=gnu11) for FMA formation.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "../fft3d_api.h"

/* BUILD-FLAG NOTE (panel_r8): L45_pfa (panel_r7) found the scored build
 * lacks tryout.sh's -funroll-loops and lost 10% to it; their fix is a
 * file-level "#pragma GCC optimize(unroll-loops)".  Tested here and
 * REJECTED on measurement (wallaby, same-window 3-way, forced 512t pfw=0,
 * B=1): unroll ON via command line 9.29-9.37 us, unroll OFF 9.36-9.49,
 * pragma 9.55-9.62.  The flag itself is worth <1% for this file (the hot
 * tile loops fully unroll under -O3's complete peeling and the 2-trip
 * kernel loops are deliberately kept rolled with asm-opaque bounds), and
 * the pragma form actively costs ~2% -- the optimize attribute perturbs
 * codegen beyond the named flag.  Do not add it back. */

#ifndef L17R_TEMPLATE_PASS
/* ===========================================================================
 * COMMON SECTION: constants, transposes, plan, tuner.
 * =========================================================================== */

#define LN     17
#define NPL    289            /* 17*17  */
#define NVOL   4913           /* 17^3   */

/* batch size at which the X-first pass order (bit class B) is FORCED.  Since
 * panel_r5 this is a dev/monitor override only: at batch >= 64 the plan now
 * MEASURES both classes on the streaming arena and X-first must beat the
 * X-last incumbent by >3% to be selected (see fft3d_create).  The classes
 * round differently, so a selection is a bit-class choice: deterministic per
 * plan (rule 4 holds); across processes a flip changes output bits, which
 * the margin + X-last default confine to genuinely-winning cases.
 *
 * Wallaby background: the r4 strided-y-store X-first lost there by 11-60%;
 * the panel_r11 staged rebuild ("xfs": y pass -> L1-hot staging plane,
 * deferred dense 4.6 KB flush) closes most of that but still loses ~7%
 * L3-resident (nv=256: 11.56 vs 10.76) and ~18% streaming (B=1024 forced:
 * 17.0 vs 14.4 us/t, best config xfs+pf+pfw).  Wallaby CANNOT rank this
 * shape honestly: its prefetchers price the x pass's 17-row-stream read of
 * `in` at 1.28-1.31x a sequential read, while the node measures the same
 * shape at 0.81-0.83x (L17_matrixsimd's sbw s17/rd, panel_r10 -- the
 * largest cross-machine inversion the panel has measured), and the node is
 * also where dense finished-plane stores won -10.8% (their r3 X-first).
 * Only the node's own plan-time race (automatic, 3% margin, and since r11
 * also raced through the (pf,pfw) grid) can settle it; the stage-1 numbers
 * go out in the description string as `xrace xl/xfs=...`. */
#ifndef L17R_XF_CUT
#define L17R_XF_CUT (1 << 30)
#endif

/* per-width strides, duplicated here for the allocator (the template derives
 * the same numbers from VW) */
#define TR4  20
#define PS4  292
#define TR8  24
#define PS8  296
#define ABUF4 (LN * PS4)
#define TBUF4 (LN * TR4)
#define ABUF8 (LN * PS8)
#define TBUF8 (LN * TR8)

/* ---------------------------------------------------------------- constants
 * c[r] = cos(2 pi 3^r/17), s[r] = sin(2 pi 3^r/17), r = 0..7 indexing the
 * order-8 quotient group.  cp/cm are the cyclic-4 / negacyclic-4 kernels of
 * the sign-only split of the cyclic-8 half:
 *   cp[t] = (c[t] + c[t+4])/2      cm[t] = (c[t] - c[t+4])/2
 * Values verbatim from impl/L17_winograd.c (computed there in long double);
 * SN[t] = s[t] for t = 0..7, negacyclic wrap SN[t+8] = -SN[t]. */
#define CP0 ( 5.12370294433828866e-01)
#define CP1 ( 8.60376828522276954e-02)
#define CP2 (-1.21982091231621334e-01)
#define CP3 (-7.26425886054435255e-01)
#define CM0 ( 4.20101934970526891e-01)
#define CM1 ( 3.59700672924310572e-01)
#define CM2 (-8.60991008452280493e-01)
#define CM3 (-1.23791249675178877e-01)
#define SN0 ( 3.61241666187152921e-01)
#define SN1 ( 8.95163291355062340e-01)
#define SN2 (-1.83749517816570340e-01)
#define SN3 (-5.26432162877355836e-01)
#define SN4 (-9.95734176295034468e-01)
#define SN5 ( 9.61825643172819045e-01)
#define SN6 (-6.73695643646557207e-01)
#define SN7 (-7.98017227280239494e-01)

typedef double    v4d __attribute__((vector_size(32), aligned(8)));
typedef long long v4l __attribute__((vector_size(32)));

/* clang spells the two-operand shuffle differently; gcc is the graded
 * compiler but keeping both costs nothing. */
#if defined(__clang__)
#  define SH4(a,b,m0,m1,m2,m3) __builtin_shufflevector((a),(b),m0,m1,m2,m3)
#else
#  define SH4(a,b,m0,m1,m2,m3) __builtin_shuffle((a),(b),(v4l){m0,m1,m2,m3})
#endif

/* ------------------------------------------------------------- transposes */

/* d[j*ds + i] = s[i*ss + j], i,j in [0,17): 4x4 ymm blocks + scalar edges.
 * Split by SOURCE ROW range (= destination column range): part 0 writes
 * exactly d[.][0..7], part 1 writes d[.][8..16].  The ov exec bodies use the
 * halves to slot the shuffle work into kernel-block shadows; part 0 alone is
 * what a lane block over columns 0..7 needs. */
static inline __attribute__((always_inline))
void transpose17_part(const double *s, long ss, double *d, long ds,
                      const int part)
{
    const int i0lo = part ? 8 : 0, i0hi = part ? 16 : 8;
    for (int i0 = i0lo; i0 < i0hi; i0 += 4)
        for (int j0 = 0; j0 < 16; j0 += 4) {
            const double *p = s + (long)i0*ss + j0;
            v4d r0 = *(const v4d *)(const void *)(p);
            v4d r1 = *(const v4d *)(const void *)(p + ss);
            v4d r2 = *(const v4d *)(const void *)(p + 2*ss);
            v4d r3 = *(const v4d *)(const void *)(p + 3*ss);
            v4d h0 = SH4(r0,r1, 0,4,2,6), h1 = SH4(r0,r1, 1,5,3,7);
            v4d h2 = SH4(r2,r3, 0,4,2,6), h3 = SH4(r2,r3, 1,5,3,7);
            double *q = d + (long)j0*ds + i0;
            *(v4d *)(void *)(q)        = SH4(h0,h2, 0,1,4,5);
            *(v4d *)(void *)(q + ds)   = SH4(h1,h3, 0,1,4,5);
            *(v4d *)(void *)(q + 2*ds) = SH4(h0,h2, 2,3,6,7);
            *(v4d *)(void *)(q + 3*ds) = SH4(h1,h3, 2,3,6,7);
        }
    if (!part) {
        for (int i = 0; i < 8; ++i)  d[16*ds + i] = s[(long)i*ss + 16];
    } else {
        for (int i = 8; i < 16; ++i) d[16*ds + i] = s[(long)i*ss + 16];
        for (int j = 0; j < LN; ++j) d[(long)j*ds + 16] = s[16*ss + j];
    }
}

static inline __attribute__((always_inline))
void transpose17(const double *s, long ss, double *d, long ds)
{
    transpose17_part(s, ss, d, ds, 0);
    transpose17_part(s, ss, d, ds, 1);
}

/* dr[j*ds+i] = Re src[i*17+j], di likewise; src is interleaved complex.
 * Per 4x4 tile: transpose the COMPLEX values first, at 128-bit granularity
 * (one shuffle moves two doubles that stay together), and split re/im after.
 * 8 + 8 = 16 shuffles per 16 complex elements instead of the 8 deinterleaves
 * + 16 real transposes (24) of the obvious order (round 1: 5.31 -> 3.34 us).
 * Split by SOURCE ROW range like transpose17_part; the two parts write
 * disjoint destination regions, so any interleaving with other work on
 * distinct buffers is bit-identical to the back-to-back order. */
static inline __attribute__((always_inline))
void deint_transpose17_part(const double *s, double *dr, double *di, long ds,
                            const int part)
{
    const int i0lo = part ? 8 : 0, i0hi = part ? 16 : 8;
    for (int i0 = i0lo; i0 < i0hi; i0 += 4)
        for (int j0 = 0; j0 < 16; j0 += 4) {
            const double *p = s + 2*((long)i0*LN + j0);
            v4d a0 = *(const v4d *)(const void *)(p);              /* row i0+0, cols j0+0,1 */
            v4d b0 = *(const v4d *)(const void *)(p + 4);          /*           cols j0+2,3 */
            v4d a1 = *(const v4d *)(const void *)(p + 2*LN);
            v4d b1 = *(const v4d *)(const void *)(p + 2*LN + 4);
            v4d a2 = *(const v4d *)(const void *)(p + 4*LN);
            v4d b2 = *(const v4d *)(const void *)(p + 4*LN + 4);
            v4d a3 = *(const v4d *)(const void *)(p + 6*LN);
            v4d b3 = *(const v4d *)(const void *)(p + 6*LN + 4);
            /* out row jj holds the complex values of source rows i0..i0+3 */
            v4d l0 = SH4(a0,a1, 0,1,4,5), h0 = SH4(a2,a3, 0,1,4,5);   /* jj=0 */
            v4d l1 = SH4(a0,a1, 2,3,6,7), h1 = SH4(a2,a3, 2,3,6,7);   /* jj=1 */
            v4d l2 = SH4(b0,b1, 0,1,4,5), h2 = SH4(b2,b3, 0,1,4,5);   /* jj=2 */
            v4d l3 = SH4(b0,b1, 2,3,6,7), h3 = SH4(b2,b3, 2,3,6,7);   /* jj=3 */
            double *qr = dr + (long)j0*ds + i0, *qi = di + (long)j0*ds + i0;
            *(v4d *)(void *)(qr)        = SH4(l0,h0, 0,2,4,6);
            *(v4d *)(void *)(qi)        = SH4(l0,h0, 1,3,5,7);
            *(v4d *)(void *)(qr + ds)   = SH4(l1,h1, 0,2,4,6);
            *(v4d *)(void *)(qi + ds)   = SH4(l1,h1, 1,3,5,7);
            *(v4d *)(void *)(qr + 2*ds) = SH4(l2,h2, 0,2,4,6);
            *(v4d *)(void *)(qi + 2*ds) = SH4(l2,h2, 1,3,5,7);
            *(v4d *)(void *)(qr + 3*ds) = SH4(l3,h3, 0,2,4,6);
            *(v4d *)(void *)(qi + 3*ds) = SH4(l3,h3, 1,3,5,7);
        }
    if (!part) {
        for (int i = 0; i < 8; ++i) {
            dr[16*ds + i] = s[2*((long)i*LN + 16)];
            di[16*ds + i] = s[2*((long)i*LN + 16) + 1];
        }
    } else {
        for (int i = 8; i < 16; ++i) {
            dr[16*ds + i] = s[2*((long)i*LN + 16)];
            di[16*ds + i] = s[2*((long)i*LN + 16) + 1];
        }
        for (int j = 0; j < LN; ++j) {
            dr[(long)j*ds + 16] = s[2*(16*LN + j)];
            di[(long)j*ds + 16] = s[2*(16*LN + j) + 1];
        }
    }
}

static inline __attribute__((always_inline))
void deint_transpose17(const double *s, double *dr, double *di, long ds)
{
    deint_transpose17_part(s, dr, di, ds, 0);
    deint_transpose17_part(s, dr, di, ds, 1);
}

/* ---------------- 512-bit transposes (panel_r8, w8 pipeline only) ----------
 * The panel_r7 VERDICT's one-directional L=17 evidence: the node rejected
 * three scheduling attacks on the non-FP residue (ov r5, dz r7, matrixsimd's
 * deferred-Z at the small cells) and rewarded exactly one mechanism -- uop
 * DELETION (L17_winograd's g8, -8..-10% in all four cells).  My residue is
 * dominated by these serialized transpose loops, which were still 4x4 ymm
 * tiles from round 1 even inside the 512-bit pipeline.  An 8x8 zmm block is
 * 8 loads + 24 shuffles + 8 stores = 40 uops per 64 elements against the
 * 4x4 tiling's 64 (and half the load/store slots on a 2-load-port CLX);
 * the interleaved-complex tile drops from 128 to 80 uops per 64 elements.
 * Pure data movement: same values to the same places, so every class-A
 * candidate stays bit-identical by construction.  On a non-AVX512 host gcc
 * emulates the 64-byte vectors (correct, slow) and the w8 candidates
 * self-eliminate in the tuner, as before. */

typedef double    v8d __attribute__((vector_size(64), aligned(8)));
typedef long long v8l __attribute__((vector_size(64)));

#if defined(__clang__)
#  define SH8(a,b,m0,m1,m2,m3,m4,m5,m6,m7) \
        __builtin_shufflevector((a),(b),m0,m1,m2,m3,m4,m5,m6,m7)
#else
#  define SH8(a,b,m0,m1,m2,m3,m4,m5,m6,m7) \
        __builtin_shuffle((a),(b),(v8l){m0,m1,m2,m3,m4,m5,m6,m7})
#endif
#define VL8(p)   (*(const v8d *)(const void *)(p))
#define VS8(p,x) (*(v8d *)(void *)(p) = (x))

/* d[j*ds + i] = s[i*ss + j], i,j in [0,8): the classic 3-stage network
 * (vunpck / 128-bit-granule permute / 256-bit-granule permute), 24 two-source
 * shuffles, all single-uop on SKX/CLX/SPR. */
static inline __attribute__((always_inline))
void tz8x8(const double *s, long ss, double *d, long ds)
{
    v8d r0 = VL8(s),        r1 = VL8(s + ss),   r2 = VL8(s + 2*ss),
        r3 = VL8(s + 3*ss), r4 = VL8(s + 4*ss), r5 = VL8(s + 5*ss),
        r6 = VL8(s + 6*ss), r7 = VL8(s + 7*ss);
    v8d t0 = SH8(r0,r1, 0,8,2,10,4,12,6,14);
    v8d t1 = SH8(r0,r1, 1,9,3,11,5,13,7,15);
    v8d t2 = SH8(r2,r3, 0,8,2,10,4,12,6,14);
    v8d t3 = SH8(r2,r3, 1,9,3,11,5,13,7,15);
    v8d t4 = SH8(r4,r5, 0,8,2,10,4,12,6,14);
    v8d t5 = SH8(r4,r5, 1,9,3,11,5,13,7,15);
    v8d t6 = SH8(r6,r7, 0,8,2,10,4,12,6,14);
    v8d t7 = SH8(r6,r7, 1,9,3,11,5,13,7,15);
    v8d u0 = SH8(t0,t2, 0,1,8,9,4,5,12,13);
    v8d u2 = SH8(t0,t2, 2,3,10,11,6,7,14,15);
    v8d u1 = SH8(t1,t3, 0,1,8,9,4,5,12,13);
    v8d u3 = SH8(t1,t3, 2,3,10,11,6,7,14,15);
    v8d u4 = SH8(t4,t6, 0,1,8,9,4,5,12,13);
    v8d u6 = SH8(t4,t6, 2,3,10,11,6,7,14,15);
    v8d u5 = SH8(t5,t7, 0,1,8,9,4,5,12,13);
    v8d u7 = SH8(t5,t7, 2,3,10,11,6,7,14,15);
    VS8(d,        SH8(u0,u4, 0,1,2,3,8,9,10,11));
    VS8(d + ds,   SH8(u1,u5, 0,1,2,3,8,9,10,11));
    VS8(d + 2*ds, SH8(u2,u6, 0,1,2,3,8,9,10,11));
    VS8(d + 3*ds, SH8(u3,u7, 0,1,2,3,8,9,10,11));
    VS8(d + 4*ds, SH8(u0,u4, 4,5,6,7,12,13,14,15));
    VS8(d + 5*ds, SH8(u1,u5, 4,5,6,7,12,13,14,15));
    VS8(d + 6*ds, SH8(u2,u6, 4,5,6,7,12,13,14,15));
    VS8(d + 7*ds, SH8(u3,u7, 4,5,6,7,12,13,14,15));
}

/* Same contract and part split as transpose17_part (part 0 writes d[.][0..7],
 * part 1 writes d[.][8..16]); only the block size changed. */
static inline __attribute__((always_inline))
void transpose17z_part(const double *s, long ss, double *d, long ds,
                       const int part)
{
    const long i0 = part ? 8 : 0;
    tz8x8(s + i0*ss,     ss, d + i0,        ds);
    tz8x8(s + i0*ss + 8, ss, d + 8*ds + i0, ds);
    if (!part) {
        for (int i = 0; i < 8; ++i)  d[16*ds + i] = s[(long)i*ss + 16];
    } else {
        for (int i = 8; i < 16; ++i) d[16*ds + i] = s[(long)i*ss + 16];
        for (int j = 0; j < LN; ++j) d[(long)j*ds + 16] = s[16*ss + j];
    }
}

static inline __attribute__((always_inline))
void transpose17z(const double *s, long ss, double *d, long ds)
{
    transpose17z_part(s, ss, d, ds, 0);
    transpose17z_part(s, ss, d, ds, 1);
}

/* 8x8-complex deinterleaving transpose tile: s points at interleaved complex
 * (row 0, col 0) of the tile, row stride LN complex; writes the transposed
 * re/im 8x8 blocks at dr/di (row stride ds).  Same scheme as the ymm tile:
 * transpose the COMPLEX values first at 128-bit granularity (two quadrant
 * 4x4 granule transposes per register half), split re/im after.
 * 16 loads + 48 shuffles + 16 stores per 64 complex, against 128 uops for
 * the 4x4 tiling. */
static inline __attribute__((always_inline))
void dz8x8(const double *s, double *dr, double *di, long ds)
{
    v8d a0 = VL8(s),           b0 = VL8(s + 8);
    v8d a1 = VL8(s + 2*LN),    b1 = VL8(s + 2*LN + 8);
    v8d a2 = VL8(s + 4*LN),    b2 = VL8(s + 4*LN + 8);
    v8d a3 = VL8(s + 6*LN),    b3 = VL8(s + 6*LN + 8);
    v8d a4 = VL8(s + 8*LN),    b4 = VL8(s + 8*LN + 8);
    v8d a5 = VL8(s + 10*LN),   b5 = VL8(s + 10*LN + 8);
    v8d a6 = VL8(s + 12*LN),   b6 = VL8(s + 12*LN + 8);
    v8d a7 = VL8(s + 14*LN),   b7 = VL8(s + 14*LN + 8);
    /* 4x4 128-bit-granule transpose of each quadrant: l_j = complex of source
     * rows 0..3 at tile column j, h_j = rows 4..7 at column j. */
#define QT4(m0,m1,m2,m3, n0,n1,n2,n3) do {                                     \
        v8d p0 = SH8(m0,m1, 0,1,8,9,4,5,12,13);                                \
        v8d p1 = SH8(m0,m1, 2,3,10,11,6,7,14,15);                              \
        v8d p2 = SH8(m2,m3, 0,1,8,9,4,5,12,13);                                \
        v8d p3 = SH8(m2,m3, 2,3,10,11,6,7,14,15);                              \
        n0 = SH8(p0,p2, 0,1,2,3,8,9,10,11);                                    \
        n2 = SH8(p0,p2, 4,5,6,7,12,13,14,15);                                  \
        n1 = SH8(p1,p3, 0,1,2,3,8,9,10,11);                                    \
        n3 = SH8(p1,p3, 4,5,6,7,12,13,14,15);                                  \
    } while (0)
    v8d l0,l1,l2,l3,l4,l5,l6,l7, h0,h1,h2,h3,h4,h5,h6,h7;
    QT4(a0,a1,a2,a3, l0,l1,l2,l3);
    QT4(a4,a5,a6,a7, h0,h1,h2,h3);
    QT4(b0,b1,b2,b3, l4,l5,l6,l7);
    QT4(b4,b5,b6,b7, h4,h5,h6,h7);
#undef QT4
#define SPL(j, lj, hj) do {                                                    \
        VS8(dr + (j)*ds, SH8(lj,hj, 0,2,4,6,8,10,12,14));                      \
        VS8(di + (j)*ds, SH8(lj,hj, 1,3,5,7,9,11,13,15));                      \
    } while (0)
    SPL(0, l0, h0);  SPL(1, l1, h1);  SPL(2, l2, h2);  SPL(3, l3, h3);
    SPL(4, l4, h4);  SPL(5, l5, h5);  SPL(6, l6, h6);  SPL(7, l7, h7);
#undef SPL
}

/* Same contract and part split as deint_transpose17_part. */
static inline __attribute__((always_inline))
void deint_transpose17z_part(const double *s, double *dr, double *di, long ds,
                             const int part)
{
    const long i0 = part ? 8 : 0;
    dz8x8(s + 2*(i0*LN),     dr + i0,        di + i0,        ds);
    dz8x8(s + 2*(i0*LN + 8), dr + 8*ds + i0, di + 8*ds + i0, ds);
    if (!part) {
        for (int i = 0; i < 8; ++i) {
            dr[16*ds + i] = s[2*((long)i*LN + 16)];
            di[16*ds + i] = s[2*((long)i*LN + 16) + 1];
        }
    } else {
        for (int i = 8; i < 16; ++i) {
            dr[16*ds + i] = s[2*((long)i*LN + 16)];
            di[16*ds + i] = s[2*((long)i*LN + 16) + 1];
        }
        for (int j = 0; j < LN; ++j) {
            dr[(long)j*ds + 16] = s[2*(16*LN + j)];
            di[(long)j*ds + 16] = s[2*(16*LN + j) + 1];
        }
    }
}

static inline __attribute__((always_inline))
void deint_transpose17z(const double *s, double *dr, double *di, long ds)
{
    deint_transpose17z_part(s, dr, di, ds, 0);
    deint_transpose17z_part(s, dr, di, ds, 1);
}

/* --------------------- ice_r4: the graded chain's map ---------------------
 * The graded step is now  state <- (z + c) / (1 + |z + c|),  z = FFT(state),
 * and the driver times fft3d_chain (weak symbol) when it exists.  The map
 * arithmetic below is ADOPTED VERBATIM from the rival 1.00-scorer's `mapc`
 * (ext/reference/fft_v4_solutions/1000f989_score1.00/implementation.c): the
 * corpus-consensus shape (sec 10.2, discovered independently 4/7) -- burn the
 * divider EXACTLY ONCE per point (the final vdivpd, which also keeps the map
 * exact to ~1 ulp so the 1e-13/step chain budget is met with ~700x margin at
 * m=98) and do the magnitude with vrsqrt14pd + 2 Newton steps on the FMA
 * pipes.  The 1e-300 bias makes s=0 and denormal-range |w|^2 safe without
 * touching MXCSR.  All chain variants share this one map, so the plan-time
 * variant race can never change output bits. */

static inline __attribute__((always_inline))
void l17r_map_sc2(const double *zp, const double *cp, double *mr, double *mi)
{
    double wr = zp[0] + cp[0], wi = zp[1] + cp[1];
    double sc = 1.0 / (1.0 + __builtin_sqrt(wr*wr + wi*wi));
    *mr = wr * sc;
    *mi = wi * sc;
}

#if defined(__AVX512F__)
#include <immintrin.h>

/* map ladder on split re/im vectors of 8 points */
#define L17R_MAP8(wr, wi, orr, oii) do {                                       \
        v8d s_ = ((v8d){0} + 1e-300);                                          \
        s_ += (wr)*(wr);                                                       \
        s_ += (wi)*(wi);                                                       \
        v8d q_ = (v8d)_mm512_rsqrt14_pd((__m512d)s_);                          \
        v8d h_ = ((v8d){0} + 0.5) * s_;                                        \
        q_ = q_ * (((v8d){0} + 1.5) - h_*q_*q_);                               \
        q_ = q_ * (((v8d){0} + 1.5) - h_*q_*q_);                               \
        v8d d_ = ((v8d){0} + 1.0) + s_*q_;    /* 1 + |w|, |w| = s*rsqrt(s) */  \
        v8d r_ = ((v8d){0} + 1.0) / d_;       /* the one exact divide      */  \
        (orr) = (wr)*r_;                                                       \
        (oii) = (wi)*r_;                                                       \
    } while (0)

/* 8 consecutive interleaved complex points: op = (zp+cp)/(1+|zp+cp|) */
static inline __attribute__((always_inline))
void l17r_map8(const double *zp, const double *cp, double *op)
{
    v8d w0 = VL8(zp)     + VL8(cp);
    v8d w1 = VL8(zp + 8) + VL8(cp + 8);
    v8d wr = SH8(w0,w1, 0,2,4,6,8,10,12,14);
    v8d wi = SH8(w0,w1, 1,3,5,7,9,11,13,15);
    v8d mr, mi;
    L17R_MAP8(wr, wi, mr, mi);
    VS8(op,     SH8(mr,mi, 0,8,1,9,2,10,3,11));
    VS8(op + 8, SH8(mr,mi, 4,12,5,13,6,14,7,15));
}
#endif /* __AVX512F__ */

/* contiguous interleaved map over n complex points (n = 289 per plane in the
 * chain step's lazy-map pre-pass, n = 4913 for the final materialization).
 * The tail is one OVERLAPPED 8-group (n >= 8 always here; both call sites are
 * out-of-place, so re-mapping a point from the same raw source is bit-free)
 * -- every point in every chain candidate goes through the one ladder, whose
 * per-lane bits are width-independent, so the whole candidate set stays
 * bit-identical and the plan-time race is free. */
static void l17r_map_run(const double *zp, const double *cp, double *op, long n)
{
#if defined(__AVX512F__)
    long i = 0;
    for (; i + 8 <= n; i += 8)
        l17r_map8(zp + 2*i, cp + 2*i, op + 2*i);
    if (i < n)
        l17r_map8(zp + 2*(n - 8), cp + 2*(n - 8), op + 2*(n - 8));
#else
    for (long i = 0; i < n; ++i)
        l17r_map_sc2(zp + 2*i, cp + 2*i, op + 2*i, op + 2*i + 1);
#endif
}

#if defined(__AVX512VL__)
/* 4-point (ymm) ladder for the x pass's w4 tail block -- vrsqrt14pd is
 * elementwise-identical across widths, so this produces the same bits the
 * 8-wide form would. */
static inline __attribute__((always_inline))
void l17r_map4(const double *zp, const double *cp, double *op)
{
    v4d w0 = *(const v4d *)(const void *)zp
           + *(const v4d *)(const void *)cp;
    v4d w1 = *(const v4d *)(const void *)(zp + 4)
           + *(const v4d *)(const void *)(cp + 4);
    v4d wr = SH4(w0,w1, 0,2,4,6);
    v4d wi = SH4(w0,w1, 1,3,5,7);
    v4d s = ((v4d){0} + 1e-300);
    s += wr*wr;
    s += wi*wi;
    v4d q = (v4d)_mm256_rsqrt14_pd((__m256d)s);
    v4d h = ((v4d){0} + 0.5) * s;
    q = q * (((v4d){0} + 1.5) - h*q*q);
    q = q * (((v4d){0} + 1.5) - h*q*q);
    v4d d = ((v4d){0} + 1.0) + s*q;
    v4d r = ((v4d){0} + 1.0) / d;
    v4d mr = wr*r, mi = wi*r;
    *(v4d *)(void *)op       = SH4(mr,mi, 0,4,1,5);
    *(v4d *)(void *)(op + 4) = SH4(mr,mi, 2,6,3,7);
}

/* EAGER map epilogue for one x-pass block ("xm" chain shape): the block just
 * stored raw z+... raw z rows at dst + 2*(k*NPL + m0); map them IN PLACE
 * (loads hit the store queue / L1) with c streamed from the same offsets.
 * Run per block, the ladder's FMA work and the one divide per row interleave
 * with the adjacent kernel blocks' 296-FP drains -- the corpus-consensus
 * placement (sec 10.2: pipeline the divider under FFT work), one level
 * coarser than the rivals' in-kernel fusion but with zero kernel surgery. */
static inline __attribute__((always_inline))
void l17r_mapblk(double *dst, const double *cm, long m0, const int tail)
{
    for (int k = 0; k < LN; ++k) {
        double *q0 = dst + 2*((long)k*NPL + m0);
        const double *c0 = cm + 2*((long)k*NPL + m0);
        if (tail) l17r_map4(q0, c0, q0);
        else      l17r_map8(q0, c0, q0);
    }
}
#endif /* __AVX512VL__ */

#if defined(__AVX512F__)
#if defined(__AVX512VL__)
/* w4-width eager epilogue: every x-pass block is 4 lanes wide */
static inline __attribute__((always_inline))
void l17r_mapblk4v(double *dst, const double *cm, long m0)
{
    for (int k = 0; k < LN; ++k) {
        double *q0 = dst + 2*((long)k*NPL + m0);
        const double *c0 = cm + 2*((long)k*NPL + m0);
        l17r_map4(q0, c0, q0);
    }
}
#endif

/* Map-FUSED deinterleaving transpose tile (ice_r4): dz8x8 with w = z + c
 * formed in interleaved space at the loads and the map ladder applied on the
 * split re/im rows right before the store.  Against the mp pre-pass shape
 * this deletes, per 8x8-complex tile: 16 buffer stores + 16 reloads + 16
 * reinterleave shuffles + 16 re-deint shuffles, and the mp store->load
 * junction.  Same map values, same destinations: bit-identical to mp+deint. */
static inline __attribute__((always_inline))
void dz8x8m(const double *s, const double *cs, double *dr, double *di, long ds)
{
    v8d a0 = VL8(s)         + VL8(cs);
    v8d b0 = VL8(s + 8)     + VL8(cs + 8);
    v8d a1 = VL8(s + 2*LN)  + VL8(cs + 2*LN);
    v8d b1 = VL8(s + 2*LN + 8)  + VL8(cs + 2*LN + 8);
    v8d a2 = VL8(s + 4*LN)  + VL8(cs + 4*LN);
    v8d b2 = VL8(s + 4*LN + 8)  + VL8(cs + 4*LN + 8);
    v8d a3 = VL8(s + 6*LN)  + VL8(cs + 6*LN);
    v8d b3 = VL8(s + 6*LN + 8)  + VL8(cs + 6*LN + 8);
    v8d a4 = VL8(s + 8*LN)  + VL8(cs + 8*LN);
    v8d b4 = VL8(s + 8*LN + 8)  + VL8(cs + 8*LN + 8);
    v8d a5 = VL8(s + 10*LN) + VL8(cs + 10*LN);
    v8d b5 = VL8(s + 10*LN + 8) + VL8(cs + 10*LN + 8);
    v8d a6 = VL8(s + 12*LN) + VL8(cs + 12*LN);
    v8d b6 = VL8(s + 12*LN + 8) + VL8(cs + 12*LN + 8);
    v8d a7 = VL8(s + 14*LN) + VL8(cs + 14*LN);
    v8d b7 = VL8(s + 14*LN + 8) + VL8(cs + 14*LN + 8);
#define QT4(m0,m1,m2,m3, n0,n1,n2,n3) do {                                     \
        v8d p0 = SH8(m0,m1, 0,1,8,9,4,5,12,13);                                \
        v8d p1 = SH8(m0,m1, 2,3,10,11,6,7,14,15);                              \
        v8d p2 = SH8(m2,m3, 0,1,8,9,4,5,12,13);                                \
        v8d p3 = SH8(m2,m3, 2,3,10,11,6,7,14,15);                              \
        n0 = SH8(p0,p2, 0,1,2,3,8,9,10,11);                                    \
        n2 = SH8(p0,p2, 4,5,6,7,12,13,14,15);                                  \
        n1 = SH8(p1,p3, 0,1,2,3,8,9,10,11);                                    \
        n3 = SH8(p1,p3, 4,5,6,7,12,13,14,15);                                  \
    } while (0)
    v8d l0,l1,l2,l3,l4,l5,l6,l7, h0,h1,h2,h3,h4,h5,h6,h7;
    QT4(a0,a1,a2,a3, l0,l1,l2,l3);
    QT4(a4,a5,a6,a7, h0,h1,h2,h3);
    QT4(b0,b1,b2,b3, l4,l5,l6,l7);
    QT4(b4,b5,b6,b7, h4,h5,h6,h7);
#undef QT4
#define SPLM(j, lj, hj) do {                                                   \
        v8d wr_ = SH8(lj,hj, 0,2,4,6,8,10,12,14);                              \
        v8d wi_ = SH8(lj,hj, 1,3,5,7,9,11,13,15);                              \
        v8d mr_, mi_;                                                          \
        L17R_MAP8(wr_, wi_, mr_, mi_);                                         \
        VS8(dr + (j)*ds, mr_);                                                 \
        VS8(di + (j)*ds, mi_);                                                 \
    } while (0)
    SPLM(0, l0, h0);  SPLM(1, l1, h1);  SPLM(2, l2, h2);  SPLM(3, l3, h3);
    SPLM(4, l4, h4);  SPLM(5, l5, h5);  SPLM(6, l6, h6);  SPLM(7, l7, h7);
#undef SPLM
}

/* whole-plane map-fused deint+transpose: same contract as deint_transpose17z
 * but reading raw state + c and writing MAPPED split re/im.  Tile offsets
 * mirror deint_transpose17z_part's exactly; every point is mapped once. */
static inline __attribute__((always_inline))
void deint_transpose17zm(const double *s, const double *cs,
                         double *dr, double *di, long ds)
{
    dz8x8m(s,                cs,                dr,            di,            ds);
    dz8x8m(s + 16,           cs + 16,           dr + 8*ds,     di + 8*ds,     ds);
    dz8x8m(s + 2*(8*LN),     cs + 2*(8*LN),     dr + 8,        di + 8,        ds);
    dz8x8m(s + 2*(8*LN + 8), cs + 2*(8*LN + 8), dr + 8*ds + 8, di + 8*ds + 8, ds);
    for (int i = 0; i < 16; ++i)         /* source col 16 -> dest row 16 */
        l17r_map_sc2(s + 2*((long)i*LN + 16), cs + 2*((long)i*LN + 16),
                     dr + 16*ds + i, di + 16*ds + i);
    for (int j = 0; j < LN; ++j)         /* source row 16 -> dest col 16 */
        l17r_map_sc2(s + 2*(16*LN + j), cs + 2*(16*LN + j),
                     dr + (long)j*ds + 16, di + (long)j*ds + 16);
}
#endif /* __AVX512F__ */

/* ------------------------------------------------------------------- plan */

/* fused-chain step: (plan, src_volume, c_volume, dst_raw_volume, domap) */
typedef void (*l17r_chfn)(struct fft3d_plan *, const double *, const double *,
                          double *, int);

struct fft3d_plan {
    int batch;
    int pf;                        /* cross-volume input prefetch enabled */
    int pfw;                       /* paced prefetchw on the x pass's out */
    double *mem;
    /* disjoint scratch per width, so each width's pad lanes stay zero */
    double *ar_w4, *ai_w4, *tr_w4, *ti_w4, *ur_w4, *ui_w4;
    double *ar_w8, *ai_w8, *tr_w8, *ti_w8, *ur_w8, *ui_w8;
    double *ar2_w8, *ai2_w8;       /* ping-pong A pair for the sp pipeline */
    double *tr2_w8, *ti2_w8;       /* second T pair for the dz plane schedule */
    double *vo_w8, *vo2_w8;        /* staging volumes for the st/stp flush
                                    * (ping-pong pair; stp flushes b-1 while
                                    * b computes) */
    double *sc0, *sc1;             /* ice_r4: chain raw-state ping-pong (one
                                    * volume each; the buffer stays RAW z
                                    * between steps -- lazy map) */
    double *mv;                    /* ice_r4: mapped volume, fallback path */
    double *mp;                    /* ice_r4: mapped plane (L1-hot, 640 dbl) */
    void (*exec)(struct fft3d_plan *, const double _Complex *,
                 double _Complex *);
    l17r_chfn chexec;              /* ice_r4: fused-chain step variant */
    int cheager;                   /* ice_r4: chexec maps at the x-pass store */
    /* ice_r5: ported L17_matrixsimd class-R chain engine ("msr") */
    double *msc;                   /* nested cosine table, 4x8 splatted 8x   */
    double *mss;                   /* 8 sine constants splatted 8x (K0..K7)  */
    double *mspb, *mspb2;          /* ping-pong plane buffers, rows 20 cplx  */
    double *mst1;                  /* padded t1: 17 planes x 640 dbl + shift */
    int ch_ms;                     /* 1 = the msr chain won the create race  */
    /* ice_r7: ported L17_matrixsimd chain v6 ("ms6") -- padded in-place
     * strided engine (their ice_r6, structure from rival 1760b1bf) */
    double *msv6a;                 /* padded state arena: 17 slabs x 808 dbl */
    double *msv6c;                 /* the volume's c field, same layout      */
    int ch_v6;                     /* 1 = the ms6 chain won the create race  */
    unsigned char msas[256];       /* per-volume t1 base shift (addr-safe)   */
    double _Complex *tin, *tout;   /* transient tuner buffers */
    size_t tn;
};

/* ---- instantiate the pipeline at both vector widths by self-#include ----
 * (mechanism adopted from L17_matrixsimd; a quoted #include is searched in
 * the includer's own directory first, so the first form works from any
 * working directory). */
#if defined(__has_include)
#  if __has_include("L17_rader.c")
#    define L17R_SELF "L17_rader.c"
#  elif __has_include("impl/L17_rader.c")
#    define L17R_SELF "impl/L17_rader.c"
#  endif
#else
#  define L17R_SELF "L17_rader.c"
#endif
#ifndef L17R_SELF
#  error "L17_rader.c must be able to #include itself"
#endif

#define L17R_TEMPLATE_PASS 1
#define VW 4
#define SFX(x) x##_w4
#include L17R_SELF
#undef SFX
#undef VW
#undef L17R_TEMPLATE_PASS

#define L17R_TEMPLATE_PASS 2
#define VW 8
#define SFX(x) x##_w8
#include L17R_SELF
#undef SFX
#undef VW
#undef L17R_TEMPLATE_PASS

/* ===========================================================================
 * ice_r5: PORTED CHAIN ENGINE ("msr") -- L17_matrixsimd's scored class-R
 * X-first pipeline, ADOPTED WHOLESALE from impl/L17_matrixsimd.c (their
 * ice_r2/r3/r4 rounds; see strategies/L17_matrixsimd.md).  Their scored chain
 * runs 13.0 us/step against my 17.5, and the whole gap is the FFT engine
 * (their structure-only skeleton prices at 11.33 us/step vs my nm=16.37):
 * lanes-as-lines interleaved-complex chunks with real coefficients need no
 * deinterleave and only two in-register plane transposes per volume, where my
 * plane pipeline pays deint+transpose+transpose junctions at L2 latency in
 * 70% of the cell (five rounds of probes).  Ported pieces, all verbatim in
 * arithmetic:
 *   - chunk17zr (their ice_r3 merged-reordered kernel, 148 FP ops/chunk,
 *     17 row loads, ONE parked cosine u, pinned sine constants K0..K7);
 *   - the X-first pass order with the addr-safe shifted t1 (their panel_r8
 *     collision model, mode 1) and padded 5120 B plane stride;
 *   - chain schedule v2 (deferred-Z: Y(x+1) between Y(x) and Z(x) on
 *     ping-pong plane buffers) with the per-plane s6 map after each Z;
 *   - the s6 map itself (pair-shared |z+c|^2 packing, rsqrt14 + 2 Newton,
 *     ONE vdivpd per 8 points, exact tier: their measured chain-98 drift
 *     2.1e-14 vs the 9.8e-12 budget);
 *   - volume-major IN-PLACE chaining (final_out is the state arena; legal
 *     because the X pass drains the volume into t1 before any Z store).
 * ONE deliberate deviation: their ymm tail chunk (chunk17n_w2, the old
 * phase-serial kernel + its table set) is replaced by a 5th OVERLAPPED zmm
 * chunk17zr at offset 13 (their own pre-panel_r5 off17 shape {0,4,8,12,13}).
 * On 2x512-bit-pipe ICX the FP time is identical and the overlapped chunk
 * deletes the tail's extra 16 phase-serial row loads; the recomputed lanes
 * are bit-identical (lane-independent arithmetic), so the engine stays one
 * clean bit class.  The engine is raced at create() against the ice_r4 xk
 * incumbent AFTER a numerical self-check against the tuned exec (a
 * transcription bug must fall back, never ship: a fast wrong chain scores
 * nothing).  The msr chain is a different rounding class than xk -- legal:
 * the harness compares the chain end state to numpy only, never across
 * processes (L17_winograd ice_r4 precedent, mx8 vs mxc8).
 * =========================================================================== */
#if defined(__AVX512VL__)

enum { MS_T1SP = 640, MS_PBROW = 40 };

#define MS_SIGN64 ((long long)0x8000000000000000LL)
/* -i * t on interleaved complex.  ice_r6, ADOPTED from L17_matrixsimd ice_r5
 * (their sign-fold, -1.3% matched): every use of MS_MULI's result is an FMA
 * against a sine pin (S +-= K * w), and IEEE multiply carries sign as an XOR
 * ((-k)*x is bit-identical to k*(-x), FMAs included), so the odd-lane sign
 * flip is FOLDED INTO THE K TABLES ((+k,-k,+k,-k,...) splats -- see the mss
 * fill in create()) and MULI is just the swap.  Deletes 8 vpxor-class uops
 * per chunk from ports 0/5 -- exactly the two FMA pipes the kernel is bound
 * on -- across ~243 chunks/step.  -DL17R_NO_SFOLD restores the XOR form +
 * unsigned pins for matched A/Bs; both forms are one bit class. */
#ifdef L17R_NO_SFOLD
#define MS_MULI(t) ((v8d)((v8l)SH8((t),(t),1,0,3,2,5,4,7,6) ^                  \
                          (v8l){0, MS_SIGN64, 0, MS_SIGN64,                    \
                                0, MS_SIGN64, 0, MS_SIGN64}))
#else
#define MS_MULI(t) SH8((t),(t),1,0,3,2,5,4,7,6)
#endif
#define MS_CGET(base, i) VL8((base) + (size_t)(i) * 8)

/* One chunk: length-17 DFT of 4 interleaved-complex lines at once (their
 * chunk17zr, WC=4, verbatim).  src,rs: element j=0 lane 0, rs doubles between
 * successive j.  tr=0: output m at dst + m*db (lanes contiguous).  tr=1:
 * transposing store, lane f's outputs land at dst + f*da + m*2. */
static inline __attribute__((always_inline)) void
ms_chunk17zr(const double *restrict src, long rs, double *restrict dst,
             long da, long db, const double *restrict cn,
             v8d K0, v8d K1, v8d K2, v8d K3, v8d K4, v8d K5, v8d K6, v8d K7,
             const int tr)
{
    v8d A0, A1, A2, A3, X0a, B0, B1, B2, B3;
    v8d S0, S1, S2, S3, S4, S5, S6, S7;
    v8d U;
    {
        v8d x0 = VL8(src);
        A0 = x0; A1 = x0; A2 = x0; A3 = x0; X0a = x0;
    }
    { /* m=0, rows (1,16): u parked for pair 0, w feeds sine row 0 */
        v8d a = VL8(src + 1 * rs), b = VL8(src + 16 * rs);
        U = a + b;
        v8d w = MS_MULI(a - b);
        S0 = K0 * w; S1 = K1 * w; S2 = K2 * w; S3 = K3 * w;
        S4 = K4 * w; S5 = K5 * w; S6 = K6 * w; S7 = K7 * w;
    }
    { /* m=4, rows (13,4): closes cosine pair 0 (U dies, then re-parks) */
        v8d a = VL8(src + 13 * rs), b = VL8(src + 4 * rs);
        v8d u = a + b;
        v8d w = MS_MULI(a - b);
        S0 += K4 * w; S1 += K5 * w; S2 += K6 * w; S3 += K7 * w;
        S4 -= K0 * w; S5 -= K1 * w; S6 -= K2 * w; S7 -= K3 * w;
        v8d pp = U + u, q = U - u;
        X0a += pp;
        A0 += MS_CGET(cn, 0) * pp; A1 += MS_CGET(cn, 1) * pp;
        A2 += MS_CGET(cn, 2) * pp; A3 += MS_CGET(cn, 3) * pp;
        B0 = MS_CGET(cn, 4) * q;  B1 = MS_CGET(cn, 5) * q;
        B2 = MS_CGET(cn, 6) * q;  B3 = MS_CGET(cn, 7) * q;
    }
    { /* m=1, rows (3,14): u parked for pair 1 */
        v8d a = VL8(src + 3 * rs), b = VL8(src + 14 * rs);
        U = a + b;
        v8d w = MS_MULI(a - b);
        S0 += K1 * w; S1 += K2 * w; S2 += K3 * w; S3 += K4 * w;
        S4 += K5 * w; S5 += K6 * w; S6 += K7 * w; S7 -= K0 * w;
    }
    { /* m=5, rows (5,12): closes cosine pair 1 */
        v8d a = VL8(src + 5 * rs), b = VL8(src + 12 * rs);
        v8d u = a + b;
        v8d w = MS_MULI(a - b);
        S0 += K5 * w; S1 += K6 * w; S2 += K7 * w; S3 -= K0 * w;
        S4 -= K1 * w; S5 -= K2 * w; S6 -= K3 * w; S7 -= K4 * w;
        v8d pp = U + u, q = U - u;
        const double *r = cn + (size_t)8 * 8;
        X0a += pp;
        A0 += MS_CGET(r, 0) * pp; A1 += MS_CGET(r, 1) * pp;
        A2 += MS_CGET(r, 2) * pp; A3 += MS_CGET(r, 3) * pp;
        B0 += MS_CGET(r, 4) * q;  B1 += MS_CGET(r, 5) * q;
        B2 += MS_CGET(r, 6) * q;  B3 += MS_CGET(r, 7) * q;
    }
    { /* m=2, rows (9,8): u parked for pair 2 */
        v8d a = VL8(src + 9 * rs), b = VL8(src + 8 * rs);
        U = a + b;
        v8d w = MS_MULI(a - b);
        S0 += K2 * w; S1 += K3 * w; S2 += K4 * w; S3 += K5 * w;
        S4 += K6 * w; S5 += K7 * w; S6 -= K0 * w; S7 -= K1 * w;
    }
    { /* m=6, rows (15,2): closes cosine pair 2 */
        v8d a = VL8(src + 15 * rs), b = VL8(src + 2 * rs);
        v8d u = a + b;
        v8d w = MS_MULI(a - b);
        S0 += K6 * w; S1 += K7 * w; S2 -= K0 * w; S3 -= K1 * w;
        S4 -= K2 * w; S5 -= K3 * w; S6 -= K4 * w; S7 -= K5 * w;
        v8d pp = U + u, q = U - u;
        const double *r = cn + (size_t)16 * 8;
        X0a += pp;
        A0 += MS_CGET(r, 0) * pp; A1 += MS_CGET(r, 1) * pp;
        A2 += MS_CGET(r, 2) * pp; A3 += MS_CGET(r, 3) * pp;
        B0 += MS_CGET(r, 4) * q;  B1 += MS_CGET(r, 5) * q;
        B2 += MS_CGET(r, 6) * q;  B3 += MS_CGET(r, 7) * q;
    }
    { /* m=3, rows (10,7): u parked for pair 3 */
        v8d a = VL8(src + 10 * rs), b = VL8(src + 7 * rs);
        U = a + b;
        v8d w = MS_MULI(a - b);
        S0 += K3 * w; S1 += K4 * w; S2 += K5 * w; S3 += K6 * w;
        S4 += K7 * w; S5 -= K0 * w; S6 -= K1 * w; S7 -= K2 * w;
    }
    { /* m=7, rows (11,6): closes cosine pair 3 */
        v8d a = VL8(src + 11 * rs), b = VL8(src + 6 * rs);
        v8d u = a + b;
        v8d w = MS_MULI(a - b);
        S0 += K7 * w; S1 -= K0 * w; S2 -= K1 * w; S3 -= K2 * w;
        S4 -= K3 * w; S5 -= K4 * w; S6 -= K5 * w; S7 -= K6 * w;
        v8d pp = U + u, q = U - u;
        const double *r = cn + (size_t)24 * 8;
        X0a += pp;
        A0 += MS_CGET(r, 0) * pp; A1 += MS_CGET(r, 1) * pp;
        A2 += MS_CGET(r, 2) * pp; A3 += MS_CGET(r, 3) * pp;
        B0 += MS_CGET(r, 4) * q;  B1 += MS_CGET(r, 5) * q;
        B2 += MS_CGET(r, 6) * q;  B3 += MS_CGET(r, 7) * q;
    }
    v8d C0 = A0 + B0, C1 = A1 + B1, C2 = A2 + B2, C3 = A3 + B3;
    v8d C4 = A0 - B0, C5 = A1 - B1, C6 = A2 - B2, C7 = A3 - B3;
    v8d X0v = X0a;

/* output slot table, identical to their chunk17z/zr */
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
        VS8(dst + 0 * db, E0);   VS8(dst + 1 * db, E1);
        VS8(dst + 2 * db, E2);   VS8(dst + 3 * db, E3);
        VS8(dst + 4 * db, E4);   VS8(dst + 5 * db, E5);
        VS8(dst + 6 * db, E6);   VS8(dst + 7 * db, E7);
        VS8(dst + 8 * db, E8);   VS8(dst + 9 * db, E9);
        VS8(dst + 10 * db, E10); VS8(dst + 11 * db, E11);
        VS8(dst + 12 * db, E12); VS8(dst + 13 * db, E13);
        VS8(dst + 14 * db, E14); VS8(dst + 15 * db, E15);
        VS8(dst + 16 * db, E16);
    } else if (tr == 1) {
        /* 4x4-complex tile transpose at 128-bit granularity (their TTILE),
         * lane f's 17 outputs land as rows of dst.  The m0=13 tile rewrites
         * columns 13..15 of the m0=12 tile with bit-identical values. */
#define MS_TILER(m0, e0, e1, e2, e3)                                           \
        do {                                                                   \
            v8d q0_ = SH8((e0), (e1), 0, 1, 2, 3, 8, 9, 10, 11);               \
            v8d q1_ = SH8((e2), (e3), 0, 1, 2, 3, 8, 9, 10, 11);               \
            v8d q2_ = SH8((e0), (e1), 4, 5, 6, 7, 12, 13, 14, 15);             \
            v8d q3_ = SH8((e2), (e3), 4, 5, 6, 7, 12, 13, 14, 15);             \
            VS8(dst + 0 * da + (m0) * 2, SH8(q0_, q1_, 0, 1, 4, 5, 8, 9, 12, 13)); \
            VS8(dst + 1 * da + (m0) * 2, SH8(q0_, q1_, 2, 3, 6, 7, 10, 11, 14, 15)); \
            VS8(dst + 2 * da + (m0) * 2, SH8(q2_, q3_, 0, 1, 4, 5, 8, 9, 12, 13)); \
            VS8(dst + 3 * da + (m0) * 2, SH8(q2_, q3_, 2, 3, 6, 7, 10, 11, 14, 15)); \
        } while (0)
        MS_TILER(0, E0, E1, E2, E3);
        MS_TILER(4, E4, E5, E6, E7);
        MS_TILER(8, E8, E9, E10, E11);
        MS_TILER(12, E12, E13, E14, E15);
        MS_TILER(13, E13, E14, E15, E16);
#undef MS_TILER
    } else {
        /* tr == 2 (ice_r6): TAIL-ONLY transposing store.  In the overlapped
         * 5th group chunk (base lane 13) lanes 0..2 recompute rows 13..15,
         * which the f0=12 chunk already stored bit-identically; only lane 3
         * (row 16) is new.  Store just that row: 3 shuffles + 1 store per
         * 4-column block instead of the full tile's 8 + 4 -- deletes 25
         * port-5 zmm shuffles and 15 stores per Y/Z group (x34 groups/step)
         * relative to tr=1.  Stored bits identical (same E lane-3 values;
         * the overlapping m0=12/13 columns carry the same registers). */
#define MS_TILE3(m0, e0, e1, e2, e3)                                           \
        do {                                                                   \
            v8d t0_ = SH8((e0), (e1), 6, 7, 14, 15, 6, 7, 14, 15);             \
            v8d t1_ = SH8((e2), (e3), 6, 7, 14, 15, 6, 7, 14, 15);             \
            VS8(dst + 3 * da + (m0) * 2,                                       \
                SH8(t0_, t1_, 0, 1, 2, 3, 8, 9, 10, 11));                      \
        } while (0)
        MS_TILE3(0, E0, E1, E2, E3);
        MS_TILE3(4, E4, E5, E6, E7);
        MS_TILE3(8, E8, E9, E10, E11);
        MS_TILE3(12, E12, E13, E14, E15);
        MS_TILE3(13, E13, E14, E15, E16);
#undef MS_TILE3
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

/* One 17-long free index, transposing store: 4 zmm chunks at 0,4,8,12 plus
 * the overlapped 5th at 13 (lanes 13..15 recomputed bit-identically; ice_r6
 * stores only its new row 16 -- tr=2 -- instead of the full 4-row tile).
 * The asm-opaque bound keeps the 4-chunk loop rolled (I-footprint
 * discipline).  -DL17R_MS_TAILFULL=1 restores the r5 full-tile tail store
 * for matched A/Bs; both store the same bytes to the needed rows. */
#ifndef L17R_MS_TAILFULL
#  define L17R_MS_TAILFULL 0
#endif
#define MS_GROUP(SRCB, RS, DSTB, DA)                                           \
    do {                                                                       \
        int nt_ = 4;                                                           \
        __asm__("" : "+r"(nt_));                                               \
        for (int t_ = 0; t_ < nt_; ++t_) {                                     \
            long f0_ = 4L * t_;                                                \
            ms_chunk17zr((SRCB) + 2 * f0_, (RS), (DSTB) + f0_ * (DA), (DA), 2, \
                         cn, K0, K1, K2, K3, K4, K5, K6, K7, 1);               \
        }                                                                      \
        ms_chunk17zr((SRCB) + 2 * 13, (RS), (DSTB) + 13 * (DA), (DA), 2,       \
                     cn, K0, K1, K2, K3, K4, K5, K6, K7,                       \
                     L17R_MS_TAILFULL ? 1 : 2);                                \
    } while (0)

/* X pass over 289 lanes: 72 zmm chunks + the overlapped tail at 285. */
#define MS_XPASS(SRCB, SRS, DSTB, DRS)                                         \
    do {                                                                       \
        int nx_ = 72;                                                          \
        __asm__("" : "+r"(nx_));                                               \
        for (int i_ = 0; i_ < nx_; ++i_) {                                     \
            long f0_ = 4L * i_;                                                \
            ms_chunk17zr((SRCB) + 2 * f0_, (SRS), (DSTB) + 2 * f0_, 2, (DRS),  \
                         cn, K0, K1, K2, K3, K4, K5, K6, K7, 0);               \
        }                                                                      \
        ms_chunk17zr((SRCB) + 2 * 285, (SRS), (DSTB) + 2 * 285, 2, (DRS),      \
                     cn, K0, K1, K2, K3, K4, K5, K6, K7, 0);                   \
    } while (0)

/* Their s6 map, verbatim: pair-shared |z+c|^2 packing, rsqrt14 + 2 Newton
 * (sqrt exact to ~1 ulp), ONE correctly-rounded vdivpd per 8 points, in
 * place on the interleaved state.  noinline: ONE codegen for every call
 * site, so no schedule can change the map's bits.  i0..i1 are PAIR indices
 * (one pair = 16 doubles); a 289-complex plane is 36 pairs + the tail. */
static __attribute__((noinline)) void
ms_map_vecs(double *restrict s, const double *restrict cp, int i0, int i1)
{
    for (int i = i0; i < i1; ++i) {
        double *p = s + 16 * (long)i;
        const double *q = cp + 16 * (long)i;
        v8d z1 = VL8(p), z2 = VL8(p + 8);
        v8d c1 = VL8(q), c2 = VL8(q + 8);
        v8d v1 = z1 + c1, v2 = z2 + c2;
        v8d re = SH8(v1, v2, 0, 2, 4, 6, 8, 10, 12, 14);
        v8d im = SH8(v1, v2, 1, 3, 5, 7, 9, 11, 13, 15);
        v8d tiny = {1e-300, 1e-300, 1e-300, 1e-300,
                    1e-300, 1e-300, 1e-300, 1e-300};
        v8d t = im * im + (re * re + tiny);
        v8d y = (v8d)_mm512_rsqrt14_pd((__m512d)t);
        v8d th = 0.5 * t;
        y = y * (1.5 - th * (y * y));
        y = y * (1.5 - th * (y * y));
        v8d d = t * y + 1.0;
        v8d one = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
        v8d w = one / d;
        v8d w1 = SH8(w, w, 0, 0, 1, 1, 2, 2, 3, 3);
        v8d w2 = SH8(w, w, 4, 4, 5, 5, 6, 6, 7, 7);
        VS8(p, v1 * w1);
        VS8(p + 8, v2 * w2);
    }
}

/* the 289th complex of a plane: exact 128-bit sqrt + div, once per plane */
static __attribute__((noinline)) void
ms_map_tail(double *restrict s, const double *restrict cp)
{
    __m128d z = _mm_loadu_pd(s + 576), cv = _mm_loadu_pd(cp + 576);
    __m128d v = _mm_add_pd(z, cv);
    __m128d sq = _mm_mul_pd(v, v);
    __m128d t = _mm_add_pd(sq, _mm_shuffle_pd(sq, sq, 1));
    __m128d d = _mm_add_pd(_mm_sqrt_pd(t), _mm_set1_pd(1.0));
    _mm_storeu_pd(s + 576, _mm_div_pd(v, d));
}

/* Addr-safe t1 base-shift table, mode 1 only (X-first: loads walk `in` rows
 * at 4624 B, stores walk t1 rows at 5120 B; find, per 16-byte residue class
 * of (t1 - vin) mod 4096, the 64-byte shift minimizing weighted 4K-aliasing
 * collisions).  Their l17_as_build, verbatim; pure integer arithmetic. */
static void l17r_ms_as_build(unsigned char astab[256])
{
    for (int cls = 0; cls < 256; ++cls) {
        long bests = 0x7fffffffL;
        int bestj = 0;
        for (int j = 0; j < 64; ++j) {
            long t = (16L * cls + 64L * j) & 4095L;
            long score = 0;
            for (int x = 0; x < 17 && score < bests; ++x)
                for (int k = 0; k < 17; ++k)
                    for (int g = 1; g <= 3; ++g) {
                        long diff = 4624L * x + 64L * g - t - 5120L * k;
                        diff &= 4095L;
                        if (diff >= 2048) diff -= 4096;
                        if (diff > -64 && diff < 64) score += 4 - g;
                    }
            if (score < bests) { bests = score; bestj = j; }
        }
        astab[cls] = (unsigned char)bestj;
    }
}

/* One volume through all m chain steps, IN PLACE on vst (their l17_chain_v2:
 * X pass to the shifted t1, then deferred-Z Y/Z planes with the map after
 * each Z).  domap=0 is the create-time probe form only (raw FFT, no map --
 * callers must not feed its output back, values are unnormalized). */
static __attribute__((noinline)) void
l17r_chain_ms_vol(fft3d_plan *p, const double *vx0, const double *vc,
                  double *vst, int m, int domap)
{
    const double *restrict cn = p->msc;
    double *restrict pb = p->mspb;
    double *restrict pb2 = p->mspb2;
    v8d K0 = VL8(p->mss + 0),  K1 = VL8(p->mss + 8),
        K2 = VL8(p->mss + 16), K3 = VL8(p->mss + 24),
        K4 = VL8(p->mss + 32), K5 = VL8(p->mss + 40),
        K6 = VL8(p->mss + 48), K7 = VL8(p->mss + 56);
    __asm__("" : "+v"(K0), "+v"(K1), "+v"(K2), "+v"(K3),
                 "+v"(K4), "+v"(K5), "+v"(K6), "+v"(K7));
    for (int s = 0; s < m; ++s) {
        const double *vin = s ? vst : vx0;
        double *t1 = p->mst1
            + 8L * p->msas[(((uintptr_t)p->mst1 - (uintptr_t)vin) & 4095u) >> 4];
        MS_XPASS(vin, 578, t1, MS_T1SP);
        MS_GROUP(t1, 34, pb, MS_PBROW);                          /* Y(0) */
        for (int x = 0; x < 17; ++x) {
            double *pba = (x & 1) ? pb2 : pb;
            double *pbn = (x & 1) ? pb : pb2;
            double *pt = vst + (long)x * 578;
            if (x + 1 < 17)
                MS_GROUP(t1 + (long)(x + 1) * MS_T1SP, 34, pbn, MS_PBROW);
            MS_GROUP(pba, MS_PBROW, pt, 34);                     /* Z(x) */
            if (domap) {
                ms_map_vecs(pt, vc + (long)x * 578, 0, 36);
                ms_map_tail(pt, vc + (long)x * 578);
            }
        }
    }
}

/* ===========================================================================
 * ice_r7: PORTED CHAIN ENGINE v6 ("ms6") -- L17_matrixsimd's ice_r6 padded
 * in-place strided engine, ADOPTED WHOLESALE (their l17_chain_v6 /
 * chunk17zri / L17_V6_A2G / l17_map_vec1, structure originally from rival
 * pipeline 1760b1bf's generator).  Their matched A/Bs price it ~-0.85
 * us/step over the shared v2-family engine this file scored with in r6;
 * their r6 scored 11.935 vs my 12.284 on engines that differ ONLY by this.
 *
 * The shape (see their strategy record, round ice_r6, for the full story):
 *   (1) state rows padded 17 -> 20 complex (320 B) in a private arena, so
 *       every strided access in every pass is a full aligned cache line;
 *   (2) slab stride 808 doubles = 6464 B = 101 lines, ODD -> the cross-slab
 *       pass spreads over L1 sets;
 *   (3) passes b (axis j0) and a1 (axis j1) transform IN PLACE along the
 *       stride -- zero shuffles, no t1 buffer, no addr-safe machinery;
 *   (4) pass a2 (axis j2, the in-row direction) runs 4-row groups through a
 *       transposed 20-vector stack array (4x4-complex tile transposes both
 *       ways), with the s6 map interleaved at GROUP granularity;
 *   (5) the 17th row rides a CROSS-SLAB fringe (4 full zmm groups over 16
 *       slabs' row-16 lines + 1 overlap group storing only slab 16's lane);
 *   (6) a1(s0+1) is software-pipelined into a2(s0)'s groups (port mix:
 *       a1 is load/FMA, a2 is shuffle-heavy);
 *   (7) the state lives in the arena for the whole volume-chain: one unpack,
 *       one pack, amortized by m.
 * Padding lanes hold zeros forever: zero in -> zero out through every linear
 * pass, map(0+0) == 0 (the 1e-300 bias keeps rsqrt14 finite), so pads never
 * contaminate and never denormalize.
 *
 * ONE deviation from the donor, bit-identical: the a2 stack array io_[] is
 * declared __attribute__((aligned(64))).  Their vd_w4 typedef (like this
 * file's v8d) is aligned(8), so nothing guarantees their stack tiles sit on
 * cache lines; an explicit 64-B alignment is free and can only help (the r6
 * arena-misalignment find, applied to the stack).
 *
 * Same safety pattern as the r5 msr port: ms6 enters the create-time race
 * only after a numerical cross-check against the already-gated msr chain
 * (different pass order and a different row-289 map site -> different
 * rounding class ~1e-15 apart; a transcription bug would be ~1e0, and the
 * candidate falls back instead of shipping a fast wrong chain).
 * =========================================================================== */

/* -DL17R_V6_XS=0: per-slab row-16 fringe (17 recompute groups) instead of
 * the cross-slab form.  -DL17R_V6_P1=0: disable the a1(s0+1) pipelining.
 * Both are the donor's own A/B hooks; defaults are their shipped winners. */
#ifndef L17R_V6_XS
#  define L17R_V6_XS 1
#endif
#ifndef L17R_V6_P1
#  define L17R_V6_P1 1
#endif
#if L17R_V6_P1 && !L17R_V6_XS
#  error "L17R_V6_P1 requires L17R_V6_XS=1"
#endif
/* -DL17R_V6_FB=1: enable the fringe/b overlap (ice_r7, built and MEASURED
 * A NEGATIVE -- default OFF): the cross-slab row-16 fringe of step s-1
 * interleaved into pass b of step s (legal: pass b's j1 = 0..15 chunks
 * touch rows the fringe never writes, and the 5 j1 = 16 chunks run last;
 * a pure reorder of independent lanewise work -- chain outputs
 * cmp-verified BIT-IDENTICAL both ways).  This was the donor's r6 "open"
 * item, est. -0.2..-0.3 us; same-lease matched pairs read fb1
 * 11.943/11.947/11.969 vs fb0 11.919/11.927/11.942 -- a 0.02-0.05 us LOSS,
 * 3/3.  The fringe evidently is not an exposed tail: the OoO window
 * already overlaps it with the next pass b at the seam, and the interleave
 * only disturbs pass b's stride stream.  Kept as a hook for re-A/Bs. */
#ifndef L17R_V6_FB
#  define L17R_V6_FB 0
#endif
#if L17R_V6_FB && !L17R_V6_XS
#  error "L17R_V6_FB requires L17R_V6_XS=1"
#endif

/* In-place twin of ms_chunk17zr (their chunk17zri, verbatim arithmetic):
 * same 17 loads in the same pair order, same accumulator update order, same
 * slot table, but load and store go through ONE non-restrict pointer at the
 * same strided addresses -- chunk17zr with src == dst would violate its
 * restrict qualifiers (UB).  All 17 loads precede all 17 stores in program
 * order, so in-place is well-defined.  Plain-store path only; the v6
 * transposes live in the a2 glue macro, never in the kernel. */
static inline __attribute__((always_inline)) void
ms_chunk17zri(double *b, long rs, const double *restrict cn,
              v8d K0, v8d K1, v8d K2, v8d K3, v8d K4, v8d K5, v8d K6, v8d K7)
{
    v8d A0, A1, A2, A3, X0a, B0, B1, B2, B3;
    v8d S0, S1, S2, S3, S4, S5, S6, S7;
    v8d U;
    {
        v8d x0 = VL8(b);
        A0 = x0; A1 = x0; A2 = x0; A3 = x0; X0a = x0;
    }
    { /* m=0, rows (1,16) */
        v8d a = VL8(b + 1 * rs), bb = VL8(b + 16 * rs);
        U = a + bb;
        v8d w = MS_MULI(a - bb);
        S0 = K0 * w; S1 = K1 * w; S2 = K2 * w; S3 = K3 * w;
        S4 = K4 * w; S5 = K5 * w; S6 = K6 * w; S7 = K7 * w;
    }
    { /* m=4, rows (13,4) */
        v8d a = VL8(b + 13 * rs), bb = VL8(b + 4 * rs);
        v8d u = a + bb;
        v8d w = MS_MULI(a - bb);
        S0 += K4 * w; S1 += K5 * w; S2 += K6 * w; S3 += K7 * w;
        S4 -= K0 * w; S5 -= K1 * w; S6 -= K2 * w; S7 -= K3 * w;
        v8d pp = U + u, q = U - u;
        X0a += pp;
        A0 += MS_CGET(cn, 0) * pp; A1 += MS_CGET(cn, 1) * pp;
        A2 += MS_CGET(cn, 2) * pp; A3 += MS_CGET(cn, 3) * pp;
        B0 = MS_CGET(cn, 4) * q;  B1 = MS_CGET(cn, 5) * q;
        B2 = MS_CGET(cn, 6) * q;  B3 = MS_CGET(cn, 7) * q;
    }
    { /* m=1, rows (3,14) */
        v8d a = VL8(b + 3 * rs), bb = VL8(b + 14 * rs);
        U = a + bb;
        v8d w = MS_MULI(a - bb);
        S0 += K1 * w; S1 += K2 * w; S2 += K3 * w; S3 += K4 * w;
        S4 += K5 * w; S5 += K6 * w; S6 += K7 * w; S7 -= K0 * w;
    }
    { /* m=5, rows (5,12) */
        v8d a = VL8(b + 5 * rs), bb = VL8(b + 12 * rs);
        v8d u = a + bb;
        v8d w = MS_MULI(a - bb);
        S0 += K5 * w; S1 += K6 * w; S2 += K7 * w; S3 -= K0 * w;
        S4 -= K1 * w; S5 -= K2 * w; S6 -= K3 * w; S7 -= K4 * w;
        v8d pp = U + u, q = U - u;
        const double *r = cn + (size_t)8 * 8;
        X0a += pp;
        A0 += MS_CGET(r, 0) * pp; A1 += MS_CGET(r, 1) * pp;
        A2 += MS_CGET(r, 2) * pp; A3 += MS_CGET(r, 3) * pp;
        B0 += MS_CGET(r, 4) * q;  B1 += MS_CGET(r, 5) * q;
        B2 += MS_CGET(r, 6) * q;  B3 += MS_CGET(r, 7) * q;
    }
    { /* m=2, rows (9,8) */
        v8d a = VL8(b + 9 * rs), bb = VL8(b + 8 * rs);
        U = a + bb;
        v8d w = MS_MULI(a - bb);
        S0 += K2 * w; S1 += K3 * w; S2 += K4 * w; S3 += K5 * w;
        S4 += K6 * w; S5 += K7 * w; S6 -= K0 * w; S7 -= K1 * w;
    }
    { /* m=6, rows (15,2) */
        v8d a = VL8(b + 15 * rs), bb = VL8(b + 2 * rs);
        v8d u = a + bb;
        v8d w = MS_MULI(a - bb);
        S0 += K6 * w; S1 += K7 * w; S2 -= K0 * w; S3 -= K1 * w;
        S4 -= K2 * w; S5 -= K3 * w; S6 -= K4 * w; S7 -= K5 * w;
        v8d pp = U + u, q = U - u;
        const double *r = cn + (size_t)16 * 8;
        X0a += pp;
        A0 += MS_CGET(r, 0) * pp; A1 += MS_CGET(r, 1) * pp;
        A2 += MS_CGET(r, 2) * pp; A3 += MS_CGET(r, 3) * pp;
        B0 += MS_CGET(r, 4) * q;  B1 += MS_CGET(r, 5) * q;
        B2 += MS_CGET(r, 6) * q;  B3 += MS_CGET(r, 7) * q;
    }
    { /* m=3, rows (10,7) */
        v8d a = VL8(b + 10 * rs), bb = VL8(b + 7 * rs);
        U = a + bb;
        v8d w = MS_MULI(a - bb);
        S0 += K3 * w; S1 += K4 * w; S2 += K5 * w; S3 += K6 * w;
        S4 += K7 * w; S5 -= K0 * w; S6 -= K1 * w; S7 -= K2 * w;
    }
    { /* m=7, rows (11,6) */
        v8d a = VL8(b + 11 * rs), bb = VL8(b + 6 * rs);
        v8d u = a + bb;
        v8d w = MS_MULI(a - bb);
        S0 += K7 * w; S1 -= K0 * w; S2 -= K1 * w; S3 -= K2 * w;
        S4 -= K3 * w; S5 -= K4 * w; S6 -= K5 * w; S7 -= K6 * w;
        v8d pp = U + u, q = U - u;
        const double *r = cn + (size_t)24 * 8;
        X0a += pp;
        A0 += MS_CGET(r, 0) * pp; A1 += MS_CGET(r, 1) * pp;
        A2 += MS_CGET(r, 2) * pp; A3 += MS_CGET(r, 3) * pp;
        B0 += MS_CGET(r, 4) * q;  B1 += MS_CGET(r, 5) * q;
        B2 += MS_CGET(r, 6) * q;  B3 += MS_CGET(r, 7) * q;
    }
    v8d C0 = A0 + B0, C1 = A1 + B1, C2 = A2 + B2, C3 = A3 + B3;
    v8d C4 = A0 - B0, C5 = A1 - B1, C6 = A2 - B2, C7 = A3 - B3;
    VS8(b + 0 * rs, X0a);
    VS8(b + 1 * rs, C0 + S0);
    VS8(b + 2 * rs, C6 - S6);
    VS8(b + 3 * rs, C1 + S1);
    VS8(b + 4 * rs, C4 - S4);
    VS8(b + 5 * rs, C5 + S5);
    VS8(b + 6 * rs, C7 - S7);
    VS8(b + 7 * rs, C3 - S3);
    VS8(b + 8 * rs, C2 - S2);
    VS8(b + 9 * rs, C2 + S2);
    VS8(b + 10 * rs, C3 + S3);
    VS8(b + 11 * rs, C7 + S7);
    VS8(b + 12 * rs, C5 - S5);
    VS8(b + 13 * rs, C4 + S4);
    VS8(b + 14 * rs, C1 - S1);
    VS8(b + 15 * rs, C6 + S6);
    VS8(b + 16 * rs, C0 - S0);
}

/* Map one zmm vector (4 complex) in place: the fringe rows' odd vector.
 * Same scalar ladder per point as ms_map_vecs (rsqrt14 + 2 Newton, one
 * correctly rounded vdivpd, mul out) => same bits per point; the packed w
 * lanes 4..7 are duplicates discarded by the expand shuffle.  noinline: one
 * codegen for every call site (their l17_map_vec1, verbatim). */
static __attribute__((noinline)) void
ms_map_vec1(double *restrict s, const double *restrict cp)
{
    v8d z = VL8(s), cc = VL8(cp);
    v8d v = z + cc;
    v8d re = SH8(v, v, 0, 2, 4, 6, 0, 2, 4, 6);
    v8d im = SH8(v, v, 1, 3, 5, 7, 1, 3, 5, 7);
    v8d tiny = {1e-300, 1e-300, 1e-300, 1e-300,
                1e-300, 1e-300, 1e-300, 1e-300};
    v8d t = im * im + (re * re + tiny);
    v8d y = (v8d)_mm512_rsqrt14_pd((__m512d)t);
    v8d th = 0.5 * t;
    y = y * (1.5 - th * (y * y));
    y = y * (1.5 - th * (y * y));
    v8d d = t * y + 1.0;
    v8d one = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    v8d w = one / d;
    v8d w1 = SH8(w, w, 0, 0, 1, 1, 2, 2, 3, 3);
    VS8(s, v * w1);
}

/* 4x4 complex tile transpose, 8 two-source shuffles (gcc lowers each to one
 * vpermt2pd; the index vectors are loop-invariant and hoisted). */
#define MS_TP4(a0, a1, a2, a3, o0, o1, o2, o3)                                 \
    do {                                                                       \
        v8d t0_ = SH8((a0), (a2), 0, 1, 8, 9, 2, 3, 10, 11);                   \
        v8d t1_ = SH8((a1), (a3), 0, 1, 8, 9, 2, 3, 10, 11);                   \
        v8d t2_ = SH8((a0), (a2), 4, 5, 12, 13, 6, 7, 14, 15);                 \
        v8d t3_ = SH8((a1), (a3), 4, 5, 12, 13, 6, 7, 14, 15);                 \
        (o0) = SH8(t0_, t1_, 0, 1, 8, 9, 2, 3, 10, 11);                        \
        (o1) = SH8(t0_, t1_, 4, 5, 12, 13, 6, 7, 14, 15);                      \
        (o2) = SH8(t2_, t3_, 0, 1, 8, 9, 2, 3, 10, 11);                        \
        (o3) = SH8(t2_, t3_, 4, 5, 12, 13, 6, 7, 14, 15);                      \
    } while (0)

/* One a2 group: 4 rows at stride RST from RP, transformed along j2 via a
 * transposed 20-vector stack io[].  RST = 40 gives 4 adjacent j1 rows of one
 * slab; RST = 808 gives the row-16 line of 4 adjacent SLABS (the cross-slab
 * fringe -- lanes are independent lines, so any 4 lines at a common stride
 * make a group).  ONLYLAST != 0 stores just the row-3 lane (the fringe's
 * overlap group, whose other three lanes are recomputed throwaways).
 * io_ is EXPLICITLY 64-B aligned (see the deviation note above). */
#define MS_V6_A2G(RP, RST, ONLYLAST)                                           \
    do {                                                                       \
        v8d io_[20] __attribute__((aligned(64)));                              \
        for (int t_ = 0; t_ < 5; ++t_) {                                       \
            v8d r0_ = VL8((RP) + 0 * (RST) + 8 * t_);                          \
            v8d r1_ = VL8((RP) + 1 * (RST) + 8 * t_);                          \
            v8d r2_ = VL8((RP) + 2 * (RST) + 8 * t_);                          \
            v8d r3_ = VL8((RP) + 3 * (RST) + 8 * t_);                          \
            MS_TP4(r0_, r1_, r2_, r3_, io_[4 * t_], io_[4 * t_ + 1],           \
                   io_[4 * t_ + 2], io_[4 * t_ + 3]);                          \
        }                                                                      \
        ms_chunk17zri((double *)io_, 8, cn,                                    \
                      K0, K1, K2, K3, K4, K5, K6, K7);                         \
        for (int t_ = 0; t_ < 5; ++t_) {                                       \
            v8d b0_, b1_, b2_, b3_;                                            \
            MS_TP4(io_[4 * t_], io_[4 * t_ + 1], io_[4 * t_ + 2],              \
                   io_[4 * t_ + 3], b0_, b1_, b2_, b3_);                       \
            if (!(ONLYLAST)) {                                                 \
                VS8((RP) + 0 * (RST) + 8 * t_, b0_);                           \
                VS8((RP) + 1 * (RST) + 8 * t_, b1_);                           \
                VS8((RP) + 2 * (RST) + 8 * t_, b2_);                           \
            }                                                                  \
            VS8((RP) + 3 * (RST) + 8 * t_, b3_);                               \
        }                                                                      \
    } while (0)

/* One cross-slab fringe unit, noinline so pass b's loop body carries ONE
 * call site per unit instead of five inlined kernel+transpose copies
 * (I-footprint discipline; the K pins ride the zmm argument registers).
 * onlylast=0: a full group over 4 slabs' row-16 lines at rp, then their
 * maps.  onlylast=1: the overlap group (slabs 13..16, store lane 3 = slab
 * 16 only) and slab 16's row map.  ONLYLAST is a runtime flag inside the
 * A2G store guard, so one codegen serves both. */
static __attribute__((noinline)) void
ms_v6_frg(double *rp, const double *rc, const double *restrict cn,
          int onlylast, int domap,
          v8d K0, v8d K1, v8d K2, v8d K3, v8d K4, v8d K5, v8d K6, v8d K7)
{
    MS_V6_A2G(rp, 808, onlylast);
    if (domap)
        for (int k = onlylast ? 3 : 0; k < 4; ++k) {
            ms_map_vecs(rp + (long)k * 808, rc + (long)k * 808, 0, 2);
            ms_map_vec1(rp + (long)k * 808 + 32, rc + (long)k * 808 + 32);
        }
}

/* The whole batched chain, volume-major: unpack x0+c into the padded
 * arenas, m in-arena steps, pack the mapped end state into final_out.
 * domap=0 is the create-time FFT-residual probe only (raw unnormalized FFT;
 * over 12 probe steps the state grows ~1e22, harmlessly finite -- callers
 * must not read the output as data). */
static __attribute__((noinline)) void
l17r_chain_v6(fft3d_plan *p, const double *x0, const double *cf,
              double *fo, int m, int nb, int domap)
{
    const double *restrict cn = p->msc;
    double *restrict pa = p->msv6a;
    double *restrict ca = p->msv6c;
    v8d K0 = VL8(p->mss + 0),  K1 = VL8(p->mss + 8),
        K2 = VL8(p->mss + 16), K3 = VL8(p->mss + 24),
        K4 = VL8(p->mss + 32), K5 = VL8(p->mss + 40),
        K6 = VL8(p->mss + 48), K7 = VL8(p->mss + 56);
    __asm__("" : "+v"(K0), "+v"(K1), "+v"(K2), "+v"(K3),
                 "+v"(K4), "+v"(K5), "+v"(K6), "+v"(K7));
    for (int b = 0; b < nb; ++b) {
        const double *vx0 = x0 + (size_t)2 * NVOL * b;
        const double *vc  = cf + (size_t)2 * NVOL * b;
        double *vst = fo + (size_t)2 * NVOL * b;
        for (int j0 = 0; j0 < 17; ++j0)
            for (int j1 = 0; j1 < 17; ++j1) {
                memcpy(pa + (long)j0 * 808 + j1 * 40,
                       vx0 + (long)j0 * 578 + j1 * 34, 34 * sizeof(double));
                memcpy(ca + (long)j0 * 808 + j1 * 40,
                       vc + (long)j0 * 578 + j1 * 34, 34 * sizeof(double));
            }
        for (int s = 0; s < m; ++s) {
            /* pass b: axis j0, elements at the 101-line slab stride */
#if L17R_V6_FB
            /* rows 0..15 first, with the PREVIOUS step's fringe units
             * interleaved (they write only row 16, which no chunk here
             * touches); the 5 row-16 chunks run last, after the fringe has
             * finished producing their input */
            {
                const int fr = (s > 0);
                for (int j1 = 0; j1 < 16; ++j1) {
                    double *cb = pa + (long)j1 * 40;
                    for (int ch = 0; ch < 5; ++ch)
                        ms_chunk17zri(cb + 8 * ch, 808,
                                      cn, K0, K1, K2, K3, K4, K5, K6, K7);
                    if (fr && j1 == 2)
                        ms_v6_frg(pa + 640, ca + 640, cn, 0, domap,
                                  K0, K1, K2, K3, K4, K5, K6, K7);
                    else if (fr && j1 == 5)
                        ms_v6_frg(pa + 4L * 808 + 640, ca + 4L * 808 + 640,
                                  cn, 0, domap,
                                  K0, K1, K2, K3, K4, K5, K6, K7);
                    else if (fr && j1 == 8)
                        ms_v6_frg(pa + 8L * 808 + 640, ca + 8L * 808 + 640,
                                  cn, 0, domap,
                                  K0, K1, K2, K3, K4, K5, K6, K7);
                    else if (fr && j1 == 11)
                        ms_v6_frg(pa + 12L * 808 + 640, ca + 12L * 808 + 640,
                                  cn, 0, domap,
                                  K0, K1, K2, K3, K4, K5, K6, K7);
                    else if (fr && j1 == 14)
                        ms_v6_frg(pa + 13L * 808 + 640, ca + 13L * 808 + 640,
                                  cn, 1, domap,
                                  K0, K1, K2, K3, K4, K5, K6, K7);
                }
                double *cb = pa + 16L * 40;
                for (int ch = 0; ch < 5; ++ch)
                    ms_chunk17zri(cb + 8 * ch, 808,
                                  cn, K0, K1, K2, K3, K4, K5, K6, K7);
            }
#else
            for (int j1 = 0; j1 < 17; ++j1) {
                double *cb = pa + (long)j1 * 40;
                for (int ch = 0; ch < 5; ++ch)
                    ms_chunk17zri(cb + 8 * ch, 808,
                                  cn, K0, K1, K2, K3, K4, K5, K6, K7);
            }
#endif
            /* slab-resident phase: a1 + a2 rows 0..15, map at a2-group
             * granularity (their matched A/B: flat per-slab map sweep loses
             * ~0.12 us/step) */
#if L17R_V6_P1
            for (int ch = 0; ch < 5; ++ch)
                ms_chunk17zri(pa + 8 * ch, 40,
                              cn, K0, K1, K2, K3, K4, K5, K6, K7);
            for (int s0 = 0; s0 < 17; ++s0) {
                double *sl = pa + (long)s0 * 808;
                const double *cl = ca + (long)s0 * 808;
                double *nx = sl + 808;
                int hn = s0 + 1 < 17;
                for (int g = 0; g < 4; ++g) {
                    MS_V6_A2G(sl + (long)g * 4 * 40, 40, 0);
                    if (hn)
                        ms_chunk17zri(nx + 8 * g, 40,
                                      cn, K0, K1, K2, K3, K4, K5, K6, K7);
                    if (domap)
                        ms_map_vecs(sl + (long)g * 160, cl + (long)g * 160,
                                    0, 10);
                }
                if (hn)
                    ms_chunk17zri(nx + 8 * 4, 40,
                                  cn, K0, K1, K2, K3, K4, K5, K6, K7);
            }
#else
            for (int s0 = 0; s0 < 17; ++s0) {
                double *sl = pa + (long)s0 * 808;
                const double *cl = ca + (long)s0 * 808;
                for (int ch = 0; ch < 5; ++ch)
                    ms_chunk17zri(sl + 8 * ch, 40,
                                  cn, K0, K1, K2, K3, K4, K5, K6, K7);
#if !L17R_V6_XS /* per-slab fringe: a rows-13..16 overlap group produces row
                 * 16 (3 recomputed throwaway lanes per slab) */
                MS_V6_A2G(sl + 13 * 40, 40, 1);
#endif
                for (int g = 0; g < 4; ++g) {
                    MS_V6_A2G(sl + (long)g * 4 * 40, 40, 0);
                    if (domap)
                        ms_map_vecs(sl + (long)g * 160, cl + (long)g * 160,
                                    0, 10);
                }
#if !L17R_V6_XS
                if (domap) {
                    ms_map_vecs(sl + 640, cl + 640, 0, 2); /* row 16 */
                    ms_map_vec1(sl + 672, cl + 672);
                }
#endif
            }
#endif
#if L17R_V6_XS
            /* cross-slab row-16 fringe: 4 full groups over 16 slabs' row-16
             * lines + one overlap group (store slab 16's lane only).
             * Co-lanes of the overlap group are already-transformed (and
             * mapped) rows: bounded garbage in, discarded out -- every
             * vector op is lanewise and the tile transpose only moves
             * complex units, so lane values never mix.  Under FB the fringe
             * of every non-final step rides the NEXT step's pass b instead;
             * only the last step (nothing left to overlap with) runs it
             * here. */
#if L17R_V6_FB
            if (s == m - 1)
#endif
            {
                for (int sb = 0; sb < 4; ++sb)
                    ms_v6_frg(pa + (long)sb * 4 * 808 + 640,
                              ca + (long)sb * 4 * 808 + 640, cn, 0, domap,
                              K0, K1, K2, K3, K4, K5, K6, K7);
                ms_v6_frg(pa + 13L * 808 + 640, ca + 13L * 808 + 640,
                          cn, 1, domap,
                          K0, K1, K2, K3, K4, K5, K6, K7);
            }
#endif
        }
        for (int j0 = 0; j0 < 17; ++j0)
            for (int j1 = 0; j1 < 17; ++j1)
                memcpy(vst + (long)j0 * 578 + j1 * 34,
                       pa + (long)j0 * 808 + j1 * 40, 34 * sizeof(double));
    }
}
#endif /* __AVX512VL__ (msr ported engine) */

/* ------------------------------------------------------------- interface */

const char *fft3d_name(void) { return "L17_rader"; }

static char g_desc[704] =   /* ice_r7: grown for the ms6 telemetry fields */
    "Rader-17 in cyclic/negacyclic form (kernel from L17_winograd), split "
    "re/im, plane-fused, plan-time width tuning";

const char *fft3d_description(void) { return g_desc; }

int fft3d_supports(int L) { return L == LN; }

/* -------------------------------------------------------------- the tuner */

typedef void (*l17r_fn)(fft3d_plan *, const double _Complex *,
                        double _Complex *);

/* Candidate classes.  Class A (X-last) serves batch < L17R_XF_CUT, class B
 * (X-first) serves batch >= L17R_XF_CUT; the class is fixed by the batch size
 * (bit-equivalence discipline from L17_matrixsimd panel_r3), the tuner ranks
 * only within it.  The pinned kernels enter only where the EVEX file exists;
 * elsewhere the plain widths compete (the emulated 512-bit build on an AVX2
 * host measures slow and eliminates itself).  "t" = mixed ymm tail. */
#if defined(__AVX512VL__)
#  define L17R_NCA 24
#  define L17R_NCB 4
#else
#  define L17R_NCA 3
#  define L17R_NCB 3
#endif
static const l17r_fn l17r_cand_a[L17R_NCA] = {
    exec_np_w4, exec_np_w8, exec_npm_w8,
#if defined(__AVX512VL__)
    exec_pin_w8, exec_pinm_w8, exec_ovm_w8, exec_ovmpin_w8, exec_spm_w8,
    exec_dzm_w8, exec_dzmpin_w8, exec_dzspm_w8,
    exec_npmdy_w8, exec_spmdy_w8,          /* panel_r9: ymm-deint twins */
    exec_stm_w8, exec_stmdy_w8,            /* panel_r10: staged dense out */
    exec_stpm_w8, exec_stpmdy_w8,          /* panel_r10: + pipelined flush */
    exec_npty_w8, exec_pinty_w8,           /* ice_r1: ymm transposes (p5   */
    exec_npmty_w8, exec_pinmty_w8,         /* relief on ICX)               */
    exec_spmpin_w8, exec_spmdypin_w8,      /* ice_r1: pin twins of sp      */
    exec_pin_w4,                           /* ice_r1: MUST STAY LAST (the
                                            * width discipline knows the w4
                                            * entries as 0 and NCA-1)      */
#endif
};
static const char *const l17r_tag_a[L17R_NCA] = {
    "xl 256", "xl 512", "xl 512t",
#if defined(__AVX512VL__)
    "xl 512 pin", "xl 512t pin", "xl 512t ov", "xl 512t ov pin", "xl 512t sp",
    "xl 512t dz", "xl 512t dz pin", "xl 512t dzsp",
    "xl 512t dy", "xl 512t sp dy",
    "xl 512t st", "xl 512t st dy",
    "xl 512t stp", "xl 512t stp dy",
    "xl 512 ty", "xl 512 pin ty",
    "xl 512t ty", "xl 512t pin ty",
    "xl 512t sp pin", "xl 512t sp dy pin",
    "xl 256 pin",
#endif
};
static const l17r_fn l17r_cand_b[L17R_NCB] = {
    exec_xf_w4, exec_xf_w8, exec_xfm_w8,
#if defined(__AVX512VL__)
    exec_xfpinm_w8,
#endif
};
static const char *const l17r_tag_b[L17R_NCB] = {
    "xfs 256", "xfs 512", "xfs 512t",
#if defined(__AVX512VL__)
    "xfs 512t pin",
#endif
};

/* --------------- ice_r4: fused-chain candidates and runner ----------------
 * All candidates share the one map ladder and identical FFT arithmetic
 * (pin/dy/fd are codegen and data-movement choices only), so the timing race
 * is free: a cross-process flip cannot change the chain's output bits. */
#if defined(__AVX512VL__)
#define L17R_NCH 14
static const l17r_chfn l17r_chcand[L17R_NCH] = {
    chain_xk_w8, chain_xkpin_w8,
    chain_xm_w8, chain_xmpin_w8, chain_xmty_w8, chain_xmtypin_w8,
    chain_dzxm_w8, chain_dzxmpin_w8, chain_xm_w4, chain_xmpin_w4,
    chain_mps_w8, chain_mpspin_w8, chain_mp_w8, chain_mppin_w8,
};
static const char *const l17r_chtag[L17R_NCH] = {
    "xk", "xk pin",
    "xm", "xm pin", "xm ty", "xm ty pin", "xm dz", "xm dz pin",
    "xm 256", "xm 256 pin", "mps", "mps pin", "mp", "mp pin",
};
/* eager = the step maps AT THE X-PASS STORE (state buffers hold mapped
 * state, the last step writes final_out directly, no materialization) */
static const int l17r_cheager[L17R_NCH] = { 1,1, 1,1,1,1,1,1,1,1, 0,0,0,0 };

/* One volume through the whole m-step chain, everything L2-resident: the
 * corpus-consensus locality design (sec 10.3: "iterate each volume through
 * all m steps while cache-resident; never sweep passes across volumes" --
 * the driver's fallback sweeps, keeping 3 full batches live in L3).  Lazy
 * shapes keep the state RAW between steps, map it inside the next step's
 * first pass and materialize once at the end; eager shapes map at each
 * step's x-pass store and finish in place. */
static void l17r_chain_vol(fft3d_plan *p, l17r_chfn f, int eager,
                           const double *x0, const double *cf, double *out,
                           int m)
{
    const double *cur = x0;
    if (eager) {
        for (int s = 0; s < m; ++s) {
            double *d = (s == m - 1) ? out : ((s & 1) ? p->sc1 : p->sc0);
            f(p, cur, cf, d, 1);
            cur = d;
        }
    } else {
        for (int s = 0; s < m; ++s) {
            double *d = (s & 1) ? p->sc1 : p->sc0;
            f(p, cur, cf, d, s > 0);
            cur = d;
        }
        l17r_map_run(cur, cf, out, NVOL);
    }
}
#endif

static void l17r_tune_free(fft3d_plan *p)
{
    free(p->tin);
    free(p->tout);
    p->tin = NULL;
    p->tout = NULL;
    p->tn = 0;
}

/* Deterministic pseudo-random scratch input: the tuner needs realistic
 * magnitudes, not the real data. */
static int l17r_tune_alloc(fft3d_plan *p, int nv)
{
    size_t n = (size_t)nv * NVOL;
    if (p->tn >= n) return 1;
    l17r_tune_free(p);
    if (posix_memalign((void **)&p->tin, 64, n * sizeof *p->tin) != 0) {
        p->tin = NULL;
        return 0;
    }
    if (posix_memalign((void **)&p->tout, 64, n * sizeof *p->tout) != 0) {
        free(p->tin);
        p->tin = NULL;
        p->tout = NULL;
        return 0;
    }
    p->tn = n;
    unsigned sr = 987654321u;
    for (size_t i = 0; i < n; ++i) {
        sr = sr * 1103515245u + 12345u;
        double a = (double)(sr >> 8) / 8388608.0 - 1.0;
        sr = sr * 1103515245u + 12345u;
        double b = (double)(sr >> 8) / 8388608.0 - 1.0;
        p->tin[i] = a + b * (double _Complex)I;
    }
    memset(p->tout, 0, n * sizeof *p->tout);
    return 1;
}

static int l17r_verbose(void)
{
#if defined(L17R_VERBOSE_BUILD)
    return 1;      /* tryout runs over ssh where env vars do not survive */
#else
    const char *e = getenv("L17R_VERBOSE");
    return e && *e && *e != '0';
#endif
}

static double l17r_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Streaming-arena size in volumes: min(batch, clamp(2.5*L3/157KB, 384, 1024)).
 * Adopted from L17_matrixsimd panel_r4 (itself from L36_mixedradix): a fixed
 * 384-volume arena is exactly wallaby's 60 MB L3 and tunes the L3-resident
 * regime there; the node (22 MB) still gets 384, bit-for-bit as before. */
static int l17r_arena_nv(void)
{
#ifdef _SC_LEVEL3_CACHE_SIZE
    long l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
#else
    long l3 = 0;
#endif
    if (l3 <= 0) return 384;
    double nv = 2.5 * (double)l3 / (157.0 * 1024.0);
    if (nv < 384.0) nv = 384.0;
    if (nv > 1024.0) nv = 1024.0;
    return (int)nv;
}

/* Sustained core clock at each vector width, measured at plan time
 * (unscored) and reported in fft3d_description() as clk256/clk512 -- the
 * panel_r4 VERDICT's explicit L=17 ask: every L=17 kernel is zmm and nobody
 * has measured the AVX-512 licence clock on the node.  Method ADOPTED FROM
 * L6_unrolled (panel_r4): a serially dependent FMA chain, latency 4 cycles
 * at BOTH widths on SKX/CLX/ICL/SPR, so freq = iters*4/time; best of 5
 * trials after the tournament has warmed the core.  (Haswell dev host: FMA
 * latency 5 and the 512-bit type is emulated -- both numbers are dev-only
 * noise there; wallaby and the node are what matters.) */
typedef double l17r_v8d __attribute__((vector_size(64), aligned(8)));

static double l17r_probe_ghz(int wide)
{
    double best = 0.0;
    if (wide) {
        l17r_v8d x = {1,1,1,1,1,1,1,1};
        const l17r_v8d a = ((l17r_v8d){0} + (1.0 + 1e-15));
        const l17r_v8d b = ((l17r_v8d){0} + 1e-300);
        double warm_until = l17r_now() + 4e-3;
        do {
            for (int i = 0; i < 8192; ++i) x = x * a + b;
        } while (l17r_now() < warm_until);
        for (int trial = 0; trial < 5; ++trial) {
            double t0 = l17r_now();
            for (int i = 0; i < 262144; ++i) x = x * a + b;
            double dt = l17r_now() - t0;
            double ghz = 262144.0 * 4.0 / dt * 1e-9;
            if (ghz > best) best = ghz;
        }
        if (!(x[0] > 0.0)) best = 0.0;     /* keep the chain observable */
    } else {
        v4d x = {1,1,1,1};
        const v4d a = ((v4d){0} + (1.0 + 1e-15));
        const v4d b = ((v4d){0} + 1e-300);
        double warm_until = l17r_now() + 4e-3;
        do {
            for (int i = 0; i < 8192; ++i) x = x * a + b;
        } while (l17r_now() < warm_until);
        for (int trial = 0; trial < 5; ++trial) {
            double t0 = l17r_now();
            for (int i = 0; i < 262144; ++i) x = x * a + b;
            double dt = l17r_now() - t0;
            double ghz = 262144.0 * 4.0 / dt * 1e-9;
            if (ghz > best) best = ghz;
        }
        if (!(x[0] > 0.0)) best = 0.0;
    }
    if (best > 9.9) best = 0.0;
    return best;
}

/* Spin a real zmm exec for ~150 ms before ranking, so the turbo/licence
 * clock has settled by the first candidate.  Without this the first two or
 * three candidates in a rank are measured on a still-ramping clock (seen on
 * wallaby at nv=256: early candidates 21-30 us/t, late ones 11.9 for
 * bit-identical work) and the pick is an artifact of table order. */
static void l17r_settle(fft3d_plan *p, int nv)
{
    int sb = p->batch, sp = p->pf, sw = p->pfw;
    p->batch = nv;
    p->pf = 0;
    p->pfw = 0;
    double t0 = l17r_now();
    do
        l17r_cand_a[L17R_NCA > 3 ? 1 : 0](p, p->tin, p->tout);
    while (l17r_now() - t0 < 0.15);
    p->batch = sb;
    p->pf = sp;
    p->pfw = sw;
}

/* Time each candidate in a block of >= 64 consecutive volume transforms,
 * never interleaved (L17_matrixsimd round-1 item 12: interleaving ISA widths
 * mis-ranked candidates by 35% on the node).  Returns the fastest index. */
static int l17r_rank(fft3d_plan *p, int nv, double *best_us,
                     const l17r_fn *cand, int ncand, int reps)
{
    int inner = (64 + nv - 1) / nv;
    if (inner < 1) inner = 1;
    int sb = p->batch, sp = p->pf, sw = p->pfw;
    p->batch = nv;
    p->pf = 0;
    p->pfw = 0;
    int bestv = 0;
    /* Two full fixed-order sweeps, per-candidate min across both (panel_r7):
     * even with the settle spin and per-candidate warmups, a machine whose
     * clock ramps over the whole tournament (wallaby contended: one verbose
     * table spanned 27 -> 10 us/t monotonically for near-identical work)
     * biases early table slots.  A second sweep runs on the settled clock,
     * so the min per candidate halves the order bias at ~2x plan time.
     * Within a sweep each candidate is still timed in its own contiguous
     * block (never interleaved -- L17_matrixsimd round-1 item 12). */
    for (int v = 0; v < ncand; ++v) best_us[v] = 1e30;
    for (int pass = 0; pass < 2; ++pass) {
        for (int v = 0; v < ncand; ++v) {
            double best = 1e30;
            /* Warmup: page in, and let the turbo/licence level settle to THIS
             * candidate's own.  Time-based (>= 1.5 ms > Intel's ~670 us
             * licence-up dwell): at the B=1 stage one exec is ~200-350 us, so
             * the old 2-exec warmup left a ymm candidate measured inside the
             * AVX-512 licence of the previous zmm candidate (clk256 = 3.89 vs
             * clk512 = 2.89 on the node, r5 probes). */
            double w0 = l17r_now();
            cand[v](p, p->tin, p->tout);
            cand[v](p, p->tin, p->tout);
            while (l17r_now() - w0 < 1.5e-3)
                cand[v](p, p->tin, p->tout);
            for (int r = 0; r < reps; ++r) {
                double t0 = l17r_now();
                for (int q = 0; q < inner; ++q) cand[v](p, p->tin, p->tout);
                double dt = l17r_now() - t0;
                if (dt < best) best = dt;
            }
            double us = best * 1e6 / ((double)nv * inner);
            if (us < best_us[v]) best_us[v] = us;
        }
    }
    for (int v = 0; v < ncand; ++v)
        if (best_us[v] < best_us[bestv]) bestv = v;
    p->batch = sb;
    p->pf = sp;
    p->pfw = sw;
    return bestv;
}

/* Time one exec-shaped function on the tuner arena at (pf, pfw) = (0, 0):
 * licence/turbo warmup >= 1.5 ms, then min over `reps` whole-arena passes.
 * Used by the panel_r9 streaming decomposition probe (and nothing scored). */
static double __attribute__((unused))
l17r_time_fn(fft3d_plan *p, int nv, l17r_fn f, int reps)
{
    int sb = p->batch, spf = p->pf, sw = p->pfw;
    p->batch = nv;
    p->pf = 0;
    p->pfw = 0;
    double w0 = l17r_now();
    f(p, p->tin, p->tout);
    f(p, p->tin, p->tout);
    while (l17r_now() - w0 < 1.5e-3)
        f(p, p->tin, p->tout);
    double best = 1e30;
    for (int r = 0; r < reps; ++r) {
        double t0 = l17r_now();
        f(p, p->tin, p->tout);
        double dt = l17r_now() - t0;
        if (dt < best) best = dt;
    }
    p->batch = sb;
    p->pf = spf;
    p->pfw = sw;
    return best * 1e6 / nv;
}

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != LN || batch <= 0) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->batch = batch;
    p->pf = 0;
    p->pfw = 0;
    size_t nd = (size_t)2*ABUF4 + (size_t)4*TBUF4
              + (size_t)4*ABUF8 + (size_t)6*TBUF8 + (size_t)10*NVOL + 640
              /* ice_r5 msr engine: cn table + K splats + pb pair + padded t1
               * (17 planes x 640 dbl + 512 dbl of addr-safe shift slack) */
              + 256 + 64 + 2*704 + ((size_t)17*640 + 512 + 64)
              + 8    /* ice_r6: slack for the 64 B realign before msc */
              /* ice_r7 ms6 engine: two padded 17x808-double arenas (state +
               * c field), plus realign slack.  r6's transfer warning applied:
               * re-check alignment whenever buffer counts change. */
              + (size_t)2*17*808 + 8;
    void *raw = NULL;
    if (posix_memalign(&raw, 64, nd * sizeof(double)) != 0 || !raw) {
        free(p);
        return NULL;
    }
    p->mem = raw;
    memset(p->mem, 0, nd * sizeof(double));      /* pad lanes := 0, forever */
    double *q = p->mem;
    p->ar_w4 = q; q += ABUF4;
    p->ai_w4 = q; q += ABUF4;
    p->tr_w4 = q; q += TBUF4;
    p->ti_w4 = q; q += TBUF4;
    p->ur_w4 = q; q += TBUF4;
    p->ui_w4 = q; q += TBUF4;
    p->ar_w8 = q; q += ABUF8;
    p->ai_w8 = q; q += ABUF8;
    p->tr_w8 = q; q += TBUF8;
    p->ti_w8 = q; q += TBUF8;
    p->ur_w8 = q; q += TBUF8;
    p->ui_w8 = q; q += TBUF8;
    p->ar2_w8 = q; q += ABUF8;
    p->ai2_w8 = q; q += ABUF8;
    p->tr2_w8 = q; q += TBUF8;
    p->ti2_w8 = q; q += TBUF8;
    p->vo_w8 = q; q += (size_t)2*NVOL;  /* fully rewritten before every read;
                                         * no pad invariant on either */
    p->vo2_w8 = q; q += (size_t)2*NVOL;
    p->sc0 = q; q += (size_t)2*NVOL;    /* ice_r4 chain buffers: fully */
    p->sc1 = q; q += (size_t)2*NVOL;    /* rewritten before every read */
    p->mv  = q; q += (size_t)2*NVOL;
    p->mp  = q; q += 640;
    /* ice_r6: REALIGN to a 64 B line before the msr buffers.  2*NVOL = 9826
     * = 2 (mod 8 doubles), so the five volume-sized buffers above left q
     * 16 B past a line boundary -- which silently made EVERY msr access a
     * cache-line-SPLIT one: the 32 cosine-splat FMA memory operands per
     * chunk (~7.8k split loads per graded step), every t1 store/load and
     * every pb store/load.  This was the ported engine's whole 15.1-vs-12.7
     * gap to L17_matrixsimd, whose own arena keeps these aligned. */
    q += (size_t)((8 - ((q - (double *)p->mem) & 7)) & 7);
    p->msc = q; q += 256;               /* ice_r5 msr engine buffers   */
    p->mss = q; q += 64;
    p->mspb = q; q += 704;
    p->mspb2 = q; q += 704;
    p->mst1 = q; q += (size_t)17*640 + 512 + 64;
    /* ice_r7: realign (defensively -- the block above is currently a
     * multiple of 8 doubles) so the ms6 arenas sit on 64 B lines; slab
     * stride 808 and row stride 40 are both multiples of 8, so EVERY row of
     * both arenas is line-aligned.  memset(mem) above already zeroed the pad
     * lanes, which stay zero forever (linear passes map 0 -> 0, map(0+0)=0). */
    q += (size_t)((8 - ((q - (double *)p->mem) & 7)) & 7);
    p->msv6a = q; q += (size_t)17*808;
    p->msv6c = q;

#if defined(__AVX512VL__)
    /* msr coefficient tables (long double trig, their exact recipe): the
     * nested-kernel cosine rows cp/cm with the negacyclic signs baked in,
     * splatted 8x so an FMA reads them as full-width memory operands, and
     * the 8 sine constants K0..K7 (row m=0 of their nts8). */
    {
        const long double twopi = 6.283185307179586476925286766559005768394L;
        long double c8v[8], s8v[8], cpv[4], cmv[4];
        int p3 = 1;
        for (int r = 0; r < 8; ++r) {
            c8v[r] = cosl(twopi * (long double)p3 / 17.0L);
            s8v[r] = sinl(twopi * (long double)p3 / 17.0L);
            p3 = (p3 * 3) % 17;
        }
        for (int t = 0; t < 4; ++t) {
            cpv[t] = (c8v[t] + c8v[t + 4]) * 0.5L;
            cmv[t] = (c8v[t] - c8v[t + 4]) * 0.5L;
        }
        for (int mm = 0; mm < 4; ++mm)
            for (int n = 0; n < 8; ++n) {
                double v;
                if (n < 4)
                    v = (double)cpv[(mm + n) & 3];
                else {
                    int nn = n - 4;
                    v = (double)cmv[(mm + nn) & 3] * ((mm + nn) < 4 ? 1.0 : -1.0);
                }
                for (int t = 0; t < 8; ++t) p->msc[(mm * 8 + n) * 8 + t] = v;
            }
        for (int n = 0; n < 8; ++n)
            for (int t = 0; t < 8; ++t)
#ifdef L17R_NO_SFOLD
                p->mss[n * 8 + t] = (double)s8v[n];
#else
                /* ice_r6 sign-fold: odd lanes carry MULI's old sign flip */
                p->mss[n * 8 + t] = (t & 1) ? -(double)s8v[n]
                                            :  (double)s8v[n];
#endif
        l17r_ms_as_build(p->msas);
    }
#endif

    int bestv = 0;
    const char *const *tags = l17r_tag_a;
    p->exec = l17r_cand_a[0];
    double pr_ph = 0.0, pr_xp = 0.0, pr_fu = 0.0;   /* panel_r9 probe */
    double pr_xa = 0.0, pr_xb = 0.0;   /* panel_r11: stage-1 class race,
                                        * published so a decline is data */

    /* ice_r1: the graded cell is B=32 (chain 98, both buffers L3-resident).
     * That used to fall into the small-batch path -- class A only, ranked at
     * nv=16, no pf race, no joint (variant,pf,pfw) grid.  The full batched
     * machinery now starts at batch >= 32 so the graded cell gets the joint
     * grid (pf in particular targets exactly this regime: every src plane is
     * an L3 hit that a one-plane-ahead prefetch can hide under compute). */
    if (batch < 32 && batch < L17R_XF_CUT) {
        /* Small batches: class A (X-last) only, ranked at (a cap of) the
         * plan's own batch size -- for B=1 this times exactly what the driver
         * times: one L2-resident volume. */
        double us1[L17R_NCA];
        int nv = batch < 16 ? batch : 16;
        if (l17r_tune_alloc(p, nv)) {
            l17r_settle(p, nv);
            bestv = l17r_rank(p, nv, us1, l17r_cand_a, L17R_NCA, 3);
#if defined(__AVX512VL__)
            /* Width discipline (ice_r1): all class-A candidates are
             * bit-identical, so a near-tie switch is free -- and the
             * ice_smoke round showed a noisy create picking "xl 256" on a
             * machine with TWO 512-bit FMA pipes (scored 1.47x behind the
             * leader).  The lone w4 candidate must beat the best w8
             * candidate by >3% to be selected; on the CLX node the winner
             * was already w8 (512t family), so this changes nothing there. */
            if (bestv == 0 || bestv == L17R_NCA - 1) {
                int bw8 = 1;
                for (int v = 2; v < L17R_NCA - 1; ++v)
                    if (us1[v] < us1[bw8]) bw8 = v;
                if (!(us1[bestv] < 0.97 * us1[bw8])) bestv = bw8;
            }
#endif
            p->exec = l17r_cand_a[bestv];
            if (l17r_verbose())
                for (int v = 0; v < L17R_NCA; ++v)
                    fprintf(stderr, "[L17_rader tune] nv=%d  %-14s %8.3f us/transform%s\n",
                            nv, l17r_tag_a[v], us1[v], v == bestv ? "  <== kept" : "");
#if defined(L17R_FORCE)
            bestv = (L17R_FORCE) % L17R_NCA;
            p->exec = l17r_cand_a[bestv];
#endif
            /* pfw A/B on the winner at the plan's own batch (panel_r6):
             * prefetchw measured -6.8% at B=1 on wallaby for this structure
             * (the x pass's strided 128 B stores expose the out RFO even
             * L2-resident), but L36_pfa measured +13% at B=1 for theirs --
             * so it is measured here, blocked, 3% margin to switch on. */
            double bw[2] = { 1e30, 1e30 };
            int inner = (64 + nv - 1) / nv;
            int sb1 = p->batch;
            p->batch = nv;
            for (int pass = 0; pass < 2; ++pass)   /* two sweeps: order bias */
            for (int f = 0; f < 2; ++f) {
                p->pfw = f;
                p->exec(p, p->tin, p->tout);
                p->exec(p, p->tin, p->tout);
                for (int r = 0; r < 3; ++r) {
                    double t0 = l17r_now();
                    for (int q = 0; q < inner; ++q)
                        p->exec(p, p->tin, p->tout);
                    double dt = l17r_now() - t0;
                    if (dt < bw[f]) bw[f] = dt;
                }
            }
            p->batch = sb1;
            p->pfw = bw[1] < 0.97 * bw[0];
            if (l17r_verbose())
                fprintf(stderr, "[L17_rader tune] nv=%d  pfw off %.3f  on %.3f"
                                "  -> pfw=%d\n", nv, bw[0] * 1e6 / (nv * inner),
                        bw[1] * 1e6 / (nv * inner), p->pfw);
        }
#if defined(L17R_FORCE)
        bestv = (L17R_FORCE) % L17R_NCA;
        p->exec = l17r_cand_a[bestv];
#endif
    } else {
        /* Batched: rank BOTH classes on a working set past L3 (the candidate
         * that wins L2-resident can lose in the streaming regime --
         * L17_winograd's round-2 lesson), arena scaled to the machine's L3
         * (L17_matrixsimd panel_r4).  The class choice is measured: X-first
         * must beat the X-last incumbent by >3% to be selected (on wallaby it
         * loses outright, r4 record; the node's 22 MB L3 makes B=256 truly
         * stream, so only the node can answer -- this runs r4's requested A/B
         * at every plan).  Then A/B the cross-volume input prefetch on the
         * winner, blocked, never alternating, 3% margin to switch pf on
         * (panel_r3: a near-tie pf=1 pick at B=256 cost 7.4% steady-state). */
        int nv2 = batch < l17r_arena_nv() ? batch : l17r_arena_nv();
        if (l17r_tune_alloc(p, nv2)) {
            double usa[L17R_NCA], usb[L17R_NCB];
            int ba = 0, bb, use_b;
            l17r_settle(p, nv2);
            if (batch >= L17R_XF_CUT) {
                use_b = 1;                       /* forced (dev/monitor A/B) */
                bb = l17r_rank(p, nv2, usb, l17r_cand_b, L17R_NCB, 4);
            } else {
                ba = l17r_rank(p, nv2, usa, l17r_cand_a, L17R_NCA, 4);
                bb = l17r_rank(p, nv2, usb, l17r_cand_b, L17R_NCB, 4);
#if defined(__AVX512VL__)
                /* Width discipline, same as the small-batch path: the lone
                 * w4 candidate in each class must beat that class's best w8
                 * candidate by >3% (bit-identical within a class, so a
                 * near-tie switch is free; ice_smoke's noisy "xl 256" pick
                 * is the target). */
                if (ba == 0 || ba == L17R_NCA - 1) {
                    int bw8 = 1;
                    for (int v = 2; v < L17R_NCA - 1; ++v)
                        if (usa[v] < usa[bw8]) bw8 = v;
                    if (!(usa[ba] < 0.97 * usa[bw8])) ba = bw8;
                }
                if (bb == 0) {
                    int bw8 = 1;
                    for (int v = 2; v < L17R_NCB; ++v)
                        if (usb[v] < usb[bw8]) bw8 = v;
                    if (!(usb[0] < 0.97 * usb[bw8])) bb = bw8;
                }
#endif
                use_b = usb[bb] < 0.97 * usa[ba];
                pr_xa = usa[ba];        /* stage-1 (0,0) race, us/vol -- */
                pr_xb = usb[bb];        /* goes out in the description  */
                if (l17r_verbose()) {
                    for (int v = 0; v < L17R_NCA; ++v)
                        fprintf(stderr, "[L17_rader tune] nv=%d %-14s %8.3f us/transform%s\n",
                                nv2, l17r_tag_a[v], usa[v],
                                (!use_b && v == ba) ? "  <== kept" : "");
                    for (int v = 0; v < L17R_NCB; ++v)
                        fprintf(stderr, "[L17_rader tune] nv=%d %-14s %8.3f us/transform%s\n",
                                nv2, l17r_tag_b[v], usb[v],
                                (use_b && v == bb) ? "  <== kept" : "");
                }
            }
            if (use_b) {
                tags = l17r_tag_b;
                bestv = bb;
                p->exec = l17r_cand_b[bestv];
            } else {
                bestv = ba;
                p->exec = l17r_cand_a[bestv];
            }
#if defined(L17R_FORCE)
            bestv = (L17R_FORCE) % (use_b ? L17R_NCB : L17R_NCA);
            p->exec = use_b ? l17r_cand_b[bestv] : l17r_cand_a[bestv];
#endif
            /* Joint (variant, pf, pfw) grid (panel_r9) -- ADOPTED FROM
             * L23_rader's panel_r8 node result: racing knobs only on the
             * stage-1 winner is a documented, repeated mistake (their
             * plain-xf + pf=2 + pw=1 combo took B=128 where a
             * stage-1-then-grid tuner could never find it).  The sp
             * write-spreading variant's mechanism composes with pfw
             * (prefetchw the rows it is about to scatter across the plane
             * phase), so it gets its (pf, pfw) shot here alongside the
             * incumbent's.  Class A only: all class-A candidates are
             * bit-identical, so a measured variant switch cannot change
             * output bits.  Disabled under L17R_FORCE so forced A/Bs stay
             * clean.  The 3% margin vs the incumbent at (0,0) stands
             * (panel_r3: a near-tie pf=1 pick cost 7.4% steady state;
             * L36_pfa r5: prefetchw on resident lines is +13% uop tax). */
            int part = -1;
#if defined(__AVX512VL__) && !defined(L17R_FORCE)
            /* panel_r10: the grid's partner slot goes to the staged dense
             * out flush ("st"), dy-matched to the winner.  sp held this slot
             * in r9 and was declined 6/6 by the node's grid, alongside q+pfw
             * at L17_winograd -- write SPREADING is falsified on this
             * machine; st instead DELETES the partial-line RFO waste of the
             * 17-stream scattered store (the r9 VERDICT's named direction:
             * "fund traffic deletion at B=256/2048").  If st itself won
             * stage 1, race its unstaged twin so the grid still spans both. */
            if (!use_b) {
                if (bestv >= 13 && bestv <= 16) {
                    /* a staged variant won stage 1 outright: race its
                     * unstaged twin so the grid still spans both shapes */
                    part = (bestv == 14 || bestv == 16) ? 11 : 2;
                } else {
                    /* race the best staged candidate from stage 1's own
                     * streaming rank (st / st dy / stp / stp dy) */
                    part = 13;
                    for (int v = 14; v <= 16; ++v)
                        if (usa[v] < usa[part]) part = v;
                }
            }
#endif
            /* panel_r11: class B (the staged X-first, "xfs") rides the grid
             * as a third variant when it did not already win stage 1 -- the
             * same L23_rader lesson one level up: xfs + pfw is the DESIGNED
             * combination (the one-plane-ahead prefetchw feeds the deferred
             * plane flush), so a (0,0)-only class race can never find it.
             * The cross-class pick keeps the 3% margin AGAINST THE BEST
             * CLASS-A CONFIG, not against (0,0): bits change only when xfs
             * wins the honest joint race.  Gated off under L17R_FORCE. */
            int partb = -1;
#if !defined(L17R_FORCE)
            if (!use_b && batch < L17R_XF_CUT) partb = bb;
#endif
            /* ice_r1: pf is now three-valued in the grid -- 0 off, 1 the
             * volume-ahead shape (DRAM regime), 2 the plane-ahead shape
             * (graded chain: src is L3-resident and only its hit latency
             * needs hiding).  6 configs per variant. */
            double bcfg[18];
            for (int g = 0; g < 18; ++g) bcfg[g] = 1e30;
            const int inc = bestv;              /* incumbent, for the log */
            l17r_fn gfn[3];
            int gcls[3], gvar[3], nvar = 0;
            gfn[0] = p->exec; gcls[0] = use_b; gvar[0] = bestv; nvar = 1;
            if (part >= 0) {
                gfn[nvar] = l17r_cand_a[part];
                gcls[nvar] = 0; gvar[nvar] = part; ++nvar;
            }
            if (partb >= 0) {
                gfn[nvar] = l17r_cand_b[partb];
                gcls[nvar] = 1; gvar[nvar] = partb; ++nvar;
            }
            int sb = p->batch;
            p->batch = nv2;
            for (int pass = 0; pass < 2; ++pass)   /* two sweeps: order bias */
            for (int g = 0; g < 6*nvar; ++g) {
                l17r_fn fn = gfn[g / 6];
                p->pf = g % 3;
                p->pfw = (g / 3) & 1;
                fn(p, p->tin, p->tout);
                for (int r = 0; r < 2; ++r) {
                    double t0 = l17r_now();
                    fn(p, p->tin, p->tout);
                    double dt = l17r_now() - t0;
                    if (dt < bcfg[g]) bcfg[g] = dt;
                }
            }
            p->batch = sb;
            /* same-class configs: 3% margin vs the incumbent at (0,0) */
            int bg = 0;
            for (int g = 1; g < 6*nvar; ++g)
                if (gcls[g / 6] == gcls[0]
                    && bcfg[g] < bcfg[bg] && bcfg[g] < 0.97 * bcfg[0]) bg = g;
            /* cross-class challenger: 3% margin vs the same-class winner */
            int bgx = -1;
            for (int g = 0; g < 6*nvar; ++g)
                if (gcls[g / 6] != gcls[0]
                    && (bgx < 0 || bcfg[g] < bcfg[bgx])) bgx = g;
            if (bgx >= 0 && bcfg[bgx] < 0.97 * bcfg[bg]) bg = bgx;
            p->pf = bg % 3;
            p->pfw = (bg / 3) & 1;
            if (bg >= 6) {
                bestv = gvar[bg / 6];
                use_b = gcls[bg / 6];
                tags = use_b ? l17r_tag_b : l17r_tag_a;
                p->exec = use_b ? l17r_cand_b[bestv] : l17r_cand_a[bestv];
            }
            if (l17r_verbose()) {
                for (int v = 0; v < nvar; ++v)
                    fprintf(stderr, "[L17_rader tune] nv=%d  %s cfg "
                                    "p0w0 %.3f p1w0 %.3f p2w0 %.3f "
                                    "p0w1 %.3f p1w1 %.3f p2w1 %.3f\n", nv2,
                            (gcls[v] ? l17r_tag_b : l17r_tag_a)[v == 0 ? inc : gvar[v]],
                            bcfg[6*v + 0] * 1e6 / nv2, bcfg[6*v + 1] * 1e6 / nv2,
                            bcfg[6*v + 2] * 1e6 / nv2, bcfg[6*v + 3] * 1e6 / nv2,
                            bcfg[6*v + 4] * 1e6 / nv2, bcfg[6*v + 5] * 1e6 / nv2);
                fprintf(stderr, "[L17_rader tune] nv=%d  grid -> %s pf=%d pfw=%d\n",
                        nv2, tags[bestv], p->pf, p->pfw);
            }

            /* Streaming decomposition probe (panel_r9; pattern ADOPTED FROM
             * L36_pfa panel_r8: when the monitor cannot run your counter,
             * put the discriminating measurement inside create() and route
             * it out through the description string).  Times, on the same
             * streaming arena at (pf,pfw)=(0,0): the plane phase alone
             * (reads `in` cold, fills A -- the input-side exposure), the x
             * pass alone (reads hot A, writes `out` -- the output-side
             * exposure), and the full plain exec of the same family.
             * fu - ph - xp ~ 0 means the phases simply add; xp >> its
             * ~4.1 us compute floor quantifies the un-hidden out-burst RFO
             * on the scoring machine, which is the standing suspect for the
             * ~4 us batched gap to the fused rivals.  ~0.1 s, unscored. */
#if defined(__AVX512VL__)
            if (!use_b) {
                const int dyf = (bestv == 11 || bestv == 12);
                pr_ph = l17r_time_fn(p, nv2, dyf ? exec_phdy_w8 : exec_ph_w8, 3);
                pr_xp = l17r_time_fn(p, nv2, exec_xp_w8, 3);
                pr_fu = l17r_time_fn(p, nv2, l17r_cand_a[dyf ? 11 : 2], 3);
                if (l17r_verbose())
                    fprintf(stderr, "[L17_rader tune] nv=%d  probe ph=%.3f "
                                    "xp=%.3f fu=%.3f us/vol\n",
                            nv2, pr_ph, pr_xp, pr_fu);
            }
#endif
        }
    }
    /* ice_r4: rank the fused-chain step variants, chain-shaped (one volume
     * iterated m steps, L2-resident -- exactly how fft3d_chain runs).  The
     * tuner input doubles as the c field: any well-scaled data times the
     * same, and the map contracts every state below 1 so nothing drifts
     * toward denormals.  ~40 ms, and it runs in the scoring window's quiet
     * like the rest of create(). */
    double ch_us = 0.0, nm_us = 0.0, ms_us = 0.0, msnm_us = 0.0;
    double ms6_us = 0.0, ms6nm_us = 0.0;
    int ch_v = 0, ms_ok = 0, v6_ok = 0;
#if defined(__AVX512VL__)
    p->chexec = l17r_chcand[0];
    p->cheager = l17r_cheager[0];
    if (l17r_tune_alloc(p, 1)) {   /* no-op when the tuner arena exists */
        const double *cx0 = (const double *)p->tin;
        double *cout = (double *)p->tout;
        const int mrace = 12;
        double chb[L17R_NCH];
        for (int v = 0; v < L17R_NCH; ++v) chb[v] = 1e30;
        for (int pass = 0; pass < 2; ++pass)   /* two sweeps: order bias */
            for (int v = 0; v < L17R_NCH; ++v) {
                double w0 = l17r_now();
                do
                    l17r_chain_vol(p, l17r_chcand[v], l17r_cheager[v],
                                   cx0, cx0, cout, 4);
                while (l17r_now() - w0 < 1.5e-3);
                for (int r = 0; r < 3; ++r) {
                    double t0 = l17r_now();
                    l17r_chain_vol(p, l17r_chcand[v], l17r_cheager[v],
                                   cx0, cx0, cout, mrace);
                    double dt = l17r_now() - t0;
                    if (dt < chb[v]) chb[v] = dt;
                }
            }
        for (int v = 1; v < L17R_NCH; ++v)
            if (chb[v] < chb[ch_v]) ch_v = v;
#if defined(L17R_FORCE_CH)
        ch_v = (L17R_FORCE_CH) % L17R_NCH;   /* dev A/B: force a chain pick */
#endif
        p->chexec = l17r_chcand[ch_v];
        p->cheager = l17r_cheager[ch_v];
        ch_us = chb[ch_v] * 1e6 / mrace;

        /* ice_r5: the ported msr engine races the xk-family winner.  FIRST
         * a numerical self-check against the tuned exec (different bit
         * class, same mathematics: one FFT-only msr step vs one plain
         * execute on the same volume must agree to ~1e-15 rel L2; a
         * transcription bug in the port shows as a large error and the
         * candidate is excluded -- a fast wrong chain scores nothing). */
        {
            l17r_chain_ms_vol(p, cx0, cx0, p->mv, 1, 0);
            int sb2 = p->batch;
            p->batch = 1;
            p->exec(p, p->tin, p->tout);
            p->batch = sb2;
            const double *ma = p->mv, *mb = (const double *)p->tout;
            double se = 0.0, sn = 0.0;
            for (long i = 0; i < 2 * (long)NVOL; ++i) {
                double d = ma[i] - mb[i];
                se += d * d;
                sn += mb[i] * mb[i];
            }
            ms_ok = (sn > 0.0) && (se <= 1e-26 * sn);
        }
        if (ms_ok) {
            double msb = 1e30;
            for (int pass = 0; pass < 2; ++pass) {
                double w0 = l17r_now();
                do
                    l17r_chain_ms_vol(p, cx0, cx0, cout, 4, 1);
                while (l17r_now() - w0 < 1.5e-3);
                for (int r = 0; r < 3; ++r) {
                    double t0 = l17r_now();
                    l17r_chain_ms_vol(p, cx0, cx0, cout, mrace, 1);
                    double dt = l17r_now() - t0;
                    if (dt < msb) msb = dt;
                }
            }
            ms_us = msb * 1e6 / mrace;
            /* msr is the pick unless the incumbent beats it by >3% (the
             * ported engine is the better-scored one; a create-window
             * near-tie must not keep the slower incumbent).  Chain bits
             * differ across the two engines -- legal, the harness compares
             * the end state to numpy only (winograd ice_r4 precedent). */
            p->ch_ms = ms_us < 1.03 * ch_us;
            /* msr FFT residual (map off), same probe shape as nm below:
             * repeat a single out-of-place step, never feeding raw z back */
            double w0 = l17r_now();
            do
                l17r_chain_ms_vol(p, cx0, cx0, p->mv, 1, 0);
            while (l17r_now() - w0 < 1.5e-3);
            double bmn = 1e30;
            for (int r = 0; r < 3; ++r) {
                double t0 = l17r_now();
                for (int s = 0; s < mrace; ++s)
                    l17r_chain_ms_vol(p, cx0, cx0, p->mv, 1, 0);
                double dt = l17r_now() - t0;
                if (dt < bmn) bmn = dt;
            }
            msnm_us = bmn * 1e6 / mrace;
        }
#if defined(L17R_FORCE_MS)
        p->ch_ms = (L17R_FORCE_MS) && ms_ok;   /* dev A/B override */
#endif

        /* ice_r7: the ported ms6 engine (matrixsimd chain v6) races the
         * winner so far.  Self-check FIRST, against the already-gated msr
         * chain: a full m=3 chain (FFT + map each step; m=3 exercises the
         * arena re-entry through pass b and the pack path) must agree to
         * 1e-12 rel L2.  The two engines share the kernel arithmetic but
         * differ in pass order and in the row-289 map site (msr maps it via
         * the exact 128-bit sqrt+div tail, ms6 through the rsqrt14 ladder),
         * so they are distinct rounding classes ~1e-15 apart; a
         * transcription bug would read ~1e0 and the candidate stays out --
         * a fast wrong chain scores nothing (r5 gate pattern). */
        if (ms_ok) {
            l17r_chain_ms_vol(p, cx0, cx0, p->mv, 3, 1);
            l17r_chain_v6(p, cx0, cx0, p->sc0, 3, 1, 1);
            const double *ma = p->sc0, *mb = p->mv;
            double se = 0.0, sn = 0.0;
            for (long i = 0; i < 2 * (long)NVOL; ++i) {
                double d = ma[i] - mb[i];
                se += d * d;
                sn += mb[i] * mb[i];
            }
            v6_ok = (sn > 0.0) && (se <= 1e-24 * sn);
        }
        if (v6_ok) {
            double m6b = 1e30;
            for (int pass = 0; pass < 2; ++pass) {
                double w0 = l17r_now();
                do
                    l17r_chain_v6(p, cx0, cx0, (double *)cout, 4, 1, 1);
                while (l17r_now() - w0 < 1.5e-3);
                for (int r = 0; r < 3; ++r) {
                    double t0 = l17r_now();
                    l17r_chain_v6(p, cx0, cx0, (double *)cout, mrace, 1, 1);
                    double dt = l17r_now() - t0;
                    if (dt < m6b) m6b = dt;
                }
            }
            ms6_us = m6b * 1e6 / mrace;
            /* ms6 is the pick unless the winner so far beats it by >15%.
             * The band is deliberately wide: same-lease alternating-binary
             * pairs at the driver's steady state read ms6 11.92-11.97 vs
             * msr 12.26-12.36 (7/7 wins, -0.35 us), while the create-time
             * race on a colder core read ms6 2.9% BEHIND msr in the same
             * process -- create races are ranks-not-levels (r5 lesson) and
             * here even the rank is cold-biased (the v6 arenas are
             * first-touched moments earlier; msr's buffers are hot from
             * the whole earlier tuner).  The donor ships v6 with no race
             * at all; the band only guards against a catastrophic
             * machine-side anomaly, with msr as the gate-verified
             * fallback. */
            p->ch_v6 = ms6_us < 1.15 * (p->ch_ms ? ms_us : ch_us);
            /* ms6 FFT residual (map off), one m=12 in-arena call so the
             * unpack/pack is amortized as in the real chain */
            double w0 = l17r_now();
            do
                l17r_chain_v6(p, cx0, cx0, p->sc0, 4, 1, 0);
            while (l17r_now() - w0 < 1.5e-3);
            double bmn = 1e30;
            for (int r = 0; r < 3; ++r) {
                double t0 = l17r_now();
                l17r_chain_v6(p, cx0, cx0, p->sc0, mrace, 1, 0);
                double dt = l17r_now() - t0;
                if (dt < bmn) bmn = dt;
            }
            ms6nm_us = bmn * 1e6 / mrace;
        }
#if defined(L17R_FORCE_V6)
        p->ch_v6 = (L17R_FORCE_V6) && v6_ok;   /* dev A/B override */
#endif

        /* unscored probe: the same step with the map OFF (domap=0 through
         * chain_mp = one plain FFT volume step) -- ch_us - nm_us is the
         * map's real embedded cost on the scoring machine, published in the
         * description so the next round knows the remaining headroom. */
        {
            double w0 = l17r_now();
            do
                chain_mp_w8(p, cx0, cx0, p->sc0, 0);
            while (l17r_now() - w0 < 1.5e-3);
            double bnm = 1e30;
            for (int r = 0; r < 3; ++r) {
                double t0 = l17r_now();
                for (int s = 0; s < mrace; ++s)
                    chain_mp_w8(p, cx0, cx0, p->sc0, 0);
                double dt = l17r_now() - t0;
                if (dt < bnm) bnm = dt;
            }
            nm_us = bnm * 1e6 / mrace;
        }
        if (l17r_verbose()) {
            for (int v = 0; v < L17R_NCH; ++v)
                fprintf(stderr, "[L17_rader tune] chain %-7s %8.3f us/step%s\n",
                        l17r_chtag[v], chb[v] * 1e6 / mrace,
                        (!p->ch_ms && v == ch_v) ? "  <== kept" : "");
            fprintf(stderr, "[L17_rader tune] chain msr     %8.3f us/step"
                            " (ok=%d, msnm=%.3f)%s\n", ms_us, ms_ok, msnm_us,
                    (p->ch_ms && !p->ch_v6) ? "  <== kept" : "");
            fprintf(stderr, "[L17_rader tune] chain ms6     %8.3f us/step"
                            " (ok=%d, ms6nm=%.3f)%s\n", ms6_us, v6_ok,
                    ms6nm_us, p->ch_v6 ? "  <== kept" : "");
            fprintf(stderr, "[L17_rader tune] chain nm (map off) %8.3f us/step"
                            "  -> map cost %.3f\n", nm_us, ch_us - nm_us);
        }
    }
#endif
    l17r_tune_free(p);

#if defined(L17R_FORCE_PF)
    p->pf = (L17R_FORCE_PF);
#endif
#if defined(L17R_FORCE_PFW)
    p->pfw = (L17R_FORCE_PFW);
#endif

    /* measured sustained clock at both widths (unscored; VERDICT r4's ask) */
    double g256 = l17r_probe_ghz(0);
    double g512 = l17r_probe_ghz(1);

    {
        int n = snprintf(g_desc, sizeof g_desc,
                 "Rader-17 cyclic/negacyclic (kernel from L17_winograd), "
                 "plane-fused; tuned: %s, pf=%d, pfw=%d, clk256=%.2f clk512=%.2f",
                 tags[bestv], p->pf, p->pfw, g256, g512);
        if (pr_xb > 0.0 && n > 0 && (size_t)n < sizeof g_desc)
            n += snprintf(g_desc + n, sizeof g_desc - (size_t)n,
                          ", xrace xl/xfs=%.2f/%.2f", pr_xa, pr_xb);
        if (pr_fu > 0.0 && n > 0 && (size_t)n < sizeof g_desc)
            n += snprintf(g_desc + n, sizeof g_desc - (size_t)n,
                          ", probe ph/xp/fu=%.2f/%.2f/%.2f us/vol",
                          pr_ph, pr_xp, pr_fu);
#if defined(__AVX512VL__)
        if (ch_us > 0.0 && n > 0 && (size_t)n < sizeof g_desc)
            snprintf(g_desc + n, sizeof g_desc - (size_t)n,
                     ", chain ch=%s %.2f nm=%.2f msr=%.2f msnm=%.2f msok=%d"
                     " ms6=%.2f ms6nm=%.2f ok6=%d us/step",
                     p->ch_v6 ? "ms6" : (p->ch_ms ? "msr" : l17r_chtag[ch_v]),
                     p->ch_v6 ? ms6_us : (p->ch_ms ? ms_us : ch_us),
                     nm_us, ms_us, msnm_us, ms_ok, ms6_us, ms6nm_us, v6_ok);
#else
        (void)ch_us;
        (void)nm_us;
        (void)ch_v;
        (void)ms_us;
        (void)msnm_us;
        (void)ms_ok;
        (void)ms6_us;
        (void)ms6nm_us;
        (void)v6_ok;
#endif
    }
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    l17r_tune_free(p);   /* normally already freed at the end of create() */
    free(p->mem);
    free(p);
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    p->exec(p, in, out);
}

/* ice_r4: the fused-chain entry point (weak-detected by the driver).  Owns
 * the whole m-step graded chain:
 *     state <- (FFT(state) + c) / (1 + |FFT(state) + c|),  state_0 = x0,
 * final state -> final_out.  Structure: per-VOLUME iteration (each volume's
 * chain is independent, so one volume runs all m steps while its state
 * ping-pong (2 x 78.6 KB), its c volume and the A scratch all sit in the
 * 1.25 MB L2 -- the fallback path sweeps volumes per step and lives in L3),
 * lazy map (the buffer stays raw between steps; the map is applied in the
 * next step's first pass and materialized once at the end).  Step 1 reads
 * x0 directly (x0 is already a state: no map).  x0 and c are never written. */
void fft3d_chain(fft3d_plan *p, const double _Complex *x0,
                 const double _Complex *c, double _Complex *final_out, int m)
{
    if (m < 1) {
        memcpy(final_out, x0, (size_t)NVOL * (size_t)p->batch * sizeof *x0);
        return;
    }
#if defined(__AVX512VL__)
    if (p->ch_v6) {
        /* ice_r7: the ported ms6 engine (matrixsimd chain v6) -- padded
         * in-place strided passes, state in a private arena for the whole
         * volume-chain; x0 and c never written */
        l17r_chain_v6(p, (const double *)x0, (const double *)c,
                      (double *)final_out, m, p->batch, 1);
        return;
    }
    if (p->ch_ms) {
        /* ice_r5: the ported msr engine, volume-major and IN PLACE on
         * final_out (X-first makes in-place legal; x0 and c never written) */
        for (int b = 0; b < p->batch; ++b) {
            const size_t off = (size_t)2 * NVOL * (size_t)b;
            l17r_chain_ms_vol(p, (const double *)x0 + off,
                              (const double *)c + off,
                              (double *)final_out + off, m, 1);
        }
        return;
    }
    for (int b = 0; b < p->batch; ++b) {
        const size_t off = (size_t)2 * NVOL * (size_t)b;
        l17r_chain_vol(p, p->chexec, p->cheager, (const double *)x0 + off,
                       (const double *)c + off, (double *)final_out + off, m);
    }
#else
    /* portable fallback (non-AVX512 dev hosts): still per-volume for the L2
     * locality, through the tuned exec + the scalar map into mv. */
    int sb = p->batch;
    p->batch = 1;
    for (int b = 0; b < sb; ++b) {
        const double *xb = (const double *)x0 + (size_t)2*NVOL*b;
        const double *cb = (const double *)c  + (size_t)2*NVOL*b;
        double *ob       = (double *)final_out + (size_t)2*NVOL*b;
        const double *cur = xb;
        for (int s = 0; s < m; ++s) {
            double *d = (s & 1) ? p->sc1 : p->sc0;
            const double *fin = cur;
            if (s > 0) {
                l17r_map_run(cur, cb, p->mv, NVOL);
                fin = p->mv;
            }
            p->exec(p, (const double _Complex *)fin, (double _Complex *)d);
            cur = d;
        }
        l17r_map_run(cur, cb, ob, NVOL);
    }
    p->batch = sb;
#endif
}

#else  /* L17R_TEMPLATE_PASS ============================================== */
/* ===========================================================================
 * TEMPLATE BODY -- instantiated once per vector width.
 * Inputs: VW (doubles per vector), SFX(x) (name mangler).
 * =========================================================================== */

#define TR  (((LN  + VW - 1) / VW) * VW)   /* plane row stride:  20 / 24  */
#define PS  (((NPL + VW - 1) / VW) * VW)   /* volume plane pitch: 292 / 296 */

typedef double    SFX(vdt) __attribute__((vector_size(VW * 8), aligned(8)));
typedef long long SFX(vlt) __attribute__((vector_size(VW * 8)));
#define vd SFX(vdt)
#define vl SFX(vlt)

#if defined(__clang__)
#  if VW == 8
#    define ILO(r,i) __builtin_shufflevector((r),(i),0,8,1,9,2,10,3,11)
#    define IHI(r,i) __builtin_shufflevector((r),(i),4,12,5,13,6,14,7,15)
#    define DLE(a,b) __builtin_shufflevector((a),(b),0,2,4,6,8,10,12,14)
#    define DLO(a,b) __builtin_shufflevector((a),(b),1,3,5,7,9,11,13,15)
#  else
#    define ILO(r,i) __builtin_shufflevector((r),(i),0,4,1,5)
#    define IHI(r,i) __builtin_shufflevector((r),(i),2,6,3,7)
#    define DLE(a,b) __builtin_shufflevector((a),(b),0,2,4,6)
#    define DLO(a,b) __builtin_shufflevector((a),(b),1,3,5,7)
#  endif
#else
#  if VW == 8
#    define ILO(r,i) __builtin_shuffle((r),(i),(vl){0,8,1,9,2,10,3,11})
#    define IHI(r,i) __builtin_shuffle((r),(i),(vl){4,12,5,13,6,14,7,15})
#    define DLE(a,b) __builtin_shuffle((a),(b),(vl){0,2,4,6,8,10,12,14})
#    define DLO(a,b) __builtin_shuffle((a),(b),(vl){1,3,5,7,9,11,13,15})
#  else
#    define ILO(r,i) __builtin_shuffle((r),(i),(vl){0,4,1,5})
#    define IHI(r,i) __builtin_shuffle((r),(i),(vl){2,6,3,7})
#    define DLE(a,b) __builtin_shuffle((a),(b),(vl){0,2,4,6})
#    define DLO(a,b) __builtin_shuffle((a),(b),(vl){1,3,5,7})
#  endif
#endif

#define VL(p)    (*(const vd *)(const void *)(p))
#define VS(p,x)  (*(vd *)(void *)(p) = (x))

/* Width-selected plane movement (panel_r8): the w8 pipeline uses the 8x8 zmm
 * transpose/deinterleave blocks, the w4 pipeline keeps the 4x4 ymm tiles.
 * Pure data movement either way -- identical values to identical places, so
 * the bit classes are untouched. */
#if VW == 8
#  define TP17(s,ss,d,ds)      transpose17z((s),(ss),(d),(ds))
#  define TP17P(s,ss,d,ds,pt)  transpose17z_part((s),(ss),(d),(ds),(pt))
#  define DT17(s,dr,di,ds)     deint_transpose17z((s),(dr),(di),(ds))
#  define DT17P(s,dr,di,ds,pt) deint_transpose17z_part((s),(dr),(di),(ds),(pt))
#else
#  define TP17(s,ss,d,ds)      transpose17((s),(ss),(d),(ds))
#  define TP17P(s,ss,d,ds,pt)  transpose17_part((s),(ss),(d),(ds),(pt))
#  define DT17(s,dr,di,ds)     deint_transpose17((s),(dr),(di),(ds))
#  define DT17P(s,dr,di,ds,pt) deint_transpose17_part((s),(dr),(di),(ds),(pt))
#endif

/* ymm deinterleave inside the w8 pipeline ("dy" candidates, panel_r9).  The
 * deint is the ONE movement stage whose source loads are unaligned AND cold
 * at batch (`in` streams from DRAM there); its 64-byte loads land on the
 * 16-byte alignment classes of an odd-length complex plane and split a cache
 * line 3 times in 4, against ~1 in 2 for the 4x4 ymm tile's 32-byte loads.
 * panel_r8 measured the zmm transposes -3.0% at B=1/B=8 but +1.7%/+1.4% at
 * B=256/2048 on the node -- a batch-only regression, which is exactly the
 * cold-split-load signature (T/U/A accesses are 64-B aligned and L1-hot at
 * every batch; the deint source is the only unaligned zmm stream r8 added).
 * "dy" keeps every zmm transpose and swaps ONLY the deint back to the ymm
 * tile.  Same values to the same places: bit-identical, tuner-ranked. */
#if VW == 8
#  define DT17Y(s,dr,di,ds)   deint_transpose17((s),(dr),(di),(ds))
#else
#  define DT17Y(s,dr,di,ds)   DT17(s,dr,di,ds)
#endif

/* "ty" (ice_r1): ymm 4x4-tile transposes inside the w8 pipeline, selected by
 * dey == 2 (which also selects the ymm deint via DT17Y's truthiness).  Ice
 * Lake motivation: every 512-bit two-source shuffle issues on port 5 ONLY,
 * while the 4x4 tile's 256-bit shuffles dual-issue on p1/p5 -- so the 8x8
 * zmm block's 24 shuffles serialize at 24 cycles where the 4x4 tiling's ~48
 * spread over two ports at ~24.  On ICX the two 512-bit FMA pipes also
 * halve the kernel-drain shadow that used to hide the p5 burst (the r5 "ov"
 * lesson in reverse).  Pure data movement, identical values to identical
 * places: every class-A candidate stays bit-identical, the tuner ranks it. */
#if VW == 8
#  define TP17D(s,ss,d,ds,deyf) do {                                          \
        if ((deyf) == 2) transpose17((s),(ss),(d),(ds));                      \
        else             transpose17z((s),(ss),(d),(ds));                     \
    } while (0)
#else
#  define TP17D(s,ss,d,ds,deyf) transpose17((s),(ss),(d),(ds))
#endif

/* width-matched rsqrt14 seed for the in-kernel mapped store ("xk", ice_r4);
 * on a non-AVX512VL host the identity keeps the dead branch compiling (mst
 * is never set there). */
#if defined(__AVX512VL__)
#  if VW == 8
#    define VRSQ14(x) ((vd)_mm512_rsqrt14_pd((__m512d)(x)))
#  else
#    define VRSQ14(x) ((vd)_mm256_rsqrt14_pd((__m256d)(x)))
#  endif
#else
#  define VRSQ14(x) (x)
#endif

/* KPIN(c): force a broadcast constant into a register with an empty asm
 * barrier.  Without it gcc folds every constant into an FMA memory operand
 * and materialises a separate negated .LC copy for each -= (it canonicalises
 * a -= b*c into a += b*(-c)): ~148 constant-load uops per kernel
 * instantiation.  MEASURED on wallaby round panel_r2 (3 load ports): pinning
 * LOSES 5% there; a 2-load-port Cascade Lake may disagree, so both kernels
 * are built and the plan-time tuner decides. */
#if defined(__AVX512VL__)
#  define KPIN(c) ({ vd _t = ((vd){0} + (c)); __asm__("" : "+v"(_t)); _t; })
#else
#  define KPIN(c) ((vd){0} + (c))
#endif

/* -------------------------------------------------------------- the kernel */

/* VW independent 17-point DFTs.
 *   load  mode lmode 0 : split inputs at xr/xi + k*xs, k = 0..16
 *   load  mode lmode 1 : interleaved inputs at isrc + 2*(k*NPL + im0)
 *                        (deinterleaving shuffle pair per input -- the
 *                        X-first x pass reads the caller's `in` directly)
 *   store mode 0 : split store at orr/oii + k*os
 *   store mode 1 : interleaved vector store at dst + 2*(k*os + m0)
 * `pin` (compile-time 0/1) selects the pinned-S-constant variant.
 * `mst` (compile-time 0/1, ice_r4, mode 1 only) fuses the graded chain's map
 * into the interleaving store: w = (vr,vi) + c (c split from `cm` at the
 * destination offsets), the rsqrt14+2-Newton+vdivpd ladder on the split
 * registers already in hand, mapped value stored instead of raw -- the
 * ladder's ops interleave with this kernel's own FMA drain at instruction
 * granularity (corpus sec 10.2's in-kernel fusion).  Elementwise-identical
 * arithmetic to l17r_map8/l17r_map4, so xk chain candidates stay in the one
 * chain bit class (cmp-verified).
 * Every input is loaded before any output is stored, so lmode 0 / mode 0 may
 * be in place. */
static inline __attribute__((always_inline)) void SFX(wino17)(
        const double *xr, const double *xi, long xs,
        const double *isrc, long im0,
        double *orr, double *oii,
        double *dst, long m0,
        long os, const int lmode, const int mode, const int pin,
        const double *cm, const int mst)
{
#define LDP(k, vr, vi) do {                                                    \
        if (lmode == 0) {                                                      \
            (vr) = VL(xr + (long)(k)*xs);  (vi) = VL(xi + (long)(k)*xs);       \
        } else {                                                               \
            const double *_q = isrc + 2*((long)(k)*NPL + im0);                 \
            vd _a = VL(_q), _b = VL(_q + VW);                                  \
            (vr) = DLE(_a,_b);  (vi) = DLO(_a,_b);                             \
        }                                                                      \
    } while (0)

#define ST(k, vr, vi) do {                                                     \
        if (mode == 0) {                                                       \
            VS(orr + (long)(k)*os, (vr));  VS(oii + (long)(k)*os, (vi));       \
        } else if (!mst) {                                                     \
            double *_p = dst + 2*((long)(k)*os + m0);                          \
            *(vd *)(void *)(_p)      = ILO((vr),(vi));                         \
            *(vd *)(void *)(_p + VW) = IHI((vr),(vi));                         \
        } else {                                                               \
            double *_p = dst + 2*((long)(k)*os + m0);                          \
            const double *_c = cm + 2*((long)(k)*os + m0);                     \
            vd _ca = VL(_c), _cb = VL(_c + VW);                                \
            vd _wr = (vr) + DLE(_ca,_cb), _wi = (vi) + DLO(_ca,_cb);           \
            vd _s = ((vd){0} + 1e-300);                                        \
            _s += _wr*_wr;                                                     \
            _s += _wi*_wi;                                                     \
            vd _q = VRSQ14(_s);                                                \
            vd _h = ((vd){0} + 0.5) * _s;                                      \
            _q = _q * (((vd){0} + 1.5) - _h*_q*_q);                            \
            _q = _q * (((vd){0} + 1.5) - _h*_q*_q);                            \
            vd _d = ((vd){0} + 1.0) + _s*_q;                                   \
            vd _r = ((vd){0} + 1.0) / _d;                                      \
            vd _mr = _wr*_r, _mi = _wi*_r;                                     \
            *(vd *)(void *)(_p)      = ILO(_mr,_mi);                           \
            *(vd *)(void *)(_p + VW) = IHI(_mr,_mi);                           \
        }                                                                      \
    } while (0)

    vd vvr[8], vvi[8];       /* V_m = sigma[m] * (x_folded[m] - x_(17-folded[m])) */
    vd ccr[8], cci[8];       /* C_n: cyclic-8 correlation of U with c, x0-seeded  */
    vd a0r,a1r,a2r,a3r, a0i,a1i,a2i,a3i;
    vd b0r,b1r,b2r,b3r, b0i,b1i,b2i,b3i;

    /* Fold-stage constants are never pinned: that stage has 16 live
     * accumulators, and pinning its 8 constants too pushes past 32 registers
     * (round panel_r2, measured: 361 stack moves, slower than either
     * alternative). */
    const vd kCP0 = ((vd){0} + CP0), kCP1 = ((vd){0} + CP1),
             kCP2 = ((vd){0} + CP2), kCP3 = ((vd){0} + CP3);
    const vd kCM0 = ((vd){0} + CM0), kCM1 = ((vd){0} + CM1),
             kCM2 = ((vd){0} + CM2), kCM3 = ((vd){0} + CM3);

    vd x0r, x0i;
    LDP(0, x0r, x0i);
    vd dcr = x0r, dci = x0i;

    /* Fold stage: block t serves quotient slots m = t and m = t+4.
     *   vv[t] = x[j0]-x[k0], vv[t+4] = x[j1]-x[k1]   (sigma folded into order)
     *   P_t = U_t + U_(t+4) -> cyclic-4 accumulators a (x0-seeded)
     *   Q_t = U_t - U_(t+4) -> negacyclic-4 accumulators b                    */
#define FOLDCOM(t, j0,k0, j1,k1)                                               \
        vd e0r, e0i, f0r, f0i, e1r, e1i, f1r, f1i;                             \
        LDP(j0, e0r, e0i);  LDP(k0, f0r, f0i);                                 \
        LDP(j1, e1r, e1i);  LDP(k1, f1r, f1i);                                 \
        vvr[t]     = e0r - f0r;  vvi[t]     = e0i - f0i;                       \
        vvr[(t)+4] = e1r - f1r;  vvi[(t)+4] = e1i - f1i;                       \
        vd u0r = e0r + f0r, u0i = e0i + f0i;                                   \
        vd u1r = e1r + f1r, u1i = e1i + f1i;                                   \
        vd pr = u0r + u1r, pi = u0i + u1i;                                     \
        vd qr = u0r - u1r, qi = u0i - u1i;                                     \
        dcr += pr;  dci += pi;

#define FOLD0(t, j0,k0, j1,k1, w0,w1,w2,w3, z0,z1,z2,z3) do {                  \
        FOLDCOM(t, j0,k0, j1,k1)                                               \
        a0r = x0r + pr*(w0);  a1r = x0r + pr*(w1);                             \
        a2r = x0r + pr*(w2);  a3r = x0r + pr*(w3);                             \
        a0i = x0i + pi*(w0);  a1i = x0i + pi*(w1);                             \
        a2i = x0i + pi*(w2);  a3i = x0i + pi*(w3);                             \
        b0r = qr*(z0);  b1r = qr*(z1);  b2r = qr*(z2);  b3r = qr*(z3);         \
        b0i = qi*(z0);  b1i = qi*(z1);  b2i = qi*(z2);  b3i = qi*(z3);         \
    } while (0)

/* sN are += / -= tokens: negacyclic wrap terms use -= with the POSITIVE
 * constant so gcc emits vfnmadd against the shared .LC slot. */
#define FOLDN(t, j0,k0, j1,k1, w0,w1,w2,w3, s0,z0, s1,z1, s2,z2, s3,z3) do {   \
        FOLDCOM(t, j0,k0, j1,k1)                                               \
        a0r += pr*(w0);  a1r += pr*(w1);  a2r += pr*(w2);  a3r += pr*(w3);     \
        a0i += pi*(w0);  a1i += pi*(w1);  a2i += pi*(w2);  a3i += pi*(w3);     \
        b0r s0 qr*(z0);  b1r s1 qr*(z1);  b2r s2 qr*(z2);  b3r s3 qr*(z3);     \
        b0i s0 qi*(z0);  b1i s1 qi*(z1);  b2i s2 qi*(z2);  b3i s3 qi*(z3);     \
    } while (0)

    /* folded = [1,3,8,7,4,5,2,6], sigma = [+,+,-,-,-,+,-,-]; the subtraction
     * order below realises sigma, and the row rotations realise the
     * (nega)circulant structure. */
    FOLD0(0,  1,16, 13, 4,  kCP0,kCP1,kCP2,kCP3,  kCM0, kCM1, kCM2, kCM3);
    FOLDN(1,  3,14,  5,12,  kCP1,kCP2,kCP3,kCP0,  +=,kCM1, +=,kCM2, +=,kCM3, -=,kCM0);
    FOLDN(2,  9, 8, 15, 2,  kCP2,kCP3,kCP0,kCP1,  +=,kCM2, +=,kCM3, -=,kCM0, -=,kCM1);
    FOLDN(3, 10, 7, 11, 6,  kCP3,kCP0,kCP1,kCP2,  +=,kCM3, -=,kCM0, -=,kCM1, -=,kCM2);
#undef FOLD0
#undef FOLDN
#undef FOLDCOM

    ST(0, dcr, dci);                       /* X[0] = x0 + sum P_t */

    ccr[0] = a0r + b0r;  cci[0] = a0i + b0i;
    ccr[4] = a0r - b0r;  cci[4] = a0i - b0i;
    ccr[1] = a1r + b1r;  cci[1] = a1i + b1i;
    ccr[5] = a1r - b1r;  cci[5] = a1i - b1i;
    ccr[2] = a2r + b2r;  cci[2] = a2i + b2i;
    ccr[6] = a2r - b2r;  cci[6] = a2i - b2i;
    ccr[3] = a3r + b3r;  cci[3] = a3i + b3i;
    ccr[7] = a3r - b3r;  cci[7] = a3i - b3i;

    /* Dense negacyclic-8 on V, four outputs at a time:
     *   Stilde[n] = sum_m SN[(m+n) mod 8] * (-1)^floor((m+n)/8) * V_m
     * The wrapped (negated) terms are written as -= with the POSITIVE
     * constant, so gcc emits vfnmadd against the same .LC slot instead of
     * materialising 8 negated duplicates (measured: without this gcc emits
     * zero vfnmadd and doubles the constant footprint).                       */
#define SACC1(m, w0,w1,w2,w3) do {                                             \
        vd tr_ = vvr[m], ti_ = vvi[m];                                         \
        a0r = tr_*(w0);  a1r = tr_*(w1);  a2r = tr_*(w2);  a3r = tr_*(w3);     \
        a0i = ti_*(w0);  a1i = ti_*(w1);  a2i = ti_*(w2);  a3i = ti_*(w3);     \
    } while (0)
#define SACC(m, s0,w0, s1,w1, s2,w2, s3,w3) do {                               \
        vd tr_ = vvr[m], ti_ = vvi[m];                                         \
        a0r s0 tr_*(w0); a1r s1 tr_*(w1); a2r s2 tr_*(w2); a3r s3 tr_*(w3);    \
        a0i s0 ti_*(w0); a1i s1 ti_*(w1); a2i s2 ti_*(w2); a3i s3 ti_*(w3);    \
    } while (0)

    const vd kSN0 = pin ? KPIN(SN0) : ((vd){0} + SN0);
    const vd kSN1 = pin ? KPIN(SN1) : ((vd){0} + SN1);
    const vd kSN2 = pin ? KPIN(SN2) : ((vd){0} + SN2);
    const vd kSN3 = pin ? KPIN(SN3) : ((vd){0} + SN3);
    const vd kSN4 = pin ? KPIN(SN4) : ((vd){0} + SN4);
    const vd kSN5 = pin ? KPIN(SN5) : ((vd){0} + SN5);
    const vd kSN6 = pin ? KPIN(SN6) : ((vd){0} + SN6);
    const vd kSN7 = pin ? KPIN(SN7) : ((vd){0} + SN7);

    /* first half: n = 0..3 */
    SACC1(0,     kSN0,    kSN1,    kSN2,    kSN3);
    SACC(1,  +=, kSN1, +=,kSN2, +=,kSN3, +=,kSN4);
    SACC(2,  +=, kSN2, +=,kSN3, +=,kSN4, +=,kSN5);
    SACC(3,  +=, kSN3, +=,kSN4, +=,kSN5, +=,kSN6);
    SACC(4,  +=, kSN4, +=,kSN5, +=,kSN6, +=,kSN7);
    SACC(5,  +=, kSN5, +=,kSN6, +=,kSN7, -=,kSN0);
    SACC(6,  +=, kSN6, +=,kSN7, -=,kSN0, -=,kSN1);
    SACC(7,  +=, kSN7, -=,kSN0, -=,kSN1, -=,kSN2);

    /* X[folded[n]] = C_n - i*sigma[n]*Stilde_n ; X[17-folded[n]] the conjugate
     * combine.  With split re/im the *(-i) is a rename + sign in the add. */
    { vd tr_ = ccr[0], ti_ = cci[0];
      ST( 1, tr_ + a0i, ti_ - a0r);  ST(16, tr_ - a0i, ti_ + a0r); }
    { vd tr_ = ccr[1], ti_ = cci[1];
      ST( 3, tr_ + a1i, ti_ - a1r);  ST(14, tr_ - a1i, ti_ + a1r); }
    { vd tr_ = ccr[2], ti_ = cci[2];
      ST( 8, tr_ - a2i, ti_ + a2r);  ST( 9, tr_ + a2i, ti_ - a2r); }
    { vd tr_ = ccr[3], ti_ = cci[3];
      ST( 7, tr_ - a3i, ti_ + a3r);  ST(10, tr_ + a3i, ti_ - a3r); }

    /* second half: n = 4..7 */
    SACC1(0,     kSN4,    kSN5,    kSN6,    kSN7);
    SACC(1,  +=, kSN5, +=,kSN6, +=,kSN7, -=,kSN0);
    SACC(2,  +=, kSN6, +=,kSN7, -=,kSN0, -=,kSN1);
    SACC(3,  +=, kSN7, -=,kSN0, -=,kSN1, -=,kSN2);
    SACC(4,  -=, kSN0, -=,kSN1, -=,kSN2, -=,kSN3);
    SACC(5,  -=, kSN1, -=,kSN2, -=,kSN3, -=,kSN4);
    SACC(6,  -=, kSN2, -=,kSN3, -=,kSN4, -=,kSN5);
    SACC(7,  -=, kSN3, -=,kSN4, -=,kSN5, -=,kSN6);
#undef SACC
#undef SACC1

    { vd tr_ = ccr[4], ti_ = cci[4];
      ST( 4, tr_ - a0i, ti_ + a0r);  ST(13, tr_ + a0i, ti_ - a0r); }
    { vd tr_ = ccr[5], ti_ = cci[5];
      ST( 5, tr_ + a1i, ti_ - a1r);  ST(12, tr_ - a1i, ti_ + a1r); }
    { vd tr_ = ccr[6], ti_ = cci[6];
      ST( 2, tr_ - a2i, ti_ + a2r);  ST(15, tr_ + a2i, ti_ - a2r); }
    { vd tr_ = ccr[7], ti_ = cci[7];
      ST( 6, tr_ - a3i, ti_ + a3r);  ST(11, tr_ + a3i, ti_ - a3r); }
#undef ST
#undef LDP
}

/* ------------------------------------------------------------- the passes */

/* lane-block starts for a 17-wide lane space whose stores must not overrun
 * lane 16 -- the last block overlaps the previous one and recomputes a few
 * transforms, which costs nothing because the block count is unchanged. */
#define NLB ((LN + VW - 1) / VW)
static const int SFX(LBOFF)[NLB] = {
#if VW == 8
    0, 8, 9
#else
    0, 4, 8, 12, 13
#endif
};
#define NXB ((NPL + VW - 1) / VW)

/* Dense per-plane flush for the staged X-first y store (panel_r11).  The old
 * X-first y pass wrote DRAM-destined `out` through 17-row strided 128-B
 * partial-line pieces from inside the compute loop -- the exact store shape
 * the r10 VERDICT identified as the wrong baseline (both fused rivals write
 * finished planes densely; L17_matrixsimd r3 won -10.8% at B=256 with it).
 * Now the y pass stores into an L1-hot 4.6 KB staging plane and this routine
 * emits the finished plane as ONE sequential stream of full lines.  It runs
 * at the TOP of the NEXT plane's compute (staging double-buffered by kx
 * parity), so the RFO misses resolve under ~1.5 us of independent kernel
 * work -- the "stage densely, pace the flush under compute" construction the
 * monitor named, applied to the structure that has dense finished planes.
 * 2*NPL = 578 = VW*n + 2 at both widths. */
static inline __attribute__((always_inline)) void SFX(pl_flush)(
        const double *sp, double *dp)
{
    long q = 0;
    for (; q + VW <= 2L*NPL; q += VW)
        VS(dp + q, VL(sp + q));
    dp[q]     = sp[q];
    dp[q + 1] = sp[q + 1];
}

#if VW == 8
/* Write-intent prefetch of one x-pass block's 17 destination row regions in
 * `out` (panel_r6, mechanism from L8_fusedaxes/L36_pfa's node-winning pfw:
 * hide the RFO, don't avoid it).  A zmm block writes 128 B per row starting
 * at a 16-byte-aligned address, touching at most 3 cache lines; the ymm
 * tail writes 64 B (2 lines). */
static inline __attribute__((always_inline)) void SFX(pfw_rows)(
        double *dst, long m0t, const int tail)
{
    for (int k = 0; k < LN; ++k) {
        char *q = (char *)(dst + 2*((long)k*NPL + m0t));
        __builtin_prefetch(q, 1, 2);
        if (tail) {
            __builtin_prefetch(q + 63, 1, 2);
        } else {
            __builtin_prefetch(q + 64, 1, 2);
            __builtin_prefetch(q + 127, 1, 2);
        }
    }
}

/* One x-pass block (mixed shape: blk 0..35 are zmm at m0 = 8*blk, blk 36 is
 * the ymm tail at NPL-4), with optional paced prefetchw of block blk+2's
 * destination.  noinline+noclone: the sp pipeline calls this from three
 * sites, and a real call keeps ONE instantiation of the two kernels in L1i
 * (r2: 38 KB of exec body kills; the ~5-cycle call is noise against a
 * ~300-cycle block).  Arithmetic identical to exec_body's mixed x pass
 * (pin=0), so any exec built from this is bit-identical to "xl 512t". */
static __attribute__((noinline, noclone)) void SFX(xblk_run)(
        const double *ar, const double *ai, double *dst, int blk,
        const int pfw)
{
    if (pfw) {
        if (blk == 0) { SFX(pfw_rows)(dst, 0, 0); SFX(pfw_rows)(dst, 8, 0); }
        if (blk <= 33)      SFX(pfw_rows)(dst, 8L*(blk + 2), 0);
        else if (blk == 34) SFX(pfw_rows)(dst, NPL - 4, 1);
        /* blk 35's zmm region [280,288) covers the tail's [285,289) rows,
         * so the tail is fully prefetched by blk 33 and 34. */
    }
    if (blk < 36) {
        long m0 = 8L*blk;
        SFX(wino17)(ar + m0, ai + m0, PS, 0, 0, 0, 0, dst, m0,
                    NPL, 0, 1, 0, 0, 0);
    } else {
        wino17_w4(ar + (NPL - 4), ai + (NPL - 4), PS, 0, 0, 0, 0,
                  dst, NPL - 4, NPL, 0, 1, 0, 0, 0);
    }
}

/* Dense volume flush for the staged x pass ("st", panel_r10).  The scattered
 * x-pass store is 17 concurrent k-row streams of 16-B-aligned 128-B pieces:
 * at batch those are partial-line DRAM RFOs whose write-combining lines are
 * evicted half-filled (17 streams against ~12 fill buffers), i.e. the same
 * `out` line is fetched and written back more than once.  The st variants
 * point those stores at the L2-resident vo staging volume instead and emit
 * `out` as ONE sequential stream of full lines here -- prefetchable RFO,
 * perfect row locality, no partial-line waste; this is what the fused
 * rivals' dense chunk/plane stores do structurally (L17_matrixsimd r3,
 * L17_winograd g8).  The rolled loop is 1228 iterations, ~30 B of code.
 * pfw composes: one write-intent prefetch per stored line, 512 B ahead
 * (prefetch never faults, so running past the last volume's end is safe). */
static inline __attribute__((always_inline)) void SFX(vo_flush)(
        const double *vo, double *dst, const int pfw)
{
    long q = 0;
    if (pfw) {
        for (; q + 8 <= 2L*NVOL; q += 8) {
            __builtin_prefetch((const char *)(dst + q) + 512, 1, 2);
            VS(dst + q, VL(vo + q));
        }
    } else {
        for (; q + 8 <= 2L*NVOL; q += 8)
            VS(dst + q, VL(vo + q));
    }
    dst[q]     = vo[q];              /* 2*NVOL = 9826 = 8*1228 + 2 */
    dst[q + 1] = vo[q + 1];
}

/* One of 17 dense flush chunks (the "stp" pipelined flush): 576 doubles
 * (4.5 KB, 72 zmm) per plane slot, the 17th takes the 610-double remainder.
 * Same stream as vo_flush, cut so that volume b-1's flush rides under
 * volume b's plane phase one chunk per plane -- the wallaby A/B of the
 * immediate flush showed the serial burst fully exposed (no compute behind
 * it), which is exactly what pacing it under the FMA stream removes. */
static inline __attribute__((always_inline)) void SFX(vo_flush_chunk)(
        const double *vo, double *dst, const int x, const int pfw)
{
    long q = 576L*x;
    const long qe = (x == 16) ? 2L*NVOL : q + 576;
    if (pfw) {
        for (; q + 8 <= qe; q += 8) {
            __builtin_prefetch((const char *)(dst + q) + 512, 1, 2);
            VS(dst + q, VL(vo + q));
        }
    } else {
        for (; q + 8 <= qe; q += 8)
            VS(dst + q, VL(vo + q));
    }
    if (x == 16) { dst[q] = vo[q]; dst[q + 1] = vo[q + 1]; }
}
#endif

/* Compile-time flags:
 *   pin    -- S constants pinned in registers (EVEX only, tuner candidate)
 *   xfirst -- X-first pass order (bit class B, batch >= L17R_XF_CUT only)
 *   mixed  -- VW==8 only: 17-lane passes run 2 zmm blocks + 1 ymm tail
 *             (wino17_w4), and the x pass's clamped last block runs at ymm.
 *             On a 1-FMA-unit part the ymm tail halves the tail block's
 *             cycles; recomputed overlap lanes are bit-identical.
 *   dey    -- w8 pipeline with the ymm deint tile (see DT17Y above); pure
 *             data movement, bit-identical.
 *   ph     -- probe phase (plan-time only, NEVER a candidate): 0 = full,
 *             1 = plane phase only (reads `in`, fills A, never touches out),
 *             2 = x pass only (reads whatever A holds, writes out).  Used by
 *             fft3d_create()'s streaming decomposition probe (panel_r9,
 *             pattern from L36_pfa panel_r8: put the discriminating
 *             measurement in create() and route it out via description()).
 *   stg    -- VW==8 mixed only: the x pass stores into the vo staging volume
 *             and the volume is then flushed to `out` as one dense sequential
 *             stream (see vo_flush).  Same kernel operands, same values to
 *             the same final places: bit-identical, class A, tuner-ranked. */
static inline __attribute__((always_inline)) void SFX(exec_body)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out,
        const int pin, const int xfirst, const int mixed,
        const int dey, const int ph, const int stg)
{
    double *const ar = p->SFX(ar), *const ai = p->SFX(ai);
    double *const tr = p->SFX(tr), *const ti = p->SFX(ti);
    double *const ur = p->SFX(ur), *const ui = p->SFX(ui);
    (void)mixed;
    (void)stg;      /* w4 instantiation: the staged x pass is VW==8-only */

    for (int b = 0; b < p->batch; ++b) {
        const double *src = (const double *)in + (size_t)2*NVOL*b;
        double *dst       = (double *)out      + (size_t)2*NVOL*b;
        const double *nxt =
            (p->pf && b + 1 < p->batch) ? src + (size_t)2*NVOL : NULL;

        if (xfirst) {
            /* ---- x pass first: interleaved `in` -> split A[kx][y][z] ---- */
#if VW == 8
            if (mixed) {
#pragma GCC unroll 1
                for (long m0 = 0; m0 + VW <= NPL; m0 += VW)
                    SFX(wino17)(0, 0, 0, src, m0, ar + m0, ai + m0,
                                0, 0, PS, 1, 0, pin, 0, 0);
                wino17_w4(0, 0, 0, src, NPL - 4, ar + (NPL - 4), ai + (NPL - 4),
                          0, 0, PS, 1, 0, 0, 0, 0);
            } else
#endif
            for (int blk = 0; blk < NXB; ++blk) {
                long m0 = (long)blk*VW;
                if (m0 > NPL - VW) m0 = NPL - VW;
                SFX(wino17)(0, 0, 0, src, m0, ar + m0, ai + m0,
                            0, 0, PS, 1, 0, pin, 0, 0);
            }

            /* ---- finish one kx plane at a time (L17_matrixsimd panel_r3's
             * X-first reorder).  panel_r11: the y pass stores into an L1-hot
             * staging plane (vp0/vp1, kx parity); the PREVIOUS plane's
             * staged output is flushed to `out` as one dense sequential
             * 4.6 KB stream at the top of this plane's compute, and pfw
             * prefetches plane kx's out region one plane ahead of its own
             * flush.  Values and final addresses unchanged: bit class B is
             * preserved, all class-B candidates stay cmp-identical. ---- */
            double *const vp0 = p->vo_w8, *const vp1 = p->vo_w8 + 1024;
            const int pfw = p->pfw;
            for (int kx = 0; kx < LN; ++kx) {
                if (nxt) {
                    const double *pp = nxt + 2*(long)kx*NPL;
                    for (int q = 0; q < 2*NPL; q += 8)
                        __builtin_prefetch(pp + q, 0, 2);
                }
                if (pfw) {
                    char *q0 = (char *)(dst + 2*(long)kx*NPL);
                    for (long qq = 0; qq < 16L*NPL; qq += 64)
                        __builtin_prefetch(q0 + qq, 1, 2);
                    __builtin_prefetch(q0 + 16L*NPL - 1, 1, 2);
                }
                if (kx > 0)
                    SFX(pl_flush)((kx & 1) ? vp0 : vp1,
                                  dst + 2*(long)(kx - 1)*NPL);

                /* A[kx][y][z] -> T[z][y] */
                TP17D(ar + (long)kx*PS, LN, tr, TR, dey);
                TP17D(ai + (long)kx*PS, LN, ti, TR, dey);

                /* z pass, in place on T: axis stride TR, lanes = y */
#if VW == 8
                if (mixed) {
                    long mlim = 8;              /* opaque: keep the 2-trip   */
                    __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
                    for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                        SFX(wino17)(tr + o, ti + o, TR, 0, 0, tr + o, ti + o,
                                    0, 0, TR, 0, 0, pin, 0, 0);
                    wino17_w4(tr + 16, ti + 16, TR, 0, 0, tr + 16, ti + 16,
                              0, 0, TR, 0, 0, 0, 0, 0);
                } else
#endif
                for (int lb = 0; lb < TR/VW; ++lb) {
                    long o = (long)lb*VW;
                    SFX(wino17)(tr + o, ti + o, TR, 0, 0, tr + o, ti + o,
                                0, 0, TR, 0, 0, pin, 0, 0);
                }

                TP17D(tr, TR, ur, TR, dey); /* T[kz][y] -> U[y][kz] */
                TP17D(ti, TR, ui, TR, dey);

                /* y pass: axis stride TR, lanes = kz, interleaving store
                 * into the staging plane (row stride 17 complex, L1-hot) */
                double *dp = (kx & 1) ? vp1 : vp0;
#if VW == 8
                if (mixed) {
                    long mlim = 8;              /* opaque: keep the 2-trip   */
                    __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
                    for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                        SFX(wino17)(ur + o, ui + o, TR, 0, 0, 0, 0, dp, o,
                                    LN, 0, 1, pin, 0, 0);
                    wino17_w4(ur + 13, ui + 13, TR, 0, 0, 0, 0, dp, 13,
                              LN, 0, 1, 0, 0, 0);
                } else
#endif
                for (int lb = 0; lb < NLB; ++lb) {
                    long o = SFX(LBOFF)[lb];
                    SFX(wino17)(ur + o, ui + o, TR, 0, 0, 0, 0, dp, o,
                                LN, 0, 1, pin, 0, 0);
                }
            }
            SFX(pl_flush)(vp0, dst + 2L*(LN - 1)*NPL);   /* plane 16 (even) */
        } else {
            /* ---------------- X-last (round-1 order) ---------------- */
            if (ph != 2)
            for (int x = 0; x < LN; ++x) {
                /* pf=1: pull the NEXT VOLUME's plane x while this plane's
                 * two compute passes run (L17_winograd round 2: -4.4% at
                 * B=2048; a no-op at B=1) -- the DRAM-regime shape.
                 * pf=2 (ice_r1): pull the NEXT PLANE of THIS volume, ~0.9 us
                 * ahead -- sized for the graded chain, where every src plane
                 * is an L3 HIT whose ~60-cycle latency is all there is to
                 * hide (a whole-volume lookahead there only churns L2); at
                 * x=16 it pulls the next volume's plane 0.  Prefetches never
                 * change values: all bit classes untouched.  73 lines. */
                if (p->pf) {
                    const double *pp;
                    if (p->pf == 2)
                        pp = (x + 1 < LN) ? src + 2*((long)(x + 1)*NPL) : nxt;
                    else
                        pp = nxt ? nxt + 2*(long)x*NPL : NULL;
                    if (pp)
                        for (int q = 0; q < 2*NPL; q += 8)
                            __builtin_prefetch(pp + q, 0, 2);
                }

                /* in[x][y][z] -> T[z][y] */
                if (dey) DT17Y(src + 2*(long)x*NPL, tr, ti, TR);
                else     DT17(src + 2*(long)x*NPL, tr, ti, TR);

                /* z pass, in place on T: axis stride TR, lanes = y */
#if VW == 8
                if (mixed) {
                    long mlim = 8;              /* opaque: keep the 2-trip   */
                    __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
                    for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                        SFX(wino17)(tr + o, ti + o, TR, 0, 0, tr + o, ti + o,
                                    0, 0, TR, 0, 0, pin, 0, 0);
                    wino17_w4(tr + 16, ti + 16, TR, 0, 0, tr + 16, ti + 16,
                              0, 0, TR, 0, 0, 0, 0, 0);
                } else
#endif
                for (int lb = 0; lb < TR/VW; ++lb) {
                    long o = (long)lb*VW;
                    SFX(wino17)(tr + o, ti + o, TR, 0, 0, tr + o, ti + o,
                                0, 0, TR, 0, 0, pin, 0, 0);
                }

                TP17D(tr, TR, ur, TR, dey); /* T[kz][y] -> U[y][kz] */
                TP17D(ti, TR, ui, TR, dey);

                /* y pass: axis stride TR, lanes = kz, straight into A[x][ky][kz] */
                double *dr = ar + (long)x*PS, *di = ai + (long)x*PS;
#if VW == 8
                if (mixed) {
                    long mlim = 8;              /* opaque: keep the 2-trip   */
                    __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
                    for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                        SFX(wino17)(ur + o, ui + o, TR, 0, 0, dr + o, di + o,
                                    0, 0, LN, 0, 0, pin, 0, 0);
                    wino17_w4(ur + 13, ui + 13, TR, 0, 0, dr + 13, di + 13,
                              0, 0, LN, 0, 0, 0, 0, 0);
                } else
#endif
                for (int lb = 0; lb < NLB; ++lb) {
                    long o = SFX(LBOFF)[lb];
                    SFX(wino17)(ur + o, ui + o, TR, 0, 0, dr + o, di + o,
                                0, 0, LN, 0, 0, pin, 0, 0);
                }
            }

            /* x pass: axis stride PS, 289 contiguous lanes, interleaving store.
             * The last block is clamped to start at NPL-VW: it overlaps the
             * previous block and recomputes a few lanes bit-identically (each
             * lane's arithmetic is independent of m0). */
            if (ph == 1) continue;
#if VW == 8
            if (mixed) {
                /* stg: kernel stores go to the L2-hot staging volume (no pfw
                 * there -- nothing DRAM-destined), then one dense flush. */
                double *const xd = stg ? p->vo_w8 : dst;
                const int pfwl = stg ? 0 : p->pfw;
                if (pfwl) { SFX(pfw_rows)(dst, 0, 0); SFX(pfw_rows)(dst, 8, 0); }
#pragma GCC unroll 1
                for (long m0 = 0; m0 + VW <= NPL; m0 += VW) {
                    if (pfwl) {
                        if (m0 <= 264)      SFX(pfw_rows)(dst, m0 + 16, 0);
                        else if (m0 == 272) SFX(pfw_rows)(dst, NPL - 4, 1);
                    }
                    SFX(wino17)(ar + m0, ai + m0, PS, 0, 0, 0, 0, xd, m0,
                                NPL, 0, 1, pin, 0, 0);
                }
                wino17_w4(ar + (NPL - 4), ai + (NPL - 4), PS, 0, 0, 0, 0,
                          xd, NPL - 4, NPL, 0, 1, 0, 0, 0);
                if (stg) SFX(vo_flush)(p->vo_w8, dst, p->pfw);
            } else
#endif
            for (int blk = 0; blk < NXB; ++blk) {
                long m0 = (long)blk*VW;
                if (m0 > NPL - VW) m0 = NPL - VW;
                SFX(wino17)(ar + m0, ai + m0, PS, 0, 0, 0, 0, dst, m0,
                            NPL, 0, 1, pin, 0, 0);
            }
        }
    }
}

#if VW == 8
/* Overlapped-shuffle X-last body ("ov"), mixed-width shape only -- panel_r5.
 * The SAME operations as exec_body(pin, 0, 1), reordered so that every
 * serialized shuffle burst (the transposes) sits in the shadow of a zmm
 * kernel block: on the node's single 512-bit FMA unit a zmm block drains
 * ~296 port-0 uops at 1/cycle while alloc (4/cycle) runs ahead and issues
 * the younger, independent shuffle/load uops on ports 5/2/3.  Specifically:
 *   - the z pass runs its ymm tail FIRST and the two zmm blocks LAST, so
 *     the T->U transpose that follows lands in a zmm drain;
 *   - T->U is emitted in halves: columns 0..7 (all the y pass's first block
 *     needs) before the y loop, columns 8..16 after the first y kernel;
 *   - the NEXT plane's deinterleave (T is dead once T->U is complete) is
 *     emitted in halves after the y pass's second zmm block and after its
 *     ymm tail; plane 0's deinterleave runs at the top of the volume, in
 *     the shadow of the previous volume's x pass.
 * All moved pieces write regions disjoint from anything concurrently live,
 * so the output is BIT-IDENTICAL to the other class-A variants (verified by
 * cmp); the tuner may rank it freely.  Wallaby (two 512-bit FMA units ->
 * drains half as long) understates the node benefit, like r4's "t" bet. */
static inline __attribute__((always_inline)) void SFX(exec_ov_body)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out,
        const int pin)
{
    double *const ar = p->SFX(ar), *const ai = p->SFX(ai);
    double *const tr = p->SFX(tr), *const ti = p->SFX(ti);
    double *const ur = p->SFX(ur), *const ui = p->SFX(ui);

    for (int b = 0; b < p->batch; ++b) {
        const double *src = (const double *)in + (size_t)2*NVOL*b;
        double *dst       = (double *)out      + (size_t)2*NVOL*b;
        const double *nxt =
            (p->pf && b + 1 < p->batch) ? src + (size_t)2*NVOL : NULL;

        /* plane 0: shadowed by the previous volume's x-pass drain (b > 0) */
        DT17P(src, tr, ti, TR, 0);
        DT17P(src, tr, ti, TR, 1);

        for (int x = 0; x < LN; ++x) {
            if (nxt) {
                const double *pp = nxt + 2*(long)x*NPL;
                for (int q = 0; q < 2*NPL; q += 8)
                    __builtin_prefetch(pp + q, 0, 2);
            }

            /* z pass, in place on T (column-disjoint blocks, any order):
             * ymm tail first, zmm blocks last */
            wino17_w4(tr + 16, ti + 16, TR, 0, 0, tr + 16, ti + 16,
                      0, 0, TR, 0, 0, 0, 0, 0);
            {
                long mlim = 8;              /* opaque: keep the 2-trip   */
                __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
                for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                    SFX(wino17)(tr + o, ti + o, TR, 0, 0, tr + o, ti + o,
                                0, 0, TR, 0, 0, pin, 0, 0);
            }

            /* T->U, columns 0..7 -- in the z pass's final zmm drain */
            TP17P(tr, TR, ur, TR, 0);
            TP17P(ti, TR, ui, TR, 0);

            /* y pass into A[x][ky][kz]; fillers ride each block's drain */
            double *dr = ar + (long)x*PS, *di = ai + (long)x*PS;
            const double *nsrc = src + 2*(long)(x + 1)*NPL;
            {
                long mlim = 8;
                __asm__("" : "+r"(mlim));
                for (long o = 0; o <= mlim; o += 8) {
                    SFX(wino17)(ur + o, ui + o, TR, 0, 0, dr + o, di + o,
                                0, 0, LN, 0, 0, pin, 0, 0);
                    if (o == 0) {
                        /* T->U columns 8..16 (T still live, U cols >= 8) */
                        TP17P(tr, TR, ur, TR, 1);
                        TP17P(ti, TR, ui, TR, 1);
                    } else if (x < 16) {
                        /* T is dead now: next plane's deinterleave, half 1 */
                        DT17P(nsrc, tr, ti, TR, 0);
                    }
                }
            }
            wino17_w4(ur + 13, ui + 13, TR, 0, 0, dr + 13, di + 13,
                      0, 0, LN, 0, 0, 0, 0, 0);
            if (x < 16)
                DT17P(nsrc, tr, ti, TR, 1);
        }

        /* x pass: unchanged mixed shape (zmm blocks + clamped ymm tail) */
#pragma GCC unroll 1
        for (long m0 = 0; m0 + VW <= NPL; m0 += VW)
            SFX(wino17)(ar + m0, ai + m0, PS, 0, 0, 0, 0, dst, m0,
                        NPL, 0, 1, pin, 0, 0);
        wino17_w4(ar + (NPL - 4), ai + (NPL - 4), PS, 0, 0, 0, 0,
                  dst, NPL - 4, NPL, 0, 1, 0, 0, 0);
    }
}

/* Software-pipelined X-last body ("sp"), mixed-width shape, panel_r6 -- r4's
 * "Next" item 3 and the monitor's named remaining lever for L=17 batched.
 * At batch, volume b's x pass is a dedicated burst writing 78.6 KB of
 * DRAM-destined `out` while the FMA stream sits mostly idle, and the plane
 * phase is compute-bound while the store path sits mostly idle.  This body
 * interleaves them ACROSS volumes: after each plane's y pass of volume b,
 * 2-3 x-pass blocks of volume b-1 run (37 blocks spread over 17 planes),
 * reading the OTHER buffer of a ping-pong A pair.  Per-volume kernel calls
 * and operands are exactly exec_body(pin=0, xfirst=0, mixed=1)'s, only the
 * global order moves across volume boundaries and volumes are independent,
 * so the output is BIT-IDENTICAL to every class-A candidate.  The x blocks
 * run behind the noinline xblk_run, so this whole body adds no inlined
 * kernel copy.  pfw composes: each block prefetchw's block k+2's rows. */
static inline __attribute__((always_inline)) void SFX(exec_sp_body)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out,
        const int pin, const int dey)
{
    double *const tr = p->SFX(tr), *const ti = p->SFX(ti);
    double *const ur = p->SFX(ur), *const ui = p->SFX(ui);
    const int pfw = p->pfw;
    const int nb = p->batch;

    for (int b = 0; b < nb; ++b) {
        const double *src = (const double *)in + (size_t)2*NVOL*b;
        double *const ar = (b & 1) ? p->ar2_w8 : p->SFX(ar);
        double *const ai = (b & 1) ? p->ai2_w8 : p->SFX(ai);
        const double *const par = (b & 1) ? p->SFX(ar) : p->ar2_w8;
        const double *const pai = (b & 1) ? p->SFX(ai) : p->ai2_w8;
        double *const pdst = (b > 0)
            ? (double *)out + (size_t)2*NVOL*(b - 1) : (double *)out;
        const double *nxt =
            (p->pf && b + 1 < nb) ? src + (size_t)2*NVOL : NULL;
        int xq = 0;

        for (int x = 0; x < LN; ++x) {
            if (nxt) {
                const double *pp = nxt + 2*(long)x*NPL;
                for (int q = 0; q < 2*NPL; q += 8)
                    __builtin_prefetch(pp + q, 0, 2);
            }

            if (dey) DT17Y(src + 2*(long)x*NPL, tr, ti, TR);
            else     DT17(src + 2*(long)x*NPL, tr, ti, TR);

            /* z pass, in place on T (identical to exec_body's mixed order) */
            {
                long mlim = 8;              /* opaque: keep the 2-trip   */
                __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
                for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                    SFX(wino17)(tr + o, ti + o, TR, 0, 0, tr + o, ti + o,
                                0, 0, TR, 0, 0, pin, 0, 0);
            }
            wino17_w4(tr + 16, ti + 16, TR, 0, 0, tr + 16, ti + 16,
                      0, 0, TR, 0, 0, 0, 0, 0);

            TP17(tr, TR, ur, TR);       /* T[kz][y] -> U[y][kz] */
            TP17(ti, TR, ui, TR);

            /* y pass into this volume's A half of the ping-pong pair */
            double *dr = ar + (long)x*PS, *di = ai + (long)x*PS;
            {
                long mlim = 8;
                __asm__("" : "+r"(mlim));
                for (long o = 0; o <= mlim; o += 8)
                    SFX(wino17)(ur + o, ui + o, TR, 0, 0, dr + o, di + o,
                                0, 0, LN, 0, 0, pin, 0, 0);
            }
            wino17_w4(ur + 13, ui + 13, TR, 0, 0, dr + 13, di + 13,
                      0, 0, LN, 0, 0, 0, 0, 0);

            /* 2-3 x-pass blocks of the PREVIOUS volume (37 over 17 planes) */
            if (b > 0) {
                int jn = ((x + 1)*37)/LN - (x*37)/LN;
                __asm__("" : "+r"(jn));
                for (int j = 0; j < jn; ++j, ++xq)
                    SFX(xblk_run)(par, pai, pdst, xq, pfw);
            }
        }
    }

    /* drain: the last volume's x pass */
    {
        const double *const lar = ((nb - 1) & 1) ? p->ar2_w8 : p->SFX(ar);
        const double *const lai = ((nb - 1) & 1) ? p->ai2_w8 : p->SFX(ai);
        double *ldst = (double *)out + (size_t)2*NVOL*(nb - 1);
        int jn = 37;
        __asm__("" : "+r"(jn));
        for (int j = 0; j < jn; ++j)
            SFX(xblk_run)(lar, lai, ldst, j, pfw);
    }
}

/* Staged-pipelined X-last body ("stp"), mixed-width shape, panel_r10.  The
 * staged x pass (stores into vo, see exec_body's stg flag) with the dense
 * flush of the PREVIOUS volume's vo paced across this volume's plane phase:
 * one 4.5 KB sequential chunk after each plane's y pass, vo ping-ponged so
 * vo(b-1) stays live while volume b's x pass fills vo(b).  This composes the
 * two lessons of the round's forced wallaby A/Bs: the scattered 17-stream
 * store is the traffic to delete (st), and an un-paced flush is a fully
 * exposed serial burst (st loses ~23% forced on wallaby streaming).  Unlike
 * sp -- whose paced work was kernel blocks with scattered partial-line
 * stores -- each paced piece here is ONE dense sequential stream, which is
 * the store pattern the hardware prefetcher and fill buffers handle at full
 * rate.  Per-volume kernel calls and operand values are exactly the staged
 * exec's; only the out-write order moves across independent volumes, so the
 * output is BIT-IDENTICAL to every class-A candidate. */
static inline __attribute__((always_inline)) void SFX(exec_stp_body)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out,
        const int pin, const int dey)
{
    double *const ar = p->SFX(ar), *const ai = p->SFX(ai);
    double *const tr = p->SFX(tr), *const ti = p->SFX(ti);
    double *const ur = p->SFX(ur), *const ui = p->SFX(ui);
    const int pfw = p->pfw;
    const int nb = p->batch;

    for (int b = 0; b < nb; ++b) {
        const double *src = (const double *)in + (size_t)2*NVOL*b;
        double *const vo  = (b & 1) ? p->vo2_w8 : p->vo_w8;
        const double *const pvo = (b & 1) ? p->vo_w8 : p->vo2_w8;
        double *const pdst = (b > 0)
            ? (double *)out + (size_t)2*NVOL*(b - 1) : (double *)out;
        const double *nxt =
            (p->pf && b + 1 < nb) ? src + (size_t)2*NVOL : NULL;

        for (int x = 0; x < LN; ++x) {
            if (nxt) {
                const double *pp = nxt + 2*(long)x*NPL;
                for (int q = 0; q < 2*NPL; q += 8)
                    __builtin_prefetch(pp + q, 0, 2);
            }

            if (dey) DT17Y(src + 2*(long)x*NPL, tr, ti, TR);
            else     DT17(src + 2*(long)x*NPL, tr, ti, TR);

            /* z pass, in place on T (identical to exec_body's mixed order) */
            {
                long mlim = 8;              /* opaque: keep the 2-trip   */
                __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
                for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                    SFX(wino17)(tr + o, ti + o, TR, 0, 0, tr + o, ti + o,
                                0, 0, TR, 0, 0, pin, 0, 0);
            }
            wino17_w4(tr + 16, ti + 16, TR, 0, 0, tr + 16, ti + 16,
                      0, 0, TR, 0, 0, 0, 0, 0);

            TP17(tr, TR, ur, TR);       /* T[kz][y] -> U[y][kz] */
            TP17(ti, TR, ui, TR);

            /* y pass into A[x] */
            double *dr = ar + (long)x*PS, *di = ai + (long)x*PS;
            {
                long mlim = 8;
                __asm__("" : "+r"(mlim));
                for (long o = 0; o <= mlim; o += 8)
                    SFX(wino17)(ur + o, ui + o, TR, 0, 0, dr + o, di + o,
                                0, 0, LN, 0, 0, pin, 0, 0);
            }
            wino17_w4(ur + 13, ui + 13, TR, 0, 0, dr + 13, di + 13,
                      0, 0, LN, 0, 0, 0, 0, 0);

            /* one dense flush chunk of the PREVIOUS volume's staged output */
            if (b > 0)
                SFX(vo_flush_chunk)(pvo, pdst, x, pfw);
        }

        /* staged x pass: kernel stores into vo (L2-hot, no pfw needed) */
#pragma GCC unroll 1
        for (long m0 = 0; m0 + VW <= NPL; m0 += VW)
            SFX(wino17)(ar + m0, ai + m0, PS, 0, 0, 0, 0, vo, m0,
                        NPL, 0, 1, pin, 0, 0);
        wino17_w4(ar + (NPL - 4), ai + (NPL - 4), PS, 0, 0, 0, 0,
                  vo, NPL - 4, NPL, 0, 1, 0, 0, 0);
    }

    /* drain: the last volume's flush */
    SFX(vo_flush)(((nb - 1) & 1) ? p->vo2_w8 : p->vo_w8,
                  (double *)out + (size_t)2*NVOL*(nb - 1), pfw);
}

/* Deferred-junction plane schedule ("dz"), mixed-width shape, panel_r7 --
 * ADOPTED FROM L17_matrixsimd panel_r6's deferred-Z (group-level deferral
 * with a double-buffered plane buffer; wallaby -3.0%/-5.7% at B=1/B=8 for
 * their structure).  exec_body's plane phase runs deint(x) -> z(x) ->
 * transpose(x) -> y(x) back to back: three store->load junctions per plane
 * where a group's tail stores are immediately re-read with no independent
 * work behind them.  This body runs the SAME groups pipelined one plane
 * deep, T double-buffered by plane parity (U single: its producer and
 * consumer are one kernel group apart in the same iteration):
 *
 *     deint(0) -> T0
 *     for x = 0..17:
 *         x>=1 : transpose(x-1)  T[(x-1)&1] -> U      [z(x) before y(x-1)]
 *         x<17 : z(x)            in place on T[x&1]
 *         x>=1 : y(x-1)          U -> A[x-1]  (+ sp: prev volume's x blocks)
 *         x<16 : deint(x+1)      in[x+1] -> T[(x+1)&1]
 *     then the x pass (or, under sp, the pipelined drain)
 *
 * Junction separations: deint(x+1)->z(x+1) has transpose(x) between;
 * z(x)->transpose(x) has y(x-1) and deint(x+1) between; transpose(x-1)->
 * y(x-1) has the whole z(x) kernel group (~740 zmm cycles) between.  The
 * only exposed junctions left are plane 0's deint->z and the final
 * transpose(16)->y(16).  Pure reordering of whole groups on disjoint
 * buffers -- zero extra instructions (the ov lesson: splitting loops costs
 * more than the shadow pays), same kernel operand values, so the output is
 * BIT-IDENTICAL to every class-A candidate.  `sp` (compile-time) composes
 * the panel_r6 cross-volume x-block pipeline into the same schedule. */
static inline __attribute__((always_inline)) void SFX(exec_dz_body)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out,
        const int pin, const int sp)
{
    double *const ur = p->SFX(ur), *const ui = p->SFX(ui);
    double *const t0r = p->SFX(tr),  *const t0i = p->SFX(ti);
    double *const t1r = p->tr2_w8,   *const t1i = p->ti2_w8;
    const int pfw = p->pfw;
    const int nb = p->batch;

    for (int b = 0; b < nb; ++b) {
        const double *src = (const double *)in + (size_t)2*NVOL*b;
        double *dst       = (double *)out      + (size_t)2*NVOL*b;
        double *const ar = (sp && (b & 1)) ? p->ar2_w8 : p->SFX(ar);
        double *const ai = (sp && (b & 1)) ? p->ai2_w8 : p->SFX(ai);
        const double *const par = (b & 1) ? p->SFX(ar) : p->ar2_w8;
        const double *const pai = (b & 1) ? p->SFX(ai) : p->ai2_w8;
        double *const pdst = (b > 0)
            ? (double *)out + (size_t)2*NVOL*(b - 1) : (double *)out;
        const double *nxt =
            (p->pf && b + 1 < nb) ? src + (size_t)2*NVOL : NULL;
        int xq = 0;

        DT17(src, t0r, t0i, TR);            /* plane 0 -> T0 */

        for (int x = 0; x <= LN; ++x) {
            if (nxt && x < LN) {
                const double *pp = nxt + 2*(long)x*NPL;
                for (int q = 0; q < 2*NPL; q += 8)
                    __builtin_prefetch(pp + q, 0, 2);
            }

            /* T[(x-1)&1] -> U; its consumer y(x-1) sits behind z(x) */
            if (x >= 1) {
                const double *sr = ((x - 1) & 1) ? t1r : t0r;
                const double *si = ((x - 1) & 1) ? t1i : t0i;
                TP17(sr, TR, ur, TR);
                TP17(si, TR, ui, TR);
            }

            /* z pass, in place on T[x&1] (exec_body's mixed order) */
            if (x < LN) {
                double *zr = (x & 1) ? t1r : t0r;
                double *zi = (x & 1) ? t1i : t0i;
                long mlim = 8;              /* opaque: keep the 2-trip   */
                __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
                for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                    SFX(wino17)(zr + o, zi + o, TR, 0, 0, zr + o, zi + o,
                                0, 0, TR, 0, 0, pin, 0, 0);
                wino17_w4(zr + 16, zi + 16, TR, 0, 0, zr + 16, zi + 16,
                          0, 0, TR, 0, 0, 0, 0, 0);
            }

            /* y pass of plane x-1: U -> A[x-1] */
            if (x >= 1) {
                double *dr = ar + (long)(x - 1)*PS, *di = ai + (long)(x - 1)*PS;
                long mlim = 8;
                __asm__("" : "+r"(mlim));
                for (long o = 0; o <= mlim; o += 8)
                    SFX(wino17)(ur + o, ui + o, TR, 0, 0, dr + o, di + o,
                                0, 0, LN, 0, 0, pin, 0, 0);
                wino17_w4(ur + 13, ui + 13, TR, 0, 0, dr + 13, di + 13,
                          0, 0, LN, 0, 0, 0, 0, 0);

                /* sp: 2-3 x-pass blocks of the PREVIOUS volume per y pass */
                if (sp && b > 0) {
                    const int k = x - 1;
                    int jn = ((k + 1)*37)/LN - (k*37)/LN;
                    __asm__("" : "+r"(jn));
                    for (int j = 0; j < jn; ++j, ++xq)
                        SFX(xblk_run)(par, pai, pdst, xq, pfw);
                }
            }

            /* NEXT plane's deinterleave -> T[(x+1)&1] (T[x-1] is dead:
             * transpose(x-1) consumed it above, in program order) */
            if (x + 1 < LN) {
                double *tdr = ((x + 1) & 1) ? t1r : t0r;
                double *tdi = ((x + 1) & 1) ? t1i : t0i;
                DT17(src + 2*(long)(x + 1)*NPL, tdr, tdi, TR);
            }
        }

        if (!sp) {
            /* x pass, mixed shape with pfw -- identical to exec_body's */
            if (pfw) { SFX(pfw_rows)(dst, 0, 0); SFX(pfw_rows)(dst, 8, 0); }
#pragma GCC unroll 1
            for (long m0 = 0; m0 + VW <= NPL; m0 += VW) {
                if (pfw) {
                    if (m0 <= 264)      SFX(pfw_rows)(dst, m0 + 16, 0);
                    else if (m0 == 272) SFX(pfw_rows)(dst, NPL - 4, 1);
                }
                SFX(wino17)(ar + m0, ai + m0, PS, 0, 0, 0, 0, dst, m0,
                            NPL, 0, 1, pin, 0, 0);
            }
            wino17_w4(ar + (NPL - 4), ai + (NPL - 4), PS, 0, 0, 0, 0,
                      dst, NPL - 4, NPL, 0, 1, 0, 0, 0);
        }
    }

    if (sp) {
        /* drain: the last volume's x pass */
        const double *const lar = ((nb - 1) & 1) ? p->ar2_w8 : p->SFX(ar);
        const double *const lai = ((nb - 1) & 1) ? p->ai2_w8 : p->SFX(ai);
        double *ldst = (double *)out + (size_t)2*NVOL*(nb - 1);
        int jn = 37;
        __asm__("" : "+r"(jn));
        for (int j = 0; j < jn; ++j)
            SFX(xblk_run)(lar, lai, ldst, j, pfw);
    }
}
#endif

/* exec variants.  VW==4 contributes the plain X-last and X-first entries;
 * VW==8 additionally contributes the pinned and mixed-tail ("t") ones. */
static void __attribute__((unused)) SFX(exec_np)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 0, 0, 0, 0);
}

static void __attribute__((unused)) SFX(exec_xf)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 1, 0, 0, 0, 0);
}

/* pin at BOTH widths since ice_r1: with AVX-512VL the w4 pipeline has
 * ymm16-31 to pin into, and on the ICX node "xl 256" has twice won the
 * stage-1 rank at the graded cell -- its ~200 32-B .LC memory-operand
 * constants per kernel block are exactly what pinning deletes.  On a
 * non-AVX512 host KPIN degrades to a plain constant and the variant is
 * simply absent from the table. */
static void __attribute__((unused)) SFX(exec_pin)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 1, 0, 0, 0, 0, 0);
}

#if VW == 8
static void __attribute__((unused)) SFX(exec_npm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1, 0, 0, 0);
}

static void __attribute__((unused)) SFX(exec_pinm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 1, 0, 1, 0, 0, 0);
}

static void __attribute__((unused)) SFX(exec_xfm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 1, 1, 0, 0, 0);
}

static void __attribute__((unused)) SFX(exec_xfpinm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 1, 1, 1, 0, 0, 0);
}

/* "dy": mixed X-last with the ymm deint tile (panel_r9) */
static void __attribute__((unused)) SFX(exec_npmdy)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1, 1, 0, 0);
}

/* "ty" (ice_r1): ymm 4x4-tile transposes AND ymm deint inside the w8
 * pipeline (dey = 2) -- port-5 relief for Ice Lake, where every 512-bit
 * two-source shuffle is p5-only while 256-bit shuffles dual-issue on p1/p5,
 * and the second 512-bit FMA pipe halves the kernel-drain shadow that used
 * to hide the zmm transpose bursts on CLX.  Data movement only, identical
 * values to identical places: bit-identical to every class-A candidate. */
static void __attribute__((unused)) SFX(exec_npty)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 0, 2, 0, 0);
}

static void __attribute__((unused)) SFX(exec_pinty)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 1, 0, 0, 2, 0, 0);
}

static void __attribute__((unused)) SFX(exec_npmty)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1, 2, 0, 0);
}

static void __attribute__((unused)) SFX(exec_pinmty)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 1, 0, 1, 2, 0, 0);
}

/* "st": mixed X-last with the staged dense out flush (panel_r10); the dy
 * twin composes with the node's r9 batched pick (`xl 512t dy`). */
static void __attribute__((unused)) SFX(exec_stm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1, 0, 0, 1);
}

static void __attribute__((unused)) SFX(exec_stmdy)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1, 1, 0, 1);
}

/* "stp": staged x pass + the previous volume's flush paced across the plane
 * phase (panel_r10) */
static void __attribute__((unused)) SFX(exec_stpm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_stp_body)(p, in, out, 0, 0);
}

static void __attribute__((unused)) SFX(exec_stpmdy)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_stp_body)(p, in, out, 0, 1);
}

/* plan-time probes (panel_r9): NEVER candidates -- exec_xp reads whatever A
 * holds, so its output is not the transform.  They exist so create() can
 * decompose the streaming exposure into plane-phase (input-side) and x-pass
 * (output-side) shares ON THE SCORING MACHINE and report it in the
 * description string (L36_pfa panel_r8 pattern). */
static void __attribute__((unused)) SFX(exec_ph)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1, 0, 1, 0);
}

static void __attribute__((unused)) SFX(exec_phdy)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1, 1, 1, 0);
}

static void __attribute__((unused)) SFX(exec_xp)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_body)(p, in, out, 0, 0, 1, 0, 2, 0);
}

static void __attribute__((unused)) SFX(exec_ovm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_ov_body)(p, in, out, 0);
}

static void __attribute__((unused)) SFX(exec_ovmpin)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_ov_body)(p, in, out, 1);
}

static void __attribute__((unused)) SFX(exec_spm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_sp_body)(p, in, out, 0, 0);
}

/* "sp dy": the cross-volume x-block pipeline with the ymm deint (panel_r9) */
static void __attribute__((unused)) SFX(exec_spmdy)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_sp_body)(p, in, out, 0, 1);
}

/* pin twins of the sp family (ice_r1): "xl 512t sp dy" won the first honest
 * nv=32 rank on the ICX node and the sp family had no pinned variants while
 * pin helped several others in the same table. */
static void __attribute__((unused)) SFX(exec_spmpin)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_sp_body)(p, in, out, 1, 0);
}

static void __attribute__((unused)) SFX(exec_spmdypin)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_sp_body)(p, in, out, 1, 1);
}

static void __attribute__((unused)) SFX(exec_dzm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_dz_body)(p, in, out, 0, 0);
}

static void __attribute__((unused)) SFX(exec_dzmpin)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_dz_body)(p, in, out, 1, 0);
}

static void __attribute__((unused)) SFX(exec_dzspm)(
        fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    SFX(exec_dz_body)(p, in, out, 0, 1);
}
#endif  /* VW == 8 */

/* ------------- ice_r4: fused-chain step bodies (both widths) ---------------
 * One chain step on ONE volume: FFT(map(src)) -> dst (raw, interleaved).
 * The map is LAZY (corpus sec 10.3): src holds the previous step's RAW FFT
 * output and (z + c)/(1 + |z + c|) is applied here, in the first pass over
 * the plane, where src and c both stream sequentially and are L2 hits (the
 * chain iterates one volume through all m steps -- see fft3d_chain).  domap=0
 * (step 1, reading the caller's x0, which is already a state) takes the plain
 * deint.  Two map placements, plan-time raced, bit-identical (same one map
 * ladder, same destinations):
 *   fd -- map fused INTO the zmm deint tile (deint_transpose17zm): no mp
 *         buffer, no extra shuffles, the ladder's FMA work interleaves with
 *         the tile's p5 shuffle bursts and the one vdivpd per 8 points rides
 *         the otherwise-idle divider under them.
 *   mp -- map as a per-plane pre-pass into the L1-hot mp buffer, then the
 *         normal deint (composes with dy's ymm tile).
 * The rest is exec_body's mixed X-last shape (2 zmm + ymm tail per pass); no
 * pf/pfw (everything is an L2 hit in the per-volume chain regime). */
#if defined(__AVX512VL__)
static inline __attribute__((always_inline)) void SFX(chain_body)(
        fft3d_plan *p, const double *src, const double *cf, double *dst,
        const int domap, const int pin, const int dey, const int fd,
        const int mps, const int xm, const int xk)
{
    double *const ar = p->SFX(ar), *const ai = p->SFX(ai);
    double *const tr = p->SFX(tr), *const ti = p->SFX(ti);
    double *const ur = p->SFX(ur), *const ui = p->SFX(ui);

    /* xm / xk (EAGER shapes): src already holds MAPPED state (this body
     * mapped it at the previous step's x-pass store), so the plane phase is
     * plain and domap is ignored; the map happens in the x pass below --
     * per-block epilogue (xm) or inside the kernel's ST macro (xk). */
    const int lazymap = domap && !xm && !xk;

    /* mps: plane 0's map is the loop prologue; plane x+1's map runs inside
     * iteration x, right after the z pass, so its 36 vdivpds and ~700 ladder
     * uops issue under the z drain and the transpose/y junction stalls
     * instead of as an exposed serial burst (sec 10.2: pipeline the divider
     * under FFT work).  Single mp buffer: deint(x) consumes mp before z(x),
     * map(x+1) overwrites it after -- no overlap.  Pure reordering of the
     * same operations: bit-identical to the mp shape. */
    if (lazymap && mps)
        l17r_map_run(src, cf, p->mp, NPL);

    for (int x = 0; x < LN; ++x) {
        const double *pl = src + 2*(long)x*NPL;
        if (lazymap && fd) {
            deint_transpose17zm(pl, cf + 2*(long)x*NPL, tr, ti, TR);
        } else {
            if (lazymap) {
                if (!mps)
                    l17r_map_run(pl, cf + 2*(long)x*NPL, p->mp, NPL);
                pl = p->mp;
            }
            if (dey) DT17Y(pl, tr, ti, TR);
            else     DT17(pl, tr, ti, TR);
        }

        /* z pass, in place on T (exec_body's mixed order at w8) */
#if VW == 8
        {
            long mlim = 8;              /* opaque: keep the 2-trip   */
            __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
            for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                SFX(wino17)(tr + o, ti + o, TR, 0, 0, tr + o, ti + o,
                            0, 0, TR, 0, 0, pin, 0, 0);
            wino17_w4(tr + 16, ti + 16, TR, 0, 0, tr + 16, ti + 16,
                      0, 0, TR, 0, 0, 0, 0, 0);
        }
#else
        for (int lb = 0; lb < TR/VW; ++lb) {
            long o = (long)lb*VW;
            SFX(wino17)(tr + o, ti + o, TR, 0, 0, tr + o, ti + o,
                        0, 0, TR, 0, 0, pin, 0, 0);
        }
#endif

        if (lazymap && mps && x + 1 < LN)
            l17r_map_run(src + 2*(long)(x + 1)*NPL,
                         cf + 2*(long)(x + 1)*NPL, p->mp, NPL);

        TP17D(tr, TR, ur, TR, dey); /* T[kz][y] -> U[y][kz] */
        TP17D(ti, TR, ui, TR, dey);

        /* y pass into A[x] */
        double *dr = ar + (long)x*PS, *di = ai + (long)x*PS;
#if VW == 8
        {
            long mlim = 8;
            __asm__("" : "+r"(mlim));
            for (long o = 0; o <= mlim; o += 8)
                SFX(wino17)(ur + o, ui + o, TR, 0, 0, dr + o, di + o,
                            0, 0, LN, 0, 0, pin, 0, 0);
            wino17_w4(ur + 13, ui + 13, TR, 0, 0, dr + 13, di + 13,
                      0, 0, LN, 0, 0, 0, 0, 0);
        }
#else
        for (int lb = 0; lb < NLB; ++lb) {
            long o = SFX(LBOFF)[lb];
            SFX(wino17)(ur + o, ui + o, TR, 0, 0, dr + o, di + o,
                        0, 0, LN, 0, 0, pin, 0, 0);
        }
#endif
    }

    /* x pass -> dst, interleaved (no staging, no pfw).  xm: each block's
     * epilogue maps the 17 rows just stored, in place; a later overlapping
     * block RE-STORES its lanes raw (overlap recompute, bit-identical) before
     * its own epilogue maps them, so every lane leaves mapped exactly once. */
#if VW == 8
#pragma GCC unroll 1
    for (long m0 = 0; m0 + VW <= NPL; m0 += VW) {
        SFX(wino17)(ar + m0, ai + m0, PS, 0, 0, 0, 0, dst, m0,
                    NPL, 0, 1, pin, cf, xk);
        if (xm) l17r_mapblk(dst, cf, m0, 0);
    }
    wino17_w4(ar + (NPL - 4), ai + (NPL - 4), PS, 0, 0, 0, 0,
              dst, NPL - 4, NPL, 0, 1, 0, cf, xk);
    if (xm) l17r_mapblk(dst, cf, NPL - 4, 1);
#else
    for (int blk = 0; blk < NXB; ++blk) {
        long m0 = (long)blk*VW;
        if (m0 > NPL - VW) m0 = NPL - VW;
        SFX(wino17)(ar + m0, ai + m0, PS, 0, 0, 0, 0, dst, m0,
                    NPL, 0, 1, pin, cf, xk);
        if (xm) l17r_mapblk4v(dst, cf, m0);
    }
#endif
}

static void __attribute__((unused)) SFX(chain_fd)(
        fft3d_plan *p, const double *s, const double *c, double *d, int domap)
{
    SFX(chain_body)(p, s, c, d, domap, 0, 0, 1, 0, 0, 0);
}

static void __attribute__((unused)) SFX(chain_fdpin)(
        fft3d_plan *p, const double *s, const double *c, double *d, int domap)
{
    SFX(chain_body)(p, s, c, d, domap, 1, 0, 1, 0, 0, 0);
}

static void __attribute__((unused)) SFX(chain_mp)(
        fft3d_plan *p, const double *s, const double *c, double *d, int domap)
{
    SFX(chain_body)(p, s, c, d, domap, 0, 0, 0, 0, 0, 0);
}

static void __attribute__((unused)) SFX(chain_mppin)(
        fft3d_plan *p, const double *s, const double *c, double *d, int domap)
{
    SFX(chain_body)(p, s, c, d, domap, 1, 0, 0, 0, 0, 0);
}

static void __attribute__((unused)) SFX(chain_mpdy)(
        fft3d_plan *p, const double *s, const double *c, double *d, int domap)
{
    SFX(chain_body)(p, s, c, d, domap, 0, 1, 0, 0, 0, 0);
}

static void __attribute__((unused)) SFX(chain_mps)(
        fft3d_plan *p, const double *s, const double *c, double *d, int domap)
{
    SFX(chain_body)(p, s, c, d, domap, 0, 0, 0, 1, 0, 0);
}

static void __attribute__((unused)) SFX(chain_mpspin)(
        fft3d_plan *p, const double *s, const double *c, double *d, int domap)
{
    SFX(chain_body)(p, s, c, d, domap, 1, 0, 0, 1, 0, 0);
}

static void __attribute__((unused)) SFX(chain_xm)(
        fft3d_plan *p, const double *s, const double *c, double *d, int domap)
{
    SFX(chain_body)(p, s, c, d, domap, 0, 0, 0, 0, 1, 0);
}

static void __attribute__((unused)) SFX(chain_xmpin)(
        fft3d_plan *p, const double *s, const double *c, double *d, int domap)
{
    SFX(chain_body)(p, s, c, d, domap, 1, 0, 0, 0, 1, 0);
}

#if VW == 8
/* dz + xm: exec_dz_body's deferred-junction plane schedule (panel_r7,
 * adopted from L17_matrixsimd's deferred-Z) composed with the eager x-pass
 * map.  The chain regime is L2-resident and latency-shaped like B=1/B=8 --
 * where dz measured -3.0%/-5.7% on wallaby -- not like the L3 batched cells
 * where the node declined it.  Same kernel operands, same map, T
 * double-buffered by plane parity: bit-identical to the other candidates. */
static inline __attribute__((always_inline)) void SFX(chain_dz_body)(
        fft3d_plan *p, const double *src, const double *cf, double *dst,
        const int pin)
{
    double *const ar = p->SFX(ar), *const ai = p->SFX(ai);
    double *const ur = p->SFX(ur), *const ui = p->SFX(ui);
    double *const t0r = p->SFX(tr), *const t0i = p->SFX(ti);
    double *const t1r = p->tr2_w8,  *const t1i = p->ti2_w8;

    DT17(src, t0r, t0i, TR);            /* plane 0 -> T0 */

    for (int x = 0; x <= LN; ++x) {
        /* T[(x-1)&1] -> U; its consumer y(x-1) sits behind z(x) */
        if (x >= 1) {
            const double *sr = ((x - 1) & 1) ? t1r : t0r;
            const double *si = ((x - 1) & 1) ? t1i : t0i;
            TP17(sr, TR, ur, TR);
            TP17(si, TR, ui, TR);
        }

        /* z pass, in place on T[x&1] */
        if (x < LN) {
            double *zr = (x & 1) ? t1r : t0r;
            double *zi = (x & 1) ? t1i : t0i;
            long mlim = 8;              /* opaque: keep the 2-trip   */
            __asm__("" : "+r"(mlim));   /* loop rolled (1 inlined    */
            for (long o = 0; o <= mlim; o += 8)   /* kernel copy)   */
                SFX(wino17)(zr + o, zi + o, TR, 0, 0, zr + o, zi + o,
                            0, 0, TR, 0, 0, pin, 0, 0);
            wino17_w4(zr + 16, zi + 16, TR, 0, 0, zr + 16, zi + 16,
                      0, 0, TR, 0, 0, 0, 0, 0);
        }

        /* y pass of plane x-1: U -> A[x-1] */
        if (x >= 1) {
            double *dr = ar + (long)(x - 1)*PS, *di = ai + (long)(x - 1)*PS;
            long mlim = 8;
            __asm__("" : "+r"(mlim));
            for (long o = 0; o <= mlim; o += 8)
                SFX(wino17)(ur + o, ui + o, TR, 0, 0, dr + o, di + o,
                            0, 0, LN, 0, 0, pin, 0, 0);
            wino17_w4(ur + 13, ui + 13, TR, 0, 0, dr + 13, di + 13,
                      0, 0, LN, 0, 0, 0, 0, 0);
        }

        /* NEXT plane's deinterleave -> T[(x+1)&1] */
        if (x + 1 < LN) {
            double *tdr = ((x + 1) & 1) ? t1r : t0r;
            double *tdi = ((x + 1) & 1) ? t1i : t0i;
            DT17(src + 2*(long)(x + 1)*NPL, tdr, tdi, TR);
        }
    }

    /* x pass with the eager map epilogues (chain_body's w8 shape) */
#pragma GCC unroll 1
    for (long m0 = 0; m0 + VW <= NPL; m0 += VW) {
        SFX(wino17)(ar + m0, ai + m0, PS, 0, 0, 0, 0, dst, m0,
                    NPL, 0, 1, pin, 0, 0);
        l17r_mapblk(dst, cf, m0, 0);
    }
    wino17_w4(ar + (NPL - 4), ai + (NPL - 4), PS, 0, 0, 0, 0,
              dst, NPL - 4, NPL, 0, 1, 0, 0, 0);
    l17r_mapblk(dst, cf, NPL - 4, 1);
}

static void __attribute__((unused)) SFX(chain_dzxm)(
        fft3d_plan *p, const double *s, const double *c, double *d, int domap)
{
    (void)domap;
    SFX(chain_dz_body)(p, s, c, d, 0);
}

static void __attribute__((unused)) SFX(chain_dzxmpin)(
        fft3d_plan *p, const double *s, const double *c, double *d, int domap)
{
    (void)domap;
    SFX(chain_dz_body)(p, s, c, d, 1);
}
#endif /* VW == 8 */

/* xk: the eager map INSIDE the x-pass kernel's ST macro (mst=1) -- ladder on
 * the split registers already in hand, no epilogue reload of stored rows */
static void __attribute__((unused)) SFX(chain_xk)(
        fft3d_plan *p, const double *s, const double *c, double *d, int domap)
{
    SFX(chain_body)(p, s, c, d, domap, 0, 0, 0, 0, 0, 1);
}

static void __attribute__((unused)) SFX(chain_xkpin)(
        fft3d_plan *p, const double *s, const double *c, double *d, int domap)
{
    SFX(chain_body)(p, s, c, d, domap, 1, 0, 0, 0, 0, 1);
}

/* ty: ymm 4x4 transposes + ymm deint inside the w8 chain (p5 relief; the
 * chain regime is L2-resident and closer to compute-bound than the batched
 * cells where ty washed out in ice_r1) */
static void __attribute__((unused)) SFX(chain_xmty)(
        fft3d_plan *p, const double *s, const double *c, double *d, int domap)
{
    SFX(chain_body)(p, s, c, d, domap, 0, 2, 0, 0, 1, 0);
}

static void __attribute__((unused)) SFX(chain_xmtypin)(
        fft3d_plan *p, const double *s, const double *c, double *d, int domap)
{
    SFX(chain_body)(p, s, c, d, domap, 1, 2, 0, 0, 1, 0);
}
#endif /* __AVX512VL__ */

#undef NXB
#undef NLB
#undef VRSQ14
#undef KPIN
#undef TP17D
#undef DT17Y
#undef DT17P
#undef DT17
#undef TP17P
#undef TP17
#undef VS
#undef VL
#undef DLO
#undef DLE
#undef IHI
#undef ILO
#undef vl
#undef vd
#undef PS
#undef TR

#endif /* L17R_TEMPLATE_PASS */
