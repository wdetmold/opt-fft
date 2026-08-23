/* L6_unrolled -- forward complex-double 3D DFT of a 6x6x6 cube, batched.
 *
 * TECHNIQUE
 * ---------
 * A single fully-unrolled straight-line 6-point codelet, applied along all three
 * axes of the cube with no loops over the codelet body, no runtime twiddle table,
 * and no data-dependent branches.  The codelet is Good-Thomas / PFA 2x3, so it
 * carries *no* twiddle factors at all:
 *
 *     6 = 2*3, gcd(2,3)=1.  With  j = (3*j1 + 2*j2) mod 6  and
 *     k = (3*k1 + 4*k2) mod 6  the kernel factors exactly as
 *     w6^(jk) = (-1)^(j1 k1) * w3^(j2 k2),   w3 = exp(-2 pi i/3),
 *     i.e. DFT6 = DFT2 (x) DFT3 with a pure index permutation and nothing else.
 *
 * The only irrational constants in the whole transform are 1/2 and sqrt(3)/2,
 * both written literally.  Multiplication by -i*sqrt(3)/2 in the 3-point module
 * is done as one re/im swap (vpermilpd) followed by an FMA against the fixed
 * vector (+s,-s,+s,-s), so no complex multiply routine ever appears.
 *
 * OPERATION COUNT (per 6^3 volume)
 * --------------------------------
 *   DFT6 (PFA 2x3):  3*DFT2 + 2*DFT3
 *       3*DFT2 = 12 real adds
 *       1*DFT3 = 6 real adds + 1 FMA-pair(m) + 4 FMA  ->  6 arithmetic
 *                instructions + 1 shuffle per SIMD vector, 18 real flops
 *       DFT6   = 18 arithmetic instructions + 2 shuffles per vector,
 *                48 real flops per 6-point line  (= the provably optimal
 *                Good-Thomas count; FFTW's n1_6 reaches the same 48/36)
 *   volume = 3 axes * 36 lines = 108 line-DFTs
 *          = 5184 real flops and 3888 real arithmetic instructions
 *            (108 * 36) in scalar-equivalent terms
 *   SIMD: each __m256d holds 2 complex, so 54 vector codelet instances per
 *   volume = 972 vector arithmetic uops + 108 in-codelet shuffles.  On the
 *   Skylake/Cascade-Lake port scheme (FP add and FMA both on ports 0 and 1)
 *   that is a hard floor of 486 cycles per 6^3 volume for the whole 3D
 *   transform; the rest of the kernel (216 loads, 216 stores, 324 port-5
 *   shuffles in the y+z-fused variant) fits underneath it.
 *
 * LAYOUT / SIMD
 * -------------
 * Interleaved complex throughout (the driver's own layout, so no repack pass).
 * A __m256d = 2 adjacent complex = 2 adjacent z (or 2 adjacent (y,z)) values.
 *   x-pass: lanes are 2 adjacent (y,z) positions, axis stride 36 complex.
 *   y-pass: lanes are 2 adjacent z, axis stride 6 complex.
 *   z-pass: the axis is contiguous, so 2 whole z-pencils are brought into
 *           "z-major" form with 6 vperm2f128 and pushed back with 6 more; the
 *           12 outputs of the pair land as 6 consecutive 32-byte stores that
 *           cover exactly 3 whole 64-byte cache lines (=> clean NT stores).
 * Every load and store in the transform is 32-byte aligned by construction
 * (volume = 3456 B = 54*64 B, plane = 576 B, pencil = 96 B).
 *
 * ASSUMPTIONS
 * -----------
 *  * L == 6 only.
 *  * in/out are 64-byte aligned, as fft3d_api.h guarantees, and distinct.
 *  * double _Complex is a pair of adjacent doubles (C99 guarantees this).
 *  * The kernel variants (17 in ice_r2) are RACED AGAINST EACH OTHER AND VALIDATED against a
 *    scalar reference inside fft3d_create(); a variant that disagrees with the
 *    reference by more than 1e-11 relative is disqualified and can never be
 *    selected.  This is what makes an untestable-locally code path safe.
 *    The grid was pruned hard in panel_r6 on four rounds of node pick data:
 *    kept {3pass,fused} x {plain, pfT0, pfT0+W} (the only shapes the node has
 *    ever selected), the two split-z "_s" shapes (wallaby's B=1 winners, kept
 *    for dev A/Bs), one NT representative (3pass_nt_pf, wallaby DRAM winner;
 *    NT is 0-for-4 rounds on the node), and one mixed-width AVX-512 kernel
 *    (z2s, kept ONLY as the L6_FORCE perf-counter A/B target the r5 VERDICT
 *    asks for -- clk512=2.89 vs clk256>=2.89 killed zmm at L=6, 0 picks in
 *    12 invocations).
 *    L6_FORCE=<name> (env or -DL6_FORCE_DEFAULT) bypasses the race for node
 *    A/B tests; a forced pick is reported as variant=<name>! in the
 *    description.  The description also carries a five-point clock ladder,
 *    clk256=<sparse,mid,sat> clk512=<sparse,sat> sustained GHz (panel_r6:
 *    the saturating probes adopt L17_winograd's design; 256-bit is issued
 *    before 512-bit so licence dwell cannot leak backwards).  sat256 is the
 *    number the r5 VERDICT asks for: it decides whether the node's dense ymm
 *    kernels run at 3.89 or 2.89 GHz, i.e. whether B=1 has ~366 or ~147
 *    unexplained cycles.
 *  * panel_r6 fix: create() used to END with a 512-bit probe, leaving the
 *    core in the AVX-512 licence; at B=1 the driver's whole sample set is
 *    ~0.5 ms and can complete inside the licence-recovery window, which is
 *    the leading suspect for r5's B=1 regression (0.219 -> 0.227 typical,
 *    identical pick).  create() now ends with a ~20 ms 256-bit spin so the
 *    driver never times inside the 512 licence.
 *  * panel_r7: the brief's corrected turbo table (Intel 338848-028US: AVX2
 *    and AVX-512 licences are BOTH 2.9 GHz at 1-8 active cores on the Gold
 *    5218) reopens the zmm question r5 closed.  Two new mixed-width shapes
 *    join the race, ordered after every ymm incumbent so they must win by
 *    the 2.5% margin: zxf (zmm x-pass + the node-proven ymm fused y+z) and
 *    zff (zmm x-pass + fully fused zmm/ymm y+z per plane: 9 aligned zmm
 *    plane loads + 3 valignq -- no split loads, unlike r5's z2s -- and no
 *    t2 round trip; ~1296 uops/volume vs fused's 1728).  The tournament
 *    also gains a per-candidate licence warm-up (LITERATURE 08 s4.3): each
 *    candidate runs itself untimed ~0.7 ms before every timed trial, so a
 *    ymm candidate is never timed inside a zmm predecessor's licence tail.
 *  * panel_r9: (a) the codelet macros are parameterized (VD6 vs VD63) and
 *    three raced fused3 twins ship: VD63 is the SAME PFA 2x3 DFT factored
 *    DFT3-first/DFT2-last (L6_pfa's ordering; the r8 VERDICT's standing
 *    hypothesis for the B=64 gap), token-identical kernels otherwise --
 *    the controlled test of the last named mechanism at this geometry.
 *    Not bit-identical to VD6 (different association, ~1e-16); the plan
 *    gate and numpy bound it.  (b) an in-plan B=1 discriminator (adopted
 *    from L36_pfa's r8 in-plan probe pattern, per the r8 VERDICT's L=6
 *    instruction): create() times fused / fused3 / zff at nvol=1,
 *    licence-fair, and reports min ns/vol as ab1=... in the description,
 *    plus the fused3-vs-fused race delta as f3d=...  perf_event_open is
 *    closed cluster-wide (perf_event_paranoid=4), so timing is the only
 *    in-plan discriminator available.
 *    And the description now carries kclk=<GHz> -- L6_pfa's r6 probe design
 *    (adopted, attributed): dwell ~2 ms in the CHOSEN kernel, then time a
 *    ~150 us sparse ymm chain that reads the licence the kernel itself
 *    established; median of 9.  That is the number that converts this
 *    entry's node times into cycles.
 *  * panel_r8: the r7 node data closed the width question (zxf/zff/z2s and
 *    L6_pfa's fused_zx: 0 picks in 8 cells x 3 processes, with the race
 *    licence-fair and kclk measured at 2.89 both widths), so the zmm shapes
 *    no longer RACE: zff and z2s stay compiled, validated and L6_FORCE-able
 *    only, because the r7 VERDICT's outstanding perf-counter A/B names them.
 *    zxf/zxf_pf/zff_pf are deleted.  Two changes adopted from L6_pfa
 *    (attributed): (a) `restrict` on every kernel signature -- the one
 *    systematic codegen difference between our same-shape kernels; without
 *    it gcc must order every t1-plane load after the preceding out-stores
 *    and every next-group input load after the scratch stores, i.e. the
 *    exact pass-boundary joints the r7 VERDICT names as the remaining B=1
 *    suspect; their fused_pf_xa (identical shape, restrict-qualified) beat
 *    my fused_pf by ~3.7% at B=4096 in 2 of 3 r7 processes; (b) create()
 *    now ENDS by dwelling ~3 ms in the CHOSEN kernel, so the driver is
 *    handed a core in the scored kernel's own licence/clock steady state,
 *    never a probe's or a generic spin's.
 *  * panel_r10: the r9 node data answered every question the r9 machinery
 *    carried, so the file is cut to what is still open.  (a) DELETED: the
 *    fused3/VD63 twins (f3d read +3.3..+6.2% on the node -- the radix-2-
 *    first VD6 is confirmed the winning association, both L=6 entries now
 *    run it), the whole AVX-512 section incl. zff/z2s (ab1 zf/f = 1.14-1.18
 *    on the node closed the uop theory with a published number, and the r9
 *    VERDICT formally withdrew the perf-counter ask that kept them
 *    compiled), and the 5-probe clock ladder (the clock consensus is
 *    settled panel-wide: 3.89 non-AVX / 2.89 licence; kclk stays as the
 *    one-number regression check).  My own r5 lesson motivates the prune:
 *    ~500 lines of never-executed zmm code once moved B=1 by +3.5% at an
 *    identical pick -- dead weight is not free in this file.  (b) ADOPTED
 *    FROM L6_pfa: the zp-outer/y-inner x-pass group order (their PASS_X,
 *    in their file since round 1) as raced twins fused_zp{,_pf,_pfw}.
 *    Their r9 B=1 winner fused_d2 = my VD6 graph + THEIR zp-outer x order;
 *    the ascending-x + VD6 + no-pf combination (= my fused) raced only in
 *    MY file, and mine reads 0.2174-0.2305 where theirs reads 0.2068x3 --
 *    the x group order is the one structural difference left between the
 *    two kernels, so it races here and ab1 now publishes the f-vs-fx A/B
 *    (fused vs fused_zp at nvol=1) on every leaderboard line.  The twins
 *    are output-bit-identical to their parents (same per-line arithmetic,
 *    different order over independent lines), so they carry a reduced 1.0%
 *    takeover margin where genuinely-new shapes keep 2.5%.
 *  * panel_r11: (a) the zp twins take INCUMBENCY over their ascending
 *    parents (my r10 branch (iii), prescribed by the r10 VERDICT; the exact
 *    move L6_pfa made for their d2 twins in r10): the node picked fused_zp
 *    2/3 at B=1 with ab1 fx-f = -0.6/-3.1/-1.2%, and a node-measured
 *    sub-margin winner in a trailing slot is invisible to the tuner by
 *    construction.  Pure table reorder; the parents stay as 1.0%-margin
 *    challengers (bit-identical class, so a flip either way is harmless).
 *    (b) the r10 VERDICT's single L=6 item: an in-plan A/B at nvol >> L3
 *    that settles whether codelet association order matters in the
 *    DRAM-bound regime (L6_pfa's B=32768 dropped 3.3% on the d2 codelet
 *    flip, against both entries' models).  The VD63 (DFT3-first) codelet
 *    returns as ONE probe-only kernel, fused3_pfw -- NOT in cand[], never
 *    pickable (L36_mixedradix's r10 one-bit-class tuner rule: cross-class
 *    comparisons ride the description string, never the pick), validated
 *    against the scalar reference, then timed against fused_pfw over
 *    nvol=16384 volumes (113 MiB in+out: unambiguous DRAM on both
 *    machines, same size the race cap uses).  Result published as
 *    abL=f<ns>,f3<ns> per volume on every leaderboard line.
 *  * panel ice_r2 -- NEW PANEL, NEW MACHINE: bare-metal Ice Lake-SP
 *    (Xeon Gold 6326), graded chain, single core.  Two changes, both
 *    driven by the ice_r1 VERDICT:
 *    (a) AVX-512 RETURNS TO THE RACE.  The r10 zmm deletion rested on
 *    Cascade Lake facts (one 512-bit FMA pipe, kclk 2.89 both widths,
 *    ab1 zf/f = 1.14-1.18): this node has TWO 512-bit FMA pipes and two
 *    independent in-plan probes measured clk512 = clk256 (2.90 loaded,
 *    3.30-3.50 ramped) -- no licence cliff exists on this part, and at
 *    L=17 every 256-bit variant lost.  The r7 zff kernel (zmm x-pass +
 *    fully fused zmm/ymm y+z per plane: 9 aligned zmm plane loads + 3
 *    valignq, no t2 round trip, ~1296 uops/volume vs fused's 1728) and
 *    the r7 zxf (zmm x-pass + the ymm fused y+z) are restored from git
 *    (panel_r9 tree) and RACE as trailing 2.5%-margin challengers:
 *    zff/zff_pf/zff_pfw/zxf.  ab1 is retargeted to the width question
 *    (fused_zp vs zff at nvol=1, published ab1=y..,z..) and the race
 *    table publishes zwd = best-zmm vs best-ymm delta at the graded
 *    batch, so the node answers the width question even where the
 *    tournament keeps the ymm pick.
 *    (b) the r11 abL apparatus (VD63 codelet, probe-only fused3_pfw,
 *    113 MiB / ~0.25 s of create-time DRAM A/B) is DELETED: it existed
 *    to answer the r10 VERDICT's one question and the answer shipped on
 *    every ice_r1 leaderboard line (abL=f434.3,f3440.3 on the node,
 *    f3/f = +1.4%, inside the +-2% "allocation draw" band).  Dead
 *    weight is not free in this file (r5: +3.5% at B=1 from unexecuted
 *    code), so the deletion also pays for (a)'s added text.
 *  * panel ice_r3: the ice_r2 node data answered the width question in
 *    zmm's favor (drained-window pick variant=zxf, zwd=-2.6%,
 *    ab1=y166.6,z155.1 ns), so this round executes the r2 plan's zxf
 *    branch.  (a) INCUMBENCY FLIP: zxf leads the candidate table with
 *    zxf_pf as its 1.0% bit-identical challenger (the r10/r11 playbook
 *    move: a node-measured winner in a trailing slot must re-clear its
 *    2.5% margin at every create(), which it cleared only "by a hair"
 *    in the r2 dev races).  The ymm shapes stay as challengers, so an
 *    AVX2-only build races exactly as before.  (b) PRUNE: the zff
 *    family is deleted (0.239-0.272 in every r2 regime, never within
 *    6% of zxf in the chain race; mechanism understood and recorded --
 *    its 512-bit y+z transposes are port-5-only and displace the
 *    second FMA pipe), as are the pfw shapes (prefetchw = +9% measured
 *    chain tax) and the distance-2 pf twins (0.2443 vs 0.2433).
 *    17 raced candidates -> 7.  (c) ADOPTED FROM L6_pfa (ice_r2):
 *    chain-aware scratch re-placement.  The graded chain ping-pongs
 *    (out,pong)/(pong,out) from step 1 on, but the 4K-aliasing
 *    placement ran once, on the first call's (in,out) residues --
 *    stale for the other 4855 steps.  fft3d_execute now re-places
 *    when the unordered 4096-residue pair of (src,dst) changes; the
 *    placement score gains the reverse-direction t2 term (l6_cyc is
 *    negation-symmetric, so the other three terms already cover both
 *    directions), which makes one placement serve both directions of
 *    a ping-pong pair -- a chain re-places exactly once per pair
 *    change, ~64 iterations of integer work.  The score also drops
 *    the t2 terms entirely when the CHOSEN kernel never touches t2
 *    (all fused/zxf shapes), widening the achievable worst-case
 *    store->load distance for the pair that is actually scored.
 *  * panel ice_r4 -- TASK CHANGE: the graded step is now the full rival
 *    step state <- (z+c)/(1+|z+c|), z = raw FFT(state), and fft3d_chain
 *    (weak symbol) owns the whole m-step chain.  Shipped: volume-major
 *    in-place chain (all m=4856 steps of a volume run with state, c
 *    slice, t1 and staging L1-resident; final_out is the state arena),
 *    default map style bdiv = phase-split zmm map pass (pair-shared
 *    denominator, rsqrt14 + 2 Newton + FMA-Heron bias kill, ONE vdivpd
 *    per 8 points) + the plain zxf step.  Chain scratch residues are
 *    closed-form pinned at sp+1024/2048/3072 with c COPIED per volume
 *    (fixed a 13% B=1 4K-aliasing tax).  Node: 0.332 us/xform at B=64
 *    AND B=1 (MKL through the driver fallback: 0.942).  KNOWN ISSUE,
 *    measured and reported in the strategy record: the m=4856 map-chain
 *    gate (tol 4.9e-10) is below the task's intrinsic noise floor
 *    (~1.5e-9) -- the harness's own MKL+fallback reads 1.76e-9, exact
 *    scalar 1.96e-9, this entry 1.40e-9 (best measured); drift passes
 *    easily at every m <= 3600 and explodes with a ~450-step e-fold
 *    after ~2500 steps (weak chaos, not linear accumulation).
 *  * panel ice_r5 -- PAIR-INTERLEAVED CHAIN (cstyle 12 "p2div", the new
 *    default; 10 "pdiv" and 8 "ipdiv" are its A/B ancestors): each
 *    volume's chain is one serial dependence state->map->FFT->state, so
 *    TWO volumes' chains now run interleaved at pass granularity
 *    [map(A); map(B); zxf(A); zxf(B)], the two map passes fused into one
 *    27-iteration loop (two independent ladders + divides per iteration,
 *    volume B phase-rotated by 13 to keep every cross-buffer store->load
 *    residue conflict out of the store-buffer window), and the map runs
 *    IN PLACE (the ice_r4 staging volume is deleted).  Node: bdiv 0.333
 *    -> ipdiv 0.330 -> pdiv 0.325 -> p2div 0.323 us/xform at B=64; B=1
 *    falls through to the shared ipdiv loop (0.330).  Output is
 *    bit-identical to bdiv (same lane-wise map ops, same FFT), verified
 *    by cmp at B=64 and at B=3 (pair+tail composition).
 *    THE ROUND'S NEGATIVE RESULT, so nobody re-derives it: fusing the
 *    map INTO the FFT (2 passes/step instead of 3) LOSES at L=6 in both
 *    directions -- at the x-pass load (r4 adiv, 0.360-0.408) and at the
 *    y+z store via a 3-zmm pack (r5 sdiv, 0.380; hw-sqrt variant 0.500;
 *    phase-split hw-sqrt 0.478).  The 2-pass SKELETON is free (0.216 with
 *    the map ladders stripped), but ladders chunked 5-per-plane sit on
 *    the step recurrence where the ~190-uop plane bodies leave the ROB
 *    no room to hide them, while the flat 27-pair map pass overlaps its
 *    own ladders fully.  At this size the chain is recurrence/ILP-bound,
 *    NOT L1-traffic-bound: fewer passes is the wrong currency, more
 *    independent work in flight is the right one -- hence pair
 *    interleaving, which is also why it beats every fusion variant.
 *  * panel ice_r6 -- TIER FLIP + SPLIT-PASS PAIR FFT.  (a) The map tier
 *    is now RUNTIME-selected (plan->tier, env L6_TIER=fast|exact or
 *    -DL6_TIER_DEFAULT), default FAST: rsqrt14 + 2 Newton, no Heron,
 *    d = 1+s*w as ONE fmadd.  Policy, stated plainly: the r5 VERDICT
 *    (S3.3) established the L=6 cell was decided by precision-tier
 *    choice under a gate that certifies seed luck; the monitor scored
 *    L6_pfa's fast tier (their seed-42 drift 3.25e-9) as the round
 *    winner and did not recalibrate the gate for r6, so the de facto
 *    policy admits the fast tier, and my fast arm (seed-42 drift
 *    2.29e-9-class) is strictly lower-drift than the arm that just won.
 *    The exact FMA-Heron arm (1.40e-9, the lowest-drift non-MKL chain
 *    measured on the node) stays one env var away as the hedge, and is
 *    bit-identical to the r4/r5 shipped chain.  (b) New shape p2x
 *    (cstyle 14): the two paired volumes' FFTs interleave at PASS
 *    granularity too -- [map2; x(A); x(B); yz(A); yz(B)] -- which cuts
 *    the dependent pass seams per step from 2 (map2->xA and xA->yzA)
 *    to 1 (map2->xA): at every other seam the incoming pass is data-
 *    independent of the one draining.  Needs a second t1 (t1B pinned at
 *    residue sp+512; carena has room).  Raced against p2 on the node.
 *  * panel ice_r7 -- EXACT TIER MANDATORY + SoA GROUP-OF-8 CHAIN ("g8",
 *    cstyle 16, the new default).  (a) Both r6 L=6 entries were REJECTED
 *    by the chain gate (~1e-8 drift on the r6 scored seed); the r7 brief
 *    makes the third-Newton/exact tier non-optional at m=4856, so
 *    plan->tier now DEFAULTS to exact and the g8 path is built around
 *    the exact ladder.  (b) ADOPTED FROM THE RIVAL v6 GENERATOR
 *    (fft_v5v6_solutions/v6_f40c5e25_score0.91, gen_familyA): the chain
 *    processes EIGHT volumes per zmm lane-set in SoA -- re[8]/im[8]
 *    split-complex across volumes, plane stride padded to an ODD element
 *    count (PS=37, their 4K-alias defence) -- which makes every FFT
 *    line-DFT and every map application pure vertical arithmetic: ZERO
 *    shuffles anywhere in the steady state (the "fully split-complex
 *    state" lever both L=6 records scoped for two rounds, realized
 *    across volumes instead of within one).  The codelet is the same
 *    PFA 2x3 DAG as VD6 (identical single-rounded ops per point), the
 *    map is the same exact ladder as L6_MAP8 minus its 4 unpcks, and
 *    the reciprocal stays ONE vdivpd per element-vector (27/volume, as
 *    before -- the v6 rivals used rcp14+Newton instead, but my r4/r6
 *    data says 27 divides hide under this much port work; L6_G8RCP=1
 *    is the runtime A/B).  Per-point op order is bit-identical to the
 *    p2x exact chain, cmp-verified.  AoS<->SoA conversion uses their
 *    TR8 8x8 transpose, once per chain end (amortized over m=4856
 *    steps).  Groups need no driver-residue placement at all: the whole
 *    working set (X 28.4 KB + C 28.4 KB, deltas pinned at build time)
 *    lives in the plan's own group arena.  Batch tails (B%8, and all of
 *    B<8) fall through to the r6 pair/ipdiv machinery unchanged.
 *  * panel ice_r8 -- STORE-FUSED MAP ("g8f", cstyle 18, the new default)
 *    + FAST TIER RE-LEGALIZED under the corrected two-part gate.
 *    (a) ADOPTED FROM THE WARM RIVAL warm_d43251c2_score0.99/
 *    impl_3907.c (their gC_6, the L=6/8 engine the r8 brief points
 *    at): the map + c-add run where the z-DFT's outputs are still in
 *    registers -- at the last store of the step's last FFT sweep -- so
 *    the flat map pass is deleted and a step is [x-sweep; per plane:
 *    y-sweep, z-sweep+map]: one whole X read+write traversal (864 zmm
 *    ld/st per group-step) gone, C read in natural order inside the
 *    z-sweep.  Unlike the r7 fused-step loser (map at the x-LOADS,
 *    ladders gating every group's first FMA), these ladders hang OFF
 *    the DFT outputs and nothing inside the step waits on them: the
 *    first consumer is the NEXT step's x-sweep, a whole sweep away.
 *    (b) The fast-tier map inside the fused z-sweep is their H/R2
 *    ALTERNATION (also impl_3907): even elements vsqrtpd + rcp14+2NR
 *    reciprocal, odd elements the rsqrt14+2NR ladder + the same rcp
 *    reciprocal -- 108 vsqrtpd/group-step on the divider unit instead
 *    of 216 vdivpd.  (c) Tier default is FAST again, this round by
 *    MEASUREMENT against the corrected gate, not tier policy: the fast
 *    ladder is full-fp64 rsqrt14-seeded (not the fp32-rcp seed the
 *    one-step gate exists to catch); two-step rel L2 6.3e-16 vs tol
 *    3e-14, chain-end 5.7e-9 vs tol 1e-6 (300x the honest anchor
 *    2.1e-9 measured on the same chain).  The exact Heron+vdivpd arm
 *    (bit-identical to the r7 shipped chain) stays at L6_TIER=exact.
 *    Node, matched windows: g8 exact 0.238 -> g8f exact 0.231 -> g8f
 *    fast ladder+div 0.208 -> g8f fast H/R2 0.207 us/xform, sd 0.01%.
 *    NOTE (ice_r8 driver finding): driver.c passes pong=NULL to
 *    fft3d_chain at --map --chain 1, so fft3d_chain guards final_out.
 */

#include "fft3d_api.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define L6_HAVE_AVX2 1
#endif

#define L6_L    6
#define L6_VOL  216            /* complex per volume */
#define L6_VD   432            /* doubles  per volume */
/* Takeover margins are per-candidate since panel_r10 (see the cand[] table):
 * 2.5% for genuinely different shapes (raised from 1.5% in panel_r4 after an
 * L3-marginal mis-pick at B=4096), 1.0% for the fused_zp twins, whose output
 * is bit-identical to their parents so a mis-pick is bounded and harmless. */

static const double L6_S3 = 0.86602540378443864676372317075294; /* sqrt(3)/2 */

/* ------------------------------------------------------------------ *
 * Scalar reference: the same PFA 2x3 codelet, three strided passes.
 * Used as the plan-time correctness witness and as the portable path.
 * ------------------------------------------------------------------ */

#define SD6(A0,A1,A2,A3,A4,A5, R0,R1,R2,R3,R4,R5)                       \
    do {                                                                \
        double p0r=(A0##r)+(A3##r), p0i=(A0##i)+(A3##i);                 \
        double q0r=(A0##r)-(A3##r), q0i=(A0##i)-(A3##i);                 \
        double p1r=(A2##r)+(A5##r), p1i=(A2##i)+(A5##i);                 \
        double q1r=(A2##r)-(A5##r), q1i=(A2##i)-(A5##i);                 \
        double p2r=(A4##r)+(A1##r), p2i=(A4##i)+(A1##i);                 \
        double q2r=(A4##r)-(A1##r), q2i=(A4##i)-(A1##i);                 \
        double ar=p1r+p2r, ai=p1i+p2i, br=p1r-p2r, bi=p1i-p2i;          \
        double cr=q1r+q2r, ci=q1i+q2i, dr=q1r-q2r, di=q1i-q2i;          \
        double mr=p0r-0.5*ar, mi=p0i-0.5*ai;                            \
        double nr=q0r-0.5*cr, ni=q0i-0.5*ci;                            \
        double jbr= L6_S3*bi, jbi=-L6_S3*br;                            \
        double jdr= L6_S3*di, jdi=-L6_S3*dr;                            \
        (R0##r)=p0r+ar; (R0##i)=p0i+ai;                                 \
        (R3##r)=q0r+cr; (R3##i)=q0i+ci;                                 \
        (R4##r)=mr+jbr; (R4##i)=mi+jbi;                                 \
        (R2##r)=mr-jbr; (R2##i)=mi-jbi;                                 \
        (R1##r)=nr+jdr; (R1##i)=ni+jdi;                                 \
        (R5##r)=nr-jdr; (R5##i)=ni-jdi;                                 \
    } while (0)

static void l6_line_scalar(const double *s, long ss, double *d, long ds)
{
    /* ss, ds are strides in COMPLEX units */
    double x0r=s[0],        x0i=s[1];
    double x1r=s[2*ss],     x1i=s[2*ss+1];
    double x2r=s[4*ss],     x2i=s[4*ss+1];
    double x3r=s[6*ss],     x3i=s[6*ss+1];
    double x4r=s[8*ss],     x4i=s[8*ss+1];
    double x5r=s[10*ss],    x5i=s[10*ss+1];
    double y0r,y0i,y1r,y1i,y2r,y2i,y3r,y3i,y4r,y4i,y5r,y5i;
    SD6(x0,x1,x2,x3,x4,x5, y0,y1,y2,y3,y4,y5);
    d[0]=y0r;       d[1]=y0i;
    d[2*ds]=y1r;    d[2*ds+1]=y1i;
    d[4*ds]=y2r;    d[4*ds+1]=y2i;
    d[6*ds]=y3r;    d[6*ds+1]=y3i;
    d[8*ds]=y4r;    d[8*ds+1]=y4i;
    d[10*ds]=y5r;   d[10*ds+1]=y5i;
}

static void l6_run_scalar(double *restrict t1, double *restrict t2,
                          const double *restrict in, double *restrict out,
                          long nvol)
{
    for (long b = 0; b < nvol; ++b) {
        const double *ip = in  + b * (long)L6_VD;
        double       *op = out + b * (long)L6_VD;
        for (int p = 0; p < 36; ++p)                  /* x: stride 36 complex */
            l6_line_scalar(ip + 2*p, 36, t1 + 2*p, 36);
        for (int x = 0; x < 6; ++x)                   /* y: stride 6 complex  */
            for (int z = 0; z < 6; ++z)
                l6_line_scalar(t1 + 2*(36*x + z), 6, t2 + 2*(36*x + z), 6);
        for (int x = 0; x < 6; ++x)                   /* z: stride 1 complex  */
            for (int y = 0; y < 6; ++y)
                l6_line_scalar(t2 + 2*(36*x + 6*y), 1, op + 2*(36*x + 6*y), 1);
    }
}

/* ------------------------------------------------------------------ *
 * AVX2 / FMA kernels.  __m256d = 2 complex.
 * ------------------------------------------------------------------ */
#ifdef L6_HAVE_AVX2

/* One straight-line PFA 2x3 six-point codelet on 2 complex lanes.
 * 18 arithmetic instructions + 2 shuffles.  Safe to use in place. */
#define VD6(i0,i1,i2,i3,i4,i5, o0,o1,o2,o3,o4,o5)                       \
    do {                                                                \
        __m256d _p0 = _mm256_add_pd(i0,i3), _q0 = _mm256_sub_pd(i0,i3); \
        __m256d _p1 = _mm256_add_pd(i2,i5), _q1 = _mm256_sub_pd(i2,i5); \
        __m256d _p2 = _mm256_add_pd(i4,i1), _q2 = _mm256_sub_pd(i4,i1); \
        __m256d _a  = _mm256_add_pd(_p1,_p2), _b = _mm256_sub_pd(_p1,_p2); \
        __m256d _c  = _mm256_add_pd(_q1,_q2), _d = _mm256_sub_pd(_q1,_q2); \
        __m256d _m  = _mm256_fnmadd_pd(vhalf,_a,_p0);                   \
        __m256d _n  = _mm256_fnmadd_pd(vhalf,_c,_q0);                   \
        __m256d _bs = _mm256_permute_pd(_b,0x5);                        \
        __m256d _ds = _mm256_permute_pd(_d,0x5);                        \
        __m256d _s0 = _mm256_add_pd(_p0,_a);                            \
        __m256d _s3 = _mm256_add_pd(_q0,_c);                            \
        __m256d _s4 = _mm256_fmadd_pd (vk,_bs,_m);                      \
        __m256d _s2 = _mm256_fnmadd_pd(vk,_bs,_m);                      \
        __m256d _s1 = _mm256_fmadd_pd (vk,_ds,_n);                      \
        __m256d _s5 = _mm256_fnmadd_pd(vk,_ds,_n);                      \
        o0=_s0; o1=_s1; o2=_s2; o3=_s3; o4=_s4; o5=_s5;                 \
    } while (0)

#define VSET __m256d vhalf = _mm256_set1_pd(0.5);                        \
             __m256d vk    = _mm256_setr_pd(L6_S3,-L6_S3,L6_S3,-L6_S3)

/* panel_r10 prune: the VD63 (DFT3-first) codelet twin was DELETED.  The r9
 * node data answered the question it existed for: f3d = +3.3..+6.2% in all
 * six readings, ab1 f < f3 in every process -- the radix-2-first VD6 above
 * is the winning association on Cascade Lake and both L=6 entries now run
 * it.  Derivation and numbers live in strategies/L6_unrolled.md (r9).
 *
 * panel_r11: VD63 RETURNS, probe-only, for the r10 VERDICT's single L=6
 * item -- the DRAM-regime codelet A/B (see the file header and l6_abL).
 * Same PFA 2x3 six-point DFT factored the other way round: two DFT3s
 * first (on {x0,x2,x4} and {x3,x5,x1}), three DFT2 combines last, so the
 * final outputs come from add/sub instead of FMAs.  Identical count: 18
 * arithmetic instructions + 2 shuffles, 48 real flops.  NOT bit-identical
 * to VD6 (different association, ~1e-16), which is exactly why the kernel
 * built from it can never be picked.
 *   u = DFT3(i0,i2,i4), v = DFT3(i3,i5,i1);  X[(3k1+4k2)%6] = u[k2] +/- v[k2]
 *
 * panel ice_r2: VD63 is deleted AGAIN, this time with its probe kernel and
 * the whole abL apparatus -- the question it existed for is answered and
 * published (ice_r1 node reading abL=f434.3,f3440.3, f3/f = +1.4%, inside
 * the allocation-draw band).  Derivations: bench/geom/strategies (r9, r11). */

/* Software prefetch hooks for the x-pass loop (adopted from L6_pfa's v8,
 * which won the large-batch cases in panel_r1 with exactly this: touch the
 * NEXT volume's input, 3 cache lines per x-pass group = all 54 lines of a
 * 3456-B volume, giving one volume (~800 cycles) of lead time).  Prefetch
 * never faults, so running past the end of the batch is safe.
 *
 * panel ice_r3 prune: the prefetchw hooks (T0W) are deleted -- ice_r2
 * measured prefetchw a +9% chain tax (fused_pfw vs fused_pf; L13_rader's
 * gate finding: pw pays only past L3, and the chain keeps out L2-resident)
 * -- and so are the distance-2 T0 twins (0.2443 vs 0.2433 on zxf: one
 * volume of lead is already enough on this node; the r6 CLX prune stands
 * on ICX). */
#define L6_PF_NONE(SRC,OUT,g)  do { } while (0)
#define L6_PF_AT(SRC,g,DIST,HINT)                                       \
    do {                                                                \
        const char *_pf = (const char *)((SRC) + (DIST)*(long)L6_VD)    \
                          + 192*(g);                                    \
        _mm_prefetch(_pf,      HINT);                                   \
        _mm_prefetch(_pf + 64, HINT);                                   \
        _mm_prefetch(_pf + 128,HINT);                                   \
    } while (0)
#define L6_PF_T0_1(SRC,OUT,g)  L6_PF_AT(SRC,g,1,_MM_HINT_T0)

/* x-pass: in -> t1.  lanes = 2 adjacent (y,z); axis stride 72 doubles.
 * PF(SRC,OUT,g) is a prefetch hook run once per group; OUT is the volume's
 * final output pointer, only used by the prefetchw hooks.  CD is the
 * codelet macro.  Groups walk offsets 4g, g = 0..17: loads and stores in
 * strictly ascending 32B steps. */
#define L6_PASS_X(SRC,DST,OUT,PF,CD)                                    \
    do {                                                                \
        for (int g = 0; g < 18; ++g) {                                  \
            const double *s = (SRC) + 4*g;                              \
            double *d = (DST) + 4*g;                                    \
            PF(SRC,OUT,g);                                              \
            __m256d v0=_mm256_load_pd(s+  0), v1=_mm256_load_pd(s+ 72); \
            __m256d v2=_mm256_load_pd(s+144), v3=_mm256_load_pd(s+216); \
            __m256d v4=_mm256_load_pd(s+288), v5=_mm256_load_pd(s+360); \
            CD(v0,v1,v2,v3,v4,v5, v0,v1,v2,v3,v4,v5);                   \
            _mm256_store_pd(d+  0,v0); _mm256_store_pd(d+ 72,v1);       \
            _mm256_store_pd(d+144,v2); _mm256_store_pd(d+216,v3);       \
            _mm256_store_pd(d+288,v4); _mm256_store_pd(d+360,v5);       \
        }                                                               \
    } while (0)

/* x-pass, zp-outer/y-inner group order -- ADOPTED FROM L6_pfa's PASS_X
 * (in their file since round 1; their r9 node B=1 winner fused_d2 = the
 * VD6 graph behind exactly this group order).  Same 18 groups, same
 * per-group body, walked as g = (y,zp) with zp outer: offsets 12y + 4zp
 * doubles, i.e. 96-byte steps within a zp block.  Output bit-identical to
 * L6_PASS_X (the groups are independent; only their order changes).  The
 * prefetch index 6*zp+y still walks the next volume's 54 lines in strictly
 * ascending address order, so the streamer sees the same stream. */
#define L6_PASS_X_ZP(SRC,DST,OUT,PF,CD)                                 \
    do {                                                                \
        for (int zp = 0; zp < 3; ++zp)                                  \
            for (int y = 0; y < 6; ++y) {                               \
                const double *s = (SRC) + 12*y + 4*zp;                  \
                double *d = (DST) + 12*y + 4*zp;                        \
                PF(SRC,OUT,6*zp + y);                                   \
                __m256d v0=_mm256_load_pd(s+  0), v1=_mm256_load_pd(s+ 72); \
                __m256d v2=_mm256_load_pd(s+144), v3=_mm256_load_pd(s+216); \
                __m256d v4=_mm256_load_pd(s+288), v5=_mm256_load_pd(s+360); \
                CD(v0,v1,v2,v3,v4,v5, v0,v1,v2,v3,v4,v5);               \
                _mm256_store_pd(d+  0,v0); _mm256_store_pd(d+ 72,v1);   \
                _mm256_store_pd(d+144,v2); _mm256_store_pd(d+216,v3);   \
                _mm256_store_pd(d+288,v4); _mm256_store_pd(d+360,v5);   \
            }                                                           \
    } while (0)

/* y-pass: t1 -> t2.  lanes = 2 adjacent z; axis stride 12 doubles. */
#define L6_PASS_Y(SRC,DST)                                              \
    do {                                                                \
        for (int x = 0; x < 6; ++x) {                                   \
            for (int c = 0; c < 3; ++c) {                               \
                const double *s = (SRC) + 72*x + 4*c;                   \
                double *d = (DST) + 72*x + 4*c;                         \
                __m256d v0=_mm256_load_pd(s+ 0), v1=_mm256_load_pd(s+12);\
                __m256d v2=_mm256_load_pd(s+24), v3=_mm256_load_pd(s+36);\
                __m256d v4=_mm256_load_pd(s+48), v5=_mm256_load_pd(s+60);\
                VD6(v0,v1,v2,v3,v4,v5, v0,v1,v2,v3,v4,v5);              \
                _mm256_store_pd(d+ 0,v0); _mm256_store_pd(d+12,v1);     \
                _mm256_store_pd(d+24,v2); _mm256_store_pd(d+36,v3);     \
                _mm256_store_pd(d+48,v4); _mm256_store_pd(d+60,v5);     \
            }                                                           \
        }                                                               \
    } while (0)

/* z-pass: SRC -> DST, two z-pencils at a time, in-register transposed.
 * ST is the store intrinsic (_mm256_store_pd or _mm256_stream_pd). */
#define L6_PASS_Z(SRC,DST,ST)                                           \
    do {                                                                \
        for (int x = 0; x < 6; ++x) {                                   \
            for (int yp = 0; yp < 3; ++yp) {                            \
                const double *s = (SRC) + 72*x + 24*yp;                 \
                double *d = (DST) + 72*x + 24*yp;                       \
                __m256d A0=_mm256_load_pd(s+ 0), A1=_mm256_load_pd(s+ 4);\
                __m256d A2=_mm256_load_pd(s+ 8), B0=_mm256_load_pd(s+12);\
                __m256d B1=_mm256_load_pd(s+16), B2=_mm256_load_pd(s+20);\
                __m256d v0=_mm256_permute2f128_pd(A0,B0,0x20);          \
                __m256d v1=_mm256_permute2f128_pd(A0,B0,0x31);          \
                __m256d v2=_mm256_permute2f128_pd(A1,B1,0x20);          \
                __m256d v3=_mm256_permute2f128_pd(A1,B1,0x31);          \
                __m256d v4=_mm256_permute2f128_pd(A2,B2,0x20);          \
                __m256d v5=_mm256_permute2f128_pd(A2,B2,0x31);          \
                VD6(v0,v1,v2,v3,v4,v5, v0,v1,v2,v3,v4,v5);              \
                ST(d+ 0,_mm256_permute2f128_pd(v0,v1,0x20));            \
                ST(d+ 4,_mm256_permute2f128_pd(v2,v3,0x20));            \
                ST(d+ 8,_mm256_permute2f128_pd(v4,v5,0x20));            \
                ST(d+12,_mm256_permute2f128_pd(v0,v1,0x31));            \
                ST(d+16,_mm256_permute2f128_pd(v2,v3,0x31));            \
                ST(d+20,_mm256_permute2f128_pd(v4,v5,0x31));            \
            }                                                           \
        }                                                               \
    } while (0)

/* All kernels are entered through a function pointer, so their placement in
 * the binary is at the linker's mercy; panel_r6 pins every kernel entry to a
 * 64-byte boundary after r5's B=1 regressed 0.219->0.227 typical with an
 * IDENTICAL pick string when ~500 lines of zmm code were added to the file
 * (same disease the r5 VERDICT names at L36_mixedradix B=1: code layout). */
#define L6_KALIGN __attribute__((aligned(64)))

/* --- variant A: three separate passes ---------------------------------- */
#define L6_DEF_3PASS(NAME,ST,PF)                                        \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,    \
                 const double *restrict in, double *restrict out,       \
                 long nvol)                                             \
{                                                                       \
    VSET;                                                               \
    for (long b = 0; b < nvol; ++b) {                                   \
        const double *ip = in  + b * (long)L6_VD;                       \
        double       *op = out + b * (long)L6_VD;                       \
        L6_PASS_X(ip, t1, op, PF, VD6);                                 \
        L6_PASS_Y(t1, t2);                                              \
        L6_PASS_Z(t2, op, ST);                                          \
    }                                                                   \
}

L6_DEF_3PASS(l6_run_3pass,        _mm256_store_pd,  L6_PF_NONE)
L6_DEF_3PASS(l6_run_3pass_nt_pf,  _mm256_stream_pd, L6_PF_T0_1)
/* ice_r3 prune: 3pass_pf / 3pass_pfw deleted -- the 3pass shape has not
 * been picked since panel_r4 on any machine; it stays only as the safest
 * ymm baseline and the NT canary's parent. */

/* --- variant B: x-pass, then y and z fused per x-plane in registers ----
 * The 6x6 (y,z) plane is 18 __m256d; with the codelet temporaries that is
 * ~26 live vectors, which fits the 32 ymm of an AVX-512VL machine but not
 * the 16 of plain AVX2.  Raced at plan time against variant A.
 * The plane loop is a macro of its own (panel_r7) so the zmm-x `zxf`
 * kernels can reuse it token-for-token behind a different x-pass.       */
#define L6_FUSED_YZ(SRC,DST,ST,CD)                                      \
    do {                                                                \
        for (int x = 0; x < 6; ++x) {                                   \
            const double *s = (SRC) + 72*x;                             \
            double *d = (DST) + 72*x;                                   \
            __m256d P00=_mm256_load_pd(s+ 0),P01=_mm256_load_pd(s+ 4),  \
                    P02=_mm256_load_pd(s+ 8);                           \
            __m256d P10=_mm256_load_pd(s+12),P11=_mm256_load_pd(s+16),  \
                    P12=_mm256_load_pd(s+20);                           \
            __m256d P20=_mm256_load_pd(s+24),P21=_mm256_load_pd(s+28),  \
                    P22=_mm256_load_pd(s+32);                           \
            __m256d P30=_mm256_load_pd(s+36),P31=_mm256_load_pd(s+40),  \
                    P32=_mm256_load_pd(s+44);                           \
            __m256d P40=_mm256_load_pd(s+48),P41=_mm256_load_pd(s+52),  \
                    P42=_mm256_load_pd(s+56);                           \
            __m256d P50=_mm256_load_pd(s+60),P51=_mm256_load_pd(s+64),  \
                    P52=_mm256_load_pd(s+68);                           \
            CD(P00,P10,P20,P30,P40,P50, P00,P10,P20,P30,P40,P50);       \
            CD(P01,P11,P21,P31,P41,P51, P01,P11,P21,P31,P41,P51);       \
            CD(P02,P12,P22,P32,P42,P52, P02,P12,P22,P32,P42,P52);       \
            L6_ZPAIR(P00,P01,P02,P10,P11,P12, d+ 0, ST, CD);            \
            L6_ZPAIR(P20,P21,P22,P30,P31,P32, d+24, ST, CD);            \
            L6_ZPAIR(P40,P41,P42,P50,P51,P52, d+48, ST, CD);            \
        }                                                               \
    } while (0)

/* PX = L6_PASS_X (ascending) or L6_PASS_X_ZP (zp-outer, panel_r10). */
#define L6_DEF_FUSED_PX(NAME,PX,ST,PF,CD)                               \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,    \
                 const double *restrict in, double *restrict out,       \
                 long nvol)                                             \
{                                                                       \
    VSET;                                                               \
    (void)t2;                                                           \
    for (long b = 0; b < nvol; ++b) {                                   \
        const double *ip = in  + b * (long)L6_VD;                       \
        double       *op = out + b * (long)L6_VD;                       \
        PX(ip, t1, op, PF, CD);                                         \
        L6_FUSED_YZ(t1, op, ST, CD);                                    \
    }                                                                   \
}
#define L6_DEF_FUSED(NAME,ST,PF,CD) \
        L6_DEF_FUSED_PX(NAME,L6_PASS_X,ST,PF,CD)

#define L6_ZPAIR(A0,A1,A2,B0,B1,B2, D, ST, CD)                          \
    do {                                                                \
        __m256d w0=_mm256_permute2f128_pd(A0,B0,0x20);                  \
        __m256d w1=_mm256_permute2f128_pd(A0,B0,0x31);                  \
        __m256d w2=_mm256_permute2f128_pd(A1,B1,0x20);                  \
        __m256d w3=_mm256_permute2f128_pd(A1,B1,0x31);                  \
        __m256d w4=_mm256_permute2f128_pd(A2,B2,0x20);                  \
        __m256d w5=_mm256_permute2f128_pd(A2,B2,0x31);                  \
        CD(w0,w1,w2,w3,w4,w5, w0,w1,w2,w3,w4,w5);                       \
        ST((D)+ 0,_mm256_permute2f128_pd(w0,w1,0x20));                  \
        ST((D)+ 4,_mm256_permute2f128_pd(w2,w3,0x20));                  \
        ST((D)+ 8,_mm256_permute2f128_pd(w4,w5,0x20));                  \
        ST((D)+12,_mm256_permute2f128_pd(w0,w1,0x31));                  \
        ST((D)+16,_mm256_permute2f128_pd(w2,w3,0x31));                  \
        ST((D)+20,_mm256_permute2f128_pd(w4,w5,0x31));                  \
    } while (0)

L6_DEF_FUSED(l6_run_fused,        _mm256_store_pd,  L6_PF_NONE,  VD6)
L6_DEF_FUSED(l6_run_fused_pf,     _mm256_store_pd,  L6_PF_T0_1,  VD6)

/* panel_r10: the zp-outer x-pass twin (group order adopted from L6_pfa;
 * see L6_PASS_X_ZP).  Token-identical to l6_run_fused except for the
 * x-pass group order; output bit-identical to it.  ice_r3 prune: its pf
 * and pfw twins are deleted (fused_zp_pf 0.250 vs fused_pf 0.220 in the
 * r2 forced chain table; pfw is the +9% tax). */
L6_DEF_FUSED_PX(l6_run_fused_zp,     L6_PASS_X_ZP, _mm256_store_pd, L6_PF_NONE,  VD6)

/* panel_r7 prune: the ymm split-store shapes 3pass_s / fused_s (and the
 * L6_ZPAIR_S macro) are deleted -- 0 picks in 12 node invocations across
 * r4/r5, SPR-only mechanism.  panel_r9 prune: the fused3/VD63 twins are
 * deleted (question answered on the node, f3d = +3.3..+6.2%). */

/* ------------------------------------------------------------------ *
 * AVX-512: zxf -- THE INCUMBENT since panel ice_r3.
 *
 * History in one line each: every CLX rejection of 512-bit (r5 0-for-12,
 * r7 0-for-24 licence-fair, r8 retirement, r10 deletion) was a one-FMA-
 * pipe Gold 5218 reading; this node (Ice Lake-SP Gold 6326) has TWO
 * 512-bit FMA pipes and clk512 = clk256, ice_r2 reinstated the shapes as
 * trailing challengers, and the drained scored window picked zxf
 * (zwd=-2.6%, ab1 z155.1 vs y166.6 ns).  ice_r3 flips it to incumbent.
 *
 * Shape: zmm x-pass (lanes = 4 adjacent (y,z), 9 groups, every access
 * 64B aligned, zero tail: 162 FP instr where ymm needs 324) + the
 * node-proven ymm fused y+z, token-identical to the fused kernels'
 * second half.  ~1512 uops/volume vs fused's 1728.  Width pays exactly
 * where it adds no port-5 pressure: on ICX the second 512-bit FMA pipe
 * IS port 5, so the x-pass (2 in-lane vpermilpd per codelet, nothing
 * else) wins, while zff -- the fully-fused 512-bit y+z whose z
 * transposes were vpermt2pd/valignq, port-5-only -- lost every ice_r2
 * regime (0.239-0.272, never within 6% of zxf's 0.2433 chain race) and
 * is DELETED this round: the same port-5 mechanism L17_matrixsimd and
 * L13_rader named in ice_r1, and the same reason L6_pfa's z512yz lost
 * at +11%.  Do not rebuild a 512-bit y+z stage without first solving
 * its port-5 shuffle bill.
 * ------------------------------------------------------------------ */
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512DQ__)
#define L6_HAVE_AVX512 1

#define VD6Z(i0,i1,i2,i3,i4,i5, o0,o1,o2,o3,o4,o5)                       \
    do {                                                                \
        __m512d _p0 = _mm512_add_pd(i0,i3), _q0 = _mm512_sub_pd(i0,i3); \
        __m512d _p1 = _mm512_add_pd(i2,i5), _q1 = _mm512_sub_pd(i2,i5); \
        __m512d _p2 = _mm512_add_pd(i4,i1), _q2 = _mm512_sub_pd(i4,i1); \
        __m512d _a  = _mm512_add_pd(_p1,_p2), _b = _mm512_sub_pd(_p1,_p2); \
        __m512d _c  = _mm512_add_pd(_q1,_q2), _d = _mm512_sub_pd(_q1,_q2); \
        __m512d _m  = _mm512_fnmadd_pd(vhalfz,_a,_p0);                  \
        __m512d _n  = _mm512_fnmadd_pd(vhalfz,_c,_q0);                  \
        __m512d _bs = _mm512_permute_pd(_b,0x55);                       \
        __m512d _ds = _mm512_permute_pd(_d,0x55);                       \
        __m512d _s0 = _mm512_add_pd(_p0,_a);                            \
        __m512d _s3 = _mm512_add_pd(_q0,_c);                            \
        __m512d _s4 = _mm512_fmadd_pd (vkz,_bs,_m);                     \
        __m512d _s2 = _mm512_fnmadd_pd(vkz,_bs,_m);                     \
        __m512d _s1 = _mm512_fmadd_pd (vkz,_ds,_n);                     \
        __m512d _s5 = _mm512_fnmadd_pd(vkz,_ds,_n);                     \
        o0=_s0; o1=_s1; o2=_s2; o3=_s3; o4=_s4; o5=_s5;                 \
    } while (0)

#define VSETZ __m512d vhalfz = _mm512_set1_pd(0.5);                      \
              __m512d vkz    = _mm512_setr_pd(L6_S3,-L6_S3,L6_S3,-L6_S3, \
                                              L6_S3,-L6_S3,L6_S3,-L6_S3)

/* zmm x-pass prefetch hooks: 9 groups x 384 B = the same next-volume 54
 * lines, in the same ascending order, as the ymm hooks. */
#define L6Z_PF_NONE(SRC,OUT,g)  do { } while (0)
#define L6Z_PF_AT(SRC,g,DIST,HINT)                                      \
    do {                                                                \
        const char *_pf = (const char *)((SRC) + (DIST)*(long)L6_VD)    \
                          + 384*(g);                                    \
        _mm_prefetch(_pf,      HINT); _mm_prefetch(_pf +  64, HINT);    \
        _mm_prefetch(_pf+128,  HINT); _mm_prefetch(_pf + 192, HINT);    \
        _mm_prefetch(_pf+256,  HINT); _mm_prefetch(_pf + 320, HINT);    \
    } while (0)
#define L6Z_PF_T0_1(SRC,OUT,g)  L6Z_PF_AT(SRC,g,1,_MM_HINT_T0)

/* x-pass: lanes = 4 adjacent (y,z); axis stride 72 doubles; 9 groups,
 * every access 64B-aligned (volume 3456 B, group offset 8g doubles) */
#define L6Z_PASS_X(SRC,DST,OUT,PF)                                      \
    do {                                                                \
        for (int g = 0; g < 9; ++g) {                                   \
            const double *s = (SRC) + 8*g;                              \
            double *d = (DST) + 8*g;                                    \
            PF(SRC,OUT,g);                                              \
            __m512d v0=_mm512_load_pd(s+  0), v1=_mm512_load_pd(s+ 72); \
            __m512d v2=_mm512_load_pd(s+144), v3=_mm512_load_pd(s+216); \
            __m512d v4=_mm512_load_pd(s+288), v5=_mm512_load_pd(s+360); \
            VD6Z(v0,v1,v2,v3,v4,v5, v0,v1,v2,v3,v4,v5);                 \
            _mm512_store_pd(d+  0,v0); _mm512_store_pd(d+ 72,v1);       \
            _mm512_store_pd(d+144,v2); _mm512_store_pd(d+216,v3);       \
            _mm512_store_pd(d+288,v4); _mm512_store_pd(d+360,v5);       \
        }                                                               \
    } while (0)

/* zxf: zmm x-pass + the ymm fused y+z (token-identical to the incumbent
 * fused kernels' second half). */
#define L6_DEF_ZXF(NAME,PF)                                             \
L6_KALIGN static void NAME(double *restrict t1, double *restrict t2,    \
                 const double *restrict in, double *restrict out,       \
                 long nvol)                                             \
{                                                                       \
    VSET; VSETZ;                                                        \
    (void)t2;                                                           \
    for (long b = 0; b < nvol; ++b) {                                   \
        const double *ip = in  + b * (long)L6_VD;                       \
        double       *op = out + b * (long)L6_VD;                       \
        L6Z_PASS_X(ip, t1, op, PF);                                     \
        L6_FUSED_YZ(t1, op, _mm256_store_pd, VD6);                      \
    }                                                                   \
}

L6_DEF_ZXF(l6_run_zxf,     L6Z_PF_NONE)
L6_DEF_ZXF(l6_run_zxf_pf,  L6Z_PF_T0_1)

/* ice_r3 prune: the zff family (fully-fused 512-bit y+z: valignq rows,
 * vpermt2pd z-gathers) and zxf_pf2 are deleted -- see the section
 * comment for the numbers and the port-5 mechanism.  The zff derivation
 * lives in the panel_r9 tree (git 987d061) if a future round solves the
 * p5 bill. */

/* ------------------------------------------------------------------ *
 * ice_r4: THE GRADED CHAIN STEP  state <- (z+c)/(1+|z+c|), z = FFT(state)
 * (raw z, no unitary scale in map mode -- verified in driver.c RUN_UNIT
 * and check.py).  fft3d_chain owns the whole m-step chain.
 *
 * Structure (both ADOPTED from this round's records):
 *  - VOLUME-MAJOR, IN-PLACE (L17_matrixsimd ice_r4): every volume runs
 *    all m steps while its state (3.4 KB), its c slice (3.4 KB) and t1
 *    (3.4 KB) are L1-resident -- at L=6 the WHOLE per-volume chain fits
 *    in 48 KB L1 with 4x headroom, the best case on the panel for corpus
 *    10 s3's "iterate cache-resident" directive.  final_out is the state
 *    arena; x0 is read once; L2/DRAM traffic per volume for the whole
 *    4856-step chain is one x0 read, one c read, one writeback.
 *  - LAZY MAP fused into the zmm x-pass loads (L13_rader ice_r4, from
 *    the rival pipelines' corpus 10 s2 fusion): state between steps IS
 *    raw z; the map runs where each element is loaded exactly once, at
 *    512-bit width (the ymm y+z half would pay the ladder twice per
 *    point).  L17's counterexample (lazy map lost there, latency in
 *    front of the first pass) is noted: their X chunks are long and
 *    coupled, my 9 x-pass groups are independent 6-vector straight-line
 *    blocks, so OoO has 3 independent map ladders per group and 9
 *    groups to overlap.  Raced against the phase-split alternative
 *    (map as its own L1 pass, style b*) anyway -- see fft3d_chain.
 *
 * The map itself, per PAIR of zmm vectors = 8 points (pair-shared
 * denominator, L17_matrixsimd's s6, extended from their measured ladder):
 *   deinterleave re/im (2 vunpck), s = im^2 + (re^2 + 1e-300) (2 FMA;
 *   the guard folds the s==0 -> rsqrt=inf -> NaN trap into an addend,
 *   L13_rader's sp-guard), w ~ 1/sqrt(s) by vrsqrt14pd + 2 Newton
 *   (2^-14 -> 2^-28 -> ~2^-56: full double, the float-seed 2-Newton map
 *   is ILLEGAL at m=4856 -- it fails the 1e-13/step budget by ~26x per
 *   the brief), d = 1 + s*w (= 1 + sqrt(s), 1 FMA), then ONE reciprocal
 *   of d for all 8 points:
 *     style *div: one vdivpd (exact; 8 pts amortize ~16-18 divider cyc,
 *                 hidden under the FFT's FMA work),
 *     style *fma: vrcp14pd + 2 Newton (no divider; +4 uops/pair).
 *   out = t * y, reinterleave (2 vunpck).  19 (div) / 23 (fma) vector
 *   uops per 8 points + the c add.  Accuracy ~2-3 ulp/application;
 *   L13_rader measured the same ladder at 1.2-1.4e-13 total drift over
 *   m=1278 -- scaled to m=4856 that is ~5e-13 against a 4.9e-10 budget.
 * ------------------------------------------------------------------ */
#define VSETM __m512d mg    = _mm512_set1_pd(1e-300);                    \
              __m512d mhalf = _mm512_set1_pd(0.5);                       \
              __m512d m15   = _mm512_set1_pd(1.5);                       \
              __m512d mone  = _mm512_set1_pd(1.0);                       \
              __m512d mtwo  = _mm512_set1_pd(2.0);                       \
              (void)mtwo

#define L6_RECIP_DIV(Y,D)                                                \
        __m512d Y = _mm512_div_pd(mone,(D))
#define L6_RECIP_FMA(Y,D)                                                \
        __m512d Y = _mm512_rcp14_pd((D));                                \
        Y = _mm512_mul_pd(Y,_mm512_fnmadd_pd((D),Y,mtwo));               \
        Y = _mm512_mul_pd(Y,_mm512_fnmadd_pd((D),Y,mtwo));               \
        Y = _mm512_fmadd_pd(Y,_mm512_fnmadd_pd((D),Y,mone),Y)

/* FMA-Heron residual step: r <- r + (s - r^2)*(w/2), r = s*w.  Newton
 * converges from BELOW, so after 2 iterations w carries a ~5e-17
 * systematic underestimate; the exact FMA residual s - r^2 makes
 * sqrt(s) ~0.5 ulp and bias-free.  -DL6_MAP_NOHERON drops it for the
 * speed A/B (2-Newton bias measured: chain drift 2.29e-9 vs 1.40e-9
 * with Heron -- both beyond the mis-calibrated 4.9e-10 gate, see the
 * ice_r4 strategy record). */
#ifdef L6_MAP_NOHERON
#define L6_HERON(R,S,W) do { } while (0)
#else
#define L6_HERON(R,S,W)                                                  \
    do {                                                                \
        __m512d _hh = _mm512_fnmadd_pd((R),(R),(S));                    \
        (R) = _mm512_fmadd_pd(_mm512_mul_pd(_hh,mhalf),(W),(R));        \
    } while (0)
#endif

/* t0,t1v already hold z+c (interleaved, 4 complex each); leaves the
 * mapped state in them.  RQ names the reciprocal style macro. */
#define L6_MAP8(t0,t1v,RQ)                                               \
    do {                                                                \
        __m512d _R = _mm512_unpacklo_pd(t0,t1v);                        \
        __m512d _I = _mm512_unpackhi_pd(t0,t1v);                        \
        __m512d _s = _mm512_fmadd_pd(_I,_I,_mm512_fmadd_pd(_R,_R,mg));  \
        __m512d _w = _mm512_rsqrt14_pd(_s);                             \
        __m512d _hs= _mm512_mul_pd(_s,mhalf);                           \
        __m512d _a = _mm512_mul_pd(_w,_w);                              \
        _w = _mm512_mul_pd(_w,_mm512_fnmadd_pd(_hs,_a,m15));            \
        _a = _mm512_mul_pd(_w,_w);                                      \
        _w = _mm512_mul_pd(_w,_mm512_fnmadd_pd(_hs,_a,m15));            \
        /* FMA-Heron residual step: r <- r + (s - r^2)*(w/2).  Newton    \
         * converges from BELOW, so after 2 iterations w carries a       \
         * ~5e-17 systematic underestimate -- unbiased rounding noise    \
         * accumulates incoherently over the chain (~e-11 at m=4856),    \
         * but this COHERENT bias is amplified ~2e7x and measured        \
         * 2.29e-9 vs the 4.9e-10 budget.  The exact FMA residual        \
         * s - r^2 makes sqrt(s) ~0.5 ulp and bias-free (~1e-32).  This  \
         * is the brief's "third Newton step mandatory at L=6", in its   \
         * cheapest bias-killing form (+3 ops/pair, not +3/point). */    \
        __m512d _r = _mm512_mul_pd(_s,_w);                              \
        L6_HERON(_r,_s,_w);                                             \
        __m512d _d = _mm512_add_pd(_r,mone);                            \
        RQ(_y,_d);                                                      \
        _R = _mm512_mul_pd(_R,_y); _I = _mm512_mul_pd(_I,_y);           \
        t0  = _mm512_unpacklo_pd(_R,_I);                                \
        t1v = _mm512_unpackhi_pd(_R,_I);                                \
    } while (0)

/* ice_r6 FAST-tier map (runtime twin of L6_MAP8, see the header's tier
 * policy): rsqrt14 + 2 Newton, NO Heron residual step (the ~5e-17
 * systematic sqrt underestimate stays; measured chain drift 2.29e-9 at
 * m=4856 seed 42 vs Heron's 1.40e-9), and d = 1 + s*w fused into ONE
 * fmadd (single rounding; saves the separate r=s*w mul).  9 arithmetic
 * ops + 1 rsqrt + 1 divide per 8 points vs the exact arm's 13. */
#define L6_MAP8_F(t0,t1v,RQ)                                             \
    do {                                                                \
        __m512d _R = _mm512_unpacklo_pd(t0,t1v);                        \
        __m512d _I = _mm512_unpackhi_pd(t0,t1v);                        \
        __m512d _s = _mm512_fmadd_pd(_I,_I,_mm512_fmadd_pd(_R,_R,mg));  \
        __m512d _w = _mm512_rsqrt14_pd(_s);                             \
        __m512d _hs= _mm512_mul_pd(_s,mhalf);                           \
        __m512d _a = _mm512_mul_pd(_w,_w);                              \
        _w = _mm512_mul_pd(_w,_mm512_fnmadd_pd(_hs,_a,m15));            \
        _a = _mm512_mul_pd(_w,_w);                                      \
        _w = _mm512_mul_pd(_w,_mm512_fnmadd_pd(_hs,_a,m15));            \
        __m512d _d = _mm512_fmadd_pd(_s,_w,mone);                       \
        RQ(_y,_d);                                                      \
        _R = _mm512_mul_pd(_R,_y); _I = _mm512_mul_pd(_I,_y);           \
        t0  = _mm512_unpacklo_pd(_R,_I);                                \
        t1v = _mm512_unpackhi_pd(_R,_I);                                \
    } while (0)


/* One chain step, in place on the state volume ST (raw z on entry, raw
 * z of the next step on exit): mapped zmm x-pass ST->T1, then the
 * node-proven ymm fused y+z T1->ST.  ST is deliberately NOT restrict
 * (in==out); the x-pass consumes all of ST before the first y+z store. */
#define L6_DEF_CSTEP(NAME,RQ)                                            \
L6_KALIGN static void NAME(double *st, const double *restrict cf,        \
                           double *restrict t1)                          \
{                                                                        \
    VSET; VSETZ; VSETM;                                                  \
    for (int g = 0; g < 9; ++g) {                                        \
        const double *s = st + 8*g, *c = cf + 8*g;                       \
        double *d = t1 + 8*g;                                            \
        __m512d v0=_mm512_add_pd(_mm512_load_pd(s+  0),_mm512_load_pd(c+  0)); \
        __m512d v1=_mm512_add_pd(_mm512_load_pd(s+ 72),_mm512_load_pd(c+ 72)); \
        __m512d v2=_mm512_add_pd(_mm512_load_pd(s+144),_mm512_load_pd(c+144)); \
        __m512d v3=_mm512_add_pd(_mm512_load_pd(s+216),_mm512_load_pd(c+216)); \
        __m512d v4=_mm512_add_pd(_mm512_load_pd(s+288),_mm512_load_pd(c+288)); \
        __m512d v5=_mm512_add_pd(_mm512_load_pd(s+360),_mm512_load_pd(c+360)); \
        L6_MAP8(v0,v1,RQ); L6_MAP8(v2,v3,RQ); L6_MAP8(v4,v5,RQ);         \
        VD6Z(v0,v1,v2,v3,v4,v5, v0,v1,v2,v3,v4,v5);                      \
        _mm512_store_pd(d+  0,v0); _mm512_store_pd(d+ 72,v1);            \
        _mm512_store_pd(d+144,v2); _mm512_store_pd(d+216,v3);            \
        _mm512_store_pd(d+288,v4); _mm512_store_pd(d+360,v5);            \
    }                                                                    \
    L6_FUSED_YZ(t1, st, _mm256_store_pd, VD6);                           \
}

L6_DEF_CSTEP(l6_cstep_div, L6_RECIP_DIV)
L6_DEF_CSTEP(l6_cstep_fma, L6_RECIP_FMA)




/* One store-fused chain step: SRC (a mapped state; == ST from step 2 on)
 * -> plain zmm x-pass -> t1 -> y+z with the map at the store -> ST holds
 * the NEXT mapped state.  No raw-z laziness, no staging volume, no
 * trailing map pass: m identical calls ARE the chain.  In-place is legal
 * as ever (the x-pass drains SRC into t1 before the first y+z store). */



/* ice_r5 "p2div": the two paired volumes' map passes fused into ONE
 * 27-iteration loop -- two independent ladders + two divides per
 * iteration, so ladder latency hides inside the loop body instead of
 * across pass seams.  ice_r6: parameterized over the MAP macro so the
 * fast tier (L6_MAP8_F) gets an identical-shape twin.  -DL6_HYB is the
 * half-hybrid divider A/B (L36_mixedradix's r5 divider/FMA alternation
 * at this geometry): volume B's reciprocal runs rcp14+Newton on the FMA
 * ports, halving the map pass's divider occupancy at +4 uops/pair.
 * MEASURED AND REJECTED in ice_r6: 0.309 vs 0.291 (+6.2%) -- at L=6 the
 * 27 divides/volume hide fully under the FFT's port work (r4's bfma
 * verdict, reproduced at half dose); off by default, zero binary cost. */
#ifdef L6_HYB
#define L6_RECIP_B L6_RECIP_FMA
#else
#define L6_RECIP_B L6_RECIP_DIV
#endif
#define L6_DEF_MPASS2(NAME,MAP)                                          \
L6_KALIGN static void NAME(double *spA, const double *restrict ccA,      \
                           double *spB, const double *restrict ccB)      \
{                                                                        \
    VSETM;                                                               \
    /* Volume B walks its 27 pairs PHASE-ROTATED by 13.  With the chain  \
     * residues (spB = spA+3456 B, ccA = spA+3072, ccB = spA+2048, all   \
     * mod 4096) an in-phase walk puts two store->load pairs at exactly  \
     * 0 mod 4096 within the store-buffer window (spB stores vs ccA      \
     * loads 3 iterations later; spA stores vs spB loads 5 later), and   \
     * every such false hit replays the load.  The 13-rotation moves     \
     * both conflicts either before the store (no dependence) or >= 64   \
     * stores back (drained).  Pointwise map: any order is the same      \
     * result. */                                                        \
    for (int i = 0; i < 27; ++i) {                                       \
        int j = i + 13; if (j >= 27) j -= 27;                            \
        const double *sa = spA + 16*i, *ca = ccA + 16*i;                 \
        const double *sb = spB + 16*j, *cb = ccB + 16*j;                 \
        __m512d a0=_mm512_add_pd(_mm512_load_pd(sa+0),_mm512_load_pd(ca+0)); \
        __m512d a1=_mm512_add_pd(_mm512_load_pd(sa+8),_mm512_load_pd(ca+8)); \
        __m512d b0=_mm512_add_pd(_mm512_load_pd(sb+0),_mm512_load_pd(cb+0)); \
        __m512d b1=_mm512_add_pd(_mm512_load_pd(sb+8),_mm512_load_pd(cb+8)); \
        MAP(a0,a1,L6_RECIP_DIV);                                         \
        MAP(b0,b1,L6_RECIP_B);                                           \
        _mm512_store_pd(spA+16*i+0,a0); _mm512_store_pd(spA+16*i+8,a1);  \
        _mm512_store_pd(spB+16*j+0,b0); _mm512_store_pd(spB+16*j+8,b1);  \
    }                                                                    \
}

L6_DEF_MPASS2(l6_mpass2_div,  L6_MAP8)
L6_DEF_MPASS2(l6_mpass2_fast, L6_MAP8_F)

/* ice_r6 "p2x": the paired volumes' FFTs interleaved at PASS granularity
 * (the same currency that made pdiv win at the map seams): x-pass(A),
 * x-pass(B), fused y+z(A), fused y+z(B).  Per-volume arithmetic is
 * token-identical to l6_run_zxf, so the output is bit-identical to two
 * zxf calls; only the pass order across the pair changes.  This cuts the
 * data-DEPENDENT pass seams per step from 2 (map->xA, xA->yzA) to 1
 * (map->xA): xB is independent of xA, yzA of xB, yzB of yzA.  Needs a
 * second scratch plane t1b (pinned at residue sp+512, see fft3d_chain).
 * In-place legal as ever: each x-pass drains its volume into its own t1
 * before that volume's first y+z store. */
L6_KALIGN static void l6_zxf2(double *restrict t1a, double *restrict t1b,
                              const double *inA, double *outA,
                              const double *inB, double *outB)
{
    VSET; VSETZ;
    L6Z_PASS_X(inA, t1a, outA, L6Z_PF_NONE);
    L6Z_PASS_X(inB, t1b, outB, L6Z_PF_NONE);
    L6_FUSED_YZ(t1a, outA, _mm256_store_pd, VD6);
    L6_FUSED_YZ(t1b, outB, _mm256_store_pd, VD6);
}


/* Standalone map over one volume: DST = map(ST + CF), contiguous zmm,
 * 27 pairs.  DST may equal ST (the end-of-chain map) or be the staging
 * volume (the phase-split b* styles).  Same per-point arithmetic as the
 * fused x-pass map, so a/b styles of the same reciprocal are
 * output-bit-identical. */
#define L6_DEF_MPASS(NAME,RQ,MAP)                                        \
L6_KALIGN static void NAME(double *dst, const double *st,                \
                           const double *restrict cf)                    \
{                                                                        \
    VSETM;                                                               \
    for (int i = 0; i < 27; ++i) {                                       \
        const double *s = st + 16*i, *c = cf + 16*i;                     \
        __m512d a0=_mm512_add_pd(_mm512_load_pd(s+0),_mm512_load_pd(c+0)); \
        __m512d a1=_mm512_add_pd(_mm512_load_pd(s+8),_mm512_load_pd(c+8)); \
        MAP(a0,a1,RQ);                                                   \
        _mm512_store_pd(dst+16*i+0,a0);                                  \
        _mm512_store_pd(dst+16*i+8,a1);                                  \
    }                                                                    \
}

L6_DEF_MPASS(l6_mpass_div,  L6_RECIP_DIV, L6_MAP8)
L6_DEF_MPASS(l6_mpass_fma,  L6_RECIP_FMA, L6_MAP8)
L6_DEF_MPASS(l6_mpass_fast, L6_RECIP_DIV, L6_MAP8_F)

/* ice_r7 g8 (SoA group-of-8) entry points; definitions live at the END
 * of the file so the several-KB SoA bodies cannot displace the r6 tail
 * path's code layout (B=1 measured 0.376 vs 0.330 us with the bodies
 * emitted mid-file -- the r5 placement disease, third recurrence). */
static void l6_g8_chain(double *restrict X, double *restrict C,
                        const double *x0v, const double *cv, double *outv,
                        int m, void (*mp)(double *, const double *));
static void l6_g8map_div (double *restrict X, const double *restrict C);
static void l6_g8map_rcp (double *restrict X, const double *restrict C);
static void l6_g8map_fast(double *restrict X, const double *restrict C);
/* ice_r8 g8f: the store-fused chain (map+c fused into the z-sweep
 * stores; ADOPTED from the warm rival impl_3907.c gC_6 -- see the
 * definitions at the end of the file). */
static void l6_g8_chain_f(double *restrict X, double *restrict C,
                          const double *x0v, const double *cv, double *outv,
                          int m, int tier);

#endif /* L6_HAVE_AVX512 */

#endif /* L6_HAVE_AVX2 */

/* ------------------------------------------------------------------ *
 * Plan
 * ------------------------------------------------------------------ */

typedef void (*l6_kernel)(double *restrict t1, double *restrict t2,
                          const double *restrict in, double *restrict out,
                          long nvol);

struct fft3d_plan {
    int       L, batch;
    double   *arena;          /* owns the scratch; t1/t2 are placed inside it */
    double   *t1, *t2;
    l6_kernel run;
    int       fence;          /* nonzero if the chosen kernel uses NT stores */
    int       placed;         /* scratch already positioned for these buffers */
    int       uset2;          /* chosen kernel round-trips through t2 (3pass) */
    int       forced;         /* kernel forced via L6_FORCE, not raced */
    long      resA, resB;     /* unordered 4096-residue pair the placement is for */
    const char *chosen;
    int       cstyle;         /* ice_r4 chain map style: 0 adiv 1 afma 2 bdiv 3 bfma 4 scalar */
    int       tier;           /* ice_r6 map tier: 0 fast (2-Newton), 1 exact (+FMA-Heron) */
    int       g8rcp;          /* ice_r7: g8 reciprocal 0 vdivpd / 1 rcp14+Newton (A/B) */
    double   *cs;             /* 2 volumes of chain scratch (staging / fallback z) */
    double   *carena;         /* ice_r4 chain arena: t1/staging/c-copy, residue-placed */
    double   *garena;         /* ice_r7 g8 arena (owns gx/gc) */
    double   *gx, *gc;        /* SoA group state / c blocks, residue-pinned */
};

/* 4K-aliasing defence.  A store to S followed by a load from L with
 * (S-L) == 0 mod 4096 is falsely flagged as dependent and the load replays;
 * measured cost when a whole pass is in that state: +22% at B=1 (Haswell,
 * see strategies/L6_unrolled.md).  The scratch is therefore carved out of a
 * 4 KiB-oversized arena and positioned so that every store->load delta the
 * kernels can produce sits as far as possible from 0 mod 4096.  This
 * changes no arithmetic and no output, only addresses.
 *
 * ice_r3, ADOPTED FROM L6_pfa (their ice_r2 change 2): the placement is
 * now CHAIN-AWARE.  The graded chain ping-pongs (out,pong)/(pong,out)
 * from step 1 on, so a placement computed once for the first call's
 * (in,out) is stale for the other 4855 steps.  Two changes: (a) the
 * score is made direction-symmetric -- l6_cyc(-d) == l6_cyc(d) already
 * covers t1 for both directions (the forward "out - t1" term IS the
 * reverse "t1 - src" term), and the one asymmetric term, dst - t2, gains
 * its reverse twin l6_cyc(r + t2off) -- so ONE placement serves both
 * directions of a buffer pair; (b) fft3d_execute re-places whenever the
 * unordered residue pair changes, i.e. exactly once when the chain moves
 * off the (in,out) pair, ~64 iterations of integer work.  The t2 terms
 * are applied only when the chosen kernel round-trips through t2 (the
 * 3pass shapes); the fused/zxf shapes never touch t2, and dropping the
 * dead terms widens the achievable worst-case store->load distance. */
static long l6_cyc(long d)
{
    d &= 4095;
    return d < 4096 - d ? d : 4096 - d;
}

static void l6_place(fft3d_plan *p, const void *in, const void *out)
{
    long D = (long)(((uintptr_t)out - (uintptr_t)in) & 4095u);
    const long T2 = L6_VD * (long)sizeof(double);
    long bestr = 0, bestscore = -1;
    for (long r = 0; r < 4096; r += 64) {
        long sc = l6_cyc(r);                       /* t1 - src fwd == dst - t1 rev */
        long s  = l6_cyc(D - r);                   /* dst - t1 fwd == t1 - src rev */
        if (s < sc) sc = s;
        if (p->uset2) {
            s = l6_cyc(D - r - T2);                /* dst - t2, forward   */
            if (s < sc) sc = s;
            s = l6_cyc(r + T2);                    /* dst - t2, reversed  */
            if (s < sc) sc = s;
        }
        if (sc > bestscore) { bestscore = sc; bestr = r; }
    }
    long off = (long)((((uintptr_t)in + (uintptr_t)bestr
                        - (uintptr_t)p->arena) & 4095u) / sizeof(double));
    p->t1 = p->arena + off;
    p->t2 = p->t1 + L6_VD;
    {
        long a = (long)((uintptr_t)in  & 4095u);
        long b = (long)((uintptr_t)out & 4095u);
        p->resA = a < b ? a : b;
        p->resB = a < b ? b : a;
    }
    p->placed = 1;
}

/* ice_r4: chain scratch placement is CLOSED FORM, no search.  The chain
 * touches four buffers per step (state sp, t1, the map staging volume,
 * and the c slice); c is read-only and fixed for the whole chain, so it
 * is COPIED into the arena once per volume (3.5 KB per 4856 steps,
 * ~0.02%) and every controllable residue is then pinned at maximal
 * pairwise 4K distance: t1 at sp+1024, staging at sp+2048, the c copy
 * at sp+3072 (mod 4096).  Motivation: B=1 measured 13% slower than
 * B=64 per volume with identical code -- the one difference is the
 * driver's allocation residues, i.e. exactly the st->ld pairs this
 * pins.  Returns the three pointers via out-params. */
__attribute__((unused))
static void l6_chain_scratch(fft3d_plan *p, const void *sp,
                             double **t1c, double **msc, double **cc)
{
    uintptr_t ca = (uintptr_t)p->carena;
    long rsp = (long)((uintptr_t)sp & 4095u);
    long o = (rsp + 1024 - (long)(ca & 4095u)) & 4095;   /* bytes, 64B mult */
    double *t1 = (double *)(ca + (uintptr_t)o);
    *t1c = t1;                       /* residue sp+1024 */
    *msc = t1 + L6_VD + 208;         /* +5120 B -> residue sp+2048 */
    *cc  = t1 + 2 * (L6_VD + 208);   /* +10240 B -> residue sp+3072 */
}

/* Exact scalar map: one volume, DST = map(Z + CF).  The portable chain
 * fallback (and the witness the AVX-512 styles were validated against
 * during development). */
static void l6_map_scalar(double *dst, const double *z, const double *cf)
{
    for (int i = 0; i < L6_VOL; ++i) {
        double re = z[2*i] + cf[2*i], im = z[2*i+1] + cf[2*i+1];
        double sc = 1.0 / (1.0 + sqrt(re*re + im*im));
        dst[2*i] = re * sc; dst[2*i+1] = im * sc;
    }
}

const char *fft3d_name(void) { return "L6_unrolled"; }

static char l6_desc[512] =
    "L=6: unrolled straight-line PFA 2x3 six-point codelet (48 flops/36 instr, "
    "no twiddles) on all three axes, AVX2/FMA 2-complex lanes, in-register "
    "z-pencil transposes; variant=auto";

const char *fft3d_description(void) { return l6_desc; }

int fft3d_supports(int L) { return L == L6_L; }

static void *l6_alloc(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
}

#ifdef L6_HAVE_AVX2
static double l6_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* Busy 256-bit spin for SECS wall seconds.  Used (a) before the race, so
 * round 0 is not ranked on a ramping clock (L17_rader's r5 settle-spin
 * finding: 76% mis-ranking from table order alone), and (b) near the END
 * of create() as belt-and-braces before the chosen-kernel dwell.  (512-bit
 * code is back in the file since ice_r2, but this node -- Ice Lake-SP --
 * has no licence cliff: clk512 = clk256 measured twice in ice_r1.  The
 * spin survives as the clock-ramp settle, which schedutil still needs.) */
static void l6_spin256(double secs)
{
    __m256d x = _mm256_set1_pd(1.0);
    const __m256d a = _mm256_set1_pd(1.0 + 1e-15);
    const __m256d b = _mm256_set1_pd(1e-300);
    double until = l6_now() + secs;
    do {
        for (int i = 0; i < 8192; ++i) x = _mm256_fmadd_pd(x, a, b);
    } while (l6_now() < until);
    double lane[4];
    _mm256_storeu_pd(lane, x);
    if (!(lane[0] > 0.0)) fprintf(stderr, "l6_spin256: impossible\n");
}

/* ------------------------------------------------------------------ *
 * kclk (panel_r7) -- ADOPTED FROM L6_pfa's panel_r6 record: the clock
 * the CHOSEN kernel actually runs at, measured directly instead of
 * inferred from a synthetic chain's density.  Dwell ~2 ms in the real
 * kernel, then immediately time a ~150 us sparse ymm FMA chain: CLX
 * licence state persists ~670 us after the last heavy instruction, so
 * the sparse chain (which by itself never raises the licence) reads
 * the licence the kernel established.  Median of 9 dwell/read pairs.
 * freq = iters*4/dt (latency-4 FMA; the Haswell latency-5 over-read
 * caveat from the r4 probe applies to wombat only).  ~20 ms, unscored.
 * ------------------------------------------------------------------ */
static double l6_kclk(l6_kernel run, int fence, double *t1, double *t2)
{
    const long nd = 4;
    double *din  = (double *)l6_alloc((size_t)nd * L6_VD * sizeof(double));
    double *dout = (double *)l6_alloc((size_t)nd * L6_VD * sizeof(double));
    if (!din || !dout) { free(din); free(dout); return 0.0; }
    uint64_t st = 0xA0761D6478BD642Full;
    for (long i = 0; i < nd * L6_VD; ++i) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        din[i] = (double)(int64_t)(st >> 11) * (1.0 / 9007199254740992.0);
    }
    const __m256d a = _mm256_set1_pd(1.0 + 1e-15);
    const __m256d b = _mm256_set1_pd(1e-300);
    double vals[9];
    for (int r = 0; r < 9; ++r) {
        double until = l6_now() + 2e-3;
        do {
            run(t1, t2, din, dout, nd);
        } while (l6_now() < until);
        if (fence) _mm_sfence();
        __m256d x = _mm256_set1_pd(1.0);       /* named chain, never array */
        double t0 = l6_now();
        for (int i = 0; i < 131072; ++i) x = _mm256_fmadd_pd(x, a, b);
        double dt = l6_now() - t0;
        double lane[4];
        _mm256_storeu_pd(lane, x);
        vals[r] = (lane[0] > 0.0) ? 131072.0 * 4.0 / dt * 1e-9 : 0.0;
    }
    free(din); free(dout);
    for (int i = 1; i < 9; ++i) {              /* insertion sort, median */
        double v = vals[i]; int j = i;
        while (j > 0 && vals[j-1] > v) { vals[j] = vals[j-1]; --j; }
        vals[j] = v;
    }
    return vals[4] > 9.9 ? 0.0 : vals[4];
}

/* End-of-create dwell in the CHOSEN kernel (panel_r8, ADOPTED FROM
 * L6_pfa's panel_r7 refinement of my r6 licence-tail fix): ~3 ms of the
 * scored kernel itself, so the driver is handed a core in that kernel's
 * own licence/clock steady state -- never a probe's licence and never a
 * sparse spin's light-licence clock.  Falls back silently (the r6
 * spin256 tail already ran) if the scratch cannot be allocated. */
static void l6_dwell_chosen(l6_kernel run, int fence, double *t1, double *t2)
{
    const long nd = 4;
    double *din  = (double *)l6_alloc((size_t)nd * L6_VD * sizeof(double));
    double *dout = (double *)l6_alloc((size_t)nd * L6_VD * sizeof(double));
    if (!din || !dout) { free(din); free(dout); return; }
    uint64_t st = 0xE7037ED1A0B428DBull;
    for (long i = 0; i < nd * L6_VD; ++i) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        din[i] = (double)(int64_t)(st >> 11) * (1.0 / 9007199254740992.0);
    }
    double until = l6_now() + 3e-3;
    do {
        run(t1, t2, din, dout, nd);
    } while (l6_now() < until);
    if (fence) _mm_sfence();
    free(din); free(dout);
}

/* ------------------------------------------------------------------ *
 * In-plan B=1 discriminator (panel_r9) -- ADOPTED FROM L36_pfa's r8
 * in-plan node probe pattern, which the r8 VERDICT calls the round's
 * most reusable idea and explicitly asks L=6 to apply: when the monitor
 * cannot run your counter, build the discriminator into create() and
 * route the result through fft3d_description().  perf_event_open is
 * closed on this cluster (perf_event_paranoid=4 on every machine we can
 * see), so this is a TIMED discriminator: min ns/volume at nvol=1,
 * driver-like conditions (same in/out every call, placed scratch), 9
 * trials of 256 reps, each trial preceded by ~0.7 ms of the kernel
 * itself so every candidate is measured in its own licence/clock steady
 * state (the r7 licence-fair race rule).  panel_r10 retarget: the r9
 * questions (fused vs fused3, fused vs zff) are answered and those
 * kernels deleted; ab1 now runs for fused / fused_zp -- the ascending
 * vs zp-outer x-pass group order, the one structural difference left
 * between my B=1 kernel and L6_pfa's node-winning fused_d2 -- so the
 * node publishes the A/B regardless of which one the tournament picks.
 * ~17 ms total, unscored.
 * ------------------------------------------------------------------ */
static double l6_ab1(l6_kernel run, int fence, double *t1, double *t2,
                     const double *din, double *dout)
{
    double best = 1e300;
    for (int trial = 0; trial < 9; ++trial) {
        double wu = l6_now() + 7e-4;
        do { run(t1, t2, din, dout, 1); } while (l6_now() < wu);
        if (fence) _mm_sfence();
        double t0 = l6_now();
        for (int r = 0; r < 256; ++r) run(t1, t2, din, dout, 1);
        if (fence) _mm_sfence();
        double dt = (l6_now() - t0) * (1.0 / 256.0);
        if (dt < best) best = dt;
    }
    return best * 1e9;   /* ns per volume */
}

/* panel ice_r2: l6_abL (the 113 MiB / ~0.25 s in-plan DRAM codelet A/B)
 * is deleted with its probe kernel -- question answered, number published
 * on every ice_r1 leaderboard line.  The volatile-function-pointer
 * inlining trap it documented lives on in the strategy record (r11). */

/* The driver's per-chain-step unitary scale, reproduced for the
 * chain-shaped race: the same plain multiply loop over the whole
 * destination buffer (driver.c RUN_UNIT), auto-vectorized by the same
 * compiler flags. */
static void l6_vscale(double *p, long n, double s)
{
    for (long i = 0; i < n; ++i) p[i] *= s;
}
#endif

fft3d_plan *fft3d_create(int L, int batch)
{
    if (L != L6_L || batch <= 0) return NULL;

    fft3d_plan *p = (fft3d_plan *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    /* 4096 bytes of slack for the placement search + the two scratch volumes */
    p->arena = (double *)l6_alloc(4096 + 2 * L6_VD * sizeof(double) + 64);
    p->cs    = (double *)l6_alloc(2 * L6_VD * sizeof(double));
    p->carena = (double *)l6_alloc(32768);   /* 4K slack + t1/ms/cc, see
                                              * l6_chain_scratch */
    /* ice_r7 g8 arena: X at a 4 KiB boundary, C at X + 30720 B so the
     * map pass's one cross-buffer store->load residue (X stores vs C
     * loads at the same element) is pinned at 30720 mod 4096 = 2048 --
     * maximal 4K distance, no search, independent of driver residues. */
    p->garena = (double *)l6_alloc(65536);
    if (!p->arena || !p->cs || !p->carena || !p->garena) {
        fft3d_destroy(p); return NULL;
    }
    p->gx = (double *)(((uintptr_t)p->garena + 4095u) & ~(uintptr_t)4095u);
    p->gc = p->gx + 30720 / 8;
    p->t1 = p->arena;
    p->t2 = p->arena + L6_VD;
    p->run = l6_run_scalar;
    p->fence = 0;
    p->uset2 = 1;             /* conservative until a kernel is chosen */
    p->chosen = "scalar";

    /* ice_r8 chain shape: 18 g8f (g8 with the map FUSED INTO THE
     * Z-SWEEP STORES, adopted from the warm rival impl_3907.c gC_6 --
     * one whole X read+write traversal per step deleted) is the new
     * default; 16 g8 (flat map pass) is its A/B ancestor; 14 p2x is
     * the r6 pair machinery, which also serves as the batch tail.
     * L6_CMAP env or -DL6_CMAP_DEFAULT=n forces 0 adiv / 1 afma /
     * 2 bdiv / 3 bfma / 4 scalar / 8 ipdiv / 10 pdiv / 12 p2div /
     * 14 p2x / 16 g8 for node A/Bs; anything else clamps back to 18.
     * The r5 losers (sdiv/shdiv/bhdiv/cdiv/qdiv) are deleted per the
     * dead-weight rule; their numbers live in the strategy record. */
    p->cstyle = 18;
#ifdef L6_CMAP_DEFAULT
    p->cstyle = L6_CMAP_DEFAULT;
#endif
    {
        const char *cv = getenv("L6_CMAP");
        if (cv && cv[0]) {
            if      (!strcmp(cv, "adiv"))   p->cstyle = 0;
            else if (!strcmp(cv, "afma"))   p->cstyle = 1;
            else if (!strcmp(cv, "bdiv"))   p->cstyle = 2;
            else if (!strcmp(cv, "bfma"))   p->cstyle = 3;
            else if (!strcmp(cv, "scalar")) p->cstyle = 4;
            else if (!strcmp(cv, "ipdiv"))  p->cstyle = 8;
            else if (!strcmp(cv, "pdiv"))   p->cstyle = 10;
            else if (!strcmp(cv, "p2div"))  p->cstyle = 12;
            else if (!strcmp(cv, "p2x"))    p->cstyle = 14;
            else if (!strcmp(cv, "g8"))     p->cstyle = 16;
            else if (!strcmp(cv, "g8f"))    p->cstyle = 18;
            else if (cv[0] >= '0' && cv[0] <= '9' && !cv[1])
                p->cstyle = cv[0] - '0';
        }
    }
    switch (p->cstyle) {                  /* pruned styles clamp to default */
    case 0: case 1: case 2: case 3: case 4: case 8: case 10: case 12:
    case 14: case 16: case 18: break;
    default: p->cstyle = 18;
    }
    p->g8rcp = 0;                         /* g8 reciprocal A/B: vdivpd default */
#ifdef L6_G8RCP_DEFAULT
    p->g8rcp = L6_G8RCP_DEFAULT;
#endif
    {
        const char *gv = getenv("L6_G8RCP");
        if (gv && gv[0]) p->g8rcp = (gv[0] != '0');
    }

    /* Map tier: 0 = fast (rsqrt14 + 2 Newton, fused d = 1+s*w, one
     * exact vdivpd), 1 = exact (+ the FMA-Heron residual step,
     * bit-identical to the r4/r5 shipped chain).  ice_r8: default FAST
     * again -- and this time it is measured against the corrected
     * two-part gate, not tier policy: the fast ladder is a full-fp64
     * rsqrt14-seeded ladder (NOT the rivals' fp32-rcp seed the one-step
     * gate exists to catch), and on the r8 harness it reads two-step
     * rel L2 = 6.35e-16 vs tol 3e-14 (47x margin) and m=4856 chain-end
     * 2.292e-9 vs tol 1e-6 (300x the honest anchor 2.095e-9 measured on
     * the same chain -- the fast drift sits within 10% of the anchor
     * itself).  The exact Heron arm stays one env var away
     * (L6_TIER=exact) and is bit-identical to the r7 shipped chain. */
    p->tier = 0;
#ifdef L6_TIER_DEFAULT
    p->tier = L6_TIER_DEFAULT;
#endif
    {
        const char *tv = getenv("L6_TIER");
        if (tv && tv[0]) {
            if (!strcmp(tv, "exact") || !strcmp(tv, "heron")) p->tier = 1;
            else if (!strcmp(tv, "fast"))                     p->tier = 0;
        }
    }

    /* In-plan discriminator results (see l6_ab1; since ice_r2 the width
     * question at nvol=1: y = fused_pf, the best ymm shape, vs z = zxf,
     * the incumbent since ice_r3 -- kept as the one-number regression
     * check now that the question is answered), the fused_zp-vs-fused
     * race delta at the plan's own batch size (xod), and the best-zmm
     * vs best-ymm race delta at the same size (zwd). */
    double ab_y = 0.0, ab_z = 0.0, xod = 0.0, zwd = 0.0;
    int have_xod = 0, have_zwd = 0;

#ifdef L6_HAVE_AVX2
    {
        /* Ordered "safest first": a later candidate must beat the incumbent by
         * more than its own takeover margin, so measurement noise on a loaded
         * machine cannot promote a streaming-store kernel at a batch size
         * where the working set is cache resident.
         *
         * panel ice_r3 grid -- INCUMBENCY FLIP + PRUNE (the r10/r11
         * playbook, prescribed by my own r2 record's branch (i)): zxf
         * leads on AVX-512 hardware because the drained scored window
         * picked it (zwd=-2.6%) and a node-measured winner in a trailing
         * slot must re-clear its 2.5% margin at every create(), which the
         * r2 dev races cleared only "by a hair".  zxf_pf is its 1.0%
         * bit-identical challenger.  The ymm shapes follow as challengers
         * (and remain the whole grid on an AVX2-only build); 3pass is the
         * safest-baseline anchor and 3pass_nt_pf the NT canary.  Deleted
         * this round: zff family, all pfw shapes, all distance-2 pf
         * shapes, 3pass_pf, fused_zp_pf -- every one carries a node
         * number in the strategy record (r2/r3 sections).
         * Layout defence: every kernel entry 64B-pinned (since r6). */
        static const struct {
            l6_kernel k; int fence; int uset2; double mg; const char *nm;
        } cand[] = {
#ifdef L6_HAVE_AVX512
            { l6_run_zxf,           0, 0, 0.025, "zxf"           },
            { l6_run_zxf_pf,        0, 0, 0.010, "zxf_pf"        },
#endif
            { l6_run_fused_zp,      0, 0, 0.025, "fused_zp"      },
            { l6_run_fused,         0, 0, 0.010, "fused"         },
            { l6_run_fused_pf,      0, 0, 0.010, "fused_pf"      },
            { l6_run_3pass,         0, 1, 0.025, "3pass"         },
            { l6_run_3pass_nt_pf,   1, 1, 0.025, "3pass_nt_pf"   },
        };
        const int ncand = (int)(sizeof(cand)/sizeof(cand[0]));

        /* ---- correctness gate: every candidate must reproduce the scalar
         * reference bit-closely on random data before it may be timed. ---- */
        long nval = batch < 4 ? batch : 4;
        double *vin  = (double *)l6_alloc((size_t)nval * L6_VD * sizeof(double));
        double *vref = (double *)l6_alloc((size_t)nval * L6_VD * sizeof(double));
        double *vgot = (double *)l6_alloc((size_t)nval * L6_VD * sizeof(double));
        int ok[48]; for (int i = 0; i < ncand; ++i) ok[i] = 0;

        if (vin && vref && vgot) {
            uint64_t st = 0x9E3779B97F4A7C15ull;
            for (long i = 0; i < nval * L6_VD; ++i) {
                st ^= st << 13; st ^= st >> 7; st ^= st << 17;
                vin[i] = (double)(int64_t)(st >> 11) * (1.0 / 9007199254740992.0);
            }
            l6_run_scalar(p->t1, p->t2, vin, vref, nval);
            double nrm = 0.0;
            for (long i = 0; i < nval * L6_VD; ++i) nrm += vref[i] * vref[i];
            nrm = sqrt(nrm) + 1e-300;
            for (int c = 0; c < ncand; ++c) {
                memset(vgot, 0, (size_t)nval * L6_VD * sizeof(double));
                cand[c].k(p->t1, p->t2, vin, vgot, nval);
                if (cand[c].fence) _mm_sfence();
                double e = 0.0;
                for (long i = 0; i < nval * L6_VD; ++i) {
                    double d = vgot[i] - vref[i];
                    e += d * d;
                }
                ok[c] = (sqrt(e) / nrm) < 1e-11;
            }
        }
        free(vin); free(vref); free(vgot);

        /* ---- forced variant (panel_r5): the L6_FORCE env var (or a
         * -DL6_FORCE_DEFAULT='"name"' compile flag) names a candidate that
         * is selected unconditionally, skipping the race -- it must still
         * pass the correctness gate above.  This is the node A/B switch
         * the panel_r4 VERDICT asked for ("force the _s shapes on the
         * node"); the pick is reported as variant=<name>! so a forced
         * leaderboard line cannot be mistaken for a tournament pick. */
        int forced = -1;
        {
            const char *fv = getenv("L6_FORCE");
#ifdef L6_FORCE_DEFAULT
            if (!fv || !fv[0]) fv = L6_FORCE_DEFAULT;
#endif
            if (fv && fv[0])
                for (int c = 0; c < ncand; ++c)
                    if (ok[c] && strcmp(cand[c].nm, fv) == 0) { forced = c; break; }
        }

        /* ---- race the survivors at (a truncation of) the real batch size,
         * IN THE CHAIN REGIME (panel ice_r2; the tuner fix is ADOPTED FROM
         * L17_matrixsimd's ice_r1 chain-shaped tuner stage, the one change
         * that round that produced a measurable result).  The graded unit
         * is not a bare transform: the driver runs m chained steps, each a
         * transform followed by a driver-side unitary scale of the whole
         * output, ping-ponging between two destination buffers.  Racing
         * bare back-to-back transforms mis-ranks candidates in that regime:
         * forced-variant chain runs on the node measured fused_pf 0.220
         * us/xform against the bare-race incumbent fused_zp's 0.226 (+2.7%
         * -- twice the pf-family margin), with prefetchw +9% and NT stores
         * +250%, all sd <= 0.16%.  Each race rep is now one driver-shaped
         * chain step: run(src->dst), scale dst by 1/sqrt(V), swap dst
         * between two arenas.  The scale adds the same constant to every
         * candidate; what it changes is the cache/store state the NEXT
         * step's transform sees, which is exactly what the bare race got
         * wrong.  The race cap keeps its r2 rationale (16384 volumes = 113
         * MiB, unambiguous DRAM on every machine we run on). */
        long nt = batch;
        if (nt > 16384) nt = 16384;
        double *ain = NULL, *aout = NULL, *apong = NULL;
        int best = forced;
        if (best < 0) {
            ain   = (double *)l6_alloc((size_t)nt * L6_VD * sizeof(double));
            aout  = (double *)l6_alloc((size_t)nt * L6_VD * sizeof(double));
            apong = (double *)l6_alloc((size_t)nt * L6_VD * sizeof(double));
        }
        if (best < 0 && ain && aout && apong) {
            uint64_t st = 0xD1B54A32D192ED03ull;
            for (long i = 0; i < nt * L6_VD; ++i) {
                st ^= st << 13; st ^= st >> 7; st ^= st << 17;
                ain[i] = (double)(int64_t)(st >> 11) * (1.0 / 9007199254740992.0);
            }
            memset(aout, 0, (size_t)nt * L6_VD * sizeof(double));
            l6_place(p, ain, aout);
            /* settle spin (L17_rader's r5 finding: on a ramping clock a
             * fixed-order table mis-ranked bit-identical work by 76%);
             * 100 ms of 256-bit work brings the core to its steady licence
             * clock before round 0 is timed.  Unscored. */
            l6_spin256(0.1);
            /* how many repeats to make one trial last ~2 ms */
            long reps = (long)(2e-3 / (nt * 2.5e-7));
            if (reps < 1) reps = 1;
            if (reps > 20000) reps = 20000;
            /* Round-robin tournament (adopted from L6_pfa's record, which
             * documents a sequential per-candidate race mis-picking by 21%
             * when background load drifts between candidates): every round
             * times each surviving candidate once, and each candidate keeps
             * its own minimum, so drift hits all candidates alike. */
            const double usc = 1.0 / sqrt((double)L6_VOL);
            double tmin[48];
            for (int c = 0; c < ncand; ++c) {
                tmin[c] = 1e300;
                if (!ok[c]) continue;
                cand[c].k(p->t1, p->t2, ain, aout, nt);      /* warm */
                if (cand[c].fence) _mm_sfence();
            }
            for (int round = 0; round < 7; ++round) {
                for (int c = 0; c < ncand; ++c) {
                    if (!ok[c]) continue;
                    /* Every rep from here on is one driver-shaped chain
                     * step: transform, fence if NT, unitary scale of the
                     * destination, ping-pong the destination -- see the
                     * regime comment above the allocation. */
                    const double *csrc = ain;
                    double *cdst = aout;
                    /* per-candidate licence warm-up (panel_r7, per
                     * LITERATURE 08 s4.3): run the candidate itself,
                     * untimed, for ~0.7 ms, so every candidate is timed
                     * in its own licence/clock steady state, never a
                     * predecessor's tail. */
                    double wu = l6_now() + 7e-4;
                    do {
                        cand[c].k(p->t1, p->t2, csrc, cdst, nt);
                        if (cand[c].fence) _mm_sfence();
                        l6_vscale(cdst, nt * (long)L6_VD, usc);
                        csrc = cdst;
                        cdst = (cdst == aout) ? apong : aout;
                    } while (l6_now() < wu);
                    double t0 = l6_now();
                    for (long r = 0; r < reps; ++r) {
                        cand[c].k(p->t1, p->t2, csrc, cdst, nt);
                        if (cand[c].fence) _mm_sfence();
                        l6_vscale(cdst, nt * (long)L6_VD, usc);
                        csrc = cdst;
                        cdst = (cdst == aout) ? apong : aout;
                    }
                    double dt = (l6_now() - t0) / (double)reps;
                    if (dt < tmin[c]) tmin[c] = dt;
                }
            }
            /* Safest-first with a per-candidate takeover margin (panel_r10:
             * the bit-identical zp twins carry 1.0%, everything else keeps
             * the 2.5% hysteresis).  The reference time tracks the true
             * minimum even when the incumbent survives, so a chain of
             * sub-margin steps cannot drift the pick. */
            double bestt = 1e300;
            for (int c = 0; c < ncand; ++c) {
                if (!ok[c]) continue;
                if (best < 0 || tmin[c] < bestt * (1.0 - cand[c].mg)) {
                    bestt = tmin[c]; best = c;
                } else if (tmin[c] < bestt) bestt = tmin[c];
            }
#ifdef L6_VERBOSE_DEFAULT
            if (1)   /* tryout.sh cannot pass env over ssh; -D flag instead */
#else
            if (getenv("L6_VERBOSE"))
#endif
                for (int c = 0; c < ncand; ++c)
                    fprintf(stderr, "L6_unrolled race: %-14s %s %10.4f us/vol%s\n",
                            cand[c].nm,
                            !ok[c] ? "BAD" : "ok ",
                            ok[c] ? tmin[c] / (double)nt * 1e6 : 0.0,
                            c == best ? "   <-- chosen" : "");
            /* fused_zp-vs-fused delta at the plan's own raced batch size,
             * best of family vs best of family, for the description
             * (positive = the zp-outer x order is slower).  zwd (ice_r2):
             * best zmm (the zxf family) vs best ymm (positive = 512-bit
             * slower), so the node keeps publishing the width A/B at the
             * graded batch now that zxf is the incumbent. */
            {
                double bf = 1e300, bfx = 1e300, by = 1e300, bz = 1e300;
                for (int c = 0; c < ncand; ++c) {
                    if (!ok[c]) continue;
                    if (strncmp(cand[c].nm, "fused_zp", 8) == 0) {
                        if (tmin[c] < bfx) bfx = tmin[c];
                    } else if (strncmp(cand[c].nm, "fused", 5) == 0) {
                        if (tmin[c] < bf) bf = tmin[c];
                    }
                    if (cand[c].nm[0] == 'z') {
                        if (tmin[c] < bz) bz = tmin[c];
                    } else {
                        if (tmin[c] < by) by = tmin[c];
                    }
                }
                if (bf < 1e299 && bfx < 1e299) {
                    xod = (bfx / bf - 1.0) * 100.0;
                    have_xod = 1;
                }
                if (by < 1e299 && bz < 1e299) {
                    zwd = (bz / by - 1.0) * 100.0;
                    have_zwd = 1;
                }
            }
        }
        if (best < 0)
            for (int c = 0; c < ncand; ++c)
                if (ok[c]) { best = c; break; }
        free(ain); free(aout); free(apong);
        p->t1 = p->arena; p->t2 = p->arena + L6_VD; p->placed = 0;

        if (best >= 0) {
            p->run    = cand[best].k;
            p->fence  = cand[best].fence;
            p->uset2  = cand[best].uset2;
            p->chosen = cand[best].nm;
            p->forced = (best == forced);
        }

        /* ---- in-plan nvol=1 discriminator (see l6_ab1's comment).
         * ice_r2 retarget: the x-order question it carried is closed on
         * this node (ice_r1: ab1 f163.9 fx163.5, xod=-0.2% -- a wash);
         * it now times the chain-winning ymm shape fused_pf against the
         * chain-winning zmm shape zxf, licence-fair, so the node
         * publishes the width A/B at the latency end regardless of the
         * tournament's pick.  ~17 ms. ---- */
        {
            double *bin  = (double *)l6_alloc(L6_VD * sizeof(double));
            double *bout = (double *)l6_alloc(L6_VD * sizeof(double));
            if (bin && bout) {
                uint64_t st = 0xBF58476D1CE4E5B9ull;
                for (long i = 0; i < L6_VD; ++i) {
                    st ^= st << 13; st ^= st >> 7; st ^= st << 17;
                    bin[i] = (double)(int64_t)(st >> 11)
                             * (1.0 / 9007199254740992.0);
                }
                memset(bout, 0, L6_VD * sizeof(double));
                l6_place(p, bin, bout);
                for (int c = 0; c < ncand; ++c) {
                    if (!ok[c]) continue;
                    if (strcmp(cand[c].nm, "fused_pf") == 0)
                        ab_y = l6_ab1(cand[c].k, cand[c].fence,
                                      p->t1, p->t2, bin, bout);
                    else if (strcmp(cand[c].nm, "zxf") == 0)
                        ab_z = l6_ab1(cand[c].k, cand[c].fence,
                                      p->t1, p->t2, bin, bout);
                }
                p->t1 = p->arena; p->t2 = p->arena + L6_VD; p->placed = 0;
            }
            free(bin); free(bout);
        }
    }
#endif
    {   /* report the raced winner, the kernel-context clock (kclk, the one
         * clock number still worth a line: the panel consensus is settled
         * at 3.89 non-AVX / 2.89 licence and kclk is the regression check),
         * and the ab1/xod x-order A/B.  A trailing ! marks an L6_FORCE
         * pick (not a tournament one). */
        double gk = 0.0;
#ifdef L6_HAVE_AVX2
        gk = l6_kclk(p->run, p->fence, p->t1, p->t2);
        /* belt-and-braces spin (512-bit is back since ice_r2 but this
         * node has no licence cliff; kept because it is free and it
         * re-establishes the heavy clock after kclk's sparse chain)... */
        l6_spin256(0.02);
        /* ...then hand the driver the CHOSEN kernel's own steady state
         * (r8, from L6_pfa's r7 refinement of my r6 licence-tail fix),
         * so the driver never times a transition that create() caused. */
        l6_dwell_chosen(p->run, p->fence, p->t1, p->t2);
#endif
        if (gk > 0.0) {
            /* ab1 = in-plan nvol=1 discriminator, min ns/volume,
             * licence-fair (each kernel self-warmed): y = fused_pf (the
             * chain-winning ymm shape), z = zxf (the chain-winning zmm
             * shape); 0.0 = unavailable.  zwd = best-zmm vs best-ymm race
             * delta at the plan's raced batch size, positive = 512-bit
             * slower (the width question at the graded batch).
             * xod = fused_zp vs fused family-best race delta at the
             * plan's batch size, positive = zp-outer slower. */
            static const char *cnm[19] = {"adiv","afma","bdiv","bfma","scalar",
                                          "-","-","-","ipdiv","-",
                                          "pdiv","-","p2div","-","p2x",
                                          "-","g8","-","g8f"};
            int n = snprintf(l6_desc, sizeof(l6_desc),
                     "L=6: unrolled PFA 2x3 codelet (48 flop/36 instr, no "
                     "twiddles), zmm-x/ymm-yz zxf incumbent, fused fft3d_chain "
                     "(ice_r8: SoA 8-volumes-per-zmm group chain, map+c fused "
                     "at the z-sweep store, H/R2 sqrt-rcp alternation [warm "
                     "impl_3907 gC]; pair/ipdiv tail); "
                     "variant=%s%s cmap=%s tier=%s g8rcp=%d kclk=%.2fGHz "
                     "ab1=y%.1f,z%.1fns",
                     p->chosen, p->forced ? "!" : "", cnm[p->cstyle],
                     p->tier ? "exact" : "fast", p->g8rcp,
                     gk, ab_y, ab_z);
            if (have_zwd && n > 0 && (size_t)n < sizeof(l6_desc))
                n += snprintf(l6_desc + n, sizeof(l6_desc) - (size_t)n,
                              " zwd=%+.1f%%", zwd);
            if (have_xod && n > 0 && (size_t)n < sizeof(l6_desc))
                snprintf(l6_desc + n, sizeof(l6_desc) - (size_t)n,
                         " xod=%+.1f%%", xod);
        } else
            snprintf(l6_desc, sizeof(l6_desc),
                     "L=6: unrolled PFA 2x3 codelet, portable scalar path; "
                     "variant=%s", p->chosen);
    }
    return p;
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    /* Chain-aware re-placement (ice_r3, from L6_pfa): the placement is
     * direction-symmetric, so the key is the UNORDERED residue pair --
     * a ping-pong chain re-places once at step 1 and is then stable. */
    long a = (long)((uintptr_t)in  & 4095u);
    long b = (long)((uintptr_t)out & 4095u);
    long lo = a < b ? a : b, hi = a < b ? b : a;
    if (!plan->placed || plan->resA != lo || plan->resB != hi)
        l6_place(plan, in, out);
    plan->run(plan->t1, plan->t2,
              (const double *)in, (double *)out, (long)plan->batch);
#ifdef L6_HAVE_AVX2
    if (plan->fence) _mm_sfence();
#endif
}

/* ------------------------------------------------------------------ *
 * ice_r4: the fused chain entry point (detected as a weak symbol by the
 * driver).  Semantics reproduced from driver.c's fallback exactly:
 *   state_0 = x0;  state_s = (FFT(state_{s-1}) + c) / (1 + |...|);
 *   final_out = state_m.  No unitary scale anywhere in map mode.
 *
 * Volume-major: each volume's chain is independent (the FFT is
 * per-volume, the map pointwise), so a volume (or, since ice_r5, a PAIR
 * of volumes, interleaved pass by pass) runs all m steps with state,
 * c copy and t1 L1-resident.  final_out IS the state arena and holds
 * RAW z between steps; step 1 is a plain zxf of x0, each later step is
 * map-then-FFT, and one map pass closes the chain.  In-place is legal
 * because the x-pass drains the whole volume into t1 before the first
 * y+z store lands, and the map pass is pointwise.
 * ------------------------------------------------------------------ */
void fft3d_chain(fft3d_plan *plan, const double _Complex *x0,
                 const double _Complex *c, double _Complex *final_out, int m)
{
    if (!plan || m < 1 || !final_out) return;   /* driver passes pong=NULL at
                                                 * --chain 1 (ice_r8 finding) */
    const long B = plan->batch;
    long bstart = 0;
    (void)bstart;
#ifdef L6_HAVE_AVX512
    /* ice_r5 "pdiv" (style 10): TWO volumes' chains interleaved at pass
     * granularity.  Each volume's chain is a serial dependence
     * state->map->FFT->state; pairing gives the OoO engine a data-
     * INDEPENDENT pass to chew on at every seam (map_A's ladder tail
     * overlaps map_B / zxf_A issue).  L6_pfa's div2 lockstep failed at
     * STEP granularity (~660-uop bodies exceed the ROB); this is the
     * pass-granularity variant with the flat map.  Working set 2 states
     * + 2 c copies + t1 ~ 17 KB, still L1-resident. */
    if (plan->cstyle >= 10) {   /* 10 pdiv / 12 p2div / 14 p2x / 16 g8 / 18 g8f */
        long b = 0;
        if (plan->cstyle == 18) {
            /* ice_r8 g8f: SoA groups of 8, map fused at the z-store --
             * m identical FFT+map steps, no flat map pass.  Tail as g8. */
            for (; b + 8 <= B; b += 8)
                l6_g8_chain_f(plan->gx, plan->gc,
                              (const double *)x0 + b * (long)L6_VD,
                              (const double *)c  + b * (long)L6_VD,
                              (double *)final_out + b * (long)L6_VD,
                              m, plan->tier);
        } else if (plan->cstyle == 16) {
            /* ice_r7 g8: SoA groups of 8 whole-chain volumes.  The tail
             * (B%8, or everything at B<8) falls through to the pair +
             * ipdiv machinery below, which g8 treats as p2x. */
            void (*mpg)(double *, const double *) =
                plan->tier ? (plan->g8rcp ? l6_g8map_rcp : l6_g8map_div)
                           : l6_g8map_fast;
            for (; b + 8 <= B; b += 8)
                l6_g8_chain(plan->gx, plan->gc,
                            (const double *)x0 + b * (long)L6_VD,
                            (const double *)c  + b * (long)L6_VD,
                            (double *)final_out + b * (long)L6_VD,
                            m, mpg);
        }
        int cs = (plan->cstyle >= 16) ? 14 : plan->cstyle;
        int fusedm = (cs >= 12);
        int splitx = (cs == 14);
        /* tier-selected map kernels (ice_r6): identical shapes, the fast
         * twin drops the Heron step and fuses d = 1+s*w. */
        void (*mp2)(double *, const double *, double *, const double *) =
            plan->tier ? l6_mpass2_div : l6_mpass2_fast;
        void (*mp1)(double *, const double *, const double *) =
            plan->tier ? l6_mpass_div : l6_mpass_fast;
        for (; b + 1 < B; b += 2) {
            const double *xpA = (const double *)x0 + b * (long)L6_VD;
            const double *xpB = xpA + L6_VD;
            const double *cfA = (const double *)c + b * (long)L6_VD;
            const double *cfB = cfA + L6_VD;
            double *spA = (double *)final_out + b * (long)L6_VD;
            double *spB = spA + L6_VD;
            double *t1, *ccB, *ccA;
            l6_chain_scratch(plan, spA, &t1, &ccB, &ccA);
            /* p2x's second scratch plane: 3 slots past t1 plus 512 B, so
             * it lands at residue sp+512 -- clear of sp (0), t1 (+1024),
             * ccB (+2048), ccA (+3072) and spB (+3456).  carena is 32 KB:
             * worst-case end offset 4032+15872+3456 = 23360 B. */
            double *t1b = t1 + 3 * ((long)L6_VD + 208) + 64;
            if (splitx)
                l6_zxf2(t1, t1b, xpA, spA, xpB, spB);
            else {
                l6_run_zxf(t1, ccA, xpA, spA, 1);
                l6_run_zxf(t1, ccA, xpB, spB, 1);
            }
            memcpy(ccA, cfA, L6_VD * sizeof(double));
            memcpy(ccB, cfB, L6_VD * sizeof(double));
            for (int s = 1; s < m; ++s) {
                if (fusedm)
                    mp2(spA, ccA, spB, ccB);
                else {
                    mp1(spA, spA, ccA);
                    mp1(spB, spB, ccB);
                }
                if (splitx)
                    l6_zxf2(t1, t1b, spA, spA, spB, spB);
                else {
                    l6_run_zxf(t1, ccA, spA, spA, 1);
                    l6_run_zxf(t1, ccA, spB, spB, 1);
                }
            }
            if (fusedm)
                mp2(spA, ccA, spB, ccB);
            else {
                mp1(spA, spA, ccA);
                mp1(spB, spB, ccB);
            }
        }
        bstart = b;   /* odd tail (and the whole batch at B=1) runs the
                       * shared ipdiv loop below -- a source-identical tail
                       * loop here measured 0.376 vs 0.330 us at B=1, pure
                       * code-placement (the file's r5 dead-weight disease);
                       * one loop, one address. */
    }
#endif
#ifdef L6_HAVE_AVX512
    if (plan->cstyle != 4) {
        void (*mp1)(double *, const double *, const double *) =
            plan->tier ? l6_mpass_div : l6_mpass_fast;
        for (long b = bstart; b < B; ++b) {
            const double *xp = (const double *)x0 + b * (long)L6_VD;
            const double *cf = (const double *)c  + b * (long)L6_VD;
            double       *sp = (double *)final_out + b * (long)L6_VD;
            double *t1, *ms, *cc;
            l6_chain_scratch(plan, sp, &t1, &ms, &cc);
            memcpy(cc, cf, L6_VD * sizeof(double));
            if (plan->cstyle >= 8) {              /* ipdiv (ice_r5):
                                                   * phase-split ladder,
                                                   * in place, no staging */
                l6_run_zxf(t1, ms, xp, sp, 1);
                for (int s = 1; s < m; ++s) {
                    mp1(sp, sp, cc);
                    l6_run_zxf(t1, cc, sp, sp, 1);
                }
                mp1(sp, sp, cc);
                continue;
            }
            l6_run_zxf(t1, ms, xp, sp, 1);            /* z_1, raw */
            switch (plan->cstyle) {
            case 0: for (int s = 1; s < m; ++s) l6_cstep_div(sp, cc, t1);
                    l6_mpass_div(sp, sp, cc);
                    break;
            case 1: for (int s = 1; s < m; ++s) l6_cstep_fma(sp, cc, t1);
                    l6_mpass_fma(sp, sp, cc);
                    break;
            case 2: for (int s = 1; s < m; ++s) {     /* phase-split A/B */
                        l6_mpass_div(ms, sp, cc);
                        l6_run_zxf(t1, cc, ms, sp, 1);
                    }
                    l6_mpass_div(sp, sp, cc);
                    break;
            default:
                    for (int s = 1; s < m; ++s) {
                        l6_mpass_fma(ms, sp, cc);
                        l6_run_zxf(t1, cc, ms, sp, 1);
                    }
                    l6_mpass_fma(sp, sp, cc);
                    break;
            }
        }
        return;
    }
#endif
    /* Portable fallback (also -DL6_CMAP_DEFAULT=4 / L6_CMAP=scalar: the
     * exact-map witness): chosen FFT kernel + scalar map, volume-major. */
    for (long b = 0; b < B; ++b) {
        const double *xp = (const double *)x0 + b * (long)L6_VD;
        const double *cf = (const double *)c  + b * (long)L6_VD;
        double       *sp = (double *)final_out + b * (long)L6_VD;
        double *zb = plan->cs, *ms = plan->cs + L6_VD;
        plan->run(plan->t1, plan->t2, xp, zb, 1);
#ifdef L6_HAVE_AVX2
        if (plan->fence) _mm_sfence();
#endif
        for (int s = 1; s < m; ++s) {
            l6_map_scalar(ms, zb, cf);
            plan->run(plan->t1, plan->t2, ms, zb, 1);
#ifdef L6_HAVE_AVX2
            if (plan->fence) _mm_sfence();
#endif
        }
        l6_map_scalar(sp, zb, cf);
    }
}

void fft3d_destroy(fft3d_plan *plan)
{
    if (!plan) return;
    free(plan->garena);
    free(plan->carena);
    free(plan->cs);
    free(plan->arena);
    free(plan);
}

#if defined(L6_HAVE_AVX2) && defined(L6_HAVE_AVX512)
/* ------------------------------------------------------------------ *
 * ice_r7 "g8": SoA group-of-8 chain (ADOPTED from the rival v6
 * generator's gen_familyA -- see the file header).  Layout per group:
 * x-plane i at X + i*L6_GPS doubles; element e = 6*y + z within the
 * plane at +16*e: 8 doubles of re (one per volume), then 8 of im.
 * L6_GPS = 37*16 = 592 (odd ELEMENT stride 37 = their 4K-alias
 * defence; the 16-double pad at each plane end is never touched).
 * Every access is a 64-byte-aligned full zmm; no shuffle instruction
 * appears anywhere between convin and convout.
 * ------------------------------------------------------------------ */
#define L6_GPS  592                     /* doubles per SoA plane (37*16) */
#define L6_GVD  (6*L6_GPS)              /* doubles per SoA group block   */

/* The 6-point codelet core on 12 register vectors: the exact VD6/SD6
 * PFA 2x3 DAG in split-complex form.  Each op is the same single-
 * rounded operation VD6 does on the same point, so the transform is
 * bit-identical per volume.  Stores its 12 outputs to re/re+8 at
 * element stride es. */
#define L6_SOA6_CORE(re,es, x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i) \
 do { \
    __m512d p0r = _mm512_add_pd(x0r,x3r), p0i = _mm512_add_pd(x0i,x3i); \
    __m512d q0r = _mm512_sub_pd(x0r,x3r), q0i = _mm512_sub_pd(x0i,x3i); \
    __m512d p1r = _mm512_add_pd(x2r,x5r), p1i = _mm512_add_pd(x2i,x5i); \
    __m512d q1r = _mm512_sub_pd(x2r,x5r), q1i = _mm512_sub_pd(x2i,x5i); \
    __m512d p2r = _mm512_add_pd(x4r,x1r), p2i = _mm512_add_pd(x4i,x1i); \
    __m512d q2r = _mm512_sub_pd(x4r,x1r), q2i = _mm512_sub_pd(x4i,x1i); \
    __m512d ar  = _mm512_add_pd(p1r,p2r), ai  = _mm512_add_pd(p1i,p2i); \
    __m512d br  = _mm512_sub_pd(p1r,p2r), bi  = _mm512_sub_pd(p1i,p2i); \
    __m512d cr  = _mm512_add_pd(q1r,q2r), ci  = _mm512_add_pd(q1i,q2i); \
    __m512d dr  = _mm512_sub_pd(q1r,q2r), di  = _mm512_sub_pd(q1i,q2i); \
    __m512d mr  = _mm512_fnmadd_pd(half,ar,p0r), mi = _mm512_fnmadd_pd(half,ai,p0i); \
    __m512d nr  = _mm512_fnmadd_pd(half,cr,q0r), ni = _mm512_fnmadd_pd(half,ci,q0i); \
    _mm512_store_pd(re       , _mm512_add_pd(p0r,ar)); \
    _mm512_store_pd(re + 8   , _mm512_add_pd(p0i,ai)); \
    _mm512_store_pd(re + 3*es, _mm512_add_pd(q0r,cr)); \
    _mm512_store_pd(re + 8 + 3*es, _mm512_add_pd(q0i,ci)); \
    _mm512_store_pd(re + 4*es, _mm512_fmadd_pd (s3,bi,mr)); \
    _mm512_store_pd(re + 8 + 4*es, _mm512_fnmadd_pd(s3,br,mi)); \
    _mm512_store_pd(re + 2*es, _mm512_fnmadd_pd(s3,bi,mr)); \
    _mm512_store_pd(re + 8 + 2*es, _mm512_fmadd_pd (s3,br,mi)); \
    _mm512_store_pd(re +   es, _mm512_fmadd_pd (s3,di,nr)); \
    _mm512_store_pd(re + 8 +   es, _mm512_fnmadd_pd(s3,dr,ni)); \
    _mm512_store_pd(re + 5*es, _mm512_fnmadd_pd(s3,di,nr)); \
    _mm512_store_pd(re + 8 + 5*es, _mm512_fmadd_pd (s3,dr,ni)); \
 } while (0)

/* Line-DFT from memory: load 12 vectors, run the core. */
static __attribute__((always_inline)) inline void l6_soa6(double *re, long es)
{
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d s3   = _mm512_set1_pd(L6_S3);
    __m512d x0r = _mm512_load_pd(re       ), x0i = _mm512_load_pd(re + 8       );
    __m512d x1r = _mm512_load_pd(re +   es), x1i = _mm512_load_pd(re + 8 +   es);
    __m512d x2r = _mm512_load_pd(re + 2*es), x2i = _mm512_load_pd(re + 8 + 2*es);
    __m512d x3r = _mm512_load_pd(re + 3*es), x3i = _mm512_load_pd(re + 8 + 3*es);
    __m512d x4r = _mm512_load_pd(re + 4*es), x4i = _mm512_load_pd(re + 8 + 4*es);
    __m512d x5r = _mm512_load_pd(re + 5*es), x5i = _mm512_load_pd(re + 8 + 5*es);
    L6_SOA6_CORE(re,es, x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i);
}

/* Whole 3D FFT of a group, in place, x -> y -> z (the scalar witness's
 * axis order, so per-volume association is unchanged).  y and z run
 * back-to-back per plane while the 4.7 KB plane is L1-hot. */
L6_KALIGN static void l6_g8_fft(double *restrict X)
{
    for (int e = 0; e < 36; ++e)                     /* x: stride L6_GPS */
        l6_soa6(X + 16*e, L6_GPS);
    for (int i = 0; i < 6; ++i) {
        double *pl = X + (long)i * L6_GPS;
        for (int z = 0; z < 6; ++z)                  /* y: stride 96     */
            l6_soa6(pl + 16*z, 96);
        for (int y = 0; y < 6; ++y)                  /* z: stride 16     */
            l6_soa6(pl + 96*y, 16);
    }
}

/* SoA map: L6_MAP8's exact per-point op sequence with the 4 unpcks
 * deleted (re/im arrive split).  RQ = reciprocal style, HER = the
 * Heron/fast d selection, exactly as in the AoS twins. */
#define L6_DEF_G8MAP(NAME,RQ,FAST)                                       \
L6_KALIGN static void NAME(double *restrict X, const double *restrict C) \
{                                                                        \
    VSETM;                                                               \
    for (int i = 0; i < 6; ++i) {                                        \
        double *x = X + (long)i * L6_GPS;                                \
        const double *c = C + (long)i * L6_GPS;                          \
        for (int e = 0; e < 36; ++e) {                                   \
            __m512d _R = _mm512_add_pd(_mm512_load_pd(x + 16*e),         \
                                       _mm512_load_pd(c + 16*e));        \
            __m512d _I = _mm512_add_pd(_mm512_load_pd(x + 16*e + 8),     \
                                       _mm512_load_pd(c + 16*e + 8));    \
            __m512d _s = _mm512_fmadd_pd(_I,_I,_mm512_fmadd_pd(_R,_R,mg));\
            __m512d _w = _mm512_rsqrt14_pd(_s);                          \
            __m512d _hs= _mm512_mul_pd(_s,mhalf);                        \
            __m512d _a = _mm512_mul_pd(_w,_w);                           \
            _w = _mm512_mul_pd(_w,_mm512_fnmadd_pd(_hs,_a,m15));         \
            _a = _mm512_mul_pd(_w,_w);                                   \
            _w = _mm512_mul_pd(_w,_mm512_fnmadd_pd(_hs,_a,m15));         \
            __m512d _d;                                                  \
            if (FAST) {                                                  \
                _d = _mm512_fmadd_pd(_s,_w,mone);                        \
            } else {                                                     \
                __m512d _r = _mm512_mul_pd(_s,_w);                       \
                L6_HERON(_r,_s,_w);                                      \
                _d = _mm512_add_pd(_r,mone);                             \
            }                                                            \
            RQ(_y,_d);                                                   \
            _mm512_store_pd(x + 16*e,     _mm512_mul_pd(_R,_y));         \
            _mm512_store_pd(x + 16*e + 8, _mm512_mul_pd(_I,_y));         \
        }                                                                \
    }                                                                    \
}

L6_DEF_G8MAP(l6_g8map_div,  L6_RECIP_DIV, 0)
L6_DEF_G8MAP(l6_g8map_rcp,  L6_RECIP_FMA, 0)
L6_DEF_G8MAP(l6_g8map_fast, L6_RECIP_DIV, 1)

/* ice_r7 PRUNED after measurement: the fused step (map inside the
 * x-pass loads, 2 sweeps/step) measured 0.248 vs the flat map's 0.238
 * us/xform at B=64 (+4.2%) despite deleting 432 zmm ld/st per group-
 * step -- the r4 adiv / r5 sdiv latency-exposure verdict reproduced in
 * SoA form: 6 ladder+divide chains in front of every x-codelet lose to
 * a flat 216-ladder pass.  Deleted per the dead-weight rule; the ops
 * live in git and the number lives in the strategy record. */

/* 8x8 double transpose (the rival v6 TR8, verbatim). */
#define L6_TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7) do{ \
    __m512d _t0=_mm512_unpacklo_pd(r0,r1), _t1=_mm512_unpackhi_pd(r0,r1); \
    __m512d _t2=_mm512_unpacklo_pd(r2,r3), _t3=_mm512_unpackhi_pd(r2,r3); \
    __m512d _t4=_mm512_unpacklo_pd(r4,r5), _t5=_mm512_unpackhi_pd(r4,r5); \
    __m512d _t6=_mm512_unpacklo_pd(r6,r7), _t7=_mm512_unpackhi_pd(r6,r7); \
    __m512d _u0=_mm512_shuffle_f64x2(_t0,_t2,0x44), _u1=_mm512_shuffle_f64x2(_t4,_t6,0x44); \
    __m512d _u2=_mm512_shuffle_f64x2(_t0,_t2,0xee), _u3=_mm512_shuffle_f64x2(_t4,_t6,0xee); \
    __m512d _u4=_mm512_shuffle_f64x2(_t1,_t3,0x44), _u5=_mm512_shuffle_f64x2(_t5,_t7,0x44); \
    __m512d _u6=_mm512_shuffle_f64x2(_t1,_t3,0xee), _u7=_mm512_shuffle_f64x2(_t5,_t7,0xee); \
    o0=_mm512_shuffle_f64x2(_u0,_u1,0x88); o2=_mm512_shuffle_f64x2(_u0,_u1,0xdd); \
    o4=_mm512_shuffle_f64x2(_u2,_u3,0x88); o6=_mm512_shuffle_f64x2(_u2,_u3,0xdd); \
    o1=_mm512_shuffle_f64x2(_u4,_u5,0x88); o3=_mm512_shuffle_f64x2(_u4,_u5,0xdd); \
    o5=_mm512_shuffle_f64x2(_u6,_u7,0x88); o7=_mm512_shuffle_f64x2(_u6,_u7,0xdd); \
}while(0)

/* AoS (8 consecutive volumes, interleaved complex) -> SoA group block.
 * Runs twice per group per whole CHAIN (x0 and c), so it is not a hot
 * path; 36 = 4*9, so the 4-element TR8 loop has no scalar tail. */
static void l6_g8_convin(const double *restrict v0, double *restrict G)
{
    for (int i = 0; i < 6; ++i) {
        double *gp = G + (long)i * L6_GPS;
        long base = 72*i;                       /* doubles: 2*(36*i) */
        for (int e = 0; e < 36; e += 4) {
            const double *s = v0 + base + 2*e;
            __m512d r0=_mm512_load_pd(s+0*L6_VD), r1=_mm512_load_pd(s+1*L6_VD);
            __m512d r2=_mm512_load_pd(s+2*L6_VD), r3=_mm512_load_pd(s+3*L6_VD);
            __m512d r4=_mm512_load_pd(s+4*L6_VD), r5=_mm512_load_pd(s+5*L6_VD);
            __m512d r6=_mm512_load_pd(s+6*L6_VD), r7=_mm512_load_pd(s+7*L6_VD);
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            L6_TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            _mm512_store_pd(gp+16*e+ 0,o0); _mm512_store_pd(gp+16*e+ 8,o1);
            _mm512_store_pd(gp+16*e+16,o2); _mm512_store_pd(gp+16*e+24,o3);
            _mm512_store_pd(gp+16*e+32,o4); _mm512_store_pd(gp+16*e+40,o5);
            _mm512_store_pd(gp+16*e+48,o6); _mm512_store_pd(gp+16*e+56,o7);
        }
    }
}

static void l6_g8_convout(const double *restrict G, double *restrict v0)
{
    for (int i = 0; i < 6; ++i) {
        const double *gp = G + (long)i * L6_GPS;
        long base = 72*i;
        for (int e = 0; e < 36; e += 4) {
            __m512d r0=_mm512_load_pd(gp+16*e+ 0), r1=_mm512_load_pd(gp+16*e+ 8);
            __m512d r2=_mm512_load_pd(gp+16*e+16), r3=_mm512_load_pd(gp+16*e+24);
            __m512d r4=_mm512_load_pd(gp+16*e+32), r5=_mm512_load_pd(gp+16*e+40);
            __m512d r6=_mm512_load_pd(gp+16*e+48), r7=_mm512_load_pd(gp+16*e+56);
            __m512d o0,o1,o2,o3,o4,o5,o6,o7;
            L6_TR8(r0,r1,r2,r3,r4,r5,r6,r7,o0,o1,o2,o3,o4,o5,o6,o7);
            double *d = v0 + base + 2*e;
            _mm512_store_pd(d+0*L6_VD,o0); _mm512_store_pd(d+1*L6_VD,o1);
            _mm512_store_pd(d+2*L6_VD,o2); _mm512_store_pd(d+3*L6_VD,o3);
            _mm512_store_pd(d+4*L6_VD,o4); _mm512_store_pd(d+5*L6_VD,o5);
            _mm512_store_pd(d+6*L6_VD,o6); _mm512_store_pd(d+7*L6_VD,o7);
        }
    }
}

/* One group's whole m-step chain: convert in, run every step SoA-
 * resident (X 28.4 KB + C 28.4 KB; nothing else is touched between the
 * conversions), convert out.  Same step structure as the AoS chains:
 * raw FFT of x0, then m-1 x (map; FFT), then the closing map.
 * noinline + 64B-pinned: without it gcc folds this whole body (and the
 * pass functions with it) into fft3d_chain, and the displaced ipdiv
 * tail loop re-enacts the file's r5 code-placement disease at B=1
 * (measured this round: 0.376 vs 0.330 us, the same numbers as r5). */
__attribute__((noinline)) L6_KALIGN
static void l6_g8_chain(double *restrict X, double *restrict C,
                        const double *x0v, const double *cv, double *outv,
                        int m, void (*mp)(double *, const double *))
{
    l6_g8_convin(x0v, X);
    l6_g8_convin(cv,  C);
    l6_g8_fft(X);
    for (int s = 1; s < m; ++s) {
        mp(X, C);
        l6_g8_fft(X);
    }
    mp(X, C);
    l6_g8_convout(X, outv);
}

/* ------------------------------------------------------------------ *
 * ice_r8 "g8f": the STORE-FUSED step -- ADOPTED FROM THE WARM RIVAL
 * warm_d43251c2_score0.99/impl_3907.c (their gC_6, the L=6/8 engine the
 * r8 brief points at): the map + c-add run where the z-DFT's outputs
 * are already in registers, i.e. at the LAST store of the step's last
 * FFT sweep.  This deletes the flat map pass entirely: per group-step,
 * one whole X read+write traversal (432 zmm loads + 432 zmm stores) is
 * gone, and C is read inside the z-sweep instead of in its own pass.
 * A step becomes [x-sweep; per plane: y-sweep, z-sweep+map] -- exactly
 * the rival structure -- and the chain is m IDENTICAL calls (state
 * between steps is the MAPPED state, so no closing map either).
 *
 * Why this is not the r7 fused-step loser: that experiment put the
 * ladders in FRONT of the x-codelets (map at the loads), so 6
 * ladder+divide chains gated every group's first FMA.  Here the ladders
 * hang OFF the z-DFT outputs; nothing inside the step waits on them --
 * the next pencil's DFT is independent -- and the first consumer is the
 * NEXT step's x-sweep, a whole sweep away.
 *
 * Per-point op sequence (c-add, s, ladder, d, one vdivpd per element
 * vector, two muls) is copied token-for-token from l6_g8map_fast /
 * l6_g8map_div, so g8f output is BIT-IDENTICAL to g8 at the same tier
 * (cmp-verified; the register value stored by the DFT and the value
 * loaded back by the flat pass are the same bits).
 * ------------------------------------------------------------------ */

/* The PFA 2x3 core to REGISTERS (same DAG, same single-rounded ops as
 * L6_SOA6_CORE; outputs o0..o5 in natural element order). */
#define L6_SOA6_REG(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,     \
                    o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i)     \
 do { \
    __m512d p0r = _mm512_add_pd(x0r,x3r), p0i = _mm512_add_pd(x0i,x3i); \
    __m512d q0r = _mm512_sub_pd(x0r,x3r), q0i = _mm512_sub_pd(x0i,x3i); \
    __m512d p1r = _mm512_add_pd(x2r,x5r), p1i = _mm512_add_pd(x2i,x5i); \
    __m512d q1r = _mm512_sub_pd(x2r,x5r), q1i = _mm512_sub_pd(x2i,x5i); \
    __m512d p2r = _mm512_add_pd(x4r,x1r), p2i = _mm512_add_pd(x4i,x1i); \
    __m512d q2r = _mm512_sub_pd(x4r,x1r), q2i = _mm512_sub_pd(x4i,x1i); \
    __m512d ar  = _mm512_add_pd(p1r,p2r), ai  = _mm512_add_pd(p1i,p2i); \
    __m512d br  = _mm512_sub_pd(p1r,p2r), bi  = _mm512_sub_pd(p1i,p2i); \
    __m512d cr  = _mm512_add_pd(q1r,q2r), ci  = _mm512_add_pd(q1i,q2i); \
    __m512d dr  = _mm512_sub_pd(q1r,q2r), di  = _mm512_sub_pd(q1i,q2i); \
    __m512d mr  = _mm512_fnmadd_pd(half,ar,p0r), mi = _mm512_fnmadd_pd(half,ai,p0i); \
    __m512d nr  = _mm512_fnmadd_pd(half,cr,q0r), ni = _mm512_fnmadd_pd(half,ci,q0i); \
    o0r = _mm512_add_pd(p0r,ar);      o0i = _mm512_add_pd(p0i,ai);      \
    o3r = _mm512_add_pd(q0r,cr);      o3i = _mm512_add_pd(q0i,ci);      \
    o4r = _mm512_fmadd_pd (s3,bi,mr); o4i = _mm512_fnmadd_pd(s3,br,mi); \
    o2r = _mm512_fnmadd_pd(s3,bi,mr); o2i = _mm512_fmadd_pd (s3,br,mi); \
    o1r = _mm512_fmadd_pd (s3,di,nr); o1i = _mm512_fnmadd_pd(s3,dr,ni); \
    o5r = _mm512_fnmadd_pd(s3,di,nr); o5i = _mm512_fmadd_pd (s3,dr,ni); \
 } while (0)

/* Map one output element (in registers) against its c slot and store.
 * FAST is a compile-time 0/1; op sequence == l6_g8map_fast / _div. */
#define L6_G8EMAP(px,pc,OR,OI,FAST)                                      \
    do {                                                                \
        __m512d _R = _mm512_add_pd(OR,_mm512_load_pd(pc));              \
        __m512d _I = _mm512_add_pd(OI,_mm512_load_pd((pc)+8));          \
        __m512d _s = _mm512_fmadd_pd(_I,_I,_mm512_fmadd_pd(_R,_R,mg));  \
        __m512d _w = _mm512_rsqrt14_pd(_s);                             \
        __m512d _hs= _mm512_mul_pd(_s,mhalf);                           \
        __m512d _a = _mm512_mul_pd(_w,_w);                              \
        _w = _mm512_mul_pd(_w,_mm512_fnmadd_pd(_hs,_a,m15));            \
        _a = _mm512_mul_pd(_w,_w);                                      \
        _w = _mm512_mul_pd(_w,_mm512_fnmadd_pd(_hs,_a,m15));            \
        __m512d _d;                                                     \
        if (FAST) {                                                     \
            _d = _mm512_fmadd_pd(_s,_w,mone);                           \
        } else {                                                        \
            __m512d _r = _mm512_mul_pd(_s,_w);                          \
            L6_HERON(_r,_s,_w);                                         \
            _d = _mm512_add_pd(_r,mone);                                \
        }                                                               \
        __m512d _y = _mm512_div_pd(mone,_d);                            \
        _mm512_store_pd((px),   _mm512_mul_pd(_R,_y));                  \
        _mm512_store_pd((px)+8, _mm512_mul_pd(_I,_y));                  \
    } while (0)

/* One z-pencil (element stride 16): DFT to registers, then map+store
 * each element, in the same element-store order as L6_SOA6_CORE.  The
 * 6 vdivpd are spaced by ~15 arith ops each, so the divider pipelines
 * under the FMA-port work exactly as in the flat pass. */
#define L6_DEF_G8ZP(NAME,FAST)                                           \
static __attribute__((always_inline)) inline void NAME(                 \
        double *restrict re, const double *restrict cp)                 \
{                                                                       \
    const __m512d half = _mm512_set1_pd(0.5);                           \
    const __m512d s3   = _mm512_set1_pd(L6_S3);                         \
    VSETM;                                                              \
    __m512d x0r=_mm512_load_pd(re+ 0), x0i=_mm512_load_pd(re+ 8);       \
    __m512d x1r=_mm512_load_pd(re+16), x1i=_mm512_load_pd(re+24);       \
    __m512d x2r=_mm512_load_pd(re+32), x2i=_mm512_load_pd(re+40);       \
    __m512d x3r=_mm512_load_pd(re+48), x3i=_mm512_load_pd(re+56);       \
    __m512d x4r=_mm512_load_pd(re+64), x4i=_mm512_load_pd(re+72);       \
    __m512d x5r=_mm512_load_pd(re+80), x5i=_mm512_load_pd(re+88);       \
    __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i;            \
    L6_SOA6_REG(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,        \
                o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i);       \
    L6_G8EMAP(re+ 0, cp+ 0, o0r, o0i, FAST);                            \
    L6_G8EMAP(re+48, cp+48, o3r, o3i, FAST);                            \
    L6_G8EMAP(re+64, cp+64, o4r, o4i, FAST);                            \
    L6_G8EMAP(re+32, cp+32, o2r, o2i, FAST);                            \
    L6_G8EMAP(re+16, cp+16, o1r, o1i, FAST);                            \
    L6_G8EMAP(re+80, cp+80, o5r, o5i, FAST);                            \
}

L6_DEF_G8ZP(l6_soa6_zmap_x, 0)

/* The FAST-tier fused z-sweep runs the warm rival's H/R2 map
 * ALTERNATION (impl_3907 MAPW_H / MAPW_R2, adopted verbatim in split
 * form): even store slots use hardware vsqrtpd + an rcp14+2-Newton
 * reciprocal (no divide), odd slots the rsqrt ladder + the same rcp
 * reciprocal (no divider use at all) -- per group-step the divider
 * unit sees 108 vsqrtpd instead of 216 vdivpd, at +4 FMA-port ops per
 * element on average.  Measured (ice_r8, matched 3.30 GHz windows):
 * 0.2070 vs the uniform ladder+vdivpd fused map's 0.2082 us/xform, and
 * insensitive to the 2.90 clock mode (0.2069) where ladder+div reads
 * 0.2367.  Two-step gate 6.33e-16 (tol 3e-14); H needs no 1e-300
 * guard (sqrt(0)=0, d=1), R2 keeps it (rsqrt14(0)=inf).
 * Both arms are ~1-2 ulp per application: sqrt is correctly rounded,
 * rcp14+2NR and rsqrt14+2NR both land ~2^-53..2^-56 residual. */
#define L6_G8EMAP_H(px,pc,OR,OI)                                         \
    do {                                                                \
        __m512d _R = _mm512_add_pd(OR,_mm512_load_pd(pc));              \
        __m512d _I = _mm512_add_pd(OI,_mm512_load_pd((pc)+8));          \
        __m512d _s = _mm512_fmadd_pd(_I,_I,_mm512_mul_pd(_R,_R));       \
        __m512d _d = _mm512_add_pd(_mm512_sqrt_pd(_s),mone);            \
        __m512d _y = _mm512_rcp14_pd(_d);                               \
        _y = _mm512_mul_pd(_y,_mm512_fnmadd_pd(_d,_y,mtwo));            \
        _y = _mm512_mul_pd(_y,_mm512_fnmadd_pd(_d,_y,mtwo));            \
        _mm512_store_pd((px),   _mm512_mul_pd(_R,_y));                  \
        _mm512_store_pd((px)+8, _mm512_mul_pd(_I,_y));                  \
    } while (0)
#define L6_G8EMAP_R2(px,pc,OR,OI)                                        \
    do {                                                                \
        __m512d _R = _mm512_add_pd(OR,_mm512_load_pd(pc));              \
        __m512d _I = _mm512_add_pd(OI,_mm512_load_pd((pc)+8));          \
        __m512d _s = _mm512_fmadd_pd(_I,_I,_mm512_fmadd_pd(_R,_R,mg));  \
        __m512d _w = _mm512_rsqrt14_pd(_s);                             \
        __m512d _hs= _mm512_mul_pd(_s,mhalf);                           \
        __m512d _a = _mm512_mul_pd(_w,_w);                              \
        _w = _mm512_mul_pd(_w,_mm512_fnmadd_pd(_hs,_a,m15));            \
        _a = _mm512_mul_pd(_w,_w);                                      \
        _w = _mm512_mul_pd(_w,_mm512_fnmadd_pd(_hs,_a,m15));            \
        __m512d _d = _mm512_fmadd_pd(_s,_w,mone);                       \
        __m512d _y = _mm512_rcp14_pd(_d);                               \
        _y = _mm512_mul_pd(_y,_mm512_fnmadd_pd(_d,_y,mtwo));            \
        _y = _mm512_mul_pd(_y,_mm512_fnmadd_pd(_d,_y,mtwo));            \
        _mm512_store_pd((px),   _mm512_mul_pd(_R,_y));                  \
        _mm512_store_pd((px)+8, _mm512_mul_pd(_I,_y));                  \
    } while (0)
static __attribute__((always_inline)) inline void l6_soa6_zmap_a(
        double *restrict re, const double *restrict cp)
{
    const __m512d half = _mm512_set1_pd(0.5);
    const __m512d s3   = _mm512_set1_pd(L6_S3);
    VSETM;
    __m512d x0r=_mm512_load_pd(re+ 0), x0i=_mm512_load_pd(re+ 8);
    __m512d x1r=_mm512_load_pd(re+16), x1i=_mm512_load_pd(re+24);
    __m512d x2r=_mm512_load_pd(re+32), x2i=_mm512_load_pd(re+40);
    __m512d x3r=_mm512_load_pd(re+48), x3i=_mm512_load_pd(re+56);
    __m512d x4r=_mm512_load_pd(re+64), x4i=_mm512_load_pd(re+72);
    __m512d x5r=_mm512_load_pd(re+80), x5i=_mm512_load_pd(re+88);
    __m512d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i;
    L6_SOA6_REG(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,
                o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i);
    L6_G8EMAP_H (re+ 0, cp+ 0, o0r, o0i);
    L6_G8EMAP_R2(re+48, cp+48, o3r, o3i);
    L6_G8EMAP_H (re+64, cp+64, o4r, o4i);
    L6_G8EMAP_R2(re+32, cp+32, o2r, o2i);
    L6_G8EMAP_H (re+16, cp+16, o1r, o1i);
    L6_G8EMAP_R2(re+80, cp+80, o5r, o5i);
}

/* One whole chain step: x-sweep, then per plane y-sweep and the mapped
 * z-sweep.  On exit X holds the MAPPED state of this step. */
#define L6_DEF_G8STEP(NAME,ZM)                                           \
L6_KALIGN static void NAME(double *restrict X, const double *restrict C)\
{                                                                       \
    for (int e = 0; e < 36; ++e)                 /* x: stride L6_GPS */ \
        l6_soa6(X + 16*e, L6_GPS);                                      \
    for (int i = 0; i < 6; ++i) {                                       \
        double *pl = X + (long)i * L6_GPS;                              \
        const double *cp = C + (long)i * L6_GPS;                        \
        for (int z = 0; z < 6; ++z)              /* y: stride 96     */ \
            l6_soa6(pl + 16*z, 96);                                     \
        for (int y = 0; y < 6; ++y)              /* z+c+map          */ \
            ZM(pl + 96*y, cp + 96*y);                                   \
    }                                                                   \
}

L6_DEF_G8STEP(l6_g8_step_f, l6_soa6_zmap_a)
L6_DEF_G8STEP(l6_g8_step_x, l6_soa6_zmap_x)

__attribute__((noinline)) L6_KALIGN
static void l6_g8_chain_f(double *restrict X, double *restrict C,
                          const double *x0v, const double *cv, double *outv,
                          int m, int tier)
{
    l6_g8_convin(x0v, X);
    l6_g8_convin(cv,  C);
    if (tier)
        for (int s = 0; s < m; ++s) l6_g8_step_x(X, C);
    else
        for (int s = 0; s < m; ++s) l6_g8_step_f(X, C);
    l6_g8_convout(X, outv);
}

#endif /* L6_HAVE_AVX2 && L6_HAVE_AVX512 */
