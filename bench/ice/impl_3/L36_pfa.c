/* L36_pfa.c -- forward complex 3D DFT of a 36^3 cube, batched, out-of-place.
 *
 * ROUND panel_r2.  The round-1 version of this file (three passes over the
 * volume through a slab-blocked 830 KB intermediate, split-complex SoA) lost
 * to L36_mixedradix on the node, 225 vs 118 us at B=1, because it crosses
 * memory three times where two suffice.  This rewrite adopts the structure
 * that won and adds what it lacked in the batched regime.
 *
 * TECHNIQUE
 *   Row-column 3D DFT; every 36-point line is a Good-Thomas / prime-factor
 *   4 x 9 codelet (zero twiddles between the stages), on INTERLEAVED complex
 *   vectors whose lanes are a spectator axis.  Two sweeps over the volume:
 *
 *   phase 1, per x-plane (36x36 complex = 20.25 KB, L1-resident):
 *       z transform: lanes = PW y-rows, via PWxPW complex-granule register
 *                    transposes on load and store (the one unavoidable
 *                    transpose pair), into a plane scratch pl[y][kz]
 *       y transform: lanes = PW kz (contiguous in pl), store to mid[x][ky][kz]
 *   phase 2:
 *       x transform: lanes = PW kz, stride 2592 doubles between x, from mid
 *                    into out (or in place when mid == out)
 *
 *   `mid` is either `out` itself (in-place phase 2: smallest resident set,
 *   746 KB, fits the node's 1 MB L2 -- the L36_mixedradix arrangement) or a
 *   plan-owned scratch volume S that is REUSED for every volume of the batch,
 *   so S stays cache-resident across the whole batch and never costs DRAM
 *   traffic; phase 2 then writes `out` with non-temporal stores.  At large B
 *   that cuts DRAM traffic per volume to the compulsory read-in + write-out
 *   (1.5 MB), where in-place pays an extra RFO + writeback round (2.2 MB).
 *   fft3d_create() times {PW=2, PW=4} x {inplace, scratch, scratch+NT} on a
 *   dummy batch, verifies every candidate against a reference, and installs
 *   the fastest -- so the AVX-512-licence and NT questions are settled by
 *   measurement on the machine that matters, per candidate, per batch size.
 *
 * ATTRIBUTION (round-2 rules: say what you borrowed)
 *   - Two-sweep plane-fused pass structure, interleaved-complex lanes, and the
 *     6-op DFT3 / 8-op DFT4 / 2-op CMUL forms: from L36_mixedradix round 1.
 *   - NT stores on the final write at large batch (+53% at B=32) and the
 *     create-time correctness interlock for variants: from L36_pencilfused.
 *   - The reused-scratch + NT combination and the PFA index maps folded into
 *     compile-time addressing: this file.
 *
 * OPERATION COUNT (per 36-point line over PW lanes, FMA-port vector ops)
 *   9 x DFT4 (8 ops, x(-i) folded into two +-1-pair FMAs)          =  72
 *   4 x DFT9 = CT 3x3: 6 DFT3 (6 ops) + 4 twiddle CMUL (2 ops)     = 176
 *   total 248 FMA-port ops + 49 swaps (port 5) per PW lines
 *   Per volume: 3888 lines -> 241 056 FMA-port ops at PW=4; on a 1-FMA-unit
 *   Gold 5218 that is ~241k cycles = 105 us of pure port-0 work -- the floor.
 *
 * ACCURACY: same modules as round 1, expect ~4e-16 relative L2 vs numpy.
 *
 * ROUND panel_r3: memory-level parallelism for the batched regime.
 *   The r2 node result (B=1 119.3 us vs B=256 238.8 us) means ~119 us/volume of
 *   the batched time is UN-overlapped memory: compulsory traffic is only
 *   1.49 MB/volume and other entries sustain 10.5-12.1 GB/s single-core on this
 *   node (monitor's r2 verdict).  Fix: (a) phase 1 software-prefetches its own
 *   input stream at a fixed byte distance (FFT36_PFD), paced evenly across both
 *   subloops -- the stream is perfectly linear over the volume, but the L2
 *   streamer alone does not hide DRAM across 4 KB page boundaries; (b) phase 2
 *   prefetches the first ~62 KB of the NEXT volume's input, so the phase-1
 *   cursor never starts cold (cross-volume overlap, the shape L6_unrolled/
 *   L6_pfa proved at L=6: next-volume coverage with prefetchtX; NTA is their
 *   documented catastrophe, not retried).  Prefetch is a tuner dimension (pf),
 *   so the B=1 in-place path can keep its r2 behaviour if prefetch costs there.
 *
 * ROUND panel_r4: cross-volume ping-pong pipeline as an ADDITIONAL candidate,
 *   tuner pick reported, tuner hysteresis.
 *   The r3 node result (B=256 227.5 us vs a ~124 us bandwidth ceiling) says the
 *   read burst of phase 1 and the write burst of phase 2 still run serially:
 *   prefetch alone bought 4.7% on the node against 24% on wallaby.  New mode
 *   M_PIPE: two plan-owned scratch volumes S0/S1; while phase 2 of volume b
 *   drains S_a to `out` with NT stores, phase 1 of volume b+1 fills S_b --
 *   interleaved at plane granularity (both phases have exactly 36 outer
 *   iterations per volume), so the DRAM read stream (in[b+1]) and the NT write
 *   stream (out[b]) are busy SIMULTANEOUSLY at ~6.3 GB/s each instead of
 *   alternating 12+ GB/s bursts.  Cost: 1.5 MB of scratch, which no longer fits
 *   the node's 1 MB L2 -- the r3 record deferred this design on exactly that
 *   ground, so it ships as one more tuner candidate (the r3 verdict's process
 *   lesson: ADD candidates, never replace structures), correctness-gated and
 *   selected only if it wins on the machine that matters.
 *   Also per the r3 verdict: fft3d_description() now reports the tuner's pick
 *   (mechanism borrowed from L36_mixedradix/L6_pfa); the tuner prefers the
 *   simplest candidate within 2% of the best time (hysteresis, against the
 *   pf0/pf1 coin-flips the verdict measured costing 3.9-6.7% elsewhere); and
 *   FFT36_PW / FFT36_MODE / FFT36_PF environment variables force the choice at
 *   plan time so the monitor can run forced-variant controls without recompiling.
 *
 * ROUND panel_r5: write-intent prefetch (pf=2), and the per-plane refactor
 *   inlined away.
 *   The r4 node picks settled the mode question: INPLACE + paced read prefetch
 *   won BOTH streaming cells (B=32 174.2 us, B=256 218.9 us, first on the
 *   board) -- every scratch/NT/pipe candidate lost on the machine that counts.
 *   What in-place leaves exposed is the RFO: phase 1 stores to a cold `out`
 *   volume, so every 64-byte line of 746 KB costs a demand read-for-ownership
 *   from DRAM that nothing overlaps.  L6_unrolled's r3 headline (adopted by
 *   L6_pfa, confirmed on the node: `prefetchw` variants selected at all DRAM
 *   batch sizes) is exactly this fix: __builtin_prefetch(p,1,3) emits
 *   `prefetchw` on PRFCHW machines (Cascade Lake and Sapphire Rapids both),
 *   acquiring the line exclusive AHEAD of the store while keeping the
 *   normal-store shape the node demonstrably prefers over NT.  Here it ships
 *   as tuner level pf=2: phase 1 paces a write-intent cursor one plane
 *   (FFT36_PFWD = 2592 doubles) ahead of its mid-plane stores when mid==out
 *   (M_INPLACE), and phase 2 write-prefetches its 36 out-streams one line
 *   ahead when out is store-cold (M_SCRATCH); NT/PIPE modes gain nothing from
 *   it (NT stores do not RFO), so pf=2 is only instantiated for those two
 *   modes.  L6's warning is respected by construction: prefetchw LOST 17% at
 *   cache-resident sizes there, so it enters as a candidate the tournament
 *   must select, never a default.
 *   Second change: phase1_plane/phase2_yplane (the r4 per-plane refactor that
 *   M_PIPE needed) are now always_inline, restoring r3's monolithic code
 *   layout for the non-pipe paths -- the r4 verdict names that refactor the
 *   only candidate cause for this entry's +2.4% B=4 node regression
 *   (129.2 -> 132.3 us against a 0.5% spread, same pick string).
 *
 * ROUND panel_r6: NTA read stream to protect `out`'s L2 residency (pf=3/4).
 *   The r4/r5 verdicts settled the node clock at ~3.89 GHz sustained, so the
 *   B=1 port floor is ~62 us against 120.4 measured -- and the extra ~230k
 *   cycles are node-specific (wallaby's B=1 sits exactly on ITS port-5 model:
 *   120.5k half-FMA + ~95k shuffles = 215k cycles = the 51.4 us measured).
 *   The node's difference from wallaby at B=1 is one number: L2 = 1 MB vs
 *   2 MB, against in+out = 1.5 MB.  The sequential in-read (demand or T1
 *   prefetch) allocates every line into L2, evicting `out` mid-execute, so
 *   the in-place phase-1 stores RFO from L3, phase 2 re-reads from L3, and
 *   modified out lines write back -- 2-3 MB of L2<->L3 round trips per
 *   execute where the compulsory traffic is 746 KB.  Fix: prefetch the in
 *   stream with hint NTA (fills L1, bypasses L2 on SKX-class cores) at a
 *   SHORT constant lead (FFT36_PFDN = 512 doubles = 4 KB), paced at exactly
 *   the consumption rate inside the yb-subloop only (2*PFSTEP per iteration,
 *   nothing in the zb-subloop) so the lead never swings.  A demand load that
 *   finds its line already in L1 allocates nothing into L2, so `out` (746 KB
 *   + the 20 KB plane) stays L2-resident across the whole execute -- and at
 *   B=1, across EXECUTES: steady state leaves out's lines L2-modified, so
 *   the RFO and the writeback disappear entirely and L3 traffic collapses to
 *   the compulsory 746 KB in-read.  The r3 NTA catastrophe (135.1 vs 104.4
 *   at B=256) was a 32 KB lead on DRAM-latency streams -- lines evicted from
 *   L1 before use, and nothing behind them in L2 by design; a 4 KB lead on
 *   L3-resident data is a different regime, and it enters as a gated
 *   candidate the tournament must select, never a default.  Two levels:
 *     pf=3  NTA in-read + the pf=2 write-intent mechanism (streaming cells)
 *     pf=4  NTA in-read alone (B=1/B=4: out is already L2-M in steady state,
 *           so prefetchw is pure uop tax there -- ranked SIMPLER than pf=3)
 *   Both only for INPLACE/SCRATCH; PFNX is off (cache-resident target).
 *
 * ROUND panel_r7: NT stores resurrected under NTA read protection
 *   (scratch+nt, pf=4 -- a combination no round has ever fielded).
 *   B=256 in-place is AT its own traffic floor (node r5: 182.6 us measured
 *   vs ~183 modeled for 2.2 MB/vol at the ~12 GB/s single-core rate), so the
 *   only structure with a lower floor is scratch+NT: compulsory 1.5 MB/vol
 *   = ~125 us.  NT has lost on the node four rounds running, but every one
 *   of those losses shares one unaddressed defect: the T1/demand in-read
 *   fills L2 and continuously evicts the 746 KB scratch S (S alone is 3/4 of
 *   the node's 1 MB L2), so S's phase-1 writes and phase-2 re-reads become
 *   ~1.5 MB/vol of extra L2<->L3 round trips that erase NT's DRAM-traffic
 *   advantage.  r6's pf=4 NTA in-read (L1-fill, L2-bypass) removes exactly
 *   that defect, but r6 gated pf>=2 to INPLACE/SCRATCH on the "NT stores do
 *   not RFO" argument -- true, and the wrong filter: the READ side is what
 *   NT mode needed all along.  New candidates {pw2,pw4} x scratch+nt x pf=4:
 *   phase 1 NTA-reads `in` (S stays L2-resident), phase 2 reads S from L2
 *   and NT-writes `out`; per-volume DRAM traffic = the compulsory 1.5 MB.
 *   PFNX stays off (the cold window is only FFT36_PFDN = 4 KB/volume, and
 *   T1 pre-coverage would refill L2 with in-stream lines, defeating the
 *   design).  The NTA cursor keeps r6's yb-subloop-only pacing at a 4 KB
 *   constant lead: pacing it across both subloops would need a >=16 KB lead
 *   to absorb the yb consumption deficit, and 16 KB of quick-evict NTA lines
 *   + the 20 KB plane buffer overflow L1 -- the r3 catastrophe mechanism.
 *   The read stream is therefore bursty (2x average rate during the yb half,
 *   idle during zb), which a ~14 GB/s burst demand against the node's peak
 *   should absorb; if it does not, the tournament rejects the candidate and
 *   the pick string says so.  Ranked most complex in the hysteresis (must
 *   beat every simpler candidate by >3%).  Wallaby cannot price this bet
 *   either direction (its 2 MB L2 holds S regardless of read pollution, and
 *   its tuner already overfavours scratch modes via the 60 MB L3 arena
 *   artifact) -- the node tournament is the only honest judge, same epistemic
 *   situation as r6's pf=4 itself.
 *
 * ROUND panel_r8: two-level phase-2 read prefetch (pf=5/6), a regime-aware
 *   hysteresis band, and in-plan node probes reported via the description.
 *   r7 on the node: NT-pf4 and NTA rejected everywhere (verdict: NT is now
 *   0-for-everything on this node; prefetchw-not-NT is a rule).  B=1 sits at
 *   120.25 us = 1.45x the ~83 us port floor, all three L36 entries cluster at
 *   118.5-124.0, and the discriminating perf counter has still never been
 *   read.  Three changes, all additive:
 *   (1) pf=5: deep T1 prefetch of phase 2's 36 source streams, FFT36_PF2D=4
 *       lines (4 tiles ~ 1000 cycles) ahead, on top of the existing 1-line T0.
 *       Why it is not redundant with PF36: at a tile boundary PF36 issues 36
 *       T0 prefetches that all contend for the ~12 L1 fill buffers, so lines
 *       coming from L3 (out's early planes, evicted by the phase-1 in-read at
 *       B=1) drain at ~12-at-a-time x ~70 cycles -- comparable to the whole
 *       248-cycle tile, i.e. marginally too late.  A T1 prefetch 4 tiles
 *       ahead stages L3->L2 through the L2 superqueue (16+ entries, no L1 FB
 *       pressure), and the T0 then only moves L2->L1.  Shape adopted from
 *       L64_radix8's slabpf (next-slab T1 staging, node-selected at B=2/B=8
 *       alongside pfw; their r7 Next explicitly suggests a deeper lead for
 *       L3 latency).  pf=6 = pf=5 + the pf=1 paced T1 in-read, in case the
 *       two ~1% terms add.  Both INPLACE-only (B=1 is the target; the
 *       verdict orders the batched cells left alone).
 *   (2) The 3% hysteresis band is regime-aware: when the arena IS the scored
 *       regime (nv == batch <= 8, i.e. B=1/B=4 -- same buffers, same 1.5 MB
 *       steady loop as the driver's timing loop), the band drops to 1%.  The
 *       3% band was built for streaming cells where the arena misrepresents
 *       the end-to-end run (r3's B=32 regression); at B=1 it has been
 *       systematically installing pf=0 over any candidate that wins by 1-3%,
 *       which is exactly the size of this entry's gap to L36_mixedradix
 *       (whose node pick string at B=1 shows pf1 sometimes winning).
 *   (3) fft3d_create() now times four fixed probes at nv=1 steady state
 *       (phase 1 alone; phase 2 alone L2-warm; phase 2 warm + deep prefetch;
 *       full pf0-inplace execute) and appends them to fft3d_description(),
 *       so the next leaderboard carries the node's own phase split -- the
 *       measurement three rounds of records have asked the monitor for.
 *
 * ROUND panel_r9: front-end round (the r8 verdict's order for L=36: the probe
 *   read fu - p1 - p2w = -3 us on the node, so the B=1 residual is INSIDE the
 *   phases -- "stop looking at caches; go at the front end. Code size, not
 *   caches.").  No arithmetic change; three things:
 *   (1) Compile-time specialization of the pf=0 hot path (the node's B=1/B=4
 *       pick in every round since r5).  run_vols used to thread pfr/p1w/pfw/
 *       p2d as RUNTIME ints into the always_inline plane functions, so the
 *       scored body carried every prefetch mechanism as live code: general
 *       phase1_pw4 = 1482 instrs (108 prefetches, flag tests in both subloop
 *       bodies), general phase2_pw4 = 1357 instrs with TWO full codelet copies
 *       (the NT loop sits in the same function, fp count 496 = 2x248).  New
 *       FN(phase1_pf0)/FN(phase2_pf0) call the plane functions with literal
 *       constant flags: gcc dead-codes every prefetch block, every flag test,
 *       the PFNX cursor and the whole NT copy out of the body the front end
 *       actually streams at B=1/B=4 (measured on the .o under node flags:
 *       phase1 1482 -> 1336 instrs, phase2 1357 -> 633, per-plane walked
 *       footprint ~8.8 KB -> ~7.9 KB against a ~1.5k-uop CLX DSB).  This is
 *       L45_pfa's r8 "compile-time exec-variant specialization" mechanism
 *       (their 3535 -> 3087-instr scoring path, all three L=45 cells taken);
 *       their other r8 mechanism, the opaque-base asm barrier, was checked
 *       for and is NOT needed here (this file's streams are single-base +
 *       disp32: 5-8 leas and no GPR spill-reload chains in either phase).
 *       It also deletes the r8 `if (pfd)' test from the phase-2 tile loop --
 *       the leading suspect for r8's +2.1% B=4 regression (verdict: refactor-
 *       around-an-untouched-hot-path class).  Streaming picks (pf=2) keep
 *       routing through the general bodies, byte-for-byte the same code as
 *       r8: the verdict orders the batched cells left alone.
 *   (2) The probe line now carries a code-size A/B the node itself runs:
 *       fu (specialized pf0 execute) vs fug (general-body pf0 execute, same
 *       arithmetic, same address streams, ~2x walked code).  fug - fu is a
 *       direct node measurement of what instruction footprint costs at B=1,
 *       readable off the leaderboard whatever the tuner picks.  (p2wd is
 *       dropped from the string: r8 answered its question -- deep-T1 tax
 *       +6.5 us, zero picks -- and the 224-byte string needed the room.)
 *   (3) An in-plan front-end counter probe: perf_event_open (syscall, no
 *       library) on {cycles, IDQ.DSB_UOPS 0x0879, IDQ.MITE_UOPS 0x0479,
 *       UOPS_ISSUED.ANY 0x010e} around the specialized fu loop, reported as
 *       k-events/volume in the description -- the counter read the verdict
 *       has asked for three rounds running ("the single thing").  Gated: if
 *       perf_event_open is denied (wallaby: perf_event_paranoid=4), the
 *       string says fe=na and the fug-fu differential still delivers.
 *
 * ROUND panel_r10: the DFT9 module replaced by genfft's n1_9 FMA DAG, and
 *   the pick lottery closed.  The r9 verdict's L=36 findings: fug - fu =
 *   +0.3% on the node (code size is NOT the B=1 residual; front-end theory
 *   dead alongside r8's cache theory), and the one unfalsified lever is the
 *   n1_9 transcription L45_pfa landed in r9 (44 -> 40 FMA-port vector ops
 *   per DFT9, correct first build, accuracy improved).  Taken verbatim from
 *   L45_pfa (same macro dialect): DFT9F replaces the CT 3x3 (6 DFT3M +
 *   4 CMULW) inside PFA36.  Per 36-line: 9 DFT4 (72) + 4 DFT9F (160) =
 *   232 FMA-port ops + ~57 swaps (was 248 + 49); per volume at PW=4:
 *   225 504 port-0 vector ops (was 241 056, -6.5%); port floor ~78 us at
 *   the 2.89 GHz AVX-512 licence clock (was ~83).
 *   Second change: the r9 verdict rules my B=1/B=4 numbers unusable until
 *   the pick lottery is closed (B=1 122.58/131.06/122.64, B=4
 *   131.95/184.11/132.48 -- the board's worst measurement).  At
 *   nv == batch <= 8 the candidate list is now restricted to the shapes the
 *   node has actually picked there in five consecutive scored rounds
 *   (r5-r9): {inplace, scratch} x pf<=2 x {pw2, pw4}.  NT, pipe, NTA
 *   (pf=3/4) and deep-T1 (pf=5/6) are 0-for-every-tournament at B=1/B=4 and
 *   contribute only tail risk there; they all remain candidates at
 *   streaming batch, and any FFT36_* / FFT_FORCE_* override restores the
 *   full list so the monitor's forced controls still work.  The r8 1% band
 *   at small nv is reverted to the uniform 3% (r9 verdict: "+-2% at these
 *   cells is the code-layout noise floor... stop building fixes for it"),
 *   and small-nv tournaments run 9 timing rounds instead of 5.
 *   Retired: the fug A/B (answered: +0.3%) and the perf_event group (fe=na
 *   on the node too; the verdict withdrew the counters).
 *
 * ROUND panel_r11: split the phase-1 excess, unbundle prefetchw, sweep PFWD.
 *   r10 node: B=1 113.128 (tied 1st; the n1_9 cut priced at ~full value --
 *   port 0 binds at L=36), still 1.45x the 78 us floor with p1 = 86-89 of
 *   fu = 115-117 (phase 1 is 76% of B=1 at ~1.7x its 52 us port share).
 *   The r10 verdict's order: attack phase 1's structure, checking
 *   L45_mixedradix's three-term costing (plane round trip / split accesses /
 *   compulsory L3) at L=36 first.  Checked by inspection: at L=36 the y-row
 *   stride is 576 B = 9 cache lines exactly, so every vector access in both
 *   phases is 64-B aligned and the split-access term -- L45's largest -- is
 *   structurally ZERO here.  What remains is compulsory L2/L3 movement, and
 *   the z/y subloop barrier inside a plane is real (the y transform needs
 *   all 36 y from the z pass), so cross-subloop fusion cannot be built
 *   without the cross-plane pipelining pencilfused already measured dead
 *   (+1-3% pw4).  Three additive changes:
 *   (1) probes p1z/p1y: the two phase-1 subloops timed separately at nv=1
 *       (z-subloop: in-read + transposes + codelet + pl write, an asm sink
 *       keeping the dead pl stores; y-subloop: codelet + mid stores, reading
 *       a fixed L1-resident 20.25 KB plane from S).  p1z carries the in-read
 *       latency, p1y carries the mid-store RFO; each side's port share is
 *       ~26 us, and p1z + p1y - p1 prices the intra-plane overlap.  Each is
 *       a LOWER bound for its side (the other stream's L2 pressure absent).
 *       Rides the description string like p1/p2w/fu (r8 mechanism).
 *   (2) pf=7: write-intent-only prefetch -- the pf=2 mechanism with the
 *       paced T1 read cursor REMOVED.  At B=1 the RFO half has only ever
 *       been fielded bundled with the read half (pure uop tax there: the
 *       in-read is one linear HW-prefetchable stream); the unbundled shape
 *       has never been tried in any tournament.  Gated candidate, INPLACE
 *       (phase-1 PFWMID) and SCRATCH (phase-2 PFW36), ranked one-mechanism
 *       (just above pf=1); enters the restricted small-nv list (inplace
 *       only, nc 12 -> 14) where the 3% band still demands a real >3% win
 *       over pf=0 before it can install.
 *   (3) FFT36_PFWD becomes a runtime plan parameter; at streaming batch the
 *       tournament adds {pw2,pw4} x inplace-pf2 x pfwd={1296,5184} beside
 *       the 2592 default -- the node PFWD sweep the records have requested
 *       since r5, now self-served.  Non-default pfwd ranks as a tie-break
 *       below its default twin (same bit class, prefetch-only difference).
 *
 * ROUND ice_r2: transpose-free phase 1 for Ice Lake (tr=1) -- both TRNC
 *   passes replaced by masked 128-bit broadcast LOADS, moving the lane
 *   transposes off port 5 and into the (idle) load ports.
 *   New machine, new binding resource.  On the graded node (Xeon Gold 6326,
 *   bare-metal ICX-SP) the second 512-bit FMA pipe lives on PORT 5 -- the
 *   same port that executes every 512-bit shuffle.  On CLX (1 FMA pipe,
 *   port 0) this file's TRNC transposes and codelet SWAPs were free riders
 *   on an otherwise idle port 5; on ICX every one of them displaces an FMA.
 *   Count per volume at PW=4: F = 225 504 FMA-class ops, S = 46 656 TRNC
 *   shuffles + 55 404 codelet swaps; two-port floor = (F+S)/2 = 163 782 cyc
 *   = 56.5 us at the node's 2.90 GHz (no AVX-512 downclock on this die,
 *   r1 leaderboard clk fields), against fu = 81.0 measured in r1.
 *   The fix uses the corpus's bare-metal proviso directly (S10: ICX-SP
 *   "folds broadcasts into the load uop for free", 2x64B loads/cyc):
 *   vbroadcastf64x2 (mem), zmm{k} is ONE load-port uop, no shuffle.  tr=1
 *   phase 1 builds each lane-transposed vector as 4 masked broadcast loads:
 *     z-subloop: LD(n) = 4 broadcasts straight from the in-plane rows
 *                (was: 4 full-row loads + TRNC's 8 shuffles per 4 vecs),
 *                ST(k) = plain vec store to pl in [ygroup][kz] order,
 *                lanes = y (was: codelet -> TRNC back -> pl[y][kz]);
 *     y-subloop: LD(n) = 4 broadcasts re-transposing pl's 128-bit granules
 *                on the fly (was: contiguous vec loads of pl[y][kz]);
 *                stores to mid unchanged.
 *   Port arithmetic per yb-group: p5-only drops 201 -> 57 ops, p0/p5 bound
 *   217 -> 145 cyc; the 144 broadcast loads cost 72 load-port cycles, under
 *   the 145 bound.  New floor (F + 55 404)/2 = 140 454 cyc = 48.4 us.
 *   GCC 11.4 verified (objdump): the 4-broadcast builder compiles to exactly
 *   4 vbroadcastf64x2-from-memory, kmovs hoisted, zero port-5 uops.
 *   Ships as a plan dimension tr (PW=4 only; PW=2's TRNC is 2 shuffles/2
 *   vecs and PW=2 loses every tournament anyway): tr=1 twins for pw4 x
 *   {inplace, scratch} x pf {0,1,2,7}, equal cand_rank to their tr=0 twins
 *   so raw min time decides between KERNELS while the 3% band still guards
 *   prefetch knobs; FFT36_TR forces it either way; the r5-r10 streaming
 *   picks and every pw2/NT/pipe path are byte-for-byte untouched.
 *   Probe p1t (tr=1 phase 1 alone, nv=1) joins the description next to p1.
 *   Context: the ice_r1 worker for this entry crashed before touching the
 *   file, so this is the first ICX-aware change; r1 scored the CLX r11 code
 *   at 119.163 us/transform (first, but only 0.3% ahead of L36_mixedradix).
 */
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <immintrin.h>

#include "fft3d_api.h"

#ifndef L36_PFA_ONCE            /* ============ COMMON, first pass ============ */
#define L36_PFA_ONCE

#define L    36
#define LSQ  1296                /* 36*36                                       */
#define VDBL ((size_t)2 * L * LSQ)   /* doubles per volume = 93312             */

#define CAT_(a,b) a##b
#define CAT(a,b)  CAT_(a,b)

/* phase-1 input prefetch distance, in doubles (32 KB): must exceed the ~10.4 KB
 * pacing deficit the yb-subloop accumulates (it consumes 2x faster than the
 * prefetch cursor advances; the zb-subloop catches up), and stay small enough
 * that the in-flight window never evicts the scratch volume from the node's
 * 1 MB L2. */
#ifndef FFT36_PFD
# define FFT36_PFD 4096
#endif
/* prefetch hint: 3=T0 (all levels), 2=T1 (L2+), 1=T2, 0=NTA */
#ifndef FFT36_PFH
# define FFT36_PFH 2
#endif
/* cache lines of the NEXT volume's input prefetched per phase-2 tile */
#ifndef FFT36_PFN
# define FFT36_PFN 3
#endif
/* pf=2: distance (doubles) the phase-1 write-intent cursor runs ahead of the
 * mid-plane stores.  One plane = 2592 doubles = 20.25 KB gives every line an
 * 18-27-iteration lead (0.5-1.5 planes, ~1.5-4 us) -- long enough to cover a
 * DRAM RFO, short enough that nothing prefetched is evicted before its store. */
#ifndef FFT36_PFWD
# define FFT36_PFWD 2592
#endif
/* pf=3/4: constant lead (doubles) of the NTA in-read cursor.  512 doubles =
 * 4 KB = 64 lines: ~18x the ~220 B a ~70-cycle L3 latency needs at the
 * observed consumption rate, small enough that the L1 footprint (4 KB of
 * NTA lines against the 20 KB plane buffer) does not evict lines before use.
 * NTA lines evicted from L1 were never in L2, so a too-large lead silently
 * doubles the L3 read -- keep this SHORT. */
#ifndef FFT36_PFDN
# define FFT36_PFDN 512
#endif
/* pf=5/6: lead, in 64-B lines per stream, of the deep T1 prefetch of phase
 * 2's 36 source streams (1 line = 1 tile = ~250 port-0 cycles at PW=4).
 * 4 lines ~ 1000 cycles: enough for L3 (~70 cyc) even when all 36 streams
 * miss at once; footprint 36*4 lines = 9 KB of L2, negligible. */
#ifndef FFT36_PF2D
# define FFT36_PF2D 4
#endif

/* W3 = exp(-2*pi*i/3): sqrt(3)/2 (also the radix-3 spine of the 9-point DAG) */
#define KS3  0.86602540378443864676372317075294
/* genfft n1_9 (FMA form) DAG constants, fftw-3.3.10 dft/scalar/codelets/
 * n1_9.c -- transcription taken verbatim from L45_pfa round panel_r9 */
#define K176 0.17632698070846497347109038686862  /* tan(pi/18)                */
#define K839 0.83909963117728001176312729812318
#define K777 0.77786191343020616002817797731863
#define K984 0.98480775301220805936674302458952
#define K492 0.49240387650610402968337151229476
#define K852 0.85286853195244320962825096394007
#define K363 0.36397023426620236135104788277683  /* tan(pi/9)                 */
#define K954 0.95418889413867113349926836418725

/* instantiate the kernel template at 256-bit, and at 512-bit where possible */
#define PW 2
#include __FILE__
#undef PW
#ifdef __AVX512F__
# define PW 4
# include __FILE__
# undef PW
# define HAVE_PW4 1
#endif

/* ---- plan, tuner, API ---------------------------------------------------- */

enum { M_INPLACE = 0, M_SCRATCH = 1, M_SCRATCH_NT = 2, M_PIPE = 3 };
static const char *const mode_name[] = {"inplace", "scratch", "scratch+nt", "pipe"};

struct fft3d_plan {
    int     batch;
    int     pw;                  /* 2 or 4                         */
    int     mode;                /* one of M_*                     */
    int     pf;                  /* software-prefetch the in stream */
    int     pfwd;                /* phase-1 write-intent cursor lead (doubles) */
    int     tr;                  /* 1 = transpose-free phase 1 (ICX, pw4 only) */
    double *S;                   /* reused scratch volume          */
    double *S1;                  /* second scratch (M_PIPE ping-pong) */
    void   *rawS, *rawS1;
};

const char *fft3d_name(void) { return "L36_pfa"; }

/* Filled by fft3d_create() with the tuner's pick, so the leaderboard shows
 * which variant actually ran (r3 verdict requirement; mechanism borrowed from
 * L36_mixedradix / L6_pfa). */
static char g_desc[288];

const char *fft3d_description(void)
{
    return g_desc[0] ? g_desc
                     : "Good-Thomas PFA 4x9 (genfft n1_9 DFT9 DAG), "
                       "interleaved-complex lanes, two sweeps; {inplace, "
                       "scratch(+NT), pipe} x {pw,pf} autotuned in create()";
}
int fft3d_supports(int Lq) { return Lq == L; }

static void run_vols(int pw, int mode, int pf, int pfwd, int tr,
                     double *S, double *S1,
                     const double *in, double *out, int nvol)
{
    if (mode == M_PIPE) {
#ifdef HAVE_PW4
        if (pw == 4) pipe_vols_pw4(in, out, S, S1, nvol, pf);
        else
#endif
        { (void)pw; pipe_vols_pw2(in, out, S, S1, nvol, pf); }
        _mm_sfence();
        return;
    }
    /* pf=2/3 add write-intent prefetch to whichever pass stores cold lines:
     * phase 1 when mid==out (INPLACE), phase 2 when out is store-cold and the
     * stores are normal (SCRATCH).  NT stores do not RFO, nothing to hide.
     * pf=7 (r11) is the write-intent mechanism ALONE, no read cursor.
     * pfr is the read-prefetch kind: 0 none, 1 paced T1 (pf=1/2/6), 2 NTA at
     * consumption rate (pf=3/4).  p2d (pf=5/6) is the deep T1 staging of
     * phase 2's source streams. */
    const int p1w = ((pf == 2 || pf == 3 || pf == 7) && mode == M_INPLACE);
    const int p2w = ((pf == 2 || pf == 3 || pf == 7) && mode == M_SCRATCH);
    const int p2d = (pf >= 5);
    const int pfr = (pf == 3 || pf == 4) ? 2
                  : (pf == 1 || pf == 2 || pf == 6) ? 1 : 0;
    for (int b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDBL;
        double       *o = out + (size_t)b * VDBL;
        double     *mid = (mode == M_INPLACE) ? o : S;
        const int    nt = (mode == M_SCRATCH_NT);
        /* next volume's input, pre-covered from inside phase 2 (pf=1/2 only:
         * pf=3/4 target cache-resident regimes where the cold window is an
         * L3 hit, and 62 KB of pre-coverage would churn L2 for nothing) */
        const double *nx = (pf == 1 || pf == 2) && b + 1 < nvol ? i + VDBL : NULL;
        /* FFT36_SKIP1/2: temporary diagnostics, wrong answers, timing only */
#ifndef FFT36_SKIP1
# define P1(call) (call)
#else
# define P1(call) ((void)0)
#endif
#ifndef FFT36_SKIP2
# define P2(call) (call)
#else
# define P2(call) ((void)0)
#endif
        /* r9: the pf=0 non-NT path (the node's B=1/B=4 pick since r5) runs
         * the compile-time-specialized bodies -- same operations in the same
         * order, with every prefetch mechanism, flag test and the NT codelet
         * copy dead-coded out of the instruction stream.  Everything else
         * (pf>0, NT) keeps the general bodies, unchanged from r8. */
        if (pf == 0 && !nt) {
#ifdef HAVE_PW4
            if (pw == 4) {
                if (tr) P1(phase1_tr1_pf0_pw4(i, mid));
                else    P1(phase1_pf0_pw4(i, mid));
                P2(phase2_pf0_pw4(mid, o));
            }
            else
#endif
            { (void)pw; (void)tr; P1(phase1_pf0_pw2(i, mid)); P2(phase2_pf0_pw2(mid, o)); }
        } else {
#ifdef HAVE_PW4
            if (pw == 4) {
                if (tr) P1(phase1_tr1_pw4(i, mid, pfr, p1w, pfwd));
                else    P1(phase1_pw4(i, mid, pfr, p1w, pfwd));
                P2(phase2_pw4(mid, o, nt, nx, p2w, p2d));
            }
            else
#endif
            { (void)pw; (void)tr; P1(phase1_pw2(i, mid, pfr, p1w, pfwd)); P2(phase2_pw2(mid, o, nt, nx, p2w, p2d)); }
        }
    }
    if (mode == M_SCRATCH_NT) _mm_sfence();
}

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* hysteresis rank: lower = simpler; a candidate must beat a simpler one by
 * >3% to be installed (r3 verdict: tuner coin-flips cost 3.9-6.7% elsewhere,
 * and pf=1 winning in-arena at B=32 while losing end-to-end is the leading
 * hypothesis for this entry's own r3 regression there; genuine wins measured
 * on wallaby -- pf at B=256, pipe vs scratch+nt -- are all >=10%, so the band
 * costs nothing where the choice is real) */
static int cand_rank(int mode, int pf, int nondef_pfwd)
{
    /* pf complexity: 0 < 1 < 7 < 5 < 2 < 6 < 4 < 3 -- one-mechanism levels
     * (pf=1 paced read, pf=7 write-intent alone, pf=5 deep phase-2 staging)
     * before two-mechanism ones (pf=2 = read + write-intent; pf=6 = 1+5;
     * pf=4 NTA; pf=3 NTA + write-intent).  A non-default PFWD ranks as a
     * tie-break below its default twin (same bit class, prefetch-only). */
    static const int pfc[8] = {0, 1, 4, 7, 6, 3, 5, 2};
    return (pfc[pf] * 4 + mode) * 2 + (nondef_pfwd != 0);
}

fft3d_plan *fft3d_create(int Lq, int batch)
{
    if (Lq != L || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->batch = batch;
    if (posix_memalign(&p->rawS, 64, VDBL * sizeof(double)) != 0) {
        free(p); return NULL;
    }
    p->S = (double *)p->rawS;
    memset(p->S, 0, VDBL * sizeof(double));
    if (posix_memalign(&p->rawS1, 64, VDBL * sizeof(double)) == 0) {
        p->S1 = (double *)p->rawS1;
        memset(p->S1, 0, VDBL * sizeof(double));
    }                            /* on failure: no pipe candidates, rest works */
    p->pw = 2; p->mode = M_SCRATCH; p->pf = 0; p->pfwd = FFT36_PFWD;   /* safe default */

    /* candidate list, first entry doubles as the numerical reference.
     * pf=2/3 (write-intent prefetch) and pf=4 (NTA read alone) exist only
     * where normal stores hit cold lines / where L2 residency of the store
     * target is the prize: INPLACE (phase-1 mid==out) and SCRATCH.
     * r7: SCRATCH_NT additionally gets pf=4 -- NTA-protected in-read keeps S
     * L2-resident while the NT final write avoids the RFO; the only shape
     * whose compulsory traffic (1.5 MB/vol) beats in-place's 2.2 MB floor.
     * (pf=3 for NT would be identical: the write-intent half is mode-gated
     * off, so only the pf=4 spelling is instantiated.) */
    struct cand { int pw, mode, pf, pfwd, tr; } cands[80];
    int nc = 0;
    for (int pf = 0; pf <= 4; ++pf) {
        cands[nc++] = (struct cand){2, M_SCRATCH,    pf, FFT36_PFWD, 0};
        cands[nc++] = (struct cand){2, M_INPLACE,    pf, FFT36_PFWD, 0};
        if (pf <= 1 || pf == 4)
            cands[nc++] = (struct cand){2, M_SCRATCH_NT, pf, FFT36_PFWD, 0};
        if (pf <= 1 && p->S1)
            cands[nc++] = (struct cand){2, M_PIPE, pf, FFT36_PFWD, 0};
#ifdef HAVE_PW4
        cands[nc++] = (struct cand){4, M_SCRATCH,    pf, FFT36_PFWD, 0};
        cands[nc++] = (struct cand){4, M_INPLACE,    pf, FFT36_PFWD, 0};
        if (pf <= 1 || pf == 4)
            cands[nc++] = (struct cand){4, M_SCRATCH_NT, pf, FFT36_PFWD, 0};
        if (pf <= 1 && p->S1)
            cands[nc++] = (struct cand){4, M_PIPE, pf, FFT36_PFWD, 0};
        /* ice_r2: tr=1 (transpose-free phase 1, port-5 relief on ICX) twins
         * for the shapes the node actually picks; pf=3/4 (NTA) excluded --
         * 0-for-every-tournament, and their phase-1 pacing is orthogonal. */
        if (pf <= 2) {
            cands[nc++] = (struct cand){4, M_SCRATCH, pf, FFT36_PFWD, 1};
            cands[nc++] = (struct cand){4, M_INPLACE, pf, FFT36_PFWD, 1};
        }
#endif
    }
    /* r8: pf=5 (deep T1 staging of phase 2's 36 source streams, on top of
     * the always-on 1-line T0) and pf=6 (= pf=5 + the pf=1 paced in-read).
     * INPLACE only: B=1 is the target cell; the r7 verdict orders the
     * batched cells left alone, and inplace is the node's pick in every
     * cell where phase 2 reads potentially-evicted lines. */
    for (int pf = 5; pf <= 6; ++pf) {
        cands[nc++] = (struct cand){2, M_INPLACE, pf, FFT36_PFWD, 0};
#ifdef HAVE_PW4
        cands[nc++] = (struct cand){4, M_INPLACE, pf, FFT36_PFWD, 0};
#endif
    }
    /* r11: pf=7, the write-intent prefetch UNBUNDLED from the paced read
     * cursor (pf=2 minus PFIN).  The one B=1 shape never fielded: at B=1
     * the read half is pure uop tax (single linear HW-prefetchable stream)
     * and has ridden along in every pf=2 tournament since r5. */
    cands[nc++] = (struct cand){2, M_INPLACE, 7, FFT36_PFWD, 0};
    cands[nc++] = (struct cand){2, M_SCRATCH, 7, FFT36_PFWD, 0};
#ifdef HAVE_PW4
    cands[nc++] = (struct cand){4, M_INPLACE, 7, FFT36_PFWD, 0};
    cands[nc++] = (struct cand){4, M_SCRATCH, 7, FFT36_PFWD, 0};
    cands[nc++] = (struct cand){4, M_INPLACE, 7, FFT36_PFWD, 1};
    cands[nc++] = (struct cand){4, M_SCRATCH, 7, FFT36_PFWD, 1};
#endif
    /* r11: the FFT36_PFWD sweep at streaming batch, self-served (the node
     * A/B the records have requested since r5).  Only for the shape the
     * node has actually picked in streaming cells (inplace-pf2). */
    if (batch > 8) {
        static const int wds[2] = {1296, 5184};
        for (int w = 0; w < 2; ++w) {
            cands[nc++] = (struct cand){2, M_INPLACE, 2, wds[w], 0};
#ifdef HAVE_PW4
            cands[nc++] = (struct cand){4, M_INPLACE, 2, wds[w], 0};
#endif
        }
    }
    /* r10: close the pick lottery (r9 verdict: B=1 122.58/131.06/122.64 and
     * B=4 131.95/184.11/132.48 make this entry's small-batch numbers
     * unusable).  At nv == batch <= 8 the list is restricted to the shapes
     * the node has actually picked there in five consecutive scored rounds
     * (r5-r9): {inplace, scratch} x pf<=2.  NT, pipe, NTA (pf=3/4) and
     * deep-T1 (pf=5/6) are 0-for-every-tournament at B=1/B=4 and contribute
     * only tail risk in those cells; they all remain candidates at streaming
     * batch.  Any FFT36_* / FFT_FORCE_* override keeps the full list so the
     * monitor's forced controls still work. */
    {
        int small_list = (batch <= 8) &&
            !getenv("FFT36_PW") && !getenv("FFT36_MODE") && !getenv("FFT36_PF") &&
            !getenv("FFT36_TR");
#if defined(FFT_FORCE_PW) || defined(FFT_FORCE_MODE) || defined(FFT36_FORCE_PF)
        small_list = 0;
#endif
        if (small_list) {
            /* r11: pf=7 (inplace) joins the restricted list -- the one
             * never-tried B=1 shape; the 3% band still requires it to beat
             * pf=0 by >3% before it can install.  PFWD sweep stays out
             * (small-nv arena cannot price a streaming-store knob). */
            int w = 0;
            for (int c = 0; c < nc; ++c)
                if ((cands[c].mode == M_INPLACE || cands[c].mode == M_SCRATCH)
                    && (cands[c].pf <= 2
                        || (cands[c].pf == 7 && cands[c].mode == M_INPLACE))
                    && cands[c].pfwd == FFT36_PFWD) cands[w++] = cands[c];
            nc = w;
        }
    }
    /* run-time forcing for the monitor's control jobs (no recompile needed) */
    { const char *e;
      if ((e = getenv("FFT36_PW"))) {
          int v = atoi(e), w = 0;
          for (int c = 0; c < nc; ++c) if (cands[c].pw == v) cands[w++] = cands[c];
          if (w) nc = w;
      }
      if ((e = getenv("FFT36_MODE"))) {
          int num = (e[0] >= '0' && e[0] <= '9'), w = 0;
          for (int c = 0; c < nc; ++c)
              if (num ? cands[c].mode == atoi(e)
                      : !strcmp(mode_name[cands[c].mode], e)) cands[w++] = cands[c];
          if (w) nc = w;
      }
      if ((e = getenv("FFT36_PF"))) {
          int v = atoi(e), w = 0;
          for (int c = 0; c < nc; ++c) if (cands[c].pf == v) cands[w++] = cands[c];
          if (w) nc = w;
      }
      if ((e = getenv("FFT36_TR"))) {      /* ice_r2: force the phase-1 kernel */
          int v = atoi(e), w = 0;
          for (int c = 0; c < nc; ++c) if (cands[c].tr == v) cands[w++] = cands[c];
          if (w) nc = w;
      }
      if ((e = getenv("FFT36_PFWDF"))) {      /* r11: force a pfwd value */
          int v = atoi(e), w = 0;
          for (int c = 0; c < nc; ++c) {
              if (cands[c].pf == 2 || cands[c].pf == 3 || cands[c].pf == 7) {
                  cands[c].pfwd = v; cands[w++] = cands[c];
              }
          }
          if (w) nc = w;
      } }
#ifdef FFT_FORCE_PW
    { int w = 0;
      for (int c = 0; c < nc; ++c) if (cands[c].pw == FFT_FORCE_PW) cands[w++] = cands[c];
      if (w) nc = w; }
#endif
#ifdef FFT_FORCE_MODE
    { int w = 0;
      for (int c = 0; c < nc; ++c) if (cands[c].mode == FFT_FORCE_MODE) cands[w++] = cands[c];
      if (w) nc = w; }
#endif
#ifdef FFT36_FORCE_PF
    { int w = 0;
      for (int c = 0; c < nc; ++c) if (cands[c].pf == FFT36_FORCE_PF) cands[w++] = cands[c];
      if (w) nc = w; }
#endif

    /* tuning arena: must actually leave L3 at large batch, or the NT-vs-
     * cached-store ranking inverts (a 16-volume arena fit wallaby's 60 MB L3
     * and mis-picked cached stores at B=256; 64 volumes = 96 MB does not) */
    const int nv = batch < 64 ? batch : 64;
    void *ri = NULL, *ro = NULL, *rr = NULL;
    if (posix_memalign(&ri, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&ro, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&rr, 64, (size_t)nv * VDBL * sizeof(double))) {
        free(ri); free(ro);                 /* keep the safe default */
        p->pw = cands[0].pw; p->mode = cands[0].mode; p->pf = cands[0].pf;
        p->pfwd = cands[0].pfwd; p->tr = cands[0].tr;
        snprintf(g_desc, sizeof g_desc,
                 "GT-PFA 4x9 two-sweep; tuner SKIPPED (arena alloc failed): "
                 "pw=%d mode=%s pf=%d", p->pw, mode_name[p->mode], p->pf);
        return p;
    }
    double *tin = ri, *tout = ro, *ref = rr;
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        tin[i] = (double)(long long)(s >> 11) * 0x1p-53;
    }

    run_vols(cands[0].pw, cands[0].mode, cands[0].pf, cands[0].pfwd, cands[0].tr,
             p->S, p->S1, tin, ref, nv);

    int    ok[80];
    double tc[80];
    for (int c = 0; c < nc; ++c) {
        run_vols(cands[c].pw, cands[c].mode, cands[c].pf, cands[c].pfwd, cands[c].tr,
                 p->S, p->S1, tin, tout, nv);
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
            double d = tout[i] - ref[i];
            num += d * d; den += ref[i] * ref[i];
        }
        ok[c] = (num <= den * 1e-26);       /* rel L2 < 1e-13 vs reference */
        tc[c] = 1e300;
    }
    const int R = (nv >= 8) ? 1 : (nv >= 2 ? 3 : 8);
    /* r10: 9 rounds at small nv (was 5) -- min-of-more against the window
     * wobble that fed the pick lottery; setup stays well under a second. */
    const int NROUND = (nv <= 8) ? 9 : 5;
    for (int round = 0; round < NROUND; ++round)
        for (int c = 0; c < nc; ++c) {
            if (!ok[c]) continue;
            /* self-warming (adopted from L36_pencilfused r5): one untimed
             * exec so each candidate is timed from its OWN steady-state cache,
             * not its predecessor's -- an NT/pipe candidate flushes tout and
             * was charging its successor a deterministic cold-RFO penalty
             * (their measured artifact: 167.4 vs a true 89.8 us/vol). */
            run_vols(cands[c].pw, cands[c].mode, cands[c].pf, cands[c].pfwd,
                     cands[c].tr, p->S, p->S1, tin, tout, nv);
            double t0 = now_s();
            for (int r = 0; r < R; ++r)
                run_vols(cands[c].pw, cands[c].mode, cands[c].pf, cands[c].pfwd,
                         cands[c].tr, p->S, p->S1, tin, tout, nv);
            double t = (now_s() - t0) / R;
            if (t < tc[c]) tc[c] = t;
        }
    int best = 0;
    for (int c = 1; c < nc; ++c) if (ok[c] && tc[c] < tc[best]) best = c;
    if (ok[best]) {
        /* hysteresis: any simpler candidate within the band wins.  r10:
         * uniform 3% again -- the r8 1% band at small nv was built to admit
         * 1-3% pf wins that r9 showed do not exist (prefetch instruments
         * 0-for-three-tournaments at B=1; the verdict's rule: "+-2% at these
         * cells is the code-layout noise floor... stop building fixes for
         * it").  All it bought was pick instability. */
        const double band = 1.03;
        int pick = best;
        for (int c = 0; c < nc; ++c)
            if (ok[c] && tc[c] <= tc[best] * band &&
                cand_rank(cands[c].mode, cands[c].pf, cands[c].pfwd != FFT36_PFWD) <
                cand_rank(cands[pick].mode, cands[pick].pf, cands[pick].pfwd != FFT36_PFWD)) pick = c;
        p->pw = cands[pick].pw; p->mode = cands[pick].mode; p->pf = cands[pick].pf;
        p->pfwd = cands[pick].pfwd; p->tr = cands[pick].tr;
    }

    /* r8 node probes: the phase split at nv=1 steady state, reported through
     * fft3d_description() so the leaderboard carries the measurement the
     * records have requested from the monitor for three rounds.  Fixed
     * config (pw4 inplace pf0 = the node's standing B=1 pick) regardless of
     * the tuner's choice, so the numbers are comparable across rounds:
     *   p1  = phase 1 alone, steady state (honest: repeating it re-streams
     *         in->tout, 1.5 MB, the same L2 pressure as the real execute)
     *   p2w = phase 2 alone, repeated in place on tout (OPTIMISTIC: its
     *         746 KB working set fits the 1 MB L2 once warm, so this is the
     *         compute + L1/L2 floor of phase 2, with no phase-1 eviction)
     *   p2wd= same with the pf=5 deep T1 staging (its pure uop tax shows
     *         here; any gain it has needs the cold-L2 context, i.e. fu)
     *   fu  = phase1 + phase2, the full pf0-inplace execute
     * fu - p1 - p2w = the memory penalty phase 2 pays for the phase-1
     * boundary; if it is small, the B=1 residual is NOT L2 thrash and the
     * lever is front-end/scheduling, closing the fork in the r7 verdict. */
#ifdef HAVE_PW4
# define PRB1(a,b)       phase1_pf0_pw4(a, b)
# define PRB2(a,b)       phase2_pf0_pw4(a, b)
# define PRBZ(a)         phase1_zonly_pw4(a)
# define PRBY(s,b)       phase1_yonly_pw4(s, b)
# define PRBT(a,b)       phase1_tr1_pf0_pw4(a, b)
#else
# define PRB1(a,b)       phase1_pf0_pw2(a, b)
# define PRB2(a,b)       phase2_pf0_pw2(a, b)
# define PRBZ(a)         phase1_zonly_pw2(a)
# define PRBY(s,b)       phase1_yonly_pw2(s, b)
#endif
    /* r10: fug and the perf_event group are retired -- both questions they
     * were built to answer closed in r9 (node fug - fu = +0.3%: code size is
     * not the B=1 residual; fe=na on the node too, counters withdrawn).
     * p1/p2w/fu stay for cross-round phase-split comparability.
     * r11: p1z/p1y split phase 1 into its two subloops (verdict's ask: locate
     * the 35 us excess).  p1z = z-subloop only (in-read + transposes +
     * codelet + pl write, pl kept live by an asm sink); p1y = y-subloop only
     * (codelet + mid stores = the RFO side), reading a fixed L1-resident
     * 20.25 KB plane from S.  Each is a lower bound for its side (the other
     * stream's L2 pressure is absent); port share ~26 us each at 2.89 GHz. */
    double pb1 = 1e300, pb2 = 1e300, pbf = 1e300, pbz = 1e300, pby = 1e300;
    double pbt = 1e300;
    {
        double t;
        const int PR = 4;
        for (int round = 0; round < 3; ++round) {
            PRB1(tin, tout);                                /* self-warm */
            t = now_s();
            for (int r = 0; r < PR; ++r) PRB1(tin, tout);
            t = (now_s() - t) / PR;  if (t < pb1) pb1 = t;
#ifdef HAVE_PW4
            PRBT(tin, tout);                                /* ice_r2: tr=1 */
            t = now_s();
            for (int r = 0; r < PR; ++r) PRBT(tin, tout);
            t = (now_s() - t) / PR;  if (t < pbt) pbt = t;
#endif
            PRBZ(tin);
            t = now_s();
            for (int r = 0; r < PR; ++r) PRBZ(tin);
            t = (now_s() - t) / PR;  if (t < pbz) pbz = t;
            PRBY(p->S, tout);
            t = now_s();
            for (int r = 0; r < PR; ++r) PRBY(p->S, tout);
            t = (now_s() - t) / PR;  if (t < pby) pby = t;
            PRB2(tout, tout);
            t = now_s();
            for (int r = 0; r < PR; ++r) PRB2(tout, tout);
            t = (now_s() - t) / PR;  if (t < pb2) pb2 = t;
            PRB1(tin, tout); PRB2(tout, tout);
            t = now_s();
            for (int r = 0; r < PR; ++r) { PRB1(tin, tout); PRB2(tout, tout); }
            t = (now_s() - t) / PR;  if (t < pbf) pbf = t;
        }
    }
    {
        char wd[20];
        wd[0] = 0;
        if (p->pfwd != FFT36_PFWD) snprintf(wd, sizeof wd, " pfwd=%d", p->pfwd);
        snprintf(g_desc, sizeof g_desc,
                 "GT-PFA 4x9 (n1_9 DAG) two-sweep; tuner pick: pw=%d mode=%s pf=%d"
                 " tr=%d%s (B=%d, nv=%d, nc=%d); probe us p1=%.1f p1t=%.1f"
                 " p1z=%.1f p1y=%.1f p2w=%.1f fu=%.1f",
                 p->pw, mode_name[p->mode], p->pf, p->tr, wd, batch, nv, nc,
                 pb1 * 1e6, pbt < 1e299 ? pbt * 1e6 : -1.0,
                 pbz * 1e6, pby * 1e6, pb2 * 1e6, pbf * 1e6);
    }

#ifdef FFT36_LOUD
    if (1) {
#else
    if (getenv("FFT36_VERBOSE")) {
#endif
        for (int c = 0; c < nc; ++c)
            fprintf(stderr, "L36_pfa tuner: pw=%d mode=%-10s pf=%d pfwd=%-4d tr=%d %s  %.1f us/vol\n",
                    cands[c].pw, mode_name[cands[c].mode], cands[c].pf,
                    cands[c].pfwd, cands[c].tr,
                    ok[c] ? "ok " : "BAD", ok[c] ? tc[c] * 1e6 / nv : 0.0);
        fprintf(stderr, "L36_pfa tuner: chose pw=%d mode=%s pf=%d pfwd=%d tr=%d (nv=%d)\n",
                p->pw, mode_name[p->mode], p->pf, p->pfwd, p->tr, nv);
    }
    free(ri); free(ro); free(rr);
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    free(p->rawS); free(p->rawS1); free(p);
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    run_vols(plan->pw, plan->mode, plan->pf, plan->pfwd, plan->tr,
             plan->S, plan->S1,
             (const double *)in, (double *)out, plan->batch);
}

#else /* ================= KERNEL TEMPLATE, PW = 2 or 4 ==================== */

#define vec   CAT(vec_pw,  PW)
#define veci  CAT(veci_pw, PW)
#define FN(n) CAT(n, CAT(_pw, PW))

typedef double    vec  __attribute__((vector_size(PW * 16)));
typedef long long veci __attribute__((vector_size(PW * 16)));

#ifdef __clang__
# define VSH(a,b,...) __builtin_shufflevector(a, b, __VA_ARGS__)
#else
# define VSH(a,b,...) __builtin_shuffle(a, b, (veci){__VA_ARGS__})
#endif

#define NVR (L / PW)             /* vectors per 36-complex row: 18 or 9 */

#if PW == 4
# define VSPLAT(a)  ((vec){(a),(a),(a),(a),(a),(a),(a),(a)})
# define VPAIR(a,b) ((vec){(a),(b),(a),(b),(a),(b),(a),(b)})
# define SWAP(v)    VSH((v),(v), 1,0,3,2,5,4,7,6)
# define VFMA(a,b,c)  ((vec)_mm512_fmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
# define VFNMA(a,b,c) ((vec)_mm512_fnmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
# define STREAM_ST(p,v) _mm512_stream_pd((p), (__m512d)(v))
#else
# define VSPLAT(a)  ((vec){(a),(a),(a),(a)})
# define VPAIR(a,b) ((vec){(a),(b),(a),(b)})
# define SWAP(v)    VSH((v),(v), 1,0,3,2)
# ifdef __FMA__
#  define VFMA(a,b,c)  ((vec)_mm256_fmadd_pd((__m256d)(a),(__m256d)(b),(__m256d)(c)))
#  define VFNMA(a,b,c) ((vec)_mm256_fnmadd_pd((__m256d)(a),(__m256d)(b),(__m256d)(c)))
# else
#  define VFMA(a,b,c)  ((a)*(b) + (c))
#  define VFNMA(a,b,c) ((c) - (a)*(b))
# endif
# ifdef __AVX__
#  define STREAM_ST(p,v) _mm256_stream_pd((p), (__m256d)(v))
# else
#  define STREAM_ST(p,v) (*(vec *)(p) = (v))
# endif
#endif

/* PW x PW transpose of 128-bit complex granules; involution, 8 shuffles per
 * 4 vectors at PW=4, 2 per 2 vectors at PW=2. */
#if PW == 4
# define TRNC(r, c) do {                                                     \
    vec u0_ = VSH((r)[0], (r)[1], 0,1,8,9,4,5,12,13);                        \
    vec u1_ = VSH((r)[0], (r)[1], 2,3,10,11,6,7,14,15);                      \
    vec u2_ = VSH((r)[2], (r)[3], 0,1,8,9,4,5,12,13);                        \
    vec u3_ = VSH((r)[2], (r)[3], 2,3,10,11,6,7,14,15);                      \
    (c)[0] = VSH(u0_, u2_, 0,1,2,3,8,9,10,11);                               \
    (c)[2] = VSH(u0_, u2_, 4,5,6,7,12,13,14,15);                             \
    (c)[1] = VSH(u1_, u3_, 0,1,2,3,8,9,10,11);                               \
    (c)[3] = VSH(u1_, u3_, 4,5,6,7,12,13,14,15);                             \
} while (0)
#else
# define TRNC(r, c) do {                                                     \
    vec u0_ = VSH((r)[0], (r)[1], 0,1,4,5);                                  \
    vec u1_ = VSH((r)[0], (r)[1], 2,3,6,7);                                  \
    (c)[0] = u0_; (c)[1] = u1_;                                              \
} while (0)
#endif

/* 9-point DFT on interleaved-complex vectors: genfft's FMA n1_9 DAG
 * (fftw-3.3.10, 24 add + 56 fma = 80 scalar FMA-port ops) transcribed
 * pairwise -- each scalar re/im line pair is one vector op; each re/im
 * crossing (mult by i, and the (1 +- c*i) spiral factors on p/q) is one
 * SWAP with the signs folded into a VPAIR constant.
 * 40 FMA-port ops + 12 swaps (the CT 3x3 it replaces: 44 + 10).
 * Transcription taken VERBATIM from L45_pfa round panel_r9 (their DFT9F;
 * same vec/VPAIR/SWAP/VFMA dialect -- this file's macros are its lineage).
 * Scalar-to-vector key, stage A (radix-3 columns {n, n+3, n+6}):
 *   sJ_ = column sum, SJ_ = full sum, aJ_ = xJ - sJ/2, eJ_ = x(J+3)-x(J+6),
 *   iJ_ = SWAP(eJ_);  p/q = aJ -+ 866*i*eJ  (the two rotated DFT3 outputs).
 * Blocks: k={0,3,6} is a DFT3 on the sums; k={1,4,7} and k={2,5,8} build
 * w = (1 +- c*i)*p (one SWAP+FMA each), cross them (u, z), and fan out. */
#define DFT9F(x0,x1,x2,x3,x4,x5,x6,x7,x8, o0,o1,o2,o3,o4,o5,o6,o7,o8) do {   \
    vec s0_ = (x3) + (x6), e0_ = (x3) - (x6);                                \
    vec S0_ = (x0) + s0_,  a0_ = VFNMA(s0_, VSPLAT(0.5), (x0));              \
    vec i0_ = SWAP(e0_);                                                     \
    vec s1_ = (x4) + (x7), e1_ = (x4) - (x7);                                \
    vec S1_ = (x1) + s1_,  a1_ = VFNMA(s1_, VSPLAT(0.5), (x1));              \
    vec i1_ = SWAP(e1_);                                                     \
    vec p1_ = VFMA (i1_, VPAIR(KS3, -KS3), a1_);                             \
    vec q1_ = VFNMA(i1_, VPAIR(KS3, -KS3), a1_);                             \
    vec s2_ = (x5) + (x8), e2_ = (x5) - (x8);                                \
    vec S2_ = (x2) + s2_,  a2_ = VFNMA(s2_, VSPLAT(0.5), (x2));              \
    vec i2_ = SWAP(e2_);                                                     \
    vec p2_ = VFMA (i2_, VPAIR(KS3, -KS3), a2_);                             \
    vec q2_ = VFNMA(i2_, VPAIR(KS3, -KS3), a2_);                             \
    /* k = 0, 3, 6: DFT3 on the column sums */                               \
    vec sg_ = S1_ + S2_, d3_ = S2_ - S1_, id_ = SWAP(d3_);                   \
    vec b0_ = VFNMA(sg_, VSPLAT(0.5), S0_);                                  \
    (o0) = S0_ + sg_;                                                        \
    (o3) = VFNMA(id_, VPAIR(KS3, -KS3), b0_);                                \
    (o6) = VFMA (id_, VPAIR(KS3, -KS3), b0_);                                \
    /* k = 1, 4, 7 */                                                        \
    {                                                                        \
    vec v1_ = VFMA (i0_, VPAIR(KS3, -KS3), a0_);                             \
    vec w2_ = VFMA (SWAP(p2_), VPAIR(-K176, K176), p2_);                     \
    vec w1_ = VFMA (SWAP(p1_), VPAIR(K839, -K839), p1_);                     \
    vec u1_ = VFMA (w1_, VPAIR(K777, -K777), SWAP(w2_));                     \
    vec z1_ = VFMA (SWAP(w1_), VPAIR(K777, -K777), w2_);                     \
    (o1) = VFMA (u1_, VPAIR(K984, -K984), v1_);                              \
    vec r1_ = VFNMA(u1_, VPAIR(K492, -K492), v1_);                           \
    (o4) = VFMA (z1_, VSPLAT(K852), r1_);                                    \
    (o7) = VFNMA(z1_, VSPLAT(K852), r1_);                                    \
    }                                                                        \
    /* k = 2, 5, 8 */                                                        \
    {                                                                        \
    vec v2_ = VFNMA(i0_, VPAIR(KS3, -KS3), a0_);                             \
    vec wA_ = VFMA (q1_, VPAIR(K176, -K176), SWAP(q1_));                     \
    vec wB_ = VFNMA(SWAP(q2_), VPAIR(K363, -K363), q2_);                     \
    vec uB_ = VFMA (wB_, VPAIR(-K954, K954), wA_);                           \
    vec zB_ = VFMA (SWAP(wB_), VPAIR(-K954, K954), SWAP(wA_));               \
    (o2) = VFMA (uB_, VPAIR(K984, -K984), v2_);                              \
    vec rB_ = VFNMA(uB_, VPAIR(K492, -K492), v2_);                           \
    (o5) = VFNMA(zB_, VSPLAT(K852), rB_);                                    \
    (o8) = VFMA (zB_, VSPLAT(K852), rB_);                                    \
    }                                                                        \
} while (0)

/* The 36-point Good-Thomas 4x9 codelet over PW interleaved-complex lanes.
 * LD(n) must yield input element n as a vec rvalue; ST(k, v) must consume
 * output element k.  Both index maps fold to compile-time constants once the
 * loops unroll.  All LD reads happen before the first ST, so LD/ST may alias. */
#define PFA36(LD, ST) do {                                                   \
    vec A_[36];                            /* A_[9*k1 + n2] */               \
    _Pragma("GCC unroll 9")                                                  \
    for (int n2_ = 0; n2_ < 9; ++n2_) {                                      \
        vec x0_ = LD(( 0 + 4*n2_) % 36), x1_ = LD(( 9 + 4*n2_) % 36);        \
        vec x2_ = LD((18 + 4*n2_) % 36), x3_ = LD((27 + 4*n2_) % 36);        \
        vec t0_ = x0_ + x2_, t1_ = x0_ - x2_;                                \
        vec t2_ = x1_ + x3_, t3_ = x1_ - x3_;                                \
        vec sw_ = SWAP(t3_);                                                 \
        A_[     n2_] = t0_ + t2_;                                            \
        A_[18 + n2_] = t0_ - t2_;                                            \
        A_[ 9 + n2_] = VFMA (sw_, VPAIR(1.0, -1.0), t1_);                    \
        A_[27 + n2_] = VFNMA(sw_, VPAIR(1.0, -1.0), t1_);                    \
    }                                                                        \
    _Pragma("GCC unroll 4")                                                  \
    for (int k1_ = 0; k1_ < 4; ++k1_) {                                      \
        const vec *g_ = A_ + 9 * k1_;                                        \
        vec Y_[9];                         /* DFT9 outputs, natural order */ \
        DFT9F(g_[0], g_[1], g_[2], g_[3], g_[4], g_[5], g_[6], g_[7], g_[8],\
              Y_[0], Y_[1], Y_[2], Y_[3], Y_[4], Y_[5], Y_[6], Y_[7], Y_[8]);\
        _Pragma("GCC unroll 9")                                              \
        for (int m_ = 0; m_ < 9; ++m_)                                       \
            ST((9*k1_ + 28*m_) % 36, Y_[m_]);                                \
    }                                                                        \
} while (0)

/* Paced prefetch of the (perfectly linear) `in` stream.  Each of the 2*NVR
 * loop iterations per x-plane advances the cursor by PFSTEP doubles, so one
 * plane's worth of prefetches issues per plane processed, spread evenly over
 * both subloops (the zb subloop touches no `in` bytes, so pacing through it
 * keeps the DRAM read stream busy during the y transform too). */
#define PFSTEP (36 * PW)                   /* doubles per iteration            */
#define PFIN(p) do {                                                          \
    _Pragma("GCC unroll 18")                                                  \
    for (int q_ = 0; q_ < PFSTEP / 8; ++q_)                                   \
        __builtin_prefetch((p) + 8 * q_, 0, FFT36_PFH);                       \
} while (0)

/* pf=3/4: NTA prefetch of the in stream at EXACTLY the consumption rate of
 * the yb-subloop (which reads 2*PFSTEP doubles per iteration); the zb-subloop
 * touches no in bytes and issues nothing, so the cursor lead is a constant
 * FFT36_PFDN.  NTA fills L1 and bypasses L2 on SKX-class cores, so the read
 * stream stops evicting `out` / S from L2 -- the whole point (r6 header). */
#define PFIN_NTA(p) do {                                                      \
    _Pragma("GCC unroll 36")                                                  \
    for (int q_ = 0; q_ < PFSTEP / 4; ++q_)                                   \
        __builtin_prefetch((p) + 8 * q_, 0, 0);                               \
} while (0)

/* pf=2 (M_INPLACE): paced WRITE-INTENT cursor over the mid-plane store
 * stream, same pacing arithmetic as PFIN (one plane's worth per plane
 * processed), one plane ahead.  __builtin_prefetch(p,1,3) emits `prefetchw`
 * on PRFCHW machines (Cascade Lake, Sapphire Rapids), acquiring the line
 * exclusive before the store so the cold-out RFO overlaps compute instead of
 * stalling the store buffer.  Adopted from L6_unrolled r3 (their fused_pfw,
 * -29% at DRAM sizes; node-confirmed via L6_pfa r4). */
#define PFWMID(p) do {                                                        \
    _Pragma("GCC unroll 18")                                                  \
    for (int q_ = 0; q_ < PFSTEP / 8; ++q_)                                   \
        __builtin_prefetch((p) + 8 * q_, 1, 3);                               \
} while (0)

/* phase 1, ONE x-plane: z transform (transposed lanes) then y transform.
 * Reads plane x of `in` sequentially; writes mid[x][ky][kz].  The paced
 * prefetch cursor is stateless across planes: each plane issues exactly one
 * plane's worth (2*LSQ doubles) of prefetches, 2*NVR iterations x PFSTEP,
 * so cursor(x) = in + PFD + x*2*LSQ reproduces the r3 per-volume pacing
 * exactly while letting M_PIPE call planes individually. */
static inline __attribute__((always_inline))
void FN(phase1_plane)(const double *restrict in, double *restrict mid,
                      int x, int pfr, int pfw, int pfwd)
{
    vec pl[L * NVR];                       /* plane [y][kz], 20.25 KB */
    const double *pfc = in  + FFT36_PFD  + (size_t)x * (2 * LSQ);
    const double *pfn = in  + FFT36_PFDN + (size_t)x * (2 * LSQ);
    double       *pwc = mid + pfwd + (size_t)x * (2 * LSQ);
    const double *px  = in  + (size_t)x * (2 * LSQ);
    double       *mx  = mid + (size_t)x * (2 * LSQ);

    for (int yb = 0; yb < L; yb += PW) {
        if (pfr == 1)      { PFIN(pfc);     pfc += PFSTEP; }
        else if (pfr == 2) { PFIN_NTA(pfn); pfn += 2 * PFSTEP; }
        if (pfw) { PFWMID(pwc); pwc += PFSTEP; }
        vec Zv[36], Wv[36];
        _Pragma("GCC unroll 18")
        for (int zb = 0; zb < NVR; ++zb) {
            vec r_[PW];
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j)
                r_[j] = *(const vec *)(px + ((size_t)(yb + j) * L + (size_t)zb * PW) * 2);
            TRNC(r_, &Zv[zb * PW]);
        }
#define LD1(n)    Zv[n]
#define ST1(k, v) (Wv[k] = (v))
        PFA36(LD1, ST1);
#undef LD1
#undef ST1
        _Pragma("GCC unroll 18")
        for (int zb = 0; zb < NVR; ++zb) {
            vec r_[PW];
            TRNC(&Wv[zb * PW], r_);
            _Pragma("GCC unroll 4")
            for (int j = 0; j < PW; ++j)
                pl[(size_t)(yb + j) * NVR + zb] = r_[j];
        }
    }

    for (int zb = 0; zb < NVR; ++zb) {
        if (pfr == 1) { PFIN(pfc); pfc += PFSTEP; }
        if (pfw) { PFWMID(pwc); pwc += PFSTEP; }
#define LD2(n)    pl[(size_t)(n) * NVR + zb]
#define ST2(k, v) (*(vec *)(mx + ((size_t)(k) * L + (size_t)zb * PW) * 2) = (v))
        PFA36(LD2, ST2);
#undef LD2
#undef ST2
    }
}

static void FN(phase1)(const double *restrict in, double *restrict mid,
                       int pfr, int pfw, int pfwd)
{
    for (int x = 0; x < L; ++x)
        FN(phase1_plane)(in, mid, x, pfr, pfw, pfwd);
}

#if PW == 4
/* ice_r2, tr=1: transpose-free phase 1.  On ICX-SP the second 512-bit FMA
 * pipe shares port 5 with every 512-bit shuffle, so TRNC's 8 shuffles per 4
 * vectors displace FMAs; but vbroadcastf64x2 (mem), zmm{k} is a single
 * LOAD-port uop (broadcast folded into the load on bare metal, corpus S10).
 * Each lane-transposed vector is built as 4 masked 128-bit broadcast loads:
 * lane j gets the complex granule at g_j.  4-deep merge chain per vector,
 * ~36 independent vectors in flight -- OOO covers the latency.  The kmov
 * mask setups are loop-invariant and hoisted by gcc (verified on the .o). */
static inline __attribute__((always_inline))
vec FN(ldtr4)(const double *g0, const double *g1,
              const double *g2, const double *g3)
{
    __m512d v = _mm512_broadcast_f64x2(_mm_load_pd(g0));
    v = _mm512_mask_broadcast_f64x2(v, 0x0C, _mm_load_pd(g1));
    v = _mm512_mask_broadcast_f64x2(v, 0x30, _mm_load_pd(g2));
    v = _mm512_mask_broadcast_f64x2(v, 0xC0, _mm_load_pd(g3));
    return (vec)v;
}

/* Same two subloops and prefetch pacing as phase1_plane, zero TRNC:
 *   z-subloop: LD = 4 broadcasts from the in-plane rows (lanes = y), the
 *              codelet output goes to pl UNtransposed, pl[ygroup*36 + kz]
 *              with lanes = y (a plain full-width store);
 *   y-subloop: LD = 4 broadcasts re-gathering pl's 128-bit granules into
 *              lanes = kz on the fly; mid stores unchanged.
 * Element (y, kz) lives at pl[(y>>2)*36 + kz], 128-bit lane (y&3). */
static inline __attribute__((always_inline))
void FN(phase1_plane_tr1)(const double *restrict in, double *restrict mid,
                          int x, int pfr, int pfw, int pfwd)
{
    vec pl[L * NVR];                       /* [ygroup][kz], 20.25 KB */
    const double *pfc = in  + FFT36_PFD  + (size_t)x * (2 * LSQ);
    const double *pfn = in  + FFT36_PFDN + (size_t)x * (2 * LSQ);
    double       *pwc = mid + pfwd + (size_t)x * (2 * LSQ);
    const double *px  = in  + (size_t)x * (2 * LSQ);
    double       *mx  = mid + (size_t)x * (2 * LSQ);
    const double *pld = (const double *)pl;

    for (int yb = 0; yb < L; yb += PW) {
        if (pfr == 1)      { PFIN(pfc);     pfc += PFSTEP; }
        else if (pfr == 2) { PFIN_NTA(pfn); pfn += 2 * PFSTEP; }
        if (pfw) { PFWMID(pwc); pwc += PFSTEP; }
#define LD1T(n)    FN(ldtr4)(px + ((size_t)(yb + 0) * L + (n)) * 2,          \
                             px + ((size_t)(yb + 1) * L + (n)) * 2,          \
                             px + ((size_t)(yb + 2) * L + (n)) * 2,          \
                             px + ((size_t)(yb + 3) * L + (n)) * 2)
#define ST1T(k, v) (pl[(size_t)(yb >> 2) * 36 + (k)] = (v))
        PFA36(LD1T, ST1T);
#undef LD1T
#undef ST1T
    }

    for (int zb = 0; zb < NVR; ++zb) {
        if (pfr == 1) { PFIN(pfc); pfc += PFSTEP; }
        if (pfw) { PFWMID(pwc); pwc += PFSTEP; }
#define LD2T(n)    FN(ldtr4)(                                                \
        pld + ((((n) >> 2) * 36 + (size_t)zb * 4 + 0) * 8 + ((n) & 3) * 2),  \
        pld + ((((n) >> 2) * 36 + (size_t)zb * 4 + 1) * 8 + ((n) & 3) * 2),  \
        pld + ((((n) >> 2) * 36 + (size_t)zb * 4 + 2) * 8 + ((n) & 3) * 2),  \
        pld + ((((n) >> 2) * 36 + (size_t)zb * 4 + 3) * 8 + ((n) & 3) * 2))
#define ST2T(k, v) (*(vec *)(mx + ((size_t)(k) * L + (size_t)zb * PW) * 2) = (v))
        PFA36(LD2T, ST2T);
#undef LD2T
#undef ST2T
    }
}

static void FN(phase1_tr1)(const double *restrict in, double *restrict mid,
                           int pfr, int pfw, int pfwd)
{
    for (int x = 0; x < L; ++x)
        FN(phase1_plane_tr1)(in, mid, x, pfr, pfw, pfwd);
}

/* tr=1, pf=0: literal-constant flags so every prefetch block dead-codes out
 * of the scored body (the r9 specialization mechanism, same as phase1_pf0). */
static void FN(phase1_tr1_pf0)(const double *restrict in, double *restrict mid)
{
    for (int x = 0; x < L; ++x)
        FN(phase1_plane_tr1)(in, mid, x, 0, 0, FFT36_PFWD);
}
#endif /* PW == 4 */

/* one x-line-group tile: 36 strided loads -> codelet -> W_arr */
#define TILE(srcp, W_arr) do {                                               \
    const double *s_ = (srcp);                                               \
    vec *W_ = (W_arr);                                                       \
    PF36(s_);                                                                \
    PFA36(LD3, ST3);                                                         \
} while (0)
#define LD3(n)    (*(const vec *)(s_ + (size_t)(n) * (2 * LSQ)))
#define ST3(k, v) (W_[k] = (v))
#ifndef FFT36_NOPF
# define PF36(s_) do {                                                       \
    _Pragma("GCC unroll 36")                                                 \
    for (int n_ = 0; n_ < 36; ++n_)                                          \
        __builtin_prefetch(s_ + (size_t)n_ * (2 * LSQ) + 8, 0, 3);           \
} while (0)
#else
# define PF36(s_) do { } while (0)
#endif

/* Per-tile pre-coverage of the NEXT volume's input: 3 lines per tile fills
 * the first 62 KB (>= FFT36_PFD + the phase-1 pacing deficit) of in[b+1]
 * before its phase 1 starts, so the prefetch cursor never starts cold. */
#define PFNX() do { if (pn_) {                                                \
    _Pragma("GCC unroll 9")                                                   \
    for (int q_ = 0; q_ < FFT36_PFN; ++q_)                                    \
        __builtin_prefetch(pn_ + 8 * q_, 0, FFT36_PFH);                       \
    pn_ += 8 * FFT36_PFN; } } while (0)

/* phase 2, ONE y-plane: x transform, 36 sequential source streams of stride
 * 20736 B.  Safe when mid == out: the codelet reads all 36 inputs before its
 * first store, so results go straight from registers to `out` with no
 * staging.  The next-volume prefetch cursor is recomputed per plane from the
 * per-plane tile count, reproducing the r3 whole-volume walk exactly. */
/* pf=2 (M_SCRATCH): write-intent prefetch of the 36 out-streams one 64-B
 * line ahead of the tile being stored -- mirrors PF36 on the read side. */
#define PFW36(d_) do {                                                       \
    _Pragma("GCC unroll 36")                                                 \
    for (int n_ = 0; n_ < 36; ++n_)                                          \
        __builtin_prefetch((d_) + (size_t)n_ * (2 * LSQ) + 8, 1, 3);         \
} while (0)

/* pf=5/6 (r8): deep T1 staging of the 36 phase-2 source streams, FFT36_PF2D
 * lines (= tiles) ahead of the always-on 1-line T0 in PF36.  PF36's 36 T0
 * prefetches at a tile boundary contend for the ~12 L1 fill buffers, so
 * L3-resident lines (out's early planes, evicted by the phase-1 in-read at
 * B=1) arrive marginally late; a T1 issue 4 tiles (~1000 port-0 cycles)
 * ahead moves L3->L2 through the L2 superqueue instead, and the T0 then only
 * has L2->L1 to do.  Shape from L64_radix8's slabpf (node-selected there). */
#define PFT1D(s_) do {                                                       \
    _Pragma("GCC unroll 36")                                                 \
    for (int n_ = 0; n_ < 36; ++n_)                                          \
        __builtin_prefetch((s_) + (size_t)n_ * (2 * LSQ) + 8 * FFT36_PF2D,   \
                           0, 2);                                            \
} while (0)

static inline __attribute__((always_inline))
void FN(phase2_yplane)(const double *mid, double *out, int nt,
                       const double *pnext, int y, int pfw, int pfd)
{
#if PW == 2
    const size_t tiles_ = nt ? (size_t)(NVR / 2) : (size_t)NVR;
#else
    const size_t tiles_ = (size_t)NVR;
#endif
    const double *pn_ = pnext ? pnext + (size_t)y * tiles_ * 8 * FFT36_PFN : NULL;
    if (!nt) {
        for (int zb = 0; zb < NVR; ++zb) {
            const size_t o = ((size_t)y * L + (size_t)zb * PW) * 2;
            const double *s_ = mid + o;
            double       *d_ = out + o;
            PF36(s_);
            if (pfd) PFT1D(s_);
            if (pfw) PFW36(d_);
            PFNX();
#define ST3D(k, v) (*(vec *)(d_ + (size_t)(k) * (2 * LSQ)) = (v))
            PFA36(LD3, ST3D);
#undef ST3D
        }
    } else {
#if PW == 2
        /* pair two z-blocks so every NT write completes a 64-byte line */
        for (int zb = 0; zb < NVR; zb += 2) {
            const size_t o = ((size_t)y * L + (size_t)zb * PW) * 2;
            vec Wa[36], Wb[36];
            TILE(mid + o,     Wa);
            TILE(mid + o + 4, Wb);
            PFNX();
            double *dst = out + o;
            _Pragma("GCC unroll 36")
            for (int k = 0; k < 36; ++k) {
                STREAM_ST(dst + (size_t)k * (2 * LSQ),     Wa[k]);
                STREAM_ST(dst + (size_t)k * (2 * LSQ) + 4, Wb[k]);
            }
        }
#else
        for (int zb = 0; zb < NVR; ++zb) {
            const size_t o = ((size_t)y * L + (size_t)zb * PW) * 2;
            const double *s_ = mid + o;
            double       *d_ = out + o;
            PF36(s_);
            PFNX();
#define ST3N(k, v) STREAM_ST(d_ + (size_t)(k) * (2 * LSQ), (v))
            PFA36(LD3, ST3N);
#undef ST3N
        }
#endif
    }
}

static void FN(phase2)(const double *mid, double *out, int nt,
                       const double *pnext, int pfw, int pfd)
{
    for (int y = 0; y < L; ++y)
        FN(phase2_yplane)(mid, out, nt, pnext, y, pfw, pfd);
}

/* r9: the pf=0 hot path with every flag a literal constant, so gcc dead-codes
 * all prefetch machinery, the flag tests and the NT codelet copy out of the
 * body the front end streams (L45_pfa r8's compile-time exec-variant
 * specialization).  Identical operations in identical order to the general
 * bodies called with (0,0)/(0,NULL,0,0) -- output is bit-identical. */
static void FN(phase1_pf0)(const double *restrict in, double *restrict mid)
{
    for (int x = 0; x < L; ++x)
        FN(phase1_plane)(in, mid, x, 0, 0, FFT36_PFWD);
}
static void FN(phase2_pf0)(const double *mid, double *out)   /* mid may == out */
{
    for (int y = 0; y < L; ++y)
        FN(phase2_yplane)(mid, out, 0, NULL, y, 0, 0);
}

/* r11 TIMING-ONLY probes: phase 1's two subloops separately, at nv=1, so the
 * description can split p1's excess between the in-read side (z) and the
 * mid-store RFO side (y).  Never installed; wrong answers by construction.
 *
 * phase1_zonly = the yb-subloop of every plane exactly as in phase1_plane
 * (in-read, TRNC pair, PFA36, pl writes); the asm sink keeps the pl stores
 * live -- without it gcc dead-codes the entire body (r9 lesson: any A/B probe
 * must launder what the compiler could otherwise remove). */
static void FN(phase1_zonly)(const double *restrict in)
{
    for (int x = 0; x < L; ++x) {
        const double *px = in + (size_t)x * (2 * LSQ);
        vec pl[L * NVR];
        for (int yb = 0; yb < L; yb += PW) {
            vec Zv[36], Wv[36];
            _Pragma("GCC unroll 18")
            for (int zb = 0; zb < NVR; ++zb) {
                vec r_[PW];
                _Pragma("GCC unroll 4")
                for (int j = 0; j < PW; ++j)
                    r_[j] = *(const vec *)(px + ((size_t)(yb + j) * L + (size_t)zb * PW) * 2);
                TRNC(r_, &Zv[zb * PW]);
            }
#define LDZ(n)    Zv[n]
#define STZ(k, v) (Wv[k] = (v))
            PFA36(LDZ, STZ);
#undef LDZ
#undef STZ
            _Pragma("GCC unroll 18")
            for (int zb = 0; zb < NVR; ++zb) {
                vec r_[PW];
                TRNC(&Wv[zb * PW], r_);
                _Pragma("GCC unroll 4")
                for (int j = 0; j < PW; ++j)
                    pl[(size_t)(yb + j) * NVR + zb] = r_[j];
            }
        }
        __asm__ volatile("" : : "r"(pl) : "memory");
    }
}

/* phase1_yonly = the zb-subloop of every plane exactly as in phase1_plane
 * (PFA36 from a plane buffer, stores to mid[x]), except the plane read comes
 * from a fixed caller-supplied 20.25 KB buffer (L1-resident after plane 0,
 * matching the just-written pl of the real body; must hold normal doubles --
 * the plan's S serves).  The store side, the RFO-exposed one, is identical
 * to the real phase 1. */
static void FN(phase1_yonly)(const double *plsrc, double *restrict mid)
{
    const vec *pl = (const vec *)plsrc;
    for (int x = 0; x < L; ++x) {
        double *mx = mid + (size_t)x * (2 * LSQ);
        for (int zb = 0; zb < NVR; ++zb) {
#define LDY(n)    pl[(size_t)(n) * NVR + zb]
#define STY(k, v) (*(vec *)(mx + ((size_t)(k) * L + (size_t)zb * PW) * 2) = (v))
            PFA36(LDY, STY);
#undef LDY
#undef STY
        }
    }
}

/* M_PIPE: cross-volume ping-pong pipeline (round 4).  While phase 2 drains
 * volume b from scratch S_a to `out` with NT stores, phase 1 of volume b+1
 * fills S_b -- interleaved at plane granularity (both phases have exactly L
 * outer iterations per volume), so the compulsory DRAM read (in[b+1]) and NT
 * write (out[b]) streams run CONCURRENTLY at ~half rate each instead of as
 * alternating full-rate bursts with the other stream idle.  Phase 2 goes
 * first in each unit so its NT stores drain asynchronously under phase 1's
 * compute.  PFNX is off (pnext = NULL): phase 1 of b+1 is itself touching
 * in[b+1] concurrently, which supersedes the pre-coverage trick. */
static void FN(pipe_vols)(const double *restrict in, double *restrict out,
                          double *S0, double *S1, int nvol, int pf)
{
    double *Sa = S0, *Sb = S1;
    for (int x = 0; x < L; ++x)                     /* prologue: volume 0 */
        FN(phase1_plane)(in, Sa, x, pf, 0, FFT36_PFWD);
    for (int b = 0; b < nvol; ++b) {
        const double *nin = (b + 1 < nvol) ? in + (size_t)(b + 1) * VDBL : NULL;
        double *o = out + (size_t)b * VDBL;
        for (int u = 0; u < L; ++u) {
            FN(phase2_yplane)(Sa, o, 1, NULL, u, 0, 0);
            if (nin) FN(phase1_plane)(nin, Sb, u, pf, 0, FFT36_PFWD);
        }
        double *sw = Sa; Sa = Sb; Sb = sw;
    }
}

#undef TILE
#undef LD3
#undef ST3
#undef PF36
#undef PFW36
#undef PFT1D
#undef PFNX
#undef PFIN
#undef PFIN_NTA
#undef PFWMID
#undef PFSTEP
#undef PFA36
#undef DFT9F
#undef TRNC
#undef STREAM_ST
#undef VFMA
#undef VFNMA
#undef SWAP
#undef VPAIR
#undef VSPLAT
#undef NVR
#undef VSH
#undef FN
#undef veci
#undef vec

#endif /* template */
