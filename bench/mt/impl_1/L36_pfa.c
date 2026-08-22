/* Carried over from the SINGLE-THREAD competition, where this file finished as
 * written below. Your job in the multicore phase is to parallelise it across
 * 32 cores without losing its single-core efficiency -- read
 * ../PANEL_BRIEF.md, and read ../../geom/strategies/L36_pfa.md for the full
 * history of how this kernel got here.
 */
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
 * ROUND mt_r1: the 32-core phase.  The serial kernel (codelets, passes,
 *   prefetch machinery) is UNTOUCHED; what is added is a threading layer:
 *   (1) A persistent pinned pthread SPIN POOL built in fft3d_create():
 *       T-1 workers pinned to the CPUs the harness's close/cores OMP mapping
 *       uses (read back from a throwaway omp region via sched_getcpu),
 *       parked on an epoch spin, synchronised by flag-array barriers (each
 *       arriver writes its OWN padded line; participant 0 scans and
 *       publishes one release word).  Adopted from L23_matrixsimd mt_r1 /
 *       L17_winograd mt_r1, whose records price the alternatives: one GOMP
 *       parallel region = 5-8 us per call on wallaby (more than a whole
 *       parallel volume here), central-counter barrier = 1.2-5 us.
 *   (2) Batched path K_VOLS: thread t owns a contiguous volume block and
 *       runs the settled serial run_vols on it with its OWN scratch
 *       (first-touched by the owning thread -> NUMA-local).  Candidates
 *       {inplace pf0, inplace pf2, scratch pf0, scratch+nt pf0} x {pw4,pw2}
 *       x team {32,16}: NT is re-raced because L23's mt_r1 measured the
 *       phase-1 "NT always loses" rule INVERTING at the 32-thread bandwidth
 *       wall (nt 1.51 vs plain 1.89 us/vol); team 16 exists because the
 *       driver first-touches in/out on one socket and the far 16 threads
 *       may pay UPI for nothing -- measured, not guessed.
 *   (3) B=1 path: within-volume decomposition on the pool.
 *       K_FUSED2 = phase 1 as 36 x-plane units, barrier, phase 2 as 324
 *       (y,zb-granule) tile units.  K_FUSED3 splits phase 1's two subloops
 *       into separate passes over a 746 KB plane arena (z pass: 324
 *       4-y-row units; y pass: 324 zb-granule units; then the same 324
 *       x-tiles), trading one more barrier + an L2-level arena round trip
 *       for near-perfect balance (36 units over 32 threads = a 2-wave
 *       span, 56% efficiency; 324 units = 98%).  Pass A and pass B use the
 *       IDENTICAL static unit partition, so each thread reads back the
 *       arena planes it itself wrote (x-major units: u = x*9 + i).
 *       All units write whole 64-byte lines at both PW -- no false sharing
 *       at any partition cut, by construction.
 *       Serial stays a candidate: the 3% hysteresis band must be beaten.
 *   Every candidate is correctness-gated against a serial reference in
 *   create(); parallelisation only reassigns whole units to threads, so the
 *   output is bit-identical to the serial kernel at the same PW.
 */
#ifndef _GNU_SOURCE
# define _GNU_SOURCE            /* sched_getcpu, cpu_set_t, setaffinity */
#endif
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#ifdef _OPENMP
# include <omp.h>
#endif

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

/* mt_r1 execution kinds */
enum { K_SERIAL = 0, K_VOLS = 1, K_FUSED2 = 2, K_FUSED3 = 3 };
static const char *const kind_name[] = {"serial", "vols", "fused2", "fused3"};

struct fft3d_plan {
    int     batch;
    int     kind;                /* K_SERIAL / K_VOLS / K_FUSED2 / K_FUSED3 */
    int     tuse;                /* team size for the pool kinds   */
    int     pw;                  /* 2 or 4                         */
    int     mode;                /* one of M_* (serial/vols paths) */
    int     pf;                  /* software-prefetch the in stream */
    int     pfwd;                /* phase-1 write-intent cursor lead (doubles) */
    double *S;                   /* participant-0 scratch (pool-owned) */
};

const char *fft3d_name(void) { return "L36_pfa"; }

/* Filled by fft3d_create() with the tuner's pick, so the leaderboard shows
 * which variant actually ran (r3 verdict requirement; mechanism borrowed from
 * L36_mixedradix / L6_pfa). */
static char g_desc[320];

const char *fft3d_description(void)
{
    return g_desc[0] ? g_desc
                     : "Good-Thomas PFA 4x9 (genfft n1_9 DAG) on a pinned "
                       "spin pool; {serial, vol-parallel, fused2, fused3} x "
                       "{T, pw, mode, pf} autotuned in create()";
}
int fft3d_supports(int Lq) { return Lq == L; }

static void run_vols(int pw, int mode, int pf, int pfwd, double *S, double *S1,
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
            if (pw == 4) { P1(phase1_pf0_pw4(i, mid)); P2(phase2_pf0_pw4(mid, o)); }
            else
#endif
            { (void)pw; P1(phase1_pf0_pw2(i, mid)); P2(phase2_pf0_pw2(mid, o)); }
        } else {
#ifdef HAVE_PW4
            if (pw == 4) { P1(phase1_pw4(i, mid, pfr, p1w, pfwd)); P2(phase2_pw4(mid, o, nt, nx, p2w, p2d)); }
            else
#endif
            { (void)pw; P1(phase1_pw2(i, mid, pfr, p1w, pfwd)); P2(phase2_pw2(mid, o, nt, nx, p2w, p2d)); }
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

/* ==== mt_r1: persistent pinned spin pool ================================== *
 * One GOMP parallel region costs 5-8 us per dispatch on wallaby (measured by
 * L23_matrixsimd mt_r1) -- comparable to this kernel's whole 32-way-parallel
 * volume -- so the pool is raw pthreads created once in fft3d_create(),
 * pinned to the CPUs the harness's OMP_PROC_BIND=close / OMP_PLACES=cores
 * mapping gives (read back from a throwaway omp region via sched_getcpu),
 * parked on an epoch spin.  Barriers are flag-array (L17_winograd mt_r1:
 * central atomic counter 1.2 us/barrier at T=16, flag array 0.3-0.4 us):
 * each arriver release-stores its OWN padded line, participant 0 scans them
 * and publishes the next phase on a single release word.  Main never
 * dispatches again until every worker (idle ones included) has posted its
 * final flag, so g_job is never written while any worker might read it. */

#define MT_MAXT 32

enum { J_INIT = 1, J_VOLS, J_FUSED2, J_FUSED3, J_QUIT };

struct mt_job {
    int kind, tuse, nph;
    int pw, mode, pf, pfwd, nvol;
    int skip;                     /* diagnostic: bit p set = phase p is a no-op */
    const double *in;
    double *out;
};

static struct {
    int T;                        /* pool size incl. participant 0 (main) */
    int started;
    pthread_t th[MT_MAXT];
    int cpu[MT_MAXT];
    double *S[MT_MAXT];           /* per-thread scratch volume, owner-touched */
    void   *rawS[MT_MAXT];
    double *pla;                  /* fused3 plane arena, 1 volume, partition-touched */
    void   *rawpla;
    struct mt_job job;
    unsigned long epoch;          /* main-side epoch base                  */
    _Atomic unsigned long release __attribute__((aligned(64)));
    struct { _Atomic unsigned long v; char pad[56]; } arr[MT_MAXT]
        __attribute__((aligned(64)));
} g_pool;

static void mt_run_phase(const struct mt_job *j, int p, int t)
{
    const int T = j->tuse;
    if ((j->skip >> p) & 1) return;      /* FFT36_PROBE timing shapes only */
    switch (j->kind) {
    case J_INIT: {
        /* per-thread scratch: allocated AND first-touched by its owner, so
         * it is NUMA-local on the two-socket node (t=0 runs on main) */
        if (!g_pool.S[t]) {
            void *raw = NULL;
            if (posix_memalign(&raw, 4096, VDBL * sizeof(double)) == 0) {
                memset(raw, 0, VDBL * sizeof(double));
                g_pool.rawS[t] = raw;
                g_pool.S[t] = (double *)raw;
            }
        }
        /* fused3 arena: first-touch with the same static u-partition pass A
         * writes it with (unit u owns doubles [288u, 288(u+1)) -- x-major) */
        if (g_pool.pla) {
            size_t u0 = (size_t)324 * t / T, u1 = (size_t)324 * (t + 1) / T;
            memset(g_pool.pla + u0 * 288, 0, (u1 - u0) * 288 * sizeof(double));
        }
        break; }
    case J_VOLS: {
        /* thread t owns a contiguous volume block; zero synchronisation
         * inside the phase.  Scratchless fallback keeps OOM runs correct. */
        int b0 = (int)((long long)j->nvol * t / T);
        int b1 = (int)((long long)j->nvol * (t + 1) / T);
        if (b1 > b0) {
            int mode = j->mode;
            double *S = g_pool.S[t];
            if ((mode == M_SCRATCH || mode == M_SCRATCH_NT) && !S)
                mode = M_INPLACE;
            run_vols(j->pw, mode, j->pf, j->pfwd, S, NULL,
                     j->in  + (size_t)b0 * VDBL,
                     j->out + (size_t)b0 * VDBL, b1 - b0);
        }
        break; }
    case J_FUSED2:
        if (p == 0) {              /* phase 1, in-place: nvol*36 plane units */
            const int nu = j->nvol * L;
            int u0 = (int)((long long)nu * t / T);
            int u1 = (int)((long long)nu * (t + 1) / T);
            for (int u = u0; u < u1; ++u) {
                const int b = u / L, x = u - b * L;
                const double *i = j->in  + (size_t)b * VDBL;
                double       *o = j->out + (size_t)b * VDBL;
                /* pf=2: phase-1 write-intent cursor (prefetchw) against the
                 * cross-execute invalidation storm; pf=3: NT mid stores
                 * (see mt_p1plane_nt), PW=4 only. */
#ifdef HAVE_PW4
                if (j->pw == 4) {
                    if (j->pf == 3) mt_p1plane_nt_pw4(i, o, x);
                    else phase1_plane_pw4(i, o, x, 0, j->pf == 2, j->pfwd);
                } else
#endif
                phase1_plane_pw2(i, o, x, 0, j->pf == 2, j->pfwd);
            }
            if (j->pf == 3) _mm_sfence();   /* NT stores are not ordered by
                                             * the barrier's release store */
        } else {                   /* phase 2, in-place: nvol*324 tile units */
            const int nu = j->nvol * 324;
            int u0 = (int)((long long)nu * t / T);
            int u1 = (int)((long long)nu * (t + 1) / T);
            int u = u0;
            while (u < u1) {
                const int b  = u / 324;
                const int ue = (b + 1) * 324 < u1 ? (b + 1) * 324 : u1;
                double *o = j->out + (size_t)b * VDBL;
#ifdef HAVE_PW4
                if (j->pw == 4) mt_xpass_pw4(o, o, u - b * 324, ue - b * 324);
                else
#endif
                mt_xpass_pw2(o, o, u - b * 324, ue - b * 324);
                u = ue;
            }
        }
        break;
    case J_FUSED3: {               /* nvol == 1 only; identical partition in
                                    * every pass, so pass B reads back the
                                    * arena planes this thread wrote in A */
        int u0 = 324 * t / T, u1 = 324 * (t + 1) / T;
        if (u1 <= u0) break;
#ifdef HAVE_PW4
        if (j->pw == 4) {
            if      (p == 0) mt_zpass_pw4(j->in, g_pool.pla, u0, u1);
            else if (p == 1) mt_ypass_pw4(g_pool.pla, j->out, u0, u1);
            else             mt_xpass_pw4(j->out, j->out, u0, u1);
        } else
#endif
        {
            if      (p == 0) mt_zpass_pw2(j->in, g_pool.pla, u0, u1);
            else if (p == 1) mt_ypass_pw2(g_pool.pla, j->out, u0, u1);
            else             mt_xpass_pw2(j->out, j->out, u0, u1);
        }
        break; }
    default: break;
    }
}

static void *mt_worker(void *arg)
{
    const int t = (int)(intptr_t)arg;
    if (g_pool.cpu[t] >= 0) {
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(g_pool.cpu[t], &cs);
        pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    }
    unsigned long e = 0;          /* last completed epoch */
    for (;;) {
        while (atomic_load_explicit(&g_pool.release, memory_order_acquire) < e + 1)
            _mm_pause();
        struct mt_job j = g_pool.job;
        if (j.kind == J_QUIT) {  /* threads t >= tuse leave; the rest stay */
            atomic_store_explicit(&g_pool.arr[t].v, e + j.nph, memory_order_release);
            if (t >= j.tuse) return NULL;
            e += j.nph;
            continue;
        }
        if (t < j.tuse) {
            for (int p = 0; p < j.nph; ++p) {
                if (p)
                    while (atomic_load_explicit(&g_pool.release,
                                                memory_order_acquire) < e + 1 + p)
                        _mm_pause();
                mt_run_phase(&j, p, t);
                atomic_store_explicit(&g_pool.arr[t].v, e + 1 + p,
                                      memory_order_release);
            }
        } else {
            /* idle this job: post the final flag only (main scans idle
             * workers at the last phase, never in between) */
            atomic_store_explicit(&g_pool.arr[t].v, e + j.nph, memory_order_release);
        }
        e += j.nph;
    }
}

/* dispatch one job and run it to completion; main is participant 0 */
static void mt_pool_run(struct mt_job *j)
{
    if (j->tuse > g_pool.T) j->tuse = g_pool.T;
    if (j->tuse < 1) j->tuse = 1;
    const unsigned long e = g_pool.epoch;
    g_pool.job = *j;
    for (int p = 0; p < j->nph; ++p) {
        atomic_store_explicit(&g_pool.release, e + 1 + p, memory_order_release);
        mt_run_phase(j, p, 0);
        const int lim = (p == j->nph - 1) ? g_pool.T : j->tuse;
        for (int t = 1; t < lim; ++t)
            while (atomic_load_explicit(&g_pool.arr[t].v, memory_order_acquire)
                   < e + 1 + p)
                _mm_pause();
    }
    g_pool.epoch = e + j->nph;
}

static void mt_pool_start(void)
{
    if (g_pool.started) return;
    int T = MT_MAXT;              /* never MORE than the harness gives */
#ifdef _OPENMP
    { int m = omp_get_max_threads(); if (m < T) T = m; }
#else
    { const char *e = getenv("OMP_NUM_THREADS");
      T = e ? atoi(e) : 1;
      if (T < 1) T = 1;
      if (T > MT_MAXT) T = MT_MAXT; }
#endif
    for (int t = 0; t < MT_MAXT; ++t) g_pool.cpu[t] = -1;
#ifdef _OPENMP
    /* discover the close/cores placement so the pool pins to the exact CPUs
     * OMP would have used (L23_matrixsimd mt_r1 mechanism) */
    #pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num();
        if (t < MT_MAXT) g_pool.cpu[t] = sched_getcpu();
    }
#endif
    g_pool.T = T;
    g_pool.epoch = 0;
    atomic_store(&g_pool.release, 0);
    for (int t = 0; t < MT_MAXT; ++t) atomic_store(&g_pool.arr[t].v, 0);
    if (g_pool.cpu[0] >= 0) {     /* pin participant 0 = the driver thread */
        cpu_set_t cs;
        CPU_ZERO(&cs);
        CPU_SET(g_pool.cpu[0], &cs);
        pthread_setaffinity_np(pthread_self(), sizeof cs, &cs);
    }
    for (int t = 1; t < T; ++t)
        if (pthread_create(&g_pool.th[t], NULL, mt_worker,
                           (void *)(intptr_t)t) != 0) {
            g_pool.T = t;         /* run with what we got */
            break;
        }
    g_pool.started = 1;
}

/* retire workers t >= keep: a spinning worker burns its core and drags the
 * all-core clock, so after the tuner installs its pick the pool is shrunk to
 * the picked team (L17_winograd mt_r1 mechanism).  The remaining team's
 * epochs stay in lockstep because J_QUIT is a normal 1-phase job. */
static void mt_pool_shrink(int keep)
{
    if (!g_pool.started || keep >= g_pool.T) return;
    if (keep < 1) keep = 1;
    struct mt_job j;
    memset(&j, 0, sizeof j);
    j.kind = J_QUIT; j.tuse = keep; j.nph = 1;
    int oldT = g_pool.T;
    mt_pool_run(&j);
    for (int t = keep; t < oldT; ++t) pthread_join(g_pool.th[t], NULL);
    g_pool.T = keep;
}

static void mt_pool_stop(void)
{
    if (!g_pool.started) return;
    mt_pool_shrink(1);
    for (int t = 0; t < MT_MAXT; ++t) {
        free(g_pool.rawS[t]);
        g_pool.rawS[t] = NULL;
        g_pool.S[t] = NULL;
    }
    free(g_pool.rawpla);
    g_pool.rawpla = NULL;
    g_pool.pla = NULL;
    g_pool.started = 0;
    g_pool.epoch = 0;
    atomic_store(&g_pool.release, 0);
}

/* run one execution shape (candidate or installed pick) over a buffer set */
static void run_mcand(int kind, int tuse, int pw, int mode, int pf, int pfwd,
                      double *S0, const double *in, double *out, int nvol)
{
    if (kind == K_SERIAL) {
        run_vols(pw, mode, pf, pfwd, S0, NULL, in, out, nvol);
        return;
    }
    struct mt_job j;
    memset(&j, 0, sizeof j);
    j.kind = (kind == K_VOLS) ? J_VOLS : (kind == K_FUSED2) ? J_FUSED2 : J_FUSED3;
    j.nph  = (kind == K_VOLS) ? 1 : (kind == K_FUSED2) ? 2 : 3;
    j.tuse = tuse;
    j.pw = pw; j.mode = mode; j.pf = pf; j.pfwd = pfwd;
    j.nvol = nvol; j.in = in; j.out = out;
    mt_pool_run(&j);
}

struct mcand { int kind, tuse, pw, mode, pf, pfwd; };

/* hysteresis rank: lower = simpler; a candidate must beat a simpler one by
 * >3% to be installed (phase-1 r3 discipline: tuner coin-flips cost 3.9-6.7%
 * elsewhere).  Simpler = serial < vols < fused2 < fused3, then FEWER
 * threads, then inplace < scratch < nt, then lower pf. */
static int mcand_rank(const struct mcand *c)
{
    int mo = (c->mode == M_INPLACE) ? 0 : (c->mode == M_SCRATCH) ? 1 : 2;
    /* pw tie-break prefers pw4, the node's standing L=36 pick in every
     * phase-1 round -- keeps a noisy pw2 win from installing (pick lottery,
     * r10 lesson; observed once on a loaded wallaby in this round). */
    return (((c->kind * 40 + c->tuse) * 4 + mo) * 8 + c->pf) * 2 + (c->pw == 2);
}

fft3d_plan *fft3d_create(int Lq, int batch)
{
    if (Lq != L || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->batch = batch;

    mt_pool_start();

    /* fused3 plane arena (one volume, 746 KB), allocated by main and
     * first-touched inside J_INIT by the pool's own static partition */
    if (!g_pool.pla && posix_memalign(&g_pool.rawpla, 4096,
                                      VDBL * sizeof(double)) == 0)
        g_pool.pla = (double *)g_pool.rawpla;

    {
        struct mt_job j;
        memset(&j, 0, sizeof j);
        j.kind = J_INIT; j.tuse = g_pool.T; j.nph = 1;
        mt_pool_run(&j);
    }
    p->S = g_pool.S[0];

    /* safe default: the node's standing serial pick since panel_r5 */
    p->kind = K_SERIAL; p->tuse = 1; p->mode = M_INPLACE; p->pf = 0;
    p->pfwd = FFT36_PFWD;
#ifdef HAVE_PW4
    p->pw = 4;
#else
    p->pw = 2;
#endif

    const int T = g_pool.T;
    int pws[2], npw = 0;
#ifdef HAVE_PW4
    pws[npw++] = 4;
#endif
    pws[npw++] = 2;

    /* candidate list by regime.  batch==1: serial vs within-volume fused
     * over a team-size axis; batch>=32: volume-parallel with the mode/NT
     * question re-raced at 32 threads and the T=16 one-socket control;
     * in between (unscored): volume-parallel with a fused2 fallback. */
    struct mcand cands[40];
    int nc = 0;
    if (batch == 1) {
        for (int w = 0; w < npw; ++w) {
            cands[nc++] = (struct mcand){K_SERIAL, 1, pws[w], M_INPLACE, 0, FFT36_PFWD};
            cands[nc++] = (struct mcand){K_SERIAL, 1, pws[w], M_INPLACE, 2, FFT36_PFWD};
        }
        static const int ts[4] = {32, 24, 16, 8};
        int prev = -1;
        for (int i = 0; i < 4; ++i) {
            int t = ts[i] <= T ? ts[i] : T;
            if (t < 2 || t == prev) continue;
            prev = t;
            cands[nc++] = (struct mcand){K_FUSED2, t, pws[0], M_INPLACE, 0, FFT36_PFWD};
            if (g_pool.pla)
                cands[nc++] = (struct mcand){K_FUSED3, t, pws[0], M_INPLACE, 0, FFT36_PFWD};
        }
        if (T >= 2) {             /* phase-1 prefetchw / NT-mid variants
                                     against the fused coherence storm */
            cands[nc++] = (struct mcand){K_FUSED2, T, pws[0], M_INPLACE, 2, FFT36_PFWD};
#ifdef HAVE_PW4
            cands[nc++] = (struct mcand){K_FUSED2, T, 4, M_INPLACE, 3, FFT36_PFWD};
            if (T > 24)
                cands[nc++] = (struct mcand){K_FUSED2, 24, 4, M_INPLACE, 3, FFT36_PFWD};
#endif
        }
        if (npw > 1 && T >= 2) {
            cands[nc++] = (struct mcand){K_FUSED2, T, 2, M_INPLACE, 0, FFT36_PFWD};
            if (g_pool.pla)
                cands[nc++] = (struct mcand){K_FUSED3, T, 2, M_INPLACE, 0, FFT36_PFWD};
        }
    } else if (batch < 32) {
        int tb = batch < T ? batch : T;
        for (int w = 0; w < npw; ++w)
            cands[nc++] = (struct mcand){K_VOLS, tb, pws[w], M_INPLACE, 0, FFT36_PFWD};
        if (T >= 2) {
            cands[nc++] = (struct mcand){K_FUSED2, T, pws[0], M_INPLACE, 0, FFT36_PFWD};
            if (T > 16)
                cands[nc++] = (struct mcand){K_FUSED2, 16, pws[0], M_INPLACE, 0, FFT36_PFWD};
        }
        cands[nc++] = (struct mcand){K_SERIAL, 1, pws[0], M_INPLACE, 0, FFT36_PFWD};
    } else {
        static const int ts[2] = {32, 16};
        int prev = -1;
        for (int i = 0; i < 2; ++i) {
            int t = ts[i] <= T ? ts[i] : T;
            if (t < 1 || t == prev) continue;
            prev = t;
            for (int w = 0; w < npw; ++w) {
                cands[nc++] = (struct mcand){K_VOLS, t, pws[w], M_INPLACE,    0, FFT36_PFWD};
                cands[nc++] = (struct mcand){K_VOLS, t, pws[w], M_INPLACE,    2, FFT36_PFWD};
                cands[nc++] = (struct mcand){K_VOLS, t, pws[w], M_SCRATCH_NT, 0, FFT36_PFWD};
                /* nt+pf1: paced T1 read prefetch under NT (r3: +24% on
                 * wallaby DRAM streams); nt+pf4: the r7 NTA-protected-
                 * scratch shape -- on the node S is 3/4 of the 1 MB L2 and
                 * the T1 in-read evicts it; NTA bypasses L2.  0-for-node at
                 * one core, but 32-thread DRAM saturation is a new regime. */
                cands[nc++] = (struct mcand){K_VOLS, t, pws[w], M_SCRATCH_NT, 1, FFT36_PFWD};
                cands[nc++] = (struct mcand){K_VOLS, t, pws[w], M_SCRATCH_NT, 4, FFT36_PFWD};
                cands[nc++] = (struct mcand){K_VOLS, t, pws[w], M_SCRATCH,    0, FFT36_PFWD};
            }
        }
    }

    /* run-time forcing for the monitor's control jobs (no recompile).
     * Filters keep phase-1 semantics: an empty result leaves the list. */
    { const char *e;
      if ((e = getenv("FFT36_MT"))) {
          int num = (e[0] >= '0' && e[0] <= '9'), w = 0;
          for (int c = 0; c < nc; ++c)
              if (num ? cands[c].kind == atoi(e)
                      : !strcmp(kind_name[cands[c].kind], e)) cands[w++] = cands[c];
          if (w) nc = w;
      }
      if ((e = getenv("FFT36_T"))) {
          int v = atoi(e), w = 0;
          for (int c = 0; c < nc; ++c)
              if (cands[c].tuse == v || cands[c].kind == K_SERIAL)
                  cands[w++] = cands[c];
          if (w) nc = w;
      }
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
      } }

    /* tuning arena.  Deliberately SERIAL fill: the driver freads `in` and
     * memsets `out` on its main thread, so on the two-socket node every
     * caller page is first-touched on ONE socket -- the arena must reproduce
     * that placement or the T=32-vs-16 race lies (L23_matrixsimd mt_r1). */
    const int nv = batch == 1 ? 1 : (batch < 128 ? batch : 128);
    void *ri = NULL, *ro = NULL, *rr = NULL;
    if (posix_memalign(&ri, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&ro, 64, (size_t)nv * VDBL * sizeof(double)) ||
        posix_memalign(&rr, 64, (size_t)nv * VDBL * sizeof(double))) {
        free(ri); free(ro);
        snprintf(g_desc, sizeof g_desc,
                 "GT-PFA 4x9 mt; tuner SKIPPED (arena alloc failed): "
                 "kind=%s pw=%d mode=%s pf=%d",
                 kind_name[p->kind], p->pw, mode_name[p->mode], p->pf);
        return p;
    }
    double *tin = ri, *tout = ro, *ref = rr;
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        tin[i] = (double)(long long)(s >> 11) * 0x1p-53;
    }
    memset(tout, 0, (size_t)nv * VDBL * sizeof(double));

    /* serial reference at max width */
    run_vols(pws[0], M_INPLACE, 0, FFT36_PFWD, p->S, NULL, tin, ref, nv);

    int    ok[40];
    double tc[40], est[40];
    for (int c = 0; c < nc; ++c) {
        double t0 = now_s();
        run_mcand(cands[c].kind, cands[c].tuse, cands[c].pw, cands[c].mode,
                  cands[c].pf, cands[c].pfwd, p->S, tin, tout, nv);
        est[c] = now_s() - t0;            /* rough single-exec estimate */
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < (size_t)nv * VDBL; ++i) {
            double d = tout[i] - ref[i];
            num += d * d; den += ref[i] * ref[i];
        }
        ok[c] = (num <= den * 1e-26);     /* rel L2 < 1e-13 vs reference */
        tc[c] = 1e300;
    }
    const int NROUND = (nv == 1) ? 7 : 3;
    for (int round = 0; round < NROUND; ++round)
        for (int c = 0; c < nc; ++c) {
            if (!ok[c]) continue;
            /* per-candidate reps sized to its own cost, so pool dispatch is
             * timed at the real per-execute granularity */
            int R = est[c] < 30e-6 ? 16 : est[c] < 100e-6 ? 6
                  : est[c] < 1e-3  ? 2  : 1;
            /* self-warming (L36_pencilfused r5): each candidate is timed
             * from its OWN steady-state cache, not its predecessor's */
            run_mcand(cands[c].kind, cands[c].tuse, cands[c].pw, cands[c].mode,
                      cands[c].pf, cands[c].pfwd, p->S, tin, tout, nv);
            double t0 = now_s();
            for (int r = 0; r < R; ++r)
                run_mcand(cands[c].kind, cands[c].tuse, cands[c].pw,
                          cands[c].mode, cands[c].pf, cands[c].pfwd,
                          p->S, tin, tout, nv);
            double t = (now_s() - t0) / R;
            if (t < tc[c]) tc[c] = t;
        }
    int best = 0;
    for (int c = 1; c < nc; ++c) if (ok[c] && tc[c] < tc[best]) best = c;
    if (ok[best]) {
        const double band = 1.03;
        int pick = best;
        for (int c = 0; c < nc; ++c)
            if (ok[c] && tc[c] <= tc[best] * band &&
                mcand_rank(&cands[c]) < mcand_rank(&cands[pick])) pick = c;
        p->kind = cands[pick].kind; p->tuse = cands[pick].tuse;
        p->pw = cands[pick].pw;     p->mode = cands[pick].mode;
        p->pf = cands[pick].pf;     p->pfwd = cands[pick].pfwd;
    }

    /* description: the pick plus the tournament's own key A/Bs, so the
     * leaderboard carries the scaling evidence whatever was installed */
    {
        double b_ser = 1e300, b_f2 = 1e300, b_f3 = 1e300;
        double vip32 = 1e300, vip16 = 1e300, vnt32 = 1e300, vnt16 = 1e300;
        int tf2 = 0, tf3 = 0;
        for (int c = 0; c < nc; ++c) {
            if (!ok[c] || cands[c].pw != pws[0]) continue;
            double us = tc[c] * 1e6 / nv;
            if (cands[c].kind == K_SERIAL && cands[c].pf == 0 && us < b_ser)
                b_ser = us;
            if (cands[c].kind == K_FUSED2 && us < b_f2) { b_f2 = us; tf2 = cands[c].tuse; }
            if (cands[c].kind == K_FUSED3 && us < b_f3) { b_f3 = us; tf3 = cands[c].tuse; }
            if (cands[c].kind == K_VOLS && cands[c].tuse >= 32) {
                if (cands[c].mode == M_INPLACE && cands[c].pf == 0 && us < vip32) vip32 = us;
                if (cands[c].mode == M_SCRATCH_NT && us < vnt32) vnt32 = us;
            }
            if (cands[c].kind == K_VOLS && cands[c].tuse == 16) {
                if (cands[c].mode == M_INPLACE && cands[c].pf == 0 && us < vip16) vip16 = us;
                if (cands[c].mode == M_SCRATCH_NT && us < vnt16) vnt16 = us;
            }
        }
        int n = snprintf(g_desc, sizeof g_desc,
                 "GT-PFA 4x9 (n1_9) spin-pool mt; pick: %s T=%d pw=%d %s pf=%d"
                 " (B=%d nv=%d nc=%d Tpool=%d)",
                 kind_name[p->kind], p->tuse, p->pw, mode_name[p->mode], p->pf,
                 batch, nv, nc, T);
        if (batch == 1 && n > 0 && (size_t)n < sizeof g_desc)
            snprintf(g_desc + n, sizeof g_desc - n,
                     "; us ser=%.1f f2t%d=%.1f f3t%d=%.1f",
                     b_ser, tf2, b_f2, tf3, b_f3);
        else if (batch >= 32 && n > 0 && (size_t)n < sizeof g_desc)
            snprintf(g_desc + n, sizeof g_desc - n,
                     "; us/vol ip32=%.2f nt32=%.2f ip16=%.2f nt16=%.2f",
                     vip32, vnt32, vip16, vnt16);
    }

    /* FFT36_PROBE: phase-level timing of the fused shapes (dev diagnostic;
     * skipped phases give wrong answers, so nothing here touches the pick) */
    if (getenv("FFT36_PROBE") && batch == 1 && T >= 2) {
        static const char *const nm[8] = {
            "f2 null (barriers only)", "f2 p1 only", "f2 p2 only", "f2 full",
            "f3 null (barriers only)", "f3 pA only", "f3 pB only", "f3 pC only"};
        static const int kd[8]   = {J_FUSED2, J_FUSED2, J_FUSED2, J_FUSED2,
                                    J_FUSED3, J_FUSED3, J_FUSED3, J_FUSED3};
        static const int sk[8]   = {3, 2, 1, 0, 7, 6, 5, 3};
        for (int q = 0; q < 8; ++q) {
            struct mt_job j;
            memset(&j, 0, sizeof j);
            j.kind = kd[q]; j.nph = (kd[q] == J_FUSED2) ? 2 : 3;
            j.tuse = T; j.pw = pws[0]; j.pfwd = FFT36_PFWD;
            j.nvol = 1; j.in = tin; j.out = tout; j.skip = sk[q];
            double bt = 1e300;
            for (int round = 0; round < 5; ++round) {
                mt_pool_run(&j);
                double t0 = now_s();
                for (int r = 0; r < 32; ++r) mt_pool_run(&j);
                double tt = (now_s() - t0) / 32;
                if (tt < bt) bt = tt;
            }
            fprintf(stderr, "L36_pfa probe: %-26s T=%-2d  %.2f us\n", nm[q], T, bt * 1e6);
        }
    }

#ifdef FFT36_LOUD
    if (1) {
#else
    if (getenv("FFT36_VERBOSE")) {
#endif
        for (int c = 0; c < nc; ++c)
            fprintf(stderr, "L36_pfa tuner: %-6s T=%-2d pw=%d %-10s pf=%d %s  %.2f us/vol\n",
                    kind_name[cands[c].kind], cands[c].tuse, cands[c].pw,
                    mode_name[cands[c].mode], cands[c].pf,
                    ok[c] ? "ok " : "BAD", ok[c] ? tc[c] * 1e6 / nv : 0.0);
        fprintf(stderr, "L36_pfa tuner: chose %s T=%d pw=%d %s pf=%d (nv=%d Tpool=%d)\n",
                kind_name[p->kind], p->tuse, p->pw, mode_name[p->mode], p->pf,
                nv, T);
    }
    /* give back the cores the pick does not use: spinning workers depress
     * the all-core clock (the serial candidate ran 77 us here vs 51.4 us in
     * phase 1 on wallaby for exactly this reason) */
    mt_pool_shrink(p->kind == K_SERIAL ? 1 : p->tuse);

    free(ri); free(ro); free(rr);
    return p;
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    mt_pool_stop();
    free(p);
}

void fft3d_execute(fft3d_plan *plan, const double _Complex *in, double _Complex *out)
{
    run_mcand(plan->kind, plan->tuse, plan->pw, plan->mode, plan->pf,
              plan->pfwd, plan->S, (const double *)in, (double *)out,
              plan->batch);
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
static __attribute__((unused)) void FN(phase1_zonly)(const double *restrict in)
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
static __attribute__((unused)) void FN(phase1_yonly)(const double *plsrc, double *restrict mid)
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

/* ---- mt_r1: ranged unit kernels for the pool's within-volume paths ------- *
 * Unit space: 324 units per volume per pass, u = plane*9 + i.  Each unit
 * covers a 4-y-row granule (z pass) or a 4-z-double-column granule (y and x
 * passes), so the unit count and the store granularity are identical at
 * PW=2 and PW=4 and every unit writes whole 64-byte lines -- no false
 * sharing at any partition cut, by construction.  The z and y passes are
 * the two subloops of phase1_plane with the local L1 plane buffer replaced
 * by a caller-supplied whole-volume arena (pla, x-major so a unit range is
 * a contiguous byte range); the x pass is phase2_yplane's tile loop.  Same
 * codelets, same order per line -> bit-identical to serial at the same PW. */
static void FN(mt_zpass)(const double *restrict in, double *restrict pla,
                         int u0, int u1)
{
    for (int u = u0; u < u1; ++u) {
        const int x = u / 9, uy = u % 9;
        const double *px = in + (size_t)x * (2 * LSQ);
        vec *pl = (vec *)(pla + (size_t)x * (2 * LSQ));
        for (int yb = uy * 4; yb < uy * 4 + 4; yb += PW) {
            vec Zv[36], Wv[36];
            _Pragma("GCC unroll 18")
            for (int zb = 0; zb < NVR; ++zb) {
                vec r_[PW];
                _Pragma("GCC unroll 4")
                for (int j = 0; j < PW; ++j)
                    r_[j] = *(const vec *)(px + ((size_t)(yb + j) * L + (size_t)zb * PW) * 2);
                TRNC(r_, &Zv[zb * PW]);
            }
#define LDZM(n)    Zv[n]
#define STZM(k, v) (Wv[k] = (v))
            PFA36(LDZM, STZM);
#undef LDZM
#undef STZM
            _Pragma("GCC unroll 18")
            for (int zb = 0; zb < NVR; ++zb) {
                vec r_[PW];
                TRNC(&Wv[zb * PW], r_);
                _Pragma("GCC unroll 4")
                for (int j = 0; j < PW; ++j)
                    pl[(size_t)(yb + j) * NVR + zb] = r_[j];
            }
        }
    }
}

static void FN(mt_ypass)(const double *restrict pla, double *restrict mid,
                         int u0, int u1)
{
    for (int u = u0; u < u1; ++u) {
        const int x = u / 9, uz = u % 9;
        const vec *pl = (const vec *)(pla + (size_t)x * (2 * LSQ));
        double *mx = mid + (size_t)x * (2 * LSQ);
        for (int zb = uz * (NVR / 9); zb < (uz + 1) * (NVR / 9); ++zb) {
#define LDYM(n)    pl[(size_t)(n) * NVR + zb]
#define STYM(k, v) (*(vec *)(mx + ((size_t)(k) * L + (size_t)zb * PW) * 2) = (v))
            PFA36(LDYM, STYM);
#undef LDYM
#undef STYM
        }
    }
}

/* phase1_plane with NT mid stores (fused pf=3): the fused regime's cost is
 * cross-core coherence -- phase 1's normal stores RFO lines the previous
 * execute's phase-2 readers hold, and phase 2 then reads them remote-M.
 * NT stores are fire-and-forget (snoop-invalidate, no ownership wait) and
 * leave the lines in LLC/DRAM where phase 2's readers pay a shared hit
 * instead of a dirty cross-core transfer.  PW=4 only in practice: there
 * every PFA36 store is one full 64-byte line, so WC buffers never hold
 * partial lines (at PW=2 the k-scattered 32-byte halves would thrash them).
 * Caller must sfence before signalling the barrier. */
static __attribute__((unused)) void
FN(mt_p1plane_nt)(const double *restrict in, double *restrict mid, int x)
{
    vec pl[L * NVR];
    const double *px = in  + (size_t)x * (2 * LSQ);
    double       *mx = mid + (size_t)x * (2 * LSQ);
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
#define LDZN(n)    Zv[n]
#define STZN(k, v) (Wv[k] = (v))
        PFA36(LDZN, STZN);
#undef LDZN
#undef STZN
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
#define LD2N(n)    pl[(size_t)(n) * NVR + zb]
#define ST2N(k, v) STREAM_ST(mx + ((size_t)(k) * L + (size_t)zb * PW) * 2, (v))
        PFA36(LD2N, ST2N);
#undef LD2N
#undef ST2N
    }
}

/* mid may alias out (in-place x pass): PFA36 reads all 36 before storing */
static void FN(mt_xpass)(const double *mid, double *out, int u0, int u1)
{
    for (int u = u0; u < u1; ++u) {
        const int y = u / 9, uz = u % 9;
        for (int zb = uz * (NVR / 9); zb < (uz + 1) * (NVR / 9); ++zb) {
            const size_t o = ((size_t)y * L + (size_t)zb * PW) * 2;
            const double *s_ = mid + o;
            double       *d_ = out + o;
            PF36(s_);
            /* mt_r1: one more tile of read lead.  In the fused path the 36
             * source lines of a tile are remote-M (just written by other
             * cores' phase 1), ~2x the latency PF36's single-tile lead was
             * sized for; a second cursor two tiles out keeps ~72 transfers
             * in flight per thread instead of ~36. */
            _Pragma("GCC unroll 36")
            for (int n_ = 0; n_ < 36; ++n_)
                __builtin_prefetch(s_ + (size_t)n_ * (2 * LSQ) + 16, 0, 3);
#define STXM(k, v) (*(vec *)(d_ + (size_t)(k) * (2 * LSQ)) = (v))
            PFA36(LD3, STXM);
#undef STXM
        }
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
