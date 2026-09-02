/* gen_powp -- prime-power Cooley-Tukey with general exact twiddles:
 * L = 25 (5x5), 27 (3x9, DFT9 = 3x3), 50 (PFA 2 x 25), 100 (PFA 4 x 25),
 * and from gen_r3 the whole odd-p^k class: 49 (7x7), 81 (9x9), 121 (11x11),
 * 125 (5x25) -- any prime-power draw in round 6's 14..127 range that is not
 * gen_pow2's 2^k is served here.
 *
 * ROUND gen_r14 (the execute()/chain() plumbing-seam round; benchFFT's
 * single-shot fft3d_execute at B=1 is the metric) -- WV SINGLE-SHOT EXECUTE
 * at L=100.  The seam on this entry was exactly one cell: 25/27/50 B=1
 * execute already rides the same interleaved engine as their B=1 chains
 * (node: 34.6/36.5/346 us vs fftw3's best 71.4/138.7/775 -- 2.1/3.8/2.2x,
 * nothing to reroute), but at 100 the chain has run the within-volume
 * engine since r12 (~4.05 ms/step INCLUDING the map) while execute() stayed
 * interleaved (5.23 ms).  fft3d_execute now routes through the wv engine
 * whenever the banked chain verdict is wv (p->exec_wv rides p->use_wv, so
 * determinism rides the existing wisdom pinning), with the single-shot's
 * unamortized pack/unpack FUSED AWAY rather than paid: pack slab s -> sweep
 * it L2-hot (entry); the x-pass pencil exits DIRECTLY to the driver's
 * interleaved out from the Pn scratch, 2 shuffles + 2 aligned 64 B stores
 * per slot with the lanex permutation baked into the shuffle indices
 * (gen_layout gen_r14's fuse-the-exit prescription) -- xcol_store's trans8
 * scatter and the whole 16 MB unpack pass are deleted.  Gated at create()
 * vs refnd (first + last volume) BEFORE the wv verdict can be stored (a
 * failure kills the candidate, so warm hits stay trustworthy).  Modes:
 * GENPWP_WVEXEC=1 fused (default) / 2 fused+NT out stores / 3 unfused
 * control; GENPWP_NOWVEXEC keeps the interleaved execute (A/B arm).  Chain
 * paths and all other sizes are byte-untouched.
 *
 * ROUND gen_r13 (quick-fix round; my cells' duty is PROTECTION) -- POOL
 * HYGIENE at the scored 50/100 cells plus the standing tag bump.  The r12
 * board shipped L=50 at 480.6 us where ~415 was available (r11 board 413.9;
 * gen_pfa_large's same-suite 413.4 proves the window was normal): wisdom
 * forensics (wisdom_a80n0.json, gen_powp/chain9/L50/B4 -> l50-ipm0, tie=1,
 * margin -1.9%) show the scoring window's cold race put ipm0 trial-best,
 * "decided" its challenger playoff against rank-0 in the same biased
 * window, and the r10 authoritative override installed it -- a family that
 * has never won 50 or 100 on any host in twelve rounds shipping a scored
 * cell.  The spread gate cannot catch sustained bias (r9 boundary, third
 * observation); the deterministic fix is to stop offering never-winners on
 * scored cells: the DEFAULT pools at 50/100 now carry only families with a
 * recorded win at that size on some host (50: ipp0/ipp1/ip0/ip1/ipf;
 * 100: ipk1/ipp1/ipp0/ip0/ip1/wv).  GENPWP_FULLPOOL=1 restores the r12
 * pools (cross-arch forensics; sig-separated, verdicts never shared);
 * wisdom tag chain9 -> chain10 so the poisoned verdict can never replay.
 * Cold create() also shortens (fewer arms: 13->5 at 50, 14->6 at 100 --
 * the also-ran trim queued since r10).  Kernel arithmetic is untouched at
 * every size: all r7-r12 gate digits must reproduce exactly.
 *
 * ROUND gen_r12 (all hands on L=100, round 2) -- WITHIN-VOLUME SoA CHAIN
 * CANDIDATE at L=100 ("l100-wv"), ADOPTED from gen_batchlane gen_r11 close
 * to verbatim (their engine took the r11 board's 100 cell at 4059 us vs
 * this entry's 4465 -- the cumulative round working as intended).  Eight
 * zmm lanes carry eight X-PLANES of one volume: z- and y-pencils are
 * elementwise batch-lane code (zero shuffles / twiddle swaps / spills where
 * the interleaved shell pays 569 shuffles + ~458 spill slots per z-line, my
 * own r8 census), both sweeps run back to back per ~1.3 MB slab (two-axes-
 * per-pass fusion for free), only the x-pass crosses lanes (trans8-
 * bracketed 104-sv scratch pencil per (y, 8z) column) and carries the map
 * fused at the dft100 stage-2 stores; c packs once per chain into
 * x-consumption order (lanex baked into the deinterleave shuffles).
 * Pencil = PFA(4 x 25) with safe placement (naturally in-place stage-2
 * DFT4s, natural output order); DFT25 = 5x5 CT through a 25-sv L1 scratch,
 * 9 compiled-in w25 constants; slot tables gen_batchlane's numpy-verified
 * set, re-gated here (create()-time m=2 chain composition gate).  Raced as
 * a 14th L=100 candidate, ranked last (the soa 3%-hurdle rule); its OWN
 * long-horizon playoff vs the challenger-playoff winner (r6/r10 machinery
 * generalized) prevents short-trial under-reads.  Node (a80n0): quiet
 * windows 4070.0-4074.2 us/xform graded (sd 0.08-0.13%; the r11 board was
 * 4465.2), race margin +6.9-9.5% over ipp0/ipp1, 3/3 busy-window held-lease
 * pairs vs forced ipp1; per-512-step counters vs ipp1 in one window:
 * l1d.replacement 1.65M vs 2.01M lines/step (-18% -- the L1-round-trip
 * diet is the mechanism), p2_3 -24%, dtlb walk-active ~0 on both (the wv
 * arena is gl_map_huge from birth, so the r11 TLB verdict comes free).
 * Gates: m=2 3.090e-15 (tol 3e-14), graded m=64 4.422e-14 vs anchor
 * 2.416e-14 (tol 1e-10), single/chain bit-repeatable, B=2 multi-volume
 * PASS.  25/27/50/lite paths untouched (historical gate digits reproduce).
 *
 * ROUND gen_r11 (all hands on L=100) -- THP RE-HOME of the chain-hot state
 * and c streams at volumes past L3 scale (ADOPTED from gen_layout gen_r11,
 * their explicit adoption recipe for this entry), plus this engine's numbers
 * for the round's open uop-saturation question.  Counter protocol first
 * (a80n0, /tmp/perf, forced-ipp1 512-chain-step delta so create() drops out
 * of the counts): TOTAL vector dispatch p0+p1+p5+p2_3+p4_9 = 1.04
 * uops/cycle (p0+p5 0.67) -- this engine at 100 is NOWHERE NEAR the ~2.1
 * all-port cap; it is traffic-bound, corroborating the PMU audit's
 * "headroom" reading and gen_pfa_large's r11 1.20 step-average on the
 * shared shell (gen_bluestein's 2.03-at-cap is real but engine-specific).
 * So uop deletion does not pay here; the round's lever is effective
 * bandwidth.  gen_layout r11 measured the mechanism: a80n0 runs
 * THP=madvise, so the driver's posix_memalign buffers get ZERO huge pages
 * and the chain streams state (16 MB) + c (16 MB) through 4K pages every
 * step -- on THIS binary a 512-step dtlb delta reads ~2.5K completed load
 * walks + ~163K walk-active cycles (~0.8% of the step) + ~405K STLB-hit
 * lookups per step.  "r11 THP re-home" is the build marker:
 *
 *   1. At vbytes >= 8 MiB (100 graded; 81/121/125 lite) the plan owns a
 *      gl_map_huge arena (2 MiB pages, prefaulted at create) holding TWO
 *      buffers: STV (state volumes) and CV (c volumes).  fft3d_chain gates
 *      per call on gl_thp_bytes(out) < half (gen_layout's primitive --
 *      verify, don't assume; a THP=always host or a huge-backed caller
 *      buffer disables the re-home for free): the interleaved chain then
 *      runs steps on STV with c staged ONCE per volume into CV (one 16 MB
 *      memcpy vs m=64 strided 4K re-reads), and the LAST write goes to the
 *      driver's out directly (trailing map_span for the deferred families,
 *      last-step dst for the plain ones) -- ZERO extra state copies, the
 *      state buffer only changes address.  Knobs: GENPWP_NOREHOME (kills
 *      arena + re-home, the A/B control arm), GENPWP_NOCV (state-only:
 *      c stays on driver pages).  Values are BIT-IDENTICAL (same
 *      arithmetic, same order, memcpy is exact); all gate drifts reproduce.
 *   2. TRIAL-REGIME FIDELITY (the r10 lesson applied forward): the race's
 *      tout/tcf at re-home sizes are the SAME arena buffers (STV/CV), so
 *      candidates are trialed in exactly the cache+TLB regime the graded
 *      chain now runs -- and trial buffer 4K phases become deterministic
 *      (gl_arena_take's 576 B stagger) instead of malloc luck.  Wisdom tag
 *      chain8 -> chain9: no stale 4K-regime verdict may replay.
 *   3. soa (25/27), the execute path, and the dense fallback are untouched;
 *      below 8 MiB nothing changes at all (50 B=4 stays per-volume
 *      L3-resident with a ~2K-page working set -- STLB-covered already).
 *
 * ROUND gen_r10 (changes on top of r9) -- PLAYOFF-AUTHORITATIVE PICKS: the
 * r9 board recovered 25/27/50 (the r9 banking design worked) but shipped
 * L=100 at 4807.4 us where ~4550 was available (-6.3% vs gen_pfa_large's
 * 4521.7, shared shell).  Cause, read straight out of wisdom_a81n2.json:
 * the scoring window's challenger playoff put ipp1 AHEAD of rank-0 ipk1 by
 * 1.6% (tight verdict, banked "tie") and the 3% rank hysteresis handed the
 * slot back to ipk1 anyway.  The rank prior is a80n0 r6 evidence (5/5
 * pairs); on a81n2 -- same Gold 6326 SKU, the r9/r10 scoring node --
 * held-lease graded pairs measure the OPPOSITE: ipp1 wins 5/6, floors
 * 4538-4619 vs ipk1's 4712-4780 (-2.2..-4.9%).  All kernel arithmetic is
 * BIT-IDENTICAL to r9/r8/r7; only tune() changes ("r10 playoff-authoritative
 * picks" is the build marker):
 *
 *   1. A DECIDED challenger playoff (noise-gated exit reached: Q <= tol or
 *      margin >= 2Q) is now AUTHORITATIVE between its two arms: the loser
 *      cannot take the slot back through the 3% rank hysteresis.  Rank
 *      priors encode ONE host's held-lease history; the playoff is the
 *      running host's own long-horizon measurement and outranks it.  Third
 *      candidates, undecided (still-noisy) playoffs, and the soa family's
 *      3% hurdle are all unchanged.
 *   2. The challenger playoff runs even when rank 0 IS the trial best (arms
 *      = the two trial LEADERS then): on a81n2, 3/5 cold races put ipk1
 *      first in the short trials and installed it with no long-horizon
 *      check at all.
 *   3. TRIAL/PLAYOFF CACHE-REGIME FIDELITY at volumes past L3 scale: base
 *      trials warm 3 steps (was 1), playoff arms warm 6 (was 2) and time
 *      PS=24 (was 12).  The candidates alternate on the SAME tout/tcf, so
 *      a c-custody arm (ipk1 CLFLUSHOPTs the whole 16 MB c stream) hands
 *      its successor -- at 100 that is always ipp1, table order -- a c-cold
 *      hierarchy that one warm step does not re-converge; the old playoff
 *      charged ipp1 for ipk1's flushes (measured: playoff ipk1 5151 vs ipp1
 *      5363 in the same window where graded held-lease pairs read ipp1
 *      4538-4619 vs ipk1 4712-4780).
 *   4. WISDOM TAG chain7 -> chain8: the r9 window's banked l100-ipk1
 *      hysteresis tie can never replay.
 *
 *   Window boundary, recorded honestly: on a81n2 QUIET windows favor ipp1
 *   by 2-5% (graded pairs, 5/6; the r9 scoring window's own playoff read
 *   ipp1 +1.6%); BUSY windows compress the pair to a tie or a small ipk1
 *   edge (contention armor, the same mechanism as a80n0 r6's 5/5).  The
 *   playoff reads its window honestly; the design outcome is that the
 *   monitor's quiet scoring window now races cold (chain8), decides on
 *   faithful long-horizon evidence, and BANKS its own verdict.
 *
 * ROUND gen_r9 (changes on top of r8) -- BANK THE PICKS (PMU_AUDIT.md
 * avenue 1; the node was queued-busy all round, so this is the pure-logic
 * round the brief said it could be).  The r8 board's L=25 cell shipped
 * 41.0 us where 31.4 was available (-29.8%): the scoring window's own cold
 * race banked l25-ip0 as a -0.97% "tie" from a contended window, and the
 * entry was STILL LIVE in wisdom_a80n0.json at r9 start -- an unchanged
 * candidate set would have warm-hit it and replayed the regression.  All
 * kernel arithmetic is BIT-IDENTICAL to r8/r7; only tune() changes:
 *
 *   1. WISDOM TAG chain6 -> chain7: the poisoned r8 verdict (and any other
 *      short-horizon chain6 verdict) can never replay.
 *   2. ADAPTIVE NOISE-GATED TRIALS: the deciding measurements -- the base
 *      min-of-rounds race for the two leaders, the soa-vs-interleaved
 *      playoff, and the rank-0 challenger playoff -- now run EXTRA rounds
 *      until each arm has demonstrated its floor (spread of its 3 smallest
 *      round times <= GENPWP_NQTOL, default 5%) or the margin dwarfs the
 *      noise (margin >= 2Q), capped (base +6 rounds, playoffs 9 rounds) so
 *      cold create() stays far under the 60 s budget.
 *   3. QUALITY-MARKED VERDICTS: a verdict that met the gate stores as a
 *      plain name -- BANKED, honored until the tag/sig machinery re-keys
 *      it.  A verdict that stayed noisy after the extension stores as
 *      "name~q<pct>@<unixtime>" -- PROVISIONAL, honored only within
 *      GENPWP_NQHORIZON (default 1800 s): long enough to pin the pick
 *      across the driver's two repeatability processes (seconds apart, the
 *      reason an unpinned noisy race was never an option), short enough
 *      that the NEXT scoring window re-races a coin flip instead of
 *      replaying it.  Round-end wisdom-strip protocol RETIRES: banking
 *      tight verdicts is now the design (the brief's avenue 1), and
 *      provisional ones expire by themselves.
 *   Determinism proof (the brief's acceptance test: 5 consecutive create()
 *   cycles pick identically) is in the strategy record.
 *
 * ROUND gen_r8 (changes on top of r7) -- the tools round: shipped chain
 * arithmetic is BIT-IDENTICAL to r7 at every size; the round's output is
 * model+counter attribution plus one model-refuted candidate kept as a
 * cross-arch knob:
 *
 *   1. n1_5 SPLIT-COMPLEX DFT5 (-DGENPWP_N15): the interleaved DFT5M's
 *      KIG factorization ported to the SoA site layout -- 32 FP ops vs the
 *      Winograd split core's 36 per DFT5.  BUILT, MODEL-REFUTED ON ICL,
 *      DEFAULT OFF: the folded output FMAs are destructive with both +-
 *      partners needing the same addend, so gcc emits ~72 extra zmm
 *      reg-movs per 25-pencil and ICL does not eliminate vector movs --
 *      port-0/5 uops go 442 -> 474/pencil (llvm-mca + objdump; node A/B
 *      and PMU numbers in the strategy record).  "FP op count" is the
 *      wrong metric in split complex; PORT UOPS is the right one -- the
 *      Winograd core's +- pairs ride non-destructive vaddpd/vsubpd, which
 *      is why it beats every "fewer-ops" DFT5 rewrite here (same mechanism
 *      class as the r7 LIFT5 wash).  Kept as an SPR race knob: Golden Cove
 *      eliminates vector movs at rename, where n1_5 is 368 vs 442 uops.
 *   2. Attribution with the new r8 tools (llvm-mca; PMU once it went live
 *      mid-round): at 100 the ipk1 step models ~2.1 ms of port work under
 *      a ~4.5 ms measured step -- the cell sits on its ~80 MB/step DRAM
 *      floor and the z-subpass's 569 port-5 shuffles/line (400 = TRNC)
 *      are hidden under misses; p1-attribution closed WITHOUT code (the
 *      answer to gen_pfa_large r7's next-list #1, from the model side).
 *      The SoA engine at 25 models ~75% port-bound -- which is what made
 *      item 1 worth building and what its mov tax then killed.
 *
 * ROUND gen_r7 (changes on top of r6) -- the SoA op diet, both items from
 * the queued literature backlog the rounds-7/8 brief exists to spend:
 *
 *   1. LIFTED DFT5 v-pair in the SoA split-complex core (ADOPTED from
 *      gen_batchlane gen_r7, their DFT5VPAIR verbatim): sin(2pi/5) =
 *      phi * sin(pi/5) EXACTLY, so the Winograd v-pair factors through
 *      u = sa - PHI*sb -- 2 fewer FMA-port ops and 2 fewer live temps per
 *      DFT5 at the same dependency depth; 10 DFT5s per L=25 pencil.
 *      BUILT, MEASURED, DEFAULT OFF: on these 5-wide pencils it is a
 *      -0.3% wash alone and LOSES ~0.3% on top of item 2 (7/8 held-lease
 *      pairs) -- their win does not transfer to this codelet shape.
 *      -DGENPWP_LIFT5 opts in (cross-arch race knob for CLX/SPR).
 *   2. 3-SHEAR LIFTED TWIDDLE ROTATIONS at the SoA twiddle stores
 *      (literature 08 6.3, queued in this record since r1; nobody in the
 *      corpus has run it in performant code -- interleaved engines pay a
 *      shuffle per shear, split-complex batch lanes pay none, which is why
 *      it finally pays HERE): a split-complex twiddle is a pure plane
 *      rotation, so the 4-op store (2 vmul + 2 vfma) becomes 3 FMA-port
 *      ops via u = re + T*im, im' = im - S*u, re' = u + T*im' with
 *      T = S/(1+C); half-turn reduction for C < 0 (factor through
 *      rot(theta-pi), negations folded into FNMSUB opcodes) keeps
 *      |T| <= 1 everywhere.  16 twiddles/pencil at 25, 28 at 27: shipped
 *      pencil FP drops 404 -> 388 (-4.0%) at 25 and 436 -> 408 (-6.4%)
 *      at 27.  Node held-lease verdict (soa forced, cold race): -1.6% at
 *      25 (30.87 vs 31.36 min-of-mins), -2.0% at 27 (43.08 vs 44.38,
 *      4/4 pairs).  Compile-time long-double tables,
 *      generated + fp-verified by build/tryout/gen_powp/gentw3.c;
 *      -DGENPWP_NOTW3 restores r6 (the depth-3 shear chain is a latency-
 *      for-port-pressure trade -- a cross-arch race knob, not a given).
 *      Neither item is bit-identical to r6 (same exact values, different
 *      rounding): all gates re-run, margins unchanged (strategy record).
 *
 * ROUND gen_r6 (changes on top of r5):
 *
 *   1. SOA-vs-INTERLEAVED PLAYOFF in the create() race (OWN fix; the wound
 *      is on the r5 board): the monitor's quiet-window cold race at 27
 *      installed l27-ip0 at a 58.6 us/vol trial (wisdom margin -0.6%,
 *      "tie") where soa's true graded number is ~44.5 -- the interleaved
 *      trials evict the SoA arenas + i-cache between rounds and a 2-step
 *      soa trial re-pays the refill every round, where the graded 200-step
 *      chain pays it once (my r5 busy-core race lesson, now demonstrated
 *      in the SCORING window: 51.44 shipped vs ~44.5 available, -15%).
 *      After the short trials, the soa candidate and the BEST interleaved
 *      candidate are re-timed head-to-head in alternation, 24 steady
 *      steps per arm per round x 3 rounds (amortizing the refill exactly
 *      as the graded chain does); playoff numbers feed back into tc[]
 *      with min() on the interleaved side, so the playoff can only make
 *      the comparison fairer to it, and soa still must clear the 3%
 *      simplest-first hysteresis.  At sizes without soa the same playoff
 *      runs between the trial best and the lowest-rank live candidate
 *      when they differ by < 15% (rank encodes graded-shape held-lease
 *      evidence; a 4-step trial dilutes steady-state benefits like ipk1's
 *      c-flush at 100, which the same cold race read +5.3% while 5/5
 *      graded m=64 pairs put it ahead).  Wisdom tag chain5 -> chain6
 *      (race protocol changed; no stale short-trial verdict may replay).
 *   2. TWO-COLUMN map-fused x-pass at L=27: BUILT, MEASURED, REJECTED as a
 *      default (closes the item queued since r2).  p27m2 processes two
 *      ADJACENT flat columns per pencil (column B's stage loads precede
 *      column A's stores in program order; map divides pair ACROSS columns,
 *      3 vdivpd per 6 sites vs 4).  Quiet-window held-lease pairs at 27
 *      B=16 m=200: two-column 47.12-47.39 vs single-column 45.18-45.57
 *      us/xform (+4.4%, 4/5 pairs) -- the doubled straight-line body costs
 *      more in the front end than the paired loads and saved divide buy.
 *      Kept behind -DGENPWP_XCOL2 (opt-in) for cross-arch races; the
 *      shipped x-pass is the r5 single-column form, bit-identical to r5.
 *   3. ipk1 ranked FIRST at 100 (settles my r5 open rank question): 5/5
 *      held-lease pairs, ipk1 5027-5606 vs ipp1 5317-6577 us, the margin
 *      widening as the window went busy -- flushing clean c lines frees
 *      L3 that contention squeezes.  50 ranks unchanged (bypass loses at
 *      B=4 where the batch's c is the L3 reuse set).
 *
 * ROUND gen_r5 (changes on top of r4):
 *
 *   1. PAIR-PACKED map ladder in map_span (ADOPTED from gen_pfa_large
 *      gen_r5, their map_step_pair): the sequential map paths ran the NR
 *      ladder on vectors where each |z|^2 is duplicated in both complex
 *      lanes -- half the ladder lanes were redundant.  Two vectors' 8
 *      distinct |z|^2 now pack into ONE zmm (2 shuffles), one ladder runs
 *      for both, reciprocals unpack pair-duplicated (2 shuffles): ~21
 *      arithmetic ops + 4 shuffles per PAIR vs 36 + 2.  BIT-IDENTICAL per
 *      element (q_re + q_im commutes; max/rsqrt14/rcp14/fma elementwise),
 *      so every ip* and ipp chain family and the trailing map keep their r4
 *      gate numbers.  Applies to ip* map passes, the ipp prepass, and the
 *      deferred-chain trailing span at ALL sizes.
 *   2. c-stream L3-BYPASS chain families ipq1/ipk1/iqn1 at 50/100 and ipq0
 *      at the lite sizes (ADOPTED from gen_pfa_large gen_r5): c is read
 *      exactly once per step; its only reuse is across steps, so caching it
 *      is pure pollution wherever state+c exceeds the 24 MB L3 (L=100:
 *      32 MB; 121/125: 57/62 MB).  ipq* = ipp* with the prepass fetching c
 *      via PREFETCHNTA (SKX+ NTA fills L1 only, skips L2/LLC on eviction);
 *      ipk1 = CLFLUSHOPT-ing clean c lines after use (architectural
 *      semantics where NTA policy is implementation-defined); iqn1 = ip1
 *      with the NTA-c map pass.  All bit-identical (hints/flushes move no
 *      data); ranked last, the per-host race arbitrates -- at 50 B=4 the
 *      whole batch's c is L3-resident reuse and they should lose there.
 *      pf ids 11/12/13/14; wisdom tag chain4 -> chain5.
 *   3. PAIRED-vdivpd SoA x-pass map at 25/27 (reciprocal-product trick from
 *      gen_layout gen_r5's gl_map16, adapted to the split-complex site):
 *      the final-stage map shares ONE vdivpd between two sites via
 *      q = 1/(d0 d1), r0 = q d1, r1 = q d0 -- x-pass divider occupancy
 *      -40% at 25 (15 vs 25 vdivpd/pencil), -33% at 27, for 3 extra vmulpd
 *      per pair (~1-2 ulp on the reciprocal, budget is ~60 ulp/step).
 *      Knob -DGENPWP_NOMAPPAIR restores the r2-r4 one-div-per-site form.
 *
 * ROUND gen_r4 (changes on top of r3):
 *
 *   1. VOLUME-MAJOR chain schedule for the interleaved families (ADOPTED
 *      from gen_pfa_large gen_r4, who adopted it from gen_dense_prime /
 *      gen_rader / gen_layout): fft3d_chain runs ALL m steps on one volume
 *      before touching the next, so the per-step working set drops from the
 *      whole batch's state+c to one volume's slice (0.5 MB at 25 -- L2! --
 *      4 MB at 50 B=4).  Per-volume FFT op order unchanged; bit-identical
 *      at 50/100.  The soa engine was group-major from birth (unchanged).
 *   2. ipp* plane-prepass deferred-map chain family (ADOPTED from
 *      gen_pfa_large gen_r4): the ipm schedule, but the map runs as
 *      map_span's sequential per-plane prepass into an L2-resident scratch
 *      plane (M's base) that p1body then consumes -- same traffic accounting
 *      as ipm without the ladder's port/latency footprint inside the
 *      granule-load stream (their measured ipm failure at L=100, confirmed
 *      by my r3 numbers).  Raced as ipp0/ipp1 at 50/100 (ranked between the
 *      base families and ipm), ipp0 at 25/27 and the lite sizes.
 *   3. The create() race times the VOLUME-MAJOR shape (their race fix): per
 *      volume one unmeasured warm step then R timed steps in place on that
 *      volume (R = 8/6/4 by volume bytes <=2/<=8/>8 MiB), min over 4
 *      interleaved rounds.  Wisdom tag chain2 -> chain4 so a stale
 *      step-major verdict can never be installed.
 *
 * ROUND gen_r3 (changes on top of r2):
 *
 *   1. ipm* deferred-map chain family ADOPTED from gen_pfa_large gen_r3
 *      (their file, this morning -- the r2 lesson about re-reading the seed
 *      entry's CURRENT source, applied): the graded map z/(1+|z|) is applied
 *      to the NEXT step's phase-1 z-subpass LOADS (v = map(z' + c) right
 *      after the granule load, before TRNC), so the state buffer holds the
 *      raw FFT output between steps and the separate map pass's full-volume
 *      read + RFO + write disappears (at L=100 that pass is ~32 MB of the
 *      ~112 MB DRAM-bound step).  fft3d_chain runs their schedule: step 1 =
 *      plain execute, steps 2..m = p1m+p2ip, one trailing map_span.  Raced
 *      as ipm0/ipm1 at 50/100 and ipm0 at the new lite sizes; bit-identical
 *      to the ip* families' op order.  Their two race fixes come along
 *      (both bit them first): a DISTINCT c-field buffer tcf (tin-as-c made
 *      ipm's state+c loads share lines and halved its apparent traffic),
 *      and trials run IN PLACE on tout (cfn(tout,tout,tcf) -- the graded
 *      steady state) instead of tin->tout.
 *   2. Round-3 any-size-in-class duty: the engine template is instantiated
 *      at 49/81/121/125 with new line codelets -- DFT49C = 7x7 CT (DFT7K
 *      conjugate-pair fold module, 33 FMA-port ops), DFT81C = 9x9 CT (DFT9K
 *      = 3x3 CT reusing the compile-time W9 tables), DFT121C = 11x11 CT
 *      (DFT11K fold, 75 ops), DFT125C = 5x25 CT (stage A 25 x DFT5 + 96
 *      W125 twiddles, stage B 5 x DFT25M).  Twiddle/constant tables for the
 *      new sizes are computed ONCE at create() in long double (cosl/sinl,
 *      ~19 digits -- gen_pfa_small r3's precedent for satisfying the
 *      brief's exactness rule with runtime tables), stored as {cos} +
 *      {sin,-sin} pairs so a twiddle store is VSPLAT + one 128-bit
 *      broadcast (no per-use sign flip).  The lite sizes instantiate a
 *      REDUCED template (GENLITE: ip0/ipf/f0/ipm0 candidates only) to keep
 *      compile size sane; all four pass the same create()-time refnd gate
 *      and race as the tuned sizes.  Unscored; they exist for round 6.
 *
 * ROUND gen_r2 (changes on top of the r1 design described below):
 *
 *   1. SoA-8 BATCH-LANE chain engine for L = 25/27 at batch % 8 == 0 (the
 *      graded 25/27 cases are B=16), raced as a 7th candidate "soa":
 *      8 volumes fill the zmm lanes, split-complex 128 B sites (re[8]|im[8]),
 *      ZERO shuffles inside the transform (structure ADOPTED from
 *      gen_pfa_small gen_r2 / gen_batchlane gen_r1, lineage ice bl8; their
 *      records explicitly invited this entry to take it, twiddles becoming
 *      broadcast-FMA operands).  NEW here -- the twiddle problem inside that
 *      structure: prime-power CT needs a digit permutation somewhere, so the
 *      pencil is IN PLACE via classic DIF (natural in -> digit-reversed out)
 *      on odd steps and DIT (digit-reversed in -> natural out) on even steps;
 *      every stage's read and write slot sets coincide (no whole-pencil
 *      register buffering, no self-sort pass).  The map is fused into the
 *      x-pass stores; c is packed ONCE per chain in BOTH layouts (natural for
 *      DIT steps, digit-reversed for DIF steps).  Slot algebra and twiddle
 *      placement verified against numpy before coding (DIF/DIT, both sizes).
 *      L=25: 2 stages of 5 x DFT5 (Winograd 4-constant split form, from
 *      gen_pfa_small) + 16 W25 twiddles; L=27: 3 stages of 9 x DFT3 + 28
 *      W27/W9 twiddles; twiddle store = 4 broadcast FMAs, exponents fold at
 *      compile time through the same product-indexed tables as r1.
 *   2. DFT25 stage-B outputs fused straight through the PFA wrappers' ST
 *      (ADOPTED from gen_pfa_large gen_r2, who did it first -- it was queued
 *      in both records): dft25v + R_[25] round-trip replaced by their
 *      DFT25M(LDX, STO, KMAP) macro; helps L=50/100.
 *   3. gen_race ADOPTION: the create()-race result is persisted per host in
 *      results/wisdom_<host>.json via gr_keyf/gr_sig/gr_wisdom_lookup/store.
 *      Warm create() is a file read (50 ms budget); and the pick is PINNED
 *      across processes, which is what makes a non-bit-identical candidate
 *      (soa computes in split-complex order) safe against the driver's
 *      two-process repeatability cmp -- gen_race's documented rationale.
 *   4. gen_layout ADOPTION: THP arena (gl_arena) with staggered mod-4096
 *      phases for the three SoA buffers; gl_pack8/gl_unpack8/gl_tr8x8 for the
 *      lane pack/unpack at the chain ends.
 *
 * ROUND gen_r1 (first real round; the previous file was the dense O(L^4)
 * validation stub).  This class is the campaign's twiddle problem: 25 and 27
 * are prime powers, so Good-Thomas cannot split them and every factorization
 * pays general W_N twiddles.  Technique:
 *
 *   Row-column 3D DFT, two sweeps per volume:
 *     phase 1, per x-plane:
 *       z transform: lanes = 4 y-rows, 4x4 complex-granule register
 *                    transposes on load and store, into plane scratch
 *                    pl[y][kz] (row pitch an ODD number of cache lines);
 *       y transform: lanes = 4 kz (contiguous in pl), store into the plan's
 *                    padded mid volume M (+64 B per plane -> odd line count,
 *                    no fixed mod-4096 relation to the driver's buffers).
 *     phase 2:
 *       x transform, lanes = 4 kz, tiled over the FLAT (y,z) index.  25^2
 *       and 27^2 are 1 mod 4, so the last tile OVERLAPS (recomputes 3
 *       lanes).  Out of place (M -> out) the overlap is trivially
 *       idempotent; IN PLACE the tail's inputs coincide with the previous
 *       tile's outputs, so the tail's GENL input vectors are STASHED before
 *       any tile runs and the tail recomputes from the stash.
 *
 *   The graded chain step (z = FFT3(x) + c; x <- z/(1+|z|)) is raced at
 *   create() across THREE families (winner differs by size and host):
 *     ip*: everything in place + a sequential vectorized map pass; the
 *          state volume is the only volume-sized object touched besides c
 *          (gen_pfa_large's measured winner at L=100, where fusing the map
 *          into the GENL-stream x-pass doubles the miss-stream count);
 *     ipf: in place AND the map fused into phase 2's stores -- no mid
 *          volume, no separate map pass (NEW; targets the cache-resident
 *          cases 25/27 where the extra pass is pure cost);
 *     f*:  map fused into phase 2 routed through the padded mid volume M
 *          (+64 B/plane -> odd line count, no fixed mod-4096 relation to
 *          the driver's buffers).
 *
 *   Line codelets (interleaved complex, lanes = a spectator axis, all index
 *   maps folded at compile time):
 *     L = 25:  DFT25 = 5x5 Cooley-Tukey, W25 twiddles: 16 nontrivial per
 *              line, exact long-double literals.  DFT5 is the FFTW n1_5 FMA
 *              form (16 FMA-port ops).  192 FMA-port ops + 36 swaps / line.
 *     L = 27:  DFT27 = 3x9 Cooley-Tukey (NEW this round): stage A = 9 x DFT3
 *              + 16 nontrivial W27 twiddles into U[9*k1+n2]; stage B = 3 x
 *              DFT9, where DFT9 is itself 3x3 CT with 4 nontrivial W9
 *              twiddles (short live ranges, 15 live vectors max).
 *              218 FMA-port ops + 55 swaps / line.
 *     L = 50 = 25 x 2 (coprime -> Good-Thomas, no inter-stage twiddles):
 *              stage 1: 25 x DFT2, stage 2: 2 x DFT25.  434 ops / line.
 *     L = 100 = 25 x 4: stage 1: 25 x DFT4, stage 2: 4 x DFT25.  968 / line.
 *
 * ATTRIBUTION (this file is deliberately cumulative):
 *   - The ENTIRE engine shell is adopted from gen_pfa_large (this round):
 *     two-sweep plane-fused structure, spectator lanes of 4, TRNC granule
 *     transpose, opaque-base asm barrier, heap plane scratch, padded mid
 *     volume (their measured L=100 4K-alias fix), fused NR map chain
 *     (rsqrt14/rcp14 + 2 Newton steps), create()-time scalar-reference gate
 *     + interleaved min-of-rounds race with simplest-first hysteresis, and
 *     the PFA50C/PFA100C/DFT25/DFT5/DFT4 codelets verbatim.  gen_pfa_large
 *     in turn credits L45_pfa (panel_r11), L45_mixedradix, L23_rader.
 *   - Fused map at the final-axis stores (not a separate pass): the
 *     campaign-wide lesson via gen_pfa_small / gen_batchlane / gen_pfa_large.
 *   - NEW here: the DFT27 = 3x9(3x3) codelet with its exact W27/W9 tables,
 *     the direct DFT25 line codelet for L=25, and the overlapping phase-2
 *     tail that lets the engine run at L^2 % 4 != 0 (odd prime powers).
 *
 * OPERATION COUNT (vector FMA-port ops per volume, lanes of 4):
 *   L=25:  3 *  625 * 192 =   360,000    (plus 2 granule transposes/elt)
 *   L=27:  3 *  729 * 218 =   476,766
 *   L=50:  3 * 1250 * 434 / 2 = 813,750  (+ ~8% phase-1 overlap recompute)
 *   L=100: 3 * 2500 * 968 = 7,260,000
 *
 * ACCURACY: twiddles are compile-time literals from long-double cosl/sinl
 * (~19 correct digits, exact-to-0.5ulp doubles).  create() gates every
 * candidate against an independent scalar O(L^2)-per-line reference at
 * 1e-13 rel L2 (first AND last arena volume) and gates the fused chain step
 * against execute + the driver's scalar map, falling back on any mismatch.
 *
 * Falls back to the dense O(L^4) matrix path if AVX-512 is unavailable.
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "../fft3d_api.h"

#ifndef GEN_POWP_ONCE               /* ============ COMMON, first pass ===== */
#define GEN_POWP_ONCE

/* library layers adopted this round (gen_r2); both are all-static includes */
#define GEN_LAYOUT_LIB_ONLY
#include "gen_layout.c"       /* gl_arena/gl_map_huge, gl_tr8x8, gl_(un)pack8 */
#define GEN_RACE_LIB_ONLY
#include "gen_race.c"         /* gr_keyf/gr_bucket/gr_sig + per-host wisdom   */

#ifdef __AVX512F__
#include <immintrin.h>

/* ---- one vector layer: 4 interleaved complex per zmm ------------------- */
typedef double    vec  __attribute__((vector_size(64)));
typedef double    uvec __attribute__((vector_size(64), aligned(8)));
typedef long long veci __attribute__((vector_size(64)));

#ifdef __clang__
# define VSH(a,b,...) __builtin_shufflevector(a, b, __VA_ARGS__)
#else
# define VSH(a,b,...) __builtin_shuffle(a, b, (veci){__VA_ARGS__})
#endif
#define LDU(p)      ((vec)*(const uvec *)(p))
#define STU(p, v)   (*(uvec *)(p) = (uvec)(v))
#define VSPLAT(a)   ((vec){(a),(a),(a),(a),(a),(a),(a),(a)})
#define VPAIR(a,b)  ((vec){(a),(b),(a),(b),(a),(b),(a),(b)})
#define SWAP(v)     VSH((v),(v), 1,0,3,2,5,4,7,6)
#define VFMA(a,b,c)  ((vec)_mm512_fmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
#define VFNMA(a,b,c) ((vec)_mm512_fnmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
#define VFNMS(a,b,c) ((vec)_mm512_fnmsub_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))

/* 4x4 transpose of 128-bit complex granules (involution) -- L45_pfa's TRNC */
#define TRNC(r, c) do {                                                      \
    vec u0_ = VSH((r)[0], (r)[1], 0,1,8,9,4,5,12,13);                        \
    vec u1_ = VSH((r)[0], (r)[1], 2,3,10,11,6,7,14,15);                      \
    vec u2_ = VSH((r)[2], (r)[3], 0,1,8,9,4,5,12,13);                        \
    vec u3_ = VSH((r)[2], (r)[3], 2,3,10,11,6,7,14,15);                      \
    (c)[0] = VSH(u0_, u2_, 0,1,2,3,8,9,10,11);                               \
    (c)[2] = VSH(u0_, u2_, 4,5,6,7,12,13,14,15);                             \
    (c)[1] = VSH(u1_, u3_, 0,1,2,3,8,9,10,11);                               \
    (c)[3] = VSH(u1_, u3_, 4,5,6,7,12,13,14,15);                             \
} while (0)

/* ---- module constants --------------------------------------------------- */
/* 5-point (FFTW n1_5 FMA form, via L45_pfa / gen_pfa_large) */
#define K59  0.55901699437494742410229341718282   /* sqrt(5)/4               */
#define KIG  0.61803398874989484820458683436564   /* sin(4pi/5)/sin(2pi/5)   */
#define KS5  0.95105651629515357211665325776975   /* sin(2pi/5)              */
/* 3-point */
#define KS3  0.86602540378443864676372317075294   /* sin(2pi/3) = sqrt(3)/2  */

/* W25^j = cos(2 pi j / 25) - i sin(2 pi j / 25), j = n2*k1 for n2,k1 in
 * 1..4 -> j in {1,2,3,4,6,8,9,12,16}.  Literals from long-double cosl/sinl
 * (~19 correct digits).  Indexed by the compile-time product, so every
 * access folds to a constant after unrolling. */
static const double C25T[17] = {
    [1]  =  9.685831611286311195e-01, [2]  =  8.763066800438635873e-01,
    [3]  =  7.289686274214115231e-01, [4]  =  5.358267949789966183e-01,
    [6]  =  6.279051952931337601e-02, [8]  = -4.257792915650726488e-01,
    [9]  = -6.374239897486897102e-01, [12] = -9.921147013144778311e-01,
    [16] = -6.374239897486897102e-01,
};
static const double S25T[17] = {
    [1]  =  2.486898871648547882e-01, [2]  =  4.817536741017152750e-01,
    [3]  =  6.845471059286886738e-01, [4]  =  8.443279255020150785e-01,
    [6]  =  9.980267284282715619e-01, [8]  =  9.048270524660195277e-01,
    [9]  =  7.705132427757892308e-01, [12] =  1.253332335643042452e-01,
    [16] = -7.705132427757892307e-01,
};

/* W27^j, j = n2*k1 for n2 in 1..8, k1 in 1..2 -> j in {1..8,10,12,14,16} */
static const double C27T[17] = {
    [1]  =  9.730448705798238388e-01, [2]  =  8.936326403234122482e-01,
    [3]  =  7.660444431189780352e-01, [4]  =  5.971585917027861649e-01,
    [5]  =  3.960797660391568237e-01, [6]  =  1.736481776669303488e-01,
    [7]  = -5.814482891047582855e-02, [8]  = -2.868032327110902531e-01,
    [10] = -6.862416378687335857e-01, [12] = -9.396926207859083841e-01,
    [14] = -9.932383577419429885e-01, [16] = -8.354878114129364197e-01,
};
static const double S27T[17] = {
    [1]  =  2.306158707424401784e-01, [2]  =  4.487991802004621728e-01,
    [3]  =  6.427876096865393263e-01, [4]  =  8.021231927550437851e-01,
    [5]  =  9.182161068802740148e-01, [6]  =  9.848077530122080594e-01,
    [7]  =  9.983081582712682080e-01, [8]  =  9.579895123154888744e-01,
    [10] =  7.273736415730486960e-01, [12] =  3.420201433256687330e-01,
    [14] = -1.160929141252302297e-01, [16] = -5.495089780708060352e-01,
};

/* W9^j, j = b*q for b,q in 1..2 -> j in {1,2,4} */
static const double C9T[5] = {
    [1] =  7.660444431189780352e-01, [2] =  1.736481776669303488e-01,
    [4] = -9.396926207859083841e-01,
};
static const double S9T[5] = {
    [1] =  6.427876096865393263e-01, [2] =  9.848077530122080594e-01,
    [4] =  3.420201433256687330e-01,
};

/* v * (C - iS): 2 FMA-port ops + 1 swap */
#define CMULC(v, C, S) VFMA(SWAP(v), VPAIR((S), -(S)), (v) * VSPLAT(C))

/* One step of the graded chain map on a vector of 4 complex:
 *     z -> z / (1 + |z|)
 * |z| and the reciprocal via rsqrt14/rcp14 + TWO Newton steps each (final
 * relative error ~1e-16, comfortably inside the 1.5e-14/step contract),
 * instead of vsqrtpd+vdivpd whose zmm throughput is not pipelined.  The
 * max() guard keeps z = 0 exact (rsqrt(0) would make 0 * inf = NaN). */
static void map_scalar(const double *z, const double *c, double *o, size_t npts);

static inline __attribute__((always_inline))
vec map_step_v(vec z)
{
    vec q  = z * z;
    vec ms = q + SWAP(q);                        /* |z|^2 in both lanes */
    ms = (vec)_mm512_max_pd((__m512d)ms, (__m512d)VSPLAT(1e-300));
    vec y = (vec)_mm512_rsqrt14_pd((__m512d)ms);
    vec t_ = ms * y;
    y = (y * VSPLAT(0.5)) * VFNMA(t_, y, VSPLAT(3.0));
    t_ = ms * y;
    y = (y * VSPLAT(0.5)) * VFNMA(t_, y, VSPLAT(3.0));
    vec d = VFMA(ms, y, VSPLAT(1.0));            /* 1 + |z|             */
    vec r = (vec)_mm512_rcp14_pd((__m512d)d);
    r = r * VFNMA(d, r, VSPLAT(2.0));
    r = r * VFNMA(d, r, VSPLAT(2.0));
    return z * r;
}

/* One map step for TWO vectors at once (gen_r5, ADOPTED from gen_pfa_large
 * gen_r5's map_step_pair).  map_step_v runs the NR ladder on a vector where
 * each |z|^2 sits duplicated in both complex lanes -- half its ladder lanes
 * compute nothing new.  Here the 8 distinct |z|^2 of a vector PAIR pack
 * into one zmm (2 shuffles), ONE ladder runs for both vectors, and the
 * reciprocals unpack pair-duplicated (2 shuffles): ~21 arithmetic ops + 4
 * shuffles per pair vs 36 + 2 with two map_step_v calls.  BIT-IDENTICAL to
 * map_step_v per element: q_re + q_im = q_im + q_re exactly (IEEE addition
 * commutes), max/rsqrt14/rcp14/fma are elementwise, and the NR expressions
 * are verbatim -- so mixed use across chain families cannot break
 * repeatability. */
static inline __attribute__((always_inline))
void map_step_pair(vec v0, vec v1, vec *o0, vec *o1)
{
    vec q0 = v0 * v0, q1 = v1 * v1;
    vec ms = VSH(q0, q1, 0,2,4,6,8,10,12,14)
           + VSH(q0, q1, 1,3,5,7,9,11,13,15);        /* 8 distinct |z|^2 */
    ms = (vec)_mm512_max_pd((__m512d)ms, (__m512d)VSPLAT(1e-300));
    vec y = (vec)_mm512_rsqrt14_pd((__m512d)ms);
    vec t_ = ms * y;
    y = (y * VSPLAT(0.5)) * VFNMA(t_, y, VSPLAT(3.0));
    t_ = ms * y;
    y = (y * VSPLAT(0.5)) * VFNMA(t_, y, VSPLAT(3.0));
    vec d = VFMA(ms, y, VSPLAT(1.0));                /* 1 + |z|          */
    vec r = (vec)_mm512_rcp14_pd((__m512d)d);
    r = r * VFNMA(d, r, VSPLAT(2.0));
    r = r * VFNMA(d, r, VSPLAT(2.0));
    *o0 = v0 * VSH(r, r, 0,0,1,1,2,2,3,3);
    *o1 = v1 * VSH(r, r, 4,4,5,5,6,6,7,7);
}

/* sequential vectorized map over one contiguous span: o = (z+c)/(1+|z+c|).
 * 2-3 perfectly sequential streams (in-place z==o legal).  BORROWED from
 * gen_pfa_large's ip* chain family: at sizes where the volume does not stay
 * cache-resident, folding the map into the GENL-stream x-pass doubles the
 * miss-stream count; a separate sequential pass is cheaper.  The 25^3/27^3
 * volumes are not multiples of 8 doubles: exact scalar tail (1-3 complex).
 * gen_r5: pair-packed ladder (map_step_pair); odd vector counts (the L=50
 * plane prepass is 625 vectors) fall to one map_step_v. */
static void map_span(const double *z, const double *c, double *o, size_t nd)
{
    const size_t nv = nd / 8;
    size_t i = 0;
    for (; i + 2 <= nv; i += 2) {
        vec v0 = LDU(z + 8 * i)     + LDU(c + 8 * i);
        vec v1 = LDU(z + 8 * i + 8) + LDU(c + 8 * i + 8);
        vec o0, o1;
        map_step_pair(v0, v1, &o0, &o1);
        STU(o + 8 * i,     o0);
        STU(o + 8 * i + 8, o1);
    }
    if (i < nv) {
        vec v = LDU(z + 8 * i) + LDU(c + 8 * i);
        STU(o + 8 * i, map_step_v(v));
    }
    map_scalar(z + nv * 8, c + nv * 8, o + nv * 8, (nd - nv * 8) / 2);
}

/* map_span with the c stream fetched PREFETCHNTA (gen_r5, the ipq/iqn
 * families, ADOPTED from gen_pfa_large gen_r5): on SKX+ non-inclusive-LLC
 * parts, NTA lines fill L1 only and are dropped on eviction instead of
 * allocated into L2/LLC.  c has NO reuse within a step -- keeping it out of
 * L3 leaves the whole LLC to the state volume, which p2 and the next step's
 * prepass re-read.  Values identical to map_span (a hint moves no data). */
static void map_span_nta(const double *z, const double *c, double *o,
                         size_t nd)
{
    const size_t nv = nd / 8;
    size_t i = 0;
    for (; i + 2 <= nv; i += 2) {
        _mm_prefetch((const char *)(c + 8 * i + 512), _MM_HINT_NTA);
        _mm_prefetch((const char *)(c + 8 * i + 520), _MM_HINT_NTA);
        vec v0 = LDU(z + 8 * i)     + LDU(c + 8 * i);
        vec v1 = LDU(z + 8 * i + 8) + LDU(c + 8 * i + 8);
        vec o0, o1;
        map_step_pair(v0, v1, &o0, &o1);
        STU(o + 8 * i,     o0);
        STU(o + 8 * i + 8, o1);
    }
    if (i < nv) {
        vec v = LDU(z + 8 * i) + LDU(c + 8 * i);
        STU(o + 8 * i, map_step_v(v));
    }
    map_scalar(z + nv * 8, c + nv * 8, o + nv * 8, (nd - nv * 8) / 2);
}

/* map_span with the c lines CLFLUSHOPT-ed one pair after use (gen_r5, the
 * ipk family, ADOPTED from gen_pfa_large gen_r5): the guaranteed-semantics
 * variant of the same L3 bypass -- NTA fill policy is implementation-
 * defined, an explicit flush of a clean line is architectural and writes
 * nothing back.  Safe: nothing writes c, so a flush racing an in-flight
 * load re-reads identical bytes. */
static void map_span_cfl(const double *z, const double *c, double *o,
                         size_t nd)
{
    const size_t nv = nd / 8;
    size_t i = 0;
    for (; i + 2 <= nv; i += 2) {
        vec v0 = LDU(z + 8 * i)     + LDU(c + 8 * i);
        vec v1 = LDU(z + 8 * i + 8) + LDU(c + 8 * i + 8);
        vec o0, o1;
        map_step_pair(v0, v1, &o0, &o1);
        STU(o + 8 * i,     o0);
        STU(o + 8 * i + 8, o1);
#ifdef __CLFLUSHOPT__
        if (i >= 2) {
            _mm_clflushopt((void *)(c + 8 * (i - 2)));
            _mm_clflushopt((void *)(c + 8 * (i - 2) + 8));
        }
#endif
    }
    if (i < nv) {
        vec v = LDU(z + 8 * i) + LDU(c + 8 * i);
        STU(o + 8 * i, map_step_v(v));
    }
    map_scalar(z + nv * 8, c + nv * 8, o + nv * 8, (nd - nv * 8) / 2);
}

/* DFT5, FFTW n1_5 FMA form: 16 FMA-port ops + 2 swaps.  Outputs are
 * lvalues; temps block-scoped so the macro can be used repeatedly. */
#define DFT5M(x0,x1,x2,x3,x4, o0,o1,o2,o3,o4) do {                           \
    vec t1_ = (x1) + (x4), t4_ = (x1) - (x4);                                \
    vec t2_ = (x2) + (x3), t7_ = (x2) - (x3);                                \
    vec te_ = t1_ + t2_,   ta_ = t1_ - t2_;                                  \
    (o0) = (x0) + te_;                                                       \
    vec tm_ = VFNMA(te_, VSPLAT(0.25), (x0));                                \
    vec tp_ = VFMA (ta_, VSPLAT(K59), tm_);                                  \
    vec tq_ = VFNMA(ta_, VSPLAT(K59), tm_);                                  \
    vec tv_ = VFMA (t7_, VSPLAT(KIG), t4_);                                  \
    vec tw_ = VFNMA(t4_, VSPLAT(KIG), t7_);                                  \
    vec sv_ = SWAP(tv_), sw_ = SWAP(tw_);                                    \
    (o1) = VFMA (sv_, VPAIR(KS5, -KS5), tp_);                                \
    (o2) = VFNMA(sw_, VPAIR(KS5, -KS5), tq_);                                \
    (o3) = VFMA (sw_, VPAIR(KS5, -KS5), tq_);                                \
    (o4) = VFNMA(sv_, VPAIR(KS5, -KS5), tp_);                                \
} while (0)

/* DFT4: 8 FMA-port ops + 1 swap */
#define DFT4M(x0,x1,x2,x3, o0,o1,o2,o3) do {                                 \
    vec c0_ = (x0) + (x2), c1_ = (x0) - (x2);                                \
    vec c2_ = (x1) + (x3), c3_ = (x1) - (x3);                                \
    vec cm_ = SWAP(c3_);                                                     \
    (o0) = c0_ + c2_;                                                        \
    (o2) = c0_ - c2_;                                                        \
    (o1) = VFMA (cm_, VPAIR(1.0, -1.0), c1_);                                \
    (o3) = VFNMA(cm_, VPAIR(1.0, -1.0), c1_);                                \
} while (0)

/* DFT3: 6 FMA-port ops + 1 swap.
 *   X1 = (x0 - t/2) - i*KS3*(x1-x2),  X2 = conj-pair form with +i. */
#define DFT3M(x0,x1,x2, o0,o1,o2) do {                                       \
    vec ts_ = (x1) + (x2), ds_ = (x1) - (x2);                                \
    (o0) = (x0) + ts_;                                                       \
    vec tm3_ = VFNMA(ts_, VSPLAT(0.5), (x0));                                \
    vec sd3_ = SWAP(ds_);                                                    \
    (o1) = VFMA (sd3_, VPAIR(KS3, -KS3), tm3_);                              \
    (o2) = VFNMA(sd3_, VPAIR(KS3, -KS3), tm3_);                              \
} while (0)

/* DFT25 = 5x5 Cooley-Tukey with exact twiddles, stage-B outputs handed
 * STRAIGHT to the caller's store macro (gen_r2: ADOPTED from gen_pfa_large
 * gen_r2, replacing the dft25v function whose r[25] every PFA wrapper then
 * re-read to route through the CRT map -- a 25-store + 25-load L1 round-trip
 * per call).  LDX(n) yields input n (stride 1 in n), STO(k, v) consumes
 * natural-order output k; KMAP is applied by the caller inside STO.  Stage A
 * stores U_[5*k1 + n2] so stage B reads 5 contiguous hot slots.
 * 192 FMA-port ops + 36 swaps. */
#define DFT25M(LDX, STO, KMAP) do {                                          \
    vec U_[25];                                                              \
    _Pragma("GCC unroll 5")                                                  \
    for (int c5_ = 0; c5_ < 5; ++c5_) {                                      \
        vec y0_, y1_, y2_, y3_, y4_;                                         \
        DFT5M(LDX(c5_), LDX(c5_ + 5), LDX(c5_ + 10), LDX(c5_ + 15),          \
              LDX(c5_ + 20), y0_, y1_, y2_, y3_, y4_);                       \
        U_[c5_]      = y0_;                                                  \
        U_[5 + c5_]  = c5_ ? CMULC(y1_, C25T[c5_],     S25T[c5_])     : y1_; \
        U_[10 + c5_] = c5_ ? CMULC(y2_, C25T[2 * c5_], S25T[2 * c5_]) : y2_; \
        U_[15 + c5_] = c5_ ? CMULC(y3_, C25T[3 * c5_], S25T[3 * c5_]) : y3_; \
        U_[20 + c5_] = c5_ ? CMULC(y4_, C25T[4 * c5_], S25T[4 * c5_]) : y4_; \
    }                                                                        \
    _Pragma("GCC unroll 5")                                                  \
    for (int k1_ = 0; k1_ < 5; ++k1_) {                                      \
        vec r0_, r1_, r2_, r3_, r4_;                                         \
        DFT5M(U_[5 * k1_], U_[5 * k1_ + 1], U_[5 * k1_ + 2],                 \
              U_[5 * k1_ + 3], U_[5 * k1_ + 4],                              \
              r0_, r1_, r2_, r3_, r4_);                                      \
        STO(KMAP(k1_),      r0_);                                            \
        STO(KMAP(k1_ + 5),  r1_);                                            \
        STO(KMAP(k1_ + 10), r2_);                                            \
        STO(KMAP(k1_ + 15), r3_);                                            \
        STO(KMAP(k1_ + 20), r4_);                                            \
    }                                                                        \
} while (0)

/* stage-A input and CRT output-index helpers for the two DFT25M users;
 * expanded inside the PFA wrappers where T_ and k2_ are in scope (their
 * preprocessor note applies: the store macro must reach DFT25M as a
 * PARAMETER, not by textual name) */
#define LDT25(n)   T_[25 * k2_ + (n)]
#define K50MAP(k)  ((26 * (k) + 25 * k2_) % 50)
#define K100MAP(k) ((76 * (k) + 25 * k2_) % 100)

/* ---- per-size line codelets, LD/ST as macro parameters ------------------ */

/* L=25: direct DFT25 = 5x5 CT, stage-B outputs handed straight to ST
 * (no result-array round trip).  X[k1 + 5*k2]. */
#define DFT25C(LD, ST) do {                                                  \
    vec U_[25];                                                              \
    _Pragma("GCC unroll 5")                                                  \
    for (int c_ = 0; c_ < 5; ++c_) {                                         \
        vec y0_, y1_, y2_, y3_, y4_;                                         \
        DFT5M(LD(c_), LD(c_ + 5), LD(c_ + 10), LD(c_ + 15), LD(c_ + 20),     \
              y0_, y1_, y2_, y3_, y4_);                                      \
        U_[c_]      = y0_;                                                   \
        U_[5 + c_]  = c_ ? CMULC(y1_, C25T[c_],     S25T[c_])     : y1_;     \
        U_[10 + c_] = c_ ? CMULC(y2_, C25T[2 * c_], S25T[2 * c_]) : y2_;     \
        U_[15 + c_] = c_ ? CMULC(y3_, C25T[3 * c_], S25T[3 * c_]) : y3_;     \
        U_[20 + c_] = c_ ? CMULC(y4_, C25T[4 * c_], S25T[4 * c_]) : y4_;     \
    }                                                                        \
    _Pragma("GCC unroll 5")                                                  \
    for (int k1_ = 0; k1_ < 5; ++k1_) {                                      \
        vec o0_, o1_, o2_, o3_, o4_;                                         \
        DFT5M(U_[5 * k1_], U_[5 * k1_ + 1], U_[5 * k1_ + 2],                 \
              U_[5 * k1_ + 3], U_[5 * k1_ + 4],                              \
              o0_, o1_, o2_, o3_, o4_);                                      \
        ST(k1_,      o0_); ST(k1_ +  5, o1_); ST(k1_ + 10, o2_);             \
        ST(k1_ + 15, o3_); ST(k1_ + 20, o4_);                                \
    }                                                                        \
} while (0)

/* L=27: DFT27 = 3x9 Cooley-Tukey (NEW).  n = 9*n1 + n2, k = k1 + 3*k2:
 *   X[k1+3k2] = DFT9_{n2}( W27^{n2 k1} * DFT3_{n1}( x[9n1+n2] )[k1] )[k2]
 * Stage A: 9 x DFT3 + 16 nontrivial W27 twiddles into U[9*k1 + n2].
 * Stage B: 3 x DFT9, DFT9 itself 3x3 CT (n2 = 3a+b, k2 = q+3r) with 4
 * nontrivial W9 twiddles.  218 FMA-port ops + 55 swaps. */
#define DFT27C(LD, ST) do {                                                  \
    vec U_[27];                                                              \
    _Pragma("GCC unroll 9")                                                  \
    for (int c_ = 0; c_ < 9; ++c_) {                                         \
        vec y0_, y1_, y2_;                                                   \
        DFT3M(LD(c_), LD(c_ + 9), LD(c_ + 18), y0_, y1_, y2_);               \
        U_[c_]      = y0_;                                                   \
        U_[9 + c_]  = c_ ? CMULC(y1_, C27T[c_],     S27T[c_])     : y1_;     \
        U_[18 + c_] = c_ ? CMULC(y2_, C27T[2 * c_], S27T[2 * c_]) : y2_;     \
    }                                                                        \
    _Pragma("GCC unroll 3")                                                  \
    for (int k1_ = 0; k1_ < 3; ++k1_) {                                      \
        const vec *u_ = U_ + 9 * k1_;                                        \
        vec T9_[9];                                                          \
        _Pragma("GCC unroll 3")                                              \
        for (int b_ = 0; b_ < 3; ++b_) {                                     \
            vec t0_, t1_, t2_;                                               \
            DFT3M(u_[b_], u_[3 + b_], u_[6 + b_], t0_, t1_, t2_);            \
            T9_[b_]     = t0_;                                               \
            T9_[3 + b_] = b_ ? CMULC(t1_, C9T[b_],     S9T[b_])     : t1_;   \
            T9_[6 + b_] = b_ ? CMULC(t2_, C9T[2 * b_], S9T[2 * b_]) : t2_;   \
        }                                                                    \
        _Pragma("GCC unroll 3")                                              \
        for (int q_ = 0; q_ < 3; ++q_) {                                     \
            vec v0_, v1_, v2_;                                               \
            DFT3M(T9_[3 * q_], T9_[3 * q_ + 1], T9_[3 * q_ + 2],             \
                  v0_, v1_, v2_);                                            \
            ST(k1_ + 3 * q_,      v0_);                                      \
            ST(k1_ + 3 * q_ +  9, v1_);                                      \
            ST(k1_ + 3 * q_ + 18, v2_);                                      \
        }                                                                    \
    }                                                                        \
} while (0)

/* L=50 = 25x2 Good-Thomas.  Stage 1: 25 x DFT2 into T_[25*k2 + n1]; stage
 * 2: 2 x DFT25, stage-B outputs stored straight through ST via the CRT map
 * (gen_r2: ADOPTED from gen_pfa_large gen_r2 -- no R_[25] round-trip). */
#define PFA50C(LD, ST) do {                                                  \
    vec T_[50];                                                              \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 25; ++n1_) {                                     \
        vec a_ = LD((2 * n1_     ) % 50);                                    \
        vec b_ = LD((2 * n1_ + 25) % 50);                                    \
        T_[n1_]      = a_ + b_;                                              \
        T_[25 + n1_] = a_ - b_;                                              \
    }                                                                        \
    _Pragma("GCC unroll 2")                                                  \
    for (int k2_ = 0; k2_ < 2; ++k2_)                                        \
        DFT25M(LDT25, ST, K50MAP);                                           \
} while (0)

/* L=100 = 25x4 Good-Thomas.  Stage 1: 25 x DFT4 into T_[25*k2 + n1]; stage
 * 2: 4 x DFT25, stage-B outputs stored straight through ST (gen_r2). */
#define PFA100C(LD, ST) do {                                                 \
    vec T_[100];                                                             \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 25; ++n1_) {                                     \
        vec a0_ = LD((4 * n1_     ) % 100);                                  \
        vec a1_ = LD((4 * n1_ + 25) % 100);                                  \
        vec a2_ = LD((4 * n1_ + 50) % 100);                                  \
        vec a3_ = LD((4 * n1_ + 75) % 100);                                  \
        DFT4M(a0_, a1_, a2_, a3_,                                            \
              T_[n1_], T_[25 + n1_], T_[50 + n1_], T_[75 + n1_]);            \
    }                                                                        \
    _Pragma("GCC unroll 4")                                                  \
    for (int k2_ = 0; k2_ < 4; ++k2_)                                        \
        DFT25M(LDT25, ST, K100MAP);                                          \
} while (0)

/* ==== gen_r3: runtime exact tables + modules for the odd-p^k class ========
 * Tables are filled once at create() from long double cosl/sinl (~19 correct
 * digits, exact-to-0.5ulp doubles -- gen_pfa_small r3's precedent for the
 * brief's exactness rule).  Twiddle tables are stored as C[j] plus the pair
 * {sin, -sin} so W^j = C - iS costs VSPLAT + ONE 128-bit broadcast. */
static double C7F[7],     S7F[7];             /* module constants, DFT7K    */
static double C11F[11],   S11F[11];           /* module constants, DFT11K   */
static double C49F[49],   S49P[49][2];        /* W49 twiddles               */
static double C81F[81],   S81P[81][2];        /* W81 twiddles               */
static double C121F[121], S121P[121][2];      /* W121 twiddles              */
static double C125F[125], S125P[125][2];      /* W125 twiddles              */

static void powp_rt_tabs(void)
{
    static int done;
    if (done) return;
    const long double TP = 6.283185307179586476925286766559L;   /* 2 pi */
    for (int j = 0; j < 7; ++j) {
        C7F[j]  = (double)cosl(TP * j / 7);
        S7F[j]  = (double)sinl(TP * j / 7);
    }
    for (int j = 0; j < 11; ++j) {
        C11F[j] = (double)cosl(TP * j / 11);
        S11F[j] = (double)sinl(TP * j / 11);
    }
#define PWPTAB(N, C, SP) do {                                                \
    for (int j_ = 0; j_ < (N); ++j_) {                                       \
        C[j_] = (double)cosl(TP * j_ / (N));                                 \
        double s_ = (double)sinl(TP * j_ / (N));                             \
        SP[j_][0] = s_; SP[j_][1] = -s_;                                     \
    }                                                                        \
} while (0)
    PWPTAB(49,  C49F,  S49P);
    PWPTAB(81,  C81F,  S81P);
    PWPTAB(121, C121F, S121P);
    PWPTAB(125, C125F, S125P);
#undef PWPTAB
    done = 1;
}

#ifdef __AVX512F__
/* broadcast a 16 B {s,-s} pair to all four complex lanes (AVX512DQ) */
#define VPAIRP(p)  ((vec)_mm512_broadcast_f64x2(_mm_loadu_pd(p)))
/* v * (C - iS) with C/S from runtime tables: 2 FMA-port ops + 1 swap + the
 * two broadcasts (VSPLAT of a scalar is a vbroadcastsd memory operand) */
#define CMULCT(v, CT, SP, j) \
    VFMA(SWAP(v), VPAIRP((SP)[(j)]), (v) * VSPLAT((CT)[(j)]))

#define KID(k) (k)                            /* identity output map        */

/* DFT7, conjugate-pair fold over runtime C7F/S7F: X_k = A_k - i B_k with
 * A_k = x0 + sum_j cos(2pi jk/7) t_j, B_k = sum_j sin(2pi jk/7) u_j,
 * t_j = x_j + x_{7-j}, u_j = x_j - x_{7-j}; X_{7-k} is the conjugate-pair
 * partner.  Full tables indexed by (j k) mod 7 carry the sine signs.
 * 33 FMA-port ops + 3 swaps.  Outputs routed STO(KM(k), simple-var). */
#define DFT7K(LDX, STO, KM) do {                                             \
    vec x70_ = LDX(0);                                                       \
    vec t71_ = LDX(1) + LDX(6), u71_ = LDX(1) - LDX(6);                      \
    vec t72_ = LDX(2) + LDX(5), u72_ = LDX(2) - LDX(5);                      \
    vec t73_ = LDX(3) + LDX(4), u73_ = LDX(3) - LDX(4);                      \
    vec y70_ = x70_ + t71_ + t72_ + t73_;                                    \
    STO(KM(0), y70_);                                                        \
    _Pragma("GCC unroll 3")                                                  \
    for (int k7_ = 1; k7_ <= 3; ++k7_) {                                     \
        vec A7_ = VFMA(t71_, VSPLAT(C7F[k7_]),                               \
                  VFMA(t72_, VSPLAT(C7F[(2 * k7_) % 7]),                     \
                  VFMA(t73_, VSPLAT(C7F[(3 * k7_) % 7]), x70_)));            \
        vec B7_ = VFMA(u71_, VSPLAT(S7F[k7_]),                               \
                  VFMA(u72_, VSPLAT(S7F[(2 * k7_) % 7]),                     \
                       u73_ * VSPLAT(S7F[(3 * k7_) % 7])));                  \
        vec sb7_ = SWAP(B7_);                                                \
        vec o7a_ = VFMA (sb7_, VPAIR(1.0, -1.0), A7_);                       \
        vec o7b_ = VFNMA(sb7_, VPAIR(1.0, -1.0), A7_);                       \
        STO(KM(k7_),     o7a_);                                              \
        STO(KM(7 - k7_), o7b_);                                              \
    }                                                                        \
} while (0)

/* DFT11, same fold shape, h = 5: 75 FMA-port ops + 5 swaps */
#define DFT11K(LDX, STO, KM) do {                                            \
    vec xb0_ = LDX(0);                                                       \
    vec tb1_ = LDX(1) + LDX(10), ub1_ = LDX(1) - LDX(10);                    \
    vec tb2_ = LDX(2) + LDX(9),  ub2_ = LDX(2) - LDX(9);                     \
    vec tb3_ = LDX(3) + LDX(8),  ub3_ = LDX(3) - LDX(8);                     \
    vec tb4_ = LDX(4) + LDX(7),  ub4_ = LDX(4) - LDX(7);                     \
    vec tb5_ = LDX(5) + LDX(6),  ub5_ = LDX(5) - LDX(6);                     \
    vec yb0_ = xb0_ + tb1_ + tb2_ + tb3_ + tb4_ + tb5_;                      \
    STO(KM(0), yb0_);                                                        \
    _Pragma("GCC unroll 5")                                                  \
    for (int kb_ = 1; kb_ <= 5; ++kb_) {                                     \
        vec Ab_ = VFMA(tb1_, VSPLAT(C11F[kb_]),                              \
                  VFMA(tb2_, VSPLAT(C11F[(2 * kb_) % 11]),                   \
                  VFMA(tb3_, VSPLAT(C11F[(3 * kb_) % 11]),                   \
                  VFMA(tb4_, VSPLAT(C11F[(4 * kb_) % 11]),                   \
                  VFMA(tb5_, VSPLAT(C11F[(5 * kb_) % 11]), xb0_)))));        \
        vec Bb_ = VFMA(ub1_, VSPLAT(S11F[kb_]),                              \
                  VFMA(ub2_, VSPLAT(S11F[(2 * kb_) % 11]),                   \
                  VFMA(ub3_, VSPLAT(S11F[(3 * kb_) % 11]),                   \
                  VFMA(ub4_, VSPLAT(S11F[(4 * kb_) % 11]),                   \
                       ub5_ * VSPLAT(S11F[(5 * kb_) % 11])))));              \
        vec sbb_ = SWAP(Bb_);                                                \
        vec oba_ = VFMA (sbb_, VPAIR(1.0, -1.0), Ab_);                       \
        vec obb_ = VFNMA(sbb_, VPAIR(1.0, -1.0), Ab_);                       \
        STO(KM(kb_),      oba_);                                             \
        STO(KM(11 - kb_), obb_);                                             \
    }                                                                        \
} while (0)

/* DFT9 = 3x3 CT (n = 3 n1 + n2, k = k1 + 3 k2), the compile-time W9 tables
 * from the L=27 codelet: stage a DFT3 over n1 per n2 + W9^{n2 k1} into
 * V9_[3 k1 + n2], stage b DFT3 over n2 per k1.  32 FMA-port ops + 8 swaps. */
#define DFT9K(LDX, STO, KM) do {                                             \
    vec V9_[9];                                                              \
    _Pragma("GCC unroll 3")                                                  \
    for (int n9_ = 0; n9_ < 3; ++n9_) {                                      \
        vec y90_, y91_, y92_;                                                \
        DFT3M(LDX(n9_), LDX(n9_ + 3), LDX(n9_ + 6), y90_, y91_, y92_);       \
        V9_[n9_]     = y90_;                                                 \
        V9_[3 + n9_] = n9_ ? CMULC(y91_, C9T[n9_],     S9T[n9_])     : y91_; \
        V9_[6 + n9_] = n9_ ? CMULC(y92_, C9T[2 * n9_], S9T[2 * n9_]) : y92_; \
    }                                                                        \
    _Pragma("GCC unroll 3")                                                  \
    for (int k9_ = 0; k9_ < 3; ++k9_) {                                      \
        vec z90_, z91_, z92_;                                                \
        DFT3M(V9_[3 * k9_], V9_[3 * k9_ + 1], V9_[3 * k9_ + 2],              \
              z90_, z91_, z92_);                                             \
        STO(KM(k9_),     z90_);                                              \
        STO(KM(k9_ + 3), z91_);                                              \
        STO(KM(k9_ + 6), z92_);                                              \
    }                                                                        \
} while (0)

/* stage helpers for the composite codelets (reference locals X_/U_ and the
 * loop vars n2_/k1_/q5_ in scope at the expansion site, like LDT25) */
#define LA49(j)     X_[7 * (j) + n2_]
#define SA49(k, v)  (U_[7 * (k) + n2_] = ((k) && n2_)                        \
                     ? CMULCT((v), C49F, S49P, ((k) * n2_) % 49) : (v))
#define LB49(j)     U_[7 * k1_ + (j)]
#define K49B(k)     (k1_ + 7 * (k))
#define LA81(j)     X_[9 * (j) + n2_]
#define SA81(k, v)  (U_[9 * (k) + n2_] = ((k) && n2_)                        \
                     ? CMULCT((v), C81F, S81P, (k) * n2_) : (v))
#define LB81(j)     U_[9 * k1_ + (j)]
#define K81B(k)     (k1_ + 9 * (k))
#define LA121(j)    X_[11 * (j) + n2_]
#define SA121(k, v) (U_[11 * (k) + n2_] = ((k) && n2_)                       \
                     ? CMULCT((v), C121F, S121P, (k) * n2_) : (v))
#define LB121(j)    U_[11 * k1_ + (j)]
#define K121B(k)    (k1_ + 11 * (k))
#define L125B(n)    T_[25 * q5_ + (n)]     /* T_, NOT U_: DFT25M's internal
                                              U_[25] would shadow it (the
                                              PFA50C/T_ rule)              */
#define K125B(k)    (q5_ + 5 * (k))

/* L=49: 7x7 CT, X[k1 + 7 k2], 36 nontrivial W49 twiddles.  534 FMA-port. */
#define DFT49C(LD, ST) do {                                                  \
    vec X_[49], U_[49];                                                      \
    for (int n_ = 0; n_ < 49; ++n_) X_[n_] = LD(n_);                         \
    _Pragma("GCC unroll 7")                                                  \
    for (int n2_ = 0; n2_ < 7; ++n2_) DFT7K(LA49, SA49, KID);                \
    _Pragma("GCC unroll 7")                                                  \
    for (int k1_ = 0; k1_ < 7; ++k1_) DFT7K(LB49, ST, K49B);                 \
} while (0)

/* L=81: 9x9 CT, 64 nontrivial W81 twiddles.  ~850 FMA-port. */
#define DFT81C(LD, ST) do {                                                  \
    vec X_[81], U_[81];                                                      \
    for (int n_ = 0; n_ < 81; ++n_) X_[n_] = LD(n_);                         \
    _Pragma("GCC unroll 9")                                                  \
    for (int n2_ = 0; n2_ < 9; ++n2_) DFT9K(LA81, SA81, KID);                \
    _Pragma("GCC unroll 9")                                                  \
    for (int k1_ = 0; k1_ < 9; ++k1_) DFT9K(LB81, ST, K81B);                 \
} while (0)

/* L=121: 11x11 CT, 100 nontrivial W121 twiddles.  ~1850 FMA-port. */
#define DFT121C(LD, ST) do {                                                 \
    vec X_[121], U_[121];                                                    \
    for (int n_ = 0; n_ < 121; ++n_) X_[n_] = LD(n_);                        \
    _Pragma("GCC unroll 11")                                                 \
    for (int n2_ = 0; n2_ < 11; ++n2_) DFT11K(LA121, SA121, KID);            \
    _Pragma("GCC unroll 11")                                                 \
    for (int k1_ = 0; k1_ < 11; ++k1_) DFT11K(LB121, ST, K121B);             \
} while (0)

/* L=125: 5x25 CT (n = 25 n1 + n2, k = k1 + 5 k2): stage A 25 x DFT5 + 96
 * W125 twiddles into U_[25 k1 + n2], stage B 5 x DFT25M straight through
 * ST via k1 + 5 k2.  1552 FMA-port. */
#define DFT125C(LD, ST) do {                                                 \
    vec X_[125], T_[125];                                                    \
    for (int n_ = 0; n_ < 125; ++n_) X_[n_] = LD(n_);                        \
    _Pragma("GCC unroll 25")                                                 \
    for (int n2_ = 0; n2_ < 25; ++n2_) {                                     \
        vec y0_, y1_, y2_, y3_, y4_;                                         \
        DFT5M(X_[n2_], X_[25 + n2_], X_[50 + n2_], X_[75 + n2_],             \
              X_[100 + n2_], y0_, y1_, y2_, y3_, y4_);                       \
        T_[n2_]       = y0_;                                                 \
        T_[25 + n2_]  = n2_ ? CMULCT(y1_, C125F, S125P, n2_)     : y1_;      \
        T_[50 + n2_]  = n2_ ? CMULCT(y2_, C125F, S125P, 2 * n2_) : y2_;      \
        T_[75 + n2_]  = n2_ ? CMULCT(y3_, C125F, S125P, 3 * n2_) : y3_;      \
        T_[100 + n2_] = n2_ ? CMULCT(y4_, C125F, S125P, 4 * n2_) : y4_;      \
    }                                                                        \
    _Pragma("GCC unroll 5")                                                  \
    for (int q5_ = 0; q5_ < 5; ++q5_)                                        \
        DFT25M(L125B, ST, K125B);                                            \
} while (0)
#endif /* __AVX512F__ */

#define GCAT_(a,b) a##b
#define GCAT(a,b)  GCAT_(a,b)

/* instantiate the engine for the four sizes */
#define GENL 25
#define GPP  28                       /* pl row pitch: 448 B = 7 lines, odd  */
#define PFAL DFT25C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 27
#define GPP  28                       /* 448 B = 7 lines, odd                */
#define PFAL DFT27C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 50
#define GPP  52                       /* 832 B = 13 lines, odd               */
#define PFAL PFA50C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 100
#define GPP  108                      /* 1728 B = 27 lines, odd              */
#define PFAL PFA100C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

/* gen_r3 odd-p^k class sizes: REDUCED template (GENLITE -> ip0/ipf/f0/ipm0
 * candidates only; halves the compile of these unscored round-6 sizes) */
#define GENLITE 1

#define GENL 49
#define GPP  52                       /* 832 B = 13 lines, odd               */
#define PFAL DFT49C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 81
#define GPP  84                       /* 1344 B = 21 lines, odd              */
#define PFAL DFT81C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 121
#define GPP  124                      /* 1984 B = 31 lines, odd              */
#define PFAL DFT121C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 125
#define GPP  132                      /* 2112 B = 33 lines, odd              */
#define PFAL DFT125C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#undef GENLITE

/* ================= SoA-8 batch-lane chain engine, L = 25 / 27 =============
 * (gen_r2) 8 volumes in the zmm lanes, split-complex 128 B sites
 * (re[8] | im[8]), zero shuffles inside the transform -- the structure of
 * gen_pfa_small gen_r2 / gen_batchlane gen_r1 (ice bl8 lineage), extended to
 * prime powers.  The prime-power novelty: general CT needs a digit
 * permutation, so pencils are the classic IN-PLACE forms -- DIF (natural in,
 * digit-reversed out) and DIT (digit-reversed in, natural out) -- whose
 * stages read and write the SAME slot sets (no whole-pencil buffering).  The
 * chain alternates DIF/DIT steps; c is packed once per chain in both site
 * layouts.  Twiddles are the same exact product-indexed tables as the
 * interleaved engine, applied at the stores as 4 broadcast FMAs (re' =
 * C*r + S*i, im' = C*i - S*r for W^j = C - iS).  Slot algebra and twiddle
 * placement verified against numpy (both sizes, both directions). */

/* site slot k of a pencil at base p, stride st DOUBLES between sites */
#define BR(k) (*(vec *)((p) + (size_t)(k) * (st)))
#define BI(k) (*(vec *)((p) + (size_t)(k) * (st) + 8))

/* ---- gen_r7: 3-shear LIFTED twiddle rotations (literature 08 6.3, queued
 * in this record since r1; the SoA store is where it finally pays -- split-
 * complex twiddles are pure plane rotations, no lane swaps needed).  The
 * 4-op twiddled store (2 vmul + 2 vfma) becomes 3 FMA-port ops:
 *     u = re + T*im;  im' = im - S*u;  re' = u + T*im'      (C >= 0)
 * with T = S/(1+C) = -tan(theta/2) sign-folded.  HALF-TURN REDUCTION keeps
 * |T| <= 1 (tan(theta/2) blows up near theta = pi): for C < 0 the rotation
 * factors through rot(theta - pi) and BOTH output negations fold into
 * FNMSUB opcodes (T = S/(1-C); u = re - T*im; im' = -S*u - im;
 * re' = -T*im' - u) -- every case is exactly 3 FMA-family instructions,
 * all signs inside constants/opcodes.  Tables CT_##_TS / CT_##_HT are
 * compile-time long-double literals generated AND fp-verified by
 * build/tryout/gen_powp/gentw3.c; the _HT branch folds after unrolling
 * (j is a literal).  Rounding differs from the r6 form in the last bits
 * (~2 ulp/twiddle vs ~1, budget 1.5e-14/step): NOT bit-identical to r6,
 * all gates re-run.  -DGENPWP_NOTW3 restores the r6 arithmetic (cross-
 * arch race knob: 3 dependent FMAs trade latency for port pressure). */
static const double __attribute__((unused)) C25T_TS[17] = {
    [1]  = 1.2632937844610817478e-01, [2]  = 2.5675636036772678331e-01,
    [3]  = 3.9592800879772126053e-01, [4]  = 5.4975465219277007428e-01,
    [6]  = 9.3906250581749235257e-01, [8]  = 6.3461929754414810072e-01,
    [9]  = 4.7056428121225149310e-01, [12] = 6.2914667253649757156e-02,
    [16] = -4.7056428121225149302e-01,
};
static const int __attribute__((unused)) C25T_HT[17] = {
    [8] = 1, [9] = 1, [12] = 1, [16] = 1,
};
static const double __attribute__((unused)) C27T_TS[17] = {
    [1]  = 1.1688323675815263715e-01, [2]  = 2.3700435377149607112e-01,
    [3]  = 3.6397023426620236136e-01, [4]  = 5.0221887602274513721e-01,
    [5]  = 6.5771034665545046420e-01, [6]  = 8.3909963117728001184e-01,
    [7]  = 9.4345134143799696028e-01, [8]  = 7.4447241657697499112e-01,
    [10] = 4.3135789393291657054e-01, [12] = 1.7632698070846497342e-01,
    [14] = -5.8243367469983406299e-02, [16] = -2.9938034709574052659e-01,
};
static const int __attribute__((unused)) C27T_HT[17] = {
    [7] = 1, [8] = 1, [10] = 1, [12] = 1, [14] = 1, [16] = 1,
};
static const double __attribute__((unused)) C9T_TS[5] = {
    [1] = 3.6397023426620236136e-01, [2] = 8.3909963117728001184e-01,
    [4] = 1.7632698070846497342e-01,
};
static const int __attribute__((unused)) C9T_HT[5] = { [4] = 1, };

#ifndef GENPWP_NOTW3
#define TWROT3(j, CT_, ST_, rr, ii, ORV, OIV) do {                           \
    if (!CT_##_HT[j]) {                                                      \
        vec u3_ = VFMA (VSPLAT(CT_##_TS[j]), (ii), (rr));                    \
        (OIV)   = VFNMA(VSPLAT(ST_[j]), u3_, (ii));                          \
        (ORV)   = VFMA (VSPLAT(CT_##_TS[j]), (OIV), u3_);                    \
    } else {                                                                 \
        vec u3_ = VFNMA(VSPLAT(CT_##_TS[j]), (ii), (rr));                    \
        (OIV)   = VFNMS(VSPLAT(ST_[j]), u3_, (ii));                          \
        (ORV)   = VFNMS(VSPLAT(CT_##_TS[j]), (OIV), u3_);                    \
    }                                                                        \
} while (0)
#else  /* the r2-r6 4-op form (2 vmul + 2 vfma, r/i independent) */
#define TWROT3(j, CT_, ST_, rr, ii, ORV, OIV) do {                           \
    (ORV) = VFMA (VSPLAT(ST_[j]), (ii), (rr) * VSPLAT(CT_[j]));              \
    (OIV) = VFNMA(VSPLAT(ST_[j]), (rr), (ii) * VSPLAT(CT_[j]));              \
} while (0)
#endif

/* store slot k, twiddled by W^j via CT_'s tables (j == 0 compile-time:
 * plain store) */
#define SSTW(k, j, CT_, ST_, rr, ii) do {                                    \
    vec rt_ = (rr), it_ = (ii);                                              \
    if ((j) != 0) {                                                          \
        vec or_, oi_;                                                        \
        TWROT3(j, CT_, ST_, rt_, it_, or_, oi_);                             \
        BR(k) = or_; BI(k) = oi_;                                            \
    } else { BR(k) = rt_; BI(k) = it_; }                                     \
} while (0)

/* split-lane map step: state (zr,zi) <- map(z + c), c site at cp (re at +0,
 * im at +8, same stride as the pencil).  Same rsqrt14/rcp14 + 2-Newton
 * ladder as the interleaved engine (additive 1e-300 guard as in
 * gen_pfa_small/gen_batchlane -- folds into the |z|^2 FMA chain). */
static inline __attribute__((always_inline))
void map8s(vec *zr, vec *zi, const double *cp)
{
    vec a  = *zr + LDU(cp);
    vec b  = *zi + LDU(cp + 8);
    vec ms = VFMA(a, a, VFMA(b, b, VSPLAT(1e-300)));
    vec y  = (vec)_mm512_rsqrt14_pd((__m512d)ms);
    vec t_ = ms * y;
    y = (y * VSPLAT(0.5)) * VFNMA(t_, y, VSPLAT(3.0));
    t_ = ms * y;
    y = (y * VSPLAT(0.5)) * VFNMA(t_, y, VSPLAT(3.0));
    vec d = VFMA(ms, y, VSPLAT(1.0));
#ifdef GENPWP_MAPRCP
    /* the interleaved engine's rcp14 + 2-Newton reciprocal, kept as the A/B
     * control.  Node verdict (this round, control-first pairs): the single
     * exact vdivpd below wins at 25 (34.2 vs 34.9-35.7, and 32.0 vs 35.6)
     * and ties at 27 -- gen_pfa_small's "divider is idle in this pass"
     * argument transfers to this engine; gen_batchlane's opposite result
     * does not. */
    vec r = (vec)_mm512_rcp14_pd((__m512d)d);
    r = r * VFNMA(d, r, VSPLAT(2.0));
    r = r * VFNMA(d, r, VSPLAT(2.0));
#else
    /* ADOPTED from gen_pfa_small: ONE exact vdivpd on the otherwise-idle
     * divider unit (also exact-tier: better rounding than the ladder) */
    vec r = (vec)_mm512_div_pd((__m512d)VSPLAT(1.0), (__m512d)d);
#endif
    *zr = a * r;
    *zi = b * r;
}

/* TWO split-lane map steps sharing ONE vdivpd (gen_r5): the reciprocal-
 * product trick from gen_layout gen_r5's gl_map16 (built there from
 * gen_dense_prime r4's item 3), adapted to the split-complex site.  Both
 * NR rsqrt ladders run (their lanes are all distinct -- nothing to pack in
 * split complex); only the divide is shared: q = 1/(d0 d1), r0 = q d1,
 * r1 = q d0.  Costs 3 extra vmulpd per pair for one saved vdivpd (~13-16
 * occupied divider cycles on ICL) and adds ~1-2 ulp to the reciprocal --
 * the per-step budget is 1.5e-14 (~60 ulp) and the m=2 soa gate checks the
 * composition at 1e-13.  |z| stays O(1) in the graded chain (the map is a
 * contraction into the unit disc), so d0*d1 cannot overflow. */
static inline __attribute__((always_inline))
void map8s_pair(vec *zr0, vec *zi0, const double *cp0,
                vec *zr1, vec *zi1, const double *cp1)
{
    vec a0 = *zr0 + LDU(cp0),  b0 = *zi0 + LDU(cp0 + 8);
    vec a1 = *zr1 + LDU(cp1),  b1 = *zi1 + LDU(cp1 + 8);
    vec m0 = VFMA(a0, a0, VFMA(b0, b0, VSPLAT(1e-300)));
    vec m1 = VFMA(a1, a1, VFMA(b1, b1, VSPLAT(1e-300)));
    vec y0 = (vec)_mm512_rsqrt14_pd((__m512d)m0);
    vec y1 = (vec)_mm512_rsqrt14_pd((__m512d)m1);
    vec t0 = m0 * y0, t1 = m1 * y1;
    y0 = (y0 * VSPLAT(0.5)) * VFNMA(t0, y0, VSPLAT(3.0));
    y1 = (y1 * VSPLAT(0.5)) * VFNMA(t1, y1, VSPLAT(3.0));
    t0 = m0 * y0; t1 = m1 * y1;
    y0 = (y0 * VSPLAT(0.5)) * VFNMA(t0, y0, VSPLAT(3.0));
    y1 = (y1 * VSPLAT(0.5)) * VFNMA(t1, y1, VSPLAT(3.0));
    vec d0 = VFMA(m0, y0, VSPLAT(1.0));
    vec d1 = VFMA(m1, y1, VSPLAT(1.0));
    vec q  = (vec)_mm512_div_pd((__m512d)VSPLAT(1.0), (__m512d)(d0 * d1));
    vec r0 = q * d1, r1 = q * d0;
    *zr0 = a0 * r0; *zi0 = b0 * r0;
    *zr1 = a1 * r1; *zi1 = b1 * r1;
}

/* map store for the final stage of the x-pass (j/CT_/ST_ ignored: the final
 * stage is twiddle-free); cp/st are in scope in the *_m pencils */
#define SSTM(k, j, CT_, ST_, rr, ii) do {                                    \
    vec zr_ = (rr), zi_ = (ii);                                              \
    map8s(&zr_, &zi_, cp + (size_t)(k) * (st));                              \
    BR(k) = zr_; BI(k) = zi_;                                                \
} while (0)

/* DFT5 split-complex core.  Declares Z0r..Z4i, the five spectral outputs.
 *
 * gen_r8: the FFTW n1_5 FMA factorization ported to split complex is
 * available behind -DGENPWP_N15 (32 FMA-port ops for both components, was
 * 36): B1 = s1*u1 + s2*u2 and B2 = s2*u1 - s1*u2 factor through
 * KIG = s2/s1 = sin(4pi/5)/sin(2pi/5), and the KS5 scale FOLDS INTO the
 * +-i cross FMAs at the outputs (X1 = A1 - iB1 -> Z1r = A1r + KS5*tvi).
 * BUILT, MODEL-REFUTED ON ICL, DEFAULT OFF: every folded output FMA is
 * destructive and its addend (tp/tq) is live in the +- partner, so gcc
 * emits ~72 extra zmm reg-movs per 25-pencil, and ICL does not eliminate
 * vector movs at rename -- port-0/5 uops go 442 -> 474 (llvm-mca + objdump
 * audit; the 36-op Winograd core's +- pairs ride NON-destructive
 * vaddpd/vsubpd, which is why it is uop-optimal here even at a higher
 * "FP op" count).  Kept as a race knob for SPR, whose Golden Cove cores
 * DO eliminate vector movs (there n1_5 is 368 vs 442 port uops, -17%).
 * The default remains the r2-r7 Winograd core, bit-identical to the r7
 * ship; -DGENPWP_LIFT5 applies within it as before. */
#define KS52 0.58778525229247312917   /* sin(4pi/5)                          */
/* gen_r7 LIFTED v-pair (ADOPTED from gen_batchlane gen_r7, their
 * DFT5VPAIR): KS5/KS52 = sin(2pi/5)/sin(pi/5) = 2cos(pi/5) = PHI, the
 * golden ratio, EXACTLY -- so v1 = KS5*sa + KS52*sb, v2 = KS52*sa - KS5*sb
 * factor through u = sa - PHI*sb as v2 = KS52*u, v1 = KS5*u + KL5*sb with
 * KL5 = (KS5^2 + KS52^2)/KS52 = 1.25/sin(pi/5): 2 fewer FMA-port ops and
 * 2 fewer live temps per DFT5, same dependency depth (2).  Constants
 * exact to the last bit of double (their 50-digit Decimal series).
 * DEFAULT OFF here (-DGENPWP_LIFT5 opts in): on THIS engine's 5-wide
 * pencils the lift LOSES ~0.3% on top of the 3-shear twiddles (node,
 * shears-only wins 7/8 held-lease pairs at 25; lift-only vs r6 is a
 * -0.3% wash) -- the serialized v1/v2-through-u chain has too little
 * surrounding ILP to hide, unlike gen_batchlane's wider pencils.  Kept
 * compilable for CLX/SPR races (their r7: weaker FMA throughput should
 * widen the lift's win). */
#define KPHI5 1.61803398874989484820
#define KL5C  2.12662702088009983045
#ifdef GENPWP_N15
/* n1_5 split form (gen_r8, 32 FP ops -- see the header note: on ICL it
 * LOSES by the port-uop count, +72 zmm reg-movs; a race knob for SPR
 * whose Golden Cove cores eliminate vector movs at rename) */
#define D5SC(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i)                        \
    vec tar_ = (x1r) + (x4r), tai_ = (x1i) + (x4i);                          \
    vec tbr_ = (x2r) + (x3r), tbi_ = (x2i) + (x3i);                          \
    vec sar_ = (x1r) - (x4r), sai_ = (x1i) - (x4i);                          \
    vec sbr_ = (x2r) - (x3r), sbi_ = (x2i) - (x3i);                          \
    vec pr_  = tar_ + tbr_,   pi_  = tai_ + tbi_;                            \
    vec qr_  = tar_ - tbr_,   qi_  = tai_ - tbi_;                            \
    vec Z0r  = (x0r) + pr_,   Z0i  = (x0i) + pi_;                            \
    vec fr_  = (x0r) - 0.25 * pr_, fi_ = (x0i) - 0.25 * pi_;                 \
    vec tpr_ = fr_ + K59 * qr_,   tpi_ = fi_ + K59 * qi_;                    \
    vec tqr_ = fr_ - K59 * qr_,   tqi_ = fi_ - K59 * qi_;                    \
    vec tvr_ = sar_ + KIG * sbr_, tvi_ = sai_ + KIG * sbi_;                  \
    vec twr_ = sbr_ - KIG * sar_, twi_ = sbi_ - KIG * sai_;                  \
    vec Z1r = tpr_ + KS5 * tvi_,  Z1i = tpi_ - KS5 * tvr_;                   \
    vec Z2r = tqr_ - KS5 * twi_,  Z2i = tqi_ + KS5 * twr_;                   \
    vec Z3r = tqr_ + KS5 * twi_,  Z3i = tqi_ - KS5 * twr_;                   \
    vec Z4r = tpr_ - KS5 * tvi_,  Z4i = tpi_ + KS5 * tvr_;
#else  /* r2-r7 Winograd 4-constant split core (36 FP ops; 34 with LIFT5).
        * DEFAULT: the +-output pairs ride non-destructive vaddpd/vsubpd,
        * so it is 442 port-0/5 uops/pencil vs n1_5's 474 on ICL. */
#ifdef GENPWP_LIFT5
#define D5VPAIR                                                              \
    vec u5r_ = sar_ - KPHI5 * sbr_,      u5i_ = sai_ - KPHI5 * sbi_;         \
    vec v1r  = KS5 * u5r_ + KL5C * sbr_, v1i  = KS5 * u5i_ + KL5C * sbi_;    \
    vec v2r  = KS52 * u5r_,              v2i  = KS52 * u5i_;
#else
#define D5VPAIR                                                              \
    vec v1r  = KS5 * sar_ + KS52 * sbr_, v1i = KS5 * sai_ + KS52 * sbi_;     \
    vec v2r  = KS52 * sar_ - KS5 * sbr_, v2i = KS52 * sai_ - KS5 * sbi_;
#endif
#define D5SC(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i)                        \
    vec tar_ = (x1r) + (x4r), tai_ = (x1i) + (x4i);                          \
    vec tbr_ = (x2r) + (x3r), tbi_ = (x2i) + (x3i);                          \
    vec sar_ = (x1r) - (x4r), sai_ = (x1i) - (x4i);                          \
    vec sbr_ = (x2r) - (x3r), sbi_ = (x2i) - (x3i);                          \
    vec pr_  = tar_ + tbr_,   pi_  = tai_ + tbi_;                            \
    vec qr_  = tar_ - tbr_,   qi_  = tai_ - tbi_;                            \
    vec X0r  = (x0r) + pr_,   X0i  = (x0i) + pi_;                            \
    vec fr_  = (x0r) - 0.25 * pr_, fi_ = (x0i) - 0.25 * pi_;                 \
    vec A1r  = fr_ + K59 * qr_,  A1i = fi_ + K59 * qi_;                      \
    vec A2r  = fr_ - K59 * qr_,  A2i = fi_ - K59 * qi_;                      \
    D5VPAIR                                                                  \
    vec Z0r = X0r,        Z0i = X0i;                                         \
    vec Z1r = A1r + v1i,  Z1i = A1i - v1r;                                   \
    vec Z2r = A2r + v2i,  Z2i = A2i - v2r;                                   \
    vec Z3r = A2r - v2i,  Z3i = A2i + v2r;                                   \
    vec Z4r = A1r - v1i,  Z4i = A1i + v1r;
#endif /* GENPWP_N15 */

/* one in-place DFT5 stage over slots {BASE + STEP*i}: output k gets twiddle
 * exponent (TJ)*k in the W25 tables through the MST store macro */
#define D5STAGE(BASE, STEP, TJ, MST) do {                                    \
    const int b5_ = (BASE), s5_ = (STEP);                                    \
    vec x0r_ = BR(b5_),           x0i_ = BI(b5_);                            \
    vec x1r_ = BR(b5_ + s5_),     x1i_ = BI(b5_ + s5_);                      \
    vec x2r_ = BR(b5_ + 2 * s5_), x2i_ = BI(b5_ + 2 * s5_);                  \
    vec x3r_ = BR(b5_ + 3 * s5_), x3i_ = BI(b5_ + 3 * s5_);                  \
    vec x4r_ = BR(b5_ + 4 * s5_), x4i_ = BI(b5_ + 4 * s5_);                  \
    D5SC(x0r_, x0i_, x1r_, x1i_, x2r_, x2i_, x3r_, x3i_, x4r_, x4i_)         \
    MST(b5_,           0,        C25T, S25T, Z0r, Z0i);                      \
    MST(b5_ +     s5_, (TJ),     C25T, S25T, Z1r, Z1i);                      \
    MST(b5_ + 2 * s5_, 2 * (TJ), C25T, S25T, Z2r, Z2i);                      \
    MST(b5_ + 3 * s5_, 3 * (TJ), C25T, S25T, Z3r, Z3i);                      \
    MST(b5_ + 4 * s5_, 4 * (TJ), C25T, S25T, Z4r, Z4i);                      \
} while (0)

/* the map-fused FINAL DFT5 stage with paired divides (gen_r5): same loads,
 * same D5SC core, same store slots and per-element map values as
 * D5STAGE(..., SSTM) except outputs 1+2 and 3+4 each share one vdivpd via
 * map8s_pair -- 3 divides per 5 sites instead of 5. */
#define D5STAGEMP(BASE, STEP) do {                                           \
    const int b5_ = (BASE), s5_ = (STEP);                                    \
    vec x0r_ = BR(b5_),           x0i_ = BI(b5_);                            \
    vec x1r_ = BR(b5_ + s5_),     x1i_ = BI(b5_ + s5_);                      \
    vec x2r_ = BR(b5_ + 2 * s5_), x2i_ = BI(b5_ + 2 * s5_);                  \
    vec x3r_ = BR(b5_ + 3 * s5_), x3i_ = BI(b5_ + 3 * s5_);                  \
    vec x4r_ = BR(b5_ + 4 * s5_), x4i_ = BI(b5_ + 4 * s5_);                  \
    D5SC(x0r_, x0i_, x1r_, x1i_, x2r_, x2i_, x3r_, x3i_, x4r_, x4i_)         \
    vec z0r_ = Z0r,       z0i_ = Z0i;                                        \
    map8s(&z0r_, &z0i_, cp + (size_t)b5_ * st);                              \
    BR(b5_) = z0r_; BI(b5_) = z0i_;                                          \
    vec z1r_ = Z1r, z1i_ = Z1i;                                              \
    vec z2r_ = Z2r, z2i_ = Z2i;                                              \
    map8s_pair(&z1r_, &z1i_, cp + (size_t)(b5_ +     s5_) * st,              \
               &z2r_, &z2i_, cp + (size_t)(b5_ + 2 * s5_) * st);             \
    BR(b5_ +     s5_) = z1r_; BI(b5_ +     s5_) = z1i_;                      \
    BR(b5_ + 2 * s5_) = z2r_; BI(b5_ + 2 * s5_) = z2i_;                      \
    vec z3r_ = Z3r, z3i_ = Z3i;                                              \
    vec z4r_ = Z4r, z4i_ = Z4i;                                              \
    map8s_pair(&z3r_, &z3i_, cp + (size_t)(b5_ + 3 * s5_) * st,              \
               &z4r_, &z4i_, cp + (size_t)(b5_ + 4 * s5_) * st);             \
    BR(b5_ + 3 * s5_) = z3r_; BI(b5_ + 3 * s5_) = z3i_;                      \
    BR(b5_ + 4 * s5_) = z4r_; BI(b5_ + 4 * s5_) = z4i_;                      \
} while (0)

/* DFT3 split-complex core (12 FMA-port ops; gen_pfa_small's D3S algebra) */
#define D3SC(x0r,x0i,x1r,x1i,x2r,x2i)                                        \
    vec t3r_ = (x1r) + (x2r), t3i_ = (x1i) + (x2i);                          \
    vec u3r_ = (x1r) - (x2r), u3i_ = (x1i) - (x2i);                          \
    vec h3r_ = (x0r) - 0.5 * t3r_, h3i_ = (x0i) - 0.5 * t3i_;                \
    vec Y0r  = (x0r) + t3r_, Y0i = (x0i) + t3i_;                             \
    vec Y1r  = h3r_ + KS3 * u3i_, Y1i = h3i_ - KS3 * u3r_;                   \
    vec Y2r  = h3r_ - KS3 * u3i_, Y2i = h3i_ + KS3 * u3r_;

/* one in-place DFT3 stage over slots {BASE + STEP*i}: outputs 0/1/2 get
 * twiddle exponents J0/J1/J2 in the CT_/ST_ tables through MST.  J0 is
 * nonzero in exactly one place -- DIT27 stage 2, whose twiddle
 * W27^{n1 (k1 + 3 k2)} carries the constant offset n1*k1 at k2 = 0. */
#define D3STAGE(BASE, STEP, J0, J1, J2, CT_, ST_, MST) do {                  \
    const int b3_ = (BASE), s3_ = (STEP);                                    \
    vec x0r_ = BR(b3_),           x0i_ = BI(b3_);                            \
    vec x1r_ = BR(b3_ + s3_),     x1i_ = BI(b3_ + s3_);                      \
    vec x2r_ = BR(b3_ + 2 * s3_), x2i_ = BI(b3_ + 2 * s3_);                  \
    D3SC(x0r_, x0i_, x1r_, x1i_, x2r_, x2i_)                                 \
    MST(b3_,           (J0), CT_, ST_, Y0r, Y0i);                            \
    MST(b3_ +     s3_, (J1), CT_, ST_, Y1r, Y1i);                            \
    MST(b3_ + 2 * s3_, (J2), CT_, ST_, Y2r, Y2i);                            \
} while (0)

/* the map-fused FINAL DFT3 stage with a paired divide (gen_r5): same loads,
 * same D3SC core, same store slots and per-element map values as
 * D3STAGE(..., SSTM) except outputs 1+2 share one vdivpd via map8s_pair --
 * 2 divides per 3 sites instead of 3. */
#define D3STAGEMP(BASE, STEP) do {                                           \
    const int b3_ = (BASE), s3_ = (STEP);                                    \
    vec x0r_ = BR(b3_),           x0i_ = BI(b3_);                            \
    vec x1r_ = BR(b3_ + s3_),     x1i_ = BI(b3_ + s3_);                      \
    vec x2r_ = BR(b3_ + 2 * s3_), x2i_ = BI(b3_ + 2 * s3_);                  \
    D3SC(x0r_, x0i_, x1r_, x1i_, x2r_, x2i_)                                 \
    vec z0r_ = Y0r, z0i_ = Y0i;                                              \
    map8s(&z0r_, &z0i_, cp + (size_t)b3_ * st);                              \
    BR(b3_) = z0r_; BI(b3_) = z0i_;                                          \
    vec z1r_ = Y1r, z1i_ = Y1i;                                              \
    vec z2r_ = Y2r, z2i_ = Y2i;                                              \
    map8s_pair(&z1r_, &z1i_, cp + (size_t)(b3_ +     s3_) * st,              \
               &z2r_, &z2i_, cp + (size_t)(b3_ + 2 * s3_) * st);             \
    BR(b3_ +     s3_) = z1r_; BI(b3_ +     s3_) = z1i_;                      \
    BR(b3_ + 2 * s3_) = z2r_; BI(b3_ + 2 * s3_) = z2i_;                      \
} while (0)

/* ---- gen_r6: TWO-COLUMN machinery for the L=27 x pass --------------------
 * (see the header's round note).  Column A is the pencil at p (BR/BI),
 * column B the adjacent flat column at p + 16 doubles (one 128 B site).
 * Loads of column B sit BEFORE the stores of column A in program order and
 * st is a runtime stride, so the compiler must keep them there: both
 * columns' site loads are issued per stage visit before any store. */
#define BR_B(k) (*(vec *)((p) + (size_t)(k) * (st) + 16))
#define BI_B(k) (*(vec *)((p) + (size_t)(k) * (st) + 24))

/* twiddled store through explicit slot macros (rr/ii must be simple vars);
 * gen_r7: routed through the same 3-shear TWROT3 as SSTW */
#define TWST2(BRX, BIX, k, j, CT_, ST_, rr, ii) do {                         \
    if ((j) != 0) {                                                          \
        vec or2_, oi2_;                                                      \
        TWROT3(j, CT_, ST_, (rr), (ii), or2_, oi2_);                         \
        BRX(k) = or2_; BIX(k) = oi2_;                                        \
    } else { BRX(k) = (rr); BIX(k) = (ii); }                                 \
} while (0)

/* one in-place DFT3 stage over TWO adjacent columns */
#define D3STAGE2(BASE, STEP, J0, J1, J2, CT_, ST_) do {                      \
    const int b3_ = (BASE), s3_ = (STEP);                                    \
    vec a0r_, a0i_, a1r_, a1i_, a2r_, a2i_;                                  \
    vec c0r_, c0i_, c1r_, c1i_, c2r_, c2i_;                                  \
    { D3SC(BR(b3_), BI(b3_), BR(b3_ + s3_), BI(b3_ + s3_),                   \
           BR(b3_ + 2 * s3_), BI(b3_ + 2 * s3_))                             \
      a0r_ = Y0r; a0i_ = Y0i; a1r_ = Y1r; a1i_ = Y1i;                        \
      a2r_ = Y2r; a2i_ = Y2i; }                                              \
    { D3SC(BR_B(b3_), BI_B(b3_), BR_B(b3_ + s3_), BI_B(b3_ + s3_),           \
           BR_B(b3_ + 2 * s3_), BI_B(b3_ + 2 * s3_))                         \
      c0r_ = Y0r; c0i_ = Y0i; c1r_ = Y1r; c1i_ = Y1i;                        \
      c2r_ = Y2r; c2i_ = Y2i; }                                              \
    TWST2(BR,   BI,   b3_,           (J0), CT_, ST_, a0r_, a0i_);            \
    TWST2(BR_B, BI_B, b3_,           (J0), CT_, ST_, c0r_, c0i_);            \
    TWST2(BR,   BI,   b3_ +     s3_, (J1), CT_, ST_, a1r_, a1i_);            \
    TWST2(BR_B, BI_B, b3_ +     s3_, (J1), CT_, ST_, c1r_, c1i_);            \
    TWST2(BR,   BI,   b3_ + 2 * s3_, (J2), CT_, ST_, a2r_, a2i_);            \
    TWST2(BR_B, BI_B, b3_ + 2 * s3_, (J2), CT_, ST_, c2r_, c2i_);            \
} while (0)

/* the map-fused final DFT3 stage over two columns: divides pair ACROSS the
 * columns (same output slot, adjacent columns) -- 3 vdivpd per 6 sites */
#define D3STAGEMP2(BASE, STEP) do {                                          \
    const int b3_ = (BASE), s3_ = (STEP);                                    \
    vec a0r_, a0i_, a1r_, a1i_, a2r_, a2i_;                                  \
    vec c0r_, c0i_, c1r_, c1i_, c2r_, c2i_;                                  \
    { D3SC(BR(b3_), BI(b3_), BR(b3_ + s3_), BI(b3_ + s3_),                   \
           BR(b3_ + 2 * s3_), BI(b3_ + 2 * s3_))                             \
      a0r_ = Y0r; a0i_ = Y0i; a1r_ = Y1r; a1i_ = Y1i;                        \
      a2r_ = Y2r; a2i_ = Y2i; }                                              \
    { D3SC(BR_B(b3_), BI_B(b3_), BR_B(b3_ + s3_), BI_B(b3_ + s3_),           \
           BR_B(b3_ + 2 * s3_), BI_B(b3_ + 2 * s3_))                         \
      c0r_ = Y0r; c0i_ = Y0i; c1r_ = Y1r; c1i_ = Y1i;                        \
      c2r_ = Y2r; c2i_ = Y2i; }                                              \
    map8s_pair(&a0r_, &a0i_, cp + (size_t)b3_ * st,                          \
               &c0r_, &c0i_, cp + (size_t)b3_ * st + 16);                    \
    map8s_pair(&a1r_, &a1i_, cp + (size_t)(b3_ + s3_) * st,                  \
               &c1r_, &c1i_, cp + (size_t)(b3_ + s3_) * st + 16);            \
    map8s_pair(&a2r_, &a2i_, cp + (size_t)(b3_ + 2 * s3_) * st,              \
               &c2r_, &c2i_, cp + (size_t)(b3_ + 2 * s3_) * st + 16);        \
    BR(b3_)             = a0r_; BI(b3_)             = a0i_;                  \
    BR_B(b3_)           = c0r_; BI_B(b3_)           = c0i_;                  \
    BR(b3_ + s3_)       = a1r_; BI(b3_ + s3_)       = a1i_;                  \
    BR_B(b3_ + s3_)     = c1r_; BI_B(b3_ + s3_)     = c1i_;                  \
    BR(b3_ + 2 * s3_)   = a2r_; BI(b3_ + 2 * s3_)   = a2i_;                  \
    BR_B(b3_ + 2 * s3_) = c2r_; BI_B(b3_ + 2 * s3_) = c2i_;                  \
} while (0)

/* ---- L=25 pencils: n = n1 + 5 n2, k = 5 k1 + k2 (DIF) / k1 + 5 k2 (DIT).
 * DIF, natural in -> digit-swapped out (X[k] lands at slot (k%5)*5 + k/5):
 *   stage 1, group n1: slots {n1 + 5i}, DFT5 over n2 -> k2, tw W25^{n1 k2}
 *   stage 2, group k2: slots {5k2 + i}, DFT5 over n1 -> k1, plain (or map)
 * DIT, digit-swapped in -> natural out:
 *   stage 1, group n1: slots {5n1 + i}, DFT5 over n2 -> k1, tw W25^{n1 k1}
 *   stage 2, group k1: slots {k1 + 5i}, DFT5 over n1 -> k2, plain (or map) */
/* final-stage forms for the two DFT5 pencil flavours: plain (z/y passes)
 * and map-fused (x pass).  gen_r5: the map-fused form defaults to the
 * paired-divide D5STAGEMP; -DGENPWP_NOMAPPAIR restores the r2-r4
 * one-vdivpd-per-site SSTM form for A/B and cross-arch races. */
#define D5FS_PLAIN(B, S) D5STAGE(B, S, 0, SSTW)
#ifdef GENPWP_NOMAPPAIR
#define D5FS_MAP(B, S)   D5STAGE(B, S, 0, SSTM)
#else
#define D5FS_MAP(B, S)   D5STAGEMP(B, S)
#endif

#define DEF_P25(NAME, FS5)                                                   \
static inline __attribute__((always_inline))                                 \
void NAME##_dif(double *restrict p, const ptrdiff_t st,                      \
                const double *restrict cp)                                   \
{                                                                            \
    (void)cp;                                                                \
    _Pragma("GCC unroll 5")                                                  \
    for (int n1_ = 0; n1_ < 5; ++n1_) D5STAGE(n1_, 5, n1_, SSTW);            \
    _Pragma("GCC unroll 5")                                                  \
    for (int k2_ = 0; k2_ < 5; ++k2_) FS5(5 * k2_, 1);                       \
}                                                                            \
static inline __attribute__((always_inline))                                 \
void NAME##_dit(double *restrict p, const ptrdiff_t st,                      \
                const double *restrict cp)                                   \
{                                                                            \
    (void)cp;                                                                \
    _Pragma("GCC unroll 5")                                                  \
    for (int n1_ = 0; n1_ < 5; ++n1_) D5STAGE(5 * n1_, 1, n1_, SSTW);        \
    _Pragma("GCC unroll 5")                                                  \
    for (int k1_ = 0; k1_ < 5; ++k1_) FS5(k1_, 5);                           \
}
DEF_P25(p25,  D5FS_PLAIN)                 /* plain final stage (z/y passes) */
DEF_P25(p25m, D5FS_MAP)                   /* map-fused final stage (x pass) */

/* ---- L=27 pencils: n = n1 + 3 n2 + 9 n3; trit-reversal is the involution.
 * DIF, natural in -> trit-reversed out (X[k] at slot rev(k)):
 *   stage 1, group (n2,n3): slots {3n2+n3 + 9i}, DFT3 over n1 -> k1,
 *            tw W27^{k1 (3n2+n3)}
 *   stage 2, group (k1,n3): slots {9k1+n3 + 3i}, DFT3 over n2 -> k2,
 *            tw W9^{k2 n3}
 *   stage 3, group (k1,k2): slots {9k1+3k2 + i}, DFT3 over n3 -> k3, plain
 * DIT, trit-reversed in -> natural out:
 *   stage 1, group (n1,n2): slots {9n1+3n2 + i}, DFT3 over n3 -> k1,
 *            tw W9^{n2 k1}
 *   stage 2, group (n1,k1): slots {9n1+k1 + 3i}, DFT3 over n2 -> k2,
 *            tw W27^{n1 (k1+3k2)}
 *   stage 3, group (k1,k2): slots {3k2+k1 + 9i}, DFT3 over n1 -> k3, plain */
/* final-stage forms for the DFT3 pencils (see D5FS_* above) */
#define D3FS_PLAIN(B, S) D3STAGE(B, S, 0, 0, 0, C27T, S27T, SSTW)
#ifdef GENPWP_NOMAPPAIR
#define D3FS_MAP(B, S)   D3STAGE(B, S, 0, 0, 0, C27T, S27T, SSTM)
#else
#define D3FS_MAP(B, S)   D3STAGEMP(B, S)
#endif

#define DEF_P27(NAME, FS3)                                                   \
static inline __attribute__((always_inline))                                 \
void NAME##_dif(double *restrict p, const ptrdiff_t st,                      \
                const double *restrict cp)                                   \
{                                                                            \
    (void)cp;                                                                \
    _Pragma("GCC unroll 3")                                                  \
    for (int n2_ = 0; n2_ < 3; ++n2_)                                        \
        _Pragma("GCC unroll 3")                                              \
        for (int n3_ = 0; n3_ < 3; ++n3_)                                    \
            D3STAGE(3 * n2_ + n3_, 9, 0,                                     \
                    3 * n2_ + n3_, 2 * (3 * n2_ + n3_), C27T, S27T, SSTW);   \
    _Pragma("GCC unroll 3")                                                  \
    for (int k1_ = 0; k1_ < 3; ++k1_)                                        \
        _Pragma("GCC unroll 3")                                              \
        for (int n3_ = 0; n3_ < 3; ++n3_)                                    \
            D3STAGE(9 * k1_ + n3_, 3, 0, n3_, 2 * n3_, C9T, S9T, SSTW);      \
    _Pragma("GCC unroll 3")                                                  \
    for (int k1_ = 0; k1_ < 3; ++k1_)                                        \
        _Pragma("GCC unroll 3")                                              \
        for (int k2_ = 0; k2_ < 3; ++k2_)                                    \
            FS3(9 * k1_ + 3 * k2_, 1);                                       \
}                                                                            \
static inline __attribute__((always_inline))                                 \
void NAME##_dit(double *restrict p, const ptrdiff_t st,                      \
                const double *restrict cp)                                   \
{                                                                            \
    (void)cp;                                                                \
    _Pragma("GCC unroll 3")                                                  \
    for (int n1_ = 0; n1_ < 3; ++n1_)                                        \
        _Pragma("GCC unroll 3")                                              \
        for (int n2_ = 0; n2_ < 3; ++n2_)                                    \
            D3STAGE(9 * n1_ + 3 * n2_, 1, 0, n2_, 2 * n2_, C9T, S9T, SSTW);  \
    _Pragma("GCC unroll 3")                                                  \
    for (int n1_ = 0; n1_ < 3; ++n1_)                                        \
        _Pragma("GCC unroll 3")                                              \
        for (int k1_ = 0; k1_ < 3; ++k1_)                                    \
            D3STAGE(9 * n1_ + k1_, 3, n1_ * k1_,                             \
                    n1_ * (k1_ + 3), n1_ * (k1_ + 6), C27T, S27T, SSTW);     \
    _Pragma("GCC unroll 3")                                                  \
    for (int k1_ = 0; k1_ < 3; ++k1_)                                        \
        _Pragma("GCC unroll 3")                                              \
        for (int k2_ = 0; k2_ < 3; ++k2_)                                    \
            FS3(3 * k2_ + k1_, 9);                                           \
}
DEF_P27(p27,  D3FS_PLAIN)
DEF_P27(p27m, D3FS_MAP)

/* gen_r6: two-column map-fused x-pass pencils (same stage/slot/twiddle
 * algebra as p27m, expanded per pair of adjacent flat columns).  MEASURED
 * AND REJECTED on the node as a default: quiet-window held-lease pairs at
 * 27 B=16 m=200 put the two-column form at 47.12-47.39 vs single-column
 * 45.18-45.57 (+4.4%, 4/5 pairs) -- the doubled straight-line body costs
 * more in the front end than the paired loads and the 3-vs-4 divides buy;
 * the x-pass streams were already covered by the OoO window.  Kept behind
 * -DGENPWP_XCOL2 (opt-in) for cross-arch races only. */
#ifdef GENPWP_XCOL2
static inline __attribute__((always_inline))
void p27m2_dif(double *restrict p, const ptrdiff_t st,
               const double *restrict cp)
{
    _Pragma("GCC unroll 3")
    for (int n2_ = 0; n2_ < 3; ++n2_)
        _Pragma("GCC unroll 3")
        for (int n3_ = 0; n3_ < 3; ++n3_)
            D3STAGE2(3 * n2_ + n3_, 9, 0,
                     3 * n2_ + n3_, 2 * (3 * n2_ + n3_), C27T, S27T);
    _Pragma("GCC unroll 3")
    for (int k1_ = 0; k1_ < 3; ++k1_)
        _Pragma("GCC unroll 3")
        for (int n3_ = 0; n3_ < 3; ++n3_)
            D3STAGE2(9 * k1_ + n3_, 3, 0, n3_, 2 * n3_, C9T, S9T);
    _Pragma("GCC unroll 3")
    for (int k1_ = 0; k1_ < 3; ++k1_)
        _Pragma("GCC unroll 3")
        for (int k2_ = 0; k2_ < 3; ++k2_)
            D3STAGEMP2(9 * k1_ + 3 * k2_, 1);
}
static inline __attribute__((always_inline))
void p27m2_dit(double *restrict p, const ptrdiff_t st,
               const double *restrict cp)
{
    _Pragma("GCC unroll 3")
    for (int n1_ = 0; n1_ < 3; ++n1_)
        _Pragma("GCC unroll 3")
        for (int n2_ = 0; n2_ < 3; ++n2_)
            D3STAGE2(9 * n1_ + 3 * n2_, 1, 0, n2_, 2 * n2_, C9T, S9T);
    _Pragma("GCC unroll 3")
    for (int n1_ = 0; n1_ < 3; ++n1_)
        _Pragma("GCC unroll 3")
        for (int k1_ = 0; k1_ < 3; ++k1_)
            D3STAGE2(9 * n1_ + k1_, 3, n1_ * k1_,
                     n1_ * (k1_ + 3), n1_ * (k1_ + 6), C27T, S27T);
    _Pragma("GCC unroll 3")
    for (int k1_ = 0; k1_ < 3; ++k1_)
        _Pragma("GCC unroll 3")
        for (int k2_ = 0; k2_ < 3; ++k2_)
            D3STAGEMP2(3 * k2_ + k1_, 9);
}
#endif /* GENPWP_XCOL2 */

/* padded plane strides, sites: L^2 rounded up to == 2 (mod 32) so plane
 * bytes == 256 (mod 4096) -- gen_pfa_small/gen_batchlane's anti-alias pad */
#define SOAPL25 642
#define SOAPL27 738

/* per-axis digit permutations (X[k] slot after a DIF pass; involutions) */
static const unsigned char g_pm25[25] = {
     0,  5, 10, 15, 20,  1,  6, 11, 16, 21,  2,  7, 12, 17, 22,
     3,  8, 13, 18, 23,  4,  9, 14, 19, 24 };
static const unsigned char g_pm27[27] = {
     0,  9, 18,  3, 12, 21,  6, 15, 24,  1, 10, 19,  4, 13, 22,
     7, 16, 25,  2, 11, 20,  5, 14, 23,  8, 17, 26 };

/* one chain step over one 8-volume group: zy sweep per x-plane (in place),
 * then the x pass with the map fused into the final-stage stores.  DIF and
 * DIT variants; C must be packed in the matching site layout (reversed for
 * DIF, natural for DIT). */
/* -DGENPWP_SOASCHED: pre-RA pressure scheduling on the soa_step wrappers
 * (gen_batchlane r3 measured -5.7..-10% on their register-explicit pencils).
 * Node verdict this round, control-first pairs with MKL steady: LOSES 32%
 * at 25 (34.3 -> 45.4) and 22% at 27 (47.8 -> 58.3) -- their win is specific
 * to spill-heavy codelets; these pencils hold <= 16 live vecs and gcc's
 * default schedule already forwards ~106 loads / kills ~61 stores per step
 * function (objdump audit).  Default stays OFF. */
#ifdef GENPWP_SOASCHED
#define SOAOPT __attribute__((optimize("schedule-insns", "sched-pressure")))
#else
#define SOAOPT
#endif

/* gen_r6: the x pass pairs adjacent flat columns through PZM2 when XP2 != 0
 * (a compile-time constant: the pair loop vanishes at XP2 == 0); the odd
 * L*L count leaves one trailing single column through PZM either way. */
#define DEF_SOASTEP(L, PLV, PZ, PZM, PZM2, XP2)                              \
static void SOAOPT soa_step_dif_##L(double *restrict S, const double *restrict C) \
{                                                                            \
    for (int x = 0; x < (L); ++x) {                                          \
        double *pl = S + (size_t)x * ((PLV) * 16);                           \
        for (int y = 0; y < (L); ++y)                                        \
            PZ##_dif(pl + (size_t)y * ((L) * 16), 16, 0);                    \
        for (int z = 0; z < (L); ++z)                                        \
            PZ##_dif(pl + (size_t)z * 16, (L) * 16, 0);                      \
    }                                                                        \
    int c = 0;                                                               \
    if (XP2)                                                                 \
        for (; c + 2 <= (L) * (L); c += 2)                                   \
            PZM2##_dif(S + (size_t)c * 16, (ptrdiff_t)((PLV) * 16),          \
                       C + (size_t)c * 16);                                  \
    for (; c < (L) * (L); ++c)                                               \
        PZM##_dif(S + (size_t)c * 16, (ptrdiff_t)((PLV) * 16),               \
                  C + (size_t)c * 16);                                       \
}                                                                            \
static void SOAOPT soa_step_dit_##L(double *restrict S, const double *restrict C) \
{                                                                            \
    for (int x = 0; x < (L); ++x) {                                          \
        double *pl = S + (size_t)x * ((PLV) * 16);                           \
        for (int y = 0; y < (L); ++y)                                        \
            PZ##_dit(pl + (size_t)y * ((L) * 16), 16, 0);                    \
        for (int z = 0; z < (L); ++z)                                        \
            PZ##_dit(pl + (size_t)z * 16, (L) * 16, 0);                      \
    }                                                                        \
    int c = 0;                                                               \
    if (XP2)                                                                 \
        for (; c + 2 <= (L) * (L); c += 2)                                   \
            PZM2##_dit(S + (size_t)c * 16, (ptrdiff_t)((PLV) * 16),          \
                       C + (size_t)c * 16);                                  \
    for (; c < (L) * (L); ++c)                                               \
        PZM##_dit(S + (size_t)c * 16, (ptrdiff_t)((PLV) * 16),               \
                  C + (size_t)c * 16);                                       \
}
DEF_SOASTEP(25, SOAPL25, p25, p25m, p25m, 0)
#ifdef GENPWP_XCOL2
DEF_SOASTEP(27, SOAPL27, p27, p27m, p27m2, 1)
#else
DEF_SOASTEP(27, SOAPL27, p27, p27m, p27m, 0)
#endif

/* pack 8 interleaved volumes (volume stride vstr complex) into the padded
 * SoA arena; pm = per-axis site permutation, NULL = natural.  Natural rows
 * go through gen_layout's gl_pack8; permuted rows scatter sites through the
 * same gl_tr8x8 network. */
static void soa_pack(const double _Complex *src, size_t vstr, double *S,
                     int L, int PL, const unsigned char *pm)
{
    for (int x = 0; x < L; ++x)
        for (int y = 0; y < L; ++y) {
            const double _Complex *row = src + ((size_t)x * L + y) * L;
            double *dst = S + ((size_t)(pm ? pm[x] : x) * PL
                             + (size_t)(pm ? pm[y] : y) * L) * 16;
            if (!pm) { gl_pack8(row, vstr, dst, (size_t)L); continue; }
            int s = 0;
            for (; s + 4 <= L; s += 4) {
                __m512d r[8];
                for (int t = 0; t < 8; ++t)
                    r[t] = _mm512_loadu_pd((const double *)(row + t * vstr + s));
                gl_tr8x8(r);
                for (int q = 0; q < 4; ++q) {
                    _mm512_storeu_pd(dst + (size_t)pm[s + q] * 16,     r[2 * q]);
                    _mm512_storeu_pd(dst + (size_t)pm[s + q] * 16 + 8, r[2 * q + 1]);
                }
            }
            for (; s < L; ++s)
                for (int t = 0; t < 8; ++t) {
                    dst[(size_t)pm[s] * 16 + t]     = creal(row[t * vstr + s]);
                    dst[(size_t)pm[s] * 16 + 8 + t] = cimag(row[t * vstr + s]);
                }
        }
}

/* exact inverse: SoA arena (site layout pm, NULL = natural) -> 8 volumes */
static void soa_unpack(const double *S, double _Complex *dst, size_t vstr,
                       int L, int PL, const unsigned char *pm)
{
    for (int x = 0; x < L; ++x)
        for (int y = 0; y < L; ++y) {
            double _Complex *row = dst + ((size_t)x * L + y) * L;
            const double *sr = S + ((size_t)(pm ? pm[x] : x) * PL
                                  + (size_t)(pm ? pm[y] : y) * L) * 16;
            if (!pm) { gl_unpack8(sr, row, vstr, (size_t)L); continue; }
            int s = 0;
            for (; s + 4 <= L; s += 4) {
                __m512d r[8];
                for (int q = 0; q < 4; ++q) {
                    r[2 * q]     = _mm512_loadu_pd(sr + (size_t)pm[s + q] * 16);
                    r[2 * q + 1] = _mm512_loadu_pd(sr + (size_t)pm[s + q] * 16 + 8);
                }
                gl_tr8x8(r);
                for (int t = 0; t < 8; ++t)
                    _mm512_storeu_pd((double *)(row + t * vstr + s), r[t]);
            }
            for (; s < L; ++s)
                for (int t = 0; t < 8; ++t)
                    row[t * vstr + s] = sr[(size_t)pm[s] * 16 + t]
                                      + I * sr[(size_t)pm[s] * 16 + 8 + t];
        }
}

#endif /* __AVX512F__ */

/* ============ gen_r12: WITHIN-VOLUME SoA engine at L = 100 ("wv") =========
 * ADOPTED from gen_batchlane gen_r11, close to verbatim -- their within-
 * volume engine took the r11 L=100 cell at 4059 us where this entry's best
 * interleaved family (ipp1 + THP re-home) reads ~4420-4465: the eight zmm
 * lanes carry eight X-PLANES of ONE volume (B=1 native), so
 *   - z-pencils (stride 1 site-vector) and y-pencils (stride ZP100 sv) are
 *     pure elementwise batch-lane code -- zero shuffles, zero twiddle swaps,
 *     zero spills where my interleaved shell pays 569 shuffles + ~458 spill
 *     slots per z-line (my own r8 port census priced the prize; their r11
 *     engine is the "fewer L1 round trips" answer my r10 record asked for);
 *   - both sweeps run back to back per ~1.3 MB slab: two-axes-per-pass
 *     fusion (the lit 11 Tier 2 item) for free;
 *   - only the x-pass crosses lanes: per (y, 8z) column, 26 trans8 in + 26
 *     out around a 104-sv scratch pencil, hidden under the pass's DRAM
 *     streams; the graded map fuses in-register at the dft100 stage-2
 *     stores (no separate map pass);
 *   - c is packed ONCE per chain into x-consumption order with the trans8
 *     lane permutation (lanex) baked into the deinterleave shuffles.
 * Pencil module: PFA(4 x 25), n = (4a + 25b) %% 100, stage 1 = 4 x DFT25
 * (map-free), safe placement 19 = 4^-1 mod 25 baked into stage-A stores so
 * stage 2 = 25 x DFT4 is NATURALLY in place with natural output order;
 * DFT25 = 5x5 CT through a 25-sv L1 scratch with 9 compiled-in w25
 * constant pairs (lit 11 Tier 1 routing).  2016 vector FP per pencil per 8
 * lanes, zero shuffles.  Slot tables are gen_batchlane's, verified by them
 * against numpy (gen100.py, 3 seeds) and re-gated here at create() (m=2
 * chain composition vs a gated execute + the driver's exact scalar map).
 * Raced as candidate "l100-wv", ranked LAST (the soa rule: a non-bit-
 * identical engine must beat the best interleaved by > 3%); the per-host
 * race + wisdom pinning make its different rounding safe across the
 * driver's two-process repeatability cmp (the r2 soa rationale).
 * DIFFERENCES from their engine, deliberate and small: my LDU/STU/VSH
 * primitives instead of their LD/ST/SH (same semantics), and no L=50 form
 * (their r11 measured it 14%% BEHIND my ipp at the L3-resident cell --
 * "do not spend r12 trying to close it with knobs", their words). */
#ifdef __AVX512F__

typedef double v8d  __attribute__((vector_size(64)));

#define V8C(x) { (x), (x), (x), (x), (x), (x), (x), (x) }
#define AIN static inline __attribute__((always_inline))

/* lane primitives + transpose network, gen_batchlane's (bl8 lineage) */
#define T1_LO 0,8,2,10,4,12,6,14
#define T1_HI 1,9,3,11,5,13,7,15
#define T2_LO 0,1,2,3,8,9,10,11
#define T2_HI 4,5,6,7,12,13,14,15
#define T3_LO 0,1,4,5,8,9,12,13
#define T3_HI 2,3,6,7,10,11,14,15

#define BF(a, b, LO, HI)                                                      \
    do {                                                                      \
        const v8d bf_ = VSH((a), (b), LO);                                    \
        (b) = VSH((a), (b), HI);                                              \
        (a) = bf_;                                                            \
    } while (0)

/* 8x8 transpose: out reg j lane l = in reg lanex[l] element j,
 * lanex = {0,1,4,5,2,3,6,7} (self-inverse) -- gen_batchlane's measured
 * trans8 semantics; their lane discipline notes apply verbatim. */
static inline void trans8(v8d *restrict m)
{
    BF(m[0], m[4], T2_LO, T2_HI);  BF(m[1], m[5], T2_LO, T2_HI);
    BF(m[2], m[6], T2_LO, T2_HI);  BF(m[3], m[7], T2_LO, T2_HI);
    BF(m[0], m[2], T3_LO, T3_HI);  BF(m[1], m[3], T3_LO, T3_HI);
    BF(m[4], m[6], T3_LO, T3_HI);  BF(m[5], m[7], T3_LO, T3_HI);
    BF(m[0], m[1], T1_LO, T1_HI);  BF(m[2], m[3], T1_LO, T1_HI);
    BF(m[4], m[5], T1_LO, T1_HI);  BF(m[6], m[7], T1_LO, T1_HI);
}

static inline void untrans_interleave(v8d *restrict r, v8d *restrict q)
{
    BF(r[0], r[1], T3_LO, T3_HI);  BF(r[2], r[3], T3_LO, T3_HI);
    BF(r[4], r[5], T3_LO, T3_HI);  BF(r[6], r[7], T3_LO, T3_HI);
    BF(q[0], q[1], T3_LO, T3_HI);  BF(q[2], q[3], T3_LO, T3_HI);
    BF(q[4], q[5], T3_LO, T3_HI);  BF(q[6], q[7], T3_LO, T3_HI);

    BF(r[0], r[2], T3_LO, T3_HI);  BF(r[1], r[3], T3_LO, T3_HI);
    BF(r[4], r[6], T3_LO, T3_HI);  BF(r[5], r[7], T3_LO, T3_HI);
    BF(q[0], q[2], T3_LO, T3_HI);  BF(q[1], q[3], T3_LO, T3_HI);
    BF(q[4], q[6], T3_LO, T3_HI);  BF(q[5], q[7], T3_LO, T3_HI);

    BF(r[0], q[0], T1_LO, T1_HI);  BF(r[1], q[1], T1_LO, T1_HI);
    BF(r[2], q[2], T1_LO, T1_HI);  BF(r[3], q[3], T1_LO, T1_HI);
    BF(r[4], q[4], T1_LO, T1_HI);  BF(r[5], q[5], T1_LO, T1_HI);
    BF(r[6], q[6], T1_LO, T1_HI);  BF(r[7], q[7], T1_LO, T1_HI);
}

static const unsigned char UO_V[16] = { 0,4,2,6,0,4,2,6,1,5,3,7,1,5,3,7 };
static const unsigned char UO_H[16] = { 0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1 };

/* pack/unpack between driver AoS (interleaved complex per lane source) and
 * the SoA slab (site = re[8]|im[8]); tails re-do an overlapping block */
static void pack_plane(const double *const vp[8], double *restrict Spl,
                       int nsites)
{
    for (int s = 0; s < nsites; s += 4) {
        if (s + 4 > nsites) s = nsites - 4;
        v8d m[8];
        for (int l = 0; l < 8; ++l) m[l] = LDU(vp[l] + 2 * (size_t)s);
        trans8(m);
        double *d = Spl + (size_t)s * 16;
        for (int j = 0; j < 8; ++j)
            STU(d + (size_t)(j >> 1) * 16 + (size_t)(j & 1) * 8, m[j]);
    }
}

static void unpack_plane(double *const vp[8], const double *restrict Spl,
                         int nsites, int nvol)
{
    for (int s = 0; s < nsites; s += 8) {
        if (s + 8 > nsites) s = nsites - 8;
        v8d r[8], q[8];
        const double *b = Spl + (size_t)s * 16;
        for (int j = 0; j < 8; ++j) {
            r[j] = *(const v8d *)(b + (size_t)j * 16);
            q[j] = *(const v8d *)(b + (size_t)j * 16 + 8);
        }
        untrans_interleave(r, q);
        for (int t = 0; t < 16; ++t) {
            const int v = UO_V[t];
            if (v < nvol)
                STU(vp[v] + 2 * (size_t)s + (size_t)UO_H[t] * 8,
                    t < 8 ? r[t] : q[t - 8]);
        }
    }
}

/* the graded map in split complex, gen_batchlane's ladder (rsqrt14 + 2
 * quadratic Newtons for |w|; then ONE exact vdivpd for 1/(1+|w|) -- their
 * memory-form-pencil verdict, and the same divider verdict as my r2 soa
 * A/B on this class's own engine) */
AIN void map8(v8d *restrict vr, v8d *restrict vi, const double *restrict cp,
              const int usediv)
{
    static const v8d eps = V8C(1e-300), half = V8C(0.5), c15 = V8C(1.5),
                     one = V8C(1.0);
    const v8d wr = *vr + *(const v8d *)cp;
    const v8d wi = *vi + *(const v8d *)(cp + 8);
    v8d s = wr * wr + eps;
    s = wi * wi + s;
    v8d y = (v8d)_mm512_rsqrt14_pd((__m512d)s);
    const v8d hs = s * half;
    v8d u = y * y;
    y = y * (c15 - hs * u);
    u = y * y;
    y = y * (c15 - hs * u);
    const v8d d = s * y + one;          /* 1 + |w| */
    v8d t;
    if (usediv) {
        t = one / d;
    } else {
        t = (v8d)_mm512_rcp14_pd((__m512d)d);
        t = t + t * (one - d * t);
        t = t + t * (one - d * t);
    }
    *vr = wr * t;
    *vi = wi * t;
}
#define WVDIV 1                          /* their MAPTAIL_GEN default        */

/* site-vector accessors: slot k of a pencil at base p, stride st doubles */
#define PR(p, st, k)  (*(v8d *)((p) + (size_t)(k) * (st)))
#define PI_(p, st, k) (*(v8d *)((p) + (size_t)(k) * (st) + 8))
#define SLR(p, st, k) ((v8d)LDU((p) + (size_t)(k) * (st)))
#define SLI(p, st, k) ((v8d)LDU((p) + (size_t)(k) * (st) + 8))

/* DFT5 core constants (theirs; exact to the last bit of double) */
static const v8d K25W = V8C(0.25);
static const v8d KQ5  = V8C(0.55901699437494742410);   /* sqrt(5)/4         */
static const v8d KS1  = V8C(0.95105651629515357212);   /* sin(2pi/5)        */
static const v8d KS2  = V8C(0.58778525229247312917);   /* sin(4pi/5)        */
static const v8d KPHI = V8C(1.61803398874989484820);   /* lifted v-pair     */
static const v8d KL5  = V8C(2.12662702088009983045);

#define M5BIND(T, p, st, i0, i1, i2, i3, i4)                                   \
        const v8d T##x0r = SLR(p, st, i0), T##x0i = SLI(p, st, i0);            \
        const v8d T##x1r = SLR(p, st, i1), T##x1i = SLI(p, st, i1);            \
        const v8d T##x2r = SLR(p, st, i2), T##x2i = SLI(p, st, i2);            \
        const v8d T##x3r = SLR(p, st, i3), T##x3i = SLI(p, st, i3);            \
        const v8d T##x4r = SLR(p, st, i4), T##x4i = SLI(p, st, i4);

/* gen_batchlane r7 LIFTED v-pair (their engine's measured default; on THIS
 * codelet shape -- 5 independent DFT25 scratch groups, not my in-place slot
 * pencils -- their verdict transfers with the engine) */
#define DFT5VPAIR(T)                                                           \
        const v8d T##uur = T##sar - KPHI * T##sbr;                             \
        const v8d T##uui = T##sai - KPHI * T##sbi;                             \
        const v8d T##v2r = KS2 * T##uur, T##v2i = KS2 * T##uui;                \
        const v8d T##v1r = KS1 * T##uur + KL5 * T##sbr;                        \
        const v8d T##v1i = KS1 * T##uui + KL5 * T##sbi;

#define DFT5CORE(T)                                                            \
        const v8d T##tar = T##x1r + T##x4r, T##tai = T##x1i + T##x4i;          \
        const v8d T##tbr = T##x2r + T##x3r, T##tbi = T##x2i + T##x3i;          \
        const v8d T##sar = T##x1r - T##x4r, T##sai = T##x1i - T##x4i;          \
        const v8d T##sbr = T##x2r - T##x3r, T##sbi = T##x2i - T##x3i;          \
        const v8d T##pr = T##tar + T##tbr, T##pi = T##tai + T##tbi;            \
        const v8d T##qr = T##tar - T##tbr, T##qi = T##tai - T##tbi;            \
        const v8d T##X0r = T##x0r + T##pr, T##X0i = T##x0i + T##pi;            \
        const v8d T##fr = T##x0r - K25W * T##pr, T##fi = T##x0i - K25W * T##pi;\
        const v8d T##A1r = T##fr + KQ5 * T##qr, T##A1i = T##fi + KQ5 * T##qi;  \
        const v8d T##A2r = T##fr - KQ5 * T##qr, T##A2i = T##fi - KQ5 * T##qi;  \
        DFT5VPAIR(T)

#define DFT5STORE(T, p, st, o0, o1, o2, o3, o4)                                \
        PR(p, st, o0) = T##X0r;              PI_(p, st, o0) = T##X0i;          \
        PR(p, st, o1) = T##A1r + T##v1i;     PI_(p, st, o1) = T##A1i - T##v1r; \
        PR(p, st, o4) = T##A1r - T##v1i;     PI_(p, st, o4) = T##A1i + T##v1r; \
        PR(p, st, o2) = T##A2r + T##v2i;     PI_(p, st, o2) = T##A2i - T##v2r; \
        PR(p, st, o3) = T##A2r - T##v2i;     PI_(p, st, o3) = T##A2i + T##v2r;

/* in-place DFT-4 on slots a0..a3 (dft100 stage 2, plain) */
#define M4IP(p, st, a0, a1, a2, a3)                                            \
    do {                                                                       \
        const v8d x0r = SLR(p, st, a0), x0i = SLI(p, st, a0);                  \
        const v8d x1r = SLR(p, st, a1), x1i = SLI(p, st, a1);                  \
        const v8d x2r = SLR(p, st, a2), x2i = SLI(p, st, a2);                  \
        const v8d x3r = SLR(p, st, a3), x3i = SLI(p, st, a3);                  \
        const v8d t0r = x0r + x2r, t0i = x0i + x2i;                            \
        const v8d t1r = x0r - x2r, t1i = x0i - x2i;                            \
        const v8d t2r = x1r + x3r, t2i = x1i + x3i;                            \
        const v8d t3r = x1r - x3r, t3i = x1i - x3i;                            \
        PR(p, st, a0) = t0r + t2r;  PI_(p, st, a0) = t0i + t2i;                \
        PR(p, st, a1) = t1r + t3i;  PI_(p, st, a1) = t1i - t3r;                \
        PR(p, st, a2) = t0r - t2r;  PI_(p, st, a2) = t0i - t2i;                \
        PR(p, st, a3) = t1r - t3i;  PI_(p, st, a3) = t1i + t3r;                \
    } while (0)

/* in-place DFT-4 + fused graded map (dft100 stage 2, chain x-pass) */
#define M4STM(p, st, cp, UD, i0, i1, i2, i3, o0, o1, o2, o3)                   \
    do {                                                                       \
        const v8d x0r = SLR(p, st, i0), x0i = SLI(p, st, i0);                  \
        const v8d x1r = SLR(p, st, i1), x1i = SLI(p, st, i1);                  \
        const v8d x2r = SLR(p, st, i2), x2i = SLI(p, st, i2);                  \
        const v8d x3r = SLR(p, st, i3), x3i = SLI(p, st, i3);                  \
        const v8d t0r = x0r + x2r, t0i = x0i + x2i;                            \
        const v8d t1r = x0r - x2r, t1i = x0i - x2i;                            \
        const v8d t2r = x1r + x3r, t2i = x1i + x3i;                            \
        const v8d t3r = x1r - x3r, t3i = x1i - x3i;                            \
        v8d zr, zi;                                                            \
        zr = t0r + t2r;  zi = t0i + t2i;                                       \
        map8(&zr, &zi, (cp) + (size_t)(o0) * (st), UD);                        \
        PR(p, st, o0) = zr;  PI_(p, st, o0) = zi;                              \
        zr = t1r + t3i;  zi = t1i - t3r;                                       \
        map8(&zr, &zi, (cp) + (size_t)(o1) * (st), UD);                        \
        PR(p, st, o1) = zr;  PI_(p, st, o1) = zi;                              \
        zr = t0r - t2r;  zi = t0i - t2i;                                       \
        map8(&zr, &zi, (cp) + (size_t)(o2) * (st), UD);                        \
        PR(p, st, o2) = zr;  PI_(p, st, o2) = zi;                              \
        zr = t1r - t3i;  zi = t1i + t3r;                                       \
        map8(&zr, &zi, (cp) + (size_t)(o3) * (st), UD);                        \
        PR(p, st, o3) = zr;  PI_(p, st, o3) = zi;                              \
    } while (0)

/* DFT25 = 5x5 CT stages through a 25-sv L1 scratch (their derivation:
 * stage A codelet a2 = DFT5 over group positions {5a1+a2}, output d1
 * twiddled by w25^(a2*d1), stored at scratch a2 + 5*d1; stage B codelet d1
 * = DFT5 over scratch row {5d1+a2}, output d2 to the PFA slot of
 * d = d1 + 5*d2 with safe placement baked in) */
#define M5CTA0(T, p, st, sc, a2, i0, i1, i2, i3, i4)                           \
    do {                                                                       \
        M5BIND(T, p, st, i0, i1, i2, i3, i4)                                   \
        DFT5CORE(T)                                                            \
        DFT5STORE(T, (sc) + (a2) * 16, 80, 0, 1, 2, 3, 4)                      \
    } while (0)

#define M5CTA(T, p, st, sc, a2, i0, i1, i2, i3, i4,                            \
              W1R, W1I, W2R, W2I, W3R, W3I, W4R, W4I)                          \
    do {                                                                       \
        M5BIND(T, p, st, i0, i1, i2, i3, i4)                                   \
        DFT5CORE(T)                                                            \
        double *const scA_ = (sc) + (a2) * 16;                                 \
        PR(scA_, 80, 0) = T##X0r;            PI_(scA_, 80, 0) = T##X0i;        \
        { const v8d t1r_ = T##A1r + T##v1i, t1i_ = T##A1i - T##v1r;            \
          PR(scA_, 80, 1) = t1r_ * (W1R) - t1i_ * (W1I);                       \
          PI_(scA_, 80, 1) = t1r_ * (W1I) + t1i_ * (W1R); }                    \
        { const v8d t4r_ = T##A1r - T##v1i, t4i_ = T##A1i + T##v1r;            \
          PR(scA_, 80, 4) = t4r_ * (W4R) - t4i_ * (W4I);                       \
          PI_(scA_, 80, 4) = t4r_ * (W4I) + t4i_ * (W4R); }                    \
        { const v8d t2r_ = T##A2r + T##v2i, t2i_ = T##A2i - T##v2r;            \
          PR(scA_, 80, 2) = t2r_ * (W2R) - t2i_ * (W2I);                       \
          PI_(scA_, 80, 2) = t2r_ * (W2I) + t2i_ * (W2R); }                    \
        { const v8d t3r_ = T##A2r - T##v2i, t3i_ = T##A2i + T##v2r;            \
          PR(scA_, 80, 3) = t3r_ * (W3R) - t3i_ * (W3I);                       \
          PI_(scA_, 80, 3) = t3r_ * (W3I) + t3i_ * (W3R); }                    \
    } while (0)

#define M5CTB(T, p, st, sc, d1, o0, o1, o2, o3, o4)                            \
    do {                                                                       \
        M5BIND(T, (sc) + (d1) * 80, 16, 0, 1, 2, 3, 4)                         \
        DFT5CORE(T)                                                            \
        DFT5STORE(T, p, st, o0, o1, o2, o3, o4)                                \
    } while (0)

/* W25 twiddles for the 5x5 CT (their compiled-in constants, exact to the
 * last bit; values agree with my C25T/S25T literals to the digit) */
static const v8d KW25R_1  = V8C(0.96858316112863108);
static const v8d KW25I_1  = V8C(-0.24868988716485479);
static const v8d KW25R_2  = V8C(0.87630668004386358);
static const v8d KW25I_2  = V8C(-0.48175367410171527);
static const v8d KW25R_3  = V8C(0.72896862742141155);
static const v8d KW25I_3  = V8C(-0.68454710592868873);
static const v8d KW25R_4  = V8C(0.53582679497899666);
static const v8d KW25I_4  = V8C(-0.84432792550201508);
static const v8d KW25R_6  = V8C(0.062790519529313374);
static const v8d KW25I_6  = V8C(-0.99802672842827156);
static const v8d KW25R_8  = V8C(-0.42577929156507266);
static const v8d KW25I_8  = V8C(-0.90482705246601958);
static const v8d KW25R_9  = V8C(-0.63742398974868975);
static const v8d KW25I_9  = V8C(-0.77051324277578925);
static const v8d KW25R_12 = V8C(-0.99211470131447788);
static const v8d KW25I_12 = V8C(-0.12533323356430426);
static const v8d KW25R_16 = V8C(-0.63742398974868975);
static const v8d KW25I_16 = V8C(0.77051324277578925);

/* the L=100 pencil: PFA(4 x 25), slot tables gen_batchlane gen_r11 verbatim
 * (numpy-verified by them; re-gated here).  Plain form (zy sweeps) and
 * map-fused form (chain x-pass). */
AIN void dft100_pencil(double *restrict p, const ptrdiff_t st)
{
    double sc[25 * 16] __attribute__((aligned(64)));
    M5CTA0(a_, p, st, sc, 0, 0, 20, 40, 60, 80);
    M5CTA(b_, p, st, sc, 1, 4, 24, 44, 64, 84,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 8, 28, 48, 68, 88,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 12, 32, 52, 72, 92,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 16, 36, 56, 76, 96,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 0, 80, 60, 40, 20);
    M5CTB(g_, p, st, sc, 1, 76, 56, 36, 16, 96);
    M5CTB(h_, p, st, sc, 2, 52, 32, 12, 92, 72);
    M5CTB(i_, p, st, sc, 3, 28, 8, 88, 68, 48);
    M5CTB(j_, p, st, sc, 4, 4, 84, 64, 44, 24);
    M5CTA0(a_, p, st, sc, 0, 25, 45, 65, 85, 5);
    M5CTA(b_, p, st, sc, 1, 29, 49, 69, 89, 9,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 33, 53, 73, 93, 13,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 37, 57, 77, 97, 17,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 41, 61, 81, 1, 21,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 25, 5, 85, 65, 45);
    M5CTB(g_, p, st, sc, 1, 1, 81, 61, 41, 21);
    M5CTB(h_, p, st, sc, 2, 77, 57, 37, 17, 97);
    M5CTB(i_, p, st, sc, 3, 53, 33, 13, 93, 73);
    M5CTB(j_, p, st, sc, 4, 29, 9, 89, 69, 49);
    M5CTA0(a_, p, st, sc, 0, 50, 70, 90, 10, 30);
    M5CTA(b_, p, st, sc, 1, 54, 74, 94, 14, 34,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 58, 78, 98, 18, 38,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 62, 82, 2, 22, 42,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 66, 86, 6, 26, 46,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 50, 30, 10, 90, 70);
    M5CTB(g_, p, st, sc, 1, 26, 6, 86, 66, 46);
    M5CTB(h_, p, st, sc, 2, 2, 82, 62, 42, 22);
    M5CTB(i_, p, st, sc, 3, 78, 58, 38, 18, 98);
    M5CTB(j_, p, st, sc, 4, 54, 34, 14, 94, 74);
    M5CTA0(a_, p, st, sc, 0, 75, 95, 15, 35, 55);
    M5CTA(b_, p, st, sc, 1, 79, 99, 19, 39, 59,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 83, 3, 23, 43, 63,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 87, 7, 27, 47, 67,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 91, 11, 31, 51, 71,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 75, 55, 35, 15, 95);
    M5CTB(g_, p, st, sc, 1, 51, 31, 11, 91, 71);
    M5CTB(h_, p, st, sc, 2, 27, 7, 87, 67, 47);
    M5CTB(i_, p, st, sc, 3, 3, 83, 63, 43, 23);
    M5CTB(j_, p, st, sc, 4, 79, 59, 39, 19, 99);
    M4IP(p, st, 0, 25, 50, 75);
    M4IP(p, st, 76, 1, 26, 51);
    M4IP(p, st, 52, 77, 2, 27);
    M4IP(p, st, 28, 53, 78, 3);
    M4IP(p, st, 4, 29, 54, 79);
    M4IP(p, st, 80, 5, 30, 55);
    M4IP(p, st, 56, 81, 6, 31);
    M4IP(p, st, 32, 57, 82, 7);
    M4IP(p, st, 8, 33, 58, 83);
    M4IP(p, st, 84, 9, 34, 59);
    M4IP(p, st, 60, 85, 10, 35);
    M4IP(p, st, 36, 61, 86, 11);
    M4IP(p, st, 12, 37, 62, 87);
    M4IP(p, st, 88, 13, 38, 63);
    M4IP(p, st, 64, 89, 14, 39);
    M4IP(p, st, 40, 65, 90, 15);
    M4IP(p, st, 16, 41, 66, 91);
    M4IP(p, st, 92, 17, 42, 67);
    M4IP(p, st, 68, 93, 18, 43);
    M4IP(p, st, 44, 69, 94, 19);
    M4IP(p, st, 20, 45, 70, 95);
    M4IP(p, st, 96, 21, 46, 71);
    M4IP(p, st, 72, 97, 22, 47);
    M4IP(p, st, 48, 73, 98, 23);
    M4IP(p, st, 24, 49, 74, 99);
}

AIN void dft100_pencil_map(double *restrict p, const ptrdiff_t st,
                           const double *restrict cp)
{
    double sc[25 * 16] __attribute__((aligned(64)));
    M5CTA0(a_, p, st, sc, 0, 0, 20, 40, 60, 80);
    M5CTA(b_, p, st, sc, 1, 4, 24, 44, 64, 84,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 8, 28, 48, 68, 88,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 12, 32, 52, 72, 92,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 16, 36, 56, 76, 96,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 0, 80, 60, 40, 20);
    M5CTB(g_, p, st, sc, 1, 76, 56, 36, 16, 96);
    M5CTB(h_, p, st, sc, 2, 52, 32, 12, 92, 72);
    M5CTB(i_, p, st, sc, 3, 28, 8, 88, 68, 48);
    M5CTB(j_, p, st, sc, 4, 4, 84, 64, 44, 24);
    M5CTA0(a_, p, st, sc, 0, 25, 45, 65, 85, 5);
    M5CTA(b_, p, st, sc, 1, 29, 49, 69, 89, 9,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 33, 53, 73, 93, 13,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 37, 57, 77, 97, 17,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 41, 61, 81, 1, 21,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 25, 5, 85, 65, 45);
    M5CTB(g_, p, st, sc, 1, 1, 81, 61, 41, 21);
    M5CTB(h_, p, st, sc, 2, 77, 57, 37, 17, 97);
    M5CTB(i_, p, st, sc, 3, 53, 33, 13, 93, 73);
    M5CTB(j_, p, st, sc, 4, 29, 9, 89, 69, 49);
    M5CTA0(a_, p, st, sc, 0, 50, 70, 90, 10, 30);
    M5CTA(b_, p, st, sc, 1, 54, 74, 94, 14, 34,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 58, 78, 98, 18, 38,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 62, 82, 2, 22, 42,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 66, 86, 6, 26, 46,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 50, 30, 10, 90, 70);
    M5CTB(g_, p, st, sc, 1, 26, 6, 86, 66, 46);
    M5CTB(h_, p, st, sc, 2, 2, 82, 62, 42, 22);
    M5CTB(i_, p, st, sc, 3, 78, 58, 38, 18, 98);
    M5CTB(j_, p, st, sc, 4, 54, 34, 14, 94, 74);
    M5CTA0(a_, p, st, sc, 0, 75, 95, 15, 35, 55);
    M5CTA(b_, p, st, sc, 1, 79, 99, 19, 39, 59,
           KW25R_1, KW25I_1, KW25R_2, KW25I_2, KW25R_3, KW25I_3, KW25R_4, KW25I_4);
    M5CTA(c_, p, st, sc, 2, 83, 3, 23, 43, 63,
           KW25R_2, KW25I_2, KW25R_4, KW25I_4, KW25R_6, KW25I_6, KW25R_8, KW25I_8);
    M5CTA(d_, p, st, sc, 3, 87, 7, 27, 47, 67,
           KW25R_3, KW25I_3, KW25R_6, KW25I_6, KW25R_9, KW25I_9, KW25R_12, KW25I_12);
    M5CTA(e_, p, st, sc, 4, 91, 11, 31, 51, 71,
           KW25R_4, KW25I_4, KW25R_8, KW25I_8, KW25R_12, KW25I_12, KW25R_16, KW25I_16);
    M5CTB(f_, p, st, sc, 0, 75, 55, 35, 15, 95);
    M5CTB(g_, p, st, sc, 1, 51, 31, 11, 91, 71);
    M5CTB(h_, p, st, sc, 2, 27, 7, 87, 67, 47);
    M5CTB(i_, p, st, sc, 3, 3, 83, 63, 43, 23);
    M5CTB(j_, p, st, sc, 4, 79, 59, 39, 19, 99);
    M4STM(p, st, cp, WVDIV, 0, 25, 50, 75,   0, 25, 50, 75);
    M4STM(p, st, cp, WVDIV, 76, 1, 26, 51,   76, 1, 26, 51);
    M4STM(p, st, cp, WVDIV, 52, 77, 2, 27,   52, 77, 2, 27);
    M4STM(p, st, cp, WVDIV, 28, 53, 78, 3,   28, 53, 78, 3);
    M4STM(p, st, cp, WVDIV, 4, 29, 54, 79,   4, 29, 54, 79);
    M4STM(p, st, cp, WVDIV, 80, 5, 30, 55,   80, 5, 30, 55);
    M4STM(p, st, cp, WVDIV, 56, 81, 6, 31,   56, 81, 6, 31);
    M4STM(p, st, cp, WVDIV, 32, 57, 82, 7,   32, 57, 82, 7);
    M4STM(p, st, cp, WVDIV, 8, 33, 58, 83,   8, 33, 58, 83);
    M4STM(p, st, cp, WVDIV, 84, 9, 34, 59,   84, 9, 34, 59);
    M4STM(p, st, cp, WVDIV, 60, 85, 10, 35,   60, 85, 10, 35);
    M4STM(p, st, cp, WVDIV, 36, 61, 86, 11,   36, 61, 86, 11);
    M4STM(p, st, cp, WVDIV, 12, 37, 62, 87,   12, 37, 62, 87);
    M4STM(p, st, cp, WVDIV, 88, 13, 38, 63,   88, 13, 38, 63);
    M4STM(p, st, cp, WVDIV, 64, 89, 14, 39,   64, 89, 14, 39);
    M4STM(p, st, cp, WVDIV, 40, 65, 90, 15,   40, 65, 90, 15);
    M4STM(p, st, cp, WVDIV, 16, 41, 66, 91,   16, 41, 66, 91);
    M4STM(p, st, cp, WVDIV, 92, 17, 42, 67,   92, 17, 42, 67);
    M4STM(p, st, cp, WVDIV, 68, 93, 18, 43,   68, 93, 18, 43);
    M4STM(p, st, cp, WVDIV, 44, 69, 94, 19,   44, 69, 94, 19);
    M4STM(p, st, cp, WVDIV, 20, 45, 70, 95,   20, 45, 70, 95);
    M4STM(p, st, cp, WVDIV, 96, 21, 46, 71,   96, 21, 46, 71);
    M4STM(p, st, cp, WVDIV, 72, 97, 22, 47,   72, 97, 22, 47);
    M4STM(p, st, cp, WVDIV, 48, 73, 98, 23,   48, 73, 98, 23);
    M4STM(p, st, cp, WVDIV, 24, 49, 74, 99,   24, 49, 74, 99);
}

/* geometry: 13 x-slabs (slab 12 = 4 pad lanes, replicated, never unpacked);
 * row stride ZP100 = 101 sv; slab stride SLST100 = 10114 sv (slab bytes
 * == 256 mod 4096, house rule); c columns CST100 = 101 sv */
#define ZP100   101
#define SLST100 10114
#define CST100  101

static void sweep_zy_100(double *restrict slab)
{
    for (int y = 0; y < 100; ++y)
        dft100_pencil(slab + (size_t)y * (ZP100 * 16), 16);      /* z-pencils */
    for (int z = 0; z < 100; ++z)
        dft100_pencil(slab + (size_t)z * 16, ZP100 * 16);        /* y-pencils */
}

static const int LX8[8] = { 0, 1, 4, 5, 2, 3, 6, 7 };            /* lanex */

AIN void xcol_load_100(const double *restrict S, double *restrict Pn,
                       int y, int zg, int ndz)
{
    const v8d vz = V8C(0.0);
    for (int xb = 0; xb < 13; ++xb) {
        const double *base = S + ((size_t)xb * SLST100 + (size_t)y * ZP100
                                  + (size_t)zg * 8) * 16;
        v8d r[8], q[8];
        for (int dz = 0; dz < 8; ++dz) {
            if (dz < ndz) {
                r[dz] = LDU(base + (size_t)dz * 16);
                q[dz] = LDU(base + (size_t)dz * 16 + 8);
            } else {
                r[dz] = vz;
                q[dz] = vz;
            }
        }
        trans8(r);
        trans8(q);
        double *dst = Pn + (size_t)xb * (8 * 16);
        for (int j = 0; j < 8; ++j) {
            STU(dst + (size_t)LX8[j] * 16, r[j]);
            STU(dst + (size_t)LX8[j] * 16 + 8, q[j]);
        }
    }
}

AIN void xcol_store_100(double *restrict S, const double *restrict Pn,
                        int y, int zg, int ndz)
{
    for (int xb = 0; xb < 13; ++xb) {
        const double *src = Pn + (size_t)xb * (8 * 16);
        v8d r[8], q[8];
        for (int i = 0; i < 8; ++i) {
            r[i] = LDU(src + (size_t)i * 16);
            q[i] = LDU(src + (size_t)i * 16 + 8);
        }
        trans8(r);
        trans8(q);
        double *base = S + ((size_t)xb * SLST100 + (size_t)y * ZP100
                            + (size_t)zg * 8) * 16;
        for (int j = 0; j < 8; ++j) {
            const int dz = LX8[j];
            if (dz < ndz) {
                STU(base + (size_t)dz * 16, r[j]);
                STU(base + (size_t)dz * 16 + 8, q[j]);
            }
        }
    }
}

static void xpass_map_100(double *restrict S, const double *restrict CT)
{
    double Pn[104 * 16] __attribute__((aligned(64)));
    for (int y = 0; y < 100; ++y)
        for (int zg = 0; zg < 13; ++zg) {
            const int ndz = (zg == 12) ? 4 : 8;
            const double *cp = CT + ((size_t)y * 13 + zg) * (CST100 * 16);
            xcol_load_100(S, Pn, y, zg, ndz);
            dft100_pencil_map(Pn, 16, cp);
            xcol_store_100(S, Pn, y, zg, ndz);
        }
}

static void pack_slab_100(const double *restrict vin, double *restrict S,
                          int s)
{
    const int nl = (s == 12) ? 4 : 8;
    for (int y = 0; y < 100; ++y) {
        const double *vp[8];
        for (int l = 0; l < 8; ++l) {
            const int x = 8 * s + (l < nl ? l : nl - 1);
            vp[l] = vin + ((size_t)x * 100 + y) * 200;
        }
        pack_plane(vp, S + ((size_t)s * SLST100 + (size_t)y * ZP100) * 16,
                   100);
    }
}

static void pack_vol_100(const double *restrict vin, double *restrict S)
{
    for (int s = 0; s < 13; ++s)
        pack_slab_100(vin, S, s);
}

static void unpack_vol_100(double *restrict vout, const double *restrict S)
{
    for (int s = 0; s < 13; ++s) {
        const int nl = (s == 12) ? 4 : 8;
        for (int y = 0; y < 100; ++y) {
            double *op[8];
            for (int l = 0; l < 8; ++l) {
                const int x = 8 * s + (l < nl ? l : nl - 1);
                op[l] = vout + ((size_t)x * 100 + y) * 200;
            }
            unpack_plane(op, S + ((size_t)s * SLST100 + (size_t)y * ZP100) * 16,
                         100, nl);
        }
    }
}

/* c packed once per chain into x-consumption order: column (y, zg), slot x,
 * lane l = z = 8zg + lanex[l]; zg=12 pulls its high half from a zero vector
 * (avoids reading past the caller's array, zeroes the pad lanes) */
static void pack_ct_100(const double *restrict c, double *restrict CT)
{
    const v8d vz = V8C(0.0);
    for (int y = 0; y < 100; ++y)
        for (int zg = 0; zg < 13; ++zg) {
            double *col = CT + ((size_t)y * 13 + zg) * (CST100 * 16);
            const size_t zb = (size_t)zg * 8;
            for (int x = 0; x < 100; ++x) {
                const double *src = c + (((size_t)x * 100 + y) * 100 + zb) * 2;
                const v8d a = LDU(src);
                const v8d b = (zg == 12) ? vz : (v8d)LDU(src + 8);
                STU(col + (size_t)x * 16, VSH(a, b, 0, 2, 8, 10, 4, 6, 12, 14));
                STU(col + (size_t)x * 16 + 8,
                    VSH(a, b, 1, 3, 9, 11, 5, 7, 13, 15));
            }
        }
}

static void chainsteps_100(double *restrict S, const double *restrict CT,
                           int m)
{
    for (int step = 0; step < m; ++step) {
        for (int s = 0; s < 13; ++s)
            sweep_zy_100(S + (size_t)s * (SLST100 * 16));
        xpass_map_100(S, CT);
    }
}

/* ==== gen_r14: single-shot fft3d_execute() through the wv engine ==========
 * The r14 seam, on this entry, is exactly L=100 B=1: the chain has run the
 * within-volume engine since r12 (~4.05 ms/step INCLUDING the map) while
 * execute() stayed on the interleaved x_ip path (measured 5.23 ms single,
 * node) -- benchFFT times execute().  A naive reroute (pack_vol + sweeps +
 * x-pass + unpack_vol) pays the pack/unpack 16 MB round trips UNAMORTIZED,
 * which the chain never sees; both are therefore fused away:
 *   - ENTRY: pack slab s, sweep it while it is still L2/L3-hot (pack_vol's
 *     whole-volume pass would round-trip all 16 MB before the first pencil);
 *   - EXIT: the x-pass pencil's outputs leave for the driver's interleaved
 *     out DIRECTLY from the Pn scratch -- 2 shuffles + 2 aligned 64 B stores
 *     per slot with the lanex permutation baked into the shuffle indices
 *     (Pn slot k lane l = z = 8zg + lanex[l]; the pack_ct convention).
 *     This deletes xcol_store's trans8 scatter AND the whole unpack pass
 *     (gen_layout gen_r14's fuse-the-exit prescription, their gl_sti8x8
 *     rationale applied to this engine's own exit).  gen_bluestein r14's
 *     cold-source lesson is honored by construction: the one compulsory
 *     cold read (pack) stays sequential per plane row.
 * No c, no map: the plain dft100_pencil.  Modes (GENPWP_WVEXEC, read once
 * at create): 1 = fused exit, plain stores; 2 = fused exit, NT streaming
 * stores for the cold out (every store is a full 64 B line; falls back to
 * plain when out is not 64 B-aligned); 3 = unfused control (pack_vol +
 * sweeps + xpass_plain + unpack_vol) for A/B attribution. */
static void xpass_plain_100(double *restrict S)
{
    double Pn[104 * 16] __attribute__((aligned(64)));
    for (int y = 0; y < 100; ++y)
        for (int zg = 0; zg < 13; ++zg) {
            const int ndz = (zg == 12) ? 4 : 8;
            xcol_load_100(S, Pn, y, zg, ndz);
            dft100_pencil(Pn, 16);
            xcol_store_100(S, Pn, y, zg, ndz);
        }
}

#define WVEX_LO 0, 8, 1, 9, 4, 12, 5, 13     /* z offsets 0..3, lanex baked */
#define WVEX_HI 2, 10, 3, 11, 6, 14, 7, 15   /* z offsets 4..7              */

static void xpass_exit_100(double *restrict S, double *restrict vout,
                           const int nt)
{
    double Pn[104 * 16] __attribute__((aligned(64)));
    for (int y = 0; y < 100; ++y)
        for (int zg = 0; zg < 13; ++zg) {
            const int ndz = (zg == 12) ? 4 : 8;
            xcol_load_100(S, Pn, y, zg, ndz);
            dft100_pencil(Pn, 16);
            double *orow = vout + ((size_t)y * 100 + (size_t)zg * 8) * 2;
            if (nt) {
                for (int k = 0; k < 100; ++k) {
                    const v8d rv = *(const v8d *)(Pn + (size_t)k * 16);
                    const v8d qv = *(const v8d *)(Pn + (size_t)k * 16 + 8);
                    double *op = orow + (size_t)k * 20000;
                    _mm512_stream_pd(op, (__m512d)VSH(rv, qv, WVEX_LO));
                    if (ndz == 8)
                        _mm512_stream_pd(op + 8, (__m512d)VSH(rv, qv, WVEX_HI));
                }
            } else {
                for (int k = 0; k < 100; ++k) {
                    const v8d rv = *(const v8d *)(Pn + (size_t)k * 16);
                    const v8d qv = *(const v8d *)(Pn + (size_t)k * 16 + 8);
                    double *op = orow + (size_t)k * 20000;
                    STU(op, VSH(rv, qv, WVEX_LO));
                    if (ndz == 8)
                        STU(op + 8, VSH(rv, qv, WVEX_HI));
                }
            }
        }
    if (nt) _mm_sfence();
}

static void wv_exec_100(double *restrict S, const double *restrict vin,
                        double *restrict vout, long nvol, int mode)
{
    for (long v = 0; v < nvol; ++v) {
        const double *iv = vin + (size_t)v * 2000000;
        double       *ov = vout + (size_t)v * 2000000;
        if (mode == 3) {                            /* unfused control arm  */
            pack_vol_100(iv, S);
            for (int s = 0; s < 13; ++s)
                sweep_zy_100(S + (size_t)s * (SLST100 * 16));
            xpass_plain_100(S);
            unpack_vol_100(ov, S);
            continue;
        }
        for (int s = 0; s < 13; ++s) {              /* slab-fused entry     */
            pack_slab_100(iv, S, s);
            sweep_zy_100(S + (size_t)s * (SLST100 * 16));
        }
        xpass_exit_100(S, ov,
                       mode == 2 && ((uintptr_t)ov & 63) == 0);
    }
}

#endif /* __AVX512F__ */

/* ---- plan, gate, race, API ---------------------------------------------- */

typedef void (*execpl_fn)(const double *in, double *out, long nvol,
                          double *M, double *P);
typedef void (*chainpl_fn)(const double *cur, double *dst, const double *cf,
                           long nvol, double *M, double *P);

struct fft3d_plan {
    int L, batch;
    execpl_fn  fn;                /* NULL -> dense fallback                  */
    chainpl_fn cfn;               /* NULL -> execute + scalar map            */
    int use_soa;                  /* chain owned by the SoA-8 lane engine    */
    int use_wv;                   /* chain owned by the within-volume engine
                                     (gen_r12, ADOPTED from gen_batchlane)   */
    int exec_wv;                  /* gen_r14: fft3d_execute() through the wv
                                     engine (0 off; else the wv_exec mode) --
                                     set only with use_wv, so the wisdom-
                                     pinned chain verdict keys it            */
    int dm;                       /* cfn is a deferred-map (ipm) step        */
    double *P;                    /* plane scratch (heap: stack 4K-aliases)  */
    double *M;                    /* padded mid volume (phase 1 -> phase 2)  */
    double *X;                    /* batch state volumes for chain ping-pong */
    double *SA, *CN, *CR;         /* SoA state + c (natural / digit-reversed)*/
    gl_arena soa_ar;              /* THP arena for the three (gen_layout)    */
    int soa_live;
    double *STV, *CV;             /* gen_r11 re-home: THP state + c volumes
                                     (also the race's tout/tcf at these
                                     sizes -- trial-regime fidelity)         */
    gl_arena ch_ar;               /* their arena (gen_layout gl_map_huge)    */
    int ch_live;
    double *WS, *WC;              /* gen_r12 within-volume slabs + x-order c */
    gl_arena wv_ar;               /* THP arena for the two (gen_layout)      */
    int wv_live;
    void   *rawP, *rawM, *rawX;
    double _Complex *w, *tmp;     /* dense fallback state                    */
};

/* the driver's MAP_STEP, verbatim semantics: z -> (z+c)/(1+|z+c|) */
static void map_scalar(const double *z, const double *c, double *o, size_t npts)
{
    for (size_t i = 0; i < npts; ++i) {
        double re = z[2 * i]     + c[2 * i];
        double im = z[2 * i + 1] + c[2 * i + 1];
        double sc = 1.0 / (1.0 + sqrt(re * re + im * im));
        o[2 * i]     = re * sc;
        o[2 * i + 1] = im * sc;
    }
}

const char *fft3d_name(void) { return "gen_powp"; }

static char g_desc[224];
const char *fft3d_description(void)
{
    return g_desc[0] ? g_desc
        : "prime-power CT, exact twiddles: OWN 25 (5x5), 27 (3x9), 50 (GT 25x2), "
          "100 (GT 25x4) + odd-p^k class 49/81/121/125 (round 6); two-sweep "
          "zmm lanes + SoA-8 DIF/DIT lane chain at 25/27 (paired-div map, "
          "r7 3-shear twiddles, r8 opt-in n1_5/lifted DFT5), volume-major chain, deferred-map "
          "ipm/ipp (pair-packed ladder), c-bypass ipq/ipk/iqn, "
          "gate+race+r9 noise-gated banked verdicts, "
          "r10 playoff-authoritative picks in create(), "
          "r11 THP re-home of chain state+c past L3 scale (gl_thp_bytes-gated), "
          "r12 within-volume SoA chain candidate at 100 (gen_batchlane r11 "
          "adopted: 8 x-planes in lanes, slab-fused zy sweeps, trans8 x-pass, "
          "PFA 4x25 map-fused pencils) raced vs the interleaved families, "
          "r13 recorded-winner default pools at 50/100, "
          "r14 wv single-shot execute (slab-fused pack, direct lanex exit)";
}
int fft3d_supports(int L)
{
    return L == 25 || L == 27 || L == 50 || L == 100 ||
           L == 49 || L == 81 || L == 121 || L == 125;
}

/* scalar O(L^2)-per-line reference: independent ground truth for the
 * create()-time gate (from gen_pfa_large / L45_pfa's ref3d) */
static void __attribute__((unused))       /* unused in the no-AVX512 build */
refnd(int L, const double _Complex *in, double _Complex *out)
{
    const size_t NP = (size_t)L * L;
    double _Complex Wt[128], buf[128];
    for (int k = 0; k < L; ++k)
        Wt[k] = cexp(-2.0 * M_PI * I * (double)k / (double)L);
    for (int x = 0; x < L; ++x)                       /* z axis: in -> out  */
        for (int y = 0; y < L; ++y) {
            const double _Complex *r = in  + ((size_t)x * L + y) * L;
            double _Complex       *w = out + ((size_t)x * L + y) * L;
            for (int k = 0; k < L; ++k) {
                double _Complex s = 0;
                for (int j = 0; j < L; ++j) s += r[j] * Wt[(j * k) % L];
                w[k] = s;
            }
        }
    for (int x = 0; x < L; ++x)                       /* y axis, in place   */
        for (int z = 0; z < L; ++z) {
            double _Complex *base = out + (size_t)x * NP + z;
            for (int j = 0; j < L; ++j) buf[j] = base[(size_t)j * L];
            for (int k = 0; k < L; ++k) {
                double _Complex s = 0;
                for (int j = 0; j < L; ++j) s += buf[j] * Wt[(j * k) % L];
                base[(size_t)k * L] = s;
            }
        }
    for (int y = 0; y < L; ++y)                       /* x axis, in place   */
        for (int z = 0; z < L; ++z) {
            double _Complex *base = out + (size_t)y * L + z;
            for (int j = 0; j < L; ++j) buf[j] = base[(size_t)j * NP];
            for (int k = 0; k < L; ++k) {
                double _Complex s = 0;
                for (int j = 0; j < L; ++j) s += buf[j] * Wt[(j * k) % L];
                base[(size_t)k * NP] = s;
            }
        }
}

/* dense fallback (the old stub): floor of last resort, known correct */
static void dense_contract(const double _Complex *w, int L,
                           const double _Complex *in, double _Complex *out,
                           int inner)
{
    for (int k = 0; k < L; ++k)
        for (int c = 0; c < inner; ++c) {
            double _Complex acc = 0.0;
            for (int j = 0; j < L; ++j)
                acc += w[(size_t)k * L + j] * in[(size_t)j * inner + c];
            out[(size_t)k * inner + c] = acc;
        }
}

static void dense_exec(const fft3d_plan *p, const double _Complex *in,
                       double _Complex *out)
{
    const int L = p->L;
    const size_t volume = (size_t)L * L * L;
    for (int b = 0; b < p->batch; ++b) {
        const double _Complex *src = in + (size_t)b * volume;
        double _Complex *dst = out + (size_t)b * volume;
        dense_contract(p->w, L, src, dst, L * L);
        for (int x = 0; x < L; ++x)
            dense_contract(p->w, L, dst + (size_t)x * L * L,
                           p->tmp + (size_t)x * L * L, L);
        for (size_t row = 0; row < volume / (size_t)L; ++row)
            dense_contract(p->w, L, p->tmp + row * (size_t)L,
                           dst + row * (size_t)L, 1);
    }
}

static int dense_setup(fft3d_plan *p)
{
    const int L = p->L;
    p->w   = malloc((size_t)L * L * sizeof *p->w);
    p->tmp = malloc((size_t)L * L * L * sizeof *p->tmp);
    if (!p->w || !p->tmp) return 0;
    for (int k = 0; k < L; ++k)
        for (int j = 0; j < L; ++j) {
            double phase = -2.0 * M_PI * (double)((k * j) % L) / (double)L;
            p->w[(size_t)k * L + j] = cos(phase) + I * sin(phase);
        }
    return 1;
}

#ifdef __AVX512F__

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int rel_ok(const double *got, const double *ref, size_t n)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = got[i] - ref[i];
        num += d * d; den += ref[i] * ref[i];
    }
    return num <= den * 1e-26;                        /* rel L2 < 1e-13 */
}

struct candpl { execpl_fn fn; chainpl_fn cfn; int pf, rank, dm;
                const char *nm; int wv; };

/* candidates are raced on the CHAIN step (the graded workload); the exec fn
 * rides along with the winning candidate.  FIVE chain families:
 *   ip*: everything in place + a sequential map pass (one volume-sized
 *        working set; wins when miss streams bind -- gen_pfa_large L=100)
 *   ipf: in place AND map fused into phase 2's stores (no map pass)
 *   f*:  map fused into phase 2, routed through the padded mid volume M
 *   ipm: (gen_r3, ADOPTED from gen_pfa_large gen_r3) deferred map applied
 *        on the NEXT step's p1 loads -- dm == 1, needs the whole-chain
 *        schedule in fft3d_chain; highest rank (must clearly win)
 *   ipp: (gen_r4, ADOPTED from gen_pfa_large gen_r4) deferred map as a
 *        per-plane sequential PREPASS into an L2 scratch plane (M's base);
 *        dm == 1 like ipm, ranked between the base families and ipm
 *   soa: (25/27, batch %% 8 == 0) the SoA-8 batch-lane whole-chain engine;
 *        cfn == NULL marks it, must beat the best interleaved by > 3% */
static const struct candpl g_c25[]  = {
    { x_ip0_25,  xc_ip0_25,   0, 0, 0, "l25-ip0", 0   },
    { x_ip1_25,  xc_ip1_25,   1, 1, 0, "l25-ip1", 0   },
    { x_ip0_25,  xc_ipf_25,   2, 2, 0, "l25-ipf", 0   },
    { x_pf0_25,  xc_pf0_25,   3, 3, 0, "l25-f0", 0    },
    { x_pf1_25,  xc_pfr_25,   4, 4, 0, "l25-fr", 0    },
    { x_pf1_25,  xc_pfrw_25,  5, 5, 0, "l25-frw", 0   },
    { x_ip0_25,  0,           6, 6, 0, "l25-soa", 0   },
    { x_ip0_25,  xc_ipp0_25,  9, 7, 1, "l25-ipp0", 0  } };
static const struct candpl g_c27[]  = {
    { x_ip0_27,  xc_ip0_27,   0, 0, 0, "l27-ip0", 0   },
    { x_ip1_27,  xc_ip1_27,   1, 1, 0, "l27-ip1", 0   },
    { x_ip0_27,  xc_ipf_27,   2, 2, 0, "l27-ipf", 0   },
    { x_pf0_27,  xc_pf0_27,   3, 3, 0, "l27-f0", 0    },
    { x_pf1_27,  xc_pfr_27,   4, 4, 0, "l27-fr", 0    },
    { x_pf1_27,  xc_pfrw_27,  5, 5, 0, "l27-frw", 0   },
    { x_ip0_27,  0,           6, 6, 0, "l27-soa", 0   },
    { x_ip0_27,  xc_ipp0_27,  9, 7, 1, "l27-ipp0", 0  } };
/* rank note (gen_r4): at 50/100 the ipp family is ranked FIRST -- 5-pair
 * same-core alternation on the node (batchlane r4's protocol) puts ipp1 at
 * -4.0..-4.5% vs ip0 at 100 (quiet floors 4905-4929 vs 5110-5160 us), and
 * both the node race and wallaby put ipp ahead at 50; a busy-window race
 * margin can shrink under the 3% hysteresis, and the tie must then fall to
 * the measured winner, not the simpler loser.
 *
 * gen_r13 POOL HYGIENE at the scored 50/100 cells.  The r12 scoring
 * window's cold race at 50 B=4 banked l50-ipm0 (tie=1, margin -1.9%):
 * ipm0 was trial-best in that window, "decided" its playoff vs rank-0 in
 * the same window, and the r10 authoritative override honored it -- and
 * the graded cell shipped 480.6 us where the ipp class runs ~415 (r12
 * board: gen_pfa_large 413.4; my own r11 board number 413.9).  ipm has
 * never won 50 or 100 on ANY host in twelve rounds; neither have f0/fr/
 * frw (at 50/100), ipq1/iqn1 (anywhere), or the c-bypass family at 50
 * where the batch's c IS the L3 reuse set (r5: ipk1 548 vs ip1 469).
 * Every never-winner a biased window can elevate is a scored-cell risk
 * with zero recorded upside, and every extra arm lengthens the cold race
 * (37 s worst case under r10 contention).  The DEFAULT pools now carry
 * only families with at least one recorded WIN at that size on some host
 * (50: ipp0/ipp1/ip0/ip1/ipf; 100: ipk1/ipp1/ipp0/ip0/ip1/wv);
 * GENPWP_FULLPOOL=1 restores the r12 pools (cross-arch forensics, and
 * GENPWP_PF forcing of a trimmed id needs it).  The wisdom sig covers the
 * offered name set, so the two pools never share verdicts; pf ids are
 * unchanged so forcing stays comparable across rounds. */
static const struct candpl g_c50[]  = {
    { x_ip0_50,  xc_ipp0_50,  9, 0, 1, "l50-ipp0", 0  },
    { x_ip1_50,  xc_ipp1_50, 10, 1, 1, "l50-ipp1", 0  },
    { x_ip0_50,  xc_ip0_50,   0, 2, 0, "l50-ip0", 0   },
    { x_ip1_50,  xc_ip1_50,   1, 3, 0, "l50-ip1", 0   },
    { x_ip0_50,  xc_ipf_50,   2, 4, 0, "l50-ipf", 0   } };
static const struct candpl g_c50f[] = {
    { x_ip0_50,  xc_ipp0_50,  9, 0, 1, "l50-ipp0", 0  },
    { x_ip1_50,  xc_ipp1_50, 10, 1, 1, "l50-ipp1", 0  },
    { x_ip0_50,  xc_ip0_50,   0, 2, 0, "l50-ip0", 0   },
    { x_ip1_50,  xc_ip1_50,   1, 3, 0, "l50-ip1", 0   },
    { x_ip0_50,  xc_ipf_50,   2, 4, 0, "l50-ipf", 0   },
    { x_pf0_50,  xc_pf0_50,   3, 5, 0, "l50-f0", 0    },
    { x_pf1_50,  xc_pfr_50,   4, 6, 0, "l50-fr", 0    },
    { x_pf1_50,  xc_pfrw_50,  5, 7, 0, "l50-frw", 0   },
    { x_ip0_50,  xc_ipm0_50,  7, 8, 1, "l50-ipm0", 0  },
    { x_ip1_50,  xc_ipm1_50,  8, 9, 1, "l50-ipm1", 0  },
    { x_ip1_50,  xc_ipq1_50, 11, 10, 1, "l50-ipq1", 0 },
    { x_ip1_50,  xc_ipk1_50, 12, 11, 1, "l50-ipk1", 0 },
    { x_ip1_50,  xc_iqn1_50, 13, 12, 0, "l50-iqn1", 0 } };
/* rank note (gen_r6): ipk1 is ranked FIRST at 100 -- the r5 rank question
 * settled by 5/5 held-lease pairs in a window that ran quiet -> busy
 * (pf12 vs pf10 mins: 5282/5317, 5027/5399, 5606/6355, 5499/6577,
 * 5458/5686; -0.6% to -16%, the margin WIDENING under contention: flushing
 * clean c lines frees L3 that contention would otherwise squeeze --
 * gen_pfa_large r4's smaller-footprint-is-contention-armor, confirmed).
 * At 50 B=4 the bypass still loses big (r5: ipk1 548 vs ip1 469); 50
 * ranks unchanged.
 * gen_r10 CAVEAT: the ipk1-first evidence is a80n0's; on a81n2 (the r9/r10
 * scoring node, same Gold 6326 SKU) held-lease graded pairs go the OTHER
 * way -- ipp1 wins 5/6, floors 4538-4619 vs ipk1's 4712-4780 us.  Ranks
 * stay (a80n0's history is real); the r10 playoff-authoritative rule in
 * tune() is what lets each host's own long-horizon race overrule them. */
static const struct candpl g_c100[] = {
    { x_ip1_100, xc_ipk1_100,12, 0, 1, "l100-ipk1", 0 },
    { x_ip1_100, xc_ipp1_100,10, 1, 1, "l100-ipp1", 0 },
    { x_ip0_100, xc_ipp0_100, 9, 2, 1, "l100-ipp0", 0 },
    { x_ip0_100, xc_ip0_100,  0, 3, 0, "l100-ip0", 0  },
    { x_ip1_100, xc_ip1_100,  1, 4, 0, "l100-ip1", 0  },
    /* gen_r12: within-volume SoA engine (ADOPTED from gen_batchlane gen_r11
     * -- the r11 board's L=100 winner).  cfn == NULL + wv marker; ranked
     * LAST like soa: the non-bit-identical engine must beat the best
     * interleaved by > 3% in its own long-horizon playoff. */
    { x_ip1_100, 0,          15, 5, 0, "l100-wv", 1 } };
static const struct candpl g_c100f[] = {
    { x_ip1_100, xc_ipk1_100,12, 0, 1, "l100-ipk1", 0 },
    { x_ip1_100, xc_ipp1_100,10, 1, 1, "l100-ipp1", 0 },
    { x_ip0_100, xc_ipp0_100, 9, 2, 1, "l100-ipp0", 0 },
    { x_ip0_100, xc_ip0_100,  0, 3, 0, "l100-ip0", 0  },
    { x_ip1_100, xc_ip1_100,  1, 4, 0, "l100-ip1", 0  },
    { x_ip0_100, xc_ipf_100,  2, 5, 0, "l100-ipf", 0  },
    { x_pf0_100, xc_pf0_100,  3, 6, 0, "l100-f0", 0   },
    { x_pf1_100, xc_pfr_100,  4, 7, 0, "l100-fr", 0   },
    { x_pf1_100, xc_pfrw_100, 5, 8, 0, "l100-frw", 0  },
    { x_ip0_100, xc_ipm0_100, 7, 9, 1, "l100-ipm0", 0 },
    { x_ip1_100, xc_ipm1_100, 8, 10, 1, "l100-ipm1", 0 },
    { x_ip1_100, xc_ipq1_100,11, 11, 1, "l100-ipq1", 0 },
    { x_ip1_100, xc_iqn1_100,13, 12, 0, "l100-iqn1", 0 },
    { x_ip1_100, 0,          15, 13, 0, "l100-wv", 1 } };
/* gen_r3 lite sizes (round-6 coverage): reduced candidate pool */
static const struct candpl g_c49[]  = {
    { x_ip0_49,  xc_ip0_49,   0, 0, 0, "l49-ip0", 0   },
    { x_ip0_49,  xc_ipp0_49,  9, 1, 1, "l49-ipp0", 0  },
    { x_ip0_49,  xc_ipf_49,   2, 2, 0, "l49-ipf", 0   },
    { x_pf0_49,  xc_pf0_49,   3, 3, 0, "l49-f0", 0    },
    { x_ip0_49,  xc_ipm0_49,  7, 4, 1, "l49-ipm0", 0  },
    { x_ip0_49,  xc_ipq0_49, 14, 5, 1, "l49-ipq0", 0  } };
static const struct candpl g_c81[]  = {
    { x_ip0_81,  xc_ip0_81,   0, 0, 0, "l81-ip0", 0   },
    { x_ip0_81,  xc_ipp0_81,  9, 1, 1, "l81-ipp0", 0  },
    { x_ip0_81,  xc_ipf_81,   2, 2, 0, "l81-ipf", 0   },
    { x_pf0_81,  xc_pf0_81,   3, 3, 0, "l81-f0", 0    },
    { x_ip0_81,  xc_ipm0_81,  7, 4, 1, "l81-ipm0", 0  },
    { x_ip0_81,  xc_ipq0_81, 14, 5, 1, "l81-ipq0", 0  } };
static const struct candpl g_c121[] = {
    { x_ip0_121, xc_ip0_121,  0, 0, 0, "l121-ip0", 0  },
    { x_ip0_121, xc_ipp0_121, 9, 1, 1, "l121-ipp0", 0 },
    { x_ip0_121, xc_ipf_121,  2, 2, 0, "l121-ipf", 0  },
    { x_pf0_121, xc_pf0_121,  3, 3, 0, "l121-f0", 0   },
    { x_ip0_121, xc_ipm0_121, 7, 4, 1, "l121-ipm0", 0 },
    { x_ip0_121, xc_ipq0_121,14, 5, 1, "l121-ipq0", 0 } };
static const struct candpl g_c125[] = {
    { x_ip0_125, xc_ip0_125,  0, 0, 0, "l125-ip0", 0  },
    { x_ip0_125, xc_ipp0_125, 9, 1, 1, "l125-ipp0", 0 },
    { x_ip0_125, xc_ipf_125,  2, 2, 0, "l125-ipf", 0  },
    { x_pf0_125, xc_pf0_125,  3, 3, 0, "l125-f0", 0   },
    { x_ip0_125, xc_ipm0_125, 7, 4, 1, "l125-ipm0", 0 },
    { x_ip0_125, xc_ipq0_125,14, 5, 1, "l125-ipq0", 0 } };
#define NCMAX 14

/* the whole graded chain through the SoA-8 lane engine: per group of 8
 * volumes, pack once (x0 natural; c in BOTH site layouts), alternate
 * DIF/DIT steps in place, unpack once (layout by m's parity) */
static void soa_chain_n(fft3d_plan *p, const double _Complex *x0,
                        const double _Complex *c, double _Complex *out,
                        int m, long nvol)
{
    const int L  = p->L;
    const int PL = (L == 25) ? SOAPL25 : SOAPL27;
    const unsigned char *pm = (L == 25) ? g_pm25 : g_pm27;
    void (*stepd)(double *restrict, const double *restrict) =
        (L == 25) ? soa_step_dif_25 : soa_step_dif_27;
    void (*stept)(double *restrict, const double *restrict) =
        (L == 25) ? soa_step_dit_25 : soa_step_dit_27;
    const size_t vstr = (size_t)L * L * L;
    for (long g = 0; g < nvol / 8; ++g) {
        const double _Complex *xg = x0 + (size_t)(8 * g) * vstr;
        const double _Complex *cg = c  + (size_t)(8 * g) * vstr;
        soa_pack(xg, vstr, p->SA, L, PL, NULL);
        soa_pack(cg, vstr, p->CN, L, PL, NULL);
        soa_pack(cg, vstr, p->CR, L, PL, pm);
        for (int s = 0; s < m; ++s)
            if (s & 1) stept(p->SA, p->CN);
            else       stepd(p->SA, p->CR);
        soa_unpack(p->SA, out + (size_t)(8 * g) * vstr, vstr, L, PL,
                   (m & 1) ? pm : NULL);
    }
}

/* one within-volume chain step on the packed arena (gen_r12): 13 zy slab
 * sweeps + the map-fused x-pass -- the unit the trials and playoff time */
static void wv_step_100(fft3d_plan *p)
{
    for (int s = 0; s < 13; ++s)
        sweep_zy_100(p->WS + (size_t)s * (SLST100 * 16));
    xpass_map_100(p->WS, p->WC);
}

/* the whole graded chain through the within-volume engine: per volume,
 * pack state + c once, run all m steps in the THP slabs, unpack once
 * (volume-major; per-volume chains are independent) */
static void wv_chain_n(fft3d_plan *p, const double _Complex *x0,
                       const double _Complex *c, double _Complex *out,
                       int m, long nvol)
{
    const size_t vstr = (size_t)p->L * p->L * p->L;
    for (long v = 0; v < nvol; ++v) {
        pack_vol_100((const double *)(x0 + (size_t)v * vstr), p->WS);
        pack_ct_100((const double *)(c + (size_t)v * vstr), p->WC);
        chainsteps_100(p->WS, p->WC, m);
        unpack_vol_100((double *)(out + (size_t)v * vstr), p->WS);
    }
}

static const char *powp_fam(int L)
{
    switch (L) {
    case 25:  return "5x5";          case 27:  return "3x9(3x3)";
    case 50:  return "GT 25x2(5x5)"; case 100: return "GT 25x4(5x5)";
    case 49:  return "7x7";          case 81:  return "9x9(3x3)";
    case 121: return "11x11";        default:  return "5x25(5x5)";
    }
}

static void install_pick(fft3d_plan *p, const struct candpl *cd, int pick,
                         int trust_chain)
{
    p->fn = cd[pick].fn;
    p->cfn = NULL;
    p->use_soa = 0;
    p->use_wv = 0;
    p->exec_wv = 0;
    p->dm = 0;
    if (!getenv("GENPWP_NOFUSE")) {
        if (cd[pick].wv)             p->use_wv = 1;
        /* gen_r14: p->exec_wv (execute through the wv engine) is decided
         * SEPARATELY by wv_exec_lookup / the exec race after this install --
         * the single-shot winner is a per-host question distinct from the
         * chain's (ICL: interleaved wins; SPR: wv-NT by ~40%). */
        else if (!cd[pick].cfn)      p->use_soa = 1;
        else if (trust_chain) {      p->cfn = cd[pick].cfn;
                                     p->dm  = cd[pick].dm; }
        /* trust_chain == 0: the cold path gates cfn itself afterwards */
    }
    snprintf(g_desc, sizeof g_desc,
             "powp CT %s exact tw%s; pick: %s (B=%d)",
             powp_fam(p->L),
             p->use_wv  ? ", within-volume SoA chain (r12, gen_batchlane)"
             : p->use_soa ? ", SoA-8 lane chain (DIF/DIT in place)"
             : p->dm    ? ", two-sweep, deferred map"
                        : ", two-sweep",
             cd[pick].nm, p->batch);
}

/* ==== gen_r14: the execute-path verdict at the wv size ====================
 * benchFFT times single-shot fft3d_execute(); the chain's wv advantage is
 * steady-state (pack amortized over m, map fused into the x-pass), so the
 * single-shot winner is a SEPARATE per-host question: on a80n0 (ICL) the
 * interleaved execute wins (4.30-4.55 ms vs wv-NT 4.60-4.63 in the same
 * window), on wallaby (SPR) wv-NT wins by ~40% (3.36 vs 5.62 ms).  Two
 * arms -- "x-il" (the interleaved p->fn) and "wv-nt" (wv_exec mode 2) --
 * raced at create() right after the chain verdict, banked under their own
 * key (tag exec11, sig over the two arm names) so the driver's two-process
 * repeatability cmp stays pinned; wv-nt, the non-bit-identical arm, must
 * clear the 3% simplest-first hysteresis (the soa/wv rule). */
static void wv_exec_keyf(char *out, size_t n, const fft3d_plan *p)
{
    char key[GR_KEY_MAX];
    gr_cand sc[2];
    memset(sc, 0, sizeof sc);
    sc[0].name = "x-il";
    sc[1].name = "wv-nt";
    gr_keyf(key, sizeof key, "gen_powp", "exec11", p->L, gr_bucket(p->batch));
    snprintf(out, n, "%s#%08x", key, gr_sig(sc, 2));
}

/* env force first, then wisdom; returns 1 if a verdict was applied (the
 * cold path races on 0).  A warm-path miss leaves the interleaved execute
 * (simplest-first) -- it cannot happen in the driver's two-process pair,
 * because the cold create stores the exec verdict in the same call that
 * stores the chain verdict the warm process is hitting. */
static int wv_exec_lookup(fft3d_plan *p)
{
    p->exec_wv = 0;
    if (!p->use_wv || getenv("GENPWP_NOWVEXEC")) return 1;
    { const char *m_ = getenv("GENPWP_WVEXEC");
      if (m_) { int v = atoi(m_);
                p->exec_wv = (v >= 1 && v <= 3) ? v : 0;
                return 1; } }
    if (getenv("GEN_RACE_NO_WISDOM") || getenv("GEN_RACE_REFRESH")) return 0;
    char fk[GR_KEY_MAX + 16], wname[64];
    int widx = -1, wtie = 0;
    double wus = 0.0;
    wv_exec_keyf(fk, sizeof fk, p);
    if (gr_wisdom_lookup(fk, wname, sizeof wname, &widx, &wtie, &wus)) {
        if (!strcmp(wname, "wv-nt")) p->exec_wv = 2;
        return 1;
    }
    return 0;
}

/* ==== gen_r9: noise-gated verdict machinery (PMU audit avenue 1) =========
 * Floor-stability metric: spread of the 3 smallest samples.  A trial arm has
 * "demonstrated its floor" when its 3 best rounds agree to within the
 * tolerance -- the min-of-mins methodology every held-lease A/B in this
 * record uses, applied to the race's own evidence.  n < 3 samples = floor
 * not demonstrated (returns 1.0 = maximally untrusted). */
static double pwp_spread3(const double *v, int n)
{
    if (n < 3) return 1.0;
    double s1 = 1e300, s2 = 1e300, s3 = 1e300;
    for (int i = 0; i < n; ++i) {
        double x = v[i];
        if (x < s1)      { s3 = s2; s2 = s1; s1 = x; }
        else if (x < s2) { s3 = s2; s2 = x; }
        else if (x < s3) { s3 = x; }
    }
    return (s3 - s1) / s1;
}

/* one trial round of candidate c, exactly the shape the graded chain runs
 * (volume-major in place for the interleaved families, whole-group steps on
 * the packed arenas for soa); returns seconds per VOLUME-step.  Factored out
 * of the r4 race loop so the r9 stability extension re-runs the identical
 * measurement. */
static double pwp_trial_once(const struct candpl *cd, int c, fft3d_plan *p,
                             double *tout, const double *tcf, int nv, int R,
                             size_t VD,
                             void (*sd_)(double *restrict,
                                         const double *restrict),
                             void (*st_)(double *restrict,
                                         const double *restrict))
{
    if (cd[c].wv) {                    /* gen_r12 within-volume whole-chain
                                        * steps on the packed THP slabs (the
                                        * graded regime by construction)     */
        wv_step_100(p);                                       /* warm x2    */
        wv_step_100(p);
        double t0 = now_s();
        for (int r = 0; r < R; ++r) wv_step_100(p);
        return (now_s() - t0) / (double)R;    /* one volume-step per step   */
    }
    if (!cd[c].cfn) {                                 /* soa whole-chain    */
        for (long g_ = 0; g_ < nv / 8; ++g_)
            { sd_(p->SA, p->CR); st_(p->SA, p->CN); }         /* warm       */
        double t0 = now_s();
        for (int r = 0; r < R; ++r)
            for (long g_ = 0; g_ < nv / 8; ++g_)
                { sd_(p->SA, p->CR); st_(p->SA, p->CN); }
        return (now_s() - t0) / ((double)R * 2.0 * (double)nv);
    }
    double tsum = 0.0;                                /* volume-major       */
    for (int b = 0; b < nv; ++b) {
        double       *tv = tout + (size_t)b * VD;
        const double *cv = tcf  + (size_t)b * VD;
        cd[c].cfn(tv, tv, cv, 1, p->M, p->P);                 /* warm       */
        /* gen_r10: at volumes past L3 scale, two extra warm steps so each
         * candidate is timed in ITS OWN cache regime -- the round-robin
         * hands a c-custody predecessor's flushed hierarchy to whoever runs
         * next (at 100 that is always ipp1 after ipk1, table order), and
         * one warm step does not re-converge 16 MB of c. */
        if (VD * sizeof(double) > ((size_t)8 << 20)) {
            cd[c].cfn(tv, tv, cv, 1, p->M, p->P);
            cd[c].cfn(tv, tv, cv, 1, p->M, p->P);
        }
        double t0 = now_s();
        for (int r = 0; r < R; ++r)
            cd[c].cfn(tv, tv, cv, 1, p->M, p->P);
        tsum += now_s() - t0;
    }
    return tsum / ((double)R * (double)nv);
}

/* gate + race + wisdom; installs p->fn (or leaves NULL: dense fallback) */
static void tune(fft3d_plan *p)
{
    const int L = p->L;
    const size_t VD = (size_t)2 * L * L * L;
    const struct candpl *cd;
    int NC;
    const int fullpool = getenv("GENPWP_FULLPOOL") != NULL;
    switch (L) {
    case 25:  cd = g_c25;  NC = 8; break;
    case 27:  cd = g_c27;  NC = 8; break;
    case 50:  cd = fullpool ? g_c50f  : g_c50;
              NC = fullpool ? 13 : 5;  break;
    case 100: cd = fullpool ? g_c100f : g_c100;
              NC = fullpool ? 14 : 6;  break;
    case 49:  cd = g_c49;  NC = 6; break;
    case 81:  cd = g_c81;  NC = 6; break;
    case 121: cd = g_c121; NC = 6; break;
    default:  cd = g_c125; NC = 6; break;
    }

    int    live[NCMAX];
    double tc[NCMAX];
    for (int c = 0; c < NC; ++c) { live[c] = 1; tc[c] = 1e300; }
    for (int c = 0; c < NC; ++c) {
        if (!cd[c].cfn && !cd[c].wv && !p->SA) live[c] = 0;  /* soa arena   */
        if (cd[c].wv && !p->wv_live)           live[c] = 0;  /* wv arena    */
    }

    { const char *e = getenv("GENPWP_PF");            /* monitor forcing */
      if (e) { int v = atoi(e);
               for (int c = 0; c < NC; ++c) if (cd[c].pf != v) live[c] = 0;
               int any = 0;
               for (int c = 0; c < NC; ++c) any |= live[c];
               if (!any) for (int c = 0; c < NC; ++c)
                   live[c] = cd[c].wv ? p->wv_live
                           : (cd[c].cfn || p->SA ? 1 : 0); } }

    /* gen_r11: THP re-home arena (ADOPTED from gen_layout gen_r11, their
     * adoption recipe for this entry verbatim).  On THP=madvise hosts (the
     * scoring nodes) the driver's posix_memalign buffers are 4K-backed at
     * any size; at volumes past L3 scale the chain streams state+c through
     * ~12K page walks per step (measured: ~163K walk-active cycles + ~405K
     * STLB-hit lookups per step at 100, ~0.8% of the step).  The plan owns
     * a prefaulted 2 MiB-page arena sized for the RACE buffers (nvr trial
     * volumes each for state and c; the graded chain uses volume 0 of
     * each), so trials and graded chain share one cache+TLB regime.
     * Allocated BEFORE the wisdom lookup: a warm-hit plan must re-home
     * exactly like the cold-raced plan that banked the verdict. */
    if (VD * sizeof(double) >= ((size_t)8 << 20) &&
        !getenv("GENPWP_NOREHOME")) {
        const int cap_ = (int)(1 + (size_t)32 * 1024 * 1024
                                       / (VD * sizeof(double)));
        const int nvr_ = p->batch < cap_ ? p->batch : cap_;
        const size_t tb_ = (size_t)nvr_ * VD * sizeof(double);
        if (gl_arena_init(&p->ch_ar, 2 * (tb_ + 8192) + (64 << 10)) == 0) {
            p->STV = (double *)gl_arena_take(&p->ch_ar, tb_ + 4096);
            p->CV  = (double *)gl_arena_take(&p->ch_ar, tb_ + 4096);
            if (p->STV && p->CV) p->ch_live = 1;
            else { gl_arena_destroy(&p->ch_ar); p->STV = p->CV = NULL; }
        }
    }

    /* per-host wisdom (ADOPTED: gen_race).  The key carries the signature of
     * the OFFERED candidate set, so a changed pool (or a GENPWP_PF filter)
     * misses instead of silently replaying stale wisdom.  A hit is the 50 ms
     * warm-create path AND pins the pick across processes -- which is what
     * makes the non-bit-identical soa family safe against the driver's
     * two-process repeatability cmp. */
    char fullkey[GR_KEY_MAX + 16];
    { char key[GR_KEY_MAX];
      gr_cand sc[NCMAX]; int ns = 0;
      memset(sc, 0, sizeof sc);
      for (int c = 0; c < NC; ++c)
          if (live[c]) sc[ns++].name = cd[c].nm;
      /* gen_r4: tag bumped chain2 -> chain4 (gen_pfa_large r4's practice) --
       * the race now times the VOLUME-MAJOR schedule, so a stale step-major
       * verdict must never be installed from wisdom.  gen_r5: chain4 ->
       * chain5 (pool gained the c-bypass families; the soa x-pass map is
       * paired-div, so a stale soa-vs-interleaved margin must re-race).
       * gen_r6: chain5 -> chain6 (the race gained the soa playoff and the
       * 27 x-pass went two-column: a stale short-trial verdict -- notably
       * the r5 board's mis-picked l27-ip0 -- must never replay).  gen_r9:
       * chain6 -> chain7 (the race gained noise-gated adaptive trials and
       * quality-marked verdicts; the r8 scoring window's contended l25-ip0
       * "tie" was still live in wisdom_a80n0.json and must never replay).
       * gen_r10: chain7 -> chain8 (a decided challenger playoff is now
       * authoritative between its arms; the r9 scoring window's banked
       * l100-ipk1 rank-hysteresis "tie" -- a measured -6.3% on a81n2 --
       * must never replay).  gen_r11: chain8 -> chain9 (trials at re-home
       * sizes now run on the THP arena -- the graded regime; a chain8
       * verdict was measured in the 4K regime and must re-race).
       * gen_r13: chain9 -> chain10 (the 50/100 default pools are trimmed to
       * recorded winners; the r12 scoring window's banked l50-ipm0 "tie" --
       * a measured ~-15% on the r12 board -- must never replay.  The pool
       * change also re-sigs 50/100; the tag bump re-keys 25/27 as well so
       * every cell's scoring verdict is re-raced under the new protocol.)
       * gen_r14: chain10 -> chain11 (a banked l100-wv now implies the NEW
       * exec gate passed on that host -- an r13-era wv verdict was banked
       * before wv_exec existed and must not enable the ungated exec path). */
      gr_keyf(key, sizeof key, "gen_powp", "chain11", L, gr_bucket(p->batch));
      snprintf(fullkey, sizeof fullkey, "%s#%08x", key, gr_sig(sc, ns)); }

    if (!getenv("GEN_RACE_NO_WISDOM") && !getenv("GEN_RACE_REFRESH")) {
        char wname[64]; int widx = -1, wtie = 0; double wus = 0.0;
        if (gr_wisdom_lookup(fullkey, wname, sizeof wname, &widx, &wtie,
                             &wus)) {
            /* gen_r9: verdicts carry a quality marker.  A plain name is a
             * TIGHT (noise-gated) verdict -- banked, honored until the
             * tag/sig machinery re-keys it.  "name~q<pct>@<unixtime>" is
             * PROVISIONAL (the race stayed noisy after its extension):
             * honored only within GENPWP_NQHORIZON so it pins the driver's
             * two repeatability processes but cannot survive into a later
             * scoring window -- that window re-races instead. */
            double wq = 0.0; long wt = -1;
            char *mk = strchr(wname, '~');
            if (mk) { *mk = 0; sscanf(mk + 1, "q%lf@%ld", &wq, &wt); }
            long hor = getenv("GENPWP_NQHORIZON")
                     ? atol(getenv("GENPWP_NQHORIZON")) : 1800;
            if (wt < 0 || (long)time(NULL) - wt <= hor)
                for (int c = 0; c < NC; ++c)
                    if (live[c] && !strcmp(cd[c].nm, wname)) {
                        install_pick(p, cd, c, 1); /* warm: no gate, no race */
                        wv_exec_lookup(p);         /* gen_r14: exec verdict  */
                        return;
                    }
        }
    }

    /* arena: stream realistically at large batch, cap the footprint; a
     * multiple of 8 volumes whenever the soa candidate is offered.  tcf is
     * a DISTINCT c-field buffer (gen_r3, ADOPTED from gen_pfa_large's r3
     * fix: passing tin as both state and c made ipm's back-to-back state+c
     * loads hit the same lines, halving its apparent read traffic -- race
     * what is graded, including the STREAMS). */
    const int cap = (int)(1 + (size_t)32 * 1024 * 1024 / (VD * 8));
    int nv  = p->batch < cap ? p->batch : cap;
    int soa_on = 0, wv_c = -1;
    for (int c = 0; c < NC; ++c) {
        if (live[c] && !cd[c].cfn && !cd[c].wv) soa_on = 1;
        if (live[c] && cd[c].wv) wv_c = c;
    }
    if (soa_on && nv >= 8) nv &= ~7;
    void *ri = NULL, *ro = NULL, *rc = NULL, *r0 = NULL, *r1 = NULL;
    /* gen_r11: at re-home sizes tout/tcf ARE the plan's THP arena buffers
     * (STV/CV), so every trial and playoff runs in the cache+TLB regime the
     * graded chain now has (r10's fidelity lesson, applied forward), and
     * their 4K phases are gl_arena_take-deterministic instead of malloc
     * luck.  nv == the arena's nvr by construction (soa is never offered at
     * these sizes).  ri (trial input, read once per gate) stays heap. */
    const int ro_heap = !p->ch_live;
    if (!ro_heap) { ro = p->STV; rc = p->CV; }
    if (posix_memalign(&ri, 64, (size_t)nv * VD * sizeof(double)) ||
        (ro_heap &&
         (posix_memalign(&ro, 64, (size_t)nv * VD * sizeof(double)) ||
          posix_memalign(&rc, 64, (size_t)nv * VD * sizeof(double)))) ||
        posix_memalign(&r0, 64, VD * sizeof(double)) ||
        (nv > 1 && posix_memalign(&r1, 64, VD * sizeof(double)))) {
        free(ri);
        if (ro_heap) { free(ro); free(rc); }
        free(r0);
        return;                                       /* dense fallback */
    }
    double *tin = ri, *tout = ro, *tcf = rc, *ref0 = r0, *refN = r1;
    unsigned long long s = 0x9E3779B97F4A7C15ull;
    for (size_t i = 0; i < (size_t)nv * VD; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        tin[i] = (double)(long long)(s >> 11) * 0x1p-53;
    }
    for (size_t i = 0; i < (size_t)nv * VD; ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        tcf[i] = 0.1 * (double)(long long)(s >> 11) * 0x1p-53;
    }
    refnd(L, (const double _Complex *)tin, (double _Complex *)ref0);
    if (nv > 1)          /* gate the LAST volume too: M/P-reuse bugs cannot hide */
        refnd(L, (const double _Complex *)(tin + (size_t)(nv - 1) * VD),
              (double _Complex *)refN);

    for (int c = 0; c < NC; ++c) {
        if (!live[c] || (!cd[c].cfn && !cd[c].wv)) continue;
        memset(tout, 0, (size_t)nv * VD * sizeof(double));
        cd[c].fn(tin, tout, nv, p->M, p->P);
        if (!rel_ok(tout, ref0, VD)) live[c] = 0;
        if (nv > 1 && live[c] &&
            !rel_ok(tout + (size_t)(nv - 1) * VD, refN, VD)) live[c] = 0;
    }
    if (wv_c >= 0 && !live[wv_c]) wv_c = -1;

    /* soa gate: the whole m=2 chain on the first 8 volumes against an
     * INDEPENDENT scalar reference (refnd + the driver's exact map) -- this
     * exercises pack, DIF, the reversed-c map, DIT, the natural-c map and
     * unpack in one shot.  Lanes 1..7 catch pack/unpack lane bugs. */
    if (soa_on && nv >= 8) {
        void *w1 = NULL, *w2 = NULL;
        if (!posix_memalign(&w1, 64, (size_t)8 * VD * sizeof(double)) &&
            !posix_memalign(&w2, 64, (size_t)8 * VD * sizeof(double))) {
            double *z1 = w1, *z2 = w2;
            for (int v = 0; v < 8; ++v)
                refnd(L, (const double _Complex *)(tin + (size_t)v * VD),
                      (double _Complex *)(z1 + (size_t)v * VD));
            map_scalar(z1, tcf, z1, 8 * VD / 2);
            for (int v = 0; v < 8; ++v)
                refnd(L, (const double _Complex *)(z1 + (size_t)v * VD),
                      (double _Complex *)(z2 + (size_t)v * VD));
            map_scalar(z2, tcf, z2, 8 * VD / 2);
            soa_chain_n(p, (const double _Complex *)tin,
                        (const double _Complex *)tcf,
                        (double _Complex *)tout, 2, 8);
            for (int c = 0; c < NC; ++c)
                if (!cd[c].cfn && !rel_ok(tout, z2, 8 * VD)) live[c] = 0;
        } else {
            for (int c = 0; c < NC; ++c) if (!cd[c].cfn) live[c] = 0;
        }
        free(w1); free(w2);
    }

    /* gen_r12 wv gate: the whole m=2 chain COMPOSITION on volume 0 -- pack,
     * two full steps (zy sweeps + map-fused x-pass), unpack -- against the
     * refnd ground truth composed with a gated execute and the driver's
     * exact scalar map (the dm-pick gate's reference shape).  This
     * exercises pack_vol, both sweep pencil strides, the trans8 x-column
     * bracket, the lanex c pack, the fused map, and unpack in one shot.
     * It also leaves WS/WC packed for the trials below. */
    /* gen_r14 exec gate (runs FIRST -- the chain gate below repacks WS/WC
     * and leaves them for the trials): the single-shot wv execute, in the
     * mode install_pick would install, against the refnd ground truth on
     * volume 0 AND the last volume.  A failure kills the wv candidate
     * outright, so a banked l100-wv verdict always implies the exec path
     * passed on this host -- warm hits stay trustworthy and the driver's
     * two-process repeatability cmp stays pinned. */
    if (wv_c >= 0) {
        const char *m_ = getenv("GENPWP_WVEXEC");
        int wmode = m_ ? atoi(m_) : 2;      /* default = the raced arm      */
        if (wmode < 1 || wmode > 3) wmode = 2;
        memset(tout, 0, (size_t)nv * VD * sizeof(double));
        wv_exec_100(p->WS, tin, tout, nv, wmode);
        if (!rel_ok(tout, ref0, VD) ||
            (nv > 1 && !rel_ok(tout + (size_t)(nv - 1) * VD, refN, VD))) {
            live[wv_c] = 0;
            wv_c = -1;
        }
    }

    if (wv_c >= 0) {
        void *e1 = NULL, *e2 = NULL;
        if (!posix_memalign(&e1, 64, VD * sizeof(double)) &&
            !posix_memalign(&e2, 64, VD * sizeof(double))) {
            double *z1 = e1, *z2 = e2;
            map_scalar(ref0, tcf, z1, VD / 2);               /* x1 ref      */
            cd[wv_c].fn(z1, z2, 1, p->M, p->P);
            map_scalar(z2, tcf, z2, VD / 2);                 /* x2 ref      */
            pack_vol_100(tin, p->WS);
            pack_ct_100(tcf, p->WC);
            chainsteps_100(p->WS, p->WC, 2);
            unpack_vol_100(tout, p->WS);
            if (!rel_ok(tout, z2, VD)) { live[wv_c] = 0; wv_c = -1; }
        } else { live[wv_c] = 0; wv_c = -1; }
        free(e1); free(e2);
    }

    /* min over interleaved rounds; race the CHAIN step in its VOLUME-MAJOR
     * shape (gen_r4, ADOPTED from gen_pfa_large gen_r4's race fix, as their
     * fix to MY r2 tin-as-c bug came the other way): the graded chain now
     * runs all m steps on one volume before the next, so the trial working
     * set must be ONE volume's state + c slice, not the whole batch --
     * racing the old step-major shape would rank candidates on a working
     * set the chain no longer has (at resident sizes the fused families'
     * economics change).  Per volume: one unmeasured warm step, then R
     * timed steps IN PLACE on that tout volume with the DISTINCT tcf slice
     * as the c field (the graded steady state); R = 8/6/4 by volume bytes
     * <=2/<=8/>8 MiB; min over NR interleaved rounds.  tc[] is seconds per
     * VOLUME-step.  State values stay bounded across trials: every family
     * re-maps before the FFT (ipm/ipp) or right after it (all others).
     * The soa trial runs one DIF + one DIT step per group (2*nv
     * volume-steps), on the arenas the m=2 gate just packed -- the soa
     * engine was group-major from birth, so its shape is unchanged. */
    void (*sd_)(double *restrict, const double *restrict) =
        (L == 25) ? soa_step_dif_25 : soa_step_dif_27;
    void (*st_)(double *restrict, const double *restrict) =
        (L == 25) ? soa_step_dit_25 : soa_step_dit_27;
    memcpy(tout, tin, (size_t)nv * VD * sizeof(double));
    const size_t vbytes = VD * sizeof(double);
    const int R  = vbytes <= ((size_t)2 << 20) ? 8
                 : vbytes <= ((size_t)8 << 20) ? 6 : 4;
    const int NR = 4;
    const double nqtol = getenv("GENPWP_NQTOL")
                       ? atof(getenv("GENPWP_NQTOL")) : 0.05;
    double trc[NCMAX][NR + 6]; int nrr[NCMAX];
    for (int c = 0; c < NC; ++c) nrr[c] = 0;
    for (int round = 0; round < NR; ++round)
        for (int c = 0; c < NC; ++c) {
            if (!live[c]) continue;
            double t = pwp_trial_once(cd, c, p, tout, tcf, nv, R, VD,
                                      sd_, st_);
            trc[c][nrr[c]++] = t;
            if (t < tc[c]) tc[c] = t;
        }
    /* gen_r9 stability extension: the two LEADERS must have demonstrated
     * their floors (pwp_spread3 <= tol) before any verdict is trusted; a
     * noisy leader buys extra rounds for the leader pair only -- the
     * also-rans' exact times decide nothing.  Leaders are recomputed each
     * extra round (a settling floor can change who leads). */
    for (int extra = 0; extra < 6; ++extra) {
        int b1 = -1, b2 = -1;
        for (int c = 0; c < NC; ++c) {
            if (!live[c]) continue;
            if (b1 < 0 || tc[c] < tc[b1]) { b2 = b1; b1 = c; }
            else if (b2 < 0 || tc[c] < tc[b2]) b2 = c;
        }
        if (b1 < 0 || b2 < 0) break;
        if (pwp_spread3(trc[b1], nrr[b1]) <= nqtol &&
            pwp_spread3(trc[b2], nrr[b2]) <= nqtol) break;
        const int cs[2] = { b1, b2 };
        for (int i = 0; i < 2; ++i) {
            int c = cs[i];
            double t = pwp_trial_once(cd, c, p, tout, tcf, nv, R, VD,
                                      sd_, st_);
            trc[c][nrr[c]++] = t;
            if (t < tc[c]) tc[c] = t;
        }
    }
    /* gen_r6 PLAYOFF: soa vs the best interleaved candidate, long-run
     * head-to-head.  The interleaved trials evict the SoA arenas and i-cache
     * between rounds, and a 2-step soa trial re-pays that refill EVERY round
     * where the graded 200-256-step chain pays it once: the r5-board scoring
     * race mis-ranked soa at 27 (l27-ip0 installed at 58.6 us/vol, wisdom
     * margin -0.6% "tie"; soa's true graded number ~44.5 -- a 15% scored
     * regression from trial infidelity, my r5 busy-core lesson biting the
     * monitor's own window).  Re-time only the two finalists in alternation
     * with 24 steady steps per arm per round, min over 3 rounds, and feed
     * the numbers back into tc[]: min() keeps the interleaved side's
     * short-trial floor, so the playoff can only make the comparison fairer
     * to it, and soa still must clear the 3% simplest-first hysteresis.
     * gen_r9: both playoff flavours are ADAPTIVE -- rounds continue past 3
     * until each arm's floor is demonstrated (pwp_spread3 <= tol) or the
     * margin dwarfs the noise (margin >= 2Q, when more rounds cannot flip
     * the verdict), capped at 9; the final quality feeds the store gate. */
    double po_q = -1.0;             /* quality of the deciding playoff      */
    int po_w = -1, po_l = -1;       /* gen_r10: DECIDED interleaved-playoff
                                       winner/loser (-1 = no decided playoff) */
    {
        int csoa = -1, cbi = -1;
        for (int c = 0; c < NC; ++c) {
            if (!live[c] || cd[c].wv) continue;       /* wv: own playoff    */
            if (!cd[c].cfn) csoa = c;
            else if (cbi < 0 || tc[c] < tc[cbi]) cbi = c;
        }
        if (csoa >= 0 && cbi >= 0 && nv >= 8) {
            const int PP = 12;            /* SoA passes per arm (2 steps ea) */
            double psoa[9], pint[9]; int npr = 0;
            double tsoa = 1e300, tint = 1e300;
            for (int pr = 0; pr < 9; ++pr) {
                for (long g_ = 0; g_ < nv / 8; ++g_) {         /* 2 warm    */
                    sd_(p->SA, p->CR); st_(p->SA, p->CN);
                    sd_(p->SA, p->CR); st_(p->SA, p->CN);
                }
                double t0 = now_s();
                for (int r = 0; r < PP; ++r)
                    for (long g_ = 0; g_ < nv / 8; ++g_) {
                        sd_(p->SA, p->CR); st_(p->SA, p->CN);
                    }
                double t = (now_s() - t0) / ((double)PP * 2.0 * (double)nv);
                psoa[npr] = t;
                if (t < tsoa) tsoa = t;
                double tsum = 0.0;
                for (int b = 0; b < nv; ++b) {
                    double       *tv = tout + (size_t)b * VD;
                    const double *cv = tcf  + (size_t)b * VD;
                    cd[cbi].cfn(tv, tv, cv, 1, p->M, p->P);    /* 2 warm    */
                    cd[cbi].cfn(tv, tv, cv, 1, p->M, p->P);
                    double t1 = now_s();
                    for (int r = 0; r < 2 * PP; ++r)
                        cd[cbi].cfn(tv, tv, cv, 1, p->M, p->P);
                    tsum += now_s() - t1;
                }
                t = tsum / ((double)(2 * PP) * (double)nv);
                pint[npr++] = t;
                if (t < tint) tint = t;
                if (npr >= 3) {                       /* gen_r9 noise gate  */
                    double q = fmax(pwp_spread3(psoa, npr),
                                    pwp_spread3(pint, npr));
                    double mg = fabs(tsoa - tint) / fmin(tsoa, tint);
                    po_q = q;
                    if (q <= nqtol || mg >= 2.0 * q) break;
                }
            }
            tc[csoa] = tsoa;
            if (tint < tc[cbi]) tc[cbi] = tint;
            if (getenv("GENPWP_VERBOSE"))
                fprintf(stderr,
                        "gen_powp L=%d playoff: %s %.2f vs %s %.2f us/vol "
                        "(%d rounds, Q %.1f%%)\n",
                        L, cd[csoa].nm, tsoa * 1e6, cd[cbi].nm, tint * 1e6,
                        npr, po_q * 100.0);
        } else if (csoa < 0 && cbi >= 0) {
            /* rank-0 challenger playoff (same disease, interleaved case):
             * rank order encodes graded-shape held-lease evidence (ipk1 at
             * 100 wins 5/5 m=64 pairs) but a 4-step trial dilutes benefits
             * that accrue in steady state (ipk1's c-flush keeps L3 clean
             * for LATER steps' state re-reads) -- the same cold race read
             * ipk1 +5.3% and hysteresis could never reach it.  When the
             * lowest-rank live candidate is not the trial best but is
             * within 15%, re-judge the two on long steady runs. */
            int crk = -1;
            for (int c = 0; c < NC; ++c)
                if (live[c] && cd[c].cfn &&
                    (crk < 0 || cd[c].rank < cd[crk].rank)) crk = c;
            /* gen_r10: when rank 0 IS the trial best, the playoff runs
             * against the runner-up-by-trial instead of being skipped --
             * on a81n2 the contaminated short trials put ipk1 first in
             * 3/5 cold races and the mis-pick installed with no long-
             * horizon check at all.  The two LEADERS always get the
             * steady-state head-to-head; the decided verdict is
             * authoritative either way. */
            if (crk == cbi) {
                crk = -1;
                for (int c = 0; c < NC; ++c)
                    if (live[c] && cd[c].cfn && c != cbi &&
                        (crk < 0 || tc[c] < tc[crk])) crk = c;
            }
            if (crk >= 0 && crk != cbi && tc[crk] <= tc[cbi] * 1.15) {
                /* gen_r10 fidelity fix, measured on a81n2 at L=100: with 2
                 * warm steps and PS=12 this playoff read ipk1 AHEAD of ipp1
                 * (5151 vs 5363; 4801 vs 4805) in the same windows where
                 * held-lease GRADED m=64 pairs put ipp1 ahead 5/6 (floors
                 * 4538-4619 vs 4712-4780).  Mechanism: the arms alternate on
                 * the SAME tout/tcf, and a c-custody arm (ipk1 CLFLUSHOPTs
                 * the whole 16 MB c stream) hands its successor a c-cold
                 * hierarchy that 2 warm steps do not re-converge -- the
                 * measurement charged ipp1 for ipk1's flushes.  At volumes
                 * past L3 scale each arm now warms until ITS OWN cache
                 * regime is established (6 steps) and is timed on a graded-
                 * scale horizon (PS=24); small volumes keep the r6 shape. */
                const int big = VD * sizeof(double) > ((size_t)8 << 20);
                const int PS = 24;
                const int NW = big ? 6 : 2;
                double pa[9], pb[9]; int npr = 0;
                double ta = 1e300, tb = 1e300;
                for (int pr = 0; pr < 9; ++pr) {
                    for (int oi = 0; oi < 2; ++oi) {
                        const int cc = oi ? crk : cbi;
                        double tsum = 0.0;
                        for (int b = 0; b < nv; ++b) {
                            double       *tv = tout + (size_t)b * VD;
                            const double *cv = tcf  + (size_t)b * VD;
                            for (int w = 0; w < NW; ++w)
                                cd[cc].cfn(tv, tv, cv, 1, p->M, p->P);
                            double t1 = now_s();
                            for (int r = 0; r < PS; ++r)
                                cd[cc].cfn(tv, tv, cv, 1, p->M, p->P);
                            tsum += now_s() - t1;
                        }
                        double t = tsum / ((double)PS * (double)nv);
                        if (oi) { pb[npr] = t; if (t < tb) tb = t; }
                        else    { pa[npr] = t; if (t < ta) ta = t; }
                    }
                    ++npr;
                    if (npr >= 3) {                   /* gen_r9 noise gate  */
                        double q = fmax(pwp_spread3(pa, npr),
                                        pwp_spread3(pb, npr));
                        double mg = fabs(ta - tb) / fmin(ta, tb);
                        po_q = q;
                        if (q <= nqtol || mg >= 2.0 * q) {
                            /* gen_r10: a DECIDED playoff (noise-gated exit
                             * reached) records its winner/loser so the rank
                             * hysteresis cannot hand the slot back to the
                             * arm the long-horizon evidence just beat. */
                            po_w = ta <= tb ? cbi : crk;
                            po_l = ta <= tb ? crk : cbi;
                            break;
                        }
                    }
                }
                if (ta < tc[cbi]) tc[cbi] = ta;
                tc[crk] = tb;
                if (getenv("GENPWP_VERBOSE"))
                    fprintf(stderr,
                            "gen_powp L=%d playoff: %s %.2f vs %s %.2f "
                            "us/vol (%d rounds, Q %.1f%%)\n",
                            L, cd[crk].nm, tb * 1e6, cd[cbi].nm, ta * 1e6,
                            npr, po_q * 100.0);
            }
        }
    }
    /* gen_r12 WV PLAYOFF: the within-volume engine vs the best interleaved
     * candidate, long-run head-to-head AFTER the challenger playoff has
     * settled the interleaved slot -- the soa playoff's shape (short trials
     * under-read a whole-chain engine whose pack/refill amortizes over the
     * graded m=64) with the r10 cache-regime fidelity (6 warm steps for the
     * >L3 interleaved arm, each arm timed at a graded-scale horizon).
     * min() on the interleaved side keeps its short-trial floor: the
     * playoff can only make the comparison FAIRER to the incumbent, and wv
     * (ranked last) still must clear the 3% simplest-first hysteresis. */
    if (wv_c >= 0) {
        int cbi2 = -1;
        for (int c = 0; c < NC; ++c)
            if (live[c] && cd[c].cfn && (cbi2 < 0 || tc[c] < tc[cbi2]))
                cbi2 = c;
        if (cbi2 >= 0) {
            const int PS = 24;
            double pw[9], pn[9]; int npr = 0;
            double tw = 1e300, tn = 1e300;
            double qwv = -1.0;
            for (int pr = 0; pr < 9; ++pr) {
                for (int w = 0; w < 3; ++w) wv_step_100(p);   /* own regime */
                double t0 = now_s();
                for (int r = 0; r < PS; ++r) wv_step_100(p);
                double t = (now_s() - t0) / (double)PS;
                pw[npr] = t;
                if (t < tw) tw = t;
                double tsum = 0.0;
                for (int b = 0; b < nv; ++b) {
                    double       *tv = tout + (size_t)b * VD;
                    const double *cv = tcf  + (size_t)b * VD;
                    for (int w = 0; w < 6; ++w)
                        cd[cbi2].cfn(tv, tv, cv, 1, p->M, p->P);
                    double t1 = now_s();
                    for (int r = 0; r < PS; ++r)
                        cd[cbi2].cfn(tv, tv, cv, 1, p->M, p->P);
                    tsum += now_s() - t1;
                }
                t = tsum / ((double)PS * (double)nv);
                pn[npr++] = t;
                if (t < tn) tn = t;
                if (npr >= 3) {                       /* gen_r9 noise gate  */
                    double q = fmax(pwp_spread3(pw, npr),
                                    pwp_spread3(pn, npr));
                    double mg = fabs(tw - tn) / fmin(tw, tn);
                    qwv = q;
                    if (q <= nqtol || mg >= 2.0 * q) break;
                }
            }
            tc[wv_c] = tw;
            if (tn < tc[cbi2]) tc[cbi2] = tn;
            if (qwv >= 0.0) po_q = po_q < 0.0 ? qwv : fmax(po_q, qwv);
            if (getenv("GENPWP_VERBOSE"))
                fprintf(stderr,
                        "gen_powp L=%d wv playoff: %s %.2f vs %s %.2f us/vol "
                        "(%d rounds, Q %.1f%%)\n",
                        L, cd[wv_c].nm, tw * 1e6, cd[cbi2].nm, tn * 1e6,
                        npr, qwv * 100.0);
        }
    }
    int best = -1;
    for (int c = 0; c < NC; ++c)
        if (live[c] && (best < 0 || tc[c] < tc[best])) best = c;
    if (best >= 0) {
        int pick = best;                              /* 3% simplest-first */
        for (int c = 0; c < NC; ++c)
            if (live[c] && tc[c] <= tc[best] * 1.03 &&
                cd[c].rank < cd[pick].rank) pick = c;
        /* gen_r10: between the two arms of a DECIDED challenger playoff the
         * playoff verdict is AUTHORITATIVE -- the loser may not take the
         * slot back through the 3% rank hysteresis.  The wound is the r9
         * board's L=100 cell: on a81n2 the playoff put ipp1 ahead of rank-0
         * ipk1 by 1.6% (tight, banked) and the hysteresis replayed ipk1
         * anyway -- 4807.4 us shipped vs gen_pfa_large's ipp1 at 4521.7
         * (-6.3%).  Held-lease graded pairs on a81n2 confirm: ipp1 wins 5/6,
         * floors 4538-4619 vs ipk1's 4712-4780 (-2.2..-4.9%) -- the OPPOSITE
         * of a80n0's r6 5/5 evidence that set the rank.  Rank priors are
         * one host's history; a decided playoff is THIS host's long-horizon
         * measurement and outranks them (per-host truth is what the whole
         * race layer is for).  The soa family is untouched by this rule: its
         * playoff feeds tc[] and soa still must clear the 3% hysteresis (it
         * is the non-bit-identical candidate; the hurdle is deliberate). */
        if (po_w >= 0 && best == po_w && pick == po_l) pick = po_w;
        install_pick(p, cd, pick, 0);
        /* gate the picked interleaved chain step against execute + the
         * driver's own scalar map on volume 0 (tcf is the c field); on any
         * mismatch fall back to the execute+scalar path.  Deferred-map
         * (ipm) picks gate the full m=2 chain COMPOSITION (execute, one ipm
         * step, trailing map_span) -- a single ipm step's output is a raw
         * FFT volume, not a chain state.  The soa pick was already gated
         * above (m=2, independent reference). */
        int chain_ok = p->use_soa || p->use_wv;   /* both m=2-gated above */
        if (cd[pick].cfn && !getenv("GENPWP_NOFUSE")) {
            void *e1 = NULL, *e2 = NULL;
            if (!posix_memalign(&e1, 64, VD * sizeof(double)) &&
                !posix_memalign(&e2, 64, VD * sizeof(double))) {
                double *zed = e1, *exp_ = e2;
                if (cd[pick].dm) {
                    p->fn(tin, zed, 1, p->M, p->P);
                    map_scalar(zed, tcf, exp_, VD / 2);          /* x1      */
                    p->fn(exp_, zed, 1, p->M, p->P);
                    map_scalar(zed, tcf, exp_, VD / 2);          /* x2 ref  */
                    p->fn(tin, tout, 1, p->M, p->P);
                    cd[pick].cfn(tout, tout, tcf, 1, p->M, p->P);
                    map_span(tout, tcf, tout, VD);               /* x2 got  */
                } else {
                    p->fn(tin, zed, 1, p->M, p->P);
                    map_scalar(zed, tcf, exp_, VD / 2);
                    cd[pick].cfn(tin, tout, tcf, 1, p->M, p->P);
                }
                if (rel_ok(tout, exp_, VD)) {
                    p->cfn = cd[pick].cfn;
                    p->dm  = cd[pick].dm;
                    chain_ok = 1;
                }
            }
            free(e1); free(e2);
            snprintf(g_desc, sizeof g_desc,
                     "powp CT %s exact tw, two-sweep%s; pick: %s (B=%d)",
                     powp_fam(L),
                     p->cfn ? (p->dm ? " + owned chain (deferred NR map)"
                                     : " + owned chain (NR map)") : "",
                     cd[pick].nm, p->batch);
        }
        /* persist the decision (gen_race wisdom) -- only a pick whose chain
         * path passed its gate, so a warm hit can trust it.  gen_r9: the
         * verdict carries its quality.  Q = floor-stability of the DECIDING
         * measurement (the playoff arms when one ran, else the base trials
         * of pick and runner); tight iff Q <= tol or the pick's margin
         * dwarfs the noise.  Tight verdicts bank as a plain name; noisy
         * ones store provisional (name~q<pct>@<time>) so they pin the
         * repeatability pair but expire before the next scoring window. */
        if (chain_ok && !getenv("GEN_RACE_NO_WISDOM")) {
            double runner = -1.0; int ridx = -1;
            for (int c = 0; c < NC; ++c)
                if (live[c] && c != pick && (ridx < 0 || tc[c] < runner))
                    { runner = tc[c]; ridx = c; }
            double margin = runner > 0 ? (runner - tc[pick]) / tc[pick] : 0.0;
            double Q = po_q >= 0.0 ? po_q
                     : fmax(pwp_spread3(trc[pick], nrr[pick]),
                            ridx >= 0 ? pwp_spread3(trc[ridx], nrr[ridx])
                                      : 0.0);
            int tight = Q <= nqtol || (margin > 0 && margin >= 2.0 * Q);
            char snm[64];
            if (tight) snprintf(snm, sizeof snm, "%s", cd[pick].nm);
            else       snprintf(snm, sizeof snm, "%s~q%.0f@%ld",
                                cd[pick].nm, Q * 100.0, (long)time(NULL));
            gr_wisdom_store(fullkey, snm, pick,
                            margin < 0.03 ? 1 : 0,
                            tc[pick] * 1e6, margin);
            if (getenv("GENPWP_VERBOSE"))
                fprintf(stderr,
                        "gen_powp L=%d verdict: %s margin %+.1f%% Q %.1f%% "
                        "-> %s\n", L, cd[pick].nm, margin * 100.0, Q * 100.0,
                        tight ? "TIGHT (banked)" : "noisy (provisional)");
        }
        /* gen_r14: race the two execute arms (out-of-place tin -> tout, the
         * benchFFT shape, min-of-3 alternating rounds, interleaved first)
         * and bank the verdict under its own key -- see wv_exec_keyf. */
        if (p->use_wv && !wv_exec_lookup(p)) {
            /* trial-regime fidelity (the r10/r11 lesson): the graded
             * benchFFT execute writes the DRIVER's 4K-paged buffer, so the
             * race's out must be heap, not the huge-page arena tout. */
            void *eo = NULL;
            double *eout = tout;
            if (!posix_memalign(&eo, 4096, (size_t)nv * VD * sizeof(double)))
                eout = eo;
            double ta = 1e300, tb = 1e300;
            p->fn(tin, eout, nv, p->M, p->P);                 /* warm both  */
            wv_exec_100(p->WS, tin, eout, nv, 2);
            for (int r = 0; r < 3; ++r) {
                double t0 = now_s();
                p->fn(tin, eout, nv, p->M, p->P);
                double t = (now_s() - t0) / (double)nv;
                if (t < ta) ta = t;
                t0 = now_s();
                wv_exec_100(p->WS, tin, eout, nv, 2);
                t = (now_s() - t0) / (double)nv;
                if (t < tb) tb = t;
            }
            free(eo);
            p->exec_wv = (tb < ta * 0.97) ? 2 : 0;  /* 3% simplest-first    */
            if (!getenv("GEN_RACE_NO_WISDOM")) {
                char fk[GR_KEY_MAX + 16];
                wv_exec_keyf(fk, sizeof fk, p);
                gr_wisdom_store(fk, p->exec_wv ? "wv-nt" : "x-il",
                                p->exec_wv ? 1 : 0,
                                fabs(ta - tb) < 0.03 * fmin(ta, tb) ? 1 : 0,
                                fmin(ta, tb) * 1e6,
                                (ta - tb) / fmin(ta, tb));
            }
            if (getenv("GENPWP_VERBOSE"))
                fprintf(stderr,
                        "gen_powp L=%d exec race: x-il %.1f vs wv-nt %.1f us "
                        "-> %s\n", L, ta * 1e6, tb * 1e6,
                        p->exec_wv ? "wv-nt" : "x-il");
        }
    }
    if (getenv("GENPWP_VERBOSE")) {
        for (int c = 0; c < NC; ++c)
            fprintf(stderr, "gen_powp L=%d: %-9s %s %.1f us/vol\n",
                    L, cd[c].nm, live[c] ? "ok " : "OUT",
                    live[c] ? tc[c] * 1e6 : 0.0);
    }
    free(ri);
    if (ro_heap) { free(ro); free(rc); }              /* else: plan-owned    */
    free(r0); free(r1);
}
#endif /* __AVX512F__ */

fft3d_plan *fft3d_create(int L, int batch)
{
    if (!fft3d_supports(L) || batch < 1) return NULL;
    fft3d_plan *p = calloc(1, sizeof *p);
    if (!p) return NULL;
    p->L = L;
    p->batch = batch;
    const size_t VD = (size_t)2 * L * L * L;
    /* chain ping-pong state for the non-fused fallback path */
    if (posix_memalign(&p->rawX, 4096,
                       (size_t)batch * VD * sizeof(double)) != 0) {
        free(p);
        return NULL;
    }
    p->X = (double *)p->rawX;
#ifdef __AVX512F__
    /* SoA-8 lane arenas (gen_layout THP arena, staggered mod-4096 phases):
     * only for 25/27 at whole groups of 8 -- the roster's B >= 8 scope */
    if ((L == 25 || L == 27) && batch >= 8 && batch % 8 == 0) {
        const int PL = (L == 25) ? SOAPL25 : SOAPL27;
        const size_t bytes = (size_t)L * PL * 16 * sizeof(double);
        if (gl_arena_init(&p->soa_ar, 3 * (bytes + 8192) + (64 << 10)) == 0) {
            p->soa_live = 1;
            p->SA = (double *)gl_arena_take(&p->soa_ar, bytes + 4096);
            p->CN = (double *)gl_arena_take(&p->soa_ar, bytes + 4096);
            p->CR = (double *)gl_arena_take(&p->soa_ar, bytes + 4096);
            if (!p->SA || !p->CN || !p->CR) {
                gl_arena_destroy(&p->soa_ar);
                p->soa_live = 0;
                p->SA = p->CN = p->CR = NULL;
            }
        }
    }
    /* gen_r12: within-volume engine slabs at L=100 (ADOPTED from
     * gen_batchlane gen_r11) -- one volume of state (13 slabs) + the
     * x-consumption-order c, both on prefaulted 2 MiB pages (their engine
     * ran on gl_map_huge from birth; the r11 TLB verdict comes for free).
     * ~34 MB; any batch (the chain loops volumes through it). */
    if (L == 100) {
        const size_t sb = (size_t)13 * SLST100 * 16 * sizeof(double);
        const size_t cb = (size_t)100 * 13 * CST100 * 16 * sizeof(double);
        if (gl_arena_init(&p->wv_ar, sb + cb + 4 * 8192 + (64 << 10)) == 0) {
            p->WS = (double *)gl_arena_take(&p->wv_ar, sb + 4096);
            p->WC = (double *)gl_arena_take(&p->wv_ar, cb + 4096);
            if (p->WS && p->WC) p->wv_live = 1;
            else {
                gl_arena_destroy(&p->wv_ar);
                p->wv_live = 0;
                p->WS = p->WC = NULL;
            }
        }
    }
    powp_rt_tabs();                   /* gen_r3 runtime exact tables (~us)   */
    const int pp = (L == 25) ? 28 : (L == 27) ? 28 : (L == 50) ? 52
                 : (L == 49) ? 52 : (L == 81) ? 84 : (L == 121) ? 124
                 : (L == 125) ? 132 : 108;
    if (posix_memalign(&p->rawP, 4096,
                       (size_t)L * pp * 2 * sizeof(double)) == 0) {
        p->P = (double *)p->rawP;
        memset(p->P, 0, (size_t)L * pp * 2 * sizeof(double));
        if (posix_memalign(&p->rawM, 4096,
                           ((size_t)(2 * L * L + 8) * L + 296)
                               * sizeof(double)) == 0) {
            p->M = (double *)p->rawM + 296;       /* padded mid volume: +64 B
                                                     per plane, base 2368 B
                                                     off the page           */
            tune(p);
        }
    }
#endif
    if (!p->fn && !dense_setup(p)) {                  /* fallback of last resort */
        free(p->rawP); free(p->rawM); free(p->rawX); free(p->w); free(p->tmp);
        free(p);
        return NULL;
    }
    return p;
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
#ifdef __AVX512F__
    if (p->exec_wv) {                 /* gen_r14: the wv engine owns the
                                       * single-shot execute wherever the
                                       * banked chain verdict is wv          */
        wv_exec_100(p->WS, (const double *)in, (double *)out, p->batch,
                    p->exec_wv);
        return;
    }
#endif
    if (p->fn) p->fn((const double *)in, (double *)out, p->batch, p->M, p->P);
    else       dense_exec(p, in, out);
}

/* the whole graded m-step chain: state <- (FFT(state)+c)/(1+|FFT(state)+c|).
 *
 * Fused path: the state lives in `out` for the whole chain (the engine
 * consumes each input volume completely -- into the plane scratch, then into
 * the mid volume M -- before writing the matching output volume, so
 * cur == dst is safe per volume).  One state buffer instead of a ping-pong
 * pair keeps the chain working set at state+M+c volumes, which matters at
 * L=100 where that is already the size of the node's L3.
 *
 * Fallback path (no fused kernel): fft3d_execute needs distinct buffers, so
 * steps alternate between `out` and the plan's X, ending in `out`. */
void fft3d_chain(fft3d_plan *p, const double _Complex *x0,
                 const double _Complex *c, double _Complex *out, int m)
{
    const size_t VDv = (size_t)2 * p->L * p->L * p->L;   /* doubles/volume  */
    const size_t VD  = VDv * (size_t)p->batch;
    if (m < 1) { memmove(out, x0, VD * sizeof(double)); return; }
#ifdef __AVX512F__
    if (p->use_wv) {                  /* gen_r12 within-volume engine owns
                                       * the chain (volume-major: pack once,
                                       * all m steps in the THP slabs,
                                       * unpack once per volume)            */
        wv_chain_n(p, x0, c, out, m, p->batch);
        return;
    }
    if (p->use_soa) {                 /* SoA-8 lane engine owns the chain
                                       * (group-major from birth: all m
                                       * steps per 8-volume group)          */
        soa_chain_n(p, x0, c, out, m, p->batch);
        return;
    }
    /* gen_r4: VOLUME-MAJOR schedule for the interleaved families (ADOPTED
     * from gen_pfa_large gen_r4, who took it from gen_dense_prime /
     * gen_rader / gen_layout -- the corpus-consensus chain shape).  Volumes
     * are independent in the chain algebra, so each volume runs ALL m steps
     * while its state + c slice stays resident (0.5 MB at L=25 -- inside
     * L2 -- to 4 MB at L=50 B=4), instead of every step churning the whole
     * batch working set.  Per-volume FFT op order is unchanged; at 50/100
     * (VDv % 8 == 0) outputs are bit-identical to the r3 step-major
     * schedule, at 25/27 the per-volume map_span tail moves 1-3 complex
     * from the vector ladder to the exact scalar map (gate-checked as
     * always; the graded 25/27 picks are soa and unaffected). */
    /* gen_r11: THP re-home of the chain-hot streams (ADOPTED from
     * gen_layout gen_r11).  Gated per call on the MEASURED THP backing of
     * the driver's out buffer (gl_thp_bytes, ~100 us against a >= 100 ms
     * call): on THP=madvise hosts it reads 0%-backed and the steps run with
     * the state in the plan's 2 MiB-page STV volume and c staged once per
     * volume into CV; the LAST write of every volume goes to the driver's
     * out directly, so the state pays ZERO extra copies -- only its address
     * changes.  Values bit-identical (same arithmetic, same order).  On a
     * host/caller whose buffers are already huge-backed the gate turns the
     * whole mechanism off. */
    int reh = 0, nocv = 0;
    if (p->ch_live && p->cfn) {
        const size_t obytes = VD * sizeof(double);
        const long long hb = gl_thp_bytes(out, obytes);
        reh  = hb < (long long)(obytes / 2);   /* -1 (unreadable): re-home */
        nocv = getenv("GENPWP_NOCV") != NULL;
    }
    if (p->cfn && p->dm) {
        /* deferred-map schedule (gen_r3 ipm / gen_r4 ipp): between steps
         * the state volume holds the RAW FFT output z' = FFT3(x_s); the map
         * z/(1+|z|) with z = z' + c is applied by the NEXT step's phase 1
         * (at its loads for ipm, in a per-plane prepass for ipp), and once
         * by map_span after the last step.  m FFT passes + m map
         * applications, exactly the chain's algebra, zero separate map
         * passes in steady state. */
        for (int b = 0; b < p->batch; ++b) {
            const double *xb = (const double *)x0 + (size_t)b * VDv;
            const double *cb = (const double *)c  + (size_t)b * VDv;
            double       *ob = (double *)out      + (size_t)b * VDv;
            if (reh) {
                const double *cvp = cb;
                if (!nocv) { memcpy(p->CV, cb, VDv * sizeof(double));
                             cvp = p->CV; }
                p->fn(xb, p->STV, 1, p->M, p->P);
                for (int s = 1; s < m; ++s)
                    p->cfn(p->STV, p->STV, cvp, 1, p->M, p->P);
                map_span(p->STV, cvp, ob, VDv);       /* exit to out */
                continue;
            }
            p->fn(xb, ob, 1, p->M, p->P);
            for (int s = 1; s < m; ++s)
                p->cfn(ob, ob, cb, 1, p->M, p->P);
            map_span(ob, cb, ob, VDv);
        }
        return;
    }
#endif
    if (p->cfn) {
        for (int b = 0; b < p->batch; ++b) {
            const double *xb = (const double *)x0 + (size_t)b * VDv;
            const double *cb = (const double *)c  + (size_t)b * VDv;
            double       *ob = (double *)out      + (size_t)b * VDv;
#ifdef __AVX512F__
            if (reh) {
                const double *cvp = cb;
                if (!nocv) { memcpy(p->CV, cb, VDv * sizeof(double));
                             cvp = p->CV; }
                const double *cur = xb;
                for (int s = 0; s < m - 1; ++s) {
                    p->cfn(cur, p->STV, cvp, 1, p->M, p->P);
                    cur = p->STV;
                }
                p->cfn(cur, ob, cvp, 1, p->M, p->P);  /* last step to out */
                continue;
            }
#endif
            const double *cur = xb;
            for (int s = 0; s < m; ++s) {
                p->cfn(cur, ob, cb, 1, p->M, p->P);
                cur = ob;
            }
        }
        return;
    }
    const double _Complex *cur = x0;
    for (int s = 0; s < m; ++s) {
        double _Complex *dst = (((m - 1 - s) & 1) == 0) ? out
                             : (double _Complex *)p->X;
        fft3d_execute(p, cur, dst);
        map_scalar((const double *)dst, (const double *)c, (double *)dst,
                   VD / 2);
        cur = dst;
    }
}

void fft3d_destroy(fft3d_plan *p)
{
    if (!p) return;
    if (p->soa_live) gl_arena_destroy(&p->soa_ar);
    if (p->ch_live)  gl_arena_destroy(&p->ch_ar);
    if (p->wv_live)  gl_arena_destroy(&p->wv_ar);
    free(p->rawP);
    free(p->rawM);
    free(p->rawX);
    free(p->w);
    free(p->tmp);
    free(p);
}

#else /* ================ ENGINE TEMPLATE, one size (GENL) ================== */

#define NPL   (GENL * GENL)
#define PLNDL (2 * GENL * GENL)          /* doubles per x-plane              */
#define MPLND (PLNDL + 8)                /* mid volume plane pitch: +64 B ->
                                            an ODD cache-line count, no fixed
                                            mod-4096 relation to in/out      */
#define VDL   ((size_t)2 * GENL * NPL)   /* doubles per volume               */
#define NFULL (GENL / 4)                 /* full 4-lane groups per subpass   */
#define NYG   (NFULL + (GENL % 4 ? 1 : 0))
#define NTFL  (NPL / 4)                  /* full flat tiles in phase 2       */
#define FN(n) GCAT(n, GCAT(_, GENL))

/* pre-RA scheduling with register-pressure awareness helps the L=25 family
 * only (measured on the node: 25: -5%, 27: +2%, 100: +48%(!) when applied
 * globally) -- gen_batchlane's SCHED15 trick, same mechanism: the 5x5 CT
 * body holds 25+ live vectors and spills; pressure-aware scheduling cuts
 * the spill traffic where the codelet is pressure-bound. */
#if GENL == 25
#define PWPOPT __attribute__((optimize("schedule-insns", "sched-pressure")))
#else
#define PWPOPT
#endif

/* phase 1, ONE x-plane: z transform (lanes = 4 y-rows, granule transposes)
 * into plane scratch pl[y][kz], then y transform (lanes = 4 kz, contiguous)
 * into the mid volume M at the padded pitch.  GENL % 4 != 0 (25, 27, 50)
 * adds one OVERLAPPING group per subpass (recompute of 4 - GENL%4 lanes;
 * every store is idempotent).  gen_r4: the body is factored over (px, mx)
 * plane bases so the ipp family can feed it a mapped scratch plane
 * (gen_pfa_large gen_r4's shape). */
static inline __attribute__((always_inline))
void FN(p1body)(const double *restrict px, double *restrict mx,
                double *restrict pld)
{
    for (int yg = 0; yg < NYG; ++yg) {
        const int yb = (yg == NFULL) ? (GENL - 4) : 4 * yg;
        /* one runtime base per block (the L45 r7 single-base fix) */
        const double *rows = px  + (size_t)yb * (2 * GENL);
        double       *prow = pld + (size_t)yb * (2 * GPP);
        vec Zv[GENL], Wv[GENL];
        _Pragma("GCC unroll 25")
        for (int zg = 0; zg < NFULL; ++zg) {
            vec r_[4];
            _Pragma("GCC unroll 4")
            for (int j = 0; j < 4; ++j)
                r_[j] = LDU(rows + (size_t)j * (2 * GENL) + 8 * zg);
            TRNC(r_, &Zv[4 * zg]);
        }
#if GENL % 4
        {   /* overlapping last z-granule: columns GENL-4 .. GENL-1 */
            vec r_[4];
            _Pragma("GCC unroll 4")
            for (int j = 0; j < 4; ++j)
                r_[j] = LDU(rows + (size_t)j * (2 * GENL) + 2 * (GENL - 4));
            TRNC(r_, &Zv[GENL - 4]);
        }
#endif
#define LD1(n)    Zv[n]
#define ST1(k, v) (Wv[k] = (v))
        PFAL(LD1, ST1);
#undef LD1
#undef ST1
        _Pragma("GCC unroll 25")
        for (int zg = 0; zg < NFULL; ++zg) {
            vec r_[4];
            TRNC(&Wv[4 * zg], r_);
            _Pragma("GCC unroll 4")
            for (int j = 0; j < 4; ++j)
                STU(prow + (size_t)j * (2 * GPP) + 8 * zg, r_[j]);
        }
#if GENL % 4
        {   vec r_[4];
            TRNC(&Wv[GENL - 4], r_);
            _Pragma("GCC unroll 4")
            for (int j = 0; j < 4; ++j)
                STU(prow + (size_t)j * (2 * GPP) + 2 * (GENL - 4), r_[j]);
        }
#endif
    }

    for (int zg = 0; zg < NYG; ++zg) {
        const int zb = (zg == NFULL) ? (GENL - 4) : 4 * zg;
        const double *pcol = pld + 2 * zb;
        double       *mcol = mx  + 2 * zb;
        /* opaque-base barrier: stops gcc hoisting+spilling GENL row leas
         * (the L45 r6 offset-table pathology) */
        __asm__("" : "+r"(pcol), "+r"(mcol));
#define LD2(n)    LDU(pcol + (size_t)(n) * (2 * GPP))
#define ST2(k, v) STU(mcol + (size_t)(k) * (2 * GENL), (v))
        PFAL(LD2, ST2);
#undef LD2
#undef ST2
    }
}

static inline __attribute__((always_inline))
void FN(p1)(const double *restrict in, double *restrict mid,
            double *restrict pld, int x, const long mpln)
{
    FN(p1body)(in + (size_t)x * PLNDL, mid + (size_t)x * mpln, pld);
}

/* p1 with the graded map APPLIED TO THE Z-SUBPASS LOADS (gen_r3, the ipm
 * deferred-map family, ADOPTED from gen_pfa_large gen_r3): the state
 * entering this step is the PREVIOUS step's raw FFT output z' (no +c, no
 * map); every element is loaded here exactly once, so v = map(z' + c)
 * right after the granule load IS the chain map, in map_span's exact op
 * order (bit-identical results).  The ladder latency sits behind loads
 * feeding compute -- thousands of independent chains per plane the OoO
 * window runs ahead of -- NOT gating stores in the miss-bound x-pass (the
 * measured ipf failure at L=100).  Deletes the separate map pass entirely.
 * Overlap groups at GENL % 4 != 0 recompute the map on the repeated lanes:
 * pure function, idempotent. */
static inline __attribute__((always_inline))
void FN(p1m)(const double *restrict in, double *restrict mid,
             double *restrict pld, int x, const long mpln,
             const double *restrict cf)
{
    const double *px = in  + (size_t)x * PLNDL;
    const double *pc = cf  + (size_t)x * PLNDL;
    double       *mx = mid + (size_t)x * mpln;

    for (int yg = 0; yg < NYG; ++yg) {
        const int yb = (yg == NFULL) ? (GENL - 4) : 4 * yg;
        const double *rows = px  + (size_t)yb * (2 * GENL);
        const double *crow = pc  + (size_t)yb * (2 * GENL);
        double       *prow = pld + (size_t)yb * (2 * GPP);
        vec Zv[GENL], Wv[GENL];
        _Pragma("GCC unroll 25")
        for (int zg = 0; zg < NFULL; ++zg) {
            vec r_[4];
            _Pragma("GCC unroll 4")
            for (int j = 0; j < 4; ++j)
                r_[j] = map_step_v(LDU(rows + (size_t)j * (2 * GENL) + 8 * zg)
                                 + LDU(crow + (size_t)j * (2 * GENL) + 8 * zg));
            TRNC(r_, &Zv[4 * zg]);
        }
#if GENL % 4
        {   /* overlapping last z-granule: columns GENL-4 .. GENL-1 */
            vec r_[4];
            _Pragma("GCC unroll 4")
            for (int j = 0; j < 4; ++j)
                r_[j] = map_step_v(
                    LDU(rows + (size_t)j * (2 * GENL) + 2 * (GENL - 4))
                  + LDU(crow + (size_t)j * (2 * GENL) + 2 * (GENL - 4)));
            TRNC(r_, &Zv[GENL - 4]);
        }
#endif
#define LD1(n)    Zv[n]
#define ST1(k, v) (Wv[k] = (v))
        PFAL(LD1, ST1);
#undef LD1
#undef ST1
        _Pragma("GCC unroll 25")
        for (int zg = 0; zg < NFULL; ++zg) {
            vec r_[4];
            TRNC(&Wv[4 * zg], r_);
            _Pragma("GCC unroll 4")
            for (int j = 0; j < 4; ++j)
                STU(prow + (size_t)j * (2 * GPP) + 8 * zg, r_[j]);
        }
#if GENL % 4
        {   vec r_[4];
            TRNC(&Wv[GENL - 4], r_);
            _Pragma("GCC unroll 4")
            for (int j = 0; j < 4; ++j)
                STU(prow + (size_t)j * (2 * GPP) + 2 * (GENL - 4), r_[j]);
        }
#endif
    }

    for (int zg = 0; zg < NYG; ++zg) {
        const int zb = (zg == NFULL) ? (GENL - 4) : 4 * zg;
        const double *pcol = pld + 2 * zb;
        double       *mcol = mx  + 2 * zb;
        __asm__("" : "+r"(pcol), "+r"(mcol));
#define LD2(n)    LDU(pcol + (size_t)(n) * (2 * GPP))
#define ST2(k, v) STU(mcol + (size_t)(k) * (2 * GENL), (v))
        PFAL(LD2, ST2);
#undef LD2
#undef ST2
    }
}

/* phase 2: x transform OUT of place, mid (padded pitch) -> out (contract
 * layout), tiled over the FLAT (y,z) index.  NPL % 4 != 0 (25, 27) adds one
 * OVERLAPPING tail tile at flat base NPL-4 -- idempotent because the source
 * M is not written here.  pf=1 pokes the GENL read streams one cache line
 * ahead (more streams than the L2 prefetcher tracks). */
static inline __attribute__((always_inline))
void FN(p2o)(const double *restrict mid, double *restrict dst, const int pf)
{
#define LD3(n)    LDU(s_ + (size_t)(n) * MPLND)
#define ST3(k, v) STU(d_ + (size_t)(k) * PLNDL, (v))
    for (int t = 0; t < NTFL; ++t) {
        const double *s_ = mid + (size_t)t * 8;
        double       *d_ = dst + (size_t)t * 8;
        if (pf) {
            _Pragma("GCC unroll 100")
            for (int n_ = 0; n_ < GENL; ++n_)
                __builtin_prefetch(s_ + (size_t)n_ * MPLND + 8, 0, 3);
        }
        PFAL(LD3, ST3);
    }
#if NPL % 4
    {   const double *s_ = mid + (size_t)(NPL - 4) * 2;
        double       *d_ = dst + (size_t)(NPL - 4) * 2;
        PFAL(LD3, ST3);
    }
#endif
#undef LD3
#undef ST3
}

/* phase 2 IN PLACE (the ip* chain family and its execute).  The line
 * codelet reads all GENL inputs before its first store, so in-place is safe
 * per tile -- EXCEPT the overlapping tail at NPL % 4 != 0 (25, 27), whose
 * inputs partially coincide with the previous tile's outputs.  Fix: stash
 * the tail tile's GENL input vectors BEFORE any tile runs; the tail then
 * recomputes from pristine inputs and its stores repeat the overlap
 * columns' values exactly (idempotent). */
static inline __attribute__((always_inline))
void FN(p2ip)(double *io, const int pf)
{
#define LD3(n)    LDU(s_ + (size_t)(n) * PLNDL)
#define ST3(k, v) STU(s_ + (size_t)(k) * PLNDL, (v))
#if NPL % 4
    vec tl_[GENL];
    {   const double *s_ = io + (size_t)(NPL - 4) * 2;
        _Pragma("GCC unroll 100")
        for (int n_ = 0; n_ < GENL; ++n_)
            tl_[n_] = LD3(n_);
    }
#endif
    for (int t = 0; t < NTFL; ++t) {
        double *s_ = io + (size_t)t * 8;
        if (pf) {
            _Pragma("GCC unroll 100")
            for (int n_ = 0; n_ < GENL; ++n_)
                __builtin_prefetch(s_ + (size_t)n_ * PLNDL + 8, 1, 3);
        }
        PFAL(LD3, ST3);
    }
#if NPL % 4
    {   double *s_ = io + (size_t)(NPL - 4) * 2;
#define LDT(n) tl_[n]
        PFAL(LDT, ST3);
#undef LDT
    }
#endif
#undef LD3
#undef ST3
}

/* phase 2 IN PLACE with the chain map fused into the stores (NEW variant:
 * in-place working set AND no separate map pass).  Wins when the volume is
 * cache-resident (the map pass is pure extra L2 traffic); expected to lose
 * at L=100 where the fused map doubles the miss-stream count
 * (gen_pfa_large's measured lesson) -- the race decides. */
static inline __attribute__((always_inline))
void FN(p2ipf)(double *io, const double *restrict cf, const int pf)
{
#define LD3(n)    LDU(s_ + (size_t)(n) * PLNDL)
#define ST3(k, v) do {                                                       \
    vec z_ = (v) + LDU(g_ + (size_t)(k) * PLNDL);                            \
    STU(s_ + (size_t)(k) * PLNDL, map_step_v(z_));                           \
} while (0)
#if NPL % 4
    vec tl_[GENL];
    {   const double *s_ = io + (size_t)(NPL - 4) * 2;
        _Pragma("GCC unroll 100")
        for (int n_ = 0; n_ < GENL; ++n_)
            tl_[n_] = LD3(n_);
    }
#endif
    for (int t = 0; t < NTFL; ++t) {
        double       *s_ = io + (size_t)t * 8;
        const double *g_ = cf + (size_t)t * 8;
        if (pf) {
            _Pragma("GCC unroll 100")
            for (int n_ = 0; n_ < GENL; ++n_)
                __builtin_prefetch(s_ + (size_t)n_ * PLNDL + 8, 1, 3);
        }
        PFAL(LD3, ST3);
    }
#if NPL % 4
    {   double       *s_ = io + (size_t)(NPL - 4) * 2;
        const double *g_ = cf + (size_t)(NPL - 4) * 2;
#define LDT(n) tl_[n]
        PFAL(LDT, ST3);
#undef LDT
    }
#endif
#undef LD3
#undef ST3
}

/* phase 2 for the fused chain: x transform OUT of place, mid -> dst, with
 * z = v + c and the map z/(1+|z|) applied at every store -- deletes the
 * driver's separate full-volume map pass (read z + read c + write state).
 * Overlapping tail is idempotent for the same reason as p2o (map reads only
 * v and c, never dst).  pf pokes the mid read streams / dst write streams
 * (RFO) one line ahead; c and the tile bases advance contiguously. */
static inline __attribute__((always_inline))
void FN(p2c)(const double *restrict mid, double *restrict dst,
             const double *restrict cf, const int pfr, const int pfw)
{
#define LD3(n)    LDU(s_ + (size_t)(n) * MPLND)
#define ST3(k, v) do {                                                       \
    vec z_ = (v) + LDU(g_ + (size_t)(k) * PLNDL);                            \
    STU(d_ + (size_t)(k) * PLNDL, map_step_v(z_));                           \
} while (0)
    for (int t = 0; t < NTFL; ++t) {
        const double *s_ = mid + (size_t)t * 8;
        double       *d_ = dst + (size_t)t * 8;
        const double *g_ = cf  + (size_t)t * 8;
        if (pfr) {
            _Pragma("GCC unroll 100")
            for (int n_ = 0; n_ < GENL; ++n_)
                __builtin_prefetch(s_ + (size_t)n_ * MPLND + 8, 0, 3);
        }
        if (pfw) {
            _Pragma("GCC unroll 100")
            for (int n_ = 0; n_ < GENL; ++n_)
                __builtin_prefetch(d_ + (size_t)n_ * PLNDL + 8, 1, 3);
        }
        PFAL(LD3, ST3);
    }
#if NPL % 4
    {   const double *s_ = mid + (size_t)(NPL - 4) * 2;
        double       *d_ = dst + (size_t)(NPL - 4) * 2;
        const double *g_ = cf  + (size_t)(NPL - 4) * 2;
        PFAL(LD3, ST3);
    }
#endif
#undef LD3
#undef ST3
}

/* fused-through-M chain step for the whole batch: state cur -> state dst.
 * Per volume: phase 1 into the plan's single mid volume M (padded pitch),
 * then the fused map phase 2 into dst. */
static void PWPOPT FN(xc_pf0)(const double *cur, double *dst, const double *cf,
                       long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, M, P, x, MPLND);
        FN(p2c)(M, dst + (size_t)b * VDL, cf + (size_t)b * VDL, 0, 0);
    }
}

#ifndef GENLITE
static void PWPOPT FN(xc_pfr)(const double *cur, double *dst, const double *cf,
                       long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, M, P, x, MPLND);
        FN(p2c)(M, dst + (size_t)b * VDL, cf + (size_t)b * VDL, 1, 0);
    }
}

static void PWPOPT FN(xc_pfrw)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, M, P, x, MPLND);
        FN(p2c)(M, dst + (size_t)b * VDL, cf + (size_t)b * VDL, 1, 1);
    }
}
#endif /* !GENLITE */

/* in-place chain step (BORROWED from gen_pfa_large's ip* family): p1 in
 * place (each plane is fully consumed into the plane scratch before being
 * rewritten; on the first chain step cur != dst and p1 simply reads cur,
 * writes dst), p2 in place, then the sequential vectorized map in place --
 * the state buffer is the ONLY volume-sized object touched besides c. */
static void PWPOPT FN(xc_ip0)(const double *cur, double *dst, const double *cf,
                       long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, d, P, x, PLNDL);
        FN(p2ip)(d, 0);
        map_span(d, cf + (size_t)b * VDL, d, VDL);
    }
}

#ifndef GENLITE
static void PWPOPT FN(xc_ip1)(const double *cur, double *dst, const double *cf,
                       long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, d, P, x, PLNDL);
        FN(p2ip)(d, 1);
        map_span(d, cf + (size_t)b * VDL, d, VDL);
    }
}
#endif /* !GENLITE */

/* deferred-map chain step (gen_r3, ipm family, ADOPTED from gen_pfa_large
 * gen_r3): the incoming state is the previous step's RAW FFT output z';
 * p1m applies map(z' + c) at its loads, p2 runs plain, and NO map pass
 * follows -- the volume written is again a raw FFT output, mapped by the
 * NEXT step's p1m (or by fft3d_chain's one trailing map_span after the
 * last step).  In-place safe exactly as ip* (each plane fully consumed
 * into the scratch before being rewritten). */
static void PWPOPT FN(xc_ipm0)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1m)(i, d, P, x, PLNDL, g);
        FN(p2ip)(d, 0);
    }
}

#ifndef GENLITE
static void PWPOPT FN(xc_ipm1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1m)(i, d, P, x, PLNDL, g);
        FN(p2ip)(d, 1);
    }
}
#endif /* !GENLITE */

/* plane-prepass deferred-map chain step (gen_r4, the ipp family, ADOPTED
 * from gen_pfa_large gen_r4): the ipm schedule (state holds the raw FFT
 * output between steps; fft3d_chain runs step 1 plain and one trailing
 * map_span) but the map runs as map_span's perfectly sequential per-plane
 * PREPASS into the plan's M base -- an L2-resident scratch plane (10-250 KB
 * at these sizes) that p1's z-subpass then consumes.  Their measured
 * rationale at L=100: ipm and ipp have IDENTICAL per-step traffic
 * accounting, and ipm still lost 11-16% to ip -- so ipm's loss is the map
 * ladder's port/latency footprint INSIDE the granule-load stream, not
 * traffic; ipp keeps the traffic cut and pays the interference only at the
 * plane seam (~plane-sized granularity, invisible to the OoO window).
 * In-place safe: the prepass copies the plane out of the state before
 * p1body overwrites it. */
static void PWPOPT FN(xc_ipp0)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x) {
            map_span(i + (size_t)x * PLNDL, g + (size_t)x * PLNDL, M, PLNDL);
            FN(p1body)(M, d + (size_t)x * PLNDL, P);
        }
        FN(p2ip)(d, 0);
    }
}

#ifndef GENLITE
static void PWPOPT FN(xc_ipp1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x) {
            map_span(i + (size_t)x * PLNDL, g + (size_t)x * PLNDL, M, PLNDL);
            FN(p1body)(M, d + (size_t)x * PLNDL, P);
        }
        FN(p2ip)(d, 1);
    }
}
#endif /* !GENLITE */

/* ipp with the prepass c stream NTA-prefetched (gen_r5, the ipq family,
 * ADOPTED from gen_pfa_large gen_r5): c never enters L2/L3, so the LLC
 * holds the state volume across the whole step and p2's reads hit L3
 * instead of DRAM.  Wins only where state+c exceed L3 (100: 32 MB;
 * 121/125: 57/62 MB); at 50 B=4 the batch's c is L3-resident reuse across
 * all m steps and bypassing it costs a full c re-read per step -- ranked
 * last, the race arbitrates. */
static void PWPOPT FN(xc_ipq0)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x) {
            map_span_nta(i + (size_t)x * PLNDL, g + (size_t)x * PLNDL,
                         M, PLNDL);
            FN(p1body)(M, d + (size_t)x * PLNDL, P);
        }
        FN(p2ip)(d, 0);
    }
}

#ifndef GENLITE
static void PWPOPT FN(xc_ipq1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x) {
            map_span_nta(i + (size_t)x * PLNDL, g + (size_t)x * PLNDL,
                         M, PLNDL);
            FN(p1body)(M, d + (size_t)x * PLNDL, P);
        }
        FN(p2ip)(d, 1);
    }
}

/* ipp with the prepass c lines CLFLUSHOPT-ed after use (gen_r5, ipk):
 * the architectural-semantics variant of ipq's L3 bypass. */
static void PWPOPT FN(xc_ipk1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x) {
            map_span_cfl(i + (size_t)x * PLNDL, g + (size_t)x * PLNDL,
                         M, PLNDL);
            FN(p1body)(M, d + (size_t)x * PLNDL, P);
        }
        FN(p2ip)(d, 1);
    }
}

/* ip1 with the NTA-c map pass (gen_r5, iqn): the same c bypass for the
 * non-deferred schedule (hosts/sizes where ip beats ipp). */
static void PWPOPT FN(xc_iqn1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, d, P, x, PLNDL);
        FN(p2ip)(d, 1);
        map_span_nta(d, cf + (size_t)b * VDL, d, VDL);
    }
}
#endif /* !GENLITE */

/* in-place chain step with the map fused into phase 2's stores (NEW):
 * no mid volume, no separate map pass. */
static void PWPOPT FN(xc_ipf)(const double *cur, double *dst, const double *cf,
                       long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, d, P, x, PLNDL);
        FN(p2ipf)(d, cf + (size_t)b * VDL, 0);
    }
}

static void PWPOPT FN(x_pf0)(const double *in, double *out, long nvol,
                      double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDL;
        double       *o = out + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, M, P, x, MPLND);
        FN(p2o)(M, o, 0);
    }
}

#ifndef GENLITE
static void PWPOPT FN(x_pf1)(const double *in, double *out, long nvol,
                      double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDL;
        double       *o = out + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, M, P, x, MPLND);
        FN(p2o)(M, o, 1);
    }
}
#endif /* !GENLITE */

/* out-of-place execute via the in-place phase 2: p1 in -> out, p2 in place
 * in out (rides with the ip* chain candidates) */
static void PWPOPT FN(x_ip0)(const double *in, double *out, long nvol,
                      double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDL;
        double       *o = out + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, o, P, x, PLNDL);
        FN(p2ip)(o, 0);
    }
}

#ifndef GENLITE
static void PWPOPT FN(x_ip1)(const double *in, double *out, long nvol,
                      double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDL;
        double       *o = out + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, o, P, x, PLNDL);
        FN(p2ip)(o, 1);
    }
}
#endif /* !GENLITE */

#undef PWPOPT
#undef FN
#undef NTFL
#undef NYG
#undef NFULL
#undef VDL
#undef MPLND
#undef PLNDL
#undef NPL

#endif /* template */
