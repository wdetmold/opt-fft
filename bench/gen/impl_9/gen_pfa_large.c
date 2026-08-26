/* gen_pfa_large -- PFA of coprime pairs, large.
 * OWNED acceptance sizes 40, 50, 100 (plus 80 since r3); from ROUND gen_r6
 * the class pays its surprise-round duty in full: EVERY two-stage coprime
 * composite in 14..127 with modules in {2,3,4,5,7,8,9,11,13,16,25} and
 * L >= 44 is supported -- 44, 48, 52, 55, 56, 63, 65, 72, 75, 77, 88, 91,
 * 99, 104, 112, 117.
 *
 * ROUND gen_r9 (counter-directed round, avenue 1: BANK THE PICKS): the
 * tune() verdict is now NOISE-GATED before it can displace the rank prior
 * or enter the wisdom cache.  Per-round trial times are kept (tr[][]) and
 * their relative spread (max-min)/min is the noise measure:
 *   1. UPSET RULE: a trial winner may displace the rank-0 prior only when
 *      both spreads are tight (<= 10%), its margin over rank 0 exceeds
 *      max(the larger spread, 6%), AND it holds >= 6% again on one FRESH
 *      long-run alternation (chain_longrun) it has not seen -- otherwise
 *      the pick reverts to rank 0, whose order encodes the held-lease
 *      quiet-floor evidence (r5/r7 pairs).  Margins under 6% are window
 *      coin flips on every host measured (+-8-15% drift between
 *      minutes-apart reps vs 1-3% real gaps among the leading families);
 *      the genuine cross-host upsets the pool exists for (ipk1 on
 *      CLX/SPR, NT under contention) measured 9-16% and clear the bar.
 *      Consecutive cold create() cycles therefore pick identically; a
 *      churning window cannot flip the pick (r7's one-off ipk1 install).
 *   2. STORAGE RULE: only tight, non-reverted verdicts persist to wisdom;
 *      a noisy trial is installed for the current plan but NOT stored, so
 *      the next create() re-races (the audit's "re-race, never trust, a
 *      noisy trial").  Engine picks were already routed through gen_race
 *      wisdom since r3 -- this round closes the noisy-verdict hole the
 *      PMU audit identified as avenue 1 (gen_powp's L=25 25% swing).
 *   All families compute bit-identical results, so neither rule can affect
 *   correctness, gates, or cross-process repeatability; generated FFT and
 *   chain code is untouched (tune()-only change).  Determinism proven on
 *   the node: 5 consecutive cold create() cycles pick identically at
 *   40/50/100 (see strategy record).
 *
 * ROUND gen_r8 (the tools round): ATTRIBUTION DONE, CODE UNCHANGED BY
 * MEASUREMENT.  The four-rounds-queued "port 5 vs DRAM?" question about p1
 * was settled with the new static analyzers plus a leased-core rdtscp
 * microbench and per-phase counters on a dev copy (PMU stays locked):
 *   1. Machine model calibrated: zmm FMA 2.0/cycle (two 512-bit pipes,
 *      ports 0+5), zmm shuffle 1.0/cycle (port 5), mixed FMA+shuffle
 *      streams match the p0/p5 model exactly.  llvm-mca's ICX tables
 *      (512-bit FMA port0-only, 1/cycle) are WRONG on this SKU -- use it
 *      only for relative schedule questions, never absolute floors.
 *   2. Phase attribution (graded chains, forced picks): at L=100/ipp1 the
 *      step splits prepass 35% / p2 29% (both DRAM-bound) / zsub 22% /
 *      ysub 14%; at L=50/ip1: p1 ~52% / map ~29% / p2 ~19%.  The hot
 *      subpass loops all run at the node's ~2.1 vector-uops/cycle GLOBAL
 *      dispatch cap (zsub@100: 2971 uops / 1410 cyc = 2.11), far above
 *      the 2-FMA port floor -- neither port 5 nor DRAM is the p1 binder;
 *      TOTAL UOPS are.
 *   3. Frontend-diet experiment (rolled stage loops + CRT index tables,
 *      DSB-resident bodies, bit-identical algebra) BUILT AND REFUTED:
 *      hot-loop code shrank 17.8->3.7 KB (100) yet zsub/ysub cycles did
 *      not move (1410->1479 hot-window, ysub 898->952); at 50 the added
 *      index/staging uops cost +11% (434->483, 4/4 pairs), at 100 +3%
 *      (min 4592 vs 4730).  MITE 16B/c fetch is NOT the binder; the
 *      unrolled r7 code already minimizes total uops, which is the real
 *      currency under the dispatch cap.  Reverted; this header is the
 *      only r8 change, so all generated code is instruction-identical
 *      to r7.
 *   4. Load-TRNC-into-stage-1 fusion (the one remaining uop cut) closed
 *      by arithmetic: the GT stage-1 strides (25 at L=50/100) are not
 *      granule-aligned, so consuming transposed granules in registers
 *      still needs the full Zv staging.  The engine is saturated at the
 *      machine's uop-throughput cap with minimal uops.
 *
 * ROUND gen_r7 changes (on top of r6 below):
 *   1. CHALLENGER PLAYOFF in tune() (ADOPTED from gen_powp gen_r6): the
 *      R-step race trials systematically under-read candidates whose win
 *      accrues across steps (the ipk1 c-flush keeps L3 clean for LATER
 *      steps' state re-reads; a 4-step trial pays its flush cost and never
 *      collects).  When the rank-0 live candidate is not the trial best
 *      but within 15%, the two are re-judged on long steady runs (24
 *      steps/arm/round at <= 8 MiB volumes, 12 above; 3 rounds,
 *      alternating) and the results feed tc[] via min() -- fair to both
 *      arms; the 3% hysteresis still decides.  Portable by construction:
 *      the playoff runs wherever the race runs (CLX/SPR wisdom included).
 *   2. ipk1 rank-first at L=100 TRIED (gen_powp r6's 5/5 on their side)
 *      and REFUTED here: my held-lease pairs went 4/5 to ipp1 (min-of-mins
 *      4645 vs 4804, -3.3%), matching my r5 session.  ipp1 keeps rank 0;
 *      ipk1 moves to rank 2, ahead of plain ip* (trial-best in every
 *      interleaved race since r5; the CLX/SPR bypass insurance).  Wisdom
 *      tag chain5 -> chain6 so no stale verdict replays against the new
 *      race shape.
 *   3. ipk1 extended to the 16 LEAN coverage sizes (rank last): the
 *      L >= 91 volumes are 12-26 MB -- the same beyond-L3 c-pollution
 *      regime as L=100.  Pool signature changes give the lean sizes fresh
 *      wisdom keys by construction.
 *   4. Two-axes-per-pass fusion (lit 11 Tier 2, the queued backlog item)
 *      was worked through and CLOSED WITHOUT CODE: this engine's phase 1
 *      already transforms two axes (z, y) per DRAM pass through the
 *      L2-resident plane scratch; re-cutting the passes as z | y+x has
 *      bit-identical traffic (16r state + 16r c + 16w, then 16r + 16w at
 *      L=100), and no cut can remove the second pass because the x-stage
 *      DFT25 couples 25 planes.  The floor-changing lever for the 32 MB
 *      L3 spill is CUSTODY of the state across the step, which is exactly
 *      the ipk1 c-bypass (items 1-2).  Accounting in the strategy record.
 *
 * ROUND gen_r6 changes (on top of r5 below):
 *   1. SIXTEEN new class-coverage sizes (above), each a lean template
 *      instantiation of the same two-sweep GT-PFA engine: stage maps
 *      validated against numpy for all 16 (script in the strategy record),
 *      CRT constants A === 1 mod N1, 0 mod N2 / B === 1 mod N2, 0 mod N1.
 *      New modules: DFTODDM -- ONE generic direct symmetric odd-N DFT
 *      macro (fold x_j/x_{N-j} into a/b, X_k = R_k -+ i I_k) serving
 *      N = 3, 7, 9, 11, 13 with exact long-double cos/sin tables indexed
 *      by (j*k) % N (compile-time folded, the C25T trick); DFT8M -- the
 *      PFA40C stage-2 radix-2 DIT DFT8 refactored into a reusable
 *      fused-store macro (PFA40C itself untouched).  7/33/52/75/102
 *      FMA-port ops at N = 3/7/9/11/13.
 *   2. GEN_LEAN instantiation mode: coverage sizes emit only the families
 *      that actually win on this node (ip0/ip1/ip2/ipr1 + ipp0/ipp1) and
 *      compile the two heavy bodies (p1body, p2) ONCE as noinline
 *      functions: ~4 codelet expansions per size instead of ~50, so 16
 *      extra sizes stay compile-cheap.  The four scored sizes keep the
 *      full 18-20 candidate pool and always_inline bodies -- their
 *      generated code is unchanged from r5.
 *   3. Odd-L phase-2 TAIL (NPL % 4 == 1 for odd L, first odd-L sizes in
 *      this entry): the last 4 flat (y,z) columns are STASHED raw before
 *      the in-place sweep, full tiles cover columns 0..NPL-2, and one
 *      final tile transforms the stash into columns NPL-4..NPL-1 -- the 3
 *      recomputed columns read RAW stashed inputs, so the in-place
 *      recompute is idempotent (gen_powp's odd-L^2 stash idea).  Phase
 *      1's overlap groups already handled GENL % 4 in {1,2,3} since r1.
 *   4. tune()/create()/supports() are table-driven (g_sizes): per-size
 *      factors, scratch pitch (odd cache-line rule kept: GPP/4 odd), and
 *      candidate pool in one row; wisdom keys unchanged (chain5 + pool
 *      signature -- new sizes get their own keys by construction).
 *
 * ROUND gen_r5 changes (on top of r4 below):
 *   1. PAIR-PACKED map ladder (map_step_pair): the sequential map paths
 *      (map_vec, map_vec_rev, the ipp prepass) ran the ~18-op NR ladder on
 *      vectors where each |z|^2 is DUPLICATED in both complex lanes -- half
 *      the ladder lanes were redundant.  Two data vectors' 8 distinct
 *      |z|^2 values are now packed into ONE zmm (2 shuffles), one ladder
 *      runs for both, and the 8 reciprocals unpack pair-duplicated (2
 *      shuffles): ~21 arithmetic ops + 4 shuffles per PAIR vs 36 + 2
 *      before, a ~35% cut in map-pass uops.  Bit-identical outputs: every
 *      per-element operation, value and order, is unchanged (q_re + q_im
 *      is commutative; rsqrt14/rcp14/max are elementwise), so all chain
 *      families still agree bitwise and the r4 gate numbers stand.  The
 *      map is 46%/41% of total step ops at L=40/50 (both L3-resident,
 *      compute-bound) -- this is where the cut pays.
 *   2. NEW chain families ipq1 / ipk1 / iqn1 (c-stream L3-bypass, aimed at
 *      L=100 where state+c = 32 MB streams through the 24 MB non-inclusive
 *      L3 every step and evicts the state before p2 re-reads it).  c is
 *      read exactly once per step and its only reuse is ACROSS steps --
 *      caching it is pure pollution whenever state+c exceeds L3.  ipq1 =
 *      ipp1 with the prepass prefetching c via PREFETCHNTA (SKX+ NTA lines
 *      fill L1 only and skip L2/LLC on eviction); ipk1 = ipp1 with the
 *      prepass CLFLUSHOPT-ing c lines one iteration after use (guaranteed
 *      semantics where NTA policy is implementation-defined); iqn1 = ip1
 *      with the NTA-c map pass (the same idea for the non-deferred
 *      schedule).  If the bypass holds, L3 keeps the 16 MB state volume
 *      across the whole step: p2's 16 MB DRAM read and most state
 *      writebacks disappear (~80 -> ~48-64 MB/step).  All three compute
 *      bit-identical values (prefetch hints and flushes move no data);
 *      raced at every size, ranked last -- at 40/50 c IS the reuse set
 *      (L3-resident across all m steps) and they should and do lose.
 *      pf ids 17/18/19.  Wisdom tag chain4 -> chain5.
 *
 * ROUND gen_r4 changes (on top of r3 below):
 *   1. VOLUME-MAJOR chain schedule (ADOPTED from the corpus consensus that
 *      gen_dense_prime / gen_rader / gen_layout all ship and this entry never
 *      did): volumes are independent in the chain algebra, so fft3d_chain now
 *      runs ALL m steps on one volume before touching the next, instead of
 *      one step across all volumes.  At L=40 B=8 / L=50 B=4 the per-step
 *      working set drops from state+c for the whole batch (16 MB, at the
 *      shared-L3 mercy of the neighbours) to ONE volume's state+c slice
 *      (2-4 MB, L2/L3-resident across all 128 steps).  Identical per-volume
 *      op order -- outputs bit-identical to the step-major schedule.  L=100
 *      is B=1 and unchanged by this.
 *   2. NEW chain family ipp* (plane-granularity deferred map): same deferred
 *      schedule as ipm (state holds raw FFT output z' between steps; step 1
 *      plain execute, one trailing map_vec) but the map z/(1+|z|) runs as a
 *      sequential per-plane PREPASS into an L2-resident scratch plane (the
 *      mid volume M's base) which p1's z-subpass then consumes.  Deletes the
 *      separate map pass's full-volume DRAM round trip exactly like ipm
 *      (~112 -> ~80 MB/step at L=100) WITHOUT ipm's measured failure mode:
 *      the ladder's uops interleave with p1's TRNC/codelet only at the plane
 *      seam (~50k-uop granularity, invisible to the OoO window) instead of
 *      per-granule, so port 5 sees the map and the transform sequentially,
 *      each at its own bottleneck.  Bit-identical (map_vec's exact op order).
 *      Raced as ipp0/ipp1 at every size; pf ids 15/16.
 *   3. The create() race now times the VOLUME-MAJOR schedule (one warm step
 *      per volume, then R timed steps on that volume, volumes outer) --
 *      racing the step-major shape would rank candidates on a working set
 *      the graded chain no longer has.  Wisdom tag bumped chain3 -> chain4
 *      so stale step-major verdicts can never be installed.
 *
 * ROUND gen_r3 changes (on top of r2 below):
 *   1. NEW chain family ipm* (map-on-NEXT-step's-loads, deferred map): the
 *      graded map z/(1+|z|) is NOT run as its own pass and NOT fused into
 *      any store; it is applied to the NEXT step's phase-1 z-subpass LOADS
 *      (v = map(state + c) right after the 4x4 granule load, before TRNC).
 *      Every element is loaded there exactly once, so the algebra is the
 *      chain's, in the ip* families' exact op order (outputs bit-identical).
 *      Deletes the separate map pass's full-volume read + RFO + write per
 *      step (32 MB of the ~112 MB step at L=100, which is DRAM-bound); c
 *      moves from the map pass into p1 at the same total read volume.  The
 *      ladder latency sits behind LOADS feeding compute (thousands of
 *      independent chains per plane the OoO window can run ahead of), NOT
 *      gating stores in the miss-bound x-pass -- the measured ipf/ipe
 *      failure mode at L=100 (r2: +75% / +27%).  fft3d_chain runs the
 *      deferred-map schedule: step 1 = plain execute, steps 2..m = p1m+p2,
 *      one trailing map_vec.  Raced as ipm0/ipm1 at every size.
 *   2. gen_race ADOPTED (their r2 offer, this record's #1 deferred item):
 *      the tune() verdict (winner name + chain-gate verdict) is persisted
 *      per (host, L, B-bucket, candidate-set signature) via
 *      gr_wisdom_get_str/put_str.  Warm create() = one wisdom read, no
 *      arena, no refnd, no race: meets the brief's 50 ms budget (cold was
 *      0.4-6.5 s).  Safe by construction: all families compute bit-identical
 *      results, and only gate-passed verdicts are ever stored.  Forcing
 *      (GENPFL_PF / GENPFL_NOFUSE) bypasses wisdom both ways; GEN_RACE_*
 *      env pins are honored by the layer.
 *   3. NEW size L=80 = 16x5 (round-3 any-size-in-class duty, the clean
 *      no-tail candidate: 80%4==0, 6400%4==0): PFA80C = 16 x DFT5 into
 *      T_[16*k2+n1], then 5 x DFT16 storing straight through ST via
 *      k = (65*k1 + 16*k2) % 80.  DFT16 = 4x4 CT, exact W16 twiddles (8
 *      CMULC + one -i), 81 FMA-port ops.  661 ops/line, 3.17M ops/volume.
 *      pl pitch 84 complex = 21 lines (odd).  Unscored; exists for the
 *      round-6 assembled library.
 *
 * ROUND gen_r2 changes (on top of the r1 design described below):
 *   1. DFT25 stage-B stores fused straight through the PFA wrappers' ST
 *      (macro DFT25M replaces the dft25v function): deletes the R_[25]
 *      L1 round-trip per DFT25 call -- 50 vector L1 accesses per line at
 *      L=50, 200 at L=100 -- and cuts peak live vectors in stage B (the
 *      old shape kept U[25] + R_[25] in flight; now each stage-B DFT5's
 *      five outputs go straight to memory).  gen_powp's r1 record queued
 *      the same item.
 *   2. NEW chain family ipf* (BORROWED from gen_powp gen_r1, who built it
 *      on this entry's engine): in place AND the map fused into phase 2's
 *      stores -- no mid volume, no separate map pass.  Their node race
 *      picked it at L=50 (473 us vs this entry's 481 with ip*): it
 *      deletes the map pass's read-state + read-c + write-state traffic
 *      while keeping stores on the lines phase 2 just read (no fresh RFO
 *      stream, unlike the out-of-place fused f*).
 *   3. NEW chain families ipnt and ipfnt (GENL % 4 == 0 sizes only): phase
 *      1's y-subpass stores through vmovntpd -- at L=40/100 the y-pass
 *      writes every output line in full at 64B alignment, so NT deletes
 *      the RFO read of the state volume; one sfence per volume before
 *      phase 2.  Aimed at L=100, which is DRAM-bound (~112 MB traffic
 *      per chain step at ~21 GB/s accounts for the whole 5.25 ms; the
 *      ipfnt accounting is ~64 MB).  64B base alignment is checked at
 *      runtime, falling back to the ordinary in-place body.
 *   All candidates remain gated + raced per (size, host) in create();
 *   every family computes bit-identical results (same op order, same NR
 *   map ladder), so a per-host pick flip cannot break repeatability.
 *
 * ROUND gen_r1 (first real round; the previous file was the dense O(L^4)
 * validation stub).  Technique, seeded directly from the fixed-size winner
 * L45_pfa (panel_r11) and generalized to this class's three sizes:
 *
 *   Row-column 3D DFT, two sweeps per volume (the structure that won L=36
 *   and L=45 on the node):
 *     phase 1, per x-plane:
 *       z transform: lanes = 4 y-rows, 4x4 complex-granule register
 *                    transposes on load and store, into plane scratch
 *                    pl[y][kz] (row pitch an ODD number of cache lines,
 *                    the L23_rader / corpus anti-aliasing rule);
 *       y transform: lanes = 4 kz (contiguous in pl), store to out[x][ky][kz].
 *     phase 2:
 *       x transform in place in `out`, lanes = 4 kz, tiled over the FLAT
 *       (y,z) index -- L*L is divisible by 4 for all three sizes, so there
 *       is NO tail here.  The codelet reads all L inputs before its first
 *       store, so in-place is safe.
 *
 *   Every L-point line is a Good-Thomas prime-factor codelet on interleaved
 *   complex vectors (lanes = a spectator axis), maps folded at compile time:
 *     L =  40 = 8 x 5:   n = (5 n1 + 8 n2) % 40,  k = (25 k1 + 16 k2) % 40
 *                        stage 1: 8 x DFT5 (FFTW n1_5 FMA form, 16 ops),
 *                        stage 2: 5 x DFT8 (radix-2 DIT, 26 ops + 6 swaps)
 *                        storing straight through ST (the r11 stage-order
 *                        lesson: short-live-range module first, long module
 *                        reads contiguous hot slots and stores directly).
 *                        278 FMA-port ops / line.
 *     L =  50 = 25 x 2:  n = (2 n1 + 25 n2) % 50,  k = (26 k1 + 25 k2) % 50
 *                        stage 1: 25 x DFT2, stage 2: 2 x DFT25.
 *                        434 FMA-port ops / line.
 *     L = 100 = 25 x 4:  n = (4 n1 + 25 n2) % 100, k = (76 k1 + 25 k2) % 100
 *                        stage 1: 25 x DFT4 (8 ops + 1 swap),
 *                        stage 2: 4 x DFT25.  968 FMA-port ops / line.
 *
 *   DFT25 is where this class meets the campaign's twiddle problem: 25 is a
 *   prime power, so it is a 5x5 Cooley-Tukey INSIDE the PFA --
 *   X[k1+5k2] = DFT5_{n2}( W25^{n2 k1} * DFT5_m( x[n2+5m] )[k1] ), 16
 *   nontrivial twiddles per call, each 2 FMA-port ops + 1 swap.  Twiddles
 *   are compile-time literals computed in long double (cosl/sinl of
 *   2*pi*j/25, ~19 correct digits, exact-to-0.5ulp doubles), laid out in a
 *   16-entry-indexed const table the unrolled loops fold.
 *   192 FMA-port ops + 36 swaps per DFT25.
 *
 *   Tails: 40 and 100 are multiples of 4 -- NO tails anywhere.  50 = 12*4+2:
 *   phase 1's z and y subpasses run 12 full groups plus ONE overlapping
 *   group at yb/zb = 46 (recomputes 2 lanes; idempotent because every store
 *   site writes values that do not depend on prior contents of the target).
 *
 *   pl row pitches (complex): 44 (40), 52 (50), 108 (100) = 11/13/27 cache
 *   lines, all odd and 64B-aligned rows.
 *
 * ATTRIBUTION (this file is deliberately cumulative):
 *   - Two-sweep plane-fused structure, spectator lanes, TRNC granule
 *     transpose, opaque-base asm barrier in the y-subloop, heap (not stack)
 *     plane scratch, create()-time scalar-reference gate + interleaved
 *     min-of-rounds race with simplest-first hysteresis: L45_pfa (seed).
 *   - DFT5 = FFTW n1_5 FMA form, verbatim from L45_pfa's PFA45 stage 1.
 *   - Store-direct stage order (short module first): L45_pfa r11 /
 *     L45_mixedradix ST1G/ST2G.
 *   - Odd-cache-line scratch pitch: L23_rader r6/r7 via L45_pfa.
 *   - Flat phase-2 tiling: L45_mixedradix r7 via L45_pfa.
 *
 * OPERATION COUNT (vector FMA-port ops per volume, lanes of 4):
 *   L=40:  3 * 400 * 278  =   333,600     (plus 2 granule transposes/elt)
 *   L=50:  3 * 625 * 434  =   813,750 + overlap recompute ~8%
 *   L=100: 3 * 2500 * 968 = 7,260,000
 *
 * OWNED CHAIN (fft3d_chain, strong symbol): the graded workload is a chain
 * z = FFT3(x) + c; x <- z/(1+|z|), and the driver's fallback pays a separate
 * full-volume map pass (read z + read c + write state) plus an initial
 * memcpy per unit.  This entry owns the chain, and create() RACES two chain
 * step families (the winner differs by size -- both are kept):
 *   ip*: everything in place.  p1 is in-place safe per plane (each plane is
 *        fully consumed into the plane scratch before being rewritten), p2
 *        is in-place safe per line, then a SEQUENTIAL vectorized map pass
 *        in place.  The state buffer is the only volume-sized object
 *        touched besides c, so it stays cache-resident: wins at L=100
 *        (5.25 ms vs 8.24 unfused vs 12.0 for the fused variant, whose
 *        map-in-the-x-pass doubles the miss-stream count to 100 reads +
 *        100 RFO writes with one fresh line each per tile) and at 50 and
 *        40 too on the node.
 *   f*:  map fused into phase 2's stores, M (padded planes: MPLND odd in
 *        cache lines, base 2368 B off the page) -> state out of place.
 *        Kept in the pool: on cache-resident cases on other hosts the
 *        deleted map pass can win (it did on wallaby-local small batches
 *        before the ip variants existed).
 * |z| and 1/(1+|z|) via rsqrt14/rcp14 + two Newton steps each (~1e-16 rel,
 * inside the 1.5e-14/step contract) -- zmm vsqrtpd/vdivpd are not pipelined
 * and would cost ~2x the map's FMA count.  The chain state stays in `out`
 * the whole time (cur == dst is safe as above): no ping-pong buffer.
 * Measured on the node (graded chain m, min over samples):
 *   L=40  B=8: 354 -> 207 us/xform   B=1: 377 -> 236
 *   L=50  B=4: 798 -> 497            B=1: 757 -> 558
 *   L=100 B=1: 8244 -> 5248          (raw execute, no chain: 4563)
 * create() gates the picked chain step against execute + the driver's
 * scalar map and silently falls back to that exact path if it disagrees.
 *
 * ACCURACY: gate at create() vs an independent scalar O(L^2)-per-line
 * reference at 1e-13 rel L2 (first AND last arena volume).  Measured via
 * check.py: ~1e-15 rel L2 vs numpy at all three sizes.
 *
 * Falls back to the dense O(L^4) matrix path if AVX-512 is unavailable or
 * (never observed) a candidate fails the gate.
 */
#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

#include "../fft3d_api.h"

#ifndef GEN_PFA_LARGE_ONCE          /* ============ COMMON, first pass ===== */
#define GEN_PFA_LARGE_ONCE

/* gen_race library (r3 adoption): wisdom cache for the tune() verdict */
#define GEN_RACE_LIB_ONLY
#include "gen_race.c"

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
#define STNT(p, v)  _mm512_stream_pd((double *)(p), (__m512d)(v))  /* 64B-aligned */
#define VSPLAT(a)   ((vec){(a),(a),(a),(a),(a),(a),(a),(a)})
#define VPAIR(a,b)  ((vec){(a),(b),(a),(b),(a),(b),(a),(b)})
#define SWAP(v)     VSH((v),(v), 1,0,3,2,5,4,7,6)
#define VFMA(a,b,c)  ((vec)_mm512_fmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))
#define VFNMA(a,b,c) ((vec)_mm512_fnmadd_pd((__m512d)(a),(__m512d)(b),(__m512d)(c)))

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
/* 5-point (FFTW n1_5 FMA form, from L45_pfa) */
#define K59  0.55901699437494742410229341718282   /* sqrt(5)/4               */
#define KIG  0.61803398874989484820458683436564   /* sin(4pi/5)/sin(2pi/5)   */
#define KS5  0.95105651629515357211665325776975   /* sin(2pi/5)              */
/* 8-point */
#define KR2  0.70710678118654752440084436210485   /* sqrt(1/2)               */
/* 16-point (r3, for L=80) */
#define KC16 0.92387953251128675612818318939679   /* cos(pi/8)               */
#define KS16 0.38268343236508977172845998403040   /* sin(pi/8)               */

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

/* r6: cos/sin(2 pi m / N) tables for the DFTODDM modules (N = 3, 7, 9, 11,
 * 13), long-double literals (~19 correct digits, exact-to-0.5ulp doubles),
 * FULL index range m = 0..N-1 so any (j*k) % N lands on a valid entry
 * (index 0 occurs, e.g. j = k = 3 at N = 9: cos 1, sin 0, exact). */
static const double C3T[3] = {
    1.0, -4.9999999999999999997e-01, -5.0000000000000000005e-01 };
static const double S3T[3] = {
    0.0,  8.6602540378443864679e-01, -8.6602540378443864673e-01 };
static const double C7T[7] = {
    1.0,
    6.2348980185873353053e-01, -2.2252093395631440434e-01,
   -9.0096886790241912632e-01, -9.0096886790241912616e-01,
   -2.2252093395631440446e-01,  6.2348980185873353085e-01 };
static const double S7T[7] = {
    0.0,
    7.8183148246802980873e-01,  9.7492791218182360698e-01,
    4.3388373911755812029e-01, -4.3388373911755812059e-01,
   -9.7492791218182360698e-01, -7.8183148246802980846e-01 };
static const double C9T[9] = {
    1.0,
    7.6604444311897803519e-01,  1.7364817766693034880e-01,
   -5.0000000000000000016e-01, -9.3969262078590838410e-01,
   -9.3969262078590838405e-01, -4.9999999999999999970e-01,
    1.7364817766693034870e-01,  7.6604444311897803530e-01 };
static const double S9T[9] = {
    0.0,
    6.4278760968653932633e-01,  9.8480775301220805940e-01,
    8.6602540378443864668e-01,  3.4202014332566873297e-01,
   -3.4202014332566873305e-01, -8.6602540378443864695e-01,
   -9.8480775301220805940e-01, -6.4278760968653932616e-01 };
static const double C11T[11] = {
    1.0,
    8.4125353283118116886e-01,  4.1541501300188642549e-01,
   -1.4231483827328514046e-01, -6.5486073394528506413e-01,
   -9.5949297361449738990e-01, -9.5949297361449738990e-01,
   -6.5486073394528506407e-01, -1.4231483827328514024e-01,
    4.1541501300188642568e-01,  8.4125353283118116892e-01 };
static const double S11T[11] = {
    0.0,
    5.4064081745559758210e-01,  9.0963199535451837142e-01,
    9.8982144188093273236e-01,  7.5574957435425828370e-01,
    2.8173255684142969765e-01, -2.8173255684142969773e-01,
   -7.5574957435425828376e-01, -9.8982144188093273241e-01,
   -9.0963199535451837136e-01, -5.4064081745559758199e-01 };
static const double C13T[13] = {
    1.0,
    8.8545602565320989587e-01,  5.6806474673115580249e-01,
    1.2053668025532305325e-01, -3.5460488704253562600e-01,
   -7.4851074817110109868e-01, -9.7094181742605202719e-01,
   -9.7094181742605202719e-01, -7.4851074817110109863e-01,
   -3.5460488704253562600e-01,  1.2053668025532305346e-01,
    5.6806474673115580238e-01,  8.8545602565320989609e-01 };
static const double S13T[13] = {
    0.0,
    4.6472317204376854566e-01,  8.2298386589365639458e-01,
    9.9270887409805399284e-01,  9.3501624268541482345e-01,
    6.6312265824079520232e-01,  2.3931566428755776695e-01,
   -2.3931566428755776706e-01, -6.6312265824079520243e-01,
   -9.3501624268541482345e-01, -9.9270887409805399278e-01,
   -8.2298386589365639469e-01, -4.6472317204376854531e-01 };

/* v * (C - iS): 2 FMA-port ops + 1 swap */
#define CMULC(v, C, S) VFMA(SWAP(v), VPAIR((S), -(S)), (v) * VSPLAT(C))
/* v * (-i): 1 mul + 1 swap (r3) */
#define MULNI(v) (SWAP(v) * VPAIR(1.0, -1.0))

/* One step of the graded chain map on a vector of 4 complex:
 *     z -> z / (1 + |z|)
 * |z| and the reciprocal via rsqrt14/rcp14 + TWO Newton steps each (final
 * relative error ~1e-16, comfortably inside the 1.5e-14/step contract),
 * instead of vsqrtpd+vdivpd whose zmm throughput is not pipelined.  ~18
 * FMA-port ops per vector.  The max() guard keeps z = 0 exact (rsqrt(0)
 * would make 0 * inf = NaN); 1e-300 shifts the result by ~1e-150. */
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

/* One map step for TWO vectors at once (r5).  map_step_v runs the NR
 * ladder on a vector where each |z|^2 sits duplicated in both complex
 * lanes -- half its ladder lanes compute nothing new.  Here the 8 distinct
 * |z|^2 of a vector PAIR pack into one zmm (2 shuffles), ONE ladder runs
 * for both vectors, and the reciprocals unpack pair-duplicated (2
 * shuffles): ~21 arithmetic ops + 4 shuffles per pair vs 36 + 2 with two
 * map_step_v calls.  BIT-IDENTICAL to map_step_v per element: q_re + q_im
 * = q_im + q_re exactly (IEEE addition is commutative), max/rsqrt14/
 * rcp14/fma are elementwise, and the NR expressions are verbatim -- so
 * mixed use across chain families cannot break repeatability. */
static inline __attribute__((always_inline))
void map_step_pair(vec v0, vec v1, vec *o0, vec *o1)
{
    vec q0 = v0 * v0, q1 = v1 * v1;
    vec ms = VSH(q0, q1, 0,2,4,6,8,10,12,14)
           + VSH(q0, q1, 1,3,5,7,9,11,13,15);       /* 8 distinct |z|^2 */
    ms = (vec)_mm512_max_pd((__m512d)ms, (__m512d)VSPLAT(1e-300));
    vec y = (vec)_mm512_rsqrt14_pd((__m512d)ms);
    vec t_ = ms * y;
    y = (y * VSPLAT(0.5)) * VFNMA(t_, y, VSPLAT(3.0));
    t_ = ms * y;
    y = (y * VSPLAT(0.5)) * VFNMA(t_, y, VSPLAT(3.0));
    vec d = VFMA(ms, y, VSPLAT(1.0));               /* 1 + |z|          */
    vec r = (vec)_mm512_rcp14_pd((__m512d)d);
    r = r * VFNMA(d, r, VSPLAT(2.0));
    r = r * VFNMA(d, r, VSPLAT(2.0));
    *o0 = v0 * VSH(r, r, 0,0,1,1,2,2,3,3);
    *o1 = v1 * VSH(r, r, 4,4,5,5,6,6,7,7);
}

/* r6: masked sub-vector map tail for the ODD-L coverage sizes.  A span of
 * nd doubles has nd % 8 leftover doubles (1-3 complex: 2*L^3 % 8 == 6 and
 * 2*L^2 % 8 == 2 for odd L) that the map_vec family's /8 vector count
 * never touches -- caught by the create() chain gate at L=75 (verdict
 * l75-ip1.x, rel ~e-3: three unmapped elements).  Zero-masked loads keep
 * the unused ladder lanes finite (the 1e-300 max guard).  Same ladder ops
 * as map_step_v: bit-identical to every family's map on those elements.
 * For the scored sizes nd % 8 == 0 and the guarded calls compile out. */
static inline __attribute__((always_inline))
void map_span_tail(const double *z, const double *c, double *o, size_t nd)
{
    const unsigned rem = (unsigned)(nd % 8);
    if (!rem) return;
    const size_t base = nd - rem;
    const __mmask8 m = (__mmask8)((1u << rem) - 1);
    __m512d zv = _mm512_maskz_loadu_pd(m, z + base);
    __m512d cv = _mm512_maskz_loadu_pd(m, c + base);
    _mm512_mask_storeu_pd(o + base, m,
                          (__m512d)map_step_v((vec)_mm512_add_pd(zv, cv)));
}

/* sequential vectorized map over one contiguous span: o = (z+c)/(1+|z+c|).
 * 2-3 perfectly sequential streams (in-place z==o legal): the chain variant
 * for sizes where the volume does NOT stay cache-resident and folding the
 * map into the 100-stream x-pass would double the miss-stream count.
 * r5: pair-packed ladder (see map_step_pair); odd tails (the L=50 plane
 * prepass is 625 vectors) fall to one map_step_v. */
static void map_vec(const double *z, const double *c, double *o, size_t nvec)
{
    size_t i = 0;
    for (; i + 2 <= nvec; i += 2) {
        vec v0 = LDU(z + 8 * i)     + LDU(c + 8 * i);
        vec v1 = LDU(z + 8 * i + 8) + LDU(c + 8 * i + 8);
        vec o0, o1;
        map_step_pair(v0, v1, &o0, &o1);
        STU(o + 8 * i,     o0);
        STU(o + 8 * i + 8, o1);
    }
    if (i < nvec) {
        vec v = LDU(z + 8 * i) + LDU(c + 8 * i);
        STU(o + 8 * i, map_step_v(v));
    }
}

/* map_vec with the c stream fetched PREFETCHNTA (r5, the ipq/iqn
 * families): on SKX+ non-inclusive-LLC parts, NTA lines fill L1 only and
 * are dropped on eviction instead of allocated into L2/LLC.  c has NO
 * reuse within a step -- keeping it out of L3 leaves the whole LLC to the
 * state volume, which p2 and the next step's prepass re-read.  Values
 * identical to map_vec (a prefetch hint moves no data). */
static void map_vec_nta(const double *z, const double *c, double *o,
                        size_t nvec)
{
    size_t i = 0;
    for (; i + 2 <= nvec; i += 2) {
        _mm_prefetch((const char *)(c + 8 * i + 512), _MM_HINT_NTA);
        _mm_prefetch((const char *)(c + 8 * i + 520), _MM_HINT_NTA);
        vec v0 = LDU(z + 8 * i)     + LDU(c + 8 * i);
        vec v1 = LDU(z + 8 * i + 8) + LDU(c + 8 * i + 8);
        vec o0, o1;
        map_step_pair(v0, v1, &o0, &o1);
        STU(o + 8 * i,     o0);
        STU(o + 8 * i + 8, o1);
    }
    if (i < nvec) {
        vec v = LDU(z + 8 * i) + LDU(c + 8 * i);
        STU(o + 8 * i, map_step_v(v));
    }
}

/* map_vec with the c lines CLFLUSHOPT-ed one iteration after use (r5, the
 * ipk family): the guaranteed-semantics variant of the same L3-bypass --
 * NTA fill policy is implementation-defined, an explicit flush of a clean
 * line is architectural and writes nothing back.  Safe: nothing writes c,
 * so a flush racing an in-flight load re-reads identical bytes. */
static void map_vec_cfl(const double *z, const double *c, double *o,
                        size_t nvec)
{
    size_t i = 0;
    for (; i + 2 <= nvec; i += 2) {
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
    if (i < nvec) {
        vec v = LDU(z + 8 * i) + LDU(c + 8 * i);
        STU(o + 8 * i, map_step_v(v));
    }
}

/* map_vec walking BACKWARD (r3, the ipr candidate): phase 2 finishes its
 * in-place forward sweep with the TAIL of the state L3-hot; a reverse map
 * pass reaps that residue instead of missing on the evicted head, and
 * leaves the head hot for the NEXT step's forward p1.  Only matters when
 * the state half-fits L3 (L=100); raced, not assumed.  r5: pair-packed. */
static inline __attribute__((always_inline))
void map_vec_rev(const double *z, const double *c, double *o,
                        size_t nvec)
{
    size_t i = nvec;
    while (i >= 2) {
        i -= 2;
        vec v0 = LDU(z + 8 * i)     + LDU(c + 8 * i);
        vec v1 = LDU(z + 8 * i + 8) + LDU(c + 8 * i + 8);
        vec o0, o1;
        map_step_pair(v0, v1, &o0, &o1);
        STU(o + 8 * i,     o0);
        STU(o + 8 * i + 8, o1);
    }
    if (i == 1) {
        vec v = LDU(z) + LDU(c);
        STU(o, map_step_v(v));
    }
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

/* DFT25 = 5x5 Cooley-Tukey with exact twiddles, stage-B outputs handed
 * STRAIGHT to the caller's store macro (r2 change: was a function writing
 * r[25], which every PFA wrapper then re-read to route through the CRT
 * map -- a 25-store + 25-load L1 round-trip per call).  LDX(n) yields
 * input n (stride 1 in n), STO(k, v) consumes natural-order output k;
 * KMAP is applied by the caller inside STO.  Stage A stores U_[5*k1 + n2]
 * so stage B reads 5 contiguous hot slots.  192 FMA-port ops + 36 swaps. */
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

/* DFT16 = 4x4 Cooley-Tukey with exact W16 twiddles (r3, for L=80), same
 * fused-store shape as DFT25M: stage A = 4 x DFT4 over c (x[4c+d], d the
 * spectator digit), twiddle W16^{d*k1} folded at the U store; stage B =
 * 4 x DFT4 over d reading 4 contiguous hot slots, outputs straight to the
 * caller's store macro.  Twiddle products d*k1 in {1,2,3,4,6,9}: 8 CMULC
 * (2 ops + swap) plus one exact -i (1 mul + swap).  81 FMA-port ops. */
#define DFT16M(LDX, STO, KMAP) do {                                          \
    vec U_[16];                                                              \
    _Pragma("GCC unroll 4")                                                  \
    for (int d_ = 0; d_ < 4; ++d_) {                                         \
        vec y0_, y1_, y2_, y3_;                                              \
        DFT4M(LDX(d_), LDX(d_ + 4), LDX(d_ + 8), LDX(d_ + 12),               \
              y0_, y1_, y2_, y3_);                                           \
        U_[d_]      = y0_;                                                   \
        U_[4 + d_]  = d_ == 0 ? y1_                                          \
                    : d_ == 1 ? CMULC(y1_, KC16, KS16)                       \
                    : d_ == 2 ? CMULC(y1_, KR2, KR2)                         \
                    :           CMULC(y1_, KS16, KC16);                      \
        U_[8 + d_]  = d_ == 0 ? y2_                                          \
                    : d_ == 1 ? CMULC(y2_, KR2, KR2)                         \
                    : d_ == 2 ? MULNI(y2_)                                   \
                    :           CMULC(y2_, -KR2, KR2);                       \
        U_[12 + d_] = d_ == 0 ? y3_                                          \
                    : d_ == 1 ? CMULC(y3_, KS16, KC16)                       \
                    : d_ == 2 ? CMULC(y3_, -KR2, KR2)                        \
                    :           CMULC(y3_, -KC16, -KS16);                    \
    }                                                                        \
    _Pragma("GCC unroll 4")                                                  \
    for (int k1_ = 0; k1_ < 4; ++k1_) {                                      \
        vec r0_, r1_, r2_, r3_;                                              \
        DFT4M(U_[4 * k1_], U_[4 * k1_ + 1], U_[4 * k1_ + 2],                 \
              U_[4 * k1_ + 3], r0_, r1_, r2_, r3_);                          \
        STO(KMAP(k1_),      r0_);                                            \
        STO(KMAP(k1_ + 4),  r1_);                                            \
        STO(KMAP(k1_ + 8),  r2_);                                            \
        STO(KMAP(k1_ + 12), r3_);                                            \
    }                                                                        \
} while (0)

/* stage-A input and CRT output-index helpers for the DFT25/DFT16 users;
 * expanded inside the PFA wrappers where T_ and k2_ are in scope (the
 * store macro itself is passed through as DFT25M's STO parameter so the
 * PFA wrapper's ST argument reaches it) */
#define LDT25(n)   T_[25 * k2_ + (n)]
#define K50MAP(k)  ((26 * (k) + 25 * k2_) % 50)
#define K100MAP(k) ((76 * (k) + 25 * k2_) % 100)
#define LDT16(n)   T_[16 * k2_ + (n)]
#define K80MAP(k)  ((65 * (k) + 16 * k2_) % 80)

/* ---- r6 generic modules for the class-coverage sizes -------------------- */

/* Direct symmetric odd-N DFT (any odd NN >= 3, prime or not), FMA form:
 * fold a_j = x_j + x_{NN-j}, b_j = x_j - x_{NN-j}; then
 *     X_k = R_k - i I_k,   X_{NN-k} = R_k + i I_k,
 *     R_k = x0 + sum_j a_j cos(2 pi jk / NN),
 *     I_k =      sum_j b_j sin(2 pi jk / NN).
 * Algebra validated against numpy at N = 3,7,9,11,13 (strategy record).
 * Tables are exact literals indexed by (j*k) % NN -- every access folds to
 * a constant after unrolling (the C25T trick).  LDX(IMAP(n)) yields input
 * n (IMAP = a compile-time index map; IM_ID for hot T_ slots), STO(KMAP(k),
 * v) consumes natural-order output k.  Both LDX and STO must arrive as
 * macro PARAMETERS (the r2 rescan lesson).  FMA-port ops:
 * (NN-1) + h + h*(2h+2), h = (NN-1)/2, + h swaps:
 * 7 / 33 / 52 / 75 / 102 at NN = 3 / 7 / 9 / 11 / 13. */
#define DFTODDM(NN, CTT, STT, LDX, IMAP, STO, KMAP) do {                     \
    vec x0_ = LDX(IMAP(0));                                                  \
    vec a_[((NN) - 1) / 2], b_[((NN) - 1) / 2];                              \
    _Pragma("GCC unroll 8")                                                  \
    for (int j_ = 1; j_ <= ((NN) - 1) / 2; ++j_) {                           \
        vec u_ = LDX(IMAP(j_)), w_ = LDX(IMAP((NN) - j_));                   \
        a_[j_ - 1] = u_ + w_;                                                \
        b_[j_ - 1] = u_ - w_;                                                \
    }                                                                        \
    vec s0_ = x0_;                                                           \
    _Pragma("GCC unroll 8")                                                  \
    for (int j_ = 1; j_ <= ((NN) - 1) / 2; ++j_) s0_ = s0_ + a_[j_ - 1];     \
    STO(KMAP(0), s0_);                                                       \
    _Pragma("GCC unroll 8")                                                  \
    for (int k_ = 1; k_ <= ((NN) - 1) / 2; ++k_) {                           \
        vec r_ = VFMA(a_[0], VSPLAT(CTT[k_ % (NN)]), x0_);                   \
        vec i_ = b_[0] * VSPLAT(STT[k_ % (NN)]);                             \
        _Pragma("GCC unroll 8")                                              \
        for (int j_ = 2; j_ <= ((NN) - 1) / 2; ++j_) {                       \
            r_ = VFMA(a_[j_ - 1], VSPLAT(CTT[(j_ * k_) % (NN)]), r_);        \
            i_ = VFMA(b_[j_ - 1], VSPLAT(STT[(j_ * k_) % (NN)]), i_);        \
        }                                                                    \
        vec si_ = SWAP(i_);                                                  \
        STO(KMAP(k_),        VFMA (si_, VPAIR(1.0, -1.0), r_));              \
        STO(KMAP((NN) - k_), VFNMA(si_, VPAIR(1.0, -1.0), r_));              \
    }                                                                        \
} while (0)

/* DFT8, radix-2 DIT with W8 folded into the output butterflies -- the
 * PFA40C stage-2 body refactored into a reusable fused-store macro
 * (PFA40C itself is untouched: its generated code is scored).  26 FMA-port
 * ops + 6 swaps.  Natural-order outputs through STO(KMAP(k), v). */
#define DFT8M(LDX, IMAP, STO, KMAP) do {                                     \
    vec f0_ = LDX(IMAP(0)), f1_ = LDX(IMAP(1));                              \
    vec f2_ = LDX(IMAP(2)), f3_ = LDX(IMAP(3));                              \
    vec f4_ = LDX(IMAP(4)), f5_ = LDX(IMAP(5));                              \
    vec f6_ = LDX(IMAP(6)), f7_ = LDX(IMAP(7));                              \
    vec t0_ = f0_ + f4_, t1_ = f0_ - f4_;                                    \
    vec t2_ = f2_ + f6_, t3_ = f2_ - f6_;                                    \
    vec s0_ = f1_ + f5_, s1_ = f1_ - f5_;                                    \
    vec s2_ = f3_ + f7_, s3_ = f3_ - f7_;                                    \
    vec E0_ = t0_ + t2_, E2_ = t0_ - t2_;                                    \
    vec m1_ = SWAP(t3_);                                                     \
    vec E1_ = VFMA (m1_, VPAIR(1.0, -1.0), t1_);                             \
    vec E3_ = VFNMA(m1_, VPAIR(1.0, -1.0), t1_);                             \
    vec O0_ = s0_ + s2_, O2_ = s0_ - s2_;                                    \
    vec m2_ = SWAP(s3_);                                                     \
    vec O1_ = VFMA (m2_, VPAIR(1.0, -1.0), s1_);                             \
    vec O3_ = VFNMA(m2_, VPAIR(1.0, -1.0), s1_);                             \
    vec w1_ = VFMA(SWAP(O1_), VPAIR(1.0, -1.0), O1_);       /* O1*(1-i) */   \
    vec w3_ = VFMA(SWAP(O3_), VPAIR(1.0, -1.0), O3_);       /* O3*(1-i) */   \
    vec sw3_ = SWAP(w3_);                                                    \
    vec sO2_ = SWAP(O2_);                                                    \
    STO(KMAP(0), E0_ + O0_);                                                 \
    STO(KMAP(4), E0_ - O0_);                                                 \
    STO(KMAP(1), VFMA (w1_, VSPLAT(KR2), E1_));                              \
    STO(KMAP(5), VFNMA(w1_, VSPLAT(KR2), E1_));                              \
    STO(KMAP(2), VFMA (sO2_, VPAIR(1.0, -1.0), E2_));                        \
    STO(KMAP(6), VFNMA(sO2_, VPAIR(1.0, -1.0), E2_));                        \
    STO(KMAP(3), VFMA (sw3_, VPAIR(KR2, -KR2), E3_));                        \
    STO(KMAP(7), VFNMA(sw3_, VPAIR(KR2, -KR2), E3_));                        \
} while (0)

/* shared index helpers for the coverage codelets: IM_ID for hot T_ slots;
 * LDTn reads stage-2 input n of block k2_; TSTn(k, v) is the stage-1 store
 * T_[N1*k + n1_] (n1_/k2_/T_ are in scope at the expansion site). */
#define IM_ID(n)    (n)
#define LDT8(n)     T_[8  * k2_ + (n)]
#define LDT9(n)     T_[9  * k2_ + (n)]
#define LDT11(n)    T_[11 * k2_ + (n)]
#define LDT13(n)    T_[13 * k2_ + (n)]
#define TST8(k, v)  (T_[8  * (k) + n1_] = (v))
#define TST9(k, v)  (T_[9  * (k) + n1_] = (v))
#define TST11(k, v) (T_[11 * (k) + n1_] = (v))
#define TST13(k, v) (T_[13 * (k) + n1_] = (v))
#define TST16(k, v) (T_[16 * (k) + n1_] = (v))
#define TST25(k, v) (T_[25 * (k) + n1_] = (v))

/* ---- per-size Good-Thomas line codelets, LD/ST as macro parameters ------ */

/* L=40 = 8x5.  Stage 1: 8 x DFT5 into T_[8*k2 + n1]; stage 2: 5 x DFT8
 * (radix-2 DIT, W8 twiddles folded into the output butterflies) reading 8
 * contiguous slots and handing every output straight to ST. */
#define PFA40C(LD, ST) do {                                                  \
    vec T_[40];                                                              \
    _Pragma("GCC unroll 8")                                                  \
    for (int n1_ = 0; n1_ < 8; ++n1_) {                                      \
        vec a0_ = LD((5 * n1_     ) % 40);                                   \
        vec a1_ = LD((5 * n1_ +  8) % 40);                                   \
        vec a2_ = LD((5 * n1_ + 16) % 40);                                   \
        vec a3_ = LD((5 * n1_ + 24) % 40);                                   \
        vec a4_ = LD((5 * n1_ + 32) % 40);                                   \
        DFT5M(a0_, a1_, a2_, a3_, a4_,                                       \
              T_[n1_], T_[8 + n1_], T_[16 + n1_], T_[24 + n1_],              \
              T_[32 + n1_]);                                                 \
    }                                                                        \
    _Pragma("GCC unroll 5")                                                  \
    for (int k2_ = 0; k2_ < 5; ++k2_) {                                      \
        const vec *f_ = T_ + 8 * k2_;                                        \
        vec t0_ = f_[0] + f_[4], t1_ = f_[0] - f_[4];                        \
        vec t2_ = f_[2] + f_[6], t3_ = f_[2] - f_[6];                        \
        vec s0_ = f_[1] + f_[5], s1_ = f_[1] - f_[5];                        \
        vec s2_ = f_[3] + f_[7], s3_ = f_[3] - f_[7];                        \
        vec E0_ = t0_ + t2_, E2_ = t0_ - t2_;                                \
        vec m1_ = SWAP(t3_);                                                 \
        vec E1_ = VFMA (m1_, VPAIR(1.0, -1.0), t1_);                         \
        vec E3_ = VFNMA(m1_, VPAIR(1.0, -1.0), t1_);                         \
        vec O0_ = s0_ + s2_, O2_ = s0_ - s2_;                                \
        vec m2_ = SWAP(s3_);                                                 \
        vec O1_ = VFMA (m2_, VPAIR(1.0, -1.0), s1_);                         \
        vec O3_ = VFNMA(m2_, VPAIR(1.0, -1.0), s1_);                         \
        vec w1_ = VFMA(SWAP(O1_), VPAIR(1.0, -1.0), O1_);   /* O1*(1-i)  */  \
        vec w3_ = VFMA(SWAP(O3_), VPAIR(1.0, -1.0), O3_);   /* O3*(1-i)  */  \
        vec sw3_ = SWAP(w3_);                                                \
        vec sO2_ = SWAP(O2_);                                                \
        ST((         16 * k2_) % 40, E0_ + O0_);                             \
        ST((25 * 4 + 16 * k2_) % 40, E0_ - O0_);                             \
        ST((25     + 16 * k2_) % 40, VFMA (w1_, VSPLAT(KR2), E1_));          \
        ST((25 * 5 + 16 * k2_) % 40, VFNMA(w1_, VSPLAT(KR2), E1_));          \
        ST((25 * 2 + 16 * k2_) % 40, VFMA (sO2_, VPAIR(1.0, -1.0), E2_));    \
        ST((25 * 6 + 16 * k2_) % 40, VFNMA(sO2_, VPAIR(1.0, -1.0), E2_));    \
        ST((25 * 3 + 16 * k2_) % 40, VFMA (sw3_, VPAIR(KR2, -KR2), E3_));    \
        ST((25 * 7 + 16 * k2_) % 40, VFNMA(sw3_, VPAIR(KR2, -KR2), E3_));    \
    }                                                                        \
} while (0)

/* L=50 = 25x2.  Stage 1: 25 x DFT2 into T_[25*k2 + n1]; stage 2: 2 x DFT25
 * on contiguous slots, stage-B outputs stored straight through ST via the
 * CRT map (r2: no R_[25] round-trip). */
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

/* L=80 = 16x5 (r3).  n = (5 n1 + 16 n2) % 80, k = (65 k1 + 16 k2) % 80
 * (65 = 1 mod 16 and 0 mod 5; 16 = 1 mod 5 and 0 mod 16).  Stage 1:
 * 16 x DFT5 into T_[16*k2 + n1]; stage 2: 5 x DFT16 on contiguous slots,
 * stage-B outputs straight through ST via the CRT map.  661 ops/line. */
#define PFA80C(LD, ST) do {                                                  \
    vec T_[80];                                                              \
    _Pragma("GCC unroll 16")                                                 \
    for (int n1_ = 0; n1_ < 16; ++n1_) {                                     \
        vec a0_ = LD((5 * n1_     ) % 80);                                   \
        vec a1_ = LD((5 * n1_ + 16) % 80);                                   \
        vec a2_ = LD((5 * n1_ + 32) % 80);                                   \
        vec a3_ = LD((5 * n1_ + 48) % 80);                                   \
        vec a4_ = LD((5 * n1_ + 64) % 80);                                   \
        DFT5M(a0_, a1_, a2_, a3_, a4_,                                       \
              T_[n1_], T_[16 + n1_], T_[32 + n1_], T_[48 + n1_],             \
              T_[64 + n1_]);                                                 \
    }                                                                        \
    _Pragma("GCC unroll 5")                                                  \
    for (int k2_ = 0; k2_ < 5; ++k2_)                                        \
        DFT16M(LDT16, ST, K80MAP);                                           \
} while (0)

/* L=100 = 25x4.  Stage 1: 25 x DFT4 into T_[25*k2 + n1]; stage 2: 4 x DFT25,
 * stage-B outputs stored straight through ST via the CRT map (r2). */
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

/* ---- r6 class-coverage codelets (lean sizes) ----------------------------
 * Uniform GT-PFA shape, maps validated against numpy for all 16 sizes:
 *   n = (N2*n1 + N1*n2) % L,  stage 1 = N1 x DFT_{N2} into T_[N1*k2 + n1],
 *   stage 2 = N2 x DFT_{N1} (the long module, reading contiguous hot slots
 *   and storing straight through ST),  k = (A*k1 + B*k2) % L with
 *   A === 1 mod N1, 0 mod N2 and B === 1 mod N2, 0 mod N1. */

/* L=44 = 11x4: 11 x DFT4, then 4 x DFT11.  388 FMA-port ops/line. */
#define K44MAP(k)  ((12 * (k) + 33 * k2_) % 44)
#define PFA44C(LD, ST) do {                                                  \
    vec T_[44];                                                              \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 11; ++n1_)                                       \
        DFT4M(LD((4 * n1_) % 44),      LD((4 * n1_ + 11) % 44),              \
              LD((4 * n1_ + 22) % 44), LD((4 * n1_ + 33) % 44),              \
              T_[n1_], T_[11 + n1_], T_[22 + n1_], T_[33 + n1_]);            \
    _Pragma("GCC unroll 4")                                                  \
    for (int k2_ = 0; k2_ < 4; ++k2_)                                        \
        DFTODDM(11, C11T, S11T, LDT11, IM_ID, ST, K44MAP);                   \
} while (0)

/* L=48 = 16x3: 16 x DFT3, then 3 x DFT16.  355 ops/line. */
#define IM48(n)    ((3 * n1_ + 16 * (n)) % 48)
#define K48MAP(k)  ((33 * (k) + 16 * k2_) % 48)
#define PFA48C(LD, ST) do {                                                  \
    vec T_[48];                                                              \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 16; ++n1_)                                       \
        DFTODDM(3, C3T, S3T, LD, IM48, TST16, IM_ID);                        \
    _Pragma("GCC unroll 3")                                                  \
    for (int k2_ = 0; k2_ < 3; ++k2_)                                        \
        DFT16M(LDT16, ST, K48MAP);                                           \
} while (0)

/* L=52 = 13x4: 13 x DFT4, then 4 x DFT13.  512 ops/line. */
#define K52MAP(k)  ((40 * (k) + 13 * k2_) % 52)
#define PFA52C(LD, ST) do {                                                  \
    vec T_[52];                                                              \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 13; ++n1_)                                       \
        DFT4M(LD((4 * n1_) % 52),      LD((4 * n1_ + 13) % 52),              \
              LD((4 * n1_ + 26) % 52), LD((4 * n1_ + 39) % 52),              \
              T_[n1_], T_[13 + n1_], T_[26 + n1_], T_[39 + n1_]);            \
    _Pragma("GCC unroll 4")                                                  \
    for (int k2_ = 0; k2_ < 4; ++k2_)                                        \
        DFTODDM(13, C13T, S13T, LDT13, IM_ID, ST, K52MAP);                   \
} while (0)

/* L=55 = 11x5: 11 x DFT5, then 5 x DFT11.  551 ops/line. */
#define K55MAP(k)  ((45 * (k) + 11 * k2_) % 55)
#define PFA55C(LD, ST) do {                                                  \
    vec T_[55];                                                              \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 11; ++n1_)                                       \
        DFT5M(LD((5 * n1_) % 55),      LD((5 * n1_ + 11) % 55),              \
              LD((5 * n1_ + 22) % 55), LD((5 * n1_ + 33) % 55),              \
              LD((5 * n1_ + 44) % 55),                                       \
              T_[n1_], T_[11 + n1_], T_[22 + n1_], T_[33 + n1_],             \
              T_[44 + n1_]);                                                 \
    _Pragma("GCC unroll 5")                                                  \
    for (int k2_ = 0; k2_ < 5; ++k2_)                                        \
        DFTODDM(11, C11T, S11T, LDT11, IM_ID, ST, K55MAP);                   \
} while (0)

/* L=56 = 8x7: 8 x DFT7, then 7 x DFT8.  446 ops/line. */
#define IM56(n)    ((7 * n1_ + 8 * (n)) % 56)
#define K56MAP(k)  ((49 * (k) + 8 * k2_) % 56)
#define PFA56C(LD, ST) do {                                                  \
    vec T_[56];                                                              \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 8; ++n1_)                                        \
        DFTODDM(7, C7T, S7T, LD, IM56, TST8, IM_ID);                         \
    _Pragma("GCC unroll 25")                                                 \
    for (int k2_ = 0; k2_ < 7; ++k2_)                                        \
        DFT8M(LDT8, IM_ID, ST, K56MAP);                                      \
} while (0)

/* L=63 = 9x7: 9 x DFT7, then 7 x DFT9.  661 ops/line. */
#define IM63(n)    ((7 * n1_ + 9 * (n)) % 63)
#define K63MAP(k)  ((28 * (k) + 36 * k2_) % 63)
#define PFA63C(LD, ST) do {                                                  \
    vec T_[63];                                                              \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 9; ++n1_)                                        \
        DFTODDM(7, C7T, S7T, LD, IM63, TST9, IM_ID);                         \
    _Pragma("GCC unroll 25")                                                 \
    for (int k2_ = 0; k2_ < 7; ++k2_)                                        \
        DFTODDM(9, C9T, S9T, LDT9, IM_ID, ST, K63MAP);                       \
} while (0)

/* L=65 = 13x5: 13 x DFT5, then 5 x DFT13.  718 ops/line. */
#define K65MAP(k)  ((40 * (k) + 26 * k2_) % 65)
#define PFA65C(LD, ST) do {                                                  \
    vec T_[65];                                                              \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 13; ++n1_)                                       \
        DFT5M(LD((5 * n1_) % 65),      LD((5 * n1_ + 13) % 65),              \
              LD((5 * n1_ + 26) % 65), LD((5 * n1_ + 39) % 65),              \
              LD((5 * n1_ + 52) % 65),                                       \
              T_[n1_], T_[13 + n1_], T_[26 + n1_], T_[39 + n1_],             \
              T_[52 + n1_]);                                                 \
    _Pragma("GCC unroll 5")                                                  \
    for (int k2_ = 0; k2_ < 5; ++k2_)                                        \
        DFTODDM(13, C13T, S13T, LDT13, IM_ID, ST, K65MAP);                   \
} while (0)

/* L=72 = 9x8: 9 x DFT8, then 8 x DFT9.  650 ops/line. */
#define IM72(n)    ((8 * n1_ + 9 * (n)) % 72)
#define K72MAP(k)  ((64 * (k) + 9 * k2_) % 72)
#define PFA72C(LD, ST) do {                                                  \
    vec T_[72];                                                              \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 9; ++n1_)                                        \
        DFT8M(LD, IM72, TST9, IM_ID);                                        \
    _Pragma("GCC unroll 25")                                                 \
    for (int k2_ = 0; k2_ < 8; ++k2_)                                        \
        DFTODDM(9, C9T, S9T, LDT9, IM_ID, ST, K72MAP);                       \
} while (0)

/* L=75 = 25x3: 25 x DFT3, then 3 x DFT25.  751 ops/line. */
#define IM75(n)    ((3 * n1_ + 25 * (n)) % 75)
#define K75MAP(k)  ((51 * (k) + 25 * k2_) % 75)
#define PFA75C(LD, ST) do {                                                  \
    vec T_[75];                                                              \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 25; ++n1_)                                       \
        DFTODDM(3, C3T, S3T, LD, IM75, TST25, IM_ID);                        \
    _Pragma("GCC unroll 3")                                                  \
    for (int k2_ = 0; k2_ < 3; ++k2_)                                        \
        DFT25M(LDT25, ST, K75MAP);                                           \
} while (0)

/* L=77 = 11x7: 11 x DFT7, then 7 x DFT11.  888 ops/line. */
#define IM77(n)    ((7 * n1_ + 11 * (n)) % 77)
#define K77MAP(k)  ((56 * (k) + 22 * k2_) % 77)
#define PFA77C(LD, ST) do {                                                  \
    vec T_[77];                                                              \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 11; ++n1_)                                       \
        DFTODDM(7, C7T, S7T, LD, IM77, TST11, IM_ID);                        \
    _Pragma("GCC unroll 25")                                                 \
    for (int k2_ = 0; k2_ < 7; ++k2_)                                        \
        DFTODDM(11, C11T, S11T, LDT11, IM_ID, ST, K77MAP);                   \
} while (0)

/* L=88 = 11x8: 11 x DFT8, then 8 x DFT11.  886 ops/line. */
#define IM88(n)    ((8 * n1_ + 11 * (n)) % 88)
#define K88MAP(k)  ((56 * (k) + 33 * k2_) % 88)
#define PFA88C(LD, ST) do {                                                  \
    vec T_[88];                                                              \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 11; ++n1_)                                       \
        DFT8M(LD, IM88, TST11, IM_ID);                                       \
    _Pragma("GCC unroll 25")                                                 \
    for (int k2_ = 0; k2_ < 8; ++k2_)                                        \
        DFTODDM(11, C11T, S11T, LDT11, IM_ID, ST, K88MAP);                   \
} while (0)

/* L=91 = 13x7: 13 x DFT7, then 7 x DFT13.  1143 ops/line. */
#define IM91(n)    ((7 * n1_ + 13 * (n)) % 91)
#define K91MAP(k)  ((14 * (k) + 78 * k2_) % 91)
#define PFA91C(LD, ST) do {                                                  \
    vec T_[91];                                                              \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 13; ++n1_)                                       \
        DFTODDM(7, C7T, S7T, LD, IM91, TST13, IM_ID);                        \
    _Pragma("GCC unroll 25")                                                 \
    for (int k2_ = 0; k2_ < 7; ++k2_)                                        \
        DFTODDM(13, C13T, S13T, LDT13, IM_ID, ST, K91MAP);                   \
} while (0)

/* L=99 = 11x9: 11 x DFT9, then 9 x DFT11.  1247 ops/line. */
#define IM99(n)    ((9 * n1_ + 11 * (n)) % 99)
#define K99MAP(k)  ((45 * (k) + 55 * k2_) % 99)
#define PFA99C(LD, ST) do {                                                  \
    vec T_[99];                                                              \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 11; ++n1_)                                       \
        DFTODDM(9, C9T, S9T, LD, IM99, TST11, IM_ID);                        \
    _Pragma("GCC unroll 25")                                                 \
    for (int k2_ = 0; k2_ < 9; ++k2_)                                        \
        DFTODDM(11, C11T, S11T, LDT11, IM_ID, ST, K99MAP);                   \
} while (0)

/* L=104 = 13x8: 13 x DFT8, then 8 x DFT13.  1154 ops/line. */
#define IM104(n)   ((8 * n1_ + 13 * (n)) % 104)
#define K104MAP(k) ((40 * (k) + 65 * k2_) % 104)
#define PFA104C(LD, ST) do {                                                 \
    vec T_[104];                                                             \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 13; ++n1_)                                       \
        DFT8M(LD, IM104, TST13, IM_ID);                                      \
    _Pragma("GCC unroll 25")                                                 \
    for (int k2_ = 0; k2_ < 8; ++k2_)                                        \
        DFTODDM(13, C13T, S13T, LDT13, IM_ID, ST, K104MAP);                  \
} while (0)

/* L=112 = 16x7: 16 x DFT7, then 7 x DFT16.  1095 ops/line. */
#define IM112(n)   ((7 * n1_ + 16 * (n)) % 112)
#define K112MAP(k) ((49 * (k) + 64 * k2_) % 112)
#define PFA112C(LD, ST) do {                                                 \
    vec T_[112];                                                             \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 16; ++n1_)                                       \
        DFTODDM(7, C7T, S7T, LD, IM112, TST16, IM_ID);                       \
    _Pragma("GCC unroll 25")                                                 \
    for (int k2_ = 0; k2_ < 7; ++k2_)                                        \
        DFT16M(LDT16, ST, K112MAP);                                          \
} while (0)

/* L=117 = 13x9: 13 x DFT9, then 9 x DFT13.  1594 ops/line. */
#define IM117(n)   ((9 * n1_ + 13 * (n)) % 117)
#define K117MAP(k) ((27 * (k) + 91 * k2_) % 117)
#define PFA117C(LD, ST) do {                                                 \
    vec T_[117];                                                             \
    _Pragma("GCC unroll 25")                                                 \
    for (int n1_ = 0; n1_ < 13; ++n1_)                                       \
        DFTODDM(9, C9T, S9T, LD, IM117, TST13, IM_ID);                       \
    _Pragma("GCC unroll 25")                                                 \
    for (int k2_ = 0; k2_ < 9; ++k2_)                                        \
        DFTODDM(13, C13T, S13T, LDT13, IM_ID, ST, K117MAP);                  \
} while (0)

#define GCAT_(a,b) a##b
#define GCAT(a,b)  GCAT_(a,b)

/* instantiate the engine for the three sizes.
 * (r2 A/B, killed: pre-RA scheduling via optimize("schedule-insns",
 * "sched-pressure") on the L=50 family -- gen_powp's -5% at their L=25 --
 * measured 662 vs 472 us/xform at L=50 B=4 on the node, +40%.  Do not
 * rediscover.) */
#define GENL 40
#define GPP  44                       /* pl row pitch: 704 B = 11 lines, odd */
#define PFAL PFA40C
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

#define GENL 80
#define GPP  84                       /* 1344 B = 21 lines, odd              */
#define PFAL PFA80C
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

/* r6: LEAN instantiations for the 16 class-coverage sizes (surprise-round
 * duty).  GEN_LEAN emits only the ip and ipp chain families and compiles the
 * heavy bodies once as noinline functions (see the template).  GPP keeps
 * the odd-cache-line rule: GPP % 4 == 0 and GPP/4 odd. */
#define GEN_LEAN 1

#define GENL 44
#define GPP  44                       /* 704 B = 11 lines, odd               */
#define PFAL PFA44C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 48
#define GPP  52                       /* 832 B = 13 lines, odd               */
#define PFAL PFA48C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 52
#define GPP  52                       /* 13 lines, odd                       */
#define PFAL PFA52C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 55
#define GPP  60                       /* 15 lines, odd                       */
#define PFAL PFA55C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 56
#define GPP  60                       /* 15 lines, odd                       */
#define PFAL PFA56C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 63
#define GPP  68                       /* 17 lines, odd                       */
#define PFAL PFA63C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 65
#define GPP  68                       /* 17 lines, odd                       */
#define PFAL PFA65C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 72
#define GPP  76                       /* 19 lines, odd                       */
#define PFAL PFA72C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 75
#define GPP  76                       /* 19 lines, odd                       */
#define PFAL PFA75C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 77
#define GPP  84                       /* 21 lines, odd                       */
#define PFAL PFA77C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 88
#define GPP  92                       /* 23 lines, odd                       */
#define PFAL PFA88C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 91
#define GPP  92                       /* 23 lines, odd                       */
#define PFAL PFA91C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 99
#define GPP  100                      /* 25 lines, odd                       */
#define PFAL PFA99C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 104
#define GPP  108                      /* 27 lines, odd                       */
#define PFAL PFA104C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 112
#define GPP  116                      /* 29 lines, odd                       */
#define PFAL PFA112C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#define GENL 117
#define GPP  124                      /* 31 lines, odd                       */
#define PFAL PFA117C
#include __FILE__
#undef GENL
#undef GPP
#undef PFAL

#undef GEN_LEAN

#endif /* __AVX512F__ */

/* ---- plan, gate, race, API ---------------------------------------------- */

typedef void (*execpl_fn)(const double *in, double *out, long nvol, double *P);
typedef void (*chainpl_fn)(const double *cur, double *dst, const double *cf,
                           long nvol, double *M, double *P);

struct fft3d_plan {
    int L, batch;
    execpl_fn  fn;                /* NULL -> dense fallback                  */
    chainpl_fn cfn;               /* NULL -> execute + scalar map            */
    int dm;                       /* cfn is the deferred-map (ipm) schedule  */
    double *P;                    /* plane scratch (heap: stack 4K-aliases)  */
    double *M;                    /* one mid volume for the fused chain      */
    double *X;                    /* batch state volumes for chain ping-pong */
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

const char *fft3d_name(void) { return "gen_pfa_large"; }

static char g_desc[224];
const char *fft3d_description(void)
{
    return g_desc[0] ? g_desc
        : "PFA coprime pairs, large: OWN 40,50,80,100 + 16 r6 coverage "
          "sizes 44..117 (every 2-stage coprime composite, modules "
          "2/3/4/5/7/8/9/11/13/16/25, exact twiddles); GT-PFA two-sweep "
          "zmm lanes, volume-major owned chain (pair-packed map, c-flush "
          "L3 bypass at >L3 sizes), gate+race+playoff in create(), "
          "noise-gated wisdom (r9: only tight-spread verdicts persist)";
}
static const int g_Ls[] = { 40, 44, 48, 50, 52, 55, 56, 63, 65, 72, 75,
                            77, 80, 88, 91, 99, 100, 104, 112, 117 };
int fft3d_supports(int L)
{
    for (size_t i = 0; i < sizeof g_Ls / sizeof g_Ls[0]; ++i)
        if (g_Ls[i] == L) return 1;
    return 0;
}

/* scalar O(L^2)-per-line reference (L45_pfa's ref3d, generalized in L):
 * independent ground truth for the create()-time gate */
static void refnd(int L, const double _Complex *in, double _Complex *out)
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

struct candpl { execpl_fn fn; chainpl_fn cfn; int pf, rank, dm; const char *nm; };

/* candidates are raced on the CHAIN step (the graded workload) when the mid
 * volume exists; the exec fn rides along with the chain winner.  Two chain
 * families: ip* keeps everything in place (one volume-sized working set +
 * c; wins when the volume does not fit near caches and miss streams are the
 * bottleneck -- L=100), fused f* folds the map into phase 2's stores (wins
 * when the volume is cache-resident and the map's extra pass is pure cost
 * -- L=40/50). */
/* pf ids (GENPFL_PF forcing): 0 ip0, 1 ip1, 2 f0, 3 fr, 4 frw, 5 ipf0,
 * 6 ipf1, 7 ipnt, 8 ipfnt, 9 ipe0, 10 ipe1, 11 ipm0, 12 ipm1, 13 ip2,
 * 14 ipr1 (r3), 15 ipp0, 16 ipp1 (r4), 17 ipq1, 18 ipk1, 19 iqn1 (r5).
 * rank = simplest-first tie preference within the 3% hysteresis band, set
 * PER SIZE from the r2/r3 node race tables (ip pair leads everywhere
 * measured; the deferred-map pairs ipm/ipp rank right behind ip so they
 * must beat it by >3% to displace it; ipf loses outright at 100, NT loses
 * ~10% there -- kept for the cross-arch hosts). */
static const struct candpl g_c40[]  = {
    { x_pf0_40,     xc_ip0_40,      0,  0, 0, "l40-ip0"   },
    { x_pf1_40,     xc_ip1_40,      1,  1, 0, "l40-ip1"   },
    { x_pf1_40,     xc_ip2_40,     13,  2, 0, "l40-ip2"   },
    { x_pf1_40,     xc_ipr1_40,    14,  3, 0, "l40-ipr1"  },
    { x_pf1_40,     xc_ipm1_40,    12,  4, 1, "l40-ipm1"  },
    { x_pf0_40,     xc_ipm0_40,    11,  5, 1, "l40-ipm0"  },
    { x_pf1_40,     xc_ipp1_40,    16,  6, 1, "l40-ipp1"  },
    { x_pf0_40,     xc_ipp0_40,    15,  7, 1, "l40-ipp0"  },
    { x_pf1_40,     xc_ipe1_40,    10,  8, 0, "l40-ipe1"  },
    { x_pf0_40,     xc_ipe0_40,     9,  9, 0, "l40-ipe0"  },
    { x_pf1_40,     xc_ipnt_40,     7, 10, 0, "l40-ipnt"  },
    { x_pf1_40,     xc_ipfnt_40,    8, 11, 0, "l40-ipfnt" },
    { x_pf1_40,     xc_ipf1_40,     6, 12, 0, "l40-ipf1"  },
    { x_pf0_40,     xc_ipf0_40,     5, 13, 0, "l40-ipf0"  },
    { x_pf0_40,     xc_pf0_40,      2, 14, 0, "l40-f0"    },
    { x_pf1_40,     xc_pfr_40,      3, 15, 0, "l40-fr"    },
    { x_pf1_40,     xc_pfrw_40,     4, 16, 0, "l40-frw"   },
    { x_pf1_40,     xc_ipq1_40,    17, 17, 1, "l40-ipq1"  },
    { x_pf1_40,     xc_ipk1_40,    18, 18, 1, "l40-ipk1"  },
    { x_pf1_40,     xc_iqn1_40,    19, 19, 0, "l40-iqn1"  } };
static const struct candpl g_c50[]  = {
    { x_pf1_50,     xc_ip1_50,      1,  0, 0, "l50-ip1"   },
    { x_pf0_50,     xc_ip0_50,      0,  1, 0, "l50-ip0"   },
    { x_pf1_50,     xc_ip2_50,     13,  2, 0, "l50-ip2"   },
    { x_pf1_50,     xc_ipr1_50,    14,  3, 0, "l50-ipr1"  },
    { x_pf1_50,     xc_ipm1_50,    12,  4, 1, "l50-ipm1"  },
    { x_pf0_50,     xc_ipm0_50,    11,  5, 1, "l50-ipm0"  },
    { x_pf1_50,     xc_ipp1_50,    16,  6, 1, "l50-ipp1"  },
    { x_pf0_50,     xc_ipp0_50,    15,  7, 1, "l50-ipp0"  },
    { x_pf1_50,     xc_ipe1_50,    10,  8, 0, "l50-ipe1"  },
    { x_pf0_50,     xc_ipe0_50,     9,  9, 0, "l50-ipe0"  },
    { x_pf1_50,     xc_ipf1_50,     6, 10, 0, "l50-ipf1"  },
    { x_pf0_50,     xc_ipf0_50,     5, 11, 0, "l50-ipf0"  },
    { x_pf0_50,     xc_pf0_50,      2, 12, 0, "l50-f0"    },
    { x_pf1_50,     xc_pfr_50,      3, 13, 0, "l50-fr"    },
    { x_pf1_50,     xc_pfrw_50,     4, 14, 0, "l50-frw"   },
    { x_pf1_50,     xc_ipq1_50,    17, 15, 1, "l50-ipq1"  },
    { x_pf1_50,     xc_ipk1_50,    18, 16, 1, "l50-ipk1"  },
    { x_pf1_50,     xc_iqn1_50,    19, 17, 0, "l50-iqn1"  } };
static const struct candpl g_c80[]  = {
    { x_pf1_80,     xc_ip1_80,      1,  0, 0, "l80-ip1"   },
    { x_pf0_80,     xc_ip0_80,      0,  1, 0, "l80-ip0"   },
    { x_pf1_80,     xc_ip2_80,     13,  2, 0, "l80-ip2"   },
    { x_pf1_80,     xc_ipr1_80,    14,  3, 0, "l80-ipr1"  },
    { x_pf1_80,     xc_ipm1_80,    12,  4, 1, "l80-ipm1"  },
    { x_pf0_80,     xc_ipm0_80,    11,  5, 1, "l80-ipm0"  },
    { x_pf1_80,     xc_ipp1_80,    16,  6, 1, "l80-ipp1"  },
    { x_pf0_80,     xc_ipp0_80,    15,  7, 1, "l80-ipp0"  },
    { x_pf1_80,     xc_ipe1_80,    10,  8, 0, "l80-ipe1"  },
    { x_pf0_80,     xc_ipe0_80,     9,  9, 0, "l80-ipe0"  },
    { x_pf1_80,     xc_ipnt_80,     7, 10, 0, "l80-ipnt"  },
    { x_pf1_80,     xc_ipfnt_80,    8, 11, 0, "l80-ipfnt" },
    { x_pf1_80,     xc_ipf1_80,     6, 12, 0, "l80-ipf1"  },
    { x_pf0_80,     xc_ipf0_80,     5, 13, 0, "l80-ipf0"  },
    { x_pf0_80,     xc_pf0_80,      2, 14, 0, "l80-f0"    },
    { x_pf1_80,     xc_pfr_80,      3, 15, 0, "l80-fr"    },
    { x_pf1_80,     xc_pfrw_80,     4, 16, 0, "l80-frw"   },
    { x_pf1_80,     xc_ipq1_80,    17, 17, 1, "l80-ipq1"  },
    { x_pf1_80,     xc_ipk1_80,    18, 18, 1, "l80-ipk1"  },
    { x_pf1_80,     xc_iqn1_80,    19, 19, 0, "l80-iqn1"  } };
/* r5 rank reorder at 100 ONLY (gen_powp's r4 move, their evidence rule):
 * the pair-packed map narrowed the race gap ipp1-vs-ip1 to <1% and the
 * hysteresis reverted to ip1, but held-lease paired minima say ipp1's
 * quiet floor is lower (4657-4668 vs ip1's 4842 this session; r4 paired
 * quiet -3.4%, busy -11%) and the score is measured on full quiet.  A
 * margin that shrinks under the tie band falls to the measured winner.
 * r7: gen_powp's r6 ranked ipk1 FIRST on their side (5/5 held-lease
 * pairs); tried here and REFUTED on this engine -- my r7 pairs went 4/5
 * to ipp1 (min-of-mins 4645 vs 4804, -3.3%), matching my r5 session
 * (4/4).  ipp1 keeps rank 0: ITS quiet floor is the lower one here, and
 * the score is measured on full quiet.  ipk1 moves up to rank 2 (ahead
 * of plain ip*): it is trial-best in every interleaved race since r5 and
 * is the bypass insurance the CLX/SPR wisdom races exist for. */
static const struct candpl g_c100[]  = {
    { x_pf1_100,    xc_ipp1_100,   16,  0, 1, "l100-ipp1"  },
    { x_pf0_100,    xc_ipp0_100,   15,  1, 1, "l100-ipp0"  },
    { x_pf1_100,    xc_ipk1_100,   18,  2, 1, "l100-ipk1"  },
    { x_pf1_100,    xc_ip1_100,     1,  3, 0, "l100-ip1"   },
    { x_pf0_100,    xc_ip0_100,     0,  4, 0, "l100-ip0"   },
    { x_pf1_100,    xc_ip2_100,    13,  5, 0, "l100-ip2"   },
    { x_pf1_100,    xc_ipr1_100,   14,  6, 0, "l100-ipr1"  },
    { x_pf1_100,    xc_ipm1_100,   12,  7, 1, "l100-ipm1"  },
    { x_pf0_100,    xc_ipm0_100,   11,  8, 1, "l100-ipm0"  },
    { x_pf1_100,    xc_ipe1_100,   10,  9, 0, "l100-ipe1"  },
    { x_pf0_100,    xc_ipe0_100,    9, 10, 0, "l100-ipe0"  },
    { x_pf1_100,    xc_ipnt_100,    7, 11, 0, "l100-ipnt"  },
    { x_pf1_100,    xc_ipfnt_100,   8, 12, 0, "l100-ipfnt" },
    { x_pf1_100,    xc_ipf1_100,    6, 13, 0, "l100-ipf1"  },
    { x_pf0_100,    xc_ipf0_100,    5, 14, 0, "l100-ipf0"  },
    { x_pf0_100,    xc_pf0_100,     2, 15, 0, "l100-f0"    },
    { x_pf1_100,    xc_pfr_100,     3, 16, 0, "l100-fr"    },
    { x_pf1_100,    xc_pfrw_100,    4, 17, 0, "l100-frw"   },
    { x_pf1_100,    xc_ipq1_100,   17, 18, 1, "l100-ipq1"  },
    { x_pf1_100,    xc_iqn1_100,   19, 19, 0, "l100-iqn1"  } };
#define NCMAX 20

/* r6: lean pools for the 16 coverage sizes -- the family/knob variants
 * that actually win on this node (full-pool insurance candidates exist only
 * for the scored sizes, to keep the extra instantiations compile-cheap).
 * pf ids match the legacy meaning; the race + wisdom arbitrate per host.
 * r7: + ipk1 (rank last) -- the L >= 91 volumes are beyond-L3 like L=100,
 * where the c-flush bypass carries the cell (gen_powp r6 + this r7). */
#define GEN_LEAN_TBL(L)                                                      \
static const struct candpl g_cl##L[] = {                                     \
    { x_pf1_##L, xc_ip1_##L,   1, 0, 0, "l" #L "-ip1"  },                    \
    { x_pf0_##L, xc_ip0_##L,   0, 1, 0, "l" #L "-ip0"  },                    \
    { x_pf1_##L, xc_ip2_##L,  13, 2, 0, "l" #L "-ip2"  },                    \
    { x_pf1_##L, xc_ipr1_##L, 14, 3, 0, "l" #L "-ipr1" },                    \
    { x_pf1_##L, xc_ipp1_##L, 16, 4, 1, "l" #L "-ipp1" },                    \
    { x_pf0_##L, xc_ipp0_##L, 15, 5, 1, "l" #L "-ipp0" },                    \
    { x_pf1_##L, xc_ipk1_##L, 18, 6, 1, "l" #L "-ipk1" } };
GEN_LEAN_TBL(44)
GEN_LEAN_TBL(48)
GEN_LEAN_TBL(52)
GEN_LEAN_TBL(55)
GEN_LEAN_TBL(56)
GEN_LEAN_TBL(63)
GEN_LEAN_TBL(65)
GEN_LEAN_TBL(72)
GEN_LEAN_TBL(75)
GEN_LEAN_TBL(77)
GEN_LEAN_TBL(88)
GEN_LEAN_TBL(91)
GEN_LEAN_TBL(99)
GEN_LEAN_TBL(104)
GEN_LEAN_TBL(112)
GEN_LEAN_TBL(117)

/* r6: one row per supported size -- factors (stage-2 module first in the
 * name f1 x f2), plane-scratch pitch (GPP rule: multiple of 4 complex,
 * GPP/4 odd cache lines), candidate pool. */
struct sizetab { int L, f1, f2, gpp; const struct candpl *cd; int nc; };
#define GEN_SZ(LV, F1, F2, GPPV, TBL)                                        \
    { LV, F1, F2, GPPV, TBL, (int)(sizeof TBL / sizeof TBL[0]) }
static const struct sizetab g_sizes[] = {
    GEN_SZ( 40,  8,  5,  44, g_c40),
    GEN_SZ( 50, 25,  2,  52, g_c50),
    GEN_SZ( 80, 16,  5,  84, g_c80),
    GEN_SZ(100, 25,  4, 108, g_c100),
    GEN_SZ( 44, 11,  4,  44, g_cl44),
    GEN_SZ( 48, 16,  3,  52, g_cl48),
    GEN_SZ( 52, 13,  4,  52, g_cl52),
    GEN_SZ( 55, 11,  5,  60, g_cl55),
    GEN_SZ( 56,  8,  7,  60, g_cl56),
    GEN_SZ( 63,  9,  7,  68, g_cl63),
    GEN_SZ( 65, 13,  5,  68, g_cl65),
    GEN_SZ( 72,  9,  8,  76, g_cl72),
    GEN_SZ( 75, 25,  3,  76, g_cl75),
    GEN_SZ( 77, 11,  7,  84, g_cl77),
    GEN_SZ( 88, 11,  8,  92, g_cl88),
    GEN_SZ( 91, 13,  7,  92, g_cl91),
    GEN_SZ( 99, 11,  9, 100, g_cl99),
    GEN_SZ(104, 13,  8, 108, g_cl104),
    GEN_SZ(112, 16,  7, 116, g_cl112),
    GEN_SZ(117, 13,  9, 124, g_cl117),
};
static const struct sizetab *sz_find(int L)
{
    for (size_t i = 0; i < sizeof g_sizes / sizeof g_sizes[0]; ++i)
        if (g_sizes[i].L == L) return &g_sizes[i];
    return NULL;
}

/* install one candidate's verdict on the plan + refresh the description */
static void install_pick(fft3d_plan *p, const struct candpl *cd, int pick,
                         int with_chain, const char *how)
{
    p->fn = cd[pick].fn;
    if (with_chain) { p->cfn = cd[pick].cfn; p->dm = cd[pick].dm; }
    const struct sizetab *sz_ = sz_find(p->L);
    snprintf(g_desc, sizeof g_desc,
             "GT-PFA %dx%d two-sweep (exact-tw modules)%s; pick: %s%s (B=%d)",
             sz_ ? sz_->f1 : p->L, sz_ ? sz_->f2 : 1,
             p->cfn ? (p->dm ? " + owned chain (deferred NR map)"
                             : " + owned chain (NR map)") : "",
             cd[pick].nm, how, p->batch);
}

/* r9: relative spread (max-min)/min of one candidate's per-round trial
 * times -- the noise measure for the pick/storage gates below.  Unfilled
 * rounds (t == 0) are skipped; no valid samples reads as infinitely noisy. */
static double trial_spread(const double *t, int nr)
{
    double lo = 1e300, hi = 0.0;
    for (int r = 0; r < nr; ++r) {
        if (t[r] <= 0.0) continue;
        if (t[r] < lo) lo = t[r];
        if (t[r] > hi) hi = t[r];
    }
    return (lo < 1e300) ? (hi - lo) / lo : 1e300;
}
/* a verdict is "tight" (storable, upset-capable) below this relative
 * spread; above it the window is churning and no trial is trusted */
#define GEN_TIGHT 0.10
/* minimum margin for a trial winner to displace the rank-0 prior: margins
 * below this are window coin flips on every host measured (the r4-r8
 * held-lease sessions drifted +-8-15% between minutes-apart reps while the
 * quiet-floor gaps among the leading families are 1-3%); the genuine
 * cross-host upsets this pool exists for (ipk1 on CLX/SPR, NT under
 * contention) have measured 9-16% margins and clear it comfortably */
#define GEN_UPSET_MIN 0.06

/* one long-run chain-step timing arm (volume-major, warm step first per
 * volume, PS timed steps, times summed over volumes and normalized per
 * step) -- shared by the r7 challenger playoff and the r9 upset
 * confirmation so both judge on the same evidence shape */
static double chain_longrun(const struct candpl *cd, int c, double *tout,
                            const double *tcf, int nv, size_t VD,
                            double *M, double *P, int PS)
{
    double tsum = 0.0;
    for (int b = 0; b < nv; ++b) {
        double       *sb = tout + (size_t)b * VD;
        const double *cb = tcf  + (size_t)b * VD;
        cd[c].cfn(sb, sb, cb, 1, M, P);                  /* warm */
        double t0 = now_s();
        for (int r = 0; r < PS; ++r)
            cd[c].cfn(sb, sb, cb, 1, M, P);
        tsum += now_s() - t0;
    }
    return tsum / PS;
}

/* gate + race; installs p->fn (or leaves it NULL for the dense fallback) */
static void tune(fft3d_plan *p)
{
    const int L = p->L;
    const size_t VD = (size_t)2 * L * L * L;
    const struct sizetab *sz = sz_find(L);
    if (!sz) return;                                  /* dense fallback */
    const struct candpl *cd = sz->cd;
    const int nc = sz->nc;

    /* r3, ADOPTED from gen_race: the tune() verdict (winner + chain-gate
     * outcome) is persisted per (host, L, B-bucket, candidate-set sig).  A
     * wisdom hit installs it with NO arena, NO refnd, NO race: warm create()
     * is one file read (~ms vs the brief's 50 ms budget; cold was up to
     * 6.5 s).  Safe: every family computes bit-identical results and only
     * gate-passed verdicts are stored.  Forcing envs bypass wisdom. */
    const int wisdom_ok = !getenv("GENPFL_PF") && !getenv("GENPFL_NOFUSE");
    char fullkey[GR_KEY_MAX + 16] = { 0 };
    if (wisdom_ok) {
        gr_cand sc[NCMAX];
        memset(sc, 0, sizeof sc);
        for (int c = 0; c < nc; ++c) sc[c].name = cd[c].nm;
        char key[GR_KEY_MAX];
        gr_keyf(key, sizeof key, "gen_pfa_large", "chain7", L,
                gr_bucket(p->batch));         /* r9: tag bump, race shape +
                                                 storage policy changed  */
        snprintf(fullkey, sizeof fullkey, "%s#%08x", key, gr_sig(sc, nc));
        char wname[GR_NAME_MAX];
        if (p->M && gr_wisdom_get_str(fullkey, wname, sizeof wname)) {
            char *sep = strrchr(wname, '.');
            if (sep && (!strcmp(sep, ".ch") || !strcmp(sep, ".x"))) {
                const int ch = sep[1] == 'c';
                *sep = 0;
                for (int c = 0; c < nc; ++c)
                    if (!strcmp(cd[c].nm, wname)) {
                        install_pick(p, cd, c, ch, " (wisdom)");
                        if (getenv("GENPFL_VERBOSE"))
                            fprintf(stderr, "gen_pfa_large L=%d: wisdom hit"
                                    " %s%s\n", L, wname, ch ? ".ch" : ".x");
                        return;
                    }
            }
        }
    }

    int    live[NCMAX];
    double tc[NCMAX];
    double tr[NCMAX][4];                  /* r9: per-round trial times */
    for (int c = 0; c < nc; ++c) {
        live[c] = 1; tc[c] = 1e300;
        for (int r = 0; r < 4; ++r) tr[c][r] = 0.0;
    }

    { const char *e = getenv("GENPFL_PF");            /* monitor forcing */
      if (e) { int v = atoi(e);
               for (int c = 0; c < nc; ++c) if (cd[c].pf != v) live[c] = 0;
               int any = 0;
               for (int c = 0; c < nc; ++c) any |= live[c];
               if (!any) for (int c = 0; c < nc; ++c) live[c] = 1; } }

    /* arena: stream realistically at large batch, cap the footprint.
     * tcf is a DISTINCT c-field buffer (r3 fix): passing tin as both the
     * state and c made ipm's back-to-back state+c loads hit the same
     * line, halving its apparent read traffic -- the race then picked
     * l50-ipm0 while the real chain ran +15% (574.8 vs ~500 us same
     * window).  Race what is graded, including the STREAMS. */
    const int cap = (int)(1 + (size_t)32 * 1024 * 1024 / (VD * 8));
    const int nv  = p->batch < cap ? p->batch : cap;
    void *ri = NULL, *ro = NULL, *rc = NULL, *r0 = NULL, *r1 = NULL;
    if (posix_memalign(&ri, 64, (size_t)nv * VD * sizeof(double)) ||
        posix_memalign(&ro, 64, (size_t)nv * VD * sizeof(double)) ||
        posix_memalign(&rc, 64, (size_t)nv * VD * sizeof(double)) ||
        posix_memalign(&r0, 64, VD * sizeof(double)) ||
        (nv > 1 && posix_memalign(&r1, 64, VD * sizeof(double)))) {
        free(ri); free(ro); free(rc); free(r0);
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
    if (nv > 1)          /* gate the LAST volume too: P-reuse bugs cannot hide */
        refnd(L, (const double _Complex *)(tin + (size_t)(nv - 1) * VD),
              (double _Complex *)refN);

    for (int c = 0; c < nc; ++c) {
        if (!live[c]) continue;
        memset(tout, 0, (size_t)nv * VD * sizeof(double));
        cd[c].fn(tin, tout, nv, p->P);
        if (!rel_ok(tout, ref0, VD)) live[c] = 0;
        if (nv > 1 && live[c] &&
            !rel_ok(tout + (size_t)(nv - 1) * VD, refN, VD)) live[c] = 0;
    }
    /* min over interleaved rounds (wallaby's fast/slow toggle, L45 lesson).
     * Race the CHAIN step when the mid volume exists -- every graded case is
     * chained, and the poke policy that wins in-place execute is not the one
     * that wins the out-of-place fused step (measured: l40 B=8 chain pf0
     * 340 vs pf1 274 us/xform while the execute arena preferred them within
     * 6%).  r3: the step runs IN PLACE on tout with the distinct tcf as c --
     * exactly the graded steady state (fft3d_chain calls cfn(out, out, c));
     * the old out-of-place cur->dst arena touched one volume set too many
     * and the tin-as-c aliasing favoured ipm (see above).  State values stay
     * bounded across trials: every family re-maps (|x|<1) before the FFT or
     * right after it. */
    /* r4: the graded chain is VOLUME-MAJOR now (all m steps on one volume
     * while its state+c slice is cache-hot -- see fft3d_chain), so the race
     * times that schedule: per volume, one UNMEASURED warm step (the graded
     * chain's 1-cold-in-m fraction, not 1-in-R), then R timed steps on the
     * same volume; volumes outer, times summed.  Racing the old step-major
     * shape would rank candidates on a working set the chain no longer has
     * (at 40/50 the resident families ipf/f* may now win -- gen_powp's
     * measured residency rule -- and the race is the arbiter). */
    memcpy(tout, tin, (size_t)nv * VD * sizeof(double));
    const size_t vbytes = VD * sizeof(double);
    const int R  = (vbytes <= ((size_t)2 << 20)) ? 8
                 : (vbytes <= ((size_t)8 << 20)) ? 6 : 4;
    const int NR = (nv >= 8) ? 3 : 4;
    for (int round = 0; round < NR; ++round)
        for (int c = 0; c < nc; ++c) {
            if (!live[c]) continue;
            double t;
            if (p->M) {
                double tsum = 0.0;
                for (int b = 0; b < nv; ++b) {
                    double       *sb = tout + (size_t)b * VD;
                    const double *cb = tcf  + (size_t)b * VD;
                    cd[c].cfn(sb, sb, cb, 1, p->M, p->P);    /* warm */
                    double t0 = now_s();
                    for (int r = 0; r < R; ++r)
                        cd[c].cfn(sb, sb, cb, 1, p->M, p->P);
                    tsum += now_s() - t0;
                }
                t = tsum / R;
            } else {
                cd[c].fn(tin, tout, nv, p->P);               /* warm */
                double t0 = now_s();
                for (int r = 0; r < R; ++r)
                    cd[c].fn(tin, tout, nv, p->P);
                t = (now_s() - t0) / R;
            }
            tr[c][round] = t;                        /* r9: keep per round */
            if (t < tc[c]) tc[c] = t;
        }
    /* r7 CHALLENGER PLAYOFF (ADOPTED from gen_powp gen_r6): a candidate
     * whose benefit accrues across steps (the ipk1 c-flush keeps L3 clean
     * for LATER steps' state re-reads) is systematically under-read by the
     * R-step trials -- gen_powp measured ipk1 +5.3% in a 4-step cold race
     * vs -0.6..-16% in 5/5 graded m=64 pairs.  When the rank-0 live
     * candidate is not the trial best but within 15%, re-judge the two on
     * long steady runs (PS steps per arm per round, alternating, min over
     * 3 rounds) and feed the results back into tc[] via min() -- fair to
     * both arms; the 3% simplest-first hysteresis below still decides. */
    if (p->M) {
        int tbest = -1, chal = -1;
        for (int c = 0; c < nc; ++c)
            if (live[c]) {
                if (tbest < 0 || tc[c] < tc[tbest]) tbest = c;
                if (chal < 0 || cd[c].rank < cd[chal].rank) chal = c;
            }
        if (tbest >= 0 && chal >= 0 && chal != tbest &&
            tc[chal] < tc[tbest] * 1.15) {
            const int PS = (vbytes <= ((size_t)8 << 20)) ? 24 : 12;
            const int arm[2] = { tbest, chal };
            for (int round = 0; round < 3; ++round)
                for (int a = 0; a < 2; ++a) {
                    const int c = arm[a];
                    double t = chain_longrun(cd, c, tout, tcf, nv, VD,
                                             p->M, p->P, PS);
                    if (t < tc[c]) tc[c] = t;
                }
        }
    }
    int best = -1;
    for (int c = 0; c < nc; ++c)
        if (live[c] && (best < 0 || tc[c] < tc[best])) best = c;
    if (best >= 0) {
        int pick = best;                              /* 3% simplest-first */
        for (int c = 0; c < nc; ++c)
            if (live[c] && tc[c] <= tc[best] * 1.03 &&
                cd[c].rank < cd[pick].rank) pick = c;
        /* r9 NOISE GATE (PMU audit avenue 1: plan-time pick instability --
         * a lucky create() once picked gen_powp's L=25 path worth 25% and
         * nothing persisted it; my own r7 cold race installed ipk1 from one
         * churning window).  Two rules, both keyed to the measured
         * per-round trial spread:
         *   1. UPSET RULE: a candidate may displace the rank-0 prior only
         *      if the window is tight, its margin over rank 0 exceeds
         *      max(the larger of the two spreads, GEN_UPSET_MIN), and it
         *      confirms the minimum margin on one fresh long-run
         *      alternation.  Otherwise the deterministic prior (the
         *      held-lease quiet-floor evidence encoded in the rank order)
         *      wins: a churning window cannot flip the pick, so
         *      consecutive cold create() cycles pick identically.
         *   2. STORAGE RULE (at the put below): only tight verdicts
         *      persist.  A reverted upset is a forced default, not a
         *      measurement -- installed for THIS plan, never stored, so
         *      the next create() re-races (re-race, never trust, a noisy
         *      trial).  All families are bit-identical, so neither rule
         *      can affect correctness or cross-process repeatability. */
        int rank0 = -1;
        for (int c = 0; c < nc; ++c)
            if (live[c] && (rank0 < 0 || cd[c].rank < cd[rank0].rank))
                rank0 = c;
        int reverted = 0;
        double nse = trial_spread(tr[pick], NR);
        if (rank0 >= 0 && pick != rank0) {
            const double n0  = trial_spread(tr[rank0], NR);
            const double nn  = nse > n0 ? nse : n0;
            const double bar = nn > GEN_UPSET_MIN ? nn : GEN_UPSET_MIN;
            int ok = nn <= GEN_TIGHT && tc[rank0] >= tc[pick] * (1.0 + bar);
            if (ok && p->M) {
                /* CONFIRMATION: one fresh long-run alternation the upset
                 * has not seen -- it must hold the minimum margin again.
                 * A single window reading, however decisive it looked,
                 * never displaces the banked rank prior on its own. */
                const int PS = (vbytes <= ((size_t)8 << 20)) ? 24 : 12;
                const double tu = chain_longrun(cd, pick, tout, tcf, nv,
                                                VD, p->M, p->P, PS);
                const double tp = chain_longrun(cd, rank0, tout, tcf, nv,
                                                VD, p->M, p->P, PS);
                ok = tp >= tu * (1.0 + GEN_UPSET_MIN);
            }
            if (!ok) {
                pick = rank0;                  /* prior wins under noise */
                reverted = 1;
                nse = n0;
            } else
                nse = nn;                      /* confirmed upset: trusted */
        }
        const int tight = !reverted && nse <= GEN_TIGHT;
        /* gate the picked chain step against execute + the driver's own
         * scalar map on volume 0 (tin doubles as the c field); on any
         * mismatch fall back to the always-correct execute+scalar path.
         * Deferred-map (ipm) picks gate the full m=2 chain composition
         * (execute, one ipm step, trailing map_vec) instead of one step. */
        int chain_ok = 0;
        if (p->M && !getenv("GENPFL_NOFUSE")) {
            void *e1 = NULL, *e2 = NULL;
            if (!posix_memalign(&e1, 64, VD * sizeof(double)) &&
                !posix_memalign(&e2, 64, VD * sizeof(double))) {
                double *zed = e1, *exp_ = e2;
                if (cd[pick].dm) {
                    cd[pick].fn(tin, zed, 1, p->P);
                    map_scalar(zed, tin, exp_, VD / 2);      /* x1        */
                    cd[pick].fn(exp_, zed, 1, p->P);
                    map_scalar(zed, tin, exp_, VD / 2);      /* x2 (ref)  */
                    cd[pick].fn(tin, tout, 1, p->P);
                    cd[pick].cfn(tout, tout, tin, 1, p->M, p->P);
                    map_vec(tout, tin, tout, VD / 8);        /* x2 (got)  */
                    map_span_tail(tout, tin, tout, VD);      /* odd-L tail */
                } else {
                    cd[pick].fn(tin, zed, 1, p->P);
                    map_scalar(zed, tin, exp_, VD / 2);
                    cd[pick].cfn(tin, tout, tin, 1, p->M, p->P);
                }
                chain_ok = rel_ok(tout, exp_, VD);
            }
            free(e1); free(e2);
        }
        install_pick(p, cd, pick, chain_ok, "");
        if (wisdom_ok && fullkey[0] && tight) {       /* persist the verdict */
            char val[GR_NAME_MAX];
            snprintf(val, sizeof val, "%s%s", cd[pick].nm,
                     chain_ok ? ".ch" : ".x");
            gr_wisdom_put_str(fullkey, val);
        } else if (wisdom_ok && fullkey[0] && getenv("GENPFL_VERBOSE"))
            fprintf(stderr, "gen_pfa_large L=%d: verdict %s NOT stored "
                    "(%s, spread %.1f%%)\n", L, cd[pick].nm,
                    reverted ? "noisy upset reverted to rank 0"
                             : "trial spread over gate",
                    nse < 1e300 ? nse * 100 : 999.0);
    }
    if (getenv("GENPFL_VERBOSE")) {
        for (int c = 0; c < nc; ++c) {
            fprintf(stderr,
                    "gen_pfa_large L=%d: %-9s %s %.1f us/vol (spread %.1f%%"
                    ", rounds", L, cd[c].nm, live[c] ? "ok " : "OUT",
                    live[c] ? tc[c] * 1e6 / nv : 0.0,
                    live[c] ? trial_spread(tr[c], NR) * 100 : 0.0);
            for (int r = 0; r < NR; ++r)
                fprintf(stderr, " %.1f", tr[c][r] * 1e6 / nv);
            fprintf(stderr, ")\n");
        }
    }
    free(ri); free(ro); free(rc); free(r0); free(r1);
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
    const struct sizetab *szc = sz_find(L);
    const int pp = szc ? szc->gpp : 0;
    if (pp &&
        posix_memalign(&p->rawP, 4096,
                       (size_t)L * pp * 2 * sizeof(double)) == 0) {
        p->P = (double *)p->rawP;
        memset(p->P, 0, (size_t)L * pp * 2 * sizeof(double));
        if (posix_memalign(&p->rawM, 4096,
                           ((size_t)(2 * L * L + 8) * L + 296)
                               * sizeof(double)) == 0)
            p->M = (double *)p->rawM + 296;       /* fused chain mid volume:
                                                     padded planes, base
                                                     2368 B off the page   */
        tune(p);
    }
#endif
    if (!p->fn && !dense_setup(p)) {                  /* fallback of last resort */
        free(p->rawP); free(p->w); free(p->tmp); free(p);
        return NULL;
    }
    return p;
}

void fft3d_execute(fft3d_plan *p, const double _Complex *in, double _Complex *out)
{
    if (p->fn) p->fn((const double *)in, (double *)out, p->batch, p->P);
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
    const size_t VDv = (size_t)2 * p->L * p->L * p->L;   /* doubles/volume */
    const size_t VD  = VDv * (size_t)p->batch;
    if (m < 1) { memmove(out, x0, VD * sizeof(double)); return; }
#ifdef __AVX512F__
    /* r4: VOLUME-MAJOR schedule (adopted from gen_dense_prime / gen_rader /
     * gen_layout's corpus-consensus chain shape).  Volumes are independent
     * in the chain algebra -- z_b = FFT3(x_b) + c_b per volume, mapped per
     * element -- so each volume runs ALL m steps while its state + c slice
     * (2-4 MB at L=40/50) stays cache-resident, instead of every step
     * churning the whole batch working set (16 MB at 40/50, shared-L3
     * hostage).  Per-volume op order is unchanged: outputs bit-identical
     * to the old step-major schedule. */
    if (p->cfn && p->dm) {
        /* deferred-map schedule (r3, ipm/ipp): between steps the state
         * volume holds the RAW FFT output z' = FFT3(x_s); the map z/(1+|z|)
         * with z = z' + c is applied by the NEXT step's phase-1 (at its
         * loads for ipm, in a per-plane prepass for ipp), and once by
         * map_vec after the last step.  m FFT passes + m map applications,
         * exactly the chain's algebra, zero separate map passes in steady
         * state. */
        for (int b = 0; b < p->batch; ++b) {
            const double *xb = (const double *)x0 + (size_t)b * VDv;
            const double *cb = (const double *)c  + (size_t)b * VDv;
            double       *ob = (double *)out      + (size_t)b * VDv;
            p->fn(xb, ob, 1, p->P);
            for (int s = 1; s < m; ++s)
                p->cfn(ob, ob, cb, 1, p->M, p->P);
            map_vec(ob, cb, ob, VDv / 8);
            map_span_tail(ob, cb, ob, VDv);      /* odd-L: last 1-3 complex */
        }
        return;
    }
#endif
    if (p->cfn) {
        for (int b = 0; b < p->batch; ++b) {
            const double *xb = (const double *)x0 + (size_t)b * VDv;
            const double *cb = (const double *)c  + (size_t)b * VDv;
            double       *ob = (double *)out      + (size_t)b * VDv;
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
#define FN(n) GCAT(n, GCAT(_, GENL))

/* r6: GEN_LEAN instantiations (the coverage sizes) emit only the ip + ipp
 * chain families and compile the two heavy bodies (p1body, p2) ONCE as
 * noinline functions -- ~4 codelet expansions per size instead of ~50.
 * The scored sizes keep always_inline: their generated code is unchanged. */
#ifdef GEN_LEAN
#define GENPFA_HEAVY static __attribute__((noinline))
#else
#define GENPFA_HEAVY static inline __attribute__((always_inline))
#endif

/* r6: per-size noinline map spans with COMPILE-TIME counts.  In r5 gcc
 * emitted map_vec.constprop clones (constant trip count) and the chain fns
 * CALLED them; adding 16 coverage sizes doubled the unit and flipped gcc
 * to inlining the map body into every chain fn instead -- measured +0.9%
 * at L=100 B=1 (4/4 held-lease pairs, medians 4766-4781 vs 4716-4741).
 * These wrappers reproduce the r5 shape by construction, independent of
 * the inliner's unit-growth mood: map_vec inlines INTO the wrapper with a
 * constant count (= the constprop clone), the chain fn makes one call.
 * The odd-L masked tail (1-3 complex, see map_span_tail) lives here too. */
static __attribute__((noinline))
void FN(map_vol)(const double *z, const double *c, double *o)
{
    map_vec(z, c, o, VDL / 8);
#if GENL % 2
    map_span_tail(z, c, o, VDL);
#endif
}
static __attribute__((noinline))
void FN(map_pln)(const double *z, const double *c, double *o)
{
    map_vec(z, c, o, PLNDL / 8);
#if PLNDL % 8
    map_span_tail(z, c, o, PLNDL);
#endif
}

/* phase 1, ONE x-plane: z transform (lanes = 4 y-rows, granule transposes)
 * into plane scratch pl[y][kz], then y transform (lanes = 4 kz, contiguous)
 * into out[x][ky][kz].  GENL % 4 == 2 (L=50) adds one OVERLAPPING group per
 * subpass (recompute of 2 lanes; every store is idempotent).  r4: the body
 * is factored out over (px, mx) plane bases so p1pm can feed it a mapped
 * scratch plane. */
GENPFA_HEAVY
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
void FN(p1)(const double *restrict in, double *restrict out,
            double *restrict pld, int x, const long mpln)
{
    FN(p1body)(in + (size_t)x * PLNDL, out + (size_t)x * mpln, pld);
}

/* p1 with a PLANE-GRANULARITY deferred-map prepass (r4, the ipp family):
 * the state entering this step is the previous step's raw FFT output z'
 * (the ipm schedule); the map v = map(z' + c) runs over the whole plane
 * FIRST, as map_vec's perfectly sequential loop, into the caller's scratch
 * plane mp (L2-resident, never DRAM), and the ordinary p1 body consumes mp.
 * Same algebra and op order as ipm/map_vec -- bit-identical outputs -- but
 * the ladder's uops meet p1's TRNC/codelet port-5 pressure only at the
 * plane seam instead of per-granule (the measured ipm failure at L=100).
 * Traffic per step matches ipm's accounting: the state is read once (by the
 * prepass), written once (by the y-subpass, into lines the prepass just
 * pulled into L2 -- no fresh RFO), and the separate map pass is gone.
 * In-place safe: mp is a copy, so overwriting state plane x is fine. */
static inline __attribute__((always_inline))
void FN(p1pm)(const double *restrict in, double *restrict out,
              double *restrict pld, int x, const long mpln,
              const double *restrict cf, double *restrict mp)
{
    FN(map_pln)(in + (size_t)x * PLNDL, cf + (size_t)x * PLNDL, mp);
    FN(p1body)(mp, out + (size_t)x * mpln, pld);
}

#ifndef GEN_LEAN
/* p1pm with the c stream NTA-prefetched (r5, ipq): see map_vec_nta. */
static inline __attribute__((always_inline))
void FN(p1pmq)(const double *restrict in, double *restrict out,
               double *restrict pld, int x, const long mpln,
               const double *restrict cf, double *restrict mp)
{
    map_vec_nta(in + (size_t)x * PLNDL, cf + (size_t)x * PLNDL, mp,
                PLNDL / 8);
    FN(p1body)(mp, out + (size_t)x * mpln, pld);
}
#endif /* !GEN_LEAN (p1pmq) */

/* p1pm with the c lines flushed after use (r5, ipk): see map_vec_cfl.
 * r7: emitted for the LEAN coverage sizes too -- the L >= 91 volumes are
 * 12-26 MB, the same beyond-L3 c-pollution regime the flush targets.
 * Odd-L note: PLNDL % 8 != 0 there; the masked map_span_tail below keeps
 * the last 1-3 complex of the prepass plane mapped (the r6 lesson). */
static inline __attribute__((always_inline))
void FN(p1pmk)(const double *restrict in, double *restrict out,
               double *restrict pld, int x, const long mpln,
               const double *restrict cf, double *restrict mp)
{
    map_vec_cfl(in + (size_t)x * PLNDL, cf + (size_t)x * PLNDL, mp,
                PLNDL / 8);
#if PLNDL % 8
    map_span_tail(in + (size_t)x * PLNDL, cf + (size_t)x * PLNDL, mp, PLNDL);
#endif
    FN(p1body)(mp, out + (size_t)x * mpln, pld);
}

#ifndef GEN_LEAN
/* p1 with the graded map APPLIED TO THE Z-SUBPASS LOADS (r3, the ipm
 * deferred-map family): the state entering this step is the PREVIOUS
 * step's raw FFT output z' (no +c, no map); every element is loaded here
 * exactly once, so v = map(z' + c) right after the granule load IS the
 * chain map, in map_vec's exact op order (bit-identical results).  The
 * ladder latency sits behind loads feeding compute -- thousands of
 * independent chains per plane the OoO window runs ahead of -- NOT gating
 * stores in the miss-bound x-pass (the measured ipf/ipe failure at
 * L=100).  Deletes the separate map pass entirely.  Overlap groups at
 * GENL % 4 != 0 recompute the map on 2 lanes: pure function, idempotent. */
static inline __attribute__((always_inline))
void FN(p1m)(const double *restrict in, double *restrict out,
             double *restrict pld, int x, const long mpln,
             const double *restrict cf)
{
    const double *px = in  + (size_t)x * PLNDL;
    const double *pc = cf  + (size_t)x * PLNDL;
    double       *mx = out + (size_t)x * mpln;

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
#endif /* !GEN_LEAN (p1m) */

/* phase 2: x transform in place, tiled over the FLAT (y,z) index.  NPL is
 * a multiple of 4 for the scored sizes: no tail there.  r6 odd-L sizes
 * have NPL % 4 == 1: the last 4 flat columns are STASHED raw before the
 * in-place sweep, full tiles cover columns 0..NPL-2, and one final tile
 * transforms the stash into columns NPL-4..NPL-1 -- the 3 recomputed
 * columns read RAW stashed inputs, so the in-place recompute is idempotent
 * (gen_powp's odd-L^2 stash idea).  The codelet reads all GENL inputs
 * before its first store.  pf = poke DISTANCE (r3: was a 0/1 flag): pokes
 * the GENL read streams pf cache lines ahead (more streams than the L2
 * prefetcher tracks). */
GENPFA_HEAVY
void FN(p2)(double *io, const long pln, const int pf)
{
#if NPL % 4
    vec S_[GENL];
    {
        const double *s_ = io + (size_t)(NPL - 4) * 2;
        for (int n_ = 0; n_ < GENL; ++n_)
            S_[n_] = LDU(s_ + (size_t)n_ * pln);
    }
#endif
    for (int t = 0; t < NPL / 4; ++t) {
        double *s_ = io + (size_t)t * 8;
        if (pf) {
            _Pragma("GCC unroll 100")
            for (int n_ = 0; n_ < GENL; ++n_)
                __builtin_prefetch(s_ + (size_t)n_ * pln + 8 * pf, 0, 3);
        }
#define LD3(n)    LDU(s_ + (size_t)(n) * pln)
#define ST3(k, v) STU(s_ + (size_t)(k) * pln, (v))
        PFAL(LD3, ST3);
#undef LD3
#undef ST3
    }
#if NPL % 4
    {
        double *d_ = io + (size_t)(NPL - 4) * 2;
        __asm__("" : "+r"(d_));
#define LD3(n)    S_[n]
#define ST3(k, v) STU(d_ + (size_t)(k) * pln, (v))
        PFAL(LD3, ST3);
#undef LD3
#undef ST3
    }
#endif
}

#ifndef GEN_LEAN
/* phase 2 for the fused chain: x transform OUT of place, mid -> dst, with
 * z = v + c and the map z/(1+|z|) applied at every store -- deletes the
 * driver's separate full-volume map pass (read z + read c + write state).
 * The mid volume M is read at the PADDED plane pitch MPLND (odd cache-line
 * count, +64 B/plane): with M at out's exact layout and a page-aligned
 * base, every M load was page-offset-congruent with an in-flight out store
 * (and every p1 M-store with an out read) -- systematic 4K false
 * dependencies, measured 11-13 ms/step at L=100 vs 8.2 for the UNFUSED
 * path.  pf pokes the mid read streams / dst write streams (RFO) one line
 * ahead; c and the tile bases advance contiguously (hardware-prefetchable). */
static inline __attribute__((always_inline))
void FN(p2c)(const double *restrict mid, double *restrict dst,
             const double *restrict cf, const int pfr, const int pfw)
{
    for (int t = 0; t < NPL / 4; ++t) {
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
#define LD3(n)    LDU(s_ + (size_t)(n) * MPLND)
#define ST3(k, v) do {                                                       \
    vec z_ = (v) + LDU(g_ + (size_t)(k) * PLNDL);                            \
    STU(d_ + (size_t)(k) * PLNDL, map_step_v(z_));                           \
} while (0)
        PFAL(LD3, ST3);
#undef LD3
#undef ST3
    }
}

/* phase 2 IN PLACE with the chain map fused into the stores (r2, BORROWED
 * from gen_powp gen_r1's ipf family, which they built on this engine): no
 * mid volume, no separate map pass.  Stores land on the lines the tile
 * just read (no fresh RFO stream, unlike the out-of-place fused p2c); c
 * is the only extra read set.  In-place safe: the codelet reads all GENL
 * inputs (stage 1 into T_) before its first store.  pf pokes the
 * read+write lines with WRITE intent one line ahead. */
static inline __attribute__((always_inline))
void FN(p2ipf)(double *io, const double *restrict cf, const int pf)
{
    for (int t = 0; t < NPL / 4; ++t) {
        double       *s_ = io + (size_t)t * 8;
        const double *g_ = cf + (size_t)t * 8;
        if (pf) {
            _Pragma("GCC unroll 100")
            for (int n_ = 0; n_ < GENL; ++n_)
                __builtin_prefetch(s_ + (size_t)n_ * PLNDL + 8, 1, 3);
        }
#define LD3(n)    LDU(s_ + (size_t)(n) * PLNDL)
#define ST3(k, v) do {                                                       \
    vec z_ = (v) + LDU(g_ + (size_t)(k) * PLNDL);                            \
    STU(s_ + (size_t)(k) * PLNDL, map_step_v(z_));                           \
} while (0)
        PFAL(LD3, ST3);
#undef LD3
#undef ST3
    }
}

/* phase 2 in place with a MAP EPILOGUE per tile (r2, new): the FFT stores
 * stay plain (no map ladder gating the DAG's stores -- the measured
 * failure of ipf at L=100, where each store waiting on a ~40-cycle
 * rsqrt/rcp NR chain collapses the x-pass's memory-level parallelism),
 * then the GENL just-written L1-hot lines are immediately re-read, mapped
 * with c, and re-stored in an independent-iteration loop the OoO window
 * can pipeline.  Deletes the separate map pass's full-volume DRAM round
 * trip (read z + read c + write state).  pf pokes next tile's state lines
 * (write intent) and THIS tile's c lines (the epilogue needs them ~1k
 * cycles after tile start). */
static inline __attribute__((always_inline))
void FN(p2ipe)(double *io, const double *restrict cf, const int pf)
{
    for (int t = 0; t < NPL / 4; ++t) {
        double       *s_ = io + (size_t)t * 8;
        const double *g_ = cf + (size_t)t * 8;
        if (pf) {
            _Pragma("GCC unroll 100")
            for (int n_ = 0; n_ < GENL; ++n_) {
                __builtin_prefetch(s_ + (size_t)n_ * PLNDL + 8, 1, 3);
                __builtin_prefetch(g_ + (size_t)n_ * PLNDL, 0, 3);
            }
        }
#define LD3(n)    LDU(s_ + (size_t)(n) * PLNDL)
#define ST3(k, v) STU(s_ + (size_t)(k) * PLNDL, (v))
        PFAL(LD3, ST3);
#undef LD3
#undef ST3
        _Pragma("GCC unroll 100")
        for (int n_ = 0; n_ < GENL; ++n_) {
            vec z_ = LDU(s_ + (size_t)n_ * PLNDL) +
                     LDU(g_ + (size_t)n_ * PLNDL);
            STU(s_ + (size_t)n_ * PLNDL, map_step_v(z_));
        }
    }
}

#if GENL % 4 == 0
/* p1 with the y-subpass stores NON-TEMPORAL (vmovntpd), r2.  Legal only
 * when GENL % 4 == 0 (row stride 2*GENL*8 B is then a whole number of
 * lines, so every ST2 is a full aligned 64B line) and the volume base is
 * 64B-aligned (the caller checks and falls back).  The y-pass writes
 * every output line of the state in full, so NT deletes the RFO read of
 * the state volume -- 16 MB per chain step at L=100, where the step is
 * DRAM-bandwidth-bound.  Callers MUST sfence before reading `out`. */
static inline __attribute__((always_inline))
void FN(p1nt)(const double *restrict in, double *restrict out,
              double *restrict pld, int x, const long mpln)
{
    const double *px = in  + (size_t)x * PLNDL;
    double       *mx = out + (size_t)x * mpln;

    for (int yg = 0; yg < NFULL; ++yg) {
        const int yb = 4 * yg;
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
    }

    for (int zg = 0; zg < NFULL; ++zg) {
        const int zb = 4 * zg;
        const double *pcol = pld + 2 * zb;
        double       *mcol = mx  + 2 * zb;
        __asm__("" : "+r"(pcol), "+r"(mcol));
#define LD2(n)    LDU(pcol + (size_t)(n) * (2 * GPP))
#define ST2(k, v) STNT(mcol + (size_t)(k) * (2 * GENL), (v))
        PFAL(LD2, ST2);
#undef LD2
#undef ST2
    }
}
#endif /* GENL % 4 == 0 */
#endif /* !GEN_LEAN (p2c/p2ipf/p2ipe/p1nt) */

#ifndef GEN_LEAN
/* one full chain step for the whole batch: state cur -> state dst.
 * Per volume: phase 1 into the plan's single mid volume M (stays hot),
 * then the fused map phase 2 into dst. */
static void FN(xc_pf0)(const double *cur, double *dst, const double *cf,
                       long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, M, P, x, MPLND);
        FN(p2c)(M, dst + (size_t)b * VDL, cf + (size_t)b * VDL, 0, 0);
    }
}

static void FN(xc_pfr)(const double *cur, double *dst, const double *cf,
                       long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, M, P, x, MPLND);
        FN(p2c)(M, dst + (size_t)b * VDL, cf + (size_t)b * VDL, 1, 0);
    }
}

static void FN(xc_pfrw)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, M, P, x, MPLND);
        FN(p2c)(M, dst + (size_t)b * VDL, cf + (size_t)b * VDL, 1, 1);
    }
}
#endif /* !GEN_LEAN (xc_pf*) */

/* in-place chain step: p1 in place (each plane is fully consumed into the
 * plane scratch before being rewritten), p2 in place, then the sequential
 * vectorized map in place -- the state buffer is the ONLY volume-sized
 * object touched besides c, so at L=100 it stays L3-resident across all
 * three phases instead of ping-ponging 3 volume sets through DRAM.  The
 * mid volume M is not used at all. */
static void FN(xc_ip0)(const double *cur, double *dst, const double *cf,
                       long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, d, P, x, PLNDL);
        FN(p2)(d, PLNDL, 0);
        FN(map_vol)(d, cf + (size_t)b * VDL, d);
    }
}

static void FN(xc_ip1)(const double *cur, double *dst, const double *cf,
                       long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, d, P, x, PLNDL);
        FN(p2)(d, PLNDL, 1);
        FN(map_vol)(d, cf + (size_t)b * VDL, d);
    }
}

/* in-place chain step, poke distance 2 (r3): identical to ip1 except p2
 * pokes its read streams TWO lines ahead -- the r2 record measured 10-15%
 * pf wobble at nv=1, so the distance is now a raced knob, not a guess. */
static void FN(xc_ip2)(const double *cur, double *dst, const double *cf,
                       long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, d, P, x, PLNDL);
        FN(p2)(d, PLNDL, 2);
        FN(map_vol)(d, cf + (size_t)b * VDL, d);
    }
}

/* in-place chain step with the REVERSE map pass (r3): p2's forward sweep
 * ends with the state's tail L3-hot; the backward map reaps it and leaves
 * the head hot for the next step's p1. */
static void FN(xc_ipr1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, d, P, x, PLNDL);
        FN(p2)(d, PLNDL, 1);
        map_vec_rev(d, cf + (size_t)b * VDL, d, VDL / 8);
#if GENL % 2
        map_span_tail(d, cf + (size_t)b * VDL, d, VDL);
#endif
    }
}

#ifndef GEN_LEAN
/* deferred-map chain step (r3, ipm family): the incoming state is the
 * previous step's RAW FFT output z'; p1m applies map(z' + c) at its loads,
 * p2 runs plain, and NO map pass follows -- the volume written is again a
 * raw FFT output, mapped by the NEXT step's p1m (or by fft3d_chain's one
 * trailing map_vec after the last step).  In-place safe exactly as ip*
 * (each plane fully consumed into the scratch before being rewritten). */
static void FN(xc_ipm0)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1m)(i, d, P, x, PLNDL, g);
        FN(p2)(d, PLNDL, 0);
    }
}

static void FN(xc_ipm1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1m)(i, d, P, x, PLNDL, g);
        FN(p2)(d, PLNDL, 1);
    }
}
#endif /* !GEN_LEAN (xc_ipm*) */

/* plane-prepass deferred-map chain step (r4, ipp family): the ipm schedule
 * (incoming state = raw FFT output z', no trailing map pass) with the map
 * batched per plane through the mid volume M's base as an L2-resident
 * scratch plane -- see p1pm.  M's base is 2368 B off the page (odd line
 * count), no fixed mod-4096 relation to the state planes. */
static void FN(xc_ipp0)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1pm)(i, d, P, x, PLNDL, g, M);
        FN(p2)(d, PLNDL, 0);
    }
}

static void FN(xc_ipp1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1pm)(i, d, P, x, PLNDL, g, M);
        FN(p2)(d, PLNDL, 1);
    }
}

#ifndef GEN_LEAN
/* ipp with the prepass c stream NTA-prefetched (r5, ipq): c never enters
 * L2/L3, so the LLC holds the state volume across the whole step and p2's
 * reads hit L3 instead of DRAM.  Wins only where state+c exceed L3
 * (L=100, 80 at B>1); at 40/50 c is L3-resident reuse across all m steps
 * and bypassing it costs a full c re-read per step -- ranked last, the
 * race arbitrates. */
static void FN(xc_ipq1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1pmq)(i, d, P, x, PLNDL, g, M);
        FN(p2)(d, PLNDL, 1);
    }
}
#endif /* !GEN_LEAN (xc_ipq1) */

/* ipp with the prepass c lines CLFLUSHOPT-ed after use (r5, ipk): the
 * architectural-semantics variant of ipq's L3 bypass.  r7: lean too. */
static void FN(xc_ipk1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        const double *g = cf  + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1pmk)(i, d, P, x, PLNDL, g, M);
        FN(p2)(d, PLNDL, 1);
    }
}

#ifndef GEN_LEAN
/* ip1 with the NTA-c map pass (r5, iqn): the same c bypass for the
 * non-deferred schedule (hosts where ip beats ipp). */
static void FN(xc_iqn1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, d, P, x, PLNDL);
        FN(p2)(d, PLNDL, 1);
        map_vec_nta(d, cf + (size_t)b * VDL, d, VDL / 8);
    }
}

/* in-place chain step with the map fused into phase 2's stores (r2, the
 * gen_powp ipf family): the map pass disappears entirely.  ipf0/ipf1
 * differ only in the p2 write-intent pokes. */
static void FN(xc_ipf0)(const double *cur, double *dst, const double *cf,
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

static void FN(xc_ipf1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, d, P, x, PLNDL);
        FN(p2ipf)(d, cf + (size_t)b * VDL, 1);
    }
}

/* in-place chain step with the per-tile map epilogue (r2, new family). */
static void FN(xc_ipe0)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, d, P, x, PLNDL);
        FN(p2ipe)(d, cf + (size_t)b * VDL, 0);
    }
}

static void FN(xc_ipe1)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, d, P, x, PLNDL);
        FN(p2ipe)(d, cf + (size_t)b * VDL, 1);
    }
}

#if GENL % 4 == 0
/* NT-store variants (r2): p1's y-subpass streams the state out with
 * vmovntpd (no RFO read of the state volume), sfence, then either the
 * plain in-place p2 + sequential map (ipnt) or the fused-map in-place p2
 * (ipfnt).  Falls back to the ordinary p1 when the volume base is not
 * 64B-aligned (never observed; the arena and driver buffers are). */
static void FN(xc_ipnt)(const double *cur, double *dst, const double *cf,
                        long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        if (((uintptr_t)d & 63) == 0) {
            for (int x = 0; x < GENL; ++x)
                FN(p1nt)(i, d, P, x, PLNDL);
            _mm_sfence();
        } else {
            for (int x = 0; x < GENL; ++x)
                FN(p1)(i, d, P, x, PLNDL);
        }
        FN(p2)(d, PLNDL, 1);
        FN(map_vol)(d, cf + (size_t)b * VDL, d);
    }
}

static void FN(xc_ipfnt)(const double *cur, double *dst, const double *cf,
                         long nvol, double *M, double *P)
{
    (void)M;
    for (long b = 0; b < nvol; ++b) {
        const double *i = cur + (size_t)b * VDL;
        double       *d = dst + (size_t)b * VDL;
        if (((uintptr_t)d & 63) == 0) {
            for (int x = 0; x < GENL; ++x)
                FN(p1nt)(i, d, P, x, PLNDL);
            _mm_sfence();
        } else {
            for (int x = 0; x < GENL; ++x)
                FN(p1)(i, d, P, x, PLNDL);
        }
        FN(p2ipf)(d, cf + (size_t)b * VDL, 1);
    }
}
#endif /* GENL % 4 == 0 */
#endif /* !GEN_LEAN (xc_ipq/ipk/iqn/ipf/ipe/nt) */

static void FN(x_pf0)(const double *in, double *out, long nvol, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDL;
        double       *o = out + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, o, P, x, PLNDL);
        FN(p2)(o, PLNDL, 0);
    }
}

static void FN(x_pf1)(const double *in, double *out, long nvol, double *P)
{
    for (long b = 0; b < nvol; ++b) {
        const double *i = in  + (size_t)b * VDL;
        double       *o = out + (size_t)b * VDL;
        for (int x = 0; x < GENL; ++x)
            FN(p1)(i, o, P, x, PLNDL);
        FN(p2)(o, PLNDL, 1);
    }
}

#undef GENPFA_HEAVY
#undef FN
#undef NYG
#undef NFULL
#undef VDL
#undef PLNDL
#undef NPL

#endif /* template */
